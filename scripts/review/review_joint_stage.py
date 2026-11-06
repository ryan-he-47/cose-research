"""2026-10-20：本机图文阶段验收，真实源分区、教师强制、冻结基座对照与GMM闭环。"""
import argparse
from collections import defaultdict
import gc
import html
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F
from scripts.demo.demo_text_readout import load_joint_readout
from scripts.demo.demo_pretrain_generation import save_native_svg,source_prefix
from scripts.review.review_formal_pipeline import ReviewDrawings,ReviewRecipes
from scripts.review.review_formal_r import stroke_records
from scripts.review.diagnose_formal_r_health import model_hash
from scripts.review.render_formal_r_review import svg_pages
from src.st.formal_data import FormalDocumentDataset
from src.st.model import gmm_nll
from src.st.pipeline_pretrain import collate_documents,to_device
from src.st.text_auxiliary import annotate_documents
from src.text_augmentation import TextAugmentationConfig


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


class SnapshotDrawings(ReviewDrawings):
    def __len__(self):
        # 仿射源索引必须在原完整池上计算，不能换成导出子集的长度。
        return self.info['count']


def dataset_for(bundle,split,font):
    data=bundle['splits'][split]
    path=Path('datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg')
    dataset=FormalDocumentDataset(font,(),SnapshotDrawings(data),font_path=path,
        seed=bundle['source_manifest']['seed'],split=split,
        augmentation=TextAugmentationConfig(enabled=False),object_sampling='affine_pass_v1')
    recipes=data['recipes']
    dataset.recipes=ReviewRecipes({int(k):v for k,v in recipes.items()} if isinstance(recipes,dict) else recipes)
    dataset.recipe_count=data['recipe_count']
    return dataset


def audit_stage(checkpoint,parent):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    old=torch.load(parent,map_location='cpu',weights_only=False)
    frozen=[k for k in old['model'] if k.startswith(('encoder.','decoder.','dct_head.'))]
    groups=[]
    for group in state['optimizer']['param_groups']:
        steps=[int(state['optimizer']['state'][p]['step']) for p in group['params']]
        groups.append(dict(role=group.get('role','R'),entries=len(steps),min_step=min(steps),max_step=max(steps)))
    result=dict(step=state['step'],phase_step=state['phase_step'],successful_inputs=state['scheduler']['inputs'],
        parent_step=old['step'],parent_checkpoint=str(parent.resolve()),
        target_inputs=state['train_args']['total_inputs'],source_commit=state['source_commit'],optimizer_groups=groups,
        frozen_ae_equal=all(torch.equal(state['model'][k],old['model'][k]) for k in frozen),
        weights_finite=all(bool(torch.isfinite(v).all()) for v in state['model'].values()),
        scheduler=state['scheduler'],scaler=state['scaler'],counters=state['counters'],data=state['data'])
    assert result['frozen_ae_equal'] and result['weights_finite']
    assert all(g['min_step']==g['max_step']==state['step'] for g in groups)
    assert result['successful_inputs']==state['counters']['successful_inputs']
    return result


@torch.no_grad()
def teacher(generator,head,document):
    annotate_documents([document],head.catalog)
    cpu=collate_documents([document],max_strokes=generator.model.cfg.max_context_strokes)
    batch=to_device(cpu,generator.device)
    model=generator.model
    with generator.autocast():
        emb=model.encode_valid(batch)
        hidden=model.hidden_states(emb[:,:-1],batch['slice_id'][:,:-1],batch['stroke_pad'][:,:-1],right_padded=True)
        selected=model.target_mask(batch)
        mu,sigma,pi=model.prediction_head(hidden[selected])
        target=emb[:,1:][selected].float()
        nll=gmm_nll(target,mu,sigma,pi,reduction='none')
        choice=pi.argmax(-1);ids=torch.arange(len(choice),device=generator.device)
        predicted=mu[ids,choice]
        error=(predicted-target).square().mean(-1)
        output=head(hidden[selected])
        kinds=batch['kind_id'][:,1:][selected]
        valid=batch['text_aux_valid'][:,1:][selected]
        shape=batch['text_aux_shape'][:,1:][selected]
        space=batch['text_aux_space'][:,1:][selected]
    result={}
    first_text=torch.zeros_like(kinds,dtype=torch.bool)
    positions=torch.nonzero(kinds==0).flatten()
    if len(positions):first_text[positions[0]]=True
    for name,mask in (('text',kinds==0),('answer_initial',first_text),('sketch',kinds==1),('selection',kinds==2),('control',kinds>=3)):
        count=int(mask.sum())
        if not count:continue
        labeled=mask & valid
        space_mask=labeled & ((kinds==0)|(kinds>=3))
        result[name]=dict(count=count,nll_sum=float(nll[mask].sum()),
            mse_sum=float(error[mask].sum()),sigma_sum=float(sigma[ids,choice][mask].mean(-1).sum()),
            shape_count=int(labeled.sum()),
            shape_ce_sum=float(F.cross_entropy(output['shape'][labeled].float(),shape[labeled],reduction='sum')) if labeled.any() else 0.,
            shape_wrong=int((output['shape'][labeled].argmax(-1)!=shape[labeled]).sum()),
            space_count=int(space_mask.sum()),
            space_wrong=int(((output['space'][space_mask]>0)!=space[space_mask].bool()).sum()))
    return result,predicted,target


def aggregate(rows):
    totals=defaultdict(lambda:defaultdict(float))
    for row in rows:
        for kind,values in row['metrics'].items():
            bucket=totals[(row['split'],row['task'],kind)]
            for key,value in values.items():bucket[key]+=value
    result=[]
    for (split,task,kind),v in totals.items():
        n=v['count'];s=v['shape_count'];p=v['space_count']
        result.append(dict(split=split,task=task,kind=kind,count=int(n),nll=v['nll_sum']/n,
            greedy_latent_mse=v['mse_sum']/n,sigma=v['sigma_sum']/n,
            shape_ce=v['shape_ce_sum']/s if s else None,
            shape_accuracy=1-v['shape_wrong']/s if s else None,
            space_accuracy=1-v['space_wrong']/p if p else None))
    return result


def render(out,summary):
    cells=[]
    for model,rows in summary['metrics'].items():
        for r in rows:
            ce='—' if r['shape_ce'] is None else f"{r['shape_ce']:.3f}"
            cells.append(f"<tr><td>{model}</td><td>{r['split']}</td><td>{r['task']}</td><td>{r['kind']}</td>"
                f"<td>{r['count']}</td><td>{r['nll']:.2f}</td><td>{r['greedy_latent_mse']:.5f}</td><td>{ce}</td></tr>")
    cards=[]
    for case in summary['cases']:
        p=out/case['name'];reference=json.loads((p/'reference.json').read_text())
        cols=[f"<div><h3>真实回答</h3>{svg_pages([r for r in reference['strokes'] if r.get('targetable')])}</div>"]
        for name in ('teacher_forced','ae_reference','greedy','sample'):
            file=p/(name+'.json')
            if file.exists():
                value=json.loads(file.read_text());records=value['strokes']
                if name in ('greedy','sample'):records=[r for r in records if r.get('generated')]
                cols.append(f"<div><h3>{name} · {html.escape(value.get('stop_reason',''))}</h3>{svg_pages(records)}</div>")
        cards.append(f"<section><h2>{case['name']} · {html.escape(case.get('label',''))}</h2>"
            f"<details><summary>条件与完整源文档</summary>{svg_pages(reference['strokes'])}</details><div class='cols'>{''.join(cols)}</div></section>")
    page="""<!doctype html><meta charset='utf-8'><title>图文阶段验收</title><style>
body{font:15px/1.55 system-ui;margin:24px;color:#203047}table{border-collapse:collapse}td,th{border:1px solid #ccd4df;padding:5px 10px}
.cols{display:flex;gap:16px;align-items:flex-start}.cols>div{width:240px;flex-shrink:0}.page{width:230px;margin:8px 0;border:1px solid #ccd4df}.page svg{width:100%}.pageid{font-size:12px}section{border-top:2px solid #ccd4df;margin:24px 0;overflow:auto}details .page{display:inline-block}
</style><h1>图文联合训练阶段验收</h1><p>本机CUDA/FP16，只读。教师强制使用真实前文；greedy为最高π分量均值，sample温度0.5，均曲线解码再编码闭环。</p>
<p>训练与留出均关闭字形增强以便比较；对象来自真实源分区，场景复用正式配方。非符号类的低CE只表示类别区分，不表示绘画正确。</p>
<p><a href='text/index.html'>文字教师强制与分类/GMM续写</a> · <a href='curves.png'>训练曲线</a></p>
<table><tr><th>模型</th><th>分区</th><th>任务</th><th>笔画</th><th>目标数</th><th>NLL</th><th>均值λ MSE</th><th>形状CE</th></tr>"""
    if (out/'curves/index.html').exists():page=page.replace("href='curves.png'","href='curves/index.html'")
    (out/'index.html').write_text(page+''.join(cells)+'</table>'+''.join(cards),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reuse_parent_teacher',action='store_true',help='复用本目录已完成的同起点教师强制结果')
    parser.add_argument('--compare_checkpoint',type=Path,help='另测中期检查点，同一源文档对照')
    parser.add_argument('--scene_examples',type=int,default=1)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'joint_summary.json').exists():raise FileExistsError('已有阶段摘要')
    torch.set_num_threads(4);started=time.monotonic()
    bundle=json.loads(args.bundle.read_text());summary=dict(metrics={},cases=[],optimizer_updates=0)
    audit=audit_stage(args.checkpoint,args.parent);write(args.out/'audit.json',audit);gc.collect()
    models=[('text_base',args.parent)]
    if args.compare_checkpoint:models.append(('joint_861m',args.compare_checkpoint))
    models.append(('joint_stage',args.checkpoint))
    for name,path in models:
        if name=='text_base' and args.reuse_parent_teacher:
            rows=json.loads((args.out/'text_base_teacher.json').read_text(encoding='utf-8'))
            summary['metrics'][name]=aggregate(rows);summary[name+'_step']=audit['parent_step']
            continue
        generator,head,vocab,layout,source=load_joint_readout(path)
        generator.model.encode_chunk_size=128
        rows=[]
        for split in ('train','valid'):
            dataset=dataset_for(bundle,split,vocab.font)
            indices=bundle['splits'][split].get('object_indices',list(range(bundle['object_count'])))
            scenes=bundle['splits'][split].get('scene_indices',list(range(bundle['scene_count'])))
            requests=[(task,i) for task in ('O1','O2') for i in indices]
            requests += [(task,i) for task in ('S1','S2','S3','S4','S5') for i in scenes]
            requests += [(task,s['index']) for task in ('O1','O2') for s in bundle['splits'][split]['samples']]
            for task,i in requests:
                document=dataset.get_for_task(task,i)
                metrics,predicted,target=teacher(generator,head,document)
                rows.append(dict(split=split,task=task,index=i,metrics=metrics,source_id=document.metadata.get('source_id')))
                sample=next((s for s in bundle['splits'][split]['samples'] if s['index']==i),None)
                selected=(task in ('O1','O2') and sample is not None) or (task.startswith('S') and i in scenes[:args.scene_examples])
                if name=='joint_stage' and split=='valid' and selected:
                    case=args.out/f"{task}_{sample['label'] if sample else 'scene'+str(i)}";case.mkdir(exist_ok=True)
                    write(case/'reference.json',dict(strokes=stroke_records(document),
                        source={k:v for k,v in document.metadata.items() if not k.startswith('_')}))
                    if task in ('O1','S2','S4'):
                        elements=[e for e in document.to_sequence()[1:] if e.targetable]
                        for method,emb in (('teacher_forced',predicted),('ae_reference',target)):
                            points=generator.decode(emb)
                            records=[dict(page_id=e.page_id,kind=e.kind,generated=True,points=p.tolist()) for e,p in zip(elements,points)]
                            write(case/(method+'.json'),dict(strokes=records))
                    summary['cases'].append(dict(name=case.name,task=task,label=sample['label'] if sample else '',index=i))
            print(json.dumps(dict(model=name,split=split,documents=len(requests))),flush=True)
        summary['metrics'][name]=aggregate(rows);write(args.out/(name+'_teacher.json'),rows)
        if name=='joint_stage':
            dataset=dataset_for(bundle,'valid',vocab.font)
            for case in summary['cases']:
                document=dataset.get_for_task(case['task'],case['index'])
                for mode in ('greedy','sample'):
                    value=generator.generate(source_prefix(document,case['task']),mode=mode,temperature=.5,
                        seed=42,max_new_strokes=320 if case['task']=='S1' else 160,max_new_pages=8)
                    write(args.out/case['name']/(mode+'.json'),value)
                    save_native_svg(value,args.out/case['name']/(mode+'.svg'),case['name'])
                    print(json.dumps(dict(case=case['name'],mode=mode,stop=value['stop_reason'],
                                          strokes=sum(r['generated'] for r in value['strokes']))),flush=True)
        summary[name+'_step']=source['step']
        assert model_hash(generator.model)==source['r_model_hash'],'只读推理改变了权重'
        del generator,head,vocab;gc.collect();torch.cuda.empty_cache()
    summary['elapsed_seconds']=time.monotonic()-started
    summary['weights_unchanged']=True
    write(args.out/'joint_summary.json',summary);render(args.out,summary)
    print(json.dumps(dict(complete=True,seconds=summary['elapsed_seconds'])),flush=True)


if __name__=='__main__':main()
