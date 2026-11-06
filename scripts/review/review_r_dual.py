"""2026-10-09：本机只读验收T1语言探索与纠偏R的实际条件生成。"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.demo.demo_pretrain_generation import load_generator, save_native_svg, source_prefix, text_prefix
from scripts.review.diagnose_formal_r_health import model_hash, target_predictions, mixture_rows, summarize_mixture, condition_matrix, matched_conditions
from scripts.review.review_formal_r import write_json, stroke_records
from src.page_document import build_document, layout_text_pages, StrokeElement
from src.st.model import gmm_nll


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(path, ae_path, require_complete):
    state = torch.load(path,map_location='cpu',weights_only=False)
    ae = torch.load(ae_path,map_location='cpu',weights_only=False)
    selected = {k:v for k,v in state['model'].items() if k.startswith(('encoder.','decoder.','dct_head.'))}
    frozen = set(selected)==set(ae['model']) and all(torch.equal(v,ae['model'][k]) for k,v in selected.items())
    steps = [int(s['step']) for s in state['optimizer']['state'].values()]
    phase = state.get('phase_step',state['step'])
    complete = state['scheduler']['inputs']>=state['train_args']['total_inputs']
    digest = hashlib.sha256(ae_path.read_bytes()).hexdigest()
    result = {k:state[k] for k in ['step','phase_step','counters','train_args','data','data_args','data_manifest','cfg','ae_source','parent_source','source_commit','scaler']}
    result.update(successful_inputs=state['scheduler']['inputs'],final_lr=state['optimizer']['param_groups'][0]['lr'],
        optimizer_step_min=min(steps),optimizer_step_max=max(steps),optimizer_entries=len(steps),
        frozen_ae_equal=frozen,frozen_ae_tensors=len(selected),ae_hash_equal=digest==state['ae_source']['sha256'],
        weights_finite=all(bool(torch.isfinite(v).all()) for v in state['model'].values()),
        budget_complete=complete,review_stage='final' if complete else 'interim_checkpoint')
    if not (frozen and result['ae_hash_equal'] and result['weights_finite'] and min(steps)==max(steps)==phase
            and state['scheduler']['inputs']==state['counters']['successful_inputs'] and (complete or not require_complete)):
        raise RuntimeError('训练完整状态核对异常')
    return result


@torch.no_grad()
def teacher(generator, records, task, out):
    mu,sigma,pi,target,kinds = target_predictions(generator,records,task)
    rows = mixture_rows(mu,sigma,pi,target,kinds,task,sigma_eps=generator.model.cfg.sigma_eps_r,
                        sigma_clamp_lo=generator.model.cfg.sigma_clamp_lo)
    report = summarize_mixture(rows)
    picked = mu[torch.arange(len(pi),device=generator.device),pi.argmax(-1)]
    elements = [r for i,r in enumerate(records) if i and r['targetable']]
    for name,latent in [('teacher_forced',picked),('ae_reference',target)]:
        with generator.autocast():
            points = np.concatenate([generator.decode(latent[i:i+64]) for i in range(0,len(latent),64)])
        value = {'strokes':[dict(page_id=r['page_id'],kind=r['kind'],points=p.tolist(),generated=True)
                            for r,p in zip(elements,points)],'conditioning':'true_previous_strokes' if name=='teacher_forced' else 'true_target_embedding'}
        write_json(out/(name+'.json'),value)
        save_native_svg(value,out/(name+'.svg'),name)
    write_json(out/'teacher_metrics.json',report)
    return report,rows


def generate_case(generator, out, name, prefix, reference, *, task, prompt, max_strokes, latent=False):
    directory = out/name
    directory.mkdir()
    write_json(directory/'reference.json',reference)
    result = {'name':name,'task':task,'prompt':prompt,'generation':{}}
    for mode,feedback in [('greedy','curve'),('sample','curve')]+([('greedy','latent')] if latent else []):
        label = mode if feedback=='curve' else 'latent'
        value = generator.generate(prefix,mode=mode,temperature=.5,seed=42,max_new_strokes=max_strokes,
            max_new_pages=8,feedback_mode=feedback)
        write_json(directory/(label+'.json'),value)
        save_native_svg(value,directory/(label+'.svg'),name+' / '+label)
        result['generation'][label] = {k:value[k] for k in ['stop_reason','generated_strokes','generated_pages','outside_page_strokes']}
        print(json.dumps({'case':name,'variant':label,**result['generation'][label]}),flush=True)
    return result


@torch.no_grad()
def language_pairs(generator,font):
    # 同一续句的点列/页号在两种前文下完全相同；观察条件对候选的相对作用。
    pairs = [
        ('voice',['A little cat was sitting here. It said','A little dog was sitting here. It said'],['meow.','woof.']),
        ('need',['Lily was very hungry.','Lily was very tired.'],['She ate some food.','She went to sleep.']),
        ('weather',['It was raining outside.','It was sunny outside.'],['She took her umbrella.','She put on her sunglasses.'])]
    result=[]
    for name,contexts,candidates in pairs:
        pages = [layout_text_pages(c,font,targetable=False) for c in contexts]
        if len(pages[0])!=len(pages[1]):
            raise AssertionError('交换前文时页数须一致')
        scores=[]
        for context,condition in zip(contexts,pages):
            row=[]
            for continuation in candidates:
                doc = build_document((*condition,*layout_text_pages(continuation,font,targetable=True)),complete=True)
                records = stroke_records(doc)
                mu,sigma,pi,target,kinds = target_predictions(generator,records,'T1')
                values = gmm_nll(target,mu,sigma,pi,reduction='none')[kinds==0]
                row.append({'sum_nll':float(values.sum()),'mean_nll':float(values.mean()),'strokes':len(values)})
            scores.append(row)
        # 正值：第二前文相对第一前文更偏向第二候选。逐候选作前文差，消除静态字形难度。
        shifts=[scores[1][i]['mean_nll']-scores[0][i]['mean_nll'] for i in range(2)]
        result.append(dict(name=name,contexts=contexts,candidates=candidates,scores=scores,
            context_swap_preference_shift=shifts[0]-shifts[1],same_candidate_geometry=True,
            interpretation='正值表示条件交换方向与示例设想一致；不是自由输出的可读性结果'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=['t1','corrective','baseline_t1'],required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--ae',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=ROOT/'tmp_smoke/r_formal_review_20261009')
    parser.add_argument('--out_dir',type=Path,required=True)
    parser.add_argument('--interim',action='store_true',help='2026-10-11：允许核对未完成预算的阶段检查点')
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    checked=audit(args.checkpoint,args.ae,args.arm!='corrective' and not args.interim)
    write_json(args.out_dir/'checks.json',checked)
    generator,font,metadata=load_generator(args.checkpoint,'cuda')
    generator.model.encode_chunk_size=128
    before=model_hash(generator.model)
    summary=[];all_rows=[]
    if args.arm=='corrective':
        originals=read(args.baseline/'summary.json')
        sources={f"{c['task']}_{c['label']}":read(args.baseline/f"{c['task']}_{c['label']}"/'reference.json')['strokes'] for c in originals}
        for item in originals:
            name=f"{item['task']}_{item['label']}"
            reference=read(args.baseline/name/'reference.json')
            old=read(args.baseline/name/'greedy.json')
            prefix=tuple(StrokeElement(np.asarray(r['points'],np.float32),r['page_id'],False,r['kind'])
                         for r in old['strokes'] if not r['generated'])
            result=generate_case(generator,args.out_dir,name,prefix,reference,task=item['task'],prompt=item['source'],max_strokes=old['max_new_strokes'])
            result['teacher_metrics'],rows=teacher(generator,reference['strokes'],item['task'],args.out_dir/name)
            all_rows+=rows;summary.append(result)
            write_json(args.out_dir/'summary.json',summary)
        write_json(args.out_dir/'condition_O1_fixed_prompt.json',condition_matrix(generator,matched_conditions(sources,font,'O1'),'O1'))
    else:
        for index in range(2):
            text=read(args.baseline/f'T1_story{index}'/'reference.json')['source']['text']
            doc=build_document(layout_text_pages(text,font,targetable=True),complete=True).window(0,1024)
            reference={'strokes':stroke_records(doc),'source':doc.metadata,'page_texts':{p.page_id:p.text for p in doc.pages},'full_story_text':text}
            for kind in ['word','sentence']:
                prompt=text.split(' ')[0] if kind=='word' else re.split(r'(?<=[.!?])\s+',text)[0]
                prefix=source_prefix(doc,'T1') if kind=='word' else text_prefix('T1',prompt,font)
                name=f'T1_story{index}_{kind}'
                result=generate_case(generator,args.out_dir,name,prefix,reference,task='T1',prompt=prompt,max_strokes=480,latent=args.arm=='t1' and kind=='sentence')
                if kind=='word':
                    result['teacher_metrics'],rows=teacher(generator,reference['strokes'],'T1',args.out_dir/name)
                    all_rows+=rows
                summary.append(result)
                write_json(args.out_dir/'summary.json',summary)
        if args.arm=='t1':
            prompts=['Once upon a time, there was a little cat named Tom.',
                'Once upon a time, there was a little dog named Max.',
                'Lily was hungry. She went to the kitchen to get',
                'Tom dropped his toy into the lake. He felt']
            for index,prompt in enumerate(prompts):
                prefix=text_prefix('T1',prompt,font)
                reference={'strokes':[],'source':{'manual_prompt':True,'text':prompt}}
                summary.append(generate_case(generator,args.out_dir,f'T1_prompt{index}',prefix,reference,
                    task='T1',prompt=prompt,max_strokes=480))
                write_json(args.out_dir/'summary.json',summary)
        write_json(args.out_dir/'language_pairs.json',language_pairs(generator,font))
    write_json(args.out_dir/'mixture_targets.json',all_rows)
    write_json(args.out_dir/'mixture_summary.json',summarize_mixture(all_rows))
    after=model_hash(generator.model)
    if before!=after: raise AssertionError('只读验收改动了权重')
    write_json(args.out_dir/'completion.json',dict(arm=args.arm,review_stage=checked['review_stage'],training=False,
        optimizer_steps=0,weight_hash_before=before,weight_hash_after=after,weights_unchanged=True,
        cases=len(summary),stop_counts={variant:dict(Counter(c['generation'][variant]['stop_reason'] for c in summary if variant in c['generation']))
            for variant in ['greedy','sample','latent']},cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
    print(json.dumps({'complete':True,'arm':args.arm,'cases':len(summary)}),flush=True)


if __name__=='__main__':
    main()
