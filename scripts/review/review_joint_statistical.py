"""2026-10-27：扩大验证对象/完整场景池，只读评估和可恢复逐例记录。"""
import argparse
from collections import Counter,defaultdict
import gc
import html
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.demo.demo_text_readout import load_joint_readout
from scripts.demo.demo_pretrain_generation import source_prefix
from scripts.review.joint_eval_batch import load_dataset,text_batch,graphic_batch
from scripts.review.probe_joint_scene_abilities import answer_match,caption_classes,layout_stats
from scripts.review.diagnose_formal_r_health import model_hash
from src.st.pipeline_pretrain import collate_documents,to_device
from src.visual_grounding_scenes import paired_scene


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def wilson(correct,count):
    if not count:return None
    z=1.959963984540054;p=correct/count;den=1+z*z/count
    center=(p+z*z/(2*count))/den
    half=z*np.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [float(center-half),float(center+half)]


def accuracy(rows,key='correct'):
    n=len(rows);k=sum(bool(r.get(key)) for r in rows)
    return dict(count=n,correct=k,accuracy=k/n if n else None,ci95_wilson=wilson(k,n),
                errors=sum('error' in r for r in rows),stops=dict(Counter(r.get('stop') for r in rows)))


def class_statistics(by_class,populations=None):
    values=list(by_class.values());n=np.asarray([r['count'] for r in values]);p=np.asarray([r['accuracy'] for r in values])
    rng=np.random.default_rng(20261027)
    draws=rng.binomial(n,p,size=(1000,len(n)))/n
    result=dict(macro_accuracy=float(p.mean()),macro_ci95_stratified_bootstrap=np.quantile(draws.mean(1),[.025,.975]).tolist())
    if populations:
        weights=np.asarray([populations[c] for c in by_class],float);weights/=weights.sum()
        result.update(raw_population_weighted_accuracy=float(p@weights),
            raw_population_weighted_ci95=np.quantile(draws@weights,[.025,.975]).tolist())
    return result


def mean_interval(values,seed=20261027):
    x=np.asarray(values,dtype=np.float64)
    if not len(x):return dict(count=0)
    rng=np.random.default_rng(seed)
    # 独立场景为重采样单位；避免创建1000×20000的巨大常驻数组。
    means=[float(x[rng.integers(0,len(x),len(x))].mean()) for _ in range(1000)]
    return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),
                ci95_bootstrap_mean=np.quantile(means,[.025,.975]).tolist())


class PrefixDocument:
    def __init__(self,prefix,metadata):self.prefix=prefix;self.metadata=metadata
    def to_sequence(self):return self.prefix


@torch.no_grad()
def hidden_batch(generator,prefixes):
    # 字体/控制符缓存跨文档共享，仍使用同一冻结E。
    flat=[e for p in prefixes for e in p]
    with generator.autocast():
        encoded=generator.encode_cached([e.rawxy for e in flat])
        embeddings=encoded.new_zeros((len(prefixes),max(map(len,prefixes)),encoded.shape[-1]))
        pages=torch.zeros(embeddings.shape[:2],dtype=torch.long,device=generator.device)
        padding=torch.ones(embeddings.shape[:2],dtype=torch.bool,device=generator.device)
        cursor=0
        for i,p in enumerate(prefixes):
            embeddings[i,:len(p)]=encoded[cursor:cursor+len(p)];cursor+=len(p)
            pages[i,:len(p)]=torch.tensor([e.page_id for e in p],device=generator.device);padding[i,:len(p)]=False
        hidden=generator.model.hidden_states(embeddings,pages,padding,right_padded=True)
    return hidden[torch.arange(len(prefixes),device=generator.device),torch.tensor([len(p)-1 for p in prefixes],device=generator.device)]


def box_metrics(expected,predicted,objects,target):
    a=np.asarray(expected);p=np.asarray(predicted);b=np.r_[p.min(0),p.max(0)]
    center=(b[:2]+b[2:])/2
    intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union=(a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection
    nearest=min(objects,key=lambda o:np.linalg.norm(np.asarray(o['center'])-center))
    return dict(correct=nearest['instance_id']==target,nearest_label=nearest['label'],
                center_error=float(np.linalg.norm(center-(a[:2]+a[2:])/2)),bbox_iou=float(intersection/max(1e-12,union)))


def cached_rows(path):
    if not path.exists():return []
    result=[]
    good_bytes=0;needs_newline=False
    with path.open('rb') as f:
        for line in f:
            try:
                if line.strip():result.append(json.loads(line))
            except json.JSONDecodeError:
                if f.read(1):raise
                path.with_suffix(path.suffix+'.incomplete_tail').write_bytes(line)
                with path.open('r+b') as target:target.truncate(good_bytes)
                break
            good_bytes+=len(line)
            needs_newline=not line.endswith(b'\n')
    if needs_newline:
        with path.open('ab') as f:f.write(b'\n')
    return result


def progress(out,phase,done,total,started):
    value=dict(phase=phase,done=done,total=total,seconds=time.monotonic()-started)
    write(out/'progress.json',value);print(json.dumps(value),flush=True)


def feature_phase(generator,dataset,info,split,out,batch_size):
    rows=[]
    for task,indices in [('O2',[o['index'] for o in info['objects']]),('S3',info['scene_indices']),('S4',info['scene_indices'])]:
        folder=out/'features'/f'{split}_{task}';folder.mkdir(parents=True,exist_ok=True)
        started=time.monotonic()
        for begin in range(0,len(indices),batch_size):
            path=folder/f'{begin:07d}.npz'
            if path.exists():continue
            documents=[];prefixes=[];meta=[]
            for i in indices[begin:begin+batch_size]:
                doc=dataset.get_for_task(task,i);prefix=source_prefix(doc,task)
                item=dict(split=split,task=task,index=i,kind=doc.metadata.get('query',{}).get('kind',task),
                    label=doc.metadata.get('answer',doc.metadata.get('label')),recipe_id=doc.metadata.get('recipe_id'))
                if task=='S4':
                    obj=next(o for o in doc.metadata['objects'] if o['instance_id']==doc.metadata['target_instance_id'])
                    item['center']=obj['center']
                if len(prefix)>1024:
                    item['error']='context_overflow';meta.append(item);continue
                prefixes.append(prefix);documents.append(doc);meta.append(item)
            valid=[m for m in meta if 'error' not in m]
            h=hidden_batch(generator,prefixes).float().cpu().numpy() if prefixes else np.empty((0,generator.model.cfg.r_d_model),np.float32)
            temporary=path.with_name(path.stem+'.tmp.npz')
            np.savez_compressed(temporary,h=h,metadata=json.dumps(valid,ensure_ascii=False),excluded=json.dumps([m for m in meta if 'error' in m]))
            temporary.replace(path)
            if begin%(batch_size*10)==0:progress(out,f'features/{split}/{task}',begin+len(meta),len(indices),started)
        progress(out,f'features/{split}/{task}',len(indices),len(indices),started)


def fit_readouts(out,classes,populations=None):
    data={}
    for split in ('train','valid'):
        for task in ('O2','S3','S4'):
            xs=[];metadata=[]
            for path in sorted((out/'features'/f'{split}_{task}').glob('*.npz')):
                if '.tmp.' in path.name:continue
                with np.load(path) as value:xs.append(value['h']);metadata+=json.loads(str(value['metadata']))
            data[split,task]=(np.concatenate(xs),metadata)
    reports=[]
    for kind,task in [('O2','O2'),('count','S3'),('existence','S3'),('direction','S3'),('S4','S4')]:
        groups={}
        for split in ('train','valid'):
            x,meta=data[split,task];mask=[i for i,m in enumerate(meta) if m['kind']==kind]
            groups[split]=(x[mask].astype(np.float64),[meta[i] for i in mask])
        x,train=groups['train'];v,valid=groups['valid']
        if not len(x) or not len(v):continue
        xm=x.mean(0);x=x-xm;v=v-xm
        labels=list(classes) if kind=='O2' else sorted({str(m['label']) for m in train+valid})
        if kind=='S4':
            y=np.array([m['center'] for m in train]);truth=np.array([m['center'] for m in valid])
        else:
            lookup={s:i for i,s in enumerate(labels)}
            y=np.eye(len(labels))[[lookup[str(m['label'])] for m in train]]
            truth=np.array([lookup[str(m['label'])] for m in valid])
        ym=y.mean(0)
        weight=np.linalg.solve(x.T@x+10*np.eye(x.shape[1]),x.T@(y-ym))
        prediction=v@weight+ym
        if kind=='S4':
            errors=np.linalg.norm(prediction-truth,axis=1)
            result=dict(center_error=mean_interval(errors),train_mean_baseline=mean_interval(np.linalg.norm(y.mean(0)-truth,axis=1)))
        else:
            predicted=prediction.argmax(-1);baseline=int(y.sum(0).argmax())
            cases=[dict(index=m['index'],label=m['label'],prediction=labels[int(p)],correct=bool(p==t)) for m,p,t in zip(valid,predicted,truth)]
            write(out/f'readout_{kind}_cases.json',cases)
            result=accuracy(cases);result['majority_accuracy']=float((truth==baseline).mean())
            if kind=='O2':
                by_class={label:accuracy([r for r in cases if r['label']==label]) for label in labels}
                result['per_class']=by_class
                result.update(class_statistics(by_class,populations))
        reports.append(dict(kind=kind,train=len(train),valid=len(valid),method='ridge lambda=10, train-centered, fixed settings',**result))
    write(out/'readouts.json',reports);return reports


def free_phase(generator,head,vocab,layout,dataset,info,task,out,batch_size,scene_subset=None):
    indices=[o['index'] for o in info['objects']] if task=='O2' else (scene_subset or info['scene_indices'])
    path=out/f'free_{task}.jsonl';done=cached_rows(path);start=len(done);started=time.monotonic()
    with path.open('a',encoding='utf-8') as f:
        for begin in range(start,len(indices),batch_size):
            docs=[dataset.get_for_task(task,i) for i in indices[begin:begin+batch_size]]
            prefixes=[source_prefix(d,task) for d in docs]
            legal=[i for i,p in enumerate(prefixes) if len(p)<1024]
            selected=[prefixes[i] for i in legal]
            values=(graphic_batch(generator,selected,limit=8) if task=='S4' else
                    text_batch(generator,head,vocab,layout,selected,limit=320 if task=='S1' else 96)) if selected else []
            outputs=dict(zip(legal,values))
            for i,doc in enumerate(docs):
                row=dict(task=task,index=indices[begin+i],kind=doc.metadata.get('query',{}).get('kind',task),
                         question=doc.metadata.get('question'),expected=doc.metadata.get('answer',doc.metadata.get('label')),correct=False)
                row.update(source_id=doc.metadata.get('source_id'),recipe_id=doc.metadata.get('recipe_id'))
                if i not in outputs:row.update(error='context_overflow',stop='context_full')
                else:
                    value=outputs[i];row['stop']=value['stop_reason']
                    if task=='S4':
                        body=[r for r in value['strokes'] if r['generated'] and r['kind'] not in ('bos','eop','eod')]
                        row.update(query_kind=doc.metadata.get('query_kind'),generated_body_strokes=len(body))
                        if body:row.update(box_metrics(doc.metadata['selection_box'],body[0]['points'],doc.metadata['objects'],doc.metadata['target_instance_id']))
                    else:
                        row['generated']=value['generated'];row['correct']=answer_match(value['generated'],str(row['expected']),task)
                        if task=='S1':
                            wanted={o['label'] for o in doc.metadata['objects']};found=caption_classes(value['generated'],dataset.drawings.classes)
                            row.update(class_precision=len(wanted&found)/max(1,len(found)),class_recall=len(wanted&found)/len(wanted))
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            f.flush();progress(out,f'free/{task}',begin+len(docs),len(indices),started)
    rows=cached_rows(path);report=accuracy(rows)
    if task=='O2':
        report['per_class']={label:accuracy([r for r in rows if r['expected']==label]) for label in dataset.drawings.classes}
        report.update(class_statistics(report['per_class'],{r['label']:r['population'] for r in info['class_counts']}))
    if task=='S3':
        report['by_kind']={}
        for kind in sorted({r['kind'] for r in rows}):
            selected=[r for r in rows if r['kind']==kind]
            report['by_kind'][kind]={**accuracy(selected),'expected_distribution':dict(Counter(str(r['expected']) for r in selected)),
                'generated_distribution':dict(Counter(r.get('generated','<error>') for r in selected))}
    if task=='S5':report['per_class']={label:accuracy([r for r in rows if r['expected']==label]) for label in sorted({r['expected'] for r in rows})}
    if task=='S4':
        report.update(center_error=mean_interval([r['center_error'] for r in rows if 'center_error' in r]),
                      bbox_iou=mean_interval([r['bbox_iou'] for r in rows if 'bbox_iou' in r]))
    if task=='S1':report.update(class_precision=mean_interval([r['class_precision'] for r in rows if 'class_precision' in r]),
                               class_recall=mean_interval([r['class_recall'] for r in rows if 'class_recall' in r]))
    write(out/f'free_{task}_summary.json',report);return report


@torch.no_grad()
def candidate_phase(generator,dataset,info,out,batch_size):
    path=out/'candidates.jsonl';start=len(cached_rows(path));indices=info['scene_indices'];started=time.monotonic()
    with path.open('a',encoding='utf-8') as f:
        for begin in range(start,len(indices),batch_size):
            docs=[dataset.get_for_task('S4',i) for i in indices[begin:begin+batch_size]]
            prefixes=[]
            for doc in docs:
                seq=doc.to_sequence();boundary=next(i for i,e in enumerate(seq) if e.targetable and e.kind=='selection')
                prefixes.append(seq[:boundary])
            hidden=hidden_batch(generator,prefixes)
            with generator.autocast():
                mu,sigma,pi=generator.model.prediction_head(hidden)
                curves=generator.decode_tensor(mu.flatten(0,1)).reshape(len(docs),mu.shape[1],generator.max_points,2).float().cpu().numpy()
            probs=pi.cpu().numpy()
            for i,doc in enumerate(docs):
                points=curves[i];centers=(points.min(1)+points.max(1))/2
                objects=doc.metadata['objects'];target=doc.metadata['target_instance_id']
                nearest=np.linalg.norm(centers[:,None,:]-np.asarray([o['center'] for o in objects])[None,:,:],axis=-1).argmin(-1)
                correct=np.array([objects[int(j)]['instance_id']==target for j in nearest]);order=np.argsort(-probs[i])
                box=np.asarray(doc.metadata['selection_box']);errors=np.linalg.norm(centers-(box[:2]+box[2:])/2,axis=-1)
                row=dict(index=indices[begin+i],object_count=len(objects),target=target,top_component=int(order[0]),
                    target_mass=float(probs[i,correct].sum()),top_center_error=float(errors[order[0]]),best_center_error=float(errors.min()),
                    topk={str(k):bool(correct[order[:k]].any()) for k in (1,2,4,8,32)})
                f.write(json.dumps(row)+'\n')
            f.flush();progress(out,'candidates',begin+len(docs),len(indices),started)
    rows=cached_rows(path)
    report=dict(count=len(rows),condition='true_target_page_controls_only; no_target_ink',
        center_assignment_topk={str(k):accuracy([dict(correct=r['topk'][str(k)]) for r in rows]) for k in (1,2,4,8,32)},
        target_mass=mean_interval([r['target_mass'] for r in rows]),top_center_error=mean_interval([r['top_center_error'] for r in rows]),
        best_center_error=mean_interval([r['best_center_error'] for r in rows]))
    write(out/'candidates_summary.json',report);return report


def pair_phase(generator,head,vocab,layout,dataset,info,out,batch_size,count):
    eligible=[i for i in info['scene_indices'] if len({o['label'] for o in dataset.recipes[i]['objects']})>1]
    rng=np.random.default_rng(20261028);selected=rng.choice(eligible,5*count,replace=False).tolist()
    path=out/'paired.jsonl';done=len(cached_rows(path));started=time.monotonic()
    with path.open('a',encoding='utf-8') as f:
        specs=[]
        for family in range(5):
            for j in range(count):
                specs.extend((family,j,variant) for variant in (0,1))
        for begin in range(done,len(specs),batch_size):
            chunk=specs[begin:begin+batch_size];docs=[]
            for family,j,variant in chunk:
                task='S3' if family<3 else 'S4' if family==3 else 'S5'
                pair_id=3*j+family if family<3 else j
                doc=paired_scene(task,dataset.recipes[selected[family*count+j]],dataset.drawings,vocab.font,pair_id=pair_id,variant=variant)
                docs.append(doc)
            # 每批不混文字和几何；家族边界为2*count，默认批量可整除。
            if len({d.metadata['task'] for d in docs})!=1:raise ValueError('pair批量跨任务边界，请使用可整除2*count的batch_size')
            task=docs[0].metadata['task'];prefixes=[source_prefix(d,task) for d in docs]
            values=graphic_batch(generator,prefixes,limit=8) if task=='S4' else text_batch(generator,head,vocab,layout,prefixes,limit=96)
            for spec,doc,value in zip(chunk,docs,values):
                family,j,variant=spec
                row=dict(family=('direction','count','existence','S4','S5')[family],pair=j,variant=variant,
                    source_recipe=doc.metadata['recipe_id'],expected=doc.metadata['answer'],stop=value['stop_reason'],correct=False)
                if task=='S4':
                    body=[r for r in value['strokes'] if r['generated'] and r['kind'] not in ('bos','eop','eod')]
                    if body:row.update(box_metrics(doc.metadata['selection_box'],body[0]['points'],doc.metadata['objects'],doc.metadata['target_instance_id']))
                else:row.update(generated=value['generated'],correct=answer_match(value['generated'],str(row['expected']),task))
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            f.flush();progress(out,'paired',begin+len(chunk),len(specs),started)
    rows=cached_rows(path);report={}
    for family in ('direction','count','existence','S4','S5'):
        groups=defaultdict(list)
        for r in rows:
            if r['family']==family:groups[r['pair']].append(r)
        complete=[g for g in groups.values() if len(g)==2]
        report[family]=accuracy([dict(correct=all(r['correct'] for r in g)) for g in complete])
        report[family]['metric']='both_variants_correct; pair is statistical unit'
    write(out/'paired_summary.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('tmp_smoke/r_joint_statistical_data_20261027'))
    p.add_argument('--out',type=Path,default=Path('tmp_smoke/r_joint_statistical_review_20261027'))
    p.add_argument('--checkpoint',type=Path,default=Path('tmp_smoke/r_joint_scratch_final_review_20261026/model.pt'))
    p.add_argument('--batch_size',type=int,default=40)
    p.add_argument('--feature_batch_size',type=int,default=64)
    p.add_argument('--pairs_per_family',type=int,default=1000)
    p.add_argument('--phases',default='features,readouts,free,candidates,pairs')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    generator,head,vocab,layout,source=load_joint_readout(args.checkpoint);generator.model.encode_chunk_size=128
    manifest_hash=hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()
    method_sources={'evaluation':Path(__file__),'batch':Path(__file__).with_name('joint_eval_batch.py'),
                    'pairs':Path(paired_scene.__code__.co_filename)}
    method_hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in method_sources.items()}
    run=dict(source=source,data=str(args.data.resolve()),manifest_sha256=manifest_hash,
        method_sha256=method_hashes,
        arguments=vars(args)|{'data':str(args.data),'out':str(args.out),'checkpoint':str(args.checkpoint)})
    previous=args.out/'run.json'
    if previous.exists():
        before=json.loads(previous.read_text(encoding='utf-8'))
        if before['source']['r_model_hash']!=source['r_model_hash'] or before.get('manifest_sha256')!=manifest_hash:
            raise ValueError('不能混合不同权重或数据清单的评估结果')
        if before.get('method_sha256')!=method_hashes:raise ValueError('评估代码或配对定义已改变，使用独立输出目录')
        for name in ('feature_batch_size','pairs_per_family'):
            if before['arguments'][name]!=getattr(args,name):raise ValueError(f'{name}须沿用原运行配置')
    write(previous,run)
    (args.out/'methods').mkdir(exist_ok=True)
    for name,path in method_sources.items():(args.out/'methods'/f'{name}.py').write_bytes(path.read_bytes())
    phases=args.phases.split(',')
    if 'features' in phases:
        for split in ('train','valid'):
            dataset,info=load_dataset(args.data,split,vocab.font)
            feature_phase(generator,dataset,info,split,args.out,args.feature_batch_size)
            del dataset;gc.collect()
    dataset,info=load_dataset(args.data,'valid',vocab.font)
    if 'readouts' in phases:fit_readouts(args.out,dataset.drawings.classes,{r['label']:r['population'] for r in info['class_counts']})
    if 'free' in phases:
        for task in ('O2','S3','S4','S5'):free_phase(generator,head,vocab,layout,dataset,info,task,args.out,args.batch_size)
    if 'candidates' in phases:candidate_phase(generator,dataset,info,args.out,min(32,args.batch_size))
    if 'pairs' in phases:pair_phase(generator,head,vocab,layout,dataset,info,args.out,args.batch_size,args.pairs_per_family)
    assert model_hash(generator.model)==source['r_model_hash']
    write(args.out/'completion.json',dict(weights_unchanged=True,optimizer_updates=0,source=source,phases=phases))
    if set(('features','readouts','free','candidates','pairs'))<=set(phases):
        from scripts.review.render_joint_statistical import render
        render(args.out,args.data)
    print(json.dumps(dict(completed=True,optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
