"""2026-11-05：追加 1B 后的只读 R 噪声、条件、回馈和采样诊断。"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.review.page_diffusion_probe_core import (TIMES,load_model,selected,history,cache_histories,
    predict_blocks,targets,common_noise,sample,decode,errors)
from src.st.page_diffusion_runtime import write_json
from src.st.page_diffusion_codec import file_sha256


def load_rows(root):
    return [(p.stem,torch.load(p,map_location='cpu',weights_only=False,mmap=True))
            for p in sorted(root.glob('*.pt')) if p.stem=='natural' or p.stem.startswith('valid_')]


def principal_axes(path):
    a=np.load(path)
    x=a['z'][a['target'] & np.isin(a['kind'],['text','sketch','selection'])].astype(np.float64)
    values,vectors=np.linalg.eigh(np.cov(x,rowvar=False))
    return values[::-1].copy(),vectors[:,::-1].copy()


@torch.inference_mode()
def denoising(args):
    started=time.monotonic()
    model,codec,state=load_model(args.checkpoint,args.weights)
    device=next(model.parameters()).device
    eigenvalues,basis=principal_axes(args.geometry)
    basis=torch.tensor(basis,dtype=torch.float32,device=device)
    out=args.out/args.name
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'source.json',dict(checkpoint=str(args.checkpoint.resolve()),sha256=file_sha256(args.checkpoint),
        weights=args.weights,step=state['step'],inputs=state.get('successful_inputs',state.get('counters',{}).get('successful_inputs')),
        times=TIMES,modes=args.modes.split(','),eigenvalues=eigenvalues.tolist(),batch=args.batch))
    del state
    times=torch.tensor(TIMES,dtype=torch.float32,device=device)
    decode_ids=[TIMES.index(t) for t in (.001,.003,.01,.03125,.0625,.125,.25,.5,1.)]
    shape_probe=None
    if args.shapes:
        from scripts.review.page_diffusion_shape_probe import ShapeProbe
        shape_probe=ShapeProbe('E:/Trash_things/cose/datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg',args.shapes,device)
    for group,rows in load_rows(args.data):
        if args.groups and group not in args.groups.split(','):continue
        destination=out/f'{group}.npz'
        if destination.exists():continue
        if args.limit:rows=rows[:args.limit]
        # 同任务来源置换；对象标签尽量不同。用 ID 显式保存配对关系。
        donors={}
        for task in sorted(set(r['task'] for r in rows)):
            pool=[r for r in rows if r['task']==task]
            for i,row in enumerate(pool):
                j=(i+max(1,len(pool)//2))%len(pool)
                donors[row['id']]=pool[j]
        rows=sorted(rows,key=lambda r:selected(r)['start'])
        all_metrics={mode:[] for mode in args.modes.split(',')}
        controls,pads,curves,counts,ids,source,labels,tasks,weights=[],[],[],[],[],[],[],[],[]
        shape_correct,shape_counts,shape_base=[],[],[]
        shape_classes=np.zeros(113,dtype=np.int64)
        shape_class_correct=np.zeros((len(decode_ids),113),dtype=np.int64)
        for begin in range(0,len(rows),args.batch):
            batch=rows[begin:begin+args.batch]
            target,body,control,padding=targets(model,batch)
            noise=common_noise(batch).to(device)
            values=(1-times[None,:,None,None])*target[:,None]+times[None,:,None,None]*noise[:,None]
            pages=torch.tensor([selected(r)['page'] for r in batch],device=device)
            local=torch.tensor([selected(r)['local'] for r in batch],device=device)
            for mode in all_metrics:
                histories=[history(r,mode,donors[r['id']]) for r in batch]
                cache,valid=cache_histories(model,histories)
                pred=predict_blocks(model,cache,valid,values,times,pages,local)
                all_metrics[mode].append(errors(pred,target,body,basis))
                if mode=='full':
                    per=(pred-target[:,None]).square().mean(-1)
                    controls.append(((per*control[:,None]).sum(-1)/control.sum(-1)[:,None].clamp_min(1)).cpu().numpy())
                    pads.append(((per*padding[:,None]).sum(-1)/padding.sum(-1)[:,None].clamp_min(1)).cpu().numpy())
                    if args.skip_decode:
                        continue
                    reference=decode(codec,model,target)
                    output=decode(codec,model,pred[:,decode_ids])
                    difference=output-reference[:,None]
                    shift=difference.mean(-2)
                    shape=difference-shift[...,None,:]
                    metric=torch.stack((difference.square().sum(-1).mean(-1).sqrt(),
                        shift.square().sum(-1).sqrt(),shape.square().sum(-1).mean(-1).sqrt()),-1)
                    curves.append(((metric*body[:,None,:,None]).sum(2)/body.sum(1)[:,None,None]).cpu().numpy())
                    if shape_probe is not None:
                        labeled=[shape_probe.target_labels(r,selected(r)) for r in batch]
                        truth=torch.stack([v[0] for v in labeled]).to(device)
                        scale=torch.stack([v[1] for v in labeled]).to(device)
                        keep=truth>=0
                        classification=(shape_probe.classify(output,scale[:,None].expand(-1,len(decode_ids),-1))
                                        if keep.any() else torch.full(output.shape[:3],-2,device=device))
                        base=(shape_probe.classify(reference,scale) if keep.any() else torch.full(reference.shape[:2],-2,device=device))
                        correct=(classification==truth[:,None])&keep[:,None]
                        shape_correct.append(correct.sum(-1).cpu().numpy())
                        shape_counts.extend(keep.sum(-1).cpu().tolist())
                        shape_base.extend(((base==truth)&keep).sum(-1).cpu().tolist())
                        shape_classes+=torch.bincount(truth[keep],minlength=113).cpu().numpy()
                        for ti in range(len(decode_ids)):
                            shape_class_correct[ti]+=torch.bincount(truth[correct[:,ti]],minlength=113).cpu().numpy()
                del cache,pred
            counts.extend(torch.stack((body.sum(1),control.sum(1),padding.sum(1)),1).cpu().tolist())
            ids.extend(r['id'] for r in batch);source.extend(r['source'] for r in batch)
            labels.extend(r['label'] or '' for r in batch);tasks.extend(r['task'] for r in batch)
            weights.extend(r['block_inclusion_weight'] for r in batch)
            if begin//args.batch%32==0:
                print(json.dumps(dict(stage='denoise',model=args.name,group=group,done=begin+len(batch),
                    total=len(rows),seconds=time.monotonic()-started)),flush=True)
        np.savez_compressed(destination,**{k:np.concatenate(v) for k,v in all_metrics.items()},
            control=np.concatenate(controls),padding=np.concatenate(pads),curves=np.concatenate(curves) if curves else np.empty((len(rows),0,3)),
            counts=counts,ids=ids,source=source,labels=labels,tasks=tasks,inclusion_weights=weights,
            times=np.asarray(TIMES),decode_times=np.asarray(TIMES)[decode_ids] if curves else np.empty(0),
            shape_correct=np.concatenate(shape_correct) if shape_correct else np.empty((0,len(decode_ids))),
            shape_counts=shape_counts,shape_baseline=shape_base,shape_classes=shape_classes,shape_class_correct=shape_class_correct)
        print(json.dumps(dict(stage='saved',model=args.name,group=group,n=len(rows),seconds=time.monotonic()-started)),flush=True)


@torch.inference_mode()
def rollout(args):
    """固定真实页/位置下替换历史；测下一块去噪和流形邻近度，另留原始轨迹。"""
    started=time.monotonic()
    model,codec,state=load_model(args.checkpoint,args.weights)
    device=next(model.parameters()).device
    from src.st.generation_pretrain import PageGenerator
    geometry=PageGenerator(codec)
    eigenvalues,basis=principal_axes(args.geometry)
    basis=torch.tensor(basis,dtype=torch.float32,device=device)
    out=args.out/args.name
    out.mkdir(parents=True,exist_ok=True)
    times=torch.tensor([.03125,.125,.5,1.],device=device)
    for group,rows in load_rows(args.data):
        if args.groups and group not in args.groups.split(','):continue
        if group=='natural':continue
        destination=out/f'rollout_{group}.pt'
        if destination.exists():continue
        eligible=[r for r in rows if r['rollout']]
        rng=np.random.default_rng(20261106)
        if len(eligible)>args.rollout_documents:
            eligible=[eligible[i] for i in rng.choice(len(eligible),args.rollout_documents,replace=False)]
        results=[]
        for begin in range(0,len(eligible),args.batch):
            batch=eligible[begin:begin+args.batch]
            first=[r['blocks'][r['rollout'][0]] for r in batch]
            histories=[history(r,block=b) for r,b in zip(batch,first)]
            truth_hist=[tuple(v.clone() for v in h) for h in histories]
            # 参考附近去噪回馈：沿同一真实未来行进，减少自由续写多解带来的歧义。
            reconstruct_hist=[tuple(v.clone() for v in h) for h in histories]
            per_batch=[]
            for horizon in range(4):
                parts=[r['blocks'][r['rollout'][0]+horizon] for r in batch]
                target,body,_,_=targets(model,batch,parts)
                noise=common_noise(batch,salt=100+horizon).to(device)
                values=(1-times[None,:,None,None])*target[:,None]+times[None,:,None,None]*noise[:,None]
                pages=torch.tensor([p['page'] for p in parts],device=device)
                local=torch.tensor([p['local'] for p in parts],device=device)
                metrics={}
                for name,hs in (('truth',truth_hist),('generated',histories),('reconstructed',reconstruct_hist)):
                    cache,valid=cache_histories(model,hs)
                    pred=predict_blocks(model,cache,valid,values,times,pages,local)
                    metrics[name]=errors(pred,target,body,basis)
                    if name=='generated':
                        output,_=sample(model,cache,valid,noise,pages,local,steps=32)
                    if name=='reconstructed':
                        # 参考路径 t=.125 的一次 x 预测，作为近参考、低失真历史回馈。
                        near=pred[:,1].cpu()
                    del cache
                decoded=decode(codec,model,output,points=76)
                reference=decode(codec,model,target,points=76)
                templates=geometry.template_tensor
                control_distance=(decoded[:,:,None,None]-templates[None,None]).square().sum(-1).mean(-1).sqrt().amin(dim=(-1,-2))
                metrics['generated_control']=(control_distance<=geometry.control_tolerance).cpu()
                diff=decoded-reference
                shape=diff-diff.mean(-2,keepdim=True)
                metrics['decoded_curve']=diff.square().sum(-1).mean((-1,-2)).sqrt().cpu().numpy()
                metrics['decoded_shape']=shape.square().sum(-1).mean((-1,-2)).sqrt().cpu().numpy()
                metrics['generated_latent']=output.cpu()
                metrics['target_latent']=target.cpu()
                metrics['generated_curves']=decoded.cpu()
                metrics['reference_curves']=reference.cpu()
                per_batch.append(metrics)
                for i,(part,row) in enumerate(zip(parts,batch)):
                    pp=torch.full((16,),part['page'],dtype=torch.long)
                    ll=torch.arange(part['local'],part['local']+16)
                    for hs,value in ((histories,output[i].cpu()),(truth_hist,target[i].cpu()),(reconstruct_hist,near[i])):
                        hs[i]=(torch.cat((hs[i][0],value)),torch.cat((hs[i][1],pp)),torch.cat((hs[i][2],ll)))
            for i,row in enumerate(batch):
                results.append(dict(id=row['id'],task=row['task'],source=row['source'],label=row['label'],
                    horizons=[{k:v[i] for k,v in x.items()} for x in per_batch]))
            print(json.dumps(dict(stage='rollout',model=args.name,group=group,done=len(results),
                total=len(eligible),seconds=time.monotonic()-started)),flush=True)
        torch.save(dict(times=times.cpu(),results=results,eligible=len([r for r in rows if r['rollout']]),
                        scope='同一真实页内64笔固定位置推进；停止符和跨页另看自由生成面板'),destination)


@torch.inference_mode()
def sampling(args):
    started=time.monotonic()
    model,codec,state=load_model(args.checkpoint,args.weights)
    device=next(model.parameters()).device
    out=args.out/args.name
    out.mkdir(parents=True,exist_ok=True)
    for group,rows in load_rows(args.data):
        if args.groups and group not in args.groups.split(','):continue
        if group=='natural':continue
        destination=out/f'sampling_{group}.pt'
        if destination.exists():continue
        rng=np.random.default_rng(20261107)
        count=len(rows) if group=='valid_S4' else args.rollout_documents
        chosen=rows if len(rows)<=count else [rows[i] for i in rng.choice(len(rows),count,replace=False)]
        results=[]
        for begin in range(0,len(chosen),args.batch):
            batch=chosen[begin:begin+args.batch]
            target,body,_,_=targets(model,batch)
            noise=common_noise(batch,salt=200).to(device)
            cache,valid=cache_histories(model,[history(r) for r in batch])
            pages=torch.tensor([selected(r)['page'] for r in batch],device=device)
            local=torch.tensor([selected(r)['local'] for r in batch],device=device)
            outputs={}
            for label,steps,power in (('16',16,1),('32',32,1),('64',64,1),('32_quadratic',32,2)):
                value,trajectory=sample(model,cache,valid,noise,pages,local,steps=steps,power=power,record=label=='32')
                curves=decode(codec,model,value)
                outputs[label]=dict(z=value.cpu(),curves=curves.cpu())
                if label=='32':
                    for entry in trajectory:
                        # 给出实际更新幅度、相邻 x 预测变化；每文档正文槽位平均。
                        delta=entry.pop('update');current=entry.pop('value');mask=body.cpu()
                        entry['update_rms_all_slots']=delta.square().mean((1,2)).sqrt()
                        entry['update_rms']=((delta.square().mean(-1)*mask).sum(-1)/mask.sum(-1)).sqrt()
                        entry['value_rms']=((current.square().mean(-1)*mask).sum(-1)/mask.sum(-1)).sqrt()
                    path=trajectory
            target_curves=decode(codec,model,target).cpu()
            for i,row in enumerate(batch):
                results.append(dict(id=row['id'],task=row['task'],source=row['source'],label=row['label'],
                    body=body[i].cpu(),target=target[i].cpu(),reference_curves=target_curves[i],
                    outputs={k:{a:b[i] for a,b in v.items()} for k,v in outputs.items()},
                    trajectory=[{k:v if isinstance(v,float) else v[i] for k,v in step.items()} for step in path]))
            print(json.dumps(dict(stage='sampling',model=args.name,group=group,done=len(results),
                total=len(chosen),seconds=time.monotonic()-started)),flush=True)
        torch.save(results,destination)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('denoising','rollout','sampling'),default='denoising')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--weights',default='model')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--geometry',type=Path,default=Path('tmp_smoke/ae_latent_geometry_20261103/r_encoded.npz'))
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--modes',default='full,wrong,none')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--rollout_documents',type=int,default=128)
    p.add_argument('--groups',default='')
    p.add_argument('--shapes',type=Path)
    p.add_argument('--skip_decode',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4)
    globals()[args.mode](args)


if __name__=='__main__':main()
