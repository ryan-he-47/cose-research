"""2026-10-09：冻结λ几何、最小高斯宽度与FP16梯度的本机只读测量。"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import string
import sys

import numpy as np
import torch
from scipy.special import ndtr
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scripts.demo.demo_pretrain_generation import load_generator
from scripts.review.diagnose_formal_r_health import document, model_hash, target_predictions, group_norms
from src.page_document import layout_text_pages, control_stroke
from src.st.model import gmm_nll
from src.st.pipeline_pretrain import collate_documents, to_device
from src.st.formal_recipe import TASKS, LOSS_WEIGHTS

EPSILONS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def stats(values):
    a = np.asarray(values, np.float64).ravel()
    if not len(a):
        return {'count': 0}
    return dict(count=len(a), mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
                **{f'p{p}': float(np.percentile(a, p)) for p in (5, 10, 25, 50, 75, 90, 95, 99)})


def gather(generator, font, samples, t1_review):
    a = np.load(samples)
    raw = [a['ink'][i, :n]*.6+np.array([.5, .6], np.float32) for i,n in enumerate(a['lengths'])]
    buckets = ['aug_text' if b == 'text' else str(b) for b in a['bucket']]
    labels = a['metadata'].tolist()
    for name in ('T1_story0_word', 'T1_story1_word'):
        ref = read(t1_review/name/'reference.json')
        for i,r in enumerate(ref['strokes']):
            if r['kind'] == 'text':
                raw.append(np.asarray(r['points'], np.float32)); buckets.append('plain_text')
                labels.append(json.dumps({'case':name,'stroke':i}))
    # 相同首字槽内的不同字形笔画：避免把跨行/跨字槽的布局间距当字形分辨率。
    for char in string.ascii_letters+string.digits+".,!?'-":
        for i,e in enumerate(layout_text_pages(char, font)[0].body_elements):
            raw.append(e.rawxy); buckets.append('same_slot_glyph')
            labels.append(json.dumps({'char':char,'stroke':i}))
    for kind in ('bos','eop','eod'):
        raw.append(control_stroke(kind)); buckets.append(kind); labels.append(kind)
    return raw, np.asarray(buckets), np.asarray(labels)


@torch.no_grad()
def encode(generator, raw, fp16):
    values=[]
    for start in range(0,len(raw),128):
        with torch.autocast('cuda',dtype=torch.float16,enabled=fp16):
            values.append(generator.encode(raw[start:start+128]).float().cpu().numpy())
    return np.concatenate(values)


def geometry(raw, z16, z32, buckets):
    output={}
    for bucket in sorted(set(buckets)):
        idx=np.flatnonzero(buckets==bucket); z=z16[idx].astype(np.float64)
        unique_raw={hashlib.sha256(raw[i].tobytes()).hexdigest() for i in idx}
        unique=np.unique(z,axis=0)
        delta=z16[idx]-z32[idx]
        entry=dict(count=len(idx), unique_raw=len(unique_raw), unique_embeddings=len(unique),
                   coordinate_std=stats(z.std(0)), std_by_dimension=z.std(0).tolist(),
                   coordinate_mad=stats(1.4826*np.median(np.abs(z-np.median(z,axis=0)),axis=0)),
                   fp16_fp32_rms=stats(np.sqrt(np.mean(delta**2,axis=1))),
                   fp16_fp32_by_dimension=np.sqrt(np.mean(delta**2,axis=0)).tolist())
        if len(unique)>1:
            with threadpool_limits(limits=4):
                nearest=NearestNeighbors(n_neighbors=2,algorithm='brute',n_jobs=4).fit(unique)
                distances,_=nearest.kneighbors(unique)
            euclidean=distances[:,1]
            eig=np.linalg.eigvalsh(np.cov(z,rowvar=False)).clip(0)
            entry.update(nearest_unique_rms=stats(euclidean/math.sqrt(z.shape[1])),
                covariance_eigenvalues=eig.tolist(),
                covariance_participation_ratio=float(eig.sum()**2/max(float((eig**2).sum()),1e-30)),
                nearest_pair_gaussian_error={str(e):stats(ndtr(-euclidean/(2*e))) for e in EPSILONS})
        output[bucket]=entry
    return output


@torch.no_grad()
def decode(generator, embeddings, fp16=False):
    values=[]
    for start in range(0,len(embeddings),32):
        with torch.autocast('cuda',dtype=torch.float16,enabled=fp16):
            values.append(generator.decode(embeddings[start:start+32]))
    return np.concatenate(values)


@torch.no_grad()
def perturbations(generator, z, buckets, raw, out):
    rng=np.random.default_rng(42); chosen=[]
    for bucket in sorted(set(buckets)):
        idx=np.flatnonzero(buckets==bucket)
        chosen.extend(rng.choice(idx,min(len(idx),32),replace=False))
    chosen=np.array(chosen); tags=buckets[chosen]
    base=decode(generator,torch.from_numpy(z[chosen]).to(generator.device))
    noise=rng.normal(size=(len(chosen),8,z.shape[1])).astype(np.float32)
    curves=[]; report={}
    for eps in EPSILONS:
        latent=z[chosen,None,:]+eps*noise
        candidate=decode(generator,torch.from_numpy(latent.reshape(-1,z.shape[1])).to(generator.device))
        candidate=candidate.reshape(len(chosen),8,generator.max_points,2)
        difference=candidate-base[:,None]
        shift=difference.mean(2)
        rms=np.sqrt(np.mean(difference**2,axis=(2,3)))
        shape=np.sqrt(np.mean((difference-shift[:,:,None])**2,axis=(2,3)))
        entry={}
        for bucket in sorted(set(tags)):
            mask=tags==bucket
            entry[bucket]=dict(samples=int(mask.sum()),directions=8,point_coordinate_rms=stats(rms[mask]),
                centered_shape_rms=stats(shape[mask]),centroid_l2=stats(np.linalg.norm(shift[mask],axis=-1)),
                coordinate_rms_at_1024px=stats(rms[mask]*1024))
        report[str(eps)]=entry;curves.append(candidate[:,0])
        print(json.dumps({'decoder_noise_sigma':eps,'done':True}),flush=True)
    np.savez_compressed(out/'perturbations.npz',indices=chosen,bucket=tags,baseline=base,
        epsilons=np.asarray(EPSILONS),curves=np.stack(curves),raw=np.asarray([raw[i] for i in chosen],dtype=object))
    return report


@torch.no_grad()
def fixed_mixture(generator, references):
    results={}
    for task,records in references:
        mu,sigma,pi,target,kinds=target_predictions(generator,records,task)
        bare=(sigma-generator.model.cfg.sigma_eps_r).clamp_min(0.)
        rows={}
        for eps in EPSILONS:
            # 沿用当前 additive 参数化；另列硬截断，区分最小宽度与对全部σ的平移。
            for parameterization,wide in [('add',bare+eps),('clamp',sigma.clamp_min(eps)),
                                          ('quadrature',(bare.square()+eps**2).sqrt())]:
                residual=(target[:,None]-mu)/wide
                gamma=((pi+1e-12).log()-wide.log().sum(-1)-.5*residual.square().sum(-1)).softmax(-1)
                derivative=gamma[...,None]*(mu-target[:,None])/wide.square()
                nll=gmm_nll(target,mu,wide,pi,reduction='none')
                row={}
                for kind,mask in [('body',kinds<3),('control',kinds>=3)]:
                    if bool(mask.any()):
                        row[kind]=dict(nll=stats(nll[mask].cpu().numpy()),
                            mu_derivative_l2=stats(derivative[mask].flatten(1).norm(dim=1).cpu().numpy()),
                            dimensions_below_floor_fraction=float((sigma[mask]<eps).float().mean()),
                            posterior_sigma=stats(wide[mask, gamma[mask].argmax(-1)].cpu().numpy()))
                rows[f'{parameterization}:{eps}']=row
        results[task]=rows
    return results


def gradient_sweep(generator, records, task, *, quadrature=False):
    first=next(r['page_id'] for i,r in enumerate(records) if i and r['targetable'])
    records=[r for r in records if r['page_id']<=first]
    model=generator.model; named=list(model.r_pred.named_parameters()); parameters=[p for _,p in named]
    batch=to_device(collate_documents([document(records,task)]),generator.device)
    batch['task_id']=torch.tensor([TASKS.index(task)],device=generator.device)
    original=model.r_pred.head.sigma_eps
    report={}
    try:
        for eps in EPSILONS:
            model.r_pred.head.sigma_eps=0. if quadrature else eps
            hook=(model.r_pred.head.register_forward_hook(
                lambda module,inputs,result: (result[0],(result[1].square()+eps**2).sqrt(),result[2]))
                  if quadrature else None)
            try:
                with generator.autocast(): output=model(batch)
            finally:
                if hook is not None: hook.remove()
            losses={'body':LOSS_WEIGHTS[task]*output['content_sum']/output['content_count'],
                    'control':.05*output['control_sum']/output['control_count']}
            entry={}
            for label,loss in losses.items():
                gradients=torch.autograd.grad(loss*512,parameters,retain_graph=label=='body',allow_unused=True)
                finite=all(bool(torch.isfinite(g).all()) for g in gradients if g is not None)
                values=[g/512 if g is not None else None for g in gradients]
                entry[label]=dict(finite=finite,loss=float(loss),
                    parameter_gradient_l2=float(torch.stack([g.float().square().sum() for g in values if g is not None]).sum().sqrt()),
                    groups=group_norms(named,values,latent_units=model.cfg.latent_units,components=model.cfg.gmm_r))
                del gradients,values
            report[str(eps)]=entry
            del output,losses,loss
            print(json.dumps({'gradient_probe':task,'sigma_floor':eps,'parameterization':'quadrature' if quadrature else 'add',
                'finite':all(v['finite'] for v in entry.values())}),flush=True)
    finally:
        model.r_pred.head.sigma_eps=original
    return {'normalization':'单个首回答页：正文均值×任务权重、控制均值×0.05；FP16上游，loss scale512；不更新参数',
            'strokes':len(records),'values':report}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'tmp_smoke/r_dual_review_assets_20261009/t1_final.pt')
    parser.add_argument('--multitask',type=Path,default=ROOT/'tmp_smoke/r_formal_density_assets_20261009/final.pt')
    parser.add_argument('--samples',type=Path,default=ROOT/'tmp_smoke/ae_tau_film_review_20261007/geometry/film_strokes.npz')
    parser.add_argument('--out_dir',type=Path,default=ROOT/'tmp_smoke/latent_sigma_20261009')
    parser.add_argument('--add_quadrature',action='store_true',help='复用已有几何，只补平方和下界的只读分布/梯度测量')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.add_quadrature:
        report=read(args.out_dir/'measurements.json')
        for arm,path in [('t1',args.checkpoint),('multitask',args.multitask)]:
            generator,_,_=load_generator(path,'cuda');generator.model.encode_chunk_size=128
            before=model_hash(generator.model)
            refs=([('T1',read(ROOT/'tmp_smoke/r_dual_review_20261009/t1/T1_story0_word/reference.json')['strokes'])] if arm=='t1' else
                  [(task,read(ROOT/'tmp_smoke/r_formal_review_20261009'/name/'reference.json')['strokes']) for task,name in
                   [('T1','T1_story0'),('O1','O1_cat'),('S2','S2_scene0'),('S4','S4_scene0')]])
            report['t1_fixed_mixture' if arm=='t1' else 'original_multitask_fixed_mixture']=fixed_mixture(generator,refs)
            report[arm+'_quadrature_gradients']={task:gradient_sweep(generator,records,task,quadrature=True)
                                                for task,records in refs if arm=='t1' or task in ('O1','S4')}
            assert before==model_hash(generator.model)
            del generator;gc.collect();torch.cuda.empty_cache()
        write(args.out_dir/'measurements.json',report)
        print(json.dumps({'quadrature_complete':True}),flush=True)
        return
    args.out_dir.mkdir(parents=True,exist_ok=False)
    t1=ROOT/'tmp_smoke/r_dual_review_20261009/t1'; baseline=ROOT/'tmp_smoke/r_formal_review_20261009'
    generator,font,metadata=load_generator(args.checkpoint,'cuda');generator.model.encode_chunk_size=128
    before=model_hash(generator.model)
    raw,buckets,labels=gather(generator,font,args.samples,t1)
    z16=encode(generator,raw,True);z32=encode(generator,raw,False)
    print(json.dumps({'encoded_strokes':len(raw),'buckets':dict(Counter(buckets))}),flush=True)
    report={'precision':'AE FP16/FP32同笔比较；解码扰动FP32；R/GMM FP16上游+FP32头/损失',
            'checkpoint':str(args.checkpoint),'ae_source':metadata['ae_source'],
            'samples_sha256':hashlib.sha256(args.samples.read_bytes()).hexdigest(),
            'sigma_values':EPSILONS,'geometry':geometry(raw,z16,z32,buckets)}
    np.savez_compressed(args.out_dir/'embeddings.npz',fp16=z16,fp32=z32,bucket=buckets,metadata=labels)
    report['decoder_noise']=perturbations(generator,z32,buckets,raw,args.out_dir)
    refs=[('T1',read(t1/'T1_story0_word/reference.json')['strokes'])]
    report['t1_fixed_mixture']=fixed_mixture(generator,refs)
    report['t1_gradient']=gradient_sweep(generator,refs[0][1],'T1')
    report['t1_weights_unchanged']=before==model_hash(generator.model)
    assert report['t1_weights_unchanged']
    write(args.out_dir/'measurements.json',report)
    del generator;gc.collect();torch.cuda.empty_cache()
    generator,_,metadata=load_generator(args.multitask,'cuda');generator.model.encode_chunk_size=128
    before=model_hash(generator.model)
    refs=[(task,read(baseline/name/'reference.json')['strokes']) for task,name in
          [('T1','T1_story0'),('O1','O1_cat'),('S2','S2_scene0'),('S4','S4_scene0')]]
    report['original_multitask_fixed_mixture']=fixed_mixture(generator,refs)
    report['multitask_gradients']={task:gradient_sweep(generator,records,task) for task,records in refs if task in ('O1','S4')}
    report['multitask_weights_unchanged']=before==model_hash(generator.model)
    assert report['multitask_weights_unchanged']
    report['optimizer_steps']=0;report['cuda_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    write(args.out_dir/'measurements.json',report)
    print(json.dumps({'complete':True,'output':str(args.out_dir)}),flush=True)


if __name__=='__main__':
    main()
