"""冻结 D 的实际 GMM 读出、固定分量分支与局部 Jacobian；只读。"""
from __future__ import annotations

import gc
import json
import numpy as np
import torch
from scipy.spatial import cKDTree

from scripts.review.latent_geometry_metrics import quantiles


def run(args):
    from scripts.review.review_latent_geometry import load_codec,normalization,write
    destination=args.out/'smoothness.json'
    if destination.exists():
        return
    codec,_=load_codec(args)
    mean_np,scale=normalization(args)
    mean=torch.tensor(mean_np,device=args.device)
    tau=torch.linspace(0,1,76,device=args.device)
    rng=np.random.default_rng(args.seed)
    result=dict(version=2,perturbation='normalized latent per-coordinate RMS; direction length = sigma*sqrt(48)',groups={})
    partial=args.out/'smoothness_partial.json'
    if partial.exists():
        previous=json.loads(partial.read_text(encoding='utf8'))
        if previous.get('version')==2:
            result=previous
        else:
            write(args.out/'smoothness_before_components.json',previous)

    @torch.no_grad()
    def decode(z,base_ids=None,half=False):
        values,branches,ids=[],[],[]
        for start in range(0,len(z),64):
            latent=torch.as_tensor(z[start:start+64],device=args.device)*scale+mean
            with torch.autocast('cuda',dtype=torch.float16,enabled=half and args.device=='cuda'):
                mu,_,pi=codec.decoder(latent,tau[None].expand(len(latent),-1))
            mu=mu.float()
            selected=pi.argmax(-1)
            def gather(i):
                return (mu.gather(2,i[...,None,None].expand(-1,-1,1,2)).squeeze(2)*.6+
                        mu.new_tensor([.5,.6])).cpu().numpy()
            values.append(gather(selected));ids.append(selected.cpu().numpy())
            if base_ids is not None:
                branches.append(gather(torch.as_tensor(base_ids[start:start+64],device=args.device)))
        return np.concatenate(values),np.concatenate(ids),np.concatenate(branches) if branches else None

    for name in ('object','scene','text'):
        if name in result['groups']:
            continue
        corpus='text' if name=='text' else 'ae'
        if not (args.out/f'{corpus}_encoded.npz').exists():
            print(json.dumps(dict(stage='smoothness_deferred',group=name,reason='source_not_encoded')),flush=True)
            continue
        rng=np.random.default_rng(args.seed+1009*('object','scene','text').index(name))
        with np.load(args.out/f'{corpus}_encoded.npz') as data:
            zall,groups,units=data['z'],data['group'],data['unit']
        with np.load(args.out/'data'/f'{corpus}_pipeline.npz') as data:
            raw_points,offsets=data['points'],data['offsets']
        selected=np.flatnonzero(groups==name)
        # 每个来源最多一笔用于尾部统计，避免某张复杂图形贡献几十个相关样本。
        shuffled=rng.permutation(selected)
        _,positions=np.unique(units[shuffled],return_index=True)
        chosen=shuffled[np.sort(positions)[:1024]]
        z=zall[chosen]
        base,base_ids,_=decode(z)
        half,half_ids,_=decode(z,half=True)
        baseline_span=[];reconstruction=[]
        for i,index in enumerate(chosen):
            points=raw_points[offsets[index]:offsets[index+1]]
            baseline_span.append(np.linalg.norm(np.ptp(points,axis=0)))
            ds=np.linalg.norm(base[i,:,None,:]-points[None,:,:],axis=-1)
            reconstruction.append((ds.min(1).mean()+ds.min(0).mean())/2)
        span=np.asarray(baseline_span)
        tree=cKDTree(np.asarray(zall[selected],np.float64))
        distances,neighbors=tree.query(np.asarray(z,np.float64),k=65,workers=4)
        patches=zall[selected[neighbors[:,1:]]]-z[:,None,:]
        patches=patches-patches.mean(1,keepdims=True)
        _,_,vh=np.linalg.svd(patches.astype(np.float64),full_matrices=False)
        random=rng.normal(size=z.shape);random/=np.linalg.norm(random,axis=1,keepdims=True)
        directions={'random':random,'local_high_variance':vh[:,0,:],'local_low_variance':vh[:,-1,:]}
        stats=dict(n=len(z),unique_sources=len(chosen),reconstruction_chamfer=quantiles(reconstruction),
                   raw_span=quantiles(span),fp16_vs_fp32_rms=quantiles(np.sqrt(((half-base)**2).sum(-1).mean(-1))),
                   fp16_component_change=quantiles((half_ids!=base_ids).mean(-1)),perturbations={})
        preview_changed=None
        preview_ids=None
        for direction,unit in directions.items():
            stats['perturbations'][direction]={}
            for sigma in (.0001,.001,.003,.01,.03,.1):
                changed,ids,fixed=decode(z+np.float32(sigma*np.sqrt(48))*unit.astype(np.float32),base_ids)
                rms=np.sqrt(((changed-base)**2).sum(-1).mean(-1))
                delta=changed-base
                translation=delta.mean(1)
                shape=np.sqrt(((delta-translation[:,None,:])**2).sum(-1).mean(-1))
                position=np.linalg.norm(translation,axis=-1)
                endpoints=np.sqrt((delta[:,[0,-1],:]**2).sum(-1).mean(-1))
                fixed_rms=np.sqrt(((fixed-base)**2).sum(-1).mean(-1))
                switched=(ids!=base_ids).mean(-1)
                if direction=='random' and sigma==.01:
                    preview_changed=changed.copy()
                    order=np.argsort(rms/np.maximum(span,1e-6))
                    preview_ids=order[[len(order)//2,int(.95*(len(order)-1)),len(order)-1]]
                stats['perturbations'][direction][str(sigma)]=dict(curve_rms=quantiles(rms),
                    relative_to_span=quantiles(rms/np.maximum(span,1e-6)),
                    fixed_branch_rms=quantiles(fixed_rms),component_switch_fraction=quantiles(switched),
                    translation_rms=quantiles(position),centered_shape_rms=quantiles(shape),
                    endpoint_rms=quantiles(endpoints),shape_relative_to_span=quantiles(shape/np.maximum(span,1e-6)),
                    rms_decomposition_max_error=float(np.abs(rms**2-position**2-shape**2).max()),
                    switch_extra_rms=quantiles(np.sqrt(((changed-fixed)**2).sum(-1).mean(-1))))
            print(json.dumps(dict(stage='smoothness',group=name,direction=direction,n=len(z))),flush=True)
        # Jacobian 对实际 argmax 所选连续分支求导；有限扰动另行捕获分量切换。
        def single(value):
            mu,_,pi=codec.decoder(value[None]*scale+mean,tau[None])
            index=pi.argmax(-1)
            curve=mu.gather(2,index[...,None,None].expand(-1,-1,1,2)).squeeze(2)*.6
            return curve.reshape(-1)/np.sqrt(76.)
        spectra=[]
        for start in range(0,min(128,len(z)),4):
            batch=torch.as_tensor(z[start:start+4],device=args.device)
            jac=torch.vmap(torch.func.jacfwd(single))(batch)
            spectra.extend(torch.linalg.svdvals(jac).detach().cpu().numpy())
        singular=np.asarray(spectra)
        stats['jacobian']=dict(n=len(singular),largest_singular=quantiles(singular[:,0]),
            frobenius=quantiles(np.sqrt((singular**2).sum(1))),
            median_spectrum=np.median(singular,axis=0).tolist(),
            units='page-coordinate curve RMS per normalized latent Euclidean unit; fixed argmax branch derivative')
        result['groups'][name]=stats
        np.savez_compressed(args.out/f'smoothness_{name}.npz',indices=chosen,z=z,base=base,
                            span=span,jacobian_singular_values=singular,
                            preview_changed=preview_changed,preview_ids=preview_ids)
        write(args.out/'smoothness_partial.json',result)
    if len(result['groups'])==3:
        write(destination,result)
