"""字形近邻间隔与真实 D 重建后的可辨性；不训练分类器或 R。"""
from __future__ import annotations

import json
import numpy as np
import torch
from scripts.review.latent_geometry_metrics import quantiles,stratified_bootstrap_accuracy


def run(args):
    from scripts.review.review_latent_geometry import load_codec,normalization,write,retrieve
    destination=args.out/'glyph_geometry.json'
    if destination.exists():
        return
    data=np.load(args.out/'glyphs.npz')
    ref=data['splits']==0;query=~ref
    parts,labels=data['parts'],data['labels']
    varied_raw=data['raw'].copy()
    shifts=data['origins']-np.array([.35,.45])
    for i,n in enumerate(parts):
        varied_raw[i,:n]+=shifts[i]
    result=dict(relationships={},decoded={})
    for mode,features in (('latent_fixed',data['latent']),('latent_varied',data['varied']),
                          ('raw_fixed',data['raw']),('raw_varied',varied_raw)):
        q_labels=labels[query]
        margins=np.empty(query.sum());same=np.empty(query.sum());different=np.empty(query.sum());purity=np.empty(query.sum())
        for n in np.unique(parts):
            ri=np.flatnonzero(parts[ref]==n);qi=np.flatnonzero(parts[query]==n)
            r=torch.as_tensor(features[ref][ri,:n].reshape(len(ri),-1),device=args.device)
            rlabels=torch.as_tensor(labels[ref][ri],device=args.device)
            for start in range(0,len(qi),128):
                indices=qi[start:start+128]
                q=torch.as_tensor(features[query][indices,:n].reshape(len(indices),-1),device=args.device)
                ds=torch.cdist(q,r,compute_mode='donot_use_mm_for_euclid_dist')
                eq=torch.as_tensor(q_labels[indices],device=args.device)[:,None]==rlabels[None]
                nearest_same=ds.masked_fill(~eq,float('inf')).min(1).values
                nearest_other=ds.masked_fill(eq,float('inf')).min(1).values
                ids=ds.topk(5,largest=False).indices
                same[indices]=nearest_same.cpu();different[indices]=nearest_other.cpu()
                purity[indices]=eq.gather(1,ids).float().mean(1).cpu()
        margins=(different-same)/(different+same+1e-12)
        result['relationships'][mode]=dict(nearest_same=quantiles(same),nearest_different=quantiles(different),
            relative_margin=quantiles(margins),five_neighbor_same_label_fraction=quantiles(purity),
            accuracy=stratified_bootstrap_accuracy(same<different,q_labels,seed=args.seed))
        print(json.dumps(dict(stage='glyph_relationship',mode=mode,accuracy=result['relationships'][mode]['accuracy']['macro_recall'])),flush=True)
    codec,geometry=load_codec(args)
    mean,scale=normalization(args)
    for mode in ('latent','varied'):
        predicted=np.zeros_like(data['raw'][query])
        features=data[mode][query]
        rows=np.repeat(np.arange(len(features)),parts[query])
        positions=np.concatenate([np.arange(n) for n in parts[query]])
        latent=features[rows,positions]*scale+mean
        for begin in range(0,len(latent),128):
            z=torch.as_tensor(latent[begin:begin+128],device=args.device)
            with torch.no_grad(),geometry.autocast():
                curves=geometry.decode(z)
            for offset,curve in enumerate(curves):
                index=begin+offset
                sampled=np.column_stack([np.interp(np.linspace(0,1,32),np.linspace(0,1,len(curve)),curve[:,axis]) for axis in range(2)])
                if mode=='varied':
                    sampled-=shifts[query][rows[index]]
                predicted[rows[index],positions[index]]=sampled
        labels_pred,distances=retrieve(data['raw'][ref],predicted,parts[ref],parts[query],labels[ref],device=args.device)
        result['decoded'][mode]=stratified_bootstrap_accuracy(labels_pred==labels[query],labels[query],seed=args.seed)
        result['decoded'][mode]['nearest_raw_distance']=quantiles(distances)
        result['decoded'][mode]['readout']='production FP16 D highest-pi means, 76 tau points; known origin removed for varied location'
        print(json.dumps(dict(stage='glyph_decoded',mode=mode,macro=result['decoded'][mode]['macro_recall'])),flush=True)
    write(destination,result)
