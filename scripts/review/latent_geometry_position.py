"""几何主方向与真实弧长质心的线性关联；不拟合预测模型。"""
import json
import numpy as np


def run(args):
    from scripts.review.review_latent_geometry import write
    destination=args.out/'position_geometry.json'
    result=json.loads(destination.read_text(encoding='utf8')) if destination.exists() else {}
    for corpus in ('ae','r','text','validation'):
        if not (args.out/f'{corpus}_encoded.npz').exists(): continue
        expected={'ae':('overall','body','text','object','scene'),
                  'r':('overall','targets','text','object','scene'),
                  'text':('stratified_text',),'validation':('object','scene')}[corpus]
        if all(f'{corpus}_{name}' in result for name in expected): continue
        with np.load(args.out/f'{corpus}_encoded.npz') as encoded:
            z,group,kind,target=encoded['z'],encoded['group'],encoded['kind'],encoded['target']
        with np.load(args.out/'data'/f'{corpus}_pipeline.npz') as raw:
            points,offsets=raw['points'].astype(np.float64),raw['offsets']
        segment=np.r_[np.linalg.norm(np.diff(points,axis=0),axis=1),0.]
        segment[offsets[1:]-1]=0
        length=np.add.reduceat(segment,offsets[:-1])
        weighted=(points+np.roll(points,-1,axis=0))*.5*segment[:,None]
        center=np.add.reduceat(weighted,offsets[:-1],axis=0)/np.maximum(length[:,None],1e-30)
        zero=length==0
        center[zero]=points[offsets[:-1][zero]]
        span=np.linalg.norm(np.maximum.reduceat(points,offsets[:-1],axis=0)-np.minimum.reduceat(points,offsets[:-1],axis=0),axis=1)
        masks={'overall':np.ones(len(z),bool),**{g:group==g for g in ('object','scene','text')}}
        if corpus=='r': masks['targets']=target&~np.isin(kind,['bos','eop','eod'])
        if corpus=='ae': masks['body']=np.isin(kind,['text','sketch'])
        if corpus=='text': masks={'stratified_text':np.ones(len(z),bool)}
        if corpus=='validation': masks={g:group==g for g in ('object','scene')}
        for name,mask in masks.items():
            key=f'{corpus}_{name}'
            if key in result or mask.sum()<50: continue
            x=z[mask].astype(np.float64);c=center[mask]
            x-=x.mean(0);c-=c.mean(0)
            eig,vec=np.linalg.eigh(x.T@x/(len(x)-1));pc=x@vec[:,-2:]
            # 对两个主方向与两个质心坐标求典型相关；只使用协方差的闭式运算。
            cov=np.cov(np.c_[pc,c],rowvar=False)
            def invsqrt(a):
                e,v=np.linalg.eigh(a);return (v/np.sqrt(np.maximum(e,1e-15)))@v.T
            canonical=np.linalg.svd(invsqrt(cov[:2,:2])@cov[:2,2:]@invsqrt(cov[2:,2:]),compute_uv=False)
            cross=pc.T@c
            explained=np.trace(cross@np.linalg.pinv(c.T@c)@cross.T)/np.sum(pc**2)
            result[key]=dict(n=len(x),top2_variance_fraction=float(eig[-2:].sum()/eig.sum()),
                top2_centroid_canonical_correlations=canonical.tolist(),
                top2_variance_linearly_explained_by_centroid=float(explained),
                point_like_strokes=int((span[mask]<1e-6).sum()),tiny_span_below_001=int((span[mask]<.01).sum()),
                interpretation='descriptive covariance association with arc-length centroid; no neural readout trained')
        write(destination,result)
        print(json.dumps(dict(stage='position_geometry',corpus=corpus)),flush=True)
