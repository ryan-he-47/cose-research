"""逐笔字形探针：同形模板合并，去掉完整字界和笔数提供的类别捷径。"""
import json
import numpy as np
import torch
from scripts.review.latent_geometry_metrics import grouped_macro_accuracy


def run(args):
    from scripts.review.review_latent_geometry import write
    from src.text_renderer import VectorFont
    from src.st.text_readout import StrokeVocabulary
    from src.st.text_shape_readout import ShapeVocabulary
    destination=args.out/'stroke_discrimination.json'
    if destination.exists(): return
    vocab=StrokeVocabulary(VectorFont.from_svg(args.font))
    shapes=ShapeVocabulary(vocab)
    data=np.load(args.out/'glyphs.npz')
    parts=data['parts'];characters=data['characters'];labels=data['labels']
    rows=np.repeat(np.arange(len(parts)),parts)
    positions=np.concatenate([np.arange(n) for n in parts])
    shape_labels=np.array([shapes.identity_to_shape[str(characters[labels[r]]),int(p)] for r,p in zip(rows,positions)])
    ref=data['splits'][rows]==0;query=~ref
    features=data['latent'][rows,positions]
    varied=data['varied'][rows,positions]
    raw=data['raw'][rows,positions]
    result=dict(classes=shapes.ink_classes,shape_vocabulary=shapes.dump(),modes={},noise={},
                scope='single-stroke shape identity, same-shape aliases merged; no glyph-boundary or stroke-count conditioning')

    def classify(reference,queries):
        r=torch.as_tensor(reference.reshape(len(reference),-1),device=args.device,dtype=torch.float32)
        predictions=[]
        for start in range(0,len(queries),128):
            q=torch.as_tensor(queries[start:start+128].reshape(len(queries[start:start+128]),-1),device=args.device,dtype=torch.float32)
            indices=torch.cdist(q,r,compute_mode='donot_use_mm_for_euclid_dist').argmin(1).cpu().numpy()
            predictions.extend(shape_labels[ref][indices])
        return grouped_macro_accuracy(np.asarray(predictions)==shape_labels[query],shape_labels[query],rows[query],seed=args.seed)

    for name,values in (('latent_fixed',features),('latent_varied',varied),('raw_fixed',raw),('raw_centered',raw-raw.mean(1,keepdims=True))):
        result['modes'][name]=classify(values[ref],values[query])
        print(json.dumps(dict(stage='stroke_knn',mode=name,macro=result['modes'][name]['macro_recall'])),flush=True)
    epsilon=np.random.default_rng(args.seed+3).normal(size=features[query].shape).astype(np.float32)
    for t in (0.,.01,.02,.04,.08,.16,.32,.5):
        result['noise'][str(t)]=classify(features[ref],features[query]+t/(1-t)*epsilon)
        print(json.dumps(dict(stage='stroke_noise',t=t,macro=result['noise'][str(t)]['macro_recall'])),flush=True)
    write(destination,result)
