"""2026-10-09：末R的缓存/常规前向和GMM分量只读核对，不改变生成方法。"""
from __future__ import annotations

# 2026-10-12：分类目录中的脚本也可从仓库根直接运行。
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path

import torch

from scripts.demo.demo_pretrain_generation import load_generator, save_native_svg, source_prefix
from src.st.generation_pretrain import PageKVCache
from src.st.pipeline_pretrain import build_document_dataset, collate_documents
from scripts.review.review_formal_r import write_json


@torch.no_grad()
def compare_cache(generator, prefix):
    model,device = generator.model,generator.device
    with generator.autocast():
        embeddings=generator.encode([e.rawxy for e in prefix]).unsqueeze(0)
        ids=torch.tensor([[e.page_id for e in prefix]],device=device)
        pad=torch.zeros(ids.shape,device=device,dtype=torch.bool)
        full=model.hidden_states(embeddings,ids,pad,right_padded=True)
        cache=PageKVCache(model)
        cached=cache.forward(embeddings,ids,prefill=True)
        a=model.prediction_head(full[:,-1]); b=model.prediction_head(cached[:,-1])
        result=dict(prefix_strokes=len(prefix),prefill_hidden_max_error=float((full-cached).abs().max()),
                    prefill_head_max_errors=[float((x-y).abs().max()) for x,y in zip(a,b)],
                    prefill_component_equal=int(a[2][0].argmax())==int(b[2][0].argmax()),
                    first_component=int(a[2][0].argmax()),first_pi=float(a[2][0].max()))
        mu=a[0][:,int(a[2][0].argmax())]
        encoded=generator.encode([generator.decode(mu)[0]]).unsqueeze(0)
        next_ids=ids[:,-1:]+int(prefix[-1].kind=='eop')
        expanded=torch.cat((embeddings,encoded),1)
        expanded_ids=torch.cat((ids,next_ids),1)
        full=model.hidden_states(expanded,expanded_ids,torch.zeros_like(expanded_ids,dtype=torch.bool),right_padded=True)
        cached=cache.forward(encoded,next_ids)
        a=model.prediction_head(full[:,-1]); b=model.prediction_head(cached[:,-1])
        result.update(next_hidden_max_error=float((full[:,-1:]-cached).abs().max()),
                      next_head_max_errors=[float((x-y).abs().max()) for x,y in zip(a,b)],
                      next_component_equal=int(a[2][0].argmax())==int(b[2][0].argmax()))
    return result


@torch.no_grad()
def mixture_diagnostics(generator,document,output):
    model=generator.model
    cpu=collate_documents([document],max_strokes=1024,max_points=generator.max_points)
    batch={k:v.to(generator.device) if torch.is_tensor(v) else v for k,v in cpu.items()}
    with generator.autocast():
        emb=model.encode_valid(batch)
        hidden=model.hidden_states(emb[:,:-1],batch['slice_id'][:,:-1],batch['stroke_pad'][:,:-1],right_padded=True)
        selected=model.target_mask(batch)
        mu,sigma,pi=model.prediction_head(hidden[selected])
        target=emb[:,1:][selected].float()
        kinds=batch['kind_id'][:,1:][selected]
        with torch.autocast('cuda',enabled=False):
            score=torch.log(pi+1e-12)-torch.log(sigma).sum(-1)-.5*(((target[:,None]-mu)/sigma)**2).sum(-1)
        likely=pi.argmax(-1); posterior=score.argmax(-1)
        row=torch.arange(len(likely),device=generator.device)
        result={}
        for name,mask in (('body',kinds<3),('control',kinds>=3)):
            result[name]=dict(count=int(mask.sum()),
                top_pi_matches_target_posterior=float((likely[mask]==posterior[mask]).float().mean()),
                top_pi_mean=float(pi[row,likely][mask].mean()),
                posterior_component_pi_mean=float(pi[row,posterior][mask].mean()),
                top_pi_latent_mse=float(((mu[row,likely]-target)[mask]**2).mean()),
                posterior_component_latent_mse=float(((mu[row,posterior]-target)[mask]**2).mean()),
                top_pi_sigma_mean=float(sigma[row,likely][mask].mean()),
                posterior_component_sigma_mean=float(sigma[row,posterior][mask].mean()))
        # 文字首个目标页可视化：仍只用最高pi分量；未用真实目标挑分量。
        if document.metadata['task'] in ('T1','O2'):
            elements=[e for e in document.to_sequence()[1:] if e.targetable]
            page=elements[0].page_id
            selected_indices=[i for i,e in enumerate(elements) if e.page_id==page]
            greedy=generator.decode(mu[row,likely][selected_indices])
            reconstructed=generator.decode(target[selected_indices])
            for name,points in (('teacher_forced',greedy),('ae_reference',reconstructed)):
                value=dict(strokes=[dict(page_id=page,kind=elements[i].kind,generated=True,points=p.tolist()) for i,p in zip(selected_indices,points)],
                           teacher_forced=True,kind_labels='expected_target_kind',conditioning='true_previous_strokes' if name=='teacher_forced' else 'true_target_embedding')
                write_json(output/f'{name}.json',value)
                save_native_svg(value,output/f'{name}.svg',name)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out_dir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    generator,font,metadata=load_generator(args.checkpoint,'cuda')
    dataset=build_document_dataset(**{**metadata['data_args'],'split':'valid'})
    cases=[('O1',label,dataset.drawings.classes.index(label)) for label in ('cat','fish','airplane','bicycle')]
    cases += [('O2','cat',dataset.drawings.classes.index('cat')),('T1','story0',0),('S2','scene0',0)]
    result={}
    for task,label,index in cases:
        document=dataset.get_for_task(task,index)
        if task=='T1': document=document.window(0,1024)
        name=f'{task}_{label}'
        result[name]=dict(cache=compare_cache(generator,source_prefix(document,task)),
                          mixture=mixture_diagnostics(generator,document,args.out_dir/name))
        print(json.dumps({name:result[name]},ensure_ascii=False),flush=True)
    write_json(args.out_dir/'inference_diagnostics.json',result)


if __name__=='__main__': main()
