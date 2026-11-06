"""在一份实际训练更新上计算各任务对两个共享权重矩阵的梯度，权重全程只读。"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.review.page_diffusion_probe_core import cache_histories,history,load_model,targets
from scripts.review.replay_diffusion_balance import FIXTURE,as_tasks,array_by_task
from src.st.formal_recipe import TASKS
from src.st.page_diffusion_runtime import autocast,write_json


def summarize(gram,weights):
    norm=weights*np.sqrt(np.diag(gram))
    along=weights*(gram@weights)
    return dict(norm_share=as_tasks(norm/norm.sum()),signed_projection=as_tasks(along/along.sum()))


def apply_proposal(result,review,out):
    recipe=json.loads((review/'run/recipe.json').read_text(encoding='utf8'))
    proposal=json.loads((out/'balance_and_blocks.json').read_text(encoding='utf8'))['balance']
    for values in result['matrices'].values():
        gram=np.asarray(values['gram'])
        for key,weights in (('previous',recipe['loss_weights']),('equal_body',proposal['equal_coefficients']),
                            ('proposed',proposal['proposed_coefficients'])):
            values[key]=summarize(gram,array_by_task(weights))
    write_json(out/'actual_shared_gradients.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reweight_only',action='store_true',help='复用同一真实更新已计算的内积，仅更新配方对比')
    a=p.parse_args()
    if a.reweight_only:
        result=json.loads((a.out/'actual_shared_gradients.json').read_text(encoding='utf8'))
        apply_proposal(result,a.review,a.out)
        return
    torch.set_num_threads(4)
    model,codec,state=load_model(a.review/'run/last.pt')
    del codec,state
    device=next(model.parameters()).device
    model.requires_grad_(False)
    parameters=(model.output.weight,model.layers[-1].down.weight)
    for w in parameters:w.requires_grad_(True)
    fixture=torch.load(FIXTURE,map_location='cpu',weights_only=False,mmap=True)
    encoded=torch.load(a.review/'data/natural.pt',map_location='cpu',weights_only=False,mmap=True)
    grouped=defaultdict(list)
    assert len(encoded)==len(fixture['documents'])
    for doc,row in zip(fixture['documents'],encoded):
        assert doc.metadata['task']==row['task'] and len(doc.to_sequence())==len(row['z'])
        source=doc.metadata.get('source_id',doc.metadata.get('task_index'))
        assert str(source)==row['source']
        layout=doc.metadata['_page_diffusion']
        for index,part in enumerate(layout['blocks']):
            if any(i>=0 and row['kinds'][i]<3 for i in part['target']):
                grouped[row['task']].append((row,part,layout['noise'][index],layout['times'][index]))
    gradients=[[] for _ in parameters]
    counts=fixture['counts']['body']
    task_mse={}
    for task in TASKS:
        accum=[torch.zeros_like(w) for w in parameters]
        seen=0;error_sum=0.
        items=sorted(grouped[task],key=lambda x:x[1]['start'])
        for begin in range(0,len(items),16):
            rows,parts,noise,times=zip(*items[begin:begin+16])
            z,mask,_,_=targets(model,rows,parts)
            tau=torch.stack(times).to(device)
            value=(1-tau[:,None,None])*z+tau[:,None,None]*torch.stack(noise).to(device)
            past,valid=cache_histories(model,[history(r,block=b) for r,b in zip(rows,parts)])
            batch,k,_=value.shape
            pages=torch.tensor([b['page'] for b in parts],device=device)[:,None].expand(-1,k)
            local=torch.tensor([b['local'] for b in parts],device=device)[:,None]+torch.arange(k,device=device)
            allowed=torch.cat((valid[:,None,:].expand(-1,k,-1),torch.ones(batch,k,k,dtype=torch.bool,device=device)),dim=-1)
            with autocast(device):
                y=model(value,pages,local,torch.ones_like(local),tau[:,None].expand(-1,k),
                        allowed[:,None],past=past)['prediction']
            sq=((y.float()-z).square().mean(-1)*mask).sum()
            derivatives=torch.autograd.grad(sq*(256/counts[task]),parameters)
            for summed,g in zip(accum,derivatives):summed.add_(g.detach()/256)
            seen+=int(mask.sum());error_sum+=float(sq.detach())
        assert seen==counts[task]
        task_mse[task]=error_sum/seen
        for dest,g in zip(gradients,accum):
            assert torch.isfinite(g).all()
            dest.append(g.cpu().flatten())
    result=dict(fixture_inputs=sum(fixture['inputs'].values()),fixture_tasks=dict(counts),mse=task_mse,
                fixture_source=fixture['source'],matrices={})
    for name,parts in zip(('output.weight','layers[-1].down.weight'),gradients):
        flat=torch.stack(parts).double()
        gram=(flat@flat.T).numpy()
        result['matrices'][name]=dict(gram=gram.tolist())
    apply_proposal(result,a.review,a.out)
    print(json.dumps({k:v['proposed']['norm_share'] for k,v in result['matrices'].items()}))


if __name__=='__main__':
    main()
