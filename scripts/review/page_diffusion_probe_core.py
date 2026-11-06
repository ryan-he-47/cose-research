"""只读诊断：变长历史缓存和相互隔离的多噪声查询，共用生产 R。"""
from __future__ import annotations

import numpy as np
import torch

from src.st.page_diffusion import PageDiffusionConfig, PageDiffusionR
from src.st.page_diffusion_codec import FrozenPageCodec
from src.st.page_diffusion_runtime import autocast

TIMES = (0., .001, .003, .01, .03125, .0625, .125, .25, .5, .75, .9, 1.)


def load_model(path, weights='model', device='cuda'):
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    model = PageDiffusionR(PageDiffusionConfig(**state['model_cfg'])).to(device).eval()
    model.load_state_dict(state[weights], strict=True)
    codec = FrozenPageCodec.from_state(state['codec']).to(device).eval()
    return model, codec, state


def selected(row):
    return row['blocks'][row['selected']]


def history(row, mode='full', donor=None, block=None):
    block = block or selected(row)
    end = block['start']
    z, pages, local = row['z'][:end], row['pages'][:end], row['local'][:end]
    if mode=='none':
        return z[:0], pages[:0], local[:0]
    if mode=='wrong':
        part = selected(donor)
        n = part['start']
        # 保留一份完整、真实的同任务条件，页号整体对齐到当前目标页。
        return donor['z'][:n], donor['pages'][:n]+block['page']-part['page'], donor['local'][:n]
    if mode=='wrong_condition':
        if row['task']=='T1':
            return history(row,'wrong',donor,block)
        other=selected(donor)
        keep=donor['pages'][:other['start']]<other['page']
        own=pages==block['page']
        return (torch.cat((donor['z'][:other['start']][keep],z[own])),
                torch.cat((donor['pages'][:other['start']][keep]+block['page']-other['page'],pages[own])),
                torch.cat((donor['local'][:other['start']][keep],local[own])))
    if mode=='previous_pages_removed':
        keep = pages==block['page']
        return z[keep], pages[keep], local[keep]
    if mode=='visual_removed':
        keep=(row['kinds'][:end]!=1)&(row['kinds'][:end]!=2)
        return z[keep],pages[keep],local[keep]
    return z, pages, local


@torch.inference_mode()
def cache_histories(model, histories):
    device = next(model.parameters()).device
    lengths = torch.tensor([len(x[0]) for x in histories], device=device)
    n, batch = max(1, int(lengths.max())), len(histories)
    values = torch.zeros(batch,n,48,device=device)
    pages = torch.zeros(batch,n,dtype=torch.long,device=device)
    local = torch.zeros_like(pages)
    for i,(z,p,l) in enumerate(histories):
        values[i,:len(z)] = z.to(device)
        pages[i,:len(z)] = p.to(device)
        local[i,:len(z)] = l.to(device)
    valid = torch.arange(n,device=device)[None,:] < lengths[:,None]
    causal = torch.ones(n,n,dtype=torch.bool,device=device).tril()
    allowed = causal[None] & valid[:,None,:]
    allowed |= torch.eye(n,dtype=torch.bool,device=device)[None] & ~valid[:,:,None]
    with autocast(device):
        result=model(values,pages,local,torch.zeros_like(pages),torch.zeros_like(values[:,:,0]),
                     allowed[:,None],return_cache=True)
    return result['cache'], valid


@torch.inference_mode()
def predict_blocks(model, cache, valid, values, times, pages, local):
    """values [B,N,16,48]；N 个查询块只共享历史、互相不可见。"""
    b,n,k,d=values.shape
    device=values.device
    groups=torch.arange(n,device=device).repeat_interleave(k)
    block_mask=groups[:,None]==groups[None,:]
    allowed=torch.cat((valid[:,None,:].expand(-1,n*k,-1),block_mask[None].expand(b,-1,-1)),dim=-1)
    p=pages[:,None,None].expand(-1,n,k).reshape(b,n*k)
    l=(local[:,None,None]+torch.arange(k,device=device)[None,None,:]).expand(-1,n,-1).reshape(b,n*k)
    t=times[None,:,None].expand(b,-1,k).reshape(b,n*k) if times.ndim==1 else times[:,:,None].expand(-1,-1,k).reshape(b,n*k)
    with autocast(device):
        result=model(values.reshape(b,n*k,d),p,l,torch.ones_like(p),t,allowed[:,None],past=cache)
    return result['prediction'].float().reshape(b,n,k,d)


def targets(model, rows, blocks=None):
    eop=model.normalize(model.eop_latent).detach().cpu()
    z,body,control,padding=[],[],[],[]
    for row,part in zip(rows,blocks or [selected(r) for r in rows]):
        ids=part['target']
        z.append(torch.stack([row['z'][i] if i>=0 else eop for i in ids]))
        body.append([i>=0 and row['kinds'][i]<3 for i in ids])
        control.append([i>=0 and row['kinds'][i]>=4 for i in ids])
        padding.append([i<0 for i in ids])
    device=next(model.parameters()).device
    return tuple(torch.as_tensor(np.asarray(x) if name else torch.stack(x),device=device)
                 for name,x in enumerate((z,body,control,padding)))


def common_noise(rows, seed=20261105, salt=0):
    import hashlib
    arrays=[]
    for row in rows:
        digest=int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:4],'big')
        rng=np.random.default_rng(np.random.SeedSequence([seed,digest,salt]))
        arrays.append(rng.standard_normal((16,48)).astype(np.float32))
    return torch.from_numpy(np.stack(arrays))


@torch.inference_mode()
def sample(model,cache,valid,noise,pages,local,steps=32,power=1.,record=False):
    value=noise[:,None].float()
    records=[]
    grid=np.linspace(1.,0.,steps+1)**power
    # 与生产sample_block一样，在整条轨迹内复用FP16权重转换缓存。
    with autocast(value.device):
        for t,next_t in zip(grid[:-1],grid[1:]):
            pred=predict_blocks(model,cache,valid,value,torch.tensor([t],device=value.device),pages,local)
            updated=(next_t/t)*value+(1-next_t/t)*pred
            if record:
                records.append(dict(t=float(t),next_t=float(next_t),value=value[:,0].cpu(),
                                    prediction=pred[:,0].cpu(),update=(updated-value)[:,0].cpu()))
            value=updated
    return value[:,0],records


@torch.inference_mode()
def decode(codec,model,z,points=32):
    shape=z.shape[:-1]
    z=z.reshape(-1,48)
    curves=[]
    device=z.device
    tau=torch.linspace(0,1,points,device=device)
    for i in range(0,len(z),64):
        latent=model.denormalize(z[i:i+64])
        with autocast(device):
            mu,_,pi=codec.decoder(latent,tau[None].expand(len(latent),-1))
        ids=pi.argmax(-1)[...,None,None].expand(-1,-1,1,2)
        curves.append(mu.float().gather(2,ids).squeeze(2)*.6+mu.new_tensor([.5,.6]))
    return torch.cat(curves).reshape(*shape,points,2)


def errors(pred,target,body,basis):
    err=pred-target[:,None]
    axes=err@basis
    per_stroke=torch.stack((err.square().mean(-1),axes[...,:2].square().sum(-1)/48,
                            axes[...,2:].square().sum(-1)/48),dim=-1)
    return ((per_stroke*body[:,None,:,None]).sum(2)/body.sum(1)[:,None,None].clamp_min(1)).cpu().numpy()


def bootstrap(values,seed=20261105,repeats=1200):
    values=np.asarray(values,dtype=np.float64)
    rng=np.random.default_rng(seed)
    means=[]
    for i in range(0,repeats,100):
        ix=rng.integers(0,len(values),(min(100,repeats-i),len(values)))
        means.extend(values[ix].mean(1))
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n=len(values))
