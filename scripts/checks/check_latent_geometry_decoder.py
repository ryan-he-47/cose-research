"""实际冻结D的前向Jacobian与固定分量中心差分核对。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.st.page_diffusion_codec import FrozenPageCodec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ae',type=Path,required=True)
    p.add_argument('--normalization',type=Path,required=True)
    p.add_argument('--encoded',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    codec=FrozenPageCodec.from_state(torch.load(args.ae,map_location='cpu',weights_only=False)).cuda().eval()
    norm=json.loads(args.normalization.read_text(encoding='utf8'))
    mean=torch.tensor(norm['mean'],device='cuda');scale=norm['scale']
    data=np.load(args.encoded);index=np.flatnonzero(data['group']=='object')[0]
    z=torch.as_tensor(data['z'][index],device='cuda')
    tau=torch.linspace(0,1,76,device='cuda')[None]
    with torch.no_grad():
        _,_,pi=codec.decoder(z[None]*scale+mean,tau)
        selected=pi.argmax(-1)
    def fixed(value):
        mu,_,_=codec.decoder(value[None]*scale+mean,tau)
        return (mu.gather(2,selected[...,None,None].expand(-1,-1,1,2)).squeeze(2)*.6).flatten()/np.sqrt(76.)
    jac=torch.func.jacfwd(fixed)(z)
    unit=torch.arange(1,49,device='cuda',dtype=torch.float32).sin();unit/=unit.norm()
    h=.001
    exact=jac@unit
    estimate=(fixed(z+h*unit)-fixed(z-h*unit))/(2*h)
    error=float((estimate-exact).norm()/exact.norm().clamp_min(1e-9))
    assert torch.isfinite(jac).all() and jac.shape==(152,48)
    assert error<.03,error
    print({'jacobian_shape':list(jac.shape),'relative_center_difference_error':error,'status':'passed'})


if __name__=='__main__': main()
