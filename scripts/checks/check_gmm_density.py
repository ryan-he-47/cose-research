"""2026-10-09：众数解码的混合密度、窄峰选择和页生成接线短核对。"""
from pathlib import Path
import json
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.st.model import gmm_nll
from src.st.sampling import gmm_density_mode
from src.st.generation_pretrain import PageGenerator
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.pretrain_model import FrozenSketchLM
from src.page_document import StrokeElement, control_stroke


def main():
    torch.set_num_threads(2)
    report = {}
    for name, means, scales, weights in (
        ('overlap',[-.7,.7],[1.,1.],[.5,.5]),
        ('narrow_low_weight',[-3.,3.],[2.,.1],[.9,.1])):
        mu=torch.tensor(means).reshape(1,2,1)
        sigma=torch.tensor(scales).reshape(1,2,1)
        pi=torch.tensor([weights])
        result=gmm_density_mode(mu,sigma,pi,n_iter=128,tolerance=1e-5)
        grid=torch.linspace(-6,6,120001).reshape(-1,1)
        grid_density=-gmm_nll(grid,mu.expand(len(grid),-1,-1),sigma.expand(len(grid),-1,-1),pi.expand(len(grid),-1),reduction='none')
        score=float(-gmm_nll(result,mu,sigma,pi))
        assert score >= float(grid_density.max())-2e-6
        if name=='overlap': assert abs(float(result[0,0]))<1e-4
        else: assert float(result[0,0])>2.9
        report[name]=dict(mode=float(result[0,0]),log_density=score,grid_best=float(grid_density.max()))
    torch.manual_seed(42)
    mu=torch.randn(3,32,48)
    sigma=torch.rand_like(mu)+.01
    pi=torch.softmax(torch.randn(3,32),-1)
    mode=gmm_density_mode(mu,sigma,pi,n_iter=128,tolerance=1e-5)
    for k in range(32):
        assert bool((-gmm_nll(mode,mu,sigma,pi,reduction='none') >= -gmm_nll(mu[:,k],mu,sigma,pi,reduction='none')-1e-4).all())
    assert torch.equal(mode,gmm_density_mode(mu,sigma,pi,n_iter=128,tolerance=1e-5))
    report['48d_all_means_and_determinism']='passed'
    cfg=gpt2_small_pretrain_config()
    cfg.r_d_model,cfg.r_dff,cfg.r_layers,cfg.r_heads=32,64,2,4
    generator=PageGenerator(FrozenSketchLM(cfg).eval())
    prefix=(StrokeElement(control_stroke('bos'),0,False,'bos'),)
    for mode,decoder in (('greedy','greedy'),('density','greedy'),('density','density')):
        value=generator.generate(prefix,mode=mode,decoder_mode=decoder,max_new_strokes=2)
        assert value['generated_strokes']==2
        if mode=='density':
            assert min(r['density_log_gain_vs_top_pi'] for r in value['strokes'] if r['generated'])>=-1e-4
        report[f'page_{mode}_{decoder}']=value['stop_reason']
    print(json.dumps(report))


if __name__=='__main__': main()
