"""诊断缓存/查询隔离须与生产单条路径相等，防止未来目标泄漏。"""
import torch
from scripts.review.page_diffusion_probe_core import cache_histories, predict_blocks, sample
from src.st.page_diffusion import PageDiffusionR, PageDiffusionConfig
from src.st.page_diffusion_sampling import clean_cache, sample_block


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model=PageDiffusionR(PageDiffusionConfig(width=64,layers=2,heads=4,ffn=128,condition=32)).eval()
    histories=[(torch.randn(n,48),torch.arange(n)//8,torch.arange(n)%8) for n in (11,19)]
    cache,valid=cache_histories(model,histories)
    value=torch.randn(2,3,16,48)
    times=torch.tensor([.03,.5,1.])
    pages,local=torch.tensor([2,3]),torch.tensor([5,1])
    output=predict_blocks(model,cache,valid,value,times,pages,local)
    maximum=0.
    for row,(z,p,l) in enumerate(histories):
        single=clean_cache(model,z[None],p[None],l[None])
        for col,t in enumerate(times):
            pp=pages[row].expand(1,16);ll=local[row]+torch.arange(16)[None]
            direct=model(value[row:row+1,col],pp,ll,torch.ones_like(pp),t.expand(1,16),
                         torch.ones(1,1,16,len(z)+16,dtype=torch.bool),past=single)['prediction']
            maximum=max(maximum,float((direct-output[row:row+1,col]).abs().max()))
    assert maximum<2e-6,maximum
    changed=value.clone();changed[:,1]*=100
    isolated=predict_blocks(model,cache,valid,changed,times,pages,local)
    assert torch.equal(output[:,0],isolated[:,0])
    noise=torch.randn(2,16,48)
    result,_=sample(model,cache,valid,noise,pages,local,steps=8)
    for row,(z,p,l) in enumerate(histories):
        single=clean_cache(model,z[None],p[None],l[None])
        direct,_,_=sample_block(model,single,int(pages[row]),int(local[row]),noise[row:row+1],steps=8)
        assert torch.allclose(direct,result[row:row+1],atol=2e-6,rtol=2e-5)
    # 无历史的填充缓存不能影响输出。
    empty=[(z[:0],p[:0],l[:0]) for z,p,l in histories]
    blank,mask=cache_histories(model,empty)
    result=predict_blocks(model,blank,mask,value,times,pages,local)
    assert torch.isfinite(result).all()
    print({'batch_vs_production_max_abs':maximum,'query_isolation':True,'sampler_equivalence':True,'empty_history_finite':True})


if __name__=='__main__':main()
