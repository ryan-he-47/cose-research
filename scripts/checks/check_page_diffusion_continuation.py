"""接续WSD的全局时钟、旧日程兼容，以及退火边界检查点的游标/RNG保真。"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
from src.st.formal_recipe import InputWSD
from src.st.page_diffusion_runtime import save_checkpoint, _rng_state, _restore_rng
from scripts.hpc.prepare_page_diffusion_continuation import state_hash, file_hash


def scheduler(recipe,start_lr=0.):
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=.001)
    return InputWSD(optimizer,recipe,start_lr)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    old=dict(total_inputs=360000000,peak_lr=.0003,min_lr=.00003,warmup_inputs=3600000,decay_inputs=72000000)
    prior=scheduler(old)
    expected={0:0.,1800000:.00015,3600000:.0003,288000000:.0003,324000000:.000165,360000000:.00003}
    for count,lr in expected.items():assert abs(prior.lr_at(count)-lr)<1e-15
    assert 'warmup_start_inputs' not in prior.state_dict()['spec']
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    restored=scheduler(state['recipe'],state['scheduler']['spec']['start_lr'])
    restored.load_state_dict(state['scheduler'])
    start=restored.inputs;spec=restored.spec;end=spec['total_inputs']
    samples={count:restored.lr_at(count) for count in
             (start,start+1800000,start+3600000,end-72000000,end-36000000,end)}
    assert start==360000013 and end==1360000013
    assert abs(samples[start]-.00003)<1e-15
    assert abs(samples[start+1800000]-.000165)<1e-15
    assert samples[start+3600000]==samples[end-72000000]==.0003
    assert abs(samples[end]-.00003)<1e-15
    restored.prepare_update(65536);restored.step(65536)
    again=scheduler(state['recipe'],spec['start_lr']);again.load_state_dict(restored.state_dict())
    assert again.inputs==start+65536 and again.state_dict()==restored.state_dict()
    del state

    class Model(torch.nn.Linear):
        def configuration(self):return dict(test=True)
    model=Model(2,1);optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
    model(torch.ones(1,2)).sum().backward();optimizer.step();optimizer.zero_grad()
    clock=InputWSD(optimizer,dict(total_inputs=100,peak_lr=.0003,min_lr=.00003,warmup_inputs=10,decay_inputs=20),0.)
    clock.step(80)
    noise=torch.Generator().manual_seed(42)
    before=dict(stream={'cursor':80},noise_rng=noise.get_state(),rng=_rng_state())
    # 模拟训练线程已取下一批，但参数、Adam和输入时钟仍处于最后稳定边界。
    expected_noise=torch.randn(4,generator=noise);expected_global=torch.randn(4)
    stream=SimpleNamespace(state_dict=lambda:{'cursor':85})
    codec=SimpleNamespace(checkpoint=lambda:{'frozen':torch.ones(2)})
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    kwargs=dict(model=model,codec=codec,optimizer=optimizer,scaler=scaler,ema=deepcopy(model.state_dict()),
                scheduler=clock,stream=stream,noise_generator=noise,recipe={},data_args={},normalization={},
                source={},step=1,counters={'successful_inputs':80})
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);regular=root/'inference.pt';regular.write_bytes(b'previous_snapshot')
        save_checkpoint(root/'pre_decay.pt',**kwargs,boundary_state=before,inference_name='pre_decay_inference.pt')
        stable_hash=file_hash(root/'pre_decay.pt')
        snapshot=torch.load(root/'pre_decay.pt',weights_only=False)
        assert snapshot['stream']==before['stream']
        assert state_hash(snapshot['rng'])==state_hash(before['rng'])
        assert state_hash(snapshot['optimizer'])==state_hash(optimizer.state_dict())
        assert snapshot['scheduler']['inputs']==80 and regular.read_bytes()==b'previous_snapshot'
        noise.set_state(snapshot['noise_rng']);_restore_rng(snapshot['rng'])
        assert torch.equal(torch.randn(4,generator=noise),expected_noise)
        assert torch.equal(torch.randn(4),expected_global)
        with torch.no_grad():model.weight.add_(1)
        save_checkpoint(root/'last.pt',**kwargs)
        assert file_hash(root/'pre_decay.pt')==stable_hash
        assert (root/'pre_decay_inference.pt').exists()
    result=dict(passed=True,original_schedule_compatible=True,continued_schedule_restorable=True,
                boundary_cursor_and_rng_verified=True,stable_checkpoint_retained=True,
                schedule_samples=samples)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__':main()
