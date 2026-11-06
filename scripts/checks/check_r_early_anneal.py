"""2026-10-12：提前退火的LR连续性、完整FP16状态不变和分段恢复核对。"""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.st.formal_recipe import InputWSD


def exact(a,b):
    if torch.is_tensor(a): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: exact(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): exact(x,y)
    else: assert a==b,(a,b)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out_dir',type=Path,required=True)
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    def run(name,resume,steps,anneal=False):
        cmd=[sys.executable,'-u','src/st/train_formal.py','--data_manifest',
             str(ROOT/'tmp_smoke/formal_check_20261003/data_args.json'),
             '--resume',str(resume),'--output_dir',str(args.out_dir/name),
             '--steps',str(steps),'--log_every','1']
        if anneal: cmd += ['--anneal_to','1e-5']
        with (args.out_dir/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,(name,result.returncode)
    def load(name):return torch.load(args.out_dir/name/'final.pt',map_location='cpu',weights_only=False)
    run('before',args.parent,4)
    before=load('before')
    run('transition_only',args.out_dir/'before/final.pt',4,True)
    after=load('transition_only')
    for key in ('model','optimizer','scaler','rng','data','counters','cfg','data_manifest','step','phase_step'):
        exact(before[key],after[key])
    assert before['scheduler']['inputs']==after['scheduler']['inputs']
    allowed={'peak_lr','min_lr','decay_inputs','anneal_transition'}
    for key in set(before['train_args'])|set(after['train_args']):
        if key not in allowed:exact(before['train_args'][key],after['train_args'][key])
    optimizer=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=1e-4)
    scheduler=InputWSD(optimizer,after['train_args'],after['scheduler']['spec']['start_lr'])
    scheduler.load_state_dict(after['scheduler'])
    start=scheduler.inputs;end=after['train_args']['total_inputs']
    assert scheduler.lr_at(start)==1e-4
    assert abs(scheduler.lr_at((start+end)/2)-5.5e-5)<1e-15
    assert abs(scheduler.lr_at(end)-1e-5)<1e-15
    assert scheduler.lr_at(end+100)==scheduler.lr_at(end)
    run('continuous',args.out_dir/'before/final.pt',6,True)
    run('split',args.out_dir/'before/final.pt',5,True)
    run('resumed',args.out_dir/'split/final.pt',6)
    a,b=load('continuous'),load('resumed')
    for key in ('model','optimizer','scheduler','scaler','rng','data','counters','cfg','train_args'):
        exact(a[key],b[key])
    assert a['counters'].get('skipped_updates',0)==0
    from src.st.formal_assets import write_json
    report={'transition_keeps_full_model_adam_scaler_rng_cursor_exact':True,
        'lr_continuity_and_linear_endpoint':True,'same_original_total_inputs':True,
        'fp16_split_resume_exact':True,'success_steps':a['phase_step']}
    write_json(args.out_dir/'checks.json',report)
    print(report)


if __name__=='__main__':main()
