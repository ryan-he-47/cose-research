"""2026-10-05：历史LR对齐、λ48 FP16完整接续与中途恢复检查；小型fixture。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.st.ae_schedule import StepLRReplay,WallTimeCosine
from scripts.review.review_ae_pair import lines


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir',type=Path,default=ROOT/'tmp_smoke/ae_lambda48_continue_check_20261005')
    parser.add_argument('--reference_log',type=Path,default=ROOT/'tmp_smoke/ae_text5_continue_review_20261005/run/train.log')
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    spec=json.loads((ROOT/'configs/ae_lambda16_text5_continue_lr.json').read_text())
    replay=StepLRReplay(spec)
    reference=[v for v in lines(args.reference_log) if v.get('success')]
    assert all(replay.lr(v['step'])==v['lr'] for v in reference)
    assert replay.end_step-replay.start_step==24038
    assert replay.lr(24154)==replay.lr(40000)==1e-4 and replay.lr(48192)==5e-6
    assert all(replay.lr(step+1)<=replay.lr(step) for step in range(24154,48192))
    restored=StepLRReplay.from_state_dict(replay.state_dict())
    assert all(restored.lr(step)==replay.lr(step) for step in (24154,40600,40625,40700,48192))
    # 原墙钟类的恒定/余弦及保存恢复仍沿用。
    wall=WallTimeCosine([1e-4],5e-6,21000,hold_seconds=14400)
    assert wall.lrs(14400)==[1e-4] and wall.lrs(21000)==[5e-6]
    assert WallTimeCosine.from_state_dict(wall.state_dict(16000)).lrs(1000)==wall.lrs(17000)
    fixture=ROOT/'tmp_smoke/ae_direct_check_20261004'
    original=ROOT/'tmp_smoke/ae_lambda48_check_20261005/dct_resumed/final.pt'
    toy={'kind':'successful_step_lr_replay','start_step':3,'end_step':6,'start_lr':1e-4,'end_lr':5e-6,
         'reference':{'fixture':True},'points':[[3,1e-4],[4,1e-4],[6,5e-6]]}
    path=args.out_dir/'lr_fixture.json';path.write_text(json.dumps(toy),encoding='utf-8')
    for name,end,resume,new_lr in (('first',5,original,True),
                                   ('restored',6,args.out_dir/'first/final.pt',False),
                                   ('continuous',6,original,True)):
        command=[sys.executable,'-u','scripts/hpc/train_ae_text5.py','--manifest',str(fixture/'data_args.json'),
                 '--validation_samples',str(fixture/'samples.npz'),'--recipe',str(ROOT/'configs/pretrain_ae_lambda48_dct_recipe.json'),
                 '--resume',str(resume),'--steps',str(end),'--max_seconds','0','--log_every','1',
                 '--output_dir',str(args.out_dir/name)]
        if new_lr: command.extend(('--lr_replay',str(path)))
        with (args.out_dir/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    start=torch.load(original,map_location='cpu',weights_only=False)
    first=torch.load(args.out_dir/'first/final.pt',map_location='cpu',weights_only=False)
    resumed=torch.load(args.out_dir/'restored/final.pt',map_location='cpu',weights_only=False)
    whole=torch.load(args.out_dir/'continuous/final.pt',map_location='cpu',weights_only=False)
    assert first['step']==5 and resumed['step']==whole['step']==6
    assert first['optimizer']['param_groups'][0]['lr']==StepLRReplay(toy).lr(5)
    assert resumed['optimizer']['param_groups'][0]['lr']==5e-6
    for state in (first,resumed,whole):
        assert state['scheduler']==start['scheduler'] and state['cosine_annealing'] is None
        assert state['step_lr_schedule']==toy
        assert state['cfg']==start['cfg'] and state['train_args']==start['train_args']
        assert state['initial_ed_sha256']==start['initial_ed_sha256']
        assert len(state['optimizer']['state'])==116
        assert all(int(v['step'])==state['step'] for v in state['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for v in state['model'].values())
    for name,value in resumed['model'].items():
        torch.testing.assert_close(value,whole['model'][name],rtol=0,atol=0)
    for name,state in resumed['optimizer']['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(value,whole['optimizer']['state'][name][key],rtol=0,atol=0)
    for key in ('data','collator','stroke_counts','scaler','step_lr_schedule'):
        assert resumed[key]==whole[key],key
    updates=[v for v in lines(args.out_dir/'continuous.log') if v.get('success')]
    assert [v['lr'] for v in updates]==[StepLRReplay(toy).lr(step) for step in (4,5,6)]
    restored_start=next(v for v in lines(args.out_dir/'restored.log') if v.get('ae_direct_start'))
    assert restored_start['step']==5 and restored_start['scale']==first['scaler']['scale']
    assert restored_start['optimizer_state_entries']==116
    assert restored_start['kinds_cumulative']==first['stroke_counts']
    report={'reference_lr_records_exact':len(reference),'new_updates':24038,'target_step':48192,
            'fp16_fixture_steps':[3,5,6],'partial_lr':first['optimizer']['param_groups'][0]['lr'],
            'end_lr':5e-6,'adam_items':116,'plateau_unchanged':True,
            'resumed_model_and_adam_bitwise_equal_to_continuous':True,
            'data_collator_scaler_equal':True}
    (args.out_dir/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
