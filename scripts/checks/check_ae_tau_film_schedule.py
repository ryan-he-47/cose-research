"""2026-10-06：FiLM单次48192步的升温/历史LR合并、FP16中途恢复短检查。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.st.ae_schedule import StepLRReplay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', type=Path, default=ROOT/'tmp_smoke/ae_tau_film_full_schedule_check_20261006')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    full = StepLRReplay(json.loads((ROOT/'configs/ae_tau_film_full_lr.json').read_text(encoding='utf-8')))
    tail = StepLRReplay(json.loads((ROOT/'configs/ae_lambda16_text5_continue_lr.json').read_text()))
    assert full.start_step == 0 and full.end_step == 48192 and full.warmup_steps == 1000
    assert all(full.lr(step) == 1e-4*step/1000 for step in range(1, 1001))
    assert all(full.lr(step) == 1e-4 for step in range(1000, 24155))
    assert all(full.lr(step) == tail.lr(step) for step in range(24154, 48193))
    restored = StepLRReplay.from_state_dict(full.state_dict())
    assert all(restored.lr(step) == full.lr(step) for step in range(48193))
    first_decay_step = next(step for step in range(24154, 48193) if full.lr(step) < 1e-4)
    recipe = json.loads((ROOT/'configs/pretrain_ae_lambda48_dct_tau_film_recipe.json').read_text())
    recipe.update(batch_size=64, microbatch_strokes=16, padded_point_budget=1024,
                  workers=2, warmup=2, validate_steps=3, save_steps=3)
    toy = {'kind':'successful_step_lr_replay', 'start_step':0, 'end_step':6,
           'start_lr':1e-4, 'end_lr':5e-6, 'warmup_steps':2,
           'points':[[0,1e-4],[3,1e-4],[6,5e-6]], 'reference':{'fixture':True}}
    recipe_path = args.out_dir/'recipe.json'
    profile_path = args.out_dir/'lr_fixture.json'
    recipe_path.write_text(json.dumps(recipe), encoding='utf-8')
    profile_path.write_text(json.dumps(toy), encoding='utf-8')
    fixture = ROOT/'tmp_smoke/ae_direct_check_20261004'
    for name, end, resume in (('first',3,None), ('resumed',6,args.out_dir/'first/final.pt'), ('continuous',6,None)):
        command = [sys.executable,'-u','scripts/hpc/train_ae_text5.py', '--manifest',str(fixture/'data_args.json'),
                   '--validation_samples',str(fixture/'samples.npz'), '--recipe',str(recipe_path),
                   '--output_dir',str(args.out_dir/name), '--steps',str(end), '--max_seconds','0', '--log_every','1']
        command.extend(('--resume',str(resume)) if resume else ('--lr_replay',str(profile_path)))
        with (args.out_dir/f'{name}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    first = torch.load(args.out_dir/'first/final.pt', map_location='cpu', weights_only=False)
    resumed = torch.load(args.out_dir/'resumed/final.pt', map_location='cpu', weights_only=False)
    whole = torch.load(args.out_dir/'continuous/final.pt', map_location='cpu', weights_only=False)
    assert first['optimizer']['param_groups'][0]['lr'] == 1e-4
    assert resumed['step'] == whole['step'] == 6 and resumed['optimizer']['param_groups'][0]['lr'] == 5e-6
    assert len(resumed['optimizer']['state']) == 124
    for key, value in resumed['model'].items():
        torch.testing.assert_close(value, whole['model'][key], rtol=0, atol=0)
    for key, state in resumed['optimizer']['state'].items():
        for field, value in state.items():
            torch.testing.assert_close(value, whole['optimizer']['state'][key][field], rtol=0, atol=0)
    for key in ('data','collator','stroke_counts','scaler','scheduler','step_lr_schedule'):
        assert resumed[key] == whole[key], key
    assert resumed['step_lr_schedule'] == toy and resumed['cosine_annealing'] is None
    assert all(int(state['step']) == 6 for state in resumed['optimizer']['state'].values())
    updates = [record for line in (args.out_dir/'continuous.log').read_text().splitlines()
               if (record:=json.loads(line)).get('success')]
    assert [record['lr'] for record in updates] == [StepLRReplay(toy).lr(step) for step in range(1,7)]
    report = {'target_steps':48192, 'successful_input_strokes':48192*4096,
              'warmup_updates':1000, 'first_decay_step':first_decay_step, 'end_lr':full.lr(48192),
              'all_original_tail_steps_exact':True, 'fp16_fixture_updates':6,
              'resumed_model_adam_data_scaler_exact':True, 'adam_entries':124}
    (args.out_dir/'checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
