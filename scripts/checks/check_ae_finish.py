"""2026-10-07：新15k余弦端点、明确替换LR及FP16完整恢复短检查。"""
from pathlib import Path
import json
import math
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.st.ae_schedule import StepLRReplay


def main():
    out = ROOT/'tmp_smoke/ae_finish_check_20261007'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    full = StepLRReplay(json.loads((ROOT/'configs/ae_tau_film_finish_lr.json').read_text()))
    assert full.lr(40000) == 1e-4 and full.lr(55000) == 5e-6
    assert math.isclose(full.lr(47500), 5.25e-5, abs_tol=1e-16)
    assert all(full.lr(i) >= full.lr(i+1) for i in range(40000,55000))
    restored = StepLRReplay.from_state_dict(full.state_dict())
    assert all(full.lr(i) == restored.lr(i) for i in range(40000,55001))
    fixture = ROOT/'tmp_smoke/ae_direct_check_20261004'
    old = ROOT/'tmp_smoke/ae_tau_film_full_schedule_check_20261006_r1'
    profile = {'kind':'successful_step_cosine','start_step':3,'end_step':6,'start_lr':1e-4,'end_lr':5e-6}
    path = out/'lr.json'
    path.write_text(json.dumps(profile), encoding='utf-8')
    common = [sys.executable,'-u','scripts/hpc/train_ae_text5.py','--manifest',str(fixture/'data_args.json'),
              '--validation_samples',str(fixture/'samples.npz'),'--recipe',str(old/'recipe.json'),
              '--max_seconds','0','--log_every','1']
    rejected = subprocess.run(common+['--resume',str(old/'first/final.pt'),'--lr_replay',str(path),
                              '--steps','6','--output_dir',str(out/'rejected')],cwd=ROOT,capture_output=True,text=True)
    assert rejected.returncode != 0 and 'LR轨迹不同' in rejected.stderr
    for name, end, resume, replace in (('first',5,old/'first/final.pt',True),
                                       ('resumed',6,out/'first/final.pt',False),
                                       ('continuous',6,old/'first/final.pt',True)):
        command = common+['--resume',str(resume),'--output_dir',str(out/name),'--steps',str(end)]
        if replace:
            command += ['--lr_replay',str(path),'--replace_lr_schedule']
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    a = torch.load(out/'resumed/final.pt',map_location='cpu',weights_only=False)
    b = torch.load(out/'continuous/final.pt',map_location='cpu',weights_only=False)
    for key,value in a['model'].items():
        torch.testing.assert_close(value,b['model'][key],rtol=0,atol=0)
    for key,state in a['optimizer']['state'].items():
        for field,value in state.items():
            torch.testing.assert_close(value,b['optimizer']['state'][key][field],rtol=0,atol=0)
    for key in ('data','collator','stroke_counts','scaler','scheduler','step_lr_schedule'):
        assert a[key] == b[key],key
    assert a['step'] == 6 and a['optimizer']['param_groups'][0]['lr'] == 5e-6
    report = {'start_step':40000,'end_step':55000,'end_lr':5e-6,'additional_inputs':15000*4096,
              'schedule_restore_exact':True,'replacement_requires_flag':True,'fp16_adam_model_data_restore_exact':True}
    (out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
