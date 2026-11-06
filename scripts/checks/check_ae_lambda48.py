"""2026-10-05：λ48两臂的共同初始化、无DCT前后向、FP16恢复及冻结R接口检查。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.hpc.train_ae_text5 import configure_sources
from src.st.ae_direct import ArcAECollator, DirectAEStream
from src.st.pretrain_model import FrozenSketchLM, config_from_ae_checkpoint
from src.st.train_ae import StrokeAutoencoder
from src.st.train_ae_direct import direct_ae_config, ed_fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,default=ROOT/'tmp_smoke/ae_direct_check_20261004')
    parser.add_argument('--out_dir',type=Path,default=ROOT/'tmp_smoke/ae_lambda48_check_20261005')
    parser.add_argument('--fp16',action='store_true')
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    recipes={arm:json.loads((ROOT/f'configs/pretrain_ae_lambda48_{arm}_recipe.json').read_text()) for arm in ('dct','nodct')}
    normalized=[]
    for recipe in recipes.values():
        shared={k:v for k,v in recipe.items() if k not in ('ae_config','experiment')}
        shared['ae_config']={k:v for k,v in recipe['ae_config'].items() if not k.startswith('dct_')}
        normalized.append(shared)
    assert normalized[0]==normalized[1]
    models={}
    for arm,recipe in recipes.items():
        torch.manual_seed(42)
        models[arm]=StrokeAutoencoder(direct_ae_config(recipe))
    assert ed_fingerprint(models['dct'])==ed_fingerprint(models['nodct'])
    assert models['nodct'].dct_head is None
    assert not any(name.startswith('dct_head.') for name in models['nodct'].state_dict())
    configure_sources(recipes['dct'])
    data_args=json.loads((args.fixture/'data_args.json').read_text())
    stream=DirectAEStream(data_args,batch_size=64,workers=0)
    records=stream.next_batch();stream.close()
    batches={arm:ArcAECollator(model.cfg,42)(records) for arm,model in models.items()}
    assert 'dct_target' not in batches['nodct']
    for key,value in batches['nodct'].items():
        if torch.is_tensor(value): torch.testing.assert_close(value,batches['dct'][key],rtol=0,atol=0)
    counts={}
    for arm,model in models.items():
        result=model(batches[arm]);result['loss'].backward()
        for name in ('encoder','decoder'):
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in getattr(model,name).parameters())
        if arm=='nodct':
            assert float(result['dct_mse'])==0
            torch.testing.assert_close(result['loss'],result['recon_total'],rtol=0,atol=0)
        else:
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.dct_head.parameters())
        counts[arm]=sum(p.numel() for p in model.parameters())
        frozen=FrozenSketchLM(config_from_ae_checkpoint({'cfg':asdict(model.cfg)}))
        frozen.load_ae({'model':model.state_dict()})
        assert frozen.cfg.latent_units==48
        for name,value in model.state_dict().items():
            torch.testing.assert_close(frozen.state_dict()[name],value,rtol=0,atol=0)
        assert all(not p.requires_grad for name,p in frozen.named_parameters() if name.startswith(('encoder.','decoder.','dct_head.')))
        assert (frozen.dct_head is None)==(arm=='nodct')
        del frozen;gc.collect()
    checks={'parameters':counts,'shared_ed_initial_sha256':ed_fingerprint(models['dct']),
            'only_dct_recipe_difference':True,'no_dct_head_or_target':True,'same_points_tau_targets':True,
            'cpu_gradients_finite':True,'frozen_r_load_both':True}
    del models,batches,result;gc.collect()
    if args.fp16:
        for arm,recipe in recipes.items():
            recipe.update(batch_size=64,microbatch_strokes=16,padded_point_budget=1024,
                          workers=2,warmup=2,validate_steps=2,save_steps=1)
            recipe_path=args.out_dir/f'{arm}_recipe.json'
            recipe_path.write_text(json.dumps(recipe),encoding='utf-8')
            for phase,steps,resume in (('first',2,None),('resumed',3,args.out_dir/f'{arm}_first/final.pt')):
                command=[sys.executable,'-u','scripts/hpc/train_ae_text5.py','--manifest',str(args.fixture/'data_args.json'),
                         '--validation_samples',str(args.fixture/'samples.npz'),'--recipe',str(recipe_path),
                         '--output_dir',str(args.out_dir/f'{arm}_{phase}'),'--steps',str(steps),'--max_seconds','0','--log_every','1']
                if resume: command.extend(('--resume',str(resume)))
                with (args.out_dir/f'{arm}_{phase}.log').open('w',encoding='utf-8') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            first=torch.load(args.out_dir/f'{arm}_first/final.pt',map_location='cpu',weights_only=False)
            state=torch.load(args.out_dir/f'{arm}_resumed/final.pt',map_location='cpu',weights_only=False)
            assert state['step']==3 and all(int(v['step'])==3 for v in state['optimizer']['state'].values())
            assert state['initial_ed_sha256']==first['initial_ed_sha256']==checks['shared_ed_initial_sha256']
            assert all(torch.isfinite(v).all() for v in state['model'].values())
            assert state['optimizer']['param_groups'][0]['lr']==recipe['lr']
            assert state['scheduler']==first['scheduler'] and state['cosine_annealing'] is None
            assert bool(any(k.startswith('dct_head.') for k in state['model']))==(arm=='dct')
            assert state['stroke_counts']['sketch']==3*state['data']['quotas']['object']+3*state['data']['quotas']['scene']
            checks[arm+'_fp16_steps']=[first['step'],state['step']]
            checks[arm+'_optimizer_state_entries']=len(state['optimizer']['state'])
    (args.out_dir/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checks,ensure_ascii=False))


if __name__=='__main__':
    main()
