"""续训状态核验和训练窗口统计；仅 CPU 读取。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.review.review_page_diffusion import summarize_window
from src.st.page_diffusion_runtime import write_json


def rows(path):
    return [json.loads(l) for l in path.read_text(encoding='utf8').splitlines() if l.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--parent',type=Path,default=Path('E:/Trash_things/cose/tmp_smoke/r_page_diffusion_final_20261103/run'))
    a=p.parse_args()
    torch.set_num_threads(4)
    s=torch.load(a.out/'run/last.pt',map_location='cpu',weights_only=False,mmap=True)
    old=torch.load(a.parent/'last.pt',map_location='cpu',weights_only=False,mmap=True)
    logs=rows(a.out/'run/train.jsonl');before=rows(a.parent/'train.jsonl')
    end=s['counters']['successful_inputs'];decay=s['scheduler']['spec']['total_inputs']-s['scheduler']['spec']['decay_inputs']
    audit=dict(step=s['step'],inputs=end,extra_inputs=end-old['counters']['successful_inputs'],
        adam_steps=sorted(set(int(v['step']) for v in s['optimizer']['state'].values())),scheduler=s['scheduler'],
        codec_unchanged=all(torch.equal(v,old['codec']['model'][k]) for k,v in s['codec']['model'].items()),
        model_cfg_unchanged=s['model_cfg']==old['model_cfg'],normalization_unchanged=s['normalization']==old['normalization'],
        finite_model=all(bool(torch.isfinite(v).all()) for v in s['model'].values()),
        finite_ema=all(bool(torch.isfinite(v).all()) for v in s['ema'].values()),
        logged_updates=len(logs),first_logged_step=logs[0]['step'],last_logged_step=logs[-1]['step'],
        seconds_per_update_median=float(np.median([r['seconds'] for r in logs])),
        loop_seconds=logs[-1]['wall_seconds'],inputs_per_second=(end-old['counters']['successful_inputs'])/logs[-1]['wall_seconds'],
        retries=sum(r.get('retries',0) for r in logs),oom=sum(r.get('oom_retries',0) for r in logs),
        windows=dict(parent_final200=summarize_window(before[-200:]),
            pre_decay200=summarize_window([r for r in logs if r['successful_inputs']<=decay][-200:]),
            final200=summarize_window(logs[-200:]),
            stable_earlier50m=summarize_window([r for r in logs if decay-100_000_000<r['successful_inputs']<=decay-50_000_000]),
            stable_last50m=summarize_window([r for r in logs if decay-50_000_000<r['successful_inputs']<=decay])))
    assert audit['adam_steps']==[s['step']]
    assert all(audit[k] for k in ('codec_unchanged','model_cfg_unchanged','normalization_unchanged','finite_model','finite_ema'))
    write_json(a.out/'training_audit.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('windows','scheduler')}))
    print(json.dumps(audit['windows']))


if __name__=='__main__':main()
