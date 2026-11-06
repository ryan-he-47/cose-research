"""顺序执行本机只读诊断；失败即停，可按已完成分项恢复。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    data=out/'data';results=out/'metrics'
    parent=Path('E:/Trash_things/cose/tmp_smoke/r_page_diffusion_final_20261103/run/last.pt')
    glyph=Path('tmp_smoke/ae_latent_geometry_20261103/glyphs.npz').resolve()
    jobs=[
        ('final_raw','denoising',out/'run/last.pt',['--modes','full,wrong_condition,previous_pages_removed,none','--shapes',str(glyph)]),
        ('parent_raw','denoising',parent,['--modes','full,wrong_condition,previous_pages_removed,none','--shapes',str(glyph)]),
        ('final_ema','denoising',out/'run/inference.pt',['--modes','full','--skip_decode']),
        ('pre_ema','denoising',out/'run/pre_decay_inference.pt',['--modes','full','--skip_decode']),
        ('final_raw','rollout',out/'run/last.pt',[]),
        ('parent_raw','rollout',parent,[]),
        ('final_raw','sampling',out/'run/last.pt',[]),
        ('parent_raw','sampling',parent,['--groups','valid_S4']),
    ]
    for name,mode,checkpoint,extra in jobs:
        command=[sys.executable,'-u','-m','scripts.review.probe_page_diffusion','--name',name,'--mode',mode,
                 '--checkpoint',str(checkpoint),'--data',str(data),'--out',str(results),'--batch','16',*extra]
        stamp=time.time()
        print(json.dumps(dict(event='start',model=name,mode=mode,command=command)),flush=True)
        with (out/f'{name}_{mode}.log').open('a',encoding='utf8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(event='done',model=name,mode=mode,seconds=time.time()-stamp)),flush=True)
    print(json.dumps(dict(event='complete')),flush=True)


if __name__=='__main__':main()
