"""顺序补充保留文字问题的视觉移除检查，以及固定来源自由生成。"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    parent=Path('E:/Trash_things/cose/tmp_smoke/r_page_diffusion_final_20261103')
    def run(command,label):
        print(json.dumps(dict(event='start',name=label)),flush=True)
        with (out/f'{label}.log').open('a',encoding='utf8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(event='done',name=label)),flush=True)
    if not (out/'data_uniform_text/valid_T1.pt').exists():
        run([sys.executable,'-u','-m','scripts.review.page_diffusion_uniform_text','--out',str(out),'--phase','encode'],'uniform_text_encode')
    glyph=Path('tmp_smoke/ae_latent_geometry_20261103/glyphs.npz').resolve()
    for name,checkpoint in (('final_raw',out/'run/last.pt'),('parent_raw',parent/'run/last.pt'),
                            ('final_ema',out/'run/inference.pt'),('pre_ema',out/'run/pre_decay_inference.pt')):
        extra=['--modes','full,wrong_condition,previous_pages_removed,none','--shapes',str(glyph)] if name.endswith('raw') else ['--modes','full','--skip_decode']
        run([sys.executable,'-u','-m','scripts.review.probe_page_diffusion','--name',name,'--checkpoint',str(checkpoint),
             '--data',str(out/'data_uniform_text'),'--out',str(out/'metrics_uniform_text'),'--batch','16',*extra],name+'_uniform_text')
    for name,mode,checkpoint in (('final_raw','rollout',out/'run/last.pt'),('parent_raw','rollout',parent/'run/last.pt'),('final_raw','sampling',out/'run/last.pt')):
        run([sys.executable,'-u','-m','scripts.review.probe_page_diffusion','--name',name,'--mode',mode,'--checkpoint',str(checkpoint),
             '--data',str(out/'data_uniform_text'),'--out',str(out/'metrics_uniform_text'),'--batch','16'],name+'_uniform_'+mode)
    for name,checkpoint in (('final_visual',out/'run/last.pt'),('parent_visual',parent/'run/last.pt')):
        command=[sys.executable,'-u','-m','scripts.review.probe_page_diffusion','--name',name,
            '--checkpoint',str(checkpoint),'--data',str(out/'data'),'--out',str(out/'metrics'),
            '--batch','16','--groups','valid_O2,valid_S1,valid_S3,valid_S4,valid_S5',
            '--modes','full,visual_removed','--skip_decode']
        print(json.dumps(dict(event='start',name=name)),flush=True)
        with (out/f'{name}.log').open('a',encoding='utf8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(event='done',name=name)),flush=True)
    run([sys.executable,'-u','-m','scripts.review.page_diffusion_feedback_roundtrip','--out',str(out)],'feedback_roundtrip')
    panel=out/'panel';panel.mkdir(exist_ok=True)
    if not (panel/'cases.json').exists():shutil.copyfile(parent/'cases.json',panel/'cases.json')
    command=[sys.executable,'-u','-m','scripts.review.review_page_diffusion','--out',str(panel),
        '--run',str(out/'run'),'--checkpoint',str(out/'run/last.pt'),'--steps','32','--max_new_strokes','512',
        '--font','E:/Trash_things/cose/datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg',
        '--data','E:/Trash_things/cose/tmp_smoke/r_joint_statistical_data_20261027']
    print(json.dumps(dict(event='start',name='panel')),flush=True)
    with (out/'panel.log').open('a',encoding='utf8') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(event='complete')),flush=True)


if __name__=='__main__':main()
