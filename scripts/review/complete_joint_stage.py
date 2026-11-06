"""2026-10-20：补充图形条件下的标准文字闭环，以及图形对照概览。"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.demo.demo_text_readout import load_joint_readout
from scripts.demo.demo_pretrain_generation import source_prefix
from scripts.review.review_joint_stage import dataset_for,write
from scripts.review.diagnose_formal_r_health import model_hash
from src.st.text_readout_generation import generate_text_readout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--localization_only',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--bundle',type=Path)
    args=parser.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=True)
    summary=None if args.localization_only else json.loads((out/'joint_summary.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4)
    generator,head,vocab,layout,source=load_joint_readout(args.checkpoint or out/'model.pt')
    generator.model.encode_chunk_size=128
    bundle=json.loads((args.bundle or out/'probe_bundle.json').read_text(encoding='utf-8'))
    dataset=dataset_for(bundle,'valid',vocab.font)
    if args.localization_only:
        rows=[]
        scenes=bundle['splits']['valid'].get('scene_indices',list(range(bundle['scene_count'])))
        for i in scenes:
            doc=dataset.get_for_task('S4',i)
            expected=np.concatenate([e.rawxy for e in doc.to_sequence() if e.targetable and e.kind=='selection'])
            saved=out/f'localization_{i:02d}.json'
            value=json.loads(saved.read_text(encoding='utf-8'))['result'] if saved.exists() else generator.generate(
                source_prefix(doc,'S4'),mode='greedy',max_new_strokes=8,max_new_pages=2)
            body=[r for r in value['strokes'] if r['generated'] and r['kind'] not in ('bos','eop','eod')]
            row=dict(index=i,question=doc.metadata['question'],stop=value['stop_reason'],generated_body_strokes=len(body),
                     query_kind=doc.metadata.get('query_kind'),
                     explicit_geometry_words=any(w in doc.metadata['question'].lower() for w in
                         ('leftmost','rightmost','uppermost','lowermost','left of','right of','above','below','bigger','smaller','biggest','smallest')),
                     target_instance_id=doc.metadata['target_instance_id'],nearest_object_correct=False,center_error=None,bbox_iou=None)
            if body:
                predicted=np.asarray(body[0]['points']);a=np.r_[expected.min(0),expected.max(0)];b=np.r_[predicted.min(0),predicted.max(0)]
                intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
                union=(a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection
                row.update(center_error=float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)),bbox_iou=float(intersection/max(1e-12,union)))
                nearest=min(doc.metadata['objects'],key=lambda o:np.linalg.norm(np.asarray(o['center'])-(b[:2]+b[2:])/2))
                row.update(predicted_nearest_label=nearest['label'],nearest_object_correct=nearest['instance_id']==row['target_instance_id'])
            rows.append(row);write(out/f'localization_{i:02d}.json',dict(metrics=row,result=value))
        assert model_hash(generator.model)==source['r_model_hash']
        report=dict(cases=rows,count=len(rows),measured=sum(r['center_error'] is not None for r in rows),
            nearest_object_correct=sum(r['nearest_object_correct'] for r in rows),
            median_center_error=float(np.median([r['center_error'] for r in rows if r['center_error'] is not None])),
            median_bbox_iou=float(np.median([r['bbox_iou'] for r in rows if r['bbox_iou'] is not None])),
            weights_unchanged=True,optimizer_updates=0,coordinate_system='unit page; bbox is ellipse bounding box')
        write(out/'localization.json',report)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,4,figsize=(13,10))
        import textwrap
        for i,ax in enumerate(axes.flat):
            doc=dataset.get_for_task('S4',i)
            ax.set(xlim=(0,1),ylim=(1,0),aspect='equal',xticks=[],yticks=[])
            for e in doc.to_sequence():
                if e.kind in ('sketch','selection'):ax.plot(*e.rawxy.T,color='#2f9e71' if e.kind=='selection' else '#bbb',lw=1)
            value=json.loads((out/f'localization_{i:02d}.json').read_text(encoding='utf-8'))['result']
            for r in value['strokes']:
                if r['generated'] and r['kind'] not in ('bos','eod','eop'):ax.plot(*np.asarray(r['points']).T,color='#2859a4',lw=1.2)
            ax.set_title(f'{i}: '+ '\n'.join(textwrap.wrap(rows[i]['question'],32)),fontsize=9)
        fig.suptitle('S4 / green: target ring; blue: generated ring',fontsize=13)
        fig.tight_layout();fig.savefig(out/'localization_overview.png',dpi=140);plt.close(fig)
        print(json.dumps(report),flush=True);return
    rows=[];started=time.monotonic()
    for case in summary['cases']:
        if case['task'] not in ('O2','S1','S3','S5'):continue
        document=dataset.get_for_task(case['task'],case['index'])
        value=generate_text_readout(generator,head,vocab,'',time.monotonic()+90,
            SimpleNamespace(constraint='prefix',generation_strokes=240),layout,prefix=source_prefix(document,case['task']))
        write(out/case['name']/'canonical.json',value)
        text=' '.join(value['generated_page_texts'].values()).strip()
        expected=' '.join(p.text for p in document.pages if p.targetable and p.text)
        row=dict(case=case['name'],task=case['task'],expected=expected,generated=text,stop=value['stop_reason'],
                 conditioning_refreshes=value['conditioning_refreshes'],font_audit=value['font_audit'])
        rows.append(row);print(json.dumps(row),flush=True)
    assert model_hash(generator.model)==source['r_model_hash']
    write(out/'conditional_text.json',dict(cases=rows,weights_unchanged=True,optimizer_updates=0,seconds=time.monotonic()-started))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=['O1_cat','O1_fish','O1_airplane','O1_bicycle','S2_scene0','S4_scene0']
    fig,axes=plt.subplots(len(selected),4,figsize=(12,16))
    for row,name in enumerate(selected):
        reference=json.loads((out/name/'reference.json').read_text(encoding='utf-8'))['strokes']
        for col,method in enumerate(('reference','teacher_forced','greedy','sample')):
            ax=axes[row,col];ax.set(xlim=(0,1),ylim=(1,0),aspect='equal',xticks=[],yticks=[])
            records=reference if method=='reference' else json.loads((out/name/(method+'.json')).read_text(encoding='utf-8'))['strokes']
            records=[r for r in records if r.get('targetable') or r.get('generated')]
            page=next((r['page_id'] for r in records if r['kind'] not in ('bos','eop','eod')),None)
            if name=='S4_scene0':
                for r in reference:
                    if r['kind']=='sketch':ax.plot(*np.asarray(r['points']).T,color='#d0d0d0',lw=.7)
            for r in records:
                if r['page_id']==page and r['kind'] not in ('bos','eop','eod'):
                    ax.plot(*np.asarray(r['points']).T,color='#2563a8',lw=1)
            ax.set_title(f'{name} / {method}',fontsize=9)
    fig.tight_layout();fig.savefig(out/'graph_overview.png',dpi=140);plt.close(fig)
    page=(out/'index.html').read_text(encoding='utf-8')
    table='<h2>图形条件下的分类文字闭环</h2><p>只给真实问题和图形前缀；字体笔顺约束，无词典，无真实回答回馈；只读推理。</p><table><tr><th>任务</th><th>真实回答</th><th>生成</th><th>停止</th></tr>'
    import html
    for r in rows:table+='<tr>'+''.join('<td>'+html.escape(r[k])+'</td>' for k in ('case','expected','generated','stop'))+'</tr>'
    (out/'index.html').write_text(page+table+'</table>',encoding='utf-8')


if __name__=='__main__':main()
