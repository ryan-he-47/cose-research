"""2026-10-22：自由场景描述/问答/圈内识别及场景生成布局统计，本机零更新。"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from PIL import Image,ImageDraw
from scripts.demo.demo_text_readout import load_joint_readout
from scripts.demo.demo_pretrain_generation import source_prefix
from scripts.review.review_joint_stage import dataset_for,write
from scripts.review.diagnose_formal_r_health import model_hash
from src.st.text_readout_generation import generate_text_readout


def normalized(text):
    text=re.sub(r'[^a-z0-9]+',' ',text.lower()).strip()
    return re.sub(r'^(?:a|an|the) ','',text)


def answer_match(actual,expected,task):
    a,b=normalized(actual),normalized(expected)
    numbers=dict(zero='0',one='1',two='2',three='3',four='4',five='5')
    if task=='S3':
        if b in ('yes','no'):return bool(a.split()) and a.split()[0]==b
        if b.isdecimal():return numbers.get(a,a)==b
    return a==b


def caption_classes(text,classes):
    value=normalized(text);used=set();found=[]
    for label in sorted(classes,key=lambda c:len(normalized(c)),reverse=True):
        for match in re.finditer(r'(?<!\w)'+re.escape(normalized(label))+r'(?!\w)',value):
            positions=set(range(match.start(),match.end()))
            if not positions & used:
                used|=positions;found.append(label)
    return set(found)


def layout_stats(value,objects):
    body=[r for r in value['strokes'] if r['generated'] and r['kind'] not in ('bos','eop','eod')]
    page=body[0]['page_id'] if body else None
    ink=Image.new('1',(256,256));draw=ImageDraw.Draw(ink)
    for row in body:
        if row['page_id']==page:
            points=np.asarray(row['points'])*255
            draw.line([tuple(p) for p in points],fill=1,width=1)
    bitmap=np.asarray(ink,dtype=bool);mask=np.zeros_like(bitmap);hit=[]
    for obj in objects:
        x0,y0,x1,y1=np.clip(np.asarray(obj['bbox'])*255,0,255).astype(int)
        mask[y0:y1+1,x0:x1+1]=True
        hit.append(bool(bitmap[y0:y1+1,x0:x1+1].any()))
    return dict(first_page_body_strokes=sum(r['page_id']==page for r in body),
        total_body_strokes=len(body),ink_pixels=int(bitmap.sum()),target_boxes=len(objects),boxes_hit=sum(hit),
        ink_inside_target_boxes=float((bitmap & mask).sum()/max(1,bitmap.sum())),
        stop=value['stop_reason'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--captions',type=int,default=24)
    parser.add_argument('--scene_drawings',type=int,default=24)
    parser.add_argument('--density_only',action='store_true',help='少量条件的整个R混合密度众数解码')
    args=parser.parse_args();out=args.out
    if (out/'scene_abilities.json').exists() and not args.density_only:raise FileExistsError('已有场景能力摘要')
    torch.set_num_threads(4);started=time.monotonic()
    generator,head,vocab,layout,source=load_joint_readout(out/'model.pt')
    generator.model.encode_chunk_size=128
    bundle=json.loads((out/'probe_bundle.json').read_text(encoding='utf-8'));dataset=dataset_for(bundle,'valid',vocab.font)
    scenes=bundle['splits']['valid'].get('scene_indices',list(range(bundle['scene_count'])))
    if args.density_only:
        examples=[('O1',s['index'],s['label']) for s in bundle['splits']['valid']['samples'] if s['label'] in ('cat','bicycle')]
        examples.append(('S2',scenes[0],'scene0'));rows=[]
        for task,i,label in examples:
            doc=dataset.get_for_task(task,i)
            value=generator.generate(source_prefix(doc,task),mode='density',density_iterations=128,
                max_new_strokes=48,max_new_pages=3)
            row=dict(task=task,label=label,stop=value['stop_reason'],
                     generated_strokes=sum(r['generated'] for r in value['strokes']),
                     body_strokes=sum(r['generated'] and r['kind'] not in ('bos','eod','eop') for r in value['strokes']))
            rows.append(row);write(out/f'density_{task}_{label}.json',dict(metrics=row,result=value))
        assert model_hash(generator.model)==source['r_model_hash']
        write(out/'density_probe.json',dict(cases=rows,iterations=128,weights_unchanged=True,optimizer_updates=0))
        print(json.dumps(rows),flush=True);return
    (out/'scene_abilities').mkdir(exist_ok=True)
    rows=[]
    for task in ('S3','S5','S1'):
        indices=scenes[:args.captions] if task=='S1' else scenes
        for i in indices:
            document=dataset.get_for_task(task,i);prefix=source_prefix(document,task)
            value=generate_text_readout(generator,head,vocab,'',time.monotonic()+90,
                SimpleNamespace(constraint='prefix',generation_strokes=320 if task=='S1' else 96),layout,prefix=prefix)
            assert all(not a['generated'] and a['points']==e.rawxy.tolist() and a['page_id']==e.page_id
                       for a,e in zip(value['strokes'],prefix))
            actual=' '.join(value['generated_page_texts'].values()).strip()
            expected=' '.join(p.text for p in document.pages if p.targetable and p.text)
            row=dict(task=task,index=i,question=document.metadata.get('question'),expected=expected,generated=actual,
                query_kind=document.metadata.get('query',{}).get('kind',task),stop=value['stop_reason'],
                correct=answer_match(actual,expected,task))
            if task=='S1':
                wanted={o['label'] for o in document.metadata['objects']};found=caption_classes(actual,dataset.drawings.classes)
                correct=len(wanted & found)
                row.update(expected_classes=sorted(wanted),mentioned_classes=sorted(found),
                           class_precision=correct/max(1,len(found)),class_recall=correct/len(wanted))
                positions={o['position'] for o in document.metadata['objects']}
                mentioned_positions=caption_classes(actual,('upper left','upper center','upper right','left','center','right','lower left','lower center','lower right'))
                row.update(expected_positions=sorted(positions),mentioned_positions=sorted(mentioned_positions),
                           position_word_recall=len(positions & mentioned_positions)/len(positions))
            rows.append(row);write(out/'scene_abilities'/f'{task}_{i}.json',dict(metrics=row,result=value))
            print(json.dumps({k:row[k] for k in ('task','index','correct','stop')}),flush=True)
    drawings=[]
    for i in scenes[:args.scene_drawings]:
        document=dataset.get_for_task('S2',i)
        for mode in ('greedy','sample'):
            value=generator.generate(source_prefix(document,'S2'),mode=mode,temperature=.5,seed=42,max_new_strokes=192,max_new_pages=3)
            row=dict(index=i,mode=mode,**layout_stats(value,document.metadata['objects']))
            drawings.append(row);write(out/'scene_abilities'/f'S2_{i}_{mode}.json',dict(metrics=row,result=value))
    groups=defaultdict(list)
    for r in rows:groups[(r['task'],r['query_kind'])].append(r)
    by_query=[]
    for (t,k),v in groups.items():
        expected=Counter(normalized(r['expected']) for r in v)
        by_query.append(dict(task=t,kind=k,count=len(v),correct=sum(r['correct'] for r in v),
                             expected_distribution=dict(expected),majority_correct=max(expected.values())))
    metrics={t:dict(count=sum(r['task']==t for r in rows),correct=sum(r['task']==t and r['correct'] for r in rows)) for t in ('S1','S3','S5')}
    captions=[r for r in rows if r['task']=='S1']
    metrics['S1'].update(class_precision=float(np.mean([r['class_precision'] for r in captions])),
                         class_recall=float(np.mean([r['class_recall'] for r in captions])),
                         position_word_recall=float(np.mean([r['position_word_recall'] for r in captions])))
    metrics['S1']['reference_exact_match']=metrics['S1'].pop('correct')
    metrics['S2']={mode:dict(count=sum(r['mode']==mode for r in drawings),
        natural_end=sum(r['mode']==mode and r['stop']=='eod_eop' for r in drawings),
        mean_box_hit_fraction=float(np.mean([r['boxes_hit']/r['target_boxes'] for r in drawings if r['mode']==mode])),
        mean_ink_inside_target_boxes=float(np.mean([r['ink_inside_target_boxes'] for r in drawings if r['mode']==mode]))) for mode in ('greedy','sample')}
    assert model_hash(generator.model)==source['r_model_hash']
    report=dict(metrics=metrics,by_query=by_query,cases=rows,drawings=drawings,
        optimizer_updates=0,weights_unchanged=True,conditioning_unchanged=True,seconds=time.monotonic()-started)
    write(out/'scene_abilities.json',report);print(json.dumps(metrics),flush=True)


if __name__=='__main__':main()
