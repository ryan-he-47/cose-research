"""2026-10-20：真实valid图形条件下自由写类别，补充逐笔教师强制指标。"""
import argparse
import html
import json
from pathlib import Path
import sys
import time
import re
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.demo.demo_text_readout import load_joint_readout
from scripts.demo.demo_pretrain_generation import source_prefix
from scripts.review.review_joint_stage import dataset_for,write
from scripts.review.diagnose_formal_r_health import model_hash
from src.st.text_readout_generation import generate_text_readout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--all_classes',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--bundle',type=Path)
    args=parser.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=True)
    if (out/'recognition.json').exists():raise FileExistsError('已存在识图探针结果')
    torch.set_num_threads(4);start=time.monotonic()
    checkpoint=args.checkpoint or out/'model.pt'
    generator,head,vocab,layout,source=load_joint_readout(checkpoint)
    generator.model.encode_chunk_size=128
    bundle=json.loads((args.bundle or out/'probe_bundle.json').read_text(encoding='utf-8'))
    dataset=dataset_for(bundle,'valid',vocab.font);rows=[]
    balanced=bundle['splits']['valid'].get('balanced_indices',[])
    indices=list(dict.fromkeys([*range(bundle['object_count']),*balanced])) if args.all_classes else list(range(bundle['object_count']))
    for i in indices:
        document=dataset.get_for_task('O2',i);prefix=source_prefix(document,'O2')
        value=generate_text_readout(generator,head,vocab,'',time.monotonic()+60,
            SimpleNamespace(constraint='prefix',generation_strokes=96),layout,prefix=prefix)
        reference=[dict(page_id=e.page_id,kind=e.kind,points=e.rawxy.tolist()) for e in prefix]
        assert all(not a['generated'] and a['points']==b['points'] and a['page_id']==b['page_id']
                   for a,b in zip(value['strokes'],reference))
        text=' '.join(value['generated_page_texts'].values()).strip()
        expected=document.metadata['label']
        row=dict(index=i,expected=expected,generated=text,correct=text.lower().strip(' .!?')==expected.lower(),
                 stop=value['stop_reason'],source_id=document.metadata['source_id'])
        norm=lambda s:re.sub(r'^(?:a|an|the) ','',re.sub(r'[^a-z0-9]+',' ',s.lower()).strip())
        row['normalized_correct']=norm(text)==norm(expected)
        rows.append(row);print(json.dumps(row),flush=True)
        write(out/f'recognition_{i:02d}.json',dict(**row,result=value))
    assert model_hash(generator.model)==source['r_model_hash']
    report=dict(count=len(rows),correct=sum(r['correct'] for r in rows),cases=rows,
                checkpoint=str(checkpoint.resolve()),step=source['step'],
                weights_unchanged=True,conditioning_unchanged=True,optimizer_updates=0,seconds=time.monotonic()-start)
    if args.all_classes:
        report['balanced']=dict(count=len(balanced),correct=sum(r['correct'] for r in rows if r['index'] in balanced),
            normalized_correct=sum(r['normalized_correct'] for r in rows if r['index'] in balanced))
        report['anchor32']=dict(count=bundle['object_count'],correct=sum(r['correct'] for r in rows if r['index']<bundle['object_count']))
    write(out/'recognition.json',report)
    table=f'<h2>{len(rows)}例valid自由识图</h2><p>原图条件、标准字体闭环，无词典和真实标签回馈。</p><table><tr><th>编号</th><th>真实类别</th><th>生成类别</th><th>一致</th></tr>'
    for r in rows:table+='<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ('index','expected','generated','correct'))+'</tr>'
    if not (out/'index.html').exists():(out/'index.html').write_text('<!doctype html><meta charset="utf-8">',encoding='utf-8')
    with (out/'index.html').open('a',encoding='utf-8') as file:file.write(table+'</table>')
    print(json.dumps({k:report[k] for k in ('count','correct','seconds')}),flush=True)


if __name__=='__main__':main()
