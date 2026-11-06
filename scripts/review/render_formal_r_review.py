"""2026-10-09：将正式R只读验收JSON渲染为可切换样张页和概览图。"""
from __future__ import annotations

# 2026-10-12：分类目录中的脚本也可从仓库根直接运行。
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))


import argparse
from collections import Counter
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def generated(value):
    return [r for r in value['strokes'] if r.get('generated')]


def pages(records):
    return sorted({r['page_id'] for r in records})


def svg_pages(records):
    result = []
    for page in pages(records):
        ink = []
        for record in records:
            if record['page_id'] != page:
                continue
            color = '#ba3c6b' if record['kind'] in ('eod','eop','bos') else '#126f9e' if record.get('generated') else '#333'
            points = ' '.join(f'{x:.4f},{y:.4f}' for x,y in record['points'])
            ink.append(f'<polyline points="{points}" stroke="{color}" fill="none" stroke-width=".0023"/>')
        result.append(f'<div class="page"><div class="pageid">第{page}页</div><svg viewBox="0 0 1 1"><rect width="1" height="1" fill="white"/><rect y=".9" width="1" height=".1" fill="#f1f5f9"/>{"".join(ink)}</svg></div>')
    return ''.join(result) or '<p>没有输出笔画</p>'


def plot_page(axis, records, overlay=()):
    axis.set(xlim=(0,1),ylim=(1,0),aspect='equal',xticks=[],yticks=[])
    axis.axhspan(.9,1,color='#f1f5f9')
    for record in overlay:
        if record['kind']=='sketch':
            points=np.asarray(record['points'])
            axis.plot(*points.T,color='#bbb',linewidth=.6)
    ids=pages(records)
    if not ids:
        return
    page=ids[0]
    for r in records:
        if r['page_id']!=page:
            continue
        points=np.asarray(r['points'])
        color='#ba3c6b' if r['kind'] in ('eod','eop','bos') else '#126f9e' if r.get('generated') else '#222'
        axis.plot(*points.T,color=color,linewidth=.8)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    args=parser.parse_args()
    root=args.review
    summary=read(root/'summary.json')
    checked=read(root/'checks.json')
    cases=[]
    for item in summary:
        name=f"{item['task']}_{item['label']}"
        directory=root/name
        ref=read(directory/'reference.json')
        greedy=read(directory/'greedy.json')
        sample=read(directory/'sample.json')
        target=[r for r in ref['strokes'] if r['targetable']]
        condition=[r for r in greedy['strokes'] if not r['generated']]
        case=dict(name=name,task=item['task'],label=item['label'],source=item['source'],reference=ref,
                  condition=condition,target=target,greedy=greedy,sample=sample,metrics=item['teacher_metrics'])
        for kind in ('teacher_forced','ae_reference'):
            if (directory/f'{kind}.json').exists():
                case[kind]=read(directory/f'{kind}.json')['strokes']
        cases.append(case)
    for title,selected,columns,filename in (
        ('Object sketches: first answer page', [c for c in cases if c['task']=='O1'],
         ('target','ae_reference','teacher_forced','greedy','sample'),'objects.png'),
        ('Scene drawing / selection: first answer page',[c for c in cases if c['task'] in ('S2','S4')],
         ('target','teacher_forced','greedy','sample'),'scenes.png'),
        ('Text outputs: first answer page',[c for c in cases if c['name'] in ('T1_story0','T1_story1','O2_cat','O2_fish','S1_scene0','S3_scene0','S5_scene0')],
         ('target','greedy','sample'),'text.png'),
        ('Text prediction with true previous strokes',[c for c in cases if c['task'] in ('T1','O2') and 'teacher_forced' in c],
         ('target','ae_reference','teacher_forced'),'text_teacher.png')):
        if not selected:
            continue
        fig,axes=plt.subplots(len(selected),len(columns),figsize=(2.8*len(columns),2.8*len(selected)),squeeze=False)
        for i,c in enumerate(selected):
            for j,column in enumerate(columns):
                records=generated(c[column]) if column in ('greedy','sample') else c[column]
                overlay=c['condition'] if c['task']=='S4' else ()
                plot_page(axes[i,j],records,overlay)
                extra=f" / {c[column]['stop_reason']}" if column in ('greedy','sample') else ''
                axes[i,j].set_title(f"{c['name']} / {column}{extra}",fontsize=8)
        fig.suptitle(title+'\nblack: source, blue: prediction, pink: control; selection overlaid on input scene',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.965))
        fig.savefig(root/filename,dpi=150)
        plt.close(fig)
    data=[]
    for c in cases:
        cdata=dict(name=c['name'],task=c['task'],label=c['label'],source=c['source'],metrics=c['metrics'],
                   condition=svg_pages(c['condition']),reference=svg_pages(c['target']),
                   greedy=svg_pages(generated(c['greedy'])),sample=svg_pages(generated(c['sample'])),
                   greedy_summary={k:c['greedy'][k] for k in ('stop_reason','generated_strokes','generated_pages','outside_page_strokes')},
                   sample_summary={k:c['sample'][k] for k in ('stop_reason','generated_strokes','generated_pages','outside_page_strokes')})
        condition_pages={r['page_id'] for r in c['reference']['strokes'] if not r['targetable'] and r['kind']=='text'}
        cdata['condition_text']=' '.join(value for key,value in c['reference']['page_texts'].items() if int(key) in condition_pages)
        if c['task']=='T1':
            cdata['condition_text']=c['source']['text'].split(' ')[0]
        for kind in ('teacher_forced','ae_reference'):
            cdata[kind]=svg_pages(c[kind]) if kind in c else None
        data.append(cdata)
    encoded=json.dumps(data,ensure_ascii=False).replace('</',r'<\/')
    header=f'R step {checked["step"]} · {checked["successful_inputs"]:,} 输入 · valid源20例 · sample温度0.5 / seed42'
    document=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>正式R收尾样张</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f4f6f8;color:#172535}header{padding:22px 30px;background:white;border-bottom:1px solid #dce2e8}h1{font-size:23px;margin:0 0 9px}main{padding:22px;max-width:1700px;margin:auto}select{font:inherit;padding:8px;margin-right:12px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}.panel{padding:15px;background:white;border:1px solid #dce2e8;border-radius:10px}.panel h3{margin:0 0 10px;font-size:17px}.page svg{width:100%;border:1px solid #dce2e8;box-sizing:border-box}.pageid{font-size:12px;color:#64748b;margin:8px 0}.page{margin-bottom:12px}.extra{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#edf2f6;padding:13px}details{margin:15px 0}#status{padding:12px 0;color:#48576b}#source{font-size:14px;white-space:pre-wrap;padding:12px 0;line-height:1.7}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.grid,.extra{grid-template-columns:1fr}}</style>
<header><h1>正式R收尾：实际生成样张</h1><div>HEADER</div></header><main>
<select id="task"></select><select id="example"></select><div id="source"></div><div id="status"></div>
<div class="grid"><section class="panel"><h3>实际条件</h3><div id="condition"></div></section><section class="panel"><h3>源回答参考</h3><div id="reference"></div></section><section class="panel"><h3>Greedy生成</h3><div id="greedy"></div></section><section class="panel"><h3>采样生成</h3><div id="sample"></div></section></div>
<div class="extra" id="extra"><section class="panel"><h3>AE重建源回答</h3><div id="ae_reference"></div></section><section class="panel"><h3>真实前文下预测下一笔</h3><p>每一步使用源笔画前文；控制符颜色取源目标类型。</p><div id="teacher_forced"></div></section></div>
<details><summary>本例前向指标与完整元信息</summary><pre id="metrics"></pre></details></main><script>
const cases=DATA;const byId=id=>document.getElementById(id);const task=byId('task'),example=byId('example');
for(const name of ['T1','O1','O2','S1','S2','S3','S4','S5']){const o=document.createElement('option');o.value=o.textContent=name;task.appendChild(o)}
function options(){example.replaceChildren();for(const c of cases.filter(c=>c.task===task.value)){const o=document.createElement('option');o.value=c.name;o.textContent=c.label;example.appendChild(o)}render()}
function render(){const c=cases.find(c=>c.name===example.value);for(const id of ['condition','reference','greedy','sample','ae_reference','teacher_forced'])byId(id).innerHTML=c[id]||'';byId('extra').style.display=c.teacher_forced?'grid':'none';const s=c.source;byId('source').textContent='条件文字：'+c.condition_text+(s.answer!==undefined?('\n源回答：'+JSON.stringify(s.answer)):s.label?('\n类别：'+s.label):('\n故事源：'+s.source_id));byId('status').textContent='Greedy：'+JSON.stringify(c.greedy_summary)+'　采样：'+JSON.stringify(c.sample_summary);byId('metrics').textContent=JSON.stringify({metrics:c.metrics,source:c.source},null,2)}
task.addEventListener('change',options);example.addEventListener('change',render);task.value='O1';options();
</script></html>'''.replace('HEADER',html.escape(header)).replace('DATA',encoded)
    (root/'index.html').write_text(document,encoding='utf-8')
    result=dict(cases=len(cases),page='index.html',montages=[p.name for p in (root/'objects.png',root/'scenes.png',root/'text.png',root/'text_teacher.png') if p.exists()],
                stop_counts={mode:dict(Counter(c[mode]['stop_reason'] for c in cases)) for mode in ('greedy','sample')})
    (root/'rendered.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
