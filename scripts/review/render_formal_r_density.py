"""2026-10-09：原最高π与全GMM众数样张并排展示。"""
from __future__ import annotations

# 2026-10-12：分类目录中的脚本也可从仓库根直接运行。
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))


import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.review.render_formal_r_review import generated, plot_page, read, svg_pages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args()
    cases=[]
    for item in read(args.review/'summary.json'):
        old=args.baseline/item['name']
        directory=args.review/item['name']
        source=read(old/'reference.json')
        value=dict(**item, target=[r for r in source['strokes'] if r['targetable']],reference=source)
        for mode in ('greedy','sample'):
            path=directory/f'{mode}.json'
            value[mode]=read(path if path.exists() else old/f'{mode}.json')
        for mode in ('density','density_both'):
            value[mode]=read(directory/f'{mode}.json')
        value['condition']=[r for r in value['greedy']['strokes'] if not r['generated']]
        for mode in ('teacher_forced','teacher_density','teacher_density_both'):
            path=(old if mode=='teacher_forced' else directory)/f'{mode}.json'
            if path.exists(): value[mode]=read(path)['strokes']
        cases.append(value)
    columns=('target','greedy','density','density_both')
    titles=('Source answer','Highest pi','GMM mode: R','GMM mode: R + D')
    montages=[]
    for group,selected in (
        ('objects',[c for c in cases if c['task']=='O1']),
        ('scenes',[c for c in cases if c['task'] in ('S2','S4')]),
        ('text',[c for c in cases if c['name'] in ('T1_story0','O2_cat','O2_fish','S1_scene0','S3_scene0','S5_scene0')])):
        fig,axes=plt.subplots(len(selected),4,figsize=(12,3*len(selected)),squeeze=False)
        for i,c in enumerate(selected):
            for j,column in enumerate(columns):
                records=c[column] if column=='target' else generated(c[column])
                plot_page(axes[i,j],records,c['condition'] if c['task']=='S4' else ())
                extra='' if column=='target' else f"\n{c[column]['generated_strokes']} strokes / {c[column]['stop_reason']}"
                axes[i,j].set_title(f"{c['name']} / {titles[j]}{extra}",fontsize=8)
        fig.suptitle('Same R / same conditions / first answer page\nblack: source, blue: prediction, pink: controls',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.965))
        name=f'{group}.png'
        fig.savefig(args.review/name,dpi=150)
        plt.close(fig)
        montages.append(name)
    data=[]
    for c in cases:
        value={k:c[k] for k in ('name','task','label','source','first_distribution','generation')}
        value.update(condition=svg_pages(c['condition']),reference=svg_pages(c['target']))
        for mode in ('greedy','sample','density','density_both'):
            value[mode]=svg_pages(generated(c[mode]))
            value[f'{mode}_summary']={k:c[mode][k] for k in ('stop_reason','generated_strokes','generated_pages','outside_page_strokes')}
        for mode in ('teacher_forced','teacher_density','teacher_density_both'):
            value[mode]=svg_pages(c[mode]) if mode in c else None
        value['teacher_metrics']=c.get('teacher_modes')
        text_ids={r['page_id'] for r in c['reference']['strokes'] if not r['targetable'] and r['kind']=='text'}
        value['condition_text']=' '.join(v for k,v in c['reference']['page_texts'].items() if int(k) in text_ids)
        if c['task']=='T1': value['condition_text']=c['source']['text'].split(' ')[0]
        data.append(value)
    encoded=json.dumps(data,ensure_ascii=False).replace('</',r'<\/')
    meta=read(args.review/'metadata.json')
    header=f'同一R step {meta["step"]} · 原valid条件20例 · 128轮多起点数值众数 · 原样回馈/控制符RMS 0.018'
    document=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全GMM众数解码对照</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f3f6f8;color:#203044}header{background:white;padding:22px 28px;border-bottom:1px solid #ddd}h1{font-size:24px;margin:0 0 10px}main{padding:20px;max-width:1800px;margin:auto}select{font:inherit;padding:8px;margin:0 12px 12px 0}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.teacher{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.panel{background:white;padding:14px;border:1px solid #dae1e7;border-radius:9px}h3{font-size:16px;margin:0 0 9px}.pageid{font-size:12px;color:#627488;margin:8px 0}.page svg{width:100%;border:1px solid #ddd;box-sizing:border-box}.page{margin-bottom:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;padding:12px;background:#eaf0f5}#source{white-space:pre-wrap;line-height:1.7;margin-bottom:12px}#status{font-size:13px;line-height:1.7;margin:12px 0}details{margin-top:18px}@media(max-width:1000px){.grid,.teacher{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.grid,.teacher{grid-template-columns:1fr}}</style>
<header><h1>全 GMM 众数解码：同条件对照</h1><div>HEADER</div><p>R众数：最大化整组32分量的48维λ密度。R+D众数：D的每个τ也取整组20分量的二维密度峰。</p></header>
<main><select id="task"></select><select id="example"></select><div id="source"></div><div id="status"></div>
<div class="grid"><section class="panel"><h3>源回答</h3><div id="reference"></div></section><section class="panel"><h3>原最高π生成</h3><div id="greedy"></div></section><section class="panel"><h3>R 全GMM众数</h3><div id="density"></div></section><section class="panel"><h3>R+D 全GMM众数</h3><div id="density_both"></div></section></div>
<details><summary>实际输入条件和原随机采样</summary><div class="teacher"><section class="panel"><h3>实际条件</h3><div id="condition"></div></section><section class="panel"><h3>原采样温度0.5</h3><div id="sample"></div></section></div></details>
<details id="tf"><summary>真实前文下预测下一笔</summary><p>各步使用源前文；源目标仅用于误差计算，未用于挑选分量。</p><div class="teacher"><section class="panel"><h3>原最高π</h3><div id="teacher_forced"></div></section><section class="panel"><h3>R众数</h3><div id="teacher_density"></div></section><section class="panel"><h3>R+D众数</h3><div id="teacher_density_both"></div></section></div></details>
<details><summary>数值核对与元信息</summary><pre id="metrics"></pre></details></main><script>
const cases=DATA;const byId=id=>document.getElementById(id),task=byId('task'),example=byId('example');
for(const name of ['T1','O1','O2','S1','S2','S3','S4','S5']){const o=document.createElement('option');o.value=o.textContent=name;task.appendChild(o)}
function options(){example.replaceChildren();for(const c of cases.filter(c=>c.task===task.value)){const o=document.createElement('option');o.value=c.name;o.textContent=c.label;example.appendChild(o)}render()}
function render(){const c=cases.find(c=>c.name===example.value);for(const id of ['condition','reference','greedy','sample','density','density_both','teacher_forced','teacher_density','teacher_density_both'])byId(id).innerHTML=c[id]||'';byId('tf').style.display=c.teacher_density?'block':'none';byId('source').textContent='条件文字：'+c.condition_text+(c.source.answer!==undefined?'\n源回答：'+JSON.stringify(c.source.answer):c.source.label?'\n类别：'+c.source.label:'');byId('status').textContent=['greedy','density','density_both'].map(m=>m+'：'+JSON.stringify(c[m+'_summary'])).join('　');byId('metrics').textContent=JSON.stringify({first_distribution:c.first_distribution,teacher_metrics:c.teacher_metrics,generation:c.generation,source:c.source},null,2)}
task.addEventListener('change',options);example.addEventListener('change',render);task.value='O1';options();
</script></html>'''.replace('HEADER',html.escape(header)).replace('DATA',encoded)
    (args.review/'index.html').write_text(document,encoding='utf-8')
    result={**read(args.review/'completion.json'), 'page':'index.html', 'montages':montages}
    (args.review/'rendered.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__': main()
