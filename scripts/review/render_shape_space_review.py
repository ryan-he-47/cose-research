"""2026-10-14：形状分类/逐笔空格读出曲线与旧版同前文完整分页。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import argparse
import html
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.review.render_formal_r_review import svg_pages
from scripts.review.render_r_dual_review import plot_page


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def render(root,baseline):
    history=read(root/'training_history.json')
    current=read(root/'classification.json');old=read(root/'baseline_classification.json')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_shape_ce'] for r in history],label='train shape')
    axes[0].plot(epochs,[r['valid']['all']['cross_entropy'] for r in history],label='valid shape')
    axes[1].plot(epochs,[r['train_space_bce'] for r in history],label='train every stroke')
    axes[1].plot(epochs,[r['valid']['space_all']['cross_entropy'] for r in history],label='valid every stroke')
    axes[1].plot(epochs,[r['valid']['space_first_stroke']['cross_entropy'] for r in history],label='valid first stroke')
    for key in ['text','first_stroke','continuation','control']:
        axes[2].plot(epochs,[r['valid'][key]['shape_error'] for r in history],label=key)
    for axis,ylabel in zip(axes,['Shape CE / nats','Space BCE / nats','Shape class error']):
        axis.set(xlabel='Epoch',ylabel=ylabel);axis.legend()
    fig.tight_layout();fig.savefig(root/'head_curves.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(8,14),squeeze=False)
    cases=[];totals=[]
    old_root=baseline.parent/'r_text_readout_constrained_20261013'
    for index in range(4):
        values=dict(new=read(root/f'prompt{index}.json'),old=read(old_root/f'prompt{index}.json'))
        panels={k:svg_pages([r for r in v['strokes'] if r['generated']]) for k,v in values.items()}
        panels['condition']=svg_pages([r for r in values['new']['strokes'] if not r['generated']])
        for col,key in enumerate(['old','new']):
            plot_page(axes[index,col],values[key]['strokes']);axes[index,col].set_title(f'prompt{index} / {key}')
        brief={k:{name:value for name,value in v.items() if name not in ('strokes','constraint_trace')}
               for k,v in values.items()}
        cases.append(dict(index=index,prompt=values['new']['prompt'],panels=panels,summary=brief))
        totals.append(brief['new']['constraint_summary'])
    fig.tight_layout();fig.savefig(root/'generation.png',dpi=150);plt.close(fig)
    total={k:sum(r[k] for r in totals) for k in ['steps','forced_steps','raw_illegal_steps',
        'placement_ties','placement_gmm_scored','space_suppressed_steps','identity_revisions']}
    total['font_violating_steps']=sum(c['summary']['new']['font_audit']['violating_steps'] for c in cases)
    (root/'totals.json').write_text(json.dumps(total,indent=2)+'\n',encoding='utf-8')
    rows=[]
    for key in ['all','text','first_stroke','continuation','control','space_all','space_first_stroke','space_continuation','space_control']:
        rate='error' if key.startswith('space_') else 'shape_error'
        values=[key,current[key]['count'],f"{old[key]['cross_entropy']:.4f}",f"{current[key]['cross_entropy']:.4f}",
                f"{old[key][rate]:.2%}",f"{current[key][rate]:.2%}"]
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    data=json.dumps(cases,ensure_ascii=False).replace('</',r'<\/')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>笔画形状分类＋逐笔空格二分类</title><style>body{font:16px system-ui,"Microsoft YaHei";color:#243442;background:#f3f5f7;max-width:1500px;margin:20px auto;padding:20px}td,th{border:1px solid #ccd;padding:8px}table{border-collapse:collapse}pre{white-space:pre-wrap;overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.panel{padding:12px;background:white;border:1px solid #ddd}svg{width:100%}select{font:inherit;padding:10px}@media(max-width:950px){.grid{grid-template-columns:1fr}}</style>
<h1>冻结R：笔画形状分类＋逐笔空格二分类</h1><p>113种有向笔画形状＋3个控制符；空格每一步独立输出、每一步BCE监督。头只读取768维h，单个256宽隐藏层；原E/D/R冻结。</p>
<p>形状只去平移，保留方向、尺度和旋转。旧245类概率先合并到同一116类，表中CE可以按同一标签空间比较；旧空格栏是概率边缘化后的二分类。</p>
<p><a href="head_curves.png">训练曲线</a> · <a href="generation.png">首个生成页概览</a> · <a href="geometry_comparison.json">同4096位置形状与GMM对比</a> · <a href="completion.json">冻结来源与缓存</a></p>
<table><tr><th>分桶</th><th>数量</th><th>旧CE/BCE</th><th>新CE/BCE</th><th>旧错误</th><th>新错误</th></tr>ROWS</table>
<p>推理保持字体前缀与控制符策略。多个落点共享已胜出的形状时，冻结R原GMM仅给标准候选的E嵌入打分，选落点；不经过D解码，不改形状或空格决策。后续笔/控制符上的非法空格会被策略压掉，次数单独记录。</p>
<select id="case"></select><p id="prompt"></p><div class="grid"><section class="panel"><h3>形状＋空格新头</h3><div id="new"></div></section><section class="panel"><h3>旧联合类别头＋前缀策略</h3><div id="old"></div></section><section class="panel"><h3>新头实际前文</h3><div id="condition"></div></section></div><details open><summary>完整文字与推理记录</summary><pre id="metrics"></pre></details>
<script>const data=DATA;const el=id=>document.getElementById(id);data.forEach(c=>{let o=document.createElement('option');o.value=c.index;o.textContent='prompt'+c.index;el('case').appendChild(o)});function render(){let c=data[+el('case').value];el('prompt').textContent=c.prompt;Object.keys(c.panels).forEach(k=>el(k).innerHTML=c.panels[k]);el('metrics').textContent=JSON.stringify(c.summary,null,2)}el('case').onchange=render;render()</script></html>'''
    (root/'index.html').write_text(page.replace('ROWS',''.join(rows)).replace('DATA',data),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=Path('tmp_smoke/r_text_readout_20261013'))
    args=parser.parse_args();render(args.root,args.baseline)
