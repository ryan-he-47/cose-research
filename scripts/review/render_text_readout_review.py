"""2026-10-13：冻结R文字读出的CE/几何类别对比及同前文闭环分页。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import argparse
import html
import json
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.review.render_formal_r_review import svg_pages
from scripts.review.render_r_dual_review import plot_page


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paired_analysis(root):
    """只汇总已落盘的同位置预测；不再次运行模型或采样。"""
    vocabulary=read(root/'vocabulary.json')
    steps=read(root/'geometry_steps.json')
    report={}
    for group,predicate in [('text',lambda t:not t['control']),
                             ('first_stroke',lambda t:not t['control'] and t['part']==0),
                             ('continuation',lambda t:not t['control'] and t['part']>0),
                             ('control',lambda t:bool(t['control']))]:
        rows=[r for r in steps if predicate(vocabulary[r['gold']])]
        if not rows:continue
        outcome=dict(both_correct=0,head_only_correct=0,gmm_only_correct=0,both_wrong=0)
        for r in rows:
            h=r['gold'] in r['head_aliases'];g=r['gold'] in r['greedy']['aliases']
            outcome['both_correct' if h and g else 'head_only_correct' if h else
                    'gmm_only_correct' if g else 'both_wrong']+=1
        values={}
        for method in ['head','source','ae','greedy','sample']:
            predictions=[]
            for r in rows:
                candidates=([dict(prediction=r['head'],aliases=r['head_aliases'],**r['head_geometry'])]
                    if method=='head' else r['sample'] if method=='sample' else [r[method]])
                predictions.extend(dict(gold=r['gold'],**value) for value in candidates)
            values[method]={key:mean(p[key] for p in predictions) for key in
                ['target_canvas_mse','target_position_mse','target_shape_mse']}
            identity=lambda index:tuple(vocabulary[index][k] for k in ['character','part','control'])
            values[method].update(steps=len(rows),
                joint_error=mean(p['prediction']!=p['gold'] for p in predictions),
                identity_error=mean(identity(p['prediction'])!=identity(p['gold']) for p in predictions),
                same_shape_error=mean(p['gold'] not in p['aliases'] for p in predictions))
            values[method]['false_control_rate']=(mean(bool(vocabulary[p['prediction']]['control']) for p in predictions)
                if group!='control' else None)
        report[group]=dict(steps=len(rows),paired_same_shape_outcomes=outcome,methods=values,
            source_ambiguous_steps=sum(len(r['source']['aliases'])>1 for r in rows))
    (root/'paired_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=Path('tmp_smoke/r_t1_final_review_20261013/t1'))
    args=parser.parse_args()
    root=args.root
    paired=paired_analysis(root)
    history,classification,geometry=map(read,[root/'training_history.json',root/'classification.json',root/'geometry_comparison.json'])
    if 'continuation' not in geometry['groups']:
        geometry['groups']['continuation']=paired['continuation']['methods']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    epochs=[r['epoch'] for r in history]
    for label,values in [('train',[r['train_ce'] for r in history]),
                         ('valid text',[r['valid']['text']['cross_entropy'] for r in history]),
                         ('valid first stroke',[r['valid']['first_stroke']['cross_entropy'] for r in history])]:
        axes[0].plot(epochs,values,label=label)
    axes[0].set(xlabel='Epoch',ylabel='Cross entropy / nats');axes[0].legend()
    for key in ['text','first_stroke','control']:
        if key in history[0]['valid']:
            axes[1].plot(epochs,[r['valid'][key]['identity_error'] for r in history],label=key)
    axes[1].set(xlabel='Epoch',ylabel='Stroke identity error');axes[1].legend()
    fig.tight_layout();fig.savefig(root/'head_curves.png',dpi=150);plt.close(fig)
    cases=[]
    fig,axes=plt.subplots(4,3,figsize=(11,14),squeeze=False)
    for index in range(4):
        values={'head':read(root/f'prompt{index}.json')}
        values.update({key:read(args.baseline/f'T1_prompt{index}'/(key+'.json')) for key in ['greedy','sample']})
        panels={key:svg_pages([r for r in value['strokes'] if r.get('generated')]) for key,value in values.items()}
        panels['condition']=svg_pages([r for r in values['head']['strokes'] if not r.get('generated')])
        cases.append(dict(index=index,prompt=values['head']['prompt'],panels=panels,
            readout={k:v for k,v in values['head'].items() if k!='strokes'},
            baseline_stops={key:values[key]['stop_reason'] for key in ['greedy','sample']}))
        for col,key in enumerate(['greedy','head','sample']):
            plot_page(axes[index,col],values[key]['strokes']);axes[index,col].set_title(f'prompt{index} / {key}')
    fig.tight_layout();fig.savefig(root/'generation.png',dpi=150);plt.close(fig)
    rows=''
    for group,values in geometry['groups'].items():
        for method,metrics in values.items():
            rows+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
                [group,method,metrics['steps'],f"{metrics['identity_error']:.2%}",f"{metrics['joint_error']:.2%}",f"{metrics['same_shape_error']:.2%}"] )+'</tr>'
    encoded=json.dumps(cases,ensure_ascii=False).replace('</',r'<\/')
    summary=json.dumps(classification,ensure_ascii=False,indent=2)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>冻结R文字笔画读出</title><style>body{font:16px system-ui,"Microsoft YaHei";color:#243442;background:#f3f5f7;margin:20px auto;max-width:1550px;padding:20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd}pre{white-space:pre-wrap;overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.panel{background:white;padding:12px;border:1px solid #ddd}svg{width:100%}select{padding:10px;font:inherit}@media(max-width:950px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<h1>冻结R：文字笔画分类读出</h1><p>正文笔画、字符内子路径、前导空格与控制符。头只读取R末层h，标准字体回馈E。</p>
<p><a href="head_curves.png">分类头曲线</a> · <a href="generation.png">同前文生成</a> · <a href="geometry_comparison.json">单步原始指标</a> · <a href="paired_analysis.json">同位置得失与形状/位置误差</a></p>
<h2>相同真实前文：分类与解码曲线的类别误差</h2><p>source/ae用于查看模板匹配和重建歧义，greedy/sample使用原GMM及D；sample取4次温度0.5采样的平均错误。几何匹配使用已知版面条件，同形集合误差允许相同几何候选。</p>
<table><tr><th>分桶</th><th>方法</th><th>位置数</th><th>笔画身份错误</th><th>联合标签错误</th><th>同形集合错误</th></tr>ROWS</table>
<details><summary>分类头完整CE与分桶数值</summary><pre>SUMMARY</pre></details>
<h2>闭环续写</h2><p>本次符号渲染使用已有引擎处理当前前缀。字追加引起的排版回流会刷新上下文，次数保存在每例记录中。文本记录来自类别轨迹；若字符内跳笔，画布只显示实际预测的子笔画，不补齐整个字。</p>
<select id="case"></select><p id="prompt"></p><div class="grid"><section class="panel"><h3>分类头</h3><div id="head"></div></section><section class="panel"><h3>原GMM greedy</h3><div id="greedy"></div></section><section class="panel"><h3>原GMM sample0.5</h3><div id="sample"></div></section><section class="panel"><h3>分类头实际前文</h3><div id="condition"></div></section></div><pre id="metrics"></pre>
<script>const data=DATA;const el=id=>document.getElementById(id);data.forEach(c=>{let o=document.createElement('option');o.value=c.index;o.textContent='prompt'+c.index;el('case').appendChild(o)});function render(){let c=data[+el('case').value];el('prompt').textContent=c.prompt;Object.keys(c.panels).forEach(k=>el(k).innerHTML=c.panels[k]);el('metrics').textContent=JSON.stringify(c.readout,null,2)}el('case').onchange=render;render()</script></html>'''
    page=page.replace('取4次',f"取{geometry['sample_draws']}次")
    page=page.replace('ROWS',rows).replace('SUMMARY',html.escape(summary)).replace('DATA',encoded)
    (root/'index.html').write_text(page,encoding='utf-8')
    print(str(root/'index.html'))


if __name__=='__main__':main()
