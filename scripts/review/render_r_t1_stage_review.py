"""2026-10-11：T1长训阶段曲线、同条件历史样张与完整分页查看页。"""

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
import numpy as np

from scripts.review.render_r_dual_review import plot_page, records, read
from scripts.review.render_formal_r_review import svg_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--earlier_log', type=Path, action='append', default=[], help='2026-10-12：按恢复顺序拼接之前的日志')
    parser.add_argument('--recent_start_million', type=float, default=650, help='局部曲线放大的起始成功输入（百万笔）')
    args = parser.parse_args()
    root, current, previous = args.root, args.root/'t1', args.previous
    checks, previous_checks = read(current/'checks.json'), read(previous/'checks.json')
    current_m, previous_m = checks['successful_inputs']/1e6, previous_checks['successful_inputs']/1e6
    current_label, previous_label = f'{current_m:.1f}M', f'{previous_m:.1f}M'
    anneal = checks['train_args'].get('anneal_transition')
    anneal_m = anneal['start_inputs']/1e6 if anneal else None
    merged, retries = {}, []
    for path in args.earlier_log + [args.log]:
        log_rows, start_step = [], None
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.startswith('{'): continue
            try: row = json.loads(line)
            except json.JSONDecodeError: continue
            if row.get('formal_start'): start_step = row['step']
            # 完整保存边界可能额外打印非诊断步，其gmm_buckets为空。
            if row.get('success') and 'text' in row.get('gmm_buckets', {}): log_rows.append(row)
            if row.get('capacity_oom_retry'): retries.append(row)
        # 后续日志从完整检查点恢复；之前未保存的更新不重复画进真实轨迹。
        if start_step is not None: merged = {k:r for k,r in merged.items() if k <= start_step}
        merged.update({r['step']:r for r in log_rows})
    rows = sorted(merged.values(), key=lambda r:r['step'])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = [r['successful_inputs']/1e6 for r in rows]
    plots = [('content_nll', 'Text NLL'), ('posterior_mu_mse', 'Posterior mean latent MSE'),
             ('top_pi_mu_mse', 'Top-pi mean latent MSE'), ('posterior_sigma', 'Posterior sigma'),
             ('top_pi_matches_posterior', 'Top-pi / posterior component agreement'), ('lr', 'Actual LR')]
    for axis, (key, title) in zip(axes.flat, plots):
        y = [r[key] if key in r else r['gmm_buckets']['text'][key] for r in rows]
        axis.plot(x, y, alpha=.25, linewidth=.5)
        width = min(20, len(y))
        smoothed = np.convolve(y, np.ones(width)/width, mode='valid')
        axis.plot(x[width-1:], smoothed, linewidth=1.2)
        axis.axvline(previous_m, color='#8192a5', linestyle='--', linewidth=.8)
        axis.axvline(current_m, color='#b64268', linestyle='--', linewidth=.8)
        if anneal_m is not None:
            axis.axvline(anneal_m, color='#329575', linestyle=':', linewidth=1, label='Anneal starts')
        axis.set(title=title, xlabel='Successful input strokes / million')
        axis.grid(alpha=.2)
    if anneal_m is not None:
        axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/'training_curves.png', dpi=160)
    plt.close(fig)
    # 2026-10-12：平台期单独放大，避免早期大幅下降压扁最近的波动。
    recent = [r for r in rows if r['successful_inputs'] >= args.recent_start_million*1e6]
    windows = []
    # 2026-10-13：收尾曲线继续按50M窗口汇总，不把统计截断在14k阶段的1B处。
    last_m = rows[-1]['successful_inputs']/1e6
    intervals = [(480,530),(600,650)] + [(lo,lo+50) for lo in range(700,int(last_m)+1,50)]
    for lo, hi in intervals:
        selected = [r for r in rows if lo*1e6 <= r['successful_inputs'] < hi*1e6]
        if not selected: continue
        weights = np.array([r['gmm_buckets']['text']['count'] for r in selected])
        fields = ('posterior_mu_mse','top_pi_mu_mse','top_pi_matches_posterior','posterior_sigma','pi_entropy')
        value = {k:float(np.average([r['gmm_buckets']['text'][k] for r in selected],weights=weights)) for k in fields}
        value.update(start_million=lo,end_million=min(hi,rows[-1]['successful_inputs']/1e6),logged_batches=len(selected),
            loss_mean=float(np.mean([r['loss'] for r in selected])),
            text_nll=float(np.average([r['content_nll'] for r in selected],weights=weights)),
            text_nll_median=float(np.median([r['content_nll'] for r in selected])))
        windows.append(value)
    (root/'training_windows.json').write_text(json.dumps(windows,indent=2),encoding='utf-8')
    if recent:
        fig, axes = plt.subplots(2,3,figsize=(15,8))
        xx = [r['successful_inputs']/1e6 for r in recent]
        for axis, (key,title) in zip(axes.flat,[('loss','Actual weighted loss'),*plots[:5]]):
            yy = [r[key] if key in r else r['gmm_buckets']['text'][key] for r in recent]
            width = min(20,len(yy))
            axis.plot(xx,yy,alpha=.25,linewidth=.5)
            axis.plot(xx[width-1:],np.convolve(yy,np.ones(width)/width,mode='valid'),linewidth=1.2)
            axis.set(title=title,xlabel='Successful input strokes / million')
            axis.axvline(current_m,color='#b64268',linestyle='--',linewidth=.8)
            axis.grid(alpha=.2)
        fig.tight_layout();fig.savefig(root/'recent_curves.png',dpi=160);plt.close(fig)
    cases = read(current/'summary.json')
    # 展示每例第一张实际生成页，完整生成保留在HTML中。
    for subset, filename in [(cases[:4], 'generation.png'), (cases[4:], 'manual_generation.png')]:
        if not subset: continue
        fig, axes = plt.subplots(len(subset), 3, figsize=(11, 3.5*len(subset)), squeeze=False)
        for row, item in enumerate(subset):
            for col, (directory, file, label) in enumerate([
                (previous, 'greedy', previous_label+' greedy'), (current, 'greedy', current_label+' greedy'),
                (current, 'sample', current_label+' sample 0.5')]):
                plot_page(axes[row, col], records(directory, item['name'], file))
                axes[row, col].set_title(item['name']+' / '+label, fontsize=9)
        fig.tight_layout();fig.savefig(root/filename, dpi=170);plt.close(fig)
    selected = [c for c in cases if 'teacher_metrics' in c]
    fig, axes = plt.subplots(len(selected), 4, figsize=(14, 7), squeeze=False)
    for row, item in enumerate(selected):
        for col, (directory, file, label) in enumerate([
            (current, 'reference', 'Source'), (current, 'ae_reference', 'Frozen AE'),
            (previous, 'teacher_forced', previous_label+' true history'),
            (current, 'teacher_forced', current_label+' true history')]):
            plot_page(axes[row, col], records(directory, item['name'], file))
            axes[row, col].set_title(item['name']+' / '+label, fontsize=9)
    fig.tight_layout()
    fig.savefig(root/'teacher.png', dpi=180)
    plt.close(fig)
    comparisons = {}
    for name, directory in [('previous', previous), ('current', current)]:
        mixture = read(directory/'mixture_summary.json')['text']
        comparisons[name] = {key:mixture[key]['mean'] if isinstance(mixture[key], dict) else mixture[key]
            for key in ['count', 'nll', 'top_pi_mse', 'posterior_mse', 'posterior_sigma', 'top_pi_matches_posterior']}
        comparisons[name]['language_pairs'] = read(directory/'language_pairs.json')
        comparisons[name]['checkpoint'] = {k:read(directory/'checks.json')[k] for k in ['step','successful_inputs']}
    comparisons['checkpoint'] = {k:checks[k] for k in ['step', 'successful_inputs', 'final_lr', 'review_stage', 'frozen_ae_equal']}
    comparisons['runtime_oom_retries'] = retries
    comparisons['anneal_transition'] = anneal
    comparisons['last_training_log'] = {k:rows[-1][k] for k in ['step', 'successful_inputs', 'content_nll', 'lr']}
    (root/'comparison.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding='utf-8')
    data = []
    for item in cases:
        name = item['name']
        generated = records(current, name, 'greedy')
        panels = {'condition':svg_pages([r for r in generated if not r.get('generated')])}
        for key, directory, file in [
            ('previous', previous, 'greedy'), ('greedy', current, 'greedy'),
            ('sample', current, 'sample'), ('teacher', current, 'teacher_forced'),
            ('ae', current, 'ae_reference'), ('latent', current, 'latent')]:
            panels[key] = svg_pages([r for r in records(directory, name, file) if r.get('generated')])
        panels['reference'] = svg_pages([r for r in records(current, name, 'reference') if r.get('targetable')])
        data.append({**item, 'panels':panels})
    encoded = json.dumps(data, ensure_ascii=False).replace('</', r'<\/')
    pair_rows = ''.join('<tr><td>'+html.escape(p['name'])+'</td><td>'+f"{p['context_swap_preference_shift']:+.6f}"+'</td></tr>'
        for p in comparisons['current']['language_pairs'])
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>T1长训阶段查看</title><style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f7;color:#253544;margin:0}header,main{padding:22px;max-width:1600px;margin:auto}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.panel{background:white;padding:12px;border:1px solid #dae1e8;border-radius:8px}
.page svg{width:100%;border:1px solid #e3e6ea}.pageid{color:#677;font-size:12px}h2{font-size:17px}select{font:inherit;padding:8px}
pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd}a{color:#176b98}
@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<header><h1>T1长训：CURRENT_STAGE</h1><p>冻结55k FiLM AE，字形增强，σ下界0.003方差平方和。本机FP16只读推理。
上一阶段：PREVIOUS_STAGE；本页复用相同前文和参考点列。</p>
<p>灰：实际前文；蓝：生成；粉：识别到的控制符。完整分页可向下查看。</p>
<a href="training_curves.png">完整训练曲线</a> · <a href="recent_curves.png">局部曲线放大</a> · <a href="generation.png">故事前文续写</a> · <a href="manual_generation.png">短前文续写</a> · <a href="teacher.png">真实前文预测</a> · <a href="comparison.json">原始对比指标</a>
<p>上下文交换的相对候选偏好变化（正值朝示例预想方向）：</p><table><tr><th>例子</th><th>变化</th></tr>PAIR_ROWS</table></header>
<main><select id="case"></select><p id="prompt"></p><pre id="stops"></pre><div class="grid">
<section class="panel"><h2>条件</h2><div id="condition"></div></section><section class="panel"><h2>原文参考</h2><div id="reference"></div></section>
<section class="panel"><h2>Greedy</h2><div id="greedy"></div></section><section class="panel"><h2>Sample 0.5</h2><div id="sample"></div></section></div>
<details><summary>历史生成、真实前文预测和AE参考</summary><div class="grid">
<section class="panel"><h2>上一阶段 greedy</h2><div id="previous"></div></section><section class="panel"><h2>真实前文下下一笔</h2><div id="teacher"></div></section>
<section class="panel"><h2>AE重建</h2><div id="ae"></div></section><section class="panel"><h2>直接λ回馈</h2><div id="latent"></div></section></div></details><pre id="metrics"></pre></main>
<script>const data=DATA;const el=id=>document.getElementById(id);for(const c of data){const o=document.createElement('option');o.value=c.name;o.textContent=c.name;el('case').appendChild(o)}
function render(){const c=data.find(c=>c.name===el('case').value);el('prompt').textContent='前文：'+c.prompt;el('stops').textContent=JSON.stringify(c.generation,null,2);
for(const id of Object.keys(c.panels))el(id).innerHTML=c.panels[id]||'<p>本例未生成此项</p>';el('metrics').textContent=JSON.stringify(c.teacher_metrics||{},null,2)}el('case').onchange=render;render();</script></html>'''
    page = page.replace('CURRENT_STAGE', f"{checks['step']}步 / {current_m/100:.3f}亿输入")
    page = page.replace('PREVIOUS_STAGE', f"{previous_checks['step']}步 / {previous_m/100:.3f}亿输入")
    if checks['review_stage'] == 'final':
        page = page.replace('T1长训阶段查看','T1长训收尾验收').replace('T1长训：','T1长训收尾：')
    (root/'index.html').write_text(page.replace('PAIR_ROWS', pair_rows).replace('DATA', encoded), encoding='utf-8')
    print(json.dumps({'page':str(root/'index.html'), 'cases':len(cases), 'runtime_oom_retries':len(retries)}))


if __name__ == '__main__':
    main()
