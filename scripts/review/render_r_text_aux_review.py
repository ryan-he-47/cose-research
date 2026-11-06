"""2026-10-14：联合R收尾曲线、密度分解与前后完整分页。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import html
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.review.render_formal_r_review import svg_pages
from scripts.review.render_r_dual_review import plot_page


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def brief(value):
    return {k: v for k, v in value.items() if k not in ('strokes', 'constraint_trace')}


def render(root, before_readout, before_r):
    log = []
    for line in (root/'train.log').read_text(encoding='utf-8').splitlines():
        try: row = json.loads(line)
        except ValueError: continue
        if row.get('success'): log.append(row)
    valid = [read(root/'validation_initial.json')]+[json.loads(line) for line in (root/'validation.jsonl').read_text().splitlines()]
    alpha = read(root/'review_checks.json')['cfg']['r_text_aux_weight']
    x = np.array([r['successful_inputs']/1e6 for r in valid])
    tx = np.array([r['successful_inputs']/1e6 for r in log])
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes[0, 0].plot(tx, [r['content_nll'] for r in log], alpha=.35, label='train / augmented')
    axes[0, 0].plot(x, [r['content_nll'] for r in valid], 'o-', label='valid / standard font')
    axes[0, 0].set_ylabel('Text GMM NLL / nats')
    for bucket in ['all', 'first', 'continuation', 'control']:
        axes[0, 1].plot(x, [r['text_aux'][bucket]['shape_error']*100 for r in valid], 'o-', label=bucket)
        axes[0, 2].plot(x, [r['text_aux'][bucket]['space_error']*100 for r in valid], 'o-', label=bucket)
    axes[0, 1].set_ylabel('Shape error / %'); axes[0, 2].set_ylabel('Every-stroke space error / %')
    geometry = [.35*r['content_nll']+.05*r['control_nll'] for r in valid]
    auxiliary = [alpha*(r['text_aux']['all']['shape_ce']+r['text_aux']['all']['space_bce']) for r in valid]
    axes[1, 0].plot(x, np.array(geometry)-geometry[0], 'o-', label='weighted GMM change')
    axes[1, 0].plot(x, np.array(auxiliary)-auxiliary[0], 'o-', label='weighted CE+BCE change')
    axes[1, 0].plot(x, np.array(geometry)+auxiliary-geometry[0]-auxiliary[0], 'o-', label='joint change')
    axes[1, 0].axhline(0, color='#888', linewidth=.7); axes[1, 0].set_ylabel('Loss change from initial')
    axes[1, 1].plot(tx, [r['lr'] for r in log]); axes[1, 1].set_ylabel('Learning rate')
    axes[1, 2].plot(tx, [r['gmm_buckets']['text']['posterior_sigma'] for r in log]); axes[1, 2].set_ylabel('Text posterior mean sigma')
    for axis in axes.flat:
        axis.set_xlabel('New successful inputs / million'); axis.grid(alpha=.15)
        axis.axvspan(0, 8, color='#9abbd0', alpha=.12); axis.axvspan(180, 200, color='#86b9a6', alpha=.12)
        if axis.get_legend_handles_labels()[0]: axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(root/'training_curves.png', dpi=150); plt.close(fig)

    density = read(root/'density_comparison.json')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    groups = ['text', 'first', 'continuation', 'control']; offsets = np.arange(4)
    for i, key in enumerate(['volume', 'residual', 'mixture_kl']):
        delta = [density['after'][g][key]-density['before'][g][key] for g in groups]
        axes[0].bar(offsets+(i-1)*.24, delta, width=.24, label=key)
    axes[0].set_xticks(offsets, groups); axes[0].axhline(0, color='#999'); axes[0].set_ylabel('NLL decomposition change / nats'); axes[0].legend()
    for key in ['posterior_mse', 'top_pi_mse', 'posterior_sigma']:
        delta = [100*(density['after'][g][key]/density['before'][g][key]-1) for g in groups]
        axes[1].plot(offsets, delta, 'o-', label=key)
    axes[1].set_xticks(offsets, groups); axes[1].axhline(0, color='#999'); axes[1].set_ylabel('Relative change / %'); axes[1].legend()
    fig.tight_layout(); fig.savefig(root/'density_decomposition.png', dpi=150); plt.close(fig)

    cases = []
    fig, axes = plt.subplots(4, 2, figsize=(8, 14), squeeze=False)
    for i in range(4):
        old, new = read(before_readout/f'prompt{i}.json'), read(root/f'prompt{i}.json')
        assert old['prompt']==new['prompt']
        for column, value in enumerate([old, new]):
            plot_page(axes[i, column], value['strokes']); axes[i, column].set_title(f'prompt{i} / '+('before' if column==0 else 'after'))
        cases.append(dict(prompt=new['prompt'], old=brief(old), new=brief(new),
            panels=dict(old=svg_pages([r for r in old['strokes'] if r['generated']]),
                        new=svg_pages([r for r in new['strokes'] if r['generated']]),
                        condition=svg_pages([r for r in new['strokes'] if not r['generated']]))))
    fig.tight_layout(); fig.savefig(root/'readout_generation.png', dpi=150); plt.close(fig)
    gmm = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
    for i in range(2):
        values = dict(old=read(before_r/'t1'/f'T1_prompt{i}'/'greedy.json'),
                      new=read(root/f'gmm_prompt{i}'/'greedy.json'), sample=read(root/f'gmm_prompt{i}'/'sample.json'))
        for column, key in enumerate(values):
            plot_page(axes[i, column], values[key]['strokes']); axes[i, column].set_title(f'prompt{i} / {key}')
        # 原样张和当前生成使用同一输入点列。
        condition = lambda v: [(r['page_id'], r['kind'], r['points']) for r in v['strokes'] if not r['generated']]
        assert condition(values['old'])==condition(values['new'])
        gmm.append(dict(prompt=cases[i]['prompt'], summary={k: brief(v) for k, v in values.items()},
                        panels={k: svg_pages([r for r in v['strokes'] if r['generated']]) for k, v in values.items()}))
    fig.tight_layout(); fig.savefig(root/'gmm_generation.png', dpi=150); plt.close(fig)

    old, new = valid[0], valid[-1]
    metric_rows = [('正文GMM NLL', old['content_nll'], new['content_nll']), ('控制符GMM NLL', old['control_nll'], new['control_nll'])]
    for name, bucket, field in [('全部形状CE', 'all', 'shape_ce'), ('全部形状错误率', 'all', 'shape_error'),
                               ('首笔形状错误率', 'first', 'shape_error'), ('续接笔形状错误率', 'continuation', 'shape_error'),
                               ('逐笔空格BCE', 'all', 'space_bce'), ('逐笔空格错误率', 'all', 'space_error'),
                               ('首笔空格错误率', 'first', 'space_error')]:
        a, b = old['text_aux'][bucket][field], new['text_aux'][bucket][field]
        metric_rows.append((name, f'{a:.2%}' if 'error' in field else f'{a:.4f}', f'{b:.2%}' if 'error' in field else f'{b:.4f}'))
    table = ''.join('<tr>'+''.join('<td>'+html.escape(f'{v:.4f}' if isinstance(v, float) else str(v))+'</td>' for v in row)+'</tr>' for row in metric_rows)
    text_delta = {k: density['after']['text'][k]-density['before']['text'][k] for k in ['nll', 'volume', 'residual', 'mixture_kl']}
    totals = {key: sum(c['new']['constraint_summary'][key] for c in cases) for key in ['steps', 'forced_steps', 'raw_illegal_steps', 'shape_mask_changes', 'placement_ties', 'space_suppressed_steps']}
    totals['font_violating_steps'] = sum(c['new']['font_audit']['violating_steps'] for c in cases)
    totals['stop_reasons'] = [c['new']['stop_reason'] for c in cases]
    write = lambda p, v: p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    write(root/'generation_summary.json', dict(totals=totals, readout=[dict(prompt=c['prompt'], before=c['old'], after=c['new']) for c in cases], GMM=[c['summary'] for c in gmm]))
    data = json.dumps(dict(readout=cases, gmm=gmm), ensure_ascii=False).replace('</', r'<\/')
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>R形状CE＋逐笔空格BCE：收尾验收</title><style>
body{font:16px system-ui,"Microsoft YaHei";color:#243442;background:#f4f6f8;max-width:1500px;margin:20px auto;padding:20px}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccd;padding:8px}img{width:100%;height:auto}select{font:inherit;padding:8px;max-width:100%}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.panel{background:white;padding:12px;border:1px solid #ddd}.page{margin-bottom:14px}svg{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:18px 0}@media(max-width:950px){.grid{grid-template-columns:1fr}}</style>
<h1>R形状CE＋逐笔空格BCE：收尾验收</h1>
<p>630704已完成：3032次新增更新，200,000,280成功输入；累计29235步，末LR=1e-5。冻结AE55k，R和辅助头联合训练；α=5.87618。全部验收在本机FP16推理，0次训练更新。</p>
<p>同128篇标准字体验证：正文NLL回退2.207，但形状/空格分类改善。回退集中在字符续接笔，主要来自逐维标准化残差；均值误差和最高π选择同时改善。完整生成仍须结合下面分页阅读。</p>
<p><a href="../../docs/records/r/2026-10-14_r_text_aux_review.md">完整分析</a> · <a href="review_checks.json">完整状态核对</a> · <a href="density_comparison.json">密度分解</a> · <a href="gradients_after.json">实际梯度</a> · <a href="decoded_shape_after.json">同4096位置GMM字形误差</a></p>
<table><tr><th>同一验证集指标</th><th>训练前</th><th>训练后</th></tr>TABLE</table>
<details open><summary>完整训练与在线验证曲线</summary><img src="training_curves.png"></details>
<details><summary>密度分解和均值/方差变化</summary><img src="density_decomposition.png"><pre>DELTA</pre></details>
<h2>标准字体分类读出：同前文、同字体前缀策略</h2>
<p>只用自己生成的前缀。多个合法落点共享同一形状时，原GMM给标准E模板打分；不经过D重建。既有动态词换行引发的回流与KV刷新保持同一实现，详细次数记入输出。</p>
<select id="readoutcase"></select><p id="prompt"></p><div class="grid"><section class="panel"><h3>训练前</h3><div id="old"></div></section><section class="panel"><h3>训练后</h3><div id="new"></div></section><section class="panel"><h3>实际前文</h3><div id="condition"></div></section></div>
<details open><summary>完整文字、停止原因与字体策略记录</summary><pre id="readoutmetrics"></pre></details>
<h2>原GMM生成：同前文、原D曲线回馈</h2><select id="gmmcase"></select><p id="gmmprompt"></p><div class="grid"><section class="panel"><h3>训练前最高π均值</h3><div id="gmmold"></div></section><section class="panel"><h3>训练后最高π均值</h3><div id="gmmnew"></div></section><section class="panel"><h3>训练后采样T0.5</h3><div id="gmmsample"></div></section></div>
<details><summary>原GMM完整记录</summary><pre id="gmmmetrics"></pre></details>
<script>const data=DATA;const el=id=>document.getElementById(id);function choose(id,rows,fn){rows.forEach((v,i)=>{let option=document.createElement('option');option.value=i;option.textContent='prompt'+i+' · '+v.prompt;el(id).appendChild(option)});el(id).onchange=fn;fn()}
function renderReadout(){let c=data.readout[+el('readoutcase').value];el('prompt').textContent=c.prompt;Object.entries(c.panels).forEach(([k,v])=>el(k).innerHTML=v);el('readoutmetrics').textContent=JSON.stringify({before:c.old,after:c.new},null,2)}
function renderGMM(){let c=data.gmm[+el('gmmcase').value];el('gmmprompt').textContent=c.prompt;Object.entries(c.panels).forEach(([k,v])=>el('gmm'+k).innerHTML=v);el('gmmmetrics').textContent=JSON.stringify(c.summary,null,2)}
choose('readoutcase',data.readout,renderReadout);choose('gmmcase',data.gmm,renderGMM)</script></html>'''
    (root/'index.html').write_text(page.replace('TABLE', table).replace('DELTA', html.escape(json.dumps(text_delta, indent=2))).replace('DATA', data), encoding='utf-8')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--before_readout', type=Path, default=Path('tmp_smoke/r_shape_space_readout_20261014'))
    parser.add_argument('--before_r', type=Path, default=Path('tmp_smoke/r_t1_final_review_20261013'))
    args = parser.parse_args(); render(args.root, args.before_readout, args.before_r)
