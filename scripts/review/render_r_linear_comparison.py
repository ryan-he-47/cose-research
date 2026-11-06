"""2026-10-18：13层线性头R的最终/中期分页、旧模型指标与梯度配比展示，纯CPU。"""
from pathlib import Path
import argparse
import html
import json

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def render(out, baseline, stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    current = read(out / 'summary.json')
    curve = [json.loads(s) for s in (out / 'validation.jsonl').read_text().splitlines() if s.strip()]
    old = [read(baseline / 'validation_initial.json')]
    old += [json.loads(s) for s in (baseline / 'validation.jsonl').read_text().splitlines() if s.strip()]
    grad = read(out / 'gradient_comparison.json')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for rows, label, color in [(curve, '13L linear / fresh', '#2677b9'), (old, '12L MLP / aux continuation', '#ca7531')]:
        axes[0].plot([r['step'] for r in rows], [r['text_aux']['all']['shape_ce'] for r in rows], label=label, color=color)
        axes[1].plot([r['step'] for r in rows], [r['content_nll'] for r in rows], label=label, color=color)
    axes[0].set(title='Validation shape CE', xlabel='Cumulative R updates', ylabel='CE')
    axes[1].set(title='Validation geometry NLL', xlabel='Cumulative R updates', ylabel='NLL')
    for ax in axes[:2]: ax.grid(alpha=.2); ax.legend(fontsize=8)
    for i, name in enumerate(['hidden', 'R_backbone']):
        g = grad['groups'][name]
        axes[2].bar(i-.18, g['old_ratio']*100, width=.35, color='#ca7531', label='old' if i == 0 else None)
        axes[2].bar(i+.18, g['new_ratio']*100, width=.35, color='#2677b9', label='new' if i == 0 else None)
    axes[2].set(yscale='log', xticks=[0, 1], xticklabels=['last h', 'R backbone'],
                ylabel='Weighted auxiliary / geometry gradient (%)', title='Same four augmented documents')
    axes[2].legend(); fig.tight_layout(); fig.savefig(out / 'comparison_curves.png', dpi=150); plt.close(fig)
    final, previous = current['teacher_forcing'], old[-1]
    metrics = [('形状CE', lambda r: r['text_aux']['all']['shape_ce'], '.4f'),
               ('全部形状准确率', lambda r: 1-r['text_aux']['all']['shape_error'], '.2%'),
               ('字符首笔准确率', lambda r: 1-r['text_aux']['first']['shape_error'], '.2%'),
               ('字符续接准确率', lambda r: 1-r['text_aux']['continuation']['shape_error'], '.2%'),
               ('逐笔空格准确率', lambda r: 1-r['text_aux']['all']['space_error'], '.2%'),
               ('正文几何NLL', lambda r: r['content_nll'], '.3f')]
    rows = ''.join(f'<tr><td>{title}</td><td>{fn(final):{fmt}}</td><td>{fn(previous):{fmt}}</td></tr>'
                   for title, fn, fmt in metrics)
    sections = []
    for kind, count in [('readout', 4), ('gmm', 2)]:
        for i in range(count):
            cards = []
            for folder, label in [(stage, 'step8009'), (out, f"step{current['step']}")]:
                name = f'{kind}{i}'; data = read(folder / f'{name}.json')
                texts = data.get('generated_page_texts', {})
                text = '\n'.join(v for k, v in sorted(texts.items(), key=lambda kv: int(kv[0])))
                svg = (folder / f'{name}.svg').read_text(encoding='utf-8').split('?>')[-1]
                cards.append(f'<article><h3>{label}</h3><pre>{html.escape(text)}</pre>{svg}</article>')
            sections.append(f'<h2>{kind} / 前文{i}</h2><div class="pair">{"".join(cards)}</div>')
    page = f'''<!doctype html><meta charset="utf-8"><title>13层R最终验收</title>
<style>body{{font:16px/1.6 system-ui;margin:28px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #aaa;padding:6px 18px}}.pair{{display:grid;grid-template-columns:repeat(2,minmax(500px,1fr));gap:24px}}
article{{border:1px solid #ccc;padding:12px}}svg{{width:100%;height:auto}}pre{{white-space:pre-wrap}}img{{max-width:100%}}</style>
<h1>13层线性头R最终验收 / step{current['step']}</h1>
<p>同128篇故事/161044目标。旧模型是26203步GMM长训后，接已拟合MLP头再联合3032步；
新模型和所有头从头联合训练。两轮辅助α分别为5.87618和0.0372770。</p>
<table><tr><th>指标</th><th>新最终（本机复核）</th><th>旧联合续训最终（原验证）</th></tr>{rows}</table>
<p>同四篇增强文档：新版加权辅助/几何梯度中位数，h为0.155%，R骨干为0.076%；
旧版分别为42.9%和8.49%。新一轮初始化校准时的h目标为50%。</p>
<img src="comparison_curves.png"><p>以下中期/最终使用相同前文与笔数。
readout有字体前缀合法约束，GMM使用最高π分量均值解码回馈。</p>{''.join(sections)}'''
    (out / 'comparison.html').write_text(page, encoding='utf-8')
    return dict(step=current['step'], alpha_new=current['gradient_probe']['alpha'],
                alpha_old_over_new=grad['alpha_ratio_old_over_new'],
                current=final, old_final=previous, gradients=grad)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--baseline', type=Path, default=ROOT / 'tmp_smoke/r_text_aux_review_20261014')
    p.add_argument('--stage', type=Path, default=ROOT / 'tmp_smoke/r_linear13_stage8009_20261015')
    a = p.parse_args()
    result = render(a.out, a.baseline, a.stage)
    (a.out / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
