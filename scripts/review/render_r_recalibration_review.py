"""2026-10-19：α持续校准续训的同终点对照、权重审计与分页样张，CPU。"""
from __future__ import annotations

import argparse
import gc
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def audit(out,baseline):
    import torch
    torch.set_num_threads(4)
    current=torch.load(out/'model.pt',map_location='cpu',weights_only=False)
    old=torch.load(baseline/'model.pt',map_location='cpu',weights_only=False)
    assert current['model'].keys()==old['model'].keys()
    frozen=[k for k in old['model'] if k.startswith(('encoder.','decoder.','dct_head.'))]
    frozen_equal=all(torch.equal(current['model'][k],old['model'][k]) for k in frozen)
    cfg_diff={k:dict(old=old['cfg'].get(k),new=current['cfg'].get(k))
              for k in current['cfg'] if current['cfg'][k]!=old['cfg'].get(k)}
    assert set(cfg_diff)=={'r_text_aux_weight'} and frozen_equal
    assert current['scheduler']['spec']==old['scheduler']['spec']
    assert current['step']==old['step'] and current['phase_step']==old['phase_step']
    assert current['text_aux_manifest']==old['text_aux_manifest']
    assert current['train_args']['total_inputs']==old['train_args']['total_inputs']
    assert current['train_args']['text_aux_alpha']==current['cfg']['r_text_aux_weight']
    assert current['train_args']['text_aux_gradient_ratio']==.1
    assert current['train_args']['text_aux_calibration_interval_inputs']==20000000
    assert not current['train_args']['text_aux_calibration_pending']
    groups=[]
    for group in current['optimizer']['param_groups']:
        steps=[int(current['optimizer']['state'][p]['step']) for p in group['params']]
        assert min(steps)==max(steps)==current['step']
        groups.append(dict(role=group.get('role','R'),parameters=len(steps),step=steps[0],lr=group['lr']))
    finite=all(bool(torch.isfinite(value).all()) for value in current['model'].values())
    assert finite and current['scheduler']['inputs']>=current['train_args']['total_inputs']
    result=dict(step=current['step'],successful_inputs=current['scheduler']['inputs'],
        input_budget=current['train_args']['total_inputs'],counters=current['counters'],
        frozen_ae_equal=frozen_equal,frozen_tensor_count=len(frozen),weights_finite=finite,
        architecture_and_sigma_unchanged=True,cfg_diff=cfg_diff,optimizer_groups=groups,
        scaler=current['scaler'],scheduler=current['scheduler'],
        calibration_count=current['train_args']['text_aux_calibration_count'],
        calibration_last_inputs=current['train_args']['text_aux_calibration_last_inputs'],
        parent_source=current['parent_source'],source_commit=current['source_commit'])
    write(out/'checkpoint_audit.json',result)
    del current,old;gc.collect()
    return result


def render(out,baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    current=read(out/'summary.json');old=read(baseline/'summary.json')
    curve=[read(out/'validation_initial.json')]+[
        json.loads(s) for s in (out/'validation.jsonl').read_text().splitlines() if s.strip()]
    old_curve=[json.loads(s) for s in (out/'baseline_validation.jsonl').read_text().splitlines() if s.strip()]
    calibrations=[json.loads(s) for s in (out/'text_aux_calibration.jsonl').read_text().splitlines() if s.strip()]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    start=curve[0]['step']
    for rows,label,color in (([r for r in old_curve if r['step']>=start-400],'Fixed alpha','#888888'),
                             (curve,'10% recurrent calibration','#1976b7')):
        axes[0].plot([r['step'] for r in rows],[r['text_aux']['all']['shape_ce'] for r in rows],
                     marker='.',label=label,color=color)
        axes[1].plot([r['step'] for r in rows],[r['content_nll'] for r in rows],marker='.',label=label,color=color)
    for ax,title in zip(axes[:2],('Validation stroke-shape CE','Validation geometry NLL')):
        ax.set(title=title,xlabel='Cumulative successful updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[2].step([r['step'] for r in calibrations],[r['alpha'] for r in calibrations],where='post',label='Measured alpha')
    axes[2].axhline(old['gradient_probe']['alpha'],color='#888888',ls='--',label='Old fixed alpha')
    axes[2].set(title='Auxiliary loss weight',xlabel='Cumulative successful updates',yscale='log')
    axes[2].grid(alpha=.2);axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'comparison_curves.png',dpi=150);plt.close(fig)
    # 直接从原始点列绘制静态预览，分页坐标与原SVG一致。
    for case in range(2):
        pairs=[(read(folder/f'gmm{case}.json'),label) for folder,label in
               ((baseline,'Fixed alpha'),(out,'10% recurrent calibration'))]
        columns=max(len({s['page_id'] for s in d['strokes']}) for d,_ in pairs)
        fig,axes=plt.subplots(2,columns,figsize=(4*columns,8),squeeze=False)
        for row,(data,label) in enumerate(pairs):
            pages=sorted({s['page_id'] for s in data['strokes']})
            for column,ax in enumerate(axes[row]):
                ax.set(xlim=(0,1),ylim=(1,0),aspect='equal',xticks=[],yticks=[])
                if column>=len(pages):ax.set_visible(False);continue
                page=pages[column];ax.set_title(f'{label} / page {page}',fontsize=10)
                for stroke in data['strokes']:
                    if stroke['page_id']!=page:continue
                    control=stroke['kind'] in ('bos','eop','eod')
                    color=('#bc3e70' if control else '#156e9b') if stroke['generated'] else '#929292'
                    ax.plot(*zip(*stroke['points']),color=color,linewidth=.8)
        fig.tight_layout();fig.savefig(out/f'gmm{case}_comparison.png',dpi=140);plt.close(fig)
    metrics=[('笔画形状CE',lambda r:r['text_aux']['all']['shape_ce'],'.4f'),
             ('全部形状准确率',lambda r:1-r['text_aux']['all']['shape_error'],'.2%'),
             ('字符首笔形状准确率',lambda r:1-r['text_aux']['first']['shape_error'],'.2%'),
             ('字符续接形状准确率',lambda r:1-r['text_aux']['continuation']['shape_error'],'.2%'),
             ('逐笔空格准确率',lambda r:1-r['text_aux']['all']['space_error'],'.2%'),
             ('正文几何NLL',lambda r:r['content_nll'],'.3f')]
    table=''.join(f'<tr><td>{name}</td><td>{fn(old["teacher_forcing"]):{fmt}}</td>'
                  f'<td>{fn(current["teacher_forcing"]):{fmt}}</td></tr>' for name,fn,fmt in metrics)
    cases=[]
    for kind,count in (('readout',4),('gmm',2)):
        for i in range(count):
            columns=[]
            for folder,label in ((baseline,'原固定α'),(out,'持续校准10%')):
                d=read(folder/f'{kind}{i}.json')
                texts=d.get('generated_page_texts',{})
                if isinstance(texts,dict):texts=[v for k,v in sorted(texts.items(),key=lambda kv:int(kv[0]))]
                svg=(folder/f'{kind}{i}.svg').read_text(encoding='utf-8').split('?>')[-1]
                columns.append(f'<article><h3>{label}</h3><p>停止：{html.escape(d["stop_reason"])}</p>'
                               f'<pre>{html.escape(" | ".join(texts))}</pre>{svg}</article>')
            title='约束分类闭环' if kind=='readout' else 'GMM最高π均值闭环'
            cases.append(f'<section><h2>{title} / 前文{i+1}</h2><div class="pair">{"".join(columns)}</div></section>')
    page=f'''<!doctype html><meta charset="utf-8"><title>R α校准续训验收</title>
<style>body{{font:16px/1.6 system-ui;margin:28px;max-width:1800px}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:6px 14px}}img{{max-width:100%}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}
article{{min-width:0}}svg{{width:100%;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:10px}}
section{{border-top:2px solid #bbb;margin-top:25px}}@media(max-width:900px){{.pair{{grid-template-columns:1fr}}}}</style>
<h1>R α持续校准续训验收 · 2026-10-19</h1>
<p>从24948步完整恢复到28702步；原模型与10%校准模型使用相同架构、原输入终点和WSD。
同128篇验证故事、226个窗口、161044个目标；准确率取原始logits。</p>
<table><tr><th>指标</th><th>原固定α收尾</th><th>10%持续校准收尾</th></tr>{table}</table>
<p>最终α={current['gradient_probe']['alpha']:.4f}；同四篇增强文档上，末h辅助/几何梯度比
{current['gradient_probe']['median_ratios']['hidden']:.2%}，R骨干梯度比
{current['gradient_probe']['median_ratios']['R_backbone']:.2%}。</p>
<img src="comparison_curves.png"><p>下方使用相同前文。分类闭环沿用字体笔顺约束和标准笔画回馈，GMM闭环沿用最高π均值解码。</p>
{''.join(cases)}'''
    (out/'comparison.html').write_text(page,encoding='utf-8')
    write(out/'comparison.json',dict(old=old['teacher_forcing'],current=current['teacher_forcing'],
        old_gradient=old['gradient_probe'],current_gradient=current['gradient_probe'],
        calibration_count=len(calibrations),alpha_min=min(r['alpha'] for r in calibrations),
        alpha_max=max(r['alpha'] for r in calibrations)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=ROOT/'tmp_smoke/r_linear13_final_review_20261018')
    args=parser.parse_args()
    checks=audit(args.out,args.baseline);render(args.out,args.baseline)
    print(json.dumps(dict(audit_passed=True,step=checks['step'],successful_inputs=checks['successful_inputs'])))


if __name__=='__main__':main()
