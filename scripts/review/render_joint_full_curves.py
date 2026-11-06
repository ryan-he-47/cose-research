"""2026-10-22：完整R主线与图文全指标曲线；离线交互、PNG和原始CSV。"""
import argparse
from collections import defaultdict
import csv
import html
import json
import math
from pathlib import Path
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs


TASKS=('T1','O1','O2','S1','S2','S3','S4','S5')
KINDS=('text','sketch','selection','control')


def read_rows(path):
    result=[]
    if not path.exists():return result
    for line in path.read_text(encoding='utf-8').splitlines():
        try:result.append(json.loads(line))
        except ValueError:pass
    return result


def get(row,key):
    for part in key.split('.'):
        if not isinstance(row,dict):return None
        row=row.get(part)
    return row if isinstance(row,(int,float)) and math.isfinite(row) else None


def trace(rows,key,name=None,*,xkey='successful_inputs',scale=1e6,yscale=1,bin_size=10000000,raw=False):
    selected=[(get(r,xkey),get(r,key)) for r in rows]
    selected=[(x,y/yscale) for x,y in selected if x is not None and y is not None]
    if raw or not bin_size:
        return dict(x=[x/scale for x,y in selected],y=[y for x,y in selected],name=name or key,raw=raw)
    bins=defaultdict(list)
    for x,y in selected:bins[int(x//bin_size)].append((x,y))
    return dict(x=[statistics.median(x for x,y in bins[b])/scale for b in sorted(bins)],
                y=[statistics.median(y for x,y in bins[b]) for b in sorted(bins)],name=name or key)


def flatten(value,prefix=''):
    result={}
    for key,v in value.items():
        name=prefix+key
        if isinstance(v,dict):result.update(flatten(v,name+'.'))
        elif isinstance(v,(str,int,float,bool)) or v is None:result[name]=v
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();root=args.out
    out=root/'curves';out.mkdir(exist_ok=True)
    logs={key:read_rows(root/name) for key,name in (
        ('text_initial','cose_r_linear13_633815.out'),('text_recal','cose_r_recal10p_657077.out'),
        ('joint_initial','cose_r_joint_661550.out'),('joint_pipeline','cose_r_joint_661737.out'))}
    successful={k:[r for r in v if r.get('success')] for k,v in logs.items()}
    joint=sorted({r['step']:r for k in ('joint_initial','joint_pipeline') for r in successful[k]}.values(),key=lambda r:r['step'])
    failed=[r for k in ('joint_initial','joint_pipeline') for r in logs[k] if r.get('success') is False]
    initial=[r for r in successful['text_initial'] if r['step']<=24948]
    replaced=[r for r in successful['text_initial'] if r['step']>24948]
    recal=[r for r in successful['text_recal'] if r['step']>24948]
    base_inputs=json.loads((root/'text_recal/latest.json').read_text(encoding='utf-8'))['successful_inputs']
    lineage=[dict(r,history_inputs=r['successful_inputs'],regime='text_initial') for r in initial]
    lineage += [dict(r,history_inputs=r['successful_inputs'],regime='text_recal') for r in recal]
    lineage += [dict(r,history_inputs=base_inputs+r['successful_inputs'],regime='joint') for r in joint]
    valid={key:read_rows(root/key/'validation.jsonl') for key in ('text_initial','text_recal','joint')}
    initial_valid=json.loads((root/'joint/validation_initial.json').read_text(encoding='utf-8'))
    jvalid=[initial_valid,*valid['joint']]
    jvalid=list({r['step']:r for r in jvalid}.values())
    calibration=read_rows(root/'joint/text_aux_calibration.jsonl')
    flat=[flatten(r) for r in joint]
    columns=sorted({key for r in flat for key in r})
    with (out/'joint_logged_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,columns);writer.writeheader();writer.writerows(flat)
    attempts=[flatten(dict(r,source_segment=k)) for k in ('joint_initial','joint_pipeline') for r in logs[k] if 'success' in r]
    attempt_columns=sorted({k for r in attempts for k in r})
    with (out/'joint_all_attempts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,attempt_columns);writer.writeheader();writer.writerows(attempts)
    (out/'joint_logged_metrics.json').write_text(json.dumps(joint),encoding='utf-8')
    (out/'lineage.json').write_text(json.dumps(lineage),encoding='utf-8')
    figures={};catalog=[]

    def grid(key,title,panels,cols=2,xlabel='Joint inputs (million)'):
        nrows=math.ceil(len(panels)/cols)
        fig,axes=plt.subplots(nrows,cols,figsize=(7*cols,3.1*nrows),squeeze=False)
        interactive=make_subplots(rows=nrows,cols=cols,subplot_titles=[p['title'] for p in panels])
        for i,p in enumerate(panels):
            ax=axes.flat[i]
            for t in p['traces']:
                if not t['x']:continue
                mode='markers' if t.get('raw') else 'lines'
                ax.plot(t['x'],t['y'],label=t['name'],alpha=.18 if t.get('raw') else .9,
                        lw=.6 if t.get('raw') else 1.3,marker='.' if t.get('raw') else None,ms=1)
                interactive.add_trace(go.Scattergl(x=t['x'],y=t['y'],name=t['name'],mode=mode,
                    opacity=.22 if t.get('raw') else 1.,showlegend=True,
                    marker=dict(size=2),line=dict(width=1.5)),row=i//cols+1,col=i%cols+1)
            if p.get('log'):ax.set_yscale('log');interactive.update_yaxes(type='log',row=i//cols+1,col=i%cols+1)
            ax.set(title=p['title'],xlabel=xlabel,ylabel=p.get('ylabel',''));ax.grid(alpha=.2)
            if any(t['x'] for t in p['traces']):ax.legend(fontsize=7)
        for ax in axes.flat[len(panels):]:ax.set_visible(False)
        fig.suptitle(title,fontsize=14);fig.tight_layout();fig.savefig(out/(key+'.png'),dpi=140);plt.close(fig)
        interactive.update_layout(title=title,height=max(500,nrows*300),template='plotly_white',
            margin=dict(l=65,r=25,t=80,b=55),legend=dict(orientation='h',y=-.08))
        figures[key]=json.loads(interactive.to_json());catalog.append(dict(key=key,title=title))

    history_panels=[]
    for key,label in (('task_nll.T1','T1 geometric NLL'),('text_aux.text.shape_ce','Text shape CE (training)'),
                      ('lr','Learning rate'),('text_aux_alpha','Alpha'),('grad_norm','R gradient norm'),('control_nll','Control NLL')):
        traces=[]
        for name,rs in (('text_initial',initial),('text_recal',recal),('joint',joint),('replaced_tail',replaced)):
            traces.append(trace(rs,key,name,xkey='step',scale=1,bin_size=200,raw=False))
        history_panels.append(dict(title=label,traces=traces,log=key in ('lr','grad_norm','text_aux_alpha')))
    grid('01_lineage','R lineage: text from scratch -> recalibration -> joint',history_panels,xlabel='Global optimizer step')
    vregimes=(('text_initial',[r for r in valid['text_initial'] if r['step']<=24948]),
              ('text_recal',[r for r in valid['text_recal'] if r['step']>24948]),('joint',jvalid),
              ('replaced_tail',[r for r in valid['text_initial'] if r['step']>24948]))
    grid('01b_lineage_validation','R lineage: held-out text validation',[
        dict(title=label,traces=[trace(rs,key,name,xkey='step',scale=1,bin_size=0) for name,rs in vregimes])
        for key,label in (('text_aux.all.shape_ce','Held-out shape CE'),('text_aux.first.shape_ce','First stroke CE'),
                          ('text_aux.continuation.shape_ce','Continuation CE'),('text_aux.all.space_error','Space error'),
                          ('content_nll','Text geometric NLL'),('control_nll','Control geometric NLL'))],xlabel='Global optimizer step')
    grid('02_tasks','Joint: all eight task body NLL',[
        dict(title=t,traces=[trace(joint,'task_nll.'+t,t+' raw',raw=True),trace(joint,'task_nll.'+t,t+' median')]) for t in TASKS])
    aux_groups=('all','text','first','continuation','control','sketch')
    grid('03_auxiliary','Joint auxiliary heads: shape and per-stroke space',[
        dict(title=field,traces=[trace(joint,f'text_aux.{b}.{field}',b) for b in aux_groups],log=field in ('shape_ce','space_bce'))
        for field in ('shape_ce','shape_error','space_bce','space_error','count','space_positive')])
    grid('04_validation','Held-out text: same 128 stories, 161044 targets',[
        dict(title=field,traces=[trace(jvalid,f'text_aux.{b}.{field}',b,bin_size=0) for b in ('all','first','continuation','control')])
        for field in ('shape_ce','shape_error','space_bce','space_error')]+[
        dict(title='Geometric NLL',traces=[trace(jvalid,'content_nll','text',bin_size=0),trace(jvalid,'control_nll','control',bin_size=0)]),
        dict(title='Calibration alpha',traces=[trace(calibration,'alpha','alpha',bin_size=0)])])
    fields=('posterior_sigma','posterior_mu_mse','top_pi_mu_mse','top_pi_matches_posterior','pi_entropy')
    grid('05_gmm','GMM: posterior component fit versus top-pi decoding',[
        dict(title=f'{b} / {f}',traces=[trace(joint,f'gmm_buckets.{b}.{f}',b)],log=f in ('posterior_sigma','posterior_mu_mse','top_pi_mu_mse'))
        for b in KINDS for f in fields],cols=5)
    fig,axes=plt.subplots(4,1,figsize=(16,12));interactive=make_subplots(rows=4,cols=1,subplot_titles=list(KINDS))
    for i,bucket in enumerate(KINDS):
        bins=defaultdict(lambda:np.zeros(32))
        for r in joint:
            h=r.get('gmm_buckets',{}).get(bucket,{}).get('posterior_components',{})
            for c,v in h.items():bins[int(r['successful_inputs']//10000000)][int(c)]+=v
        keys=sorted(bins);z=np.asarray([bins[k]/max(1,bins[k].sum()) for k in keys]).T
        axes[i].imshow(z,aspect='auto',origin='lower',extent=(keys[0]*10,(keys[-1]+1)*10,-.5,31.5),vmin=0,vmax=max(.1,float(z.max())))
        axes[i].set(title=bucket,xlabel='Joint inputs (million)',ylabel='Posterior component')
        interactive.add_trace(go.Heatmap(x=[k*10+5 for k in keys],y=list(range(32)),z=z,colorscale='Viridis',showscale=False),row=i+1,col=1)
    fig.tight_layout();fig.savefig(out/'06_components.png',dpi=140);plt.close(fig)
    interactive.update_layout(height=1100,title='GMM posterior component use',template='plotly_white')
    figures['06_components']=json.loads(interactive.to_json());catalog.append(dict(key='06_components',title='GMM posterior component use'))
    grid('07_system','Training state and pipeline',[
        dict(title='Weighted total loss',traces=[trace(joint,'loss','loss')]),
        dict(title='LR / alpha',traces=[trace(joint,'lr','lr'),trace(joint,'text_aux_alpha','alpha')],log=True),
        dict(title='Gradient norms',traces=[trace(joint,'grad_norm','R'),trace(joint,'text_aux_grad_norm','aux')],log=True),
        dict(title='FP16 scale',traces=[trace(joint,'scale','scale',bin_size=0),trace(failed,'scale','failed updates',raw=True)]),
        dict(title='GPU memory GiB',traces=[trace(joint,'device_free_gib','free GiB'),trace(joint,'peak_memory_bytes','allocated GiB',yscale=2**30),trace(joint,'reserved_memory_bytes','reserved GiB',yscale=2**30)]),
        dict(title='Capacity and microbatches',traces=[trace(joint,'padded_stroke_budget','padding budget'),trace(joint,'microbatches','microbatches')],log=True),
        dict(title='Prepare / full update seconds',traces=[trace(joint,'prepare_seconds','prepare'),trace(joint,'update_seconds','whole update')]),
        dict(title='Successful inputs / second',traces=[trace(joint,'inputs_per_second','inputs/s')])])
    shares=[];targets=[];exposures=[]
    for r in joint:
        total=sum(r['task_inputs'].values());body=sum(r['body_targets'].values())
        shares.append(dict(successful_inputs=r['successful_inputs'],**{t:r['task_inputs'][t]/max(1,total) for t in TASKS}))
        targets.append(dict(successful_inputs=r['successful_inputs'],efficiency=(body+r['control_targets'])/total,
                            **{t:r['body_targets'][t]/max(1,body) for t in TASKS}))
        cov=r.get('coverage') or {}
        exposures.append(dict(successful_inputs=r['successful_inputs'],**{t:cov.get('completed_sources',{}).get(t,0)/n for t,n in cov.get('source_limits',{}).items()}))
    grid('08_data','Input shares, supervised targets and source exposure',[
        dict(title='Input cost share',traces=[trace(shares,t,t) for t in TASKS]),
        dict(title='Body target share',traces=[trace(targets,t,t) for t in TASKS]),
        dict(title='Supervised target / input efficiency',traces=[trace(targets,'efficiency','body+control / all inputs')]),
        dict(title='Completed source passes (uncapped)',traces=[trace(exposures,t,t) for t in TASKS])])
    summary=dict(joint_first_step=joint[0]['step'],joint_last_step=joint[-1]['step'],
        joint_phase_step=joint[-1]['phase_step'],joint_inputs=joint[-1]['successful_inputs'],
        base_inputs=base_inputs,total_lineage_inputs=base_inputs+joint[-1]['successful_inputs'],
        joint_logged_success=len(joint),joint_logged_failed=sum(r.get('success') is False for key in ('joint_initial','joint_pipeline') for r in logs[key]),
        joint_oom_retries=sum(bool(r.get('capacity_oom_retry')) for key in ('joint_initial','joint_pipeline') for r in logs[key]),
        expected_logging_stride=10,mainline_text_cutoff=24948,replaced_tail_points=len(replaced),
        validation_points=len(jvalid),calibration_points=len(calibration),metric_columns=len(columns),
        last_logged_update=joint[-1],plots=catalog)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'figures.js').write_text('window.COSE_CURVES='+json.dumps(figures)+';',encoding='utf-8')
    (out/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    options=''.join(f'<option value="{v["key"]}">{html.escape(v["title"])}</option>' for v in catalog)
    images=''.join(f'<details><summary>{html.escape(v["title"])} · PNG</summary><a href="{v["key"]}.png"><img src="{v["key"]}.png" loading="lazy"></a></details>' for v in catalog)
    page=f'''<!doctype html><meta charset="utf-8"><title>完整训练曲线 / step {joint[-1]['step']}</title>
<style>body{{font:16px/1.6 system-ui;margin:22px;color:#24344b}}select{{padding:10px;max-width:90%}}img{{max-width:100%}}details{{border-top:1px solid #ccd4df;padding:12px}}.note{{background:#edf3fa;padding:16px}}#chart{{width:100%}}</style>
<h1>完整训练曲线：R从随机初始化到图文暂停点</h1><div class="note"><p>文字从头→24948；α修正24948→28702；图文接续28702→{joint[-1]['step']}（阶段{joint[-1]['phase_step']}步，{joint[-1]['successful_inputs']:,}输入）。
未采用的原文字尾段单独标作replaced_tail。图文前263步和后续流水线段均已接齐。</p>
<p>横轴标Global step时为累计Adam更新；其他图默认图文阶段百万输入。实线为每1000万输入中位数，原始日志每10步采样；训练阶段边界不跨段平滑。图例可点击，图框可拖选放大。</p>
<p>训练保持停止；本页没有启动退火。完整CSV有{len(columns)}列。</p></div>
<details><summary>指标口径</summary><ul>
<li>NLL为48维嵌入的联合负对数密度，不是字节CE；可以为负。形状CE/错误率按笔画形状类别。</li>
<li>总loss含任务权重、控制符权重和动态α；α变化时，不把总loss变化直接解读为能力变化。</li>
<li>posterior指标使用真实下一笔选责任度最高分量，只用于诊断。top-pi只按模型π选分量，接近最高π均值推理。</li>
<li>σ是标准差。posterior_mu_mse/top_pi_mu_mse均为λ空间均值误差；pi_entropy单位为nat。</li>
<li>梯度范数是裁剪前范数；R和辅助头分别裁剪。FP16 scale不是学习率。</li>
<li>显存peak是日志记录的CUDA累计峰值，free为当时整卡空余；完整update耗时已含prepare。</li>
<li>输入份额按包括条件笔画的成本计；body target份额按被监督正文计。源池遍数按已消费源游标，不把预取算曝光。</li>
</ul></details>
<p><a href="joint_logged_metrics.csv">成功更新指标 CSV</a> · <a href="joint_all_attempts.csv">含失败尝试 CSV</a> · <a href="joint_logged_metrics.json">原始图文指标 JSON</a> · <a href="summary.json">来源/完整性摘要</a></p>
<select id="select">{options}</select><div id="chart"></div>{images}
<script src="plotly.min.js"></script><script src="figures.js"></script><script>
const s=document.getElementById('select');function show(){{const f=COSE_CURVES[s.value];Plotly.react('chart',f.data,f.layout,{{responsive:true,scrollZoom:true}})}}s.onchange=show;show();
</script>'''
    (out/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('joint_first_step','joint_last_step','joint_phase_step','joint_inputs','total_lineage_inputs','joint_logged_success','joint_logged_failed','metric_columns')}))


if __name__=='__main__':main()
