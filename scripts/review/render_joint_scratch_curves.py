"""2026-10-26：从头图文全程及退火段曲线，离线交互、PNG、CSV；不加载模型。"""
import argparse
from collections import defaultdict
import csv
import html
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from scripts.review.render_joint_full_curves import read_rows, get, flatten
from scripts.review.review_joint_scratch_stage import read, write

TASKS = ('T1','O1','O2','S1','S2','S3','S4','S5')


def csv_rows(path, rows):
    values = [flatten(r) for r in rows]
    columns = sorted({key for row in values for key in row})
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader();writer.writerows(values)
    return len(columns)


def metrics(row):
    return dict(step=row['step'], inputs=row['successful_inputs'], content_nll=row['content_nll'],
                control_nll=row['control_nll'], shape_ce=row['text_aux']['all']['shape_ce'],
                shape_accuracy=1-row['text_aux']['all']['shape_error'],
                first_accuracy=1-row['text_aux']['first']['shape_error'],
                continuation_accuracy=1-row['text_aux']['continuation']['shape_error'],
                space_accuracy=1-row['text_aux']['all']['space_error'])


def render(out, baseline):
    destination = out/'curves';destination.mkdir(exist_ok=True)
    recipe = read(out/'recipe.json')
    total = recipe['total_inputs'];start = total-recipe['decay_inputs']
    raw = [row for job in (674183,685064) for row in read_rows(out/f'cose_r_joint_scratch_{job}.out')]
    attempts = [r for r in raw if isinstance(r,dict) and isinstance(r.get('success'),bool)]
    # 重启边界如有重复成功记录，只保留同一步最后一条。
    unique = {r['step']:r for r in attempts if r['success'] and 'update_seconds' in r}
    updates = [unique[k] for k in sorted(unique)]
    validation = read_rows(out/'validation.jsonl')
    initial = dict(read(out/'validation_initial.json'), step=0, successful_inputs=0)
    validation = [initial, *validation]
    display_end=max(total,validation[-1]['successful_inputs'])/1e8
    calibration = read_rows(out/'text_aux_calibration.jsonl')
    previous = read(baseline/'audit.json')
    prior_input = previous['successful_inputs']
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'], 'axes.unicode_minus':False})
    figures = {};catalog = []

    def grid(key, title, rows, panels, *, smooth=False, columns=4):
        nr = math.ceil(len(panels)/columns)
        fig, axes = plt.subplots(nr, columns, figsize=(4.2*columns,3.1*nr), squeeze=False)
        interactive = make_subplots(rows=nr, cols=columns, subplot_titles=[p[0] for p in panels], vertical_spacing=.12)
        panel_values={}
        for i,(name,path,transform,logarithmic) in enumerate(panels):
            values = [(get(r,'successful_inputs'),get(r,path)) for r in rows]
            values = [(x/1e8,transform(y)) for x,y in values if x is not None and y is not None]
            if not values:continue
            panel_values[i]=values
            x,y = zip(*values);ax=axes.flat[i]
            ax.plot(x,y,color='#64748b',alpha=.25 if smooth else .8,lw=.6 if smooth else 1.3)
            interactive.add_trace(go.Scattergl(x=x,y=y,mode='lines',name=name+'原值',
                line=dict(width=.7,color='#94a3b8'),opacity=.4 if smooth else 1.,showlegend=False),row=i//columns+1,col=i%columns+1)
            if smooth:
                bins=defaultdict(list)
                for xx,yy in values:bins[int(xx*10)].append((xx,yy))
                xs=[statistics.median(x for x,y in bins[b]) for b in sorted(bins)]
                ys=[statistics.median(y for x,y in bins[b]) for b in sorted(bins)]
                ax.plot(xs,ys,color='#146c94',lw=1.4)
                interactive.add_trace(go.Scatter(x=xs,y=ys,mode='lines',name=name+'每1000万输入中位数',
                    line=dict(width=1.6,color='#146c94'),showlegend=False),row=i//columns+1,col=i%columns+1)
            ax.axvspan(start/1e8,total/1e8,color='#efaa40',alpha=.15)
            ax.axvline(start/1e8,color='#d97822',ls='--',lw=.9)
            ax.axvline(prior_input/1e8,color='#45956c',ls=':',lw=.8)
            ax.set(title=name,xlabel='成功输入 / 亿');ax.grid(alpha=.2)
            interactive.add_vrect(x0=start/1e8,x1=total/1e8,fillcolor='#efaa40',opacity=.12,line_width=0,row=i//columns+1,col=i%columns+1)
            interactive.add_vline(x=start/1e8,line_dash='dash',line_color='#d97822',row=i//columns+1,col=i%columns+1)
            interactive.add_vline(x=prior_input/1e8,line_dash='dot',line_color='#45956c',row=i//columns+1,col=i%columns+1)
            if logarithmic:
                ax.set_yscale('log');interactive.update_yaxes(type='log',row=i//columns+1,col=i%columns+1)
        for ax in axes.flat[len(panels):]:ax.set_visible(False)
        fig.suptitle(title+'；橙色为退火，绿线为50021步',fontsize=13)
        fig.tight_layout();fig.savefig(destination/(key+'.png'),dpi=140)
        if key in ('training','task_nll','validation','gmm'):
            fig.suptitle(title+'：退火前后；橙色为退火',fontsize=13)
            lo,hi=(start-1.2e8)/1e8,display_end
            for i,ax in enumerate(axes.flat[:len(panels)]):
                ax.set_xlim(lo,hi)
                visible=[y for x,y in panel_values.get(i,[]) if lo<=x<=hi and math.isfinite(y)]
                if not visible:continue
                if panels[i][3]:
                    values=[math.log10(y) for y in visible if y>0]
                    if not values:continue
                    a,b=min(values),max(values);pad=max((b-a)*.08,.025)
                    ax.set_ylim(10**(a-pad),10**(b+pad))
                else:
                    a,b=min(visible),max(visible);pad=max((b-a)*.08,max(abs(a),abs(b))*.0005,1e-7)
                    ax.set_ylim(a-pad,b+pad)
            fig.savefig(destination/('annealing_'+key+'.png'),dpi=150)
        plt.close(fig)
        interactive.update_layout(title=title,height=max(600,nr*310),template='plotly_white',margin=dict(t=80,b=50))
        figures[key]=json.loads(interactive.to_json());catalog.append((key,title))

    identity=lambda x:x
    percent_error=lambda x:100*(1-x)
    spec=lambda name,path,transform=identity,log=False:(name,path,transform,log)
    grid('training','训练几何、辅助项与调度',updates,[
        spec('总loss','loss'),spec('正文几何NLL','content_nll'),spec('控制符NLL','control_nll'),
        spec('形状CE（混合任务）','text_aux.all.shape_ce'),spec('空格BCE','text_aux.all.space_bce'),
        spec('学习率','lr',log=True),spec('动态辅助α','text_aux_alpha'),spec('R梯度范数','grad_norm',log=True)],smooth=True)
    grid('task_nll','八任务训练正文NLL',updates,[spec(task,'task_nll.'+task) for task in TASKS],smooth=True)
    grid('validation','同128篇文字valid，原始验证点',validation,[
        spec('文字正文几何NLL','content_nll'),spec('控制符NLL','control_nll'),spec('形状CE','text_aux.all.shape_ce'),
        spec('全部形状准确率 / %','text_aux.all.shape_error',percent_error),
        spec('字符首笔准确率 / %','text_aux.first.shape_error',percent_error),
        spec('字符续接准确率 / %','text_aux.continuation.shape_error',percent_error),
        spec('空格准确率 / %','text_aux.all.space_error',percent_error),spec('空格BCE','text_aux.all.space_bce')])
    grid('gmm','混合分布尺度、均值误差与分量选择',updates,[
        spec(kind+' / '+label,'gmm_buckets.'+kind+'.'+metric,log=metric!='top_pi_matches_posterior')
        for kind in ('text','sketch','selection','control')
        for label,metric in (('责任分量σ','posterior_sigma'),('责任分量均值MSE','posterior_mu_mse'),
                             ('最高π均值MSE','top_pi_mu_mse'),('最高π与责任分量一致率','top_pi_matches_posterior'))],smooth=True)
    grid('runtime','组批与运行记录',updates,[
        spec('输入/秒','inputs_per_second'),spec('准备秒数','prepare_seconds'),spec('更新秒数','update_seconds'),
        spec('整卡空余GiB','device_free_gib'),spec('R padding比','r_padding_ratio'),spec('E点padding比','e_point_padding_ratio'),
        spec('微批数量','microbatches'),spec('容量padding预算','padded_stroke_budget')],smooth=True)
    last_before = max((r for r in validation if r['successful_inputs']<=start),key=lambda r:r['successful_inputs'])
    final = validation[-1]
    a,b = metrics(last_before),metrics(final)
    windows = {'退火前末1亿':[r for r in updates if start-1e8<=r['successful_inputs']<start],
               '退火末5000万':[r for r in updates if total-5e7<=r['successful_inputs']<=total+1e5]}
    statistic_paths=['loss','content_nll','text_aux.all.shape_ce','text_aux_alpha',*[f'task_nll.{t}' for t in TASKS],
        *[f'gmm_buckets.{kind}.{metric}' for kind in ('text','sketch','selection','control')
          for metric in ('posterior_sigma','posterior_mu_mse','top_pi_mu_mse','top_pi_matches_posterior','pi_entropy')]]
    statistics_by_window={name:{path:statistics.median(v for v in (get(r,path) for r in rows) if v is not None)
        for path in statistic_paths}
        for name,rows in windows.items()}
    summary=dict(anneal_start_inputs=start,total_inputs=total,decay_inputs=recipe['decay_inputs'],
        peak_lr=recipe['peak_lr'],min_lr=recipe['min_lr'],last_before= a,final=b,
        validation_delta={k:b[k]-a[k] for k in a if k not in ('step','inputs')},
        validation_points=len(validation)-1,anneal_validation_points=sum(r['successful_inputs']>start for r in validation),
        logged_successful_updates=len(updates),failed_attempt_records=sum(not r['success'] for r in attempts),
        first_lr_drop=next({k:r[k] for k in ('step','successful_inputs','lr')} for r in updates
                           if r['successful_inputs']>start and r['lr']<recipe['peak_lr']),
        previous_snapshot_inputs=prior_input,window_medians=statistics_by_window,
        training_columns=csv_rows(destination/'training.csv',updates))
    csv_rows(destination/'validation.csv',validation);csv_rows(destination/'all_attempts.csv',attempts)
    csv_rows(destination/'calibration.csv',calibration)
    write(destination/'summary.json',summary)
    (destination/'plotly.min.js').write_text(get_plotlyjs(),encoding='utf-8')
    cards=''.join(f'<section><h2>{html.escape(title)}</h2><div id="{key}"></div><p><a href="{key}.png">全程PNG</a>'+
                  (f' · <a href="annealing_{key}.png">退火前后PNG</a>' if key in ('training','task_nll','validation','gmm') else '')+
                  '</p></section>' for key,title in catalog)
    page='<!doctype html><meta charset="utf-8"><title>从头图文全程与退火曲线</title><script src="plotly.min.js"></script>'
    page+='<style>body{font:16px/1.6 system-ui;margin:24px;color:#24344b}nav{position:sticky;top:0;background:#fff;padding:12px;z-index:5}button{padding:7px 14px;margin:4px}section{margin:30px 0;border-top:1px solid #ddd}</style>'
    page+='<h1>从头图文：完整训练及退火段</h1><p>0→36亿成功输入；34.56亿起退火，LR从1e-4降至1e-5。橙色区为退火；绿线是50021步已验收模型。训练蓝线为每1000万输入中位数，灰线为原日志值；验证保留全部原始点。</p>'
    page+=f'<nav><button onclick="range(null)">全程</button><button onclick="range([{(start-1.2e8)/1e8},{display_end}])">退火前后</button><button onclick="range([{start/1e8},{display_end}])">仅退火</button> · <a href="training.csv">训练CSV</a> · <a href="validation.csv">验证CSV</a> · <a href="summary.json">退火变化JSON</a></nav>'+cards
    page+='<script>const figures='+json.dumps(figures,ensure_ascii=False)+''';
for(const [id,f] of Object.entries(figures)){Plotly.newPlot(id,f.data,f.layout,{responsive:true});}
function range(r){for(const [id,f] of Object.entries(figures)){const patch={};
 for(const k of Object.keys(f.layout)){
  if(/^xaxis[0-9]*$/.test(k)){patch[k+".autorange"]=!r;if(r)patch[k+".range"]=r;}
  if(/^yaxis[0-9]*$/.test(k)){
   patch[k+".autorange"]=!r;if(!r)continue;
   const axis=k.replace('axis','');let values=[];
   for(const t of f.data){if((t.yaxis||'y')!==axis)continue;
    for(let i=0;i<t.x.length;i++){if(t.x[i]>=r[0]&&t.x[i]<=r[1]&&Number.isFinite(t.y[i]))values.push(t.y[i]);}}
   if(f.layout[k].type==='log')values=values.filter(v=>v>0).map(v=>Math.log10(v));
   if(!values.length)continue;const a=Math.min(...values),b=Math.max(...values);
   const pad=Math.max((b-a)*.08,Math.max(Math.abs(a),Math.abs(b))*.0005,1e-7);
   patch[k+".range"]=[a-pad,b+pad];
  }
 }Plotly.relayout(id,patch);}}
</script>'''
    (destination/'index.html').write_text(page,encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=ROOT/'tmp_smoke/r_joint_scratch_review_20261026')
    args=parser.parse_args()
    result=render(args.out.resolve(),args.baseline.resolve())
    print(json.dumps({k:result[k] for k in ('last_before','final','validation_delta','first_lr_drop','anneal_validation_points')},ensure_ascii=False))


if __name__=='__main__':main()
