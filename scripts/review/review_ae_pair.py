"""2026-10-04：比较两条已结束AE的日志、同笔几何与λ聚类；只读，不训练。"""
from __future__ import annotations

# 2026-10-12：分类目录中的脚本也可从仓库根直接运行。
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

BUCKETS=('text','sketch_object','sketch_scene','sketch')
NAMES={'text':'文字','sketch_object':'对象图形','sketch_scene':'场景图形','sketch':'图形汇总'}
LABELS={'original45':'原轮45%文字','text5':'5%文字'}
COLORS={'original45':'#2667a8','text5':'#ce7937'}
SLURM_SECONDS={'initial':425,'base':30851,'cosine':11305,'text5':21010}


def lines(path):
    result=[]
    for line in path.read_text(encoding='utf-8').splitlines():
        try: value=json.loads(line)
        except ValueError: continue
        if isinstance(value,dict): result.append(value)
    return result


def series(root,stages):
    validation=[];updates=[];offset=0
    for stage in stages:
        folder=root/stage
        data=lines(folder/'validation.jsonl')
        stop=json.loads((folder/'run_stop.json').read_text(encoding='utf-8'))
        # 末次验证写完到run_stop仅有保存耗时；小时轴用于近似对齐GPU用时。
        start=data[-1]['unix_seconds']-stop['elapsed_seconds']
        for value in data:
            value['active_hours_approx']=(offset+value['unix_seconds']-start)/3600
            validation.append(value)
        updates.extend(v for v in lines(folder/'train.log') if 'success' in v)
        offset+=SLURM_SECONDS[stage]
    return sorted(validation,key=lambda v:v['step']),updates


def at_step(values,step,bucket,field='cd'):
    x=np.array([v['step'] for v in values])
    if not x[0]<=step<=x[-1]: raise ValueError('比较位置超出验证曲线，不能外推')
    y=np.array([v['buckets'][bucket][field]['mean'] for v in values])
    return float(np.interp(step,x,y))


def comparison(before,after):
    return {'before':before,'after':after,'change_percent':100*(after/before-1)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir',type=Path,required=True)
    parser.add_argument('--baseline_dir',type=Path,required=True)
    parser.add_argument('--samples',type=Path,required=True)
    args=parser.parse_args()
    root=args.run_dir
    values={};updates={};states={};audits={}
    for name,stages,folder in (('original45',('initial','base','cosine'),'cosine'),
                                ('text5',('text5',),'text5')):
        values[name],updates[name]=series(root,stages)
        state=torch.load(root/folder/'final.pt',map_location='cpu',weights_only=False)
        assert state['validation_source']['sha256']==hashlib.sha256(args.samples.read_bytes()).hexdigest()
        assert state['step']==values[name][-1]['step']
        assert all(torch.isfinite(t).all() for t in state['model'].values())
        assert np.isclose(state['best_validation_cd'],min(v['monitor_cd'] for v in values[name]),rtol=0,atol=1e-14)
        states[name]=state
        audits[name]={'step':state['step'],'strokes':state['stroke_counts'],
            'gpu_seconds':sum(SLURM_SECONDS[s] for s in stages),'final_lr':state['optimizer']['param_groups'][0]['lr'],
            'scale':state['scaler']['scale'],'finite':True,'validation_source':state['validation_source'],
            'recipe':state['train_args'],'quotas':state['data']['quotas'],
            'lowest_monitor_at_final':values[name][-1]['monitor_cd']==state['best_validation_cd']}
    assert states['original45']['cfg']==states['text5']['cfg']
    assert states['original45']['data_manifest']==states['text5']['data_manifest']
    assert states['original45']['train_args']=={
        k:v for k,v in states['text5']['train_args'].items() if k!='source_shares'}
    old=json.loads((args.baseline_dir/'report.json').read_text(encoding='utf-8'))['models']
    geometry=json.loads((root/'geometry/report.json').read_text(encoding='utf-8'))['models']
    end=values['text5'][-1]
    aligned={bucket:comparison(at_step(values['original45'],end['step'],bucket),
        end['buckets'][bucket]['cd']['mean']) for bucket in BUCKETS}
    exposure=[]
    # 新轮真实验证点对齐原轮图形笔画数；原轮退火前后分开标注。
    for v in values['text5']:
        if v['step']<5000: continue
        graphic_strokes=v['step']*3686
        reference_step=graphic_strokes/2048
        exposure.append({'text5_step':v['step'],'graphic_strokes':graphic_strokes,
            'original_interpolated_step':reference_step,
            'original_phase':'cosine' if reference_step>33525 else 'plateau',
            'sketch':comparison(at_step(values['original45'],reference_step,'sketch'),v['buckets']['sketch']['cd']['mean'])})
    recent={}
    for name in LABELS:
        final_step=values[name][-1]['step']
        start_step=max(v['step'] for v in values[name] if v['step']<=final_step-4000)
        recent[name]={'from_step':start_step,'to_step':final_step,
            'buckets':{b:comparison(at_step(values[name],start_step,b),at_step(values[name],final_step,b)) for b in BUCKETS}}
    summary={'audit':audits,'same_cfg_manifest':True,'comparison_interpolation':'between saved validation points; no extrapolation',
        'same_updates_reference_step':end['step'],'same_updates':aligned,'same_graphic_exposure':exposure,
        'recent':recent,'fp32_geometry':geometry,'historical_baselines':old,
        'fp16_final':{n:values[n][-1] for n in LABELS}}
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(15,8.8),layout='constrained')
    for ax,bucket in zip(axes[0],BUCKETS[:3]):
        for name in LABELS:
            v=values[name]
            ax.plot(np.array([r['step'] for r in v])/1000,[r['buckets'][bucket]['cd']['mean'] for r in v],
                    '.-',color=COLORS[name],label=LABELS[name])
        ax.axhline(old['old_frozen2']['buckets'][bucket]['cd']['mean'],ls=':',color='#555555',label='旧冻结AE（FP32）')
        ax.axvline(end['step']/1000,color='#999999',ls='--',lw=.8)
        ax.set(title=NAMES[bucket]+' CD',xlabel='成功更新（千次）',ylabel='CD',yscale='log')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0]
    for name,multiplier in (('original45',2048),('text5',3686)):
        v=[r for r in values[name] if r['step']>=5000]
        ax.plot(np.array([r['step'] for r in v])*multiplier/1e6,[r['buckets']['sketch']['cd']['mean'] for r in v],
                '.-',color=COLORS[name],label=LABELS[name])
    ax.axvline(33525*2048/1e6,color='#999999',ls='--',lw=.8,label='原轮开始余弦')
    ax.set(title='按图形曝光量看重建曲线',xlabel='累计图形笔画（百万）',ylabel='图形 CD')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1]
    for name in LABELS:
        v=[r for r in values[name] if r['step']>=10000]
        for field,style,label in (('shape_contribution','-','形状'),('position_contribution','--','位置')):
            ax.plot(np.array([r['step'] for r in v])/1000,[r['buckets']['sketch'][field]['mean'] for r in v],
                    style,color=COLORS[name],label=LABELS[name]+' '+label)
    ax.set(title='后段图形误差贡献',xlabel='成功更新（千次）',ylabel='CD贡献')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,2]
    for name in LABELS:
        u=updates[name]
        ax.plot(np.array([r['step'] for r in u])/1000,[r['lr'] for r in u],color=COLORS[name],label=LABELS[name])
    ax.set(title='实际学习率',xlabel='成功更新（千次）',ylabel='LR')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('CoSE AE：降低文字曝光后的训练结果（固定6144笔valid）',fontsize=15)
    fig.savefig(root/'curves.png',dpi=155);plt.close(fig)
    samples=dict(np.load(args.samples,allow_pickle=False))
    original=samples['cloud']
    bundles={'old':dict(np.load(args.baseline_dir/'old_frozen2_strokes.npz',allow_pickle=False))}
    bundles.update({n:dict(np.load(root/f'geometry/{n}_strokes.npz',allow_pickle=False)) for n in LABELS})
    for bundle in bundles.values():
        np.testing.assert_array_equal(bundle['cloud'],original)
    selected=[]
    for bucket in BUCKETS[:3]:
        ids=np.flatnonzero(samples['bucket']==bucket)
        ranked=ids[np.argsort(bundles['text5']['errors'][ids,0])]
        selected.extend(int(ranked[round(q*(len(ranked)-1))]) for q in (.5,.9))
    fig,axes=plt.subplots(len(selected),4,figsize=(12,2*len(selected)),layout='constrained')
    columns=(('source','原笔','#111111'),('old','旧冻结AE','#596773'),
             ('original45','原轮45%文字末步',COLORS['original45']),('text5','5%文字末步',COLORS['text5']))
    for row,index in enumerate(selected):
        curves=[original[index]]+[bundles[name]['reconstructed'][index] for name in ('old','original45','text5')]
        points=np.concatenate(curves);lo=points.min(0);hi=points.max(0)
        center=(hi+lo)/2;size=max(float((hi-lo).max())*1.25,.025)
        for col,(name,title,color) in enumerate(columns):
            ax=axes[row,col];curve=curves[col]
            if col: ax.plot(original[index,:,0],original[index,:,1],color='#cccccc',lw=2.3)
            ax.plot(curve[:,0],curve[:,1],color=color,lw=1.3)
            ax.set(xlim=(center[0]-size/2,center[0]+size/2),ylim=(center[1]+size/2,center[1]-size/2),aspect='equal',xticks=[],yticks=[])
            error='' if not col else f" CD={bundles[name]['errors'][index,0]:.4f}"
            ax.set_title(title+error,fontsize=9)
            if not col:
                metadata=json.loads(str(samples['metadata'][index]))
                ax.set_ylabel(f"{NAMES[str(samples['bucket'][index])]} / #{index}\n{metadata.get('label','文字')} / 5%桶内{'P50' if row%2==0 else 'P90'}",fontsize=9)
    fig.suptitle('同笔重建：同一行使用相同坐标范围，灰线为原笔；按5%模型桶内CD选P50/P90',fontsize=12)
    fig.savefig(root/'stroke_comparison.png',dpi=145);plt.close(fig)
    summary['stroke_examples']=selected
    tails={}
    for name,bundle in bundles.items():
        mask=np.char.startswith(samples['bucket'],'sketch');errors=bundle['errors'][mask]
        total=np.sort(errors[:,0]);tails[name]={'top10_percent_error_share':float(total[-int(len(total)*.1):].sum()/total.sum()),
            'mean_shape':float(errors[:,1].mean()),'mean_position':float(errors[:,2].mean())}
    summary['graphic_tail']=tails
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'audit':audits,'same_updates':aligned,'same_graphic_exposure':exposure,'recent':recent,'tails':tails},ensure_ascii=False))


if __name__=='__main__':
    main()
