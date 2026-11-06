"""2026-10-04：只读已落盘的AE训练/验证日志，画曲线并与同笔旧AE指标对照。"""
from __future__ import annotations

# 2026-10-12：分类目录中的脚本也可从仓库根直接运行。
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


LABELS={'text':'文字','sketch_object':'对象图形','sketch_scene':'场景图形','sketch':'图形汇总'}


def json_lines(path):
    values=[]
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        try:values.append(json.loads(line))
        except ValueError:continue  # 实时文件的最后一行可能尚未写完。
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args()
    root=args.run_dir
    log=json_lines(root/'train.log')
    updates=[v for v in log if 'success' in v]
    validation=json_lines(root/'validation.jsonl')
    initial=json_lines(root/'validation_initial.jsonl')
    series=initial+validation
    series.sort(key=lambda v:v['step'])
    latest=validation[-1];current=updates[-1]
    baseline=json.loads(args.baseline.read_text(encoding='utf-8'))['models']
    start=next(v for v in log if v.get('ae_direct_start'))
    recipe=start['recipe']
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8.2),layout='constrained')
    x=np.array([v['step'] for v in updates])/1000
    window=20
    def smooth(values):
        return np.array([np.mean(values[max(0,i-window+1):i+1]) for i in range(len(values))])
    nll=np.array([v['recon_nll'] for v in updates])
    axes[0,0].plot(x,nll,color='#2667a8',alpha=.2,lw=.7)
    axes[0,0].plot(x,smooth(nll),color='#2667a8',label='重建 NLL（约1000更新平滑）')
    axes[0,0].set(title='训练重建损失',ylabel='NLL')
    axes[0,0].legend()
    colors={'text':'#ad4683','sketch_object':'#ce7937','sketch_scene':'#238673','sketch':'#2667a8'}
    vx=np.array([v['step'] for v in series])/1000
    for name in ('text','sketch_object','sketch_scene'):
        axes[0,1].plot(vx,[v['buckets'][name]['cd']['mean'] for v in series],
                       '.-',color=colors[name],label=LABELS[name])
        axes[0,1].axhline(baseline['old_frozen2']['buckets'][name]['cd']['mean'],
                         color=colors[name],ls='--',lw=.8,alpha=.65)
    axes[0,1].set(title='固定valid笔画CD；虚线为旧冻结AE',ylabel='CD（对数轴）',yscale='log')
    axes[0,1].legend()
    # 排除随机初始化的两点，放大当前训练段；原始全曲线保留在右上图。
    view=validation
    xx=np.array([v['step'] for v in view])/1000
    for field,label,color in (('shape_contribution','形状分量','#ce7937'),
                              ('position_contribution','位置分量','#238673'),('cd','图形总CD','#2667a8')):
        axes[1,0].plot(xx,[v['buckets']['sketch'][field]['mean'] for v in view],'.-',label=label,color=color)
        axes[1,0].axhline(baseline['old_frozen2']['buckets']['sketch'][field]['mean'],
                         color=color,ls='--',lw=.8,alpha=.65)
    axes[1,0].set(title='图形CD及贡献；虚线为旧冻结AE',ylabel='页坐标误差')
    axes[1,0].legend()
    axes[1,1].plot(x,[v['lr'] for v in updates],color='#2667a8',label='实际LR')
    axes[1,1].set(title='实际学习率与DCT损失',ylabel='学习率')
    right=axes[1,1].twinx()
    dct=np.array([v['dct_mse'] for v in updates])
    right.plot(x,smooth(dct),color='#ce7937',label='DCT MSE（平滑）')
    right.set_ylabel('DCT MSE');right.spines['top'].set_visible(False)
    axes[1,1].legend(loc='upper left');right.legend(loc='upper right')
    for ax in axes.flat:
        ax.set_xlabel('成功更新（千次）');ax.grid(alpha=.18)
    timestamp=datetime.fromtimestamp(latest['unix_seconds'],ZoneInfo('Asia/Shanghai'))
    fig.suptitle(f'CoSE新AE：step{current["step"]:,}，最近验证step{latest["step"]:,}（{timestamp:%m-%d %H:%M}）',fontsize=15)
    fig.savefig(root/'curves.png',dpi=150)
    plt.close(fig)
    comparison={}
    for name in LABELS:
        now=latest['buckets'][name]
        a=baseline['old_frozen2']['buckets'][name];b=baseline['adapted6030']['buckets'][name]
        comparison[name]={'current':now,'old_cd':a['cd']['mean'],'ae6030_cd':b['cd']['mean'],
                         'vs_old_percent':(now['cd']['mean']/a['cd']['mean']-1)*100,
                         'vs_6030_percent':(now['cd']['mean']/b['cd']['mean']-1)*100}
    recent=updates[-20:]
    summary={'latest_training':current,'latest_validation_step':latest['step'],
        'validation_time_cst':timestamp.isoformat(),'recipe':recipe,'comparison':comparison,
        'last20_logged_updates':{name:float(np.mean([v[name] for v in recent]))
            for name in ('recon_nll','dct_mse','update_seconds','prepare_seconds')},
        'validation_count':len(validation),'lr_values':sorted(set(v['lr'] for v in updates if v['step']>=recipe['warmup'])),
        'scheduler_implemented':'warmup + ReduceLROnPlateau, not fixed-duration annealing',
        'proposed_annealing':{'start_lr':1e-4,'end_lr':1e-5,'curve':'cosine',
                              'budget':'remaining approved budget','applied':False}}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('latest_validation_step','validation_time_cst',
        'last20_logged_updates','validation_count','lr_values')},ensure_ascii=False))
    for name,value in comparison.items():
        print(LABELS[name],json.dumps({k:value[k] for k in ('vs_old_percent','vs_6030_percent')}),
              'shape/position',value['current']['shape_share'],value['current']['position_share'])


if __name__=='__main__':
    main()
