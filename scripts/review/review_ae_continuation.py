"""2026-10-05：5%文字AE续训的同笔验收、曲线和固定重建对照；只读，不训练。"""
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

from scripts.review.review_ae_pair import BUCKETS, NAMES, comparison, lines


def draw_strokes(path, indices, samples, bundles, title):
    columns=(('source','原笔','#111111'),('old','旧冻结AE','#596773'),
             ('before','5%续训前','#2667a8'),('after','5%续训后','#ce7937'))
    fig,axes=plt.subplots(len(indices),4,figsize=(12,2*len(indices)),layout='constrained')
    for row,index in enumerate(indices):
        source=samples['cloud'][index]
        curves=[source]+[bundles[name]['reconstructed'][index] for name in ('old','before','after')]
        points=np.concatenate(curves);lo=points.min(0);hi=points.max(0)
        center=(hi+lo)/2;size=max(float((hi-lo).max())*1.25,.025)
        for col,(_,label,color) in enumerate(columns):
            name=columns[col][0];ax=axes[row,col]
            if col: ax.plot(source[:,0],source[:,1],color='#cccccc',lw=2.3)
            ax.plot(curves[col][:,0],curves[col][:,1],color=color,lw=1.3)
            ax.set(xlim=(center[0]-size/2,center[0]+size/2),
                   ylim=(center[1]+size/2,center[1]-size/2),aspect='equal',xticks=[],yticks=[])
            error='' if not col else f" CD={bundles[name]['errors'][index,0]:.4f}"
            ax.set_title(label+error,fontsize=9)
            if not col:
                meta=json.loads(str(samples['metadata'][index]))
                ax.set_ylabel(f"{NAMES[str(samples['bucket'][index])]} / #{index}\n{meta.get('label','文字')}",fontsize=9)
    fig.suptitle(title+'；同一行坐标范围相同，灰线为原笔',fontsize=12)
    fig.savefig(path,dpi=145);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir',type=Path,required=True)
    parser.add_argument('--prior_dir',type=Path,required=True)
    parser.add_argument('--baseline_dir',type=Path,required=True)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--gpu_seconds',type=int,required=True,help='sacct实际ElapsedRaw')
    args=parser.parse_args();root=args.run_dir;run=root/'run'
    prior=json.loads((args.prior_dir/'summary.json').read_text(encoding='utf-8'))
    states={name:torch.load(path,map_location='cpu',weights_only=False) for name,path in (
        ('before',args.prior_dir/'text5/final.pt'),('after',run/'final.pt'))}
    before,after=states['before'],states['after']
    for key in ('cfg','train_args','data_manifest','validation_source'):
        assert before[key]==after[key], key
    assert after['validation_source']['sha256']==hashlib.sha256(args.samples.read_bytes()).hexdigest()
    assert all(torch.isfinite(value).all() for value in after['model'].values())
    assert before['scheduler']==after['scheduler']  # 两段调度都暂停plateau，保留旧状态。
    assert all(int(value['step'])==after['step'] for value in after['optimizer']['state'].values())
    assert before['data']['quotas']==after['data']['quotas']
    log=lines(run/'train.log');updates=[v for v in log if 'success' in v]
    start=next(v for v in log if v.get('ae_direct_start'))
    assert start['step']==before['step'] and start['kinds_cumulative']==before['stroke_counts']
    assert start['optimizer_state_entries']==len(before['optimizer']['state'])
    assert start['scale']==before['scaler']['scale']
    validation=lines(run/'validation.jsonl');end=validation[-1]
    assert end['step']==after['step']
    assert np.isclose(after['best_validation_cd'],min(v['monitor_cd'] for v in validation),rtol=0,atol=1e-14)
    duration=after['cosine_annealing']['duration_seconds']
    hold=after['cosine_annealing']['hold_seconds']
    assert after['cosine_annealing']['elapsed_seconds']==duration
    assert after['optimizer']['param_groups'][0]['lr']==after['cosine_annealing']['end_lr']
    stop=json.loads((run/'run_stop.json').read_text(encoding='utf-8'))
    # 与此前口径相同：最终验证写完到停止记录的耗时很短，阶段时间轴标为近似。
    zero=end['unix_seconds']-stop['elapsed_seconds']
    elapsed=lambda value: value['unix_seconds']-zero
    phase={}
    for bucket in BUCKETS:
        points=[prior['fp16_final']['text5']]+validation
        xs=np.array([0.]+[elapsed(v) for v in validation])
        ys=np.array([v['buckets'][bucket]['cd']['mean'] for v in points])
        at_hold=float(np.interp(hold,xs,ys))
        phase[bucket]={'hold_boundary_cd_interpolated':at_hold,
            'hold':comparison(float(ys[0]),at_hold),'cosine':comparison(at_hold,float(ys[-1])),
            'entire':comparison(float(ys[0]),float(ys[-1]))}
        phase[bucket]['cosine_components']={}
        for field in ('shape_contribution','position_contribution'):
            component=np.array([v['buckets'][bucket][field]['mean'] for v in points])
            boundary=float(np.interp(hold,xs,component))
            phase[bucket]['cosine_components'][field]=comparison(boundary,float(component[-1]))
    geometry=json.loads((root/'geometry/report.json').read_text(encoding='utf-8'))['models']['continued']
    reports={'old':prior['historical_baselines']['old_frozen2'],
             'before':prior['fp32_geometry']['text5'],'after':geometry}
    pairs={bucket:{'vs_before':comparison(reports['before']['buckets'][bucket]['cd']['mean'],geometry['buckets'][bucket]['cd']['mean']),
                   'vs_old':comparison(reports['old']['buckets'][bucket]['cd']['mean'],geometry['buckets'][bucket]['cd']['mean'])}
           for bucket in BUCKETS}
    for name,first_elapsed in (('last_approx_hour',duration-3600),('last_two_validations',elapsed(validation[-2]))):
        candidates=[v for v in validation if elapsed(v)>=first_elapsed]
        first=candidates[0]
        phase[name]={'from_step':first['step'],'to_step':end['step'],
                    'buckets':{b:comparison(first['buckets'][b]['cd']['mean'],end['buckets'][b]['cd']['mean']) for b in BUCKETS}}
    increment={k:after['stroke_counts'].get(k,0)-before['stroke_counts'].get(k,0)
               for k in after['stroke_counts']}
    delta=after['step']-before['step'];batch=after['train_args']['batch_size']
    assert increment['sketch']==delta*(after['data']['quotas']['object']+after['data']['quotas']['scene'])
    assert increment['text']==delta*after['data']['quotas']['text']
    audit={'job_id':604433,'gpu_seconds':args.gpu_seconds,'total_text5_gpu_seconds':args.gpu_seconds+prior['audit']['text5']['gpu_seconds'],
           'resume_step':before['step'],'final_step':after['step'],'new_updates':delta,
           'new_successful_strokes':delta*batch,'total_successful_strokes':after['step']*batch,
           'stroke_increment':increment,'stroke_counts':after['stroke_counts'],
           'final_lr':end['lr'],'scale':after['scaler']['scale'],'cosine':after['cosine_annealing'],
           'final_is_lowest_recorded_monitor':end['monitor_cd']==after['best_validation_cd'],
           'same_cfg_recipe_manifest_validation':True,'plateau_unchanged':True,'finite_weights':True,
           'source_commit':after['source_commit'],'validation_source':after['validation_source']}
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(2,3,figsize=(15,8.8),layout='constrained')
    history=[v for v in lines(args.prior_dir/'text5/validation.jsonl') if v['step']>=10000]
    new=[prior['fp16_final']['text5']]+validation
    for ax,bucket in zip(axes[0],BUCKETS[:3]):
        for values,label,color in ((history,'5%前段','#2667a8'),(new,'5%续训','#ce7937')):
            ax.plot(np.array([v['step'] for v in values])/1000,[v['buckets'][bucket]['cd']['mean'] for v in values],'.-',label=label,color=color)
        ax.axhline(reports['old']['buckets'][bucket]['cd']['mean'],color='#596773',ls=':',label='旧冻结AE（FP32）')
        ax.axvline(before['step']/1000,color='#999999',ls='--',lw=.8)
        ax.set(title=NAMES[bucket]+' CD',xlabel='成功更新（千次）',ylabel='CD',yscale='log')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0]
    for field,label,color in (('cd','图形总CD','#2667a8'),('shape_contribution','形状贡献','#ce7937'),('position_contribution','位置贡献','#238673')):
        ax.plot(np.array([v['step'] for v in new])/1000,[v['buckets']['sketch'][field]['mean'] for v in new],'.-',label=label,color=color)
        ax.axhline(reports['old']['buckets']['sketch'][field]['mean'],color=color,ls=':',lw=.8)
    ax.set(title='续训图形误差；虚线为旧AE',xlabel='成功更新（千次）',ylabel='CD贡献')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1]
    ax.plot(np.array([v['step'] for v in updates])/1000,[v['lr'] for v in updates],color='#2667a8')
    ax.set(title='续训实际LR：先恒定再余弦',xlabel='成功更新（千次）',ylabel='LR');ax.grid(alpha=.2)
    ax=axes[1,2]
    def smooth(data):
        return [np.mean(data[max(0,i-19):i+1]) for i in range(len(data))]
    x=np.array([v['step'] for v in updates])/1000
    ax.plot(x,smooth([v['recon_nll'] for v in updates]),color='#2667a8',label='重建NLL')
    right=ax.twinx();right.plot(x,smooth([v['dct_mse'] for v in updates]),color='#ce7937',label='DCT MSE')
    ax.set(title='续训损失（20条日志滑动平均）',xlabel='成功更新（千次）',ylabel='NLL');right.set_ylabel('DCT MSE')
    ax.legend(loc='upper left');right.legend(loc='upper right');ax.grid(alpha=.2)
    fig.suptitle('CoSE 5%文字AE：续训6小时后的重建变化',fontsize=15)
    fig.savefig(root/'curves.png',dpi=155);plt.close(fig)
    samples=dict(np.load(args.samples,allow_pickle=False))
    bundles={name:dict(np.load(path,allow_pickle=False)) for name,path in (
        ('old',args.baseline_dir/'old_frozen2_strokes.npz'),
        ('before',args.prior_dir/'geometry/text5_strokes.npz'),('after',root/'geometry/continued_strokes.npz'))}
    for bundle in bundles.values(): np.testing.assert_array_equal(bundle['cloud'],samples['cloud'])
    chosen=prior['stroke_examples']
    draw_strokes(root/'stroke_comparison.png',chosen,samples,bundles,'固定上一轮六笔，不重新挑样')
    # 另看续训前形状贡献较高的长尾；选择依据始终来自续训前，避免按改善挑图。
    tail=[]
    for bucket in ('sketch_object','sketch_scene'):
        ids=np.flatnonzero(samples['bucket']==bucket)
        ranked=ids[np.argsort(bundles['before']['errors'][ids,1])]
        tail.extend(int(ranked[round(q*(len(ranked)-1))]) for q in (.95,.99))
    draw_strokes(root/'shape_tail.png',tail,samples,bundles,'续训前形状贡献P95/P99笔画')
    draw_strokes(root/'shape_focus.png',tail[:2],samples,bundles,'续训前对象形状贡献P95/P99笔画')
    example_errors={str(index):{name:{'cd':float(bundle['errors'][index,0]),
                                  'shape':float(bundle['errors'][index,1]),'position':float(bundle['errors'][index,2])}
                     for name,bundle in bundles.items()} for index in chosen+tail}
    summary={'audit':audit,'phase_fp16':phase,'comparisons_fp32':pairs,'models_fp32':reports,
             'fp16_final':end,'fixed_examples':chosen,'shape_tail_examples':tail,
             'example_errors':example_errors,'phase_time_note':'hold boundary interpolated between validations; elapsed approximate',
             'fp16_fp32_graphic_delta_percent':100*(geometry['buckets']['sketch']['cd']['mean']/end['buckets']['sketch']['cd']['mean']-1)}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'audit':audit,'comparisons':pairs,'phase':phase,'example_errors':example_errors},ensure_ascii=False))


if __name__=='__main__':
    main()
