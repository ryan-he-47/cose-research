"""R 诊断的配对统计、曲线及中文离线报告。"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch

from src.st.page_diffusion_runtime import write_json

TASKS=('T1','O1','O2','S1','S2','S3','S4','S5')
NAMES=dict(zip(TASKS,('文字续写','按名称画对象','看图命名','场景描述','按描述画场景','场景问答','按名称圈选','圈内命名')))
MODELS=('parent_raw','final_raw','pre_ema','final_ema')


def metric_path(root,model,task,name):
    return root/('metrics_uniform_text' if task=='T1' else 'metrics')/model/name


def resamples(n,labels=None,repeats=1200):
    rng=np.random.default_rng(20261105)
    if labels is not None and all(str(x) for x in labels):
        groups=[np.flatnonzero(labels==label) for label in np.unique(labels)]
        return np.concatenate([g[rng.integers(len(g),size=(repeats,len(g)))] for g in groups],1)
    return rng.integers(n,size=(repeats,n))


def estimate(numerator,indices,denominator=None):
    a=np.asarray(numerator,dtype=np.float64)
    b=np.ones_like(a) if denominator is None else np.asarray(denominator,dtype=np.float64)
    value=float(a.sum()/max(b.sum(),1e-30))
    sims=[]
    for i in range(0,len(indices),100):
        ix=indices[i:i+100]
        sims.extend(a[ix].sum(1)/np.maximum(b[ix].sum(1),1e-30))
    return dict(mean=value,ci95=np.quantile(sims,[.025,.975]).tolist(),n=len(a))


def improvement(old,new,indices):
    return estimate(old-new,indices,old)


def binary_interval(values):
    a=np.asarray(values);n=len(a);p=float(a.mean());z=1.959963984540054
    scale=1+z*z/n;center=(p+z*z/(2*n))/scale
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/scale
    return dict(mean=p,ci95=[max(0.,float(center-radius)),min(1.,float(center+radius))],n=n,method='Wilson')


def integral(a):
    return np.trapezoid(a['full'][:,:,0],a['times'],axis=1)


def denoise_summary(root):
    arrays={name:{t:dict(np.load(metric_path(root,name,t,f'valid_{t}.npz'))) for t in TASKS} for name in MODELS}
    summary={}
    for task in TASKS:
        ref=arrays['final_raw'][task]
        ix=resamples(len(ref['ids']),ref['labels'])
        rows=torch.load(root/('data_uniform_text' if task=='T1' else 'data')/f'valid_{task}.pt',
                        map_location='cpu',weights_only=False,mmap=True)
        properties={}
        for row in rows:
            block=row['blocks'][row['selected']];end=block['start']
            properties[row['id']]=(bool((row['pages'][:end]<block['page']).any()),
                not bool(((row['pages'][:end]==block['page'])&(row['kinds'][:end]<3)).any()))
        subsets={'with_previous_page':np.asarray([properties[i][0] for i in ref['ids']]),
                 'answer_start':np.asarray([properties[i][1] for i in ref['ids']])}
        del rows
        for name in MODELS:
            assert np.array_equal(ref['ids'],arrays[name][task]['ids']), (name,task,'配对ID不一致')
            assert np.array_equal(ref['source'],arrays[name][task]['source']), (name,task,'配对来源不一致')
        task_summary=dict(n=len(ref['ids']),source_units=len(np.unique(ref['source'])),models={})
        for name in MODELS:
            a=arrays[name][task]
            integ=integral(a)
            tail=np.trapezoid(a['full'][:,:,2],a['times'],axis=1)
            pos=np.trapezoid(a['full'][:,:,1],a['times'],axis=1)
            m=dict(integrated_mse=estimate(integ,ix),tail_mse=estimate(tail,ix),position_fraction=estimate(pos,ix,integ),
                mean_mse=a['full'][:,:,0].mean(0).tolist(),mean_tail=a['full'][:,:,2].mean(0).tolist(),
                mean_position=a['full'][:,:,1].mean(0).tolist(),conditions={})
            for mode in ('wrong_condition','previous_pages_removed','none'):
                if mode in a:
                    other=np.trapezoid(a[mode][:,:,0],a['times'],axis=1)
                    m['conditions'][mode]=dict(integrated_increase=estimate(other-integ,ix,integ),
                        high_noise_increase=estimate(a[mode][:,-1,0]-a['full'][:,-1,0],ix,a['full'][:,-1,0]),
                        tail_integrated_increase=estimate(np.trapezoid(a[mode][:,:,2],a['times'],axis=1)-tail,ix,tail),
                        tail_high_noise_increase=estimate(a[mode][:,-1,2]-a['full'][:,-1,2],ix,a['full'][:,-1,2]),
                        mean_mse=a[mode][:,:,0].mean(0).tolist(),mean_tail=a[mode][:,:,2].mean(0).tolist())
            m['subsets']={}
            for label,keep in subsets.items():
                if not keep.any():continue
                subix=resamples(int(keep.sum()),a['labels'][keep])
                m['subsets'][label]={mode:estimate(a[mode][keep,-1,0]-a['full'][keep,-1,0],subix,a['full'][keep,-1,0])
                    for mode in ('wrong_condition','previous_pages_removed','none') if mode in a}
            if len(a['shape_counts']) and a['shape_counts'].sum():
                count=a['shape_counts'];keep=count>0
                shape_ix=resamples(int(keep.sum()),a['labels'][keep])
                m['glyph']=dict(baseline=estimate(a['shape_baseline'][keep],shape_ix,count[keep]),
                    times=a['decode_times'].tolist(),
                    accuracy=[estimate(a['shape_correct'][keep,i],shape_ix,count[keep]) for i in range(a['shape_correct'].shape[1])],
                    strokes=int(count.sum()),documents=int(keep.sum()),represented_shapes=int((a['shape_classes']>0).sum()),
                    class_counts=a['shape_classes'].tolist(),class_correct=a['shape_class_correct'].tolist())
            task_summary['models'][name]=m
        task_summary['continuation_improvement']=improvement(integral(arrays['parent_raw'][task]),integral(ref),ix)
        task_summary['anneal_ema_improvement']=improvement(integral(arrays['pre_ema'][task]),integral(arrays['final_ema'][task]),ix)
        summary[task]=task_summary
        visual={}
        for name in ('parent_visual','final_visual'):
            path=root/'metrics'/name/f'valid_{task}.npz'
            if path.exists():
                a=dict(np.load(path));assert np.array_equal(ref['ids'],a['ids'])
                base=integral(a);without=np.trapezoid(a['visual_removed'][:,:,0],a['times'],axis=1)
                visual[name]=dict(integrated_increase=estimate(without-base,ix,base),
                    high_noise_increase=estimate(a['visual_removed'][:,-1,0]-a['full'][:,-1,0],ix,a['full'][:,-1,0]),
                    tail_high_noise_increase=estimate(a['visual_removed'][:,-1,2]-a['full'][:,-1,2],ix,a['full'][:,-1,2]),
                    mean_full=a['full'][:,:,0].mean(0).tolist(),mean_removed=a['visual_removed'][:,:,0].mean(0).tolist())
                keep=subsets['answer_start']
                if keep.any():
                    visual[name]['answer_start_high_noise_increase']=estimate(
                        a['visual_removed'][keep,-1,0]-a['full'][keep,-1,0],resamples(int(keep.sum()),a['labels'][keep]),a['full'][keep,-1,0])
        task_summary['visual']=visual
    return summary,arrays


def loss_balance(root):
    recipe=json.loads((root/'run/recipe.json').read_text(encoding='utf8'))
    rows=[json.loads(l) for l in (root/'run/train.jsonl').read_text(encoding='utf8').splitlines() if l.strip()][-200:]
    losses,gradients={},{}
    for task in TASKS:
        weight=recipe['loss_weights'][task]
        losses[task]=float(np.mean([weight*r['body_sum:'+task]/r['counts']['body'][task] for r in rows]))
        gradients[task]=float(np.mean([4*weight**2/48*r['body_sum:'+task]/r['counts']['body'][task]**2 for r in rows]))
    return dict(weighted_loss=losses,loss_fraction={k:v/sum(losses.values()) for k,v in losses.items()},
        output_gradient_squared=gradients,output_gradient_fraction={k:v/sum(gradients.values()) for k,v in gradients.items()},
        definition='正文各任务的标量损失，以及对48维x预测输出的平方梯度范数。按末200个真实训练更新精确还原；主干参数梯度需经过网络Jacobian。')


def natural_summary(root):
    result={}
    for model in MODELS:
        a=dict(np.load(root/'metrics'/model/'natural.npz'))
        weights=a['inclusion_weights']*a['counts'][:,0]
        result[model]=dict(documents=len(weights),estimated_body_targets=int(weights.sum()),
            integrated_mse=float(np.average(integral(a),weights=weights)),
            mean_mse=np.average(a['full'][:,:,0],axis=0,weights=weights).tolist())
    return result


def feedback_summary(root,tasks=TASKS):
    from scipy.spatial import cKDTree
    geometry=np.load(root.parent/'ae_latent_geometry_20261103/r_encoded.npz')
    z=geometry['z'];kind=geometry['kind']
    rng=np.random.default_rng(20261105)
    libraries={}
    for name in ('text','sketch'):
        group=z[kind==name]
        ids=rng.choice(len(group),min(16384,len(group)),replace=False)
        lib=group[ids].astype(np.float64)
        _,basis=np.linalg.eigh(np.cov(group.astype(np.float64),rowvar=False))
        tail=basis[:,:-2]
        libraries[name]=(cKDTree(lib),cKDTree(lib@tail),tail)
    summary={}
    for model in ('parent_raw','final_raw'):
        summary[model]={}
        for task in tasks:
            path=metric_path(root,model,task,f'rollout_valid_{task}.pt')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            rows=saved['results']
            if task not in ('T1','S1'):
                summary[model][task]=dict(n=len(rows),eligible=saved['eligible'],scope='正文通常不足64笔；保存覆盖记录，长程统计使用T1和S1。');continue
            if not rows:
                summary[model][task]=dict(n=0,eligible=saved['eligible']);continue
            n=len(rows);ix=resamples(n)
            out=dict(n=n,eligible=saved['eligible'],times=saved['times'].tolist(),modes={})
            controls=np.stack([[h['generated_control'].numpy() for h in r['horizons']] for r in rows])
            block_controls=controls.any(-1)
            alive=np.ones((n,4),bool)
            for h in range(1,4):alive[:,h]=~block_controls[:,:h].any(-1)
            out['generated_control_fraction']=float(controls.mean())
            out['alive_documents']=alive.sum(0).tolist()
            for mode in ('truth','generated','reconstructed'):
                values=np.asarray([[h[mode] for h in r['horizons']] for r in rows])
                out['modes'][mode]=dict(mean=values.mean(0).tolist(),by_horizon=[])
                truth=np.asarray([[h['truth'] for h in r['horizons']] for r in rows])
                for h in range(4):
                    out['modes'][mode]['by_horizon'].append(dict(
                        at_noise_half_increase=estimate(values[:,h,2,0]-truth[:,h,2,0],ix,truth[:,h,2,0]),
                        at_noise_one_increase=estimate(values[:,h,3,0]-truth[:,h,3,0],ix,truth[:,h,3,0])))
                out['modes'][mode]['alive_by_horizon']=[]
                for h in range(4):
                    keep=alive[:,h]
                    out['modes'][mode]['alive_by_horizon'].append(
                        estimate(values[keep,h,2,0]-truth[keep,h,2,0],resamples(int(keep.sum())),truth[keep,h,2,0]) if keep.any() else None)
            library=libraries['sketch' if task in ('O1','S2') else 'text']
            generated=np.stack([[h['generated_latent'].numpy() for h in r['horizons']] for r in rows])
            target=np.stack([[h['target_latent'].numpy() for h in r['horizons']] for r in rows])
            out['neighbors']={}
            for label,tree,axes in (('full',library[0],None),('tail',library[1],library[2])):
                g=generated.reshape(-1,48);t=target.reshape(-1,48)
                if axes is not None:g,t=g@axes,t@axes
                gd=tree.query(g,workers=4)[0].reshape(n,4,16)
                td=tree.query(t,workers=4)[0].reshape(n,4,16)
                # 只看正常正文历史仍存活的块，并排除其中识别出的控制笔，避免把合法EOP记作流形异常。
                keep=(np.cumsum(controls,axis=2)==0)&alive[:,:,None]
                threshold=float(np.quantile(td[keep],.99))
                out['neighbors'][label]=dict(target_median=float(np.median(td[keep])),generated_median=float(np.median(gd[keep])),
                    median_ratio=float(np.median(gd[keep])/max(np.median(td[keep]),1e-15)),target_p99=threshold,
                    generated_above_target_p99=float((gd[keep]>threshold).mean()),body_strokes=int(keep.sum()),
                    by_horizon_ratio=[float(np.median(gd[:,h][keep[:,h]])/max(np.median(td[:,h][keep[:,h]]),1e-15)) if keep[:,h].any() else None for h in range(4)])
                count=keep.sum((1,2));usable=count>0;boot=resamples(int(usable.sum()))
                out['neighbors'][label]['mean_distance_ratio']=estimate(
                    ((gd*keep).sum((1,2))/np.maximum(count,1))[usable],boot,
                    ((td*keep).sum((1,2))/np.maximum(count,1))[usable])
                out['neighbors'][label]['outlier_rate']=estimate(((gd>threshold)&keep).sum((1,2))[usable],boot,count[usable])
            summary[model][task]=out
    return summary


def roundtrip_summary(root):
    result={}
    for task in ('T1','S1'):
        raw=torch.load(metric_path(root,'final_raw',task,f'rollout_valid_{task}.pt'),map_location='cpu',weights_only=False)
        changed=torch.load(metric_path(root,'final_raw',task,f'roundtrip_valid_{task}.pt'),map_location='cpu',weights_only=False)
        assert [(r['id'],r['source']) for r in raw['results']]==[(r['id'],r['source']) for r in changed['results']]
        controls=np.stack([[h['generated_control'].numpy().any() for h in r['horizons']] for r in raw['results']])
        alive=np.concatenate((np.ones((len(controls),1),bool),np.cumsum(controls[:,:-1],axis=1)==0),axis=1)
        item=dict(n=len(controls),by_horizon=[])
        for horizon in range(4):
            keep=alive[:,horizon];ix=resamples(int(keep.sum()))
            truth=np.asarray([r['horizons'][horizon]['truth'][2,0] for r in raw['results']])[keep]
            generated=np.asarray([r['horizons'][horizon]['generated'][2,0] for r in raw['results']])[keep]
            a=np.asarray([r['horizons'][horizon]['generated_roundtrip'][2,0] for r in changed['results']])[keep]
            b=np.asarray([r['horizons'][horizon]['target_roundtrip'][2,0] for r in changed['results']])[keep]
            item['by_horizon'].append(dict(
                generated_increase=estimate(generated-truth,ix,truth),
                generated_roundtrip_increase=estimate(a-truth,ix,truth),
                target_roundtrip_increase=estimate(b-truth,ix,truth),
                generated_error_reduction=estimate(generated-a,ix,generated),
                excess_error_repaired=estimate(generated-a,ix,generated-truth) if horizon else None))
        ix=resamples(len(controls))
        for key in ('generated_roundtrip_latent_change','target_roundtrip_latent_change',
                    'generated_roundtrip_curve_change','target_roundtrip_curve_change'):
            values=np.asarray([[h[key] for h in r['horizons']] for r in changed['results']])
            item[key]=estimate(values.mean(1),ix)
        result[task]=item
    return result


def box_metrics(curve,metadata):
    curve=np.asarray(curve)
    box=np.r_[curve.min(0),curve.max(0)]
    center=(box[:2]+box[2:])/2
    objects=metadata['objects']
    nearest=objects[int(np.argmin([np.linalg.norm(center-np.asarray(o['center'])) for o in objects]))]
    target=np.asarray(metadata['selection_box'])
    wh=np.maximum(0,np.minimum(box[2:],target[2:])-np.maximum(box[:2],target[:2]))
    intersection=float(np.prod(wh))
    union=float(np.prod(box[2:]-box[:2])+np.prod(target[2:]-target[:2])-intersection)
    return float(nearest['instance_id']==metadata['target_instance_id']),float(np.linalg.norm(center-(target[:2]+target[2:])/2)),intersection/max(union,1e-12)


def sampling_summary(root):
    metadata=json.loads((root/'scene_metadata.json').read_text(encoding='utf8'))
    output={};s4_pairs={};s4_ids={}
    for model in ('parent_raw','final_raw'):
        output[model]={}
        paths=sorted((root/'metrics'/model).glob('sampling_valid_*.pt'))
        for original in paths:
            path=metric_path(root,model,original.stem.split('_')[-1],original.name)
            task=path.stem.split('_')[-1]
            rows=torch.load(path,map_location='cpu',weights_only=False)
            ix=resamples(len(rows))
            if task=='S4':s4_ids[model]=[r['id'] for r in rows]
            result=dict(n=len(rows),methods={})
            for method in ('16','32','64','32_quadratic'):
                reference=[];convergence=[];shape_convergence=[];scores=[]
                for r in rows:
                    mask=r['body'].numpy().astype(bool)
                    p=r['outputs'][method]['curves'].numpy()[mask]
                    q=r['reference_curves'].numpy()[mask]
                    baseline=r['outputs']['64']['curves'].numpy()[mask]
                    d=p-q;reference.append(float(np.sqrt((d*d).sum(-1).mean())))
                    d=p-baseline;convergence.append(float(np.sqrt((d*d).sum(-1).mean())))
                    d-=d.mean(-2,keepdims=True);shape_convergence.append(float(np.sqrt((d*d).sum(-1).mean())))
                    if task=='S4':scores.append(box_metrics(p[0],metadata[r['source']]))
                item=dict(reference_curve_rms=estimate(reference,ix),distance_to_64=estimate(convergence,ix),
                    shape_distance_to_64=estimate(shape_convergence,ix))
                if scores:
                    scores=np.asarray(scores)
                    item.update(accuracy=binary_interval(scores[:,0]),iou_above_half=binary_interval(scores[:,2]>=.5),
                                center_error=estimate(scores[:,1],ix),iou=estimate(scores[:,2],ix))
                    s4_pairs[model,method]=scores[:,0]
                result['methods'][method]=item
            updates=np.asarray([[float(s['update_rms']) for s in r['trajectory']] for r in rows])
            pred=np.stack([[s['prediction'].numpy() for s in r['trajectory']] for r in rows])
            masks=np.stack([r['body'].numpy() for r in rows])
            deltas=np.sqrt((np.square(np.diff(pred,axis=1)).mean(-1)*masks[:,None]).sum(-1)/masks.sum(-1)[:,None])
            result['trajectory']=dict(times=[s['t'] for s in rows[0]['trajectory']],
                update_median=np.median(updates,axis=0).tolist(),update_p95=np.quantile(updates,.95,axis=0).tolist(),
                prediction_change_median=np.median(deltas,axis=0).tolist())
            if task=='S4':
                ae=[box_metrics(r['reference_curves'].numpy()[r['body'].numpy().astype(bool)][0],metadata[r['source']])[0] for r in rows]
                result['ae_reference_accuracy']=binary_interval(ae)
                result['random_object_baseline']=float(np.mean([1/len(metadata[r['source']]['objects']) for r in rows]))
                center_guess=[]
                for r in rows:
                    meta=metadata[r['source']]
                    guessed=min(meta['objects'],key=lambda o: np.linalg.norm(np.asarray(o['center'])-np.array([.5,.5])))
                    center_guess.append(float(guessed['instance_id']==meta['target_instance_id']))
                result['nearest_page_center_baseline']=binary_interval(center_guess)
            output[model][task]=result
    assert s4_ids['parent_raw']==s4_ids['final_raw'],'S4配对来源顺序不同'
    ix=resamples(len(s4_pairs['final_raw','32']))
    output['s4_paired']=dict(continuation_accuracy_difference=estimate(s4_pairs['final_raw','32']-s4_pairs['parent_raw','32'],ix),
        sampler_differences={m:estimate(s4_pairs['final_raw',m]-s4_pairs['final_raw','32'],ix) for m in ('16','64','32_quadratic')})
    return output


def figures(root,summary,arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    def save(fig,name):
        fig.savefig(root/name,dpi=165,bbox_inches='tight');plt.close(fig)
    times=arrays['final_raw']['T1']['times']
    colors={'parent_raw':'#858b98','final_raw':'#087f8c','pre_ema':'#c07a39','final_ema':'#8352a0'}
    fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
    for task,ax in zip(TASKS,axes.flat):
        for model in ('parent_raw','final_raw'):
            y=summary['denoising'][task]['models'][model]['mean_mse']
            ax.plot(times[1:],y[1:],label='3.6亿' if model=='parent_raw' else'13.6亿',color=colors[model])
        ax.set(xscale='log',yscale='log',title=f'{task} {NAMES[task]}',xlabel='噪声比例 t',ylabel='正文潜变量 MSE')
        ax.axvline(1/32,color='#ccc',ls=':');ax.grid(alpha=.2)
    axes.flat[0].legend();save(fig,'denoising_curves.png')
    fig,axes=plt.subplots(2,4,figsize=(17,8),constrained_layout=True)
    for task,ax in zip(TASKS,axes.flat):
        m=summary['denoising'][task]['models']['final_raw']
        ax.plot(times,m['mean_mse'],label='真实历史',color=colors['final_raw'])
        for key,label,c in (('wrong_condition','置换条件','#be695e'),('previous_pages_removed','移除前页','#a88c35'),('none','移除全部历史','#8d93a1')):
            ax.plot(times,m['conditions'][key]['mean_mse'],label=label,color=c)
        ax.set(title=f'{task} {NAMES[task]}',xlabel='噪声比例 t',ylabel='正文 MSE');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8);save(fig,'condition_curves.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
    visual_tasks=['O2','S1','S3','S4','S5']
    for name,offset,c,label in (('parent_visual',-.16,'#858b98','3.6亿'),('final_visual',.16,'#087f8c','13.6亿')):
        vals=[summary['denoising'][t]['visual'][name]['high_noise_increase'] for t in visual_tasks]
        y=np.array([v['mean']*100 for v in vals]);ci=np.array([v['ci95'] for v in vals])*100
        axes[0].bar(np.arange(5)+offset,y,width=.3,label=label,color=c,yerr=[y-ci[:,0],ci[:,1]-y],capsize=3)
    axes[0].set(xticks=np.arange(5),xticklabels=visual_tasks,ylabel='去掉视觉笔画后的误差增加（%）',title='保留文字问题：t=1时的图形条件收益');axes[0].legend()
    for key,offset,c,label in (('loss_fraction',-.16,'#087f8c','正文损失'),('output_gradient_fraction',.16,'#c07a39','输出端平方梯度')):
        axes[1].bar(np.arange(8)+offset,[summary['loss_balance'][key][t]*100 for t in TASKS],width=.3,color=c,label=label)
    axes[1].set(xticks=np.arange(8),xticklabels=TASKS,ylabel='份额（%）',title='当前训练目标的实际重心');axes[1].legend()
    save(fig,'visual_and_loss_balance.png')
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for task,ax in zip(('T1','O2','S1','S3','S5'),axes.flat):
        for model in ('parent_raw','final_raw'):
            g=summary['denoising'][task]['models'][model]['glyph']
            y=np.array([x['mean'] for x in g['accuracy']]);ci=np.array([x['ci95'] for x in g['accuracy']])
            ax.plot(g['times'],100*y,color=colors[model],label='3.6亿' if model=='parent_raw' else'13.6亿')
            ax.fill_between(g['times'],ci[:,0]*100,ci[:,1]*100,color=colors[model],alpha=.15)
        ax.axhline(g['baseline']['mean']*100,color='#999',ls=':',label='干净AE基线')
        ax.set(xscale='log',ylim=(0,100),title=f'{task} {NAMES[task]}',xlabel='噪声比例 t',ylabel='逐笔字形辨认率（%）');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8);axes.flat[-1].axis('off');save(fig,'glyph_recovery.png')
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),constrained_layout=True)
    for task in ('T1','O1','O2','S1','S2','S4'):
        a=arrays['final_raw'][task]
        y=a['full'][:,:,0].mean(0)
        axes[0].plot(times[1:],y[1:]/times[1:]**2,label=task)
        axes[1].plot(a['decode_times'],a['curves'][:,:,2].mean(0),label=task)
        m=summary['sampling']['final_raw'][task]['trajectory']
        axes[2].plot(m['times'],m['update_median'],label=task)
    axes[0].set(xscale='log',yscale='log',xlabel='噪声比例 t',ylabel='x误差 / t²',title='由 x 误差折算的速度误差')
    axes[1].set(xscale='log',yscale='log',xlabel='噪声比例 t',ylabel='去掉平移后的曲线 RMS',title='实际笔形误差')
    axes[2].set(xlabel='噪声比例 t',ylabel='正文每步潜变量更新 RMS',title='32步生成：实际更新幅度');axes[2].invert_xaxis()
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'low_noise_and_updates.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for mi,model in enumerate(('parent_raw','final_raw')):
        for task in ('T1','S1'):
            data=summary['feedback'][model].get(task,{})
            if not data.get('n'):continue
            y=[x['at_noise_half_increase']['mean']*100 for x in data['modes']['generated']['by_horizon']]
            axes[mi].plot(range(1,5),y,marker='o',label=f'{task}, n={data["n"]}')
        axes[mi].set(title='3.6亿' if mi==0 else'13.6亿',xlabel='后续块序号',ylabel='生成历史使下一块去噪 MSE 增加（%）',xticks=[1,2,3,4]);axes[mi].legend();axes[mi].grid(alpha=.2)
    save(fig,'history_feedback.png')
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    for task,ax in zip(('T1','S1'),axes):
        rows=summary['roundtrip'][task]['by_horizon']
        for key,label,color in (('generated_increase','原生成历史','#be695e'),
                                ('generated_roundtrip_increase','同一生成曲线D→E','#087f8c'),
                                ('target_roundtrip_increase','真实目标D→E','#858b98')):
            y=np.asarray([r[key]['mean'] for r in rows])*100
            ci=np.asarray([r[key]['ci95'] for r in rows])*100
            ax.plot(range(1,5),y,marker='o',color=color,label=label)
            ax.fill_between(range(1,5),ci[:,0],ci[:,1],color=color,alpha=.12)
        ax.set(title=f'{task} {NAMES[task]}',xlabel='后续块序号',xticks=[1,2,3,4],
               ylabel='相对真实历史的去噪MSE增量（%）')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    save(fig,'feedback_roundtrip.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    labels=['16步','32步','64步','32步末段加密'];keys=['16','32','64','32_quadratic']
    for model,offset in (('parent_raw',-.17),('final_raw',.17)):
        values=[summary['sampling'][model]['S4']['methods'][k]['accuracy'] for k in keys]
        y=np.array([v['mean'] for v in values])*100;ci=np.array([v['ci95'] for v in values])*100
        axes[0].bar(np.arange(4)+offset,y,width=.32,color=colors[model],label='3.6亿' if model=='parent_raw' else'13.6亿',yerr=[y-ci[:,0],ci[:,1]-y],capsize=3)
    axes[0].axhline(summary['sampling']['final_raw']['S4']['random_object_baseline']*100,ls=':',color='#888',label='均匀选对象')
    axes[0].set(xticks=np.arange(4),xticklabels=labels,ylabel='首笔中心归属正确对象（%）',title='512个配对验证场景');axes[0].legend(fontsize=8)
    tasks=list(TASKS)
    for method,offset,c in (('16',-.22,'#c07a39'),('32',0.,'#087f8c'),('32_quadratic',.22,'#8352a0')):
        values=[summary['sampling']['final_raw'][t]['methods'][method]['distance_to_64']['mean'] for t in tasks]
        axes[1].bar(np.arange(8)+offset,values,width=.21,label=method,color=c)
    axes[1].set(xticks=np.arange(8),xticklabels=tasks,ylabel='与同噪声64步输出的曲线 RMS',title='采样轨迹随步数的变化');axes[1].legend(fontsize=8)
    save(fig,'sampler_comparison.png')
    # 连续训练曲线直接来自两个完整日志。
    paths=[Path('E:/Trash_things/cose/tmp_smoke/r_page_diffusion_final_20261103/run/train.jsonl'),root/'run/train.jsonl']
    logs=[json.loads(line) for path in paths for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    x=np.array([r['successful_inputs']/1e8 for r in logs])
    def smooth(v,n=200):
        a=np.asarray(v,float);cs=np.r_[0,np.cumsum(a)];i=np.arange(len(a));start=np.maximum(0,i-n+1)
        return (cs[i+1]-cs[start])/(i-start+1)
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    axes[0,0].plot(x,smooth([r['loss'] for r in logs]),color='#087f8c');axes[0,0].set_title('总损失（200次更新滑动平均）')
    for task in TASKS:
        y=[r.get('body_sum:'+task,0)/max(1,r['counts']['body'].get(task,0)) for r in logs]
        axes[0,1].plot(x,smooth(y),label=task)
    axes[0,1].set_title('任务正文 MSE');axes[0,1].legend(ncol=4,fontsize=8)
    for t in (0.,.25,.5,.75):
        y=[r.get(f'noise_{t}_sum',0)/max(1,r.get(f'noise_{t}_count',0)) for r in logs]
        axes[1,0].plot(x,smooth(y),label=f'{t:g}–{t+.25:g}')
    axes[1,0].set(yscale='log',title='各噪声档正文 MSE');axes[1,0].legend(fontsize=8)
    axes[1,1].plot(x,[r['lr'] for r in logs],color='#c07a39');axes[1,1].set_title('实际学习率：重新升温、稳定、退火')
    for ax in axes.flat:
        ax.axvline(3.6,color='#bbb',ls='--');ax.axvspan(12.88,13.6,color='#bd8c4d',alpha=.13)
        ax.set_xlabel('累计成功输入（亿）');ax.grid(alpha=.2)
    save(fig,'training_curves.png')
    continuation=[r for r in logs if r['successful_inputs']>360000013]
    cx=np.asarray([r['successful_inputs'] for r in continuation])/1e8
    baseline=summary['training']['windows']['parent_final200']['task_mse'] if 'parent_final200' in summary['training']['windows'] else None
    # 单独放大续训阶段，保留原始完整曲线以核对升温和两次退火。
    fig,axes=plt.subplots(1,2,figsize=(14,4.8),constrained_layout=True)
    axes[0].plot(cx,smooth([r['loss'] for r in continuation]),color='#087f8c')
    axes[0].set_title('追加10亿：总损失局部放大')
    if baseline is None:
        parent=[r for r in logs if r['successful_inputs']<=360000013][-200:]
        baseline={task:sum(r['body_sum:'+task] for r in parent)/sum(r['counts']['body'][task] for r in parent) for task in TASKS}
    for task in TASKS:
        y=[r['body_sum:'+task]/max(1,r['counts']['body'][task])/baseline[task]*100 for r in continuation]
        axes[1].plot(cx,smooth(y),label=task)
    axes[1].set_title('各任务正文误差：原3.6亿末段=100%')
    axes[1].axhline(100,color='#999',ls=':');axes[1].set_ylabel('相对误差（%）');axes[1].legend(ncol=4,fontsize=8)
    for ax in axes:
        ax.axvspan(12.88,13.6,color='#bd8c4d',alpha=.13);ax.set_xlabel('累计成功输入（亿）');ax.grid(alpha=.2)
    save(fig,'training_continuation_zoom.png')


def html_report(root,summary):
    rows=[];visual_rows=[]
    def pct(x):
        return f'{100*x["mean"]:.1f}%（{100*x["ci95"][0]:.1f}～{100*x["ci95"][1]:.1f}）'
    for task in TASKS:
        d=summary['denoising'][task]
        rows.append(f'<tr><td>{task} {NAMES[task]}</td><td>{d["n"]:,}</td><td>{pct(d["continuation_improvement"])}</td><td>{pct(d["anneal_ema_improvement"])}</td></tr>')
        if d['visual']:
            visual_rows.append(f'<tr><td>{task} {NAMES[task]}</td><td>{pct(d["visual"]["parent_visual"]["high_noise_increase"])}</td><td>{pct(d["visual"]["final_visual"]["high_noise_increase"])}</td><td>{pct(d["models"]["final_raw"]["conditions"]["wrong_condition"]["high_noise_increase"])}</td></tr>')
    notes=(root/'observations.html').read_text(encoding='utf8') if (root/'observations.html').exists() else ''
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>追加10亿后的R诊断</title>
<style>body{font:17px/1.8 system-ui,"Microsoft YaHei";color:#192b3c;background:#f3f6f8;margin:0}main{max-width:1160px;margin:auto;padding:36px 24px}h1,h2{line-height:1.4}section{background:white;padding:25px 30px;border-radius:12px;margin:24px 0}img{width:100%;border-radius:6px}table{width:100%;border-collapse:collapse;font-size:15px}td,th{padding:10px;border-bottom:1px solid #dde3e7;text-align:left}a{color:#096b84}details{margin:16px 0}.note{color:#536574;font-size:15px}.lead{font-size:20px}</style><main>
<h1>追加10亿输入后：R到底学会了什么，卡在哪里</h1><p>2026-11-05 · 最终20,713步 / 13.60000017亿输入 · 指标附95%区间；变化按来源配对重采样，圈选准确率使用Wilson区间</p>'''+notes+'''
<section><h2>先看训练与配对验证</h2><p>本次10,484份验证任务文档。对象每类10个来源，场景512个，文字1024篇；场景的五种任务共享来源并分项统计。对象区间保留每类的抽样份额。另有991份真实训练流窗口作为自然配比参照。</p>
<table><tr><th>任务</th><th>文档数</th><th>加训后误差降低</th><th>本轮退火期间EMA误差降低</th></tr>'''+''.join(rows)+'''</table><p class="note">使用相同验证块、相同噪声，在t=0至1上积分。</p><img src="training_continuation_zoom.png"><details><summary>从零开始的完整训练曲线</summary><img src="training_curves.png"></details><img src="denoising_curves.png"></section>
<section><h2>条件利用：看误差增加在哪些任务、哪些噪声档</h2><table><tr><th>任务</th><th>3.6亿：移除视觉</th><th>13.6亿：移除视觉</th><th>13.6亿：换整份前页条件</th></tr>'''+''.join(visual_rows)+'''</table><p class="note">表内是t=1时正文MSE的增幅。移除视觉保留文字问题、本页前缀、控制符与原位置编号；换前页条件会同时更换其中的问题和图形。全部噪声档见下图。</p><img src="condition_curves.png"><img src="visual_and_loss_balance.png"><p>右图来自末200个实际训练更新；输出端梯度按现有损失公式精确计算。主干参数梯度还取决于网络Jacobian。</p><p>原始结果同时保留前两主方向和其余46方向的误差。前两主方向与位置的关系见上一轮AE几何报告。</p></section>
<section><h2>笔形在去噪的哪一段恢复</h2><p>用上一轮113类合并笔形参考库，去掉位置和页面字号后做逐笔近邻辨认。每份文档的全部目标笔画一起重采样；虚线为同一批真实潜变量经AE解码后的辨认率。</p><img src="glyph_recovery.png"><img src="low_noise_and_updates.png"></section>
<section><h2>生成历史如何影响后续块</h2><p>在有四个完整正文块的页面上推进64笔，每块都测相同真实目标的去噪误差。另保留参考附近去噪历史和训练分布近邻距离，用于区分偏离参考与离开常见笔形区域。</p><img src="history_feedback.png"><p>下面固定已有生成曲线，经同一D→E编码后回馈；真实目标也做相同往返作为对照。只统计此前尚未生成结束符的轨迹，避免把结束后的槽位当成正文。</p><img src="feedback_roundtrip.png"></section>
<section><h2>更多采样步数和加密末段的收益</h2><p>四种采样共用初始噪声。S4使用全部512个验证场景；其他任务每项128个来源。末段加密使用t=(j/32)²。圈选按第一条输出笔画的包围盒中心归属最近对象计算，并同时记录中心误差和IoU。</p><img src="sampler_comparison.png"></section>
<section><h2>原始生成与复现</h2><p><a href="panel/report.html">固定来源的完整自由生成样张</a> · <a href="summary.json">所有统计与区间</a> · <a href="training_audit.json">训练核验</a> · <a href="selection.json">验证来源</a></p>
<details><summary>实际公式和文献</summary><p>本项目t=1为噪声，z_t=(1-t)x+tε；L_v=L_x/t²。生产更新为z_next=(t_next/t)z_t+(1-t_next/t)x_hat。最后一步直接输出x预测。</p><p>JiT最终采用x输出和v损失：<a href="https://arxiv.org/html/2511.13720v1">Back to Basics，3.2、4.2、4.3节</a>。历史噪声的相关方法：<a href="https://arxiv.org/html/2407.01392v2">Diffusion Forcing</a>。</p></details></section></main></html>'''
    (root/'report.html').write_text(page,encoding='utf8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--render_only',action='store_true');a=p.parse_args();torch.set_num_threads(4)
    if a.render_only:
        summary=json.loads((a.out/'summary.json').read_text(encoding='utf8'));html_report(a.out,summary);return
    denoising,arrays=denoise_summary(a.out)
    summary=dict(denoising=denoising,feedback=feedback_summary(a.out),roundtrip=roundtrip_summary(a.out),sampling=sampling_summary(a.out),
                 loss_balance=loss_balance(a.out),natural=natural_summary(a.out),
                 training=json.loads((a.out/'training_audit.json').read_text(encoding='utf8')))
    write_json(a.out/'summary.json',summary)
    figures(a.out,summary,arrays);html_report(a.out,summary)
    print(json.dumps(dict(stage='report_complete',tasks=len(denoising))))


if __name__=='__main__':main()
