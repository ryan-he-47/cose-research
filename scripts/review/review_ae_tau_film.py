"""2026-10-07：FiLM完整AE验收，沿用同笔几何/聚类和固定角点样张；不训练。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.st.ae_schedule import StepLRReplay
from src.st.ae_geometry import arc_resample, error_components
from src.st.train_ae import StrokeAutoencoder, config_from_checkpoint, load_ae_weights
from src.st.sampling import gmm_density_mode
from src.st.pretrain_model import FrozenSketchLM, config_from_ae_checkpoint
from scripts.review.review_ae_pair import BUCKETS, NAMES, comparison, lines, at_step
import review_ae_lambda48_continuation as layout

layout.LABELS = {'small':'λ16末步', 'before':'λ48原τ拼接', 'after':'λ48正弦τ/FiLM'}
layout.COLORS = {'small':'#9a7ab3', 'before':'#2667a8', 'after':'#ce7937'}


def decode_check(root, samples, indices, baseline, device):
    """固定难笔用256τ重推理，检查48τ绘图及GMM分量选择；不改变主表口径。"""
    fig, axes = plt.subplots(2*len(indices), 4, figsize=(12, 3.6*len(indices)), layout='constrained')
    result = {}
    for arm, checkpoint in (('before',baseline), ('after',root/'run/final.pt')):
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = StrokeAutoencoder(config_from_checkpoint(state)).eval().to(device)
        load_ae_weights(model, state)
        result[arm] = {}
        with torch.no_grad():
            ink = torch.from_numpy(samples['ink'][indices]).to(device)
            lengths = torch.from_numpy(samples['lengths'][indices]).to(device)
            embedding = model.encoder(ink, lengths)
            curves = {}
            for mode, count in (('pi48',48),('pi256',256),('density256',256)):
                tau = torch.linspace(0,1,count,device=device).expand(len(indices),-1)
                mu, sigma, pi = model.decoder(embedding,tau)
                selected = (gmm_density_mode(mu,sigma,pi) if mode=='density256' else
                            mu.gather(2,pi.argmax(-1)[...,None,None].expand(-1,-1,1,2)).squeeze(2))
                raw = (selected*.6+selected.new_tensor([.5,.6])).cpu().numpy()
                curves[mode] = raw
                result[arm][mode] = {str(index): {'cd':float(error_components(samples['cloud'][index],arc_resample(raw[i],32))[0]),
                    'centered_shape_cd':float(error_components(samples['cloud'][index],arc_resample(raw[i],32))[3])}
                    for i,index in enumerate(indices)}
        for i,index in enumerate(indices):
            row = 2*i+(arm=='after')
            source = samples['cloud'][index]
            values = [source]+[curves[mode][i]-arc_resample(curves[mode][i],32).mean(0)+source.mean(0)
                               for mode in ('pi48','pi256','density256')]
            all_points = np.concatenate(values); lo,hi=all_points.min(0),all_points.max(0)
            center=(lo+hi)/2;size=max(float((hi-lo).max())*1.25,.025)
            for col,(title,curve) in enumerate(zip(('原笔','48τ / 最高π','256τ / 最高π','256τ / 密度众数'),values)):
                ax=axes[row,col]
                if col: ax.plot(source[:,0],source[:,1],color='#cccccc',lw=2.3)
                ax.plot(curve[:,0],curve[:,1],color='#111111' if not col else layout.COLORS[arm],lw=1.3)
                ax.set(xlim=(center[0]-size/2,center[0]+size/2),ylim=(center[1]+size/2,center[1]-size/2),aspect='equal',xticks=[],yticks=[])
                ax.set_title(title,fontsize=9)
                if not col: ax.set_ylabel(f'{layout.LABELS[arm]} / #{index}',fontsize=9)
        del model,state
        gc.collect()
        if device=='cuda': torch.cuda.empty_cache()
    fig.suptitle('固定难笔的解码检查：去质心对齐，灰线为原笔；不训练',fontsize=12)
    layout.save_plot(fig,root/'decode_check.png');plt.close(fig)
    report={'indices':indices,'precision':'float32','metrics':result,'tau_counts':[48,256],'density_mode_iterations':64}
    (root/'decode_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir',type=Path,required=True)
    parser.add_argument('--concat_dir',type=Path,required=True)
    parser.add_argument('--concat_first_dir',type=Path,required=True)
    parser.add_argument('--lambda16_dir',type=Path,required=True)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--gpu_seconds',type=int,required=True)
    parser.add_argument('--decode_check',action='store_true')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args();torch.set_num_threads(4)
    root=args.run_dir
    prior=json.loads((args.concat_dir/'summary.json').read_text(encoding='utf-8'))
    small=json.loads((args.lambda16_dir/'summary.json').read_text(encoding='utf-8'))
    baseline=torch.load(args.concat_dir/'run/final.pt',map_location='cpu',weights_only=False)
    state=torch.load(root/'run/final.pt',map_location='cpu',weights_only=False)
    cfg_without_tau=lambda cfg:{key:value for key,value in cfg.items() if not key.startswith('dec_tau_')}
    assert cfg_without_tau(state['cfg'])==cfg_without_tau(baseline['cfg'])
    a=dict(state['train_args']);b=dict(baseline['train_args'])
    a.pop('experiment');b.pop('experiment')
    a['ae_config']=cfg_without_tau(a['ae_config']);b['ae_config']=cfg_without_tau(b['ae_config'])
    assert a==b
    assert state['cfg']['dec_tau_mode']=='film' and state['cfg']['dec_tau_pe_dim']==64 and state['cfg']['dec_tau_max_cycles']==32
    assert state['step']==baseline['step']==48192
    for key in ('data_args','data_manifest','validation_source','scheduler'):
        assert state[key]==baseline[key],key
    assert state['data']==baseline['data']
    assert state['validation_source']['sha256']==hashlib.sha256(args.samples.read_bytes()).hexdigest()
    assert all(torch.isfinite(value).all() for value in state['model'].values())
    assert len(state['optimizer']['state'])==124
    assert all(int(value['step'])==48192 for value in state['optimizer']['state'].values())
    assert state['source_commit']=='77ed20d040f78dfde6e487962df42a3fb7977d40'
    updates=[record for record in lines(root/'run/train.log') if 'success' in record]
    start=next(record for record in lines(root/'run/train.log') if record.get('ae_direct_start'))
    assert start['step']==0 and start['optimizer_state_entries']==0 and start['scale']==256 and start['resume_from'] is None
    assert start['parameters']==6521446 and start['target_successful_steps']==48192
    profile=json.loads((root/'run/lr_replay_requested.json').read_text(encoding='utf-8'))
    assert state['step_lr_schedule']==profile==start['lr_schedule']
    replay=StepLRReplay(profile)
    assert all(record['lr']==replay.lr(record['lr_replay_step']) for record in updates)
    assert all(record['lr_replay_step']==record['step']+(not record['success']) for record in updates)
    assert all(np.isfinite(record[field]) for record in updates for field in ('loss','recon_nll','dct_mse'))
    assert state['optimizer']['param_groups'][0]['lr']==5e-6 and state['cosine_annealing'] is None
    validation=lines(root/'run/validation.jsonl');end=validation[-1]
    assert [record['step'] for record in validation]==[0]+list(range(2000,48001,2000))+[48192]
    best=min(validation,key=lambda record:record['monitor_cd'])
    assert state['best_validation_cd']==best['monitor_cd']
    stop=json.loads((root/'run/run_stop.json').read_text(encoding='utf-8'))
    assert stop['step']==48192 and stop['reason']=='steps' and stop['max_seconds']==0
    counts=state['stroke_counts']
    assert counts['sketch']==48192*3686 and counts['text']==48192*205
    assert sum(counts.get(key,0) for key in ('bos','eop','eod','selection'))==48192*205
    assert counts['skipped_updates']==sum(not record['success'] for record in updates)
    # 正式checkpoint还应能通过后续R使用的冻结接口完整载入FiLM和频率buffer。
    frozen=FrozenSketchLM(config_from_ae_checkpoint(state));frozen.load_ae(state)
    assert all(not parameter.requires_grad for name,parameter in frozen.named_parameters()
               if name.startswith(('encoder.','decoder.','dct_head.')))
    for key,value in state['model'].items():
        torch.testing.assert_close(value,frozen.state_dict()[key],rtol=0,atol=0)
    del frozen,baseline;gc.collect()
    audit={'job_id':606348,'gpu_seconds':args.gpu_seconds,'loop_seconds':stop['elapsed_seconds'],
           'final_step':48192,'successful_inputs':197394432,'stroke_counts':counts,'optimizer_entries':124,
           'scale':state['scaler']['scale'],'final_lr':5e-6,'lr_records_exact':len(updates),'warmup_steps':1000,
           'last_constant_step':40600,'best_step':best['step'],'best_monitor_cd':best['monitor_cd'],
           'final_monitor_cd':end['monitor_cd'],'source_commit':state['source_commit'],
           'validation_source':state['validation_source'],'data_cursors':state['data'],
           'same_other_cfg_recipe_data':True,'same_final_source_cursors':True,'finite_weights':True,'frozen_load_exact':True,
           'peak_memory_bytes':max(record['peak_memory_bytes'] for record in updates),
           'late_median_update_seconds':float(np.median([record['update_seconds'] for record in updates if record['step']>=44000])),
           'final_logged_losses':{key:updates[-1][key] for key in ('step','loss','recon_nll','dct_mse')},
           'training_wall_change_percent':100*(args.gpu_seconds/prior['audit']['total_lambda48_gpu_seconds']-1)}
    del state;gc.collect()
    geometry=json.loads((root/'geometry/report.json').read_text(encoding='utf-8'))['models']['film']
    reports={'old':prior['models_fp32']['old'],'small':small['models_fp32']['after'],
             'before':prior['models_fp32']['after'],'after':geometry}
    fields=('cd','shape_contribution','position_contribution','centered_shape_cd','centroid_distance')
    comparisons={bucket:{name:{field:comparison(reports[name]['buckets'][bucket][field]['mean'],
        geometry['buckets'][bucket][field]['mean']) for field in fields} for name in ('old','small','before')} for bucket in BUCKETS}
    phase={bucket:{field:{'hold_after_24154':comparison(at_step(validation,24154,bucket,field),at_step(validation,40600,bucket,field)),
                           'cosine':comparison(at_step(validation,40600,bucket,field),end['buckets'][bucket][field]['mean']),
                           'last2192':comparison(at_step(validation,46000,bucket,field),end['buckets'][bucket][field]['mean']),
                           'last192':comparison(at_step(validation,48000,bucket,field),end['buckets'][bucket][field]['mean'])}
                    for field in fields[:4]} for bucket in BUCKETS}
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    history=lines(args.concat_first_dir/'dct/validation.jsonl')+lines(args.concat_dir/'run/validation.jsonl')
    fig,axes=plt.subplots(2,3,figsize=(15,8.8),layout='constrained')
    for ax,bucket in zip(axes[0],BUCKETS[:3]):
        for name,rows in (('before',history),('after',validation)):
            rows=[record for record in rows if record['step']>=10000]
            ax.plot(np.array([record['step'] for record in rows])/1000,
                    [record['buckets'][bucket]['cd']['mean'] for record in rows],'.-',color=layout.COLORS[name],label=layout.LABELS[name])
        ax.set(title=NAMES[bucket]+' CD',xlabel='成功更新（千次）',ylabel='CD',yscale='log');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,0]
    for field,label,color in (('cd','图形总CD','#2667a8'),('shape_contribution','形状贡献','#ce7937'),('position_contribution','位置贡献','#238673')):
        ax.plot(np.array([record['step'] for record in validation[5:]])/1000,[record['buckets']['sketch'][field]['mean'] for record in validation[5:]],'.-',label=label,color=color)
    ax.set(title='FiLM图形误差分解',xlabel='成功更新（千次）',ylabel='CD贡献');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1];ax.plot(np.array([record['step'] for record in updates])/1000,[record['lr'] for record in updates],color='#2667a8')
    ax.axvline(40.6,color='#999999',ls='--',lw=.8);ax.set(title='实际LR：单次完整训练',xlabel='成功更新（千次）',ylabel='LR');ax.grid(alpha=.2)
    ax=axes[1,2]
    for field,label,color in (('recon_nll','重建NLL','#2667a8'),('dct_mse','DCT MSE','#ce7937')):
        target=ax if field=='recon_nll' else ax.twinx();values=[record[field] for record in updates]
        smooth=[np.mean(values[max(0,i-19):i+1]) for i in range(len(values))]
        target.plot(np.array([record['step'] for record in updates])/1000,smooth,color=color,label=label)
        target.set_ylabel(label);target.legend(loc='upper left' if field=='recon_nll' else 'upper right',fontsize=8)
    ax.set(title='训练损失（20条日志平均）',xlabel='成功更新（千次）');ax.grid(alpha=.2)
    fig.suptitle('λ48+DCT：原τ拼接与连续正弦τ/FiLM，同48192成功更新',fontsize=14)
    layout.save_plot(fig,root/'curves.png');plt.close(fig)
    samples=dict(np.load(args.samples,allow_pickle=False))
    bundles={name:dict(np.load(path,allow_pickle=False)) for name,path in (
        ('small',args.lambda16_dir/'geometry/continued_strokes.npz'),('before',args.concat_dir/'geometry/continued_strokes.npz'),('after',root/'geometry/film_strokes.npz'))}
    for bundle in bundles.values(): np.testing.assert_array_equal(bundle['cloud'],samples['cloud'])
    chosen=[273,893,3882,2828,5123,6032];tail=[3369,3913,4357,5960]
    layout.draw_strokes(root/'stroke_comparison.png',chosen,samples,bundles)
    layout.draw_strokes(root/'shape_aligned.png',tail,samples,bundles,aligned=True)
    layout.draw_strokes(root/'shape_focus.png',tail[:2],samples,bundles,aligned=True)
    examples={str(index):{name:{field:float(bundle['errors'][index,column]) for field,column in (
        ('cd',0),('shape',1),('position',2),('centered_shape_cd',3))} for name,bundle in bundles.items()} for index in chosen+tail}
    decode=decode_check(root,samples,tail[:2],args.concat_dir/'run/final.pt',args.device) if args.decode_check else None
    summary={'audit':audit,'models_fp32':reports,'comparisons_fp32':comparisons,'phase_fp16':phase,'fp16_final':end,
             'fixed_examples':chosen,'shape_tail_examples':tail,'example_errors':examples,'decode_check':decode,
             'phase_note':'24154/40600边界在相邻验证记录间插值；主表仍用同48τ历史口径',
             'fp16_fp32_graphic_delta_percent':100*(geometry['buckets']['sketch']['cd']['mean']/end['buckets']['sketch']['cd']['mean']-1)}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'audit':{key:value for key,value in audit.items() if key!='data_cursors'},'comparisons':comparisons,
                      'phase':phase,'examples':examples,'decode':decode},ensure_ascii=False))


if __name__=='__main__':
    main()
