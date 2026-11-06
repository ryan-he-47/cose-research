"""2026-10-04：大池独立笔画源、随机初始化AE、FP16完整恢复和收敛曲线。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.st.ae_direct import ArcAECollator,DirectAEStream,microbatches
from src.st.formal_assets import write_json
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.run_budget import RunBudget
from src.st.train_ae import StrokeAutoencoder,_rng_state,_restore_rng,_clip_and_check
from src.st.ae_geometry import error_components,summarize,arc_resample
from src.st.ae_schedule import WallTimeCosine,StepLRReplay


DEFAULT={'version':'direct_ae_v1','batch_size':4096,'microbatch_strokes':2048,
         'padded_point_budget':131072,'workers':4,'lr':1e-4,'warmup':1000,
         'n_t_samples':16,'clip_norm':1.,'seed':42,'validate_seconds':1800,'save_seconds':3600,
         'plateau_factor':.5,'plateau_patience':3,'min_lr':1e-6,'max_points':76}


@torch.no_grad()
def validate(model,samples,device,output,step,*,lr=None):
    model.eval()
    reconstructed=[]
    for start in range(0,len(samples['ink']),128):
        batch={'ink':torch.from_numpy(samples['ink'][start:start+128]).to(device),
               'stroke_len':torch.from_numpy(samples['lengths'][start:start+128]).to(device)}
        with torch.autocast('cuda',dtype=torch.float16):
            points=model.reconstruct(batch,num_points=48)
        reconstructed.extend(arc_resample(points,32) for points in points.float().cpu().numpy())
    reconstruction=np.stack(reconstructed)
    errors=np.asarray([error_components(a,b) for a,b in zip(samples['cloud'],reconstruction)])
    if not np.isfinite(errors[:,:6]).all():
        raise RuntimeError('新AE验证重建含非有限坐标')
    buckets={name:summarize(errors[samples['bucket']==name]) for name in sorted(set(samples['bucket']))}
    if 'sketch_object' in buckets:
        mask=np.char.startswith(samples['bucket'],'sketch')
        buckets['sketch']=summarize(errors[mask])
    # 只用于LR调度/保存较低CD的状态；各桶曲线完整保留，不代替用户判断收敛。
    monitor=float(np.mean([buckets[name]['cd']['mean'] for name in ('text','sketch_object','sketch_scene')]))
    record={'step':step,'buckets':buckets,'monitor_cd':monitor,
            'unix_seconds':time.time(),'precision':'FP16 trunk/FP32 head'}
    if lr is not None:
        record['lr']=lr
    write_json(output/'validation_latest.json',record)
    with (output/'validation.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record)+'\n')
    np.savez_compressed(output/'validation_latest.npz',reconstructed=reconstruction,
        errors=errors,bucket=samples['bucket'],original=samples['cloud'])
    print(json.dumps({'ae_validation':record}),flush=True)
    model.train()
    return monitor


def save(path,model,optimizer,plateau,scaler,stream,collator,*,step,counters,recipe,
         data_args,manifest,validation_source,best,validation_age,cosine_state=None,initial_ed_sha256=None,
         step_lr_state=None):
    state={'stage':'ae','format_version':3,'phase':'ae_direct_v1','cfg':asdict(model.cfg),
           'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':plateau.state_dict(),
           'scaler':scaler.state_dict(),'data':stream.state_dict(),'collator':collator.state_dict(),
           'rng':_rng_state(),'step':step,'stroke_counts':dict(counters),'train_args':recipe,
           'data_args':data_args,'data_manifest':manifest,'ae_source':None,
           'initialization':'random','best_validation_cd':best,'validation_source':validation_source,
           'validation_age_seconds':validation_age,'source_commit':subprocess.check_output(
               ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    # plateau状态保留兼容旧checkpoint；启用cosine后不再调用plateau.step。
    state['cosine_annealing']=cosine_state
    state['step_lr_schedule']=step_lr_state
    state['initial_ed_sha256']=initial_ed_sha256
    temporary=path.with_suffix('.pt.tmp')
    torch.save(state,temporary)
    temporary.replace(path)
    write_json(path.parent/'latest.json',{'checkpoint':path.name,'step':step,
        'successful_strokes':sum(counters[name] for name in ('sketch','text','bos','eop','eod','selection')),
        'best_validation_cd':best,'source_commit':state['source_commit']})


def direct_ae_config(recipe):
    """2026-10-06：配方覆盖AE尺寸/DCT/τ条件方式，默认与历史入口相同。"""
    cfg=gpt2_small_pretrain_config()
    cfg.freeze_ed=False;cfg.n_t_samples=recipe['n_t_samples']
    allowed={'latent_units','enc_d_model','enc_layers','enc_heads','enc_dff',
             'dec_hidden','dec_layers','dec_tau_mode','dec_tau_pe_dim','dec_tau_max_cycles',
             'dct_coeffs','dct_dc','dct_weight'}
    overrides=recipe.get('ae_config',{})
    if set(overrides)-allowed:
        raise ValueError('ae_config只能修改AE尺寸/DCT/τ条件方式，不能覆盖R/坐标/页协议')
    for key,value in overrides.items():
        setattr(cfg,key,value)
    if cfg.dct_coeffs==0 and (cfg.dct_dc or cfg.dct_weight!=0):
        raise ValueError('完全无DCT臂须coeffs=0、dc=False、weight=0')
    return cfg


def ed_fingerprint(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name.startswith(('encoder.','decoder.')):
            digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def capacity(model,records,collator,recipe,scaler,device):
    rng=collator.state_dict()
    model_rng=_rng_state()
    trials=[]
    for budget,micro in ((65536,1024),(131072,2048),(262144,4096)):
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started=time.monotonic()
        try:
            count=0
            for cpu in microbatches(records,collator,documents=micro,point_budget=budget):
                batch={key:value.to(device) if torch.is_tensor(value) else value for key,value in cpu.items()}
                with torch.autocast('cuda',dtype=torch.float16):
                    result=model(batch)
                    loss=result['loss']*len(batch['stroke_len'])/len(records)
                scaler.scale(loss).backward()
                count+=1
                del result,loss,batch,cpu
            torch.cuda.synchronize(device)
            trials.append({'padded_point_budget':budget,'microbatch_strokes':micro,
                'seconds':time.monotonic()-started,'microbatches':count,'oom':False,
                'allocated_bytes':torch.cuda.max_memory_allocated(device),
                'reserved_bytes':torch.cuda.max_memory_reserved(device)})
        except torch.cuda.OutOfMemoryError:
            trials.append({'padded_point_budget':budget,'microbatch_strokes':micro,'oom':True})
        finally:
            result=loss=batch=cpu=None
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            collator.load_state_dict(rng)
            _restore_rng(model_rng)
    viable=[trial for trial in trials if not trial['oom']]
    if not viable:
        raise RuntimeError('AE容量试算均OOM')
    selected=min(viable,key=lambda trial:trial['seconds'])
    recipe.update({key:selected[key] for key in ('padded_point_budget','microbatch_strokes')})
    torch.cuda.reset_peak_memory_stats(device)
    return {'trials':trials,'selected':selected,'device':torch.cuda.get_device_name(device)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--validation_samples',type=Path,required=True)
    parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--recipe',type=Path)
    parser.add_argument('--steps',type=int,default=1000000000)
    parser.add_argument('--max_seconds',type=float,default=42600)
    parser.add_argument('--cosine_end_lr',type=float,
                        help='从恢复后的实际LR开始，在max_seconds内余弦降到此值')
    parser.add_argument('--cosine_hold_seconds',type=float,
                        help='2026-10-05：新余弦阶段先保持当前LR的秒数，包含在max_seconds内')
    parser.add_argument('--lr_replay',type=Path,
                        help='按成功步数/LR轨迹训练；2026-10-06支持step0升温与完整单段训练')
    parser.add_argument('--replace_lr_schedule',action='store_true',
                        help='2026-10-07：明确更换恢复点之后的LR；起始步数/LR必须与完整checkpoint一致')
    parser.add_argument('--calibrate',action='store_true')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--microbatch_strokes',type=int)
    parser.add_argument('--padded_point_budget',type=int)
    parser.add_argument('--log_every',type=int,default=50)
    args=parser.parse_args()
    if os.getenv("COSE_REQUIRE_SLURM") == "1" and not os.getenv('SLURM_JOB_ID'):
        parser.error('服务器训练必须经Slurm计算节点')
    if not torch.cuda.is_available():
        parser.error('独立AE正式入口要求CUDA FP16')
    recipe=dict(DEFAULT)
    if args.recipe:
        recipe.update(json.loads(args.recipe.read_text()))
    state=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    saved_cosine=state.get('cosine_annealing') if state else None
    saved_step_lr=state.get('step_lr_schedule') if state else None
    step_lr=StepLRReplay.from_state_dict(saved_step_lr) if saved_step_lr else None
    if args.replace_lr_schedule and (not state or not args.lr_replay):
        parser.error('更换LR需要完整恢复checkpoint并指定新轨迹')
    if args.lr_replay:
        requested=StepLRReplay(json.loads(args.lr_replay.read_text(encoding='utf-8')))
        if saved_cosine or args.cosine_end_lr is not None or args.cosine_hold_seconds is not None:
            parser.error('按步LR轨迹不能叠加墙钟余弦')
        if step_lr and step_lr.state_dict()!=requested.state_dict() and not args.replace_lr_schedule:
            parser.error('恢复的LR轨迹不同，不能重开或更换收尾进度')
        if not state and (requested.start_step!=0 or requested.lr(0)!=recipe['lr'] or
                          requested.warmup_steps!=recipe['warmup']):
            parser.error('随机初始化的完整LR轨迹需从step0开始，峰值/升温与配方一致')
        if state and (not step_lr or args.replace_lr_schedule) and (state['step']!=requested.start_step or
                           state['optimizer']['param_groups'][0]['lr']!=requested.lr(requested.start_step)):
            parser.error('新按步轨迹的起点/LR与恢复checkpoint不符')
        step_lr=requested
    if step_lr:
        if saved_cosine or args.cosine_end_lr is not None or args.cosine_hold_seconds is not None:
            parser.error('保存的按步LR不能叠加墙钟余弦')
        if args.steps==1000000000:
            args.steps=step_lr.end_step
        current_step=state['step'] if state else 0
        if not step_lr.start_step<=current_step<=args.steps<=step_lr.end_step:
            parser.error('训练步数超出已对齐LR轨迹范围')
    if args.cosine_end_lr is not None and (not state or saved_cosine):
        parser.error('开启余弦收尾需恢复已有AE；已在退火的checkpoint直接恢复，不能重开曲线')
    if args.cosine_hold_seconds is not None and args.cosine_end_lr is None:
        parser.error('恒定段仅在开启新余弦阶段时指定；恢复已有曲线时沿用保存的时长')
    if state:
        if state.get('phase')!='ae_direct_v1':
            parser.error('此入口只恢复新的独立源AE，不加载旧AE/R')
        if args.recipe and recipe.get('ae_config') is not None and recipe['ae_config']!=state['train_args'].get('ae_config'):
            parser.error('恢复的AE尺寸/DCT臂与请求配方不同，不能混接两臂')
        recipe=dict(state['train_args'])
    for name in ('workers','batch_size','microbatch_strokes','padded_point_budget'):
        value=getattr(args,name)
        if value is not None:
            recipe[name]=value
    if recipe['workers'] < 0 or any(recipe[key] <= 0 for key in (
            'batch_size','microbatch_strokes','padded_point_budget','n_t_samples',
            'validate_seconds','save_seconds','lr','clip_norm')):
        parser.error('训练容量、时间间隔、学习率必须为正，workers不能为负')
    if args.steps < 0 or args.log_every <= 0 or recipe['warmup'] < 0:
        parser.error('steps/warmup不能为负，log_every必须为正')
    for key in ('validate_steps','save_steps'):
        if recipe.get(key,0)<0:
            parser.error('按步验证/保存间隔不能为负')
    schedule_kind=recipe.get('lr_schedule','warmup_then_plateau')
    if schedule_kind not in ('warmup_then_plateau','warmup_constant'):
        parser.error('未知独立AE学习率调度')
    if recipe['max_points'] != 76:
        parser.error('本阶段沿用76点稀疏化，不通过配方改动此协议')
    torch.set_num_threads(4)
    random.seed(recipe['seed']);np.random.seed(recipe['seed']);torch.manual_seed(recipe['seed'])
    cfg=direct_ae_config(recipe)
    if state:
        for key,value in state['cfg'].items():
            setattr(cfg,key,value)
    device=torch.device('cuda')
    model=StrokeAutoencoder(cfg)
    initial_ed_sha256=state.get('initial_ed_sha256') if state else ed_fingerprint(model)
    model=model.to(device)
    if not state and recipe.get('reset_rng_after_init',False):
        # 两臂E/D在head之前初始化；重置运行RNG，避免有无head改变后续随机状态。
        torch.manual_seed(recipe['seed'])
    optimizer=torch.optim.Adam(model.parameters(),lr=recipe['lr'])
    plateau=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=recipe['plateau_factor'],
        patience=recipe['plateau_patience'],min_lr=recipe['min_lr'])
    scaler=torch.amp.GradScaler('cuda',init_scale=256)
    data_args=json.loads(args.manifest.read_text())
    if data_args.get('split') != 'train':
        parser.error('训练manifest必须指定train源')
    stream=DirectAEStream(data_args,batch_size=recipe['batch_size'],workers=recipe['workers'])
    manifest=stream.dataset.manifest()
    collator=ArcAECollator(cfg,recipe['seed'])
    validation=dict(np.load(args.validation_samples,allow_pickle=False))
    validation_source={'sha256':hashlib.sha256(args.validation_samples.read_bytes()).hexdigest(),
                       'path':str(args.validation_samples.resolve()),'count':len(validation['ink'])}
    step=0;counters=Counter();best=float('inf');validation_age=0.
    if state:
        if state['data_manifest']!=manifest or state['validation_source']['sha256']!=validation_source['sha256']:
            raise ValueError('独立AE恢复的数据/观测样本不同')
        model.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']);plateau.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']);stream.load_state_dict(state['data'])
        collator.load_state_dict(state['collator']);_restore_rng(state['rng'])
        step=int(state['step']);counters=Counter(state['stroke_counts'])
        best=state['best_validation_cd'];validation_age=state['validation_age_seconds']
        del state;gc.collect()
    cosine=None
    if saved_cosine:
        cosine=WallTimeCosine.from_state_dict(saved_cosine)
    elif args.cosine_end_lr is not None:
        cosine=WallTimeCosine([group['lr'] for group in optimizer.param_groups],
                              args.cosine_end_lr,args.max_seconds,
                              hold_seconds=args.cosine_hold_seconds or 0.)
    if cosine:
        cosine.apply(optimizer,0.)
    elif step_lr:
        step_lr.apply(optimizer,step)
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    write_json(output/'recipe.json',recipe);write_json(output/'data_manifest.json',manifest)
    print(json.dumps({'ae_direct_start':True,'initialization':'random','step':step,
        'parameters':sum(p.numel() for p in model.parameters()),'recipe':recipe,'quotas':stream.quotas,
        'precision':'FP16/GradScaler + FP32 GMM/DCT','source':'direct full-pool strokes',
        'resume_from':str(args.resume) if args.resume else None,
        'optimizer_state_entries':len(optimizer.state),'scale':scaler.get_scale(),
        'kinds_cumulative':dict(counters),
        'ae_config':{key:getattr(cfg,key) for key in ('latent_units','enc_d_model','enc_layers','enc_heads','enc_dff','dec_hidden','dec_layers','dec_tau_mode','dec_tau_pe_dim','dec_tau_max_cycles','dct_coeffs','dct_dc','dct_weight')},
        'initial_ed_sha256':initial_ed_sha256,'target_successful_steps':args.steps,
        'lr_schedule':cosine.state_dict(0.) if cosine else step_lr.state_dict() if step_lr else {'kind':schedule_kind}}),flush=True)
    model.train();stream.prefetch()
    budget=RunBudget(args.max_seconds)
    phase_start_step=step
    last_validation=time.monotonic()-validation_age;last_save=time.monotonic()
    last_validation_step=step;last_save_step=step
    if not args.resume:
        best=validate(model,validation,device,output,step,lr=optimizer.param_groups[0]['lr'])
        last_validation=time.monotonic()
    needs_capacity=args.calibrate and not args.resume
    failed=0
    def checkpoint(name,age=None):
        save(output/name,model,optimizer,plateau,scaler,stream,collator,step=step,
            counters=counters,recipe=recipe,data_args=data_args,manifest=manifest,
            validation_source=validation_source,best=best,
            validation_age=time.monotonic()-last_validation if age is None else age,
            cosine_state=cosine.state_dict(budget.elapsed) if cosine else None,
            initial_ed_sha256=initial_ed_sha256,
            step_lr_state=step_lr.state_dict() if step_lr else None)

    try:
        while step<args.steps and budget.reason is None:
            started=time.monotonic();records=stream.next_batch();prepared=time.monotonic()
            if needs_capacity:
                report=capacity(model,records,collator,recipe,scaler,device)
                write_json(output/'capacity.json',report);write_json(output/'recipe.json',recipe)
                print(json.dumps({'ae_capacity':report}),flush=True)
                needs_capacity=False
            if step_lr:
                # 溢出时step不推进；下一次尝试继续同一个LR，恢复也不重开恒定段。
                step_lr.apply(optimizer,step+1)
            elif cosine:
                cosine.apply(optimizer,budget.elapsed)
            elif step<recipe['warmup']:
                for group in optimizer.param_groups:
                    group['lr']=recipe['lr']*(step+1)/recipe['warmup']
            optimizer.zero_grad(set_to_none=True)
            totals=Counter();kinds=Counter(record.kind for record in records);micro=0
            for cpu in microbatches(records,collator,documents=recipe['microbatch_strokes'],
                                   point_budget=recipe['padded_point_budget']):
                batch={key:value.to(device,non_blocking=True) if torch.is_tensor(value) else value for key,value in cpu.items()}
                fraction=len(batch['stroke_len'])/len(records)
                with torch.autocast('cuda',dtype=torch.float16):
                    result=model(batch);loss=result['loss']*fraction
                if not torch.isfinite(loss):
                    raise RuntimeError('新AE损失非有限')
                scaler.scale(loss).backward()
                for key,value in result.items():
                    totals[key]+=float(value.detach())*fraction
                micro+=1
            scaler.unscale_(optimizer)
            finite,norms=_clip_and_check(model,recipe['clip_norm'])
            old_scale=scaler.get_scale()
            if finite:
                scaler.step(optimizer);scaler.update()
                success=scaler.get_scale()>=old_scale
            else:
                scaler.update(new_scale=old_scale/2);success=False
            torch.cuda.synchronize(device)
            if success:
                step+=1;counters.update(kinds);failed=0
            else:
                failed+=1;counters['skipped_updates']+=1
            elapsed=time.monotonic()-started
            if not success or step<=phase_start_step+3 or step%args.log_every==0:
                print(json.dumps({'step':step,'success':success,**totals,'kinds':dict(kinds),
                    'kinds_cumulative':dict(counters),'lr':optimizer.param_groups[0]['lr'],'scale':scaler.get_scale(),
                    'annealing_progress':cosine.progress(budget.elapsed) if cosine else None,
                    'lr_replay_step':step+1 if step_lr and not success else step if step_lr else None,
                    'microbatches':micro,'prepare_seconds':prepared-started,'update_seconds':elapsed,
                    'strokes_per_second':len(records)/elapsed,'peak_memory_bytes':torch.cuda.max_memory_allocated(device),
                    'reserved_memory_bytes':torch.cuda.max_memory_reserved(device)}),flush=True)
            now=time.monotonic()
            validation_due=(step>0 and step%recipe['validate_steps']==0 and step!=last_validation_step if recipe.get('validate_steps',0)
                            else now-last_validation>=recipe['validate_seconds'])
            if validation_due:
                metric=validate(model,validation,device,output,step,lr=optimizer.param_groups[0]['lr'])
                if not cosine and not step_lr and schedule_kind=='warmup_then_plateau' and step>=recipe['warmup']:
                    before=optimizer.param_groups[0]['lr'];plateau.step(metric)
                    if before!=optimizer.param_groups[0]['lr']:
                        print(json.dumps({'ae_lr':optimizer.param_groups[0]['lr'],'trigger':'validation CD plateau'}),flush=True)
                last_validation=time.monotonic()
                last_validation_step=step
                if metric<best:
                    best=metric
                    checkpoint('best.pt',age=0.)
            save_due=(step>0 and step%recipe['save_steps']==0 and step!=last_save_step if recipe.get('save_steps',0)
                      else time.monotonic()-last_save>=recipe['save_seconds'])
            if save_due:
                checkpoint(f'ckpt_{step:07d}.pt')
                last_save=time.monotonic()
                last_save_step=step
            if failed>=20:
                raise RuntimeError('新AE连续20次梯度溢出')
        if cosine:
            cosine.apply(optimizer,budget.elapsed)
        elif step_lr:
            step_lr.apply(optimizer,step)
        metric=validate(model,validation,device,output,step,lr=optimizer.param_groups[0]['lr'])
        if metric<best:
            best=metric
            checkpoint('best.pt',age=0.)
        checkpoint('final.pt',age=0.)
        budget.finish(output,step)
    except Exception:
        # 保留最后完成的更新和已消费游标，便于检查失败状态；不自动重投。
        checkpoint('interrupted.pt')
        raise
    finally:
        stream.close()


if __name__=='__main__':
    main()
