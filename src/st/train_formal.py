"""正式FP16训练：冻结AE、按成功输入计量、任务损失与WSD。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.st.formal_assets import write_json
from src.st.formal_recipe import DEFAULT_RECIPE, InputWSD, CoverageWSD, TASKS
from src.st.formal_capacity import backward_documents, calibrate_headroom, limit_cuda_memory
from src.st.formal_stream import InputDocumentStream, collect_documents, iter_microbatches
from src.st.pipeline_pretrain import build_document_dataset
from src.st.pretrain_model import FrozenSketchLM, config_from_ae_checkpoint, config_from_r_checkpoint
from src.st.pretrain_config import apply_r_architecture,check_chinchilla_budget
from src.st.run_budget import RunBudget
from src.st.train_pretrain import _rng_state, _restore_rng
from src.st.formal_observation import observation_report
from src.st.text_auxiliary import (bind_auxiliary,annotate_documents,calibrate_auxiliary,
                                   auxiliary_report,preflight_auxiliary,validate_auxiliary,
                                   configure_recalibration,auxiliary_calibration_due,apply_auxiliary_calibration,
                                   extend_non_symbol,space_target_count,auxiliary_loss,weight_hash)


def phase_data_args(data_args, recipe):
    """显式控制字形增强；已有augmentation字典也必须关闭enabled。"""
    result = dict(data_args)
    if 'object_sampling' in recipe:result['object_sampling']=recipe['object_sampling']
    if 'paired_scene_share' in recipe:result['paired_scene_share']=recipe['paired_scene_share']
    if 'augment_text' in recipe:
        result['augment_text'] = recipe['augment_text']
        if result.get('augmentation') is not None:
            result['augmentation'] = {**result['augmentation'], 'enabled':recipe['augment_text']}
    return result


def materialize_adam(optimizer):
    """容量校准前预留新Adam的完整状态；计数仍为0，不执行更新。"""
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if not optimizer.state[parameter]:
                optimizer.state[parameter] = {
                    'step':torch.tensor(0.),
                    'exp_avg':torch.zeros_like(parameter),
                    'exp_avg_sq':torch.zeros_like(parameter)}


def ae_source_record(path, state, *, expected_step=None, expected_lr=None, completion=None):
    """检查依赖AE的完整收尾并记录实际冻结来源；也用于本机短核对。"""
    if state.get('phase') != 'ae_direct_v1':
        raise ValueError('正式R需要独立源AE checkpoint')
    step = int(state['step'])
    lr = float(state['optimizer']['param_groups'][0]['lr'])
    if expected_step is not None and step != expected_step:
        raise ValueError(f'AE尚未完成：期望{expected_step}，实际{step}')
    if expected_lr is not None and not np.isclose(lr,expected_lr,rtol=1e-12,atol=0):
        raise ValueError(f'AE收尾LR不同：{lr}')
    if completion is not None:
        stop = json.loads(Path(completion).read_text(encoding='utf-8'))
        if stop['step'] != step or stop['reason'] != 'steps':
            raise ValueError(f'AE提前结束：{stop}')
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return {'path':str(Path(path).resolve()),'step':step,'lr':lr,'sha256':digest.hexdigest(),
            'cfg':state['cfg'],'source_commit':state.get('source_commit'),
            'slurm_dependency':os.getenv('COSE_AE_JOB')}


def task_loss(result, body_counts, control_count, recipe,*,auxiliary_count=None,auxiliary_weight=None,
              auxiliary_space_count=None):
    loss = result["pred_sum"]*0
    for i, task in enumerate(TASKS):
        if body_counts[task]:
            loss = loss + recipe["loss_weights"][task]*result["task_sums"][i]/body_counts[task]
    if control_count:
        loss = loss + recipe["control_weight"]*result["control_sum"]/control_count
    if recipe.get('visual_grounding'):
        from src.st.visual_grounding_auxiliary import grounding_loss
        loss=loss+recipe.get('grounding_alpha',0.)*grounding_loss(result,recipe)
    if auxiliary_count:
        weight=recipe.get('text_aux_alpha',0.) if auxiliary_weight is None else auxiliary_weight
        if weight is None:raise ValueError('辅助权重尚未校准')
        loss=loss+weight*auxiliary_loss(result,auxiliary_count,auxiliary_space_count)
    return loss


def gpu_batch(batch, device):
    return {key:value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key,value in batch.items()}


def save_formal(path, model, optimizer, scheduler, scaler, stream, *, step, phase_step,
                counters, data_args, manifest, recipe, ae_source, parent_source,
                data_state=None, rng_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workspace_commit = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True).stdout.strip()
    # 2026-10-09：Git对象入口从已验证提交加载src，运行工作树保持不动。
    commit = globals().get('__cose_commit__', workspace_commit)
    state = {"stage":"r_pretrain", "format_version":3, "phase":"formal_v1",
             "model":model.state_dict(), "cfg":asdict(model.cfg), "step":step,
             "phase_step":phase_step, "optimizer":optimizer.state_dict(),
             "scheduler":scheduler.state_dict(), "scaler":scaler.state_dict(), "rng":_rng_state() if rng_state is None else rng_state,
             "data":stream.state_dict() if data_state is None else data_state, "counters":dict(counters), "data_args":data_args,
             "data_manifest":manifest, "train_args":recipe, "ae_source":ae_source,
             "parent_source":parent_source, "source_commit":commit,
             "workspace_commit":workspace_commit}
    if model.text_aux is not None:
        state['text_aux_manifest']=model.text_aux_manifest
    if model.visual_grounding is not None:
        state['grounding_classes']=model.grounding_classes
    state['lifetime_successful_inputs']=(parent_source or {}).get('lifetime_successful_inputs',
        (parent_source or {}).get('successful_inputs',0))+counters.get('successful_inputs',0)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
    write_json(path.parent/'latest.json', {"checkpoint":path.name, "step":step,
        "phase_step":phase_step, "successful_inputs":scheduler.inputs, "source_commit":commit,
        "target_inputs":recipe.get('total_inputs'), "training_goal":recipe.get('training_goal')})
    limits=(recipe.get('training_goal') or {}).get('limits') or recipe.get('coverage_source_limits')
    if limits:
        write_json(path.parent/'coverage.json',stream.coverage(limits))
    keep=recipe.get('keep_recent_checkpoints')
    if keep is not None:
        root=path.parent.resolve()
        files=sorted(root.glob('ckpt_[0-9][0-9][0-9][0-9][0-9][0-9][0-9].pt'))
        for old in files[:-keep] if keep else files:
            if old.is_symlink() or old.resolve().parent!=root:raise RuntimeError('检查点保留路径异常')
            old.unlink()


def calibrate_capacity(model, documents, body, control, recipe, device,auxiliary_count=None,auxiliary_space_count=None):
    """同一批真实文档的前后向容量测量；不更新权重/Adam，不推进训练游标。"""
    if recipe.get('memory_headroom_gib') is not None:
        return calibrate_headroom(model,documents,body,control,recipe,device,
                                  partial(task_loss,auxiliary_count=auxiliary_count,
                                          auxiliary_space_count=auxiliary_space_count),gpu_batch,_rng_state,_restore_rng)
    original = dict(recipe)
    calibration_rng = _rng_state()
    trials = []
    candidates = [(recipe["padded_stroke_budget"], recipe["activation_checkpoint"])]
    # 扩大到65536只在起步占用较低且更快时采用；容量不足时回退。
    for _ in range(4):
        budget, activation = candidates.pop(0)
        recipe.update(padded_stroke_budget=budget, activation_checkpoint=activation)
        model.activation_checkpoint = activation
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        try:
            microbatches = 0
            for cpu in iter_microbatches(documents,recipe,max_strokes=model.cfg.max_context_strokes,pin=True):
                with torch.autocast('cuda',dtype=torch.float16):
                    result = model(gpu_batch(cpu,device))
                    loss = task_loss(result,body,control,recipe,auxiliary_count=auxiliary_count,
                                     auxiliary_space_count=auxiliary_space_count)
                loss.backward()
                microbatches += 1
                del result, loss
            torch.cuda.synchronize(device)
            trial = {"padded_stroke_budget":budget, "activation_checkpoint":activation,
                     "seconds":time.monotonic()-started, "microbatches":microbatches,
                     "allocated_bytes":torch.cuda.max_memory_allocated(device),
                     "reserved_bytes":torch.cuda.max_memory_reserved(device), "oom":False}
            trials.append(trial)
            if len(trials) == 1 and trial["reserved_bytes"] < 20*2**30 and budget < 65536:
                candidates.append((65536,False))
            else:
                break
        except torch.cuda.OutOfMemoryError:
            trials.append({"padded_stroke_budget":budget,"activation_checkpoint":activation,"oom":True})
            candidates.append((max(8192,budget//2), True))
        finally:
            result = loss = None
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            _restore_rng(calibration_rng)
    viable = [item for item in trials if not item["oom"]]
    if not viable:
        recipe.update(original)
        raise RuntimeError("容量校准回退后仍OOM")
    # 先保留更快的可用配置；测量包含真实padding开销，避免为占满显存而放大。
    selected = min(viable,key=lambda item:item["seconds"])
    recipe.update(padded_stroke_budget=selected["padded_stroke_budget"],
                  activation_checkpoint=selected["activation_checkpoint"])
    model.activation_checkpoint = recipe["activation_checkpoint"]
    torch.cuda.reset_peak_memory_stats(device)
    return {"trials":trials, "selected":selected, "device":torch.cuda.get_device_name(device)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_manifest',type=Path,required=True)
    parser.add_argument('--recipe',type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--new_phase',type=Path,help='沿用R/Adam/scaler，显式启用新正式分布/WSD')
    source.add_argument('--resume',type=Path,help='完整恢复同一正式阶段')
    source.add_argument('--init_ae',type=Path,help='冻结指定AE，从头初始化R')
    source.add_argument('--warm_start',type=Path,help='保留正式R/AE与数据游标，新建Adam/scaler/WSD')
    parser.add_argument('--expected_ae_step',type=int)
    parser.add_argument('--expected_ae_lr',type=float)
    parser.add_argument('--ae_completion',type=Path)
    parser.add_argument('--output_dir',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1000000000)
    parser.add_argument('--max_seconds',type=float,default=0)
    parser.add_argument('--calibrate',action='store_true')
    parser.add_argument('--continue_data',action='store_true',help='新阶段保留源游标/RNG，与--new_phase一起使用')
    parser.add_argument('--init_text_aux',type=Path,help='2026-10-14：在新阶段接形状/逐笔空格头，保留R/Adam')
    parser.add_argument('--extend_aux_non_symbol',action='store_true',help='新阶段显式追加非符号类并扩展Adam')
    parser.add_argument('--text_aux_validation',type=Path,help='原读出128篇valid文本，当前R在线验证')
    parser.add_argument('--memory_headroom_gib',type=float)
    parser.add_argument('--log_every',type=int,default=10)
    parser.add_argument('--anneal_to',type=float,help='2026-10-12：完整恢复后显式提前退火，走完剩余输入')
    parser.add_argument('--input_budget_to',type=int,help='覆盖阶段完整恢复后改为累计成功输入终点')
    parser.add_argument('--runtime_tuning',type=Path,help='仅覆盖预取/微批工程参数的JSON')
    parser.add_argument('--aux_recalibrate_ratio',type=float,help='完整恢复后显式修改辅助/h几何梯度RMS目标比例')
    parser.add_argument('--aux_recalibrate_every_inputs',type=int,help='按成功输入持续低频校准α')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--microbatch_documents',type=int)
    parser.add_argument('--padded_stroke_budget',type=int)
    parser.add_argument('--activation_checkpoint',action=argparse.BooleanOptionalAction,default=None)
    args = parser.parse_args()
    if (args.input_budget_to is not None or args.runtime_tuning is not None) and not args.resume:
        parser.error('输入终点和工程调参仅用于同阶段完整恢复')
    if (args.continue_data or args.init_text_aux) and not args.new_phase:
        parser.error('--continue_data/--init_text_aux需要--new_phase')
    if args.extend_aux_non_symbol and not args.new_phase:
        parser.error('--extend_aux_non_symbol仅用于--new_phase')
    if args.anneal_to is not None and not args.resume:
        parser.error('--anneal_to只用于同阶段完整恢复')
    if args.aux_recalibrate_ratio is not None or args.aux_recalibrate_every_inputs is not None:
        if not args.resume or args.aux_recalibrate_ratio is None or args.aux_recalibrate_every_inputs is None:
            parser.error('辅助重校准两个参数需同时指定，且仅用于--resume')
        if not math.isfinite(args.aux_recalibrate_ratio) or args.aux_recalibrate_ratio<=0 or args.aux_recalibrate_every_inputs<=0:
            parser.error('辅助重校准比例与间隔必须为有限正数')
    if not args.init_ae and any(value is not None for value in
                               (args.expected_ae_step,args.expected_ae_lr,args.ae_completion)):
        parser.error('AE收尾检查只用于新R起跑')
    if args.steps < 1 or args.log_every < 1:
        parser.error('steps/log_every必须为正')
    if os.getenv("COSE_REQUIRE_SLURM") == "1" and not os.getenv('SLURM_JOB_ID'):
        parser.error('服务器训练请经Slurm计算节点')
    recipe = {**DEFAULT_RECIPE}
    if args.recipe:
        recipe.update(json.loads(args.recipe.read_text(encoding='utf-8')))
    state = torch.load(args.resume or args.new_phase or args.warm_start or args.init_ae,map_location='cpu',weights_only=False)
    if recipe.get('budget_status')=='proposal':
        parser.error('该配方仍为建议预算；用户给定新增训练量后生成budget_status=approved的实际配方')
    if args.warm_start and state.get('phase') != 'formal_v1':
        parser.error('纠偏接续需要正式R完整checkpoint')
    if args.init_ae:
        ae_source = ae_source_record(args.init_ae,state,expected_step=args.expected_ae_step,
                                     expected_lr=args.expected_ae_lr,completion=args.ae_completion)
    if args.resume:
        if state.get('phase') != 'formal_v1':
            parser.error('旧阶段继续请使用--new_phase')
        if args.recipe:
            immutable = ('total_inputs','input_budget','max_points','seed','peak_lr','min_lr',
                         'warmup_inputs','decay_inputs','input_shares','loss_weights','control_weight')
            if any(recipe[name] != state['train_args'][name] for name in immutable):
                parser.error('恢复时正式配方不同；改变分布/损失/调度请显式使用--new_phase')
            for name in ('augment_text','sigma_eps_r'):
                before = state['train_args'].get(name)
                after = recipe.get(name)
                if before != after:
                    parser.error(f'恢复时{name}不同，请显式启动新阶段')
            if recipe.get('r_sigma_floor_mode','add') != state['train_args'].get('r_sigma_floor_mode','add'):
                parser.error('恢复时R的σ参数化不同，请显式启动新阶段')
            for name in ('text_auxiliary','text_aux_alpha','text_aux_gradient_ratio','text_aux_probe_batches',
                         'text_aux_calibration_interval_inputs','text_aux_non_symbol','training_goal',
                         'object_sampling','r_architecture','chinchilla'):
                if recipe.get(name)!=state['train_args'].get(name):
                    parser.error(f'恢复时{name}不同')
            for name in ('paired_scene_share','visual_grounding','text_aux_normalization','text_aux_task_weights','grounding_alpha'):
                if recipe.get(name)!=state['train_args'].get(name):parser.error(f'恢复时{name}不同')
        recipe = dict(state['train_args'])
    runtime_transition=None
    if args.runtime_tuning:
        tuning=json.loads(args.runtime_tuning.read_text(encoding='utf-8'))
        allowed={'workers','microbatch_documents','prefetch_documents','prepare_batch_documents',
                 'prepare_updates','encoder_length_buckets','cache_validation_embeddings'}
        if not tuning or set(tuning)-allowed:
            parser.error('工程调参只允许供数、微批及E点长分组参数')
        runtime_transition=dict(previous={k:recipe.get(k) for k in tuning},current=tuning)
        recipe.update(tuning)
    for name in ('workers','microbatch_documents','padded_stroke_budget','activation_checkpoint'):
        value = getattr(args,name)
        if value is not None:
            recipe[name] = value
    if recipe.get('initialization_policy') == 'random_R_and_heads' and not (args.init_ae or args.resume):
        parser.error('从头配方只接受独立AE初始化或本轮完整恢复，禁止加载旧R作为起点')
    if args.memory_headroom_gib is not None:
        if not math.isfinite(args.memory_headroom_gib) or args.memory_headroom_gib <= 0:
            parser.error('显存余量必须为有限正数GiB')
        recipe['memory_headroom_gib'] = args.memory_headroom_gib
    if any(recipe[name] <= 0 for name in ('input_budget','padded_stroke_budget','microbatch_documents','encode_chunk_size')) or recipe['workers'] < 0:
        parser.error('batch/编码分块须为正，workers须非负')
    for name in ('prepare_updates','cache_validation_embeddings'):
        if name in recipe and not isinstance(recipe[name],bool):parser.error(f'{name}须为布尔值')
    buckets=recipe.get('encoder_length_buckets',[])
    if (not isinstance(buckets,(list,tuple)) or any(type(x) is not int or not 0<x<=recipe['max_points'] for x in buckets)
            or list(buckets)!=sorted(set(buckets))):
        parser.error('E点长分组须为不超过max_points的递增正整数列表')
    data_args = phase_data_args(json.loads(args.data_manifest.read_text(encoding='utf-8')), recipe)
    if not data_args.get('formal'):
        parser.error('正式入口需要formal数据manifest')
    torch.set_num_threads(4)
    random.seed(recipe['seed'])
    np.random.seed(recipe['seed'])
    torch.manual_seed(recipe['seed'])
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('正式入口要求CUDA FP16；CPU工程核对使用check_formal_pretraining.py')
    dataset = build_document_dataset(**data_args)
    manifest = dataset.manifest()
    if args.init_ae and recipe.get('initialization_policy') == 'random_R_and_heads':
        recipe['coverage_source_limits'] = dataset.coverage_limits()
    source_before,source_after=dict(state.get('data_manifest',{})),dict(manifest)
    if args.new_phase and args.continue_data:
        for name in ('paired_scene_share','paired_scene_version'):
            source_before.pop(name,None);source_after.pop(name,None)
    if (args.resume or args.warm_start or args.continue_data) and source_before != source_after:
        raise ValueError('正式恢复的数据来源/配方不同')
    cfg = config_from_ae_checkpoint(state) if args.init_ae else config_from_r_checkpoint(state)
    apply_r_architecture(cfg,recipe,fresh=bool(args.init_ae))
    floor_mode = recipe.get('r_sigma_floor_mode',cfg.r_sigma_floor_mode)
    if floor_mode not in ('add','quadrature'):
        parser.error('r_sigma_floor_mode须为add或quadrature')
    if args.resume and cfg.r_sigma_floor_mode != floor_mode:
        parser.error('恢复配方与checkpoint的R σ参数化不同')
    cfg.r_sigma_floor_mode = floor_mode
    if 'sigma_eps_r' in recipe:
        if not math.isfinite(recipe['sigma_eps_r']) or recipe['sigma_eps_r'] <= 0:
            parser.error('sigma_eps_r须为有限正数')
        if args.resume and cfg.sigma_eps_r != recipe['sigma_eps_r']:
            parser.error('恢复配方与checkpoint的σ下限不同')
        cfg.sigma_eps_r = recipe['sigma_eps_r']
    model = FrozenSketchLM(cfg,encode_chunk_size=recipe['encode_chunk_size'],
                          activation_checkpoint=recipe['activation_checkpoint']).to(device)
    if args.init_ae:
        model.load_ae(state)
        step, phase_step, counters = 0, 0, Counter()
        parent_source = None
    else:
        model.load_state_dict(state['model'],strict=True)
        step = int(state['step'])
        phase_step = int(state.get('phase_step',0)) if args.resume else 0
        counters = Counter(state['counters']) if args.resume else Counter()
        ae_source = state['ae_source']
        parent_source = state.get('parent_source') if args.resume else {
            'path':str((args.warm_start or args.new_phase).resolve()),'step':step,
            'successful_inputs':state.get('counters',{}).get('successful_inputs'),
            'lifetime_successful_inputs':state.get('lifetime_successful_inputs',state.get('counters',{}).get('successful_inputs',0)),
            'optimizer_reset':bool(args.warm_start)}
    optimizer = torch.optim.Adam(model.r_pred.parameters(),lr=recipe['peak_lr'])
    # 恢复新阶段的两组Adam；初次接头则先恢复原R的一组，随后新增头组。
    if model.text_aux is not None:
        optimizer.add_param_group({'params':model.text_aux.parameters(),'role':'text_aux'})
    if model.visual_grounding is not None:
        optimizer.add_param_group({'params':model.visual_grounding.parameters(),'role':'visual_grounding'})
        model.grounding_classes=state['grounding_classes']
        if model.grounding_classes!=list(dataset.drawings.classes):raise ValueError('对象辅助类别表不同')
    scaler = torch.amp.GradScaler('cuda',init_scale=256)
    if args.resume or args.new_phase:
        optimizer.load_state_dict(state['optimizer'])
        if state.get('scaler'):
            scaler.load_state_dict(state['scaler'])
    if recipe.get('visual_grounding') and model.visual_grounding is None:
        if not args.new_phase:parser.error('接入对象辅助头须显式new_phase')
        from src.st.visual_grounding_auxiliary import install_grounding
        install_grounding(model,optimizer,recipe['visual_grounding'],dataset.drawings.classes)
        recipe.update(text_aux_alpha=0.,text_aux_calibration_pending=True)
        cfg.r_text_aux_weight=0.
    if bool(recipe.get('visual_grounding'))!=(model.visual_grounding is not None):
        parser.error('对象辅助结构与实际配方不同')
    auxiliary_catalog=None;auxiliary_manifest=None
    if args.init_text_aux:
        if not recipe.get('text_auxiliary'):parser.error('配方未启用文字辅助损失')
        auxiliary_catalog,auxiliary_manifest=bind_auxiliary(model,dataset.font,manifest['font_sha256'],
                                                          head_path=args.init_text_aux)
        optimizer.add_param_group({'params':model.text_aux.parameters(),
                                  'lr':optimizer.param_groups[0]['lr'],'role':'text_aux'})
    elif args.init_ae and model.text_aux is not None:
        auxiliary_catalog,auxiliary_manifest=bind_auxiliary(model,dataset.font,manifest['font_sha256'],
            initialize_random=True,initialization_seed=recipe['seed'],
            include_non_symbol=bool(recipe.get('text_aux_non_symbol',False)))
    elif model.text_aux is not None:
        auxiliary_catalog,auxiliary_manifest=bind_auxiliary(model,dataset.font,manifest['font_sha256'],
                                                          saved_manifest=state.get('text_aux_manifest'))
    if bool(recipe.get('text_auxiliary',False))!=(model.text_aux is not None):
        parser.error('辅助训练配方与模型结构不同')
    if args.extend_aux_non_symbol:
        auxiliary_catalog,auxiliary_manifest=extend_non_symbol(model,optimizer,dataset.font,auxiliary_manifest)
        recipe.update(text_aux_alpha=cfg.r_text_aux_weight,text_aux_calibration_pending=True)
    if bool(recipe.get('text_aux_non_symbol',False))!=(auxiliary_catalog is not None and auxiliary_catalog.non_symbol_class is not None):
        parser.error('非符号类配方与checkpoint不同；新阶段须显式扩展')
    if args.init_ae:
        recipe['initialization'] = dict(kind='random_R_and_heads',seed=recipe['seed'],
            ae_sha256=ae_source['sha256'],r_sha256=weight_hash(model.r_pred),
            auxiliary_sha256=weight_hash(model.text_aux) if model.text_aux is not None else None,
            inherited_R=False,inherited_optimizer=False,inherited_data=False)
    parameter_budget=check_chinchilla_budget(model,recipe)
    validation_path=args.text_aux_validation or recipe.get('text_aux_validation_path')
    if validation_path:
        if model.text_aux is None:parser.error('文字辅助验证需要辅助头')
        validation_path=Path(validation_path).resolve()
        digest=hashlib.sha256(validation_path.read_bytes()).hexdigest()
        if args.resume and digest!=auxiliary_manifest.get('validation_sha256'):
            parser.error('恢复的验证文本不同')
        auxiliary_manifest.update(validation_path=str(validation_path),validation_sha256=digest)
        recipe['text_aux_validation_path']=str(validation_path)
    if args.resume and model.text_aux is not None and recipe.get('text_aux_alpha')!=cfg.r_text_aux_weight:
        parser.error('恢复的辅助损失权重与模型cfg不同')
    if args.calibrate:
        materialize_adam(optimizer)
    if recipe.get('memory_headroom_gib') is not None:
        print(json.dumps({'memory_limit':limit_cuda_memory(device,recipe['memory_headroom_gib'])}),flush=True)
    start_lr = state['scheduler']['spec']['start_lr'] if args.resume else (
        optimizer.param_groups[0]['lr'] if args.new_phase else 0.)
    goal=recipe.get('training_goal')
    if goal:
        if goal.get('kind')!='all_task_pools' or goal.get('passes')!=1 or not 0<goal['decay_fraction']<1:
            parser.error('数据目标必须为各任务源池至少一遍，且退火份额合法')
        limits=dataset.coverage_limits()
        if 'limits' in goal and goal['limits']!=limits:parser.error('恢复的数据池规模不同')
        goal['limits']=limits
    scheduler = CoverageWSD(optimizer,recipe,start_lr) if goal else InputWSD(optimizer,recipe,start_lr)
    stream = InputDocumentStream(dataset,data_args,recipe['input_shares'],
                                max_strokes=cfg.max_context_strokes,workers=recipe['workers'],
                                prefetch_documents=recipe.get('prefetch_documents'),
                                prepare_batch_documents=recipe.get('prepare_batch_documents'))
    if args.resume:
        scheduler.load_state_dict(state['scheduler'])
        stream.load_state_dict(state['data'])
        _restore_rng(state['rng'])
    elif args.warm_start or args.continue_data:
        stream.continue_positions(state['data'])
        _restore_rng(state['rng'])
    elif args.new_phase:
        _restore_rng(state['rng'])
    if args.init_ae:
        if (step or phase_step or counters or parent_source is not None or scheduler.inputs
            or any(stream.indices.values()) or any(stream.offsets.values())
            or any(float(v['step']) != 0 for v in optimizer.state.values())):
            raise RuntimeError('从AE新建R必须以零步数、零Adam步数、零预算和零消费游标起跑')
    input_transition=None
    if args.input_budget_to is not None:
        scheduler,input_transition=InputWSD.from_coverage(scheduler,recipe,args.input_budget_to)
        input_transition['source_checkpoint']=str(args.resume.resolve())
        goal=None
    coverage_limits=(goal or {}).get('limits') or recipe.get('coverage_source_limits')
    transition = None
    if args.anneal_to is not None:
        if goal:parser.error('覆盖阶段按最后覆盖进度退火，不接受输入终点退火')
        transition = scheduler.begin_decay(args.anneal_to)
        transition['source_checkpoint'] = str(args.resume.resolve())
        recipe.update(peak_lr=scheduler.spec['peak_lr'],min_lr=scheduler.spec['min_lr'],
                      decay_inputs=scheduler.spec['decay_inputs'],anneal_transition=transition)
    auxiliary_transition=None
    if args.aux_recalibrate_ratio is not None:
        if model.text_aux is None:
            parser.error('重校准需要已训练的文字辅助头')
        auxiliary_transition=configure_recalibration(recipe,args.aux_recalibrate_ratio,
            args.aux_recalibrate_every_inputs,inputs=scheduler.inputs,step=step,source=args.resume.resolve())
    del state
    gc.collect()
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'recipe.json',recipe)
    write_json(output/'data_manifest.json',manifest)
    if recipe.get('initialization') is not None:
        write_json(output/'initialization.json',recipe['initialization'])
    for name,value in (('runtime_transition',runtime_transition),('input_budget_transition',input_transition)):
        if value is not None:
            write_json(output/(name+'.json'),value)
            print(json.dumps({name:value}),flush=True)
    if auxiliary_transition is not None:
        write_json(output/'auxiliary_transition.json',auxiliary_transition)
        print(json.dumps({'auxiliary_transition':auxiliary_transition}),flush=True)
    if parameter_budget is not None:
        write_json(output/'parameter_budget.json',parameter_budget)
        print(json.dumps({'parameter_budget':parameter_budget}),flush=True)
    if transition is not None:
        write_json(output/'anneal_transition.json',transition)
        print(json.dumps({'anneal_transition':transition}),flush=True)
    if auxiliary_manifest is not None:
        write_json(output/'text_aux_manifest.json',auxiliary_manifest)
    model.train()
    model.cache_validation_embeddings=recipe.get('cache_validation_embeddings',False)
    if model.text_aux is not None:
        preflight=preflight_auxiliary(model,dataset.font,auxiliary_catalog,device,_rng_state,_restore_rng)
        write_json(output/'text_aux_preflight.json',preflight)
        print(json.dumps({'text_aux_preflight':preflight}),flush=True)
    if validation_path:
        initial_valid=validate_auxiliary(model,dataset.font,auxiliary_catalog,validation_path,
            manifest.get('page_layout',{}),device,_rng_state,_restore_rng)
        initial_valid.update(step=step,successful_inputs=scheduler.inputs)
        write_json(output/'validation_initial.json',initial_valid)
        print(json.dumps({'text_aux_validation':initial_valid}),flush=True)
    print(json.dumps({'formal_start':True,'step':step,'phase_step':phase_step,'recipe':recipe,
                      'precision':'FP16 + FP32 GMM/loss','parent':parent_source,'ae_source':ae_source,
                      'r_parameters':sum(p.numel() for p in model.r_pred.parameters()),
                      'r_layers':cfg.r_layers,'r_head_mlp':cfg.r_head_mlp,
                      'text_aux_hidden':cfg.r_text_aux_hidden if model.text_aux else None,
                      'sigma_eps_r':cfg.sigma_eps_r,'r_sigma_floor_mode':cfg.r_sigma_floor_mode,
                      'source_commit':globals().get('__cose_commit__'), 'active_tasks':stream.active_tasks,
                      'augmentation_enabled':dataset.augmentation.enabled,
                      'optimizer_state_entries':len(optimizer.state),
                      'text_auxiliary':model.text_aux is not None,
                      'text_aux_parameters':sum(p.numel() for p in model.text_aux.parameters()) if model.text_aux else 0,
                      'optimizer_groups':[{'role':g.get('role','R'),'parameters':len(g['params']),
                          'step_max':max((float(optimizer.state[p]['step']) for p in g['params'] if optimizer.state[p]),default=0.)}
                          for g in optimizer.param_groups],
                      'optimizer_step_max':max((float(v['step']) for v in optimizer.state.values() if v),default=0.),
                      'data_positions':{'indices':stream.indices,'offsets':stream.offsets},
                      'frozen_ae_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad)}),flush=True)
    stream.prefetch()
    if recipe.get('prepare_updates', False):
        from src.st.formal_prefetch import PreparedUpdates
        stream = PreparedUpdates(stream, recipe, auxiliary_catalog)
    if recipe.get('visual_grounding') and phase_step==0:
        save_formal(output/'stage_start.pt',model,optimizer,scheduler,scaler,stream,
            step=step,phase_step=phase_step,counters=counters,data_args=data_args,manifest=manifest,
            recipe=recipe,ae_source=ae_source,parent_source=parent_source)
    budget = RunBudget(args.max_seconds)
    last_save = time.monotonic()
    throughput_inputs, throughput_seconds, throughput_updates = 0, 0., 0
    need_calibration = args.calibrate
    calibration_checkpoint_pending=any(value is not None for value in
        (runtime_transition,input_transition,transition,auxiliary_transition))
    def complete():
        return scheduler.progress>=1. if goal else scheduler.inputs>=recipe['total_inputs']
    try:
        while not complete() and phase_step < args.steps and budget.reason is None:
            started = time.monotonic()
            preparation_rng = _rng_state()
            preparation_boundary=stream.state_dict()
            try:
                input_budget = recipe['input_budget'] if goal else min(recipe['input_budget'],recipe['total_inputs']-scheduler.inputs)
                if recipe.get('prepare_updates', False):
                    documents,inputs,body,control,aux_count,aux_space_count = stream.take(input_budget)
                else:
                    documents, inputs, body, control = collect_documents(stream,input_budget)
                    aux_count=annotate_documents(documents,auxiliary_catalog) if auxiliary_catalog is not None else None
                    aux_space_count=space_target_count(documents) if auxiliary_catalog is not None else None
                    from src.st.visual_grounding_auxiliary import annotate_learning
                    annotate_learning(documents,recipe,dataset.drawings.classes)
            except Exception as error:
                # 2026-10-11：stream已回滚到批前；保存最后成功更新，避免再丢一小时。
                _restore_rng(preparation_rng)
                stream.rewind(preparation_boundary)
                save_formal(output/'data_error.pt',model,optimizer,scheduler,scaler,stream,
                    step=step,phase_step=phase_step,counters=counters,data_args=data_args,
                    manifest=manifest,recipe=recipe,ae_source=ae_source,parent_source=parent_source)
                write_json(output/'data_error.json',{'step':step,'successful_inputs':scheduler.inputs,
                    'error':repr(error),'data':stream.state_dict(),'checkpoint':'data_error.pt'})
                raise
            prepared = time.monotonic()
            prepare_seconds = prepared-started
            if model.text_aux is not None and auxiliary_calibration_due(recipe,scheduler.inputs):
                try:
                    calibration=calibrate_auxiliary(model,documents,body,control,recipe,device,
                                                   task_loss,gpu_batch,_rng_state,_restore_rng)
                    calibration=apply_auxiliary_calibration(recipe,calibration,inputs=scheduler.inputs,step=step)
                    cfg.r_text_aux_weight=recipe['text_aux_alpha']
                except Exception as error:
                    _restore_rng(preparation_rng);stream.rewind(preparation_boundary)
                    save_formal(output/'calibration_error.pt',model,optimizer,scheduler,scaler,stream,
                        step=step,phase_step=phase_step,counters=counters,data_args=data_args,
                        manifest=manifest,recipe=recipe,ae_source=ae_source,parent_source=parent_source)
                    write_json(output/'calibration_error.json',{'step':step,'successful_inputs':scheduler.inputs,
                        'error':repr(error),'checkpoint':'calibration_error.pt'})
                    raise
                write_json(output/'text_aux_calibration.json',calibration)
                with (output/'text_aux_calibration.jsonl').open('a',encoding='utf-8') as log:
                    log.write(json.dumps(calibration,ensure_ascii=False)+'\n')
                write_json(output/'recipe.json',recipe)
                calibration_checkpoint_pending=True
                print(json.dumps({'text_aux_calibration':calibration}),flush=True)
                started=time.monotonic()-prepare_seconds
            if need_calibration:
                report = calibrate_capacity(model,documents,body,control,recipe,device,aux_count,aux_space_count)
                write_json(output/'capacity.json',report)
                write_json(output/'recipe.json',recipe)
                print(json.dumps({'capacity':report}),flush=True)
                need_calibration = False
                # 容量试算不计入持续训练速度；正文准备时间保留。
                started = time.monotonic()-prepare_seconds
            optimizer.zero_grad(set_to_none=True)
            actual_inputs = sum(inputs.values())
            if recipe.get('visual_grounding') and not recipe.get('anneal_start_saved') and scheduler.inputs+actual_inputs>=recipe['total_inputs']-recipe['decay_inputs']:
                recipe['anneal_start_saved']=True
                # 文档已经准备，固定快照须保存消费前位置，以便从这里完整恢复。
                save_formal(output/'anneal_start.pt',model,optimizer,scheduler,scaler,stream,
                        step=step,phase_step=phase_step,counters=counters,data_args=data_args,manifest=manifest,
                        recipe=recipe,ae_source=ae_source,parent_source=parent_source,
                        data_state=preparation_boundary,rng_state=preparation_rng)
            coverage=stream.coverage(coverage_limits) if coverage_limits else None
            if goal:scheduler.prepare_update(actual_inputs,coverage['progress'])
            else:scheduler.prepare_update(actual_inputs)
            model.observe_distribution = phase_step < 3 or (phase_step+1) % args.log_every == 0
            model.observe_auxiliary = model.observe_distribution
            sums,total_loss,finite,microbatches = backward_documents(
                model,optimizer,scaler,documents,body,control,recipe,device,
                partial(task_loss,auxiliary_count=aux_count,auxiliary_space_count=aux_space_count),gpu_batch,TASKS,_rng_state,_restore_rng)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.r_pred.parameters(),recipe['clip_norm'])
            aux_norm=(torch.nn.utils.clip_grad_norm_(model.text_aux.parameters(),recipe['clip_norm'])
                      if model.text_aux is not None else norm.new_zeros(()))
            grounding_norm=(torch.nn.utils.clip_grad_norm_(model.visual_grounding.parameters(),recipe['clip_norm'])
                            if model.visual_grounding is not None else norm.new_zeros(()))
            old_scale = scaler.get_scale()
            if finite and bool(torch.isfinite(norm)) and bool(torch.isfinite(aux_norm)) and bool(torch.isfinite(grounding_norm)):
                scaler.step(optimizer)
                scaler.update()
                success = scaler.get_scale() >= old_scale
            else:
                scaler.update(new_scale=old_scale/2)
                success = False
            counters['attempts'] += 1
            counters['read_inputs'] += actual_inputs
            if success:
                if goal:scheduler.step(actual_inputs,coverage['progress'])
                else:scheduler.step(actual_inputs)
                step += 1
                phase_step += 1
                counters['successful_inputs'] += actual_inputs
                counters['successful_body'] += sum(body.values())
                counters['successful_control'] += control
                for task in TASKS:
                    counters['inputs:'+task] += inputs[task]
                    counters['body:'+task] += body[task]
                counters['consecutive_skips'] = 0
            else:
                counters['skipped_updates'] += 1
                counters['consecutive_skips'] += 1
                if goal or recipe.get('retry_failed_documents'):
                    _restore_rng(preparation_rng);stream.rewind(preparation_boundary)
            torch.cuda.synchronize(device)
            elapsed = time.monotonic()-started
            if success:
                throughput_inputs += actual_inputs
                throughput_seconds += elapsed
                throughput_updates += 1
                counters['training_seconds']+=elapsed
                rate = throughput_inputs/throughput_seconds
                write_json(output/'throughput.json', {
                    'updates':throughput_updates, 'inputs':throughput_inputs,
                    'seconds':throughput_seconds, 'inputs_per_second':rate,
                    'successful_inputs':scheduler.inputs,
                    'remaining_inputs':None if goal else max(0,recipe['total_inputs']-scheduler.inputs),
                    'estimated_remaining_gpu_hours':((counters['training_seconds']/scheduler.progress*
                        (1-scheduler.progress)/3600 if scheduler.progress>0 else None)
                        if goal else max(0,recipe['total_inputs']-scheduler.inputs)/rate/3600),
                    'coverage':stream.coverage(coverage_limits) if coverage_limits else None,
                    'includes_data_preparation':True, 'excludes_capacity_trials':True})
            if not success or phase_step <= 3 or phase_step % args.log_every == 0 or budget.reason:
                print(json.dumps({'step':step,'phase_step':phase_step,'success':success,'loss':total_loss,
                    'coverage':stream.coverage(coverage_limits) if coverage_limits else None,
                    'inputs':actual_inputs,'successful_inputs':scheduler.inputs,'task_inputs':dict(inputs),
                    'body_targets':dict(body),'control_targets':control,
                    'content_nll':sums['content_sum']/max(1,sum(body.values())),
                    'control_nll':sums['control_sum']/max(1,control),
                    'task_nll':{task:sums[task]/body[task] if body[task] else None for task in TASKS},
                    'gmm_buckets':observation_report(sums),
                    'microbatches':microbatches,'prepare_seconds':prepare_seconds,'update_seconds':elapsed,
                    'r_padding_ratio':sums['r_padded_inputs']/actual_inputs,
                    'e_point_padding_ratio':sums['e_point_slots']/max(1,sums['e_valid_points']),
                    'inputs_per_second':actual_inputs/elapsed,'lr':optimizer.param_groups[0]['lr'],
                    'scale':scaler.get_scale(),'grad_norm':float(norm),'peak_memory_bytes':torch.cuda.max_memory_allocated(device),
                    'text_aux':auxiliary_report(sums) if model.text_aux is not None else None,
                    'text_aux_alpha':recipe.get('text_aux_alpha'),'text_aux_grad_norm':float(aux_norm),
                    'grounding':{k:v for k,v in sums.items() if k.startswith('ground_')},
                    'grounding_alpha':recipe.get('grounding_alpha'),'grounding_grad_norm':float(grounding_norm),
                    'reserved_memory_bytes':torch.cuda.max_memory_reserved(device),
                    'padded_stroke_budget':recipe['padded_stroke_budget'],
                    'device_free_gib':torch.cuda.mem_get_info(device)[0]/2**30,
                    'memory_headroom_gib':recipe.get('memory_headroom_gib'),
                    'skipped_documents':dict(stream.skipped)}),flush=True)
            checkpoint_kwargs = dict(step=step,phase_step=phase_step,counters=counters,
                data_args=data_args,manifest=manifest,recipe=recipe,ae_source=ae_source,parent_source=parent_source)
            valid_every=recipe.get('text_aux_validation_every_inputs',20000000)
            if validation_path and success and budget.reason is None and (
                scheduler.inputs-recipe.get('text_aux_last_valid_inputs',0)>=valid_every or
                complete()):
                valid=validate_auxiliary(model,dataset.font,auxiliary_catalog,validation_path,
                    manifest.get('page_layout',{}),device,_rng_state,_restore_rng)
                valid.update(step=step,successful_inputs=scheduler.inputs)
                write_json(output/f'validation_{step:07d}.json',valid)
                with (output/'validation.jsonl').open('a',encoding='utf-8') as log:
                    log.write(json.dumps(valid,ensure_ascii=False)+'\n')
                recipe['text_aux_last_valid_inputs']=scheduler.inputs
                print(json.dumps({'text_aux_validation':valid}),flush=True)
            if counters['consecutive_skips'] >= 20:
                save_formal(output/'numeric_stop.pt',model,optimizer,scheduler,scaler,stream,**checkpoint_kwargs)
                raise RuntimeError('连续20次FP16溢出；状态已保存')
            if (success and (calibration_checkpoint_pending or (model.text_aux is not None and phase_step==1))) or time.monotonic()-last_save >= recipe['save_seconds']:
                save_formal(output/f'ckpt_{step:07d}.pt',model,optimizer,scheduler,scaler,stream,**checkpoint_kwargs)
                calibration_checkpoint_pending=False
                last_save = time.monotonic()
        save_formal(output/'final.pt',model,optimizer,scheduler,scaler,stream,step=step,
            phase_step=phase_step,counters=counters,data_args=data_args,manifest=manifest,
            recipe=recipe,ae_source=ae_source,parent_source=parent_source)
        budget.finish(output,step,completed_reason=("data_coverage" if goal else "input_budget") if complete() else "steps")
        print(json.dumps({'formal_saved':str(output/'final.pt'),'successful_inputs':scheduler.inputs,
                          'input_budget_complete':complete()}),flush=True)
    finally:
        stream.close()


if __name__ == '__main__':
    main()
