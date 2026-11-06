"""2026-10-08：用户指定整卡空余6GiB，按完整恢复状态放大正式R微批。"""
from collections import Counter
import gc
import json
import time

import torch

from .formal_stream import iter_microbatches


def limit_cuda_memory(device, headroom_gib):
    torch.cuda.empty_cache()
    free,total = torch.cuda.mem_get_info(device)
    outside = max(0,total-free-torch.cuda.memory_reserved(device))
    limit = total-outside-int(headroom_gib*2**30)
    if limit <= 0:
        raise ValueError('GPU空余不足以保留指定显存')
    index = device if isinstance(device,int) else torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(limit/total,index)
    return {'device_bytes':total,'outside_allocator_bytes':outside,
            'allocator_limit_bytes':limit,'headroom_gib':headroom_gib}


def calibrate_headroom(model, documents, body, control, recipe, device, loss_fn, move_batch, rng_state, restore_rng):
    original_budget = recipe['padded_stroke_budget']
    rng = rng_state()
    trials = []
    quantum = 1024
    lower,upper = 0,recipe['microbatch_documents']*model.cfg.max_context_strokes+quantum
    candidate = min(original_budget,upper-quantum)
    for _ in range(12):
        model.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
        limits = limit_cuda_memory(device,recipe['memory_headroom_gib'])
        torch.cuda.reset_peak_memory_stats(device)
        recipe['padded_stroke_budget'] = candidate
        started = time.monotonic()
        result = loss = None
        repeat = False
        try:
            microbatches = 0
            for cpu in iter_microbatches(documents,recipe,max_strokes=model.cfg.max_context_strokes,pin=True):
                with torch.autocast('cuda',dtype=torch.float16):
                    result = model(move_batch(cpu,device))
                    loss = loss_fn(result,body,control,recipe)
                loss.backward()
                microbatches += 1
                result = loss = None
            torch.cuda.synchronize(device)
            free,total = torch.cuda.mem_get_info(device)
            outside = max(0,total-free-torch.cuda.memory_reserved(device))
            peak_reserved = torch.cuda.max_memory_reserved(device)
            headroom = (total-outside-peak_reserved)/2**30
            fits = headroom >= recipe['memory_headroom_gib']
            trials.append({'padded_stroke_budget':candidate,'activation_checkpoint':model.activation_checkpoint,
                           'seconds':time.monotonic()-started,'microbatches':microbatches,
                           'allocated_bytes':torch.cuda.max_memory_allocated(device),'reserved_bytes':peak_reserved,
                           'physical_peak_headroom_gib':headroom,'oom':False,'fits_headroom':fits})
            if fits:
                lower = candidate
            elif outside > limits['outside_allocator_bytes']:
                # 初次cuBLAS等外部分配完成后，刷新额度并复测同一候选。
                repeat = True
            else:
                upper = candidate
        except torch.cuda.OutOfMemoryError:
            trials.append({'padded_stroke_budget':candidate,'oom':True,'fits_headroom':False})
            upper = candidate
        finally:
            result = loss = None
            model.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
            restore_rng(rng)
        if repeat:
            continue
        if upper-lower <= quantum:
            break
        candidate = max(quantum,((lower+upper)//(2*quantum))*quantum)
    viable = [trial for trial in trials if trial.get('fits_headroom')]
    if not viable:
        recipe['padded_stroke_budget'] = original_budget
        raise RuntimeError('指定显存余量下没有可用微批')
    selected = max(viable,key=lambda trial:trial['padded_stroke_budget'])
    recipe['padded_stroke_budget'] = selected['padded_stroke_budget']
    limits = limit_cuda_memory(device,recipe['memory_headroom_gib'])
    torch.cuda.reset_peak_memory_stats(device)
    return {'trials':trials,'selected':selected,'memory_limit':limits,
            'device':torch.cuda.get_device_name(device),'policy':'largest_microbatch_with_headroom'}


def backward_documents(model, optimizer, scaler, documents, body, control, recipe, device,
                       loss_fn, move_batch, tasks, rng_state, restore_rng):
    rng = rng_state()
    while True:
        optimizer.zero_grad(set_to_none=True)
        sums,scalar_names,scalar_values,microbatches = Counter(),[],[],0
        def accumulate(name, value):
            scalar_names.append(name)
            scalar_values.append(value.detach())
        result = loss = None
        try:
            for cpu in iter_microbatches(documents,recipe,max_strokes=model.cfg.max_context_strokes,pin=True):
                if isinstance(cpu,dict) and 'ink' in cpu:
                    sums['r_padded_inputs']+=cpu['stroke_pad'].shape[0]*(cpu['stroke_pad'].shape[1]-1)
                    sums['e_point_slots']+=sum((end-start)*width for start,end,width in
                        (cpu.get('encode_groups') or ((0,len(cpu['ink']),cpu['ink'].shape[1]),)))
                    sums['e_valid_points']+=int(cpu['stroke_len'].sum())
                with torch.autocast('cuda',dtype=torch.float16):
                    result = model(move_batch(cpu,device))
                    loss = loss_fn(result,body,control,recipe)
                accumulate('_nonfinite', ~torch.isfinite(loss))
                scaler.scale(loss).backward()
                accumulate('_loss', loss)
                for name in ('content_sum','control_sum'):
                    accumulate(name, result[name])
                for i,task in enumerate(tasks):
                    accumulate(task, result['task_sums'][i])
                sums.update(result.get('distribution_sums',{}))
                for name,value in result.items():
                    if name.startswith(('aux_','ground_')) and torch.is_tensor(value):
                        accumulate(name, value)
                microbatches += 1
                result = loss = None
            values=torch.stack(scalar_values).cpu().tolist()
            for name,value in zip(scalar_names,values):sums[name]+=value
            return sums, sums.pop('_loss'), sums.pop('_nonfinite') == 0, microbatches
        except torch.cuda.OutOfMemoryError:
            previous = recipe['padded_stroke_budget']
            if previous <= 4096:
                raise
            recipe['padded_stroke_budget'] = max(4096,(int(previous*.9)//1024)*1024)
            print(json.dumps({'capacity_oom_retry':True,'previous_budget':previous,
                              'padded_stroke_budget':recipe['padded_stroke_budget'],
                              'same_documents':True}),flush=True)
            result = loss = None
            scalar_values.clear()
            optimizer.zero_grad(set_to_none=True)
            gc.collect(); torch.cuda.empty_cache()
            restore_rng(rng)
