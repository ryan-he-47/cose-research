"""在阶段起点用新曝光、原生块与字形度量作只读定标，再恢复正式消费边界。"""
from collections import Counter
import gc
import json
import math

import torch

from .formal_recipe import TASKS
from .page_diffusion_balance import block_clock, calibrated_weights, validate_balance
from .page_diffusion_data import iter_diffusion_batches, prepare_update
from .page_diffusion_runtime import _restore_rng, _rng_state, forward_batch, shrink_microbatch


def _batch_sums(model,codec,docs,counts,recipe,device):
    sums = torch.zeros(2,len(TASKS),device=device)
    for batch in iter_diffusion_batches(docs,recipe):
        result = forward_batch(model,codec,batch,counts,recipe,device)["metrics"]
        sums += torch.stack([result[prefix+task] for prefix in ("body_sum:","metric_sum:") for task in TASKS]).reshape_as(sums)
    if not torch.isfinite(sums).all():
        raise FloatingPointError("定标误差非有限")
    return sums.cpu().tolist()


@torch.inference_mode()
def calibrate(model,codec,stream,noise,recipe,device,*,should_stop=lambda:False):
    policy = recipe["balance"]
    if "reference_mse" in policy:
        raise ValueError("已有正式标尺，普通恢复直接使用已保存的系数")
    boundary = stream.state_dict()
    rng_before = _rng_state()
    calibration_noise = torch.Generator(device="cpu")
    calibration_noise.set_state(noise.get_state())
    was_training = model.training
    records = []
    oom_retries = 0
    fit_count = policy["calibration_updates"]
    number = fit_count+policy["check_updates"]
    model.eval()
    try:
        for index in range(number):
            if should_stop():
                raise InterruptedError("定标收到停止请求")
            docs,inputs,*_ = stream.take(recipe["input_budget"])
            width = block_clock(recipe,policy["parent_step"]+index)["next_width"]
            counts = prepare_update(docs,calibration_noise,width,supervision_block=policy["supervision_block"])
            if any(counts["body"].get(t,0)<=0 for t in TASKS):
                raise ValueError("定标完整更新中有任务缺少正文目标")
            while True:
                try:
                    raw,metric = _batch_sums(model,codec,docs,counts,recipe,device)
                    break
                except torch.OutOfMemoryError:
                    shrink_microbatch(recipe)
                    oom_retries += 1
                gc.collect()
                if device.type=="cuda":
                    torch.cuda.empty_cache()
            records.append(dict(width=width,inputs=dict(inputs),body=dict(counts["body"]),
                raw_mse={t:v/counts["body"][t] for t,v in zip(TASKS,raw)},
                metric_mse={t:v/counts["body"][t] for t,v in zip(TASKS,metric)}))
            if (index+1)%len(policy["width_cycle"])==0:
                print(json.dumps(dict(phase="balance_calibration",measured=index+1,total=number)),flush=True)
    finally:
        model.train(was_training)
        try:
            stream.rewind(boundary)
        finally:
            _restore_rng(rng_before)
    references = {t:sum(r["metric_mse"][t] for r in records[:fit_count])/fit_count for t in TASKS}
    weights = calibrated_weights(policy,references)
    held = records[fit_count:]
    terms = {t:weights[t]*sum(r["metric_mse"][t] for r in held)/len(held) for t in TASKS}
    total = sum(terms.values())
    if not math.isfinite(total) or total<=0 or stream.state_dict()!=boundary:
        raise RuntimeError("定标核验或消费边界回退失败")
    policy["reference_mse"] = references
    recipe["loss_weights"] = weights
    validate_balance(recipe)
    return dict(version=policy["version"],fit_updates=fit_count,check_updates=len(held),
        measured_inputs=sum(sum(r["inputs"].values()) for r in records),
        fit_widths=dict(Counter(r["width"] for r in records[:fit_count])),
        check_widths=dict(Counter(r["width"] for r in held)),body_scale=policy["body_scale"],
        reference_mse=references,loss_weights=weights,heldout_body_loss=total,
        heldout_share={t:v/total for t,v in terms.items()},calibration_oom_retries=oom_retries,
        committed_inputs=0,optimizer_updates=0,cursor_restored=True,measurements=records)
