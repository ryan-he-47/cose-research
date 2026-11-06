"""2026-11-05获批阶段的来源、损失尺度及成功更新块长时钟。"""
from copy import deepcopy
import hashlib
import math

import numpy as np
import torch

from .formal_recipe import TASKS


VERSION = "page_diffusion_balance_20261105_v1"


def _digest(values):
    return hashlib.sha256(np.asarray(values,dtype="<f4").tobytes()).hexdigest()


def _check_shares(shares):
    if set(shares) != set(TASKS) or any(not math.isfinite(x) or x <= 0 for x in shares.values()):
        raise ValueError("八项任务份额须齐全且为有限正值")
    if not math.isclose(sum(shares.values()),1.,abs_tol=1e-9):
        raise ValueError("任务份额合计须为1")


def validate_balance(recipe):
    policy = recipe.get("balance")
    if policy is None:
        if recipe["model"].get("training_blocks"):
            raise ValueError("混合块训练检查点缺少配方")
        return
    if policy["version"] != VERSION:
        raise ValueError("不支持的损失平衡版本")
    _check_shares(policy["target_shares"])
    _check_shares(policy["input_shares"])
    if recipe["input_shares"] != policy["input_shares"]:
        raise ValueError("恢复曝光与阶段配方不一致")
    widths = policy["width_cycle"]
    if (not widths or any(type(b) is not int or b <= 0 or policy["supervision_block"] % b for b in widths)
            or sorted(set(widths)) != list(recipe["model"]["training_blocks"])
            or recipe["model"]["block"] != policy["sampling_block"]
            or policy["calibration_updates"] < len(widths) or policy["check_updates"] < len(widths)
            or policy["calibration_updates"] % len(widths) or policy["check_updates"] % len(widths)):
        raise ValueError("块长周期、定标周期或模型配置不一致")
    if not (0 <= policy["t_full"] < policy["t_zero"] <= 1 and 0 <= policy["tail_extra"] <= 1
            and policy["shape_tasks"] == ["T1","S1"] and math.isfinite(policy["body_scale"]) and policy["body_scale"]>0):
        raise ValueError("文字度量或正文标尺无效")
    axes = np.asarray(policy["translation_axes"],dtype=np.float32)
    if (axes.shape != (recipe["model"]["latent"],2) or _digest(axes) != policy["axes_sha256"]
            or not np.allclose(axes.T@axes,np.eye(2),atol=2e-6)):
        raise ValueError("平移基指纹或正交性不一致")
    if policy["additional_inputs"] != 1_000_000_000 or recipe["total_inputs"] != policy["parent_inputs"]+policy["additional_inputs"]:
        raise ValueError("新阶段的全局预算终点不一致")
    if "reference_mse" in policy:
        weights = calibrated_weights(policy,policy["reference_mse"])
        if recipe["loss_weights"] != weights:
            raise ValueError("训练系数与已保存的定标结果不同")


def enter_balanced_stage(parent, spec, parent_hash, parent_path):
    """只替换新阶段的元数据，保留完整父状态中的参数、优化器、随机源和源游标。"""
    if (parent_hash != spec["parent_sha256"] or parent["kind"] != "page_diffusion_training_v2"
            or parent["step"] != spec["parent_step"] or parent["counters"]["successful_inputs"] != spec["parent_inputs"]
            or parent["scheduler"]["inputs"] != spec["parent_inputs"]):
        raise ValueError("起点与批准的稳定段完整检查点不符")
    old = parent["scheduler"]["spec"]
    inputs = spec["parent_inputs"]
    saved = parent["source"].get("stable_checkpoints",[])
    if (old["total_inputs"] != spec["parent_target_inputs"] or old["peak_lr"] != .0003
            or old["decay_inputs"] != 72000000 or inputs > old["total_inputs"]-old["decay_inputs"]
            or not any(x["step"]==parent["step"] and x["successful_inputs"]==inputs and
                       x["target_inputs"]==old["total_inputs"] for x in saved)):
        raise ValueError("父WSD时钟或稳定段留档记录不符")
    if not parent["optimizer"]["state"] or any(int(s["step"])!=parent["step"] for s in parent["optimizer"]["state"].values()):
        raise ValueError("父Adam状态不完整")
    if any(group["lr"]!=old["peak_lr"] for group in parent["optimizer"]["param_groups"]):
        raise ValueError("父优化器未处于稳定学习率")
    coordinate = torch.cat((parent["model"]["latent_mean"],parent["model"]["latent_scale"].reshape(1))).cpu().numpy()
    if parent["source"]["ae_sha256"] != spec["ae_sha256"] or _digest(coordinate) != spec["coordinate_sha256"]:
        raise ValueError("冻结AE或实际坐标与平移基的来源不同")
    if parent["model_cfg"]["block"] != spec["supervision_block"] or parent["model_cfg"] != parent["recipe"]["model"]:
        raise ValueError("父模型块宽或配置不一致")
    policy = deepcopy(spec)
    recipe = deepcopy(parent["recipe"])
    recipe.update(version=VERSION, approval="approved_20261105_balanced_plus_1b", balance=policy,
                  total_inputs=inputs+spec["additional_inputs"], input_shares=dict(spec["input_shares"]),
                  loss_weights=dict(spec["target_shares"]))
    recipe["model"].update(block=spec["sampling_block"],training_blocks=sorted(set(spec["width_cycle"])))
    scheduler = deepcopy(parent["scheduler"])
    scheduler["spec"]["total_inputs"] = recipe["total_inputs"]
    source = deepcopy(parent["source"])
    source["balance_origin"] = dict(checkpoint=str(parent_path),sha256=parent_hash,step=parent["step"],
        inputs=inputs,target_inputs=recipe["total_inputs"],parent_stream=deepcopy(parent["stream"]),
        prior_wsd=deepcopy(old),new_wsd=deepcopy(scheduler["spec"]))
    validate_balance(recipe)
    return dict(parent, recipe=recipe, model_cfg=deepcopy(recipe["model"]), scheduler=scheduler, source=source)


def block_clock(recipe, completed_step):
    policy = recipe["balance"]
    completed = completed_step-policy["parent_step"]
    if completed < 0:
        raise ValueError("块长时钟早于新阶段起点")
    widths = policy["width_cycle"]
    cycle, offset = divmod(completed,len(widths))
    rng = torch.Generator(device="cpu").manual_seed(policy["width_seed"]+cycle)
    index = torch.randperm(len(widths),generator=rng).tolist()[offset]
    return dict(completed_updates=completed,cycle=cycle,offset=offset,next_width=widths[index])


def calibrated_weights(policy, references):
    if set(references)!=set(TASKS) or any(not math.isfinite(v) or v<=0 for v in references.values()):
        raise ValueError("各任务的参考误差必须为有限正值")
    return {task:policy["body_scale"]*policy["target_shares"][task]/references[task] for task in TASKS}


def bind_axes(model,recipe):
    if "balance" in recipe:
        axes = model.latent_mean.new_tensor(recipe["balance"]["translation_axes"])
        model.register_buffer("balance_axes",axes,persistent=False)


def stage_metrics(recipe,counts,metrics):
    raw = {t:metrics["body_sum:"+t]/n for t,n in counts["body"].items() if n}
    metric = {t:metrics["metric_sum:"+t]/n for t,n in counts["body"].items() if n}
    terms = {t:recipe["loss_weights"][t]*value for t,value in metric.items()}
    body = sum(terms.values())
    return dict(body_terms=terms,body_share={t:v/body if body else 0. for t,v in terms.items()},
                raw_body_mse=raw,training_body_mse=metric,
                weighted_body=body)


def window_metrics(rows):
    terms = {task:sum(r["body_terms"].get(task,0.) for r in rows) for task in TASKS}
    inputs = {task:sum(r["input_counts"].get(task,0) for r in rows) for task in TASKS}
    widths = {}
    for width in sorted({r["native_block"] for r in rows}):
        selected = [r for r in rows if r["native_block"]==width]
        seconds = sum(r["seconds"]+r["preparation_seconds"] for r in selected)
        headroom = [r["physical_peak_headroom_gib"] for r in selected if r["physical_peak_headroom_gib"] is not None]
        widths[width] = dict(updates=len(selected),
            inputs_per_second=sum(sum(r["input_counts"].values()) for r in selected)/seconds,
            body_targets_per_second=sum(sum(r["counts"]["body"].values()) for r in selected)/seconds,
            min_headroom_gib=min(headroom) if headroom else None,
            clipped_updates=sum(r["grad_clipped"] for r in selected),
            oom_retries=sum(r["oom_retries"] for r in selected))
    total_body, total_inputs = sum(terms.values()), sum(inputs.values())
    return dict(first_step=rows[0]["step"],last_step=rows[-1]["step"],updates=len(rows),
        input_share={t:v/total_inputs for t,v in inputs.items()},
        body_share={t:v/total_body if total_body else 0. for t,v in terms.items()},
        mean_body_loss=total_body/len(rows),mean_control_loss=sum(r["weighted_control"] for r in rows)/len(rows),
        widths=widths)
