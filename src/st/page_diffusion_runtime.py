"""块扩散的冻结编码、更新、EMA和完整恢复；不改变旧训练入口的计算。"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import time

import torch

from .formal_prefetch import PreparedUpdates
from .formal_stream import InputDocumentStream
from .page_diffusion import PageDiffusionConfig, PageDiffusionR
from .page_diffusion_balance import bind_axes, block_clock, validate_balance
from .page_diffusion_data import iter_diffusion_batches, diffusion_inputs, losses, prepare_update
from .pipeline_pretrain import build_document_dataset
from .train_pretrain import _rng_state, _restore_rng


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def create_stream(data_args, recipe, state=None, *, new_exposure=False):
    dataset = build_document_dataset(**data_args)
    stream = InputDocumentStream(dataset, data_args, recipe["input_shares"],
        max_strokes=recipe["context"], workers=recipe["workers"],
        prefetch_documents=recipe.get("prefetch_documents"),
        prepare_batch_documents=recipe.get("prepare_batch_documents"))
    if new_exposure:
        stream.continue_positions(state)
    elif state is not None:
        stream.load_state_dict(state)
    # 扩散使用自己的双流打包；旧单流批张量既不读取，也无需预先构造。
    return PreparedUpdates(stream, recipe, None, pin=torch.cuda.is_available(), prepare_microbatches=False)


@torch.no_grad()
def estimate_normalization(codec, data_args, recipe, device):
    """独立的有限训练流，只统计真实正文，既不推进正式游标也不消耗正式噪声源。"""
    from .formal_stream import iter_microbatches
    count, sum_x, sum_sq = 0, torch.zeros(48, dtype=torch.float64, device=device), torch.zeros((), dtype=torch.float64, device=device)
    probe_recipe = {**recipe, "input_budget": min(recipe["input_budget"], 16384)}
    stream = create_stream(data_args, probe_recipe)
    target_count = recipe["normalization_targets"]
    try:
        while count < target_count:
            documents, *_ = stream.take(probe_recipe["input_budget"])
            for batch in iter_microbatches(documents, probe_recipe, max_strokes=recipe["context"], pin=False):
                batch = move(batch, device)
                with autocast(device):
                    latent = codec.encode_valid(batch)
                valid = (~batch["stroke_pad"]) & (batch["targetable"] > .5) & (batch["kind_id"] < 3)
                values = latent[valid][:target_count-count].double()
                count += len(values)
                sum_x += values.sum(0)
                sum_sq += values.square().sum()
                if count >= target_count:
                    break
    finally:
        stream.close()
    mean = sum_x/count
    scale = (sum_sq/(count*48)-mean.square().mean()).clamp_min(0).sqrt()
    if not torch.isfinite(mean).all() or not torch.isfinite(scale) or scale <= 0:
        raise ValueError("冻结AE标准化统计无效")
    return dict(count=count, mean=mean.float().cpu().tolist(), scale=float(scale),
                source="train_real_body_only", data_args=data_args)


def validate_recipe(recipe):
    obsolete = {"text_aux_alpha", "text_aux_gradient_ratio", "calibration_interval", "calibrated_inputs"} & recipe.keys()
    if obsolete:
        raise ValueError(f"无分类辅助配方不能包含旧辅助字段：{sorted(obsolete)}")
    validate_balance(recipe)
    return PageDiffusionConfig(**recipe["model"])


def new_training_state(recipe, device):
    cfg = validate_recipe(recipe)
    torch.manual_seed(recipe["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(recipe["seed"])
    model = PageDiffusionR(cfg).to(device)
    bind_axes(model,recipe)
    decay, other = [], []
    for name, parameter in model.named_parameters():
        (other if parameter.ndim < 2 or name.endswith("bias") else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": .1},
                                   {"params": other, "weight_decay": 0.}],
                                  lr=recipe["peak_lr"], betas=(.9, .95), eps=1e-8, foreach=False)
    # 在首个前向前分配完整状态，避免optimizer.step中途才因状态分配OOM。
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer.state[parameter] = dict(step=torch.tensor(0.),
                exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = {k: value.detach().clone() for k, value in model.state_dict().items()}
    return model, optimizer, scaler, ema


@torch.no_grad()
def update_ema(model, ema, decay):
    parameters = dict(model.named_parameters())
    for key, value in model.state_dict().items():
        if key in parameters:
            ema[key].lerp_(value, 1-decay)
        else:
            ema[key].copy_(value)


def initialize_coordinates(model, ema, normalization, eop):
    with torch.no_grad():
        model.latent_mean.copy_(torch.tensor(normalization["mean"], device=model.latent_mean.device))
        model.latent_scale.fill_(normalization["scale"])
        model.eop_latent.copy_(eop)
        for key in ("latent_mean", "latent_scale", "eop_latent"):
            ema[key].copy_(model.state_dict()[key])


def forward_batch(model, codec, batch, counts, recipe, device):
    codec.encode_chunk_size = int(recipe.get("encode_chunk_size", 256))
    batch = move(batch, device)
    with autocast(device):
        with torch.profiler.record_function("diffusion.frozen_encoder"):
            encoded = codec.encode_valid(batch)
        values, target = diffusion_inputs(model, encoded, batch)
        with torch.profiler.record_function("diffusion.backbone"):
            output = model(values, batch["pages"], batch["local"], batch["roles"],
                           batch["times"], batch["allowed"], clean_size=batch["clean_size"],
                           attention_layout=recipe.get("attention_layout", "dense"))
    # 扩散正文与控制符的均方误差使用FP32。
    return losses(model, output, target, batch, counts, recipe)


def shrink_microbatch(recipe):
    if recipe.get("memory_policy") == "soft" and recipe["padded_stroke_budget"] > 1024:
        recipe["padded_stroke_budget"] = max(1024, recipe["padded_stroke_budget"]//2)
    elif recipe["microbatch_documents"] > 1:
        recipe["microbatch_documents"] = max(1, recipe["microbatch_documents"]//2)
    elif recipe["padded_stroke_budget"] > 1024:
        recipe["padded_stroke_budget"] = max(1024, recipe["padded_stroke_budget"]//2)
    else:
        raise RuntimeError("单文档最低微批仍超显存，停止并保留恢复点")


def summarize_profile(profile):
    rows = []
    for item in profile.key_averages():
        rows.append(dict(name=item.key, calls=item.count,
            self_device_us=getattr(item, "self_device_time_total", 0.),
            device_us=getattr(item, "device_time_total", 0.), self_cpu_us=item.self_cpu_time_total))
    return dict(scope="first_microbatch_of_update", torch=torch.__version__,
        top_device=sorted(rows, key=lambda row:row["self_device_us"], reverse=True)[:25],
        attention=[r for r in rows if "attention" in r["name"]],
        phases=[r for r in rows if r["name"].startswith("diffusion.")])


def train_update(model, codec, optimizer, scaler, ema, documents, counts, recipe, device, *, profile_path=None):
    if "balance" in recipe and "reference_mse" not in recipe["balance"]:
        raise ValueError("新配方须完成只读定标后才能更新参数")
    start = time.monotonic()
    retry, oom_retries, metrics = 0, 0, Counter()
    while True:
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            free_start, total = torch.cuda.mem_get_info(device)
            outside = max(0, total-free_start-torch.cuda.memory_reserved(device))
            base_allocated = torch.cuda.memory_allocated(device)
        metrics.clear()
        accumulated, metric_names = None, None
        finite_losses = torch.ones((), dtype=torch.bool, device=device)
        microbatches, padded_positions = 0, 0
        try:
            for batch in iter_diffusion_batches(documents, recipe):
                capture = profile_path is not None and microbatches == 0 and retry == 0
                profiling = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) if capture else nullcontext()
                with profiling as profiler:
                    result = forward_batch(model, codec, batch, counts, recipe, device)
                    with torch.profiler.record_function("diffusion.backward"):
                        scaler.scale(result["total"]).backward()
                if capture:
                    write_json(profile_path, summarize_profile(profiler))
                    del profiler, profiling
                finite_losses.logical_and_(torch.isfinite(result["total"]).detach())
                values = {**result["metrics"], "loss":result["total"].detach()}
                names = tuple(values)
                if metric_names is not None and names != metric_names:
                    raise RuntimeError("微批日志字段不一致")
                metric_names = names
                vector = torch.stack([values[k].float() for k in names])
                accumulated = vector if accumulated is None else accumulated+vector
                microbatches += 1
                padded_positions += batch["pages"].numel()
                del result
            if not bool(finite_losses):
                raise FloatingPointError("loss非有限，整批未更新或消费预算")
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["clip_norm"])
            if not torch.isfinite(grad):
                scaler.update()
                retry += 1
                if retry > 8:
                    raise FloatingPointError("同批多次梯度溢出，未消费预算")
                continue
            if device.type == "cuda":
                free, total = torch.cuda.mem_get_info(device)
                outside = max(outside, total-free-torch.cuda.memory_reserved(device))
                if recipe.get("memory_policy", "strict") == "strict" and free < recipe.get("minimum_free_gib", 0.)*1024**3:
                    raise torch.OutOfMemoryError("整卡空余不足约定值")
            scaler.step(optimizer)
            scaler.update()
            update_ema(model, ema, recipe["ema"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            metrics.update(dict(zip(metric_names, accumulated.cpu().tolist())))
            return dict(metrics, seconds=time.monotonic()-start, grad_norm=float(grad), retries=retry,
                        runtime_profiled=profile_path is not None,
                        oom_retries=oom_retries,
                        microbatches=microbatches, padded_positions=padded_positions,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                        gpu_free_before_step_bytes=free_start if device.type == "cuda" else None,
                        gpu_free_after_backward_bytes=free if device.type == "cuda" else None,
                        gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                        gpu_base_allocated_bytes=base_allocated if device.type == "cuda" else None,
                        gpu_total_bytes=total if device.type == "cuda" else None,
                        gpu_outside_bytes=outside if device.type == "cuda" else None,
                        physical_peak_headroom_gib=(total-outside-torch.cuda.max_memory_reserved(device))/2**30 if device.type == "cuda" else None,
                        microbatch_documents=recipe["microbatch_documents"],
                        padded_stroke_budget=recipe["padded_stroke_budget"],
                        packing=recipe.get("packing", "legacy"),
                        activation_checkpoint=model.cfg.activation_checkpoint)
        except torch.OutOfMemoryError:
            # 丢弃本次临时计算图，同一文档/块/噪声保留；不推进optimizer/EMA/时钟。
            if "result" in locals():
                del result
            optimizer.zero_grad(set_to_none=True)
            # unscale_可能已发生，清理scaler本轮状态而不改变其比例。
            scaler =_reset_scaler_attempt(scaler)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            shrink_microbatch(recipe)
            retry += 1
            oom_retries += 1


def _reset_scaler_attempt(scaler):
    if scaler.is_enabled() and getattr(scaler, "_scale", None) is not None:
        # public update(new_scale) resets per-optimizer bookkeeping after an aborted attempt.
        scaler.update(new_scale=scaler.get_scale())
    return scaler


def save_checkpoint(path, *, model, codec, optimizer, scaler, ema, scheduler, stream,
                    noise_generator, recipe, data_args, normalization, source, step, counters,
                    boundary_state=None, inference_name="inference.pt"):
    path = Path(path)
    payload = dict(kind="page_diffusion_training_v2", model=model.state_dict(), model_cfg=model.configuration(),
        codec=codec.checkpoint(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), ema=ema,
        scheduler=scheduler.state_dict(), stream=stream.state_dict(), noise_rng=noise_generator.get_state(),
        rng=_rng_state(), recipe=deepcopy(recipe), data_args=data_args, normalization=normalization,
        source=source, step=step, counters=dict(counters))
    if boundary_state is not None:
        payload.update(stream=boundary_state["stream"], noise_rng=boundary_state["noise_rng"],
                       rng=boundary_state["rng"])
    if "balance" in recipe:
        payload["block_clock"] = block_clock(recipe,step)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    inference_path = path.parent/inference_name
    inference_temporary = inference_path.with_suffix(".tmp")
    torch.save(dict(kind="page_diffusion_inference_v2", model={k: v.detach().cpu() for k, v in ema.items()},
                    model_cfg=model.configuration(), codec=codec.checkpoint(), source=source,
                    step=step, successful_inputs=scheduler.inputs), inference_temporary)
    os.replace(inference_temporary, inference_path)
