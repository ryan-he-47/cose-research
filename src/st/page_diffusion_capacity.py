"""按成功的正式更新调整执行批量；6GiB是规划目标，不是失败判据。"""
import statistics

import torch

from .formal_capacity import limit_cuda_memory


def configure_memory(device, recipe):
    target = float(recipe["minimum_free_gib"])
    if recipe.get("memory_policy", "strict") != "soft":
        return limit_cuda_memory(device, target)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    outside = max(0, total-free-torch.cuda.memory_reserved(device))
    # 使用物理容量处理真实OOM；不把6GiB目标转成分配器硬上限。
    index = device if isinstance(device, int) else torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(1., index)
    return dict(policy="soft", device_bytes=total, outside_allocator_bytes=outside,
                allocator_limit_bytes=total, target_headroom_gib=target,
                soft_tolerance_gib=2.)


def adapt_batching(recipe, stats, inputs):
    """仅改变下一步的计算拆分；本步所有成功输入照常计入训练。"""
    if not recipe.get("adaptive_microbatch"):
        return None
    state = recipe.setdefault("runtime_batch_tuning", {"samples":[], "done":False,
        "warmup_remaining":int(recipe.get("batch_tuning_warmup", 1)), "history":[]})
    current = int(recipe["padded_stroke_budget"])
    target = float(recipe["minimum_free_gib"])
    free = float(stats["physical_peak_headroom_gib"])
    if stats.get("oom_retries", 0):
        state.update(done=True, samples=[])
        return dict(reason="actual_oom_shrunk_same_batch", selected_budget=current)
    if free < target-2.:
        # 当前更新已完成，不因余量不足丢弃；按实测峰值温和缩小下一步。
        base = stats["gpu_base_allocated_bytes"]
        available = stats["gpu_total_bytes"]-stats["gpu_outside_bytes"]
        ratio = max(.5, (available-target*2**30-base)/max(1,stats["gpu_peak_reserved_bytes"]-base))
        selected = max(1024, min(current-1024, int(current*ratio)//1024*1024))
        recipe["padded_stroke_budget"] = selected
        state.update(done=True, samples=[])
        return dict(reason="soft_target_adjust_next_update", before=current, after=selected, headroom_gib=free)
    if state["done"]:
        return None
    if stats.get("runtime_profiled"):
        return None
    if state.get("warmup_remaining", 0):
        state["warmup_remaining"] -= 1
        return None
    seconds = stats["seconds"]+stats.get("preparation_seconds", 0.)
    state["samples"].append(dict(rate=inputs/seconds, headroom=free))
    if len(state["samples"]) < int(recipe.get("batch_tuning_samples", 5)):
        return None
    rate = statistics.median(x["rate"] for x in state["samples"])
    free = min(x["headroom"] for x in state["samples"])
    state["samples"] = []
    packing = recipe.get("packing", "legacy")
    state.setdefault("history", []).append(dict(budget=current, packing=packing, rate=rate, headroom_gib=free))
    best = state.get("best", state.get("previous"))
    if best is None or rate > best["rate"]:
        best = dict(budget=current, packing=packing, rate=rate, headroom_gib=free)
    state["best"] = best
    compare_packing = recipe.setdefault("batch_tuning_compare_packing", "packing" in recipe)
    probe = state.get("packing_probe")
    if compare_packing and probe is None:
        # 补齐更少未必更快；同一容量下，用正式更新实测两种分组。
        selected = "legacy" if packing == "cost" else "cost"
        recipe["packing"] = selected
        state.update(packing_probe="running", warmup_remaining=1)
        return dict(reason="compare_packing_on_successful_updates", before=current, after=current,
                    before_packing=packing, after_packing=selected, inputs_per_second=rate)
    if probe == "running":
        state["packing_probe"] = "done"
        recipe["packing"] = best.get("packing", packing)
        selected = current if best.get("headroom_gib", free) <= target or current >= 131072 else min(131072, current*2)
        recipe["padded_stroke_budget"] = selected
        state.update(done=selected == current, warmup_remaining=1)
        return dict(reason="packing_comparison_finished", before=current, after=selected,
                    before_packing=packing, after_packing=recipe["packing"],
                    inputs_per_second=rate, best_inputs_per_second=best["rate"])
    tolerance = float(recipe.get("batch_tuning_slowdown", .03))
    if rate < best["rate"]*(1-tolerance):
        # 与历史最快档比较，不能逐档接受小幅变慢后遗忘更快的设置。
        recipe["padded_stroke_budget"] = best["budget"]
        recipe["packing"] = best.get("packing", packing)
        state["done"] = True
        return dict(reason="larger_batch_slower", before=current, after=best["budget"],
                    before_packing=packing, after_packing=recipe["packing"],
                    inputs_per_second=rate, best_inputs_per_second=best["rate"])
    if free <= target or current >= 131072:
        state["done"] = True
        recipe["padded_stroke_budget"] = best["budget"]
        recipe["packing"] = best.get("packing", packing)
        return dict(reason="soft_target_or_batch_limit", before=current, after=best["budget"],
                    selected_budget=best["budget"], headroom_gib=free, inputs_per_second=rate)
    selected = min(131072, current*2)
    state["previous"] = dict(budget=current, rate=rate)
    recipe["padded_stroke_budget"] = selected
    return dict(reason="grow_on_successful_updates", before=current, after=selected,
                headroom_gib=free, inputs_per_second=rate)
