"""软显存目标只影响下一步拆分，不把略低于6GiB当作失败。"""
from copy import deepcopy
import json

from src.st.page_diffusion_capacity import adapt_batching


def main():
    template = dict(adaptive_microbatch=True, memory_policy="soft", minimum_free_gib=6.,
                    microbatch_documents=512, padded_stroke_budget=16384,
                    batch_tuning_samples=3, batch_tuning_warmup=0)
    def sample(free=20., seconds=10.):
        return dict(physical_peak_headroom_gib=free, seconds=seconds, gpu_base_allocated_bytes=2*2**30,
            gpu_total_bytes=32*2**30, gpu_outside_bytes=0, gpu_peak_reserved_bytes=(32-free)*2**30)
    recipe = deepcopy(template)
    for _ in range(3):
        changed = adapt_batching(recipe, sample(), 65536)
    assert changed["reason"] == "grow_on_successful_updates" and recipe["padded_stroke_budget"] == 32768
    for _ in range(3):
        changed = adapt_batching(recipe, sample(seconds=13.), 65536)
    assert changed["reason"] == "larger_batch_slower" and recipe["padded_stroke_budget"] == 16384
    recipe = deepcopy(template)
    for _ in range(3):
        changed = adapt_batching(recipe, sample(free=5.), 65536)
    assert recipe["padded_stroke_budget"] == 16384 and recipe["runtime_batch_tuning"]["done"]
    recipe = deepcopy(template)
    changed = adapt_batching(recipe, sample(free=3.), 65536)
    assert 8192 < recipe["padded_stroke_budget"] < 16384
    assert changed["reason"] == "soft_target_adjust_next_update"
    recipe = deepcopy(template)
    recipe["padded_stroke_budget"] = 8192
    changed = adapt_batching(recipe, {**sample(), "oom_retries":1}, 65536)
    assert changed["reason"] == "actual_oom_shrunk_same_batch" and recipe["runtime_batch_tuning"]["done"]
    recipe = {**deepcopy(template), "batch_tuning_slowdown":.1}
    for duration in (10., 10.8, 11.6):
        for _ in range(3):
            changed = adapt_batching(recipe, sample(seconds=duration), 65536)
    assert recipe["padded_stroke_budget"] == 16384
    assert changed["after"] == recipe["runtime_batch_tuning"]["best"]["budget"]
    recipe = {**deepcopy(template), "packing":"cost"}
    for _ in range(3):
        changed = adapt_batching(recipe, sample(seconds=10.), 65536)
    assert changed["after_packing"] == "legacy" and recipe["padded_stroke_budget"] == 16384
    adapt_batching(recipe, sample(seconds=100.), 65536)  # 跳过切换后的首次冷启动。
    for _ in range(3):
        changed = adapt_batching(recipe, sample(seconds=8.), 65536)
    assert changed["reason"] == "packing_comparison_finished"
    assert recipe["packing"] == "legacy" and recipe["padded_stroke_budget"] == 32768
    assert recipe["runtime_batch_tuning"]["best"]["packing"] == "legacy"
    print(json.dumps(dict(grow_using_successful_updates=True, stop_when_larger_is_slower=True,
        accepts_five_gib_headroom=True, deeper_overshoot_adjusts_next_update=True, actual_oom_stops_growth=True,
        returns_to_global_best_after_gradual_slowdown=True, compares_both_packings_using_formal_updates=True)))


if __name__ == "__main__":
    main()
