"""冻结完整批的执行时间、梯度与实际注意力算子检查；不消费正式训练状态。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import time

import torch

from scripts.checks.check_page_diffusion_performance import load_baseline, compare_gradients
from src.page_document import control_stroke
from src.st.generation_pretrain import PageGenerator
from src.st.page_diffusion_codec import FrozenPageCodec
from src.st import page_diffusion_data as data
from src.st import page_diffusion_runtime as runtime


def profile_summary(profile):
    rows = []
    for item in profile.key_averages():
        rows.append(dict(name=item.key, calls=item.count,
            self_device_us=getattr(item, "self_device_time_total", 0.),
            device_us=getattr(item, "device_time_total", 0.),
            self_cpu_us=item.self_cpu_time_total))
    return dict(top_device=sorted(rows, key=lambda x:x["self_device_us"], reverse=True)[:25],
                attention=[r for r in rows if "attention" in r["name"]],
                phases=[r for r in rows if r["name"].startswith("diffusion.")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("legacy", "fast"), default="fast")
    parser.add_argument("--budget", type=int, default=8192)
    parser.add_argument("--packing", choices=("legacy", "cost"), default="cost")
    parser.add_argument("--layout", choices=("dense", "aligned", "split"), default="split")
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--encoder_chunk", type=int, default=1024)
    parser.add_argument("--reference_gradients", type=Path)
    parser.add_argument("--save_gradients", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--allocator_fraction", type=float, default=.8,
                        help="本机Windows避免显存溢出到共享内存；不用于服务器soft策略")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    packet = torch.load(args.data, map_location="cpu", weights_only=False)
    documents, counts = packet["documents"], {**packet["counts"], "aux":0, "space":0}
    for document in documents:
        document.metadata.pop("_text_aux_targets", None)
    recipe = deepcopy(packet["recipe"])
    for key in ("text_aux_alpha", "text_aux_gradient_ratio", "calibration_interval", "calibrated_inputs"):
        recipe.pop(key, None)
    recipe["model"].pop("shapes", None)
    recipe["model"]["version"] = "page_diffusion_v2"
    recipe.update(microbatch_documents=512, padded_stroke_budget=args.budget,
        minimum_free_gib=0., memory_policy="soft",
        packing=args.packing, attention_layout=args.layout, encode_chunk_size=args.encoder_chunk)
    recipe["model"]["activation_checkpoint"] = args.checkpoint
    active_runtime, active_data = runtime, data
    if args.variant == "legacy":
        recipe["model"].update(version="page_diffusion_v1", shapes=117)
        recipe["text_aux_alpha"] = 0.
        active_data = load_baseline(args.baseline, "page_diffusion_data")
        model_module = load_baseline(args.baseline, "page_diffusion")
        active_runtime = load_baseline(args.baseline, "page_diffusion_runtime")
        active_runtime.PageDiffusionConfig = model_module.PageDiffusionConfig
        active_runtime.PageDiffusionR = model_module.PageDiffusionR
        active_runtime.losses = active_data.losses
        active_runtime.iter_diffusion_batches = active_data.iter_diffusion_batches
        recipe.update(packing="legacy", attention_layout="dense", encode_chunk_size=256)
    device = torch.device("cuda:0")
    torch.cuda.set_per_process_memory_fraction(args.allocator_fraction, device)
    ae = torch.load("tmp_smoke/r_dual_review_assets_20261009/ae_final.pt", map_location="cpu", weights_only=False)
    codec = FrozenPageCodec.from_state(ae, encode_chunk_size=recipe["encode_chunk_size"]).to(device)
    del ae
    eop = PageGenerator(codec).encode([control_stroke("eop")])[0].float()
    normalization = json.loads((args.data.parent/"normalization.json").read_text(encoding="utf8"))
    model, optimizer, scaler, ema = active_runtime.new_training_state(recipe, device)
    active_runtime.initialize_coordinates(model, ema, normalization, eop)
    with torch.no_grad():
        for layer in model.layers:
            width = model.cfg.width
            layer.modulation.bias[2*width:3*width].fill_(.1)
            layer.modulation.bias[5*width:6*width].fill_(.1)
    warm = next(active_data.iter_diffusion_batches(documents[:2], recipe))
    warmed = active_runtime.forward_batch(model, codec, warm, counts, recipe, device)
    warmed["total"].backward()
    del warm, warmed
    optimizer.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    print(json.dumps(dict(phase="start", variant=args.variant, budget=args.budget,
        checkpoint=args.checkpoint, layout=recipe["attention_layout"], packing=recipe["packing"])), flush=True)
    started = time.monotonic()
    result = active_runtime.train_update(model, codec, optimizer, scaler, ema, documents, counts, recipe, device)
    result.update(measured_seconds=time.monotonic()-started, inputs=sum(packet["inputs"].values()),
        variant=args.variant, gpu=torch.cuda.get_device_name(), torch=torch.__version__, recipe=recipe)
    if args.reference_gradients:
        result["gradient_difference"] = compare_gradients(model, args.reference_gradients)
    if args.save_gradients:
        torch.save({n:p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p,device="cpu")
                    for n,p in model.named_parameters()}, args.out/"gradients.pt")
    runtime.write_json(args.out/"result.json", result)
    print(json.dumps({k:result[k] for k in ["measured_seconds", "loss", "microbatches", "padded_positions",
        "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "retries"]} |
        {"gradient_difference":result.get("gradient_difference")}), flush=True)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            probe = next(active_data.iter_diffusion_batches(documents[:2], recipe))
            optimizer.zero_grad(set_to_none=True)
            profiled = active_runtime.forward_batch(model, codec, probe, counts, recipe, device)
            profiled["total"].backward()
        report = profile_summary(prof)
        runtime.write_json(args.out/"profile.json", report)
        print(json.dumps(dict(profile_attention=report["attention"])), flush=True)


if __name__ == "__main__":
    main()
