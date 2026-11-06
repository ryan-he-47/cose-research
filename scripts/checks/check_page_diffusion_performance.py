"""同一真实完整批的旧/新执行对照；仅本机临时状态，不消费正式输入。"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

from src.st.generation_pretrain import PageGenerator
from src.page_document import control_stroke
from src.st.page_diffusion_runtime import (new_training_state, initialize_coordinates,
    forward_batch, train_update, write_json)
from src.st import page_diffusion_data as current_data
from src.st.page_diffusion_codec import FrozenPageCodec


def load_baseline(path, name):
    spec = importlib.util.spec_from_file_location("src.st._perf_"+name, path/(name+".py"))
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def compare_gradients(model, path):
    expected = torch.load(path, map_location="cpu", weights_only=False)
    square, delta, dot, actual_square, maximum = 0., 0., 0., 0., 0.
    for name, parameter in model.named_parameters():
        a = parameter.grad.detach().cpu().double() if parameter.grad is not None else torch.zeros_like(expected[name], dtype=torch.float64)
        b = expected[name].double()
        square += float(b.square().sum())
        actual_square += float(a.square().sum())
        delta += float((a-b).square().sum())
        dot += float((a*b).sum())
        maximum = max(maximum, float((a-b).abs().max()))
    return dict(relative_l2=(delta/max(square, 1e-30))**.5,
                cosine=dot/max((square*actual_square)**.5, 1e-30), max_abs=maximum)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--variants", default="baseline,batched,optimized")
    p.add_argument("--budget", type=int, default=8192)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    packet = torch.load(args.data, map_location="cpu", weights_only=False)
    documents, counts = packet["documents"], {**packet["counts"], "aux":0, "space":0}
    for document in documents:
        document.metadata.pop("_text_aux_targets", None)
    # 当前检查只对照正文/控制符扩散，旧分类辅助不再属于比较目标。
    recipe = {k:v for k,v in packet["recipe"].items()
              if k not in ("text_aux_alpha", "text_aux_gradient_ratio", "calibration_interval", "calibrated_inputs")}
    recipe.update(minimum_free_gib=0.)
    recipe["model"] = {k:v for k,v in recipe["model"].items() if k != "shapes"}
    recipe["model"]["version"] = "page_diffusion_v2"
    norm = json.loads((args.data.parent/"normalization.json").read_text(encoding="utf-8"))
    old_data = load_baseline(args.baseline, "page_diffusion_data")
    old_runtime = load_baseline(args.baseline, "page_diffusion_runtime")
    old_runtime.losses = old_data.losses
    old_runtime.iter_diffusion_batches = old_data.iter_diffusion_batches
    device = torch.device("cuda")
    ae = torch.load("tmp_smoke/r_dual_review_assets_20261009/ae_final.pt", map_location="cpu", weights_only=False)
    codec = FrozenPageCodec.from_state(ae).to(device)
    del ae
    eop = PageGenerator(codec).encode([control_stroke("eop")])[0].float()
    for variant in args.variants.split(","):
        current = dict(recipe)
        if variant != "baseline":
            current.update(microbatch_documents=128, padded_stroke_budget=args.budget)
        runtime = old_runtime if variant in ("baseline", "batched") else None
        data_module = old_data if runtime else current_data
        run = runtime.train_update if runtime else train_update
        fwd = runtime.forward_batch if runtime else forward_batch
        model, optimizer, scaler, ema = new_training_state(current, device)
        initialize_coordinates(model, ema, norm, eop)
        # 注意力不能处在全零门控，以便真正检查换微批后的梯度。
        with torch.no_grad():
            for layer in model.layers:
                width = model.cfg.width
                layer.modulation.bias[2*width:3*width].fill_(.1)
                layer.modulation.bias[5*width:6*width].fill_(.1)
        warm = next(data_module.iter_diffusion_batches(documents[:2], current))
        warmed = fwd(model, codec, warm, counts, current, device)
        warmed["total"].backward()
        del warm, warmed
        optimizer.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        result = run(model, codec, optimizer, scaler, ema, documents, counts, current, device)
        result["measured_seconds"] = time.monotonic()-started
        result.update(variant=variant, peak_bytes=torch.cuda.max_memory_allocated(),
            gpu=torch.cuda.get_device_name(), torch=torch.__version__, documents=len(documents),
            raw_inputs=sum(packet["inputs"].values()))
        shapes = [(b["pages"].shape, b["clean_size"]) for b in data_module.iter_diffusion_batches(documents, current)]
        result.update(microbatches=len(shapes), padded_positions=sum(s[0]*s[1] for s,_ in shapes))
        gradient_path = args.out/"baseline_gradients.pt"
        if variant == "baseline":
            torch.save({n:p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p,device="cpu")
                        for n,p in model.named_parameters()}, gradient_path)
        elif gradient_path.exists():
            result["gradient_difference"] = compare_gradients(model, gradient_path)
        write_json(args.out/(variant+".json"), result)
        print(json.dumps({k:result[k] for k in ("variant","measured_seconds","loss","microbatches",
            "padded_positions","peak_bytes","retries")}|{"gradient_difference":result.get("gradient_difference")}), flush=True)
        del model, optimizer, scaler, ema
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
