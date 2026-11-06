"""固定真批子集，核对新旧损失及中途OOM后的完整更新一致性。"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.checks.check_page_diffusion_performance import load_baseline
from src.page_document import control_stroke
from src.st.generation_pretrain import PageGenerator
from src.st.page_diffusion import PageDiffusionConfig, PageDiffusionR
from src.st import page_diffusion_runtime as runtime
from src.st.page_diffusion_codec import FrozenPageCodec
from src.st.page_diffusion_data import prepare_update, pack_documents, diffusion_inputs, losses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, help="新检查结果目录，避免覆盖原诊断记录")
    p.add_argument("--fast", action="store_true", help="核对双流注意力和成本分组的实际路径")
    args = p.parse_args()
    torch.set_num_threads(4)
    packet = torch.load(args.root/"actual_batch.pt", map_location="cpu", weights_only=False)
    docs = [deepcopy(next(d for d in packet["documents"] if d.metadata["task"] == t)) for t in ("O1","O2","S3")]
    counts = prepare_update(docs, torch.Generator().manual_seed(77))
    cfg = PageDiffusionConfig(width=64, layers=2, heads=4, ffn=128, condition=16, activation_checkpoint=False)
    recipe = {k:v for k,v in packet["recipe"].items()
              if k not in ("text_aux_alpha", "text_aux_gradient_ratio", "calibration_interval", "calibrated_inputs")}
    recipe.update(model=asdict(cfg), minimum_free_gib=0.)
    if args.fast:
        recipe.update(packing="cost", attention_layout="split", encode_chunk_size=1024)
    model = PageDiffusionR(cfg)
    batch = pack_documents(docs, recipe)
    encoded = torch.randn(3, batch["clean_size"], 48)
    value, target = diffusion_inputs(model, encoded, batch)
    output = model(value, batch["pages"], batch["local"], batch["roles"], batch["times"], batch["allowed"],
                   clean_size=batch["clean_size"], attention_layout=recipe.get("attention_layout", "dense"))
    old = load_baseline(args.root/"baseline", "page_diffusion_data")
    # 旧目标只保留正文与控制符项，与新纯扩散目标对齐。
    old_batch = {**batch, "aux_valid":torch.zeros_like(batch["real_body"])}
    left = old.losses(model, output, target, old_batch, {**counts, "aux":0, "space":0}, recipe)["total"]
    right = losses(model, output, target, batch, counts, recipe)["total"]
    parameters = tuple(model.parameters())
    ga = torch.autograd.grad(left, parameters, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(right, parameters, allow_unused=True)
    torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-6)
    difference = 0.
    for a,b in zip(ga,gb):
        if a is None and b is None:
            continue
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        difference = max(difference, float((a-b).abs().max()))
    device = torch.device("cuda")
    codec = FrozenPageCodec.from_state(torch.load("tmp_smoke/r_dual_review_assets_20261009/ae_final.pt",
        map_location="cpu", weights_only=False)).to(device)
    norm = json.loads((args.root/"normalization.json").read_text(encoding="utf-8"))
    eop = PageGenerator(codec).encode([control_stroke("eop")])[0].float()
    expected, retried = None, None
    for inject in (False, True):
        active = {**recipe, "microbatch_documents":2 if inject else 1}
        model, optimizer, scaler, ema = runtime.new_training_state(active, device)
        runtime.initialize_coordinates(model, ema, norm, eop)
        original, calls = runtime.forward_batch, [0]
        def checked(*a, **kw):
            calls[0] += 1
            if inject and calls[0] == 2:
                raise torch.OutOfMemoryError("仅诊断：第一次反向之后强制重试")
            return original(*a, **kw)
        with patch.object(runtime, "forward_batch", checked):
            result = runtime.train_update(model, codec, optimizer, scaler, ema, docs, counts, active, device)
        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        averages = {k:v.detach().cpu().clone() for k,v in ema.items()}
        moments = {k:{name:value.detach().cpu().clone() for name,value in fields.items()}
                   for k,fields in optimizer.state_dict()["state"].items()}
        if not inject:
            expected = state
            expected_ema, expected_moments, expected_scaler = averages, moments, deepcopy(scaler.state_dict())
        else:
            assert result["retries"] == 1 and active["microbatch_documents"] == 1
            for key in expected:
                torch.testing.assert_close(state[key], expected[key], atol=0, rtol=0)
                torch.testing.assert_close(averages[key], expected_ema[key], atol=0, rtol=0)
            for key,fields in moments.items():
                for name,value in fields.items():
                    torch.testing.assert_close(value, expected_moments[key][name], atol=0, rtol=0)
            assert scaler.state_dict() == expected_scaler
            retried = result["retries"]
    report = dict(fp32_loss_difference=float(abs(left-right).detach()), fp32_gradient_max_difference=difference,
                  oom_retries=retried, oom_same_parameters_exact=True, oom_same_adam_ema_scaler=True,
                  attention_layout=recipe.get("attention_layout", "dense"), packing=recipe.get("packing", "legacy"))
    runtime.write_json((args.out or args.root)/"optimization_checks.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
