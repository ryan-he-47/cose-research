"""实际冻结AE、正式尺寸R的FP16更新和完整状态恢复核对。"""
from __future__ import annotations

from copy import deepcopy
import gc
import json
from pathlib import Path
import time

import torch

from scripts.checks.check_page_diffusion import document
from src.page_document import control_stroke
from src.st.generation_pretrain import PageGenerator
from src.st.page_diffusion_codec import FrozenPageCodec
from src.st.page_diffusion_data import prepare_update
from src.st.page_diffusion_runtime import (new_training_state, initialize_coordinates,
    train_update, _rng_state, _restore_rng)
from src.st.page_diffusion_sampling import PageDiffusionGenerator


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    output = Path("tmp_smoke/r_page_diffusion_no_aux_20261102")/time.strftime("gpu_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    recipe = json.loads(Path("configs/r_page_diffusion_20261102.json").read_text(encoding="utf-8"))
    recipe.update(minimum_free_gib=0., workers=0)
    ae_path = Path("tmp_smoke/r_dual_review_assets_20261009/ae_final.pt")
    ae = torch.load(ae_path, map_location="cpu", weights_only=False)
    codec = FrozenPageCodec.from_state(ae).to(device)
    del ae
    geometry = PageGenerator(codec)
    eop = geometry.encode([control_stroke("eop")])[0].float()
    model, optimizer, scaler, ema = new_training_state(recipe, device)
    initialize_coordinates(model, ema, {"mean":[0.]*48, "scale":1.}, eop)
    model.train()
    noise = torch.Generator().manual_seed(99)
    docs = [document(5), document(19)]
    counts = prepare_update(docs, noise)
    assert model.cfg.version == "page_diffusion_v2"
    assert not any("text_aux" in key for key in model.state_dict())
    assert not any("aux" in key or "space" in key for key in counts)
    assert all(int(v["step"]) == 0 for v in optimizer.state.values())
    first = train_update(model, codec, optimizer, scaler, ema, docs, counts, recipe, device)
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                 ema=ema, noise=noise.get_state(), rng=_rng_state(), recipe=recipe)
    restore_path, expected_path = output/"restore.pt", output/"expected.pt"
    torch.save(state, restore_path)
    del state
    counts = prepare_update(docs, noise)
    noise_expected = docs[0].metadata["_page_diffusion"]["noise"].clone()
    second = train_update(model, codec, optimizer, scaler, ema, docs, counts, recipe, device)
    torch.save({"model":model.state_dict(), "ema":ema, "optimizer":optimizer.state_dict()}, expected_path)
    parameters = sum(p.numel() for p in model.parameters())
    del model, optimizer, scaler, ema
    gc.collect(); torch.cuda.empty_cache()
    restored = torch.load(restore_path, map_location="cpu", weights_only=False)
    recipe = restored["recipe"]
    model, optimizer, scaler, ema = new_training_state(recipe, device)
    model.load_state_dict(restored["model"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    scaler.load_state_dict(restored["scaler"])
    ema = {k:v.to(device) for k,v in restored["ema"].items()}
    noise.set_state(restored["noise"])
    _restore_rng(restored["rng"])
    del restored
    counts = prepare_update(docs, noise)
    torch.testing.assert_close(docs[0].metadata["_page_diffusion"]["noise"], noise_expected, rtol=0, atol=0)
    resumed = train_update(model, codec, optimizer, scaler, ema, docs, counts, recipe, device)
    expected = torch.load(expected_path, map_location="cpu", weights_only=False)
    maximum = 0.
    for k, v in model.state_dict().items():
        maximum = max(maximum, float((v.cpu()-expected["model"][k]).abs().max()))
        torch.testing.assert_close(v.cpu(), expected["model"][k], atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(ema[k].cpu(), expected["ema"][k], atol=1e-7, rtol=1e-5)
    actual_optimizer = optimizer.state_dict()
    for key, values in actual_optimizer["state"].items():
        for field, value in values.items():
            torch.testing.assert_close(value.cpu(), expected["optimizer"]["state"][key][field], atol=1e-7, rtol=1e-5)
    del expected
    sample = PageDiffusionGenerator(model, codec).generate(docs[0].pages[0].elements,
        seed=42, steps=2, max_new_strokes=16)
    assert all(torch.isfinite(torch.tensor(r["points"])).all() for r in sample["records"])
    assert all(not p.requires_grad for p in codec.parameters())
    result = dict(parameters=parameters, first=first, second=second, resumed=resumed,
        auxiliary_heads=False, max_resume_parameter_difference=maximum,
        optimizer_and_ema_restored=True, noise_and_cut_restored=True,
        frozen_ae=True, sample_stop=sample["stop"], sample_strokes=sample["generated_strokes"],
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (output/"result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"result":str(output/"result.json"), **result}))
    # 仅移除本检查刚创建的两个临时快照；记录和结果保留。
    for path in (restore_path, expected_path):
        if path.resolve().parent != output.resolve():
            raise RuntimeError("临时快照路径越界")
        path.unlink()


if __name__ == "__main__":
    main()
