"""块扩散的遮罩泄漏、页位置、缓存、损失分母及恢复定向检查。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from src.page_document import Page, StrokeElement, build_document, control_stroke
from src.st.page_diffusion import PageDiffusionConfig, PageDiffusionR, rotate_pages
from src.st.page_diffusion_data import document_layout, pack_documents, prepare_update, diffusion_inputs, losses
from src.st.page_diffusion_sampling import clean_cache, sample_block
from src.st.page_diffusion_runtime import new_training_state, train_update, initialize_coordinates


def document(body=34, complete=True):
    line = np.asarray([[.2, .2], [.3, .25]], np.float32)
    condition = Page(0, "sketch", (StrokeElement(line, 0, False, "sketch"),), False)
    target = Page(1, "sketch", tuple(StrokeElement(line.copy(), 1, True, "sketch") for _ in range(body)), True)
    doc = build_document((condition, target), complete=complete, metadata={"task": "O1"})
    return doc


def run():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    cfg = PageDiffusionConfig(width=64, layers=2, heads=4, ffn=128, condition=16,
                              activation_checkpoint=False)
    recipe = json.loads(Path("configs/r_page_diffusion_20261102.json").read_text(encoding="utf-8"))
    recipe.update(model=asdict(cfg), minimum_free_gib=0., microbatch_documents=2,
                  padded_stroke_budget=4096, attention_layout="dense")
    model = PageDiffusionR(cfg).eval()
    # 零初始化门控会让注意力无效；先启用它，泄漏/缓存检查才有意义。
    with torch.no_grad():
        for layer in model.layers:
            layer.modulation.bias[2*cfg.width:3*cfg.width].fill_(1.)
            layer.modulation.bias[5*cfg.width:6*cfg.width].fill_(1.)
    doc = document()
    rng = torch.Generator().manual_seed(42)
    counts = prepare_update([doc], rng)
    doc.metadata["_page_diffusion"] = {**doc.metadata["_page_diffusion"], **document_layout(doc)}
    # 手工取消prefix jitter后，重新给定统计与噪声，确保至少三个预测块。
    layout = doc.metadata["_page_diffusion"]
    layout["noise"] = torch.randn(len(layout["blocks"]), 16, 48, generator=rng)
    layout["times"] = torch.full((len(layout["blocks"]),), .7)
    counts = dict(body={"O1":34}, control=2, padded=12, blocks=3)
    batch = pack_documents([doc], recipe)
    encoded = torch.randn(1, doc.stroke_count, 48)
    values, target = diffusion_inputs(model, encoded, batch)
    def forward(v, b=batch):
        return model(v, b["pages"], b["local"], b["roles"], b["times"], b["allowed"])
    out = forward(values)
    c, start = batch["clean_size"], layout["blocks"][0]["start"]
    changed = values.clone()
    changed[:, start:c] = torch.randn_like(changed[:, start:c])*10
    changed[:, c+16:] = torch.randn_like(changed[:, c+16:])*10
    after = forward(changed)
    torch.testing.assert_close(out["prediction"][:, c:c+16], after["prediction"][:, c:c+16], atol=1e-7, rtol=0)
    # 当前块干净真值和未来带噪块改变，第一块预测必须完全不变。
    with torch.no_grad():
        cache = clean_cache(model, values[:, :start], batch["pages"][:, :start], batch["local"][:, :start])
        part = model(values[:, c:c+16], batch["pages"][:, c:c+16], batch["local"][:, c:c+16],
                     batch["roles"][:, c:c+16], batch["times"][:, c:c+16],
                     torch.ones(1, 1, 16, start+16, dtype=torch.bool), past=cache)
    torch.testing.assert_close(out["prediction"][:, c:c+16], part["prediction"], atol=2e-7, rtol=2e-5)
    # 页号共同平移不改变旋转内积；页内先后由另一输入提供。
    q, k = torch.randn(1, 4, 5, 16), torch.randn(1, 4, 5, 16)
    pages = torch.tensor([[0, 0, 1, 2, 2]])
    score = rotate_pages(q, pages, 10000.) @ rotate_pages(k, pages, 10000.).transpose(-1, -2)
    shifted = rotate_pages(q, pages+17, 10000.) @ rotate_pages(k, pages+17, 10000.).transpose(-1, -2)
    torch.testing.assert_close(score, shifted, atol=1e-5, rtol=2e-5)
    window = doc.window(1)
    assert document_layout(window)["local"][0] == len(doc.pages[0].elements)-1
    assert document_layout(window)["local"][1] == 0
    group_totals = []
    for body in (0, 14):
        item = document(body, complete=False)
        prepare_update([item], torch.Generator().manual_seed(42))
        packed = pack_documents([item], recipe)
        group_totals.append(float(packed["control_weight"][packed["synthetic"]].sum()))
        assert abs(group_totals[-1]-.25) < 1e-7
        assert float(packed["control_weight"][packed["real_control"]].sum()) == 1.
        loc = packed["local"][0, packed["clean_size"]:]
        assert torch.equal(loc[1:]-loc[:-1], torch.ones_like(loc[:-1]))
        assert packed["allowed"][0, 0, -16:, -16:].all()
    # 相同完整批分母下，按文档拆分的损失/梯度与整批一致。
    docs = [document(5), document(19)]
    stats = prepare_update(docs, torch.Generator().manual_seed(88))
    latents = [torch.randn(d.stroke_count, 48) for d in docs]
    def evaluate(items, zs):
        packed = pack_documents(items, recipe)
        enc = torch.zeros(len(items), packed["clean_size"], 48)
        for row, z in enumerate(zs): enc[row, :len(z)] = z
        inp, tgt = diffusion_inputs(model, enc, packed)
        prediction = forward(inp, packed)
        return losses(model, prediction, tgt, packed, stats, recipe)["total"]
    full = evaluate(docs, latents)
    full_grad = torch.autograd.grad(full, model.output.weight)[0]
    split = sum(evaluate([d], [z]) for d, z in zip(docs, latents))
    split_grad = torch.autograd.grad(split, model.output.weight)[0]
    torch.testing.assert_close(full, split, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(full_grad, split_grad, atol=2e-6, rtol=1e-5)
    # 即使旧资产带有错误分类标签，也不能进入新目标或产生分类梯度。
    assert not hasattr(model, "text_aux")
    assert not any("aux" in key or "space" in key for key in stats)
    for item in docs:
        n = item.stroke_count
        item.metadata["_text_aux_targets"] = dict(shape=np.full(n, -999, np.int64),
            space=np.full(n, 999., np.float32), first=np.zeros(n, bool), valid=np.ones(n, bool))
    with_labels = evaluate(docs, latents)
    with_labels_grad = torch.autograd.grad(with_labels, model.output.weight)[0]
    torch.testing.assert_close(with_labels, full, atol=0, rtol=0)
    torch.testing.assert_close(with_labels_grad, full_grad, atol=0, rtol=0)
    print(json.dumps(dict(passed=["no_future_or_clean_target_leak", "clean_kv_cache_equivalence",
        "page_rope_relative", "window_keeps_local_position", "eop_weight_independent_of_padding",
        "eop_slots_participate_in_attention", "full_batch_microbatch_loss_and_gradient",
        "legacy_auxiliary_labels_cannot_change_loss_or_gradient"],
        padding_group_totals=group_totals)))


if __name__ == "__main__":
    run()
