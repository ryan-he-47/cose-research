"""2026-10-03：短 CPU 核对右 padding fast path 与变长组批的梯度。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.page_document import Page, StrokeElement, build_document
from src.st.pipeline_pretrain import collate_documents
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.pretrain_model import FrozenSketchLM
from src.st.train_pretrain import collect_update_batches


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = gpt2_small_pretrain_config()
    cfg.r_d_model, cfg.r_dff, cfg.r_layers, cfg.r_heads = 32, 64, 2, 4
    model = FrozenSketchLM(cfg, activation_checkpoint=True).train()
    documents = []
    for count in (3, 11, 5):
        elements = tuple(StrokeElement(np.array([[.1 + i * .015, .2], [.12 + i * .015, .3]], np.float32),
                                       0, True, "text") for i in range(count))
        documents.append(build_document([Page(0, "text", elements)], metadata={"task": "T1"}))
    batch = collate_documents(documents)
    embeddings = model.encode_valid(batch)
    ids, pad = batch["slice_id"][:, :-1], batch["stroke_pad"][:, :-1]
    expected = model.hidden_states(embeddings[:, :-1], ids, pad)
    actual = model.hidden_states(embeddings[:, :-1], ids, pad, right_padded=True)
    torch.testing.assert_close(actual[~pad], expected[~pad], atol=2e-5, rtol=2e-5)
    parameters = list(model.r_pred.tok_proj.parameters()) + list(model.r_pred.layers.parameters())
    weights = torch.randn_like(expected)
    g1 = torch.autograd.grad((expected[~pad] * weights[~pad]).sum(), parameters)
    g2 = torch.autograd.grad((actual[~pad] * weights[~pad]).sum(), parameters)
    for p, q in zip(g1, g2):
        torch.testing.assert_close(p, q, atol=3e-5, rtol=3e-5)
    grouped = deepcopy(model)
    target_count = int(model.target_mask(batch).sum())
    model(batch)["pred_sum"].div(target_count).backward()
    args = {"microbatch_documents": 2, "target_budget": target_count, "max_points": 76,
            "batch_by_length": True, "padded_stroke_budget": 20}
    batches, count, tasks = collect_update_batches(iter(documents), args, 1024)
    assert count == target_count and tasks["T1"] == target_count
    sums, content, control = 0., 0., 0.
    for chunk in batches:
        result = grouped(chunk)
        result["pred_sum"].div(count).backward()
        sums += float(result["pred_sum"].detach())
        content += float(result["content_sum"])
        control += float(result["control_sum"])
    assert abs(sums - content - control) < 1e-3
    max_error = 0.
    for p, q in zip(model.r_pred.parameters(), grouped.r_pred.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4)
        max_error = max(max_error, float((p.grad - q.grad).abs().max()))
    assert all(p.grad is None for module in (model.encoder, model.decoder, model.dct_head)
               for p in module.parameters())
    print(json.dumps({"right_padding_max_abs": float((actual[~pad] - expected[~pad]).detach().abs().max()),
                      "grouped_gradient_max_abs": max_error, "targets": count,
                      "microbatches": len(batches), "frozen_ae": "passed"}))


if __name__ == "__main__":
    main()
