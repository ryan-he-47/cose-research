"""2026-10-03：核对页切换、缓存与完整因果前向；可选实际 FP16 权重。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.st.generation_pretrain import PageCursor, PageKVCache
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.pretrain_model import FrozenSketchLM


def check_cache(model, autocast):
    device = next(model.parameters()).device
    embeddings = torch.randn(1, 9, model.cfg.latent_units, device=device)
    ids = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2]], device=device)
    pad = torch.zeros(1, 9, dtype=torch.bool, device=device)
    with torch.no_grad(), autocast:
        full = model.hidden_states(embeddings, ids, pad)
        cache = PageKVCache(model)
        parts = [cache.forward(embeddings[:, :4], ids[:, :4], prefill=True)]
        for index in range(4, 9):
            parts.append(cache.forward(embeddings[:, index:index + 1], ids[:, index:index + 1]))
        actual = torch.cat(parts, 1)
        torch.testing.assert_close(actual, full, atol=.015 if device.type == "cuda" else 2e-5,
                                   rtol=.015 if device.type == "cuda" else 2e-5)
        for expected, value in zip(model.prediction_head(full[:, -1]),
                                   model.prediction_head(actual[:, -1])):
            torch.testing.assert_close(value, expected, atol=.003, rtol=.015)
    return float((actual - full).abs().max())


def main():
    from contextlib import nullcontext
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = gpt2_small_pretrain_config()
    cfg.r_d_model, cfg.r_dff, cfg.r_layers, cfg.r_heads = 32, 64, 2, 4
    tiny = FrozenSketchLM(cfg).eval()
    report = {"cpu_cache_max_abs": check_cache(tiny, nullcontext())}
    cursor = PageCursor(0)
    cursor.append("eop")
    assert cursor.page_id == 1
    cursor.append("eod")
    cursor.append("body")
    cursor.append("eop")
    assert cursor.page_id == 2 and not cursor.complete
    cursor.append("eod")
    cursor.append("eop")
    assert cursor.complete and cursor.page_id == 2
    report["page_cursor"] = "passed"
    if args.ckpt:
        from scripts.demo.demo_pretrain_generation import load_generator, text_prefix
        from src.page_document import control_stroke
        generator, font, _ = load_generator(args.ckpt, "cuda")
        report["fp16_cache_max_abs"] = check_cache(generator.model, generator.autocast())
        for kind, templates in generator.templates.items():
            for template in templates:
                assert generator.recognize(template)[0] == kind
        for x in (.2, .5, .85):
            line = np.column_stack((np.linspace(x, x + .08, 76), np.full(76, .5))).astype(np.float32)
            assert generator.recognize(line)[0] == "body"
        prefix = text_prefix("O1", "cat", font)
        assert prefix[-1].kind == "eop"
        cursor = PageCursor(prefix[-1].page_id)
        cursor.append(prefix[-1].kind)
        assert cursor.page_id == prefix[-1].page_id + 1
        # 实际笔画编码后的缓存结果与既有 next_distribution 同源。
        with torch.no_grad(), generator.autocast():
            e = generator.encode([p.rawxy for p in prefix]).unsqueeze(0)
            ids = torch.tensor([[p.page_id for p in prefix]], device="cuda")
            pad = torch.zeros(ids.shape, dtype=torch.bool, device="cuda")
            full = generator.model.hidden_states(e, ids, pad)
            cached = PageKVCache(generator.model).forward(e, ids, prefill=True)
            torch.testing.assert_close(full, cached, atol=.015, rtol=.015)
        report["controls_and_prefix"] = "passed"
        # 同种子下，有/无确定性排版缓存必须逐点相等；缓存内容不可被增强污染。
        from functools import partial
        import src.page_document as page_module
        from src.text_augmentation import augment_rendered_text
        cached_render = page_module._render_page_text
        for seed in range(4):
            hook = lambda: partial(augment_rendered_text, rng=np.random.default_rng(seed))
            actual_pages = page_module.layout_text_pages("Draw: cat. Once upon a time, there was a fish.",
                                                         font, augment=hook())
            try:
                page_module._render_page_text = cached_render.__wrapped__
                expected_pages = page_module.layout_text_pages("Draw: cat. Once upon a time, there was a fish.",
                                                               font, augment=hook())
            finally:
                page_module._render_page_text = cached_render
            assert len(actual_pages) == len(expected_pages)
            for a, b in zip(actual_pages, expected_pages):
                assert a.metadata == b.metadata
                assert all(np.array_equal(x.rawxy, y.rawxy) for x, y in zip(a.elements, b.elements))
        report["text_layout_cache"] = "exact_match"
    print(json.dumps(report))


if __name__ == "__main__":
    main()
