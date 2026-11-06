"""冻结AE真实控制笔画回环、跨块EOD/EOP及块尾丢弃检查。"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.checks.check_page_diffusion import document
from src.page_document import control_stroke
from src.st.generation_pretrain import PageGenerator
from src.st.page_diffusion import PageDiffusionConfig, PageDiffusionR
from src.st.page_diffusion_codec import FrozenPageCodec
from src.st.page_diffusion_sampling import PageDiffusionGenerator


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda")
    ae = torch.load("tmp_smoke/r_dual_review_assets_20261009/ae_final.pt",
                    map_location="cpu", weights_only=False)
    codec = FrozenPageCodec.from_state(ae).to(device)
    del ae
    model = PageDiffusionR(PageDiffusionConfig(width=64, layers=2, heads=4, ffn=128,
        condition=16, activation_checkpoint=False)).to(device).eval()
    doc = document(1)
    prefix = doc.pages[0].elements
    geometry = PageGenerator(codec)
    encoded = geometry.encode([doc.pages[1].elements[0].rawxy,
                               control_stroke("eod"), control_stroke("eop")])
    for expected, points in zip(("body", "eod", "eop"), geometry.decode(encoded)):
        kind, _ = geometry.recognize(points)
        assert kind == expected, (kind, expected)

    def run(sequence):
        calls = []
        def fixed_block(m, cache, page, local, noise, steps):
            ids = sequence[len(calls)]
            calls.append(dict(page=page, local=local, history=cache[0][0].shape[-2]))
            return (encoded[ids][None], torch.full((1,16), page, device=device),
                    torch.arange(local, local+16, device=device)[None])
        with patch("src.st.page_diffusion_sampling.sample_block", fixed_block):
            result = PageDiffusionGenerator(model, codec).generate(prefix,
                max_new_strokes=48, max_new_pages=4)
        return result, calls

    across, calls = run([[0]*15+[1], [2]*16])
    assert across["stop"] == "eod_eop" and len(across["records"]) == 17
    assert calls[1]["page"] == calls[0]["page"]
    assert calls[1]["local"] == calls[0]["local"]+16
    assert across["blocks"][1]["retained"] == 1

    tail, calls = run([[0,2]+[0]*14, [1,2]+[2]*14])
    assert tail["stop"] == "eod_eop" and len(tail["records"]) == 4
    assert tail["blocks"][0]["discarded_non_eop"] == 14
    assert calls[1]["page"] == calls[0]["page"]+1 and calls[1]["local"] == 0
    assert calls[1]["history"] == calls[0]["history"]+2
    result = dict(ae_control_roundtrip=True, cross_block_eod_eop=True,
                  first_eop_discards_entire_tail=True, cache_keeps_only_retained=True,
                  next_page_local_zero=True)
    path = Path("tmp_smoke/r_page_diffusion_checks_20261102/stopping.json")
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
