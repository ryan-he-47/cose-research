"""2026-10-03：查看试训 AE 原笔/重建及冻结 R 的来源，CPU 执行。"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.demo.demo_ae_reconstruction import COLORS, reconstruct_elements
from src.st.pipeline_pretrain import build_document_dataset
from src.st.train_ae import StrokeAutoencoder, config_from_checkpoint, load_ae_weights


def load_model(state):
    model = StrokeAutoencoder(config_from_checkpoint(state)).eval()
    load_ae_weights(model, state)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae", type=Path, required=True)
    parser.add_argument("--r", type=Path, required=True)
    parser.add_argument("--old_ae", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    state = torch.load(args.ae, map_location="cpu", weights_only=False)
    current = load_model(state)
    report = {"ae_step": state["step"], "stroke_counts": state["stroke_counts"],
              "ae_train_args": state["train_args"], "ae": str(args.ae.resolve()),
              "r": str(args.r.resolve()), "old_ae": str(args.old_ae.resolve())}
    data_args = {**state["data_args"], "split": "valid"}
    r_state = torch.load(args.r, map_location="cpu", weights_only=False)
    report.update(r_step=r_state["step"], r_counters=r_state["counters"],
                  r_train_args=r_state["train_args"], r_ae_source=r_state["ae_source"],
                  r_weights_finite=all(bool(torch.isfinite(v).all()) for v in r_state["model"].values()),
                  frozen_ae_equal=all(torch.equal(v, r_state["model"][k])
                                      for k, v in state["model"].items()))
    if not report["frozen_ae_equal"] or not report["r_weights_finite"]:
        raise RuntimeError("R 权重或冻结 AE 来源异常")
    del state, r_state
    gc.collect()
    old_state = torch.load(args.old_ae, map_location="cpu", weights_only=False)
    old = load_model(old_state)
    del old_state
    gc.collect()
    dataset = build_document_dataset(**data_args)
    classes = dataset.drawings.classes
    examples = [("T1", 0, 0), ("T1", 0, 1)]
    for label in ("cat", "fish", "airplane", "bicycle"):
        examples.append(("O1", classes.index(label), -1))
    examples.extend((("S2", 0, -1), ("S4", 0, -1)))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(9, 3 * len(examples)), dpi=130)
    FigureCanvasAgg(figure)
    rows = []
    for row, (task, index, page_index) in enumerate(examples):
        document = dataset.get_for_task(task, index)
        page = document.pages[page_index]
        elements = tuple(e for e in document.prefix if e.page_id == page.page_id) + page.elements
        source, before = reconstruct_elements(old, elements)
        _, after = reconstruct_elements(current, elements)
        label = document.metadata.get("label", task)
        for column, (title, strokes) in enumerate((("Source", source), ("Old AE", before),
                                                   ("Adapted AE / step 6030", after))):
            axis = figure.add_subplot(len(examples), 3, row * 3 + column + 1)
            axis.set(xlim=(0, 1), ylim=(1, 0), aspect="equal", xticks=[], yticks=[])
            axis.axhspan(.90, 1, color="#f1f5f9")
            for points, element in zip(strokes, elements):
                axis.plot(points[:, 0], points[:, 1], color=COLORS[element.kind], linewidth=.9)
            axis.set_title(f"{title}\n{label}, page {page.page_id}, {len(elements)} strokes", fontsize=9)
        rows.append({"task": task, "index": index, "page": page.page_id,
                     "text": page.text, "metadata": document.metadata,
                     "kinds": dict(Counter(e.kind for e in elements))})
    figure.tight_layout()
    figure.savefig(args.out_dir / "ae_comparison.png", facecolor="white")
    figure.savefig(args.out_dir / "ae_comparison.svg", facecolor="white")
    report.update(examples=rows, split="valid", decoder="highest_pi_component_mean",
                  device="cpu", dtype="float32")
    (args.out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
