"""用独立 AE checkpoint 逐页输出源/重建曲线；CPU 渲染，不训练（2026-10-02）。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.st.ae_data import cap_stroke  # noqa: E402
from src.st.pipeline_pretrain import build_document_dataset  # noqa: E402
from src.st.train_ae import (StrokeAutoencoder, config_from_checkpoint,  # noqa: E402
                             load_ae_weights)


COLORS = {"text": "#171717", "sketch": "#171717", "selection": "#ad258c",
          "bos": "#2d69ad", "eop": "#2d69ad", "eod": "#2d69ad"}


@torch.no_grad()
def reconstruct_elements(model, elements, *, max_points=76, chunk_size=32):
    """最终页坐标 → 同一固定映射 → E/D；有效整笔编码，不拼接不同笔画。"""
    source = [cap_stroke(element.rawxy, max_points) for element in elements]
    output = []
    for start in range(0, len(source), chunk_size):
        strokes = source[start:start + chunk_size]
        lengths = np.array([len(stroke) for stroke in strokes], dtype=np.int64)
        ink = np.zeros((len(strokes), int(lengths.max()), 2), dtype=np.float32)
        for row, points in enumerate(strokes):
            ink[row, :len(points)] = (points - np.array([0.5, 0.6], np.float32)) / 0.6
        batch = {"ink": torch.from_numpy(ink), "stroke_len": torch.from_numpy(lengths)}
        output.extend(model.reconstruct(batch, num_points=max_points).cpu().numpy())
    if not all(np.isfinite(points).all() for points in output):
        raise RuntimeError("AE 重建包含非有限坐标")
    return source, output


def write_page(source, reconstructed, elements, page, output: Path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8.0, 4.4), dpi=160)
    FigureCanvasAgg(figure)
    for column, (title, strokes) in enumerate((("source", source),
                                               ("AE reconstruction", reconstructed))):
        axis = figure.add_subplot(1, 2, column + 1)
        axis.set_xlim(0, 1)
        axis.set_ylim(1, 0)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.axhspan(.90, 1, color="#f1f5f9")
        for points, element in zip(strokes, elements):
            axis.plot(points[:, 0], points[:, 1], color=COLORS[element.kind],
                      linewidth=1.1, solid_capstyle="round", solid_joinstyle="round")
        axis.set_title(f"{title}: page {page.page_id}, {page.modality}, "
                       f"{len(strokes)} strokes", fontsize=10)
    figure.suptitle("black: text/sketch   purple: selection   blue: BOS/EOP/EOD", fontsize=9)
    figure.tight_layout(pad=1.1)
    figure.savefig(output.with_suffix(".png"), facecolor="white")
    figure.savefig(output.with_suffix(".svg"), facecolor="white")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--document_index", type=int, default=0,
                        help="provider 的全局文档索引，可包含重复轮次")
    parser.add_argument("--max_pages", type=int, default=3)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--cpu_threads", type=int, default=4)
    args = parser.parse_args()
    if args.document_index < 0 or min(args.max_pages, args.chunk_size, args.cpu_threads) < 1:
        parser.error("document_index >= 0，max_pages/chunk_size/cpu_threads 必须为正数")
    torch.set_num_threads(args.cpu_threads)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "data_args" not in state:
        raise ValueError("需要 train_ae.py 保存的 checkpoint（含 data_args）")
    cfg = config_from_checkpoint(state)
    model = StrokeAutoencoder(cfg).eval()
    load_ae_weights(model, state)
    dataset = build_document_dataset(**state["data_args"])
    document = dataset[args.document_index]
    output = args.out_dir or ROOT / "tmp_smoke" / (
        "ae_reconstruction_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    max_points = int(state.get("train_args", {}).get("max_points", 76))
    pages = []
    for page in document.pages[:args.max_pages]:
        elements = tuple(element for element in document.prefix if element.page_id == page.page_id)
        elements += page.elements
        source, reconstructed = reconstruct_elements(model, elements,
            max_points=max_points, chunk_size=args.chunk_size)
        name = output / f"page_{page.page_id:04d}"
        write_page(source, reconstructed, elements, page, name)
        pages.append({"page_id": page.page_id, "modality": page.modality,
                      "strokes": len(elements), "kinds": dict(Counter(e.kind for e in elements)),
                      "png": str(name.with_suffix(".png").resolve()),
                      "svg": str(name.with_suffix(".svg").resolve())})
    report = {"checkpoint": str(args.ckpt.resolve()), "checkpoint_step": state.get("step"),
              "document_index": args.document_index, "document_metadata": document.metadata,
              "document_pages": len(document.pages), "rendered_pages": len(pages),
              "device": "cpu", "dtype": "float32", "decoder": "highest_pi_component_mean",
              "max_points": max_points, "pages": pages}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
