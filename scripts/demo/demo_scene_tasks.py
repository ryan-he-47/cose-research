"""用少量本地 RAW 对象生成 S1–S5 场景样张与布局记录（CPU，2026-10-02）。"""
from __future__ import annotations

import argparse
from datetime import datetime
from functools import partial
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.scene_tasks import build_scene_document  # noqa: E402
from src.st.ae_data import QuickDrawRawSource  # noqa: E402
from src.text_augmentation import augment_rendered_text  # noqa: E402
from src.text_renderer import VectorFont  # noqa: E402


def write_sample(document, output):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    columns = min(4, len(document.pages))
    rows = math.ceil(len(document.pages) / columns)
    figure = Figure(figsize=(3.5 * columns, 3.8 * rows), dpi=150)
    FigureCanvasAgg(figure)
    for index, page in enumerate(document.pages):
        ax = figure.add_subplot(rows, columns, index + 1)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhspan(.90, 1, color="#f1f5f9")
        for element in page.elements:
            color = "#2d69ad" if element.kind in ("bos", "eop", "eod") else "#171717"
            ax.plot(element.rawxy[:, 0], element.rawxy[:, 1], color=color, linewidth=1.2,
                    solid_capstyle="round", solid_joinstyle="round")
        role = "target" if page.targetable else "condition"
        ax.set_title(f'{document.metadata["task"]}: page {page.page_id}, {page.modality}, {role}', size=9)
    figure.tight_layout(pad=1.0)
    figure.savefig(output.with_suffix(".png"), facecolor="white")
    figure.savefig(output.with_suffix(".svg"), facecolor="white")


def smoke(document, drawings):
    metadata = document.metadata
    objects = metadata["objects"]
    assert 2 <= len(objects) <= 4 and len({obj["label"] for obj in objects}) == len(objects)
    assert [obj["key_id"] for obj in objects] == [drawing.key_id for drawing in drawings[:len(objects)]]
    assert document.to_sequence()[-2].kind == "eod" and document.to_sequence()[-1].kind == "eop"
    assert all(not e.targetable for p in document.pages if not p.targetable for e in p.elements)
    assert all(np.isfinite(e.rawxy).all() for e in document.to_sequence())
    scene = next(p for p in document.pages if p.modality == "sketch")
    body = scene.body_elements
    assert len(metadata["stroke_owner"]) == len(body)
    for obj, drawing in zip(objects, drawings):
        start, end = obj["stroke_range"]
        assert metadata["stroke_owner"][start:end] == [obj["instance_id"]] * (end - start)
        assert end - start == len(drawing.strokes)
        affine = obj["affine"]
        for source, element in zip(drawing.strokes, body[start:end]):
            assert np.allclose(source[0] * affine["scale"] + affine["translation"],
                               element.rawxy[0], atol=2e-6)
        bbox = np.array(obj["bbox"])
        assert np.all(bbox[:2] >= [0.06, 0.10]) and np.all(bbox[2:] <= [0.94, 0.86])
    for index, a in enumerate(objects):
        for b in objects[index + 1:]:
            lo = np.maximum(a["bbox"][:2], b["bbox"][:2])
            hi = np.minimum(a["bbox"][2:], b["bbox"][2:])
            assert np.any(hi <= lo)
    if metadata["task"] in ("S4", "S5"):
        target = objects[metadata["target_instance_id"]]
        box = np.array(metadata["answer"]["bbox"] if metadata["task"] == "S4" else metadata["selection_box"])
        assert np.all(box[:2] >= [0.06, 0.10]) and np.all(box[2:] <= [0.94, 0.86])
        center, radius = (box[:2] + box[2:]) / 2, (box[2:] - box[:2]) / 2
        points = np.concatenate([e.rawxy for e in body[slice(*target["stroke_range"])]])
        assert np.all(np.sum(((points - center) / radius) ** 2, axis=1) <= 1 + 1e-5)
    return {"pages": len(document.pages), "strokes": document.stroke_count,
            "target_strokes": sum(e.targetable for e in document.to_sequence())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "datasets/quick_draw")
    parser.add_argument("--font", type=Path,
                        default=ROOT / "datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg")
    parser.add_argument("--objects", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = QuickDrawRawSource(args.raw_dir, max_per_class=1)
    if len(source.class_indices) < args.objects:
        raise ValueError("本地 RAW 的类别数少于指定对象数")
    drawings = [source.get(indices[0]) for indices in source.class_indices[:args.objects]]
    font = VectorFont.from_svg(args.font)
    output = args.out or ROOT / "tmp_smoke" / ("scene_tasks_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    report = {}
    for task in ("S1", "S2", "S3", "S4", "S5"):
        rng = np.random.default_rng(args.seed)
        augment = partial(augment_rendered_text, rng=rng)
        document = build_scene_document(task, drawings, font, rng, augment=augment)
        report[task] = smoke(document, drawings)
        write_sample(document, output / task)
        (output / f"{task}_layout.json").write_text(
            json.dumps(document.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    # 顺带查看首版问答模板是否覆盖 0/1、yes/no 与几何关系分支。
    observed = {}
    for seed in range(40):
        document = build_scene_document("S3", drawings, font, np.random.default_rng(seed))
        query = document.metadata["query"]
        key = f'{query["kind"]}:{document.metadata["answer"]}'
        observed[key] = observed.get(key, 0) + 1
    report["question_examples"] = observed
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
