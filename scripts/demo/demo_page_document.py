"""生成文字增强、条件/回答页和圈选样张；CPU 核验页协议（2026-10-02）。"""
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
from src.page_document import (build_document, control_stroke, fit_sketch,  # noqa: E402
                               layout_text_pages, selection_page, sketch_page)
from src.text_augmentation import augment_rendered_text  # noqa: E402
from src.text_renderer import VectorFont  # noqa: E402


STORY = (
    'Lily found a tiny bird beside the old tree. "Can you fly?" she asked. '
    'The bird shook its head, so Lily made a little nest. '
    'She put the nest in a warm place and brought the bird some water.\n\n'
    'After a while, the bird felt better. It spread its wings and flew above the garden! '
    'Lily waved and said, "Come back tomorrow!"'
)


def cat_strokes():
    """固定笔顺小草图，避免 CPU 演示依赖整个 QuickDraw 缓存。"""
    return (
        np.array(((0, 0), (1, 0), (.8, -.65), (.55, -.25), (.2, -.65), (0, 0)), np.float32),
        np.array(((0, 0), (-.1, .7), (.3, 1), (.9, .8), (1, 0)), np.float32),
        np.array(((.15, .15), (.25, .15)), np.float32),
        np.array(((.65, .15), (.75, .15)), np.float32),
        np.array(((.4, .38), (.5, .5), (.6, .38)), np.float32),
        np.array(((.1, .4), (-.4, .25)), np.float32),
        np.array(((.9, .4), (1.4, .25)), np.float32),
    )


def write_sample(document, output: Path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    columns = min(4, len(document.pages))
    rows = math.ceil(len(document.pages) / columns)
    figure = Figure(figsize=(3.5 * columns, 3.8 * rows), dpi=150)
    FigureCanvasAgg(figure)
    colors = {"bos": "#2d69ad", "eop": "#2d69ad", "eod": "#2d69ad"}
    for index, page in enumerate(document.pages):
        ax = figure.add_subplot(rows, columns, index + 1)
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axhspan(.90, 1, color="#f1f5f9")
        for element in page.elements:
            xy = element.rawxy
            ax.plot(xy[:, 0], xy[:, 1], color=colors.get(element.kind, "#171717"),
                    linewidth=1.25, solid_capstyle="round", solid_joinstyle="round")
        role = "target" if page.targetable else "condition"
        ax.set_title(f"page {page.page_id}: {page.modality}, {role}, {len(page.elements)} strokes",
                     size=9)
    figure.tight_layout(pad=1.0)
    figure.savefig(output.with_suffix(".png"), facecolor="white")
    figure.savefig(output.with_suffix(".svg"), facecolor="white")


def protocol_smoke(font: VectorFont):
    pages = layout_text_pages(STORY, font)
    canonical = pages[0].metadata["canonical_text"]
    restored = "".join(p.metadata["separator_before"] + p.text for p in pages)
    assert restored == canonical
    for page in pages:
        assert page.metadata["rendered_text"].replace("\n", " ") == page.text
        assert all(page.text[e.char_index] != " " for e in page.elements)
    document = build_document(pages)
    sequence = document.to_sequence()
    assert sequence[0].kind == "bos" and not sequence[0].targetable
    assert [e.kind for e in sequence[-2:]] == ["eod", "eop"]
    assert sum(e.kind == "bos" for e in sequence) == 1
    assert all(p.elements[-1].kind == "eop" for p in document.pages)
    assert all(e.page_id == p.page_id for p in document.pages for e in p.elements)
    assert all(2 <= len(e.rawxy) <= 76 for e in sequence)
    budget = max(len(p.elements) for p in document.pages) + 1
    windows = list(document.iter_windows(budget))
    assert len(windows) > 1
    for window in windows[1:]:
        assert window.prefix[0].kind == "eop" and not window.prefix[0].targetable
        assert window.prefix[0].page_id + 1 == window.pages[0].page_id
        assert not any(e.kind == "bos" for e in window.to_sequence())
    long_word = "supercalifragilisticexpialidocious" * 3
    long_pages = layout_text_pages(long_word, font)
    assert len(long_pages) == 1 and long_pages[0].text == long_word
    assert long_pages[0].metadata["font_size"] < .085
    prompt = build_document(layout_text_pages("Draw a cat.", font, targetable=False), complete=False)
    assert not any(e.kind == "eod" or e.targetable for e in prompt.to_sequence())
    source = fit_sketch(cat_strokes())
    assert all(np.isfinite(s).all() for s in source)
    assert all(len(control_stroke(kind)) >= 2 for kind in ("bos", "eop", "eod"))
    return {"pages": len(document.pages), "strokes": len(sequence),
            "window_budget": budget, "window_strokes": [w.stroke_count for w in windows],
            "long_word_font_size": long_pages[0].metadata["font_size"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path,
                        default=ROOT / "datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = args.out or ROOT / "tmp_smoke" / ("page_document_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    font = VectorFont.from_svg(args.font)
    report = protocol_smoke(font)
    augmentation = partial(augment_rendered_text, rng=np.random.default_rng(args.seed))
    clean = build_document(layout_text_pages(STORY, font), metadata={"task": "text_lm"})
    augmented = build_document(layout_text_pages(STORY, font, augment=augmentation),
                               metadata={"task": "text_lm", "text_augmentation_seed": args.seed})
    for before, after in zip(clean.pages, augmented.pages):
        assert before.text == after.text
        assert before.metadata == after.metadata
        assert [e.char_index for e in before.elements] == [e.char_index for e in after.elements]
        for a, b in zip(before.elements, after.elements):
            if a.kind in ("bos", "eod", "eop"):
                assert np.array_equal(a.rawxy, b.rawxy)
    # 2026-10-03：多个种子都不能改变换行、字槽、字距或控制符。
    for seed in range(args.seed, args.seed + 16):
        changed = build_document(layout_text_pages(STORY, font, augment=partial(
            augment_rendered_text, rng=np.random.default_rng(seed))))
        assert len(changed.pages) == len(clean.pages)
        for before, after in zip(clean.pages, changed.pages):
            assert before.metadata == after.metadata and before.text == after.text
            assert [e.char_index for e in before.elements] == [e.char_index for e in after.elements]
            for a, b in zip(before.elements, after.elements):
                if a.kind in ("bos", "eod", "eop"):
                    assert np.array_equal(a.rawxy, b.rawxy)
    report["fixed_layout_seeds"] = 16
    o1 = build_document((*layout_text_pages("Draw a cat.", font, targetable=False),
                         sketch_page(cat_strokes())), metadata={"task": "O1"})
    o2 = build_document((sketch_page(cat_strokes(), targetable=False),
                         *layout_text_pages("cat", font)), metadata={"task": "O2"})
    selected = build_document((sketch_page(cat_strokes(), targetable=False),
                               *layout_text_pages("Circle the cat.", font, targetable=False),
                               selection_page((.10, .15, .90, .85))), metadata={"task": "S4"})
    for name, document in (("text_clean", clean), ("text_augmented", augmented),
                           ("object_O1", o1), ("object_O2", o2), ("selection", selected)):
        write_sample(document, output / name)
        report[name] = {"pages": len(document.pages), "strokes": document.stroke_count,
                        "target_strokes": sum(e.targetable for e in document.to_sequence())}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
