"""生成两个单线字体的矢量样张；CPU 原型核验（2026-10-02）。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
from html import escape
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.text_renderer import VectorFont, flatten_svg_path  # noqa: E402
from src.text_augmentation import (TextAugmentationConfig, augment_rendered_text,
                                   sample_text_warp)  # noqa: E402


FONTS = {
    "relief": {
        "repository": "isdat-type/Relief-SingleLine",
        "revision": "01dfc5779ec1e9e4b288d96c6c96c23bfccbaf9d",
        "font": "fonts/open_svg/ReliefSingleLineSVG-Regular.svg",
        "sha256": "75f05a5b64ed6039c9816628ee051d98e16c19148a8268c63f5eccf8382479e2",
        "extra": ["OFL.txt", "README.md", "AUTHORS.txt", "CONTRIBUTORS.txt"],
        "license": "SIL OFL 1.1",
    },
    "normstroke": {
        "repository": "octycs/norm-stroke",
        "revision": "e4f7be3ee482d3026f19b7afae88c1e0c193316a",
        "font": "font/NormStroke.svg",
        "sha256": "acc05c1a18fa3528ca4c70c0847d65b9a319dbafa38fa7f1a60436cd3d243fbc",
        "extra": ["README.md"],
        "license": "CC0 1.0 (upstream README)",
    },
}

LINES = [
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    "0123456789  Il1  O0  rn-m  vv-w",
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~",
    "The little cat saw a big dog.",
    '"Hello!" said Lily. Is it 10 or 100?',
    "i I l 1 |  . , : ; ! ?  a e g q y",
]


def download_fonts(asset_dir: Path):
    """只下载约 250 KB 的字体与来源/许可，版本和字体 SHA256 固定。"""
    for name, info in FONTS.items():
        folder = asset_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for source_path in [info["font"], *info["extra"]]:
            target = folder / Path(source_path).name
            url = f'https://raw.githubusercontent.com/{info["repository"]}/{info["revision"]}/{source_path}'
            data = urllib.request.urlopen(url, timeout=30).read()
            if source_path == info["font"] and hashlib.sha256(data).hexdigest() != info["sha256"]:
                raise ValueError(f"字体 SHA256 不匹配：{name}")
            target.write_bytes(data)
        source = {**info, "repository": "https://github.com/" + info["repository"],
                  "download_date": datetime.now().date().isoformat()}
        (folder / "SOURCE.json").write_text(json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8")


def smoke(font: VectorFont):
    ascii_text = "".join(chr(c) for c in range(32, 127))
    supported = "".join(c for c in ascii_text if c in font.glyphs)
    rendered = font.render(supported, font_size=0.04)
    assert all(2 <= len(s) <= 76 and np.isfinite(s).all() for s in rendered.strokes)
    assert list(rendered.stroke_char_indices) == sorted(rendered.stroke_char_indices)
    assert not font.render(" ").strokes and font.measure("  ", 0.1) > 0
    for char in ".,:;!?iI":
        assert font.render(char).strokes, f"短笔/标点丢失：{char!r}"
    try:
        font.render("汉")
    except ValueError:
        pass
    else:
        raise AssertionError("未知字符没有报错")
    # 笔画分段、圆弧方向、S 镜像控制点的简短解析核验。
    strokes = flatten_svg_path("M0 0 L10 0 M10 10 h5", 0.01)
    assert len(strokes) == 2 and np.allclose(strokes[1][-1], [15, 10])
    circle = flatten_svg_path("M10 0 A10 10 0 0 1 -10 0 A10 10 0 0 1 10 0", 0.01)[0]
    assert np.allclose(circle[[0, -1]], [[10, 0], [10, 0]])
    assert np.max(np.abs(np.linalg.norm(circle, axis=1) - 10)) < 1e-8
    curve = flatten_svg_path("M0 0 C0 10 10 10 10 0 S20 -10 20 0", 0.01)[0]
    assert curve[:, 1].min() < -7 and curve[:, 1].max() > 7
    return {"font": font.name, "ascii_supported": len(supported),
            "ascii_missing": list(font.missing_characters(ascii_text)),
            "strokes": len(rendered.strokes),
            "point_count": sum(map(len, rendered.strokes)),
            "min_points": min(map(len, rendered.strokes)),
            "max_points": max(map(len, rendered.strokes))}


def write_sample(font: VectorFont, output: Path, info: dict):
    """SVG 内容全部来自实际采样笔画；字体名与缺字说明用普通文本标注。"""
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1040" viewBox="0 0 1 0.866667">',
           '<rect width="1" height="0.866667" fill="white"/>',
           f'<text x="0.045" y="0.035" font-family="sans-serif" font-size="0.017">{escape(font.name)} — {escape(info["license"])}</text>']
    missing = font.missing_characters("".join(chr(c) for c in range(32, 127)))
    note = "ASCII 95/95; original stroke order" if not missing else "ASCII 92/95; unsupported (left blank below): " + " ".join(missing)
    svg.append(f'<text x="0.045" y="0.059" font-family="sans-serif" font-size="0.012">{escape(note)}</text>')
    all_strokes = []
    for index, line in enumerate(LINES):
        # 对缺字只在样张插入等宽空位，并明确标注；render API 仍严格报错。
        shown = "".join(c if c in font.glyphs else " " for c in line)
        font_size = min(0.058, 0.88 / font.measure(shown, 1.0))
        baseline = 0.145 + 0.1 * index
        rendered = font.render(shown, font_size=font_size, origin=(0.055, baseline))
        for stroke in rendered.strokes:
            points = " ".join(f"{x:.7f},{y:.7f}" for x, y in stroke)
            svg.append(f'<polyline points="{points}" fill="none" stroke="#151515" stroke-width="0.0019" stroke-linecap="round" stroke-linejoin="round"/>')
        all_strokes.extend(rendered.strokes)
    svg.append('</svg>')
    output.with_suffix(".svg").write_text("\n".join(svg), encoding="utf-8")

    # PNG 方便在工作区查看样张；真实训练输入仍为上面的矢量点列。
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(12, 10.4), dpi=120)
    FigureCanvasAgg(figure)
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0.866667, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(0.045, 0.035, f'{font.name} — {info["license"]}', size=15, va="baseline")
    ax.text(0.045, 0.059, note, size=10, va="baseline")
    for stroke in all_strokes:
        ax.plot(stroke[:, 0], stroke[:, 1], color="#151515", linewidth=1.6,
                solid_capstyle="round", solid_joinstyle="round")
    figure.savefig(output.with_suffix(".png"), dpi=120, facecolor="white")


def smoke_augmentation(font: VectorFont, seed: int):
    """核验字槽、基线、字宽和笔顺固定，字形变动且采样可复现。"""
    config = TextAugmentationConfig()
    text = "The little cat.\\nI l 1 i,.;!? WAV xyz"
    base = font.render(text, font_size=0.06)
    original = tuple(stroke.copy() for stroke in base.strokes)
    left = augment_rendered_text(base, font_size=0.06, rng=np.random.default_rng(seed))
    right = augment_rendered_text(base, font_size=0.06, rng=np.random.default_rng(seed))
    assert all(np.array_equal(a, b) for a, b in zip(left.strokes, right.strokes))
    assert all(np.array_equal(a, b) for a, b in zip(base.strokes, original))
    assert augment_rendered_text(base, font_size=.06, rng=np.random.default_rng(seed),
                                config=TextAugmentationConfig(enabled=False)) is base
    max_start_error = 0.
    for sample_seed in range(seed, seed + 32):
        changed = augment_rendered_text(base, font_size=.06, rng=np.random.default_rng(sample_seed))
        assert changed.text == base.text and changed.stroke_char_indices == base.stroke_char_indices
        assert changed.glyph_placements == base.glyph_placements
        assert changed.advance_width == base.advance_width
        assert len(changed.strokes) == len(base.strokes)
        assert all(2 <= len(s) <= 76 and np.isfinite(s).all() for s in changed.strokes)
        for placement in base.glyph_placements:
            points = [s for s, c in zip(changed.strokes, changed.stroke_char_indices)
                      if c == placement.char_index]
            if not points:
                continue
            xy = np.concatenate(points)
            error = abs(float(xy[:, 0].min()) - placement.bounds[0])
            max_start_error = max(max_start_error, error)
            assert error < 1e-7
            assert float(xy[:, 0].max()) <= max(placement.bounds[2],
                                              placement.cell_right - .025 * .06) + 1e-7
    assert any(not np.array_equal(a, b) for a, b in zip(base.strokes, left.strokes))
    warps = [sample_text_warp(font_size=.06, rng=np.random.default_rng(i))
             for i in range(seed, seed + 128)]
    scales = np.array([[w.scale_x, w.scale_y] for w in warps])
    assert np.all((scales >= .9) & (scales <= 1.1)) and np.all(scales.std(axis=0) > .04)
    return {"seed": seed, "config": asdict(config), "checked_glyph_samples": 32,
            "sampled_scale_min": scales.min(axis=0).tolist(),
            "sampled_scale_max": scales.max(axis=0).tolist(),
            "max_glyph_start_error": max_start_error,
            "layout_unchanged": True, "preserved_strokes": len(base.strokes)}


def write_augmented_sample(font: VectorFont, output: Path, seed: int):
    """原样与七次字形增强；蓝色字槽原点及基线帮助直观看排版。"""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    config = TextAugmentationConfig()
    text = "The little cat saw a big dog."
    font_size = min(.055, .77 / font.measure(text, 1.))
    rng = np.random.default_rng(seed)
    figure = Figure(figsize=(12, 10.8), dpi=120)
    FigureCanvasAgg(figure)
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(.9, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    title = f"{font.name} — fixed layout, glyph-only augmentation"
    note = "sx/sy 0.90–1.10; bend 6% em; blue guides: fixed character origins / baselines"
    ax.text(.045, .035, title, size=15)
    ax.text(.045, .060, note, size=10)
    samples = []
    for index in range(8):
        baseline = .145 + .095 * index
        base = font.render(text, font_size=font_size, origin=(.07, baseline))
        rendered = base if index == 0 else augment_rendered_text(
            base, font_size=font_size, rng=rng, config=config)
        ax.plot([.06, .92], [baseline, baseline], color="#b8d7ed", lw=.6)
        for placement in base.glyph_placements:
            x = placement.origin[0]
            ax.plot([x, x], [baseline - font_size, baseline + .005], color="#d5e7f3", lw=.5)
        for stroke in rendered.strokes:
            ax.plot(stroke[:, 0], stroke[:, 1], color="#151515", linewidth=1.6,
                    solid_capstyle="round", solid_joinstyle="round")
        ax.text(.07, baseline + .025, "Original" if index == 0 else f"Sample {index}",
                size=9, color="#555555")
        samples.append({"text": text, "font_size": font_size, "bounds": rendered.bounds,
                        "strokes": len(rendered.strokes), "advance_width": rendered.advance_width,
                        "glyph_origins": [p.origin for p in rendered.glyph_placements]})
    figure.savefig(output.with_suffix(".svg"), facecolor="white")
    figure.savefig(output.with_suffix(".png"), dpi=120, facecolor="white")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "datasets/vector_fonts")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--download-fonts", action="store_true", help="下载固定版本小字体和许可")
    parser.add_argument("--font", choices=["all", *FONTS], default="all")
    parser.add_argument("--augment", action="store_true", help="额外核验训练增强并生成连续缩放/弯曲样张")
    parser.add_argument("--seed", type=int, default=42, help="增强样张的可复现种子")
    args = parser.parse_args()
    if args.download_fonts:
        download_fonts(args.assets)
    output = args.out or ROOT / "tmp_smoke" / ("vector_text_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    report = {}
    for name, info in FONTS.items():
        if args.font != "all" and args.font != name:
            continue
        source = args.assets / name / Path(info["font"]).name
        if not source.exists():
            raise FileNotFoundError(f"字体不存在：{source}；先加 --download-fonts")
        if hashlib.sha256(source.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"字体与固定版本不一致：{source}")
        font = VectorFont.from_svg(source)
        report[name] = smoke(font)
        write_sample(font, output / name, info)
        if args.augment:
            report[name]["augmentation"] = smoke_augmentation(font, args.seed)
            report[name]["augmentation_samples"] = write_augmented_sample(font, output / (name + "_augmented"), args.seed)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "report": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
