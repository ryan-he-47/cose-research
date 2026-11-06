"""训练文字的字槽内连续增强（2026-10-03，用户要求排版固定）。

先确定字符原点、基线、字宽推进与换行，再独立改变各字形。增强不参与
排版；同一字的所有笔画共用参数，空格、字符顺序和控制符不受影响。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from src.text_renderer import GlyphPlacement, RenderedText, resample_stroke


@dataclass(frozen=True)
class TextAugmentationConfig:
    enabled: bool = True
    scale_x: tuple[float, float] = (0.90, 1.10)
    scale_y: tuple[float, float] = (0.90, 1.10)
    bend_fraction: float = 0.06
    wavelength_em: tuple[float, float] = (0.80, 1.80)
    point_spacing: float = 0.001
    max_points: int = 76

    def __post_init__(self):
        for name in ("scale_x", "scale_y", "wavelength_em"):
            low, high = getattr(self, name)
            if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
                raise ValueError(f"{name} 必须是有限的正数区间")
        if not math.isfinite(self.bend_fraction) or not 0 <= self.bend_fraction < 1:
            raise ValueError("bend_fraction 必须在 [0, 1) 内")
        if not math.isfinite(self.point_spacing) or self.point_spacing <= 0:
            raise ValueError("point_spacing 必须为有限正数")
        if self.max_points < 2:
            raise ValueError("max_points 至少为 2")


@dataclass(frozen=True)
class TextWarp:
    """字形变形参数。x 弯曲幅度为坐标单位，y 为基线距离的相对调制。"""
    scale_x: float
    scale_y: float
    amplitude_x: float
    bend_y: float
    wavelength_x: float
    wavelength_y: float
    phase_x: float
    phase_y: float


def sample_text_warp(*, font_size: float, rng: np.random.Generator,
                     config: TextAugmentationConfig | None = None) -> TextWarp:
    """采样一个字的参数；随机数由数据管线持有。"""
    config = config or TextAugmentationConfig()
    if not math.isfinite(font_size) or font_size <= 0:
        raise ValueError("font_size 必须为有限正数")
    if not config.enabled:
        return TextWarp(1., 1., 0., 0., font_size, font_size, 0., 0.)
    return TextWarp(float(rng.uniform(*config.scale_x)),
                    float(rng.uniform(*config.scale_y)),
                    float(rng.uniform(-1, 1) * config.bend_fraction * font_size / 2),
                    float(rng.uniform(-1, 1) * config.bend_fraction),
                    *map(float, rng.uniform(*config.wavelength_em, 2) * font_size),
                    *map(float, rng.uniform(0, 2 * np.pi, 2)))


def _warp_glyph(strokes, placement: GlyphPlacement, warp: TextWarp, *,
                font_size: float, config: TextAugmentationConfig):
    left, _, right, _ = placement.bounds
    width = right - left
    # 允许字体原有越界，但不增加字槽之间的侵入。字槽位置不随形变改变。
    available = max(right, placement.cell_right - .025 * font_size) - left
    amplitude = math.copysign(min(abs(warp.amplitude_x),
        max(0., available - min(config.scale_x[0], 1.) * width) / 2), warp.amplitude_x)
    scale_x = warp.scale_x
    if width > 1e-12:
        scale_x = min(scale_x, max(0., available - 2 * abs(amplitude)) / width)
    prepared = [resample_stroke(s, config.point_spacing, config.max_points)
                .astype(np.float64) for s in strokes]
    warped = []
    for points in prepared:
        x = points[:, 0] - left
        y = points[:, 1] - placement.origin[1]
        result = np.empty_like(points)
        result[:, 0] = scale_x * x + amplitude * (
            np.sin(2 * np.pi * y / warp.wavelength_x + warp.phase_x)
            - math.sin(warp.phase_x))
        result[:, 1] = placement.origin[1] + warp.scale_y * y * (
            1 + warp.bend_y * np.sin(2 * np.pi * x / warp.wavelength_y + warp.phase_y))
        warped.append(result)
    # 全字统一回到原 ink 左边缘，行首和每个字符的起始横坐标固定。
    minimum = min(float(s[:, 0].min()) for s in warped)
    result = [resample_stroke(s + [left - minimum, 0], config.point_spacing,
                              config.max_points) for s in warped]
    # 弧长重采样可能略过极值点；仅平移同字笔画，恢复精确左边缘。
    correction = left - min(float(s[:, 0].min()) for s in result)
    return [np.asarray(s + [correction, 0], dtype=np.float32) for s in result]


def augment_rendered_text(rendered: RenderedText, *, font_size: float,
                          rng: np.random.Generator, origin=None,
                          config: TextAugmentationConfig | None = None) -> RenderedText:
    """排版后改变字形。origin 保留旧 hook 签名，不作为页面变形中心。"""
    config = config or TextAugmentationConfig()
    if not config.enabled or not rendered.strokes:
        return rendered
    if not rendered.glyph_placements:
        raise ValueError("字形增强需要渲染器提供确定性 glyph_placements")
    by_char = {}
    for index, owner in enumerate(rendered.stroke_char_indices):
        by_char.setdefault(owner, []).append(index)
    strokes = list(rendered.strokes)
    for placement in rendered.glyph_placements:
        indices = by_char.get(placement.char_index)
        if not indices:
            continue
        warp = sample_text_warp(font_size=font_size, rng=rng, config=config)
        changed = _warp_glyph([strokes[i] for i in indices], placement, warp,
                              font_size=font_size, config=config)
        for index, stroke in zip(indices, changed):
            strokes[index] = stroke
    points = np.concatenate(strokes)
    bounds = (*map(float, points.min(axis=0)), *map(float, points.max(axis=0)))
    return replace(rendered, strokes=tuple(strokes), bounds=bounds)
