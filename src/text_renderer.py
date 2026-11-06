"""单线 SVG 字体 → 有序矢量笔画；训练数据和打字输入共用的原型（2026-10-02）。

仅依赖 NumPy；不读取旧手写字库。画布坐标 x 向右、y 向下，origin 是首行
基线。每个 SVG M 开始一笔，空格只移动光标；未知字符报错供数据准备层处理。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4,
          "Q": 4, "T": 2, "A": 7}


def _distance_to_segment(point, start, end):
    delta = end - start
    denominator = float(delta @ delta)
    t = 0.0 if denominator == 0 else float((point - start) @ delta) / denominator
    return float(np.linalg.norm(point - (start + np.clip(t, 0, 1) * delta)))


def _bezier(points, tolerance, out, depth=0):
    """de Casteljau 细分，以控制点到弦线段的距离限制折线误差。"""
    if depth >= 18 or max(_distance_to_segment(p, points[0], points[-1])
                          for p in points[1:-1]) <= tolerance:
        out.append(points[-1])
        return
    levels = [np.asarray(points)]
    while len(levels[-1]) > 1:
        levels.append((levels[-1][:-1] + levels[-1][1:]) / 2)
    _bezier([level[0] for level in levels], tolerance, out, depth + 1)
    _bezier([level[-1] for level in reversed(levels)], tolerance, out, depth + 1)


def _arc(start, end, rx, ry, rotation, large, sweep, tolerance, out):
    """SVG 椭圆弧端点参数转中心参数（W3C SVG implementation notes B.2）。"""
    rx, ry = abs(rx), abs(ry)
    if np.array_equal(start, end):
        return
    if rx == 0 or ry == 0:
        out.append(end)
        return
    angle = math.radians(rotation % 360)
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    transformed = rot.T @ ((start - end) / 2)
    x, y = transformed
    stretch = (x / rx) ** 2 + (y / ry) ** 2
    if stretch > 1:
        rx, ry = rx * math.sqrt(stretch), ry * math.sqrt(stretch)
    denominator = rx * rx * y * y + ry * ry * x * x
    factor = math.sqrt(max(0.0, (rx * rx * ry * ry - denominator) / denominator))
    if bool(large) == bool(sweep):
        factor = -factor
    center_local = factor * np.array([rx * y / ry, -ry * x / rx])
    center = rot @ center_local + (start + end) / 2
    u = (transformed - center_local) / [rx, ry]
    v = (-transformed - center_local) / [rx, ry]
    begin = math.atan2(u[1], u[0])
    delta = math.atan2(u[0] * v[1] - u[1] * v[0], float(u @ v))
    if sweep and delta < 0:
        delta += 2 * math.pi
    elif not sweep and delta > 0:
        delta -= 2 * math.pi

    def at(theta):
        return center + rot @ np.array([rx * math.cos(theta), ry * math.sin(theta)])

    def subdivide(a, b, pa, pb, depth=0):
        mid = (a + b) / 2
        pm = at(mid)
        if depth >= 18 or (abs(b - a) <= math.pi / 2 and
                           _distance_to_segment(pm, pa, pb) <= tolerance):
            out.append(pb)
        else:
            subdivide(a, mid, pa, pm, depth + 1)
            subdivide(mid, b, pm, pb, depth + 1)

    subdivide(begin, begin + delta, start, end)


@lru_cache(maxsize=4096)
def flatten_svg_path(data: str, tolerance: float = 1.0) -> tuple[np.ndarray, ...]:
    """将 SVG 路径拆成移笔分隔的折线，返回字体原生坐标。"""
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance 必须为正数")
    if _TOKEN.sub("", data).strip(" ,\t\r\n"):
        raise ValueError("路径包含不支持的 SVG 字符")
    tokens = _TOKEN.findall(data)
    strokes, active = [], []
    current = np.zeros(2)
    beginning = np.zeros(2)
    previous, control = "", None
    command, i = None, 0

    def flush():
        if active:
            points = np.asarray(active, dtype=np.float64)
            # 移笔后的孤立点也保留，后续采样复制为两点。
            points.flags.writeable = False
            strokes.append(points)
            active.clear()

    while i < len(tokens):
        if tokens[i].isalpha():
            command = tokens[i]
            i += 1
            if command.upper() == "Z":
                if active and not np.array_equal(current, beginning):
                    active.append(beginning.copy())
                current = beginning.copy()
                previous, control, command = "Z", None, None
                continue
        if command is None:
            raise ValueError("SVG 路径缺少命令")
        kind, relative = command.upper(), command.islower()
        size = _ARITY[kind]
        if i + size > len(tokens) or any(t.isalpha() for t in tokens[i:i + size]):
            raise ValueError(f"SVG {command} 参数不完整")
        values = np.array([float(t) for t in tokens[i:i + size]])
        i += size

        def xy(offset):
            return values[offset:offset + 2] + (current if relative else 0)

        next_control = None
        if kind == "M":
            flush()
            current = xy(0)
            beginning = current.copy()
            active.append(current.copy())
            command = "l" if relative else "L"
        elif not active:
            raise ValueError("SVG 绘制命令前缺少 M")
        elif kind in ("L", "H", "V"):
            end = xy(0) if kind == "L" else current.copy()
            if kind == "H":
                end[0] = values[0] + (current[0] if relative else 0)
            elif kind == "V":
                end[1] = values[0] + (current[1] if relative else 0)
            active.append(end)
            current = end
        elif kind in ("C", "S", "Q", "T"):
            reflected = 2 * current - control if control is not None else current
            if kind == "C":
                first, second, end = xy(0), xy(2), xy(4)
                controls = [current, first, second, end]
                next_control = second
            elif kind == "S":
                first = reflected if previous in ("C", "S") else current
                second, end = xy(0), xy(2)
                controls, next_control = [current, first, second, end], second
            elif kind == "Q":
                first, end = xy(0), xy(2)
                controls, next_control = [current, first, end], first
            else:
                first = reflected if previous in ("Q", "T") else current
                end = xy(0)
                controls, next_control = [current, first, end], first
            _bezier(controls, tolerance, active)
            current = end
        elif kind == "A":
            if values[3] not in (0, 1) or values[4] not in (0, 1):
                raise ValueError("SVG 弧线 flag 必须为 0 或 1")
            end = xy(5)
            _arc(current, end, *values[:5], tolerance, active)
            current = end
        previous, control = kind, next_control
    flush()
    return tuple(strokes)


def resample_stroke(points: np.ndarray, spacing: float, max_points: int = 76) -> np.ndarray:
    """折线上等弧长取点；保留首尾和至少两点，长笔画受 max_points 限制。"""
    if spacing <= 0 or not math.isfinite(spacing) or max_points < 2:
        raise ValueError("spacing 必须为正数，max_points 至少为 2")
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("空笔画")
    lengths = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if lengths[-1] <= 1e-12:
        return np.repeat(points[:1], 2, axis=0).astype(np.float32)
    keep = np.r_[True, np.diff(lengths) > 1e-12]
    lengths, points = lengths[keep], points[keep]
    count = min(max_points, max(2, math.ceil(lengths[-1] / spacing) + 1))
    positions = np.linspace(0, lengths[-1], count)
    return np.column_stack([np.interp(positions, lengths, points[:, j])
                            for j in range(2)]).astype(np.float32)


@dataclass(frozen=True)
class Glyph:
    character: str
    name: str
    advance: float
    path: str


@dataclass(frozen=True)
class GlyphPlacement:
    """2026-10-03：确定性字槽；空格同样记录游标推进，不由增强重新排版。"""

    char_index: int
    character: str
    origin: tuple[float, float]
    advance: float
    line_index: int
    bounds: tuple[float, float, float, float] | None
    cell_right: float


@dataclass(frozen=True)
class RenderedText:
    text: str
    strokes: tuple[np.ndarray, ...]
    stroke_char_indices: tuple[int, ...]
    advance_width: float
    bounds: tuple[float, float, float, float] | None
    glyph_placements: tuple[GlyphPlacement, ...] = ()

    def translated(self, offset) -> "RenderedText":
        """整段确定性排版平移；笔画和字槽使用完全相同的偏移。"""
        dx, dy = map(float, offset)
        shift_box = lambda box: None if box is None else (
            box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
        placements = tuple(replace(g, origin=(g.origin[0] + dx, g.origin[1] + dy),
                                  bounds=shift_box(g.bounds), cell_right=g.cell_right + dx)
                           for g in self.glyph_placements)
        return replace(self, strokes=tuple((s + [dx, dy]).astype(np.float32) for s in self.strokes),
                       bounds=shift_box(self.bounds), glyph_placements=placements)


class VectorFont:
    def __init__(self, name, units_per_em, ascent, descent, glyphs, kerning):
        self.name = name
        self.units_per_em = units_per_em
        self.ascent = ascent
        self.descent = descent
        self.glyphs = glyphs
        self.kerning = kerning

    @classmethod
    def from_svg(cls, path: str | Path) -> "VectorFont":
        root = ET.parse(path).getroot()
        font = next((e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "font"), None)
        if font is None:
            raise ValueError(f"不是 SVG font：{path}")
        face = next(e for e in font if e.tag.rsplit("}", 1)[-1] == "font-face")
        default_advance = float(font.get("horiz-adv-x", "0"))
        glyphs = {}
        for node in font:
            character = node.get("unicode")
            if node.tag.rsplit("}", 1)[-1] == "glyph" and character:
                # 首版逐字符排版，不自动合成 ligature。
                glyphs.setdefault(character, Glyph(character, node.get("glyph-name", character),
                    float(node.get("horiz-adv-x", default_advance)), node.get("d", "")))
        names = {g.name: char for char, g in glyphs.items()}
        kerning = {}
        for node in font:
            if node.tag.rsplit("}", 1)[-1] != "hkern":
                continue
            # 两个候选的 kerning 采用 g1/g2 字形名；也支持 u1/u2 字符列表。
            sides = []
            for index in (1, 2):
                chars = [names[n] for n in node.get(f"g{index}", "").split(",") if n in names]
                chars += [c for c in node.get(f"u{index}", "").split(",") if c in glyphs]
                sides.append(chars)
            for left in sides[0]:
                for right in sides[1]:
                    kerning[left, right] = float(node.get("k", "0"))
        return cls(face.get("font-family", font.get("id", "SVG font")),
                   float(face.get("units-per-em", "1000")), float(face.get("ascent", "0")),
                   abs(float(face.get("descent", "0"))), glyphs, kerning)

    def missing_characters(self, text: str) -> tuple[str, ...]:
        return tuple(sorted(set(text) - set(self.glyphs) - {"\n"}))

    def measure(self, text: str, font_size: float) -> float:
        """返回最长行的 advance 宽度；ink bbox 可通过 render 获取。"""
        missing = self.missing_characters(text)
        if missing:
            raise ValueError(f"{self.name} 缺少字符：{missing!r}")
        widths = []
        for line in text.split("\n"):
            width = sum(self.glyphs[c].advance for c in line)
            width -= sum(self.kerning.get((a, b), 0) for a, b in zip(line, line[1:]))
            widths.append(width * font_size / self.units_per_em)
        return max(widths, default=0.0)

    def render(self, text: str, *, font_size: float = 0.08, origin=(0.06, 0.14),
               line_height: float = 1.4, point_spacing: float = 0.001,
               flatness: float = 0.00015, max_points: int = 76) -> RenderedText:
        if font_size <= 0 or not math.isfinite(font_size) or line_height <= 0:
            raise ValueError("font_size / line_height 必须为正数")
        missing = self.missing_characters(text)
        if missing:
            raise ValueError(f"{self.name} 缺少字符：{missing!r}")
        scale = font_size / self.units_per_em
        x, baseline = map(float, origin)
        start_x = x
        strokes, owners, placements = [], [], []
        line_index = 0
        previous = None
        for index, character in enumerate(text):
            if character == "\n":
                x, baseline, previous = start_x, baseline + line_height * font_size, None
                line_index += 1
                continue
            glyph = self.glyphs[character]
            x -= self.kerning.get((previous, character), 0) * scale
            first_stroke = len(strokes)
            for polyline in flatten_svg_path(glyph.path, flatness / scale):
                xy = polyline * [scale, -scale] + [x, baseline]
                strokes.append(resample_stroke(xy, point_spacing, max_points))
                owners.append(index)
            glyph_bounds = None
            if len(strokes) > first_stroke:
                pts = np.concatenate(strokes[first_stroke:])
                glyph_bounds = (*map(float, pts.min(axis=0)), *map(float, pts.max(axis=0)))
            placements.append(GlyphPlacement(index, character, (x, baseline), glyph.advance * scale,
                                              line_index, glyph_bounds, x + glyph.advance * scale))
            x += glyph.advance * scale
            previous = character
        bounds = None
        if strokes:
            points = np.concatenate(strokes)
            bounds = (*map(float, points.min(axis=0)), *map(float, points.max(axis=0)))
        # 下一个字槽的起点包含确定性 kerning；末字使用原 advance。
        for index in range(len(placements) - 1):
            if placements[index + 1].line_index == placements[index].line_index:
                placements[index] = replace(placements[index], cell_right=placements[index + 1].origin[0])
        return RenderedText(text, tuple(strokes), tuple(owners), self.measure(text, font_size), bounds,
                            tuple(placements))
