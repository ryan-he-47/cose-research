"""1×1 页排版与可见文档控制笔画（2026-10-02）。

本模块只输出最终画布坐标。编码器的固定坐标重映射由 collate 统一完成；
文字、草图、圈选和控制笔画均保留确定性笔顺。文字增强 hook 在排版后执行，
控制符随后添加，不受文字变形影响。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
import re
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from src.text_renderer import RenderedText, VectorFont, resample_stroke


_CONTROL_KINDS = frozenset(("bos", "eop", "eod"))
_BODY_KINDS = frozenset(("text", "sketch", "selection"))


@dataclass(frozen=True)
class PageLayoutConfig:
    body_box: tuple[float, float, float, float] = (0.06, 0.10, 0.94, 0.86)
    font_size: float = 0.085
    line_height: float = 1.35
    soft_chars: int = 48
    candidate_chars: int = 64
    point_spacing: float = 0.001
    max_points: int = 76
    # 2026-10-03：训练/推理固定同一内框；局部字形伸缩使用预留余量，不改变字槽。
    max_text_scale: float = 1.20
    bend_em_margin: float = 0.06

    def __post_init__(self):
        _validate_box(self.body_box)
        if not (math.isfinite(self.font_size) and self.font_size > 0):
            raise ValueError("font_size 必须为正数")
        if self.line_height <= 0 or self.soft_chars <= 0:
            raise ValueError("line_height / soft_chars 必须为正数")
        if self.candidate_chars < self.soft_chars:
            raise ValueError("candidate_chars 不能小于 soft_chars")
        if self.max_text_scale < 1 or self.bend_em_margin < 0:
            raise ValueError("文字增强预留范围无效")


def _validate_box(box):
    x0, y0, x1, y1 = map(float, box)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"范围须为 1×1 页内非空 bbox：{box}")


def _points(stroke, *, copy=True):
    points = np.array(stroke, dtype=np.float32, copy=copy)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("笔画必须为非空 N×2 点列")
    if not np.isfinite(points).all():
        raise ValueError("笔画包含非有限坐标")
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)
    return points


@dataclass(frozen=True)
class StrokeElement:
    rawxy: np.ndarray
    page_id: int
    targetable: bool
    kind: str
    char_index: int | None = None

    def __post_init__(self):
        if self.kind not in _CONTROL_KINDS | _BODY_KINDS:
            raise ValueError(f"未知笔画种类：{self.kind}")
        if self.page_id < 0:
            raise ValueError("page_id 不能为负")
        points = _points(self.rawxy)
        if np.any(points < -1e-6) or np.any(points > 1 + 1e-6):
            raise ValueError(f"笔画超出 1×1 页：{self.kind}")
        points.flags.writeable = False
        object.__setattr__(self, "rawxy", points)


@dataclass(frozen=True)
class Page:
    page_id: int
    modality: str
    elements: tuple[StrokeElement, ...]
    targetable: bool = True
    text: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.modality not in _BODY_KINDS:
            raise ValueError(f"未知页模态：{self.modality}")
        if any(e.page_id != self.page_id for e in self.elements):
            raise ValueError("同页元素须共享 page_id")
        if any(e.kind not in _CONTROL_KINDS and e.kind != self.modality for e in self.elements):
            raise ValueError("每页正文只使用一种模态")

    @property
    def strokes(self) -> tuple[np.ndarray, ...]:
        return tuple(e.rawxy for e in self.elements)

    @property
    def body_elements(self) -> tuple[StrokeElement, ...]:
        return tuple(e for e in self.elements if e.kind not in _CONTROL_KINDS)


@dataclass(frozen=True)
class Document:
    pages: tuple[Page, ...]
    complete: bool = True
    metadata: dict = field(default_factory=dict)
    prefix: tuple[StrokeElement, ...] = ()

    def __post_init__(self):
        if not self.pages:
            raise ValueError("文档至少包含一页")
        ids = [p.page_id for p in self.pages]
        if any(b != a + 1 for a, b in zip(ids, ids[1:])):
            raise ValueError("文档页 id 必须连续递增")
        for page in self.pages:
            if not page.elements or page.elements[-1].kind != "eop":
                raise ValueError("EOP 必须为每页最后一笔")
        if self.complete and (len(self.pages[-1].elements) < 2 or
                              self.pages[-1].elements[-2].kind != "eod"):
            raise ValueError("完整文档最后两笔必须为 EOD → EOP")

    def to_sequence(self) -> tuple[StrokeElement, ...]:
        """序列元素的 targetable 是该笔本身的监督标记；collate 再右移目标。"""
        return self.prefix + tuple(e for p in self.pages for e in p.elements)

    @property
    def stroke_count(self):
        return len(self.prefix) + sum(len(p.elements) for p in self.pages)

    def window(self, start_page: int = 0, max_strokes: int = 1024) -> "Document":
        """取完整页窗口。中段以前一页真实 EOP 为条件，不伪造 BOS。"""
        if not 0 <= start_page < len(self.pages) or max_strokes <= 0:
            raise ValueError("窗口起点或笔画预算无效")
        if start_page:
            prefix = (replace(self.pages[start_page - 1].elements[-1], targetable=False),)
            prefix_local_ids = [len(self.pages[start_page - 1].elements)-1]
        else:
            prefix = self.prefix
            prefix_local_ids = self.metadata.get('_prefix_local_ids', [])
        selected, used = [], len(prefix)
        for page in self.pages[start_page:]:
            if used + len(page.elements) > max_strokes:
                break
            selected.append(page)
            used += len(page.elements)
        if not selected:
            count = len(self.pages[start_page].elements) + len(prefix)
            raise ValueError(f"完整页需要 {count} 笔，超过窗口 {max_strokes}；请调整排版/预算")
        end = start_page + len(selected)
        return Document(tuple(selected), complete=self.complete and end == len(self.pages),
                        metadata={**self.metadata, "window_pages": (start_page, end),
                                  "_prefix_local_ids": prefix_local_ids}, prefix=prefix)

    def iter_windows(self, max_strokes: int = 1024) -> Iterator["Document"]:
        start = 0
        while start < len(self.pages):
            window = self.window(start, max_strokes)
            yield window
            start += len(window.pages)


def control_stroke(kind: str, *, point_spacing: float = 0.001,
                   max_points: int = 76) -> np.ndarray:
    """三个页脚标记各一笔；不使用文字字形或画布外地标。"""
    paths = {
        "bos": ((0.06, 0.925, 0.08, 0.040),
                ((0, .5), (.25, .5), (.25, 0), (.75, 0), (.75, 1), (1, 1), (1, .5))),
        "eod": ((0.46, 0.925, 0.08, 0.040),
                ((0, 0), (.35, 1), (.5, .35), (.65, 1), (1, 0), (.5, .65), (.5, 0))),
        "eop": ((0.84, 0.925, 0.10, 0.040),
                ((0, .15), (.2, .15), (.4, .5), (.2, .85), (.6, .85), (1, .5), (.6, .15))),
    }
    if kind not in paths:
        raise ValueError(f"未知控制符：{kind}")
    (x, y, width, height), points = paths[kind]
    xy = np.asarray(points) * [width, height] + [x, y]
    return resample_stroke(xy, point_spacing, max_points)


def _canonical_words(text: str):
    """合并空白为普通词间空格；段落位置仍用于分页优先级。"""
    matches = list(re.finditer(r"\S+", text))
    canonical, spans, breaks, cursor = [], [], [], 0
    for index, match in enumerate(matches):
        if index:
            canonical.append(" ")
            cursor += 1
        word = match.group()
        start = cursor
        canonical.append(word)
        cursor += len(word)
        spans.append((start, cursor))
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        gap = text[match.end():next_start]
        breaks.append("\n" in gap or bool(re.search(r'[.!?]["\')\]]*$', word)))
    return "".join(canonical), spans, breaks


def _page_chunks(text: str, config: PageLayoutConfig):
    canonical, spans, breaks = _canonical_words(text)
    # 2026-10-11：异常无空格长串不能整词压进一页；仅对它作确定性字符分页。
    # 普通词沿用原词/句边界选择，不在拆开的词中补空格。
    split_spans, split_breaks = [], []
    for (lo, hi), boundary in zip(spans, breaks):
        for part in range(lo, hi, config.candidate_chars):
            end = min(part + config.candidate_chars, hi)
            split_spans.append((part, end))
            split_breaks.append(boundary and end == hi)
    spans, breaks = split_spans, split_breaks
    start = 0
    while start < len(spans):
        candidates = []
        for end in range(start + 1, len(spans) + 1):
            count = spans[end - 1][1] - spans[start][0]
            if count > config.candidate_chars and end > start + 1:
                break
            candidates.append((end, count))
            if count >= config.candidate_chars:
                break
        # 句末/段落在软目标附近优先；其余取最接近目标的完整词边界。
        preferred = [(end, count) for end, count in candidates
                     if breaks[end - 1] and count >= config.soft_chars * 0.60]
        end, _ = min(preferred or candidates, key=lambda item: abs(item[1] - config.soft_chars))
        source_start, source_end = spans[start][0], spans[end - 1][1]
        yield canonical[source_start:source_end], (source_start, source_end), canonical
        start = end


def _wrap_words(text: str, font: VectorFont, size: float, width: float) -> str:
    lines, line = [], ""
    for word in text.split(" "):
        candidate = line + " " + word if line else word
        if line and font.measure(candidate, size) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return "\n".join(lines)


@lru_cache(maxsize=2048)
def _render_page_text(text: str, font: VectorFont, config: PageLayoutConfig):
    # 2026-10-03：只缓存确定性排版；字形增强仍在调用方按原 RNG 每次重新采样。
    x0, y0, x1, y1 = config.body_box
    margin = config.bend_em_margin * config.font_size
    width = (x1 - x0 - 2 * margin) / config.max_text_scale
    height = (y1 - y0 - 2 * margin) / config.max_text_scale
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    inner_x, inner_y = center[0] - width / 2, center[1] - height / 2
    longest = max(font.measure(word, 1.0) for word in text.split(" "))
    size = min(config.font_size, width / longest if longest else config.font_size)
    for _ in range(20):
        wrapped = _wrap_words(text, font, size, width)
        rendered = font.render(wrapped, font_size=size, origin=(0, 0),
                               line_height=config.line_height, point_spacing=config.point_spacing,
                               max_points=config.max_points)
        if rendered.bounds is None:
            return rendered, size, center
        bx0, by0, bx1, by1 = rendered.bounds
        scale = min(1.0, width / max(bx1 - bx0, 1e-12), height / max(by1 - by0, 1e-12))
        if scale >= 1.0 - 1e-6:
            # 留出同样的右/下侧余量；不以正文 ink 宽度填满内框。
            offset = np.array((inner_x - bx0, inner_y - by0), dtype=np.float32)
            placed = rendered.translated(offset)
            for stroke in placed.strokes:
                stroke.flags.writeable = False
            return placed, size, center
        size *= scale * 0.99
    raise ValueError("文字页无法排入正文范围")


def layout_text_pages(text: str, font: VectorFont, *,
                      config: PageLayoutConfig = PageLayoutConfig(), page_id_start: int = 0,
                      targetable: bool = True, augment: Callable | None = None) -> tuple[Page, ...]:
    """词边界分页，返回尚未添加控制符的正文页。

    augment 可传 functools.partial(augment_rendered_text, rng=..., config=...)；
    hook 接收 rendered、font_size 和正文中心 origin，返回 RenderedText。
    字槽、字距、基线、换行和分页先确定；增强只修改各字形笔画。
    """
    pages = []
    for index, (chunk, span, canonical) in enumerate(_page_chunks(text, config)):
        rendered, size, center = _render_page_text(chunk, font, config)
        layout = rendered.glyph_placements
        if augment is not None:
            rendered = augment(rendered, font_size=size, origin=center)
        page_id = page_id_start + index
        x0, y0, x1, y1 = config.body_box
        for stroke in rendered.strokes:
            if np.any(stroke < np.array((x0, y0)) - 1e-6) or np.any(stroke > np.array((x1, y1)) + 1e-6):
                raise ValueError("文字增强超出正文范围，请增加排版余量")
        elements = tuple(StrokeElement(s, page_id, targetable, "text", owner)
                         for s, owner in zip(rendered.strokes, rendered.stroke_char_indices))
        separator = " " if span[0] and canonical[span[0] - 1] == " " else ""
        metadata = {"source_span": span, "separator_before": separator,
                    "canonical_text": canonical, "rendered_text": rendered.text,
                    "font": font.name, "font_size": size, "body_box": config.body_box,
                    "augmentation_origin": center,
                    "glyph_placements": layout, "advance_width": rendered.advance_width}
        pages.append(Page(page_id, "text", elements, targetable, chunk, metadata))
    return tuple(pages)


def fit_sketch(strokes: Iterable[np.ndarray], *,
               box: tuple[float, float, float, float] = (0.06, 0.10, 0.94, 0.86),
               max_points: int = 76) -> tuple[np.ndarray, ...]:
    """整幅草图等比居中，只稀疏超长笔；可给场景对象传任意页内 bbox。"""
    _validate_box(box)
    if max_points < 2:
        raise ValueError("max_points 至少为 2")
    strokes = tuple(_points(s) for s in strokes)
    if not strokes:
        return ()
    points = np.concatenate(strokes)
    lo, hi = points.min(axis=0), points.max(axis=0)
    x0, y0, x1, y1 = box
    size = np.array((x1 - x0, y1 - y0))
    span = hi - lo
    nonzero = span > 1e-12
    scale = float(np.min(size[nonzero] / span[nonzero])) if np.any(nonzero) else 1.0
    offset = np.array(((x0 + x1) / 2, (y0 + y1) / 2)) - (lo + hi) * scale / 2
    return tuple(_cap_sketch_stroke(s * scale + offset, max_points) for s in strokes)


def _cap_sketch_stroke(points: np.ndarray, max_points: int) -> np.ndarray:
    """保留短 RAW 采样；超过 cap 后按弧长稀疏，保留首尾。"""
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= max_points:
        return points.copy()
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points.astype(np.float64), axis=0), axis=1))]
    if distance[-1] <= 1e-12:
        return np.repeat(points[:1], 2, axis=0)
    keep = np.r_[True, np.diff(distance) > 1e-12]
    grid = np.linspace(0, distance[-1], max_points)
    return np.column_stack([np.interp(grid, distance[keep], points[keep, axis])
                            for axis in range(2)]).astype(np.float32)


def sketch_page(strokes: Iterable[np.ndarray], *, page_id: int = 0, targetable: bool = True,
                box: tuple[float, float, float, float] = (0.06, 0.10, 0.94, 0.86),
                metadata: dict | None = None, fit: bool = True) -> Page:
    """fit=False 用于已处于最终页坐标的场景，不对布局再次归一化。"""
    raw = fit_sketch(strokes, box=box) if fit else tuple(_points(s) for s in strokes)
    elements = tuple(StrokeElement(s, page_id, targetable, "sketch") for s in raw)
    return Page(page_id, "sketch", elements, targetable, metadata=metadata or {})


def selection_page(box: tuple[float, float, float, float], *, page_id: int = 0,
                   targetable: bool = True, metadata: dict | None = None) -> Page:
    """与原图共享坐标的单笔椭圆圈选页。"""
    _validate_box(box)
    x0, y0, x1, y1 = box
    theta = np.linspace(-math.pi / 2, 3 * math.pi / 2, 76)
    xy = np.column_stack(((x0 + x1) / 2 + (x1 - x0) / 2 * np.cos(theta),
                          (y0 + y1) / 2 + (y1 - y0) / 2 * np.sin(theta))).astype(np.float32)
    return Page(page_id, "selection", (StrokeElement(xy, page_id, targetable, "selection"),),
                targetable, metadata={"selection_box": box, **(metadata or {})})


def build_document(pages: Sequence[Page], *, page_id_start: int = 0, complete: bool = True,
                   add_bos: bool = True, metadata: dict | None = None) -> Document:
    """组合条件页与回答页；条件页用 targetable=False，提示文档用 complete=False。"""
    if not pages:
        raise ValueError("文档至少包含一页")
    # 提示末页不加 EOD；其 EOP 由环境提供。BOS 只有真实文档起点才添加。
    finished = bool(complete and pages[-1].targetable)
    result = []
    for index, page in enumerate(pages):
        page_id = page_id_start + index
        elements = [replace(e, page_id=page_id, targetable=page.targetable) for e in page.body_elements]
        if index == 0 and add_bos:
            elements.insert(0, StrokeElement(control_stroke("bos"), page_id, False, "bos"))
        if index == len(pages) - 1 and finished:
            elements.append(StrokeElement(control_stroke("eod"), page_id, page.targetable, "eod"))
        elements.append(StrokeElement(control_stroke("eop"), page_id, page.targetable, "eop"))
        result.append(replace(page, page_id=page_id, elements=tuple(elements)))
    return Document(tuple(result), finished, metadata or {})


def iter_document_strokes(document: Document) -> Iterator[StrokeElement]:
    """AE 读取所有元素，不使用 R 的 targetable 筛选。"""
    return iter(document.to_sequence())
