"""
CoSE 多区域草图数据模型 — 重构版
===============================
三元组分区模型: Graphic (targetable) + Text (context) + Padding (discarded)

核心类:
  StrokeRegion       — 一类笔画的集合, 携带预处理策略
  TextLayoutConfig   — 画布固定排版配置 (训练/推理单一事实源)
  CompositeDiagram   — 多区域草图 (grpahic + text)
  DiagramComposer    — 图形+文字固定排版合成器
  StrokeDiagram      — 模型就绪格式 (translate + z-score 后)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Forward reference for letter database
if "src.letter_database" not in sys.modules:
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)


# ═══════════════════════════════════════════════════════════════════════
# 1. StrokeRegion — 一类笔画的集合
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StrokeRegion:
    """一类笔画的集合，携带自身预处理策略.

    Attributes:
        name:        区域名称 ("graphic", "text", "padding")
        ink:         笔画数据 (n_strokes, max_len, 4) — (x, y, t, pen)
        lens:        每笔有效点数 (n_strokes,)
        start_coord: 每笔起笔坐标 (n_strokes, 2) — 原始空间, 不 z-score
        role:        采样角色:
                       "targetable" — 可做 target, 可做 context
                       "context"    — 永不做 target, 固定入选 gi
                       "padding"    — 不参与任何计算
        augment:     增强函数: (ink, lens, rng) -> (ink, lens), None = 不增强
    """
    name: str
    ink: np.ndarray                      # (n_strokes, max_len, 4)
    lens: np.ndarray                     # (n_strokes,)
    start_coord: np.ndarray             # (n_strokes, 2)
    role: str                            # "targetable" | "context"
    augment: Optional[Callable] = None

    @property
    def n_strokes(self) -> int:
        return self.ink.shape[0]

    @property
    def max_len(self) -> int:
        return self.ink.shape[1]

    def apply_augment(self, rng: np.random.RandomState):
        """执行增强 (原地修改)."""
        if self.augment is not None and self.n_strokes > 0:
            self.ink, self.lens = self.augment(self.ink, self.lens, rng)

    def translate_to_origin(self):
        """每笔平移归零 (原地修改). 在记录 start_coord 之后调用."""
        self.ink[:, :, :3] -= self.ink[:, 0:1, :3]

    def z_score(self, mean_ch: np.ndarray, std_ch: np.ndarray):
        """Z-score 归一化 ink 通道 (原地修改). start_coord 不变."""
        self.ink[:, :, :4] = (self.ink[:, :, :4] - mean_ch) / std_ch

    def temporal_resample(self, factor: int):
        """时间降采样."""
        if factor <= 1 or self.n_strokes == 0:
            return
        from src.precache import pp_temporal_resampling
        self.ink, self.lens = pp_temporal_resampling(
            self.ink, self.lens, factor)


# ═══════════════════════════════════════════════════════════════════════
# 2. CompositeDiagram — 多区域草图
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CompositeDiagram:
    """一张包含图形+文字的多区域草图.

    布局约定:
      - Graphic Region 索引 [0, ng)
      - Text Region    索引 [ng, ng+nt)
      - 是 translate_to_origin / z-score 之前的格式
    """
    graphic: StrokeRegion
    text: StrokeRegion
    label: Optional[str] = None

    # ── 派生属性 ──

    @property
    def total_strokes(self) -> int:
        return self.ng + self.nt

    @property
    def ng(self) -> int:
        return self.graphic.n_strokes

    @property
    def nt(self) -> int:
        return self.text.n_strokes

    @property
    def all_ink(self) -> np.ndarray:
        """(ng+nt, max_len, 4) 拼接后的 ink."""
        if self.nt == 0:
            return self.graphic.ink
        g_len = self.graphic.max_len
        t_len = self.text.max_len
        target_len = max(g_len, t_len)
        g = self.graphic.ink
        t = self.text.ink
        if g_len < target_len:
            g = np.concatenate([g, np.zeros((self.ng, target_len - g_len, 4), dtype=np.float32)], axis=1)
        if t_len < target_len:
            t = np.concatenate([t, np.zeros((self.nt, target_len - t_len, 4), dtype=np.float32)], axis=1)
        return np.concatenate([g, t], axis=0)

    @property
    def all_lens(self) -> np.ndarray:
        if self.nt == 0:
            return self.graphic.lens
        return np.concatenate([self.graphic.lens, self.text.lens], axis=0)

    @property
    def context_mask(self) -> np.ndarray:
        """[False×ng, True×nt]."""
        return np.concatenate([
            np.zeros(self.ng, dtype=bool),
            np.ones(self.nt, dtype=bool),
        ])

    @property
    def all_start_coord(self) -> np.ndarray:
        return np.concatenate([
            self.graphic.start_coord,
            self.text.start_coord,
        ], axis=0)

    def apply_augment(self, rng: np.random.RandomState):
        """对 graphic 增强, text 微抖 (原地)."""
        self.graphic.apply_augment(rng)
        self.text.apply_augment(rng)


# ═══════════════════════════════════════════════════════════════════════
# 3. 固定排版 — TextLayoutConfig / fit_graphic_to_box / typeset_text
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TextLayoutConfig:
    """画布固定排版配置 — 训练侧 (DiagramComposer / SyntheticTextProvider) 与
    推理侧 (eval_text2sketch) 的单一事实源 (Y4/X1 修复, docs/15).

    布局 (size_normalized 画布, y 向下):

        ┌────────────────────────┐
        │ 图形 box                │  y ∈ [0, graphic_box_height]
        │ (fit-to-box, 居中适配)  │  x ∈ [0, graphic_box_width]
        ├────────────────────────┤  ← gap
        │ 文本 box                │  y ∈ [text_box_top, +text_box_height]
        │ (边框对齐, 多行可扩展)   │  行高 line_height, 最多 max_lines 行
        └────────────────────────┘

    多行扩展: 文本按词排版, 超出行宽换行; 行数上限 max_lines, 超出截断计数.
    """

    # ── 图形 box ──
    graphic_box_width: float = 1.0
    graphic_box_height: float = 1.0   # size_normalize 后图形天然 y ∈ [0,1]
    # ── 图形增强 (fit-to-box, 放弃旋转/平移/翻转/剪切) ──
    gfx_aspect_jitter: float = 0.1    # 长宽比微抖幅度 (±)
    gfx_fill_min: float = 0.8         # 填充率下限, fill ∈ [fill_min, 1.0]
    # ── 文本 box ──
    gap: float = 0.1                  # 文本框顶与图形框底的距离
    text_x_margin: float = 0.05       # 文本框内左右边距
    # ── 文本排版 ──
    # 默认值经 FS-COCO 全量 caption 扫描选定 (scripts/diag_caption_typography.py):
    # text_scale=0.09 + max_lines=5 → 94.8% caption 无截断排版;
    # 2026-08-09 用户审阅后等比缩小 50% (text_scale/letter_spacing/line_height/
    # pos_jitter 同步减半, 视觉比例不变), 空格 = 一个字符宽, 细体字母不横向拉伸.
    text_scale: float = 0.045         # 字母 canonical 高度 (归一化目标, 见 letter_aspect)
    letter_aspect: float = 0.65       # 字母 canonical 长宽比 (w/h); 实测字库 1038 样本
                                       # 原始长宽比方差极大 (0.03~2.1, median=0.662) ——
                                       # 增强前先把每个字母按此比例+text_scale 归一化,
                                       # 避免排版歪斜难辨认 (issue: 2026-08-08 用户反馈)
    min_letter_aspect: float = 0.45   # 细体字母 (i/l/j) 横向拉伸上限: 自然长宽比低于此值
                                       # (实测 i≈0.16, l≈0.22, j≈0.30) 时不再拉伸笔画,
                                       # 仅高度对齐的均匀缩放, 字形在等宽槽位内居中排版
    letter_spacing: float = 0.01      # 词内字母间距 (随字号缩放)
    word_spacing: Optional[float] = None  # 词间空格宽度; None → 一个字符宽 (随字号缩放)
    line_height: float = 0.06         # 行高 (行顶到下一行行顶, 随字号缩放)
    max_lines: int = 5                # 最大行数 (超出截断并计数)
    # ── 文本增强 (沿用原值) ──
    scale_jitter: float = 0.08
    pos_jitter: float = 0.005         # 位置抖动 (随字号等比缩小)

    @property
    def text_box_top(self) -> float:
        return self.graphic_box_height + self.gap

    @property
    def char_width(self) -> float:
        """单个字母的完整进宽 (字母槽位 + 词内间距), 即"一个字符宽"."""
        return self.text_scale * self.letter_aspect + self.letter_spacing

    @property
    def text_box_height(self) -> float:
        """容纳 max_lines 行 + 缩放/位置抖动余量."""
        jitter = self.text_scale * self.scale_jitter + self.pos_jitter
        return self.line_height * self.max_lines + 2 * jitter

    @property
    def text_box_width(self) -> float:
        return self.graphic_box_width

    @property
    def text_line_width(self) -> float:
        """单行可用宽度 (扣除左右边距)."""
        return self.graphic_box_width - 2 * self.text_x_margin


def _valid_bbox(ink: np.ndarray, lens: np.ndarray):
    """有效点 (尊重 lens) 的 bbox. 返回 (min_x, max_x, min_y, max_y)."""
    xs, ys = [], []
    for i in range(ink.shape[0]):
        L = int(lens[i])
        if L > 0:
            xs.extend(ink[i, :L, 0].tolist())
            ys.extend(ink[i, :L, 1].tolist())
    if not xs:
        return 0.0, 1.0, 0.0, 1.0
    return min(xs), max(xs), min(ys), max(ys)


def fit_graphic_to_box(
    ink: np.ndarray,
    lens: np.ndarray,
    rng: np.random.RandomState,
    layout: TextLayoutConfig,
    augment: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """图形增强 + 固定 box 居中适配 (Y4/X1 修复).

    放弃对象级仿射 (旋转/平移/翻转/剪切会破坏场景级草图语义, docs/15 §2.2);
    仅小幅度随机长宽比 + 随机填充率缩放, 居中适配进固定图形 box, 保证不越界.

    Args:
        ink:     (n, L, 4) size_normalized 图形笔画
        lens:    (n,) 有效长度 (不变)
        rng:     随机数
        layout:  TextLayoutConfig
        augment: False → 确定性适配 (fill=1.0, 无抖动), 验证集口径

    Returns:
        (ink', lens) — 图形 bbox 居中放入 [0, box_w] × [0, box_h]
    """
    out = ink.copy()
    if ink.shape[0] == 0:
        return out, lens

    min_x, max_x, min_y, max_y = _valid_bbox(ink, lens)
    bw, bh = max_x - min_x, max_y - min_y
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

    if augment:
        sx = 1.0 + rng.uniform(-layout.gfx_aspect_jitter, layout.gfx_aspect_jitter)
        sy = 1.0 + rng.uniform(-layout.gfx_aspect_jitter, layout.gfx_aspect_jitter)
        fill = rng.uniform(layout.gfx_fill_min, 1.0)
    else:
        sx = sy = 1.0
        fill = 1.0

    eps = 1e-6
    f = fill * min(layout.graphic_box_width / max(bw * sx, eps),
                   layout.graphic_box_height / max(bh * sy, eps))
    out[:, :, 0] = (ink[:, :, 0] - cx) * (sx * f) + layout.graphic_box_width / 2
    out[:, :, 1] = (ink[:, :, 1] - cy) * (sy * f) + layout.graphic_box_height / 2
    return out, lens


def typeset_text(
    label: str,
    letter_db,
    layout: TextLayoutConfig,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """在固定文本框内排版文本 (Y4/X1 修复: 边框对齐; 多词 + 自动换行, 多行可扩展).

    排版规则:
      - 每个字母先按自身 bbox 归一化到 canonical 尺寸 (text_scale 高 ×
        text_scale*letter_aspect 宽), 再叠加缩放抖动 —— 原始笔画 (取自不同单词/
        书写时机) 天然长宽比方差极大 (实测 0.03~2.1), 不归一化直接增强会导致
        排版歪斜难以辨认;
      - 细体字母 (i/l/j, 自然长宽比 < min_letter_aspect) 不横向拉伸, 仅均匀缩放
        对齐高度, 字形在等宽槽位内居中 —— 避免竖笔被拉成粗条 (issue: 2026-08-09);
      - 行顶 y = text_box_top + line_idx * line_height; 字母 bbox 顶边对齐行顶
        (废弃旧的中心线对齐), 左边缘对齐 cursor;
      - 词内字母按 first/middle/last 位置采样, 字母间距 letter_spacing;
      - 词间空格 word_spacing (None → 一个字符宽, 随字号缩放); cursor 超出行宽则换行;
      - 行数超过 max_lines 时截断 (丢弃剩余字符, 计数返回);
      - 单个词超出行宽时不拆分, 允许溢出右边界 (罕见, 由调用方统计).

    Returns:
        ink (n_strokes, max_len, 4), lens (n_strokes,), start (n_strokes, 2),
        n_truncated (被截断丢弃的字符数)
    """
    from src.letter_database import POS_FIRST, POS_MIDDLE, POS_LAST, POS_ANY

    empty = (np.zeros((0, 1, 4), dtype=np.float32),
             np.zeros(0, dtype=np.int32),
             np.zeros((0, 2), dtype=np.float32), 0)
    if label is None:
        return empty
    words = str(label).lower().split()
    if not words:
        return empty

    def _pos_hint(i, n):
        if n == 1:
            return POS_ANY
        if i == 0:
            return POS_FIRST
        if i == n - 1:
            return POS_LAST
        return POS_MIDDLE

    def _sample_letter(ch, pos):
        try:
            return letter_db.sample(ch, pos, rng)
        except KeyError:
            try:
                return letter_db.sample(ch, POS_ANY, rng)
            except KeyError:
                return None

    all_inks, all_lenses = [], []
    n_truncated = 0
    cursor_x = layout.text_x_margin
    line_idx = 0
    line_right = layout.text_x_margin + layout.text_line_width

    for word in words:
        if line_idx >= layout.max_lines:
            n_truncated += len(word)
            continue

        # ── 采样整个词 (换行判断需要词宽) ──
        letters = []
        for i, ch in enumerate(word):
            sampled = _sample_letter(ch, _pos_hint(i, len(word)))
            if sampled is None:
                continue  # 字库无此字符, 跳过 (与历史行为一致)
            ink_ch, lens_ch = sampled
            # ── 先按 bbox 归一化到 canonical 尺寸 (长宽比 + 绝对大小一致),
            #    再叠加缩放抖动 (issue: 原始笔画天然长宽比方差极大 [0.03,2.1],
            #    不先归一化直接叠加增强会导致排版歪斜、难以辨认) ──
            min_x, max_x, min_y, max_y = _valid_bbox(ink_ch, lens_ch)
            bw, bh = max_x - min_x, max_y - min_y
            eps = 1e-6
            jitter = 1.0 + rng.uniform(-layout.scale_jitter, layout.scale_jitter)
            target_h = layout.text_scale * jitter
            target_w = layout.text_scale * layout.letter_aspect * jitter
            aspect = bw / max(bh, eps)
            if aspect < layout.min_letter_aspect:
                # 细体字母 (i/l/j): 不横向拉伸 (避免竖笔被拉成粗条), 均匀缩放只对齐
                # 高度, 字形在等宽槽位 (target_w) 内居中 —— 等宽排版 (issue: 2026-08-09)
                sy = target_h / max(bh, eps)
                sx = sy
                x_off = (target_w - bw * sx) / 2.0
            else:
                sx = target_w / max(bw, eps)
                sy = target_h / max(bh, eps)
                x_off = 0.0
            letters.append((ink_ch, lens_ch, sx, sy, min_x, min_y, target_w, x_off))
        if not letters:
            continue
        word_w = sum(cw for (*_, cw, _) in letters) \
            + layout.letter_spacing * (len(letters) - 1)

        # ── 超宽换行 (不在行首才换) ──
        if cursor_x > layout.text_x_margin and cursor_x + word_w > line_right:
            line_idx += 1
            cursor_x = layout.text_x_margin
            if line_idx >= layout.max_lines:
                n_truncated += len(word)
                continue

        # ── 逐字母摆放: bbox 顶边对齐行顶, 左边缘对齐 cursor ──
        line_top = layout.text_box_top + line_idx * layout.line_height
        for ink_ch, lens_ch, sx, sy, min_x, min_y, char_w, x_off in letters:
            ink_adj = ink_ch.copy()
            ink_adj[:, :, 0] = (ink_ch[:, :, 0] - min_x) * sx + cursor_x + x_off
            ink_adj[:, :, 1] = (ink_ch[:, :, 1] - min_y) * sy + line_top
            if layout.pos_jitter > 0:
                ink_adj[:, :, 0] += rng.uniform(-layout.pos_jitter, layout.pos_jitter)
                ink_adj[:, :, 1] += rng.uniform(-layout.pos_jitter, layout.pos_jitter)
            all_inks.append(ink_adj)
            all_lenses.append(lens_ch)
            cursor_x += char_w + layout.letter_spacing
        word_gap = layout.word_spacing if layout.word_spacing is not None \
            else layout.char_width
        cursor_x += word_gap - layout.letter_spacing

    if not all_inks:
        return empty

    # Pad to uniform max_len
    max_len = max(ink.shape[1] for ink in all_inks)
    padded = []
    for ink in all_inks:
        if ink.shape[1] < max_len:
            ink = np.concatenate(
                [ink, np.zeros((ink.shape[0], max_len - ink.shape[1], 4), dtype=np.float32)], axis=1)
        padded.append(ink)

    all_inks_np = np.concatenate(padded, axis=0)
    all_lenses_np = np.concatenate(all_lenses, axis=0)
    all_start = all_inks_np[:, 0, :2].copy()

    return all_inks_np, all_lenses_np, all_start, n_truncated


# ═══════════════════════════════════════════════════════════════════════
# 4. DiagramComposer — 固定排版合成器
# ═══════════════════════════════════════════════════════════════════════

class DiagramComposer:
    """图形+文字固定排版合成器 (Y4/X1 修复后).

    排版逻辑 (固定布局, 与图形内容无关):
      1. 图形: fit_graphic_to_box — 小幅长宽比/缩放增强, 居中适配进固定图形 box;
      2. 文本: typeset_text — 固定文本框, 边框对齐, 多词/换行 (上限 max_lines).
      两个区域各自增强、互不影响; 全部布局参数来自 TextLayoutConfig (单一事实源).

    Config: 见 TextLayoutConfig (docs/15 §3).
    """

    def __init__(
        self,
        letter_db,
        layout: Optional[TextLayoutConfig] = None,
        seed: int = 42,
    ):
        self.db = letter_db
        self.layout = layout or TextLayoutConfig()
        self._rng = np.random.RandomState(seed)

    def compose(
        self,
        graphic_ink: np.ndarray,
        graphic_lens: np.ndarray,
        label: str,
        rng: np.random.RandomState,
        augment: bool = True,
    ) -> CompositeDiagram:
        """合成草图.

        Args:
            graphic_ink:  (ng, Lg, 4) — 图形笔画 (size_normalized)
            graphic_lens: (ng,) — 有效长度
            label:        文本标签, 支持多词 ("a cat on a mat")
            rng:          外部随机数 (不消耗, 排版用内部 RNG — 与历史行为一致)
            augment:      False → 图形确定性适配 (fill=1.0 无抖动), 验证集口径

        Returns:
            CompositeDiagram
        """
        g_ink, g_lens = fit_graphic_to_box(
            graphic_ink, graphic_lens, self._rng, self.layout, augment=augment)

        graphic = StrokeRegion(
            name="graphic",
            ink=g_ink,
            lens=g_lens.copy(),
            start_coord=g_ink[:, 0, :2].copy(),
            role="targetable",
        )

        t_ink, t_lens, t_start, _n_trunc = typeset_text(
            label, self.db, self.layout, self._rng)

        text = StrokeRegion(
            name="text",
            ink=t_ink,
            lens=t_lens,
            start_coord=t_start,
            role="context",
        )

        return CompositeDiagram(graphic=graphic, text=text, label=label)


# ═══════════════════════════════════════════════════════════════════════
# 5. StrokeDiagram — 模型就绪格式
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StrokeDiagram:
    """translate_to_origin + z-score 后的统一格式, 可直接转为模型 batch.

    所有笔画统一存放, context_mask 区分 targetable vs context.
    """
    ink: np.ndarray                        # (S, L, 4) — z-scored
    lens: np.ndarray                       # (S,)
    start_coord: np.ndarray                # (S, 2) — 原始空间
    end_coord: np.ndarray                 # (S, 2) — 原始空间
    context_mask: np.ndarray               # (S,) bool
    label: Optional[str] = None

    @classmethod
    def from_composite(
        cls,
        comp: CompositeDiagram,
        mean_ch: np.ndarray,
        std_ch: np.ndarray,
        affine_fn: Optional[Callable] = None,
        temporal_factor: int = 2,
    ) -> "StrokeDiagram":
        """完整管线: 增强 → translate → z-score → 统一视图.

        Args:
            comp:         CompositeDiagram (增强前)
            mean_ch/std_ch: z-score 统计量
            affine_fn:   图形仿射增强函数: (ink, lens, rng) -> (ink, lens)
            temporal_factor: 时间降采样因子 (0=不降采样)

        Steps:
          1. 对 graphic 执行仿射增强 (affine_fn)
          2. 记录 start_coord / end_coord (在 translate 之前)
          3. translate_to_origin (所有区域)
          4. z-score (仅 ink)
          5. temporal_resampling (可选)
        """
        ink = comp.all_ink
        lens = comp.all_lens
        ng = comp.ng

        # Step 1: 仿射增强 — 仅 graphic
        if affine_fn is not None and ng > 0:
            g_ink = ink[:ng]
            g_lens = lens[:ng]
            g_ink, g_lens = affine_fn(g_ink, g_lens)
            ink = np.concatenate([g_ink, ink[ng:]], axis=0) if comp.nt > 0 else g_ink
            lens = np.concatenate([g_lens, lens[ng:]], axis=0) if comp.nt > 0 else g_lens

        # Step 2: 记录 start_coord / end_coord
        start_coord = ink[:, 0, :2].copy()
        end_coord = np.zeros((ink.shape[0], 2), dtype=np.float32)
        for si in range(ink.shape[0]):
            L = int(lens[si])
            if L > 0:
                end_coord[si] = (ink[si, :L, :2] * ink[si, :L, 3:4]).sum(axis=0)

        # Step 3: 逐笔平移归零
        ink[:, :, :3] -= ink[:, 0:1, :3]

        # Step 4: z-score (仅 ink)
        ink[:, :, :4] = (ink[:, :, :4] - mean_ch) / std_ch

        # Step 5: temporal_resampling (可选)
        if temporal_factor > 1:
            from src.precache import pp_temporal_resampling
            ink, lens = pp_temporal_resampling(ink, lens, temporal_factor)

        return cls(
            ink=ink,
            lens=lens,
            start_coord=start_coord,
            end_coord=end_coord,
            context_mask=comp.context_mask,
            label=comp.label,
        )

    def to_model_batch(self, n_t_samples: int, mask_pen: bool,
                       rng: np.random.RandomState):
        """转为 (inputs, targets) dict. 适配现有 to_model_batch_single."""
        from src.precache import to_model_batch_single, pp_get_t_targets

        t_values, t_targets = pp_get_t_targets(self.ink, self.lens, n_t_samples, rng)
        return to_model_batch_single(
            self.ink, self.lens,
            self.start_coord, self.end_coord,
            t_values, t_targets, mask_pen,
            dct_coeffs=None,
            context_mask=self.context_mask,
        )


# ═══════════════════════════════════════════════════════════════════════
# 图形增强（自 src/pipeline_a2.py 迁入，2026-09-30 项目清理）
# ═══════════════════════════════════════════════════════════════════════

def a2_graphic_augment(ink, lens, rng, layout, affine_prob, augment=True):
    """图形增强 (A2): 绕 bbox 中心旋转 + 水平翻转 + fit-to-box 居中.

    采样幅度复刻旧管线 pp_random_affine_all, 但不动点从坐标原点改为 bbox 中心:
      - 旋转:    θ ∈ [-π/2, π/2], prob=affine_prob (旧管线绕原点);
      - 水平翻转: scale_x *= -1,   prob=affine_prob (旧管线绕原点);
      - 垂直翻转: scale_y *= -1,   prob=affine_prob (旧管线绕原点).
    随后交回新版 fit_graphic_to_box 做长宽比/填充抖动 + 居中 (图形 bbox 恒居中).
    无平移分量 (中心恒在固定框中心), 满足"取消随机平移".
    """
    out = ink.copy()
    if out.shape[0] == 0:
        return out, lens

    if augment:
        min_x, max_x, min_y, max_y = _valid_bbox(ink, lens)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2

        rot_angle = rng.uniform(-np.pi / 2, np.pi / 2)
        if rng.random() > affine_prob:
            rot_angle = 0.0
        flip_x = rng.random() < affine_prob
        flip_y = rng.random() < affine_prob

        if rot_angle != 0.0 or flip_x or flip_y:
            xy = out[:, :, :2].astype(np.float64)
            xy -= np.array([cx, cy], dtype=np.float64)
            if flip_x:
                xy[:, :, 0] *= -1.0
            if flip_y:
                xy[:, :, 1] *= -1.0
            if rot_angle != 0.0:
                c = np.cos(rot_angle)
                s = np.sin(rot_angle)
                rot = np.array([[c, -s], [s, c]], dtype=np.float64)
                xy = xy @ rot.T
            out[:, :, :2] = xy + np.array([cx, cy], dtype=np.float64)

    return fit_graphic_to_box(out, lens, rng, layout, augment=augment)


