"""时空切片数据管线（src/st/pipeline_st.py）。

从零重构 —— 契约见 docs/22 §6。与旧管线（src/data_pipeline.py）并存，旧管线不动。

本期范围（G2/G3 冒烟）：QD 9 类**单片纯图形**样本（CoSE 基线的等价形态）。
A1'/B1' 的多片构造（文字片/指针片/标签片）随各自 milestone 加入。

关键约定：
- precache 存 RAW（inks: (N,) object，每条 (S,L,4) = [x,y,t,pen] 绝对坐标 + 时间戳；
  lenses: (N,) object 每条 (S,)）。统计量 mean_ch/std_ch (4,)，z-score 只用 x,y 通道。
- 在线处理：逐笔平移到起点（encoder 输入相对坐标）→ xy z-score → Δt 构造
  （切片任务：0 + 抖动；`use_real_dt` 时用真实时间戳，每笔画 Δt 相对笔首）→
  τ 采样与插值目标（相对坐标）→ 端点目标。
- start_coord 三维 (x, y, Δt₀) 永不 z-score。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config_st import STConfig
from .sampling import riffle_shuffle, first_stroke_weights


def _random_affine(pts: np.ndarray, prob: float,
                   rng: np.random.Generator) -> np.ndarray:
    """整图统一仿射增强（严格移植旧管线 pp_random_affine_all + _make_affine_matrix）。

    pts (S,L,2) 绝对坐标；线性变换（无平移分量）故 padding 零点不受影响。
    顺序：旋转 → 缩放 → 翻转 → 剪切，每步以 prob（剪切 prob/3）概率激活。
    """
    return np.einsum("ij,slj->sli",
                     _sample_affine_matrix(prob, rng), pts).astype(np.float32)


def _sample_affine_matrix(prob: float, rng: np.random.Generator) -> np.ndarray:
    """采样一个 (2,2) 仿射矩阵（供跨片联合施加，保持片间相对布局）。"""
    rot = rng.uniform(-np.pi / 2, np.pi / 2) if rng.random() < prob else 0.0
    scale = rng.uniform(0.5, 2.5) if rng.random() < prob else 1.0
    sx = -scale if rng.random() < prob else scale
    sy = -scale if rng.random() < prob else scale
    shear = rng.uniform(-0.3, 0.3) if rng.random() < prob / 3.0 else 0.0
    cos_t, sin_t = np.cos(rot), np.sin(rot)
    shear_mat = np.array([[1.0, shear], [shear, 1.0]], dtype=np.float32)
    rot_scale = np.array([[sx * cos_t, -sy * sin_t],
                          [sx * sin_t, sy * cos_t]], dtype=np.float32)
    return shear_mat @ rot_scale


def _temporal_stride(rel: np.ndarray, lens: np.ndarray, factor: int):
    """长笔 stride 抽稀（对齐旧 pp_temporal_resampling：>100 点笔按 factor 抽，
    其余不动）。作用于平移后的 (S,L,C) 相对坐标；padding 由 lens 掩码，无需保零。
    """
    S, L, C = rel.shape
    out = np.zeros_like(rel)
    new_lens = lens.copy()
    for i in range(S):
        L_i = int(lens[i])
        if L_i > 100:
            pts = rel[i, :L_i:factor]
            out[i, :len(pts)] = pts
            new_lens[i] = len(pts)
        else:
            out[i, :L_i] = rel[i, :L_i]
    return out, new_lens


def _terminator_stroke(cfg: STConfig, mean_ch: np.ndarray, std_ch: np.ndarray,
                       rng: np.random.Generator):
    """规范 e 形终止笔画（D25；docs/22 §11）：315° 内旋小环，首点 [0,0]（相对坐标）。

    返回 rel3 (L,3)（xy 与真实笔画同一 z-score；Δt=0+抖动——e 是瞬时符号笔画，
    use_real_dt 模式下也保持 0）与 start (3,)（raw 空间固定 [4,4,0]，不 z-score）。
    """
    n = cfg.term_points
    r = cfg.term_radius
    th = np.linspace(np.pi, np.pi + 1.75 * np.pi, n, dtype=np.float32)
    rr = r * (1.0 - 0.3 * (th - th[0]) / (th[-1] - th[0]))       # 轻微内旋收尾
    xy = np.stack([rr * np.cos(th) + r, rr * np.sin(th)], -1)   # θ=π 时首点 [0,0]
    xy = (xy - mean_ch[:2]) / std_ch[:2]
    if cfg.coord_dim == 3:
        if cfg.dt_jitter > 0:
            dt = rng.normal(0, cfg.dt_jitter, (n, 1)).astype(np.float32)
        else:
            dt = np.zeros((n, 1), dtype=np.float32)
        rel = np.concatenate([xy, dt], -1)
        start = np.array([cfg.term_pos[0], cfg.term_pos[1], 0.0], dtype=np.float32)
    else:
        rel = xy
        start = np.array([cfg.term_pos[0], cfg.term_pos[1]], dtype=np.float32)
    return rel, start


class PrecacheSlices(Dataset):
    """QD 预缓存 → 单切片图形样本（冒烟用）。"""

    def __init__(self, cache_dir: str, split: str, cfg: STConfig,
                 max_diagrams: Optional[int] = None, seed: int = 42):
        d = np.load(Path(cache_dir) / f"precached_{split}.npz", allow_pickle=True)
        self.inks = d["inks"]
        self.lenses = d["lenses"]
        self.labels = d["labels"] if "labels" in d else None
        self.mean_ch = d["mean_ch"].astype(np.float32)   # (4,) [x,y,t,pen]
        self.std_ch = d["std_ch"].astype(np.float32)
        if max_diagrams is not None:
            self.inks = self.inks[:max_diagrams]
            self.lenses = self.lenses[:max_diagrams]
            if self.labels is not None:
                self.labels = self.labels[:max_diagrams]
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.inks)

    # ── 共享处理件（单片/两片数据集共用）──

    def _process_group(self, ink: np.ndarray, lens: np.ndarray, affine: bool,
                       affine_mat: Optional[np.ndarray] = None):
        """一组笔画的共享处理：超长截断 → 仿射（可选）→ 逐笔平移 → 长笔抽稀 →
        xy z-score →（coord_dim=3 时）Δt 构造。

        ink (S,L,C≥2) 绝对坐标（含 t 通道时 use_real_dt 可用），lens (S,)。
        affine_mat 给定时直接施加该矩阵（多片联合仿射，保持片间相对布局）；
        否则 affine=True 时自行采样。
        返回 rel (S,L,coord_dim)、start_coord (S,coord_dim)、lens。
        """
        cfg = self.cfg
        if ink.shape[1] > cfg.max_stroke_len:      # encoder PE 表长上限
            ink = ink[:, :cfg.max_stroke_len]
            lens = np.minimum(lens, cfg.max_stroke_len)
        L = ink.shape[1]
        pts = ink[:, :, :2].astype(np.float32)
        if affine_mat is not None:
            pts = np.einsum("ij,slj->sli", affine_mat, pts).astype(np.float32)
        elif affine and cfg.affine_prob > 0:           # 在线仿射（对齐旧管线）
            pts = _random_affine(pts, cfg.affine_prob, self.rng)
        start_xy = pts[:, 0, :]
        if cfg.unified_emb:
            # 计划1 统一嵌入（总纲：画布排版上显示的形状位置即编码器输入）。
            # 保留绝对显示坐标，不再平移；归一化用固定画布重映射：
            # 图形框 [0,1]² + 下方文本框（y≤~1.2）→ 以零为中心的画布（等比保长宽比）
            assert cfg.coord_dim == 2, "unified_emb 仅支持 coord_dim=2"
            work = pts.copy()
        else:
            work = pts - start_xy[:, None, :]               # 平移到起点

        # 长笔 stride 抽稀（对齐旧 pp_temporal_resampling：>100 点按 factor 抽）
        if cfg.temporal_resampling_factor > 1:
            work, lens = _temporal_stride(work, lens, cfg.temporal_resampling_factor)

        if cfg.unified_emb:
            work = (work - np.array([0.5, 0.6], np.float32)) / 0.6
        else:
            work = (work - self.mean_ch[:2]) / self.std_ch[:2]   # z-score 仅 xy

        # DCT 目标：直接在最终显示坐标上计算（= 编码器输入空间）。
        # k=1..K 纯形状（去均值，平移不变）；cfg.dct_dc 时末尾追加质心 (x̄,ȳ)
        # ——直流分量等价量，统一画布下即笔画位置，显式位置+形状解耦监督（B 臂）
        dct_target = None
        if cfg.dct_coeffs > 0:
            from src.st.dct import compute_dct_for_strokes
            dct_target = compute_dct_for_strokes(work, lens, 128, cfg.dct_coeffs,
                                                 include_mean=cfg.dct_dc)

        rel = work

        if cfg.coord_dim == 3:
            if getattr(cfg, "use_real_dt", False) and ink.shape[2] >= 3:
                dt = ink[:, :L, 2] - ink[:, 2:3, 2]    # 相对笔首
            else:
                dt = np.zeros((ink.shape[0], L), dtype=np.float32)
                if cfg.dt_jitter > 0:
                    dt = dt + self.rng.normal(0, cfg.dt_jitter, dt.shape).astype(np.float32)
            rel = np.concatenate([rel, dt[:, :, None]], axis=-1)
            start_dt = np.zeros((ink.shape[0], 1), dtype=np.float32)
            start_coord = np.concatenate([start_xy, start_dt], -1)   # 不 z-score
        else:
            start_coord = start_xy                                   # 2D，不 z-score
        return (rel.astype(np.float32), start_coord.astype(np.float32), lens,
                dct_target)

    def _terminator_row(self, L: int):
        """e 终止笔画行（pad 到 L 点）。"""
        e_rel, e_start = _terminator_stroke(self.cfg, self.mean_ch, self.std_ch, self.rng)
        e_pad = np.zeros((L, self.cfg.coord_dim), dtype=np.float32)
        e_pad[:self.cfg.term_points] = e_rel
        return e_pad, e_start

    def _finalize(self, rel3, start_coord, lens, slice_id, targetable,
                  dct_target=None):
        """τ 采样与插值目标 + 端点目标 + 首笔加权 → batch dict。
        dct_target (S,2K)：DCT 纯形状辅助目标（计划1，None=不启用）。"""
        cfg = self.cfg
        S, L = rel3.shape[0], rel3.shape[1]
        n_t = cfg.n_t_samples
        tau = self.rng.random((S, n_t)).astype(np.float32)
        pos = tau * np.maximum(lens - 1, 1).astype(np.float32)[:, None]
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, L - 1)
        frac = (pos - lo).astype(np.float32)[:, :, None]
        lo_pts = np.take_along_axis(rel3, lo[:, :, None], axis=1)
        hi_pts = np.take_along_axis(rel3, hi[:, :, None], axis=1)
        tau_target = lo_pts + frac * (hi_pts - lo_pts)

        end_idx = np.maximum(lens - 1, 0)
        end_target = np.stack([
            rel3[:, 0], rel3[np.arange(S), end_idx]], axis=1)

        loss_weight = first_stroke_weights(slice_id, cfg.w_first)
        out = {
            "ink": rel3, "stroke_len": lens, "start_coord": start_coord,
            "slice_id": slice_id, "targetable": targetable,
            "tau": tau, "tau_target": tau_target, "end_target": end_target,
            "loss_weight": loss_weight,
        }
        if dct_target is not None:
            out["dct_target"] = dct_target
        return out

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        ink = self.inks[i].astype(np.float32)            # (S,L,4) 绝对坐标+时间戳
        lens = self.lenses[i].astype(np.int64)
        rel3, start_coord, lens, dct_t = self._process_group(ink, lens, affine=True)

        # ── 单片：slice_id 全 0；riffle 洗牌（图形片，按生成序洗牌增广）──
        S0 = rel3.shape[0]
        slice_id = np.zeros(S0, dtype=np.int64)
        order = riffle_shuffle(np.arange(S0), cfg.p_shuffle_graphic, self.rng)
        rel3, lens = rel3[order], lens[order]
        start_coord, slice_id = start_coord[order], slice_id[order]
        if dct_t is not None:
            dct_t = dct_t[order]

        # ── 终止符 e（D25）：洗牌后追加在片尾，永远最后 ──
        # 语义：e 的预测上下文恒为完整笔画集 ⇒ R_pos 输出近 [4,4] 即"画完"信号
        if cfg.use_terminator:
            e_pad, e_start = self._terminator_row(rel3.shape[1])
            rel3 = np.concatenate([rel3, e_pad[None]], axis=0)
            start_coord = np.concatenate([start_coord, e_start[None]], axis=0)
            lens = np.concatenate([lens, [cfg.term_points]])
            slice_id = np.concatenate([slice_id, slice_id[-1:]])
            if dct_t is not None:
                dct_t = np.concatenate(
                    [dct_t, np.zeros((1, dct_t.shape[-1]), np.float32)], axis=0)

        # targetable：图形笔与 e 可做预测目标；文字片=0（纯上下文）
        targetable = np.ones(rel3.shape[0], dtype=np.float32)
        return self._finalize(rel3, start_coord, lens, slice_id, targetable, dct_t)


class WordGraphicSlices(PrecacheSlices):
    """A1' 两片样本：文字片（类别词，纯上下文）+ 图形片（target）。

    布局对齐旧 A1（composite.TextLayoutConfig 默认）：图形 y∈[0,1]，文字框在
    图形下方（y≈[1.1,1.16]）。**文字与图形共享同一个随机仿射矩阵**（保持词-画相对
    布局，对齐旧 A1 的整图联合仿射；2026-08-22 修复）；文字片不洗牌（字母序即生成序，
    p_shuffle_text=0），字形抖动由 typeset 自带（scale/pos jitter）；图形片 riffle。
    每片末尾各一个 e；文字片与 text-e 全部 targetable=0。
    序列：[text..., e_t, graphic..., e_g]，slice_id 0/0/1/1。
    """

    def __init__(self, cache_dir: str, split: str, cfg: STConfig,
                 letter_db_path: str, max_diagrams: Optional[int] = None, seed: int = 42):
        super().__init__(cache_dir, split, cfg, max_diagrams=max_diagrams, seed=seed)
        from src.letter_database import LetterDatabase
        from src.composite import TextLayoutConfig
        self.letter_db = LetterDatabase(letter_db_path)
        # A2-v3 排版（2026-08-13 用户审定）：字体放大版 text_scale=0.09，
        # 随字号缩放的间距/行高/位置抖动同步翻倍（还原 2026-08-09 缩小前的值）
        self.layout = TextLayoutConfig(text_scale=0.09, letter_spacing=0.02,
                                       line_height=0.12, pos_jitter=0.01)
        # typeset_text / letter_db 走旧 RandomState API（SyntheticTextProvider 验证过的组合）
        self.text_rng = np.random.RandomState(seed)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        cfg = self.cfg
        from src.composite import typeset_text

        # ── 图形片：A2-v3 版增强（2026-08-22 对齐，废除此前的全仿射错误理解）──
        # bbox 中心旋转/水平/垂直翻转（p=affine_prob，采样幅度复刻旧管线）
        # + fit-to-box 居中（长宽比/填充率抖动，图形恒居中于固定 [0,1]² 框，
        # 不满天飞）。文本框固定不动。
        from src.composite import a2_graphic_augment
        ink = self.inks[i].astype(np.float32)
        lens = self.lenses[i].astype(np.int64)
        ink_aug, lens_aug = a2_graphic_augment(
            ink, lens, self.rng, self.layout,
            affine_prob=cfg.affine_prob, augment=True)
        g_rel3, g_start, g_lens, g_dct = self._process_group(ink_aug, lens_aug, affine=False)
        if cfg.first_stroke_prior and g_rel3.shape[0] > 1:
            # 首笔先验（2026-09-25 用户裁定）：50% 弧长最长笔 / 50% 真实首笔
            # 放图形序列第一位，第一位豁免 riffle。弧长在画布映射坐标上算
            # （映射等比，不改变长短排序）。
            if self.rng.random() < 0.5:
                arc = np.array([np.hypot(
                    *np.diff(g_rel3[s, :g_lens[s], :2], axis=0).T).sum()
                    if g_lens[s] > 1 else 0.0
                    for s in range(g_rel3.shape[0])])
                first = int(np.argmax(arc))
            else:
                first = 0
            rest = np.array([j for j in range(g_rel3.shape[0]) if j != first])
            rest = riffle_shuffle(rest, cfg.p_shuffle_graphic, self.rng)
            order = np.concatenate([[first], rest])
        else:
            order = riffle_shuffle(np.arange(g_rel3.shape[0]),
                                   cfg.p_shuffle_graphic, self.rng)
        g_rel3, g_start, g_lens = g_rel3[order], g_start[order], g_lens[order]
        if g_dct is not None:
            g_dct = g_dct[order]

        # ── 文字片（固定文本框 + typeset 抖动，不做任何仿射）──
        label = str(self.labels[i]) if self.labels is not None else ""
        t_ink, t_lens, _, _ = typeset_text(label, self.letter_db, self.layout, self.text_rng)
        if t_ink.shape[0] == 0:                    # 字库缺字符等异常 → 退化为单片图形
            return super().__getitem__(i)
        t_rel3, t_start, t_lens, t_dct = self._process_group(
            t_ink.astype(np.float32), t_lens.astype(np.int64), affine=False)

        # ── 统一 L 维并装配 [text, e_t, graphic, e_g] ──
        L = max(g_rel3.shape[1], t_rel3.shape[1], cfg.term_points)

        def pad_L(a):
            out = np.zeros((a.shape[0], L, a.shape[-1]), dtype=np.float32)
            out[:, :a.shape[1]] = a
            return out

        # r_pe=none（A2 复现模式）时退化为全局单片：slice_id 全 0
        g_sid, t_sid = (0, 0) if cfg.r_pe_mode == "none" else (1, 0)

        def dct_zeros():
            w = cfg.dct_coeffs * 2 + (2 if cfg.dct_dc else 0)
            return (np.zeros((1, w), np.float32)
                    if cfg.dct_coeffs > 0 else None)

        parts = []
        parts.append((pad_L(t_rel3), t_start, t_lens, t_sid, 0.0, t_dct))
        if cfg.use_terminator:
            e_pad, e_start = self._terminator_row(L)
            parts.append((e_pad[None], e_start[None],
                          np.array([cfg.term_points]), t_sid, 0.0, dct_zeros()))
        parts.append((pad_L(g_rel3), g_start, g_lens, g_sid, 1.0, g_dct))
        if cfg.use_terminator:
            e_pad, e_start = self._terminator_row(L)
            parts.append((e_pad[None], e_start[None],
                          np.array([cfg.term_points]), g_sid, 1.0, dct_zeros()))

        rel3 = np.concatenate([p[0] for p in parts], axis=0)
        start_coord = np.concatenate([p[1] for p in parts], axis=0)
        lens = np.concatenate([p[2] for p in parts], axis=0)
        slice_id = np.concatenate([np.full(len(p[2]), p[3], np.int64) for p in parts])
        targetable = np.concatenate([np.full(len(p[2]), p[4], np.float32) for p in parts])
        dct_t = (np.concatenate([p[5] for p in parts], axis=0)
                 if cfg.dct_coeffs > 0 else None)
        return self._finalize(rel3, start_coord, lens, slice_id, targetable, dct_t)


class BalancedClassBatchSampler:
    """类别均衡无限流 batch 采样器（替代旧 round-robin 的队尾余量债）。

    每类一条独立洗牌队列，耗尽即重洗续上（类内无放回循环）；每轮 n_classes 个
    槽位、轮内类序重洗；按 batch_size 连续切批。任意 ≥n_classes 的连续窗口内
    各类曝光 ±1——没有"余量"，因为队列永不同时耗尽（2026-08-21 用户设计）。
    """

    def __init__(self, labels, batch_size: int, seed: int = 42):
        labels = np.asarray(labels)
        self.classes = np.unique(labels)
        self.pools = {c: np.flatnonzero(labels == c) for c in self.classes}
        self._state = {c: (np.empty(0, dtype=np.int64), 0) for c in self.classes}
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.n = len(labels)

    def __len__(self):
        return self.n // self.batch_size          # 名义长度（实际为无限流）

    def _draw(self, c) -> int:
        q, p = self._state[c]
        if p >= len(q):
            q = self.pools[c].copy()
            self.rng.shuffle(q)
            p = 0
        self._state[c] = (q, p + 1)
        return int(q[p])

    def __iter__(self):
        buf = []
        while True:
            order = self.classes.copy()
            self.rng.shuffle(order)
            for c in order:
                buf.append(self._draw(c))
            while len(buf) >= self.batch_size:
                yield buf[:self.batch_size]
                buf = buf[self.batch_size:]


def collate_st(samples) -> Dict[str, torch.Tensor]:
    """变长 padding 成 batch。padding 笔画：ink=0、len=0、pad=True。"""
    B = len(samples)
    S_max = max(s["ink"].shape[0] for s in samples)
    L_max = max(s["ink"].shape[1] for s in samples)
    F = samples[0]["ink"].shape[-1]
    n_t = samples[0]["tau"].shape[1]

    ink = np.zeros((B, S_max, L_max, F), dtype=np.float32)
    lens = np.zeros((B, S_max), dtype=np.int64)
    start = np.zeros((B, S_max, samples[0]["start_coord"].shape[-1]), dtype=np.float32)
    slice_id = np.zeros((B, S_max), dtype=np.int64)
    targetable = np.zeros((B, S_max), dtype=np.float32)
    pad = np.ones((B, S_max), dtype=bool)
    tau = np.zeros((B, S_max, n_t), dtype=np.float32)
    tau_target = np.zeros((B, S_max, n_t, F), dtype=np.float32)
    end_target = np.zeros((B, S_max, 2, F), dtype=np.float32)
    loss_weight = np.ones((B, S_max), dtype=np.float32)
    has_dct = "dct_target" in samples[0]
    dct_arr = None
    if has_dct:
        dk = samples[0]["dct_target"].shape[-1]
        dct_arr = np.zeros((B, S_max, dk), dtype=np.float32)

    for b, s in enumerate(samples):
        n = s["ink"].shape[0]
        L = s["ink"].shape[1]
        ink[b, :n, :L] = s["ink"]
        lens[b, :n] = s["stroke_len"]
        start[b, :n] = s["start_coord"]
        slice_id[b, :n] = s["slice_id"]
        targetable[b, :n] = s["targetable"]
        pad[b, :n] = False
        tau[b, :n] = s["tau"]
        tau_target[b, :n, :, :F] = s["tau_target"]
        end_target[b, :n] = s["end_target"]
        loss_weight[b, :n] = s["loss_weight"]
        if has_dct:
            dct_arr[b, :n] = s["dct_target"]

    out = {
        "ink": torch.from_numpy(ink),
        "stroke_len": torch.from_numpy(lens),
        "start_coord": torch.from_numpy(start),
        "slice_id": torch.from_numpy(slice_id),
        "targetable": torch.from_numpy(targetable),
        "stroke_pad": torch.from_numpy(pad),
        "tau": torch.from_numpy(tau),
        "tau_target": torch.from_numpy(tau_target),
        "end_target": torch.from_numpy(end_target),
        "loss_weight": torch.from_numpy(loss_weight),
    }
    if has_dct:
        out["dct_target"] = torch.from_numpy(dct_arr)
    return out


def _st_worker_init(worker_id: int):
    """num_workers>0 时按 worker 重撒数据集 RNG（否则 fork/spawn 副本同种子，
    各 worker 的仿射/洗牌/排版抖动序列相关——标准增强相关坑）。"""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds.rng = np.random.default_rng(42 + 1000 * worker_id)
    if hasattr(ds, "text_rng"):
        ds.text_rng = np.random.RandomState(42 + 1000 * worker_id)


def build_st_loader(cache_dir: str, split: str, cfg: STConfig, batch_size: int,
                    max_diagrams: Optional[int] = None, shuffle: bool = True,
                    task: str = "graphic", letter_db_path: Optional[str] = None,
                    class_balanced: Optional[bool] = None,
                    num_workers: int = 0) -> DataLoader:
    """task: "graphic" 单片基线 / "word_graphic" A1' 两片（需 letter_db_path）。
    class_balanced 默认取 cfg.class_balanced；启用且有标签时用 BalancedClassBatchSampler
    （无限流均衡，无队尾余量债）。"""
    if task == "word_graphic":
        assert letter_db_path is not None, "word_graphic 需要 letter_db_path"
        ds = WordGraphicSlices(cache_dir, split, cfg, letter_db_path,
                               max_diagrams=max_diagrams)
    else:
        ds = PrecacheSlices(cache_dir, split, cfg, max_diagrams=max_diagrams)
    cb = getattr(cfg, "class_balanced", True) if class_balanced is None else class_balanced
    if cb and ds.labels is not None and len(np.unique(ds.labels)) > 1:
        sampler = BalancedClassBatchSampler(ds.labels, batch_size)
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_st,
                          num_workers=num_workers, worker_init_fn=_st_worker_init)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_st, drop_last=True,
                      num_workers=num_workers, worker_init_fn=_st_worker_init)
