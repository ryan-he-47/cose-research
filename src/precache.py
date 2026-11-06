"""
预缓存构建（Phase 1 离线，src/precache.py）。

自 src/data_pipeline.py 抽取的活跃子集（2026-09-30 项目清理）——
原文件的在线处理（pp_*/batch/loader）随旧栈归档 _archive/。

Phase 1: NDJSON → size_normalize → 20ms 重采样 → 存 RAW（增强只在线上做）。
不变量：start_coord 永不 z-score；precache 只存 RAW。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

@dataclass
class DataConfig:
    """数据管线配置，严格匹配 TF 默认值."""
    # 过滤
    min_stroke_len: int = 4
    max_stroke_len: int = 200
    min_strokes_per_diagram: int = 4

    # 重采样
    resample_timestep_ms: float = 20.0
    temporal_resampling_factor: int = 2  # 0=不降采样, 2=stride-2

    # 增强
    affine_prob: float = 0.3

    # 解码器 t 采样
    n_t_samples: int = 4

    # GT targets
    gt_targets: bool = True

    # Pen
    mask_encoder_pen: bool = True

    # DCT 辅助解码器 (0 = 禁用)
    dct_n_coeffs: int = 0            # 每通道保留的非直流 DCT 系数个数
    dct_n_resampled: int = 128       # 等弧长重采样点数

    # 文本合成钩子 (Phase 2: 多模态实验)
    text_provider: Optional["SyntheticTextProvider"] = None
    # 新版合成器 (重构后, 优先使用)
    composer: Optional["DiagramComposer"] = None

    # 类别均衡采样 (Phase 2: 多类别 QuickDraw)
    class_balanced: bool = False


def size_normalization(drawing: List) -> List:
    """尺寸归一化：除以高度，保持宽高比。

    严格匹配 TF didi_json_to_tfrecords.py::size_normalization
    """
    # 计算全局 bounding box
    minx = min(min(s[0]) for s in drawing)
    miny = min(min(s[1]) for s in drawing)
    maxx = max(max(s[0]) for s in drawing)
    maxy = max(max(s[1]) for s in drawing)

    height = maxy - miny
    if height < 1e-6:
        height = 1.0

    return [
        [
            [(x - minx) / height for x in s[0]],
            [(y - miny) / height for y in s[1]],
            list(s[2]),  # 时间戳不变
        ]
        for s in drawing
    ]


def resample_stroke(stroke, timestep: float = 20.0):
    """时间维度线性插值重采样。

    严格匹配 TF didi_json_to_tfrecords.py::resample_stroke
    关键: 使用 while current_time < final_time (不是 for range)
    """
    x, y, t = stroke
    if len(t) < 3:
        return stroke

    def _interpolate(t_val, t_prev, t_next, v0, v1):
        d0 = abs(t_val - t_prev)
        d1 = abs(t_val - t_next)
        dist_sum = d0 + d1
        d0 /= dist_sum
        d1 /= dist_sum
        return d1 * v0 + d0 * v1

    r_x, r_y, r_t = [x[0]], [y[0]], [t[0]]
    final_time = t[-1]
    current_time = t[0] + timestep
    i = 1

    while current_time < final_time:
        while i < len(t) - 1 and current_time > t[i]:
            i += 1
        r_x.append(_interpolate(current_time, t[i-1], t[i], x[i-1], x[i]))
        r_y.append(_interpolate(current_time, t[i-1], t[i], y[i-1], y[i]))
        r_t.append(current_time)
        current_time += timestep

    return [r_x, r_y, r_t]


def resample_stroke_to_cap(stroke, cap: int):
    """超长笔画等弧长重采样到 cap 点（保形状、均匀化时间密度；docs/22 §17.1）。

    stroke: [x_list, y_list, t_list]（resample_stroke 之后、diagram_to_strokes 之前）。
    短于 cap 原样返回。
    """
    x, y, t = stroke
    n = len(x)
    if n <= cap:
        return stroke
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    ta = np.asarray(t, dtype=np.float64)
    ds = np.sqrt(np.diff(xa) ** 2 + np.diff(ya) ** 2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] < 1e-12:                      # 全重合退化：均匀取点
        idx = np.linspace(0, n - 1, cap).astype(int)
        return [[float(xa[i]) for i in idx], [float(ya[i]) for i in idx],
                [float(ta[i]) for i in idx]]
    su = np.linspace(0.0, s[-1], cap)
    return [np.interp(su, s, xa).tolist(),
            np.interp(su, s, ya).tolist(),
            np.interp(su, s, ta).tolist()]


def diagram_to_strokes(drawing: List) -> Tuple[np.ndarray, np.ndarray]:
    """将一张图的笔画列表转为 numpy 数组。

    Returns:
        ink: (num_strokes, max_len, 4) — [x, y, t, pen]
        stroke_lens: (num_strokes,) — 每条笔画的有效长度
    """
    max_len = max(len(s[0]) for s in drawing)
    n_strokes = len(drawing)

    ink = np.zeros((n_strokes, max_len, 4), dtype=np.float32)
    stroke_lens = np.zeros(n_strokes, dtype=np.int32)

    for i, s in enumerate(drawing):
        L = len(s[0])
        ink[i, :L, 0] = s[0]
        ink[i, :L, 1] = s[1]
        ink[i, :L, 2] = s[2]
        ink[i, L - 1, 3] = 1.0  # pen-up at last point
        stroke_lens[i] = L

    return ink, stroke_lens


def compute_statistics(diagrams: List) -> Dict:
    """计算逐通道均值和方差（仅训练集）。

    输入: tuples of (ink_array, lens_array) 或直接 ink arrays.
    Pen 通道固定 mean=0, var=1。
    """
    n_channel = 0.0
    mean_channel = np.zeros(4, dtype=np.float64)
    m2_channel = np.zeros(4, dtype=np.float64)

    for item in tqdm(diagrams, desc="Computing statistics"):
        # 兼容 (ink, lens) tuple 和裸 ink array
        if isinstance(item, tuple):
            ink = item[0]
            lens = item[1]
        else:
            ink = item
            lens = None

        for i in range(ink.shape[0]):
            L = int(lens[i]) if lens is not None else ink.shape[1]
            if L <= 0:
                continue
            sample = ink[i, :L, :].astype(np.float64)

            seq_len = sample.shape[0]
            n_channel += seq_len
            delta = sample - mean_channel
            mean_channel += delta.sum(axis=0) / n_channel
            m2_channel += (delta * (sample - mean_channel)).sum(axis=0)

    var_channel = m2_channel / (n_channel - 1) if n_channel > 1 else np.ones(4)

    # Pen 通道不归一化
    mean_channel[3] = 0.0
    var_channel[3] = 1.0

    return {
        "mean_channel": mean_channel.astype(np.float32),
        "var_channel": var_channel.astype(np.float32),
        "std_channel": np.sqrt(var_channel).astype(np.float32),
        "n_samples": int(n_channel),
    }


def build_quickdraw_precache(
    ndjson_paths: List[str],
    output_dir: str,
    category_names: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
    config: Optional[DataConfig] = None,
    resample_cap: Optional[int] = None,
) -> Dict:
    """多类别 QuickDraw NDJSON → 统一预缓存 (.npz)

    与 DiDi 的区别:
      - 支持多个 .ndjson 文件 (每类别一个)
      - 没有预定义 split，按比例随机划分 train/valid/test
      - 存储类别标签 (labels) 用于后续多模态任务

    Args:
        ndjson_paths: 每类的 .ndjson 文件路径列表
        output_dir: 输出目录
        category_names: 类别名 (默认从文件名推断, e.g. "full_raw_cat" → "cat")
        train_ratio / valid_ratio: 划分比例 (test = 1 - train - valid)
        seed: 随机种子
        config: 管线配置
        resample_cap: 超长笔画等弧长重采样上限（None=不启用；docs/22 §17.1，
            保形状丢时间密度，替代在线截断丢尾部）

    Returns:
        stats: z-score 统计量
    """
    if config is None:
        config = DataConfig()
    if category_names is None:
        category_names = [_infer_category(p) for p in ndjson_paths]

    os.makedirs(output_dir, exist_ok=True)

    # 收集所有 diagram (ink, lens, label)
    all_data = []
    total_drawings = 0

    for path, cat_name in zip(ndjson_paths, category_names):
        print(f"Reading: {path}  ({cat_name})")
        with open(path, 'r', encoding='utf-8') as f:
            n_lines = sum(1 for _ in f)
        print(f"  {n_lines:,} drawings")

        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=n_lines, desc=f"  Processing {cat_name}"):
                sample = json.loads(line)
                drawing = sample["drawing"]

                # 只保留 recognized=True 的
                if not sample.get("recognized", True):
                    continue

                # 预处理管线 (与 DiDi 相同)
                drawing = size_normalization(drawing)
                drawing = [resample_stroke(s, config.resample_timestep_ms)
                           for s in drawing]
                # 超长笔画重采样回 cap 内（保形状；docs/22 §17.1。cap<200 时
                # 下方 max_stroke_len 截断自然失活）
                if resample_cap is not None:
                    drawing = [resample_stroke_to_cap(s, resample_cap)
                               for s in drawing]
                ink, lens = diagram_to_strokes(drawing)

                # 过滤：仅丢弃过短笔画 (≤4点)，长笔画后续截断而非整笔丢弃
                keep = (lens > config.min_stroke_len)
                if keep.sum() >= config.min_strokes_per_diagram:
                    ink = ink[keep]
                    lens = lens[keep]
                    clip_len = min(ink.shape[1], config.max_stroke_len)
                    ink = ink[:, :clip_len, :]
                    lens = np.minimum(lens, clip_len)  # 同步截断
                    all_data.append((ink, lens, cat_name))
                    total_drawings += 1

    print(f"\nTotal valid diagrams: {total_drawings:,} across {len(category_names)} categories")

    # 统计各类数量
    cat_counts = {}
    for _, _, cat in all_data:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt:,}")

    # 随机划分 train/valid/test
    rng = np.random.RandomState(seed)
    n = len(all_data)
    indices = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    splits = {
        "train": [all_data[i] for i in indices[:n_train]],
        "valid": [all_data[i] for i in indices[n_train:n_train + n_valid]],
        "test":  [all_data[i] for i in indices[n_train + n_valid:]],
    }

    for split_name, data in splits.items():
        print(f"  {split_name}: {len(data):,} diagrams")

    # 计算 z-score 统计量 (仅训练集, translate_to_origin 后)
    stats_data = []
    for ink, lens, _ in splits["train"]:
        ink_t = ink.copy()
        ink_t[:, :, :3] -= ink_t[:, 0:1, :3]
        stats_data.append((ink_t, lens))
    stats = compute_statistics(stats_data)
    print(f"Stats: mean_ch={stats['mean_channel']}, std_ch={stats['std_channel']}")

    # 保存 — 字段兼容 CachedDiagramDataset
    for split_name, data in splits.items():
        path = os.path.join(output_dir, f"precached_{split_name}.npz")
        inks_flat = [d[0] for d in data]
        lenses_flat = [d[1] for d in data]
        labels_flat = np.array([d[2] for d in data], dtype=object)
        np.savez_compressed(
            path,
            inks=np.array(inks_flat, dtype=object),
            lenses=np.array(lenses_flat, dtype=object),
            labels=labels_flat,
            mean_ch=stats["mean_channel"],
            std_ch=stats["std_channel"],
        )
        print(f"  {split_name}: {len(data)} diagrams → {path}")

    # 保存元数据
    meta = {
        "category_names": sorted(cat_counts.keys()),
        "category_counts": {k: int(v) for k, v in cat_counts.items()},
        "n_drawings_total": int(total_drawings),
        "split_seed": seed,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return stats


def _infer_category(path: str) -> str:
    """从文件名推断类别名: 'full_raw_cat.ndjson' → 'cat'"""
    import re
    name = os.path.splitext(os.path.basename(path))[0]
    # Strip common prefixes
    for prefix in ["full_raw_", "full_simplified_", "raw_", "simplified_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def pp_temporal_resampling(
    ink: np.ndarray,
    stroke_lens: np.ndarray,
    factor: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """时间维度 stride 降采样。

    匹配 TF stroke_dataset.py::pp_temporal_resampling
    - 短笔画 (<20点): 不降采样
    - 中等笔画 (20-100): factor=1
    - 长笔画 (>100): factor=factor
    """
    if factor <= 1:
        return ink, stroke_lens

    new_inks = []
    new_lens = []
    for i in range(ink.shape[0]):
        L = int(stroke_lens[i])
        if L < 20:
            f = 1
        elif L > 100:
            f = factor
        else:
            f = 1

        sampled = ink[i, :L:f].copy()
        # 保留最后一个 pen-up 事件
        sampled[-1, 3] = ink[i, L - 1, 3]

        # Pad 回原长度
        padded = np.zeros_like(ink[i])
        padded[:len(sampled)] = sampled
        new_inks.append(padded)
        new_lens.append(len(sampled))

    return np.stack(new_inks), np.array(new_lens, dtype=np.int32)


def pp_get_t_targets(
    ink: np.ndarray,
    stroke_lens: np.ndarray,
    n_t: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """随机采样 t ∈ [0,1]，线性插值获取对应 (x,y,pen)。

    匹配 TF stroke_dataset.py::pp_get_t_targets

    Returns:
        t_values: (n_strokes, n_t) — t 值
        t_targets: (n_strokes, n_t, 3) — (x, y, pen)
    """
    n_strokes = ink.shape[0]
    t_values = np.zeros((n_strokes, n_t), dtype=np.float32)
    t_targets = np.zeros((n_strokes, n_t, 3), dtype=np.float32)

    for i in range(n_strokes):
        L = int(stroke_lens[i])
        if L < 2:
            continue

        t = rng.uniform(0.0, 1.0, n_t)
        t_values[i] = t

        len_t = t * (L - 1)
        lower_idx = np.floor(len_t).astype(np.int32)
        upper_idx = np.minimum(np.ceil(len_t).astype(np.int32), L - 1)
        factor = (len_t - lower_idx)[:, None]  # (n_t, 1)

        lower = ink[i, lower_idx, :3]
        upper = ink[i, upper_idx, :3]

        inter = factor * upper + (1.0 - factor) * lower
        # Pen: max of lower and upper (pen 通道索引=3)
        max_pen = np.maximum(ink[i, lower_idx, 3], ink[i, upper_idx, 3])

        t_targets[i, :, :2] = inter[:, :2]
        t_targets[i, :, 2] = max_pen

    return t_values, t_targets


def to_model_batch_single(
    ink: np.ndarray,
    stroke_lens: np.ndarray,
    start_coord: np.ndarray,
    end_coord: np.ndarray,
    t_values: np.ndarray,
    t_targets: np.ndarray,
    mask_pen: bool = True,
    dct_coeffs: Optional[np.ndarray] = None,
    context_mask: Optional[np.ndarray] = None,
) -> Tuple[Dict, Dict]:
    """将单张图的处理结果转为模型输入/目标格式。

    EXACT match with smartink_torch/data/diagram_dataset.py::to_model_batch_single.
    """
    S, L, D = ink.shape
    n_t = t_values.shape[1]

    # Build 3-channel ink: [x, y, pen]
    ink_3ch = np.concatenate([ink[:, :, :2], ink[:, :, 3:4]], axis=-1)  # (S, L, 3)

    if mask_pen:
        seq_len = stroke_lens - 1
        encoder_inputs = np.zeros((S, L - 1, 3), dtype=np.float32)
        for i in range(S):
            sl = int(seq_len[i])
            encoder_inputs[i, :sl, :] = ink_3ch[i, :sl, :]
    else:
        seq_len = stroke_lens.copy()
        encoder_inputs = ink_3ch.copy()

    # decoder_inputs: teacher forcing [zeros(1), ink_3ch[:-1]]
    decoder_inputs = np.concatenate(
        [np.zeros((S, 1, 3), dtype=np.float32), ink_3ch[:, :-1, :]], axis=1)

    # start_coord: (S, 1, 2) with singleton dim
    sc = start_coord.reshape(S, 1, 2).astype(np.float32)
    ec = end_coord.reshape(S, 1, 2).astype(np.float32)

    # Flattened t-targets
    t_target_ink_flat = t_targets.reshape(S * n_t, 3).astype(np.float32)

    inputs = {
        "seq_len": seq_len.astype(np.int64),
        "encoder_inputs": encoder_inputs.astype(np.float32),
        "decoder_inputs": decoder_inputs.astype(np.float32),
        "start_coord": sc,
        "end_coord": ec,
        "num_strokes": np.array([S], dtype=np.int32),
        "t_input": t_values.astype(np.float32),
        "t_target_ink": t_target_ink_flat,
        "context_mask": context_mask,
    }

    targets = {
        "stroke": ink_3ch[:, :, :2].copy().astype(np.float32),
        "pen": ink_3ch[:, :, 2:3].copy().astype(np.float32),
        "seq_len": seq_len.astype(np.int64),
        "num_strokes": np.array([S], dtype=np.int32),
        "start_coord": sc,
        "end_coord": ec,
        "t_target_ink": t_target_ink_flat,
        "t_target_stroke": t_target_ink_flat[:, :2],
        "t_target_pen": t_target_ink_flat[:, 2:3],
    }

    # DCT 目标 (可选)
    if dct_coeffs is not None:
        targets["dct_target"] = dct_coeffs.astype(np.float32)

    return inputs, targets
