# 自 smartink_torch/util/dct.py 迁入（2026-09-30 项目清理）；smartink_torch 已归档。
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.fft import dct


def stroke_to_dct(
    points: np.ndarray,
    num_resampled: int = 128,
    num_coeffs: int = 8,
    return_mean: bool = True,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    """单条笔画 → DCT 低频系数.

    Args:
        points: (N, 2) 笔画轨迹, [:, 0]=X, [:, 1]=Y
        num_resampled: 等弧长重采样后的点数
        num_coeffs: 每通道保留的非直流 DCT 系数个数 (k=1..num_coeffs)
        return_mean: 是否返回去中心化的均值

    Returns:
        coeffs: (num_coeffs, 2) DCT 系数, 第一列=X, 第二列=Y
        mean: (x_mean, y_mean) 如果 return_mean=True, 否则 None

    Raises:
        ValueError: 当 N < 2 时
    """
    if points.shape[0] < 2:
        raise ValueError(f"笔画至少需要 2 个点, 实际 {points.shape[0]}")

    # ── Step 1: 等弧长重采样 ──
    x_orig = points[:, 0].astype(np.float64)
    y_orig = points[:, 1].astype(np.float64)

    # 累计弧长
    dx = np.diff(x_orig)
    dy = np.diff(y_orig)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.zeros(len(x_orig), dtype=np.float64)
    s[1:] = np.cumsum(ds)
    total_len = s[-1]

    if total_len < 1e-12:
        # 所有点重合 → DCT 输出全零
        coeffs = np.zeros((num_coeffs, 2), dtype=np.float32)
        mean = (float(x_orig[0]), float(y_orig[0])) if return_mean else None
        return coeffs, mean

    # 均匀采样弧长
    s_uniform = np.linspace(0.0, total_len, num_resampled)
    x_resampled = np.interp(s_uniform, s, x_orig)
    y_resampled = np.interp(s_uniform, s, y_orig)

    # ── Step 2: 去中心化 ──
    x_mean = np.mean(x_resampled)
    y_mean = np.mean(y_resampled)
    x_centered = x_resampled - x_mean
    y_centered = y_resampled - y_mean

    # ── Step 3: DCT-II 变换 (正交归一化) ──
    dct_x = dct(x_centered, type=2, norm="ortho")
    dct_y = dct(y_centered, type=2, norm="ortho")

    # ── Step 4: 截取非直流系数 k=1..num_coeffs ──
    coeffs_x = dct_x[1 : num_coeffs + 1]
    coeffs_y = dct_y[1 : num_coeffs + 1]

    coeffs = np.stack([coeffs_x, coeffs_y], axis=-1).astype(np.float32)

    mean = (float(x_mean), float(y_mean)) if return_mean else None
    return coeffs, mean


def compute_dct_for_strokes(
    ink: np.ndarray,
    stroke_lens: np.ndarray,
    num_resampled: int = 128,
    num_coeffs: int = 8,
    include_mean: bool = False,
) -> np.ndarray:
    """批量计算笔画的 DCT 系数 (供数据管线使用).

    Args:
        ink: (S, L, 4) 笔画数据, channels=[x, y, t, pen]
        stroke_lens: (S,) 每条笔画的有效长度
        num_resampled: 等弧长重采样点数
        num_coeffs: 每通道保留的非直流 DCT 系数个数
        include_mean: True 时末尾追加质心 (x̄, ȳ)（直流分量的等价格式：
            去中心化前的均值 = k=0 的 1/√N；统一画布坐标下即笔画位置。
            2026-09-24 用户裁定：显式位置+形状解耦监督，B 臂）

    Returns:
        dct_coeffs: (S, num_coeffs * 2 [+2]) — [x_k1..x_kK, y_k1..y_kK, (x̄, ȳ)]
    """
    S = ink.shape[0]
    w = num_coeffs * 2 + (2 if include_mean else 0)
    out = np.zeros((S, w), dtype=np.float32)

    for i in range(S):
        L = int(stroke_lens[i])
        if L < 2:
            continue

        points = ink[i, :L, :2]  # (L, 2) — 仅取 x, y

        try:
            coeffs, mean = stroke_to_dct(
                points,
                num_resampled=num_resampled,
                num_coeffs=num_coeffs,
                return_mean=include_mean,
            )
            # coeffs: (num_coeffs, 2)
            out[i, :num_coeffs] = coeffs[:, 0]  # X 系数
            out[i, num_coeffs:2 * num_coeffs] = coeffs[:, 1]  # Y 系数
            if include_mean and mean is not None:
                out[i, 2 * num_coeffs:] = mean      # 质心（位置监督）
        except ValueError:
            # 不足 2 点, 保持零
            pass

    return out
