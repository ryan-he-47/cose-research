"""采样与序列构造（src/st/sampling.py）—— 替代 32 子集机制（docs/22 §5）。

- riffle 洗牌：以 p 触发，按 1/2^n 概率分 n+1 段、段间重排（用户设计 2026-08-20）。
- 首笔加权：每片第一笔 loss 权重 w_first，其余 1。
- 片内顺序策略：target 片按生成序；context 片 riffle 增广（集合性）。
"""
from __future__ import annotations

import numpy as np

from .config_st import STConfig


def riffle_shuffle(order: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """riffle 洗牌（docs/22 §5-3）。

    以概率 p 触发；触发后：P(分成 n+1 段) = 1/2^n（n=1,2,...）；段间随机重排；
    若 n+1 > 笔画数则完全随机打乱。未触发原样返回。
    """
    n_items = len(order)
    if n_items <= 1 or rng.random() >= p:
        return order
    # 抽段数：n ~ 几何分布 P(n)=1/2^n，n≥1
    n = 1
    while rng.random() < 0.5 and n < 8:  # 上限 8 防爆
        n += 1
    n_seg = n + 1
    if n_seg > n_items:
        return rng.permutation(order)
    cuts = np.sort(rng.choice(np.arange(1, n_items), size=n_seg - 1, replace=False))
    segments = np.split(order, cuts)
    perm = rng.permutation(len(segments))
    return np.concatenate([segments[i] for i in perm])


def first_stroke_weights(slice_id: np.ndarray, w_first: float) -> np.ndarray:
    """每片第一笔权重 w_first，其余 1。slice_id (S,) 已按序列顺序排列。"""
    w = np.ones(len(slice_id), dtype=np.float32)
    if len(slice_id) > 0:
        w[0] = w_first
        new_slice = slice_id[1:] != slice_id[:-1]
        w[1:][new_slice] = w_first
    return w


def shuffle_within_slices(slice_id: np.ndarray, stroke_order: np.ndarray,
                          p_graphic: float, rng: np.random.Generator,
                          text_slice_ids: frozenset = frozenset(),
                          p_text: float = 0.0) -> np.ndarray:
    """对 context 片内笔画做 riffle 洗牌（文字片与图形片分率）。

    slice_id (S,)、stroke_order (S,) 当前排列下的片标与原始索引。
    返回新的笔画索引排列（片间顺序不变，仅片内重排）。
    """
    out = []
    for sid in dict.fromkeys(slice_id.tolist()):  # 保序取唯一值
        idx = stroke_order[slice_id == sid]
        p = p_text if sid in text_slice_ids else p_graphic
        out.append(riffle_shuffle(idx, p, rng))
    return np.concatenate(out) if out else stroke_order


def sample_gmm(mu, sigma, pi, *, generator, mixture_temperature=None,
               noise_temperature=.5, top_p=1., count=1):
    """分量概率温度与分量内噪声独立；缺省分量温度保留原采样序列。"""
    import math
    import torch
    if (not math.isfinite(noise_temperature) or noise_temperature < 0 or
            not 0 < top_p <= 1 or count < 1 or
            (mixture_temperature is not None and (not math.isfinite(mixture_temperature) or mixture_temperature < 0))):
        raise ValueError('温度须为有限非负数，top_p在(0,1]内，候选数为正')
    if mu.shape[0] != 1:raise ValueError('分量采样要求单个预测分布')
    probabilities = pi[0]
    if mixture_temperature not in (None, 0):
        probabilities = (probabilities.clamp(min=1e-12).log()/mixture_temperature).softmax(-1)
    if top_p < 1:
        values, order = probabilities.sort(descending=True, stable=True)
        values = values.masked_fill(values.cumsum(0)-values >= top_p, 0.)
        probabilities = torch.zeros_like(values).scatter(0, order, values)
    vectors, components = [], []
    for _ in range(count):
        index = (pi[0].argmax().reshape(1) if mixture_temperature == 0 else
                 torch.multinomial(probabilities, 1, generator=generator))
        mean, width = mu[0].index_select(0,index), sigma[0].index_select(0,index)
        noise = torch.randn(mean.shape, device=mean.device, generator=generator)
        vectors.append(mean + width*noise*noise_temperature)
        components.append(index)
    return torch.cat(vectors), torch.cat(components)


def gmm_density_mode(mu, sigma, pi, n_iter: int = 64, *, tolerance: float = 0.0):
    """GMM 全密度数值众数（2026-09-27 用户提议，分量=基函数视角）。

    模型从未被约束"一分量一模态"——分量可以基函数叠加逼近一个局部复杂分布，
    密度峰可不在任何 μ_k 上。贪心取 argmax π 的 μ 会偏。本函数用
    Carreira-Perpiñán 固定点迭代（对角协方差）多起点（起点=各分量均值）
    搜索 log p(x) 的极大点，再按整个混合密度选择最高候选。
    有限多起点是数值搜索，不能保证找到任意 GMM 的所有局部峰。
    tolerance>0 时每8轮检查各起点的相对位移，已稳定则提前结束。

    mu/sigma: (..., K, F)；pi: (..., K)（自动归一）→ 返回 (..., F)。
    """
    import torch
    if not isinstance(mu, torch.Tensor):
        mu = torch.as_tensor(mu)
        sigma = torch.as_tensor(sigma)
        pi = torch.as_tensor(pi)
    if n_iter < 1 or tolerance < 0:
        raise ValueError("迭代次数必须为正数，tolerance不能为负数")
    # 2026-10-09：众数求解不随主干AMP降成FP16；保留float64数值核对。
    if mu.dtype not in (torch.float32, torch.float64):
        mu, sigma, pi = mu.float(), sigma.float(), pi.float()
    K = mu.shape[-2]
    F = mu.shape[-1]
    pi = (pi / pi.sum(-1, keepdim=True).clamp(min=1e-12))
    log_pi = pi.clamp(min=1e-12).log()
    log_s = sigma.clamp(min=1e-12).log()
    s2 = sigma.clamp(min=1e-12) ** 2
    const = F * 0.5 * float(torch.tensor(2 * torch.pi).log())

    x = mu.clone()                                  # (..., K, F) 多起点
    mu_e = mu.unsqueeze(-3)                         # (...,1,K,F) 分量轴
    s2_e = s2.unsqueeze(-3)
    s_e = s2_e.sqrt()
    log_pi_e = log_pi.unsqueeze(-2)                 # (...,1,K)
    log_s_sum = log_s.sum(-1).unsqueeze(-2)         # (...,1,K)

    def log_p_at(y):
        """任意点集 (...,M,F) 的 log p(y)（...,M）。"""
        dd = (y.unsqueeze(-2) - mu_e) / s_e
        lp = (dd ** 2).sum(-1).mul(-0.5).add(-log_s_sum).add(-const) + log_pi_e
        return torch.logsumexp(lp, dim=-1)

    for iteration in range(n_iter):
        d = (x.unsqueeze(-2) - mu_e) / s_e  # (...,K_r,K_c,F) 标准化差
        log_n = (-0.5 * (d ** 2).sum(-1) - log_s_sum - const)
        w = (log_n + log_pi_e).softmax(-1)          # 责任度 γ (...,K_r,K_c)
        # 对角协方差固定点：x_f ← Σγ_k μ_kf/s_kf² / Σγ_k/s_kf²
        num = (w.unsqueeze(-1) * mu_e / s2_e).sum(-2)
        den = (w.unsqueeze(-1) / s2_e).sum(-2).clamp(min=1e-12)
        updated = num / den
        if tolerance and (iteration + 1) % 8 == 0:
            stable = bool(((updated - x).abs() / sigma.clamp(min=1e-12)).max() <= tolerance)
            x = updated
            if stable:
                break
        else:
            x = updated
    # 平台抛光：融合分量的密度顶是 quartic 平坦平台，固定点会停在平台各侧
    best = log_p_at(x).max(-1, keepdim=True)
    x_best = (torch.nn.functional.one_hot(best.indices.squeeze(-1), K)
              .unsqueeze(-1).to(x.dtype) * x).sum(-2, keepdim=True)  # (...,1,F)
    s_bar = (pi.unsqueeze(-1) * sigma).sum(-2, keepdim=True)        # π 加权 σ 尺度
    d_norm = ((x - x_best) / s_bar.clamp(min=1e-12)).norm(dim=-1)   # (...,K_r)
    w = torch.where((log_p_at(x) >= best.values - 0.1) & (d_norm <= 2.0),
                    torch.ones_like(d_norm), torch.zeros_like(d_norm))
    w = (w / w.sum(-1, keepdim=True).clamp(min=1e-12)).unsqueeze(-1)
    polished = (w * x).sum(-2, keepdim=True)                        # (...,1,F)
    # 候选极大竞赛（2026-09-30 修复）：{起点(含 greedy 点), 收敛终点, 抛光点} 中
    # 取密度最大者——输出严格是找到的候选中最大值点，且必不劣于任何分量均值
    cand = torch.cat([mu, x, polished], dim=-2)                     # (...,2K+1,F)
    lp = log_p_at(cand)
    idx = lp.argmax(-1, keepdim=True).unsqueeze(-1)
    return cand.gather(-2, idx.expand(*cand.shape[:-2], 1, F)).squeeze(-2)
