"""Transformer 学习率调度（src/st/lr.py）。

自 src/train.py 迁入（2026-09-30 项目清理）；行为不变。
"""
from __future__ import annotations

import torch


class TransformerLR:
    """Transformer 学习率调度: warmup + 1/sqrt(step).

    严格匹配论文使用的 Transformer 调度。
    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int = 64,
        warmup_steps: int = 4000,
        lr_scale: float = 1.0,
        group_scales=None,
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        # lr_scale：batch 缩放的线性法则校正（2026-08-23 docs/24 §3 探针：
        # v4=B=80，本地 B=24 → lr_scale≈0.3；默认 1.0 不影响旧管线）
        self.lr_scale = lr_scale
        # group_scales：分组 LR（docs/24 阶段3：AE 组以 Ḡ/32=0.234 速率训练，
        # 复刻原管线 AE:R = 7.5笔:32预测 的信号比；None=所有组同 LR，旧行为）
        self.group_scales = group_scales
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._compute_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            scale = (self.group_scales[i] if self.group_scales is not None else 1.0)
            param_group["lr"] = lr * scale
        return lr

    def _compute_lr(self) -> float:
        step = self._step
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** (-1.5))
        return self.lr_scale * (self.d_model ** (-0.5)) * min(arg1, arg2)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]
