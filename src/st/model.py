"""时空笔画模型（src/st/model.py）。

从零重构 —— 设计契约见 docs/22（已批准 2026-08-20）；旧库 smartink_torch/ 仅作只读参考。

核心约定：
- 点通道 [x, y, Δt]，无 pen 通道（docs/22 §2-1）；encoder 看全序列（无末点屏蔽）。
- 笔画坐标相对 start_coord 平移；start_coord 永不 z-score，三维含片内 Δt 起点。
- GMM σ为标准差；旧头用exp(clamp(raw))+eps，新T1的R头显式启用方差平方和。
- 端点辅助损失：τ∈{0,1} 固定采样（docs/21 §5.2a）。
- R_pred/R_pos 双模型（D-a）：R_pred 以目标起点位置为条件（两阶段：先哪里、后画什么）。
- 片级 RoPE（D-b）：同片笔画共享旋转角 ⇒ 片内注意力无顺序泄漏；片间扁平因果掩码。
- stop_predictive_grad=True（不变量 #2）：R 的输入嵌入与目标嵌入全部 detach。
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_st import STConfig

LOG_2PI = math.log(2.0 * math.pi)


# ═══════════════════════════════════════════════════════════════
# GMM 输出头与 NLL
# ═══════════════════════════════════════════════════════════════

class GMMDense(nn.Module):
    """对角 GMM 输出头：mu/sigma/pi。σ为标准差，下界参数也为标准差。

    sigma_eps=None 时用 cfg.sigma_eps；R 头传 cfg.sigma_eps_r（更严下限，
    防 λ 空间 NLL 跑飞——A1 崩塌事故对策，docs/22 §13）。
    """

    def __init__(self, in_dim: int, out_dim: int, n_comp: int, cfg: STConfig,
                 sigma_eps: Optional[float] = None, sigma_floor_mode: str = "add"):
        super().__init__()
        self.out_dim = out_dim
        self.n_comp = n_comp
        self.cfg = cfg
        self.sigma_eps = cfg.sigma_eps if sigma_eps is None else sigma_eps
        if sigma_floor_mode not in ("add", "quadrature"):
            raise ValueError(f"未知GMM标准差下界参数化：{sigma_floor_mode}")
        self.sigma_floor_mode = sigma_floor_mode
        self.head = nn.Linear(in_dim, (2 * out_dim + 1) * n_comp)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h (B,T,D) → mu/sigma (B,T,K,F), pi (B,T,K)。

        参数解析与概率计算使用 fp32；新训练 gmm_head_fp32=True 时，
        线性头也退出 autocast，避免在线性层后才转型。旧配置保留原行为。
        """
        B, T, _ = h.shape
        if self.cfg.gmm_head_fp32:
            with torch.autocast(device_type=h.device.type, enabled=False):
                out = self.head(h.float())
        else:
            out = self.head(h).float()
        K, Fd = self.n_comp, self.out_dim
        mu, raw_sigma, pi = torch.split(out, [Fd * K, Fd * K, K], dim=-1)
        mu = mu.view(B, T, K, Fd)
        sigma = torch.exp(torch.clamp(raw_sigma, self.cfg.sigma_clamp_lo,
                                      self.cfg.sigma_clamp_hi)).view(B, T, K, Fd)
        if self.sigma_floor_mode == "quadrature":
            # 2026-10-09：在FP32中叠加固定方差，保留宽分量的表达范围。
            sigma = torch.sqrt(sigma.square() + self.sigma_eps ** 2)
        else:
            sigma = sigma + self.sigma_eps
        pi = F.softmax(pi, dim=-1)
        return mu, sigma, pi


def gmm_nll(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
            pi: torch.Tensor, mask: Optional[torch.Tensor] = None,
            weight: Optional[torch.Tensor] = None, *, reduction: str = "mean") -> torch.Tensor:
    """对角 GMM 负对数似然。全程 fp32（见 GMMDense 注）。

    x (B,F)；mu/sigma/pi (B,K,F)/(B,K,F)/(B,K)。mask (B,) bool，True=计入；
    weight (B,) float，逐样本权重（首笔加权，docs/22 §5）。
    """
    x = x.float().unsqueeze(1)  # (B,1,F)
    log_n = (-0.5 * ((x - mu) / sigma) ** 2).sum(-1) \
            - torch.log(sigma).sum(-1) \
            - 0.5 * mu.shape[-1] * LOG_2PI
    log_prob = torch.logsumexp(torch.log(pi + 1e-12) + log_n, dim=1)  # (B,)
    nll = -log_prob
    if mask is not None:
        nll = nll[mask]
        if weight is not None:
            weight = weight[mask]
    # 2026-10-03：正式混合按任务归一化，复用同一次GMM概率计算。
    if reduction == "none":
        return nll if weight is None else nll * weight
    if reduction != "mean":
        raise ValueError("reduction 只能为mean或none")
    if nll.numel() == 0:
        return torch.zeros((), device=x.device)
    if weight is not None:
        return (nll * weight).sum() / weight.sum().clamp(min=1e-9)
    return nll.mean()


# ═══════════════════════════════════════════════════════════════
# 基础组件：因果自注意力 + 前馈
# ═══════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """多头自注意力；可选 RoPE（按给定位置索引旋转 q/k）与因果掩码。"""

    def __init__(self, d_model: int, num_heads: int, rope_theta: Optional[float] = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.rope_theta = rope_theta
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def _apply_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x (B,H,T,D)，pos (B,T) 任意非负位置索引（片索引）。同一 pos 共享旋转角。"""
        D = x.shape[-1]
        freqs = 1.0 / (self.rope_theta ** (torch.arange(0, D, 2, device=x.device,
                                                        dtype=torch.float32) / D))
        ang = pos.float().unsqueeze(-1) * freqs            # (B,T,D/2)
        cos = torch.cos(ang).unsqueeze(1)                  # (B,1,T,D/2)
        sin = torch.sin(ang).unsqueeze(1)
        x1, x2 = x[..., 0::2], x[..., 1::2]                # 交错拆半
        xr1 = x1 * cos - x2 * sin
        xr2 = x1 * sin + x2 * cos
        out = torch.stack([xr1, xr2], dim=-1).flatten(-2)  # 交错合并回 (B,H,T,D)
        return out

    def forward(self, x: torch.Tensor, causal: bool,
                key_padding_mask: Optional[torch.Tensor],
                rope_pos: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.num_heads, self.depth).transpose(1, 2)
        k = self.wk(x).view(B, T, self.num_heads, self.depth).transpose(1, 2)
        v = self.wv(x).view(B, T, self.num_heads, self.depth).transpose(1, 2)

        if rope_pos is not None:
            q = self._apply_rope(q, rope_pos)
            k = self._apply_rope(k, rope_pos)

        # SDPA（flash/mem-efficient 内核自动选择；掩码为布尔"keep"语义：
        # True=可注意。docs/22 §11——替代旧的手搓加性 float mask）
        keep = None
        if causal and key_padding_mask is not None:
            keep = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        if key_padding_mask is not None:
            # key_padding_mask: (B,T) True=padding
            keep_k = ~key_padding_mask[:, None, :]                    # (B,1,T)
            keep = keep_k if keep is None else keep.unsqueeze(0) & keep_k
        if keep is not None and keep.dim() == 3:
            keep = keep.unsqueeze(1)                                  # (B,1,T,T)
        # 2026-10-03：无 key padding 时直接走 causal 内核，不分配 T×T 掩码。
        if causal and key_padding_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.dense(out)


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dff: int, rate: float,
                 rope_theta: Optional[float] = None):
        super().__init__()
        self.mha = CausalSelfAttention(d_model, num_heads, rope_theta)
        self.ffn = nn.Sequential(nn.Linear(d_model, dff), nn.ReLU(),
                                 nn.Linear(dff, d_model))
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(rate)

    def forward(self, x, causal, key_padding_mask, rope_pos):
        x = self.ln1(x + self.drop(self.mha(x, causal, key_padding_mask, rope_pos)))
        x = self.ln2(x + self.drop(self.ffn(x)))
        return x


def _sin_pe(max_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def _page_pe(page_id: torch.Tensor, d_model: int) -> torch.Tensor:
    """2026-10-02：直接计算每笔所属页的正弦 PE，不依赖固定 max_slices 表。"""
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32,
                                 device=page_id.device)
                    * (-math.log(10000.0) / d_model))
    angles = page_id.float().unsqueeze(-1) * div
    return torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)


# ═══════════════════════════════════════════════════════════════
# E：笔画编码器（点级因果，看全序列，无 pen 通道）
# ═══════════════════════════════════════════════════════════════

class StrokeEncoder(nn.Module):
    """点序列 [x,y,Δt]（相对 start_coord）→ 笔画嵌入 λ。"""

    def __init__(self, cfg: STConfig):
        super().__init__()
        self.cfg = cfg
        self.inp = nn.Linear(cfg.coord_dim, cfg.enc_d_model)
        self.register_buffer("pos_encoding",
                             _sin_pe(cfg.max_stroke_len, cfg.enc_d_model))
        self.layers = nn.ModuleList([
            Block(cfg.enc_d_model, cfg.enc_heads, cfg.enc_dff, cfg.enc_dropout)
            for _ in range(cfg.enc_layers)
        ])
        self.out = nn.Linear(cfg.enc_d_model, cfg.latent_units)

    def forward(self, points: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """points (B,L,3) 相对坐标；lens (B,) 有效长度。返回 λ (B,latent)。"""
        B, L, _ = points.shape
        x = self.inp(points) * math.sqrt(self.cfg.enc_d_model)
        x = x + self.pos_encoding[:L].unsqueeze(0)
        for layer in self.layers:
            # 只右padding且只读最后有效位置；因果关系已经隔离未来padding。
            x = layer(x, causal=True, key_padding_mask=None, rope_pos=None)
        idx = (lens - 1).clamp(min=0).long()
        last = x[torch.arange(B, device=points.device), idx]   # (B,D)
        return self.out(last)


# ═══════════════════════════════════════════════════════════════
# D：曲线解码器（旧[τ,λ]或连续τ编码+FiLM → GMM；共用原重建损失）
# ═══════════════════════════════════════════════════════════════

class ContinuousTauEncoding(nn.Module):
    """2026-10-06：连续弧长τ的固定sin/cos编码，0.5到max_cycles对数频率。"""

    def __init__(self, dim: int, max_cycles: float):
        super().__init__()
        if dim < 4 or dim % 2 or not math.isfinite(max_cycles) or max_cycles <= .5:
            raise ValueError('τ编码维数需为至少4的偶数，最高周期数需大于0.5')
        cycles = torch.logspace(math.log10(.5), math.log10(max_cycles), dim // 2)
        self.register_buffer('angular_frequencies', cycles * (2 * math.pi))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        # AMP下仍以FP32计算相位和三角函数，随后交给FiLM线性层的autocast。
        with torch.autocast(device_type=tau.device.type, enabled=False):
            angles = tau.float().unsqueeze(-1) * self.angular_frequencies.float()
            return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


class CurveDecoder(nn.Module):
    """τ∈[0,1]参数化曲线；film模式用τ调制λ主干，输出沿用现有GMM。"""

    def __init__(self, cfg: STConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.dec_tau_mode not in ('concat', 'film'):
            raise ValueError('解码器τ模式需要concat或film')
        in_dim = cfg.latent_units + (cfg.dec_tau_mode == 'concat')
        layers = []
        for _ in range(cfg.dec_layers):
            layers += [nn.Linear(in_dim, cfg.dec_hidden), nn.ReLU()]
            in_dim = cfg.dec_hidden
        self.mlp = nn.Sequential(*layers)
        self.head = GMMDense(cfg.dec_hidden, cfg.coord_dim, cfg.gmm_dec, cfg)
        if cfg.dec_tau_mode == 'film':
            self.tau_encoding = ContinuousTauEncoding(cfg.dec_tau_pe_dim, cfg.dec_tau_max_cycles)
            self.film = nn.ModuleList([
                nn.Linear(cfg.dec_tau_pe_dim, 2 * cfg.dec_hidden) for _ in range(cfg.dec_layers)
            ])
            # γ为相对恒等缩放的增量，小非零权重让初始化时就能分辨τ。
            for modulation in self.film:
                nn.init.normal_(modulation.weight, std=1e-3)
                nn.init.zeros_(modulation.bias)

    def forward(self, emb: torch.Tensor, tau: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """emb (B,latent)；tau (B,n_t)。返回 GMM 参数 (B,n_t,K,F) / (B,n_t,K)。"""
        B, n_t = tau.shape
        e = emb.unsqueeze(1).expand(B, n_t, self.cfg.latent_units)
        if self.cfg.dec_tau_mode == 'concat':
            x = torch.cat([tau.unsqueeze(-1), e], dim=-1)
            return self.head(self.mlp(x))
        condition = self.tau_encoding(tau)
        x = e
        for index, modulation in enumerate(self.film):
            features = self.mlp[2 * index](x)
            gamma, beta = modulation(condition).chunk(2, dim=-1)
            x = self.mlp[2 * index + 1]((1 + gamma) * features + beta)
        return self.head(x)

    def recon_loss(self, emb: torch.Tensor, tau: torch.Tensor,
                   target_pts: torch.Tensor, end_pts: torch.Tensor,
                   loss_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """target_pts (B,n_t,F) 随机 τ 处的 GT 插值点；end_pts (B,2,F) τ=0/1 处 GT；
        loss_mask (B,) True=计入（padding 笔画排除）。"""
        mu, sigma, pi = self.forward(emb, tau)
        nll = gmm_nll(target_pts.reshape(-1, self.cfg.coord_dim),
                      mu.reshape(-1, self.cfg.gmm_dec, self.cfg.coord_dim),
                      sigma.reshape(-1, self.cfg.gmm_dec, self.cfg.coord_dim),
                      pi.reshape(-1, self.cfg.gmm_dec),
                      mask=loss_mask.repeat_interleave(tau.shape[1]))
        # 端点辅助损失（τ=0/1 固定采样；endpoint_loss_w=0 时整个跳过——
        # 2026-08-22 用户裁定撤回该监督项：端点定位先学会扭曲 λ 空间自然成形）
        if self.cfg.endpoint_loss_w > 0:
            tau_end = torch.zeros(emb.shape[0], 2, device=emb.device)
            tau_end[:, 1] = 1.0
            mu_e, sigma_e, pi_e = self.forward(emb, tau_end)
            nll_end = gmm_nll(end_pts.reshape(-1, self.cfg.coord_dim),
                              mu_e.reshape(-1, self.cfg.gmm_dec, self.cfg.coord_dim),
                              sigma_e.reshape(-1, self.cfg.gmm_dec, self.cfg.coord_dim),
                              pi_e.reshape(-1, self.cfg.gmm_dec),
                              mask=loss_mask.repeat_interleave(2))
        else:
            nll_end = torch.zeros((), device=emb.device)
        return {"recon_nll": nll,
                "endpoint_nll": nll_end,
                "recon_total": nll + self.cfg.endpoint_loss_w * nll_end}


# ═══════════════════════════════════════════════════════════════
# R：关系模型（片级 RoPE + 扁平因果；双模型，R_pred 以目标起点为条件）
# ═══════════════════════════════════════════════════════════════

class RelationalModel(nn.Module):
    """笔画级因果 transformer。token = [λ, start_coord]（+ 模式相关的 PE/查询）。

    位置编码模式（cfg.r_pe_mode，D27；docs/19 §D27 / docs/22 §15）：
    - rope:        片级 RoPE 旋转 q/k（D-b，已证伪：旋转固定 λ 点云 → 混叠）
    - rope_stroke: 逐笔 RoPE（对照臂，预期复现 CoSE 论文顺序编码的性能损失）
    - none:        无位置编码（消融臂）
    - concat:      片正弦 PE 拼接到 token 特征（额外自由度承载时序，不动嵌入
                   流形）；目标片 PE 作查询条件显式告知"是否进入下一时间片"。
                   本模式下 R_pos 附带片末 logit 头（终止符 e 的退化形态）。
    - page:        2026-10-02 新页协议：λ 拼当前页正弦 PE，无目标页查询，
                   无结束二分类头；页索引不受 max_slices 表长限制。

    query_pos=True（R_pred）：额外拼接 GT 下一笔起点（两阶段，docs/22 §9 D-a）。
    """

    def __init__(self, cfg: STConfig, out_dim: int, query_pos: bool,
                 dual_out_dim: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.r_pe_mode
        self.query_pos = query_pos
        self.dual_out_dim = dual_out_dim               # D28：双头模式挂嵌入 GMM 头
        if self.mode == "page":
            if not cfg.unified_emb or query_pos or dual_out_dim is not None:
                raise ValueError("page 模式使用统一 λ 的单一 R，不接受起点查询或双头")
            if cfg.slice_pe_dim <= 0 or cfg.slice_pe_dim % 2:
                raise ValueError("page 模式的 slice_pe_dim 必须是正偶数")
        self.pe_dim = cfg.slice_pe_dim if self.mode in ("concat", "page") else 0
        # 计划1 统一嵌入：位置已在 λ 内，token 只有 λ（不再拼 start_coord）
        # 2026-09-27 free_input_encoder：输入 token 是新编码器直投影的
        # r_d_model 维特征（不过 16 维瓶颈），tok_in 相应换口径
        tok_in = ((cfg.r_d_model if cfg.free_input_encoder else cfg.latent_units)
                  if cfg.unified_emb else cfg.latent_units) \
            + (0 if cfg.unified_emb else cfg.coord_dim) \
            + self.pe_dim
        self.tok_proj = nn.Linear(tok_in, cfg.r_d_model)
        if query_pos and not cfg.r_head_query:
            self.q_proj = nn.Linear(cfg.coord_dim + self.pe_dim, cfg.r_d_model)
        elif self.mode == "concat":
            self.q_proj = nn.Linear(self.pe_dim, cfg.r_d_model)
        if self.mode == "concat":
            self.register_buffer("pos_encoding",
                                 _sin_pe(cfg.max_slices, self.pe_dim))
        rope_theta = cfg.rope_theta if self.mode in ("rope", "rope_stroke") else None
        self.layers = nn.ModuleList([
            Block(cfg.r_d_model, cfg.r_heads, cfg.r_dff, cfg.r_dropout,
                  rope_theta=rope_theta)
            for _ in range(cfg.r_layers)
        ])
        # 输出通路：r_head_mlp 时加缓冲 MLP+ReLU+LayerNorm（还原 v4 的
        # DenseLayers([512,256])→GMM 通路；双头时两个头各自独立缓冲）。
        # D29：r_head_query 时目标起点拼到 GMM 头输入（查询塔专属，输入维
        # 多 coord_dim）；r_head_extra 追加 512 宽隐层加深条件解码器。
        hq = cfg.r_head_query and query_pos
        buf_in = cfg.r_d_model + (cfg.coord_dim if hq else 0)
        head_in = 256 if cfg.r_head_mlp else buf_in
        if cfg.r_head_mlp:
            buf_layers: list = []
            d_in = buf_in
            for _ in range(1 + cfg.r_head_extra):
                buf_layers += [nn.Linear(d_in, 512), nn.ReLU()]
                d_in = 512
            buf_layers += [nn.Linear(512, 256), nn.ReLU(), nn.LayerNorm(256)]
            self.head_buf = nn.Sequential(*buf_layers)
        else:
            self.head_buf = None
        self.head = GMMDense(head_in, out_dim, cfg.gmm_r, cfg,
                             sigma_eps=cfg.sigma_eps_r,
                             sigma_floor_mode=cfg.r_sigma_floor_mode)
        # D28 双头模式的嵌入 GMM 头（与位置头并联在同一 trunk 输出上）
        if dual_out_dim is not None:
            if cfg.r_head_mlp:
                self.head_emb_buf = nn.Sequential(
                    nn.Linear(cfg.r_d_model, 512), nn.ReLU(),
                    nn.Linear(512, 256), nn.ReLU(), nn.LayerNorm(256))
            else:
                self.head_emb_buf = None
            self.head_emb = GMMDense(head_in, dual_out_dim, cfg.gmm_r, cfg,
                                     sigma_eps=cfg.sigma_eps_r,
                                     sigma_floor_mode=cfg.r_sigma_floor_mode)
        else:
            self.head_emb = None
        # 片末 logit 头（concat 模式的 R_pos）：预测"下一笔是其片末笔"→1
        self.end_head = (nn.Linear(cfg.r_d_model, 1)
                         if (self.mode == "concat" and not query_pos) else None)

    def forward(self, emb: torch.Tensor, start: torch.Tensor,
                slice_id: torch.Tensor, pad_mask: torch.Tensor,
                next_start: Optional[torch.Tensor] = None,
                tgt_sid: Optional[torch.Tensor] = None):
        """emb (B,S,latent)；start (B,S,3)；slice_id (B,S) 片索引；
        pad_mask (B,S) True=padding；next_start (B,S,3) 仅 query_pos=True；
        tgt_sid (B,S) 目标片索引（仅 concat 模式查询条件；page 模式不读取）。
        返回 (mu, sigma, pi, end_logit, emb_gmm)；位置 i 的输出 = 对第 i+1 笔的预测。
        emb_gmm 仅双头模式非 None（同 trunk 同位置的嵌入 GMM 参数元组）。"""
        B, S, _ = emb.shape
        feats = [emb] if self.cfg.unified_emb else [emb, start]
        if self.mode == "concat":
            feats.append(self.pos_encoding[slice_id])          # 上下文片 PE 拼接
        elif self.mode == "page":
            feats.append(_page_pe(slice_id, self.pe_dim).to(emb.dtype))  # 当前页 PE
        tok = self.tok_proj(torch.cat(feats, dim=-1))
        # 阶段2高层注入（docs/24 §3）：late_query_layer>0 时查询不进输入层，
        # 改在第 K 层 Block 前加性注入（高层隐向量已聚合全局语义，自偏置可接受）。
        # D29：r_head_query 时 transformer 侧零注入，条件改拼到 GMM 头输入。
        q_vec = None
        if self.query_pos and not self.cfg.r_head_query:
            assert next_start is not None
            q = [next_start]
            if self.mode == "concat":
                q.append(self.pos_encoding[tgt_sid])           # 目标片 PE 查询
            q_vec = self.q_proj(torch.cat(q, dim=-1))
            if self.cfg.late_query_layer <= 0:
                tok = tok + q_vec
        elif self.mode == "concat":
            tok = tok + self.q_proj(self.pos_encoding[tgt_sid])

        if self.mode == "rope":
            rope_pos = slice_id
        elif self.mode == "rope_stroke":
            rope_pos = torch.arange(S, device=emb.device).unsqueeze(0).expand(B, S)
        else:
            rope_pos = None
        x = tok
        # 阶段1错层头（docs/24 §3）：r_mid_head_layer>0 时位置头读第 N 层输出，
        # 嵌入头仍读最终层——强迫位置信息提前凝聚，模仿"先哪里后画什么"时序
        mid = self.cfg.r_mid_head_layer
        x_pos = None
        for i, layer in enumerate(self.layers, start=1):
            if q_vec is not None and i == self.cfg.late_query_layer:
                x = x + q_vec                                # 阶段2：高层条件注入
            x = layer(x, causal=True, key_padding_mask=pad_mask, rope_pos=rope_pos)
            if mid > 0 and i == mid:
                x_pos = x
        if x_pos is None:
            x_pos = x
        # D29：查询塔的目标起点拼到 GMM 头输入（transformer 侧零注入）
        h_in = x_pos
        if self.query_pos and self.cfg.r_head_query:
            assert next_start is not None
            h_in = torch.cat([h_in, next_start], dim=-1)
        mu, sigma, pi = self.head(self.head_buf(h_in) if self.head_buf is not None else h_in)
        end_logit = self.end_head(x).squeeze(-1) if self.end_head is not None else None
        emb_gmm = (self.head_emb(self.head_emb_buf(x) if self.head_emb_buf is not None else x)
                   if self.head_emb is not None else None)
        return mu, sigma, pi, end_logit, emb_gmm


# ═══════════════════════════════════════════════════════════════
# 顶层模型
# ═══════════════════════════════════════════════════════════════

class STModel(nn.Module):
    """E + D + R_pred + R_pos。batch 契约见 forward docstring（由 pipeline_st 供给）。"""

    def __init__(self, cfg: STConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = StrokeEncoder(cfg)
        self.decoder = CurveDecoder(cfg)
        # 计划1 DCT 纯形状辅助头（docs/24 计划1；尺寸取旧实验 v1(3×128) 与
        # v2(1×32) 之间，默认 2×64——v2 单层瓶颈惩罚可能过强）
        if cfg.dct_coeffs > 0:
            layers = []
            d_in = cfg.latent_units
            for _ in range(cfg.dct_layers):
                layers += [nn.Linear(d_in, cfg.dct_hidden), nn.ReLU()]
                d_in = cfg.dct_hidden
            layers += [nn.Linear(d_in, cfg.dct_coeffs * 2
                                     + (2 if cfg.dct_dc else 0))]
            self.dct_head = nn.Sequential(*layers)
        else:
            self.dct_head = None
        if cfg.unified_emb:
            # 计划1：单一关系模型（λ-only token、无查询、无 R_pos 分离塔）
            self.r_pred = RelationalModel(cfg, out_dim=cfg.latent_units,
                                          query_pos=False)
            self.r_pos = None
            if cfg.free_input_encoder:
                # 预测专用输入编码器（2026-09-27 用户构想）：同架构随机初始化，
                # 块栈内部已有 Pre-LN；输出接口补 LayerNorm（"层间 LN"）+
                # 直投影到 r_d_model。梯度不截断，端到端由预测目标训练；
                # self.encoder 保持预训练冻结、只产 λ 目标。
                self.input_encoder = StrokeEncoder(cfg)
                self.input_encoder.out = nn.Sequential(
                    nn.LayerNorm(cfg.enc_d_model, eps=1e-6),
                    nn.Linear(cfg.enc_d_model, cfg.r_d_model))
        elif cfg.r_dual_head:
            # D28 方案1：单 trunk 双头（位置 + 嵌入并联，无查询注入）
            self.r_pos = RelationalModel(cfg, out_dim=cfg.coord_dim, query_pos=False,
                                         dual_out_dim=cfg.latent_units)
            self.r_pred = None
        else:
            self.r_pos = RelationalModel(cfg, out_dim=cfg.coord_dim, query_pos=False)
            self.r_pred = RelationalModel(cfg, out_dim=cfg.latent_units, query_pos=True)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch:
          ink         (B,S,L,3)  相对坐标点序列（padding=0）
          stroke_len  (B,S)      有效长度
          start_coord (B,S,3)    绝对起点（含片内 Δt 起点；不 z-score）
          slice_id    (B,S)      片索引
          stroke_pad  (B,S)      True=padding 笔画
          tau         (B,S,n_t)  重建采样 τ
          tau_target  (B,S,n_t,3) 对应 GT 插值点（相对坐标）
          end_target  (B,S,2,3)  τ=0/1 处 GT 点
        """
        cfg = self.cfg
        B, S, L, Fd = batch["ink"].shape
        ink = batch["ink"].reshape(B * S, L, Fd)
        lens = batch["stroke_len"].reshape(B * S)

        # ── E/D：全部笔画一次编码 + 重建损失（freeze_ed 时 decoder 整个跳过）──
        emb = self.encoder(ink, lens)                       # (B*S, latent)
        emb_d = emb.view(B, S, cfg.latent_units)
        stroke_valid = ~batch["stroke_pad"].reshape(B * S)  # True=有效笔画

        if cfg.freeze_ed:
            dec = {"recon_nll": torch.zeros((), device=emb.device),
                   "endpoint_nll": torch.zeros((), device=emb.device),
                   "recon_total": torch.zeros((), device=emb.device)}
        else:
            dec = self.decoder.recon_loss(
                emb, batch["tau"].reshape(B * S, -1),
                batch["tau_target"].reshape(B * S, cfg.n_t_samples, Fd),
                batch["end_target"].reshape(B * S, 2, Fd),
                loss_mask=stroke_valid)

        # ── DCT 纯形状辅助损失（计划1；监督 E 的 λ 线性编码低频形状，
        #    用 detach 前的 emb——这是 AE 侧损失，梯度本就只流向 E/D。
        #    freeze_ed 时 E 冻结，目标恒定，与 decoder 同步跳过）──
        if (self.dct_head is not None and "dct_target" in batch
                and not cfg.freeze_ed):
            dt_pred = self.dct_head(emb)                     # (B*S, 2K)
            dt_gt = batch["dct_target"].reshape(emb.shape[0], -1)
            per = ((dt_pred - dt_gt) ** 2).mean(-1)          # (B*S,)
            w = stroke_valid.float()
            dct_mse = (per * w).sum() / w.sum().clamp(min=1.0)
        else:
            dct_mse = torch.zeros((), device=emb.device)

        # ── R：预测下一笔（shift 语义），全位置监督 ──
        # 位置 i 预测 i+1；stop_predictive_grad：R 输入与目标全部 detach
        # 2026-09-27 free_input_encoder：R 输入 = 新编码器输出（不截梯度，
        # 端到端训练），目标仍 = 冻结编码器的 λ（detach 兜底）
        if cfg.free_input_encoder:
            emb_in = self.input_encoder(ink, lens).view(B, S, cfg.r_d_model)
        else:
            emb_in = emb_d.detach() if cfg.stop_predictive_grad else emb_d
        start = batch["start_coord"]
        slice_id = batch["slice_id"]
        pad = batch["stroke_pad"]

        emb_ctx = emb_in[:, :-1]
        start_ctx = start[:, :-1]
        slice_ctx = slice_id[:, :-1]
        pad_ctx = pad[:, :-1]

        tgt_emb = emb_d[:, 1:].detach()                     # (B,S-1,latent)
        tgt_start = start[:, 1:]                            # (B,S-1,3)
        valid_next = (~pad[:, 1:]) & (~pad_ctx)             # 预测与目标都有效
        # targetable=0 的笔画（文字片等纯上下文）永不做预测目标（docs/22 §6 角色语义）
        if "targetable" in batch:
            valid_next = valid_next & (batch["targetable"][:, 1:] > 0.5)
        # 首笔加权：权重按目标位置取（loss_weight[b,s] = 第 s 笔的权重，
        # 作为目标时是 s+1 位置被预测 → 取 [:,1:]）
        w_next = None
        if "loss_weight" in batch:
            w_next = batch["loss_weight"][:, 1:].reshape(-1).float()

        # R_pos：下一笔起点（concat 模式下带目标片 PE 查询 + 片末 logit 头）
        # r_skip_pos（docs/24 §3 阶段2修正案）：单塔诊断只训嵌入塔，位置塔整个跳过
        # 计划1 unified_emb：位置已在 λ 内，无位置塔（pos_nll 恒零）
        tgt_sid = slice_id[:, 1:]
        if cfg.unified_emb or cfg.r_skip_pos:
            pos_nll = torch.zeros((), device=emb.device)
            end_logit = None
            emb_gmm = None
        else:
            mu_p, sigma_p, pi_p, end_logit, emb_gmm = self.r_pos(
                emb_ctx, start_ctx, slice_ctx, pad_ctx, tgt_sid=tgt_sid)
            pos_nll = gmm_nll(tgt_start.reshape(-1, cfg.coord_dim),
                              mu_p.reshape(-1, cfg.gmm_r, cfg.coord_dim),
                              sigma_p.reshape(-1, cfg.gmm_r, cfg.coord_dim),
                              pi_p.reshape(-1, cfg.gmm_r),
                              mask=valid_next.reshape(-1), weight=w_next)

        if cfg.unified_emb:
            # 计划1：单一 R 直接预测下一笔统一嵌入（无查询、无位置条件）
            mu_e, sigma_e, pi_e, _, _ = self.r_pred(
                emb_ctx, start_ctx, slice_ctx, pad_ctx, tgt_sid=tgt_sid)
        elif cfg.r_dual_head:
            # D28：嵌入 GMM 与位置共享 trunk（无查询注入）
            assert emb_gmm is not None
            mu_e, sigma_e, pi_e = emb_gmm
        else:
            # R_pred：以 GT 下一笔起点为查询条件（教师强制；推理时换 R_pos 预测值）
            mu_e, sigma_e, pi_e, _, _ = self.r_pred(
                emb_ctx, start_ctx, slice_ctx, pad_ctx,
                next_start=tgt_start, tgt_sid=tgt_sid)
        pred_nll = gmm_nll(tgt_emb.reshape(-1, cfg.latent_units),
                           mu_e.reshape(-1, cfg.gmm_r, cfg.latent_units),
                           sigma_e.reshape(-1, cfg.gmm_r, cfg.latent_units),
                           pi_e.reshape(-1, cfg.gmm_r),
                           mask=valid_next.reshape(-1), weight=w_next)

        total = (dec["recon_total"] + pred_nll + cfg.pos_loss_w * pos_nll
                 + cfg.dct_weight * dct_mse)
        out = {"loss": total,
               "recon_nll": dec["recon_nll"],
               "endpoint_nll": dec["endpoint_nll"],
               "pred_nll": pred_nll,
               "pos_nll": pos_nll,
               "dct_mse": dct_mse}
        # 片末 logit 头（concat 模式）：目标=下一笔是否其片末笔（结构信号，
        # 与 targetable 无关——文字片边界也监督）
        if end_logit is not None:
            sentinel = torch.full_like(slice_id[:, :1], -1)
            sid_after = torch.cat([slice_id[:, 2:], sentinel], dim=1)
            y_end = (sid_after != slice_id[:, 1:]).float()
            m_end = ((~pad[:, 1:]) & (~pad_ctx)).float()
            bce = F.binary_cross_entropy_with_logits(end_logit, y_end,
                                                     reduction="none")
            end_bce = (bce * m_end).sum() / m_end.sum().clamp(min=1.0)
            out["loss"] = out["loss"] + cfg.end_head_w * end_bce
            out["end_bce"] = end_bce
        # pred 分桶日志：e 目标 vs 普通目标（pred_nll 是双峰混合均值，
        # 窗口振荡排查需要分开看；docs/22 §13 后续）
        if cfg.use_terminator:
            with torch.no_grad():
                is_e = ((tgt_start[..., 0] - cfg.term_pos[0]).abs() < 1e-3) & \
                       ((tgt_start[..., 1] - cfg.term_pos[1]).abs() < 1e-3)
                flat = lambda m: (m).reshape(-1)
                out["pred_e_nll"] = gmm_nll(
                    tgt_emb.reshape(-1, cfg.latent_units),
                    mu_e.reshape(-1, cfg.gmm_r, cfg.latent_units),
                    sigma_e.reshape(-1, cfg.gmm_r, cfg.latent_units),
                    pi_e.reshape(-1, cfg.gmm_r),
                    mask=flat(valid_next & is_e))
                out["pred_g_nll"] = gmm_nll(
                    tgt_emb.reshape(-1, cfg.latent_units),
                    mu_e.reshape(-1, cfg.gmm_r, cfg.latent_units),
                    sigma_e.reshape(-1, cfg.gmm_r, cfg.latent_units),
                    pi_e.reshape(-1, cfg.gmm_r),
                    mask=flat(valid_next & ~is_e))
        return out
