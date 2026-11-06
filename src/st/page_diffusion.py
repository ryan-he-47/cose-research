"""2026-11-02：页间 RoPE、页内正弦拼接的连续块扩散 R。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import _page_pe


@dataclass(frozen=True)
class PageDiffusionConfig:
    version: str = "page_diffusion_v2"
    latent: int = 48
    local_pe: int = 32
    width: int = 768
    layers: int = 13
    heads: int = 12
    ffn: int = 2048
    condition: int = 128
    block: int = 16
    rope_theta: float = 10000.
    context: int = 1024
    activation_checkpoint: bool = True
    training_blocks: tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "training_blocks", tuple(self.training_blocks))
        if self.training_blocks and (self.block not in self.training_blocks or
                any(type(width) is not int or width < 1 for width in self.training_blocks) or
                len(set(self.training_blocks)) != len(self.training_blocks)):
            raise ValueError("原生块长清单必须包含默认块长，并由互异正整数组成")
        if self.version != "page_diffusion_v2":
            raise ValueError("当前为无分类辅助的page_diffusion_v2；旧模型请使用其绑定的源码提交")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("注意力头须整除且每头维度为偶数")
        if min(self.latent, self.local_pe, self.width, self.layers, self.block, self.ffn) < 1:
            raise ValueError("模型维度必须为正")
        if self.local_pe % 2 or self.rope_theta <= 1:
            raise ValueError("位置编码配置无效")


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return (value * self.weight.float()).to(x.dtype)


def page_rotary(pages, dim, theta):
    freq = theta ** (-torch.arange(0, dim, 2, device=pages.device, dtype=torch.float32) / dim)
    angle = pages.float()[:, None, :, None] * freq
    return angle.cos(), angle.sin()


def rotate_pages(value, pages, theta, rotary=None):
    """旋转 Q/K，不旋转 λ 或 V；不同页的角差对应页号差。"""
    a, b = value[..., 0::2].float(), value[..., 1::2].float()
    c, s = page_rotary(pages, value.shape[-1], theta) if rotary is None else rotary
    return torch.stack((a*c-b*s, a*s+b*c), -1).flatten(-2).to(value.dtype)


def attention_bias(allowed, dtype):
    """一次转换并对齐行跨度；所有层共用，不在每层重复转换布尔掩码。"""
    if allowed.dtype != torch.bool:
        return allowed
    width = allowed.shape[-1]
    pitch = (width+7)//8*8
    bias = torch.full((*allowed.shape[:-1], pitch), -torch.inf,
                      dtype=dtype, device=allowed.device)[..., :width]
    return bias.masked_fill_(allowed, 0.)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.theta = cfg.heads, cfg.rope_theta
        self.qkv = nn.Linear(cfg.width, 3*cfg.width, bias=False)
        self.output = nn.Linear(cfg.width, cfg.width, bias=False)

    def forward(self, x, pages, allowed, past=None, *, rotary=None, clean_size=None):
        b, n, w = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, w//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = rotate_pages(q, pages, self.theta, rotary), rotate_pages(k, pages, self.theta, rotary)
        if past is not None:
            k = torch.cat((past[0], k), -2)
            v = torch.cat((past[1], v), -2)
        if clean_size is None:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.)
        else:
            if past is not None:
                raise ValueError("双流训练注意力不接收推理缓存")
            # 有效干净query只看过去；其右填充输出不被目标读取，也不进入损失。
            clean = F.scaled_dot_product_attention(q[:, :, :clean_size], k[:, :, :clean_size],
                v[:, :, :clean_size], is_causal=True, dropout_p=0.)
            if clean_size < n:
                noisy = F.scaled_dot_product_attention(q[:, :, clean_size:], k, v,
                    attn_mask=allowed, dropout_p=0.)
                y = torch.cat((clean, noisy), dim=2)
            else:
                y = clean
        return self.output(y.transpose(1, 2).reshape(b, n, w)), (k, v)


class DiffusionLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(cfg.width), RMSNorm(cfg.width)
        self.attention = Attention(cfg)
        self.up = nn.Linear(cfg.width, 2*cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.width, bias=False)
        self.modulation = nn.Linear(cfg.condition, 6*cfg.width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, condition, pages, allowed, past=None, *, rotary=None, clean_size=None):
        s1, a1, g1, s2, a2, g2 = self.modulation(condition).chunk(6, -1)
        update, kv = self.attention(self.norm1(x)*(1+a1)+s1, pages, allowed, past,
                                    rotary=rotary, clean_size=clean_size)
        x = x + g1*update
        u, v = self.up(self.norm2(x)*(1+a2)+s2).chunk(2, -1)
        return x + g2*self.down(F.silu(u)*v), kv


class PageDiffusionR(nn.Module):
    def __init__(self, cfg: PageDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(cfg.latent+cfg.local_pe, cfg.width, bias=False)
        self.roles = nn.Embedding(2, cfg.width)
        self.condition_roles = nn.Embedding(2, cfg.condition)
        self.time = nn.Sequential(nn.Linear(256, cfg.condition), nn.SiLU(),
                                  nn.Linear(cfg.condition, cfg.condition))
        self.layers = nn.ModuleList(DiffusionLayer(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.width)
        self.final_modulation = nn.Linear(cfg.condition, 2*cfg.width)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        self.output = nn.Linear(cfg.width, cfg.latent)
        nn.init.normal_(self.output.weight, std=.001)
        nn.init.zeros_(self.output.bias)
        self.register_buffer("latent_mean", torch.zeros(cfg.latent))
        self.register_buffer("latent_scale", torch.ones(()))
        self.register_buffer("eop_latent", torch.zeros(cfg.latent))

    def normalize(self, latent):
        return (latent.float()-self.latent_mean)/self.latent_scale

    def denormalize(self, x):
        return x.float()*self.latent_scale+self.latent_mean

    def forward(self, x, pages, local, roles, times, allowed, *, past=None,
                return_cache=False, clean_size=None, attention_layout="dense"):
        """没有真实长度或填充身份输入。roles: 0=干净历史，1=带噪槽位。"""
        cfg = self.cfg
        pe = _page_pe(local, cfg.local_pe).to(x.dtype)
        h = self.input(torch.cat((x, pe), -1))
        h = h + self.roles(roles).to(h.dtype)
        freq = torch.exp(-math.log(10000)*torch.arange(128, device=x.device).float()/128)
        angles = times.float()[..., None]*freq*1000
        t = torch.cat((angles.cos(), angles.sin()), -1)
        condition = self.time(t).to(h.dtype)*(roles == 1)[..., None] + self.condition_roles(roles).to(h.dtype)
        condition = F.silu(condition)
        rotary = page_rotary(pages, cfg.width//cfg.heads, cfg.rope_theta)
        split = None
        if attention_layout not in ("dense", "aligned", "split"):
            raise ValueError("未知注意力执行方式")
        if attention_layout == "split":
            if clean_size is None or past is not None or return_cache:
                raise ValueError("双流训练注意力需要clean_size且不能使用缓存")
            split = int(clean_size)
            if allowed.shape[-2] == x.shape[1]:
                allowed = allowed[..., split:, :]
            if allowed.shape[-2:] != (x.shape[1]-split, x.shape[1]):
                raise ValueError("带噪query掩码形状与双流长度不符")
        if attention_layout != "dense":
            allowed = attention_bias(allowed, h.dtype)
        caches = []
        for index, layer in enumerate(self.layers):
            previous = None if past is None else past[index]
            if self.training and cfg.activation_checkpoint and torch.is_grad_enabled():
                if past is not None or return_cache:
                    raise ValueError("训练检查点路径不使用推理缓存")
                def run(value, cond, module=layer):
                    return module(value, cond, pages, allowed, rotary=rotary, clean_size=split)[0]
                h = checkpoint(run, h, condition, use_reentrant=False)
            else:
                h, kv = layer(h, condition, pages, allowed, previous, rotary=rotary, clean_size=split)
                if return_cache:
                    caches.append(kv)
        shift, scale = self.final_modulation(condition).chunk(2, -1)
        hidden = self.norm(h)*(1+scale)+shift
        return {"prediction": self.output(hidden), "hidden": hidden,
                "cache": caches if return_cache else None}

    def configuration(self):
        cfg = asdict(self.cfg)
        if self.cfg.training_blocks:
            cfg["training_blocks"] = list(self.cfg.training_blocks)
        else:
            del cfg["training_blocks"]
        return cfg
