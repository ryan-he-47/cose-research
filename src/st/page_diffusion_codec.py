"""只构造原冻结 E/D/DCT，不构造旧 R；严格继承 AE 的 cfg 和权重。"""
from dataclasses import fields, asdict
import hashlib
from pathlib import Path

import torch
from torch import nn

from .config_st import STConfig
from .model import StrokeEncoder, CurveDecoder
from .pretrain_model import FrozenSketchLM


class FrozenPageCodec(nn.Module):
    encode_valid = FrozenSketchLM.encode_valid

    def __init__(self, cfg, encode_chunk_size=256):
        super().__init__()
        self.cfg, self.encode_chunk_size = cfg, encode_chunk_size
        self.encoder, self.decoder = StrokeEncoder(cfg), CurveDecoder(cfg)
        layers, width = [], cfg.latent_units
        if cfg.dct_coeffs > 0:
            for _ in range(cfg.dct_layers):
                layers += [nn.Linear(width, cfg.dct_hidden), nn.ReLU()]
                width = cfg.dct_hidden
            layers += [nn.Linear(width, cfg.dct_coeffs*2+(2 if cfg.dct_dc else 0))]
        self.dct_head = nn.Sequential(*layers) if layers else None
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @classmethod
    def from_state(cls, state, **kwargs):
        raw = state["cfg"]
        raw = raw if isinstance(raw, dict) else vars(raw)
        known = {f.name for f in fields(STConfig)}
        cfg = STConfig(**{k: v for k, v in raw.items() if k in known})
        if cfg.latent_units != 48 or not cfg.unified_emb:
            raise ValueError("本版需要已接受的统一48维AE")
        model = cls(cfg, **kwargs)
        selected = {k: v for k, v in state["model"].items()
                    if k.startswith(("encoder.", "decoder.", "dct_head."))}
        model.load_state_dict(selected, strict=True)
        return model

    def checkpoint(self):
        return dict(cfg=asdict(self.cfg), model={k: v.detach().cpu() for k, v in self.state_dict().items()})


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
