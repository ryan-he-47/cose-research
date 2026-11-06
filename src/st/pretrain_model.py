"""页协议冻结 AE → R：只编码有效笔画，只对有效目标计算 GMM（2026-10-02）。"""
from __future__ import annotations

from dataclasses import fields
from contextlib import nullcontext

import torch
from torch.utils.checkpoint import checkpoint

from .config_st import STConfig
from .model import STModel, _page_pe, gmm_nll
from .pretrain_config import gpt2_small_pretrain_config
from .formal_observation import distribution_sums


def config_from_ae_checkpoint(state: dict) -> STConfig:
    """继承旧/新 AE 的实际规格，R 保持已批准的 GPT-2 small 尺寸。"""
    cfg = gpt2_small_pretrain_config()
    source = state.get("cfg", {})
    source = source if isinstance(source, dict) else vars(source)
    known = {field.name for field in fields(STConfig)}
    ae_fields = {"coord_dim", "latent_units", "max_stroke_len", "gmm_dec", "n_t_samples",
                 "endpoint_loss_w", "sigma_clamp_lo", "sigma_clamp_hi", "sigma_eps",
                 "unified_emb"}
    for name, value in source.items():
        if name in known and (name.startswith(("enc_", "dec_", "dct_")) or name in ae_fields):
            setattr(cfg, name, value)
    return cfg


def config_from_r_checkpoint(state: dict) -> STConfig:
    cfg = gpt2_small_pretrain_config()
    known = {field.name for field in fields(STConfig)}
    for name, value in state["cfg"].items():
        if name in known:
            setattr(cfg, name, value)
    return cfg


class FrozenSketchLM(STModel):
    """参数名与 STModel/独立 AE checkpoint 兼容；AE 始终 eval/no_grad。"""

    def __init__(self, cfg: STConfig | None = None, *, encode_chunk_size: int = 256,
                 activation_checkpoint: bool = True):
        cfg = cfg or gpt2_small_pretrain_config()
        if not (cfg.unified_emb and cfg.coord_dim == 2 and cfg.r_pe_mode == "page"
                and cfg.freeze_ed and not cfg.free_input_encoder and cfg.gmm_head_fp32):
            raise ValueError("冻结预训练需要统一 λ/page PE/freeze_ed/FP32 GMM，无旁路 E")
        if encode_chunk_size <= 0:
            raise ValueError("encode_chunk_size 必须为正数")
        super().__init__(cfg)
        self.encode_chunk_size = encode_chunk_size
        self.activation_checkpoint = activation_checkpoint
        self.text_aux = None
        self.visual_grounding = None
        if cfg.r_grounding_enabled:
            self.install_grounding(cfg.r_grounding_classes, cfg.r_grounding_key_dim)
        if cfg.r_text_aux_enabled:
            self.install_text_aux(cfg.r_text_aux_classes,cfg.r_text_aux_hidden)
        for module in (self.encoder, self.decoder, self.dct_head):
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def install_text_aux(self,classes,hidden=256):
        from .text_shape_readout import ShapeSpaceReadout
        if self.text_aux is not None:raise ValueError('文字辅助头已存在')
        self.cfg.r_text_aux_enabled=True
        self.cfg.r_text_aux_classes=int(classes);self.cfg.r_text_aux_hidden=int(hidden)
        self.text_aux=ShapeSpaceReadout(self.cfg.r_d_model,classes,hidden).to(next(self.r_pred.parameters()).device)

    def install_grounding(self,classes,key_dim=128):
        from .visual_grounding_auxiliary import VisualGroundingHead
        if self.visual_grounding is not None:raise ValueError('对象辅助头已存在')
        self.cfg.r_grounding_enabled=True
        self.cfg.r_grounding_classes=int(classes);self.cfg.r_grounding_key_dim=int(key_dim)
        self.visual_grounding=VisualGroundingHead(self.cfg.r_d_model,classes,key_dim).to(next(self.r_pred.parameters()).device)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in (self.encoder, self.decoder, self.dct_head):
            if module is not None:
                module.eval()
        return self

    def load_ae(self, state: dict):
        self._validation_cache = None
        weights = state.get("model", state)
        selected = {name: value for name, value in weights.items()
                    if name.startswith(("encoder.", "decoder.", "dct_head."))}
        expected = {name for name in self.state_dict()
                    if name.startswith(("encoder.", "decoder.", "dct_head."))}
        if set(selected) != expected:
            raise ValueError(f"AE 权重不匹配：缺少 {sorted(expected - set(selected))}；"
                             f"多出 {sorted(set(selected) - expected)}")
        self.load_state_dict(selected, strict=False)

    @torch.no_grad()
    def encode_valid(self, batch: dict) -> torch.Tensor:
        ink, lens = batch["ink"], batch["stroke_len"]
        if ink.ndim != 3 or ink.shape[-1] != self.cfg.coord_dim or len(ink) == 0:
            raise ValueError("ink 应为非空的 [有效笔画,点,2]")
        if ink.shape[1] > self.cfg.max_stroke_len:
            raise ValueError("点数超过 AE 位置表；请在 collate 中稀疏化")
        parts = []
        groups = batch.get('encode_groups') or ((0, len(ink), ink.shape[1]),)
        for begin, end, width in groups:
            for start in range(begin, end, self.encode_chunk_size):
                stop = min(end, start + self.encode_chunk_size)
                parts.append(self.encoder(ink[start:stop, :width], lens[start:stop]))
        encoded = torch.cat(parts)
        batch_size, padded_size = batch["stroke_pad"].shape
        embeddings = encoded.new_zeros((batch_size * padded_size, self.cfg.latent_units))
        embeddings.index_copy_(0, batch["stroke_batch_index"], encoded)
        return embeddings.view(batch_size, padded_size, self.cfg.latent_units)

    def hidden_states(self, embeddings: torch.Tensor, page_id: torch.Tensor,
                      pad_mask: torch.Tensor, *, right_padded: bool = False) -> torch.Tensor:
        """只有当前笔所属页 PE；不读取目标页号/目标坐标。"""
        r = self.r_pred
        pe = _page_pe(page_id, r.pe_dim).to(embeddings.dtype)
        hidden = r.tok_proj(torch.cat((embeddings, pe), dim=-1))
        # 2026-10-03：collate 只右填充，真实 query 的所有 padding key 都在未来。
        # 因果掩码已屏蔽它们；padding 输出不进 loss，故可省去显式 key mask。
        key_mask = None if right_padded else pad_mask
        for layer in r.layers:
            if self.training and self.activation_checkpoint and torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, True, key_mask, None,
                                    use_reentrant=False)
            else:
                hidden = layer(hidden, causal=True, key_padding_mask=key_mask, rope_pos=None)
        return hidden

    def prediction_head(self, selected_hidden: torch.Tensor):
        """selected_hidden [N,d]；缓冲头、GMM 线性头和概率计算均为 FP32。"""
        with torch.autocast(device_type=selected_hidden.device.type, enabled=False):
            hidden = selected_hidden.float().unsqueeze(0)
            if self.r_pred.head_buf is not None:
                hidden = self.r_pred.head_buf(hidden)
            return tuple(value.squeeze(0) for value in self.r_pred.head(hidden))

    @staticmethod
    def target_mask(batch: dict) -> torch.Tensor:
        pad = batch["stroke_pad"]
        return (~pad[:, :-1]) & (~pad[:, 1:]) & (batch["targetable"][:, 1:] > 0.5)

    def forward(self, batch: dict, *, auxiliary_probe=False) -> dict:
        if batch["stroke_pad"].shape[1] > self.cfg.max_context_strokes:
            raise ValueError("文档超过 R 上下文；先取文本完整页窗口")
        embeddings = batch.get('encoded_embeddings')
        if embeddings is None:embeddings = self.encode_valid(batch)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not auxiliary_probe):
            hidden = self.hidden_states(embeddings[:, :-1], batch["slice_id"][:, :-1],
                                        batch["stroke_pad"][:, :-1], right_padded=True)
        indices = batch.get('target_index')
        if indices is None:
            indices = self.target_mask(batch).flatten().nonzero().flatten()
        count = torch.tensor(indices.numel(), device=hidden.device)
        if indices.numel() == 0:
            zero = hidden.sum().float() * 0
            return {"loss": zero, "pred_nll": zero, "pred_sum": zero, "pred_count": count}
        full_probe=auxiliary_probe and self.visual_grounding is not None
        if full_probe:hidden=hidden.detach().float().requires_grad_(True)
        selected_hidden=hidden.reshape(-1, hidden.shape[-1]).index_select(0, indices)
        targets=embeddings[:, 1:].reshape(-1, embeddings.shape[-1]).index_select(0, indices)
        if auxiliary_probe and not full_probe:selected_hidden=selected_hidden.detach().float().requires_grad_(True)
        mu, sigma, pi = self.prediction_head(selected_hidden)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            individual = gmm_nll(targets.float(), mu, sigma, pi, reduction="none")
            mean = individual.mean()
        metrics = {}
        if "kind_id" in batch:
            kinds = batch.get('target_kind')
            if kinds is None:kinds = batch['kind_id'][:, 1:].reshape(-1).index_select(0, indices)
            for name, mask in (("content", kinds < 3), ("control", kinds >= 3)):
                metrics[name + "_count"] = mask.sum()
                metrics[name + "_sum"] = torch.where(mask, individual, 0.).sum()
            if "task_id" in batch:
                task_ids = batch['task_id'].index_select(0, indices // (batch['stroke_pad'].shape[1]-1))
                metrics["task_sums"] = {i: torch.where((kinds < 3) & (task_ids == i), individual, 0.).sum()
                                        for i in range(8)}
                if getattr(self, 'observe_distribution', False):
                    metrics['distribution_sums'] = distribution_sums(
                        targets,mu,sigma,pi,kinds)
            else:
                metrics = {name: value.detach() for name, value in metrics.items()}
        if self.text_aux is not None:
            from .text_auxiliary import auxiliary_statistics
            required=('shape','space','first','valid')
            if any('text_aux_'+key not in batch for key in required):
                raise ValueError('已启用文字辅助监督，但batch缺少母模板标签')
            aux_indices = batch.get('aux_target_index')
            if aux_indices is None:
                aux_indices = batch['text_aux_valid'][:, 1:].reshape(-1).index_select(0, indices).nonzero().flatten()
            if aux_indices.numel():
                # 校准在FP32测头对h的梯度，正式训练仍走FP16头/FP32 CE+BCE。
                with torch.autocast(device_type=hidden.device.type,enabled=False) if auxiliary_probe else nullcontext():
                    output=self.text_aux(selected_hidden.index_select(0, aux_indices))
                values=[batch['aux_target_'+key] if 'aux_target_'+key in batch else
                        batch['text_aux_'+key][:,1:].reshape(-1).index_select(0, indices).index_select(0, aux_indices)
                        for key in ('shape','space','first')]
                kinds=batch['kind_id'][:,1:].reshape(-1).index_select(0, indices).index_select(0, aux_indices)
                metrics.update(auxiliary_statistics(output,*values,kinds,
                    detailed=auxiliary_probe or not self.training or getattr(self, 'observe_auxiliary', True)))
                if 'learning_shape' in batch:
                    weights=[batch['learning_'+key][:,1:].reshape(-1).index_select(0,indices).index_select(0,aux_indices)
                             for key in ('shape','space')]
                    with torch.autocast(device_type=hidden.device.type,enabled=False):
                        metrics['aux_weighted_sum']=(torch.nn.functional.cross_entropy(output['shape'].float(),values[0],reduction='none')*weights[0]).sum()+(
                            torch.nn.functional.binary_cross_entropy_with_logits(output['space'].float(),values[1].float(),reduction='none')*weights[1]).sum()
        if self.visual_grounding is not None:
            metrics.update(self.visual_grounding.statistics(hidden,batch))
        if auxiliary_probe:metrics['probe_hidden']=hidden if full_probe else selected_hidden
        return {"loss": mean, "pred_nll": mean, "pred_sum": mean * count,
                "pred_count": count, **metrics}

    @torch.no_grad()
    def next_distribution(self, batch: dict):
        """完整上下文最后一笔预测下一笔；外部按生成的控制符更新页号。"""
        embeddings = self.encode_valid(batch)
        hidden = self.hidden_states(embeddings, batch["slice_id"], batch["stroke_pad"], right_padded=True)
        last = (~batch["stroke_pad"]).sum(1) - 1
        selected = hidden[torch.arange(len(last), device=hidden.device), last]
        return self.prediction_head(selected)
