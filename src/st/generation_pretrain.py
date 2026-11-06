"""2026-10-03：页 LM 自回归生成、逐笔 KV 缓存和几何控制符识别。"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from collections import OrderedDict
import time

import numpy as np
import torch
from torch.nn import functional as F

from src.page_document import control_stroke
from .ae_data import cap_stroke
from .model import _page_pe
from .sampling import gmm_density_mode, sample_gmm


@dataclass
class PageCursor:
    """EOP 属于原页；只有后续笔画换页，EOD → EOP 才结束文档。"""
    page_id: int
    pending_eod: bool = False
    complete: bool = False

    def append(self, kind: str):
        if kind == "eop":
            self.complete = self.pending_eod
            self.pending_eod = False
            if not self.complete:
                self.page_id += 1
        else:
            self.pending_eod = kind == "eod"


class PageKVCache:
    """单条无 padding 上下文；复用原层权重和 post-norm，训练前向不变。"""
    def __init__(self, model):
        if model.training:
            raise ValueError("KV cache 只用于 eval 推理")
        self.model = model
        self.entries = []
        self.length = 0

    def truncate(self, length):
        if not 0 <= length <= self.length:
            raise ValueError('缓存截断位置超出有效前缀')
        self.length = length

    def fork(self):
        other = PageKVCache(self.model)
        other.length = self.length
        other.entries = [(k.clone(), v.clone()) for k, v in self.entries]
        return other

    @torch.no_grad()
    def forward(self, embeddings, page_ids, *, prefill=False):
        if embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError("缓存要求单条 [1,T,latent] 上下文")
        if not prefill and not self.entries:
            raise ValueError("先prefill，再追加笔画")
        past = 0 if prefill else self.length
        length = past + embeddings.shape[1]
        if length > self.model.cfg.max_context_strokes:
            raise ValueError("缓存超过上下文")
        r = self.model.r_pred
        pe = _page_pe(page_ids, r.pe_dim).to(embeddings.dtype)
        x = r.tok_proj(torch.cat((embeddings, pe), -1))
        keep = None
        if past and embeddings.shape[1] > 1:
            keep = torch.arange(length, device=x.device)[None, :] <= (
                past + torch.arange(embeddings.shape[1], device=x.device)[:, None])
        for index, layer in enumerate(r.layers):
            attention = layer.mha
            if attention.rope_theta is not None:
                raise ValueError("此缓存仅支持新 page PE 模式")
            b, t, d = x.shape
            split = lambda z: z.view(b, t, attention.num_heads, attention.depth).transpose(1, 2)
            q, k, v = (split(projection(x)) for projection in
                       (attention.wq, attention.wk, attention.wv))
            if index == len(self.entries):
                shape = (b, attention.num_heads, self.model.cfg.max_context_strokes, attention.depth)
                self.entries.append((k.new_empty(shape), v.new_empty(shape)))
            keys, values = self.entries[index]
            keys[:, :, past:length].copy_(k)
            values[:, :, past:length].copy_(v)
            # 后缀query的因果对角线偏移past；单笔query可看全部有效key。
            out = F.scaled_dot_product_attention(q, keys[:, :, :length], values[:, :, :length],
                attn_mask=keep, is_causal=past == 0)
            out = attention.dense(out.transpose(1, 2).contiguous().view(b, t, d))
            x = layer.ln1(x + layer.drop(out))
            x = layer.ln2(x + layer.drop(layer.ffn(x)))
        self.length = length
        return x


class PageGenerator:
    def __init__(self, model, *, max_points=76, control_tolerance=.018):
        if max_points < 2 or max_points > model.cfg.max_stroke_len:
            raise ValueError("max_points 超出 AE 点数范围")
        if control_tolerance <= 0:
            raise ValueError("control_tolerance 必须为正数")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.max_points, self.control_tolerance = max_points, control_tolerance
        self.templates = {}
        self.encoding_cache = OrderedDict()
        with torch.no_grad(), self.autocast():
            for kind in ("eop", "eod"):
                raw = control_stroke(kind, max_points=max_points)
                embedding = self.encode([raw])
                decoded = self.decode(embedding)[0]
                # 同时参考真实控制笔和本 AE 的解码形状，容纳重建误差。
                raw_uniform = self.uniform_points(raw)
                self.templates[kind] = (raw_uniform, decoded)
        self.template_tensor = torch.as_tensor(np.asarray(list(self.templates.values())), device=self.device)

    def autocast(self):
        return (torch.autocast("cuda", dtype=torch.float16)
                if self.device.type == "cuda" else nullcontext())

    def uniform_points(self, points):
        positions = np.linspace(0, len(points) - 1, self.max_points)
        return np.column_stack([np.interp(positions, np.arange(len(points)), points[:, dim])
                                for dim in range(2)]).astype(np.float32)

    @torch.no_grad()
    def encode(self, strokes):
        strokes = [cap_stroke(p, self.max_points) for p in strokes]
        lengths = [len(p) for p in strokes]
        ink = np.zeros((len(strokes), max(lengths), 2), dtype=np.float32)
        for index, points in enumerate(strokes):
            ink[index, :len(points)] = points
        return self.encode_tensor(torch.from_numpy(ink).to(self.device),
                                  torch.tensor(lengths, device=self.device))

    @torch.no_grad()
    def encode_tensor(self, points, lengths=None):
        ink = (points.float() - points.new_tensor([.5, .6], dtype=torch.float32)) / .6
        if lengths is None:
            lengths = torch.full((len(points),), points.shape[1], device=points.device, dtype=torch.long)
        return torch.cat([self.model.encoder(ink[i:i + self.model.encode_chunk_size],
                                             lengths[i:i + self.model.encode_chunk_size])
                          for i in range(0, len(ink), self.model.encode_chunk_size)])

    @torch.no_grad()
    def encode_cached(self, strokes):
        """标准曲线缓存包含绝对坐标、有效点数及本生成器的冻结E/精度。"""
        keys, missing = [], {}
        for curve in strokes:
            curve = cap_stroke(curve, self.max_points)
            key = (torch.is_autocast_enabled(), curve.shape, curve.astype(np.float32).tobytes())
            keys.append(key)
            if key not in self.encoding_cache:
                missing[key] = curve
        if missing:
            encoded = self.encode(list(missing.values()))
            self.encoding_cache.update(zip(missing, encoded.unbind()))
        result = torch.stack([self.encoding_cache[key] for key in keys])
        for key in keys:self.encoding_cache.move_to_end(key)
        while len(self.encoding_cache) > 4096:self.encoding_cache.popitem(last=False)
        return result

    @torch.no_grad()
    def decode(self, embedding, *, mode="greedy", density_iterations=128):
        return self.decode_tensor(embedding, mode=mode, density_iterations=density_iterations).cpu().numpy()

    @torch.no_grad()
    def decode_tensor(self, embedding, *, mode="greedy", density_iterations=128):
        if mode not in ("greedy", "density"):
            raise ValueError("decoder mode 必须为 greedy/density")
        tau = torch.linspace(0, 1, self.max_points, device=self.device).expand(len(embedding), -1)
        mu, sigma, pi = self.model.decoder(embedding, tau)
        if mode == "density":
            points = gmm_density_mode(mu, sigma, pi, n_iter=density_iterations, tolerance=1e-5)
        else:
            best = pi.argmax(-1)[..., None, None].expand(-1, -1, 1, 2)
            points = mu.gather(2, best).squeeze(2)
        return points * .6 + mu.new_tensor([.5, .6])

    def recognize(self, points):
        scores = {kind: min(float(np.sqrt(np.square(points - template).sum(-1).mean()))
                            for template in templates)
                  for kind, templates in self.templates.items()}
        nearest = min(scores, key=scores.get)
        kind = nearest if scores[nearest] <= self.control_tolerance else "body"
        return kind, scores

    @staticmethod
    def _window_start(page_ids, limit):
        """溢出时从旧完整页末笔 EOP 重建窗口，不更改页号或丢半页。"""
        for index in range(1, len(page_ids)):
            if page_ids[index] != page_ids[index - 1] and len(page_ids) - index + 1 <= limit:
                return index - 1
        return None

    @torch.no_grad()
    def generate(self, prefix, *, max_new_strokes=128, max_new_pages=4,
                 mode="greedy", temperature=.5, seed=42, callback=None, cancelled=None,
                 decoder_mode="greedy", density_iterations=128, feedback_mode="curve",
                 mixture_temperature=None, noise_temperature=None, top_p=1.,
                 candidate_count=1, cycle_penalty=1., outside_penalty=1.):
        if not prefix or max_new_strokes < 1 or max_new_pages < 1 or temperature < 0:
            raise ValueError("需要非空条件和正数生成预算；temperature >= 0")
        if mode not in ("greedy", "sample", "density"):
            raise ValueError("mode 必须为 greedy/sample/density")
        if decoder_mode not in ("greedy", "density") or density_iterations < 1:
            raise ValueError("decoder_mode必须为greedy/density，众数迭代数必须为正数")
        if feedback_mode not in ("curve", "latent"):
            raise ValueError("feedback_mode 必须为 curve/latent")
        if candidate_count < 1 or cycle_penalty < 0 or outside_penalty < 0:
            raise ValueError('候选数须为正，闭环/越界惩罚须非负')
        if candidate_count > 1 and mode != 'sample':
            raise ValueError('多候选闭环评分用于sample模式')
        noise_temperature = temperature if noise_temperature is None else noise_temperature
        records = [{"page_id": e.page_id, "kind": e.kind, "generated": False,
                    "points": e.rawxy.tolist()} for e in prefix]
        page_ids = [e.page_id for e in prefix]
        cursor = PageCursor(page_ids[-1])
        cursor.append(prefix[-1].kind)
        first_generated_page = cursor.page_id
        random = torch.Generator(device=self.device).manual_seed(seed)
        limit = self.model.cfg.max_context_strokes
        start = 0 if len(prefix) <= limit else self._window_start(page_ids, limit)
        if start is None:
            raise ValueError("条件的单页超过上下文，请缩短条件")
        cache = PageKVCache(self.model)
        started = time.monotonic()
        stop = "max_new_strokes"
        pending_curves = []
        with self.autocast():
            embeddings = self.encode([e.rawxy for e in prefix]).unsqueeze(0)
            ids = torch.tensor([page_ids], device=self.device)
            hidden = cache.forward(embeddings[:, start:], ids[:, start:], prefill=True)
            for _ in range(max_new_strokes):
                if cancelled is not None and cancelled():
                    stop = "cancelled"
                    break
                if cursor.complete:
                    stop = "eod_eop"
                    break
                if cursor.page_id - first_generated_page >= max_new_pages:
                    stop = "max_new_pages"
                    break
                mu, sigma, pi = self.model.prediction_head(hidden[:, -1])
                density_info = {}
                if mode == "density":
                    next_embedding = gmm_density_mode(mu, sigma, pi, n_iter=density_iterations, tolerance=1e-5)
                    # 仅记录峰处责任度最高的分量；输出本身可不在任何分量均值上。
                    joint = pi.clamp(min=1e-12).log() - sigma.log().sum(-1) - .5 * (((next_embedding[:, None] - mu) / sigma)**2).sum(-1)
                    component = int(joint[0].argmax())
                    greedy_component = int(pi[0].argmax())
                    greedy_joint = pi.clamp(min=1e-12).log() - sigma.log().sum(-1) - .5 * (((mu[:, greedy_component, None] - mu) / sigma)**2).sum(-1)
                    density_info = dict(density_log_gain_vs_top_pi=float(torch.logsumexp(joint, -1)[0] - torch.logsumexp(greedy_joint, -1)[0]),
                                        top_pi_component=greedy_component,
                                        mode_mean_distance=float((next_embedding[0] - mu[0, component]).norm()))
                elif mode == "greedy":
                    component = pi[0].argmax()
                    next_embedding = mu[0].index_select(0,component.reshape(1))
                else:
                    candidates, components = sample_gmm(mu, sigma, pi, generator=random,
                        mixture_temperature=mixture_temperature, noise_temperature=noise_temperature,
                        top_p=top_p, count=candidate_count)
                    next_embedding = candidates
                    component = components[0]
                curve = self.decode_tensor(next_embedding, mode=decoder_mode, density_iterations=density_iterations)
                returned = self.encode_tensor(curve) if feedback_mode=='curve' or candidate_count>1 else next_embedding
                if candidate_count > 1:
                    widths = sigma[0].index_select(0, components)
                    residual = ((returned.float()-candidates.float())/widths).square().mean(-1)
                    outside = ((curve < 0)|(curve > 1)).float().mean((1,2))
                    scores = pi[0].index_select(0, components).clamp(min=1e-12).log() - cycle_penalty*residual - outside_penalty*outside
                    winner = scores.masked_fill(~torch.isfinite(scores),-torch.inf).argmax().reshape(1)
                    next_embedding = candidates.index_select(0, winner)
                    curve, returned = curve.index_select(0, winner), returned.index_select(0, winner)
                    component = components.index_select(0, winner)[0]
                encoded = (returned if feedback_mode == 'curve' else next_embedding).unsqueeze(0)
                distances = (curve[0] - self.template_tensor).square().sum(-1).mean(-1).sqrt().amin(-1)
                component_index=torch.as_tensor(component,device=self.device).reshape(1)
                packet = torch.stack((distances[0], distances[1],
                    torch.isfinite(next_embedding).all(), torch.isfinite(curve).all(),
                    ((curve < 0)|(curve > 1)).any(), pi[0].index_select(0,component_index)[0],
                    sigma[0].index_select(0,component_index).mean(),component_index[0])).float().cpu().tolist()
                if not packet[2] or not packet[3]:
                    stop = 'nonfinite_embedding' if not packet[2] else 'nonfinite_coordinates'
                    break
                scores = dict(zip(('eop','eod'), packet[:2]))
                nearest = min(scores, key=scores.get)
                kind = nearest if scores[nearest] <= self.control_tolerance else 'body'
                page_id = cursor.page_id
                record = {"page_id": page_id, "kind": kind, "generated": True,
                          "points": None, "control_distances": scores,
                          "component": int(packet[7]), "pi": packet[5],
                          "sigma_mean": packet[6], "outside_page": bool(packet[4]), **density_info}
                if callback is not None:record['points'] = curve[0].cpu().tolist()
                else:pending_curves.append((record, curve[0]))
                records.append(record)
                cursor.append(kind)
                if callback is not None:
                    callback(record)
                if cursor.complete:
                    stop = "eod_eop"
                    break
                # 默认 D → E 回馈：让 R 看到实际绘出的曲线；不夹坐标或替换控制符。
                # 2026-10-09：latent 仅供接口漂移诊断，默认仍回馈实际绘出的曲线。
                embeddings = torch.cat((embeddings, encoded), 1)
                page_ids.append(page_id)
                ids = torch.tensor([page_ids], device=self.device)
                if cache.length == limit:
                    start = self._window_start(page_ids, limit)
                    if start is None:
                        stop = "context_full"
                        break
                    hidden = cache.forward(embeddings[:, start:], ids[:, start:], prefill=True)
                else:
                    hidden = cache.forward(encoded, ids[:, -1:])
        if pending_curves:
            for (record, _), points in zip(pending_curves, torch.stack([p for _, p in pending_curves]).cpu().tolist()):
                record['points'] = points
        generated = [r for r in records if r["generated"]]
        return {"strokes": records, "stop_reason": stop, "generated_strokes": len(generated),
                "generated_pages": len(set(r["page_id"] for r in generated)),
                "outside_page_strokes": sum(r["outside_page"] for r in generated),
                "control_events": [{"index": i, "page_id": r["page_id"], "kind": r["kind"],
                                    "distances": r["control_distances"]}
                                   for i, r in enumerate(records) if r["generated"] and r["kind"] != "body"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "mode": mode, "temperature": temperature, "seed": seed,
                'mixture_temperature':mixture_temperature, 'noise_temperature':noise_temperature,
                'top_p':top_p,'candidate_count':candidate_count,'cycle_penalty':cycle_penalty,'outside_penalty':outside_penalty,
                "decoder_mode": decoder_mode, "density_iterations": density_iterations if mode == "density" or decoder_mode == "density" else None,
                "control_tolerance": self.control_tolerance,
                "feedback": "decoded_curve_to_encoder" if feedback_mode == "curve" else "predicted_latent",
                "precision": "fp16 + fp32 GMM" if self.device.type == "cuda" else "fp32",
                "context_limit": limit, "max_new_strokes": max_new_strokes,
                "max_new_pages": max_new_pages}
