"""按检查点的原生块长在页内去噪，遇到首个EOP即结束本页。"""
from __future__ import annotations

import torch

from .generation_pretrain import PageGenerator
from .page_diffusion_runtime import autocast


@torch.no_grad()
def clean_cache(model, x, pages, local, past=None):
    n = x.shape[1]
    previous = 0 if past is None else past[0][0].shape[-2]
    allowed = torch.arange(previous+n, device=x.device)[None, :] <= previous+torch.arange(n, device=x.device)[:, None]
    roles, times = torch.zeros_like(pages), torch.zeros_like(pages, dtype=torch.float32)
    result = model(x, pages, local, roles, times, allowed[None, None], past=past, return_cache=True)
    return result["cache"]


@torch.no_grad()
def sample_block(model, cache, page, start_local, noise, steps=32):
    supported = getattr(model.cfg, "training_blocks", ()) or (model.cfg.block,)
    if (steps < 1 or noise.ndim != 3 or noise.shape[0] != 1 or noise.shape[1] not in supported
            or noise.shape[2] != model.cfg.latent):
        raise ValueError("无效去噪步数或初始噪声形状")
    width = noise.shape[1]
    device = noise.device
    pages = torch.full((1, width), page, dtype=torch.long, device=device)
    local = torch.arange(start_local, start_local+width, device=device)[None]
    roles = torch.ones_like(pages)
    n_history = 0 if cache is None else cache[0][0].shape[-2]
    allowed = torch.ones((1, 1, width, n_history+width), dtype=torch.bool, device=device)
    value = noise.float()
    for index in range(steps, 0, -1):
        t, next_t = index/steps, (index-1)/steps
        times = torch.full_like(pages, t, dtype=torch.float32)
        predicted = model(value, pages, local, roles, times, allowed, past=cache)["prediction"].float()
        value = (next_t/t)*value+(1-next_t/t)*predicted
    return value, pages, local


class PageDiffusionGenerator:
    def __init__(self, model, codec):
        self.model, self.codec = model.eval(), codec.eval()
        self.device = next(model.parameters()).device
        self.geometry = PageGenerator(codec)

    @torch.no_grad()
    def generate(self, prefix, *, seed=42, steps=32, max_new_strokes=128, max_new_pages=4,
                 prefix_local_ids=None, block_width=None):
        if not prefix or max_new_strokes < 1 or max_new_pages < 1:
            raise ValueError("无效前缀或生成上限")
        width = self.model.cfg.block if block_width is None else block_width
        if width not in (getattr(self.model.cfg, "training_blocks", ()) or (self.model.cfg.block,)):
            raise ValueError("所选推理宽度超出检查点训练过的块长")
        history_pages, local_ids = [], []
        previous, position = None, 0
        for element in prefix:
            if previous != element.page_id:
                position = 0
            history_pages.append(element.page_id)
            local_ids.append(position)
            previous, position = element.page_id, position+1
        if prefix_local_ids is not None:
            if len(prefix_local_ids) != len(prefix):
                raise ValueError("前缀局部位置长度不匹配")
            local_ids = list(prefix_local_ids)
        history = self.model.normalize(self.geometry.encode([e.rawxy for e in prefix]))[None]
        pages = torch.tensor([history_pages], device=self.device)
        local = torch.tensor([local_ids], device=self.device)
        if len(prefix) > self.model.cfg.context:
            begin = self._window_start(pages[0].tolist())
            history, pages, local = history[:, begin:], pages[:, begin:], local[:, begin:]
        with autocast(self.device):
            cache = clean_cache(self.model, history, pages, local)
        current_page = history_pages[-1]+int(prefix[-1].kind == "eop")
        next_local = 0 if prefix[-1].kind == "eop" else local_ids[-1]+1
        first_page = current_page
        pending_eod = prefix[-1].kind == "eod"
        if len(prefix) > 1 and prefix[-1].kind == "eop" and prefix[-2].kind == "eod":
            raise ValueError("输入已是结束文档")
        rng = torch.Generator(device="cpu").manual_seed(seed)
        records, blocks, stop = [], [], "max_new_strokes"
        while len(records) < max_new_strokes:
            noise = torch.randn(1, width, self.model.cfg.latent, generator=rng).to(self.device)
            with autocast(self.device):
                sampled, block_pages, block_local = sample_block(self.model, cache, current_page, next_local, noise, steps)
                latents = self.model.denormalize(sampled[0])
                curves = self.geometry.decode(latents)
            kinds, scores = zip(*(self.geometry.recognize(curve) for curve in curves))
            take, closed, complete = 0, False, False
            block_record = dict(page=current_page, first_local=next_local, kinds=list(kinds),
                                latents=latents.cpu().tolist())
            for index, (kind, curve, score) in enumerate(zip(kinds, curves, scores)):
                if len(records) >= max_new_strokes:
                    break
                records.append(dict(kind=kind, page_id=current_page, local_id=next_local+index,
                                    points=curve.tolist(), scores=score))
                take += 1
                if kind == "eop":
                    closed, complete = True, pending_eod
                    pending_eod = False
                    break
                pending_eod = kind == "eod"
            block_record.update(retained=take,
                discarded_non_eop=sum(k != "eop" for k in kinds[take:]) if closed else 0,
                eop_found=closed)
            blocks.append(block_record)
            kept = sampled[:, :take]
            new_pages, new_local = block_pages[:, :take], block_local[:, :take]
            history = torch.cat((history, kept), 1)
            pages, local = torch.cat((pages, new_pages), 1), torch.cat((local, new_local), 1)
            if complete:
                stop = "eod_eop"
                break
            if len(records) >= max_new_strokes:
                break
            if closed:
                current_page += 1
                next_local = 0
                if current_page-first_page >= max_new_pages:
                    stop = "max_new_pages"
                    break
            else:
                next_local += take
            with autocast(self.device):
                if history.shape[1] > self.model.cfg.context:
                    try:
                        begin = self._window_start(pages[0].tolist())
                    except ValueError:
                        stop = "context_page_overflow"
                        break
                    history, pages, local = history[:, begin:], pages[:, begin:], local[:, begin:]
                    cache = clean_cache(self.model, history, pages, local)
                else:
                    cache = clean_cache(self.model, kept, new_pages, new_local, cache)
        return dict(seed=seed, steps=steps, block_width=width, stop=stop, records=records, blocks=blocks,
                    generated_strokes=len(records), generated_pages=len({r['page_id'] for r in records}))

    def _window_start(self, page_ids):
        for index in range(1, len(page_ids)):
            if page_ids[index] != page_ids[index-1] and len(page_ids)-index+1 <= self.model.cfg.context:
                return index-1
        raise ValueError("单页超过上下文上限")
