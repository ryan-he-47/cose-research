"""2026-11-02：固定槽位、页末 EOP 填充及完整批损失分母。"""
from __future__ import annotations

from collections import Counter
import math
import numpy as np
import torch

from .formal_recipe import TASKS
from .pipeline_pretrain import collate_documents, KIND_IDS


def document_layout(document, block=16, *, prefix_cut=0):
    seq = document.to_sequence()
    local, previous, position = [], None, 0
    prefix_ids = document.metadata.get("_prefix_local_ids", [])
    if document.prefix and len(prefix_ids) != len(document.prefix):
        raise ValueError("窗口前缀缺少原始页内序号")
    for index, element in enumerate(seq):
        if index < len(document.prefix):
            position = int(prefix_ids[index])
        elif element.page_id != previous:
            position = 0
        local.append(position)
        previous, position = element.page_id, position+1
    blocks = []
    i = 1
    while i < len(seq):
        if not seq[i].targetable:
            i += 1
            continue
        start, page = i, seq[i].page_id
        while i < len(seq) and seq[i].page_id == page and seq[i].targetable:
            i += 1
        end = i
        # 长页随机提供至多15笔真实前缀，覆盖页中续写；短答案完整监督。
        offset = min(int(prefix_cut), block-1) if end-start > 2*block else 0
        start += offset
        for begin in range(start, end, block):
            real = list(range(begin, min(begin+block, end)))
            if len(real) < block and seq[real[-1]].kind != "eop":
                raise ValueError("短块仅允许在真实 EOP 后填充")
            blocks.append(dict(start=begin, page=page, local=local[begin],
                               target=real+[-1]*(block-len(real))))
    return dict(local=local, blocks=blocks)


def prepare_update(documents, generator, block=16, *, supervision_block=None):
    """一次固定完整批的切块、噪声、t；微批重排/OOM不得重抽。"""
    base = block if supervision_block is None else supervision_block
    if block < 1 or base % block or base < block:
        raise ValueError("去噪块长必须整除监督块长")
    counts = dict(body=Counter(), control=0, padded=0, blocks=0)
    # 新阶段先固定整批监督起点，后续各宽度的噪声数量不会改变本批其他文档的真前缀。
    cuts = torch.randint(0, base, (len(documents),),generator=generator).tolist() if supervision_block is not None else None
    for row,document in enumerate(documents):
        cut = cuts[row] if cuts is not None else int(torch.randint(0, base, (), generator=generator))
        layout = document_layout(document, base, prefix_cut=cut)
        if block != base:
            pieces = []
            for part in layout["blocks"]:
                real = len(part["target"])-part["target"].count(-1)
                for offset in range(0, real, block):
                    pieces.append(dict(start=part["start"]+offset, page=part["page"],
                                       local=part["local"]+offset, target=part["target"][offset:offset+block]))
            layout["blocks"] = pieces
        layout["width"] = block
        n = len(layout["blocks"])
        layout["noise"] = torch.randn(n, block, 48, generator=generator)
        layout["times"] = torch.rand(n, generator=generator)
        document.metadata["_page_diffusion"] = layout
        seq, task = document.to_sequence(), document.metadata["task"]
        for part in layout["blocks"]:
            counts["blocks"] += 1
            for index in part["target"]:
                if index < 0:
                    counts["padded"] += 1
                elif seq[index].kind in ("eop", "eod"):
                    counts["control"] += 1
                elif seq[index].kind != "bos":
                    counts["body"][task] += 1
    return counts


def pack_documents(documents, recipe):
    batch = collate_documents(documents, max_strokes=recipe["context"], max_points=76,
                              encoder_length_buckets=recipe["encoder_length_buckets"])
    block = documents[0].metadata["_page_diffusion"].get("width", recipe["model"]["block"])
    b, clean_size = batch["stroke_pad"].shape
    layouts = [d.metadata["_page_diffusion"] for d in documents]
    noisy_size = max(len(x["blocks"])*block for x in layouts)
    total = clean_size+noisy_size
    pages = np.zeros((b, total), np.int64)
    local = np.zeros((b, total), np.int64)
    roles = np.ones((b, total), np.int64)
    roles[:, :clean_size] = 0
    split = recipe.get("attention_layout", "dense") == "split"
    allowed = np.zeros((b, 1, noisy_size if split else total, total), bool)
    targets = np.zeros((b, noisy_size), np.int64)
    weight = np.zeros((b, noisy_size), np.float32)
    real_body = np.zeros((b, noisy_size), bool)
    real_control = np.zeros_like(real_body)
    synthetic = np.zeros_like(real_body)
    times = torch.zeros(b, total)
    noise = torch.zeros(b, noisy_size, 48)
    for row, (document, layout) in enumerate(zip(documents, layouts)):
        seq = document.to_sequence()
        n = len(seq)
        pages[row, :n] = [e.page_id for e in seq]
        local[row, :n] = layout["local"]
        if not split:
            allowed[row, 0, :n, :n] = np.tri(n, dtype=bool)
        # 无效query只看自身，避免全遮罩行；有效query永远看不到这些PAD。
        if split:
            diagonal = np.arange(noisy_size)
            allowed[row, 0, diagonal, clean_size+diagonal] = True
        else:
            np.fill_diagonal(allowed[row, 0], True)
        for part_id, part in enumerate(layout["blocks"]):
            lo, hi = part_id*block, (part_id+1)*block
            qlo, qhi = clean_size+lo, clean_size+hi
            pages[row, qlo:qhi] = part["page"]
            local[row, qlo:qhi] = np.arange(part["local"], part["local"]+block)
            query_lo, query_hi = (lo, hi) if split else (qlo, qhi)
            allowed[row, 0, query_lo:query_hi, :part["start"]] = True
            allowed[row, 0, query_lo:query_hi, qlo:qhi] = True
            times[row, qlo:qhi] = layout["times"][part_id]
            noise[row, lo:hi] = layout["noise"][part_id]
            padding = part["target"].count(-1)
            for col, index in enumerate(part["target"], lo):
                if index < 0:
                    synthetic[row, col] = True
                    weight[row, col] = recipe["padding_group_weight"]/padding
                    continue
                targets[row, col] = index
                kind = seq[index].kind
                if kind in ("eop", "eod"):
                    real_control[row, col] = True
                    weight[row, col] = 1.
                elif kind != "bos":
                    real_body[row, col] = True
    arrays = dict(pages=pages, local=local, roles=roles, allowed=allowed,
                  targets=targets, control_weight=weight, real_body=real_body,
                  real_control=real_control, synthetic=synthetic)
    batch.update({k: torch.from_numpy(v) for k, v in arrays.items()})
    batch.update(noise=noise, times=times, clean_size=clean_size,
                 task_id=torch.tensor([TASKS.index(d.metadata["task"]) for d in documents]))
    return batch


def diffusion_groups(documents, recipe):
    """只重排执行；文档、监督、切分、t与噪声在进入这里之前已经固定。"""
    block = documents[0].metadata["_page_diffusion"].get("width", recipe["model"]["block"])
    if any(d.metadata["_page_diffusion"].get("width", recipe["model"]["block"]) != block for d in documents):
        raise ValueError("一次更新应使用同一原生块长")
    noisy_length = lambda d: len(d.metadata["_page_diffusion"]["blocks"])*block
    if recipe.get("packing", "legacy") == "cost":
        ordered = sorted(documents, key=lambda d: (
            math.floor(math.log2(max(noisy_length(d), 1)/max(d.stroke_count, 1))*2),
            d.stroke_count+noisy_length(d)))
        lengths = np.array([(d.stroke_count, noisy_length(d)) for d in ordered], dtype=np.int64)
        size = len(ordered)
        best, parent = np.full(size+1, np.inf), np.zeros(size+1, np.int64)
        best[0] = 0.
        width, ffn = recipe["model"]["width"], recipe["model"]["ffn"]
        attention_weight = 2*width/(4*width*width+3*width*ffn)
        # 更大的容量档同时允许少拆微批，最终由实测吞吐选择；不能只追求最低补齐量。
        overhead = float(recipe.get("microbatch_overhead_positions", max(2048., recipe["padded_stroke_budget"]/4)))
        for end in range(1, size+1):
            limit = min(end, recipe["microbatch_documents"])
            sizes = np.arange(1, limit+1)
            clean, noisy = np.maximum.accumulate(lengths[end-limit:end][::-1], axis=0).T
            positions = sizes*(clean+noisy)
            pairs = sizes*(clean*clean*.5+noisy*(clean+noisy))
            cost = best[end-sizes]+overhead+positions+attention_weight*pairs
            cost[(positions > recipe["padded_stroke_budget"]) & (sizes > 1)] = np.inf
            selected = int(cost.argmin())
            best[end], parent[end] = cost[selected], end-sizes[selected]
        spans, end = [], size
        while end:
            begin = int(parent[end])
            spans.append((begin, end))
            end = begin
        for begin, end in reversed(spans):
            yield ordered[begin:end]
        return
    # 切分与噪声已经固定；这里只重排计算，分桶兼顾干净/带噪两种长度。
    ordered = sorted(documents, key=lambda d: (d.stroke_count//64, noisy_length(d)//64, d.stroke_count))
    chunk, largest_clean, largest_noisy = [], 0, 0
    for doc in ordered:
        clean, noisy = doc.stroke_count, noisy_length(doc)
        projected = (max(largest_clean, clean)+max(largest_noisy, noisy))*(len(chunk)+1)
        if chunk and (len(chunk) >= recipe["microbatch_documents"] or projected > recipe["padded_stroke_budget"]):
            yield chunk
            chunk, largest_clean, largest_noisy = [], 0, 0
        chunk.append(doc)
        largest_clean, largest_noisy = max(largest_clean, clean), max(largest_noisy, noisy)
    if chunk:
        yield chunk


def iter_diffusion_batches(documents, recipe):
    for chunk in diffusion_groups(documents, recipe):
        yield pack_documents(chunk, recipe)


def diffusion_inputs(model, encoded, batch):
    clean = model.normalize(encoded)
    indices = batch["targets"][..., None].expand(-1, -1, clean.shape[-1])
    target = clean.gather(1, indices)
    eop = model.normalize(model.eop_latent)
    target = torch.where(batch["synthetic"][..., None], eop, target)
    t = batch["times"][:, batch["clean_size"]:, None]
    noisy = (1-t)*target + t*batch["noise"]
    return torch.cat((clean, noisy), 1), target


def losses(model, output, target, batch, counts, recipe):
    start = batch["clean_size"]
    hidden = output["hidden"]
    prediction = output["prediction"][:, start:].float()
    residual = prediction-target.float()
    error = residual.square().mean(-1)
    times = batch["times"][:, start:]
    policy = recipe.get("balance")
    if policy is not None:
        # U来自同字形的受控平移。真实文字正文只增补正交余量，停止符始终用原欧氏误差。
        with torch.autocast(device_type=prediction.device.type,enabled=False):
            position = (residual @ model.balance_axes).square().sum(-1)/residual.shape[-1]
        tail = (error-position).clamp_min(0.)
        ramp = ((policy["t_zero"]-times)/(policy["t_zero"]-policy["t_full"])).clamp(0.,1.)
        text = batch["kind_id"].gather(1,batch["targets"]) == KIND_IDS["text"]
        shape_error = error+policy["tail_extra"]*ramp*tail*text
    zero = prediction.sum()*0
    body, control = zero, zero
    metrics = {}
    for task_id, task in enumerate(TASKS):
        selected = batch["real_body"] & (batch["task_id"] == task_id)[:, None]
        value = (error*selected).sum()
        metrics["body_sum:"+task] = value.detach()
        if policy is not None:
            if task in policy["shape_tasks"]:
                value = (shape_error*selected).sum()
            metrics["metric_sum:"+task] = value.detach()
        if counts["body"].get(task, 0):
            body = body + recipe["loss_weights"][task]*value/counts["body"][task]
    if counts["control"]:
        control = (error*batch["control_weight"]).sum()/counts["control"]
    if policy is not None:
        metrics["weighted_control"] = (recipe["control_loss_weight"]*control).detach()
    for name in ("real_body", "real_control", "synthetic"):
        metrics[name+"_sum"] = (error*batch[name]).sum().detach()
    for lo, hi in ((0., .25), (.25, .5), (.5, .75), (.75, 1.)):
        selected = batch["real_body"] & (times >= lo) & (times < hi)
        metrics[f"noise_{lo}_sum"] = (error*selected).sum().detach()
        metrics[f"noise_{lo}_count"] = selected.sum().detach()
    total = body + recipe["control_loss_weight"]*control
    return dict(total=total, body=body, control=control,
                hidden=hidden, metrics=metrics)
