"""冻结 AE 的页协议 R 训练入口（2026-10-02）。

默认 steps=0 只显示配置；--check 使用小 R 做 CPU 通路检查。正式单卡训练
为 GPT-2 small 尺寸、1024 上下文、FP16 + GradScaler / FP32 GMM。
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import inspect
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.st.pipeline_pretrain import build_document_dataset, collate_documents, to_device
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.pretrain_model import FrozenSketchLM, config_from_ae_checkpoint, config_from_r_checkpoint
from src.st.run_budget import RunBudget


class DocumentStream:
    """按有效目标笔画混合任务；保存每个任务的消费位置和文本窗口位置。"""

    def __init__(self, dataset, max_strokes=1024):
        self.dataset, self.max_strokes = dataset, max_strokes
        tasks = tuple(dataset.tasks)
        candidates = {"text": tuple(t for t in tasks if t == "T1"),
                      "object": tuple(t for t in tasks if t.startswith("O")),
                      "scene": tuple(t for t in tasks if t.startswith("S"))}
        self.groups = {name: values for name, values in candidates.items() if values}
        if not self.groups:
            raise ValueError("没有可用预训练任务")
        shares = {"text": 0.4, "object": 0.4, "scene": 0.2} if "scene" in self.groups else {
            "text": 0.5, "object": 0.5}
        denominator = sum(shares[name] for name in self.groups)
        self.shares = {name: shares[name] / denominator for name in self.groups}
        self.task_indices = {task: 0 for task in tasks}
        self.task_targets = {task: 0 for task in tasks}
        self.group_targets = {name: 0 for name in self.groups}
        self.group_cursor = {name: 0 for name in self.groups}
        self.pending = {}
        self.source_index = 0
        self.skipped = Counter()
        self.skipped_examples = []

    def _windows(self, document):
        if document.metadata["task"] == "T1":
            return list(document.iter_windows(self.max_strokes))
        if document.stroke_count > self.max_strokes:
            raise ValueError(f"条件任务整文 {document.stroke_count} 笔超过 {self.max_strokes}")
        return [document]

    def __next__(self):
        for _ in range(1000):
            group = min(self.groups, key=lambda name: self.group_targets[name] / self.shares[name])
            tasks = self.groups[group]
            cursor = self.group_cursor[group]
            rotated = tasks[cursor % len(tasks):] + tasks[:cursor % len(tasks)]
            task = min(rotated, key=lambda name: self.task_targets[name])
            self.group_cursor[group] = tasks.index(task) + 1
            if task not in self.pending:
                index = self.task_indices[task]
                self.task_indices[task] += 1
                self.source_index += 1
                document = self.dataset.get_for_task(task, index)
                try:
                    windows = self._windows(document)
                except ValueError as error:
                    self.skipped[task + ":too_long"] += 1
                    self.skipped_examples.append({"task": task, "index": index,
                        "strokes": document.stroke_count, "reason": str(error)})
                    self.skipped_examples = self.skipped_examples[-20:]
                    continue
                self.pending[task] = {"index": index, "offset": 0, "windows": windows}
            item = self.pending[task]
            document = item["windows"][item["offset"]]
            item["offset"] += 1
            if item["offset"] == len(item["windows"]):
                del self.pending[task]
            sequence = document.to_sequence()
            count = sum(element.targetable for element in sequence[1:])
            if not count:
                self.skipped[task + ":no_target"] += 1
                continue
            self.group_targets[group] += count
            self.task_targets[task] += count
            return document
        raise RuntimeError("连续 1000 个文档没有可用目标；检查条件任务长度和数据源")

    def state_dict(self):
        return {"source_index": self.source_index, "task_indices": dict(self.task_indices),
                "task_targets": dict(self.task_targets),
                "group_targets": dict(self.group_targets), "group_cursor": dict(self.group_cursor),
                "shares": self.shares, "max_strokes": self.max_strokes,
                "pending": {task: {"index": item["index"], "offset": item["offset"]}
                            for task, item in self.pending.items()},
                "skipped": dict(self.skipped), "skipped_examples": self.skipped_examples}

    def load_state_dict(self, state):
        if state["shares"] != self.shares or state["max_strokes"] != self.max_strokes:
            raise ValueError("恢复时任务份额或上下文不同")
        if set(state["task_indices"]) != set(self.task_indices):
            raise ValueError("恢复时任务清单不同")
        self.source_index = state["source_index"]
        self.task_indices = dict(state["task_indices"])
        self.task_targets = dict(state["task_targets"])
        self.group_targets = dict(state["group_targets"])
        self.group_cursor = dict(state["group_cursor"])
        self.skipped = Counter(state["skipped"])
        self.skipped_examples = list(state["skipped_examples"])
        self.pending = {task: {**item, "windows": self._windows(
            self.dataset.get_for_task(task, item["index"]))} for task, item in state["pending"].items()}


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def collect_update_batches(stream, train_args, max_strokes):
    """2026-10-03：先取本次更新的文档，再按长度组批；页内/笔画顺序不动。"""
    documents, count, task_targets = [], 0, Counter()
    grouped = train_args.get("batch_by_length", False)
    micro = train_args["microbatch_documents"]
    while count < train_args["target_budget"]:
        for _ in range(1 if grouped else micro):
            document = next(stream)
            targets = sum(e.targetable for e in document.to_sequence()[1:])
            documents.append(document)
            count += targets
            task_targets[document.metadata["task"]] += targets
    if grouped:
        documents.sort(key=lambda d: d.stroke_count)
    batches, chunk = [], []
    padded_budget = train_args.get("padded_stroke_budget", 8192)
    for document in documents:
        next_length = max([document.stroke_count] + [d.stroke_count for d in chunk])
        if chunk and (len(chunk) == micro or grouped and next_length * (len(chunk) + 1) > padded_budget):
            batches.append(collate_documents(chunk, max_strokes=max_strokes,
                                             max_points=train_args["max_points"]))
            chunk = []
        chunk.append(document)
    if chunk:
        batches.append(collate_documents(chunk, max_strokes=max_strokes,
                                         max_points=train_args["max_points"]))
    return batches, count, task_targets


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _optimizer(model, train_args):
    optimizer = torch.optim.Adam(model.r_pred.parameters(), lr=train_args["lr"])
    warmup = max(1, train_args["warmup"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min((step + 1) / warmup, (warmup / (step + 1)) ** 0.5))
    return optimizer, scheduler


def save_checkpoint(path, model, optimizer, scheduler, scaler, *, step, stream,
                    counters, data_args, train_args, ae_source, data_manifest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"stage": "r_pretrain", "format_version": 1,
             "model": model.state_dict(), "cfg": asdict(model.cfg), "step": step,
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict() if scaler is not None else None,
             "rng": _rng_state(), "data": stream.state_dict(), "counters": counters,
             "data_args": data_args, "train_args": train_args, "ae_source": ae_source,
             "data_manifest": data_manifest}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore_checkpoint(state, model, optimizer, scheduler, scaler, stream):
    if state.get("stage") != "r_pretrain":
        raise ValueError("--resume 需要本入口的完整 R checkpoint；AE 来源用 --init_ae")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state["scaler"]:
        scaler.load_state_dict(state["scaler"])
    stream.load_state_dict(state["data"])
    _restore_rng(state["rng"])
    return int(state["step"]), Counter(state["counters"])


def cpu_check(model, dataset, stream, output, data_args, train_args, ae_source):
    from src.page_document import build_document, layout_text_pages
    # AE 规格和权重保持真实来源；仅 R 缩小，正文用短句检查右移监督/右 padding。
    documents = [build_document(layout_text_pages(text, dataset.font), metadata={"task": "T1"})
                 for text in ("Hi.", "A little cat.")]
    batch = collate_documents(documents)
    model.train()
    result = model(batch)
    if not bool(torch.isfinite(result["loss"])):
        raise RuntimeError("CPU R loss 非有限")
    result["loss"].backward()
    assert all(not module.training for module in (model.encoder, model.decoder, model.dct_head)
               if module is not None)
    assert all(parameter.grad is None and not parameter.requires_grad for module in
               (model.encoder, model.decoder, model.dct_head) if module is not None
               for parameter in module.parameters())
    grads = [parameter.grad for parameter in model.r_pred.parameters() if parameter.grad is not None]
    assert grads and all(bool(torch.isfinite(grad).all()) for grad in grads)
    optimizer, scheduler = _optimizer(model, train_args)
    optimizer.step()
    scheduler.step()
    model.eval()
    with torch.no_grad():
        fake = torch.randn(1, 7, model.cfg.latent_units)
        sid = torch.tensor([[0, 0, 0, 1, 1, 1, 2]])
        pad = torch.zeros((1, 7), dtype=torch.bool)
        left = model.hidden_states(fake, sid, pad)
        changed = fake.clone()
        changed[:, 4:] += 5
        right = model.hidden_states(changed, sid, pad)
        assert torch.allclose(left[:, :4], right[:, :4], atol=1e-6, rtol=1e-6)
        mu, sigma, pi = model.next_distribution(batch)
        assert mu.shape == (2, model.cfg.gmm_r, model.cfg.latent_units)
        assert sigma.dtype == pi.dtype == mu.dtype == torch.float32
        singles = [model(collate_documents([document])) for document in documents]
        combined = model(batch)
        assert torch.allclose(sum(value["pred_sum"] for value in singles), combined["pred_sum"],
                              atol=1e-4, rtol=1e-5)
        assert sum(value["pred_count"] for value in singles) == combined["pred_count"]
    next(stream)  # 建立真实 provider 游标；不对这一长文做 CPU 网络计算。
    checkpoint_path = output / "check.pt"
    counters = Counter(read_targets=int(result["pred_count"]), successful_targets=int(result["pred_count"]))
    save_checkpoint(checkpoint_path, model, optimizer, scheduler, None, step=1, stream=stream,
        counters=counters, data_args=data_args, train_args=train_args, ae_source=ae_source,
        data_manifest=dataset.manifest())
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = FrozenSketchLM(config_from_r_checkpoint(saved),
        encode_chunk_size=train_args["encode_chunk_size"], activation_checkpoint=train_args["activation_checkpoint"])
    restored_optimizer, restored_scheduler = _optimizer(restored, train_args)
    restored_stream = DocumentStream(dataset, model.cfg.max_context_strokes)
    restored_step, _ = restore_checkpoint(saved, restored, restored_optimizer,
                                           restored_scheduler, None, restored_stream)
    assert restored_step == 1
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())
    expected, replayed = next(stream), next(restored_stream)
    expected_batch, replayed_batch = collate_documents([expected]), collate_documents([replayed])
    assert all(torch.equal(value, replayed_batch[name]) for name, value in expected_batch.items()
               if torch.is_tensor(value))
    print(json.dumps({"check": "ok", "small_r": model.cfg.r_d_model == 48,
        "pred_nll": float(result["pred_nll"].detach()),
        "targets": int(result["pred_count"]), "r_parameters": sum(p.numel() for p in model.r_pred.parameters()),
        "shares": stream.shares, "source_index": saved["data"]["source_index"],
        "checkpoint": str(checkpoint_path.resolve())}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init_ae", help="已有旧 AE 或新独立 AE checkpoint；完整恢复时无需重复提供")
    parser.add_argument("--resume", help="完整 R 训练状态")
    parser.add_argument("--steps", type=int, default=0, help="目标成功 optimizer 总步数")
    parser.add_argument("--check", action="store_true", help="小 R CPU 通路检查")
    parser.add_argument("--font", default="datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg")
    parser.add_argument("--raw_dir", default="datasets/quick_draw")
    parser.add_argument("--stories_path")
    parser.add_argument("--text_file")
    parser.add_argument("--max_stories", type=int, default=1024)
    parser.add_argument("--max_per_class", type=int, default=256)
    parser.add_argument("--no_text_augmentation", action="store_true")
    parser.add_argument("--manifest", help="build_document_dataset 数据参数 JSON")
    parser.add_argument("--microbatch_documents", type=int, default=1)
    parser.add_argument("--batch_by_length", action="store_true",
                        help="本次更新内按文档长度组批，保留各文档笔顺")
    parser.add_argument("--padded_stroke_budget", type=int, default=8192,
                        help="长度组批时每个微批 B×S 的上限")
    parser.add_argument("--resume_microbatch_documents", type=int,
                        help="恢复时明确改微批大小；优化器、有效目标预算与数据游标照常恢复")
    parser.add_argument("--target_budget", type=int, default=8192, help="每次更新的有效目标笔画预算，可略超出")
    parser.add_argument("--max_points", type=int, default=76)
    parser.add_argument("--encode_chunk_size", type=int, default=256)
    parser.add_argument("--no_activation_checkpoint", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--max_consecutive_skips", type=int, default=20)
    parser.add_argument("--output_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--max_seconds", type=float, default=0,
                        help="本次训练循环时间上限；0 不限时，更新边界保存 final.pt")
    args = parser.parse_args()
    if args.steps < 0 or args.lr <= 0 or args.clip_norm <= 0:
        parser.error("steps >= 0，lr/clip_norm > 0")
    if any(getattr(args, key) < 1 for key in ("microbatch_documents", "target_budget",
            "encode_chunk_size", "save_every", "log_every", "max_consecutive_skips", "cpu_threads",
            "padded_stroke_budget")):
        parser.error("批大小、预算、日志/保存间隔与线程数必须为正数")
    if args.resume_microbatch_documents is not None and (not args.resume or args.resume_microbatch_documents < 1):
        parser.error("resume_microbatch_documents 需要 --resume 且为正数")
    if args.max_points < 2 or args.max_stories < 0 or args.max_per_class < 0:
        parser.error("max_points >= 2，数据数量不能为负")
    if not np.isfinite(args.max_seconds) or args.max_seconds < 0:
        parser.error("max_seconds 必须为有限非负数")
    if args.check and args.steps:
        parser.error("--check 与 --steps 分开运行")
    if (args.check or args.steps) and not (args.init_ae or args.resume):
        parser.error("新训练需 --init_ae；续跑需 --resume")
    return args


def main():
    args = parse_args()
    if not args.check and args.steps == 0:
        cfg = gpt2_small_pretrain_config()
        with torch.device("meta"):
            model = FrozenSketchLM(cfg)
        print(json.dumps({"training_started": False, "cfg": asdict(cfg), "init_ae": args.init_ae,
            "r_parameters": sum(parameter.numel() for parameter in model.r_pred.parameters()),
            "target_budget": args.target_budget, "microbatch_documents": args.microbatch_documents},
            ensure_ascii=False, indent=2))
        return
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.check else args.device)
    if not args.check and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("正式 R 训练使用 CUDA FP16；CPU 检查使用 --check")
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume is not None and resume.get("stage") != "r_pretrain":
        raise ValueError("--resume 需要本入口的完整 R checkpoint")
    initial = torch.load(args.init_ae, map_location="cpu", weights_only=False) if resume is None else None
    cfg = config_from_r_checkpoint(resume) if resume else config_from_ae_checkpoint(initial)
    if args.check and resume is None:
        cfg = replace(cfg, r_layers=2, r_d_model=48, r_heads=4, r_dff=96)
    data_args = {"font_path": args.font, "raw_dir": args.raw_dir,
        "stories_path": args.stories_path, "text_file": args.text_file,
        "max_stories": min(args.max_stories, 8) if args.check else args.max_stories,
        "max_per_class": min(args.max_per_class, 1) if args.check else args.max_per_class,
        "seed": args.seed, "augment_text": not args.no_text_augmentation}
    if args.manifest:
        supplied = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        unknown = set(supplied) - set(inspect.signature(build_document_dataset).parameters)
        if unknown:
            raise ValueError(f"manifest 不认识的参数：{sorted(unknown)}")
        data_args.update(supplied)
    train_args = {key: getattr(args, key) for key in ("microbatch_documents", "target_budget", "max_points",
        "encode_chunk_size", "lr", "warmup", "clip_norm")}
    train_args["activation_checkpoint"] = not args.no_activation_checkpoint
    train_args.update(batch_by_length=args.batch_by_length,
                      padded_stroke_budget=args.padded_stroke_budget)
    ae_source = {"path": str(Path(args.init_ae).resolve()), "stage": initial.get("stage", "legacy"),
                 "step": initial.get("step"), "cfg": initial.get("cfg")} if initial else resume["ae_source"]
    if resume:
        data_args, train_args = resume["data_args"], dict(resume["train_args"])
        if args.resume_microbatch_documents is not None:
            train_args["microbatch_documents"] = args.resume_microbatch_documents
        if args.batch_by_length:
            train_args.update(batch_by_length=True, padded_stroke_budget=args.padded_stroke_budget)
    dataset = build_document_dataset(**data_args)
    model = FrozenSketchLM(cfg, encode_chunk_size=train_args["encode_chunk_size"],
                           activation_checkpoint=train_args["activation_checkpoint"]).to(device)
    if initial is not None:
        model.load_ae(initial)
        del initial
    stream = DocumentStream(dataset, cfg.max_context_strokes)
    output = Path(args.output_dir or time.strftime("tmp_outputs/r_pretrain_%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    if args.check:
        if resume is not None:
            # 2026-10-03：检查恢复权重时保留真实 R 尺寸，并确实加载其中的冻结 AE。
            model.load_state_dict(resume["model"], strict=True)
        cpu_check(model, dataset, stream, output, data_args, train_args, ae_source)
        return
    optimizer, scheduler = _optimizer(model, train_args)
    scaler = torch.amp.GradScaler("cuda", init_scale=256)
    step, counters = restore_checkpoint(resume, model, optimizer, scheduler, scaler, stream) if resume else (0, Counter())
    manifest = dataset.manifest()
    if resume and resume["data_manifest"] != manifest:
        raise ValueError("恢复时数据 manifest/hash 不同")
    print(json.dumps({"r_parameters": sum(p.numel() for p in model.r_pred.parameters()),
        "context": cfg.max_context_strokes, "precision": "fp16 + FP32 head", "shares": stream.shares,
        "step": step, "train_args": train_args, "ae_source": ae_source["path"]}, ensure_ascii=False), flush=True)
    model.train()
    started, consecutive_skips = time.monotonic(), 0
    budget = RunBudget(args.max_seconds)
    while step < args.steps and budget.reason is None:
        update_started = time.monotonic()
        cpu_batches, target_count, task_targets = collect_update_batches(stream, train_args,
                                                                         cfg.max_context_strokes)
        prepared = time.monotonic()
        counters["attempts"] += 1
        counters["read_targets"] += target_count
        optimizer.zero_grad(set_to_none=True)
        total_loss, finite_loss = 0.0, True
        metric_sums, metric_counts = Counter(), Counter()
        for cpu_batch in cpu_batches:
            batch = to_device(cpu_batch, device)
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(batch)
            finite_loss &= bool(torch.isfinite(result["pred_sum"]))
            scaler.scale(result["pred_sum"] / target_count).backward()
            total_loss += float(result["pred_sum"].detach())
            for name in ("content", "control"):
                if name + "_sum" in result:
                    metric_sums[name] += float(result[name + "_sum"])
                    metric_counts[name] += int(result[name + "_count"])
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.r_pred.parameters(), train_args["clip_norm"])
        scale_before = scaler.get_scale()
        if finite_loss and bool(torch.isfinite(norm)):
            scaler.step(optimizer)
            scaler.update()
            success = scaler.get_scale() >= scale_before
        else:
            # norm 非有限也可能是范数求和溢出；明确跳过，避免错误执行 optimizer。
            scaler.update(new_scale=scale_before * 0.5)
            success = False
        if not success:
            counters["skipped_updates"] += 1
            counters["skipped_targets"] += target_count
            consecutive_skips += 1
            print(json.dumps({"skipped_update": True, "step": step, "targets": target_count,
                "scale": scaler.get_scale(), "source_index": stream.source_index}, ensure_ascii=False), flush=True)
            if consecutive_skips >= args.max_consecutive_skips:
                save_checkpoint(output / "numeric_stop.pt", model, optimizer, scheduler, scaler,
                    step=step, stream=stream, counters=counters, data_args=data_args,
                    train_args=train_args, ae_source=ae_source, data_manifest=manifest)
                raise RuntimeError("连续数值溢出；状态已保存到 numeric_stop.pt")
            continue
        consecutive_skips = 0
        counters["successful_targets"] += target_count
        for task, count in task_targets.items():
            counters["successful_targets:" + task] += count
        scheduler.step()
        step += 1
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"step": step, "pred_nll": total_loss / target_count,
                "content_nll": metric_sums["content"] / metric_counts["content"] if metric_counts["content"] else None,
                "control_nll": metric_sums["control"] / metric_counts["control"] if metric_counts["control"] else None,
                "kind_targets": dict(metric_counts), "microbatches": len(cpu_batches),
                "prepare_seconds": round(prepared - update_started, 3),
                "update_seconds": round(time.monotonic() - update_started, 3),
                "targets": target_count, "task_targets": dict(task_targets), "counters": dict(counters),
                "group_read_targets": stream.group_targets, "task_read_targets": stream.task_targets,
                "source_index": stream.source_index,
                "skipped_documents": dict(stream.skipped), "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(norm), "scale": scaler.get_scale(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device)}, ensure_ascii=False), flush=True)
        if step % args.save_every == 0:
            save_checkpoint(output / f"ckpt_{step:06d}.pt", model, optimizer, scheduler, scaler,
                step=step, stream=stream, counters=counters, data_args=data_args,
                train_args=train_args, ae_source=ae_source, data_manifest=manifest)
    save_checkpoint(output / "final.pt", model, optimizer, scheduler, scaler, step=step,
        stream=stream, counters=counters, data_args=data_args, train_args=train_args,
        ae_source=ae_source, data_manifest=manifest)
    print(f"[train_pretrain] saved {output / 'final.pt'}", flush=True)
    budget.finish(output, step)


if __name__ == "__main__":
    main()
