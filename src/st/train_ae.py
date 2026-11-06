"""新分布独立 AE 入口：沿用 E/D/DCT 权重，完全不实例化 R（2026-10-02）。

正式训练用 FP16 主干 + GradScaler，GMM 头与损失使用 FP32。默认 steps=0，
不启动训练；--check 做 CPU 前后向/保存恢复检查。完整恢复 checkpoint 的
model 键仍为 encoder./decoder./dct_head.，可由 R 阶段的 --init_ae 读取。
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields
import json
import inspect
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.st.ae_data import build_ae_loader
from src.st.config_st import STConfig
from src.st.model import CurveDecoder, StrokeEncoder
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.run_budget import RunBudget


class StrokeAutoencoder(nn.Module):
    """只包含现有 E/D/DCT；参数名与 STModel 的 AE 部分完全一致。"""

    def __init__(self, cfg: STConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = StrokeEncoder(cfg)
        self.decoder = CurveDecoder(cfg)
        self.dct_head = None
        if cfg.dct_coeffs > 0:
            layers = []
            size = cfg.latent_units
            for _ in range(cfg.dct_layers):
                layers.extend((nn.Linear(size, cfg.dct_hidden), nn.ReLU()))
                size = cfg.dct_hidden
            layers.append(nn.Linear(size, cfg.dct_coeffs * 2 + (2 if cfg.dct_dc else 0)))
            self.dct_head = nn.Sequential(*layers)

    def forward(self, batch: dict) -> dict:
        embeddings = self.encoder(batch["ink"], batch["stroke_len"])
        reconstructed = self.decoder.recon_loss(embeddings, batch["tau"],
            batch["tau_target"], batch["end_target"],
            torch.ones(len(embeddings), dtype=torch.bool, device=embeddings.device))
        if self.dct_head is not None:
            # 与 GMM 一样，辅助目标的数值计算脱离 FP16 autocast。
            with torch.autocast(device_type=embeddings.device.type, enabled=False):
                dct_prediction = self.dct_head(embeddings.float())
                dct_mse = (dct_prediction - batch["dct_target"].float()).square().mean()
        else:
            dct_mse = embeddings.new_zeros((), dtype=torch.float32)
        return {**reconstructed, "dct_mse": dct_mse,
                "loss": reconstructed["recon_total"] + self.cfg.dct_weight * dct_mse}

    @torch.no_grad()
    def reconstruct(self, batch: dict, num_points: int = 76) -> torch.Tensor:
        """返回页坐标 [B,num_points,2]，用于观察现有 AE 的重建样张。"""
        embeddings = self.encoder(batch["ink"], batch["stroke_len"])
        tau = torch.linspace(0, 1, num_points, device=embeddings.device).expand(
            len(embeddings), -1)
        mu, _, pi = self.decoder(embeddings, tau)
        best = pi.argmax(-1)[..., None, None].expand(-1, -1, 1, self.cfg.coord_dim)
        normalized = mu.gather(2, best).squeeze(2)
        return normalized * 0.6 + normalized.new_tensor([0.5, 0.6])


def config_from_checkpoint(state: dict, *, for_initialization: bool = False) -> STConfig:
    """初始化只继承 AE 的实际规格；新 R 配置不被旧小 R 覆盖。"""
    cfg = gpt2_small_pretrain_config()
    source = state.get("cfg", {})
    if not isinstance(source, dict):
        source = vars(source)
    known = {field.name for field in fields(STConfig)}
    for key, value in source.items():
        if key not in known:
            continue
        if not for_initialization or key.startswith(("enc_", "dec_", "dct_")) or key in {
            "coord_dim", "latent_units", "max_stroke_len", "gmm_dec", "n_t_samples",
            "endpoint_loss_w", "sigma_clamp_lo", "sigma_clamp_hi", "sigma_eps", "unified_emb"}:
            setattr(cfg, key, value)
    cfg.freeze_ed = False
    cfg.free_input_encoder = False
    cfg.gmm_head_fp32 = True
    cfg.amp = True
    cfg.amp_dtype = "fp16"
    if not cfg.unified_emb or cfg.coord_dim != 2:
        raise ValueError("新分布入口需要现有统一坐标 AE（unified_emb=True、coord_dim=2）")
    return cfg


def load_ae_weights(model: StrokeAutoencoder, state: dict):
    weights = state.get("model", state)
    selected = {key: value for key, value in weights.items()
                if key.startswith(("encoder.", "decoder.", "dct_head."))}
    model.load_state_dict(selected, strict=True)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def _restore_rng(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_checkpoint(path: str | Path, model: StrokeAutoencoder, optimizer,
                    scheduler, scaler, *, step: int, stream, collator,
                    data_args: dict, train_args: dict, data_manifest=None, stroke_counts=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": "ae", "format_version": 1,
               "model": model.state_dict(), "cfg": asdict(model.cfg), "step": step,
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "scaler": scaler.state_dict() if scaler is not None else None,
               "rng": _rng_state(), "data": stream.state_dict(),
               "collator": collator.state_dict(), "data_args": data_args,
               "train_args": train_args, "data_manifest": data_manifest}
    payload["stroke_counts"] = dict(stroke_counts or {})
    payload["ae_source"] = train_args.get("ae_source")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(state: dict, model, optimizer, scheduler, scaler, stream, collator):
    if state.get("stage") != "ae":
        raise ValueError("--resume 使用本独立 AE 入口的完整 checkpoint；旧权重用 --init_ae")
    load_ae_weights(model, state)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    stream.load_state_dict(state["data"])
    collator.load_state_dict(state["collator"])
    _restore_rng(state["rng"])
    return int(state["step"])


def _optimizer(model, lr: float, warmup: int):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda step: min((step + 1) / max(warmup, 1), 1.0))
    return optimizer, scheduler


def _clip_and_check(model, max_norm: float) -> tuple[bool, dict]:
    norms = {}
    for name in ("encoder", "decoder", "dct_head"):
        module = getattr(model, name)
        if module is not None:
            norms[name] = torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)
    return all(bool(torch.isfinite(norm)) for norm in norms.values()), norms


def _data_manifest(dataset):
    manifest = getattr(dataset, "manifest", None)
    return manifest() if callable(manifest) else manifest


def cpu_check(model, loader, stream, collator, output_dir, data_args, train_args, dataset):
    """单批 CPU 前后向、DCT 梯度与 checkpoint 恢复；不更新参数。"""
    model.train()
    batch = next(iter(loader))
    result = model(batch)
    if not all(bool(torch.isfinite(value)) for value in result.values()):
        raise RuntimeError("AE CPU 前向出现非有限损失")
    result["loss"].backward()
    gradients = {}
    for name in ("encoder", "decoder", "dct_head"):
        module = getattr(model, name)
        if module is None:
            continue
        parameters = [p for p in module.parameters() if p.grad is not None]
        if not parameters or not all(bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise RuntimeError(f"{name} 梯度缺失或非有限")
        gradients[name] = float(sum(p.grad.detach().abs().sum() for p in parameters))
    model.eval()
    reconstructed = model.reconstruct(batch)
    if not bool(torch.isfinite(reconstructed).all()):
        raise RuntimeError("AE 重建坐标非有限")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "reconstruction.npz", ink=batch["ink"].numpy(),
        stroke_len=batch["stroke_len"].numpy(), reconstructed=reconstructed.numpy(),
        kinds=np.array(batch["kinds"]))
    optimizer, scheduler = _optimizer(model, train_args["lr"], train_args["warmup"])
    checkpoint = output_dir / "check.pt"
    save_checkpoint(checkpoint, model, optimizer, scheduler, None, step=0,
        stream=stream, collator=collator, data_args=data_args, train_args=train_args,
        data_manifest=_data_manifest(dataset))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored = StrokeAutoencoder(config_from_checkpoint(saved))
    load_ae_weights(restored, saved)
    if not all(torch.equal(value, restored.state_dict()[key])
               for key, value in model.state_dict().items()):
        raise RuntimeError("AE checkpoint 权重恢复不一致")
    # 数据游标与 τ RNG 一起恢复，下一批应逐项一致。
    expected = next(iter(loader))
    stream.load_state_dict(saved["data"])
    collator.load_state_dict(saved["collator"])
    replayed = next(iter(loader))
    if (expected["kinds"] != replayed["kinds"] or
            not all(torch.equal(value, replayed[key]) for key, value in expected.items()
                    if torch.is_tensor(value))):
        raise RuntimeError("AE checkpoint 下一批数据恢复不一致")
    print(json.dumps({"check": "ok", "parameters": sum(p.numel() for p in model.parameters()),
        "losses": {key: float(value.detach()) for key, value in result.items()},
        "gradient_abs_sum": gradients, "stroke_kinds": dict(Counter(batch["kinds"])),
        "checkpoint": str(checkpoint)}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=0, help="目标 optimizer 更新总步数，0 不训练")
    parser.add_argument("--check", action="store_true", help="CPU 单批前后向和保存恢复检查")
    parser.add_argument("--init_ae", default="tmp_outputs/st_uni_frozen2/final.pt")
    parser.add_argument("--random_init", action="store_true", help="显式不用已有权重；通常保持关闭")
    parser.add_argument("--resume", help="恢复本入口保存的完整训练状态")
    parser.add_argument("--font", default="datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg")
    parser.add_argument("--raw_dir", default="datasets/quick_draw")
    parser.add_argument("--stories_path", help="TinyStories parquet 目录；省略时使用数据源默认")
    parser.add_argument("--text_file", help="以本地 UTF-8 文本代替 TinyStories 的小型输入")
    parser.add_argument("--manifest", help="JSON 对象：font_path/raw_dir/stories_path/text_file 等数据参数")
    parser.add_argument("--max_stories", type=int, default=1024)
    parser.add_argument("--max_per_class", type=int, default=256)
    parser.add_argument("--no_text_augmentation", action="store_true")
    parser.add_argument("--strokes_per_batch", type=int, default=256)
    parser.add_argument("--stroke_shares", nargs=3, type=float, default=(.50, .45, .05),
                        metavar=("SKETCH", "TEXT", "AUX"),
                        help="AE 按笔画混合：草图/文字/控制符与圈选，默认 50/45/5")
    parser.add_argument("--max_points", type=int, default=76)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--max_seconds", type=float, default=0,
                        help="本次训练循环时间上限；0 不限时，更新边界保存 final.pt")
    args = parser.parse_args()
    if args.steps < 0 or args.accumulate < 1 or args.max_points < 2 or args.lr <= 0:
        parser.error("steps >= 0，accumulate >= 1，max_points >= 2，lr > 0")
    if args.strokes_per_batch < 1 or args.log_every < 1 or args.save_every < 1:
        parser.error("batch/log_every/save_every 必须为正数")
    if any(not np.isfinite(value) or value < 0 for value in args.stroke_shares) or sum(args.stroke_shares) <= 0:
        parser.error("stroke_shares 必须为有限非负数，且至少一组大于零")
    if args.cpu_threads < 1 or args.warmup < 0 or args.clip_norm <= 0:
        parser.error("cpu_threads/clip_norm 必须为正数，warmup >= 0")
    if not np.isfinite(args.max_seconds) or args.max_seconds < 0:
        parser.error("max_seconds 必须为有限非负数")
    if args.check and args.steps:
        parser.error("--check 与 --steps 分开运行")
    return args


def main():
    args = parse_args()
    if not args.check and args.steps == 0:
        print("未启动训练。用 --check 检查 CPU 通路，或显式 --steps N 在 CUDA 上训练。")
        return
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.check else args.device)
    if not args.check and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("正式 AE 训练使用 CUDA FP16；CPU 通路使用 --check")
    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    initial = (None if args.random_init or resume is not None else
               torch.load(args.init_ae, map_location="cpu", weights_only=False))
    cfg = config_from_checkpoint(resume or initial or {}, for_initialization=resume is None)
    model = StrokeAutoencoder(cfg).to(device)
    if initial is not None:
        load_ae_weights(model, initial)
        print(f"[train_ae] 已沿用 AE 权重：{args.init_ae}", flush=True)
        del initial
    data_args = {"font_path": args.font, "raw_dir": args.raw_dir,
        "stories_path": args.stories_path, "text_file": args.text_file,
        "max_stories": args.max_stories, "max_per_class": args.max_per_class,
        "seed": args.seed, "augment_text": not args.no_text_augmentation}
    if args.manifest:
        supplied = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        if not isinstance(supplied, dict):
            raise ValueError("manifest 需要 JSON 对象")
        from src.st.pipeline_pretrain import build_document_dataset
        unknown = set(supplied) - set(inspect.signature(build_document_dataset).parameters)
        if unknown:
            raise ValueError(f"manifest 不认识的参数：{sorted(unknown)}")
        data_args.update(supplied)
    train_args = {key: getattr(args, key) for key in ("strokes_per_batch", "max_points",
        "accumulate", "lr", "warmup", "clip_norm")}
    train_args["ae_source"] = None if args.random_init else args.init_ae
    train_args["stroke_shares"] = dict(zip(("sketch", "text", "aux"), args.stroke_shares))
    if resume is not None:
        # 恢复实际训练设置；--steps 可延长总步数，设备/输出目录可迁移。
        data_args, train_args = resume["data_args"], resume["train_args"]
    if args.check and resume is None:
        train_args = {**train_args, "strokes_per_batch": min(train_args["strokes_per_batch"], 8)}
    from src.st.pipeline_pretrain import build_document_dataset
    dataset = build_document_dataset(**data_args)
    loader, stream, collator = build_ae_loader(dataset, cfg,
        strokes_per_batch=train_args["strokes_per_batch"], max_points=train_args["max_points"],
        seed=data_args["seed"], stroke_shares=train_args.get("stroke_shares"))
    output_dir = Path(args.output_dir or time.strftime("tmp_outputs/ae_%Y%m%d_%H%M%S"))
    if args.check:
        if resume is not None:
            load_ae_weights(model, resume)
            stream.load_state_dict(resume["data"])
            collator.load_state_dict(resume["collator"])
            _restore_rng(resume["rng"])
        cpu_check(model, loader, stream, collator, output_dir, data_args, train_args, dataset)
        return
    optimizer, scheduler = _optimizer(model, train_args["lr"], train_args["warmup"])
    # 旧 AE 的尖锐曲线 GMM 在 65536 起始倍率下会连续跳步；直接从实测可用倍率开始。
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    step = restore_checkpoint(resume, model, optimizer, scheduler, scaler, stream, collator) if resume else 0
    stroke_counts = Counter(resume.get("stroke_counts", {}) if resume else {})
    if resume is not None and resume.get("data_manifest") != _data_manifest(dataset):
        raise ValueError("续跑数据 manifest 与 checkpoint 不一致；检查字体/原始数据版本")
    print(f"[train_ae] {sum(p.numel() for p in model.parameters())/1e6:.3f}M AE 参数；"
          f"不实例化 R；FP16+GradScaler/FP32 GMM；batch={train_args['strokes_per_batch']} "
          f"accumulate={train_args['accumulate']}；step={step}->{args.steps}", flush=True)
    print(json.dumps({"ae_sampling": {"stroke_shares": train_args.get("stroke_shares"),
          "batch_quotas": getattr(stream, "quotas", None),
          "tasks": getattr(stream, "tasks", None)}}, ensure_ascii=False), flush=True)
    model.train()
    iterator = iter(loader)
    failed = 0
    started = time.monotonic()
    budget = RunBudget(args.max_seconds)
    while step < args.steps and budget.reason is None:
        batches = [next(iterator) for _ in range(train_args["accumulate"])]
        count = sum(len(batch["stroke_len"]) for batch in batches)
        optimizer.zero_grad(set_to_none=True)
        totals = Counter()
        kinds = Counter()
        for cpu_batch in batches:
            kinds.update(cpu_batch["kinds"])
            weight = len(cpu_batch["stroke_len"]) / count
            batch = _to_device(cpu_batch, device)
            with torch.autocast("cuda", dtype=torch.float16):
                result = model(batch)
            if not bool(torch.isfinite(result["loss"])):
                raise RuntimeError(f"AE loss 非有限：step={step}，kinds={dict(kinds)}")
            scaler.scale(result["loss"] * weight).backward()
            for key, value in result.items():
                totals[key] += float(value.detach()) * weight
        scaler.unscale_(optimizer)
        finite, _ = _clip_and_check(model, train_args["clip_norm"])
        if not finite:
            # norm 本身也可能溢出，即使单个梯度有限也不更新 Adam 动量。
            scaler.update(new_scale=scaler.get_scale() * 0.5)
            failed += 1
            print(f"[train_ae] 非有限梯度，GradScaler 跳过更新；scale={scaler.get_scale():g}", flush=True)
            if failed >= 10:
                raise RuntimeError("AE 连续梯度溢出，请查看当前损失与采样")
            continue
        scaler.step(optimizer)
        scaler.update()
        failed = 0
        scheduler.step()
        step += 1
        stroke_counts.update(kinds)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps({"step": step, **dict(totals), "lr": optimizer.param_groups[0]["lr"],
                "strokes": count, "kinds": dict(kinds), "kinds_cumulative": dict(stroke_counts),
                "scale": scaler.get_scale(),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
                "elapsed_seconds": round(time.monotonic() - started, 3)}, ensure_ascii=False), flush=True)
        if step % args.save_every == 0:
            save_checkpoint(output_dir / f"ckpt_{step:06d}.pt", model, optimizer, scheduler,
                scaler, step=step, stream=stream, collator=collator, data_args=data_args,
                train_args=train_args, data_manifest=_data_manifest(dataset), stroke_counts=stroke_counts)
    save_checkpoint(output_dir / "final.pt", model, optimizer, scheduler, scaler, step=step,
        stream=stream, collator=collator, data_args=data_args, train_args=train_args,
        data_manifest=_data_manifest(dataset), stroke_counts=stroke_counts)
    print(f"[train_ae] saved {output_dir / 'final.pt'}", flush=True)
    budget.finish(output_dir, step)


if __name__ == "__main__":
    main()
