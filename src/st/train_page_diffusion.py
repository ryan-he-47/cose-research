"""2026-11-02：以独立状态执行用户批准的从头训练预算。"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import signal
import time

import torch

from .page_diffusion_capacity import configure_memory, adapt_batching
from .page_diffusion_balance import block_clock, enter_balanced_stage, stage_metrics, window_metrics
from .page_diffusion_calibration import calibrate
from .formal_recipe import InputWSD, TASKS
from .generation_pretrain import PageGenerator
from .page_diffusion_codec import FrozenPageCodec, file_sha256
from .page_diffusion_data import prepare_update
from .page_diffusion_runtime import (autocast, create_stream, estimate_normalization,
    initialize_coordinates, new_training_state, train_update,
    save_checkpoint, validate_recipe, write_json, _rng_state, _restore_rng)
from .pipeline_pretrain import build_document_dataset


def budget_from_profile(profile, deadline, now=None):
    now = time.time() if now is None else now
    useful = deadline-now-300  # 原子保存与状态写入余量；14:00样张另在外部预留一小时。
    stable = profile[-min(8, len(profile)):]
    rate = sum(x["inputs"] for x in stable)/sum(x["seconds"] for x in stable)
    # 留15%吞吐波动余量，不把最好的单步当成稳定吞吐。
    updates = math.floor(useful*.85*rate/65536)
    if updates < 4:
        raise RuntimeError("截止前不足以完成独立正式短跑，保存实测结果后停止")
    return dict(total_inputs=updates*65536, estimated_updates=updates,
                measured_inputs_per_second=rate, available_seconds=useful,
                throughput_factor=.85, deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--data_manifest", type=Path)
    parser.add_argument("--init_ae", type=Path)
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument("--resume", type=Path)
    origin.add_argument("--balance_from", type=Path, help="按获批新配方从指定稳定段完整状态分出新阶段")
    parser.add_argument("--balance_config", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "deadline", "train"), default="train")
    parser.add_argument("--deadline", help="含时区的训练停止时间；采样准备时间已另预留")
    parser.add_argument("--profile_steps", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime_microbatch_documents", type=int,
                        help="完整恢复时显式修改每个微批文档上限，不改变一次更新的输入预算")
    parser.add_argument("--runtime_padded_stroke_budget", type=int,
                        help="完整恢复时显式修改双流打包预算")
    parser.add_argument("--runtime_memory_policy", choices=("strict", "soft"))
    parser.add_argument("--runtime_adaptive_microbatch", action=argparse.BooleanOptionalAction, default=None,
                        help="在成功正式更新中根据显存和吞吐调整拆分，不增加测试更新")
    parser.add_argument("--runtime_activation_checkpoint", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--runtime_encode_chunk_size", type=int)
    parser.add_argument("--runtime_packing", choices=("legacy", "cost"))
    parser.add_argument("--runtime_attention_layout", choices=("dense", "aligned", "split"))
    parser.add_argument("--runtime_profile", action="store_true",
                        help="只观察首个正式更新中的一个微批，不增加训练更新")
    args = parser.parse_args()
    starting_balance = args.balance_from is not None
    if starting_balance != (args.balance_config is not None):
        parser.error("新阶段起点和重平衡配方须一起指定")
    if starting_balance:
        if args.mode != "train" or args.deadline:
            parser.error("重平衡接续采用固定输入预算")
        args.resume = args.balance_from
    runtime_override = {key: value for key, value in (
        ("microbatch_documents", args.runtime_microbatch_documents),
        ("padded_stroke_budget", args.runtime_padded_stroke_budget),
        ("encode_chunk_size", args.runtime_encode_chunk_size)) if value is not None}
    if any(value <= 0 for value in runtime_override.values()):
        parser.error("运行参数必须为正数")
    if args.runtime_memory_policy is not None:
        runtime_override["memory_policy"] = args.runtime_memory_policy
    if args.runtime_adaptive_microbatch is not None:
        runtime_override["adaptive_microbatch"] = args.runtime_adaptive_microbatch
    for key in ("activation_checkpoint", "packing", "attention_layout"):
        value = getattr(args, "runtime_"+key)
        if value is not None:
            runtime_override[key] = value
    if runtime_override and not args.resume:
        parser.error("运行参数覆盖仅限完整检查点恢复；从头训练直接使用获批配方")
    if not 1 <= args.profile_steps <= 16:
        parser.error("工程验证限1–16次更新")
    if os.getenv("COSE_REQUIRE_SLURM") == "1" and not os.getenv("SLURM_JOB_ID"):
        parser.error("禁止登录节点重计算")
    if args.resume and any((args.init_ae, args.recipe, args.data_manifest)):
        parser.error("完整恢复只读取检查点配方，不混入新基础JSON")
    if not args.resume and not all((args.init_ae, args.recipe, args.data_manifest)):
        parser.error("从头起跑需要AE、配方及数据清单")
    deadline = None
    if args.deadline:
        instant = datetime.fromisoformat(args.deadline)
        if instant.tzinfo is None:
            parser.error("截止时间必须包含时区")
        deadline = instant.timestamp()
    if args.mode == "deadline" and deadline is None:
        parser.error("按时间预算必须指定deadline")
    if deadline and time.time() >= deadline-600:
        parser.error("已没有足够时间进行初始化与安全保存")
    output = args.output_dir.resolve()
    if (starting_balance or not args.resume) and output.exists() and any(output.iterdir()):
        parser.error("新训练必须使用独立空目录")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    stopping = {"reason": None}
    def request_stop(signum, frame):
        stopping["reason"] = "signal_"+str(signum)
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)

    if args.resume:
        restored = torch.load(args.resume, map_location="cpu", weights_only=False)
        if restored.get("kind") != "page_diffusion_training_v2":
            parser.error("当前只恢复无分类辅助的v2完整检查点；旧含辅助模型须用其绑定提交")
        if starting_balance:
            spec = json.loads(args.balance_config.read_text(encoding="utf-8"))
            restored = enter_balanced_stage(restored, spec, file_sha256(args.resume), args.resume)
        elif "balance" in restored["recipe"]:
            policy = restored["recipe"]["balance"]
            phase_inputs = restored["scheduler"]["inputs"] - policy["parent_inputs"]
            if ("reference_mse" not in policy or phase_inputs != sum(restored["stream"]["inputs"].values())
                    or restored["scheduler"]["inputs"] != restored["counters"]["successful_inputs"]
                    or restored.get("block_clock") != block_clock(restored["recipe"], restored["step"])):
                raise ValueError("恢复时的标尺、成功输入数或混合块时钟不一致")
        recipe, data_args = deepcopy(restored["recipe"]), restored["data_args"]
        codec = FrozenPageCodec.from_state(restored["codec"])
        source, normalization = restored["source"], restored["normalization"]
        if deadline is not None and source.get("deadline") != args.deadline:
            parser.error("恢复不得静默更换已确定的时间预算")
        if deadline is None and source.get("deadline"):
            deadline = datetime.fromisoformat(source["deadline"]).timestamp()
        runtime_commit = os.getenv("COSE_GIT_RUNTIME_COMMIT")
        if starting_balance:
            source["balance_origin"].update(code_commit=runtime_commit, job_id=os.getenv("SLURM_JOB_ID"))
        if runtime_commit and runtime_commit != source.get("code_commit"):
            source.setdefault("resume_code_commits", []).append(runtime_commit)
        source.setdefault("resume_events", []).append(dict(job_id=os.getenv("SLURM_JOB_ID"),
            code_commit=runtime_commit, checkpoint=str(args.resume), step=int(restored["step"]),
            successful_inputs=int(restored["scheduler"]["inputs"])))
    else:
        restored = None
        recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
        if recipe.get("approval") != "approved_20261102_360m_no_aux":
            parser.error("缺少本次移除分类辅助并从头训练的批准标记")
        if deadline or recipe.get("total_inputs") != 360000000:
            parser.error("本轮批准固定3.6亿输入，不能混入旧时间截止或更换总量")
        data_args = json.loads(args.data_manifest.read_text(encoding="utf-8"))
        data_args.update(split="train", object_sampling="affine_pass_v1", paired_scene_share=0.)
        ae = torch.load(args.init_ae, map_location="cpu", weights_only=False)
        if ae.get("step") != 55000:
            parser.error("必须使用已接受的55k冻结AE")
        codec = FrozenPageCodec.from_state(ae)
        del ae
        source = dict(code_commit=os.getenv("COSE_GIT_RUNTIME_COMMIT"), ae_path=str(args.init_ae),
                      ae_sha256=file_sha256(args.init_ae), deadline=args.deadline,
                      initialization="random_R_without_auxiliary_new_optimizer_new_stream",
                      job_id=os.getenv("SLURM_JOB_ID"), started_utc=datetime.now(timezone.utc).isoformat())
        normalization = None
    if runtime_override:
        defaults = {"memory_policy":"strict", "adaptive_microbatch":False,
                    "encode_chunk_size":256, "packing":"legacy", "attention_layout":"dense"}
        changes = {}
        for key, value in runtime_override.items():
            owner = recipe["model"] if key == "activation_checkpoint" else recipe
            before = owner.get(key, True if key == "activation_checkpoint" else defaults.get(key))
            if before != value:
                changes[("model." if key == "activation_checkpoint" else "")+key] = dict(before=before, after=value)
            owner[key] = value
        if changes:
            recipe.pop("runtime_batch_tuning", None)
            source.setdefault("runtime_changes", []).append(dict(step=int(restored["step"]),
                successful_inputs=int(restored["scheduler"]["inputs"]), changes=changes,
                code_commit=os.getenv("COSE_GIT_RUNTIME_COMMIT")))
    validate_recipe(recipe)
    if recipe.get("adaptive_microbatch") and recipe.get("memory_policy") != "soft":
        parser.error("正式更新内自适应批量需要soft显存策略")
    if device.type == "cuda":
        memory = configure_memory(device, recipe)
        write_json(output/"gpu_admission.json", memory)
    codec.encode_chunk_size = int(recipe.get("encode_chunk_size", 256))
    codec.to(device)
    dataset = build_document_dataset(**data_args)
    manifest = dataset.manifest()
    if restored and restored["source"].get("data_manifest") != manifest:
        raise ValueError("恢复数据清单与检查点不同")
    source.update(data_manifest=manifest)
    del dataset
    write_json(output/"source.json", source)
    if normalization is None:
        normalization = estimate_normalization(codec, data_args, recipe, device)
        write_json(output/"normalization.json", normalization)
    generator = PageGenerator(codec)
    from src.page_document import control_stroke
    eop = generator.encode([control_stroke("eop")])[0].float()
    del generator

    def initialize():
        m, opt, sc, avg = new_training_state(recipe, device)
        initialize_coordinates(m, avg, normalization, eop)
        m.train()
        rng = torch.Generator(device="cpu").manual_seed(recipe["seed"]+100003)
        return m, opt, sc, avg, rng

    profile = []
    if not restored and args.mode in ("profile", "deadline"):
        profile_started = time.monotonic()
        model, optimizer, scaler, ema, noise = initialize()
        stream = create_stream(data_args, recipe)
        try:
            for step in range(1, args.profile_steps+1):
                if (stopping["reason"] or time.monotonic()-profile_started >= 3600
                        or (deadline and time.time() >= deadline-300)):
                    break
                update_started = time.monotonic()
                documents, inputs, *_ = stream.take(recipe["input_budget"])
                counts = prepare_update(documents, noise, recipe["model"]["block"])
                metrics = train_update(model, codec, optimizer, scaler, ema, documents, counts, recipe, device)
                record = dict(step=step, inputs=sum(inputs.values()), **metrics)
                record["seconds"] = time.monotonic()-update_started
                profile.append(record)
                print(json.dumps({"phase": "profile", **record}), flush=True)
                write_json(output/"profile.json", profile)
        finally:
            stream.close()
        if len(profile) != args.profile_steps:
            raise RuntimeError("工程验证未完成，不启动正式训练")
        profile_report = dict(updates=len(profile), parameters=sum(p.numel() for p in model.parameters()),
            last_microbatch=recipe["microbatch_documents"], padded_stroke_budget=recipe["padded_stroke_budget"],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            normalization_targets=normalization["count"], optimizer_states=len(optimizer.state),
            finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            formal_optimizer_updates=0)
        write_json(output/"profile_summary.json", profile_report)
        if not profile_report["finite"]:
            raise RuntimeError("工程验证参数非有限")
        if args.mode == "profile":
            return
        del model, optimizer, scaler, ema, noise, documents
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        budget = budget_from_profile(profile, deadline)
        recipe.update(total_inputs=budget["total_inputs"], warmup_inputs=max(1, round(budget["total_inputs"]*.01)),
                      decay_inputs=max(1, round(budget["total_inputs"]*.2)))
        write_json(output/"deadline_budget.json", budget)
        print(json.dumps({"phase": "budget", **budget}), flush=True)

    if not recipe.get("total_inputs"):
        parser.error("正式训练缺少已确定的输入预算")
    model, optimizer, scaler, ema, noise = initialize()
    scheduler = InputWSD(optimizer, recipe, start_lr=restored["scheduler"]["spec"]["start_lr"] if restored else 0.)
    counters, step = Counter(), 0
    if restored:
        expected_cfg = deepcopy(restored["model_cfg"])
        if args.runtime_activation_checkpoint is not None:
            expected_cfg["activation_checkpoint"] = args.runtime_activation_checkpoint
        if expected_cfg != model.configuration():
            raise ValueError("检查点模型配置与配方不一致")
        model.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        scaler.load_state_dict(restored["scaler"])
        ema = {k: v.to(device) for k, v in restored["ema"].items()}
        scheduler.load_state_dict(restored["scheduler"])
        noise.set_state(restored["noise_rng"])
        _restore_rng(restored["rng"])
        counters, step = Counter(restored["counters"]), int(restored["step"])
    stream = create_stream(data_args, recipe, restored["stream"] if restored else None, new_exposure=starting_balance)
    policy = recipe.get("balance")
    write_json(output/"recipe.json", recipe)
    if not restored:
        initialized = dict(source=source, parameters=sum(p.numel() for p in model.parameters()),
            inherited_R=False, inherited_optimizer=False, inherited_stream=False, inherited_rng=False,
            initial_step=0, successful_inputs=0, target_inputs=recipe["total_inputs"],
            model_version=model.cfg.version, auxiliary_heads=False,
            optimizer_step_min=min(int(v["step"]) for v in optimizer.state.values()),
            optimizer_step_max=max(int(v["step"]) for v in optimizer.state.values()))
        write_json(output/"initialization.json", initialized)
        print(json.dumps({"phase":"initialized", **initialized}), flush=True)
    else:
        optimizer_steps = [int(values["step"]) for values in optimizer.state.values()]
        resumed = dict(step=step, successful_inputs=scheduler.inputs, target_inputs=recipe["total_inputs"],
            optimizer_step_min=min(optimizer_steps), optimizer_step_max=max(optimizer_steps),
            optimizer_states=len(optimizer_steps), ema_entries=len(ema),
            microbatch_documents=recipe["microbatch_documents"], padded_stroke_budget=recipe["padded_stroke_budget"],
            memory_policy=recipe.get("memory_policy", "strict"), adaptive_microbatch=recipe.get("adaptive_microbatch", False),
            activation_checkpoint=model.cfg.activation_checkpoint, encode_chunk_size=codec.encode_chunk_size,
            packing=recipe.get("packing", "legacy"), attention_layout=recipe.get("attention_layout", "dense"))
        if policy:
            boundary = stream.state_dict()
            if starting_balance and (any(boundary["inputs"].values()) or
                    any(boundary[key] != restored["stream"][key] for key in ("indices", "offsets"))):
                raise ValueError("接续时的来源游标或新曝光起点不符")
            resumed.update(block_clock=block_clock(recipe,step), phase_inputs=sum(boundary["inputs"].values()),
                           current_lr=optimizer.param_groups[0]["lr"],
                           decay_start_inputs=recipe["total_inputs"]-scheduler.spec["decay_inputs"])
        write_json(output/"resume_audit.json", resumed)
        print(json.dumps({"phase":"resumed", **resumed}), flush=True)
    del restored
    last_save, start_time = time.monotonic(), time.monotonic()
    profile_pending = args.runtime_profile
    tuning = recipe.get("runtime_batch_tuning")
    if args.resume and tuning and not tuning.get("done"):
        tuning["warmup_remaining"] = max(1, tuning.get("warmup_remaining", 0))
    def save(name="last.pt", boundary_state=None):
        save_checkpoint(output/name, model=model, codec=codec, optimizer=optimizer, scaler=scaler,
            ema=ema, scheduler=scheduler, stream=stream, noise_generator=noise, recipe=recipe,
            data_args=data_args, normalization=normalization, source=source, step=step, counters=counters,
            boundary_state=boundary_state,
            inference_name="inference.pt" if name == "last.pt" else "pre_decay_inference.pt")
        write_json(output/("latest.json" if name == "last.pt" else "pre_decay.json"), dict(step=step, successful_inputs=scheduler.inputs,
            target_inputs=recipe["total_inputs"], saved_utc=datetime.now(timezone.utc).isoformat()))
    decay_start = scheduler.spec["total_inputs"]-scheduler.spec["decay_inputs"]
    stable_saved = any(item["target_inputs"] == recipe["total_inputs"]
                       for item in source.get("stable_checkpoints", []))
    reason = "input_budget"
    recent = deque(maxlen=32)
    ready = not starting_balance
    try:
        if starting_balance:
            report = calibrate(model, codec, stream, noise, recipe, device,
                               should_stop=lambda: stopping["reason"] or (output/"STOP").exists())
            write_json(output/"balance_calibration.json", report)
            source["balance_origin"]["calibration_sha256"] = file_sha256(output/"balance_calibration.json")
            write_json(output/"source.json", source)
            write_json(output/"recipe.json", recipe)
            print(json.dumps(dict(phase="balance_calibrated", heldout_share=report["heldout_share"],
                                  loss_weights=recipe["loss_weights"])), flush=True)
            ready = True
            save()
            last_save = time.monotonic()
        while scheduler.inputs < recipe["total_inputs"]:
            if stopping["reason"] or (output/"STOP").exists():
                reason = stopping["reason"] or "stop_file"
                break
            if deadline and time.time() >= deadline-180:
                reason = "deadline"
                break
            if args.max_steps and step >= args.max_steps:
                reason = "max_steps"
                break
            before, noise_before, rng_before = stream.state_dict(), noise.get_state(), _rng_state()
            try:
                preparation_started = time.monotonic()
                documents, inputs, *_ = stream.take(min(recipe["input_budget"], recipe["total_inputs"]-scheduler.inputs))
                consumed = sum(inputs.values())
                if not stable_saved and scheduler.inputs <= decay_start < scheduler.inputs+consumed:
                    stable = dict(path=str(output/"pre_decay.pt"), step=step,
                                  successful_inputs=scheduler.inputs, target_inputs=recipe["total_inputs"],
                                  decay_start_inputs=decay_start)
                    source.setdefault("stable_checkpoints", []).append(stable)
                    try:
                        save("pre_decay.pt", dict(stream=before, noise_rng=noise_before, rng=rng_before))
                    except Exception:
                        source["stable_checkpoints"].pop()
                        raise
                    stable_saved = True
                    print(json.dumps({"phase":"stable_checkpoint", **stable}), flush=True)
                if policy:
                    width = block_clock(recipe,step)["next_width"]
                    counts = prepare_update(documents, noise, width, supervision_block=policy["supervision_block"])
                else:
                    counts = prepare_update(documents, noise, recipe["model"]["block"])
                preparation_seconds = time.monotonic()-preparation_started
                scheduler.prepare_update(consumed)
                metrics = train_update(model, codec, optimizer, scaler, ema, documents, counts, recipe, device,
                    profile_path=output/"runtime_profile.json" if profile_pending else None)
                profile_pending = False
            except Exception:
                stream.rewind(before)
                noise.set_state(noise_before)
                _restore_rng(rng_before)
                scheduler.prepare_update(0)
                raise
            scheduler.step(consumed)
            step += 1
            counters.update(successful_inputs=consumed, body_targets=sum(counts["body"].values()),
                control_targets=counts["control"], padding_slots=counts["padded"], blocks=counts["blocks"])
            counters.update({"inputs:"+key: value for key, value in inputs.items()})
            row = dict(step=step, successful_inputs=scheduler.inputs, lr=optimizer.param_groups[0]["lr"],
                wall_seconds=time.monotonic()-start_time,
                preparation_seconds=preparation_seconds,
                counts={**counts, "body": dict(counts["body"])}, **metrics)
            if policy:
                row.update(native_block=width, phase_step=step-policy["parent_step"],
                           phase_inputs=scheduler.inputs-policy["parent_inputs"], input_counts=dict(inputs),
                           grad_clipped=metrics["grad_norm"]>recipe["clip_norm"],
                           **stage_metrics(recipe,counts,metrics))
                recent.append(row)
                if row["phase_step"] % 4 == 0 and (row["phase_step"] <= 32 or row["phase_step"] % 128 == 0):
                    write_json(output/"balance_progress.json", window_metrics(recent))
            adjustment = adapt_batching(recipe, {**metrics, "preparation_seconds":preparation_seconds}, consumed) if device.type == "cuda" else None
            if adjustment:
                if ((adjustment.get("after", 0) and adjustment["after"] < adjustment.get("before", 0)) or
                        adjustment.get("before_packing") != adjustment.get("after_packing")):
                    torch.cuda.empty_cache()
                row["runtime_adjustment"] = adjustment
                write_json(output/"runtime_state.json", dict(step=step, successful_inputs=scheduler.inputs,
                    microbatch_documents=recipe["microbatch_documents"], padded_stroke_budget=recipe["padded_stroke_budget"],
                    packing=recipe.get("packing", "legacy"),
                    memory_policy=recipe.get("memory_policy", "strict"), tuning=recipe.get("runtime_batch_tuning"),
                    adjustment=adjustment))
                print(json.dumps({"phase":"runtime_adjustment", **adjustment}), flush=True)
            with (output/"train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row)+"\n")
            if step == 1 or step % recipe["log_every"] == 0 or (policy and row["phase_step"] <= 4):
                print(json.dumps({"phase": "train", **row}), flush=True)
                write_json(output/"progress.json", row)
            if time.monotonic()-last_save >= recipe["save_seconds"]:
                save()
                last_save = time.monotonic()
    except Exception as error:
        reason = "error:"+type(error).__name__
        write_json(output/"failure.json", dict(error=repr(error), step=step, successful_inputs=scheduler.inputs))
        raise
    finally:
        try:
            if ready:
                save()
            if recent:
                write_json(output/"balance_progress.json", window_metrics(recent))
            write_json(output/"run_stop.json", dict(reason=reason, step=step, successful_inputs=scheduler.inputs,
                target_inputs=recipe["total_inputs"], counters=dict(counters), stopped_utc=datetime.now(timezone.utc).isoformat()))
        finally:
            stream.close()


if __name__ == "__main__":
    main()
