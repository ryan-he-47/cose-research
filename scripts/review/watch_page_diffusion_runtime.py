"""只读跟随Slurm日志，稳定后取一次元数据并写运行验收；不调用模型、不修改训练。"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(value, encoding="utf8")
    os.replace(temporary, path)


def ssh_command(host, command):
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=2", host, command]


def snapshot_command(run, job):
    source = f'''import json,pathlib,subprocess
p=pathlib.Path({run!r})
d={{}}
for name in ["initialization","resume_audit","runtime_state","gpu_admission","runtime_profile","recipe","progress","latest","failure","run_stop"]:
 f=p/(name+".json");d[name]=json.loads(f.read_text()) if f.exists() else None
f=p/"source.json"
if f.exists():
 s=json.loads(f.read_text());d["source"]={{k:s.get(k) for k in ["code_commit","resume_events","runtime_changes","ae_sha256"]}}
d["train"]=[]
f=p/"train.jsonl"
if f.exists():
 for line in f.read_text().splitlines():
  if line.strip():
   try:d["train"].append(json.loads(line))
   except json.JSONDecodeError:pass
d["slurm"]=subprocess.run(["scontrol","show","job",{str(job)!r},"-o"],capture_output=True,text=True).stdout
print(json.dumps(d))
'''
    encoded = base64.b64encode(source.encode()).decode()
    return "python -c "+shlex.quote("import base64;exec(base64.b64decode("+repr(encoded)+"))")


def summarize_rows(rows):
    if not rows:
        return None
    result = dict(count=len(rows), first_step=rows[0]["step"], last_step=rows[-1]["step"],
        mean_update_seconds=statistics.mean(r["seconds"] for r in rows),
        retries=sum(r.get("retries", 0) for r in rows),
        oom_retries=sum(r.get("oom_retries", 0) for r in rows))
    for key in ["preparation_seconds", "physical_peak_headroom_gib", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "microbatches", "padded_positions"]:
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            result[key] = dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values))
    if len(rows)>1:
        elapsed = rows[-1]["wall_seconds"]-rows[0]["wall_seconds"]
        result.update(end_to_end_seconds=elapsed/(len(rows)-1),
                      inputs_per_second=(rows[-1]["successful_inputs"]-rows[0]["successful_inputs"])/elapsed)
    return result


def make_report(snapshot, args, reason):
    audit = snapshot.get("initialization" if args.fresh else "resume_audit") or {}
    recipe = snapshot.get("recipe") or {}
    source = snapshot.get("source") or {}
    events = source.get("resume_events") or []
    rows = snapshot.get("train") or []
    changes = [r["step"] for r in rows if r.get("runtime_adjustment")]
    boundary = max(changes, default=args.resume_step)
    stable = [r for r in rows if r["step"]>boundary and not r.get("runtime_profiled")][-args.stable_updates:]
    checks = dict(resume_step=audit.get("initial_step" if args.fresh else "step")==args.resume_step,
        resume_inputs=audit.get("successful_inputs")==args.resume_inputs,
        optimizer_steps=audit.get("optimizer_step_min")==audit.get("optimizer_step_max")==args.resume_step,
        target_inputs=audit.get("target_inputs")==360000000,
        code_commit=(source.get("code_commit")==args.commit if args.fresh else bool(events) and events[-1].get("code_commit")==args.commit),
        activation_checkpoint_disabled=(recipe.get("model", {}).get("activation_checkpoint") if args.fresh else audit.get("activation_checkpoint")) is False,
        split_attention=(recipe.get("attention_layout") if args.fresh else audit.get("attention_layout"))=="split",
        continued=bool(rows) and rows[-1]["step"]>args.resume_step,
        no_reported_failure=snapshot.get("failure") is None,
        capacity_settled=bool(((snapshot.get("runtime_state") or {}).get("tuning") or {}).get("done")),
        stable_window=len(stable)>=args.stable_updates)
    if args.fresh:
        checks.update(fresh_states=all(audit.get(k) is False for k in
            ("inherited_R", "inherited_optimizer", "inherited_stream", "inherited_rng")) and not events,
            no_auxiliary=audit.get("auxiliary_heads") is False and audit.get("model_version")=="page_diffusion_v2"
                and not any(k in recipe for k in ("text_aux_alpha", "text_aux_gradient_ratio", "calibration_interval"))
                and not any("alpha" in row or "shape_ce_sum" in row or "space_bce_sum" in row for row in rows))
    summary = dict(observed_at=now(), trigger=reason, job_id=args.job, checks=checks,
                   stable=summarize_rows(stable), runtime_state=snapshot.get("runtime_state"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8-sig"))
    summary["before"] = summarize_rows(baseline["last_100"])
    passed = all(checks.values())
    lines = ["# 扩散管线服务器运行验收", "", f"更新时间：{summary['observed_at']}。作业{args.job}，代码`{args.commit}`。", "",
             "已采集稳定运行数据并通过起点核对。" if passed else "当前记录尚不能完成全部运行验收，未通过的项目列在下方。", "",
             f"{'从头初始化' if args.fresh else '恢复'}记录：第{audit.get('initial_step' if args.fresh else 'step')}步、{audit.get('successful_inputs')}个成功输入，总目标{audit.get('target_inputs')}。"]
    if args.fresh:
        lines += ["", "本轮已移除分类辅助并从头训练；下表只比较运行效率，两轮损失值不按同一目标比较。"]
    if summary["stable"] and "inputs_per_second" in summary["stable"]:
        current, previous = summary["stable"], summary["before"]
        change = (current["inputs_per_second"]/previous["inputs_per_second"]-1)*100
        lines += ["", "| 指标 | 上一版 | 新管线稳定段 |", "|---|---:|---:|",
            f"| 记录更新数 | {previous['count']} | {current['count']} |",
            f"| 端到端秒/步 | {previous['end_to_end_seconds']:.3f} | {current['end_to_end_seconds']:.3f} |",
            f"| 有效输入/秒 | {previous['inputs_per_second']:.0f} | {current['inputs_per_second']:.0f} |",
            f"| 重试次数 | {previous['retries']} | {current['retries']} |", "",
            f"当前稳定段为第{current['first_step']}–{current['last_step']}步，吞吐比上一版{'提高' if change>=0 else '下降'}{abs(change):.1f}%。"]
        if change<0:
            lines += ["这段记录的吞吐低于上一版，需要进一步定位；观察器不会修改训练。"]
        memory = current.get("physical_peak_headroom_gib")
        if memory:
            lines += [f"计入分配器缓存及外部CUDA占用后的峰值余量为{memory['minimum']:.2f}–{memory['maximum']:.2f}GiB。"]
        selected = rows[-1]
        lines += [f"记录末尾的执行设置：分组`{selected.get('packing')}`，位置预算{selected.get('padded_stroke_budget')}，激活重计算{selected.get('activation_checkpoint')}。"]
    failed = [key for key,value in checks.items() if not value]
    if failed:
        labels = dict(resume_step="恢复步数", resume_inputs="恢复输入计数", optimizer_steps="Adam步数",
            target_inputs="总预算", code_commit="源码提交", activation_checkpoint_disabled="关闭激活重计算",
            split_attention="双流注意力", continued="已继续训练", no_reported_failure="训练错误记录",
            capacity_settled="容量调整已结束", stable_window="连续稳定更新数量",
            fresh_states="各状态从头初始化", no_auxiliary="无分类辅助")
        lines += ["", "仍需检查："+"、".join(labels[k] for k in failed)+"。"]
    if snapshot.get("failure"):
        lines += ["", "训练错误：`"+str(snapshot["failure"].get("error"))+"`。"]
    profile = snapshot.get("runtime_profile") or {}
    operators = sorted({item["name"] for item in profile.get("attention", []) if "_scaled_dot_product_" in item["name"]})
    if operators:
        lines += ["", "一个正式微批中观察到的注意力算子："+"、".join("`"+name+"`" for name in operators)+"。"]
    lines += ["", "原始元数据与连续训练记录保存在同目录的`server_snapshot.json`，统计及逐项核对在`summary.json`。",
              "本观察器只跟随日志，达到稳定窗口后读取一次JSON元数据；不调用模型，不改训练，不提交任何作业。", ""]
    return summary, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--fresh", action="store_true", help="核验从头初始化，替代旧接续核验")
    parser.add_argument("--resume_step", type=int, default=0)
    parser.add_argument("--resume_inputs", type=int, default=0)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stable_updates", type=int, default=40)
    args = parser.parse_args()
    if args.fresh and (args.resume_step or args.resume_inputs):
        parser.error("从头初始化的步数和成功输入必须为0")
    args.out.mkdir(parents=True, exist_ok=True)
    report = args.out/"server_followup.md"
    origin = "从头初始化，分类辅助已移除" if args.fresh else f"从第{args.resume_step}步、{args.resume_inputs}输入接续"
    write(report, f"# 扩散管线服务器运行验收\n\n{now()}：等待作业{args.job}启动并形成稳定记录。\n\n预期{origin}，代码`{args.commit}`。\n\n只读观察器PID为{os.getpid()}，不查询队列、不调用模型、不修改训练、不追加作业；完成核对后自动结束。\n")
    shell = "tail -n +1 -F "+shlex.quote(args.log)+" & task_tail_pid=$!; read -r task_stop || true; kill \"$task_tail_pid\" 2>/dev/null || true; wait \"$task_tail_pid\" 2>/dev/null || true"
    command = "timeout 172800 bash -c "+shlex.quote(shell)
    process = subprocess.Popen(ssh_command(args.host, command), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf8", errors="replace")
    step, settled, trigger = args.resume_step, None, "log_connection_ended"
    try:
        with (args.out/"observed_events.jsonl").open("a", encoding="utf8") as log:
            for line in process.stdout:
                if "Traceback (most recent call last)" in line or "slurmstepd: error:" in line:
                    trigger = "training_error_in_log"
                    break
                try:event = json.loads(line)
                except json.JSONDecodeError:continue
                log.write(json.dumps(event, ensure_ascii=False)+"\n");log.flush()
                phase = event.get("phase")
                if phase == "initialized" and args.fresh:
                    write(report, f"# 扩散管线服务器运行验收\n\n{now()}：作业{args.job}已报告从头初始化，步数{event.get('initial_step')}、输入{event.get('successful_inputs')}，辅助头{event.get('auxiliary_heads')}。正在等待稳定运行记录。\n")
                if phase == "resumed":
                    write(report, f"# 扩散管线服务器运行验收\n\n{now()}：作业{args.job}已报告从第{event.get('step')}步、{event.get('successful_inputs')}输入恢复，正在等待容量调整后连续{args.stable_updates}步的记录。\n\n激活重计算：{event.get('activation_checkpoint')}；注意力：{event.get('attention_layout')}。\n")
                if phase == "runtime_adjustment":
                    reason = event.get("reason")
                    if reason in ("larger_batch_slower", "soft_target_or_batch_limit", "actual_oom_shrunk_same_batch", "soft_target_adjust_next_update") or (reason=="packing_comparison_finished" and event.get("after")==event.get("before")):
                        settled = step
                if phase == "train":
                    step = event["step"]
                    if (settled is not None and step>=settled+args.stable_updates+20) or step>=args.resume_step+250:
                        trigger = "stable_updates_observed"
                        break
        completed = subprocess.run(ssh_command(args.host, snapshot_command(args.run, args.job)),
            capture_output=True, text=True, encoding="utf8", timeout=60, check=True)
        snapshot = json.loads(completed.stdout)
        write(args.out/"server_snapshot.json", json.dumps(snapshot, ensure_ascii=False, indent=2))
        summary, markdown = make_report(snapshot, args, trigger)
        write(args.out/"summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        write(report, markdown)
        print(json.dumps(dict(finished_at=now(), report=str(report), checks=summary["checks"])), flush=True)
    except Exception as error:
        write(report, f"# 扩散管线服务器运行验收\n\n{now()}：观察器未能完成取数，需检查连接或作业状态。\n\n错误：`{error!r}`。\n\n观察器没有修改训练。\n")
        raise
    finally:
        try:
            process.stdin.write("\n");process.stdin.flush();process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            process.kill();process.wait()


if __name__ == "__main__":
    main()
