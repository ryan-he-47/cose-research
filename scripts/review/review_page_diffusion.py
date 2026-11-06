"""本机训练观察：固定来源原始D解码样张、停止记录及训练曲线，不计算小样本准确率。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import html
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from scripts.demo.demo_pretrain_generation import source_prefix, text_prefix, save_native_svg
from scripts.review.joint_eval_batch import load_dataset
from scripts.review.render_formal_r_review import svg_pages
from src.page_document import StrokeElement
from src.st.page_diffusion import PageDiffusionConfig, PageDiffusionR
from src.st.page_diffusion_codec import FrozenPageCodec, file_sha256
from src.st.page_diffusion_runtime import write_json
from src.st.page_diffusion_sampling import PageDiffusionGenerator
from src.text_renderer import VectorFont


TASKS = ("T1", "O1", "O2", "S1", "S2", "S3", "S4", "S5")
NAMES = dict(zip(TASKS, ("文字续写", "按名称画物体", "看图命名", "场景描述", "按描述画场景",
                        "场景问答", "按名称圈选", "圈内对象命名")))


def stroke_records(elements):
    return [dict(points=e.rawxy.tolist(), page_id=e.page_id, kind=e.kind, generated=False)
            for e in elements]


def prepare_cases(data, output, font_path):
    destination = output/"cases.json"
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    font = VectorFont.from_svg(font_path)
    dataset, info = load_dataset(data, "valid", font)
    rng = np.random.default_rng(20261102)
    objects = [info["objects"][int(i)] for i in rng.choice(len(info["objects"]), 4, replace=False)]
    scenes = [int(x) for x in rng.choice(info["scene_indices"], 4, replace=False)]
    cases = []
    prompts = ("Once upon a time, a little girl", "Tom and his dog went to the",
               "The red ball was under the", "Lily wanted to help her friend")
    for index, prompt in enumerate(prompts):
        cases.append(dict(id=f"T1_{index}", task="T1", source="手写续写前缀", prompt=prompt,
            prefix=stroke_records(text_prefix("T1", prompt, font)), reference=[], seed=2026110200+index))
    for task in TASKS[1:]:
        ids = [int(x["index"]) for x in objects] if task.startswith("O") else scenes
        for position, index in enumerate(ids):
            doc = dataset.get_for_task(task, index)
            prefix = source_prefix(doc, task)
            cases.append(dict(id=f"{task}_{index}", task=task, source="扩大评估valid来源",
                source_index=index, label=doc.metadata.get("label"),
                page_text=[p.text for p in doc.pages if p.text],
                prefix=stroke_records(prefix), reference=stroke_records(doc.to_sequence()),
                seed=2026110200+position))
    value = dict(version="page_diffusion_initial_panel_v1", selection_seed=20261102,
        data_manifest_sha256=file_sha256(data/"manifest.json"), font_sha256=file_sha256(font_path),
        selection="固定种子选4个对象来源、4个场景来源，O1/O2和S1–S5分别共用来源；T1为4条手写前缀。",
        objects=objects, scenes=scenes, cases=cases)
    write_json(destination, value)
    del dataset
    gc.collect()
    return value


def summarize_window(rows):
    if not rows:
        return None
    def ratio(numerator, denominator):
        return sum(numerator(r) for r in rows)/max(1, sum(denominator(r) for r in rows))
    return dict(updates=len(rows), first_step=rows[0]["step"], last_step=rows[-1]["step"],
        first_inputs=rows[0]["successful_inputs"], last_inputs=rows[-1]["successful_inputs"],
        mean_loss=float(np.mean([r["loss"] for r in rows])),
        body_mse=ratio(lambda r:r["real_body_sum"], lambda r:sum(r["counts"]["body"].values())),
        task_mse={t:ratio(lambda r:r.get("body_sum:"+t,0), lambda r:r["counts"]["body"].get(t,0)) for t in TASKS},
        noise_mse={str(t):ratio(lambda r:r.get(f"noise_{t}_sum",0), lambda r:r.get(f"noise_{t}_count",0))
                   for t in (0., .25, .5, .75)},
        control_mse=ratio(lambda r:r["real_control_sum"], lambda r:r["counts"]["control"]),
        padding_mse=ratio(lambda r:r["synthetic_sum"], lambda r:r["counts"]["padded"]),
        mean_update_seconds=float(np.mean([r["seconds"] for r in rows])))


def curves(run, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    path = run/"train.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return dict(updates=0)
    x = np.asarray([r["successful_inputs"]/1e6 for r in rows])
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    def smooth(values):
        a = np.asarray(values, float)
        window = max(1, min(20, len(a)//30))
        return np.asarray([np.nanmean(a[max(0,i-window+1):i+1]) for i in range(len(a))])
    for task in TASKS:
        y = [r.get("body_sum:"+task, 0)/max(1, r["counts"]["body"].get(task,0)) for r in rows]
        axes[0,0].plot(x, smooth(y), label=task)
    axes[0,0].set_title("正文误差（按任务，不含填充）")
    for lo in (0., .25, .5, .75):
        y = [r.get(f"noise_{lo}_sum", 0)/max(1,r.get(f"noise_{lo}_count",0)) for r in rows]
        axes[0,1].plot(x, smooth(y), label=f"t={lo:g}–{lo+.25:g}")
    axes[0,1].set_title("正文误差（按加噪时间）")
    for key, count, label in (("real_control_sum", "control", "真实控制"), ("synthetic_sum", "padded", "重复EOP")):
        axes[1,0].plot(x, smooth([r[key]/max(1,r["counts"][count]) for r in rows]), label=label)
    axes[1,0].set_title("控制/填充每槽误差（未乘损失系数）")
    axes[1,1].plot(x, smooth([r["grad_norm"] for r in rows]), label="裁剪前梯度范数")
    axes[1,1].set_yscale("log")
    axes[1,1].set_title("梯度范数（当前无分类辅助）")
    axes[2,0].plot(x, smooth([r["loss"] for r in rows]), label="总损失")
    twin = axes[2,0].twinx()
    twin.plot(x, [r["lr"] for r in rows], color="#b55335", alpha=.6, label="学习率")
    twin.set_ylabel("学习率", color="#b55335")
    axes[2,0].set_title("总损失与学习率")
    wall = np.asarray([r["wall_seconds"] for r in rows])
    seconds = np.diff(np.r_[0., wall])
    # 完整恢复会开始新的运行时钟，不能把负时间差压成极小数制造吞吐尖峰。
    seconds = np.where(seconds > 0, seconds, wall)
    inputs = np.diff([0]+[r["successful_inputs"] for r in rows])
    axes[2,1].plot(x, smooth(inputs/np.maximum(seconds, .001)), label="原始输入/秒")
    axes[2,1].set_title("端到端吞吐（含取数与中间保存）")
    recipe = json.loads((run/"recipe.json").read_text(encoding="utf8"))
    decay_start = recipe["total_inputs"]-recipe["decay_inputs"]
    for axis in axes.flat:
        axis.grid(alpha=.15)
        axis.set_xlabel("百万成功原始输入")
        axis.legend(fontsize=8, ncol=2)
        axis.axvspan(decay_start/1e6, recipe["total_inputs"]/1e6, color="#d9a441", alpha=.08)
        axis.axvline(decay_start/1e6, color="#b58a35", linewidth=.8, linestyle="--")
    fig.suptitle(f"无分类辅助扩散R · 阴影为末{recipe['decay_inputs']/1e4:g}万输入退火段", fontsize=16)
    fig.savefig(output/"training_curves.png", dpi=160)
    plt.close(fig)
    late = x >= max(0, decay_start/1e6-40)
    if late.sum() > 20:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        axes[0,0].plot(x[late], smooth([r["loss"] for r in rows])[late], label="总损失")
        axes[0,0].set_title("末段总损失")
        for task in TASKS:
            y = smooth([r.get("body_sum:"+task,0)/max(1,r["counts"]["body"].get(task,0)) for r in rows])
            axes[0,1].plot(x[late], y[late], label=task)
        axes[0,1].set_title("各任务正文MSE")
        for t in (0., .25, .5, .75):
            y = smooth([r.get(f"noise_{t}_sum",0)/max(1,r.get(f"noise_{t}_count",0)) for r in rows])
            axes[1,0].plot(x[late], y[late], label=f"t={t:g}–{t+.25:g}")
        axes[1,0].set_title("各噪声时段正文MSE")
        for key,count,label in (("real_control_sum","control","真实控制"),("synthetic_sum","padded","重复EOP")):
            y = smooth([r[key]/max(1,r["counts"][count]) for r in rows])
            axes[1,1].plot(x[late], y[late], label=label)
        axes[1,1].set_title("控制与填充每槽MSE")
        for axis in axes.flat:
            axis.axvspan(decay_start/1e6, recipe["total_inputs"]/1e6, color="#d9a441", alpha=.12)
            axis.axvline(decay_start/1e6, color="#b58a35", linewidth=.8, linestyle="--")
            axis.set_xlabel("百万成功原始输入")
            axis.grid(alpha=.15);axis.legend(fontsize=8,ncol=2)
        fig.suptitle("退火段细看 · 使用训练日志，20步滑动平均", fontsize=15)
        fig.savefig(output/"late_training_curves.png", dpi=160)
        plt.close(fig)
    stop = json.loads((run/"run_stop.json").read_text(encoding="utf8")) if (run/"run_stop.json").exists() else None
    return dict(updates=len(rows), last_step=rows[-1]["step"], successful_inputs=rows[-1]["successful_inputs"],
        wall_seconds=float(seconds.sum()), first_loss=rows[0]["loss"], last_loss=rows[-1]["loss"],
        inputs_per_second=float(sum(inputs)/seconds.sum()), retries=sum(r.get("retries",0) for r in rows),
        oom_retries=sum(r.get("oom_retries",0) for r in rows),
        mean_step_seconds=float(seconds.mean()),
        minimum_peak_headroom_gib=min((r["physical_peak_headroom_gib"] for r in rows
            if r.get("physical_peak_headroom_gib") is not None), default=None),
        log_directory=Path(os.path.relpath(run, output)).as_posix(),
        run_stop=stop, decay_start_inputs=decay_start,
        windows=dict(before_decay=summarize_window([r for r in rows if r["successful_inputs"] <= decay_start][-200:]),
                     final_200=summarize_window(rows[-200:])))


def render_overviews(output, cases):
    """用实际笔画生成浏览图；圈选识别的输入选圈也叠加在条件场景上。"""
    import matplotlib.pyplot as plt
    items = cases["cases"]
    if not all((output/f"{c['id']}_raw/result.json").exists() for c in items):
        return
    results = {c["id"]:json.loads((output/f"{c['id']}_raw/result.json").read_text(encoding="utf8")) for c in items}
    controls = ("eop", "eod", "bos")
    def draw(axis, records, title, page=None, color="#165ead", controls_visible=True):
        axis.set(xlim=(0,1), ylim=(1,0), xticks=[], yticks=[])
        axis.set_aspect("equal")
        axis.set_title(title, fontsize=8)
        for r in records:
            if (page is not None and r["page_id"]!=page) or (not controls_visible and r["kind"] in controls):
                continue
            xy = np.asarray(r["points"])
            tone = "#b53b64" if r["kind"] in controls else r.get("display_color", color)
            axis.plot(xy[:,0],xy[:,1],color=tone,lw=.8)
    def save(figure, name):
        figure.savefig(output/name, dpi=160)
        plt.close(figure)
    selected = [c for c in items if c["task"]=="T1"]
    fig, axes = plt.subplots(len(selected),3,figsize=(13,4*len(selected)),constrained_layout=True,squeeze=False)
    for row,c in enumerate(selected):
        value = results[c["id"]]
        records = [{**r,"display_color":"#a2a8ad"} for r in c["prefix"]]+value["records"]
        pages = sorted({r["page_id"] for r in records})
        for col,axis in enumerate(axes[row]):
            page = pages[col] if col<len(pages) else -1
            draw(axis,records,f"{c['id']} · 页{page} · {value['stop']}",page=page)
    save(fig,"text_samples.png")
    selected = [c for c in items if c["task"]=="O1"]
    fig, axes = plt.subplots(len(selected),3,figsize=(12,4*len(selected)),constrained_layout=True,squeeze=False)
    for row,c in enumerate(selected):
        name = " / ".join(c.get("page_text",[]))
        target = [r for r in c["reference"] if r["page_id"]>max(x["page_id"] for x in c["prefix"])]
        groups = [(target,"O1参考"),(results[c["id"]]["records"],"O1生成"),
                  (results[c["id"].replace("O1_","O2_")]["records"],"O2生成类名")]
        for axis,(records,label) in zip(axes[row],groups):
            draw(axis,records,label+" · "+name,page=min((r["page_id"] for r in records),default=0))
    save(fig,"object_samples.png")
    selected = [c for c in items if c["task"]=="S4"]
    fig, axes = plt.subplots(len(selected),4,figsize=(16,4*len(selected)),constrained_layout=True,squeeze=False)
    for col,task in enumerate(("S4","S5")):
        for row,c in enumerate(c for c in items if c["task"]==task):
            prefix = [{**r,"display_color":"#bd8b37" if r["kind"]=="selection" else "#a4a9ae"}
                      for r in c["prefix"] if r["kind"] not in ("text",*controls)]
            last = max(r["page_id"] for r in c["reference"])
            target = [r for r in c["reference"] if r["page_id"]==last]
            for sub,(strokes,label,color) in enumerate(((target,"参考","#287c4d"),(results[c["id"]]["records"],"生成","#165ead"))):
                title = c["id"]+" "+label+"\n"+" / ".join(c.get("page_text",[]))[-72:]
                draw(axes[row,col*2+sub],prefix+strokes,title,color=color,controls_visible=False)
    save(fig,"grounding_samples.png")
    selected = [c for c in items if c["task"]=="S1"]
    fig, axes = plt.subplots(len(selected),3,figsize=(12,4*len(selected)),constrained_layout=True,squeeze=False)
    for row,c in enumerate(selected):
        scene = [r for r in c["prefix"] if r["kind"]=="sketch"]
        other = c["id"].replace("S1_","S2_")
        groups = [(scene,"条件/参考场景"),(results[c["id"]]["records"],"S1生成描述"),(results[other]["records"],"S2按描述生成场景")]
        for axis,(records,label) in zip(axes[row],groups):
            draw(axis,records,c["id"].split("_")[1]+" · "+label,page=min((r["page_id"] for r in records),default=0))
    save(fig,"scene_samples.png")


def render(output, cases, summary):
    render_overviews(output, cases)
    entries = []
    stops = Counter()
    for case in cases["cases"]:
        methods = []
        for weights in ("raw", "ema"):
            path = output/f"{case['id']}_{weights}"/"result.json"
            if not path.exists():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            stops[weights+":"+value["stop"]] += 1
            generated = [{**r, "generated":True} for r in value["records"]]
            tail = sum(b["discarded_non_eop"] for b in value["blocks"])
            description = "当前权重" if weights == "raw" else "EMA权重"
            methods.append(f'<h4>{description}：{html.escape(value["stop"])}，{len(generated)}笔</h4>'
                f'<p>EOP后丢弃的非EOP槽位：{tail}；耗时{value["sampling_seconds"]:.1f}秒。</p>'
                f'<div class="pages">{svg_pages(case["prefix"]+generated)}</div>'
                f'<a href="{case["id"]}_{weights}/result.json">完整槽位与停止记录</a>')
        detail = case.get("prompt") or " / ".join(case.get("page_text", []))
        entries.append(f'<article data-task="{case["task"]}"><h3>{case["id"]} · {NAMES[case["task"]]}</h3>'
            f'<p>{html.escape(detail)}</p><p>来源：{case["source"]}；固定噪声种子{case["seed"]}</p>'
            f'<details><summary>条件与原始参考</summary><div class="pages">'
            f'{svg_pages(case["reference"] or case["prefix"])}</div></details>'+"".join(methods)+"</article>")
    summary["sample_stop_counts"] = dict(stops)
    write_json(output/"summary.json", summary)
    options = '<option value="all">全部任务</option>'+''.join(f'<option value="{t}">{t} {NAMES[t]}</option>' for t in TASKS)
    source = json.loads((output/"review_source.json").read_text(encoding="utf-8")) if (output/"review_source.json").exists() else {}
    finished = (summary.get("run_stop") or {}).get("reason") == "input_budget"
    heading = "完整训练结果" if finished else "训练观察"
    window_before = summary.get("windows", {}).get("before_decay")
    window_after = summary.get("windows", {}).get("final_200")
    window_table = ""
    if window_before and window_after:
        comparisons = [("总损失", window_before["mean_loss"], window_after["mean_loss"]),
            ("正文MSE", window_before["body_mse"], window_after["body_mse"]),
            ("真实控制MSE", window_before["control_mse"], window_after["control_mse"]),
            ("重复EOP每槽MSE", window_before["padding_mse"], window_after["padding_mse"])]
        comparisons += [(t+" "+NAMES[t],window_before["task_mse"][t],window_after["task_mse"][t]) for t in TASKS]
        cells = ''.join(f'<tr><td>{label}</td><td>{a:.5f}</td><td>{b:.5f}</td><td>{(b/a-1)*100:+.1f}%</td></tr>' for label,a,b in comparisons if a)
        window_table = '<h3>退火前与训练末尾</h3><p>各取200次更新，正文按实际目标数汇总。两窗使用各自训练批次，反映训练误差走势；此表不是固定验证集的生成能力比较。</p><table><tr><th>指标</th><th>退火前200步</th><th>最后200步</th><th>变化</th></tr>'+cells+'</table>'
    notes_path = output/"observations.html"
    notes = notes_path.read_text(encoding="utf8") if notes_path.exists() else ""
    log_directory = html.escape(summary.get("log_directory", "run"), quote=True)
    late_plot = '<img src="late_training_curves.png" alt="退火段细看">' if (output/"late_training_curves.png").exists() else ""
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CoSE 分块扩散{heading}</title>
    <style>body{{max-width:1450px;margin:30px auto;padding:0 24px;font:16px/1.7 system-ui;color:#17202c;background:#f4f6f8}}table{{border-collapse:collapse;min-width:650px}}th,td{{text-align:left;padding:8px 16px;border-bottom:1px solid #ddd}}
    article,.panel{{background:white;border-radius:12px;padding:24px;margin:24px 0}}h1,h2,h3{{line-height:1.3}}.pages{{display:flex;flex-wrap:wrap;gap:12px}}.page{{width:300px;max-width:100%}}.pageid{{font-size:13px;color:#666}}svg{{width:100%;border:1px solid #ddd}}img{{width:100%}}select{{font-size:16px;padding:8px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
    <h1>CoSE 分块扩散：{heading}</h1><div class="panel">
    <p>检查点第{source.get('step','—')}步，成功原始输入{source.get('successful_inputs',0):,}。
    R从头初始化，冻结原55k自编码器。样张直接解码连续潜变量，固定{source.get('steps',32)}步去噪；蓝色为生成正文，粉色为控制笔画。</p>
    <p>{cases['selection']}所有预选样例依次展示。此处观察生成形态与停止情况，不从32份样张计算准确率。
    每例最多{source.get('max_new_strokes',128)}笔、3页，触及上限的结果原样保留。旧GMM的NLL与这里的MSE尺度不同。</p>
    <p>先查看最终当前权重；另展示每任务首例的EMA，使用相同来源与噪声种子。</p>
    <a href="cases.json">预先固定的来源与前缀</a> · <a href="summary.json">训练与停止摘要</a></div>
    {notes}<div class="panel"><h2>训练曲线</h2><p>短窗平均用于阅读；正文、控制和重复填充分别显示。<a href="{log_directory}/train.jsonl">完整逐步日志</a> · <a href="{log_directory}/run_stop.json">结束记录</a></p>
    <img src="training_curves.png" alt="训练曲线">{late_plot}{window_table}</div><h2>固定样张</h2><select id="filter">{options}</select>
    {''.join(entries)}<script>document.getElementById('filter').onchange=function(){{document.querySelectorAll('article').forEach(a=>a.hidden=this.value!=='all'&&a.dataset.task!==this.value)}};</script></html>'''
    (output/"report.html").write_text(page, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("tmp_smoke/r_joint_statistical_data_20261027"))
    p.add_argument("--font", type=Path, default=Path("datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--prepare_only", action="store_true")
    p.add_argument("--render_only", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="仅工程试运行时限制例数，正式面板为32份")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--block_width", type=int, help="使用检查点支持的原生块长；默认取模型配置")
    p.add_argument("--max_new_strokes", type=int, default=128)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cases = prepare_cases(args.data, args.out, args.font)
    if args.prepare_only:
        print(json.dumps(dict(cases=len(cases["cases"]), objects=cases["objects"], scenes=cases["scenes"])))
        return
    if not args.run:
        p.error("需要本次训练日志目录--run")
    summary = curves(args.run, args.out)
    if args.render_only:
        render(args.out, cases, summary)
        return
    if not args.checkpoint:
        p.error("生成需要完整本轮检查点--checkpoint")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("kind") != "page_diffusion_training_v2":
        p.error("首段对照需要包含当前参数与EMA的完整检查点")
    for key in ("optimizer", "scaler", "stream", "rng", "noise_rng"):
        state.pop(key, None)
    source = dict(checkpoint=str(args.checkpoint.resolve()), sha256=file_sha256(args.checkpoint),
        step=state["step"], successful_inputs=state["counters"]["successful_inputs"],
        training_source=state["source"], created_utc=datetime.now(timezone.utc).isoformat(),
        steps=args.steps, block_width=args.block_width if args.block_width is not None else state["model_cfg"]["block"],
        max_new_strokes=args.max_new_strokes, limit=args.limit,
        cases_sha256=file_sha256(args.out/"cases.json"))
    if source["block_width"] not in (state["model_cfg"].get("training_blocks") or [state["model_cfg"]["block"]]):
        p.error("采样块长须取自此检查点支持的宽度")
    before = args.out/"review_source.json"
    if before.exists():
        old = json.loads(before.read_text(encoding="utf-8"))
        if old.get("block_width",state["model_cfg"]["block"]) != source["block_width"]:
            raise ValueError("原生采样块长改变，请用独立输出目录")
        for name in ("sha256", "steps", "max_new_strokes", "limit", "cases_sha256"):
            if old[name] != source[name]:
                raise ValueError("权重、前缀或采样设置改变，请用独立输出目录")
    else:
        write_json(before, source)
    codec = FrozenPageCodec.from_state(state["codec"]).to("cuda")
    model = PageDiffusionR(PageDiffusionConfig(**state["model_cfg"])).to("cuda").eval()
    selected = cases["cases"][:args.limit] if args.limit else cases["cases"]
    for weights, key in (("raw", "model"), ("ema", "ema")):
        model.load_state_dict(state[key], strict=True)
        generator = PageDiffusionGenerator(model, codec)
        seen = set()
        for case in selected:
            if weights == "ema" and case["task"] in seen:
                continue
            seen.add(case["task"])
            out = args.out/f"{case['id']}_{weights}"
            if (out/"result.json").exists():
                continue
            prefix = [StrokeElement(np.asarray(r["points"], np.float32), r["page_id"], False, r["kind"]) for r in case["prefix"]]
            start = time.monotonic()
            result = generator.generate(prefix, seed=case["seed"], steps=args.steps, block_width=source["block_width"],
                                        max_new_strokes=args.max_new_strokes, max_new_pages=3)
            result["sampling_seconds"] = time.monotonic()-start
            write_json(out/"result.json", result)
            save_native_svg({"strokes":case["prefix"]+[{**r,"generated":True} for r in result["records"]]},
                            out/"pages.svg", case["id"]+" / "+weights)
            print(json.dumps(dict(case=case["id"], weights=weights, stop=result["stop"],
                                  seconds=result["sampling_seconds"])), flush=True)
    render(args.out, cases, summary)


if __name__ == "__main__":
    main()
