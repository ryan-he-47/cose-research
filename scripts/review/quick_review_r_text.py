"""2026-10-15：本机短验收，当前R在线教师强制与标准笔画/GMM闭环，零训练更新。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
from dataclasses import asdict
import html
import json
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from scripts.demo.demo_text_readout import load_joint_readout
from scripts.demo.demo_pretrain_generation import save_native_svg, text_prefix
from scripts.review.diagnose_formal_r_health import model_hash
from src.st.text_auxiliary import validate_auxiliary
from src.st.text_readout_generation import generate_text_readout


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()


def restore_rng(state):
    random.setstate(state[0]); np.random.set_state(state[1])
    torch.set_rng_state(state[2]); torch.cuda.set_rng_state_all(state[3])


def render(out, summary):
    rows = []
    for name, v in summary["teacher_forcing"]["text_aux"].items():
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{v['count']}</td>"
                    f"<td>{v['shape_ce']:.4f}</td><td>{v['shape_accuracy']:.2%}</td>"
                    f"<td>{v['space_accuracy']:.2%}</td></tr>")
    cases = []
    for c in summary["cases"]:
        svg = (out / c["svg"]).read_text(encoding="utf-8")
        page_texts = c.get('generated_page_texts', [])
        if isinstance(page_texts, dict):
            page_texts = [v for k, v in sorted(page_texts.items(), key=lambda kv: int(kv[0]))]
        cases.append(f"<section><h2>{html.escape(c['method'])} / {c['index']}</h2>"
                     f"<p>前文：{html.escape(c['prompt'])}</p><p>停止：{html.escape(c['stop_reason'])}</p>"
                     f"<pre>{html.escape(' | '.join(page_texts))}</pre>{svg}</section>")
    page = f"""<!doctype html><meta charset="utf-8"><title>R step {summary['step']} 短验收</title>
<style>body{{font:16px/1.6 system-ui;margin:28px;max-width:1400px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #bbb;padding:6px 14px}}section{{border-top:2px solid #ccc;margin-top:24px}}
svg{{width:100%;height:auto}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:12px}}</style>
<h1>R step {summary['step']} 短验收</h1><p>{summary['teacher_forcing']['stories']}篇原验证故事；当前R在线FP16，标准字体、真实前缀，
准确率来自原始logits。形状类准确率不是整字正确率。</p>
<table><tr><th>分桶</th><th>目标数</th><th>形状CE</th><th>形状准确率</th><th>逐笔空格准确率</th></tr>
{''.join(rows)}</table><p>分类闭环沿用字体前缀/控制符约束、标准笔画回馈；同形落点由GMM选。
GMM闭环用原最高π均值解码/重编码，未加字形约束。无外部词典或语言模型。</p>{''.join(cases)}"""
    (out / "index.html").write_text(page, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--valid", type=Path, default=ROOT / "tmp_smoke/text_readout_assets_20261013/valid.jsonl")
    p.add_argument("--generation_strokes", type=int, default=240)
    p.add_argument("--gmm_strokes", type=int, default=160)
    p.add_argument("--max_seconds", type=float, default=600)
    p.add_argument("--gradient_probe", action="store_true", help="四篇既有增强文档，只求导不更新")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "summary.json").exists():
        raise FileExistsError("已有验收摘要，使用独立输出目录")
    torch.set_num_threads(4)
    start = time.monotonic(); deadline = start + args.max_seconds
    generator, head, vocab, layout, source = load_joint_readout(args.checkpoint)
    generator.model.encode_chunk_size = 128
    before = source["r_model_hash"]
    report = validate_auxiliary(generator.model, vocab.font, head.catalog, args.valid, asdict(layout),
                                generator.device, rng_state, restore_rng)
    first, cont = report['text_aux']['first'], report['text_aux']['continuation']
    n = first['count'] + cont['count']
    text = dict(count=n, space_positive=first['space_positive'] + cont['space_positive'])
    for key in ('shape_ce', 'space_bce', 'shape_error', 'space_error'):
        text[key] = (first[key] * first['count'] + cont[key] * cont['count']) / max(1, n)
    report['text_aux']['text'] = text
    for v in report["text_aux"].values():
        v["shape_accuracy"] = 1 - v["shape_error"]
        v["space_accuracy"] = 1 - v["space_error"]
        v['no_space_baseline_accuracy'] = 1 - v['space_positive'] / v['count']
    write(args.out / "teacher_forcing.json", report)
    print(json.dumps({"teacher_forcing": report}, ensure_ascii=False), flush=True)
    gradient_summary = None
    if args.gradient_probe:
        import statistics
        from scripts.review.review_r_text_aux import gradient_observation
        rows = [json.loads(line) for line in args.valid.read_text(encoding='utf-8').splitlines() if line.strip()]
        gradient_observation(generator, head, vocab, layout, rows,
                             generator.model.cfg.r_text_aux_weight, args.out, 'current', loss_scale=8.)
        probe = json.loads((args.out / 'gradients_current.json').read_text(encoding='utf-8'))
        gradient_summary = dict(alpha=probe['alpha'], median_ratios={
            group: statistics.median(r['groups'][group]['weighted_aux_to_geometry'] for r in probe['rows'])
            for group in ('hidden', 'R_backbone')})
        print(json.dumps({'gradient_probe': gradient_summary}), flush=True)
    prompts = ["Once upon a time, there was a little cat named Tom.",
               "Once upon a time, there was a little dog named Max.",
               "Lily was hungry. She went to the kitchen to get",
               "Tom dropped his toy into the lake. He felt"]
    cases = []
    for i, prompt in enumerate(prompts):
        value = generate_text_readout(generator, head, vocab, prompt, deadline,
                                     SimpleNamespace(constraint="prefix", generation_strokes=args.generation_strokes), layout)
        name = f"readout{i}"; write(args.out / f"{name}.json", value)
        save_native_svg(value, args.out / f"{name}.svg", f"R / shape+space / {i}")
        case = dict(index=i, method="形状/空格分类闭环", prompt=prompt, stop_reason=value["stop_reason"],
                    generated_page_texts=value["generated_page_texts"], svg=f"{name}.svg")
        cases.append(case); print(json.dumps(case, ensure_ascii=False), flush=True)
    for i, prompt in enumerate(prompts[:2]):
        if time.monotonic() >= deadline:
            raise TimeoutError("短验收到达工程时间上限")
        value = generator.generate(text_prefix("T1", prompt, vocab.font), mode="greedy", temperature=.5,
                                   seed=42, max_new_strokes=args.gmm_strokes, max_new_pages=4)
        name = f"gmm{i}"; write(args.out / f"{name}.json", value)
        save_native_svg(value, args.out / f"{name}.svg", f"R / GMM greedy / {i}")
        cases.append(dict(index=i, method="GMM闭环", prompt=prompt, stop_reason=value["stop_reason"], svg=f"{name}.svg"))
        print(json.dumps(cases[-1], ensure_ascii=False), flush=True)
    after = model_hash(generator.model)
    assert before == after, "推理前后权重不同"
    summary = dict(step=source['step'], checkpoint=str(args.checkpoint.resolve()), teacher_forcing=report, cases=cases,
                   elapsed_seconds=time.monotonic()-start, optimizer_updates=0, weights_unchanged=True,
                   model_hash=after, generation_strokes=args.generation_strokes, gmm_strokes=args.gmm_strokes)
    summary['gradient_probe'] = gradient_summary
    write(args.out / "summary.json", summary); render(args.out, summary)
    print(json.dumps({"complete": True, "step": summary["step"], "seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
