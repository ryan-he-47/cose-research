"""2026-10-09：渲染R健康诊断的数值图与回馈样张，CPU。"""
from __future__ import annotations

# 2026-10-12：分类目录中的脚本也可从仓库根直接运行。
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))


import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.review.render_formal_r_review import plot_page, svg_pages


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--density_baseline",type=Path,required=True)
    args = parser.parse_args()
    root = args.diagnosis
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    gradients = read(root/"gradient_probe.json")
    mixture = read(root/"mixture_summary.json")
    cycle = read(root/"cycle_summary.json")
    figure,axes = plt.subplots(2,2,figsize=(13,8.4))
    names = list(gradients["cases"])
    norm = lambda values: sum(v*v for k,v in values.items() if not k.startswith("gmm_"))**.5
    x = np.arange(len(names))
    axes[0,0].bar(x-.19,[norm(gradients["cases"][n]["body_norms"]) for n in names],width=.38,label="正文（已乘任务权重）")
    axes[0,0].bar(x+.19,[norm(gradients["cases"][n]["control_norms"]) for n in names],width=.38,label="控制符（已乘0.05）")
    axes[0,0].set_xticks(x,[n.split("_")[0] for n in names])
    axes[0,0].set_ylabel("裁剪前参数梯度范数")
    axes[0,0].set_title("8任务首回答页诊断批")
    axes[0,0].legend(fontsize=8)
    kinds = ("text","sketch","eop","eod")
    x = np.arange(len(kinds))
    for offset,key,label in ((-.25,"mu_derivative_norm","当前ε=1e-4"),(0,"eps0.01_mu_derivative_norm","只改ε=1e-2"),(.25,"eps0.02_mu_derivative_norm","只改ε=2e-2")):
        axes[0,1].bar(x+offset,[mixture[k][key]["mean"] for k in kinds],width=.25,label=label)
    axes[0,1].set_xticks(x,["文字","图形","EOP","EOD"])
    axes[0,1].set_yscale("log")
    axes[0,1].set_ylabel("单目标 ∂NLL/∂μ 范数均值")
    axes[0,1].set_title("固定μ/π的σ下限敏感性（未训练）")
    axes[0,1].legend(fontsize=8)
    x = np.arange(2)
    for offset,key,label in ((-.25,"dense_rms","原笔→76点"),(0,"cycle_rms","原λ→D→E（76点）"),(.25,"oracle_length_cycle_rms","D→原笔点数→E")):
        axes[1,0].bar(x+offset,[cycle[k+"/all"][key]["mean"] for k in ("text","sketch")],width=.25,label=label)
    axes[1,0].set_xticks(x,["文字3376笔","图形293笔"])
    axes[1,0].set_ylabel("λ每维RMS均值")
    axes[1,0].set_title("点数与解码回馈造成的λ偏移")
    axes[1,0].legend(fontsize=8)
    matrix = read(root/"condition_O2_sketch_only.json")
    values = np.array([[v[c]["first_nll"]-v[r]["first_nll"] for c in matrix] for r,v in matrix.items()])
    extent = max(abs(values).max(),1e-6)
    image = axes[1,1].imshow(values,cmap="RdBu_r",vmin=-extent,vmax=extent)
    axes[1,1].set_xticks(range(4),[n.split("_")[1] for n in matrix])
    axes[1,1].set_yticks(range(4),[n.split("_")[1] for n in matrix])
    axes[1,1].set_xlabel("输入草图类别（指令保持本例）")
    axes[1,1].set_ylabel("真实类别文字回答")
    axes[1,1].set_title("O2首笔NLL相对正确草图的变化")
    for i,j in np.ndindex(values.shape):
        axes[1,1].text(j,i,f"{values[i,j]:+.2f}",ha="center",va="center",fontsize=9)
    figure.colorbar(image,ax=axes[1,1],shrink=.7)
    figure.tight_layout()
    figure.savefig(root/"diagnostic_metrics.png",dpi=160)
    plt.close(figure)
    chosen = ("O1_cat","O2_cat","T1_story0","S2_scene0")
    figure,axes = plt.subplots(4,3,figsize=(10.5,13))
    sections = []
    equality = {}
    for i,name in enumerate(chosen):
        reference = read(args.baseline/name/"reference.json")["strokes"]
        target = [r for r in reference if r["targetable"]]
        curves = {mode:read(root/name/f"{mode}.json") for mode in ("curve","latent")}
        old = read(args.density_baseline/name/"greedy.json")
        equality[name] = [(r["page_id"],r["kind"],r["points"]) for r in curves["curve"]["strokes"]] == [(r["page_id"],r["kind"],r["points"]) for r in old["strokes"]]
        for j,(label,records) in enumerate((("源回答首个目标页",target),*( ("D→E回馈" if mode=="curve" else "直接λ回馈",[r for r in value["strokes"] if r["generated"]]) for mode,value in curves.items()))):
            plot_page(axes[i,j],records)
            axes[i,j].set_title(name+" / "+label,fontsize=9)
        panels = [f'<section><h3>条件</h3>{svg_pages([r for r in curves["curve"]["strokes"] if not r["generated"]])}</section>',
                  f'<section><h3>源回答</h3>{svg_pages(target)}</section>']
        for mode,value in curves.items():
            summary = {k:value[k] for k in ("feedback","stop_reason","generated_strokes","generated_pages")}
            panels.append(f'<section><h3>{"D→E回馈" if mode=="curve" else "直接λ回馈"}</h3><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre>{svg_pages([r for r in value["strokes"] if r["generated"]])}</section>')
        sections.append(f'<h2>{name}</h2><div class="grid">'+"".join(panels)+"</div>")
    figure.suptitle("同一R/AE、最高π解码；直接λ回馈仅作诊断\n黑：源回答，蓝：生成，粉：控制符",fontsize=11)
    figure.tight_layout(rect=(0,0,1,.96))
    figure.savefig(root/"feedback_comparison.png",dpi=150)
    plt.close(figure)
    if not all(equality.values()):
        raise AssertionError(f"默认回馈与本机已有最高π样张不一致：{equality}")
    (root/"render_checks.json").write_text(json.dumps(dict(curve_matches_existing_local_greedy=equality),indent=2),encoding="utf-8")
    (root/"index.html").write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>R训练健康诊断</title>
<style>body{margin:24px;font:15px/1.6 system-ui,"Microsoft YaHei";background:#f5f7fa;color:#18263a}h1{font-size:26px}h2{margin-top:35px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}section{background:white;padding:14px;border:1px solid #dce2e8;border-radius:8px}h3{margin:0 0 12px}.page svg{width:100%;border:1px solid #ddd}.pageid{color:#64748b;font-size:12px}pre{white-space:pre-wrap;font-size:12px}img{width:100%;max-width:1300px}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<h1>R训练健康诊断 · 2026-10-09</h1><p>正式末R step3043 / 冻结AE55000 / 本机FP16；诊断不更新权重。</p><img src="diagnostic_metrics.png"><p>直接λ回馈没有恢复类别或文字生成；下方保留四例全部输出页。</p>'''+"".join(sections)+"</html>",encoding="utf-8")
    print(json.dumps(dict(rendered=True,default_feedback_exact_match=equality)))


if __name__=="__main__":
    main()
