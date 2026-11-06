"""2026-10-03：新页 LM 的文字续写/O1/O2 样张和本机交互窗口。"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.page_document import StrokeElement, build_document, control_stroke, layout_text_pages, sketch_page
from src.st.generation_pretrain import PageGenerator
from src.st.pipeline_pretrain import build_document_dataset
from src.st.pretrain_model import FrozenSketchLM, config_from_r_checkpoint
from src.text_normalization import normalize_text
from src.text_renderer import VectorFont


def load_generator(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("stage") != "r_pretrain":
        raise ValueError("需要新页协议的 R checkpoint")
    cfg = config_from_r_checkpoint(state)
    model = FrozenSketchLM(cfg, activation_checkpoint=False).eval()
    model.load_state_dict(state["model"], strict=True)
    metadata = {k: state[k] for k in ("step", "data_args", "data_manifest", "ae_source")}
    if 'text_aux_manifest' in state:
        metadata['text_aux_manifest']=state['text_aux_manifest']
    metadata["checkpoint"] = str(Path(checkpoint).resolve())
    max_points = state["train_args"]["max_points"]
    del state
    gc.collect()
    model.to(device)
    font_path = Path(metadata["data_args"]["font_path"])
    if not font_path.exists() and metadata['data_manifest'].get('font_sha256'):
        # 服务器绝对资产路径在本机不成立时，只接受内容完全相同的字体。
        candidate = ROOT/'datasets/vector_fonts/relief'/font_path.name
        if candidate.exists() and hashlib.sha256(candidate.read_bytes()).hexdigest() == metadata['data_manifest']['font_sha256']:
            font_path = candidate
    font = VectorFont.from_svg(font_path)
    return PageGenerator(model, max_points=max_points), font, metadata


def text_prefix(task, text, font, sketch=None):
    if task == "O1":
        pages = layout_text_pages("Draw: " + normalize_text(text).text, font, targetable=False)
        return build_document(pages, complete=False).to_sequence()
    if task == "O2":
        if not sketch:
            raise ValueError("O2 需要草图点列 JSON")
        pages = (*layout_text_pages("Name this sketch:", font, targetable=False),
                 sketch_page(sketch, targetable=False))
        return build_document(pages, complete=False).to_sequence()
    if not text.strip():
        return (StrokeElement(control_stroke("bos"), 0, False, "bos"),)
    pages = layout_text_pages(normalize_text(text).text, font, targetable=False)
    # 续写停在最后一页正文，不事先给这页放 EOP。
    return build_document(pages, complete=False).to_sequence()[:-1]


def source_prefix(document, task):
    sequence = document.to_sequence()
    if task == "T1":
        # 条件截止于首个完整单词（从字符归属确定），不喂余下故事。
        first_page = document.pages[0]
        boundary = len((first_page.text or "").split(" ")[0])
        prefix = [sequence[0]]
        for element in sequence[1:]:
            if element.page_id != first_page.page_id or element.kind != "text" or element.char_index >= boundary:
                break
            prefix.append(element)
        return tuple(prefix)
    boundary = next(i for i, e in enumerate(sequence) if e.targetable)
    return sequence[:boundary]


def save_result(result, output, *, title="Page LM", reference=None):
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        save_native_svg(result,output/'pages.svg',title)
        if reference is not None:
            save_native_svg({'strokes':[{'page_id':e.page_id,'kind':e.kind,'generated':False,
                'points':e.rawxy.tolist()} for e in reference.to_sequence()]},output/'reference.svg','source reference')
        return
    pages = sorted(set(r["page_id"] for r in result["strokes"]))
    columns = min(4, len(pages))
    rows = (len(pages) + columns - 1) // columns
    figure = Figure(figsize=(max(6.4, 3.2 * columns), 3.4 * rows), dpi=150)
    FigureCanvasAgg(figure)
    for index, page_id in enumerate(pages):
        axis = figure.add_subplot(rows, columns, index + 1)
        axis.set(xlim=(0, 1), ylim=(1, 0), aspect="equal", xticks=[], yticks=[])
        axis.axhspan(.90, 1, color="#f1f5f9")
        strokes = [r for r in result["strokes"] if r["page_id"] == page_id]
        for record in strokes:
            points = np.array(record["points"])
            color = "#156e9b" if record["generated"] else "#929292"
            if record["kind"] in ("eop", "eod", "bos"):
                color = "#bc3e70" if record["generated"] else "#929292"
            axis.plot(points[:, 0], points[:, 1], color=color, linewidth=1)
        count = sum(r["generated"] for r in strokes)
        axis.set_title(f"page {page_id} / {count} generated strokes", fontsize=9)
    figure.suptitle(f"{title} | {result['stop_reason']}\ngray: condition, blue: generation, pink: generated control", fontsize=10)
    figure.tight_layout()
    figure.savefig(output / "pages.png", facecolor="white")
    figure.savefig(output / "pages.svg", facecolor="white")
    if reference is not None:
        from scripts.demo.demo_ae_reconstruction import COLORS
        reference_pages = reference.pages[:3]
        ref_figure = Figure(figsize=(3.2 * len(reference_pages), 3.5), dpi=150)
        FigureCanvasAgg(ref_figure)
        for index, page in enumerate(reference_pages):
            axis = ref_figure.add_subplot(1, len(reference_pages), index + 1)
            axis.set(xlim=(0, 1), ylim=(1, 0), aspect="equal", xticks=[], yticks=[])
            for element in page.elements:
                axis.plot(element.rawxy[:, 0], element.rawxy[:, 1], color=COLORS[element.kind], linewidth=1)
            axis.set_title(f"source reference / page {page.page_id}", fontsize=9)
        ref_figure.tight_layout()
        ref_figure.savefig(output / "reference.png", facecolor="white")


def save_native_svg(result, path, title):
    """服务器没有matplotlib时，直接导出可在本机打开的分页矢量图。"""
    import xml.etree.ElementTree as ET
    pages = sorted(set(record['page_id'] for record in result['strokes']))
    columns = min(4,len(pages))
    rows = (len(pages)+columns-1)//columns
    svg = ET.Element('svg',xmlns='http://www.w3.org/2000/svg',
        width=str(columns*320),height=str(rows*350+35),viewBox=f'0 0 {columns*320} {rows*350+35}')
    ET.SubElement(svg,'rect',width='100%',height='100%',fill='white')
    ET.SubElement(svg,'text',x='10',y='22',attrib={'font-size':'14'}).text=title
    for index,page in enumerate(pages):
        x,y=(index%columns)*320+10,(index//columns)*350+55
        group=ET.SubElement(svg,'g',transform=f'translate({x},{y})')
        ET.SubElement(group,'rect',width='300',height='300',fill='white',stroke='#ddd')
        ET.SubElement(group,'text',x='0',y='-8',attrib={'font-size':'12'}).text=f'page {page}'
        for record in result['strokes']:
            if record['page_id'] != page:
                continue
            color=('#bc3e70' if record['kind'] in ('bos','eop','eod') else '#156e9b') if record['generated'] else '#929292'
            points=' '.join(f'{a*300:.4f},{b*300:.4f}' for a,b in record['points'])
            ET.SubElement(group,'polyline',points=points,fill='none',stroke=color,attrib={'stroke-width':'1.2'})
    ET.ElementTree(svg).write(path,encoding='utf-8',xml_declaration=True)


def sampling_options(args):
    return {name:getattr(args,name) for name in ('mixture_temperature','noise_temperature','top_p',
            'candidate_count','cycle_penalty','outside_penalty')}


def run_gui(generator, font, metadata, args):
    import queue
    import threading
    import tkinter as tk
    from tkinter import filedialog, ttk
    root = tk.Tk()
    root.title(f"CoSE 页 LM · step {metadata['step']}")
    task = tk.StringVar(value="O1")
    text = tk.StringVar(value="cat")
    mode = tk.StringVar(value="greedy")
    status = tk.StringVar(value="文字用训练字体排版；O2 可加载草图 JSON。")
    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    ttk.Combobox(top, textvariable=task, values=("O1", "O2", "T1"), state="readonly", width=5).pack(side="left")
    ttk.Entry(top, textvariable=text, width=45).pack(side="left", padx=6)
    ttk.Combobox(top, textvariable=mode, values=("greedy", "sample"), state="readonly", width=8).pack(side="left")
    canvas = tk.Canvas(root, width=610, height=610, background="white")
    canvas.pack()
    controls = ttk.Frame(root, padding=8)
    controls.pack(fill="x")
    ttk.Label(root, textvariable=status, padding=8).pack(fill="x")
    messages, cancel = queue.Queue(), threading.Event()
    view = {"result": None, "page": 0, "sketch": None, "busy": False}

    def redraw(delta=0):
        canvas.delete("all")
        if view["result"] is None:
            return
        pages = sorted(set(r["page_id"] for r in view["result"]["strokes"]))
        view["page"] = max(0, min(view["page"] + delta, len(pages) - 1))
        page = pages[view["page"]]
        canvas.create_rectangle(10, 10, 600, 600, outline="#aaa")
        for r in view["result"]["strokes"]:
            if r["page_id"] != page:
                continue
            points = np.array(r["points"]) * 590 + 10
            color = "#156e9b" if r["generated"] else "#929292"
            if r["generated"] and r["kind"] in ("eop", "eod"):
                color = "#bc3e70"
            canvas.create_line(*points.flatten().tolist(), fill=color, width=1.6)
        canvas.create_text(305, 595, text=f"第 {page} 页", fill="#999")

    def load_sketch():
        path = filedialog.askopenfilename(filetypes=[("草图点列 JSON", "*.json")])
        if path:
            try:
                view["sketch"] = read_sketch(path)
                task.set("O2")
                status.set(f"已载入 {len(view['sketch'])} 笔")
            except Exception as error:
                status.set(str(error))

    def start():
        if view["busy"]:
            return
        try:
            prefix = text_prefix(task.get(), text.get(), font, view["sketch"])
        except Exception as error:
            status.set(str(error))
            return
        cancel.clear()
        view.update(busy=True, result={"strokes": [{"page_id": e.page_id, "kind": e.kind,
                         "points": e.rawxy.tolist(), "generated": False} for e in prefix]}, page=0)
        redraw()
        status.set("生成中…")
        selected_mode, selected_task = mode.get(), task.get()

        def worker():
            try:
                result = generator.generate(prefix, max_new_strokes=args.max_new_strokes,
                    max_new_pages=args.max_new_pages, mode=selected_mode, temperature=args.temperature,
                    seed=args.seed, callback=lambda r: messages.put(("stroke", r)), cancelled=cancel.is_set,
                    **sampling_options(args))
                result.update(checkpoint=metadata["checkpoint"], checkpoint_step=metadata["step"], task=selected_task)
                output = ROOT / "tmp_smoke" / ("page_generation_gui_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                save_result(result, output, title=f"{selected_task} / step {metadata['step']}")
                messages.put(("done", (result, output)))
            except Exception as error:
                messages.put(("error", str(error)))
        threading.Thread(target=worker, daemon=True).start()

    def drain():
        while not messages.empty():
            kind, value = messages.get_nowait()
            if kind == "stroke":
                view["result"]["strokes"].append(value)
                view["page"] = len(set(r["page_id"] for r in view["result"]["strokes"])) - 1
                redraw()
            elif kind == "done":
                result, output = value
                view.update(busy=False, result=result)
                status.set(f"{result['generated_strokes']} 笔，{result['stop_reason']}；已保存 {output.name}")
                redraw()
            else:
                view["busy"] = False
                status.set(value)
        root.after(100, drain)

    ttk.Button(controls, text="生成", command=start).pack(side="left")
    ttk.Button(controls, text="停止", command=cancel.set).pack(side="left", padx=6)
    ttk.Button(controls, text="加载草图", command=load_sketch).pack(side="left")
    ttk.Button(controls, text="上一页", command=lambda: redraw(-1)).pack(side="right")
    ttk.Button(controls, text="下一页", command=lambda: redraw(1)).pack(side="right", padx=6)
    root.protocol("WM_DELETE_WINDOW", lambda: (cancel.set(), root.destroy()))
    root.after(100, drain)
    root.mainloop()


def read_sketch(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data["strokes"]
        data = [r["points"] for r in data if r.get("kind", "body") in ("body", "sketch")]
    return [np.asarray(points, dtype=np.float32) for points in data]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--task", choices=("T1", "O1", "O2"), default="O1")
    parser.add_argument("--text", default="cat")
    parser.add_argument("--sketch_json", type=Path)
    parser.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=.5)
    parser.add_argument('--mixture_temperature',type=float,default=None,help='分量选择温度；0取最大π；缺省沿用原π')
    parser.add_argument('--noise_temperature',type=float,default=None,help='分量内噪声；缺省沿用temperature')
    parser.add_argument('--top_p',type=float,default=1.)
    parser.add_argument('--candidate_count',type=int,default=1,help='sample模式的D→E闭环候选数')
    parser.add_argument('--cycle_penalty',type=float,default=1.)
    parser.add_argument('--outside_penalty',type=float,default=1.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_strokes", type=int, default=128)
    parser.add_argument("--max_new_pages", type=int, default=4)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--demo", action="store_true", help="同一 valid 源生成对象/文字样本；正式checkpoint追加S1–S5")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.demo and args.gui:
        parser.error("demo/gui 选一个")
    torch.set_num_threads(4)
    generator, font, metadata = load_generator(args.ckpt, args.device)
    if args.gui:
        run_gui(generator, font, metadata, args)
        return
    output = args.out_dir or ROOT / "tmp_smoke" / ("page_generation_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if args.demo:
        output.mkdir(parents=True, exist_ok=False)
        dataset = build_document_dataset(**{**metadata["data_args"], "split": "valid"})
        examples = [("O1", label, dataset.drawings.classes.index(label))
                    for label in ("cat", "fish", "airplane", "bicycle")]
        examples += [("O2", "cat", dataset.drawings.classes.index("cat")), ("T1", "story", 0)]
        if metadata['data_args'].get('formal'):
            examples += [(f'S{i}','scene',0) for i in range(1,6)]
        results = []
        for task, label, index in examples:
            document = dataset.get_for_task(task, index)
            result = generator.generate(source_prefix(document, task), mode=args.mode,
                temperature=args.temperature, seed=args.seed, max_new_strokes=args.max_new_strokes,
                max_new_pages=args.max_new_pages,**sampling_options(args))
            result.update(task=task, source=document.metadata, checkpoint_step=metadata["step"],
                          checkpoint=metadata["checkpoint"])
            save_result(result, output / f"{task}_{label}", title=f"{task}: {label} / step {metadata['step']}",
                        reference=document)
            summary = {k: v for k, v in result.items() if k not in ("strokes", "source")}
            results.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        (output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        sketch = read_sketch(args.sketch_json) if args.sketch_json else None
        prefix = text_prefix(args.task, args.text, font, sketch)
        result = generator.generate(prefix, mode=args.mode, temperature=args.temperature,
            seed=args.seed, max_new_strokes=args.max_new_strokes, max_new_pages=args.max_new_pages,**sampling_options(args))
        result.update(task=args.task, input_text=args.text, checkpoint_step=metadata["step"],
                      checkpoint=metadata["checkpoint"])
        save_result(result, output, title=f"{args.task}: {args.text} / step {metadata['step']}")
        print(json.dumps({k: v for k, v in result.items() if k != "strokes"}, ensure_ascii=False))
    print("输出：", output.resolve())


if __name__ == "__main__":
    main()
