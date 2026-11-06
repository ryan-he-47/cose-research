"""2026-10-07：正式RAW/场景配方的人工审核页；调用训练同款文档构建器。"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import partial
import html
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.formal_scene_tasks import CAPTION_VERSION, _caption
from src.page_document import build_document, layout_text_pages
from src.text_augmentation import TextAugmentationConfig, augment_rendered_text
from src.text_renderer import VectorFont
from src.st.ae_data import RawDrawing
from src.st.formal_data import FormalDocumentDataset
from src.st.pipeline_pretrain import Story


class ReviewDrawings:
    def __init__(self,bundle):
        self.classes = bundle['raw_manifest']['classes']
        self.info = bundle['raw_manifest']
        self.records = {int(key):value for key,value in bundle['records'].items()}
        self.class_indices = [[] for _ in self.classes]
        for index,record in self.records.items():
            self.class_indices[self.classes.index(record['word'])].append(index)

    def __len__(self):
        return len(self.records)

    def get(self,index):
        record = self.records[int(index)]
        strokes = tuple(np.column_stack((s[0],s[1])).astype(np.float32) for s in record['drawing'])
        return RawDrawing(strokes,record['word'],str(record['key_id']))

    def manifest(self):
        return self.info


class ReviewRecipes:
    def __init__(self,recipes):
        self.recipes = recipes

    def __getitem__(self,index):
        return self.recipes[index]


def page_svg(page,prefix=()):
    paths = []
    for element in (*prefix,*page.elements):
        color = '#2563eb' if element.kind in ('bos','eop','eod') else '#a21caf' if element.kind == 'selection' else '#151b26'
        path = 'M'+' L'.join(f'{x*400:.2f},{y*400:.2f}' for x,y in element.rawxy)
        paths.append(f'<path d="{path}" stroke="{color}"/>')
    return '<svg viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg"><rect width="400" height="400" fill="white"/><g fill="none" stroke-width="1.2" stroke-linejoin="round" stroke-linecap="round">'+''.join(paths)+'</g></svg>'


def document_html(document):
    pages = []
    for page in document.pages:
        role = '预测目标' if page.targetable else '条件'
        prefix = tuple(e for e in document.prefix if e.page_id == page.page_id)
        pages.append(f'<figure><figcaption>页{page.page_id} · {role} · {page.modality} · {len(page.elements)}笔</figcaption>'+
                     page_svg(page,prefix)+f'<div class="page-text">{html.escape(page.text or "")}</div></figure>')
    meta = document.metadata
    question = html.escape(str(meta.get('question','')))
    answer = html.escape(str(meta.get('answer',meta.get('label',''))))
    return f'<div class="question">{question}</div><div class="answer">答案：{answer}</div><div class="pages">'+''.join(pages)+'</div>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=ROOT/'tmp_smoke/ae_finish_20261007/formal_review_inputs.json')
    parser.add_argument('--out_dir',type=Path,default=ROOT/'tmp_smoke/formal_pipeline_review_20261007')
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    bundle = json.loads(args.bundle.read_text(encoding='utf-8'))
    font_path = ROOT/'datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg'
    font = VectorFont.from_svg(font_path)
    source = ReviewDrawings(bundle)
    dataset = FormalDocumentDataset(font,(),source,font_path=font_path,seed=42,
                                    augmentation=TextAugmentationConfig())
    dataset.recipes = ReviewRecipes(bundle['recipes'])
    dataset.recipe_count = len(bundle['recipes'])
    cards, records, lengths, query_kinds = [], [], [], Counter()
    for i,recipe in enumerate(bundle['recipes']):
        docs = {task:dataset.get_for_task(task,i) for task in ('S1','S2','S3','S4','S5')}
        objects = docs['S1'].metadata['objects']
        counts = Counter(obj['label'] for obj in objects)
        inventory = ', '.join(f'{label} × {count}' for label,count in counts.items())
        caption = docs['S1'].metadata['answer']
        object_rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in
            (obj['instance_id'],obj['label'],obj['position'],tuple(round(x,3) for x in obj['bbox'])))+'</tr>' for obj in objects)
        details = []
        for task,doc in docs.items():
            sequence = doc.to_sequence()
            assert sequence[-2].kind == 'eod' and sequence[-1].kind == 'eop'
            assert all(all(e.page_id == page.page_id for e in page.elements) for page in doc.pages)
            query_kinds[doc.metadata.get('query',{}).get('kind',doc.metadata.get('query_kind',task))] += 1
            lengths.append(doc.stroke_count)
            records.append({'task':task,'index':i,'strokes':doc.stroke_count,'metadata':doc.metadata,
                            'pages':[{'id':p.page_id,'kind':p.modality,'target':p.targetable,'text':p.text} for p in doc.pages]})
            details.append(f'<details data-task="{task}"><summary>{task} · {len(doc.pages)}页 · {doc.stroke_count}笔</summary>'+document_html(doc)+'</details>')
        scene_page = next(page for page in docs['S1'].pages if page.modality == 'sketch')
        ring_meta = docs['S4'].metadata
        rings = docs['S4'].pages[-1]
        # 叠加图仅方便核对圈选坐标，下面实际训练文档仍为独立圈选页。
        from dataclasses import replace
        ring_elements = tuple(replace(e,page_id=scene_page.page_id) for e in rings.elements if e.kind == 'selection')
        overlay = page_svg(scene_page,ring_elements)
        cards.append(f'<section id="scene-{i}"><h2>场景{i} · {html.escape(inventory)}</h2><div class="overview"><div>'+overlay+
            f'<p>核对圈选：{html.escape(ring_meta["question"])}</p></div><div><p class="caption">{html.escape(caption)}</p>'+
            f'<p>简略caption：{html.escape(_caption(objects,detailed=False))}</p><table><tr><th>实例</th><th>类别</th><th>粗位置</th><th>bbox</th></tr>{object_rows}</table></div></div>'+''.join(details)+'</section>')
    extras = []
    for task in ('O1','O2'):
        cls = next(i for i,entries in enumerate(source.class_indices) if entries)
        doc = dataset.get_for_task(task,cls)
        extras.append(f'<details><summary>{task} 对象页样例</summary>'+document_html(doc)+'</details>')
    story = Story(**bundle['stories'][0])
    augment = partial(augment_rendered_text,rng=np.random.default_rng(np.random.SeedSequence([42,0,0])),config=TextAugmentationConfig())
    doc = build_document(layout_text_pages(story.text,font,augment=augment),metadata={'task':'T1'},complete=True)
    window = next(doc.iter_windows(1024))
    extras.append('<details><summary>T1 正式TinyStories首篇 · 第一个1024笔画窗口</summary>'+document_html(window)+'</details>')
    summary = {'caption_version':CAPTION_VERSION,'selection':bundle['selection'],'scenes':len(bundle['recipes']),
               'source_drawings':len(source),'scene_documents':len(records),'min_strokes':min(lengths),
               'max_strokes':max(lengths),'over_context':sum(x>1024 for x in lengths),
               'object_count_distribution':dict(Counter(len(r['objects']) for r in bundle['recipes'])),
               'repeated_scenes':sum(len({o['label'] for o in r['objects']})<len(r['objects']) for r in bundle['recipes']),
               'query_kinds':dict(query_kinds),'source_manifest':bundle['source_manifest']}
    (args.out_dir/'samples.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.out_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    css = 'body{font-family:system-ui;margin:0;background:#f4f6f8;color:#182434}main{max-width:1200px;margin:auto;padding:28px}header{position:sticky;top:0;background:#f4f6f8;padding:12px 0;z-index:1}section{background:white;padding:24px;border-radius:12px;margin:22px 0}h1{font-size:25px}h2{font-size:19px}.overview{display:grid;grid-template-columns:360px 1fr;gap:30px}.overview svg{width:330px;border:1px solid #ddd}.caption{font-size:19px;line-height:1.6}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #dce3ea;text-align:left}details{margin:15px 0}summary{cursor:pointer;padding:10px;background:#edf2f7;border-radius:5px}.pages{display:flex;overflow:auto;gap:12px;padding:12px 0}figure{margin:0;width:235px;flex:0 0 235px}figure svg{width:235px;border:1px solid #ddd}figcaption,.page-text{font-size:12px;line-height:1.5}.page-text{padding:5px}.answer,.question{margin:8px;overflow-wrap:anywhere}button{padding:6px 14px;margin-right:8px;cursor:pointer}select{padding:6px}@media(max-width:700px){.overview{grid-template-columns:1fr}}'
    controls = '<select id="sceneJump"><option value="">跳到场景</option>'+''.join(f'<option value="scene-{i}">{i}</option>' for i in range(len(cards)))+'</select>'
    controls += ''.join(f'<button onclick="showTask(\'{task}\')">展开{task}</button>' for task in ('S1','S2','S3','S4','S5'))
    controls += '<button onclick="showTask(\'\')">收起全部</button>'
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>正式R场景与caption审核</title><style>'+css+'</style><main><h1>正式R场景与caption审核 · 2026-10-07</h1><p>正式训练池前24个配方，未筛选外观。黑色为正文，蓝色为控制笔画，紫色为圈选。概览叠加圈只用于核对；实际训练圈选保持独立页。</p><p>先核对caption的类别、数量和空间描述，再展开任务检查指令、答案与分页。字形增强保持训练设置。</p><header>'+controls+'</header>'+''.join(extras)+''.join(cards)+'</main><script>function showTask(t){document.querySelectorAll("details").forEach(d=>d.open=!!t&&d.dataset.task===t)}document.getElementById("sceneJump").onchange=e=>{if(e.target.value)document.getElementById(e.target.value).scrollIntoView()}</script></html>'
    (args.out_dir/'index.html').write_text(page,encoding='utf-8')
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import textwrap
    figure = Figure(figsize=(14,10),dpi=150)
    FigureCanvasAgg(figure)
    for i in range(6):
        document = dataset.get_for_task('S1',i)
        scene = next(p for p in document.pages if p.modality == 'sketch')
        axis = figure.add_subplot(2,3,i+1)
        axis.set_xlim(0,1); axis.set_ylim(1,0); axis.set_aspect('equal')
        axis.set_xticks([]); axis.set_yticks([])
        for element in scene.elements:
            axis.plot(element.rawxy[:,0],element.rawxy[:,1],color='#2563eb' if element.kind in ('eop','eod') else '#171717',linewidth=1)
        axis.set_title(f'Scene {i}',fontsize=11)
        axis.set_xlabel('\n'.join(textwrap.wrap(document.metadata['answer'],64)),fontsize=8)
    figure.tight_layout(pad=2.0,h_pad=3.5)
    figure.savefig(args.out_dir/'overview.png',facecolor='white')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
