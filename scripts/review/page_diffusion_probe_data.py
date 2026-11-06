"""2026-11-05：冻结 R 诊断用文档；按正式布局、划分与字形增强准备。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.review.joint_eval_batch import load_dataset
from src.st.ae_data import source_in_split
from src.st.generation_pretrain import PageGenerator
from src.st.page_diffusion_codec import FrozenPageCodec, file_sha256
from src.st.page_diffusion_data import document_layout
from src.st.page_diffusion_runtime import autocast, write_json
from src.st.pipeline_pretrain import Story, PretrainDocumentDataset, KIND_IDS
from src.text_augmentation import TextAugmentationConfig
from src.text_normalization import normalize_text
from src.text_renderer import VectorFont

TASKS = ('T1', 'O1', 'O2', 'S1', 'S2', 'S3', 'S4', 'S5')
SEED = 20261105


def valid_stories(root, n):
    """从所有Parquet行范围均匀抽候选，再按正式归一化/去重/哈希保留验证故事。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    paths = sorted(Path(root).rglob('train-*.parquet'))
    files=[pq.ParquetFile(path) for path in paths]
    offsets=np.r_[0,np.cumsum([p.metadata.num_rows for p in files])]
    rng=np.random.default_rng(SEED)
    candidates=np.sort(rng.choice(int(offsets[-1]),min(int(offsets[-1]),max(4096,n*128)),replace=False))
    pools, seen = [], set()
    for fi,(path,file) in enumerate(zip(paths,files)):
        ids=candidates[(candidates>=offsets[fi])&(candidates<offsets[fi+1])]-offsets[fi]
        begin=0
        for group in range(file.num_row_groups):
            end=begin+file.metadata.row_group(group).num_rows
            selected=ids[(ids>=begin)&(ids<end)]
            if not len(selected):begin=end;continue
            values=file.read_row_group(group,columns=['text']).column(0).take(pa.array(selected-begin)).to_pylist()
            for row,raw in zip(selected,values):
                source = f'{path.name}:{int(row)}'
                norm = normalize_text(raw)
                text = norm.text.strip()
                identity = hashlib.sha256(text.encode()).hexdigest()
                if norm.unsupported or not text or identity in seen:
                    continue
                seen.add(identity)
                if source_in_split(text, 'valid', .02):
                    pools.append(Story(text, source))
            begin=end
    if len(pools)<n:raise ValueError(f'验证候选不足：{len(pools)} < {n}')
    return [pools[i] for i in rng.choice(len(pools), n, replace=False)]


@torch.inference_mode()
def encode_documents(documents, geometry, mean, scale, cohort, offset):
    curves = [e.rawxy for d in documents for e in d.to_sequence()]
    # 仅按有效点长分组 E 的执行，不补点、不改变输入笔序。
    order = sorted(range(len(curves)), key=lambda i: min(len(curves[i]), 76))
    encoded = torch.empty(len(curves), 48, dtype=torch.float32)
    for start in range(0, len(order), 384):
        ids = order[start:start+384]
        with autocast(geometry.device):
            z = geometry.encode([curves[i] for i in ids])
        encoded[ids] = (z.float().cpu()-mean)/scale
    result, begin = [], 0
    for number, doc in enumerate(documents, offset):
        seq = doc.to_sequence()
        z = encoded[begin:begin+len(seq)].clone()
        begin += len(seq)
        rng = np.random.default_rng(np.random.SeedSequence([SEED, number, TASKS.index(doc.metadata['task'])]))
        layout = document_layout(doc, prefix_cut=int(rng.integers(16)))
        candidates = [b for b in layout['blocks'] if any(i >= 0 and seq[i].kind in ('text','sketch','selection') for i in b['target'])]
        if not candidates:
            continue
        selected = int(rng.integers(len(candidates)))
        body = [KIND_IDS[e.kind] for e in seq]
        # 连续四个完整正文块用于历史回馈诊断；固定页和位置，隔离停止/对齐影响。
        rollout = []
        for j in range(len(candidates)-3):
            parts = candidates[j:j+4]
            ids = [i for part in parts for i in part['target']]
            if all(i >= 0 and body[i] < 3 for i in ids) and len({b['page'] for b in parts}) == 1:
                if all(parts[k+1]['start'] == parts[k]['start']+16 for k in range(3)):
                    rollout.append(j)
        result.append(dict(id=f'{cohort}_{doc.metadata["task"]}_{number}', cohort=cohort,
            task=doc.metadata['task'], source=str(doc.metadata.get('source_id',doc.metadata.get('task_index',number))),
            label=doc.metadata.get('label'), z=z, pages=torch.tensor([e.page_id for e in seq]),
            local=torch.tensor(layout['local']), kinds=torch.tensor(body), blocks=candidates,
            selected=selected, rollout=rollout, text=[p.text for p in doc.pages if p.text],
            block_inclusion_weight=len(candidates), raw_inputs=len(seq)-1))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--assets', type=Path, default=Path('E:/Trash_things/cose'))
    p.add_argument('--objects_per_class', type=int, default=10)
    p.add_argument('--scenes', type=int, default=512)
    p.add_argument('--stories', type=int, default=1024)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    mean, scale = state['model']['latent_mean'], state['model']['latent_scale']
    codec = FrozenPageCodec.from_state(state['codec'], encode_chunk_size=384).cuda()
    geometry = PageGenerator(codec)
    font_path = args.assets/'datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg'
    font = VectorFont.from_svg(font_path)
    data = args.assets/'tmp_smoke/r_joint_statistical_data_20261027'
    dataset, info = load_dataset(data, 'valid', font)
    dataset.augmentation = TextAugmentationConfig(enabled=True)
    rng = np.random.default_rng(SEED)
    classes = defaultdict(list)
    for item in info['objects']:
        classes[item['label']].append(item)
    objects = [items[i] for _,items in sorted(classes.items())
               for i in rng.choice(len(items), args.objects_per_class, replace=False)]
    scene_ids = rng.choice(info['scene_indices'], args.scenes, replace=False).tolist()
    stories = valid_stories(args.assets/'datasets/datasets--roneneldan--TinyStories/snapshots', args.stories)
    texts = PretrainDocumentDataset(font, stories, seed=42, split='valid', augmentation=TextAugmentationConfig(enabled=True))
    selection = dict(seed=SEED, object_indices=objects, scene_indices=scene_ids,
        stories=[dict(source_id=s.source_id, sha256=hashlib.sha256(s.text.encode()).hexdigest()) for s in stories],
        augmentation='正式字形增强：±10%缩放、6%平滑弯曲',
        selection='每类对象无放回10个来源；场景无放回；文字按正式正文哈希的valid划分，跨4分片。每文档随机完整页窗口和一个正文块。',
        data_manifest_sha256=file_sha256(data/'manifest.json'), font_sha256=file_sha256(font_path))
    write_json(args.out/'selection.json', selection)
    started = time.monotonic()
    for task in TASKS:
        path = args.out/f'valid_{task}.pt'
        if path.exists():
            continue
        ids = list(range(len(stories))) if task=='T1' else [o['index'] for o in objects] if task.startswith('O') else scene_ids
        rows, pending, skipped = [], [], []
        for i,index in enumerate(ids):
            doc = (texts if task=='T1' else dataset).get_for_task(task, int(index))
            if task=='T1':
                windows = list(doc.iter_windows(1024))
                doc = windows[int(rng.integers(len(windows)))]
            if doc.stroke_count > 1024:
                skipped.append(dict(index=int(index), strokes=doc.stroke_count))
                continue
            pending.append(doc)
            if len(pending)>=16 or i==len(ids)-1:
                rows.extend(encode_documents(pending, geometry, mean, scale, 'valid', len(rows)))
                pending=[]
            if (i+1)%256==0:
                print(json.dumps(dict(stage='encode', task=task, documents=i+1, seconds=time.monotonic()-started)),flush=True)
        torch.save(rows,path)
        write_json(path.with_suffix('.json'),dict(task=task,documents=len(rows),skipped=skipped,
            source_units=len(set(x['source'] for x in rows)), blocks=sum(len(x['blocks']) for x in rows)))
        print(json.dumps(dict(stage='saved',task=task,documents=len(rows),seconds=time.monotonic()-started)),flush=True)
    # 已导出的真实训练更新：只做自然配比参照，主体判断来自上面的独立valid来源。
    path=args.out/'natural.pt'
    if not path.exists():
        fixture=args.assets/'tmp_smoke/r_page_diffusion_performance_20261102/actual_batch.pt'
        original=torch.load(fixture,map_location='cpu',weights_only=False,mmap=True)
        rows=[]
        for i in range(0,len(original['documents']),16):
            rows.extend(encode_documents(original['documents'][i:i+16],geometry,mean,scale,'natural',i))
        torch.save(rows,path)
        write_json(path.with_suffix('.json'),dict(documents=len(rows),source=original['source'],
            inputs=original['inputs'],fixture_sha256=file_sha256(fixture),
            scope='65,000余输入的真实训练首批窗口；按可选正文块数进行加权，单独显示。'))
    print(json.dumps(dict(stage='complete',seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    main()
