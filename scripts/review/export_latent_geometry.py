"""2026-11-03：从绑定提交的 AE/R 训练流导出只读几何样本；不加载到 GPU。"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def bind(commit, loader):
    spec = importlib.util.spec_from_file_location('_geometry_git_loader', loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path.cwd()
    sys.path.insert(0, str(root))
    sys.meta_path.insert(0, module.GitSource(root, commit))


class Reservoir:
    def __init__(self, capacity, seed):
        self.capacity, self.rng = capacity, np.random.default_rng(seed)
        self.rows, self.seen, self.counts = [], 0, Counter()

    def add(self, points, *, group, kind, task='', unit='', label='', target=False, window=0):
        self.seen += 1
        self.counts[f'{group}:{kind}'] += 1
        index = self.seen-1 if self.seen <= self.capacity else int(self.rng.integers(self.seen))
        if index >= self.capacity:
            return
        row = (np.asarray(points, np.float32).copy(), group, kind, task, str(unit), label, target, window)
        if index == len(self.rows):
            self.rows.append(row)
        else:
            self.rows[index] = row

    def save(self, path):
        points, group, kind, task, unit, label, target, window = zip(*self.rows)
        lengths = np.asarray([len(x) for x in points], np.int32)
        temporary=Path(path).with_name(Path(path).stem+'.tmp.npz')
        np.savez_compressed(temporary, points=np.concatenate(points),
            offsets=np.r_[0, np.cumsum(lengths)], group=np.asarray(group), kind=np.asarray(kind),
            task=np.asarray(task), unit=np.asarray(unit), label=np.asarray(label),
            target=np.asarray(target, bool), window=np.asarray(window, np.int16))
        temporary.replace(path)
        return dict(seen=self.seen, kept=len(self.rows), population_counts=dict(self.counts),
                    sample_counts=dict(Counter(f'{r[1]}:{r[2]}' for r in self.rows)),
                    sampling='uniform_without_replacement_reservoir', sha256=digest(path))


def export_ae(state, args, out):
    from src.st.ae_direct import DirectAEStream
    saved = state['data']
    data_args = state['data_args']
    fractions = (np.arange(args.windows)+.5)/args.windows
    cursors, counts = [], Counter()
    stream = DirectAEStream(data_args, batch_size=saved['batch_size'], workers=0)
    # 历史启动配方覆盖过模块默认份额。检查点的实际整数配额是本次复现来源。
    stream.quotas = dict(saved['quotas'])
    try:
        for window, fraction in enumerate(fractions):
            stream.cache.clear()
            cursor = {**saved, 'indices': {k:int(v*fraction) for k,v in saved['indices'].items()},
                      'offsets': {k:0 for k in saved['offsets']}}
            stream.load_state_dict(cursor)
            start = stream.state_dict()
            groups = {}
            original_next = stream._next
            def tagged(name):
                record = original_next(name)
                groups[id(record)] = name
                return record
            stream._next = tagged
            for _ in range(args.ae_batches):
                groups.clear()
                records = stream.next_batch()
                for record in records:
                    group = groups[id(record)]
                    label = stream.dataset.drawings.classes[record.document_index % len(stream.dataset.drawings.classes)] if group == 'object' else ''
                    out.add(record.rawxy, group=group, kind=record.kind, label=label,
                            unit=f'{group}:{record.document_index}', target=True, window=window)
                    counts[group] += 1
            stream._next = original_next
            cursors.append(dict(fraction=float(fraction), start=start, end=stream.state_dict()))
            print(json.dumps(dict(mode='ae', window=window, seen=out.seen)), flush=True)
    finally:
        stream.close()
    return dict(data_args=data_args, actual_quotas=saved['quotas'], counts=dict(counts), cursors=cursors)


def export_r(state, args, out):
    from src.st.pipeline_pretrain import build_document_dataset
    from src.st.formal_stream import InputDocumentStream, collect_documents
    from src.st.page_diffusion_data import prepare_update
    from src.st.ae_data import cap_stroke
    recipe, data_args, saved = state['recipe'], state['data_args'], state['stream']
    dataset = build_document_dataset(**data_args)
    stream = InputDocumentStream(dataset, data_args, recipe['input_shares'],
                                 max_strokes=recipe['context'], workers=0)
    fractions = (np.arange(args.windows)+.5)/args.windows
    noise = torch.Generator().manual_seed(args.seed+100003)
    cursors, totals = [], Counter()
    try:
        for window, fraction in enumerate(fractions):
            stream.windows.clear()
            cursor = {**saved, 'indices': {k:int(v*fraction) for k,v in saved['indices'].items()},
                      'offsets': {k:0 for k in saved['offsets']},
                      'inputs': {k:0 for k in saved['inputs']}, 'skipped':{}, 'skip_examples':[]}
            stream.load_state_dict(cursor)
            start = stream.state_dict()
            for _ in range(args.r_updates):
                documents, inputs, _, _ = collect_documents(stream, recipe['input_budget'])
                counts = prepare_update(documents, noise, recipe['model']['block'])
                totals.update({f'inputs:{k}':v for k,v in inputs.items()})
                totals.update({f'targets:{k}':v for k,v in counts['body'].items()})
                totals['padding_slots'] += counts['padded']
                totals['control_targets'] += counts['control']
                for doc in documents:
                    task = doc.metadata['task']
                    targets = {i for part in doc.metadata['_page_diffusion']['blocks'] for i in part['target'] if i >= 0}
                    source = doc.metadata.get('source_id', doc.metadata.get('recipe_id', 'unknown'))
                    for index, element in enumerate(doc.to_sequence()):
                        group = 'text' if element.kind == 'text' else (
                            ('object' if task.startswith('O') else 'scene') if element.kind == 'sketch' else 'aux')
                        out.add(cap_stroke(element.rawxy, 76), group=group, kind=element.kind,
                                task=task, unit=f'{task[0]}:{source}', label=str(doc.metadata.get('label','')),
                                target=index in targets, window=window)
            cursors.append(dict(fraction=float(fraction), start=start, end=stream.state_dict()))
            print(json.dumps(dict(mode='r', window=window, seen=out.seen)), flush=True)
    finally:
        stream.close()
    return dict(data_args=data_args, recipe=recipe, counts=dict(totals), cursors=cursors,
                input_view='all_real_encoded_strokes_including_context_and_controls',
                target_view='actual_real_targets_after_page_block_prefix_cut; synthetic_padding_excluded')


def export_text(state,args,out):
    from src.st.ae_direct import source_block
    from src.st.pipeline_pretrain import build_document_dataset
    dataset=build_document_dataset(**state['data_args'])
    rng=np.random.default_rng(args.seed)
    ids=rng.choice(state['data']['indices']['text'],1024,replace=False)
    for number,index in enumerate(ids):
        records=source_block(dataset,'text',int(index))
        selected=rng.choice(len(records),min(32,len(records)),replace=False)
        for i in sorted(selected):
            record=records[i]
            out.add(record.rawxy,group='text',kind='text',unit=f'text:{index}',target=True)
        if (number+1)%128==0:
            print(json.dumps(dict(mode='text',documents=number+1,seen=out.seen)),flush=True)
    return dict(data_args=state['data_args'],source_indices=ids.tolist(),documents=len(ids),
                selection='1024 independent consumed AE text sources, up to 32 uniform strokes per source; stratified text view, not natural mixture')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('ae','r','text'), required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--git_loader', type=Path, required=True)
    p.add_argument('--windows', type=int, default=8)
    p.add_argument('--ae_batches', type=int, default=8)
    p.add_argument('--r_updates', type=int, default=2)
    p.add_argument('--capacity', type=int, default=262144)
    p.add_argument('--seed', type=int, default=20261103)
    args = p.parse_args()
    if not os.getenv('SLURM_JOB_ID') and os.getenv("COSE_REQUIRE_SLURM") == "1":
        p.error('服务器抽样必须经 Slurm')
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/f'{args.mode}_pipeline.npz'
    if path.exists():
        p.error('输出已存在，请保留旧产物并使用新目录')
    torch.set_num_threads(1)
    started = time.monotonic()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    commit = state['source']['code_commit'] if args.mode == 'r' else state['source_commit']
    if args.mode == 'r' and 'recipe' not in state:
        raise ValueError('R 检查点缺少实际配方')
    metadata = dict(mode=args.mode, checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint),
                    pipeline_commit=commit, export_commit=os.getenv('COSE_CODE_COMMIT'),
                    job=os.getenv('SLURM_JOB_ID'), seed=args.seed, step=state['step'],
                    cursor_selection='stratified_windows_across_consumed_source_cursor_ranges_not_exact_step_replay')
    for key in ('model','optimizer','ema','scaler','rng','noise_rng'):
        state.pop(key, None)
    gc.collect()
    bind(commit, args.git_loader)
    reservoir = Reservoir(args.capacity, args.seed)
    details = {'ae':export_ae,'r':export_r,'text':export_text}[args.mode](state,args,reservoir)
    metadata.update(details)
    metadata.update(reservoir.save(path))
    metadata['seconds'] = time.monotonic()-started
    (args.out/f'{args.mode}_pipeline.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(mode=args.mode, seen=metadata['seen'], kept=metadata['kept'], seconds=metadata['seconds'])),flush=True)


if __name__ == '__main__':
    main()
