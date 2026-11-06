"""2026-10-09：同一正式R和原验收条件，试全混合密度众数解码。"""
from __future__ import annotations

# 2026-10-12：分类目录中的脚本也可从仓库根直接运行。
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))


import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from scripts.demo.demo_pretrain_generation import load_generator, save_native_svg
from scripts.review.review_formal_r import write_json
from src.page_document import StrokeElement
from src.st.generation_pretrain import PageKVCache
from src.st.model import gmm_nll
from src.st.sampling import gmm_density_mode


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@torch.no_grad()
def first_distribution(generator, prefix):
    with generator.autocast():
        emb = generator.encode([e.rawxy for e in prefix]).unsqueeze(0)
        ids = torch.tensor([[e.page_id for e in prefix]], device=generator.device)
        hidden = PageKVCache(generator.model).forward(emb, ids, prefill=True)
        mu, sigma, pi = generator.model.prediction_head(hidden[:, -1])
        a = gmm_density_mode(mu, sigma, pi, n_iter=128, tolerance=1e-5)
        b = gmm_density_mode(mu, sigma, pi, n_iter=512)
        likely = int(pi[0].argmax())
        values = [float(-gmm_nll(x, mu, sigma, pi)) for x in (mu[:, likely], a, b)]
    return dict(top_pi_log_density=values[0], mode128_log_density=values[1], mode512_log_density=values[2],
                refine512_log_gain=values[2]-values[1], refine512_latent_rms=float(((a-b)**2).mean().sqrt()))


@torch.no_grad()
def teacher_modes(generator, reference, output):
    """源前文下逐笔预测，只导出首个回答页，目标仅供误差计算。"""
    source = reference['strokes']
    targets = [i for i, r in enumerate(source) if i and r['targetable']]
    page = source[targets[0]]['page_id']
    targets = [i for i in targets if source[i]['page_id'] == page]
    last = max(targets)
    with generator.autocast():
        emb = generator.encode([np.asarray(r['points'], dtype=np.float32) for r in source[:last+1]]).unsqueeze(0)
        ids = torch.tensor([[r['page_id'] for r in source[:last]]], device=generator.device)
        hidden = PageKVCache(generator.model).forward(emb[:, :last], ids, prefill=True)
        indices = torch.tensor(targets, device=generator.device)
        mu, sigma, pi = generator.model.prediction_head(hidden[0, indices-1])
        selected = gmm_density_mode(mu, sigma, pi, n_iter=128, tolerance=1e-5)
        top = mu[torch.arange(len(targets), device=generator.device), pi.argmax(-1)]
        target = emb[0, indices].float()
        body = torch.tensor([source[i]['kind'] not in ('bos', 'eod', 'eop') for i in targets], device=generator.device)
        gains = -gmm_nll(selected, mu, sigma, pi, reduction='none') + gmm_nll(top, mu, sigma, pi, reduction='none')
        metrics = dict(page=page, count=len(targets), body_count=int(body.sum()),
            body_top_pi_latent_mse=float(((top-target)[body]**2).mean()),
            body_density_latent_mse=float(((selected-target)[body]**2).mean()),
            mean_log_density_gain=float(gains.mean()), min_log_density_gain=float(gains.min()))
        for name, mode in (('teacher_density', 'greedy'), ('teacher_density_both', 'density')):
            points = generator.decode(selected, mode=mode)
            value = dict(strokes=[dict(page_id=page,kind=source[i]['kind'],generated=True,points=p.tolist()) for i,p in zip(targets,points)],
                         teacher_forced=True, conditioning='true_previous_strokes', kind_labels='expected_target_kind')
            write_json(output/f'{name}.json', value)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--rerun_greedy', action='store_true', help='更换推理设备时在同设备重跑最高π基线')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        raise FileExistsError('输出目录已有产物，需使用独立目录')
    torch.set_num_threads(4)
    generator, _, metadata = load_generator(args.checkpoint, 'cuda')
    baseline = read(args.baseline/'checks.json')
    serialized_ae_source = json.loads(json.dumps(metadata['ae_source']))
    if metadata['step'] != baseline['step'] or serialized_ae_source != baseline['ae_source']:
        raise ValueError('与原验收权重/AE来源不一致')
    write_json(args.out_dir/'metadata.json', dict(**metadata, baseline=str(args.baseline), training=False,
        numerical_method='all component means / diagonal precision fixed point / full density candidate selection',
        iterations=128, tolerance=1e-5, control_templates='unchanged raw and greedy AE templates',
        device_name=torch.cuda.get_device_name(), greedy_rerun=args.rerun_greedy))
    summary = []
    for item in read(args.baseline/'summary.json'):
        name = f"{item['task']}_{item['label']}"
        old = args.baseline/name
        output = args.out_dir/name
        output.mkdir()
        original = read(old/'greedy.json')
        prefix = tuple(StrokeElement(np.asarray(r['points'], dtype=np.float32), r['page_id'], False, r['kind'])
                       for r in original['strokes'] if not r['generated'])
        reference = read(old/'reference.json')
        record = dict(name=name, task=item['task'], label=item['label'], source=item['source'],
                      first_distribution=first_distribution(generator,prefix), generation={})
        variants = ([('greedy','greedy')] if args.rerun_greedy else []) + [('density','greedy'), ('density_both','density')]
        for name_mode, decoder_mode in variants:
            value = generator.generate(prefix, mode='greedy' if name_mode=='greedy' else 'density', decoder_mode=decoder_mode, seed=original['seed'],
                temperature=original['temperature'], max_new_strokes=original['max_new_strokes'],
                max_new_pages=original['max_new_pages'])
            value.update(task=item['task'], label=item['label'], source=item['source'], checkpoint_step=metadata['step'])
            write_json(output/f'{name_mode}.json', value)
            save_native_svg(value, output/f'{name_mode}.svg', f"{name} / {name_mode}")
            ink = [r for r in value['strokes'] if r['generated']]
            gains = [r.get('density_log_gain_vs_top_pi',0.0) for r in ink]
            record['generation'][name_mode] = {k:v for k,v in value.items() if k not in ('strokes','source')}
            record['generation'][name_mode].update(mean_log_density_gain=float(np.mean(gains)), min_log_density_gain=min(gains),
                dominant_component_changes=sum(r['component'] != r.get('top_pi_component',r['component']) for r in ink))
            print(json.dumps(dict(case=record['name'], variant=name_mode, **record['generation'][name_mode]),ensure_ascii=False),flush=True)
        if item['task'] in ('O1','S2','S4') or record['name'] in ('T1_story0','O2_cat'):
            record['teacher_modes'] = teacher_modes(generator,reference,output)
        summary.append(record)
        write_json(args.out_dir/'summary.json', summary)
    stop_counts = {mode:dict(Counter(r['generation'][mode]['stop_reason'] for r in summary)) for mode,_ in variants}
    write_json(args.out_dir/'completion.json',dict(cases=len(summary),generations=len(variants)*len(summary),stop_counts=stop_counts,training=False))
    print(json.dumps(dict(complete=True,stop_counts=stop_counts)),flush=True)


if __name__ == '__main__':
    main()
