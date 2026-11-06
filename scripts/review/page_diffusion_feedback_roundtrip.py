"""固定已生成曲线，经同一 D→E 回馈，检查后续误差能否由嵌入修复。"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import torch

from scripts.review.page_diffusion_probe_core import (
    load_model, history, cache_histories, predict_blocks, targets,
    common_noise, errors, decode,
)
from scripts.review.probe_page_diffusion import principal_axes
from src.st.generation_pretrain import PageGenerator
from src.st.page_diffusion_runtime import autocast


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, codec, state = load_model(args.out/'run/last.pt')
    del state
    geometry = PageGenerator(codec)
    device = next(model.parameters()).device
    _, axes = principal_axes(Path('tmp_smoke/ae_latent_geometry_20261103/r_encoded.npz'))
    basis = torch.tensor(axes, dtype=torch.float32, device=device)
    for task in ('T1', 'S1'):
        data_dir = args.out/('data_uniform_text' if task == 'T1' else 'data')
        metric_dir = args.out/('metrics_uniform_text' if task == 'T1' else 'metrics')/'final_raw'
        destination = metric_dir/f'roundtrip_valid_{task}.pt'
        if destination.exists():
            continue
        data = torch.load(data_dir/f'valid_{task}.pt', map_location='cpu', weights_only=False)
        rows_by_id = {row['id']: row for row in data}
        original = torch.load(metric_dir/f'rollout_valid_{task}.pt', map_location='cpu', weights_only=False)
        times = original['times'].to(device)
        results = []
        for begin in range(0, len(original['results']), 16):
            saved = original['results'][begin:begin+16]
            batch = [rows_by_id[r['id']] for r in saved]
            assert all(r['source'] == s['source'] for r, s in zip(batch, saved))
            first = [r['blocks'][r['rollout'][0]] for r in batch]
            initial = [history(r, block=b) for r, b in zip(batch, first)]
            histories = {name: [tuple(v.clone() for v in h) for h in initial]
                         for name in ('generated_roundtrip', 'target_roundtrip')}
            by_horizon = []
            for horizon in range(4):
                parts = [r['blocks'][r['rollout'][0]+horizon] for r in batch]
                target, body, _, _ = targets(model, batch, parts)
                noise = common_noise(batch, salt=100+horizon).to(device)
                values = (1-times[None,:,None,None])*target[:,None]+times[None,:,None,None]*noise[:,None]
                pages = torch.tensor([p['page'] for p in parts], device=device)
                local = torch.tensor([p['local'] for p in parts], device=device)
                metrics = {}
                for name, hs in histories.items():
                    cache, valid = cache_histories(model, hs)
                    pred = predict_blocks(model, cache, valid, values, times, pages, local)
                    metrics[name] = errors(pred, target, body, basis)
                    del cache
                    curve_key = 'generated_curves' if name.startswith('generated') else 'reference_curves'
                    latent_key = 'generated_latent' if name.startswith('generated') else 'target_latent'
                    curves = torch.stack([r['horizons'][horizon][curve_key] for r in saved]).to(device)
                    before = torch.stack([r['horizons'][horizon][latent_key] for r in saved]).to(device)
                    with autocast(device):
                        recoded = model.normalize(geometry.encode_tensor(curves.reshape(-1,76,2))).float().reshape(-1,16,48)
                    metrics[name+'_latent_change'] = (recoded-before).square().mean((-1,-2)).cpu().numpy()
                    after = decode(codec, model, recoded, points=76)
                    metrics[name+'_curve_change'] = (after-curves).square().sum(-1).mean((-1,-2)).sqrt().cpu().numpy()
                    for i, part in enumerate(parts):
                        pp = torch.full((16,),part['page'],dtype=torch.long)
                        ll = torch.arange(part['local'],part['local']+16)
                        hs[i] = (torch.cat((hs[i][0],recoded[i].cpu())),
                                 torch.cat((hs[i][1],pp)),torch.cat((hs[i][2],ll)))
                by_horizon.append(metrics)
            for i, row in enumerate(batch):
                results.append(dict(id=row['id'],source=row['source'],
                    horizons=[{k:v[i] for k,v in h.items()} for h in by_horizon]))
            print(json.dumps(dict(task=task,done=len(results),total=len(original['results']))),flush=True)
        torch.save(dict(times=times.cpu(),results=results,
            scope='固定已有生成曲线；原始目标D→E作往返重采样对照；全部使用76点解码曲线'),destination)


if __name__ == '__main__':
    main()
