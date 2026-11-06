"""2026-10-05：λ48/DCT两臂的同笔分析、运行核对和固定重建图；只读。"""
from __future__ import annotations

# 分类目录脚本的直接执行入口。
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))


import argparse
import gc
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.review.review_ae_pair import BUCKETS, NAMES, comparison, lines

LABELS = {'small6h': 'λ16首轮6h', 'small12h': 'λ16续训末步',
          'dct': 'λ48有DCT', 'nodct': 'λ48无DCT'}
COLORS = {'small6h': '#596773', 'small12h': '#9a7ab3',
          'dct': '#2667a8', 'nodct': '#ce7937'}


def check_decoder(root, samples_path, indices, device='cuda'):
    """只检查几笔：增加绘图τ和改用既有密度众数，排查显示/分量选择。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.st.ae_geometry import arc_resample, error_components
    from src.st.sampling import gmm_density_mode
    from src.st.train_ae import StrokeAutoencoder, config_from_checkpoint, load_ae_weights
    torch.set_num_threads(4)
    samples = dict(np.load(samples_path, allow_pickle=False))
    results, curves = {}, {}
    for arm in ('dct', 'nodct'):
        state = torch.load(root / arm / 'final.pt', map_location='cpu', weights_only=False)
        model = StrokeAutoencoder(config_from_checkpoint(state)).eval().to(device)
        load_ae_weights(model, state)
        latent = np.load(root / f'geometry/{arm}_strokes.npz', allow_pickle=False)['latent'][indices]
        curves[arm], results[arm] = {}, {}
        with torch.no_grad():
            embeddings = torch.from_numpy(latent).to(device)
            for mode, count in (('pi48', 48), ('pi256', 256), ('density256', 256)):
                tau = torch.linspace(0, 1, count, device=device).expand(len(indices), -1)
                mu, sigma, pi = model.decoder(embeddings, tau)
                if mode == 'density256':
                    predicted = gmm_density_mode(mu, sigma, pi)
                else:
                    component = pi.argmax(-1)[..., None, None].expand(-1, -1, 1, 2)
                    predicted = mu.gather(2, component).squeeze(2)
                raw = (predicted * .6 + predicted.new_tensor([.5, .6])).cpu().numpy()
                curves[arm][mode] = raw
                results[arm][mode] = {str(index): {'cd': float(error_components(samples['cloud'][index], arc_resample(raw[row], 32))[0]),
                    'centered_shape_cd': float(error_components(samples['cloud'][index], arc_resample(raw[row], 32))[3])}
                    for row, index in enumerate(indices)}
        del model, state
        if device == 'cuda':
            torch.cuda.empty_cache()
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(4, 4, figsize=(12, 8), layout='constrained')
    for row, (arm, selected_row) in enumerate((('dct', 0), ('nodct', 0), ('dct', 1), ('nodct', 1))):
        index = indices[selected_row]
        source = samples['cloud'][index]
        names = ('source', 'pi48', 'pi256', 'density256')
        values = [source] + [curves[arm][n][selected_row] for n in names[1:]]
        values = [source] + [v - arc_resample(v, 32).mean(0) + source.mean(0) for v in values[1:]]
        points = np.concatenate(values)
        lo, hi = points.min(0), points.max(0)
        center, size = (lo + hi) / 2, max(float((hi - lo).max()) * 1.25, .025)
        for col, (name, curve) in enumerate(zip(names, values)):
            ax = axes[row, col]
            if col:
                ax.plot(source[:, 0], source[:, 1], color='#cccccc', lw=2.3)
            ax.plot(curve[:, 0], curve[:, 1], color='#111111' if not col else COLORS[arm], lw=1.3)
            ax.set(xlim=(center[0] - size / 2, center[0] + size / 2),
                   ylim=(center[1] + size / 2, center[1] - size / 2), aspect='equal', xticks=[], yticks=[])
            titles = {'source': '原笔', 'pi48': '48τ / 最高π', 'pi256': '256τ / 最高π', 'density256': '256τ / 密度众数'}
            ax.set_title(titles[name], fontsize=9)
            if not col:
                ax.set_ylabel(f'{LABELS[arm]} / #{index}', fontsize=9)
    fig.suptitle('解码检查：仅重新推理，不训练；去质心对齐', fontsize=12)
    fig.savefig(root / 'decode_check.png', dpi=150)
    plt.close(fig)
    report = {'indices': indices, 'precision': 'float32', 'metrics': results,
              'density_mode_iterations': 64, 'tau_counts': [48, 256]}
    (root / 'decode_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def draw_strokes(path, indices, samples, bundles, *, aligned=False):
    names = ('small12h', 'dct', 'nodct')
    fig, axes = plt.subplots(len(indices), 4, figsize=(12, 2 * len(indices)),
                             layout='constrained', squeeze=False)
    for row, index in enumerate(indices):
        source = samples['cloud'][index]
        curves = [source] + [bundles[n]['reconstructed'][index].copy() for n in names]
        if aligned:
            curves = [source] + [p - p.mean(0) + source.mean(0) for p in curves[1:]]
        points = np.concatenate(curves)
        lo, hi = points.min(0), points.max(0)
        center = (lo + hi) / 2
        size = max(float((hi - lo).max()) * 1.25, .025)
        for col, (name, curve) in enumerate(zip(('source',) + names, curves)):
            ax = axes[row, col]
            if col:
                ax.plot(source[:, 0], source[:, 1], color='#cccccc', lw=2.3)
            ax.plot(curve[:, 0], curve[:, 1], color='#111111' if not col else COLORS[name], lw=1.3)
            ax.set(xlim=(center[0] - size / 2, center[0] + size / 2),
                   ylim=(center[1] + size / 2, center[1] - size / 2), aspect='equal', xticks=[], yticks=[])
            if col:
                field = 3 if aligned else 0
                ax.set_title(LABELS[name] + f" {'形状CD' if aligned else 'CD'}={bundles[name]['errors'][index, field]:.4f}", fontsize=9)
            else:
                meta = json.loads(str(samples['metadata'][index]))
                ax.set_title('原笔', fontsize=9)
                ax.set_ylabel(f"{NAMES[str(samples['bucket'][index])]} / #{index}\n{meta.get('label', '文字')}", fontsize=9)
    title = '去质心只看形状' if aligned else '原页坐标重建'
    fig.suptitle(title + '；固定此前索引，同一行坐标范围相同，灰线为原笔', fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def audit_arm(root, arm, sample_sha):
    folder = root / arm
    state = torch.load(folder / 'final.pt', map_location='cpu', weights_only=False)
    log = lines(folder / 'train.log')
    start = next(v for v in log if v.get('ae_direct_start'))
    updates = [v for v in log if 'success' in v]
    validation = lines(folder / 'validation.jsonl')
    end = validation[-1]
    stop = json.loads((folder / 'run_stop.json').read_text(encoding='utf-8'))
    recipe = state['train_args']
    cfg = state['cfg']
    assert state['step'] == end['step'] == stop['step'] == 24154
    assert stop['reason'] == 'steps' and stop['max_seconds'] == 0
    assert state['validation_source']['sha256'] == sample_sha
    assert state['source_commit'] == '943b9b0aa89776151ddf051cc87135f745b2a876'
    assert cfg['latent_units'] == 48 and cfg['enc_d_model'] == 192 and cfg['dec_hidden'] == 1024
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    assert all(int(v['step']) == state['step'] for v in state['optimizer']['state'].values())
    assert start['step'] == 0 and start['optimizer_state_entries'] == 0 and start['scale'] == 256
    assert state['cosine_annealing'] is None and recipe['lr_schedule'] == 'warmup_constant'
    assert state['optimizer']['param_groups'][0]['lr'] == 1e-4
    assert all(np.isclose(v['lr'], 1e-4 * min(v['step'] / 1000, 1)) for v in updates if v['success'])
    assert [v['step'] for v in validation] == [0] + list(range(2000, 24001, 2000)) + [24154]
    counts = state['stroke_counts']
    assert counts['sketch'] == 24154 * 3686 and counts['text'] == 24154 * 205
    auxiliary = sum(counts.get(k, 0) for k in ('bos', 'eop', 'eod', 'selection'))
    assert auxiliary == 24154 * 205
    head_keys = [k for k in state['model'] if k.startswith('dct_head.')]
    assert bool(head_keys) == (arm == 'dct')
    if arm == 'nodct':
        assert cfg['dct_coeffs'] == 0 and cfg['dct_weight'] == 0 and not cfg['dct_dc']
        assert all(v['dct_mse'] == 0 and v['loss'] == v['recon_nll'] for v in updates)
    assert np.isclose(state['best_validation_cd'], min(v['monitor_cd'] for v in validation), atol=1e-14)
    best = min(validation, key=lambda v: v['monitor_cd'])
    info = {'job_id': 604682 if arm == 'dct' else 604683,
            'slurm_seconds': 25175 if arm == 'dct' else 18613,
            'loop_seconds': stop['elapsed_seconds'], 'final_step': state['step'],
            'successful_inputs': 24154 * 4096, 'stroke_counts': counts,
            'optimizer_entries': len(state['optimizer']['state']), 'scale': state['scaler']['scale'],
            'final_lr': end['lr'], 'best_step': best['step'], 'best_monitor_cd': best['monitor_cd'],
            'final_monitor_cd': end['monitor_cd'], 'initial_ed_sha256': state['initial_ed_sha256'],
            'parameter_count': start['parameters'],
            'peak_memory_bytes': max(v['peak_memory_bytes'] for v in updates),
            'late_median_update_seconds': float(np.median([v['update_seconds'] for v in updates if v['step'] >= 20000])),
            'finite_weights': True, 'validation_source': state['validation_source'],
            'source_commit': state['source_commit'], 'data_cursors': state['data'],
            'best_fp16': best, 'first_batch_recon_nll': updates[0]['recon_nll']}
    shared = {k: state[k] for k in ('data_args', 'data_manifest', 'validation_source', 'scheduler')}
    shared['cfg'] = {k: v for k, v in cfg.items() if k not in ('dct_coeffs', 'dct_dc', 'dct_weight')}
    shared['recipe'] = {k: v for k, v in recipe.items() if k not in ('experiment', 'ae_config')}
    shared['ae_config'] = {k: v for k, v in recipe['ae_config'].items() if not k.startswith('dct_')}
    del state
    gc.collect()
    return info, shared, validation, updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--prior_dir', type=Path, required=True)
    parser.add_argument('--continued_dir', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--check_decoder', action='store_true', help='几笔48/256τ与密度众数的推理检查')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda', help='仅用于额外解码检查')
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.run_dir
    prior = json.loads((args.prior_dir / 'summary.json').read_text(encoding='utf-8'))
    continued = json.loads((args.continued_dir / 'summary.json').read_text(encoding='utf-8'))
    sha = hashlib.sha256(args.samples.read_bytes()).hexdigest()
    audits, common, validation, updates = {}, {}, {}, {}
    for arm in ('dct', 'nodct'):
        audits[arm], common[arm], validation[arm], updates[arm] = audit_arm(root, arm, sha)
    assert common['dct'] == common['nodct']
    reference = torch.load(args.prior_dir / 'text5/final.pt', map_location='cpu', weights_only=False)
    for key in ('data_args', 'data_manifest', 'validation_source'):
        assert common['dct'][key] == reference[key]
    del reference
    gc.collect()
    assert audits['dct']['initial_ed_sha256'] == audits['nodct']['initial_ed_sha256']
    assert audits['dct']['first_batch_recon_nll'] == audits['nodct']['first_batch_recon_nll']
    geometry = json.loads((root / 'geometry/report.json').read_text(encoding='utf-8'))['models']
    reports = {'old': prior['historical_baselines']['old_frozen2'],
               'small6h': prior['fp32_geometry']['text5'],
               'small12h': continued['models_fp32']['after'], **geometry}
    comparisons = {arm: {bucket: {name: comparison(reports[name]['buckets'][bucket]['cd']['mean'],
                     reports[arm]['buckets'][bucket]['cd']['mean']) for name in ('old', 'small6h', 'small12h')}
                     for bucket in BUCKETS} for arm in ('dct', 'nodct')}
    dct_difference = {bucket: {field: comparison(reports['dct']['buckets'][bucket][field]['mean'],
                               reports['nodct']['buckets'][bucket][field]['mean'])
                               for field in ('cd', 'shape_contribution', 'position_contribution', 'centered_shape_cd', 'centroid_distance')}
                      for bucket in BUCKETS}
    late = {}
    for arm in ('dct', 'nodct'):
        later = [v for v in validation[arm] if v['step'] >= 18000]
        late[arm] = {b: {field: {'values': [v['buckets'][b][field]['mean'] for v in later],
                    'from_18000_to_final': comparison(later[0]['buckets'][b][field]['mean'], later[-1]['buckets'][b][field]['mean'])}
                    for field in ('cd', 'shape_contribution', 'position_contribution')} for b in BUCKETS}
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    prior_valid = lines(args.prior_dir / 'text5/validation.jsonl')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), layout='constrained')
    for ax, bucket in zip(axes[0], BUCKETS[:3]):
        for name, rows in (('small6h', prior_valid), ('dct', validation['dct']), ('nodct', validation['nodct'])):
            rows = [v for v in rows if v['step'] >= 4000]
            ax.plot(np.array([v['step'] for v in rows]) / 1000, [v['buckets'][bucket]['cd']['mean'] for v in rows],
                    '.-', color=COLORS[name], label=LABELS[name])
        for name, style in (('old', ':'), ('small12h', '--')):
            ax.axhline(reports[name]['buckets'][bucket]['cd']['mean'], color='#555555' if name == 'old' else COLORS[name],
                       ls=style, lw=1, label='旧冻结AE' if name == 'old' else LABELS[name])
        ax.set(title=NAMES[bucket] + ' CD', xlabel='成功更新（千次）', ylabel='CD', yscale='log')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    for ax, field, title in zip(axes[1, :2], ('shape_contribution', 'position_contribution'), ('图形形状贡献', '图形位置贡献')):
        for name, rows in (('small6h', prior_valid), ('dct', validation['dct']), ('nodct', validation['nodct'])):
            rows = [v for v in rows if v['step'] >= 10000]
            ax.plot(np.array([v['step'] for v in rows]) / 1000, [v['buckets']['sketch'][field]['mean'] for v in rows],
                    '.-', color=COLORS[name], label=LABELS[name])
        ax.axhline(reports['small12h']['buckets']['sketch'][field]['mean'], color=COLORS['small12h'], ls='--', label=LABELS['small12h'])
        ax.set(title=title, xlabel='成功更新（千次）', ylabel='CD贡献')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    ax = axes[1, 2]
    for name in ('dct', 'nodct'):
        rows = updates[name]
        ys = np.array([v['recon_nll'] for v in rows])
        smoothed = [float(np.mean(ys[max(0, i - 19):i + 1])) for i in range(len(ys))]
        ax.plot(np.array([v['step'] for v in rows]) / 1000, smoothed, color=COLORS[name], label=LABELS[name])
    ax.set(title='重建NLL（20条日志平均）', xlabel='成功更新（千次）', ylabel='NLL')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.suptitle('λ48扩容与DCT有无：相同步数、固定6144笔验证', fontsize=15)
    fig.savefig(root / 'curves.png', dpi=155)
    plt.close(fig)
    samples = dict(np.load(args.samples, allow_pickle=False))
    bundles = {'small12h': dict(np.load(args.continued_dir / 'geometry/continued_strokes.npz', allow_pickle=False))}
    bundles.update({arm: dict(np.load(root / f'geometry/{arm}_strokes.npz', allow_pickle=False)) for arm in ('dct', 'nodct')})
    for bundle in bundles.values():
        np.testing.assert_array_equal(bundle['cloud'], samples['cloud'])
    fixed, tail = continued['fixed_examples'], continued['shape_tail_examples']
    draw_strokes(root / 'stroke_comparison.png', fixed, samples, bundles)
    draw_strokes(root / 'shape_tail.png', tail, samples, bundles)
    draw_strokes(root / 'shape_aligned.png', tail, samples, bundles, aligned=True)
    draw_strokes(root / 'shape_focus.png', tail[:2], samples, bundles, aligned=True)
    example_errors = {str(i): {name: {'cd': float(v['errors'][i, 0]), 'shape': float(v['errors'][i, 1]),
                                    'position': float(v['errors'][i, 2]), 'shape_only': float(v['errors'][i, 3])}
                             for name, v in bundles.items()} for i in fixed + tail}
    if args.check_decoder:
        check_decoder(root, args.samples, tail, args.device)
    decode_path = root / 'decode_check.json'
    decoding_check = json.loads(decode_path.read_text(encoding='utf-8')) if decode_path.is_file() else None
    summary = {'audit': audits, 'shared_configuration_and_initialization': True,
               'same_data_as_small6h': True, 'samples_sha256': sha,
               'models_fp32': reports, 'comparisons_fp32': comparisons, 'no_dct_vs_dct_fp32': dct_difference,
               'fp16_late_curve': late, 'fp16_final': {n: validation[n][-1] for n in ('dct', 'nodct')},
               'fixed_examples': fixed, 'shape_tail_examples': tail, 'example_errors': example_errors,
               'fp16_fp32_cd_delta_percent': {n: {b: 100 * (reports[n]['buckets'][b]['cd']['mean'] /
                       validation[n][-1]['buckets'][b]['cd']['mean'] - 1) for b in BUCKETS} for n in ('dct', 'nodct')},
               'no_dct_walltime_change_percent': 100 * (audits['nodct']['slurm_seconds'] / audits['dct']['slurm_seconds'] - 1),
               'decoding_check': decoding_check}
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'audit': audits, 'comparisons': comparisons, 'no_dct_vs_dct': dct_difference,
                      'example_errors': example_errors}, ensure_ascii=False))


if __name__ == '__main__':
    main()
