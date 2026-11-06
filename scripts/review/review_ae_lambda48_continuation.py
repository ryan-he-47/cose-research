"""2026-10-06：λ48有DCT接续验收，复用同笔几何和固定样张；只读。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.st.ae_schedule import StepLRReplay
from scripts.review.review_ae_pair import BUCKETS, NAMES, comparison, lines, at_step

LABELS = {'small': 'λ16续训末步', 'before': 'λ48续训前', 'after': 'λ48续训后'}
COLORS = {'small': '#9a7ab3', 'before': '#2667a8', 'after': '#ce7937'}


def save_plot(fig, path):
    # 2026-10-06：Windows预览可能映射旧PNG；先写新文件再替换，避免截断映射文件。
    temporary = path.with_suffix('.new.png')
    fig.savefig(temporary, dpi=150)
    temporary.replace(path)


def draw_strokes(path, indices, samples, bundles, *, aligned=False):
    names = ('small', 'before', 'after')
    fig, axes = plt.subplots(len(indices), 4, figsize=(12, 2 * len(indices)),
                             layout='constrained', squeeze=False)
    for row, index in enumerate(indices):
        source = samples['cloud'][index]
        curves = [source] + [bundles[name]['reconstructed'][index].copy() for name in names]
        if aligned:
            curves = [source] + [curve - curve.mean(0) + source.mean(0) for curve in curves[1:]]
        points = np.concatenate(curves)
        lo, hi = points.min(0), points.max(0)
        center, size = (lo + hi) / 2, max(float((hi - lo).max()) * 1.25, .025)
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
    fig.suptitle(('去质心只看形状' if aligned else '原页坐标重建') +
                 '；沿用此前索引，同一行坐标范围相同，灰线为原笔', fontsize=12)
    save_plot(fig, path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--lambda48_dir', type=Path, required=True)
    parser.add_argument('--lambda16_dir', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--gpu_seconds', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.run_dir
    prior = json.loads((args.lambda48_dir / 'summary.json').read_text(encoding='utf-8'))
    small = json.loads((args.lambda16_dir / 'summary.json').read_text(encoding='utf-8'))
    before = torch.load(args.lambda48_dir / 'dct/final.pt', map_location='cpu', weights_only=False)
    after = torch.load(root / 'run/final.pt', map_location='cpu', weights_only=False)
    for key in ('cfg', 'train_args', 'data_args', 'data_manifest', 'validation_source',
                'scheduler', 'initial_ed_sha256'):
        assert before[key] == after[key], key
    assert after['validation_source']['sha256'] == hashlib.sha256(args.samples.read_bytes()).hexdigest()
    assert after['step'] == 48192 and before['step'] == 24154
    assert all(torch.isfinite(value).all() for value in after['model'].values())
    assert len(after['optimizer']['state']) == len(before['optimizer']['state']) == 116
    assert all(int(value['step']) == after['step'] for value in after['optimizer']['state'].values())
    assert after['data']['quotas'] == before['data']['quotas']
    log = lines(root / 'run/train.log')
    start = next(value for value in log if value.get('ae_direct_start'))
    updates = [value for value in log if 'success' in value]
    assert start['step'] == before['step'] and start['kinds_cumulative'] == before['stroke_counts']
    assert start['optimizer_state_entries'] == 116 and start['scale'] == before['scaler']['scale']
    replay = StepLRReplay(after['step_lr_schedule'])
    requested = json.loads((root / 'run/lr_replay_requested.json').read_text(encoding='utf-8'))
    assert after['step_lr_schedule'] == requested
    assert start['lr_schedule'] == after['step_lr_schedule']
    assert all(value['lr'] == replay.lr(value['lr_replay_step']) for value in updates)
    assert all(value['lr_replay_step'] == value['step'] + (not value['success']) for value in updates)
    assert after['optimizer']['param_groups'][0]['lr'] == 5e-6
    assert after['cosine_annealing'] is None
    validation = lines(root / 'run/validation.jsonl')
    history = lines(args.lambda48_dir / 'dct/validation.jsonl')
    end = validation[-1]
    assert [value['step'] for value in validation] == list(range(26000, 48001, 2000)) + [48192]
    assert end['step'] == after['step'] and end['lr'] == 5e-6
    best = min(history + validation, key=lambda value: value['monitor_cd'])
    assert after['best_validation_cd'] == best['monitor_cd']
    stop = json.loads((root / 'run/run_stop.json').read_text(encoding='utf-8'))
    assert stop['step'] == after['step'] and stop['reason'] == 'steps' and stop['max_seconds'] == 0
    delta = after['step'] - before['step']
    increment = {key: after['stroke_counts'].get(key, 0) - before['stroke_counts'].get(key, 0)
                 for key in after['stroke_counts']}
    assert increment['sketch'] == delta * 3686 and increment['text'] == delta * 205
    assert sum(increment.get(key, 0) for key in ('bos', 'eop', 'eod', 'selection')) == delta * 205
    assert sum(not value['success'] for value in updates) == increment['skipped_updates']
    audit = {'job_id': 605767, 'gpu_seconds': args.gpu_seconds, 'loop_seconds': stop['elapsed_seconds'],
             'total_lambda48_gpu_seconds': args.gpu_seconds + prior['audit']['dct']['slurm_seconds'],
             'resume_step': before['step'], 'final_step': after['step'], 'new_updates': delta,
             'new_successful_inputs': delta * 4096, 'total_successful_inputs': after['step'] * 4096,
             'stroke_increment': increment, 'stroke_counts': after['stroke_counts'],
             'optimizer_entries': 116, 'scale': after['scaler']['scale'], 'final_lr': end['lr'],
             'lr_records_exact': len(updates),
             'logged_successful_updates': sum(value['success'] for value in updates),
             'lr_profile': after['step_lr_schedule'],
             'best_step': best['step'], 'best_monitor_cd': best['monitor_cd'],
             'final_monitor_cd': end['monitor_cd'], 'final_is_lowest_monitor': best['step'] == after['step'],
             'same_cfg_recipe_data_validation': True, 'plateau_unchanged': True, 'finite_weights': True,
             'source_commit': after['source_commit'], 'validation_source': after['validation_source'],
             'data_cursors': after['data'], 'peak_memory_bytes': max(value['peak_memory_bytes'] for value in updates),
             'late_median_update_seconds': float(np.median([value['update_seconds'] for value in updates if value['step'] >= 44000]))}
    geometry = json.loads((root / 'geometry/report.json').read_text(encoding='utf-8'))['models']['continued']
    reports = {'old': prior['models_fp32']['old'], 'small': small['models_fp32']['after'],
               'before': prior['models_fp32']['dct'], 'after': geometry}
    fields = ('cd', 'shape_contribution', 'position_contribution', 'centered_shape_cd', 'centroid_distance')
    comparisons = {bucket: {name: {field: comparison(reports[name]['buckets'][bucket][field]['mean'],
        geometry['buckets'][bucket][field]['mean']) for field in fields} for name in ('old', 'small', 'before')}
        for bucket in BUCKETS}
    new = [history[-1]] + validation
    # 恒定段末节点40600；验证间插值仅用于观察分段变化。
    boundary = 40600
    phase = {bucket: {field: {'hold': comparison(new[0]['buckets'][bucket][field]['mean'],
        at_step(new, boundary, bucket, field)), 'cosine': comparison(at_step(new, boundary, bucket, field),
        end['buckets'][bucket][field]['mean'])} for field in fields[:3]} for bucket in BUCKETS}
    late = {bucket: {field: comparison(validation[-2]['buckets'][bucket][field]['mean'],
        end['buckets'][bucket][field]['mean']) for field in fields[:3]} for bucket in BUCKETS}
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 10})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), layout='constrained')
    small_history = lines(args.lambda16_dir / 'run/validation.jsonl')
    for ax, bucket in zip(axes[0], BUCKETS[:3]):
        for name, rows in (('small', small_history), ('before', history), ('after', new)):
            rows = [value for value in rows if value['step'] >= 10000]
            ax.plot(np.array([value['step'] for value in rows]) / 1000,
                    [value['buckets'][bucket]['cd']['mean'] for value in rows], '.-',
                    label=LABELS[name], color=COLORS[name])
        ax.axhline(reports['old']['buckets'][bucket]['cd']['mean'], color='#555555', ls=':', label='旧冻结AE（FP32）')
        ax.set(title=NAMES[bucket] + ' CD', xlabel='成功更新（千次）', ylabel='CD', yscale='log')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    ax = axes[1, 0]
    for field, label, color in (('cd', '图形总CD', '#2667a8'), ('shape_contribution', '形状贡献', '#ce7937'),
                                ('position_contribution', '位置贡献', '#238673')):
        ax.plot(np.array([value['step'] for value in new]) / 1000,
                [value['buckets']['sketch'][field]['mean'] for value in new], '.-', label=label, color=color)
    ax.set(title='λ48接续图形误差', xlabel='成功更新（千次）', ylabel='CD贡献')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    ax = axes[1, 1]
    ax.plot(np.array([value['step'] for value in updates]) / 1000, [value['lr'] for value in updates], color='#2667a8')
    ax.axvline(boundary / 1000, color='#999999', ls='--', lw=.8)
    ax.set(title='实际LR：对齐λ16续训步数', xlabel='成功更新（千次）', ylabel='LR')
    ax.grid(alpha=.2)
    ax = axes[1, 2]
    for field, label, color in (('recon_nll', '重建NLL', '#2667a8'), ('dct_mse', 'DCT MSE', '#ce7937')):
        target = ax if field == 'recon_nll' else ax.twinx()
        values = [value[field] for value in updates]
        smooth = [np.mean(values[max(0, i - 19):i + 1]) for i in range(len(values))]
        target.plot(np.array([value['step'] for value in updates]) / 1000, smooth, color=color, label=label)
        target.set_ylabel(label)
        target.legend(loc='upper left' if field == 'recon_nll' else 'upper right', fontsize=8)
    ax.set(title='续训损失（20条日志平均）', xlabel='成功更新（千次）')
    ax.grid(alpha=.2)
    fig.suptitle('CoSE λ48+DCT接续：与λ16对齐成功更新和LR', fontsize=14)
    save_plot(fig, root / 'curves.png')
    plt.close(fig)
    samples = dict(np.load(args.samples, allow_pickle=False))
    bundles = {name: dict(np.load(path, allow_pickle=False)) for name, path in (
        ('small', args.lambda16_dir / 'geometry/continued_strokes.npz'),
        ('before', args.lambda48_dir / 'geometry/dct_strokes.npz'),
        ('after', root / 'geometry/continued_strokes.npz'))}
    for bundle in bundles.values():
        np.testing.assert_array_equal(bundle['cloud'], samples['cloud'])
    chosen = [273, 893, 3882, 2828, 5123, 6032]
    tail = [3369, 3913, 4357, 5960]
    draw_strokes(root / 'stroke_comparison.png', chosen, samples, bundles)
    draw_strokes(root / 'shape_aligned.png', tail, samples, bundles, aligned=True)
    draw_strokes(root / 'shape_focus.png', tail[:2], samples, bundles, aligned=True)
    example_errors = {str(index): {name: {'cd': float(bundle['errors'][index, 0]),
        'shape': float(bundle['errors'][index, 1]), 'position': float(bundle['errors'][index, 2]),
        'centered_shape_cd': float(bundle['errors'][index, 3])} for name, bundle in bundles.items()}
        for index in chosen + tail}
    summary = {'audit': audit, 'models_fp32': reports, 'comparisons_fp32': comparisons,
               'phase_fp16': phase, 'last_two_validations_fp16': late, 'fp16_final': end,
               'fixed_examples': chosen, 'shape_tail_examples': tail, 'example_errors': example_errors,
               'phase_boundary_step': boundary, 'phase_note': 'validation interpolation at last constant-LR node',
               'fp16_fp32_graphic_delta_percent': 100 * (geometry['buckets']['sketch']['cd']['mean'] / end['buckets']['sketch']['cd']['mean'] - 1)}
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'audit': {key: value for key, value in audit.items() if key not in ('lr_profile', 'data_cursors')},
                      'comparisons': comparisons, 'phase': phase, 'late': late, 'examples': example_errors}, ensure_ascii=False))


if __name__ == '__main__':
    main()
