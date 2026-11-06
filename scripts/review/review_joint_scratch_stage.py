"""2026-10-26：从头图文R与初版joint的同源阶段检查，复用已有检查资料。"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def audit(out, baseline):
    import torch
    current = torch.load(out/'model.pt', map_location='cpu', weights_only=False)
    old = torch.load(baseline/'model.pt', map_location='cpu', weights_only=False)
    frozen = [k for k in old['model'] if k.startswith(('encoder.', 'decoder.', 'dct_head.'))]
    groups = []
    for group in current['optimizer']['param_groups']:
        steps = [int(current['optimizer']['state'][p]['step']) for p in group['params']
                 if current['optimizer']['state'].get(p)]
        groups.append(dict(role=group.get('role', 'R'), entries=len(steps),
                           min_step=min(steps), max_step=max(steps)))
    report = dict(step=current['step'], phase_step=current['phase_step'],
                  successful_inputs=current['scheduler']['inputs'],
                  source_commit=current['source_commit'], target_inputs=current['train_args']['total_inputs'],
                  comparison_step=old['step'], comparison_phase_step=old['phase_step'],
                  comparison_joint_inputs=old['scheduler']['inputs'],
                  comparison_cfg=old['cfg'],
                  frozen_ae_equal=all(torch.equal(current['model'][k], old['model'][k]) for k in frozen),
                  weights_finite=all(bool(torch.isfinite(v).all()) for v in current['model'].values()),
                  optimizer_groups=groups, cfg=current['cfg'], scheduler=current['scheduler'],
                  counters=current['counters'], data=current['data'], train_args=current['train_args'],
                  optimizer_updates=0, comparison_checkpoint=str((baseline/'model.pt').resolve()),
                  checkpoint=str((out/'model.pt').resolve()))
    assert report['frozen_ae_equal'] and report['weights_finite']
    assert all(g['min_step'] == g['max_step'] == report['step'] for g in groups)
    assert report['successful_inputs'] == report['counters']['successful_inputs']
    write(out/'audit.json', report)
    print(json.dumps({k: report[k] for k in ('step', 'successful_inputs', 'frozen_ae_equal', 'weights_finite')}), flush=True)


def teacher_review(out, baseline):
    import torch
    from scripts.demo.demo_text_readout import load_joint_readout
    from scripts.review.review_joint_stage import dataset_for, teacher, aggregate
    from scripts.review.diagnose_formal_r_health import model_hash
    torch.set_num_threads(4)
    started = time.monotonic()
    bundle = read(out/'probe_bundle.json')
    baseline_teacher = baseline/'joint_stage_teacher.json'
    if not baseline_teacher.exists():
        baseline_teacher = baseline/'scratch_teacher.json'
    old = read(baseline_teacher)
    generator, head, vocab, layout, source = load_joint_readout(out/'model.pt')
    generator.model.encode_chunk_size = 128
    destination = out/'scratch_teacher.json'
    rows = read(destination) if destination.exists() else []
    for i, row in enumerate(rows):
        assert all(row[k] == old[i][k] for k in ('split', 'task', 'index', 'source_id'))
    datasets = {split: dataset_for(bundle, split, vocab.font) for split in ('train', 'valid')}
    for i in range(len(rows), len(old)):
        reference = old[i]
        document = datasets[reference['split']].get_for_task(reference['task'], reference['index'])
        assert document.metadata.get('source_id') == reference['source_id']
        metrics, _, _ = teacher(generator, head, document)
        rows.append(dict(split=reference['split'], task=reference['task'], index=reference['index'],
                         source_id=document.metadata.get('source_id'), metrics=metrics))
        if (i + 1) % 128 == 0 or i == len(old) - 1:
            write(destination, rows)
            print(json.dumps(dict(teacher_documents=len(rows), total=len(old), seconds=time.monotonic()-started)), flush=True)
    assert model_hash(generator.model) == source['r_model_hash']

    def balanced(values):
        seen = set()
        for row in values:
            split, task, index = row['split'], row['task'], row['index']
            key = split, task, index
            if key in seen:
                continue
            seen.add(key)
            if task in ('O1', 'O2') and index not in bundle['splits'][split]['balanced_indices']:
                continue
            yield row
    write(out/'teacher_comparison.json', dict(
        old=aggregate(list(balanced(old))), scratch=aggregate(list(balanced(rows))),
        documents=len(rows), step=source['step'], model_hash=source['r_model_hash'],
        conditioning_source_equal=True, weights_unchanged=True,
        optimizer_updates=0, seconds=time.monotonic()-started))


def drawings(out, baseline):
    import torch
    from scripts.demo.demo_text_readout import load_joint_readout
    from scripts.demo.demo_pretrain_generation import save_native_svg, source_prefix
    from scripts.review.review_joint_stage import dataset_for
    from scripts.review.review_formal_r import stroke_records
    from scripts.review.diagnose_formal_r_health import model_hash
    from scripts.review.render_joint_full_review import previews
    torch.set_num_threads(4)
    generator, head, vocab, layout, source = load_joint_readout(out/'model.pt')
    generator.model.encode_chunk_size = 128
    dataset = dataset_for(read(out/'probe_bundle.json'), 'valid', vocab.font)
    cases = [c for c in read(baseline/'joint_summary.json')['cases'] if c['task'] in ('O1', 'S2')]
    for case in cases:
        destination = out/case['name']
        destination.mkdir(exist_ok=True)
        document = dataset.get_for_task(case['task'], case['index'])
        reference = read(baseline/case['name']/'reference.json')
        actual = stroke_records(document)
        assert len(actual) == len(reference['strokes'])
        assert all(all(a[k] == b[k] for k in ('page_id','kind','points','targetable'))
                   for a,b in zip(actual, reference['strokes']))
        # 逐点核对同一文档后保留原始参考，旧产物仍在原目录。
        write(destination/'reference.json', reference)
        for mode in ('greedy', 'sample'):
            path = destination/(mode+'.json')
            if path.exists():
                continue
            value = generator.generate(source_prefix(document, case['task']), mode=mode,
                                       temperature=.5, seed=42, max_new_strokes=160, max_new_pages=8)
            write(path, value)
            save_native_svg(value, destination/(mode+'.svg'), case['name'])
            print(json.dumps(dict(drawing=case['name'], mode=mode, stop=value['stop_reason'])), flush=True)
    assert model_hash(generator.model) == source['r_model_hash']
    write(out/'joint_summary.json', dict(cases=cases, step=source['step'], optimizer_updates=0, weights_unchanged=True))
    previews(out)


def baseline_replay(out, baseline):
    """用当前同一推理实现重放旧模型样张，记录短回答兼容性。"""
    import torch
    from types import SimpleNamespace
    from scripts.demo.demo_text_readout import load_joint_readout
    from scripts.demo.demo_pretrain_generation import source_prefix
    from scripts.review.review_joint_stage import dataset_for
    from src.st.text_readout_generation import generate_text_readout
    from scripts.review.diagnose_formal_r_health import model_hash
    replay = out/'baseline_replay'
    replay.mkdir(exist_ok=True)
    if not (replay/'model.pt').exists():
        shutil.copyfile(baseline/'model.pt', replay/'model.pt')
    if not (replay/'probe_bundle.json').exists():
        write(replay/'probe_bundle.json', read(out/'probe_bundle.json'))
    if not (replay/'joint_summary.json').exists():
        drawings(replay, baseline)
        gc.collect();torch.cuda.empty_cache()
    torch.set_num_threads(4)
    generator, head, vocab, layout, source = load_joint_readout(baseline/'model.pt')
    generator.model.encode_chunk_size = 128
    dataset = dataset_for(read(out/'probe_bundle.json'), 'valid', vocab.font)
    cases = []
    for old in read(baseline/'text/summary.json')['cases'][:4]:
        value = generate_text_readout(generator, head, vocab, old['prompt'], time.monotonic()+120,
            SimpleNamespace(constraint='prefix', generation_strokes=240), layout)
        text_pages = {str(k): v for k,v in value['generated_page_texts'].items()}
        case = dict(old, generated_page_texts=text_pages, stop_reason=value['stop_reason'])
        write(replay/f'text_{old["index"]}.json', value)
        cases.append(dict(task='T1', index=old['index'], same_text=text_pages==old['generated_page_texts'], case=case))
        print(json.dumps({k:cases[-1][k] for k in ('task','index','same_text')}), flush=True)
    # 取旧资料中固定的前16个对象、每个场景文字任务的前8个条件；同输入检查推理实现。
    recognition = read(baseline/'recognition.json')['cases'][:16]
    scene_cases = read(baseline/'scene_abilities.json')['cases']
    selected = [('O2', row) for row in recognition]
    selected += [(task, row) for task in ('S1','S3','S5')
                 for row in [r for r in scene_cases if r['task']==task][:8]]
    for task, old in selected:
        document = dataset.get_for_task(task, old['index'])
        value = generate_text_readout(generator, head, vocab, '', time.monotonic()+90,
            SimpleNamespace(constraint='prefix', generation_strokes=320 if task=='S1' else 96),
            layout, prefix=source_prefix(document, task))
        actual = ' '.join(value['generated_page_texts'].values()).strip()
        cases.append(dict(task=task, index=old['index'], expected=old['expected'],
                          old=old['generated'], generated=actual, same_text=actual==old['generated']))
        write(replay/f'{task}_{old["index"]}.json', value)
    assert model_hash(generator.model) == source['r_model_hash']
    report = dict(cases=cases, same=sum(r['same_text'] for r in cases), count=len(cases),
                  weights_unchanged=True, optimizer_updates=0)
    write(out/'baseline_runtime_replay.json', report)
    print(json.dumps({k:report[k] for k in ('same','count')}), flush=True)
    changed_tasks = {r['task'] for r in cases if not r['same_text']}
    del generator, head, vocab
    gc.collect();torch.cuda.empty_cache()
    commands = [('localization', ['-m', 'scripts.review.complete_joint_stage', '--out', str(replay), '--localization_only'])]
    if 'O2' in changed_tasks:
        commands.append(('recognition', ['-m', 'scripts.review.probe_joint_recognition', '--out', str(replay), '--all_classes']))
    if changed_tasks & {'S1','S3','S5'}:
        commands.append(('scenes', ['-m', 'scripts.review.probe_joint_scene_abilities', '--out', str(replay)]))
    for name, command in commands:
        if (replay/('scene_abilities.json' if name=='scenes' else name+'.json')).exists():
            continue
        with (replay/(name+'_run.log')).open('a', encoding='utf-8') as stdout, (replay/(name+'_error.log')).open('a', encoding='utf-8') as stderr:
            subprocess.run([sys.executable, '-X', 'utf8', *command], cwd=ROOT, stdout=stdout, stderr=stderr, check=True)


def run_pipeline(out, baseline):
    """各检查单独进程、单任务使用CUDA；产物存在时继续未完成部分。"""
    status = out/'review_status.json'
    commands = [
        ('audit', out/'audit.json', ['-m', 'scripts.review.review_joint_scratch_stage', '--phase', 'audit']),
        ('text', out/'text/summary.json', ['-m', 'scripts.review.quick_review_r_text', '--checkpoint', str(out/'model.pt'),
                                        '--out', str(out/'text'), '--max_seconds', '2400']),
        ('teacher', out/'teacher_comparison.json', ['-m', 'scripts.review.review_joint_scratch_stage', '--phase', 'teacher']),
        ('drawings', out/'joint_summary.json', ['-m', 'scripts.review.review_joint_scratch_stage', '--phase', 'drawings']),
        ('recognition', out/'recognition.json', ['-m', 'scripts.review.probe_joint_recognition', '--out', str(out), '--all_classes']),
        ('scenes', out/'scene_abilities.json', ['-m', 'scripts.review.probe_joint_scene_abilities', '--out', str(out)]),
        ('localization', out/'localization.json', ['-m', 'scripts.review.complete_joint_stage', '--out', str(out), '--localization_only']),
    ]
    for name, result, command in commands:
        if result.exists():
            continue
        if name in ('audit', 'teacher', 'drawings'):
            command += ['--out', str(out), '--baseline', str(baseline)]
        write(status, dict(phase=name, state='running', started_at=time.time()))
        with (out/(name+'_run.log')).open('a', encoding='utf-8') as stdout, (out/(name+'_error.log')).open('a', encoding='utf-8') as stderr:
            completed = subprocess.run([sys.executable, '-X', 'utf8', *command], cwd=ROOT, stdout=stdout, stderr=stderr)
        if completed.returncode:
            write(status, dict(phase=name, state='failed', returncode=completed.returncode, at=time.time()))
            raise RuntimeError(f'{name}失败，检查独立日志')
        gc.collect()
    write(status, dict(phase='complete', state='complete', at=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT/'tmp_smoke/r_joint_full_review_20261022')
    parser.add_argument('--phase', choices=('all', 'audit', 'teacher', 'drawings', 'baseline_replay'), default='all')
    args = parser.parse_args()
    out, baseline = args.out.resolve(), args.baseline.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'all':
        run_pipeline(out, baseline)
    else:
        dict(audit=audit, teacher=teacher_review, drawings=drawings, baseline_replay=baseline_replay)[args.phase](out, baseline)


if __name__ == '__main__':
    main()
