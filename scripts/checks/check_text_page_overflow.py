"""2026-10-11：复现长串分页故障，核对普通排版、增强和准备失败回滚。"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.page_document import build_document, layout_text_pages
from src.text_renderer import VectorFont
from src.text_augmentation import augment_rendered_text, TextAugmentationConfig
from src.st.formal_recipe import TASKS
from src.st.formal_stream import InputDocumentStream, collect_documents
from src.st.pipeline_pretrain import build_document_dataset, collate_documents


def records(document):
    return [(e.page_id, e.kind, e.targetable, e.rawxy) for e in document.to_sequence()]


def equal(left, right):
    a, b = records(left), records(right)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x[:3] == y[:3]
        np.testing.assert_array_equal(x[3], y[3])


def fp16_emergency_check(parent, out):
    """两个短更新，注入准备故障后完整恢复；不使用正式训练配额。"""
    assert torch.cuda.is_available()
    fixture = ROOT/'tmp_smoke/formal_check_20261003/data_args.json'
    def run(name, resume, steps, inject=False):
        code = 'from src.st import train_formal as t\n'
        if inject:
            code += '''original=t.collect_documents
calls=0
def fail_second(*args, **kwargs):
    global calls
    calls+=1
    if calls==2: raise RuntimeError('injected preparation failure')
    return original(*args, **kwargs)
t.collect_documents=fail_second
'''
        code += 't.main()'
        command = [sys.executable, '-u', '-c', code, '--data_manifest', str(fixture),
                   '--resume', str(resume), '--output_dir', str(out/name),
                   '--steps', str(steps), '--log_every', '1']
        with (out/(name+'.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        assert (result.returncode != 0) == inject, (name, result.returncode)
    def exact(a, b):
        for name in ('model', 'optimizer'):
            def compare(x, y):
                if torch.is_tensor(x): torch.testing.assert_close(x, y, rtol=0, atol=0)
                elif isinstance(x, dict):
                    assert x.keys() == y.keys()
                    for key in x: compare(x[key], y[key])
                elif isinstance(x, list):
                    assert len(x) == len(y)
                    for left, right in zip(x, y): compare(left, right)
                else: assert x == y
            compare(a[name], b[name])
        for name in ('data', 'scheduler', 'scaler', 'counters', 'step', 'phase_step'):
            assert a[name] == b[name], name
        assert a['rng']['python'] == b['rng']['python']
        np.testing.assert_array_equal(a['rng']['numpy'][1], b['rng']['numpy'][1])
        assert a['rng']['numpy'][0] == b['rng']['numpy'][0]
        assert a['rng']['numpy'][2:] == b['rng']['numpy'][2:]
        torch.testing.assert_close(a['rng']['torch'], b['rng']['torch'], rtol=0, atol=0)
        for x, y in zip(a['rng']['cuda'], b['rng']['cuda']):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    run('one', parent, 1)
    run('continuous', parent, 2)
    run('injected', parent, 2, True)
    a = torch.load(out/'one/final.pt', map_location='cpu', weights_only=False)
    b = torch.load(out/'injected/data_error.pt', map_location='cpu', weights_only=False)
    exact(a, b)
    run('resumed', out/'injected/data_error.pt', 2)
    a = torch.load(out/'continuous/final.pt', map_location='cpu', weights_only=False)
    b = torch.load(out/'resumed/final.pt', map_location='cpu', weights_only=False)
    exact(a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--old_commit', default='ffbb511')
    parser.add_argument('--out_dir', type=Path, required=True)
    parser.add_argument('--fp16_parent', type=Path, help='已有小模型短检查的完整初始状态')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sources = json.loads(args.sources.read_text(encoding='utf-8'))
    font = VectorFont.from_svg(ROOT/'datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg')
    old = types.ModuleType('old_page_document_check')
    sys.modules[old.__name__] = old
    source = subprocess.check_output(['git', 'show', f'{args.old_commit}:src/page_document.py'], cwd=ROOT)
    exec(compile(source, str(ROOT/'src/page_document.py'), 'exec'), old.__dict__)
    ordinary = [sources['checkpoint_story']['text'], 'A cat sat here.\nThen it went home.',
                ' '.join(['short', 'words'] * 60)]
    for text in ordinary:
        before, after = old.layout_text_pages(text, font), layout_text_pages(text, font)
        assert len(before) == len(after)
        for a, b in zip(before, after):
            assert a.text == b.text and a.metadata == b.metadata
            equal(old.build_document([a]), build_document([b]))
    bad = sources['exceptional_stories'][0]
    old_doc = old.build_document(old.layout_text_pages(bad['text'], font))
    try:
        list(old_doc.iter_windows(1024))
    except ValueError as error:
        original_error = str(error)
        assert '2214' in original_error
    else:
        raise AssertionError('没有复现原故障')
    augment = partial(augment_rendered_text, rng=np.random.default_rng(42), config=TextAugmentationConfig())
    plain = layout_text_pages(bad['text'], font)
    warped = layout_text_pages(bad['text'], font, augment=augment)
    canonical = plain[0].metadata['canonical_text']
    assert ''.join(p.metadata['separator_before']+p.text for p in plain) == canonical
    for a, b in zip(plain, warped):
        assert a.text == b.text and a.metadata == b.metadata
    fixed = build_document(warped)
    windows = list(fixed.iter_windows(1024))
    assert all(d.stroke_count <= 1024 for d in windows)
    for d in windows:
        batch = collate_documents([d])
        assert len(set(batch['stroke_len'].tolist())) > 1
        for i, e in enumerate(d.to_sequence()):
            assert batch['stroke_len'][i] == len(e.rawxy)
    # 真实消费游标位于正常故事的第2窗口：修复后该窗口逐点不变。
    normal = sources['checkpoint_story']['text']
    a = list(old.build_document(old.layout_text_pages(normal, font)).iter_windows(1024))
    b = list(build_document(layout_text_pages(normal, font)).iter_windows(1024))
    for left, right in zip(a, b):
        equal(left, right)
    fixture = json.loads((ROOT/'tmp_smoke/formal_check_20261003/data_args.json').read_text(encoding='utf-8'))
    story_root = args.out_dir/'stories'
    story_root.mkdir()
    story_rows = [sources['checkpoint_story'], {'text':bad['text'], 'source_id':bad['source_id']},
                  {'text':'A dog ran home.', 'source_id':'check:short'}]
    offsets = []
    with (story_root/'stories.jsonl').open('wb') as f:
        for row in story_rows:
            offsets.append(f.tell())
            f.write((json.dumps(row)+'\n').encode('utf-8'))
    np.save(story_root/'offsets.npy', np.array(offsets, np.uint64))
    np.save(story_root/'train.npy', np.arange(3, dtype=np.uint64))
    np.save(story_root/'valid.npy', np.array([], dtype=np.uint64))
    manifest = json.loads((Path(fixture['story_index'])/'manifest.json').read_text(encoding='utf-8'))
    manifest['splits'] = {'train':3, 'valid':0}
    manifest['sha256'] = hashlib.sha256((story_root/'stories.jsonl').read_bytes()).hexdigest()
    (story_root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    data_args = {**fixture, 'story_index':str(story_root.resolve()), 'augment_text':True}
    dataset = build_document_dataset(**data_args)
    shares = {task:float(task == 'T1') for task in TASKS}
    serial = InputDocumentStream(dataset, data_args, shares, workers=0)
    serial.next_document()
    boundary = serial.state_dict()
    parallel = InputDocumentStream(dataset, data_args, shares, workers=4)
    parallel.load_state_dict(boundary)
    try:
        # 覆盖恢复的正常故事及异常长串故事的完整窗口。
        for _ in range(len(windows)+len(a)+3):
            equal(serial.next_document(), parallel.next_document())
            assert serial.state_dict() == parallel.state_dict()
    finally:
        parallel.close()
    boundary = serial.state_dict()
    original = serial.next_document
    calls = 0
    def injected(missing=()):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError('injected preparation failure')
        return original(missing)
    serial.next_document = injected
    try:
        collect_documents(serial, 65536)
    except ValueError:
        assert serial.state_dict() == boundary
        assert not serial.pending and not serial.windows
    else:
        raise AssertionError('没有触发注入故障')
    serial.next_document = original
    restored = InputDocumentStream(dataset, data_args, shares, workers=0)
    restored.load_state_dict(boundary)
    equal(serial.next_document(), restored.next_document())
    report = dict(original_error=original_error,task_index=bad['task_index'],source_id=bad['source_id'],
        max_word_chars=bad['max_word_chars'],fixed_pages=len(fixed.pages),
        max_fixed_page_strokes=max(len(p.elements) for p in fixed.pages),
        window_lengths=[d.stroke_count for d in windows],ordinary_layout_exact=True,
        original_text_restored=True,augmentation_layout_exact=True,native_variable_points=True,
        checkpoint_story_windows_exact=True,spawn4_resume_and_augmented_geometry_exact=True,
        failed_batch_cursor_rollback_and_replay_exact=True)
    if args.fp16_parent:
        fp16_emergency_check(args.fp16_parent, args.out_dir)
        report['fp16_emergency_checkpoint_and_resume_exact'] = True
    (args.out_dir/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
