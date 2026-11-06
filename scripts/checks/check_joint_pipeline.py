"""CPU核对成批预取的坐标/消费恢复、累计输入调度和Slurm接续终点。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.st.formal_recipe import CoverageWSD, InputWSD, TASKS
from src.st.formal_stream import InputDocumentStream
from src.st.formal_segment import segment_action
from src.st.pipeline_pretrain import build_document_dataset


def signature(document):
    digest = hashlib.sha256()
    digest.update(str(document.metadata.get('task')).encode())
    for element in document.to_sequence():
        digest.update(str((element.kind, element.page_id, element.char_index)).encode())
        digest.update(element.rawxy.tobytes())
    return digest.hexdigest()


def compare_streams(data_args, shares, *, context, count):
    dataset = build_document_dataset(**data_args)
    active = [task for task in TASKS if shares[task]]
    serial = InputDocumentStream(dataset, data_args, shares, workers=0, max_strokes=context)
    parallel = InputDocumentStream(dataset, data_args, shares, workers=2, max_strokes=context,
        prefetch_documents={t: 8 for t in active}, prepare_batch_documents={t: 4 for t in active})
    restored = None
    try:
        parallel.prefetch()
        for _ in range(count):
            assert signature(serial.next_document()) == signature(parallel.next_document())
            assert serial.state_dict() == parallel.state_dict()
            assert len(parallel.pending) <= 8 * len(active)
        boundary = deepcopy(parallel.state_dict())
        expected = [signature(serial.next_document()) for _ in range(8)]
        assert [signature(parallel.next_document()) for _ in range(8)] == expected
        parallel.rewind(boundary)
        assert [signature(parallel.next_document()) for _ in range(8)] == expected
        restored = InputDocumentStream(dataset, data_args, shares, workers=0, max_strokes=context)
        restored.load_state_dict(boundary)
        assert [signature(restored.next_document()) for _ in range(8)] == expected
        return dict(consumed=count, active=active, restored_window_offset=boundary['offsets']['T1'])
    finally:
        serial.close(); parallel.close()
        if restored: restored.close()


def main():
    torch.set_num_threads(2)
    root = Path('tmp_smoke/r_joint_pipeline_check_20261019');root.mkdir(parents=True, exist_ok=True)
    data_args = json.loads(Path('tmp_smoke/formal_check_20261003/data_args.json').read_text())
    shares = json.loads(Path('configs/r_joint_full_20261019.json').read_text())['input_shares']
    mixed = compare_streams(data_args, shares, context=1024, count=32)
    stories=root/'long_story';stories.mkdir(exist_ok=True)
    (stories/'stories.jsonl').write_text(json.dumps({'text':'Once a cat met a dog. They played near a tree. '*20,
                                                  'source_id':'long_fixture'})+'\n',encoding='utf-8')
    for name in ('offsets','train','valid'):np.save(stories/(name+'.npy'),np.array([0],dtype=np.uint64))
    (stories/'manifest.json').write_text(json.dumps({'kind':'indexed_stories_v1','fixture':True,'count':1}))
    text = compare_streams({**data_args,'story_index':str(stories)},
                          {t: float(t == 'T1') for t in TASKS}, context=128, count=1)
    assert text['restored_window_offset'] > 0
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    recipe = dict(peak_lr=1e-4, min_lr=1e-5, warmup_inputs=8000000,
        training_goal=dict(kind='all_task_pools', passes=1, decay_fraction=.04, limits={'T1':1686530}))
    old = CoverageWSD(optimizer, recipe, 1e-5);old.step(18000000, .001)
    before_lr = optimizer.param_groups[0]['lr']
    new, transition = InputWSD.from_coverage(old, recipe, 3600000000)
    assert new.inputs == 18000000 and new.spec['decay_inputs'] == 144000000
    assert optimizer.param_groups[0]['lr'] == before_lr
    assert recipe['coverage_source_limits'] == {'T1':1686530} and recipe['retry_failed_documents']
    assert new.lr_at(3456000000) == 1e-4
    assert abs(new.lr_at(3600000000) - 1e-5) < 1e-12
    clone = InputWSD(optimizer, recipe, 1e-5);clone.load_state_dict(new.state_dict())
    assert clone.inputs == new.inputs
    for inputs, reason, action in ((18, 'SIGUSR1', 'continue'), (3600000000, 'input_budget', 'complete'),
                                  (18, 'SIGTERM', 'stop'), (18, 'steps', 'stop')):
        assert segment_action(dict(target_inputs=3600000000, successful_inputs=inputs),
                              dict(reason=reason), dict(progress=1)) == action
    assert segment_action({}, dict(reason='data_coverage'), dict(progress=1)) == 'complete'
    report = dict(mixed=mixed, text_window=text, budget_transition=transition,
                  batched_geometry_and_cursor_match=True, bounded_prefetch=True,
                  rewind_and_checkpoint_cursor_match=True, segmented_budget_stop=True)
    (root/'checks.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__': main()
