"""2026-10-04：独立AE源的消费恢复、弧长目标、梯度与R冻结接口核验。"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.st.ae_data import StrokeRecord
from src.st.ae_direct import DirectAEStream, ArcAECollator, source_block
from src.st.ae_geometry import arc_resample, error_components, summarize
from src.st.formal_assets import write_json
from src.st.formal_assets import OffsetJSONReader
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.pretrain_model import FrozenSketchLM, config_from_ae_checkpoint
from src.st.train_ae import StrokeAutoencoder, load_ae_weights


def same_records(left,right):
    assert len(left)==len(right)
    for a,b in zip(left,right):
        assert (a.kind,a.document_index,a.page_id)==(b.kind,b.document_index,b.page_id)
        np.testing.assert_array_equal(a.rawxy,b.rawxy)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,default=ROOT/'tmp_smoke/formal_check_20261003')
    parser.add_argument('--out_dir',type=Path,default=ROOT/'tmp_smoke/ae_direct_check_20261004')
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        cfg=config_from_ae_checkpoint(state)
        frozen=FrozenSketchLM(cfg)
        frozen.load_ae(state)
        assert all(not p.requires_grad for name,p in frozen.named_parameters()
                   if name.startswith(('encoder.','decoder.','dct_head.')))
        for name,value in state['model'].items():
            torch.testing.assert_close(frozen.state_dict()[name],value,rtol=0,atol=0)
        assert all(torch.isfinite(value).all() for value in state['model'].values())
        print(json.dumps({'r_fresh_init_ae_equal':True,'ae_step':state['step'],
            'phase':state['phase'],'optimizer_state_entries':len(state['optimizer']['state']),
            'strokes':state['stroke_counts']}))
        return
    data_args=json.loads((args.fixture/'data_args.json').read_text())
    data_args['scene_index']=str((args.fixture/'materialized/train').resolve())
    write_json(args.out_dir/'data_args.json',data_args)
    stream=DirectAEStream(data_args,batch_size=64,workers=0)
    # 新mmap读取对照原有seek/readline；覆盖RAW、正文、配方及LRU关闭/重开。
    reader=OffsetJSONReader(max_files=1)
    for source,path,offsets in (
        (stream.dataset.drawings,None,stream.dataset.drawings.indices[:20]),
        (stream.dataset.stories,stream.dataset.stories.root/'stories.jsonl',stream.dataset.stories.offsets),
        (stream.dataset.recipes,stream.dataset.recipes.root/'recipes.jsonl',stream.dataset.recipes.offsets[:20])):
        for value in offsets:
            actual_path,offset=(source.files[int(value[0])],int(value[1])) if path is None else (path,int(value))
            with actual_path.open('rb') as original:
                original.seek(offset);expected_json=json.loads(original.readline())
            assert reader.read(actual_path,offset)==expected_json
    reader.close()
    first=stream.next_batch()
    cursor=stream.state_dict()
    stream.prefetch()
    expected=stream.next_batch()
    resumed=DirectAEStream(data_args,batch_size=64,workers=2)
    resumed.load_state_dict(cursor);resumed.prefetch()
    try:
        same_records(expected,resumed.next_batch())
    finally:
        resumed.close()
    assert Counter(record.kind for record in first)['sketch']==32
    cfg=gpt2_small_pretrain_config();cfg.freeze_ed=False;cfg.n_t_samples=16
    collator=ArcAECollator(cfg)
    state=collator.state_dict();batch=collator(first)
    restored=ArcAECollator(cfg);restored.load_state_dict(state)
    for name,value in restored(first).items():
        if torch.is_tensor(value):
            torch.testing.assert_close(batch[name],value,rtol=0,atol=0)
    # 点间距极不均匀的直线：目标应随弧长τ推进，不随原始点索引推进。
    raw=np.array([[.1,.2],[.101,.2],[.9,.2]],np.float32)
    special=collator([StrokeRecord(raw,'sketch',0,0)])
    x=(.1+.8*special['tau'].numpy()[0]-.5)/.6
    np.testing.assert_allclose(special['tau_target'].numpy()[0,:,0],x,atol=2e-7)
    model=StrokeAutoencoder(cfg)
    result=model(batch);result['loss'].backward()
    for name in ('encoder','decoder','dct_head'):
        grads=[p.grad for p in getattr(model,name).parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    # 小型有效几何，只用于本机FP16通路检查。
    raw=[];buckets=[]
    for source,bucket in (('text','text'),('object','sketch_object'),('scene','sketch_scene')):
        records=source_block(stream.dataset,source,0)[:16]
        raw.extend(record.rawxy for record in records);buckets.extend([bucket]*len(records))
    ink=np.zeros((len(raw),76,2),np.float32)
    lengths=np.array(list(map(len,raw)),np.int64)
    for i,points in enumerate(raw):
        ink[i,:len(points)]=(points-np.array([.5,.6],np.float32))/.6
    np.savez_compressed(args.out_dir/'samples.npz',ink=ink,lengths=lengths,
        cloud=np.stack([arc_resample(p,32) for p in raw]),bucket=np.array(buckets))
    point=error_components(np.ones((32,2)),np.ones((32,2)))
    assert np.isnan(point[-1]) and summarize(np.array([point]))['degenerate_point_strokes']==1
    recipe=json.loads((ROOT/'configs/pretrain_ae_direct_recipe.json').read_text())
    recipe.update(batch_size=64,microbatch_strokes=16,padded_point_budget=1024,
                  workers=2,warmup=2,validate_seconds=3600,save_seconds=.001)
    write_json(args.out_dir/'recipe.json',recipe)
    write_json(args.out_dir/'checks.json',{'worker_independent_replay':True,'arc_length_targets':True,
        'mmap_matches_buffered_json':True,
        'encoder_decoder_dct_gradients_finite':True,'quotas':stream.quotas,'kind_counts':dict(Counter(r.kind for r in first))})
    stream.close()
    print((args.out_dir/'checks.json').read_text())


if __name__=='__main__':
    main()
