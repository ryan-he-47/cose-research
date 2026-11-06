"""2026-10-07：FiLM冻结来源、正式R尺寸/WSD、真实合成问答与FP16起跑短核对。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.review.review_formal_pipeline import ReviewDrawings
from src.formal_scene_tasks import _area, _contains, _descriptors, _caption
from src.st.formal_recipe import DEFAULT_RECIPE, InputWSD
from src.st.pretrain_model import FrozenSketchLM, config_from_ae_checkpoint
from src.st.train_formal import ae_source_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fp16',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = ROOT/'tmp_smoke/formal_launch_check_20261007'
    out.mkdir(parents=True,exist_ok=True)
    path = ROOT/'tmp_smoke/ae_finish_20261007/parent40000.pt'
    state = torch.load(path,map_location='cpu',weights_only=False)
    font = ROOT/'datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg'
    assert hashlib.sha256(font.read_bytes()).hexdigest() == state['data_manifest']['font_sha256']
    source = ae_source_record(path,state,expected_step=40000,expected_lr=1e-4)
    try:
        ae_source_record(path,state,expected_step=55000,expected_lr=5e-6)
    except ValueError:
        pass
    else:
        raise AssertionError('未完成AE错误通过正式检查')
    cfg = config_from_ae_checkpoint(state)
    model = FrozenSketchLM(cfg,encode_chunk_size=1024,activation_checkpoint=False)
    model.load_ae(state)
    assert cfg.latent_units == 48 and cfg.dec_tau_mode == 'film'
    assert (cfg.r_layers,cfg.r_d_model,cfg.r_heads,cfg.r_dff,cfg.max_context_strokes) == (12,768,12,3072,1024)
    assert all(not p.requires_grad for name,p in model.named_parameters() if not name.startswith('r_pred.'))
    r_count = sum(p.numel() for p in model.r_pred.parameters())
    optimizer = torch.optim.Adam(model.r_pred.parameters(),lr=1e-4)
    schedule = InputWSD(optimizer,DEFAULT_RECIPE,0.)
    schedule.prepare_update(65536)
    assert optimizer.param_groups[0]['lr'] == 1e-4*65536/8000000
    assert schedule.inputs == 0
    schedule.step(65536)
    assert schedule.inputs == 65536
    assert schedule.lr_at(8000000) == schedule.lr_at(192000000) == 1e-4
    assert np.isclose(schedule.lr_at(200000000),1e-5)
    del optimizer,model,state
    samples = json.loads((ROOT/'tmp_smoke/formal_pipeline_review_20261007/samples.json').read_text(encoding='utf-8'))
    verified = Counter()
    for sample in samples:
        meta = sample['metadata']
        objects = meta['objects']
        names = _descriptors(objects)
        if sample['task'] == 'S1':
            assert meta['answer'] == _caption(objects)
            for obj in objects:
                assert obj['label'] in meta['answer']
            verified['captions'] += 1
        if sample['task'] == 'S3':
            query = meta['query']
            if query['kind'] in ('count','existence'):
                count = sum(obj['label'] == query['label'] for obj in objects)
                assert query['value'] == count
                assert meta['answer'] == (str(count) if query['kind'] == 'count' else 'yes' if count else 'no')
            if query['kind'] in ('size','distance','reference'):
                assert meta['answer'] == names[query['target']]
            verified['questions'] += 1
        if sample['task'] in ('S4','S5'):
            target = next(obj for obj in objects if obj['instance_id'] == meta['target_instance_id'])
            assert all(not _contains(meta['selection_box'],obj['center']) for obj in objects if obj is not target)
            if sample['task'] == 'S5':
                assert meta['answer'] == target['label']
            verified['selections'] += 1
    if args.fp16:
        recipe = {**DEFAULT_RECIPE,'total_inputs':8192,'warmup_inputs':2048,'decay_inputs':2048,
                  'input_budget':1024,'padded_stroke_budget':1024,'microbatch_documents':4,
                  'encode_chunk_size':256,'workers':0}
        recipe_path = out/'recipe.json'
        recipe_path.write_text(json.dumps(recipe),encoding='utf-8')
        command = [sys.executable,'-u','src/st/train_formal.py','--data_manifest',
            str(ROOT/'tmp_smoke/formal_check_20261003/data_args.json'),'--recipe',str(recipe_path),
            '--init_ae',str(path),'--expected_ae_step','40000','--expected_ae_lr','1e-4',
            '--output_dir',str(out/'fp16'),'--steps','1','--log_every','1','--max_seconds','0']
        with (out/'fp16.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        saved = torch.load(out/'fp16/final.pt',map_location='cpu',weights_only=False)
        assert saved['step'] == 1 and saved['counters']['successful_inputs'] > 0
        assert saved['scheduler']['inputs'] == saved['counters']['successful_inputs']
        assert saved['optimizer']['param_groups'][0]['lr'] > 0
        assert saved['ae_source']['sha256'] == source['sha256']
        assert all(int(s['step']) == 1 for s in saved['optimizer']['state'].values())
        frozen = torch.load(path,map_location='cpu',weights_only=False)
        for name,tensor in frozen['model'].items():
            torch.testing.assert_close(tensor,saved['model'][name],rtol=0,atol=0)
    report = {'r_parameters':r_count,'ae_source':source,'frozen_film_load':True,'incomplete_ae_rejected':True,
              'wsd_first_lr_nonzero':True,'real_samples_checked':dict(verified),'fp16_one_update':args.fp16}
    (out/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key != 'ae_source'}))


if __name__ == '__main__':
    main()
