"""2026-10-09：两臂入口、T1预取、原点数、Adam重置及FP16完整恢复短核对。"""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.st.formal_recipe import DEFAULT_RECIPE, InputWSD, TASKS
from src.st.formal_stream import InputDocumentStream, collect_documents
from src.st.pipeline_pretrain import build_document_dataset, collate_documents
from src.st.pretrain_model import FrozenSketchLM, config_from_ae_checkpoint
from src.st.train_formal import ae_source_record, materialize_adam, phase_data_args, save_formal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir',type=Path,default=ROOT/'tmp_smoke/r_dual_check_20261009')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = args.out_dir
    out.mkdir(parents=True,exist_ok=False)
    fixture = ROOT/'tmp_smoke/formal_check_20261003/data_args.json'
    data_args = json.loads(fixture.read_text(encoding='utf-8'))
    t1_recipe = json.loads((ROOT/'configs/r_t1_plain_20261009.json').read_text())
    corrective = json.loads((ROOT/'configs/r_corrective_20261009.json').read_text())
    for recipe in (t1_recipe,corrective):
        assert recipe['total_inputs']==200000000 and recipe['memory_headroom_gib']==6
    assert t1_recipe['loss_weights']==DEFAULT_RECIPE['loss_weights']
    assert 'sigma_eps_r' not in t1_recipe and corrective['sigma_eps_r']==.01
    # 即使manifest包含增强配置，false仍关闭全部字形变换。
    plain_args = phase_data_args({**data_args,'augmentation':{'enabled':True}},t1_recipe)
    plain = build_document_dataset(**plain_args)
    assert not plain.augmentation.enabled
    a,b = plain.get_for_task('T1',0),plain.get_for_task('T1',len(plain.stories))
    assert a.metadata['text']==b.metadata['text']
    for x,y in zip(a.to_sequence(),b.to_sequence()):
        np.testing.assert_array_equal(x.rawxy,y.rawxy)
    batched = collate_documents([a])
    lengths = batched['stroke_len'].tolist()
    assert len(set(lengths))>1 and any(n<76 for n in lengths)
    for i,element in enumerate(a.to_sequence()):
        if len(element.rawxy)<=76:
            assert lengths[i]==len(element.rawxy)
            np.testing.assert_allclose(batched['ink'][i,:lengths[i]].numpy(),
                (element.rawxy-np.array([.5,.6],np.float32))/.6,atol=0,rtol=0)
    serial = InputDocumentStream(plain,plain_args,t1_recipe['input_shares'],workers=0)
    docs,inputs,body,control = collect_documents(serial,1024)
    assert set(inputs)==set(body)=={'T1'} and control>0
    saved = serial.state_dict()
    expected = serial.next_document()
    parallel = InputDocumentStream(plain,plain_args,t1_recipe['input_shares'],workers=4)
    parallel.load_state_dict(saved)
    parallel.prefetch()
    assert len(parallel.pending)==4 and all(task=='T1' for task,index in parallel.pending)
    replay = parallel.next_document()
    for x,y in zip(expected.to_sequence(),replay.to_sequence()):
        np.testing.assert_array_equal(x.rawxy,y.rawxy)
    parallel.close()
    dataset = build_document_dataset(**data_args)
    stream = InputDocumentStream(dataset,data_args,DEFAULT_RECIPE['input_shares'],workers=0)
    collect_documents(stream,1024)
    parent_data = stream.state_dict()
    changed = InputDocumentStream(dataset,data_args,corrective['input_shares'],workers=0)
    changed.continue_positions(parent_data)
    assert changed.indices==stream.indices and changed.offsets==stream.offsets and sum(changed.inputs.values())==0
    ae_path = ROOT/'tmp_smoke/ae_finish_20261007/parent40000.pt'
    ae = torch.load(ae_path,map_location='cpu',weights_only=False)
    cfg = replace(config_from_ae_checkpoint(ae),r_layers=2,r_d_model=32,r_heads=4,r_dff=64)
    model = FrozenSketchLM(cfg,encode_chunk_size=256,activation_checkpoint=False)
    model.load_ae(ae)
    optimizer = torch.optim.Adam(model.r_pred.parameters(),lr=1e-4)
    materialize_adam(optimizer)
    assert all(float(s['step'])==0 and not bool(s['exp_avg'].any()) for s in optimizer.state.values())
    for value in optimizer.state.values():
        value['step'].fill_(7)
        value['exp_avg'].fill_(.1)
        value['exp_avg_sq'].fill_(.2)
    parent_recipe = {**DEFAULT_RECIPE,'workers':0}
    scheduler = InputWSD(optimizer,parent_recipe,0.)
    scheduler.step(200000833)
    save_formal(out/'parent.pt',model,optimizer,scheduler,torch.amp.GradScaler('cuda'),stream,
        step=7,phase_step=7,counters=Counter(successful_inputs=200000833),data_args=data_args,
        manifest=dataset.manifest(),recipe=parent_recipe,ae_source=ae_source_record(ae_path,ae),parent_source=None)
    del model,optimizer,ae
    small = {**corrective,'total_inputs':8192,'warmup_inputs':2048,'decay_inputs':2048,
        'input_budget':1024,'padded_stroke_budget':1024,'microbatch_documents':4,
        'encode_chunk_size':256,'workers':0,'memory_headroom_gib':.5}
    (out/'corrective.json').write_text(json.dumps(small),encoding='utf-8')
    def run(name,source,steps,recipe):
        command = [sys.executable,'-u','src/st/train_formal.py','--data_manifest',str(fixture),
            '--recipe',str(recipe),'--output_dir',str(out/name),'--steps',str(steps),
            '--log_every','1','--max_seconds','0',*source]
        if name == 'split':
            command.append('--calibrate')
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    run('continuous',['--warm_start',str(out/'parent.pt')],2,out/'corrective.json')
    run('split',['--warm_start',str(out/'parent.pt')],1,out/'corrective.json')
    run('resumed',['--resume',str(out/'split/final.pt')],2,out/'corrective.json')
    continuous = torch.load(out/'continuous/final.pt',map_location='cpu',weights_only=False)
    resumed = torch.load(out/'resumed/final.pt',map_location='cpu',weights_only=False)
    for key,value in continuous['model'].items():
        torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)
    for key,value in continuous['optimizer']['state'].items():
        for field,tensor in value.items():
            torch.testing.assert_close(tensor,resumed['optimizer']['state'][key][field],rtol=0,atol=0)
    for key in ('data','counters','scheduler','scaler'):
        assert continuous[key]==resumed[key],key
    assert resumed['step']==9 and resumed['phase_step']==2
    assert resumed['cfg']['sigma_eps_r']==.01
    assert all(float(s['step'])==2 for s in resumed['optimizer']['state'].values())
    starts = [json.loads(line) for line in (out/'split.log').read_text().splitlines() if line.startswith('{')]
    start = next(row for row in starts if row.get('formal_start'))
    assert start['optimizer_state_entries']>0 and start['optimizer_step_max']==0
    assert start['data_positions']['indices']==parent_data['indices']
    assert start['data_positions']['offsets']==parent_data['offsets']
    assert start['parent']['successful_inputs']==200000833
    update = next(row for row in starts if row.get('success'))
    assert set(update['gmm_buckets'])=={'text','sketch','selection','control'}
    plain_small = {**small,**{key:t1_recipe[key] for key in ('version','augment_text','input_shares','loss_weights')}}
    plain_small.pop('sigma_eps_r')
    (out/'t1.json').write_text(json.dumps(plain_small),encoding='utf-8')
    run('t1',['--init_ae',str(ae_path),'--expected_ae_step','40000','--expected_ae_lr','1e-4'],1,out/'t1.json')
    trained = torch.load(out/'t1/final.pt',map_location='cpu',weights_only=False)
    assert trained['step']==trained['phase_step']==1 and trained['cfg']['sigma_eps_r']==1e-4
    assert not trained['data_manifest']['augmentation']['enabled']
    assert set(k.removeprefix('inputs:') for k,v in trained['counters'].items() if k.startswith('inputs:') and v)=={'T1'}
    assert set(k.removeprefix('body:') for k,v in trained['counters'].items() if k.startswith('body:') and v)=={'T1'}
    frozen = torch.load(ae_path,map_location='cpu',weights_only=False)
    for state in (trained,resumed):
        for name,value in frozen['model'].items():
            torch.testing.assert_close(value,state['model'][name],rtol=0,atol=0)
    report = {'t1_only_collect_and_four_worker_prefetch':True,'text_augmentation_disabled':True,
        'variable_point_counts_unchanged':True,'warm_start_weights_and_positions':True,
        'adam_and_stage_budget_reset':True,'fp16_resume_model_adam_data_wsd_scaler_exact':True,
        'frozen_ae_unchanged':True,'t1_actual_gpt2_dimensions_fp16_one_update':True,
        't1_original_sigma_and_loss_weights':True,'gmm_buckets_logged':True}
    (out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    main()
