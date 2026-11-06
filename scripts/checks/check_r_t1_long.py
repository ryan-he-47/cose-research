"""2026-10-09：R方差平方和、旧头兼容、冻结AE和FP16分段恢复短核对。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.st.config_st import STConfig
from src.st.formal_recipe import DEFAULT_RECIPE, InputWSD
from src.st.formal_stream import InputDocumentStream
from src.st.model import GMMDense, gmm_nll
from src.st.pipeline_pretrain import build_document_dataset, collate_documents
from src.st.pretrain_model import FrozenSketchLM, config_from_ae_checkpoint, config_from_r_checkpoint
from src.st.train_formal import ae_source_record, phase_data_args, save_formal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir',type=Path,default=ROOT/'tmp_smoke/r_t1_long_check_20261009')
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    recipe = json.loads((ROOT/'configs/r_t1_long_20261009.json').read_text(encoding='utf-8'))
    assert recipe['total_inputs']==20*86439968 and recipe['sigma_eps_r']==.003
    assert recipe['r_sigma_floor_mode']=='quadrature' and not recipe['augment_text']
    old_cfg = config_from_r_checkpoint({'cfg':{'sigma_eps_r':1e-4}})
    assert old_cfg.r_sigma_floor_mode=='add'
    head = GMMDense(8,48,32,old_cfg).cuda()
    h = torch.randn(1,4,8,device='cuda',requires_grad=True)
    with torch.autocast('cuda',dtype=torch.float16):
        mu, sigma, pi = head(h)
    raw = head.head(h.float())
    pieces = raw.split((48*32,48*32,32),dim=-1)
    torch.testing.assert_close(sigma, pieces[1].clamp(-15,5).exp().view_as(sigma)+old_cfg.sigma_eps,rtol=0,atol=0)
    # 与只读测量时的forward hook计算及梯度逐项相同。
    reference = GMMDense(8,48,32,old_cfg,sigma_eps=0).cuda()
    reference.load_state_dict(head.state_dict())
    hook = reference.register_forward_hook(lambda module,inputs,result:
        (result[0],torch.sqrt(result[1].square()+.003**2),result[2]))
    head.sigma_eps, head.sigma_floor_mode = .003,'quadrature'
    for module in (reference,head):
        with torch.autocast('cuda',dtype=torch.float16):
            prediction = module(h)
            loss = gmm_nll(torch.randn(4,48,device='cuda')*0,
                           *(v.squeeze(0) for v in prediction))
        loss.backward()
        if module is reference:
            expected = tuple(v.detach().clone() for v in prediction)
            expected_grad = reference.head.weight.grad.clone()
        else:
            for a,b in zip(prediction,expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(module.head.weight.grad,expected_grad,rtol=0,atol=0)
    hook.remove()
    for raw_scale in (-40.,-15.,-6.,0.,5.,40.):
        head.zero_grad(set_to_none=True)
        with torch.no_grad():
            head.head.weight.zero_()
            head.head.bias.zero_()
            head.head.bias[48*32:2*48*32].fill_(raw_scale)
        with torch.autocast('cuda',dtype=torch.float16):
            mu,sigma,pi = head(h)
            loss = gmm_nll(torch.full((4,48),.001,device='cuda'),mu.squeeze(0),sigma.squeeze(0),pi.squeeze(0))
        loss.backward()
        assert sigma.dtype==torch.float32 and bool((sigma>=.003).all())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    del head, reference, h, raw, pieces, prediction, expected, expected_grad, loss, mu, sigma, pi
    gc.collect()
    torch.cuda.empty_cache()
    fixture = ROOT/'tmp_smoke/formal_check_20261003/data_args.json'
    data_args = phase_data_args(json.loads(fixture.read_text(encoding='utf-8')),recipe)
    dataset = build_document_dataset(**data_args)
    doc = dataset.get_for_task('T1',0)
    batch = collate_documents([doc])
    assert not dataset.augmentation.enabled and len(set(batch['stroke_len'].tolist()))>1
    for i,element in enumerate(doc.to_sequence()):
        if len(element.rawxy)<=76:
            assert batch['stroke_len'][i]==len(element.rawxy)
            np.testing.assert_array_equal(batch['ink'][i,:len(element.rawxy)].numpy(),
                (element.rawxy-np.array([.5,.6],np.float32))/.6)
    ae_path = ROOT/'tmp_smoke/r_dual_review_assets_20261009/ae_final.pt'
    ae = torch.load(ae_path,map_location='cpu',weights_only=False)
    cfg = replace(config_from_ae_checkpoint(ae),r_layers=2,r_d_model=32,r_heads=4,r_dff=64,
                  sigma_eps_r=.003,r_sigma_floor_mode='quadrature')
    model = FrozenSketchLM(cfg,encode_chunk_size=256,activation_checkpoint=False)
    model.load_ae(ae)
    assert model.decoder.head.sigma_floor_mode=='add' and model.r_pred.head.sigma_floor_mode=='quadrature'
    optimizer = torch.optim.Adam(model.r_pred.parameters(),lr=1e-4)
    small = {**recipe,'total_inputs':8192,'warmup_inputs':1024,'decay_inputs':1024,
        'input_budget':256,'padded_stroke_budget':1024,'microbatch_documents':4,
        'encode_chunk_size':256,'workers':0,'memory_headroom_gib':.5}
    stream = InputDocumentStream(dataset,data_args,small['input_shares'],workers=0)
    scheduler = InputWSD(optimizer,small,0.)
    save_formal(out/'parent.pt',model,optimizer,scheduler,torch.amp.GradScaler('cuda',init_scale=256),stream,
        step=0,phase_step=0,counters=Counter(),data_args=data_args,manifest=dataset.manifest(),recipe=small,
        ae_source=ae_source_record(ae_path,ae),parent_source=None)
    del model,optimizer,ae
    (out/'recipe.json').write_text(json.dumps(small),encoding='utf-8')
    def run(name,source,steps,recipe_path=None,success=True):
        command = [sys.executable,'-u','src/st/train_formal.py','--data_manifest',str(fixture),
            '--recipe',str(recipe_path or out/'recipe.json'),'--output_dir',str(out/name),
            '--steps',str(steps),'--log_every','1',*source]
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            result = subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                    env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        assert (result.returncode==0)==success,(name,result.returncode)
    run('continuous',['--resume',str(out/'parent.pt')],2)
    run('split',['--resume',str(out/'parent.pt')],1)
    run('resumed',['--resume',str(out/'split/final.pt')],2)
    a = torch.load(out/'continuous/final.pt',map_location='cpu',weights_only=False)
    b = torch.load(out/'resumed/final.pt',map_location='cpu',weights_only=False)
    for name,value in a['model'].items():
        torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
    for key,value in a['optimizer']['state'].items():
        for field,tensor in value.items():
            torch.testing.assert_close(tensor,b['optimizer']['state'][key][field],rtol=0,atol=0)
    for name in ('data','counters','scheduler','scaler'):
        assert a[name]==b[name],name
    assert a['rng']['python']==b['rng']['python']
    assert a['rng']['numpy'][0]==b['rng']['numpy'][0]
    np.testing.assert_array_equal(a['rng']['numpy'][1],b['rng']['numpy'][1])
    assert a['rng']['numpy'][2:]==b['rng']['numpy'][2:]
    torch.testing.assert_close(a['rng']['torch'],b['rng']['torch'],rtol=0,atol=0)
    for left,right in zip(a['rng']['cuda'],b['rng']['cuda']):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    frozen = torch.load(ae_path,map_location='cpu',weights_only=False)
    for name,value in frozen['model'].items():
        torch.testing.assert_close(value,b['model'][name],rtol=0,atol=0)
    bad = {**small,'r_sigma_floor_mode':'add'}
    (out/'bad.json').write_text(json.dumps(bad),encoding='utf-8')
    run('reject_changed_semantics',['--resume',str(out/'split/final.pt')],2,out/'bad.json',False)
    assert 'R的σ参数化不同' in (out/'reject_changed_semantics.log').read_text(encoding='utf-8')
    del a,b,frozen
    gc.collect()
    # 实际12L/d768/λ48模型从55k AE初始化，FP16只做一个短更新。
    run('actual_size',['--init_ae',str(ae_path),'--expected_ae_step','55000','--expected_ae_lr','5e-6'],1)
    trained = torch.load(out/'actual_size/final.pt',map_location='cpu',weights_only=False)
    assert trained['step']==1 and trained['cfg']['r_sigma_floor_mode']=='quadrature'
    assert trained['cfg']['sigma_eps_r']==.003 and trained['cfg']['r_layers']==12
    assert trained['cfg']['r_d_model']==768 and not trained['data_manifest']['augmentation']['enabled']
    lines = [json.loads(line) for line in (out/'actual_size.log').read_text(encoding='utf-8').splitlines() if line.startswith('{')]
    start = next(row for row in lines if row.get('formal_start'))
    assert start['r_parameters']==86439968 and start['active_tasks']==['T1']
    assert trained['counters'].get('skipped_updates',0)==0
    frozen = torch.load(ae_path,map_location='cpu',weights_only=False)
    for name,value in frozen['model'].items():
        torch.testing.assert_close(value,trained['model'][name],rtol=0,atol=0)
    report = {'old_cfg_and_add_formula_unchanged':True,'measurement_output_and_gradient_exact':True,
        'fp16_extreme_scales_finite_std_floor':True,'ae_head_add_and_frozen_weights_unchanged':True,
        'native_variable_points_and_t1_no_augmentation':True,'fp16_segment_resume_exact':True,
        'changed_semantics_resume_rejected':True,'actual_86439968_parameter_r_fp16_update':True,
        'actual_size_successful_inputs':trained['counters']['successful_inputs']}
    (out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    main()
