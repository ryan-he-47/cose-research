"""2026-10-14：辅助监督回传R、拆微批归一、完整状态和梯度校准的CPU检查。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict,replace
import json
import numpy as np
import torch
from src.page_document import build_document,layout_text_pages
from src.text_renderer import VectorFont
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.pretrain_model import FrozenSketchLM,config_from_r_checkpoint
from src.st.pipeline_pretrain import collate_documents
from src.st.text_readout import StrokeVocabulary,document_trace
from src.st.text_shape_readout import ShapeVocabulary,ShapeSpaceReadout
from src.st.text_auxiliary import (annotate_documents,bind_auxiliary,weight_hash,calibrate_auxiliary,
                                   validate_auxiliary,configure_recalibration,
                                   auxiliary_calibration_due,apply_auxiliary_calibration)
from src.st.train_formal import task_loss,materialize_adam,save_formal,gpu_batch
from src.st.formal_recipe import DEFAULT_RECIPE,InputWSD,TASKS
from src.st.formal_stream import InputDocumentStream,collect_documents
from src.st.pipeline_pretrain import PretrainDocumentDataset,Story
from src.st.train_pretrain import _rng_state,_restore_rng


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir',type=Path,default=Path('tmp_smoke/r_text_aux_check_20261014'))
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(42)
    font=VectorFont.from_svg('datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg')
    vocab=StrokeVocabulary(font);catalog=ShapeVocabulary(vocab)
    documents=[build_document(layout_text_pages(t,font),complete=True,metadata={'task':'T1'})
               for t in ['A cat sat. It was happy!', 'The little kitten ran to her home. '*3]]
    n=annotate_documents(documents,catalog)
    for doc in documents:
        labels,_=document_trace(doc,vocab)
        targets=doc.metadata['_text_aux_targets']
        assert np.array_equal(targets['shape'],catalog.mapping[labels])
        assert np.array_equal(targets['space'],catalog.leading[labels])
    cfg=replace(gpt2_small_pretrain_config(),enc_d_model=24,enc_layers=1,enc_heads=4,enc_dff=48,
                latent_units=16,dec_layers=1,dec_hidden=32,gmm_dec=2,r_d_model=32,
                r_layers=2,r_heads=4,r_dff=64,gmm_r=3,max_context_strokes=1024)
    model=FrozenSketchLM(cfg)
    optimizer=torch.optim.Adam(model.r_pred.parameters(),lr=1e-5);materialize_adam(optimizer)
    model(collate_documents(documents))['loss'].backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    original_state=deepcopy(model.state_dict())
    original_optimizer=deepcopy(optimizer.state_dict())
    fake_head=ShapeSpaceReadout(cfg.r_d_model,catalog,16)
    head_path=args.out_dir/'fixture_head.pt'
    torch.save(dict(kind='shape_space',head=fake_head.state_dict(),hidden=16,input_width=32,
                    shape_vocabulary=catalog.dump(),font_sha256='fixture',r_model_hash=weight_hash(model)),head_path)
    _,manifest=bind_auxiliary(model,font,'fixture',head_path=head_path)
    optimizer.add_param_group({'params':model.text_aux.parameters(),'role':'text_aux'})
    materialize_adam(optimizer)
    assert all(torch.equal(a,b) for p,a in original_state.items() for b in [model.state_dict()[p]])
    assert optimizer.state_dict()['param_groups'][0]==original_optimizer['param_groups'][0]
    batch=collate_documents(documents);batch['task_id']=torch.zeros(len(documents),dtype=torch.long)
    model.train();result=model(batch)
    aux=(result['aux_all_ce_sum']+result['aux_all_bce_sum'])/n
    aux.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.r_pred.layers.parameters())
    assert all(p.grad is None for p in model.r_pred.head.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.text_aux.parameters())
    assert result['aux_all_count']==n and result['aux_control_count']>0 and result['aux_continuation_count']>0
    model.zero_grad(set_to_none=True)
    body=Counter(T1=sum(e.targetable and e.kind=='text' for d in documents for e in d.to_sequence()[1:]))
    control=n-body['T1'];recipe={**DEFAULT_RECIPE,'text_auxiliary':True,'text_aux_alpha':.7,
        'microbatch_documents':2,'padded_stroke_budget':8192,'text_aux_probe_batches':8}
    loss=task_loss(model(batch),body,control,recipe,auxiliary_count=n);loss.backward()
    reference={name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    for d in documents:
        part=collate_documents([d]);part['task_id']=torch.zeros(1,dtype=torch.long)
        task_loss(model(part),body,control,recipe,auxiliary_count=n).backward()
    assert all(torch.allclose(p.grad,reference[name],atol=2e-5,rtol=2e-4)
               for name,p in model.named_parameters() if name in reference)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert all(torch.equal(a,model.state_dict()[p]) for p,a in original_state.items() if p.startswith(('encoder.','decoder.','dct_head.')))
    before_hash=weight_hash(model);rng=torch.get_rng_state().clone()
    calibration=calibrate_auxiliary(model,documents,body,control,recipe,torch.device('cpu'),
                                    task_loss,gpu_batch,_rng_state,_restore_rng)
    assert calibration['alpha']>0 and weight_hash(model)==before_hash and torch.equal(rng,torch.get_rng_state())
    valid_path=args.out_dir/'valid.jsonl'
    valid_path.write_text(json.dumps({'text':'A cat sat.','source_id':'fixture-valid'})+'\n',encoding='utf-8')
    valid=validate_auxiliary(model,font,catalog,valid_path,{},torch.device('cpu'),_rng_state,_restore_rng)
    assert valid['features']=='online_current_R' and valid['text_aux']['all']['count']>0
    assert weight_hash(model)==before_hash and model.training and torch.equal(rng,torch.get_rng_state())
    recipe['text_aux_alpha']=calibration['alpha'];model.cfg.r_text_aux_weight=calibration['alpha']
    original_total=recipe['total_inputs']
    configure_recalibration(recipe,.1,20000000,inputs=123,step=7,source='fixture.pt')
    assert auxiliary_calibration_due(recipe,123) and recipe['total_inputs']==original_total
    recalibration=calibrate_auxiliary(model,documents,body,control,recipe,torch.device('cpu'),
                                      task_loss,gpu_batch,_rng_state,_restore_rng)
    assert abs(recalibration['alpha']/calibration['alpha']-.2)<1e-7
    assert weight_hash(model)==before_hash and torch.equal(rng,torch.get_rng_state())
    recalibration=apply_auxiliary_calibration(recipe,recalibration,inputs=123,step=7)
    model.cfg.r_text_aux_weight=recipe['text_aux_alpha']
    assert not auxiliary_calibration_due(recipe,20000122) and auxiliary_calibration_due(recipe,20000123)
    unchanged=dict(recipe)
    configure_recalibration(recipe,.1,20000000,inputs=456,step=8,source='resumed.pt')
    assert recipe==unchanged
    dataset=PretrainDocumentDataset(font,[Story('A cat sat.','a'),Story('A dog ran.','b')],include_scenes=False)
    dataset.tasks=TASKS  # 正式池的八任务清单，仅T1的份额非零。
    shares={k:float(k=='T1') for k in TASKS};recipe.update(input_shares=shares)
    source=InputDocumentStream(dataset,{},shares,workers=0)
    collect_documents(source,20);old_positions=source.state_dict()
    continued=InputDocumentStream(dataset,{},shares,workers=0);continued.continue_positions(old_positions)
    assert continued.indices==source.indices and continued.offsets==source.offsets
    assert sum(continued.inputs.values())==0
    scheduler=InputWSD(optimizer,recipe,1e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    save_formal(args.out_dir/'fixture_full.pt',model,optimizer,scheduler,scaler,continued,
        step=26204,phase_step=1,counters=Counter(successful_inputs=20),data_args={},manifest={},
        recipe=recipe,ae_source={},parent_source={})
    saved=torch.load(args.out_dir/'fixture_full.pt',map_location='cpu',weights_only=False)
    assert saved['train_args']==recipe and saved['cfg']['r_text_aux_weight']==recipe['text_aux_alpha']
    assert not auxiliary_calibration_due(saved['train_args'],20000122)
    assert auxiliary_calibration_due(saved['train_args'],20000123)
    restored=FrozenSketchLM(config_from_r_checkpoint(saved));restored.load_state_dict(saved['model'],strict=True)
    bind_auxiliary(restored,font,'fixture',saved_manifest=saved['text_aux_manifest'])
    restored_optimizer=torch.optim.Adam(restored.r_pred.parameters(),lr=1e-5)
    restored_optimizer.add_param_group({'params':restored.text_aux.parameters(),'role':'text_aux'})
    restored_optimizer.load_state_dict(saved['optimizer'])
    assert weight_hash(restored)==weight_hash(model) and len(restored_optimizer.param_groups)==2
    assert len(restored_optimizer.state)==len(optimizer.state)
    # 旧cfg/权重继续严格加载，辅助默认关闭，参数集合不变。
    old_cfg=asdict(cfg);old_cfg['r_text_aux_enabled']=False
    old=FrozenSketchLM(config_from_r_checkpoint({'cfg':old_cfg}));old.load_state_dict(original_state,strict=True)
    report=dict(auxiliary_gradient_reaches_R=True,frozen_AE_unchanged=True,
        auxiliary_only_does_not_train_GMM_head=True,labels_match_original_readout=True,
        space_supervised_every_target=True,split_microbatch_gradients_match=True,
        old_R_optimizer_preserved=True,continue_data_indices_offsets_preserved=True,
        gradient_calibration_no_updates=True,full_checkpoint_optimizer_roundtrip=True,
        recalibration_target_10_percent=True,recalibration_successful_input_boundary=True,
        recalibration_policy_progress_roundtrip=True,
        old_checkpoint_strict_compatible=True,online_validation_preserves_weights_RNG=True,calibration=calibration)
    (args.out_dir/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print({k:v for k,v in report.items() if k!='calibration'})


if __name__=='__main__':main()
