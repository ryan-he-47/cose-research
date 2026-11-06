"""CPU核对非符号扩展/Adam移植、图形空格屏蔽、全池遍历与覆盖WSD。"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
import sys
import json
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from src.text_renderer import VectorFont
from src.page_document import build_document,layout_text_pages,sketch_page
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.pretrain_model import FrozenSketchLM,config_from_r_checkpoint
from src.st.text_readout import StrokeVocabulary
from src.st.text_shape_readout import ShapeVocabulary
from src.st.text_auxiliary import (bind_auxiliary,extend_non_symbol,annotate_documents,
    auxiliary_statistics,space_target_count,auxiliary_loss)
from src.st.pipeline_pretrain import collate_documents,build_document_dataset
from src.st.formal_data import FormalDocumentDataset
from src.st.formal_recipe import CoverageWSD
from src.st.formal_stream import InputDocumentStream
from src.st.train_formal import save_formal


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    font_path='datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg'
    font=VectorFont.from_svg(font_path);catalog=ShapeVocabulary(StrokeVocabulary(font))
    cfg=replace(gpt2_small_pretrain_config(),latent_units=4,enc_d_model=16,enc_layers=1,enc_heads=2,enc_dff=32,
        dec_hidden=16,dec_layers=1,gmm_dec=2,r_d_model=16,r_layers=1,r_heads=2,r_dff=32,
        gmm_r=2,r_head_mlp=False,r_text_aux_hidden=0,max_context_strokes=1024)
    model=FrozenSketchLM(cfg);model.install_text_aux(len(catalog),0)
    catalog,manifest=bind_auxiliary(model,font,'fixture',initialize_random=True)
    optimizer=torch.optim.Adam(model.r_pred.parameters(),lr=1e-4)
    optimizer.add_param_group({'params':model.text_aux.parameters(),'role':'text_aux'})
    text=build_document(layout_text_pages('A cat.',font),complete=True,metadata={'task':'T1'})
    annotate_documents([text],catalog)
    result=model(collate_documents([text]));n=int(result['aux_all_count'])
    loss=result['pred_nll']+auxiliary_loss(result,n);loss.backward();optimizer.step();optimizer.zero_grad()
    old_weight=model.text_aux.shape_head.weight.detach().clone()
    old_moment=optimizer.state[model.text_aux.shape_head.weight]['exp_avg'].clone()
    old_space=model.text_aux.space_head.weight
    r_state=deepcopy(optimizer.state_dict()['state'])
    catalog,manifest=extend_non_symbol(model,optimizer,font,manifest)
    assert len(catalog)==117 and catalog.non_symbol_class==116
    assert torch.equal(old_weight,model.text_aux.shape_head.weight[:-1])
    state=optimizer.state[model.text_aux.shape_head.weight]
    assert torch.equal(old_moment,state['exp_avg'][:-1]) and not state['exp_avg'][-1].any()
    assert model.text_aux.space_head.weight is old_space and int(state['step'])==1
    for i,p in enumerate(optimizer.param_groups[0]['params']):
        assert all(torch.equal(v,r_state[i][k]) for k,v in optimizer.state[p].items())
    points=np.array([[.1,.1],[.3,.8],[.8,.2]],np.float32)
    graphic=build_document((sketch_page([points]),),complete=True,metadata={'task':'O1'})
    count=annotate_documents([text,graphic],catalog);spaces=space_target_count([text,graphic])
    labels=graphic.metadata['_text_aux_targets'];seq=graphic.to_sequence()
    assert all(labels['shape'][i]==116 and labels['valid'][i] for i,e in enumerate(seq) if e.kind=='sketch')
    result=model(collate_documents([text,graphic]))
    assert result['aux_sketch_count']>0 and result['aux_sketch_space_count']==0
    assert result['aux_sketch_bce_sum']==0 and count>spaces
    logits=dict(shape=torch.randn(3,117,requires_grad=True),space=torch.randn(3,requires_grad=True))
    stats=auxiliary_statistics(logits,torch.tensor([0,116,116]),torch.zeros(3),
                               torch.tensor([True,False,False]),torch.tensor([0,1,2]))
    stats['aux_all_bce_sum'].backward()
    assert logits['space'].grad[1:].eq(0).all() and logits['space'].grad[0]!=0
    params=[p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True);auxiliary_loss(result,count,spaces).backward()
    whole=[None if p.grad is None else p.grad.clone() for p in params]
    model.zero_grad(set_to_none=True)
    for doc in (text,graphic):
        part=model(collate_documents([doc]));auxiliary_loss(part,count,spaces).backward()
    assert all((a is None and p.grad is None) or torch.allclose(a,p.grad,atol=1e-6,rtol=1e-4)
               for a,p in zip(whole,params))
    saved=dict(cfg=asdict(cfg),model=model.state_dict())
    restored=FrozenSketchLM(config_from_r_checkpoint(saved));restored.load_state_dict(saved['model'],strict=True)
    bind_auxiliary(restored,font,'fixture',saved_manifest=manifest)
    opt=torch.optim.Adam(restored.r_pred.parameters(),lr=1e-4)
    opt.add_param_group({'params':restored.text_aux.parameters(),'role':'text_aux'})
    opt.load_state_dict(optimizer.state_dict())
    assert opt.state[restored.text_aux.shape_head.weight]['exp_avg'].shape[0]==117
    class Drawings:
        classes=('cat','dog');class_indices=(range(3),range(3,5))
        def __len__(self):return 5
        def get(self,i):return SimpleNamespace(label=self.classes[i>=3],key_id=str(i),strokes=[points])
    dataset=FormalDocumentDataset(font,(),Drawings(),font_path=font_path,object_sampling='affine_pass_v1')
    with patch('src.st.formal_data.build_formal_dataset',return_value=dataset) as factory:
        assert build_document_dataset(formal=True,split='train',object_sampling='affine_pass_v1') is dataset
        assert factory.call_args.kwargs['object_sampling']=='affine_pass_v1'
    for task in ('O1','O2'):
        ids=[dataset.get_for_task(task,i).metadata['source_id'] for i in range(5)]
        assert set(ids)==set(map(str,range(5)))
    recipe=dict(peak_lr=1e-4,min_lr=1e-5,warmup_inputs=100,
                training_goal=dict(kind='all_task_pools',passes=1,decay_fraction=.04,limits={'T1':10}))
    scheduler=CoverageWSD(optimizer,recipe,1e-5)
    scheduler.step(100,.96);assert abs(optimizer.param_groups[0]['lr']-1e-4)<1e-12
    scheduler.step(1,.98);assert abs(optimizer.param_groups[0]['lr']-5.5e-5)<1e-12
    clone=CoverageWSD(optimizer,recipe,1e-5);clone.load_state_dict(scheduler.state_dict())
    assert clone.inputs==101 and clone.progress==.98
    clone.step(1,1.);assert abs(optimizer.param_groups[0]['lr']-1e-5)<1e-12
    out=Path('tmp_smoke/r_joint_aux_check_20261019');out.mkdir(parents=True,exist_ok=True)
    shares=json.loads(Path('configs/r_joint_full_20261019.json').read_text())['input_shares']
    stream=InputDocumentStream(dataset,{},shares,workers=0)
    stream.indices['T1']=10
    assert stream.coverage({'T1':10})['progress']==1
    recipe['keep_recent_checkpoints']=2
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    for step in (1,2,3):
        save_formal(out/f'ckpt_{step:07d}.pt',model,optimizer,clone,scaler,stream,
            step=step,phase_step=step,counters={'successful_inputs':clone.inputs},data_args={},
            manifest={},recipe=recipe,ae_source={},parent_source={})
    assert len(list(out.glob('ckpt_*.pt')))==2 and not (out/'ckpt_0000001.pt').exists()
    full=torch.load(out/'ckpt_0000003.pt',map_location='cpu',weights_only=False)
    stream2=InputDocumentStream(dataset,{},shares,workers=0);stream2.load_state_dict(full['data'])
    assert stream2.coverage({'T1':10})==stream.coverage({'T1':10})
    clone2=CoverageWSD(optimizer,recipe,1e-5);clone2.load_state_dict(full['scheduler'])
    assert clone2.progress==1 and full['cfg']['r_text_aux_classes']==117
    report=dict(old_rows_and_Adam_preserved=True,non_symbol_class=116,total_classes=117,
        sketch_and_region_space_gradient_zero=True,symbol_space_normalization=True,
        mixed_microbatch_gradients_match=True,extended_checkpoint_roundtrip=True,
        objects_without_replacement=True,coverage_WSD_roundtrip=True,
        full_checkpoint_coverage_stream_roundtrip=True,bounded_checkpoint_retention=True,
        dataset_entry_forwards_object_sampling=True)
    (out/'checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(report)


if __name__=='__main__':main()
