"""2026-10-27：配对真值、无答案泄漏、微批梯度及新增头恢复工程检查。"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.st.pipeline_pretrain import build_document_dataset,collate_documents as _collate
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.pretrain_model import FrozenSketchLM,config_from_r_checkpoint
from src.st.text_auxiliary import bind_auxiliary,annotate_documents,space_target_count
from src.st.visual_grounding_auxiliary import annotate_learning,install_grounding,calibrate_learning
from src.st.train_formal import task_loss,materialize_adam,_rng_state,_restore_rng
from src.st.formal_stream import collect_documents,InputDocumentStream
from src.st.formal_recipe import InputWSD
from src.st.formal_prefetch import PreparedUpdates
from src.visual_grounding_scenes import paired_scene
from src.formal_scene_tasks import make_recipe


def collate_documents(documents):
    value=_collate(documents)
    tasks=('T1','O1','O2','S1','S2','S3','S4','S5')
    value['task_id']=torch.tensor([tasks.index(d.metadata['task']) for d in documents])
    return value


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    args=json.loads((ROOT/'tmp_smoke/formal_check_20261003/data_args.json').read_text())
    args['paired_scene_share']=1.
    dataset=build_document_dataset(**args)
    recipe=json.loads((ROOT/'configs/r_visual_grounding_20261027.proposed.json').read_text())
    recipe.update(text_aux_alpha=.1,grounding_alpha=.2,padded_stroke_budget=2048,
                  microbatch_documents=2,workers=0,text_aux_probe_batches=2)
    cfg=replace(gpt2_small_pretrain_config(),latent_units=4,enc_d_model=16,enc_layers=1,
        enc_heads=2,enc_dff=32,dec_hidden=16,dec_layers=1,gmm_dec=2,r_d_model=16,
        r_layers=1,r_heads=2,r_dff=32,gmm_r=2,r_head_mlp=False,r_text_aux_hidden=0,
        sigma_eps_r=.003,r_sigma_floor_mode='quadrature')
    model=FrozenSketchLM(cfg,activation_checkpoint=False)
    catalog,_=bind_auxiliary(model,dataset.font,dataset.manifest()['font_sha256'],initialize_random=True,
                             include_non_symbol=True,initialization_seed=42)
    optimizer=torch.optim.Adam(model.r_pred.parameters(),lr=1e-4)
    optimizer.add_param_group(dict(params=model.text_aux.parameters(),role='text_aux'))
    materialize_adam(optimizer)
    # 迁移检查使用已有非零Adam历史，避免只核对全零槽位。
    for group in optimizer.param_groups:
        for parameter in group['params']:parameter.grad=torch.full_like(parameter,.01)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    old_state=deepcopy(optimizer.state_dict());old_weights=deepcopy(model.state_dict())
    rng=torch.get_rng_state().clone()
    install_grounding(model,optimizer,recipe['visual_grounding'],dataset.drawings.classes)
    assert torch.equal(rng,torch.get_rng_state())
    assert len(optimizer.param_groups)==3
    for key,value in old_weights.items():assert torch.equal(value,model.state_dict()[key])
    for key,value in old_state['state'].items():
        for name,tensor in value.items():assert torch.equal(tensor,optimizer.state_dict()['state'][key][name])
    source_recipe=next(make_recipe(dataset.drawings,42,i) for i in range(30)
                       if len({o['label'] for o in make_recipe(dataset.drawings,42,i)['objects']})>1)
    documents=[]
    for task,pair_id in (('S3',0),('S3',1),('S4',6),('S5',9)):
        pair=[paired_scene(task,source_recipe,dataset.drawings,dataset.font,pair_id=pair_id,variant=v) for v in (0,1)]
        assert pair[0].metadata['question']==pair[1].metadata['question']
        assert pair[0].metadata['answer']!=pair[1].metadata['answer']
        for doc in pair:
            assert len(doc.metadata['stroke_owner'])==sum(e.kind=='sketch' for e in doc.to_sequence())
            assert len(doc.metadata['relations'])==len(doc.metadata['objects'])*(len(doc.metadata['objects'])-1)
        documents+=pair
    count=annotate_documents(documents,catalog)
    annotate_learning(documents,recipe,dataset.drawings.classes)
    body={task:sum(e.targetable and e.kind in ('text','sketch','selection') for d in documents if d.metadata['task']==task for e in d.to_sequence()[1:]) for task in recipe['loss_weights']}
    control=sum(e.targetable and e.kind in ('bos','eop','eod') for d in documents for e in d.to_sequence()[1:])
    for doc in documents:
        data=doc.metadata['_grounding_targets'];seq=doc.to_sequence()
        if data['query_position'] is not None:
            assert all(p<=data['query_position'] for p in data['key_positions'])
            assert not any(e.targetable for e in seq[:data['query_position']+1])
        assert all(not seq[p].kind=='text' for p in data['object_positions'])
    model.eval()
    loss=lambda result:task_loss(result,body,control,recipe,auxiliary_count=count,auxiliary_space_count=space_target_count(documents))
    optimizer.zero_grad(set_to_none=True)
    full=loss(model(collate_documents(documents)));full.backward()
    gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    assert any(n.startswith('visual_grounding.') and g.abs().sum()>0 for n,g in gradients.items())
    assert any(n.startswith('r_pred.layers.') and g.abs().sum()>0 for n,g in gradients.items())
    assert all(p.grad is None for n,p in model.named_parameters() if n.startswith(('encoder.','decoder.','dct_head.')))
    optimizer.zero_grad(set_to_none=True);total=0.
    for doc in documents:
        value=loss(model(collate_documents([doc])));total+=float(value.detach());value.backward()
    torch.testing.assert_close(torch.tensor(total),full.detach(),atol=1e-4,rtol=1e-5)
    for name,p in model.named_parameters():
        if name in gradients:torch.testing.assert_close(p.grad,gradients[name],atol=2e-4,rtol=3e-4)
    optimizer.step()
    state=dict(model=deepcopy(model.state_dict()),cfg=asdict(model.cfg),optimizer=deepcopy(optimizer.state_dict()))
    restored=FrozenSketchLM(config_from_r_checkpoint(state),activation_checkpoint=False)
    restored.load_state_dict(state['model'],strict=True);restored.eval()
    second=torch.optim.Adam(restored.r_pred.parameters(),lr=1e-4)
    second.add_param_group(dict(params=restored.text_aux.parameters(),role='text_aux'))
    second.add_param_group(dict(params=restored.visual_grounding.parameters(),role='visual_grounding'))
    second.load_state_dict(state['optimizer'])
    for candidate,opt in ((model,optimizer),(restored,second)):
        opt.zero_grad(set_to_none=True);loss(candidate(collate_documents(documents))).backward();opt.step()
    for name,p in model.state_dict().items():torch.testing.assert_close(p,restored.state_dict()[name],atol=0,rtol=0)
    for index,values in optimizer.state_dict()['state'].items():
        for name,value in values.items():torch.testing.assert_close(value,second.state_dict()['state'][index][name],atol=0,rtol=0)
    calibration=calibrate_learning(model,documents,body,control,recipe,torch.device('cpu'),task_loss,
                                  lambda x,d:x,_rng_state,_restore_rng)
    assert calibration['alpha']>0 and calibration['grounding_alpha']>0
    # 异步准备器和直接供数使用相同索引、配对变体及全批标签。
    data=InputDocumentStream(dataset,args,recipe['input_shares'],workers=0)
    prepared=PreparedUpdates(data,recipe,catalog,pin=False)
    boundary=prepared.state_dict();values=prepared.take(1024)
    signature=lambda docs:[(d.metadata['task'],d.metadata.get('pair_id'),d.metadata.get('variant_id'),d.metadata['task_index']) for d in docs]
    first=signature(values[0]);prepared.rewind(boundary);again=prepared.take(1024)
    assert first==signature(again[0])
    prepared.close()
    report=dict(paired_documents=len(documents),loss=float(full.detach()),microbatch_gradient_equal=True,
        frozen_AE_no_gradient=True,old_weights_Adam_and_RNG_preserved=True,three_group_resume_exact=True,
        common_full_hidden_calibration=calibration,paired_prepare_rewind_equal=True)
    out=ROOT/'tmp_smoke/r_visual_grounding_20261027';out.mkdir(parents=True,exist_ok=True)
    (out/'engineering.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
