"""2026-10-22：本机CPU核对真实55k AE起新R，不执行训练。"""
from pathlib import Path
from hashlib import sha256
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from src.text_renderer import VectorFont
from src.st.pretrain_model import FrozenSketchLM,config_from_ae_checkpoint
from src.st.pretrain_config import apply_r_architecture
from src.st.text_auxiliary import bind_auxiliary,weight_hash
from src.st.formal_recipe import InputWSD
from src.st.train_formal import ae_source_record


def main():
    torch.set_num_threads(2)
    recipe=json.loads((ROOT/'configs/r_joint_scratch_20261022.json').read_text())
    assert recipe['initialization_policy']=='random_R_and_heads'
    assert recipe['total_inputs']==3600000000 and recipe['retry_failed_documents']
    source=ROOT/'tmp_smoke/r_dual_review_assets_20261009/ae_final.pt'
    state=torch.load(source,map_location='cpu',weights_only=False)
    ae=ae_source_record(source,state,expected_step=55000,expected_lr=5e-6)
    assert ae['sha256']=='fdde9575d7570e098517b2b232174b074b938c29fa8039bf04591eff2098f781'
    try:
        ae_source_record(source,{'phase':'formal_v1'})
    except ValueError:
        pass
    else:
        raise AssertionError('不能将R检查点用作独立AE来源')
    torch.manual_seed(recipe['seed'])
    cfg=apply_r_architecture(config_from_ae_checkpoint(state),recipe,fresh=True)
    cfg.sigma_eps_r=recipe['sigma_eps_r'];cfg.r_sigma_floor_mode=recipe['r_sigma_floor_mode']
    model=FrozenSketchLM(cfg,activation_checkpoint=False)
    r_hash=weight_hash(model.r_pred);aux_hash=weight_hash(model.text_aux)
    # 即使源字典意外带R键，AE加载也不能碰R/辅助头。
    key=next(k for k in model.state_dict() if k.startswith('r_pred.'))
    state['model'][key]=torch.full_like(model.state_dict()[key],123.)
    model.load_ae(state)
    assert r_hash==weight_hash(model.r_pred) and aux_hash==weight_hash(model.text_aux)
    actual=model.state_dict()
    ae_keys=[k for k in actual if k.startswith(('encoder.','decoder.','dct_head.'))]
    assert all(torch.equal(actual[k],state['model'][k]) for k in ae_keys)
    assert all(not p.requires_grad for m in (model.encoder,model.decoder,model.dct_head) if m is not None for p in m.parameters())
    font_path=ROOT/'datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg'
    font=VectorFont.from_svg(str(font_path));font_hash=sha256(font_path.read_bytes()).hexdigest()
    catalog,manifest=bind_auxiliary(model,font,font_hash,initialize_random=True,
        initialization_seed=42,include_non_symbol=True)
    assert len(catalog)==117 and catalog.non_symbol_class==116
    assert r_hash==weight_hash(model.r_pred) and aux_hash==weight_hash(model.text_aux)
    # 词表完整恢复不重新扩类，不初始化旧116类权重。
    restored,again=bind_auxiliary(model,font,font_hash,saved_manifest=manifest)
    assert restored.dump()==catalog.dump() and again==manifest
    assert model.r_pred.head_buf is None and isinstance(model.text_aux.trunk,torch.nn.Identity)
    assert (cfg.r_layers,cfg.r_d_model,cfg.r_heads,cfg.r_dff,cfg.latent_units,cfg.gmm_r)==(13,768,12,3072,48,32)
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert parameters==94682262,parameters
    optimizer=torch.optim.Adam(model.r_pred.parameters(),lr=recipe['peak_lr'])
    optimizer.add_param_group({'params':model.text_aux.parameters(),'role':'text_aux'})
    scheduler=InputWSD(optimizer,recipe,0.)
    assert not optimizer.state and scheduler.inputs==0 and all(g['lr']==0 for g in optimizer.param_groups)
    assert scheduler.lr_at(recipe['warmup_inputs'])==1e-4
    assert abs(scheduler.lr_at(recipe['total_inputs'])-1e-5)<1e-12
    report=dict(actual_trainable_parameters=parameters,ae_step=55000,ae_sha256=ae['sha256'],
        r_sha256=r_hash,auxiliary_sha256=aux_hash,random_R_untouched_by_AE_load=True,
        all_AE_tensors_equal=True,AE_frozen=True,R_checkpoint_rejected_as_AE=True,
        classes=len(catalog),non_symbol_class=catalog.non_symbol_class,
        fresh_117_class_binding_and_restore=True,optimizer_states=0,scheduler_inputs=0,
        total_inputs=recipe['total_inputs'],checks='CPU，无训练')
    out=ROOT/'tmp_smoke/r_joint_scratch_check_20261022';out.mkdir(parents=True,exist_ok=True)
    (out/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
