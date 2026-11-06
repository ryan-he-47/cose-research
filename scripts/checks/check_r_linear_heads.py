"""2026-10-14：13层/线性头从头训练结构、预算、随机绑定和恢复；实际FP16零更新。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from copy import deepcopy
from dataclasses import replace,asdict
import json
import torch
from src.st.pretrain_model import FrozenSketchLM,config_from_ae_checkpoint,config_from_r_checkpoint
from src.st.pretrain_config import apply_r_architecture,check_chinchilla_budget
from src.st.text_auxiliary import bind_auxiliary,preflight_auxiliary,weight_hash
from src.st.train_pretrain import _rng_state,_restore_rng
from src.text_renderer import VectorFont
from src.page_document import build_document,layout_text_pages
from src.st.text_auxiliary import annotate_documents
from src.st.pipeline_pretrain import collate_documents


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('tmp_smoke/r_linear_heads_check_20261014'))
    parser.add_argument('--ae', type=Path, default=Path('tmp_smoke/r_dual_review_assets_20261009/ae_final.pt'))
    parser.add_argument('--fp16', action='store_true', help='本机实际13层模型前后向，不执行optimizer.step')
    args=parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(42)
    recipe=json.loads(Path('configs/r_t1_linear_heads_13l_20261014.json').read_text())
    ae=torch.load(args.ae,map_location='cpu',weights_only=False)
    cfg=apply_r_architecture(config_from_ae_checkpoint(ae),recipe,fresh=True)
    cfg.r_sigma_floor_mode=recipe['r_sigma_floor_mode']; cfg.sigma_eps_r=recipe['sigma_eps_r']
    model=FrozenSketchLM(cfg,activation_checkpoint=False);model.load_ae(ae)
    assert len(model.r_pred.layers)==13 and model.r_pred.head_buf is None
    assert model.r_pred.head.head.in_features==768 and model.r_pred.head.head.out_features==3104
    assert isinstance(model.text_aux.trunk,torch.nn.Identity)
    assert model.text_aux.shape_head.in_features==model.text_aux.space_head.in_features==768
    assert all(not p.requires_grad for module in (model.encoder,model.decoder,model.dct_head) for p in module.parameters())
    ae_equal=all(torch.equal(model.state_dict()[k],v) for k,v in ae['model'].items())
    assert ae_equal
    budget=check_chinchilla_budget(model,recipe)
    counts=dict(R=sum(p.numel() for p in model.r_pred.parameters()),auxiliary=sum(p.numel() for p in model.text_aux.parameters()),
        all_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert counts==dict(R=94591520,auxiliary=89973,all_trainable=94681493)
    original_cfg=deepcopy(cfg)
    try: apply_r_architecture(replace(cfg,r_layers=12),recipe,fresh=False)
    except ValueError: mismatch_rejected=True
    else: raise AssertionError('恢复时静默改变层数')
    assert cfg==original_cfg
    bad={**recipe,'total_inputs':recipe['total_inputs']+1}
    try: check_chinchilla_budget(model,bad)
    except ValueError: budget_mismatch_rejected=True
    else: raise AssertionError('预算与实际参数未对齐')
    font=VectorFont.from_svg('datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg')
    catalog,manifest=bind_auxiliary(model,font,'fixture-font',initialize_random=True,initialization_seed=42)
    assert manifest['head_source'] is None and manifest['initialization']=='fresh_R_and_auxiliary'
    write(args.out/'architecture.json',dict(cfg=asdict(cfg),counts=counts,parameter_budget=budget,
        linear_heads=True,frozen_AE_equal=ae_equal,restore_architecture_mismatch_rejected=mismatch_rejected,
        budget_mismatch_rejected=budget_mismatch_rejected,random_head_not_from_old_readout=True))
    if args.fp16:
        if not torch.cuda.is_available():raise RuntimeError('需要本机CUDA')
        model.to('cuda');model.train();model.encode_chunk_size=128
        before=weight_hash(model)
        report=preflight_auxiliary(model,font,catalog,torch.device('cuda'),_rng_state,_restore_rng)
        doc=build_document(layout_text_pages('A cat sat. It was happy!',font),complete=True,metadata={'task':'T1'})
        n=annotate_documents([doc],catalog)
        cpu=collate_documents([doc]);cpu['task_id']=torch.zeros(1,dtype=torch.long)
        batch={k:v.to('cuda') if torch.is_tensor(v) else v for k,v in cpu.items()}
        with torch.autocast('cuda',dtype=torch.float16):
            result=model(batch)
            loss=.35*result['content_sum']/result['content_count']+.05*result['control_sum']/result['control_count']
            loss+=.1*(result['aux_all_ce_sum']+result['aux_all_bce_sum'])/n
        (loss*128).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.r_pred.head.parameters())
        assert all(p.grad is None for p in model.encoder.parameters())
        model.zero_grad(set_to_none=True)
        report['combined_geometry_and_auxiliary_backward_finite']=True
        assert before==weight_hash(model)
        write(args.out/'actual_fp16.json',{**report,'weights_unchanged':True,'layers':13,'linear_heads':True})
        print(json.dumps({'actual_fp16':report}),flush=True)
    del model,ae
    # 小尺寸执行一次工程更新并恢复两组Adam，避免额外实际R训练。
    tiny_cfg=replace(cfg,enc_d_model=24,enc_layers=1,enc_heads=4,enc_dff=48,dec_layers=1,dec_hidden=32,
        r_d_model=32,r_layers=2,r_heads=4,r_dff=64,gmm_r=3)
    tiny=FrozenSketchLM(tiny_cfg,activation_checkpoint=False)
    bind_auxiliary(tiny,font,'fixture-font',initialize_random=True,initialization_seed=42)
    optimizer=torch.optim.Adam(tiny.r_pred.parameters(),lr=1e-5)
    optimizer.add_param_group({'params':tiny.text_aux.parameters(),'role':'text_aux'})
    doc=build_document(layout_text_pages('A cat sat.',font),complete=True,metadata={'task':'T1'})
    n=annotate_documents([doc],catalog);result=tiny(collate_documents([doc]))
    loss=result['loss']+(result['aux_all_ce_sum']+result['aux_all_bce_sum'])/n
    loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    saved=dict(cfg=asdict(tiny.cfg),model=tiny.state_dict(),optimizer=optimizer.state_dict(),text_aux_manifest=tiny.text_aux_manifest)
    torch.save(saved,args.out/'fixture.pt')
    saved=torch.load(args.out/'fixture.pt',map_location='cpu',weights_only=False)
    restored=FrozenSketchLM(config_from_r_checkpoint(saved));restored.load_state_dict(saved['model'],strict=True)
    bind_auxiliary(restored,font,'fixture-font',saved_manifest=saved['text_aux_manifest'])
    opt=torch.optim.Adam(restored.r_pred.parameters(),lr=1e-5)
    opt.add_param_group({'params':restored.text_aux.parameters(),'role':'text_aux'});opt.load_state_dict(saved['optimizer'])
    assert weight_hash(tiny)==weight_hash(restored) and len(opt.param_groups)==2
    assert all(int(s['step'])==1 for s in opt.state.values())
    assert restored.r_pred.head_buf is None and isinstance(restored.text_aux.trunk,torch.nn.Identity)
    # 不配置新字段时，旧非线性头的严格权重路径保持。
    old_cfg=replace(tiny_cfg,r_head_mlp=True,r_text_aux_enabled=False)
    old=FrozenSketchLM(old_cfg);old_weights=old.state_dict()
    old_reload=FrozenSketchLM(config_from_r_checkpoint({'cfg':asdict(old_cfg)}));old_reload.load_state_dict(old_weights,strict=True)
    write(args.out/'checks.json',dict(linear_head_checkpoint_strict_roundtrip=True,two_Adam_groups_restored=True,
        old_buffer_checkpoint_strict_compatible=True,actual_R_optimizer_updates=0,tiny_fixture_updates=1))
    print(json.dumps(dict(counts=counts,parameter_budget=budget,checks=True)),flush=True)


if __name__=='__main__':main()
