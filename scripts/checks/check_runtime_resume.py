"""2026-10-24：小型FP16完整训练入口，核对预取跨检查点恢复及两组Adam。"""
from dataclasses import asdict,replace
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.st.formal_recipe import InputWSD
from src.st.formal_stream import InputDocumentStream
from src.st.pipeline_pretrain import build_document_dataset
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.pretrain_model import FrozenSketchLM
from src.st.text_auxiliary import bind_auxiliary
from src.st.train_formal import save_formal,phase_data_args,materialize_adam


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out_dir',type=Path,default=ROOT/'tmp_smoke/runtime_resume_20261024')
    p.add_argument('--grounding',action='store_true',help='核对新三组Adam、配对消费和文字分桶损失')
    args=p.parse_args();out=args.out_dir;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(42)
    data_path=ROOT/'tmp_smoke/formal_check_20261003/data_args.json'
    recipe=json.loads((ROOT/'configs/r_joint_scratch_20261022.json').read_text())
    if args.grounding:
        recipe=json.loads((ROOT/'configs/r_visual_grounding_20261027.proposed.json').read_text())
        recipe['budget_status']='engineering_check'
    data_args=phase_data_args(json.loads(data_path.read_text()),recipe)
    dataset=build_document_dataset(**data_args)
    cfg=replace(gpt2_small_pretrain_config(),latent_units=4,enc_d_model=16,enc_layers=1,
        enc_heads=2,enc_dff=32,dec_hidden=16,dec_layers=1,gmm_dec=2,r_d_model=16,
        r_layers=1,r_heads=2,r_dff=32,gmm_r=2,r_head_mlp=False,r_text_aux_hidden=0,
        sigma_eps_r=recipe['sigma_eps_r'],r_sigma_floor_mode=recipe['r_sigma_floor_mode'])
    model=FrozenSketchLM(cfg,activation_checkpoint=False).cuda()
    manifest=dataset.manifest()
    catalog,_=bind_auxiliary(model,dataset.font,manifest['font_sha256'],initialize_random=True,
                             include_non_symbol=True,initialization_seed=42)
    recipe.update(input_budget=512,microbatch_documents=4,padded_stroke_budget=2048,
        workers=2,text_aux_alpha=.1,text_aux_calibration_pending=False,
        text_aux_calibration_last_inputs=0,save_seconds=3600,memory_headroom_gib=.5)
    recipe['r_architecture']={name:getattr(cfg,name) for name in recipe['r_architecture']}
    recipe['prefetch_documents']={task:8 for task in recipe['input_shares']}
    recipe['prepare_batch_documents']={task:2 for task in recipe['input_shares']}
    cfg.r_text_aux_weight=.1
    optimizer=torch.optim.Adam(model.r_pred.parameters(),lr=1e-4)
    optimizer.add_param_group({'params':model.text_aux.parameters(),'role':'text_aux'})
    materialize_adam(optimizer)
    scheduler=InputWSD(optimizer,recipe,1e-5)
    scaler=torch.amp.GradScaler('cuda',init_scale=256)
    parent_recipe=dict(recipe)
    if args.grounding:
        for name in ('visual_grounding','grounding_alpha','paired_scene_share','text_aux_normalization','text_aux_task_weights'):
            parent_recipe.pop(name,None)
        recipe['text_aux_calibration_pending']=True
        (out/'new_phase_recipe.json').write_text(json.dumps(recipe),encoding='utf-8')
        data_args={**data_args,'paired_scene_share':0.}
        dataset=build_document_dataset(**data_args);manifest=dataset.manifest()
    stream=InputDocumentStream(dataset,data_args,recipe['input_shares'],workers=0)
    save_formal(out/'initial.pt',model,optimizer,scheduler,scaler,stream,step=0,phase_step=0,
        counters={},data_args=data_args,manifest=manifest,recipe=parent_recipe,ae_source={'kind':'恢复检查小模型'},parent_source=None)
    del model,optimizer,scaler;torch.cuda.empty_cache()
    runtime=ROOT/'configs/r_runtime_efficiency_20261024.json'
    for name,source,steps,tune in [('continuous',out/'initial.pt',2,True),
                                  ('split_a',out/'initial.pt',1,True),
                                  ('split_b',out/'split_a/final.pt',2,False)]:
        command=[sys.executable,'-X','utf8','-u','src/st/train_formal.py','--resume',str(source),
                 '--data_manifest',str(data_path),'--output_dir',str(out/name),
                 '--steps',str(steps),'--max_seconds','0','--log_every','1']
        if args.grounding and tune:
            command[command.index('--resume')]='--new_phase'
            command+=['--continue_data','--recipe',str(out/'new_phase_recipe.json')]
        elif tune:command+=['--runtime_tuning',str(runtime)]
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            print((out/(name+'.log')).read_text(encoding='utf-8',errors='replace')[-7000:])
            raise RuntimeError(f'{name}入口检查失败')
        print(json.dumps({'completed':name}),flush=True)
    continuous=torch.load(out/'continuous/final.pt',map_location='cpu',weights_only=False)
    resumed=torch.load(out/'split_b/final.pt',map_location='cpu',weights_only=False)
    for name,value in continuous['model'].items():
        torch.testing.assert_close(value,resumed['model'][name],atol=0,rtol=0)
    for index,values in continuous['optimizer']['state'].items():
        for name,value in values.items():
            torch.testing.assert_close(value,resumed['optimizer']['state'][index][name],atol=0,rtol=0)
    for name in ['cfg','data','scheduler','scaler']:
        assert continuous[name]==resumed[name],name
    for name in ['step','phase_step']:assert continuous[name]==resumed[name]==2
    for name,value in continuous['counters'].items():
        if name!='training_seconds':assert value==resumed['counters'][name],name
    def equal_tree(a,b):
        import numpy as np
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal_tree(v,b[k]) for k,v in a.items())
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
        return a==b
    assert equal_tree(continuous['rng'],resumed['rng'])
    report=dict(successful_updates=2,fp16=True,workers=2,model_exact=True,both_Adam_groups_exact=True,
        optimizer_groups=len(continuous['optimizer']['param_groups']),all_Adam_groups_exact=True,
        scaler_scheduler_rng_cursor_exact=True,successful_inputs=continuous['scheduler']['inputs'],
        recipe_performance_flags_restored={k:resumed['train_args'][k] for k in
            ['prepare_updates','encoder_length_buckets','cache_validation_embeddings']})
    (out/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
