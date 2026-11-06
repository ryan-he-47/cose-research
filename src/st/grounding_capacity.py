"""2026-10-27：恢复真实模型/Adam，零优化器更新的对象辅助容量与速度核对。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from .pretrain_model import FrozenSketchLM,config_from_r_checkpoint
from .pipeline_pretrain import build_document_dataset
from .formal_stream import InputDocumentStream,collect_documents
from .text_auxiliary import bind_auxiliary,annotate_documents,space_target_count,calibrate_auxiliary,apply_auxiliary_calibration
from .visual_grounding_auxiliary import install_grounding,annotate_learning
from .train_formal import phase_data_args,materialize_adam,calibrate_capacity,task_loss,gpu_batch,_rng_state,_restore_rng


def weight_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data_manifest',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if os.getenv("COSE_REQUIRE_SLURM") == "1" and not os.getenv('SLURM_JOB_ID'):
        p.error('服务器重计算须经Slurm')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(42)
    recipe=json.loads(args.recipe.read_text())
    data_args=phase_data_args(json.loads(args.data_manifest.read_text()),recipe)
    dataset=build_document_dataset(**data_args)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=config_from_r_checkpoint(state)
    model=FrozenSketchLM(cfg,encode_chunk_size=recipe['encode_chunk_size'],activation_checkpoint=recipe['activation_checkpoint']).cuda()
    model.load_state_dict(state['model'],strict=True)
    catalog,_=bind_auxiliary(model,dataset.font,dataset.manifest()['font_sha256'],saved_manifest=state['text_aux_manifest'])
    optimizer=torch.optim.Adam(model.r_pred.parameters(),lr=recipe['peak_lr'])
    optimizer.add_param_group(dict(params=model.text_aux.parameters(),role='text_aux'))
    optimizer.load_state_dict(state['optimizer'])
    parent_step=state['step'];del state
    install_grounding(model,optimizer,recipe['visual_grounding'],dataset.drawings.classes)
    materialize_adam(optimizer)
    before=weight_hash(model)
    steps_before=[float(v['step']) for v in optimizer.state.values()]
    model.train()
    stream=InputDocumentStream(dataset,data_args,recipe['input_shares'],workers=recipe['workers'],
        prefetch_documents=recipe.get('prefetch_documents'),prepare_batch_documents=recipe.get('prepare_batch_documents'))
    try:
        started=time.monotonic()
        documents,inputs,body,control=collect_documents(stream,recipe['input_budget'])
        count=annotate_documents(documents,catalog);spaces=space_target_count(documents)
        annotate_learning(documents,recipe,dataset.drawings.classes)
        prepare_seconds=time.monotonic()-started
        calibration=calibrate_auxiliary(model,documents,body,control,recipe,torch.device('cuda'),task_loss,gpu_batch,_rng_state,_restore_rng)
        apply_auxiliary_calibration(recipe,calibration,inputs=0,step=parent_step)
        cfg.r_text_aux_weight=recipe['text_aux_alpha']
        capacity=calibrate_capacity(model,documents,body,control,recipe,torch.device('cuda'),count,spaces)
        assert weight_hash(model)==before
        assert steps_before==[float(v['step']) for v in optimizer.state.values()]
        actual=sum(inputs.values());seconds=capacity['selected']['seconds']
        report=dict(parent_step=parent_step,optimizer_updates=0,weights_unchanged=True,Adam_steps_unchanged=True,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            grounding_parameters=sum(p.numel() for p in model.visual_grounding.parameters()),
            committed_input_consumption=0,probe_inputs=actual,documents=len(documents),prepare_seconds=prepare_seconds,
            input_counts=dict(inputs),calibration=calibration,capacity=capacity,
            estimated_forward_backward_hours=recipe['total_inputs']/actual*seconds/3600,
            timing_scope='forward/backward capacity trial; excludes Adam step/checkpoints/validation; preparation measured separately',
            optimizer_groups=[dict(role=g.get('role','R'),parameters=len(g['params'])) for g in optimizer.param_groups],
            source_commit=globals().get('__cose_commit__'),slurm_job_id=os.getenv('SLURM_JOB_ID'))
        (args.out/'capacity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        (args.out/'measured_recipe.json').write_text(json.dumps(recipe,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:stream.close()


if __name__=='__main__':main()
