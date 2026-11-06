"""2026-10-09：正式R收尾只读核对、valid逐笔前向与真实自回归样张。"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.demo.demo_pretrain_generation import save_native_svg, source_prefix
from src.st.generation_pretrain import PageGenerator
from src.st.model import gmm_nll
from src.st.pipeline_pretrain import build_document_dataset, collate_documents
from src.st.pretrain_model import FrozenSketchLM, config_from_r_checkpoint


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def audit(state, ae_path):
    ae = torch.load(ae_path, map_location='cpu', weights_only=False)
    keys = [k for k in state['model'] if k.startswith(('encoder.', 'decoder.', 'dct_head.'))]
    ae_keys = [k for k in ae['model'] if k.startswith(('encoder.', 'decoder.', 'dct_head.'))]
    equal = set(keys) == set(ae_keys) and all(torch.equal(state['model'][k], ae['model'][k]) for k in keys)
    finite = all(bool(torch.isfinite(v).all()) for v in state['model'].values())
    steps = [int(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
    digest = hashlib.sha256()
    with Path(ae_path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    counters = dict(state['counters'])
    result = dict(step=state['step'], successful_inputs=state['scheduler']['inputs'],
                  counters=counters, stream=state['data'], recipe=state['train_args'], cfg=state['cfg'],
                  ae_source=state['ae_source'], ae_step=ae['step'], frozen_ae_equal=equal,
                  frozen_ae_tensors=len(keys), weights_finite=finite,
                  actual_ae_sha256=digest.hexdigest(), ae_hash_equal=digest.hexdigest()==state['ae_source']['sha256'],
                  optimizer_step_min=min(steps), optimizer_step_max=max(steps), optimizer_entries=len(steps),
                  final_lr=state['optimizer']['param_groups'][0]['lr'], scaler=state['scaler'],
                  source_commit=state['source_commit'], budget_complete=state['scheduler']['inputs']>=state['train_args']['total_inputs'])
    if not (equal and finite and result['ae_hash_equal'] and result['budget_complete']
            and min(steps)==max(steps)==state['step']
            and counters['successful_inputs']==state['scheduler']['inputs']):
        raise RuntimeError('完整状态或冻结来源异常：'+json.dumps(result,ensure_ascii=False))
    return result


def stroke_records(document):
    return [dict(page_id=e.page_id, kind=e.kind, generated=False, targetable=e.targetable,
                 points=e.rawxy.tolist()) for e in document.to_sequence()]


@torch.no_grad()
def teacher_forced(generator, document, output, draw):
    model = generator.model
    cpu = collate_documents([document], max_strokes=model.cfg.max_context_strokes, max_points=generator.max_points)
    batch = {k:v.to(generator.device) if torch.is_tensor(v) else v for k,v in cpu.items()}
    with generator.autocast():
        emb = model.encode_valid(batch)
        hidden = model.hidden_states(emb[:,:-1],batch['slice_id'][:,:-1],batch['stroke_pad'][:,:-1],right_padded=True)
        selected = model.target_mask(batch)
        mu,sigma,pi = model.prediction_head(hidden[selected])
        target = emb[:,1:][selected].float()
        individual = gmm_nll(target,mu,sigma,pi,reduction='none')
        kinds = batch['kind_id'][:,1:][selected]
        choice = pi.argmax(-1)
        ids = torch.arange(len(choice),device=generator.device)
        greedy = mu[ids,choice]
        greedy_sigma = sigma[ids,choice]
        metrics = {}
        for name,mask in (('body',kinds<3),('control',kinds>=3)):
            count = int(mask.sum())
            metrics[name] = dict(count=count, nll=float(individual[mask].mean()) if count else None,
                                 sigma_mean=float(greedy_sigma[mask].mean()) if count else None,
                                 greedy_latent_mse=float(((greedy-target)[mask]**2).mean()) if count else None)
        if draw:
            predicted = generator.decode(greedy)
            reconstructed = generator.decode(target)
            elements = [e for e in document.to_sequence()[1:] if e.targetable]
            for name,points in (('teacher_forced',predicted),('ae_reference',reconstructed)):
                records = [dict(page_id=e.page_id,kind=e.kind,generated=True,points=p.tolist()) for e,p in zip(elements,points)]
                value = dict(strokes=records,teacher_forced=True,kind_labels='expected_target_kind',
                             conditioning='true_previous_strokes' if name=='teacher_forced' else 'true_target_embedding')
                write_json(output/f'{name}.json',value)
                save_native_svg(value,output/f'{name}.svg',name)
    write_json(output/'teacher_metrics.json',metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--ae',type=Path,required=True)
    parser.add_argument('--out_dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    checked = audit(state,args.ae)
    write_json(args.out_dir/'checks.json',checked)
    print(json.dumps({'audit':{k:checked[k] for k in ('step','successful_inputs','frozen_ae_equal','weights_finite','final_lr')} }),flush=True)
    data_args = {**state['data_args'],'split':'valid'}
    model = FrozenSketchLM(config_from_r_checkpoint(state),encode_chunk_size=1024,activation_checkpoint=False).eval()
    model.load_state_dict(state['model'],strict=True)
    max_points = state['train_args']['max_points']
    del state
    gc.collect()
    model.to(args.device)
    generator = PageGenerator(model,max_points=max_points)
    dataset = build_document_dataset(**data_args)
    labels = ('cat','fish','airplane','bicycle')
    cases = [(task,label,dataset.drawings.classes.index(label)) for task in ('O1','O2') for label in labels]
    cases += [('T1',f'story{i}',i) for i in range(2)]
    cases += [(f'S{task}',f'scene{i}',i) for i in range(2) for task in range(1,6)]
    summary = []
    for task,label,index in cases:
        document = dataset.get_for_task(task,index)
        if task=='T1':
            document = document.window(0,model.cfg.max_context_strokes)
        case = args.out_dir/f'{task}_{label}'
        case.mkdir()
        reference = dict(strokes=stroke_records(document),source=document.metadata,
                         page_texts={p.page_id:p.text for p in document.pages if p.text})
        write_json(case/'reference.json',reference)
        save_native_svg(reference,case/'reference.svg',f'{task}: {label} / valid source')
        metrics = teacher_forced(generator,document,case,task in ('O1','S2','S4'))
        max_strokes = 480 if task=='S1' else 320 if task=='T1' else 192 if task in ('O2','S3','S5') else 128
        records = dict(task=task,label=label,index=index,source=document.metadata,
                       reference_strokes=document.stroke_count,teacher_metrics=metrics,generation={})
        for mode in ('greedy','sample'):
            result = generator.generate(source_prefix(document,task),mode=mode,temperature=.5,seed=42,
                                        max_new_strokes=max_strokes,max_new_pages=8)
            result.update(task=task,label=label,source=document.metadata,checkpoint_step=checked['step'])
            write_json(case/f'{mode}.json',result)
            save_native_svg(result,case/f'{mode}.svg',f'{task}: {label} / {mode}')
            value = {k:v for k,v in result.items() if k not in ('strokes','source')}
            records['generation'][mode] = value
            print(json.dumps({'case':case.name,**value},ensure_ascii=False),flush=True)
        summary.append(records)
        write_json(args.out_dir/'summary.json',summary)
    stop_counts = {mode:dict(Counter(r['generation'][mode]['stop_reason'] for r in summary)) for mode in ('greedy','sample')}
    write_json(args.out_dir/'completion.json',dict(cases=len(summary),generations=len(summary)*2,split='valid',
        mode_temperatures={'greedy':None,'sample':.5},seed=42,stop_counts=stop_counts,
        checkpoint=str(args.checkpoint),checkpoint_step=checked['step'],training=False))
    print(json.dumps({'review_complete':True,'cases':len(summary),'stop_counts':stop_counts}),flush=True)


if __name__=='__main__':
    main()
