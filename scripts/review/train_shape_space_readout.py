"""2026-10-14：复用冻结R特征，训练形状/逐笔空格双分支并查看同前文闭环。"""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from dataclasses import asdict
import hashlib
import json
import time
import numpy as np
import torch

from scripts.demo.demo_text_readout import load_readout
from scripts.demo.demo_pretrain_generation import save_native_svg
from scripts.review.diagnose_formal_r_health import model_hash
from scripts.review.train_text_readout import check_time, logits_for, write
from src.st.text_readout_generation import generate_text_readout as generate
from src.page_document import control_stroke
from src.st.text_readout import arc_points
from src.st.text_shape_readout import (ShapeVocabulary, ShapeSpaceReadout, shape_space_loss,
    shape_space_metrics, remap_joint_probabilities, probability_metrics)


@torch.no_grad()
def output_for(head, features, device):
    values = {'shape': [], 'space': []}
    for part in features.split(8192):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            output = head(part.to(device))
        for key in values:
            values[key].append(output[key].float().cpu())
    return {key: torch.cat(parts) for key, parts in values.items()}


def fit(head, train, valid, args, deadline, out, catalog, device):
    features = train['features'].to(device)
    shape, space, _ = catalog.targets(train['labels'], device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=4,
        threshold=1e-3, min_lr=1e-5)
    random = torch.Generator().manual_seed(args.seed)
    best = significant_best = float('inf')
    stale = updates = skipped = 0
    history = []
    stop = 'epoch_guard'
    for epoch in range(args.epochs):
        head.train(); losses = []
        for ids in torch.randperm(len(shape), generator=random).to(device).split(args.batch):
            check_time(deadline)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type=='cuda'):
                output = head(features[ids])
            loss, ce, bce = shape_space_loss(output, shape[ids], space[ids])
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            success = scaler.get_scale() >= old_scale
            updates += success; skipped += not success
            losses.append((float(ce.detach()), float(bce.detach())))
        head.eval()
        metrics = shape_space_metrics(output_for(head, valid['features'], device), valid['labels'], catalog)
        score = metrics['all']['cross_entropy']+metrics['space_all']['cross_entropy']
        if score < significant_best-1e-4:
            significant_best=score; stale=0
        else:
            stale+=1
        if score < best:
            best=score
            torch.save(dict(head=head.state_dict(), epoch=epoch+1, valid=metrics, score=score), out/'head_best.pt')
        scheduler.step(score)
        lr=optimizer.param_groups[0]['lr']
        averages=np.mean(losses, axis=0)
        row=dict(epoch=epoch+1, train_shape_ce=float(averages[0]), train_space_bce=float(averages[1]),
                 valid=metrics, lr=lr, updates=updates, skipped=skipped, stale_epochs=stale)
        history.append(row); write(out/'training_history.json', history)
        torch.save(dict(head=head.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        scheduler=scheduler.state_dict(), shuffle_rng=random.get_state(), epoch=epoch+1,
                        updates=updates, skipped=skipped, catalog=catalog.dump()), out/'head_latest.pt')
        print(json.dumps(dict(epoch=epoch+1, score=score, shape_error=metrics['text']['shape_error'],
                              space_error=metrics['space_all']['error'], lr=lr)), flush=True)
        if lr<=1.000001e-5 and stale>=10:
            stop='valid_ce_bce_plateau'; break
    best_state=torch.load(out/'head_best.pt', map_location=device, weights_only=False)
    head.load_state_dict(best_state['head']); head.eval()
    write(out/'fit_stop.json', dict(reason=stop, epochs=len(history), best_epoch=best_state['epoch'],
          best_score=best, final_lr=lr, successful_updates=updates, fp16_skips=skipped))
    return history


@torch.no_grad()
def geometry(generator, vocab, catalog, valid, output, old_logits, args, deadline, out):
    ids=np.unique(np.linspace(0,len(valid['labels'])-1,args.compare_steps,dtype=int))
    shape_truth, _, _=catalog.targets(valid['labels'])
    random=torch.Generator(device=generator.device).manual_seed(args.seed)
    old_prob,_=remap_joint_probabilities(old_logits,catalog)
    rows=[]
    controls=np.stack([arc_points(control_stroke(k),catalog.count) for k in catalog.controls])
    controls-=controls.mean(1,keepdims=True)
    for start in range(0,len(ids),32):
        check_time(deadline)
        selected=ids[start:start+32]
        mu,sigma,pi=generator.model.prediction_head(valid['features'][selected].to(generator.device))
        batch_ids=torch.arange(len(selected),device=generator.device)
        latent=[valid['targets'][selected].to(generator.device),mu[batch_ids,pi.argmax(-1)]]
        for _ in range(args.draws):
            component=torch.multinomial(pi,1,generator=random).squeeze(1)
            noise=torch.randn(mu[batch_ids,component].shape,device=generator.device,generator=random)
            latent.append(mu[batch_ids,component]+sigma[batch_ids,component]*noise*.5)
        with generator.autocast():
            decoded=[generator.decode(value) for value in latent]
        for j,index in enumerate(selected):
            size=valid['frames'][index].size if valid['frames'][index] else .085
            templates=np.concatenate([np.stack(catalog.templates)*size, controls])
            def match(points):
                curve=arc_points(points,catalog.count).astype(np.float64);curve-=curve.mean(0)
                distance=np.mean((templates-curve)**2,axis=(1,2))
                choice=int(distance.argmin())
                return dict(shape=choice, error=choice!=int(shape_truth[index]), mse=float(distance[choice]))
            row=dict(index=int(index), gold_shape=int(shape_truth[index]),
                new_head=int(output['shape'][index].argmax()),
                old_head_merged=int(old_prob[index].argmax()),
                old_head_joint_map=int(catalog.mapping[int(old_logits[index].argmax())]),
                source=match(valid['points'][index]), ae=match(decoded[0][j]),
                greedy=match(decoded[1][j]), sample=[match(v[j]) for v in decoded[2:]])
            rows.append(row)
        if start%512==0:
            print(json.dumps(dict(geometry_steps=min(start+32,len(ids)))),flush=True)
    report={}
    for name,predicate in [('all',lambda t:True),('text',lambda t:not t.control),
        ('first_stroke',lambda t:not t.control and t.part==0),
        ('continuation',lambda t:not t.control and t.part>0),('control',lambda t:bool(t.control))]:
        part=[r for r in rows if predicate(vocab.tokens[int(valid['labels'][r['index']])])]
        if not part:continue
        result=dict(count=len(part))
        for key in ['new_head','old_head_merged','old_head_joint_map']:
            result[key]=float(np.mean([r[key]!=r['gold_shape'] for r in part]))
        for key in ['source','ae','greedy','sample']:
            outcomes=[value for r in part for value in (r[key] if key=='sample' else [r[key]])]
            result[key]=float(np.mean([v['error'] for v in outcomes]))
        report[name]=result
    write(out/'geometry_steps.json',rows)
    write(out/'geometry_comparison.json',dict(groups=report, sample_draws=args.draws, temperature=.5,
        metric='directed_arc64_centered_canvas_MSE_at_reference_font_size',
        reference_layout_only_for_external_matching=True, AE_points_unchanged=True))
    if report['all']['source']!=0:
        raise AssertionError('源标准曲线的形状量化未对齐')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('tmp_smoke/r_text_readout_20261013'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max_seconds',type=float,default=7200)
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--batch',type=int,default=4096)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--hidden',type=int,default=256)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--compare_steps',type=int,default=4096)
    parser.add_argument('--draws',type=int,default=4)
    parser.add_argument('--generation_strokes',type=int,default=480)
    args=parser.parse_args()
    if min(args.max_seconds,args.epochs,args.batch,args.compare_steps,args.draws,args.generation_strokes)<=0:
        parser.error('运行参数必须为正')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'requested.json').exists():raise FileExistsError('已有实验记录，不覆盖')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    started=time.monotonic();deadline=started+args.max_seconds
    write(args.out/'requested.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    generator,old_head,vocab,layout,state=load_readout(args.baseline/'head_final.pt')
    generator.model.encode_chunk_size=128
    before=model_hash(generator.model)
    catalog=ShapeVocabulary(vocab)
    write(args.out/'shape_vocabulary.json',catalog.dump())
    cached={split:torch.load(args.baseline/f'features_{split}.pt',map_location='cpu',weights_only=False)
            for split in ['train','valid']}
    for split,value in cached.items():
        if value['features'].shape[1]!=state['input_width'] or int(value['labels'].max())>=len(vocab):
            raise ValueError('缓存形状/词表不符')
    old_logits=logits_for(old_head,cached['valid']['features'],generator.device)
    old_probability,old_space=remap_joint_probabilities(old_logits,catalog)
    baseline=probability_metrics(old_probability,old_space,cached['valid']['labels'],catalog)
    old_flag=torch.as_tensor(catalog.leading)[old_logits.argmax(-1)]
    truth=torch.as_tensor(catalog.leading)[cached['valid']['labels']]
    baseline['old_joint_argmax_space_error']=float((old_flag!=truth).float().mean())
    write(args.out/'baseline_classification.json',baseline)
    del old_head
    head=ShapeSpaceReadout(state['input_width'],catalog,args.hidden).to(generator.device)
    history=fit(head,cached['train'],cached['valid'],args,deadline,args.out,catalog,generator.device)
    output=output_for(head,cached['valid']['features'],generator.device)
    metrics=shape_space_metrics(output,cached['valid']['labels'],catalog)
    write(args.out/'classification.json',metrics)
    torch.save(dict(kind='shape_space',head=head.state_dict(), hidden=args.hidden,input_width=state['input_width'],
        shape_vocabulary=catalog.dump(), vocabulary=vocab.dump(),r_source=state['r_source'],
        r_model_hash=before,font_sha256=state['font_sha256'],layout=asdict(layout),
        input='last_R_hidden_only', loss='CE_shape_all+BCE_space_all_equal_weight',valid=metrics),args.out/'head_final.pt')
    geometry(generator,vocab,catalog,cached['valid'],output,old_logits,args,deadline,args.out)
    prompts=['Once upon a time, there was a little cat named Tom.',
             'Once upon a time, there was a little dog named Max.',
             'Lily was hungry. She went to the kitchen to get',
             'Tom dropped his toy into the lake. He felt']
    for index,prompt in enumerate(prompts):
        value=generate(generator,head,vocab,prompt,deadline,args,layout)
        write(args.out/f'prompt{index}.json',value)
        save_native_svg(value,args.out/f'prompt{index}.svg',f'Shape / space / prompt{index}')
        print(json.dumps(dict(prompt=index,stop=value['stop_reason'],strokes=value['generated_strokes'])),flush=True)
    after=model_hash(generator.model)
    if before!=after:raise AssertionError('冻结模型发生变化')
    cache_files={split:dict(path=str((args.baseline/f'features_{split}.pt').resolve()),
                           bytes=(args.baseline/f'features_{split}.pt').stat().st_size) for split in cached}
    write(args.out/'completion.json',dict(complete=True,seconds=time.monotonic()-started,
        R_unchanged=True,weight_hash_before=before,weight_hash_after=after,
        old_head_file_sha256=hashlib.sha256((args.baseline/'head_final.pt').read_bytes()).hexdigest(),
        classes=len(catalog),ink_classes=catalog.ink_classes,head_parameters=sum(p.numel() for p in head.parameters()),
        train_strokes=len(cached['train']['labels']),valid_strokes=len(cached['valid']['labels']),
        epochs=len(history),gpu='local',precision='FP16 head/backbone; FP32 CE/BCE',
        cache_reused=cache_files, train_augmentation=True,valid_augmentation=False,
        no_character_classification=True,space_supervised_every_stroke=True))
    from scripts.review.render_shape_space_review import render
    render(args.out,args.baseline)
    print(str(args.out/'index.html'),flush=True)


if __name__=='__main__':main()
