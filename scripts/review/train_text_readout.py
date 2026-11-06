"""2026-10-13：本机冻结R读出实验；时间预算必须显式指定，只优化小分类头。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from scripts.demo.demo_pretrain_generation import load_generator, save_native_svg
from scripts.review.diagnose_formal_r_health import model_hash
from src.page_document import PageLayoutConfig, build_document, layout_text_pages
from src.st.generation_pretrain import PageKVCache
from src.st.pipeline_pretrain import collate_documents
from src.st.text_readout import (StrokeVocabulary, StrokeReadout, SymbolicRenderer,
    document_trace, candidate_templates, match_curve, classification_metrics, arc_points)
from src.st.text_readout_policy import GlyphPrefixPolicy, audit_font_sequence
from src.text_augmentation import TextAugmentationConfig, augment_rendered_text


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('已到本次显式预算；保存已有产物，不追加计算')


@torch.no_grad()
def extract(generator, vocab, rows, split, deadline, layout):
    features, labels, targets, frames, points = [], [], [], [], []
    for index, row in enumerate(rows):
        check_time(deadline)
        rng = np.random.default_rng(np.random.SeedSequence([42,0,row['logical_index']]))
        augment = partial(augment_rendered_text,rng=rng,config=TextAugmentationConfig()) if split=='train' else None
        document = build_document(layout_text_pages(row['text'],vocab.font,config=layout,augment=augment),
                                  complete=True,metadata={'source_id':row['source_id']})
        for window in document.iter_windows(generator.model.cfg.max_context_strokes):
            check_time(deadline)
            symbolic, positions = document_trace(window,vocab)
            cpu = collate_documents([window],max_strokes=generator.model.cfg.max_context_strokes)
            batch = {k:v.to(generator.device) if torch.is_tensor(v) else v for k,v in cpu.items()}
            with generator.autocast():
                embedding = generator.model.encode_valid(batch)
                hidden = generator.model.hidden_states(embedding[:,:-1],batch['slice_id'][:,:-1],
                    batch['stroke_pad'][:,:-1],right_padded=True)
            selected = generator.model.target_mask(batch)
            indices = torch.where(selected[0])[0].cpu().numpy()+1
            features.append(hidden[selected].cpu().half())
            labels.append(torch.from_numpy(symbolic[indices]))
            if split=='valid':
                targets.append(embedding[:,1:][selected].cpu().float())
                sequence = window.to_sequence()
                frames.extend(positions[i] for i in indices)
                points.extend(sequence[i].rawxy for i in indices)
        if index%16==0:
            print(json.dumps({'extract':split,'stories':index+1,'strokes':sum(len(v) for v in labels)}),flush=True)
    result = dict(features=torch.cat(features),labels=torch.cat(labels))
    if split=='valid':
        result.update(targets=torch.cat(targets),frames=frames,points=points)
    return result


@torch.no_grad()
def logits_for(head, features, device):
    values=[]
    for part in features.split(8192):
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            values.append(head(part.to(device)).float().cpu())
    return torch.cat(values)


def fit(head, train, valid, args, deadline, out, vocab, device):
    features, labels = train['features'].to(device),train['labels'].to(device)
    optimizer = torch.optim.AdamW(head.parameters(),lr=args.lr,weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=4,
        threshold=1e-3,min_lr=1e-5)
    successful, history = 0, []
    best=float('inf');significant_best=float('inf');stale=0;stop='epoch_guard'
    random = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(len(labels),generator=random).to(device)
        losses=[]
        for ids in order.split(args.batch):
            check_time(deadline)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                logits = head(features[ids])
            loss = torch.nn.functional.cross_entropy(logits.float(),labels[ids])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            successful += scaler.get_scale() >= old_scale
            losses.append(float(loss.detach()))
        head.eval()
        metrics = classification_metrics(logits_for(head,valid['features'],device),valid['labels'],vocab)
        score=metrics['all']['cross_entropy']
        if score < significant_best-1e-4:
            significant_best=score;stale=0
        else:stale+=1
        if score < best:
            best=score
            torch.save({'head':head.state_dict(),'epoch':epoch+1,'valid':metrics,
                        'hidden':args.hidden,'input_width':head.net[0].in_features if args.hidden else head.net.in_features},
                       out/'head_best.pt')
        scheduler.step(score)
        lr=optimizer.param_groups[0]['lr']
        row = dict(epoch=epoch+1,train_ce=float(np.mean(losses)),valid=metrics,successful_updates=successful,
                   lr=lr,best_valid_ce=best,stale_epochs=stale)
        history.append(row)
        write(out/'training_history.json',history)
        print(json.dumps({'epoch':epoch+1,'train_ce':row['train_ce'],'valid':metrics['text'],'lr':lr}),flush=True)
        torch.save({'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'scheduler':scheduler.state_dict(),'epoch':epoch+1,'updates':successful,
                    'hidden':args.hidden,'vocabulary':vocab.dump()},out/'head_latest.pt')
        if lr<=1.000001e-5 and stale>=10:
            stop='valid_ce_plateau';break
    best_state=torch.load(out/'head_best.pt',map_location=device,weights_only=False)
    head.load_state_dict(best_state['head']);head.eval()
    write(out/'fit_stop.json',dict(reason=stop,epochs=len(history),best_epoch=best_state['epoch'],
        best_valid_ce=best,final_lr=lr,successful_updates=successful))
    return history


def token_identity(vocab, index):
    t=vocab.tokens[int(index)]
    return t.character,t.part,t.control


@torch.no_grad()
def geometry_comparison(generator, vocab, valid, logits, args, deadline, out):
    ids = np.unique(np.linspace(0,len(valid['labels'])-1,min(args.compare_steps,len(valid['labels'])),dtype=int))
    random = torch.Generator(device=generator.device).manual_seed(args.seed)
    result=[]
    for start in range(0,len(ids),32):
        check_time(deadline)
        batch_ids=ids[start:start+32]
        hidden=valid['features'][batch_ids].to(generator.device)
        mu,sigma,pi=generator.model.prediction_head(hidden)
        row=torch.arange(len(batch_ids),device=generator.device)
        latent=[valid['targets'][batch_ids].to(generator.device),mu[row,pi.argmax(-1)]]
        for _ in range(args.draws):
            component=torch.multinomial(pi,1,generator=random).squeeze(1)
            noise=torch.randn(mu[row,component].shape,device=generator.device,generator=random)
            latent.append(mu[row,component]+sigma[row,component]*noise*.5)
        with generator.autocast():
            decoded=[generator.decode(value) for value in latent]
        for index, target_index in enumerate(batch_ids):
            frame=valid['frames'][target_index]
            templates=candidate_templates(vocab,frame)
            gold=int(valid['labels'][target_index]);head=int(logits[target_index].argmax())
            matches=[match_curve(valid['points'][target_index],templates)]
            matches += [match_curve(value[index],templates) for value in decoded]
            # 同一个候选集合/画布条件对所有方法作同样匹配；不使用目标笔画选候选。
            aliases=np.flatnonzero(np.mean((templates-templates[head])**2,axis=(1,2))<=1e-12).tolist()
            truth=arc_points(valid['points'][target_index])
            all_predictions=[templates[head],truth]+[arc_points(value[index]) for value in decoded]
            target_errors=[]
            for points in all_predictions:
                total=float(np.mean((points-truth)**2))
                position=float(np.mean((points.mean(0)-truth.mean(0))**2))
                target_errors.append(dict(target_canvas_mse=total,target_position_mse=position,
                                          target_shape_mse=total-position))
            for value,error in zip(matches,target_errors[1:]):value.update(error)
            result.append(dict(index=int(target_index),gold=gold,head=head,head_aliases=aliases,
                               head_geometry=target_errors[0],
                               source=matches[0],ae=matches[1],greedy=matches[2],sample=matches[3:]))
        if start%512==0:
            print(json.dumps({'geometry_steps':min(start+32,len(ids)),'target_steps':len(ids)}),flush=True)
    report={}
    for group, predicate in [('all',lambda t:True),('text',lambda t:not t.control),
                             ('control',lambda t:bool(t.control)),('first_stroke',lambda t:not t.control and t.part==0),
                             ('continuation',lambda t:not t.control and t.part>0)]:
        selected=[r for r in result if predicate(vocab.tokens[r['gold']])]
        if not selected:continue
        report[group]={}
        for method in ['head','source','ae','greedy','sample']:
            errors,identity_errors,alias_errors,distances,canvas_errors=[],[],[],[],[]
            for r in selected:
                values=([dict(prediction=r['head'],aliases=r['head_aliases'],**r['head_geometry'])] if method=='head' else
                        r[method] if method=='sample' else [r[method]])
                for value in values:
                    errors.append(value['prediction']!=r['gold'])
                    identity_errors.append(token_identity(vocab,value['prediction'])!=token_identity(vocab,r['gold']))
                    alias_errors.append(r['gold'] not in value['aliases'])
                    if 'canvas_mse' in value:distances.append(value['canvas_mse'])
                    canvas_errors.append(value['target_canvas_mse'])
            report[group][method]=dict(steps=len(selected),joint_error=float(np.mean(errors)),
                identity_error=float(np.mean(identity_errors)),same_shape_error=float(np.mean(alias_errors)),
                curve_match_mse=float(np.mean(distances)) if distances else None,
                target_canvas_mse=float(np.mean(canvas_errors)))
    write(out/'geometry_steps.json',result)
    write(out/'geometry_comparison.json',dict(groups=report,sample_draws=args.draws,temperature=.5,
        template_matching='ordered_arc32_canvas_mse',alias_tolerance=1e-12,
        layout_condition='known_reference_page_font_size_and_line_geometry; class_head_uses_hidden_only',
        space_at_new_line_or_page_is_geometrically_ambiguous=True))
    return report


@torch.no_grad()
def generate(generator, head, vocab, prompt, deadline, args, layout):
    document=build_document(layout_text_pages(prompt,vocab.font,config=layout,targetable=False),complete=False)
    renderer=SymbolicRenderer(vocab,document,layout)
    constraint=getattr(args,'constraint','none')
    if constraint not in ('none','prefix'):raise ValueError('constraint须为none/prefix')
    factorized=getattr(head,'factorized',False)
    if factorized:
        from src.st.text_shape_readout import ShapePrefixPolicy, placement_ranker
        policy=ShapePrefixPolicy(vocab,head.catalog,placement_ranker(generator))
        constraint='shape_prefix'
    else:
        policy=GlyphPrefixPolicy(vocab) if constraint=='prefix' else None
    records=renderer.initial
    cache=PageKVCache(generator.model)
    refreshes=condition_refreshes=0
    def prefill(rows):
        embeddings=generator.encode([np.asarray(r['points'],np.float32) for r in rows]).unsqueeze(0)
        pages=torch.tensor([[r['page_id'] for r in rows]],device=generator.device)
        return cache.forward(embeddings,pages,prefill=True)
    with generator.autocast():
        hidden=prefill(records)
        max_parts=max(vocab.parts.values())
        for step in range(args.generation_strokes+(max_parts+1 if policy else 0)):
            check_time(deadline)
            if policy and (step>=args.generation_strokes or len(records)>=generator.model.cfg.max_context_strokes-max_parts):
                finished=policy.finish_at_limit(renderer)
                if finished is not None:
                    records=finished
                    stop='max_new_strokes' if step>=args.generation_strokes else 'context_full'
                    break
            output=head(hidden[:,-1])
            if factorized:
                output={k:v.float()[0].detach().cpu().numpy() for k,v in output.items()}
                updated=policy.append(renderer,output,hidden[:,-1])
            elif policy:updated=policy.append(renderer,output.float()[0].detach().cpu().numpy())
            else:updated=renderer.append(int(output.float()[0].argmax()))
            moved=[i for i,(old,new) in enumerate(zip(records,updated)) if old['points']!=new['points']]
            if moved:
                refreshes+=1
                condition_refreshes+=any(not records[i]['generated'] for i in moved)
            records=updated
            if renderer.complete:
                stop='eod_eop';break
            if renderer.page_id-document.pages[-1].page_id>=8:
                stop='max_new_pages';break
            if len(records)>=generator.model.cfg.max_context_strokes:
                stop='context_full';break
            if moved:
                hidden=prefill(records)
            else:
                encoded=generator.encode([np.asarray(records[-1]['points'],np.float32)]).unsqueeze(0)
                pages=torch.tensor([[records[-1]['page_id']]],device=generator.device)
                hidden=cache.forward(encoded,pages)
        else:
            stop='max_new_strokes'
    text_pages=dict(renderer.page_texts)
    if not renderer.complete:text_pages[renderer.page_id]=renderer.text
    return dict(strokes=records,prompt=prompt,stop_reason=stop,generated_strokes=sum(r['generated'] for r in records),
                generated_page_texts=text_pages,invalid_part_transitions=renderer.invalid_parts,
                layout_refreshes=refreshes,conditioning_refreshes=condition_refreshes,
                feedback='canonical_font_to_frozen_E',future_reference_used=False,
                constraint=constraint,font_audit=audit_font_sequence(records,vocab),
                constraint_summary=policy.summary() if policy else None,
                constraint_trace=policy.trace if policy else [],
                rendering='existing_word_wrap_and_ink_alignment_on_current_symbolic_prefix')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--stories',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max_seconds',type=float,required=True)
    parser.add_argument('--hidden',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=300,help='工程上限；验证CE平台且LR到1e-5后提前停止')
    parser.add_argument('--batch',type=int,default=4096)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--compare_steps',type=int,default=4096)
    parser.add_argument('--draws',type=int,default=4)
    parser.add_argument('--generation_strokes',type=int,default=480)
    args=parser.parse_args()
    if args.max_seconds<=0 or args.epochs<1 or args.batch<1 or args.draws<1 or args.compare_steps<1:
        parser.error('必须指定正的预算和运行参数')
    if not torch.cuda.is_available():parser.error('本实验使用本机CUDA/FP16环境')
    # 后台启动器先把stdout/stderr放入运行目录；已有实验仍不得覆盖。
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'requested.json').exists():
        raise FileExistsError(f'运行目录已有实验记录：{args.out}')
    started=time.monotonic();deadline=started+args.max_seconds
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    write(args.out/'requested.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    generator,font,metadata=load_generator(args.checkpoint,'cuda')
    generator.model.requires_grad_(False);generator.model.encode_chunk_size=128
    before=model_hash(generator.model)
    layout_spec=dict(metadata['data_manifest'].get('page_layout',{}))
    if 'body_box' in layout_spec:layout_spec['body_box']=tuple(layout_spec['body_box'])
    layout=PageLayoutConfig(**layout_spec)
    vocab=StrokeVocabulary(font)
    write(args.out/'vocabulary.json',vocab.dump())
    data={split:[json.loads(line) for line in (args.stories/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
          for split in ['valid','train']}
    if set(r['source_id'] for r in data['train']) & set(r['source_id'] for r in data['valid']):
        raise ValueError('读出训练/验证故事重叠')
    cached={}
    head=None
    try:
        for split in ['valid','train']:
            cached[split]=extract(generator,vocab,data[split],split,deadline,layout)
            torch.save(cached[split],args.out/('features_'+split+'.pt'))
        head=StrokeReadout(generator.model.cfg.r_d_model,len(vocab),args.hidden).to(generator.device)
        history=fit(head,cached['train'],cached['valid'],args,deadline,args.out,vocab,generator.device)
        logits=logits_for(head,cached['valid']['features'],generator.device)
        valid=classification_metrics(logits,cached['valid']['labels'],vocab)
        write(args.out/'classification.json',valid)
        torch.save({'head':head.state_dict(),'hidden':args.hidden,'vocabulary':vocab.dump(),
            'input_width':generator.model.cfg.r_d_model,
            'r_source':metadata,'font_sha256':metadata['data_manifest']['font_sha256'],
            'r_model_hash':before,
            'layout':asdict(layout),'input':'last_R_hidden_only','valid':valid},args.out/'head_final.pt')
        geometry=geometry_comparison(generator,vocab,cached['valid'],logits,args,deadline,args.out)
        prompts=['Once upon a time, there was a little cat named Tom.',
                 'Once upon a time, there was a little dog named Max.',
                 'Lily was hungry. She went to the kitchen to get',
                 'Tom dropped his toy into the lake. He felt']
        generations=[]
        head.eval()
        for index,prompt in enumerate(prompts):
            value=generate(generator,head,vocab,prompt,deadline,args,layout)
            write(args.out/f'prompt{index}.json',value)
            save_native_svg(value,args.out/f'prompt{index}.svg',f'Readout / prompt{index}')
            generations.append({k:v for k,v in value.items() if k!='strokes'})
            write(args.out/'generations.json',generations)
        after=model_hash(generator.model)
        assert before==after
        write(args.out/'completion.json',dict(complete=True,seconds=time.monotonic()-started,
            max_seconds=args.max_seconds,R_unchanged=True,weight_hash_before=before,weight_hash_after=after,
            optimized_parameters='readout_only',head_parameters=sum(p.numel() for p in head.parameters()),
            classes=len(vocab),train_strokes=len(cached['train']['labels']),valid_strokes=len(cached['valid']['labels']),
            epochs=len(history),precision='FP16 backbone/head; FP32 CE',gpu='local',
            train_augmentation=True,valid_augmentation=False))
    except TimeoutError as error:
        write(args.out/'budget_stop.json',dict(reason=str(error),seconds=time.monotonic()-started,
            complete=False,cached_splits=list(cached),head_checkpoint=(args.out/'head_latest.pt').exists()))
        raise
    print(json.dumps({'complete':True,'out':str(args.out),'valid':valid['text']}),flush=True)


if __name__=='__main__':main()
