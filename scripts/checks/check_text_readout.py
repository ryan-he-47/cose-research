"""2026-10-13：读出标签、模板量化、符号渲染及单层头的CPU短检查。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import argparse
import json
from src.page_document import build_document, layout_text_pages
from src.text_renderer import VectorFont
from src.st.text_readout import (StrokeVocabulary, StrokeReadout, document_trace,
    candidate_templates, match_curve, SymbolicRenderer, classification_metrics, StrokeToken)
from src.st.text_readout_policy import GlyphPrefixPolicy, audit_font_sequence


def check_prefix_policy(font,vocab):
    def scores(*items):
        logits=np.full(len(vocab),-100.,np.float64)
        for token,value in items:logits[vocab.index[token]]=value
        return logits
    prompt=build_document(layout_text_pages('A cat',font),complete=False)
    renderer=SymbolicRenderer(vocab,prompt);policy=GlyphPrefixPolicy(vocab)
    policy.append(renderer,scores((StrokeToken('m',0,True),8),(StrokeToken('n',0,True),7),
                                  (StrokeToken('b',1),20)))
    assert {'m','n'}<=set(policy.candidates) and renderer.text.endswith('m')
    policy.append(renderer,scores((StrokeToken('m',1),8),(StrokeToken('n',1),7),
                                  (StrokeToken('a',0),20)))
    assert {'m','n'}<=set(policy.candidates) and policy.prefix_length==2
    policy.append(renderer,scores((StrokeToken('a',0),8),(StrokeToken('m',2),-10)))
    assert renderer.text.endswith('na') and policy.trace[-1]['finished_character']=='n'
    assert audit_font_sequence(renderer.records(),vocab)['violations']==0
    renderer=SymbolicRenderer(vocab,prompt);policy=GlyphPrefixPolicy(vocab)
    policy.append(renderer,scores((StrokeToken('h',0,True),8),(StrokeToken('k',0,True),7)))
    assert {'h','k'}<=set(policy.candidates)
    policy.append(renderer,scores((StrokeToken('k',1),8),(StrokeToken('h',1),-10)))
    assert renderer.text.endswith('k') and audit_font_sequence(renderer.records(),vocab)['violations']==0
    policy.append(renderer,scores((StrokeToken(control='eod'),8)))
    policy.append(renderer,scores((StrokeToken('a',0),20)))
    assert renderer.complete and policy.trace[-1]['forced']
    assert audit_font_sequence(renderer.records(),vocab)['violations']==0
    # 局部路径相同仍不够：此处点号/冒号的kerning不同，应排除实际位置不同的候选。
    renderer=SymbolicRenderer(vocab,prompt);policy=GlyphPrefixPolicy(vocab)
    policy.append(renderer,scores((StrokeToken('.',0,True),8),(StrokeToken(':',0,True),7)))
    assert ':' not in policy.candidates and renderer.text.endswith('.')
    # 字形未画完时不能停在半字；m/n共有前两笔，停止时可落到完整的n。
    renderer=SymbolicRenderer(vocab,prompt);policy=GlyphPrefixPolicy(vocab)
    policy.append(renderer,scores((StrokeToken('m',0,True),8),(StrokeToken('n',0,True),7)))
    assert policy.finish_at_limit(renderer) is None
    policy.append(renderer,scores((StrokeToken('m',1),8),(StrokeToken('n',1),7)))
    assert policy.finish_at_limit(renderer) is not None and renderer.text.endswith('n')
    assert audit_font_sequence(renderer.records(),vocab)['unfinished_character'] is None
    # 原字体数据的完整文档也要通过独立审计；不是只查新策略自己写的flag。
    full=build_document(layout_text_pages('Hello, my name is Kim. I am happy!',font),complete=True)
    labels,_=document_trace(full,vocab)
    rows=[dict(token=int(i)) for i in labels]
    assert audit_font_sequence(rows,vocab)['violations']==0
    assert audit_font_sequence(rows+[dict(token=vocab.index[StrokeToken('a',0)])],vocab)['violations']>0
    return dict(shared_first_stroke=True,short_terminal_competes_with_continuation=True,
                delayed_identity_revision=True,forced_eod_eop=True,independent_font_audit=True,
                actual_position_alias_filter=True,complete_glyph_at_stroke_limit=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,help='可选：实际冻结R的FP16前向/因果/梯度接线，0优化更新')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    font = VectorFont.from_svg('datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg')
    vocab = StrokeVocabulary(font)
    document = build_document(layout_text_pages('A little cat sat. It said meow! '*8,font), complete=True)
    labels, frames = document_trace(document,vocab)
    source_alias = 0
    for index, element in enumerate(document.to_sequence()):
        templates = candidate_templates(vocab,frames[index])
        result = match_curve(element.rawxy,templates)
        if int(labels[index]) not in result['aliases']:
            raise AssertionError(f'源字形模板未对齐：{index} {vocab.tokens[labels[index]]} {result}')
        source_alias += len(result['aliases']) > 1
    for window in document.iter_windows(1024):
        wlabels, _ = document_trace(window,vocab)
        assert len(wlabels) == window.stroke_count
    prompt = build_document(layout_text_pages('A cat',font),complete=False)
    renderer = SymbolicRenderer(vocab,prompt)
    assert len(renderer.initial) == len(prompt.to_sequence())-1
    for old,new in zip(prompt.to_sequence()[:-1],renderer.initial):
        assert np.array_equal(old.rawxy,np.asarray(new['points'],np.float32))
    continuation = ' sat.'
    for char_index,char in enumerate(continuation):
        if char == ' ':continue
        for part in range(vocab.parts[char]):
            renderer.append(vocab.index[StrokeToken(char,part,char_index==1 and part==0)])
    expected = build_document(layout_text_pages('A cat sat.',font),complete=False).to_sequence()[:-1]
    actual = renderer.records()
    assert len(expected) == len(actual)
    for old,new in zip(expected,actual):
        assert np.array_equal(old.rawxy,np.asarray(new['points'],np.float32))
    renderer.append(vocab.index[StrokeToken(control='eod')])
    renderer.append(vocab.index[StrokeToken(control='eop')])
    assert renderer.complete and renderer.invalid_parts == 0
    features = torch.randn(16,768)
    head = StrokeReadout(768,len(vocab))
    targets = torch.tensor(labels[1:17])
    loss = torch.nn.functional.cross_entropy(head(features),targets)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    metrics = classification_metrics(head(features).detach(),targets,vocab)
    report=dict(classes=len(vocab),source_strokes=len(labels),source_alias_steps=source_alias,
               exact_renderer=True,gradient_finite=True,metric_groups=list(metrics))
    report['prefix_policy']=check_prefix_policy(font,vocab)
    if args.checkpoint:
        from scripts.demo.demo_pretrain_generation import load_generator
        from src.st.pipeline_pretrain import collate_documents
        generator,_,_=load_generator(args.checkpoint,'cuda')
        model=generator.model.requires_grad_(False)
        sample=build_document(layout_text_pages('A cat sat.',font),complete=True)
        symbolic,_=document_trace(sample,vocab)
        batch=collate_documents([sample])
        batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.no_grad(),generator.autocast():
            embedded=model.encode_valid(batch)
            h=model.hidden_states(embedded[:,:-1],batch['slice_id'][:,:-1],batch['stroke_pad'][:,:-1],right_padded=True)
            changed=embedded.clone();changed[:,-1]+=1
            h2=model.hidden_states(changed[:,:-1],batch['slice_id'][:,:-1],batch['stroke_pad'][:,:-1],right_padded=True)
        assert torch.equal(h,h2)
        head=StrokeReadout(768,len(vocab)).cuda()
        mask=model.target_mask(batch)
        y=torch.from_numpy(symbolic[1:]).cuda()[mask[0]]
        with generator.autocast():logits=head(h[mask].detach())
        loss=torch.nn.functional.cross_entropy(logits.float(),y);loss.backward()
        assert all(p.grad is None for p in model.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        if args.out:
            from scripts.review.train_text_readout import geometry_comparison, generate
            from types import SimpleNamespace
            import time
            geometry_out=args.out.parent/'geometry_smoke'
            geometry_out.mkdir(parents=True,exist_ok=True)
            indices=torch.where(mask[0])[0].cpu().numpy()+1
            _,frames=document_trace(sample,vocab)
            valid=dict(features=h[mask].cpu().half(),labels=y.cpu(),
                       targets=embedded[:,1:][mask].cpu().float(),
                       frames=[frames[i] for i in indices],points=[sample.to_sequence()[i].rawxy for i in indices])
            options=SimpleNamespace(compare_steps=16,draws=1,seed=42,generation_strokes=16)
            compared=geometry_comparison(generator,vocab,valid,logits.detach().cpu(),options,
                                        time.monotonic()+60,geometry_out)
            assert compared['all']['source']['same_shape_error']==0
            scripted=[]
            for index,char in enumerate('sat.'):
                scripted.extend(vocab.index[StrokeToken(char,p,index==0 and p==0)] for p in range(vocab.parts[char]))
            scripted += [vocab.index[StrokeToken(control=kind)] for kind in ['eod','eop']]
            class ScriptedHead(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.index=0
                def forward(self,x):
                    values=x.new_full((1,len(vocab)),-10)
                    values[0,scripted[self.index]]=10;self.index+=1
                    return values
            from src.page_document import PageLayoutConfig
            output=generate(generator,ScriptedHead(),vocab,'A cat',time.monotonic()+60,options,PageLayoutConfig())
            assert output['stop_reason']=='eod_eop' and output['invalid_part_transitions']==0
            assert output['generated_page_texts'][0]=='A cat sat.'
            report.update(template_comparison_wiring=True,scripted_generation_wiring=True)
        report.update(fp16_forward=True,no_future_embedding=True,R_gradients_none=True,optimizer_updates=0,
                      actual_targets=len(y),head_parameters=sum(p.numel() for p in head.parameters()))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
