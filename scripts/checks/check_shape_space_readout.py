"""2026-10-14：共享形状标签、逐笔空格梯度及字体前缀接线的CPU检查。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import argparse
import json
import numpy as np
import torch
from src.text_renderer import VectorFont
from src.page_document import build_document,layout_text_pages
from src.st.text_readout import StrokeVocabulary,StrokeToken,SymbolicRenderer,document_trace,arc_points
from src.st.text_readout_policy import audit_font_sequence
from src.st.text_shape_readout import (ShapeVocabulary,ShapeSpaceReadout,ShapePrefixPolicy,
    shape_space_loss,shape_space_metrics,remap_joint_probabilities)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();torch.set_num_threads(2)
    font=VectorFont.from_svg('datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg')
    vocab=StrokeVocabulary(font);catalog=ShapeVocabulary(vocab)
    assert catalog.identity_to_shape['h',0]==catalog.identity_to_shape['k',0]
    assert catalog.identity_to_shape['h',1]==catalog.identity_to_shape['m',1]==catalog.identity_to_shape['n',1]
    assert catalog.mapping[vocab.index[StrokeToken('h',0)]]==catalog.mapping[vocab.index[StrokeToken('h',0,True)]]
    sample=build_document(layout_text_pages('A cat sat. A kitten!',font),complete=True)
    labels,_=document_trace(sample,vocab)
    shapes,spaces,first=catalog.targets(labels)
    assert torch.all(spaces[~first]==0)
    head=ShapeSpaceReadout(768,catalog)
    output=head(torch.randn(len(labels),768))
    output['space'].retain_grad()
    loss,ce,bce=shape_space_loss(output,shapes,spaces);loss.backward()
    assert torch.all(output['space'].grad!=0), '续接笔/控制符必须也有空格监督梯度'
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    joint=torch.randn(len(labels),len(vocab))
    shape_p,space_p=remap_joint_probabilities(joint,catalog)
    assert torch.allclose(shape_p.sum(-1),torch.ones(len(labels)),atol=1e-6)
    metrics=shape_space_metrics(output,torch.tensor(labels),catalog)
    assert metrics['space_all']['count']==len(labels)
    prompt=build_document(layout_text_pages('A cat',font),complete=False)
    renderer=SymbolicRenderer(vocab,prompt)
    trial=renderer.fork()
    reference=arc_points(trial.append(vocab.index[StrokeToken('h',0,True)])[-1]['points'])
    def choose_h_position(curves,hidden):
        return [-np.mean((arc_points(curve)-reference)**2) for curve in curves]
    policy=ShapePrefixPolicy(vocab,catalog,choose_h_position)
    def step(character,part,leading=False):
        logits=np.full(len(catalog),-100.)
        logits[catalog.identity_to_shape[character,part]]=10
        return policy.append(renderer,dict(shape=logits,space=10 if leading else -10))
    step('h',0,True)
    assert {'h','k'}<=set(policy.candidates)
    step('k',1,True) # 每步二分类输出；字体约束会抑制续接笔的非法空格。
    assert renderer.text=='A cat k' and policy.trace[-1]['space_suppressed']
    assert not policy.trace[-1]['space_applied']
    logits=np.full(len(catalog),-100.);logits[catalog.controls['eod']]=10
    policy.append(renderer,dict(shape=logits,space=-10))
    policy.append(renderer,dict(shape=logits,space=10))
    assert renderer.complete and audit_font_sequence(renderer.records(),vocab)['violations']==0
    # 位置歧义只在已胜出的同一形状内调用打分，不能偷偷改选其它形状。
    renderer=SymbolicRenderer(vocab,prompt);calls=[]
    def rank(curves,hidden):
        calls.append(len(curves));return np.arange(len(curves))
    policy=ShapePrefixPolicy(vocab,catalog,rank)
    logits=np.full(len(catalog),-100.);sid=catalog.identity_to_shape['.',0];logits[sid]=10
    policy.append(renderer,dict(shape=logits,space=10))
    assert calls and policy.trace[-1]['selected_shape']==sid
    report=dict(shape_classes=len(catalog),ink_classes=catalog.ink_classes,
        head_parameters=sum(p.numel() for p in head.parameters()),shared_h_k=True,
        shared_nonfirst_strokes=True,space_not_part_of_shape_class=True,
        space_gradient_every_stroke=True,probability_remap_normalized=True,
        delayed_character_resolution=True,illegal_continuation_space_recorded=True,
        placement_rank_only_inside_winning_shape=True,EOS_protocol=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(report)


if __name__=='__main__':main()
