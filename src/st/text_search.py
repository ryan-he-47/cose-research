"""2026-10-24：标准笔画的后缀缓存、确定续笔合并与小束搜索。"""
from dataclasses import dataclass
import time
import numpy as np
import torch

from .generation_pretrain import PageKVCache


def changed_prefix(old, new):
    return [i for i,(a,b) in enumerate(zip(old,new))
            if a['page_id']!=b['page_id'] or a['points']!=b['points']]


def encode_rows(generator, rows):
    values=generator.encode_cached([np.asarray(r['points'],np.float32) for r in rows]).unsqueeze(0)
    pages=torch.tensor([[r['page_id'] for r in rows]],device=generator.device)
    return values,pages


@dataclass
class TextState:
    renderer: object
    policy: object
    records: list
    cache: object
    hidden: object
    steps: int=0
    score: float=0.
    refreshes: int=0
    condition_refreshes: int=0
    forced_blocks: int=0
    forced_strokes: int=0
    stop: str | None=None

    def fork(self):
        return TextState(self.renderer.fork(),self.policy.fork(),list(self.records),self.cache.fork(),
                         self.hidden,self.steps,self.score,self.refreshes,self.condition_refreshes,
                         self.forced_blocks,self.forced_strokes,self.stop)

    def advance(self, generator, rows):
        moved=changed_prefix(self.records,rows)
        if moved:
            self.refreshes+=1
            self.condition_refreshes+=any(not self.records[i]['generated'] for i in moved)
        start=min(moved) if moved else len(self.records)
        self.cache.truncate(start)
        self.hidden=self.cache.forward(*encode_rows(generator,rows[start:]))[:,-1:]
        self.records=rows


def limit_reason(state, generator, args, first_page, max_parts, *, before=False):
    if state.renderer.complete:return 'eod_eop'
    if state.renderer.page_id-first_page>=8:return 'max_new_pages'
    if len(state.records)>=generator.model.cfg.max_context_strokes:return 'context_full'
    if before and state.policy and (state.steps>=args.generation_strokes or
            len(state.records)>=generator.model.cfg.max_context_strokes-max_parts):
        finished=state.policy.finish_at_limit(state.renderer)
        if finished is not None:
            state.records=finished
            return 'max_new_strokes' if state.steps>=args.generation_strokes else 'context_full'
    return None


def head_arrays(head, hidden):
    return {k:v.float().detach().cpu().numpy() for k,v in head(hidden).items()}


def forced_block(generator, head, state, args, first_page, max_parts):
    remaining=min(16,args.generation_strokes-state.steps,
                  generator.model.cfg.max_context_strokes-max_parts-len(state.records))
    if remaining<2:return False
    # 字首/已可完字时必有多个新字分支，无需为了探测确定续笔再试排整页。
    if not state.renderer.pending_eod and (not state.policy.candidates or state.policy.terminal()):return False
    choices=state.policy.choices(state.renderer)
    if len(choices)!=1 or choices[0].action not in ('continue','control'):return False
    planner=state.policy.fork();renderer=state.renderer.fork();rows=state.records
    dummy={'shape':np.zeros(len(planner.catalog)), 'space':0.}
    count=0
    for _ in range(remaining):
        if not renderer.pending_eod and (not planner.candidates or planner.terminal()):break
        choices=planner.choices(renderer)
        if len(choices)!=1 or choices[0].action not in ('continue','control'):break
        updated=planner.append(renderer,dummy,choice_override=choices[0])
        if changed_prefix(rows,updated):break
        rows=updated;count+=1
        if renderer.complete or renderer.page_id-first_page>=8:break
    if count<2:return False
    appended=rows[len(state.records):]
    hidden=state.cache.forward(*encode_rows(generator,appended))
    predecessors=torch.cat((state.hidden[:,-1:],hidden[:,:-1]),1)
    output=head_arrays(head,predecessors)
    for i in range(count):
        row={k:v[0,i] for k,v in output.items()}
        state.records=state.policy.append(state.renderer,row,predecessors[:,i])
    state.hidden=hidden[:,-1:];state.steps+=count
    state.forced_blocks+=1;state.forced_strokes+=count
    return True


def run_shape_search(generator, head, renderer, policy, deadline, args):
    width=getattr(args,'beam_width',1)
    penalty=getattr(args,'beam_length_penalty',1.)
    if width not in (1,2,3,4) or not np.isfinite(penalty) or not 0<=penalty<=2:
        raise ValueError('beam_width须为1–4，长度归一指数须在0–2')
    first_page=renderer.page_id;max_parts=max(policy.vocab.parts.values())
    records=renderer.initial
    cache=PageKVCache(generator.model)
    state=TextState(renderer,policy,records,cache,cache.forward(*encode_rows(generator,records),prefill=True)[:,-1:])
    frontier=[state];completed=[]
    for _ in range(args.generation_strokes+max_parts+1):
        if time.monotonic()>=deadline:raise TimeoutError('读出推理已到本次时间预算')
        proposals=[]
        for state in frontier:
            state.stop=limit_reason(state,generator,args,first_page,max_parts,before=True)
            if state.stop:
                completed.append(state);continue
            if width==1 and getattr(args,'batch_forced',True) and forced_block(generator,head,state,args,first_page,max_parts):
                state.stop=limit_reason(state,generator,args,first_page,max_parts)
                if state.stop:completed.append(state)
                else:proposals.append((state.score,state,None,None))
                continue
            output={k:v[0] for k,v in head_arrays(head,state.hidden[:,-1]).items()}
            if width==1:
                proposals.append((state.score,state,None,output))
            else:
                for choice,score in state.policy.ranked_choices(state.renderer,output,state.hidden[:,-1],width):
                    proposals.append((state.score+score,state,choice,output))
        frontier=[]
        for score,parent,choice,output in sorted(proposals,key=lambda x:x[0],reverse=True)[:width]:
            state=parent if width==1 else parent.fork()
            if output is not None:
                rows=state.policy.append(state.renderer,output,state.hidden[:,-1],choice_override=choice)
                state.steps+=1;state.score=score
                # 结束后不用把最后控制笔再次推进R。
                previous=state.records;state.records=rows
                state.stop=limit_reason(state,generator,args,first_page,max_parts)
                if not state.stop:
                    state.records=previous;state.advance(generator,rows)
            if state.stop:completed.append(state)
            else:frontier.append(state)
        if not frontier:break
        completed=sorted(completed,key=lambda s:s.score/max(1,s.steps)**penalty,reverse=True)[:width]
    for state in frontier:state.stop='max_new_strokes'
    candidates=completed+frontier
    if not candidates:raise RuntimeError('没有合法文字生成分支')
    return max(candidates,key=lambda s:s.score/max(1,s.steps)**penalty)
