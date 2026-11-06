"""2026-10-14：冻结R文字读出共用的标准笔画闭环，新旧头按类型分派。"""
import time
import numpy as np
import torch
from src.page_document import Page, build_document, layout_text_pages
from .generation_pretrain import PageKVCache
from .text_readout import SymbolicRenderer
from .text_readout_policy import GlyphPrefixPolicy, audit_font_sequence


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('读出推理已到本次时间预算')


def readout_result(renderer,policy,records,prompt,prefix,constraint,refreshes,condition_refreshes,stop):
    text_pages=dict(renderer.page_texts)
    if not renderer.complete:text_pages[renderer.page_id]=renderer.text
    return dict(strokes=records,prompt=prompt,stop_reason=stop,generated_strokes=sum(r['generated'] for r in records),
                generated_page_texts=text_pages,invalid_part_transitions=renderer.invalid_parts,
                layout_refreshes=refreshes,conditioning_refreshes=condition_refreshes,
                conditioning='provided_stroke_prefix' if prefix is not None else 'typed_text',
                font_audit_scope='generated_text_only' if prefix is not None else 'full_text_sequence',
                feedback='canonical_font_to_frozen_E',future_reference_used=False,
                constraint=constraint,font_audit=audit_font_sequence(
                    [r for r in records if r['generated']] if prefix is not None else records,vocab=renderer.vocab),
                constraint_summary=policy.summary() if policy else None,
                constraint_trace=policy.trace if policy else [],
                rendering='existing_word_wrap_and_ink_alignment_on_current_symbolic_prefix')


@torch.no_grad()
def generate_text_readout(generator, head, vocab, prompt, deadline, args, layout, *, prefix=None):
    if prefix is not None:
        if not prefix or prefix[-1].kind!='eop':
            raise ValueError('图文读出前缀须截止于条件页EOP')
        prompt=''
    pages=(Page(0,'text',(),False,text=''),) if prefix is not None else layout_text_pages(
        prompt,vocab.font,config=layout,targetable=False)
    document=build_document(pages,complete=False)
    renderer=SymbolicRenderer(vocab,document,layout)
    if prefix is not None:
        # 条件页原样闭合，只在下一空白页生成文字；不读取真实回答或重排条件。
        renderer.closed=[dict(page_id=e.page_id,kind=e.kind,generated=False,points=e.rawxy.tolist()) for e in prefix]
        renderer.current=[];renderer.text='';renderer.page_texts={}
        renderer.page_id=prefix[-1].page_id+1
        renderer.initial=renderer.records()
    first_page=renderer.page_id
    constraint=getattr(args,'constraint','none')
    if constraint not in ('none','prefix'):raise ValueError('constraint须为none/prefix')
    factorized=getattr(head,'factorized',False)
    if factorized:
        from src.st.text_shape_readout import ShapePrefixPolicy, placement_ranker
        policy=ShapePrefixPolicy(vocab,head.catalog,placement_ranker(generator))
        constraint='shape_prefix'
    else:
        policy=GlyphPrefixPolicy(vocab) if constraint=='prefix' else None
    if factorized:
        from .text_search import run_shape_search
        with generator.autocast():
            state=run_shape_search(generator,head,renderer,policy,deadline,args)
        result=readout_result(state.renderer,state.policy,state.records,prompt,prefix,constraint,
                              state.refreshes,state.condition_refreshes,state.stop)
        result.update(beam_width=getattr(args,'beam_width',1),beam_length_penalty=getattr(args,'beam_length_penalty',1.),
                      search_log_probability=state.score if getattr(args,'beam_width',1)>1 else None,
                      forced_blocks=state.forced_blocks,batched_forced_strokes=state.forced_strokes)
        return result
    if getattr(args,'beam_width',1)!=1:raise ValueError('多路径文字搜索需要形状/空格读出头')
    records=renderer.initial
    cache=PageKVCache(generator.model)
    refreshes=condition_refreshes=0
    def prefill(rows):
        embeddings=generator.encode_cached([np.asarray(r['points'],np.float32) for r in rows]).unsqueeze(0)
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
            moved=[i for i,(old,new) in enumerate(zip(records,updated))
                   if old['points']!=new['points'] or old['page_id']!=new['page_id']]
            if moved:
                refreshes+=1
                condition_refreshes+=any(not records[i]['generated'] for i in moved)
            records=updated
            if renderer.complete:
                stop='eod_eop';break
            if renderer.page_id-first_page>=8:
                stop='max_new_pages';break
            if len(records)>=generator.model.cfg.max_context_strokes:
                stop='context_full';break
            if moved:
                start=min(moved)
                cache.truncate(start)
                suffix=records[start:]
                encoded=generator.encode_cached([np.asarray(r['points'],np.float32) for r in suffix]).unsqueeze(0)
                pages=torch.tensor([[r['page_id'] for r in suffix]],device=generator.device)
                hidden=cache.forward(encoded,pages)
            else:
                encoded=generator.encode_cached([np.asarray(records[-1]['points'],np.float32)]).unsqueeze(0)
                pages=torch.tensor([[records[-1]['page_id']]],device=generator.device)
                hidden=cache.forward(encoded,pages)
        else:
            stop='max_new_strokes'
    return readout_result(renderer,policy,records,prompt,prefix,constraint,refreshes,condition_refreshes,stop)
