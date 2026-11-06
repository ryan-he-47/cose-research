"""2026-10-27：仅评估使用的变长批量KV；沿用现有字体策略和图形回馈。"""
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from src.page_document import Page,build_document
from src.st.model import _page_pe
from src.st.model import gmm_nll
from src.st.text_readout import SymbolicRenderer
from src.st.text_shape_readout import ShapePrefixPolicy,placement_ranker
from src.st.text_search import changed_prefix
from src.st.generation_pretrain import PageCursor
from scripts.review.review_joint_stage import SnapshotDrawings
from scripts.review.review_formal_pipeline import ReviewRecipes
from src.st.formal_data import FormalDocumentDataset
from src.text_augmentation import TextAugmentationConfig


def load_dataset(root,split,font):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    info=manifest['splits'][split]
    records={}
    expected={r['file']:r['sha256'] for r in info.get('records_shards',[])}
    for name in info.get('records_files',[info.get('records_file')]):
        path=root/name
        if name in expected:
            check=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):check.update(block)
            if check.hexdigest()!=expected[name]:raise ValueError('评估原始分片hash不同')
        with path.open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if row['index'] in records:raise ValueError('评估源索引重复')
                records[row['index']]=row['record']
    if 'unique_raw_records' in info and len(records)!=info['unique_raw_records']:raise ValueError('评估原始记录数量不同')
    drawings=SnapshotDrawings(dict(raw_manifest=info['raw_manifest'],records=records))
    dataset=FormalDocumentDataset(font,(),drawings,
        font_path='datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg',
        seed=manifest['source_manifest']['seed'],split=split,augmentation=TextAugmentationConfig(enabled=False),
        object_sampling='affine_pass_v1')
    recipes={}
    with (root/info['recipes_file']).open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);recipes[row['index']]=row['recipe']
    dataset.recipes=ReviewRecipes(recipes);dataset.recipe_count=info['recipe_count']
    return dataset,info


class RaggedKV:
    def __init__(self,generator,size):
        self.generator=generator;self.model=generator.model;self.size=size
        self.entries=[];self.lengths=[0]*size;self.last=None

    @torch.no_grad()
    def forward(self,rows,starts=None):
        device=self.generator.device
        active=[i for i,r in enumerate(rows) if r]
        if not active:return self.last
        bases_full=self.lengths if starts is None else starts
        bases=[bases_full[i] for i in active]
        rows=[rows[i] for i in active];b=len(active)
        counts=[len(r) for r in rows];t=max(counts)
        active_ids=torch.tensor(active,device=device)
        lengths=[s+n for s,n in zip(bases,counts)]
        limit=self.model.cfg.max_context_strokes
        if max(lengths)>limit:raise ValueError('批量缓存超过上下文')
        flat=[row for group in rows for row in group]
        encoded=self.generator.encode_cached([np.asarray(row['points'],np.float32) for row in flat])
        ink=encoded.new_zeros((b,t,encoded.shape[-1]));pages=torch.zeros((b,t),dtype=torch.long,device=device)
        source=0;row_ids=[];time_ids=[];positions=[]
        for i,n in enumerate(counts):
            if n:
                ink[i,:n]=encoded[source:source+n];source+=n
                pages[i,:n]=torch.tensor([r['page_id'] for r in rows[i]],device=device)
                row_ids.extend([i]*n);time_ids.extend(range(n));positions.extend(range(bases[i],lengths[i]))
        row_ids=torch.tensor(row_ids,device=device);time_ids=torch.tensor(time_ids,device=device)
        positions=torch.tensor(positions,device=device)
        r=self.model.r_pred;pe=_page_pe(pages,r.pe_dim).to(ink.dtype)
        x=r.tok_proj(torch.cat((ink,pe),-1))
        prefill=not self.entries
        if prefill:
            keep=None
        else:
            end=max(lengths)
            queries=torch.tensor(bases,device=device)[:,None]+torch.arange(t,device=device)[None,:]
            keys=torch.arange(end,device=device)[None,None,:]
            keep=((keys<=queries[:,:,None])&(keys<torch.tensor(lengths,device=device)[:,None,None]))[:,None]
        for index,layer in enumerate(r.layers):
            a=layer.mha
            if a.rope_theta is not None:raise ValueError('仅支持现有page PE')
            split=lambda z:z.view(b,t,a.num_heads,a.depth).transpose(1,2)
            q,k,v=(split(proj(x)) for proj in (a.wq,a.wk,a.wv))
            if index==len(self.entries):
                shape=(self.size,a.num_heads,limit,a.depth)
                # 变长行的padding槽也必须有限；masked attention不能依赖未初始化值。
                self.entries.append((k.new_zeros(shape),v.new_zeros(shape)))
            cache_k,cache_v=self.entries[index]
            cache_k[active_ids[row_ids],:,positions]=k[row_ids,:,time_ids]
            cache_v[active_ids[row_ids],:,positions]=v[row_ids,:,time_ids]
            keys,values=cache_k[:,:,:max(lengths)],cache_v[:,:,:max(lengths)]
            if b!=self.size:keys,values=keys[active_ids],values[active_ids]
            attended=F.scaled_dot_product_attention(q,keys,values,
                attn_mask=keep,is_causal=prefill)
            output=a.dense(attended.transpose(1,2).contiguous().view(b,t,-1))
            x=layer.ln1(x+layer.drop(output));x=layer.ln2(x+layer.drop(layer.ffn(x)))
        last=x[torch.arange(b,device=device),torch.tensor([max(0,n-1) for n in counts],device=device)]
        if self.last is None:self.last=last.new_zeros((self.size,last.shape[-1]))
        self.last[active_ids]=last
        for i,length in zip(active,lengths):self.lengths[i]=length
        return self.last


def prefix_records(prefix):
    return [dict(page_id=e.page_id,kind=e.kind,generated=False,points=e.rawxy.tolist()) for e in prefix]


def batch_choices(generator,policies,renderers,output,hidden,stops):
    chosen={};ties={};curves=[];owners=[]
    for i,(policy,renderer) in enumerate(zip(policies,renderers)):
        if stops[i]:continue
        leading=float(output['space'][i])>0
        choices=[c for c in policy.choices(renderer) if c.action!='start' or policy.vocab.tokens[c.ids[0]].leading_space==leading]
        ids=[int(policy.catalog.mapping[c.ids[0]]) for c in choices]
        winning=max(set(ids),key=lambda k:(float(output['shape'][i,k]),-k))
        tied=[c for c,k in zip(choices,ids) if k==winning]
        if len(tied)==1:chosen[i]=tied[0];continue
        begin=len(curves)
        for choice in tied:
            trial=renderer.fork()
            if choice.finish:trial.retarget_glyph(choice.finish)
            if choice.action=='continue':trial.retarget_glyph(policy.vocab.tokens[choice.ids[0]].character)
            curves.append(np.asarray(trial.append(choice.ids[0])[-1]['points'],np.float32));owners.append(i)
        ties[i]=(begin,len(curves),tied)
    if curves:
        candidates=generator.encode_cached(curves)
        mu,sigma,pi=generator.model.prediction_head(hidden)
        owner=torch.tensor(owners,device=hidden.device)
        ranks=(-gmm_nll(candidates.float(),mu[owner],sigma[owner],pi[owner],reduction='none')).cpu().numpy()
        for i,(start,end,choices) in ties.items():chosen[i]=choices[int(np.argmax(ranks[start:end]))]
    return chosen


@torch.no_grad()
def text_batch(generator,head,vocab,layout,prefixes,*,limit=96):
    renderers=[];policies=[];records=[]
    base_policy=ShapePrefixPolicy(vocab,head.catalog,placement_ranker(generator))
    for prefix in prefixes:
        renderer=SymbolicRenderer(vocab,build_document((Page(0,'text',(),False,text=''),),complete=False),layout)
        renderer.closed=prefix_records(prefix);renderer.current=[];renderer.text='';renderer.page_texts={}
        renderer.page_id=prefix[-1].page_id+1;renderer.initial=renderer.records()
        renderers.append(renderer);policies.append(base_policy.fork());records.append(renderer.initial)
    first=[r.page_id for r in renderers];stops=[None]*len(prefixes)
    cache=RaggedKV(generator,len(prefixes));max_parts=max(vocab.parts.values())
    with generator.autocast():
        hidden=cache.forward(records)
        for step in range(limit+max_parts+1):
            output={k:v.float().cpu().numpy() for k,v in head(hidden).items()}
            suffixes=[[] for _ in prefixes];starts=list(cache.lengths)
            for i,(renderer,policy) in enumerate(zip(renderers,policies)):
                if stops[i]:continue
                if step>=limit or len(records[i])>=generator.model.cfg.max_context_strokes-max_parts:
                    finished=policy.finish_at_limit(renderer)
                    if finished is not None:
                        records[i]=finished;stops[i]='max_new_strokes' if step>=limit else 'context_full';continue
            choices=batch_choices(generator,policies,renderers,output,hidden,stops)
            for i,(renderer,policy) in enumerate(zip(renderers,policies)):
                if stops[i]:continue
                row={k:v[i] for k,v in output.items()}
                updated=policy.append(renderer,row,hidden[i:i+1],choice_override=choices[i])
                reason=('eod_eop' if renderer.complete else 'max_new_pages' if renderer.page_id-first[i]>=8
                        else 'context_full' if len(updated)>=generator.model.cfg.max_context_strokes else None)
                if reason:records[i]=updated;stops[i]=reason;continue
                moved=changed_prefix(records[i],updated)
                start=min(moved) if moved else len(records[i]);starts[i]=start
                suffixes[i]=updated[start:];records[i]=updated
            if all(stops):break
            hidden=cache.forward(suffixes,starts)
    results=[]
    for renderer,rows,stop in zip(renderers,records,stops):
        texts=dict(renderer.page_texts)
        if not renderer.complete:texts[renderer.page_id]=renderer.text
        results.append(dict(generated=' '.join(texts.values()).strip(),generated_page_texts=texts,
            stop_reason=stop or 'max_new_strokes',strokes=rows,
            generated_strokes=sum(r['generated'] for r in rows)))
    return results


@torch.no_grad()
def graphic_batch(generator,prefixes,*,limit=24,max_pages=2):
    records=[prefix_records(p) for p in prefixes]
    cursors=[PageCursor(p[-1].page_id) for p in prefixes]
    for c,p in zip(cursors,prefixes):c.append(p[-1].kind)
    first=[c.page_id for c in cursors];stops=[None]*len(prefixes)
    cache=RaggedKV(generator,len(prefixes))
    with generator.autocast():
        hidden=cache.forward(records)
        for step in range(limit):
            mu,sigma,pi=generator.model.prediction_head(hidden)
            latent=mu[torch.arange(len(prefixes),device=generator.device),pi.argmax(-1)]
            curves=generator.decode_tensor(latent).cpu().numpy()
            suffixes=[[] for _ in prefixes]
            for i,(curve,cursor) in enumerate(zip(curves,cursors)):
                if stops[i]:continue
                if not np.isfinite(curve).all():stops[i]='nonfinite_coordinates';continue
                if cursor.complete:stops[i]='eod_eop';continue
                if cursor.page_id-first[i]>=max_pages:stops[i]='max_new_pages';continue
                kind,_=generator.recognize(curve)
                row=dict(page_id=cursor.page_id,kind=kind,generated=True,points=curve.tolist())
                records[i].append(row);cursor.append(kind)
                if cursor.complete:stops[i]='eod_eop';continue
                if len(records[i])>=generator.model.cfg.max_context_strokes:stops[i]='context_full';continue
                suffixes[i]=[row]
            if all(stops):break
            hidden=cache.forward(suffixes)
    return [dict(strokes=rows,stop_reason=stop or 'max_new_strokes') for rows,stop in zip(records,stops)]
