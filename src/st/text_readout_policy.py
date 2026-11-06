"""2026-10-13：字体笔顺约束；共享可见前缀的字符延迟到分岔时确定。"""
from __future__ import annotations

from dataclasses import dataclass
from copy import copy
import numpy as np
from .text_readout import StrokeToken, arc_points


def visible_signature(renderer, records):
    # 只比较当前页。原点、点数、笔顺与FP32坐标都相同才合并候选；不是近似字形纠错。
    return tuple((len(row['points']),np.asarray(row['points'],np.float32).tobytes())
                 for row in records[len(renderer.closed):])


@dataclass
class StrokeChoice:
    ids: tuple[int, ...]
    action: str
    finish: str | None = None


class GlyphPrefixPolicy:
    """有限字体前缀：没有外部词典；仅约束完整字形、BOS和EOD/EOP协议。"""

    def __init__(self,vocab):
        self.vocab=vocab
        self.candidates=()
        self.prefix_length=0
        self.last_scores={}
        self.trace=[]
        chars=list(vocab.parts)
        first=np.stack([arc_points(vocab.glyph(c,1.)[0]) for c in chars])
        first=first-first[:,:1]
        used=set();self.first_groups=[]
        for index,char in enumerate(chars):
            if char in used:continue
            group=tuple(chars[i] for i in np.where(np.mean((first-first[index])**2,axis=(1,2))<=1e-12)[0])
            used.update(group);self.first_groups.append(group)
        self.identity_revisions=0
        self.cap_finalizations=0
        self._choice_cache=None

    def terminal(self):
        return [c for c in self.candidates if self.vocab.parts[c]==self.prefix_length]

    def fork(self):
        result=copy(self)
        result.trace=list(self.trace)
        result.last_scores=dict(self.last_scores)
        return result

    def terminal_character(self):
        return max(self.terminal(),key=lambda c:(self.last_scores[c],-ord(c)))

    def _trial(self,renderer,token_id,action,finish):
        trial=renderer.fork()
        if finish:trial.retarget_glyph(finish)
        if action=='continue':trial.retarget_glyph(self.vocab.tokens[token_id].character)
        rows=trial.append(token_id)
        return visible_signature(trial,rows)

    def _partition(self,renderer,ids,action,finish=None):
        if len(ids)==1:return [StrokeChoice(tuple(ids),action,finish)]
        groups={}
        for index in ids:
            key=self._trial(renderer,index,action,finish)
            groups.setdefault(key,[]).append(index)
        return [StrokeChoice(tuple(group),action,finish) for group in groups.values()]

    def choices(self,renderer):
        # 同一步的确定续笔规划和实际读出共享合法分支，避免重复试排整页。
        key=(id(renderer.vocab),id(renderer.config),renderer.page_id,renderer.text,
             renderer.pending_eod,len(renderer.closed),
             tuple((item['token'],item['owner']) for item in renderer.current),
             self.candidates,self.prefix_length,tuple(sorted(self.last_scores.items())))
        if self._choice_cache is not None and self._choice_cache[0]==key:
            return list(self._choice_cache[1])
        result=self._choices(renderer)
        self._choice_cache=(key,tuple(result))
        return result

    def _choices(self,renderer):
        v=self.vocab
        if renderer.pending_eod:
            return [StrokeChoice((v.index[StrokeToken(control='eop')],),'control')]
        choices=[]
        if self.candidates:
            ids=[v.index[StrokeToken(c,self.prefix_length)] for c in self.candidates
                 if v.parts[c]>self.prefix_length]
            if ids:choices+=self._partition(renderer,ids,'continue')
            if not self.terminal():return choices
            finish=self.terminal_character()
        else:finish=None
        for leading in [False,True]:
            for group in self.first_groups:
                ids=[v.index[StrokeToken(c,0,leading)] for c in group]
                choices+=self._partition(renderer,ids,'start',finish)
        if any(not item['token'].control for item in renderer.current):
            choices += [StrokeChoice((v.index[StrokeToken(control=kind)],),'control',finish)
                        for kind in ['eop','eod']]
        return choices

    def _resolve(self,renderer,character):
        before=visible_signature(renderer,renderer.records())
        changed=renderer.retarget_glyph(character)
        if visible_signature(renderer,renderer.records())!=before:
            raise AssertionError('延迟身份改写移动了已画前缀')
        self.identity_revisions+=bool(changed)
        return changed

    def append(self,renderer,logits):
        logits=np.asarray(logits,np.float64).reshape(-1)
        choices=self.choices(renderer)
        if not choices:raise AssertionError('字体约束没有合法转移')
        def score(choice):
            values=logits[list(choice.ids)];maximum=values.max()
            return float(maximum+np.log(np.exp(values-maximum).sum()))
        choice=max(choices,key=score)
        selected=max(choice.ids,key=lambda i:(logits[i],-i))
        raw=int(logits.argmax())
        legal=sorted({index for option in choices for index in option.ids})
        probability=np.exp(logits-logits.max());probability/=probability.sum()
        before=list(self.candidates)
        revised=0
        if choice.finish:revised+=self._resolve(renderer,choice.finish)
        token=self.vocab.tokens[selected]
        if choice.action=='continue':
            revised+=self._resolve(renderer,token.character)
            length=self.prefix_length+1
        else:length=1
        rows=renderer.append(selected)
        if token.control:
            self.candidates=();self.prefix_length=0;self.last_scores={}
        else:
            self.candidates=tuple(self.vocab.tokens[i].character for i in choice.ids)
            self.prefix_length=length
            self.last_scores={self.vocab.tokens[i].character:float(logits[i]) for i in choice.ids}
            if len(self.candidates)==1 and self.terminal():
                self.candidates=();self.prefix_length=0
        self.trace.append(dict(step=len(self.trace),raw=raw,selected=selected,action=choice.action,
            raw_legal=raw in legal,legal_labels=len(legal),legal_branches=len(choices),
            removed_probability=float(max(0.,1-probability[legal].sum())),
            forced=len(choices)==1,shared_stroke=len(choice.ids)>1,
            candidates_before=before,candidates_after=list(self.candidates),
            prefix_length_after=self.prefix_length,finished_character=choice.finish,
            relabeled_strokes=revised))
        return rows

    def finish_at_limit(self,renderer):
        """停止在完整字形；若候选中有短字终点即可定字，否则继续完成当前字。"""
        if renderer.pending_eod:return None
        if not self.candidates:return renderer.records()
        if not self.terminal():return None
        self._resolve(renderer,self.terminal_character())
        self.candidates=();self.prefix_length=0;self.cap_finalizations+=1
        return renderer.records()

    def summary(self):
        return dict(steps=len(self.trace),forced_steps=sum(t['forced'] for t in self.trace),
            raw_illegal_steps=sum(not t['raw_legal'] for t in self.trace),
            shared_stroke_steps=sum(t['shared_stroke'] for t in self.trace),
            ambiguous_starts=sum(t['shared_stroke'] and t['action']=='start' for t in self.trace),
            grouped_choice_overrides=sum(t['raw_legal'] and t['raw']!=t['selected'] for t in self.trace),
            mean_removed_probability=float(np.mean([t['removed_probability'] for t in self.trace])) if self.trace else 0.,
            identity_revisions=self.identity_revisions,cap_finalizations=self.cap_finalizations,
            first_shape_groups=[list(g) for g in self.first_groups if len(g)>1],
            prefix_equality='identical_current_page_float32_points_and_lengths',
            ranking='logsumexp_of_legal_labels_for_same_visible_stroke',uses_dictionary=False)


def audit_font_sequence(records,vocab):
    """独立检查最终字形身份：也数提前中断字符，不只数跳到后续笔。"""
    active=None;next_part=0;pending_eod=False;page_has_ink=False;document_complete=False
    violations=[];completed=0
    for index,row in enumerate(records):
        token=vocab.tokens[row['token']]
        if token.control=='bos' and index==0:continue
        if document_complete:
            violations.append(dict(index=index,reason='after_document_end'))
        if pending_eod and token.control!='eop':
            violations.append(dict(index=index,reason='eod_not_followed_by_eop'))
        if active and (token.control or token.character!=active or token.part!=next_part):
            violations.append(dict(index=index,reason='interrupted_glyph',expected_character=active,expected_part=next_part))
            active=None;next_part=0
        if token.control:
            if token.control=='bos':violations.append(dict(index=index,reason='generated_bos'))
            if token.control in ('eop','eod') and not page_has_ink:
                violations.append(dict(index=index,reason='empty_page_control'))
            if token.control=='eop':
                document_complete=pending_eod
                page_has_ink=False
            pending_eod=token.control=='eod'
            continue
        pending_eod=False;page_has_ink=True
        if active is None and token.part:
            violations.append(dict(index=index,reason='missing_glyph_prefix'))
        active=token.character;next_part=token.part+1
        if next_part==vocab.parts[active]:active=None;next_part=0;completed+=1
    return dict(violations=len(violations),violating_steps=len({v['index'] for v in violations}),
                details=violations,completed_glyphs=completed,
                unfinished_character=active,pending_eod=pending_eod,document_complete=document_complete)
