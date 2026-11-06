"""2026-10-14：按平移无关笔画形状分类，前导空格独立二分类。"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .text_readout import arc_points
from .text_readout_policy import GlyphPrefixPolicy


class ShapeVocabulary:
    """标准字体的有向形状；保留尺度/旋转，只去平移，不改AE输入点列。"""

    def __init__(self, vocab, count=64, tolerance=1e-12,include_non_symbol=False):
        self.symbolic = vocab
        self.count, self.tolerance = count, tolerance
        self.templates, self.members = [], []
        self.identity_to_shape = {}
        for character, parts in vocab.parts.items():
            for part in range(parts):
                curve = arc_points(vocab.glyph(character, 1.)[part], count).astype(np.float64)
                curve -= curve.mean(0)
                shape = next((i for i, template in enumerate(self.templates)
                              if np.mean((template-curve)**2) <= tolerance), None)
                if shape is None:
                    shape = len(self.templates)
                    self.templates.append(curve)
                    self.members.append([])
                self.members[shape].append(dict(character=character, part=part))
                self.identity_to_shape[character, part] = shape
        self.ink_classes = len(self.templates)
        self.controls = {kind: self.ink_classes+i for i, kind in enumerate(('bos', 'eop', 'eod'))}
        self.non_symbol_class=self.ink_classes+len(self.controls) if include_non_symbol else None
        self.mapping = np.asarray([self.controls[t.control] if t.control else
                                  self.identity_to_shape[t.character, t.part] for t in vocab.tokens], np.int64)
        self.leading = np.asarray([t.leading_space for t in vocab.tokens], bool)
        self.first = np.asarray([not t.control and t.part == 0 for t in vocab.tokens], bool)
        self.body = np.asarray([not t.control for t in vocab.tokens], bool)

    def __len__(self):
        return self.ink_classes+len(self.controls)+int(self.non_symbol_class is not None)

    def dump(self):
        result=dict(ink_classes=self.ink_classes, classes=len(self), members=self.members,
                    controls=self.controls, symbolic_to_shape=self.mapping.tolist(),
                    matching='directed_arc64_centroid_translation_removed',
                    resample_points=self.count, mean_square_tolerance=self.tolerance,
                    preserves_scale_rotation_direction=True, modifies_AE_points=False)
        if self.non_symbol_class is not None:result['non_symbol_class']=self.non_symbol_class
        return result

    def targets(self, symbolic_labels, device=None):
        labels = torch.as_tensor(symbolic_labels, dtype=torch.long, device=device)
        def lookup(values):
            return torch.as_tensor(values, device=labels.device)[labels]
        return lookup(self.mapping), lookup(self.leading).float(), lookup(self.first)


class ShapeSpaceReadout(nn.Module):
    """hidden>0共享一个隐藏层；hidden=0两分支直接线性读h，不复制空格类别。"""
    factorized = True

    def __init__(self, width, shapes, hidden=256):
        super().__init__()
        # 正式checkpoint按cfg建头；字体标签表在数据加载后校验/绑定。
        self.catalog = None if isinstance(shapes,int) else shapes
        classes = shapes if isinstance(shapes,int) else len(shapes)
        self.trunk = nn.Sequential(nn.Linear(width, hidden), nn.ReLU()) if hidden else nn.Identity()
        self.shape_head = nn.Linear(hidden or width, classes)
        self.space_head = nn.Linear(hidden or width, 1)

    def forward(self, hidden):
        values = self.trunk(hidden)
        return dict(shape=self.shape_head(values), space=self.space_head(values).squeeze(-1))


def shape_space_loss(output, shape, space):
    # 用户2026-10-14明确：每笔并行判断/监督空格，后续笔及控制符目标也参与。
    shape_loss = nn.functional.cross_entropy(output['shape'].float(), shape)
    space_loss = nn.functional.binary_cross_entropy_with_logits(output['space'].float(), space)
    return shape_loss+space_loss, shape_loss, space_loss


@torch.no_grad()
def shape_space_metrics(output, symbolic_labels, catalog):
    labels = torch.as_tensor(symbolic_labels).cpu().long()
    probabilities = output['shape'].float().cpu().softmax(-1)
    space_probability = output['space'].float().cpu().sigmoid()
    return probability_metrics(probabilities, space_probability, labels, catalog)


def probability_metrics(probabilities, space_probability, labels, catalog):
    """旧245类头先合并概率到相同形状，再比较；不直接对比不同词表CE。"""
    shape, space, first = catalog.targets(labels)
    prediction = probabilities.argmax(-1)
    body = torch.as_tensor(catalog.body)[labels]
    report = {}
    for name, mask in [('all', torch.ones_like(first)), ('text', body), ('control', ~body),
                       ('first_stroke', first), ('continuation', body & ~first)]:
        p = probabilities[mask, shape[mask]].clamp_min(1e-30)
        report[name] = dict(count=int(mask.sum()), cross_entropy=float(-p.log().mean()),
                            shape_error=float((prediction[mask]!=shape[mask]).float().mean()))
    for name, mask in [('all', torch.ones_like(first)), ('first_stroke', first),
                       ('continuation', body & ~first), ('control', ~body)]:
        p, truth = space_probability[mask], space[mask].bool()
        selected = p > .5
        tp, fp = int((selected & truth).sum()), int((selected & ~truth).sum())
        fn, tn = int((~selected & truth).sum()), int((~selected & ~truth).sum())
        report['space_'+name] = dict(count=len(p), positives=int(truth.sum()),
            cross_entropy=float(nn.functional.binary_cross_entropy(p.clamp(1e-7, 1-1e-7), truth.float())),
            error=float((selected!=truth).float().mean()), true_positive=tp, false_positive=fp,
            false_negative=fn, true_negative=tn, precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn))
    return report


def remap_joint_probabilities(logits, catalog):
    joint = logits.float().cpu().softmax(-1)
    shape = joint.new_zeros((len(joint), len(catalog)))
    ids = torch.as_tensor(catalog.mapping).expand(len(joint), -1)
    shape.scatter_add_(1, ids, joint)
    space = joint[:, torch.as_tensor(catalog.leading)].sum(-1)
    return shape, space


class ShapePrefixPolicy(GlyphPrefixPolicy):
    """逐笔并行读出空格；字体合法前缀约束其应用，形状概率不按别名重复计。"""

    def __init__(self, vocab, catalog, position_ranker=None):
        super().__init__(vocab)
        self.catalog, self.position_ranker = catalog, position_ranker

    def ranked_choices(self, renderer, output, hidden, width):
        """在形状/空格分岔保留候选；同形状同空格的落点仍由原GMM选择。"""
        logits=np.asarray(output['shape'],np.float64)
        logp=logits-logits.max();logp-=np.log(np.exp(logp).sum())
        space=float(output['space'])
        groups={}
        for choice in self.choices(renderer):
            sid=int(self.catalog.mapping[choice.ids[0]])
            leading=choice.action=='start' and self.vocab.tokens[choice.ids[0]].leading_space
            groups.setdefault((sid,leading),[]).append(choice)
        scored=sorted(((float(logp[sid]-np.logaddexp(0.,-space if leading else space)), tied)
                       for (sid,leading),tied in groups.items()),key=lambda item:item[0],reverse=True)
        result=[]
        for score,tied in scored[:width]:
            choice=tied[0]
            if len(tied)>1 and self.position_ranker is not None:
                curves=[]
                for candidate in tied:
                    trial=renderer.fork()
                    if candidate.finish:trial.retarget_glyph(candidate.finish)
                    if candidate.action=='continue':trial.retarget_glyph(self.vocab.tokens[candidate.ids[0]].character)
                    curves.append(np.asarray(trial.append(candidate.ids[0])[-1]['points'],np.float32))
                choice=tied[int(np.argmax(self.position_ranker(curves,hidden)))]
            result.append((choice,score))
        return result

    def append(self, renderer, output, hidden=None, *, choice_override=None):
        logits = np.asarray(output['shape'], np.float64).reshape(-1)
        space_probability = float(torch.sigmoid(torch.as_tensor(float(output['space']))))
        leading = space_probability > .5
        choices = [c for c in self.choices(renderer)
                   if choice_override is not None or c.action != 'start' or self.vocab.tokens[c.ids[0]].leading_space == leading]
        if not choices:
            raise AssertionError('形状字体策略没有合法转移')
        shape_ids = [int(self.catalog.mapping[c.ids[0]]) for c in choices]
        for c, sid in zip(choices, shape_ids):
            if any(self.catalog.mapping[i] != sid for i in c.ids):
                raise AssertionError('同一可见分支包含不同形状')
        if choice_override is not None and choice_override not in choices:
            raise ValueError('搜索候选不属于当前合法字体前缀')
        winning = (int(self.catalog.mapping[choice_override.ids[0]]) if choice_override is not None else
                   max(set(shape_ids), key=lambda i: (logits[i], -i)))
        tied = [c for c, sid in zip(choices, shape_ids) if sid == winning]
        location_scored = False
        if choice_override is not None:
            choice=choice_override
        elif len(tied)>1 and self.position_ranker is not None:
            curves = []
            for choice in tied:
                trial = renderer.fork()
                if choice.finish:
                    trial.retarget_glyph(choice.finish)
                if choice.action == 'continue':
                    trial.retarget_glyph(self.vocab.tokens[choice.ids[0]].character)
                curves.append(np.asarray(trial.append(choice.ids[0])[-1]['points'], np.float32))
            ranks = self.position_ranker(curves, hidden)
            choice = tied[int(np.argmax(ranks))]
            location_scored = True
        else:
            choice = tied[0]
        selected = choice.ids[0]
        before = list(self.candidates)
        revised = self._resolve(renderer, choice.finish) if choice.finish else 0
        token = self.vocab.tokens[selected]
        if choice.action == 'continue':
            revised += self._resolve(renderer, token.character)
            length = self.prefix_length+1
        else:
            length = 1
        rows = renderer.append(selected)
        if token.control:
            self.candidates=(); self.prefix_length=0; self.last_scores={}
        else:
            self.candidates=tuple(self.vocab.tokens[i].character for i in choice.ids)
            self.prefix_length=length
            self.last_scores={c: 0. for c in self.candidates}
            if len(self.candidates)==1 and self.terminal():
                self.candidates=(); self.prefix_length=0
        raw = int(logits.argmax())
        probabilities = np.exp(logits-logits.max()); probabilities /= probabilities.sum()
        legal = set(shape_ids)
        self.trace.append(dict(step=len(self.trace), raw=raw, selected=selected,
            selected_shape=winning, action=choice.action, raw_legal=raw in legal,
            legal_labels=len(legal), legal_branches=len(choices),
            removed_probability=float(max(0.,1-probabilities[list(legal)].sum())),
            forced=len(legal)==1, shared_stroke=len(choice.ids)>1,
            candidates_before=before, candidates_after=list(self.candidates),
            prefix_length_after=self.prefix_length, finished_character=choice.finish,
            relabeled_strokes=revised, space_probability=space_probability,
            raw_space=leading, space_applied=token.leading_space,
            space_suppressed=leading and choice.action!='start',
            placement_tie=len(tied)>1, placement_gmm_scored=location_scored))
        return rows

    def summary(self):
        report = super().summary()
        report.update(ranking='unique_shape_logit; independent_space_threshold_0.5',
            grouped_choice_overrides=0,
            shape_mask_changes=sum(t['raw']!=t['selected_shape'] for t in self.trace),
            placement_ties=sum(t['placement_tie'] for t in self.trace),
            placement_gmm_scored=sum(t['placement_gmm_scored'] for t in self.trace),
            space_suppressed_steps=sum(t['space_suppressed'] for t in self.trace),
            placement_rule='frozen_R_GMM_density_on_standard_E_embeddings_only_within_winning_shape',
            character_logits_used=False)
        return report


def placement_ranker(generator):
    """只选已胜出形状的标准落点，不由GMM改选形状或空格；不调用D。"""
    from .model import gmm_nll

    @torch.no_grad()
    def rank(curves, hidden):
        with generator.autocast():
            candidates = generator.encode_cached(curves)
            mu, sigma, pi = generator.model.prediction_head(hidden.reshape(1, -1))
        return (-gmm_nll(candidates.float(), mu.expand(len(curves), -1, -1),
                         sigma.expand(len(curves), -1, -1), pi.expand(len(curves), -1),
                         reduction='none')).cpu().numpy()
    return rank
