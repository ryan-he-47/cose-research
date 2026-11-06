"""2026-10-13：冻结R隐状态的文字笔画读出、确定性模板匹配与符号渲染。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import copy
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from src.page_document import (Document, PageLayoutConfig, StrokeElement,
                               _render_page_text, control_stroke)


@dataclass(frozen=True)
class StrokeToken:
    character: str = ''
    part: int = 0
    leading_space: bool = False
    control: str = ''


class StrokeVocabulary:
    """字符内的SVG子路径为笔画身份；空格只在字符第一笔上编码。"""

    def __init__(self, font):
        self.font = font
        self.tokens, self.parts = [], {}
        for character in sorted(c for c in font.glyphs if len(c) == 1 and 33 <= ord(c) <= 126):
            count = len(font.render(character).strokes)
            self.parts[character] = count
            for part in range(count):
                self.tokens.append(StrokeToken(character, part))
                if part == 0:
                    self.tokens.append(StrokeToken(character, part, True))
        self.tokens += [StrokeToken(control=kind) for kind in ('bos', 'eop', 'eod')]
        self.index = {token: index for index, token in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def dump(self):
        return [asdict(t) for t in self.tokens]

    @lru_cache(maxsize=512)
    def glyph(self, character, size):
        # 标准字形保持原采样；只有几何误差计算另外作等弧长采样。
        return self.font.render(character, font_size=size, origin=(0, 0)).strokes


class StrokeReadout(nn.Module):
    """输入仅为R最后一层h；hidden=0为线性头，默认一个256宽ReLU隐藏层。"""

    def __init__(self, width, classes, hidden=256):
        super().__init__()
        self.net = (nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, classes))
                    if hidden else nn.Linear(width, classes))

    def forward(self, hidden):
        return self.net(hidden)


@dataclass(frozen=True)
class TemplateFrame:
    """单步量化的已知排版条件；不向读出头输入这些字段。"""
    size: float
    line_origin: tuple[float, float]
    previous_character: str = ''
    previous_origin: tuple[float, float] = (0, 0)
    previous_advance: float = 0
    line_break: bool = True
    previous_part: int = -1


def document_trace(document: Document, vocab: StrokeVocabulary):
    """与原文档逐笔对应；标签来自字形归属，既不看λ也不猜重建形状。"""
    labels, frames = [], []
    for element in document.prefix:
        labels.append(vocab.index[StrokeToken(control=element.kind)])
        frames.append(None)
    for page in document.pages:
        placements = {g.char_index: g for g in page.metadata.get('glyph_placements', ())}
        rendered = page.metadata.get('rendered_text', '')
        size = page.metadata.get('font_size', PageLayoutConfig().font_size)
        line_origins = {}
        for g in placements.values():
            if g.character != ' ':
                line_origins.setdefault(g.line_index, g.origin)
        parts, previous, previous_part = {}, None, -1
        for element in page.elements:
            if element.kind != 'text':
                labels.append(vocab.index[StrokeToken(control=element.kind)])
                frames.append(TemplateFrame(size, line_origins[previous.line_index],
                    previous.character, previous.origin, previous.advance, False, previous_part)
                    if previous else None)
                continue
            owner = element.char_index
            g = placements[owner]
            part = parts.get(owner, 0)
            parts[owner] = part + 1
            if part == 0:
                leading = ((owner > 0 and rendered[owner-1].isspace()) or
                           (owner == 0 and bool(page.metadata.get('separator_before'))))
            else:
                leading = False
            labels.append(vocab.index[StrokeToken(g.character, part, leading)])
            frames.append(TemplateFrame(size, line_origins[g.line_index],
                previous.character if previous else '', previous.origin if previous else (0, 0),
                previous.advance if previous else 0,
                previous is None or g.line_index != previous.line_index, previous_part))
            previous = g
            previous_part = part
    if len(labels) != document.stroke_count:
        raise AssertionError('符号标签和文档笔画数未对齐')
    return np.asarray(labels, np.int64), frames


def arc_points(points, count=32):
    """仅用于误差匹配，保留笔顺方向；不修改AE输入。"""
    points = np.asarray(points, np.float64)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distance[-1] < 1e-12:
        return np.repeat(points[:1], count, axis=0).astype(np.float32)
    keep = np.r_[True, np.diff(distance) > 1e-12]
    grid = np.linspace(0, distance[-1], count)
    return np.stack([np.interp(grid, distance[keep], points[keep, dim]) for dim in range(2)], -1).astype(np.float32)


def candidate_templates(vocab: StrokeVocabulary, frame: TemplateFrame | None, count=32):
    """枚举全词表，原笔顺/绝对坐标匹配，不按真值筛选候选类别。"""
    font = vocab.font
    templates = []
    for token in vocab.tokens:
        if token.control:
            points = control_stroke(token.control)
        else:
            if frame is None:
                # 控制符目标仍枚举正文候选，使用默认首行，避免用目标kind排除竞争者。
                frame = TemplateFrame(.085, (.14, .18))
            partial = frame.previous_part + 1 < vocab.parts.get(frame.previous_character, 0)
            if token.part and partial and token.character == frame.previous_character and not frame.line_break:
                origin = np.array(frame.previous_origin)
            elif frame.line_break:
                origin = np.array(frame.line_origin)
            else:
                origin = np.array(frame.previous_origin) + [frame.previous_advance, 0]
                scale = frame.size / font.units_per_em
                if token.leading_space:
                    origin[0] += (font.glyphs[' '].advance - font.kerning.get((frame.previous_character, ' '), 0)
                                  - font.kerning.get((' ', token.character), 0)) * scale
                else:
                    origin[0] -= font.kerning.get((frame.previous_character, token.character), 0) * scale
            points = vocab.glyph(token.character, frame.size)[token.part] + origin
        templates.append(arc_points(points, count))
    return np.stack(templates)


def match_curve(points, templates, *, alias_tolerance=1e-12):
    """有序弧长MSE确定标签；同时返回同形候选集合，平票按固定词表顺序。"""
    sampled = arc_points(points, templates.shape[1])
    scores = np.mean((templates - sampled)**2, axis=(1, 2))
    choice = int(scores.argmin())
    aliases = np.flatnonzero(np.mean((templates - templates[choice])**2, axis=(1, 2)) <= alias_tolerance)
    delta = templates[choice].mean(0) - sampled.mean(0)
    position = float(np.mean(delta**2))
    return dict(prediction=choice, aliases=aliases.tolist(), canvas_mse=float(scores[choice]),
                position_mse=position, shape_mse=float(scores[choice])-position)


class SymbolicRenderer:
    """复用当前整词换行/整页对齐；只从已出现的符号渲染，EOP由读出头输出。

    新字改变当前页排版时，返回的既有坐标也可能改变。调用方必须重编码并刷新
    KV上下文；不会把未来参考文字或坐标传入这个闭环。
    """
    def __init__(self, vocab, prompt_document: Document, config=PageLayoutConfig()):
        self.vocab, self.config = vocab, config
        self.closed = []
        self.current = []
        self.text = ''
        self.pending_eod = self.complete = False
        self.invalid_parts = 0
        self.page_texts = {}
        self.page_id = prompt_document.pages[-1].page_id
        labels, _ = document_trace(prompt_document, vocab)
        sequence = prompt_document.to_sequence()
        # 打字前文的最后一页不添加EOP，其他页保持原始坐标和控制符。
        for index, (element, label) in enumerate(zip(sequence, labels)):
            if index == len(sequence)-1 and element.kind == 'eop':
                break
            record = dict(page_id=element.page_id, kind=element.kind, generated=False,
                          points=element.rawxy.tolist(), token=int(label))
            if element.page_id != self.page_id or element.kind == 'bos':
                self.closed.append(record)
            else:
                token = vocab.tokens[label]
                self._append_body(token, False)
        # 记录首次渲染对应点列，闭环前缀必须与标准输入一致。
        self.initial = self.records()

    def _append_body(self, token, generated):
        if token.control:
            self.current.append(dict(token=token, owner=None, generated=generated))
            return
        can_continue = (self.current and not self.current[-1]['token'].control and
                        self.current[-1]['token'].character == token.character and
                        self.current[-1]['token'].part + 1 == token.part)
        if token.part and not can_continue:
            self.invalid_parts += 1
        if can_continue:
            owner = self.current[-1]['owner']
        else:
            if token.leading_space and self.text:
                self.text += ' '
            owner = len(self.text)
            self.text += token.character
        self.current.append(dict(token=token, owner=owner, generated=generated))

    def records(self):
        if not self.current:
            return list(self.closed)
        by_owner = {}
        if self.text:
            rendered, _, _ = _render_page_text(self.text, self.vocab.font, self.config)
            for points, owner in zip(rendered.strokes, rendered.stroke_char_indices):
                by_owner.setdefault(owner, []).append(points)
        result = list(self.closed)
        for item in self.current:
            token = item['token']
            points = (control_stroke(token.control) if token.control else by_owner[item['owner']][token.part])
            result.append(dict(page_id=self.page_id, kind=token.control or 'text', generated=item['generated'],
                               points=points.tolist(), token=self.vocab.index[token]))
        return result

    def append(self, token_id):
        token = self.vocab.tokens[token_id]
        self._append_body(token, True)
        if token.control == 'eop':
            self.page_texts[self.page_id] = self.text
            self.closed = self.records()
            self.current, self.text = [], ''
            self.complete = self.pending_eod
            self.pending_eod = False
            if not self.complete:
                self.page_id += 1
        else:
            self.pending_eod = token.control == 'eod'
        return self.records()

    def fork(self):
        """用于字体前缀试放；字体/排版缓存共享，当前页状态独立。"""
        result=copy(self)
        result.closed=list(self.closed)
        result.current=[dict(item) for item in self.current]
        result.page_texts=dict(self.page_texts)
        return result

    def retarget_glyph(self, character):
        """共享笔画前缀分岔时改当前字的身份；调用方先核对已画曲线相同。"""
        if not self.current or self.current[-1]['owner'] is None:
            raise ValueError('没有可延迟确定身份的当前字')
        owner=self.current[-1]['owner']
        items=[item for item in self.current if item['owner']==owner]
        if [item['token'].part for item in items]!=list(range(len(items))):
            raise ValueError('当前字的笔顺前缀不完整')
        if len(items)>self.vocab.parts[character]:
            raise ValueError('已画前缀比候选字的笔画数更长')
        changed=sum(item['token'].character!=character for item in items)
        if changed:
            self.text=self.text[:owner]+character+self.text[owner+1:]
            for item in items:
                old=item['token']
                item['token']=StrokeToken(character,old.part,old.leading_space)
        return changed


@torch.no_grad()
def classification_metrics(logits, labels, vocab):
    logits, labels = logits.float(), labels.long()
    probabilities = logits.softmax(-1)
    prediction = logits.argmax(-1)
    controls = torch.tensor([bool(t.control) for t in vocab.tokens], device=labels.device)
    first = torch.tensor([not t.control and t.part == 0 for t in vocab.tokens], device=labels.device)
    identity = {key: i for i, key in enumerate(dict.fromkeys((t.character, t.part, t.control) for t in vocab.tokens))}
    identities = torch.tensor([identity[t.character,t.part,t.control] for t in vocab.tokens], device=labels.device)
    body = ~controls[labels]
    report = {}
    for name, mask in [('all', torch.ones_like(labels, dtype=torch.bool)), ('text', body),
                       ('control', ~body), ('first_stroke', first[labels]), ('continuation', body & ~first[labels])]:
        if not mask.any():
            continue
        nll = -probabilities[mask, labels[mask]].clamp_min(1e-30).log()
        report[name] = dict(count=int(mask.sum()), cross_entropy=float(nll.mean()),
            joint_error=float((prediction[mask]!=labels[mask]).float().mean()),
            identity_error=float((identities[prediction[mask]]!=identities[labels[mask]]).float().mean()))
    leading = torch.tensor([t.leading_space for t in vocab.tokens], device=labels.device)
    mask = first[labels]
    space_probability = probabilities[:, leading].sum(-1)
    truth = leading[labels]
    if mask.any():
        report['leading_space'] = dict(count=int(mask.sum()), positives=int(truth[mask].sum()),
            cross_entropy=float(nn.functional.binary_cross_entropy(space_probability[mask], truth[mask].float())),
            error=float(((space_probability[mask]>.5)!=truth[mask]).float().mean()),
            joint_argmax_flag_error=float((leading[prediction[mask]]!=truth[mask]).float().mean()))
    return report
