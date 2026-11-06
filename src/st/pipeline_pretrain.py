"""2026-10-02：按页的文本/对象预训练数据，文字增强在每次实际消费时重采样。"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.page_document import Document, PageLayoutConfig, build_document, layout_text_pages, sketch_page
from src.text_augmentation import TextAugmentationConfig, augment_rendered_text
from src.text_normalization import NORMALIZATION_VERSION, normalize_text
from src.text_renderer import VectorFont
from .ae_data import QuickDrawRawSource, cap_stroke, source_in_split, validate_source_split


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FONT = ROOT / "datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg"
DEFAULT_STORIES = ROOT / "datasets/datasets--roneneldan--TinyStories/snapshots"
KIND_IDS = {"text": 0, "sketch": 1, "selection": 2, "bos": 3, "eop": 4, "eod": 5}


@dataclass(frozen=True)
class Story:
    text: str
    source_id: str


def load_stories(path: str | Path, limit: int = 128) -> tuple[list[Story], dict]:
    """只保留有限首阶段文本池；Parquet 按小 batch 读取，异常字符留在隔离报告。"""
    if limit < 0:
        raise ValueError("故事数量不能为负")
    path = Path(path)
    report = {"version": NORMALIZATION_VERSION, "accepted": 0, "changed": 0,
              "duplicates": 0, "repairs": {}, "quarantined": []}
    seen, stories = set(), []

    def accept(raw, source):
        result = normalize_text(raw)
        for kind, count in result.change_counts.items():
            report["repairs"][kind] = report["repairs"].get(kind, 0) + count
        if result.unsupported:
            report["quarantined"].append({"source": source,
                "characters": list(result.unsupported),
                "original_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest()})
            return
        text = result.text.strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not text or digest in seen:
            report["duplicates"] += 1
            return
        seen.add(digest)
        stories.append(Story(text, source))
        report["changed"] += int(text != raw)

    if limit == 0:
        return [], report
    if path.suffix.lower() in (".txt", ".jsonl", ".ndjson"):
        if path.suffix.lower() == ".txt":
            records = path.read_text(encoding="utf-8").split("\n\n\n")
        else:
            with path.open(encoding="utf-8") as stream:
                for index, line in enumerate(stream):
                    if line.strip():
                        accept(json.loads(line)["text"], f"{path.name}:{index}")
                    if len(stories) >= limit:
                        break
            report["accepted"] = len(stories)
            return stories, report
        for index, raw in enumerate(records):
            accept(raw, f"{path.name}:{index}")
            if len(stories) >= limit:
                break
    else:
        import pyarrow.parquet as pq
        files = [path] if path.is_file() else sorted(path.rglob("train-*.parquet"))
        if not files:
            raise FileNotFoundError(f"未找到 TinyStories train parquet：{path}")
        done = False
        for file in files:
            row = 0
            for batch in pq.ParquetFile(file).iter_batches(batch_size=256, columns=["text"]):
                for raw in batch.column(0).to_pylist():
                    accept(raw, f"{file.name}:{row}")
                    row += 1
                    if len(stories) >= limit:
                        done = True
                        break
                if done:
                    break
            if done:
                break
    report["accepted"] = len(stories)
    return stories, report


class PretrainDocumentDataset:
    """索引可直接恢复；global_index 的高位表示重复轮次，几何每轮重新采样。

    AE 按任务交错消费，R 可按有效目标量选任务。相同索引可复现；后续索引
    的几何重新采样。保留所有页，R 再按上下文取窗口。
    """

    def __init__(self, font: VectorFont, stories: Sequence[Story], drawings=None, *,
                 seed: int = 42, augment_text: bool = True,
                 augmentation: TextAugmentationConfig | None = None,
                 normalization_report: dict | None = None, include_scenes: bool = True,
                 split: str = "all", validation_fraction: float = .02):
        validate_source_split(split, validation_fraction)
        self.font, self.stories, self.drawings = font, tuple(stories), drawings
        self.seed = seed
        self.split, self.validation_fraction = split, validation_fraction
        self.augmentation = augmentation or TextAugmentationConfig(enabled=augment_text)
        self.normalization_report = normalization_report or {}
        self.tasks = (("T1",) if self.stories else ()) + (("O1", "O2") if drawings else ())
        if include_scenes and drawings and len(drawings.class_indices) >= 2:
            self.tasks += ("S1", "S2", "S3", "S4", "S5")
        if not self.tasks:
            raise ValueError("文档源为空，至少提供故事或 QuickDraw")

    def __len__(self):
        return max(len(self.stories), len(self.drawings) if self.drawings else 0) * len(self.tasks)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, global_index: int) -> Document:
        if global_index < 0:
            raise IndexError("global_index 必须非负")
        task = self.tasks[global_index % len(self.tasks)]
        document = self.get_for_task(task, global_index // len(self.tasks))
        document.metadata["global_index"] = global_index
        return document

    def get_for_task(self, task: str, index: int) -> Document:
        """任务内独立游标；R 的目标预算采样器和 AE 交错流共用同一生成函数。"""
        if task not in self.tasks or index < 0:
            raise ValueError(f"无效任务/索引：{task}/{index}")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.tasks.index(task), index]))
        augment = partial(augment_rendered_text, rng=rng, config=self.augmentation)

        def text_pages(text, target):
            return layout_text_pages(text, self.font, targetable=target,
                                     augment=augment if self.augmentation.enabled else None)

        metadata = {"task_index": index, "task": task,
                    "text_augmentation": self.augmentation.enabled}
        if task == "T1":
            story = self.stories[index % len(self.stories)]
            metadata.update(task="T1", source_id=story.source_id, text=story.text)
            pages = text_pages(story.text, True)
        else:
            if task.startswith("S"):
                from src.scene_tasks import build_scene_document
                count = int(rng.integers(2, min(4, len(self.drawings.class_indices)) + 1))
                classes = rng.choice(len(self.drawings.class_indices), count, replace=False)
                drawings = [self.drawings.get(int(rng.choice(self.drawings.class_indices[int(cls)])))
                            for cls in classes]
                document = build_scene_document(task, drawings, self.font, rng=rng,
                    augment=augment if self.augmentation.enabled else None)
                document.metadata.update(metadata)
                return document
            cls = index % len(self.drawings.class_indices)
            entries = self.drawings.class_indices[cls]
            drawing = self.drawings.get(entries[(index // len(self.drawings.class_indices)) % len(entries)])
            label = normalize_text(drawing.label).text
            metadata.update(source_id=drawing.key_id, label=label)
            if task == "O1":
                metadata["task"] = "O1"
                pages = (*text_pages(f"Draw: {label}", False),
                         sketch_page(drawing.strokes, targetable=True))
            else:
                metadata["task"] = "O2"
                pages = (*text_pages("Name this sketch:", False),
                         sketch_page(drawing.strokes, targetable=False),
                         *text_pages(label, True))
        return build_document(pages, complete=True, metadata=metadata)

    def manifest(self):
        from dataclasses import asdict
        glyphs = [(key, asdict(glyph)) for key, glyph in sorted(self.font.glyphs.items())]
        font_spec = {"name": self.font.name, "units_per_em": self.font.units_per_em,
                     "ascent": self.font.ascent, "descent": self.font.descent, "glyphs": glyphs,
                     "kerning": sorted((a, b, value) for (a, b), value in self.font.kerning.items())}
        digest = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                      sort_keys=True).encode("utf-8")).hexdigest()
        manifest = {"seed": self.seed, "documents_per_pass": len(self),
                "format_version": 2, "tasks": list(self.tasks),
                "geometry_version": "2026-10-03-glyph-v2",
                "page_layout": asdict(PageLayoutConfig()),
                "stories": len(self.stories),
                "stories_sha256": digest([(story.source_id, story.text) for story in self.stories]),
                "drawings": len(self.drawings) if self.drawings is not None else 0,
                "raw_source": self.drawings.manifest() if self.drawings else None,
                "classes": sorted(getattr(self.drawings, "classes", [])),
                "font": self.font.name, "font_sha256": digest(font_spec),
                "augmentation": asdict(self.augmentation),
                "normalization": self.normalization_report}
        if self.split != "all":
            manifest.update(split=self.split, validation_fraction=self.validation_fraction)
        return manifest


def build_document_dataset(*, font_path=DEFAULT_FONT, raw_dir=None,
                           stories_path=None, text_file=None, max_stories=128,
                           max_per_class=64, seed=42, augment_text=True,
                           augmentation=None, include_scenes=True, split="all",
                           validation_fraction=.02, formal=False, story_index=None,
                           raw_index=None, scene_index=None, scene_recipes=1000000,
                           object_sampling='with_replacement', paired_scene_share=0.) -> PretrainDocumentDataset:
    """先建立有限源池再划分；all 保持已有冒烟的数据与 manifest（2026-10-03）。"""
    validate_source_split(split, validation_fraction)
    if formal:
        from .formal_data import build_formal_dataset
        return build_formal_dataset(font_path=font_path, raw_dir=raw_dir, story_index=story_index,
            raw_index=raw_index, scene_index=scene_index, scene_recipes=scene_recipes,
            seed=seed, split=split, augmentation=augmentation, augment_text=augment_text,
            validation_fraction=validation_fraction,object_sampling=object_sampling,paired_scene_share=paired_scene_share)
    if isinstance(augmentation, dict):
        augmentation = TextAugmentationConfig(**augmentation)
    font = VectorFont.from_svg(font_path)
    path = text_file or stories_path or DEFAULT_STORIES
    stories, report = load_stories(path, max_stories) if max_stories else ([], {})
    # 规范化/去重后的完整故事是划分单位，随后所有分页和增强继承此划分。
    stories = [story for story in stories
               if source_in_split(story.text, split, validation_fraction)]
    drawings = QuickDrawRawSource(raw_dir, max_per_class=max_per_class, split=split,
                                 validation_fraction=validation_fraction) if raw_dir else None
    if not stories and drawings is None and split != "all":
        raise ValueError(f"文本 split={split} 在前 {max_stories} 篇基础池中为空；"
                         "增大 max_stories 或调整 validation_fraction")
    return PretrainDocumentDataset(font, stories, drawings, seed=seed,
        augment_text=augment_text, augmentation=augmentation, normalization_report=report,
        include_scenes=include_scenes, split=split, validation_fraction=validation_fraction)


def collate_documents(documents: Sequence[Document], *, max_strokes=1024,
                      max_points=76, encoder_length_buckets=()) -> dict:
    """仅为有效笔画分配 E 行；R 按序列右 padding，坐标先增强再固定映射。"""
    sequences = [document.to_sequence() for document in documents]
    if not sequences or any(len(seq) < 2 for seq in sequences):
        raise ValueError("R batch 至少包含两笔的文档")
    size = max(map(len, sequences))
    if size > max_strokes:
        raise ValueError(f"{size} 笔超过上下文 {max_strokes}；先取 Document 窗口")
    batch = len(sequences)
    pad = np.ones((batch, size), dtype=bool)
    sid = np.zeros((batch, size), dtype=np.int64)
    target = np.zeros((batch, size), dtype=np.float32)
    kinds = np.zeros((batch, size), dtype=np.int64)
    start = np.zeros((batch, size, 2), dtype=np.float32)
    points, owners = [], []
    for b, sequence in enumerate(sequences):
        for s, element in enumerate(sequence):
            raw = cap_stroke(element.rawxy, max_points=max_points)
            points.append((raw - np.array([0.5, 0.6], np.float32)) / 0.6)
            owners.append(b * size + s)
            pad[b, s] = False
            sid[b, s] = element.page_id
            target[b, s] = element.targetable
            kinds[b, s] = KIND_IDS[element.kind]
            start[b, s] = raw[0]
    lengths = np.array(list(map(len, points)), dtype=np.int64)
    groups = []
    if encoder_length_buckets:
        bounds = np.asarray(encoder_length_buckets, dtype=np.int64)
        if bounds.ndim != 1 or np.any(bounds <= 0) or np.any(np.diff(bounds) <= 0):
            raise ValueError('E点长分组边界须为递增正整数')
        bins = np.searchsorted(bounds, lengths)
        order = np.argsort(bins, kind='stable')
        points = [points[i] for i in order]
        owners = [owners[i] for i in order]
        lengths, bins = lengths[order], bins[order]
        begin = 0
        for end in list(np.flatnonzero(np.diff(bins)) + 1) + [len(points)]:
            groups.append((int(begin), int(end), int(lengths[begin:end].max())))
            begin = end
    ink = np.zeros((len(points), int(lengths.max()), 2), dtype=np.float32)
    for row, pts in enumerate(points):
        ink[row, :len(pts)] = pts
    tensors = {"ink": ink, "stroke_len": lengths,
               "stroke_batch_index": np.array(owners, dtype=np.int64),
               "stroke_pad": pad, "slice_id": sid, "targetable": target,
               "kind_id": kinds, "start_coord": start}
    if any('_text_aux_targets' in doc.metadata for doc in documents):
        for key,dtype,fill in [('shape',np.int64,0),('space',np.float32,0),
                               ('first',bool,False),('valid',bool,False)]:
            labels=np.full((batch,size),fill,dtype=dtype)
            for b,(doc,sequence) in enumerate(zip(documents,sequences)):
                data=doc.metadata.get('_text_aux_targets')
                if data is None or len(data[key])!=len(sequence):
                    raise ValueError('辅助标签缺失或与文档长度不同')
                labels[b,:len(sequence)]=data[key]
            tensors['text_aux_'+key]=labels
    if any('_learning_weights' in doc.metadata for doc in documents):
        for key in ('shape','space'):
            weights=np.zeros((batch,size),np.float32)
            for b,doc in enumerate(documents):
                if '_learning_weights' not in doc.metadata:raise ValueError('全批文字权重缺失')
                weights[b,:len(sequences[b])]=doc.metadata['_learning_weights'][key]
            tensors['learning_'+key]=weights
    indices = np.flatnonzero(((~pad[:, :-1]) & (~pad[:, 1:]) & (target[:, 1:] > .5)).ravel())
    tensors['target_index'] = indices
    tensors['target_kind'] = kinds[:, 1:].ravel()[indices]
    if 'text_aux_valid' in tensors:
        valid = np.flatnonzero(tensors['text_aux_valid'][:, 1:].ravel()[indices])
        tensors['aux_target_index'] = valid
        for key in ('shape', 'space', 'first'):
            tensors['aux_target_' + key] = tensors['text_aux_' + key][:, 1:].ravel()[indices][valid]
    result = {**{key: torch.from_numpy(value) for key, value in tensors.items()},
            'encode_groups': tuple(groups),
            "metadata": [document.metadata for document in documents]}
    if any('_grounding_targets' in doc.metadata for doc in documents):
        result['ground_labels']=[doc.metadata['_grounding_targets'] for doc in documents]
    return result


def to_device(batch: dict, device) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}
