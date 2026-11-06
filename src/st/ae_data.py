"""新分布 AE 数据：RAW 小索引与逐笔流，不缓存神经网络嵌入（2026-10-02）。

页坐标/文字增强由 page_document 和 pipeline_pretrain 产生。本模块只把最终
可见笔画映射到既有统一 E 的坐标系，再从同一轨迹产生 τ 与 DCT 目标。
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .config_st import STConfig
from .dct import compute_dct_for_strokes


@dataclass(frozen=True)
class RawDrawing:
    strokes: tuple[np.ndarray, ...]
    label: str
    key_id: str


def validate_source_split(split: str, validation_fraction: float):
    if split not in ("all", "train", "valid"):
        raise ValueError("split 必须为 all、train 或 valid")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction 必须在 0 与 1 之间")


def source_in_split(source_key: str, split: str, validation_fraction: float) -> bool:
    """2026-10-03：按源身份稳定划分；几何增强和场景合成不参与划分。"""
    if split == "all":
        return True
    digest = hashlib.sha256(source_key.encode("utf-8")).digest()
    is_valid = int.from_bytes(digest[:8], "big") < int(validation_fraction * (1 << 64))
    return is_valid if split == "valid" else not is_valid


class QuickDrawRawSource:
    """索引每类前 max_per_class 条 recognized RAW，坐标按需读取。

    索引只保存文件偏移；不会把现有 650k object-array NPZ 解压到 RAM。
    max_per_class=None 显式表示扫描全部输入。get 的顺序固定，可供数据源
    以全局样本索引复现；sample 额外提供类别均衡随机取图。train/valid
    在限定的基础池内按 key_id 划分，合成场景只能读取所属划分的对象。
    """

    def __init__(self, root: str | Path, max_per_class: int | None = 256,
                 *, files: Iterable[str | Path] | None = None, split: str = "all",
                 validation_fraction: float = .02):
        validate_source_split(split, validation_fraction)
        self.split, self.validation_fraction = split, validation_fraction
        root = Path(root)
        # 2026-10-03：按文件名字符串固定顺序；WindowsPath 会忽略大小写，
        # 导致 The_* 三类与 Linux 的顺序不同，影响随机取样和跨机器恢复。
        candidates = (Path(p) for p in files) if files else root.glob("full_raw_*.ndjson")
        self.files = tuple(sorted(candidates, key=lambda p: (p.name, p.as_posix())))
        if not self.files:
            raise FileNotFoundError(f"没有 QuickDraw RAW 文件：{root}")
        if max_per_class is not None and max_per_class <= 0:
            raise ValueError("max_per_class 必须为正数或 None")
        self.max_per_class = max_per_class
        self.indices: list[tuple[int, int]] = []
        self.class_indices: list[list[int]] = []
        self.classes: list[str] = []
        self.indexed_files: list[dict] = []
        self.malformed_lines = 0
        for file_index, path in enumerate(self.files):
            entries = []
            base_count = 0
            accepted_hash = hashlib.sha256()
            with path.open("rb") as stream:
                # 先限定每类基础池，再划分；valid 不为凑数量继续扫全类。
                while max_per_class is None or base_count < max_per_class:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeError):
                        self.malformed_lines += 1
                        continue
                    if not record.get("recognized", False) or not record.get("drawing"):
                        continue
                    base_count += 1
                    key_id = str(record.get("key_id", f"{path.name}:{offset}"))
                    if not source_in_split(key_id, split, validation_fraction):
                        continue
                    entries.append(len(self.indices))
                    self.indices.append((file_index, offset))
                    accepted_hash.update(line.strip() + b"\n")
            if entries:
                self.class_indices.append(entries)
                self.classes.append(path.stem.removeprefix("full_raw_").replace("_", " "))
                self.indexed_files.append({"name": path.name, "accepted": len(entries),
                                           "indexed_sha256": accepted_hash.hexdigest()})
        if not self.indices:
            if split != "all":
                raise ValueError(f"QuickDraw split={split} 在每类前 {max_per_class} 条基础池中为空；"
                                 "增大 max_per_class 或调整 validation_fraction")
            raise ValueError("QuickDraw 输入没有可用 recognized 图形")

    def __len__(self):
        return len(self.indices)

    def get(self, index: int) -> RawDrawing:
        file_index, offset = self.indices[index]
        path = self.files[file_index]
        record = self._read_record(path, int(offset))
        strokes = []
        for stroke in record["drawing"]:
            if len(stroke) < 2 or not stroke[0] or len(stroke[0]) != len(stroke[1]):
                continue
            xy = np.column_stack((stroke[0], stroke[1])).astype(np.float32)
            if not np.isfinite(xy).all():
                raise ValueError(f"QuickDraw 非有限坐标：{path}:{record.get('key_id')}")
            if len(xy) == 1:
                xy = np.repeat(xy, 2, axis=0)
            strokes.append(xy)
        if not strokes:
            raise ValueError(f"QuickDraw 空轨迹：{path}:{record.get('key_id')}")
        return RawDrawing(tuple(strokes), str(record.get("word") or
            path.stem.removeprefix("full_raw_").replace("_", " ")),
            str(record.get("key_id", f"{path.name}:{offset}")))

    def _read_record(self, path, offset):
        with path.open("rb") as stream:
            stream.seek(offset)
            return json.loads(stream.readline())

    def __getitem__(self, index: int) -> RawDrawing:
        return self.get(index)

    def sample(self, rng: np.random.Generator) -> RawDrawing:
        indices = self.class_indices[int(rng.integers(len(self.class_indices)))]
        return self.get(indices[int(rng.integers(len(indices)))])

    def manifest(self) -> dict:
        manifest = {"kind": "quickdraw_raw_offset_index", "count": len(self),
                "max_per_class": self.max_per_class,
                "classes": self.classes,
                # 用实际消费记录的 hash；数据目录迁移到服务器不因路径/mtime 变化而拒绝恢复。
                "files": self.indexed_files,
                "malformed_lines": self.malformed_lines}
        if self.split != "all":
            manifest.update(split=self.split, validation_fraction=self.validation_fraction)
        return manifest


def cap_stroke(points: np.ndarray, max_points: int = 76) -> np.ndarray:
    """仅超长笔画等弧长抽稀；点笔保留两个重合点，首尾始终保留。"""
    points = np.asarray(points, dtype=np.float32)
    if max_points < 2 or points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("需要非空 [N,2] 笔画，max_points 至少为 2")
    if not np.isfinite(points).all():
        raise ValueError("笔画含非有限坐标")
    if len(points) == 1:
        return np.repeat(points, 2, axis=0)
    if len(points) <= max_points:
        return points.copy()
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points.astype(
        np.float64), axis=0), axis=1))]
    if distance[-1] <= 1e-12:
        return np.repeat(points[:1], 2, axis=0)
    keep = np.r_[True, np.diff(distance) > 1e-12]
    grid = np.linspace(0, distance[-1], max_points)
    return np.column_stack([np.interp(grid, distance[keep], points[keep, d])
                            for d in range(2)]).astype(np.float32)


# 供页/RAW 调用者使用的清楚别名；只控制 cap，短笔画不重采样。
resample_stroke = cap_stroke


@dataclass(frozen=True)
class StrokeRecord:
    rawxy: np.ndarray
    kind: str = "stroke"
    document_index: int = 0
    page_id: int = 0


class AEStrokeStream(IterableDataset):
    """indexed 文档提供者 → 所有有效笔画；包含上下文和可见控制符。

    provider(index) 应对同一 index 返回相同文档，并接受超出基础池长度的
    全局索引；pipeline_pretrain 的数据集采用此接口实现重复轮次的新增强。
    当前入口使用 workers=0，因此游标代表已经消费的笔画，可直接续跑。
    """

    def __init__(self, document_provider: Callable[[int], object], *,
                 max_points: int = 76):
        self.document_provider = document_provider
        self.max_points = max_points
        self.document_index = 0
        self.stroke_index = 0

    def state_dict(self) -> dict:
        return {"document_index": self.document_index, "stroke_index": self.stroke_index,
                "max_points": self.max_points, "version": 1}

    def load_state_dict(self, state: dict):
        if int(state["max_points"]) != self.max_points:
            raise ValueError("AE 续跑的笔画 cap 与 checkpoint 不一致")
        self.document_index = int(state["document_index"])
        self.stroke_index = int(state["stroke_index"])

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("AE 有状态逐笔流当前使用 workers=0")
        while True:
            document = self.document_provider(self.document_index)
            elements = document.to_sequence() if hasattr(document, "to_sequence") else tuple(document)
            if not elements:
                raise ValueError(f"AE 文档 {self.document_index} 没有笔画")
            if self.stroke_index >= len(elements):
                raise ValueError("AE checkpoint 的笔画游标超出对应文档；检查数据版本")
            for index in range(self.stroke_index, len(elements)):
                element = elements[index]
                rawxy = element.rawxy if hasattr(element, "rawxy") else element
                record = StrokeRecord(cap_stroke(rawxy, self.max_points),
                    getattr(element, "kind", "stroke"), self.document_index,
                    int(getattr(element, "page_id", 0)))
                # 在 yield 前更新，最后一条刚好推进到下一个文档。
                if index + 1 == len(elements):
                    self.document_index += 1
                    self.stroke_index = 0
                else:
                    self.stroke_index = index + 1
                yield record


AE_KIND_GROUPS = {"sketch": ("sketch",), "text": ("text",),
                  "aux": ("bos", "eop", "eod", "selection")}


class BalancedAEStrokeStream(IterableDataset):
    """2026-10-03：按笔画配额组成 AE batch，各组内部保持源文档笔顺。

    三组使用独立文档游标，草图组跳过纯文本任务，不为取图扫描长故事。
    缓存仅为当前文档的矢量笔画；恢复时按任务/文档/笔画游标重建。
    与 R 的整份文档序列、目标配比及页索引无关。
    """

    def __init__(self, dataset, *, strokes_per_batch: int, shares: dict,
                 max_points: int = 76):
        super().__init__()
        if set(shares) != set(AE_KIND_GROUPS):
            raise ValueError("AE stroke_shares 必须包含 sketch/text/aux 三组")
        values = {group: float(shares[group]) for group in AE_KIND_GROUPS}
        if any(not np.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("AE 笔画比例必须为有限非负数")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("AE 笔画比例不能全部为零")
        self.shares = {group: value / total for group, value in values.items()}
        self.dataset, self.max_points = dataset, max_points
        self.strokes_per_batch = strokes_per_batch
        tasks = tuple(dataset.tasks)
        self.tasks = {group: tuple(task for task in tasks if group != "sketch" or task != "T1")
                      for group in AE_KIND_GROUPS if self.shares[group] > 0}
        if any(not tasks for tasks in self.tasks.values()):
            raise ValueError("AE 配比要求草图，但当前数据没有对象/场景任务")
        if strokes_per_batch < len(self.tasks):
            raise ValueError("AE batch 至少需要与启用的笔画组数一样多的笔画")
        wanted = {group: self.shares[group] * strokes_per_batch for group in self.tasks}
        self.quotas = {group: int(np.floor(value)) for group, value in wanted.items()}
        order = sorted(wanted, key=lambda group: wanted[group] - self.quotas[group], reverse=True)
        for group in order[:strokes_per_batch - sum(self.quotas.values())]:
            self.quotas[group] += 1
        # 小 batch 也保留每个已启用组；正式 B=512 无需该调整。
        for group in self.quotas:
            if self.quotas[group] == 0:
                donors = [other for other in self.quotas if self.quotas[other] > 1]
                donor = max(donors, key=lambda other: self.quotas[other] - wanted[other])
                self.quotas[donor] -= 1
                self.quotas[group] = 1
        self.cursors = {group: {"document_index": 0, "stroke_index": 0} for group in self.tasks}
        self.batch_offset = 0
        self._cache = {}

    def state_dict(self) -> dict:
        return {"version": 2, "sampler": "balanced_kind", "max_points": self.max_points,
                "strokes_per_batch": self.strokes_per_batch, "shares": dict(self.shares),
                "quotas": dict(self.quotas), "tasks": dict(self.tasks),
                "cursors": copy.deepcopy(self.cursors), "batch_offset": self.batch_offset}

    def load_state_dict(self, state: dict):
        expected = self.state_dict()
        for key in ("version", "sampler", "max_points", "strokes_per_batch", "shares", "quotas", "tasks"):
            if state.get(key) != expected[key]:
                raise ValueError(f"AE 续跑采样设置不一致：{key}")
        cursors = copy.deepcopy(state["cursors"])
        offset = int(state["batch_offset"])
        if (set(cursors) != set(self.tasks) or not 0 <= offset < self.strokes_per_batch or
                any(min(cursor["document_index"], cursor["stroke_index"]) < 0 for cursor in cursors.values())):
            raise ValueError("AE 续跑笔画游标无效")
        self.cursors, self.batch_offset = cursors, offset
        self._cache.clear()

    def _take(self, group: str) -> StrokeRecord:
        cursor, tasks = self.cursors[group], self.tasks[group]
        for _ in range(16 * len(tasks)):
            document_index = cursor["document_index"]
            cached = self._cache.get(group)
            if cached is None or cached[0] != document_index:
                task = tasks[document_index % len(tasks)]
                document = self.dataset.get_for_task(task, document_index // len(tasks))
                elements = tuple(element for element in document.to_sequence()
                                 if element.kind in AE_KIND_GROUPS[group])
                cached = self._cache[group] = (document_index, elements)
            elements = cached[1]
            if not elements:
                cursor.update(document_index=document_index + 1, stroke_index=0)
                continue
            index = cursor["stroke_index"]
            if index >= len(elements):
                raise ValueError("AE 笔画游标超过源文档；检查数据版本")
            element = elements[index]
            record = StrokeRecord(cap_stroke(element.rawxy, self.max_points), element.kind,
                                  document_index, int(element.page_id))
            if index + 1 == len(elements):
                cursor.update(document_index=document_index + 1, stroke_index=0)
                self._cache.pop(group, None)
            else:
                cursor["stroke_index"] += 1
            return record
        raise ValueError(f"AE 连续文档没有 {group} 笔画；检查源任务和配比")

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("AE 有状态逐笔流当前使用 workers=0")
        while True:
            end = 0
            for group, quota in self.quotas.items():
                end += quota
                if self.batch_offset < end:
                    record = self._take(group)
                    self.batch_offset = (self.batch_offset + 1) % self.strokes_per_batch
                    yield record
                    break


class AECollator:
    """有效笔画 [B,L,2]；E/D 不接触 padding 笔画行。"""

    def __init__(self, cfg: STConfig, seed: int = 42):
        if cfg.coord_dim != 2 or not cfg.unified_emb:
            raise ValueError("新分布 AE 使用 unified_emb=True、coord_dim=2")
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "version": 1}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]

    def __call__(self, records: list[StrokeRecord]) -> dict:
        if not records:
            raise ValueError("AE batch 为空")
        cfg = self.cfg
        lengths = np.array([len(r.rawxy) for r in records], dtype=np.int64)
        if lengths.max() > cfg.max_stroke_len:
            raise ValueError("AE cap 超过 encoder 的 max_stroke_len")
        ink = np.zeros((len(records), int(lengths.max()), 2), dtype=np.float32)
        for i, record in enumerate(records):
            # 继续沿用旧统一 AE 的固定画布映射。
            ink[i, :lengths[i]] = (record.rawxy - [0.5, 0.6]) / 0.6
        tau = self.rng.random((len(records), cfg.n_t_samples)).astype(np.float32)
        pos = tau * (lengths - 1)[:, None]
        low = np.floor(pos).astype(np.int64)
        high = np.minimum(low + 1, (lengths - 1)[:, None])
        fraction = (pos - low)[..., None]
        rows = np.arange(len(records))[:, None]
        targets = ink[rows, low] + fraction * (ink[rows, high] - ink[rows, low])
        endpoints = np.stack((ink[:, 0], ink[np.arange(len(records)), lengths - 1]), axis=1)
        batch = {"ink": torch.from_numpy(ink), "stroke_len": torch.from_numpy(lengths),
                 "tau": torch.from_numpy(tau),
                 "tau_target": torch.from_numpy(targets.astype(np.float32)),
                 "end_target": torch.from_numpy(endpoints),
                 "kinds": tuple(record.kind for record in records)}
        if cfg.dct_coeffs > 0:
            batch["dct_target"] = torch.from_numpy(compute_dct_for_strokes(
                ink, lengths, 128, cfg.dct_coeffs, include_mean=cfg.dct_dc))
        return batch


def build_ae_loader(document_provider, cfg: STConfig, *, strokes_per_batch: int = 256,
                    max_points: int = 76, seed: int = 42, stroke_shares: dict | None = None):
    if strokes_per_batch <= 0:
        raise ValueError("strokes_per_batch 必须为正数")
    # 新入口传 dataset；旧调用的 callable / 旧 checkpoint 仍使用原逐文档流。
    if stroke_shares is None:
        provider = document_provider if callable(document_provider) else document_provider.__getitem__
        stream = AEStrokeStream(provider, max_points=max_points)
    else:
        stream = BalancedAEStrokeStream(document_provider, strokes_per_batch=strokes_per_batch,
                                       shares=stroke_shares, max_points=max_points)
    collator = AECollator(cfg, seed)
    loader = DataLoader(stream, batch_size=strokes_per_batch, num_workers=0,
                        collate_fn=collator, pin_memory=False,
                        # workers=0 不消费 worker RNG；迭代器 base_seed 与模型 RNG 分离。
                        generator=torch.Generator().manual_seed(seed))
    return loader, stream, collator
