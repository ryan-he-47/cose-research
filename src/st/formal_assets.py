"""2026-10-03：正式数据的磁盘偏移索引；正文与RAW坐标按需读取。"""
from __future__ import annotations

import hashlib
import json
import mmap
from collections import OrderedDict
from pathlib import Path

import numpy as np

from src.text_normalization import NORMALIZATION_VERSION, normalize_text
from .ae_data import QuickDrawRawSource, source_in_split
from .pipeline_pretrain import Story


class OffsetJSONReader:
    """2026-10-04：进程内复用只读mmap，避免GPFS逐条open/缓冲readline。"""
    def __init__(self, max_files=384):
        self.max_files = max_files
        self.maps = OrderedDict()

    def read(self, path, offset):
        path = Path(path)
        mapped = self.maps.get(path)
        if mapped is None:
            if len(self.maps) >= self.max_files:
                _, oldest = self.maps.popitem(last=False)
                oldest.close()
            with path.open('rb') as stream:
                mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            self.maps[path] = mapped
        self.maps.move_to_end(path)
        mapped.seek(int(offset))
        return json.loads(mapped.readline())

    def close(self):
        for mapped in self.maps.values():
            mapped.close()
        self.maps.clear()

    def __getstate__(self):
        return {'max_files': self.max_files}

    def __setstate__(self, state):
        self.__init__(**state)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_story_index(snapshot, output, *, validation_fraction=.02, limit=None):
    """计算节点流式清洗四分片；报告和索引发布完成后才能被训练读取。"""
    import pyarrow.parquet as pq
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(snapshot).rglob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(snapshot)
    spec = {"normalization": NORMALIZATION_VERSION, "validation_fraction": validation_fraction,
            "limit": limit, "shards": [{"name": p.name, "bytes": p.stat().st_size} for p in files]}
    ready = output / "manifest.json"
    if ready.exists():
        existing = json.loads(ready.read_text(encoding="utf-8"))
        if existing["spec"] != spec:
            raise ValueError("故事索引已有不同配方，请使用独立输出目录")
        return existing
    report = {"rows": 0, "accepted": 0, "duplicates": 0, "unsupported": 0, "changed": 0}
    offsets, partitions, seen = [], {"train": [], "valid": []}, set()
    digest = hashlib.sha256()
    with (output / "stories.jsonl.tmp").open("wb") as destination:
        for file in files:
            row = 0
            for batch in pq.ParquetFile(file).iter_batches(batch_size=1024, columns=["text"]):
                for raw in batch.column(0).to_pylist():
                    source = f"{file.name}:{row}"
                    row += 1
                    report["rows"] += 1
                    if not isinstance(raw, str):
                        report["invalid_rows"] = report.get("invalid_rows", 0)+1
                        continue
                    result = normalize_text(raw)
                    if result.unsupported:
                        report["unsupported"] += 1
                        continue
                    text = result.text.strip()
                    identity = hashlib.sha256(text.encode("utf-8")).digest()
                    if not text or identity in seen:
                        report["duplicates"] += 1
                        continue
                    seen.add(identity)
                    index = len(offsets)
                    offsets.append(destination.tell())
                    split = "train" if source_in_split(text, "train", validation_fraction) else "valid"
                    partitions[split].append(index)
                    line = (json.dumps({"text": text, "source_id": source}, ensure_ascii=False) + "\n").encode("utf-8")
                    destination.write(line)
                    digest.update(line)
                    report["accepted"] += 1
                    report["changed"] += int(raw != text)
                    if limit is not None and report["accepted"] >= limit:
                        break
                if limit is not None and report["accepted"] >= limit:
                    break
            print(json.dumps({"story_shard": file.name, **report}), flush=True)
            if limit is not None and report["accepted"] >= limit:
                break
    np.save(output / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
    for split, entries in partitions.items():
        np.save(output / f"{split}.npy", np.asarray(entries, dtype=np.uint64))
    (output / "stories.jsonl.tmp").replace(output / "stories.jsonl")
    manifest = {"kind": "indexed_stories_v1", "spec": spec, "report": report,
                "sha256": digest.hexdigest(), "splits": {k: len(v) for k, v in partitions.items()}}
    write_json(ready, manifest)
    return manifest


class IndexedStories:
    def __init__(self, root, split="train"):
        self.root, self.split = Path(root), split
        self.info = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.offsets = np.load(self.root / "offsets.npy", mmap_mode="r")
        self.entries = (np.arange(len(self.offsets), dtype=np.uint64) if split == "all" else
                        np.load(self.root / f"{split}.npy", mmap_mode="r"))
        self.reader = OffsetJSONReader()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        item = self.reader.read(self.root / 'stories.jsonl', self.offsets[int(self.entries[index])])
        return Story(**item)

    def __getstate__(self):
        return {"root": str(self.root), "split": self.split}

    def __setstate__(self, state):
        self.__init__(**state)


def prepare_raw_index(raw_dir, output, *, max_per_class=10000, validation_fraction=.02):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "manifest.json"
    files = sorted(Path(raw_dir).glob("full_raw_*.ndjson"), key=lambda p: p.name)
    spec = {"max_per_class": max_per_class, "validation_fraction": validation_fraction,
            "files": [{"name": p.name, "bytes": p.stat().st_size} for p in files]}
    if ready.exists():
        manifest = json.loads(ready.read_text(encoding="utf-8"))
        if manifest["spec"] != spec:
            raise ValueError("RAW索引已有不同来源，请使用独立目录")
        return manifest
    manifests = {}
    for split in ("train", "valid"):
        source = QuickDrawRawSource(raw_dir, max_per_class, split=split,
                                   validation_fraction=validation_fraction)
        np.save(output / f"{split}_offsets.npy", np.asarray(source.indices, dtype=np.uint64))
        ends = np.cumsum([0] + [len(v) for v in source.class_indices], dtype=np.uint64)
        np.save(output / f"{split}_class_ends.npy", ends)
        manifests[split] = {"source": source.manifest(), "files": [p.name for p in source.files]}
        del source
    manifest = {"kind": "indexed_raw_v1", "spec": spec, "splits": manifests}
    write_json(ready, manifest)
    return manifest


class IndexedRawSource(QuickDrawRawSource):
    def __init__(self, raw_dir, index_dir, split="train"):
        self.root, self.index_root, self.split = Path(raw_dir), Path(index_dir), split
        info = json.loads((self.index_root / "manifest.json").read_text(encoding="utf-8"))
        source = info["splits"][split]
        self.files = tuple(self.root / name for name in source["files"])
        self.classes = source["source"]["classes"]
        self.indices = np.load(self.index_root / f"{split}_offsets.npy", mmap_mode="r")
        ends = np.load(self.index_root / f"{split}_class_ends.npy", mmap_mode="r")
        self.class_indices = [range(int(a), int(b)) for a, b in zip(ends[:-1], ends[1:])]
        self.info = source["source"]
        self.reader = OffsetJSONReader()

    def _read_record(self, path, offset):
        return self.reader.read(path, offset)

    def get(self, index):
        # 基类只访问 files/indices；将 mmap 的整数转成合法文件下标和 seek offset。
        return super().get(int(index))

    def manifest(self):
        return self.info

    def __getstate__(self):
        return {"raw_dir": str(self.root), "index_dir": str(self.index_root), "split": self.split}

    def __setstate__(self, state):
        self.__init__(**state)
