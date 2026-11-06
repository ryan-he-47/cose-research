"""
Letter Database — Phase 2b
===========================
从手写采集的 word JSON → 归一化 → 按 (char, pos) 索引的字母库.

用法:
  from src.letter_database import LetterDatabase
  db = LetterDatabase("data/letter_database.npz")
  ink, lens = db.sample("c", "first", rng)   # → (n_strokes, max_len, 4), (n_strokes,)
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root on path for data_pipeline imports
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from src.precache import size_normalization, resample_stroke, diagram_to_strokes


# ══════════════════════════════════════════════════════════════════
# Position encoding
# ══════════════════════════════════════════════════════════════════
POS_FIRST = 0
POS_MIDDLE = 1
POS_LAST = 2
POS_ANY = 3  # Used only for online sampling

POS_NAME = {0: "first", 1: "middle", 2: "last", 3: "any"}


def _infer_position(n_letters: int, pos_in_word: int) -> int:
    """将词内位置映射到 first/middle/last."""
    if n_letters == 1:
        return POS_ANY
    if pos_in_word == 0:
        return POS_FIRST
    if pos_in_word == n_letters - 1:
        return POS_LAST
    return POS_MIDDLE


# ══════════════════════════════════════════════════════════════════
# Database builder
# ══════════════════════════════════════════════════════════════════

def _extract_letters_from_dir(
    json_dir: str,
    writer_id: int,
    resample_ms: float,
) -> List[dict]:
    """从一个书写者的 word JSON 目录提取全部字母 (build_letter_database 的核心逻辑,
    抽出以支持多书写者合并, 见 build_letter_database_multi)."""
    json_files = sorted([
        f for f in os.listdir(json_dir) if f.endswith(".json")
    ])
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")

    print(f"Found {len(json_files)} word JSON files in {json_dir} (writer_id={writer_id})")

    all_letters = []
    for fname in json_files:
        path = os.path.join(json_dir, fname)
        word_data = json.load(open(path, encoding="utf-8"))
        word = word_data["word"]
        n = len(word_data["letters"])

        for letter in word_data["letters"]:
            raw_strokes = letter["strokes"]
            char = letter["char"]
            pos_raw = letter["pos_in_word"]
            pos = _infer_position(n, pos_raw)

            # Convert to CoSE drawing format: [[x], [y], [t]]
            # Use real timestamps from HTML (ms since stroke start), fallback to simulated
            drawing = []
            for s in raw_strokes:
                xs = list(s["x"])
                ys = list(s["y"])
                ts = s.get("t_ms", None)
                if ts is None:
                    # Backward compat: old JSON without time data → simulate
                    ts = [i * resample_ms for i in range(len(xs))]
                else:
                    ts = list(ts)
                drawing.append([xs, ys, ts])

            # Normalize & resample (same as CoSE pipeline)
            drawing = size_normalization(drawing)
            drawing = [resample_stroke(s, resample_ms) for s in drawing]

            # Convert to numpy
            ink, lens = diagram_to_strokes(drawing)

            all_letters.append({
                "char": char,
                "position": pos,
                "word": word,
                "ink": ink,
                "lens": lens,
                "writer": writer_id,
            })

    print(f"  {len(all_letters)} letters extracted from {json_dir}")
    return all_letters


def _pack_and_save(all_letters: List[dict], output_path: str) -> str:
    """打包 all_letters → letter_database.npz (chars/positions/writers/words/inks/lenses)."""
    n_letters = len(all_letters)
    print(f"Total letters: {n_letters}")

    count_by_char_pos = defaultdict(lambda: [0, 0, 0, 0])  # [first, middle, last, any]
    for a in all_letters:
        count_by_char_pos[a["char"]][a["position"]] += 1

    for ch in "abcdefghijklmnopqrstuvwxyz":
        cnts = count_by_char_pos.get(ch, [0, 0, 0, 0])
        print(f"  {ch}: first={cnts[0]} middle={cnts[1]} last={cnts[2]} any={cnts[3]}")

    n = len(all_letters)
    chars = np.empty(n, dtype=object)
    positions = np.empty(n, dtype=np.int8)
    writers = np.empty(n, dtype=np.int16)
    words = np.empty(n, dtype=object)
    inks = np.empty(n, dtype=object)
    lenses = np.empty(n, dtype=object)
    for i, a in enumerate(all_letters):
        chars[i] = a["char"]
        positions[i] = a["position"]
        writers[i] = a["writer"]
        words[i] = a["word"]
        inks[i] = a["ink"]
        lenses[i] = a["lens"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        chars=chars,
        positions=positions,
        writers=writers,
        words=words,
        inks=inks,
        lenses=lenses,
        meta=np.array([n_letters, len(set(a["writer"] for a in all_letters))]),
    )
    print(f"Saved {n_letters} letters to {output_path}")
    return output_path


def build_letter_database(
    json_dir: str,
    output_path: str,
    resample_ms: float = 20.0,
    writer_id: int = 0,
) -> str:
    """汇总单个书写者的 word JSON 文件 → letter_database.npz

    Args:
        json_dir: 包含 *_v*.json 文件的目录
        output_path: 输出 .npz 路径
        resample_ms: 时间重采样步长 (ms). 采集器无时间戳，模拟每笔匀速
        writer_id: 书写者 ID (多志愿者时递增)

    Returns:
        output_path
    """
    all_letters = _extract_letters_from_dir(json_dir, writer_id, resample_ms)
    return _pack_and_save(all_letters, output_path)


def build_letter_database_multi(
    sources: List[Tuple[str, int]],
    output_path: str,
    resample_ms: float = 20.0,
) -> str:
    """汇总多个书写者的 word JSON 目录 → 合并的 letter_database.npz.

    Args:
        sources: [(json_dir, writer_id), ...] — 每个书写者一个目录 + 唯一 writer_id
        output_path: 输出 .npz 路径
        resample_ms: 时间重采样步长 (ms)

    Returns:
        output_path
    """
    all_letters = []
    for json_dir, writer_id in sources:
        all_letters.extend(_extract_letters_from_dir(json_dir, writer_id, resample_ms))
    return _pack_and_save(all_letters, output_path)


# ══════════════════════════════════════════════════════════════════
# Query interface
# ══════════════════════════════════════════════════════════════════

class LetterDatabase:
    """字母库查询接口."""

    def __init__(self, db_path: str):
        data = np.load(db_path, allow_pickle=True)
        self.chars = data["chars"]
        self.positions = data["positions"]
        self.writers = data["writers"]
        self.inks = data["inks"]
        self.lenses = data["lenses"]
        self.n_letters = len(self.chars)

        # Build index: (char, position) → list of indices
        self._index: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        for i in range(self.n_letters):
            key = (str(self.chars[i]), int(self.positions[i]))
            self._index[key].append(i)

        # Build ANY index
        char_positions: Dict[str, List[int]] = defaultdict(list)
        for i in range(self.n_letters):
            char_positions[str(self.chars[i])].append(i)
        for ch, idx_list in char_positions.items():
            self._index[(ch, POS_ANY)] = idx_list

        print(f"LetterDatabase: {self.n_letters} letters, "
              f"{len(self._index)} index entries")

    def sample(
        self,
        char: str,
        position: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """随机抽取一个字母变体.

        Args:
            char: "a"-"z"
            position: POS_FIRST, POS_MIDDLE, POS_LAST, or POS_ANY
            rng: RandomState for reproducibility

        Returns:
            (ink, lens): ink shape (n_strokes, max_len, 4), lens shape (n_strokes,)
        """
        if rng is None:
            rng = np.random

        key = (char.lower(), position)
        candidates = self._index.get(key, [])
        if not candidates:
            # Fallback: try POS_ANY
            key_any = (char.lower(), POS_ANY)
            candidates = self._index.get(key_any, [])
        if not candidates:
            raise KeyError(f"No samples for char='{char}', position={POS_NAME.get(position, position)}")

        idx = int(rng.choice(candidates))
        return self.inks[idx].copy(), self.lenses[idx].copy()

    def n_samples(self, char: str, position: int) -> int:
        """返回指定 (char, position) 的样本数."""
        key = (char.lower(), position)
        return len(self._index.get(key, []))
