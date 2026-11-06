"""QuickDraw 对象合成首批场景任务 S1–S5（2026-10-02）。

布局真值留在 Document.metadata；实际模型输入是任务指定的文字、图与圈选页。
实例顺序沿用输入顺序，每个实例保留源笔顺，不执行片内 shuffle。
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from src.page_document import (Document, build_document, fit_sketch, layout_text_pages,
                               selection_page, sketch_page)
from src.st.ae_data import RawDrawing
from src.text_normalization import normalize_text
from src.text_renderer import VectorFont


_BODY_BOX = (0.06, 0.10, 0.94, 0.86)
_CELLS = ((0.07, 0.12, 0.48, 0.46), (0.52, 0.12, 0.93, 0.46),
          (0.07, 0.50, 0.48, 0.84), (0.52, 0.50, 0.93, 0.84))
_POSITIONS = ("upper left", "upper right", "lower left", "lower right")
_ABSENT_LABELS = ("cat", "dog", "tree", "house", "bird", "car", "flower", "fish")


def _bbox(strokes):
    points = np.concatenate(strokes)
    return tuple(float(v) for v in (*points.min(axis=0), *points.max(axis=0)))


def _area(obj):
    x0, y0, x1, y1 = obj["bbox"]
    return (x1 - x0) * (y1 - y0)


def _circle_box(strokes):
    """以实际 ink 外包框为起点，扩为覆盖全部点的椭圆圈选。"""
    points = np.concatenate(strokes)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    radius = (hi - lo) / 2 + 0.012
    extent = float(np.sqrt(np.max(np.sum(((points - center) / radius) ** 2, axis=1))))
    radius *= max(1.0, extent) * 1.02
    return tuple(float(v) for v in (*(center - radius), *(center + radius)))


def _make_scene(drawings: Sequence[RawDrawing], rng: np.random.Generator):
    unique, seen = [], set()
    for drawing in drawings:
        result = normalize_text(drawing.label)
        if result.unsupported or not result.text.strip():
            raise ValueError(f"场景类别无法用当前字体渲染：{drawing.label!r}")
        label = result.text.strip()
        if label not in seen:
            seen.add(label)
            unique.append((drawing, label))
        if len(unique) == 4:
            break
    if len(unique) < 2:
        raise ValueError("首版场景需要 2–4 个不同类别的对象")
    slots = rng.choice(4, len(unique), replace=False)
    scene_strokes, owners, objects, instance_strokes = [], [], [], []
    for index, ((drawing, label), slot) in enumerate(zip(unique, slots)):
        x0, y0, x1, y1 = _CELLS[int(slot)]
        center = np.array(((x0 + x1) / 2, (y0 + y1) / 2)) + rng.uniform(-.01, .01, 2)
        # 小尺度和随机平移覆盖场景 AE 分布；四槽位之间保持空隙。
        width, height = rng.uniform(.19, .24), rng.uniform(.15, .19)
        target_box = (*map(float, center - [width / 2, height / 2]),
                      *map(float, center + [width / 2, height / 2]))
        fitted = fit_sketch(drawing.strokes, box=target_box)
        if not fitted:
            raise ValueError(f"场景源对象无有效笔画：{drawing.key_id}")
        bbox = _bbox(fitted)
        actual_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        raw_points = np.concatenate(drawing.strokes)
        span = raw_points.max(axis=0) - raw_points.min(axis=0)
        nonzero = span > 1e-12
        dimensions = np.array((width, height))
        scale = float(np.min(dimensions[nonzero] / span[nonzero])) if np.any(nonzero) else 1.0
        translation = np.asarray(fitted[0][0]) - np.asarray(drawing.strokes[0][0]) * scale
        start = len(scene_strokes)
        scene_strokes.extend(fitted)
        owners.extend([index] * len(fitted))
        instance_strokes.append(fitted)
        objects.append({"instance_id": index, "key_id": str(drawing.key_id), "label": label,
                        "bbox": bbox, "center": actual_center, "slot": int(slot),
                        "position": _POSITIONS[int(slot)], "stroke_range": (start, len(scene_strokes)),
                        "affine": {"scale": scale, "translation": translation.tolist()}})
    relations = []
    for left in objects:
        for right in objects:
            if left["instance_id"] == right["instance_id"]:
                continue
            dx = left["center"][0] - right["center"][0]
            dy = left["center"][1] - right["center"][1]
            relations.append({"subject": left["instance_id"], "reference": right["instance_id"],
                              "left": dx < -.15, "right": dx > .15,
                              "above": dy < -.15, "below": dy > .15,
                              "larger_bbox": _area(left) > _area(right)})
    description = "The scene has " + ", ".join(
        f'a {obj["label"]} at the {obj["position"]}' for obj in objects) + "."
    return tuple(scene_strokes), objects, owners, relations, description, tuple(instance_strokes)


def _question(objects, relations, rng):
    kind = str(rng.choice(("count", "horizontal", "vertical", "size")))
    if kind == "count":
        present = bool(rng.integers(2))
        if present:
            obj = objects[int(rng.integers(len(objects)))]
            label = obj["label"]
            count = 1
        else:
            labels = {obj["label"] for obj in objects}
            label = next(label for label in _ABSENT_LABELS if label not in labels)
            count = 0
        return f"How many {label} objects are in the scene?", str(count), {
            "kind": "count", "label": label, "value": count}
    if kind == "size":
        indices = rng.choice(len(objects), 2, replace=False)
        a, b = (objects[int(index)] for index in indices)
        label = a["label"] if _area(a) > _area(b) else b["label"]
        if math.isclose(_area(a), _area(b), abs_tol=1e-10):
            label = "same size"
        return f'Which is bigger, the {a["label"]} or the {b["label"]}?', label, {
            "kind": "size", "instance_ids": [a["instance_id"], b["instance_id"]],
            "bbox_areas": [_area(a), _area(b)], "value": label}
    axis = ("left", "right") if kind == "horizontal" else ("above", "below")
    eligible = [r for r in relations if any(r[direction] for direction in axis)]
    if not eligible:
        axis = ("above", "below") if kind == "horizontal" else ("left", "right")
        eligible = [r for r in relations if any(r[direction] for direction in axis)]
    relation = eligible[int(rng.integers(len(eligible)))]
    direction = str(rng.choice(axis))
    a, b = objects[relation["subject"]], objects[relation["reference"]]
    connector = f"to the {direction} of" if direction in ("left", "right") else direction
    answer = "yes" if relation[direction] else "no"
    return f'Is the {a["label"]} {connector} the {b["label"]}?', answer, {
        "kind": direction, "subject": a["instance_id"], "reference": b["instance_id"],
        "value": answer}


def _grounding_query(objects, relations, rng):
    unique_relations = []
    for reference in objects:
        for direction in ("left", "right", "above", "below"):
            candidates = [r for r in relations if r["reference"] == reference["instance_id"]
                          and r[direction]]
            if len(candidates) == 1:
                unique_relations.append((candidates[0], direction))
    choices = ("class", "position", "relation") if unique_relations else ("class", "position")
    mode = str(rng.choice(choices))
    if mode == "relation":
        relation, direction = unique_relations[int(rng.integers(len(unique_relations)))]
        target, reference = objects[relation["subject"]], objects[relation["reference"]]
        connector = f"to the {direction} of" if direction in ("left", "right") else direction
        question = f'Circle the object {connector} the {reference["label"]}.'
        query = {"kind": "relation", "direction": direction, "reference": reference["instance_id"]}
    else:
        target = objects[int(rng.integers(len(objects)))]
        if mode == "class":
            question = f'Circle the {target["label"]}.'
            query = {"kind": "class", "label": target["label"]}
        else:
            question = f'Circle the object in the {target["position"]}.'
            query = {"kind": "position", "position": target["position"]}
    return question, target, query


def build_scene_document(task: str, drawings: Sequence[RawDrawing], font: VectorFont,
                         rng: np.random.Generator, augment: Callable | None = None) -> Document:
    """S1 图生文、S2 文生图、S3 问答、S4 圈选定位、S5 圈选识别。"""
    task = task.upper()
    if task not in ("S1", "S2", "S3", "S4", "S5"):
        raise ValueError(f"首批场景任务不支持 {task}")
    strokes, objects, owners, relations, description, instances = _make_scene(drawings, rng)
    metadata = {"task": task, "objects": objects, "stroke_owner": owners,
                "stroke_owner_scope": "scene_body", "description": description,
                "relations": relations, "scene_bbox": _bbox(strokes),
                "draw_order": [obj["instance_id"] for obj in objects]}

    def text_pages(text, target=False):
        return layout_text_pages(text, font, targetable=target, augment=augment)

    def scene(target=False):
        return sketch_page(strokes, targetable=target, fit=False)

    if task == "S1":
        metadata.update(question="Describe this scene.", answer=description)
        pages = (*text_pages(metadata["question"]), scene(), *text_pages(description, True))
    elif task == "S2":
        question = "Draw this scene: " + description
        metadata.update(question=question, answer={"kind": "scene", "instance_ids": metadata["draw_order"]})
        pages = (*text_pages(question), scene(True))
    elif task == "S3":
        question, answer, query = _question(objects, relations, rng)
        metadata.update(question=question, answer=answer, query=query)
        pages = (*text_pages("Look at this scene."), scene(), *text_pages(question), *text_pages(answer, True))
    elif task == "S4":
        question, target, query = _grounding_query(objects, relations, rng)
        circle_box = _circle_box(instances[target["instance_id"]])
        metadata.update(question=question, query=query, target_instance_id=target["instance_id"],
                        answer={"kind": "selection", "bbox": circle_box})
        pages = (*text_pages("Find an object in this scene."), scene(), *text_pages(question),
                 selection_page(circle_box))
    else:
        target = objects[int(rng.integers(len(objects)))]
        circle_box = _circle_box(instances[target["instance_id"]])
        metadata.update(question="Name the object in the circle.", answer=target["label"],
                        target_instance_id=target["instance_id"], selection_box=circle_box)
        pages = (*text_pages(metadata["question"]), scene(), selection_page(circle_box, targetable=False),
                 *text_pages(target["label"], True))
    return build_document(pages, complete=True, metadata=metadata)
