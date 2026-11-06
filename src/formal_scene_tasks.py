"""2026-10-03：已审批的连续布局、重复对象和S1–S5；只使用可见几何事实。"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

from src.page_document import build_document, fit_sketch, layout_text_pages, selection_page, sketch_page
from src.scene_tasks import _bbox, _circle_box
from src.text_normalization import normalize_text
from src.st.ae_data import cap_stroke


SCENE_VERSION = "2026-10-03-continuous-v1"
CAPTION_VERSION = "2026-10-07-count-position-relation-v2"


def instruction(canonical, variants, rng):
    return canonical if rng.random() < .7 else str(rng.choice(variants))


def _position(center):
    x, y = center
    horizontal = "left" if x < .38 else "right" if x > .62 else "center"
    vertical = "upper" if y < .36 else "lower" if y > .60 else ""
    return " ".join(part for part in (vertical, horizontal) if part)


def _area(obj):
    a, b, c, d = obj["bbox"]
    return max(1e-8, (c-a)*(d-b))


def _apart(a, b, margin=.012):
    return a[2]+margin < b[0] or b[2]+margin < a[0] or a[3]+margin < b[1] or b[3]+margin < a[1]


def _contains(box, center):
    x0, y0, x1, y1 = box
    return ((center[0]-(x0+x1)/2)/((x1-x0)/2))**2 + ((center[1]-(y0+y1)/2)/((y1-y0)/2))**2 <= 1


def make_recipe(source, seed, recipe_id):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 891, recipe_id]))
    count = int(rng.choice((2, 3, 4), p=(.30, .40, .30)))
    classes = rng.choice(len(source.classes), count, replace=len(source.classes) < count).tolist()
    repeated = bool(rng.random() < .25)
    if repeated:
        repeats = int(rng.integers(2, count+1))
        classes[:repeats] = [classes[0]] * repeats
    chosen, used = [], set()
    for cls in classes:
        entries = source.class_indices[int(cls)]
        index = int(entries[int(rng.integers(len(entries)))])
        for _ in range(16):
            if index not in used:
                break
            index = int(entries[int(rng.integers(len(entries)))])
        if index in used:
            raise ValueError("重复类别需要不同的源草图")
        used.add(index)
        chosen.append((index, source.get(index)))
    for _ in range(64):
        items = []
        for source_index, drawing in chosen:
            side = float(rng.uniform(.16, .36))
            for _ in range(128):
                center = rng.uniform((.10+side/2, .14+side/2), (.90-side/2, .82-side/2))
                box = (*map(float, center-side/2), *map(float, center+side/2))
                strokes = fit_sketch(drawing.strokes, box=box)
                actual = _bbox(strokes)
                if all(_apart(actual, obj["bbox"]) for obj in items):
                    points = np.concatenate(drawing.strokes)
                    span = np.ptp(points, axis=0)
                    nonzero = span > 1e-12
                    scale = float(np.min(side/span[nonzero])) if nonzero.any() else 1.0
                    translation = strokes[0][0] - drawing.strokes[0][0]*scale
                    circ = _circle_box(strokes)
                    if min(circ[:2]) < .025 or max(circ[2:]) > .975:
                        continue
                    items.append({"source_index": source_index, "key_id": drawing.key_id,
                        "label": normalize_text(drawing.label).text, "bbox": actual,
                        "center": ((actual[0]+actual[2])/2, (actual[1]+actual[3])/2),
                        "affine": {"scale": scale, "translation": translation.tolist()},
                        "selection_box": circ, "stroke_count": len(strokes)})
                    break
            else:
                break
        if len(items) == count and all(not _contains(a["selection_box"], b["center"])
                for a in items for b in items if a is not b):
            items.sort(key=lambda obj: (obj["center"][1], obj["center"][0], obj["source_index"]))
            for i, obj in enumerate(items):
                obj.update(instance_id=i, position=_position(obj["center"]))
            relations = []
            for a in items:
                for b in items:
                    if a is b:
                        continue
                    dx,dy = np.asarray(a['center'])-b['center']
                    relations.append({'subject':a['instance_id'],'reference':b['instance_id'],
                        'left':bool(dx < -.12),'right':bool(dx > .12),
                        'above':bool(dy < -.12),'below':bool(dy > .12),
                        'distance':float(np.hypot(dx,dy)), 'bbox_area_ratio':_area(a)/_area(b)})
            return {"version": SCENE_VERSION, "recipe_id": recipe_id, "seed": seed,
                    "objects": items, "relations":relations, "repeated": repeated}
    raise ValueError(f"场景{recipe_id}布局未放下{count}个对象")


def realize_recipe(recipe, source):
    strokes, owners, objects = [], [], []
    for obj in recipe["objects"]:
        drawing = source.get(obj["source_index"])
        if drawing.key_id != obj["key_id"]:
            raise ValueError("场景配方引用的RAW索引/来源不同")
        scale = obj["affine"]["scale"]
        translation = np.asarray(obj["affine"]["translation"], np.float32)
        instance = tuple(cap_stroke(s*scale + translation) for s in drawing.strokes)
        start = len(strokes)
        strokes.extend(instance)
        owners.extend([obj["instance_id"]]*len(instance))
        objects.append({**obj, "stroke_range": (start, len(strokes))})
    return tuple(strokes), objects, owners


def _descriptors(objects):
    counts = Counter(obj["label"] for obj in objects)
    result = {}
    for obj in objects:
        same = sorted((x for x in objects if x["label"] == obj["label"]), key=lambda x: (x["center"][0], x["center"][1]))
        rank = next(i for i, x in enumerate(same) if x["instance_id"] == obj["instance_id"])
        result[obj["instance_id"]] = obj["label"] if counts[obj["label"]] == 1 else (
            ("leftmost" if rank == 0 else "rightmost" if rank == len(same)-1 else
             ("second from the left", "third from the left")[rank-1]) + " " + obj["label"])
    return result


def _caption(objects, *, detailed=True):
    groups = {}
    for obj in objects:
        groups.setdefault(obj["label"], []).append(obj)
    parts = []
    for label, group in groups.items():
        amount = ('One','Two','Three','Four')[len(group)-1]
        noun = f"sketch{'es' if len(group)>1 else ''} of {label}" if label.startswith('The ') else (
            label if len(group)==1 else f'{label} sketches')
        phrase = f'{amount} {noun}'
        if detailed:
            phrase += " at " + " and ".join(obj["position"] for obj in group)
        parts.append(phrase)
    caption = "; ".join(parts) + "."
    # 2026-10-07：完整列出实例数；补一个明确几何关系。简略版省去绝对位置，仍保留关系。
    names = _descriptors(objects)
    candidates = []
    for i,a in enumerate(objects):
        for b in objects[i+1:]:
            dx,dy = np.asarray(a['center'])-b['center']
            if abs(dx) > .12:
                candidates.append((abs(dx),a,b,'left of' if dx < 0 else 'right of'))
            if abs(dy) > .12:
                candidates.append((abs(dy),a,b,'above' if dy < 0 else 'below'))
    if candidates:
        _,a,b,relation = max(candidates,key=lambda item:item[0])
        def definite(name):
            return name if name.startswith('The ') else 'the '+name.replace(' The ',' ')
        subject = definite(names[a['instance_id']])
        subject = subject[0].upper()+subject[1:]
        caption += f" {subject} is {relation} {definite(names[b['instance_id']])}."
    return caption


def _questions(objects, labels, names, rng):
    family = str(rng.choice(("count", "direction", "size", "distance", "reference"), p=(.35,.30,.15,.10,.10)))
    if family == "count":
        existing = {obj["label"] for obj in objects}
        absent = [label for label in labels if label not in existing]
        label = str(rng.choice(absent)) if absent and rng.random() < .25 else str(rng.choice(sorted(existing)))
        count = sum(obj["label"] == label for obj in objects)
        if rng.random() < .25:
            return f"Is there a {label}?", "yes" if count else "no", {"kind": "existence", "label": label, "value": count}
        return f"How many {label} sketches are there?", str(count), {"kind": "count", "label": label, "value": count}
    a, b = (objects[int(i)] for i in rng.choice(len(objects), 2, replace=False))
    if family == "size" and abs(math.log(_area(a)/_area(b))) > .2:
        bigger = a if _area(a) > _area(b) else b
        return f"Which is bigger, the {names[a['instance_id']]} or the {names[b['instance_id']]}?", names[bigger["instance_id"]], {"kind": "size", "target": bigger["instance_id"]}
    if family == "distance" and len(objects) > 2:
        candidates = sorted((obj for obj in objects if obj is not a), key=lambda obj: np.linalg.norm(np.asarray(obj["center"])-a["center"]))
        d0, d1 = (np.linalg.norm(np.asarray(obj["center"])-a["center"]) for obj in candidates[:2])
        if d1-d0 > .04:
            target = candidates[0]
            return f"Which object is closest to the {names[a['instance_id']]}?", names[target["instance_id"]], {"kind": "distance", "target": target["instance_id"]}
    pairs = []
    for a in objects:
        for b in objects:
            if a is b:
                continue
            dx, dy = np.asarray(a["center"])-b["center"]
            if abs(dx) > .12:
                pairs.append((a, b, "left" if dx < 0 else "right"))
            if abs(dy) > .12:
                pairs.append((a, b, "above" if dy < 0 else "below"))
    if not pairs:
        label = a["label"]
        count = sum(obj["label"] == label for obj in objects)
        return f"How many {label} sketches are there?", str(count), {"kind": "count", "label": label, "value": count}
    a, b, direction = pairs[int(rng.integers(len(pairs)))]
    if family == "reference":
        matching = [x for x in pairs if x[1] is b and x[2] == direction]
        if len(matching) == 1:
            connector = f"to the {direction} of" if direction in ("left", "right") else direction
            return f"What is {connector} the {names[b['instance_id']]}?", names[a["instance_id"]], {"kind": "reference", "target": a["instance_id"]}
    yes = bool(rng.integers(2))
    if not yes:
        direction = {"left":"right", "right":"left", "above":"below", "below":"above"}[direction]
    connector = f"to the {direction} of" if direction in ("left", "right") else direction
    return f"Is the {names[a['instance_id']]} {connector} the {names[b['instance_id']]}?", "yes" if yes else "no", {"kind": "direction", "value": yes}


def build_formal_scene(task, recipe, source, font, *, rng, augment=None):
    strokes, objects, owners = realize_recipe(recipe, source)
    names = _descriptors(objects)
    metadata = {"task": task, "recipe_id": recipe["recipe_id"], "scene_version": SCENE_VERSION,
                "caption_version": CAPTION_VERSION,
                "objects": objects, "relations":recipe.get('relations',[]),
                "stroke_owner": owners, "draw_order": list(range(len(objects)))}
    def text(value, target=False):
        return layout_text_pages(value, font, targetable=target, augment=augment)
    scene = lambda target=False: sketch_page(strokes, targetable=target, fit=False)
    if task == "S1":
        prompt = instruction("Describe this scene.", ("Describe the objects and their positions.", "What is in this scene?"), rng)
        answer = _caption(objects)
        pages = (*text(prompt), scene(), *text(answer, True))
    elif task == "S2":
        description = _caption(objects, detailed=rng.random() < .8)
        prompt = instruction("Draw this scene:", ("Sketch this scene:", "Draw the following scene:"), rng)
        pages = (*text(prompt + " " + description), scene(True))
        answer = {"kind": "scene", "instances": list(names)}
    elif task == "S3":
        question, answer, query = _questions(objects, source.classes, names, rng)
        prompt = instruction("Look at this scene.", ("Study this scene.", "Answer about this scene."), rng)
        pages = (*text(prompt), scene(), *text(question), *text(answer, True))
        metadata.update(question=question, query=query)
    else:
        target = objects[int(rng.integers(len(objects)))]
        box = target["selection_box"]
        # 同一目标的圈形可连续变化；保持覆盖全部点且不罩住另一对象中心。
        points = np.concatenate(strokes[slice(*target["stroke_range"])])
        original_center = np.array(((box[0]+box[2])/2,(box[1]+box[3])/2))
        original_radius = np.array(((box[2]-box[0])/2,(box[3]-box[1])/2))
        for _ in range(8):
            center = original_center+rng.uniform(-.002,.002,2)
            radius = original_radius*rng.uniform(1.01,1.06,2)
            proposed = tuple(float(value) for value in (*center-radius,*center+radius))
            if min(proposed[:2]) >= .01 and max(proposed[2:]) <= .99 and \
                    np.max(np.sum(((points-center)/radius)**2,axis=1)) <= 1 and \
                    all(not _contains(proposed,obj["center"]) for obj in objects if obj is not target):
                box = proposed
                break
        if task == "S4":
            mode = str(rng.choice(("class", "position", "relation_size"), p=(.4,.35,.25)))
            same = [obj for obj in objects if obj["label"] == target["label"]]
            descriptor = names[target["instance_id"]]
            if mode == "position" and len(same) > 1:
                vertical = sorted(same, key=lambda obj: obj["center"][1])
                if target is vertical[0] and vertical[1]["center"][1]-target["center"][1] > .12:
                    descriptor = "uppermost " + target["label"]
                elif target is vertical[-1] and target["center"][1]-vertical[-2]["center"][1] > .12:
                    descriptor = "lowermost " + target["label"]
            if mode == "relation_size" and len(same) > 1:
                sizes = sorted(same, key=_area)
                if target is sizes[0] and _area(sizes[1])/_area(target) > 1.3:
                    descriptor = "smallest " + target["label"]
                elif target is sizes[-1] and _area(target)/_area(sizes[-2]) > 1.3:
                    descriptor = "largest " + target["label"]
            question = f"Circle the {descriptor}."
            if mode == "relation_size" and rng.random() < .5:
                options = []
                for reference in objects:
                    if reference is target:
                        continue
                    dx, dy = np.asarray(target["center"])-reference["center"]
                    for axis, delta, directions in ((0,dx,("left","right")), (1,dy,("above","below"))):
                        if abs(delta) <= .12:
                            continue
                        sign = -1 if delta < 0 else 1
                        matches = [obj for obj in same if sign*(obj["center"][axis]-reference["center"][axis]) > .12]
                        if len(matches) == 1 and matches[0] is target:
                            direction = directions[0 if sign < 0 else 1]
                            connector = f"to the {direction} of" if axis == 0 else direction
                            options.append(f"Circle the {target['label']} {connector} the {names[reference['instance_id']]}.")
                if options:
                    question = str(rng.choice(options))
            prompt = instruction("Find an object in this scene.", ("Locate the requested object.", "Find the target object."), rng)
            pages = (*text(prompt), scene(), *text(question), selection_page(box))
            answer = {"kind": "selection", "bbox": box}
            metadata.update(question=question, query_kind=mode)
        else:
            prompt = instruction("Name the object in the circle.", ("What object is circled?", "Identify the circled object."), rng)
            pages = (*text(prompt), scene(), selection_page(box, targetable=False), *text(target["label"], True))
            answer = target["label"]
        metadata.update(target_instance_id=target["instance_id"], selection_box=box)
    metadata["answer"] = answer
    return build_document(pages, complete=True, metadata=metadata)
