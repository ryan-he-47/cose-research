"""2026-10-27：同一问题随可见场景变化的确定性配对文档。"""
from copy import deepcopy
import numpy as np

from src.formal_scene_tasks import _circle_box, _position, realize_recipe
from src.page_document import build_document, layout_text_pages, selection_page, sketch_page
from src.st.ae_data import cap_stroke

PAIR_VERSION = '2026-10-27-visible-pairs-v1'


def paired_scene(task, recipe, source, font, *, pair_id, variant, augment=None):
    """pair_id决定问题；variant只改变画面及其真实答案，不改变问题文字。"""
    if variant not in (0, 1) or task not in ('S3', 'S4', 'S5'):
        raise ValueError('配对任务/变体无效')
    originals = recipe['objects']
    a = originals[0]
    b = next((o for o in originals if o['label'] != a['label']), None)
    if b is None:
        raise ValueError('配对源需要两个不同类别')
    family = ('direction', 'count', 'existence')[pair_id % 3] if task == 'S3' else task
    if family in ('count', 'existence'):
        count = (pair_id // 3) % 3 + variant if family == 'count' else variant
        chosen = [a] * count + [b]
        centers = ((.25, .29), (.75, .29), (.25, .71), (.75, .71))
        question = (f"How many {a['label']} sketches are there?" if family == 'count'
                    else f"Is there a {a['label']}?")
        answer = str(count) if family == 'count' else ('yes' if count else 'no')
        target_source = None
    else:
        chosen = [a, b] if variant == 0 else [b, a]
        centers = ((.25, .5), (.75, .5))
        question = (f"Is the {a['label']} to the left of the {b['label']}?" if family == 'direction'
                    else f"Circle the {a['label']}." if task == 'S4' else 'What object is circled?')
        answer = ('yes' if variant == 0 else 'no') if family == 'direction' else (
            chosen[0]['label'] if task == 'S5' else None)
        target_source = a['source_index'] if task == 'S4' else chosen[0]['source_index']
    objects = []
    for original, center in zip(chosen, centers):
        drawing = source.get(original['source_index'])
        points = np.concatenate(drawing.strokes)
        lower, upper = points.min(0), points.max(0)
        scale = .26 / max(float((upper-lower).max()), 1e-8)
        translation = np.asarray(center) - (lower+upper)*scale/2
        strokes = tuple(cap_stroke(s*scale+translation) for s in drawing.strokes)
        pts = np.concatenate(strokes)
        box = [*pts.min(0).tolist(), *pts.max(0).tolist()]
        obj = deepcopy(original)
        obj.update(bbox=box, center=list(center), position=_position(center),
                   affine=dict(scale=scale, translation=translation.tolist()),
                   selection_box=_circle_box(strokes), stroke_count=len(strokes))
        objects.append(obj)
    objects.sort(key=lambda o: (o['center'][1], o['center'][0], o['source_index']))
    for i, obj in enumerate(objects):
        obj['instance_id'] = i
    relations=[]
    for first in objects:
        for second in objects:
            if first is second:continue
            dx,dy=np.asarray(first['center'])-second['center']
            area=lambda o:max(1e-8,(o['bbox'][2]-o['bbox'][0])*(o['bbox'][3]-o['bbox'][1]))
            relations.append(dict(subject=first['instance_id'],reference=second['instance_id'],
                left=bool(dx<-.12),right=bool(dx>.12),above=bool(dy<-.12),below=bool(dy>.12),
                distance=float(np.hypot(dx,dy)),bbox_area_ratio=area(first)/area(second)))
    paired_recipe = dict(recipe, objects=objects, relations=relations, repeated=len(chosen) != len({o['label'] for o in chosen}))
    strokes, objects, owners = realize_recipe(paired_recipe, source)
    metadata = dict(task=task, recipe_id=recipe['recipe_id'], objects=objects, stroke_owner=owners,
                    draw_order=list(range(len(objects))), pair_id=pair_id, variant_id=variant,
                    relations=relations,
                    paired_scene_version=PAIR_VERSION, question=question,
                    query=dict(kind=family), answer=answer)
    text = lambda value, target=False: layout_text_pages(value, font, targetable=target, augment=augment)
    scene = sketch_page(strokes, targetable=False, fit=False)
    if task == 'S3':
        pages = (*text('Look at this scene.'), scene, *text(question), *text(answer, True))
    else:
        target = next(o for o in objects if o['source_index'] == target_source)
        box = target['selection_box'] if task == 'S4' else (.08, .33, .42, .67)
        metadata.update(target_instance_id=target['instance_id'], selection_box=box)
        if task == 'S4':
            metadata['answer'] = dict(kind='selection', bbox=box)
            pages = (*text('Find an object in this scene.'), scene, *text(question), selection_page(box))
        else:
            pages = (*text(question), scene, selection_page(box, targetable=False), *text(answer, True))
    return build_document(pages, complete=True, metadata=metadata)
