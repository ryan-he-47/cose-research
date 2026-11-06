"""2026-10-27：全更新归一的文字、对象类别和可见目标匹配监督。"""
from collections import Counter
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VisualGroundingHead(nn.Module):
    def __init__(self, width, classes, key_dim=128):
        super().__init__()
        if classes<=0 or key_dim<=0:raise ValueError('对象辅助尺寸须为正')
        self.object_class = nn.Linear(width, classes)
        self.query = nn.Linear(width, key_dim, bias=False)
        self.key = nn.Linear(width, key_dim, bias=False)

    def statistics(self, hidden, batch):
        result = {k: hidden.sum().float()*0 for k in ('ground_class_sum','ground_pointer_sum','ground_class_weight',
                                    'ground_pointer_weight','ground_class_correct','ground_pointer_correct')}
        labels = batch.get('ground_labels')
        if labels is None: return result
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            for b, data in enumerate(labels):
                h = hidden[b].float()
                positions, classes = data['object_positions'], data['object_classes']
                if positions:
                    logits = self.object_class(h[positions])
                    target = torch.tensor(classes, device=h.device)
                    weight = data['object_weight']
                    result['ground_class_sum'] += F.cross_entropy(logits, target, reduction='sum')*weight
                    result['ground_class_weight'] += len(classes)*weight
                    result['ground_class_correct'] += (logits.argmax(-1)==target).sum()*weight
                if data['query_position'] is not None:
                    keys = h[data['key_positions']]
                    query = h[data['query_position']]
                    logits = self.key(keys) @ self.query(query) / math.sqrt(self.key.out_features)
                    target = torch.tensor(data['positive'], dtype=torch.bool, device=h.device)
                    weight = data['query_weight']
                    result['ground_pointer_sum'] += (torch.logsumexp(logits,0)-torch.logsumexp(logits[target],0))*weight
                    result['ground_pointer_weight'] += weight
                    result['ground_pointer_correct'] += target[logits.argmax()].float()*weight
        return result


def annotate_learning(documents, recipe, classes):
    """须在完整更新首次collate前调用；微批/OOM重试共享固定分母。"""
    balanced = recipe.get('text_aux_normalization') == 'task_document_buckets_v1'
    grounding = bool(recipe.get('visual_grounding'))
    if not balanced and not grounding: return
    from src.text_normalization import normalize_text
    class_index = {normalize_text(label).text:i for i,label in enumerate(classes)}
    if len(class_index)!=len(classes):raise ValueError('对象类别规范化后发生重名')
    task_counts = Counter(d.metadata['task'] for d in documents)
    task_weights = recipe.get('text_aux_task_weights', recipe['loss_weights'])
    for doc in documents:
        seq = doc.to_sequence(); size = len(seq)
        target = np.asarray([e.targetable for e in seq], bool); target[0] = False
        task = doc.metadata['task']
        if balanced:
            labels = doc.metadata['_text_aux_targets']
            valid = labels['valid'] & target
            kinds = np.asarray([e.kind for e in seq])
            buckets = [valid & (kinds=='text') & labels['first'],
                       valid & (kinds=='text') & ~labels['first'],
                       valid & (kinds!='text')]
            present = sum(bool(x.any()) for x in buckets)
            weight = task_weights[task]/task_counts[task]
            shape = np.zeros(size, np.float32); space = np.zeros(size, np.float32)
            for mask in buckets:
                if mask.any(): shape[mask] = weight/max(1,present)/mask.sum()
            mask = valid & np.isin(kinds, ('text','bos','eop','eod'))
            if mask.any(): space[mask] = weight/mask.sum()
            doc.metadata['_learning_weights'] = dict(shape=shape, space=space)
        if not grounding: continue
        sketch_positions = [i for i,e in enumerate(seq[:-1]) if e.kind=='sketch']
        owners = doc.metadata.get('stroke_owner')
        objects = doc.metadata.get('objects')
        if owners is None and sketch_positions and task in ('O1','O2'):
            owners = [0]*len(sketch_positions)
            objects = [dict(instance_id=0,label=doc.metadata['label'])]
        data = dict(object_positions=[], object_classes=[], object_weight=0.,
                    query_position=None, key_positions=[], positive=[], query_weight=0.)
        if owners is not None and objects and len(owners)==len(sketch_positions):
            for obj in objects:
                positions = [p for p,o in zip(sketch_positions,owners) if o==obj['instance_id']]
                label=normalize_text(obj['label']).text
                if positions and label in class_index:
                    data['object_positions'].append(positions[-1])
                    data['object_classes'].append(class_index[label])
            data['object_weight'] = 1/max(1,len(data['object_classes']))
            target_id = doc.metadata.get('target_instance_id', doc.metadata.get('query',{}).get('target'))
            boundary = next((i for i,e in enumerate(seq) if e.targetable), 0)
            keys = [(p,o) for p,o in zip(sketch_positions,owners) if p < boundary]
            positive = [o==target_id for _,o in keys]
            if target_id is not None and boundary > 0 and any(positive):
                data.update(query_position=boundary-1,key_positions=[p for p,_ in keys],
                            positive=positive, query_weight=1.)
        doc.metadata['_grounding_targets'] = data
    object_docs = sum(bool(d.metadata['_grounding_targets']['object_positions']) for d in documents) if grounding else 0
    query_docs = sum(d.metadata['_grounding_targets']['query_position'] is not None for d in documents) if grounding else 0
    if grounding:
        for doc in documents:
            data=doc.metadata['_grounding_targets']
            data['object_weight'] /= max(1,object_docs)
            data['query_weight'] /= max(1,query_docs)


def grounding_loss(result, recipe):
    zero=result['pred_sum']*0
    spec=recipe.get('visual_grounding') or {}
    return spec.get('class_weight',.5)*result.get('ground_class_sum',zero)+spec.get('pointer_weight',.5)*result.get('ground_pointer_sum',zero)


def install_grounding(model, optimizer, spec, classes):
    """只在旧组恢复后新增组；同阶段恢复由cfg在构造时创建头。"""
    if model.visual_grounding is not None: raise ValueError('新阶段接头时原模型已包含对象辅助头')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(spec.get('seed',42))
        model.install_grounding(len(classes), spec.get('key_dim',128))
    model.grounding_classes = list(classes)
    optimizer.add_param_group(dict(params=model.visual_grounding.parameters(), role='visual_grounding',
                                   lr=optimizer.param_groups[0]['lr']))


def calibrate_learning(model, documents, body, control, recipe, device, loss_fn, move_batch,
                       rng_state, restore_rng):
    """文字与对象均在同一完整h叶子上测梯度；条件位置的零梯度也进入分母。"""
    from .formal_stream import iter_microbatches
    from .text_auxiliary import auxiliary_loss, space_target_count
    snapshot=rng_state(); rows=[]
    probe={**recipe,'padded_stroke_budget':min(8192,recipe['padded_stroke_budget'])}
    selected=[]
    for task in recipe['input_shares']:
        candidates=[d for d in documents if d.metadata['task']==task]
        candidates.sort(key=lambda d:d.metadata['_grounding_targets']['query_position'] is None)
        selected.extend(candidates[:2])
    selected.sort(key=lambda d:d.stroke_count)
    count=sum(sum(v and e.targetable for v,e in zip(d.metadata['_text_aux_targets']['valid'][1:],d.to_sequence()[1:])) for d in documents)
    try:
        for cpu in iter_microbatches(selected,probe,max_strokes=model.cfg.max_context_strokes,pin=device.type=='cuda'):
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                result=model(move_batch(cpu,device),auxiliary_probe=True)
            h=result['probe_hidden']
            geo=loss_fn(result,body,control,{**recipe,'grounding_alpha':0.},auxiliary_count=count,auxiliary_weight=0.)
            losses=[geo,auxiliary_loss(result,count,space_target_count(documents)),grounding_loss(result,recipe)]
            gradients=[torch.autograd.grad(loss,h,retain_graph=True)[0].float() for loss in losses]
            rms=[float(g.square().mean().sqrt()) for g in gradients]
            if rms[0] > 0 and rms[1] > 0 and all(math.isfinite(x) for x in rms):
                rows.append(dict(geometry_rms=rms[0],auxiliary_rms=rms[1],grounding_rms=rms[2],ratio=rms[0]/rms[1],
                                 targets=int(result['aux_all_count']),hidden_positions=h.numel()//h.shape[-1]))
            if len(rows)>=recipe.get('text_aux_probe_batches',8): break
    finally:
        model.zero_grad(set_to_none=True); restore_rng(snapshot)
    ground=[r['geometry_rms']/r['grounding_rms'] for r in rows if r['grounding_rms']>0]
    if not rows or not ground: raise RuntimeError('本批没有可校准的文字/对象梯度')
    alpha=recipe.get('text_aux_gradient_ratio',.1)*float(np.median([r['ratio'] for r in rows]))
    recipe['grounding_alpha']=recipe['visual_grounding'].get('gradient_ratio',.05)*float(np.median(ground))
    return dict(alpha=alpha,grounding_alpha=recipe['grounding_alpha'],probe_rows=rows,
                probe_tasks=[d.metadata['task'] for d in selected],
                probe_target_documents=sum(d.metadata['_grounding_targets']['query_position'] is not None for d in selected),
                target_ratio=recipe.get('text_aux_gradient_ratio',.1),
                gradient_target='auxiliary_to_geometry_common_full_hidden_RMS', optimizer_updates=0,
                preserves_weights_optimizer_RNG_and_documents=True)
