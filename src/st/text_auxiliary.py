"""正式R的文字母模板标签、分类统计与梯度权重校准。"""
from __future__ import annotations

import hashlib
import math
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .formal_stream import iter_microbatches


def weight_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def bind_auxiliary(model,font,font_sha256,*,head_path=None,saved_manifest=None,
                   initialize_random=False,initialization_seed=None,include_non_symbol=None):
    from .text_readout import StrokeVocabulary
    from .text_shape_readout import ShapeVocabulary
    if include_non_symbol is not None and not initialize_random:
        raise ValueError('非符号类新建开关只用于从头初始化；恢复按已保存词表')
    non_symbol = bool(include_non_symbol) if initialize_random else bool(
        saved_manifest and saved_manifest['shape_vocabulary'].get('non_symbol_class') is not None)
    catalog=ShapeVocabulary(StrokeVocabulary(font),include_non_symbol=non_symbol)
    if initialize_random:
        if head_path is not None or saved_manifest is not None:
            raise ValueError('从头辅助头不能同时加载旧头/旧来源')
        if model.text_aux is None:
            model.install_text_aux(len(catalog),model.cfg.r_text_aux_hidden)
        if model.cfg.r_text_aux_classes!=len(catalog):
            raise ValueError('从头辅助头的类别数与字体不同')
        manifest=dict(version='2026-10-14-shape-space-from-scratch',shape_vocabulary=catalog.dump(),
            font_sha256=font_sha256,head_source=None,head_file_sha256=None,
            source_R_hash=weight_hash(model),initialization='fresh_R_and_auxiliary',seed=initialization_seed,
            space_supervision='every_valid_target_including_continuations_and_controls')
    elif head_path is not None:
        state=torch.load(head_path,map_location='cpu',weights_only=False)
        if state.get('kind')!='shape_space' or state['input_width']!=model.cfg.r_d_model:
            raise ValueError('需要与R末层同宽的形状/空格头')
        if state['shape_vocabulary']!=catalog.dump() or state['font_sha256']!=font_sha256:
            raise ValueError('母模板标签表或字体来源与辅助头不同')
        original_hash=weight_hash(model)
        if original_hash!=state['r_model_hash']:
            raise ValueError('辅助头的冻结R来源与本次完整状态不同')
        model.install_text_aux(len(catalog),state['hidden'])
        model.text_aux.load_state_dict(state['head'],strict=True)
        manifest=dict(version='2026-10-14-shape-space-all-strokes',shape_vocabulary=catalog.dump(),
            font_sha256=font_sha256,head_source=str(Path(head_path).resolve()),
            head_file_sha256=hashlib.sha256(Path(head_path).read_bytes()).hexdigest(),
            source_R_hash=original_hash,space_supervision='every_valid_target_including_continuations_and_controls')
    else:
        if model.text_aux is None or saved_manifest is None:
            raise ValueError('新R缺少辅助头/标签来源记录')
        if (saved_manifest['shape_vocabulary']!=catalog.dump() or
            saved_manifest['font_sha256']!=font_sha256 or model.cfg.r_text_aux_classes!=len(catalog)):
            raise ValueError('恢复的辅助标签/字体/类别数不同')
        manifest=saved_manifest
    model.text_aux.catalog=catalog
    model.text_aux_manifest=manifest
    return catalog,manifest


def extend_non_symbol(model,optimizer,font,manifest):
    """显式追加一行，保留旧116行/空格头与Adam动量；新行Adam动量为零。"""
    from .text_readout import StrokeVocabulary
    from .text_shape_readout import ShapeVocabulary
    catalog=ShapeVocabulary(StrokeVocabulary(font),include_non_symbol=True)
    old=model.text_aux.shape_head
    if old.out_features+1!=len(catalog):raise ValueError('只允许从旧词表追加非符号类')
    rng=torch.get_rng_state()
    new=torch.nn.Linear(old.in_features,len(catalog),device='cpu',dtype=old.weight.dtype).to(old.weight.device)
    torch.set_rng_state(rng)
    with torch.no_grad():
        new.weight.zero_();new.bias.fill_(float(old.bias.mean())-2.)
        new.weight[:-1].copy_(old.weight);new.bias[:-1].copy_(old.bias)
    for before,after in ((old.weight,new.weight),(old.bias,new.bias)):
        for group in optimizer.param_groups:
            group['params']=[after if p is before else p for p in group['params']]
        previous=optimizer.state.pop(before,{})
        expanded={}
        for key,value in previous.items():
            if torch.is_tensor(value) and value.shape==before.shape:
                target=torch.zeros_like(after);target[:-1].copy_(value);expanded[key]=target
            else:expanded[key]=value.clone() if torch.is_tensor(value) else value
        if previous:optimizer.state[after]=expanded
    model.text_aux.shape_head=new;model.cfg.r_text_aux_classes=len(catalog)
    model.text_aux.catalog=catalog
    manifest={**manifest,'version':'2026-10-19-symbol-and-non-symbol',
        'shape_vocabulary':catalog.dump(),'space_supervision':'every_symbol_target; graphic_targets_masked',
        'non_symbol_extension':{'old_classes':old.out_features,'new_classes':len(catalog),
                                'preserves_old_rows_space_head_and_Adam':True}}
    model.text_aux_manifest=manifest
    return catalog,manifest


def annotate_documents(documents,catalog):
    """标签取母字形与原笔顺；只添监督侧车，不修改坐标或数据manifest。"""
    total=0
    for document in documents:
        shapes=[];spaces=[];first=[];valid=[]
        def add_control(element):
            graphic=element.kind not in catalog.controls
            label=catalog.non_symbol_class if graphic else catalog.controls[element.kind]
            shapes.append(0 if label is None else label)
            spaces.append(False);first.append(False);valid.append(label is not None)
        for element in document.prefix:add_control(element)
        for page in document.pages:
            placements={g.char_index:g for g in page.metadata.get('glyph_placements',())}
            text=page.metadata.get('rendered_text','');parts={}
            for element in page.elements:
                if element.kind!='text':
                    add_control(element);continue
                owner=element.char_index;glyph=placements[owner]
                part=parts.get(owner,0);parts[owner]=part+1
                leading=part==0 and ((owner>0 and text[owner-1].isspace()) or
                                     (owner==0 and bool(page.metadata.get('separator_before'))))
                shapes.append(catalog.identity_to_shape[glyph.character,part]);spaces.append(leading)
                first.append(part==0);valid.append(True)
        sequence=document.to_sequence()
        if len(shapes)!=len(sequence):raise AssertionError('辅助标签与实际笔画未对齐')
        document.metadata['_text_aux_targets']=dict(shape=np.asarray(shapes,np.int64),
            space=np.asarray(spaces,np.float32),first=np.asarray(first,bool),valid=np.asarray(valid,bool))
        total+=sum(v and e.targetable for v,e in zip(valid[1:],sequence[1:]))
    return total


def space_target_count(documents):
    return sum(bool(v) and element.targetable and element.kind in ('text','bos','eop','eod')
               for doc in documents for v,element in
               zip(doc.metadata['_text_aux_targets']['valid'][1:],doc.to_sequence()[1:]))


def auxiliary_loss(result,count,space_count=None):
    if 'aux_weighted_sum' in result:return result['aux_weighted_sum']
    zero=result['pred_sum']*0
    return result.get('aux_all_ce_sum',zero)/max(1,count)+result.get('aux_all_bce_sum',zero)/max(1,
        count if space_count is None else space_count)


def auxiliary_statistics(output,shape,space,first,kinds,*,detailed=True):
    symbol=(kinds==0)|(kinds>=3)
    with torch.autocast(device_type=output['shape'].device.type,enabled=False):
        ce=torch.nn.functional.cross_entropy(output['shape'].float(),shape,reduction='none')
        bce=torch.nn.functional.binary_cross_entropy_with_logits(
            output['space'].float(),space.float(),reduction='none')
        bce=torch.where(symbol,bce,0.)
    if not detailed:
        return {'aux_all_ce_sum':ce.sum(),'aux_all_bce_sum':bce.sum()}
    shape_error=output['shape'].argmax(-1)!=shape
    space_error=(output['space']>0)!=space.bool()
    result={}
    for name,mask in [('all',torch.ones_like(first)),('text',kinds==0),('first',first),
                      ('continuation',(kinds==0)&~first),('control',kinds>=3),
                      ('sketch',(kinds>0)&(kinds<3))]:
        prefix='aux_'+name+'_'
        space_mask=mask&symbol
        result.update({prefix+'count':mask.sum(),prefix+'ce_sum':torch.where(mask,ce,0.).sum(),
            prefix+'space_count':space_mask.sum(),
            prefix+'bce_sum':torch.where(space_mask,bce,0.).sum(),prefix+'shape_errors':(shape_error&mask).sum(),
            prefix+'space_errors':(space_error&space_mask).sum(),
            prefix+'space_positive':torch.where(space_mask,space,0.).sum()})
    return result


def auxiliary_report(sums):
    result={}
    for name in ('all','text','first','continuation','control','sketch'):
        p='aux_'+name+'_';n=sums.get(p+'count',0)
        if n:
            spaces=sums.get(p+'space_count',n)
            result[name]=dict(count=int(n),space_count=int(spaces),shape_ce=sums[p+'ce_sum']/n,
                space_bce=sums[p+'bce_sum']/spaces if spaces else None,shape_error=sums[p+'shape_errors']/n,
                space_error=sums[p+'space_errors']/spaces if spaces else None,space_positive=int(sums[p+'space_positive']))
    return result


def preflight_auxiliary(model,font,catalog,device,rng_state,restore_rng):
    from src.page_document import build_document,layout_text_pages
    from .pipeline_pretrain import collate_documents
    snapshot=rng_state()
    document=build_document(layout_text_pages('A little cat sat. It was happy!',font),
                            complete=True,metadata={'task':'T1'})
    count=annotate_documents([document],catalog)
    cpu=collate_documents([document],max_strokes=model.cfg.max_context_strokes)
    batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in cpu.items()}
    optimizer=torch.optim.SGD([*model.r_pred.parameters(),*model.text_aux.parameters()],lr=0.)
    scaler=torch.amp.GradScaler('cuda',init_scale=256,enabled=device.type=='cuda')
    try:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            result=model(batch)
            loss=(result['aux_all_ce_sum']+result['aux_all_bce_sum'])/count
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        core=[p.grad for p in model.r_pred.parameters() if p.grad is not None]
        aux=[p.grad for p in model.text_aux.parameters()]
        if not core or not all(torch.isfinite(g).all() for g in core) or not any(g.abs().max()>0 for g in core):
            raise RuntimeError('分类梯度没有有限地进入R')
        if not all(g is not None and torch.isfinite(g).all() for g in aux):
            raise RuntimeError('辅助头梯度缺失/非有限')
        for module in (model.encoder,model.decoder,model.dct_head):
            if module is not None and any(p.requires_grad or p.grad is not None for p in module.parameters()):
                raise RuntimeError('冻结AE出现梯度')
        return dict(precision='FP16 forward; FP32 CE/BCE',targets=count,optimizer_updates=0,
                    auxiliary_gradient_reaches_R=True,auxiliary_head_gradient_finite=True,
                    frozen_AE_gradient_none=True,space_supervised_every_target=catalog.non_symbol_class is None,
                    space_supervised_every_symbol_target=True,graphics_space_masked=catalog.non_symbol_class is not None)
    finally:
        model.zero_grad(set_to_none=True);restore_rng(snapshot)


def configure_recalibration(recipe,ratio,interval,*,inputs,step,source):
    """显式升级旧配方；同策略恢复保留校准进度，不重置输入预算/WSD。"""
    if not math.isfinite(ratio) or ratio<=0 or interval<=0:
        raise ValueError('辅助梯度比例和校准输入间隔必须为有限正数')
    unchanged=(recipe.get('text_aux_gradient_ratio')==ratio and
               recipe.get('text_aux_calibration_interval_inputs')==interval)
    transition=dict(source_checkpoint=str(source),step=step,successful_inputs=inputs,
                    previous_alpha=recipe.get('text_aux_alpha'),target_ratio=ratio,
                    interval_inputs=interval,policy_changed=not unchanged)
    if not unchanged:
        recipe.update(text_aux_gradient_ratio=ratio,text_aux_calibration_interval_inputs=interval,
                      text_aux_calibration_pending=True,text_aux_calibration_count=0,
                      text_aux_calibration_transition=transition)
    return transition


def auxiliary_calibration_due(recipe,inputs):
    if recipe.get('text_aux_alpha') is None or recipe.get('text_aux_calibration_pending',False):
        return True
    interval=recipe.get('text_aux_calibration_interval_inputs')
    return interval is not None and inputs-recipe.get('text_aux_calibration_last_inputs',0)>=interval


def apply_auxiliary_calibration(recipe,calibration,*,inputs,step):
    alpha=calibration['alpha']
    if not math.isfinite(alpha) or alpha<=0:
        raise RuntimeError(f'辅助校准α无效：{alpha}')
    result={**calibration,'step':step,'successful_inputs':inputs,
            'previous_alpha':recipe.get('text_aux_alpha'),
            'calibration_index':recipe.get('text_aux_calibration_count',0)+1}
    for row in result['probe_rows']:
        row['weighted_ratio_after']=alpha/row['ratio']
        row['weighted_ratio_before']=(result['previous_alpha']/row['ratio']
                                      if result['previous_alpha'] is not None else None)
    recipe.update(text_aux_alpha=alpha,text_aux_calibration_pending=False,
                  text_aux_calibration_last_inputs=inputs,text_aux_calibration_last_step=step,
                  text_aux_calibration_count=result['calibration_index'])
    return result


def calibrate_auxiliary(model,documents,body,control,recipe,device,loss_fn,move_batch,rng_state,restore_rng):
    """R在线前向，FP32头/leaf h测梯度；不更新权重/Adam，保留随机状态。"""
    if model.visual_grounding is not None:
        from .visual_grounding_auxiliary import calibrate_learning
        return calibrate_learning(model,documents,body,control,recipe,device,loss_fn,move_batch,rng_state,restore_rng)
    snapshot=rng_state();rows=[]
    probe={**recipe,'padded_stroke_budget':min(8192,recipe['padded_stroke_budget'])}
    aux_count=sum(int(doc.metadata['_text_aux_targets']['valid'][1:][
        np.asarray([bool(e.targetable) for e in doc.to_sequence()[1:]])].sum()) for doc in documents)
    try:
        for cpu in iter_microbatches(documents,probe,max_strokes=model.cfg.max_context_strokes,pin=device.type=='cuda'):
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                result=model(move_batch(cpu,device),auxiliary_probe=True)
            h=result['probe_hidden']
            geo=loss_fn(result,body,control,recipe,auxiliary_count=aux_count,auxiliary_weight=0.)
            aux=auxiliary_loss(result,aux_count,space_target_count(documents))
            dg=torch.autograd.grad(geo,h,retain_graph=True)[0].float()
            da=torch.autograd.grad(aux,h)[0].float()
            g=float(dg.square().mean().sqrt());a=float(da.square().mean().sqrt())
            if not (math.isfinite(g) and math.isfinite(a) and g>0 and a>0):
                raise RuntimeError(f'辅助梯度校准无有效梯度：{g}/{a}')
            rows.append(dict(microbatch=len(rows)+1,targets=int(result['aux_all_count']),
                             geometry_rms=g,auxiliary_rms=a,ratio=g/a))
            result=h=dg=da=None
            if len(rows)>=recipe.get('text_aux_probe_batches',8):break
    finally:
        model.zero_grad(set_to_none=True);restore_rng(snapshot)
    if not rows:raise RuntimeError('辅助梯度校准没有文档')
    alpha=recipe.get('text_aux_gradient_ratio',.5)*float(np.median([r['ratio'] for r in rows]))
    return dict(alpha=alpha,probe_rows=rows,probe_precision='FP32 heads and leaf h; FP16 frozen-E/R forward',
        gradient_target='auxiliary_to_geometry_R_hidden_RMS',
        target_ratio=recipe.get('text_aux_gradient_ratio',.5),optimizer_updates=0,
        preserves_weights_optimizer_RNG_and_documents=True)


@torch.no_grad()
def validate_auxiliary(model,font,catalog,path,layout_spec,device,rng_state,restore_rng):
    """复用旧128篇故事，当前R在线计算；不使用旧h，不更新/消费训练游标。"""
    from src.page_document import build_document,layout_text_pages,PageLayoutConfig
    from .pipeline_pretrain import collate_documents
    snapshot=rng_state();training=model.training;observe=getattr(model,'observe_distribution',False)
    model.eval();model.observe_distribution=False;model.zero_grad(set_to_none=True)
    layout=dict(layout_spec)
    if 'body_box' in layout:layout['body_box']=tuple(layout['body_box'])
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    started=time.monotonic()
    cache_enabled=getattr(model,'cache_validation_embeddings',False)
    stat=Path(path).stat()
    cache_key=(str(Path(path).resolve()),stat.st_mtime_ns,stat.st_size,str(device),
               json.dumps(layout,sort_keys=True),json.dumps(catalog.dump(),sort_keys=True))
    cache=getattr(model,'_validation_cache',None)
    reused=bool(cache_enabled and cache is not None and cache[0]==cache_key)
    cached_batches=[]
    def batches():
        if reused:
            yield from cache[1]
            return
        for row in rows:
            document=build_document(layout_text_pages(row['text'],font,config=PageLayoutConfig(**layout)),
                                    complete=True,metadata={'task':'T1','source_id':row['source_id']})
            for window in document.iter_windows(model.cfg.max_context_strokes):
                annotate_documents([window],catalog)
                yield collate_documents([window],max_strokes=model.cfg.max_context_strokes)
    sums=Counter();windows=0
    try:
        for cpu in batches():
            batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in cpu.items()}
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                if cache_enabled and not reused:
                    batch['encoded_embeddings']=model.encode_valid(batch)
                result=model(batch)
            if cache_enabled and not reused:
                cached={k:v for k,v in cpu.items() if k not in ('ink','stroke_len','stroke_batch_index','encode_groups','metadata')}
                cached['encoded_embeddings']=batch['encoded_embeddings'].cpu()
                cached_batches.append(cached)
            values={k:v for k,v in result.items() if k.startswith('aux_') or
                    k in ('content_sum','control_sum','content_count','control_count')}
            for key,value in zip(values,torch.stack(list(values.values())).cpu().tolist()):sums[key]+=value
            windows+=1
        if cache_enabled and not reused:model._validation_cache=(cache_key,cached_batches)
    finally:
        model.train(training);model.observe_distribution=observe;restore_rng(snapshot)
    return dict(stories=len(rows),windows=windows,augmentation=False,features='online_current_R',
        frozen_E_cache_reused=reused,elapsed_seconds=time.monotonic()-started,
        content_nll=sums['content_sum']/max(1,sums['content_count']),
        control_nll=sums['control_sum']/max(1,sums['control_count']),text_aux=auxiliary_report(sums))
