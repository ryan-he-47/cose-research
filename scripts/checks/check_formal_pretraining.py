"""2026-10-03：正式分布的坐标/mask/损失归一化/预取恢复核对，并出AE重建样张。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from src.formal_scene_tasks import _bbox, make_recipe, realize_recipe
from src.st.formal_assets import IndexedRawSource, prepare_raw_index, write_json
from src.st.formal_recipe import DEFAULT_RECIPE, InputWSD, TASKS
from src.st.formal_stream import InputDocumentStream, collect_documents, iter_microbatches
from src.st.pipeline_pretrain import build_document_dataset, collate_documents
from src.st.pretrain_model import FrozenSketchLM, config_from_ae_checkpoint, config_from_r_checkpoint
from src.st.train_formal import save_formal, task_loss
from src.st.train_ae import StrokeAutoencoder, config_from_checkpoint, load_ae_weights
from scripts.demo.demo_ae_reconstruction import reconstruct_elements, write_page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir',type=Path,default=ROOT/'tmp_smoke/formal_check_20261003')
    parser.add_argument('--raw_dir',type=Path,default=ROOT/'datasets/quickdraw_345_raw_1000')
    parser.add_argument('--ae',type=Path,default=ROOT/'tmp_outputs/ae_stage1_balanced_20261003/final.pt')
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    raw=out/'raw'
    raw.mkdir(exist_ok=True)
    for label in ('cat','dog','house','tree'):
        source=args.raw_dir/f'full_raw_{label}.ndjson'
        target=raw/source.name
        if not target.exists():
            os.link(source,target)
    index=out/'raw_index'
    prepare_raw_index(raw,index,max_per_class=128)
    stories=out/'stories'
    stories.mkdir(exist_ok=True)
    texts=['Once a cat met a dog. They played near a tree. The cat went home.',
           'A small bird saw a house. It flew to the tree and sang.',
           'The dog ran to the left. The cat stayed by the house.']
    offsets=[]
    with (stories/'stories.jsonl').open('wb') as stream:
        for i,text in enumerate(texts):
            offsets.append(stream.tell())
            stream.write((json.dumps({'text':text,'source_id':f'fixture:{i}'})+'\n').encode())
    np.save(stories/'offsets.npy',np.asarray(offsets,dtype=np.uint64))
    np.save(stories/'train.npy',np.arange(3,dtype=np.uint64))
    np.save(stories/'valid.npy',np.asarray([0],dtype=np.uint64))
    write_json(stories/'manifest.json',{'kind':'indexed_stories_v1','fixture':True,'count':3})
    data_args={'formal':True,'font_path':str(ROOT/'datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg'),
               'raw_dir':str(raw),'raw_index':str(index),'story_index':str(stories),
               'split':'train','scene_recipes':96,'seed':42}
    write_json(out/'data_args.json',data_args)
    dataset=build_document_dataset(**data_args)
    source=IndexedRawSource(raw,index)
    duplicates=0
    counts=Counter()
    for i in range(96):
        recipe=make_recipe(source,42,i)
        strokes,objects,owners=realize_recipe(recipe,source)
        counts[len(objects)]+=1
        duplicates+=len({obj['label'] for obj in objects}) < len(objects)
        assert len(set(obj['source_index'] for obj in objects)) == len(objects)
        for obj in objects:
            start,stop=obj['stroke_range']
            np.testing.assert_allclose(_bbox(strokes[start:stop]),obj['bbox'],atol=2e-6)
            points=np.concatenate(strokes[start:stop])
            x0,y0,x1,y1=obj['selection_box']
            norm=((points-[(x0+x1)/2,(y0+y1)/2])/[(x1-x0)/2,(y1-y0)/2])**2
            assert np.max(norm.sum(1)) < 1.001
            assert np.min(points) >= 0 and np.max(points) <= 1
    assert duplicates and set(counts) == {2,3,4}
    documents=[dataset.get_for_task(task,0) for task in TASKS]
    scene_objects=[doc.metadata['objects'] for doc in documents if doc.metadata['task'].startswith('S')]
    assert all(objects == scene_objects[0] for objects in scene_objects)
    for document in documents:
        sequence=document.to_sequence()
        assert sequence[-2].kind == 'eod' and sequence[-1].kind == 'eop'
        for page in document.pages:
            assert len({e.page_id for e in page.elements}) == 1
    recipe={**DEFAULT_RECIPE,'input_budget':2048,'microbatch_documents':8,'padded_stroke_budget':8192,'workers':0}
    stream=InputDocumentStream(dataset,data_args,recipe['input_shares'],workers=0)
    _,inputs,_,_=collect_documents(stream,recipe['input_budget'])
    saved=stream.state_dict()
    expected=stream.next_document()
    restored=InputDocumentStream(dataset,data_args,recipe['input_shares'],workers=2)
    restored.load_state_dict(saved)
    restored.prefetch()
    replayed=restored.next_document()
    a,b=collate_documents([expected]),collate_documents([replayed])
    assert all(torch.equal(value,b[name]) for name,value in a.items() if torch.is_tensor(value))
    restored.close()
    initial=torch.load(args.ae,map_location='cpu',weights_only=False)
    cfg=replace(config_from_ae_checkpoint(initial),r_layers=2,r_d_model=48,r_heads=4,r_dff=96)
    model=FrozenSketchLM(cfg,encode_chunk_size=128,activation_checkpoint=False)
    model.load_ae(initial)
    optimizer=torch.optim.Adam(model.r_pred.parameters(),lr=1e-4)
    scheduler=InputWSD(optimizer,recipe,1.757363402156491e-5)
    assert scheduler.lr_at(8000000) == 1e-4
    assert scheduler.lr_at(192000000) == 1e-4
    assert abs(scheduler.lr_at(200000000)-1e-5) < 1e-12
    counts_body=Counter()
    control=0
    for doc in documents:
        for element in doc.to_sequence()[1:]:
            if element.targetable:
                if element.kind in ('eod','eop','bos'):
                    control+=1
                else:
                    counts_body[doc.metadata['task']]+=1
    model.train()
    total_loss=0.
    for cpu in iter_microbatches(sorted(documents,key=lambda d:d.stroke_count),recipe):
        result=model(cpu)
        loss=task_loss(result,counts_body,control,recipe)
        assert torch.isfinite(loss)
        loss.backward()
        total_loss+=float(loss.detach())
    assert all(p.grad is None for module in (model.encoder,model.decoder,model.dct_head) for p in module.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.r_pred.parameters() if p.grad is not None)
    gradients={name:p.grad.clone() for name,p in model.r_pred.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    for cpu in iter_microbatches(sorted(documents,key=lambda d:d.stroke_count),{**recipe,'microbatch_documents':1}):
        result=model(cpu)
        task_loss(result,counts_body,control,recipe).backward()
    for name,p in model.r_pred.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad,gradients[name],rtol=5e-4,atol=3e-6)
    optimizer.step()
    scheduler.step(sum(inputs.values()))
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    save_formal(out/'check.pt',model,optimizer,scheduler,scaler,stream,step=1,phase_step=1,
        counters={'successful_inputs':scheduler.inputs},data_args=data_args,manifest=dataset.manifest(),
        recipe=recipe,ae_source={'path':str(args.ae)},parent_source=None)
    state=torch.load(out/'check.pt',map_location='cpu',weights_only=False)
    copy=FrozenSketchLM(config_from_r_checkpoint(state),activation_checkpoint=False)
    copy.load_state_dict(state['model'],strict=True)
    second_optimizer=torch.optim.Adam(copy.r_pred.parameters(),lr=1e-4)
    second_optimizer.load_state_dict(state['optimizer'])
    second_schedule=InputWSD(second_optimizer,recipe,state['scheduler']['spec']['start_lr'])
    second_schedule.load_state_dict(state['scheduler'])
    assert second_schedule.inputs == scheduler.inputs
    assert all(torch.equal(v,copy.state_dict()[k]) for k,v in model.state_dict().items())
    ae=StrokeAutoencoder(config_from_checkpoint(initial)).eval()
    load_ae_weights(ae,initial)
    preview=out/'ae_preview'
    preview.mkdir(exist_ok=True)
    for task in ('S2','S5'):
        document=dataset.get_for_task(task,7)
        write_json(preview/f'{task}_metadata.json',document.metadata)
        for page in document.pages:
            elements=[element for element in document.to_sequence() if element.page_id == page.page_id]
            source,reconstruction=reconstruct_elements(ae,elements)
            write_page(source,reconstruction,elements,page,preview/f'{task}_page_{page.page_id:02d}')
    report={'checks':'passed','scene_counts':dict(counts),'duplicate_scenes':duplicates,
            'prefetch_restore_equal':True,'microbatch_gradient_equal':True,'frozen_ae':True,'checkpoint_restore_equal':True,
            'successful_inputs':scheduler.inputs,'task_body':dict(counts_body),'control_targets':control}
    write_json(out/'report.json',report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
