"""2026-10-27：扩大评估的批量推理与现有单例推理逐例核对。"""
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.demo.demo_text_readout import load_joint_readout
from scripts.demo.demo_pretrain_generation import source_prefix
from scripts.review.review_joint_stage import dataset_for
from scripts.review.joint_eval_batch import text_batch,graphic_batch,RaggedKV,prefix_records
from src.st.text_readout_generation import generate_text_readout
from src.st.generation_pretrain import PageKVCache
from scripts.review.diagnose_formal_r_health import model_hash


def main():
    torch.set_num_threads(4)
    out=Path('tmp_smoke/r_joint_statistical_review_20261027');out.mkdir(parents=True,exist_ok=True)
    generator,head,vocab,layout,source=load_joint_readout('tmp_smoke/r_joint_scratch_final_review_20261026/model.pt')
    generator.model.encode_chunk_size=128
    bundle=json.loads(Path('tmp_smoke/r_joint_full_review_20261022/probe_bundle.json').read_text(encoding='utf-8'))
    dataset=dataset_for(bundle,'valid',vocab.font)
    cases=[('O2',i) for i in bundle['splits']['valid']['balanced_indices'][:8]]+[(t,i) for t in ('S3','S5') for i in range(4)]
    prefixes=[source_prefix(dataset.get_for_task(t,i),t) for t,i in cases]
    start=time.monotonic();batched=text_batch(generator,head,vocab,layout,prefixes,limit=96)
    batch_seconds=time.monotonic()-start
    rows=[];start=time.monotonic()
    for (task,index),prefix,batch in zip(cases,prefixes,batched):
        single=generate_text_readout(generator,head,vocab,'',time.monotonic()+120,
            SimpleNamespace(constraint='prefix',generation_strokes=96,batch_forced=False),layout,prefix=prefix)
        text=' '.join(single['generated_page_texts'].values()).strip()
        rows.append(dict(task=task,index=index,single=text,batch=batch['generated'],equal=text==batch['generated'],
                         single_stop=single['stop_reason'],batch_stop=batch['stop_reason']))
    single_seconds=time.monotonic()-start
    (out/'batch_text_check.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    graphics=[source_prefix(dataset.get_for_task('S4',i),'S4') for i in range(4)]
    batch=graphic_batch(generator,graphics,limit=16)
    errors=[]
    for i,(prefix,value) in enumerate(zip(graphics,batch)):
        single=generator.generate(prefix,mode='greedy',max_new_strokes=16,max_new_pages=2)
        s=[r for r in single['strokes'] if r['generated']];b=[r for r in value['strokes'] if r['generated']]
        errors.append(dict(index=i,single_count=len(s),batch_count=len(b),single_stop=single['stop_reason'],batch_stop=value['stop_reason'],
            max_coordinate_error=max((float((torch.tensor(a['points'])-torch.tensor(c['points'])).abs().max()) for a,c in zip(s,b)),default=0.),
            control_sequence_equal=[r['kind'] for r in s]==[r['kind'] for r in b]))
    assert model_hash(generator.model)==source['r_model_hash']
    result=dict(text_cases=rows,text_equal=sum(r['equal'] for r in rows),text_count=len(rows),
                batch_seconds=batch_seconds,single_seconds=single_seconds,graphics=errors,
                weights_unchanged=True,optimizer_updates=0)
    (out/'batch_equivalence.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
