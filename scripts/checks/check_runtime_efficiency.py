"""2026-10-24：预取提交/回滚、变长E、静态KV后缀和采样兼容性短检查。"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys
import subprocess
import time
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.st.formal_prefetch import PreparedUpdates
from src.st.formal_stream import InputDocumentStream, collect_documents, iter_microbatches
from src.st.pipeline_pretrain import build_document_dataset, collate_documents
from src.st.pretrain_config import gpt2_small_pretrain_config
from src.st.pretrain_model import FrozenSketchLM
from src.st.generation_pretrain import PageGenerator, PageKVCache
from src.st.sampling import sample_gmm
from src.st.text_auxiliary import auxiliary_statistics
from src.st.text_readout import StrokeVocabulary
from src.st.text_shape_readout import ShapeVocabulary
from scripts.checks.check_joint_pipeline import signature


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir',type=Path,default=ROOT/'tmp_smoke/runtime_efficiency_20261024')
    parser.add_argument('--checkpoint',type=Path,help='可选：既有实际R，只做短推理比较')
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(42)
    data_args=json.loads((ROOT/'tmp_smoke/formal_check_20261003/data_args.json').read_text())
    dataset=build_document_dataset(**data_args)
    recipe=json.loads((ROOT/'configs/r_joint_scratch_20261022.json').read_text())
    recipe.update(input_budget=512,microbatch_documents=8,padded_stroke_budget=2048,
                  encoder_length_buckets=[16,32,48,64,76])
    catalog=ShapeVocabulary(StrokeVocabulary(dataset.font),include_non_symbol=True)
    def source():
        return InputDocumentStream(dataset,data_args,recipe['input_shares'],workers=0)
    serial=source();ahead=PreparedUpdates(source(),recipe,catalog,pin=False)
    report={}
    try:
        boundary=ahead.state_dict()
        expected=collect_documents(serial,512)
        actual=ahead.take(512)
        assert [signature(d) for d in expected[0]] == [signature(d) for d in actual[0]]
        assert actual[1:4] == expected[1:]
        assert ahead.state_dict()==serial.state_dict()
        public=ahead.state_dict();ahead.future.result()
        assert ahead.state_dict()==public and ahead.source.state_dict()!=public
        ahead.rewind(boundary)
        repeated=ahead.take(512)
        assert [signature(d) for d in actual[0]]==[signature(d) for d in repeated[0]]
        restored=source();restored.load_state_dict(public)
        next_expected=collect_documents(restored,512)
        next_actual=ahead.take(512)
        assert [signature(d) for d in next_expected[0]]==[signature(d) for d in next_actual[0]]
        assert ahead.state_dict()==restored.state_dict()
        tail_boundary=ahead.state_dict();tail_source=source();tail_source.load_state_dict(tail_boundary)
        tail=collect_documents(tail_source,128);taken=ahead.take(128)
        assert [signature(d) for d in tail[0]]==[signature(d) for d in taken[0]]
        report['prefetch']=dict(geometry=True,public_cursor=True,rewind=True,restore=True,tail_budget=True)
    finally:
        ahead.close();serial.close()
    cfg=gpt2_small_pretrain_config()
    cfg.enc_d_model=32;cfg.enc_dff=64;cfg.enc_heads=4;cfg.enc_layers=2
    cfg.r_d_model=32;cfg.r_dff=64;cfg.r_heads=4;cfg.r_layers=2
    model=FrozenSketchLM(cfg,activation_checkpoint=False).eval()
    docs=actual[0][:3]
    plain=collate_documents(docs);bucket=collate_documents(docs,encoder_length_buckets=[16,32,48,64,76])
    with torch.no_grad():
        old=model.encode_valid(plain);new=model.encode_valid(bucket)
    torch.testing.assert_close(old,new,atol=3e-6,rtol=3e-6)
    report['encoder_bucket_max_abs']=float((old-new).abs().max())
    emb=torch.randn(1,15,cfg.latent_units);pages=torch.arange(15)[None]//4
    with torch.no_grad():
        full=model.hidden_states(emb,pages,torch.zeros_like(pages,dtype=torch.bool),right_padded=True)
        cache=PageKVCache(model)
        cache.forward(emb[:,:5],pages[:,:5],prefill=True)
        pointers=[(k.data_ptr(),v.data_ptr()) for k,v in cache.entries]
        block=cache.forward(emb[:,5:10],pages[:,5:10])
        torch.testing.assert_close(block,full[:,5:10],atol=3e-6,rtol=3e-6)
        clone=cache.fork();clone.truncate(7)
        changed=emb.clone();changed[:,7:]+=0.1
        suffix=clone.forward(changed[:,7:],pages[:,7:])
        expected=model.hidden_states(changed,pages,torch.zeros_like(pages,dtype=torch.bool),right_padded=True)
        torch.testing.assert_close(suffix,expected[:,7:],atol=3e-6,rtol=3e-6)
        assert cache.length==10 and pointers==[(k.data_ptr(),v.data_ptr()) for k,v in cache.entries]
    report['static_kv']=dict(block_causal=True,suffix_rebuild=True,independent_fork=True,stable_storage=True)
    mu=torch.randn(1,8,48);sigma=torch.rand_like(mu)*.1;pi=torch.randn(1,8).softmax(-1)
    old_rng=torch.Generator().manual_seed(12);new_rng=torch.Generator().manual_seed(12)
    for _ in range(10):
        index=int(torch.multinomial(pi[0],1,generator=old_rng))
        expected=mu[:,index]+sigma[:,index]*torch.randn((1,48),generator=old_rng)*.5
        sampled,component=sample_gmm(mu,sigma,pi,generator=new_rng)
        assert int(component[0])==index and torch.equal(sampled,expected)
    deterministic,_=sample_gmm(mu,sigma,pi,generator=new_rng,mixture_temperature=0,noise_temperature=0)
    assert torch.equal(deterministic,mu[:,int(pi[0].argmax())])
    report['sampling']=dict(legacy_seed_exact=True,independent_zero_temperatures=True)
    for kinds in [torch.tensor([0,1,3,2]),torch.tensor([1,2,1,2])]:
        logits=torch.randn(4,7,requires_grad=True);spaces=torch.randn(4,requires_grad=True)
        targets=torch.tensor([1,2,3,4]);truth=torch.tensor([1.,0.,0.,0.]);first=torch.tensor([True,False,False,False])
        current=auxiliary_statistics(dict(shape=logits,space=spaces),targets,truth,first,kinds,detailed=False)
        symbols=(kinds==0)|(kinds>=3)
        reference=torch.nn.functional.cross_entropy(logits,targets,reduction='sum')
        reference=reference+torch.nn.functional.binary_cross_entropy_with_logits(spaces[symbols],truth[symbols],reduction='sum')
        value=current['aux_all_ce_sum']+current['aux_all_bce_sum']
        torch.testing.assert_close(value,reference)
        g1=torch.autograd.grad(value,(logits,spaces),retain_graph=True)
        g2=torch.autograd.grad(reference,(logits,spaces))
        for x,y in zip(g1,g2):torch.testing.assert_close(x,y)
    report['auxiliary_mask_gradient']=True
    if torch.cuda.is_available():
        model=model.cuda();generator=PageGenerator(model)
        strokes=[e.rawxy for e in docs[0].to_sequence()[:8]]
        with generator.autocast():
            a=generator.encode(strokes);cached=generator.encode_cached(strokes)
            assert torch.equal(a,cached)
            curve=generator.decode_tensor(a)
            direct=generator.encode_tensor(curve);roundtrip=generator.encode(list(curve.cpu().numpy()))
            torch.testing.assert_close(direct,roundtrip,atol=0,rtol=0)
        report['fp16_gpu_curve_feedback_exact']=True
        from src.st.formal_capacity import backward_documents
        from src.st.formal_recipe import TASKS
        from src.st.train_formal import gpu_batch,task_loss
        from src.st.train_pretrain import _rng_state,_restore_rng
        model.train();reference=deepcopy(model)
        optimizer=torch.optim.Adam(model.r_pred.parameters(),lr=1e-4)
        scaler=torch.amp.GradScaler('cuda',init_scale=256)
        retry_recipe={**recipe,'encoder_length_buckets':[],'microbatch_documents':1,'padded_stroke_budget':8192}
        docs,inputs,body,control=actual[:4]
        snapshot=_rng_state();reference_loss=0.
        for cpu in iter_microbatches(docs,retry_recipe,pin=True):
            with torch.autocast('cuda',dtype=torch.float16):
                result=reference(gpu_batch(cpu,'cuda'));loss=task_loss(result,body,control,retry_recipe)
            scaler.scale(loss).backward();reference_loss+=float(loss.detach())
        _restore_rng(snapshot)
        calls=0;injected=False
        def moving(batch,device):
            nonlocal calls,injected
            calls+=1
            if calls==2:
                injected=True
                raise torch.cuda.OutOfMemoryError('定点验证同批重试')
            return gpu_batch(batch,device)
        sums,total,finite,microbatches=backward_documents(model,optimizer,scaler,docs,body,control,
            retry_recipe,torch.device('cuda'),task_loss,moving,TASKS,_rng_state,_restore_rng)
        assert injected and finite and abs(total-reference_loss)<1e-5
        for p,q in zip(model.r_pred.parameters(),reference.r_pred.parameters()):
            torch.testing.assert_close(p.grad,q.grad,atol=0,rtol=0)
        report['fp16_oom_retry']=dict(same_gradients=True,same_loss=True,microbatches=microbatches)
        from src.st.text_auxiliary import validate_auxiliary
        from unittest.mock import patch
        model.install_text_aux(len(catalog),hidden=0)
        model.cache_validation_embeddings=True
        validation_path=args.out_dir/'validation_fixture.jsonl'
        validation_path.write_text('\n'.join(json.dumps(dict(text=t,source_id=i))
            for i,t in enumerate(['A little cat.', 'It was happy!'])),encoding='utf-8')
        validation_args=(model,dataset.font,catalog,validation_path,{},torch.device('cuda'),_rng_state,_restore_rng)
        first=validate_auxiliary(*validation_args)
        with (patch.object(model.encoder,'forward',side_effect=AssertionError('验证缓存仍调用了E')),
              patch.object(model,'hidden_states',wraps=model.hidden_states) as current_r):
            second=validate_auxiliary(*validation_args)
        assert second['frozen_E_cache_reused'] and current_r.call_count==second['windows']
        for key in ['stories','windows','content_nll','control_nll','text_aux']:
            assert first[key]==second[key],key
        report['validation_cache']=dict(frozen_E_skipped=True,current_R_evaluated=True,metrics_exact=True)
        del reference,optimizer,scaler,model,generator
        torch.cuda.empty_cache()
    if args.checkpoint:
        from scripts.demo.demo_pretrain_generation import load_generator,text_prefix
        from scripts.review.review_r_text_aux import load_joint_readout
        from src.st.text_readout_generation import generate_text_readout
        def old_module(path,name):
            code=subprocess.check_output(['git','show','67e7d4f:'+path],cwd=ROOT,text=True,encoding='utf-8')
            module=types.ModuleType(name);sys.modules[name]=module
            exec(compile(code,path,'exec'),module.__dict__)
            return module
        old_generation=old_module('src/st/generation_pretrain.py','src.st._before_generation')
        generator,font,_=load_generator(args.checkpoint,'cuda')
        old_generator=old_generation.PageGenerator(generator.model)
        prefix=text_prefix('O1','cat',font)
        comparisons=[]
        for mode in ['greedy','sample']:
            before=old_generator.generate(prefix,mode=mode,max_new_strokes=32)
            after=generator.generate(prefix,mode=mode,max_new_strokes=32)
            same=len(before['strokes'])==len(after['strokes']) and all(
                x['kind']==y['kind'] and np.array_equal(x['points'],y['points'])
                for x,y in zip(before['strokes'],after['strokes']))
            pairs=list(zip(before['strokes'],after['strokes']))
            max_abs=max(float(np.max(np.abs(np.asarray(x['points'])-np.asarray(y['points'])))) for x,y in pairs)
            decisions_equal=len(before['strokes'])==len(after['strokes']) and all(
                x['kind']==y['kind'] and x.get('component')==y.get('component') for x,y in pairs)
            assert decisions_equal and max_abs<2e-4,(mode,max_abs)
            comparisons.append(dict(mode=mode,points_exact=same,point_max_abs=max_abs,decisions_equal=decisions_equal,
                                    before_seconds=before['elapsed_seconds'],
                                    after_seconds=after['elapsed_seconds'],stop=after['stop_reason']))
        selected=generator.generate(prefix,mode='sample',max_new_strokes=8,candidate_count=3,
                                    mixture_temperature=.7,noise_temperature=.2,top_p=.95)
        assert selected['generated_strokes']>0 and all(np.isfinite(r['points']).all() for r in selected['strokes'])
        report['actual_curves']=comparisons
        del old_generator,generator;torch.cuda.empty_cache()
        generator,head,vocab,layout,_=load_joint_readout(args.checkpoint)
        old_text=old_module('src/st/text_readout_generation.py','src.st._before_text')
        prompt='Once upon a time, there was a little cat named Tom.'
        options=types.SimpleNamespace(generation_strokes=120,constraint='prefix',beam_width=1,batch_forced=True)
        start=time.monotonic()
        before=old_text.generate_text_readout(generator,head,vocab,prompt,time.monotonic()+300,options,layout)
        before_seconds=time.monotonic()-start
        start=time.monotonic()
        after=generate_text_readout(generator,head,vocab,prompt,time.monotonic()+300,options,layout)
        report['actual_text']=dict(same_text=before['generated_page_texts']==after['generated_page_texts'],
            same_points=before['strokes']==after['strokes'],layout_refreshes=after['layout_refreshes'],
            batched_forced_strokes=after['batched_forced_strokes'],before_seconds=before_seconds,
            seconds=time.monotonic()-start)
        assert after['font_audit']['valid'] if 'valid' in after['font_audit'] else not after['invalid_part_transitions']
        beam=generate_text_readout(generator,head,vocab,prompt,time.monotonic()+300,
            types.SimpleNamespace(generation_strokes=24,beam_width=2,beam_length_penalty=1.),layout)
        assert beam['invalid_part_transitions']==0
        report['beam']=dict(width=beam['beam_width'],strokes=beam['generated_strokes'],font_audit=beam['font_audit'])
        (args.out_dir/'text_greedy.json').write_text(json.dumps(after,ensure_ascii=False),encoding='utf-8')
        (args.out_dir/'text_beam.json').write_text(json.dumps(beam,ensure_ascii=False),encoding='utf-8')
    (args.out_dir/'checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
