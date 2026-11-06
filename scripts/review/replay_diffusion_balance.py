"""从稳定段日志和固定真实更新回放任务权重、目标数与不同块长的填充开销。"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from src.st.formal_recipe import TASKS
from src.st.page_diffusion_runtime import write_json
from src.st.text_readout import StrokeVocabulary
from src.text_renderer import VectorFont


NEW_EXPOSURE = dict(T1=.20,O1=.15,O2=.15,S1=.08,S2=.12,S3=.10,S4=.10,S5=.10)
TARGET_BODY_SHARE = dict(T1=.15,O1=.12,O2=.15,S1=.15,S2=.12,S3=.08,S4=.08,S5=.15)
FONT = 'E:/Trash_things/cose/datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg'
FIXTURE = 'E:/Trash_things/cose/tmp_smoke/r_page_diffusion_performance_20261102/actual_batch.pt'


def array_by_task(values):
    return np.array([values[t] for t in TASKS])


def as_tasks(values):
    return {t:float(value) for t,value in zip(TASKS,values)}


def task_losses(rows):
    counts = np.asarray([[r['counts']['body'].get(t,0) for t in TASKS] for r in rows])
    sums = np.asarray([[r['body_sum:'+t] for t in TASKS] for r in rows])
    if not (counts>0).all():
        raise ValueError('回放窗口包含缺失任务')
    return sums/counts,counts


def balance(run):
    recipe = json.loads((run/'recipe.json').read_text(encoding='utf8'))
    logs = [json.loads(line) for line in (run/'train.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    stable = json.loads((run/'pre_decay.json').read_text(encoding='utf8'))['successful_inputs']
    earlier = [r for r in logs if stable-100_000_000<r['successful_inputs']<=stable-50_000_000]
    later = [r for r in logs if stable-50_000_000<r['successful_inputs']<=stable]
    prior = array_by_task(recipe['loss_weights'])
    reference = task_losses(earlier)[0].mean(0)
    scale = float(reference@prior)
    equal = scale/(len(TASKS)*reference)
    proposed = scale*array_by_task(TARGET_BODY_SHARE)/reference
    exposure_ratio = array_by_task(NEW_EXPOSURE)/array_by_task(recipe['input_shares'])
    views = {}
    for name,rows in (('calibration',earlier),('heldout_stable',later),('final_200',logs[-200:])):
        mse,counts = task_losses(rows)
        old = mse.mean(0)*prior
        even = mse.mean(0)*equal
        new = mse.mean(0)*proposed
        # 新曝光按单位原始输入的产出率推算。输出端平方梯度与w²*MSE/N成正比。
        projected_counts = counts*exposure_ratio
        output_gradient = (proposed**2*mse/projected_counts).mean(0)
        equal_output_gradient = (equal**2*mse/projected_counts).mean(0)
        views[name] = dict(steps=[rows[0]['step'],rows[-1]['step']],updates=len(rows),
            old_share=as_tasks(old/old.sum()),equal_share=as_tasks(even/even.sum()),new_share=as_tasks(new/new.sum()),
            target_count=as_tasks(counts.mean(0)),new_target_count_estimate=as_tasks(projected_counts.mean(0)),
            old_body_loss=float(old.sum()),new_body_loss=float(new.sum()),
            equal_output_gradient_share_estimate=as_tasks(equal_output_gradient/equal_output_gradient.sum()),
            new_output_gradient_share_estimate=as_tasks(output_gradient/output_gradient.sum()))
    return dict(reference_mse=as_tasks(reference),old_body_scale=scale,equal_coefficients=as_tasks(equal),
                proposed_coefficients=as_tasks(proposed),target_body_share=TARGET_BODY_SHARE,
                actual_exposure=recipe['input_shares'],proposed_exposure=NEW_EXPOSURE,views=views)


def query_cost(fixture):
    results = {}
    docs = fixture['documents']
    for width in (4,8,16):
        blocks = padding = real = padded_groups = 0
        kinds = Counter()
        for doc in docs:
            seq = doc.to_sequence()
            for part in doc.metadata['_page_diffusion']['blocks']:
                ids = [i for i in part['target'] if i>=0]
                original = []
                for j in range(0,len(ids),width):
                    chunk = ids[j:j+width]
                    assert all(seq[i].page_id==part['page'] for i in chunk)
                    if len(chunk)<width:
                        assert seq[chunk[-1]].kind=='eop'
                        padded_groups += 1
                    original.extend(chunk)
                    blocks += 1; padding += width-len(chunk); real += len(chunk)
                    kinds.update(seq[i].kind for i in chunk)
                assert original==ids
        results[width] = dict(blocks=blocks,noisy_slots=blocks*width,padding=padding,real=real,
                              padded_groups=padded_groups,real_kinds=dict(kinds))
    assert results[16]['padding']==fixture['counts']['padded']
    assert results[16]['blocks']==fixture['counts']['blocks']
    assert results[4]['real_kinds']==results[8]['real_kinds']==results[16]['real_kinds']
    return dict(documents=len(docs),raw_inputs=sum(fixture['inputs'].values()),widths=results)


def text_span(review):
    vocab = StrokeVocabulary(VectorFont.from_svg(FONT))
    docs = torch.load(review/'data_uniform_text/valid_T1.pt',map_location='cpu',weights_only=False,mmap=True)
    rng = np.random.default_rng(20261105)
    samples = {b:[] for b in (1,4,8,16)}
    for doc in docs:
        parts = [b for b in doc['blocks'] if all(i>=0 and int(doc['kinds'][i])==0 for i in b['target'])]
        if not parts:continue
        block = parts[int(rng.integers(len(parts)))]
        pages = sorted(set(doc['pages'][doc['kinds']==0].tolist()))
        text = doc['text'][pages.index(block['page'])]
        indices = torch.where((doc['pages']==block['page']) & (doc['kinds']==0))[0].tolist()
        offset = 0
        glyphs = []
        for ch in text:
            amount = vocab.parts.get(ch,0)
            glyphs.append(indices[offset:offset+amount])
            offset+=amount
        assert offset==len(indices)
        words = [[i for glyph in glyphs[m.start():m.end()] for i in glyph] for m in re.finditer('[A-Za-z]+',text)]
        for width,values in samples.items():
            ids = set(block['target'][:width])
            whole = sum(1 for glyph in glyphs if glyph and set(glyph)<=ids)
            complete_words = sum(1 for word in words if word and set(word)<=ids)
            cut = any(max(ids) in g[:-1] for g in glyphs)
            values.append((whole,complete_words,cut))
    results = {}
    for width,values in samples.items():
        v=np.asarray(values)
        results[width]=dict(documents=len(v),complete_glyphs_mean=float(v[:,0].mean()),
            complete_glyphs_median=float(np.median(v[:,0])),at_least_one_word=float((v[:,1]>0).mean()),
            end_inside_glyph=float(v[:,2].mean()))
    return results


def text_evidence(root,review):
    result={}
    for task in ('T1','O2','S1','S3','S5'):
        path=root if task=='T1' else root/task
        summary=json.loads((path/'text_stage.json').read_text(encoding='utf8'))
        free=json.loads((path/'text_free_rows.json').read_text(encoding='utf8'))
        rows=[r for r in free if r['changed']>0]
        counts=np.array([r['effects'] for r in rows])
        rng=np.random.default_rng(20261105)
        ix=rng.integers(len(rows),size=(1200,len(rows)))
        band_totals=counts[ix].sum(1)
        rates=band_totals[:,:,0]/band_totals[:,:,1]
        rate=counts[:,:,0].sum(0)/counts[:,:,1].sum(0)
        source=review/('metrics_uniform_text' if task=='T1' else 'metrics')/'final_raw'/f'valid_{task}.npz'
        with np.load(source) as raw:
            t=raw['times'];mse=raw['full'][:,:,0].mean(0)
        mass=[]
        for lo in (0.,.25,.5,.75):
            keep=(t>=lo)&(t<=lo+.25)
            mass.append(np.trapezoid(mse[keep],t[keep]))
        with np.load(path/'text_readout_gradients.npz') as raw:
            folds=raw['folds'].astype(np.float64);axes=raw['axes'].astype(np.float64)
        total=folds.sum(0)
        project=axes@axes.T
        energy=np.square(axes.T@total).sum((-1,-2))/np.square(total).sum((-1,-2))
        angles=defaultdict(list)
        for fold in folds:
            training=total[0]-fold[0]
            residual=training-project@training
            for key,validation in (('shape',fold[1]),('center',fold[2])):
                for beta in (1.,1.5,2.):
                    candidate=training+(beta-1)*residual
                    numerator=(candidate*validation).sum((-1,-2))
                    denominator=np.linalg.norm(candidate,axis=(-1,-2))*np.linalg.norm(validation,axis=(-1,-2))
                    angles[key+str(beta)].append(numerator/np.maximum(denominator,1e-20))
        result[task]=dict(times=summary['times'],input_documents=summary['documents'],
            edited_teacher=summary['teacher']['edited'],free=summary['free']['edited'],free_count=len(rows),
            high_minus_low=dict(mean=float(rate[0]-rate[-1]),ci95=np.quantile(rates[:,0]-rates[:,-1],[.025,.975]).tolist()),
            x_loss_quarters_low_to_high=(mass/np.sum(mass)).tolist(),translation_energy=energy.tolist(),
            crossfit_cosine={k:np.mean(v,axis=0).tolist() for k,v in angles.items()},
            half_boost_fold_wins={k:(np.array(angles[k+'1.5'])>np.array(angles[k+'1.0'])).sum(0).tolist()
                                 for k in ('shape','center')})
    write_json(root/'text_evidence.json',result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--summarize_text',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.summarize_text:
        text_evidence(args.out,args.review)
        return
    fixture=torch.load(FIXTURE,map_location='cpu',weights_only=False,mmap=True)
    result=dict(balance=balance(args.review/'run'),block_cost=query_cost(fixture),text_span=text_span(args.review))
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'balance_and_blocks.json',result)
    print(json.dumps(result['block_cost']))
    print(json.dumps(result['text_span']))


if __name__=='__main__':
    main()
