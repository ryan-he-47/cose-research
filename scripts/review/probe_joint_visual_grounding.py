"""2026-10-27：冻结R的配对场景、圈选候选与线性读出诊断。"""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.demo.demo_text_readout import load_joint_readout
from scripts.demo.demo_pretrain_generation import source_prefix
from scripts.review.review_joint_stage import dataset_for, write
from scripts.review.probe_joint_scene_abilities import answer_match
from scripts.review.diagnose_formal_r_health import model_hash
from src.st.pipeline_pretrain import collate_documents, to_device
from src.st.text_readout_generation import generate_text_readout
from src.visual_grounding_scenes import paired_scene


@torch.no_grad()
def features(generator, document, candidate_kind=None):
    batch = to_device(collate_documents([document]), generator.device)
    boundary = next(i for i, e in enumerate(document.to_sequence()) if e.targetable)
    candidate = (next(i for i,e in enumerate(document.to_sequence()) if e.targetable and e.kind==candidate_kind)
                 if candidate_kind else boundary)
    with generator.autocast():
        emb = generator.model.encode_valid(batch)
        h = generator.model.hidden_states(emb[:, :candidate], batch['slice_id'][:, :candidate],
                                         batch['stroke_pad'][:, :candidate], right_padded=True)
        mu, sigma, pi = generator.model.prediction_head(h[:, -1])
    return h[0, boundary-1].float().cpu().numpy(), (mu[0], sigma[0], pi[0]), emb[0, candidate].float()


def scene_svg(document, curves=()):
    lines = ['<svg viewBox="0 0 400 400" width="360" height="360" style="background:white">']
    for e in document.to_sequence():
        if e.kind in ('sketch','selection'):
            points = ' '.join(f'{x*400:.2f},{y*400:.2f}' for x, y in e.rawxy)
            lines.append(f'<polyline points="{points}" fill="none" stroke="#334155" stroke-width="1.2"/>')
    for curve, color in curves:
        points = ' '.join(f'{x*400:.2f},{y*400:.2f}' for x, y in curve)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    return ''.join(lines) + '</svg>'


def ridge(train_x, train_y, valid_x, regularization=10.):
    # 使用训练集均值中心化；截距不惩罚。对偶解适合样本少于768维的缓存。
    xm, ym = train_x.mean(0), train_y.mean(0)
    x, y = train_x-xm, train_y-ym
    weights = np.linalg.solve(x@x.T + regularization*np.eye(len(x)), y)
    return (valid_x-xm)@x.T@weights + ym


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('tmp_smoke/r_joint_scratch_final_review_20261026/model.pt'))
    parser.add_argument('--bundle', type=Path, default=Path('tmp_smoke/r_joint_full_review_20261022/probe_bundle.json'))
    parser.add_argument('--out', type=Path, default=Path('tmp_smoke/r_visual_grounding_20261027'))
    parser.add_argument('--pairs', type=int, default=12)
    parser.add_argument('--probe_scenes', type=int, default=96)
    parser.add_argument('--probe_objects', type=int, default=345)
    args = parser.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    (out/'cases').mkdir(exist_ok=True); (out/'features').mkdir(exist_ok=True)
    torch.set_num_threads(4)
    generator, head, vocab, layout, source = load_joint_readout(args.checkpoint)
    generator.model.encode_chunk_size = 128
    bundle = json.loads(args.bundle.read_text(encoding='utf-8'))
    datasets = {s: dataset_for(bundle, s, vocab.font) for s in ('train', 'valid')}
    indices = bundle['splits']['valid'].get('scene_indices', list(range(bundle['scene_count'])))
    rows, panels = [], []
    for pair_id in range(args.pairs):
        task = 'S3' if pair_id < 6 else 'S4' if pair_id < 9 else 'S5'
        dataset = datasets['valid']; recipe = dataset.recipes[indices[pair_id % len(indices)]]
        # 有全同类的原配方时，定点寻找包含两个类别的已有配方。
        for offset in range(len(indices)):
            recipe = dataset.recipes[indices[(pair_id+offset) % len(indices)]]
            if len({o['label'] for o in recipe['objects']}) > 1: break
        for variant in (0, 1):
            key = f'{task}_{pair_id}_{variant}'; path = out/'cases'/f'{key}.json'
            doc = paired_scene(task, recipe, dataset.drawings, vocab.font, pair_id=pair_id, variant=variant)
            cached=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
            if cached is not None and (task!='S4' or cached.get('candidate_kind')=='selection'):
                row = cached
            else:
                h, (mu, sigma, pi), target = features(generator, doc, 'selection' if task=='S4' else None)
                prefix = source_prefix(doc, task)
                if task == 'S4':
                    value = generator.generate(prefix, mode='greedy', max_new_strokes=16, max_new_pages=2)
                    with generator.autocast(): curves = generator.decode(mu).tolist()
                    mse = (mu.float()-target).square().mean(-1)
                    logp = pi.float().clamp_min(1e-30).log() - sigma.float().log().sum(-1) - ((target-mu.float())/sigma.float()).square().sum(-1)/2
                    responsibility = logp.softmax(0)
                    box=doc.metadata['selection_box'];center=np.asarray([(box[0]+box[2])/2,(box[1]+box[3])/2])
                    errors=[float(np.linalg.norm((np.min(c,axis=0)+np.max(c,axis=0))/2-center)) for c in curves]
                    row = dict(task=task, pair_id=pair_id, variant=variant, metadata=doc.metadata,
                        candidate_kind='selection',candidate_context='true_target_page_controls_only; no_target_ink',
                        result=value, candidates=[dict(component=i, pi=float(pi[i]), sigma=float(sigma[i].mean()),
                            latent_mse=float(mse[i]), center_error=errors[i],responsibility=float(responsibility[i]), curve=curves[i]) for i in range(len(pi))],
                        top_pi=int(pi.argmax()), nearest_latent=int(mse.argmin()),nearest_center=int(np.argmin(errors)),target_posterior=int(responsibility.argmax()))
                else:
                    value = generate_text_readout(generator, head, vocab, '', time.monotonic()+120,
                        SimpleNamespace(constraint='prefix', generation_strokes=96), layout, prefix=prefix)
                    actual = ' '.join(value['generated_page_texts'].values()).strip()
                    row = dict(task=task, pair_id=pair_id, variant=variant, metadata=doc.metadata,
                               generated=actual, correct=answer_match(actual, doc.metadata['answer'], task), result=value)
                write(path, row)
                print(json.dumps(dict(case=key, generated=row.get('generated'), top_pi=row.get('top_pi'))), flush=True)
            rows.append(row)
            curves = []
            if task == 'S4':
                curves = [(row['candidates'][row['top_pi']]['curve'], '#ef4444'),
                          (row['candidates'][row['nearest_latent']]['curve'], '#16a34a')]
            panels.append('<article><h3>'+html.escape(key)+'</h3>'+scene_svg(doc, curves)+
                '<p>'+html.escape(doc.metadata['question'])+'</p><p>正确：'+html.escape(str(doc.metadata['answer']))+
                '</p><p>输出：'+html.escape(row.get('generated', '红：最高π；绿：真实目标潜变量最近分量（诊断）'))+'</p></article>')
    records = []
    for split, dataset in datasets.items():
        data = bundle['splits'][split]
        cases = [('O2', int(i)) for i in data['balanced_indices'][:args.probe_objects]]
        scenes = data.get('scene_indices', list(range(bundle['scene_count'])))[:args.probe_scenes]
        cases += [(task, int(i)) for i in scenes for task in ('S3', 'S4')]
        for task, index in cases:
            name = f'{split}_{task}_{index}'
            path = out/'features'/f'{name}.npz'; meta_path = path.with_suffix('.json')
            if path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
            else:
                doc = dataset.get_for_task(task, index)
                if doc.stroke_count > 1024: continue
                h, _, _ = features(generator, doc)
                kind = doc.metadata.get('query', {}).get('kind', task)
                label = doc.metadata.get('answer', doc.metadata.get('label'))
                center = None
                if task == 'S4':
                    obj = next(o for o in doc.metadata['objects'] if o['instance_id'] == doc.metadata['target_instance_id'])
                    center = obj['center']
                meta = dict(split=split, task=task, index=index, kind=kind, label=label, center=center,
                            feature='last_condition_before_first_target', recipe_id=doc.metadata.get('recipe_id'))
                np.savez_compressed(path, h=h); write(meta_path, meta)
            records.append((meta, np.load(path)['h'].astype(np.float64)))
        print(json.dumps(dict(features_split=split, count=sum(m['split']==split for m, _ in records))), flush=True)
    readouts = []
    for kind in ('O2', 'count', 'existence', 'direction', 'S4'):
        groups = {s: [(m, h) for m, h in records if m['split']==s and m['kind']==kind] for s in ('train','valid')}
        if not all(groups.values()): continue
        x = {s: np.stack([h for _, h in g]) for s, g in groups.items()}
        if kind == 'S4':
            y = {s: np.array([m['center'] for m, _ in g]) for s, g in groups.items()}
            prediction = ridge(x['train'], y['train'], x['valid'])
            report = dict(mean_center_error=float(np.linalg.norm(prediction-y['valid'], axis=1).mean()),
                          train_mean_baseline=float(np.linalg.norm(y['train'].mean(0)-y['valid'], axis=1).mean()))
        else:
            labels = sorted({str(m['label']) for m, _ in records if m['kind']==kind})
            y = {s: np.array([labels.index(str(m['label'])) for m, _ in g]) for s, g in groups.items()}
            prediction = ridge(x['train'], np.eye(len(labels))[y['train']], x['valid']).argmax(-1)
            majority = Counter(y['train']).most_common(1)[0][0]
            report = dict(correct=int((prediction==y['valid']).sum()), accuracy=float((prediction==y['valid']).mean()),
                          majority_accuracy=float((y['valid']==majority).mean()), labels=labels)
        readouts.append(dict(kind=kind, train=len(groups['train']), valid=len(groups['valid']), **report))
    pairs = [dict(pair_id=i, task=next(r['task'] for r in rows if r['pair_id']==i),
                  both_correct=all(r['correct'] for r in rows if r['pair_id']==i) if next(r['task'] for r in rows if r['pair_id']==i)!='S4' else None,
                  generated=[r.get('generated') for r in rows if r['pair_id']==i]) for i in range(args.pairs)]
    unchanged = model_hash(generator.model) == source['r_model_hash']; assert unchanged
    report = dict(checkpoint=str(args.checkpoint.resolve()), source=source, optimizer_updates=0,
                  weights_unchanged=unchanged, readouts=readouts, pairs=pairs, cases=rows)
    write(out/'diagnosis.json', report)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>视觉条件诊断</title>'
        '<style>body{background:#f1f5f9;font-family:system-ui;margin:24px}main{display:grid;grid-template-columns:repeat(2,minmax(380px,1fr));gap:16px}article{background:white;padding:16px}pre{white-space:pre-wrap}</style>'
        '<h1>最终54825步：视觉条件诊断</h1><p>红色最高π；绿色真实目标最近候选，仅供诊断。</p><pre>'+
        html.escape(json.dumps(readouts, ensure_ascii=False, indent=2))+'</pre><main>'+''.join(panels)+'</main>', encoding='utf-8')
    print(json.dumps(dict(pairs=pairs, readouts=readouts, weights_unchanged=unchanged), ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
