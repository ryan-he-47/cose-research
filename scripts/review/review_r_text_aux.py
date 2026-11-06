"""2026-10-14：联合R收尾的密度分解、梯度观察和本机标准字体续写；零训练更新。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.demo.demo_text_readout import load_readout, load_joint_readout
from scripts.demo.demo_pretrain_generation import save_native_svg, text_prefix
from scripts.review.diagnose_formal_r_health import model_hash
from scripts.review.train_text_readout import extract
from src.page_document import build_document, layout_text_pages, control_stroke
from src.st.model import gmm_nll
from src.st.pipeline_pretrain import collate_documents, to_device
from src.st.text_readout import arc_points
from src.st.text_auxiliary import annotate_documents
from src.st.text_readout_generation import generate_text_readout
from src.st.text_shape_readout import shape_space_metrics
from src.text_augmentation import TextAugmentationConfig, augment_rendered_text
from functools import partial


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def audit(checkpoint, parent):
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    old = torch.load(parent, map_location='cpu', weights_only=False)
    frozen = [k for k in old['model'] if k.startswith(('encoder.', 'decoder.', 'dct_head.'))]
    groups = []
    for group in state['optimizer']['param_groups']:
        steps = [int(state['optimizer']['state'][p]['step']) for p in group['params']]
        groups.append(dict(role=group.get('role', 'R'), parameters=len(steps),
                           min_step=min(steps), max_step=max(steps), lr=group['lr']))
    result = dict(step=state['step'], phase_step=state['phase_step'], counters=state['counters'],
        successful_inputs=state['scheduler']['inputs'], recipe=state['train_args'],
        optimizer_groups=groups, scaler=state['scaler'], source_commit=state['source_commit'],
        cfg=state['cfg'], data=state['data'], ae_source=state['ae_source'],
        frozen_ae_equal=all(torch.equal(state['model'][k], old['model'][k]) for k in frozen),
        weights_finite=all(bool(torch.isfinite(v).all()) for v in state['model'].values()),
        auxiliary_manifest_equal=state['text_aux_manifest']==read(checkpoint.parent/'text_aux_manifest.json'))
    assert result['frozen_ae_equal'] and result['weights_finite'] and result['auxiliary_manifest_equal']
    assert state['scheduler']['inputs']==state['counters']['successful_inputs']>=state['train_args']['total_inputs']
    assert groups[0]['min_step']==groups[0]['max_step']==state['step']
    assert groups[1]['min_step']==groups[1]['max_step']==state['phase_step']
    assert state['step']==old['step']+state['phase_step']
    for key in ['sigma_eps_r', 'r_sigma_floor_mode', 'latent_units', 'max_context_strokes']:
        assert state['cfg'][key]==old['cfg'][key]
    return result


def density_parts(target, mu, sigma, pi):
    """NLL = 常数 + 后验平均logσ + 后验平均标准化残差/2 + KL(q||π)。"""
    actual = gmm_nll(target, mu, sigma, pi, reduction='none')
    # 诊断分解用double，避免q从接近-200的对数密度相减时的FP32归一误差。
    target, mu, sigma, pi = (v.double() for v in (target, mu, sigma, pi))
    # 与实际训练相同的pi+1e-12；logsumexp得到实际归一化的后验。
    residual = ((target[:, None]-mu)/sigma).square().sum(-1)*.5
    volume = sigma.log().sum(-1)
    logpi = (pi+1e-12).log()
    joint = logpi-volume-residual
    logq = joint-joint.logsumexp(-1, keepdim=True)
    q = logq.exp()
    constant = .5*target.shape[-1]*math.log(2*math.pi)
    fields = dict(volume=(q*volume).sum(-1), residual=(q*residual).sum(-1),
                  mixture_kl=(q*(logq-logpi)).sum(-1))
    rebuilt = fields['volume']+fields['residual']+fields['mixture_kl']+constant
    assert torch.allclose(actual.double(), rebuilt, atol=.001, rtol=1e-5)
    fields['fp32_reconstruction_error'] = actual.double()-rebuilt
    row = torch.arange(len(target), device=target.device)
    posterior, prior = q.argmax(-1), pi.argmax(-1)
    fields.update(nll=actual, posterior_sigma=sigma[row, posterior].mean(-1),
        posterior_mse=(mu[row, posterior]-target).square().mean(-1),
        top_pi_mse=(mu[row, prior]-target).square().mean(-1),
        pi_matches=(prior==posterior).float(), pi_entropy=-(pi*logpi).sum(-1),
        posterior_mean_sigma=q.mul(sigma.mean(-1)).sum(-1))
    return fields, constant


@torch.no_grad()
def observe_features(generator, head, cache, catalog, out, tag):
    arrays, shape_out, space_out = {}, [], []
    for start in range(0, len(cache['labels']), 512):
        h = cache['features'][start:start+512].to(generator.device)
        target = cache['targets'][start:start+512].to(generator.device)
        with generator.autocast():
            output = head(h)
            mu, sigma, pi = generator.model.prediction_head(h)
        fields, constant = density_parts(target, mu, sigma, pi)
        for key, value in fields.items():
            arrays.setdefault(key, []).append(value.cpu().numpy())
        shape_out.append(output['shape'].float().cpu()); space_out.append(output['space'].float().cpu())
    arrays = {k: np.concatenate(v) for k, v in arrays.items()}
    output = dict(shape=torch.cat(shape_out), space=torch.cat(space_out))
    classification = shape_space_metrics(output, cache['labels'], catalog)
    shape, space, first = catalog.targets(cache['labels'])
    body = np.asarray(catalog.body)[cache['labels'].numpy()]
    masks = dict(all=np.ones(len(body), bool), text=body, first=first.numpy(),
                 continuation=body & ~first.numpy(), control=~body)
    decomposition = {}
    for name, mask in masks.items():
        decomposition[name] = dict(count=int(mask.sum()), gaussian_constant=constant,
            **{k: float(v[mask].mean()) for k, v in arrays.items()},
            top_pi_mse_median=float(np.median(arrays['top_pi_mse'][mask])),
            top_pi_mse_p90=float(np.quantile(arrays['top_pi_mse'][mask], .9)))
    arrays['shape_correct'] = (output['shape'].argmax(-1)==shape).numpy()
    arrays['space_correct'] = ((output['space']>0)==space.bool()).numpy()
    np.savez_compressed(out/f'density_{tag}.npz', **arrays)
    write(out/f'classification_{tag}.json', classification)
    write(out/f'density_{tag}.json', decomposition)
    return arrays, decomposition


def paired_observation(out, labels, catalog):
    before = np.load(out/'density_before.npz')
    after = np.load(out/'density_after.npz')
    labels = torch.as_tensor(labels).cpu()
    _, _, first = catalog.targets(labels)
    body = np.asarray(catalog.body)[labels.numpy()]
    delta = after['nll']-before['nll']
    groups = {}
    for name, mask in [('body', body), ('first', first.numpy()), ('continuation', body & ~first.numpy())]:
        groups[name] = {}
        for key, condition in [('both_correct', before['shape_correct'] & after['shape_correct']),
                               ('corrected', ~before['shape_correct'] & after['shape_correct']),
                               ('regressed', before['shape_correct'] & ~after['shape_correct']),
                               ('both_wrong', ~before['shape_correct'] & ~after['shape_correct'])]:
            selected = mask & condition
            groups[name][key] = dict(count=int(selected.sum()), nll_delta=float(delta[selected].mean()) if selected.any() else None,
                contribution_to_group_delta=float(delta[selected].sum()/mask.sum()))
    ids = np.unique(np.linspace(0, len(labels)-1, 4096, dtype=int))
    groups['same_4096_classification'] = {}
    for name, mask in [('all', np.ones(len(labels), bool)), ('text', body), ('first', first.numpy()),
                       ('continuation', body & ~first.numpy()), ('control', ~body)]:
        selected = ids[mask[ids]]
        groups['same_4096_classification'][name] = dict(count=len(selected),
            before_error=float((~before['shape_correct'][selected]).mean()),
            after_error=float((~after['shape_correct'][selected]).mean()))
    write(out/'paired_density_shapes.json', groups)


def gradient_observation(generator, head, vocab, layout, rows, alpha, out, tag, loss_scale=64.):
    """同四个增强文档，两损失分别求导；只读真实R参数，含AMP反传量化。"""
    model = generator.model
    parameters = [(name, p) for name, p in model.r_pred.named_parameters()]
    model.r_pred.requires_grad_(True); head.requires_grad_(True)
    reports = []
    try:
        for item in rows[:4]:
            rng = np.random.default_rng(np.random.SeedSequence([42, 0, item['logical_index']]))
            augment = partial(augment_rendered_text, rng=rng, config=TextAugmentationConfig())
            document = build_document(layout_text_pages(item['text'], vocab.font, config=layout,
                augment=augment), complete=True, metadata={'task': 'T1'}).window(0, model.cfg.max_context_strokes)
            annotate_documents([document], head.catalog)
            batch = to_device(collate_documents([document]), generator.device)
            model.zero_grad(set_to_none=True)
            with generator.autocast():
                embeddings = model.encode_valid(batch)
                hidden = model.hidden_states(embeddings[:, :-1], batch['slice_id'][:, :-1],
                    batch['stroke_pad'][:, :-1], right_padded=True)
                selected = model.target_mask(batch)
                h = hidden[selected]
                mu, sigma, pi = model.prediction_head(h)
                output = head(h)
            target = embeddings[:, 1:][selected].float()
            kinds = batch['kind_id'][:, 1:][selected]
            nll = gmm_nll(target, mu, sigma, pi, reduction='none')
            body, control = kinds==0, kinds>=3
            geo = .35*nll[body].mean()+.05*nll[control].mean()
            ce = torch.nn.functional.cross_entropy(output['shape'].float(), batch['text_aux_shape'][:, 1:][selected])
            bce = torch.nn.functional.binary_cross_entropy_with_logits(output['space'].float(), batch['text_aux_space'][:, 1:][selected])
            aux = alpha*(ce+bce)
            # 固定较低scale，只防FP16反传下溢；没有scaler.step或optimizer.step。
            scale = float(loss_scale)
            dg = torch.autograd.grad(geo*scale, [h, *[p for _, p in parameters]], retain_graph=True, allow_unused=True)
            da = torch.autograd.grad(aux*scale, [h, *[p for _, p in parameters]], allow_unused=True)
            groups = {}
            for name, indices in [('hidden', [0]), ('R_backbone', [i+1 for i, (n, _) in enumerate(parameters) if not n.startswith(('head.', 'head_buf.'))]),
                                  ('GMM_head', [i+1 for i, (n, _) in enumerate(parameters) if n.startswith(('head.', 'head_buf.'))]),
                                  ('R_all', list(range(1, len(parameters)+1)))]:
                gg = aa = dot = 0.
                for i in indices:
                    g = None if dg[i] is None else dg[i].detach().float()/scale
                    a = None if da[i] is None else da[i].detach().float()/scale
                    if g is not None:
                        assert bool(torch.isfinite(g).all()); gg += float(g.double().square().sum())
                    if a is not None:
                        assert bool(torch.isfinite(a).all()); aa += float(a.double().square().sum())
                    if g is not None and a is not None: dot += float((g.double()*a.double()).sum())
                groups[name] = dict(geometry_norm=math.sqrt(gg), weighted_aux_norm=math.sqrt(aa),
                    weighted_aux_to_geometry=math.sqrt(aa/max(gg, 1e-30)),
                    cosine=dot/math.sqrt(max(gg*aa, 1e-60)), dot=dot,
                    joint_norm=math.sqrt(max(0, gg+aa+2*dot)))
            reports.append(dict(source_id=item['source_id'], targets=int(selected.sum()),
                geometry_loss=float(geo.detach()), shape_ce=float(ce.detach()), space_bce=float(bce.detach()), groups=groups))
            del dg, da, output, mu, sigma, pi, h, hidden, embeddings, nll, geo, aux, ce, bce
        write(out/f'gradients_{tag}.json', dict(augmentation=True, alpha=alpha, loss_scale=loss_scale,
            optimizer_updates=0, precision='FP16 backbone/head; FP32 NLL/CE/BCE', rows=reports))
    finally:
        model.zero_grad(set_to_none=True); model.r_pred.requires_grad_(False); head.requires_grad_(False)


@torch.no_grad()
def decoded_shape_errors(generator, cache, catalog, out, tag, count=4096):
    # 沿用10-14已有同4096位置、四次采样的量化，参考排版只进入外部匹配器。
    ids = np.unique(np.linspace(0, len(cache['labels'])-1, count, dtype=int))
    truth, _, first = catalog.targets(cache['labels'])
    controls = np.stack([arc_points(control_stroke(k), catalog.count) for k in catalog.controls])
    controls -= controls.mean(1, keepdims=True)
    random = torch.Generator(device=generator.device).manual_seed(42)
    rows = []
    for start in range(0, len(ids), 32):
        selected = ids[start:start+32]
        mu, sigma, pi = generator.model.prediction_head(cache['features'][selected].to(generator.device))
        batch = torch.arange(len(selected), device=generator.device)
        latent = [mu[batch, pi.argmax(-1)]]
        for _ in range(4):
            component = torch.multinomial(pi, 1, generator=random).squeeze(1)
            noise = torch.randn(mu[batch, component].shape, device=generator.device, generator=random)
            latent.append(mu[batch, component]+sigma[batch, component]*noise*.5)
        with generator.autocast(): decoded = [generator.decode(value) for value in latent]
        for j, index in enumerate(selected):
            size = cache['frames'][index].size if cache['frames'][index] else .085
            templates = np.concatenate([np.stack(catalog.templates)*size, controls])
            predictions = []
            for value in decoded:
                curve = arc_points(value[j], catalog.count).astype(np.float64); curve -= curve.mean(0)
                predictions.append(int(np.mean((templates-curve)**2, axis=(1, 2)).argmin()))
            rows.append(dict(index=int(index), truth=int(truth[index]), first=bool(first[index]),
                body=bool(catalog.body[int(cache['labels'][index])]), greedy=predictions[0], samples=predictions[1:]))
    groups = {}
    for name, mask in [('all', lambda r: True), ('text', lambda r: r['body']),
                       ('first', lambda r: r['first']), ('continuation', lambda r: r['body'] and not r['first']),
                       ('control', lambda r: not r['body'])]:
        selected = [r for r in rows if mask(r)]
        groups[name] = dict(count=len(selected), greedy=float(np.mean([r['greedy']!=r['truth'] for r in selected])),
            sample=float(np.mean([v!=r['truth'] for r in selected for v in r['samples']])))
    write(out/f'decoded_shape_{tag}.json', dict(groups=groups, metric='directed_arc64_centered_canvas_MSE_at_reference_font_size',
        reference_layout_only_for_external_matching=True, sample_draws=4, temperature=.5, AE_points_unchanged=True))


@torch.no_grad()
def language_pairs(generator, head, catalog, font):
    pairs = [('voice', ['A little cat was sitting here. It said', 'A little dog was sitting here. It said'], ['meow.', 'woof.']),
             ('need', ['Lily was very hungry.', 'Lily was very tired.'], ['She ate some food.', 'She went to sleep.']),
             ('weather', ['It was raining outside.', 'It was sunny outside.'], ['She took her umbrella.', 'She put on her sunglasses.'])]
    result = []
    for name, contexts, candidates in pairs:
        scores = []
        for context in contexts:
            condition = layout_text_pages(context, font, targetable=False)
            row = []
            for continuation in candidates:
                doc = build_document((*condition, *layout_text_pages(continuation, font)), complete=True)
                annotate_documents([doc], catalog)
                batch = to_device(collate_documents([doc]), generator.device)
                with generator.autocast():
                    emb = generator.model.encode_valid(batch)
                    hidden = generator.model.hidden_states(emb[:, :-1], batch['slice_id'][:, :-1], batch['stroke_pad'][:, :-1], right_padded=True)
                    selected = generator.model.target_mask(batch)
                    h = hidden[selected]; mu, sigma, pi = generator.model.prediction_head(h); output = head(h)
                body = batch['kind_id'][:, 1:][selected]==0
                shape = batch['text_aux_shape'][:, 1:][selected]
                space = batch['text_aux_space'][:, 1:][selected]
                values = dict(geometry_nll=gmm_nll(emb[:, 1:][selected], mu, sigma, pi, reduction='none'),
                    shape_ce=torch.nn.functional.cross_entropy(output['shape'].float(), shape, reduction='none'),
                    space_bce=torch.nn.functional.binary_cross_entropy_with_logits(output['space'].float(), space, reduction='none'))
                row.append({k: float(v[body].mean()) for k, v in values.items()})
            scores.append(row)
        shift = {key: (scores[1][0][key]-scores[0][0][key])-(scores[1][1][key]-scores[0][1][key]) for key in scores[0][0]}
        result.append(dict(name=name, contexts=contexts, candidates=candidates, scores=scores, preference_shift=shift))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='已下载final.pt与日志的独立目录')
    parser.add_argument('--parent', type=Path, default=ROOT/'tmp_smoke/r_t1_final_review_20261013/final.pt')
    parser.add_argument('--old_head', type=Path, default=ROOT/'tmp_smoke/r_shape_space_readout_20261014/head_final.pt')
    parser.add_argument('--old_cache', type=Path, default=ROOT/'tmp_smoke/r_text_readout_20261013/features_valid.pt')
    parser.add_argument('--valid', type=Path, default=ROOT/'tmp_smoke/text_readout_assets_20261013/valid.jsonl')
    parser.add_argument('--max_seconds', type=float, default=2400, help='本机只读验收工程上限，不是训练预算')
    args = parser.parse_args(); out = args.out
    if (out/'completion.json').exists(): raise FileExistsError('已有完整验收，不覆盖')
    torch.set_num_threads(4)
    started = time.monotonic(); deadline = started+args.max_seconds
    checks = audit(out/'final.pt', args.parent); write(out/'review_checks.json', checks)
    print(json.dumps(dict(checkpoint_audit=True, step=checks['step'], inputs=checks['successful_inputs'])), flush=True)
    rows = [json.loads(line) for line in args.valid.read_text(encoding='utf-8').splitlines()]
    alpha = checks['cfg']['r_text_aux_weight']
    summaries = {}
    for tag in ['before', 'after']:
        if tag=='before':
            generator, head, vocab, layout, _ = load_readout(args.old_head, args.parent)
            cache = torch.load(args.old_cache, map_location='cpu', weights_only=False)
        else:
            generator, head, vocab, layout, _ = load_joint_readout(out/'final.pt')
            generator.model.encode_chunk_size = 128
            cache = extract(generator, vocab, rows, 'valid', deadline, layout)
        generator.model.encode_chunk_size = 128
        before = model_hash(generator.model)
        arrays, density = observe_features(generator, head, cache, head.catalog, out, tag)
        summaries[tag] = density
        write(out/'density_comparison.json', summaries)
        print(json.dumps(dict(density_complete=tag, text=density['text'])), flush=True)
        gradient_observation(generator, head, vocab, layout, rows, alpha, out, tag)
        write(out/f'language_pairs_{tag}.json', language_pairs(generator, head, head.catalog, vocab.font))
        if tag=='after':
            paired_observation(out, cache['labels'], head.catalog)
            decoded_shape_errors(generator, cache, head.catalog, out, tag)
            prompts = [read(ROOT/f'tmp_smoke/r_shape_space_readout_20261014/prompt{i}.json')['prompt'] for i in range(4)]
            gen_args = SimpleNamespace(constraint='prefix', generation_strokes=480)
            for index, prompt in enumerate(prompts):
                value = generate_text_readout(generator, head, vocab, prompt, deadline, gen_args, layout)
                write(out/f'prompt{index}.json', value); save_native_svg(value, out/f'prompt{index}.svg', f'Joint R / prompt{index}')
                print(json.dumps(dict(readout_prompt=index, stop=value['stop_reason'], text=value['generated_page_texts'])), flush=True)
            for index in range(2):
                case = out/f'gmm_prompt{index}'; case.mkdir(exist_ok=True)
                prefix = text_prefix('T1', prompts[index], vocab.font)
                for mode in ['greedy', 'sample']:
                    value = generator.generate(prefix, mode=mode, temperature=.5, seed=42, max_new_strokes=480, max_new_pages=8)
                    write(case/f'{mode}.json', value); save_native_svg(value, case/f'{mode}.svg', f'Joint GMM / prompt{index} / {mode}')
                    print(json.dumps(dict(gmm_prompt=index, mode=mode, stop=value['stop_reason'])), flush=True)
        after = model_hash(generator.model)
        assert before==after
        write(out/f'inference_{tag}.json', dict(weights_unchanged=True, weight_hash_before=before,
            weight_hash_after=after, optimizer_updates=0, cache='same_R_frozen_cache' if tag=='before' else 'online_current_R',
            targets=len(cache['labels']), layout=asdict(layout)))
        del generator, head, cache, arrays; gc.collect(); torch.cuda.empty_cache()
    write(out/'completion.json', dict(complete=True, seconds=time.monotonic()-started, training=False,
        optimizer_updates=0, device='local RTX4060', precision='FP16 backbone; FP32 density/loss',
        validation_stories=len(rows), readout_prompts=4, GMM_prompts=2, cuda_peak_gib=torch.cuda.max_memory_allocated()/2**30))
    print(json.dumps(dict(complete=True, seconds=time.monotonic()-started)), flush=True)


if __name__=='__main__': main()
