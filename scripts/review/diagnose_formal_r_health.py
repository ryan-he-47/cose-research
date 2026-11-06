"""2026-10-09：复用本机正式末R/验收原笔，检查GMM、回馈、条件及梯度；不更新权重。"""
from __future__ import annotations

# 2026-10-12：分类目录中的脚本也可从仓库根直接运行。
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))


import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.demo.demo_pretrain_generation import load_generator, save_native_svg, text_prefix
from scripts.review.review_formal_r import write_json
from src.page_document import StrokeElement
from src.st.ae_geometry import arc_resample, chamfer
from src.st.formal_recipe import LOSS_WEIGHTS, TASKS
from src.st.model import gmm_nll
from src.st.pipeline_pretrain import collate_documents, to_device


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def document(records, task):
    elements = tuple(StrokeElement(np.asarray(r["points"], np.float32), r["page_id"],
                                  r["targetable"], r["kind"]) for r in records)
    return SimpleNamespace(to_sequence=lambda: elements, metadata={"task": task})


def stats(values):
    a = np.asarray(values, np.float64)
    if not len(a):
        return {"count": 0}
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)),
                p99=float(np.quantile(a, .99)), min=float(a.min()), max=float(a.max()))


def model_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def decode_chunks(generator, embeddings):
    return np.concatenate([generator.decode(embeddings[i:i+64])
                           for i in range(0, len(embeddings), 64)])


@torch.no_grad()
def target_predictions(generator, records, task, *, context=None):
    model = generator.model
    batch = to_device(collate_documents([document(records, task)]), generator.device)
    with generator.autocast():
        emb = model.encode_valid(batch)
        supplied = emb if context is None else context.unsqueeze(0)
        hidden = model.hidden_states(supplied[:, :-1], batch["slice_id"][:, :-1],
                                     batch["stroke_pad"][:, :-1], right_padded=True)
        selected = model.target_mask(batch)
        mu, sigma, pi = model.prediction_head(hidden[selected])
    target = emb[:, 1:][selected].float()
    kinds = batch["kind_id"][:, 1:][selected]
    return mu, sigma, pi, target, kinds


def mixture_rows(mu, sigma, pi, target, kinds, task, *, sigma_eps, sigma_clamp_lo):
    residual = (target[:, None] - mu) / sigma
    joint = (pi + 1e-12).log() - sigma.log().sum(-1) - .5*residual.square().sum(-1)
    gamma = joint.softmax(-1)
    posterior, likely = gamma.argmax(-1), pi.argmax(-1)
    row = torch.arange(len(pi), device=pi.device)
    nll = gmm_nll(target, mu, sigma, pi, reduction="none")
    # 输出坐标上的NLL导数；这里只作只读尺度观测，不回传E。
    d_mu = gamma[..., None] * (mu-target[:, None]) / sigma.square()
    d_log_sigma = gamma[..., None] * (1-residual.square())
    names = {0:"text", 1:"sketch", 2:"selection", 3:"bos", 4:"eop", 5:"eod"}
    arrays = dict(nll=nll, prior_component=likely, posterior_component=posterior,
        posterior_peak=gamma.max(-1).values,
        prior_entropy=-(pi*(pi+1e-12).log()).sum(-1),
        posterior_entropy=-(gamma*(gamma+1e-12).log()).sum(-1),
        top_pi_mse=(mu[row,likely]-target).square().mean(-1),
        posterior_mse=(mu[row,posterior]-target).square().mean(-1),
        top_pi_sigma=sigma[row,likely].mean(-1),
        posterior_sigma=sigma[row,posterior].mean(-1),
        posterior_sigma_min=sigma[row,posterior].min(-1).values,
        all_sigma_min=sigma.flatten(1).min(-1).values,
        floor_hit_fraction=(sigma <= sigma_eps+math.exp(sigma_clamp_lo)*1.01).float().mean((1,2)),
        posterior_standardized_rms=residual[row,posterior].square().mean(-1).sqrt(),
        mu_derivative_norm=d_mu.flatten(1).norm(dim=1),
        log_sigma_derivative_norm=d_log_sigma.flatten(1).norm(dim=1),
        pi_logit_derivative_norm=(pi-gamma).norm(dim=1))
    for new_eps in (.01,.02):
        # 只改σ下限的数值敏感性：固定μ/π，不重新训练，不把本项当作新模型效果。
        wider = sigma-sigma_eps+new_eps
        standardized = (target[:,None]-mu)/wider
        post = ((pi+1e-12).log()-wider.log().sum(-1)-.5*standardized.square().sum(-1)).softmax(-1)
        derivative = post[...,None]*(mu-target[:,None])/wider.square()
        tag = f"eps{new_eps:g}"
        arrays[tag+"_mu_derivative_norm"] = derivative.flatten(1).norm(dim=1)
        arrays[tag+"_nll"] = gmm_nll(target,mu,wider,pi,reduction="none")
    host = {k:v.cpu().numpy() for k,v in arrays.items()}
    return [dict(task=task, kind=names[int(k)], **{a:float(v[i]) for a,v in host.items()})
            for i,k in enumerate(kinds.cpu().tolist())]


def summarize_mixture(rows):
    result = {}
    for kind in sorted(set(r["kind"] for r in rows)):
        selected = [r for r in rows if r["kind"] == kind]
        counts = Counter(int(r["posterior_component"]) for r in selected)
        probability = np.array(list(counts.values()))/len(selected)
        result[kind] = dict(count=len(selected), posterior_component_counts=dict(counts),
            posterior_effective_components=float(np.exp(-(probability*np.log(probability)).sum())),
            top_pi_matches_posterior=float(np.mean([r["prior_component"]==r["posterior_component"] for r in selected])),
            **{key:stats([r[key] for r in selected]) for key in selected[0]
               if key not in ("task", "kind", "prior_component", "posterior_component")})
    return result


@torch.no_grad()
def cycles(generator, records, task):
    raw = [np.asarray(r["points"], np.float32) for r in records]
    with generator.autocast():
        encoded = generator.encode(raw)
        dense = generator.encode([arc_resample(p,76) for p in raw])
        decoded = decode_chunks(generator, encoded)
        dense_decoded = decode_chunks(generator, dense)
        cycle = generator.encode(decoded)
        oracle = generator.encode([arc_resample(p,len(q)) for p,q in zip(decoded,raw)])
    rows = []
    for i,record in enumerate(records):
        rows.append(dict(task=task, kind=record["kind"], original_points=len(raw[i]),
            dense_rms=float((dense[i].float()-encoded[i].float()).square().mean().sqrt()),
            cycle_rms=float((cycle[i].float()-encoded[i].float()).square().mean().sqrt()),
            oracle_length_cycle_rms=float((oracle[i].float()-encoded[i].float()).square().mean().sqrt()),
            dense_decoded_cd=chamfer(arc_resample(raw[i],64),arc_resample(dense_decoded[i],64)),
            decoded_cd=chamfer(arc_resample(raw[i],64),arc_resample(decoded[i],64))))
    baseline = target_predictions(generator,records,task)
    kinds = baseline[-1]
    mu,sigma,pi,target,_ = baseline
    nlls = {"original":gmm_nll(target,mu,sigma,pi,reduction="none")}
    # 目标始终是原笔λ，仅更换R可见前文；这是训练/推理接口敏感性检查。
    for name, supplied in (("dense76",dense), ("decoded_cycle",cycle), ("oracle_length_cycle",oracle)):
        mu,sigma,pi,target,_ = target_predictions(generator,records,task,context=supplied)
        nlls[name] = gmm_nll(target,mu,sigma,pi,reduction="none")
    sensitivity = {}
    for kind,mask in (("text",kinds==0),("sketch",kinds==1),("selection",kinds==2),("control",kinds>=3)):
        if bool(mask.any()):
            sensitivity[kind] = {name:stats(values[mask].cpu().tolist()) for name,values in nlls.items()}
    return rows, sensitivity


@torch.no_grad()
def condition_matrix(generator, sources, task):
    names = [name for name in sources if name.startswith(task+"_")]
    results = {}
    for target_name in names:
        original = sources[target_name]
        boundary = next(i for i,r in enumerate(original) if i and r["targetable"])
        answer = original[boundary:]
        scores = {}
        for condition_name in [*names, "last_eop_only"]:
            if condition_name == "last_eop_only":
                prefix = [dict(original[boundary-1],targetable=False)]
            else:
                alternate = sources[condition_name]
                n = next(i for i,r in enumerate(alternate) if i and r["targetable"])
                prefix = alternate[:n]
            mu,sigma,pi,target,kinds = target_predictions(generator,[*prefix,*answer],task)
            mask = kinds < 3
            scores[condition_name] = dict(body_nll=float(gmm_nll(target[mask],mu[mask],sigma[mask],pi[mask])),
                first_nll=float(gmm_nll(target[:1],mu[:1],sigma[:1],pi[:1])),
                first_top_pi_mse=float((mu[0,pi[0].argmax()]-target[0]).square().mean()),
                first_top_pi=int(pi[0].argmax()), first_pi=float(pi[0].max()),
                first_top_pi_mean=mu[0,pi[0].argmax()].cpu().tolist())
        results[target_name] = scores
    return results


def matched_conditions(sources, font, task):
    """O1固定Draw模板；O2保持本例指令，仅交换草图，排除指令风格干扰。"""
    result = {}
    names = [name for name in sources if name.startswith(task+"_")]
    for name in names:
        records = sources[name]
        boundary = next(i for i,r in enumerate(records) if i and r["targetable"])
        if task == "O1":
            prefix = [dict(points=e.rawxy.tolist(),page_id=e.page_id,kind=e.kind,targetable=False)
                      for e in text_prefix("O1",name.split("_",1)[1],font)]
            result[name] = [*prefix,*records[boundary:]]
        else:
            result[name] = records
    return result


@torch.no_grad()
def sketch_only_matrix(generator, sources):
    names = [name for name in sources if name.startswith("O2_")]
    output = {}
    for name in names:
        records = sources[name]
        sketch_page = next(r["page_id"] for r in records if r["kind"]=="sketch")
        output[name] = {}
        for other in names:
            sketch = [dict(r,page_id=sketch_page,targetable=False) for r in sources[other] if r["kind"]=="sketch"]
            start = next(i for i,r in enumerate(records) if r["kind"]=="sketch")
            end = max(i for i,r in enumerate(records) if r["kind"]=="sketch")+1
            mu,sigma,pi,target,kinds = target_predictions(generator,[*records[:start],*sketch,*records[end:]],"O2")
            body = kinds < 3
            output[name][other] = dict(body_nll=float(gmm_nll(target[body],mu[body],sigma[body],pi[body])),
                first_nll=float(gmm_nll(target[:1],mu[:1],sigma[:1],pi[:1])),
                first_top_pi_mse=float((mu[0,pi[0].argmax()]-target[0]).square().mean()),
                first_top_pi=int(pi[0].argmax()), first_top_pi_mean=mu[0,pi[0].argmax()].cpu().tolist())
    return output


@torch.no_grad()
def layer_sensitivity(generator, sources):
    from src.st.model import _page_pe
    states = []
    for name in ("O1_cat","O1_fish","O1_airplane","O1_bicycle"):
        source = sources[name]
        boundary = next(i for i,r in enumerate(source) if r["targetable"])
        prefix = source[:boundary]
        with generator.autocast():
            emb = generator.encode([np.asarray(r["points"],np.float32) for r in prefix]).unsqueeze(0)
            ids = torch.tensor([[r["page_id"] for r in prefix]],device=generator.device)
            r = generator.model.r_pred
            x = r.tok_proj(torch.cat((emb,_page_pe(ids,r.pe_dim).to(emb.dtype)),-1))
            values = [x[0,-1].float().cpu()]
            for layer in r.layers:
                x = layer(x,causal=True,key_padding_mask=None,rope_pos=None)
                values.append(x[0,-1].float().cpu())
            states.append(values)
    return [dict(layer=i-1, relative_rms_to_cat=[float((s[i]-states[0][i]).norm()/states[0][i].norm()) for s in states],
        cosine_to_cat=[float(torch.nn.functional.cosine_similarity(s[i],states[0][i],dim=0)) for s in states])
        for i in range(len(states[0]))]


def group_norms(named, gradients, *, latent_units=48, components=32):
    squared = defaultdict(float)
    for (name,_), value in zip(named,gradients):
        if value is None:
            continue
        if name.startswith("layers."):
            group = ".".join(name.split(".")[:2])
        else:
            group = name.split(".")[0]
        squared[group] += float(value.float().square().sum())
        if name.startswith("head.head."):
            # GMMDense最后线性层的行布局：K×F均值、K×F尺度、K混合logit。
            width = latent_units*components
            for label, part in zip(("mu","raw_sigma","pi_logits"),value.split([width,width,components],0)):
                squared["gmm_"+label] += float(part.float().square().sum())
    return {key:math.sqrt(value) for key,value in squared.items()}


def gradient_probe(generator, sources):
    # 每任务取首个回答页；明确是诊断批，不伪装成正式大批的历史梯度。
    chosen = ("T1_story0","O1_cat","O2_cat","S1_scene0","S2_scene0","S3_scene0","S4_scene0","S5_scene0")
    docs = []
    total_controls = 0
    for name in chosen:
        original = sources[name]
        first = next(r["page_id"] for i,r in enumerate(original) if i and r["targetable"])
        records = [r for r in original if r["page_id"] <= first]
        task = name.split("_")[0]
        total_controls += sum(r["targetable"] and r["kind"] in ("eop","eod") for r in records)
        docs.append((name,task,records))
    named = list(generator.model.r_pred.named_parameters())
    head_shape = dict(latent_units=generator.model.cfg.latent_units, components=generator.model.cfg.gmm_r)
    parameters = [p for _,p in named]
    accumulated = [[torch.zeros_like(p,device="cpu") for p in parameters] for _ in range(2)]
    details = {}
    for name,task,records in docs:
        batch = to_device(collate_documents([document(records,task)]),generator.device)
        batch["task_id"] = torch.tensor([TASKS.index(task)],device=generator.device)
        with generator.autocast():
            output = generator.model(batch)
        body = LOSS_WEIGHTS[task]*output["content_sum"]/output["content_count"]
        control = .05*output["control_sum"]/total_controls
        entry = dict(strokes=len(records), body_count=int(output["content_count"]),
                     control_count=int(output["control_count"]), weight=LOSS_WEIGHTS[task])
        for label,loss,acc in zip(("body","control"),(body,control),accumulated):
            scaled = torch.autograd.grad(loss*512,parameters,retain_graph=label=="body",allow_unused=True)
            gradients = [g/512 if g is not None else None for g in scaled]
            if any(not bool(torch.isfinite(g).all()) for g in gradients if g is not None):
                raise FloatingPointError(f"{name}/{label} 的诊断梯度非有限")
            entry[label+"_norms"] = group_norms(named,gradients,**head_shape)
            for buffer,gradient in zip(acc,gradients):
                if gradient is not None:
                    buffer.add_(gradient.detach().cpu())
            del scaled, gradients
        details[name] = entry
        del output, body, control, batch
        print(json.dumps(dict(gradient_case=name,**entry),ensure_ascii=False),flush=True)
    body,control = accumulated
    b2 = sum(float(g.square().sum()) for g in body)
    c2 = sum(float(g.square().sum()) for g in control)
    dot = sum(float(x.flatten().dot(y.flatten())) for x,y in zip(body,control))
    combined = [b+c for b,c in zip(body,control)]
    return dict(cases=details, body_norm=math.sqrt(b2), control_norm=math.sqrt(c2),
                gradient_cosine=dot/math.sqrt(b2*c2), combined_norm=math.sqrt(b2+c2+2*dot),
                body_groups=group_norms(named,body,**head_shape), control_groups=group_norms(named,control,**head_shape),
                combined_groups=group_norms(named,combined,**head_shape), control_denominator=total_controls,
                normalization="每任务正文均值×原任务权重，控制符sum×0.05/全诊断批控制数；不执行optimizer.step")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--out_dir",type=Path,required=True)
    parser.add_argument("--generation_only",action="store_true",help="复用已保存诊断，只补生成；核对同权重")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if any(args.out_dir.iterdir()) and not args.generation_only:
        raise FileExistsError("需要独立的空输出目录")
    torch.set_num_threads(4)
    generator,font,metadata = load_generator(args.checkpoint,"cuda")
    generator.model.encode_chunk_size = 128
    before = model_hash(generator.model)
    if args.generation_only and read(args.out_dir/"metadata.json")["initial_weight_hash"] != before:
        raise ValueError("补生成必须复用同权重")
    sources = {f"{item['task']}_{item['label']}":read(args.baseline/f"{item['task']}_{item['label']}"/"reference.json")["strokes"]
               for item in read(args.baseline/"summary.json")}
    write_json(args.out_dir/"metadata.json",dict(**metadata,cfg=asdict(generator.model.cfg),training=False,
        device=torch.cuda.get_device_name(), precision="FP16 trunk / FP32 GMM", initial_weight_hash=before))
    if args.generation_only:
        all_rows = read(args.out_dir/"mixture_targets.json")
    else:
        all_rows,cycle_rows,sensitivity = [],[],{}
        for name,records in sources.items():
            task = name.split("_")[0]
            with torch.no_grad():
                mu,sigma,pi,target,kinds = target_predictions(generator,records,task)
                all_rows.extend(mixture_rows(mu,sigma,pi,target,kinds,task,
                    sigma_eps=generator.model.cfg.sigma_eps_r,
                    sigma_clamp_lo=generator.model.cfg.sigma_clamp_lo))
                current,context = cycles(generator,records,task)
            cycle_rows.extend(current)
            sensitivity[name] = context
            print(json.dumps(dict(case=name,targets=len(target),context=context),ensure_ascii=False),flush=True)
        write_json(args.out_dir/"mixture_targets.json",all_rows)
        write_json(args.out_dir/"mixture_summary.json",summarize_mixture(all_rows))
        write_json(args.out_dir/"cycle_strokes.json",cycle_rows)
        grouped = {}
        for kind in sorted(set(r["kind"] for r in cycle_rows)):
            for bucket in ("all","below76","at76"):
                subset = [r for r in cycle_rows if r["kind"]==kind and
                          (bucket=="all" or (r["original_points"]<76 if bucket=="below76" else r["original_points"]==76))]
                grouped[kind+"/"+bucket] = {key:stats([r[key] for r in subset]) for key in
                    ("original_points","dense_rms","cycle_rms","oracle_length_cycle_rms","decoded_cd","dense_decoded_cd")}
        write_json(args.out_dir/"cycle_summary.json",grouped)
        write_json(args.out_dir/"context_sensitivity.json",sensitivity)
        for task in ("O1","O2"):
            write_json(args.out_dir/f"condition_{task}.json",condition_matrix(generator,sources,task))
        write_json(args.out_dir/"layer_condition_sensitivity.json",layer_sensitivity(generator,sources))
        matched = matched_conditions(sources,font,"O1")
        write_json(args.out_dir/"condition_O1_fixed_prompt.json",condition_matrix(generator,matched,"O1"))
        write_json(args.out_dir/"condition_O2_sketch_only.json",sketch_only_matrix(generator,sources))
        write_json(args.out_dir/"layer_fixed_prompt_sensitivity.json",layer_sensitivity(generator,matched))
        write_json(args.out_dir/"gradient_probe.json",gradient_probe(generator,sources))
    generation = {}
    for name in ("O1_cat","O2_cat","T1_story0","S2_scene0"):
        old = read(args.baseline/name/"greedy.json")
        prefix = tuple(StrokeElement(np.asarray(r["points"],np.float32),r["page_id"],False,r["kind"])
                       for r in old["strokes"] if not r["generated"])
        generation[name] = {}
        (args.out_dir/name).mkdir(exist_ok=True)
        for feedback in ("curve","latent"):
            value = generator.generate(prefix, max_new_strokes=old["max_new_strokes"],max_new_pages=old["max_new_pages"],
                seed=old["seed"],temperature=old["temperature"],feedback_mode=feedback)
            write_json(args.out_dir/name/f"{feedback}.json",value)
            save_native_svg(value,args.out_dir/name/f"{feedback}.svg",name+" / "+feedback)
            generation[name][feedback] = {key:value[key] for key in ("stop_reason","generated_strokes","generated_pages","outside_page_strokes")}
        print(json.dumps(dict(generation_case=name,**generation[name]),ensure_ascii=False),flush=True)
    write_json(args.out_dir/"feedback_generation.json",generation)
    after = model_hash(generator.model)
    if before != after:
        raise AssertionError("诊断改变了权重")
    write_json(args.out_dir/"completion.json",dict(training=False,optimizer_steps=0,weight_hash_before=before,
        weight_hash_after=after,weights_unchanged=before==after,cases=len(sources),targets=len(all_rows),
        gradient_probe="复用已保存结果" if args.generation_only else "只回传，不更新",
        hash_scope="generation_only" if args.generation_only else "full_diagnosis",
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))
    print(json.dumps(dict(complete=True,weights_unchanged=True,targets=len(all_rows))),flush=True)


if __name__=="__main__":
    main()
