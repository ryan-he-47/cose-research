"""2026-10-04：AE原坐标Chamfer、形变/平移贡献和分桶λ轮廓系数。"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from src.st.ae_data import cap_stroke
from src.st.formal_assets import write_json
from src.st.pipeline_pretrain import build_document_dataset
from src.st.train_ae import StrokeAutoencoder, config_from_checkpoint, load_ae_weights
from src.formal_scene_tasks import realize_recipe
from scripts.legacy.diag_emb_compare import PartA
from src.st.ae_geometry import arc_resample, chamfer, error_components, summarize


def silhouette(embeddings,limit,seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from threadpoolctl import threadpool_limits
    indices=np.random.default_rng(seed).choice(len(embeddings),min(limit,len(embeddings)),replace=False)
    values=embeddings[indices].astype(np.float64)
    results={}
    with threadpool_limits(limits=4):
        for metric in ('euclidean','sqeuclidean','cosine'):
            points=values.copy()
            if metric=='cosine':
                points/=np.maximum(np.linalg.norm(points,axis=1,keepdims=True),1e-12)
            scores={}
            for k in (5,10,15,20,25):
                labels=KMeans(n_clusters=k,random_state=0,n_init=10).fit_predict(points)
                scores[str(k)]=float(silhouette_score(points,labels,metric=metric))
            results[metric]={'by_k':scores,'mean':float(np.mean(list(scores.values())))}
    return {'samples':len(values),'clusters':'KMeans n_init=10, random_state=0',
            'cosine_input':'unit_norm','metrics':results}


def encode_records(raw,bucket,metadata):
    capped=[cap_stroke(points,76) for points in raw]
    ink=np.zeros((len(raw),76,2),np.float32)
    lengths=np.array([len(points) for points in capped],np.int64)
    for i,points in enumerate(capped):
        ink[i,:len(points)]=(points-np.array([.5,.6],np.float32))/.6
    return {'ink':ink,'lengths':lengths,'cloud':np.stack([arc_resample(points,32) for points in raw]),
            'cap_cloud':np.stack([arc_resample(points,32) for points in capped]),
            'bucket':np.asarray(bucket),'metadata':np.asarray([json.dumps(m,ensure_ascii=False) for m in metadata])}


def prepare_history(path):
    # 现行工具的原始历史样本/顺序；只取同一cycle_cd使用的前300笔。
    data=PartA('cpu')
    records=encode_records(data.canvas[:300],['historical_graphic']*300,
        [{'legacy_index':i,'class_id':int(data.cls[i])} for i in range(300)])
    np.savez_compressed(path,**records)
    print(json.dumps({'history_samples':str(path),'count':300,
        'class_counts':dict(Counter(int(i) for i in data.cls[:300]))}),flush=True)


def prepare_formal(data_args,count,path):
    data=build_document_dataset(**{**data_args,'split':'valid'})
    rng=np.random.default_rng(8765)
    raw,buckets,metadata=[],[],[]
    # 各故事随机取64笔，避免只取前几行或把一个长故事当全部文字分布。
    text_count,story_index=0,0
    while text_count<count:
        document=data.get_for_task('T1',story_index)
        elements=[element for element in document.to_sequence() if element.kind=='text']
        chosen=rng.choice(len(elements),min(64,len(elements),count-text_count),replace=False)
        for index in chosen:
            element=elements[int(index)]
            raw.append(element.rawxy); buckets.append('text')
            metadata.append({'story':document.metadata['source_id'],'page_id':element.page_id,'char_index':element.char_index})
        text_count+=len(chosen); story_index+=1
    # 图形分对象页/连续场景两组，每组与文字同样笔数；对象页覆盖全部345类。
    for index in range(count):
        document=data.get_for_task('O1',index)
        elements=[element for element in document.to_sequence() if element.kind=='sketch']
        element=elements[int(rng.integers(len(elements)))]
        raw.append(element.rawxy); buckets.append('sketch_object')
        metadata.append({'label':document.metadata['label'],'key_id':document.metadata['source_id']})
    for index in range(count):
        recipe=data.recipes[index]
        strokes,objects,_=realize_recipe(recipe,data.drawings)
        obj=objects[int(rng.integers(len(objects)))]
        start,end=obj['stroke_range']
        points=strokes[int(rng.integers(start,end))]
        raw.append(points); buckets.append('sketch_scene')
        metadata.append({'label':obj['label'],'key_id':obj['key_id'],'recipe_id':index,'instance_id':obj['instance_id']})
    records=encode_records(raw,buckets,metadata)
    np.savez_compressed(path,**records)
    write_json(path.with_suffix('.json'),{'dataset':data.manifest(),'counts':dict(Counter(buckets)),
        'text_stories':story_index,'object_classes':len(set(m['label'] for m,b in zip(metadata,buckets) if b=='sketch_object'))})
    print(json.dumps({'formal_samples':str(path),'counts':dict(Counter(buckets))}),flush=True)
    return records


def evaluate(checkpoint,records,out,label,device,cluster_count):
    started=time.monotonic()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=config_from_checkpoint(state)
    model=StrokeAutoencoder(cfg).eval().to(device)
    load_ae_weights(model,state)
    step=state.get('step')
    del state; gc.collect()
    latents,reconstructed=[],[]
    with torch.no_grad():
        for start in range(0,len(records['ink']),64):
            ink=torch.from_numpy(records['ink'][start:start+64]).to(device)
            lengths=torch.from_numpy(records['lengths'][start:start+64]).to(device)
            embeddings=model.encoder(ink,lengths)
            # 历史cycle_cd的48个解码τ，再等弧长重采样到32点。
            tau=torch.linspace(0,1,48,device=device).expand(len(ink),-1)
            mu,_,pi=model.decoder(embeddings,tau)
            component=pi.argmax(-1)[...,None,None].expand(-1,-1,1,2)
            points=mu.gather(2,component).squeeze(2)*.6+mu.new_tensor([.5,.6])
            latents.append(embeddings.cpu().numpy())
            reconstructed.extend(arc_resample(points,32) for points in points.cpu().numpy())
    latent=np.concatenate(latents)
    reconstruction=np.stack(reconstructed)
    errors=np.array([error_components(a,b) for a,b in zip(records['cloud'],reconstruction)])
    report={'checkpoint':str(Path(checkpoint).resolve()),'step':step,'precision':'float32',
            'decode':'highest_pi_component_mean,48 tau ->32 equal_arclength',
            'buckets':{},'seconds':time.monotonic()-started}
    for bucket in sorted(set(records['bucket']))+(['sketch'] if 'sketch_object' in records['bucket'] else []):
        mask=np.char.startswith(records['bucket'],'sketch') if bucket=='sketch' else records['bucket']==bucket
        report['buckets'][bucket]=summarize(errors[mask])
        # 图形聚类在汇总桶上做；各子桶仍汇报CD，避免重复计算同一测量。
        if bucket in ('text','sketch','historical_graphic'):
            report['buckets'][bucket]['silhouette']=silhouette(latent[mask],cluster_count,42)
        report['buckets'][bucket]['sparsify_cd']=float(np.mean([chamfer(a,b) for a,b in zip(records['cloud'][mask],records['cap_cloud'][mask])]))
    np.savez_compressed(out/f'{label}_strokes.npz',latent=latent,reconstructed=reconstruction,errors=errors,**records)
    report['seconds']=time.monotonic()-started
    write_json(out/f'{label}.json',report)
    print(json.dumps({'model':label,'step':step,'seconds':report['seconds'],'buckets':{
        name:{key:bucket[key] for key in ('count','shape_share','position_share')}|{'cd':bucket['cd']['mean'],
        'cosine_sc':bucket.get('silhouette',{}).get('metrics',{}).get('cosine',{}).get('mean')}
        for name,bucket in report['buckets'].items()}}),flush=True)
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def check_formulas():
    points=np.column_stack((np.linspace(0,1,32),np.sin(np.linspace(0,3,32))))
    for other in (points,points+[.1,.2],(points-points.mean(0))*1.2+points.mean(0)):
        values=error_components(points,other)
        assert abs(values[0]-values[1]-values[2])<1e-12
    translated=error_components(points,points+[.1,.2])
    assert abs(translated[1])<1e-12 and abs(translated[3])<1e-12


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare_history',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--samples',type=Path,help='2026-10-04：直接复用已保存的同笔样本，避免重建数据池')
    parser.add_argument('--history_samples',type=Path)
    parser.add_argument('--history_only',action='store_true')
    parser.add_argument('--checkpoint',action='append',metavar='LABEL=PATH')
    parser.add_argument('--out_dir',type=Path)
    parser.add_argument('--strokes',type=int,default=2048)
    parser.add_argument('--cluster_strokes',type=int,default=1024)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    torch.set_num_threads(4)
    check_formulas()
    if args.prepare_history:
        args.prepare_history.parent.mkdir(parents=True,exist_ok=True)
        prepare_history(args.prepare_history)
        return
    if (not args.manifest and not args.samples and not args.history_only) or not args.checkpoint or not args.out_dir:
        parser.error('需要manifest或samples，以及checkpoint/out_dir')
    if args.samples and (args.manifest or args.history_only):
        parser.error('samples直接复用样本，不同时指定manifest/history_only')
    if os.getenv("COSE_REQUIRE_SLURM") == "1" and not os.getenv('SLURM_JOB_ID'):
        parser.error('服务器诊断重计算必须经Slurm计算节点')
    args.out_dir.mkdir(parents=True,exist_ok=False)
    records=(dict(np.load(args.samples,allow_pickle=False)) if args.samples else
             dict(np.load(args.history_samples,allow_pickle=False)) if args.history_only else
             prepare_formal(json.loads(args.manifest.read_text()),args.strokes,args.out_dir/'samples.npz'))
    reports={}
    for item in args.checkpoint:
        label,path=item.split('=',1)
        reports[label]=evaluate(path,records,args.out_dir,label,args.device,args.cluster_strokes)
        if args.history_samples and not args.history_only:
            historical=dict(np.load(args.history_samples,allow_pickle=False))
            reports[label]['historical']=evaluate(path,historical,args.out_dir,label+'_historical',args.device,args.cluster_strokes)
    write_json(args.out_dir/'report.json',{'models':reports,'definition':{
        'cd':'mean_a min_b ||a-b|| + mean_b min_a ||a-b||, raw page coordinates',
        'shape_only':'CD(original,reconstruction - reconstructed_centroid + original_centroid)',
        'position_only':'CD(original,original + reconstructed_centroid - original_centroid)',
        'shape_contribution':'(shape_only + total - position_only)/2',
        'position_contribution':'(position_only + total - shape_only)/2',
        'shares':'sum(bucket_contribution)/sum(bucket_total_cd), signed and not clipped',
        'relative_cd':'CD / original stroke bbox diagonal; point strokes <=1e-6 excluded only here','silhouette':'KMeans k=5,10,15,20,25; each metric mean'}})


if __name__=='__main__':
    main()
