"""2026-11-03：冻结 AE 的训练分布几何、字形区分和解码局部稳定性。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist
import torch

from scripts.review.latent_geometry_metrics import (spectrum, dimension_profile, clusters,
    quantiles, stratified_bootstrap_accuracy, wilson_interval)
from src.st.page_diffusion_codec import FrozenPageCodec, file_sha256
from src.st.generation_pretrain import PageGenerator
from src.st.ae_geometry import arc_resample
from src.st.ae_data import cap_stroke
from src.text_renderer import VectorFont
from src.text_augmentation import augment_rendered_text


def write(path, value):
    path=Path(path)
    temporary=path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    temporary.replace(path)


def load_codec(args):
    state=torch.load(args.ae,map_location='cpu',weights_only=False)
    codec=FrozenPageCodec.from_state(state,encode_chunk_size=512).to(args.device).eval()
    del state
    gc.collect()
    return codec,PageGenerator(codec)


def normalization(args):
    data=json.loads(args.normalization.read_text(encoding='utf8'))
    return np.asarray(data['mean'],np.float32),data['scale']


def check_sources(args):
    path=args.out/'review_source.json'
    old=json.loads(path.read_text(encoding='utf8')) if path.exists() else None
    files={}
    for name,source in (('ae',args.ae),('normalization',args.normalization),('font',args.font)):
        stat=source.stat()
        item=dict(path=str(source.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        previous=old['files'].get(name) if old else None
        if previous and all(previous.get(k)==v for k,v in item.items()):
            item['sha256']=previous['sha256']
        else:
            item['sha256']=file_sha256(source)
        if previous and previous['sha256']!=item['sha256']:
            raise ValueError('已有输出的AE、字体或归一化来源改变，请使用独立目录')
        files[name]=item
    if old and old['seed']!=args.seed:
        raise ValueError('已有输出的抽样种子不同，请使用独立目录')
    for p in args.out.glob('*_encoding.json'):
        metadata=json.loads(p.read_text(encoding='utf8'))
        if metadata['ae_sha256']!=files['ae']['sha256'] or metadata['normalization_sha256']!=files['normalization']['sha256']:
            raise ValueError(f'{p.name}的嵌入缓存来源不一致')
    glyph=args.out/'glyph_source.json'
    if glyph.exists():
        metadata=json.loads(glyph.read_text(encoding='utf8'))
        if any(metadata[k]!=v for k,v in [('ae_sha256',files['ae']['sha256']),('font_sha256',files['font']['sha256']),('seed',args.seed)]):
            raise ValueError('字形缓存来源不一致')
    expected=dict(seed=args.seed,files=files,scope='frozen_AE_only; no_R_inference_or_training')
    if old!=expected:
        write(path,expected)


def prepare_validation(args):
    from scripts.review.joint_eval_batch import load_dataset
    from scripts.review.export_latent_geometry import Reservoir
    destination=args.out/'data'/'validation_pipeline.npz'
    if destination.exists():
        return
    destination.parent.mkdir(exist_ok=True)
    font=VectorFont.from_svg(args.font)
    dataset,info=load_dataset(args.validation,'valid',font)
    plan=json.loads((args.validation/'valid_plan.json').read_text(encoding='utf8'))
    rng=np.random.default_rng(args.seed)
    sample=Reservoir(100000,args.seed)
    labels=Counter()
    for number,entry in enumerate(plan['objects']):
        doc=dataset.get_for_task('O1',entry['index'])
        strokes=[e for e in doc.to_sequence() if e.kind=='sketch']
        element=strokes[int(rng.integers(len(strokes)))]
        sample.add(cap_stroke(element.rawxy,76),group='object',kind='sketch',task='O1',
                   unit=f'object:{doc.metadata["source_id"]}',label=entry['label'],target=True)
        labels[entry['label']]+=1
        if (number+1)%5000==0:
            print(json.dumps(dict(stage='validation_objects',objects=number+1)),flush=True)
    scene_ids=rng.choice(plan['scene_indices'],5000,replace=False)
    for number,index in enumerate(scene_ids):
        doc=dataset.get_for_task('S2',int(index))
        strokes=[e for e in doc.to_sequence() if e.kind=='sketch']
        chosen=rng.choice(len(strokes),min(2,len(strokes)),replace=False)
        for i in sorted(chosen):
            sample.add(cap_stroke(strokes[i].rawxy,76),group='scene',kind='sketch',task='S2',
                       unit=f'scene:{index}',target=True)
        if (number+1)%1000==0:
            print(json.dumps(dict(stage='validation_scenes',scenes=number+1)),flush=True)
    report=sample.save(destination)
    report.update(object_class_counts=dict(labels),scenes=len(scene_ids),seed=args.seed,
                  source_manifest_sha256=file_sha256(args.validation/'manifest.json'),
                  sampling='all 345x100 independent valid objects, one uniformly sampled stroke each; 5000 uniform valid scenes, up to two strokes each')
    write(destination.with_suffix('.json'),report)


@torch.no_grad()
def encode_curves(geometry, curves, *, fp16=True, label=''):
    lengths=np.asarray([len(x) for x in curves])
    result=np.empty((len(curves),48),np.float32)
    finished=0
    for lower,upper in ((0,8),(8,16),(16,32),(32,48),(48,76)):
        ids=np.flatnonzero((lengths>lower)&(lengths<=upper))
        for start in range(0,len(ids),512):
            selected=ids[start:start+512]
            with torch.autocast('cuda',dtype=torch.float16,enabled=fp16 and geometry.device.type=='cuda'):
                z=geometry.encode([curves[i] for i in selected])
            result[selected]=z.float().cpu().numpy()
            finished+=len(selected)
        print(json.dumps(dict(stage='encode',label=label,bucket=upper,finished=finished,total=len(curves))),flush=True)
    if finished!=len(curves):
        raise ValueError('曲线点数超出真实 AE 输入范围')
    return result


def encode_pipeline(args):
    codec,geometry=load_codec(args)
    mean,scale=normalization(args)
    for name in ('ae','r','text','validation'):
        source=args.out/'data'/f'{name}_pipeline.npz'
        destination=args.out/f'{name}_encoded.npz'
        if destination.exists() or not source.exists() or not source.with_suffix('.json').exists():
            continue
        with np.load(source) as data:
            points,offsets=data['points'],data['offsets']
            curves=[points[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)]
            latent=encode_curves(geometry,curves,label=name)
            z=(latent-mean)/scale
            rng=np.random.default_rng(args.seed)
            fp32_indices=rng.choice(len(curves),min(8000,len(curves)),replace=False)
            z32=(encode_curves(geometry,[curves[i] for i in fp32_indices],fp16=False,label=name+'_fp32')-mean)/scale
            np.savez_compressed(destination,z=z,latent=latent,fp32_indices=fp32_indices,z_fp32=z32,
                **{k:data[k] for k in ('group','kind','task','unit','label','target','window')})
            write(args.out/f'{name}_encoding.json',dict(n=len(z),source_sha256=file_sha256(source),
                ae_sha256=file_sha256(args.ae),normalization_sha256=file_sha256(args.normalization),
                fp32_difference_norm=quantiles(np.linalg.norm(z32-z[fp32_indices],axis=1)),
                dtype='FP16 E, FP32 stored latent and saved normalization',counts=dict(Counter(data['kind']))))
            print(json.dumps(dict(stage='encoded',mode=name,n=len(z))),flush=True)
        del curves,points,latent,z
        gc.collect()


def profile_one(args,name,z,kind,unit,mask):
    destination=args.out/'profiles'/f'{name}.json'
    if destination.exists():
        return
    destination.parent.mkdir(exist_ok=True)
    indices=np.flatnonzero(mask)
    points=z[indices]
    continuous=~np.isin(kind[indices],['bos','eop','eod'])
    result=dict(name=name,n=len(points),source_units=len(np.unique(unit[indices])),
                kinds=dict(Counter(kind[indices])),spectrum=spectrum(points),
                continuous_n=int(continuous.sum()),
                continuous_spectrum=spectrum(points[continuous]),
                dimension=dimension_profile(points[continuous],seed=args.seed),
                clustering=clusters(points[continuous],seed=args.seed))
    write(destination,result)
    print(json.dumps(dict(stage='profile',name=name,n=len(points),pr=result['spectrum']['participation_ratio'],
        two_nn=[r['two_nn_mle'] for r in result['dimension']['runs']])),flush=True)


def analyze_pipeline(args):
    for corpus in ('ae','r','text','validation'):
        if not (args.out/f'{corpus}_encoded.npz').exists():
            continue
        with np.load(args.out/f'{corpus}_encoded.npz') as data:
            z,kind,unit,group=data['z'],data['kind'],data['unit'],data['group']
            profiles={'overall':np.ones(len(z),bool),
                      **{g:group==g for g in ('text','object','scene')}}
            if corpus=='ae':
                profiles['body']=np.isin(kind,['text','sketch'])
            if corpus=='text':
                profiles={'stratified_text':np.ones(len(z),bool)}
            if corpus=='validation':
                profiles={g:group==g for g in ('object','scene')}
            if corpus=='r':
                profiles['targets']=data['target']&~np.isin(kind,['bos','eop','eod'])
            for name,mask in profiles.items():
                if mask.sum()>50:
                    profile_one(args,f'{corpus}_{name}',z,kind,unit,mask)
            precision=args.out/'profiles'/f'{corpus}_precision.json'
            if not precision.exists():
                keep=~np.isin(kind[data['fp32_indices']],['bos','eop','eod'])
                write(precision,dict(fp16=dimension_profile(z[data['fp32_indices']][keep],sizes=(4000,),repeats=1),
                    fp32=dimension_profile(data['z_fp32'][keep],sizes=(4000,),repeats=1)))
    calibration=args.out/'profiles'/'calibration.json'
    if not calibration.exists():
        rng=np.random.default_rng(args.seed)
        result={}
        for dimension in (3,8,24,48):
            rotation=np.linalg.qr(rng.normal(size=(48,dimension)))[0]
            x=rng.normal(size=(16000,dimension))@rotation.T
            result[str(dimension)]=dimension_profile(x,sizes=(4000,16000),repeats=1)
            print(json.dumps(dict(stage='calibration',dimension=dimension)),flush=True)
        write(calibration,result)


def calibrate_covariance(args):
    rng=np.random.default_rng(args.seed+21)
    for path in sorted((args.out/'profiles').glob('*.json')):
        data=json.loads(path.read_text(encoding='utf8'))
        if 'spectrum' not in data:
            continue
        output=path.with_name(path.stem+'_gaussian_dimension.json')
        if output.exists():
            continue
        if 'continuous_spectrum' in data:
            cov=np.asarray(data['continuous_spectrum']['covariance'])
        elif data['continuous_n']!=data['n']:
            corpus=data['name'].split('_')[0]
            with np.load(args.out/f'{corpus}_encoded.npz') as encoded:
                keep=~np.isin(encoded['kind'],['bos','eop','eod'])
                if data['name']!=f'{corpus}_overall' or int(keep.sum())!=data['continuous_n']:
                    raise ValueError('需显式定位与维度估计相同的连续样本')
                cov=np.cov(encoded['z'][keep].astype(np.float64),rowvar=False)
        else:
            cov=np.asarray(data['spectrum']['covariance'])
        eig,vec=np.linalg.eigh(cov)
        synthetic=rng.normal(size=(16000,48))@(vec*np.sqrt(np.maximum(eig,0))).T
        write(output,dict(name=data['name'],purpose='finite-sample dimension baseline with matched continuous-sample covariance',
            covariance_rank=int(np.linalg.matrix_rank(cov)),
            dimension=dimension_profile(synthetic,sizes=(4000,16000),repeats=1,seed=args.seed)))
        print(json.dumps(dict(stage='covariance_dimension_calibration',name=data['name'])),flush=True)


def control_geometry(args):
    destination=args.out/'control_geometry.json'
    if destination.exists():
        return
    with np.load(args.out/'ae_encoded.npz') as data:
        z,kind=data['z'],data['kind']
        names=('bos','eop','eod')
        prototypes=np.stack([z[kind==name].mean(0) for name in names])
        body=z[np.isin(kind,['text','sketch'])]
        distances=cdist(prototypes,body)
        result=dict(names=names,prototype_distances=cdist(prototypes,prototypes).tolist(),
            nearest_body={name:float(distances[i].min()) for i,name in enumerate(names)},
            body_distance={name:quantiles(distances[i]) for i,name in enumerate(names)},noise={})
        rng=np.random.default_rng(args.seed)
        truth=np.repeat(np.arange(3),4096)
        eps=rng.normal(size=(len(truth),48))
        for t in (0.,.01,.03,.1,.2,.4,.6,.8):
            query=(1-t)*prototypes[truth]+t*eps
            prediction=cdist(query,(1-t)*prototypes).argmin(1)
            result['noise'][str(t)]={name:float((prediction[truth==i]==i).mean()) for i,name in enumerate(names)}
        result['draws_per_prototype']=4096
        result['noise_ci95']={t:{name:wilson_interval(round(value*4096),4096) for name,value in rates.items()}
                              for t,rates in result['noise'].items()}
        result['interval']='Wilson binomial interval for independent Gaussian draws, conditional on fixed prototypes'
        result['scope']='three fixed control prototypes; 4096 independent Gaussian draws per prototype; not stopping accuracy'
        write(destination,result)


def make_glyphs(args):
    output=args.out/'glyphs.npz'
    if output.exists():
        return
    codec,geometry=load_codec(args)
    font=VectorFont.from_svg(args.font)
    characters=sorted(c for c in font.glyphs if len(c)==1 and 33<=ord(c)<=126 and font.render(c).strokes)
    rng=np.random.default_rng(args.seed)
    curves,raw,labels,parts,splits,origins=[],[],[],[],[],[]
    for label,c in enumerate(characters):
        base=font.render(c,font_size=.085,origin=(.35,.45))
        for split,count in ((0,128),(1,256)):
            for _ in range(count):
                rendered=augment_rendered_text(base,font_size=.085,rng=rng)
                original=[cap_stroke(s,76) for s in rendered.strokes]
                curves.extend(original)
                raw.append(np.stack([arc_resample(s,32) for s in original]))
                parts.append(len(original));labels.append(label);splits.append(split)
                origins.append([rng.uniform(.06,.84),rng.uniform(.15,.82)])
        if label%16==0:
            print(json.dumps(dict(stage='glyph_prepare',characters=label+1)),flush=True)
    mean,scale=normalization(args)
    z=(encode_curves(geometry,curves,label='glyph_fixed_position')-mean)/scale
    offsets=np.r_[0,np.cumsum(parts)]
    max_parts=max(parts)
    latent=np.zeros((len(parts),max_parts,48),np.float32)
    raw_features=np.zeros((len(parts),max_parts,32,2),np.float32)
    moved=[]
    for i,(begin,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        latent[i,:parts[i]]=z[begin:end]
        raw_features[i,:parts[i]]=raw[i]
        shift=np.asarray(origins[i],np.float32)-[.35,.45]
        moved.extend([(x+shift).astype(np.float32) for x in curves[begin:end]])
    moved_z=(encode_curves(geometry,moved,label='glyph_varied_position')-mean)/scale
    varied=np.zeros_like(latent)
    for i,(begin,end) in enumerate(zip(offsets[:-1],offsets[1:])):
        varied[i,:parts[i]]=moved_z[begin:end]
    np.savez_compressed(output,latent=latent,varied=varied,raw=raw_features,parts=parts,labels=labels,
                        splits=splits,origins=origins,characters=characters)
    write(args.out/'glyph_source.json',dict(characters=characters,reference_per_class=128,query_per_class=256,
        font_sha256=file_sha256(args.font),ae_sha256=file_sha256(args.ae),seed=args.seed,
        augmentation='production augment_rendered_text; fixed and independent varied origin; font_size=.085',
        comparison='ordered full-glyph stroke representations, comparisons restricted to same actual stroke count'))


def retrieve(reference,query,reference_parts,query_parts,reference_labels,*,device='cuda'):
    prediction=np.empty(len(query),np.int64)
    best_distance=np.empty(len(query),np.float64)
    # 每类笔数固定，因此分组避免 padding 长度伪造距离；笔数是此 AE 读出探针的已知条件。
    for n in np.unique(query_parts):
        ri=np.flatnonzero(reference_parts==n); qi=np.flatnonzero(query_parts==n)
        r=torch.as_tensor(reference[ri,:n].reshape(len(ri),-1),device=device,dtype=torch.float32)
        for start in range(0,len(qi),128):
            selected=qi[start:start+128]
            q=torch.as_tensor(query[selected,:n].reshape(len(selected),-1),device=device,dtype=torch.float32)
            distances=torch.cdist(q,r,compute_mode='donot_use_mm_for_euclid_dist')
            values,ids=distances.min(1)
            prediction[selected]=reference_labels[ri[ids.cpu().numpy()]]
            best_distance[selected]=values.cpu().numpy()
    return prediction,best_distance


def analyze_glyphs(args):
    destination=args.out/'glyph_metrics.json'
    if destination.exists():
        return
    with np.load(args.out/'glyphs.npz') as data:
        ref=data['splits']==0; query=~ref
        labels,parts=data['labels'],data['parts']
        result=dict(characters=data['characters'].tolist(),modes={},noise={},
            chance_given_stroke_count=float(len(np.unique(parts))/len(np.unique(labels))),
            reference_per_class=128,query_per_class=256,
            conditioning='known true glyph boundaries and stroke count; no prediction model or learned classifier')
        predictions={}
        for name,features in (('latent_fixed',data['latent']),('latent_varied',data['varied']),('raw_fixed',data['raw'])):
            prediction,distance=retrieve(features[ref],features[query],parts[ref],parts[query],labels[ref],device=args.device)
            result['modes'][name]=stratified_bootstrap_accuracy(prediction==labels[query],labels[query],seed=args.seed)
            result['modes'][name]['nearest_distance']=quantiles(distance)
            confusion=np.zeros((len(result['characters']),)*2,np.int64)
            np.add.at(confusion,(labels[query],prediction),1)
            result['modes'][name]['confusion']=confusion.tolist()
            predictions[name]=prediction
            print(json.dumps(dict(stage='glyph_knn',mode=name,macro=result['modes'][name]['macro_recall'])),flush=True)
        rng=np.random.default_rng(args.seed+1)
        epsilon=rng.normal(size=data['latent'][query].shape).astype(np.float32)
        for t in (0.,.005,.01,.02,.04,.08,.16,.32,.5):
            # 缩放全部参考为(1-t)z等价于查询z+t/(1-t)epsilon，保持实际线性加噪路径。
            noisy=data['latent'][query]+t/(1-t)*epsilon
            prediction,_=retrieve(data['latent'][ref],noisy,parts[ref],parts[query],labels[ref],device=args.device)
            result['noise'][str(t)]=stratified_bootstrap_accuracy(prediction==labels[query],labels[query],seed=args.seed)
            print(json.dumps(dict(stage='glyph_noise',t=t,macro=result['noise'][str(t)]['macro_recall'])),flush=True)
        result['noise_sampling']='same independent Gaussian per query reused across t; unused padded slots excluded'
        write(destination,result)
        np.savez_compressed(args.out/'glyph_predictions.npz',truth=labels[query],**predictions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=('encode','analyze','glyph_prepare','glyph_analyze','glyph_geometry','strokes','position','validation_prepare','smooth','calibrate_covariance','controls'),required=True)
    p.add_argument('--ae',type=Path,default=Path('E:/Trash_things/cose/tmp_smoke/r_dual_review_assets_20261009/ae_final.pt'))
    p.add_argument('--normalization',type=Path,default=Path('E:/Trash_things/cose/tmp_smoke/r_page_diffusion_final_20261103/run/normalization.json'))
    p.add_argument('--font',type=Path,default=Path('E:/Trash_things/cose/datasets/vector_fonts/relief/ReliefSingleLineSVG-Regular.svg'))
    p.add_argument('--validation',type=Path,default=Path('E:/Trash_things/cose/tmp_smoke/r_joint_statistical_data_20261027'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=20261103)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    check_sources(args)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    started=time.monotonic()
    from scripts.review.latent_geometry_smooth import run as smooth
    from scripts.review.latent_geometry_glyph import run as glyph_geometry
    from scripts.review.latent_geometry_strokes import run as strokes
    from scripts.review.latent_geometry_position import run as position
    {'encode':encode_pipeline,'analyze':analyze_pipeline,'smooth':smooth,
     'glyph_prepare':make_glyphs,'glyph_analyze':analyze_glyphs,'glyph_geometry':glyph_geometry,
     'strokes':strokes,'position':position,'validation_prepare':prepare_validation,'calibrate_covariance':calibrate_covariance,
     'controls':control_geometry}[args.phase](args)
    print(json.dumps(dict(stage='finished',phase=args.phase,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    main()
