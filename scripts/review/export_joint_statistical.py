"""2026-10-27：计算节点上的有界分片导出；原始索引与源文件只读。"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import random
import socket
import struct
import sys
import time


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def memory_status():
    result={}
    try:
        import resource
        result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:pass
    try:
        relative=next(line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        base=Path('/sys/fs/cgroup')/relative.lstrip('/')
        result['cgroup']=relative
        for key in ('memory.current','memory.peak','memory.max','memory.events'):
            p=base/key
            if p.exists():result[key]=p.read_text().strip()
    except (OSError,StopIteration):pass
    return result


class Table:
    def __init__(self,path,columns=1):
        self.file=Path(path).open('rb');f=self.file
        if f.read(6)!=b'\x93NUMPY':raise ValueError('非NPY索引')
        major,minor=f.read(2)
        size=struct.unpack('<H' if major==1 else '<I',f.read(2 if major==1 else 4))[0]
        header=ast.literal_eval(f.read(size).decode('latin1'))
        if header['descr'] not in ('<u8','=u8') or header['fortran_order']:raise ValueError('索引格式不同')
        if math.prod(header['shape'][1:])!=columns:raise ValueError('索引列数不同')
        self.start=f.tell();self.columns=columns;self.count=header['shape'][0]
        self.map=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)

    def get(self,index):
        if not 0<=index<self.count:raise IndexError(index)
        return struct.unpack_from('<'+'Q'*self.columns,self.map,self.start+index*self.columns*8)

    def close(self):
        self.map.close();self.file.close()


def read_record(handle,offset,limit):
    handle.seek(offset)
    line=handle.readline(limit+1)
    if len(line)>limit:raise ValueError('单条RAW记录超过有界读取限制；停止且不替换样本')
    return json.loads(line)


def check_host():
    host=socket.gethostname()
    if host.startswith('hpclogin'):
        raise RuntimeError('禁止在登录节点导出；请通过Slurm CPU作业运行')
    if os.name!='nt' and os.getenv("COSE_REQUIRE_SLURM") == "1" and not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('服务器导出必须具有Slurm作业环境')
    return host


def export_split(args,manifest,source,split,split_no,valid_count):
    root=args.out;index=Path(manifest['raw_index']);raw=Path(manifest['raw_dir'])
    info=source['splits'][split];size=info['source']['count'];classes=info['source']['classes']
    ends=Table(index/f'{split}_class_ends.npy')
    offsets=Table(index/f'{split}_offsets.npy',2)
    scene=Path(manifest['scene_index'])/split
    scene_info=json.loads((scene/'manifest.json').read_text())
    scene_offsets=Table(scene/'offsets.npy')
    try:
        plan_path=root/f'{split}_plan.json'
        if plan_path.exists():
            plan=json.loads(plan_path.read_text())
        else:
            rng=random.Random(args.seed+split_no)
            multiplier=104729
            while math.gcd(multiplier,size)!=1:multiplier+=2
            inverse=pow(multiplier,-1,size);objects=[];counts=[]
            for c,label in enumerate(classes):
                start,stop=ends.get(c)[0],ends.get(c+1)[0]
                selected=rng.sample(range(start,stop),min(args.objects_per_class,stop-start))
                counts.append(dict(label=label,population=stop-start,sampled=len(selected)))
                objects.extend(dict(label=label,raw_index=i,index=((i-manifest['seed'])*inverse)%size) for i in selected)
            scene_ids=(list(range(scene_info['count'])) if split=='valid' else
                       sorted(rng.sample(range(valid_count,scene_info['count']),args.train_scenes)))
            plan=dict(objects=objects,class_counts=counts,scene_indices=scene_ids)
            atomic_json(plan_path,plan)
        ids={o['raw_index'] for o in plan['objects']}
        recipes=root/f'{split}_recipes.jsonl';temporary=recipes.with_suffix('.jsonl.tmp')
        with temporary.open('w',encoding='utf-8') as f, (scene/'recipes.jsonl').open('rb') as source_file:
            for i in plan['scene_indices']:
                recipe=read_record(source_file,scene_offsets.get(i)[0],args.max_record_bytes)
                ids.update(o['source_index'] for o in recipe['objects'])
                f.write(json.dumps(dict(index=i,recipe=recipe),ensure_ascii=False,separators=(',',':'))+'\n')
        temporary.replace(recipes)
        ordered=sorted(ids);state_path=root/f'{split}_state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else dict(shards=[])
        keys=set();recognized=Counter();shards=[]
        for begin in range(0,len(ordered),args.shard_records):
            target=root/f'{split}_records_{begin:07d}.jsonl'
            existing=next((s for s in state['shards'] if s['file']==target.name),None)
            if existing:
                if not target.exists() or digest(target)!=existing['sha256']:raise ValueError('已完成分片校验失败')
                for key in existing['keys']:
                    if key in keys:raise ValueError('同分区绘图ID重复')
                    keys.add(key)
                recognized.update(existing['recognized']);shards.append(existing);continue
            part=target.with_suffix('.jsonl.tmp');part_keys=[];part_recognized=Counter()
            current_file=None;handle=None;started=time.monotonic()
            try:
                with part.open('w',encoding='utf-8') as f:
                    for i in ordered[begin:begin+args.shard_records]:
                        file_id,offset=offsets.get(i)
                        if file_id!=current_file:
                            if handle is not None:handle.close()
                            handle=(raw/info['files'][file_id]).open('rb')
                            if hasattr(os,'posix_fadvise'):
                                try:os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_RANDOM)
                                except OSError:pass
                            current_file=file_id
                        record=read_record(handle,offset,args.max_record_bytes)
                        record={k:record[k] for k in ('word','key_id','drawing','recognized') if k in record}
                        key=str(record['key_id'])
                        if key in keys:raise ValueError('同分区绘图ID重复')
                        keys.add(key);part_keys.append(key);part_recognized[str(record.get('recognized'))]+=1
                        f.write(json.dumps(dict(index=i,record=record),ensure_ascii=False,separators=(',',':'))+'\n')
            finally:
                if handle is not None:handle.close()
            part.replace(target)
            item=dict(file=target.name,count=len(part_keys),bytes=target.stat().st_size,
                      sha256=digest(target),keys=part_keys,recognized=dict(part_recognized))
            shards.append(item);recognized.update(part_recognized)
            atomic_json(state_path,dict(shards=shards))
            print(json.dumps(dict(split=split,completed=min(begin+args.shard_records,len(ordered)),total=len(ordered),
                seconds=time.monotonic()-started,memory=memory_status())),flush=True)
        result=dict(raw_manifest=info['source'],recipe_count=scene_info['count'],scene_indices=plan['scene_indices'],
                    objects=plan['objects'],class_counts=plan['class_counts'],records_files=[s['file'] for s in shards],
                    records_shards=[{k:v for k,v in s.items() if k!='keys'} for s in shards],recipes_file=recipes.name,
                    recipes_sha256=digest(recipes),unique_raw_records=len(ordered),unique_drawing_keys=len(keys),
                    recognized=dict(recognized),record_bytes=sum(s['bytes'] for s in shards))
        return result,keys
    finally:
        ends.close();offsets.close();scene_offsets.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--objects_per_class',type=int,default=100);p.add_argument('--train_scenes',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20261027);p.add_argument('--shard_records',type=int,default=4000)
    p.add_argument('--max_record_bytes',type=int,default=16*1024*1024);p.add_argument('--resume',action='store_true')
    args=p.parse_args();host=check_host()
    if min(args.objects_per_class,args.train_scenes,args.shard_records,args.max_record_bytes)<=0:p.error('数量与边界须为正')
    args.out.mkdir(parents=True,exist_ok=args.resume)
    spec=dict(seed=args.seed,objects_per_class=args.objects_per_class,train_scenes=args.train_scenes,
              shard_records=args.shard_records,source_manifest_sha256=digest(args.manifest))
    state=args.out/'export_spec.json'
    if state.exists() and json.loads(state.read_text())!=spec:raise ValueError('恢复时抽样配置或来源不同')
    atomic_json(state,spec)
    manifest=json.loads(args.manifest.read_text());index=Path(manifest['raw_index'])
    source=json.loads((index/'manifest.json').read_text())
    valid_count=json.loads((Path(manifest['scene_index'])/'valid'/'manifest.json').read_text())['count']
    result=dict(version='2026-10-27-expanded-evaluation-sharded-v2',seed=args.seed,source_manifest=manifest,
                object_sampling='uniform_without_replacement_within_each_class',splits={},
                execution=dict(hostname=host,slurm_job_id=os.getenv('SLURM_JOB_ID'),memory=memory_status()))
    keys={}
    for split_no,split in enumerate(('train','valid')):
        result['splits'][split],keys[split]=export_split(args,manifest,source,split,split_no,valid_count)
        atomic_json(args.out/'manifest.partial.json',result)
    result['train_valid_drawing_key_overlap']=len(keys['train']&keys['valid'])
    if result['train_valid_drawing_key_overlap']:raise ValueError('train/valid绘图ID重叠')
    result['execution']['final_memory']=memory_status()
    atomic_json(args.out/'manifest.json',result)
    print(json.dumps(dict(completed=True,execution=result['execution'],splits={s:dict(objects=len(v['objects']),
        scenes=len(v['scene_indices']),records=v['unique_raw_records'],bytes=v['record_bytes'])
        for s,v in result['splits'].items()})),flush=True)


if __name__=='__main__':main()
