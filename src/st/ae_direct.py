"""2026-10-04：正式AE独立笔画源，不通过R任务文档轮转采样。"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing as mp
import os
import signal

import numpy as np
import torch

from src.formal_scene_tasks import realize_recipe
from src.page_document import control_stroke,fit_sketch,selection_page
from .ae_data import StrokeRecord,cap_stroke
from .dct import compute_dct_for_strokes
from .pipeline_pretrain import build_document_dataset


SOURCES=('object','scene','text','aux')
SHARES=(.25,.25,.45,.05)
VERSION='direct_ae_sources_v1'


def _permuted(index,count,seed):
    multiplier=104729
    while math.gcd(multiplier,count)!=1:
        multiplier+=2
    return (index*multiplier+seed)%count


def source_block(dataset,source,index):
    records=[]
    if source=='text':
        document=dataset.get_for_task('T1',index)
        return [StrokeRecord(cap_stroke(e.rawxy),e.kind,index,e.page_id)
                for e in document.to_sequence() if e.kind=='text']
    if source=='object':
        for item in range(index*32,(index+1)*32):
            cls=item%len(dataset.drawings.classes)
            entries=dataset.drawings.class_indices[cls]
            row=_permuted(item//len(dataset.drawings.classes),len(entries),dataset.seed+cls)
            drawing=dataset.drawings.get(int(entries[row]))
            records.extend(StrokeRecord(cap_stroke(points),'sketch',item,0)
                           for points in fit_sketch(drawing.strokes,box=(.06,.10,.94,.86)))
    elif source=='scene':
        for item in range(index*16,(index+1)*16):
            recipe=dataset.recipes[_permuted(item,len(dataset.recipes),dataset.seed)]
            strokes,_,_=realize_recipe(recipe,dataset.drawings)
            records.extend(StrokeRecord(cap_stroke(points),'sketch',item,0) for points in strokes)
    elif source=='aux':
        for item in range(index*64,(index+1)*64):
            kind=('bos','eop','eod','selection')[item%4]
            if kind=='selection':
                recipe=dataset.recipes[_permuted(item//4,len(dataset.recipes),dataset.seed)]
                obj=recipe['objects'][(item//4)%len(recipe['objects'])]
                points=selection_page(obj['selection_box']).elements[0].rawxy
            else:
                points=control_stroke(kind)
            records.append(StrokeRecord(cap_stroke(points),kind,item,0))
    else:
        raise ValueError(source)
    return records


def _worker_init(data_args):
    global worker_dataset
    torch.set_num_threads(1)
    # Slurm停机预告由训练主进程处理；准备进程留给主进程正常关闭。
    if hasattr(signal,'SIGUSR1'):
        signal.signal(signal.SIGUSR1,signal.SIG_IGN)
    worker_dataset=build_document_dataset(**data_args)


def _worker_block(source,index):
    return source_block(worker_dataset,source,index)


class DirectAEStream:
    def __init__(self,data_args,*,batch_size=2048,workers=4):
        if batch_size < 1 or workers < 0:
            raise ValueError('batch_size必须为正，workers不能为负')
        self.dataset=build_document_dataset(**data_args)
        if not data_args.get('formal') or not self.dataset.recipes:
            raise ValueError('独立AE源要求完整正式索引和场景配方')
        self.batch_size=batch_size
        wanted=dict(zip(SOURCES,np.asarray(SHARES)*batch_size))
        self.quotas={name:int(value) for name,value in wanted.items()}
        order=sorted(SOURCES,key=lambda name:wanted[name]-self.quotas[name],reverse=True)
        for name in order[:batch_size-sum(self.quotas.values())]:
            self.quotas[name]+=1
        if min(self.quotas.values())<1:
            raise ValueError('batch太小，至少保证四个源各一笔')
        self.indices={name:0 for name in SOURCES}
        self.offsets={name:0 for name in SOURCES}
        self.pending={}
        self.cache={}
        self.pool=None
        if workers:
            os.environ['OMP_NUM_THREADS']='1'
            os.environ['MKL_NUM_THREADS']='1'
            self.pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                initializer=_worker_init,initargs=(data_args,))

    def _submit(self,name):
        if self.pool:
            queue=self.pending.setdefault(name,{})
            # 准备多个源块并行读盘；游标仍只在实际消费后推进。
            for index in range(self.indices[name],self.indices[name]+4):
                if index in queue or (index==self.indices[name] and name in self.cache):
                    continue
                queue[index]=self.pool.submit(_worker_block,name,index)

    def prefetch(self):
        for name in SOURCES:
            self._submit(name)

    def _next(self,name):
        if name not in self.cache:
            if self.pool:
                self._submit(name)
                self.cache[name]=self.pending[name].pop(self.indices[name]).result()
            else:
                self.cache[name]=source_block(self.dataset,name,self.indices[name])
        block=self.cache[name]
        if not block:
            raise ValueError(f'AE源{name}返回空块')
        record=block[self.offsets[name]]
        self.offsets[name]+=1
        if self.offsets[name]==len(block):
            self.indices[name]+=1
            self.offsets[name]=0
            del self.cache[name]
            self._submit(name)
        return record

    def next_batch(self):
        records=[]
        for name in SOURCES:
            records.extend(self._next(name) for _ in range(self.quotas[name]))
        # 只按整笔长度分组，笔画内部点序保持原序。
        records.sort(key=lambda record:len(record.rawxy))
        return records

    def state_dict(self):
        return {'version':VERSION,'batch_size':self.batch_size,'quotas':self.quotas,
                'indices':dict(self.indices),'offsets':dict(self.offsets)}

    def load_state_dict(self,state):
        if any(state[key]!=self.state_dict()[key] for key in ('version','batch_size','quotas')):
            raise ValueError('AE恢复时独立采样配方不同')
        if self.pending or self.cache:
            raise ValueError('AE游标必须在预取前恢复')
        self.indices,self.offsets=dict(state['indices']),dict(state['offsets'])

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True,cancel_futures=True)
        for source in (self.dataset.drawings,self.dataset.stories,self.dataset.recipes):
            reader=getattr(source,'reader',None)
            if reader:
                reader.close()


class ArcAECollator:
    """统一弧长τ参数，分层随机覆盖整笔；输入坐标/字槽/笔顺保持原协议。"""
    def __init__(self,cfg,seed=42):
        self.cfg=cfg
        self.rng=np.random.default_rng(seed)

    def state_dict(self):
        return {'version':'arc_stratified_v1','rng':self.rng.bit_generator.state}

    def load_state_dict(self,state):
        if state['version']!='arc_stratified_v1':
            raise ValueError('AE恢复时τ参数化不同')
        self.rng.bit_generator.state=state['rng']

    def __call__(self,records):
        lengths=np.asarray([len(record.rawxy) for record in records],np.int64)
        ink=np.zeros((len(records),int(lengths.max()),2),np.float32)
        n=self.cfg.n_t_samples
        tau=((np.arange(n)[None,:]+self.rng.random((len(records),n)))/n).astype(np.float32)
        target=np.zeros((len(records),n,2),np.float32)
        endpoints=[]
        for i,record in enumerate(records):
            points=(record.rawxy-np.array([.5,.6],np.float32))/.6
            ink[i,:len(points)]=points
            distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
            if distance[-1]<1e-12:
                target[i]=points[0]
            else:
                keep=np.r_[True,np.diff(distance)>0]
                target[i]=np.column_stack([np.interp(tau[i]*distance[-1],distance[keep],points[keep,axis])
                                          for axis in range(2)])
            endpoints.append(points[[0,-1]])
        result={'ink':torch.from_numpy(ink),'stroke_len':torch.from_numpy(lengths),
                'tau':torch.from_numpy(tau),'tau_target':torch.from_numpy(target),
                'end_target':torch.from_numpy(np.stack(endpoints)),'kinds':tuple(r.kind for r in records)}
        # 2026-10-05：用户指定无DCT臂，头/损失/目标计算一并移除。
        if self.cfg.dct_coeffs > 0:
            result['dct_target']=torch.from_numpy(compute_dct_for_strokes(ink,lengths,128,
                self.cfg.dct_coeffs,include_mean=self.cfg.dct_dc))
        return result


def microbatches(records,collator,*,documents=1024,point_budget=65536):
    chunk=[]
    for record in records:
        if chunk and (len(chunk)>=documents or len(record.rawxy)*(len(chunk)+1)>point_budget):
            yield collator(chunk)
            chunk=[]
        chunk.append(record)
    if chunk:
        yield collator(chunk)
