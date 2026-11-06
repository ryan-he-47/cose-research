"""2026-10-03：按输入成本选任务，4进程有界预取，恢复按已消费游标重建。"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing as mp
import os
import signal

import torch

from .formal_recipe import TASKS
from .pipeline_pretrain import build_document_dataset, collate_documents


def worker_init(data_args, max_strokes):
    global worker_dataset, worker_context
    torch.set_num_threads(1)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    worker_dataset = build_document_dataset(**data_args)
    worker_context = max_strokes


def worker_document(task, index):
    document = worker_dataset.get_for_task(task, index)
    if task == "T1":
        return list(document.iter_windows(worker_context))
    return [document]


def worker_documents(task, start, count):
    """一次进程通信准备多个源文档，文档种子和片内笔顺保持原定义。"""
    return [worker_document(task, index) for index in range(start, start + count)]


class InputDocumentStream:
    def __init__(self, dataset, data_args, shares, *, max_strokes=1024, workers=4,
                 prefetch_documents=None, prepare_batch_documents=None):
        self.dataset, self.data_args = dataset, data_args
        self.shares, self.max_strokes = dict(shares), max_strokes
        if set(self.shares) != set(dataset.tasks):
            raise ValueError("正式份额与任务清单不同")
        if any(not math.isfinite(value) or value < 0 for value in self.shares.values()):
            raise ValueError("任务份额须为有限非负数")
        if not math.isclose(sum(self.shares.values()), 1., abs_tol=1e-9):
            raise ValueError("输入份额之和须为1")
        self.active_tasks = tuple(task for task in TASKS if self.shares[task] > 0)
        self.indices = {task: 0 for task in TASKS}
        self.offsets = {task: 0 for task in TASKS}
        self.inputs = Counter({task: 0 for task in TASKS})
        self.skipped = Counter()
        self.skip_examples = []
        self.pending = {}
        self.windows = {}
        self.pool = None
        default_depth = max(1, workers) if len(self.active_tasks) == 1 else 1
        self.prefetch_depth = {task: int((prefetch_documents or {}).get(task, default_depth))
                               for task in self.active_tasks}
        self.prepare_batch = {task: int((prepare_batch_documents or {}).get(task, 1))
                              for task in self.active_tasks}
        if any(self.prepare_batch[t] < 1 or self.prefetch_depth[t] < self.prepare_batch[t]
               for t in self.active_tasks):
            raise ValueError('预取深度须不小于准备批量，且两者为正')
        if workers:
            # spawn子进程重新打开mmap；不复制父进程CUDA上下文或全量故事。
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                initializer=worker_init, initargs=(data_args, max_strokes))

    def _submit(self, task):
        if self.pool:
            start = self.indices[task] + int(task in self.windows)
            queued = [index for name, index in self.pending if name == task]
            depth, batch = self.prefetch_depth[task], self.prepare_batch[task]
            # 低水位整批补充；避免每消费一篇就退化为一次单篇IPC。
            if len(queued) > depth // 2:
                return
            frontier = max(queued, default=start - 1) + 1
            end = start + depth
            while frontier < end:
                count = min(batch, end - frontier)
                if count < batch and queued:
                    break
                future = self.pool.submit(worker_documents, task, frontier, count)
                for offset in range(count):
                    self.pending[task, frontier + offset] = (future, offset)
                frontier += count

    def prefetch(self):
        for task in self.active_tasks:
            self._submit(task)

    def _load(self, task):
        if task not in self.windows:
            try:
                if self.pool:
                    self._submit(task)
                    future, offset = self.pending.pop((task,self.indices[task]))
                    documents = future.result()[offset]
                else:
                    document = self.dataset.get_for_task(task, self.indices[task])
                    documents = list(document.iter_windows(self.max_strokes)) if task == "T1" else [document]
            except Exception as error:
                raise RuntimeError(f"{task}源索引{self.indices[task]}准备失败：{error}") from error
            self.windows[task] = documents

    def next_document(self, missing=()):
        for _ in range(1000):
            task = min(missing or self.active_tasks, key=lambda name: self.inputs[name]/self.shares[name])
            self._load(task)
            document = self.windows[task][self.offsets[task]]
            self.offsets[task] += 1
            if self.offsets[task] == len(self.windows[task]):
                self.indices[task] += 1
                self.offsets[task] = 0
                del self.windows[task]
                self._submit(task)
            if document.stroke_count > self.max_strokes:
                self.skipped[task + ":too_long"] += 1
                self.skip_examples.append({"task":task, "strokes":document.stroke_count,
                    "source":document.metadata.get("source_id"), "recipe_id":document.metadata.get("recipe_id")})
                self.skip_examples = self.skip_examples[-32:]
                continue
            count = document.stroke_count-1
            if count <= 0:
                self.skipped[task + ":empty"] += 1
                continue
            self.inputs[task] += count
            return document
        raise RuntimeError("连续1000份文档无法用于正式训练")

    def state_dict(self):
        return {"kind":"formal_input_stream_v1", "indices":dict(self.indices),
                "offsets":dict(self.offsets), "inputs":dict(self.inputs), "shares":self.shares,
                "max_strokes":self.max_strokes, "skipped":dict(self.skipped),
                "skip_examples":list(self.skip_examples)}

    def coverage(self,limits):
        fractions={task:min(1.,self.indices[task]/limit) for task,limit in limits.items()}
        return dict(progress=min(fractions.values()),fractions=fractions,
                    completed_sources=dict(self.indices),source_limits=dict(limits),
                    skipped=dict(self.skipped))

    def load_state_dict(self, state):
        if state["shares"] != self.shares or state["max_strokes"] != self.max_strokes:
            raise ValueError("恢复时任务份额或上下文不同")
        if self.pending or self.windows:
            raise ValueError("先恢复消费游标，再启动预取")
        self.indices, self.offsets = dict(state["indices"]), dict(state["offsets"])
        self.inputs, self.skipped = Counter(state["inputs"]), Counter(state["skipped"])
        self.skip_examples = list(state["skip_examples"])

    def rewind(self, state):
        """2026-10-11：准备失败时回滚到整批消费前；丢弃未消费预取结果。"""
        for future in {entry[0] for entry in self.pending.values()}:
            future.cancel()
        self.pending.clear()
        self.windows.clear()
        self.load_state_dict(state)

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

    def continue_positions(self, state):
        """2026-10-09：新配方接续源游标，份额欠账和统计从新阶段重新累计。"""
        if state['kind'] != 'formal_input_stream_v1' or state['max_strokes'] != self.max_strokes:
            raise ValueError('纠偏阶段需要同上下文的正式消费游标')
        self.load_state_dict({**state, 'shares':self.shares,
            'inputs':{task:0 for task in TASKS}, 'skipped':{}, 'skip_examples':[]})


def collect_documents(stream, input_budget):
    # 一批尚未训练时，任何准备异常都不能留下部分推进的消费游标。
    boundary = stream.state_dict()
    try:
        return _collect_documents(stream, input_budget)
    except Exception:
        stream.rewind(boundary)
        raise


def _collect_documents(stream, input_budget):
    documents, inputs, body, control = [], Counter(), Counter(), 0
    missing = set(stream.active_tasks)
    while sum(inputs.values()) < input_budget or missing:
        # 显式固定任务顺序，避免Python集合排序在跨机器恢复时改变采样顺序。
        document = stream.next_document(tuple(task for task in TASKS if task in missing))
        task = document.metadata["task"]
        missing.discard(task)
        documents.append(document)
        inputs[task] += document.stroke_count-1
        for element in document.to_sequence()[1:]:
            if element.targetable:
                if element.kind in ("bos", "eop", "eod"):
                    control += 1
                else:
                    body[task] += 1
    documents.sort(key=lambda doc: doc.stroke_count)
    return documents, inputs, body, control


def iter_microbatches(documents, recipe, *, max_strokes=1024, pin=False):
    signature = (recipe['microbatch_documents'], recipe['padded_stroke_budget'],
                 recipe['max_points'], max_strokes, pin,
                 tuple(recipe.get('encoder_length_buckets', ())))
    if getattr(documents, 'batch_signature', None) == signature:
        yield from documents.cpu_batches
        return
    batches = []
    chunk = []
    for document in documents:
        length = document.stroke_count  # documents已经按长度排序
        if chunk and (len(chunk) >= recipe["microbatch_documents"] or
                      length*(len(chunk)+1) > recipe["padded_stroke_budget"]):
            batch = _batch(chunk, recipe, max_strokes, pin)
            batches.append(batch)
            yield batch
            chunk = []
        chunk.append(document)
    if chunk:
        batch = _batch(chunk, recipe, max_strokes, pin)
        batches.append(batch)
        yield batch
    if hasattr(documents, '__dict__'):
        documents.batch_signature, documents.cpu_batches = signature, batches


def _batch(chunk, recipe, max_strokes, pin):
    batch = collate_documents(chunk, max_strokes=max_strokes, max_points=recipe["max_points"],
                              encoder_length_buckets=recipe.get('encoder_length_buckets', ()))
    batch["task_id"] = torch.tensor([TASKS.index(doc.metadata["task"]) for doc in chunk],dtype=torch.long)
    if pin:
        batch = {key:value.pin_memory() if torch.is_tensor(value) else value for key,value in batch.items()}
    return batch
