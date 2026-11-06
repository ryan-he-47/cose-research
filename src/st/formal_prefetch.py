"""2026-10-24：有界完整批预取；公开游标只包含已交给训练线程的文档。"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from .formal_stream import collect_documents, iter_microbatches


class PreparedDocuments(list):
    pass


class PreparedUpdates:
    def __init__(self, source, recipe, catalog=None, *, pin=True, prepare_microbatches=True):
        self.source, self.recipe, self.catalog, self.pin = source, recipe, catalog, pin
        self.prepare_microbatches = prepare_microbatches
        self.visible = deepcopy(source.state_dict())
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cose-prepare')
        self.future = None
        self.requested_budget = None

    def __getattr__(self, name):
        return getattr(self.source, name)

    @property
    def skipped(self):
        return Counter(self.visible['skipped'])

    def state_dict(self):
        return deepcopy(self.visible)

    def coverage(self, limits):
        indices = self.visible['indices']
        fractions = {task: min(1., indices[task]/limit) for task, limit in limits.items()}
        return dict(progress=min(fractions.values()), fractions=fractions,
                    completed_sources=dict(indices), source_limits=dict(limits),
                    skipped=dict(self.skipped))

    def _prepare(self, budget, recipe):
        from .text_auxiliary import annotate_documents, space_target_count
        documents, inputs, body, control = collect_documents(self.source, budget)
        documents = PreparedDocuments(documents)
        aux = annotate_documents(documents, self.catalog) if self.catalog is not None else None
        spaces = space_target_count(documents) if self.catalog is not None else None
        from .visual_grounding_auxiliary import annotate_learning
        if recipe.get('visual_grounding') or recipe.get('text_aux_normalization'):
            annotate_learning(documents,recipe,self.source.dataset.drawings.classes)
        # 第一次生成后由iter_microbatches缓存；OOM改变切分参数会自动失效。
        if self.prepare_microbatches:
            list(iter_microbatches(documents, recipe, max_strokes=self.source.max_strokes, pin=self.pin))
        return documents, inputs, body, control, aux, spaces, self.source.state_dict()

    def _submit(self, budget):
        self.requested_budget = budget
        self.future = self.executor.submit(self._prepare, budget, deepcopy(self.recipe))

    def take(self, budget):
        if self.future is not None and self.requested_budget != budget:
            self.rewind(self.visible)
        if self.future is None:
            self._submit(budget)
        values = self.future.result()
        self.future = None
        self.visible = deepcopy(values[-1])
        self._submit(self.recipe['input_budget'])
        return values[:-1]

    def rewind(self, state):
        # 先结束唯一生产者，避免后台继续推进已经回滚的源游标。
        if self.future is not None:
            self.future.cancel()
            try:
                self.future.result()
            except Exception:
                pass
            self.future = None
        self.source.rewind(state)
        self.visible = deepcopy(state)

    def close(self):
        if self.future is not None:
            self.future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.source.close()
