"""2026-10-03：审批配方与按成功输入笔画计时的WSD。"""
from __future__ import annotations


TASKS = ("T1", "O1", "O2", "S1", "S2", "S3", "S4", "S5")
INPUT_SHARES = dict(zip(TASKS, (.4, .175, .175, .075, .0875, .0375, .03, .02)))
LOSS_WEIGHTS = dict(zip(TASKS, (.35, .25, .15, .07, .10, .03, .03, .02)))
DEFAULT_RECIPE = {
    "version": "2026-10-03-approved-v1", "total_inputs": 200000000,
    "input_budget": 65536, "microbatch_documents": 128, "padded_stroke_budget": 32768,
    "encode_chunk_size": 1024, "activation_checkpoint": False, "workers": 4,
    "max_points": 76, "peak_lr": .0001, "min_lr": .00001,
    "warmup_inputs": 8000000, "decay_inputs": 8000000,
    "clip_norm": 1.0, "control_weight": .05, "input_shares": INPUT_SHARES,
    "loss_weights": LOSS_WEIGHTS, "seed": 42, "save_seconds": 3600,
}


class CoverageWSD:
    """全数据池目标：升温按成功输入，稳定/末4%退火按最慢任务的成功覆盖进度。"""
    def __init__(self,optimizer,recipe,start_lr):
        self.optimizer=optimizer
        self.spec={key:recipe[key] for key in ('peak_lr','min_lr','warmup_inputs')}
        self.spec.update(start_lr=float(start_lr),training_goal=recipe['training_goal'])
        self.inputs=0;self.progress=0.;self._apply()

    def lr_at(self,inputs,progress=None):
        s=self.spec;progress=self.progress if progress is None else progress
        if inputs<s['warmup_inputs']:
            return s['start_lr']+(s['peak_lr']-s['start_lr'])*inputs/max(1,s['warmup_inputs'])
        fraction=s['training_goal']['decay_fraction']
        decay=min(1.,max(0.,(progress-(1-fraction))/fraction))
        return s['peak_lr']+(s['min_lr']-s['peak_lr'])*decay

    def _apply(self):
        for group in self.optimizer.param_groups:group['lr']=self.lr_at(self.inputs)

    def prepare_update(self,inputs,progress=None):
        for group in self.optimizer.param_groups:group['lr']=self.lr_at(self.inputs+inputs,progress)

    def step(self,inputs,progress=None):
        self.inputs+=int(inputs)
        if progress is not None:self.progress=float(progress)
        self._apply()

    def state_dict(self):
        return dict(kind='coverage_wsd_v1',spec=dict(self.spec),inputs=self.inputs,progress=self.progress)

    def load_state_dict(self,state):
        if state['kind']!='coverage_wsd_v1' or state['spec']!=self.spec:raise ValueError('覆盖WSD配方不同')
        self.inputs=int(state['inputs']);self.progress=float(state['progress']);self._apply()


class InputWSD:
    def __init__(self, optimizer, recipe, start_lr):
        self.optimizer = optimizer
        self.spec = {key: recipe[key] for key in
                     ("total_inputs", "peak_lr", "min_lr", "warmup_inputs", "decay_inputs")}
        self.spec["start_lr"] = float(start_lr)
        if "warmup_start_inputs" in recipe:
            self.spec["warmup_start_inputs"] = int(recipe["warmup_start_inputs"])
            if self.spec["warmup_start_inputs"] < 0:
                raise ValueError("升温起点必须为非负成功输入数")
        if self.spec.get("warmup_start_inputs", 0) + self.spec["warmup_inputs"] + self.spec["decay_inputs"] >= self.spec["total_inputs"]:
            raise ValueError("WSD稳定段不能为空")
        self.inputs = 0
        self._apply()

    @classmethod
    def from_coverage(cls, previous, recipe, total_inputs):
        """显式缩短同一阶段终点，保留已成功输入与原升温时钟。"""
        if not isinstance(previous, CoverageWSD) or total_inputs <= previous.inputs:
            raise ValueError('需要未耗尽的新输入终点和覆盖调度器')
        old = previous.state_dict()
        fraction = old['spec']['training_goal']['decay_fraction']
        recipe.update(total_inputs=int(total_inputs), training_goal=None,
                      decay_inputs=round(total_inputs * fraction),
                      coverage_source_limits=old['spec']['training_goal']['limits'],
                      retry_failed_documents=True)
        result = cls(previous.optimizer, recipe, old['spec']['start_lr'])
        result.inputs = previous.inputs
        result._apply()
        return result, dict(previous=old, current=result.state_dict(),
                           includes_previous_successful_inputs=True)

    def lr_at(self, inputs):
        s = self.spec
        warmup_start = s.get("warmup_start_inputs", 0)
        if inputs < warmup_start+s["warmup_inputs"]:
            ratio = max(0, inputs-warmup_start) / max(1, s["warmup_inputs"])
            return s["start_lr"] + (s["peak_lr"]-s["start_lr"])*ratio
        decay_start = s["total_inputs"]-s["decay_inputs"]
        if inputs <= decay_start:
            return s["peak_lr"]
        ratio = min(1., (inputs-decay_start)/s["decay_inputs"])
        return s["peak_lr"]+(s["min_lr"]-s["peak_lr"])*ratio

    def _apply(self):
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr_at(self.inputs)

    def step(self, successful_inputs):
        self.inputs += int(successful_inputs)
        self._apply()

    def prepare_update(self, inputs):
        # 2026-10-07：本次更新使用包含本批成功输入的LR；首批不以零LR空走。
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr_at(self.inputs + int(inputs))

    def state_dict(self):
        return {"kind": "input_wsd_v1", "spec": dict(self.spec), "inputs": self.inputs}

    def load_state_dict(self, state):
        if state["kind"] != "input_wsd_v1" or state["spec"] != self.spec:
            raise ValueError("恢复时WSD配方不同；阶段切换请显式使用新阶段入口")
        self.inputs = int(state["inputs"])
        self._apply()

    def begin_decay(self, min_lr):
        """2026-10-12：显式批准后，从当前LR开始走完剩余成功输入的线性退火。"""
        import math
        current_lr = self.lr_at(self.inputs)
        remaining = self.spec['total_inputs'] - self.inputs
        if self.inputs <= self.spec.get('warmup_start_inputs', 0)+self.spec['warmup_inputs'] or remaining <= 0:
            raise ValueError('提前退火须在升温结束后、输入预算完成前')
        if not math.isfinite(min_lr) or not 0 < min_lr <= current_lr:
            raise ValueError('退火终值须为正且不高于当前LR')
        before = dict(self.spec)
        self.spec.update(peak_lr=current_lr,min_lr=float(min_lr),decay_inputs=remaining)
        self._apply()
        return {'kind':'approved_early_linear_decay','start_inputs':self.inputs,
                'start_lr':current_lr,'end_lr':float(min_lr),
                'end_inputs':self.spec['total_inputs'],'decay_inputs':remaining,'previous_spec':before}
