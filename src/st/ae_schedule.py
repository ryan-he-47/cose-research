"""2026-10-04：AE余下预算按训练墙钟时间做余弦收尾，支持完整恢复。"""
from __future__ import annotations

import math
from bisect import bisect_right


class WallTimeCosine:
    def __init__(self, start_lrs, end_lr, duration_seconds, elapsed_seconds=0., *, hold_seconds=0.):
        self.start_lrs = tuple(float(lr) for lr in start_lrs)
        self.end_lr = float(end_lr)
        self.duration_seconds = float(duration_seconds)
        self.elapsed_seconds = float(elapsed_seconds)
        # 2026-10-05：同一批准预算内先保持当前LR，再平滑退火；恢复保留两段进度。
        self.hold_seconds = float(hold_seconds)
        values = (*self.start_lrs, self.end_lr, self.duration_seconds, self.elapsed_seconds, self.hold_seconds)
        if not self.start_lrs or not all(math.isfinite(value) for value in values):
            raise ValueError('余弦参数必须有限且至少有一个优化器组')
        if self.end_lr <= 0 or any(lr < self.end_lr for lr in self.start_lrs):
            raise ValueError('收尾LR必须为正且不高于当前LR')
        if self.duration_seconds <= 0 or self.elapsed_seconds < 0:
            raise ValueError('余弦时长必须为正，已用时间不能为负')
        if not 0 <= self.hold_seconds < self.duration_seconds:
            raise ValueError('恒定段必须非负且短于本阶段总时长')

    def progress(self, run_elapsed):
        return min(1., max(0., (self.elapsed_seconds + run_elapsed - self.hold_seconds) /
                              (self.duration_seconds - self.hold_seconds)))

    def lrs(self, run_elapsed):
        weight = .5 * (1. + math.cos(math.pi * self.progress(run_elapsed)))
        return [self.end_lr + (lr - self.end_lr) * weight for lr in self.start_lrs]

    def apply(self, optimizer, run_elapsed):
        if len(optimizer.param_groups) != len(self.start_lrs):
            raise ValueError('余弦恢复时优化器组数改变')
        for group, lr in zip(optimizer.param_groups, self.lrs(run_elapsed)):
            group['lr'] = lr

    def state_dict(self, run_elapsed):
        return {'kind': 'wall_time_cosine', 'start_lrs': list(self.start_lrs),
                'end_lr': self.end_lr, 'duration_seconds': self.duration_seconds,
                'hold_seconds': self.hold_seconds,
                'elapsed_seconds': min(self.duration_seconds, max(0., self.elapsed_seconds + run_elapsed))}

    @classmethod
    def from_state_dict(cls, state):
        if state['kind'] != 'wall_time_cosine':
            raise ValueError('未知AE退火类型')
        return cls(state['start_lrs'], state['end_lr'], state['duration_seconds'],
                   state['elapsed_seconds'], hold_seconds=state.get('hold_seconds', 0.))


class StepLRReplay:
    """按成功更新数重放LR；2026-10-06支持从step0升温后连续跑完整轨迹。"""

    def __init__(self, spec):
        self.cosine = spec.get('kind') == 'successful_step_cosine'
        if spec.get('kind') not in ('successful_step_lr_replay', 'successful_step_cosine'):
            raise ValueError('未知按步LR轨迹')
        raw_points = ([[spec['start_step'], spec['start_lr']], [spec['end_step'], spec['end_lr']]]
                      if self.cosine else spec['points'])
        points = tuple((int(step), float(lr)) for step, lr in raw_points)
        if len(points) < 2 or any(step < 0 or not math.isfinite(lr) or lr <= 0 for step, lr in points):
            raise ValueError('LR轨迹需至少两个有效步数/正LR记录')
        if any(b[0] <= a[0] or b[1] > a[1] for a, b in zip(points, points[1:])):
            raise ValueError('成功步数需递增，收尾LR不能递增')
        if points[0] != (spec['start_step'], spec['start_lr']) or points[-1] != (spec['end_step'], spec['end_lr']):
            raise ValueError('LR轨迹的起止记录不符')
        self.spec = dict(spec)
        self.points = points
        self.steps = tuple(step for step, _ in points)
        self.start_step, self.end_step = self.steps[0], self.steps[-1]
        self.warmup_steps = int(spec.get('warmup_steps', 0))
        if self.cosine and self.warmup_steps:
            raise ValueError('接续余弦不重走升温')
        if self.warmup_steps < 0 or (self.warmup_steps and (
                self.start_step != 0 or self.warmup_steps > points[1][0] or points[0][1] != points[1][1])):
            raise ValueError('完整轨迹升温需从step0开始，且位于初始恒定LR段内')

    def lr(self, step):
        if 0 < step <= self.warmup_steps:
            # 与原独立AE的逐成功步warmup使用相同运算顺序；step0保留启动LR。
            return self.points[0][1] * step / self.warmup_steps
        if step <= self.start_step:
            return self.points[0][1]
        if step >= self.end_step:
            return self.points[-1][1]
        if self.cosine:
            fraction = (step - self.start_step) / (self.end_step - self.start_step)
            return self.points[-1][1] + (self.points[0][1] - self.points[-1][1]) * .5 * (1. + math.cos(math.pi * fraction))
        right = bisect_right(self.steps, step)
        before, after = self.points[right - 1], self.points[right]
        fraction = (step - before[0]) / (after[0] - before[0])
        return before[1] + fraction * (after[1] - before[1])

    def apply(self, optimizer, step):
        if len(optimizer.param_groups) != 1:
            raise ValueError('按步轨迹沿用独立AE的单Adam组')
        optimizer.param_groups[0]['lr'] = self.lr(step)

    def state_dict(self):
        if self.cosine:
            return dict(self.spec)
        return {**self.spec, 'points': [list(point) for point in self.points]}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state)
