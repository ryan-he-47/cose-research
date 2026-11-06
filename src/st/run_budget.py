"""2026-10-03：按时间/调度信号在完整更新边界停下，保存可恢复状态。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import signal
import time


class RunBudget:
    def __init__(self, max_seconds: float = 0):
        if not math.isfinite(max_seconds) or max_seconds < 0:
            raise ValueError("max_seconds 必须为有限非负数")
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.signal_name = None
        for name in ("SIGINT", "SIGTERM", "SIGUSR1"):
            value = getattr(signal, name, None)
            if value is not None:
                signal.signal(value, self._on_signal)

    def _on_signal(self, signum, frame):
        self.signal_name = signal.Signals(signum).name

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    @property
    def reason(self):
        if self.signal_name:
            return self.signal_name
        if self.max_seconds and self.elapsed >= self.max_seconds:
            return "max_seconds"
        return None

    def finish(self, output: Path, step: int, *, completed_reason="steps"):
        record = {"step": step, "reason": self.reason or completed_reason,
                  "elapsed_seconds": round(self.elapsed, 3),
                  "max_seconds": self.max_seconds}
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_stop.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"run_stop": record}, ensure_ascii=False), flush=True)
