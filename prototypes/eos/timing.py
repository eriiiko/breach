"""
Timing instrumentation for the tick loop -- FIRST-CLASS per PLAN.md: the
question this whole spike must answer ("is rung B cheap enough to run in
realtime?") is a cost-table question as much as a by-eye one. Reusable by
every scheme: `TickTimer` only needs a wall-clock bracket and the solver's
reported substep count, both solver-agnostic.
"""

import time

import numpy as np


class TickTimer:
    """Accumulates per-tick wall-clock ms and solver substep counts."""

    def __init__(self):
        self._ms: list[float] = []
        self._substeps: list[int] = []
        self._t0: float | None = None

    def tick_start(self) -> None:
        self._t0 = time.perf_counter()

    def tick_end(self, substeps: int) -> None:
        assert self._t0 is not None, "tick_end called without a matching tick_start"
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        self._ms.append(elapsed_ms)
        self._substeps.append(substeps)
        self._t0 = None

    def summary(self) -> dict:
        ms = np.asarray(self._ms, dtype=np.float64)
        substeps = np.asarray(self._substeps, dtype=np.float64)
        return {
            "n_ticks": len(ms),
            "ms_min": float(ms.min()),
            "ms_mean": float(ms.mean()),
            "ms_max": float(ms.max()),
            "substeps_mean": float(substeps.mean()),
        }

    def print_table(self, *, grid_w: int, grid_h: int, scenario: str, scheme: str) -> None:
        s = self.summary()
        print("-" * 64)
        print(f"timing  scenario={scenario}  scheme={scheme}  grid={grid_w}x{grid_h}")
        print(f"  ticks          : {s['n_ticks']}")
        print(f"  ms/tick   min  : {s['ms_min']:.3f}")
        print(f"  ms/tick   mean : {s['ms_mean']:.3f}")
        print(f"  ms/tick   max  : {s['ms_max']:.3f}")
        print(f"  substeps/tick  : {s['substeps_mean']:.2f}")
        print("-" * 64)
