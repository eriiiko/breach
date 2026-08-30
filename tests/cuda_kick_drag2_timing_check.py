"""Drag-law v2 (docs/drag_law_v2_design_2026-08-23.md §7/§8 gate 5) — the
k_drag2 TIMING gate: stage Q is a NEW always-on per-open-cell cost (isqrt +
2 int64 divisions) once armed. This measures its overhead ARMED and WINDY
(§7 — a calm field would measure only the fast-path compare, never the real
cost class) on BOTH backends, against a < 3% tick-budget bound (§7's
"Budget: < 3% tick time on BOTH backends").

CPU leg: a real engine tick (Simulation.step()) on a windy, open scenario,
timed with k_drag2=0 (baseline) vs k_drag2=1.0 (armed) — the delta as a
fraction of the SHIPPED tick budget (1/ticks_per_second, read live per R5,
never a printed number) must stay under the bound.

CUDA leg: EOS GPU dispatch is not yet wired into the live tick (P6.5
pending — cuda_kick_compression.h's own docstring: "no dispatch site
consumes this yet"), so there is no GPU Simulation.step() to time
end-to-end. The isolated `cuda_eos_kick_compression` call is timed directly
(armed vs dormant) on the same windy field instead — exactly the
incremental cost class §7 names, ahead of dispatch — against the same
absolute budget.

Run via cuda_harness.run_cuda_script (tests/test_cuda_kick_drag2_timing.py
is the pytest wrapper). The CPU leg runs fine in that same subprocess — it
is the same interpreter, with the CUDA build's `breach_physics` also
carrying every plain CPU solver (the cuda_kick_check.py PART 3 precedent).
Prints DRAG2_TIMING_RESULT: PASS/FAIL and exits 0/1.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import breach_physics as bp

FP_ONE = 65536
BUDGET_FRACTION = 0.03


def _quantize(x):
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _tick_budget_ms():
    """The live tick budget (R5: never a printed number) — config's
    [clock] ticks_per_second, shipped 24."""
    from config import CFG
    return 1000.0 / float(CFG.clock.ticks_per_second)


# ---------------------------------------------------------------------------
# CPU leg — a real engine tick, windy + armed vs windy + dormant.
# ---------------------------------------------------------------------------
def _make_windy_scenario(h=160, w=160, seed=20260823):
    from level_loader import LevelData
    from simulation import Simulation

    tm = np.full((h, w), 4, dtype=np.int32)   # open interior air
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    level = LevelData(name="drag2_timing_bench", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False)
    sim.set_paused(False)
    g = sim.gmap
    # A strong, uniform, diagonal wind field over the whole open interior —
    # ARMED and WINDY (design §7): stage Q's divide must actually execute
    # every open-cell tick, not just the fast-path compare.
    open_mask = ~(g.solid | g.is_vacuum)
    g.wind_x[open_mask] = int(_quantize(60.0))
    g.wind_y[open_mask] = int(_quantize(-45.0))
    return sim


def _time_ticks(sim, n_ticks, warmup=5):
    for _ in range(warmup):
        sim.step()
    times = np.empty(n_ticks, dtype=np.float64)
    for i in range(n_ticks):
        t0 = time.perf_counter()
        sim.step()
        times[i] = (time.perf_counter() - t0) * 1000.0
    return times


def bench_cpu_tick(n_ticks=60):
    sim_off = _make_windy_scenario()
    sim_off.physics_runner.eos.k_drag2 = 0.0
    ms_off = _time_ticks(sim_off, n_ticks)

    sim_on = _make_windy_scenario()
    sim_on.physics_runner.eos.k_drag2 = 1.0
    ms_on = _time_ticks(sim_on, n_ticks)

    p50_off, p50_on = float(np.median(ms_off)), float(np.median(ms_on))
    delta = p50_on - p50_off
    budget = _tick_budget_ms()
    return dict(p50_off=p50_off, p50_on=p50_on, delta_ms=delta,
                frac=delta / budget, budget_ms=budget)


# ---------------------------------------------------------------------------
# CUDA leg — the isolated cuda_eos_kick_compression call, armed vs dormant.
# ---------------------------------------------------------------------------
# arc #54 (P-G2): step 4c (compression work) is DELETED — `temperature`,
# `t_min`/`t_work_clamp`/`t_max_phys` (moved to the §2.6 recovery, outside
# this isolated tail) and `k_drag_heat_frac`/`c_v` (the retired deposit
# formula, D5) are gone from `cuda_eos_kick_compression`'s signature; the
# kick now runs in place on `gas_energy` (arc #54 §2.2/§2.3) instead, and
# `t_amb_k` is LOAD-BEARING (folds the derived k_ke constant).
CONSTS = dict(
    c_max=300.0, dx=1.0 / 3.0, adiabatic_index=1.4, absorb_strength=8.0,
    n_floor_solver=1e-3, u_max=1000.0,
    k_drag=0.0, t_amb_k=290.0,
)


def _make_windy_field(h=160, w=160):
    rng = np.random.default_rng(1)
    wind_x = np.full((h, w), int(_quantize(60.0)), dtype=np.int32)
    wind_y = np.full((h, w), int(_quantize(-45.0)), dtype=np.int32)
    # arc #54: the field the kick's KE brackets debit/credit now — a plausible
    # ambient-ish magnitude (N~1 raw-count worth of air at T_AMB_K), timing
    # only, no correctness claim.
    gas_energy = np.full((h, w), FP_ONE * FP_ONE * 290, dtype=np.int64)
    p_new = _quantize(rng.random((h, w)) * 0.4 - 0.2).astype(np.int32)
    gas = np.zeros((3, h, w), dtype=np.int32)
    gas[0] = _quantize(0.21)
    gas[1] = _quantize(0.79)
    gas_conservative = np.array([True, True, False])
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    absorb = np.zeros((h, w), dtype=np.float32)
    cap2 = np.full((h, w), int(_quantize(2300.0)) ** 2, dtype=np.int64)
    return dict(wind_x=wind_x, wind_y=wind_y, gas_energy=gas_energy,
                p_new=p_new, gas=gas, gas_conservative=gas_conservative,
                solid=solid, is_vacuum=is_vacuum, absorb=absorb, cap2=cap2)


def _time_cuda_call(fields, k_drag2, dt, n_iter):
    args_tail = (fields["p_new"], fields["gas"], fields["gas_conservative"],
                 fields["solid"], fields["is_vacuum"], fields["absorb"])

    def call():
        wx, wy = fields["wind_x"].copy(), fields["wind_y"].copy()
        ge = fields["gas_energy"].copy()
        bp.cuda_eos_kick_compression(wx, wy, ge, *args_tail, dt, fields["cap2"],
                                     k_drag2=k_drag2, **CONSTS)

    for _ in range(5):
        call()   # warmup
    t0 = time.perf_counter()
    for _ in range(n_iter):
        call()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter * 1000.0


def bench_cuda_isolated(n_iter=200):
    fields = _make_windy_field()
    dt = _tick_budget_ms() / 1000.0
    ms_off = _time_cuda_call(fields, 0.0, dt, n_iter)
    ms_on = _time_cuda_call(fields, 1.0, dt, n_iter)
    delta = ms_on - ms_off
    budget = _tick_budget_ms()
    return dict(p50_off=ms_off, p50_on=ms_on, delta_ms=delta,
                frac=delta / budget, budget_ms=budget)


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("DRAG2_TIMING_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())

    ok = True
    print("\n--- CPU leg: real engine tick, windy scenario, armed vs dormant ---")
    cpu = bench_cpu_tick()
    print(f"  dormant p50={cpu['p50_off']:.4f}ms  armed p50={cpu['p50_on']:.4f}ms  "
          f"delta={cpu['delta_ms']:.4f}ms  ({cpu['frac'] * 100:.2f}% of "
          f"{cpu['budget_ms']:.3f}ms budget)")
    if cpu["frac"] >= BUDGET_FRACTION:
        ok = False
        print(f"  CPU leg FAILS the < {BUDGET_FRACTION * 100:.0f}% budget")

    print("\n--- CUDA leg: isolated cuda_eos_kick_compression, armed vs dormant ---")
    cuda = bench_cuda_isolated()
    print(f"  dormant={cuda['p50_off']:.4f}ms  armed={cuda['p50_on']:.4f}ms  "
          f"delta={cuda['delta_ms']:.4f}ms  ({cuda['frac'] * 100:.2f}% of "
          f"{cuda['budget_ms']:.3f}ms budget)")
    if cuda["frac"] >= BUDGET_FRACTION:
        ok = False
        print(f"  CUDA leg FAILS the < {BUDGET_FRACTION * 100:.0f}% budget")

    print("\nDRAG2_TIMING_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
