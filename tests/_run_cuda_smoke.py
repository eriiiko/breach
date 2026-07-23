"""Throwaway headless check: run the LIVE engine with ALL GPU backends ON.

This is the real verification for the GPU launch path — the windowed game can't
run headless here, so we exercise the PHYSICS end-to-end instead. It must be run
in the anaconda-3.11 interpreter that owns the cp311 CUDA .pyd; the harness below
re-execs itself there via tests/cuda_harness.run_cuda_script (which prepends the
DLL-dir + sys.path bootstrap, so `import breach_physics` resolves to the CUDA
build in cpp/build_cuda).

EOS P6.0: the wave/atmosphere backends are RETIRED (cuda_wave.cu /
cuda_atmosphere.cu deleted — their CPU solvers were replaced by the EOS solve
in P3), and the remaining kernels are STALE until their P6 ports re-prove them
(tests/cuda_harness.py EOS_P6_PENDING_KERNELS pins the gates meanwhile).

What it asserts:
  (a) no crash building + stepping a real Simulation 30 ticks with the live GPU
      backends on (temperature/water/smoke/fire);
  (b) the backends actually report CUDA (get_*_backend() True) + cuda_available();
  (c) fields EVOLVE (smoke/fire/temperature/atmosphere/water not frozen) once we
      deposit a fire + smoke + water + overpressure source;
  (d) sanity: a fresh CPU-backend run of the same seeded scenario agrees field-for
      -field with the GPU run (the solvers are bit-identical by design).

Run:  C:/Users/steen/anaconda3/python.exe tests/_run_cuda_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuda_harness import run_cuda_script, cuda_available  # noqa: E402


# This body runs INSIDE the anaconda-3.11 subprocess with the CUDA build on the
# path (cuda_harness._bootstrap prepends DLL dir + sys.path: build_cuda, src,
# root, tests). It prints "RESULT: ..." lines the parent asserts on.
_BODY = r"""
import numpy as np
import breach_physics as bp

print("HAS_CUDA", bool(getattr(bp, "HAS_CUDA", False)))
assert getattr(bp, "HAS_CUDA", False), "imported the CPU build, not build_cuda"
assert bp.cuda_available(), "no usable CUDA device"
print("DEVICE", bp.cuda_device_info())

from config import CFG
from level_loader import load as load_level
from simulation import Simulation
from simulation.unit import Unit
from simulation.field_edit import FieldEdit, Region, EditMode, Falloff

LEVEL = getattr(CFG.display, "level", "playground")
# The 4 field solvers + the raycaster (the live fire->heat cast, CUDA-S2 live).
# With the raycaster ON, cast_fire_heat deposits `heat` on the GPU.
# EOS P6.0: wave/atmos backends retired (kernels deleted with their solvers).
# EOS P6.5: the four EOS kernel-surface flags (bulk_flux, sl_advection,
# mg_solve, kick_compression) are now LIVE-DISPATCHED — with all four on,
# run_substeps routes the whole eos.step tick to the chained GPU orchestration
# (cuda_eos_step.cu). All six EOS-era setters are in the all-on set.
SETTERS = ["set_temperature_backend", "set_water_backend", "set_smoke_backend",
           "set_fire_backend", "set_raycaster_backend", "set_bulk_flux_backend",
           "set_sl_advection_backend", "set_mg_solve_backend",
           "set_kick_compression_backend"]
GETTERS = ["get_temperature_backend", "get_water_backend", "get_smoke_backend",
           "get_fire_backend", "get_raycaster_backend", "get_bulk_flux_backend",
           "get_sl_advection_backend", "get_mg_solve_backend",
           "get_kick_compression_backend"]


def find_open_cell(gmap):
    # A flammable, interior cell (not solid, not vacuum) so the fluid solvers —
    # fire/smoke/atmosphere/wave/water — have somewhere to act. Falls back to
    # any open cell.
    open_mask = (~gmap.solid) & (~gmap.is_vacuum)
    pref = open_mask & gmap.flammable
    ys, xs = np.nonzero(pref if pref.any() else open_mask)
    if len(ys) == 0:
        raise RuntimeError("no open cell in level")
    mid = len(ys) // 2  # deterministic central-ish pick
    return int(ys[mid]), int(xs[mid])


def find_solid_cell_near(gmap, r, c):
    # The temperature solver converts heat -> temperature on SOLID tiles only
    # (walls hold heat; air does not — temperature_solver.cpp pass 1). Find the
    # nearest solid cell to (r, c) so a heat deposit provably drives temperature.
    ys, xs = np.nonzero(gmap.solid)
    if len(ys) == 0:
        return r, c
    d = (ys - r) ** 2 + (xs - c) ** 2
    k = int(np.argmin(d))
    return int(ys[k]), int(xs[k])


def build_sim():
    level = load_level(LEVEL)
    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=False)
    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team, footprint=s.footprint))
    return sim


def deposit_sources(sim, r, c):
    # Fire + smoke + overpressure + water + a wave pulse, so EVERY solver has
    # something to move. Authored through the canonical FieldEdit write path.
    sim.edit(FieldEdit("fire",        Region.DISC, (r, c, 2), 1.0,  EditMode.MAX))
    sim.edit(FieldEdit("smoke",       Region.DISC, (r, c, 3), 4.0,  EditMode.ADD,
                       channel=None))
    sim.edit(FieldEdit("atmosphere",  Region.DISC, (r, c, 3), 0.8,  EditMode.ADD))
    sim.edit(FieldEdit("wave_source", Region.DISC, (r, c, 2), 2.0,  EditMode.ADD))
    sim.edit(FieldEdit("water_depth", Region.DISC, (r, c, 3), 0.30, EditMode.ADD))
    # Deposit raw heat onto a SOLID cell, so the temperature solver provably
    # evolves independent of whether the fire sustains (heat -> temperature is
    # the GPU temperature kernel's job, and only happens on solid tiles).
    sr, sc = find_solid_cell_near(sim.gmap, r, c)
    sim.edit(FieldEdit("heat",        Region.TILE, (sr, sc),   400.0, EditMode.ADD))


def _snap(gm):
    return {
        "smoke": gm.smoke.copy(), "fire": gm.fire.copy(),
        "temperature": gm.temperature.copy(),
        "atmosphere": gm.atmosphere.copy(),
        "water_depth": gm.water_depth.copy(),
    }


def run(use_cuda, n_ticks=30):
    for name in SETTERS:
        getattr(bp, name)(bool(use_cuda))
    backends_report = {g: bool(getattr(bp, g)()) for g in GETTERS}
    sim = build_sim()
    sim.set_paused(False)
    r, c = find_open_cell(sim.gmap)
    gm = sim.gmap
    after1 = None
    # Track the PEAK |field| across the run — `heat`/`temperature` spike then
    # decay within a tick, so a peak makes "this kernel moved the field"
    # unambiguous (vs an end-state that has cooled back toward 0).
    peak = {k: 0.0 for k in _snap(gm)}
    for t in range(n_ticks):
        deposit_sources(sim, r, c)   # re-feed each tick so every solver stays driven
        sim.step()
        if t == 0:
            after1 = _snap(gm)
        for k in peak:
            peak[k] = max(peak[k], float(np.abs(getattr(gm, k).astype(np.float64)).max()))
    return backends_report, after1, _snap(gm), peak


# ---- GPU run -------------------------------------------------------------
gpu_backends, gpu_after1, gpu_final, gpu_peak = run(use_cuda=True)
print("BACKENDS_CUDA", all(gpu_backends.values()), gpu_backends)
print("PEAK_MAG", {k: round(v, 3) for k, v in gpu_peak.items()})

# (c) fields EVOLVE: final != after-tick-1 for the dynamic fields.
evolved = {}
for k in gpu_final:
    evolved[k] = bool(np.any(gpu_final[k] != gpu_after1[k]))
print("EVOLVED", evolved)
# At minimum smoke + temperature + atmosphere must move (fire/water can settle).
moving = sum(1 for v in evolved.values() if v)
print("N_FIELDS_MOVING", moving)

# Each field's GPU kernel must have moved it OFF zero at some point in the run
# (catch a silently-frozen / never-dispatched solver). Uses the per-run peak.
nonzero = {k: bool(gpu_peak[k] > 0.0) for k in gpu_peak}
print("NONZERO_PEAK", nonzero)
n_nonzero = sum(1 for v in nonzero.values() if v)

# ---- CPU run (same seeded scenario) for gross agreement ------------------
cpu_backends, _cpu_after1, cpu_final, _cpu_peak = run(use_cuda=False)
print("CPU_BACKENDS_OFF", (not any(cpu_backends.values())), cpu_backends)
agree = {}
for k in gpu_final:
    a, b = gpu_final[k], cpu_final[k]
    if a.shape != b.shape:
        agree[k] = "SHAPE-MISMATCH %s vs %s" % (a.shape, b.shape)
        continue
    if np.array_equal(a, b):
        agree[k] = "bit-identical"
    else:
        af = a.astype(np.float64); bf = b.astype(np.float64)
        denom = max(1.0, float(np.abs(bf).max()))
        rel = float(np.abs(af - bf).max()) / denom
        agree[k] = "rel-maxdiff %.3e" % rel
print("GPU_VS_CPU", agree)

# PASS gate: all GPU backends report CUDA, the CPU reference reports OFF, the
# fields actually evolved (>=3 moving tick1->final), every field's kernel drove
# it off zero at some point (all 5 nonzero), and GPU == CPU on every field.
all_agree = all(v in ("bit-identical",) for v in agree.values())
ok = (all(gpu_backends.values())
      and (not any(cpu_backends.values()))
      and moving >= 3
      and n_nonzero == len(gpu_peak)
      and all_agree)
print("RESULT:", "PASS" if ok else "FAIL")
"""


def main() -> int:
    if not cuda_available():
        print("SKIP: CUDA build / runtime not present. "
              "Build it with cpp/build_cuda.bat.")
        return 2
    proc = run_cuda_script(_BODY, timeout=300.0)
    sys.stdout.write(proc.stdout)
    if proc.stderr.strip():
        sys.stderr.write("\n--- subprocess stderr ---\n")
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"\nFAIL: subprocess exited {proc.returncode}")
        return 1
    if "RESULT: PASS" not in proc.stdout:
        print("\nFAIL: did not see RESULT: PASS")
        return 1
    print("\nOK: all GPU backends ran the live engine 30 ticks; fields evolved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
