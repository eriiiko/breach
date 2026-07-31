"""P-R1 gate (a), CUDA half — byte-identity of the C++ fire-plane source build
(runs inside the GPU subprocess, tests/cuda_harness.py).

docs/radiation_raycaster_extinction_ruling_2026-07-31.md A4.1-A4.2: the
per-tile source build that used to run in PhysicsRunner.cast_fire_heat's
Python loop now runs inside Raycaster.cast_from_fire_plane (CPU) /
cuda_raycaster_cast_from_fire_plane (CUDA), ONE call per tick. Mechanical
relocation only — no march/law change — so ``heat`` must be byte-identical
(tolerance ZERO) to the pre-patch per-tile loop on BOTH backends.

``_old_cast_cuda`` below is a frozen, verbatim transcription of the pre-P-R1
``cast_fire_heat`` CUDA branch (collect LightSource per tile, one
``cuda_raycaster_cast_batch`` call) — the S8c-proven batched path this patch
does not touch. It is the oracle for ``cuda_raycaster_cast_from_fire_plane``.

Three checks, on TWO scenarios (600-fire synthetic firestorm; the real
"playground" level over several ticks of evolving fire — mirrors
tests/test_pr1_fire_plane_cast.py's CPU half and
tests/cuda_s2_check.py / cuda_s2b_raycaster_live_check.py's structure):

  1. NEW CUDA (cuda_raycaster_cast_from_fire_plane) == OLD CUDA
     (cuda_raycaster_cast_batch over a Python-built source list) — the gate's
     literal requirement.
  2. NEW CUDA == NEW CPU (cast_from_fire_plane) — belt-and-suspenders cross-
     backend check (transitively implied by (1) + the pre-existing S2/S8c
     CPU==CUDA gates, but cheap to assert directly).
  3. Scenario sanity: heat is non-trivially non-zero (not a vacuous pass).

Prints ``PR1_FIRE_PLANE_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import math
import sys

import numpy as np

# Import the CUDA build FIRST (cuda_harness bootstrap put cpp/build_cuda on
# the path) so `breach_physics` is the GPU build.
import breach_physics as bp

FP_ONE_F = 65536.0   # fire_fixed.FP_ONE_F — Q16.16 scale, shared across fields


# ---------------------------------------------------------------------------
# The oracles: frozen, verbatim transcriptions of the PRE-P-R1 per-tile loop
# (physics_runner.py's old cast_fire_heat). Call only functions this patch
# does not touch (bp.LightSource, cast_source_directional,
# cuda_raycaster_cast_batch) — do NOT "modernize" these to match the patch.
# ---------------------------------------------------------------------------
def _build_old_sources(fire, k_fire_heat, fire_ray_count, range_base,
                        range_per_i, intensity_base, intensity_per_i, color):
    two_pi = 2.0 * math.pi
    ray_count = fire_ray_count
    ys, xs = np.nonzero(fire > 0)
    sources = []
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        intensity_fire = float(fire[yy, xx]) / FP_ONE_F
        src = bp.LightSource()
        src.x = float(xx) + 0.5
        src.y = float(yy) + 0.5
        src.max_range = range_base + range_per_i * intensity_fire
        src.ray_count = ray_count
        src.angle_spread = two_pi
        src.angle_center = ((xx * 7 + yy * 13) % ray_count) * (two_pi / ray_count)
        src.intensity = intensity_base + intensity_per_i * intensity_fire
        src.heat = k_fire_heat * intensity_fire
        src.jitter = 0.0
        src.color = color
        sources.append(src)
    return sources


def _old_cast_cpu(raycaster, fire, k_fire_heat, fire_ray_count, range_base,
                   range_per_i, intensity_base, intensity_per_i, color,
                   rgb, dx, dy, gas_f, gas_absorption, gas_scatter,
                   light_atten, heat, heat_atten):
    sources = _build_old_sources(fire, k_fire_heat, fire_ray_count, range_base,
                                  range_per_i, intensity_base, intensity_per_i,
                                  color)
    for src in sources:
        raycaster.cast_source_directional(
            src, rgb, dx, dy, gas_f, gas_absorption, gas_scatter, light_atten,
            heat=heat, smoke_glow=None, heat_atten=heat_atten)


def _old_cast_cuda(raycaster, fire, k_fire_heat, fire_ray_count, range_base,
                    range_per_i, intensity_base, intensity_per_i, color,
                    rgb, dx, dy, gas_f, gas_absorption, gas_scatter,
                    light_atten, heat, heat_atten):
    sources = _build_old_sources(fire, k_fire_heat, fire_ray_count, range_base,
                                  range_per_i, intensity_base, intensity_per_i,
                                  color)
    if not sources:
        return
    bp.cuda_raycaster_cast_batch(
        raycaster, sources, rgb, dx, dy, gas_f, gas_absorption, gas_scatter,
        light_atten, heat=heat, smoke_glow=None, heat_atten=heat_atten)


def _new_cast_cpu(raycaster, fire, k_fire_heat, fire_ray_count, range_base,
                   range_per_i, intensity_base, intensity_per_i, color,
                   rgb, dx, dy, gas_f, gas_absorption, gas_scatter,
                   light_atten, heat, heat_atten):
    raycaster.cast_from_fire_plane(
        fire, k_fire_heat, fire_ray_count, range_base, range_per_i,
        intensity_base, intensity_per_i, color,
        rgb, dx, dy, gas_f, gas_absorption, gas_scatter, light_atten,
        heat=heat, smoke_glow=None, heat_atten=heat_atten)


def _new_cast_cuda(raycaster, fire, k_fire_heat, fire_ray_count, range_base,
                    range_per_i, intensity_base, intensity_per_i, color,
                    rgb, dx, dy, gas_f, gas_absorption, gas_scatter,
                    light_atten, heat, heat_atten):
    bp.cuda_raycaster_cast_from_fire_plane(
        raycaster, fire, k_fire_heat, fire_ray_count, range_base, range_per_i,
        intensity_base, intensity_per_i, color,
        rgb, dx, dy, gas_f, gas_absorption, gas_scatter, light_atten,
        heat=heat, smoke_glow=None, heat_atten=heat_atten)


def _make_raycaster():
    rc = bp.Raycaster()
    rc.light_cull = 0.01
    rc.heat_cull = 0.01
    rc.smoke_absorb_scale = 1.4
    return rc


def _cast_all(rc, fire, dials, gas_f, gas_absorption, gas_scatter, light_atten,
              heat_atten, h, w):
    """Run OLD-CPU, OLD-CUDA, NEW-CPU, NEW-CUDA on the identical inputs;
    return the four heat buffers."""
    outs = {}
    for tag, fn in (("old_cpu", _old_cast_cpu), ("old_cuda", _old_cast_cuda),
                     ("new_cpu", _new_cast_cpu), ("new_cuda", _new_cast_cuda)):
        heat = np.zeros((h, w), np.int32)
        rgb = np.zeros((h, w, 3), np.float32)
        dx = np.zeros((h, w), np.float32)
        dy = np.zeros((h, w), np.float32)
        fn(rc, fire, dials["k_fire_heat"], dials["fire_ray_count"],
           dials["range_base"], dials["range_per_i"], dials["intensity_base"],
           dials["intensity_per_i"], dials["color"], rgb, dx, dy, gas_f,
           gas_absorption, gas_scatter, light_atten, heat, heat_atten)
        outs[tag] = heat
    return outs


def _compare(tag, outs, h, w) -> bool:
    ok = True
    heat_old, heat_new_cuda = outs["old_cuda"], outs["new_cuda"]
    if not np.array_equal(heat_old, heat_new_cuda):
        ok = False
        mism = int(np.count_nonzero(heat_old != heat_new_cuda))
        idx = int(np.argmax(heat_old != heat_new_cuda))
        ry, rx = divmod(idx, w)
        print(f"  {tag}: NEW-CUDA != OLD-CUDA — {mism} MISMATCH "
              f"(first @ ({ry},{rx}): old={heat_old.flat[idx]} "
              f"new={heat_new_cuda.flat[idx]})")
    if not np.array_equal(outs["new_cpu"], outs["new_cuda"]):
        ok = False
        mism = int(np.count_nonzero(outs["new_cpu"] != outs["new_cuda"]))
        print(f"  {tag}: NEW-CPU != NEW-CUDA — {mism} MISMATCH (cross-backend)")
    if not np.array_equal(outs["old_cpu"], outs["old_cuda"]):
        # Sanity on the ORACLE itself (already proven by S2/S8c, but a red
        # here means the scenario/harness is broken, not this patch).
        ok = False
        print(f"  {tag}: ORACLE INCONSISTENT — OLD-CPU != OLD-CUDA (harness bug, "
              f"not a P-R1 regression)")
    nz = int(np.count_nonzero(heat_old))
    peak = int(heat_old.max())
    if ok:
        print(f"  {tag}: all 4 casts (old-cpu/old-cuda/new-cpu/new-cuda) "
              f"byte-identical ({nz} heated / {h * w} cells, peak={peak}).")
    return ok


# ---------------------------------------------------------------------------
# Scenario (i): synthetic 600-fire firestorm.
# ---------------------------------------------------------------------------
def _synth_fire_plane(h, w, nfire, seed):
    rng = np.random.default_rng(seed)
    fire_q = np.zeros((h, w), dtype=np.int32)
    cells = set()
    while len(cells) < nfire:
        cells.add((int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))))
    for (yy, xx) in cells:
        i = float(rng.uniform(0.3, 1.0))
        v = i * FP_ONE_F
        fire_q[yy, xx] = int(np.floor(v + 0.5))   # round-half-away-from-zero

    n_gases = 2
    gas_f = (rng.random((n_gases, h, w)).astype(np.float32) * 0.6)
    gas_absorption = np.array([[1.0, 1.0, 1.0], [0.9, 0.2, 0.9]], np.float32)
    gas_scatter = np.array([[0.6, 0.6, 0.6], [0.1, 0.7, 0.1]], np.float32)
    light_atten = np.zeros((h, w, 3), np.float32)
    heat_atten = np.zeros((h, w), np.float32)
    heat_atten[h // 2, :] = 0.7
    heat_atten[:, w // 2] = 1.0
    for _ in range(40):
        ry, rx = int(rng.integers(0, h)), int(rng.integers(0, w))
        heat_atten[ry, rx] = float(rng.uniform(0.1, 1.0))
    return fire_q, gas_f, gas_absorption, gas_scatter, light_atten, heat_atten


def part1_firestorm() -> bool:
    print("PART 1 — 600-fire synthetic firestorm, all 4 casts byte-identical:")
    h, w, nfire = 128, 128, 600
    (fire_q, gas_f, gas_absorption, gas_scatter,
     light_atten, heat_atten) = _synth_fire_plane(h, w, nfire, seed=20260731)
    dials = dict(k_fire_heat=800.0, fire_ray_count=8, range_base=2.0,
                 range_per_i=6.0, intensity_base=0.3, intensity_per_i=0.7,
                 color=(1.0, 0.6, 0.2))
    rc = _make_raycaster()
    outs = _cast_all(rc, fire_q, dials, gas_f, gas_absorption, gas_scatter,
                      light_atten, heat_atten, h, w)
    n_fire_tiles = int(np.count_nonzero(fire_q))
    print(f"  {n_fire_tiles} fire sources ({h}x{w} grid)")
    ok = _compare("firestorm", outs, h, w)
    if int(np.count_nonzero(outs["old_cuda"])) == 0:
        ok = False
        print("  SCENARIO WEAK: heat never non-zero — vacuous gate")
    return ok


# ---------------------------------------------------------------------------
# Scenario (ii): the real "playground" level, several ticks, evolving fire.
# ---------------------------------------------------------------------------
N_TICKS = 15


def part2_playground_multitick() -> bool:
    print(f"PART 2 — playground level, {N_TICKS} ticks of evolving fire, "
          f"all 4 casts byte-identical (real PhysicsRunner dials):")
    from level_loader import load as load_level
    from simulation import fire_fixed
    from simulation import Simulation
    from simulation.physics_runner import PhysicsRunner
    from simulation.unit import Unit

    level = load_level("playground")
    sim = Simulation(level, seed=20260731, breach_physics=bp,
                     enable_recorder=False)
    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team,
                          footprint=s.footprint))
    g = sim.gmap
    interior = (~g.solid) & (~g.is_vacuum)
    ys, xs = np.nonzero(interior)
    rng = np.random.default_rng(20260731)
    n_fire = max(20, len(ys) // 30)
    pick = rng.choice(len(ys), size=min(n_fire, len(ys)), replace=False)
    for k in pick:
        yy, xx = int(ys[k]), int(xs[k])
        g.fire[yy, xx] = fire_fixed.quantize_scalar(float(rng.uniform(0.3, 1.0)))
    sim.set_paused(False)

    runner = sim.physics_runner
    if runner is None:
        print("  no physics_runner on the sim — cannot drive the live dials")
        return False
    rc = runner.raycaster
    h, w = g.fire.shape

    ok = True
    n_tick = 0
    max_peak = 0
    for t in range(N_TICKS):
        fire_snapshot = g.fire.copy()
        gas_f = (g.gas.astype(np.float64) / FP_ONE_F).astype(np.float32)
        dials = dict(k_fire_heat=runner.k_fire_heat,
                     fire_ray_count=runner.fire_ray_count,
                     range_base=runner.fire_range_base,
                     range_per_i=runner.fire_range_per_i,
                     intensity_base=runner.fire_intensity_base,
                     intensity_per_i=runner.fire_intensity_per_i,
                     color=runner.fire_color)
        outs = _cast_all(rc, fire_snapshot, dials, gas_f, g.gases.absorption,
                          g.gases.scatter_albedo, g.dyn_light_atten,
                          g.heat_atten, h, w)
        tick_ok = _compare(f"tick {t}", outs, h, w)
        ok = ok and tick_ok
        max_peak = max(max_peak, int(outs["old_cuda"].max()))
        n_tick += 1
        if not tick_ok:
            break

        # Advance the REAL sim one tick (real fire solver evolves the field) —
        # mirrors cuda_s2_check.py's / cuda_s2b_raycaster_live_check.py's
        # multitick pattern. Backend flags are untouched by this loop (the
        # sim's OWN internal cast_fire_heat call during sim.step() runs
        # whatever backend is currently globally set; irrelevant here since
        # we snapshot+compare BEFORE stepping).
        sim.step()

    if ok:
        print(f"  {n_tick} ticks byte-identical on the real playground level "
              f"(peak heat={max_peak}).")
    if max_peak == 0:
        ok = False
        print("  SCENARIO WEAK: heat never non-zero over the run — vacuous")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("PR1_FIRE_PLANE_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_firestorm()
    p2 = part2_playground_multitick()
    ok = p1 and p2
    print("PR1_FIRE_PLANE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
