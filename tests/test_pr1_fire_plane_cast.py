"""P-R1 gate (a), CPU half — byte-identity of the C++ fire-plane source build.

docs/radiation_raycaster_extinction_ruling_2026-07-31.md A4.1-A4.2: the
per-tile source build that used to run in PhysicsRunner.cast_fire_heat's
Python loop (one bp.LightSource() + ~10 pybind attribute writes PER BURNING
TILE, PER TICK) now runs inside Raycaster.cast_from_fire_plane (one call per
tick). This is a MECHANICAL RELOCATION — no march, deposit law, range, or fan
count changed — so ``heat`` must be byte-identical (tolerance ZERO) between
the new single call and the old per-tile loop.

This module is the CPU half of that oracle: ``_old_cast_cpu`` below is a
FROZEN, verbatim transcription of the pre-P-R1 ``cast_fire_heat`` per-tile
loop (physics_runner.py, before this patch) — it calls ONLY functions this
patch does not touch (``Raycaster.cast_source_directional``,
``bp.LightSource``), so running it through the patched build reproduces
exactly what the pre-patch build would have produced. Do NOT "modernize"
this function to match ``cast_from_fire_plane`` — its entire value as an
oracle is that it stays a faithful copy of the code being replaced.

Two scenarios, per the patch's gate (a):
  (i)  a synthetic 600-fire firestorm (mirrors
       tests/bench_s8c_fire_heat_check.py's 128x128/600-fire scale — the S8c
       payoff scenario this patch's docstring cites, ~6000 pybind writes/tick).
  (ii) the real "playground" level (docs/radiation_raycaster_extinction_
       ruling_2026-07-31.md's "default playground level"), fire seeded on
       real interior tiles with the REAL production dials (a live
       PhysicsRunner), stepped several ticks with the fire solver evolving
       the field between casts (mirrors tests/cuda_s2_check.py's
       part1b_multitick_live pattern).

The CUDA half (NEW cuda_raycaster_cast_from_fire_plane vs OLD
cuda_raycaster_cast_batch) is tests/cuda_pr1_fire_plane_check.py (skips
cleanly without a CUDA build).

Run:
    conda run -n data python -m pytest tests/test_pr1_fire_plane_cast.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import fire_fixed  # noqa: E402  S3a: gmap.fire is int32 Q16.16
from simulation import Simulation  # noqa: E402
from simulation.physics_runner import PhysicsRunner  # noqa: E402
from simulation.unit import Unit  # noqa: E402

FP_ONE_F = fire_fixed.FP_ONE_F   # 65536.0


# ---------------------------------------------------------------------------
# The oracle: a frozen, verbatim transcription of the PRE-P-R1 per-tile loop
# (physics_runner.py's old cast_fire_heat, CPU branch). Calls only
# cast_source_directional + bp.LightSource — neither touched by this patch.
# ---------------------------------------------------------------------------
def _old_cast_cpu(raycaster, fire, k_fire_heat, fire_ray_count, range_base,
                   range_per_i, intensity_base, intensity_per_i, color,
                   rgb, dx, dy, gas_f, gas_absorption, gas_scatter,
                   light_atten, heat, heat_atten):
    two_pi = 2.0 * math.pi
    ray_count = fire_ray_count
    ys, xs = np.nonzero(fire > 0)
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
        raycaster.cast_source_directional(
            src, rgb, dx, dy, gas_f, gas_absorption, gas_scatter, light_atten,
            heat=heat, smoke_glow=None, heat_atten=heat_atten)


def _new_cast_cpu(raycaster, fire, k_fire_heat, fire_ray_count, range_base,
                   range_per_i, intensity_base, intensity_per_i, color,
                   rgb, dx, dy, gas_f, gas_absorption, gas_scatter,
                   light_atten, heat, heat_atten):
    raycaster.cast_from_fire_plane(
        fire, k_fire_heat, fire_ray_count, range_base, range_per_i,
        intensity_base, intensity_per_i, color,
        rgb, dx, dy, gas_f, gas_absorption, gas_scatter, light_atten,
        heat=heat, smoke_glow=None, heat_atten=heat_atten)


def _make_raycaster():
    rc = bp.Raycaster()
    rc.light_cull = 0.01
    rc.heat_cull = 0.01
    rc.smoke_absorb_scale = 1.4
    return rc


# ---------------------------------------------------------------------------
# Scenario (i): synthetic 600-fire firestorm (raw arrays, no GameMap needed —
# cast_from_fire_plane's CPU signature is pure numpy arrays + scalar dials).
# ---------------------------------------------------------------------------
def _synth_fire_plane(h, w, nfire, seed):
    rng = np.random.default_rng(seed)
    fire_q = np.zeros((h, w), dtype=np.int32)
    cells = set()
    while len(cells) < nfire:
        cells.add((int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))))
    for (yy, xx) in cells:
        fire_q[yy, xx] = fire_fixed.quantize_scalar(float(rng.uniform(0.3, 1.0)))

    n_gases = 2
    gas_f = (rng.random((n_gases, h, w)).astype(np.float32) * 0.6)
    gas_absorption = np.array([[1.0, 1.0, 1.0], [0.9, 0.2, 0.9]], np.float32)
    gas_scatter = np.array([[0.6, 0.6, 0.6], [0.1, 0.7, 0.1]], np.float32)
    light_atten = np.zeros((h, w, 3), np.float32)
    heat_atten = np.zeros((h, w), np.float32)
    # A couple of full/partial heat walls + a scatter of random occluders, so
    # the occlusion branch (heat_survival decay, source-tile self-occlusion
    # skip) is exercised, not just open-air marches.
    heat_atten[h // 2, :] = 0.7
    heat_atten[:, w // 2] = 1.0
    for _ in range(40):
        ry, rx = int(rng.integers(0, h)), int(rng.integers(0, w))
        heat_atten[ry, rx] = float(rng.uniform(0.1, 1.0))
    return fire_q, gas_f, gas_absorption, gas_scatter, light_atten, heat_atten


def test_600_fire_firestorm_byte_identical():
    print("\nP-R1 gate (a)(i) — 600-fire synthetic firestorm, CPU heat byte-identity:")
    h, w, nfire = 128, 128, 600
    (fire_q, gas_f, gas_absorption, gas_scatter,
     light_atten, heat_atten) = _synth_fire_plane(h, w, nfire, seed=20260731)

    dials = dict(k_fire_heat=800.0, fire_ray_count=8, range_base=2.0,
                 range_per_i=6.0, intensity_base=0.3, intensity_per_i=0.7,
                 color=(1.0, 0.6, 0.2))

    rc = _make_raycaster()

    heat_old = np.zeros((h, w), np.int32)
    rgb_old = np.zeros((h, w, 3), np.float32)
    dx_old = np.zeros((h, w), np.float32)
    dy_old = np.zeros((h, w), np.float32)
    _old_cast_cpu(rc, fire_q, dials["k_fire_heat"], dials["fire_ray_count"],
                  dials["range_base"], dials["range_per_i"],
                  dials["intensity_base"], dials["intensity_per_i"],
                  dials["color"], rgb_old, dx_old, dy_old, gas_f,
                  gas_absorption, gas_scatter, light_atten, heat_old, heat_atten)

    heat_new = np.zeros((h, w), np.int32)
    rgb_new = np.zeros((h, w, 3), np.float32)
    dx_new = np.zeros((h, w), np.float32)
    dy_new = np.zeros((h, w), np.float32)
    _new_cast_cpu(rc, fire_q, dials["k_fire_heat"], dials["fire_ray_count"],
                  dials["range_base"], dials["range_per_i"],
                  dials["intensity_base"], dials["intensity_per_i"],
                  dials["color"], rgb_new, dx_new, dy_new, gas_f,
                  gas_absorption, gas_scatter, light_atten, heat_new, heat_atten)

    n_fire_tiles = int(np.count_nonzero(fire_q))
    n_heated = int(np.count_nonzero(heat_old))
    print(f"  {n_fire_tiles} fire sources -> {n_heated} heated tiles "
          f"({h * w} total cells), peak heat={int(heat_old.max())}")
    assert n_fire_tiles == nfire
    assert n_heated > 0, "scenario deposited no heat at all — vacuous gate"
    mismatches = int(np.count_nonzero(heat_old != heat_new))
    if mismatches:
        idx = int(np.argmax(heat_old != heat_new))
        ry, rx = divmod(idx, w)
        raise AssertionError(
            f"{mismatches}/{h * w} cells MISMATCH (first @ ({ry},{rx}): "
            f"old={heat_old.flat[idx]} new={heat_new.flat[idx]})")
    assert np.array_equal(heat_old, heat_new)
    print(f"  heat: byte-identical over all {h * w} cells "
          f"({n_heated} nonzero, 0 mismatches).")


# ---------------------------------------------------------------------------
# Scenario (ii): the real "playground" level, several ticks, evolving fire,
# the REAL production PhysicsRunner dials (config.toml).
# ---------------------------------------------------------------------------
N_TICKS = 15


def _fresh_playground_sim(seed):
    level = load_level("playground")
    sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False)
    for s in level.spawns:
        sim.add_unit(Unit(s.name, x=s.x, y=s.y, team=s.team, footprint=s.footprint))
    g = sim.gmap
    interior = (~g.solid) & (~g.is_vacuum)
    ys, xs = np.nonzero(interior)
    rng = np.random.default_rng(seed)
    n_fire = max(20, len(ys) // 30)
    pick = rng.choice(len(ys), size=min(n_fire, len(ys)), replace=False)
    for k in pick:
        yy, xx = int(ys[k]), int(xs[k])
        g.fire[yy, xx] = fire_fixed.quantize_scalar(float(rng.uniform(0.3, 1.0)))
    sim.set_paused(False)
    return sim


def test_playground_level_several_ticks_byte_identical():
    print(f"\nP-R1 gate (a)(ii) — playground level, {N_TICKS} ticks of evolving "
          f"fire, CPU heat byte-identity (real PhysicsRunner dials):")
    sim = _fresh_playground_sim(seed=20260731)
    g = sim.gmap
    runner = sim.physics_runner
    assert runner is not None, "no physics_runner on the sim"
    rc = runner.raycaster

    h, w = g.fire.shape
    n_tick = 0
    max_peak = 0
    total_heated = 0
    for t in range(N_TICKS):
        fire_snapshot = g.fire.copy()

        heat_old = np.zeros((h, w), np.int32)
        rgb_o = np.zeros((h, w, 3), np.float32)
        dx_o = np.zeros((h, w), np.float32)
        dy_o = np.zeros((h, w), np.float32)
        gas_f = (g.gas.astype(np.float64) / 65536.0).astype(np.float32)
        _old_cast_cpu(rc, fire_snapshot, runner.k_fire_heat, runner.fire_ray_count,
                      runner.fire_range_base, runner.fire_range_per_i,
                      runner.fire_intensity_base, runner.fire_intensity_per_i,
                      runner.fire_color, rgb_o, dx_o, dy_o, gas_f,
                      g.gases.absorption, g.gases.scatter_albedo,
                      g.dyn_light_atten, heat_old, g.heat_atten)

        heat_new = np.zeros((h, w), np.int32)
        rgb_n = np.zeros((h, w, 3), np.float32)
        dx_n = np.zeros((h, w), np.float32)
        dy_n = np.zeros((h, w), np.float32)
        _new_cast_cpu(rc, fire_snapshot, runner.k_fire_heat, runner.fire_ray_count,
                      runner.fire_range_base, runner.fire_range_per_i,
                      runner.fire_intensity_base, runner.fire_intensity_per_i,
                      runner.fire_color, rgb_n, dx_n, dy_n, gas_f,
                      g.gases.absorption, g.gases.scatter_albedo,
                      g.dyn_light_atten, heat_new, g.heat_atten)

        if not np.array_equal(heat_old, heat_new):
            mismatches = int(np.count_nonzero(heat_old != heat_new))
            idx = int(np.argmax(heat_old != heat_new))
            ry, rx = divmod(idx, w)
            raise AssertionError(
                f"tick {t}: {mismatches} cell MISMATCH (first @ ({ry},{rx}): "
                f"old={heat_old.flat[idx]} new={heat_new.flat[idx]})")

        n_fire_tiles = int(np.count_nonzero(fire_snapshot))
        n_heated = int(np.count_nonzero(heat_old))
        total_heated += n_heated
        max_peak = max(max_peak, int(heat_old.max()))
        n_tick += 1
        print(f"  tick {t}: {n_fire_tiles} fire tiles -> {n_heated} heated "
              f"tiles, byte-identical")

        # Advance the REAL sim one tick (real fire solver: growth/spread/
        # decay/extinction) so the NEXT comparison sees an evolved field —
        # mirrors tests/cuda_s2_check.py::part1b_multitick_live.
        sim.step()

    assert n_tick == N_TICKS
    assert max_peak > 0, "heat was never non-zero over the run — vacuous gate"
    print(f"  all {n_tick} ticks byte-identical on the real playground level "
          f"(peak heat={max_peak}, total heated-tile-ticks={total_heated}).")


if __name__ == "__main__":
    test_600_fire_firestorm_byte_identical()
    test_playground_level_several_ticks_byte_identical()
    print("OK — P-R1 CPU byte-identity gate (600-fire firestorm + playground level)")
