"""S3b fire DISCRETE-OUTPUT determinism + overflow stress (the fire-specific gates).

The S3b integer logistic makes two DISCRETE control-flow outputs that drive synced
state and MUST be bit-deterministic (plan §5.3, §6):

  * the EXTINGUISH FLIP — `I_next < I_min -> 0` snaps a fire out; a 1-LSB slip would
    flip it on a different tick on a peer -> desync (the renderer's fire on/off).
  * the BURN-THROUGH LIST — `wall_hp <= 0 -> destroyed` drives `destroy_wall` (a
    topology change). The (y,x) list + the tick it lands on must be identical.

Both are integer compares on integer fields -> bit-identical by construction. These
tests assert that empirically across an ignite -> firestorm -> starve -> extinguish
trajectory (run twice, same seed), AND drive a SHOCKWAVE-FANNED firestorm (a grenade
wave through a blaze, the worst-case W = |wind| via sqrt_q16) so the logistic chain +
the int64 radicand are exercised without an int64->int32 narrow overflow.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_s3b_fire_determinism.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp           # noqa: E402
from level_loader import LevelData    # noqa: E402
from simulation import Simulation     # noqa: E402
from simulation import fire_fixed, wave_fixed, atmosphere_fixed, wall_fixed  # noqa: E402
from simulation.materials import MAT_WOOD  # noqa: E402

SEED = 31337
# P-F1b (2026-08-02, docs/fire_recalibration_2026-08-02.md): 90 -> 3400. This
# trajectory's LAST discrete event is the extinguish flip, and the horizon was
# sized for fires that could not sustain themselves at all (the shipped
# k_die/k_grow demanded 3.5x more oxygen availability than the atmosphere can
# supply, so a seeded blaze snapped out inside a second). The recalibration
# restores the sustain condition, so the held-and-fanned blaze now burns its
# room down over ~2 minutes (measured: extinguish flip at tick 3205) before it
# starves. The gate itself (bit-identical
# run-to-run, and the trajectory must really ignite AND really extinguish) is
# unchanged; only the window it needs to contain both events moved.
TICKS = 3400
SY, SX = 8, 8


def _wood_room() -> LevelData:
    h = w = 20
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:19, 1:19] = 4
    return LevelData(name="s3b_det", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _make_sim() -> Simulation:
    sim = Simulation(_wood_room(), seed=SEED, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            y, x = SY + dy, SX + dx
            if 1 <= y < 19 and 1 <= x < 19:
                g.material[y, x] = MAT_WOOD
    g._update_caches()
    sim.set_paused(False)
    return sim


def _run_trajectory():
    """Ignite -> firestorm (held + wind-fanned) -> starve -> extinguish. Returns the
    per-tick fire-field snapshots, the per-tick destroyed-wall lists, and the
    per-tick lit-cell counts."""
    sim = _make_sim()
    g = sim.gmap
    seed_q = fire_fixed.quantize_scalar(0.9)
    wind_q = atmosphere_fixed.quantize_scalar(2.0)
    fire_snaps = []
    destroyed_per_tick = []
    lit = []
    hold_until = 30
    for t in range(TICKS):
        if t < hold_until:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    g.fire[SY + dy, SX + dx] = max(int(g.fire[SY + dy, SX + dx]), seed_q)
        if 5 <= t < hold_until:
            g.wind_x[:] = wind_q
        if t == 5:
            g.wave_source[SY, SX + 5] = wave_fixed.quantize_scalar(8.0)
        # Snapshot the destroyed-wall list this tick (the burn-through control output).
        solid_before = g.solid.copy()
        sim.set_paused(False)
        sim.step()
        # Walls that became non-solid this tick (burn-through or breach).
        newly = np.argwhere(solid_before & (~g.solid))
        destroyed_per_tick.append([tuple(int(v) for v in yx) for yx in newly])
        fire_snaps.append(g.fire.copy())          # int32 Q16.16 (exact compare)
        lit.append(int((g.fire > fire_fixed.quantize_scalar(0.01)).sum()))
    return fire_snaps, destroyed_per_tick, lit


def test_fire_field_and_burnthrough_list_bit_identical_run_twice():
    """The extinguish-flip tick AND the burn-through destroyed-tile list are
    bit-identical across two runs of the ignite->firestorm->starve->extinguish
    trajectory (the discrete-output determinism gate)."""
    fa, da, la = _run_trajectory()
    fb, db, lb = _run_trajectory()
    assert len(fa) == len(fb) == TICKS
    for t in range(TICKS):
        assert np.array_equal(fa[t], fb[t]), (
            f"fire field diverged run-to-run at tick {t} (discrete logistic desync)")
        assert da[t] == db[t], (
            f"burn-through destroyed-tile list diverged at tick {t}: {da[t]} != {db[t]}")
    assert la == lb, "lit-cell trajectory diverged run-to-run"
    # The trajectory MUST actually exercise the discrete events (ignite then later a
    # full extinguish), else the gate is vacuous.
    assert max(la) > 1, f"the firestorm never spread (peak lit {max(la)}) — not a real trajectory"
    assert la[-1] == 0, f"the fire never extinguished (final lit {la[-1]}) — no extinguish flip exercised"


def test_shockwave_fanned_firestorm_no_overflow():
    """Drive a grenade shockwave through a blaze (the worst-case W = |wind| via
    sqrt_q16 + the (1 + k_wind_fan*W) logistic factor) and assert the fire field
    stays in its valid Q16.16 [0,1] range every tick — an int64->int32 narrow
    overflow in the logistic chain or a wrapped sqrt would blow the field out of
    [0, FP_ONE] (negative garbage or > 1). The wall_hp field also stays sane."""
    sim = _make_sim()
    g = sim.gmap
    seed_q = fire_fixed.quantize_scalar(0.95)
    # A VIOLENT shockwave: a big wave_source kick AND a directly-spiked wind so W is
    # large (the firestorm-fan worst case). |wind| ~ 30 here -> rad ~ 2*(30*65536)^2
    # ~ 7.7e12, well inside the int64 radicand bound; sqrt_q16 -> ~2.8e6 counts.
    big_wind = atmosphere_fixed.quantize_scalar(30.0)
    FP_ONE = fire_fixed.FP_ONE
    for t in range(60):
        if t < 20:
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    g.fire[SY + dy, SX + dx] = max(int(g.fire[SY + dy, SX + dx]), seed_q)
        if t == 3:
            g.wave_source[SY, SX] = wave_fixed.quantize_scalar(30.0)   # violent blast
        if 3 <= t < 25:
            g.wind_x[:] = big_wind
            g.wind_y[:] = big_wind
        sim.set_paused(False)
        sim.step()
        f = g.fire
        assert f.dtype == np.int32, "fire must stay int32 Q16.16"
        assert int(f.min()) >= 0, (
            f"fire went NEGATIVE at tick {t} (min={int(f.min())}) — int64->int32 "
            f"narrow overflow / wrapped sqrt in the logistic chain")
        assert int(f.max()) <= FP_ONE, (
            f"fire exceeded FP_ONE (1.0) at tick {t} (max={int(f.max())}) — the "
            f"clamp01 / chain overflowed")
        # wall_hp stays a sane Q16.16 (the burn-through depletion never wraps).
        assert int(g.wall_hp.max()) < (1 << 30), "wall_hp overflowed its Q16.16 range"


# ---------------------------------------------------------------------------
# Cross-config self-consistency (P1): vary the tick length dt that feeds the
# logistic AND the [physics.fire] params; each config self-matches bit-for-bit
# run-to-run (the integer logistic is deterministic for ANY config, not just the
# shipped one — a config-dependent float leak would fail this).
# ---------------------------------------------------------------------------
def _drive_isolated_fire(dt, *, k_grow, k_die, k_wind_fan, ticks=40):
    """Drive the C++ FireSimulation.step in isolation on a 5x5 wood-centre scene
    with the given dt + fire params; return the final fire field (int32 Q16.16)."""
    from simulation.physics_runner import PhysicsRunner
    from simulation.materials import MaterialTable, MAT_AIR
    tbl = MaterialTable.from_config()
    fs = PhysicsRunner(bp).fire
    fs.params.k_grow = float(k_grow)
    fs.params.k_die = float(k_die)
    fs.params.k_wind_fan = float(k_wind_fan)
    h = w = 5
    m = np.full((h, w), MAT_AIR, dtype=np.int8)
    m[2, 2] = MAT_WOOD
    flammable = np.ascontiguousarray(tbl.flammable[m])
    solid = np.ascontiguousarray(tbl.permeability[m] <= 0.0)
    is_vac = np.zeros((h, w), dtype=bool)
    atm = np.where(solid, 0, atmosphere_fixed.quantize_scalar(1.0)).astype(np.int32)
    # EOS refactor P4: n_o2 is the O2 gate's own input, non-limiting here
    # (this test is about dt/param-config determinism, not O2 starvation).
    n_o2 = np.where(solid, 0, atmosphere_fixed.quantize_scalar(1.0)).astype(np.int32)
    smoke = np.zeros((h, w), dtype=np.int32)
    wall_hp = np.zeros((h, w), dtype=np.int32)
    wall_hp[2, 2] = wall_fixed.quantize_scalar(60.0)
    temperature = np.zeros((h, w), dtype=np.int32)
    temperature[2, 2] = int(round(500.0 * 65536))
    fire = np.zeros((h, w), dtype=np.int32)
    fire[2, 2] = fire_fixed.quantize_scalar(0.5)
    wind_x = np.full((h, w), atmosphere_fixed.quantize_scalar(1.5), dtype=np.int32)
    wind_y = np.zeros((h, w), dtype=np.int32)
    for _ in range(ticks):
        fs.step(fire, atm, n_o2, smoke, wall_hp, temperature, wind_x, wind_y,
                solid, is_vac, flammable, float(dt))
    return fire.copy()


def test_cross_config_self_match():
    """Each (dt, fire-param) config self-matches bit-for-bit run-to-run, and a
    DIFFERENT config produces a DIFFERENT field (so the sweep is not vacuous)."""
    configs = [
        dict(dt=1.0 / 24, k_grow=4.0, k_die=2.0, k_wind_fan=0.5),   # shipped
        dict(dt=1.0 / 30, k_grow=4.0, k_die=2.0, k_wind_fan=0.5),   # higher tps
        dict(dt=1.0 / 12, k_grow=6.0, k_die=1.5, k_wind_fan=1.0),   # lower tps + params
    ]
    results = []
    for cfg in configs:
        a = _drive_isolated_fire(**cfg)
        b = _drive_isolated_fire(**cfg)
        assert np.array_equal(a, b), f"config {cfg} not bit-identical run-to-run"
        results.append(a)
    # The three configs should NOT all be identical (else the params do nothing —
    # the test would be vacuously self-matching).
    assert not (np.array_equal(results[0], results[1])
                and np.array_equal(results[0], results[2])), (
        "all configs produced the identical field — the dt/param sweep is vacuous")
