"""GATE 9 and the WIRING of the shadow sweep (ray-engine-v2 P1, design v3 §2.9
gate 9, §3 "goldens unmoved at P1", §11 "is P1 wired?").

The goldens themselves are asserted unmoved by the tests that own them
(tests/test_w6_armory.py against GOLDEN_AGGREGATE, the per-field goldens);
this file adds the two conditions the design says must be CHECKED rather than
believed, plus the proof that the sweep is live:

  * the A/B default scenario is WRAP-FREE on the old cast's int32 `rad_net` —
    detected through the pre-fold ledger identity `Σ rad_net + Σ rad_amb == 0`
    (a wrap breaks it by exactly 2^32, tests/test_pf1a_radiation_books.py) and
    the measured max|rad_net| reported. FINDING: that scenario is radiatively
    INERT — its seeded fire delivers no heat (CLAUDE.md "Starting a fire"), no
    thermal solid ever leaves ambient, and the old cast's rad_net is zero on
    every tick — so the check there is exact but vacuous; the same check on a
    shipped level with a real hot tile is the non-vacuous one;
  * the shadow sweep's three-term identity holds on the live engine EVERY
    tick and its Φ plane is non-zero on every cell (the ambient ring lights
    the grid even when nothing radiates);
  * and, since P2a (design row 38), the four planes SURVIVE the tick — the
    conductor's four `fill(0)` lines are gone and `RadiationSweep::run`
    zeroes its own outputs instead, so the render-time tile inspector reads
    real numbers; the second tick must therefore OVERWRITE, not accumulate.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_radiation_sweep_shadow_wiring.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from field_ab_harness import default_scenario_sim  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation, fire_fixed  # noqa: E402
from simulation.materials import MAT_WOOD  # noqa: E402

INT32_LIMIT = 2 ** 31
_SWEEP_PLANES = ("rad_net_sweep", "rad_flux_sweep", "rad_amb_sweep", "rad_fluence")


def _sweep_dials(sim):
    """(t_amb_q, k_leak_q) exactly as PhysicsRunner hands them to step 2b — read
    off the runner, never hardcoded, so a dial change cannot make the direct-run
    comparison below silently compare two different laws."""
    runner = sim.physics_runner
    return int(runner._eos_t_amb_raw()), int(runner.k_leak_q)


def _watch_physics(sim):
    """Wrap the runner's step so every per-tick plane is read right after the
    physics tail and before the conductor wipes it."""
    runner = sim.physics_runner
    orig = runner.step
    seen = {"max_abs_rad_net": 0, "old_identity": [], "sweep_identity": [],
            "sweep_nonzero_ticks": 0, "old_nonzero_ticks": 0,
            "fluence_min": None, "fluence_max": 0, "ticks": 0}

    def wrapped(gmap, sim_time, tick=0):
        out = orig(gmap, sim_time, tick=tick)
        rn = gmap.rad_net.astype(np.int64)
        ra = gmap.rad_amb.astype(np.int64)
        seen["max_abs_rad_net"] = max(seen["max_abs_rad_net"], int(np.abs(rn).max()))
        seen["old_identity"].append(int(rn.sum()) + int(ra.sum()))
        seen["old_nonzero_ticks"] += int(bool(np.any(rn != 0)))
        s = (int(gmap.rad_net_sweep.sum()) + int(gmap.rad_flux_sweep.sum())
             + int(gmap.rad_amb_sweep.sum()))
        seen["sweep_identity"].append(s)
        seen["sweep_nonzero_ticks"] += int(bool(np.any(gmap.rad_net_sweep != 0)))
        fmin, fmax = int(gmap.rad_fluence.min()), int(gmap.rad_fluence.max())
        seen["fluence_min"] = fmin if seen["fluence_min"] is None else min(seen["fluence_min"], fmin)
        seen["fluence_max"] = max(seen["fluence_max"], fmax)
        seen["ticks"] += 1
        return out

    runner.step = wrapped
    return seen


def _ambient_fluence_s16(sim):
    """16 x amb_m: what every cell's Φ is when nothing radiates (design §2.3)."""
    e0 = int(np.asarray(sim.physics_runner.engine.emissive.table())[0])
    return 16 * ((e0 * 4096) >> 16)


def test_ab_default_scenario_is_wrap_free_and_the_sweep_runs_every_tick():
    """PROPERTY (design §3 condition (i), critique 3 §6c): over 40 ticks of the
    A/B default scenario the old cast's pre-fold identity holds every tick (no
    int32 cell wrapped), max|rad_net| < 2^31 is CHECKED and reported; the
    shadow sweep's three-term identity holds every tick; and Φ is exactly the
    ambient fluence 16·amb_m on every cell every tick (step 2b ran — and the
    scene is radiatively inert, which is the FINDING this test records).

    BREAKS IF: the old cast wraps on the goldens' scene, step 2b is not wired
    (Φ would be zero), or a sweep term stops being booked.
    """
    sim = default_scenario_sim()
    seen = _watch_physics(sim)
    for _ in range(40):
        sim.set_paused(False)
        sim.step()
    assert seen["ticks"] == 40
    assert all(v == 0 for v in seen["old_identity"]), seen["old_identity"][:5]
    assert seen["max_abs_rad_net"] < INT32_LIMIT
    assert all(v == 0 for v in seen["sweep_identity"]), seen["sweep_identity"][:5]
    amb = _ambient_fluence_s16(sim)
    assert seen["fluence_min"] == amb and seen["fluence_max"] == amb, (seen["fluence_min"], amb)
    print(f"\nA/B default scenario, 40 ticks: max|rad_net| (old cast) = "
          f"{seen['max_abs_rad_net']}; old cast non-zero on {seen['old_nonzero_ticks']}/40 "
          f"ticks; sweep rad_net_sweep non-zero on {seen['sweep_nonzero_ticks']}/40 ticks; "
          f"Phi == 16*amb_m == {amb} on every cell (the scene is radiatively inert; "
          f"the wrap check here is exact but vacuous)")


def _playground_with_a_hot_wood_tile():
    lvl = load_level("playground")
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    ys, xs = np.where((g.material == MAT_WOOD) & g.thermal_solid)
    assert len(ys) > 0, "the level carries no wood tile"
    pick = None
    for y, x in zip(ys, xs):
        if 0 < y < g.material.shape[0] - 1 and 0 < x < g.material.shape[1] - 1:
            if not g.thermal_solid[y, x + 1] or not g.thermal_solid[y + 1, x]:
                pick = (int(y), int(x))
                break
    assert pick is not None, "no wood tile with an air neighbour"
    y, x = pick
    # A thermal SOLID's temperature is its own truth (the mirror rule is about
    # gas cells); a fire is started by delivering heat -- both, so the tile is
    # burning AND hot, as CLAUDE.md's "Starting a fire" row requires.
    g.temperature[y, x] = 1263 << 16
    g.fire[y, x] = fire_fixed.quantize_scalar(0.8)
    return sim, (y, x)


def test_sweep_on_a_real_level_is_non_trivial_after_one_tick():
    """PROPERTY: on a shipped level, one physics tick after a wood tile is
    heated to the crate plateau (1263 game) and lit, the shadow sweep books a
    NEGATIVE rad_net_sweep on that tile (it radiates), a POSITIVE one on some
    other cell (it absorbs), Φ exceeds the ambient fluence somewhere, and the
    three-term identity is exact — read directly after PhysicsRunner.step,
    before the conductor's wipe. The old cast is active on the same scene
    (its rad_net is non-zero: the hot tile is above T_emit_gate), so the
    old-law identity + max|rad_net| are checked NON-vacuously here too.

    BREAKS IF: the sweep is not wired into step_tail, reads the wrong
    temperature plane, or the extinction planes are not projected from the
    material table (a wood tile with a == 0 would neither emit nor absorb).
    """
    sim, (y, x) = _playground_with_a_hot_wood_tile()
    g = sim.gmap
    # The dynamic plane's resting state equals the static plane (d == a with
    # no bodies), so the invariant 0 <= a <= d holds for a direct physics call
    # even before the conductor's first stamp; the stamp is still run here to
    # mirror the conductor's slot order.
    assert g.heat_atten_q[y, x] == 65536 and g.dyn_heat_atten_q[y, x] == 65536
    assert np.all(g.dyn_heat_atten_q >= g.heat_atten_q)
    g.stamp_units(sim.units)
    assert g.dyn_heat_atten_q[y, x] == 65536
    sim.physics_runner.step(g, 1.0 / 24.0, tick=0)
    rn, rf, ra = g.rad_net_sweep, g.rad_flux_sweep, g.rad_amb_sweep
    amb = _ambient_fluence_s16(sim)
    assert rn[y, x] < 0, "the hot wood tile must radiate (rad_net_sweep < 0)"
    assert int(rn.max()) > 0, "some cell must absorb (rad_net_sweep > 0)"
    assert int(rn.sum()) + int(rf.sum()) + int(ra.sum()) == 0
    assert np.all(g.rad_fluence > 0) and int(g.rad_fluence.max()) > amb
    old_rn = g.rad_net.astype(np.int64)
    old_ra = g.rad_amb.astype(np.int64)
    assert np.any(old_rn != 0), "the old cast must be active on this scene"
    assert int(old_rn.sum()) + int(old_ra.sum()) == 0
    assert int(np.abs(old_rn).max()) < INT32_LIMIT
    print(f"\nplayground level: hot wood at ({y},{x}) rad_net_sweep={int(rn[y, x])}, "
          f"max absorber {int(rn.max())}, sweep identity exact; old cast max|rad_net| = "
          f"{int(np.abs(old_rn).max())}, old-law identity exact")


def test_shadow_planes_survive_the_tick_and_hold_the_sweep_s_own_values():
    """PROPERTY (P2a, design row 38): after Simulation.step RETURNS, the four
    shadow planes still hold THIS tick's sweep output — the conductor no longer
    wipes them, because the sweep zeroes them itself at its own start. That is
    what lets the renderer's tile inspector show Φ, `f` and E°⁻¹(Φ) at render
    time; while the wipe lived in the conductor, Φ was 0 by then and the rows
    were dead. Nothing digested changes: none of the four is in DIGEST_FIELDS
    or SIM_FIELDS.

    Asserted against the values read MID-tick (right after PhysicsRunner.step),
    so "still there" means equal to what the sweep actually wrote, not merely
    non-zero.

    BREAKS IF: a `fill(0)` for these planes comes back into Simulation.step, or
    run() stops writing them.
    """
    sim, _ = _playground_with_a_hot_wood_tile()
    g = sim.gmap
    mid = {}

    runner = sim.physics_runner
    orig = runner.step

    def wrapped(gmap, sim_time, tick=0):
        out = orig(gmap, sim_time, tick=tick)
        for name in _SWEEP_PLANES:
            mid[name] = getattr(gmap, name).copy()
        return out

    runner.step = wrapped
    sim.set_paused(False)
    sim.step()
    assert mid, "the physics tail did not run"
    assert int(mid["rad_fluence"].min()) > 0 and np.any(mid["rad_net_sweep"] != 0)
    for name in _SWEEP_PLANES:
        plane = getattr(g, name)
        assert plane.dtype == np.int64 and plane.shape == g.temperature.shape
        assert np.array_equal(plane, mid[name]), (
            f"{name} was altered between the physics tail and the end of the tick "
            f"(the conductor must NOT wipe it any more)")
    print(f"\nafter Simulation.step: Phi min = {int(g.rad_fluence.min())}, "
          f"max = {int(g.rad_fluence.max())}; rad_net_sweep extremes "
          f"{int(g.rad_net_sweep.min())} / {int(g.rad_net_sweep.max())} — all still "
          f"readable at render time")


def test_a_second_tick_overwrites_the_planes_and_does_not_accumulate():
    """PROPERTY (P2a, the non-vacuous half): because nothing wipes the planes
    between ticks any more, the sweep MUST overwrite them. After two ticks each
    plane equals what ONE direct RadiationSweep.run on that tick's own inputs
    produces — not twice it. This is the exact bug the patch would have caused
    had run() kept accumulating into whatever the caller left behind.

    BREAKS IF: run()'s own zeroing is dropped (every plane would then be the sum
    of two ticks, which the `2 *` comparison below detects), or the sweep starts
    reading state that survives a tick.
    """
    sim, (y, x) = _playground_with_a_hot_wood_tile()
    g = sim.gmap
    sim.set_paused(False)
    sim.step()
    after_one = {n: getattr(g, n).copy() for n in _SWEEP_PLANES}
    # tick 2: snapshot the sweep's INPUTS as the physics tail sees them, then
    # let the tick finish and compare against a single direct run on them.
    grabbed = {}
    runner = sim.physics_runner
    orig = runner.step

    def wrapped(gmap, sim_time, tick=0):
        grabbed["inputs"] = (gmap.temperature.copy(), gmap.heat_atten_q.copy(),
                             gmap.dyn_heat_atten_q.copy(), gmap.heat_inv_shift.copy(),
                             gmap.thermal_solid.copy())
        return orig(gmap, sim_time, tick=tick)

    runner.step = wrapped
    sim.step()
    assert "inputs" in grabbed
    T, a_q, d_q, his, ts = grabbed["inputs"]
    eng = sim.physics_runner.engine
    h, w = T.shape
    one = [np.zeros((h, w), dtype=np.int64) for _ in range(4)]
    t_amb_q, k_leak_q = _sweep_dials(sim)
    sweep = bp.RadiationSweep()
    sweep.run(T, a_q, d_q, his, ts, eng.emissive, t_amb_q, k_leak_q,
              bp.RadiationSweep.SHEAR, 16, *one)
    for name, expect in zip(_SWEEP_PLANES, one):
        got = getattr(g, name)
        assert np.array_equal(got, expect), (
            f"{name} after tick 2 is not one tick's worth of sweep")
        if np.any(after_one[name] != 0):
            assert not np.array_equal(got, after_one[name] + expect), (
                f"{name} accumulated across ticks (run() lost its own zeroing)")
    print(f"\ntick 2 overwrites: Phi max = {int(g.rad_fluence.max())} equals a single "
          f"direct run on the same inputs; tick-1 values were not added to")


def test_stale_caller_cannot_hand_step_tail_a_narrow_plane():
    """PROPERTY (critique 3 §6a, made loud): PhysicsEngine.step_tail REFUSES an
    int32 array where an int64 shadow plane is expected (TypeError), and
    refuses to run without the six radiation arguments at all — a stale
    caller cannot write into a silently widened temporary. NOTE: without
    `noconvert`, pybind11's convert pass would still perform the SAFE int32 ->
    int64 cast into a discarded copy; the loud failure comes from the
    `.noconvert()` on those arguments, not from the absence of forcecast.

    BREAKS IF: the binding regains a default or loses noconvert on those args.
    """
    sim = default_scenario_sim()
    g = sim.gmap
    eng = sim.physics_runner.engine
    common = dict(ripple=g.ripple, ripple_v=g.ripple_v, water_depth=g.water_depth,
                  wave_p=g.wave_p, solid=g.solid, fire=g.fire, atmosphere=g.atmosphere,
                  smoke=g.smoke, wall_hp=g.wall_hp, temperature=g.temperature,
                  wind_x=g.wind_x, wind_y=g.wind_y, is_vacuum=g.is_vacuum,
                  flammable=g.flammable, heat=g.heat, heat_inv_shift=g.heat_inv_shift,
                  face_shift=g.face_shift, thermal_solid=g.thermal_solid,
                  cool_shift_grid=g.cool_shift, fuel_recip=g.fuel_recip,
                  fire_T_ext_plane=g.fire_T_ext_plane, gas=g.gas,
                  gas_conservative=g.gases.conservative,
                  o2_idx=int(g.gases.name_to_id["o2"]), sim_time=1.0 / 24.0)
    with pytest.raises(TypeError):
        eng.step_tail(**common)                                   # the six args missing
    narrow = np.zeros(g.temperature.shape, dtype=np.int32)
    with pytest.raises(TypeError):
        eng.step_tail(**common, heat_atten_q=g.heat_atten_q,
                      dyn_heat_atten_q=g.dyn_heat_atten_q,
                      rad_net_sweep=narrow, rad_flux_sweep=g.rad_flux_sweep,
                      rad_amb_sweep=g.rad_amb_sweep, rad_fluence=g.rad_fluence,
                      k_leak_q=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
