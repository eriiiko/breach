"""P5a -- SMOKE ABSORBS HEAT, WIRED AND DORMANT (ray-engine-v2, design v3 §6.3).

The mechanism is gated bit for bit against the integer reference in
tests/test_radiation_sweep_reference.py (gate 0) and the GPU twin in
tests/cuda_radiation_sweep_check.py. This file owns the WIRING and the
DORMANCY:

  * PhysicsEngine.step_tail takes the per-gas `heat_absorb` table as a REQUIRED,
    noconvert argument, so a stale caller cannot run a smoke-blind sweep once P5b
    ships non-zero values;
  * the live conductor hands the sweep the smoke term -- the gas planes, the table,
    the bulk sum -- so with a (test-fixture) coefficient the live planes ARE a
    direct sweep with the smoke term, and differ from one without it;
  * NOTHING consumes a gas cell's rad_net yet: the temperature fold's `ts` mask is
    unchanged (P5c opens it), so the smoke term moves no gas temperature.

(P5b's gas arm of the Fleck pre-pass has its own wiring and dormancy tests, in
tests/test_radiation_sweep_gas_fleck.py; the direct sweep below is handed the
engine's own gas currency, which that arm reads.)

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_radiation_sweep_smoke_wiring.py -q
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
from simulation import Simulation, fire_fixed, gas_fixed, unit_fixed  # noqa: E402
from simulation.gases import SMOKE  # noqa: E402
from simulation.materials import MAT_WOOD  # noqa: E402

_SWEEP_PLANES = ("rad_net_sweep", "rad_flux_sweep", "rad_amb_sweep", "rad_fluence")
H_SMOKE = 5.0            # a TEST-FIXTURE coefficient; every shipped row is 0.0


def _step_tail_kwargs(g, runner):
    """Every argument the live runner hands step_tail, as keywords."""
    return dict(ripple=g.ripple, ripple_v=g.ripple_v, water_depth=g.water_depth,
                wave_p=g.wave_p, solid=g.solid, fire=g.fire, atmosphere=g.atmosphere,
                smoke=g.smoke, wall_hp=g.wall_hp, temperature=g.temperature,
                wind_x=g.wind_x, wind_y=g.wind_y, is_vacuum=g.is_vacuum,
                flammable=g.flammable, heat=g.heat, heat_inv_shift=g.heat_inv_shift,
                face_shift=g.face_shift, thermal_solid=g.thermal_solid,
                fuel_recip=g.fuel_recip, fire_T_ext_plane=g.fire_T_ext_plane,
                gas=g.gas, gas_conservative=g.gases.conservative,
                o2_idx=int(g.gases.name_to_id["o2"]), sim_time=1.0 / 24.0,
                heat_atten_q=g.heat_atten_q, dyn_heat_atten_q=g.dyn_heat_atten_q,
                rad_net_sweep=g.rad_net_sweep, rad_flux_sweep=g.rad_flux_sweep,
                rad_amb_sweep=g.rad_amb_sweep, rad_fluence=g.rad_fluence,
                gas_heat_absorb_q16=g.gases.heat_absorb_q16,
                k_leak_q=int(runner.k_leak_q))


def test_step_tail_refuses_a_missing_mistyped_or_misshapen_heat_absorb_table():
    """PROPERTY: PhysicsEngine.step_tail REQUIRES the per-gas heat_absorb table
    (TypeError when it is missing), takes it only AS int32 -- a wider int64
    table is refused (numpy never performs that narrowing cast), and so is a
    NARROWER int16 one, which is a safe cast pybind11's convert pass would
    otherwise perform silently: `noconvert`, the house rule for every sweep
    argument -- and only with one entry per gas plane (ValueError). The correct
    table runs (non-vacuity: the refusals are about the table, not something
    else in the call).

    BREAKS IF: the argument regains a default (a caller that forgets it would run
    a smoke-blind sweep the day P5b ships non-zero values), loses noconvert (the
    int16 table goes through), or the length check is dropped.
    """
    sim = default_scenario_sim()
    g = sim.gmap
    eng = sim.physics_runner.engine
    kw = _step_tail_kwargs(g, sim.physics_runner)
    good = kw.pop("gas_heat_absorb_q16")
    with pytest.raises(TypeError):
        eng.step_tail(**kw)                                        # missing
    with pytest.raises(TypeError):
        eng.step_tail(**kw, gas_heat_absorb_q16=good.astype(np.int64))   # narrowing
    with pytest.raises(TypeError):
        eng.step_tail(**kw, gas_heat_absorb_q16=good.astype(np.int16))   # noconvert
    with pytest.raises(ValueError):
        eng.step_tail(**kw, gas_heat_absorb_q16=np.zeros(len(good) + 1, dtype=np.int32))
    eng.step_tail(**kw, gas_heat_absorb_q16=good)                  # NON-VACUITY


def _smoky_playground():
    """The playground with a wood tile HEATED to the 1263-game plateau and lit
    (CLAUDE.md "Starting a fire"), a smoke cloud in the open air around it -- in
    [0, ONE], so the fire step's tracer clamp leaves it exactly as the sweep will
    read it -- and a TEST-FIXTURE heat_absorb on smoke."""
    sim = Simulation(load_level("playground"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    ys, xs = np.where((g.material == MAT_WOOD) & g.thermal_solid)
    pick = None
    for y, x in zip(ys, xs):
        if 2 < y < g.material.shape[0] - 3 and 2 < x < g.material.shape[1] - 3:
            if not g.thermal_solid[y, x + 1] or not g.thermal_solid[y + 1, x]:
                pick = (int(y), int(x))
                break
    assert pick is not None, "the playground carries no wood tile beside air"
    y0, x0 = pick
    g.temperature[y0, x0] = 1263 << 16
    g.fire[y0, x0] = fire_fixed.quantize_scalar(0.8)
    air = (~g.thermal_solid) & (~g.solid) & (~g.is_vacuum)
    yy, xx = np.mgrid[0:g.material.shape[0], 0:g.material.shape[1]]
    cloud = air & (np.abs(yy - y0) <= 4) & (np.abs(xx - x0) <= 4)
    dens = gas_fixed.quantize(np.where((yy + xx) % 3 == 0, 0.27, 0.06))
    g.gas[SMOKE][cloud] = dens[cloud]
    assert np.any(cloud), "no open air near the fire for the smoke"
    hq = g.gases.heat_absorb_q16.copy()
    hq[SMOKE] = unit_fixed.quantize_scalar(H_SMOKE)
    g.gases.heat_absorb_q16 = np.ascontiguousarray(hq)
    return sim, pick, cloud


def test_the_live_sweep_reads_the_smoke_term():
    """PROPERTY: with smoke in the air and a non-zero heat_absorb on smoke (a test
    fixture), the planes the conductor's sweep writes at step 2b EQUAL a direct
    RadiationSweep.run on the very inputs step_tail received plus the smoke term
    (the gas planes, the table, the bulk O2 + N2 sum) -- and DIFFER from the same
    run without it, and a smoky gas cell reads a non-zero extinction there. So
    step_tail hands the sweep the whole smoke term, on the live path.

    BREAKS IF: step_tail stops passing the gas planes, the table or the bulk
    sum to the sweep; the runner stops passing GasTable.heat_absorb_q16; or the
    bulk sum the sweep reads is not the conservative-plane sum.
    """
    sim, pick, cloud = _smoky_playground()
    g = sim.gmap
    runner = sim.physics_runner
    grabbed = {}

    class _Capture:
        """Forward to the engine; snapshot the sweep's inputs at step_tail entry
        (the fire step before it reads temperature read-only and only clamps
        the in-range smoke, so these ARE step 2b's inputs)."""
        def __init__(self, engine):
            self._engine = engine

        def __getattr__(self, name):
            return getattr(self._engine, name)

        def step_tail(self, *args, **kwargs):
            grabbed["T"] = g.temperature.copy()
            grabbed["a"] = g.heat_atten_q.copy()
            grabbed["d"] = g.dyn_heat_atten_q.copy()
            grabbed["gas"] = g.gas.copy()
            grabbed["hq"] = np.asarray(kwargs["gas_heat_absorb_q16"]).copy()
            return self._engine.step_tail(*args, **kwargs)

    runner.engine = _Capture(runner.engine)
    sim.set_paused(False)
    sim.step()
    assert "T" in grabbed, "step_tail did not run"
    eng = runner.engine._engine
    assert int(grabbed["hq"][SMOKE]) == unit_fixed.quantize_scalar(H_SMOKE)
    n_bulk = sum(grabbed["gas"][gi].astype(np.int64)
                 for gi in np.flatnonzero(g.gases.conservative)).astype(np.int32)
    sweep = bp.RadiationSweep()
    amb = sweep.derive_ambient(g.is_vacuum, eng.emissive, int(runner.rad_amb_vacuum_q))
    t_amb_q = int(runner._eos_t_amb_raw())

    def direct(**gas_kw):
        out = [np.zeros(g.temperature.shape, dtype=np.int64) for _ in range(4)]
        sweep.run(grabbed["T"], grabbed["a"], grabbed["d"], g.heat_inv_shift,
                  g.thermal_solid, eng.emissive, amb, t_amb_q, int(runner.k_leak_q),
                  bp.RadiationSweep.SHEAR, 16, *out, **gas_kw)
        return out

    n_floor_q, _c_v_q, recip_cv = eng.gas_capacity_q()     # P5b: the gas arm's currency
    with_smoke = direct(gas=np.ascontiguousarray(grabbed["gas"]),
                        heat_absorb_q16=np.ascontiguousarray(grabbed["hq"]),
                        n_bulk=np.ascontiguousarray(n_bulk),
                        n_floor_q=int(n_floor_q), recip_cv=int(recip_cv))
    a_eff = sweep.a_eff_plane()
    without = direct()
    for name, want in zip(_SWEEP_PLANES, with_smoke):
        assert np.array_equal(getattr(g, name), want), (
            f"{name}: the live sweep is not the direct sweep WITH the smoke term")
    assert not np.array_equal(with_smoke[0], without[0]), (
        "the smoke term moved nothing -- the comparison above is vacuous")
    assert np.all(a_eff[cloud] > 0), "a smoky gas cell read no extinction"
    print(f"\nlive sweep == direct sweep with the smoke term; {int(cloud.sum())} smoky "
          f"cells, a_eff there {int(a_eff[cloud].min())}..{int(a_eff[cloud].max())} Q16")


def test_nothing_consumes_a_gas_cells_rad_net_yet():
    """PROPERTY (P5a's dormancy, second half): the temperature fold converts
    rad_net on THERMAL SOLIDS only -- the Pass-1 `ts[i]` mask is unchanged -- so a
    rad_net booked on a gas cell (which the smoke term now produces) moves no
    temperature anywhere: the fold with rad_net non-zero on every gas cell and
    zero on every solid equals the fold with no rad_net at all, cell for cell,
    counter for counter. And the same plane on the SOLIDS does move them
    (non-vacuity: the fold is live, only its gas branch is closed).

    WHEN THIS MUST CHANGE: P5c opens the mask to accountable gas cells (the
    Pass-1 gas radiation branch, the clamp on gas, the group-1 books) and
    replaces this with the closure identity. Opening it anywhere else, earlier
    -- P5b wires only the sweep's gas Fleck arm -- is the change this exists to
    catch.
    """
    sim = default_scenario_sim()
    g = sim.gmap
    ts = g.thermal_solid
    gas_cells = (~ts) & (~g.is_vacuum)
    assert np.any(gas_cells) and np.any(ts)
    rn_gas = np.where(gas_cells, np.int64(5_000_000), np.int64(0)).astype(np.int64)
    rn_solid = np.where(ts, np.int64(5_000_000), np.int64(0)).astype(np.int64)

    def fold(rad_net):
        solver = bp.TemperatureSolver()
        T = g.temperature.copy()
        solver.step(T, np.zeros_like(g.heat), g.heat_inv_shift, g.face_shift, g.solid,
                    g.is_vacuum, g.atmosphere, thermal_solid=ts, rad_net=rad_net)
        return T, (solver.t_max_phys_hits, solver.t_low_rail_hits,
                   solver.e_gas_deposit_sum, solver.e_solid_deposit_sum)

    base, c_base = fold(None)
    on_gas, c_gas = fold(rn_gas)
    on_solid, _c = fold(rn_solid)
    assert np.array_equal(on_gas, base) and c_gas == c_base, (
        "a gas cell's rad_net reached the fold -- the ts mask opened before P5b")
    assert not np.array_equal(on_solid, base), "the fold is dead: vacuous"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
