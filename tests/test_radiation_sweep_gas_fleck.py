"""P5b -- THE GAS ARM OF THE FLECK PRE-PASS: ITS CURRENCY, ITS WIRING, ITS DORMANCY
(ray-engine-v2, design v3 §2.8 / §6.3).

The arm's ARITHMETIC is held bit for bit to the integer reference by gate 0
(tests/test_radiation_sweep_reference.py) and to the CPU by the CUDA gate
(tests/cuda_radiation_sweep_check.py P10h/P10i); the PROPERTY it exists for --
a hot absorbing gas cell cools monotonically and never below ambient, where the
L = 0 arm P5a shipped overshoots in one step exactly where g > 4T/T_abs -- is
the reference's gate 14 (tests/test_ray_engine_v2_integer_reference.py). This
file owns the three things neither can see:

  * THE CURRENCY IS THE FOLD'S. The arm prices a gas cell's L in N * c_v, through
    the same N floor, per-cell reciprocal and c_v reciprocal the temperature
    fold's Pass-1 gas deposit divides by. PhysicsEngine.gas_capacity_q() derives
    them from the fold's OWN dials (temperature.c_v / .n_floor_heat); here the
    fold's own gas deposit is predicted, bit for bit, from those three integers
    -- at the shipped dials and at another pair -- so the arm and the fold
    cannot be pricing a gas cell in two currencies.
  * THE LIVE PATH HANDS IT OVER: the conductor's sweep (step 2b of step_tail)
    damps HOT smoke on the live table in exactly that currency, and follows the
    fold's dials when they move.
  * WHERE IT ENGAGES (P5c; P5b's dormancy gate, retired as its docstring
    said it would be): smoke now absorbs in the SHIPPED table, so the live game
    damps hot smoky gas cells -- and only those: a gas cell whose smoke term is
    zero stays at 2^24.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_radiation_sweep_gas_fleck.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from _radiation_sweep_harness import F_ONE, ONE, R  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation, fire_fixed, gas_fixed, unit_fixed  # noqa: E402
from simulation.gases import SMOKE  # noqa: E402
from simulation.materials import MAT_WOOD  # noqa: E402

NO_FACE = 63                     # [physics.thermal] NO_FACE: no conduction face
H_SMOKE = 5.0                    # a TEST-FIXTURE coefficient (the shipped smoke is 25.36, P5c)
_SWEEP_PLANES = ("rad_net_sweep", "rad_flux_sweep", "rad_amb_sweep", "rad_fluence")


def _engine():
    """A live engine, its temperature solver's dials bound from config exactly as
    PhysicsRunner binds them."""
    sim = Simulation(load_level("playground"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    return sim.physics_runner.engine


def _fold_gas_dT(solver, deposits, n_bulk):
    """The temperature fold's OWN gas deposit, measured: one direct
    TemperatureSolver.step on all-gas cells with no conduction faces, no vacuum
    and no ring, so each cell's final temperature is exactly Pass 1's gas dT
    (the non-energy branch: temperature += dT)."""
    h, w = deposits.shape
    T = np.zeros((h, w), dtype=np.int32)
    solver.step(T, np.ascontiguousarray(deposits.astype(np.int32)),
                np.zeros((h, w), dtype=np.int32),
                np.full((h, w, 4), NO_FACE, dtype=np.int32),
                np.zeros((h, w), dtype=bool), np.zeros((h, w), dtype=bool),
                np.full((h, w), ONE, dtype=np.int32),
                n_bulk=np.ascontiguousarray(n_bulk.astype(np.int32)),
                thermal_solid=np.zeros((h, w), dtype=bool))
    return T


def _predicted_dT(deposits, n_bulk, n_floor_q, recip_cv):
    """What the fold's gas deposit IS, in the currency gas_capacity_q hands the
    sweep: e_abs = deposit (N >= ONE) or its density share mul_q16(deposit, N),
    dT = deposit_dT_wide_q16(e_abs, reciprocal_q16(max(N, n_floor_q)), recip_cv),
    every factor through the kit's own bound functions."""
    out = np.zeros(deposits.shape, dtype=np.int64)
    for (y, x), dep in np.ndenumerate(deposits):
        n = int(n_bulk[y, x])
        e_abs = int(dep) if n >= ONE else (int(dep) * n) >> 16
        rn = bp.fp_reciprocal_q16(max(n, n_floor_q))
        out[y, x] = bp.fp_deposit_dT_wide_q16(e_abs, rn, recip_cv)
    return out


def test_the_gas_arm_is_priced_in_the_folds_own_currency():
    """PROPERTY: PhysicsEngine.gas_capacity_q() -- the (n_floor_q, c_v_q,
    recip_cv) step_tail hands the sweep's gas Fleck arm -- is the temperature
    fold's OWN gas currency: at the shipped dials it is the integer reference's
    (N_FLOOR_Q_LIVE = 655, C_V_Q_LIVE = 504, make_recip(504 / 65536)), and the
    fold's own Pass-1 gas deposit, measured on the engine's own solver, is
    predicted BIT FOR BIT from those three integers -- for bulk counts above
    ambient, at it, and below the n_floor_heat floor (where the floor, not N, is
    the capacity) -- at the shipped dials AND after both dials move (c_v 0.02,
    n_floor_heat 0.05). So the arm and the fold price a gas cell in one
    currency, and where they meet is that accessor: the SAME quantize /
    make_recip on the SAME two dials the fold's step() reads.

    BREAKS IF: gas_capacity_q derives any of the three differently from the
    fold (a second quantization of c_v instead of its one integer form, the
    floor dropped or taken from another dial, a make_recip on the raw c_v), or
    reads a dial the fold does not; or the fold's own derivation moves without
    the accessor following -- the moved-dials half catches either direction.
    """
    eng = _engine()
    nf, cvq, rcv = eng.gas_capacity_q()
    assert (nf, cvq, rcv) == (R.N_FLOOR_Q_LIVE, R.C_V_Q_LIVE,
                              R.make_recip(R.C_V_Q_LIVE / 65536.0)), (nf, cvq, rcv)
    deposits = np.array([[65536, 1 << 20, 1234567, 999, 7 << 16, 3],
                         [65536, 1 << 20, 1234567, 999, 7 << 16, 3],
                         [65536, 1 << 20, 1234567, 999, 7 << 16, 3]], dtype=np.int64)
    n_bulk = np.array([[3 * ONE] * 6,
                       [ONE] * 6,
                       [200, 400, 100, 50, 300, 1]], dtype=np.int64)
    for dials in ((None, None), (0.02, 0.05)):
        if dials[0] is not None:
            eng.temperature.c_v, eng.temperature.n_floor_heat = dials
            nf, cvq, rcv = eng.gas_capacity_q()
            assert (nf, cvq) == (R.quant(dials[1]), R.quant(dials[0]))
        got = _fold_gas_dT(eng.temperature, deposits, n_bulk)
        want = _predicted_dT(deposits, n_bulk, nf, rcv)
        assert np.array_equal(got.astype(np.int64), want), (dials, got, want)
        # non-vacuity: the floor binds on the third row and the currency is
        # load-bearing -- a neighbouring recip_cv predicts a different fold
        assert np.all(n_bulk[2] < nf)
        assert not np.array_equal(_predicted_dT(deposits, n_bulk, nf, rcv + rcv // 64),
                                  want)


def _hot_smoky_playground(fixture=True):
    """The playground with a burning wood tile (heat delivered: CLAUDE.md
    'Starting a fire'), a smoke cloud around it SEEDED HOT through the gas-energy
    seam (gamemap.seed_gas_temperature: 'Gas temperature is a mirror'), and --
    with `fixture` -- a TEST-FIXTURE heat_absorb on smoke. At 3000 game soot at
    ambient density is past g = 1 on the live table (damped from 1676 game), so
    the arm engages. Without the fixture the gas table is the SHIPPED one."""
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
    g.temperature[y0, x0] = 1263 << 16                  # a thermal solid: its own truth
    g.fire[y0, x0] = fire_fixed.quantize_scalar(0.8)
    air = (~g.thermal_solid) & (~g.solid) & (~g.is_vacuum)
    yy, xx = np.mgrid[0:g.material.shape[0], 0:g.material.shape[1]]
    cloud = air & (np.abs(yy - y0) <= 4) & (np.abs(xx - x0) <= 4)
    assert np.any(cloud), "no open air near the fire for the smoke"
    dens = gas_fixed.quantize(np.where((yy + xx) % 3 == 0, 0.27, 0.06))
    g.gas[SMOKE][cloud] = dens[cloud]
    g.seed_gas_temperature(cloud, 3000 << 16)
    if fixture:
        hq = g.gases.heat_absorb_q16.copy()
        hq[SMOKE] = unit_fixed.quantize_scalar(H_SMOKE)
        g.gases.heat_absorb_q16 = np.ascontiguousarray(hq)
    return sim, cloud


def _step_capturing(sim):
    """Step once; snapshot step 2b's inputs at step_tail entry (the fire step
    reads temperature read-only and only clamps the in-range smoke, so these ARE
    the sweep's inputs)."""
    g = sim.gmap
    runner = sim.physics_runner
    grabbed = {}
    engine = runner.engine

    class _Capture:
        def __getattr__(self, name):
            return getattr(engine, name)

        def step_tail(self, *args, **kwargs):
            grabbed["T"] = g.temperature.copy()
            grabbed["a"] = g.heat_atten_q.copy()
            grabbed["d"] = g.dyn_heat_atten_q.copy()
            grabbed["gas"] = g.gas.copy()
            grabbed["hq"] = np.asarray(kwargs["gas_heat_absorb_q16"]).copy()
            return engine.step_tail(*args, **kwargs)

    runner.engine = _Capture()
    try:
        sim.set_paused(False)
        sim.step()
    finally:
        runner.engine = engine
    assert "T" in grabbed, "step_tail did not run"
    return grabbed


def _direct(sim, grabbed, n_floor_q, recip_cv):
    """RadiationSweep.run on step 2b's captured inputs, the smoke term and the
    given gas currency: returns (four planes, Fleck plane)."""
    g = sim.gmap
    runner = sim.physics_runner
    eng = runner.engine
    n_bulk = sum(grabbed["gas"][gi].astype(np.int64)
                 for gi in np.flatnonzero(g.gases.conservative)).astype(np.int32)
    sweep = bp.RadiationSweep()
    amb = sweep.derive_ambient(g.is_vacuum, eng.emissive, int(runner.rad_amb_vacuum_q))
    out = [np.zeros(g.temperature.shape, dtype=np.int64) for _ in range(4)]
    sweep.run(grabbed["T"], grabbed["a"], grabbed["d"], g.heat_inv_shift, g.thermal_solid,
              eng.emissive, amb, int(runner._eos_t_amb_raw()), int(runner.k_leak_q),
              bp.RadiationSweep.SHEAR, 16, *out,
              gas=np.ascontiguousarray(grabbed["gas"]),
              heat_absorb_q16=np.ascontiguousarray(grabbed["hq"]),
              n_bulk=np.ascontiguousarray(n_bulk),
              n_floor_q=int(n_floor_q), recip_cv=int(recip_cv))
    return out, np.asarray(sweep.fleck_plane())


def test_the_live_sweep_damps_hot_smoke_in_the_folds_currency():
    """PROPERTY: on the live path -- the conductor's step 2b, the live table --
    HOT absorbing smoke is DAMPED (f < 2^24 on gas cells of the engine's own
    Fleck plane), and the four planes and the Fleck plane the conductor wrote
    EQUAL a direct RadiationSweep.run on the very inputs step_tail received,
    priced in the engine's gas_capacity_q -- and differ from the same run in a
    neighbouring currency. When the fold's dials move (c_v 0.002, n_floor_heat
    2.0: a floor above every damped cell's bulk count, and a capacity twice as
    stiff as the shipped one at N = 1), the next tick's live sweep follows them:
    equal to the direct run in the NEW currency, different from the old one. So
    step_tail hands the arm the fold's currency, read afresh from the fold's
    dials every tick.

    BREAKS IF: step_tail stops passing the currency (the sweep then refuses the
    gas group), passes a constant or stale one, derives it from anything but
    this->temperature's dials, or the sweep's gas arm is disconnected (no gas
    cell damped).
    """
    sim, cloud = _hot_smoky_playground()
    g = sim.gmap
    eng = sim.physics_runner.engine
    for dials in ((None, None), (0.002, 2.0)):
        if dials[0] is not None:
            eng.temperature.c_v, eng.temperature.n_floor_heat = dials
        nf, _cvq, rcv = eng.gas_capacity_q()
        grabbed = _step_capturing(sim)
        live_f = np.asarray(eng.radiation.fleck_plane())
        want, want_f = _direct(sim, grabbed, nf, rcv)
        for name, plane in zip(_SWEEP_PLANES, want):
            assert np.array_equal(getattr(g, name), plane), (dials, name)
        assert np.array_equal(live_f, want_f), dials
        gas = ~g.thermal_solid
        damped = (live_f < F_ONE) & gas
        n_damped = int(np.count_nonzero(damped))
        assert n_damped > 0, f"no gas cell damped on the live path at {dials}"
        _other, other_f = _direct(sim, grabbed, nf, rcv // 2)
        assert not np.array_equal(other_f, live_f), (
            f"the currency moved nothing at {dials}: the comparison is vacuous")
        if dials[0] is not None:
            # the moved floor is above every bulk count the damped cells hold,
            # so n_floor_q -- not N -- is their capacity: the floor is load-bearing
            n_bulk = sum(grabbed["gas"][gi].astype(np.int64)
                         for gi in np.flatnonzero(g.gases.conservative))
            assert np.all(n_bulk[damped] < nf), (int(n_bulk[damped].max()), nf)
        print(f"\ndials {dials}: currency {(nf, rcv)}, {n_damped} damped gas cells, "
              f"min f {int(live_f[gas].min())}")


def test_the_shipped_game_damps_exactly_the_smoky_gas_cells():
    """PROPERTY (P5c): with the SHIPPED gas table -- smoke absorbs, derived in
    config.toml; every other gas is 0.0 -- the live game's gas arm engages on
    exactly the gas cells whose smoke term absorbs: over live ticks of a burning
    playground with a hot smoke cloud, every DAMPED gas cell (f < 2^24 in the
    engine's own Fleck plane) held smoke with a bulk count at or above N_EPS_RAW
    at the sweep's input, and every gas cell with no smoke stayed at 2^24 -- while
    some gas cells ARE damped (non-vacuity; before P5c this gate asserted none
    were, which its docstring named P5c to retire).

    BREAKS IF: the arm reaches a gas cell whose smoke term is zero (a_gas == 0)
    -- e.g. by keying on a_eff with a material extinction on gas, or on the bulk
    count alone -- or the shipped smoke stops absorbing (nothing damped).
    """
    sim, cloud = _hot_smoky_playground(fixture=False)     # the SHIPPED gas table
    g = sim.gmap
    assert g.gases.heat_absorb_q16[SMOKE] > 0, "the shipped smoke absorbs nothing"
    eng = sim.physics_runner.engine
    n_damped = 0
    for _ in range(3):
        grabbed = _step_capturing(sim)
        fl = np.asarray(eng.radiation.fleck_plane())
        gas = ~g.thermal_solid
        smoke_in = grabbed["gas"][SMOKE] > 0
        n_bulk = sum(grabbed["gas"][gi].astype(np.int64)
                     for gi in np.flatnonzero(g.gases.conservative))
        damped = (fl < F_ONE) & gas
        n_damped += int(np.count_nonzero(damped))
        assert np.all(smoke_in[damped] & (n_bulk[damped] >= 1)), (
            "a gas cell with no absorbing smoke was damped")
        assert np.all(fl[gas & ~smoke_in] == F_ONE)
    assert n_damped > 0, "no gas cell damped in the shipped game: vacuous"
    assert np.any(g.rad_net_sweep != 0), "no radiation flowed: vacuous"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
