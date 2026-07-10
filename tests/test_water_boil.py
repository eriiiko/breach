"""Flash-boil vacuum sink + steam puff (Step W5 of docs/water_implementation_plan.md).

Inside ``_step_water``, after the pipe-model substeps and BEFORE the W3
displacement accounting (so a boiled-off column reads as receding water ->
slight decompression, the physically-right sign), standing water under low
pressure boils off and puffs ``white_smoke`` (steam):

    boiling = (atmosphere < boil_p_thresh) & (water_depth > 0)
    boiled  = where(boiling, min(water_depth, boil_rate * sim_time), 0)
    water_depth        -= boiled
    gas[white_smoke]   += steam_yield * boiled

PRESSURE-keyed, not vacuum-keyed: a drained-but-sealed room boils too. The
fire-side evaporative heat sink (wet tiles staying cool) is the temperature
solver's lane and is NOT exercised here.

The plan's four tests, with the plan's numbers (boil_rate 0.02 m/s,
boil_p_thresh 0.3, steam_yield 4.0, tick 1/24 s):
  1. single wet tile held at atmosphere 0.0, depth 0.1 m: after ONE tick
     depth == 0.1 - 0.02/24 (atol 1e-7) and the white_smoke gain
     == 4.0 * 0.02/24 (atol 1e-6).
  2. after 121 ticks the tile has boiled DRY: depth == 0.0 exactly
     (120 full-rate increments empty 0.1 m in exact arithmetic; float32
     rounding can leave a sub-depth_eps residue on tick 120, which the
     solver's dry snap kills on tick 121 — so 121 is the robust pin).
  3. a TWIN tile at atmosphere 1.0 in the same run is bit-exact unchanged
     (depth AND gas).
  4. dormancy: no water -> no gas writes (gas field bit-identical).

SCENE / VACUUM-HOLD NOTE: each tile is a sealed single-cell air column (the
test_water_displacement scene), and the boiling cell's atmosphere is painted
0.0 ONCE, directly, while paused — no ``is_vacuum``, no per-tick re-clamping.
That holds bit-exactly for the whole run because every atmosphere mover (wave
Laplacian, IMEX diffusion, wind gradient) gathers faces as
``min(perm[self], perm[n])`` and all four neighbours are hull, so nothing can
ever refill the cell; the W3 displacement only MULTIPLIES atmosphere
(0 * ratio == 0). ``is_vacuum`` is deliberately NOT used: it flips the cell to
breach/space semantics in the atmosphere & smoke solvers (drain toward
space), and the steam puff must land in ordinary interior air. The sealed
cell also makes the pipe-model substeps a no-op on the water (nowhere to
flow) and gas transport a no-op on the puff (no open face, no wind, and a
breach-less map has an all-zero smoke sink field).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_water_boil.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.gases import N_TRACE_GASES, WHITE_SMOKE  # noqa: E402
from water_q16 import q, deq  # noqa: E402  (S1: Q16.16 quantize/dequantize)
from simulation import gas_fixed  # noqa: E402  (S2b: gas Q16.16 dequantize)


def _gas_mass(plane):
    """Total real-density mass of a gas plane (S2b: int32 Q16.16 -> /65536)."""
    return float(plane.astype(np.float64).sum()) / gas_fixed.FP_ONE_F

# S1: water_depth is int32 Q16.16 (~1.5e-5 m granularity). The W5 boil is a
# FLOAT BRIDGE (it dequantizes depth, boils in float, re-quantizes the removed
# metres), so the per-tick depth carries one Q16.16 round-trip per tick. Depth
# assertions are in metres (deq) with a tolerance widened to a few Q16.16 LSB
# where the old float test used 1e-7. The STEAM puff is still float (it is read
# from the gas field directly).
Q_EPS = 1.0 / 65536.0   # one Q16.16 LSB in metres

SEED = 42

A = (2, 2)   # the vacuum-held boiling cell
B = (2, 4)   # the twin control cell at full pressure

DEPTH = 0.1            # m painted on both cells (the plan's number)
BOIL_RATE = 0.02       # m/s — the plan pins these three against the bound
BOIL_P_THRESH = 0.3    # config values in _assert_plan_params below
STEAM_YIELD = 4.0
TICK = 1.0 / 24.0      # s (guarded against CFG.clock below)
INC = BOIL_RATE * TICK     # per-tick boil-off at full rate: 0.02/24 m


# ---------------------------------------------------------------------------
# helpers (synthetic-LevelData scene, the test_water_displacement pattern;
# CSV codes: 1 = hull, 4 = interior air)
# ---------------------------------------------------------------------------
def _twin_cell_level(tile_size_m: float = 0.333) -> LevelData:
    """A 5x7 all-hull block with TWO open interior cells, A=(2,2) and
    B=(2,4), separated by hull — two independent sealed single-cell air
    columns in one map. Water painted on them cannot flow during the
    substeps and the steam puff cannot leave A: depth and gas changes come
    from the W5 boil alone."""
    tm = np.ones((5, 7), dtype=np.int32)
    tm[A] = 4
    tm[B] = 4
    return LevelData(
        name="water_boil_test",
        version="1",
        path=Path("."),
        tilemap=tm,
        tile_size_m=tile_size_m,
        diffuse_path=Path("."),
    )


def _assert_plan_params(runner):
    """The plan's W5 numbers are the config's — pin the binding explicitly so
    a config drift fails HERE with a message, not as a mystery delta."""
    assert abs(runner.water_boil_rate - BOIL_RATE) < 1e-12, (
        "config [physics.water] boil_rate drifted from the plan's 0.02")
    assert abs(runner.water_boil_p_thresh - BOIL_P_THRESH) < 1e-12, (
        "config [physics.water] boil_p_thresh drifted from the plan's 0.3")
    assert abs(runner.water_steam_yield - STEAM_YIELD) < 1e-12, (
        "config [physics.water] steam_yield drifted from the plan's 4.0")
    assert int(CFG.clock.ticks_per_second) == 24, (
        "the plan's tick = 1/24 s premise no longer holds")


def _boil_sim(depth: float = DEPTH):
    """Twin-cell scene: A vacuum-held (atmosphere painted 0.0 once — see the
    module docstring's vacuum-hold note), B at the interior default 1.0;
    ``depth`` metres of water painted DIRECTLY on both while paused (read by
    the W2a first-call seed as pre-existing — no tick-1 compression spike).
    ``depth=0.0`` paints none (the dormancy scene, pressure side still armed).
    """
    level = _twin_cell_level()
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    for (y, x) in (A, B):
        assert not g.solid[y, x]
        assert (g.solid[y - 1, x] and g.solid[y + 1, x]
                and g.solid[y, x - 1] and g.solid[y, x + 1]), (
            "each column must be wall-sealed on all four sides")
    assert not g.is_vacuum.any(), "the hold must not rely on breach semantics"
    assert float(g.atmosphere[B]) > 0.5, "twin cell has no air (vacuous twin)"
    # EOS P3: `atmosphere` (P) is solver-materialized from (N,T) every tick —
    # the old one-paint P hold would be overwritten on tick 1. The equivalent
    # hold on the REAL state: zero the bulk O2/N2 at A (one paint). A is a
    # wall-sealed 1-cell pocket, so no bulk flux ever refills it and
    # p* = C*N*T stays ~0 there for the whole run (the near-vacuum row
    # degeneracy pins P_A to p*_A each solve). The P_prev seed below is
    # cosmetic (the solver refreshes it).
    from simulation.gases import O2 as _O2, INERT_N2 as _N2
    g.gas[_O2][A] = 0
    g.gas[_N2][A] = 0
    g.atmosphere[A] = 0.0          # P_prev seed only (solver-owned now)
    if depth:
        g.water_depth[A] = q(depth)   # S1: paint in Q16.16 metres
        g.water_depth[B] = q(depth)
    sim.set_paused(False)
    return sim, g


# ---------------------------------------------------------------------------
# 1. One tick: the boil-off depth decrement and the steam puff, exactly
# ---------------------------------------------------------------------------
def test_one_tick_vacuum_boil_depth_and_steam_gain():
    sim, g = _boil_sim()
    _assert_plan_params(sim.physics_runner)

    # The steam GAIN is read as TOTAL white_smoke mass (float64 sum) — robust
    # against transport moving the puff between ticks (here transport is a
    # no-op anyway: sealed cell, zero wind, breach-less map -> zero sink).
    total0 = _gas_mass(g.gas[WHITE_SMOKE])    # S2b: dequantized real-density mass
    assert total0 == 0.0, "scene starts with stray white_smoke"

    sim.step()                                    # ONE tick = 1/24 s

    # Depth: one full-rate increment boiled off the vacuum cell. S1: the depth
    # is Q16.16 and the boil round-trips through float once, so allow a few LSB
    # of quantization slack (was 1e-7 on the pure-float build).
    d = float(deq(g.water_depth[A]))
    assert abs(d - (DEPTH - INC)) < 3 * Q_EPS, (
        f"one-tick boil depth {d} != {DEPTH - INC} (tol {3*Q_EPS:.2e})")
    # Steam: the puff is steam_yield * boiled, in white_smoke and ONLY there.
    # S1: boiled is the Q16.16-quantized full-rate amount. S2b: the steam puff
    # ITSELF is now Q16.16 too (the gas plane is integer), so the gain carries a
    # SECOND quantization (the puff quantize on top of the boil quantize) — widen
    # the slack to a few gas LSB (~1.5e-5 each) on top of the boil LSB.
    gain = _gas_mass(g.gas[WHITE_SMOKE]) - total0
    assert abs(gain - STEAM_YIELD * INC) < STEAM_YIELD * 3 * Q_EPS + 3 * Q_EPS, (
        f"one-tick steam gain {gain} != {STEAM_YIELD * INC} "
        f"(tol {STEAM_YIELD * 3 * Q_EPS + 3 * Q_EPS:.2e})")
    # Trace slices only (0..N_TRACE_GASES-1) — the bulk O2/inert_N2 pair (EOS
    # refactor P1) always carries ambient air, unrelated to the steam puff.
    for gi in range(N_TRACE_GASES):
        if gi != WHITE_SMOKE:
            assert not g.gas[gi].any(), (
                f"boil leaked into gas slice {gi} (steam is white_smoke only)")
    # The vacuum hold held (and the cell keeps boiling next tick). EOS P3:
    # the steam puff itself now carries trace MASS (trace_mass_scale), so
    # P_A is epsilon-positive rather than exactly 0 — the boil gate only
    # needs it below boil_p_thresh.
    from simulation import atmosphere_fixed as _afx
    assert float(g.atmosphere[A]) < _afx.quantize_scalar(
        sim.physics_runner.water_boil_p_thresh)


# ---------------------------------------------------------------------------
# 2. 121 ticks: the tile boils DRY — depth exactly 0.0
# ---------------------------------------------------------------------------
def test_boils_dry_exact_zero_after_121_ticks():
    sim, g = _boil_sim()

    for _ in range(119):
        sim.step()
    # Non-vacuity: ~one increment left — still wet, still boiling at 119.
    assert float(g.water_depth[A]) > 0.0, (
        "boiled dry too early — the 121-tick pin is measuring nothing")

    sim.step()                                    # 120: the last increment
    sim.step()                                    # 121: dry-snap any residue
    assert float(g.water_depth[A]) == 0.0, (
        f"depth after 121 ticks is {g.water_depth[A]!r}, not exactly 0.0")

    # Yield bookkeeping (the plan's numbers end-to-end): ALL the water left
    # as steam, total white_smoke starts at steam_yield * 0.1 = 0.4 as it is
    # puffed in across the 121 ticks. EOS refactor P4 (design §2.2/§5 v2.1,
    # decisions.md #12): the per-gas trace `decay` column is now APPLIED
    # (white_smoke decay=0.020/s, config.toml [gases.white_smoke] — "loaded
    # but never applied" pre-P4), crediting the lost mass to inert_N2 each
    # tick, so the puffed-in steam settles/condenses as it accumulates —
    # the measured total is ~4-5% below the undecayed 0.4 over this run
    # (steam puffed early has more ticks to decay than steam puffed late).
    # Widened from the pre-P4 0.1% float-rounding tolerance to comfortably
    # cover the REAL, expected decay loss (not a regression — it is EXACTLY
    # decision #12 v2.1's "burnt/settled products go to inert_N2" behaviour).
    total = _gas_mass(g.gas[WHITE_SMOKE])     # S2b: dequantized real-density mass
    assert 0.0 < total < STEAM_YIELD * DEPTH, (
        f"steam total {total} should be positive and BELOW the undecayed "
        f"steam_yield*depth {STEAM_YIELD * DEPTH} (decay only ever removes "
        f"mass from this plane)")
    assert abs(total - STEAM_YIELD * DEPTH) < 0.10 * STEAM_YIELD * DEPTH, (
        f"steam total {total} strayed too far from steam_yield*depth "
        f"{STEAM_YIELD * DEPTH} for the known white_smoke decay rate "
        f"(0.020/s over ~121 ticks) — investigate")


# ---------------------------------------------------------------------------
# 3. The twin tile at atmosphere 1.0: bit-exact unchanged (depth AND gas)
# ---------------------------------------------------------------------------
def test_twin_tile_at_full_pressure_bit_exact_unchanged():
    sim, g = _boil_sim()
    depth_pre = g.water_depth.copy()
    gas_pre = g.gas.copy()

    for _ in range(121):                          # the full boil-dry run
        sim.step()

    # Control (the asserts bite): the vacuum twin DID boil in this same run.
    assert float(g.water_depth[A]) < float(depth_pre[A])
    assert _gas_mass(g.gas[WHITE_SMOKE]) > 0.0

    # The full-pressure twin: depth bit-exact, every gas slice bit-exact.
    assert g.water_depth[B] == depth_pre[B], (
        f"twin depth changed: {depth_pre[B]!r} -> {g.water_depth[B]!r}")
    by, bx = B
    assert np.array_equal(g.gas[:, by, bx], gas_pre[:, by, bx]), (
        "twin tile gas changed — boil wrote where atmosphere >= boil_p_thresh")


# ---------------------------------------------------------------------------
# 4. Dormancy: no water -> no gas writes
# ---------------------------------------------------------------------------
def test_dormancy_no_water_no_gas_writes():
    # Pressure side of the boil mask ARMED (cell A held below boil_p_thresh)
    # but NO water anywhere: the only reason no steam appears is the absence
    # of water. With every TRACE gas slice zero the transport loop .any()-skips
    # them all, so ANY trace-gas change must be a spurious W5 write (e.g. the
    # minimum(where=...)-garbage trap the plan calls out). The bulk O2/inert_N2
    # pair (EOS refactor P1) is NOT zero — both cells are fully wall-sealed on
    # all 4 sides (see _boil_sim's assert), so their conservative transport is
    # a structural no-op regardless; the SECOND assertion below (whole-array
    # equality across the 3 ticks) is what actually pins "no writes at all".
    sim, g = _boil_sim(depth=0.0)
    assert float(g.atmosphere[A]) < sim.physics_runner.water_boil_p_thresh
    assert not g.water_depth.any()
    gas_pre = g.gas.copy()
    assert not gas_pre[:N_TRACE_GASES].any()

    for _ in range(3):
        sim.step()

    assert np.array_equal(g.gas, gas_pre), (
        "dry map wrote gas — W5 must be a no-op without water")
    assert not g.water_depth.any()
    assert not g.flow_vx.any() and not g.flow_vy.any()
