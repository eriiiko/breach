"""M3 (issue #12) -- combustion's fuel-bed deposit at its DERIVED value.

Brief: docs/ray_engine_v2_m3_brief_2026-09-22.md. Report:
docs/ray_engine_v2_scheme_study_2026-09-13/report_m3.md.

T5b §6.3 measured that the shipped `H_BED_M * 2^H_BED_SHIFT = 19827 * 128` was
the fuel-surface share of Huggett's heat of combustion times 2^16 -- T3 derived
the constant per UNIT of N_O2, and the engine multiplies a RAW Q16.16 burn
count by it. M3 sets it to the derived value. Three properties, each stating
the change that must break it:

1. The live solver pays, per raw count of O2 burned, the fuel-bed share of
   Huggett's constant in heat counts -- measured on the solver, after
   `o2_potency` has been folded in (the brief's §2 trap).
2. A corrected fire on a thin row SUSTAINS: it heats its own tile past its
   ignition temperature and holds it there while it burns a stated share of
   its fuel (brief §4).
3. Over a full corrected burn on a real level the Fleck damping never engages
   (M2b §5.3's post-M3 one-liner): nothing rails at T_MAX_PHYS any more.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_m3_derived_h_bed.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                                      # noqa: E402
from config import CFG                                           # noqa: E402
from level_loader import load as load_level                      # noqa: E402
from simulation import Simulation, fire_fixed                    # noqa: E402
from simulation.materials import (                               # noqa: E402
    MATERIAL_NAMES, MaterialTable, RHO_C_PIN, THERMAL_MASS_PIN,
)
from simulation.physics_runner import PhysicsRunner              # noqa: E402

import fire_timing_harness as bench                              # noqa: E402

ONE = 1 << 16
F_ONE = 1 << 24


# ===========================================================================
# 1. The solver-level H_bed is the derived fuel-bed share of Huggett
# ===========================================================================
# The long-form oracle, from cited constants only -- deliberately NOT the
# engine's collapsed expression, so the test and the dial are not the same
# number written twice.
HUGGETT_J_PER_KG_O2 = 13.1e6    # Huggett 1980 (docs/papers/): heat per kg O2 consumed
FUEL_BED_SHARE = 0.25           # Drysdale ch.5 burning-rate balance: flame-to-surface
                                # feedback is 20-40 % of total release (T5b §5)
P_ATM = 101325.0                # Pa: one unit of N is one atmosphere in one tile
R_GAS = 8.314462618             # J/(mol.K), CODATA 2018
M_O2 = 0.0319988                # kg/mol


def _derived_h_bed() -> float:
    """Heat counts per RAW count of N_O2 burned that pay the fuel-bed share.

    One unit of N_O2 is one atmosphere of O2 filling one tile at the engine's
    ambient; one raw count is 1/65536 of it. One heat count is `J_per_count`
    joules (R13's currency pin). Both are denominated on the SAME reference
    tile, so `V_tile` cancels -- it is written out anyway so every factor is
    visible.
    """
    t_amb_k = float(CFG.physics.temperature_scale.kelvin_ambient)
    tile = float(CFG.physics.thermal.tile_size_ref_m)
    v_tile = tile * tile * float(CFG.physics.water.ceiling_h)
    kg_o2_per_raw = P_ATM * v_tile / (R_GAS * t_amb_k) * M_O2 / ONE
    j_per_count = RHO_C_PIN * v_tile / (THERMAL_MASS_PIN * ONE)
    return FUEL_BED_SHARE * HUGGETT_J_PER_KG_O2 * kg_o2_per_raw / j_per_count


def _burn_once(solver, draw_r, max_claimants):
    """One combustion step on a 7x7 fixture: a burning flammable tile at the
    centre, open pure-O2 cells around it (o2f_j = 1, so the draw is large and
    `mul_q16`'s per-claimant truncation is negligible against it). Returns
    (O2 raw counts burned, heat raw counts deposited into `heat[]`)."""
    h = w = 7
    O2_, N2_, SMOKE_ = 0, 1, 2
    gas = np.zeros((3, h, w), dtype=np.int32)
    gas[O2_] = 2 * ONE                      # enriched: plenty to draw
    temperature = np.zeros((h, w), dtype=np.int32)
    wall_hp = np.full((h, w), 60 * ONE, dtype=np.int32)
    fire = np.zeros((h, w), dtype=np.int32)
    flammable = np.zeros((h, w), dtype=bool)
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    ign = np.full((h, w), 280 * ONE, dtype=np.int32)
    tsol = np.zeros((h, w), dtype=bool)
    hshift = np.zeros((h, w), dtype=np.int32)
    c = (3, 3)
    gas[:, c[0], c[1]] = 0                  # the burning tile holds no gas
    solid[c] = True
    flammable[c] = True
    tsol[c] = True
    hshift[c] = -3
    fire[c] = ONE                           # I = 1
    temperature[c] = 4000 * ONE             # hotf at its cap: a large, clean draw
    heat = np.zeros((h, w), dtype=np.int32)
    dem_acc = np.zeros((max_claimants, h, w), dtype=np.int32)
    o2_before = int(gas[O2_].astype(np.int64).sum())
    solver.step(gas, O2_, N2_, SMOKE_, temperature, wall_hp, fire, flammable,
                solid, is_vacuum, ign, 1.0 / 24.0, 0.0076849, 0.01,
                thermal_solid=tsol, heat_inv_shift=hshift, heat=heat,
                dem_acc=dem_acc, draw_r=draw_r, max_claimants=max_claimants)
    burned = o2_before - int(gas[O2_].astype(np.int64).sum())
    return burned, int(heat.astype(np.int64).sum())


def test_the_fuel_bed_deposit_pays_the_derived_share_of_huggett():
    """PROPERTY: the LIVE combustion solver -- built by PhysicsRunner from the
    shipped config, so after `o2_potency` is folded into the mantissa -- pays,
    per raw count of O2 it burns, the fuel-surface share of Huggett's heat of
    combustion, in heat counts: 0.25 * 13.1 MJ/kg * (kg O2 per raw count) /
    J_per_count, i.e. ~38.7. Measured on the solver's own `step`, as T5b §6.3
    measured the error, so the mantissa quantization, the shift and the potency
    fold are all inside the measurement.

    The tolerance is 0.5 %: the oracle's inputs (293 vs 293.15 K, rounded
    constants) move it by 0.05 %, and Huggett's own +/-5 % is a statement about
    the constant, not about this transcription. The error this test exists for
    is 2^16 = 65 536x.

    BREAKS IF: the dial is transcribed per UNIT of N_O2 again (the T5b §6.3
    error: 19827 * 2^7 = 38.73 * 2^16); `o2_potency` is set away from 1.0
    without re-deriving (it multiplies what the solver pays, and "potency" is
    a deliberate trade that must be restated here); the potency fold lands on
    the shift instead of the mantissa; or the deposit site in combustion.cpp
    changes its arithmetic.
    """
    pr = PhysicsRunner(bp)
    solver = pr.combustion
    max_claimants = int(getattr(CFG.physics.combustion, "max_claimants"))
    burned = deposited = 0
    for _ in range(8):
        b, d = _burn_once(solver, pr._draw_r, max_claimants)
        burned += b
        deposited += d
    # precision guard: mul_q16 floors each claimant's deposit, so the total is
    # short by < 1 heat count per claimant per step -- at most 8 * max_claimants
    # = 96 counts here. Against >= 10 000 raw counts burned (~3.9e5 heat counts)
    # that is < 0.03 %, far inside the tolerance. Measured: ~36 500 burned.
    assert burned > 10_000, f"the fixture burned only {burned} raw counts of O2"
    measured = deposited / burned
    derived = _derived_h_bed()
    bound = solver.H_BED_M * 2 ** solver.H_BED_SHIFT
    assert abs(measured / derived - 1.0) < 0.005, (
        f"the solver pays {measured:.4f} heat counts per raw count of O2 burned "
        f"(bound H_BED_M * 2^H_BED_SHIFT = {bound:.6g}, o2_potency = "
        f"{pr._o2_potency}); the derived fuel-bed share of Huggett is "
        f"{derived:.4f} -- ratio {measured / derived:.6g} "
        f"({np.log2(max(measured / derived, 1e-300)):+.2f} in log2)")


# ===========================================================================
# 2. The sustain property (brief §4): a corrected fire heats its own thin
#    tile past ignition and holds it there
# ===========================================================================
# The bench is fire_timing_harness at its own CLI defaults (main()): an 84x40
# planetside interior, the object at (12, 21), 0.333 m tiles, sky_tau_s 60.
_BENCH = dict(interior_w=84, interior_h=40, crate_xy=(12, 40 // 2 + 1),
              tile_size_m=0.333, tail_seconds=3.0, sky_tau_s=60.0,
              sponge_width=8)
SUSTAIN_FRACTION = 0.25          # "more than a quarter of the wall_hp bar" (§4)

_WOOD_FINDING = (
    "M3 finding, Erik's call (report_m3 §5-6): the 5 mm wood PANEL never heats "
    "past its own 300-game ignition point under the derived H_bed -- it falls "
    "from its seed on the first tick and the flame goes out at ~137 s with 18 % "
    "of its bar spent, none of it self-held. It is the one thin row with a "
    "convection face (kappa 0.12) AND emissivity 0.90; removing either alone "
    "(diagnostic only) lifts it to 31 % / 37 % consumed. This is M2b §5.2's "
    "knife edge landing on one row. strict: when P5's gas channel, #68's skin "
    "node or a fuel-bed-share ruling makes wood sustain, this XPASSes and the "
    "marker must go.")


@pytest.mark.parametrize("mat, budget_s", [
    ("kindling", 40.0),
    ("furniture", 100.0),
    pytest.param("wood", 30.0, marks=pytest.mark.xfail(strict=True, reason=_WOOD_FINDING)),
])
def test_a_corrected_fire_heats_its_own_thin_tile_past_ignition(mat, budget_s):
    """PROPERTY (brief §4): after ignition, a fire on a thin row SUSTAINS -- it
    heats its own tile to and past its ignition temperature and keeps it there
    while it burns more than a quarter of its fuel (`wall_hp` bar). Measured as
    the share of the bar spent on ticks where the tile is alight AND at or above
    its own ignition temperature, i.e. fuel burned by a self-held flame.

    WHY NOT "a quarter of the bar before intensity reaches 0" (the brief's
    first wording): measured, that cannot fail for the reason it exists. With
    the fuel-bed deposit REMOVED (H_BED_M = 0) kindling still spends 92 % of its
    bar, and furniture 70 %, before the flame goes out: the bench seeds the tile
    AT its ignition point, the fire logistic's `hot` gate stays open down to
    ignition - 200 game, and the burn timer eats the bar while the seed's heat
    bleeds off. So it measured the timer, not a flame holding its fuel bed.

    MEASURED under the derived H_bed (report_m3 §3): kindling self-held from
    8.4 s to 25.3 s, plateau 284.6 game, 46.5 % of the bar spent self-held;
    furniture 7.5-69.7 s, plateau 288.9 game, 49.8 %. At HALF the deposit both
    read 0 % -- the tile never rises past its seed. That is the knife edge of
    M2b §5.2 (a thin tile's radiative plateau at this power is ~285 game
    against ignition 280), and this test is where it stays visible. The run
    budgets cover each whole self-held window about twice over; they are run
    lengths, not asserted durations.

    BREAKS IF: the fuel-bed deposit is weakened (half the derived H_bed already
    reads 0 % on both rows -- validated, report_m3 §3); a loss channel grows
    (emission, convection, the k_leak deck leak) until the plateau falls below
    ignition; or the bench stops seeding its object at ignition.
    """
    fuel_mat = next(i for i, n in MATERIAL_NAMES.items() if n == mat)
    ign = float(MaterialTable.from_config().ignition_temp_q16[fuel_mat]) / ONE
    m = bench.run_one(0.0, max_seconds=budget_s, verbose=False,
                      fuel_mat=fuel_mat, **_BENCH)
    rec = m["rec"]
    I, T, hp = rec["I"], rec["T"], rec["hp"]
    hp0 = m["crate_hp0"]
    # non-vacuity: the object ignited, carried a real fuel bar, and the flame
    # grew off its seed (the harness's own stall criterion)
    assert hp0 > 0.0, f"{mat}: the bench object carries no fuel bar"
    assert not m["stalled"], f"{mat}: the flame never grew off its seed"
    spent = -np.diff(np.concatenate([[hp0], hp]))
    self_held = (T >= ign) & (I > 0.0)
    frac = float(spent[self_held].sum()) / hp0
    assert frac > SUSTAIN_FRACTION, (
        f"{mat}: the fire spent only {frac:.1%} of its bar while holding its own "
        f"tile at or above ignition ({ign:.0f} game) -- peak tile T "
        f"{T.max():.1f}, {int(self_held.sum())} self-held ticks of {len(T)}, "
        f"{(hp0 - hp[-1]) / hp0:.1%} spent in all")


# ===========================================================================
# 3. The Fleck damping stays inert through a full corrected burn (M2b §5.3)
# ===========================================================================
# The M2b live probe's scene, verbatim:
# docs/ray_engine_v2_scheme_study_2026-09-13/m2b_fleck_live_probe.py
PROBE_LEVEL = "fire_tuning"
PROBE_IGNITE = (46, 8)                 # (x, y): a furniture tile
FITTED_RAD_SCALE = 5.1427e-5           # the retired cast's key, the probe's control
FULL_BURN_CAP_S = 600.0


def _probe_sim(force_scale=None):
    level = load_level(PROBE_LEVEL)
    sim = Simulation(level, seed=12345, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    eng = sim.physics_runner.engine
    if force_scale is not None:
        eng.emissive.rad_scale = float(force_scale)
        eng.emissive.bake()
    ix, iy = PROBE_IGNITE
    delta = float(CFG.physics.fire.ignition_to_ext_delta)
    t_ext = int(g.fire_T_ext_plane[iy, ix]) / ONE
    g.temperature[iy, ix] = fire_fixed.quantize_scalar(t_ext + delta + 5.0)
    g.fire[iy, ix] = fire_fixed.quantize_scalar(0.12)
    return sim, g, eng


def _min_f(g, eng):
    f = np.asarray(eng.radiation.fleck_plane(), dtype=np.int64)
    mask = (g.heat_atten_q > 0) & g.thermal_solid
    return int(f[mask].min()) if mask.any() and f.size else F_ONE


def test_the_fleck_damping_stays_inert_through_a_full_corrected_burn():
    """PROPERTY (M2b §5.3): over a FULL burn on a real level, the live sweep's
    Fleck factor is exactly 1 (`f == 2^24`) on every emitting thermal solid,
    every tick. The damping only engages above a row's g = 1 knee (thin wood
    ~4868 game); a corrected fire plateaus near 290 game, so a live `f < 1`
    means something is railing again.

    The M2b probe reported min f = 0.0318 on this very scene at the tip: the
    burning tile railed at T_MAX_PHYS within 11 ticks, on the 2^16 H_bed.

    NON-VACUITY: the same instrument on the retired FITTED emission scale (the
    probe's own control) reads damping within two ticks, and the corrected run
    must actually burn -- the seeded tile grows off its seed and spends fuel --
    before it goes out.

    BREAKS IF: the fuel-bed deposit returns to a runaway (the tip: f = 0.0318);
    the sweep's E-table is baked at the fitted scale; or a thermal solid's
    capacity falls so far that its knee drops into the fire range.
    """
    # control: the instrument CAN see damping
    sim, g, eng = _probe_sim(force_scale=FITTED_RAD_SCALE)
    ctrl = F_ONE
    for _ in range(2):
        sim.set_paused(False)
        sim.step()
        ctrl = min(ctrl, _min_f(g, eng))
    assert ctrl < F_ONE, "control: the fitted scale shows no damping -- the probe is blind"

    # the corrected burn, to extinction
    sim, g, eng = _probe_sim()
    ix, iy = PROBE_IGNITE
    hp0 = int(g.wall_hp[iy, ix])
    tps = float(CFG.clock.ticks_per_second)
    worst, peak_i, peak_t, ticks = F_ONE, 0, 0.0, 0
    mask = (g.heat_atten_q > 0) & g.thermal_solid
    for k in range(int(FULL_BURN_CAP_S * tps)):
        sim.set_paused(False)
        sim.step()
        ticks = k + 1
        worst = min(worst, _min_f(g, eng))
        peak_i = max(peak_i, int(g.fire[iy, ix]))
        peak_t = max(peak_t, float(g.temperature[mask].max()) / ONE)
        if peak_i > fire_fixed.quantize_scalar(0.2) and not (g.fire > 0).any():
            break
    assert peak_i > fire_fixed.quantize_scalar(0.2), "the seeded fire never grew"
    assert int(g.wall_hp[iy, ix]) < hp0, "the fire burned no fuel"
    assert worst == F_ONE, (
        f"the Fleck damping engaged during a corrected burn: min f = "
        f"{worst / F_ONE:.6f} over {ticks} ticks (peak solid T {peak_t:.1f} game)")
