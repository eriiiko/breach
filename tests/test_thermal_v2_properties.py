"""Thermal model v2 — the properties §6 asks T5 to pin (issue #12, T5b).

Design: ``docs/thermal_model_v2_design_2026-09-19.md`` §6. Four of that list
land here; the rest have homes of their own and are named beside them:

  item 1  a flammable thermal solid has a loss channel
          -> tests/test_optics_ingress.py (the material door, T5b step 2)
  item 5b a face loses exactly what the gas gains          -> HERE
  item 5c a gas cell never ends hotter than the solid heating it  -> HERE
  item 6  a sealed room is no longer adiabatic (`k_leak`)  -> HERE
  item 7  nothing relaxes to ambient                       -> HERE
  item 8  conduction is physical (α·Δt/Δx², no stability anchor)
          -> tests/test_temperature_conduction.py (T5b step 3)
  item 9  the books close with the thermostat term gone
          -> tests/test_thermostat_books.py (T5b step 7)
  item 10 a marine burns, a zombie takes 4x
          -> tests/test_unit_heat_damage.py (the band, re-derived at T5b step 6)
  item 11 contents burn, structure resists
          -> NOT PINNED. report_t5b.md §6.3: after the flip a burning tile
             ratchets to the T_MAX_PHYS rail, so "does this ignite within the
             bench window" currently answers yes for everything hot enough to
             be in the scene. The property is real and wanted; it cannot be
             stated honestly until Erik rules on the combustion/emission
             imbalance, and writing a version that passes today would pin the
             runaway rather than the property.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_thermal_v2_properties.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import field_ab_harness as fab  # noqa: E402
from config import CFG  # noqa: E402
from simulation import Simulation, gas_fixed  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_HULL, MaterialTable)

FP = 65536
C_V = float(CFG.physics.thermal.c_v)
NO_FACE = int(CFG.physics.thermal.NO_FACE)
_TBL = MaterialTable.from_config()


def _solver():
    s = bp.TemperatureSolver()
    s.no_face = NO_FACE
    s.o2_vacuum_thresh = float(CFG.physics.thermal.o2_vacuum_thresh)
    s.c_v = C_V
    s.n_floor_heat = float(CFG.physics.thermal.n_floor_heat)
    return s


def _hot_gas_room(ticks, c_v=None):
    """field_ab_harness's canonical sealed 16x16 hull box, gas seeded +300
    game-deg above ambient, no fire and no breach — the minimal scene in which
    the ONLY thing happening is heat crossing solid|gas faces."""
    sim = Simulation(fab._scenario_level(), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    if c_v is not None:
        sim.physics_runner.engine.temperature.c_v = c_v
    interior = (~g.solid) & (~g.is_vacuum)
    g.seed_gas_temperature(interior, gas_fixed.quantize_scalar(300.0))
    sim.set_paused(False)
    for _ in range(ticks):
        sim.step()
    return sim


# --------------------------------------------------------------------------- #
# item 5b — a face loses exactly what the gas gains
# --------------------------------------------------------------------------- #
def test_5b_a_face_loses_exactly_what_the_gas_gains():
    """PROPERTY (design v2 §6 item 5b, Erik's own assertion 2026-09-19): across
    a solid|gas face the solid's loss and the gas's gain are the SAME energy,
    in ONE currency, up to the residuals the engine already counts.

    The two ledgers are denominated differently — the solid side books HEAT
    COUNTS (`ΔT · thermal_mass`), the gas side books `N · T_abs` — and `c_v` is
    the one exchange rate between them, so the identity is

        e_solid_cond_sum + c_v · e_gas_cond_sum  ==  e_cond_trunc_sum
                                                   + e_cond_cap_sum

    WHY IT MATTERS. Before T5a the gas booked `de` UNCONVERTED, which held only
    because `c_v == 1` made the two currencies numerically identical — T2
    measured `received/delivered = c_v` on a live sim, i.e. at the derived
    `c_v` 99.23 % of every joule conducted into air was destroyed at the face.
    This is that measurement as a gate.

    BREAKS IF: the conversion drops, is applied twice, or uses a second
    representation of `c_v` (`make_recip(c_v)` and `1/quantize(c_v)` differ by
    0.0724 % at the shipped value — report_t2.md §10.2).
    """
    sim = _hot_gas_room(60)
    ts = sim.physics_runner.engine.temperature
    solid = int(ts.e_solid_cond_sum)
    gas_counts = int(ts.e_gas_cond_sum) * C_V
    counted = int(ts.e_cond_trunc_sum) + int(ts.e_cond_cap_sum)

    assert solid > 0, "the walls never warmed -- this gate would be vacuous"
    assert gas_counts < 0, "the gas never cooled -- this gate would be vacuous"
    residue = (solid + gas_counts) - counted
    scale = abs(gas_counts)
    assert abs(residue) < 0.005 * scale, (
        f"the two sides of the solid|gas face do not balance: solid {solid:+d}, "
        f"gas {gas_counts:+.0f} counts, counted residuals {counted:+d}, "
        f"unexplained {residue:+.0f} ({100 * abs(residue) / scale:.3f} % of the "
        f"gas side)")
    # ...and the balance is a real cancellation, not two small numbers: the two
    # sides are each an order of magnitude larger than what is left over.
    assert abs(solid) > 5 * abs(counted)


# --------------------------------------------------------------------------- #
# item 5c — the maximum principle across a solid|gas face
# --------------------------------------------------------------------------- #
def test_5c_a_gas_cell_never_ends_hotter_than_the_solid_heating_it():
    """PROPERTY (design v2 §6 item 5c, Erik's own assertion): conduction is a
    convex combination, so a gas cell can approach the solid warming it and
    never overshoot it. Structurally guaranteed by `min(cap_i, cap_j)` — `full`
    is exactly the energy that brings the smaller-capacity side to the other's
    temperature — and the `c_v` fix had to preserve that.

    The gas is ~1000x lighter than the solid here, so it is the side the
    minimum binds on; if the capacity pairing were wrong in the direction that
    matters, the gas would overshoot on the first tick and this would catch it.

    BREAKS IF: the face quantum stops being priced at the smaller capacity, or
    the endpoint divide is taken against a different one than the quantum was
    built from.
    """
    fs = int(_TBL.face_shift_table[MAT_HULL, MAT_AIR])
    assert fs != NO_FACE, "the air|hull face is closed -- this gate is vacuous"
    T = np.zeros((1, 2), dtype=np.int32)
    T[0, 0] = 800 * FP
    heat = np.zeros((1, 2), dtype=np.int32)
    his = np.array([[int(_TBL.heat_inv_shift[MAT_HULL]), 0]], dtype=np.int32)
    face = np.full((1, 2, 4), NO_FACE, dtype=np.int32)
    face[0, 0, 2] = fs          # E of the solid
    face[0, 1, 3] = fs          # W of the gas
    solid = np.array([[True, False]], dtype=bool)
    tsm = np.array([[True, False]], dtype=bool)
    vac = np.zeros((1, 2), dtype=bool)
    atm = np.full((1, 2), FP, dtype=np.int32)
    s = _solver()
    worst = -(1 << 62)
    for _ in range(400):
        solid_before = int(T[0, 0])
        s.step(T, heat, his, face, solid, vac, atm, thermal_solid=tsm)
        worst = max(worst, int(T[0, 1]) - solid_before)
    assert worst <= 0, (
        f"a gas cell ended {worst / FP:.4f} game HOTTER than the solid that "
        f"warmed it -- the maximum principle is broken")
    # non-vacuity: the gas did warm, and the solid did cool.
    assert int(T[0, 1]) > 0, "no heat crossed the face at all"
    assert int(T[0, 0]) < 800 * FP, "the solid lost nothing"


# --------------------------------------------------------------------------- #
# item 6 — a sealed room is no longer adiabatic
# --------------------------------------------------------------------------- #
def test_6_k_leak_makes_a_sealed_room_lose_heat_out_of_plane():
    """PROPERTY (design v2 §6 item 6, R2): `k_leak` is the out-of-plane
    radiative path — the fraction of each ordinate's stream that leaves the deck
    plane per cell crossing, with the ceiling returning ambient. With it at 0 a
    sealed interior room is radiatively ADIABATIC, because every ordinate that
    leaves a cell arrives at another cell in the same plane. That is why R1
    (deleting `cool_shift`) made this dial load-bearing.

    Measured on the canonical sealed box with its walls seeded hot: the sweep's
    ambient export `|Σ rad_amb|` is **50 % larger** at the shipped `k_leak = 0.10`
    than at 0, and the room's solid books fall further.

    BREAKS IF: `k_leak` stops reaching the sweep, or the ambient channel stops
    being the thing it feeds.
    """
    def run(k_leak_q):
        sim = Simulation(fab._scenario_level(), seed=1, breach_physics=bp,
                         enable_recorder=False)
        g = sim.gmap
        sim.physics_runner.k_leak_q = k_leak_q
        wall = g.thermal_solid & (~g.is_vacuum)
        assert wall.any(), "no thermal-solid walls in the scenario"
        g.temperature[wall] = int(600.0 * FP)
        sim.set_paused(False)
        first = None
        for _ in range(60):
            sim.step()
            if first is None:
                first = int(sim.physics_runner.engine.temperature
                            .solid_energy_books_sum)
        amb = int(np.abs(g.rad_amb_sweep).astype(object).sum())
        s3 = int((g.rad_net_sweep.astype(object) + g.rad_flux_sweep.astype(object)
                  + g.rad_amb_sweep.astype(object)).sum())
        return first, int(sim.physics_runner.engine.temperature
                          .solid_energy_books_sum), amb, s3

    shipped_q = int(round(float(CFG.physics.radiation.k_leak) * FP))
    assert shipped_q > 0, "k_leak is 0 in config -- R2 is not live"
    b0_on, b1_on, amb_on, s3_on = run(shipped_q)
    b0_off, b1_off, amb_off, s3_off = run(0)

    # The sweep stays exactly conservative in both cases (gate 1, restated).
    assert s3_on == 0 and s3_off == 0, "the sweep's three-term identity broke"
    # The leak is a REAL extra loss channel.
    assert amb_on > amb_off, (
        f"k_leak changed nothing: |rad_amb| {amb_on} at the shipped dial vs "
        f"{amb_off} at 0")
    assert amb_on > 1.2 * amb_off, (
        f"k_leak's contribution is marginal: {amb_on} vs {amb_off}")
    # ...and the room ends colder for it.
    assert (b1_on - b0_on) < (b1_off - b0_off), (
        f"the room did not lose more heat with the leak on: "
        f"{b1_on - b0_on} vs {b1_off - b0_off}")


# --------------------------------------------------------------------------- #
# item 7 — nothing relaxes to ambient
# --------------------------------------------------------------------------- #
def test_7_nothing_relaxes_to_ambient():
    """PROPERTY (design v2 §6 item 7, R1): no solid's temperature changes except
    through a booked channel. An isolated hot solid — no live face, no heat
    deposit, no radiation fold — must hold its temperature EXACTLY, for ever.

    This is the sharpest possible form of "Pass 3 is gone". Before T5b step 7
    the same tile decayed by `T >> cool_shift` every tick and was at ambient
    within seconds; it now sits at 5000 game for as long as you run it.

    BREAKS IF: a `cool_shift` remnant survives anywhere in the pass, or any new
    unbooked relaxation is added.
    """
    T = np.zeros((1, 3), dtype=np.int32)
    T[0, 1] = 5000 * FP
    start = int(T[0, 1])
    heat = np.zeros((1, 3), dtype=np.int32)
    his = np.full((1, 3), 3, dtype=np.int32)
    face = np.full((1, 3, 4), NO_FACE, dtype=np.int32)
    solid = np.ones((1, 3), dtype=bool)
    vac = np.zeros((1, 3), dtype=bool)
    atm = np.full((1, 3), FP, dtype=np.int32)
    s = _solver()
    for _ in range(200):
        s.step(T, heat, his, face, solid, vac, atm)
    assert int(T[0, 1]) == start, (
        f"an isolated hot solid relaxed from {start} to {int(T[0, 1])} with no "
        f"booked channel -- something still decays toward ambient")
    # The same, with a VACUUM-FACING neighbour: `cool_shift_vacuum` used to make
    # this the FAST case, so it is the one a remnant would show up in first.
    T2 = np.zeros((1, 3), dtype=np.int32)
    T2[0, 1] = 5000 * FP
    vac2 = np.array([[True, False, True]], dtype=bool)
    s2 = _solver()
    for _ in range(200):
        s2.step(T2, heat, his, face, solid, vac2, atm)
    assert int(T2[0, 1]) == start, (
        f"a vacuum-facing hot solid relaxed from {start} to {int(T2[0, 1])} -- "
        f"the cool_shift_vacuum path is still alive")
    # And the two counters that priced it are gone from the surface entirely.
    assert not hasattr(s, "e_cool_sum")
    assert not hasattr(s, "e_thermostat_sum")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
