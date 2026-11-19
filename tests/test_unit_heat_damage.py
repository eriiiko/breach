"""Unit heat damage (engine/06 §4, proposal §4.2/§4.3) — STEP D.

The `Phi_rad`-only radiant-heat consumer of the per-tick `heat` deposit. A
LIVING unit, stamped as a full ray-blocker before the ray pass, samples the
already-occluded `heat` buffer at its footprint; if the felt temperature pushes
past its tolerance band it loses HP this tick. This module exercises the
consumer directly (heat injected into ``gmap.heat`` — no renderer, no ray pass)
and through a full ``Simulation.step()`` to assert the heat-clear ordering.

#215 (docs/body_temperature_design_2026-11-17.md section 2): the felt
temperature is now the FLOW into the unit's own body temperature
(simulation.body_temp), so the per-tick law below holds in the STEADY STATE:
a held flux in still air at one atmosphere and 20 C settles the body at the
felt temperature, where the row bills the old amount to a count
(``_settle``). The burst, the split and the vacuum are
tests/test_body_temp.py's.

Verifies:
  - a unit on a HOT tile loses HP, scaled by absorption * (1 - reflectivity)
    and the over-temperature ramp (once its body has settled);
  - a unit on a COLD tile takes EXACTLY zero;
  - a ZOMBIE takes ``zombie.fire_damage_multiplier`` (4x) more than a marine
    for the same flux;
  - damage SCALES with over-temperature (stronger beam -> more than linear-in-
    flux extra, via the k_over ramp);
  - a heat DEATH sets source/killed_by == "heat" and does NOT set
    killed_by_zombie (no conversion of burned corpses);
  - damage is TICK-RATE INDEPENDENT (same real DPS at 12 vs 24 tps);
  - determinism: fixed unit order -> identical result;
  - end-to-end: after a full Simulation.step() `heat` is 0 (the clear moved to
    AFTER the unit-damage consumer), and the unit-damage saw the PRE-clear value.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_unit_heat_damage.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import unit_fixed  # noqa: E402
from simulation.combat import apply_environmental_damage, HEAT_SCALE  # noqa: E402
from simulation.events import UnitHitEvent, UnitKilledEvent  # noqa: E402
from simulation.unit import Unit  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal gmap stub: apply_environmental_damage only touches gmap.heat (.shape
# + indexing). A real GameMap works too, but the stub keeps the consumer tests
# fast and free of level geometry.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _unprotected_body(monkeypatch):
    """#215: this module holds the heat row's UNSCALED law (heat_absorption
    1.0); the shipped protection only scales what reaches the body
    (tests/test_body_temp.py::test_the_protection_scales_both_inflows_alike)."""
    from simulation import body_temp
    monkeypatch.setattr(body_temp, "absorption_q", lambda env: 1 << 16)


class _HeatStub:
    """#215: the body law also reads the air on the footprint (one
    atmosphere, the engine's ambient: the steady-state anchor's conditions)
    and the tile size (the heat unit's joules)."""
    def __init__(self, h, w, tile_size_m=0.333):
        self.heat = np.zeros((h, w), dtype=np.int32)
        self.atmosphere = np.full((h, w), 1 << 16, dtype=np.int32)
        self.temperature = np.zeros((h, w), dtype=np.int32)
        self.solid = np.zeros((h, w), dtype=bool)
        self.tile_size_m = tile_size_m


#: Seconds enough for a body to settle at a held flux (#215): the time
#: constant is 2 s at 24 tps (4 s at 12: the felt scale is per tick), and 20
#: of them close any gap these fixtures open to below the law's rounding.
SETTLE_S = 80


def _settle(units, stub, tps, events=None):
    """Hold the injected flux for SETTLE_S seconds (the stub's heat buffer is
    never cleared), then run ONE more tick: each unit's damage of that last
    tick, the steady state's per-tick bill. The units carry hit points enough
    to live through it."""
    for _ in range(SETTLE_S * tps):
        apply_environmental_damage(units, stub, ticks_per_second=tps)
    before = [u.current_hp for u in units]
    apply_environmental_damage(units, stub, ticks_per_second=tps, events=events)
    return [b - u.current_hp for b, u in zip(before, units)]


def _expected_marine_dmg_raw(phi, tps):
    """Reproduce the §4.2 model for a default-human marine (temperature_max=60,
    environmental_damage_rate=1.0): the RAW (pre-snap) per-tick HP loss."""
    cmb = CFG.combat
    phi_abs = phi * float(cmb.unit_absorption) * (1.0 - float(cmb.unit_reflectivity))
    t_felt = float(cmb.heat_ambient_ref) + float(cmb.heat_flux_to_temp) * phi_abs
    over = t_felt - 60.0
    if over <= 0.0:
        return 0.0
    return 1.0 * (1.0 + float(cmb.heat_overtemp_scale) * over) * (1.0 / tps)


def _q(v):
    return unit_fixed.quantize_scalar(float(v))


def _expected_marine_dmg_q(phi, tps, mult_q=1 << 16):
    """THE row's law in Q16 (#29 P3e), written by hand: the gain and the
    dials quantized once, every product / quotient the real value rounded
    once (half away) -- for a default-human marine (temperature_max 60,
    environmental_damage_rate 1.0); ``mult_q`` a resist (the zombie's)."""
    cmb = CFG.combat
    one = 1 << 16
    peak = int(round(phi * HEAT_SCALE))
    gain_q = _q(float(cmb.heat_flux_to_temp) * float(cmb.unit_absorption)
                * (1.0 - float(cmb.unit_reflectivity)))
    t_q = _q(cmb.heat_ambient_ref) + unit_fixed.mul_q16_round(gain_q, peak)
    over_q = t_q - _q(60.0)
    if over_q <= 0:
        return 0
    k_q = _q(cmb.heat_overtemp_scale)
    dmg_q = unit_fixed.div_round(
        _q(1.0) * (one + unit_fixed.mul_q16_round(k_q, over_q)), one * tps)
    return unit_fixed.mul_q16_round(dmg_q, mult_q)


def _expected_marine_dmg(phi, tps):
    """The APPLIED per-tick HP loss: the integer law's Q16 count as HP. It
    stays within 1e-4 (relative) of the float model above -- the Q16
    quantization of the dials, the documented #29 P3e re-baseline (held by
    ``test_the_integer_law_tracks_the_float_model``)."""
    return _expected_marine_dmg_q(phi, tps) / 65536.0


def test_the_integer_law_tracks_the_float_model():
    """PROPERTY (#29 P3e): over the fixtures' flux range the integer row is
    the float model to within 1e-4 relative + one Q16 count. BREAKS IF a
    dial is quantized to a coarser grid or a rounding is dropped."""
    for phi in (phi_of(2.5), phi_of(3.0), phi_of(5.0), phi_of(10.0),
                phi_of(20.0), 1.0, 20.0, 50.0, 200.0):
        for tps in (12, 24):
            raw = _expected_marine_dmg_raw(round(phi * HEAT_SCALE) / HEAT_SCALE, tps)
            got = _expected_marine_dmg(phi, tps)
            assert abs(got - raw) <= 1e-4 * raw + 1.0 / 65536, (phi, tps)


def _place(unit, x, y):
    unit.set_pos(float(x), float(y))
    return unit


def _inject(stub, unit, phi):
    """Stamp `phi` (energy units) as a Q16.16 heat deposit on the unit's tiles."""
    raw = int(round(phi * HEAT_SCALE))
    for (tx, ty) in unit.occupied_tiles():
        stub.heat[ty, tx] = raw


# T5b step 6: PHI NOW HAS A PHYSICAL MEANING, so this module's fixtures state
# irradiance in kW/m2 instead of bare `phi` numbers.
#
# `phi` is the body's absorbed energy per tick in heat counts, so
#     irradiance = phi * 65536 * J_per_count * ticks_per_second / A_rad
#                = phi * 224.8 kW/m2
# with J_per_count = 0.4759 J (R13's pin) and A_rad = 4*0.333*2.5 = 3.33 m2.
# Before the flip `phi` came off the retired cast's FITTED emission scale
# (2419x the derived one), so the fixture values below meant nothing physical
# and "phi ~ 1 is a faint warmth" was simply the old scale's arithmetic.
# Under the derived currency phi = 1 is 225 kW/m2 -- five times the flux inside
# a compartment fire -- which is why the old `warm room` fixture now cooks.
KW_M2_PER_PHI = 224.775


def phi_of(kw_m2):
    """Irradiance in kW/m2 -> the `phi` this module injects."""
    return float(kw_m2) / KW_M2_PER_PHI


# ---------------------------------------------------------------------------
# Consumer tests (heat injected directly)
# ---------------------------------------------------------------------------
def test_hot_tile_damages_with_absorption_and_ramp():
    """A unit on a hot tile loses the §4.2 model amount once its body has
    settled (#215: the steady state IS the old law, to a count). BREAKS IF
    the derived loss stops matching heat_flux_to_temp (the body settles
    elsewhere) or the band stops reading the body."""
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    m.current_hp = 1e9
    phi = 50.0
    _inject(stub, m, phi)

    (dmg,) = _settle([m], stub, 24)

    expected = _expected_marine_dmg(phi, 24)
    assert expected > 0.0
    assert abs(dmg - expected) <= 1.0 / 65536


def test_cold_tile_zero_damage():
    """A unit on a cold (zero-heat) tile takes EXACTLY zero damage."""
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    m.current_hp = 100.0
    # heat stays 0 everywhere -> no flux.

    apply_environmental_damage([m], stub, ticks_per_second=24)

    assert m.current_hp == 100.0


def test_warm_room_survivable():
    """PROPERTY (T5b step 6): the burn band is anchored on the LITERATURE.
    2.5 kW/m2 -- the standard human pain / firefighter-exposure threshold --
    sits exactly at the edge of the survivable band, so 1 kW/m2 (bright
    sunshine) does nothing and 5 kW/m2 (blistering in ~30 s) hurts.

    This REPLACES "phi ~ 1 is a faint warmth", which was the retired cast's
    FITTED scale talking: under the derived currency phi = 1 is 225 kW/m2.
    `heat_flux_to_temp` is derived from this anchor (config.toml [combat]), so
    the test and the dial cannot drift apart -- and stating the fixture in
    kW/m2 means a future re-derivation moves the dial, not this file.

    BREAKS IF: the anchor moves, or `heat_flux_to_temp` stops being derived
    from it.
    """
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    m.current_hp = 100.0
    _inject(stub, m, phi_of(1.0))       # 1 kW/m2 -- bright sunshine

    apply_environmental_damage([m], stub, ticks_per_second=24)

    assert _expected_marine_dmg(phi_of(1.0), 24) == 0.0
    assert m.current_hp == 100.0
    # The anchor itself: 2.5 kW/m2 is the band EDGE, so the marine takes the
    # BASE rate there and nothing more -- `over` is 0, so the ramp contributes
    # nothing and the damage is exactly `environmental_damage_rate`, 1 HP/s,
    # i.e. 100 s to incapacitate. That is the right shape for a threshold the
    # literature describes as "pain in ~10 s, injury over tens of seconds".
    assert _expected_marine_dmg(phi_of(2.5), 24) * 24 == pytest.approx(1.0, abs=0.01)
    # Just below it, nothing at all -- the edge is real, not a region.
    assert _expected_marine_dmg(phi_of(2.4), 24) == 0.0
    # The literature band, stated so a reader sees what the dial buys:
    #   5 kW/m2 blistering in ~30 s, 10 severe pain / 2nd-degree in ~10 s,
    #   20 untenable. Damage must be monotone and reach lethal rates.
    d5 = _expected_marine_dmg(phi_of(5.0), 24) * 24
    d10 = _expected_marine_dmg(phi_of(10.0), 24) * 24
    d20 = _expected_marine_dmg(phi_of(20.0), 24) * 24
    assert 0.0 < d5 < d10 < d20
    assert d10 > 5.0, f"10 kW/m2 should be clearly lethal over ~15 s, got {d10} HP/s"
    assert d5 < 5.0, f"5 kW/m2 should be survivable for tens of seconds, got {d5} HP/s"


def test_zombie_takes_fire_multiplier_more():
    """A zombie takes exactly zombie.fire_damage_multiplier x a marine's hit."""
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    z = _place(Unit("Z1", x=20, y=20, team=ZOMBIE_TEAM), 20, 20)
    base_hp = 1e9
    m.current_hp = z.current_hp = base_hp  # avoid death, isolate the per-tick amount
    phi = 50.0
    _inject(stub, m, phi)
    _inject(stub, z, phi)

    # #215: the per-tick amount of the settled bodies (both settle at the
    # same felt temperature; the x4 stays in the damage pipeline).
    marine_dmg, zombie_dmg = _settle([m, z], stub, 24)
    assert marine_dmg > 0.0
    # Exact applied deltas (Q2-lift): each snapped to the Q16.16 grid AFTER
    # the zombie multiplier, mirroring combat.py's operation order — sharp.
    mult = float(CFG.zombie.fire_damage_multiplier)
    # #29 P3e: the marine's Q16 damage, the zombie's its resist product
    # rounded once (mitigate_q).
    assert abs(marine_dmg - _expected_marine_dmg_q(phi, 24) / 65536.0) <= 1 / 65536
    assert abs(zombie_dmg - _expected_marine_dmg_q(phi, 24, _q(mult)) / 65536.0) \
        <= mult / 65536
    ratio = zombie_dmg / marine_dmg
    # The two INDEPENDENT Q16.16 snaps (Q2-lift) perturb the exact multiplier
    # ratio by up to (|dz| + mult*|dm|)/marine ~= (1+mult)*(0.5/65536)/marine
    # — derive the tolerance from the actual magnitudes (x1.5 margin) instead
    # of a magic number, so a config change keeps the bound honest.
    snap = 0.5 / 65536.0
    assert abs(ratio - mult) < 1.5 * (1.0 + mult) * 3 * snap / marine_dmg
    # Report-quality numbers (visible with `pytest -s`).
    print(f"\n[marine vs zombie @ phi=50, 24tps] marine={marine_dmg:.6f} "
          f"zombie={zombie_dmg:.6f} ratio={ratio:.3f}")


def test_damage_scales_with_overtemperature():
    """A stronger beam does MORE than the flux-proportional baseline (k_over
    ramp): doubling the over-temperature more than doubles the damage."""
    stub = _HeatStub(40, 40)
    weak = _place(Unit("W", x=5, y=5, team=0), 5, 5)
    strong = _place(Unit("S", x=20, y=20, team=0), 20, 20)
    base_hp = 1e9
    weak.current_hp = strong.current_hp = base_hp
    _inject(stub, weak, 20.0)
    _inject(stub, strong, 200.0)

    weak_dmg, strong_dmg = _settle([weak, strong], stub, 24)
    assert weak_dmg > 0.0
    # 10x the flux -> well MORE than 10x the damage, because the (1 + k_over*over)
    # ramp grows with over-temperature on top of the linear flux term.
    assert strong_dmg > 10.0 * weak_dmg
    # And matches the closed-form model exactly (the Q16.16-snapped applied
    # delta — Q2-lift; _expected_marine_dmg snaps like combat.py does).
    assert abs(weak_dmg - _expected_marine_dmg(20.0, 24)) <= 1 / 65536
    assert abs(strong_dmg - _expected_marine_dmg(200.0, 24)) <= 1 / 65536


def test_heat_death_sets_source_and_no_conversion():
    """A heat kill emits source/killed_by == 'heat' and never flags
    killed_by_zombie (a burned corpse must not convert)."""
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    # One strong tick kills it. #29 P3a: an hp is an exact n/65536 (the
    # setter refuses anything else), so 1/128, not the old 0.01.
    m.current_hp = 1.0 / 128
    _inject(stub, m, 200.0)
    events = []

    apply_environmental_damage([m], stub, ticks_per_second=24, events=events)

    assert not m.alive
    assert m.killed_by_zombie is False
    hits = [e for e in events if isinstance(e, UnitHitEvent)]
    kills = [e for e in events if isinstance(e, UnitKilledEvent)]
    assert hits and hits[0].source == "heat"
    assert kills and kills[0].killed_by == "heat"


def test_tick_rate_independent_dps():
    """Same real DPS at 12 vs 24 tps: per-tick dmg halves when tps doubles, so
    dmg*tps (the per-second rate) matches."""
    phi = 50.0

    base_hp = 1e9

    stub12 = _HeatStub(40, 40)
    m12 = _place(Unit("M", x=10, y=10, team=0), 10, 10)
    m12.current_hp = base_hp
    _inject(stub12, m12, phi)
    (dmg12,) = _settle([m12], stub12, 12)

    stub24 = _HeatStub(40, 40)
    m24 = _place(Unit("M", x=10, y=10, team=0), 10, 10)
    m24.current_hp = base_hp
    _inject(stub24, m24, phi)
    (dmg24,) = _settle([m24], stub24, 24)

    # Each settled delta matches its snapped model to a count (#215).
    assert abs(dmg12 - _expected_marine_dmg(phi, 12)) <= 1 / 65536
    assert abs(dmg24 - _expected_marine_dmg(phi, 24)) <= 1 / 65536
    dps12 = dmg12 * 12
    dps24 = dmg24 * 24
    # Q2-lift: the RAW model is exactly tick-rate independent (x/12 == 2*(x/24)
    # in IEEE), but the two deltas snap to the Q16.16 grid INDEPENDENTLY, so
    # the per-second rates may differ by up to (12+24)*0.5/65536 ~= 2.7e-4 —
    # the documented quantization tolerance, imperceptible at DPS ~450.
    assert abs(dps12 - dps24) < 3 * 3e-4
    # 24-tps per-tick hit is half the 12-tps per-tick hit, to within a snap
    # of each settled body (#215).
    assert abs(dmg12 - 2.0 * dmg24) <= (3.0 / 65536) + 1e-9
    print(f"\n[tick-rate independence @ phi=50] "
          f"dmg/tick 12tps={dmg12:.6f} 24tps={dmg24:.6f} | "
          f"DPS 12tps={dps12:.4f} 24tps={dps24:.4f}")


def test_max_over_footprint():
    """Phi is the MAX over the footprint: one hot tile under the body burns it,
    even if the rest of the footprint is cold."""
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    base_hp = 1e9
    m.current_hp = base_hp
    # Only ONE of the 9 footprint tiles is hot.
    tiles = m.occupied_tiles()
    tx, ty = tiles[len(tiles) // 2]
    stub.heat[ty, tx] = int(round(50.0 * HEAT_SCALE))

    (dmg,) = _settle([m], stub, 24)

    assert abs(dmg - _expected_marine_dmg(50.0, 24)) <= 1 / 65536


def test_determinism_fixed_order():
    """Fixed unit order -> bit-identical HP after the serial apply."""
    phi = 73.0

    def run():
        stub = _HeatStub(60, 60)
        units = []
        for i in range(5):
            u = _place(Unit(f"M{i}", x=5 + 6 * i, y=5, team=i % 2),
                       5 + 6 * i, 5)
            u.current_hp = 1e9
            _inject(stub, u, phi)
            units.append(u)
        apply_environmental_damage(units, stub, ticks_per_second=24)
        return [u.current_hp for u in units]

    assert run() == run()


# ---------------------------------------------------------------------------
# End-to-end: the heat-clear ordering through a full Simulation.step()
# ---------------------------------------------------------------------------
def test_full_step_clears_heat_after_unit_damage():
    """After a full Simulation.step(): `heat` is wiped to 0 (the clear moved to
    the END of the tick, AFTER the unit-damage consumer) AND the unit-damage
    consumer saw the pre-clear value (the marine lost HP from the injected heat).
    """
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=42, breach_physics=bp, enable_recorder=False)
    # A marine somewhere passable in the interior (same spot the sim smoke test
    # uses) and un-paused so step() runs the tick body.
    m = Unit("M1", x=14, y=50, team=0)
    sim.add_unit(m)
    sim.set_paused(False)

    # The first tick stamps units / fires start-of-round explosives; let it run
    # so the unit is stamped and obstacles settle, then inject heat for tick 2.
    sim.step()

    m.current_hp = 500.0
    # Inject a strong heat deposit at the marine's footprint AFTER stamping but
    # the deposit must survive into the NEXT step's consumer. The render-side
    # ray pass normally fills `heat`; headless we fill it directly right before
    # the step so apply_environmental_damage (inside step, post-physics) reads it.
    # Physics runs the C++ heat->temperature conversion first (also a reader),
    # then unit damage, then the clear — so we inject here and check both.
    raw = int(round(200.0 * HEAT_SCALE))
    for (tx, ty) in m.occupied_tiles():
        sim.gmap.heat[ty, tx] = raw

    hp_before = m.current_hp
    sim.step()

    # The unit-damage consumer ran BEFORE the clear and saw the injected heat.
    assert m.current_hp < hp_before
    # And the clear ran at end-of-tick: heat is now zero everywhere.
    assert int(sim.gmap.heat.max()) == 0
