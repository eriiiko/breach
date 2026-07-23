"""Unit heat damage (engine/06 §4, proposal §4.2/§4.3) — STEP D.

The `Phi_rad`-only radiant-heat consumer of the per-tick `heat` deposit. A
LIVING unit, stamped as a full ray-blocker before the ray pass, samples the
already-occluded `heat` buffer at its footprint; if the felt temperature pushes
past its tolerance band it loses HP this tick. This module exercises the
consumer directly (heat injected into ``gmap.heat`` — no renderer, no ray pass)
and through a full ``Simulation.step()`` to assert the heat-clear ordering.

Verifies:
  - a unit on a HOT tile loses HP, scaled by absorption * (1 - reflectivity)
    and the over-temperature ramp;
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import load as load_level  # noqa: E402
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
class _HeatStub:
    def __init__(self, h, w):
        self.heat = np.zeros((h, w), dtype=np.int32)


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


def _expected_marine_dmg(phi, tps):
    """The APPLIED per-tick HP loss: the §4.2 model snapped to the Q16.16 grid
    (Q2-lift — combat.py quantizes every damage delta before it touches HP;
    change vs the raw model <= 1/131072 ~= 7.6e-6, the documented contract)."""
    return unit_fixed.quantize_hp_delta(_expected_marine_dmg_raw(phi, tps))


def _place(unit, x, y):
    unit.x = float(x)
    unit.y = float(y)
    return unit


def _inject(stub, unit, phi):
    """Stamp `phi` (energy units) as a Q16.16 heat deposit on the unit's tiles."""
    raw = int(round(phi * HEAT_SCALE))
    for (tx, ty) in unit.occupied_tiles():
        stub.heat[ty, tx] = raw


# ---------------------------------------------------------------------------
# Consumer tests (heat injected directly)
# ---------------------------------------------------------------------------
def test_hot_tile_damages_with_absorption_and_ramp():
    """A unit on a hot tile loses exactly the §4.2 model amount."""
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    m.current_hp = 100.0
    phi = 50.0
    _inject(stub, m, phi)

    apply_environmental_damage([m], stub, ticks_per_second=24)

    expected = _expected_marine_dmg(phi, 24)
    assert expected > 0.0
    assert abs((100.0 - m.current_hp) - expected) < 1e-9


def test_cold_tile_zero_damage():
    """A unit on a cold (zero-heat) tile takes EXACTLY zero damage."""
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    m.current_hp = 100.0
    # heat stays 0 everywhere -> no flux.

    apply_environmental_damage([m], stub, ticks_per_second=24)

    assert m.current_hp == 100.0


def test_warm_room_survivable():
    """A faint warmth (within the tolerance band) does no damage."""
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    m.current_hp = 100.0
    # Phi ~1: T_felt ~26 < temperature_max (60) -> over <= 0.
    _inject(stub, m, 1.0)

    apply_environmental_damage([m], stub, ticks_per_second=24)

    assert _expected_marine_dmg(1.0, 24) == 0.0
    assert m.current_hp == 100.0


def test_zombie_takes_fire_multiplier_more():
    """A zombie takes exactly zombie.fire_damage_multiplier x a marine's hit."""
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    z = _place(Unit("Z1", x=20, y=20, team=1), 20, 20)
    base_hp = 1000.0
    m.current_hp = z.current_hp = base_hp  # avoid death, isolate the per-tick amount
    phi = 50.0
    _inject(stub, m, phi)
    _inject(stub, z, phi)

    apply_environmental_damage([m, z], stub, ticks_per_second=24)

    marine_dmg = base_hp - m.current_hp
    zombie_dmg = base_hp - z.current_hp
    assert marine_dmg > 0.0
    # Exact applied deltas (Q2-lift): each snapped to the Q16.16 grid AFTER
    # the zombie multiplier, mirroring combat.py's operation order — sharp.
    raw = _expected_marine_dmg_raw(phi, 24)
    mult = float(CFG.zombie.fire_damage_multiplier)
    assert abs(marine_dmg - unit_fixed.quantize_hp_delta(raw)) < 1e-9
    assert abs(zombie_dmg - unit_fixed.quantize_hp_delta(raw * mult)) < 1e-9
    ratio = zombie_dmg / marine_dmg
    # The two INDEPENDENT Q16.16 snaps (Q2-lift) perturb the exact multiplier
    # ratio by up to (|dz| + mult*|dm|)/marine ~= (1+mult)*(0.5/65536)/marine
    # — derive the tolerance from the actual magnitudes (x1.5 margin) instead
    # of a magic number, so a config change keeps the bound honest.
    snap = 0.5 / 65536.0
    assert abs(ratio - mult) < 1.5 * (1.0 + mult) * snap / marine_dmg
    # Report-quality numbers (visible with `pytest -s`).
    print(f"\n[marine vs zombie @ phi=50, 24tps] marine={marine_dmg:.6f} "
          f"zombie={zombie_dmg:.6f} ratio={ratio:.3f}")


def test_damage_scales_with_overtemperature():
    """A stronger beam does MORE than the flux-proportional baseline (k_over
    ramp): doubling the over-temperature more than doubles the damage."""
    stub = _HeatStub(40, 40)
    weak = _place(Unit("W", x=5, y=5, team=0), 5, 5)
    strong = _place(Unit("S", x=20, y=20, team=0), 20, 20)
    base_hp = 1000.0
    weak.current_hp = strong.current_hp = base_hp
    _inject(stub, weak, 20.0)
    _inject(stub, strong, 200.0)

    apply_environmental_damage([weak, strong], stub, ticks_per_second=24)

    weak_dmg = base_hp - weak.current_hp
    strong_dmg = base_hp - strong.current_hp
    assert weak_dmg > 0.0
    # 10x the flux -> well MORE than 10x the damage, because the (1 + k_over*over)
    # ramp grows with over-temperature on top of the linear flux term.
    assert strong_dmg > 10.0 * weak_dmg
    # And matches the closed-form model exactly (the Q16.16-snapped applied
    # delta — Q2-lift; _expected_marine_dmg snaps like combat.py does).
    assert abs(weak_dmg - _expected_marine_dmg(20.0, 24)) < 1e-9
    assert abs(strong_dmg - _expected_marine_dmg(200.0, 24)) < 1e-9


def test_heat_death_sets_source_and_no_conversion():
    """A heat kill emits source/killed_by == 'heat' and never flags
    killed_by_zombie (a burned corpse must not convert)."""
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    m.current_hp = 0.01  # one strong tick kills it
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

    base_hp = 1000.0

    stub12 = _HeatStub(40, 40)
    m12 = _place(Unit("M", x=10, y=10, team=0), 10, 10)
    m12.current_hp = base_hp
    _inject(stub12, m12, phi)
    apply_environmental_damage([m12], stub12, ticks_per_second=12)
    dmg12 = base_hp - m12.current_hp

    stub24 = _HeatStub(40, 40)
    m24 = _place(Unit("M", x=10, y=10, team=0), 10, 10)
    m24.current_hp = base_hp
    _inject(stub24, m24, phi)
    apply_environmental_damage([m24], stub24, ticks_per_second=24)
    dmg24 = base_hp - m24.current_hp

    # Each applied delta matches its snapped model exactly (sharp).
    assert abs(dmg12 - _expected_marine_dmg(phi, 12)) < 1e-9
    assert abs(dmg24 - _expected_marine_dmg(phi, 24)) < 1e-9
    dps12 = dmg12 * 12
    dps24 = dmg24 * 24
    # Q2-lift: the RAW model is exactly tick-rate independent (x/12 == 2*(x/24)
    # in IEEE), but the two deltas snap to the Q16.16 grid INDEPENDENTLY, so
    # the per-second rates may differ by up to (12+24)*0.5/65536 ~= 2.7e-4 —
    # the documented quantization tolerance, imperceptible at DPS ~450.
    assert abs(dps12 - dps24) < 3e-4
    # 24-tps per-tick hit is half the 12-tps per-tick hit, to within one snap.
    assert abs(dmg12 - 2.0 * dmg24) <= (1.0 / 65536) + 1e-9
    print(f"\n[tick-rate independence @ phi=50] "
          f"dmg/tick 12tps={dmg12:.6f} 24tps={dmg24:.6f} | "
          f"DPS 12tps={dps12:.4f} 24tps={dps24:.4f}")


def test_max_over_footprint():
    """Phi is the MAX over the footprint: one hot tile under the body burns it,
    even if the rest of the footprint is cold."""
    stub = _HeatStub(40, 40)
    m = _place(Unit("M1", x=10, y=10, team=0), 10, 10)
    base_hp = 1000.0
    m.current_hp = base_hp
    # Only ONE of the 9 footprint tiles is hot.
    tiles = m.occupied_tiles()
    tx, ty = tiles[len(tiles) // 2]
    stub.heat[ty, tx] = int(round(50.0 * HEAT_SCALE))

    apply_environmental_damage([m], stub, ticks_per_second=24)

    assert abs((base_hp - m.current_hp) - _expected_marine_dmg(50.0, 24)) < 1e-9


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
    level = load_level("unhcr_vessel", levels_dir="prototypes")
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
