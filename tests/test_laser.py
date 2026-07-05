"""The first HITSCAN — Lance-3 (mechanics/03 §5, weapons W2) — the laser gate.

What is locked here:

  - SKEWER: one beam crosses two units on a line and BOTH take ENERGY
    packets (beam energy is NOT reduced by unit hits in v1), in march order,
    exact amounts;
  - INTEGER BEER-LAMBERT: per tile crossed the beam multiplies by
    max(0, ONE - sum_g absorb_g*density_g) in pure Q16.16 — the test places
    a gas density chosen so the transmission is EXACTLY 32768 (one half) and
    hand-computes the full integer chain: 25<<16 -> x32768>>16 -> 819200 ->
    round-half-away (819200 + 32768) >> 16 == 13;
  - the beam DIES below BEAM_MIN_ENERGY_Q16 inside a dense cloud (no
    packets, no wall chew beyond);
  - the beam STOPS at the first solid tile and chews it by exactly
    quantize(ammo.wall_damage) — the apply_explosion wall-HP path;
  - DORMANT: nothing in shipped play fires a beam (marines carry the k5) —
    no LaserFiredEvent, ever, until a test equips lance_3 explicitly;
  - integration: unit.weapon_id = "lance_3" + the ordinary ORDER_FIRE flow
    dispatches the hitscan archetype; a zombie takes the FULL 25 ENERGY
    (the bullet_damage_multiplier site rule is a BULLET rule — beams skip it).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_laser.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import unit_fixed, wall_fixed  # noqa: E402
from simulation.combat import BEAM_MIN_ENERGY_Q16, fire_beam  # noqa: E402
from simulation.events import LaserFiredEvent, UnitHitEvent  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gases import BLACK_SMOKE, WHITE_SMOKE  # noqa: E402
from simulation.materials import MAT_HULL  # noqa: E402
from simulation.orders import ORDER_FIRE, Order  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402

SEED = 20260705
FP_ONE = unit_fixed.FP_ONE


def _level(h=20, w=30, edits=()):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    for (y, x, code) in edits:
        tm[y, x] = code
    return LevelData(name="w2_laser", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _room(h=20, w=30, edits=()):
    return GameMap(_level(h, w, edits))


def _zombie(x, y, name="Z"):
    u = Unit(name, x=x, y=y, team=1)
    u.current_hp = 1e9
    return u


def _lance():
    t = get_tables()
    return t.weapons.by_name["lance_3"], t.ammo.by_name["cell_laser_standard"]


def _hits(events):
    return [e for e in events if isinstance(e, UnitHitEvent)]


def _beams(events):
    return [e for e in events if isinstance(e, LaserFiredEvent)]


def _fire(gmap, units, shooter, fx2, fy2, events, spread_deg=0.0, rng=None):
    weapon, ammo = _lance()
    fire_beam(gmap, units, shooter, 3, 9, fx2, fy2,
              tick=0, shots=[], real_time=0.0,
              rng=rng if rng is not None else np.random.default_rng(SEED),
              events=events, weapon=weapon, ammo=ammo, spread_deg=spread_deg)


# ---------------------------------------------------------------------------
# Skewer: two units, one beam, both packets, exact amounts
# ---------------------------------------------------------------------------
def test_beam_skewers_two_units_full_energy_each():
    gmap = _room()
    shooter = Unit("S", x=2, y=8, team=0)
    za = _zombie(10, 8, "ZA")      # footprint x in [10,12]
    zb = _zombie(18, 8, "ZB")      # footprint x in [18,20]
    shooter.id, za.id, zb.id = 1, 2, 3
    events = []
    _fire(gmap, [shooter, za, zb], shooter, 25, 9, events)

    hits = _hits(events)
    assert [(h.unit_id, h.damage, h.source) for h in hits] == [
        (2, 25.0, "laser"),        # march order: ZA first...
        (3, 25.0, "laser"),        # ...and the beam reaches ZB UNDIMINISHED
    ]
    beams = _beams(events)
    assert len(beams) == 1 and beams[0].kind == "laser"
    # Each skewered unit is hit exactly once despite a 3-tile-deep footprint.
    assert len(hits) == 2
    # hp moved by the quantized 25 each (neutral ENERGY mitigation).
    assert za.current_hp == 1e9 - unit_fixed.quantize_hp_delta(25)
    assert zb.current_hp == 1e9 - unit_fixed.quantize_hp_delta(25)


# ---------------------------------------------------------------------------
# Integer Beer-Lambert — the EXACT hand-computed halving
# ---------------------------------------------------------------------------
def test_gas_cloud_halves_beam_energy_exact_q16():
    """One white-smoke tile on the path, density chosen so the per-tile
    transmission is EXACTLY 32768/65536 (one half). Hand chain:
      energy = 25 << 16 = 1638400
      term   = (absorb_q * density_q) >> 16 == 32768   (asserted)
      trans  = 65536 - 32768 = 32768
      energy = (1638400 * 32768) >> 16 = 819200        (12.5 real)
      amount = (819200 + 32768) >> 16 = 13             (round half away)
    Both skewered units take 13 (the cloud sits before the first)."""
    gmap = _room()
    absorb_q = int(gmap.gases.beam_absorb_q16[WHITE_SMOKE])
    assert absorb_q > 0
    # Smallest density whose absorb product lands in the ==32768 window
    # (ceil division; window width 65536 >= absorb_q guarantees a member).
    density_q = -((-(32768 << 16)) // absorb_q)
    term = (absorb_q * density_q) >> 16
    assert term == 32768, "chosen density must halve exactly"
    assert density_q < 2**31                          # int32-safe
    gmap.gas[WHITE_SMOKE, 9, 6] = density_q           # on the march row

    shooter = Unit("S", x=2, y=8, team=0)
    za = _zombie(10, 8, "ZA")
    zb = _zombie(18, 8, "ZB")
    shooter.id, za.id, zb.id = 1, 2, 3
    events = []
    _fire(gmap, [shooter, za, zb], shooter, 25, 9, events)

    # The full integer chain, hand-computed:
    energy_q = 25 << 16
    energy_q = (energy_q * (FP_ONE - term)) >> 16
    assert energy_q == 819200
    expected_amount = (energy_q + (FP_ONE >> 1)) >> 16
    assert expected_amount == 13                      # 12.5 rounds away from 0

    hits = _hits(events)
    assert [(h.unit_id, h.damage) for h in hits] == [
        (2, float(expected_amount)),
        (3, float(expected_amount)),   # no further gas: ZB gets the same 13
    ]


def test_beam_dies_in_dense_cloud_before_the_target():
    """A black-smoke tile dense enough that transmission <= 0 kills the beam
    at that tile: no packets, no wall chew, tracer ends in the cloud."""
    gmap = _room()
    absorb_q = int(gmap.gases.beam_absorb_q16[BLACK_SMOKE])
    # density with (absorb * d) >> 16 >= ONE  ->  trans <= 0  ->  energy 0.
    density_q = -((-(FP_ONE << 16)) // absorb_q) + 1
    gmap.gas[BLACK_SMOKE, 9, 7] = density_q
    east_wall_hp = int(gmap.wall_hp[9, 29])

    shooter = Unit("S", x=2, y=8, team=0)
    z = _zombie(14, 8)
    shooter.id, z.id = 1, 2
    events = []
    _fire(gmap, [shooter, z], shooter, 25, 9, events)

    assert _hits(events) == []                        # nothing reached
    assert 0 < BEAM_MIN_ENERGY_Q16                    # the kill threshold
    beam = _beams(events)[0]
    assert int(beam.to_tile[0]) == 7                  # died in the cloud
    assert int(gmap.wall_hp[9, 29]) == east_wall_hp   # no chew anywhere
    assert z.current_hp == 1e9


# ---------------------------------------------------------------------------
# Wall stop + chew
# ---------------------------------------------------------------------------
def test_beam_stops_at_wall_and_chews_it():
    gmap = _room(edits=[(9, 8, MAT_HULL)])            # wall mid-path
    shooter = Unit("S", x=2, y=8, team=0)
    z = _zombie(12, 8)                                # behind the wall
    shooter.id, z.id = 1, 2
    q300 = wall_fixed.quantize_scalar(300.0)
    assert int(gmap.wall_hp[9, 8]) == q300
    events = []
    _fire(gmap, [shooter, z], shooter, 25, 9, events)

    assert _hits(events) == []                        # the wall shielded Z
    _, ammo = _lance()
    assert ammo.wall_damage == 15
    assert int(gmap.wall_hp[9, 8]) == q300 - wall_fixed.quantize_scalar(15.0)
    assert int(_beams(events)[0].to_tile[0]) == 8     # terminated at the wall


# ---------------------------------------------------------------------------
# Dormant in shipped play + the ORDER_FIRE integration
# ---------------------------------------------------------------------------
def test_laser_dormant_when_nobody_carries_one():
    """Marines carry the k5; a scripted firefight emits ShotFiredEvents but
    NEVER a LaserFiredEvent — the beam path exists only as data until a unit
    is explicitly equipped."""
    assert "lance_3" in get_tables().weapons.by_name  # the row ships...
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=2, y=8, team=0)
    z = Unit("Z", x=14, y=8, team=1)
    m_id = sim.add_unit(m)
    sim.add_unit(z)
    assert m.weapon_id == "k5_carbine"                # ...but nobody fires it
    assert sim.apply_action(m_id, Order(ORDER_FIRE, target_fx=15,
                                        target_fy=9, phase=0))
    lasers = []
    for _ in range(6):
        sim.set_paused(False)
        sim.step()
        lasers += _beams(sim.tick_events)
    assert lasers == []
    assert sim.bullets == []                          # k5 resolves same-tick


def test_lance3_fires_through_the_order_flow():
    """unit.weapon_id = 'lance_3' + the ordinary ORDER_FIRE: the archetype
    dispatcher routes to the beam; the zombie takes the FULL 25 ENERGY (the
    bullet site rule does not apply to beams)."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=2, y=8, team=0)
    z = Unit("Z", x=14, y=8, team=1)
    m_id = sim.add_unit(m)
    sim.add_unit(z)
    m.weapon_id = "lance_3"
    zombie_hp = z.current_hp
    assert sim.apply_action(m_id, Order(ORDER_FIRE, target_fx=15,
                                        target_fy=9, phase=0))
    sim.set_paused(False)
    sim.step()
    beams = _beams(sim.tick_events)
    assert len(beams) == 1 and beams[0].unit_id == m_id
    hits = [h for h in _hits(sim.tick_events) if h.unit_id == z.id]
    assert len(hits) == 1
    assert hits[0].damage == 25.0                     # FULL energy, no x0.25
    assert hits[0].source == "laser"
    assert z.current_hp == zombie_hp - 25.0
    assert sim.bullets == []                          # hitscan never persists


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
