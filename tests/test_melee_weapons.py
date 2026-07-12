"""The W5 MELEE gate: combat knife + arc baton (mechanics/03 §5).

What is locked here:

  - THE ADJACENCY PREDICATE (of record): Chebyshev-1 footprint contact,
    8-connected — edge contact, DIAGONAL corner contact, and overlap all
    count; a 2-tile gap does not; exact for any footprint shape (pairwise
    occupied_tiles(), not a bounding box); symmetric; no LOS term;
  - TO-HIT TRIVIALLY 1.0: melee never consults cover — the exposure roll
    does not EXIST on this path (melee_strike/melee_adjacent take no gmap:
    they physically cannot read a cover column), and a crit-0 strike
    consumes ZERO randomness while always connecting;
  - KNIFE CRIT-VS-FACING PINNED (seeded, deterministic): behind x4 -> 60%,
    flank x2 -> 30%, front x1 -> 15%; exact 70/35 packet amounts, exactly
    ONE door-4 uniform per strike, and NO zombie bullet_damage_multiplier
    on melee (that is the BULLET site rule);
  - BATON -> STUNNED 1.5 s AT THE DELIVERY SITE: 36 ticks, refresh-stacked,
    can_act/can_aim suppressed — while the DamagePacket type stays
    damage-only (field-set pinned: amount/dtype/source_id/ap, no status
    smuggling) and a killing blow stuns no corpse;
  - ZOMBIE MELEE PATH UNTOUCHED (regression): the ai_zombie bite keeps its
    shipped numbers, cadence, event silence, conversion semantics, and
    zero RNG draws;
  - COOLDOWN/AP CADENCE: rof gates the strike train at the derived tick
    intervals (knife 14, baton 19 @ 24 tps); a whiff charges nothing;
    ap_cost rides the weapon row (the shipped ORDER_FIRE consumption);
  - DORMANCY REPLICA: a melee-free scripted firefight is bit-identical
    (fields + hp + events + RNG end-state) to a twin with the melee branch
    sentinel-patched out (== the pre-W5 dispatch), and the branch is never
    entered.

Run:
    C:/Users/steen/miniconda3/python.exe -m pytest tests/test_melee_weapons.py -q
"""
from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG, ticks_from_seconds  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.combat import (  # noqa: E402
    melee_adjacent, melee_strike, process_shooting,
)
from simulation.damage import DamagePacket  # noqa: E402
from simulation.events import UnitHitEvent, UnitKilledEvent  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.orders import ORDER_FIRE, Order  # noqa: E402
from simulation.status import STUNNED, composed_flags  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402

SEED = 20260707


# ---------------------------------------------------------------------------
# Scaffolding (the test_spray_weapons shape)
# ---------------------------------------------------------------------------
def _level(h=24, w=24, edits=()):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    for (y, x, code) in edits:
        tm[y, x] = code
    return LevelData(name="w5_melee", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _room(h=24, w=24, edits=()):
    return GameMap(_level(h, w, edits))


def _weapon(name):
    return get_tables().weapons.by_name[name]


def _unit(name, x, y, team=0, uid=1, footprint=3):
    u = Unit(name, x=x, y=y, team=team, footprint=footprint)
    u.id = uid
    return u


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


# ---------------------------------------------------------------------------
# 1. The adjacency predicate — pinned semantics, incl. multi-tile + diagonal
# ---------------------------------------------------------------------------
def test_adjacency_predicate_pinned_footprint_contact():
    """3x3 vs 3x3 (anchor deltas): edge contact (dx 3) and DIAGONAL corner
    contact (dx 3, dy 3) are adjacent; a one-tile gap on the diagonal-plus
    axis (dx 4) is not; overlap trivially is. Symmetric both ways."""
    a = _unit("A", 4, 4, team=0, uid=1)

    edge = _unit("E", 7, 4, team=1, uid=2)        # tiles touch edge-to-edge
    diag = _unit("D", 7, 7, team=1, uid=3)        # corners touch diagonally
    gap = _unit("G", 8, 4, team=1, uid=4)         # one empty column between
    gap_d = _unit("GD", 8, 8, team=1, uid=5)      # one empty ring diagonally
    over = _unit("O", 5, 5, team=1, uid=6)        # footprints overlap

    assert melee_adjacent(a, edge)
    assert melee_adjacent(a, diag)                # THE DIAGONAL CASE: counts
    assert not melee_adjacent(a, gap)
    assert not melee_adjacent(a, gap_d)
    assert melee_adjacent(a, over)                # distance 0 counts trivially
    # Symmetry (pairwise tile walk — no anchor-order asymmetry).
    for other in (edge, diag, gap, gap_d, over):
        assert melee_adjacent(a, other) == melee_adjacent(other, a)


def test_adjacency_predicate_multi_size_footprints():
    """1x1 vs 3x3: the predicate walks occupied_tiles(), so mixed footprint
    sizes get exact answers — a 1x1 diagonally off the 3x3's corner is
    adjacent, two tiles off is not."""
    big = _unit("B", 4, 4, team=0, uid=1)         # tiles x,y in [4, 6]
    corner = _unit("c", 7, 7, team=1, uid=2, footprint=1)
    away = _unit("d", 8, 8, team=1, uid=3, footprint=1)
    beside = _unit("e", 3, 5, team=1, uid=4, footprint=1)
    assert melee_adjacent(big, corner)            # (7,7) touches (6,6)
    assert not melee_adjacent(big, away)          # Chebyshev 2 from (6,6)
    assert melee_adjacent(big, beside)            # (3,5) touches (4,5)


def test_adjacency_has_no_map_term():
    """Structural: neither the predicate nor the strike takes the map — the
    to-hit CANNOT consult cover/LOS (touching footprints have no tile
    between them; mechanics/03 §5 to-hit trivially 1.0)."""
    assert "gmap" not in inspect.signature(melee_adjacent).parameters
    assert "gmap" not in inspect.signature(melee_strike).parameters


# ---------------------------------------------------------------------------
# 2. To-hit 1.0, no cover roll: crit-0 melee consumes ZERO randomness
# ---------------------------------------------------------------------------
def test_baton_always_connects_and_draws_nothing():
    """The arc baton (crit_chance 0): every adjacent strike connects, no
    exposure roll, no crit roll — the RNG stream never moves (the
    lazy-roll rule: rolls that cannot matter are never drawn)."""
    baton = _weapon("arc_baton")
    rng = np.random.default_rng(SEED)
    state0 = rng.bit_generator.state
    for i in range(25):
        a = _unit("A", 4, 4, team=0, uid=1)
        z = _unit("Z", 7, 4, team=1, uid=2)
        hp0 = z.current_hp
        assert melee_strike([a, z], a, 8.0, 5.0, rng, None, baton)
        assert z.current_hp == hp0 - 10           # 10 ENERGY, exact
    assert rng.bit_generator.state == state0      # zero draws in 25 strikes


def test_whiff_semantics_no_target_or_not_adjacent():
    """No enemy on the order tile, a same-team unit there, or an enemy out
    of contact: the strike returns False and NOTHING is touched (no
    facing snap, no draw, no packet)."""
    knife = _weapon("combat_knife")
    rng = np.random.default_rng(SEED)
    state0 = rng.bit_generator.state
    a = _unit("A", 4, 4, team=0, uid=1)
    friend = _unit("F", 7, 4, team=0, uid=2)
    far = _unit("Z", 12, 4, team=1, uid=3)
    facing0 = a.facing
    assert not melee_strike([a, friend, far], a, 8.0, 5.0, rng, None, knife)
    assert not melee_strike([a, friend, far], a, 13.0, 5.0, rng, None, knife)
    assert a.facing == facing0
    assert friend.current_hp == 100.0 and far.current_hp == 100.0
    assert rng.bit_generator.state == state0


# ---------------------------------------------------------------------------
# 3. The knife's crit-vs-facing — pinned, seeded, deterministic
# ---------------------------------------------------------------------------
def _strike_from(dx_anchor, dy_anchor, seed, target_team=1):
    """One knife strike on a zombie at anchor (4,4) whose facing is EAST
    (0.0), from an attacker offset by the given anchor delta. Returns
    (damage_applied, n_draws)."""
    knife = _weapon("combat_knife")
    a = _unit("A", 4 + dx_anchor, 4 + dy_anchor, team=0, uid=1)
    z = _unit("Z", 4, 4, team=target_team, uid=2)
    z.facing = 0.0                                # facing EAST (y-up, 0=E)
    rng = np.random.default_rng(seed)
    hp0 = z.current_hp
    assert melee_strike([a, z], a, 5.0, 5.0, rng, None, knife)
    # Count the draws by replaying the twin stream.
    twin = np.random.default_rng(seed)
    n = 0
    while twin.bit_generator.state != rng.bit_generator.state and n < 4:
        twin.uniform(0.0, 1.0)
        n += 1
    assert twin.bit_generator.state == rng.bit_generator.state
    return hp0 - z.current_hp, n


def test_knife_behind_arc_crit_pinned():
    """Target faces EAST; attacker strikes from the WEST -> impact arrives
    from BEHIND -> crit% = 0.15 x 4 = 0.60 (human 120/90 arcs). Seed 1's
    first uniform is 0.5118 < 0.60 -> CRIT: 35 x 2 = 70, exactly one draw.
    Seed 5's 0.8050 >= 0.60 -> no crit: 35. THE ASSASSIN FANTASY, pinned
    — and no bullet_damage_multiplier on a zombie target (melee is not
    the bullet site rule)."""
    dmg, n = _strike_from(-3, 0, seed=1)          # attacker west, strikes east
    assert (dmg, n) == (70.0, 1)
    dmg, n = _strike_from(-3, 0, seed=5)
    assert (dmg, n) == (35.0, 1)


def test_knife_flank_and_front_arcs_pinned():
    """Flank (impact from the side): 0.15 x 2 = 0.30 — seed 2 (0.2616)
    crits, seed 1 (0.5118) does not. Front (impact head-on): 0.15 x 1 —
    seed 3 (0.0856) crits, seed 2 does not. One draw each, always."""
    assert _strike_from(0, -3, seed=2) == (70.0, 1)   # from the north: flank
    assert _strike_from(0, -3, seed=1) == (35.0, 1)
    assert _strike_from(3, 0, seed=3) == (70.0, 1)    # from the east: front
    assert _strike_from(3, 0, seed=2) == (35.0, 1)


def test_strike_snaps_attacker_facing_to_the_bearing():
    """The fire_burst facing rule holds for melee: a connecting strike due
    EAST sets facing 0 (kit atan2, y-up convention); a whiff does not."""
    baton = _weapon("arc_baton")
    a = _unit("A", 4, 4, team=0, uid=1)
    z = _unit("Z", 7, 4, team=1, uid=2)
    rng = np.random.default_rng(SEED)
    assert melee_strike([a, z], a, 8.0, 5.0, rng, None, baton)
    assert a.facing == 0.0


# ---------------------------------------------------------------------------
# 4. Baton -> STUNNED at the delivery site; packets stay damage-only
# ---------------------------------------------------------------------------
def test_damage_packet_type_is_damage_only():
    """The mechanics/06 §1 discipline, structurally: DamagePacket carries
    amount/dtype/source_id/ap and NOTHING else — no status field exists to
    smuggle CC through the pipeline. Statuses ride apply_status at the
    delivery site (the W3 teargas->BLINDED pattern)."""
    assert {f.name for f in dataclasses.fields(DamagePacket)} == {
        "amount", "dtype", "source_id", "ap"}


def test_baton_applies_stunned_36_ticks_at_the_site():
    baton = _weapon("arc_baton")
    a = _unit("A", 4, 4, team=0, uid=7)
    z = _unit("Z", 7, 4, team=1, uid=2)
    events = []
    rng = np.random.default_rng(SEED)
    assert melee_strike([a, z], a, 8.0, 5.0, rng, events, baton)
    # The packet did the damage (event stream), the status came separately.
    assert [type(e) for e in events] == [UnitHitEvent]
    assert events[0].source == "melee" and events[0].damage == 10.0
    assert len(z.statuses) == 1
    st = z.statuses[0]
    assert st.kind == STUNNED
    assert st.remaining_ticks == 36               # 1.5 s @ 24 tps, derived
    assert st.magnitude_q16 == 0                  # pure CC — no DoT payload
    assert st.source_id == 7                      # attributed to the attacker
    flags = composed_flags(z)
    assert not flags.can_act and not flags.can_aim
    # Refresh stacking: a second strike re-ups the SAME instance.
    st.remaining_ticks = 5
    assert melee_strike([a, z], a, 8.0, 5.0, rng, None, baton)
    assert len(z.statuses) == 1 and z.statuses[0].remaining_ticks == 36


def test_knife_applies_no_status_and_corpses_take_none():
    """The knife row carries no status_kind -> statuses untouched. A baton
    blow that KILLS applies no status either — corpses don't get stunned
    (statuses freeze on corpses; a corpse status would be dead digest
    weight)."""
    knife = _weapon("combat_knife")
    baton = _weapon("arc_baton")
    a = _unit("A", 4, 4, team=0, uid=1)
    z = _unit("Z", 7, 4, team=1, uid=2)
    rng = np.random.default_rng(5)                # 0.8050: no crit anywhere
    assert melee_strike([a, z], a, 8.0, 5.0, rng, None, knife)
    assert not z.statuses
    z.current_hp = 8.0                            # the next jolt kills
    assert melee_strike([a, z], a, 8.0, 5.0, rng, None, baton)
    assert not z.alive
    assert not z.statuses                         # no stun on the corpse
    assert not z.killed_by_zombie                 # player melee never converts


# ---------------------------------------------------------------------------
# 5. Cooldown / AP cadence — the established rof/last_fire machinery
# ---------------------------------------------------------------------------
def test_row_derivations_and_ap_cost():
    knife = _weapon("combat_knife")
    baton = _weapon("arc_baton")
    tps = CFG.clock.ticks_per_second
    assert knife.rof_interval_ticks == ticks_from_seconds(0.6, tps) == 14
    assert baton.rof_interval_ticks == ticks_from_seconds(0.8, tps) == 19
    assert baton.status_ticks == ticks_from_seconds(1.5, tps) == 36
    assert knife.ap_cost == 1 and baton.ap_cost == 1
    assert knife.ammo_family == "none" and baton.ammo_family == "none"
    assert knife.mag_size == 0 and baton.mag_size == 0


def test_cooldown_gates_the_strike_train_exact_ticks():
    """A standing fire order on an adjacent zombie, driven straight through
    process_shooting: baton strikes land at ticks 0/19/38/57 exactly;
    knife (seed 13: first three flank draws 0.8648/0.8553/0.8110, no
    crits) at 0/14/28 — the third kills (3 x 35 > 100) and the train
    stops on the corpse."""
    gmap = _room()
    # Baton cadence.
    a = _unit("A", 4, 9, team=0, uid=1)
    a.weapon_id = "arc_baton"
    z = _unit("Z", 7, 9, team=1, uid=2)
    a.orders = [Order(ORDER_FIRE, target_fx=8, target_fy=10, phase=0)]
    rng = np.random.default_rng(SEED)
    state0 = rng.bit_generator.state
    hit_ticks = []
    for t in range(70):
        events = []
        process_shooting(gmap, [a, z], t, [], 0.0, rng, events=events)
        if any(isinstance(e, UnitHitEvent) for e in events):
            hit_ticks.append(t)
    assert hit_ticks == [0, 19, 38, 57]
    assert z.current_hp == 100.0 - 4 * 10
    assert rng.bit_generator.state == state0      # crit-0: still zero draws

    # Knife cadence + kill stop (seed 13 — pinned no-crit prefix).
    a2 = _unit("A2", 4, 9, team=0, uid=3)
    a2.weapon_id = "combat_knife"
    z2 = _unit("Z2", 7, 9, team=1, uid=4)
    a2.orders = [Order(ORDER_FIRE, target_fx=8, target_fy=10, phase=0)]
    rng = np.random.default_rng(13)
    hits, kills = [], []
    for t in range(60):
        events = []
        process_shooting(gmap, [a2, z2], t, [], 0.0, rng, events=events)
        hits += [(t, e.damage) for e in events if isinstance(e, UnitHitEvent)]
        kills += [e for e in events if isinstance(e, UnitKilledEvent)]
    assert hits == [(0, 35.0), (14, 35.0), (28, 35.0)]
    assert len(kills) == 1 and kills[0].killed_by == "melee"
    assert not z2.alive


def test_whiff_charges_no_cadence():
    """An order aimed at an empty tile whiffs every tick: last_fire_tick
    stays untouched, so the FIRST connecting strike is never rof-delayed —
    step the target into contact and it lands that very tick."""
    gmap = _room()
    a = _unit("A", 4, 9, team=0, uid=1)
    a.weapon_id = "arc_baton"
    z = _unit("Z", 12, 9, team=1, uid=2)          # far from the order tile
    a.orders = [Order(ORDER_FIRE, target_fx=8, target_fy=10, phase=0)]
    rng = np.random.default_rng(SEED)
    for t in range(10):
        process_shooting(gmap, [a, z], t, [], 0.0, rng)
    assert a.last_fire_tick == -999 and z.current_hp == 100.0
    z.x = 7.0                                     # into contact, on the tile
    events = []
    process_shooting(gmap, [a, z], 10, [], 0.0, rng, events=events)
    assert a.last_fire_tick == 10
    assert any(isinstance(e, UnitHitEvent) for e in events)


# ---------------------------------------------------------------------------
# 6. End-to-end through the conductor: the chain-stun (and zero RNG)
# ---------------------------------------------------------------------------
def test_e2e_baton_chain_stun_suppresses_the_zombie_bite():
    """Full Simulation: a baton marine with a standing fire order on an
    ACTIVATED adjacent zombie. The tick order (statuses -> shooting ->
    zombie AI) means the tick-0 strike stuns BEFORE the first bite, and
    the 19-tick cadence re-stuns inside every 36-tick window: the zombie
    never bites (marine hp untouched), the zombie is whittled 10/strike,
    and the WHOLE FIGHT consumes zero randomness (crit-0 melee — the
    dormant-seam pattern with the weapon actively swinging)."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=5, y=9, team=0)
    m.weapon_id = "arc_baton"
    z = Unit("Z", x=8, y=9, team=1)
    mid = sim.add_unit(m)
    sim.add_unit(z)
    z.zombie_activated = True
    state0 = sim.rng.bit_generator.state
    assert sim.apply_action(mid, Order(
        ORDER_FIRE, target_fx=9, target_fy=10, phase=0))
    hp_m, hp_z = m.current_hp, z.current_hp
    _step(sim, 60)                                # strikes at 0/19/38/57
    assert m.current_hp == hp_m                   # never bitten: chain-stunned
    assert z.current_hp == hp_z - 4 * 10
    assert any(st.kind == STUNNED for st in z.statuses)
    assert not composed_flags(z).can_act
    assert sim.rng.bit_generator.state == state0  # zero draws, whole fight


# ---------------------------------------------------------------------------
# 7. Zombie melee regression — the ai_zombie path is untouched
# ---------------------------------------------------------------------------
def test_zombie_bite_keeps_its_shipped_path():
    """An activated zombie against an order-less marine: bites land on the
    shipped center-distance rule + attack_cooldown cadence for the exact
    CFG.zombie.melee_damage, emit NO events (the site never had a list),
    convert on kill (killed_by_zombie — the ONLY converting death), and
    draw nothing. W5 must not have rerouted any of it."""
    sim = Simulation(_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=5, y=9, team=0)
    z = Unit("Z", x=8, y=9, team=1)
    sim.add_unit(m)
    sim.add_unit(z)
    z.zombie_activated = True
    state0 = sim.rng.bit_generator.state
    hp0 = m.current_hp
    bite = CFG.zombie.melee_damage
    _step(sim, 1)                                 # bite lands tick 0
    assert m.current_hp == hp0 - bite
    assert not any(isinstance(e, UnitHitEvent) for e in sim.tick_events)
    cooldown = CFG.zombie.attack_cooldown_ticks
    _step(sim, cooldown)                          # exactly one more bite
    assert m.current_hp == hp0 - 2 * bite
    assert not m.alive                            # 2 x 60 >= 100
    assert m.killed_by_zombie                     # the converting death
    assert sim.rng.bit_generator.state == state0  # the bite draws nothing


# ---------------------------------------------------------------------------
# 8. Dormancy replica — no melee equipped == the pre-W5 dispatch
# ---------------------------------------------------------------------------
_FIELDS = ("gas", "atmosphere", "wave_source", "wave_p", "fire", "wall_hp",
           "material", "temperature", "water_depth", "is_vacuum")


def test_dormancy_no_melee_weapon_is_bit_identical_to_prew5():
    """A scripted k5 firefight + a live zombie bite (no melee weapon
    equipped anywhere) on the LIVE W5 code vs a twin whose melee branch is
    sentinel-patched out (== the pre-W5 dispatch): every field, every hp,
    the event stream, positions, and the RNG end-state are bit-identical,
    tick for tick — and the sentinel proves the branch is never entered.
    The W5 seams cost a melee-free trajectory nothing (the LAZY-ROLL rule
    at archetype granularity)."""
    import simulation.combat as combat_mod

    def build():
        sim = Simulation(_level(edits=[(10, 14, 2)]), seed=SEED,
                         breach_physics=bp, enable_recorder=False)
        m = Unit("M", x=3, y=9, team=0)           # k5 from config default
        z = Unit("Z", x=16, y=9, team=1)
        zb = Unit("ZB", x=6, y=9, team=1)         # biter in the marine's face
        mid = sim.add_unit(m)
        sim.add_unit(z)
        sim.add_unit(zb)
        zb.zombie_activated = True
        assert m.weapon_id == "k5_carbine"        # no melee weapon anywhere
        assert sim.apply_action(mid, Order(
            ORDER_FIRE, target_fx=17, target_fy=10, phase=0))
        return sim

    def run(sim, n=20):
        traj = []
        for _ in range(n):
            _step(sim)
            snap = {f: np.copy(getattr(sim.gmap, f)) for f in _FIELDS}
            snap["__heat__"] = np.copy(sim.gmap.heat)
            snap["__events__"] = repr(sim.tick_events)
            snap["__hp__"] = tuple(u.current_hp for u in sim.units)
            snap["__pos__"] = tuple((u.x, u.y) for u in sim.units)
            traj.append(snap)
        return traj, sim.rng.bit_generator.state

    traj_new, rng_new = run(build())

    calls = []
    saved = combat_mod.melee_strike
    combat_mod.melee_strike = lambda *a, **k: calls.append(1)  # pre-W5 twin
    try:
        traj_old, rng_old = run(build())
    finally:
        combat_mod.melee_strike = saved
    assert not calls, "the melee branch ran in a melee-free scenario"

    for t, (sn, so) in enumerate(zip(traj_new, traj_old)):
        for f in _FIELDS:
            assert np.array_equal(sn[f], so[f]), \
                f"tick {t}: field '{f}' diverged from the pre-W5 replica"
        assert np.array_equal(sn["__heat__"], so["__heat__"]), \
            f"tick {t}: heat diverged"
        assert sn["__events__"] == so["__events__"], f"tick {t}: events"
        assert sn["__hp__"] == so["__hp__"], f"tick {t}: hp"
        assert sn["__pos__"] == so["__pos__"], f"tick {t}: positions"
    assert rng_new == rng_old, "RNG stream moved vs the pre-W5 replica"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
