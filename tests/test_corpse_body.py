"""The corpse's body (arc #63 P5c, docs/swarm_P5c_impl.md; Erik, 2026-11-19:
"fire and explosion and heat still damage after death"; "a unit of 100hp hit
by a single bullet with damage 100 + its body HP ... would also be enough to
destroy the body"). ONE pool: the body is the unit's hp continuing below zero,
and a corpse is DESTROYED at ``hp <= -body_factor x max_hp``
(``corpse.destroyed``). The blast and the heat row reach corpses through
``damage.apply_packet``; a corpse takes the hp and nothing else. The CPU
build's host path; the device forms are gated by ``tests/cuda_unit_damage_check.py``
(the heat scene's corpses), ``tests/cuda_unit_payloads_check.py`` (a blast on
corpses) and ``tests/cuda_unit_kernels_check.py`` PART CORPSE (the door).

Each test names the property it protects and the change that must break it.
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

from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import (ai_zombie, body_temp, corpse, exchange,  # noqa: E402
                        unit_fixed, unit_traits)
from simulation.damage import KINETIC, DamagePacket, apply_packet  # noqa: E402
from simulation.events import UnitHitEvent, UnitKilledEvent  # noqa: E402
from simulation.materials import MAT_AIR, MAT_FURNITURE, MAT_HULL  # noqa: E402
from simulation.payloads import execute_payload  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402

ONE = unit_fixed.FP_ONE
TPS = int(CFG.clock.ticks_per_second)


def _room(n=24):
    tm = np.full((n, n), MAT_HULL, dtype=np.int32)
    tm[1:n - 1, 1:n - 1] = MAT_AIR
    return LevelData(name="corpse_body", version="2", path=Path("."), tilemap=tm,
                     tile_size_m=0.333, diffuse_path=Path("."), entities=[], wires=[])


def _sim(n=24, seed=1):
    return Simulation(_room(n), seed=seed, breach_physics=bp, enable_recorder=False,
                      ruleset=ContinuousRealtime())


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def _kill(u, extra=0.0):
    """Kill ``u`` through the one door: its hit points plus ``extra``."""
    apply_packet(u, DamagePacket(amount=u.current_hp + extra, dtype=KINETIC), None,
                 source="explosion")
    assert not u.alive


def _frag():
    from simulation.weapons import get_tables
    return get_tables().payloads.by_name["frag_standard"]


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

def test_the_body_is_the_hp_below_zero_against_the_units_own_full_health():
    """Property: a corpse is destroyed exactly when its hp reaches
    -round(body_factor x max_hp), its own class's full health times its HP
    hook (a zombie's corpse the zombie's), and a living unit never is.
    Breaks if destroyed reads anything but the hp against max_hp_q, if the
    factor is rounded twice, or if a living unit can be destroyed."""
    for team in (0, 255):
        u = Unit("U", x=2.0, y=2.0, team=team)
        full = unit_traits.max_hp_q(u)
        body = corpse.body_q(u)
        assert body == unit_fixed.mul_q16_round(full, corpse.get_settings().body_factor_q)
        assert not corpse.destroyed(u)
        _kill(u)
        u.hp_q = -body + 1
        assert not corpse.destroyed(u)
        u.hp_q = -body
        assert corpse.destroyed(u)
    v = Unit("V", x=2.0, y=2.0, team=0)
    v.hp_q = -10 * ONE * 100                     # fixture: alive at any hp
    assert not corpse.destroyed(v)


def test_the_killing_blows_overkill_counts():
    """Property (Erik: "a single bullet with damage 100 + its body HP ...
    would also be enough"): one packet of max_hp x (1 + factor) on a living
    unit kills it AND destroys its body, so it never rises, even killed by a
    zombie; one count less leaves the body. Breaks if the overkill is
    discarded at death (an hp clamp) or the destroyed check runs on a hp
    reset at death."""
    f = corpse.get_settings().body_factor_q
    u = Unit("U", x=2.0, y=2.0, team=0)
    full = unit_traits.max_hp_q(u)
    hit_q = full + unit_fixed.mul_q16_round(full, f)
    apply_packet(u, DamagePacket(amount=hit_q / ONE, dtype=KINETIC, amount_q=hit_q),
                 None, source="melee", mark_killed_by_zombie=True)
    assert not u.alive and corpse.destroyed(u)
    assert not ai_zombie.rises_as_zombie(u)
    w = Unit("W", x=2.0, y=2.0, team=0)
    apply_packet(w, DamagePacket(amount=(hit_q - 1) / ONE, dtype=KINETIC,
                                 amount_q=hit_q - 1),
                 None, source="melee", mark_killed_by_zombie=True)
    assert not w.alive and not corpse.destroyed(w)
    assert ai_zombie.rises_as_zombie(w)


def test_a_destroyed_corpse_takes_nothing_more_and_emits_nothing():
    """Property: a packet on a destroyed corpse leaves its hp where it is and
    adds no event, so its value is bounded and it hashes still. Breaks if
    apply_packet's destroyed gate is removed."""
    u = Unit("U", x=2.0, y=2.0, team=0)
    _kill(u)
    u.hp_q = -corpse.body_q(u) - 7               # fixture: destroyed
    hp0 = u.hp_q
    ev = []
    out = apply_packet(u, DamagePacket(amount=50.0, dtype=KINETIC), ev, source="bullet")
    assert out == 0.0 and u.hp_q == hp0 and ev == []


def test_a_corpse_packet_today_keeps_its_hit_event():
    """Property (today's same-slot reach, left as it was): a packet on a
    corpse that is not destroyed lowers its hp and emits its UnitHit, no
    second UnitKilled, no stress. Breaks if the new reaches' silence leaks
    into apply_packet itself."""
    u = Unit("U", x=2.0, y=2.0, team=0)
    _kill(u)
    hp0 = u.hp_q
    ev = []
    apply_packet(u, DamagePacket(amount=34.0, dtype=KINETIC, attacker_id=7), ev,
                 source="melee")
    assert u.hp_q == hp0 - 34 * ONE
    assert [type(e) for e in ev] == [UnitHitEvent]
    assert u.stress_q == 0


# ---------------------------------------------------------------------------
# The blast reaches corpses
# ---------------------------------------------------------------------------

def _frag_on(sim, u, dx=1):
    """A frag on the tile ``dx`` east of the corpse's centre tile (the frag's
    radius is 2 tiles: at 1 it takes off half its damage)."""
    cy = int(u.center_tile_y())
    cx = int(u.center_tile_x()) + dx
    ev = []
    execute_payload(sim.gmap, sim.edit_queue, sim.units, cy, cx, _frag(), sim.draws,
                    events=ev, owner_id=99)
    return ev


def test_a_frag_beside_a_corpse_lowers_its_hp_and_enough_destroy_it():
    """Property: a frag beside a marine's corpse lowers its hp by the blast's
    falloff (the same as on the living), and repeated frags destroy the
    corpse in a finite count; after that further frags leave the hp still.
    The corpse's hits add NO event and no credit. Breaks if the blast keeps
    its alive gate, if a corpse's hit emits an event, or if a destroyed
    corpse keeps taking damage."""
    sim = _sim()
    u = Unit("M", x=8.0, y=8.0, team=0)
    sim.add_unit(u)
    _step(sim)
    _kill(u)
    hp0 = u.hp_q
    ev = _frag_on(sim, u)
    assert u.hp_q < hp0
    assert not any(getattr(e, "unit_id", None) == u.id for e in ev)
    n = 1
    while not corpse.destroyed(u):
        _frag_on(sim, u)
        n += 1
        assert n < 50
    hp_d = u.hp_q
    _frag_on(sim, u)
    assert u.hp_q == hp_d and n >= 2


def test_a_corpse_takes_no_stress_status_poison_exposure_or_impulse_from_a_blast():
    """Property: a corpse hit by blasts takes the hp and nothing else: stress,
    statuses, poison, exposure, impulse and the conversion flag stay as the
    death left them, and the hit names nobody. Breaks if the corpse's packet
    runs the living's side effects."""
    sim = _sim()
    u = Unit("M", x=8.0, y=8.0, team=0)
    sim.add_unit(u)
    _step(sim)
    _kill(u)
    snap = (u.stress_q, list(u.exposure), u.poison_x0, u.poison_x1, u.imp_x, u.imp_y,
            u.killed_by_zombie, dict(u.recent_attackers or {}))
    for _ in range(3):
        _frag_on(sim, u)
    assert (u.stress_q, list(u.exposure), u.poison_x0, u.poison_x1, u.imp_x, u.imp_y,
            u.killed_by_zombie, dict(u.recent_attackers or {})) == snap


def test_a_living_units_blast_damage_is_unchanged_beside_a_corpse():
    """Property: a living marine beside a corpse takes the same blast damage
    and the same events as with no corpse there (the corpse's packet changes
    nobody else). Breaks if the corpse's reach disturbs the living's."""
    def run(with_corpse):
        sim = _sim()
        m = Unit("L", x=12.0, y=8.0, team=0)
        sim.add_unit(m)
        if with_corpse:
            c = Unit("C", x=8.0, y=8.0, team=0)
            sim.add_unit(c)
        _step(sim)
        if with_corpse:
            _kill(c)
        ev = _frag_on(sim, m, dx=-1)
        return m.hp_q, [(type(e).__name__, e.unit_id, float(getattr(e, "damage", 0)))
                        for e in ev if hasattr(e, "unit_id")]
    assert run(True) == run(False)


# ---------------------------------------------------------------------------
# Heat reaches corpses
# ---------------------------------------------------------------------------

class _Stub:
    """A map of one-atmosphere air, the 0.333 m tile, a heat buffer."""
    def __init__(self, h=30, w=30, t_game=0.0):
        self.heat = np.zeros((h, w), dtype=np.int32)
        self.atmosphere = np.full((h, w), ONE, dtype=np.int32)
        self.temperature = np.full((h, w), unit_fixed.quantize_scalar(t_game),
                                   dtype=np.int32)
        self.solid = np.zeros((h, w), dtype=bool)
        self.tile_size_m = 0.333


def _hot_corpse(team=0):
    u = Unit("C", x=10.0, y=10.0, team=team)
    u.set_pos(10.0, 10.0)
    _kill(u)
    return u


def test_a_hot_corpse_keeps_burning_and_is_destroyed_in_finite_time():
    """Property: a corpse whose body is above the band keeps taking heat
    damage (the body is kept at death and the row runs on it) and, held in
    hot air, is destroyed in finite time; then the row leaves it alone. A
    corpse in cool ambient air with an ambient body never moves. Breaks if
    the heat row keeps its alive gate, if death clears the body, or if a
    destroyed corpse keeps taking heat."""
    k = body_temp.k_temp_q() / ONE
    hot = _Stub(t_game=2000.0 / k)               # air 2000 C above ambient
    cool = _Stub()
    u = _hot_corpse()
    c = _hot_corpse()
    amb = body_temp.ambient_q()
    ev = []
    exchange.apply_environmental_damage([c], cool, TPS)
    assert c.body_temp_q == amb and c.hp_q == 0
    for t in range(1, 600 * TPS):
        exchange.apply_environmental_damage([u], hot, TPS, events=ev)
        if corpse.destroyed(u):
            break
    assert corpse.destroyed(u), (u.hp_q / ONE, (u.body_temp_q - amb) / ONE)
    assert ev == []
    hp_d, b_d = u.hp_q, u.body_temp_q
    exchange.apply_environmental_damage([u], hot, TPS)
    assert (u.hp_q, u.body_temp_q) == (hp_d, b_d)
    for _ in range(60 * TPS):
        exchange.apply_environmental_damage([c], cool, TPS)
    assert c.body_temp_q == amb and c.hp_q == 0


def test_death_keeps_the_body_temperature():
    """Property: a unit killed hot keeps its body temperature as a corpse
    (it is the same body) and the corpse goes on taking heat damage while
    its body is above the band. Breaks if the death clears the body."""
    u = Unit("C", x=10.0, y=10.0, team=0)
    u.set_pos(10.0, 10.0)
    u.body_temp_q = body_temp.ambient_q() + 500 * ONE
    _kill(u)
    assert u.body_temp_q == body_temp.ambient_q() + 500 * ONE
    hp0 = u.hp_q
    exchange.apply_environmental_damage([u], _Stub(), TPS)
    assert u.hp_q < hp0


def test_a_corpse_in_fire_is_destroyed_in_finite_time():
    """Property: a marine's corpse lying on a burning furniture pile (its
    footprint on the pile) is destroyed within two minutes (measured 65 s;
    docs/swarm_P5c_impl.md), and the heat adds no event naming it. Breaks if
    fire cannot reach a corpse (the heat row's alive gate)."""
    tm = np.full((40, 40), MAT_HULL, dtype=np.int32)
    tm[1:39, 1:39] = MAT_AIR
    tm[19:22, 19:22] = MAT_FURNITURE
    lv = LevelData(name="fire_corpse", version="2", path=Path("."), tilemap=tm,
                   tile_size_m=0.333, diffuse_path=Path("."), entities=[], wires=[])
    s = Simulation(lv, seed=3, breach_physics=bp, enable_recorder=False,
                   ruleset=ContinuousRealtime())
    _step(s)
    from simulation.weapons import get_tables
    row = get_tables().payloads.by_name["studio_ignite"]
    for y in range(19, 22):
        for x in range(19, 22):
            execute_payload(s.gmap, s.edit_queue, s.units, y, x, row, s.draws)
    _step(s, 20 * TPS)
    assert int(s.gmap.fire.max()) > 0, "the scene is moot: nothing burns"
    m = Unit("M", x=19.5, y=19.5, team=0)       # footprint on the pile
    s.add_unit(m)
    _step(s)
    _kill(m)
    for t in range(120 * TPS):
        _step(s)
        assert not any(getattr(e, "unit_id", None) == m.id for e in s.tick_events)
        if corpse.destroyed(m):
            break
    assert corpse.destroyed(m), (m.hp_q / ONE, m.body_temp_q / ONE)


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

def test_a_corpse_destroyed_by_a_blast_never_rises_by_either_trigger():
    """Property: a corpse a zombie killed and the larvae infested, destroyed
    by frags before the tick's end, rises by neither trigger and its level
    stays 0. Breaks if the door or the level ignores the hp-based destroyed
    check."""
    sim = _sim()
    u = Unit("M", x=8.0, y=8.0, team=0)
    sim.add_unit(u)
    _step(sim)
    _kill(u)
    u.killed_by_zombie = True
    u.infest_q = corpse.get_settings().threshold_q * 2   # fixture: infested
    for _ in range(50):
        if corpse.destroyed(u):
            break
        _frag_on(sim, u)
    assert corpse.destroyed(u)
    assert not ai_zombie.rises_as_zombie(u)
    corpse.infest(u, 10)
    assert u.infest_q == 0
    ai_zombie.convert_marines_to_zombies([u])
    assert not u.alive and not u.is_zombie


def test_a_risen_corpse_is_whole_again():
    """Property: a corpse that rises takes its full health back from the door
    (its body whole for the next death), even after blasts lowered it.
    Breaks if the door stops resetting the hp."""
    sim = _sim()
    u = Unit("M", x=8.0, y=8.0, team=0)
    sim.add_unit(u)
    _step(sim)
    _kill(u)
    u.killed_by_zombie = True
    _frag_on(sim, u)
    assert u.hp_q < 0 and not corpse.destroyed(u)
    ai_zombie.convert_marines_to_zombies([u])
    assert u.alive and u.hp_q == unit_traits.max_hp_q(u)
