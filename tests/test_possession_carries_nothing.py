"""#29 reference debt (the P6 slot inventory, Part E): what possession means
on the continuous path, for the device port of slot 1b (P6.4).

``Simulation._possessed_ids`` gates slot 1b (``_consume_direct_intents``): a
possessed living marine consumes its latches and carries its wind-up and
plant; any other unit has a held trigger, a wind-up and a plant DROPPED.
These tests establish that the gate carries no state of its own: a living
non-zombie unit outside the set never holds anything slot 1b would consume or
drop, because every verb that writes such a column puts its unit in the set
(``melee_intent`` too since this patch, though its latch is slot 4a's). So
the device may treat 1b as "every living marine consumes its own columns"
with no possession column.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import level_loader  # noqa: E402
from control_policy import VERB_RULES  # noqa: E402
from level_loader import LevelData  # noqa: E402
from match_replay import INTENT_KINDS  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.intents import FP_ONE  # noqa: E402
from simulation.match_session import MatchSession  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.turning import PI_Q16, place  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402

#: What slot 1b consumes or drops, and each column's "holds nothing" value.
_ONE_B_REST = (("live_trigger", False), ("pending_throw", None),
               ("pending_plant", None), ("pending_use", False),
               ("pending_reload", False), ("pending_shove", False))


def _held(u) -> list:
    """The 1b columns ``u`` holds something in (empty = holds nothing)."""
    out = [name for name, rest in _ONE_B_REST if getattr(u, name) != rest]
    if u.throw_release_tick >= 0:
        out.append("throw_release_tick")
    if u.plant_done_tick >= 0:
        out.append("plant_done_tick")
    return out


class _Gate:
    """Wraps ``sim._consume_direct_intents``: at every slot-1b entry checks
    the property, and counts what made it non-vacuous."""

    def __init__(self, sim):
        self.sim = sim
        self.unpossessed_seen = 0       # living marine outside the set
        self.possessed_holding = 0      # a possessed marine holding 1b state
        self.possessed_seen = 0         # a living possessed marine
        self._inner = sim._consume_direct_intents
        sim._consume_direct_intents = self

    def __call__(self):
        sim = self.sim
        for u in sim.units:
            if not u.alive or u.is_zombie:
                continue
            if u.id in sim._possessed_ids:
                self.possessed_seen += 1
                self.possessed_holding += bool(_held(u))
                continue
            self.unpossessed_seen += 1
            assert not _held(u), (
                f"tick {sim.tick}: unit {u.id} is not possessed but holds "
                f"{_held(u)} -- slot 1b would drop it")
        self._inner()


def _args(kind: str, uid: int) -> tuple:
    vals = {"int": 0, "int?": None, "bool": True}
    return (uid,) + tuple(vals[t] for t in INTENT_KINDS[kind][1:])


def test_every_latched_hand_verb_possesses_its_unit():
    """Property: each latched verb of the policy's ``VERB_RULES`` (throw,
    use, reload, shove, melee, plant -- every one a future latch adds)
    puts its unit in ``_possessed_ids``, so possession is "the unit has
    issued a hand verb" with no exception P6.4 must model. Breaks if a
    latch verb (``melee_intent`` before this patch) leaves its unit out."""
    latches = [k for k, (cls, _) in VERB_RULES.items() if cls == "latch"]
    assert "melee_intent" in latches
    tm = np.ones((20, 20), dtype=np.int32)
    tm[1:19, 1:19] = 4
    level = LevelData(name="possess", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))
    for kind in latches:
        sim = Simulation(level, seed=1, breach_physics=bp,
                         enable_recorder=False, ruleset=ContinuousRealtime())
        uid = sim.add_unit(Unit("M", x=8, y=8, team=0))
        assert uid not in sim._possessed_ids
        getattr(sim, kind)(*_args(kind, uid))
        assert uid in sim._possessed_ids, kind


def test_an_unpossessed_marine_holds_nothing_slot_1b_reads_scripted():
    """Property: on the continuous path, at every slot-1b entry, a living
    non-zombie unit NOT in ``_possessed_ids`` holds no held trigger, no
    pending throw / plant / use / reload / shove, no wind-up and no plant
    under way -- so the gate's drop branch never drops anything from a
    living marine and possession carries no state. Driven by every verb on
    an arena with a melee-only striker, an idle marine and a zombie; not
    vacuous (unpossessed marines are checked, possessed ones are seen
    holding wind-ups and triggers). Breaks if any writer of a 1b column
    stops possessing its unit (e.g. ``throw_grenade_intent`` without the
    ``add``: the thrower issues nothing else, so its wind-up is then held
    outside the set)."""
    h = w = 64
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    level = LevelData(name="possess", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."))
    sim = Simulation(level, seed=7, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)

    def add(name, x, y, team, facing):
        u = sim.get_unit(sim.add_unit(Unit(name, x=x, y=y, team=team)))
        place(u, facing)
        return u

    m = add("M", 20, 20, 0, 0)                      # melee only
    e = add("E", 23, 20, 1, PI_Q16)                 # adjacent enemy
    f = add("F", 20, 30, 0, -(PI_Q16 // 2))         # thrower
    p = add("P", 40, 12, 0, 0)                      # planter
    s = add("S", 40, 40, 1, 0)                      # trigger + shove + use
    add("I", 30, 50, 0, 0)                          # never driven
    add("Z", 55, 55, ZOMBIE_TEAM, 0)
    gate = _Gate(sim)
    one = FP_ONE
    kinds = len(get_tables().hand_grenade_rows)
    for t in range(160):
        if t % 3 == 0:
            sim.melee_intent(m.id)
        if t < 40:
            sim.set_move_dir(e.id, 0, -one, one)
        if t == 40:
            sim.clear_move_dir(e.id)
        for k in range(kinds):
            if t == 10 + 20 * k:
                sim.throw_grenade_intent(f.id, 0, one, 24, 5 * one, k)
        if t == 15:
            sim.plant_charge_intent(p.id, 0)
        if 20 <= t < 120:
            sim.set_aim(s.id, one, 0)
            sim.set_trigger(s.id, True)
        if t == 60:
            sim.reload_intent(s.id)
            sim.shove_intent(s.id)
            sim.use_intent(s.id)
        if t == 120:
            sim.set_trigger(s.id, False)
        if t == 30:
            sim.set_flashlight(m.id, True)
        sim.step()
    assert gate.unpossessed_seen > 0 and gate.possessed_holding > 0


def test_an_unpossessed_marine_holds_nothing_slot_1b_reads_bots_match():
    """Property: the same as the scripted arena, in a ship_demo match whose
    first team is driven by the layered bots (every verb through the
    facade) and whose other team by ``idle`` (never possessed): at every
    slot-1b entry no living unpossessed marine holds 1b state (not vacuous:
    both possessed and unpossessed marines are seen). Breaks if a
    controller writes a 1b column outside the intent facade (the idle
    team's marines are never possessed, so any such write on them shows)."""
    lv = level_loader.load("ship_demo", str(ROOT / "levels"))
    teams = tuple(replace(t, controller=("bots" if i == 0 else "idle"))
                  for i, t in enumerate(lv.match.teams))
    ses = MatchSession(lv, replace(lv.match, teams=teams), seed=1,
                       breach_physics=bp)
    gate = _Gate(ses.sim)
    for _ in range(120):
        ses.tick()
    assert gate.unpossessed_seen > 0 and gate.possessed_seen > 0
