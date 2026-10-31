"""#122 -- the horde through the control seam (``control_zombies.ZombieControl``,
match controller ``"zombies"``).

Properties: the shipped zombie law is WRAPPED, not rewritten (a match plays
bit for bit as before #122 -- the one golden-style pin below, deliberately);
every zombie has one driver per tick; a bite's kill is credited to melee and
the converted marine is then driven by the zombie controller; an idle zombie
controller leaves the horde standing; the registry builds ``zombies``; the
game path (no match) still runs the conductor's zombie slot.

Levels are written through ``level_lib`` (never a hand-written toml) and
matches run through ``run_match`` (the one match loop).
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import control_source  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import simulation.simulation as sim_mod  # noqa: E402
from control_zombies import ZombieControl  # noqa: E402
from field_ab_harness import _capture_unit_state  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.events import UnitKilledEvent  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.unit import Unit  # noqa: E402

IDLE = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                 controller="idle")
# Two marine rooms far apart, the zombie room right next to team 0 (the
# test_match.py box): with idle marines the horde eats team 0 first.
ROOMS = [
    SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=4),
    SpawnRoom(team=1, x=30, y=14, w=8, h=8, max_per_team=4),
    SpawnRoom(team=ZOMBIE_TEAM, x=12, y=2, w=8, h=8, max_per_team=4),
]
BOX_SPEC = MatchSpec(teams=(IDLE, IDLE), seed=1, zombies=3,
                     time_limit_s=20.0)


def _box(tmp_path, name, spec=BOX_SPEC, *, w=40, h=24):
    lvl = tmp_path / name
    lvl.mkdir()
    level_lib.write_level_header(lvl, name=name, tile_size_m=1.0)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", ROOMS)
    level_lib.write_match(lvl, spec)
    return level_loader.load(name, str(tmp_path))


class _UnitStream:
    """on_tick hook: a hash over every tick's synced unit records (positions,
    hp, life, team, ...) -- what the zombie law writes; events excluded, so
    the #122 melee events (which change no state) do not enter it."""

    def __init__(self):
        self.h = hashlib.sha256()

    def __call__(self, sim):
        self.h.update(repr(_capture_unit_state(sim)["units"]).encode())


# ---------------------------------------------------------------------------
# (a) the shipped law, wrapped -- pinned against the pre-#122 runner
# ---------------------------------------------------------------------------
# Captured at the start of #122 (main 5ee2712, the conductor driving the
# horde inside Simulation.step) with tools/run_match.py. This pin IS the
# property "wrap, don't rewrite": the zombie controller must reproduce the
# conductor's law tick for tick. A change of the zombie LAW (or of anything
# else these matches touch) moves it -- then re-pin deliberately, with the
# rationale, never to make the wrapping pass.
_PRE_122_BOX = dict(
    outcome="win", team=1, ticks=82,
    field_digest="87e20008aff3541d2d0b1d64113108d5"
                 "c09f85dd40ae1b58a0a6f58f12f83268",
    unit_stream="6dab6957a82fdd55786f7ec375e39761"
                "bdced4c61aa69a4cc6eb66a8cffb580a")
_PRE_122_SHIP_DEMO = dict(
    # #140 re-pin (2026-10-31, the wind law retires the pressure-gradient
    # push). SOLE CAUSE, verified on the same build: the #140 P1a tree
    # (78d3d78, which still reproduces the old pin: win 0 at 1528, bdc5ad3e /
    # 4d32170a / a5ddf313) with k_push = 0 and knockdown_dv_threshold = 1e30
    # gives exactly the values below, and so does the P1b tree with the wind
    # thresholds at 1e15 / 2e15 Pa -- the push WAS the whole change, the
    # wind law never fires in this match. The match is now won by team 1 at
    # tick 706 (no squad is shoved around a doorway any more).
    # (was win 0 at 1528; field bdc5ad3e6e6a7f797199a2783efc13276f6f515df8e7
    #  2ce0a50ae9e16d06a898; tick 4d32170a7a7ff465ba410f935c8912ae594999c903
    #  169aa66582ae2e7bcb2ab8; units a5ddf313af731dd1b7e983e68ebe448dffbe59a9
    #  fcf238eadcf262576c97f47a)
    outcome="win", team=1, ticks=706,
    field_digest="b38c2d29ffeb91d3ea5fb772fa24d597"
                 "86a53bc1c90004f7321bffbf442d3ca4",
    # #123 re-pin (credited events): the last tick holds the winning bullet's
    # UnitHit + UnitKilled, whose hashed record gained `attacker_id` (2, the
    # shooter). SOLE CAUSE, verified on the same build: hashing the record
    # without that key reproduces c69925fa... exactly; outcome, ticks,
    # field_digest and the per-tick unit stream are unchanged.
    # (was c69925fa68e8ba10fbb724eaedbcb53d56058f3b4efaf91a2601d1366be0f170)
    tick_digest="ce979185366be487732e587a851ab6e3"
                "589d0b0ea8671ab06ef7c5f41ba25c37",
    unit_stream="b55cc54e4e12bd7002a70cecd01d1377"
                "03964eb73fa63c15da59f0b415cedb4e")


def test_the_wrapped_law_plays_the_pre_122_match_on_a_small_box(tmp_path):
    """Property: idle marines + 3 zombies under the "zombies" controller give
    the pre-#122 result, final fields and per-tick unit stream. Breaks if
    the controller changes the law's unit list, its order or its slot in the
    tick, drives a zombie twice, or skips a tick."""
    stream = _UnitStream()
    o = run_match(_box(tmp_path, "box"), on_tick=stream)
    got = dict(outcome=o.result.outcome, team=o.result.team,
               ticks=o.result.ticks, field_digest=o.field_digest,
               unit_stream=stream.h.hexdigest())
    assert got == _PRE_122_BOX


def test_the_wrapped_law_plays_the_pre_122_ship_demo_match():
    """Property: ship_demo seed 1 (bots vs bots + 4 zombies) ends exactly as
    the conductor-driven law plays it (re-pinned at #140: win team 1 at tick
    706) with the same tick_digest. Breaks like the box test, on the real
    level."""
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    stream = _UnitStream()
    o = run_match(level, on_tick=stream)
    got = dict(outcome=o.result.outcome, team=o.result.team,
               ticks=o.result.ticks, field_digest=o.field_digest,
               tick_digest=o.tick_digest, unit_stream=stream.h.hexdigest())
    assert got == _PRE_122_SHIP_DEMO


def test_bots_and_zombies_one_seed_one_digest(tmp_path):
    """Property (the seed gate with the horde in it): two runs of one seed
    give one digest; the zombie controller draws nothing of its own. Breaks
    if ZombieControl reads anything unsynced or iterates an unordered set
    into the law."""
    bots = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                     controller="bots")
    lvl = _box(tmp_path, "seeded", replace(BOX_SPEC, teams=(bots, bots)))
    a, b = run_match(lvl), run_match(lvl)
    assert (a.tick_digest, a.result) == (b.tick_digest, b.result)


# ---------------------------------------------------------------------------
# (b) one driver per zombie per tick
# ---------------------------------------------------------------------------
def test_each_zombie_is_driven_once_per_tick_in_a_match(tmp_path,
                                                        monkeypatch):
    """Property: under the match path the zombie law is entered at most ONCE
    per tick (never by the conductor AND the controller), and every living
    zombie is in that call exactly once. Breaks if the conductor's slot keeps
    running for a match, if ZombieControl calls the law itself, or if a
    request names a zombie twice into the law."""
    calls = Counter()
    seen = []
    real = sim_mod.update_zombies_tick

    def counting(gmap, units, tick, events=None):
        calls[tick] += 1
        ids = [u.id for u in units if u.is_zombie and u.alive]
        seen.append((tick, ids))
        return real(gmap, units, tick, events=events)

    monkeypatch.setattr(sim_mod, "update_zombies_tick", counting)

    class _Twice(ZombieControl):
        """Asks twice a tick: the second request must not drive again."""
        def handle_frame(self, sim, renderer=None):
            super().handle_frame(sim, renderer)
            super().handle_frame(sim, renderer)

    monkeypatch.setitem(control_source._MATCH_CONTROLLERS, "zombies_twice",
                        _Twice)
    living = {}

    def watch(sim):
        living[sim.tick - 1] = sorted(u.id for u in sim.units
                                      if u.is_zombie and u.alive)

    run_match(_box(tmp_path, "once",
                   replace(BOX_SPEC, zombie_controller="zombies_twice")),
              on_tick=watch)
    assert calls and max(calls.values()) == 1
    for tick, ids in seen:
        # The law saw each zombie living at the tick's start exactly once.
        if tick - 1 in living:
            assert sorted(ids) == living[tick - 1], (tick, ids)
    # ... and no tick with a living zombie went undriven.
    for t, ids in living.items():
        if ids and t + 1 in living:
            assert calls[t + 1] == 1, t + 1


def test_drive_zombies_is_refused_on_a_game_path(tmp_path):
    """Property: without a match's hand-over the conductor drives the horde,
    and the ZOMBIE intent is refused -- a second driver is never silently
    added. Breaks if drive_zombies is accepted on a game path."""
    sim = Simulation(_box(tmp_path, "game"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    with pytest.raises(RuntimeError):
        sim.drive_zombies([0])


# ---------------------------------------------------------------------------
# (c) credited bites; a converted marine joins the controller's horde
# ---------------------------------------------------------------------------
def test_zombie_kills_are_credited_and_the_dead_join_the_horde(tmp_path):
    """Property (scope §5 item 4's gate): in a match with idle marines and
    the zombie controller, every death is a UnitKilledEvent credited to
    melee, the dead marines convert (ZOMBIE_TEAM, slot kept in their old
    roster) and are thereafter MOVED by the zombie controller while their
    old team's controller emits nothing. Breaks if the bite leaves the
    event list, a converted marine is left undriven, or its slot moves."""
    kills = []
    track = {}

    def watch(sim):
        for ev in sim.tick_events:
            if isinstance(ev, UnitKilledEvent):
                kills.append((ev.unit_id, ev.killed_by))
        for uid in {k for k, _ in kills}:
            u = sim.get_unit(uid)
            if u.alive and u.is_zombie:
                track.setdefault(uid, set()).add((u.x, u.y))

    lvl = _box(tmp_path, "credit", replace(BOX_SPEC, time_limit_s=30.0))
    o = run_match(lvl, on_tick=watch)
    assert kills, "the horde killed nobody"
    assert all(src == "melee" for _, src in kills), kills
    assert o.converted_alive >= 1
    # Each converted marine walked after it turned (team 0's controller is
    # idle: only the zombie controller can have moved it).
    moved = [uid for uid, pos in track.items() if len(pos) > 1]
    assert moved, track


def test_converted_marine_keeps_its_roster_slot(tmp_path):
    """Property: conversion (where MatchRules runs it) moves the unit to
    ZOMBIE_TEAM but its slot stays in the old team's roster. Breaks if the
    hand-over re-rosters converted units."""
    lvl = _box(tmp_path, "slot")
    rules_sim = {}
    run_match(lvl, on_tick=lambda s: rules_sim.setdefault("sim", s))
    sim = rules_sim["sim"]
    roster0 = sim.roster(0)
    assert [u.slot for u in roster0] == list(range(len(roster0)))
    converted = [u for u in roster0 if u.is_zombie]
    assert converted and all(u.team == ZOMBIE_TEAM for u in converted)


# ---------------------------------------------------------------------------
# (d) an idle horde stands
# ---------------------------------------------------------------------------
def test_an_idle_zombie_controller_leaves_the_horde_motionless(tmp_path):
    """Property: ``zombie_controller = "idle"`` (written through the writer,
    read through the loader door) gives the horde to IdleControl, so no
    zombie moves, activates or bites. Breaks if the conductor still drives
    the horde in a match, or the key is dropped at the door."""
    spec = replace(BOX_SPEC, zombie_controller="idle", time_limit_s=5.0)
    lvl = _box(tmp_path, "passive", spec)
    assert lvl.match.zombie_controller == "idle"
    text = (tmp_path / "passive" / level_loader.MATCH_FILENAME).read_text()
    assert 'zombie_controller = "idle"' in text
    start = {}
    end = {}

    def watch(sim):
        for u in sim.units:
            if u.is_zombie:
                start.setdefault(u.id, (u.x, u.y))
                end[u.id] = (u.x, u.y, u.zombie_activated)
    o = run_match(lvl, on_tick=watch)
    assert start and all(end[i][:2] == start[i] and not end[i][2]
                         for i in start)
    assert o.result.ticks == round(spec.time_limit_s * 24)
    assert o.converted_alive == 0


def test_the_default_zombie_controller_leaves_match_files_unchanged():
    """Property: the default (``zombies``) is not written, so the checked-in
    match files are byte-identical to the writer's text. Breaks if the
    writer starts emitting the default key (every generated level would
    change) or the loader's default drifts from the writer's."""
    for name in ("ship_demo", "playground"):
        level = level_loader.load(name, str(ROOT / "levels"))
        assert level.match.zombie_controller == "zombies"
        path = ROOT / "levels" / name / level_loader.MATCH_FILENAME
        assert level_lib.format_match_text(level.match).encode() == \
            path.read_bytes().replace(b"\r\n", b"\n")


# ---------------------------------------------------------------------------
# (e) the registry
# ---------------------------------------------------------------------------
def test_the_registry_builds_the_zombie_controller(tmp_path):
    """Property: ``zombies`` is a registry row building a ZombieControl for
    ZOMBIE_TEAM; an unregistered zombie_controller is refused at the door.
    Breaks if the row is dropped or the door stops checking the key."""
    assert "zombies" in control_source.registered_controllers()
    c = control_source.create_controller("zombies", ZOMBIE_TEAM)
    assert isinstance(c, ZombieControl)
    with pytest.raises(ValueError):
        control_source.create_controller("zombies", 0)
    with pytest.raises(ValueError, match="zombie_controller"):
        _box(tmp_path, "bad", replace(BOX_SPEC, zombie_controller="nobody"))


# ---------------------------------------------------------------------------
# (f) the game path keeps the conductor's slot
# ---------------------------------------------------------------------------
def test_the_game_path_still_runs_the_conductor_s_zombie_slot(tmp_path):
    """Property: a WEGO sim (the shipped default ruleset) with a zombie and a
    marine and NO controller: the zombie still activates and walks -- the
    conductor drives it. Breaks if the slot's guard ever leaves the horde
    to a controller outside a match."""
    sim = Simulation(_box(tmp_path, "wego"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    m = Unit("M", x=4, y=10, team=0, footprint=3)
    z = Unit("Z", x=16, y=10, team=ZOMBIE_TEAM, footprint=3)
    sim.add_unit(m)
    sim.add_unit(z)
    sim.set_paused(False)
    x0 = z.x
    for _ in range(48):
        sim.set_paused(False)
        sim.step()
    assert z.zombie_activated
    assert z.x < x0
