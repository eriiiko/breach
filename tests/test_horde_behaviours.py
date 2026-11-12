"""#183 G9p stage 2 -- the horde's behaviours on the field.

Properties: with every behaviour switched off the horde plays the stage-1
field law exactly (the permanent gate); the wander stays within its leash,
draws no random number and depends only on (id, tick); a noise wakes exactly
the zombies within its radius, walls ignored, and a hearer walks the noise's
field and settles there; a zombie never steps onto a burning or hot tile,
not even between refreshes; a bashed door breaks after the designed number
of strikes, each strike a noise, and a hull door is never bashed; the sprint
runs only with a line of sight within its radius; the new per-unit state is
absence-transparent in the digest and cleared at conversion and death; the
settings door refuses a missing or mistyped key; shots and detonations make
noises at their rows' loudness.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import level_loader  # noqa: E402
from config import CFG, Namespace  # noqa: E402
from level_loader import EntityInstance, LevelData  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation, horde, noise  # noqa: E402
from simulation.ai_zombie import (convert_marines_to_zombies,  # noqa: E402
                                  update_zombies_tick)
from simulation.entities import door as door_mod  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import MAT_AIR, MAT_DOOR_CLOSED, MAT_HULL  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.unit import Unit  # noqa: E402

OFF = [(k, False) for k in horde.BEHAVIOUR_SWITCHES]

# The stage-1 field law's pinned matches (G9p stage 1, 2026-11-11; the
# values tests/test_zombie_control.py held before the stage-2 re-pin). This
# gate holds them while every behaviour is off; the next DELIBERATE re-pin
# of the field law by another cause retires them with it.
_STAGE1_SHIP_DEMO = dict(
    # Erik's zombie bite balance re-pin (2026-11-12: [zombie] melee_damage
    # 60 -> 34, attack_cooldown_seconds 1.0 -> 2.2, hp 400 -> 100;
    # docs/zombie_bite_balance_2026-11-12.md). SOLE CAUSE: this test passes
    # against the OLD pin on this tree (main 321babcf merged in) with the
    # three old [zombie] values set back in-process. Was:
    #   ticks 264 (now 254)
    #   field_digest 22e944fe6477d16ab0367b9562c357e96c8ed0a7
    #     a2277dde8dce6b2a486e9f23
    #   tick_digest 87dfe255f97bdc33320a7df36fea68dbfbfcd92f
    #     c9ff1fc86f91894febddd94d
    #   unit_stream c144ab6cb3bec1f33f143eaa1bfff97de346743f
    #     316ec412b60e65ec140e761a
    outcome="win", team=1, ticks=254,
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was d9c4c5cef6c04f9d80eaa2b368d88782a8fa28132f7cd39a8011c230b88cecde)
    field_digest="2e4852a4ac2077e62af6f4df06d253f1"
                 "d48cee022632f248b7356917e3b4373b",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was f6b702b5550299dcebfe2db8794a1e310b228648319b7c668b3a4c8ac1920f33)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was 9f78cd9b7ffe6d7016457f3ec8ca406d7d570114a7152a3f46153d7abdc6c6c1)
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was 82a361e85a0334a3a18dc84e63aea8bee0755ba6ac1b0d18423ea4c9f0aeea3e)
    tick_digest="55d714459d7db78897a9b8e830139548"
                "86e279cde0714acc105d39231dad457c",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was d83ee4d35c56f376d47daec6179ac3c37dfef360cd149d7ef6d48c6efa3a9f3f)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was f44bfcb5271c1768920a7727538c3c334d2c91f8ef3e9b98ce82e6bf5ff9cd20)
    # #29 P3e re-pin (2026-11-12, the match path to integer): the shot
    # cone, the exposure / crit rolls and the FieldEdit noise are integer draws
    # (simulation.draws: new random values), and the blast falloff, the payload
    # deposits, the heat and poison rows, mitigation and the hp are integer
    # (re-rounded) -- the GAME moved by design, not only the hash.
    # docs/p3e_ab_2026-11-12.md: win rate, survivors, rounds per kill and the
    # blast outcomes within seed noise over 24-72 seeds.
    # (was fd0d6bd79fbe4fa624065eda6bc9c04bc200e8ef38d7cdd5bd77bd4e5643877e)
    unit_stream="46dc6d4c5ad2ff73fd89e55430930884"
                "e4119672c4c5be7c73d88e5e3eeb87f1")
_STAGE1_BOX = dict(
    # Erik's zombie bite balance re-pin (2026-11-12: [zombie] melee_damage
    # 60 -> 34, attack_cooldown_seconds 1.0 -> 2.2, hp 400 -> 100;
    # docs/zombie_bite_balance_2026-11-12.md). SOLE CAUSE: this test passes
    # against the OLD pin on this tree (main 321babcf merged in) with the
    # three old [zombie] values set back in-process. Was:
    #   ticks 130 (now 294)
    #   field_digest 0fdc79f228cc2e64ca6936cf5a2ec41c52642adf
    #     ca545cbe51324cfad2e21631
    #   unit_stream 4a87a8bb9efdef78446729e66cebed58acfdda24
    #     3deb7ba4cdff947a55fa2740
    outcome="win", team=1, ticks=294,
    field_digest="ca6a1f289099dad155e21f9608e506f2"
                 "f19c481ca92ebd4efd0336aea658f15c",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was 05bdbac7c76250b949ef35aef81fdd403bd89af88232eebe061bf753e47b1e3d)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was a8f43fe1faf94900de0fb22c1b75e32d0ca991e25a69412e544a77b1b47343df)
    unit_stream="12cb16f75ea1727bda19f2dade7dcd98"
                "eb63e240a571e8bf4ca5f121fc2d5abf")


def _off(monkeypatch, keep=()):
    for k, v in OFF:
        if k not in keep:
            monkeypatch.setattr(CFG.zombie, k, v)


def _only(monkeypatch, *on):
    """Every behaviour off but ``on``."""
    _off(monkeypatch, keep=on)
    for k in on:
        monkeypatch.setattr(CFG.zombie, k, True)


def _room(w=40, h=24, tile=1.0, name="room"):
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    return tm, LevelData(name=name, version="3", path=Path("."), tilemap=tm,
                         tile_size_m=tile, diffuse_path=Path("."))


def _sim(level, physics=True):
    sim = Simulation(level, seed=1, breach_physics=bp if physics else None,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


# ---------------------------------------------------------------------------
# (a) all behaviours off == the stage-1 law
# ---------------------------------------------------------------------------
def test_with_every_behaviour_off_the_horde_plays_the_stage_1_law(
        monkeypatch, tmp_path):
    """Property (permanent): with wander, hearing, fear of fire, door
    bashing and the sprint all off, ship_demo seed 1 and the box play
    exactly the stage-1 field law's pinned matches -- the noise log, the
    loudness columns, the new unit state and the digest keys change nothing
    while the behaviours are off. Breaks if any behaviour leaks past its
    switch, or the stage-1 descent changes."""
    from test_zombie_control import _UnitStream, _box
    _off(monkeypatch)
    stream = _UnitStream()
    from test_bots_v1 import v1_spec
    ship = level_loader.load("ship_demo", str(ROOT / "levels"))
    # G8p: the first bots (bots_v1) played this pin; the layered bots
    # moved the shipped match, so this gate plays it with them.
    o = run_match(ship, v1_spec(ship), on_tick=stream)
    assert dict(outcome=o.result.outcome, team=o.result.team,
                ticks=o.result.ticks, field_digest=o.field_digest,
                tick_digest=o.tick_digest,
                unit_stream=stream.h.hexdigest()) == _STAGE1_SHIP_DEMO
    stream = _UnitStream()
    o = run_match(_box(tmp_path, "box"), on_tick=stream)
    assert dict(outcome=o.result.outcome, team=o.result.team,
                ticks=o.result.ticks, field_digest=o.field_digest,
                unit_stream=stream.h.hexdigest()) == _STAGE1_BOX


# ---------------------------------------------------------------------------
# (b) the wander
# ---------------------------------------------------------------------------
def _wander_run(monkeypatch, seed, ticks=1500):
    _only(monkeypatch, "wander")
    tm, lvl = _room(60, 30)
    sim = Simulation(lvl, seed=seed, breach_physics=None,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    # A prey far away behind the trigger radius (a wall of hull between).
    sim.gmap.material[1:29, 50] = MAT_HULL
    sim.gmap._update_caches()
    sim.add_unit(Unit("M", x=54, y=12, team=0))
    zs = [sim.get_unit(sim.add_unit(Unit(f"Z{i}", x=8 + 9 * i, y=12,
                                         team=ZOMBIE_TEAM)))
          for i in range(4)]
    rng0 = sim.rng.bit_generator.state
    track = []
    for _ in range(ticks):
        sim.step()
        track.append(tuple((z.tile_x, z.tile_y) for z in zs))
    return sim, zs, track, rng0


def test_the_wander_stays_on_its_leash_and_moves(monkeypatch):
    """Property: dormant zombies wander (each moves at least a few times in
    a minute) and never leave the leash around their anchor (squared tile
    distance <= the settings' leash_r2). Breaks if the wander drifts off, or
    stands still."""
    sim, zs, track, _ = _wander_run(monkeypatch, 1)
    r2 = sim.horde_settings.wander_leash_r2
    for i, z in enumerate(zs):
        assert z.zombie_anchor_y >= 0 and not z.zombie_activated
        moves = sum(1 for a, b in zip(track, track[1:]) if a[i] != b[i])
        assert moves >= 5, (i, moves)
        for pos in track:
            x, y = pos[i]
            assert ((y - z.zombie_anchor_y) ** 2
                    + (x - z.zombie_anchor_x) ** 2) <= r2


def test_the_wander_draws_no_random_number_and_is_a_hash_of_id_and_tick(
        monkeypatch):
    """Property: two runs with different sim seeds give the same wander
    tracks, the sim's generator is untouched by a wandering horde, and the
    direction is a pure function of (id, tick). Breaks if the wander draws
    from any stream or reads anything but its id and the tick."""
    _, _, t1, _ = _wander_run(monkeypatch, 1, ticks=600)
    sim, _, t2, rng0 = _wander_run(monkeypatch, 99, ticks=600)
    assert t1 == t2
    assert sim.rng.bit_generator.state == rng0
    a = [horde.wander_direction(3, t) for t in range(200)]
    assert a == [horde.wander_direction(3, t) for t in range(200)]
    assert set(a) == set(horde.NEIGHBOURS)
    assert a != [horde.wander_direction(4, t) for t in range(200)]


# ---------------------------------------------------------------------------
# (c) hearing
# ---------------------------------------------------------------------------
def test_a_noise_wakes_exactly_the_zombies_within_its_radius(monkeypatch):
    """Property: a noise at a tile with radius r gives a target to exactly
    the non-awake zombies whose centre is within r tiles (squared,
    inclusive) -- walls ignored: a zombie behind a solid wall inside the
    radius hears it, one just outside the radius does not, an awake zombie
    keeps its chase. Breaks if hearing reads line of sight or rounds the
    radius."""
    _only(monkeypatch, "hearing")
    tm, lvl = _room(70, 30)
    tm[1:29, 30] = MAT_HULL                       # a wall the noise ignores
    g = GameMap(lvl)
    # No sight wakes and no chain: hearing alone is measured.
    monkeypatch.setattr(CFG.zombie, "trigger_radius", 1)
    monkeypatch.setattr(CFG.zombie, "propagation_radius", 1)
    s = horde.horde_settings(CFG, 1.0)
    prey = Unit("M", x=64, y=24, team=0)         # far; out of every radius
    prey.id = 99
    r = 12
    cx, cy = 20, 12                               # noise tile
    zs = []
    for i, (dx, dy) in enumerate([(r, 0), (r + 1, 0), (0, r), (8, 9),
                                  (9, 8), (11, 0), (5, 5)]):
        z = Unit(f"Z{i}", x=cx + dx - 1, y=cy + dy - 1, team=ZOMBIE_TEAM)
        z.id = i
        zs.append(z)
    zs[5].set_pos(31.0, 11.0)                 # beyond the wall, d = 12
    zs[6].zombie_activated = True
    noise.emit(g, cy, cx, r)
    update_zombies_tick(g, [prey] + zs, 1, settings=s)
    for z in zs:
        d2 = ((z.center_tile_x() - cx) ** 2 + (z.center_tile_y() - cy) ** 2)
        hears = (not z.zombie_activated and d2 <= r * r)
        assert (z.zombie_noise_ticks > 0) is hears, (z.name, d2)
        if hears:
            assert (z.zombie_noise_y, z.zombie_noise_x) == (cy, cx)
    assert zs[5].zombie_noise_ticks > 0           # through the wall
    assert g.noise_log == []                      # drained by the slot


def test_a_hearer_walks_to_the_noise_and_settles_there(monkeypatch):
    """Property: a zombie that heard a noise walks its field (every step
    strictly lower), and on arrival -- without seeing prey -- drops the
    target and, with the wander on, re-anchors where it stands. Breaks if
    the hearer does not descend the noise's field or never settles."""
    _only(monkeypatch, "hearing", "wander")
    tm, lvl = _room(60, 24, tile=0.333)
    sim = _sim(lvl, physics=False)
    sim.gmap.material[1:23, 45] = MAT_HULL
    sim.gmap._update_caches()
    sim.add_unit(Unit("M", x=52, y=10, team=0))  # walled off, unseen
    z = sim.get_unit(sim.add_unit(Unit("Z", x=5, y=10, team=ZOMBIE_TEAM)))
    noise.emit(sim.gmap, 11, 30, 40)
    sim.step()
    assert z.zombie_noise_ticks > 0
    field = sim.horde_state.noise_field(sim.gmap, 3, 11, 30)
    last = horde.field_at(field, z.tile_y, z.tile_x)
    for _ in range(2000):
        sim.step()
        if z.zombie_noise_ticks == 0:
            break
        v = horde.field_at(field, z.tile_y, z.tile_x)
        assert v <= last
        last = v
    assert z.zombie_noise_ticks == 0 and not z.zombie_activated
    arrive = sim.horde_settings.noise_arrive_tiles
    assert horde.field_at(field, z.tile_y, z.tile_x) <= arrive
    assert (z.zombie_anchor_y, z.zombie_anchor_x) == (z.tile_y, z.tile_x) \
        or z.zombie_anchor_y >= 0


def test_a_noise_with_no_path_is_dropped_and_one_unreached_times_out(
        monkeypatch):
    """Property: a hearer with no path to its noise settles at once; one
    that cannot get closer keeps the target only for the timeout. Breaks if
    a hearer waits forever."""
    _only(monkeypatch, "hearing")
    tm, lvl = _room(60, 24)
    sim = _sim(lvl, physics=False)
    sim.gmap.material[1:23, 30] = MAT_HULL         # no way across
    sim.gmap._update_caches()
    sim.add_unit(Unit("M", x=55, y=3, team=0))
    monkeypatch.setattr(CFG.zombie, "trigger_radius", 1)
    z = sim.get_unit(sim.add_unit(Unit("Z", x=5, y=10, team=ZOMBIE_TEAM)))
    noise.emit(sim.gmap, 11, 45, 60)
    t = 0
    while t < 200:
        sim.step()
        t += 1
        if z.zombie_noise_ticks == 0:
            break
    assert z.zombie_noise_ticks == 0
    # The timeout bounds every chase.
    timeout = sim.horde_settings.noise_timeout_ticks
    assert timeout == round(CFG.zombie.noise_timeout_seconds
                            * CFG.clock.ticks_per_second)


def test_shots_and_detonations_are_noises_at_their_rows_loudness():
    """Property: a trigger pull of a ranged weapon records a noise at the
    shooter's centre with the row's ``loudness`` (metres) quantized to
    tiles; a detonating payload one at its tile with its own; a firing row
    without ``loudness`` is refused by the door. Breaks if a source skips
    the log or the units change."""
    from simulation.combat import _dispatch_trigger
    from simulation.payloads import execute_payload
    from simulation.field_edit import EditQueue
    from simulation.weapons import WeaponTable
    tm, lvl = _room(40, 20, tile=0.5)
    sim = _sim(lvl, physics=False)
    u = sim.get_unit(sim.add_unit(Unit("M", x=5, y=5, team=0)))
    w = sim.weapons_tables.weapons.by_name[u.weapon_id]
    assert w.loudness_tiles == noise.radius_tiles(w.loudness, 0.5) > 0
    _dispatch_trigger(sim.gmap, sim.units, u, u.center_tile_x(),
                      u.center_tile_y(), 30, 6, 1, [], 0.0, sim.rng, None,
                      None, w, 0.0)
    assert sim.gmap.noise_log[-1] == (u.center_tile_y(), u.center_tile_x(),
                                      w.loudness_tiles)
    frag = sim.weapons_tables.payloads.by_name["frag_standard"]
    assert frag.loudness_tiles == noise.radius_tiles(frag.loudness, 0.5) > 0
    execute_payload(sim.gmap, EditQueue(), sim.units, 10, 20, frag, sim.rng)
    assert sim.gmap.noise_log[-1] == (10, 20, frag.loudness_tiles)
    with pytest.raises(KeyError, match="loudness"):
        WeaponTable({"r": dict(archetype="projectile", range_tiles=10)}, 24)


# ---------------------------------------------------------------------------
# (d) fear of fire
# ---------------------------------------------------------------------------
def test_a_zombie_never_steps_onto_fire_and_waits_at_the_edge(monkeypatch):
    """Property: a burning band across the corridor between an awake zombie
    and its prey: the zombie walks up to it and never puts a body tile on
    a burning or hot tile, on any tick; with no way round it waits. Breaks
    if the mask or the step check lets fire through."""
    _only(monkeypatch, "fear_fire")
    tm = np.full((9, 50), MAT_HULL, dtype=np.int32)
    tm[2:7, 1:49] = MAT_AIR
    lvl = LevelData(name="fire", version="3", path=Path("."), tilemap=tm,
                    tile_size_m=1.0, diffuse_path=Path("."))
    sim = _sim(lvl, physics=False)
    g = sim.gmap
    g.fire[2:7, 25:27] = 65536                     # the burning band
    thr = sim.horde_settings.fear_temp_q16
    g.temperature[2:7, 22:25] = thr                # hot, not burning
    sim.add_unit(Unit("M", x=40, y=3, team=0))
    z = sim.get_unit(sim.add_unit(Unit("Z", x=3, y=3, team=ZOMBIE_TEAM)))
    z.zombie_activated = True
    for _ in range(1500):
        sim.step()
        blk = (slice(z.tile_y, z.tile_y + 3), slice(z.tile_x, z.tile_x + 3))
        assert not (g.fire[blk] > 0).any()
        assert not (g.temperature[blk] >= thr).any()
    assert z.tile_x + 3 == 22                      # at the edge, waiting


def test_a_fire_lit_between_refreshes_still_stops_the_step(monkeypatch):
    """Property: fire that appears after the field was flooded (on a tick
    the cache is kept) still blocks the step onto it. Breaks if the step
    trusts the field's mask alone."""
    _only(monkeypatch, "fear_fire")
    tm, lvl = _room(40, 9)
    sim = _sim(lvl, physics=False)
    sim.add_unit(Unit("M", x=30, y=3, team=0))
    z = sim.get_unit(sim.add_unit(Unit("Z", x=3, y=3, team=ZOMBIE_TEAM)))
    z.zombie_activated = True
    while z.tile_x == 3:
        sim.step()
    # Burn every tile of the next blocks (east, north, south) mid-refresh.
    x = z.tile_x
    sim.gmap.fire[:, x + 3] = 65536
    sim.gmap.fire[z.tile_y - 1, :] = 65536
    sim.gmap.fire[z.tile_y + 3, :] = 65536
    for _ in range(200):
        sim.step()
        assert z.tile_x == x


# ---------------------------------------------------------------------------
# (e) door bashing
# ---------------------------------------------------------------------------
def _door_inst(eid, ordinal, x, y, length_m=3.0, initial_state="closed"):
    fields = {f.name: f.default for f in door_mod.door.FIELDS}
    fields.update(x=x, y=y, orientation="v", length_m=length_m,
                  initial_state=initial_state)
    return EntityInstance(id=eid, class_name="door", ordinal=ordinal,
                          fields=fields)


def _door_level(hull=False):
    """A room split by a wall at x = 20 with a 3-tile closed door in it; or
    (``hull``) the door in the room's east hull with space beyond."""
    h, w = 12, 40
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    if hull:
        tm[:, 30:] = -1                            # space east of x = 29
        tm[1:h - 1, 29] = MAT_HULL
        tm[4:7, 29] = MAT_AIR                      # the door's span
        door = _door_inst("d", 0, 29, 4)
    else:
        tm[1:h - 1, 20] = MAT_HULL
        tm[4:7, 20] = MAT_AIR                      # the door's span
        door = _door_inst("d", 0, 20, 4)
    return LevelData(name="doors", version="3", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."),
                     entities=[door])


def test_a_bashed_door_breaks_after_the_designed_strikes_each_a_noise(
        monkeypatch):
    """Property: an awake zombie whose only route is a closed door strikes
    it at the bite's cadence until it breaks after exactly ceil(door hp /
    door_bash_damage) strikes, each strike a noise at the struck tile with
    the bash radius; then the door is destroyed and the zombie walks
    through. Breaks if the bash skips the wall-chew seam, its cadence or its
    noise, or the field does not count the door passable."""
    _only(monkeypatch, "bash_doors")
    sim = _sim(_door_level(), physics=False)
    m = sim.get_unit(sim.add_unit(Unit("M", x=32, y=5, team=0)))
    m.current_hp = 1e9
    z = sim.get_unit(sim.add_unit(Unit("Z", x=10, y=5, team=ZOMBIE_TEAM)))
    z.zombie_activated = True
    door = sim._doors[0]
    hp = float(sim.gmap.materials.hp[MAT_DOOR_CLOSED])
    want = math.ceil(hp / float(CFG.zombie.door_bash_damage))
    strikes = []
    broke = None
    for _ in range(want * 40 + 2000):
        sim.step()
        for (ty, tx, r) in sim.gmap.noise_log:
            assert (ty, tx) in set(door.span)
            assert r == sim.horde_settings.bash_loudness_tiles
            strikes.append(sim.tick)
        if broke is None and not door.alive:
            broke = len(strikes)
        if z.tile_x > 21:
            break
    assert broke == want
    gaps = {b - a for a, b in zip(strikes, strikes[1:])}
    assert gaps == {CFG.zombie.attack_cooldown_ticks}
    assert z.tile_x > 21                           # through the doorway


def test_a_hull_door_is_never_bashed(monkeypatch):
    """Property: a door whose far side is space is not a bashable tile and
    a zombie whose prey stands beyond it never strikes it (its hp holds,
    no noise). Breaks if a zombie can open the ship to vacuum."""
    _only(monkeypatch, "bash_doors")
    sim = _sim(_door_level(hull=True), physics=False)
    door = sim._doors[0]
    assert horde.is_hull_door(sim.gmap, door.span)
    assert not horde.bashable_door_tiles(sim.gmap, sim._doors).any()
    m = sim.get_unit(sim.add_unit(Unit("M", x=33, y=5, team=0)))
    m.current_hp = 1e9
    z = sim.get_unit(sim.add_unit(Unit("Z", x=20, y=5, team=ZOMBIE_TEAM)))
    z.zombie_activated = True
    hp0 = [int(sim.gmap.wall_hp[t]) for t in door.span]
    for _ in range(600):
        sim.step()
        assert sim.gmap.noise_log == []
    assert door.alive and [int(sim.gmap.wall_hp[t]) for t in door.span] == hp0


# ---------------------------------------------------------------------------
# (f) the sprint
# ---------------------------------------------------------------------------
def _first_step(monkeypatch, prey_x, wall=False, sprint=True):
    _only(monkeypatch, *(("sprint",) if sprint else ()))
    tm, lvl = _room(70, 12, tile=0.333)
    sim = _sim(lvl, physics=False)
    if wall:
        sim.gmap.material[1:11, 30] = MAT_HULL
        sim.gmap.material[1:4, 30] = MAT_AIR       # a way round, no sight
        sim.gmap._update_caches()
    sim.add_unit(Unit("M", x=prey_x, y=5, team=0))
    z = sim.get_unit(sim.add_unit(Unit("Z", x=5, y=5, team=ZOMBIE_TEAM)))
    z.zombie_activated = True
    x0 = z.x
    for t in range(1, 200):
        sim.step()
        if z.x != x0 or z.tile_y != 5:
            return t, sim
    raise AssertionError("never stepped")


def test_the_sprint_needs_sight_within_its_radius(monkeypatch):
    """Property: an awake zombie with a line of sight to prey within the
    sprint radius takes its steps at the Speed table's sprint cadence; with
    the prey beyond the radius, or behind a wall, or with the sprint off, at
    the shamble. Breaks if the sprint ignores sight or radius, or does not
    read the [speeds] row."""
    r = int(math.sqrt(horde.horde_settings(CFG, 0.333).sprint_r2))
    t, sim = _first_step(monkeypatch, 5 + r - 2)
    assert t == sim.speed_table.zombie_sprint_ticks
    t, sim = _first_step(monkeypatch, 5 + r + 4)
    assert t == sim.speed_table.zombie_shamble_ticks
    t, sim = _first_step(monkeypatch, 40, wall=True)
    assert t == sim.speed_table.zombie_shamble_ticks
    t, sim = _first_step(monkeypatch, 5 + r - 2, sprint=False)
    assert t == sim.speed_table.zombie_shamble_ticks
    assert sim.speed_table.zombie_sprint_ticks < \
        sim.speed_table.zombie_shamble_ticks


# ---------------------------------------------------------------------------
# (g) state, digest, door
# ---------------------------------------------------------------------------
def test_the_horde_state_is_absence_transparent_and_cleared(monkeypatch):
    """Property: a zombie with no noise and no anchor records no horde key
    in the unit digest; a noise target and an anchor record theirs; a death
    and a conversion clear both. Breaks if the new state hashes while
    absent or survives death or conversion."""
    from field_ab_harness import _unit_record
    from simulation.damage import DamagePacket, KINETIC, apply_packet
    z = Unit("Z", x=5, y=5, team=ZOMBIE_TEAM)
    keys = {"zombie_noise_y", "zombie_noise_x", "zombie_noise_ticks",
            "zombie_anchor_y", "zombie_anchor_x"}   # #29 P3a: the schema's
    rec = _unit_record(z)
    assert not keys & set(rec)
    z.zombie_noise_y, z.zombie_noise_x, z.zombie_noise_ticks = 3, 4, 10
    z.zombie_anchor_y, z.zombie_anchor_x = 5, 5
    rec = _unit_record(z)
    assert [rec[k] for k in ("zombie_noise_y", "zombie_noise_x",
                             "zombie_noise_ticks")] == [3, 4, 10]
    assert [rec["zombie_anchor_y"], rec["zombie_anchor_x"]] == [5, 5]
    z.current_hp = 1.0
    apply_packet(z, DamagePacket(amount=100, dtype=KINETIC), source="bullet")
    assert not z.alive
    rec = _unit_record(z)
    assert "noise" not in rec and "anchor" not in rec
    m = Unit("M", x=5, y=5, team=0)
    m.zombie_anchor_y = m.zombie_anchor_x = 2
    m.alive = False
    m.killed_by_zombie = True
    convert_marines_to_zombies([m])
    assert m.is_zombie and m.zombie_anchor_y == -1 and m.zombie_noise_ticks == 0


@pytest.mark.parametrize("key,bad", [
    ("wander", 1), ("hearing", "yes"), ("fear_temp", 0.0),
    ("field_refresh_ticks", 0), ("noise_fields_max", 1.5),
    ("door_bash_damage", -1.0), ("sprint_radius_m", "far")])
def test_the_settings_door_refuses_a_mistyped_key(key, bad):
    """Property: the [zombie] door refuses each horde key that is missing
    or of the wrong type or range, naming it. Breaks if a dial is read
    unchecked."""
    z = Namespace(dict(vars(CFG.zombie)))
    setattr(z, key, bad)
    cfg = Namespace({"zombie": z, "clock": CFG.clock})
    with pytest.raises(ValueError, match=key):
        horde.horde_settings(cfg, 1.0)
    delattr(z, key)
    with pytest.raises(ValueError, match=key):
        horde.horde_settings(cfg, 1.0)
