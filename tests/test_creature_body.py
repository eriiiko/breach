"""#106 P6d1 -- creature bodies: the limp octopus in the shared sim.

Properties protected (each test names the change that must break it):

* DORMANCY -- a creature-free sim builds no body, calls no solver, writes a
  carrier with ``n_creatures == 0`` and hashes exactly the pre-P6d1 bytes;
  its recorder dump carries no creature key. Breaks if the fold or the ring
  stops being presence-gated.
* THE LIMP GATE (the reference's G2 dead-arm signature, through Simulation on
  a flooded room): released 0.2 m above the deck, every node reaches z = 0
  within ~2 s, never goes below it, and the animal is at rest (< 1 mm/s)
  after 10 s; the crown stays in the map and off blocked tiles; the unit's
  anchor and facing ARE the pose through the converters. Breaks on a wrong
  body / arm set-up, a missing gravity or floor, or a stale write-back.
* HANDEDNESS -- the pose heading turns counter-clockwise on screen like
  ``Unit.facing``; the solver's yaw is its negative; arm 0 lies at its row
  angle from the heading. Breaks if a sign flips.
* WRITER ROUTING -- MOVE_DIR / AIM never move or turn a creature (they are the
  hub's goal); the wave push never displaces it (the creature row turns it
  into a body impulse); the round-end snap skips it. Breaks if any second
  writer touches its position.
* HUMANS UNCHANGED -- humans in the same room are byte-identical with and
  without the octopus, and the unit stamp planes differ only on the crown's
  footprint. Breaks if a creature leaks into anything but its stamp.
* THE A/B HARNESS locates a planted one-count divergence at the right arm,
  node and axis; the record decodes to the body's own arrays.
* PICKLE -- the creature system round-trips through pickle mid-run with the
  digest unchanged; WEGO -- a TwoPhaseWEGO round boundary (the tick rewind and
  the round-end snap) leaves the creature section equal to a continuous run's.
* THE SPAWN DOOR refuses an unknown species, a species without a creature row,
  a footprint off the map or off open floor, an arm laid into a wall, and a
  build without the chain solver; ``level_lib.write_creatures`` round-trips
  through the loader; match.toml carries creatures through ``spawn_match``.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")
if not hasattr(bp, "arm_chain_tick"):
    pytest.skip("breach_physics predates the arm chain solver (#106 P6c): rebuild cpp/",
                allow_module_level=True)

import level_lib  # noqa: E402
import level_loader  # noqa: E402
from field_ab_harness import capture_snapshot, diff_trajectories  # noqa: E402
from field_digest import tick_digest  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import arm_chain as AC  # noqa: E402
from simulation import creature_fixed as CF  # noqa: E402
from simulation import creature_system as CS  # noqa: E402
from simulation import water_fixed  # noqa: E402
from simulation.factions import CREATURE_TEAM  # noqa: E402
from simulation.match import CreatureSpawn, MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.ruleset import ContinuousRealtime, TwoPhaseWEGO  # noqa: E402
from simulation.unit import Unit  # noqa: E402

TILE_M = 1.0 / 3.0
TPS = 24
MM = 1e-3


# ---------------------------------------------------------------------------
# The flooded room, written through level_lib
# ---------------------------------------------------------------------------
def write_room(root: Path, name: str, *, w: int = 30, h: int = 30, creatures=(),
               walls=(), rooms=(), match=None, flood_m: float = 2.0):
    """A hull box of (w, h) tiles at the game's tile, flooded ``flood_m``
    deep everywhere inside (water everywhere is what the arm chain assumes,
    v1), with optional extra wall tiles, creatures, spawn rooms and a match,
    every file through the level_lib writers."""
    lvl = root / name
    lvl.mkdir()
    level_lib.write_level_header(lvl, name=name, tile_size_m=TILE_M)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    for (x, y) in walls:
        tm[y, x] = MAT_HULL
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    depth = np.where(tm == MAT_AIR, water_fixed.quantize_scalar(flood_m), 0).astype(np.int32)
    level_lib.write_water(lvl, depth)
    if creatures:
        level_lib.write_creatures(lvl / "level.toml", list(creatures))
    if rooms:
        level_lib.write_spawn_rooms(lvl / "level.toml", list(rooms))
    if match is not None:
        level_lib.write_match(lvl, match)
    return level_loader.load(name, str(root))


def make_sim(level, ruleset=None, recorder=False):
    return Simulation(level, seed=7, breach_physics=bp, enable_recorder=recorder,
                      ruleset=ruleset)


def run(sim, n, on_tick=None):
    """n ticks (forced unpause), the per-tick snapshot list."""
    traj = []
    for t in range(n):
        sim.set_paused(False)
        sim.step()
        if on_tick is not None:
            on_tick(t, sim)
        traj.append(capture_snapshot(sim))
    return traj


OCTO = CreatureSpawn("octopus", 13.5, 13.5, heading=30.0)


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    return write_room(tmp_path_factory.mktemp("lvl"), "flood", creatures=(OCTO,))


@pytest.fixture(scope="module")
def dry_room(tmp_path_factory):
    return write_room(tmp_path_factory.mktemp("lvl"), "dry")


def body_of(sim, unit):
    return sim._creature_system.bodies[unit.id]


# ---------------------------------------------------------------------------
# Dormancy + absence transparency
# ---------------------------------------------------------------------------
def test_a_creature_free_sim_is_dormant_and_hashes_as_before(dry_room, tmp_path, monkeypatch):
    """No creature unit: no body, the carrier says 0, the tick digest is the
    pre-P6d1 digest (the same snapshot without the key hashes the same), the
    recorder ring stays empty and the dump has no creature key. Breaks if
    the fold or the ring is not presence-gated."""
    sim = make_sim(dry_room, recorder=True)
    sim.add_unit(Unit("m", 5, 5, team=0))
    snap = run(sim, 3)[-1]
    assert not sim._creature_system.bodies
    assert snap[CS.CREATURE_DIGEST_KEY] == {"n_creatures": 0, "records": ()}
    stripped = {k: v for k, v in snap.items() if k != CS.CREATURE_DIGEST_KEY}
    assert tick_digest(snap) == tick_digest(stripped)
    assert all(s is None for s in sim.recorder.creature_snapshots)
    monkeypatch.chdir(tmp_path)
    data = np.load(sim.recorder.dump("t"))
    assert "creature_state" not in data.files


# ---------------------------------------------------------------------------
# The limp gate
# ---------------------------------------------------------------------------
LIFT_M = 0.2


@pytest.fixture(scope="module")
def limp(room):
    """The octopus released 0.2 m above the deck (crown and every node lifted
    together -- the reference's G2 set-up), 10 s of ticks; per tick the node
    z range, max node speed, the crown and the unit's anchor + facing."""
    sim = make_sim(room)
    (u,) = sim.spawn_creatures(room.creatures)
    b = body_of(sim, u)
    lift = CF.chain_from_m(LIFT_M)
    b.bufs.p[0, ..., 2] += lift
    b.bufs.body[0, 0, AC.BS_Z] += lift
    rows = []
    for t in range(10 * TPS):
        prev = b.bufs.p[0].astype(np.int64)
        sim.set_paused(False)
        sim.step()
        p, v, bd = b.bufs.p[0], b.bufs.v[0], b.bufs.body[0, 0]
        # MOTION: how far every node actually went this tick, as a speed
        moved = np.sqrt(((p.astype(np.int64) - prev).astype(np.float64) ** 2).sum(-1))
        motion = moved.max() * TPS / CF.CHAIN_ONE
        vel = np.sqrt((v.astype(np.float64) ** 2).sum(-1)) / CF.CHAIN_ONE
        rows.append(dict(zmin=int(p[..., 2].min()), zmax=int(p[..., 2].max()), motion=motion,
                         speed=vel[:, 2:].max(), root_speed=vel[:, :2].max(),
                         crown=(int(bd[AC.BS_X]), int(bd[AC.BS_Y])), heading=b.heading_q24,
                         unit=(u.x, u.y, u.facing)))
    return sim, u, b, rows


def test_limp_arms_sink_to_the_deck_and_come_to_rest(limp):
    """G2's dead-arm signature through Simulation: every node at z = 0 within
    ~2 s (2.5 s allowed), none ever below the deck, and at rest after 10 s:
    no node MOVES faster than 1 mm/s over the last second (tick-to-tick
    displacement), and every free node's stored velocity is under 1 mm/s.
    Breaks without gravity, the floor or the drag.

    The two ROOT nodes (0, 1: rigidly attached to the crown) are excluded from
    the stored-velocity bound: they carry a standing ~5 mm/s velocity that the
    attachment cancels every substep (their motion is ~0.3 mm/s, in the
    motion bound) -- a property of the integer reference's attachment +
    rebuild (held at tol 0 by P6c), reported, not a P6d1 target."""
    _, _, _, rows = limp
    assert min(r["zmin"] for r in rows) >= 0, "a node went below the deck"
    landed = next(t for t, r in enumerate(rows) if r["zmax"] == 0)
    assert landed <= int(2.5 * TPS), f"every node on the deck only at tick {landed}"
    last = rows[-TPS:]
    motion = max(r["motion"] for r in last)
    assert motion < 1.0 * MM, f"still moving at {motion / MM:.3f} mm/s"
    assert rows[-1]["speed"] < 1.0 * MM, f"a free node at {rows[-1]['speed'] / MM:.3f} mm/s"
    print(f"limp: every node on the deck at {landed / TPS:.2f} s; last second: max motion "
          f"{motion / MM:.4f} mm/s, max free-node velocity {rows[-1]['speed'] / MM:.4f} mm/s, "
          f"root-node stored velocity {rows[-1]['root_speed'] / MM:.3f} mm/s")


def test_limp_crown_stays_in_the_map_and_off_blocked_tiles(limp):
    """The crown's disc never overlaps a blocked tile and its centre stays in
    the map (the solver's body collision). Breaks if the topology is missing."""
    sim, u, b, rows = limp
    solid = sim.gmap.solid
    h, w = solid.shape
    tile_q = CF.chain_from_m(TILE_M)
    rad = CF.chain_from_m(OCTO_ROW().body.body_radius_m)
    for row in rows:
        cx, cy = row["crown"]
        assert 0 <= cx < w * tile_q and 0 <= cy < h * tile_q
        for iy in range((cy - rad) // tile_q, (cy + rad) // tile_q + 1):
            for ix in range((cx - rad) // tile_q, (cx + rad) // tile_q + 1):
                if solid[iy, ix]:
                    nx = min(max(cx, ix * tile_q), (ix + 1) * tile_q)
                    ny = min(max(cy, iy * tile_q), (iy + 1) * tile_q)
                    assert (cx - nx) ** 2 + (cy - ny) ** 2 >= (rad - 2) ** 2


def OCTO_ROW():
    from simulation.creatures import get_creature
    return get_creature("octopus")


def test_the_unit_anchor_and_facing_are_the_pose(limp):
    """Every tick, ``Unit.x/y/facing`` equal the integer pose through the
    converters -- the only per-tick writer. Breaks on a stale or second
    writer."""
    _, u, _, rows = limp
    F = u.footprint
    for row in rows:
        cx, cy = row["crown"]
        ax, ay = CF.pose_to_unit_anchor(float(CF.chain_to_m(cx)), float(CF.chain_to_m(cy)), F, TILE_M)
        assert row["unit"] == (ax, ay, CF.heading_q24_to_rad(row["heading"]))


# ---------------------------------------------------------------------------
# Handedness and the spawn pose
# ---------------------------------------------------------------------------
def test_heading_handedness_and_the_rest_pose():
    """The pose heading is screen-CCW like Unit.facing (North = -y); the
    solver's yaw is its negative; arm a lies at heading + its row angle,
    radially from the crown rim, straight, at rest length, on the deck; the
    roots are the solver's own rotation of the body-frame points. Breaks if
    any sign or the rim offset changes."""
    import math
    cdef = OCTO_ROW()
    assert CF.PI_Q24 == round(math.pi * 2 ** 24)
    centre = (CF.chain_from_m(5.0), CF.chain_from_m(5.0))
    for heading_deg in (0.0, 90.0, -135.0, 180.0):
        hq = CF.heading_q24_from_deg(heading_deg)
        p, root, th = CS.rest_pose(cdef, centre, hq)
        assert th == CF.wrap_heading_q24(-hq)
        assert (p[..., 2] == 0).all()
        for a, deg in enumerate(cdef.body.arm_angles_deg):
            ang = math.radians(heading_deg + deg)
            d = CF.chain_to_m(p[a, -1, :2].astype(np.int64) - np.array(centre))
            # screen CCW: +angle turns from East toward North, i.e. toward -y
            want = (cdef.body.body_radius_m + cdef.body.arm_length_m) * np.array(
                [math.cos(ang), -math.sin(ang)])
            assert np.allclose(d, want, atol=2e-3), (heading_deg, a, d, want)
            for k in (0, 1):
                rx, ry = CS._rot(th, int(root[a, k, 0]), int(root[a, k, 1]))
                assert (centre[0] + rx, centre[1] + ry) == (int(p[a, k, 0]), int(p[a, k, 1]))
        assert CS.rest_pose(cdef, centre, hq)[0].tobytes() == p.tobytes()   # pure integer


# ---------------------------------------------------------------------------
# Writer routing
# ---------------------------------------------------------------------------
def test_intents_are_the_goal_never_a_move(room):
    """MOVE_DIR and AIM on a creature are recorded as the hub's goal and move
    or turn nothing: the run with intents equals the run without in every
    node and pose count. Breaks if slot 3 or the AIM facing write touches a
    creature."""
    def go(with_intents):
        sim = make_sim(room, ruleset=ContinuousRealtime())
        (u,) = sim.spawn_creatures(room.creatures)
        traj = []
        for t in range(12):
            if with_intents:
                sim.set_move_dir(u.id, 65536, 0, 0)
                sim.set_aim(u.id, 0, -65536)
            sim.set_paused(False)
            sim.step()
            traj.append(CS.decode_record(
                capture_snapshot(sim)[CS.CREATURE_DIGEST_KEY]["records"][0]))
        return traj, body_of(sim, u), u

    a, _, ua = go(False)
    b, body_b, ub = go(True)
    assert body_b.goal_kind == CS.GOAL_MOVE_DIR and body_b.goal_q16 == (65536, 0)
    assert (ua.x, ua.y, ua.facing) == (ub.x, ub.y, ub.facing)
    for da, db in zip(a, b):
        assert np.array_equal(da["p"], db["p"]) and np.array_equal(da["body"], db["body"])


def test_the_wave_push_is_a_body_impulse_never_a_displacement(room):
    """With a pressure gradient across the crown's footprint, the unit push
    row leaves a creature's anchor alone and the creature row adds the push
    to the crown's velocity (x only, away from the high side). Breaks if the
    unit row displaces a creature or the creature row is lost."""
    from simulation import atmosphere_fixed
    from simulation.exchange import apply_creature_wave_impulse, apply_wave_push
    sim = make_sim(room)
    (u,) = sim.spawn_creatures(room.creatures)
    b = body_of(sim, u)
    g = sim.gmap
    tx = u.tile_x
    g.atmosphere[:, :tx + 2] += atmosphere_fixed.quantize_scalar(3.0)   # high on the west
    x0, y0 = u.x, u.y
    v0 = b.bufs.body[0, 0].copy()
    apply_wave_push(sim.units, g, TPS)
    assert (u.x, u.y) == (x0, y0)
    apply_creature_wave_impulse(sim.units, g, TPS, sim._creature_system)
    v1 = b.bufs.body[0, 0]
    assert int(v1[AC.BS_VX]) > int(v0[AC.BS_VX])
    assert int(v1[AC.BS_VY]) == int(v0[AC.BS_VY])


def test_the_round_end_snap_and_wego_rewind_leave_the_creature_alone(room):
    """TwoPhaseWEGO across a round boundary (the tick rewind and the
    round-end snap) gives the SAME creature section, tick for tick, as a
    continuous run: every creature timer is an elapsed-tick counter. Breaks
    if a timer reads sim.tick or the snap rounds the anchor."""
    n = TwoPhaseWEGO().ticks_per_round(make_sim(room)) + 8
    sw = make_sim(room, ruleset=TwoPhaseWEGO())
    sw.spawn_creatures(room.creatures)
    sc = make_sim(room, ruleset=ContinuousRealtime())
    sc.spawn_creatures(room.creatures)
    rewound = False
    for t in range(n):
        for s in (sw, sc):
            s.set_paused(False)
            s.step()
        rewound |= sw.tick < sc.tick
        a = capture_snapshot(sw)[CS.CREATURE_DIGEST_KEY]
        c = capture_snapshot(sc)[CS.CREATURE_DIGEST_KEY]
        assert a == c, f"tick {t}: {diff_trajectories([{CS.CREATURE_DIGEST_KEY: a}], [{CS.CREATURE_DIGEST_KEY: c}])[:3]}"
    assert rewound, "the WEGO run never crossed a round boundary (vacuous)"
    uw = sw.get_unit(0)
    uc = sc.get_unit(0)
    assert (uw.x, uw.y, uw.facing) == (uc.x, uc.y, uc.facing)


# ---------------------------------------------------------------------------
# Humans are unchanged
# ---------------------------------------------------------------------------
def test_humans_are_byte_identical_with_and_without_the_octopus(tmp_path):
    """Two marines walk a straight line in the east half of a wide flooded
    room under MOVE_DIR; the octopus lies limp in the west half. The humans'
    synced records are byte-identical with and without it, and the unit stamp
    planes (dyn_permeability, dyn_heat_atten_q, dyn_wave_absorb) differ ONLY
    on the crown's footprint tiles. Breaks if a creature
    leaks into any human or any tile but its stamp."""
    lvl = write_room(tmp_path, "wide", w=64, h=30)

    def go(with_octo):
        sim = make_sim(lvl, ruleset=ContinuousRealtime())
        m1 = Unit("m1", 40, 6, team=0)
        m2 = Unit("m2", 40, 20, team=0)
        sim.add_unit(m1)
        sim.add_unit(m2)
        octo = sim.spawn_creatures([CreatureSpawn("octopus", 12.5, 13.5)])[0] if with_octo else None
        out = []
        for t in range(48):
            sim.set_move_dir(m1.id, 65536, 0, 0)
            sim.set_move_dir(m2.id, 0, -65536, 0)
            sim.set_paused(False)
            sim.step()
            snap = capture_snapshot(sim)
            humans = [r for r in snap["__unit_state__"]["units"] if r["id"] in (m1.id, m2.id)]
            out.append((repr(humans), (snap["dyn_permeability"].copy(),
                                       snap["dyn_heat_atten_q"].copy(),
                                       snap["dyn_wave_absorb"].copy()),
                        set(octo.occupied_tiles()) if octo else set()))
        return out

    a, b = go(False), go(True)
    moved = a[0][0] != a[-1][0]
    assert moved, "the marines never moved (vacuous)"
    for t, ((ha, oa, _), (hb, ob, foot)) in enumerate(zip(a, b)):
        assert ha == hb, f"tick {t}: a human differs with the octopus present"
        changed = np.zeros(oa[0].shape, dtype=bool)
        for pa, pb in zip(oa, ob):
            changed |= pa != pb
        diff = {(int(x), int(y)) for y, x in np.argwhere(changed)}
        assert diff, f"tick {t}: the crown stamped nothing (vacuous)"
        assert diff <= foot, f"tick {t}: a stamp plane differs off the crown: {sorted(diff - foot)[:5]}"


# ---------------------------------------------------------------------------
# The A/B harness, the record, pickle
# ---------------------------------------------------------------------------
def test_the_ab_harness_locates_a_planted_divergence(room):
    """One chain count flipped on one copy (arm 3, node 7, x, after tick 5):
    diff_trajectories names that creature, arm, node and axis at tick 5 and
    nothing before. Breaks if the creature section is not compared or not
    decoded per arm and node."""
    def go(plant):
        sim = make_sim(room)
        (u,) = sim.spawn_creatures(room.creatures)
        b = body_of(sim, u)

        def on(t, s):
            if plant and t == 5:
                b.bufs.p[0, 3, 7, 0] += 1
        return run(sim, 8, on_tick=on), u.id

    (a, uid), (b, _) = go(False), go(True)
    diffs = diff_trajectories(a, b)
    assert diffs, "the planted count was not seen"
    first = diffs[0]
    assert first.startswith("tick 5: __creature__ unit %d: p differs at arm 3 node 7 axis x" % uid), first
    assert not any(d.startswith(("tick 0:", "tick 4:")) for d in diffs)


def test_the_record_decodes_to_the_body(room):
    """decode_record is the serializer's twin: every field comes back equal
    to the body's own arrays, the heading in the pose's sign, the row
    provenance first. Breaks if the layout and its decoder drift apart."""
    sim = make_sim(room)
    (u,) = sim.spawn_creatures(room.creatures)
    run(sim, 3)
    b = body_of(sim, u)
    d = CS.decode_record(CS.serialize_record(b))
    cdef = OCTO_ROW()
    assert (d["unit_id"], d["row_version"], d["row_hash"]) == (u.id, cdef.version, cdef.row_hash)
    assert np.array_equal(d["p"], b.bufs.p[0]) and np.array_equal(d["v"], b.bufs.v[0])
    assert d["body"][3] == -int(b.bufs.body[0, 0, AC.BS_TH]) == b.heading_q24
    assert np.array_equal(d["wave_clock"], b.bufs.wave_clock[0])
    assert d["rng"] == b.rng and d["arm_ticks"].tolist() == [3] * cdef.body.arms
    sect = CS.creature_carrier(sim._creature_system)
    assert CS.section_bytes(sect).startswith(CS.CREATURE_SECT_PREAMBLE)


def test_the_creature_system_round_trips_through_pickle(room):
    """The creature state is numpy + ints and the solver is stateless: the
    system pickled and restored at tick 12 continues to the very digests of
    an uninterrupted run. (The Simulation as a whole does not pickle today --
    its module handle and the C++ PhysicsEngine; the creature part must not
    add to that.) Breaks if a body holds hidden or unpicklable state."""
    def go(roundtrip):
        sim = make_sim(room)
        sim.spawn_creatures(room.creatures)
        out = []
        for t in range(24):
            if roundtrip and t == 12:
                sim._creature_system = pickle.loads(pickle.dumps(sim._creature_system))
            sim.set_paused(False)
            sim.step()
            out.append(tick_digest(capture_snapshot(sim)))
        return out
    assert go(False) == go(True)


def test_the_recorder_ring_holds_the_section(room, tmp_path, monkeypatch):
    """With a creature the ring holds the tick's section bytes (the digest's
    own serializer) and the dump carries them losslessly. Breaks if the ring
    records a cast or the dump truncates a record."""
    sim = make_sim(room, recorder=True)
    sim.spawn_creatures(room.creatures)
    run(sim, 2)
    want = sim._creature_system.section_bytes()
    assert sim.recorder.creature_snapshots[1] == want
    monkeypatch.chdir(tmp_path)
    data = np.load(sim.recorder.dump("t"))
    assert bytes(data["creature_state"][1]) == want


def test_a_new_wall_reaches_the_chain(room):
    """A topology change (``on_tile_changed``) is re-read into the body's
    walls on the next tick. Breaks if the body keeps a stale `solid`."""
    sim = make_sim(room)
    (u,) = sim.spawn_creatures(room.creatures)
    b = body_of(sim, u)
    run(sim, 1)
    y, x = 2, 2
    sim.gmap.material[y, x] = MAT_HULL
    sim.gmap.on_tile_changed(y, x)
    assert b.bufs.blocked[0, y, x] == 0
    run(sim, 1)
    assert b.bufs.blocked[0, y, x] == 1


# ---------------------------------------------------------------------------
# The spawn door
# ---------------------------------------------------------------------------
def test_the_spawn_door_refuses_what_cannot_stand(tmp_path):
    """Unknown species, a species without a creature row, a footprint off
    the map or on a wall, an arm laid into a wall: each refused, naming it."""
    from simulation.match import creature_spawn_from_dict
    with pytest.raises(ValueError, match="not a species"):
        creature_spawn_from_dict({"species": "kraken", "x": 1, "y": 1}, "t")
    with pytest.raises(ValueError, match="no creature row"):
        creature_spawn_from_dict({"species": "human", "x": 1, "y": 1}, "t")
    with pytest.raises(ValueError, match="heading"):
        creature_spawn_from_dict({"species": "octopus", "x": 1, "y": 1, "heading": 270}, "t")
    with pytest.raises(ValueError, match="unknown key"):
        creature_spawn_from_dict({"species": "octopus", "x": 1, "y": 1, "z": 0}, "t")
    for i, (row, walls, why) in enumerate((
            (CreatureSpawn("octopus", 28.5, 13.5), (), "leaves"),
            (CreatureSpawn("octopus", 13.5, 13.5), ((14, 14),), "open floor"),
            (CreatureSpawn("octopus", 13.5, 13.5), ((21, 12),), "arm 0"))):
        with pytest.raises(ValueError, match=why):
            write_room(tmp_path, f"bad{i}", creatures=(row,), walls=walls)


def test_spawn_refuses_without_the_chain_solver(room):
    """No compiled solver, no creature (no Python fallback chain)."""
    sim = make_sim(room)
    sim._bp = type("NoChain", (), {})()
    with pytest.raises(RuntimeError, match="arm_chain_tick"):
        sim.spawn_creatures(room.creatures)


def test_written_rows_load_and_spawn_as_written(room):
    """write_creatures -> the loader -> spawn_creatures: the row comes back
    equal, the unit is C-<slot> on CREATURE_TEAM with the row's footprint,
    standing on the quantized spawn pose."""
    assert room.creatures == [OCTO]
    sim = make_sim(room)
    (u,) = sim.spawn_creatures(room.creatures)
    assert (u.name, u.team, u.footprint) == ("C-0", CREATURE_TEAM, OCTO_ROW().body.footprint)
    assert abs(u.x - OCTO.x) < 1e-6 and abs(u.y - OCTO.y) < 1e-6
    assert abs(u.facing - np.radians(OCTO.heading)) < 1e-6


def test_a_match_spawns_its_creatures(tmp_path):
    """match.toml's [[creature]] rows go through the match writer and the
    loader's door and are spawned by spawn_match after the teams."""
    idle = MatchTeam(n=1, loadout="rifleman", env_profile="default", controller="idle")
    rooms = [SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=1),
             SpawnRoom(team=1, x=50, y=20, w=8, h=8, max_per_team=1)]
    spec = MatchSpec(teams=(idle, idle), seed=3,
                     creatures=(CreatureSpawn("octopus", 28.5, 13.5, heading=-90.0),))
    lvl = write_room(tmp_path, "m", w=64, h=30, rooms=rooms, match=spec)
    assert lvl.match.creatures == spec.creatures
    sim = Simulation(lvl, seed=3, breach_physics=bp, enable_recorder=False)
    units = sim.spawn_match()
    assert [u.name for u in units] == ["T0-0", "T1-0", "C-0"]
    assert sim._creature_system.bodies.keys() == {units[-1].id}
