"""#215 -- a thrown grenade bounces off walls (Erik, 2026-11-17: "grenades
should not pass through walls"; the rule agreed on #215).

The flight is a chain of straight legs (``combat.Projectile``): the first is
the throw, bit for bit; each tick the piece flown is walked tile by tile
(``combat.flight_walk``) against the rounds' march's stop read (solid, or off
the grid) as the map stands that tick; a leg that enters a blocked tile ends
at the last point in the free tile and a mirrored leg starts there, its
length the distance left times ``bounce_restitution`` rounded once in Q16; a
leg under a quarter tile, or a wall after ``bounce_max`` bounces, rests it;
at fuse-out the rest of the flight is resolved at once and the grenade
detonates where it rests (``combat.detonation_tile``), never on a blocked
tile.

Each test names the property it protects and the change that must break it.
Host == device every tick and env i of a batch == env i alone are the CUDA
gate's (``tests/cuda_unit_objects_check.py`` PART FLIGHT / BOUNCE, wrapper
``tests/test_cuda_unit_objects.py``).
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import combat, unit_fixed  # noqa: E402
from simulation.orders import ORDER_GRENADE  # noqa: E402

ONE = unit_fixed.FP_ONE
RQ = unit_fixed.quantize_scalar(0.4)      # the shipped restitution's Q16
BMAX = 4


class _Materials:
    blocks_thrown = np.array([False, True])      # 0 open, 1 blocks a throw only


class Map:
    """A bare map for the law: ``solid`` and the material whose
    ``blocks_thrown`` the walk reads (material 1 stops a throw, not a round)."""
    materials = _Materials()

    def __init__(self, h=40, w=60):
        self.material = np.zeros((h, w), dtype=np.int32)
        self.solid = np.zeros((h, w), dtype=bool)
        self.solid[0, :] = self.solid[-1, :] = True
        self.solid[:, 0] = self.solid[:, -1] = True


def _grenade(sx, sy, tx, ty, fuse=1000, thrown=0):
    return combat.Projectile(ORDER_GRENADE, sx, sy, tx, ty, 0.0, thrown,
                             fuse_ticks=fuse)


def _old_position(p0, tick, tps):
    """The pre-#215 ``update_position``: the straight line from the throw's
    start to its target, from the thrown tick."""
    el = tick - p0["thrown"]
    dx, dy = p0["tx"] - p0["sx"], p0["ty"] - p0["sy"]
    d = unit_fixed.isqrt(dx * dx + dy * dy)
    if d < combat._FLIGHT_EPS_Q or el * p0["v"] >= d * tps:
        return p0["tx"], p0["ty"]
    return (p0["sx"] + unit_fixed.div_round(dx * el * p0["v"], tps * d),
            p0["sy"] + unit_fixed.div_round(dy * el * p0["v"], tps * d))


def _fly_to_fuse(g, gmap, rq=RQ, bmax=BMAX, on_tick=None):
    """Slot 2's law for one grenade from its thrown tick to its fuse-out:
    ``fly`` every tick, ``settle`` at the fuse-out; the detonation tile."""
    t = g.thrown_tick
    while True:
        if on_tick is not None:
            on_tick(t)
        g.fly(gmap, t, rq, bmax)
        if t >= g.get_detonate_tick():
            g.settle(gmap, t, rq, bmax)
            return t, combat.detonation_tile(gmap, g.pos_x_q, g.pos_y_q)
        t += 1


def _tps():
    from config import CFG
    return int(CFG.clock.ticks_per_second)


# ---------------------------------------------------------------------------
# Rule 1: a throw that meets no wall lands where and when it landed before
# ---------------------------------------------------------------------------
def test_a_leg_that_meets_no_wall_flies_and_lands_as_the_pre_215_law():
    """Protects the leg law of rule 1: from a given start, a leg whose
    flight enters no blocked tile has, on every tick, the pre-#215
    straight-line position, rests on its target and detonates on its target
    tile on its fuse tick, approach its start's tile -- a sweep of 72
    directions, 6 reaches and 3 fuses (before the landing, at it, after it)
    in an open room. (The throw's START moved to the hand, Erik 2026-11-18:
    the in-game test below holds the detonation tile and tick.) Breaks when
    the first leg's arithmetic, the leg tick, the settle of an unobstructed
    flight, the detonation tile or the approach moves."""
    gmap = Map(80, 80)
    tps = _tps()
    n = 0
    for k in range(72):
        a = 2 * math.pi * k / 72
        for reach in (0.05, 0.4, 3.0, 7.5, 18.0, 30.0):
            for fuse in (2, 14, 40):
                sx, sy = 40, 40
                tx = sx + 0.5 + unit_fixed.dequantize_scalar(
                    unit_fixed.quantize_scalar(math.cos(a) * reach))
                ty = sy + 0.5 + unit_fixed.dequantize_scalar(
                    unit_fixed.quantize_scalar(math.sin(a) * reach))
                g = _grenade(sx, sy, tx, ty, fuse=fuse, thrown=100)
                p0 = dict(sx=g.start_x_q, sy=g.start_y_q, tx=g.target_x_q,
                          ty=g.target_y_q, thrown=100, v=g.travel_speed_q)

                def check(t, g=g, p0=p0):
                    if t > 100:      # the position the previous tick left
                        assert (g.pos_x_q, g.pos_y_q) == _old_position(
                            p0, t - 1, tps), (k, reach, fuse, t)
                tick, tile = _fly_to_fuse(g, gmap, on_tick=check)
                assert tick == 100 + fuse
                assert g.bounces == 0
                assert (g.pos_x_q, g.pos_y_q) == (p0["tx"], p0["ty"])
                assert tile == (unit_fixed.shr_trunc(p0["ty"]),
                                unit_fixed.shr_trunc(p0["tx"]))
                assert g.approach_tile() == (sy, sx)
                n += 1
    assert n == 72 * 6 * 3


# ---------------------------------------------------------------------------
# A wall in the way
# ---------------------------------------------------------------------------
def _walled(axis):
    """A room split by a wall: column 30 (``axis`` "x") or row 20 ("y")."""
    gmap = Map(40, 60)
    if axis == "x":
        gmap.solid[:, 30] = True
    else:
        gmap.solid[20, :] = True
    return gmap


@pytest.mark.parametrize("axis", ["x", "y"])
def test_a_wall_in_the_way_mirrors_the_leg_and_keeps_the_blast_on_the_throwers_side(axis):
    """Protects rules 2, 5 and 6 on both axes: a sweep of directions and
    reaches whose straight flight would cross the wall -- every one meets
    it, its first bounce flips exactly the component across the wall, the
    new leg starts in the free tile next to the wall and is the distance the
    old leg had LEFT times the restitution rounded once in Q16 (its vector's
    length within one Q16 unit of that), and it detonates on the thrower's
    side on a free tile. Breaks when the walk misses a wall, flips the
    wrong component, stops inside the wall or a tile short, when the bounce
    length is taken from the whole leg or rounded twice, or when the
    detonation can land beyond the wall or on it."""
    gmap = _walled(axis)
    met = 0
    for k in range(-8, 9):
        for reach in (12.0, 17.5, 23.0, 29.0):
            a = math.radians(6.5 * k)
            u, v = math.cos(a), math.sin(a)       # along x, across y
            if axis == "y":
                u, v = v, u
            sx, sy = 24, 14
            g = _grenade(sx, sy, sx + 0.5 + u * reach, sy + 0.5 + v * reach,
                         fuse=200, thrown=0)
            if axis == "x" and g.target_x_q < 30 * ONE or \
                    axis == "y" and g.target_y_q < 20 * ONE:
                continue
            seen = []
            orig = g.bounce

            def spy(px, py, fx, fy, tick, rq, bmax, g=g, orig=orig, seen=seen):
                before = (g.start_x_q, g.start_y_q, g.target_x_q, g.target_y_q)
                left = unit_fixed.isqrt((before[2] - px) ** 2 + (before[3] - py) ** 2)
                length = orig(px, py, fx, fy, tick, rq, bmax)
                seen.append((before, (px, py, fx, fy), left, length,
                             (g.start_x_q, g.start_y_q, g.target_x_q, g.target_y_q)))
                return length
            g.bounce = spy
            _, tile = _fly_to_fuse(g, gmap)
            assert seen, (k, reach)
            met += 1
            before, (px, py, fx, fy), left, length, after = seen[0]
            assert (fx, fy) == ((True, False) if axis == "x" else (False, True))
            if axis == "x":
                assert px == 30 * ONE - 1 and 29 * ONE <= px
            else:
                assert py == 20 * ONE - 1
            assert length == unit_fixed.mul_q16_round(left, RQ)
            ndx, ndy = after[2] - after[0], after[3] - after[1]
            assert abs(unit_fixed.isqrt(ndx * ndx + ndy * ndy) - length) <= 1
            odx, ody = before[2] - before[0], before[3] - before[1]
            if axis == "x":
                assert ndx < 0 < odx and (ndy > 0) == (ody > 0) or ody == 0
            else:
                assert ndy < 0 < ody and (ndx > 0) == (odx > 0) or odx == 0
            assert not gmap.solid[tile]
            assert (tile[1] < 30) if axis == "x" else (tile[0] < 20)
    assert met >= 40


def test_an_exact_corner_flips_both_components():
    """Protects rule 2's corner: a 45-degree flight from a tile corner
    crosses the lattice's corners exactly; into a blocked diagonal tile it
    stops at the corner of its free tile and both components flip (the
    flight comes straight back). Breaks when a tie of the crossing
    parameters is broken to one axis, or the corner's stop point is not the
    free tile's."""
    gmap = Map(40, 40)
    gmap.solid[15, 15] = True
    g = _grenade(10, 10, 20.0, 20.0, fuse=200)
    hit = combat.flight_walk(gmap, g.start_x_q, g.start_y_q, g.target_x_q,
                             g.target_y_q)
    assert hit == (15 * ONE - 1, 15 * ONE - 1, True, True)
    g.bounce(*hit, 5, RQ, BMAX)
    assert g.target_x_q < g.start_x_q and g.target_y_q < g.start_y_q
    assert g.target_x_q - g.start_x_q == g.target_y_q - g.start_y_q


# ---------------------------------------------------------------------------
# Restitution 0 and 1, bounce_max
# ---------------------------------------------------------------------------
def test_restitution_zero_drops_at_the_wall():
    """Protects the dial's low end: at restitution 0 the first wall rests
    the grenade at the point it hit (no new leg, no bounce counted), and it
    detonates in the free tile against the wall. Breaks when a zero-length
    leg is flown, or the rest point is not the hit point."""
    gmap = _walled("x")
    g = _grenade(24, 14, 40.5, 14.5, fuse=100)
    _, tile = _fly_to_fuse(g, gmap, rq=0)
    assert g.bounces == 0
    assert (g.start_x_q, g.start_y_q) == (g.target_x_q, g.target_y_q)
    assert g.pos_x_q == 30 * ONE - 1
    assert tile == (14, 29)


def test_restitution_one_mirrors_in_full_and_bounce_max_ends_it():
    """Protects the dial's high end and the bounce count: at restitution 1
    every bounce's leg is the whole distance left (a mirror), and down a
    corridor three tiles wide the grenade bounces exactly ``bounce_max``
    times, the next wall resting it where it hits. Breaks when restitution
    1 shortens a leg, or bounce_max is off by one either way."""
    gmap = Map(40, 40)
    gmap.solid[:, 10] = True
    gmap.solid[:, 13] = True                        # a corridor two tiles wide
    g = _grenade(11, 5, 11.5 + 29.0 * math.cos(0.6), 5.5 + 29.0 * math.sin(0.6),
                 fuse=400)
    lengths = []
    orig = g.bounce

    def spy(px, py, fx, fy, tick, rq, bmax):
        left = unit_fixed.isqrt((g.target_x_q - px) ** 2 + (g.target_y_q - py) ** 2)
        r = orig(px, py, fx, fy, tick, rq, bmax)
        lengths.append((left, r))
        return r
    g.bounce = spy
    _fly_to_fuse(g, gmap, rq=ONE, bmax=BMAX)
    assert g.bounces == BMAX
    assert len(lengths) == BMAX + 1
    assert all(r == left for left, r in lengths[:BMAX])
    assert lengths[BMAX][1] == 0 and lengths[BMAX][0] >= 4 * ONE   # not a short leg
    assert (g.start_x_q, g.start_y_q) == (g.target_x_q, g.target_y_q)
    assert 11 * ONE <= g.pos_x_q < 13 * ONE


def test_a_bounced_leg_under_a_quarter_tile_rests():
    """Protects rule 3's short leg: a wall met with less than a quarter tile
    of flight left over the restitution rests the grenade at the wall.
    Breaks when the quarter-tile threshold moves or is skipped."""
    gmap = _walled("x")
    g = _grenade(24, 14, 30.55, 14.5, fuse=100)     # 0.55 left: 0.22 after
    _fly_to_fuse(g, gmap)
    assert g.bounces == 0
    assert g.pos_x_q == 30 * ONE - 1


# ---------------------------------------------------------------------------
# The map as it stands each tick
# ---------------------------------------------------------------------------
def test_a_door_closing_mid_flight_stops_the_grenade():
    """Protects rule 4 (closing): a grenade flying at an open gap whose
    tiles turn solid while it is in the air meets them and bounces back.
    Breaks when the wall is asked once at the throw, or only of the leg's
    target."""
    gmap = _walled("x")
    gmap.solid[14, 30] = False                      # the gap, open at the throw
    g = _grenade(20, 14, 40.5, 14.5, fuse=60)

    def close(t):
        if t == 3:
            gmap.solid[14, 30] = True
    _, tile = _fly_to_fuse(g, gmap, on_tick=close)
    assert g.bounces >= 1
    assert tile[1] < 30


def test_a_wall_destroyed_mid_flight_lets_the_grenade_through():
    """Protects rule 4 (opening): a grenade flying at a wall tile that is
    destroyed while it is in the air flies through and lands on its
    target, unbounced. Breaks when the walk reads a stale map."""
    gmap = _walled("x")
    g = _grenade(20, 14, 40.5, 14.5, fuse=60)

    def open_(t):
        if t == 3:
            gmap.solid[14, 30] = False
    _, tile = _fly_to_fuse(g, gmap, on_tick=open_)
    assert g.bounces == 0
    assert tile == (14, 40)


def test_a_fuse_out_mid_leg_never_detonates_beyond_the_wall():
    """Protects rule 5: when the fuse runs out while the grenade is still
    short of the wall, the rest of its flight is resolved at once by the
    same walk -- it meets the wall and rests on the thrower's side -- for a
    sweep of fuses and reaches. Breaks when the fuse-out detonates at the
    leg's target (the pre-#215 rule) or where the grenade happens to be
    without the settle when its target lies beyond the wall."""
    gmap = _walled("x")
    for fuse in range(0, 9):
        for reach in (8.0, 15.0, 22.0, 29.0):
            g = _grenade(22, 14, 22.5 + reach, 14.5, fuse=fuse)
            _, tile = _fly_to_fuse(g, gmap)
            assert tile[1] < 30, (fuse, reach, tile)
            assert not gmap.solid[tile]
            if 22.5 + reach >= 30:
                assert g.bounces >= 1 or g.start_x_q == g.target_x_q


def test_a_grenade_resting_where_a_wall_now_stands_detonates_beside_it():
    """Protects rule 6: a grenade resting on a tile that became solid (a
    door closed on it) detonates on the first free 4-neighbour (north,
    south, west, east), never on the wall. Breaks when the detonation tile
    is the rest tile unchecked."""
    gmap = Map(20, 20)
    gmap.solid[5, 7] = True
    assert combat.detonation_tile(gmap, 7 * ONE + 5, 5 * ONE + 9) == (4, 7)
    gmap.solid[4, 7] = True
    assert combat.detonation_tile(gmap, 7 * ONE + 5, 5 * ONE + 9) == (6, 7)
    assert combat.detonation_tile(gmap, 3 * ONE, 3 * ONE) == (3, 3)


def test_the_walk_enters_every_tile_the_segment_crosses_in_order():
    """Protects the walk's tile order (the grid traversal): on random
    segments, a wall planted on any tile the exact segment passes through
    the interior of is met there first (no tile skipped, none before its
    turn). Breaks when a step is taken on the wrong axis or the walk
    leaps a tile."""
    rng = np.random.default_rng(215)
    for _ in range(300):
        ax, ay = (int(v) for v in rng.integers(4 * ONE, 36 * ONE, size=2))
        bx, by = (int(v) for v in rng.integers(4 * ONE, 36 * ONE, size=2))
        # the tiles the real segment visits, exactly: the floor of the
        # point between each two consecutive grid-line crossings
        ts = {Fraction(0), Fraction(1)}
        for a0, b0 in ((ax, bx), (ay, by)):
            if a0 != b0:
                lo, hi = sorted((a0, b0))
                for g in range(lo // ONE + 1, hi // ONE + 1):
                    ts.add(Fraction(g * ONE - a0, b0 - a0))
        ts = sorted(ts)
        tiles = []
        for s0, s1 in zip(ts, ts[1:]):
            s = (s0 + s1) / 2
            t = (math.floor((ax + (bx - ax) * s) / ONE),
                 math.floor((ay + (by - ay) * s) / ONE))
            if not tiles or tiles[-1] != t:
                tiles.append(t)
        if len(tiles) < 3:
            continue
        pick = tiles[len(tiles) // 2]
        gmap = Map(40, 40)
        gmap.solid[pick[1], pick[0]] = True
        hit = combat.flight_walk(gmap, ax, ay, bx, by)
        assert hit is not None
        px, py = hit[0], hit[1]
        here = (px >> 16, py >> 16)
        assert here == tiles[tiles.index(pick) - 1], (ax, ay, bx, by, pick, here)


def test_the_restitution_is_refused_outside_0_to_1_by_name():
    """Protects the dial's door: ``bounce_restitution`` outside 0..1 and a
    negative or non-integer ``bounce_max`` are refused at the table's load,
    naming the column. Breaks when the door accepts them."""
    from simulation.weapons import WeaponTable
    base = {"archetype": "lobbed", "ammo_family": "hand_grenade"}
    for bad in (-0.1, 1.5, True):
        with pytest.raises(ValueError, match="bounce_restitution"):
            WeaponTable({"g": dict(base, bounce_restitution=bad)}, 24)
    for bad in (-1, 2.5):
        with pytest.raises(ValueError, match="bounce_max"):
            WeaponTable({"g": dict(base, bounce_max=bad)}, 24)
    t = WeaponTable({"g": dict(base, bounce_restitution=0.4, bounce_max=4)}, 24)
    assert t.by_name["g"].bounce_restitution_q == RQ


# ---------------------------------------------------------------------------
# In the game: the sim's slot 2, the 2c finding, determinism
# ---------------------------------------------------------------------------
def _sim(tilemap, seed=215):
    import breach_physics as bp
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    lv = LevelData(name="p215", version="3", path=Path("."), tilemap=tilemap,
                   tile_size_m=1.0, diffuse_path=Path("."), entities=[])
    sim = Simulation(lv, seed=seed, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _walled_tiles():
    from simulation.materials import MAT_AIR, MAT_HULL
    tm = np.full((40, 60), MAT_HULL, dtype=np.int32)
    tm[1:39, 1:59] = MAT_AIR
    tm[2:38, 30] = MAT_HULL
    return tm


def _play(sim, throws, ticks):
    """Throw from units standing at ``throws`` ((x, y, facing_q, reach_t,
    fuse) each) on tick 1; play ``ticks``; return every execute_payload
    call of slot 2 as (tick, tile, approach, owner) and the per-tick
    grenade positions."""
    from simulation import simulation as simmod
    from simulation.turning import place
    from simulation.unit import Unit
    us = []
    for i, (x, y, f, _r, _fz) in enumerate(throws):
        u = Unit(f"G{i}", x=float(x), y=float(y), team=0)
        sim.add_unit(u)
        place(u, f)
        u.current_hp = 1.0e6
        us.append(u)
    calls, track = [], []
    orig = simmod.execute_payload

    def rec(gmap, q, units, fy, fx, payload, *a, **kw):
        calls.append((sim.tick, (fy, fx), kw.get("approach"), kw.get("owner_id")))
        return orig(gmap, q, units, fy, fx, payload, *a, **kw)
    simmod.execute_payload = rec
    try:
        for k in range(ticks):
            if k == 1:
                for u, (_x, _y, _f, r, fz) in zip(us, throws):
                    sim.throw_grenade_intent(u.id, ONE, 0, fz, reach_q=int(r * ONE))
            sim.step()
            sim.set_paused(False)
            track.append(tuple((p.pos_x_q, p.pos_y_q, p.bounces)
                               for p in sim.projectiles))
    finally:
        simmod.execute_payload = orig
    return calls, track


def test_in_the_game_a_throw_at_a_wall_detonates_on_the_throwers_side():
    """Protects the rule end to end through the sim's slot 2 (the throw
    verb, the wind-up, ``_update_grenades``' two passes, the payload call):
    throws east at a wall from three rows, two fuses each, all detonate west
    of it on a free tile; the throw that cannot reach the wall detonates on
    its target tile as before #215, its approach the thrower's tile. Breaks
    when slot 2 detonates at the throw's target or skips the walk."""
    sim = _sim(_walled_tiles())
    throws = [(20, 5, 0, 20.0, 40), (20, 15, 0, 25.0, 12), (20, 25, 0, 14.0, 6),
              (8, 32, 0, 5.0, 30)]
    calls, _ = _play(sim, throws, 90)
    assert len(calls) == len(throws)
    for (_t, (fy, fx), _ap, _o) in calls:
        assert fx < 30 and not sim.gmap.solid[fy, fx]
    short = [c for c in calls if c[3] == sim.units[-1].id][0]
    u = sim.units[-1]
    cx, cy = u.center_tile_x(), u.center_tile_y()
    assert short[1] == (cy, cx + 5)                # 5 tiles east, unbounced
    hand = sim.weapons_tables.weapons.by_name["hand_grenade"]
    # the thrower where it stood at the release (its spawn: every throw is
    # released on the same tick, before any detonation). #215 (body
    # temperature) moved the blasts' wind, which shoves it later on: its
    # position at the END of the play is not where it threw from.
    x0, y0 = throws[-1][0], throws[-1][1]
    sx, sy = combat.release_point(sim.gmap, x0 * ONE, y0 * ONE, u.footprint, ONE,
                                  0, hand.release_forward_q, hand.release_side_q)
    assert short[2] == (sy >> 16, sx >> 16)        # the leg's start: the hand


def test_in_the_game_a_fuse_out_mid_leg_detonates_where_the_rest_of_its_flight_ends():
    """Protects rule 5 in the sim's slot 2: a grenade whose fuse runs out
    while it is still flying at a wall far away detonates where the rest of
    its flight, resolved at once (bounce included), rests -- the law's own
    flight of the same throw -- and NOT where it happened to be on the
    fuse tick. Breaks when slot 2 skips the settle or detonates elsewhere."""
    from simulation import simulation as simmod
    sim = _sim(_walled_tiles())
    seen = []
    orig = simmod.Projectile.settle

    def spy(p, gmap, tick, rq, bmax):
        import copy
        law = copy.deepcopy(p)                      # the law's own flight, apart
        orig(law, gmap, tick, rq, bmax)
        seen.append(((p.pos_x_q, p.pos_y_q), (p.target_x_q, p.target_y_q),
                     combat.detonation_tile(gmap, law.pos_x_q, law.pos_y_q)))
        return orig(p, gmap, tick, rq, bmax)
    simmod.Projectile.settle = spy
    try:
        calls, _ = _play(sim, [(20, 15, 0, 25.0, 3)], 30)
    finally:
        simmod.Projectile.settle = orig
    assert len(calls) == 1 and len(seen) == 1
    at_fuse, target, want = seen[0]
    assert at_fuse != target                        # still flying at the fuse
    assert calls[0][1] == want
    assert want != (at_fuse[1] >> 16, at_fuse[0] >> 16)
    assert want[1] < 30


def test_same_seed_same_flight():
    """Protects determinism: the same throws in the same map from the same
    seed give the same positions, bounces and detonations every tick.
    Breaks when the walk or the bounce reads anything but the map and the
    grenade's own integers (an iteration order, a float)."""
    throws = [(20, 5, 0, 29.0, 50), (20, 15, 0, 25.0, 33), (20, 25, 0, 17.0, 20)]
    a = _play(_sim(_walled_tiles()), throws, 70)
    b = _play(_sim(_walled_tiles()), throws, 70)
    assert a == b
    assert any(b2 for tr in a[1] for (_x, _y, b2) in tr)       # it did bounce


def test_the_2c_finding_no_longer_crosses_the_wall(tmp_path):
    """Protects the 2c regression (docs/rl_agent_rung2_report_2c_2026-11-17.md
    section 7, point 1): in ``range_corner``, the frag thrown from (14, 13)
    in the west room at the reach that landed it at (34, 35) in the far room
    no longer detonates east of the first wall (column 19) -- either way the
    tile pair is read. Breaks when the flight stops asking the walls."""
    import gen_training_rooms as TR
    import level_loader
    from simulation.ruleset import ContinuousRealtime
    from simulation import Simulation
    import breach_physics as bp
    TR.write_all(tmp_path)
    lv = level_loader.load("range_corner", str(tmp_path))
    sim = Simulation(lv, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    hand = sim.weapons_tables.weapons.by_name["hand_grenade"]
    for (x0, y0), (x1, y1) in (((14, 13), (34, 35)), ((13, 14), (35, 34))):
        g = _grenade(x0, y0, x1 + 0.5, y1 + 0.5, fuse=48)
        _, tile = _fly_to_fuse(g, sim.gmap, rq=hand.bounce_restitution_q,
                               bmax=hand.bounce_max)
        assert tile[1] < 19, ((x0, y0), tile)
        assert not sim.gmap.solid[tile]


# ---------------------------------------------------------------------------
# The throw leaves the hand (Erik, 2026-11-18)
# ---------------------------------------------------------------------------
def test_the_grenade_leaves_the_hand_turned_with_the_facing():
    """Protects the release point: for a sweep of facings and positions the
    start is the thrower's Q16 centre plus the hand's offset -- forward
    along the throw, to the right of it by the side value (negative: the
    LEFT hand) -- within one Q16 unit of the exact rotation, and on the
    thrower's left for the shipped row. Breaks when the offset is not
    turned with the facing, the side is mirrored, the centre is the tile
    corner, or the rounding is not once."""
    from simulation.unit_fixed import cos_q16, sin_q16
    gmap = Map(80, 80)
    f_q, s_q = unit_fixed.quantize_scalar(1.362), unit_fixed.quantize_scalar(-0.864)
    for k in range(36):
        fq = unit_fixed.quantize_scalar(2 * math.pi * k / 36)
        dx, dy = cos_q16(fq), -sin_q16(fq)
        x_q, y_q = 30 * ONE + 977 * k, 31 * ONE + 13 * k
        sx, sy = combat.release_point(gmap, x_q, y_q, 3, dx, dy, f_q, s_q)
        cx, cy = x_q + 3 * ONE / 2, y_q + 3 * ONE / 2
        ux, uy = dx / ONE, dy / ONE
        ex = cx + (f_q * ux - s_q * uy)
        ey = cy + (f_q * uy + s_q * ux)
        assert abs(sx - ex) <= 1 and abs(sy - ey) <= 1, k
        # the hand is on the thrower's LEFT: right = (-uy, ux)
        assert (sx - cx) * -uy + (sy - cy) * ux < 0


def test_the_hand_never_starts_beyond_a_wall():
    """Protects the release point's walk: a thrower whose hand would reach
    past a wall (standing against it, facing it) starts the grenade at the
    last free point on its own side, not counted as a bounce; the throw
    then meets the wall and detonates on that side. Breaks when the piece
    from the centre to the hand is not walked."""
    gmap = _walled("x")                                  # the wall: column 30
    f_q, s_q = unit_fixed.quantize_scalar(1.8), unit_fixed.quantize_scalar(-0.4)
    x_q, y_q = 27 * ONE + 30000, 14 * ONE               # centre x = 28.96
    sx, sy = combat.release_point(gmap, x_q, y_q, 3, ONE, 0, f_q, s_q)
    assert sx == 30 * ONE - 1 and not gmap.solid[sy >> 16, sx >> 16]
    g = _grenade(unit_fixed.dequantize_scalar(sx), unit_fixed.dequantize_scalar(sy),
                 40.5, 15.5, fuse=40)
    assert g.bounces == 0
    _, tile = _fly_to_fuse(g, gmap)
    assert tile[1] < 30


def test_a_grille_stops_a_grenade_and_still_passes_a_round():
    """Protects Erik's 2026-11-18 ruling: a grille (``blocks_thrown``, not
    solid) stops a thrown grenade -- it bounces and detonates on the
    thrower's side -- while the rounds' read (``solid``, the march's;
    ``has_los(physical=True)``) still passes it. Breaks when the flight
    reads ``solid`` alone, or the grille's row loses ``blocks_thrown``, or
    the column leaks into the rounds' read."""
    from simulation.materials import MAT_AIR, MAT_GRILLE, MAT_HULL
    tm = np.full((40, 60), MAT_HULL, dtype=np.int32)
    tm[1:39, 1:59] = MAT_AIR
    tm[2:38, 30] = MAT_GRILLE
    sim = _sim(tm)
    gmap = sim.gmap
    assert not gmap.solid[14, 30] and combat.flight_blocked(gmap, 14, 30)
    assert gmap.has_los(14, 20, 14, 40, physical=True)        # a round passes
    assert combat.flight_plane(gmap)[14, 30] and not combat.flight_plane(gmap)[14, 29]
    hand = sim.weapons_tables.weapons.by_name["hand_grenade"]
    g = _grenade(20, 14, 40.5, 14.5, fuse=60)
    _, tile = _fly_to_fuse(g, gmap, rq=hand.bounce_restitution_q,
                           bmax=hand.bounce_max)
    assert g.bounces >= 1 and tile[1] < 30


def test_in_the_game_a_wall_free_throw_detonates_where_and_when_main_did():
    """Protects rule 1 as it reads after the hand (Erik 2026-11-18): every
    throw of a sweep (facings x reaches, an open room) whose leg from the
    hand meets no wall detonates on the tile pre-#215 main detonated it --
    the target, ``centre tile + half + direction x reach`` truncated, which
    the hand did not move -- on the thrown tick + fuse. Its per-tick
    positions DIFFER from main's (the flight starts at the hand, not at the
    centre tile's corner). Breaks when the target moves with the start, or
    a wall-free throw lands elsewhere or later."""
    from simulation import simulation as simmod
    from simulation.materials import MAT_AIR, MAT_HULL
    from simulation.turning import place
    from simulation.unit import Unit
    tm = np.full((90, 90), MAT_HULL, dtype=np.int32)
    tm[1:89, 1:89] = MAT_AIR
    sim = _sim(tm)
    throws = []
    for k in range(12):
        u = Unit(f"R{k}", x=40.0 + (k % 4) * 0.37, y=40.0 + (k // 4) * 0.21, team=0)
        sim.add_unit(u)
        place(u, unit_fixed.quantize_scalar(2 * math.pi * k / 12))
        u.current_hp = 1.0e6
        throws.append((u, 3.0 + 2.25 * k, 20 + k))
    rel, calls = [], []
    o_rel, o_ex = simmod.Simulation._release_grenade, simmod.execute_payload

    def r(s, u, kind, fuse, reach_q, dx_q, dy_q):
        n0 = len(s.projectiles)
        o_rel(s, u, kind, fuse, reach_q, dx_q, dy_q)
        for p in s.projectiles[n0:]:
            cx, cy = u.center_tile_x(), u.center_tile_y()
            half = ONE >> 1
            main_t = ((cy << 16) + half + unit_fixed.mul_q16_round(dy_q, reach_q),
                      (cx << 16) + half + unit_fixed.mul_q16_round(dx_q, reach_q))
            rel.append((p, s.tick, (main_t[0] >> 16, main_t[1] >> 16),
                        (cy << 16, cx << 16)))

    def ex(gmap, q, units, fy, fx, payload, *a, **kw):
        calls.append((sim.tick, kw.get("owner_id"), (fy, fx)))
        return o_ex(gmap, q, units, fy, fx, payload, *a, **kw)
    simmod.Simulation._release_grenade, simmod.execute_payload = r, ex
    try:
        for k in range(80):
            if k == 1:
                for u, reach, fuse in throws:
                    sim.throw_grenade_intent(u.id, ONE, 0, fuse,
                                             reach_q=int(reach * ONE))
            sim.step()
            sim.set_paused(False)
    finally:
        simmod.Simulation._release_grenade, simmod.execute_payload = o_rel, o_ex
    assert len(rel) == len(throws) and len(calls) == len(throws)
    moved = 0
    for p, t0, main_tile, main_start in rel:
        assert p.bounces == 0
        got = [c for c in calls if c[1] == p.owner_id][0]
        assert got[2] == main_tile and got[0] == t0 + p.fuse_ticks
        moved += (p.start_y_q, p.start_x_q) != main_start
    assert moved == len(rel)                      # every start is the hand now
