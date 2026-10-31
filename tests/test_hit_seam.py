"""The hit seam (#106 P6d3): ``Unit.hit_tiles()`` -- what a bullet or a beam can
strike -- against ``occupied_tiles()``, the footprint everything else reads.

Design: docs/octopus_p6_sim_port_design_2026-10-30.md section 4.2. Each test
names the property it protects and the change that must break it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_hit_seam.py -q
"""
from __future__ import annotations

import math
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
    pytest.skip("this build has no arm chain solver (#106 P6c)", allow_module_level=True)

import _creature_scenarios as S  # noqa: E402
from simulation import creature_fixed as CF  # noqa: E402
from simulation import creature_system as CS  # noqa: E402
from simulation import unit_fixed  # noqa: E402
from simulation.combat import fire_beam, fire_burst, melee_strike  # noqa: E402
from simulation.damage import DamagePacket, apply_packet  # noqa: E402
from simulation.events import UnitHitEvent  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402

ROOM = 40


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _octopus(name, idle_ticks=24):
    """A flooded ROOM x ROOM room, the octopus at its centre after
    ``idle_ticks`` of the default idle (arms seated, outstretched on the deck)."""
    sc = S.Scenario(name, size=ROOM)
    sc.step(idle_ticks)
    return sc


def _march(fx, fy, angle, w=ROOM, h=ROOM):
    """The tiles the bullet march visits from (fx, fy) along ``angle`` (march
    convention), the march's own kit steps and ``int()`` tiling, up to the hull."""
    sx, sy = unit_fixed.cos_rad(angle), unit_fixed.sin_rad(angle)
    rx, ry = float(fx), float(fy)
    out = []
    for _ in range(2 * (w + h)):
        rx += sx
        ry += sy
        t = (int(rx), int(ry))
        if not (1 <= t[0] < w - 1 and 1 <= t[1] < h - 1):
            break
        out.append(t)
    return out


def _marine(name, ax, ay, team=0):
    return Unit(name, x=float(ax), y=float(ay), team=team)


def _node_tile(body, a, k):
    """Node (a, k)'s tile, through the solver's own tile law."""
    b = body.bufs
    from simulation import arm_chain as AC
    T = int(b.world[0, AC.WD_TILE_Q24])
    ox, oy = int(b.world[0, AC.WD_ORIGIN_X_Q24]), int(b.world[0, AC.WD_ORIGIN_Y_Q24])
    x, y = int(b.p[0, a, k, 0]), int(b.p[0, a, k, 1])
    return (CF.chain_tile(x, ox, T), CF.chain_tile(y, oy, T))


def _segment(body, a, k):
    """The tiles of arm a's segment node k -> k + 1 (just node k's tile at the tip)."""
    from simulation import arm_chain as AC
    b = body.bufs
    T = int(b.world[0, AC.WD_TILE_Q24])
    ox, oy = int(b.world[0, AC.WD_ORIGIN_X_Q24]), int(b.world[0, AC.WD_ORIGIN_Y_Q24])
    p = b.p[0, a]
    if k + 1 >= p.shape[0]:
        return [_node_tile(body, a, k)]
    return CF.segment_tiles(int(p[k, 0]) - ox, int(p[k, 1]) - oy,
                            int(p[k + 1, 0]) - ox, int(p[k + 1, 1]) - oy, T)


def _crown_m(body):
    from simulation import arm_chain as AC
    bd = body.bufs.body[0, 0]
    return float(CF.chain_to_m(int(bd[AC.BS_X]))), float(CF.chain_to_m(int(bd[AC.BS_Y])))


def _cheb(t, tiles):
    return min(max(abs(t[0] - f[0]), abs(t[1] - f[1])) for f in tiles)


def _arm_hit_line(sc):
    """A row y whose first hit tile marching EAST from the west hull is an ARM
    tile (not the crown's) at least 3 tiles from the footprint, with its
    struck node at least 1 m from the crown: the shot ACROSS an outstretched
    arm. Returns (y, x of that tile, its (arm, node))."""
    u, body = sc.unit, sc.body
    shape = u.hit_shape
    fp = u.occupied_tiles()
    cx, cy = _crown_m(body)
    for y in sorted({t[1] for t in shape.tiles}):
        path = _march(3.5, y + 0.5, 0.0)
        first = next((t for t in path if t in shape.tile_set), None)
        if first is None or first in shape.body_tiles or first[0] < 6:
            continue
        a, k = shape.node_at[first]
        nx, ny = (float(CF.chain_to_m(int(v))) for v in body.bufs.p[0, a, k, :2])
        if _cheb(first, fp) >= 3 and math.hypot(nx - cx, ny - cy) >= 1.0:
            return y, first[0], (a, k)
    raise AssertionError("no outstretched arm reachable by an eastward row shot "
                         "(the idle pose changed?)")


def _near_miss(sc):
    """A shot (shooter anchor, angle) whose march crosses NO hit tile but
    passes beside an arm (a tile 8-adjacent to an arm tile) inside the arms'
    reach: the empty tiles between the arms on the crown's side. Deterministic
    search on a coarse lattice; the closest such shot to the crown wins."""
    u = sc.unit
    shape = u.hit_shape
    fp = u.occupied_tiles()
    arm = shape.tile_set - shape.body_tiles
    near = {(x + i, y + j) for (x, y) in arm for i in (-1, 0, 1) for j in (-1, 0, 1)}
    near -= shape.tile_set
    best = None
    for ay in range(2, ROOM - 4, 2):
        for ax in range(2, ROOM - 4, 2):
            mine = {(ax + i, ay + j) for i in range(3) for j in range(3)}
            if mine & shape.tile_set:
                continue
            for k in range(24):
                ang = k * math.pi / 12
                path = _march(ax + 1.5, ay + 1.5, ang)
                ps = set(path)
                if ps & shape.tile_set or not ps & near:
                    continue
                d = min(_cheb(t, fp) for t in path)
                if best is None or (d, ax, ay, k) < best:
                    best = (d, ax, ay, k)
    assert best is not None, "no near miss exists around the octopus"
    return best


def _fire_k5(units, shooter, fx, fy, angle, gmap, events):
    return fire_burst(gmap, units, shooter, fx, fy, fx + math.cos(angle), fy + math.sin(angle),
                      0, [], 0.0, np.random.default_rng(1), events,
                      spread_deg=0.0, aim_angle=angle)


# ---------------------------------------------------------------------------
# humans: the seam is the footprint
# ---------------------------------------------------------------------------
def test_a_humans_hit_tiles_are_its_footprint():
    """Property: for every unit but a creature hit_tiles() == occupied_tiles()
    and in_hit_tiles agrees with occupies() on and around the footprint.
    Breaks if the seam's default ever stops being the footprint (a human
    struck off its body, or a body tile that cannot be shot)."""
    for fp in (1, 3, 5):
        u = Unit("M", x=7.0, y=4.0, team=0, footprint=fp)
        assert u.hit_shape is None
        assert u.hit_tiles() == u.occupied_tiles()
        for y in range(1, 12):
            for x in range(4, 15):
                assert u.in_hit_tiles(x, y) == u.occupies((x, y)), (fp, x, y)


def test_the_march_picks_the_same_human_target_with_the_seam_on_and_off(monkeypatch):
    """Property: a scripted spread volley (a shotgun's pellets and a laser
    beam through a crowd of humans) resolves to the same hits, events, hp
    and tracers whether the march tests the seam or the plain footprint
    (``occupies``). Breaks if the seam changes any human hit decision."""
    lvl = S.write_room(Path(S.tempfile.mkdtemp(prefix="seam_")), "seam_humans", w=40, h=24)
    tables = get_tables()

    def run():
        from simulation import Simulation
        sim = Simulation(lvl, seed=3, breach_physics=bp, enable_recorder=False)
        units = [_marine(f"M{i}", ax, ay, team=(0 if i == 0 else 1))
                 for i, (ax, ay) in enumerate(((3, 10), (14, 9), (17, 12), (22, 8),
                                               (26, 11), (30, 13)))]
        for u in units:
            sim.add_unit(u)
        events, shots = [], []
        rng = np.random.default_rng(11)
        shooter = units[0]
        for tgt in units[1:]:
            fire_burst(sim.gmap, units, shooter, shooter.center_tile_x(), shooter.center_tile_y(),
                       tgt.center_tile_x(), tgt.center_tile_y(), 0, shots, 0.0, rng, events,
                       weapon=tables.weapons.by_name["jackhammer_8"])
            fire_beam(sim.gmap, units, shooter, shooter.center_tile_x(), shooter.center_tile_y(),
                      tgt.center_tile_x(), tgt.center_tile_y(), 0, shots, 0.0, rng, events,
                      weapon=tables.weapons.by_name["lance_3"])
        hits = [(type(e).__name__, e.unit_id, getattr(e, "damage", None),
                 getattr(e, "source", None), getattr(e, "attacker_id", None))
                for e in events if hasattr(e, "unit_id")]
        return (hits, [u.current_hp for u in units],
                [(s.fx1, s.fy1, s.fx2, s.fy2) for s in shots])

    seam = run()
    monkeypatch.setattr(Unit, "in_hit_tiles", lambda self, ix, iy: self.occupies((ix, iy)))
    footprint = run()
    assert seam == footprint
    assert any(h[0] == "UnitHitEvent" for h in seam[0]), "the volley hit nobody: no test"


# ---------------------------------------------------------------------------
# the octopus: arms can be shot, nothing else sees them
# ---------------------------------------------------------------------------
def test_hit_tiles_are_the_footprint_plus_the_rasterised_arms():
    """Property: a creature's hit_tiles() is sorted, holds its footprint, and
    is exactly the footprint plus every in-map tile its arm segments cross
    (``segment_tiles`` over the integer chain), every arm tile mapped to a
    node whose own tile or segment holds it. Breaks if the shape is built
    from node tiles only, from the float view, or misses the body."""
    sc = _octopus("seam_shape")
    u, body = sc.unit, sc.body
    tiles = u.hit_tiles()
    assert tiles == sorted(tiles) and len(set(tiles)) == len(tiles)
    assert set(u.occupied_tiles()) <= set(tiles)
    want = set(u.occupied_tiles())
    for a in range(body.arms):
        for k in range(body.nodes):
            want.update(t for t in _segment(body, a, k) if 0 <= t[0] < ROOM and 0 <= t[1] < ROOM)
    assert set(tiles) == want
    for t, (a, k) in u.hit_shape.node_at.items():
        assert t == _node_tile(body, a, k) or t in _segment(body, a, k)
    assert len(want) > len(u.occupied_tiles()), "the arms add nothing: no test"


def test_segment_tiles_is_a_connected_exact_supercover():
    """Property: the rasteriser walks from the start tile to the end tile in
    8-connected steps, never revisiting, and contains every tile whose
    interior a dense sample of the segment visits (a clipped corner
    included). Breaks if a corner clip is dropped or a step overshoots."""
    rng = np.random.default_rng(5)
    T = 1000
    for _ in range(400):
        x0, y0, x1, y1 = (int(v) for v in rng.integers(-3500, 3500, size=4))
        if _ % 5 == 0:   # end points exactly on tile corners / edges
            x0, y1 = (x0 // T) * T, (y1 // T) * T
        path = CF.segment_tiles(x0, y0, x1, y1, T)
        assert path[0] == (x0 // T, y0 // T) and path[-1] == (x1 // T, y1 // T)
        assert len(set(path)) == len(path)
        for p, q in zip(path, path[1:]):
            assert max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1
        for s in np.linspace(0.0, 1.0, 801):
            x, y = x0 + s * (x1 - x0), y0 + s * (y1 - y0)
            if x % T and y % T:   # strictly inside a tile
                assert (int(x // T), int(y // T)) in path


def test_a_shot_across_an_outstretched_arm_hits_and_a_near_miss_misses():
    """Property: a k5 round marching across an arm (no crown tile on its
    line) hits the octopus -- a UnitHitEvent with its id, its hashed
    ``last_hit`` naming an arm and a node whose tile or segment the round
    crossed -- and a round crossing only the empty tiles beside the arms
    misses. Breaks if arms are unhittable, if the whole reach disc is, or if
    the struck node is not recorded."""
    sc = _octopus("seam_shot")
    u, body = sc.unit, sc.body
    y, x_hit, (a_exp, k_exp) = _arm_hit_line(sc)
    path = _march(3.5, y + 0.5, 0.0)
    assert not set(path[:path.index((x_hit, y)) + 1]) & set(u.occupied_tiles())
    shooter = _marine("M", 2, y - 1)
    events = []
    hp0 = u.current_hp
    _fire_k5([shooter, u], shooter, 3.5, y + 0.5, 0.0, sc.sim.gmap, events)
    hits = [e for e in events if isinstance(e, UnitHitEvent)]
    assert hits and {e.unit_id for e in hits} == {u.id}   # every round of the burst
    assert {e.source for e in hits} == {"bullet"}
    assert u.current_hp < hp0
    a, k = body.last_hit
    assert (a, k) == (a_exp, k_exp)
    assert (x_hit, y) == _node_tile(body, a, k) or (x_hit, y) in _segment(body, a, k)
    # the record the digest hashes carries it
    assert CS.decode_record(CS.serialize_record(body))["last_hit"] == (a, k)

    d, ax, ay, kk = _near_miss(sc)
    assert d <= 6, f"the near miss passes {d} tiles from the crown: not crown-side"
    miss_shooter = _marine("N", ax, ay)
    events = []
    hp1 = u.current_hp
    _fire_k5([miss_shooter, u], miss_shooter, ax + 1.5, ay + 1.5, kk * math.pi / 12,
             sc.sim.gmap, events)
    assert not [e for e in events if isinstance(e, UnitHitEvent)]
    assert u.current_hp == hp1 and body.last_hit == (a, k)


def test_a_beam_skewers_an_arm():
    """Property: a laser beam across an arm skewers the octopus once (source
    "laser") and records the struck node. Breaks if the skewer keeps the
    footprint test or hits a creature twice through arm and crown."""
    sc = _octopus("seam_beam")
    u, body = sc.unit, sc.body
    y, x_hit, (a_exp, k_exp) = _arm_hit_line(sc)
    shooter = _marine("M", 2, y - 1)
    events = []
    fire_beam(sc.sim.gmap, [shooter, u], shooter, 3.5, y + 0.5, 4.5, y + 0.5, 0, [], 0.0,
              np.random.default_rng(1), events,
              weapon=get_tables().weapons.by_name["lance_3"], spread_deg=0.0, aim_angle=0.0)
    hits = [e for e in events if isinstance(e, UnitHitEvent)]
    assert [(e.unit_id, e.source) for e in hits] == [(u.id, "laser")]
    assert body.last_hit == (a_exp, k_exp)


def test_a_bite_on_an_arm_tile_does_not_land_but_one_on_the_crown_does():
    """Property: melee resolves its target on the FOOTPRINT: a human beside
    an arm (and nowhere near the crown) striking that arm tile whiffs, while
    a human beside the crown striking a crown tile connects. Breaks if melee
    is moved onto hit_tiles()."""
    sc = _octopus("seam_melee")
    u, body = sc.unit, sc.body
    shape = u.hit_shape
    fp = u.occupied_tiles()
    knife = get_tables().weapons.by_name["combat_knife"]
    # an arm tile far from the crown and a free 3x3 spot touching it
    spot = None
    for t in sorted(shape.tile_set - shape.body_tiles, key=lambda t: (-_cheb(t, fp), t)):
        for ay in range(t[1] - 3, t[1] + 2):
            for ax in range(t[0] - 3, t[0] + 2):
                mine = [(ax + i, ay + j) for i in range(3) for j in range(3)]
                if (set(mine) & shape.tile_set or _cheb(t, mine) != 1
                        or min(_cheb(m, fp) for m in mine) < 3
                        or not all(1 <= m[0] < ROOM - 1 and 1 <= m[1] < ROOM - 1 for m in mine)):
                    continue
                spot = (t, ax, ay)
                break
            if spot:
                break
        if spot:
            break
    assert spot is not None
    t, ax, ay = spot
    m = _marine("M", ax, ay)
    events = []
    hp0, last0 = u.current_hp, body.last_hit
    assert not melee_strike([m, u], m, t[0] + 0.5, t[1] + 0.5,
                            np.random.default_rng(1), events, knife)
    assert not events and u.current_hp == hp0 and body.last_hit == last0
    # the control: beside the crown, a crown tile
    x0, y0 = min(fp)
    c = _marine("C", x0 + 3, y0)
    assert melee_strike([c, u], c, x0 + 2.5, y0 + 0.5, np.random.default_rng(1), events, knife)
    assert [e.unit_id for e in events if isinstance(e, UnitHitEvent)] == [u.id]


def test_arm_tiles_leave_the_stamps_untouched():
    """Property: the stamp planes (obstacles, dyn_permeability,
    dyn_heat_atten_q) are the same with the arms' hit shape present or
    removed, and an arm-only tile carries no unit stamp. Breaks if a stamp
    starts reading hit_tiles()."""
    sc = _octopus("seam_stamps")
    u = sc.unit
    gmap = sc.sim.gmap
    units = sc.sim.units

    def stamps():
        gmap.stamp_units(units)
        return (np.array(gmap.obstacles, copy=True), np.array(gmap.dyn_permeability, copy=True),
                np.array(gmap.dyn_heat_atten_q, copy=True))

    with_arms = stamps()
    shape = u.hit_shape
    u.hit_shape = None
    try:
        without = stamps()
    finally:
        u.hit_shape = shape
    for a, b in zip(with_arms, without):
        assert np.array_equal(a, b)
    arm_only = sorted(shape.tile_set - shape.body_tiles)
    perm = np.asarray(gmap.permeability)
    for (x, y) in arm_only:
        assert with_arms[1][y, x] == perm[y, x]


def test_hit_tiles_are_rebuilt_each_tick_and_freeze_with_a_dead_creature():
    """Property: after every tick a living creature's hit_tiles() equals the
    shape built from its CURRENT chain (and the set changes as the arms
    crawl); once dead it is no longer ticked, so the set freezes with its
    pose. Breaks if the shape is built once, or rebuilt for a corpse."""
    sc = S.Scenario("seam_ticks", size=60)
    u, body = sc.unit, sc.body
    seen = set()
    for _ in range(36):
        sc.move(0.0)
        sc.step(1)
        tiles = u.hit_tiles()
        assert tiles == sorted(tiles)
        fresh = CS.build_hit_shape(body, u.occupied_tiles(), 60, 60)
        assert fresh.tiles == tuple(tiles) and fresh.node_at == u.hit_shape.node_at
        seen.add(tuple(tiles))
    assert len(seen) > 1, "the arms never moved the set"
    apply_packet(u, DamagePacket(amount=100000, dtype=0, attacker_id=-1), [], source="bullet")
    assert not u.alive
    frozen = u.hit_tiles()
    p_frozen = body.bufs.p.copy()
    sc.step(6)
    assert u.hit_tiles() == frozen and np.array_equal(body.bufs.p, p_frozen)
