"""#183 G9p stage 1 -- the horde field (the residency design's P3d, §4.9).

Properties: the C++ flood (``PhysicsEngine.horde_field``) and the Python
reference of the law (``horde.flood_reference``) both equal an INDEPENDENT
flood (Bellman-Ford relaxation over AND-ed shifted windows, written here) on
real maps -- ship_demo, the box, a recipe ship -- and after a wall is
destroyed; the field does not depend on the order of its seeds or of the
prey; every zombie step of a match is to a strictly lower field value; the
zombie law calls no A*; in a straight corridor the field walk is A*'s path
and the bite lands as it always did; the activation gate is the old sqrt
gate, integer; the cache re-floods when the topology moves.
"""
from __future__ import annotations

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
import simulation.ai_zombie as ai_zombie  # noqa: E402
import simulation.simulation as sim_mod  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation, horde  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.unit import Unit  # noqa: E402

BIG = horde.UNREACHED


def relax_flood(tile_ok, fp, seeds):
    """An independent flood: passability as the AND of fp*fp shifted
    windows, distances by relaxing ``d = min(d, neighbour + 1)`` over the
    whole grid until nothing changes (no queue, no integral image)."""
    ok = np.asarray(tile_ok, dtype=bool)
    h, w = ok.shape
    passable = np.zeros((h, w), dtype=bool)
    if fp <= h and fp <= w:
        acc = np.ones((h - fp + 1, w - fp + 1), dtype=bool)
        for dy in range(fp):
            for dx in range(fp):
                acc &= ok[dy:dy + h - fp + 1, dx:dx + w - fp + 1]
        passable[:h - fp + 1, :w - fp + 1] = acc
    d = np.full((h, w), np.int64(BIG), dtype=np.int64)
    for y, x in seeds:
        if 0 <= y < h and 0 <= x < w:
            d[y, x] = 0
    seed_mask = d == 0
    while True:
        nb = np.full((h, w), np.int64(BIG), dtype=np.int64)
        nb[1:, :] = np.minimum(nb[1:, :], d[:-1, :] + 1)
        nb[:-1, :] = np.minimum(nb[:-1, :], d[1:, :] + 1)
        nb[:, 1:] = np.minimum(nb[:, 1:], d[:, :-1] + 1)
        nb[:, :-1] = np.minimum(nb[:, :-1], d[:, 1:] + 1)
        new = np.where(passable & ~seed_mask, np.minimum(d, nb), d)
        new = np.minimum(new, BIG)
        if np.array_equal(new, d):
            break
        d = new
    return d.astype(np.int32)


def _cpp(tile_ok, fp, seeds):
    out = np.empty(tile_ok.shape, dtype=np.int32)
    bp.PhysicsEngine().horde_field(
        np.ascontiguousarray(tile_ok, dtype=np.uint8), fp,
        np.ascontiguousarray(np.asarray(seeds, dtype=np.int32).reshape(-1, 2)),
        out)
    return out


def _box_level(tmp_path, name="box", w=40, h=24):
    from test_zombie_control import _box
    return _box(tmp_path, name, w=w, h=h)


@pytest.fixture(scope="module")
def recipe_level(tmp_path_factory):
    import gen_ship as G
    plan = G.generate(1, "medium", recipe="freighter", profile="standard",
                      zombies=4)
    base = tmp_path_factory.mktemp("horde_recipe")
    G.write(plan, base / plan.name)
    return level_loader.load(plan.name, levels_dir=str(base))


def _seeds_for(gmap, fp, n=6):
    """The anchors of n open cells spread over the map (row-major picks),
    plus one out of bounds (skipped by every flood)."""
    passable = horde.block_passable(horde.walkable_tiles(gmap), fp)
    ys, xs = np.nonzero(passable)
    idx = np.linspace(0, len(ys) - 1, n).astype(int)
    return [(int(ys[i]), int(xs[i])) for i in idx] + [(-5, 3)]


def _check_map(gmap):
    for fp in (3, 1):
        tile_ok = horde.walkable_tiles(gmap)
        seeds = _seeds_for(gmap, fp)
        want = relax_flood(tile_ok, fp, seeds)
        assert (want < BIG).sum() > 50                     # a real field
        assert np.array_equal(_cpp(tile_ok, fp, seeds), want)
        assert np.array_equal(horde.flood_reference(tile_ok, fp, seeds), want)


def test_the_field_is_the_reference_flood_on_ship_demo():
    """Property: on ship_demo, for footprints 3 and 1, the C++ flood and the
    Python reference equal the independent relaxation flood cell for cell.
    Breaks if either flood's passability (the footprint block, the bounds),
    its connectivity or its seed rule changes."""
    _check_map(GameMap(level_loader.load("ship_demo", str(ROOT / "levels"))))


def test_the_field_is_the_reference_flood_on_the_box(tmp_path):
    """Property: the same on the zombie-control box. Breaks like the
    ship_demo gate."""
    _check_map(GameMap(_box_level(tmp_path)))


def test_the_field_is_the_reference_flood_on_a_recipe_ship(recipe_level):
    """Property: the same on a generated recipe ship (freighter, medium,
    seed 1): doors, grilles, furniture and the hull. Breaks like the
    ship_demo gate."""
    _check_map(GameMap(recipe_level))


def test_the_field_follows_a_destroyed_wall(tmp_path):
    """Property: after ``destroy_wall`` opens a gap in a wall that split a
    room, the flood over the new map equals the reference flood of the new
    map, and some cell is now nearer than before (the gap is used). Breaks
    if the walkable mask is cached past a topology change."""
    lvl = _box_level(tmp_path, "split")
    g = GameMap(lvl)
    g.material[1:23, 20] = MAT_HULL                       # a wall, x = 20
    g._update_caches()
    seeds = [(10, 4)]
    before = _cpp(horde.walkable_tiles(g), 3, seeds)
    assert before[10, 30] == BIG                          # the far side: no path
    for y in range(9, 14):
        g.destroy_wall(y, 20)
    after_ok = horde.walkable_tiles(g)
    after = _cpp(after_ok, 3, seeds)
    assert np.array_equal(after, relax_flood(after_ok, 3, seeds))
    assert after[10, 30] < BIG


def test_the_field_does_not_depend_on_the_order_of_seeds_or_prey(tmp_path):
    """Property (id permutation): the flood of a seed set is the same in
    any order, and the cache's prey field is the same with the prey list
    reversed. Breaks if the field depends on the order units are listed (a
    first-come label, a tie that is not unique)."""
    g = GameMap(_box_level(tmp_path, "perm"))
    seeds = _seeds_for(g, 3)
    ok = horde.walkable_tiles(g)
    a = _cpp(ok, 3, seeds)
    assert np.array_equal(a, _cpp(ok, 3, seeds[::-1]))
    assert np.array_equal(a, _cpp(ok, 3, seeds[2:] + seeds[:2]))
    prey = [Unit(f"M{i}", x=float(x), y=float(y), team=0)
            for i, (y, x) in enumerate([(3, 3), (15, 30), (8, 18)])]
    s1, s2 = horde.HordeState(), horde.HordeState()
    f1 = s1.prey_field(g, 3, prey)
    f2 = s2.prey_field(g, 3, prey[::-1])
    assert np.array_equal(f1, f2)


def test_every_zombie_step_is_to_a_strictly_lower_field_value(tmp_path,
                                                              monkeypatch):
    """Property: through a whole box match (idle marines, the zombie
    controller), every cell a zombie steps to in slot 5 holds a strictly
    lower value on the field the slot used than the cell it left. Breaks if
    the descent takes an equal or higher neighbour, or reads another field
    than the one it walks."""
    real = sim_mod.update_zombies_tick
    checked = []

    def spy(gmap, units, tick, events=None, state=None, **kw):
        before = {u.id: (u.tile_y, u.tile_x) for u in units if u.is_zombie}
        real(gmap, units, tick, events=events, state=state, **kw)
        field = state.fields.get(("prey", 3))
        for u in units:
            if u.is_zombie and u.id in before:
                old = before[u.id]
                new = (u.tile_y, u.tile_x)
                if new != old:
                    assert field is not None
                    assert int(field[new]) < int(field[old]), (tick, u.id)
                    checked.append(tick)

    monkeypatch.setattr(sim_mod, "update_zombies_tick", spy)
    run_match(_box_level(tmp_path, "steps"))
    assert len(checked) >= 5                # 1 m tiles: a 40-tick shamble


def test_the_zombie_law_calls_no_a_star(tmp_path, monkeypatch):
    """Property: no A* call remains in the zombie law -- a box match with
    idle marines runs to its end with ``pathfinding.astar`` replaced by a
    function that raises, and the law's module holds no ``astar``. Breaks if
    the horde paths by a per-zombie search again."""
    import pathfinding

    def boom(*a, **k):
        raise AssertionError("A* called by the horde")

    monkeypatch.setattr(pathfinding, "astar", boom)
    assert not hasattr(ai_zombie, "astar")
    o = run_match(_box_level(tmp_path, "noastar"))
    assert o.result.ticks > 0


def _corridor(w=40):
    tm = np.full((7, w), MAT_HULL, dtype=np.int32)
    tm[2:5, 1:w - 1] = MAT_AIR                            # 3 wide: one lane
    return LevelData(name="corridor", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def test_in_a_straight_corridor_the_field_walk_is_a_stars_path_and_bites():
    """Property (the bite law unchanged where the routes agree): in a
    3-wide corridor the zombie's anchors are exactly the prefix of the A*
    path the old law walked, one per shamble cadence, and the first bite
    lands on the first tick its prey is within ``footprint + 1`` tiles --
    a ``melee`` hit of the shipped damage credited to the biter. Breaks if
    the descent leaves the straight route, the cadence moves or the bite's
    gate, damage or credit change."""
    from pathfinding import astar
    from simulation.events import UnitHitEvent
    sim = Simulation(_corridor(), seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    m = sim.get_unit(sim.add_unit(Unit("M", x=30, y=2, team=0)))
    z = sim.get_unit(sim.add_unit(Unit("Z", x=3, y=2, team=ZOMBIE_TEAM)))
    m.current_hp = 1e9
    g = sim.gmap
    path = astar(3, 2, 30, 2, lambda x, y: not g.is_passable_block(y, x, 3),
                 g.material.shape[1], g.material.shape[0])
    visited = []
    bite_tick = None
    shamble = sim.speed_table.zombie_shamble_ticks
    for _ in range(40 * shamble):
        before = (z.tile_x, z.tile_y)
        sim.step()
        if (z.tile_x, z.tile_y) != before:
            visited.append((z.tile_x, z.tile_y))
        hits = [e for e in sim.tick_events if isinstance(e, UnitHitEvent)
                and e.unit_id == m.id]
        if hits:
            bite_tick = sim.tick - 1
            assert hits[0].source == "melee" and hits[0].attacker_id == z.id
            break
    assert visited == path[1:1 + len(visited)]
    assert m.center_tile_x() - z.center_tile_x() <= z.footprint + 1
    assert bite_tick is not None
    # The bite came on the tick of the last step's arrival in reach: the
    # step before it was out of reach.
    assert m.center_tile_x() - (z.center_tile_x() - 1) > z.footprint + 1


def test_the_activation_gate_is_the_old_sqrt_gate_in_integers():
    """Property: a dormant zombie wakes iff a prey with line of sight is
    STRICTLY inside ``trigger_radius`` tiles: at exactly the radius it
    stays dormant, one tile closer it wakes (the old ``sqrt(d2) < r``,
    integer now). Breaks if the gate becomes inclusive or loses precision."""
    r = int(CFG.zombie.trigger_radius)
    w = r + 20
    tm = np.full((9, w), MAT_HULL, dtype=np.int32)
    tm[1:8, 1:w - 1] = MAT_AIR
    lvl = LevelData(name="gate", version="1", path=Path("."), tilemap=tm,
                    tile_size_m=1.0, diffuse_path=Path("."))
    for gap, woken in ((r, False), (r - 1, True)):
        g = GameMap(lvl)
        z = Unit("Z", x=2, y=3, team=ZOMBIE_TEAM)
        m = Unit("M", x=2 + gap, y=3, team=0)
        z.id, m.id = 1, 2
        ai_zombie.update_zombies_tick(g, [m, z], 1)
        assert z.zombie_activated is woken, gap


def test_a_zombie_with_no_path_waits(tmp_path):
    """Property: a woken zombie in a sealed room (its field cell and every
    neighbour unreached) never moves -- it waits; and the cache's field
    there is UNREACHED. Breaks if the descent steps on an unreached value
    or the flood leaks through a wall."""
    tm = np.full((20, 30), MAT_HULL, dtype=np.int32)
    tm[1:19, 1:12] = MAT_AIR
    tm[1:19, 14:29] = MAT_AIR
    lvl = LevelData(name="sealed", version="1", path=Path("."), tilemap=tm,
                    tile_size_m=1.0, diffuse_path=Path("."))
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    sim.add_unit(Unit("M", x=20, y=8, team=0))
    z = sim.get_unit(sim.add_unit(Unit("Z", x=4, y=8, team=ZOMBIE_TEAM)))
    z.zombie_activated = True
    start = (z.x, z.y)
    for _ in range(120):
        sim.step()
    assert (z.x, z.y) == start
    assert int(sim.horde_state.fields[("prey", 3)][8, 4]) == BIG


def test_the_cache_refloods_when_the_topology_moves(tmp_path):
    """Property: the sim's field cache drops on a refresh tick AND on the
    first slot after a topology change, so a wall opened between refreshes
    is walked through on the next tick. Breaks if a field outlives
    ``GameMap._tile_serial``."""
    g_level = _box_level(tmp_path, "reflood")
    sim = Simulation(g_level, seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    st = sim.horde_state
    st.begin_tick(sim.gmap, 1, sim.horde_settings)
    st.fields[("prey", 3)] = "stale"
    st.begin_tick(sim.gmap, 3, sim.horde_settings)        # odd tick: kept
    assert st.fields.get(("prey", 3)) == "stale"
    sim.gmap.destroy_wall(0, 5)
    st.begin_tick(sim.gmap, 5, sim.horde_settings)        # serial moved
    assert ("prey", 3) not in st.fields
    st.fields[("prey", 3)] = "stale"
    st.begin_tick(sim.gmap, 6, sim.horde_settings)        # refresh tick
    assert ("prey", 3) not in st.fields
