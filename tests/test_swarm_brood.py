"""The brood entity and its load-time spawn (arc #63 P4, docs/swarm_P4_impl.md).

Each test names the property it protects and the change that must break it.
Levels are synthetic, written through level_lib into tmp_path (never the
repo's levels/).

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_swarm_brood.py -q
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tools",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import level_lib  # noqa: E402
import level_loader  # noqa: E402
from level_loader import EntityInstance  # noqa: E402
from simulation import swarm, swarm_brood, swarm_fixed  # noqa: E402
from simulation.draws import MatchDraws  # noqa: E402
from simulation.entities import REGISTRY, serialize_entity_state  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.rng_salts import Salt  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402

GRID = (16, 24)                     # rows x cols


def _write_level(root: Path, name: str, broods=(), extra=(), wall_cols=(),
                 match=None, spawn_rooms=()):
    """A hull box with an air interior; ``broods`` = (id, zone_id, (x0, y0,
    x1, y1), count, species) painted into zones.npy; ``extra`` = further
    EntityInstances appended after them; ``wall_cols`` = interior columns
    turned to hull."""
    lvl = root / name
    lvl.mkdir(parents=True)
    h, w = GRID
    level_lib.write_level_header(lvl, name=name, tile_size_m=0.333)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    for c in wall_cols:
        tm[1:h - 1, c] = MAT_HULL
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    z = np.zeros((h, w), dtype=np.uint8)
    ents = []
    for i, (bid, zid, (x0, y0, x1, y1), count, sp) in enumerate(broods):
        z[y0:y1 + 1, x0:x1 + 1] = zid
        ents.append(EntityInstance(
            id=bid, class_name="swarm_brood", ordinal=i,
            fields={"zone_id": zid, "species": sp, "count": count},
            authored_keys=("zone_id", "species", "count")))
    for k, e in enumerate(extra):
        ents.append(replace(e, ordinal=len(broods) + k))
    level_lib.write_zones_npy(lvl, z, npy_bak=False)
    level_lib.write_managed_blocks(
        lvl / "level.toml",
        {"entity": lambda nl: level_lib.format_entity_lines(ents, nl)})
    if spawn_rooms:
        level_lib.write_spawn_rooms(lvl / "level.toml", spawn_rooms)
    if match is not None:
        level_lib.write_match(lvl, match)
    return level_loader.load(name, str(root))


TWO = (("bw", 1, (2, 2, 9, 13), 120, "larva"),
       ("be", 2, (12, 2, 21, 13), 80, "larva"))


def _sim(lv, seed=7):
    return Simulation(lv, seed=seed, breach_physics=bp, enable_recorder=False)


def _units(sim):
    """The store's live units as (unit_id, pos_x, pos_y, heading) rows, slot
    order."""
    g = sim.gmap
    v = swarm.swarm_column(g, "larva", "valid") != 0
    return np.stack([swarm.swarm_column(g, "larva", c)[v].astype(np.int64)
                     for c in ("unit_id", "pos_x", "pos_y", "heading")], axis=1)


# ---------------------------------------------------------------------------
# The entity: registry, writer, loader, serializer
# ---------------------------------------------------------------------------

def test_the_brood_round_trips_through_the_writer_the_loader_and_the_serializer(tmp_path):
    """Property: swarm_brood is a registered intangible zone class with
    zone_id / species (default "larva") / count (default 200) and no seed
    column; level_lib writes it and the loader reads back the same fields
    and its zone; the one serializer hashes its synced fields (zone_id,
    count; the species string is not a synced kind). Breaks if the class
    leaves the registry or ZONE_CLASSES, gains a seed column, or the writer
    / loader drops a field."""
    cls = REGISTRY["swarm_brood"]
    assert cls.INTANGIBLE is True
    fields = {f.name: f for f in cls.FIELDS}
    assert set(fields) == {"zone_id", "species", "count"}
    assert fields["species"].default == "larva"
    assert fields["count"].default == 200
    assert "swarm_brood" in level_loader.ZONE_CLASSES
    lv = _write_level(tmp_path, "rt", broods=TWO)
    got = [e for e in lv.entities if e.class_name == "swarm_brood"]
    assert [(e.id, e.fields["zone_id"], e.fields["species"], e.fields["count"])
            for e in got] == [(b[0], b[1], b[4], b[3]) for b in TWO]
    assert int((lv.zone_grid == 1).sum()) == 8 * 12
    blob = serialize_entity_state(got)
    assert b"swarm_brood" in blob and b"count|" in blob and b"zone_id|" in blob
    assert b"species" not in blob


# ---------------------------------------------------------------------------
# The spawn
# ---------------------------------------------------------------------------

def test_each_brood_spawns_count_units_on_free_tiles_of_its_own_zone(tmp_path):
    """Property: every brood spawns exactly ``count`` live units, each inside
    a free (non-solid) tile its zone paints, heading canonical; broods fill
    in ordinal order (brood 0's unit ids first). Breaks if the scatter lands
    off the zone, on a wall, outside the tile, or the fill order changes."""
    lv = _write_level(tmp_path, "sp", broods=TWO, wall_cols=(5,))
    sim = _sim(lv)
    u = _units(sim)
    assert u.shape[0] == 200
    tx, ty = u[:, 1] >> swarm_fixed.FP_SHIFT, u[:, 2] >> swarm_fixed.FP_SHIFT
    zone = lv.zone_grid[ty, tx]
    assert (zone[u[:, 0] <= 120] == 1).all() and (zone[u[:, 0] > 120] == 2).all()
    assert not sim.gmap.solid[ty, tx].any(), "a unit on a wall (column 5)"
    assert (u[:, 3] > -swarm_fixed.PI_Q16).all() and (u[:, 3] <= swarm_fixed.PI_Q16).all()
    assert sorted(u[:, 0].tolist()) == list(range(1, 201))
    assert swarm.check_invariants(sim.gmap) == []


def test_the_scatter_is_a_function_of_the_seed_alone(tmp_path):
    """Property: the same match seed gives the same store twice (and after a
    reset); another seed another scatter. Breaks if the scatter reads a
    stream, the clock past tick 0, or anything but (seed, ordinal, k)."""
    lv = _write_level(tmp_path, "seed", broods=TWO)
    a, b, c = _units(_sim(lv, 7)), _units(_sim(lv, 7)), _units(_sim(lv, 8))
    assert np.array_equal(a, b)
    assert (a[:, 1:] != c[:, 1:]).any(axis=1).mean() > 0.95
    s = _sim(lv, 7)
    for _ in range(5):
        s.set_paused(False)
        s.step()
    s.reset(7)
    assert np.array_equal(_units(s), a)


def test_an_unrelated_entity_after_a_brood_moves_nothing(tmp_path):
    """Property: adding an entity after the broods (here an extraction zone
    and a heat source) leaves every brood's scatter as it was. Breaks if the
    key takes anything but the brood's own ordinal and k (an entity count, a
    shared draw counter)."""
    a = _units(_sim(_write_level(tmp_path, "plain", broods=TWO)))
    extra = (EntityInstance(id="hs", class_name="heat_source", ordinal=0,
                            fields={"x": 3, "y": 3, "heat_per_s": 1.0},
                            authored_keys=("x", "y", "heat_per_s")),)
    b = _units(_sim(_write_level(tmp_path, "more", broods=TWO, extra=extra)))
    assert np.array_equal(a, b)


def test_two_broods_never_share_a_key():
    """Property: the same tiles and count under two ordinals give different
    scatters (the key holds the ordinal), and the draw indices of one brood
    are distinct per (unit, purpose). Breaks if the ordinal leaves the key
    or two purposes share an index."""
    tiles = np.argwhere(np.ones((6, 6), dtype=bool)).astype(np.int64)
    d = MatchDraws(3)
    a = swarm_brood.scatter(d, 0, 300, tiles)
    b = swarm_brood.scatter(d, 1, 300, tiles)
    same = (a[0] == b[0]) & (a[1] == b[1])
    assert same.sum() == 0
    w = d.words_at(Salt.SWARM_BROOD, 0, np.arange(8, dtype=np.uint32))
    assert len(set(w.tolist())) == 8


def test_words_at_are_the_draws_at_gives(seed=11):
    """Property: the vectorised words are exactly ``at(...).word()`` one by
    one, counted in ``drawn``. Breaks if words_at forms another counter."""
    d = MatchDraws(seed)
    d.set_tick(5)
    idx = np.array([0, 1, 7, 1000, 4096], dtype=np.uint32)
    w = d.words_at(Salt.SWARM_BROOD, 9, idx)
    assert w.tolist() == [d.at(Salt.SWARM_BROOD, 9, int(i)).word() for i in idx]
    assert d.drawn == 2 * len(idx)
    with pytest.raises(TypeError):
        d.words_at(Salt.SWARM_BROOD, 9, idx.astype(np.int64))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def _fake_sim(lv, seed=1):
    g = GameMap(lv)
    return SimpleNamespace(entities=lv.entities, level=lv, gmap=g,
                           draws=MatchDraws(seed), swarm=swarm.SwarmUnits(g))


@pytest.mark.parametrize("case", ["no_free_tile", "unknown_species", "over_capacity"])
def test_a_brood_that_cannot_spawn_is_refused_by_name_with_nothing_spawned(tmp_path, case):
    """Property: a brood whose zone has no free tile, a species that does not
    exist, and broods asking more than the store holds are each refused by
    name before any write (the store stays empty, next_unit_id unmoved), the
    first two also by the level lint. Breaks on a clamp, a partial fill, or
    a check that runs after the first brood spawned."""
    import level_airtight
    cap = swarm.SwarmSpeciesTable.from_config().row("larva").max_units
    if case == "no_free_tile":
        broods = (TWO[0], ("walled", 2, (5, 2, 5, 13), 10, "larva"))
        lv = _write_level(tmp_path, case, broods=broods, wall_cols=(5,))
        exc, words = swarm_brood.BroodRefused, ("walled", "no free")
    elif case == "unknown_species":
        broods = (TWO[0], ("odd", 2, (12, 2, 21, 13), 10, "moth"))
        lv = _write_level(tmp_path, case, broods=broods)
        exc, words = swarm_brood.BroodRefused, ("odd", "moth")
    else:
        broods = (("a", 1, (2, 2, 9, 13), cap - 5, "larva"),
                  ("b", 2, (12, 2, 21, 13), 10, "larva"))
        lv = _write_level(tmp_path, case, broods=broods)
        exc, words = swarm.SwarmFull, ("larva", str(cap - 5 + 10))
    fake = _fake_sim(lv)
    with pytest.raises(exc) as e:
        swarm_brood.spawn_broods(fake)
    for w in words:
        assert w in str(e.value)
    assert fake.swarm.live_count("larva") == 0
    assert int(fake.gmap.swarm_next_unit_id) == swarm.FIRST_UNIT_ID
    with pytest.raises(exc):
        _sim(lv)
    if case != "over_capacity":
        assert level_airtight.check(case, str(tmp_path)) is False


def test_the_lint_passes_a_good_brood(tmp_path):
    """Property: a level whose broods can spawn passes the lint. Breaks if
    the brood check reports a problem on a good brood."""
    import level_airtight
    _write_level(tmp_path, "good", broods=TWO)
    assert level_airtight.check("good", str(tmp_path)) is True


def test_a_level_without_a_brood_draws_and_stores_nothing(tmp_path):
    """Property: a brood-free level (zones painted, an extraction zone) draws
    no word at load and holds no swarm (no ``__swarm__`` block, nothing
    present), so its digest is the pre-P4 one. Breaks if the spawn seam draws
    or allocates a present store without a brood."""
    ez = EntityInstance(id="ex", class_name="extraction_zone", ordinal=0,
                        fields={"zone_id": 3, "faction": 0},
                        authored_keys=("zone_id",))
    lv = _write_level(tmp_path, "nob", extra=(ez,))
    sim = _sim(lv)
    assert sim.draws.drawn == 0
    assert not swarm.swarm_present(sim.gmap)
    assert sim.swarm.carrier()["present"] is False
    assert swarm_brood.spawn_broods(sim) == []


# ---------------------------------------------------------------------------
# Every match path: the batched env's reset
# ---------------------------------------------------------------------------

def test_a_batched_envs_reset_respawns_from_its_own_draws(tmp_path):
    """Property: each env of the farm spawns its broods at reset from its own
    episode seed, equal to a lone MatchSession at that seed; the next episode
    re-spawns from the next seed. Breaks if the spawn is not on the path the
    env's reset takes, or reads anything shared between envs."""
    import rl_env
    from simulation.factions import ZOMBIE_TEAM
    from simulation.match import MatchSpec, MatchTeam, SpawnRoom
    from simulation.match_session import MatchSession
    idle = MatchTeam(n=1, loadout="rifleman", env_profile="default",
                     controller="idle")
    rooms = [SpawnRoom(team=0, x=2, y=2, w=4, h=4, max_per_team=1),
             SpawnRoom(team=1, x=17, y=9, w=4, h=4, max_per_team=1),
             SpawnRoom(team=ZOMBIE_TEAM, x=10, y=2, w=4, h=4, max_per_team=1)]
    spec = MatchSpec(teams=(idle, idle), seed=1, zombies=0, time_limit_s=5.0)
    lv = _write_level(tmp_path, "farm", broods=TWO, match=spec, spawn_rooms=rooms)
    run_toml = tmp_path / "run.toml"
    run_toml.write_text(
        "[run]\nseed = 11\nn_envs = 2\ndecimation = 4\nlevels = [\"farm\"]\n"
        f"policy = \"policy\"\nlevels_dir = '{tmp_path.as_posix()}'\n"
        "[obs]\nchannels = [\"pressure\"]\ncrop_radius = 2\nvector = [\"hp\"]\n"
        "[reward]\nkills = 1.0\n", encoding="utf-8")
    run = rl_env.load_run(run_toml)
    layout = rl_env.make_layout(run, [lv])

    def lone(seed):
        s = MatchSession(lv, replace(lv.match, seed=seed), seed=seed,
                         breach_physics=bp)
        return _units(s.sim)

    stores = []
    for e in (0, 1):
        r = rl_env.EnvRunner(e, run, layout, [lv], breach_physics=bp)
        r.reset()
        got = _units(r.session.sim)
        assert got.shape[0] == 200
        assert np.array_equal(got, lone(r.seed)), f"env {e}"
        stores.append(got)
        if e == 0:
            first = r.seed
            r.reset()
            assert r.seed != first
            assert np.array_equal(_units(r.session.sim), lone(r.seed))
    assert not np.array_equal(stores[0], stores[1])


def test_the_tile_inspector_counts_the_larvae_on_its_tile(tmp_path):
    """Property: ``larvae on this tile`` is the number of live units whose
    position lies in the tile, through the inspector's one seam. Breaks if
    the count reads empty slots, another tile or the wrong axis."""
    from renderer.hover_readout import pack_hover_readout
    lv = _write_level(tmp_path, "insp", broods=TWO)
    sim = _sim(lv)
    u = _units(sim)
    tx, ty = u[:, 1] >> swarm_fixed.FP_SHIFT, u[:, 2] >> swarm_fixed.FP_SHIFT
    (x, y), n = max(((int(a), int(b)), int(((tx == a) & (ty == b)).sum()))
                    for a, b in zip(tx, ty))
    assert n >= 1
    assert swarm.count_on_tile(sim.gmap, "larva", x, y) == n
    assert swarm.count_on_tile(sim.gmap, "larva", 0, 0) == 0
    r = pack_hover_readout(sim.gmap, x, y, lambda t: t + 293.0)
    assert r.larvae == n and f"larvae on this tile: {n}" in r.lines
    sim.swarm.reclaim("larva", np.flatnonzero(
        (swarm.swarm_column(sim.gmap, "larva", "pos_x") >> 16 == x)
        & (swarm.swarm_column(sim.gmap, "larva", "pos_y") >> 16 == y)
        & (swarm.swarm_column(sim.gmap, "larva", "valid") != 0)))
    assert swarm.count_on_tile(sim.gmap, "larva", x, y) == 0
    assert swarm.count_on_tile(sim.gmap, "larva", 0, 0) == 0, "a freed slot counted"
