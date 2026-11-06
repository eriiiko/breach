"""Properties of the LEVELSPEC (#136; docs/levelgen_archetype_recipe_2026-10-31.md
§5.1): a recipe ship records every generator input and what it realised in
level.toml's ``[levelgen]`` table (written by ``level_lib``, read by
``level_loader``), and is a pure function of that record
(``gen_ship.generate_from_spec`` / ``gen_ship.py --from-level``). Plus the
recipe path's spawn box: the squad stands in the middle of its slab.

Each test names the property it protects and the change that must break it.
No ship, count or set of departments / cycle types is pinned: the record is
checked against the ship it was written for.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_ship as G  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import ship_grammar as S  # noqa: E402
import ship_layout as L  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import formation_capacity, placement_cells  # noqa: E402

ARGS = dict(seed=1, size="medium", profile="standard", zombies=4)


@pytest.fixture(scope="module")
def plan():
    a = ARGS
    return G.generate(a["seed"], a["size"], recipe="freighter",
                      profile=a["profile"], zombies=a["zombies"])


@pytest.fixture(scope="module")
def written(plan, tmp_path_factory):
    """The plan written through the CLI's own writer, and loaded back."""
    base = tmp_path_factory.mktemp("levelspec")
    G.write(plan, base / plan.name)
    return base, level_loader.load(plan.name, levels_dir=str(base))


def _record_dict(plan) -> dict:
    return G.levelgen_record(plan, "test-build").as_dict()


def _raster(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        return im.size, im.mode, im.tobytes()


def _differ(a: Path, b: Path) -> list:
    """Every file of folder ``a`` that folder ``b`` does not reproduce (a PNG
    by its DECODED raster: encoder bytes are not the generator's, #172)."""
    out = []
    names = sorted(p.name for p in a.iterdir() if p.is_file())
    if names != sorted(p.name for p in b.iterdir() if p.is_file()):
        out.append(("file set", names))
    for n in names:
        if not (b / n).is_file():
            continue
        if n.endswith(".png"):
            same = _raster(a / n) == _raster(b / n)
        else:
            same = (a / n).read_bytes() == (b / n).read_bytes()
        if not same:
            out.append(n)
    return out


# ---------------------------------------------------------------------------
# The [levelgen] door
# ---------------------------------------------------------------------------

def test_the_levelgen_record_round_trips_through_the_one_writer(plan,
                                                                written):
    """PROPERTY: what ``gen_ship.write`` records through ``level_lib`` is
    exactly what ``level_loader`` reads back -- every input and every
    realised factor, the inline ``cycles`` table included -- and a second
    write of the loaded record is byte-stable. BREAKS IF the writer and the
    loader disagree on a key, a type or the inline-table form, or the family
    is not registered with the one writer."""
    base, lv = written
    rec = lv.levelgen
    assert isinstance(rec, level_loader.LevelgenRecord)
    expect = G.levelgen_record(plan, rec.generator_version)
    assert rec == expect
    toml = base / plan.name / "level.toml"
    before = toml.read_bytes()
    level_lib.write_levelgen(toml, rec)
    assert toml.read_bytes() == before
    assert "levelgen" in level_lib.MANAGED_FAMILIES


@pytest.mark.parametrize("edit, match", [
    (lambda d: d.update(bogus=1), r"unknown key\(s\) \['bogus'\]"),
    (lambda d: d.pop("level_seed"), "missing input 'level_seed'"),
    (lambda d: d.pop("recipe_hash"), "missing input 'recipe_hash'"),
    (lambda d: d.pop("spawn_distance"),
     "missing realised factor 'spawn_distance'"),
    (lambda d: d.update(max_per_team="6"), r"\[levelgen\]\.max_per_team"),
    (lambda d: d.update(level_seed=True), r"\[levelgen\]\.level_seed"),
    (lambda d: d.update(g=-1), r"\[levelgen\]\.g "),
    (lambda d: d.update(recipe_hash="abc"), r"\[levelgen\]\.recipe_hash"),
    (lambda d: d.update(generator_version=""),
     r"\[levelgen\]\.generator_version"),
    (lambda d: d.update(departments=[]), r"\[levelgen\]\.departments"),
    (lambda d: d.update(cycles={"pocket": -1}), r"\[levelgen\]\.cycles\.pocket"),
    (lambda d: d.update(cycles=[1]), r"\[levelgen\]\.cycles"),
])
def test_the_levelgen_door_refuses_by_name(plan, edit, match):
    """PROPERTY: the loader's door refuses an unknown key, a missing input
    or realised factor and a mistyped value, naming the key; the WRITER runs
    the same door, so it refuses the same record before touching disk.
    BREAKS IF a key is accepted unchecked, a check stops naming its key, or
    the writer formats without the door."""
    d = _record_dict(plan)
    edit(d)
    with pytest.raises(ValueError, match=match):
        level_loader.parse_levelgen(d, "test")
    with pytest.raises(ValueError, match=match):
        level_lib.format_levelgen_lines(d)


def test_a_level_without_levelgen_loads_as_before(tmp_path):
    """PROPERTY: the table is DORMANT when absent -- ``levels/ship_demo``
    (a v1 ship) loads with ``levelgen = None``, and a v1 plan written now
    carries no ``[levelgen]`` (v1 has no LevelSpec, so its files do not move
    by a byte; test_gen_ship_edges holds ship_demo's bytes). BREAKS IF the
    loader requires the table, or the v1 path starts writing one."""
    assert level_loader.load("ship_demo").levelgen is None
    v1 = G.generate(5, "medium", vents=True)
    G.write(v1, tmp_path / v1.name)
    raw = tomllib.loads((tmp_path / v1.name / "level.toml").read_text(
        encoding="utf-8"))
    assert "levelgen" not in raw
    assert level_loader.load(v1.name, levels_dir=str(tmp_path)).levelgen \
        is None
    with pytest.raises(ValueError, match="v1"):
        G.levelgen_record(v1)


# ---------------------------------------------------------------------------
# A ship is a pure function of its LevelSpec
# ---------------------------------------------------------------------------

def test_a_recipe_ship_is_a_pure_function_of_its_levelspec(tmp_path):
    """PROPERTY (archetype doc §5.1): a recipe ship written by the CLI and
    regenerated from its ``[levelgen]`` record ALONE (``--from-level``, no
    other argument) is the same level, file for file (a PNG by its decoded
    raster). BREAKS IF an input of ``generate`` is missing from the record,
    the record and the generator disagree on one (e.g. the base vs the
    scaled horde), or the regenerate path draws differently."""
    a = ARGS
    first, again = tmp_path / "first", tmp_path / "again"
    assert G.main(["--seed", str(a["seed"]), "--size", a["size"],
                   "--recipe", "freighter", "--profile", a["profile"],
                   "--zombies", str(a["zombies"]), "--out", str(first)]) == 0
    assert G.main(["--from-level", str(first), "--out", str(again)]) == 0
    assert _differ(first, again) == []


def test_a_stale_record_is_refused_by_name(plan):
    """PROPERTY: a record whose ``recipe_hash`` is not the current
    library's (the recipe was retuned since) is refused by name, since its
    ship can no longer be reproduced -- unless explicitly allowed, which
    generates today's ship from the same inputs. BREAKS IF the hash is not
    compared, or the refusal cannot be overridden."""
    d = _record_dict(plan)
    d["recipe_hash"] = "0" * 64
    rec = level_loader.parse_levelgen(d, "test")
    with pytest.raises(G.StaleRecipeError, match="recipe_hash.*retuned"):
        G.generate_from_spec(rec)
    import numpy as np
    assert np.array_equal(G.generate_from_spec(rec, allow_stale=True).tilemap,
                          plan.tilemap)


def test_the_realised_factors_agree_with_the_ship(plan, written):
    """PROPERTY: the recorded factors are the ship's own, read from the
    WRITTEN level where it has them: the spawn distance is the bot path the
    ``[validation]`` gate measures between the loaded spawn rooms on the
    loaded tilemap; the horde is match.toml's and the zombie room's; the
    departments, cycle applications, secrets and exits are the quotient
    graph's; ``(g, a)``, the cells and the rooms are the plan's; the inputs
    are the request plus the library's recipe hash. BREAKS IF a factor is
    measured a second way, counted on the wrong graph, or recorded before the
    profile's scaling."""
    _base, lv = written
    rec, q = lv.levelgen, plan.quotient
    loaded = SimpleNamespace(tilemap=lv.tilemap, spawn_rooms=lv.spawn_rooms)
    path = L.bot_path(loaded)
    assert path and rec.spawn_distance == len(path) - 1
    zroom = next(r for r in lv.spawn_rooms if r.team == ZOMBIE_TEAM)
    assert rec.zombies == lv.match.zombies == zroom.max_per_team
    arch = S.load_archetype("freighter")
    assert rec.zombies == arch.scaled_zombies(rec.profile, rec.zombies_base)
    assert list(rec.departments) == [d.name for d in q.departments]
    assert {rm.tags.get("dept") for rm in plan.graph.rooms
            if rm.tags.get("dept")} == set(rec.departments)
    assert sum(rec.cycles.values()) == len(q.cycles)
    assert {c.type for c in q.cycles} <= set(rec.cycles)
    assert {e.cycle for e in arch.applies if e.cycle} <= set(rec.cycles)
    assert rec.secrets == len(q.secrets) and rec.spawn_exits == q.exits
    assert (rec.g, rec.a) == (plan.g, plan.attempt)
    assert rec.rooms == len(plan.graph.rooms)
    assert rec.cells == len({rm.tags.get("cell") for rm in plan.graph.rooms})
    assert (rec.archetype, rec.size, rec.profile, rec.level_seed,
            rec.zombies_base, rec.max_per_team) == (
        "freighter", ARGS["size"], ARGS["profile"], ARGS["seed"],
        ARGS["zombies"], 6)
    assert rec.recipe_hash == S.recipe_hash(arch, rec.profile)
    hard = G.generate(0, "medium", recipe=arch, profile="hard", zombies=4)
    hrec = G.levelgen_record(hard, "test-build")
    assert (hrec.zombies_base, hrec.zombies) == (4, hard.zombies)
    assert hrec.zombies == arch.scaled_zombies("hard", 4) != 4


# ---------------------------------------------------------------------------
# The squad stands in the middle of its slab
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed, size", [(1, "medium"), (2, "large")])
def test_the_spawn_boxes_stand_centred_in_their_rooms(seed, size):
    """PROPERTY (recipe path): each spawn room written is the LEAST box that
    seats its ``max_per_team`` under ``formation`` across its room's short
    side, centred in that room within one tile along each axis, so every
    formation cell (the squad, the horde) stands in the room's middle and
    not at its fore end; the two teams' boxes are mirror twins; and the
    box's ``_centre`` (validate's and the bot path's A* endpoint) is the
    room's own, so centring moves no spawn distance and no gate verdict.
    BREAKS IF the recipe path writes the whole slab (or a section) as the
    spawn room again, the box drifts off the centre or loses the room's
    parity, or the mirror is lost."""
    plan = G.generate(seed, size, recipe="freighter", profile="standard",
                      zombies=4)
    W = plan.tilemap.shape[1]
    rooms = {r.team: r for r in plan.spawn_rooms}
    hosts = {rm.tags["spawn_team"]: rm for rm in plan.graph.rooms
             if "spawn_team" in rm.tags}
    hosts[ZOMBIE_TEAM] = next(rm for rm in plan.graph.rooms
                              if rm.tags.get("zombie"))
    for team, host in hosts.items():
        box = rooms[team]
        assert host.x <= box.x and box.x + box.w <= host.x + host.w
        assert host.y <= box.y and box.y + box.h <= host.y + host.h
        assert abs((2 * box.x + box.w) - (2 * host.x + host.w)) <= 2, team
        assert abs((2 * box.y + box.h) - (2 * host.y + host.h)) <= 2, team
        whole = box.__class__(team=team, x=host.x, y=host.y, w=host.w,
                              h=host.h, max_per_team=1)
        assert G._centre(box) == G._centre(whole), team   # A* endpoints kept
        assert formation_capacity(box) >= box.max_per_team
        vertical = host.h >= host.w
        shorter = (box.__class__(team=team, x=box.x, y=box.y, w=box.w,
                                 h=box.h - 5, max_per_team=1) if vertical
                   else box.__class__(team=team, x=box.x, y=box.y,
                                      w=box.w - 5, h=box.h, max_per_team=1))
        if (shorter.h if vertical else shorter.w) > 0:
            assert formation_capacity(shorter) < box.max_per_team, team
        for (cx, cy) in placement_cells(box, box.max_per_team):
            assert host.x <= cx and cx + 3 <= host.x + host.w
    a, b = rooms[0], rooms[1]
    assert (b.x, b.y, b.w, b.h) == (W - a.x - a.w, a.y, a.w, a.h)


def test_the_spawn_box_keeps_its_rooms_centre_on_hand_built_rooms():
    """PROPERTY: ``centred_spawn_box`` on any room (tall or wide, odd or
    even long side, any slot budget it can seat) returns a box inside the
    room that seats the budget, whose ``_centre`` is the room's exactly and
    whose middle is the room's within one tile. BREAKS IF the parity
    adjustment goes (an even box in an odd room moves the A* endpoint), or
    the box is cut along the short side."""
    from simulation.match import SpawnRoom
    for short in (3, 8, 9, 10, 13):
        for long_ in range(short + 1, 64):
            for n in range(1, 9):
                for tall in (True, False):
                    w, h = (short, long_) if tall else (long_, short)
                    room = SpawnRoom(team=0, x=3, y=4, w=w, h=h,
                                     max_per_team=n)
                    if formation_capacity(room) < n:
                        continue
                    box = G.centred_spawn_box(room)
                    assert formation_capacity(box) >= n, (room, box)
                    assert G._centre(box) == G._centre(room), (room, box)
                    assert room.x <= box.x and box.x + box.w <= room.x + w
                    assert room.y <= box.y and box.y + box.h <= room.y + h
                    assert (box.w, box.h)[1 - tall] == short   # full short side
                    assert abs((2 * box.x + box.w) - (2 * room.x + w)) <= 2
                    assert abs((2 * box.y + box.h) - (2 * room.y + h)) <= 2


# ---------------------------------------------------------------------------
# The lamps fit the frame's light door (found by the first large ship)
# ---------------------------------------------------------------------------

def test_the_lamp_budget_is_the_light_doors_own_arithmetic():
    """PROPERTY: gen_ship's lamp budget is derived from the renderer's
    light door, not a free number: the cone budget in light units (2^44 at
    L°'s currency) less the muzzle flashes' own cap less a reserve that
    covers the cursor lamp. BREAKS IF either renderer constant moves without
    gen_ship following (a ship then trips ``cone_rows`` mid-firefight)."""
    bp = pytest.importorskip("breach_physics")
    from renderer import frame_lights as fl
    units = fl.LIGHT_CONE_BUDGET / float(1 << int(bp.L_FINE_BITS))
    assert G.CONE_LIGHT_UNITS == units
    assert G.MUZZLE_FLASH_LIGHT == fl.MUZZLE_FLASH_MAX_TOTAL
    cursor = fl.CURSOR_INTENSITY * max(fl.CURSOR_COLOR)
    assert G.LAMP_LIGHT_BUDGET + G.MUZZLE_FLASH_LIGHT + cursor <= units


@pytest.mark.parametrize("seed, size", [(1, "medium"), (2, "large")])
def test_a_recipe_ships_lamps_survive_a_full_volley(seed, size):
    """PROPERTY: a recipe ship's lamps, plus a full volley of muzzle flashes
    (their cap) plus the cursor lamp, pass the frame's light door
    (``frame_lights.cone_rows``) -- the door main.py refused the first large
    freighter at, on frame 3, with its lamps alone; every lit room keeps at
    least one lamp; the lamps are mirrored. BREAKS IF the lamp budget is
    dropped, the budget pass empties a room, or it breaks the mirror."""
    bp = pytest.importorskip("breach_physics")
    from renderer import frame_lights as fl
    plan = G.generate(seed, size, recipe="freighter", profile="standard",
                      zombies=4)
    lamps = plan.embedding.lights
    assert G._lamp_light(lamps) <= G.LAMP_LIGHT_BUDGET + 1e-9
    h, w = plan.tilemap.shape
    extra = [fl.cursor_lamp_spec((w / 2.0, h / 2.0))]
    extra += fl.transient_specs([{"x": w / 2.0, "y": h / 2.0,
                                  "intensity": fl.MUZZLE_FLASH_MAX_TOTAL,
                                  "color": (1.0, 1.0, 1.0)}])
    specs = fl.frame_light_specs(lamps, total_tick=0,
                                 sim_time_per_tick=1.0 / 60, extra=extra)
    fl.cone_rows(specs.specs, w, h, int(bp.L_FINE_BITS))      # no refusal
    for rm in plan.graph.rooms:
        lit = "light" not in rm.tags or rm.tags["light"][1] > 0
        inside = [l for l in lamps if rm.x <= l.x < rm.x + rm.w
                  and rm.y <= l.y < rm.y + rm.h]
        assert bool(inside) == lit, rm.id
    spots = {(round(l.x, 6), round(l.y, 6), l.intensity) for l in lamps}
    assert spots == {(round(w - x, 6), y, i) for (x, y, i) in spots}


def test_v1_spawn_rooms_are_their_whole_rooms():
    """PROPERTY: the spawn box is the RECIPE path's; v1 still writes each
    spawn room as its whole graph room (ship_demo's spawn rooms do not move;
    its bytes are held by test_gen_ship_edges). BREAKS IF the centring leaks
    into ``embed`` or the v1 loop."""
    plan = G.generate(1, "medium", vents=True)
    for r in plan.spawn_rooms:
        if r.team == ZOMBIE_TEAM:
            host = next(rm for rm in plan.graph.rooms if rm.tags.get("zombie"))
        else:
            host = next(rm for rm in plan.graph.rooms
                        if rm.tags.get("spawn_team") == r.team)
        assert (r.x, r.y, r.w, r.h) == (host.x, host.y, host.w, host.h)
