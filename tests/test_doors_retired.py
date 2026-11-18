"""#215 (Erik, 2026-11-18): the legacy painted door is retired.

Every doorway is a door entity (closed at the start), the loader refuses a
bare painted door (``MAT_DOOR`` under no door's span), and the ship
generator's door rows emit closed entity doors. Design and census:
``docs/doors_retire_painted_impl_2026-11-18.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
from simulation import Simulation, combat  # noqa: E402
from simulation.door_system import DOOR_CLOSED, DOOR_OPEN  # noqa: E402
from simulation.entities.door import runtime_span  # noqa: E402
from simulation.materials import MAT_DOOR, MAT_DOOR_CLOSED  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402


def _levels():
    return sorted(p.parent.name for p in (ROOT / "levels").glob("*/level.toml"))


@pytest.mark.parametrize("name", _levels())
def test_no_level_in_the_repo_holds_a_bare_painted_door(name):
    """Property: no level under levels/ carries a ``MAT_DOOR`` tile outside
    every door entity's span (read from the files themselves, the loader's
    refusal switched off so the check is the test's own). Breaks when a
    level with a painted door is added or a generator emits one again."""
    lvl = level_loader.load(name, allow_bare_painted_doors=True)
    spans = set()
    for e in lvl.entities:
        if e.class_name == "door":
            spans.update(runtime_span(e.fields, lvl.tile_size_m, 1))
    bare = [(int(y), int(x)) for y, x in np.argwhere(lvl.tilemap == MAT_DOOR)
            if (int(y), int(x)) not in spans]
    assert bare == [], (name, bare[:5])


def _copy_level(tmp_path, name):
    import shutil
    d = tmp_path / name
    shutil.copytree(ROOT / "levels" / name, d)
    return d


def test_the_loader_refuses_a_bare_painted_door_by_name(tmp_path):
    """Property: one ``MAT_DOOR`` tile painted outside any door span makes
    ``level_loader.load`` refuse the level, naming the painted door and the
    migration tool; the same tile under a door's span loads (the door's
    stamp overrides it), and the migration's read may still open it. Breaks
    if the legacy kind can load again or a span's own paint is refused."""
    d = _copy_level(tmp_path, "door_test")
    lvl = level_loader.load(str(d))
    door = next(e for e in lvl.entities if e.class_name == "door")
    (sy, sx) = runtime_span(door.fields, lvl.tile_size_m, 1)[0]
    tm = lvl.tilemap.copy()
    tm[sy, sx] = MAT_DOOR                         # under a span: allowed
    level_lib.write_tilemap_csv(d, tm, csv_bak=False)
    level_loader.load(str(d))
    spans = set()
    for e in lvl.entities:
        if e.class_name == "door":
            spans.update(runtime_span(e.fields, lvl.tile_size_m, 1))
    wall = [(int(y), int(x)) for y, x in np.argwhere(tm == 1)
            if (int(y), int(x)) not in spans]
    tm[wall[len(wall) // 2]] = MAT_DOOR           # bare: refused
    level_lib.write_tilemap_csv(d, tm, csv_bak=False)
    with pytest.raises(ValueError, match="bare painted door.*migrate_level_entities"):
        level_loader.load(str(d))
    level_loader.load(str(d), allow_bare_painted_doors=True)


def test_gen_ship_door_rows_emit_closed_door_entities():
    """Property: every ``door`` / ``vented_door`` edge of a generated ship is
    ONE closed door entity over exactly its opening, stamped
    ``MAT_DOOR_CLOSED``; nothing stamps ``MAT_DOOR``. Breaks if the door rows
    paint again or emit an open door."""
    import gen_ship as G
    for kind in ("door", "vented_door"):
        assert G.EDGE_KINDS[kind].entity == "door"
        assert G.door_initial(G.EDGE_KINDS[kind]) == "closed"
    lvl = level_loader.load("ship_demo")
    assert not (lvl.tilemap == MAT_DOOR).any()
    doors = [e for e in lvl.entities if e.class_name == "door"]
    assert doors and all(e.fields["initial_state"] == "closed" for e in doors)
    span = set()
    for e in doors:
        span.update(runtime_span(e.fields, lvl.tile_size_m, 1))
    assert span == {(int(y), int(x)) for y, x in
                    np.argwhere(lvl.tilemap == MAT_DOOR_CLOSED)}


def _blocks(sim, door):
    """What the door's span does to each passer, read through the game's
    own seams: (rounds = solid, grenades = flight_blocked, gas = solid with
    permeability 0, sight = has_los across, movement = is_passable)."""
    g = sim.gmap
    span = door.span
    (y0, x0), (y1, x1) = span[0], span[-1]
    if y0 == y1:                               # a horizontal door: cross N-S
        a, b = (y0 - 2, (x0 + x1) // 2), (y0 + 2, (x0 + x1) // 2)
    else:
        a, b = ((y0 + y1) // 2, x0 - 2), ((y0 + y1) // 2, x0 + 2)
    return dict(
        rounds=all(bool(g.solid[t]) for t in span),
        grenades=all(combat.flight_blocked(g, t[0], t[1]) for t in span),
        gas=all(float(g.permeability[t]) <= 0.0 for t in span),
        sight=not g.has_los(a[0], a[1], b[0], b[1]),
        movement=not any(g.is_passable(t[0], t[1]) for t in span))


def test_a_migrated_doorway_blocks_everything_closed_and_nothing_open():
    """Property: a migrated ship_demo doorway (a closed door entity) blocks
    rounds, grenades, gas, sight and movement; used open it blocks none of
    them; used again it closes and blocks them all again. Breaks if the
    entity door lets one through closed or keeps one out open."""
    sim = Simulation(level_loader.load("ship_demo"), seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    door = sim._doors[0]
    assert door.state == DOOR_CLOSED
    assert all(_blocks(sim, door).values()), _blocks(sim, door)
    door.want_open = True
    sim.step()
    assert door.state == DOOR_OPEN
    assert not any(_blocks(sim, door).values()), _blocks(sim, door)
    door.want_open = False
    sim.set_paused(False)
    sim.step()
    assert door.state == DOOR_CLOSED
    assert all(_blocks(sim, door).values())
