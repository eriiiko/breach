"""Roof lights as shapes (#177): a recipe ship's lamps drawn from
``[levelgen.lamp_shapes.*]``, each shape's tiles sharing the lamp's light."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tools", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gen_ship as G  # noqa: E402
from config import CFG  # noqa: E402

RING = ["****", "*  *", "*  *", "****"]


def _graph(w, h, shape, x=10, span=12):
    W = 2 * x + w + 20
    rm = lambda i, rx, m: G.Room(i, rx, 5, w, h, "room",                  # noqa: E731
                                 {"lamp_span": span, "lamp_shape": shape}, mirror=m)
    return G.ShipGraph(width=W, height=h + 10, hull=(1, 1, W - 2, h + 8),
                       rooms=[rm(0, x, 1), rm(1, W - x - w, 0)], edges=[],
                       mirror_x=True, max_per_team=4, zombies=0)


def _cells(lamps):
    return {(int(l.x), int(l.y)) for l in lamps}


@pytest.mark.parametrize("w, h", [(10, 10), (11, 10), (10, 11), (11, 11)])
def test_a_named_shape_is_drawn_centred_on_its_lamp(w, h):
    """PROPERTY: ``ring4`` lands as drawn: hollow, exactly centred in the room
    at either parity of room size (its middle line doubled at the odd ones),
    its tiles sharing one lamp's light. BREAKS IF a shape lands a half tile
    off centre, fills in, or spends more light than its lamp."""
    g = _graph(w, h, "ring4")
    lamps = G._shaped_lamps(g, g.rooms[0], G.lamp_shapes())
    cells = _cells(lamps)
    assert len(cells) == len(lamps) >= 12
    xs, ys = sorted({c[0] for c in cells}), sorted({c[1] for c in cells})
    assert xs[0] + xs[-1] + 1 == 2 * g.rooms[0].x + w
    assert ys[0] + ys[-1] + 1 == 2 * g.rooms[0].y + h
    assert (xs[1], ys[1]) not in cells
    plain = G._room_lamps(g.rooms[0])
    assert sum(l.intensity for l in lamps) == pytest.approx(sum(l.intensity for l in plain))


def test_a_room_too_small_for_its_shape_keeps_its_lamp():
    """PROPERTY: a shape never touches its room's wall: no fit, one plain lamp.
    BREAKS IF the margin check goes."""
    g = _graph(5, 5, "ring4")
    assert G._shaped_lamps(g, g.rooms[0], G.lamp_shapes()) == G._room_lamps(g.rooms[0])


@pytest.mark.parametrize("seed, size", [(1, "medium"), (7, "large"), (3, "medium")])
def test_a_recipe_ship_draws_shapes_and_keeps_its_light_and_mirror(seed, size):
    """PROPERTY: on a generated ship every lit room keeps exactly the light of
    its plain lamps, its lamp tiles inside the room; the lights mirror; at
    least three shapes appear (the library is drawn from). BREAKS IF a shape
    changes a room's light, leaves its room, or a twin stops being the
    reflection."""
    plan = G.generate(seed, size, recipe="freighter", profile="standard", zombies=4)
    lamps, W = plan.embedding.lights, plan.tilemap.shape[1]
    sizes = set()
    for rm in plan.graph.rooms:
        inside = [l for l in lamps if rm.x <= l.x < rm.x + rm.w
                  and rm.y <= l.y < rm.y + rm.h]
        plain = G._room_lamps(rm)
        assert sum(l.intensity for l in inside) == pytest.approx(
            sum(l.intensity for l in plain), rel=1e-9), rm.id
        if plain:
            sizes.add(round(len(inside) / len(plain)))
    assert len(sizes) >= 3
    spots = {(round(l.x, 6), round(l.y, 6)) for l in lamps}
    assert spots == {(round(W - x, 6), y) for (x, y) in spots}
    assert len(spots) == len(lamps)


def test_the_door_refuses_a_bad_shape(monkeypatch):
    """PROPERTY: a mask that does not mirror onto itself, a row with both forms
    or neither, a stray character, are refused by name. BREAKS IF a lopsided
    mask reaches a twin room (its reflection would be another shape)."""
    for row in (NS(mask=["** ", "** "]), NS(mask=RING, width=1, fill=0.5), NS(),
                NS(mask=["#"]), NS(width=0, fill=0.5), NS(width=1, fill=0.0),
                NS(mask=["**", "*"])):
        monkeypatch.setattr(CFG.levelgen, "lamp_shapes", NS(bad=row))
        with pytest.raises(ValueError, match="bad"):
            G.lamp_shapes()
