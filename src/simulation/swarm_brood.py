"""The brood spawn (arc #63 P4, engine/17 §7 / §11): THE load-time spawn of
swarm units from the level's ``swarm_brood`` zone entities.

At level load, before tick 0, every brood scatters ``count`` units of its
``species`` over its zone's FREE (non-solid) tiles. Each unit's tile, its
position inside the tile and its heading are drawn from the match draws
(``sim.draws``, one source of draws, Erik's ruling) under ONE salt,
``Salt.SWARM_BROOD``, keyed on (match seed, the brood's entity ordinal, the
unit's index k): draw ``4 k + j`` of site (SWARM_BROOD, ordinal) at tick 0,
j = tile, x in the tile, y in the tile, heading. So a brood's scatter depends
on nothing else in the level, and two broods never share a key.

Broods fill slots in entity (ordinal) order and their units in k order, so a
unit's ``unit_id`` is reproducible. The spawn itself goes through
:meth:`simulation.swarm.SwarmUnits.spawn` and nothing else.

Refusals, by name and with NOTHING spawned (every check runs before the
first write): a brood whose zone has no free tile and a species that does
not exist (:class:`BroodRefused`; the level lint reports both through
:func:`brood_problems`), and a total over a species' capacity
(:class:`simulation.swarm.SwarmFull`). Never a clamp, never a partial fill.

The brood never ticks and leaves no runtime state: a level without one
draws nothing and spawns nothing (its digest is unchanged).

Called from ONE place, :meth:`Simulation._reset_internal`, which every path
a match starts through constructs (the played game, ``tools/run_match.py``,
``MatchSession``, each env of the batched environment at its reset).
"""
from __future__ import annotations

import numpy as np

from simulation import swarm_fixed
from simulation.philox32 import philox32_angle_q16_np, philox32_below_np
from simulation.rng_salts import Salt
from simulation.swarm import SWARM_SPECIES, SwarmFull, swarm_column

BROOD_CLASS = "swarm_brood"

# The draws one unit takes, in this order (draw index 4 k + j).
DRAW_TILE, DRAW_SUB_X, DRAW_SUB_Y, DRAW_HEADING = 0, 1, 2, 3
DRAWS_PER_UNIT = 4


class BroodRefused(ValueError):
    """A brood that cannot spawn (no free tile in its zone, or an unknown
    species), raised before anything is written."""


def broods_of(entities) -> list:
    """The level's brood instances in ordinal order."""
    return sorted((e for e in entities if e.class_name == BROOD_CLASS),
                  key=lambda e: int(e.ordinal))


def free_tiles(zone_grid, solid, brood) -> np.ndarray:
    """The brood's FREE tiles, ``(n, 2)`` int64 ``(y, x)`` in row-major order:
    the tiles its ``zone_id`` paints that are not solid. ``zone_grid`` is the
    level's paint grid at the runtime grid's shape (``--res`` replicates it),
    or None (no zones painted: no tile)."""
    if zone_grid is None:
        return np.zeros((0, 2), dtype=np.int64)
    mask = (np.asarray(zone_grid) == int(brood.fields["zone_id"])) \
        & ~np.asarray(solid, dtype=bool)
    return np.argwhere(mask).astype(np.int64)


def brood_problems(zone_grid, solid, entities) -> list:
    """Every authoring problem of the level's broods, by name: an unknown
    species, a zone with no free tile. Empty when every brood can spawn.
    The spawn refuses on the same list; the level lint prints it."""
    out = []
    for b in broods_of(entities):
        sp = b.fields.get("species")
        if sp not in SWARM_SPECIES:
            out.append(f"swarm_brood {b.id!r}: species {sp!r} does not exist "
                       f"(known: {SWARM_SPECIES})")
        if free_tiles(zone_grid, solid, b).shape[0] == 0:
            out.append(f"swarm_brood {b.id!r}: zone_id "
                       f"{int(b.fields['zone_id'])} has no free (non-solid) "
                       f"tile to spawn on")
    return out


def scatter(draws, ordinal: int, count: int, tiles: np.ndarray):
    """The brood's ``count`` units over ``tiles`` (its free tiles): Q16.16
    ``(pos_x, pos_y, heading)`` int64 arrays in k order, drawn at the draws'
    current tick (load time: tick 0). The tile is ``below(w, n)`` over the
    tile list, the position inside it the top 16 bits of a word on each axis
    (so ``[tile, tile + 1)``), the heading an angle draw, then wrapped."""
    count = int(count)
    n = int(tiles.shape[0])
    if n == 0:
        raise BroodRefused(f"brood ordinal {ordinal}: no free tile")
    if not 0 < count * DRAWS_PER_UNIT < (1 << 32):
        raise BroodRefused(f"brood ordinal {ordinal}: count {count} outside "
                           f"the draw index range")
    idx = np.arange(count * DRAWS_PER_UNIT, dtype=np.uint32)
    w = draws.words_at(Salt.SWARM_BROOD, int(ordinal), idx).reshape(
        count, DRAWS_PER_UNIT)
    pick = philox32_below_np(w[:, DRAW_TILE], np.uint32(n)).astype(np.int64)
    ty, tx = tiles[pick, 0], tiles[pick, 1]
    sub_x = (w[:, DRAW_SUB_X] >> np.uint32(16)).astype(np.int64)
    sub_y = (w[:, DRAW_SUB_Y] >> np.uint32(16)).astype(np.int64)
    pos_x = (tx << swarm_fixed.FP_SHIFT) + sub_x
    pos_y = (ty << swarm_fixed.FP_SHIFT) + sub_y
    heading = swarm_fixed.wrap_heading_q16(
        philox32_angle_q16_np(w[:, DRAW_HEADING]).astype(np.int64))
    return pos_x, pos_y, heading


def spawn_broods(sim) -> list:
    """Spawn every brood of ``sim``'s level into ``sim.swarm`` (THE seam,
    called once per construction / reset). Returns ``[(brood id, slots)]`` in
    ordinal order; a brood-free level returns ``[]`` and draws nothing."""
    broods = broods_of(sim.entities)
    if not broods:
        return []
    if int(sim.draws.tick) != 0:
        raise BroodRefused(f"broods spawn at load time (tick 0), the draws "
                           f"read tick {int(sim.draws.tick)}")
    zone_grid = getattr(sim.level, "zone_grid", None)
    problems = brood_problems(zone_grid, sim.gmap.solid, broods)
    if problems:
        raise BroodRefused("; ".join(problems))
    need: dict = {}
    for b in broods:
        need[b.fields["species"]] = need.get(b.fields["species"], 0) \
            + int(b.fields["count"])
    for sp, k in need.items():
        free = int((swarm_column(sim.gmap, sp, "valid") == 0).sum())
        if k > free:
            raise SwarmFull(f"swarm_brood: the level's broods ask {k} "
                            f"{sp!r} units but the store has {free} free "
                            f"slot(s) ([swarm.{sp}] max_units)")
    out = []
    for b in broods:
        tiles = free_tiles(zone_grid, sim.gmap.solid, b)
        px, py, hd = scatter(sim.draws, int(b.ordinal), int(b.fields["count"]),
                             tiles)
        out.append((b.id, sim.swarm.spawn(b.fields["species"], px, py, hd)))
    return out


__all__ = ["BROOD_CLASS", "BroodRefused", "broods_of", "free_tiles",
           "brood_problems", "scatter", "spawn_broods"]
