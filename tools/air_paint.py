r"""tools/air_paint.py — the AIR wand tool's hull-leak-gated enclosure fill
(Arc C5, editor doc §7, canon engine/16 §7 / A9).

`air_init.npy` governs the WHOLE map once it exists (there is no per-cell
partial-override flag) — `simulation.gamemap.GameMap` applies it to every
open interior tile, and its own pre-override default is `FP_ONE`
(1.0 atm) everywhere (`np.full((h, w), FP_ONE)`, gamemap.py's atmosphere
init). So a brand-new `air_init.npy` MUST seed every tile to that same
default before the user's own paint lands on top — else painting one room
would silently zero-pressure the rest of a previously-unpainted map. See
:func:`default_ambient_grid`.

The hull-leak validator (editor doc §7: "fill escaping to the border = warn,
don't paint") walks the SAME connectivity `tools/level_airtight.py` already
uses to detect a leaky level: flood over every NON-SOLID tile (walls block,
vacuum/SPACE does NOT — gas physically reaches vacuum through exactly that
path) and refuse whenever the flood reaches the grid border. Only a SEALED
fill is ever handed back to the caller to paint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from level_edit_common import (enclosure_fill_region,  # noqa: E402
                               region_touches_border)

FP_ONE = 1 << 16          # 1.0 atm in Q16.16 counts (gamemap's own unit)


def default_ambient_grid(shape) -> np.ndarray:
    """A brand-new `air_init.npy`'s seed: `FP_ONE` (1.0 atm) EVERYWHERE.

    `gamemap.py` only ever applies an air_init override to OPEN interior
    tiles (solid/vacuum/ambient-ring cells are excluded regardless of what
    the array holds there — `_seed_air_override`'s own `open_air` mask),
    and the engine's pre-override default there is exactly `FP_ONE` — so an
    explicit array that is `FP_ONE` everywhere except the tiles the user
    actually painted is bit-identical, everywhere else, to the level having
    no `air_init.npy` at all (`level_lib.write_air_init_npy`'s own "an
    explicit all-ambient grid is bit-identical to no grid" note)."""
    h, w = shape
    return np.full((h, w), FP_ONE, dtype=np.int32)


def quantize_atm(atm) -> int:
    """Float atm -> Q16.16 int counts, round-half-up, clamped `>= 0` (the
    loader hard-errors on negative pressures — `air_init.npy` never carries
    one)."""
    q = int(np.floor(float(atm) * FP_ONE + 0.5))
    return max(0, q)


def plan_air_fill(grid, solid_codes, tx: int, ty: int):
    """The hull-leak-gated enclosure fill (editor doc §7). Floods from
    `(tx, ty)` over every tile that is NOT solid-for-air (`solid_codes` —
    the sim-exact permeability<=0 set, `map_editor.water_solid_codes`'s own
    seam) — vacuum/SPACE tiles DO NOT block the flood (a breach's gas
    physically reaches vacuum, so the check must be able to walk through
    it, exactly like `tools/level_airtight.py`'s own leak detector) — then
    refuses whenever the flood reaches the grid BORDER.

    Returns `(region, why)`: `region` (a frozenset of `(tx, ty)`) is
    non-`None` ONLY on a SEALED fill — a leaky one always comes back with
    `region is None`, so the caller can never accidentally paint a region
    that escaped the hull.
    """
    g = np.asarray(grid)
    solid = np.isin(g, np.asarray(sorted(solid_codes), dtype=g.dtype))
    passable = ~solid
    region = enclosure_fill_region(passable, tx, ty)
    if region is None:
        return None, "start tile is solid"
    if region_touches_border(region, g.shape[1], g.shape[0]):
        return None, ("leaky room — fill reaches the map border "
                      "(hull-leak validator): NOT painted")
    return region, "ok"
