# Patch Plan — Coord System Cleanup

> **Status:** Plan locked, implementation pending.
> **Drafted:** 2026-05-20 (late Wednesday — orchestrator session).
> **Author intent:** Erik. Execution: implementation agent next session.
> **Sibling docs:** `architecture.md` §3, §16.10 · `unit_variants_design_brainstorm.md`

## 1. Motivation

The Breach codebase carries leftover terminology from an early design where
units moved on their own coarse-tile grid (1 m × 1 m blocks) layered on top of
a finer physics-tile grid (1/3 m). **That dual-grid design was abolished long
ago.** Today there is **one grid** — the physics-tile grid. Units occupy a
3×3 block of physics tiles, but units are not on a different grid.

The dead vocabulary (`coarse`, `fine`, `cx/cy`, `co`, `fine_w`, `coarse_px`,
…) still permeates production code and creates real confusion:

- `Unit(cx, cy)` takes "coarse" coords; constructor does `self.fx = cx * co`.
  A reader reasonably asks: what's `cx` — coarse? continuous? float? — and
  what is `co`?
- The integer field is named `fx` (where `f` stood for "fine"), but the
  float field is named `fxf` (where the trailing `f` means "float"). Two
  different meanings of `f` in one Unit class.
- `map_w` / `map_h` in `config.toml` look like physics-grid dimensions but
  are actually "coarse-block" dimensions (`fine_w = map_w * coarse`).
- `coarse_px` = "pixels per coarse block in the dead pygame renderer" —
  dead, name didn't survive contact with the new pyray renderer.

A specific consequence: **F6 cursor coords HUD shows physics-tile coords**
(what the user reads), **but `Unit(cx, cy)` interprets those numbers as
coarse coords** (multiplying by 3). Spawn-coord entries in `level.toml`
following the cursor readout landed 3× outside the map.

## 2. Canon (the model we are landing on)

**One grid, one coordinate system.**

- The grid is the **physics-tile grid**. Tile size = `level.tile_size_m` m
  (currently 1/3 m). Per-level — the renderer + sim read this from
  `LevelData`, not from `config.toml`.
- Physics matrices (atmosphere, smoke, fire, light, …) are `[tiles_h, tiles_w]`.
- A unit's **position is a single (x, y) pair in tile units**. Float when
  smooth motion / rendering interpolation is involved, integer when used
  as a matrix index — both are the same coord system, the integer is just
  `int(x), int(y)`.
- A unit's **footprint is a property of the unit**, not a global constant.
  Default `footprint = 3`. When the unit-variant system lands
  ([[unit_variants_design_brainstorm]]), big units get `footprint = 4` or
  larger; small units get `2`. The simulation code never says "3" outside
  the Unit constructor's default.
- The relationship to meters and to matrix indices is **explicit**, via
  helpers in `src/simulation/coords.py`. The rest of the code stays in
  tile units; meter conversions happen only when physically meaningful
  (forces, damage falloff in meters, etc.).

## 3. Naming map

Renames apply across the entire `src/`, `renderer/`, `level_loader.py`,
`main.py`, `input_handler.py`, `tests/`, and `docs/`.

| Today                | New name                              | Type        | Meaning                                  |
|----------------------|----------------------------------------|-------------|-------------------------------------------|
| `unit.fx`, `unit.fy` | **`unit.x`, `unit.y`**                | `float`     | Position on the physics grid, top-left of footprint |
| `unit.fxf`, `unit.fyf` | (folded into `unit.x`, `unit.y`)    | —           | The float field IS the source of truth   |
| `unit.cx`, `unit.cy` (properties) | **delete**               | —           | Dead concept                              |
| `unit.tile_x`, `unit.tile_y` | property `int(self.x)`, `int(self.y)` | `int`       | Convenience for matrix-index access      |
| `unit.center_fx()`, `center_fy()` | `unit.center_tile_x()`, `center_tile_y()` | `int` | Top-left + `footprint // 2`        |
| `unit.get_center_px()` | renamed `center_screen_px()` if kept, otherwise **delete** | — | Pixel-space center — renderer concern    |
| `CFG.display.coarse` | **delete**, replaced by `unit.footprint` | —      | Per-unit, default 3 (in Unit constructor) |
| `CFG.display.fine_w`, `fine_h` | **delete**                   | —           | Grid size comes from the level's CSV     |
| `CFG.display.map_w`, `map_h` | **delete**                     | —           | Were "coarse-block" units, dead          |
| `CFG.display.coarse_px` | **delete**                          | —           | Dead pygame artifact                      |
| `CFG.display.fine_tile_px` | **delete** if unused, else investigate | — | Renderer uses `world_px_per_tile` now    |
| `SpawnEntry.cx`, `.cy` | **`SpawnEntry.x`, `.y`**             | `float`     | Float so future spawns can be sub-tile   |
| `[[spawn]] cx =`, `cy =` in `level.toml` | **`x =`, `y =`**     | float       |                                           |

**Why `x`, `y` (not `tile_x`, `tile_y`) on Unit:**
We discussed both. The source-of-truth field on Unit being plain `x`, `y` is
short for the most-read field. Integer matrix indexing happens via the
`tile_x` / `tile_y` properties (or `int(x)`, `int(y)` inline). Spawn entries
match (`x`, `y`).

## 4. Coords module

New file: **`src/simulation/coords.py`**

```python
"""Coordinate helpers for the physics-tile grid.

There is ONE grid — the physics-tile grid. A position is an (x, y) pair in
tile units. The integer form (row, col) for matrix indexing comes from
flooring; the meter form comes from multiplying by ``level.tile_size_m``.
These helpers exist so the conversions are explicit and centralized — if
the physics resolution (``tile_size_m``) changes, only meter-conversion
call sites are affected. Everything in tile units stays stable.
"""
from __future__ import annotations
from typing import Tuple


def tile_to_index(x: float, y: float) -> Tuple[int, int]:
    """(x, y) in tile units → (row, col) for matrix indexing.

    Returns (row, col) — row first because numpy is row-major. Floors,
    not rounds: tile (3.7, 2.1) belongs to cell (row=2, col=3).
    """
    return int(y), int(x)


def index_to_tile(row: int, col: int) -> Tuple[float, float]:
    """(row, col) matrix index → tile-unit coords at the cell's top-left."""
    return float(col), float(row)


def tile_to_meters(x: float, y: float, tile_size_m: float) -> Tuple[float, float]:
    """Tile-unit coords → meters."""
    return x * tile_size_m, y * tile_size_m


def meters_to_tile(mx: float, my: float, tile_size_m: float) -> Tuple[float, float]:
    """Meters → tile-unit coords."""
    return mx / tile_size_m, my / tile_size_m


def tile_distance_m(x1: float, y1: float, x2: float, y2: float,
                    tile_size_m: float) -> float:
    """Euclidean distance between two tile-unit positions, in meters."""
    dx = (x2 - x1) * tile_size_m
    dy = (y2 - y1) * tile_size_m
    return (dx * dx + dy * dy) ** 0.5
```

**Where the rest of the sim uses these:**

- Physics matrix access: `mat[*tile_to_index(unit.x, unit.y)]` or use the
  `unit.tile_x` / `unit.tile_y` properties for clarity.
- Damage falloff curves (radii in meters): convert tile-distance via
  `tile_distance_m`.
- The renderer keeps its own tile→screen-px mapping (camera handles that).
  Coords.py does not deal with pixels.

## 5. Unit class — target shape

```python
class Unit:
    """Game entity. Position lives in physics-tile units."""

    def __init__(self, name: str, x: float, y: float, team: int = 0,
                 footprint: int = 3):
        self.name = name
        self.team = team
        self.is_zombie = (team == 1)

        self.id = -1

        # Position on the physics-tile grid. Float so renderer can
        # interpolate; integer indexing via tile_x / tile_y properties.
        self.x = float(x)
        self.y = float(y)

        # Side length of the unit's square footprint, in physics tiles.
        # Default 3 (size-1 human). Will become a function of the variant
        # system later — see unit_variants_design_brainstorm.md.
        self.footprint = int(footprint)

        # ... rest unchanged (hp, ap, orders, zombie AI state, …)

    @property
    def tile_x(self) -> int:
        """Integer tile index (col) — for matrix access."""
        return int(self.x)

    @property
    def tile_y(self) -> int:
        """Integer tile index (row) — for matrix access."""
        return int(self.y)

    def center_tile_x(self) -> int:
        return self.tile_x + self.footprint // 2

    def center_tile_y(self) -> int:
        return self.tile_y + self.footprint // 2
```

**What disappears from Unit:**
- `self.fx`, `self.fy` (replaced by `self.x`, `self.y`)
- `self.fxf`, `self.fyf` (folded into `self.x`, `self.y`)
- `@property def cx`, `def cy`
- `def center_fx`, `def center_fy`
- `def get_center_px` (renderer concern — move to renderer if anyone still calls it)
- `from config import CFG` reads of `coarse` (we use `self.footprint`)

## 6. Per-file change list

These are the production files that touch dead vocab. For each, the change
shape is: rename fields, replace `CFG.display.coarse` with `unit.footprint`
(or `3` only inside Unit's default arg), drop the `*co` / `// co` arithmetic.

### Simulation layer

- **`src/simulation/unit.py`** — full rewrite of class signature per §5.
  Lines 40 (`__init__`), 41 (`co =`), 53–56 (`fx = cx * co` etc.),
  110, 115 (properties), 119–123, 127 (center_px). Drop the
  `from config import CFG` import for `display.coarse`.

- **`src/simulation/gamemap.py`**
  - L116 (docstring): "3×3 block" → "the unit's footprint block"
  - L121, L206: `co = CFG.display.coarse` → remove; read `unit.footprint`
  - L130–133, L153–157: iterate over `range(unit.footprint)` instead of
    `range(co)`.
  - L217–218: hull-breach edge check. The `co` value here was guarding a
    2-tile border. The check should be re-derived — see §8 ambiguity #3.

- **`src/simulation/simulation.py`**
  - L229 docstring: `(cx, cy)` → `(x, y)`.
  - L235–239: drop `co =` and `*co`. Spawn writes `unit.x = float(x)`, etc.
  - L383: `co =` → drop; `_compute_player_paths` uses `unit.footprint`.
  - L666–667, L682–685: `u.fxf = px; u.fyf = py` → `u.x = px; u.y = py`.
    Remove all fxf/fyf references.
  - L714, L724–725: `co =` → drop; grenade spawn center uses
    `unit.footprint // 2` from the throwing unit's footprint.

- **`src/simulation/combat.py`** — L249: `co =` → drop. Walk the file for
  any other dead references during the patch.

- **`src/simulation/ai_zombie.py`**
  - L52: `co =` → drop.
  - L113–114: melee range comment "coarse + 1 fine tile" → "footprint + 1
    tiles". The threshold becomes `unit.footprint + 1`.
  - L151–152: `z.fxf = float(next_x); z.fyf = float(next_y)` →
    `z.x = float(next_x); z.y = float(next_y)`.

### Input / orchestration

- **`src/input_handler.py`**
  - L159, L181: `co = CFG.display.coarse` → drop.
  - L164: unit-selection AABB uses `unit.footprint` instead of `co`.
  - L188–192: order-target snapping uses `unit.footprint // 2`.

- **`main.py`** — drives spawn from `level.spawns`. Currently builds
  `Unit(s.name, cx=s.cx, cy=s.cy, team=s.team)`. After: `Unit(s.name,
  x=s.x, y=s.y, team=s.team, footprint=s.footprint)` (footprint optional;
  defaults to 3).

### Level

- **`level_loader.py`**
  - `SpawnEntry` dataclass: `cx`, `cy` → `x`, `y` (both `float`). Add
    optional `footprint: int = 3`.
  - `load()` parser: read `x`, `y`, optional `footprint` from `[[spawn]]`
    table entries. Update the docstring near L33–34 to remove
    "coarse-tile" wording.

- **`levels/unhcr_vessel/level.toml`** — `[[spawn]]` entries: replace
  every `cx = N` / `cy = N` with `x = X` / `y = Y` using the **physics-tile
  coords Erik read off the F6 HUD**:
  - Alpha: `x = 24`, `y = 90`
  - Bravo: `x = 24`, `y = 94`
  - Cobra: `x = 24`, `y = 98`
  - Zomb1: `x = 36`, `y = 76` (`# TODO: footprint 4 once variants land`)
  - Zomb2: `x = 24`, `y = 77`
  - Zomb3: `x = 19`, `y = 78`
  - Zomb4: `x = 25`, `y = 16`

### Config

- **`config.toml`**
  - Delete the `[display]` keys: `coarse`, `fine_w`, `fine_h`, `map_w`,
    `map_h`. Audit whether `fine_tile_px` is referenced anywhere live;
    if not, delete it too.
  - Keep `[display] level = "unhcr_vessel"` (level selection).

- **`config.py`**
  - Delete the legacy fallback derivation block (lines 57–63 in current
    file): `fine_w = map_w * coarse`, `fine_h = map_h * coarse`,
    `coarse_px = fine_tile_px * coarse`.

### Renderer

The renderer references `unit.fx` / `unit.fy` in a few places to draw
units. Those reads become `unit.x` / `unit.y` (still float, just renamed).
The renderer is **not** the source of the dead vocab — once Unit is
renamed, the renderer follows mechanically.

- **`renderer/game_renderer.py`** — `_draw_units_world` reads `m.fx, m.fy` →
  `m.x, m.y`. (Verify during implementation that no `coarse`/`fine` words
  remain in this file.)
- **`renderer/overlays.py`** — same: `draw_unit(x_tile, y_tile, …)` already
  takes tile coords, just confirm call sites pass the renamed field.

### Tests

- The current test suite (`tests/test_simulation.py`,
  `tests/test_level_loader.py`) does not grep-hit the dead vocab, but
  it WILL break the moment `Unit.__init__` signature changes. The patch
  must update any `Unit(cx=…, cy=…)` calls inside tests (full search after
  the rename) and any `unit.fx` / `.fy` / `.fxf` / `.fyf` reads.

### Docs

- **`docs/architecture.md`**
  - §3 Grid & Coordinate System: replace `(fx, fy)` with `(x, y)`. Replace
    "fine tile" / "coarse tile" wherever it appears. The "Unit footprint |
    3×3 tiles" line stays but adds: "footprint is a unit property,
    default 3".
  - §14 Configuration: remove the `fine_w = map_w * coarse` block (lines
    945–947).
  - §16.10: the `fxf/fyf` removal note now reads "completed — `x`, `y`
    are the float source of truth, integer access via property". Or just
    delete the section.

- **`docs/unit_variants_design_brainstorm.md`** — §"Size → tile footprint
  (threshold rule, brainstorm)" already says "3×3 fine tiles (the 'coarse'
  block from `display.coarse`)". Update to "3-tile footprint, default for
  size-1 units. Implemented as `unit.footprint`."

- **`docs/patch_level_pipeline_v1.md`** — line 367 mentions removing the
  coarse-tile concept from game.py. game.py is being deleted anyway —
  leave the line, but the patch plan that this doc refers to is now
  this one.

- **Other archive docs** — no changes needed (archive folder is
  historical).

### Out of scope (do not touch)

- `game.py` (~70+ coarse references). Slated for deletion in migration
  step 13. Don't waste effort.
- `prototypes/archive/*` — exploratory, not production.
- `debug_physics.py` — standalone test utility. Will break with the
  config cleanup; **fix only if it's still in use**. Otherwise note in
  the implementation report.
- The C++ raycaster's `coarse_cluster` parameter — that's a fire-source
  clustering hint in the C++ code, semantically unrelated. Leave the
  C++ name; just remove the Python-side `raycaster.coarse_cluster =
  CFG.display.coarse` assignment in `debug_physics.py` (replace with a
  hardcoded value, e.g. 3, or delete if the utility is dead).

## 7. Step-by-step execution order

For an implementation agent:

1. **Snapshot & verify clean state**: `git status`, run `pytest tests/` —
   should be 8 passing. Note baseline.
2. **Add `src/simulation/coords.py`** with the helpers from §4. No callers
   yet — this is just laying the foundation. Tests still pass.
3. **Rewrite `src/simulation/unit.py`**: new constructor signature (`x, y,
   footprint`), drop `cx/cy/fxf/fyf`. Tests will break at this point.
4. **Update `simulation.py`, `gamemap.py`, `combat.py`, `ai_zombie.py`**
   to use `unit.x`, `unit.y`, `unit.footprint`. Drop every `co =`.
5. **Update `input_handler.py`** likewise.
6. **Update `level_loader.py`** — `SpawnEntry` field rename, parser, docs.
7. **Update `levels/unhcr_vessel/level.toml`** — replace `cx/cy` with
   `x/y` per the coord list in §6.
8. **Update `main.py`** — pass new field names.
9. **Update renderer reads** — `unit.fx` → `unit.x`, `unit.fy` → `unit.y`
   in `renderer/game_renderer.py` and anywhere else.
10. **Delete dead config keys** from `config.toml` and `config.py`.
11. **Update tests** — fix any `Unit(cx=, cy=)` calls and `.fx/.fy/.fxf/.fyf`
    reads.
12. **Run `pytest tests/` — 8 passing again.**
13. **Run smoke test** (`python tests/test_main_smoke.py --auto`) — should
    pass (60 ticks, 600 frames).
14. **Launch the game** (`python main.py`) — confirm:
    - Window opens at correct monitor size (still 1920×1200)
    - F6 HUD shows physics-tile coords
    - All 7 units spawn at the coords listed in §6 (visible on the map,
      no off-map disappearance)
    - Selecting a unit, issuing a move order, pressing Space — units
      walk toward target as before
15. **Update the three doc files** per §6 → docs section.
16. **Commit** in logical chunks (one per group of files makes sense —
    coords helper, Unit class, sim core, input/main, level, config,
    renderer, tests, docs). Each commit message: short summary + which
    decision from this plan it implements.
17. **Push.**
18. **Final report** to the orchestrator: which commits landed, any
    deviations from this plan and why, anything weird that came up.

## 8. Decisions / ambiguities flagged for the implementer

- **#1 — Float vs int for spawn coords.** Spawned `SpawnEntry.x/y` is
  `float` per §3. Marines today spawn at integer tiles; the float type
  is forward-compatible. The loader should accept TOML ints transparently
  (TOML's int → Python float is automatic when annotated `float`; verify
  on the actual implementation).

- **#2 — `unit.footprint` per-unit but never varied yet.** Default is 3.
  The variant system isn't implemented (see [[unit_variants_design_brainstorm]]).
  Until then, every spawn uses 3. Don't add machinery for footprint > 3
  yet — `stamp_units` etc. should *correctly handle* arbitrary footprint
  via `range(unit.footprint)`, but the variant-system tests come later.

- **#3 — Hull-breach edge check in `destroy_wall` (gamemap.py:217–218).**
  This currently uses `if fy < co or fy >= h - co` — a 2-tile border check
  using `co=3`. Reading the line: it's checking whether the destroyed wall
  is on the **map edge** (= hull boundary). The "coarse" here is a
  coincidence; it really wants a small constant (probably 1 or 2 tiles)
  for "near the edge". **Implementer: investigate this line. Probably
  should be `if fy < 1 or fy >= h - 1 or fx < 1 or fx >= w - 1`** — a
  pure edge check, independent of unit size.

- **#4 — `get_center_px` on Unit.** This computes a pixel-space center
  using `CFG.display.coarse_px`. The renderer doesn't seem to call it
  anymore (pyray uses world-tile coords + camera). **Implementer: grep
  for any caller. If unused, delete. If used, move the logic to the
  renderer and delete from Unit.**

- **#5 — `fine_tile_px` config key.** This was the legacy "pixels per
  fine tile" (= 6). Pyray renderer uses `world_px_per_tile` instead.
  **Implementer: grep. If unused, delete the config key. Likely safe.**

- **#6 — `debug_physics.py` survival.** This utility references the
  dead config keys and will break. **Implementer: check if it still
  works pre-patch (run it). If broken already, delete it. If working,
  fix it to use the level loader's grid dimensions and drop `CFG.display`
  references.**

## 9. Verification — what "done" looks like

- `pytest tests/` — 8 passing (no regression vs. baseline).
- `python tests/test_main_smoke.py --auto` — completes with the
  "rendered 600 frames; sim.tick=60" line.
- `python main.py` — game launches, 7 units visible at the §6 coords,
  unit can be selected, move order issued, Space resumes, unit moves.
- `git grep -i 'coarse\|fine_tile\|fxf\|fyf\| cx\b\| cy\b'` (or
  equivalent ripgrep) — zero hits in production code. Hits remaining
  must be (a) in `game.py`, (b) in `prototypes/archive/`, (c) in
  `docs/archive/`, (d) in `project_spectral_methods.md` memory file
  (multigrid context), or (e) in this patch-plan doc itself.
- One paragraph in the agent's final report: "Items #3, #4, #5, #6
  from §8 — resolved as follows."

## 10. Memory updates

When the patch lands:

- Update [[breach-architecture-2026-05]] memory to remove any mention
  of the dead vocab if present, and to add: "Coord system cleanup
  completed 2026-05-XX. One grid, `unit.x/y` float, `unit.footprint`."
- Verify no other memory file references coarse/fine in a stale way.

---

When you (Erik) read this fresh, the questions to revisit before dispatch:

1. Naming: are you happy with `unit.x` / `unit.y` (float, source of truth)
   and `tile_x` / `tile_y` properties for int access? Alternative was
   `tile_x` / `tile_y` as the int field with no float source-of-truth — but
   that requires the renderer to track interpolation state externally,
   which is the deferred architecture.md §16.10 work.
2. Spawn schema: `x` / `y` vs. `tile_x` / `tile_y` vs. `pos = [x, y]`? Plan
   uses the first.
3. `footprint` as a Unit field now, default 3 — or wait until the variant
   system to introduce it? Plan adds it now since it removes the
   `CFG.display.coarse` global cleanly. If you'd rather wait, the agent
   needs an alternative source for the 3 (hardcoded constant somewhere).
