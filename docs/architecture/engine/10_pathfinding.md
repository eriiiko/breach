# Pathfinding

**Depends on:** [Grid & Coordinates](01_grid_and_coordinates.md), [Material System](03_material_system.md)

Pathfinding answers one question: given a unit at one tile and a destination at
another, what sequence of tiles takes it there without walking through walls?
Breach uses a single algorithm — grid A\* — for both sides of the board. Player
marines use it once, at order-placement time, to turn each movement waypoint into
a concrete tile path that the renderer previews and the executor replays. Zombies
use it live, during execution, to chase the nearest marine and re-plan as that
target moves.

The whole system is one module, `pathfinding.py`, with no dependency on the rest
of the simulation: it talks to the world only through a caller-supplied blocking
predicate. That isolation is deliberate — the same search runs headless inside the
`Simulation` facade, inside zombie AI, and (eventually) inside test harnesses,
without dragging in `GameMap`, units, or config.

---

## 1. The search

A\* on the tile grid, eight-connected. The signature is deliberately small:

```python
astar(start_x, start_y, goal_x, goal_y, is_blocked_fn, map_w, map_h)
    -> list[(x, y)]            # start..goal inclusive, or [] if no path
```

Coordinates are tile coordinates on the one physics grid — the same `(x, y)`
addressing every other system uses. There is no separate "fine" or "coarse"
pathfinding grid; the field names that survive in the module docstring
(`FINE_W = 120`, `FINE_H = 75`) are a holdover from the abolished dual-grid era and
serve only as default map dimensions. Callers pass the real grid shape explicitly.

The return value is a flat list of `(x, y)` tile tuples from start to goal
inclusive. An empty list means *no path* — and is also returned for the
degenerate cases (start or goal out of bounds, start or goal blocked, node budget
exhausted). A single-element list means start already equals goal. Callers must
treat `[]` as "stay put", never as a bug.

### Footprint-aware blocking

A unit is not a point — a marine is a 3×3 block of tiles. A\* never reasons about
the footprint itself; it delegates the entire question to `is_blocked_fn(x, y)`,
a boolean predicate the caller supplies. Both real callers build that predicate
from `GameMap.is_passable_block`:

```python
def is_blocked(x, y):
    return not gmap.is_passable_block(y, x, unit.footprint)
```

`is_passable_block(fy, fx, footprint)` is true only when the *entire*
`footprint × footprint` square anchored at `(fx, fy)` is air or open door — a
single wall tile inside the block makes the whole position impassable. Because the
predicate carries the footprint, **A\* itself is footprint-agnostic**: the same
search plans for a 3-tile marine, and would plan for a hypothetical 2- or 4-tile
unit, with no change to the algorithm. The footprint contract — `occupied_tiles()`
/ `occupies()` on the unit, fed through `is_passable_block` — is the single seam
where unit size enters pathfinding. (Variable footprints exist in the unit model
but only the 3×3 case is exercised in practice; see *Implementation status*.)

The internal bounds check `_in_bounds(x, y, w, h)` enforces that the *anchor*
leaves room for a 3-tile block (`0 <= x <= w - 3`). This is hardcoded to
`UNIT_SIZE = 3` and is the one place the footprint-agnostic story leaks: a
non-3 footprint is handled correctly by `is_blocked_fn` but bounds-clipped as if
it were 3. For current content this is invisible.

### Movement cost and the alternating diagonal

Cardinal steps cost 1. Diagonal steps use the **D&D 3.5 alternating rule**: the
first diagonal a unit takes costs 1, the next costs 2, the next 1, and so on. The
average diagonal therefore costs 1.5, which is a cheap, integer-friendly
approximation of √2 that avoids the "diagonals are free" artefact of uniform cost
without paying for floating-point Euclidean steps everywhere.

Implementing alternating cost correctly requires the parity to be **part of the
search state**, not a global counter. Each node is `(x, y, diag_flag)`, where
`diag_flag` says whether the *next* diagonal will be cheap. A diagonal step
flips the flag; a cardinal step leaves it untouched. Two arrivals at the same tile
with different parity are genuinely different states with different future costs,
so they are stored and expanded separately. The goal test ignores parity — either
parity at the goal tile ends the search.

The heuristic matches the cost model exactly so it stays admissible:

```
h = straight + 1.5 * diag        # octile distance with 1.5 diagonals
  where  diag     = min(|dx|, |dy|)
         straight = max(|dx|, |dy|) - diag
```

Because the real diagonal cost alternates between 1 and 2 and the heuristic
assumes the cheaper-on-average 1.5, `h` can mildly overestimate a path that is
forced onto many consecutive expensive diagonals. In practice this is harmless —
it only ever makes the search marginally greedier — and the heuristic never blocks
a valid path.

### Determinism and the node budget

The open set is a binary heap of
`(f, tie_breaker, g, x, y, diag_flag)`. `f = g + h`. The `tie_breaker` is a
monotonic insertion counter: when two nodes share `f`, the one discovered first
wins. This makes the search **fully deterministic** — identical inputs always
expand nodes in the same order and return the same path, which the deterministic
`Simulation` facade relies on for reproducible rounds and replay.

Expansion is capped at `NODE_LIMIT = 50_000` nodes. On overflow the search
abandons and returns `[]`. The cap is a safety valve against pathological
geometry (a goal walled off from the start forces A\* to flood the entire
reachable region before failing); on the current map size it is never approached
by a solvable query.

---

## 2. How player units use it

Player movement is **planned once, replayed deterministically**. When a movement
order is placed, `Simulation.apply_action` validates the target with
`is_passable_block` and then calls `_compute_player_paths`, which runs A\* and
lays down the unit's full tick-by-tick trajectory for the round. The same method
runs on undo. Nothing about a player path is recomputed during execution.

This is a deliberate choice over per-tick replanning. Because the path is
materialised at order time:

- the planning **overlay and the executor read the same data** — what the player
  previews is exactly what runs, with no drift between a "preview path" and an
  "execution path";
- execution is trivially cheap — stepping a unit is an array index, not a search;
- the round is deterministic and replayable from the order list alone.

`_compute_player_paths` does the lowering. For each living marine it walks both
phases of the round. Within a phase it gathers that phase's movement orders, runs
`astar` between consecutive waypoints, and concatenates the tile segments
(dropping each segment's duplicated first tile). It then expands the tile path
into **per-tick float positions**: each tile transition is sub-sampled into
`ticks_per_tile` linearly-interpolated steps, where the cadence comes from the
order type —

| Order | Cadence source |
|---|---|
| Move & Attack | `CFG.movement.marine_attack_ticks_per_tile` |
| Move with Cover | `CFG.movement.marine_cover_ticks_per_tile` |
| Sprint | `CFG.movement.marine_sprint_ticks_per_tile` |

The result fills `unit.move_path` (one `(x, y)` per tick) and pads with the final
resting position so the path spans the whole phase. If pathfinding is unavailable
(import guard `HAS_PATHFINDING`), the fallback degrades to a straight teleport to
the waypoint — correct enough to keep the game playable, but wall-ignoring.

At execution time `_update_player_movement` simply indexes into `move_path`:

```python
path_idx = self.tick - u.path_tick_offset
if 0 <= path_idx < len(u.move_path):
    u.x, u.y = u.move_path[path_idx]
    u.face_towards(...)
```

There is no per-tick blocking check here. A path is validated only at the moment
it is computed; if a wall is destroyed mid-execution along a marine's planned
route, the marine walks through the now-open (or still-closed) tiles regardless.
This is a known gap, recorded in *Implementation status*.

---

## 3. How zombies use it

Zombies path **live**. Each activated zombie picks the nearest living marine and,
on its movement cadence (`speed_ticks_per_tile`), advances one tile along an A\*
path toward that target. The path is recomputed when it is missing, exhausted, or
**every 5 steps** — the periodic repath is what lets a zombie track a marine who
is himself moving. The blocking predicate is the same footprint-aware
`is_passable_block` wrapper, rebuilt per query with the zombie's own footprint.

Zombie movement re-validates each step before committing: even with a fresh path,
`is_passable_block` is checked again before the zombie occupies the next tile (a
wall may have changed since the path was built). On a stale step the zombie drops
its path and re-plans next cadence tick. So unlike player movement, zombie
movement *does* honour walls per-step — a consequence of the live, re-validating
loop rather than a precomputed replay.

---

## 4. Temporal A\* — present but dormant

The module also ships `temporal_astar` and a `ReservationTable`. These implement
**cooperative pathfinding**: A\* over `(x, y, tick)` space with an explicit *wait*
action, searching against a reservation table that records which tiles each unit
occupies at each tick. The intent is collision-free multi-unit movement — marines
that route around each other instead of overlapping.

`ReservationTable` is a sparse `{(x, y, tick): unit_id}` map. Reserving a path
stamps every tile of a unit's footprint for every tick of each segment;
`is_reserved` checks a footprint block at a tick (with self-exclusion); `clear`
removes one unit's stamps or wipes the table. `temporal_astar` consults it for
both the destination tile at the arrival tick and the origin tile across the
transit ticks, so a unit cannot move into a tile another unit is passing through.

**None of this is wired in.** No call site constructs a `ReservationTable` or
calls `temporal_astar`; player paths are built by plain `astar` and marines can
freely overlap during execution. The temporal machinery is finished, tested-in-
isolation scaffolding kept against the day multi-unit coordination is wanted. It
is intentionally left dormant — enabling it is a real feature decision (it changes
how marines move around each other), not a loose end to tidy up, and it should not
be switched on opportunistically.

---

## 5. Forward design

Two directions are noted but not built:

- **Articulated / non-square bodies.** The footprint contract already routes unit
  shape through `occupied_tiles()` into `is_blocked_fn`, so a non-square or
  multi-segment body is, in principle, just a different predicate. The
  `_in_bounds` 3-anchor assumption and the absence of facing-rotation in
  `occupied_tiles()` are the two things that must change first.
- **Batch A\* on GPU.** At Civulator-scale unit counts, running one A\* per unit on
  the CPU does not hold. A batched, data-parallel A\* (all units searched together
  on the GPU) is the planned answer; it is out of scope for Breach's squad-sized
  rounds and lives in the CUDA integration plan, not here.

---

## Implementation status

**Built and in use:**

- `astar(...)` in `pathfinding.py` — eight-connected grid A\*, alternating
  diagonal cost as a search-state parity flag, octile heuristic, deterministic
  tie-breaking, 50k node cap. Returns a flat `list[(x, y)]`.
- Footprint-aware blocking via caller-supplied `is_blocked_fn` wrapping
  `GameMap.is_passable_block(fy, fx, footprint)`.
- Player path lowering: `Simulation._compute_player_paths` runs A\* at order
  time (and on undo), interpolates to per-tick float positions at the order's
  `ticks_per_tile` cadence, fills `unit.move_path`. Replayed by
  `_update_player_movement` as an array index — the precompute-once decision.
- Zombie live pathing in `ai_zombie.update_zombies_tick`: nearest-target A\*,
  repath on miss/exhaust/every-5-steps, per-step `is_passable_block`
  re-validation before each move.
- `HAS_PATHFINDING` import guard with a straight-line teleport fallback.

**Built but dormant (deliberately unused):**

- `temporal_astar(...)` and `ReservationTable` — complete cooperative-pathfinding
  implementation in `(x, y, tick)` space with wait actions and footprint
  reservations. No call site. Player units overlap freely during execution.
  Enabling it is a scoped feature decision, not cleanup.

**Designed only (not built):**

- Non-square / articulated footprints and facing-rotated `occupied_tiles()`.
  The predicate seam is ready; bounds-handling and rotation are not.
- GPU batch A\* — design lives in the CUDA integration plan.

**Known gaps / rough edges:**

- **No per-tick wall collision for player units.** `_update_player_movement`
  blindly replays `move_path`. A path is validated only when computed; a wall
  destroyed mid-round along a planned route is not re-checked, so marines can walk
  through walls during execution. The fix is a per-tick `is_passable_block` check
  in the movement loop (or, more completely, replanning on terrain change).
- **`_in_bounds` hardcodes a 3-tile anchor margin** (`UNIT_SIZE = 3`). Non-3
  footprints are blocked correctly by the predicate but clipped at the map edge as
  if they were 3×3. Untested for footprint values other than 3.
- **`FINE_W` / `FINE_H` defaults are dual-grid relics.** They are only fallback
  map dimensions (real callers pass the actual grid shape) and the names should be
  retired to match the one-grid model in *Grid & Coordinates*.
- **Constants not config-driven.** `NODE_LIMIT`, `UNIT_SIZE`, and the default map
  size are literals in `pathfinding.py` rather than reading from `CFG`.
