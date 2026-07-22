# Arc C — C2 transaction-log undo: JIT design (design-gate)

> DESIGN ONLY — authored 2026-07-22 on `arc-c-editor`. This doc is the
> design half of C2's design-gate: it is adversarially critiqued *before*
> any code lands (kickoff C2; escalation trigger 3 = "C2 undo design fails
> its critique twice"). Spec pointers: editor doc §3 pillar 6 (undo = one
> transaction log of compound ops), kickoff C2, impl-doc C0 four-ring
> inventory + C1 LIGHT port. Buildable by a Sonnet subagent from this doc
> alone.

## 0. What we are replacing

Today `run_editor()` (locals, one function — impl-doc C0) carries **four
per-domain undo rings**, all fixed-capacity LIFO snapshot rings
(`UNDO_CAPACITY = 100`):

| ring | type | snapshot payload | pushed by |
|---|---|---|---|
| `undo` | `UndoRing` | whole `grid` (int32) copy | PAINT / ROOM / CORRIDOR / DOOR |
| `spawn_undo` | `SpawnRing` | whole `spawns` list (dataclass copies) | SPAWN add/move/delete/team |
| `light_undo` | `SpawnRing` | whole `lights` list (`EditableLight` copies) | LIGHT add/move/delete/edit |
| `water_undo` | `UndoRing` | whole `water_q` (int32) copy | WATER fill |

`Ctrl+Z` pops **only the current mode's ring** (SPAWN/LIGHT/WATER in their
modes, else the tile ring). Rings are deliberately isolated so undo in one
domain can't rewind another. There is **no redo**. Dirty state is four bools
(`dirty_tiles/spawns/lights/water`), each set on mutation and all cleared on
save; the unsaved dot is `any(dirty_*)`. Undo never clears the dot even when
it returns the state to what's on disk.

C2 replaces all four rings with **one global transaction log** of compound
operations, adds **redo**, and makes the unsaved dot **position-accurate**
(undoing back to the saved state clears it). This is the whole of C2; every
later patch (C3–C6) registers its new op classes onto the same log.

Live editable state a transaction can touch (all `run_editor` locals):

- **Grids** (numpy `int32`, mutated in place): `grid` (material codes),
  `water_q` (Q16.16 depth). C5 adds `zones` and `air` grids.
- **Collections** (lists of dataclasses): `spawns` (`SpawnEntry`), `lights`
  (`EditableLight`), `entities` (`EntityInstance`, currently held as
  `other_entities` + not yet mutated by any tool). C3 places doors/sensors
  into `entities`; C6 adds a `wires` collection (`WireSpec`).

---

## 1. Op & transaction model

### 1.1 Two op primitives cover every op class

An **operation (op)** is a reversible primitive with `undo(ctx)` and
`redo(ctx)` that write stored values into fixed locations of live state.
`ctx` is a small struct holding the live state handles (the grids dict and
the collections dict — see §7). There are exactly **two op types**; every
current and future op class maps onto one of them:

**`GridCellsOp(grid_name, ys, xs, before, after)`** — a *delta* on one named
grid. `ys`, `xs` are `int32` coordinate arrays; `before`, `after` are arrays
of the grid's dtype, one entry per changed cell. `redo` writes
`grid[ys, xs] = after`; `undo` writes `grid[ys, xs] = before`. Also exposes
`bounding_rect()` (min/max of ys,xs) so the editor can re-bake exactly the
touched region (reusing today's `expand_dirty_rect`/`rebake_rect`). `nbytes`
= sum of the four arrays' `.nbytes`.

**`CollectionOp(coll_name, before, after)`** — a bounded *snapshot pair* of
one named collection. `before`/`after` are independent dataclass-copy lists
(`[replace(e) for e in coll]`) captured at gesture start / commit. `redo`
does `coll[:] = after`; `undo` does `coll[:] = before` (slice-assign keeps
the caller's list identity — important because `run_editor` closes over the
list objects). `nbytes` ≈ `len × per-dataclass` (tiny).

> **Delta vs snapshot — the deliberate split.** Grids are `O(10⁴–10⁵)` cells;
> snapshotting a whole grid per action (today's `UndoRing`) is the memory
> problem the kickoff flags — so grids **delta-encode** (changed cells only).
> Collections are `O(10¹–10²)` small dataclasses; a whole-list copy is
> microseconds and a few KB, and it makes add/move/delete/edit/multi-select/
> re-id/paste all fall out of the *same* op with zero per-op-class code. So
> collections **snapshot** (matching the existing `SpawnRing`, which already
> proved generic over any dataclass — impl-doc C1). Per-element collection
> deltas would buy nothing here and add bug surface. This is the honest
> answer to "grid deltas AND entity deltas both need a representation": each
> gets the representation its size warrants.

### 1.2 Transaction = atomic compound op

```
Transaction:
    label: str            # e.g. "paint", "door", "move light" — drives the status flash
    ops:   list[Op]       # applied in order; INVERTED in reverse order
```

- `Transaction.redo(ctx)` runs `op.redo` for each op **in order**.
- `Transaction.undo(ctx)` runs `op.undo` for each op **in reverse order**.
- `Transaction.nbytes` = Σ op.nbytes.
- `Transaction.rebake_rects()` yields the `bounding_rect()` of each
  `GridCellsOp` so undo/redo re-bakes only what moved.

A transaction may hold any mix — a door is `[GridCellsOp("material", …),
CollectionOp("entities", …)]`. Undo/redo over the op list is all-or-nothing
(§6), which is what makes a compound op atomic.

### 1.3 Op classes that must join at C2 (exist now)

| op class | maps to | notes |
|---|---|---|
| tile paint (PAINT stroke, ROOM, CORRIDOR) | `GridCellsOp("material")` | one op per gesture |
| DOOR-material stamp (current DOOR mode) | `GridCellsOp("material")` | today stamps `MAT_DOOR`; C3 changes it to a compound door (below) |
| water paint (WATER fill) | `GridCellsOp("water")` | |
| spawn add / move / delete / team-toggle | `CollectionOp("spawns")` | |
| light add / move / delete / field-edit (B/C/R/E/P/X/H) | `CollectionOp("lights")` | replaces `light_undo` |
| entity inspector field edit (LIGHT today; general C3+) | `CollectionOp(<coll holding it>)` | one txn per nudge/cycle |

### 1.4 Op classes ARRIVING later — accommodated with no redesign

Each later patch registers ops through the **same builder seam** (§2.1); the
model does not change. The extension rule: *reuse `GridCellsOp` with a
registered grid name, or `CollectionOp` with a registered collection name;
add a new grid/collection to the ctx registry (§7) only when a genuinely new
array/list appears.*

| patch | new op class | joins as |
|---|---|---|
| C3 | **DOOR placement** = grid stamp of `MAT_DOOR_CLOSED` (id 7, A6) **+** door entity add, ONE user action | `Transaction["door", GridCellsOp("material"), CollectionOp("entities")]` — the compound-op archetype |
| C3 | sensor placement (body tile + `sample_tile`) | `CollectionOp("entities")` (no grid stamp) |
| C3 | generic entity place-one | `CollectionOp("entities")` |
| C4 | multi-select move / assign-tag / clump paste (re-id, internal wires kept) | `CollectionOp("entities")` (+ `CollectionOp("wires")` when paste adds internal wires) — whole-list snapshot handles re-id and N-at-once for free |
| C5 | zone paint / air paint / vacuum paint | `GridCellsOp("zones")` / `GridCellsOp("air")` — register those two grids in ctx (§7) |
| C5 | zone *instance* create/delete (paints + the `[[entity]]` binding) | `Transaction[GridCellsOp("zones"), CollectionOp("entities")]` |
| C6 | wire add / remove | `CollectionOp("wires")` — register the `wires` collection in ctx |

No later op needs a third op type. If a future patch ever finds neither
primitive fits, it adds a new `Op` subclass implementing `undo`/`redo`/
`nbytes` — but nothing in Arc C's scope does.

---

## 2. Capture points (where a transaction opens and commits)

### 2.1 The builder seam

Every mutating gesture routes through **one open-transaction builder** on the
log:

```
log.begin(label)          # opens the pending transaction (idempotent within a gesture)
log.snapshot_grid(name)   # captures a transient before-copy of a grid (once per gesture per grid)
log.snapshot_coll(name)   # captures a transient before-list of a collection (once per gesture)
   … the gesture mutates live state exactly as today …
log.commit()              # extracts deltas vs the now-mutated state, pushes ONE Transaction (or drops it if empty)
log.abort()               # drops the pending transaction WITHOUT reverting live state (see §2.4)
```

This is **immediate-mode capture**: the gesture mutates `grid`/`lights`/…
live (unchanged from today), and `commit()` computes the delta between the
transient *before* snapshot and the live *after* state. Rationale over a
command-mode rewrite: it maps one-to-one onto the existing push-points (every
`undo.push(snap)` / `spawn_undo.push(snap)` becomes a `snapshot_*` at gesture
start + a `commit` at gesture end), so the ~1.7k-line loop keeps its exact
mutation code. The transient before-copy (a full `grid.copy()` for grids,
already taken today as `stroke_pending`/`snap`) lives **only during the
gesture** and is discarded at commit — the *log* stores only the extracted
delta, so the log stays delta-encoded.

`commit()` builds each op:
- **grid op**: `mask = before != live` restricted to the gesture's tracked
  bounding rect (`stroke_dirty` for strokes; the op's own `rect` for
  ROOM/CORRIDOR/DOOR); `ys, xs = mask.nonzero()`; `before[mask]`, `live[mask]`.
  If `mask` is empty → contributes nothing.
- **collection op**: compare `before` list to live; if dataclass-equal, drop.
- If **all** ops are empty, `commit()` pushes **nothing** (a no-op gesture —
  paint X over X, room that changed nothing, drag that didn't move — never
  dirties the log; matches today's `if changed:` / `if moved:` guards).

### 2.2 One transaction per user action — the boundaries

| gesture | opens | commits |
|---|---|---|
| PAINT stroke | mouse-down (first change) | mouse-up (or gesture force-end §2.4). **One** transaction for the whole stroke, not per tile |
| PAINT shift-line | shift-click | same click (instantaneous) |
| ROOM / CORRIDOR | drag start | mouse-up (release) — one transaction |
| DOOR click (C3: door place) | click | same click; C3's compound grid+entity door is built between one begin/commit |
| WATER / zone / air fill | click | same click (a bucket fill is one action even though it touches many cells) |
| spawn/light add, delete, team/kind/field nudge | key/click | same event (instantaneous) |
| spawn/light **drag-move** | mouse-down (capture before) | mouse-up **if position changed**, else dropped — matches today's pre-drag-snapshot-only-if-moved |
| C4 multi-select move / tag / paste | gesture start | gesture end |

### 2.3 Coalescing rules

- **Within a stroke** the same cell may be repainted many times across
  frames. Coalescing is automatic from immediate-mode capture: the stored
  `before` is the value at **gesture start** (the transient snapshot), the
  stored `after` is the value at **commit**; intermediate values never enter
  the op. So a stroke that paints A→B→C stores before=A, after=C in one op.
- **Across strokes / clicks**: each is its own transaction (no cross-gesture
  merge). Rapid repeated inspector nudges of the *same field* stay **one
  transaction per keypress** — an ACCEPTED GAP (matches today's one-push-per-
  keypress; a time-window merge is deferred, not built). Documented so the
  critique can accept it rather than the build guessing.

### 2.4 Gesture interruption (mode switch / Esc mid-gesture)

`cancel_transients()` runs on every mode switch. Rule: **if a transaction is
open with ≥1 real change, `commit()` it; otherwise `abort()`.** So "paint,
then TAB to LIGHT" leaves the paint as a committed, undoable transaction
(matches today, where the tile snapshot was already pushed mid-stroke). A
drag-move interrupted by a mode switch also commits its partial move (this is
a small, deliberate improvement over today's latent gap where an interrupted
drag pushed no undo entry — flagged in §9). `abort()` never reverts live
state; it only drops an empty pending transaction.

---

## 3. Delta representation & memory bounds

- **Grid ops**: `ys/xs` (`int32`) + `before/after` (grid dtype, `int32` for
  both material and water). ≈ 16 bytes per changed cell. A typical brush
  stroke touches tens–hundreds of cells → hundreds–low-KB. The pathological
  case — a full-canvas repaint or a whole-map bucket fill on a 256×256 grid —
  is 65 536 cells ≈ 1 MB (same order as one old snapshot, but *only* for that
  one action; every small action is now near-free instead of a full 256 KB
  snapshot). Delta-encoding wins decisively on the common path and never
  loses by more than a constant on the worst path.
- **Collection ops**: `before`+`after` dataclass-copy lists; a level's tens
  of entities → single-digit KB per op.

**Log bounds (two ceilings, evict oldest until both hold):**

- **Depth bound: `LOG_DEPTH = 256` transactions.**
- **Memory ceiling: `LOG_BYTES = 128 MB`** (Σ transaction.nbytes over the
  whole log).

When a newly committed transaction pushes the log past either bound, evict
from the **oldest (furthest-back undo) end** until both hold — but **never
evict the transaction the cursor sits just after** (always keep ≥1 undoable
action). Redo-tail entries are truncated by new commits (§4), not by these
bounds. Eviction that drops a transaction at or below `saved_cursor` makes
the saved state unreachable by undo → `saved_cursor = None` (§5). These are
generous for a single-operator dev tool; the accepted gap is that 256
full-canvas repaints (or ~128 MB of deltas) drop the oldest history.

---

## 4. Redo — linear undo/redo cursor

The log is a linear stack with a cursor; `log[:cursor]` are applied,
`log[cursor:]` are redoable.

```
TransactionLog:
    txns:         list[Transaction]
    cursor:       int              # count applied; log[cursor:] is the redo tail
    saved_cursor: int | None       # cursor value equal to on-disk state (§5)
```

- **undo()**: if `cursor > 0`: `txns[cursor-1].undo(ctx)`; `cursor -= 1`.
  Else flash "nothing to undo".
- **redo()**: if `cursor < len(txns)`: `txns[cursor].redo(ctx)`;
  `cursor += 1`. Else flash "nothing to redo".
- **commit of a NEW transaction**: **truncate the redo tail** —
  `del txns[cursor:]` — then append and `cursor += 1`. A new action after
  undo discards the redone-away future (standard linear semantics; no undo
  tree). If `saved_cursor is not None and saved_cursor > cursor` before the
  append (the saved state lived in the truncated tail), set
  `saved_cursor = None`.

Keys: `Ctrl+Z` = undo (global), `Ctrl+Y` **and** `Ctrl+Shift+Z` = redo. The
"release the mouse before undo" guard stays: undo/redo are refused while a
gesture is open (commit or force-end first). The inspector HUD "undo depth N"
becomes "undo N / redo M" from `cursor` and `len(txns) - cursor`.

---

## 5. Interaction with level_lib dirty-tracking & unsaved state

**Dirty is a cursor comparison, not four bools:**

```
dirty = (log.saved_cursor is None) or (log.cursor != log.saved_cursor)
```

- **open**: `txns=[]`, `cursor=0`, `saved_cursor=0` → clean.
- **any commit / redo / undo**: no special handling — `dirty` recomputes.
  Undoing back until `cursor == saved_cursor` clears the dot; redoing or
  editing away from it re-dirties. This is the classic "saved marker at a log
  position" pattern and it *replaces* today's four `dirty_*` bools (which
  could never clear on undo).
- **save (Ctrl+S)**: after the existing `level_lib` writeback + bake +
  `handle.record_disk_state()`, set `saved_cursor = cursor`. One marker
  suffices: a save flushes *all* domains (csv + water npy + toml families +
  bake), so a single global "state == disk" position is exactly right.
- **saved_cursor = None** (saved state fell out of the log via eviction §3):
  `dirty` stays True until the next save — correct (the log can no longer
  prove the state matches disk).

**level_lib `check_stale` (reload-or-overwrite) — unchanged contract.** The
transaction log is **purely client-side**; it does not read or write
`LevelHandle.toml_mtime_ns/toml_sha256` and does not change any level_lib API
meaning (respects escalation trigger 1 — client-side additions only). On
Ctrl+S the editor still calls `handle.check_stale()` and prompts
reload-or-overwrite exactly as designed. Two outcomes:

- **overwrite**: proceed with save; then `saved_cursor = cursor` as above.
- **reload** (discard in-memory state for disk): the in-memory grids and
  collections are rebuilt from disk, so the log's stored deltas no longer
  describe reachable states → **reset the log** (`txns=[]`, `cursor=0`,
  `saved_cursor=0`). This reset is the one new client-side line the reload
  path needs; it touches no level_lib internals.

The `.bak` once-per-session guards (`csv_bak_written`, `toml_bak_written`,
`water_bak_written`) are independent of undo and are untouched by C2.

---

## 6. Atomicity & failure

**A compound transaction fully applies or fully does not — no half-state.**
The discipline that guarantees it:

1. **Refused actions never open a committed transaction.** A gesture that the
   editor rejects (`door_check` fails, a solid sensor `sample_tile`, a fill
   with no region, a drag that didn't move) either never calls `begin()` or
   calls `abort()`. Only actions that actually changed live state commit.
2. **Ops are total assignments, not computations.** `GridCellsOp.undo/redo`
   is `grid[ys, xs] = before/after`; `CollectionOp.undo/redo` is
   `coll[:] = before/after`. Both are pure numpy fancy-assignment / list
   slice-assignment into locations whose validity is fixed for the session
   (grid shape never changes — no tool resizes the map; coordinates were
   captured from that same grid; collection slice-assign always succeeds).
   **An inverse cannot fail by construction** — there is no code path in
   `undo`/`redo` that can raise on well-formed session state, so there is no
   "inverse couldn't be applied" case to recover from. This is why we do not
   build rollback machinery: the only way to get a partial apply would be an
   op raising mid-list, and ops cannot raise.
3. **Commit is the only mutation point of the log.** `commit()` builds all
   ops from already-applied live state and appends one Transaction; it never
   applies anything (the gesture already did). So there is no window where
   half a transaction is in the log.

Accepted gap (dev tool, single trusted operator): no defence against an
out-of-band mutation of `grid`/`lights` between capture and undo (nothing in
the editor does this). If it ever happened, undo would write stale `before`
values — impossible in the current single-threaded loop, documented rather
than guarded.

---

## 7. Migration from the four rings

**Removed:** `undo`, `spawn_undo`, `light_undo`, `water_undo`, the `SpawnRing`
class (and `UndoRing` if unused elsewhere — it is only used by the editor and
`tests/test_map_editor_tool.py::test_undo_ring_restores_room_and_corridor_ops`,
which is replaced by the new log tests §8), and the four `dirty_*` bools.

**Added:** a `TransactionLog` plus a `ctx` registry mapping names to live
state handles, constructed once in `run_editor` after state is loaded:

```
ctx.grids       = {"material": grid, "water": water_q}      # + "zones","air" in C5
ctx.collections = {"spawns": spawns, "lights": lights, "entities": entities}  # + "wires" in C6
```

(The grids dict holds the array objects; `undo` writes into them in place via
fancy indexing, so the closed-over `grid`/`water_q` names stay valid.
Collections use slice-assign for the same reason.)

**Push-point rewrite** — mechanical, one gesture at a time: each
`undo.push(snap)` / `spawn_undo.push(...)` / `light_undo.push(...)` /
`water_undo.push(...)` site becomes a `snapshot_*` at the gesture's start and
a `commit` at its end (per §2.2). The mutation code between them is unchanged.

**Ctrl+Z / Ctrl+S block rewrite**: the mode-scoped `if mode == "SPAWN": …
elif "LIGHT": …` undo dispatch collapses to a single `log.undo()`;
`Ctrl+Y`/`Ctrl+Shift+Z` → `log.redo()`; on undo/redo, re-bake the union of the
transaction's `rebake_rects()` (grid ops) and refresh spawn/light overlays
(collection ops) — the editor already re-derives overlays from the live lists
each frame, so only grid re-bakes need explicit rects.

### 7.1 BEHAVIOR CHANGE — global undo history (call it out)

Today `Ctrl+Z` rewinds **only the current mode's** ring (cross-domain
isolation was deliberate). The transaction log is **one global history across
all domains**: `Ctrl+Z` undoes the *most recent action regardless of the mode
you are now in*. Example — paint a wall, TAB to LIGHT, `Ctrl+Z` now undoes
the **paint** (today it would say "nothing to undo (lights)").

This is **intended** and matches the locked design: editor doc §3 pillar 6 =
"Undo is a **single** transaction log … replacing the per-domain rings," and
kickoff C2 = "a single log of compound operations." The deliberate isolation
of the four-ring era is exactly what pillar 6 retires. This is the one
user-visible behavior change in C2 and the critique should bless it
explicitly. (It is also *more* correct for compound ops: a door's grid stamp
and its entity must undo together, which per-domain rings could never do.)

---

## 8. Test plan (the oracle — pure, headless)

All tests are headless (no raylib), on `TransactionLog` + the two op types
against plain numpy grids and dataclass lists — the same style as
`tests/test_editor_layout.py` / `test_level_editor_tool.py`. Proposed file:
`tests/test_editor_undo.py`.

1. **Op round-trip, each class**: `GridCellsOp` on a material grid and on a
   water grid — after redo the grid equals the mutated state, after undo it
   equals the exact prior state (`np.array_equal`). `CollectionOp` on spawns,
   lights, and entities — undo/redo restore dataclass-equal lists, and the
   list *object identity* is preserved (slice-assign, not rebind).
2. **Compound-op atomicity**: a `Transaction[GridCellsOp("material"),
   CollectionOp("entities")]` (the C3 door archetype); undo restores **both**
   grid and entities to the prior state; redo re-applies both; assert there
   is no observable state where one op is applied and the other is not (undo
   reverses ops in reverse order; verify grid+collection are consistent after
   each of undo and redo).
3. **Stroke coalescing**: feed a stroke that paints cell (x,y) A→B→C via the
   builder; the committed op stores before=A, after=C (one transaction, one
   op). A no-op stroke (paint X over existing X) commits **nothing** — log
   length unchanged.
4. **Redo truncation**: commit 3, undo 2 (cursor=1), commit 1 → `len(txns)==2`,
   redo() is a no-op, the redone-away transactions are gone.
5. **Saved-marker / dirty coherence**: open → `dirty` False; commit → True;
   undo to `cursor==saved_cursor` → False; redo → True; save (set
   `saved_cursor=cursor`) → False; commit → True. Then force eviction below
   `saved_cursor` (§3) → `saved_cursor is None` and `dirty` stays True at
   `cursor==0`.
6. **Global-history ordering**: interleave a `material` grid op and a
   `lights` collection op ("different modes"); a single `undo()` reverses the
   **last** committed action irrespective of any mode notion; a second undo
   reverses the earlier one.
7. **Memory-bound enforcement**: push `> LOG_DEPTH` small transactions →
   oldest evicted, `len(txns) == LOG_DEPTH`, cursor/`saved_cursor` adjusted
   consistently. Push large grid-delta transactions until `Σ nbytes` would
   exceed `LOG_BYTES` → eviction keeps the log under the ceiling and never
   evicts the last undoable transaction.
8. **Fuzz round-trip**: from a known start state (a random grid + random
   spawn/light/entity lists), apply a random sequence of ops across all grids
   and collections through the builder; `undo()` all the way → state is
   deep-equal to the start (`np.array_equal` on every grid + dataclass-equal
   on every collection); then `redo()` all the way → equal to the
   post-sequence state; and a random walk of interleaved undo/redo stays
   equal to an independent reference model that just replays the same cursor
   moves.

Gate (per arc): these unit tests + full `pytest tests -q` green + level_lib
byte-stable round-trip tests green + **zero goldens/digests touched** (the
editor is not sim path).

---

## 9. Self-identified weak points (feed the adversarial pass)

1. **Immediate-mode capture correctness on compound + bounded grid diffs.**
   `commit()` extracts the grid delta by diffing the transient before-copy
   against live *within a tracked bounding rect* (`stroke_dirty`, or each
   tool's own `rect`). If any tool ever mutates a cell outside the rect it
   reports, that cell is silently dropped from the delta and undo leaves
   residue. ROOM/CORRIDOR already compute their own rects (`diff_rect`), so
   the risk is real but localized; a reviewer may argue `commit()` should
   diff the *whole* grid for safety (O(grid) per commit — cheap at 256²) and
   drop the per-gesture rect tracking entirely. Also up for challenge: the
   immediate-mode (mutate-then-capture) choice itself vs a command-mode
   rewrite, and whether "commit-on-cancel" (§2.4) is the right call for an
   interrupted drag.

2. **Variable-size transactions under the memory bound.** Delta-encoding
   assumes edits are small, but zone/air/water **bucket fills** (C5) and
   whole-room paints can each produce large deltas — the very workflows where
   deltas approach snapshot size. Are `LOG_DEPTH=256` / `LOG_BYTES=128 MB`
   the right knobs, and is evicting-below-`saved_cursor` (which silently
   pins the dot to "dirty forever") acceptable, or should the depth be lower
   and the byte ceiling the real governor? The pathological full-repaint
   case (~1 MB/txn) is called out but the critique should sanity-check the
   numbers against the actual max grid size in play.

3. **Whole-collection snapshots + the collapse to one dirty marker.**
   `CollectionOp` copies the *entire* collection per op. For C4 (select-all
   move, big clump paste) and any level that grows to thousands of entities,
   repeated per-keypress whole-list copies could add up (memory and the
   inspector-nudge-spam path §2.3's accepted gap). And collapsing four
   independent `dirty_*` flags into one global cursor assumes no workflow
   depended on per-domain dirty state (e.g. "water saved independently") —
   believed true (save flushes everything atomically) but worth a challenge.
   Finally, the reload-or-overwrite path *must* reset the log (§5); a reviewer
   should confirm that reset is actually wired at build time, since a missed
   reset would leave undo pointing at a replaced state.
