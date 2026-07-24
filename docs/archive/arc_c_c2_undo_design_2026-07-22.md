# Arc C — C2 transaction-log undo: JIT design (design-gate) — v2

> DESIGN ONLY — authored 2026-07-22 on `arc-c-editor`. **v2** folds in the
> first-pass adversarial critique (3 lenses: data-loss, atomicity,
> scope/regression — all NEEDS-REVISION, core two-primitive + linear-cursor
> model BLESSED). Every blocker (B1–B5) and concern (C1–C9) is resolved
> normatively below; see the **Revision log (v2)** at §10. This is the design
> half of C2's design-gate; escalation trigger 3 = "C2 undo design fails its
> critique twice" (this is pass 1 → revise, not escalation). Spec pointers:
> editor doc §3 pillar 6 (undo = one transaction log of compound ops), kickoff
> C2, impl-doc C0 four-ring inventory + C1 LIGHT port. Buildable by a Sonnet
> subagent from this doc alone.

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
one named collection. `before`/`after` are **deep** copies of the collection
list, captured at gesture start / commit. `redo` does `coll[:] = after`;
`undo` does `coll[:] = before` (slice-assign keeps the caller's list identity
— important because `run_editor` closes over the list objects). `nbytes` ≈
`len × per-dataclass` (tiny).

> **The copy MUST be deep (B1).** `[replace(e) for e in coll]` copies each
> dataclass but leaves `EntityInstance.fields` (a dict) and `.tags` (a tuple)
> **aliased by reference**. That is silently fatal: an inspector field-edit
> mutates `fields` in place, so the "before" snapshot's dict mutates with it —
> `commit()` then judges before == after (no-op, dropped), and any undo
> restores an instance sharing the live dict. **Normative:** `CollectionOp`
> captures `copy.deepcopy(coll)` for both `before` and `after`. Collections
> are tens of small dataclasses; `deepcopy` cost is irrelevant, and it is
> total over every present and future mutable member (no per-field allow-list
> to keep in sync). The §8 IDENTITY test pins that a restored element's
> `fields`/`tags` are independent objects from the retained snapshot.

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
log.abort()               # REVERTS live state from the retained before-snapshots, then drops the pending txn (§2.4, C4)
```

**`abort()` reverts (C4).** Immediate-mode mutates live state during the
gesture, so a bare "drop without reverting" would leave a half-applied,
unlogged change (corruption). `abort()` therefore restores every grid/
collection it snapshotted from the retained transient before-copy
(`grid[...] = before`, `coll[:] = before`) and then discards the pending
transaction. This is cheap — the before-copies are already held for the diff
— and makes an aborted gesture a true no-op regardless of how far it had
mutated. (Complementary invariant, still required: a gesture validates
**before** its first live mutation — see §6 — so refused actions never open a
transaction at all; `abort()` is the safety net for a gesture force-ended
mid-mutation, §2.4.)

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
- **grid op (whole-grid diff — B2)**: `mask = (before != live)` over the
  **entire grid**; `ys, xs = mask.nonzero()`; `before[mask]`, `live[mask]`.
  If `mask` is all-False → contributes nothing. The diff is **never** bounded
  by the gesture's rect: at 256² a full `!=` is ~65 k comparisons
  (microseconds), and a rect bound silently drops cells for CORRIDOR (its
  walls land a tile beyond the drag-line bbox — today's code already diffs the
  *whole* grid via `diff_rect`) and for every flood/wand/zone/air/water fill
  (C5) whose changed region is not the cursor's bbox. Per-gesture rects
  (`stroke_dirty` etc.) survive **only** as a re-bake hint for the live
  preview (a render concern), never as the delta bound.
- **collection op**: compare the `before` snapshot to live; if deep-equal,
  drop.
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
drag pushed no undo entry — flagged in §9). `abort()` is used only for a
gesture with **no** real change (nothing to commit); it reverts any
snapshotted-but-unchanged state (a cheap no-op) and drops the empty pending
transaction. Esc mid-gesture behaves the same: commit if changed, else abort.

---

## 3. Delta representation & memory bounds

- **Grid ops**: `ys/xs` (`int32`) + `before/after` (grid dtype, `int32` for
  both material and water) — **~16 bytes per changed cell**, i.e. *4× a raw
  cell*. A typical brush stroke touches tens–hundreds of cells → hundreds of
  bytes to low-KB. The pathological case — a full-canvas repaint or a
  whole-map bucket fill — stores every cell as a delta, so at 256² it is
  65 536 × 16 ≈ **1 MB, ~4× the 256 KB of one old whole-grid snapshot**. So
  delta-encoding is a decisive win on the common path (near-free vs a fixed
  256 KB) and a bounded ~4× loss only when an action truly rewrites the whole
  map. **Grid size is not capped at 256²** — `new --size` has only a floor —
  so on a large map a single fill can be many MB; the **byte ceiling, not the
  depth count, is the real governor** and is what protects big-map sessions.
- **Collection ops**: `before`+`after` deep-copy lists; a level's tens of
  entities → single-digit KB per op.

**Log bounds — injectable (C7):**
`TransactionLog(depth=LOG_DEPTH, max_bytes=LOG_BYTES)` with
`LOG_DEPTH = 256`, `LOG_BYTES = 128 MB` as the production defaults but both
**constructor args** so the §8 eviction test can pass a tiny ceiling instead
of allocating 128 MB. Bound = "evict oldest until BOTH hold."

**Eviction renumbers the cursor (C3 — normative).** When a newly committed
transaction pushes the log past either bound, evict `K` transactions from the
**front** (oldest / furthest-back undo end), then adjust every index that
counts from the front:

```
del txns[:K]
cursor = max(0, cursor - K)
saved_cursor = (None if (saved_cursor is not None and saved_cursor <= K)
                else saved_cursor - K)
```

Never evict the transaction the cursor sits just after — always keep ≥1
undoable action (so a single 200 MB fill is retained even over the ceiling;
the accepted gap). Redo-tail entries are truncated by new commits (§4), not
by these bounds. `saved_cursor <= K` (saved state fell inside the evicted
prefix) means the saved state is no longer reachable by undo → `None` (§5).
Failing to renumber `cursor` after a front-eviction would apply the wrong
inverse — this adjustment is not optional.

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

Keys (C9 — explicit guards, because today's `ctrl and Z` has **no** shift
check, so `Ctrl+Shift+Z` currently also fires undo):

```
undo = ctrl and (not shift) and is_key_pressed(Z)
redo = ctrl and (is_key_pressed(Y) or (shift and is_key_pressed(Z)))
```

`Ctrl+Y` is otherwise unbound today, so it is free for redo. The "release the
mouse before undo" guard stays: undo/redo are refused while a gesture is open
(commit or force-end first). The inspector HUD "undo depth N" becomes "undo N
/ redo M" from `cursor` and `len(txns) - cursor`.

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
- **save (Ctrl+S)**: the save-time water mask is a real state mutation and
  must go **through the log** (C1, below); then, as the **last** step, set
  `saved_cursor = cursor`. One marker suffices: a save flushes *all* domains
  (csv + water npy + toml families + bake), so a single global "state == disk"
  position is exactly right.
- **saved_cursor = None** (saved state fell out of the log via eviction §3):
  `dirty` stays True until the next save — correct (the log can no longer
  prove the state matches disk).

**Save-time water mask must be logged (C1).** Ctrl+S today does
`water_q[...] = mask_water_to_open(water_q, grid, water_solid)` — an
out-of-band mutation that would violate the §6 "nothing mutates live state
outside a transaction" invariant and let the dot lie (fill water → wall over
it → save → the mask zeroes the walled cells, but the log doesn't know, so a
later undo/redo desyncs live vs disk while the dot reads clean). **Normative
fix:** wrap that mask exactly like any gesture — `log.begin("save-mask")`,
`snapshot_grid("water")`, apply the mask to `water_q`, `log.commit()`. If the
mask changed nothing, `commit()` pushes nothing (the common case, dot
unaffected). If it did change cells, they land as one committed
`GridCellsOp("water")` **before** `saved_cursor = cursor` — so the saved
position captures the post-mask state and undo/redo stay coherent. (This runs
before the `level_lib` writeback, so the file and the log agree.)

**level_lib `check_stale` — NOT wired by C2.** The transaction log is
**purely client-side**: it never reads or writes
`LevelHandle.toml_mtime_ns/toml_sha256` and changes no level_lib API meaning
(respects escalation trigger 1 — client-side additions only). Note that this
editor's Ctrl+S **does not currently call `check_stale`** — it overwrites
unconditionally and only `record_disk_state()`s. C2 wires **only** the
overwrite path that exists: `saved_cursor = cursor` after a successful save.
C2 does **not** add a stale-check/reload prompt to the save flow (that is
unauthorized save-flow scope). *Forward-looking note:* if a reload-or-
overwrite path is ever added, its **reload** branch (discard in-memory state
for disk) must both **reset the log** (`txns=[]`, `cursor=0`,
`saved_cursor=0`) and **refill the ctx handles in place** (§7 / C5) — because
the log's stored deltas no longer describe reachable states. That is a future
patch's line; nothing in C2 depends on it.

The `.bak` once-per-session guards (`csv_bak_written`, `toml_bak_written`,
`water_bak_written`) are independent of undo and are untouched by C2.

---

## 6. Atomicity & failure

**A compound transaction fully applies or fully does not — no half-state.**
The discipline that guarantees it:

1. **Validate before the first live mutation (C4).** Every (compound) gesture
   runs ALL of its validation before it mutates any live state: `door_check`,
   solid-`sample_tile` refusal, empty-fill, "did the drag move" — each is
   decided first, so a rejected gesture either never calls `begin()` or, if it
   had begun and touched nothing real, calls `abort()`. Consequence: once a
   gesture has mutated live state it is a *valid* change and MUST `commit()`
   — it never aborts a partial mutation away silently. `abort()` remains the
   safety net (it reverts from the retained before-copy, §2.1) but the
   validation ordering means it only ever fires on a no-op gesture.
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
3. **Ops within a transaction touch disjoint state and never read what a
   sibling writes.** Every op stores its own `before`/`after` and applies by
   assignment — no op *computes* its inverse from current state, so no op
   depends on the order relative to a sibling. This is exactly what makes
   "apply in order / invert in reverse order" sound (the door's grid op and
   entity op are on disjoint arrays/lists). **Normative constraint on future
   op types:** a new op type may only store-and-assign; it may never derive
   its effect from state another op in the same transaction writes. State it
   so a future op can't quietly break reverse-order inversion.
4. **Commit is the only mutation point of the log.** `commit()` builds all
   ops from already-applied live state and appends one Transaction; it never
   applies anything (the gesture already did). So there is no window where
   half a transaction is in the log.

With the save-time water mask now routed through the log (§5 / C1), **nothing
mutates live grid/collection state outside a transaction** — the invariant
holds with no exception. Accepted gap (dev tool, single trusted operator):
the single-threaded loop has no concurrent writer, so there is no
between-capture-and-undo race to guard; documented rather than defended.

---

## 7. Migration from the four rings

**Removed — `map_editor.py`-local only:** the four ring instances (`undo`,
`spawn_undo`, `light_undo`, `water_undo`), the `SpawnRing` **class** (defined
in `map_editor.py`, imported by no test — safe to delete), and the four
`dirty_*` bools.

**KEPT — `UndoRing` is NOT removable (B5).** `UndoRing` lives in the *shared*
`tools/level_edit_common.py` and is still used by the *other* editor
`tools/align_level_art.py` plus `tests/test_level_editor_tool.py` and
`tests/test_level_water.py` (all out of C2 scope). C2 only stops
`map_editor.py` from **instantiating** it; the class stays exactly as is.
`tests/test_map_editor_tool.py::test_undo_ring_restores_room_and_corridor_ops`
exercises `UndoRing` directly (not the editor's ring wiring), so it also
stays green untouched — the new log tests (§8) are *added*, not a replacement
for it.

**Added:** a `TransactionLog(depth=…, max_bytes=…)` plus a `ctx` registry
mapping names to the live state handles, constructed once in `run_editor`
after state is loaded:

```
ctx.grids       = {"material": grid, "water": water_q}      # + "zones","air" in C5
ctx.collections = {"spawns": spawns, "lights": lights, "entities": entities}  # + "wires" in C6
```

The grids dict holds the array objects; `undo` writes into them in place via
fancy indexing, so the closed-over `grid`/`water_q` names stay valid.
Collections use slice-assign for the same reason. **ctx handles are always
refilled IN PLACE (C5)** — `grid[...] = …`, `coll[:] = …`, never rebound to a
fresh object — or `ctx` would close over a dead array/list. (This matters at
any future log-reset/reload point, §5.)

*Forward note for C5:* the `zones` and `air` grids are **`None` at load**
until first allocated by their paint tool — a `GridCellsOp` on them before
allocation would target `None`. C5 must register those grids in `ctx.grids`
only once allocated (or allocate lazily on first paint, before `snapshot_grid`
runs). Relatedly, any tool that resizes/crops a grid changes its shape and
invalidates every stored `GridCellsOp` coordinate → such a tool must **reset
the log** (no Arc C tool resizes a grid, so this is a flag for whoever adds
one, not a C2 line).

**`ctx.collections["entities"]` IS the object Ctrl+S serializes (B4).** Today
`other_entities` is captured once at load (`map_editor.py:819`) as a *frozen*
list and is the object the save path merges (`format_lights_for_save(...,
other_entities)`). If the log mutated a *different* list, placed entities
would survive undo in-UI but vanish on save (and undo would restore a list
the writer ignores). Normative: **rename `other_entities` → `entities`, keep
it live and in-place-mutated** (`coll[:] = …` from `CollectionOp`), register
that same object as `ctx.collections["entities"]`, and route the light/entity
save off it. `CollectionOp`'s slice-assign preserves the object identity the
serializer holds.

**Shared `[[entity]]` id-allocator — a C2 invariant (B3).** `lights` and
`entities` are separate *logged* collections but both serialize to the ONE
`[[entity]]` family, where ids are unique-or-hard-error at load. Today
`unique_light_id` scans only `lights`; C3 place-one will scan only
`entities`; and C1 already put `light` in the ENTITY palette — so placing a
`light` via generic place-one while LIGHT mode holds `light_1` would mint a
second `light_1` → save → **unloadable level**. **Invariant:** every
collection that serializes to `[[entity]]` mints ids from ONE allocator that
scans the **union** of all such collections (`lights ∪ entities ∪ future`).
C2 defines the union-minting helper —
`unique_entity_id(prefix, *collections)` returning a `{id}`-free name over the
union — and routes `EditableLight` minting through it; C3 uses the same helper
for place-one/doors/sensors. (Full enforcement lands as C3 consumes it, but
the helper + invariant are fixed here so C3 can't reintroduce the collision.)

**Referential integrity for id-changing ops (C6 — forward rule).** Any op
that changes the *set of entity ids* (delete, clump-paste-with-re-id, future
re-id) MUST, in the **same transaction**, carry the `wires` and tag
`CollectionOp`s that reference those ids — otherwise undo/redo strands wires
on dead ids and a later save hard-errors. C2 states the rule; C4/C6 build the
ops. (This is *why* whole-collection snapshots are the right shape: a
clump-paste's re-id and its internal wires round-trip as one atomic snapshot
set.)

**Push-point rewrite** — mechanical, one gesture at a time: each
`undo.push(snap)` / `spawn_undo.push(...)` / `light_undo.push(...)` /
`water_undo.push(...)` site becomes a `snapshot_*` at the gesture's start and
a `commit` at its end (per §2.2). The mutation code between them is unchanged.

**Ctrl+Z / Ctrl+S block rewrite**: the mode-scoped `if mode == "SPAWN": …
elif "LIGHT": …` undo dispatch collapses to a single `log.undo()`;
redo (§4 keys) → `log.redo()`; on undo/redo, re-bake the union of the
transaction's grid-op `bounding_rect()`s and refresh spawn/light overlays
(collection ops) — the editor already re-derives overlays from the live lists
each frame, so only grid re-bakes need explicit rects.

### 7.1 Dirty consumers to rewire (C8 — checklist, NameError risk)

Removing the four `dirty_*` bools touches **six** sites, not just the dirty
computation; miss the last one and the editor NameErrors at quit. Each reads
a `dirty_*` name today and must move to the single `dirty` /
`log.cursor != log.saved_cursor` source (or the `saved_cursor`-set on save).
Approx. line numbers are pre-refactor `map_editor.py`:

1. `dirty_any` (~940) — the OR of all four → `dirty` (§5 formula).
2. Esc-quit guard (~945) — "unsaved changes, press Esc again" → gate on
   `dirty`.
3. Status-bar UNSAVED dot (~1854) → gate on `dirty`.
4. Save reset (~1490) — `dirty_tiles = dirty_spawns = … = False` → replace
   with `log.saved_cursor = log.cursor` (after the save-mask commit, §5).
5. Inspector "undo depth N" readout (~1839) → "undo N / redo M" (§4).
6. End-of-session PRINT (~1892-93) — prints all four `dirty_*` by name →
   **NameError at quit if missed**; replace with `dirty` (or drop the
   per-domain fields).

### 7.2 BEHAVIOR CHANGE — global undo history (call it out)

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
   lights, and entities — undo/redo restore deep-equal lists, and the list
   *object identity* is preserved (slice-assign, not rebind).
1b. **Deep-copy IDENTITY (B1)**: build a `CollectionOp` over an `entities`
   list, undo, then mutate a restored element's `.fields` dict and `.tags`;
   assert the op's retained `before`/`after` snapshots are **unaffected**
   (object-identity independence, not just equality). Also: an inspector
   field-edit that changes only a key inside `.fields` must commit a
   *non-empty* op (deep compare detects it) — the aliasing-drop bug, pinned
   as a regression test.
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
3b. **Whole-grid diff captures out-of-bbox cells (B2)**: a **CORRIDOR** op
   (walls land a tile beyond the drag-line bbox) and a **flood/bucket WATER
   fill** (changed region ≠ cursor bbox) each round-trip exactly through
   undo/redo — proving `commit()` diffs the whole grid, not the gesture rect.
   (A deliberately-wrong rect-bounded diff would leave residue here; this test
   is the guard against reintroducing it.)
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
7. **Memory-bound enforcement (injectable bounds, C7)**: construct
   `TransactionLog(depth=4, max_bytes=<tiny>)` so the test evicts without
   allocating 128 MB. Push `> depth` small transactions → oldest evicted,
   `len(txns) == depth`, and **cursor renumbering (C3)** is exact: after
   front-evicting `K`, `cursor` and `saved_cursor` are decremented by `K`
   (saved → `None` when it fell in the evicted prefix), and a subsequent
   undo/redo still applies the *correct* inverse (not an off-by-K one). Push
   grid-delta transactions past `max_bytes` → the log stays under the ceiling
   and never evicts the last undoable transaction.
7b. **Save-mask coherence (C1)**: fill water, wall over part of it, run the
   save-mask-as-committed-`GridCellsOp("water")` step, set
   `saved_cursor = cursor`; assert `dirty` is False, then undo/redo across the
   mask op keeps live water == what the mask wrote (no live-vs-disk desync).
7c. **Union id-allocator (B3)**: with `lights` holding `light_1`, mint a new
   `light` via the generic `unique_entity_id(..., lights, entities)` helper →
   assert the new id is absent from `lights ∪ entities` (no duplicate that
   would make the saved `[[entity]]` family unloadable).
8. **Fuzz round-trip**: from a known start state (a random grid + random
   spawn/light/entity lists), apply a random sequence of ops across all grids
   and collections through the builder; `undo()` all the way → state is
   deep-equal to the start (`np.array_equal` on every grid + deep dataclass
   equality on every collection); then `redo()` all the way → equal to the
   post-sequence state; and a random walk of interleaved undo/redo stays
   equal to an independent reference model that just replays the same cursor
   moves.

*Forward-looking (C4/C6, not built in C2):* a cross-collection referential
test — delete an entity that a wire references, in one transaction carrying
both the `entities` and `wires` `CollectionOp`s; undo/redo keeps wires and
their target ids consistent (no stranded wire on a dead id). Noted here so C4/
C6 add it when those ops land.

Gate (per arc): these unit tests + full `pytest tests -q` green + level_lib
byte-stable round-trip tests green + **zero goldens/digests touched** (the
editor is not sim path).

---

## 9. Residual risks & accepted gaps (post-critique)

The first-pass critique (data-loss / atomicity / scope-regression) is folded
into v2 (see Revision log §10). What remains is genuinely accepted, not open:

1. **Immediate-mode (mutate-then-capture) over a command-mode rewrite.** v2
   keeps immediate-mode because it maps one-to-one onto the existing
   push-points, keeping the 1.7k-line loop's mutation code intact. Its two
   sharp edges are now *closed*: the grid delta is a **whole-grid** diff
   (§2.1, B2 — no rect can drop a cell; `bounding_rect()` survives only as a
   re-bake hint), and `abort()` **reverts** from the retained before-copy
   (§2.1/§2.4, C4 — no half-applied gesture can survive). Commit-on-cancel for
   an interrupted drag stays (blessed as an improvement). Accepted gap: the
   whole-grid `!=` is O(cells) per commit — trivial (~65 k at 256², larger
   maps scale linearly but still sub-millisecond).

2. **Large deltas under the byte ceiling.** Bucket fills (C5 zone/air/water)
   and whole-map paints produce large deltas — up to ~4× a raw grid on a full
   rewrite (§3). This is why the **byte ceiling is the governor**, not the
   depth count, and why grid size being uncapped (`new --size` floor only) is
   handled by bytes rather than txn count. Accepted gap: a single fill larger
   than `LOG_BYTES` is still retained (never evict the last undoable action),
   and evicting the saved marker pins the dot to dirty until the next save
   (§3/§5) — correct-but-conservative for a dev tool.

3. **Whole-collection snapshots + one global dirty marker.** `CollectionOp`
   deep-copies the whole collection per op. Cheap now (tens of entities), and
   the right shape for C4 clump-paste re-id + internal-wire round-trips
   (§7 C6 rule). Accepted gap: on a level with thousands of entities,
   per-keypress inspector nudges each deep-copy the whole list — a time-window
   coalesce (§2.3) is deferred, not built. Collapsing four `dirty_*` flags to
   one global cursor is sound because save flushes every domain atomically
   (§5); no workflow depends on per-domain dirty. The log is reset only where
   in-memory state is actually replaced — and C2 has **no** such path (Ctrl+S
   overwrites unconditionally, §5/C2); the reset+refill-in-place discipline is
   documented for a *future* reload path, not wired now.

---

## 10. Revision log (v2)

v2 folds in the first-pass critique. Core model (two primitives + linear
cursor) unchanged. What changed:

- **B1 — deep-copy collections.** §1.1 `CollectionOp` now captures
  `copy.deepcopy(coll)`, not `[replace(e) …]` (which aliased
  `EntityInstance.fields`/`.tags` and silently dropped/aliased inspector
  edits). §8 adds the IDENTITY + non-empty-edit regression test (1b).
- **B2 — whole-grid delta diff.** §2.1 `commit()` diffs the **entire** grid
  (`before != live`); per-gesture rects are demoted to a re-bake hint only.
  Removed all "rect/`stroke_dirty` as the delta bound" language from §2.1/§9.
  §8 adds CORRIDOR + WATER-flood round-trip tests (3b). Corrected the §9
  inaccuracy (CORRIDOR already whole-grid-diffs today).
- **B3 — shared `[[entity]]` id-allocator.** §7 pins the invariant: all
  collections serializing to `[[entity]]` mint from one union-scanning
  helper `unique_entity_id(prefix, *collections)`. §8 adds the union test (7c).
- **B4 — save serializes the live logged list.** §7 renames
  `other_entities` → `entities`, pins `ctx.collections["entities"]` as the
  exact object Ctrl+S serializes, in-place-mutated via slice-assign.
- **B5 — `UndoRing` stays.** §7 corrected: `UndoRing` is shared
  (`level_edit_common.py`, used by align tool + 2 tests) — C2 only stops
  `map_editor.py` instantiating it; only the local `SpawnRing` class is
  deleted. §8's existing `UndoRing` test stays green.
- **C1 — save-time water mask is logged.** §5 wraps `mask_water_to_open` in a
  committed `GridCellsOp("water")` before `saved_cursor = cursor`; §6 drops
  the now-false "nothing mutates out-of-band" caveat (invariant now holds).
  §8 adds save-mask coherence test (7b).
- **C2 — no reload path invented.** §5 reframed: C2 wires only the
  unconditional-overwrite save that exists (`saved_cursor = cursor`); the
  log-reset is a forward-looking note for a future reload path; no
  `check_stale`/prompt added to the save flow.
- **C3 — eviction renumbers the cursor.** §3 gives the normative formula
  (`cursor -= K`; `saved_cursor` decremented or `None`). §8 asserts it (7).
- **C4 — validate-before-mutate + reverting abort.** §2.1/§2.4/§6: all
  validation precedes the first live mutation; a mutated gesture must commit;
  `abort()` reverts from the retained before-copy.
- **C5 — ctx handles refilled in place.** §7 states grids/collections are
  always mutated in place, never rebound; forward note for a reload path.
- **C6 — id-changing ops carry wire/tag deltas.** §7 rule for C4/C6; §8
  forward-looking cross-collection test note.
- **C7 — injectable bounds.** §3: `TransactionLog(depth=…, max_bytes=…)`;
  §8 eviction test uses a tiny ceiling.
- **C8 — dirty-consumer checklist.** §7.1 names the six `dirty_*` sites
  (incl. the end-of-session PRINT that would NameError at quit).
- **C9 — redo keybinding guard.** §4: `undo = ctrl and not shift and Z`;
  `redo = ctrl and (Y or (shift and Z))`.
- **Nits.** §3: pathological grid delta is ~4× a snapshot (not "same order"),
  and grid size is uncapped so the byte ceiling governs. §6: added the
  "no op reads a sibling op's write" constraint that makes reverse-order
  inversion sound. §7: C5 `zones`/`air` are `None` at load — register only
  once allocated; a resize/crop tool must reset the log.
