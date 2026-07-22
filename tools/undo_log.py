r"""tools/undo_log.py — the editor's single transaction-log undo (Arc C2).

Design: docs/arc_c_c2_undo_design_2026-07-22.md (v2). This module is the
headless, raylib-free core: two op primitives, a compound Transaction, the
linear-cursor TransactionLog with redo + saved-marker dirty tracking + a
builder seam, and the shared `[[entity]]` id allocator invariant helper. It
replaces the four per-domain LIFO rings (`UndoRing`/`SpawnRing`) that
`map_editor.py` used to carry with ONE global history of compound operations
(editor doc §3 pillar 6). Every later Arc C patch registers its op classes
onto this same log through the builder seam — no model change (§1.4).

The two primitives (§1.1):

  - ``GridCellsOp(grid_name, ys, xs, before, after)`` — a *delta* on one named
    numpy grid: changed cells only (``grid[ys, xs] = after/before``). Grids are
    O(10^4-10^5) cells, so a whole-grid snapshot per action is the memory
    problem the kickoff flags; delta-encoding is near-free on the common path.
  - ``CollectionOp(coll_name, before, after)`` — a bounded *snapshot pair* of
    one named dataclass list (``coll[:] = after/before``). Collections are
    O(10^1-10^2) tiny dataclasses, so a whole-list ``copy.deepcopy`` is
    microseconds and handles add/move/delete/edit/multi-select/re-id/paste with
    zero per-op-class code. The copy MUST be deep (B1): a shallow copy would
    alias ``EntityInstance.fields``/``.tags`` and silently drop inspector edits.

Each gets the representation its size warrants (§1.1 "the deliberate split").

Nothing here imports raylib; it is unit-tested against plain numpy grids and
dataclass lists in tests/test_editor_undo.py.
"""
from __future__ import annotations

import copy

import numpy as np

# Production log bounds (§3) — BOTH are constructor args so a test can pass a
# tiny ceiling instead of allocating 128 MB (C7, injectable bounds).
LOG_DEPTH = 256                       # max transactions retained
LOG_BYTES = 128 * 1024 * 1024         # 128 MB byte ceiling (the real governor)


class UndoContext:
    """The live-state registry a transaction writes into (§7). ``grids`` maps
    a name to the numpy array (mutated in place via fancy indexing);
    ``collections`` maps a name to the dataclass list (mutated via
    slice-assign). Both handles are refilled IN PLACE for the whole session,
    never rebound — or the log would close over a dead array/list."""

    def __init__(self, grids=None, collections=None):
        self.grids = dict(grids) if grids else {}
        self.collections = dict(collections) if collections else {}


class GridCellsOp:
    """A reversible delta on one named grid (§1.1). ``ys``/``xs`` are int32
    coordinate arrays; ``before``/``after`` are the grid's dtype, one entry per
    changed cell. Total assignment, never a computation — cannot raise on
    well-formed session state (§6.2)."""

    __slots__ = ("grid_name", "ys", "xs", "before", "after")

    def __init__(self, grid_name, ys, xs, before, after):
        self.grid_name = str(grid_name)
        self.ys = np.asarray(ys, dtype=np.int32)
        self.xs = np.asarray(xs, dtype=np.int32)
        self.before = np.asarray(before)
        self.after = np.asarray(after)

    def redo(self, ctx) -> None:
        ctx.grids[self.grid_name][self.ys, self.xs] = self.after

    def undo(self, ctx) -> None:
        ctx.grids[self.grid_name][self.ys, self.xs] = self.before

    def bounding_rect(self):
        """Inclusive-origin tile rect ``(tx0, ty0, tw, th)`` of the touched
        cells, or ``None`` when the op is empty. A re-bake HINT only (§2.1):
        the delta itself is the whole-grid diff, never bounded by this rect."""
        if self.ys.size == 0:
            return None
        x0, y0 = int(self.xs.min()), int(self.ys.min())
        return (x0, y0,
                int(self.xs.max()) - x0 + 1, int(self.ys.max()) - y0 + 1)

    @property
    def nbytes(self) -> int:
        return int(self.ys.nbytes + self.xs.nbytes
                   + self.before.nbytes + self.after.nbytes)


class CollectionOp:
    """A reversible whole-list snapshot pair for one named collection (§1.1).
    ``before``/``after`` are DEEP copies (B1); undo/redo slice-assign a fresh
    deep copy so the live list never shares mutable substructure
    (``EntityInstance.fields``/``.tags``) with the retained snapshot."""

    __slots__ = ("coll_name", "before", "after")

    def __init__(self, coll_name, before, after):
        self.coll_name = str(coll_name)
        self.before = before
        self.after = after

    def redo(self, ctx) -> None:
        ctx.collections[self.coll_name][:] = copy.deepcopy(self.after)

    def undo(self, ctx) -> None:
        ctx.collections[self.coll_name][:] = copy.deepcopy(self.before)

    def bounding_rect(self):
        return None                       # collections have no grid rect

    @property
    def nbytes(self) -> int:
        # Tens of small dataclasses; a cheap structural estimate is enough for
        # the byte ceiling (grids dominate it). 256 bytes/element is generous.
        return 256 * (len(self.before) + len(self.after))


class Transaction:
    """One atomic compound op (§1.2): ordered ``ops``, applied in order on
    redo, INVERTED in reverse order on undo. A door is
    ``[GridCellsOp("material"), CollectionOp("entities")]`` — the archetype.
    All-or-nothing because ops are total store-and-assign and never read a
    sibling's write (§6.3)."""

    __slots__ = ("label", "ops")

    def __init__(self, label, ops):
        self.label = str(label)
        self.ops = list(ops)

    def redo(self, ctx) -> None:
        for op in self.ops:
            op.redo(ctx)

    def undo(self, ctx) -> None:
        for op in reversed(self.ops):
            op.undo(ctx)

    @property
    def nbytes(self) -> int:
        return sum(op.nbytes for op in self.ops)

    def rebake_rects(self):
        """``(grid_name, (tx0, ty0, tw, th))`` for each non-empty grid op — so
        the editor re-bakes only what a grid op moved on undo/redo (collection
        ops yield nothing; their overlays re-derive from the live lists each
        frame)."""
        out = []
        for op in self.ops:
            r = op.bounding_rect()
            if r is not None:
                out.append((op.grid_name, r))
        return out


class _Pending:
    """The open-transaction builder state: the transient BEFORE snapshots held
    only during a gesture (discarded at commit — the log stores just the
    extracted delta, so it stays delta-encoded, §2.1)."""

    __slots__ = ("label", "grid_before", "coll_before")

    def __init__(self, label):
        self.label = label
        self.grid_before = {}     # grid name -> pre-gesture np copy
        self.coll_before = {}     # coll name -> pre-gesture deep-copied list


class TransactionLog:
    """One global linear undo/redo history of compound transactions (§4).

    ``txns[:cursor]`` are applied; ``txns[cursor:]`` are the redoable tail.
    ``saved_cursor`` is the cursor value equal to on-disk state (``None`` once
    the saved state falls out of the log via eviction). ``dirty`` is a cursor
    comparison, not a set of bools — undoing back to the saved position clears
    the unsaved dot (§5).

    The builder seam (§2.1): ``begin(label)`` / ``snapshot_grid(name)`` /
    ``snapshot_coll(name)`` open a pending transaction and capture transient
    before-copies; the gesture mutates live state exactly as before;
    ``commit()`` extracts the deltas vs the now-mutated state and pushes ONE
    transaction (or drops it if empty); ``abort()`` reverts live state from the
    retained before-copies and drops the pending transaction (§2.4).
    """

    def __init__(self, ctx, depth: int = LOG_DEPTH, max_bytes: int = LOG_BYTES):
        self.ctx = ctx
        self.depth = int(depth)
        self.max_bytes = int(max_bytes)
        self.txns: list = []
        self.cursor = 0
        self.saved_cursor = 0         # open state == on-disk state -> clean
        self._pending = None

    # ---- builder seam ----------------------------------------------------

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def begin(self, label) -> None:
        """Open the pending transaction. Idempotent within a gesture: a repeat
        call (e.g. per paint frame) keeps the first snapshots and label."""
        if self._pending is None:
            self._pending = _Pending(str(label))

    def snapshot_grid(self, name) -> None:
        """Capture the transient before-copy of a grid (once per gesture per
        grid). No-op if already snapshotted this gesture."""
        if self._pending is None:
            raise RuntimeError("snapshot_grid() without begin()")
        if name not in self._pending.grid_before:
            self._pending.grid_before[name] = np.array(
                self.ctx.grids[name], copy=True)

    def snapshot_coll(self, name) -> None:
        """Capture the transient deep before-copy of a collection (once per
        gesture). No-op if already snapshotted this gesture."""
        if self._pending is None:
            raise RuntimeError("snapshot_coll() without begin()")
        if name not in self._pending.coll_before:
            self._pending.coll_before[name] = copy.deepcopy(
                self.ctx.collections[name])

    def abort(self) -> None:
        """Revert live state from the retained before-copies and drop the
        pending transaction (§2.1/§2.4). A true no-op regardless of how far the
        gesture had mutated. Cheap — the before-copies are already held."""
        p = self._pending
        if p is None:
            return
        for name, before in p.grid_before.items():
            self.ctx.grids[name][...] = before
        for name, before in p.coll_before.items():
            self.ctx.collections[name][:] = copy.deepcopy(before)
        self._pending = None

    def commit(self):
        """Extract the delta of every snapshotted grid/collection vs the now
        mutated live state and push ONE Transaction. Returns the pushed
        Transaction, or ``None`` when the gesture changed nothing (a no-op
        gesture never dirties the log, §2.1). Never applies anything — the
        gesture already mutated live state."""
        p = self._pending
        if p is None:
            return None
        self._pending = None
        ops: list = []
        # Grid ops: whole-grid diff (B2) — the delta is NEVER bounded by the
        # gesture's rect (a rect bound silently drops CORRIDOR walls and every
        # flood/fill cell outside the cursor bbox).
        for name, before in p.grid_before.items():
            live = self.ctx.grids[name]
            mask = (before != live)
            if mask.any():
                ys, xs = mask.nonzero()
                ops.append(GridCellsOp(name, ys.astype(np.int32),
                                       xs.astype(np.int32),
                                       np.array(before[mask], copy=True),
                                       np.array(live[mask], copy=True)))
        # Collection ops: deep compare; drop if unchanged. The before snapshot
        # is already an independent deep copy; hand it to the op and deep-copy
        # only the live side.
        for name, before in p.coll_before.items():
            live = self.ctx.collections[name]
            if before != list(live):
                ops.append(CollectionOp(name, before, copy.deepcopy(list(live))))
        if not ops:
            return None
        txn = Transaction(p.label, ops)
        self._push(txn)
        return txn

    # ---- linear undo / redo ---------------------------------------------

    def undo(self):
        """Invert the most-recent applied transaction (global history — the
        last action across ALL domains, §7.2). Returns it, or ``None`` when
        there is nothing to undo."""
        if self.cursor <= 0:
            return None
        txn = self.txns[self.cursor - 1]
        txn.undo(self.ctx)
        self.cursor -= 1
        return txn

    def redo(self):
        """Re-apply the next redoable transaction. Returns it, or ``None`` when
        the redo tail is empty."""
        if self.cursor >= len(self.txns):
            return None
        txn = self.txns[self.cursor]
        txn.redo(self.ctx)
        self.cursor += 1
        return txn

    def _push(self, txn) -> None:
        """Append a freshly committed transaction: truncate the redo tail
        (§4 — a new action after undo discards the redone-away future), then
        append, advance the cursor, and enforce the memory bounds."""
        # If the saved state lived in the tail we are about to truncate, it is
        # no longer reachable -> the dot can no longer prove state==disk.
        if self.saved_cursor is not None and self.saved_cursor > self.cursor:
            self.saved_cursor = None
        del self.txns[self.cursor:]
        self.txns.append(txn)
        self.cursor += 1
        self._enforce_bounds()

    def _total_bytes(self) -> int:
        return sum(t.nbytes for t in self.txns)

    def _enforce_bounds(self) -> None:
        """Evict oldest transactions from the FRONT until BOTH the depth and
        byte bounds hold, renumbering the cursor (§3, C3). Never evict the last
        undoable action — a single fill larger than the ceiling is retained
        (the accepted gap)."""
        while len(self.txns) > 1 and (
                len(self.txns) > self.depth
                or self._total_bytes() > self.max_bytes):
            K = 1
            del self.txns[:K]
            self.cursor = max(0, self.cursor - K)
            # NEW-1 (v2 verification): guard the None case FIRST — a second
            # front-eviction after saved_cursor is already None must not do
            # `None - K` (TypeError). It simply stays None.
            if self.saved_cursor is not None:
                self.saved_cursor = (None if self.saved_cursor <= K
                                     else self.saved_cursor - K)

    # ---- saved marker / dirty (§5) --------------------------------------

    @property
    def dirty(self) -> bool:
        """Position-accurate unsaved flag: True unless the cursor sits exactly
        at the saved position. Undoing back to the saved state clears it;
        redoing or editing away re-dirties (§5)."""
        return self.saved_cursor is None or self.cursor != self.saved_cursor

    def mark_saved(self) -> None:
        """Pin the current cursor as the on-disk position — the LAST step of a
        successful save (after the save-mask commit, §5)."""
        self.saved_cursor = self.cursor

    # ---- HUD helpers -----------------------------------------------------

    @property
    def undo_count(self) -> int:
        return self.cursor

    @property
    def redo_count(self) -> int:
        return len(self.txns) - self.cursor
