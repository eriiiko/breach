"""#29 P7e.3 -- THE EDIT DOOR, STEP (c): slot 6b's flush built ON THE DEVICE
from the edit door's rows (``simulation.edit_door.DeviceEditFlush`` over
``cpp/src/cuda_edit_flush.cu``), fed to the device edit queue's K1 / K2
(``cpp/src/cuda_edit_queue.cu``), against the same K1 / K2 fed from the host
pack (``EditQueue.take_ordered`` + ``field_edit.pack_for_device`` +
``physics_runner._edit_buffer``) at tol 0.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_edit_flush.py).

PART SORT (item 1) -- hand-built windows on a real map: edits of seven
fields, several sources and the three door regions (TILE, DISC FLAT / LINEAR,
TILES; a noisy and scaled blast-smoke DISC among them) enqueued OUT of their
applied order with tied ``(field, source, region)``, at N = 1 and N = 2 (a
different window per env). The device-built buffers (the rows in applied
order, the tiles carried with their edit, the batch-global edit index, the
env offsets, the tick word / key / ambient scalars) equal the host pack's
layout row for row, and every written plane and book equals the host-pack
flush. The host's sort, pack and buffer are made to RAISE while the device
flush runs: a fallback to the host pack is a failure.

PART RANGE (item 2) -- each refusal of ``pack_for_device`` reached by a row
the host also refuses or falls back on (a water_depth edit, a gas slice past
G, a conservative slice, a den of 2^62, a contribution of 2^55, a product
past 2^125), refused BY NAME for its env while the other env of the batch is
applied; a row just inside each numeric bound (den 2^62 - 1, contribution
2^55 - 1, the last trace slice) accepted and equal to the host pack. The
door-integrity refusals (an unknown field, a mode, a noise bound, a key gap,
a tile off the grid) refused by name too.

PART WINDOW (item 3) -- rows written after a flush (a 9e producer's), the
tick row then advanced (the boundary), more rows (slots 1 to 5): the door
still holds the first rows after the boundary, and the flush applies both at
the new tick's draws, equal to the host queue's window.

PART SEQ (item 4) -- the flush resets every env's seq to 0 (a row keyed 0
continues it, the host queue's next seq); a noisy edit at an ABSOLUTE seq
(a window base of 5) draws as the host's at seq 5; ``reset_env`` resets one
env alone.

PART SCENES (items 1 to 5 on played scenes) -- the spray, payload (grenades
and a planted charge) and hazard (timed charges, heat sources) scenes, N = 1
and N = 2 with different envs: the host plays; every edit the host enqueues
is appended to the device door at its seq as a producer writes it (the
producers' own rows equal ``door_rows`` -- the door's gates), the rows of
slots 7 to 9e after the flush, so the door holds them across the tick
boundary; at each 6b the door's window equals the host queue's, the device
flush equals the host-pack flush (planes and books), and the door is empty
with seq 0 after it. A census counts the ticks with edits out of order with
ties, noisy edits and windows that span the boundary.

PROPERTY: the device flush applies the window the device door holds, in the
host's applied order, with the host's integers, books and refusals. BREAKS
IF the sort is unstable or keyed otherwise, a tile leaves its edit, the edit
index or the offsets are not batch-global, a scalar is not the tick row's /
the table's, a range check is missing or misplaced, the door is emptied
before the flush or not at it, or the seq is not reset at the flush. Each
validated by breaking it (the patch's report).
"""
from __future__ import annotations

import contextlib
import os
import sys
import time

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

import cupy as cp  # noqa: E402

from simulation import edit_door, field_edit as fe, physics_runner  # noqa: E402
from simulation import unit_draws, unit_store, unit_tables  # noqa: E402
from simulation.edit_door import (DeviceEditDoor, DeviceEditFlush,  # noqa: E402
                                  EditFlushRefused)

FAILS: list = []
SCENE_TICKS = int(os.environ.get("EDIT_FLUSH_SCENE_TICKS", "70"))
WRITTEN = ("gas", "temperature", "gas_energy", "fire", "heat")
ONE = 65536


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------
def host_planes(gmap) -> dict:
    return {name: np.ascontiguousarray(getattr(gmap, name)).copy()
            for name, _dt in edit_door.FLUSH_PLANES}


def dev_planes(per_env) -> dict:
    return {name: cp.asarray(np.stack([p[name] for p in per_env]))
            for name, _dt in edit_door.FLUSH_PLANES}


def gas_args(gmap):
    ids = gmap.gases.name_to_id
    return int(gmap.gas.shape[0]), int(ids["o2"]), int(ids["inert_n2"])


class Rig:
    """A batch's door, the device flush, the reference flush and tables."""

    def __init__(self, sims):
        self.sims = list(sims)
        n = len(self.sims)
        self.door = DeviceEditDoor(n, bp=bp)
        self.flush = DeviceEditFlush(self.door)
        self._ref_door = DeviceEditDoor(n, bp=bp)
        self.ref = DeviceEditFlush(self._ref_door)
        self.store = unit_store.DeviceStore(n, 1, bp=bp,
                                            tables=unit_tables.values_batch(self.sims))

    def tick_row(self, rows=None):
        if rows is None:
            rows = np.stack([unit_draws.tick_row(s) for s in self.sims])
        return unit_draws.TickRow.of(rows)


@contextlib.contextmanager
def no_host_pack():
    """The host's sort, pack and buffer RAISE inside: the device flush must
    not fall back to them."""
    saved = (fe.pack_for_device, fe.EditQueue.take_ordered, physics_runner._edit_buffer)

    def boom(*_a, **_k):
        raise AssertionError("the device flush fell back to the host pack")
    fe.pack_for_device = boom
    fe.EditQueue.take_ordered = boom
    physics_runner._edit_buffer = boom
    try:
        yield
    finally:
        fe.pack_for_device, fe.EditQueue.take_ordered, physics_runner._edit_buffer = saved


def compare(tag, rig, packs, planes_np, tick_row, n_edits, expect_refused=()):
    """The device flush of the rig's door vs the host packs' flush, on the
    same planes. Returns the device books (or the refusal raised)."""
    g0 = rig.sims[0].gmap
    ng, o2, n2 = gas_args(g0)
    a, b = dev_planes(planes_np), dev_planes(planes_np)
    rig.ref.run_host_pack(packs, b, ng, o2, n2)
    bref = rig.ref.books()
    refused = None
    with no_host_pack():
        rig.flush.run(tick_row, rig.store.tables, a, ng, o2, n2)
        try:
            bdev = rig.flush.books()
        except EditFlushRefused as ex:
            refused = ex
            bdev = None
    ok = True
    if refused is not None:
        if refused.env not in expect_refused:
            fail(f"{tag}: refused unexpectedly: {refused}")
            return None
    elif expect_refused:
        fail(f"{tag}: env(s) {list(expect_refused)} not refused")
        return None
    for e in range(len(rig.sims)):
        if e in expect_refused:
            for name in WRITTEN:       # a refused env keeps its planes
                if not np.array_equal(a[name][e].get(), planes_np[e][name]):
                    fail(f"{tag}: refused env {e} wrote {name}")
                    ok = False
            continue
        for name in WRITTEN:
            if not np.array_equal(a[name][e].get(), b[name][e].get()):
                d = np.argwhere(a[name][e].get() != b[name][e].get())
                fail(f"{tag}: env {e} {name} differs from the host pack at "
                     f"{len(d)} cells, first {d[0].tolist()}")
                ok = False
        if bdev is not None and not np.array_equal(bdev[e], bref[e]):
            fail(f"{tag}: env {e} books differ: {np.nonzero(bdev[e] != bref[e])[0].tolist()}")
            ok = False
    if refused is None:
        rows = rig.flush.rows(n_edits)
        for e, p in enumerate(packs):
            er, tr, env = rows[e]
            if p is None:
                if er.shape[0] or tr.shape[0]:
                    fail(f"{tag}: env {e} built rows for an empty window")
                    ok = False
                continue
            pe = p.edits.copy()
            pe[:, fe.E_ENV] = e
            if not np.array_equal(er, pe):
                fail(f"{tag}: env {e} edit rows differ from the host pack")
                ok = False
            if not np.array_equal(tr, p.tiles):
                fail(f"{tag}: env {e} tile rows differ from the host pack")
                ok = False
            want = (p.tick_word, p.key[0], p.key[1], p.t_amb_raw)
            if tuple(int(x) for x in env) != want:
                fail(f"{tag}: env {e} scalars {env.tolist()} != {want}")
                ok = False
    # THE RESET: the door empty, every seq 0
    for e, (er, tr) in enumerate(rig.door.rows()):
        if er.shape[0] or tr.shape[0]:
            fail(f"{tag}: env {e}'s door not emptied by the flush")
            ok = False
    if rig.door.seq().any():
        fail(f"{tag}: seq not reset at the flush: {rig.door.seq().tolist()}")
        ok = False
    if refused is not None:
        return refused
    return bdev if ok else None


def append_pairs(rig, e, pairs):
    if pairs:
        er, tr = edit_door.door_rows(rig.sims[e].gmap, pairs)
        rig.door.append_probe(e, er, tr)


def host_pack(sim, pairs):
    q = fe.EditQueue()
    ordered = sorted(pairs, key=q._sort_key)
    if not ordered:
        return None
    pack, reason = fe.pack_for_device(sim.gmap, ordered, sim.draws)
    if pack is None:
        raise AssertionError(f"the host pack falls back ({reason})")
    return pack


# ---------------------------------------------------------------------------
# The hand-built windows
# ---------------------------------------------------------------------------
def _map_sim(variant=0):
    from _hazard_scene import HazardScene
    sim = HazardScene(variant, bp=bp).sim
    for _ in range(2 + variant):          # a few ticks: some gas, heat, a varied map
        sim.step()
        sim.set_paused(False)
    return sim


def _trace_slices(gmap):
    return [g for g in range(gmap.gas.shape[0]) if not bool(gmap.gases.conservative[g])]


def _open_tiles(gmap, k, rng):
    free = np.argwhere(~gmap.solid[2:-2, 2:-2]) + 2
    return [tuple(int(v) for v in free[i]) for i in rng.choice(len(free), k, replace=False)]


def window_edits(sim, seed):
    """A window of edits over seven fields, several sources and the three door
    regions, in a scrambled order with tied (field, source, region)."""
    from simulation import gases as gas_ids
    g = sim.gmap
    rng = np.random.default_rng(seed)
    t = _open_tiles(g, 24, rng)
    trace = [s for s in _trace_slices(g) if s != int(gas_ids.SMOKE)] or _trace_slices(g)
    scale = np.full(g.solid.shape, ONE, dtype=np.int64)
    scale[t[5][0], t[5][1]] = ONE // 3
    E = fe.FieldEdit
    R, F, M = fe.Region, fe.Falloff, fe.EditMode
    edits = [
        E("heat", R.TILE, t[0], 3.5, source_id=4),
        E("heat", R.TILE, t[1], 1.25, source_id=4),                      # tie with the above
        E("heat", R.DISC, (t[2][0], t[2][1], 2.6), 7.0, falloff=F.LINEAR, source_id=4),
        E("fire", R.DISC, (t[3][0], t[3][1], 1.8), 0.6, mode=M.MAX, source_id=9),
        E("smoke", R.DISC, (t[5][0], t[5][1], 3.2), 0.8, noise=0.35, scale=scale,
          source_id=11),
        E("gas", R.TILES, ((t[6][0], t[6][1], 3), (t[7][0], t[7][1], 5)), 0.4,
          channel=trace[0], source_id=2, weight_den=8),
        E("atmosphere", R.DISC, (t[8][0], t[8][1], 2.2), 0.3, source_id=6),
        E("wave_source", R.TILE, t[9], 2.0, source_id=1),
        E("gas_energy", R.TILE, t[10], 0.0, amount_q=123456789, source_id=3),
        E("heat", R.TILE, t[11], 2.0, source_id=1),
        E("smoke", R.DISC, (t[12][0], t[12][1], 1.5), 0.2, source_id=11),   # tie, no noise
        E("fire", R.TILE, t[13], 0.3, source_id=9),
        E("heat", R.TILE, t[14], 0.5, source_id=4),                      # another tie
        E("gas", R.TILE, t[15], 0.1, channel=trace[-1], source_id=2),
        E("atmosphere", R.TILE, t[16], 0.05, mode=M.REMOVE, source_id=6),
    ]
    order = rng.permutation(len(edits))
    return [edits[i] for i in order]


def _pairs(edits, base=0):
    return [(e, base + k) for k, e in enumerate(edits)]


def _check_scrambled(tag, pairs):
    q = fe.EditQueue()
    ordered = sorted(pairs, key=q._sort_key)
    keys = [q._sort_key(p)[:3] for p in pairs]
    if [s for _e, s in ordered] == [s for _e, s in pairs]:
        fail(f"{tag}: the window arrived in its applied order (not a sort test)")
    if len(set(keys)) == len(keys):
        fail(f"{tag}: no tied (field, source, region) in the window")


def part_sort():
    sims = [_map_sim(0), _map_sim(2)]
    for n in (1, 2):
        rig = Rig(sims[:n])
        rig.door.begin()
        packs, counts = [], []
        for e in range(n):
            pairs = _pairs(window_edits(sims[e], 101 + e))
            _check_scrambled(f"sort N={n} env {e}", pairs)
            append_pairs(rig, e, pairs)
            packs.append(host_pack(sims[e], pairs))
            counts.append(len(pairs))
        b = compare(f"sort N={n}", rig, packs, [host_planes(s.gmap) for s in sims[:n]],
                    rig.tick_row(), counts)
        if b is not None:
            print(f"  sort N={n}: {sum(counts)} edits, "
                  f"{sum(p.tiles.shape[0] for p in packs)} tiles, noise draws "
                  f"{int(b[:, rig.flush.bp.EDIT_QUEUE_LAYOUT['noise']].sum())}: equal")


# ---------------------------------------------------------------------------
# PART RANGE
# ---------------------------------------------------------------------------
def _stage(rig, per_env):
    """Set the door from raw rows (no producer check): (edits, tiles) per env."""
    d = rig.door
    E, T = edit_door.EDIT_COLS, edit_door.TILE_COLS
    er = np.zeros((d.n_env, d.edit_cap, E), dtype=np.int64)
    tr = np.zeros((d.n_env, d.tile_cap, T), dtype=np.int64)
    ec = np.zeros(d.n_env, dtype=np.int64)
    tc = np.zeros(d.n_env, dtype=np.int64)
    seq = np.zeros(d.n_env, dtype=np.int64)
    for e, (a, t) in enumerate(per_env):
        er[e, :len(a)] = a
        tr[e, :len(t)] = t
        ec[e], tc[e] = len(a), len(t)
        seq[e] = (int(a[0][fe.E_KEY]) + len(a)) if len(a) else 0
    d.buffer.edits.upload(er, ec)
    d.buffer.tiles.upload(tr, tc)
    d.set_seq(seq)


def _rows(sim, edit):
    return edit_door.door_rows(sim.gmap, [(edit, 0)])


def _refused(tag, rig, per_env, ok_env_pack, code_name, sims):
    """Env 0 holds the refused rows, env 1 a valid window: env 0 refused by
    name, env 1 applied as the host pack applies it."""
    _stage(rig, per_env)
    planes = [host_planes(s.gmap) for s in sims]
    got = compare(tag, rig, [None, ok_env_pack], planes, rig.tick_row(),
                  [0, len(per_env[1][0])], expect_refused=(0,))
    if isinstance(got, EditFlushRefused):
        if got.name != code_name:
            fail(f"{tag}: refused as {got.name!r}, want {code_name!r}")
        else:
            print(f"  {tag}: refused by name ({got.name}), env 1 applied")
    elif got is not None:
        fail(f"{tag}: not refused")


def _accepted(tag, rig, sims, edits0):
    pairs = _pairs(edits0)
    rig.door.begin()
    append_pairs(rig, 0, pairs)
    pk = host_pack(sims[0], pairs)
    p1 = _pairs([fe.FieldEdit("heat", fe.Region.TILE, (3, 3), 1.0, source_id=1)])
    append_pairs(rig, 1, p1)
    got = compare(tag, rig, [pk, host_pack(sims[1], p1)],
                  [host_planes(s.gmap) for s in sims], rig.tick_row(), [len(pairs), 1])
    if got is not None and not isinstance(got, EditFlushRefused):
        print(f"  {tag}: accepted, equal to the host pack")


def part_range():
    sims = [_map_sim(0), _map_sim(2)]
    rig = Rig(sims)
    s0, g0 = sims[0], sims[0].gmap
    E, R = fe.FieldEdit, fe.Region
    tile = _open_tiles(g0, 1, np.random.default_rng(5))[0]
    ok_pairs = _pairs([E("heat", R.TILE, (4, 4), 2.0, source_id=3)])
    ok_er, ok_tr = edit_door.door_rows(sims[1].gmap, ok_pairs)
    ok_pack = host_pack(sims[1], ok_pairs)
    G = int(g0.gas.shape[0])
    trace = _trace_slices(g0)
    cons = [k for k in range(G) if bool(g0.gases.conservative[k])]

    def host_says(edit):
        try:
            pack, reason = fe.pack_for_device(g0, [(edit, 0)], s0.draws)
        except Exception as ex:          # the host flush raises
            return f"raises {type(ex).__name__}"
        return reason

    def case(name, edit_host, rows, code):
        why = host_says(edit_host)
        if why is None:
            fail(f"range {name}: the host pack serves the row (not a refusal case)")
            return
        _refused(f"range {name} (host: {why})", rig, [rows, (ok_er, ok_tr)], ok_pack, code,
                 sims)

    base_er, base_tr = _rows(s0, E("heat", R.TILE, tile, 2.0, source_id=7))
    # a water_depth edit: the host pack falls back
    er = base_er.copy()
    er[0, fe.E_FIELD] = fe.DEVICE_FIELD_ORDINALS["water_depth"]
    case("water_depth", E("water_depth", R.TILE, tile, 0.1), (er, base_tr), "water_depth")
    # a gas slice past G: the host pack falls back; the last trace slice is served
    g_er, g_tr = _rows(s0, E("gas", R.TILE, tile, 0.2, channel=trace[-1], source_id=7))
    er = g_er.copy()
    er[0, fe.E_SLICE] = G
    case("gas_slice", E("gas", R.TILE, tile, 0.2, channel=G), (er, g_tr), "gas_slice")
    _accepted("range gas_slice inside (the last trace slice)", rig, sims,
              [E("gas", R.TILE, tile, 0.2, channel=trace[-1], source_id=7)])
    # a conservative slice: the host raises
    if cons:
        er = g_er.copy()
        er[0, fe.E_SLICE] = cons[0]
        case("conservative", E("gas", R.TILE, tile, 0.2, channel=cons[0]), (er, g_tr),
             "conservative")
    # den 2^62: the host pack falls back; 2^62 - 1 is served
    d_in = (1 << 62) - 1
    e_in = E("heat", R.TILES, ((tile[0], tile[1], d_in),), 0.0, amount_q=1000, source_id=7,
             weight_den=d_in)
    d_er, d_tr = _rows(s0, e_in)
    if int(d_er[0, fe.E_DEN]) != d_in:
        fail(f"range den: the door wrote den {int(d_er[0, fe.E_DEN])}, not {d_in}")
    _accepted("range den 2^62 - 1", rig, sims, [e_in])
    er = d_er.copy()
    er[0, fe.E_DEN] = 1 << 62
    tr = d_tr.copy()
    tr[0, 2] = 1 << 62
    case("den", E("heat", R.TILES, ((tile[0], tile[1], 1 << 62),), 0.0, amount_q=1000,
                  source_id=7, weight_den=1 << 62), (er, tr), "den")
    # the contribution: 2^55 - 1 served, 2^55 falls back
    c_in = (1 << 55) - 1
    _accepted("range contribution 2^55 - 1", rig, sims,
              [E("heat", R.TILE, tile, 0.0, amount_q=c_in, source_id=7)])
    er = base_er.copy()
    er[0, fe.E_AMOUNT] = 1 << 55
    case("contrib", E("heat", R.TILE, tile, 0.0, amount_q=1 << 55, source_id=7),
         (er, base_tr), "contrib")
    # the product past 2^125 (a scaled TILES row): the host pack falls back
    sc = np.full(g0.solid.shape, ONE, dtype=np.int64)
    sc[tile] = 2 * ONE
    big = (1 << 63) - 1
    p_edit = E("heat", R.TILES, ((tile[0], tile[1], 1 << 62),), 0.0, amount_q=big,
               source_id=7, weight_den=1, scale=sc)
    p_er, p_tr = _rows(s0, E("heat", R.TILES, ((tile[0], tile[1], 1),), 0.0, amount_q=1,
                             source_id=7, weight_den=1, scale=sc))
    er, tr = p_er.copy(), p_tr.copy()
    er[0, fe.E_AMOUNT] = big
    tr[0, 2] = 1 << 62
    case("product", p_edit, (er, tr), "product")
    # the door's integrity (the played path's resident_edits refuses the same)
    for name, col, val in (("field", fe.E_FIELD, 8), ("mode", fe.E_MODE, 3)):
        er = base_er.copy()
        er[0, col] = val
        _refused(f"range {name}", rig, [(er, base_tr), (ok_er, ok_tr)], ok_pack, name, sims)
    n_er, n_tr = _rows(s0, E("smoke", R.DISC, (tile[0], tile[1], 1.5), 0.3, noise=0.2,
                             source_id=7))
    er = n_er.copy()
    er[0, fe.E_LOW] = ONE + 1
    _refused("range noise_low", rig, [(er, n_tr), (ok_er, ok_tr)], ok_pack, "noise_low",
             sims)
    two_er, two_tr = edit_door.door_rows(g0, _pairs([E("heat", R.TILE, tile, 1.0),
                                                     E("heat", R.TILE, tile, 2.0)]))
    er = two_er.copy()
    er[1, fe.E_KEY] = 5
    _refused("range key", rig, [(er, two_tr), (ok_er, ok_tr)], ok_pack, "key", sims)
    tr = base_tr.copy()
    tr[0, 1] = g0.solid.size
    _refused("range tile", rig, [(base_er, tr), (ok_er, ok_tr)], ok_pack, "tile", sims)


# ---------------------------------------------------------------------------
# PART WINDOW / PART SEQ
# ---------------------------------------------------------------------------
def part_window():
    sims = [_map_sim(0)]
    rig = Rig(sims)
    s = sims[0]
    edits = window_edits(s, 707)
    first, second = edits[:6], edits[6:]
    rig.door.begin()
    # 9e of tick t: rows after the flush
    append_pairs(rig, 0, _pairs(first))
    row_t = unit_draws.tick_row(s)
    trow = rig.tick_row(row_t[None, :])
    # THE BOUNDARY: the latch advances the tick row (tick, draw tick) in place;
    # nothing touches the door
    row_t1 = row_t.copy()
    row_t1[0] += 1
    row_t1[unit_draws.TICK_ROW_KEY + 2] += 1
    trow.write(row_t1[None, :])
    held = rig.door.rows()[0][0].shape[0]
    if held != len(first):
        fail(f"window: the door holds {held} rows after the boundary, not {len(first)}")
    # slots 1 to 5 of tick t + 1
    pairs = _pairs(first + second)
    append_pairs(rig, 0, pairs[len(first):])

    from simulation.draws import MatchDraws
    d = MatchDraws(s.draws.seed)           # the host draws at tick t + 1
    d.set_tick(int(row_t1[unit_draws.TICK_ROW_KEY + 2]))
    if tuple(d.key) != tuple(s.draws.key):
        fail("window: the draws' key is not the sim's")
    q = fe.EditQueue()
    ordered = sorted(pairs, key=q._sort_key)
    pack, reason = fe.pack_for_device(s.gmap, ordered, d)
    if pack is None:
        fail(f"window: the host pack falls back ({reason})")
        return
    if not any(e.noise > 0 for e, _s in pairs[:len(first)] + pairs[len(first):]):
        fail("window: no noisy edit (the tick word would not matter)")
    b = compare("window", rig, [pack], [host_planes(s.gmap)], trow, [len(pairs)])
    if b is not None:
        print(f"  window: {len(first)} rows across the boundary + {len(second)} after: "
              f"equal to the host window at the new tick")


def part_seq():
    sims = [_map_sim(0), _map_sim(2)]
    rig = Rig(sims)
    E, R = fe.FieldEdit, fe.Region
    g = sims[0].gmap
    t = _open_tiles(g, 3, np.random.default_rng(9))
    noisy = E("smoke", R.DISC, (t[0][0], t[0][1], 2.5), 0.7, noise=0.5, source_id=11)
    # an ABSOLUTE seq: a window based at 5 (the host queue's seq at the walk)
    rig.door.begin()
    rig.door.set_seq(np.array([5, 0], dtype=np.int64))
    p0 = [(noisy, 5)]
    append_pairs(rig, 0, p0)
    p1 = _pairs([E("heat", R.TILE, t[1], 1.0, source_id=1)])
    append_pairs(rig, 1, p1)
    b = compare("seq absolute", rig, [host_pack(sims[0], p0), host_pack(sims[1], p1)],
                [host_planes(s.gmap) for s in sims], rig.tick_row(), [1, 1])
    if b is not None:
        # the same edit at seq 0 draws differently: the absolute value matters
        p0z = [(noisy, 0)]
        pz = host_pack(sims[0], p0z)
        if np.array_equal(pz.edits[:, fe.E_KEY], host_pack(sims[0], p0).edits[:, fe.E_KEY]):
            fail("seq absolute: the key column does not carry the seq")
        print("  seq absolute: a noisy edit at seq 5 draws as the host's")
    # after the flush every seq is 0: rows keyed 0 continue it (the host queue's
    # next seq after take_ordered)
    q = fe.EditQueue()
    q.enqueue(noisy)
    pairs = list(q._edits)
    try:
        append_pairs(rig, 0, pairs)
    except edit_door.DoorInterleave as ex:
        fail(f"seq reset: a row keyed 0 does not continue the door after the flush ({ex})")
        return
    p2 = _pairs([E("fire", R.TILE, t[2], 0.4, source_id=2)])
    append_pairs(rig, 1, p2)
    b = compare("seq reset", rig, [host_pack(sims[0], pairs), host_pack(sims[1], p2)],
                [host_planes(s.gmap) for s in sims], rig.tick_row(), [1, 1])
    if b is not None:
        print("  seq reset: the flush resets every seq to 0")
    # reset_env: env 1 alone
    append_pairs(rig, 0, _pairs([noisy]))
    append_pairs(rig, 1, _pairs([noisy, noisy]))
    rig.door.reset_env(1)
    seq = rig.door.seq().tolist()
    rows = [r[0].shape[0] for r in rig.door.rows()]
    if seq != [1, 0] or rows != [1, 0]:
        fail(f"seq reset_env: seq {seq}, rows {rows} (want [1, 0], [1, 0])")
    else:
        print("  seq reset_env: env 1 reset alone")
    rig.door.begin()


# ---------------------------------------------------------------------------
# PART SCENES
# ---------------------------------------------------------------------------
class Tap:
    """Taps on the drivers' edit queues: every enqueue (with its seq) and the
    6b take (the window, the ordered list, the host pack, the planes and the
    tick row at that moment). Reads only."""

    def __init__(self, sims):
        self.sims = list(sims)
        self.pending = [[] for _ in self.sims]
        self.snap = [None] * len(self.sims)
        self.takes = [0] * len(self.sims)
        self._orig = (fe.EditQueue.enqueue, fe.EditQueue.take_ordered)
        tap = self
        orig_enq, orig_take = self._orig

        def enqueue(q, edit):
            seq = q._seq
            orig_enq(q, edit)
            e = tap.env_of(q)
            if e is not None:
                tap.pending[e].append((edit, seq))

        def take_ordered(q):
            e = tap.env_of(q)
            if e is None:
                return orig_take(q)
            window = list(q._edits)
            out = orig_take(q)
            sim = tap.sims[e]
            pack = None
            if out:
                pack, reason = fe.pack_for_device(sim.gmap, out, sim.draws)
                if pack is None:
                    raise AssertionError(f"scene: the host pack falls back ({reason})")
            tap.snap[e] = dict(pre=tap.pending[e], window=window, ordered=out, pack=pack,
                               planes=host_planes(sim.gmap), row=unit_draws.tick_row(sim))
            tap.pending[e] = []
            tap.takes[e] += 1
            return out
        fe.EditQueue.enqueue = enqueue
        fe.EditQueue.take_ordered = take_ordered

    def env_of(self, q):
        for e, s in enumerate(self.sims):
            if s.edit_queue is q:
                return e
        return None

    def close(self):
        fe.EditQueue.enqueue, fe.EditQueue.take_ordered = self._orig


def run_scene(name, drivers, ticks, census):
    sims = [d.sim for d in drivers]
    rig = Rig(sims)
    rig.door.begin()
    tap = Tap(sims)
    try:
        # rows enqueued before the first tick (setup) are pending, keyed from 0
        for k in range(ticks):
            for d in drivers:
                d.tick()
            if any(s is None for s in tap.snap) or any(t != k + 1 for t in tap.takes):
                fail(f"{name}: tick {k}: 6b took the queue {tap.takes} times")
                return
            # slots 1 to 5's rows join the window the door already holds
            for e, sn in enumerate(tap.snap):
                append_pairs(rig, e, sn["pre"])
            for e, (sn, (er, tr)) in enumerate(zip(tap.snap, rig.door.rows())):
                we, wt = (edit_door.door_rows(sims[e].gmap, sn["window"]) if sn["window"]
                          else (np.zeros((0, edit_door.EDIT_COLS), np.int64),
                                np.zeros((0, edit_door.TILE_COLS), np.int64)))
                if not (np.array_equal(er, we) and np.array_equal(tr, wt)):
                    fail(f"{name}: tick {k}: env {e}'s door is not the host window")
                    return
                q = fe.EditQueue()
                keys = [q._sort_key(p)[:3] for p in sn["window"]]
                n_post = len(sn["window"]) - len(sn["pre"])
                if sn["window"]:
                    census["ticks"] += 1
                    census["edits"] += len(sn["window"])
                    unsorted = ([s for _e, s in sn["ordered"]] !=
                                [s for _e, s in sn["window"]])
                    if unsorted and len(set(keys)) < len(keys):
                        census["unsorted_tied"] += 1
                    if any(ed.noise > 0 for ed, _s in sn["window"]):
                        census["noisy"] += 1
                    if n_post:
                        census["spanning"] += 1
                    census["max_tiles"] = max(census["max_tiles"],
                                              int(sn["pack"].tiles.shape[0]))
            b = compare(f"{name} tick {k}", rig, [sn["pack"] for sn in tap.snap],
                        [sn["planes"] for sn in tap.snap],
                        rig.tick_row(np.stack([sn["row"] for sn in tap.snap])),
                        [len(sn["window"]) for sn in tap.snap])
            if b is None and any(sn["window"] for sn in tap.snap):
                return
            # slots 7 to 9e's rows (a timed charge, a heat source): written after
            # the flush, they cross the tick boundary in the door
            for e in range(len(sims)):
                append_pairs(rig, e, tap.pending[e])
                tap.pending[e] = []
            tap.snap = [None] * len(sims)
    finally:
        tap.close()


def part_scenes():
    import cuda_unit_payloads_check as PC
    import cuda_unit_sprays_check as SC
    from _hazard_scene import HazardScene
    census = dict(ticks=0, edits=0, unsorted_tied=0, noisy=0, spanning=0, max_tiles=0)
    makers = (("spray", lambda v: SC.SprayDriver(v)),
              ("payload", lambda v: PC.PayloadScene(v)),
              ("hazard", lambda v: HazardScene(2 * v, bp=bp)))
    for name, make in makers:
        for n in (1, 2):
            t0 = time.perf_counter()
            n0 = len(FAILS)
            drivers = [make(v) for v in range(n)]
            if name == "payload":
                PC.share_tables([d.sim for d in drivers])
            run_scene(f"{name} N={n}", drivers, SCENE_TICKS, census)
            print(f"  {name} N={n}: {SCENE_TICKS} ticks "
                  f"{'equal' if len(FAILS) == n0 else 'DIFFERS'} "
                  f"({time.perf_counter() - t0:.0f} s)")
    print(f"  census: {census}")
    for key in ("ticks", "unsorted_tied", "noisy", "spanning"):
        if not census[key]:
            fail(f"scenes: no tick with {key} (the scenes do not reach the case)")


PARTS = ("sort", "range", "window", "seq", "scenes")


def main(parts=None) -> int:
    print(f"P7e.3 the edit door, step (c), HAS_CUDA={getattr(bp, 'HAS_CUDA', False)}")
    t0 = time.perf_counter()
    for name in PARTS:
        if parts and name not in parts:
            continue
        print(f"PART {name.upper()}")
        try:
            globals()[f"part_{name}"]()
        except Exception as ex:          # a crash is a failure, named
            import traceback
            traceback.print_exc()
            fail(f"{name}: {type(ex).__name__}: {ex}")
    print(f"({time.perf_counter() - t0:.0f} s)")
    print("EDIT_FLUSH_RESULT:", "FAIL" if FAILS else "PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
