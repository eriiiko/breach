"""agents/records.py -- THE reader of the bots' records and the loader that
deals their decisions to the copy (rung 2 note 2e, sections 2 to 4; design
3.1).

THE RECORDS are patch 1j's imitation records: one ``.npz`` per match and
door team, written by ``tools/bot_tournament.py --records`` through
``control_bots_door.RecordSink`` and read here through the game's own
``control_bots_door.read_record`` (which refuses another record format).
A file holds, per decision of the team (the env's decision ticks, in
order), every roster slot's served observation, the four masks, the
argument's mask, the five integers sent, the translation's loss bits, the
fireteam rewards and the two row flags; ``meta`` ties it to its match
(level, seed, the match spec, the spec names). Files stay PACKED on disk:
whole files are unpacked in memory by the loader's processes, never to
disk.

WHAT FITS (:func:`check_record`): a record is refused by name
(:class:`RecordRefused`) unless its observation and action spec names are
today's, its grenade rows (count and names) are today's, every array has
today's trailing shape and number format for its team size, and every kind
value of its living decisions lies inside the network's embedding rows. A
record that does not fit is never converted or padded: the records are
re-made (note 2: "the guard for the day the observation changes again").

THE CENSUS (:func:`census`): every file of a folder read once -- matches,
files and living-soldier decisions per ship, the fit verdict per file, and
one hash over the files' names and bytes (the census hash a copy's export
names).

THE SPLIT (:func:`side`): by level and seed, never by file -- every file of
one (level, seed) falls on one side; held back is every seed with
``seed % HELD_EVERY == 0``.

THE LOADER, two modes, both a function of one seed and the pass number:

- :class:`DecisionDealer` (the copy without memory): processes take whole
  files from a shuffled list and unpack them; the dealer keeps a shuffle
  buffer of about ``buffer_matches`` files and deals batches of single
  living decisions drawn at random from it. One pass deals every living
  decision of its files exactly once, in near-equal batches.
- :class:`StretchDealer` (the copy with memory, design 3.2): ``lanes``
  soldiers at once, each a whole match of one soldier in order from its
  first decision, cut into stretches by THE stretch rule
  (``agents.learner.stretches``), the next stretch of every lane dealt
  together; when a soldier's match ends its lane takes the next soldier
  (drawn at random from the buffer). A stretch never starts in mid-match
  from an empty memory: a lane's memory is carried by the trainer, zeroed
  only at a soldier's first decision (``first``).

Only the decisions of LIVING soldiers (the records' own ``valid`` flag) are
dealt; a soldier's living decisions are the first of its match (the census
counts any that are not, and the loader refuses such a soldier by name).

No torch here: the loader's processes import numpy and the game's tables
only.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agents import _game  # noqa: F401  (the game's Python on the path)
from simulation import action_spec_v1 as A
from simulation import obs_spec_v1 as O

#: The split: a (level, seed) is held back iff ``seed % HELD_EVERY == 0``.
HELD_EVERY = 5
#: The served keys a dealt decision carries (the soldier's input and the
#: masks), then the record's own per-decision arrays.
OBS_KEYS = ("vector", "list", "list_mask", "near_f", "near_u", "far_f",
            "far_u") + A.MASK_KEYS
DECISION_KEYS = OBS_KEYS + ("list_uid", "arg_mask", "actions", "loss")
#: A record file's name: ``<c0>-<c1>__z<zombies>__s<seed>__t<team>.npz``.
_NAME = re.compile(r"^(?P<match>.+__z\d+__s(?P<seed>\d+))__t(?P<team>\d+)"
                   r"\.npz$")


class RecordRefused(ValueError):
    """A record does not fit today's observation or action layout (the
    message names the file and every field)."""


# ---------------------------------------------------------------------------
# reading and fitting
# ---------------------------------------------------------------------------
def read(path) -> tuple:
    """``(arrays, meta)`` of one record (the game's reader: another record
    format is refused there)."""
    import control_bots_door as D
    return D.read_record(path)


def read_meta(path) -> dict:
    """``meta`` alone (a packed member read without the arrays)."""
    import control_bots_door as D
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"][()]))
    if meta.get("record_format") != D.RECORD_FORMAT:
        raise RecordRefused(f"{path}: record_format "
                            f"{meta.get('record_format')!r}, the game's reader "
                            f"reads {D.RECORD_FORMAT}")
    return meta


def expected(S: int, K: int) -> dict:
    """Today's ``key -> (trailing shape, dtype)`` of a record of team size
    ``S`` with ``K`` grenade rows, from the game's own shape tables."""
    E = int(S) + O.ENTRIES_BEYOND_SLOTS
    out = {}
    for k, (shape, dt) in O.shapes(1, S, S).items():
        out[k] = (tuple(shape[1:]), np.dtype(dt))
    for k, (shape, dt) in A.mask_shapes(1, S, E, K).items():
        out[k] = (tuple(shape[1:]), np.dtype(dt))
    n_heads = len(A.HEADS)
    out["arg_mask"] = ((S, A.N_ARG), np.dtype(np.uint8))
    out["actions"] = ((S, n_heads), np.dtype(np.int16))
    out["loss"] = ((S,), np.dtype(np.uint16))
    return out


def _today() -> dict:
    from simulation.weapons import get_tables
    rows = get_tables().hand_grenade_rows
    return {"obs_spec": O.SPEC_NAME, "action_spec": A.ACTION_SPEC_NAME,
            "K": len(rows), "hand_names": list(A.hand_names(rows))}


def check_meta(meta: dict, where: str) -> None:
    """Refuse (:class:`RecordRefused`, naming each field) a record whose
    spec names or grenade rows are not today's."""
    want = _today()
    bad = [f"{k}: record {meta.get(k)!r}, today {v!r}"
           for k, v in want.items() if meta.get(k) != v]
    E = int(meta.get("S", -1)) + O.ENTRIES_BEYOND_SLOTS
    if meta.get("E") != E:
        bad.append(f"E: record {meta.get('E')!r}, today S + "
                   f"{O.ENTRIES_BEYOND_SLOTS} = {E}")
    if bad:
        raise RecordRefused(f"{where} does not fit: " + "; ".join(bad))


def check_record(arrays: dict, meta: dict, where: str, sizes=None) -> None:
    """Refuse a record that does not fit today's layout (module text): the
    spec names and grenade rows (:func:`check_meta`), every array's
    trailing shape and number format, and every kind value of a living
    decision inside the network's embedding rows (``sizes``, an
    ``agents.sizes.Sizes``; default: today's for the record's ``S``)."""
    check_meta(meta, where)
    S, K = int(meta["S"]), int(meta["K"])
    bad = []
    for k, (tail, dt) in expected(S, K).items():
        a = arrays.get(k)
        if a is None:
            bad.append(f"{k}: missing")
            continue
        if tuple(a.shape[1:]) != tail or a.dtype != dt:
            bad.append(f"{k}: record {tuple(a.shape[1:])} {a.dtype}, today "
                       f"{tail} {dt}")
    if bad:
        raise RecordRefused(f"{where} does not fit: " + "; ".join(bad))
    if sizes is None:
        from agents.sizes import sizes_for
        sizes = sizes_for(S, n_kinds=K)
    from simulation.factions import ZOMBIE_TEAM
    live = arrays["valid"] != 0
    for table, cols in (("vector", sizes.vector_kinds),
                        ("list", sizes.entry_kinds)):
        x = arrays[table][live]
        if table == "list":
            x = x[arrays["list_mask"][live] != 0]
        for c in cols:
            v = x[..., c.pos]
            if c.zombie_row >= 0:
                v = v[v != ZOMBIE_TEAM]
            if v.size == 0:
                continue
            lo, hi = float(v.min()), float(v.max())
            if lo < -1 or hi + 1 >= c.rows or not np.all(v == np.round(v)):
                bad.append(f"{table} kind {c.name!r} (col {c.pos}): values "
                           f"{lo:g}..{hi:g}, the network has rows for -1.."
                           f"{c.rows - 2}")
    if bad:
        raise RecordRefused(f"{where} does not fit: " + "; ".join(bad))


# ---------------------------------------------------------------------------
# the census and the split
# ---------------------------------------------------------------------------
def side(level: str, seed: int) -> str:
    """THE split (note 4): ``"held"`` for every (level, seed) with ``seed %
    HELD_EVERY == 0``, else ``"train"``. A function of the match, never of
    the file: both teams' files, with and without the horde, fall together."""
    del level                          # every ship is split the same way
    return "held" if int(seed) % HELD_EVERY == 0 else "train"


@dataclass(frozen=True)
class RecordFile:
    """One record file as the census saw it."""
    path: str
    level: str
    seed: int
    match: str                # the match's name in its level folder
    team: int
    decisions: int
    living: int               # living-soldier decisions
    slots: tuple              # per roster slot, its living decisions
    side: str


def _prefix_ok(valid) -> bool:
    """Every slot's living decisions are the first of the match."""
    v = np.asarray(valid) != 0
    n = v.sum(axis=0)
    return bool(all(v[:k, s].all() for s, k in enumerate(n)))


def _census_one(path: str) -> dict:
    """One file's census row (run in a loader process)."""
    p = Path(path)
    data = p.read_bytes()
    out = {"path": str(p), "bytes": len(data),
           "sha256": hashlib.sha256(data).hexdigest()}
    m = _NAME.match(p.name)
    try:
        arrays, meta = read(p)
        where = f"{p.parent.name}/{p.name}"
        check_record(arrays, meta, where)
        valid = arrays["valid"]
        out.update(fits=True, level=meta["level_id"], seed=int(meta["seed"]),
                   team=int(meta["team"]), match=m.group("match") if m else
                   p.stem, decisions=int(valid.shape[0]),
                   living=int((valid != 0).sum()),
                   slots=[int(x) for x in (valid != 0).sum(axis=0)],
                   prefix=_prefix_ok(valid),
                   controllers=list(meta["controllers"]),
                   zombies=int(meta["spec"]["zombies"]),
                   build=meta.get("build"), S=int(meta["S"]),
                   E=int(meta["E"]))
        if m and int(m.group("seed")) != out["seed"]:
            out.update(fits=False, refused=f"{where}: name seed "
                       f"{m.group('seed')} against meta seed {out['seed']}")
    except (RecordRefused, ValueError, KeyError) as e:
        out.update(fits=False, refused=str(e))
    return out


def _pool(workers: int):
    return ProcessPoolExecutor(max_workers=int(workers),
                               mp_context=mp.get_context("spawn"))


def census(folder, workers: int = 4) -> dict:
    """Every ``.npz`` under ``folder`` (one level folder per ship) read once
    (module text). Returns the census: per ship the files, matches, seeds
    and living decisions on each side, the refusals, the soldiers whose
    living decisions are not a prefix, and ``hash`` (one sha256 over every
    file's name relative to ``folder``, size and own sha256, in sorted
    order)."""
    root = Path(folder)
    paths = sorted(str(p) for p in root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no record files under {root}")
    if workers <= 1:
        rows = [_census_one(p) for p in paths]
    else:
        with _pool(workers) as ex:
            rows = list(ex.map(_census_one, paths, chunksize=4))
    h = hashlib.sha256()
    for r in rows:
        rel = Path(r["path"]).relative_to(root).as_posix()
        h.update(f"{rel}\0{r['bytes']}\0{r['sha256']}\n".encode())
    ships, refusals, files = {}, [], []
    for r in rows:
        if not r["fits"]:
            refusals.append(r["refused"])
            continue
        sd = side(r["level"], r["seed"])
        files.append(RecordFile(
            path=r["path"], level=r["level"], seed=r["seed"],
            match=r["match"], team=r["team"], decisions=r["decisions"],
            living=r["living"], slots=tuple(r["slots"]), side=sd))
        s = ships.setdefault(r["level"], {
            "files": 0, "matches": set(), "seeds": set(), "decisions": 0,
            "slot_decisions": 0, "living": {"train": 0, "held": 0},
            "matches_by_side": {"train": set(), "held": set()},
            "kinds": {}, "not_prefix": 0, "bytes": 0, "builds": set()})
        s["files"] += 1
        s["matches"].add(r["match"])
        s["matches_by_side"][sd].add(r["match"])
        s["seeds"].add(r["seed"])
        s["decisions"] += r["decisions"]
        s["slot_decisions"] += r["decisions"] * r["S"]
        s["living"][sd] += r["living"]
        kind = r["match"].split("__s")[0]
        s["kinds"][kind] = s["kinds"].get(kind, 0) + 1
        s["not_prefix"] += 0 if r["prefix"] else 1
        s["bytes"] += r["bytes"]
        s["builds"].add(r["build"])
    for s in ships.values():
        s["matches"] = len(s["matches"])
        s["matches_by_side"] = {k: len(v) for k, v in
                                s["matches_by_side"].items()}
        seeds = sorted(s["seeds"])
        s["seeds"] = {"n": len(seeds), "held": [x for x in seeds
                                                if x % HELD_EVERY == 0],
                      "min": seeds[0], "max": seeds[-1]}
        s["builds"] = sorted(map(str, s["builds"]))
    return {"folder": str(root), "files": len(paths), "fit": len(files),
            "refused": refusals, "ships": ships, "hash": h.hexdigest(),
            "today": _today(), "records": files}


def census_json(c: dict) -> dict:
    """The census as plain JSON (the files listed by name and count)."""
    out = {k: v for k, v in c.items() if k != "records"}
    out["records"] = [{"path": f.path, "level": f.level, "seed": f.seed,
                       "team": f.team, "living": f.living, "side": f.side}
                      for f in c["records"]]
    return out


def split(records) -> dict:
    """``{"train": [...], "held": [...]}`` of :class:`RecordFile` by
    :func:`side`."""
    out = {"train": [], "held": []}
    for f in records:
        out[side(f.level, f.seed)].append(f)
    return out


# ---------------------------------------------------------------------------
# unpacking one file (in a loader process)
# ---------------------------------------------------------------------------
def unpack(path: str) -> dict:
    """One file's living decisions (module text), checked: ``{key: (n,
    ...)}`` for :data:`DECISION_KEYS` in the served formats, plus ``t``
    (the decision's index in its match), ``slot`` and ``first`` (the
    soldier's first decision of the match). Refuses a soldier whose living
    decisions are not the first of its match."""
    arrays, meta = read(path)
    where = f"{Path(path).parent.name}/{Path(path).name}"
    check_record(arrays, meta, where)
    valid = arrays["valid"] != 0
    if not _prefix_ok(valid):
        raise RecordRefused(f"{where}: a soldier's living decisions are not "
                            f"the first of its match")
    # slot-major: a soldier's decisions together, in time order
    s_idx, t_idx = np.nonzero(valid.T)
    out = {k: np.ascontiguousarray(arrays[k][t_idx, s_idx])
           for k in DECISION_KEYS}
    out["t"] = t_idx.astype(np.int32)
    out["slot"] = s_idx.astype(np.int8)
    out["first"] = t_idx == 0
    return out


# ---------------------------------------------------------------------------
# the loader
# ---------------------------------------------------------------------------
class _Files:
    """The files of one pass, unpacked by ``workers`` processes ahead of
    the reader IN THE PASS'S ORDER (so the order is the seed's whatever
    process finishes first), at most ``ahead`` files in flight."""

    def __init__(self, paths, workers: int, ahead: int):
        self.paths = list(paths)
        self.workers = int(workers)
        self.ahead = max(1, int(ahead))
        self.ex = _pool(workers) if self.workers > 1 else None
        self.q = deque()
        self.i = 0
        self.wait = 0.0             # seconds the reader waited for a file

    def _fill(self):
        while self.ex is not None and self.i < len(self.paths) \
                and len(self.q) < self.ahead:
            self.q.append(self.ex.submit(unpack, self.paths[self.i]))
            self.i += 1

    def next(self):
        """The next file's decisions, or None at the end of the pass."""
        import time
        if self.ex is None:
            if self.i >= len(self.paths):
                return None
            self.i += 1
            return unpack(self.paths[self.i - 1])
        self._fill()
        if not self.q:
            return None
        t = time.perf_counter()
        out = self.q.popleft().result()
        self.wait += time.perf_counter() - t
        self._fill()
        return out

    def close(self):
        if self.ex is not None:
            for f in self.q:
                f.cancel()
            self.ex.shutdown(wait=True, cancel_futures=True)
            self.ex = None


def batch_sizes(total: int, batch: int) -> list:
    """A pass of ``total`` decisions cut into the fewest near-equal batches
    of at most ``batch`` (never a short last one: the learner's rule for
    its pieces, ``agents.learner.cut_pieces``)."""
    if total <= 0:
        return []
    n = -(-int(total) // int(batch))
    return [len(x) for x in np.array_split(np.arange(total), n)]


def _order(records, seed: int, p: int) -> list:
    rng = np.random.default_rng([int(seed), int(p), 0])
    paths = sorted(f.path for f in records)
    return [paths[i] for i in rng.permutation(len(paths))]


class DecisionDealer:
    """Batches of single living decisions (module text). ``records`` are
    :class:`RecordFile` (their ``living`` counts plan the batches)."""

    def __init__(self, records, *, seed: int, batch: int,
                 buffer_matches: int = 30, workers: int = 4):
        self.records = list(records)
        self.seed, self.batch = int(seed), int(batch)
        self.buffer_matches = int(buffer_matches)
        self.workers = int(workers)
        self.total = int(sum(f.living for f in self.records))
        self.wait = 0.0

    def batches(self, p: int):
        """Pass ``p``: yields ``dict`` batches (``(B, ...)`` arrays of
        :data:`DECISION_KEYS`, ``t``, ``slot``, ``first``); every living
        decision of the records exactly once."""
        rng = np.random.default_rng([self.seed, int(p), 1])
        files = _Files(_order(self.records, self.seed, p), self.workers,
                       2 * self.workers)
        buf = {}                    # id -> arrays
        left = {}                   # id -> undealt count
        pool_m = np.zeros(0, np.int64)
        pool_i = np.zeros(0, np.int64)
        next_id = 0
        try:
            for size in batch_sizes(self.total, self.batch):
                while len(left) < self.buffer_matches or len(pool_m) < size:
                    d = files.next()
                    if d is None:
                        break
                    n = len(d["t"])
                    if n == 0:
                        continue
                    buf[next_id] = d
                    left[next_id] = n
                    pool_m = np.concatenate([pool_m,
                                             np.full(n, next_id, np.int64)])
                    pool_i = np.concatenate([pool_i, np.arange(n)])
                    next_id += 1
                if len(pool_m) < size:
                    raise RuntimeError(
                        f"the records dealt {len(pool_m)} decisions where "
                        f"the census counted {size} more (a file changed "
                        f"since the census)")
                pick = np.sort(rng.choice(len(pool_m), size, replace=False))
                bm, bi = pool_m[pick], pool_i[pick]
                keep = np.ones(len(pool_m), bool)
                keep[pick] = False
                pool_m, pool_i = pool_m[keep], pool_i[keep]
                out = {}
                for k in buf[int(bm[0])]:
                    out[k] = np.concatenate([buf[int(m)][k][bi[bm == m]]
                                             for m in np.unique(bm)])
                for m, c in zip(*np.unique(bm, return_counts=True)):
                    left[int(m)] -= int(c)
                    if left[int(m)] == 0:
                        del left[int(m)]
                        del buf[int(m)]
                yield out
            if files.next() is not None or len(pool_m):
                raise RuntimeError("the records hold more living decisions "
                                   "than the census counted")
        finally:
            self.wait += files.wait
            files.close()


@dataclass
class _Lane:
    data: dict | None = None        # one soldier's decisions, in order
    cuts: list | None = None        # its stretches (start, length)
    k: int = 0                      # the next stretch


class StretchDealer:
    """The memory copy's stretches (module text): ``lanes`` soldiers at a
    time, each dealt its next stretch of ``stretch`` decisions. Every
    batch is ``(stretch, lanes, ...)`` arrays (time first, as
    ``forward_sequence`` reads them) with ``first`` (``(stretch, lanes)``:
    a soldier's first decision), ``kept`` (``(stretch, lanes)``: the steps
    that belong to the stretch, not its padding) and ``length``
    (``(lanes,)``, 0 for a lane with nothing left)."""

    def __init__(self, records, *, seed: int, lanes: int, stretch: int,
                 buffer_matches: int = 30, workers: int = 4):
        self.records = list(records)
        self.seed, self.lanes, self.stretch = int(seed), int(lanes), \
            int(stretch)
        self.buffer_matches = int(buffer_matches)
        self.workers = int(workers)
        self.total = int(sum(f.living for f in self.records))
        self.wait = 0.0

    def _soldiers(self, d: dict) -> list:
        from agents.learner import stretches
        out = []
        for s in np.unique(d["slot"]):
            idx = np.flatnonzero(d["slot"] == s)
            sol = {k: v[idx] for k, v in d.items()}
            if not sol["first"][0]:
                raise RecordRefused("a soldier's match does not start at its "
                                    "first decision")
            first = sol["first"][:, None].astype(np.uint8)
            start, _, length = stretches(first, self.stretch)
            out.append((sol, list(zip(start.tolist(), length.tolist()))))
        return out

    def batches(self, p: int):
        """Pass ``p``: yields stretch batches until every living decision of
        the records was dealt once."""
        rng = np.random.default_rng([self.seed, int(p), 2])
        files = _Files(_order(self.records, self.seed, p), self.workers,
                       2 * self.workers)
        waiting = []                # soldiers not yet in a lane
        lanes = [_Lane() for _ in range(self.lanes)]
        n_files = 0
        S = self.stretch
        try:
            while True:
                while n_files < len(self.records) and \
                        len(waiting) < self.buffer_matches * 4:
                    d = files.next()
                    if d is None:
                        break
                    n_files += 1
                    if len(d["t"]):
                        waiting.extend(self._soldiers(d))
                for ln in lanes:
                    if ln.data is None or ln.k >= len(ln.cuts):
                        if waiting:
                            j = int(rng.integers(len(waiting)))
                            waiting[j], waiting[-1] = waiting[-1], waiting[j]
                            ln.data, ln.cuts = waiting.pop()
                            ln.k = 0
                        else:
                            ln.data = None
                active = [ln for ln in lanes if ln.data is not None]
                if not active:
                    break
                ref = active[0].data
                out = {k: np.zeros((S, self.lanes) + ref[k].shape[1:],
                                   ref[k].dtype) for k in ref}
                kept = np.zeros((S, self.lanes), bool)
                length = np.zeros(self.lanes, np.int64)
                for j, ln in enumerate(lanes):
                    src = ln if ln.data is not None else active[0]
                    a, n = src.cuts[min(src.k, len(src.cuts) - 1)]
                    t = np.minimum(a + np.arange(S), a + n - 1)
                    for k in out:
                        out[k][:, j] = src.data[k][t]
                    if ln.data is not None:
                        kept[:n, j] = True
                        length[j] = n
                        ln.k += 1
                    else:
                        out["first"][:, j] = False
                out["kept"] = kept
                out["length"] = length
                yield out
            if files.next() is not None:
                raise RuntimeError("records left after the lanes ran dry")
        finally:
            self.wait += files.wait
            files.close()


def decisions_of(records, workers: int = 4, limit_files: int | None = None):
    """Every file's living decisions, file by file in the records' sorted
    order (for measuring): yields ``(RecordFile, decisions)``."""
    recs = sorted(records, key=lambda f: f.path)[:limit_files]
    files = _Files([f.path for f in recs], workers, 2 * max(1, workers))
    try:
        for f in recs:
            yield f, files.next()
    finally:
        files.close()


__all__ = ["HELD_EVERY", "OBS_KEYS", "DECISION_KEYS", "RecordRefused",
           "read", "read_meta", "expected", "check_meta", "check_record",
           "side", "RecordFile", "census", "census_json", "split", "unpack",
           "batch_sizes", "DecisionDealer", "StretchDealer", "decisions_of"]
