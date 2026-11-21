"""THE DEVICE BOOKS (#29 P8.7, residency design §4.4c "P8 RE-CUT, VERSION
2", row P8.7; inventory ``docs/residency_p8_observation_inventory_2026-11-20.md``
parts E, H, I, K5; critique findings 3, 14, 21): the reward books of reward
profile v1 (``simulation.reward_v1.RewardBooks``), the env's old event
counters (``rl_env.count_events``), the served reward groups
(``rl_env.EnvRunner._reward``), termination (``MatchRules``) and the ENDED
LATCH, per env, on the card, updated ONCE per tick in the device conductor's
walk (the ``11'`` call line, after the device team picture whose bars they
read and after the observation's blocks of the tick). The host books stay THE
reference; this is their twin (never a second reward fold).

CLASS T (training only): nothing here is synced state, no digest reads it,
and its device code writes no synced buffer.

THE ORDER (``EnvRunner.step``'s loop body, line for line, one thread per env):
the step's accumulators zeroed on the first tick of a step (``start_step``);
``begin`` is the identity on the device (nothing changes a unit between two
conductor ticks: the snapshot of the previous tick's end IS the state at the
tick's start, and no team can be eliminated between ticks); the old counters
over the tick's event ring in emission order (float32 accumulation, numpy 2's
NEP 50 rule: the damage cast to float32, then a float32 add; the rows whose
records ended before the tick frozen); ``RewardBooks.end`` -- the walk over the
ring in emission order (float64, unfused, through the observation's float
kit), the turnings and the ``RewardBooksMismatch`` check, the snapshot, the
team part (the standing in ``sim.units`` = store row order, the teams' dict
order kept as the order a team first appears), the eliminations, the flush;
then termination (fewer than two marine teams alive or the limit tick), the
result and ``finish``; on a decision tick or the ending tick the SERVED
arrays.

THE FOLDS (numpy's order, inventory K5, measured on numpy 2.5.3 by the P8.7
experiment): the named profile's ``(counters[:, :, None] * G[None]).sum(axis=1)``
adds the K products of a row SEQUENTIALLY from k = 0 starting from +0.0 (0
mismatches in 900 random (U, K) cases; a sum of all -0.0 products is +0.0);
the ``"counters"`` profile's last-axis float32 sums are numpy's unrolled
pairwise sum (11 635 of 24 000 rows differ from a serial fold at K = 18), so
that profile is REFUSED BY NAME (design "P8 RE-CUT, VERSION 2", critique 3).

THE STATE (:func:`layout`): ONE ``(N, BK_W)`` int64 block per env (floats as
their own bit patterns): the unit snapshot (``_units``), the standing
(``parts``: H, A per team, the zombies' H, the teams' dict order), the living
and eliminated teams (bit sets), the rows' ended / pending / step-ended flags,
the tick's scratch and the step's counters (``_scr``, ``counters``, ``raw``),
the old counters (float32 bits), the per-team ``rest`` / ``d_balance`` /
``outcome``, the walk's working hp and life, and the scalars (finished, the
ending tick, the result, terminated / truncated, the tick, the env's team
count).

THE SERVED ARRAYS (:attr:`DeviceBooks.served`): exactly what the env serves
(``rl_env.Layout.shapes``): ``reward`` (N, U, 5) float32, ``counters`` (N, U,
K) float32, ``team_balance`` / ``team_uncredited`` / ``team_outcome`` (N, T)
float32, ``rows_ended`` (N, U) uint8, ``terminated`` / ``truncated`` (N,)
uint8; and the result per env (:attr:`DeviceBooks.result`: outcome code, team,
ticks).

THE ENDED LATCH (``DeviceObservation.ended``): set on the card at the tick an
env's match ends; from the next tick on every block skips the env and its
books are FROZEN (the host stops stepping an ended episode). The ending tick
is kept on the card (scalar ``ENDED_TICK``): the hook of the terminal
observation (P8.8b).

THE ONE READ: the books' fault words (a ``RewardBooksMismatch`` count, fault
bits) and the per-env "ended this tick" flags ride the conductor's ONE books
read (kind ``reward_books``): termination reaches the host with no third sync
(``DeviceConductor.ended_now``).
"""
from __future__ import annotations

import numpy as np

#: A unit's snapshot (``reward_v1._Unit``).
UNIT_FIELDS = ("HP_Q", "ALIVE", "TEAM", "ZOMBIE", "BAR_Q")
#: The scalars.
SCALAR_FIELDS = ("FINISHED", "ENDED_TICK", "RES_OUTCOME", "RES_TEAM", "RES_TICKS",
                 "TERMINATED", "TRUNCATED", "TICK", "N_MATCH", "TERM_SEEN")
#: The result's outcome codes (``MatchResult.outcome``; 0 while running).
OUTCOME_CODES = {"": 0, "win": 1, "draw": 2}
#: The most marine teams the books' standing holds (a refusal past it).
T_MAX = 8
#: The term order of the obs table's BOOK_TERMS rows (every named profile
#: has exactly these terms; refused otherwise).
TERM_NAMES = ("win", "eliminated", "clock", "marine_bar", "marine_alive", "zombie_bar",
              "ff_bar", "ff_kill")
#: The fault bits (the books read names each); the mismatch is its own count.
FAULT_BITS = (
    ("SKIPPED", "the books were not at the previous tick (a skipped or doubled update)"),
    ("TEAM", "a unit's team is not a marine team of the layout"),
)
#: THE BREAKS the check switches (``brk``): 1 the tick's events walked in
#: reverse order, 2 the books not frozen at the end, 3 termination decided
#: one tick late. (The weight break is the check's: the obs table's term rows
#: rolled by one.)
BREAK_CODES = {"": 0, "bk_order": 1, "bk_freeze": 2, "bk_late": 3}
#: The deferred words per env: the mismatch count, the fault bits, ended now.
WORDS = 3


def counter_names() -> tuple:
    """THE served counters (``rl_env.COUNTER_NAMES``: the old ones, then the
    books')."""
    import rl_env
    return tuple(rl_env.COUNTER_NAMES)


def n_old() -> int:
    import rl_env
    return int(rl_env.N_OLD_COUNTERS)


def layout(n_teams: int, n_units: int, n_rows: int) -> list:
    """``[(section, offset, shape)]`` of one env's block: THE layout the
    kernel computes (``cuda_obs_books.cu``'s ``bk_layout``; the build checks
    the compiled one against it)."""
    from simulation.reward_v1 import BOOK_COUNTERS
    T, U, R = int(n_teams), int(n_units), int(n_rows)
    nb, k0 = len(BOOK_COUNTERS), n_old()
    secs = [("UNIT", (U, len(UNIT_FIELDS))), ("H", (T,)), ("A", (T,)), ("HZ", (1,)),
            ("ORD", (T,)), ("ORD_N", (1,)), ("LIVING", (1,)), ("ELIM", (1,)),
            ("ENDED", (R,)), ("PEND", (R,)), ("ROWS_ENDED", (R,)),
            ("SCR", (R, nb)), ("CNT", (R, nb)), ("RAW", (R, nb)), ("OLD", (R, k0)),
            ("REST", (T,)), ("DBAL", (T,)), ("OUTC", (T,)),
            ("WORK_H", (U,)), ("WORK_AL", (U,)), ("SC", (len(SCALAR_FIELDS),))]
    out, off = [], 0
    for name, shape in secs:
        out.append((name, off, shape))
        off += int(np.prod(shape))
    return out


def width(n_teams: int, n_units: int, n_rows: int) -> int:
    name, off, shape = layout(n_teams, n_units, n_rows)[-1]
    return off + int(np.prod(shape))


def constants() -> list:
    """The books' generated constants (``obs_layout.h``, through
    ``device_obs._constants``): the counters' columns, the unit and scalar
    fields, the terms' order, the codes, faults and breaks."""
    from simulation.events import CAUSE_NAMES
    from simulation.reward_v1 import BOOK_COUNTERS, REWARD_GROUPS
    names = counter_names()
    k0 = n_old()
    out = [("OBS_BK_NB", len(BOOK_COUNTERS)), ("OBS_BK_K0", k0), ("OBS_BK_K", len(names)),
           ("OBS_BK_GROUPS", len(REWARD_GROUPS)), ("OBS_BK_TMAX", T_MAX),
           ("OBS_BK_N_TERMS", len(TERM_NAMES)), ("OBS_BK_WORDS", WORDS)]
    out += [(f"OBS_BK_C_{n.upper()}", i) for i, n in enumerate(BOOK_COUNTERS)]
    for n in ("hits_dealt", "damage_dealt", "team_hits", "kills", "team_kills",
              "hits_taken", "damage_taken"):
        out.append((f"OBS_BK_OLD_{n.upper()}", names.index(n)))
    out.append(("OBS_BK_OLD_DEATHS0", names.index(f"deaths_{CAUSE_NAMES[0]}")))
    out.append(("OBS_BK_N_CAUSES", len(CAUSE_NAMES)))
    out += [(f"OBS_BK_UNIT_{f}", i) for i, f in enumerate(UNIT_FIELDS)]
    out.append(("OBS_BK_UNIT_W", len(UNIT_FIELDS)))
    out += [(f"OBS_BK_S_{f}", i) for i, f in enumerate(SCALAR_FIELDS)]
    out.append(("OBS_BK_S_W", len(SCALAR_FIELDS)))
    out += [(f"OBS_BK_TERM_{n.upper()}", i) for i, n in enumerate(TERM_NAMES)]
    out += [(f"OBS_BK_OUTCOME_{k.upper()}", v) for k, v in OUTCOME_CODES.items() if k]
    out += [(f"OBS_BK_FAULT_{n}", 1 << i) for i, (n, _) in enumerate(FAULT_BITS)]
    out += [(f"OBS_BK_BREAK_{k.upper()}", v) for k, v in BREAK_CODES.items() if k]
    return out


class BooksRefused(RuntimeError):
    """A run the device books do not serve (by name)."""


def profile_terms(profile: str, overrides=None) -> dict:
    """The term table the device books read: a NAMED profile's
    (``reward_v1.profile_terms``, the run file's overrides applied); the
    ``"counters"`` profile REFUSED BY NAME (numpy sums its fold pairwise, a
    serial kernel cannot equal it: critique finding 3)."""
    from simulation.reward_v1 import COUNTERS_PROFILE
    from simulation.reward_v1 import profile_terms as _terms
    if profile == COUNTERS_PROFILE:
        raise BooksRefused(
            f"device books: the {COUNTERS_PROFILE!r} reward profile is refused on the device "
            f"backend (its float32 fold is numpy's pairwise sum, which a serial kernel cannot "
            f"equal); name a profile of reward_v1.PROFILES")
    terms = _terms(profile, overrides, where="device books")
    if tuple(sorted(terms)) != tuple(sorted(TERM_NAMES)):
        raise BooksRefused(f"device books: profile {profile!r}'s terms {sorted(terms)} are not "
                           f"the books' {sorted(TERM_NAMES)}")
    return terms


def table_rows(terms: dict) -> tuple:
    """The obs table's book rows from the reference's own objects: the terms
    (``TERM_NAMES`` order) and the group matrix (``reward_v1.group_weights``
    over ``rl_env.COUNTER_NAMES``, ``(K, 5)`` row-major), floats as their own
    float64 bit patterns."""
    from simulation.reward_v1 import group_weights
    t = np.array([float(terms[n]) for n in TERM_NAMES], dtype=np.float64)
    g = group_weights(terms, counter_names()).astype(np.float64)
    return t.view(np.int64).copy(), g.reshape(-1).view(np.int64).copy()


def fault_text(bits: int) -> str:
    return "; ".join(f"{n}: {why}" for i, (n, why) in enumerate(FAULT_BITS)
                     if int(bits) & (1 << i))


# ---------------------------------------------------------------------------
# THE NORMAL FORM: the books' whole state as plain values (floats as their
# bits), keyed by unit id, so the host books and the device block compare
# field by field (the gate)
# ---------------------------------------------------------------------------
def _bits64(a) -> list:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64)).view(np.uint64).tolist()


def host_form(books, old) -> dict:
    """The REFERENCE books' normal form: ``books`` a ``RewardBooks`` between
    two ticks (its scratch empty), ``old`` the env's old counters of the step
    (``(U, K0)`` float32)."""
    if books._scr.any() or books._pending_end.any():
        raise BooksRefused("device books: the host books hold a tick's scratch")
    p = books.parts
    return {
        "units": {int(u): (int(s.hp_q), int(bool(s.alive)), int(s.team), int(bool(s.zombie)),
                           int(s.bar_q)) for u, s in books._units.items()},
        "parts": [(int(t), _bits64([h])[0], int(p.A[t])) for t, h in p.H.items()],
        "hz": _bits64([p.H_zombie])[0],
        "living": sorted(int(t) for t in books.living),
        "eliminated": sorted(int(t) for t in books._eliminated),
        "ended": np.flatnonzero(books.ended).tolist(),
        "rows_ended": np.flatnonzero(books.rows_ended).tolist(),
        "counters": [_bits64(r) for r in books.counters],
        "raw": [_bits64(r) for r in books.raw],
        "old": np.ascontiguousarray(np.asarray(old, dtype=np.float32)).view(np.uint32).tolist(),
        "rest": _bits64(books.rest),
        "d_balance": _bits64(books.d_balance),
        "outcome": _bits64(books.outcome),
        "finished": bool(books.finished),
    }


def device_form(block: np.ndarray, uids, T: int, U: int, R: int) -> dict:
    """One env's DEVICE block (downloaded) as the normal form; ``uids``: the
    store's rows' unit ids (row order)."""
    sec = {n: block[o:o + int(np.prod(s))].reshape(s) for n, o, s in layout(T, U, R)}
    Q = {f: i for i, f in enumerate(UNIT_FIELDS)}
    S = {f: i for i, f in enumerate(SCALAR_FIELDS)}
    units = {int(uids[r]): tuple(int(sec["UNIT"][r, Q[f]]) for f in UNIT_FIELDS)
             for r in range(len(uids))}
    order = [int(t) for t in sec["ORD"][:int(sec["ORD_N"][0])]]
    u64 = lambda a: np.ascontiguousarray(a).view(np.uint64)  # noqa: E731
    living = [t for t in range(T) if int(sec["LIVING"][0]) >> t & 1]
    elim = [t for t in range(T) if int(sec["ELIM"][0]) >> t & 1]
    return {
        "units": units,
        "parts": [(t, int(u64(sec["H"][t:t + 1])[0]), int(sec["A"][t])) for t in order],
        "hz": int(u64(sec["HZ"])[0]),
        "living": living,
        "eliminated": elim,
        "ended": np.flatnonzero(sec["ENDED"]).tolist(),
        "rows_ended": np.flatnonzero(sec["ROWS_ENDED"]).tolist(),
        "counters": [u64(r).tolist() for r in sec["CNT"]],
        "raw": [u64(r).tolist() for r in sec["RAW"]],
        "old": (sec["OLD"] & 0xffffffff).astype(np.uint32).tolist(),
        "rest": u64(sec["REST"]).tolist(),
        "d_balance": u64(sec["DBAL"]).tolist(),
        "outcome": u64(sec["OUTC"]).tolist(),
        "finished": bool(sec["SC"][S["FINISHED"]]),
    }


# ---------------------------------------------------------------------------
# The device books
# ---------------------------------------------------------------------------
class DeviceBooks:
    """The reward books of a device observation's N envs (see the module
    docstring). Built by ``DeviceObservation`` (its ``reward_profile``);
    :meth:`start` takes each env's first state on the card (the
    ``RewardBooks`` constructor: the snapshot, the standing, the living
    teams)."""

    def __init__(self, obs, terms: dict):
        import cupy as cp
        c = obs.c
        self.obs, self.c, self.cp, self.bp = obs, c, cp, c.bp
        self.N, self.T, self.U, self.R = int(obs.N), int(obs.T), int(c.U), int(obs.n_rows)
        if self.T > T_MAX:
            raise BooksRefused(f"device books: {self.T} marine teams, the books hold {T_MAX}")
        self.terms = dict(terms)
        self.W = width(self.T, self.U, self.R)
        self.layout = {n: (o, s) for n, o, s in layout(self.T, self.U, self.R)}
        compiled = dict(self.bp.obs_books_layout(self.T, self.U, self.R))
        mine = {n: int(o) for n, (o, _s) in self.layout.items()}
        mine["W"] = self.W
        if compiled != mine:
            raise BooksRefused(f"device books: the compiled block layout {compiled} is not "
                               f"device_books.layout's {mine} (rebuild cpp/build_cuda)")
        self.K = len(counter_names())
        block = np.zeros((self.N, self.W), dtype=np.int64)
        o = self.layout["SC"][0]
        S = {f: i for i, f in enumerate(SCALAR_FIELDS)}
        block[:, o + S["ENDED_TICK"]] = -1
        block[:, o + S["RES_TEAM"]] = -1
        block[:, o + S["TICK"]] = -1
        block[:, o + S["N_MATCH"]] = obs.env_teams
        self.state = cp.asarray(block)
        N, R, T = self.N, self.R, self.T
        from simulation.reward_v1 import REWARD_GROUPS
        self.served = {
            "reward": cp.zeros((N, R, len(REWARD_GROUPS)), dtype=cp.float32),
            "counters": cp.zeros((N, R, self.K), dtype=cp.float32),
            "team_balance": cp.zeros((N, T), dtype=cp.float32),
            "team_uncredited": cp.zeros((N, T), dtype=cp.float32),
            "team_outcome": cp.zeros((N, T), dtype=cp.float32),
            "rows_ended": cp.zeros((N, R), dtype=cp.uint8),
            "terminated": cp.zeros((N,), dtype=cp.uint8),
            "truncated": cp.zeros((N,), dtype=cp.uint8),
        }
        self.words = cp.zeros((WORDS, N), dtype=cp.int64)
        self.brk = 0                          # THE BREAKS (BREAK_CODES), the check's
        self.updates = 0

    def section(self, name: str):
        o, s = self.layout[name]
        n = int(np.prod(s))
        return self.state[:, o:o + n].reshape((self.N,) + tuple(s))

    def _launch(self, first: bool) -> bool:
        o, c, sv = self.obs, self.c, self.served
        pic = o.picture
        return self.bp.obs_books(
            int(self.N), int(c.store.device_ptr), int(self.U), int(c.store.shape[2]),
            c.ring.buffer, int(o.rowmap.data.ptr), int(self.R), int(o.S), int(self.T),
            int(o.table.data.ptr), int(o.table.shape[1]), int(c.trow.ptr),
            int(o.params.data.ptr), int(o.ended.data.ptr), int(pic.state.data.ptr), int(pic.W),
            int(self.state.data.ptr), int(self.W), int(sv["reward"].data.ptr),
            int(sv["counters"].data.ptr), int(sv["team_balance"].data.ptr),
            int(sv["team_uncredited"].data.ptr), int(sv["team_outcome"].data.ptr),
            int(sv["rows_ended"].data.ptr), int(sv["terminated"].data.ptr),
            int(sv["truncated"].data.ptr), int(self.words.data.ptr), bool(first),
            int(self.brk))

    def start(self) -> None:
        """Every env's first state (the books' constructor), at the build."""
        deferred = self._launch(True)
        if not deferred:
            self.check_words(self.words.get())

    def update(self) -> None:
        """THE stage (the ``11'`` call line, after the picture and the
        blocks): every env's books brought to the tick. Inside a conductor
        tick the words ride the books read; outside it they are read here."""
        deferred = self._launch(False)
        self.updates += 1
        if not deferred:
            self.check_words(self.words.get())

    def check_words(self, w) -> None:
        """Raise by name what the deferred words name (the conductor's books
        read calls this with the words it downloaded)."""
        from simulation.reward_v1 import RewardBooksMismatch
        w = np.asarray(w, dtype=np.int64).reshape(WORDS, self.N)
        if w[0].any():
            e = int(np.flatnonzero(w[0])[0])
            ex = RewardBooksMismatch(
                f"device books (env {e}, tick {int(self.c.sims[e].tick)}): {int(w[0, e])} "
                f"unit(s) whose events do not reproduce their hp or life (an hp or life write "
                f"outside apply_packet or an event with no packet)")
            ex.env = e
            raise ex
        if w[1].any():
            from simulation.conductor import ConductorFault
            e = int(np.flatnonzero(w[1])[0])
            ex = ConductorFault(f"device books (env {e}, tick {int(self.c.sims[e].tick)}): "
                                f"{fault_text(int(w[1, e]))}")
            ex.env = e
            raise ex

    # -- readers (the gate's downloads: a sync) ------------------------------
    def forms(self) -> list:
        blk = self.state.get()
        return [device_form(blk[e], self.c.uids[e], self.T, self.U, self.R)
                for e in range(self.N)]

    def scalars(self) -> np.ndarray:
        """``(N, len(SCALAR_FIELDS))`` int64: finished, the ending tick, the
        result (outcome code, team, ticks), terminated, truncated, ..."""
        return self.section("SC").get()

    def host_served(self) -> dict:
        return {k: v.get() for k, v in self.served.items()}


__all__ = ["UNIT_FIELDS", "SCALAR_FIELDS", "OUTCOME_CODES", "T_MAX", "TERM_NAMES",
           "FAULT_BITS", "BREAK_CODES", "WORDS", "counter_names", "n_old", "layout", "width",
           "constants", "BooksRefused", "profile_terms", "table_rows", "fault_text",
           "host_form", "device_form", "DeviceBooks"]
