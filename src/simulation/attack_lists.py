"""THE ATTACK SLOT'S PER-ENV LISTS beside the effect list (#29 P7a.3;
``docs/residency_batching_design_2026-11-07.md`` §4.4c decisions 1 and 4).

Two lists, each an instance of the record list (``cpp/src/cuda_unit_events.h``'s
``UnitRecordList``, bound as ``breach_physics.UnitRecordListBuffer``: never a
second buffer mechanism), per env per tick, with a capacity: past it nothing
is written, the env's dropped count rises and the Python face refuses the run
naming the list and the env (:class:`ListOverflow`).

- **THE CELL BUFFER** (:data:`CELL_W` = 3: ``(y, x, div)``): the tiles slot 4b's
  phase A decided for each sprayer's cone (``combat.spray_cone_tiles``, against
  the walls standing at the slot's start), in the cone's row-major order, one
  span per sprayer; a SPRAY_CELLS record of the effect list
  (``simulation.applier``) holds its span. The walk reads the span and never
  recomputes a cone. Begun empty with the effect list each tick.
- **THE CLAIMS LIST** (:data:`CLAIM_W` = 4: ``(channel, y, x, attacker)``, the
  channel ``unit_events.CREDIT_HEAT`` for the heat buffer, a gas slice's index
  for a gas deposit): ``damage.TickCredit.claim`` as records, EVERY claim in
  the reference's order (duplicates included; the environment's claims, an
  attacker of -1, never recorded). "First claim wins" is the reader's: the
  device ``credit_at`` (``cpp/src/cuda_unit_damage.cuh``) scans first match
  first, so the list is a claims table as it stands. It lives one tick and
  goes with the tick-end clears (decision 4).

Host side: :func:`cells_of` (the record function of the cell buffer: the host
pending list's spray cells, in record order), :func:`claims_since` (the claims
the host's ``TickCredit`` gained since a mark, in their first-claim order) and
:func:`first_claims` (a claims list folded to its first claim per key, the
order kept: what a ``TickCredit`` holds after the same calls).

NOT PART OF THE PLAYED TICK: Class T (training only). Import-light: numpy;
``breach_physics`` only inside :class:`DeviceRecordList`.
"""
from __future__ import annotations

import numpy as np

#: The cell buffer's record: (y, x, div) -- cpp/src/cuda_unit_spray.cuh CELL_W.
CELL_W = 3
#: Cells per env per tick: STORAGE. A sprayer's cone holds about range^2 x
#: half-angle tiles (the Dragon-7 at 10 tiles: ~25; the Dragon-9 heavy at 60
#: tiles in a 0.333 m level: ~940); four heavy sprayers at once fit.
CELL_CAP = 4096
#: The claims list's record: (channel, y, x, attacker) -- credit_at's table.
CLAIM_W = 4
#: Claims per env per tick: STORAGE (a sprayed cell claims at most twice).
CLAIM_CAP = 2 * CELL_CAP


class ListOverflow(RuntimeError):
    """More records than a list's capacity (named: the list and the env)."""


def cells_of(sim, pending) -> np.ndarray:
    """THE CELL BUFFER's record function: the host pending list's spray cells
    (``(C, 3)`` int64, in record order -- each SPRAY_CELLS record's span)."""
    from simulation import applier
    return applier.records_and_cells_of(sim, pending)[1]


def claim_rows(tc) -> list:
    """A ``damage.TickCredit``'s claims as ``(channel, y, x, attacker)`` rows
    in its dict's insertion order (= first-claim order), the heat channel as
    ``unit_events.CREDIT_HEAT``."""
    from simulation.damage import CREDIT_HEAT as HOST_HEAT
    from simulation.unit_events import CREDIT_HEAT
    out = []
    for (ch, y, x), a in (getattr(tc, "_owner", None) or {}).items():
        out.append((CREDIT_HEAT if ch == HOST_HEAT else int(ch), int(y), int(x), int(a)))
    return out


def claims_since(gmap, n0: int) -> np.ndarray:
    """The claims ``gmap``'s TickCredit gained after its first ``n0`` keys,
    ``(R, 4)`` int64 in first-claim order."""
    rows = claim_rows(getattr(gmap, "damage_credit", None))[int(n0):]
    return np.array(rows, dtype=np.int64).reshape(len(rows), CLAIM_W)


def first_claims(recs, prior=()) -> np.ndarray:
    """A claims list (``(R, 4)``, every claim in order) folded to what a
    ``TickCredit`` holding ``prior`` keys gains from the same calls: the first
    claim of each new ``(channel, y, x)``, in order."""
    seen = {tuple(int(v) for v in r[:3]) for r in prior}
    out = []
    for r in np.asarray(recs, dtype=np.int64).reshape(-1, CLAIM_W):
        k = (int(r[0]), int(r[1]), int(r[2]))
        if k in seen:
            continue
        seen.add(k)
        out.append(tuple(int(v) for v in r))
    return np.array(out, dtype=np.int64).reshape(len(out), CLAIM_W)


def replay_claims(gmap, recs) -> None:
    """The chained form's landing: each claims record through the host's own
    ``TickCredit.claim``, in order."""
    from simulation.damage import CREDIT_HEAT as HOST_HEAT
    from simulation.damage import tick_credit
    from simulation.unit_events import CREDIT_HEAT
    tc = tick_credit(gmap)
    for r in np.asarray(recs, dtype=np.int64).reshape(-1, CLAIM_W):
        ch = HOST_HEAT if int(r[0]) == CREDIT_HEAT else int(r[0])
        tc.claim(ch, ((int(r[1]), int(r[2])),), int(r[3]))


class DeviceRecordList:
    """ONE per-env device record list ``(N, cap, width)`` int64 with counts and
    dropped counts (``breach_physics.UnitRecordListBuffer``: the event ring's
    own class), for a SERIAL per-env producer (no row staging). ``name`` is
    what a refusal names."""

    def __init__(self, name: str, n_env: int, cap: int, width: int, bp=None):
        if bp is None:
            import breach_physics as bp
        if not getattr(bp, "HAS_CUDA", False):
            raise RuntimeError(f"{name}: the device list needs the CUDA build "
                               f"(breach_physics.HAS_CUDA)")
        if int(n_env) < 1 or int(cap) < 1 or int(width) < 1:
            raise ValueError(f"{name}: n_env, cap and width must be >= 1")
        self.bp = bp
        self.name = name
        self.n_env, self.cap, self.width = int(n_env), int(cap), int(width)
        self.shape = (self.n_env, self.cap, self.width)
        self._buf = bp.UnitRecordListBuffer(self.n_env, 1, self.cap, self.width, 1)

    @property
    def buffer(self):
        return self._buf

    def begin(self) -> None:
        """Every env's list empty (counts and dropped to 0)."""
        self._buf.begin()

    def stage_lists(self, per_env) -> None:
        """Set every env's list from host records (``(R_e, width)`` each);
        more than the capacity refused by name."""
        per_env = [np.asarray(r, dtype=np.int64).reshape(-1, self.width) for r in per_env]
        if len(per_env) != self.n_env:
            raise ValueError(f"{self.name}: {len(per_env)} envs staged, the list "
                             f"holds {self.n_env}")
        recs = np.zeros(self.shape, dtype=np.int64)
        counts = np.zeros(self.n_env, dtype=np.int64)
        for e, r in enumerate(per_env):
            if r.shape[0] > self.cap:
                raise ListOverflow(f"{self.name}: {r.shape[0]} records in env {e} do "
                                   f"not fit its capacity {self.cap}")
            recs[e, :r.shape[0]] = r
            counts[e] = r.shape[0]
        self._buf.upload(recs, counts)

    def download(self):
        """``(records (N, cap, width), counts (N,), dropped (N,))`` int64."""
        return self._buf.download()

    def records(self) -> list:
        """Per env, its records as int tuples (list order)."""
        recs, counts, _ = self.download()
        return [[tuple(int(x) for x in recs[e, i]) for i in range(min(int(counts[e]), self.cap))]
                for e in range(self.n_env)]

    def check(self, what: str = "") -> None:
        """Refuse by name an env whose list dropped records."""
        _r, _c, drp = self.download()
        bad = [e for e in range(self.n_env) if int(drp[e])]
        if bad:
            raise ListOverflow(f"{self.name}{' ' + what if what else ''}: env {bad[0]} "
                               f"dropped {int(drp[bad[0]])} record(s) past its "
                               f"capacity {self.cap}")


def cell_buffer(n_env: int, cap: int = CELL_CAP, bp=None) -> DeviceRecordList:
    """THE CELL BUFFER of ``n_env`` envs."""
    return DeviceRecordList("the cell buffer", n_env, cap, CELL_W, bp=bp)


def claims_list(n_env: int, cap: int = CLAIM_CAP, bp=None) -> DeviceRecordList:
    """THE CLAIMS LIST of ``n_env`` envs."""
    return DeviceRecordList("the claims list", n_env, cap, CLAIM_W, bp=bp)


__all__ = ["CELL_W", "CELL_CAP", "CLAIM_W", "CLAIM_CAP", "ListOverflow", "cells_of",
           "claim_rows", "claims_since", "first_claims", "replay_claims",
           "DeviceRecordList", "cell_buffer", "claims_list"]
