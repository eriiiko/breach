"""simulation/reward_v1.py -- reward profile v1 as code (#195 1i).

THE contract is ``docs/rl_agent_rung1_note_1i_2026-11-13.md`` (design
``docs/rl_agent_design_2026-11-11.md`` 6.1-6.9, 7.2, 10). A reward is a
NAMED table of term weights (:data:`PROFILES`, the run file's ``[reward]
profile``) over per-row COUNTERS that the books fill from the game's own
state and credited events; the env returns one number per GROUP
(:data:`REWARD_GROUPS`) for every ``in_match`` row and never their sum
(the trainer weights the five and blends own against team).

HEALTH is counted in BARS: a unit's health over its bar (``TeamPicture.
bars``, the health it had when it came into being on its side), clipped at
one, a corpse's 0 (:func:`bar_ratio`). A marine team ``t``'s BALANCE is

    B_t = mb (H_t - H_enemy) + ma (A_t - A_enemy) - zb H_zombie

(:func:`standing` / :func:`balance`): ``H`` its living soldiers' bars, ``A``
their count, the enemy every OTHER marine team, ``H_zombie`` every living
zombie's bars. The books (:class:`RewardBooks`) split each tick's change of
it into the parts a row is credited with (OWN, FRIENDLY FIRE) and the part
no row is credited with (the team's UNCREDITED REST: the environment, the
horde, a third team, a turning, a change made outside a tick), every part
BUILT from its source -- the rest is never "the change minus the own
parts" -- and the TEAM part is the change itself read from state, shared
out over the team's roster. A tick whose events do not reproduce a unit's
hit points raises :class:`RewardBooksMismatch`.

Pure host Python: it reads the sim, the tick's events and the team
picture's bars and writes nothing to any of them. No BLAS (the No-BLAS
rule: sums are broadcasts or Python loops).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.events import UnitHitEvent, UnitKilledEvent
from simulation.ruleset import MatchRules

#: The profile names the run file's ``[reward] profile`` takes: ``"counters"``
#: (the default, the placeholder: a weight per counter name, the weighted
#: sum) and the named term tables of :data:`PROFILES`.
COUNTERS_PROFILE = "counters"

#: The named profiles: term -> weight (design 6.2's sizes). "range" and
#: "duel" (6.9) are rung 2's rows.
PROFILES = {
    "fireteam": {
        "win": 10.0,
        "eliminated": -10.0,
        "clock": -3.0,
        "marine_bar": 1.0,
        "marine_alive": 0.5,
        "zombie_bar": 0.25,
        "ff_bar": -2.0,
        "ff_kill": -3.0,
    },
}
REWARD_PROFILES = (COUNTERS_PROFILE,) + tuple(PROFILES)

#: The five groups, in this order everywhere (``reward[..., g]``).
REWARD_GROUPS = ("outcome", "own", "team", "friendly_fire", "order")

#: The books' counters, APPENDED by the env after its old ones (the
#: generated ``rl_env.COUNTER_NAMES``). Never pinned by a test.
BOOK_COUNTERS = (
    "bars_enemy",          # bars the row's unit took off enemy marines
    "kills_enemy",         # enemy marines it killed
    "bars_zombie",         # bars it took off zombies
    "bars_lost",           # bars its own unit lost (a heal: negative)
    "died",                # its unit's death (1)
    "bars_team",           # bars it took off its own team-mates
    "share_bars",          # dH_t / n_t
    "share_alive",         # dA_t / n_t
    "share_enemy_bars",    # dH_enemy / n_t
    "share_enemy_alive",   # dA_enemy / n_t
    "share_zombie_bars",   # dH_zombie / n_t
    "won",
    "eliminated",
    "clock",
)

#: counter -> (group, term, sign): THE map from a profile's terms to the
#: counters' weights (the code-fixed signs: the mirror of 6.2 holds for any
#: profile a run file writes). ``team_kills`` is the env's old counter
#: (same team at the tick's start, ``attacker != victim``), exactly 6.2's
#: friendly kill.
_COUNTER_TERMS = {
    "won": ("outcome", "win", 1.0),
    "eliminated": ("outcome", "eliminated", 1.0),
    "clock": ("outcome", "clock", 1.0),
    "bars_enemy": ("own", "marine_bar", 1.0),
    "bars_lost": ("own", "marine_bar", -1.0),
    "kills_enemy": ("own", "marine_alive", 1.0),
    "died": ("own", "marine_alive", -1.0),
    "bars_zombie": ("own", "zombie_bar", 1.0),
    "share_bars": ("team", "marine_bar", 1.0),
    "share_enemy_bars": ("team", "marine_bar", -1.0),
    "share_alive": ("team", "marine_alive", 1.0),
    "share_enemy_alive": ("team", "marine_alive", -1.0),
    "share_zombie_bars": ("team", "zombie_bar", -1.0),
    "bars_team": ("friendly_fire", "ff_bar", 1.0),
    "team_kills": ("friendly_fire", "ff_kill", 1.0),
}

_Q16 = 65536


def profile_terms(name: str, overrides: dict | None = None,
                  where: str = "[reward]") -> dict:
    """The term table of profile ``name`` with ``overrides`` (term ->
    number) applied, refusing by name an unknown profile, an unknown term
    and a non-number."""
    if name not in PROFILES:
        raise ValueError(f"{where} profile must be one of "
                         f"{list(REWARD_PROFILES)}, got {name!r}")
    terms = dict(PROFILES[name])
    for k, v in (overrides or {}).items():
        if k not in terms:
            raise ValueError(f"{where} {k!r} is not a term of profile "
                             f"{name!r} (terms: {sorted(terms)}); under a "
                             f"profile the weights live on terms, never on "
                             f"counters")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{where} {k} must be a number, got {v!r}")
        terms[k] = float(v)
    return terms


def group_weights(terms: dict, counter_names) -> np.ndarray:
    """``(K, 5)`` float64: counter ``k``'s weight in each group of
    :data:`REWARD_GROUPS` (its term's weight times its fixed sign); zero
    for every counter in no group; the ``order`` column all zero (6.7)."""
    gi = {g: i for i, g in enumerate(REWARD_GROUPS)}
    out = np.zeros((len(counter_names), len(REWARD_GROUPS)), dtype=np.float64)
    for k, name in enumerate(counter_names):
        got = _COUNTER_TERMS.get(name)
        if got is not None:
            g, term, sign = got
            out[k, gi[g]] = sign * float(terms[term])
    return out


# ---------------------------------------------------------------------------
# Health in bars; the balance
# ---------------------------------------------------------------------------
def bar_q_of(bar) -> int:
    """A bar (``TeamPicture.bars``' float, exact n/65536) on its Q16 grid."""
    return int(round(float(bar) * _Q16))


def bar_ratio(hp_q: int, bar_q: int) -> float:
    """THE health in bars: ``min(max(hp_q, 0), bar_q) / bar_q`` (float64);
    above one bar counts as one, below zero as zero; 0 without a bar."""
    if bar_q <= 0:
        return 0.0
    return min(max(int(hp_q), 0), int(bar_q)) / float(bar_q)


def _health(hp_q: int, alive: bool, bar_q: int) -> float:
    """A unit's health term: its bars while alive, 0 as a corpse."""
    return bar_ratio(hp_q, bar_q) if alive else 0.0


@dataclass(frozen=True)
class StandingParts:
    """A match's standing: per non-zombie team its living units' bars ``H``
    and their count ``A``; every living zombie's bars ``H_zombie``."""
    H: dict
    A: dict
    H_zombie: float

    def h(self, t) -> float:
        return self.H.get(int(t), 0.0)

    def a(self, t) -> int:
        return self.A.get(int(t), 0)


def standing(units, bars) -> StandingParts:
    """THE standing (one implementation of "a team's health in bars"; the
    full view's balance column reads it too): over ``units`` (any iterable;
    the dead are skipped) and the team picture's ``bars``."""
    H, A, hz = {}, {}, 0.0
    for u in units:
        if not u.alive:
            continue
        r = bar_ratio(u.hp_q, bar_q_of(bars.get(int(u.id), 0.0)))
        if u.is_zombie:
            hz += r
        else:
            t = int(u.team)
            H[t] = H.get(t, 0.0) + r
            A[t] = A.get(t, 0) + 1
    return StandingParts(H, A, hz)


def balance(parts: StandingParts, t: int, terms: dict) -> float:
    """``B_t`` (6.2; every other team an enemy at the same weight, 6.6)."""
    t = int(t)
    h_en = sum(v for k, v in parts.H.items() if k != t)
    a_en = sum(v for k, v in parts.A.items() if k != t)
    return (terms["marine_bar"] * (parts.h(t) - h_en)
            + terms["marine_alive"] * (parts.a(t) - a_en)
            - terms["zombie_bar"] * parts.H_zombie)


def _contrib(health: float, alive: bool, team: int, zombie: bool, t: int,
             terms: dict) -> float:
    """One unit's part of ``B_t``."""
    if not alive:
        return 0.0
    if zombie:
        return -terms["zombie_bar"] * health
    if team == t:
        return terms["marine_bar"] * health + terms["marine_alive"]
    return -terms["marine_bar"] * health - terms["marine_alive"]


# ---------------------------------------------------------------------------
# The books
# ---------------------------------------------------------------------------
class RewardBooksMismatch(RuntimeError):
    """A tick's events do not reproduce a unit's hit points or life: an hp
    write outside ``apply_packet`` or an event with no packet (a game
    finding, never patched over)."""


@dataclass
class _Unit:
    hp_q: int
    alive: bool
    team: int
    zombie: bool
    bar_q: int


_CI = {n: i for i, n in enumerate(BOOK_COUNTERS)}


class RewardBooks:
    """One episode's books (the note's section 2). Build after the match
    is spawned; call :meth:`begin` right before each ``MatchSession.tick``,
    :meth:`end` right after it, :meth:`finish` once with the result when
    the session is terminal. :meth:`start_step` zeroes the per-step
    accumulators a decision reads:

    - ``counters`` ``(U, len(BOOK_COUNTERS))`` float64, a row's records
      masked once they ended; ``raw`` the same unmasked (the identities);
    - ``d_balance`` ``(T,)`` the change of each team's ``B_t`` read from
      state; ``rest`` ``(T,)`` the uncredited rest (in balance units);
      ``outcome`` ``(T,)`` the outcome group paid to each of a team's rows;
    - ``rows_ended`` ``(U,)`` bool: the rows whose records ended this step.

    ``n_teams`` / ``n_slots`` are the RUN's ``T`` / ``S`` (the arrays'
    shapes); the match's own teams are ``spec.teams``. ``stats`` counts what
    the walk met (overkill hits, corpse hits, turnings, ...): the gate's
    vacuity floor reads it."""

    def __init__(self, sim, spec, n_slots: int, terms: dict,
                 n_teams: int | None = None):
        if getattr(sim, "team_picture", None) is None:
            raise ValueError("the reward books read the team picture's bars: "
                             "a match sim (spawn_match) has one")
        self.terms = dict(terms)
        self.S = int(n_slots)
        self.n_match = len(spec.teams)
        self.T = int(n_teams) if n_teams is not None else self.n_match
        self.n_rows = self.T * self.S
        self.row_of = {}
        self.rows = {}
        self.n_t = {}
        for t in range(self.n_match):
            r = sim.roster(t)
            self.rows[t] = np.array([t * self.S + u.slot for u in r],
                                    dtype=np.int64)
            self.n_t[t] = len(r)
            for u in r:
                self.row_of[int(u.id)] = (t, t * self.S + int(u.slot))
        self._units = self._snapshot(sim)
        self.parts = standing(sim.units, sim.team_picture.bars)
        self.living = set(MatchRules.living_teams(sim))
        self._eliminated = set()
        self.ended = np.zeros(self.n_rows, dtype=bool)
        self._pending_end = np.zeros(self.n_rows, dtype=bool)
        self.finished = False
        self.outside_changes = 0
        self.stats = {k: 0 for k in (
            "overkill_hits", "corpse_hits", "turnings", "zombie_bites",
            "uncredited_hits", "friendly_fire", "eliminations", "wins",
            "clocks", "posthumous_credits", "heals")}
        nb = len(BOOK_COUNTERS)
        self._scr = np.zeros((self.n_rows, nb), dtype=np.float64)
        self.counters = np.zeros((self.n_rows, nb), dtype=np.float64)
        self.raw = np.zeros((self.n_rows, nb), dtype=np.float64)
        self.rest = np.zeros(self.T, dtype=np.float64)
        self.d_balance = np.zeros(self.T, dtype=np.float64)
        self.outcome = np.zeros(self.T, dtype=np.float64)
        self.rows_ended = np.zeros(self.n_rows, dtype=bool)

    # -- the step's accumulators -------------------------------------------
    def start_step(self) -> None:
        self.counters.fill(0.0)
        self.raw.fill(0.0)
        self.rest.fill(0.0)
        self.d_balance.fill(0.0)
        self.outcome.fill(0.0)
        self.rows_ended.fill(False)

    def _flush(self) -> None:
        scr = self._scr
        self.raw += scr
        live = ~self.ended
        self.counters[live] += scr[live]
        if self._pending_end.any():
            new = self._pending_end & ~self.ended
            self.ended |= new
            self.rows_ended |= new
            self._pending_end.fill(False)
        scr.fill(0.0)

    # -- state ---------------------------------------------------------------
    @staticmethod
    def _snapshot(sim) -> dict:
        bars = sim.team_picture.bars
        return {int(u.id): _Unit(int(u.hp_q), bool(u.alive), int(u.team),
                                 bool(u.is_zombie),
                                 bar_q_of(bars.get(int(u.id), 0.0)))
                for u in sim.units}

    def _member(self, uid: int, st: _Unit):
        """``(team, row)`` of a MEMBER (on its row's team at the tick's start
        and not a zombie; a corpse included), else None."""
        tr = self.row_of.get(uid)
        if tr is None or st.zombie or st.team != tr[0]:
            return None
        return tr

    # -- settling one change of one unit -----------------------------------
    def _settle(self, v: int, st: _Unit, r: float, dead: int,
                attacker: int, *, hit: bool) -> None:
        """Book a change of unit ``v`` (stored state ``st``): ``r`` bars
        removed (negative: given back), ``dead`` its life term (1 a death,
        -1 a return, 0 none), credited to ``attacker`` when it is a member
        other than ``v`` (-1 = nobody)."""
        terms, scr = self.terms, self._scr
        mb, ma, zb = (terms["marine_bar"], terms["marine_alive"],
                      terms["zombie_bar"])
        mv = self._member(v, st)
        ma_ = None
        if attacker >= 0 and attacker != v:
            ast = self._units.get(attacker)
            if ast is not None:
                ma_ = self._member(attacker, ast)
                if ma_ is not None and not ast.alive:
                    self.stats["posthumous_credits"] += 1
        # the rows' own parts (cases 1-3)
        if mv is not None:
            scr[mv[1], _CI["bars_lost"]] += r
            scr[mv[1], _CI["died"]] += dead
        enemy = False
        if ma_ is not None:
            if st.zombie:
                scr[ma_[1], _CI["bars_zombie"]] += r
            elif mv is not None and mv[0] != ma_[0]:
                enemy = True
                scr[ma_[1], _CI["bars_enemy"]] += r
                scr[ma_[1], _CI["kills_enemy"]] += dead
            elif mv is not None and hit:
                scr[ma_[1], _CI["bars_team"]] += r
                if r != 0.0:
                    self.stats["friendly_fire"] += 1
        # case 4: per team, the change no row of the team is credited with
        for t in range(self.n_match):
            if st.zombie:
                delta = zb * r
            elif st.team == t:
                delta = -mb * r - ma * dead
            else:
                delta = mb * r + ma * dead
            covered = 0.0
            if mv is not None and mv[0] == t:
                covered += -mb * r - ma * dead
            if ma_ is not None and ma_[0] == t:
                if st.zombie:
                    covered += zb * r
                elif enemy:
                    covered += mb * r + ma * dead
            self.rest[t] += delta - covered

    def _restate(self, old_health: float, old: _Unit, new_health: float,
                 new: _Unit) -> None:
        """A change of side (the turning) or of existence: wholly
        uncredited, into every team's rest."""
        for t in range(self.n_match):
            self.rest[t] += (
                _contrib(new_health, new.alive, new.team, new.zombie, t,
                         self.terms)
                - _contrib(old_health, old.alive, old.team, old.zombie, t,
                           self.terms))

    def _eliminations(self, sim) -> None:
        now = set(MatchRules.living_teams(sim))
        for t in sorted(self.living - now):
            if t in self._eliminated or t >= self.n_match:
                continue
            self._eliminated.add(t)
            self.stats["eliminations"] += 1
            rows = self.rows[t]
            self._scr[rows, _CI["eliminated"]] += 1.0
            self.outcome[t] += self.terms["eliminated"]
            self._pending_end[rows] = True
        self.living = now

    # -- the two calls a tick ----------------------------------------------
    def begin(self, sim) -> None:
        """Right before a tick: settle any change made OUTSIDE a tick (only
        a test writes one) with no attacker, pay an elimination it caused,
        store the state. The team part's standing is NOT refreshed (the next
        :meth:`end` differences it from the last one)."""
        if self.finished:
            raise RuntimeError("the reward books are finished")
        now = self._snapshot(sim)
        for uid, st in self._units.items():
            nw = now.get(uid)
            if nw is None:
                self.outside_changes += 1
                self._restate(_health(st.hp_q, st.alive, st.bar_q), st, 0.0,
                              _Unit(0, False, st.team, st.zombie, 0))
                continue
            if (nw.team, nw.zombie) != (st.team, st.zombie):
                self.outside_changes += 1
                self._restate(_health(st.hp_q, st.alive, st.bar_q), st,
                              _health(nw.hp_q, nw.alive, nw.bar_q), nw)
                continue
            if (nw.hp_q, nw.alive, nw.bar_q) == (st.hp_q, st.alive,
                                                 st.bar_q):
                continue
            self.outside_changes += 1
            r = (_health(st.hp_q, st.alive, st.bar_q)
                 - _health(nw.hp_q, nw.alive, nw.bar_q))
            dead = int(st.alive) - int(nw.alive)
            self._settle(uid, st, r, dead, -1, hit=False)
        for uid, nw in now.items():
            if uid not in self._units:
                self.outside_changes += 1
                self._restate(0.0, _Unit(0, False, nw.team, nw.zombie, 0),
                              _health(nw.hp_q, nw.alive, nw.bar_q), nw)
        self._units = now
        self._eliminations(sim)
        self._flush()

    def end(self, sim) -> None:
        """Right after a tick: the walk over ``sim.tick_events``, the
        turning, the check, the team part, the eliminations."""
        if self.finished:
            raise RuntimeError("the reward books are finished")
        units = self._units
        h = {uid: st.hp_q for uid, st in units.items()}
        al = {uid: st.alive for uid, st in units.items()}
        for ev in sim.tick_events:
            if isinstance(ev, UnitHitEvent):
                v = int(ev.unit_id)
                st = units.get(v)
                d = float(ev.damage)
                if st is None or d == 0.0:
                    continue
                d_q = int(round(d * _Q16))
                a = int(ev.attacker_id)
                if not al[v]:
                    self.stats["corpse_hits"] += 1
                elif d_q > max(h[v], 0):
                    self.stats["overkill_hits"] += 1
                if d < 0.0:
                    self.stats["heals"] += 1
                elif a < 0:
                    self.stats["uncredited_hits"] += 1
                if (a >= 0 and a in units and units[a].zombie
                        and self._member(v, st) is not None and d > 0.0):
                    self.stats["zombie_bites"] += 1
                before = _health(h[v], al[v], st.bar_q)
                h[v] -= d_q
                r = before - _health(h[v], al[v], st.bar_q)
                self._settle(v, st, r, 0, a if d > 0.0 else -1, hit=True)
            elif isinstance(ev, UnitKilledEvent):
                v = int(ev.unit_id)
                st = units.get(v)
                if st is None or not al[v]:
                    continue
                r = _health(h[v], True, st.bar_q)
                al[v] = False
                self._settle(v, st, r, 1, int(ev.attacker_id), hit=False)
        now = self._snapshot(sim)
        tick = int(sim.tick)
        for uid, st in units.items():
            nw = now.get(uid)
            old_h = _health(h[uid], al[uid], st.bar_q)
            if nw is None:
                self._restate(old_h, _Unit(h[uid], al[uid], st.team,
                                           st.zombie, st.bar_q),
                              0.0, _Unit(0, False, st.team, st.zombie, 0))
                continue
            if (nw.team, nw.zombie) != (st.team, st.zombie):
                self.stats["turnings"] += 1
                self._restate(old_h, _Unit(h[uid], al[uid], st.team,
                                           st.zombie, st.bar_q),
                              _health(nw.hp_q, nw.alive, nw.bar_q), nw)
                continue
            if h[uid] != nw.hp_q or al[uid] != nw.alive:
                raise RewardBooksMismatch(
                    f"tick {tick}: unit {uid}'s events end at hp_q {h[uid]} "
                    f"alive {al[uid]}, the unit holds hp_q {nw.hp_q} alive "
                    f"{nw.alive} -- an hp or life write outside apply_packet "
                    f"or an event with no packet")
        for uid, nw in now.items():
            if uid not in units:
                self._restate(0.0, _Unit(0, False, nw.team, nw.zombie, 0),
                              _health(nw.hp_q, nw.alive, nw.bar_q), nw)
        self._units = now
        # the team part: the change itself, from state, over the roster
        old, new = self.parts, standing(sim.units, sim.team_picture.bars)
        self.parts = new
        h_all_old, h_all_new = sum(old.H.values()), sum(new.H.values())
        a_all_old, a_all_new = sum(old.A.values()), sum(new.A.values())
        dz = new.H_zombie - old.H_zombie
        scr = self._scr
        for t in range(self.n_match):
            self.d_balance[t] += (balance(new, t, self.terms)
                                  - balance(old, t, self.terms))
            n = self.n_t[t]
            if n == 0:
                continue
            dh = new.h(t) - old.h(t)
            da = new.a(t) - old.a(t)
            dhe = (h_all_new - new.h(t)) - (h_all_old - old.h(t))
            dae = (a_all_new - new.a(t)) - (a_all_old - old.a(t))
            rows = self.rows[t]
            scr[rows, _CI["share_bars"]] += dh / n
            scr[rows, _CI["share_alive"]] += da / n
            scr[rows, _CI["share_enemy_bars"]] += dhe / n
            scr[rows, _CI["share_enemy_alive"]] += dae / n
            scr[rows, _CI["share_zombie_bars"]] += dz / n
        self._eliminations(sim)
        self._flush()

    def finish(self, result) -> None:
        """Once, when the session is terminal: the win (``result``'s team)
        or the clock (a draw with two or more teams standing: every row of
        each, the dead included); every row's records end."""
        if self.finished:
            raise RuntimeError("finish() was already called on these books")
        if result is not None and result.outcome == "win":
            t = int(result.team)
            if t < self.n_match:
                self.stats["wins"] += 1
                self._scr[self.rows[t], _CI["won"]] += 1.0
                self.outcome[t] += self.terms["win"]
        elif result is not None and len(self.living) >= 2:
            for t in sorted(self.living):
                if t < self.n_match:
                    self.stats["clocks"] += 1
                    self._scr[self.rows[t], _CI["clock"]] += 1.0
                    self.outcome[t] += self.terms["clock"]
        for t in range(self.n_match):
            self._pending_end[self.rows[t]] = True
        self._flush()
        self.finished = True


__all__ = [
    "COUNTERS_PROFILE", "PROFILES", "REWARD_PROFILES", "REWARD_GROUPS",
    "BOOK_COUNTERS", "profile_terms", "group_weights", "bar_q_of",
    "bar_ratio", "StandingParts", "standing", "balance", "RewardBooks",
    "RewardBooksMismatch",
]
