"""``AgentPolicy`` -- the match controller ``"policy"`` (#29 P1a, residency
design §4.1 / §4.5; the ``AgentPolicy`` slot ``control_source`` reserved).

A policy drives its team's units through the SAME per-tick intent facade the
bots and the gamepad use -- ``Simulation.set_move_dir`` / ``clear_move_dir`` /
``set_aim`` / ``set_trigger`` / ``throw_grenade_intent`` / ``use_intent`` /
``reload_intent`` --
so the sim's one intent tap (``Simulation.attach_intent_log``) sees every
call and a recorded episode replays under ``tools/run_match.py --replay``.
Nothing else drives a policy's units, and the policy decides nothing: it
holds ONE ACTION ROW per roster slot of its team, handed to it by a trainer
(the batched environment's farm worker, ``src/rl_env.py``) through
:meth:`AgentPolicy.set_rows`, and issues it.

THE ACTION ROW is an int32 vector whose columns are GENERATED from
``match_replay.INTENT_KINDS`` minus ``drive_zombies`` (the horde's intent)
and the entry points' own parameter names (:func:`action_schema`):

- a kind with no argument beyond the unit (``clear_move_dir``,
  ``use_intent``, ``reload_intent``) is ONE flag column named after the kind (0 = not issued);
- every other argument is a column ``<kind>.<parameter>``, encoded by its
  ``INTENT_KINDS`` type: ``int`` as is (Q16.16 values are their raw ints),
  ``bool`` as 0/1, ``fhex`` (today's float ``fuse_seconds``) as Q16.16
  SECONDS dequantized exactly (``n / 65536``) into the float the entry point
  takes -- the integer fuse is P3e, ``INTENT_KINDS`` is unchanged here --,
  ``int?`` as the int or :data:`NONE_Q` (int32 min) for ``None``.

So ``set_move_dir.dx_q``, ``set_move_dir.dy_q`` (a Q16.16 unit vector),
``set_move_dir.throttle_q`` (Q16 in [0, ONE]: 0 walks, ONE sprints -- G1p;
the sim clamps it), ``clear_move_dir`` (a flag DISTINCT from a zero
vector: a zero vector holds a live, motionless move intent), ``set_aim.dx_q``,
``set_aim.dy_q``, ``set_trigger.held``, ``throw_grenade_intent.dx_q`` /
``.dy_q`` / ``.fuse_seconds`` / ``.reach_q``, ``use_intent``,
``reload_intent`` (G2p).

WHEN a kind is issued is its row in :data:`VERB_RULES` -- a new kind in
``INTENT_KINDS`` is refused here until it has one (the column follows for
free; the rule is a decision):

- ``hold`` kinds are issued on EVERY tick of a decimation step (the sim
  overwrites the unit's live slot each call): move, aim, the trigger HOLD;
- ``latch`` kinds only on the FIRST tick after :meth:`AgentPolicy.set_rows`
  (the sim latches them until a tick consumes them): throw, use, reload;

each gated: ``always``; ``flag`` (its own flag column is non-zero);
``unless:<kind>`` (not when that kind's flag is set: ``set_move_dir`` yields
to ``clear_move_dir``, so a row issues exactly one of the two); ``vector``
(its ``dx_q, dy_q`` pair is not zero: a throw needs a direction).

A ZERO row therefore issues ``set_move_dir(0, 0, 0)``, ``set_aim(0, 0)`` and
``set_trigger(False)`` every tick: a held, motionless stance with the facing
untouched. A row with the ``clear_move_dir`` flag issues ``clear_move_dir``,
``set_aim(0, 0)``, ``set_trigger(False)`` -- what the reference-equality gate
(G-a) drives against idle controllers.

Rows go to the living units of the team in roster-slot order (a dead or
converted unit's row is ignored); kinds in ``INTENT_KINDS`` order. Pyray-free,
headless; draws nothing (the random driver below takes its own generator).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np

from control_source import ControlSource
from match_replay import INTENT_KINDS
from simulation.intents import FP_ONE, dequantize

#: The intent kinds no policy row carries (the horde's own intent).
EXCLUDED_KINDS = ("drive_zombies",)
#: The kinds a policy row carries, in ``INTENT_KINDS`` order.
POLICY_KINDS = tuple(k for k in INTENT_KINDS if k not in EXCLUDED_KINDS)

#: kind -> (class, gate): how a policy issues it in a decimation step (see
#: the module doc). Every kind of :data:`POLICY_KINDS` needs a row.
VERB_RULES = {
    "set_move_dir": ("hold", "unless:clear_move_dir"),
    "clear_move_dir": ("hold", "flag"),
    "set_aim": ("hold", "always"),
    "set_trigger": ("hold", "always"),
    "throw_grenade_intent": ("latch", "vector"),
    "use_intent": ("latch", "flag"),
    "reload_intent": ("latch", "flag"),
}
VERB_CLASSES = ("hold", "latch")

#: The ``int?`` column value meaning ``None``.
NONE_Q = -(2 ** 31)


@dataclass(frozen=True)
class Column:
    """One action-row column: its name, the kind it feeds, the index of the
    argument it is (after the unit id; ``None`` for a kind's flag column)
    and that argument's ``INTENT_KINDS`` type (``"flag"`` for a flag)."""
    name: str
    kind: str
    arg: int | None
    type: str


@dataclass(frozen=True)
class ActionSchema:
    """The generated action-row layout (:func:`action_schema`)."""
    columns: tuple

    @property
    def names(self) -> tuple:
        return tuple(c.name for c in self.columns)

    def __len__(self) -> int:
        return len(self.columns)

    def index(self, name: str) -> int:
        for i, c in enumerate(self.columns):
            if c.name == name:
                return i
        raise KeyError(f"no action column {name!r} (columns: "
                       f"{list(self.names)})")


def _check_rules() -> None:
    missing = [k for k in POLICY_KINDS if k not in VERB_RULES]
    if missing:
        raise RuntimeError(
            f"intent kind(s) {missing} have no policy rule: a new verb is a "
            f"row in match_replay.INTENT_KINDS AND a row in "
            f"control_policy.VERB_RULES (hold/latch + its gate)")
    stale = sorted(set(VERB_RULES) - set(POLICY_KINDS))
    if stale:
        raise RuntimeError(f"control_policy.VERB_RULES names {stale}, which "
                           f"are not policy intent kinds")
    for kind, (cls, gate) in VERB_RULES.items():
        if cls not in VERB_CLASSES:
            raise RuntimeError(f"VERB_RULES[{kind!r}]: class {cls!r} is not "
                               f"one of {VERB_CLASSES}")
        if gate.startswith("unless:"):
            other = gate.split(":", 1)[1]
            if other not in POLICY_KINDS or INTENT_KINDS[other][1:]:
                raise RuntimeError(f"VERB_RULES[{kind!r}]: {gate!r} must name "
                                   f"a flag kind (no argument beyond the unit)")
        elif gate == "flag":
            if INTENT_KINDS[kind][1:]:
                raise RuntimeError(f"VERB_RULES[{kind!r}]: a 'flag' gate needs "
                                   f"a kind with no argument beyond the unit")
        elif gate == "vector":
            if "dx_q" not in _params(kind) or "dy_q" not in _params(kind):
                raise RuntimeError(f"VERB_RULES[{kind!r}]: a 'vector' gate "
                                   f"needs dx_q / dy_q arguments")
        elif gate != "always":
            raise RuntimeError(f"VERB_RULES[{kind!r}]: unknown gate {gate!r}")


def _params(kind: str) -> tuple:
    """The entry point's parameter names after ``self`` and the unit id."""
    from simulation.simulation import Simulation
    sig = inspect.signature(getattr(Simulation, kind))
    names = [p for p in sig.parameters if p != "self"]
    return tuple(names[1:])


_SCHEMA: ActionSchema | None = None


def action_schema() -> ActionSchema:
    """THE action-row layout, generated once from ``INTENT_KINDS`` (minus
    :data:`EXCLUDED_KINDS`) and the ``Simulation`` entry points' parameter
    names; refuses a kind without a :data:`VERB_RULES` row or whose arity
    disagrees with its entry point."""
    global _SCHEMA
    if _SCHEMA is not None:
        return _SCHEMA
    _check_rules()
    cols = []
    for kind in POLICY_KINDS:
        types = INTENT_KINDS[kind]
        if types[:1] != ("int",):
            raise RuntimeError(f"INTENT_KINDS[{kind!r}] must start with the "
                               f"unit id (int), got {types!r}")
        arg_types = types[1:]
        params = _params(kind)
        if len(params) != len(arg_types):
            raise RuntimeError(f"Simulation.{kind} takes {list(params)} "
                               f"after the unit, INTENT_KINDS says "
                               f"{list(arg_types)}")
        if not arg_types:
            cols.append(Column(kind, kind, None, "flag"))
        for i, (p, t) in enumerate(zip(params, arg_types)):
            if t not in ("int", "bool", "fhex", "int?"):
                raise RuntimeError(f"INTENT_KINDS[{kind!r}]: no action-row "
                                   f"encoding for type {t!r}")
            cols.append(Column(f"{kind}.{p}", kind, i, t))
    _SCHEMA = ActionSchema(tuple(cols))
    return _SCHEMA


# ---------------------------------------------------------------------------
# Issuing a row
# ---------------------------------------------------------------------------
class _Plan:
    """The schema compiled into per-kind issue steps (built once)."""

    def __init__(self, schema: ActionSchema):
        self.steps = []
        for kind in POLICY_KINDS:
            cls, gate = VERB_RULES[kind]
            cols = [(i, c) for i, c in enumerate(schema.columns)
                    if c.kind == kind]
            flag = next((i for i, c in cols if c.type == "flag"), None)
            args = [(i, c.type) for i, c in cols if c.type != "flag"]
            gate_col = None
            if gate == "flag":
                gate_col = flag
            elif gate.startswith("unless:"):
                gate_col = schema.index(gate.split(":", 1)[1])
            elif gate == "vector":
                gate_col = (schema.index(f"{kind}.dx_q"),
                            schema.index(f"{kind}.dy_q"))
            self.steps.append((kind, cls == "latch", gate.split(":")[0],
                               gate_col, args))


def _decode(value: int, typ: str):
    if typ == "int":
        return int(value)
    if typ == "bool":
        return bool(value)
    if typ == "fhex":
        return dequantize(int(value))
    if typ == "int?":
        return None if int(value) == NONE_Q else int(value)
    raise RuntimeError(f"no decoding for {typ!r}")


def issue_row(sim, unit_id: int, row, first: bool, plan: _Plan) -> None:
    """Issue one unit's action ``row`` through the intent facade: every
    ``hold`` kind whose gate passes, and the ``latch`` kinds too when
    ``first`` (the first tick of a decimation step)."""
    row = [int(v) for v in row]
    for kind, latch, gate, gate_col, args in plan.steps:
        if latch and not first:
            continue
        if gate == "flag":
            if not row[gate_col]:
                continue
        elif gate == "unless":
            if row[gate_col]:
                continue
        elif gate == "vector":
            if row[gate_col[0]] == 0 and row[gate_col[1]] == 0:
                continue
        getattr(sim, kind)(unit_id, *[_decode(row[i], t) for i, t in args])


class AgentPolicy(ControlSource):
    """The ``policy`` match controller: issues one action row per roster slot
    of ``team`` each tick (see the module doc). :meth:`set_rows` starts a
    decimation step; until the first call it issues nothing. Works with
    ``renderer=None``; calling ``handle_frame`` twice in one tick is a
    no-op."""

    def __init__(self, team: int):
        self.team = int(team)
        self.schema = action_schema()
        self._plan = _Plan(self.schema)
        self._rows = None
        self._first = False
        self._last_tick = None

    def set_rows(self, rows) -> None:
        """Hand the policy its action rows for the next decimation step:
        ``rows[s]`` is roster slot ``s``'s row (an int array ``(S, A)``,
        ``A`` = the schema's width; slots beyond the roster are ignored).
        The latch kinds go out on the next tick only."""
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.shape[1] != len(self.schema):
            raise ValueError(f"policy rows must be (slots, "
                             f"{len(self.schema)}), got {rows.shape}")
        if not np.issubdtype(rows.dtype, np.integer):
            raise ValueError(f"policy rows must be integer, got {rows.dtype}")
        self._rows = rows.astype(np.int64, copy=True)
        self._first = True

    def handle_frame(self, sim, renderer=None) -> None:
        rows = self._rows
        if rows is None:
            return
        tick = int(sim.tick)
        if tick == self._last_tick:
            return
        self._last_tick = tick
        first, self._first = self._first, False
        roster = sim.roster(self.team)
        for slot in range(min(len(roster), rows.shape[0])):
            u = roster[slot]
            if not u.alive or u.team != self.team:
                continue
            issue_row(sim, u.id, rows[slot], first, self._plan)


# ---------------------------------------------------------------------------
# The random-action rows (the smoke driver's and the gates' trainer stub)
# ---------------------------------------------------------------------------
#: The random driver's placeholder domain: per decimation step a unit clears
#: its move with p_clear, holds the trigger with p_trigger, and issues each
#: latch kind (throw, use, reload) with p_latch; every direction is a uniform random
#: unit vector, ``throttle_q`` uniform in [0, ONE] (G1p), the
#: fuse uniform in [fuse_lo_s, fuse_hi_s], the reach uniform in
#: [0, reach_hi_tiles] (the sim clamps it to the hand grenade's range).
RANDOM_DOMAIN = dict(p_clear=0.25, p_trigger=0.5, p_latch=0.05,
                     fuse_lo_s=0.5, fuse_hi_s=3.0, reach_hi_tiles=16)


def _unit_vectors(rng, n: int) -> np.ndarray:
    """``n`` uniform random directions as Q16.16 unit vectors (int64
    ``(n, 2)``): a uniform point of the square, normalised (``sqrt``, IEEE
    correctly rounded) and rounded half away from zero -- the arithmetic of
    ``quantize_stick_direction``, vectorised; a zero draw becomes +x."""
    v = rng.uniform(-1.0, 1.0, size=(n, 2))
    norm = np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])
    zero = norm == 0.0
    norm[zero] = 1.0
    u = v / norm[:, None]
    u[zero] = (1.0, 0.0)
    q = np.sign(u) * np.floor(np.abs(u) * FP_ONE + 0.5)
    return q.astype(np.int64)


def random_action_rows(rng, n: int, domain: dict | None = None) -> np.ndarray:
    """``n`` uniformly random action rows ``(n, A)`` int32 from ``rng`` (a
    seeded ``numpy.random.Generator``) over :data:`RANDOM_DOMAIN` (or
    ``domain``). The rows are what a trainer hands over: integers (the
    directions rounded as ``quantize_stick_direction`` rounds)."""
    d = dict(RANDOM_DOMAIN if domain is None else domain)
    schema = action_schema()
    rows = np.zeros((n, len(schema)), dtype=np.int64)
    pairs = {}
    for i, c in enumerate(schema.columns):
        if c.name.endswith(".dx_q"):
            pairs.setdefault(c.kind, {})["dx"] = i
        elif c.name.endswith(".dy_q"):
            pairs.setdefault(c.kind, {})["dy"] = i
    for kind, ij in pairs.items():
        v = _unit_vectors(rng, n)
        rows[:, ij["dx"]] = v[:, 0]
        rows[:, ij["dy"]] = v[:, 1]
    for i, c in enumerate(schema.columns):
        if c.name.endswith((".dx_q", ".dy_q")):
            continue
        latch = VERB_RULES[c.kind][0] == "latch"
        if c.type == "flag":
            p = d["p_latch"] if latch else d["p_clear"]
            rows[:, i] = rng.random(n) < p
        elif c.type == "bool":
            rows[:, i] = rng.random(n) < d["p_trigger"]
        elif c.type == "fhex":
            lo = int(d["fuse_lo_s"] * FP_ONE)
            hi = int(d["fuse_hi_s"] * FP_ONE)
            rows[:, i] = rng.integers(lo, hi + 1, size=n)
        elif c.type == "int?":
            rows[:, i] = rng.integers(0, int(d["reach_hi_tiles"]) * FP_ONE + 1,
                                      size=n)
        elif c.name.endswith(".throttle_q"):
            rows[:, i] = rng.integers(0, FP_ONE + 1, size=n)
        else:
            raise RuntimeError(f"random_action_rows: no domain for column "
                               f"{c.name!r} ({c.type})")
    # A latch kind gated on its vector fires with p_latch: zero the rest.
    for kind, (cls, gate) in VERB_RULES.items():
        if cls == "latch" and gate == "vector":
            off = rng.random(n) >= d["p_latch"]
            rows[off, schema.index(f"{kind}.dx_q")] = 0
            rows[off, schema.index(f"{kind}.dy_q")] = 0
    return rows.astype(np.int32)


def zero_rows(n: int, *, clear_move: bool = True) -> np.ndarray:
    """``n`` rows that drive nothing: no aim, no trigger, no latch, and the
    move CLEARED (``clear_move=True``) or held at a zero vector."""
    schema = action_schema()
    rows = np.zeros((n, len(schema)), dtype=np.int32)
    if clear_move:
        rows[:, schema.index("clear_move_dir")] = 1
    return rows


__all__ = [
    "AgentPolicy", "ActionSchema", "Column", "action_schema", "issue_row",
    "POLICY_KINDS", "EXCLUDED_KINDS", "VERB_RULES", "NONE_Q",
    "RANDOM_DOMAIN", "random_action_rows", "zero_rows",
]
