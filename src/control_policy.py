"""``AgentPolicy`` -- the match controller ``"policy"`` (#29 P1a, residency
design §4.1 / §4.5; the ``AgentPolicy`` slot ``control_source`` reserved).

A policy drives its team's units through the SAME per-tick intent facade the
bots and the gamepad use -- ``Simulation.set_move_dir`` / ``clear_move_dir`` /
``set_aim`` / ``set_trigger`` / ``throw_grenade_intent`` / ``use_intent`` /
``reload_intent`` / ``shove_intent`` / ``melee_intent`` / ``set_flashlight``
/ ``plant_charge_intent`` -- so the sim's one intent tap
(``Simulation.attach_intent_log``) sees every call and a recorded episode replays under ``tools/run_match.py --replay``.
Nothing else drives a policy's units, and the policy decides nothing: it
holds ONE ACTION ROW per roster slot of its team, handed to it by a trainer
(the batched environment's farm worker, ``src/rl_env.py``) through
:meth:`AgentPolicy.set_rows`, and issues it.

THE ACTION ROW is an int32 vector whose columns are GENERATED from
``match_replay.INTENT_KINDS`` minus ``drive_zombies`` (the horde's intent)
and the entry points' own parameter names (:func:`action_schema`):

- a kind with no argument beyond the unit (``clear_move_dir``,
  ``use_intent``, ``reload_intent``, ``shove_intent``, ``melee_intent``) is ONE flag column named after the kind (0 = not issued);
- every other argument is a column ``<kind>.<parameter>``, encoded by its
  ``INTENT_KINDS`` type: ``int`` as is (Q16.16 values are their raw ints),
  ``bool`` as 0/1, ``fhex`` (a float argument; none today since G3p made
  the fuse integer ticks) as Q16.16 dequantized exactly (``n / 65536``),
  ``int?`` as the int or :data:`NONE_Q` (int32 min) for ``None``.

So ``set_move_dir.dx_q``, ``set_move_dir.dy_q`` (a Q16.16 unit vector),
``set_move_dir.throttle_q`` (Q16 in [0, ONE]: 0 walks, ONE sprints -- G1p;
the sim clamps it), ``clear_move_dir`` (a flag DISTINCT from a zero
vector: a zero vector holds a live, motionless move intent), ``set_aim.dx_q``,
``set_aim.dy_q``, ``set_trigger.held``, ``throw_grenade_intent.dx_q`` /
``.dy_q`` / ``.fuse_ticks`` (integer ticks) / ``.reach_q`` / ``.kind`` (the
hand-grenade row index, G3p), ``use_intent``,
``reload_intent`` (G2p), ``shove_intent`` (G5p-a2), ``melee_intent`` (G6p),
``set_flashlight.off`` (G7p: the OFF sense, so a zero row keeps every light on),
``plant_charge_intent.fuse_ticks`` (#193: integer ticks, the sim clamps it to
5 .. 60 s; 0 = no plant, so a zero row plants nothing).

WHEN a kind is issued is its row in :data:`VERB_RULES` -- a new kind in
``INTENT_KINDS`` is refused here until it has one (the column follows for
free; the rule is a decision):

- ``hold`` kinds are issued on EVERY tick of a decimation step (the sim
  overwrites the unit's live slot each call): move, aim, the trigger HOLD;
- ``latch`` kinds only on the FIRST tick after :meth:`AgentPolicy.set_rows`
  (the sim latches them until a tick consumes them): throw, use, reload,
  shove, melee, plant;

each gated: ``always``; ``flag`` (its own flag column is non-zero);
``unless:<kind>`` (not when that kind's flag is set: ``set_move_dir`` yields
to ``clear_move_dir``, so a row issues exactly one of the two); ``vector``
(its ``dx_q, dy_q`` pair is not zero: a throw needs a direction);
``nonzero`` (its ONE argument is not zero: a plant needs a timer).

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

#: The intent kinds no policy row carries: the horde's own intent and the
#: commander's order (#195 1f -- a soldier's policy never orders itself).
EXCLUDED_KINDS = ("drive_zombies", "order_intent")
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
    "shove_intent": ("latch", "flag"),
    "melee_intent": ("latch", "flag"),
    # The flashlight bit is held every tick like the trigger; its argument is
    # the OFF sense (Simulation.set_flashlight) so a zero row leaves it on.
    "set_flashlight": ("hold", "always"),
    # The plant (#193) latches like the throw, gated on its one argument: a
    # zero fuse column is no plant (the sim would clamp a 0 up to 5 s).
    "plant_charge_intent": ("latch", "nonzero"),
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
        elif gate == "nonzero":
            if len(INTENT_KINDS[kind][1:]) != 1:
                raise RuntimeError(f"VERB_RULES[{kind!r}]: a 'nonzero' gate "
                                   f"needs a kind with exactly one argument "
                                   f"beyond the unit")
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
            elif gate == "nonzero":
                gate_col = args[0][0]
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
        elif gate == "nonzero":
            if not row[gate_col]:
                continue
        getattr(sim, kind)(unit_id, *[_decode(row[i], t) for i, t in args])


class _Aim:
    """One decoded aim choice (the decoder's, never the Unit's): a compass
    token (``kind`` "tok", ``value`` its direction), "walk", "keep", or a
    unit (``value`` its game id, ``last_point`` the last centre tile it
    resolved to)."""
    __slots__ = ("kind", "value", "last_point")

    def __init__(self, kind: str, value: int = -1):
        self.kind, self.value, self.last_point = kind, int(value), None


_WALK = _Aim("walk")


class _Decision:
    """One slot's decoded decision (the five choices after the fallbacks
    that need no sim)."""
    __slots__ = ("feet", "aim", "aim_index", "trigger", "hand", "arg")


class _SlotState:
    """The decoder's per-unit memory (controller state: never on the Unit,
    never synced; design 9 "the decoder's held aim")."""
    __slots__ = ("throw_aim", "last_vec", "aimed_uid", "prev", "uid")

    def __init__(self):
        self.throw_aim = None
        self.last_vec = (0, 0)
        self.aimed_uid = -1
        self.prev = None
        self.uid = -1


class AgentPolicy(ControlSource):
    """The ``policy`` match controller: issues one action row per roster slot
    of ``team`` each tick (see the module doc). :meth:`set_rows` starts a
    decimation step; until the first call it issues nothing. Works with
    ``renderer=None``; calling ``handle_frame`` twice in one tick is a
    no-op.

    THE ACTION SPEC (#195 1h, ``[action] spec``): ``"raw"`` (the default)
    takes the generated rows through :meth:`set_rows`; ``"v1"``
    (:meth:`configure`, ``simulation.action_spec_v1``) takes five integers
    per slot through :meth:`set_decisions` and DECODES them every tick into
    one raw row issued through :func:`issue_row` -- so ``VERB_RULES`` stays
    the one hold / latch rule and nothing new reaches the sim or the
    recording."""

    def __init__(self, team: int):
        self.team = int(team)
        self.schema = action_schema()
        self._plan = _Plan(self.schema)
        self._rows = None
        self._first = False
        self._last_tick = None
        self.action_spec = "raw"
        self._tables = None
        self._dec = None            # v1: per slot a _Decision
        self._slot = {}             # v1: per slot a _SlotState
        idx = self.schema.index
        self._col = {n: idx(n) for n in self.schema.names}

    # -- the action spec (#195 1h) -----------------------------------------
    def configure(self, spec: str, tables=None) -> None:
        """Choose the action spec: ``"raw"`` or ``"v1"`` with its
        ``ActionTables`` (``action_spec_v1.action_tables``). Refused once a
        row has been handed over."""
        from simulation.action_spec_v1 import ACTION_SPECS
        if spec not in ACTION_SPECS:
            raise ValueError(f"unknown action spec {spec!r} (known: "
                             f"{list(ACTION_SPECS)})")
        if self._rows is not None or self._dec is not None:
            raise RuntimeError("AgentPolicy.configure after a row was handed "
                               "over: the action spec is set once, first")
        if spec == "v1" and tables is None:
            raise ValueError("action spec 'v1' needs its ActionTables")
        self.action_spec = spec
        self._tables = tables if spec == "v1" else None

    def set_decisions(self, rows5, list_uid) -> None:
        """The ``"v1"`` twin of :meth:`set_rows`: ``rows5`` the team's
        ``(S, 5)`` block of ``(feet, aim, trigger, hand, arg)``, ``list_uid``
        its ``(S, E)`` block of the observation those rows were chosen on.
        A list pointer becomes a game id HERE, once, and is never kept as a
        position (design 2.2). Indices are trusted in range (the env's door
        refuses the rest)."""
        from simulation import action_spec_v1 as A
        if self.action_spec != "v1":
            raise RuntimeError("set_decisions is the 'v1' action spec's; "
                               "this policy is on 'raw' (set_rows)")
        tb = self._tables
        rows5 = np.asarray(rows5)
        list_uid = np.asarray(list_uid)
        if rows5.ndim != 2 or rows5.shape[1] != len(A.HEADS):
            raise ValueError(f"v1 decisions must be (slots, 5), got "
                             f"{rows5.shape}")
        if list_uid.shape != (rows5.shape[0], tb.n_entries):
            raise ValueError(f"list_uid must be ({rows5.shape[0]}, "
                             f"{tb.n_entries}), got {list_uid.shape}")
        decs = []
        for s in range(rows5.shape[0]):
            feet, aim, trig, hand, arg = (int(v) for v in rows5[s])
            d = _Decision()
            d.feet, d.trigger, d.arg = feet, trig, arg
            if aim < A.AIM_WALK:
                d.aim, d.aim_index = _Aim("tok", aim), aim
            elif aim == A.AIM_KEEP:
                d.aim, d.aim_index = _Aim("keep"), A.AIM_KEEP
            else:
                d.aim, d.aim_index = _WALK, A.AIM_WALK
                i = aim - A.AIM_TOKENS
                if i >= 0:
                    uid = int(list_uid[s, i])
                    if uid >= 0 and tb.aim_lo <= i < tb.aim_hi:
                        d.aim, d.aim_index = _Aim("unit", uid), A.AIM_TOKENS
            # the fallbacks that need no sim (note section 2): the reserved
            # flashlight and a masked argument decode as hand "nothing"
            if hand == A.HAND_FLASHLIGHT:
                hand = A.HAND_NOTHING
            elif hand >= A.HAND_THROW0:
                if arg == A.ARG_ONTO and d.aim.kind != "unit":
                    hand = A.HAND_NOTHING
            elif hand == A.HAND_PLANT and arg == A.ARG_ONTO:
                hand = A.HAND_NOTHING
            d.hand = hand
            decs.append(d)
        self._dec = decs
        self._first = True

    def aim_view(self) -> dict:
        """The decoder's view the observation reads (note section 4): per
        unit id of this team ``(aimed_uid, prev)`` -- the unit its body
        aimed at on the last tick issued (-1 for a token) and the six
        decoded choices of the last decision (``None`` before the first)."""
        return {st.uid: (st.aimed_uid, st.prev)
                for st in self._slot.values() if st.uid >= 0}

    def set_rows(self, rows) -> None:
        """Hand the policy its action rows for the next decimation step:
        ``rows[s]`` is roster slot ``s``'s row (an int array ``(S, A)``,
        ``A`` = the schema's width; slots beyond the roster are ignored).
        The latch kinds go out on the next tick only."""
        if self.action_spec != "raw":
            raise RuntimeError("set_rows is the 'raw' action spec's; this "
                               "policy is on 'v1' (set_decisions)")
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.shape[1] != len(self.schema):
            raise ValueError(f"policy rows must be (slots, "
                             f"{len(self.schema)}), got {rows.shape}")
        if not np.issubdtype(rows.dtype, np.integer):
            raise ValueError(f"policy rows must be integer, got {rows.dtype}")
        self._rows = rows.astype(np.int64, copy=True)
        self._first = True

    def handle_frame(self, sim, renderer=None) -> None:
        if self.action_spec == "v1":
            self._decode_frame(sim)
            return
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

    # -- the v1 decoder: one raw row per living unit per tick ---------------
    def _decode_frame(self, sim) -> None:
        decs = self._dec
        if decs is None:
            return
        tick = int(sim.tick)
        if tick == self._last_tick:
            return
        self._last_tick = tick
        first, self._first = self._first, False
        roster = sim.roster(self.team)
        records = None
        for slot in range(min(len(roster), len(decs))):
            u = roster[slot]
            # a dead or turned row issues NOTHING, ever (1e's note: a latch
            # for such a unit would never be consumed and stay hashed)
            if not u.alive or u.team != self.team:
                continue
            if records is None:
                records = sim.team_picture.records(self.team)
            st = self._slot.get(slot)
            if st is None:
                st = self._slot[slot] = _SlotState()
            st.uid = int(u.id)
            issue_row(sim, u.id, self._decode(u, decs[slot], st, records,
                                              tick, first),
                      first, self._plan)

    @staticmethod
    def _point(aim, records):
        """The centre an aim at a unit resolves to on this tick: its
        record's centre in the team's own picture (the live centre while
        seen, the last-seen one otherwise), else its ``last_point``."""
        rec = records.get(aim.value)
        if rec is not None:
            aim.last_point = (int(rec.centre[0]), int(rec.centre[1]))
        return aim.last_point

    def _decode(self, u, d, st, records, tick, first) -> list:
        """Decision ``d`` as one raw row for unit ``u`` on this tick (note
        section 2)."""
        from simulation import action_spec_v1 as A
        tb = self._tables
        c = self._col
        row = [0] * len(self.schema)
        # -- the aim held through a wind-up: the game's own state decides --
        if st.throw_aim is not None and int(u.throw_release_tick) < tick:
            st.throw_aim = None
        hand = d.hand
        dec_aim = d.aim if d.aim.kind != "keep" or st.throw_aim is not None \
            else _WALK
        if first and hand >= A.HAND_THROW0 and int(u.throw_release_tick) < 0:
            st.throw_aim = dec_aim
        aim = st.throw_aim if st.throw_aim is not None else dec_aim
        if aim.kind == "keep":
            aim = _WALK
        # -- feet --
        opt = A.feet_option(d.feet)
        if opt is None:
            row[c["clear_move_dir"]] = 1
        else:
            dx, dy = A.COMPASS_Q16[opt[0]]
            row[c["set_move_dir.dx_q"]] = dx
            row[c["set_move_dir.dy_q"]] = dy
            row[c["set_move_dir.throttle_q"]] = tb.throttles[opt[1]]
        # -- aim --
        if aim.kind == "tok":
            v = A.COMPASS_Q16[aim.value]
            st.aimed_uid = -1
        elif aim.kind == "unit":
            p = self._point(aim, records)
            v = A.aim_vector(u, p) if p is not None else None
            if v is None:
                v = st.last_vec
            st.aimed_uid = aim.value
        else:
            v = (0, 0)
            st.aimed_uid = -1
        st.last_vec = v
        row[c["set_aim.dx_q"]], row[c["set_aim.dy_q"]] = v
        row[c["set_trigger.held"]] = 1 if d.trigger else 0
        # -- hand (a latch: issue_row sends it on the first tick only) --
        if first:
            arg_used = -1
            if hand == A.HAND_RELOAD:
                row[c["reload_intent"]] = 1
            elif hand == A.HAND_MELEE:
                row[c["melee_intent"]] = 1
            elif hand == A.HAND_SHOVE:
                row[c["shove_intent"]] = 1
            elif hand == A.HAND_USE:
                row[c["use_intent"]] = 1
            elif hand == A.HAND_PLANT:
                row[c["plant_charge_intent.fuse_ticks"]] = \
                    tb.plant_timers[d.arg]
                arg_used = d.arg
            elif hand >= A.HAND_THROW0:
                if d.arg == A.ARG_ONTO:
                    p = self._point(d.aim, records)
                    reach = None if p is None else A.onto_reach_q(u, p)
                else:
                    reach = tb.reaches_q[d.arg]
                if reach is None:
                    hand = A.HAND_NOTHING
                else:
                    fx, fy = A.facing_vector(u)
                    row[c["throw_grenade_intent.dx_q"]] = fx
                    row[c["throw_grenade_intent.dy_q"]] = fy
                    row[c["throw_grenade_intent.fuse_ticks"]] = tb.fuse_ticks
                    row[c["throw_grenade_intent.reach_q"]] = reach
                    row[c["throw_grenade_intent.kind"]] = \
                        hand - A.HAND_THROW0
                    arg_used = d.arg
            if opt is None:
                pf, ps = A.N_DIRS, -1
            else:
                pf, ps = opt
            aim_idx = d.aim_index
            if aim_idx == A.AIM_KEEP and d.aim.kind == "keep" and \
                    dec_aim is _WALK:
                aim_idx = A.AIM_WALK
            st.prev = (pf, ps, aim_idx, 1 if d.trigger else 0, hand, arg_used)
        return row


# ---------------------------------------------------------------------------
# The random-action rows (the smoke driver's and the gates' trainer stub)
# ---------------------------------------------------------------------------
#: The random driver's placeholder domain: per decimation step a unit clears
#: its move with p_clear, holds the trigger with p_trigger, and issues each
#: latch kind (throw, use, reload, shove, melee) with p_latch (and darkens the flashlight with it); every direction is a uniform random
#: unit vector, ``throttle_q`` uniform in [0, ONE] (G1p), the
#: fuse uniform in [fuse_lo_ticks, fuse_hi_ticks] (0.5 to 3 s at 24 tps), the
#: reach uniform in [0, reach_hi_tiles] (the sim clamps it to the hand
#: grenade's range), the grenade ``kind`` uniform over the hand-grenade rows.
RANDOM_DOMAIN = dict(p_clear=0.25, p_trigger=0.5, p_latch=0.05,
                     fuse_lo_ticks=12, fuse_hi_ticks=72, reach_hi_tiles=16,
                     plant_lo_ticks=120, plant_hi_ticks=1440)


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
    nonzero = {k for k, (_, g) in VERB_RULES.items() if g == "nonzero"}
    for i, c in enumerate(schema.columns):
        if c.name.endswith((".dx_q", ".dy_q")) or c.kind in nonzero:
            continue
        latch = VERB_RULES[c.kind][0] == "latch"
        if c.type == "flag":
            p = d["p_latch"] if latch else d["p_clear"]
            rows[:, i] = rng.random(n) < p
        elif c.type == "bool":
            # The flashlight's OFF is rare (p_latch), the trigger's hold half.
            p = d["p_latch"] if c.kind == "set_flashlight" else d["p_trigger"]
            rows[:, i] = rng.random(n) < p
        elif c.name.endswith(".fuse_ticks"):
            rows[:, i] = rng.integers(int(d["fuse_lo_ticks"]),
                                      int(d["fuse_hi_ticks"]) + 1, size=n)
        elif c.name.endswith(".kind"):
            from simulation.weapons import get_tables
            rows[:, i] = rng.integers(
                0, len(get_tables().hand_grenade_rows), size=n)
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
    # A latch kind gated on its one argument (the plant, #193) fires with
    # p_latch, its fuse uniform in [plant_lo_ticks, plant_hi_ticks]: drawn
    # LAST, so every other column is the draw it was before the kind.
    for i, c in enumerate(schema.columns):
        if c.kind in nonzero:
            v = rng.integers(int(d["plant_lo_ticks"]),
                             int(d["plant_hi_ticks"]) + 1, size=n)
            v[rng.random(n) >= d["p_latch"]] = 0
            rows[:, i] = v
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
