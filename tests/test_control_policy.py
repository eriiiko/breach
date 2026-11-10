"""#29 P1a -- the ``policy`` match controller (``src/control_policy.py``).

Properties: the action row has a column for every argument of every intent
kind a policy may issue (generated from ``match_replay.INTENT_KINDS``, the
horde's intent excluded); a kind without an issue rule is refused; within a
decimation step the latch kinds (throw, use) go out on the first tick only
and the holds (move, aim, trigger) on every tick; the ``clear_move_dir``
flag is distinct from a zero move vector; only the team's living units are
driven; the float fuse and the optional reach decode exactly; ``policy`` is
registered but no match file can name it.

The sim here is a stand-in that logs every intent call -- what the policy
ISSUES is the property; that the sim acts on it is the farm gates' business
(``tests/test_rl_env.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import control_policy as cp  # noqa: E402
import control_source  # noqa: E402
from match_replay import INTENT_KINDS  # noqa: E402
from simulation.intents import FP_ONE  # noqa: E402


class _U:
    def __init__(self, uid, team, alive=True, slot=0):
        self.id, self.team, self.alive, self.slot = uid, team, alive, slot


class _Sim:
    """Logs every intent call as ``(tick, kind, args)``."""

    def __init__(self, rosters):
        self.tick = 0
        self._rosters = rosters
        self.calls = []

    def roster(self, team, n_slots=None):
        return list(self._rosters.get(team, ()))

    def __getattr__(self, kind):
        if kind not in INTENT_KINDS:
            raise AttributeError(kind)

        def call(*args):
            self.calls.append((self.tick, kind, args))
        return call


def _row(**cols):
    schema = cp.action_schema()
    row = np.zeros(len(schema), dtype=np.int32)
    for name, v in cols.items():
        row[schema.index(name.replace("__", "."))] = v
    return row


def test_every_policy_intent_argument_has_a_column():
    """Property: each intent kind except the horde's ``drive_zombies`` has
    one column per argument after the unit (named ``<kind>.<parameter>``),
    or one flag column when it takes none; ``drive_zombies`` has none.
    Breaks if a verb or an argument is missing from the action row (a new
    INTENT_KINDS row would not reach a policy)."""
    import inspect
    from simulation.simulation import Simulation
    names = set(cp.action_schema().names)
    for kind, types in INTENT_KINDS.items():
        if kind in cp.EXCLUDED_KINDS:
            assert not any(n == kind or n.startswith(kind + ".")
                           for n in names)
            continue
        params = [p for p in inspect.signature(getattr(Simulation, kind))
                  .parameters if p != "self"][1:]
        assert len(params) == len(types) - 1
        if not params:
            assert kind in names
        for p in params:
            assert f"{kind}.{p}" in names


def test_a_kind_without_an_issue_rule_is_refused(monkeypatch):
    """Property: a policy intent kind with no VERB_RULES row is refused by
    name. Breaks if a new verb would get columns but silently never be
    issued (or issued by a guessed rule)."""
    monkeypatch.setattr(cp, "POLICY_KINDS", cp.POLICY_KINDS + ("new_verb",))
    with pytest.raises(RuntimeError, match="new_verb"):
        cp._check_rules()


def test_latch_kinds_go_out_on_the_first_tick_only():
    """Property: across one decimation step (one set_rows, several ticks)
    throw and use are issued once, on the first tick, while move, aim and
    the trigger hold are issued on every tick. Breaks if a latch repeats
    (a grenade per tick) or a hold is dropped after the first tick."""
    sim = _Sim({0: [_U(5, 0)]})
    pol = cp.AgentPolicy(0)
    pol.set_rows(_row(set_move_dir__dx_q=FP_ONE, set_aim__dy_q=FP_ONE,
                      set_trigger__held=1,
                      throw_grenade_intent__dx_q=FP_ONE,
                      throw_grenade_intent__fuse_ticks=24,
                      use_intent=1)[None])
    for t in range(3):
        sim.tick = t
        pol.handle_frame(sim, None)
    by_kind = {}
    for tick, kind, _ in sim.calls:
        by_kind.setdefault(kind, []).append(tick)
    assert by_kind["throw_grenade_intent"] == [0]
    assert by_kind["use_intent"] == [0]
    for kind in ("set_move_dir", "set_aim", "set_trigger"):
        assert by_kind[kind] == [0, 1, 2]
    # the next step's set_rows re-arms the latches
    pol.set_rows(_row(use_intent=1)[None])
    sim.tick = 3
    pol.handle_frame(sim, None)
    assert [c for c in sim.calls if c[0] == 3 and c[1] == "use_intent"]


def test_clear_move_is_distinct_from_a_zero_vector():
    """Property: a row with the clear_move_dir flag issues clear_move_dir and
    never set_move_dir; an all-zero row issues set_move_dir(0, 0, 0); both
    issue set_aim(0, 0) and set_trigger(False) and no latch. Breaks if the
    flag is conflated with a zero vector (a stopped unit would hold a live
    move intent, or a zero vector would release it)."""
    for clear in (True, False):
        sim = _Sim({0: [_U(9, 0)]})
        pol = cp.AgentPolicy(0)
        pol.set_rows(cp.zero_rows(1, clear_move=clear))
        pol.handle_frame(sim, None)
        got = {(k, a) for _, k, a in sim.calls}
        move = (("clear_move_dir", (9,)) if clear
                else ("set_move_dir", (9, 0, 0, 0)))
        assert got == {move, ("set_aim", (9, 0, 0)),
                       ("set_trigger", (9, False))}


def test_rows_drive_only_the_living_units_still_on_the_team():
    """Property: a dead unit's row and a converted (now zombie-team) unit's
    row issue nothing; rows beyond the roster are ignored. Breaks if a
    policy drives a corpse or becomes a second driver of a zombie."""
    roster = [_U(1, 0), _U(2, 0, alive=False), _U(3, 255), _U(4, 0)]
    sim = _Sim({0: roster})
    pol = cp.AgentPolicy(0)
    pol.set_rows(cp.zero_rows(6))
    pol.handle_frame(sim, None)
    assert {a[0] for _, _, a in sim.calls} == {1, 4}


def test_the_fuse_kind_and_reach_columns_decode_exactly():
    """Property (G3p): the fuse column reaches the entry point as that many
    INTEGER ticks (an int, not a float), the kind column as that hand-grenade
    row index, and the reach column's NONE_Q as None, any other value as that
    int. Breaks if the fuse is formed as a float again, the kind is dropped
    or reordered, or None cannot be expressed."""
    sim = _Sim({0: [_U(7, 0)]})
    pol = cp.AgentPolicy(0)
    pol.set_rows(np.stack([
        _row(throw_grenade_intent__dx_q=FP_ONE,
             throw_grenade_intent__fuse_ticks=37,
             throw_grenade_intent__reach_q=cp.NONE_Q,
             throw_grenade_intent__kind=4)]))
    pol.handle_frame(sim, None)
    (_, _, args), = [c for c in sim.calls if c[1] == "throw_grenade_intent"]
    assert args[3] == 37 and type(args[3]) is int
    assert args[4] is None
    assert args[5] == 4 and type(args[5]) is int
    sim2 = _Sim({0: [_U(7, 0)]})
    sim2.tick = 1          # a new tick (a second call in one tick is a no-op)
    pol.set_rows(np.stack([_row(throw_grenade_intent__dy_q=-FP_ONE,
                                throw_grenade_intent__reach_q=5 * FP_ONE)]))
    pol.handle_frame(sim2, None)
    (_, _, args), = [c for c in sim2.calls
                     if c[1] == "throw_grenade_intent"]
    assert args[4] == 5 * FP_ONE


def test_random_rows_stay_in_their_domain():
    """Property: the trainer stub's random rows are integer rows of the
    schema's width, every direction pair a Q16.16 unit vector (within
    rounding) or, for a latch kind that did not fire, zero; flags 0/1.
    Breaks if the stub emits a row the sim would read as something else."""
    rows = cp.random_action_rows(np.random.default_rng(3), 400)
    schema = cp.action_schema()
    assert rows.shape == (400, len(schema)) and rows.dtype == np.int32
    for kind in ("set_move_dir", "set_aim", "throw_grenade_intent"):
        v = rows[:, [schema.index(f"{kind}.dx_q"),
                     schema.index(f"{kind}.dy_q")]].astype(np.float64)
        n = np.hypot(v[:, 0], v[:, 1]) / FP_ONE
        live = n > 0
        assert np.all(np.abs(n[live] - 1.0) < 1e-4)
        if cp.VERB_RULES[kind][0] == "hold":
            assert live.all()
    for c in schema.columns:
        if c.type in ("flag", "bool"):
            assert set(np.unique(rows[:, schema.index(c.name)])) <= {0, 1}


def test_policy_is_registered_but_no_match_file_can_name_it():
    """Property: ``policy`` builds an AgentPolicy through the registry and
    is not nameable in match.toml (nothing would hand it rows). Breaks if a
    match file could put a team under a controller that never acts."""
    assert "policy" in control_source.registered_controllers()
    assert "policy" not in control_source.match_file_controllers()
    c = control_source.create_controller("policy", 1)
    assert isinstance(c, cp.AgentPolicy) and c.team == 1
    assert c.handle_frame(_Sim({}), None) is None     # no rows: nothing
