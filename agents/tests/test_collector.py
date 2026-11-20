"""The stored decision and the collector (agents/batch.py,
agents/collector.py): note 2b, 2 and 3; design 4.2; critique lens 2 major
5 and minor 5. On the stand-in with the scripted task of ``_scripted``.
Each test names the property and the change that must break it."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from agents import collector as col_mod
from agents.batch import Batch, LearningRows
from agents.collector import Collector, MaskedChoiceSent
from agents.standin import StandIn
from agents.tasks import networks
from agents.tests._scripted import OWN, TEAM, ScriptTask, mark
from agents.trainer import learning_teams
from simulation import action_spec_v1 as A

N, T, MATCH = 2, 5, 3


def _setup(decisions=T):
    task = ScriptTask(MATCH)
    env = StandIn(task, N, seed=5)
    task.bind(env.sizes)
    soldier, _ = networks(env, 0)
    rows = LearningRows(env, learning_teams(env))
    batch = Batch(env.layout, rows, decisions)
    c = Collector(env, soldier, rows, seed=1)
    c.start()
    return env, task, rows, batch, c


def _env_row(r):
    """The environment row of batch row r (S = 2, learning teams 0 and 2)."""
    l, s = (r // 2) % 2, r % 2
    return (0, 2)[l] * 2 + s


def test_only_the_learning_teams_rows_are_stored_in_a_fixed_order():
    """Protects: the stored rows are exactly the roster places of the teams
    on ``policy`` (here teams 0 and 2, never team 1 on the bots), each at a
    fixed place ``(env * L + l) * S + s``, time first ``(T, R, ...)``, in
    the served number formats of the game's shape tables.
    Breaks when: the rows are picked by the flags (every team's), a team
    block is read at the wrong offset, or a stored array changes format."""
    env, task, rows, batch, c = _setup()
    assert rows.teams == (0, 2) and rows.n_rows == N * 2 * 2
    shapes = env.layout.obs_shapes()
    for k, a in batch.obs.items():
        assert a.shape == (T, rows.n_rows) + shapes[k][0][2:]
        assert a.dtype == np.dtype(shapes[k][1])
    c.collect(batch)
    for r in range(rows.n_rows):
        e, u = r // 4, _env_row(r)
        for t in range(T):
            if batch.valid[t, r]:
                ep, clock = divmod(t, MATCH)
                assert batch.obs["vector"][t, r, task.col] == mark(ep, clock,
                                                                    u)


def test_the_batch_holds_copies_of_what_each_step_served():
    """Protects: every stored array is a COPY of what the environment served
    at that step (the observation, the masks, the full view; the view after
    the last step), so the environment overwriting its arrays at its next
    step -- or anyone zeroing them later -- changes nothing stored.
    Breaks when: the collector keeps a reference to an environment array
    (the stored steps then all read the last step's view), or the last view
    is taken before the last step."""
    env, task, rows, batch, c = _setup()
    c.collect(batch)
    full_t = np.array(batch.full["full_match"][:, :, 0])
    for t in range(T):
        ep, clock = divmod(t, MATCH)
        for r in range(rows.n_rows):
            if batch.in_match[t, r]:
                assert full_t[t, r] == mark(ep, clock, _env_row(r))
    # the view after the last step: match 1, decision 2
    assert batch.last_full["full_match"][0, 0] == mark(1, 2, 0)
    assert batch.last_obs["vector"][0, task.col] == mark(1, 2, 0)
    before = {k: v.copy() for k, v in batch.obs.items()}
    for k in env._buf:
        env._buf[k][...] = 0
    for k, v in batch.obs.items():
        np.testing.assert_array_equal(v, before[k])


def test_the_row_flags_endings_and_rewards_are_kept_apart():
    """Protects: per row and step -- ``first`` at a match's first decision
    (the run's first and after an in-step restart); ``done`` at a match's
    end and where a row's records end while its match goes on; a dead
    soldier stays a VALUE row (paid the team's reward) but not a policy
    row; a row after its records ended and an empty roster place are in
    nothing and paid nothing; the rewards are the step's.
    Breaks when: the restart is not seen (no ``first``), ``rows_ended`` is
    ignored (the closed row goes on as a value row), a dead row is dropped
    from the value rows, or an empty place is counted."""
    env, task, rows, batch, c = _setup()
    c.collect(batch)
    first = np.zeros((T, rows.n_rows), dtype=np.uint8)
    first[[0, 3]] = 1
    np.testing.assert_array_equal(batch.first, first)
    # env 0 rows 0..3 = (team 0: u0, u1), (team 2: u4, empty u5)
    np.testing.assert_array_equal(batch.valid[:, 1], [1, 0, 0, 1, 0])
    np.testing.assert_array_equal(batch.value_row[:, 1], [1] * 5)
    np.testing.assert_array_equal(batch.reward[:, 1, TEAM], [0.5] * 5)
    np.testing.assert_array_equal(batch.reward[:, 0, OWN], [1, 2, 3, 1, 2])
    np.testing.assert_array_equal(batch.done[:, 0], [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(batch.value_row[:, 3], [0] * 5)
    np.testing.assert_array_equal(batch.reward[:, 3], np.zeros((T, 5)))
    # env 1 rows 4..7; team 2's u4 (row 6): records end at decision 1 of
    # every match (steps 1 and 4) while the match goes on
    np.testing.assert_array_equal(batch.done[:, 6], [0, 1, 1, 0, 1])
    np.testing.assert_array_equal(batch.value_row[:, 6], [1, 1, 0, 1, 1])
    np.testing.assert_array_equal(batch.in_match[:, 6], [1] * 5)
    np.testing.assert_array_equal(batch.reward[:, 6, OWN], [1, 2, 0, 1, 2])
    np.testing.assert_array_equal(batch.value_row[:, 7], [0] * 5)
    # policy rows are value rows
    assert not ((batch.valid != 0) & (batch.value_row == 0)).any()


def test_a_dead_or_empty_learning_row_draws_the_default():
    """Protects: a learning row that is dead or empty carries one legal
    option per head, so its draw is the default (stand, where I walk, no
    trigger, nothing, 0) and counts no head (probability 1, log 0).
    Breaks when: the stand-in or the game stops serving the default masks
    on a row with valid 0, or the collector draws such a row from other
    masks."""
    env, task, rows, batch, c = _setup()
    c.collect(batch)
    off = batch.valid == 0
    assert off.any()
    want = np.array([A.FEET_STAND, A.AIM_WALK, 0, A.HAND_NOTHING, 0])
    np.testing.assert_array_equal(batch.actions[off],
                                  np.broadcast_to(want, (int(off.sum()), 5)))
    assert (batch.used[off] == 0).all() and (batch.logp[off] == 0).all()


def test_rows_the_trainer_does_not_drive_are_sent_the_legal_default(
        monkeypatch):
    """Protects: every row of a team not on ``policy`` is sent the legal
    default, whatever the network would choose, and the masked-choice count
    is read on the driven living rows only.
    Breaks when: the collector sends its draws (or zeros: aim 0 is masked
    on a dead row) to rows it does not drive."""
    env, task, rows, batch, c = _setup()
    sent = []
    real = env.step

    def spy(actions):
        sent.append(np.array(actions))
        return real(actions)
    monkeypatch.setattr(env, "step", spy)
    c.collect(batch)
    for a in sent:
        team1 = a[:, 2:4]
        assert (team1[..., 0] == A.FEET_STAND).all()
        assert (team1[..., 1] == A.AIM_WALK).all()
        assert (team1[..., 2:] == [0, A.HAND_NOTHING, 0]).all()


def test_a_masked_choice_on_a_driven_row_stops_the_run(monkeypatch):
    """Protects: a masked choice counted on a driven living row stops the
    run with an error naming the env, the team and the slot.
    Breaks when: ``info["masked_choices"]`` is not read, or read on the
    wrong rows."""
    env, task, rows, batch, c = _setup()
    real = Collector._act

    def bad(self, obs):
        d = real(self, obs)
        d["actions"][:, 3] = A.HAND_FLASHLIGHT     # reserved: always masked
        return d
    monkeypatch.setattr(Collector, "_act", bad)
    with pytest.raises(MaskedChoiceSent, match=r"env 0, team 0, slot 0"):
        c.collect(batch)


def test_default_actions_are_legal_under_any_served_masks():
    """Protects: the default sent to a row not driven is legal under the
    masks served (a throw being wound up leaves "keep" as the one aim).
    Breaks when: the default is a fixed row of zeros or ignores the
    wind-up."""
    masks = {k: np.zeros((1, 2, n), dtype=np.uint8)
             for k, n in zip(A.MASK_KEYS, A.head_sizes(34, 3)[:4])}
    for i in range(2):
        A.default_masks({k: v[0] for k, v in masks.items()}, i)
    masks["mask_aim"][0, 1] = 0
    masks["mask_aim"][0, 1, A.AIM_KEEP] = 1
    a = col_mod.default_actions(masks, 5)
    assert A.masked_choices(a, masks).max() == 0
    assert a[0, 1, 1] == A.AIM_KEEP and a[0, 0, 1] == A.AIM_WALK
