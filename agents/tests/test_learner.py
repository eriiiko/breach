"""The learner, the live check and the round loop (agents/learner.py,
agents/trainer.py): note 2b, 5 to 7; design 4.4, 4.8; critique lens 2
majors 5 and 6, minors 2, 3, 9. On the stand-in with the scripted tasks of
``_scripted``. Each test names the property and the change that must break
it."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from agents import learner as learner_mod
from agents.collector import Collector
from agents.learner import Learner, LiveCheckFailed
from agents.network import ValueNet
from agents.settings import Settings
from agents.sizes import sizes_for
from agents.spec import reduced
from agents.standin import StandIn
from agents.tasks import networks
from agents.tests._scripted import NoisyTask, ScriptTask
from agents.trainer import RunRefused, Trainer
from agents.value_scale import ValueScale

N = 2


def _trainer(task_cls=NoisyTask, decisions=16, seed=0, **kw):
    task = task_cls(4)
    env = StandIn(task, N, seed=seed)
    task.bind(env.sizes)
    soldier, value = networks(env, seed)
    settings = Settings(planned_rounds=10, decisions=decisions, **kw)
    return Trainer(env, soldier, value, settings, seed=seed)


def test_a_round_learns_with_the_live_check_silent_and_hands_the_weights_over():
    """Protects: one round on the stand-in collects, grades and learns with
    the live check silent (the recomputed probabilities equal the stored
    ones, the distance before any step 0), every piece steps both networks
    when nothing moves far, and the collector plays the next round with the
    learner's new weights.
    Breaks when: acting and learning differ (a format, a mode, a mask), or
    the weights are not handed over (the next round would be played by an
    old policy: off-policy data inside an on-policy update)."""
    tr = _trainer()
    before = copy.deepcopy(tr.collector.soldier.state_dict())
    stats = tr.round()
    assert stats["live"]["max_logp_gap"] <= 1e-4
    assert abs(stats["live"]["distance"]) <= 1e-4
    assert stats["soldier_steps"] >= 1 and stats["value_steps"] >= 1
    after = tr.collector.soldier.state_dict()
    for k, v in tr.learner.soldier.state_dict().items():
        assert torch.equal(after[k].cpu(), v.cpu()), k
    assert any(not torch.equal(before[k], after[k]) for k in before)
    stats2 = tr.round()
    assert stats2["live"]["max_logp_gap"] <= 1e-4


def test_the_live_check_trips_on_a_kept_reference_to_an_env_array(
        monkeypatch):
    """Protects: the live check stops the run when the collector keeps a
    reference to an environment array in place of a copy (here: the stored
    steps written at the round's end from references to the environment's
    arrays, which by then hold its last step), naming the probability.
    Breaks when: the live check is skipped, compares the stored probability
    with itself, or its tolerance is loosened past the fault."""
    tr = _trainer()
    real = Collector.collect

    def kept_references(self, batch):
        refs = []
        gather = self._gather_step

        def remember(b, t):
            out = gather(b, t)
            refs.append((t, self._obs, self._full))   # the env's own arrays
            return out
        self._gather_step = remember
        try:
            real(self, batch)
        finally:
            del self._gather_step
        for t, obs, full in refs:                       # written too late
            o, f = self._slot(batch, t)
            self.rows.gather(obs, o, tuple(o))
            self.rows.gather(full, f, tuple(f))
        return batch
    monkeypatch.setattr(Collector, "collect", kept_references)
    with pytest.raises(LiveCheckFailed, match="probability of a stored"):
        tr.round()


def test_the_live_check_trips_on_masks_taken_from_another_step(monkeypatch):
    """Protects: the live check stops the run when the masks at learning are
    taken from another step than the one the choice was drawn under (the
    noisy task's masks change at every decision), naming which clause.
    Breaks when: the live check is skipped or does not compare which heads
    counted and the argument's mask."""
    tr = _trainer()
    R = tr.batch.R
    real = Learner._masks_for

    def other_step(self, batch, idx):
        return real(self, batch, (np.asarray(idx) + R) % (batch.T * R))
    monkeypatch.setattr(Learner, "_masks_for", other_step)
    with pytest.raises(LiveCheckFailed,
                       match="probability of a stored|heads counted"):
        tr.round()


def test_the_early_stop_steps_no_piece_past_the_distance(monkeypatch):
    """Protects: the early stop -- the first piece whose distance moved
    (from the forward pass its step would use) is over ``max_distance`` is
    NOT stepped, the soldier's network takes no further step that round,
    and the value network finishes all its passes.
    Breaks when: the over-the-line piece is stepped anyway, the stop is per
    pass instead of per round, or the value network stops with it."""
    tr = _trainer(decisions=16, piece=32, lump=16, step_size=0.05)
    steps = []
    real = tr.learner.opt_soldier.step
    monkeypatch.setattr(tr.learner.opt_soldier, "step",
                        lambda *a, **k: (steps.append(1), real(*a, **k)))
    stats = tr.round()
    d = stats["distances"]
    line = tr.settings.max_distance
    assert stats["stopped"] is not None, d
    assert all(x <= line for x in d[:-1]) and d[-1] > line
    assert len(steps) == stats["soldier_steps"] == len(d) - 1
    n_value = int((tr.batch.value_row != 0).sum())
    pieces = -(-n_value // 32)
    assert stats["value_steps"] == tr.settings.passes * pieces


def test_lumps_add_up_to_the_step_of_the_pieces_mean_loss():
    """Protects: a piece worked through in lumps steps both networks
    exactly as the whole piece at once would (the lumps' gradients add up
    to the piece's mean loss): the same weights after a round.
    Breaks when: each lump's loss is a mean over the lump (lumps of unequal
    policy rows then weigh unequally), or the optimiser steps per lump."""
    tr = _trainer(decisions=16, piece=64, lump=64, passes=1)
    batch = tr.collector.collect(tr.batch)
    whole = Learner(copy.deepcopy(tr.learner.soldier),
                    copy.deepcopy(tr.learner.value),
                    replace(tr.settings, max_distance=1e9), seed=3)
    lumps = Learner(copy.deepcopy(tr.learner.soldier),
                    copy.deepcopy(tr.learner.value),
                    replace(tr.settings, max_distance=1e9, lump=8), seed=3)
    whole.learn(batch)
    lumps.learn(batch)
    for a, b in ((whole.soldier, lumps.soldier), (whole.value, lumps.value)):
        for (k, p), q in zip(a.named_parameters(), b.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=2e-6, msg=k)


@pytest.mark.parametrize("n, piece", [(1, 8), (8, 8), (9, 8), (17, 8),
                                      (4099, 2048), (49152, 2048)])
def test_a_pass_is_cut_into_near_equal_pieces_never_a_short_last_one(n, piece):
    """Protects: a pass's rows are cut into the fewest pieces of at most
    ``piece`` rows, every row in exactly one, the sizes one row apart at
    most -- so no step is ever taken from a handful of left-over decisions
    (9 rows in pieces of 8 are 5 + 4, not 8 + 1).
    Breaks when: the pass is cut at fixed multiples of ``piece`` (a last
    piece of 1 to 3 rows then takes a full Adam step on noise), a row is
    dropped or repeated, or a piece exceeds the limit."""
    order = np.random.default_rng(n).permutation(n)
    pieces = learner_mod.cut_pieces(order, piece)
    sizes = [len(p) for p in pieces]
    assert len(pieces) == -(-n // piece)
    assert max(sizes) <= piece and max(sizes) - min(sizes) <= 1
    assert np.array_equal(np.sort(np.concatenate(pieces)), np.arange(n))
    assert learner_mod.cut_pieces(order[:0], piece) == []


def test_each_network_has_its_own_optimiser_and_its_own_clip(monkeypatch):
    """Protects: two optimisers over disjoint parameters, each covering its
    network entirely, and each gradient cut to the limit by itself (a spike
    in the value network cannot shrink the soldier's step).
    Breaks when: one optimiser or one global clip covers both networks."""
    tr = _trainer()
    lr = tr.learner
    ps = {id(p) for g in lr.opt_soldier.param_groups for p in g["params"]}
    pv = {id(p) for g in lr.opt_value.param_groups for p in g["params"]}
    assert ps == {id(p) for p in lr.soldier.parameters()}
    assert pv == {id(p) for p in lr.value.parameters()}
    assert not ps & pv
    seen = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        params = list(params)
        seen.append(({id(p) for p in params}, max_norm))
        return real(params, max_norm, *a, **k)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    tr.round()
    assert seen and all(s in (ps, pv) for s, _ in seen)
    assert {frozenset(s) for s, _ in seen} == {frozenset(ps), frozenset(pv)}
    assert all(m == tr.settings.max_grad_norm for _, m in seen)


def test_the_scale_grades_old_then_moves_then_teaches_new(monkeypatch):
    """Protects: the order of a round (note 2b, 5): the estimates that grade
    are taken with the scale as it was while collecting, THEN the scale
    moves, THEN the value network learns on targets normalised with the
    moved scale.
    Breaks when: the scale is moved before grading (an off-by-one-batch),
    or the targets are normalised with the old scale."""
    tr = _trainer()
    calls = []
    real_est = Learner._estimates
    real_upd = ValueScale.update
    real_norm = ValueScale.normalise
    monkeypatch.setattr(Learner, "_estimates", lambda self, b: (
        calls.append(("grade", self.scale.rounds)), real_est(self, b))[1])
    monkeypatch.setattr(ValueScale, "update", lambda self, t: (
        calls.append(("move", self.rounds)), real_upd(self, t))[1])
    monkeypatch.setattr(ValueScale, "normalise", lambda self, x: (
        calls.append(("teach", self.rounds)), real_norm(self, x))[1])
    tr.round()
    assert calls == [("grade", 0), ("move", 0), ("teach", 1)]


@pytest.mark.parametrize("change, name", [
    (dict(reward_profile="counters"), "profile"),
    (dict(spec="crop"), r"\[obs\] spec"),
    (dict(action_spec="raw"), r"\[action\] spec"),
    (dict(full_view=False), "full_view"),
])
def test_runs_the_trainer_cannot_train_are_refused_by_name(change, name):
    """Protects: the trainer refuses, naming why, a run on the old
    "counters" reward (the clock is no ending there), an observation or
    action spec other than "v1", and the full view switched off.
    Breaks when: a refusal is dropped (the run would train on a wrong
    contract silently)."""
    task = ScriptTask(4)
    env = StandIn(task, N)
    task.bind(env.sizes)
    soldier, value = networks(env, 0)
    env.layout = replace(env.layout, **change)
    with pytest.raises(RunRefused, match=name):
        Trainer(env, soldier, value, Settings(planned_rounds=1))


def test_networks_that_do_not_fit_are_refused():
    """Protects: a network whose record does not fit the layout (a value
    network made for another number of teams) is refused by name. (The
    soldier with the memory on, refused until patch 2d, is trained since
    2d.1: ``test_memory.py``.)
    Breaks when: ``check_fits`` is not called."""
    task = ScriptTask(4)
    env = StandIn(task, N)
    task.bind(env.sizes)
    soldier, value = networks(env, 0)
    other = ValueNet(reduced(False), sizes_for(2, 2))
    with pytest.raises(RunRefused, match="n_teams"):
        Trainer(env, soldier, other, Settings(planned_rounds=1))


def test_the_step_size_falls_to_zero_over_the_planned_rounds():
    """Protects: Adam's step size starts at the design's 0.0003 and falls
    in a straight line to zero over the run's planned rounds.
    Breaks when: the schedule is missing, steps, or ends elsewhere."""
    s = Settings(planned_rounds=4)
    assert [s.step_size_at(r) for r in range(5)] == pytest.approx(
        [3e-4, 2.25e-4, 1.5e-4, 0.75e-4, 0.0])
