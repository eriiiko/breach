"""The memory path and the value-first rounds (rung 2 note 2d, part 1,
sections 2 to 5, 7 and 8; design 4.5, 4.8, 6 row B): the stretches, the
stored memory, the collector and learner with the memory on, the live
check's second clause and its four faults, the value-first rounds. On the
stand-in with the scripted tasks of ``_scripted`` (matches of 3 decisions,
so they end and start inside every stretch; a soldier dies; records end).
Each test names the property and the change that must break it."""
from __future__ import annotations

import copy
import hashlib
import platform

import numpy as np
import pytest
import torch

from agents.collector import Collector
from agents.learner import Learner, LiveCheckFailed
from agents.network import SOLDIER_KEYS, SoldierNet
from agents.settings import Settings
from agents.spec import reduced
from agents.standin import StandIn
from agents.tasks import networks
from agents.tests._scripted import NoisyTask, ScriptTask
from agents.trainer import RunRefused, Trainer

N = 2


def _trainer(task_cls=NoisyTask, memory=True, seed=0, **kw):
    task = task_cls(3)
    env = StandIn(task, N, seed=seed)
    task.bind(env.sizes)
    soldier, value = networks(env, seed, reduced(memory))
    kw.setdefault("decisions", 32)
    kw.setdefault("piece", 64)
    kw.setdefault("lump", 32)
    settings = Settings(planned_rounds=10, **kw)
    return Trainer(env, soldier, value, settings, seed=seed)


# -- without memory: nothing moved -------------------------------------------
#: sha256 of both networks' weights after :func:`_no_memory_run`, made with
#: main's trainer code (``6c5e788c``) BEFORE patch 2d.1 touched it, keyed by
#: the machine it was made on (the processor's float kernels and PyTorch's
#: version decide the bits). On another machine the test is skipped, naming
#: how to make its key: check out main's ``agents/`` as it was, run
#: ``_no_memory_run()`` there, add the line.
NO_MEMORY_REFERENCE = {
    ("2.15.0+cu132", "AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD"):
        "424deb3f220135027a8f6e666ecef8a87eaf2b4b9708cd45d15292c049c45cef",
}


def _no_memory_run() -> str:
    """Three rounds without memory on the noisy task (pieces of 24 cut
    near-equal, lumps of 8), both networks' weights hashed in name order."""
    task = NoisyTask(5)
    env = StandIn(task, 2, seed=7)
    task.bind(env.sizes)
    soldier, value = networks(env, 7)
    s = Settings(planned_rounds=10, decisions=16, piece=24, lump=8)
    tr = Trainer(env, soldier, value, s, seed=7)
    for _ in range(3):
        tr.round()
    h = hashlib.sha256()
    for name, net in (("soldier", tr.learner.soldier),
                      ("value", tr.learner.value)):
        for k, v in sorted(net.state_dict().items()):
            h.update(f"{name}.{k}".encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_a_run_without_memory_gives_mains_weights_bit_for_bit():
    """Protects: with the memory off the patch moved nothing it did not
    mean to -- three rounds give the weights main's trainer gave, bit for
    bit on the processor (note 2d, 2).
    Breaks when: the memory-off learner changes its order of operations
    (the units, the cut, the lumps, the shuffle's draws), the collector's
    draw changes, or a value-first or stretch branch leaks into it."""
    key = (torch.__version__, platform.processor())
    if key not in NO_MEMORY_REFERENCE:
        pytest.skip(f"no reference for {key}: make one from main's agents/ "
                    f"with _no_memory_run() and add it")
    assert torch.get_num_threads() == 4      # the reference's (conftest)
    assert _no_memory_run() == NO_MEMORY_REFERENCE[key]


def test_without_memory_the_batch_keeps_no_memory():
    """Protects: the memory array is allocated only with the memory on.
    Breaks when: a memory-off batch allocates it (the collector would then
    refuse the run, or store into a dead array)."""
    tr = _trainer(memory=False)
    assert tr.batch.memory is None
    tr2 = _trainer(memory=True)
    core = tr2.learner.soldier.memory.hidden_size
    assert tr2.batch.memory.shape == (33, tr2.batch.R, core)
    assert tr2.batch.memory.dtype == np.float32


# -- the refusals --------------------------------------------------------------
@pytest.mark.parametrize("kw, name", [
    (dict(decisions=40), "decisions"),
    (dict(piece=72, lump=24), "piece"),
    (dict(lump=24), "lump"),
    (dict(stretch=12), "decisions"),
])
def test_cuts_that_are_not_whole_stretches_are_refused_by_name(kw, name):
    """Protects: with the memory on, ``decisions``, ``piece`` and ``lump``
    must each be a whole number of stretches, refused by name when the
    trainer is built (note 2d, 2); with the memory off the same numbers
    are let through (``stretch`` is not read).
    Breaks when: a refusal is dropped (the learner would cut a stretch in
    two, or read past the batch's end)."""
    with pytest.raises(RunRefused, match=f"{name} .* stretch"):
        _trainer(memory=True, **kw)
    _trainer(memory=False, **kw)


def test_a_soldier_with_memory_is_trained():
    """Protects: the memory refusal of patch 2b is gone; a soldier with the
    memory on trains, with the live check's second clause measured.
    Breaks when: ``check_run`` refuses the memory, or the second clause is
    not run."""
    st = _trainer(memory=True).round()
    assert st["live"]["max_memory_gap"] <= 1e-4
    assert st["soldier_steps"] >= 1


@pytest.mark.parametrize("kw", [dict(stretch=0), dict(value_first_rounds=-1)])
def test_settings_refuse_a_stretch_below_one_or_negative_value_rounds(kw):
    """Protects: a stretch of 0 or negative value-first rounds is refused by
    name. Breaks when the check is dropped (a stretch of 0 divides by
    zero; negative rounds read as none)."""
    with pytest.raises(ValueError, match=list(kw)[0]):
        Settings(planned_rounds=4, **kw)


# -- the stored memory is what the soldier had ---------------------------------
def test_the_stored_memory_recomputed_along_a_stretch_is_what_the_soldier_had():
    """Protects: design 4.5's test -- from ``memory[t0]`` and the stored
    observations and ``first`` flags, the memory recomputed at every step of
    every stretch equals the stored ``memory[t0 + j]`` (the one the soldier
    had before decision ``t0 + j``), with matches ending inside the
    stretches (3-decision matches) and a soldier dead mid-match; the memory
    is carried across a round's edge (``memory[0]`` of a round is
    ``memory[T]`` of the last, bit for bit, and not zero); and the network
    zeroes it exactly where a match starts.
    Breaks when: the memory is stored after the decision, not zeroed at a
    match's start, zeroed at a round's edge, or zeroed anywhere else."""
    tr = _trainer(ScriptTask, decisions=32, stretch=16)
    b = tr.batch
    tr.collector.collect(b)
    end_1 = b.memory[b.T].copy()
    tr.collector.collect(b)
    assert np.array_equal(b.memory[0], end_1)
    assert np.abs(end_1).max() > 0
    firsts = b.first.astype(bool)
    assert firsts[1:].any() and (~firsts[1:]).any()
    net = tr.collector.soldier
    S = 16
    with torch.no_grad():
        for t0 in range(0, b.T, S):
            state = torch.as_tensor(b.memory[t0])
            for j in range(S):
                t = t0 + j
                obs = {k: torch.as_tensor(b.obs[k][t]) for k in SOLDIER_KEYS}
                first = torch.as_tensor(firsts[t])
                _, state = net(obs, state, first)
                torch.testing.assert_close(
                    state, torch.as_tensor(b.memory[t + 1]), rtol=0,
                    atol=1e-5, msg=f"step {t}")
            # forward_sequence (the learner's path) agrees with the steps
            obs = {k: torch.as_tensor(b.obs[k][t0:t0 + S])
                   for k in SOLDIER_KEYS}
            _, end = net.forward_sequence(
                obs, torch.as_tensor(b.memory[t0]),
                torch.as_tensor(firsts[t0:t0 + S]))
            torch.testing.assert_close(end, torch.as_tensor(b.memory[t0 + S]),
                                       rtol=0, atol=1e-5)
    # zeroed exactly where a match starts: from a non-zero memory, one step
    # with first set equals one step from zeros, and without it does not
    t = int(np.flatnonzero(firsts[1:].any(axis=1))[0]) + 1
    r = int(np.flatnonzero(firsts[t])[0])
    assert np.abs(b.memory[t, r]).max() > 0
    obs = {k: torch.as_tensor(b.obs[k][t, r:r + 1]) for k in SOLDIER_KEYS}
    with torch.no_grad():
        _, a = net(obs, torch.as_tensor(b.memory[t, r:r + 1]),
                   torch.ones(1, dtype=torch.bool))
        _, z = net(obs, torch.zeros_like(torch.as_tensor(
            b.memory[t, r:r + 1])))
        _, c = net(obs, torch.as_tensor(b.memory[t, r:r + 1]),
                   torch.zeros(1, dtype=torch.bool))
    assert torch.equal(a, z) and not torch.equal(a, c)


# -- the live check's second clause, and its four faults -----------------------
def test_the_live_check_is_silent_on_a_correct_run_with_memory():
    """Protects: on a correct run with the memory on (matches ending inside
    every stretch, a dead soldier, rows whose records end) both clauses are
    silent for several rounds, the memory's gap measured and tiny.
    Breaks when: acting and learning differ on any stretch."""
    tr = _trainer(NoisyTask)
    for _ in range(3):
        st = tr.round()
        assert st["live"]["max_memory_gap"] <= 1e-4
        assert st["live"]["max_logp_gap"] <= 1e-4


def _fault_stored_after(monkeypatch):
    """The collector stores the memory AFTER the decision."""
    real = Collector._act_step

    def after(self, batch, t, obs):
        d = real(self, batch, t, obs)
        batch.memory[t] = self._memory.cpu().numpy()
        return d
    monkeypatch.setattr(Collector, "_act_step", after)


def _fault_no_zero(monkeypatch):
    """The collector does not zero the memory at a match's start."""
    real = Collector._act
    monkeypatch.setattr(Collector, "_act",
                        lambda self, obs, state=None, first=None:
                        real(self, obs, state, None))


def _fault_learner_ignores_first(monkeypatch):
    """The learner ignores ``first`` inside a stretch."""
    real = SoldierNet.forward_sequence
    monkeypatch.setattr(SoldierNet, "forward_sequence",
                        lambda self, obs, state=None, first=None:
                        real(self, obs, state, None))


def _fault_cut_one_off(monkeypatch):
    """The stretches cut one decision off: each replays the decisions one
    step after the stored memory it starts from."""
    real = Learner._stretch_decisions

    def shifted(self, batch, units):
        idx = real(self, batch, units)
        return np.minimum(idx + batch.R, batch.T * batch.R - 1)
    monkeypatch.setattr(Learner, "_stretch_decisions", shifted)


FAULTS = {"stored after the decision": _fault_stored_after,
          "not zeroed at a match's start": _fault_no_zero,
          "first ignored when learning": _fault_learner_ignores_first,
          "stretches cut one decision off": _fault_cut_one_off}


@pytest.mark.parametrize("fault", sorted(FAULTS))
def test_the_live_check_trips_on_each_memory_fault(fault, monkeypatch):
    """Protects: the live check stops the run on each of note 2d's four
    memory faults, before any step: the memory stored after the decision,
    not zeroed at a match's start, ``first`` ignored when learning, the
    stretches cut one decision off. The memory clause names the row and the
    stretch.
    Breaks when: the second clause is dropped or compares the stored memory
    with itself, or the first clause stops running through the sequence
    path."""
    tr = _trainer(NoisyTask)
    FAULTS[fault](monkeypatch)
    with pytest.raises(LiveCheckFailed) as e:
        tr.round()
    msg = str(e.value)
    assert ("memory at a stretch's end" in msg
            or "probability of a stored" in msg), msg
    if "memory at a stretch's end" in msg:
        assert "env " in msg and "stretch " in msg, msg


# -- the value-first rounds ----------------------------------------------------
@pytest.mark.parametrize("memory", [False, True])
def test_value_first_rounds_leave_the_soldier_bit_identical(memory):
    """Protects: in the value-first rounds the live check runs, the value
    network and its scale learn, and the soldier's network does not change
    by one bit (its optimiser untouched); in the round after, the soldier
    learns at the plan's step size for that round (the plan is not
    shifted).
    Breaks when: the soldier steps in a value-first round (at any step
    size, Adam's state would move too), the value network is frozen with
    it, the live check is skipped, or the plan restarts after them."""
    tr = _trainer(NoisyTask, memory=memory, value_first_rounds=2)
    s0 = copy.deepcopy(tr.learner.soldier.state_dict())
    v0 = copy.deepcopy(tr.learner.value.state_dict())
    for r in range(2):
        st = tr.round()
        assert st["value_first"] and st["live"] is not None
        assert st["soldier_steps"] == 0 and st["value_steps"] >= 1
        assert not tr.learner.opt_soldier.state
        for k, v in tr.learner.soldier.state_dict().items():
            assert torch.equal(v, s0[k]), k
    assert any(not torch.equal(v, v0[k])
               for k, v in tr.learner.value.state_dict().items())
    assert tr.learner.scale.rounds == 2
    st = tr.round()
    assert not st["value_first"] and st["soldier_steps"] >= 1
    assert st["step_size"] == tr.settings.step_size_at(2)
    assert any(not torch.equal(v, s0[k])
               for k, v in tr.learner.soldier.state_dict().items())
