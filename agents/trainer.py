"""agents/trainer.py -- the round loop that joins the collector and the
learner, in one process (rung 2 note 2b, 1 item 7; design 4.1, 4.6).

ONE ROUND (:meth:`Trainer.round`):

1. **Collect**: the collector plays T decisions in each of the N matches
   with the weights it was handed and fills the batch;
2. **Grade and learn**: the learner grades the batch, moves the value
   targets' scale, runs the live check and learns;
3. **Hand over**: the learner's new weights go to the collector; the batch
   is overwritten by the next round.

The game is met only through the environment (``rl_env.BatchedEnv``, or in
the tests the stand-in of :mod:`agents.standin`, which has the same names).

REFUSED BY NAME at the start (:func:`check_run`): a run whose reward is not
a named profile (the old "counters": the clock is not an ending there); an
observation or action spec other than "v1"; the full view switched off; a
network whose record does not fit the layout (``spec.check_fits``); a
network with the memory on (the memory path is patch 2d's).
"""
from __future__ import annotations

import time

import torch

from agents import _game  # noqa: F401
from agents.batch import Batch, LearningRows
from agents.collector import Collector
from agents.learner import Learner, one_arithmetic
from agents.settings import Settings
from agents.sizes import sizes_for
from agents.spec import SpecRecord, check_fits, v1
from simulation import action_spec_v1 as A
from simulation import obs_spec_v1 as O
from simulation.reward_v1 import COUNTERS_PROFILE

#: The controller name of a learning team (``[run] policy``).
LEARNING_CONTROLLER = "policy"


class RunRefused(ValueError):
    """The trainer refuses this run or network (the message names why)."""


def learning_teams(env) -> tuple:
    """The teams the run file puts on the ``policy`` controller."""
    ctrls = env.run.team_controllers(env.layout.n_teams)
    return tuple(t for t, c in enumerate(ctrls) if c == LEARNING_CONTROLLER)


def layout_sizes(layout):
    """The networks' sizes for an environment's layout."""
    return sizes_for(layout.n_slots, layout.n_teams, layout.n_kinds)


def check_run(env, soldier, value) -> None:
    """Refuse, by name, what this trainer cannot train (module text)."""
    lay = env.layout
    if lay.reward_profile == COUNTERS_PROFILE:
        raise RunRefused(f"[reward] profile {lay.reward_profile!r}: the "
                         f"trainer needs a named profile (under the old "
                         f"counters the clock is not an ending)")
    if lay.spec != O.SPEC_NAME:
        raise RunRefused(f"[obs] spec {lay.spec!r}: the trainer reads "
                         f"observation spec {O.SPEC_NAME!r}")
    if lay.action_spec != A.ACTION_SPEC_NAME:
        raise RunRefused(f"[action] spec {lay.action_spec!r}: the trainer "
                         f"answers in action spec {A.ACTION_SPEC_NAME!r}")
    if not lay.full_view:
        raise RunRefused("[obs] full_view is off: the value network reads "
                         "the full view")
    if soldier.spec.memory:
        raise RunRefused("a soldier's network with the memory on: the "
                         "memory path is patch 2d's")
    sizes = layout_sizes(lay)
    try:
        check_fits(soldier.record, SpecRecord.of(v1(), sizes))
        check_fits(value.record, SpecRecord.of(v1(), sizes, value=True))
    except ValueError as e:
        raise RunRefused(str(e)) from None
    if not learning_teams(env):
        raise RunRefused("no team is on the 'policy' controller: nothing "
                         "to learn")


class Trainer:
    """Collector and learner over ``env``, one process (module text)."""

    def __init__(self, env, soldier, value, settings: Settings,
                 device="cpu", seed: int = 0):
        one_arithmetic()
        check_run(env, soldier, value)
        self.env = env
        self.settings = settings
        self.rows = LearningRows(env, learning_teams(env))
        self.batch = Batch(env.layout, self.rows, settings.decisions)
        self.learner = Learner(soldier, value, settings, device, seed)
        self.collector = Collector(env, self.learner.soldier, self.rows,
                                   device, seed)
        self.collector.set_weights(self.learner.weights())
        self.collector.start()
        self.rounds = 0

    def round(self) -> dict:
        """One round: collect, grade and learn, hand the weights over."""
        t0 = time.perf_counter()
        batch = self.collector.collect(self.batch)
        _sync(self.learner.device)
        t1 = time.perf_counter()
        stats = self.learner.learn(batch)
        _sync(self.learner.device)
        t2 = time.perf_counter()
        self.collector.set_weights(self.learner.weights())
        _sync(self.learner.device)
        t3 = time.perf_counter()
        self.rounds += 1
        stats.update(round=self.rounds, collect_seconds=t1 - t0,
                     learn_seconds=t2 - t1, handover_seconds=t3 - t2,
                     round_seconds=t3 - t0,
                     decisions=int((batch.valid != 0).sum()))
        return stats


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


__all__ = ["LEARNING_CONTROLLER", "RunRefused", "learning_teams",
           "layout_sizes", "check_run", "Trainer"]
