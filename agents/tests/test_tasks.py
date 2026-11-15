"""The made-up tasks (agents/tasks.py): note 2b, 9; design 8's table, its
first five rows. Each task is learnt from three fixed seeds by the real
network code at reduced width on the processor, and each is shown to FAIL
with its own trap put in. Minutes each: marked ``tasks`` (the quick tests
alone: ``pytest agents/tests -q -m "not tasks"``).

THE PASS: on every seed, the mean reward per decision over the last five
rounds reaches 0.8 of the way from a random player's (measured) to the best
possible, inside :data:`BUDGET` rounds. A task that will not pass is a
finding: the line, the task and the trainer's numbers are not bent."""
from __future__ import annotations

import numpy as np
import pytest

from agents import learner as learner_mod
from agents import tasks
from agents import trainer as trainer_mod
from agents.grading import choice_grade
from agents.network import SoldierNet
from agents.settings import Settings
from agents.value_scale import ValueScale

pytestmark = pytest.mark.tasks

SEEDS = (0, 1, 2)
#: The rounds each task must pass within, also the run's planned rounds (the
#: step size falls to zero over them). Fixed before the gate run from one
#: measured run each on seed 0 at this very budget: pointer passed in 37
#: rounds, argument in 176, weights in 85, scale in 92, teams in 123.
BUDGET = {"pointer": 150, "argument": 300, "weights": 300, "scale": 300,
          "teams": 300}
_RANDOM = {}


def _random(name):
    if name not in _RANDOM:
        w = np.asarray(Settings(planned_rounds=1).weights)
        _RANDOM[name] = tasks.random_score(tasks.TASKS[name], w)
    return _RANDOM[name]


def _run(name, seed):
    res = tasks.run_task(tasks.TASKS[name], seed, BUDGET[name],
                         random=_random(name))
    print(f"\n[task] {name} seed {seed}: {'PASS' if res['passed'] else 'FAIL'}"
          f" in {res['rounds']} rounds, reached {res['reached']:.3f} "
          f"(random {res['random']:.3f}, line {res['line']:.3f}, best "
          f"{res['best']:.3f}; share {res['share']:.2f})", end="")
    return res


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", sorted(tasks.TASKS))
def test_the_task_is_learnt(name, seed):
    """Protects: the trainer learns each made-up task from a near-even
    start -- 0.8 of the way from random to best on every seed, inside the
    budget -- with the live check silent on every round (it would raise).
    Breaks when: any of the traps below is in the trainer (each test
    ``test_the_task_fails_with_its_trap`` shows which), or learning itself
    is broken (the clip, the grades' sign, the step)."""
    res = _run(name, seed)
    assert res["passed"], res


# -- the traps: each must make its task FAIL ---------------------------------
def _trap_pointer(monkeypatch):
    """The pointer follows a PLACE: each aim option reads the entry one
    place before it (the same in acting and learning, so the live check is
    silent)."""
    real = SoldierNet._outputs
    monkeypatch.setattr(SoldierNet, "_outputs", lambda self, core, ents:
                        real(self, core, ents.roll(1, dims=1)))


def _trap_argument(monkeypatch):
    """The masks at learning differ from those at acting: the learner
    evaluates the stored choices under masks where every option is legal
    (the live check, which would stop this at once, silenced)."""
    monkeypatch.setattr(learner_mod.Learner, "_masks_for",
                        lambda self, batch, idx: {
                            k: v | True for k, v in
                            _real_masks(self, batch, idx).items()})
    monkeypatch.setattr(learner_mod, "live_check",
                        lambda *a, **k: {"max_logp_gap": 0.0,
                                         "distance": 0.0})


_real_masks = learner_mod.Learner._masks_for


def _trap_weights(monkeypatch):
    """The weights applied to the wrong group: own's weight lands on team
    and team's on own."""
    monkeypatch.setattr(learner_mod, "choice_grade", lambda adv, w:
                        choice_grade(adv, np.asarray(w)[[0, 2, 1, 3, 4]]))


def _trap_scale(monkeypatch):
    """The value targets not rescaled: the scale never moves (mean 0,
    spread 1), so the value network learns targets in the hundreds."""
    monkeypatch.setattr(ValueScale, "update", lambda self, targets: None)


def _trap_teams(monkeypatch):
    """The trainer stores and grades the rows of every team, the one on the
    bots included (the literal reading by the flags)."""
    monkeypatch.setattr(trainer_mod, "learning_teams", lambda env: tuple(
        range(env.layout.n_teams)))


TRAPS = {"pointer": _trap_pointer, "argument": _trap_argument,
         "weights": _trap_weights, "scale": _trap_scale,
         "teams": _trap_teams}


@pytest.mark.parametrize("name", sorted(tasks.TASKS))
def test_the_task_fails_with_its_trap(name, monkeypatch):
    """Protects: each task can FAIL -- with its own trap in (design 8's
    "fails if" column), it does not reach the line within the same budget
    on the first seed -- so its pass means the trap is absent.
    Breaks when: the task is too easy to tell (the trap changes nothing),
    which makes its pass worthless."""
    _random(name)                       # measured without the trap
    TRAPS[name](monkeypatch)
    res = _run(name, SEEDS[0])
    assert not res["passed"], res
