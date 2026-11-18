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
from agents.collector import Collector
from agents import trainer as trainer_mod
from agents.grading import choice_grade
from agents.network import SoldierNet
from agents.settings import Settings
from agents.value_scale import ValueScale

pytestmark = pytest.mark.tasks

SEEDS = (0, 1, 2)
#: The rounds each task must pass within, also the run's planned rounds (the
#: step size falls to zero over them). Fixed before the gate run from one
#: measured run each on seed 0: pointer passed in 37 rounds, argument in 45,
#: weights in 85, scale in 92, teams in 123 (argument measured at a budget
#: of 300, the others at these).
BUDGET = {"pointer": 150, "argument": 150, "weights": 300, "scale": 300,
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
    res.pop("soldier")
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


# -- the memory tasks (note 2d, 6) ------------------------------------------
#: The rounds each memory task must pass within (also the planned rounds),
#: fixed before the gate run from one measured run each on seed 0 (the
#: measured rounds in the task's text, :class:`agents.tasks.RecallTask`).
MEMORY_BUDGET = {("recall12", 16): None, ("recall24", 32): None}
#: FINDING of 2d.1 (the task's text, :class:`agents.tasks.RecallTask`):
#: recall12 is not learnt with the design's numbers (seed 0: 220 rounds on
#: the tasks' farm and 140 on a farm of 32 matches near the random player;
#: even 4 decisions between cue and answer: 0.11 of the way after 300
#: rounds), so no budget could be measured and the three tests that hang on
#: it (the pass, recall24 at 16 failing, the reset-at-the-cut trap failing:
#: a fail of a task nobody passes proves nothing) are skipped BY NAME until
#: the lead decides what changes. Not a looser line, not an easier task.
#: 2d.1b fixed the network's part (the heads now read the memory through a
#: normalising layer: gap 0 with memory passes by reward in 87 rounds) and
#: measured the rest: the tasks' farm gives recall12 8.5 teaching answers a
#: round (gap 0: 130), still flat after 370 rounds. The farm is the lead's.
#: 2d.1c measured both candidates on seed 0 for 600 rounds each: the
#: stretches starting at a match's first decision (17.3 teaching answers a
#: round) and that with 128 decisions a round (95): both flat; the cue's
#: signal is below the noise (the task's text). Still skipped.
MEMORY_FINDING = ("FINDING 2d.1/2d.1b/2d.1c: recall12 is not learnt by "
                  "reward, neither with stretches starting at a match's "
                  "start nor with 128 decisions a round (agents/tasks.py "
                  "RecallTask); the lead decides what changes")
memory_finding = pytest.mark.skip(reason=MEMORY_FINDING)
#: recall24 with stretches of 16 is run for as long as recall24 with 32 is
#: given, and must stay near the random player: its mean over the last
#: :data:`PASS_WINDOW` rounds below this share of the way to the best.
RECALL24_AT_16_AT_MOST = 0.25


def _memory_random(name):
    if name not in _RANDOM:
        w = np.asarray(Settings(planned_rounds=1).weights)
        _RANDOM[name] = tasks.random_score(tasks.MEMORY_TASKS[name], w)
    return _RANDOM[name]


def _run_memory(name, stretch, seed, budget):
    res = tasks.run_task(tasks.MEMORY_TASKS[name], seed, budget,
                         random=_memory_random(name), memory=True,
                         stretch=stretch)
    res.pop("soldier")
    print(f"\n[task] {name} stretch {stretch} seed {seed}: "
          f"{'PASS' if res['passed'] else 'FAIL'} in {res['rounds']} rounds, "
          f"reached {res['reached']:.3f} (random {res['random']:.3f}, line "
          f"{res['line']:.3f}; share {res['share']:.2f})", end="")
    return res


@memory_finding
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name, stretch", [("recall12", 16), ("recall24", 32)])
def test_the_memory_task_is_learnt(name, stretch, seed):
    """Protects: with the memory on, the trainer learns to answer a cue 12
    decisions later with stretches of 16, and 24 later with stretches of 32
    -- 0.8 of the way from random to best on every seed inside the budget
    -- with soldiers dying mid-match, matches ending inside stretches and
    across rounds' edges, and the live check (both clauses) silent on every
    round (it would raise).
    Breaks when: the memory is stored, carried, zeroed or replayed wrong
    (each of the live check's four faults; the reset at the cut below), the
    dead rows are learnt from, or the endings are wrong."""
    res = _run_memory(name, stretch, seed, MEMORY_BUDGET[(name, stretch)])
    assert res["passed"], res


@memory_finding
def test_recall24_fails_with_stretches_of_16():
    """Protects: design 4.5's limit -- what the memory can LEARN to hold is
    one stretch long: with stretches of 16 the answer 24 decisions after
    its cue is never in the cue's stretch, so recall24 stays near the
    random player for as many rounds as it is given at 32 (asserted: a
    trainer that learns it at 16 is a finding about the stretches, not a
    success).
    Breaks when: the gradient reaches back past a stretch's start (the
    stored memory not a constant, stretches longer than said, or memory
    smuggled through the observation)."""
    res = _run_memory("recall24", 16, SEEDS[0],
                      MEMORY_BUDGET[("recall24", 32)])
    assert not res["passed"], res
    assert res["share"] <= RECALL24_AT_16_AT_MOST, res


def _trap_reset_at_the_cut(monkeypatch, stretch=16):
    """The memory silently reset at the cut: the collector marks every
    stretch's first decision as a match's start (``first``), so the memory
    is zeroed there in acting AND learning alike -- the live check, which
    compares the two, cannot see it."""
    real = Collector._act_step

    def reset_at_cut(self, batch, t, obs):
        if t % stretch == 0:
            batch.first[t] = 1
        return real(self, batch, t, obs)
    monkeypatch.setattr(Collector, "_act_step", reset_at_cut)


@memory_finding
def test_recall12_fails_with_the_memory_reset_at_the_cut(monkeypatch):
    """Protects: recall12 can FAIL on the classic memory fault the live
    check forgives -- the memory zeroed at every stretch's start instead of
    at a match's -- because its matches start anywhere inside a stretch
    (lengths that are not whole stretches), so most cues are cut off their
    answers; within the same budget it does not reach the line.
    Breaks when: the matches are made to start on stretch starts (the trap
    would then change nothing), which makes recall12's pass worthless."""
    _memory_random("recall12")
    _trap_reset_at_the_cut(monkeypatch)
    res = _run_memory("recall12", 16, SEEDS[0],
                      MEMORY_BUDGET[("recall12", 16)])
    assert not res["passed"], res


# -- the "already good start" (note 2d, 7) --------------------------------
_TAUGHT = {}


def _taught_pointer():
    """The pointer task's soldier taught until it passed (seed 0), once per
    test session."""
    if "soldier" not in _TAUGHT:
        res = tasks.run_task(tasks.PointerTask, SEEDS[0], BUDGET["pointer"],
                             random=_random("pointer"))
        assert res["passed"], {k: v for k, v in res.items()
                               if k != "soldier"}
        _TAUGHT["soldier"] = res["soldier"]
    return _TAUGHT["soldier"]


def _good_start(seed, **kw):
    res = tasks.good_start(_taught_pointer(), seed,
                           planned_rounds=BUDGET["pointer"], **kw)
    print(f"\n[good_start] seed {seed} value_first {res['value_first_rounds']}"
          f" variety {res['variety']}: start {res['start']:.3f} lowest "
          f"{res['lowest']:.3f} end {res['end']:.3f} (kept "
          f"{res['kept']:.2f})", end="")
    return res


@pytest.mark.parametrize("seed", SEEDS)
def test_a_good_start_is_not_worn_down_with_value_first_rounds(seed):
    """Protects: design 6 arm B -- a taught soldier given a FRESH value
    network learns 20 rounds with 5 value-first rounds and the design's
    numbers, and its lowest round keeps at least 0.9 of its start.
    Breaks when: the value-first rounds step the soldier (it would be
    graded by a value network that knows nothing), or learning itself
    wears a good start down."""
    res = _good_start(seed, value_first_rounds=5)
    assert res["lowest"] >= tasks.GOOD_START_KEEP * res["start"], res


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", ["no value-first rounds",
                                  "ten times the variety push"])
def test_a_good_start_without_its_guard_is_measured(kind, seed):
    """REPORTED, NOT GATED (note 2d, 7): the same start with no value-first
    rounds, and with five of them but the variety push ten times the
    design's. Nobody knows the answer yet; the numbers are printed for the
    lead (and written into ``good_start``'s text). Protects only that both
    runs complete with the live check silent."""
    if kind == "no value-first rounds":
        res = _good_start(seed, value_first_rounds=0)
    else:
        res = _good_start(seed, value_first_rounds=5,
                          variety=10 * Settings(planned_rounds=1).variety)
    assert len(res["scores"]) == tasks.GOOD_START_ROUNDS
