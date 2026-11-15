"""agents/tasks.py -- the made-up tasks that need no memory (rung 2 note
2b, 9; design 8, the first five rows of its table), each built to FAIL on
one of the trainer's own traps, and the runner that scores them.

| task | the trainer fails it if |
|---|---|
| :class:`PointerTask`: point at the list entry that carries a flag; the list is shuffled at every decision | the pointer follows a place in the list and not the entry |
| :class:`ArgumentTask`: the legal arguments depend on the hand verb and the aim just chosen; the trigger is unused on most decisions, the argument whenever the verb takes none | a choice is graded that did not count, or the masks at learning differ from those at acting |
| :class:`WeightTask`: the reward comes in the "own" group, the opposite in "team" (weight 0) | the weights are applied to the wrong group |
| :class:`ScaleTask`: rewards a hundred times larger than usual, on top of a large offset the value network must learn, beside "outcome", always 0 | the rescaling of the value targets is wrong |
| :class:`TeamTask`: a second team in the match that does not learn, paid the opposite of the learner | the trainer stores or grades rows it does not drive |

Each runs on the stand-in (:mod:`agents.standin`), the real network code at
reduced width, on the processor. THE PASS (:func:`run_task`): the mean
reward per decision (the learning team's living rows, the run's group
weights, from the stand-in's own books) over the last five rounds reaches
0.8 of the way from a random player's (:func:`random_score`: every legal
option equally likely, measured) to the best possible (the task's
:attr:`best`), inside a number of rounds the test fixes.
"""
from __future__ import annotations

import numpy as np
import torch

from agents import _game  # noqa: F401
from agents import choosing, fake
from agents.network import SOLDIER_KEYS, SoldierNet, ValueNet, as_tensors
from agents.settings import Settings
from agents.spec import reduced
from agents.standin import StandIn, Task
from agents.trainer import Trainer, layout_sizes
from simulation import action_spec_v1 as A
from simulation import obs_spec_v1 as O
from simulation.reward_v1 import REWARD_GROUPS

G_OUTCOME, G_OWN, G_TEAM = (REWARD_GROUPS.index(g)
                            for g in ("outcome", "own", "team"))
#: The share of the way from random to best a task must reach.
PASS_SHARE = 0.8
#: The rounds the pass is averaged over.
PASS_WINDOW = 5


def numeric_columns(n_columns: int, kinds, count: int) -> list:
    """The first ``count`` columns of a table that are numbers, not kinds
    (where a task writes its signal)."""
    kind_pos = {k.pos for k in kinds}
    cols = [i for i in range(n_columns) if i not in kind_pos]
    if len(cols) < count:
        raise ValueError(f"{count} number columns wanted, {len(cols)} there")
    return cols[:count]


def refresh_aim(obs: dict, i: int) -> None:
    """Row ``i``'s aim mask from its list as the game builds it: the
    compass tokens and "where I walk", every visible aimable entry."""
    tp = fake._type_pos()
    aim = obs["mask_aim"][i]
    aim[...] = 0
    aim[:A.AIM_KEEP] = 1
    aim[A.AIM_TOKENS:] = (obs["list_mask"][i] != 0) & np.isin(
        obs["list"][i, :, tp].astype(np.int64), A.AIMABLE_TYPES)


def _onehot(obs, i, cols, k):
    obs["vector"][i, cols] = -1.0
    obs["vector"][i, cols[k]] = 1.0


def _feet_option(direction: int) -> int:
    """The feet index of ``direction`` (0..15) at walking speed."""
    return 1 + direction * A.N_SPEEDS


def _sizes_cols(sizes):
    return sizes.vector, sizes.vector_kinds


class _Base(Task):
    """A task over the stand-in's sizes (set by :meth:`bind`)."""

    def bind(self, sizes):
        self.sizes = sizes
        return self


class PointerTask(_Base):
    """Point at the flagged one of ``k`` visible entries, the list shuffled
    at every decision; "own" pays 1 for the flagged entry."""

    def __init__(self, k: int = 5):
        super().__init__(name="pointer", match_len=7, best=1.0)
        self.k = k

    def bind(self, sizes):
        super().bind(sizes)
        self.flag_col = numeric_columns(sizes.entry, sizes.entry_kinds, 1)[0]
        return self

    def start(self, e, rng):
        return {"target": np.full(self.n_teams * self.n_slots, -1)}

    def observe(self, e, state, obs, full, valid, rng):
        E = obs["list_mask"].shape[1]
        tp = fake._type_pos()
        for i in np.flatnonzero(valid):
            pos = rng.permutation(E)[:self.k]
            obs["list_mask"][i] = 0
            obs["list_mask"][i, pos] = 1
            obs["list"][i, pos, tp] = O.TYPE_SEEN
            obs["list"][i, pos, self.flag_col] = -1.0
            flagged = pos[rng.integers(self.k)]
            obs["list"][i, flagged, self.flag_col] = 1.0
            state["target"][i] = flagged
            refresh_aim(obs, i)

    def pay(self, e, state, actions, valid, rng):
        u = len(valid)
        r = np.zeros((u, len(REWARD_GROUPS)))
        hit = actions[:, 1] == A.AIM_TOKENS + state["target"]
        r[:, G_OWN] = np.where(valid & hit, 1.0, 0.0)
        return r, np.zeros(u, bool), np.zeros(u, bool), False


class ArgumentTask(_Base):
    """A cue names the hand verb wanted, a plant or the first throw (four
    verbs are legal: nothing and reload too); the argument's right value
    depends on the verb AND the aim just chosen (a plant: timer 1; a throw:
    "onto" at an entry, reach 3 at a token). "own" pays half for the RIGHT
    verb and the other half when its argument is right too; a wrong verb
    pays nothing. The trigger has one legal option on most decisions (a
    head unused there), and the argument does not count whenever "nothing"
    or "reload" is drawn.

    NO VERB IS PAID WITHOUT ITS ARGUMENT (2b's findings; both forms below
    were measured and dropped). The first form paid the argument's half
    whatever the verb, so "always nothing" earned 0.667 against a random
    player's 0.44 with no cue read, and the policy sat there for good. The
    second paid only a right verb but still asked for "nothing" on a third
    of the decisions: a verb that needs no argument is paid in full by the
    verb alone, so from a near-even start the policy first fell to it
    whatever the cue (by round 60 "nothing" was chosen on 96 % of its own
    cues and plant and throw on 10 % of theirs) and took 300 rounds to come
    back, ending at 0.75 of the way. That fall is how this kind of trainer
    behaves from nothing wherever doing nothing is safe -- the reason rung 2
    starts from the copy -- and not one of the traps this task is for."""

    WANTED = (A.HAND_PLANT, A.HAND_THROW0)
    HANDS = (A.HAND_NOTHING, A.HAND_RELOAD, A.HAND_PLANT, A.HAND_THROW0)

    def __init__(self, k: int = 3, p_trigger_unused: float = 0.8):
        super().__init__(name="argument", match_len=7, best=1.0)
        self.k = k
        self.p_unused = p_trigger_unused

    def bind(self, sizes):
        super().bind(sizes)
        self.cue = numeric_columns(sizes.vector, sizes.vector_kinds,
                                   len(self.WANTED))
        return self

    def start(self, e, rng):
        return {"want": np.zeros(self.n_teams * self.n_slots, dtype=int)}

    def observe(self, e, state, obs, full, valid, rng):
        E = obs["list_mask"].shape[1]
        tp = fake._type_pos()
        for i in np.flatnonzero(valid):
            w = int(rng.integers(len(self.WANTED)))
            state["want"][i] = self.WANTED[w]
            _onehot(obs, i, self.cue, w)
            pos = rng.permutation(E)[:self.k]
            obs["list_mask"][i] = 0
            obs["list_mask"][i, pos] = 1
            obs["list"][i, pos, tp] = O.TYPE_SEEN
            refresh_aim(obs, i)
            obs["mask_hand"][i] = 0
            obs["mask_hand"][i, list(self.HANDS)] = 1
            obs["mask_trigger"][i] = 1
            if rng.uniform() < self.p_unused:
                obs["mask_trigger"][i, 1] = 0

    @staticmethod
    def right_arg(hand, aim):
        """The argument a decision must carry (-1: none counts)."""
        return np.where(hand == A.HAND_PLANT, 1,
                        np.where(hand >= A.HAND_THROW0,
                                 np.where(aim >= A.AIM_TOKENS, A.ARG_ONTO, 3),
                                 -1))

    def pay(self, e, state, actions, valid, rng):
        u = len(valid)
        hand, aim, arg = actions[:, 3], actions[:, 1], actions[:, 4]
        want = self.right_arg(hand, aim)
        arg_ok = (want < 0) | (arg == want)
        r = np.zeros((u, len(REWARD_GROUPS)))
        verb_ok = hand == state["want"]
        r[:, G_OWN] = np.where(valid & verb_ok, 0.5 + 0.5 * arg_ok, 0.0)
        return r, np.zeros(u, bool), np.zeros(u, bool), False


class WeightTask(_Base):
    """A cue names one of four directions; "own" pays 1 for walking that
    way, "team" (weight 0) pays the opposite (-1).

    Four, not eight (2b's finding): the tasks' farm gives one learning step
    256 decisions, where a real piece has 2 048. With eight cues and nine
    legal options a cue's right answer is drawn about 3.5 times per step,
    too few to grade it (0.48 of the way after 2 000 steps); with four it
    is drawn about 13 times, as often as the pointer task's one answer.
    What the task protects (which group the weights land on) does not
    depend on the number of directions."""

    def __init__(self, n_dirs: int = 4):
        super().__init__(name="weights", match_len=7, best=1.0)
        self.n_dirs = n_dirs

    def bind(self, sizes):
        super().bind(sizes)
        self.cue = numeric_columns(sizes.vector, sizes.vector_kinds,
                                   self.n_dirs)
        stride = A.N_DIRS // self.n_dirs
        self.options = [_feet_option(d * stride) for d in range(self.n_dirs)]
        return self

    def start(self, e, rng):
        return {"right": np.zeros(self.n_teams * self.n_slots, dtype=int)}

    def observe(self, e, state, obs, full, valid, rng):
        for i in np.flatnonzero(valid):
            d = int(rng.integers(self.n_dirs))
            state["right"][i] = self.options[d]
            _onehot(obs, i, self.cue, d)
            obs["mask_feet"][i] = 0
            obs["mask_feet"][i, A.FEET_STAND] = 1
            obs["mask_feet"][i, self.options] = 1

    def pay(self, e, state, actions, valid, rng):
        u = len(valid)
        right = valid & (actions[:, 0] == state["right"])
        r = np.zeros((u, len(REWARD_GROUPS)))
        r[:, G_OWN] = np.where(right, 1.0, 0.0)
        r[:, G_TEAM] = np.where(right, -1.0, 0.0)
        return r, np.zeros(u, bool), np.zeros(u, bool), False


class ScaleTask(_Base):
    """A cue says pull or release; "own" pays ``100 x (b + right)``, where
    the offset ``b`` (0..3, fixed per env, shown in the vector) is much
    larger than the choice's worth -- only a value network that learns
    ``b`` grades the choice well. "outcome" is always 0."""

    SIZE = 100.0
    LEVELS = 4

    def __init__(self):
        mean_b = (self.LEVELS - 1) / 2.0
        super().__init__(name="scale", match_len=7,
                         best=self.SIZE * (mean_b + 1.0))

    def bind(self, sizes):
        super().bind(sizes)
        self.cue, self.level = numeric_columns(sizes.vector,
                                               sizes.vector_kinds, 2)
        return self

    def start(self, e, rng):
        return {"b": e % self.LEVELS,
                "pull": np.zeros(self.n_teams * self.n_slots, dtype=int)}

    def observe(self, e, state, obs, full, valid, rng):
        for i in np.flatnonzero(valid):
            pull = int(rng.integers(2))
            state["pull"][i] = pull
            obs["vector"][i, self.cue] = 1.0 if pull else -1.0
            obs["vector"][i, self.level] = state["b"] / (self.LEVELS - 1)

    def pay(self, e, state, actions, valid, rng):
        u = len(valid)
        right = actions[:, 2] == state["pull"]
        r = np.zeros((u, len(REWARD_GROUPS)))
        r[:, G_OWN] = np.where(valid, self.SIZE * (state["b"] + right), 0.0)
        return r, np.zeros(u, bool), np.zeros(u, bool), False


class TeamTask(_Base):
    """Team 0 learns (one soldier; three empty roster places), team 1 is on
    the bots (four soldiers, shown the same cue); "own" pays the learner 1
    for walking the cued way and every soldier of team 1 the opposite.
    Four directions, for the reason given at :class:`WeightTask`."""

    def __init__(self, n_dirs: int = 4):
        super().__init__(name="teams", n_teams=2, n_slots=4,
                         controllers=("policy", "bots"), match_len=7,
                         best=1.0)
        self.n_dirs = n_dirs

    def bind(self, sizes):
        super().bind(sizes)
        self.cue = numeric_columns(sizes.vector, sizes.vector_kinds,
                                   self.n_dirs)
        stride = A.N_DIRS // self.n_dirs
        self.options = [_feet_option(d * stride) for d in range(self.n_dirs)]
        return self

    def roster(self, e, rng):
        held = np.zeros(self.n_teams * self.n_slots, dtype=bool)
        held[0] = True                           # team 0: one soldier
        held[self.n_slots:] = True               # team 1: four
        return held

    def start(self, e, rng):
        return {"right": 0}

    def observe(self, e, state, obs, full, valid, rng):
        d = int(rng.integers(self.n_dirs))
        state["right"] = self.options[d]
        for i in np.flatnonzero(valid):
            _onehot(obs, i, self.cue, d)
            obs["mask_feet"][i] = 0
            obs["mask_feet"][i, A.FEET_STAND] = 1
            obs["mask_feet"][i, self.options] = 1

    def pay(self, e, state, actions, valid, rng):
        u = len(valid)
        right = bool(valid[0] and actions[0, 0] == state["right"])
        r = np.zeros((u, len(REWARD_GROUPS)))
        r[0, G_OWN] = 1.0 if right else 0.0
        r[self.n_slots:, G_OWN] = np.where(valid[self.n_slots:],
                                           -1.0 if right else 0.0, 0.0)
        return r, np.zeros(u, bool), np.zeros(u, bool), False


TASKS = {"pointer": PointerTask, "argument": ArgumentTask,
         "weights": WeightTask, "scale": ScaleTask, "teams": TeamTask}


def make_env(task: Task, n_envs: int, seed: int) -> StandIn:
    """The stand-in for ``task``, the task bound to its sizes."""
    env = StandIn(task, n_envs, seed)
    task.bind(env.sizes)
    return env


def networks(env, seed: int, spec=None):
    """A fresh soldier's and value network for ``env``'s layout."""
    spec = spec or reduced(memory=False)
    sizes = layout_sizes(env.layout)
    torch.manual_seed(int(seed))
    return SoldierNet(spec, sizes), ValueNet(spec, sizes)


def score(env, weights) -> tuple:
    """``(paid . weights, decisions)`` of the learning teams so far."""
    return float(np.dot(env.paid, weights)), int(env.paid_decisions)


@torch.no_grad()
def random_score(task_cls, weights, n_envs: int = 64, steps: int = 64,
                 seed: int = 12345) -> float:
    """The mean reward per decision of a random player: every legal option
    of every head equally likely (``choosing.draw`` with ``explore`` 1)."""
    task = task_cls()
    env = make_env(task, n_envs, seed)
    soldier, _ = networks(env, seed)
    obs, _ = env.reset()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        flat = {k: obs[k].reshape((-1,) + obs[k].shape[2:])
                for k in SOLDIER_KEYS + A.MASK_KEYS}
        out, _ = soldier(as_tensors({k: flat[k] for k in SOLDIER_KEYS}))
        d = choosing.draw(out, choosing.masks_of(flat), gen, explore=1.0)
        actions = d.actions.numpy().reshape(env.n_envs, env.U, -1)
        obs, _, _, _, _ = env.step(actions)
    paid, n = score(env, weights)
    return paid / n


#: The stand-in farm the tasks run on: matches, and decisions per match per
#: round (256 learning decisions a round with one learner per match). The
#: trainer's numbers are the design's; this is only the farm's size, kept
#: small because on the processor the time per learning step grows with the
#: piece and the step size fixes how far one step goes.
TASK_ENVS = 8
TASK_DECISIONS = 32


def run_task(task_cls, seed: int, max_rounds: int, *,
             n_envs: int = TASK_ENVS, decisions: int = TASK_DECISIONS,
             threads: int = 4, random=None, on_round=None) -> dict:
    """Train on ``task_cls`` from seed ``seed`` until the mean over the
    last :data:`PASS_WINDOW` rounds reaches the pass line, or
    ``max_rounds`` (also the run's planned rounds). ``random`` a measured
    random score (else measured here). Returns the scores, the rounds used
    and the verdict."""
    torch.set_num_threads(threads)
    settings = Settings(planned_rounds=max_rounds, decisions=decisions)
    w = np.asarray(settings.weights)
    if random is None:
        random = random_score(task_cls, w)
    task = task_cls()
    env = make_env(task, n_envs, seed)
    soldier, value = networks(env, seed)
    trainer = Trainer(env, soldier, value, settings, seed=seed)
    line = random + PASS_SHARE * (task.best - random)
    scores = []
    passed = False
    last = score(env, w)
    for r in range(max_rounds):
        stats = trainer.round()
        now = score(env, w)
        scores.append((now[0] - last[0]) / max(1, now[1] - last[1]))
        last = now
        if on_round is not None:
            on_round(r, scores[-1], stats)
        if len(scores) >= PASS_WINDOW and \
                np.mean(scores[-PASS_WINDOW:]) >= line:
            passed = True
            break
    reached = float(np.mean(scores[-PASS_WINDOW:]))
    return {"task": task.name, "seed": seed, "passed": passed,
            "rounds": len(scores), "random": random, "best": task.best,
            "line": line, "reached": reached,
            "share": (reached - random) / (task.best - random),
            "scores": scores}


__all__ = ["PASS_SHARE", "PASS_WINDOW", "numeric_columns", "refresh_aim",
           "PointerTask", "ArgumentTask", "WeightTask", "ScaleTask",
           "TeamTask", "TASKS", "TASK_ENVS", "TASK_DECISIONS", "make_env",
           "networks", "score",
           "random_score", "run_task"]
