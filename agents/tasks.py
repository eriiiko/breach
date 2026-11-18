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

With the soldier's memory on (patch 2d.1; design 8's last rows; note 2d, 6
and 7):

| task | the trainer fails it if |
|---|---|
| :class:`Recall12`: a cue at a match's first decision, the answer asked for 12 decisions later; soldiers die mid-match, matches end mid-batch at lengths that are not whole stretches | the memory, the stretches, the dead rows or the endings are wrong; the memory is reset at a stretch's start (the trap) |
| :class:`Recall24`: the same with 24 between | it passes with stretches of 16 (the memory's limit is not where design 4.5 says); it must pass with 32 |
| :func:`good_start`: the pointer task's taught soldier with a fresh value network, 20 rounds | learning wears down a good start (the copy's danger) |

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

    def measure(self, env, weights) -> tuple:
        """``(paid . weights, decisions)`` of the learning teams so far: the
        score is per learning decision of a living soldier (the stand-in's
        own books)."""
        return float(np.dot(env.paid, weights)), int(env.paid_decisions)


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
    """At every decision the hand's mask allows "nothing" and ONE verb that
    takes an argument, a plant or the first throw, drawn by the task; the
    argument's right value depends on that verb AND on the aim just chosen
    (a plant: timer 1; a throw: "onto" at an entry, reach 3 at a token).
    "own" pays half for using the verb and the other half when its argument
    is right too; "nothing" pays nothing. The trigger has one legal option
    on most decisions (a head unused there), and the argument does not
    count whenever "nothing" is drawn.

    THE MASK SAYS WHICH VERB, NOT A CUE (2b's findings; three forms with a
    cue were measured and dropped). A cue made the task one of learning two
    answers of unequal worth from a near-even start, and that is where this
    kind of trainer is weak whatever its code: the answer that pays sooner
    crowds the other out before the cue is read.

    - Half paid for any right argument, "nothing" (no argument) always
      right: "always nothing" earned 0.667 with no cue read, and the policy
      sat there for good.
    - Only a right verb paid, "nothing" still asked for on a third of the
      decisions: by round 60 it was chosen on 96 % of its own cues and the
      other two verbs on 10 % of theirs; 0.75 of the way after 400 rounds.
    - Only plant and throw asked for: plant's one argument is learnt first,
      plant then pays double, and throw fell to 3 % of its own cues by round
      60. Two seeds came back and passed (176 and 123 rounds); the third sat
      at "always plant" (0.47) for 300 rounds.

    None of that is one of the traps this task is for, so the task no
    longer leans on it: the masks, as in the game, say what the hand may do,
    and the task asks what its row in design 8 asks. The crowding-out is
    recorded in the plan's as-built for 2b (it is why rung 2 starts from the
    copy, and what to look for when a verb dies out in a real run)."""

    VERBS = (A.HAND_PLANT, A.HAND_THROW0)

    def __init__(self, k: int = 3, p_trigger_unused: float = 0.8):
        super().__init__(name="argument", match_len=7, best=1.0)
        self.k = k
        self.p_unused = p_trigger_unused

    def start(self, e, rng):
        return {"verb": np.zeros(self.n_teams * self.n_slots, dtype=int)}

    def observe(self, e, state, obs, full, valid, rng):
        E = obs["list_mask"].shape[1]
        tp = fake._type_pos()
        for i in np.flatnonzero(valid):
            verb = self.VERBS[int(rng.integers(len(self.VERBS)))]
            state["verb"][i] = verb
            pos = rng.permutation(E)[:self.k]
            obs["list_mask"][i] = 0
            obs["list_mask"][i, pos] = 1
            obs["list"][i, pos, tp] = O.TYPE_SEEN
            refresh_aim(obs, i)
            obs["mask_hand"][i] = 0
            obs["mask_hand"][i, [A.HAND_NOTHING, verb]] = 1
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
        arg_ok = arg == self.right_arg(hand, aim)
        used = valid & (hand == state["verb"])
        r = np.zeros((u, len(REWARD_GROUPS)))
        r[:, G_OWN] = np.where(used, 0.5 + 0.5 * arg_ok, 0.0)
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
    for walking the cued way and every soldier of team 1 the opposite, ten
    times over. Four directions, for the reason given at
    :class:`WeightTask`.

    TEN TIMES OVER, so that the trap can be told (2b's finding). Team 1's
    rows are not driven by the trainer, so their pay has nothing to do with
    what it drew for them: stored and graded, they are noise, not a wrong
    lesson. At the learner's own size (four rows of -1 beside one of +1)
    that noise only slowed the trap run: 0.75 of the way in 300 rounds,
    against a clean pass in 123 -- too near the line to call the trap
    caught. A trainer that never reads those rows cannot see their size."""

    #: How many times the learner's pay each soldier of team 1 is paid,
    #: with the opposite sign.
    OPPOSITE = 10.0

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
                                           -self.OPPOSITE if right else 0.0,
                                           0.0)
        return r, np.zeros(u, bool), np.zeros(u, bool), False


class RecallTask(_Base):
    """THE MEMORY TASKS (note 2d, 6; design 8's last two rows). At a match's
    first decision a cue (one of four directions) is shown in the soldier's
    vector for that one decision; for the next ``gap - 1`` decisions the cue
    columns show nothing (0); at decision ``gap`` the feet must walk the
    cued direction, and only that decision is paid ("own" 1). The feet may
    always stand or walk one of the four ways (a random player answers
    right one time in five).

    THE SCORE is per ANSWER, not per decision (:meth:`measure`): the "own"
    paid over the answers a living learning soldier gave. Per decision the
    best would depend on the matches' lengths and the deaths.

    WHAT IS IN IT for the traps of note 2d, 5: two soldiers per match;
    slot 1 dies at a drawn decision in half the matches (before or after
    its answer; a dead row stays in the match, served zeros, paid nothing,
    still stepped by the collector); every match lasts a drawn ``gap + 1``
    to ``gap + 7`` decisions, never a multiple of 16 (so of 32 neither),
    so matches end and restart inside rounds and across a round's edge and
    start anywhere inside a stretch.

    WHAT THE STRETCH ALLOWS (design 4.5): the gradient stops at a stretch's
    start, so the cue can be LEARNT to be kept only when the cue and its
    answer fall in one stretch -- the cue at one of the stretch's first
    ``stretch - gap`` places. ``recall12`` with stretches of 16 has that on
    a quarter of its matches; ``recall24`` with 16 never (it must FAIL),
    with 32 on a quarter (it must pass).

    FINDING (2d.1, 2026-11-17): recall12 was NOT learnt with the design's
    numbers. Seed 0, stretches of 16: 220 rounds on the tasks' farm (8
    matches x 32 decisions) and 140 on a farm of 32 matches near the random
    player (0.25 is "never stand", reachable without the cue); with the
    answer 4 decisions after the cue, 0.11 of the way after 300 rounds;
    with the answer AT the cue (gap 0, memory on) 0.8 at round 180.

    WHAT 2d.1b FOUND AND FIXED (2026-11-17), in two parts:

    1. THE NETWORK (fixed). Measured by a supervised fit (the right feet at
       the answer, through ``forward_sequence``, Adam 1e-3, batches of 256,
       each window from its match's first decision): the cue's share of an
       activation is about the same from the vector reader to the feet
       (0.12 to 0.15 of its size: the 4 cue columns against each soldier's
       fixed scenery), so no layer washes it out -- but the cell's output
       starts about 8 x smaller than the join it reads (rms 0.13 against
       1.0; the chrono start makes most units keep), and the heads read it
       raw. Without memory the fit reaches 0.89 at step 80; memory on, gap
       0: 0.30 at step 80; gap 12: 0.29 after 300 steps. The heads now read
       the memory through a normalising layer (``SoldierNet.memory_norm``):
       gap 0 0.94 at step 80, gap 12 0.83 at step 180 and 0.99 at 210. By
       reward, gap 0 with memory on now passes in 87 rounds (was 0.8 at
       round 180; the weights task without memory: 85). The chrono start
       stays: measured with each window replayed under all four cues, it
       keeps 7 x more of the cue's mark after 12 decisions and 27 x more
       after 24 than PyTorch's start (the mark kept about 0.82 x a decision
       against 0.67 x). In the supervised fit at gap 12 both starts learnt
       alike (0.83 at step 180 / 150): it is kept for the longer gaps.
    2. THE FARM (NOT changed: the lead's call). With the network fixed,
       recall12 at stretch 16 on the tasks' farm is still flat after 370
       rounds (seed 0, planned 600; every tenth round 0.11 to 0.46, no
       trend) and on 32 matches after 100 rounds (0.19 to 0.28). Counted
       per round on the tasks' farm (20 rounds): gap 0 gives 130 answers,
       every one learnable; recall12 25 answers, of which 8.5 have the cue
       in the answer's stretch (the only ones that teach the keeping);
       recall24 at 32 gives 14 answers and 3.4 learnable, at 16 none. So
       recall12 sees 15 x and recall24 38 x fewer teaching answers a round
       than gap 0, which needs 87 rounds, and the supervised fit says
       holding the cue 12 decisions costs about 2 x the steps of reading it
       at once: on the order of a thousand rounds or more, past any test's
       budget at 4 s a round (under load). What a pass needs is a farm
       giving more teaching answers a round (more matches, more decisions
       a round) or a larger budget, decided by the lead and written here
       with its measurement.

    WHAT 2d.1c MEASURED (2026-11-18; FINDING, still open). Two candidates,
    seed 0, stretches of 16, planned 600 rounds, the machine shared with
    another agent's builds (4 to 24 s a round as its load came and went):

    - B, THE STRETCHES START AT A MATCH'S FIRST DECISION (the learner,
      ``Learner._stretches``, kept on the branch): teaching answers a round
      on the tasks' farm 8.5 -> 17.3. By reward: flat for all 600 rounds
      (the means of ten rounds 0.18 to 0.31; last five 0.19), 4.4 s a
      round.
    - A with B, the design's 128 decisions a round (``Settings.decisions``'
      default; the tasks' farm keeps 32 for speed): teaching answers a
      round 95 (fixed cuts at 128: 28.7; 32 matches x 32 decisions with
      fixed cuts, 2d.1b's farm, 34.6; gap 0 on the tasks' farm: 130), 15 to
      24 s a round. By reward: flat for all 600 rounds (the means of 25
      rounds 0.22 to 0.27; 19 s a round on average; an earlier run of the
      same, stopped at 150 rounds, 0.21 to 0.30). A alone (fixed cuts) gives
      fewer teaching answers than A with B at the same cost, and 2d.1b ran
      it on 32 matches for 100 rounds flat: not run again.

    The diagnosis (A with B, every 25 rounds, over the ~95 answers of the
    round's batch replayed from their match's start): the gradient of the
    right answer's log-probability reaching the memory after the cue's
    step is 0.44 to 0.66 of its size at the step before the answer (no
    vanishing: the chrono start holds); the cue read back from the memory
    at the answer by a linear probe stays near chance (0.10 to 0.43
    against 0.25, noisy at ~47 held-out answers, no trend); the feet's
    variety at the answer FALLS from 1.61 (all five even) to 1.04 to 1.4
    while the chance of the right answer stays 0.23 to 0.28: the soldier
    grows sure of a direction that does not depend on the cue; the value's
    error at the answer stays about the pay's own spread (0.36 to 0.50
    against a 0/1 pay near 0.25); no early stop in any round (8 steps a
    round). In expectation the answer's gradient is only the right answer's
    chance (0.2) times the supervised one, and the 15 unpaid decisions a
    match share the feet head and the grades' common size: the cue's
    signal is below the noise at these numbers. Neither candidate is a
    pass; nothing was tuned, loosened or shortened.

    FOLLOW-UP (2d.1c, the lead's two measurements; B on, the tasks' farm,
    stretch 16, seed 0, planned and at most 300 rounds, stopped at a pass):

    | gap | teaching answers a round | by reward |
    |---|---|---|
    | 0 | 130 | PASS in 93 rounds |
    | 1 | 100 | flat, last 30 rounds 0.25 |
    | 2 | 76 | flat, 0.25 |
    | 4 | 50 | flat, 0.26 |
    | 8 | 26 | flat, 0.24 |

    A CLIFF between gap 0 and gap 1 (at gap 1 the teaching answers are
    near gap 0's), not a growth with the gap. The noise diagnostic (a
    scratch patch, never a fix): (a) the soldier's loss from the ANSWERS
    only -- every other decision's clipped objective and variety masked,
    and with them the distance, the live check's first clause and the
    grades' common size taken over the answers; the value's loss, the
    grading and the scale untouched: gap 1 PASSES in 159 rounds (means of
    25 rounds 0.21, 0.26, 0.29, 0.52, 0.73, 0.74, 0.82); gap 12 flat on the
    tasks' farm for 300 rounds (0.22 to 0.29) and on 128 decisions a round
    for 150 (0.24 to 0.26; gap 1 needed 75 rounds before it rose, so 150 is
    not conclusive). (b) every decision keeps its grade and its loss terms,
    the grades brought to a common size with the mean and spread of the
    answers' grades over the batch: gap 1 partly learnt, 0.25 for 175
    rounds, then 0.37 to 0.48 at 300. Read: as soon as one decision comes
    before the answer in its match (gap >= 1), the answer's luck flows back
    through the grades onto decisions that cannot cause it, on the same
    feet head, and drowns the cue; at gap 0 nothing comes before the answer.
    At gap 12 the drowning is not the only obstacle within these budgets.

    SECOND FOLLOW-UP (scratch task shapes, never committed; B on, the
    tasks' farm, stretch 16, seed 0, at most 300 rounds):

    | shape | gap | by reward (means of 25 rounds) |
    |---|---|---|
    | "hold": every decision 1..gap paid for the cued way | 12 | flat, 0.22 to 0.25 (last 25: 0.25) |
    | "go": a column shows 1 at the answer only | 12 | flat, 0.20 to 0.27 (last 25: 0.24) |

    Neither shape passes gap 12; "hold" at gap 4 and "go" at gap 1 were
    still running when this was written (the lead's runs since: "hold" gap
    4 flat at 0.25 for 300 rounds; "go" gap 1 0.41 to 0.43, no pass;
    "hold" gap 12 flat at 0.25 from round 75 to round 3000).

    THIRD FOLLOW-UP (2d.1c; B on, the tasks' farm, stretch 16, seed 0;
    scratch scripts, measurements only):

    E1, the direction. On 20 fresh batches each, the soldier's gradient as
    the learner forms it (its first piece, before clipping: g_rl), the same
    with the answers only (g_ans), against the gradient of the mean
    log-probability of the right feet at the answers (g_sup) and of the
    mean probability (g_obj, what the pay rewards). Cosines, mean +- s.e.
    over batches (all weights), and of the 20-batch means:

    | gap, network | g_rl.g_obj per batch | of the means | g_ans.g_obj per batch | of the means | |mean|^2/var g_rl, g_ans |
    |---|---|---|---|---|---|
    | 12, fresh | +0.16 +- 0.04 | +0.48 | +0.43 +- 0.04 | +0.91 | 0.06, 0.18 |
    | 12, 100 rounds (plateau) | +0.11 +- 0.11 | +0.83 | +0.43 +- 0.09 | +0.48 | 0.09, 0.13 |
    | 1, fresh | +0.44 +- 0.05 | +0.93 | +0.65 +- 0.03 | +0.98 | 0.23, 0.55 |
    | 1, 100 rounds (plateau) | +0.34 +- 0.11 | +0.26 | +0.45 +- 0.10 | +0.46 | 0.02, 0.03 |

    (Fresh, g_sup and g_obj coincide; at the plateau g_sup's mean points
    away from the soldier's favourite direction and its cosines with g_rl
    and g_ans fall to about 0.1, and the means' to -0.45 for g_ans: g_obj
    is the honest comparison.) The answers' grades have the right sign
    (common size, right +2.9 to +3.5, wrong -0.2 to -0.6). Per group at the
    plateau (gap 12, g_ans against g_obj, of the means): memory cell 0.75,
    memory_norm 0.98, join 0.75, feet head 0.49.

    E2, a round at the plateau (gap 12, rounds 100 to 119): all 4 planned
    steps run every round, the early stop never fires; the distance after
    the last step is -0.002 to +0.004 (mean |.| 0.001, max_distance 0.02);
    the gradient's norm before clipping 0.13 to 0.55 (mean 0.35; 5 % of the
    steps clipped at 0.5); step size 2.0e-4 to 1.8e-4. Of the distance,
    feet 52 %, trigger 23 %, aim 18 %, hand 7 %: half goes to heads this
    task does not pay.

    E3, does reward keep a memory that exists. Supervised fit of recall12
    (32 matches x 64 decisions, Adam 1e-3) until p(right) > 0.9, then that
    soldier with a FRESH value network to the trainer for 200 rounds: seed
    0 (supervised 0.91 at step 630): round 1 0.92, 25-round means 0.91,
    0.93, 0.96, 0.96, 0.97, 0.98, 0.98, 0.97: KEPT and improved (4.2 s a
    round). Seed 1: the supervised fit stalled at 0.71 to 0.76 for 1 400
    steps and was handed over at step 2 000 (not above 0.9); by reward
    round 1 0.67, 25-round means 0.68, 0.74, 0.70, 0.71, 0.68, 0.68,
    0.67, 0.72: held, neither lost nor raised to the line.

    Reading: no sign fault (E1: the grades and the mean direction are
    right); the learner is not stopped (E2: every step runs, at about 5 %
    of its distance budget); reward keeps and sharpens a memory that exists
    (E3). The plateau is the noise case: per batch the reward gradient is
    mostly noise (|mean|^2 / variance 0.02 to 0.18, so 5 to 50 batches
    before the signal stands out), and the steps are small, so a cue-blind
    start never finds the memory."""

    N_DIRS = 4

    def __init__(self, gap: int = 12):
        self.gap = int(gap)
        self.lengths = [n for n in range(self.gap + 1, self.gap + 8)
                        if n % 16]
        super().__init__(name=f"recall{self.gap}", n_teams=1, n_slots=2,
                         controllers=("policy",),
                         match_len=max(self.lengths), best=1.0)
        self.answers = 0

    def bind(self, sizes):
        super().bind(sizes)
        self.cue = numeric_columns(sizes.vector, sizes.vector_kinds,
                                   self.N_DIRS)
        stride = A.N_DIRS // self.N_DIRS
        self.options = [_feet_option(d * stride) for d in range(self.N_DIRS)]
        return self

    def start(self, e, rng):
        u = self.n_teams * self.n_slots
        dies = int(rng.integers(1, self.gap + 4)) if rng.uniform() < 0.5 \
            else -1
        return {"clock": 0, "length": int(rng.choice(self.lengths)),
                "right": np.array([self.options[int(rng.integers(
                    self.N_DIRS))] for _ in range(u)]),
                "dies": dies}

    def observe(self, e, state, obs, full, valid, rng):
        for i in range(len(valid)):
            obs["vector"][i, self.cue] = 0.0
        for i in np.flatnonzero(valid):
            if state["clock"] == 0:
                d = self.options.index(int(state["right"][i]))
                _onehot(obs, i, self.cue, d)
            obs["mask_feet"][i] = 0
            obs["mask_feet"][i, A.FEET_STAND] = 1
            obs["mask_feet"][i, self.options] = 1

    def pay(self, e, state, actions, valid, rng):
        u = len(valid)
        clock = state["clock"]
        r = np.zeros((u, len(REWARD_GROUPS)))
        if clock == self.gap:
            right = valid & (actions[:, 0] == state["right"])
            r[:, G_OWN] = np.where(right, 1.0, 0.0)
            self.answers += int(valid.sum())
        deaths = np.zeros(u, bool)
        if clock == state["dies"]:
            deaths[1] = True
        state["clock"] = clock + 1
        over = state["clock"] >= state["length"]
        return r, deaths, np.zeros(u, bool), over

    def measure(self, env, weights) -> tuple:
        """``(paid . weights, answers)``: the score per answer given."""
        return float(np.dot(env.paid, weights)), int(self.answers)


class Recall12(RecallTask):
    """``recall12``: the answer 12 decisions after the cue."""

    def __init__(self):
        super().__init__(12)


class Recall24(RecallTask):
    """``recall24``: the answer 24 decisions after the cue."""

    def __init__(self):
        super().__init__(24)


TASKS = {"pointer": PointerTask, "argument": ArgumentTask,
         "weights": WeightTask, "scale": ScaleTask, "teams": TeamTask}
#: The memory tasks (run with the soldier's memory on, note 2d, 6).
MEMORY_TASKS = {"recall12": Recall12, "recall24": Recall24}


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
    """``(paid . weights, count)`` of the learning teams so far: per
    decision, or what the task measures per (:meth:`_Base.measure`)."""
    return env.task.measure(env, weights)


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
             threads: int = 4, random=None, on_round=None,
             memory: bool = False, stretch: int | None = None) -> dict:
    """Train on ``task_cls`` from seed ``seed`` until the mean over the
    last :data:`PASS_WINDOW` rounds reaches the pass line, or
    ``max_rounds`` (also the run's planned rounds). ``random`` a measured
    random score (else measured here); ``memory`` the soldier's memory on,
    with stretches of ``stretch`` (else the design's). Returns the scores,
    the rounds used, the verdict and the trained soldier."""
    torch.set_num_threads(threads)
    extra = {} if stretch is None else {"stretch": int(stretch)}
    settings = Settings(planned_rounds=max_rounds, decisions=decisions,
                        **extra)
    w = np.asarray(settings.weights)
    if random is None:
        random = random_score(task_cls, w)
    task = task_cls()
    env = make_env(task, n_envs, seed)
    soldier, value = networks(env, seed, reduced(memory) if memory else None)
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
            "scores": scores, "soldier": trainer.learner.soldier}


#: The rounds the "already good start" is learnt on after its start, and
#: the share of the start's score its lowest round must keep (design 8's
#: row; note 2d, 7).
GOOD_START_ROUNDS = 20
GOOD_START_KEEP = 0.9


def good_start(soldier, seed: int, *, planned_rounds: int,
               value_first_rounds: int, variety: float | None = None,
               rounds: int = GOOD_START_ROUNDS, n_envs: int = TASK_ENVS,
               decisions: int = TASK_DECISIONS, threads: int = 4) -> dict:
    """THE "ALREADY GOOD START" TASK (note 2d, 7; design 6 arm B, 8): a
    copy of ``soldier``, taught on the pointer task until it passed, starts
    a run with a FRESH value network and a fresh scale and learns
    ``rounds`` rounds with the design's numbers (``value_first_rounds`` and
    ``variety`` as given). Returns the score per round (each played by the
    weights the round started with: round 1's is the start's), the start,
    the lowest, the end (the last round's) and the lowest's share of the
    start.

    MEASURED (2d.1, 2026-11-17; the pointer soldier taught on seed 0 until
    it passed, 37 rounds, 0.82; planned rounds 150; start / lowest / end):

    | run | seed 0 | seed 1 | seed 2 |
    |---|---|---|---|
    | 5 value-first rounds (the gate) | 0.832 / 0.832 / 0.941 | 0.742 / 0.680 / 0.891 | 0.695 / 0.695 / 0.949 |
    | no value-first rounds | 0.832 / 0.832 / 0.949 | 0.742 / 0.734 / 0.934 | 0.695 / 0.695 / 0.945 |
    | 5 value-first, variety x 10 | 0.832 / 0.832 / 0.930 | 0.742 / 0.680 / 0.887 | 0.695 / 0.695 / 0.922 |

    Neither the missing value-first rounds nor ten times the variety push
    wore the start down: every run ends 0.11 to 0.25 above its start. The
    one dip (seed 1, 0.680 = 0.92 of the start) falls in round 6 of the
    runs WITH value-first rounds, while the soldier was still frozen: it is
    the noise of one round's score (256 decisions, about 0.03), not
    learning. Two limits of the measurement: the taught start is the pass
    line (0.82 on its own rooms, 0.70 to 0.74 on other seeds' scenery), not
    the best, so a trainer has room to climb and a wearing-down of a start
    at the top is not what this sees; and the gate's 0.9 is about three
    rounds' noise below a single-round start."""
    import copy
    torch.set_num_threads(threads)
    extra = {} if variety is None else {"variety": float(variety)}
    settings = Settings(planned_rounds=planned_rounds, decisions=decisions,
                        value_first_rounds=value_first_rounds, **extra)
    w = np.asarray(settings.weights)
    env = make_env(PointerTask(), n_envs, seed)
    _, value = networks(env, seed)
    trainer = Trainer(env, copy.deepcopy(soldier), value, settings,
                      seed=seed)
    scores = []
    last = score(env, w)
    for _ in range(rounds):
        trainer.round()
        now = score(env, w)
        scores.append((now[0] - last[0]) / max(1, now[1] - last[1]))
        last = now
    start, lowest = scores[0], min(scores)
    return {"seed": seed, "value_first_rounds": value_first_rounds,
            "variety": settings.variety, "scores": scores, "start": start,
            "lowest": lowest, "end": scores[-1],
            "kept": lowest / start if start else float("nan")}


__all__ = ["PASS_SHARE", "PASS_WINDOW", "numeric_columns", "refresh_aim",
           "PointerTask", "ArgumentTask", "WeightTask", "ScaleTask",
           "TeamTask", "RecallTask", "Recall12", "Recall24", "MEMORY_TASKS",
           "GOOD_START_ROUNDS", "GOOD_START_KEEP", "good_start",
           "TASKS", "TASK_ENVS", "TASK_DECISIONS", "make_env",
           "networks", "score",
           "random_score", "run_task"]
