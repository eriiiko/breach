"""agents/settings.py -- the trainer's numbers, in ONE record (rung 2 note
2b, 1; design 4.4).

Every number the trainer uses is a field here, its default the design's
starting value; nothing is typed anywhere else. Each is the start of a
sweep (design 4.4 says where each comes from). The reward group weights
are rung 2's: outcome 1, own 1, team 0, friendly fire 0, order 0 (design
4.3), in the environment's group order (``reward_v1.REWARD_GROUPS``).

On disk these numbers are a run's training file
(:mod:`agents.training_file`, patch 2c), one key per field.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agents import _game  # noqa: F401
from simulation.reward_v1 import REWARD_GROUPS

#: Rung 2's reward group weights (design 4.3), by group name.
RUNG2_WEIGHTS = {"outcome": 1.0, "own": 1.0, "team": 0.0,
                 "friendly_fire": 0.0, "order": 0.0}


def _rung2_weights() -> tuple:
    return tuple(float(RUNG2_WEIGHTS[g]) for g in REWARD_GROUPS)


@dataclass(frozen=True)
class Settings:
    """The numbers of one training run (design 4.4)."""
    #: Rounds the run is planned for: the step size falls in a straight
    #: line to zero over them. No default: every run says its own.
    planned_rounds: int
    #: Decisions per match per round (T; design 4.4: 64 matches x 128).
    decisions: int = 128
    #: The look-ahead (discount): the range's 0.99 (the duel's is 0.995).
    gamma: float = 0.99
    #: The blend of short and long grading (lambda).
    lam: float = 0.95
    #: The clip, on the probability of the whole decision.
    clip: float = 0.2
    #: Passes over a round's batch.
    passes: int = 4
    #: The early stop: the distance moved past which the soldier's network
    #: takes no further step this round.
    max_distance: float = 0.02
    #: Decisions per piece (one step of each network).
    piece: int = 2048
    #: Decisions per lump (gradients added up to the piece's step).
    lump: int = 512
    #: Adam's step size at the first round, and its epsilon.
    step_size: float = 3e-4
    adam_eps: float = 1e-5
    #: The push toward variety, per counted head (its share of its most).
    variety: float = 0.01
    #: The limit on one update's size (gradient norm), per network.
    max_grad_norm: float = 0.5
    #: The value targets' scale: the share a round moves the mean and the
    #: mean square, and the floor under the spread.
    scale_share: float = 0.05
    scale_floor: float = 0.01
    #: The floor added to the grades' spread when they are brought to a
    #: common size (keeps a piece of equal grades from dividing by zero).
    grade_floor: float = 1e-8
    #: The live check's tolerance (the probability and the distance).
    live_tolerance: float = 1e-4
    #: The memory's stretch (design 4.5): with the soldier's memory on, the
    #: unit learning cuts, shuffles and replays -- a row's ``stretch``
    #: decisions in order from the memory stored at the first. ``decisions``,
    #: ``piece`` and ``lump`` are then whole numbers of stretches (refused
    #: by name when the trainer is built). Not read with the memory off.
    stretch: int = 16
    #: The value-first rounds (design 6, arm B; note 2d, 7): the first rounds
    #: of a run in which only the value network (and its scale) learns while
    #: the soldier's network plays and takes no step. The step-size plan is
    #: not shifted by them: it falls over ``planned_rounds`` as always.
    value_first_rounds: int = 0
    #: The reward group weights, in ``REWARD_GROUPS`` order.
    weights: tuple = field(default_factory=_rung2_weights)
    #: The run's wall-clock stop (hours): a run ends at ``planned_rounds``
    #: or here, whichever comes first (note 2c, 4). None: no wall stop.
    #: The step size still falls over the PLANNED rounds, so a plan the
    #: wall cuts short never reaches its low step sizes.
    wall_hours: float | None = None
    #: Saving a run (note 2d, 9; the training file's ``[save]`` table):
    #: ``last.pt`` is written every ``save_every`` rounds and at the end;
    #: ``round_<n>.pt`` is kept every ``keep_every`` rounds (0: none).
    save_every: int = 10
    keep_every: int = 0

    def __post_init__(self):
        if len(self.weights) != len(REWARD_GROUPS):
            raise ValueError(f"settings: {len(self.weights)} group weights "
                             f"for the groups {REWARD_GROUPS}")
        if int(self.planned_rounds) < 1:
            raise ValueError(f"settings: planned_rounds "
                             f"{self.planned_rounds} < 1")
        if self.wall_hours is not None and not float(self.wall_hours) > 0:
            raise ValueError(f"settings: wall_hours {self.wall_hours} must "
                             f"be above 0 (or None: no wall stop)")
        if int(self.stretch) < 1:
            raise ValueError(f"settings: stretch {self.stretch} < 1")
        if int(self.value_first_rounds) < 0:
            raise ValueError(f"settings: value_first_rounds "
                             f"{self.value_first_rounds} < 0")
        if int(self.save_every) < 1:
            raise ValueError(f"settings: save_every {self.save_every} < 1")
        if int(self.keep_every) < 0:
            raise ValueError(f"settings: keep_every {self.keep_every} < 0")
        if self.lump < 1 or self.piece < self.lump:
            raise ValueError(f"settings: a piece ({self.piece}) is worked "
                             f"through in lumps ({self.lump})")

    def step_size_at(self, round_index: int) -> float:
        """Adam's step size in round ``round_index`` (0 = the first):
        falling in a straight line to zero over the planned rounds."""
        left = 1.0 - float(round_index) / float(self.planned_rounds)
        return self.step_size * max(0.0, left)


def _trainer_default(name: str):
    """A :class:`Settings` field's default: the copy starts from the
    trainer's numbers (note 2e, 5), never a second typed copy of them."""
    from dataclasses import fields
    return next(f.default for f in fields(Settings) if f.name == name)


@dataclass(frozen=True)
class CopySettings:
    """The numbers of one copying run (rung 2 note 2e, section 5; design
    3.2). On disk: the training file's ``[copy]`` table
    (:func:`agents.training_file.parse_copy`)."""
    #: Planned passes over the training records; the step size falls in a
    #: straight line to zero over them. The best pass's weights are kept.
    passes: int = 10
    #: Decisions per step (with the memory on: lanes x stretch).
    batch: int = _trainer_default("piece")
    #: Decisions per lump (gradients added up to the step; the 8 GB card).
    lump: int = _trainer_default("lump")
    step_size: float = _trainer_default("step_size")
    adam_eps: float = _trainer_default("adam_eps")
    max_grad_norm: float = _trainer_default("max_grad_norm")
    #: The feet's floor (design 3.2): the bot's choice is trained toward
    #: ``1 - feet_floor``, the OTHER LEGAL feet options share
    #: ``feet_floor``. The other heads have none.
    feet_floor: float = 0.02
    #: The memory's stretch (design 4.5; the trainer's).
    stretch: int = _trainer_default("stretch")
    #: The loader: processes, and the shuffle buffer in matches.
    loader_workers: int = 4
    buffer_matches: int = 30
    #: The seed of the loader's order and of the network's first weights.
    seed: int = 0
    #: The previous-choice inputs blanked at the network's input (design
    #: 3.3, "does it lean on its own past?"): in training and measuring.
    blank_prev: bool = False

    def __post_init__(self):
        if int(self.passes) < 1:
            raise ValueError(f"copy settings: passes {self.passes} < 1")
        if self.lump < 1 or self.batch < self.lump:
            raise ValueError(f"copy settings: a batch ({self.batch}) is "
                             f"worked through in lumps ({self.lump})")
        if not 0.0 <= float(self.feet_floor) < 1.0:
            raise ValueError(f"copy settings: feet_floor {self.feet_floor} "
                             f"outside [0, 1)")
        if int(self.stretch) < 1 or int(self.loader_workers) < 1 \
                or int(self.buffer_matches) < 1:
            raise ValueError("copy settings: stretch, loader_workers and "
                             "buffer_matches are at least 1")

    def step_size_at(self, step: int, steps: int) -> float:
        """Adam's step size at ``step`` of ``steps`` planned: a straight line
        to zero."""
        return self.step_size * max(0.0, 1.0 - float(step) / float(steps))


__all__ = ["RUNG2_WEIGHTS", "Settings", "CopySettings"]
