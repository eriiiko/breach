"""agents/settings.py -- the trainer's numbers, in ONE record (rung 2 note
2b, 1; design 4.4).

Every number the trainer uses is a field here, its default the design's
starting value; nothing is typed anywhere else. Each is the start of a
sweep (design 4.4 says where each comes from). The reward group weights
are rung 2's: outcome 1, own 1, team 0, friendly fire 0, order 0 (design
4.3), in the environment's group order (``reward_v1.REWARD_GROUPS``).

The training file that will carry these numbers on disk is patch 2c's;
here the record is built in code.
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
    #: The reward group weights, in ``REWARD_GROUPS`` order.
    weights: tuple = field(default_factory=_rung2_weights)

    def __post_init__(self):
        if len(self.weights) != len(REWARD_GROUPS):
            raise ValueError(f"settings: {len(self.weights)} group weights "
                             f"for the groups {REWARD_GROUPS}")
        if int(self.planned_rounds) < 1:
            raise ValueError(f"settings: planned_rounds "
                             f"{self.planned_rounds} < 1")
        if self.lump < 1 or self.piece < self.lump:
            raise ValueError(f"settings: a piece ({self.piece}) is worked "
                             f"through in lumps ({self.lump})")

    def step_size_at(self, round_index: int) -> float:
        """Adam's step size in round ``round_index`` (0 = the first):
        falling in a straight line to zero over the planned rounds."""
        left = 1.0 - float(round_index) / float(self.planned_rounds)
        return self.step_size * max(0.0, left)


__all__ = ["RUNG2_WEIGHTS", "Settings"]
