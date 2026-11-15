"""A scripted task for the stand-in (a harness, not collected): every
served row carries a MARK that says which match, which decision and which
row it is, so a test can tell a copy from a reference; and the matches
script the endings the trainer must keep apart.

Three marine teams of two roster places, teams 0 and 2 on ``policy``, team
1 on the bots; team 2 holds ONE soldier (its slot 1 is an empty place).

- env 0: team 0's slot 1 dies in its match's step 0 (dead, still in the
  match, still paid ``team``);
- env 1: team 2's records end in its match's step 1 while the match goes
  on;
- every match lasts ``match_len`` decisions (its clock), then restarts.

``own`` pays ``decision + 1`` and ``team`` pays 0.5 to every row the
environment pays (the stand-in zeroes empty places and closed records).
"""
from __future__ import annotations

import numpy as np

from agents.standin import Task
from agents.tasks import numeric_columns
from simulation.reward_v1 import REWARD_GROUPS

OWN = REWARD_GROUPS.index("own")
TEAM = REWARD_GROUPS.index("team")


def mark(episode: int, clock: int, row: int) -> float:
    """The mark a served row carries: match, decision, row."""
    return float(episode * 100 + clock) + row / 100.0


class ScriptTask(Task):
    def __init__(self, match_len: int = 3):
        super().__init__(name="script", n_teams=3, n_slots=2,
                         controllers=("policy", "bots", "policy"),
                         match_len=match_len, best=1.0)

    def bind(self, sizes):
        self.col = numeric_columns(sizes.vector, sizes.vector_kinds, 1)[0]
        return self

    def roster(self, e, rng):
        held = np.ones(6, dtype=bool)
        held[5] = False                  # team 2, slot 1: an empty place
        return held

    def start(self, e, rng):
        self._episode = getattr(self, "_episode", {})
        self._episode[e] = self._episode.get(e, -1) + 1
        return {"clock": 0, "episode": self._episode[e]}

    def observe(self, e, state, obs, full, valid, rng):
        for i in range(6):
            m = mark(state["episode"], state["clock"], i)
            obs["vector"][i, self.col] = m
            full["full_match"][i, 0] = m

    def pay(self, e, state, actions, valid, rng):
        clock = state["clock"]
        r = np.zeros((6, len(REWARD_GROUPS)))
        r[:, OWN] = clock + 1
        r[:, TEAM] = 0.5
        deaths = np.zeros(6, bool)
        rows_end = np.zeros(6, bool)
        if e == 0 and clock == 0:
            deaths[1] = True
        if e == 1 and clock == 1:
            deaths[4] = True
            rows_end[4:6] = True
        state["clock"] = clock + 1
        return r, deaths, rows_end, False


class NoisyTask(ScriptTask):
    """The scripted task with masks that change at every decision (random
    feet, hand and trigger options; a random set of aimable entries; a
    throw sometimes being wound up), so that masks taken from another step
    differ from the ones the choice was drawn under."""

    def observe(self, e, state, obs, full, valid, rng):
        super().observe(e, state, obs, full, valid, rng)
        from agents import fake
        from simulation import action_spec_v1 as A
        tp = fake._type_pos()
        E = obs["list_mask"].shape[1]
        for i in np.flatnonzero(valid):
            obs["mask_feet"][i] = rng.uniform(size=obs["mask_feet"].shape[1]) < 0.5
            obs["mask_feet"][i, A.FEET_STAND] = 1
            obs["mask_trigger"][i] = 1
            obs["mask_trigger"][i, 1] = rng.uniform() < 0.5
            pos = rng.permutation(E)[:int(rng.integers(1, 6))]
            obs["list_mask"][i] = 0
            obs["list_mask"][i, pos] = 1
            obs["list"][i, pos, tp] = 1
            aim = obs["mask_aim"][i]
            aim[...] = 0
            if rng.uniform() < 0.15:
                aim[A.AIM_KEEP] = 1
            else:
                aim[:A.AIM_KEEP] = 1
                aim[A.AIM_TOKENS + pos] = 1
            hand = obs["mask_hand"][i]
            hand[...] = rng.uniform(size=hand.shape[0]) < 0.5
            hand[A.HAND_NOTHING] = 1
            hand[A.HAND_FLASHLIGHT] = 0
