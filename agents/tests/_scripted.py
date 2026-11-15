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
