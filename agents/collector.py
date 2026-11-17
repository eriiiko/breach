"""agents/collector.py -- the collector (rung 2 note 2b, 3; design 4.1,
4.2): takes the soldier's weights, plays T decisions in each of the N
matches, returns the batch.

Per step, for all learning rows in one pass:

1. the learning rows of the served arrays (through ``env.team_rows``; the
   full view from ``info``, the same rows) are COPIED into the batch's slot
   for this step -- the environment overwrites its arrays at its next step;
2. the soldier's network reads that slot (without gradients) and
   ``choosing.draw`` draws the five choices under the slot's masks: the
   network acts on exactly the arrays that are kept for learning;
3. every row the trainer does not drive is sent the legal default (stand,
   aim where I walk -- or "keep" while a throw is wound up, the one legal
   aim then --, no trigger, hand nothing, 0); a learning row that is dead
   or empty carries one legal option per head, so its draw IS the default;
4. the step; ``info["masked_choices"]`` on the driven living rows must be
   0, anything else stops the run naming the env and the row;
5. the step's rewards and endings are stored, and the row bookkeeping
   moves: a row's story ends when its match ends or its records end
   (``rows_ended``); a match that ended was restarted inside the step, so
   the next decision is that row's ``first``.

Between rounds nothing is reset: the next round goes on from the matches as
they stand. After the last step the view the value network reads is copied
once (``last_*``) for the value after the last stored decision.

The collector holds its OWN copy of the soldier's network; the learner
hands it weights (:meth:`Collector.set_weights`) and it hands back a batch:
one hand-over each way (patch 2d adds the header).
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from agents import _game  # noqa: F401
from agents.batch import (FULL_STORED, LAST_FULL, LAST_OBS, OBS_STORED,
                          Batch, LearningRows)
from agents.choosing import draw, masks_of
from agents.network import SOLDIER_KEYS, as_tensors
from simulation import action_spec_v1 as A


class MaskedChoiceSent(RuntimeError):
    """The environment counted a masked choice on a row the trainer drives
    (it must never: the draw is under the served masks)."""


def default_actions(masks: dict, n_heads: int) -> np.ndarray:
    """The legal default for every row of the served ``masks`` (``(N, U,
    size)`` each): stand, aim where I walk (or "keep" where that is the one
    legal aim), no trigger, hand nothing, argument 0."""
    aim = masks["mask_aim"]
    out = np.zeros(aim.shape[:-1] + (n_heads,), dtype=np.int64)
    out[..., 0] = A.FEET_STAND
    out[..., 1] = np.where(aim[..., A.AIM_WALK] != 0, A.AIM_WALK, A.AIM_KEEP)
    out[..., 2] = 0
    out[..., 3] = A.HAND_NOTHING
    out[..., 4] = 0
    return out


class Collector:
    """Plays the learning rows of ``env`` (a ``BatchedEnv`` or the
    stand-in) with its own copy of ``soldier``."""

    def __init__(self, env, soldier, rows: LearningRows, device="cpu",
                 seed: int = 0):
        self.env = env
        self.rows = rows
        self.device = torch.device(device)
        self.soldier = copy.deepcopy(soldier).to(self.device).eval()
        for p in self.soldier.parameters():
            p.requires_grad_(False)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.env_of_row = rows.env_of_row()
        self._obs = self._full = None
        R = rows.n_rows
        self._first = np.ones(R, dtype=bool)
        self._story = np.ones(R, dtype=bool)
        self.decisions = 0            # learning decisions stored, all rounds

    # -- the hand-over: weights in --------------------------------------
    def set_weights(self, weights: dict) -> None:
        """The soldier's weights the next round is played with."""
        self.soldier.load_state_dict(weights)

    # -- the matches ----------------------------------------------------
    def start(self) -> None:
        """Start every match (the run's first reset). Called once; between
        rounds nothing is reset."""
        obs, info = self.env.reset()
        self._obs, self._full = obs, info["full_view"]
        self._first[...] = True
        self._story[...] = True

    def _slot(self, batch: Batch, t: int):
        """The batch's arrays for step ``t`` by served key (the soldier's
        input, the masks, the two flags) and the full view's."""
        obs = {k: batch.obs[k][t] for k in OBS_STORED}
        obs["valid"] = batch.valid[t]
        obs["in_match"] = batch.in_match[t]
        full = {k: batch.full[k][t] for k in FULL_STORED}
        return obs, full

    def _gather_step(self, batch: Batch, t: int):
        """COPY the learning rows of the served arrays into step ``t``'s
        slot; return the slot (what the network acts on)."""
        obs, full = self._slot(batch, t)
        self.rows.gather(self._obs, obs, tuple(obs))
        self.rows.gather(self._full, full, tuple(full))
        return obs, full

    @torch.no_grad()
    def _act(self, obs: dict):
        x = as_tensors({k: obs[k] for k in SOLDIER_KEYS}, self.device)
        masks = {k: v.to(self.device)
                 for k, v in masks_of({k: obs[k] for k in A.MASK_KEYS}).items()}
        out, _ = self.soldier(x)
        return draw(out, masks, self.generator)

    def collect(self, batch: Batch) -> Batch:
        """Play ``batch.T`` decisions in every match; fill and return
        ``batch``."""
        if self._obs is None:
            raise RuntimeError("collect() before start()")
        env, rows = self.env, self.rows
        for t in range(batch.T):
            obs, _ = self._gather_step(batch, t)
            batch.first[t] = self._first
            batch.value_row[t] = (batch.in_match[t] != 0) & self._story
            d = self._act(obs)
            batch.actions[t] = d.actions.cpu().numpy()
            batch.used[t] = d.used.cpu().numpy()
            batch.logp_heads[t] = d.logp_heads.cpu().numpy()
            batch.logp[t] = d.logp.cpu().numpy()
            batch.arg_mask[t] = d.arg_mask.cpu().numpy()
            sent = default_actions({k: self._obs[k] for k in A.MASK_KEYS},
                                   len(A.HEADS))
            rows.scatter(batch.actions[t], sent)
            new_obs, reward, _, _, info = env.step(sent)
            self._check_masked(info, batch.valid[t])
            got = {"reward": np.zeros_like(batch.reward[t]),
                   "rows_ended": np.zeros(rows.n_rows, dtype=bool)}
            rows.gather({"reward": reward, "rows_ended": info["rows_ended"]},
                        got, ("reward", "rows_ended"))
            batch.reward[t] = got["reward"]
            ended = np.asarray(info["ended"], dtype=bool)[self.env_of_row]
            done = ended | got["rows_ended"]
            batch.done[t] = done
            # the bookkeeping: a story ends at its ending; a match that ended
            # was restarted inside the step (its next decision is a first)
            self._story &= ~done
            self._story |= ended
            self._first = ended.copy()
            self._obs, self._full = new_obs, info["full_view"]
            self.decisions += int(batch.valid[t].sum())
        self._gather_last(batch)
        return batch

    def _gather_last(self, batch: Batch) -> None:
        """COPY the view after the last step (the value network's keys)."""
        self.rows.gather(self._obs, batch.last_obs, LAST_OBS)
        self.rows.gather(self._full, batch.last_full, LAST_FULL)

    def _check_masked(self, info: dict, valid) -> None:
        if "masked_choices" not in info:
            raise RuntimeError("the environment served no 'masked_choices' "
                               "(an action spec other than 'v1'?)")
        got = {"m": np.zeros(self.rows.n_rows, dtype=np.int64)}
        self.rows.gather({"m": info["masked_choices"]}, got, ("m",))
        bad = np.flatnonzero((got["m"] != 0) & (np.asarray(valid) != 0))
        if len(bad):
            r = int(bad[0])
            raise MaskedChoiceSent(
                f"{int(got['m'][r])} masked choice(s) sent on a driven "
                f"living row ({self.rows.name(r)}); {len(bad)} row(s) in all")


__all__ = ["MaskedChoiceSent", "default_actions", "Collector"]
