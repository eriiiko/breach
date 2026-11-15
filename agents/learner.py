"""agents/learner.py -- the learner (rung 2 note 2b, 5 to 7; design 4.3,
4.4, 4.8): takes a batch, returns the soldier's weights.

ONE ROUND OF LEARNING, top to bottom (:meth:`Learner.learn`):

1. **Grade** with the value targets' scale as it was while collecting: the
   value network's estimate of every stored decision and of the view after
   the last one, in real units, without gradients; then
   :func:`agents.grading.grade` per reward group and the weighted sum
   (:func:`agents.grading.choice_grade`).
2. **Move the scale** toward this batch's targets and correct the value
   network's last layer so its real-unit outputs do not move
   (:class:`agents.value_scale.ValueScale`).
3. **Learn**, on the targets normalised with the new scale:
   ``passes`` passes; each pass cuts a fresh shuffle of the round's value
   rows into PIECES of ``piece`` decisions; a piece is worked through in
   LUMPS of ``lump`` decisions whose gradients are added up, so that the
   step is the step of the piece's MEAN loss. Per piece:

   - its grades are brought to a common size over its policy rows (minus
     the mean, over the spread plus a small floor), before its first lump;
   - the soldier's loss on the policy rows: the clipped objective on the
     JOINT ratio (the stored and the recomputed probability of the whole
     decision, ``choosing.evaluate`` under the stored masks), minus
     ``variety`` times the sum of the counted heads' variety;
   - the value loss on the value rows: the squared error of the normalised
     estimates, summed over the five outputs (all learn, the weight-0 ones
     too);
   - the distance moved: the mean of stored minus recomputed
     log-probability over the piece's policy rows, from the same forward
     pass the step uses. The first piece over ``max_distance`` is not
     stepped and the soldier's network takes no further step this round;
     the value network finishes its passes;
   - each network's gradient is cut to ``max_grad_norm`` by itself, and
     each has its own Adam (step size falling in a straight line to zero
     over the planned rounds).

THE LIVE CHECK (:func:`live_check`), every round, on the first piece of
the first pass, before any step: on the policy rows the recomputed
probability of every stored choice equals the stored one to
``live_tolerance``, which heads counted and the argument's mask recomputed
equal the stored ones exactly, and the distance moved is 0 to the
tolerance. A failure stops the run (:class:`LiveCheckFailed`) saying which
of the three. It catches a kept reference to an environment array, masks
that differ between acting and learning, a mismatch of number formats, and
any difference between acting and learning in the choosing module.

The choosing module is the ONE place a probability is computed: this
module calls ``evaluate`` and writes no softmax, sampler or log-probability
of its own.
"""
from __future__ import annotations

import numpy as np
import torch

from agents import _game  # noqa: F401
from agents.batch import Batch
from agents.choosing import evaluate
from agents.grading import choice_grade, common_size, grade
from agents.network import SOLDIER_KEYS, VALUE_FULL_KEYS, VALUE_OBS_KEYS
from agents.settings import Settings
from agents.value_scale import ValueScale
from simulation import action_spec_v1 as A


class LiveCheckFailed(RuntimeError):
    """The live check found acting and learning apart (the message says
    which of its three clauses)."""


def one_arithmetic() -> None:
    """One arithmetic for acting and learning (note 2b, 6): 32-bit, the
    card's faster-but-different options off (TF32, the convolution
    algorithm search)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def live_check(d_logp_heads, s_logp_heads, d_used, s_used, d_arg, s_arg,
               distance: float, tolerance: float) -> dict:
    """The live check's three clauses over the first piece's policy rows
    (recomputed ``d_*`` against stored ``s_*``); raises
    :class:`LiveCheckFailed` naming every clause that fails, else returns
    what it measured."""
    gap = float((d_logp_heads - s_logp_heads).abs().max()) \
        if d_logp_heads.numel() else 0.0
    heads_ok = bool(torch.equal(d_used, s_used)) and bool(
        torch.equal(d_arg, s_arg))
    bad = []
    if not gap <= tolerance:
        bad.append(f"the probability of a stored choice (recomputed apart "
                   f"by {gap:.3g} in log-probability, tolerance "
                   f"{tolerance})")
    if not heads_ok:
        n = int((d_used != s_used).any(dim=-1).sum()
                + (d_arg != s_arg).any(dim=-1).sum())
        bad.append(f"which heads counted or the argument's mask "
                   f"(recomputed apart on {n} row(s))")
    if not abs(distance) <= tolerance:
        bad.append(f"the distance moved before any step ({distance:.3g}, "
                   f"tolerance {tolerance})")
    if bad:
        raise LiveCheckFailed("live check failed: " + "; ".join(bad))
    return {"max_logp_gap": gap, "distance": float(distance)}


class Learner:
    """Learns the soldier's and the value network from batches (see the
    module text)."""

    def __init__(self, soldier, value, settings: Settings, device="cpu",
                 seed: int = 0):
        self.settings = settings
        self.device = torch.device(device)
        self.soldier = soldier.to(self.device).eval()
        self.value = value.to(self.device).eval()
        self.scale = ValueScale(self.value.out[-1], settings.scale_share,
                                settings.scale_floor)
        self.opt_soldier = torch.optim.Adam(self.soldier.parameters(),
                                            lr=settings.step_size,
                                            eps=settings.adam_eps)
        self.opt_value = torch.optim.Adam(self.value.parameters(),
                                          lr=settings.step_size,
                                          eps=settings.adam_eps)
        self.rng = np.random.default_rng(int(seed))
        self.round = 0

    # -- the hand-over: weights out ---------------------------------------
    def weights(self) -> dict:
        """The soldier's weights, for the collector's next round."""
        return {k: v.detach().clone()
                for k, v in self.soldier.state_dict().items()}

    # -- inputs of a set of stored decisions --------------------------------
    def _tensor(self, batch: Batch, name: str, idx):
        return torch.as_tensor(batch.flat(name)[idx], device=self.device)

    def _soldier_inputs(self, batch: Batch, idx) -> dict:
        return {k: self._tensor(batch, k, idx) for k in SOLDIER_KEYS}

    def _masks_for(self, batch: Batch, idx) -> dict:
        """The masks stored with the decisions ``idx``, as bool tensors."""
        return {k: self._tensor(batch, k, idx) != 0 for k in A.MASK_KEYS}

    def _value_inputs(self, batch: Batch, idx):
        obs = {k: self._tensor(batch, k, idx) for k in VALUE_OBS_KEYS}
        full = {k: self._tensor(batch, k, idx) for k in VALUE_FULL_KEYS}
        return obs, full

    @torch.no_grad()
    def _estimates(self, batch: Batch):
        """The value network's estimates in REAL units with the scale as it
        is now: every value row's ``(T, R, G)`` (0 elsewhere) and the view
        after the last step's ``(R, G)``."""
        T, R, G = batch.reward.shape
        v = np.zeros((T * R, G))
        rows = np.flatnonzero(batch.flat("value_row") != 0)
        for i in range(0, len(rows), self.settings.lump):
            idx = rows[i:i + self.settings.lump]
            obs, full = self._value_inputs(batch, idx)
            v[idx] = self.scale.real(self.value(obs, full)).cpu().numpy()
        last = np.zeros((R, G))
        for i in range(0, R, self.settings.lump):
            sl = slice(i, i + self.settings.lump)
            obs = {k: torch.as_tensor(batch.last_obs[k][sl],
                                      device=self.device)
                   for k in VALUE_OBS_KEYS}
            full = {k: torch.as_tensor(batch.last_full[k][sl],
                                       device=self.device)
                    for k in VALUE_FULL_KEYS}
            last[sl] = self.scale.real(self.value(obs, full)).cpu().numpy()
        return v.reshape(T, R, G), last

    # -- one round ------------------------------------------------------------
    def learn(self, batch: Batch) -> dict:
        """Learn from ``batch`` (played with the current weights); returns
        the round's measurements. The weights are :meth:`weights`."""
        s = self.settings
        step_size = s.step_size_at(self.round)
        for opt in (self.opt_soldier, self.opt_value):
            for g in opt.param_groups:
                g["lr"] = step_size
        T, R, G = batch.reward.shape

        # 1. grade, with the scale as it was while collecting
        v, v_last = self._estimates(batch)
        adv, target = grade(batch.reward, v, v_last, batch.done,
                            batch.value_row, s.gamma, s.lam)
        choice = choice_grade(adv, s.weights).reshape(T * R)
        value_rows = batch.flat("value_row") != 0
        policy_rows = (batch.flat("valid") != 0) & value_rows
        targets = target.reshape(T * R, G)

        # 2. move the scale toward this batch's targets (the last layer is
        #    corrected inside, so the real-unit outputs do not move)
        self.scale.update(targets[value_rows])

        # 3. learn, on the targets normalised with the new scale
        target_n = self.scale.normalise(
            torch.as_tensor(targets, device=self.device)).float()
        stored_logp = batch.flat("logp")
        live = None
        soldier_learns = True
        stopped = None
        distances, pol_losses, val_losses, varieties, clipped = \
            [], [], [], [], []
        soldier_steps = value_steps = 0
        order_rows = np.flatnonzero(value_rows)
        for p in range(s.passes):
            order = self.rng.permutation(order_rows)
            for k, start in enumerate(range(0, len(order), s.piece)):
                piece = order[start:start + s.piece]
                checking = (p == 0 and k == 0)
                pol = piece[policy_rows[piece]]
                n_pol, n_val = len(pol), len(piece)
                # the piece's grades brought to a common size, before its
                # first lump
                grades = np.zeros(T * R)
                grades[pol] = common_size(choice[pol], s.grade_floor)
                self.opt_soldier.zero_grad(set_to_none=True)
                self.opt_value.zero_grad(set_to_none=True)
                moved = 0.0
                seen = {"d_lp": [], "s_lp": [], "d_u": [], "s_u": [],
                        "d_a": [], "s_a": []}
                for i in range(0, n_val, s.lump):
                    lump = piece[i:i + s.lump]
                    lump_pol = lump[policy_rows[lump]]
                    if soldier_learns and len(lump_pol):
                        out = self._soldier_lump(batch, lump_pol, n_pol,
                                                 grades, stored_logp,
                                                 checking, seen)
                        moved += out["moved"]
                        pol_losses.append(out["loss"])
                        varieties.append(out["variety"])
                        clipped.append(out["clipped"])
                    val_losses.append(self._value_lump(batch, lump, n_val,
                                                       target_n))
                distance = moved / n_pol if n_pol else 0.0
                if checking and soldier_learns and n_pol:
                    live = live_check(
                        torch.cat(seen["d_lp"]), torch.cat(seen["s_lp"]),
                        torch.cat(seen["d_u"]), torch.cat(seen["s_u"]),
                        torch.cat(seen["d_a"]), torch.cat(seen["s_a"]),
                        distance, s.live_tolerance)
                if soldier_learns and n_pol:
                    distances.append(distance)
                    if distance > s.max_distance:
                        # the early stop: this piece is not stepped
                        soldier_learns = False
                        stopped = (p, k)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.soldier.parameters(), s.max_grad_norm)
                        self.opt_soldier.step()
                        soldier_steps += 1
                torch.nn.utils.clip_grad_norm_(self.value.parameters(),
                                               s.max_grad_norm)
                self.opt_value.step()
                value_steps += 1
        self.round += 1
        return {
            "step_size": step_size,
            "live": live,
            "distances": distances,
            "stopped": stopped,
            "soldier_steps": soldier_steps,
            "value_steps": value_steps,
            "policy_loss": float(np.mean(pol_losses)) if pol_losses else 0.0,
            "value_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            "variety": float(np.mean(varieties)) if varieties else 0.0,
            "clipped": float(np.mean(clipped)) if clipped else 0.0,
            "policy_rows": int(policy_rows.sum()),
            "value_rows": int(value_rows.sum()),
            "scale_mean": self.scale.mean.cpu().numpy().copy(),
            "scale_spread": self.scale.spread.cpu().numpy().copy(),
        }

    def _soldier_lump(self, batch, idx, n_pol, grades, stored_logp,
                      checking, seen) -> dict:
        """One lump of the soldier's loss (its share of the piece's mean),
        backward; returns the summed distance and the measurements."""
        s = self.settings
        out, _ = self.soldier(self._soldier_inputs(batch, idx))
        masks = self._masks_for(batch, idx)
        actions = self._tensor(batch, "actions", idx).long()
        try:
            d = evaluate(out, masks, actions)
        except ValueError as e:
            if checking:
                raise LiveCheckFailed(
                    f"live check failed: the probability of a stored choice "
                    f"(the masks at learning forbid it: {e})") from None
            raise
        old = torch.as_tensor(stored_logp[idx], device=self.device)
        adv = torch.as_tensor(grades[idx], device=self.device,
                              dtype=torch.float32)
        log_ratio = d.logp - old
        ratio = log_ratio.exp()
        clipped = ratio.clamp(1.0 - s.clip, 1.0 + s.clip)
        objective = torch.minimum(ratio * adv, clipped * adv)
        variety = d.variety.sum(dim=-1)
        loss = (-objective.sum() - s.variety * variety.sum()) / n_pol
        loss.backward()
        if checking:
            seen["d_lp"].append(d.logp_heads.detach())
            seen["s_lp"].append(self._tensor(batch, "logp_heads", idx))
            seen["d_u"].append(d.used.detach())
            seen["s_u"].append(self._tensor(batch, "used", idx) != 0)
            seen["d_a"].append(d.arg_mask.detach())
            seen["s_a"].append(self._tensor(batch, "arg_mask", idx) != 0)
        return {"moved": float((old - d.logp.detach()).sum()),
                "loss": float(loss.detach()) * n_pol / len(idx),
                "variety": float(variety.detach().mean()),
                "clipped": float(((ratio - 1.0).abs() > s.clip)
                                 .float().mean())}

    def _value_lump(self, batch, idx, n_val, target_n) -> float:
        """One lump of the value loss (its share of the piece's mean),
        backward; returns the lump's mean loss."""
        obs, full = self._value_inputs(batch, idx)
        pred = self.value(obs, full)
        err = ((pred - target_n[torch.as_tensor(idx, device=self.device)])
               ** 2).sum(dim=-1)
        (err.sum() / n_val).backward()
        return float(err.detach().mean())


__all__ = ["LiveCheckFailed", "one_arithmetic", "live_check", "Learner"]
