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
   rows into the fewest near-equal PIECES of at most ``piece`` decisions
   (:func:`cut_pieces`: never a short last one); a piece is worked through in
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

WITH THE SOLDIER'S MEMORY ON (patch 2d; design 4.5), the unit a pass
shuffles and cuts is the STRETCH -- a row's ``stretch`` decisions ``k *
stretch .. k * stretch + stretch - 1`` on the batch's own time -- never a
single decision: a piece of ``piece`` decisions is ``piece / stretch``
stretches, a lump ``lump / stretch``. Only stretches holding a value row are
taken. Each stretch is replayed in order by ``forward_sequence`` from the
stored ``memory[t0]`` (a constant: the gradient stops at the stretch's
start) with its stored ``first`` flags, and the policy rows among its
decisions are evaluated; the value network learns from the value rows of
the same stretches. Grades, the clip, the early stop, the common size per
piece over its policy rows and the two optimisers are as without memory.
With the memory off nothing here changes: the same weights, bit for bit.

THE VALUE-FIRST ROUNDS (``value_first_rounds``; design 6 arm B): in a run's
first such rounds the live check runs, the value network and its scale
learn, and the soldier's network takes no step (its optimiser is not
touched, so its weights stay the same to the bit). The step-size plan is
not shifted.

THE LIVE CHECK (:func:`live_check`), every round, on the first piece of
the first pass, before any step: on the policy rows the recomputed
probability of every stored choice equals the stored one to
``live_tolerance``, which heads counted and the argument's mask recomputed
equal the stored ones exactly, and the distance moved is 0 to the
tolerance; with the memory on (its second clause, design 4.8) the memory
``forward_sequence`` returns after every stretch of the piece equals the
stored ``memory[t0 + stretch]`` to ``live_tolerance`` -- and the first
clause then runs through the sequence path, so it also catches a stretch
replayed from the wrong start. A failure stops the run
(:class:`LiveCheckFailed`) saying which clause (for the memory, the row and
the stretch). It catches a kept reference to an environment array, masks
that differ between acting and learning, a mismatch of number formats, any
difference between acting and learning in the choosing module, and a
memory stored or replayed apart from the one the soldier had.

THE BATCH'S HEADER (patch 2d.2; design 4.6): :meth:`Learner.learn` first
reads the batch's header (``Batch.check_header``) against its own
(:meth:`Learner.header`: the version of its current weights, the run file's
and the training file's sha256, the build id) and refuses a batch that
differs (``batch.ForeignBatch``, naming the field). ``learn`` moves the
weights, so it raises the version by one; the trainer hands the new weights
over with it.

The choosing module is the ONE place a probability is computed: this
module calls ``evaluate`` and writes no softmax, sampler or log-probability
of its own.

WHAT A ROUND OF LEARNING MEASURED (patch 2c's per-round line), besides the
losses: the seconds of grading (the estimates, the grades, the scale's
move) and of the passes, the card waited for at each end; per reward group,
the share of the grading targets' spread the value network's estimates
explained BEFORE this round's passes (``1 - var(target - v) / var(target)``
over the value rows; None for a group whose PAY does not vary in the round:
its targets would vary only by the network's own guesses); the
variety of each head over the decisions where it counted; the passes each
measured distance belongs to.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from agents import _game  # noqa: F401
from agents.batch import Batch, run_ids
from agents.choosing import evaluate
from agents.grading import choice_grade, common_size, grade
from agents.network import SOLDIER_KEYS, VALUE_FULL_KEYS, VALUE_OBS_KEYS
from agents.settings import Settings
from agents.value_scale import ValueScale
from simulation import action_spec_v1 as A


#: Below this spread of a group's pay over a round the group "does not
#: vary" and its explained share is None (a share of nothing is noise).
VARIES = 1e-12


def explained(v, target, reward, rows) -> list:
    """Per group: ``1 - var(target - v) / var(target)`` over ``rows`` (the
    value rows), or None where the group's pay (``reward``) does not vary
    over them, or its targets do not."""
    out = []
    for g in range(target.shape[-1]):
        y = target[..., g][rows]
        if (y.size == 0 or float(reward[..., g][rows].var()) <= VARIES
                or float(y.var()) <= VARIES):
            out.append(None)
            continue
        out.append(float(1.0 - (y - v[..., g][rows]).var() / y.var()))
    return out


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def cut_pieces(order, piece: int) -> list:
    """A pass's shuffled rows cut into the FEWEST pieces of at most
    ``piece`` rows, of near-equal size (they differ by one row at most).
    Never a short last piece: under Adam a step taken from a handful of
    decisions is as long as one taken from two thousand, and its grades,
    brought to a common size over three rows, are noise."""
    n = len(order)
    if n == 0:
        return []
    return np.array_split(order, -(-n // int(piece)))


def live_check(d_logp_heads, s_logp_heads, d_used, s_used, d_arg, s_arg,
               distance: float, tolerance: float, memory=None) -> dict:
    """The live check's clauses over the first piece's policy rows
    (recomputed ``d_*`` against stored ``s_*``); with the memory on,
    ``memory`` is ``(gap, where)``: the largest distance between a
    stretch's recomputed and stored end memory and the stretch it was found
    at. Raises :class:`LiveCheckFailed` naming every clause that fails, else
    returns what it measured."""
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
    out = {"max_logp_gap": gap, "distance": float(distance)}
    if memory is not None:
        m_gap, where = memory
        out["max_memory_gap"] = float(m_gap)
        if not m_gap <= tolerance:
            bad.append(f"the memory at a stretch's end (recomputed apart by "
                       f"{m_gap:.3g}, tolerance {tolerance}, at {where})")
    if bad:
        raise LiveCheckFailed("live check failed: " + "; ".join(bad))
    return out


class Learner:
    """Learns the soldier's and the value network from batches (see the
    module text)."""

    def __init__(self, soldier, value, settings: Settings, device="cpu",
                 seed: int = 0, ids: dict | None = None):
        self.settings = settings
        #: the run's names a batch's header must carry (run file and
        #: training file sha256, build id; None where the caller has none)
        self.ids = run_ids(ids)
        #: the version of the soldier's current weights: raised by one by
        #: every :meth:`learn` (it moves them), saved with the network
        self.version = 0
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
    def header(self) -> dict:
        """What a batch played with the current weights carries
        (``batch.HEADER_FIELDS``): the version and the run's names."""
        return {"version": self.version, **self.ids}

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
        # the header: a batch played with other weights, or for another run
        # or build, is refused by name (design 4.6)
        batch.check_header(self.header())
        _sync(self.device)
        t0 = time.perf_counter()
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
        explained_share = explained(v, target, batch.reward,
                                    batch.value_row != 0)

        # 2. move the scale toward this batch's targets (the last layer is
        #    corrected inside, so the real-unit outputs do not move)
        self.scale.update(targets[value_rows])
        _sync(self.device)
        t1 = time.perf_counter()

        # 3. learn, on the targets normalised with the new scale
        target_n = self.scale.normalise(
            torch.as_tensor(targets, device=self.device)).float()
        stored_logp = batch.flat("logp")
        live = None
        soldier_learns = True
        stopped = None
        distances, pol_losses, val_losses, varieties, clipped = \
            [], [], [], [], []
        distance_pass = []
        n_heads = len(A.HEADS)
        variety_heads = np.zeros(n_heads)
        used_heads = np.zeros(n_heads)
        soldier_steps = value_steps = 0
        value_first = self.round < s.value_first_rounds
        memory = batch.memory is not None
        if memory:
            # the units are the stretches holding a value row
            units = self._stretch_units(batch, value_rows)
            unit_piece = s.piece // s.stretch
            unit_lump = s.lump // s.stretch
        else:
            units, unit_piece, unit_lump = (np.flatnonzero(value_rows),
                                            s.piece, s.lump)
        for p in range(s.passes):
            order = self.rng.permutation(units)
            for k, piece_u in enumerate(cut_pieces(order, unit_piece)):
                checking = (p == 0 and k == 0)
                piece = (self._stretch_decisions(batch, piece_u).reshape(-1)
                         if memory else piece_u)
                pol = piece[policy_rows[piece]]
                n_pol = len(pol)
                n_val = (int(value_rows[piece].sum()) if memory
                         else len(piece))
                # the piece's grades brought to a common size, before its
                # first lump
                grades = np.zeros(T * R)
                grades[pol] = common_size(choice[pol], s.grade_floor)
                self.opt_soldier.zero_grad(set_to_none=True)
                self.opt_value.zero_grad(set_to_none=True)
                moved = 0.0
                seen = {"d_lp": [], "s_lp": [], "d_u": [], "s_u": [],
                        "d_a": [], "s_a": [], "mem": []}
                for i in range(0, len(piece_u), unit_lump):
                    lump_u = piece_u[i:i + unit_lump]
                    out = None
                    if memory:
                        idx = self._stretch_decisions(batch, lump_u)
                        lump = idx.reshape(-1)
                        lump = lump[value_rows[lump]]
                        if soldier_learns:
                            out = self._soldier_stretches(
                                batch, lump_u, idx, policy_rows, n_pol,
                                grades, stored_logp, checking, seen)
                    else:
                        lump = lump_u
                        lump_pol = lump[policy_rows[lump]]
                        if soldier_learns and len(lump_pol):
                            out = self._soldier_lump(batch, lump_pol, n_pol,
                                                     grades, stored_logp,
                                                     checking, seen)
                    if out is not None:
                        moved += out["moved"]
                        variety_heads += out["variety_heads"]
                        used_heads += out["used_heads"]
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
                        distance, s.live_tolerance,
                        self._memory_gap(batch, seen["mem"]) if memory
                        else None)
                if soldier_learns and n_pol:
                    distances.append(distance)
                    distance_pass.append(p)
                    if value_first:
                        # the value-first rounds: the live check ran; the
                        # soldier's network takes no step this round
                        soldier_learns = False
                    elif distance > s.max_distance:
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
        self.version += 1
        _sync(self.device)
        t2 = time.perf_counter()
        return {
            "seconds": {"grading": t1 - t0, "passes": t2 - t1},
            "explained": explained_share,
            "variety_heads": [float(a / b) if b else None
                              for a, b in zip(variety_heads, used_heads)],
            "distance_pass": distance_pass,
            "step_size": step_size,
            "live": live,
            "distances": distances,
            "stopped": stopped,
            "value_first": value_first,
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

    # -- the memory's stretches ------------------------------------------
    def _stretch_units(self, batch: Batch, value_rows) -> np.ndarray:
        """The stretches holding a value row, as units ``k * R + r``
        (stretch ``k`` of row ``r``)."""
        S, T, R = self.settings.stretch, batch.T, batch.R
        vr = value_rows.reshape(T // S, S, R).any(axis=1)
        return np.flatnonzero(vr.reshape(-1))

    def _stretch_decisions(self, batch: Batch, units) -> np.ndarray:
        """``(stretch, n)``: the flat decision index of every step of the
        stretches ``units``, in the order ``forward_sequence`` returns
        them (time first)."""
        S, R = self.settings.stretch, batch.R
        units = np.asarray(units)
        t0 = (units // R) * S
        r = units % R
        return (t0[None, :] + np.arange(S)[:, None]) * R + r[None, :]

    def _soldier_stretches(self, batch, units, idx, policy_rows, n_pol,
                           grades, stored_logp, checking, seen):
        """One lump of stretches: replayed in order by ``forward_sequence``
        from the stored memory at each start with the stored ``first``
        flags; the soldier's loss on their policy rows (its share of the
        piece's mean), backward. On the checking piece the memory after
        every stretch is kept for the live check's second clause. Returns
        the measurements, or None when the lump holds no policy row."""
        S, R = self.settings.stretch, batch.R
        units = np.asarray(units)
        t0 = (units // R) * S
        r = units % R
        flat = idx.reshape(-1)
        obs = {k: torch.as_tensor(batch.flat(k)[idx], device=self.device)
               for k in SOLDIER_KEYS}
        first = torch.as_tensor(batch.flat("first")[idx] != 0,
                                device=self.device)
        state = torch.as_tensor(batch.memory[t0, r], device=self.device)
        out, end = self.soldier.forward_sequence(obs, state, first)
        if checking:
            stored = torch.as_tensor(batch.memory[t0 + S, r],
                                     device=self.device)
            seen["mem"].append(((end.detach() - stored).abs().amax(dim=-1),
                                units))
        sel = np.flatnonzero(policy_rows[flat])
        if not len(sel):
            return None
        return self._soldier_loss(
            out.take(torch.as_tensor(sel, device=self.device)), batch,
            flat[sel], n_pol, grades, stored_logp, checking, seen)

    def _memory_gap(self, batch: Batch, kept) -> tuple:
        """The live check's second clause, measured: the largest distance
        between a stretch's recomputed and stored end memory over the
        checking piece, and where (the row's place, the stretch)."""
        if not kept:
            return 0.0, "no stretch"
        gaps = torch.cat([g for g, _ in kept]).cpu().numpy()
        units = np.concatenate([u for _, u in kept])
        w = int(np.argmax(gaps))
        k, r = divmod(int(units[w]), batch.R)
        S = self.settings.stretch
        return (float(gaps[w]),
                f"{batch.rows.name(r)}, stretch {k} (decisions {k * S} to "
                f"{k * S + S - 1})")

    def _soldier_lump(self, batch, idx, n_pol, grades, stored_logp,
                      checking, seen) -> dict:
        """One lump of the soldier's loss (its share of the piece's mean),
        backward; returns the summed distance and the measurements."""
        out, _ = self.soldier(self._soldier_inputs(batch, idx))
        return self._soldier_loss(out, batch, idx, n_pol, grades,
                                  stored_logp, checking, seen)

    def _soldier_loss(self, out, batch, idx, n_pol, grades, stored_logp,
                      checking, seen) -> dict:
        """The soldier's loss over the decisions ``idx`` given the
        network's outputs for them, backward (see :meth:`_soldier_lump`)."""
        s = self.settings
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
        used = d.used.detach()
        return {"moved": float((old - d.logp.detach()).sum()),
                "loss": float(loss.detach()) * n_pol / len(idx),
                "variety": float(variety.detach().mean()),
                "variety_heads": (d.variety.detach() * used).sum(dim=0)
                .cpu().numpy().astype(np.float64),
                "used_heads": used.sum(dim=0).cpu().numpy()
                .astype(np.float64),
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


__all__ = ["VARIES", "explained", "LiveCheckFailed", "one_arithmetic",
           "cut_pieces", "live_check", "Learner"]
