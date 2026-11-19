"""agents/copying.py -- copying the bots (behaviour cloning; rung 2 note 2e,
sections 5 and 6; design 3.2 and 3.3).

THE LOSS (:func:`copy_loss`), per living decision of a door bot: minus the
log-probability of the bot's recorded choice, summed over the heads that
counted -- computed by the choosing module (``choosing.evaluate``) under
the records' own masks, the argument's mask worked out by the game's rule
from the bot's hand and aim and held EQUAL to the records' stored one
(refused otherwise). The feet alone are trained toward "the bot's choice
with ``1 - feet_floor``, the OTHER LEGAL options sharing ``feet_floor``"
(design 3.2); the hand, the argument, the aim and the trigger as they are.
A head that did not count adds nothing. The loss of a step is the mean over
its decisions.

THE TRAINING (:class:`Copier`): Adam, its step size falling in a straight
line to zero over the planned passes (by decisions dealt), the gradient cut
to ``max_grad_norm``, a step's decisions worked through in lumps whose
gradients add up to the step's mean. Without memory the steps are batches
of single decisions (``records.DecisionDealer``); with memory, stretches of
``lanes`` soldiers (``records.StretchDealer``), each lane's memory carried
from one stretch to the next (detached: the gradient stops at a stretch's
start, as in the trainer) and zeroed only at a soldier's first decision.
The value network is not trained here.

THE MEASURES (:func:`measure`, design 3.3), each with the number of files
(one per match and door team) and decisions it was read on: per head,
teacher-forced agreement (each head's most likely option, given the bot's
earlier choices) against "the most common legal option" (counted on the
training records) and "repeat the previous choice" (the soldier's own
previous decision; the aim by the entry's uid), on all decisions and on
the changes only; per option; the mean angle error of feet and aim; the
pointer against "aim at the nearest enemy"; every counted head right at
once with the network's own aim and hand chosen first; agreement split by
the records' loss bits. A memory network is measured on every soldier's
match from its start.

THE RESULT is written through ``agents.saved.export_soldier`` only, naming
what it was started from (the records' folder, their census hash, the split
rule, the pass).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from agents import _game  # noqa: F401
from agents import records as RC
from agents.choosing import MASK_KEYS, chances, evaluate, mode
from agents.network import SOLDIER_KEYS
from agents.settings import CopySettings
from simulation import action_spec_v1 as A
from simulation import obs_spec_v1 as O

HEADS = A.HEADS
#: The previous-choice inputs (design 3.3, "does it lean on its own
#: past?"), by name from the spec's tables: the vector's six decoded
#: previous choices (kinds: blanked to -1, "none") and the list's
#: previous-aim flag (blanked to 0).
PREV_VECTOR = tuple(f.pos for f in O.VECTOR_COLUMNS
                    if f.name.startswith("prev["))
PREV_ENTRY = tuple(f.pos for f in O.ENTRY_COLUMNS if f.name == "prev_aim")
_ENTRY_DX = next(f.pos for f in O.ENTRY_COLUMNS if f.name == "dx")
_ENTRY_DY = next(f.pos for f in O.ENTRY_COLUMNS if f.name == "dy")
_ENTRY_DIST = next(f.pos for f in O.ENTRY_COLUMNS if f.name == "dist_c")
#: The compass step of feet and aim, in degrees.
_STEP_DEG = 360.0 / A.N_DIRS


class CopyRefused(ValueError):
    """A record's stored argument mask is not the game's rule for its hand
    and aim (the records are inconsistent: refused, never trained on)."""


def blank_previous(d: dict) -> dict:
    """``d`` with the previous-choice inputs blanked (copies of the two
    arrays; the rest shared)."""
    out = dict(d)
    v = d["vector"].copy()
    v[..., list(PREV_VECTOR)] = -1.0
    lst = d["list"].copy()
    lst[..., list(PREV_ENTRY)] = 0.0
    out["vector"], out["list"] = v, lst
    return out


def _tensors(d: dict, idx, device) -> tuple:
    """The soldier's inputs, the masks, the bot's actions and the stored
    argument mask of the decisions ``idx`` of ``d`` (numpy, first axis)."""
    obs = {k: torch.as_tensor(np.ascontiguousarray(d[k][idx]), device=device)
           for k in SOLDIER_KEYS}
    return (obs,) + _masks_actions(d, idx, device)


def _masks_actions(d: dict, idx, device) -> tuple:
    """The masks, the bot's actions and the stored argument mask of the
    decisions ``idx`` of ``d``."""
    masks = {k: torch.as_tensor(np.ascontiguousarray(d[k][idx]),
                                device=device) != 0 for k in MASK_KEYS}
    actions = torch.as_tensor(d["actions"][idx], device=device).long()
    argm = torch.as_tensor(d["arg_mask"][idx], device=device) != 0
    return masks, actions, argm


def copy_loss(out, masks: dict, actions, stored_arg, feet_floor: float):
    """Per decision, minus the log-probability of the bot's choice over the
    counted heads, the feet's term spread (module text). Returns ``(loss
    (N,), Decision)``."""
    d = evaluate(out, masks, actions)
    if not torch.equal(d.arg_mask, stored_arg):
        n = int((d.arg_mask != stored_arg).any(dim=-1).sum())
        raise CopyRefused(f"the records' argument mask differs from the "
                          f"game's rule for the bot's hand and aim on {n} "
                          f"decision(s)")
    loss = -d.logp_heads.sum(dim=-1)
    f = float(feet_floor)
    if f > 0.0:
        legal = masks["mask_feet"]
        _, lp = chances(out.feet(), legal)
        mine = torch.zeros_like(legal)
        mine.scatter_(1, actions[:, :1], True)
        others = legal & ~mine
        n_other = others.sum(dim=-1).clamp(min=1).to(lp.dtype)
        other = torch.where(others, lp, torch.zeros_like(lp)).sum(dim=-1) \
            / n_other
        lp_bot = d.logp_heads[:, 0]
        feet = (1.0 - f) * lp_bot + f * other
        loss = loss + torch.where(d.used[:, 0], lp_bot - feet,
                                  torch.zeros_like(feet))
    return loss, d


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
class Copier:
    """Trains ``soldier`` on the records (module text)."""

    def __init__(self, soldier, settings: CopySettings, device="cpu",
                 planned: int = 1):
        self.s = settings
        self.device = torch.device(device)
        self.soldier = soldier.to(self.device).train(False)
        self.opt = torch.optim.Adam(self.soldier.parameters(),
                                    lr=settings.step_size,
                                    eps=settings.adam_eps)
        #: decisions planned over the run (passes x a pass's decisions):
        #: the step size falls in a straight line to zero over them
        self.planned = max(1, int(planned))
        self.dealt = 0
        self.steps = 0

    def _prep(self, d):
        return blank_previous(d) if self.s.blank_prev else d

    def _lr(self):
        lr = self.s.step_size * max(0.0, 1.0 - self.dealt / self.planned)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def step(self, b: dict) -> dict:
        """One step on a batch of single decisions (``DecisionDealer``)."""
        b = self._prep(b)
        n = len(b["actions"])
        lr = self._lr()
        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        for i in range(0, n, self.s.lump):
            idx = slice(i, min(n, i + self.s.lump))
            obs, masks, actions, argm = _tensors(b, idx, self.device)
            out, _ = self.soldier(obs)
            loss, _ = copy_loss(out, masks, actions, argm, self.s.feet_floor)
            (loss.sum() / n).backward()
            total += float(loss.detach().sum())
        norm = float(torch.nn.utils.clip_grad_norm_(
            self.soldier.parameters(), self.s.max_grad_norm))
        self.opt.step()
        self.dealt += n
        self.steps += 1
        return {"loss": total / n, "decisions": n, "grad_norm": norm,
                "step_size": lr}

    def step_stretches(self, b: dict, state):
        """One step on a batch of stretches (``StretchDealer``) from the
        lanes' carried memory ``state`` ``(lanes, core)``; returns the
        measurements and the lanes' memory after the step's stretches
        (detached)."""
        b = self._prep(b)
        S, L = b["kept"].shape
        n = int(b["kept"].sum())
        lr = self._lr()
        self.opt.zero_grad(set_to_none=True)
        per = max(1, self.s.lump // S)
        new = state.clone()
        total = 0.0
        for j0 in range(0, L, per):
            j = slice(j0, min(L, j0 + per))
            obs = {k: torch.as_tensor(np.ascontiguousarray(b[k][:, j]),
                                      device=self.device)
                   for k in SOLDIER_KEYS}
            first = torch.as_tensor(b["first"][:, j], device=self.device)
            upto = torch.as_tensor(np.maximum(b["length"][j], 1),
                                   device=self.device)
            out, end = self.soldier.forward_sequence(obs, state[j], first,
                                                     upto=upto)
            new[j] = end.detach()
            kept = np.flatnonzero(b["kept"][:, j].reshape(-1))
            if not len(kept):
                continue
            flat = {k: b[k][:, j].reshape((-1,) + b[k].shape[2:])
                    for k in MASK_KEYS + ("actions", "arg_mask")}
            masks, actions, argm = _masks_actions(flat, kept, self.device)
            sel = torch.as_tensor(kept, device=self.device)
            loss, _ = copy_loss(out.take(sel), masks, actions, argm,
                                self.s.feet_floor)
            (loss.sum() / max(1, n)).backward()
            total += float(loss.detach().sum())
        norm = float(torch.nn.utils.clip_grad_norm_(
            self.soldier.parameters(), self.s.max_grad_norm))
        self.opt.step()
        self.dealt += n
        self.steps += 1
        return ({"loss": total / max(1, n), "decisions": n,
                 "grad_norm": norm, "step_size": lr}, new)



# ---------------------------------------------------------------------------
# the measures (design 3.3)
# ---------------------------------------------------------------------------
def _unsquash(s):
    """The inverse of the observation's one curve (``obs_spec_v1.squash``)
    below its cap: the identity on [-1, 1], ``sign(s) exp(|s| - 1)``
    beyond."""
    s = np.asarray(s, dtype=np.float64)
    big = np.abs(s) > 1.0
    out = s.copy()
    out[big] = np.sign(s[big]) * np.exp(np.abs(s[big]) - 1.0)
    return out


def _feet_deg(f):
    """A feet option's direction in degrees (NaN for stand)."""
    f = np.asarray(f, dtype=np.int64)
    d = ((f - 1) // A.N_SPEEDS).astype(np.float64) * _STEP_DEG
    return np.where(f == A.FEET_STAND, np.nan, d)


def _aim_deg(aim, lst):
    """An aim's direction in degrees: a compass token's, or the bearing of
    the list entry it points at (``atan2(-dy, dx)``, the compass's
    convention, from the entry's unsquashed offset); NaN for "where I
    walk" and "keep"."""
    aim = np.asarray(aim, dtype=np.int64)
    out = np.full(aim.shape, np.nan)
    tok = aim < A.N_DIRS
    out[tok] = aim[tok] * _STEP_DEG
    ent = np.flatnonzero(aim >= A.AIM_TOKENS)
    if len(ent):
        j = aim[ent] - A.AIM_TOKENS
        dx = _unsquash(lst[ent, j, _ENTRY_DX])
        dy = _unsquash(lst[ent, j, _ENTRY_DY])
        out[ent] = np.degrees(np.arctan2(-dy, dx)) % 360.0
    return out


def _angle_gap(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _used(d: dict) -> np.ndarray:
    """Which of the bot's heads counted ``(n, 5)``: the GAME's rule
    (``action_spec_v1.heads_used``) on the records' masks."""
    u = A.heads_used({k: d[k] for k in MASK_KEYS}, d["arg_mask"])
    return np.stack([u[h] for h in HEADS], axis=-1)


class OptionCounts:
    """How often the bots chose each option of each head where it counted,
    over the training records: THE counts of the baseline "always the most
    common legal option"."""

    def __init__(self, head_sizes):
        self.counts = [np.zeros(int(n), np.int64) for n in head_sizes]

    def add(self, d: dict) -> None:
        used = _used(d)
        for h in range(len(HEADS)):
            a = d["actions"][used[:, h], h].astype(np.int64)
            np.add.at(self.counts[h], a, 1)

    def choice(self, h: int, legal) -> np.ndarray:
        """The most common option among the legal ones (``legal`` ``(n,
        options)``; ties to the lower index)."""
        c = np.where(legal != 0, self.counts[h][None, :legal.shape[1]], -1)
        return c.argmax(axis=-1)

    def as_json(self) -> list:
        return [c.tolist() for c in self.counts]


def _legal(d: dict, h: int):
    return d["arg_mask"] if h == 4 else d[MASK_KEYS[h]]


def _previous(d: dict) -> tuple:
    """Per decision, the soldier's previous choice of each head ``(n, 5)``
    (-1 at a soldier's first decision; the aim mapped to the entry of the
    SAME uid in this decision's list, -1 when it is gone) and the first
    flags."""
    act = d["actions"].astype(np.int64)
    prev = np.full_like(act, -1)
    first = d["first"].astype(bool)
    later = np.flatnonzero(~first)
    prev[later] = act[later - 1]
    ent = later[act[later - 1, 1] >= A.AIM_TOKENS]
    for i in ent:
        uid = d["list_uid"][i - 1, act[i - 1, 1] - A.AIM_TOKENS]
        hit = np.flatnonzero((d["list_uid"][i] == uid)
                             & (d["list_mask"][i] != 0))
        prev[i, 1] = A.AIM_TOKENS + int(hit[0]) if len(hit) else -1
    return prev, first


def _option_names(h: int, n: int) -> list:
    if h == 0:
        return ["stand"] + [f"dir{k // A.N_SPEEDS}_speed{k % A.N_SPEEDS}"
                            for k in range(n - 1)]
    if h == 1:
        return ([f"compass{k}" for k in range(A.N_DIRS)]
                + ["walk", "keep", "entry"])
    if h == 3:
        return list(_hand_names())
    return [str(k) for k in range(n)]


class Tally:
    """The measures' accumulators over files (module text)."""

    def __init__(self, head_sizes, counts: OptionCounts | None):
        self.sizes = tuple(int(n) for n in head_sizes)
        self.counts = counts
        H = len(HEADS)
        self.files = 0
        self.decisions = 0

        def z():
            return np.zeros(H, np.int64)
        self.n, self.agree, self.common, self.repeat = z(), z(), z(), z()
        self.ch_n, self.ch_agree, self.ch_common, self.ch_repeat = \
            z(), z(), z(), z()
        self.opt_n = [np.zeros(n, np.int64) for n in self.option_sizes()]
        self.opt_agree = [np.zeros(n, np.int64) for n in self.option_sizes()]
        self.joint_tf = self.joint_free = 0
        self.feet_gap, self.feet_n, self.feet_stand_miss = 0.0, 0, 0
        self.aim_gap, self.aim_n = 0.0, 0
        self.ptr_n = self.ptr_agree = self.ptr_nearest = 0
        nb = len(_loss_bits())
        self.bit_n = np.zeros(nb + 1, np.int64)        # last: no loss bit
        self.bit_joint = np.zeros(nb + 1, np.int64)
        self.bit_head_n = np.zeros((nb + 1, H), np.int64)
        self.bit_head_agree = np.zeros((nb + 1, H), np.int64)

    def option_sizes(self):
        """Per head the options reported apart (the aim's entries as one)."""
        out = list(self.sizes)
        out[1] = A.AIM_TOKENS + 1
        return out

    def add(self, d: dict, tm, fm) -> None:
        """One file's living decisions ``d`` (slot-major, time order) with
        the network's teacher-forced (``tm``) and own (``fm``) most likely
        options."""
        bot = d["actions"].astype(np.int64)
        tm, fm = np.asarray(tm), np.asarray(fm)
        used = _used(d)
        prev, first = _previous(d)
        self.files += 1
        self.decisions += len(bot)
        right_tf = (tm == bot) | ~used
        for h in range(len(HEADS)):
            u = used[:, h]
            legal = _legal(d, h)
            ok = tm[:, h] == bot[:, h]
            com = (self.counts.choice(h, legal) if self.counts is not None
                   else np.full(len(bot), -1))
            p = prev[:, h]
            p_legal = np.zeros(len(bot), bool)
            has = np.flatnonzero(p >= 0)
            p_legal[has] = legal[has, p[has]] != 0
            rep = np.where(first, com, np.where(p_legal, p, -1))
            self.n[h] += u.sum()
            self.agree[h] += (ok & u).sum()
            self.common[h] += ((com == bot[:, h]) & u).sum()
            self.repeat[h] += ((rep == bot[:, h]) & u).sum()
            ch = u & ~first & (p != bot[:, h])
            self.ch_n[h] += ch.sum()
            self.ch_agree[h] += (ok & ch).sum()
            self.ch_common[h] += ((com == bot[:, h]) & ch).sum()
            self.ch_repeat[h] += ((rep == bot[:, h]) & ch).sum()
            o = bot[:, h] if h != 1 else np.minimum(bot[:, h], A.AIM_TOKENS)
            np.add.at(self.opt_n[h], o[u], 1)
            np.add.at(self.opt_agree[h], o[u & ok], 1)
        self.joint_tf += int(right_tf.all(axis=-1).sum())
        self.joint_free += int(((fm == bot) | ~used).all(axis=-1).sum())
        # feet: the angle where both move; a stand against a move apart
        u = used[:, 0]
        fb, fn = _feet_deg(bot[:, 0]), _feet_deg(tm[:, 0])
        both = u & ~np.isnan(fb) & ~np.isnan(fn)
        self.feet_gap += float(_angle_gap(fb[both], fn[both]).sum())
        self.feet_n += int(both.sum())
        self.feet_stand_miss += int((u & (np.isnan(fb) != np.isnan(fn)))
                                    .sum())
        # aim: the angle where both aims have a direction
        u = used[:, 1]
        ab = _aim_deg(bot[:, 1], d["list"])
        an = _aim_deg(tm[:, 1], d["list"])
        both = u & ~np.isnan(ab) & ~np.isnan(an)
        self.aim_gap += float(_angle_gap(ab[both], an[both]).sum())
        self.aim_n += int(both.sum())
        # the pointer, against "aim at the nearest enemy"
        ptr = u & (bot[:, 1] >= A.AIM_TOKENS)
        legal_e = d["mask_aim"][:, A.AIM_TOKENS:] != 0
        dist = np.where(legal_e, d["list"][..., _ENTRY_DIST], np.inf)
        nearest = A.AIM_TOKENS + dist.argmin(axis=-1)
        self.ptr_n += int(ptr.sum())
        self.ptr_agree += int((ptr & (tm[:, 1] == bot[:, 1])).sum())
        self.ptr_nearest += int((ptr & (nearest == bot[:, 1])).sum())
        # the loss bits
        bits = d["loss"].astype(np.int64)
        nb = len(_loss_bits())
        for b in range(nb + 1):
            sel = (bits & (1 << b)) != 0 if b < nb else bits == 0
            self.bit_n[b] += sel.sum()
            self.bit_joint[b] += (sel & right_tf.all(axis=-1)).sum()
            self.bit_head_n[b] += (used & sel[:, None]).sum(axis=0)
            self.bit_head_agree[b] += (used & sel[:, None]
                                       & (tm == bot)).sum(axis=0)

    def result(self) -> dict:
        def share(a, b):
            return None if not b else float(a) / float(b)
        H = len(HEADS)
        heads = {}
        for h in range(H):
            names = _option_names(h, self.option_sizes()[h])
            heads[HEADS[h]] = {
                "decisions": int(self.n[h]),
                "agreement": share(self.agree[h], self.n[h]),
                "most_common": share(self.common[h], self.n[h]),
                "repeat_previous": share(self.repeat[h], self.n[h]),
                "changes": {"decisions": int(self.ch_n[h]),
                            "agreement": share(self.ch_agree[h],
                                               self.ch_n[h]),
                            "most_common": share(self.ch_common[h],
                                                 self.ch_n[h]),
                            "repeat_previous": share(self.ch_repeat[h],
                                                     self.ch_n[h])},
                "per_option": {names[k]: {"decisions": int(c),
                                          "agreement": share(a, c)}
                               for k, (c, a) in enumerate(
                                   zip(self.opt_n[h], self.opt_agree[h]))
                               if c},
            }
        bits = list(_loss_bits()) + ["none"]
        return {
            "files": self.files, "decisions": self.decisions,
            "heads": heads,
            "all_heads_teacher_forced": share(self.joint_tf, self.decisions),
            "all_heads_own_choices": share(self.joint_free, self.decisions),
            "feet_angle_deg": {"mean": share(self.feet_gap, self.feet_n),
                               "decisions": self.feet_n,
                               "stand_against_move": self.feet_stand_miss},
            "aim_angle_deg": {"mean": share(self.aim_gap, self.aim_n),
                              "decisions": self.aim_n},
            "pointer": {"decisions": self.ptr_n,
                        "agreement": share(self.ptr_agree, self.ptr_n),
                        "nearest_enemy": share(self.ptr_nearest,
                                               self.ptr_n)},
            "loss_bits": {
                bits[b]: {"decisions": int(self.bit_n[b]),
                          "all_heads": share(self.bit_joint[b],
                                             self.bit_n[b]),
                          "heads": {HEADS[h]: share(
                              self.bit_head_agree[b, h],
                              self.bit_head_n[b, h]) for h in range(H)}}
                for b in range(len(bits))},
        }


def _loss_bits() -> tuple:
    import control_bots_door as D
    return D.LOSS_BITS


def _hand_names() -> tuple:
    from simulation.weapons import get_tables
    return A.hand_names(get_tables().hand_grenade_rows)


@torch.no_grad()
def most_likely(soldier, d: dict, device, lump: int = 512,
                chunk: int = 64) -> tuple:
    """The network's teacher-forced and own most likely options ``(n, 5)``
    on one file's living decisions ``d`` (slot-major, time order). With the
    memory on every soldier is run from its first decision, the memory
    carried through its whole match (``chunk`` decisions at a time)."""
    n = len(d["actions"])
    tm = np.zeros((n, len(HEADS)), np.int64)
    fm = np.zeros_like(tm)
    if soldier.memory is None:
        for i in range(0, n, lump):
            idx = slice(i, min(n, i + lump))
            obs, masks, actions, _ = _tensors(d, idx, device)
            out, _ = soldier(obs)
            tm[idx] = mode(out, masks, given=actions).cpu().numpy()
            fm[idx] = mode(out, masks).cpu().numpy()
        return tm, fm
    rows = [np.flatnonzero(d["slot"] == s) for s in np.unique(d["slot"])]
    T = max(len(r) for r in rows)
    # (T, soldiers) decision indices, padded with each soldier's last
    grid = np.stack([r[np.minimum(np.arange(T), len(r) - 1)] for r in rows],
                    axis=1)
    real = np.arange(T)[:, None] < np.array([len(r) for r in rows])[None]
    state = None
    for t0 in range(0, T, chunk):
        g = grid[t0:t0 + chunk]
        obs = {k: torch.as_tensor(d[k][g], device=device)
               for k in SOLDIER_KEYS}
        first = torch.as_tensor(d["first"][g], device=device)
        out, state = soldier.forward_sequence(obs, state, first)
        keep = np.flatnonzero(real[t0:t0 + chunk].reshape(-1))
        idx = g.reshape(-1)[keep]
        sub = out.take(torch.as_tensor(keep, device=device))
        masks, actions, _ = _masks_actions(d, idx, device)
        tm[idx] = mode(sub, masks, given=actions).cpu().numpy()
        fm[idx] = mode(sub, masks).cpu().numpy()
    return tm, fm


def measure(soldier, records, *, device="cpu", counts=None,
            blank_prev: bool = False, workers: int = 4,
            limit_files: int | None = None) -> dict:
    """The measures (module text) of ``soldier`` on ``records`` (a list of
    ``records.RecordFile``), with the baselines' ``counts`` (an
    :class:`OptionCounts` of the training records, or None)."""
    tally = Tally(soldier.sizes.heads, counts)
    for _, d in RC.decisions_of(records, workers, limit_files):
        if not len(d["actions"]):
            continue
        x = blank_previous(d) if blank_prev else d
        tm, fm = most_likely(soldier, x, device)
        tally.add(d, tm, fm)
    return tally.result()


# ---------------------------------------------------------------------------
# a run
# ---------------------------------------------------------------------------
def check_split(records) -> dict:
    """The split of the census's records, refusing a (level, seed) on both
    sides (the split is by match, never by file)."""
    parts = RC.split(records)
    keys = {s: {(f.level, f.seed) for f in parts[s]} for s in parts}
    both = keys["train"] & keys["held"]
    if both:
        raise RuntimeError(f"(level, seed) on both sides of the split: "
                           f"{sorted(both)[:5]}")
    return parts


def _state_copy(soldier) -> dict:
    return {k: v.detach().cpu().clone() for k, v in
            soldier.state_dict().items()}


def _count_batch(counts: OptionCounts, b: dict, memory: bool) -> None:
    if not memory:
        counts.add(b)
        return
    k = b["kept"].reshape(-1)
    counts.add({key: b[key].reshape((-1,) + b[key].shape[2:])[k]
                for key in MASK_KEYS + ("arg_mask", "actions")})


def run(records_dir, net, settings: CopySettings, out_dir, *,
        device="cpu", census=None, training_file_text: str | None = None,
        limit_train: int | None = None, measure_training: bool = True,
        log=print) -> dict:
    """A copying run (module text): the census, the split, ``settings.
    passes`` passes with the held-back agreement after each (every counted
    head right, teacher-forced), the best pass's weights exported through
    ``agents.saved``, then the measures on the held-back and the training
    records. Writes into ``out_dir``: ``census.json``, ``curve.jsonl`` (a
    line a pass), ``copy.pt`` (the best pass's soldier), ``measures.json``.
    ``limit_train`` keeps the first files of the training side (sorted by
    name): a short run, never a long one."""
    from agents import saved as SV
    from agents.network import SoldierNet
    from agents.sizes import sizes_for
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    c = census if census is not None else RC.census(
        records_dir, settings.loader_workers)
    (out / "census.json").write_text(json.dumps(RC.census_json(c), indent=1,
                                                default=list))
    if c["refused"]:
        raise RC.RecordRefused("records that do not fit: "
                               + " | ".join(c["refused"][:5]))
    parts = check_split(c["records"])
    train, held = parts["train"], parts["held"]
    if limit_train is not None:
        train = sorted(train, key=lambda f: f.path)[:int(limit_train)]
    S = max(int(RC.read_meta(f.path)["S"]) for f in (train[0], held[0]))
    torch.manual_seed(int(settings.seed))
    soldier = SoldierNet(net, sizes_for(S))
    total = sum(f.living for f in train)
    copier = Copier(soldier, settings, device,
                    planned=settings.passes * total)
    counts = OptionCounts(soldier.sizes.heads)
    memory = soldier.memory is not None
    lanes = max(1, settings.batch // settings.stretch)
    if memory:
        dealer = RC.StretchDealer(train, seed=settings.seed, lanes=lanes,
                                  stretch=settings.stretch,
                                  buffer_matches=settings.buffer_matches,
                                  workers=settings.loader_workers)
    else:
        dealer = RC.DecisionDealer(train, seed=settings.seed,
                                   batch=settings.batch,
                                   buffer_matches=settings.buffer_matches,
                                   workers=settings.loader_workers)
    best = (-1.0, None, None)
    started = {"copy": {"records": str(Path(records_dir)),
                        "census_sha256": c["hash"],
                        "split": f"held iff seed % {RC.HELD_EVERY} == 0",
                        "pass": None, "settings": asdict(settings),
                        "training_file": training_file_text}}
    specs = {"observation": O.SPEC_NAME, "action": A.ACTION_SPEC_NAME,
             "reward_profile": None}
    with open(out / "curve.jsonl", "a", encoding="utf-8") as curve:
        for p in range(settings.passes):
            t0 = time.perf_counter()
            busy = 0.0
            losses = []
            state = soldier.initial_state(lanes, copier.device)
            for b in dealer.batches(p):
                if p == 0:
                    _count_batch(counts, b, memory)
                tb = time.perf_counter()
                if memory:
                    stats, state = copier.step_stretches(b, state)
                else:
                    stats = copier.step(b)
                if copier.device.type == "cuda":
                    torch.cuda.synchronize(copier.device)
                busy += time.perf_counter() - tb
                losses.append(stats["loss"])
                if copier.steps % 50 == 0:
                    log(f"pass {p} step {copier.steps} loss "
                        f"{stats['loss']:.4f} dealt {copier.dealt}")
            t1 = time.perf_counter()
            agree = measure(soldier, held, device=copier.device,
                            counts=counts, blank_prev=settings.blank_prev,
                            workers=settings.loader_workers)
            t2 = time.perf_counter()
            line = {"pass": p, "steps": copier.steps,
                    "decisions_dealt": copier.dealt,
                    "loss_mean": float(np.mean(losses)) if losses else None,
                    "seconds_pass": t1 - t0, "seconds_learning": busy,
                    "seconds_waiting_for_data": (t1 - t0) - busy,
                    "seconds_measure": t2 - t1,
                    "held_all_heads": agree["all_heads_teacher_forced"],
                    "held_heads": {h: v["agreement"] for h, v in
                                   agree["heads"].items()},
                    "held_files": agree["files"],
                    "held_decisions": agree["decisions"]}
            curve.write(json.dumps(line) + "\n")
            curve.flush()
            log(json.dumps(line))
            if agree["all_heads_teacher_forced"] > best[0]:
                best = (agree["all_heads_teacher_forced"], p,
                        _state_copy(soldier))
                started["copy"]["pass"] = p
                SV.export_soldier(out / "copy.pt", soldier, specs=specs,
                                  started_from=started)
    soldier.load_state_dict(best[2])
    result = {"best_pass": best[1], "census_sha256": c["hash"],
              "settings": asdict(settings), "net": asdict(net),
              "train": {"files": len(train), "decisions": total},
              "held": measure(soldier, held, device=copier.device,
                              counts=counts, blank_prev=settings.blank_prev,
                              workers=settings.loader_workers),
              "option_counts": counts.as_json()}
    if measure_training:
        result["training"] = measure(soldier, train, device=copier.device,
                                     counts=counts,
                                     blank_prev=settings.blank_prev,
                                     workers=settings.loader_workers)
    result["seconds"] = time.perf_counter() - t_start
    result["loader_wait_seconds"] = dealer.wait
    (out / "measures.json").write_text(json.dumps(result, indent=1))
    return result


def loader_speed(records, *, memory: bool, seconds: float, workers: int,
                 batch: int, stretch: int, buffer_matches: int,
                 seed: int = 0) -> dict:
    """Decisions per second the loader deals with no network attached."""
    if memory:
        dealer = RC.StretchDealer(records, seed=seed,
                                  lanes=max(1, batch // stretch),
                                  stretch=stretch,
                                  buffer_matches=buffer_matches,
                                  workers=workers)
    else:
        dealer = RC.DecisionDealer(records, seed=seed, batch=batch,
                                   buffer_matches=buffer_matches,
                                   workers=workers)
    t0 = time.perf_counter()
    n = 0
    first = None
    gen = dealer.batches(0)
    try:
        for b in gen:
            n += int(b["kept"].sum()) if memory else len(b["actions"])
            if first is None:
                first = time.perf_counter() - t0
            if time.perf_counter() - t0 > seconds:
                break
    finally:
        gen.close()
    dt = time.perf_counter() - t0
    return {"memory": memory, "decisions": n, "seconds": dt,
            "per_second": n / dt, "first_batch_seconds": first,
            "workers": workers, "batch": batch}


def main(argv=None) -> int:
    import argparse
    from agents import training_file as TF
    ap = argparse.ArgumentParser(prog="python -m agents.copying",
                                 description="copying the bots (note 2e)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("census", help="read every record once")
    a.add_argument("--records", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--workers", type=int, default=4)
    a = sub.add_parser("speed", help="the loader's speed, no network")
    a.add_argument("--records", required=True)
    a.add_argument("--memory", action="store_true")
    a.add_argument("--seconds", type=float, default=60.0)
    a.add_argument("--workers", type=int, default=4)
    a = sub.add_parser("train", help="a copying run")
    a.add_argument("--records", required=True)
    a.add_argument("--training-file", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--device", default="cpu")
    a.add_argument("--limit-train", type=int, default=None)
    a.add_argument("--no-training-measure", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "census":
        c = RC.census(args.records, args.workers)
        j = RC.census_json(c)
        Path(args.out).write_text(json.dumps(j, indent=1, default=list))
        print(json.dumps({k: v for k, v in j.items() if k != "records"},
                         indent=1, default=list))
        return 0 if not c["refused"] else 1
    if args.cmd == "speed":
        c = RC.census(args.records, args.workers)
        train = check_split(c["records"])["train"]
        cs = CopySettings()
        print(json.dumps(loader_speed(
            train, memory=args.memory, seconds=args.seconds,
            workers=args.workers, batch=cs.batch, stretch=cs.stretch,
            buffer_matches=cs.buffer_matches)))
        return 0
    settings, net, text, _ = TF.load_copy(args.training_file)
    from agents.learner import one_arithmetic
    one_arithmetic()
    run(args.records, net, settings, args.out, device=args.device,
        training_file_text=text, limit_train=args.limit_train,
        measure_training=not args.no_training_measure)
    return 0


__all__ = ["PREV_VECTOR", "PREV_ENTRY", "CopyRefused", "blank_previous",
           "copy_loss", "Copier", "OptionCounts", "Tally", "most_likely",
           "measure", "check_split", "run", "loader_speed", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
