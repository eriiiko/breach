"""agents/choosing.py -- THE one implementation of choosing (rung 2 design
2; design 8.2; plan section 4 item 6). Used when a soldier acts, when the
trainer learns from reward and when the network copies the bots; nothing
else turns the network's raw outputs and the masks into chances.

- :func:`chances`: the chances of every option of one head under its mask
  (a forbidden option has chance exactly 0), mixed with an optional small
  chance ``explore`` of a uniformly random LEGAL option (default 0);
- :func:`draw`: the five choices drawn in the action spec's order (feet,
  aim, trigger, hand, argument), each later head told the earlier choices
  through :class:`agents.network.HeadOutputs`, the argument's mask worked
  out from the drawn hand and aim (:func:`arg_mask`);
- :func:`evaluate`: the probability of five GIVEN choices, by the SAME walk
  (a given choice the masks forbid is refused);
- :func:`heads_used`: which heads counted -- the game's rule
  (``action_spec_v1.heads_used``: more than one legal option), batched;
- the variety of each counted head as a share of the most it could be: its
  entropy over the log of its number of legal options (design 4.4).

A head that did not count adds nothing to the decision's probability and
nothing to the variety. :func:`arg_mask` and :func:`heads_used` are batched
forms of the game's own functions (``action_spec_v1.arg_mask`` /
``heads_used``) for a whole batch on the card; ``agents/tests`` holds them
EQUAL to the game's on drawn cases, which is what keeps them from becoming a
second rule. Every number comes from the game's action spec.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from agents import _game  # noqa: F401
from simulation import action_spec_v1 as A

HEADS = A.HEADS
MASK_KEYS = A.MASK_KEYS


@dataclass
class Decision:
    """Five choices of ``N`` rows and what they cost.

    ``actions`` ``(N, 5)`` int64 in :data:`HEADS` order; ``logp`` ``(N,)``
    the log-probability of the whole decision (the counted heads' sum);
    ``logp_heads`` ``(N, 5)`` per head (0 where the head did not count);
    ``used`` ``(N, 5)`` bool, which heads counted; ``variety`` ``(N, 5)``
    each counted head's entropy as a share of its most (0 where not
    counted); ``arg_mask`` ``(N, 5)`` bool, the argument's mask the walk
    worked out."""
    actions: torch.Tensor
    logp: torch.Tensor
    logp_heads: torch.Tensor
    used: torch.Tensor
    variety: torch.Tensor
    arg_mask: torch.Tensor


def masks_of(obs: dict) -> dict:
    """The four served masks (``uint8``) of ``obs`` as bool tensors."""
    return {k: torch.as_tensor(obs[k]) != 0 for k in MASK_KEYS}


def arg_mask(hand, aim):
    """The argument's mask from the drawn ``hand`` and ``aim`` (int
    tensors of one shape), ``(..., N_ARG)`` bool: the game's rule
    (``action_spec_v1.arg_mask``) batched -- a throw: the four reaches, and
    "onto" iff the aim is a list entry; a plant: the four timers; any other
    hand: option 0 alone."""
    hand = hand.long()
    aim = aim.long()
    throw = hand >= A.HAND_THROW0
    four = throw | (hand == A.HAND_PLANT)
    out = torch.zeros(hand.shape + (A.N_ARG,), dtype=torch.bool,
                      device=hand.device)
    out[..., :A.N_FRACTIONS] = four.unsqueeze(-1)
    out[..., A.ARG_ONTO] = throw & (aim >= A.AIM_TOKENS)
    out[..., 0] |= ~four
    return out


def heads_used(masks: dict, argm):
    """Which heads counted, ``(N, 5)`` bool in :data:`HEADS` order: a head
    with more than one legal option (the game's ``heads_used``, batched)."""
    cols = [masks[k].sum(dim=-1) > 1 for k in MASK_KEYS]
    cols.append(argm.sum(dim=-1) > 1)
    return torch.stack(cols, dim=-1)


def chances(logits, mask, explore: float = 0.0):
    """One head's chances under its ``mask`` (``(N, n)`` bool), mixed with
    a chance ``explore`` of a uniformly random legal option. Returns ``(p,
    logp)``: ``p`` exactly 0 and ``logp`` ``-inf`` on a forbidden option.
    Refuses a row with no legal option."""
    if mask.shape != logits.shape:
        raise ValueError(f"mask {tuple(mask.shape)} against logits "
                         f"{tuple(logits.shape)}")
    n_legal = mask.sum(dim=-1, keepdim=True)
    if bool((n_legal == 0).any()):
        raise ValueError("a head with no legal option (the masks promise "
                         "at least one)")
    logits = logits.float()
    logp = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    if explore == 0.0:
        return logp.exp(), logp
    e = float(explore)
    if not 0.0 < e <= 1.0:
        raise ValueError(f"explore {explore} outside [0, 1]")
    p = (1.0 - e) * logp.exp() + e * mask.float() / n_legal.float()
    safe = torch.where(mask, p, torch.ones_like(p))
    logp = torch.where(mask, safe.log(), torch.full_like(p, float("-inf")))
    return p, logp


def variety_share(p, logp, mask):
    """Entropy of one head over the log of its legal options, ``(N,)``; 0
    where at most one option is legal. Never computes ``0 x -inf``."""
    safe = torch.where(mask, logp, torch.zeros_like(logp))
    ent = -(p * safe).sum(dim=-1)
    n = mask.sum(dim=-1)
    denom = torch.log(n.clamp(min=2).to(ent.dtype))
    return torch.where(n > 1, ent / denom, torch.zeros_like(ent))


def _walk(out, masks: dict, pick, explore: float) -> Decision:
    """THE walk over the five heads, shared by :func:`draw` and
    :func:`evaluate`: ``pick(h, p)`` returns head ``h``'s choice ``(N,)``."""
    n = out.n_rows
    for k in MASK_KEYS:
        if masks[k].shape[0] != n:
            raise ValueError(f"{k}: {masks[k].shape[0]} rows, the network "
                             f"gave {n}")
    if masks["mask_aim"].shape[-1] != out.aim_keys.shape[1]:
        raise ValueError(f"mask_aim has {masks['mask_aim'].shape[-1]} "
                         f"options, the list gives {out.aim_keys.shape[1]}")
    chosen, logps, varieties = [], [], []

    def head(h, logits, mask):
        p, logp = chances(logits, mask, explore)
        c = pick(h, p).long()
        lp = logp.gather(1, c.unsqueeze(1))[:, 0]
        if bool(torch.isinf(lp).any()):
            raise ValueError(f"head {HEADS[h]!r}: a choice the masks forbid")
        chosen.append(c)
        logps.append(lp)
        varieties.append(variety_share(p, logp, mask))
        return c

    head(0, out.feet(), masks["mask_feet"])
    aim = head(1, out.aim(), masks["mask_aim"])
    head(2, out.trigger(aim), masks["mask_trigger"])
    hand = head(3, out.hand(aim), masks["mask_hand"])
    argm = arg_mask(hand, aim)
    head(4, out.arg(aim, hand), argm)
    used = heads_used(masks, argm)
    lp = torch.stack(logps, dim=-1)
    lp = torch.where(used, lp, torch.zeros_like(lp))
    var = torch.stack(varieties, dim=-1)
    var = torch.where(used, var, torch.zeros_like(var))
    return Decision(actions=torch.stack(chosen, dim=-1), logp=lp.sum(dim=-1),
                    logp_heads=lp, used=used, variety=var, arg_mask=argm)


def draw(out, masks: dict, generator=None, explore: float = 0.0) -> Decision:
    """Draw the five choices of every row of ``out`` (a
    :class:`agents.network.HeadOutputs`) under ``masks`` (bool, see
    :func:`masks_of`)."""
    def pick(h, p):
        return torch.multinomial(p, 1, generator=generator)[:, 0]
    return _walk(out, masks, pick, explore)


def evaluate(out, masks: dict, actions, explore: float = 0.0) -> Decision:
    """The probability of the GIVEN choices ``actions`` ``(N, 5)`` under
    ``masks``, by the same walk as :func:`draw`."""
    actions = torch.as_tensor(actions).long()

    def pick(h, p):
        return actions[:, h].to(p.device)
    return _walk(out, masks, pick, explore)


__all__ = ["HEADS", "MASK_KEYS", "Decision", "masks_of", "arg_mask",
           "heads_used", "chances", "variety_share", "draw", "evaluate"]
