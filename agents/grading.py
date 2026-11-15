"""agents/grading.py -- grading, said exactly (rung 2 note 2b, 4; design
4.3). PURE functions of arrays: no network, no environment, so the case
worked by hand tests them as they are.

For every reward group g and every row, over the batch's T steps, with v
the value network's estimate in REAL units (computed once per round, with
the scale as it was while collecting):

    next[t]   = v[t + 1]            (for the last step: the estimate of the
                                     view after the batch's last step)
    delta[t]  = reward[t] + gamma * (1 - done[t]) * next[t] - v[t]
    adv[t]    = delta[t] + gamma * lambda * (1 - done[t]) * adv[t + 1]
                                    (adv[T] = 0)
    target[t] = adv[t] + v[t]

After an ending (``done``: the row's match ended, or its records ended)
nothing is still to come, whatever the view returned (it is the next
match's). The one cut that leans on the value network is the batch's end in
the middle of a match. Only VALUE ROWS count: a step that is not a value
row (an empty roster place, a row after its records ended) neither receives
nor passes anything on -- its advantage and target are 0.

The choice's grade is the weighted sum over the groups (:func:`choice_grade`)
and is brought to a common size AFTER the sum (:func:`common_size`), by the
learner, over the policy rows of the piece it learns from.
"""
from __future__ import annotations

import numpy as np


def grade(reward, value, last_value, done, value_row, gamma: float,
          lam: float):
    """The advantage and the value target of every step, per group.

    ``reward`` ``(T, R, G)``: the group numbers paid by the step that
    carried the decision out; ``value`` ``(T, R, G)``: the estimate at each
    decision; ``last_value`` ``(R, G)``: the estimate of the view after the
    last step; ``done`` ``(T, R)``: the row's story ended in that step;
    ``value_row`` ``(T, R)``: the step is a value row. Returns ``(adv,
    target)``, both ``(T, R, G)`` float64, 0 off the value rows."""
    reward = np.asarray(reward, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    last_value = np.asarray(last_value, dtype=np.float64)
    T, R, G = reward.shape
    if value.shape != (T, R, G) or last_value.shape != (R, G):
        raise ValueError(f"grade: reward {reward.shape}, value {value.shape}, "
                         f"last value {last_value.shape}")
    live = np.asarray(value_row, dtype=bool)                  # (T, R)
    goes_on = 1.0 - np.asarray(done, dtype=np.float64)        # (T, R)
    if live.shape != (T, R) or goes_on.shape != (T, R):
        raise ValueError(f"grade: done {np.shape(done)}, value rows "
                         f"{np.shape(value_row)}, want {(T, R)}")
    v = np.where(live[..., None], value, 0.0)
    adv = np.zeros((T, R, G))
    after = np.zeros((R, G))            # adv[t + 1]
    for t in reversed(range(T)):
        nxt = last_value if t == T - 1 else v[t + 1]
        keep = goes_on[t][:, None]
        delta = reward[t] + gamma * keep * nxt - v[t]
        a = delta + gamma * lam * keep * after
        adv[t] = np.where(live[t][:, None], a, 0.0)
        after = adv[t]
    target = np.where(live[..., None], adv + v, 0.0)
    return adv, target


def choice_grade(adv, weights):
    """The choice's grade: ``sum over g of weights[g] * adv[..., g]`` (the
    groups on the last axis, in ``REWARD_GROUPS`` order)."""
    adv = np.asarray(adv, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if adv.shape[-1] != w.shape[0]:
        raise ValueError(f"choice_grade: {adv.shape[-1]} groups, "
                         f"{w.shape[0]} weights")
    return (adv * w).sum(axis=-1)


def common_size(grades, floor: float):
    """``grades`` brought to a common size: minus their mean, over their
    spread plus ``floor`` (the piece's policy rows, taken by the caller)."""
    g = np.asarray(grades, dtype=np.float64)
    if g.size == 0:
        return g
    return (g - g.mean()) / (g.std() + float(floor))


__all__ = ["grade", "choice_grade", "common_size"]
