"""agents/fake.py -- made-up inputs in the SERVED formats, for the tests
and the card-memory measurement of ``print_network``: every array has the
dtype and shape the game's own shape tables give (``obs_spec_v1.shapes``,
``full_shapes``, ``action_spec_v1.mask_shapes``), every kind column a
value its table allows, every mask the game's promises (no head without a
legal option; "nothing" and the compass tokens never masked; a row with no
unit carries ``action_spec_v1.default_masks``). The numbers are random, not
a game state.
"""
from __future__ import annotations

import numpy as np

from agents import _game  # noqa: F401
from agents.sizes import Sizes
from simulation import action_spec_v1 as A
from simulation import obs_spec_v1 as O
from simulation.factions import ZOMBIE_TEAM


def _kind_values(rng, k, shape, n_teams):
    if k.zombie_row >= 0:                       # side
        choices = np.array([-1, ZOMBIE_TEAM] + list(range(n_teams)))
        return rng.choice(choices, size=shape).astype(np.float32)
    return rng.integers(-1, k.rows - 1, size=shape).astype(np.float32)


def _table(rng, shape, kinds, n_teams, type_pos=None, n_types=0):
    x = rng.uniform(-1.5, 1.5, size=shape).astype(np.float32)
    for k in kinds:
        x[..., k.pos] = _kind_values(rng, k, shape[:-1], n_teams)
    if type_pos is not None:
        x[..., type_pos] = rng.integers(0, n_types, size=shape[:-1])
    return x


def _window(rng, window, n):
    w = window.side
    f = rng.uniform(-2, 2, size=(n, len(window.f_planes), w, w)).astype(
        np.float16)
    u = np.zeros((n, len(window.u_planes), w, w), dtype=np.uint8)
    for i, (name, codes) in enumerate(window.u_planes):
        top = max(codes) + 1 if codes else (
            4 if O.PLANE_COLUMNS[name][0] == O.COUNT else 2)
        u[:, i] = rng.integers(0, top, size=(n, w, w))
    return f, u


def _type_pos():
    return next(f.pos for f in O.ENTRY_COLUMNS if f.name == "type")


def make_obs(sizes: Sizes, n: int, rng, n_entries: int | None = None,
             p_invalid: float = 0.15, p_windup: float = 0.1) -> dict:
    """``n`` rows of a soldier's served observation and masks (numpy). A
    share ``p_invalid`` of rows carries an invalid row's masks (one legal
    option per head), ``p_windup`` a throw being wound up (aim: "keep")."""
    E = sizes.n_entries if n_entries is None else int(n_entries)
    tp = _type_pos()
    obs = {
        "vector": _table(rng, (n, sizes.vector), sizes.vector_kinds,
                         sizes.n_teams),
        "list": _table(rng, (n, E, sizes.entry), sizes.entry_kinds,
                       sizes.n_teams, tp, sizes.entry_types),
        "list_mask": (rng.uniform(size=(n, E)) < 0.5).astype(np.uint8),
        "list_uid": np.arange(n * E, dtype=np.int32).reshape(n, E),
    }
    obs["near_f"], obs["near_u"] = _window(rng, sizes.near, n)
    obs["far_f"], obs["far_u"] = _window(rng, sizes.far, n)
    heads = A.head_sizes(E, sizes.n_kinds)
    for h, k in enumerate(A.MASK_KEYS):
        obs[k] = (rng.uniform(size=(n, heads[h])) < 0.5).astype(np.uint8)
    obs["mask_feet"][:, A.FEET_STAND] = 1
    obs["mask_trigger"][np.arange(n), rng.integers(0, 2, size=n)] = 1
    aim = obs["mask_aim"]
    aim[:, :A.AIM_KEEP] = 1
    aim[:, A.AIM_KEEP] = 0
    aimable = np.isin(obs["list"][..., tp].astype(np.int64), A.AIMABLE_TYPES)
    aim[:, A.AIM_TOKENS:] &= (obs["list_mask"] & aimable).astype(np.uint8)
    hand = obs["mask_hand"]
    hand[:, A.HAND_NOTHING] = 1
    hand[:, A.HAND_FLASHLIGHT] = 0
    windup = rng.uniform(size=n) < p_windup
    aim[windup] = 0
    aim[windup, A.AIM_KEEP] = 1
    for i in np.flatnonzero(rng.uniform(size=n) < p_invalid):
        A.default_masks(obs, int(i))
    return obs


def make_full(sizes: Sizes, n: int, rng) -> dict:
    """``n`` rows of a served ``info["full_view"]`` (numpy)."""
    w = sizes.near.side
    full = {
        "full_list": _table(rng, (n, sizes.full_entries, sizes.full_entry),
                            sizes.full_kinds, sizes.n_teams, _type_pos(),
                            sizes.entry_types),
        "full_mask": (rng.uniform(size=(n, sizes.full_entries)) < 0.5).astype(
            np.uint8),
        "full_uid": np.arange(n * sizes.full_entries, dtype=np.int32).reshape(
            n, sizes.full_entries),
        "full_match": rng.uniform(-1, 1, size=(n, sizes.n_match)).astype(
            np.float32),
        "full_watched": rng.integers(0, 2, size=(n, w, w)).astype(np.uint8),
    }
    full["full_mask"][:, 0] = 1                 # the soldier itself
    return full


def pad_list(obs: dict, n_entries: int) -> dict:
    """``obs`` with its list padded to ``n_entries`` entries: the padding
    masked off (list, list mask, the aim's entry options)."""
    out = dict(obs)
    n, E = obs["list_mask"].shape
    extra = int(n_entries) - E
    if extra < 0:
        raise ValueError("pad_list only lengthens")
    out["list"] = np.concatenate(
        [obs["list"], np.zeros((n, extra, obs["list"].shape[-1]),
                               dtype=obs["list"].dtype)], axis=1)
    out["list_mask"] = np.concatenate(
        [obs["list_mask"], np.zeros((n, extra), dtype=np.uint8)], axis=1)
    out["list_uid"] = np.concatenate(
        [obs["list_uid"], np.full((n, extra), -1, dtype=np.int32)], axis=1)
    out["mask_aim"] = np.concatenate(
        [obs["mask_aim"], np.zeros((n, extra), dtype=np.uint8)], axis=1)
    return out


__all__ = ["make_obs", "make_full", "pad_list"]
