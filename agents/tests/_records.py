"""Small record files in the game's own format (``control_bots_door``'s
``.npz`` with its ``meta``), written for the loader's and the copy's tests.

Every array has today's trailing shape and number format
(``agents.records.expected``); every head of a living decision has its
legal options and the bot's choice among them; the argument's mask is the
game's rule for the drawn hand and aim. Vector column 0 carries a unique
MARK per (file, decision, slot) so a test can tell which decisions were
dealt: ``mark(file_no, t, slot)``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agents import records as RC
from simulation import action_spec_v1 as A


def mark(file_no: int, t: int, slot: int) -> float:
    return float(file_no * 100000 + t * 10 + slot)


def _meta(S, level, seed, team, n_kinds, names, over):
    import control_bots_door as D
    m = {"record_format": D.RECORD_FORMAT, "level_id": level,
         "seed": int(seed), "team": int(team), "S": S,
         "E": S + RC.O.ENTRIES_BEYOND_SLOTS, "K": n_kinds,
         "hand_names": list(names), "obs_spec": RC.O.SPEC_NAME,
         "action_spec": A.ACTION_SPEC_NAME,
         "controllers": ["bots_door", "bots"],
         "spec": {"zombies": 0}, "build": "test"}
    m.update(over)
    return m


def write(folder, level: str, seed: int, team: int, file_no: int, *,
          decisions: int = 20, living=(20, 12, 0), S: int = 3,
          rng=None, kind: str = "bots_door-bots__z0", over_meta=None,
          over_arrays=None) -> Path:
    """One record file at ``folder/level/<kind>__s<seed>__t<team>.npz``:
    ``decisions`` decisions; slot ``s`` alive for its first ``living[s]``."""
    from simulation.weapons import get_tables
    rng = np.random.default_rng(file_no) if rng is None else rng
    rows = get_tables().hand_grenade_rows
    K = len(rows)
    ex = RC.expected(S, K)
    n = int(decisions)
    a = {k: np.zeros((n,) + tail, dt) for k, (tail, dt) in ex.items()}
    a["list_uid"] = np.arange(n * S * (S + RC.O.ENTRIES_BEYOND_SLOTS),
                              dtype=np.int32).reshape(a["list_uid"].shape)
    live = np.arange(n)[:, None] < np.asarray(living)[None, :]
    a["valid"][...] = live
    a["in_match"][...] = 1
    E = S + RC.O.ENTRIES_BEYOND_SLOTS
    for t in range(n):
        for s in range(S):
            a["vector"][t, s, 0] = mark(file_no, t, s)
            a["vector"][t, s, 1:] = rng.normal(size=a["vector"].shape[-1] - 1) \
                .astype(np.float32) * 0.1
            a["near_f"][t, s] = rng.normal(size=a["near_f"].shape[2:])
            a["far_f"][t, s] = rng.normal(size=a["far_f"].shape[2:])
            # list: a few live entries of type "seen" (1), the rest padding
            k = int(rng.integers(1, 4))
            a["list_mask"][t, s, :k] = 1
            a["list"][t, s, :k, 0] = 1.0
            a["list"][t, s, :k, 1:3] = rng.normal(size=(k, 2)) * 0.5
            for key in A.MASK_KEYS:
                m = a[key][t, s]
                if not live[t, s]:
                    m[0 if key != "mask_aim" else A.AIM_WALK] = 1
                    continue
                m[:] = rng.random(m.shape) < 0.5
                if key == "mask_aim":
                    m[A.AIM_TOKENS:] = 0
                    m[A.AIM_TOKENS:A.AIM_TOKENS + k] = rng.random(k) < 0.7
                if key == "mask_hand":
                    m[A.HAND_FLASHLIGHT] = 0
                if not m.any():
                    m[0 if key != "mask_aim" else A.AIM_WALK] = 1
            choice = [int(rng.choice(np.flatnonzero(a[key][t, s])))
                      for key in A.MASK_KEYS]
            argm = A.arg_mask(choice[3], choice[1])
            choice.append(int(rng.choice(np.flatnonzero(argm))))
            a["actions"][t, s] = choice
            a["arg_mask"][t, s] = argm
            a["loss"][t, s] = int(rng.integers(0, 4))
    a["reward"] = np.zeros((n, S, 5), np.float32)
    a["rows_ended"] = np.zeros((n, S), bool)
    a["tick"] = np.arange(n, dtype=np.int64) * 4
    del E
    # the kind columns at a legal code (0), the numbers random
    from agents.sizes import sizes_for
    for c in sizes_for(S, n_kinds=K).vector_kinds:
        a["vector"][..., c.pos] = 0.0
    if over_arrays:
        a.update(over_arrays)
    meta = _meta(S, level, seed, team, K, A.hand_names(rows),
                 over_meta or {})
    path = Path(folder) / level / f"{kind}__s{seed}__t{team}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    a["meta"] = np.array(json.dumps(meta, sort_keys=True))
    with open(path, "wb") as f:
        np.savez_compressed(f, **a)
    return path


def folder(root, seeds=(1, 2, 5, 10, 11), levels=("ship_a", "ship_b"),
           **kw) -> Path:
    """A records folder: per level and seed two matches (one per side of
    the door team), each one file; file numbers in creation order."""
    root = Path(root)
    no = 0
    for lv in levels:
        for sd in seeds:
            for kind, team in (("bots_door-bots__z0", 0),
                               ("bots-bots_door__z4", 1)):
                no += 1
                write(root, lv, sd, team, no, kind=kind, **kw)
    return root
