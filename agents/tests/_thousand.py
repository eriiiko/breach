"""The 1 000-decision test of design 3.3 ("is the data path sound?"),
shared by its pytest wrapper and the patch's short card run: 1 000 living
decisions drawn across at least 50 training matches of real records,
learnt by the copy until every counted head agrees on at least
``LINE`` of them, or the step budget runs out."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from agents import copying as C
from agents import records as RC
from agents.network import SoldierNet
from agents.settings import CopySettings
from agents.sizes import sizes_for

#: Where the real records are: ``BREACH_RECORDS`` or 1j's folder.
RECORDS = Path(os.environ.get("BREACH_RECORDS", "C:/tmp/breach_1j/records"))
DECISIONS = 1000
MATCHES = 50
LINE = 0.99
#: The test's step budget (reduced width passes in about 1 750).
STEPS = 4000


def draw(records_dir=RECORDS, decisions=DECISIONS, matches=MATCHES,
         seed=0) -> dict:
    """``decisions`` living decisions, the same number from each of
    ``matches`` training files spread over every ship (by name order), as
    one dict of arrays."""
    root = Path(records_dir)
    by_ship = {}
    for p in sorted(root.rglob("*.npz")):
        m = RC.read_meta(p)
        if RC.side(m["level_id"], m["seed"]) == "train":
            by_ship.setdefault(m["level_id"], []).append(str(p))
    ships = sorted(by_ship)
    picked = []
    for i in range(matches):
        lst = by_ship[ships[i % len(ships)]]
        picked.append(lst[(i // len(ships)) * (len(lst) * len(ships)
                                               // matches) % len(lst)])
    rng = np.random.default_rng(seed)
    per = decisions // matches
    parts = []
    for p in picked:
        d = RC.unpack(p)
        idx = np.sort(rng.choice(len(d["t"]), per, replace=False))
        parts.append({k: v[idx] for k, v in d.items()})
    out = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    out["files"] = np.array(sorted(set(picked)))
    return out


def agreement(soldier, d, device) -> list:
    """Per head, teacher-forced agreement over the decisions where the
    bot's head counted (None where it never counted)."""
    tm, _ = C.most_likely(soldier, d, device)
    used = C._used(d)
    bot = d["actions"].astype(np.int64)
    return [None if not used[:, h].any()
            else float((tm[used[:, h], h] == bot[used[:, h], h]).mean())
            for h in range(len(C.HEADS))]


def learn(d, net, *, device="cpu", steps=1500, check_every=25,
          settings=None, log=None) -> dict:
    """Train a fresh soldier of spec ``net`` on ``d`` (all of it each
    step) until every counted head agrees on ``LINE`` or ``steps`` run
    out."""
    s = settings or CopySettings(seed=0)
    torch.manual_seed(s.seed)
    S = int((d["list"].shape[1] - RC.O.ENTRIES_BEYOND_SLOTS))
    soldier = SoldierNet(net, sizes_for(S))
    # a constant step size: the plan is far away
    cop = C.Copier(soldier, s, device, planned=10 ** 12)
    batch = {k: v for k, v in d.items() if k != "files"}
    t0 = time.perf_counter()
    agree = None
    for k in range(1, steps + 1):
        stats = cop.step(batch)
        if k % check_every == 0 or k == steps:
            agree = agreement(soldier, batch, cop.device)
            if log:
                log(k, stats["loss"], agree)
            if all(a is None or a >= LINE for a in agree):
                break
    return {"steps": k, "agreement": agree, "loss": stats["loss"],
            "seconds": time.perf_counter() - t0,
            "decisions": len(batch["t"]),
            "files": int(len(d["files"])),
            "used": C._used(batch).sum(axis=0).tolist()}
