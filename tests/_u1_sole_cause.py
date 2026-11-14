"""#29 U1's SOLE-CAUSE driver: the hooks enter at neutral, so the game, every
digest and every observation byte must not change.

Reuses P3a's / P3c's scenarios (``tests/_p3c_sole_cause.py``: ship_demo bots,
ship_demo bots_v1, the #122 conversion box, the scripted arena of every verb,
the one-phase WEGO timeline and TwoPhaseWEGO's path) -- per tick the schema's
unit records, the planted charges, the synced events and the field digest --
and adds ``obs_ship_demo``: a ship_demo bots-vs-bots match whose every
observation array (observation spec v1's ``vector`` / ``list`` / ``list_mask``
/ ``list_uid`` / the four windows, the full view ``full_v1``'s five arrays and
action spec v1's four masks) is built every 4 ticks over every roster row and
recorded as bytes, beside the per-tick unit records and field digests.

Run on main (a detached checkout with this file copied in) and on the branch,
then ``--compare``: equal everywhere = the hooks moved nothing.

    <py> tests/_u1_sole_cause.py OUT.pkl [scenario ...]
    <py> tests/_u1_sole_cause.py --compare A.pkl B.pkl
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import _p3c_sole_cause as C  # noqa: E402  (installs the P3c scenarios on P)

P = C.P


def scenario_obs_ship_demo(n_ticks=600, every=4):
    import level_loader
    import rl_env
    from simulation import obs_spec_v1 as O
    from simulation.action_spec_v1 import (MASK_KEYS, action_tables,
                                           build_masks, mask_shapes)
    from simulation.match_session import MatchSession
    from simulation.sensor_accessor import EntityFieldAccessor

    lv = level_loader.load("ship_demo", str(ROOT / "levels"))
    spec = replace(lv.match, teams=tuple(replace(t, controller="bots")
                                         for t in lv.match.teams))
    ses = MatchSession(lv, spec, seed=1, breach_physics=P.bp)
    sim = ses.sim
    S = max(rl_env._team_slots(lv))
    T = len(spec.teams)
    U = T * S
    ctx = O.ObsContext.build(sim, spec, S)
    tables = action_tables(sim, S)
    acc = EntityFieldAccessor(sim.gmap)
    obs_shapes = {k: v for k, v in O.shapes(1, U, S).items()
                  if k in O.OBS_KEYS}
    full_shapes = O.full_shapes(1, U, T)
    m_shapes = mask_shapes(1, U, S + O.ENTRIES_BEYOND_SLOTS,
                           len(sim.weapons_tables.hand_grenade_rows))
    tape = P._Tape()
    observed = 0
    for _ in range(n_ticks):
        if ses.is_terminal():
            break
        ses.tick()
        tape(sim)
        if sim.tick % every:
            continue
        live = [None] * U
        roster = [None] * U
        for t in range(T):
            for u in sim.roster(t):
                roster[t * S + u.slot] = u
                if u.alive and u.team == t:
                    live[t * S + u.slot] = u
        out = {k: np.zeros(shape[1:], dtype)
               for k, (shape, dtype) in obs_shapes.items()}
        out["list_uid"][...] = -1
        acc.observe("v1", live, out, sim=sim, context=ctx)
        masks = {k: np.zeros(shape[1:], dtype)
                 for k, (shape, dtype) in m_shapes.items()}
        build_masks(sim, live, out["list"][:, :, 0], out["list_mask"],
                    out["list_uid"], tables, every, masks)
        full = {k: np.zeros(shape[1:], dtype)
                for k, (shape, dtype) in full_shapes.items()}
        acc.observe("full_v1", roster, full, sim=sim, context=ctx)
        h = {}
        for k in O.OBS_KEYS:
            h[k] = hashlib.sha256(np.ascontiguousarray(out[k]).tobytes()
                                  ).hexdigest()
        for k in MASK_KEYS:
            h[k] = hashlib.sha256(np.ascontiguousarray(masks[k]).tobytes()
                                  ).hexdigest()
        for k in O.FULL_KEYS:
            h[k] = hashlib.sha256(np.ascontiguousarray(full[k]).tobytes()
                                  ).hexdigest()
        # the per-tick record gains the observation's hashes on its own slot
        tick, recs, charges, evs, fd = tape.ticks[-1]
        tape.ticks[-1] = (tick, recs, charges, evs, fd, tuple(sorted(h.items())))
        observed += 1
        tape.cov["obs_rows_live"] += sum(u is not None for u in live)
        tape.cov["obs_list_entries"] += int(out["list_mask"].sum())
    tape.cov["observations"] = observed
    tape.cov["conversions"] = P._conversions(sim)
    return tape, ("obs_ship_demo", int(sim.tick))


SCENARIOS = dict(P.SCENARIOS)
SCENARIOS["obs_ship_demo"] = scenario_obs_ship_demo
P.SCENARIOS = SCENARIOS


def compare(a_path: str, b_path: str) -> int:
    """P3a's compare, plus the observation hashes (the sixth slot of an
    observed tick)."""
    import pickle
    bad = P.compare(a_path, b_path)
    with open(a_path, "rb") as fh:
        a = pickle.load(fh)
    with open(b_path, "rb") as fh:
        b = pickle.load(fh)
    for name in sorted(set(a) & set(b)):
        n_obs = 0
        for x, y in zip(a[name]["ticks"], b[name]["ticks"]):
            if len(x) > 5 or len(y) > 5:
                n_obs += 1
                if x[5:] != y[5:]:
                    diff = [k for (k, va), (_, vb) in zip(x[5], y[5])
                            if va != vb]
                    print(f"{name}: observation differs at tick {x[0]} in "
                          f"{diff}")
                    bad = 1
                    break
        if n_obs:
            print(f"{name}: {n_obs} observations compared")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    P.run(sys.argv[1], sys.argv[2:] or None)
