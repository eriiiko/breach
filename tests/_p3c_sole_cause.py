"""#29 P3c's SOLE-CAUSE driver: positions become Q16 (``x_q`` / ``y_q``) and
the game must not change, only the digest's definition.

Reuses P3a's scenarios (``tests/_p3a_sole_cause.py``: ship_demo bots,
ship_demo bots_v1, the #122 conversion box, the scripted arena of every
verb) and adds two LEGACY ones (the one-phase WEGO timeline and TwoPhaseWEGO's
precomputed path + the round-end snap), whose interpolated positions are
NOT on the Q16 grid before P3c.

On every tree the per-tick unit record is NORMALISED to the P3c form: a
pre-P3c record's ``x`` / ``y`` (``round(v * 1e9)``) are replaced by ``x_q`` /
``y_q`` = ``round(u.x * 65536)`` read from the unit itself, and every
position that is NOT an exact n/65536 is counted (``off_grid`` in the
coverage) -- on the continuous scenarios there must be none. A P3c record
already carries ``x_q`` / ``y_q``. Then ``--compare`` main's dump with the
branch's: equal records at every tick and every field digest identical =
the re-pin moves the HASH, not the game.

    <py> tests/_p3c_sole_cause.py OUT.pkl [scenario ...]
    <py> tests/_p3c_sole_cause.py --compare A.pkl B.pkl
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import _p3a_sole_cause as P  # noqa: E402

_orig_digest_record = P.unit_schema.digest_record


def _normalised_record(u):
    rec = _orig_digest_record(u)
    if "x" in rec:                               # a pre-P3c record
        out = {}
        for k, v in rec.items():
            if k in ("x", "y"):
                f = float(getattr(u, k))
                q = round(f * 65536)
                if q / 65536 != f:
                    _OFF_GRID[0] += 1
                    _OFF_GRID[1].setdefault(
                        (getattr(u, "name", "?"), k), f)
                out[k + "_q"] = int(q)
            else:
                out[k] = v
        return out
    return rec


_OFF_GRID = [0, {}]
P.unit_schema.digest_record = _normalised_record


def _legacy_level(h=32, w=32):
    from level_loader import LevelData
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1
    return LevelData(name="p3c_legacy", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _legacy(ruleset, n_ticks, orders, order_type):
    from simulation import Simulation
    from simulation import orders as O
    from simulation.factions import ZOMBIE_TEAM
    from simulation.unit import Unit
    sim = Simulation(_legacy_level(), seed=7, breach_physics=P.bp,
                     enable_recorder=False, ruleset=ruleset)
    sim.set_paused(False)
    a = sim.get_unit(sim.add_unit(Unit("a", x=4.0, y=4.0, team=0)))
    b = sim.get_unit(sim.add_unit(Unit("b", x=4.0, y=20.0, team=0)))
    sim.add_unit(Unit("z", x=24.0, y=24.0, team=ZOMBIE_TEAM))
    for u, (tx, ty) in zip((a, b), orders):
        sim.apply_action(u.id, O.Order(getattr(O, order_type), tx, ty, phase=0))
    tape = P._Tape()
    for _ in range(n_ticks):
        sim.set_paused(False)
        sim.step()
        tape(sim)
    tape.cov["conversions"] = P._conversions(sim)
    return tape, (type(ruleset).__name__, int(sim.tick))


def scenario_onephase():
    from simulation.ruleset import OnePhaseWEGO
    return _legacy(OnePhaseWEGO(), 200, ((11, 9), (13, 22)), "ORDER_MOVE")


def scenario_twophase():
    from config import CFG
    from simulation.ruleset import TwoPhaseWEGO
    return _legacy(TwoPhaseWEGO(), CFG.clock.ticks_per_round + 20,
                   ((11, 9), (13, 22)), "ORDER_MOVE_ATTACK")


SCENARIOS = dict(P.SCENARIOS)
SCENARIOS["onephase"] = scenario_onephase
SCENARIOS["twophase"] = scenario_twophase
P.SCENARIOS = SCENARIOS


def run(out, names=None):
    import pickle
    dump = {}
    for name in names or SCENARIOS:
        _OFF_GRID[0] = 0
        _OFF_GRID[1] = {}
        tape, outcome = SCENARIOS[name]()
        tape.cov["off_grid_positions"] = _OFF_GRID[0]
        print(f"{name}: {len(tape.ticks)} ticks, outcome {outcome}")
        print("   coverage: " + ", ".join(
            f"{k}={v}" for k, v in sorted(tape.cov.items())))
        if _OFF_GRID[1]:
            print("   first off-grid values: "
                  + ", ".join(f"{k}={v!r}" for k, v in
                              list(_OFF_GRID[1].items())[:6]))
        dump[name] = dict(ticks=tape.ticks, outcome=outcome,
                          cov=dict(tape.cov))
    with open(out, "wb") as fh:
        pickle.dump(dump, fh, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(P.compare(sys.argv[2], sys.argv[3]))
    run(sys.argv[1], sys.argv[2:] or None)
