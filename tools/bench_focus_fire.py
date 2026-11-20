"""tools/bench_focus_fire.py -- rounds and seconds to kill under focus fire
(gameplay pack v1 G4p, issue #181).

n marines with the shipped loadout (``[marine] loadout``, the carbine) stand
at a fixed distance from ONE target -- a zombie, or a marine of another team
on ``[env_profiles.default]`` -- and fire at it until it dies, in one of two
modes:

- ``continuous``: every shooter holds the trigger from the same tick, so the
  volleys land together (the focus-fire case the law is for);
- ``single``: every shooter pulls the trigger once a second, on the same tick
  (one round each per second; the exposure has decayed between pulls).

Per case it reports the rounds fired until the kill (in all and per shooter),
the hits, the seconds from the first round to the kill and the peak kinetic
exposure the target reached. Reloads happen as in the game: a magazine that
empties starts the automatic reload, the loadout's spares are counted, and a
case whose shooters run dry before the kill is reported as no kill.

The sim is the real one -- a ``ContinuousRealtime`` ``Simulation`` with
physics in an open synthetic arena, the shooters driven through the intent
facade (``set_aim`` / ``set_trigger``), the horde handed to a controller that
never drives it (the zombie stands still) -- so the damage chain, the attack
slot's simultaneity and the exposure law are the game's own; nothing here
models damage. Deterministic: no window, a fixed seed, no draw of its own
(the carbine's spread draws from the sim's RNG; at the bench distance every
round of the cone lands on the 3x3 target).

    C:/Users/steen/anaconda3/envs/data/python.exe tools/bench_focus_fire.py
    C:/Users/steen/anaconda3/envs/data/python.exe tools/bench_focus_fire.py \\
        --targets zombie --max-n 4 --csv focus_fire.csv

Measurement only: it reads the sim, never writes the config.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

SEED = 181
#: The arena: hull border, open air inside (1 m tiles).
ARENA_H, ARENA_W = 40, 48
#: Where the target's footprint anchors; shooters stand on an arc around it.
TARGET_XY = (34, 18)
DISTANCE_TILES = 12
#: Shooters' bearings from the target (degrees, y down), one per shooter.
BEARINGS_DEG = (180.0, 150.0, 210.0, 120.0, 240.0, 165.0, 195.0, 135.0)
MAX_SECONDS = 60
MODES = ("continuous", "single")
TARGETS = ("zombie", "marine")


@dataclass
class Case:
    target: str
    mode: str
    n: int
    killed: bool
    rounds: int
    rounds_per_shooter: float
    hits: int
    seconds: float
    peak_exposure: float


def _arena():
    from level_loader import LevelData
    tm = np.ones((ARENA_H, ARENA_W), dtype=np.int32)      # hull
    tm[1:ARENA_H - 1, 1:ARENA_W - 1] = 4                    # air
    return LevelData(name="focus_fire_bench", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _unit_vector_q(dx: float, dy: float):
    from simulation import unit_fixed
    d = math.hypot(dx, dy)
    return (unit_fixed.quantize_scalar(dx / d),
            unit_fixed.quantize_scalar(dy / d))


def run_case(target: str, n: int, mode: str,
             distance: int = DISTANCE_TILES) -> Case:
    """One case on a fresh sim (the config as loaded, so a caller that edits
    ``CFG`` before calling measures its own dials)."""
    import breach_physics as bp
    from config import CFG
    from simulation import Simulation
    from simulation.damage import KINETIC
    from simulation.events import ShotFiredEvent, UnitHitEvent
    from simulation.factions import ZOMBIE_TEAM
    from simulation.ruleset import ContinuousRealtime
    from simulation.unit import Unit
    from simulation.unit_fixed import FP_ONE

    if target not in TARGETS or mode not in MODES:
        raise ValueError(f"target in {TARGETS}, mode in {MODES}")
    if not 1 <= n <= len(BEARINGS_DEG):
        raise ValueError(f"n in 1..{len(BEARINGS_DEG)}")
    tps = int(CFG.clock.ticks_per_second)
    sim = Simulation(_arena(), seed=SEED, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.hand_zombies_to_controller()          # nobody drives the horde
    tx, ty = TARGET_XY
    if target == "zombie":
        tgt = Unit("target", x=tx, y=ty, team=ZOMBIE_TEAM)
    else:
        tgt = Unit("target", x=tx, y=ty, team=1, env_profile="default")
    sim.add_unit(tgt)
    shooters = []
    for i in range(n):
        b = math.radians(BEARINGS_DEG[i])
        sx = round(tx + distance * math.cos(b))
        sy = round(ty + distance * math.sin(b))
        u = Unit(f"S{i}", x=sx, y=sy, team=0, env_profile="default")
        sim.add_unit(u)
        shooters.append(u)
    sim.set_paused(False)
    ids = {u.id for u in shooters}
    aims = {u.id: _unit_vector_q(tgt.x - u.x, tgt.y - u.y) for u in shooters}

    def aim_all():
        for uid, (ax, ay) in aims.items():
            sim.set_aim(uid, ax, ay)

    # Turn every shooter onto the target first (the turn-rate cap), triggers
    # released, so the first rounds of every shooter leave on one tick.
    for _ in range(4 * tps):
        aim_all()
        before = [u.facing for u in shooters]
        sim.step()
        sim.set_paused(False)
        if all(u.facing == f for u, f in zip(shooters, before)):
            break

    rounds = hits = 0
    first_tick = None
    peak = 0
    killed = False
    for k in range(MAX_SECONDS * tps):
        aim_all()
        if mode == "continuous":
            held = True
        else:
            held = (k % tps) == 0
        for uid in ids:
            sim.set_trigger(uid, held)
        sim.step()
        sim.set_paused(False)
        for e in sim.tick_events:
            if isinstance(e, ShotFiredEvent) and e.launch and e.unit_id in ids:
                rounds += 1
                if first_tick is None:
                    first_tick = sim.tick - 1
            elif isinstance(e, UnitHitEvent) and e.unit_id == tgt.id \
                    and e.attacker_id in ids:
                hits += 1
        peak = max(peak, int(tgt.exposure[KINETIC]))
        if not tgt.alive:
            killed = True
            break
    seconds = (sim.tick - first_tick) / tps if first_tick is not None else 0.0
    return Case(target=target, mode=mode, n=n, killed=killed, rounds=rounds,
                rounds_per_shooter=rounds / n, hits=hits,
                seconds=round(seconds, 3),
                peak_exposure=round(peak / FP_ONE, 4))


def run_table(targets=TARGETS, modes=MODES, max_n: int = 4,
              distance: int = DISTANCE_TILES) -> list:
    return [run_case(t, n, m, distance)
            for t in targets for m in modes for n in range(1, max_n + 1)]


def _print(cases) -> None:
    print(f"{'target':7s} {'mode':10s} {'n':>2s} {'rounds':>7s} "
          f"{'/shooter':>8s} {'hits':>5s} {'seconds':>8s} {'peak x':>7s}")
    for c in cases:
        r = str(c.rounds) if c.killed else f">{c.rounds}"
        print(f"{c.target:7s} {c.mode:10s} {c.n:2d} {r:>7s} "
              f"{c.rounds_per_shooter:8.1f} {c.hits:5d} "
              f"{c.seconds if c.killed else float('nan'):8.2f} "
              f"{c.peak_exposure:7.3f}" + ("" if c.killed else "  NO KILL"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench_focus_fire")
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--max-n", type=int, default=4)
    ap.add_argument("--distance", type=int, default=DISTANCE_TILES)
    ap.add_argument("--csv", metavar="PATH")
    args = ap.parse_args(argv)
    cases = run_table(tuple(args.targets.split(",")),
                      tuple(args.modes.split(",")), args.max_n, args.distance)
    _print(cases)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(cases[0])))
            w.writeheader()
            for c in cases:
                w.writerow(asdict(c))
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
