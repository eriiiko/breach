"""tools/bench_zombie_melee.py -- one marine against one zombie (Erik's
zombie bite balance, 2026-11-12; the sibling of ``bench_focus_fire.py``).

Erik's rule for the bite: a lone zombie that reaches a marine at contact can
kill the marine before the marine, firing full auto from that moment, kills
it -- unless the marine shoves and steps back and fires again; a marine that
starts firing from range wins.

One marine (the default loadout, ``[marine] loadout``) and one zombie on an
open arena at the shipped 0.333 m tile, the zombie either SPAWNED (a
``ZOMBIE_TEAM`` unit) or TURNED (a marine converted through the game's own
``convert_marines_to_zombies``: it keeps its kit and mass, takes
``[zombie] hp``). Cases:

- ``idle``: the zombie at contact, the marine does nothing -- the bites and
  seconds to kill the marine, and the damage each bite lands;
- ``fire_vs_standing``: the zombie at contact but not driven (it stands),
  the marine fires full auto from tick 0 -- the seconds to kill the zombie;
- ``duel``: both act from contact (the zombie bites, the marine fires);
- ``shove``: both act from contact, the marine shoves on tick 0, walks two
  tiles straight back (trigger released) and resumes full auto;
- ``shove_repeat``: as ``shove``, and again every time the zombie is back
  at contact (the shove's own cooldown applies);
- ``range``: the zombie 10 m away, awake, walking in; the marine fires full
  auto from tick 0.

The sim is the real one (``ContinuousRealtime``, physics on, the conductor's
zombie slot driving the zombie, the marine through the intent facade), so
the bite, the damage chain, the exposure law, the magazines and the unit
impulse law are the game's own. Deterministic: a fixed seed, no draw here.

    C:/Users/steen/anaconda3/envs/data/python.exe tools/bench_zombie_melee.py

Measurement only: it reads the sim, never writes the config.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

SEED = 193
TILE_M = 0.333
ARENA_H, ARENA_W = 40, 90
MARINE_XY = (20, 18)
#: Contact: the two 3x3 footprints touching side by side (centre distance
#: 3 tiles, inside the bite's reach of footprint + 1).
CONTACT_TILES = 3
RANGE_M = 10.0
STEP_BACK_TILES = 2
MAX_SECONDS = 60
CASES = ("idle", "fire_vs_standing", "duel", "shove", "shove_repeat",
         "range")
ZOMBIES = ("spawned", "turned")


@dataclass
class Result:
    zombie: str
    case: str
    winner: str             # "marine" / "zombie" / "none"
    seconds: float          # tick 0 to the kill
    bites: int              # bites that landed on the marine
    bite_damage: float      # hp one bite took (the first)
    marine_hp: float        # left at the end
    zombie_hp: float        # left at the end
    zombie_hp0: float
    rounds: int             # rounds the marine fired


def _arena():
    from level_loader import LevelData
    tm = np.ones((ARENA_H, ARENA_W), dtype=np.int32)      # hull
    tm[1:ARENA_H - 1, 1:ARENA_W - 1] = 4                    # air
    return LevelData(name="zombie_melee_bench", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=TILE_M, diffuse_path=Path("."))


def _unit_vector_q(dx: float, dy: float):
    from simulation import unit_fixed
    d = math.hypot(dx, dy)
    return (unit_fixed.quantize_scalar(dx / d),
            unit_fixed.quantize_scalar(dy / d))


def _centre(u):
    return (u.x + u.footprint / 2.0, u.y + u.footprint / 2.0)


def run_case(zombie: str, case: str) -> Result:
    import breach_physics as bp
    from config import CFG
    from simulation import Simulation
    from simulation.ai_zombie import convert_marines_to_zombies
    from simulation.events import ShotFiredEvent, UnitHitEvent
    from simulation.factions import ZOMBIE_TEAM
    from simulation.ruleset import ContinuousRealtime
    from simulation.turning import PI_Q16, place
    from simulation.unit import Unit

    if zombie not in ZOMBIES or case not in CASES:
        raise ValueError(f"zombie in {ZOMBIES}, case in {CASES}")
    tps = int(CFG.clock.ticks_per_second)
    sim = Simulation(_arena(), seed=SEED, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    mx, my = MARINE_XY
    gap = (round(RANGE_M / TILE_M) if case == "range" else CONTACT_TILES)
    marine = Unit("M", x=mx, y=my, team=0, env_profile="default")
    sim.add_unit(marine)
    if zombie == "spawned":
        z = Unit("Z", x=mx + gap, y=my, team=ZOMBIE_TEAM)
        sim.add_unit(z)
    else:
        z = Unit("Z", x=mx + gap, y=my, team=1, env_profile="default")
        sim.add_unit(z)
        z.alive = False
        z.current_hp = 0.0
        z.killed_by_zombie = True
        convert_marines_to_zombies([z])
    z.zombie_activated = True
    place(marine, 0)                 # facing the zombie (+x)
    place(z, PI_Q16)
    if case == "fire_vs_standing":
        sim.hand_zombies_to_controller()     # nobody drives it: it stands
    sim.set_paused(False)
    zhp0 = float(z.current_hp)

    fires = case != "idle"
    phase = "shove" if case.startswith("shove") else "fire"
    start_cx = _centre(marine)[0]
    bites = rounds = 0
    bite_damage = 0.0
    winner = "none"
    for _ in range(MAX_SECONDS * tps):
        mcx, mcy = _centre(marine)
        zcx, zcy = _centre(z)
        ax, ay = _unit_vector_q(zcx - mcx, zcy - mcy)
        if fires:
            sim.set_aim(marine.id, ax, ay)
        if (case == "shove_repeat" and phase == "fire"
                and math.hypot(zcx - mcx, zcy - mcy) <= CONTACT_TILES + 0.5):
            phase = "shove"
            start_cx = mcx
        if phase == "shove":
            sim.shove_intent(marine.id)
            sim.set_trigger(marine.id, False)
            phase = "back"
        elif phase == "back":
            if start_cx - mcx >= STEP_BACK_TILES:
                sim.clear_move_dir(marine.id)
                phase = "fire"
            else:
                sim.set_move_dir(marine.id, -ax, -ay, 0)
                sim.set_trigger(marine.id, False)
        if phase == "fire" and fires:
            sim.set_trigger(marine.id, True)
        hp_before = float(marine.current_hp)
        sim.step()
        sim.set_paused(False)
        for e in sim.tick_events:
            if isinstance(e, ShotFiredEvent) and e.launch \
                    and e.unit_id == marine.id:
                rounds += 1
            elif isinstance(e, UnitHitEvent) and e.unit_id == marine.id \
                    and e.attacker_id == z.id:
                bites += 1
                if bites == 1:
                    bite_damage = hp_before - max(
                        0.0, float(marine.current_hp)) if marine.alive \
                        else hp_before
        if not z.alive or (not z.is_zombie):
            winner = "marine"
            break
        if not marine.alive or marine.is_zombie:
            winner = "zombie"
            break
    return Result(zombie=zombie, case=case, winner=winner,
                  seconds=round(sim.tick / tps, 3), bites=bites,
                  bite_damage=round(bite_damage, 3),
                  marine_hp=round(max(0.0, float(marine.current_hp))
                                  if not marine.is_zombie else 0.0, 2),
                  zombie_hp=round(max(0.0, float(z.current_hp))
                                  if z.alive else 0.0, 2),
                  zombie_hp0=zhp0, rounds=rounds)


def _print(rows) -> None:
    from config import CFG
    print(f"[zombie] melee_damage = {CFG.zombie.melee_damage}, "
          f"attack_cooldown_seconds = {CFG.zombie.attack_cooldown_seconds} "
          f"({CFG.zombie.attack_cooldown_ticks} ticks)")
    print(f"{'zombie':8s} {'case':17s} {'winner':7s} {'s':>6s} {'bites':>5s}"
          f" {'bite':>6s} {'m hp':>6s} {'z hp':>7s} {'z hp0':>6s} "
          f"{'rounds':>6s}")
    for r in rows:
        print(f"{r.zombie:8s} {r.case:17s} {r.winner:7s} {r.seconds:6.2f} "
              f"{r.bites:5d} {r.bite_damage:6.1f} {r.marine_hp:6.1f} "
              f"{r.zombie_hp:7.1f} {r.zombie_hp0:6.0f} {r.rounds:6d}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench_zombie_melee")
    ap.add_argument("--zombies", default=",".join(ZOMBIES))
    ap.add_argument("--cases", default=",".join(CASES))
    args = ap.parse_args(argv)
    rows = [run_case(zk, c) for zk in args.zombies.split(",")
            for c in args.cases.split(",")]
    _print(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
