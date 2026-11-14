"""tools/bench_poison.py -- the poison law's tuning instrument (#29 PX;
``docs/poison_law_design_2026-11-14.md`` §6, Erik's).

Three tables, every number from the real law (``simulation.poison``) and the
real gas row (``exchange.apply_poison_dose``) on a 100 HP marine, no
wall-clock, no draw:

1. ONE DOSE (default one unit, a larva bite): the hit points lost by 5 / 10 /
   30 / 60 s, the total, the peak rate and when, when the law goes dormant,
   the hit events.
2. N SECONDS IN GAS at a density, then out of the cloud: the same columns,
   beside the PRE-PX law's closed form (``[exchange] poison_dps`` = 6 HP a
   second x density while in the gas, nothing after); then, per density, the
   time to kill a marine that STAYS in the gas (law vs pre-PX) and the LETHAL
   EXPOSURE (the shortest stay after which the marine dies anyway).
3. A MATCH SCENE: ``levels/ship_demo`` through ``MatchSession`` (both teams
   idle, no horde), team 0's slot 0 throwing ONE poison grenade into its own
   squad's room at 1 s; per marine the hit points lost by 10 / 30 / 60 / 120 s
   and its death tick, and the POISON hit events in all. Tree-agnostic
   (``--scene`` alone runs on a tree without the law: the "before").

    C:/Users/steen/anaconda3/envs/data/python.exe tools/bench_poison.py
    C:/Users/steen/anaconda3/envs/data/python.exe tools/bench_poison.py --scene

Measurement only: it reads the sim, never writes the config.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

MARINE_HP = 100.0
MARKS_S = (5, 10, 30, 60)
GAS_SECONDS = (2, 5, 10, 17, 30)
DENSITIES = (1.0, 0.5, 0.2)
PRE_PX_DPS = 6.0          # the deleted [exchange] poison_dps


def _tps():
    from config import CFG
    return int(CFG.clock.ticks_per_second)


def _level(h=24, w=24):
    from level_loader import LevelData
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = tm[-1, :] = 1
    tm[:, 0] = tm[:, -1] = 1
    return LevelData(name="bench_poison", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _marine(hp=MARINE_HP):
    from simulation.unit import Unit
    u = Unit("M", x=8, y=8, team=0)
    u.id = 1
    u.current_hp = hp
    return u


class Curve:
    """Per-tick hp lost (points), the tick of death, the hit events."""

    def __init__(self, lost, death_tick, hits):
        self.lost, self.death_tick, self.hits = lost, death_tick, hits

    def by(self, seconds, tps):
        return sum(self.lost[:int(seconds * tps)])

    def total(self):
        return sum(self.lost)

    def peak(self, tps):
        """The peak rate (HP/s) over a one-second window, and its time
        (the window's end, s): robust to the final flush tick."""
        best, at = 0.0, 0.0
        run = 0.0
        for i, v in enumerate(self.lost):
            run += v
            if i >= tps:
                run -= self.lost[i - tps]
            if run > best:
                best, at = run, (i + 1) / tps
        return best, at


def run_gas(density, gas_ticks, tps, hp=MARINE_HP, max_ticks=None,
            dose_q=0):
    """A marine in poison gas of ``density`` for ``gas_ticks`` ticks, then
    out (or a single ``dose_q`` at tick 0), the law run until it is dormant
    (or the marine dies, or ``max_ticks``)."""
    from simulation import gas_fixed, poison
    from simulation.events import UnitHitEvent
    from simulation.exchange import apply_poison_dose
    from simulation.gamemap import GameMap
    from simulation.gases import POISON as GAS_POISON
    g = GameMap(_level())
    u = _marine(hp)
    q_trace = gas_fixed.quantize_trace_scalar(density) if density else 0
    tiles = u.occupied_tiles()
    if dose_q:
        poison.dose(u, dose_q, -1)
    lost, hits, death = [], 0, None
    t = 0
    limit = max_ticks if max_ticks is not None else 24 * 3600
    while t < limit:
        on = t < gas_ticks
        for (tx, ty) in tiles:
            g.gas[GAS_POISON][ty, tx] = q_trace if on else 0
        events = []
        h0 = u.hp_q
        apply_poison_dose([u], g, tps, events=events)
        poison.tick([u], t, events)
        lost.append((h0 - u.hp_q) / 65536.0)
        hits += sum(1 for e in events if isinstance(e, UnitHitEvent))
        t += 1
        if not u.alive:
            death = t
            break
        if not on and not poison.holds(u) and t >= gas_ticks:
            break
    return Curve(lost, death, hits)


def table_one_dose(units=1.0):
    from simulation import poison
    tps = _tps()
    c = run_gas(0.0, 0, tps, hp=1e6, dose_q=int(round(units * 65536)))
    pk, at = c.peak(tps)
    s = poison.get_settings()
    print(f"\n1. ONE DOSE of {units:g} unit(s)  (a_onset {s.onset_q}, a_decay "
          f"{s.decay_q}, H {s.harm_q / 65536:g} HP/unit)")
    print("   lost by " + "  ".join(f"{m:>3d}s {c.by(m, tps):6.2f}"
                                    for m in MARKS_S)
          + f"   total {c.total():.3f} HP")
    print(f"   peak {pk:.3f} HP/s at {at:.2f} s; dormant after "
          f"{len(c.lost) / tps:.1f} s; {c.hits} hit events")


def table_gas():
    tps = _tps()
    print("\n2. N SECONDS IN GAS, then out (a 100 HP marine; HP lost; "
          "pre-PX = 6 HP/s x density while in the gas)")
    print(f"   {'dens':>4s} {'N s':>4s} | " + " ".join(
        f"{m:>3d}s law/pre " for m in MARKS_S)
          + "|  total  peak HP/s @s   events  dormant s")
    for d in DENSITIES:
        for n in GAS_SECONDS:
            c = run_gas(d, n * tps, tps, hp=1e6)
            pk, at = c.peak(tps)
            cols = []
            for m in MARKS_S:
                pre = PRE_PX_DPS * d * min(m, n)
                cols.append(f"{c.by(m, tps):6.1f}/{pre:4.0f}")
            print(f"   {d:4.1f} {n:4d} | " + " ".join(cols)
                  + f" | {c.total():6.1f}  {pk:5.2f} @{at:5.1f}  {c.hits:6d}"
                  f"  {len(c.lost) / tps:7.1f}")
    print("\n   STAYING in the gas (100 HP): time to kill, law vs pre-PX; "
          "the LETHAL EXPOSURE (the shortest stay that kills anyway)")
    for d in DENSITIES:
        c = run_gas(d, 24 * 3600, tps, max_ticks=24 * 3600)
        kill = c.death_tick / tps if c.death_tick else float("nan")
        pre = MARINE_HP / (PRE_PX_DPS * d)
        lethal = _lethal_exposure(d, tps)
        print(f"   density {d:3.1f}: dies after {kill:6.1f} s "
              f"(pre-PX {pre:5.1f} s); lethal exposure "
              f"{lethal / tps:6.2f} s ({lethal} ticks)")


def _lethal_exposure(density, tps):
    """The fewest ticks in the gas after which a 100 HP marine dies anyway
    (bisection over the law: the outcome is monotone in the stay)."""
    lo, hi = 1, 24 * 600
    while lo < hi:
        mid = (lo + hi) // 2
        if run_gas(density, mid, tps).death_tick is not None:
            hi = mid
        else:
            lo = mid + 1
    return lo


def run_scene(seconds=120, throw_at_s=1.0, on_tick=None):
    """ship_demo, both teams idle, no horde: team 0's slot 0 throws one
    poison grenade (handed to it: the rifleman carries none) at its own
    squad's centre at ``throw_at_s``. ``on_tick(sim)`` after every tick (the
    sole-cause driver's tape). Returns the squad, its starting hp, the hp
    lost by each mark, the death ticks, the POISON hit events and the
    throw's distance."""
    import breach_physics as bp
    import level_loader
    from simulation.events import UnitHitEvent, UnitKilledEvent
    from simulation.match_session import MatchSession
    from simulation.weapons import get_tables
    tps = _tps()
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    spec = level.match
    spec = replace(spec, zombies=0, time_limit_s=float(seconds),
                   teams=tuple(replace(t, controller="idle")
                               for t in spec.teams))
    session = MatchSession(level, spec, breach_physics=bp)
    sim = session.sim
    squad = sorted((u for u in sim.units if u.team == 0),
                   key=lambda u: u.slot)
    thrower = squad[0]
    kind = get_tables().hand_grenade_rows.index("grenade_poison")
    thrower.grenade_counts[kind] += 1           # the bench hands it one
    cx = sum(u.x for u in squad) / len(squad)
    cy = sum(u.y for u in squad) / len(squad)
    dx, dy = cx - thrower.x, cy - thrower.y
    dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = int(65536 * dx / dist), int(65536 * dy / dist)
    hp0 = {u.id: u.current_hp for u in squad}
    lost_at = {u.id: {} for u in squad}
    death = {}
    hits = 0
    marks = (10, 30, 60, 120)
    throw_tick = int(throw_at_s * tps)
    while not session.is_terminal() and sim.tick < seconds * tps:
        if sim.tick < throw_tick + 12:
            sim.set_aim(thrower.id, ux, uy)
        if sim.tick == throw_tick:
            sim.throw_grenade_intent(thrower.id, ux, uy, 24,
                                     int(dist * 65536), kind)
        session.tick()
        if on_tick is not None:
            on_tick(sim)
        for e in sim.tick_events:
            if isinstance(e, UnitHitEvent) and e.source == "poison_gas":
                hits += 1
            if isinstance(e, UnitKilledEvent) and e.unit_id in hp0:
                death[e.unit_id] = sim.tick
        for m in marks:
            if sim.tick == m * tps:
                for u in squad:
                    lost_at[u.id][m] = hp0[u.id] - max(0.0, u.current_hp)
    return squad, lost_at, death, hits, dist, marks


def table_scene(seconds=120, throw_at_s=1.0):
    tps = _tps()
    squad, lost_at, death, hits, dist, marks = run_scene(seconds, throw_at_s)
    print(f"\n3. MATCH SCENE: ship_demo, idle teams, no horde; slot 0 throws "
          f"one poison grenade at its squad's centre ({dist:.1f} tiles) at "
          f"{throw_at_s:g} s; {seconds} s")
    for u in squad:
        row = "  ".join(f"{m:>3d}s {lost_at[u.id].get(m, float('nan')):6.1f}"
                        for m in marks)
        d = death.get(u.id)
        print(f"   {u.name:6s} lost {row}   "
              + (f"dies at {d / tps:6.1f} s" if d else "lives"))
    print(f"   POISON hit events: {hits}")
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench_poison")
    ap.add_argument("--scene", action="store_true",
                    help="only the match scene (runs on a tree without the "
                         "law: the before)")
    ap.add_argument("--dose", type=float, default=1.0,
                    help="table 1's dose in units (default 1: a bite)")
    args = ap.parse_args(argv)
    if not args.scene:
        table_one_dose(args.dose)
        table_gas()
    table_scene()
    return 0


if __name__ == "__main__":
    sys.exit(main())
