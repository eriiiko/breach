"""tools/bench_bot_matches.py -- bots-vs-bots match statistics (#121).

Runs a level's match.toml with EVERY marine team on the ``bots`` controller
(the zombies as configured) over a range of seeds, through the one match loop
(``run_match.run_match`` and its ``on_tick`` hook), and prints per seed and in
total: the outcome (team 0 / team 1 wins, draw), ticks to the result, the
deaths by cause and the wall time.

    C:/Users/steen/anaconda3/python.exe tools/bench_bot_matches.py \\
        --level ship_demo --seeds 1-10

Death causes come from the tick events: a ``UnitKilledEvent``'s ``killed_by``
("bullet", "explosion" = a frag, "heat", ...). A zombie's bite emits no event
(``ai_zombie`` applies it with ``events=None``), so a marine that dies or
turns without a kill event is counted as "zombie". Marine deaths and zombie
deaths are counted apart. Measurement only: it reads the sim, never writes.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tests", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation.events import ShotFiredEvent, UnitKilledEvent  # noqa: E402


class DeathLog:
    """``on_tick`` watcher: who died (or turned) this tick, and of what."""

    def __init__(self):
        self.marine = Counter()
        self.zombie = Counter()
        self.shots = 0
        self.throws = 0
        self._prev = None        # id -> (alive, is_zombie)
        self._grenades = None

    def __call__(self, sim):
        killed = {}
        for ev in sim.tick_events:
            if isinstance(ev, UnitKilledEvent):
                killed[ev.unit_id] = ev.killed_by
            elif isinstance(ev, ShotFiredEvent) and ev.launch:
                self.shots += 1
        now = {u.id: (u.alive, u.is_zombie) for u in sim.units}
        if self._prev is None:
            # The first call follows tick 1: everyone was alive at spawn.
            self._prev = {u.id: (True, u.is_zombie) for u in sim.units}
        grenades = sum(int(getattr(u, "has_grenade", 0)) for u in sim.units)
        if self._grenades is not None and grenades < self._grenades:
            self.throws += self._grenades - grenades
        self._grenades = grenades
        for uid, (alive, zomb) in self._prev.items():
            if not alive:
                continue
            n_alive, n_zomb = now[uid]
            if zomb:
                if not n_alive:
                    self.zombie[killed.get(uid, "other")] += 1
            elif not n_alive or n_zomb:
                self.marine[killed.get(uid, "zombie")] += 1
        self._prev = now


def _seeds(text: str) -> list:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench_bot_matches")
    ap.add_argument("--level", required=True)
    ap.add_argument("--seeds", default="1-10", help="'1-10' or '1,4,7'")
    ap.add_argument("--levels-dir", default="levels")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    level = level_loader.load(args.level, args.levels_dir)
    spec = level.match
    spec = replace(spec, teams=tuple(replace(t, controller="bots")
                                     for t in spec.teams))
    print(f"level={args.level} teams n={[t.n for t in spec.teams]} "
          f"zombies={spec.zombies} time_limit_s={spec.time_limit_s}")
    outcomes = Counter()
    ticks, walls = [], []
    marine, zombie = Counter(), Counter()
    for seed in _seeds(args.seeds):
        log = DeathLog()
        t0 = time.perf_counter()
        o = run_match(level, spec, seed=seed, on_tick=log)
        wall = time.perf_counter() - t0
        r = o.result
        key = f"team {r.team} wins" if r.outcome == "win" else "draw"
        outcomes[key] += 1
        ticks.append(r.ticks)
        walls.append(wall)
        marine.update(log.marine)
        zombie.update(log.zombie)
        print(f"seed {seed:3d}: {key:12s} ticks {r.ticks:5d} "
              f"living {o.living} turned {o.converted_alive} "
              f"marine deaths {dict(log.marine)} zombie deaths "
              f"{dict(log.zombie)} shots {log.shots} frags {log.throws} "
              f"wall {wall:6.1f} s")
    n = len(ticks)
    print(f"outcomes: {dict(outcomes)}")
    print(f"mean ticks to result: {sum(ticks) / n:.0f} "
          f"({sum(ticks) / n / 24.0:.1f} s sim)")
    print(f"marine deaths by cause: {dict(marine)}")
    print(f"zombie deaths by cause: {dict(zombie)}")
    print(f"wall per match: mean {sum(walls) / n:.1f} s, max {max(walls):.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
