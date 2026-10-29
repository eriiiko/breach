"""tools/bench_bot_matches.py -- bots-vs-bots match statistics (#121).

Runs a level's match.toml with EVERY marine team on the ``bots`` controller
(the zombies as configured) over a range of seeds, through the one match loop
(``run_match.run_match`` and its ``on_tick`` hook), and prints per seed and in
total: the outcome (team 0 / team 1 wins, draw), ticks to the result, the
deaths by cause, the deaths by ATTACKER TEAM (#123) and the wall time.

    C:/Users/steen/anaconda3/python.exe tools/bench_bot_matches.py \\
        --level ship_demo --seeds 1-10

Death causes come from the tick events: a ``UnitKilledEvent``'s ``killed_by``
("bullet", "explosion" = a frag, "heat", "melee" = a zombie's bite, credited
since #122). A marine that dies or turns WITHOUT a kill event is counted as
"zombie" (the pre-#122 bucket; it should now stay empty -- an entry there is
an uncredited death for the credited-events audit). Marine deaths and zombie
deaths are counted apart.

By attacker team (#123, friendly fire is ON): the kill event's ``attacker_id``
against the victim's team at the tick before it died -- ``self`` (its own
grenade), ``own`` (a team-mate: a team-kill), ``enemy`` (any other team, the
horde included), ``environment`` (attacker -1: a fire, a cloud, a timed
charge) and ``uncredited`` (no kill event at all). The attacker's team is the
one it ACTED in: its team now for a direct cause (a shot, a bite), but its
last MARINE team for an indirect one (a blast, a jet's heat or gas) -- a
marine killed and converted while its grenade was in flight threw it as a
marine (zombies carry no weapons). Every death lands in
exactly one cause bucket and one attacker bucket, so the two tables of a
victim group have the same total. Measurement only: it reads the sim, never
writes.
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
from simulation.events import (  # noqa: E402
    NO_ATTACKER, ShotFiredEvent, UnitKilledEvent)


INDIRECT_CAUSES = ("explosion", "heat", "poison_gas")


def attacker_relation(victim_id, victim_team, attacker_id, teams) -> str:
    """Which side a kill came from (#123): ``self`` / ``own`` / ``enemy`` /
    ``environment``; ``teams`` maps a unit id to the team it acted in."""
    if attacker_id == NO_ATTACKER:
        return "environment"
    if attacker_id == victim_id:
        return "self"
    return "own" if teams.get(attacker_id) == victim_team else "enemy"


class DeathLog:
    """``on_tick`` watcher: who died (or turned) this tick, and of what."""

    def __init__(self):
        self.marine = Counter()
        self.zombie = Counter()
        self.marine_by = Counter()   # by attacker team (#123)
        self.zombie_by = Counter()
        self.shots = 0
        self.throws = 0
        self._prev = None        # id -> (alive, is_zombie, team)
        self._marine_team = {}   # id -> its team while a living marine
        self._grenades = None

    def __call__(self, sim):
        killed = {}
        for ev in sim.tick_events:
            if isinstance(ev, UnitKilledEvent):
                killed[ev.unit_id] = (ev.killed_by, ev.attacker_id)
            elif isinstance(ev, ShotFiredEvent) and ev.launch:
                self.shots += 1
        now = {u.id: (u.alive, u.is_zombie, u.team) for u in sim.units}
        teams = {uid: t for uid, (_a, _z, t) in now.items()}
        if self._prev is None:
            # The first call follows tick 1: everyone was alive at spawn.
            self._prev = {u.id: (True, u.is_zombie, u.team)
                          for u in sim.units}
        grenades = sum(int(getattr(u, "has_grenade", 0)) for u in sim.units)
        if self._grenades is not None and grenades < self._grenades:
            self.throws += self._grenades - grenades
        self._grenades = grenades
        for uid, (alive, zomb, team) in self._prev.items():
            if not alive:
                continue
            n_alive, n_zomb, _t = now[uid]
            if uid in killed:
                cause, attacker = killed[uid]
                acted = teams
                if cause in INDIRECT_CAUSES and attacker in self._marine_team:
                    acted = {attacker: self._marine_team[attacker]}
                side = attacker_relation(uid, team, attacker, acted)
            else:
                cause, side = None, "uncredited"
            if zomb:
                if not n_alive:
                    self.zombie[cause or "other"] += 1
                    self.zombie_by[side] += 1
            elif not n_alive or n_zomb:
                self.marine[cause or "zombie"] += 1
                self.marine_by[side] += 1
        for u in sim.units:
            if u.alive and not u.is_zombie:
                self._marine_team[u.id] = u.team
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
    marine_by, zombie_by = Counter(), Counter()
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
        marine_by.update(log.marine_by)
        zombie_by.update(log.zombie_by)
        print(f"seed {seed:3d}: {key:12s} ticks {r.ticks:5d} "
              f"living {o.living} turned {o.converted_alive} "
              f"marine deaths {dict(log.marine)} by {dict(log.marine_by)} "
              f"zombie deaths {dict(log.zombie)} by {dict(log.zombie_by)} "
              f"shots {log.shots} frags {log.throws} "
              f"wall {wall:6.1f} s digest {o.tick_digest[:12]}", flush=True)
    n = len(ticks)
    print(f"outcomes: {dict(outcomes)}")
    print(f"mean ticks to result: {sum(ticks) / n:.0f} "
          f"({sum(ticks) / n / 24.0:.1f} s sim)")
    print(f"marine deaths by cause: {dict(marine)}")
    print(f"zombie deaths by cause: {dict(zombie)}")
    print(f"marine deaths by attacker team: {dict(marine_by)}")
    print(f"zombie deaths by attacker team: {dict(zombie_by)}")
    print(f"wall per match: mean {sum(walls) / n:.1f} s, max {max(walls):.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
