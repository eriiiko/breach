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

TIMING (#125, ``--timing``): the wall time of every tick, split into the
controllers' share (every ``handle_frame`` of the tick plus the runner's
``is_terminal`` check: the bots' A* and vision, the horde's law) and
``sim.step()`` (physics + gameplay), as mean / p50 / p95 / max, ticks/s and the
real-time ratio (sim seconds per wall second). The split needs one more
timestamp than ``on_tick`` gives, so the watcher wraps the match's own
``sim.step`` bound method on its first call (a timer around the same call, no
sim state touched); tick 1 -- the first ``step()``, with its warm-up -- is
therefore not in the sample, and the watchers' own time is excluded.
``--controller`` puts every marine team on ``bots`` (the default) or ``idle``;
``--zombies-idle`` leaves the horde standing too (the physics-only baseline);
``--sim-seconds`` replaces the match's time limit (a match that ends sooner
is reported with the ticks it ran); ``--repeat`` runs each seed again;
``--cuda`` runs on the GPU build (``--cuda-build DIR``, default
``cpp/build_cuda``; every live backend on, ``tools/run_on_cuda.py``'s list)
and ``--resident`` adds field residency. Each timed run also prints one
``TIMING_JSON`` line for a report to collect. The #125 tick budget:

    C:/Users/steen/anaconda3/python.exe tools/bench_bot_matches.py \\
        --level ship_s1_medium --levels-dir <scratch>/levels --seeds 1 \\
        --timing --sim-seconds 60 --repeat 2
"""
from __future__ import annotations

import argparse
import json
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


class TickTimer:
    """``on_tick`` watcher (#125): per-tick wall time, split into the
    controllers' share and ``sim.step()``. ``inner`` (another watcher) is
    called first; its time is kept out of both shares."""

    def __init__(self, inner=None):
        self.inner = inner
        self.step_s = []   # sim.step() per tick
        self.ctrl_s = []   # controllers + is_terminal per tick
        self._mark = None  # end of the previous on_tick
        self._t0 = self._t1 = None

    def _wrap(self, sim):
        orig = sim.step

        def timed_step(*a, **k):
            self._t0 = time.perf_counter()
            r = orig(*a, **k)
            self._t1 = time.perf_counter()
            return r
        sim.step = timed_step

    def __call__(self, sim):
        if self.inner is not None:
            self.inner(sim)
        if self._mark is None:
            self._wrap(sim)
        elif self._t0 is not None:
            self.ctrl_s.append(self._t0 - self._mark)
            self.step_s.append(self._t1 - self._t0)
        self._mark = time.perf_counter()

    @staticmethod
    def stats(xs) -> dict:
        import numpy as np
        a = np.asarray(xs, dtype=np.float64) * 1e3
        if a.size == 0:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "max": float(a.max())}

    def summary(self, tps: float) -> dict:
        """ms statistics of total / step / ctrl, ticks/s, real-time ratio."""
        tot = [c + s for c, s in zip(self.ctrl_s, self.step_s)]
        mean_s = sum(tot) / max(1, len(tot))
        rate = 1.0 / mean_s if mean_s else 0.0
        return {"n": len(tot), "total": self.stats(tot),
                "step": self.stats(self.step_s),
                "ctrl": self.stats(self.ctrl_s),
                "ticks_per_s": rate, "realtime": rate / tps}


def _fmt(name, st) -> str:
    return (f"{name} mean {st['mean']:7.2f} p50 {st['p50']:7.2f} "
            f"p95 {st['p95']:7.2f} max {st['max']:8.2f} ms")


def setup_cuda(build_dir: Path, resident: bool) -> None:
    """Import ``breach_physics`` from the CUDA build in ``build_dir`` (before
    anything imports the CPU build) and turn every live backend on -- the
    same list and checks as ``tools/run_on_cuda.py``."""
    import os
    from cuda_harness import cuda_dll_dir
    from run_on_cuda import enable_all_backends
    if not list(Path(build_dir).glob("breach_physics*.pyd")):
        raise SystemExit(f"no CUDA build in {build_dir}")
    dll = cuda_dll_dir()
    if dll is None:
        raise SystemExit("CUDA runtime DLLs (cudart64_*.dll) not found")
    os.add_dll_directory(str(dll))
    sys.path.insert(0, str(build_dir))
    import breach_physics as bp
    if Path(bp.__file__).resolve().parent != Path(build_dir).resolve():
        raise SystemExit(f"breach_physics resolved to {bp.__file__}, "
                         f"not the CUDA build")
    enable_all_backends(bp)
    if resident:
        from simulation import physics_runner
        physics_runner.set_residency(True)
        print("[bench] GPU field residency ON")


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
    ap.add_argument("--timing", action="store_true",
                    help="per-tick wall time, controllers vs sim.step (#125)")
    ap.add_argument("--controller", default="bots",
                    help="controller for every marine team (bots / idle)")
    ap.add_argument("--zombies-idle", action="store_true",
                    help="the horde on 'idle' too (physics-only baseline)")
    ap.add_argument("--sim-seconds", type=float, default=None,
                    help="replace the match's time limit")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run every seed this many times")
    ap.add_argument("--cuda", action="store_true",
                    help="run on the CUDA build, every live backend on")
    ap.add_argument("--cuda-build", default=str(ROOT / "cpp" / "build_cuda"))
    ap.add_argument("--resident", action="store_true",
                    help="with --cuda: GPU field residency on")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.cuda:
        setup_cuda(Path(args.cuda_build), args.resident)
    from run_match import run_match
    import breach_physics as bp
    from config import CFG
    tps = float(CFG.clock.ticks_per_second)
    level = level_loader.load(args.level, args.levels_dir)
    spec = level.match
    spec = replace(spec, teams=tuple(replace(t, controller=args.controller)
                                     for t in spec.teams))
    if args.zombies_idle:
        spec = replace(spec, zombie_controller="idle")
    if args.sim_seconds is not None:
        spec = replace(spec, time_limit_s=float(args.sim_seconds))
    backend = "cpu"
    if args.cuda:
        backend = "cuda+resident" if args.resident else "cuda"
    print(f"level={args.level} teams n={[t.n for t in spec.teams]} "
          f"controller={args.controller} zombies={spec.zombies} "
          f"zombie_controller={spec.zombie_controller} "
          f"time_limit_s={spec.time_limit_s} tps={tps:g} backend={backend} "
          f"module={Path(bp.__file__).parent}")
    outcomes = Counter()
    ticks, walls = [], []
    marine, zombie = Counter(), Counter()
    marine_by, zombie_by = Counter(), Counter()
    for seed in [s for s in _seeds(args.seeds) for _ in range(args.repeat)]:
        log = DeathLog()
        timer = TickTimer(log) if args.timing else None
        t0 = time.perf_counter()
        o = run_match(level, spec, seed=seed,
                      on_tick=log if timer is None else timer)
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
        if timer is not None:
            t = timer.summary(tps)
            print(f"  timing: {t['n']} ticks timed, {t['ticks_per_s']:.1f} "
                  f"ticks/s, real-time x{t['realtime']:.2f}")
            for name in ("total", "step", "ctrl"):
                print("    " + _fmt(f"{name:5s}", t[name]))
            print("  TIMING_JSON " + json.dumps(
                {"level": args.level, "seed": seed,
                 "controller": args.controller,
                 "zombies_idle": args.zombies_idle, "backend": backend,
                 "ticks": r.ticks, "result": key, **t}), flush=True)
    n = len(ticks)
    print(f"outcomes: {dict(outcomes)}")
    print(f"mean ticks to result: {sum(ticks) / n:.0f} "
          f"({sum(ticks) / n / tps:.1f} s sim)")
    print(f"marine deaths by cause: {dict(marine)}")
    print(f"zombie deaths by cause: {dict(zombie)}")
    print(f"marine deaths by attacker team: {dict(marine_by)}")
    print(f"zombie deaths by attacker team: {dict(zombie_by)}")
    print(f"wall per match: mean {sum(walls) / n:.1f} s, max {max(walls):.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
