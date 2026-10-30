"""tools/run_match.py -- THE headless match runner (#119 P2, rl-ready §7).

Runs one match to its result with no window: the level and its match.toml
(``level_loader.load`` -> ``LevelData.match``), a ``Simulation`` seeded with
the match seed under :class:`~simulation.ruleset.MatchRules`, the units
spawned from the config into the level's spawn rooms
(``Simulation.spawn_match``), then ``step()`` until terminal, each team's
registered controller (``control_source.create_match_controllers``, the
horde's included) called once per tick with ``renderer=None``. No light is requested, no recorder allocated,
nothing imports the renderer. Tests and benches call :func:`run_match`;
never a second match loop.

    C:/Users/steen/anaconda3/python.exe tools/run_match.py --level playground
    C:/Users/steen/anaconda3/python.exe tools/run_match.py --level playground \\
        --seed 7 --ticks-per-report 240

Prints the result (``win`` + team, or ``draw``), the ticks, the living roster
slots per team, and the final state's digests: ``field_digest`` (the frozen
integer fields, ``tests/field_digest.py``) and ``tick_digest`` (those folded
with the synced unit state and the entity section -- the one a determinism
gate compares). Exit 0 on a finished match.

Every unit has ONE driver and it is a controller (#122): the teams' rows and
the horde's ``zombie_controller`` (default ``zombies``: the shipped zombie law
through the ZOMBIE intent), built by ``control_source.create_match_controllers``.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import level_loader  # noqa: E402
from control_source import create_match_controllers  # noqa: E402
from field_ab_harness import capture_snapshot  # noqa: E402
from field_digest import field_digest, tick_digest  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.ruleset import MatchResult, MatchRules  # noqa: E402


@dataclass(frozen=True)
class MatchOutcome:
    """What :func:`run_match` returns."""
    result: MatchResult
    seed: int
    #: team -> the roster slots still alive ON that team (ZOMBIE_TEAM: the
    #: spawned zombies' slots; converted marines are counted separately).
    living: dict
    #: living units on ZOMBIE_TEAM that were spawned as marines.
    converted_alive: int
    field_digest: str
    tick_digest: str


def _living(sim, spec) -> tuple[dict, int]:
    teams = list(range(len(spec.teams)))
    if spec.zombies:
        teams.append(ZOMBIE_TEAM)
    living = {t: [u.slot for u in sim.roster(t)
                  if u.alive and u.team == t] for t in teams}
    spawned_zombies = {id(u) for u in sim.roster(ZOMBIE_TEAM)}
    converted = sum(1 for u in sim.units if u.alive and u.is_zombie
                    and id(u) not in spawned_zombies)
    return living, converted


def run_match(level, spec=None, *, seed: int | None = None,
              report_every: int = 0, out=None, on_tick=None) -> MatchOutcome:
    """Run ``spec`` (default ``level.match``) on ``level`` to its result.
    ``seed`` overrides the config's seed; ``report_every`` > 0 prints the
    living slots every that many ticks through ``out`` (default print).
    ``on_tick(sim)`` (#121), if given, is called after every ``step()`` --
    while that tick's ``tick_events`` are still on the sim -- so a bench or a
    test can watch the match without a second match loop; it must not write
    sim state."""
    spec = level.match if spec is None else spec
    if spec is None:
        raise ValueError(f"level {level.name!r} has no match.toml")
    seed = spec.seed if seed is None else int(seed)
    out = print if out is None else out
    rules = MatchRules.from_spec(spec)
    sim = Simulation(level, seed=seed, breach_physics=bp,
                     enable_recorder=False, ruleset=rules)
    sim.set_paused(False)
    sim.spawn_match(spec)
    controllers = create_match_controllers(sim, spec)
    while not rules.is_terminal(sim):
        for c in controllers:
            c.handle_frame(sim, None)
        sim.step()
        if on_tick is not None:
            on_tick(sim)
        if report_every and sim.tick % report_every == 0:
            living, converted = _living(sim, spec)
            out(f"tick {sim.tick}: living {living} converted {converted}")
    living, converted = _living(sim, spec)
    snap = capture_snapshot(sim)
    return MatchOutcome(result=rules.result(sim), seed=seed, living=living,
                        converted_alive=converted,
                        field_digest=field_digest(snap),
                        tick_digest=tick_digest(snap))


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="run_match", description="Run one headless match to its result.")
    ap.add_argument("--level", required=True,
                    help="level folder name (needs a match.toml)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the match config's seed")
    ap.add_argument("--ticks-per-report", type=int, default=0,
                    help="print the living slots every N ticks (0 = never)")
    ap.add_argument("--levels-dir", default="levels",
                    help="the levels root (default: the repo's levels/)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    level = level_loader.load(args.level, args.levels_dir)
    spec = level.match
    if spec is None:
        print(f"run_match: level {args.level!r} has no "
              f"{level_loader.MATCH_FILENAME}", file=sys.stderr)
        return 2
    seed = spec.seed if args.seed is None else args.seed
    print(f"match: level={args.level} seed={seed} teams={len(spec.teams)} "
          f"n={[t.n for t in spec.teams]} zombies={spec.zombies} "
          f"objective={spec.objective} time_limit_s={spec.time_limit_s}")
    o = run_match(level, spec, seed=seed,
                  report_every=max(0, args.ticks_per_report))
    r = o.result
    print(f"result: {r.outcome}" + (f" team {r.team}" if r.team is not None
                                    else ""))
    print(f"ticks: {r.ticks}")
    for team, slots in o.living.items():
        label = "zombies" if team == ZOMBIE_TEAM else f"team {team}"
        print(f"living {label}: {slots}")
    print(f"converted zombies alive: {o.converted_alive}")
    print(f"field_digest: {o.field_digest}")
    print(f"tick_digest: {o.tick_digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
