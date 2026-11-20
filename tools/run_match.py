"""tools/run_match.py -- THE headless match runner (#119 P2, rl-ready §7).

Runs one match to its result with no window: the level and its match.toml
(``level_loader.load`` -> ``LevelData.match``), stepped through
:class:`~simulation.match_session.MatchSession` (#29 P1a) -- THE steppable
match: a ``Simulation`` seeded with the match seed under
:class:`~simulation.ruleset.MatchRules`, the units spawned from the config
into the level's spawn rooms (``Simulation.spawn_match``), each team's
registered controller (``control_source.create_match_controllers``, the
horde's included) called once per tick with ``renderer=None``, then
``step()`` -- until terminal. :func:`run_match` is the session's caller (the
batched environment's farm workers are the other). No light is requested,
no recorder allocated, nothing imports the renderer. Tests and benches call
:func:`run_match`; never a second match loop.

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

Recording and replay (#126, ``src/match_replay.py``; replay by
RE-SIMULATION, never the field recorder)::

    ... tools/run_match.py --level ship_demo --record m.replay.gz
    ... tools/run_match.py --replay m.replay.gz

``--record PATH`` writes the match it runs: level, match spec, seed, every
intent per tick (the sim's one intent tap) and a digest checkpoint every
``[replay] checkpoint_every_ticks``. ``--replay PATH`` re-runs that match
headless with the ``replay`` controller in place of the recorded ones,
refusing a recording of another level / seed (exit 2), verifying every
checkpoint; it prints ``replay: every checkpoint matched (N)`` and exits 0,
or ``replay: DIVERGED -- ...`` naming the first tick that differs and exits
1. ``--level`` defaults to the recording's level with ``--replay``.
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
import match_replay  # noqa: E402
from simulation.match import outcome_lines  # noqa: E402
from simulation.match_session import MatchSession  # noqa: E402
from simulation.ruleset import MatchResult  # noqa: E402


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
    #: the match's recording (``record=True``, #126), else None.
    recording: "match_replay.Recording | None" = None
    #: the checkpoints a replay verified (``replay=...``, #126), else 0.
    checkpoints_verified: int = 0


def run_match(level, spec=None, *, seed: int | None = None,
              report_every: int = 0, out=None, on_tick=None,
              record: bool = False, checkpoint_every_ticks: int | None = None,
              replay=None) -> MatchOutcome:
    """Run ``spec`` (default ``level.match``) on ``level`` to its result.
    ``seed`` overrides the config's seed; ``report_every`` > 0 prints the
    living slots every that many ticks through ``out`` (default print).
    ``on_tick(sim)`` (#121), if given, is called after every ``step()`` --
    while that tick's ``tick_events`` are still on the sim -- so a bench or a
    test can watch the match without a second match loop; it must not write
    sim state.

    ``record=True`` (#126) records the match into
    ``MatchOutcome.recording``, a checkpoint every ``checkpoint_every_ticks``
    (default ``[replay] checkpoint_every_ticks``). ``replay`` (a
    ``match_replay.Recording``) re-runs a recorded match instead: its own
    spec and seed (a ``spec`` / ``seed`` passed must equal them, and
    ``level`` must be the recorded one, else ``ReplayRefused``), driven by the
    ``replay`` controller, every checkpoint verified (``ReplayDivergence`` on
    the first that differs, or when the match ends at another tick)."""
    out = print if out is None else out
    session = MatchSession(level, spec, seed=seed, replay=replay,
                           breach_physics=bp)
    if record:
        session.record(checkpoint_every_ticks)
    sim = session.sim
    while not session.is_terminal():
        session.tick()
        if on_tick is not None:
            on_tick(sim)
        if report_every and sim.tick % report_every == 0:
            living, converted = session.living()
            out(f"tick {sim.tick}: living {living} converted {converted}")
    result = session.result()
    recording = session.finish()
    living, converted = session.living()
    fd, td = session.digests()
    return MatchOutcome(result=result, seed=session.seed, living=living,
                        converted_alive=converted,
                        field_digest=fd, tick_digest=td,
                        recording=recording,
                        checkpoints_verified=session.checkpoints_verified)


def _parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="run_match", description="Run one headless match to its result.")
    ap.add_argument("--level", default=None,
                    help="level folder name (needs a match.toml); with "
                         "--replay, defaults to the recording's level")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the match config's seed")
    ap.add_argument("--ticks-per-report", type=int, default=0,
                    help="print the living slots every N ticks (0 = never)")
    ap.add_argument("--levels-dir", default="levels",
                    help="the levels root (default: the repo's levels/)")
    ap.add_argument("--record", default=None, metavar="PATH",
                    help="write a recording of the match to PATH (#126)")
    ap.add_argument("--replay", default=None, metavar="PATH",
                    help="re-run the match recorded at PATH, headless, "
                         "verifying every checkpoint (#126)")
    args = ap.parse_args(argv)
    if args.replay is None and args.level is None:
        ap.error("--level is required (unless --replay)")
    return args


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    replay = None
    if args.replay is not None:
        try:
            replay = match_replay.read_recording(args.replay)
        except match_replay.ReplayRefused as e:
            print(f"run_match: {e}", file=sys.stderr)
            return 2
    level_name = args.level if args.level is not None else replay.level
    level = level_loader.load(level_name, args.levels_dir)
    spec = level.match if replay is None else replay.spec
    if spec is None:
        print(f"run_match: level {level_name!r} has no "
              f"{level_loader.MATCH_FILENAME}", file=sys.stderr)
        return 2
    if args.seed is not None:
        seed = args.seed
    else:
        seed = replay.seed if replay is not None else spec.seed
    print(f"match: level={level_name} seed={seed} teams={len(spec.teams)} "
          f"n={[t.n for t in spec.teams]} zombies={spec.zombies} "
          f"objective={spec.objective} time_limit_s={spec.time_limit_s}"
          + (f" replay={args.replay}" if replay is not None else ""))
    try:
        o = run_match(level, spec, seed=seed,
                      report_every=max(0, args.ticks_per_report),
                      record=args.record is not None, replay=replay)
    except match_replay.ReplayRefused as e:
        print(f"run_match: {e}", file=sys.stderr)
        return 2
    except match_replay.ReplayDivergence as e:
        print(f"replay: DIVERGED -- {e}")
        return 1
    for line in outcome_lines(o.result, o.living, o.converted_alive,
                              o.field_digest, o.tick_digest):
        print(line)
    if o.recording is not None:
        n = match_replay.write_recording(o.recording, args.record)
        ticks = max(1, o.recording.end_tick)
        print(f"recorded: {args.record} ({n} bytes, {n / ticks:.1f} "
              f"bytes/tick, {o.recording.n_intents()} intents, "
              f"{len(o.recording.checkpoints)} checkpoints)")
    if replay is not None:
        print(f"replay: every checkpoint matched ({o.checkpoints_verified})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
