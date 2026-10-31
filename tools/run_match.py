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
from config import CFG  # noqa: E402
from control_source import create_match_controllers  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.match import living_slots, outcome_lines  # noqa: E402
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
    if replay is not None:
        match_replay.check_header(replay, match_replay.level_id(level),
                                  seed=seed, spec=spec)
        spec, seed = replay.spec, replay.seed
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
    controllers = create_match_controllers(sim, spec, recording=replay)
    recorder = None
    if record:
        every = (match_replay.replay_settings(CFG)
                 if checkpoint_every_ticks is None else checkpoint_every_ticks)
        recorder = match_replay.MatchRecorder(
            match_replay.level_id(level), spec, seed,
            checkpoint_every_ticks=every,
            build=match_replay.build_id())
        recorder.start(sim)
    while not rules.is_terminal(sim):
        for c in controllers:
            c.handle_frame(sim, None)
        sim.step()
        if recorder is not None:
            recorder.after_step(sim)
        if on_tick is not None:
            on_tick(sim)
        if report_every and sim.tick % report_every == 0:
            living, converted = living_slots(sim, spec)
            out(f"tick {sim.tick}: living {living} converted {converted}")
    result = rules.result(sim)
    recording = recorder.finish(sim, result) if recorder is not None else None
    verified = 0
    if replay is not None:
        controllers[0].finish(sim)
        verified = controllers[0].checkpoints_verified
    living, converted = living_slots(sim, spec)
    fd, td = match_replay.match_digests(sim)
    return MatchOutcome(result=result, seed=seed, living=living,
                        converted_alive=converted,
                        field_digest=fd, tick_digest=td,
                        recording=recording, checkpoints_verified=verified)


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
