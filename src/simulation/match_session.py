"""THE steppable match (#29 P1a, residency design §4.1 / §13 b.2).

A :class:`MatchSession` is one match as ``tools/run_match.py`` runs it,
factored out so a caller can step it tick by tick: the level and its parsed
match spec, a ``Simulation`` seeded with the match seed under
:class:`~simulation.ruleset.MatchRules`, the units spawned from the spec into
the level's spawn rooms (``Simulation.spawn_match``, the one match spawn),
every team's registered controller and the horde's
(``control_source.create_match_controllers``, the one builder), and the
recording hook (``match_replay.MatchRecorder`` through the sim's one intent
tap, ``Simulation.attach_intent_log``).

:meth:`MatchSession.tick` is THE per-tick body of every headless match loop:
each controller's ``handle_frame(sim, None)`` in build order, then
``sim.step()``, then the recorder's checkpoint. ``run_match`` drives it to
the result; the batched environment's farm workers (``src/rl_env.py``) drive
it with the ``policy`` controller's action rows. Never a second loop body.

Headless: no window, no light requested, no field recorder, nothing imports
the renderer, and the physics runner is ``headless`` (#29 P4a: the
render-only ripple is skipped). The digests (``match_digests``) come from the determinism
gates' snapshot in ``tests/`` -- the caller puts that directory on
``sys.path`` (``tools/run_match.py`` and the farm workers do).
"""
from __future__ import annotations

import match_replay
from control_source import create_match_controllers
from simulation.match import living_slots
from simulation.ruleset import MatchRules
from simulation.simulation import Simulation


class MatchSession:
    """One match, steppable: build it, :meth:`tick` until
    :meth:`is_terminal`, then :meth:`result` / :meth:`finish`.

    ``spec`` defaults to ``level.match``; ``seed`` overrides the spec's.
    ``replay`` (a ``match_replay.Recording``) re-runs a recorded match: its
    own spec and seed (a ``spec`` / ``seed`` passed must equal them and
    ``level`` must be the recorded one, else ``ReplayRefused``), driven by
    the ``replay`` controller, each checkpoint verified as the sim passes it.
    ``breach_physics`` is the compiled CPU module (imported from the path
    when not given)."""

    def __init__(self, level, spec=None, *, seed: int | None = None,
                 replay=None, breach_physics=None):
        if replay is not None:
            match_replay.check_header(replay, match_replay.level_id(level),
                                      seed=seed, spec=spec)
            spec, seed = replay.spec, replay.seed
        spec = level.match if spec is None else spec
        if spec is None:
            raise ValueError(f"level {level.name!r} has no match.toml")
        seed = spec.seed if seed is None else int(seed)
        if breach_physics is None:
            import breach_physics
        self.level = level
        self.spec = spec
        self.seed = seed
        self.rules = MatchRules.from_spec(spec)
        self.sim = Simulation(level, seed=seed, breach_physics=breach_physics,
                              enable_recorder=False, ruleset=self.rules)
        # #29 P4a: a session is headless (no window ever draws it), so the
        # engine skips the render-only W6a ripple and the resident tick
        # downloads no render-only plane (PhysicsRunner.headless). Outside the
        # digest by construction: ripple / ripple_v are float, excluded.
        if self.sim.physics_runner is not None:
            self.sim.physics_runner.headless = True
        self.sim.set_paused(False)
        self.sim.spawn_match(spec)
        self.controllers = create_match_controllers(self.sim, spec,
                                                    recording=replay)
        self._replay = replay
        self._recorder = None
        self._finished = False

    # -- recording (#126) -------------------------------------------------
    def record(self, checkpoint_every_ticks: int | None = None) -> None:
        """Record this match from tick 0: attach the intent tap and take the
        tick-0 checkpoint; every :meth:`tick` then checkpoints every
        ``checkpoint_every_ticks`` (default ``[replay]
        checkpoint_every_ticks``), and :meth:`finish` returns the
        ``Recording``."""
        if self._recorder is not None:
            raise ValueError("this match is already being recorded")
        if checkpoint_every_ticks is None:
            from config import CFG
            checkpoint_every_ticks = match_replay.replay_settings(CFG)
        self._recorder = match_replay.MatchRecorder(
            match_replay.level_id(self.level), self.spec, self.seed,
            checkpoint_every_ticks=checkpoint_every_ticks,
            build=match_replay.build_id())
        self._recorder.start(self.sim)

    # -- the loop body ----------------------------------------------------
    def tick(self) -> None:
        """One tick: every controller's ``handle_frame(sim, None)`` (team
        rows in team order, then the horde's -- the order
        ``create_match_controllers`` built them), ``sim.step()``, then the
        recording's checkpoint when one is due."""
        sim = self.sim
        for c in self.controllers:
            c.handle_frame(sim, None)
        sim.step()
        if self._recorder is not None:
            self._recorder.after_step(sim)

    def is_terminal(self) -> bool:
        """The ``MatchRules`` law: fewer than two marine teams alive, or the
        time limit."""
        return self.rules.is_terminal(self.sim)

    def result(self):
        """The ``MatchResult`` once terminal, else None."""
        return self.rules.result(self.sim)

    def finish(self):
        """Close a terminal match once: the recording's last checkpoint (the
        ``Recording`` is returned, else None) and, under ``replay``, the
        check that the match ended where the recording did
        (``ReplayDivergence`` otherwise)."""
        if self._finished:
            raise ValueError("finish() was already called on this match")
        self._finished = True
        recording = None
        if self._recorder is not None:
            recording = self._recorder.finish(self.sim, self.result())
        if self._replay is not None:
            self.controllers[0].finish(self.sim)
        return recording

    # -- reading ----------------------------------------------------------
    @property
    def tick_count(self) -> int:
        return int(self.sim.tick)

    @property
    def checkpoints_verified(self) -> int:
        """The checkpoints a ``replay`` verified so far (0 otherwise)."""
        if self._replay is None:
            return 0
        return self.controllers[0].checkpoints_verified

    def living(self) -> tuple:
        """``(living, converted)``: ``simulation.match.living_slots``."""
        return living_slots(self.sim, self.spec)

    def digests(self) -> tuple:
        """``(field_digest, tick_digest)`` of the match as it stands."""
        return match_replay.match_digests(self.sim)


__all__ = ["MatchSession"]
