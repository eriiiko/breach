"""``MatchObserver`` — watch a level's match in the game window (#121), and
``ReplayObserver`` — watch a RECORDED match by re-simulating it (#127).

``main.py --level <level> --control match`` (a ``_parse_control_flag`` value)
plays the level's ``match.toml`` with the normal renderer: the match's
ruleset (:class:`~simulation.ruleset.MatchRules`), its seed, its units
spawned into the level's spawn rooms (``Simulation.spawn_match``), and every
team (and the horde) driven by its REGISTERED controller
(``control_source.create_match_controllers``),
called once per SIM TICK (:meth:`on_tick`, right before each ``sim.step()``)
exactly as the headless runner (``tools/run_match.py``) calls it, so the
window shows the same match the runner plays. The human only observes: no
possession, no selection, no orders. When the match ends the sim pauses and
the result is printed.

``main.py --replay PATH`` (#127, ``_parse_replay_flag``) is THE playback
path: :class:`ReplayObserver` is a MatchObserver whose spec and seed come
from the recording's header and whose one controller is ``replay``
(``create_match_controllers(sim, spec, recording=)``), so the window draws a
LIVE re-simulated match (light, smoke, fire VFX all work because it is a
real sim). Every checkpoint is verified as the sim passes it; a divergence
pauses the window on ``DIVERGED at tick N`` and never crashes it. Seeking is
re-simulation too (checkpoints hold digests, not state): forward
fast-forwards without drawing, backward rebuilds the sim from tick 0
(:func:`seek_plan`). The pure parts here are pyray-free; only
:meth:`ReplayObserver.handle_frame` polls keys (pyray imported lazily).
"""
from __future__ import annotations

from control_source import ControlSource, create_match_controllers
from simulation.orders import ORDER_MOVE_ATTACK
from simulation.ruleset import MatchRules


class MatchObserver(ControlSource):
    """The match's controllers drive; the window watches.

    ``spec`` (#127) overrides the level's match.toml (the replay observer
    passes the recording's)."""

    def __init__(self, level, spec=None):
        spec = getattr(level, "match", None) if spec is None else spec
        if spec is None:
            raise SystemExit(f"--control match needs a level with a "
                             f"match.toml (level {level.name!r} has none)")
        self.spec = spec
        self.rules = MatchRules.from_spec(spec)
        # Built at spawn: they need the sim (the horde is handed over there).
        self.controllers = []
        # UI-state attributes main.py reads off every ControlSource.
        self.selected_unit_id = None
        self.planning_phase = 0
        self.current_mode = ORDER_MOVE_ATTACK
        self._reported = False
        #: the sim this observer watches (set at spawn) -- the follow
        #: camera (#128) reads its units through main.py.
        self.sim = None
        self.match_result = None

    # Sim-construction hooks.
    def initial_ruleset(self):
        return self.rules

    def starts_paused(self) -> bool:
        return False

    def initial_seed(self, default: int) -> int:
        return int(self.spec.seed)

    def spawn_units(self, sim, level) -> bool:
        self.sim = sim
        sim.spawn_match(self.spec)
        self.controllers = self._build_controllers(sim)
        print(f"  match: seed {self.initial_seed(0)}, teams "
              f"{[(t.n, t.controller) for t in self.spec.teams]}, zombies "
              f"{self.spec.zombies}, limit {self.spec.time_limit_s:g} s")
        return True

    def _build_controllers(self, sim) -> list:
        return create_match_controllers(sim, self.spec)

    # Per tick / per frame.
    def on_tick(self, sim) -> None:
        """Before each ``sim.step()``: the runner's loop body. A terminal
        match pauses the sim instead (the step that follows is then a
        no-op), and reports its result once."""
        if self.rules.is_terminal(sim):
            sim.set_paused(True)
            if not self._reported:
                self._reported = True
                self._on_end(sim)
            return
        self._drive(sim)

    def _drive(self, sim) -> None:
        for c in self.controllers:
            c.handle_frame(sim, None)

    def _on_end(self, sim) -> None:
        r = self.rules.result(sim)
        self.match_result = r
        print(f"  match over: {r.outcome}"
              + (f" team {r.team}" if r.team is not None else "")
              + f" after {r.ticks} ticks")

    def handle_frame(self, sim, renderer) -> None:
        return None        # the human observes; nothing is polled

    def hud_lines(self, sim, cam=None) -> list:
        """The watched match's HUD (#128), ``(text, kind)`` rows as
        :meth:`ReplayObserver.hud_lines`: the seed, the tick, the result
        once over, and ``cam`` (the follow camera's ``cam: ...`` line)."""
        rows = [(f"MATCH  seed {self.initial_seed(0)}", "title"),
                (f"tick {int(sim.tick)}", "info")]
        r = self.match_result
        if r is not None:
            rows.append((f"match over: {r.outcome}"
                         + (f" team {r.team}" if r.team is not None else "")
                         + f" after {r.ticks} ticks", "ok"))
        if cam is not None:
            rows.append((cam, "info"))
            rows.append(("C cycle camera", "info"))
        return rows


# ---------------------------------------------------------------------------
# Replay (#127)
# ---------------------------------------------------------------------------
def replay_level_name(recording, level_flag=None) -> str:
    """The level a replay loads: the recording's level FOLDER (its header).
    ``level_flag`` (main.py's ``--level``) is not needed; given, it must name
    the same folder, else ``ReplayRefused``."""
    from match_replay import ReplayRefused
    if level_flag is not None and level_flag != recording.level:
        raise ReplayRefused(f"--level {level_flag!r} disagrees with the "
                            f"recording, which is of level "
                            f"{recording.level!r} (drop --level)")
    return recording.level


def seek_plan(current_tick: int, target_tick: int, end_tick: int) -> tuple:
    """How to reach ``target_tick`` from ``current_tick`` by re-simulation:
    ``(restart, target, n_steps)``. ``target`` is clamped to
    ``[0, end_tick]``; ``restart`` is True when it lies BEHIND the sim (the
    sim is rebuilt at tick 0 -- checkpoints hold digests, not state), and
    ``n_steps`` is how many ticks are then stepped without drawing."""
    target = max(0, min(int(target_tick), int(end_tick)))
    if target < current_tick:
        return True, target, target
    return False, target, target - int(current_tick)


class ReplayObserver(MatchObserver):
    """Plays a :class:`match_replay.Recording` in the window by
    re-simulation. ``make_sim(seed, ruleset) -> Simulation`` builds a fresh
    sim on a freshly loaded copy of the recorded level (main.py's one
    constructor); it is what a backward seek rebuilds from. ``seek_tick`` is
    where Backspace rewinds to (main.py's ``--seek``, default 0).

    Keys (existing ones, no new binding): Space pauses / resumes,
    Backspace restarts at ``seek_tick`` (re-simulated from tick 0). Speed is
    ``[display] sim_speed`` (main.py's wall-clock pacing)."""

    def __init__(self, level, recording, *, make_sim=None, seek_tick=0):
        import match_replay
        match_replay.check_header(recording, match_replay.level_id(level))
        super().__init__(level, spec=recording.spec)
        self.recording = recording
        self.make_sim = make_sim
        self.seek_tick = int(seek_tick)
        self.sim = None
        #: ``on_seek_tick(sim, target)``, called after every tick a seek
        #: steps (main.py draws its "seeking" overlay from it, #128).
        self.on_seek_tick = None
        #: bumped on every rebuild / seek: main.py re-requests light then.
        self.generation = 0
        self._reset_state()

    def _reset_state(self) -> None:
        self._reported = False
        #: the last tick whose recorded checkpoint this run verified.
        self.last_verified = None
        #: the ReplayDivergence that stopped the replay, else None.
        self.diverged = None
        #: ``(field_digest, tick_digest)`` at the end, once reported.
        self.final_digests = None
        self.result = None

    # Hooks.
    def initial_seed(self, default: int) -> int:
        return int(self.recording.seed)

    def spawn_units(self, sim, level) -> bool:
        self._reset_state()
        self.sim = sim
        return super().spawn_units(sim, level)

    def _build_controllers(self, sim) -> list:
        return create_match_controllers(sim, self.spec,
                                        recording=self.recording)

    @property
    def end_tick(self) -> int:
        return int(self.recording.end_tick)

    @property
    def ended(self) -> bool:
        return self._reported

    def _drive(self, sim) -> None:
        if self.diverged is not None:
            sim.set_paused(True)
            return
        from match_replay import ReplayDivergence
        try:
            self.controllers[0].handle_frame(sim, None)
        except ReplayDivergence as e:
            self._diverge(sim, e)
            return
        if int(sim.tick) in self.recording.checkpoints:
            self.last_verified = int(sim.tick)

    def _diverge(self, sim, e) -> None:
        self.diverged = e
        sim.set_paused(True)
        print(f"replay: DIVERGED -- {e}")

    def on_tick(self, sim) -> None:
        if self.diverged is not None:
            sim.set_paused(True)
            return
        super().on_tick(sim)

    def _on_end(self, sim) -> None:
        """The match is terminal: it must end where the recording did, on
        its last checkpoint; then report as ``run_match`` does."""
        import match_replay
        from match_replay import ReplayDivergence
        from simulation.match import living_slots, outcome_lines
        super()._on_end(sim)
        try:
            self.controllers[0].finish(sim)
        except ReplayDivergence as e:
            self._diverge(sim, e)
            return
        self.last_verified = int(sim.tick)
        self.result = self.rules.result(sim)
        self.final_digests = tuple(match_replay.match_digests(sim))
        living, converted = living_slots(sim, self.spec)
        for line in outcome_lines(self.result, living, converted,
                                  *self.final_digests):
            print(line)
        print(f"replay: every checkpoint matched "
              f"({self.controllers[0].checkpoints_verified})")

    # Re-simulation.
    def restart(self):
        """Rebuild the match at tick 0 on a fresh sim (the same hooks main.py
        builds the first one with). Returns the new sim."""
        if self.make_sim is None:
            raise RuntimeError("ReplayObserver.restart needs make_sim")
        # A finished or diverged replay restarts playing; otherwise the
        # pause state carries over.
        paused = (self.sim is not None and self.sim.is_paused()
                  and not self.ended and self.diverged is None)
        self.rules = MatchRules.from_spec(self.spec)
        sim = self.make_sim(self.initial_seed(0), self.initial_ruleset())
        self.spawn_units(sim, None)
        sim.set_paused(paused)
        self.generation += 1
        return sim

    def seek(self, target_tick: int, progress=print):
        """Reach ``target_tick`` (clamped to the recording) WITHOUT drawing:
        backward rebuilds at tick 0 first (:func:`seek_plan`), then the
        ticks are stepped through :meth:`on_tick` exactly as the window
        steps them, checkpoints verified on the way. Keeps the pause state;
        light is dropped for the fast-forward and re-requested by main.py
        (``generation``). Returns the sim (a new one after a restart)."""
        restart, target, n = seek_plan(self.sim.tick, target_tick,
                                       self.end_tick)
        sim = self.restart() if restart else self.sim
        paused = sim.is_paused()
        sim.set_light(False)
        if n and progress is not None:
            progress(f"  replay: seeking to tick {target} ({n} ticks, "
                     f"not drawn)")
        while sim.tick < target and not self.ended \
                and self.diverged is None:
            sim.set_paused(False)
            self.on_tick(sim)
            if sim.is_paused():
                break
            sim.step()
            if self.on_seek_tick is not None:
                self.on_seek_tick(sim, target)
            if progress is not None and sim.tick % 240 == 0:
                progress(f"  replay: ... tick {sim.tick}")
        if self.rules.is_terminal(sim):
            self.on_tick(sim)          # report the end now
        if not self.ended and self.diverged is None:
            sim.set_paused(paused)
        self.generation += 1
        return sim

    def toggle_pause(self, sim) -> None:
        """Space: pause / resume -- a no-op once the replay ended or
        diverged (the window stays on its last picture)."""
        if self.ended or self.diverged is not None:
            sim.set_paused(True)
            return
        sim.set_paused(not sim.is_paused())

    def handle_frame(self, sim, renderer) -> None:
        import pyray as rl
        K = rl.KeyboardKey
        if rl.is_key_pressed(K.KEY_SPACE):
            self.toggle_pause(sim)
        if rl.is_key_pressed(K.KEY_BACKSPACE):
            self.restart()
            self.seek(self.seek_tick)

    def hud_lines(self, sim, cam=None) -> list:
        """The replay HUD as ``(text, kind)`` rows, ``kind`` one of
        ``"title"``, ``"info"``, ``"ok"``, ``"bad"`` (main.py colours them).
        ``cam`` is the follow camera's ``cam: ...`` line (#128), if any."""
        rec = self.recording
        rows = [(f"REPLAY  {rec.level}  seed {rec.seed}", "title"),
                (f"tick {int(sim.tick)} / {self.end_tick}"
                 + ("  PAUSED" if sim.is_paused() and not self.ended
                    and self.diverged is None else ""), "info")]
        if self.diverged is not None:
            rows.append((f"DIVERGED at tick {self.diverged.tick}", "bad"))
        elif self.last_verified is None:
            rows.append(("no checkpoint verified yet", "info"))
        else:
            rows.append((f"checkpoint {self.last_verified} verified", "ok"))
        if self.result is not None:
            r = self.result
            rows.append((f"match over: {r.outcome}"
                         + (f" team {r.team}" if r.team is not None else "")
                         + f" after {r.ticks} ticks", "ok"))
        if cam is not None:
            rows.append((cam, "info"))
        rows.append(("Space pause  Backspace restart"
                     + (f" at tick {self.seek_tick}" if self.seek_tick
                        else "")
                     + ("  C camera" if cam is not None else ""), "info"))
        return rows


__all__ = ["MatchObserver", "ReplayObserver", "replay_level_name",
           "seek_plan"]
