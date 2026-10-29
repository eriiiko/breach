"""``MatchObserver`` — watch a level's match in the game window (#121).

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
"""
from __future__ import annotations

from control_source import ControlSource, create_match_controllers
from simulation.orders import ORDER_MOVE_ATTACK
from simulation.ruleset import MatchRules


class MatchObserver(ControlSource):
    """The match's controllers drive; the window watches."""

    def __init__(self, level):
        spec = getattr(level, "match", None)
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

    # Sim-construction hooks.
    def initial_ruleset(self):
        return self.rules

    def starts_paused(self) -> bool:
        return False

    def initial_seed(self, default: int) -> int:
        return int(self.spec.seed)

    def spawn_units(self, sim, level) -> bool:
        sim.spawn_match(self.spec)
        self.controllers = create_match_controllers(sim, self.spec)
        print(f"  match: seed {self.spec.seed}, teams "
              f"{[(t.n, t.controller) for t in self.spec.teams]}, zombies "
              f"{self.spec.zombies}, limit {self.spec.time_limit_s:g} s")
        return True

    # Per tick / per frame.
    def on_tick(self, sim) -> None:
        """Before each ``sim.step()``: the runner's loop body. A terminal
        match pauses the sim instead (the step that follows is then a
        no-op), and prints its result once."""
        if self.rules.is_terminal(sim):
            sim.set_paused(True)
            if not self._reported:
                r = self.rules.result(sim)
                print(f"  match over: {r.outcome}"
                      + (f" team {r.team}" if r.team is not None else "")
                      + f" after {r.ticks} ticks")
                self._reported = True
            return
        for c in self.controllers:
            c.handle_frame(sim, None)

    def handle_frame(self, sim, renderer) -> None:
        return None        # the human observes; nothing is polled


__all__ = ["MatchObserver"]
