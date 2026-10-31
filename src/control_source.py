"""The ``ControlSource`` seam — how intents are produced (above the facade).

Design: ``docs/control_modularity_design_2026-07-22.md`` §3b. A loadable
game is ``Ruleset`` (sim-side turn structure + cost policy, see
``simulation/ruleset.py``) + ``ControlSource`` (this module — the
control-side half) + ``AgentPolicy`` (the ML track) + config/content.
``main.py`` owns exactly one ``ControlSource`` instance, chosen at startup
from the ``--control`` launch flag (default ``wego``), and calls
``handle_frame(sim, renderer)`` on it once per frame. The renderer and
``Simulation`` never know which control scheme is driving them — they only
read a handful of UI-state attributes back off the source for the HUD
(``selected_unit_id``, ``planning_phase``, ``current_mode`` — see
``WEGOPlanningInput`` in ``input_handler.py``).

P2 (2026-07-22, pure seam + rename, NO behavior change): only
``WEGOPlanningInput`` exists — the renamed/wrapped ``InputHandler``, byte-
identical behavior, selected by the default ``--control wego``.
``GamepadDirect`` (one possessed unit, per-tick MOVE_DIR/AIM/TRIGGER
intents) and ``AgentPolicy`` (the RL path) are P3 — not built here.
"""
from __future__ import annotations

import math

from simulation.unit_fixed import quantize_scalar


def quantize_stick_direction(ax: float, ay: float, deadzone: float = 0.15):
    """Pure float-axis -> Q16.16 intent quantization (control-modularity P3,
    the control/facade seam, §3c).

    Turns one analog stick's raw axes (``ax``/``ay``, each nominally in
    ``[-1, 1]``) into a Q16.16 **unit-vector** direction plus a clamped Q16.16
    magnitude: ``(dx_q, dy_q, mag_q)``. This is THE seam where a control-layer
    float becomes a fixed-point sim intent — the sim never sees ``ax``/``ay``.

    Determinism: pure and libm-free on the sim-critical path. ``math.sqrt`` is
    IEEE-754 *correctly rounded* (a basic operation, not a libm transcendental
    like ``sin``/``atan2``), the reciprocal and products are IEEE, and
    :func:`~simulation.unit_fixed.quantize_scalar` is the documented
    round-half-away-from-zero twin — so the SAME ``(ax, ay)`` yields the SAME
    ``(dx_q, dy_q, mag_q)`` on every machine, compiler, and Python version.
    (The one thing this function cannot make deterministic is the raw axis
    value a physical pad/driver reports — see the P3 escalation note (c): that
    is why the *quantized intent*, not the raw axis, is what a networked
    lockstep would sync/replay.)

    A magnitude at or below ``deadzone`` returns ``(0, 0, 0)`` — no direction,
    the stick is centered. Magnitude is clamped to 1.0 (diagonal corners on
    some pads read > 1). Unit-testable with synthetic floats and no hardware.
    """
    mag = math.sqrt(ax * ax + ay * ay)
    if mag <= deadzone:
        return (0, 0, 0)
    inv = 1.0 / mag
    ux = ax * inv
    uy = ay * inv
    mag_c = 1.0 if mag > 1.0 else mag
    return (quantize_scalar(ux), quantize_scalar(uy), quantize_scalar(mag_c))


def analog_speed_q(ax: float, ay: float, deadzone: float = 0.15) -> int:
    """Pure stick -> ANALOG SPEED quantization (#106 P6f): the stick's
    deflection past its deadzone, rescaled to the full range -- ``(min(1,
    |a|) - deadzone) / (1 - deadzone)`` -- as Q16.16 in ``[0, ONE]``
    (``MoveDirIntent.speed_q``: the fraction of the unit's maximum speed).
    0 inside the deadzone, ONE at full deflection (a corner reading > 1 is
    clamped). Deterministic for the same reason as
    :func:`quantize_stick_direction` (IEEE sqrt / divide, the round-half-away
    quantizer); testable with synthetic floats and no hardware."""
    mag = math.sqrt(ax * ax + ay * ay)
    if mag <= deadzone:
        return 0
    mag = 1.0 if mag > 1.0 else mag
    span = 1.0 - deadzone
    frac = 1.0 if span <= 0.0 else (mag - deadzone) / span
    return max(0, min(1 << 16, quantize_scalar(frac)))


def first_controlled_unit(sim, current_id=None):
    """The unit a one-unit direct scheme drives (GamepadDirect,
    KeyboardDirect): ``current_id`` while it lives, else the first living
    unit of ``sim.player_team`` (unit order), else None. Under ``--play
    creature`` the player team is ``CREATURE_TEAM``, so this is the
    creature (one octopus in the tank)."""
    cur = sim.get_unit(current_id) if current_id is not None else None
    if cur is not None and cur.alive:
        return cur.id
    for u in sim.units:
        if u.team == sim.player_team and u.alive:
            return u.id
    return None


#: ``--play NAME`` -> the team the human's direct scheme drives (#106 P6f).
#: Absent = the squad (``player_team`` 0, unchanged).
_PLAYABLE = ("creature",)


def play_team(name: str) -> int:
    """The team ``--play NAME`` hands the human (``creature`` ->
    ``factions.CREATURE_TEAM``); an unknown name is a launch error."""
    from simulation.factions import CREATURE_TEAM
    if name == "creature":
        return CREATURE_TEAM
    raise SystemExit(f"--play must be one of {_PLAYABLE!r}, got {name!r}")


def select_play(sim, source, name) -> None:
    """Apply ``--play NAME`` to a built sim: the session's controlled team
    (``sim.player_team``) becomes the named one, so the direct scheme's
    selected unit is that team's first living unit (the creature). Only a
    DIRECT scheme (``source.drives_one_unit``: gamepad, keyboard) can play
    a creature -- a WEGO plan never reaches a creature's hub -- so any
    other scheme is refused. ``None`` changes nothing. The squad is then
    driven by nothing: it stands."""
    if name is None:
        return
    team = play_team(name)
    if not getattr(source, "drives_one_unit", False):
        raise SystemExit(f"--play {name} needs a direct control scheme "
                         f"(--control gamepad or --control keyboard)")
    sim.player_team = team


class ControlSource:
    """Strategy interface ``main.py`` selects at startup via ``--control``.

    Every implementation turns per-frame polling (keyboard/mouse today;
    gamepad sticks/buttons in P3; for the ML track, an observed
    ``sim.get_state()``) into calls against the ``Simulation`` facade
    (``sim.apply_action`` and friends). Side effects land on ``sim`` or on
    this object's own UI-only state — a ``ControlSource`` never mutates the
    sim directly, and the sim never imports this module (the dependency
    points one way, control -> facade, matching the design's "above the
    facade" placement).
    """

    def handle_frame(self, sim, renderer) -> None:
        """Run one frame's worth of input for this control scheme.

        ``sim`` is the :class:`simulation.Simulation` facade; ``renderer``
        is the :class:`renderer.GameRenderer` (used for screen<->tile
        conversion and any renderer-owned bindings this source still
        needs, e.g. ``mouse_to_tile()``) — or ``None`` in a headless match
        (#119 P2: ``tools/run_match.py`` calls each team's controller once
        per tick with no renderer). A controller registered for matches
        (:func:`register_controller`) must work with ``renderer=None``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Sim-construction hooks (P3, §3a/§3b) — a control scheme also picks the
    # turn structure it runs under. main.py queries these BEFORE building the
    # Simulation so the two halves of the loadable game (Ruleset + ControlSource)
    # are chosen together. The base returns the WEGO defaults, so
    # WEGOPlanningInput is byte-identical without overriding anything.
    # ------------------------------------------------------------------
    def initial_ruleset(self):
        """The :class:`~simulation.ruleset.Ruleset` this control scheme runs
        under, or ``None`` for the shipped default
        (:class:`~simulation.ruleset.TwoPhaseWEGO`). GamepadDirect returns a
        :class:`~simulation.ruleset.ContinuousRealtime`."""
        return None

    def starts_paused(self) -> bool:
        """Whether the sim should begin paused. WEGO plans first (``True``);
        direct control runs immediately (``False``)."""
        return True

    def initial_seed(self, default: int) -> int:
        """The sim seed main.py builds with (#121): ``default`` for every
        human scheme; ``MatchObserver`` returns the match's seed."""
        return default

    def spawn_units(self, sim, level) -> bool:
        """Spawn this scheme's units into the fresh ``sim`` and return True,
        or return False to let main.py spawn the level's ``[[spawn]]``
        entries (every human scheme). ``MatchObserver`` spawns the match."""
        return False

    def on_tick(self, sim) -> None:
        """Called by main.py right before EVERY ``sim.step()`` (#121) — the
        per-tick hook a scheme that drives units by the tick (the match
        observer's controllers) uses, so the window plays the same ticks as
        the headless runner. A no-op for every per-frame human scheme."""
        return None

    #: True for a scheme that drives ONE unit by per-tick intents (the
    #: gamepad, the keyboard): the only schemes ``--play`` accepts (#106 P6f).
    drives_one_unit = False

    #: May the renderer pan the camera with WASD? False for a scheme that
    #: steers a unit with them (KeyboardDirect); the arrows always pan.
    pans_camera_with_wasd = True

    @property
    def wants_renderer_toggles(self) -> bool:
        """May the RENDERER keep its own diagnostic key bindings (F1-F10, T,
        V, M, …)?

        ``True`` for every shipped scheme — nothing about their keymaps
        changes. ``OnePhaseWEGOInput`` returns ``self.debug``, because §17
        rules that in game mode the control scheme owns every binding, so the
        renderer's diagnostic keys are evicted with the rest unless ``--debug``
        arms them.
        """
        return True


class IdleControl(ControlSource):
    """A match controller that emits NOTHING (#119 P2): its team stands where
    it spawned. The first registered match controller; the bots (#121) and
    the horde (#122, ``zombies``) register beside it. As a match's
    ``zombie_controller`` it leaves the horde standing."""

    def __init__(self, team: int | None = None):
        self.team = team

    def handle_frame(self, sim, renderer=None) -> None:
        return None


# ---------------------------------------------------------------------------
# The MATCH controller registry (#119 P2) — name -> factory(team) building the
# ControlSource that drives one team of a headless match. match.toml's
# `controller` names a row here (validated at the level door); the headless
# runner builds one per team and calls handle_frame(sim, None) each tick. A
# new controller is a registry entry, never a second control path. Separate
# from `_KNOWN_CONTROLS` below, which names the HUMAN schemes main.py's
# --control flag picks between.
# ---------------------------------------------------------------------------
def _bots(team: int) -> "ControlSource":
    """The rule-based bot team (#121, ``control_bots.BotTeam``, settings
    ``config.toml [bots]``). Deferred import: control_bots pulls in the sim's
    tables and the pathfinder, which importing this module must not."""
    from control_bots import BotTeam
    return BotTeam(team)


def _zombies(team: int) -> "ControlSource":
    """The horde (#122, ``control_zombies.ZombieControl``): the shipped
    zombie law, run through the ZOMBIE intent for every living zombie."""
    from control_zombies import ZombieControl
    return ZombieControl(team)


def _creatures(team: int) -> "ControlSource":
    """The creatures (#106 P6d2, ``control_creature.CreatureControl``): every
    living creature unit of the team crawls at the nearest unit it sees, or
    idles. Deferred import, like the others."""
    from control_creature import CreatureControl
    return CreatureControl(team)


_MATCH_CONTROLLERS: dict = {
    "idle": IdleControl,
    "bots": _bots,
    "zombies": _zombies,
    "creature": _creatures,
}


def register_controller(name: str, factory) -> None:
    """Register ``factory`` (``team -> ControlSource``) under ``name``;
    refuses a name already taken (never a silent override)."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"controller name must be a non-empty str "
                         f"(got {name!r})")
    if name in _MATCH_CONTROLLERS:
        raise ValueError(f"controller {name!r} is already registered")
    _MATCH_CONTROLLERS[name] = factory


def registered_controllers() -> tuple:
    """The registered match-controller names, sorted."""
    return tuple(sorted(_MATCH_CONTROLLERS))


def create_controller(name: str, team: int) -> "ControlSource":
    """Build the registered controller ``name`` for ``team``."""
    try:
        factory = _MATCH_CONTROLLERS[name]
    except KeyError:
        raise ValueError(f"controller {name!r} is not registered (known: "
                         f"{list(registered_controllers())})") from None
    return factory(team)


def create_match_controllers(sim, spec) -> list:
    """THE one way a match's controllers are built (#122): one per
    ``[[team]]`` row (team i -> its ``controller``), then the horde's
    (``spec.zombie_controller`` for ``ZOMBIE_TEAM``) -- after handing the
    horde to it (``sim.hand_zombies_to_controller``), so the conductor no
    longer drives the zombies itself: every unit of a match has exactly one
    driver, and it is a ControlSource. Then, when the match has creatures
    (#106 P6d2), one controller per creature team -- ``spec.
    creature_controller`` (default ``creature``; ``idle`` leaves them to
    their own idle), in team order. Call once per match, after
    ``sim.spawn_match``; then ``handle_frame(sim, None)`` on each, every
    tick, before ``sim.step()``."""
    from simulation.factions import ZOMBIE_TEAM
    from simulation.match import (DEFAULT_CREATURE_CONTROLLER,
                                  DEFAULT_ZOMBIE_CONTROLLER)
    controllers = [create_controller(row.controller, team)
                   for team, row in enumerate(spec.teams)]
    sim.hand_zombies_to_controller()
    controllers.append(create_controller(
        getattr(spec, "zombie_controller", DEFAULT_ZOMBIE_CONTROLLER),
        ZOMBIE_TEAM))
    creature_teams = sorted({int(c.team) for c in getattr(spec, "creatures", ())})
    cname = getattr(spec, "creature_controller", DEFAULT_CREATURE_CONTROLLER)
    controllers += [create_controller(cname, team) for team in creature_teams]
    return controllers


# Launch-flag name -> factory. P3 adds "gamepad" (GamepadDirect); the ML entry
# point (AgentPolicy) is driven differently — it has no per-frame poll loop —
# so it is not expected to route through this same factory; see the design doc
# §3b.
_KNOWN_CONTROLS = ("wego", "gamepad", "keyboard", "onephase", "match")


def create_control_source(name: str, debug: bool = False,
                          level=None) -> "ControlSource":
    """Factory for the ``--control`` launch flag (§3b).

    ``name`` must be one of :data:`_KNOWN_CONTROLS`: ``"wego"`` (today's
    default WEGO planning input) or ``"gamepad"`` (P3 direct control — one
    possessed marine, per-tick intents, ContinuousRealtime). An unknown name
    is a launch-time error (``SystemExit``), never a silent fallback — the same
    convention ``main.py``'s other ``--flag`` parsers use for a bad value.
    """
    if name == "wego":
        # Deferred import: avoids a module-load cycle with input_handler.py,
        # which imports ControlSource from here to declare its base class.
        from input_handler import WEGOPlanningInput
        return WEGOPlanningInput()
    if name == "gamepad":
        # Deferred import: control_gamepad imports pyray (the raylib gamepad
        # API) at module load; keeping it lazy means importing control_source
        # (which the sim-side test path and input_handler both do) never
        # requires pyray.
        from control_gamepad import GamepadDirect
        return GamepadDirect()
    if name == "keyboard":
        # #106 P6f: one unit by keyboard + mouse per-tick intents (WASD at
        # full speed, Shift / Ctrl for the slow / mid speeds), the keyboard
        # twin of GamepadDirect. Deferred import (pyray).
        from control_keyboard import KeyboardDirect
        return KeyboardDirect()
    if name == "onephase":
        # The turn-formula redesign (onephase_wego design). Picks OnePhaseWEGO
        # as its ruleset; ``debug`` re-arms the evicted diagnostic keys (§17 +
        # Erik's kickoff ruling 2).
        from control_onephase import OnePhaseWEGOInput
        return OnePhaseWEGOInput(debug=debug)
    if name == "match":
        # #121: watch the level's match.toml — its controllers drive every
        # team, once per sim tick (control_match.MatchObserver).
        from control_match import MatchObserver
        if level is None:
            raise SystemExit("--control match needs the loaded level")
        return MatchObserver(level)
    raise SystemExit(
        f"--control must be one of {_KNOWN_CONTROLS!r}, got {name!r}")


__all__ = [
    "ControlSource", "create_control_source", "quantize_stick_direction",
    "analog_speed_q", "first_controlled_unit", "play_team", "select_play",
    "IdleControl", "register_controller", "registered_controllers",
    "create_controller", "create_match_controllers",
]
