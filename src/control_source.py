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
(Direct control is ``control_direct.DirectControl`` since the gameplay pack's
G1p: the pad and keyboard + mouse halves of one scheme.)
(``AgentPolicy`` exists since #29 P1a as the match controller ``policy``,
``control_policy.py``: a trainer's action rows through the intent facade.)
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


def quantize_point_aim(ox: float, oy: float, px: float, py: float,
                       deadzone: float = 0.0):
    """Aim at a POINT (gameplay pack G1p, the mouse): the Q16.16 unit vector
    from ``(ox, oy)`` (the possessed unit's centre, world tiles) to
    ``(px, py)`` (the cursor's world position) through THE seam,
    :func:`quantize_stick_direction` (so its normalisation, rounding and
    deadzone are the stick's), plus the distance in Q16.16 tiles (a throw's
    reach). ``(dx_q, dy_q, reach_q)``; ``(0, 0, 0)`` when the point is within
    ``deadzone`` tiles. ``math.sqrt`` is correctly rounded, so the reach is
    as deterministic as the direction."""
    vx = float(px) - float(ox)
    vy = float(py) - float(oy)
    dx_q, dy_q, mag_q = quantize_stick_direction(vx, vy, deadzone)
    if mag_q == 0:
        return (0, 0, 0)
    return (dx_q, dy_q, quantize_scalar(math.sqrt(vx * vx + vy * vy)))


def stick_throttle_q(mag_q: int, deadzone: float) -> int:
    """A stick's deflection as the move THROTTLE (G1p), Q16 in ``[0, ONE]``:
    ``mag_q`` (the clamped magnitude :func:`quantize_stick_direction`
    returned) rescaled so the edge of the ``deadzone`` is 0 (the walk) and
    full deflection is ONE (the sprint). Pure; 0 for a centred stick."""
    mag = mag_q / 65536.0
    if mag <= deadzone:
        return 0
    t = (mag - deadzone) / (1.0 - deadzone)
    return quantize_scalar(1.0 if t > 1.0 else t)


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

    def bind_match(self, sim, spec) -> None:
        """Called ONCE by :func:`create_match_controllers` on every
        team-row controller right after the team rows are built (#195 1j):
        the one way a match controller learns its match's spec (``spec``,
        the parsed ``MatchSpec`` as played; ``sim`` spawned). A no-op for
        every controller but the door bots, which build their observation
        context and action tables here and refuse a level without a room
        graph."""
        return None

    def on_tick(self, sim) -> None:
        """Called by main.py right before EVERY ``sim.step()`` (#121) — the
        per-tick hook a scheme that drives units by the tick (the match
        observer's controllers) uses, so the window plays the same ticks as
        the headless runner. A no-op for every per-frame human scheme."""
        return None

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

    @property
    def owns_camera_keys(self) -> bool:
        """Are the camera's KEYS this scheme's (gameplay pack G1p)? ``True``
        for direct control, whose movement keys are data
        (``[controls.keys]``): main.py then neither pans the view with
        WASD / arrows nor zooms with Q / E (the wheel and -/+ still zoom).
        ``False`` for every other scheme -- their keymaps are unchanged."""
        return False

    #: How main.py frames this scheme's possessed unit (``[display]
    #: camera_mode``, ``renderer/follow_camera.py``): ``None`` = no
    #: possessed-unit camera (every scheme but direct control).
    camera_mode = None


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
    """The layered bot team (#183 G8p, ``control_bots.BotTeam``: team / unit
    / position layers, settings ``config.toml [bots]``). Deferred import:
    control_bots pulls in the sim's tables and the pathfinder, which
    importing this module must not."""
    from control_bots import BotTeam
    return BotTeam(team)


def _bots_v1(team: int) -> "ControlSource":
    """The first rule-based bot team (#121, ``control_bots_v1.BotTeamV1``),
    kept unchanged beside the layered bots as a second opponent (#183 G8p:
    a ladder of opponents)."""
    from control_bots_v1 import BotTeamV1
    return BotTeamV1(team)


def _zombies(team: int) -> "ControlSource":
    """The horde (#122, ``control_zombies.ZombieControl``): the shipped
    zombie law, run through the ZOMBIE intent for every living zombie."""
    from control_zombies import ZombieControl
    return ZombieControl(team)


def _policy(team: int) -> "ControlSource":
    """The ``policy`` controller (#29 P1a, ``control_policy.AgentPolicy`` --
    the ML entry point this module reserved): issues a trainer's action rows
    through the intent facade, one row per roster slot. Built by code only
    (the batched environment, ``src/rl_env.py``): a match file cannot name it
    (registered with ``nameable=False``), since nothing would hand it rows."""
    from control_policy import AgentPolicy
    return AgentPolicy(team)


def _commander(team: int) -> "ControlSource":
    """The commander (#195 1f, ``control_bots.BotCommander``): the layered
    bots' TEAM layer alone, writing its play, objective and target as the
    team's ORDERS through ``Simulation.order_intent`` -- and nothing else.
    Built by :func:`create_match_controllers` only, for a team whose row
    names a :data:`DOOR_CONTROLLERS` controller on a level with a room
    graph; a match file cannot name it."""
    from control_bots import BotCommander
    return BotCommander(team)


def _replay(team, recording=None) -> "ControlSource":
    """The replay controller (#126, ``match_replay.ReplayControl``): feeds a
    recorded match's intents back. Built only from a recording, by
    ``create_match_controllers(sim, spec, recording=...)``; a match file
    cannot name it (registered with ``nameable=False``)."""
    if recording is None:
        raise ValueError("controller 'replay' is built from a recording: "
                         "create_match_controllers(sim, spec, recording=...) "
                         "(tools/run_match.py --replay)")
    from match_replay import ReplayControl
    return ReplayControl(recording, team)


#: The door bots (#195 1j, ``control_bots_door``): the layered bots driven
#: through action spec v1, the design run and its variants (design 2.7),
#: NAMEABLE (a match.toml or a run file may put them on a side; their
#: ``bind_match`` refuses a level without a room graph). The rows are
#: ``control_bots_door.DOOR_VARIANTS`` (held equal at its import).
DOOR_BOT_NAMES = ("bots_door", "bots_door_dirs8", "bots_door_every6",
                  "bots_door_every8", "bots_door_trigger_tick",
                  "bots_door_throw_dir")


def _bots_door(name):
    def factory(team: int) -> "ControlSource":
        """A door-bots variant (``control_bots_door.create``)."""
        from control_bots_door import create
        return create(team, name)
    return factory


_MATCH_CONTROLLERS: dict = {
    "idle": IdleControl,
    "bots": _bots,
    "bots_v1": _bots_v1,
    "zombies": _zombies,
    "replay": _replay,
    "policy": _policy,
    "commander": _commander,
}
for _name in DOOR_BOT_NAMES:
    _MATCH_CONTROLLERS[_name] = _bots_door(_name)
del _name
#: Registered controllers a match.toml may NOT name (they need more than a
#: team to be built, or something only code hands them): the match door reads
#: :func:`match_file_controllers`.
_NOT_NAMEABLE: set = {"replay", "policy", "commander"}
#: The controllers that drive a team THROUGH THE DOOR (the action spec,
#: #195 1f) WITH NO BRAIN OF THEIR OWN: a team on one of them gets the bots'
#: commander (``commander``) beside it on a level with a room graph, so its
#: units hold orders. The door bots (#195 1j, ``DOOR_BOT_NAMES``) are not
#: here: they carry their own brain and write their own orders.
DOOR_CONTROLLERS = frozenset({"policy"})


def register_controller(name: str, factory, *, nameable: bool = True) -> None:
    """Register ``factory`` (``team -> ControlSource``) under ``name``;
    refuses a name already taken (never a silent override). ``nameable=False``
    keeps it out of match files (built by code only, like ``replay``)."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"controller name must be a non-empty str "
                         f"(got {name!r})")
    if name in _MATCH_CONTROLLERS:
        raise ValueError(f"controller {name!r} is already registered")
    _MATCH_CONTROLLERS[name] = factory
    if not nameable:
        _NOT_NAMEABLE.add(name)


def registered_controllers() -> tuple:
    """The registered match-controller names, sorted."""
    return tuple(sorted(_MATCH_CONTROLLERS))


def match_file_controllers() -> tuple:
    """The registered controllers a match.toml may name, sorted."""
    return tuple(sorted(set(_MATCH_CONTROLLERS) - _NOT_NAMEABLE))


def create_controller(name: str, team: int, **options) -> "ControlSource":
    """Build the registered controller ``name`` for ``team``; ``options``
    go to its factory (only ``replay`` takes one, ``recording``)."""
    try:
        factory = _MATCH_CONTROLLERS[name]
    except KeyError:
        raise ValueError(f"controller {name!r} is not registered (known: "
                         f"{list(registered_controllers())})") from None
    return factory(team, **options)


def create_match_controllers(sim, spec, *, recording=None) -> list:
    """THE one way a match's controllers are built (#122): one per
    ``[[team]]`` row (team i -> its ``controller``), then the horde's
    (``spec.zombie_controller`` for ``ZOMBIE_TEAM``) -- after handing the
    horde to it (``sim.hand_zombies_to_controller``), so the conductor no
    longer drives the zombies itself: every unit of a match has exactly one
    driver, and it is a ControlSource. Call once per match, after
    ``sim.spawn_match``; then ``handle_frame(sim, None)`` on each, every
    tick, before ``sim.step()``.

    ``recording`` (#126, a ``match_replay.Recording``): replay it instead --
    the horde is handed over the same way, and ONE ``replay`` controller
    re-issues every recorded intent of every team and the horde, in the
    order they were issued.

    A team whose row names a :data:`DOOR_CONTROLLERS` controller gets the
    ``commander`` after the team rows, on a level whose room graph is at
    the map's shape (#195 1f); a replay re-issues its recorded orders
    instead."""
    from simulation.factions import ZOMBIE_TEAM
    from simulation.match import DEFAULT_ZOMBIE_CONTROLLER
    if recording is not None:
        sim.hand_zombies_to_controller()
        return [create_controller("replay", None, recording=recording)]
    controllers = [create_controller(row.controller, team)
                   for team, row in enumerate(spec.teams)]
    # #195 1j: every team-row controller learns its match (a no-op for all
    # but the door bots, which build their observation context here)
    for c in controllers:
        c.bind_match(sim, spec)
    # #195 1f: a team driven through the door gets the commander (orders on
    # the room graph) -- only where an order can name a place.
    from simulation.unit_orders import graph_of
    if graph_of(sim) is not None:
        controllers += [create_controller("commander", team)
                        for team, row in enumerate(spec.teams)
                        if row.controller in DOOR_CONTROLLERS]
    sim.hand_zombies_to_controller()
    controllers.append(create_controller(
        getattr(spec, "zombie_controller", DEFAULT_ZOMBIE_CONTROLLER),
        ZOMBIE_TEAM))
    return controllers


# Launch-flag name -> factory. P3 added "gamepad"; G1p (gameplay pack v1) made
# it an alias of "direct" (control_direct.DirectControl: the pad AND keyboard
# + mouse, whichever was touched last). The ML entry point (AgentPolicy) is
# driven differently — it has no per-frame poll loop — so it does not route
# through this factory: it is the match controller "policy" above (#29 P1a),
# built by the batched environment.
_KNOWN_CONTROLS = ("wego", "direct", "gamepad", "onephase", "match")


def create_control_source(name: str, debug: bool = False,
                          level=None) -> "ControlSource":
    """Factory for the ``--control`` launch flag (§3b).

    ``name`` must be one of :data:`_KNOWN_CONTROLS`: ``"wego"`` (today's
    default WEGO planning input), ``"direct"`` (G1p direct control — one
    possessed marine, per-tick intents, ContinuousRealtime, pad or keyboard +
    mouse; ``"gamepad"`` is its alias), ``"onephase"`` or ``"match"``. An unknown name
    is a launch-time error (``SystemExit``), never a silent fallback — the same
    convention ``main.py``'s other ``--flag`` parsers use for a bad value.
    """
    if name == "wego":
        # Deferred import: avoids a module-load cycle with input_handler.py,
        # which imports ControlSource from here to declare its base class.
        from input_handler import WEGOPlanningInput
        return WEGOPlanningInput()
    if name in ("direct", "gamepad"):
        # Deferred import: the direct scheme's halves import pyray (the
        # raylib input API); keeping it lazy means importing control_source
        # (which the sim-side test path and input_handler both do) never
        # requires pyray. "gamepad" is the alias kept from P3 (G1p).
        from control_direct import DirectControl
        try:
            return DirectControl()
        except ValueError as e:
            raise SystemExit(f"config.toml: {e}")
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
    "quantize_point_aim", "stick_throttle_q",
    "IdleControl", "register_controller", "registered_controllers",
    "DOOR_CONTROLLERS", "DOOR_BOT_NAMES",
    "create_controller", "create_match_controllers", "match_file_controllers",
]
