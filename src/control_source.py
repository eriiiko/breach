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
        needs, e.g. ``mouse_to_tile()``).
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


# Launch-flag name -> factory. P3 adds "gamepad" (GamepadDirect); the ML entry
# point (AgentPolicy) is driven differently — it has no per-frame poll loop —
# so it is not expected to route through this same factory; see the design doc
# §3b.
_KNOWN_CONTROLS = ("wego", "gamepad")


def create_control_source(name: str) -> "ControlSource":
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
    raise SystemExit(
        f"--control must be one of {_KNOWN_CONTROLS!r}, got {name!r}")


__all__ = [
    "ControlSource", "create_control_source", "quantize_stick_direction",
]
