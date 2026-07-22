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


# Launch-flag name -> factory. Extended in P3 with "gamepad" (GamepadDirect)
# and later the ML entry point (AgentPolicy is driven differently — it has
# no per-frame poll loop — so it is not expected to route through this same
# factory; see the design doc §3b).
_KNOWN_CONTROLS = ("wego",)


def create_control_source(name: str) -> "ControlSource":
    """Factory for the ``--control`` launch flag (§3b).

    ``name`` must be one of :data:`_KNOWN_CONTROLS` — in P2 that is only
    ``"wego"``, today's default WEGO planning input. An unknown name is a
    launch-time error (``SystemExit``), never a silent fallback — the same
    convention ``main.py``'s other ``--flag`` parsers use for a bad value.
    """
    if name == "wego":
        # Deferred import: avoids a module-load cycle with input_handler.py,
        # which imports ControlSource from here to declare its base class.
        from input_handler import WEGOPlanningInput
        return WEGOPlanningInput()
    raise SystemExit(
        f"--control must be one of {_KNOWN_CONTROLS!r}, got {name!r}")


__all__ = ["ControlSource", "create_control_source"]
