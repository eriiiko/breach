"""``ui`` — the OnePhaseWEGO interface layer (onephase_wego design §16).

"**UI architecture:** a standalone ``ui/`` package, NOT a framework. Two seams:
it *reads* sim state + action registry + inventory; it *writes* only orders
through the ControlSource path. Hotbar, overlays, DS3 menu all live there,
shared by both rulesets."

The package is split so that the seam is enforceable rather than merely
intended:

- :mod:`ui.model` is **pure and headless** — it imports no ``pyray``, touches
  no window, and mutates no simulation. Every question the interface asks
  ("what should this hotbar slot say?", "where will this marine be at 2.3 s?",
  "which enemies may I draw?") is answered here, as data. That is what makes
  the UI testable at all, and it is where essentially all of the logic lives.
- :mod:`ui.draw` turns those structures into raylib calls and does nothing
  else. It is deliberately dumb: if a decision is being made in draw.py, it is
  in the wrong file.

Nothing in this package writes to the sim. Orders are placed by the
ControlSource (``src/control_onephase.py``), which is the only writer.
"""
from __future__ import annotations

from ui.model import (  # noqa: F401
    DEFAULT_HOTBAR, DS3_PAGES, FlashlightCone, HotbarSlot, Hologram,
    MenuModel, MenuRow, PathViz, PlanOverlay, PlanningClock, WaypointMarker,
    bind_slot, default_bindings, drawable_enemies, ds3_menu, flashlight_cones,
    hotbar, plan_overlay, planning_clock, position_at,
)

__all__ = [
    "DEFAULT_HOTBAR", "DS3_PAGES", "FlashlightCone", "HotbarSlot", "Hologram",
    "MenuModel", "MenuRow", "PathViz", "PlanOverlay", "PlanningClock",
    "WaypointMarker", "bind_slot", "default_bindings", "drawable_enemies",
    "ds3_menu", "flashlight_cones", "hotbar", "plan_overlay", "planning_clock",
    "position_at",
]
