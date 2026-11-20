"""The on-screen light curve editor (graphics G5, #64): pure, headless, testable.

A Photoshop Curves panel over ``renderer/light_curve.py``. All its decisions
live here as data (the ``ui.model`` split): the layout, what a press hits, what
a drag does. The shell (``GameRenderer.update_light_curve_panel``) feeds it the
mouse; ``ui.draw.draw_light_curve_panel`` draws it. No raylib in this file.

Left-press a point and drag; left-press the plot to add a point there;
right-press a point to delete it. The first and last point stay.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from renderer.light_curve import (IDENTITY_POINTS, LightCurve, MIN_POINT_GAP,
                                  check_points)

PLOT_PX = 256
MARGIN_PX = 12
ROW_PX = 22
GRAB_PX = 8                 # a press this close to a point takes the point
PANEL_W = PLOT_PX + 2 * MARGIN_PX
PANEL_H = PLOT_PX + 2 * MARGIN_PX + 2 * ROW_PX + 8


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def contains(self, mx: float, my: float) -> bool:
        return (self.x <= mx < self.x + self.w
                and self.y <= my < self.y + self.h)


@dataclass(frozen=True)
class PanelLayout:
    panel: Rect
    plot: Rect
    toggle: Rect
    reset: Rect
    save: Rect
    dark_sat: Rect


def panel_layout(screen_w: int) -> PanelLayout:
    """Top right of the screen."""
    x0 = screen_w - PANEL_W - MARGIN_PX
    y0 = MARGIN_PX
    plot = Rect(x0 + MARGIN_PX, y0 + MARGIN_PX, PLOT_PX, PLOT_PX)
    row1 = plot.y + PLOT_PX + 8
    row2 = row1 + ROW_PX
    third = PLOT_PX // 3
    return PanelLayout(
        panel=Rect(x0, y0, PANEL_W, PANEL_H), plot=plot,
        toggle=Rect(plot.x, row1, third - 4, ROW_PX - 4),
        reset=Rect(plot.x + third, row1, third - 4, ROW_PX - 4),
        save=Rect(plot.x + 2 * third, row1, PLOT_PX - 2 * third, ROW_PX - 4),
        dark_sat=Rect(plot.x, row2, PLOT_PX, ROW_PX - 4))


def to_screen(layout: PanelLayout, x: float, y: float) -> tuple:
    p = layout.plot
    return p.x + x * p.w, p.y + (1.0 - y) * p.h


def to_curve(layout: PanelLayout, mx: float, my: float) -> tuple:
    p = layout.plot
    return (min(1.0, max(0.0, (mx - p.x) / p.w)),
            min(1.0, max(0.0, 1.0 - (my - p.y) / p.h)))


class CurvePanel:
    """The edit in progress: a ``LightCurve`` the renderer uses every frame.

    ``saved`` is what the file holds, so the shell can tell an unsaved edit
    and a config reload (``base`` moved) from a drag."""

    def __init__(self, curve: LightCurve):
        self.curve = curve
        self.base = curve
        self.saved = True
        self.drag: Optional[int] = None         # index of the held point
        self.drag_sat = False

    def _set(self, **changes) -> None:
        self.curve = replace(self.curve, **changes)
        self.saved = False

    def captures(self, layout: PanelLayout, mx: float, my: float) -> bool:
        """The mouse belongs to the panel, not the game."""
        return (self.drag is not None or self.drag_sat
                or layout.panel.contains(mx, my))

    def point_at(self, layout: PanelLayout, mx: float, my: float):
        for i, (x, y) in enumerate(self.curve.points):
            sx, sy = to_screen(layout, x, y)
            if (sx - mx) ** 2 + (sy - my) ** 2 <= GRAB_PX ** 2:
                return i
        return None

    def press(self, layout: PanelLayout, mx: float, my: float) -> Optional[str]:
        """Left press. Returns ``"save"`` when the shell must write the file."""
        if layout.toggle.contains(mx, my):
            self._set(enabled=not self.curve.enabled)
        elif layout.reset.contains(mx, my):
            self._set(points=IDENTITY_POINTS, dark_saturation=1.0)
        elif layout.save.contains(mx, my):
            return "save"
        elif layout.dark_sat.contains(mx, my):
            self.drag_sat = True
            self.move(layout, mx, my)
        elif layout.plot.contains(mx, my):
            i = self.point_at(layout, mx, my)
            if i is None:
                i = self._insert(*to_curve(layout, mx, my))
            self.drag = i
        return None

    def _insert(self, x: float, y: float) -> Optional[int]:
        pts = list(self.curve.points)
        i = sum(1 for p in pts if p[0] < x)
        if (0 < i and x - pts[i - 1][0] < MIN_POINT_GAP) or (
                i < len(pts) and pts[i][0] - x < MIN_POINT_GAP):
            return None             # on top of a neighbour: nothing to add
        pts.insert(i, (x, y))
        self._set(points=check_points(pts))
        return i

    def move(self, layout: PanelLayout, mx: float, my: float) -> None:
        """The mouse moved with the left button held."""
        if self.drag_sat:
            r = layout.dark_sat
            self._set(dark_saturation=min(1.0, max(0.0, (mx - r.x) / r.w)))
        elif self.drag is not None:
            pts = list(self.curve.points)
            i = self.drag
            x, y = to_curve(layout, mx, my)
            lo = pts[i - 1][0] + MIN_POINT_GAP if i > 0 else 0.0
            hi = pts[i + 1][0] - MIN_POINT_GAP if i < len(pts) - 1 else 1.0
            pts[i] = (min(hi, max(lo, x)), y)
            self._set(points=check_points(pts))

    def release(self) -> None:
        self.drag = None
        self.drag_sat = False

    def right_press(self, layout: PanelLayout, mx: float, my: float) -> None:
        """Delete the point under the mouse, bar the two ends."""
        i = self.point_at(layout, mx, my)
        if i is not None and 0 < i < len(self.curve.points) - 1:
            pts = list(self.curve.points)
            del pts[i]
            self._set(points=check_points(pts))

    def mark_saved(self) -> None:
        self.saved = True


__all__ = ["CurvePanel", "PANEL_H", "PANEL_W", "PanelLayout", "Rect",
           "panel_layout", "to_curve", "to_screen"]
