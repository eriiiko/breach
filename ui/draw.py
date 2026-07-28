"""UI drawing — raylib calls over :mod:`ui.model`'s structures (design §16).

Deliberately dumb. Every decision was already made in :mod:`ui.model`; this
file turns the resulting data into pixels and does nothing else. If a rule is
being applied here, it is in the wrong file — that separation is what keeps
the interface testable, since ``ui.model`` has no window and no raylib.

Two coordinate spaces, matching the renderer's existing split:

- **world** draws (paths, holograms, cones, marks) go into the world render
  target, where ``world_px_per_tile`` is the only scale involved and the
  camera transforms everything afterwards — the same contract
  ``GameRenderer._draw_orders_world`` works under;
- **screen** draws (hotbar, planning clock, DS3 menu) are plain overlay
  pixels.
"""
from __future__ import annotations

import math

import pyray as rl

from renderer.coords import tile_to_world_px

# The design names ONE colour: teal, for paths, endpoint footprints and
# holograms (§16). Everything else here is a neutral HUD grey so the teal
# stays the signal.
TEAL = rl.Color(64, 224, 208, 255)
TEAL_DIM = rl.Color(64, 224, 208, 110)
TEAL_GHOST = rl.Color(64, 224, 208, 70)
BLOCKED = rl.Color(230, 90, 70, 200)
HUD_BG = rl.Color(18, 20, 24, 220)
HUD_EDGE = rl.Color(90, 100, 110, 255)
HUD_TEXT = rl.Color(220, 228, 235, 255)
HUD_DIM = rl.Color(120, 130, 140, 255)
MARK_COLOR = rl.Color(255, 190, 60, 255)


# ---------------------------------------------------------------------------
# World-space overlays
# ---------------------------------------------------------------------------
def draw_plan_overlay(overlay, world_px_per_tile: float) -> None:
    """Teal path line, endpoint footprint + arrival label, waypoint markers,
    and shoot holograms (§16)."""
    wpt = float(world_px_per_tile)
    for path in overlay.paths:
        colour = BLOCKED if path.blocked else TEAL
        pts = path.points
        for a, b in zip(pts, pts[1:]):
            rl.draw_line_ex(
                rl.Vector2(tile_to_world_px(a[0] + 0.5, wpt),
                           tile_to_world_px(a[1] + 0.5, wpt)),
                rl.Vector2(tile_to_world_px(b[0] + 0.5, wpt),
                           tile_to_world_px(b[1] + 0.5, wpt)),
                max(1.0, 0.12 * wpt), colour)
        _footprint_box(path.endpoint, path.footprint, wpt, colour)
        _time_label(path.endpoint, path.footprint, wpt,
                    f"{path.arrival_seconds:.1f}", colour)

    for wp in overlay.waypoints:
        _footprint_box((wp.x, wp.y), wp.footprint, wpt, TEAL_DIM)
        _time_label((wp.x, wp.y), wp.footprint, wpt,
                    f"{wp.arrival_seconds:.1f}", TEAL_DIM)

    for holo in overlay.holograms:
        _footprint_fill((holo.x, holo.y), holo.footprint, wpt, TEAL_GHOST)
        _footprint_box((holo.x, holo.y), holo.footprint, wpt, TEAL_DIM)
        _time_label((holo.x, holo.y), holo.footprint, wpt,
                    f"{holo.at_seconds:.1f}", TEAL_DIM)
        if holo.target is not None:
            cx = tile_to_world_px(holo.x + holo.footprint * 0.5, wpt)
            cy = tile_to_world_px(holo.y + holo.footprint * 0.5, wpt)
            rl.draw_line_ex(
                rl.Vector2(cx, cy),
                rl.Vector2(tile_to_world_px(holo.target[0] + 0.5, wpt),
                           tile_to_world_px(holo.target[1] + 0.5, wpt)),
                max(1.0, 0.06 * wpt), TEAL_GHOST)


def _footprint_box(pos, footprint, wpt, colour) -> None:
    x = tile_to_world_px(pos[0], wpt)
    y = tile_to_world_px(pos[1], wpt)
    side = footprint * wpt
    rl.draw_rectangle_lines_ex(rl.Rectangle(x, y, side, side),
                               max(1.0, 0.08 * wpt), colour)


def _footprint_fill(pos, footprint, wpt, colour) -> None:
    x = tile_to_world_px(pos[0], wpt)
    y = tile_to_world_px(pos[1], wpt)
    side = footprint * wpt
    rl.draw_rectangle(int(x), int(y), int(side), int(side), colour)


def _time_label(pos, footprint, wpt, text, colour) -> None:
    """The arrival timestamp — "2.3" = arrives 2.3 s into the round (§16)."""
    x = tile_to_world_px(pos[0], wpt)
    y = tile_to_world_px(pos[1], wpt)
    size = max(10, int(0.7 * wpt))
    rl.draw_text(text, int(x + 2), int(y - size - 2), size, colour)


def draw_overwatch_cone(unit, world_px_per_tile: float) -> None:
    """The overwatch cone (§9) — its width is player-set target control, so it
    has to be legible on screen or the dial is unusable."""
    if getattr(unit, "overwatch_facing", None) is None:
        return
    wpt = float(world_px_per_tile)
    cx = tile_to_world_px(unit.center_tile_x() + 0.5, wpt)
    cy = tile_to_world_px(unit.center_tile_y() + 0.5, wpt)
    half = float(unit.overwatch_half_deg or 0.0)
    # Screen y is down; facing is y-up (the standard one negation).
    centre_deg = -math.degrees(float(unit.overwatch_facing))
    rl.draw_circle_sector(rl.Vector2(cx, cy), 8.0 * wpt,
                          centre_deg - half, centre_deg + half, 24,
                          rl.Color(64, 224, 208, 28))


def draw_flashlight_cones(cones, world_px_per_tile: float) -> None:
    """Marine flashlights — RENDER ONLY (§8); they never reach the sim."""
    wpt = float(world_px_per_tile)
    for cone in cones:
        cx = tile_to_world_px(cone.x + 0.5, wpt)
        cy = tile_to_world_px(cone.y + 0.5, wpt)
        centre_deg = -math.degrees(cone.facing)
        rl.draw_circle_sector(
            rl.Vector2(cx, cy), cone.range_tiles * wpt,
            centre_deg - cone.half_deg, centre_deg + cone.half_deg, 32,
            rl.Color(255, 250, 235, 16))


def draw_marks(sim, team: int, world_px_per_tile: float) -> None:
    """Marked targets (§11) — the mark has to be visible or marking is a
    keypress with no feedback."""
    marks = sim.marks.get(int(team), set())
    if not marks:
        return
    wpt = float(world_px_per_tile)
    for u in sim.units:
        if int(u.id) not in marks or not u.alive:
            continue
        cx = tile_to_world_px(u.center_tile_x() + 0.5, wpt)
        cy = tile_to_world_px(u.center_tile_y() + 0.5, wpt)
        r = 0.9 * wpt
        rl.draw_circle_lines(int(cx), int(cy), r, MARK_COLOR)
        rl.draw_circle_lines(int(cx), int(cy), r * 0.7, MARK_COLOR)


# ---------------------------------------------------------------------------
# Screen-space HUD
# ---------------------------------------------------------------------------
def draw_hotbar(slots, screen_w: int, screen_h: int) -> None:
    """The hotbar (§16) — the action registry, rendered."""
    n = len(slots)
    if not n:
        return
    cell = 54
    pad = 6
    total = n * cell + (n - 1) * pad
    x0 = (screen_w - total) // 2
    y0 = screen_h - cell - 18

    for slot in slots:
        x = x0 + slot.index * (cell + pad)
        rl.draw_rectangle(x, y0, cell, cell, HUD_BG)
        rl.draw_rectangle_lines(x, y0, cell, cell,
                                HUD_EDGE if slot.bound else
                                rl.Color(60, 66, 72, 255))
        rl.draw_text(slot.key_label, x + 4, y0 + 3, 12, HUD_DIM)
        if not slot.bound:
            continue
        colour = HUD_TEXT if slot.enabled else HUD_DIM
        rl.draw_text(_fit(slot.label, 9), x + 5, y0 + 20, 11, colour)
        if slot.count is not None:
            rl.draw_text(f"x{slot.count}", x + cell - 22, y0 + 3, 12, colour)
        if slot.cooldown_remaining > 0:
            # A cooling slot is shaded from the bottom, so the bar reads as a
            # timer at a glance rather than a number to parse.
            frac = min(1.0, slot.cooldown_remaining / 2.0)
            h = int(cell * frac)
            rl.draw_rectangle(x, y0 + cell - h, cell, h,
                              rl.Color(0, 0, 0, 130))
            rl.draw_text(f"{slot.cooldown_remaining:.1f}", x + 5,
                         y0 + cell - 15, 11, HUD_DIM)


def _fit(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n - 1] + "."


def draw_planning_clock(clock, screen_w: int) -> None:
    """The submit-within-N timer (§16). Disabled (single-player) draws
    nothing at all — an always-visible 'untimed' badge would be noise."""
    if not clock.enabled:
        return
    w, h = 260, 10
    x = (screen_w - w) // 2
    y = 14
    rl.draw_rectangle(x, y, w, h, HUD_BG)
    rl.draw_rectangle(x, y, int(w * clock.fraction), h,
                      BLOCKED if clock.remaining_seconds <= 3.0 else TEAL)
    rl.draw_rectangle_lines(x, y, w, h, HUD_EDGE)
    rl.draw_text(f"SUBMIT {clock.remaining_seconds:.0f}s", x + w + 10, y - 3,
                 14, HUD_TEXT)


def draw_round_banner(sim, screen_w: int) -> None:
    """Round number + where we are inside it. The round is short (§2), so the
    player needs the clock legible without hunting for it."""
    secs = sim.round_tick / float(sim.ticks_per_round) \
        * (sim.ticks_per_round / 24.0)
    text = (f"ROUND {sim.round_index + 1}   "
            f"{secs:.1f}s / {sim.ticks_per_round / 24.0:.1f}s"
            f"{'   [PLANNING]' if sim.is_paused() else ''}")
    rl.draw_text(text, screen_w // 2 - 130, 30, 16,
                 TEAL if sim.is_paused() else HUD_TEXT)


def draw_ds3_menu(model, screen_w: int, screen_h: int) -> None:
    """The Dark Souls 3-pattern menu (§15): page tabs across the top, rows
    beneath. Overlays; in WEGO the planning pause hosts it naturally."""
    w, h = 520, 360
    x = (screen_w - w) // 2
    y = (screen_h - h) // 2
    rl.draw_rectangle(x, y, w, h, rl.Color(12, 14, 18, 240))
    rl.draw_rectangle_lines(x, y, w, h, HUD_EDGE)

    tab_x = x + 16
    for i, page in enumerate(model.pages):
        active = (i == model.page_index)
        rl.draw_text(page, tab_x, y + 14, 16,
                     TEAL if active else HUD_DIM)
        tab_x += rl.measure_text(page, 16) + 18

    row_y = y + 52
    for row in model.rows:
        rl.draw_text(row.label, x + 20, row_y, 15, HUD_TEXT)
        if row.value:
            rl.draw_text(row.value, x + w - 20 - rl.measure_text(row.value, 15),
                         row_y, 15, HUD_DIM)
        row_y += 24
    if not model.rows:
        rl.draw_text("(empty)", x + 20, row_y, 15, HUD_DIM)


__all__ = [
    "draw_ds3_menu", "draw_flashlight_cones", "draw_hotbar", "draw_marks",
    "draw_overwatch_cone", "draw_plan_overlay", "draw_planning_clock",
    "draw_round_banner",
]
