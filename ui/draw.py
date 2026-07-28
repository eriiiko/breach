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
def _dim(colour, factor: float = 0.42):
    """A muted copy of a colour — the whole treatment for an UNSELECTED
    marine's orders (Erik: "i'd like everything to stay on screen … perhaps as
    a darker color, to show that it's not the current selected unit").

    Alpha is scaled harder than the channels: a squad's worth of plans should
    recede into the floor rather than compete, while the SHAPE of each stays
    readable so you can see the whole assault at once.
    """
    return rl.Color(int(colour.r * 0.75), int(colour.g * 0.75),
                    int(colour.b * 0.75), max(28, int(colour.a * factor)))


def draw_plan_overlay(overlay, world_px_per_tile: float,
                      dimmed: bool = False) -> None:
    """Teal path line, endpoint footprint + arrival label, waypoint markers,
    shoot holograms, fire lines and skill markers (§16).

    ``dimmed`` draws the whole overlay muted, for a marine that is not the
    current selection — so the squad's plans all stay visible without the
    selected one losing its voice.
    """
    wpt = float(world_px_per_tile)
    tone = _dim if dimmed else (lambda c, *a: c)
    for path in overlay.paths:
        colour = tone(BLOCKED if path.blocked else TEAL)
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
        _footprint_box((wp.x, wp.y), wp.footprint, wpt, tone(TEAL_DIM))
        _time_label((wp.x, wp.y), wp.footprint, wpt,
                    f"{wp.arrival_seconds:.1f}", tone(TEAL_DIM))

    for marker in overlay.action_markers:
        _action_marker(marker, wpt, tone)

    # Fire lines and target ticks last, so they read on top of any path.
    for line in overlay.fire_lines:
        _fire_line(line, wpt, tone)
    for tgt in overlay.targets:
        _target_marker(tgt, wpt, tone)

    for holo in overlay.holograms:
        _footprint_fill((holo.x, holo.y), holo.footprint, wpt,
                        tone(TEAL_GHOST))
        _footprint_box((holo.x, holo.y), holo.footprint, wpt, tone(TEAL_DIM))
        _time_label((holo.x, holo.y), holo.footprint, wpt,
                    f"{holo.at_seconds:.1f}", tone(TEAL_DIM))
        if holo.target is not None:
            cx = tile_to_world_px(holo.x + holo.footprint * 0.5, wpt)
            cy = tile_to_world_px(holo.y + holo.footprint * 0.5, wpt)
            rl.draw_line_ex(
                rl.Vector2(cx, cy),
                rl.Vector2(tile_to_world_px(holo.target[0] + 0.5, wpt),
                           tile_to_world_px(holo.target[1] + 0.5, wpt)),
                max(1.0, 0.06 * wpt), tone(TEAL_GHOST))


def _fire_line(line, wpt, tone=lambda c: c) -> None:
    """THE line of fire — where this marine's ordered shot will actually go.

    Carries the information a ring on the target cannot: who is shooting at
    whom, and from where. A hovered/aim line (what you would order if you
    clicked now, or an overwatch facing) is dashed and dim; a committed one is
    solid, with a muzzle dot at the firing end so the direction reads.
    """
    x1 = tile_to_world_px(line.from_x, wpt)
    y1 = tile_to_world_px(line.from_y, wpt)
    x2 = tile_to_world_px(line.to_x, wpt)
    y2 = tile_to_world_px(line.to_y, wpt)
    th = max(1.0, (0.07 if line.hovered else 0.11) * wpt)
    colour = tone(TEAL_GHOST if line.hovered else TEAL)
    if line.hovered:
        _dashed(x1, y1, x2, y2, th, colour, dash=0.6 * wpt)
    else:
        rl.draw_line_ex(rl.Vector2(x1, y1), rl.Vector2(x2, y2), th, colour)
        rl.draw_circle(int(x1), int(y1), max(2.0, 0.16 * wpt), colour)


def _dashed(x1, y1, x2, y2, th, colour, dash) -> None:
    dx, dy = x2 - x1, y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length <= 0:
        return
    ux, uy = dx / length, dy / length
    t = 0.0
    while t < length:
        e = min(t + dash, length)
        rl.draw_line_ex(rl.Vector2(x1 + ux * t, y1 + uy * t),
                        rl.Vector2(x1 + ux * e, y1 + uy * e), th, colour)
        t = e + dash


def _target_marker(tgt, wpt, tone=lambda c: c) -> None:
    """A light tint + tick on an ordered enemy's footprint.

    Deliberately SUBTLE: the fire line is what says "this one", so a heavy
    reticle here would just be a second voice saying the same thing over the
    sprite. A hovered target gets the tint only.
    """
    colour = tone(TEAL_DIM if tgt.hovered else TEAL)
    x = tile_to_world_px(tgt.x, wpt)
    y = tile_to_world_px(tgt.y, wpt)
    side = tgt.footprint * wpt
    rl.draw_rectangle(int(x), int(y), int(side), int(side),
                      tone(rl.Color(64, 224, 208,
                                    24 if tgt.hovered else 44)))
    if not tgt.hovered:
        rl.draw_rectangle_lines_ex(rl.Rectangle(x, y, side, side),
                                   max(1.0, 0.06 * wpt), colour)
        size = max(9, int(0.6 * wpt))
        rl.draw_text(f"{tgt.action_name} {tgt.at_seconds:.1f}",
                     int(x), int(y + side + 2), size, colour)


# Skills/items get their own colour so they never read as movement.
ORANGE = rl.Color(255, 150, 40, 255)
ORANGE_FILL = rl.Color(255, 150, 40, 60)


def _action_marker(m, wpt, tone=lambda c: c) -> None:
    """A symbol where a skill/item action happens (Erik: a charge on a door
    should show a symbol, orange, indicating when it will blow up).

    Charges are ORANGE and labelled with their DETONATION countdown — the
    number you plan the stack around, not the moment the marine finishes
    planting. Everything else uses the same shape language in teal so the
    orange stays reserved for "this is going to explode".
    """
    charge = m.kind == "charge"
    colour = tone(ORANGE if charge else TEAL_DIM)
    fill = tone(ORANGE_FILL if charge else rl.Color(64, 224, 208, 40))
    cx = tile_to_world_px(m.x + 0.5, wpt)
    cy = tile_to_world_px(m.y + 0.5, wpt)
    r = max(3.0, 0.5 * wpt)

    if charge:
        # A diamond — distinct from every round/square marker on screen.
        pts = [rl.Vector2(cx, cy - r), rl.Vector2(cx + r, cy),
               rl.Vector2(cx, cy + r), rl.Vector2(cx - r, cy)]
        rl.draw_triangle(pts[0], pts[3], pts[1], fill)
        rl.draw_triangle(pts[1], pts[3], pts[2], fill)
        for a, b in zip(pts, pts[1:] + pts[:1]):
            rl.draw_line_ex(a, b, max(1.0, 0.1 * wpt), colour)
    else:
        rl.draw_circle(int(cx), int(cy), r, fill)
        rl.draw_circle_lines(int(cx), int(cy), r, colour)

    size = max(9, int(0.62 * wpt))
    text = (f"{m.at_seconds:.1f}s" if charge
            else f"{m.label} {m.at_seconds:.1f}")
    rl.draw_text(text, int(cx - r), int(cy + r + 2), size, colour)


def draw_selected_marker(unit, world_px_per_tile: float) -> None:
    """Which marine is selected (Erik: it needs to be indicated graphically).

    An arc under the feet rather than a box around the body: it never hides
    the sprite, it survives the 3D-model toggle, and it does not compete with
    the target tint, which uses a full-footprint fill.
    """
    if unit is None:
        return
    wpt = float(world_px_per_tile)
    cx = tile_to_world_px(unit.x + unit.footprint * 0.5, wpt)
    cy = tile_to_world_px(unit.y + unit.footprint * 0.5, wpt)
    r = unit.footprint * wpt * 0.62
    rl.draw_ring(rl.Vector2(cx, cy), r * 0.86, r, 0.0, 360.0, 40,
                 rl.Color(64, 224, 208, 90))
    rl.draw_ring(rl.Vector2(cx, cy), r * 0.86, r, -40.0, 40.0, 16, TEAL)
    rl.draw_ring(rl.Vector2(cx, cy), r * 0.86, r, 140.0, 220.0, 16, TEAL)


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


def draw_cover(sim, world_px_per_tile: float) -> None:
    """Cover rectangles (§7).

    Cover is INTANGIBLE — it is a continuous-space shape, not a tile — so the
    tileset never draws it and it has to be drawn here or the player is being
    protected by something invisible. Greybox for now (art is a later pass);
    what has to read correctly is the SHAPE, since geometry is the whole
    mechanic, and the damage state, since a chewed crate is about to stop
    protecting anybody.

    A ``blocks_los`` barricade is drawn solid and taller-looking; an ordinary
    crate is drawn lower and semi-transparent, because you can see over it.
    """
    wpt = float(world_px_per_tile)
    for c in getattr(sim, "cover", ()) or ():
        if not c.alive:
            continue
        x = c.x0 * wpt
        y = c.y0 * wpt
        w = (c.x1 - c.x0) * wpt
        h = (c.y1 - c.y0) * wpt
        if c.blocks_los:
            fill = rl.Color(96, 92, 86, 255)
            edge = rl.Color(150, 146, 138, 255)
        else:
            fill = rl.Color(120, 104, 74, 180)
            edge = rl.Color(178, 156, 112, 255)
        rl.draw_rectangle(int(x), int(y), int(w), int(h), fill)
        rl.draw_rectangle_lines_ex(rl.Rectangle(x, y, w, h),
                                   max(1.0, 0.1 * wpt), edge)
        # Damage read: a red bar along the top edge as the HP goes.
        if c.hp_now < c.hp_max:
            frac = max(0.0, c.hp_now / float(c.hp_max))
            rl.draw_rectangle(int(x), int(y), int(w * frac),
                              max(2, int(0.12 * wpt)),
                              rl.Color(210, 70, 50, 220))


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


def draw_armed_readout(action_label: str, seconds, screen_w: int,
                       screen_h: int) -> None:
    """What is armed, and — for a time-carrying action — the wheel-set moment.

    Without this the wheel dial is invisible: you would be scrolling a number
    you cannot see. Sits just above the hotbar, where the eye already is.
    """
    if not action_label:
        return
    text = action_label if seconds is None else \
        f"{action_label}  @ {seconds:.2f}s   [wheel to adjust]"
    size = 16
    w = rl.measure_text(text, size) + 20
    x = (screen_w - w) // 2
    y = screen_h - 100
    rl.draw_rectangle(x, y, w, 26, HUD_BG)
    rl.draw_rectangle_lines(x, y, w, 26, ORANGE if seconds is not None
                            else HUD_EDGE)
    rl.draw_text(text, x + 10, y + 5, size,
                 ORANGE if seconds is not None else HUD_TEXT)


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
    "draw_cover", "draw_ds3_menu", "draw_flashlight_cones", "draw_hotbar",
    "draw_marks", "draw_overwatch_cone", "draw_plan_overlay",
    "draw_planning_clock", "draw_round_banner", "draw_selected_marker",
]
