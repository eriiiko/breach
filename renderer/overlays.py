"""Overlays: smoke, fire, units, orders, debug HUDs.

These are drawn on top of the lit ship layer. Most use simple rectangle/line
draws via pyray. Smoke and fire are uploaded as dynamic RGBA textures at
physics resolution and drawn stretched over the map area.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import pyray as rl

from . import core
from .blackbody import BlackbodyRamp, pack_emissive_rgba
from .coords import tile_to_world_px


# ----------------------------------------------------------------------------
# Smoke + Fire overlays (physics-resolution textures stretched to map area)
# ----------------------------------------------------------------------------

class FieldOverlay:
    """Holds a dynamic RGBA texture for a scalar physics field.

    Use for smoke (gray semi-transparent) and fire (orange glow).
    """

    def __init__(self, grid_h: int, grid_w: int, tint=(180, 180, 200),
                 max_alpha=255, gamma: float = 1.0):
        self.h = grid_h
        self.w = grid_w
        self.tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        self.packed = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
        self.tint_r, self.tint_g, self.tint_b = tint
        self.max_alpha = max_alpha
        # Render-contrast power curve applied to the rendered density in update()
        # (ch.05 §6.1 step 5 "smoke^gamma"). RENDER-ONLY — never touches the sim
        # field. gamma > 1 crushes thin smoke toward transparent and sharpens
        # wispy edges (filmic); 1.0 = identity (linear opacity, the default for
        # the base class so any other field is untouched).
        # The LEGACY smoke overlay bakes gamma=1.5 as a frozen constant
        # (game_renderer._LEGACY_SMOKE_GAMMA) — Fire & Heat Beauty B2 P2 DELETED
        # the [smoke] smoke_render_gamma config dial, the gas-medium tau-curve
        # subsumes it.
        self.gamma = gamma

    def update(self, field: np.ndarray) -> None:
        """field: (H, W) float in [0,1]. Pack to RGBA, upload.

        Smoke is drawn as a flat grey DENSITY medium: alpha is density-driven,
        the RGB tint is constant. The old ``light_modulation`` parameter (which
        multiplied the smoke colour by the local light to fake lit-smoke tint)
        is RETIRED — the god-ray glow (``GlowOverlay``, fed the in-scattered
        glow: the sweep's ``light_glow`` since ray-engine-v2 P6b, the render
        march's ``smoke_glow`` before P6c) provides lit-smoke shafts as an additive
        layer, one energy-conserving mechanism with no double-count (ch.03 C16,
        ch.05 §God-rays). Alpha is never modulated by light: smoke as a physical
        medium is always there; the glow overlay adds the colour it scatters.
        """
        # Pack as PREMULTIPLIED alpha so the overlay can be drawn with
        # BLEND_ALPHA_PREMULTIPLY (Porter-Duff "over"). Raylib's default
        # BLEND_ALPHA uses SRC_ALPHA for BOTH the colour AND alpha
        # channels, which means drawing semi-transparent smoke over an
        # opaque ship pixel REDUCES the destination alpha (e.g. ship
        # alpha 1.0 + smoke alpha 0.5 -> result alpha 0.75 instead of
        # 1.0). When the world RT is then blitted to screen, the lower
        # alpha lets the screen-fixed background bleed through what
        # should be opaque ship pixels — exactly the "galaxies through
        # the ship after a grenade" bug. PREMUL gets the correct
        # Porter-Duff alpha math: ship alpha 1.0 stays at 1.0.
        v = np.clip(field, 0.0, 1.0)
        # smoke^gamma render contrast (ch.05 §6.1 step 5): remap the rendered
        # density through a power curve so thin smoke crushes toward transparent
        # and wispy edges sharpen — a filmic look that kills flat fog. gamma=1.0
        # is identity (skip the pow for speed; also what the base class / fire
        # use). This is render-time only; gmap.smoke is never modified.
        if self.gamma != 1.0:
            v = v ** self.gamma
        alpha = v * self.max_alpha   # uint range, 0..255
        a_norm = alpha / 255.0       # 0..1 multiplier for premultiplication
        r = self.tint_r * a_norm
        g = self.tint_g * a_norm
        b = self.tint_b * a_norm
        self.packed[..., 0] = r.astype(np.uint8)
        self.packed[..., 1] = g.astype(np.uint8)
        self.packed[..., 2] = b.astype(np.uint8)
        self.packed[..., 3] = alpha.astype(np.uint8)
        core.update_rgba_texture(self.tex, self.packed)

    def draw(self, dst_x: int, dst_y: int, dst_w: int, dst_h: int) -> None:
        src = rl.Rectangle(0, 0, float(self.w), float(self.h))
        dst = rl.Rectangle(float(dst_x), float(dst_y), float(dst_w), float(dst_h))
        rl.draw_texture_pro(self.tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)


def _begin_additive_rgb_only_blend() -> None:
    """Begin an additive blend for light/glow passes — RGB only, dstA untouched.

    RULE: additive passes must not write destination alpha. Raylib's
    BLEND_ADDITIVE is ``glBlendFunc(SRC_ALPHA, ONE)`` for ALL channels, so it
    also adds ``srcA*srcA`` to dstA — one full-RT additive draw with packed
    alpha saturates the world RT's alpha to 255 everywhere, including the
    vacuum tiles the lighting shader deliberately left transparent. The
    premultiplied RT->screen blit (``world_composite.blit_to_screen``,
    ``out = rt.rgb + bg*(1 - rt.a)``) then multiplies the backdrop by zero:
    the galaxy disappears and vacuum renders opaque black.

    Fix: separate blend factors — RGB additive ``(SRC_ALPHA, ONE)``, alpha
    untouched ``(ZERO, ONE)`` so ``dstA' = dstA`` exactly. The RGB result is
    identical to BLEND_ADDITIVE. Pair with ``rl.end_blend_mode()``.
    """
    rl.rl_set_blend_factors_separate(
        rl.RL_SRC_ALPHA, rl.RL_ONE,   # RGB: classic additive (src*srcA + dst)
        rl.RL_ZERO, rl.RL_ONE,        # A:   dstA' = dstA — never written
        rl.RL_FUNC_ADD, rl.RL_FUNC_ADD)
    rl.begin_blend_mode(rl.BlendMode.BLEND_CUSTOM_SEPARATE)


class HeatFieldOverlay:
    """Emissive temperature overlay — physical black-body glow over a Q16.16 field.

    Fire & Heat Beauty arc, B1 (design
    docs/fire_b1_blackbody_fire_lights_design_2026-07-21.md §2). Render-time view
    of ``gmap.temperature`` (Q16.16 int32 ΔT above ambient, 0 = cold); it NEVER
    mutates the field. This is the tier-1, ray-free glow: the whole plume glows
    for a LUT read per tile, no rays.

    It replaces the old 5-stop hand-tuned ramp AND the ``temp_display_max`` knob
    with the physical black-body primitive (:class:`renderer.blackbody.
    BlackbodyRamp`): temperature -> pseudo-Kelvin -> normalized chroma * a
    separate T⁴ HDR intensity. The HDR emissive (``chroma * intensity``, which
    runs past 1) is shaped by the ACES curve -- the SAME curve the world
    resolve uses -- so a bright warm core reads white-hot instead of clipping
    to flat orange, and the colour reads the physics by eye (a fire yellows
    with O2, reddens as it starves).

    LEGACY HEAT LAYER, display space (graphics G0, the one named exception
    to the one-tone-map rule; see ``blackbody.pack_emissive_rgba``): its look
    was tuned as a display colour added on screen, so it is drawn AFTER the
    world resolve, into the display RT, with its pre-G0 packing. G3 F3 tried
    to retire the exception and MEASURED the move failing (design §13.8, the
    hue gate, ``tools/fire_look_gate.py``: +51..+142 % luminance on the heat-
    only blasts, a 17 deg hue shift, a hard blocky rim), so it stays -- the
    fallback -- until its own retirement patch (from F3's first commit).

    This class owns the layer's DATA only: ``tex`` (RGBA8, the display
    colour split into a peak-normalised hue (RGB) and a brightness weight
    (A), so the additive ``rgb * alpha`` reconstructs it), ``body_tex`` (the
    share below the smoke) and ``rect`` (the tiles it glows on, dilated by
    one, or None). It is DRAWN by THE fire's one shader
    (``shaders/fire_glow.fs`` stage "heat", ``FireVFXPass.draw_glow``):
    additively, RGB only (the destination alpha is never written), through
    the smoke over it (G2's glow-through) and under the flame's cover (G3:
    ``glow * (1 - eps)``, the flame covers its burning tile -- nothing is
    counted twice). Cold tiles stay invisible.
    """

    def __init__(self, grid_h: int, grid_w: int, ramp: BlackbodyRamp,
                 max_alpha: int = 220):
        self.h = grid_h
        self.w = grid_w
        # The shared black-body primitive (built once in the renderer from
        # [render.blackbody]). Render-only tuning; never touches the sim.
        self.ramp = ramp
        self.max_alpha = int(max_alpha)
        # RGBA8 display colour: the legacy heat layer is display-referred.
        self.tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        self.packed = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
        self.rect = None                  # the glowing tiles' box (G3 F3)
        # G2: the share of each tile's glow lying BELOW the smoke
        # (gas_relief.glow_below_fraction), R channel; set_below_fraction
        self.body_tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        rl.set_texture_filter(self.body_tex,
                              rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        self._below_img = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
        self._below_img[..., 3] = 255

    def set_below_fraction(self, frac: np.ndarray) -> None:
        """Graphics G2: per tile, the share of the glow that lies below the
        smoke, which the glow-through dims (``gas_relief.
        glow_below_fraction``). Uploaded every sim tick (48 x 32 bytes)."""
        self._below_img[..., 0] = np.clip(
            np.asarray(frac, dtype=np.float32) * 255.0 + 0.5, 0, 255
        ).astype(np.uint8)
        core.update_rgba_texture(self.body_tex, self._below_img)

    @property
    def brightness(self) -> np.ndarray:
        """(H, W) float32 in [0, 1]: the layer's display brightness per tile
        (the packed weight over its ceiling) -- how strongly each tile glows
        (graphics G2: the smoke over it is lit from below by this)."""
        return self.packed[..., 3].astype(np.float32) / float(max(self.max_alpha, 1))

    def update(self, temperature: np.ndarray,
               intensity_mod: Optional[np.ndarray] = None) -> None:
        """temperature: (H, W) Q16.16 int32 — ΔT above ambient (0 = cold).

        Map through the black-body ramp, shape the HDR emissive with the
        ACES curve into its display colour, and pack to an additive RGBA8
        texture (the legacy heat layer, display space). RENDER-ONLY: `temperature` is read,
        never written. The packing math lives in renderer.blackbody
        (pyray-free / headless-testable); here we only upload it.

        ``intensity_mod`` (Fire & Heat Beauty B2 P4): an optional (H, W)
        dirty-Planck speckle field (renderer.speckle) that mottles the
        black-body intensity at pack time. ``None`` (the B1 default) packs
        byte-for-byte as before — the ``off`` speckle mode + the game/harness
        that never built a SpeckleField both leave this None.
        """
        self.packed = pack_emissive_rgba(self.ramp, temperature, self.max_alpha,
                                         intensity_mod=intensity_mod)
        core.update_rgba_texture(self.tex, self.packed)
        self.rect = _glow_box(self.packed[..., 3] > 0)

    def unload(self) -> None:
        rl.unload_texture(self.tex)
        rl.unload_texture(self.body_tex)


def _glow_box(mask: np.ndarray):
    """The bounding box (x0, y0, x1, y1; tiles, end-exclusive) of ``mask``
    dilated by one tile (the texture is bilinear: a glow spills half a tile)
    and clipped to the grid -- the only area the heat stage draws -- or None
    when nothing glows."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    h, w = mask.shape
    return (max(0, int(cols[0]) - 1), max(0, int(rows[0]) - 1),
            min(w, int(cols[-1]) + 2), min(h, int(rows[-1]) + 2))


class GlowOverlay:
    """God-ray / lit-smoke glow overlay (ch.05 §God-rays).

    Draws the in-scattered glow field (the radiation sweep's ``light_glow``,
    through the light accessor and LightingPass -- the render march's
    ``smoke_glow`` until P6c) as an ADDITIVE volumetric shaft. This supersedes
    the retired ``light_modulation`` smoke surface-tint: a red beam through
    smoke casts a red shaft, energy-conserving by construction (the energy is
    exactly what the smoke removed from the ray). The additive draw must raise
    RGB *only*: plain BLEND_ADDITIVE also writes destination alpha, and with
    this texture's packed alpha=255 it saturated the world RT's alpha across
    vacuum tiles, turning the backdrop opaque black under the premultiplied
    blit — so draw() uses RGB-only separate blend factors instead (see
    ``_begin_additive_rgb_only_blend``). Unlike the alpha-blended smoke it is
    NOT premultiplied (ch.05 §Blend discipline). Drawn before units so they
    occlude it in screen space; the sweep puts no glow on non-gas cells and
    none past opaque tiles, so shafts already terminate at walls.
    """

    def __init__(self, grid_h: int, grid_w: int, gain: float = 1.0):
        self.h = grid_h
        self.w = grid_w
        # `gain` scales the glow brightness before the 0..255 quantize — a
        # render-only knob (the deposit is energy-conserving and typically dim).
        self.gain = gain
        self.tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        self.packed = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)

    def update(self, glow: np.ndarray) -> None:
        """glow: (H, W, 3) float — the in-scattered god-ray field.

        Tone-map by simple clamp (ACES is the final-slice job) and pack into
        an RGBA texture with full alpha. With the RGB factors (SRC_ALPHA, ONE)
        full alpha passes the RGB straight through as an additive
        contribution; draw() keeps the alpha channel out of the blend
        entirely (RGB-only additive — dstA must never be written, see
        ``_begin_additive_rgb_only_blend``).
        """
        glow = np.clip(glow * self.gain, 0.0, 1.0)
        self.packed[..., 0] = (glow[..., 0] * 255.0).astype(np.uint8)
        self.packed[..., 1] = (glow[..., 1] * 255.0).astype(np.uint8)
        self.packed[..., 2] = (glow[..., 2] * 255.0).astype(np.uint8)
        self.packed[..., 3] = 255
        core.update_rgba_texture(self.tex, self.packed)

    def draw(self, dst_x: int, dst_y: int, dst_w: int, dst_h: int) -> None:
        # Additive passes must not write destination alpha (BLEND_ADDITIVE
        # touches dstA; this erases the backdrop under the premultiplied
        # blit). This texture packs alpha=255 on EVERY texel, so a plain
        # BLEND_ADDITIVE here saturated the whole world RT's alpha — vacuum
        # tiles went opaque black. See _begin_additive_rgb_only_blend.
        _begin_additive_rgb_only_blend()
        src = rl.Rectangle(0, 0, float(self.w), float(self.h))
        dst = rl.Rectangle(float(dst_x), float(dst_y), float(dst_w), float(dst_h))
        rl.draw_texture_pro(self.tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_blend_mode()


# ----------------------------------------------------------------------------
# Units, orders, HUD
# ----------------------------------------------------------------------------

def draw_unit(x_tile: float, y_tile: float, world_px_per_tile: float,
              color, label: str = "", radius_tiles: float = 1.5,
              footprint_tiles: int = 3,
              sprite: Optional[rl.Texture] = None,
              light_intensity: float = 1.0,
              is_prone: bool = False) -> None:
    """Draw a unit on its footprint, in world-pixel coordinates.

    x_tile, y_tile = top-left of the unit's footprint in world-tile coords.
    world_px_per_tile = how many world pixels per tile (set by WorldComposite).
    footprint_tiles = side length of the unit's tile footprint (3 for marines).
    radius_tiles = visual radius in tile units (used only for the circle fallback).
    sprite = optional Texture; if provided, draws sprite scaled to the footprint
             instead of the circle placeholder.
    light_intensity = scalar 0..1+ (clamped) used to tint the sprite. 1.0 = full
             brightness; 0.0 = black silhouette. Lets the unit respond to the
             room's lighting without needing a per-sprite normal map yet.
    is_prone = the unit's composed ``is_prone`` behavior flag (mechanics/06 §4,
             KNOCKED_DOWN et al). Minimal v1 visual (P4): the sprite is drawn
             rotated 90° about its footprint centre — lying on its side — so
             who is knocked down reads at a glance; the circle fallback
             flattens into an ellipse. Render-only; the hitbox/stamp/exposure
             implications of prone are separate later wiring.
    """
    x_wpx = tile_to_world_px(x_tile, world_px_per_tile)
    y_wpx = tile_to_world_px(y_tile, world_px_per_tile)
    size_wpx = footprint_tiles * world_px_per_tile

    if sprite is not None:
        # Tint by local light. Clamp to [0, 1] so we never overdrive past white.
        L = max(0.0, min(1.0, float(light_intensity)))
        c = int(L * 255)
        tint = rl.Color(c, c, c, 255)
        src = rl.Rectangle(0.0, 0.0, float(sprite.width), float(sprite.height))
        if is_prone:
            # Rotate about the footprint centre: draw_texture_pro spins the
            # dest rect around `origin`, which is expressed in dest space
            # relative to (dst.x, dst.y) — so anchor dst at the centre point
            # and set origin to the half-size. Same footprint, lying sideways.
            half_wpx = size_wpx * 0.5
            dst = rl.Rectangle(x_wpx + half_wpx, y_wpx + half_wpx,
                               size_wpx, size_wpx)
            rl.draw_texture_pro(sprite, src, dst,
                                rl.Vector2(half_wpx, half_wpx), 90.0, tint)
        else:
            dst = rl.Rectangle(x_wpx, y_wpx, size_wpx, size_wpx)
            rl.draw_texture_pro(sprite, src, dst, rl.Vector2(0.0, 0.0), 0.0,
                                tint)
    else:
        # Circle fallback — also used when sprite failed to load.
        half = footprint_tiles * 0.5
        cx_wpx = x_wpx + half * world_px_per_tile
        cy_wpx = y_wpx + half * world_px_per_tile
        r_wpx  = radius_tiles * world_px_per_tile
        if is_prone:
            # Flattened ellipse — the lying-down silhouette of the circle.
            rl.draw_ellipse(int(cx_wpx), int(cy_wpx),
                            r_wpx, r_wpx * 0.45, rl.Color(*color))
        else:
            rl.draw_circle(int(cx_wpx), int(cy_wpx), r_wpx, rl.Color(*color))

    if label:
        r_wpx = radius_tiles * world_px_per_tile
        cx_wpx = x_wpx + (footprint_tiles * 0.5) * world_px_per_tile
        cy_wpx = y_wpx + (footprint_tiles * 0.5) * world_px_per_tile
        rl.draw_text(label, int(cx_wpx - r_wpx),
                     int(cy_wpx - r_wpx - 14), 12, rl.WHITE)


def draw_waypoint_line(p1_tile, p2_tile, world_px_per_tile: float,
                       color=(60, 200, 255, 200), unit_footprint_tiles: int = 3
                       ) -> None:
    """Draw a line between two waypoints in world-pixel coordinates.
    p1, p2 are (x_tile, y_tile). The line is drawn through the centers of
    the unit's footprint at each waypoint."""
    half = unit_footprint_tiles * 0.5
    x1_wpx = tile_to_world_px(p1_tile[0] + half, world_px_per_tile)
    y1_wpx = tile_to_world_px(p1_tile[1] + half, world_px_per_tile)
    x2_wpx = tile_to_world_px(p2_tile[0] + half, world_px_per_tile)
    y2_wpx = tile_to_world_px(p2_tile[1] + half, world_px_per_tile)
    rl.draw_line_ex(rl.Vector2(x1_wpx, y1_wpx),
                    rl.Vector2(x2_wpx, y2_wpx),
                    2.0, rl.Color(*color))


def draw_grid(grid_w_tile: int, grid_h_tile: int, world_px_per_tile: float,
              color=(80, 80, 100, 60), step: int = 3) -> None:
    """Faint grid overlay at every `step` tiles, drawn in world pixels."""
    color_obj = rl.Color(*color)
    px_w = grid_w_tile * world_px_per_tile
    px_h = grid_h_tile * world_px_per_tile
    for x_tile in range(0, grid_w_tile + 1, step):
        xp = tile_to_world_px(x_tile, world_px_per_tile)
        rl.draw_line_ex(rl.Vector2(xp, 0), rl.Vector2(xp, px_h), 1.0, color_obj)
    for y_tile in range(0, grid_h_tile + 1, step):
        yp = tile_to_world_px(y_tile, world_px_per_tile)
        rl.draw_line_ex(rl.Vector2(0, yp), rl.Vector2(px_w, yp), 1.0, color_obj)


def draw_text(text: str, x: int, y: int, size: int = 16, color=(220, 220, 220, 255)) -> None:
    rl.draw_text(text, x, y, size, rl.Color(*color))


def draw_panel_background(x: int, y: int, w: int, h: int, color=(20, 20, 28, 240)) -> None:
    rl.draw_rectangle(x, y, w, h, rl.Color(*color))
    rl.draw_line_ex(rl.Vector2(x, y), rl.Vector2(x, y + h), 2.0, rl.Color(120, 120, 140, 255))


__all__ = [
    "FieldOverlay", "GlowOverlay", "HeatFieldOverlay",
    "draw_unit", "draw_waypoint_line",
    "draw_grid", "draw_text", "draw_panel_background",
]
