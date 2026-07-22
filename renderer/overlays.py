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
        # the base class so FireOverlay and any other field is untouched).
        # The LEGACY smoke overlay bakes gamma=1.5 as a frozen constant
        # (game_renderer._LEGACY_SMOKE_GAMMA) — Fire & Heat Beauty B2 P2 DELETED
        # the [smoke] smoke_render_gamma config dial, the gas-medium tau-curve
        # subsumes it; WaterFieldOverlay uses gamma=0.5 to LIFT thin films.
        self.gamma = gamma

    def update(self, field: np.ndarray) -> None:
        """field: (H, W) float in [0,1]. Pack to RGBA, upload.

        Smoke is drawn as a flat grey DENSITY medium: alpha is density-driven,
        the RGB tint is constant. The old ``light_modulation`` parameter (which
        multiplied the smoke colour by the local light to fake lit-smoke tint)
        is RETIRED — the god-ray glow (``GlowOverlay``, fed by the ray march's
        ``smoke_glow`` output) now provides lit-smoke shafts as an additive
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


class FireOverlay(FieldOverlay):
    """Fire-specific: orange/yellow tint, additive blend."""

    def __init__(self, grid_h: int, grid_w: int):
        super().__init__(grid_h, grid_w, tint=(255, 140, 30), max_alpha=220)

    def update(self, fire: np.ndarray) -> None:
        # Slight color modulation by intensity (hotter = more white).
        # Fire is its own light source.
        v = np.clip(fire, 0.0, 1.0)
        self.packed[..., 0] = 255
        self.packed[..., 1] = (140 + (255 - 140) * v * 0.5).astype(np.uint8)
        self.packed[..., 2] = (30 + (180 - 30) * v * 0.5).astype(np.uint8)
        self.packed[..., 3] = (v * self.max_alpha).astype(np.uint8)
        core.update_rgba_texture(self.tex, self.packed)

    def draw(self, dst_x: int, dst_y: int, dst_w: int, dst_h: int) -> None:
        rl.begin_blend_mode(rl.BlendMode.BLEND_ADDITIVE)
        super().draw(dst_x, dst_y, dst_w, dst_h)
        rl.end_blend_mode()


class WaterFieldOverlay(FieldOverlay):
    """Water overlay v2 — depth-blue tint + ripple shading + foam + ambient sines.

    Water W6b (docs/water_implementation_plan.md; canon engine/07 §6). Still
    the TASTEFUL PLACEHOLDER, not the shipped look — the dedicated water-optics
    research pass (Fresnel, refraction, caustics) is a separate canon §6 item.
    Render-time view of the sim's water fields; it NEVER mutates any of them.
    Depth is normalised by ``depth_display_max`` (metres mapping to full tint)
    and packed PREMULTIPLIED, drawn with BLEND_ALPHA_PREMULTIPLY like smoke.

    Three render effects on top of the W2b depth tint, all CPU-side numpy at
    overlay resolution (everything stays O(h·w); no per-pixel Python loops):

    1. **Ripple shading** — a cheap directional pseudo-normal from
       d(ripple)/dx: positive slope brightens the tint, negative darkens, gain
       ``ripple_shade`` (fraction of the tint at the saturation slope
       ``_SLOPE_REF``). Brightened channels are clipped before the
       premultiply so the composite stays premultiplied-valid.
    2. **Foam** — where ``|grad ripple|`` exceeds ``foam_thresh`` (m/tile) the
       colour blends toward white and alpha lifts toward opaque froth
       (whitecaps at steep fronts, canon §6). A fresh splash (slope ~0.075
       m/tile from a 0.05 m amplitude over ~2 tiles) foams; settled water
       (slope ~0) never does. Advancing wet/dry FRONTS also foam when the
       local |flow| exceeds ``_FRONT_SPEED``.
    3. **Ambient sines** — a precomputed 3-wave sine lattice over (x, y, t)
       gives standing water idle shimmer. Canon §6 modulation rule: amplitude
       = ``ambient_base`` + local ripple ENERGY (|ripple| + |ripple_v|,
       4-neighbour smoothed), so the surface gets agitated near events and
       calms down after. ``t`` is wall-clock render time — animation only,
       deterministic-irrelevant (render-only by the locked §6 rule).

    Knobs bind from ``[display]`` (water_ripple_shade / water_foam_thresh /
    water_ambient_base / water_display_max) via getattr-defaults in
    GameRenderer. RENDER-ONLY and RESTART-BOUND: read once at renderer
    construction; Ctrl+R re-reads config.toml but never re-binds overlays
    (the W2b water_display_max precedent).

    gamma = 0.5 LIFTS thin films toward visible — the opposite of smoke's
    wisp-crushing 1.5 — because the point is to SEE where the water went,
    and a spreading pour thins to centimetres fast (0.04 m reads at ~20%
    alpha instead of 4%).

    Zero-water fast path: when ``water_depth`` is all zero AND the last
    upload already wrote all-zero texels, ``update`` returns without packing
    or uploading (a dry ship costs one ``.any()``). The GPU texture starts
    OPAQUE BLACK (core.create_dynamic_rgba_texture), so the first update
    always uploads to clear it.
    """

    # Saturation slope (m/tile): d(ripple)/dx at which the shading hits its
    # full ±ripple_shade swing — the fresh-splash scale (a 0.05 m splash over
    # ~2 tiles ≈ 0.075 m/tile reads fully shaded; gentle swell reads partial).
    _SLOPE_REF = 0.05
    # Ambient amplitude = ambient_base + _AMB_ENERGY_GAIN * energy, capped at
    # _AMB_MAX (energy is the |ripple| + |ripple_v| heuristic in ~metres; a
    # fresh splash saturates the cap, settled water decays back to the base).
    _AMB_ENERGY_GAIN = 2.0
    _AMB_MAX = 0.40
    # Wet/dry front foam: a wet tile with a dry 4-neighbour and local |flow|
    # above _FRONT_SPEED (m/s) gets at least _FRONT_FOAM whitening.
    _FRONT_SPEED = 0.10
    _FRONT_FOAM = 0.6
    # Full foam lifts alpha toward this fraction of max_alpha (opaque froth
    # stays readable even over centimetre films).
    _FOAM_ALPHA_FRAC = 0.85
    # Ambient sine lattice: (kx, ky, omega) per wave — rad/tile spatial
    # frequencies (wavelengths ~10 tiles) and rad/s temporal (slow idle,
    # periods 3–7 s). Three directions so the sum never reads as stripes.
    _WAVES = ((0.55, 0.25, 1.3),
              (-0.35, 0.45, 0.9),
              (0.20, -0.60, 1.9))

    def __init__(self, grid_h: int, grid_w: int,
                 depth_display_max: float = 1.0,
                 ripple_shade: float = 0.35,
                 foam_thresh: float = 0.03,
                 ambient_base: float = 0.06):
        super().__init__(grid_h, grid_w, tint=(40, 110, 230),
                         max_alpha=200, gamma=0.5)
        # Depth (m) that maps to the top of the tint ramp. Render-only knob;
        # depths above it just clamp to full tint (FieldOverlay clips to 1).
        self.depth_display_max = float(depth_display_max)
        self.ripple_shade = float(ripple_shade)
        self.foam_thresh = float(foam_thresh)
        self.ambient_base = float(ambient_base)
        # Precomputed sine-lattice spatial phases, one (h, w) float32 array
        # per wave — per frame only sin(phase + omega*t) remains.
        ys = np.arange(grid_h, dtype=np.float32)[:, None]
        xs = np.arange(grid_w, dtype=np.float32)[None, :]
        self._phases = [np.asarray(kx * xs + ky * ys, dtype=np.float32)
                        for (kx, ky, _w) in self._WAVES]
        # True once the last upload wrote all-zero texels (see class doc).
        self._tex_blank = False

    def update(self, water_depth: np.ndarray,
               ripple: Optional[np.ndarray] = None,
               ripple_v: Optional[np.ndarray] = None,
               flow_vx: Optional[np.ndarray] = None,
               flow_vy: Optional[np.ndarray] = None,
               t: float = 0.0) -> None:
        """Pack water_depth (+ optional ripple fields) to RGBA and upload.

        water_depth: (H, W) float metres. ripple (m) / ripple_v (m/s) enable
        the v2 shading/foam/ambient path; ``ripple=None`` falls back to the
        plain W2b depth tint (bit-identical to the old overlay). flow_vx /
        flow_vy (m/s) optionally add wet/dry-front foam. ``t`` is the render
        animation clock in seconds (any epoch).

        RENDER-ONLY: every input is read, never written.
        """
        has_water = bool(water_depth.any())
        if not has_water and self._tex_blank:
            return  # dry ship + texture already clear: skip pack AND upload
        if ripple is None:
            # Legacy W2b path — plain depth-blue tint via the base pack.
            super().update(water_depth / max(self.depth_display_max, 1e-6))
            self._tex_blank = not has_water
            return

        # --- base depth tint (the W2b ramp) -----------------------------
        v = np.clip(water_depth / max(self.depth_display_max, 1e-6), 0.0, 1.0)
        v = v ** self.gamma                       # 0.5: lift thin films
        alpha = v * self.max_alpha                # float (h, w), 0..max_alpha
        wet = water_depth > 0.0

        # --- 1. ripple shading: directional pseudo-normal from d/dx -----
        gx = np.gradient(ripple, axis=1)          # m per tile
        gy = np.gradient(ripple, axis=0)
        shade = self.ripple_shade * np.clip(gx / self._SLOPE_REF, -1.0, 1.0)

        # --- 3. ambient sines, amplitude = base + local ripple energy ---
        energy = np.abs(ripple) + np.abs(ripple_v)
        # Cheap 4-neighbour smooth (np.roll wraps at the border; borders are
        # walls/vacuum -> depth 0 -> invisible texels, so the wrap is moot).
        energy = (4.0 * energy
                  + np.roll(energy, 1, 0) + np.roll(energy, -1, 0)
                  + np.roll(energy, 1, 1) + np.roll(energy, -1, 1)) * 0.125
        amp = np.clip(self.ambient_base + self._AMB_ENERGY_GAIN * energy,
                      0.0, self._AMB_MAX)
        sines = np.sin(self._phases[0] + self._WAVES[0][2] * t)
        sines += np.sin(self._phases[1] + self._WAVES[1][2] * t)
        sines += np.sin(self._phases[2] + self._WAVES[2][2] * t)
        # Brightness factor >= 0; per-channel 0..255 clip below keeps the
        # premultiply valid (channel <= alpha) however hard this brightens.
        bright = np.maximum(1.0 + shade + amp * (sines / 3.0), 0.0)

        # --- 2. foam: whitecaps where the surface is steep ---------------
        mag = np.hypot(gx, gy)
        foam = np.clip(mag / max(self.foam_thresh, 1e-9) - 1.0, 0.0, 1.0)
        if flow_vx is not None and flow_vy is not None:
            # Advancing wet/dry front: wet tile, dry 4-neighbour, moving.
            dry = ~wet
            edge = (np.roll(dry, 1, 0) | np.roll(dry, -1, 0) |
                    np.roll(dry, 1, 1) | np.roll(dry, -1, 1))
            fast = (flow_vx * flow_vx + flow_vy * flow_vy
                    ) > self._FRONT_SPEED ** 2
            foam = np.maximum(foam, np.where(wet & edge & fast,
                                             self._FRONT_FOAM, 0.0))
        foam = np.where(wet, foam, 0.0)            # foam only ON water
        # Foam lifts alpha toward opaque froth (dry tiles stay 0: v and foam
        # are both 0 there, so alpha stays 0 and the premultiply zeroes RGB).
        alpha = np.maximum(alpha, foam * (self._FOAM_ALPHA_FRAC
                                          * self.max_alpha))
        a_norm = alpha / 255.0

        # --- compose: shaded tint -> foam blend -> premultiply LAST ------
        inv = 1.0 - foam
        white = 255.0 * foam
        cr = np.clip(self.tint_r * bright, 0.0, 255.0) * inv + white
        cg = np.clip(self.tint_g * bright, 0.0, 255.0) * inv + white
        cb = np.clip(self.tint_b * bright, 0.0, 255.0) * inv + white
        self.packed[..., 0] = (cr * a_norm).astype(np.uint8)
        self.packed[..., 1] = (cg * a_norm).astype(np.uint8)
        self.packed[..., 2] = (cb * a_norm).astype(np.uint8)
        self.packed[..., 3] = alpha.astype(np.uint8)
        core.update_rgba_texture(self.tex, self.packed)
        self._tex_blank = not has_water


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
    runs past 1) is ACES tone-mapped — the SAME curve the lighting shader uses —
    so a bright warm core reads white-hot instead of clipping to flat orange, and
    the colour reads the physics by eye (a fire yellows with O2, reddens as it
    starves).

    Drawn ADDITIVELY: the tone-mapped colour is split into a peak-normalized hue
    (packed RGB) and a brightness weight (packed alpha), so the additive result
    ``rgb * alpha`` reconstructs the tone-mapped emissive. Alpha carries the
    intensity/brightness curve, so cold tiles stay invisible and hot tiles glow.
    draw() uses RGB-only separate blend factors so destination alpha is never
    written (see ``_begin_additive_rgb_only_blend``).
    """

    def __init__(self, grid_h: int, grid_w: int, ramp: BlackbodyRamp,
                 max_alpha: int = 220):
        self.h = grid_h
        self.w = grid_w
        # The shared black-body primitive (built once in the renderer from
        # [render.blackbody]). Render-only tuning; never touches the sim.
        self.ramp = ramp
        self.max_alpha = int(max_alpha)
        self.tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        self.packed = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)

    def update(self, temperature: np.ndarray) -> None:
        """temperature: (H, W) Q16.16 int32 — ΔT above ambient (0 = cold).

        Map through the black-body ramp, ACES tone-map the HDR emissive, and
        pack to an additive RGBA texture. RENDER-ONLY: `temperature` is read,
        never written. The packing math lives in renderer.blackbody
        (pyray-free / headless-testable); here we only upload it.
        """
        self.packed = pack_emissive_rgba(self.ramp, temperature, self.max_alpha)
        core.update_rgba_texture(self.tex, self.packed)

    def draw(self, dst_x: int, dst_y: int, dst_w: int, dst_h: int) -> None:
        # Additive passes must not write destination alpha (BLEND_ADDITIVE
        # touches dstA; this erases the backdrop under the premultiplied
        # blit) — see _begin_additive_rgb_only_blend.
        _begin_additive_rgb_only_blend()
        src = rl.Rectangle(0, 0, float(self.w), float(self.h))
        dst = rl.Rectangle(float(dst_x), float(dst_y), float(dst_w), float(dst_h))
        rl.draw_texture_pro(self.tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_blend_mode()


class GlowOverlay:
    """God-ray / lit-smoke glow overlay (ch.05 §God-rays).

    Draws the ray march's ``smoke_glow`` field — the RGB light the smoke
    *absorbed*, per channel — as an ADDITIVE volumetric shaft. This supersedes
    the retired ``light_modulation`` smoke surface-tint: a red beam through
    smoke casts a red shaft, energy-conserving by construction (the energy is
    exactly what the smoke removed from the ray). The additive draw must raise
    RGB *only*: plain BLEND_ADDITIVE also writes destination alpha, and with
    this texture's packed alpha=255 it saturated the world RT's alpha across
    vacuum tiles, turning the backdrop opaque black under the premultiplied
    blit — so draw() uses RGB-only separate blend factors instead (see
    ``_begin_additive_rgb_only_blend``). Unlike the alpha-blended smoke it is
    NOT premultiplied (ch.05 §Blend discipline). Drawn before units so they
    occlude it in screen space; the march deposits no glow past opaque tiles,
    so shafts already terminate at walls.
    """

    def __init__(self, grid_h: int, grid_w: int, gain: float = 1.0):
        self.h = grid_h
        self.w = grid_w
        # `gain` scales the glow brightness before the 0..255 quantize — a
        # render-only knob (the deposit is energy-conserving and typically dim).
        self.gain = gain
        self.tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        self.packed = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)

    def update(self, smoke_glow: np.ndarray) -> None:
        """smoke_glow: (H, W, 3) float — the absorbed-light god-ray field.

        Tone-map by simple clamp (ACES is the final-slice job) and pack into
        an RGBA texture with full alpha. With the RGB factors (SRC_ALPHA, ONE)
        full alpha passes the RGB straight through as an additive
        contribution; draw() keeps the alpha channel out of the blend
        entirely (RGB-only additive — dstA must never be written, see
        ``_begin_additive_rgb_only_blend``).
        """
        glow = np.clip(smoke_glow * self.gain, 0.0, 1.0)
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
    "FieldOverlay", "FireOverlay", "GlowOverlay", "HeatFieldOverlay",
    "WaterFieldOverlay",
    "draw_unit", "draw_waypoint_line",
    "draw_grid", "draw_text", "draw_panel_background",
]
