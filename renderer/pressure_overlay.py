"""Pressure colormap overlay for the world RT.

Shared by the main game (``GameRenderer``) and the lighting demo tool
(``tools/lighting_demo.py``). Renders ``gmap.atmosphere + gmap.wave_p``
as an RGBA colour map using the stops in
``config.toml [rendering] pressure_stops``, masking out walls and vacuum.

Drawn INTO the world RT (after smoke/fire, before units) so the camera
transform applies. Uses BLEND_ALPHA_PREMULTIPLY for correct Porter-Duff
composition with the existing premultiplied overlays.

Originally ported from ``game.py:2095-2149`` into the lighting demo;
hoisted here so explosions look dramatic in the main game too.
"""
from __future__ import annotations

import numpy as np
import pyray as rl

from config import CFG
from . import core


def _load_pressure_stops() -> np.ndarray:
    """Load colormap from ``CFG.rendering.pressure_stops`` with a fallback."""
    raw = getattr(CFG.rendering, "pressure_stops", None)
    if raw is None:
        raw = [
            [0.0,    0,   0,   0,   0],
            [3.3,  255, 255, 255,   5],
            [6.0,  255, 255, 255,  15],
            [7.0,  200,  50,  30, 120],
            [8.0,  255, 140,  30, 180],
            [9.0,  255, 220,  80, 220],
            [10.0, 255, 255, 255, 255],
        ]
    return np.array(raw, dtype=np.float32)


def _default_pressure_scale() -> float:
    return float(getattr(CFG.rendering, "pressure_scale", 2.0))


class PressureOverlay:
    """Owns the per-frame pressure → colour texture for the world RT."""

    def __init__(self, grid_h: int, grid_w: int):
        self.h = grid_h
        self.w = grid_w
        self.stops = _load_pressure_stops()
        self.pressure_scale = _default_pressure_scale()
        self.tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        self._rgba = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
        # Point filter — colour map is per-tile, smoothing makes it look
        # like a soft blob rather than the pressure-cell texture it is.
        rl.set_texture_filter(self.tex, rl.TextureFilter.TEXTURE_FILTER_POINT)

    def update(self, gmap) -> None:
        """Compute and upload the pressure colour map for the current state."""
        # S2a/S2c render FLOAT BRIDGE: gmap.wave_p (S2a) AND gmap.atmosphere (S2c)
        # are now int32 Q16.16 — DEQUANTIZE BOTH to their PRE-MIGRATION real scale
        # before summing, so the pressure colour map (the explosion viz) reads the
        # original float pressure the colour stops were tuned against. Without this
        # the raw int32 counts (~65536× too large) saturate the top colour stop and
        # paint the whole frame opaque (the bug: wrong explosion colours AND smoke
        # hidden under a saturated overlay). Render-only; the int fields are the
        # synced source of truth.
        from simulation import atmosphere_fixed, wave_fixed
        total = (atmosphere_fixed.dequantize_f32(gmap.atmosphere)
                 + wave_fixed.dequantize_f32(gmap.wave_p))
        if self.pressure_scale > 0:
            p = 1.0 + (total - 1.0) * (10.0 / self.pressure_scale)
        else:
            p = total.copy()

        rgba = self._rgba
        rgba.fill(0)
        stops = self.stops
        n = len(stops)
        # Linear interp between adjacent stops.
        for i in range(n - 1):
            lo, hi = stops[i, 0], stops[i + 1, 0]
            mask = (p >= lo) & (p < hi)
            if not np.any(mask):
                continue
            t = np.clip((p[mask] - lo) / (hi - lo + 1e-9), 0.0, 1.0)
            for ch in range(4):
                rgba[mask, ch] = (
                    stops[i, ch + 1]
                    + t * (stops[i + 1, ch + 1] - stops[i, ch + 1])
                ).astype(np.uint8)
        # Anything above the last stop clamps to its colour.
        mask_last = p >= stops[-1, 0]
        if np.any(mask_last):
            for ch in range(4):
                rgba[mask_last, ch] = int(stops[-1, ch + 1])

        # Pressure is only meaningful in air: hide on walls and vacuum.
        solid = gmap.solid | gmap.is_vacuum
        rgba[solid] = 0

        # Pre-multiply alpha so the draw can use BLEND_ALPHA_PREMULTIPLY
        # and not corrupt the destination alpha (same Porter-Duff fix as
        # smoke — see renderer/overlays.py:FieldOverlay.update).
        a = rgba[..., 3:4].astype(np.float32) / 255.0
        rgba[..., 0:3] = (rgba[..., 0:3].astype(np.float32) * a).astype(np.uint8)

        core.update_rgba_texture(self.tex, rgba)

    def draw_into_world_rt(self, world_px_w: int, world_px_h: int) -> None:
        """Draw the colour map across the world RT. Caller has already
        called BeginTextureMode on the world RT."""
        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        src = rl.Rectangle(0, 0, float(self.w), float(self.h))
        dst = rl.Rectangle(0, 0, float(world_px_w), float(world_px_h))
        rl.draw_texture_pro(self.tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_blend_mode()

    def unload(self) -> None:
        rl.unload_texture(self.tex)


__all__ = ["PressureOverlay"]
