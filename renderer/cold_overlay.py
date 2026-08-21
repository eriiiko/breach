"""Cold-tier overlay — a diverging blue ramp for T_rel < 0 (P-W2, arc
`tabs-compression-work`, design docs/tabs_compression_work_design_2026-08-20.md
D-7 / B-F13 / C12).

``renderer.overlays.HeatFieldOverlay`` is additive-emissive (light only adds),
so it structurally cannot show cold — the whole point of D-7's second render
instrument. This module is the minimal cold tier: T_rel < 0 maps through a
small blue colour-stop ramp (the SAME linear-interp-between-stops idiom
``renderer.pressure_overlay._load_pressure_stops`` / ``PressureOverlay.update``
already use — reused, not reinvented) into a premultiplied RGBA texture,
alpha-blended UNDER the additive heat pass on the SAME toggle (T /
``show_temperature``). Cold and hot are mutually exclusive per pixel (T_rel
can't be both negative and positive), so "under" only matters in the sense
that this pass is drawn BEFORE ``HeatFieldOverlay.draw()`` in the frame
(design D-7's phrasing) — there is no actual overlap to arbitrate.

PLACEHOLDER CONSTANTS, EXPLICITLY PROVISIONAL (design D-7): the stop table
below is a first guess at a readable cold ramp, not a tuned look — Erik judges
the look at P-W3 alongside the physics, same as every other provisional
render pass in this codebase (WaterFieldOverlay's "tasteful placeholder"
precedent).

Kelvin-frame note (design D-7, for the brief): this overlay reads game-deg
T_rel directly (the honest sub-ambient number), NOT the canonical render
Kelvin map (K = 293 + 3*T_game) — that map is misleading below ambient
(-96.67 game-deg reads as "3 K" through it, not the physically meaningful
193 K in the EOS's own T_abs frame).

The mapping function (``pack_cold_rgba``) is pyray-free and pure numpy, so it
is headless-unit-testable in isolation, exactly like ``renderer.blackbody.
pack_emissive_rgba`` — see ``tests/test_cold_overlay.py``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Q16.16 temperature scale (materials.TEMP_SCALE / renderer.blackbody.TEMP_SCALE
# / renderer.hover_readout.TEMP_SCALE — the one shared fixed-point domain).
TEMP_SCALE = 65536.0

# Diverging blue ramp stops: (T_rel game-deg, R, G, B, A). T_rel >= 0 (ambient
# and warm) is handled separately (fully transparent) — these stops only cover
# the T_rel < 0 half, ordered from ambient (0, transparent) to deep cold.
# PLACEHOLDER (design D-7): tuned for "visibly cold, not migraine-blue" at a
# glance, not against any reference image.
COLD_STOPS = np.array([
    [0.0,    20,  40,  90,   0],   # ambient boundary: transparent
    [-10.0,  30,  70, 150,  50],   # faint chill (a rarefaction ring)
    [-50.0,  40, 110, 210, 140],   # the ~-96.67 work-clamp figure lands here
    [-150.0, 90, 170, 255, 210],   # deep cold (near the T_MIN floor)
], dtype=np.float32)

# Gas-presence gate for painting cold, as a fraction of ambient bulk N.
# HUMAN-TEST feedback (Erik, 2026-08-21, P-W3 round 1): without it the ramp
# paints every near-vacuum/vented cell — on a space map that is thousands of
# cells and the screen floods blue ("i first thought u filled everything with
# water"). The cold of near-nothing is noise, not information — and the sim
# itself agrees: the EOS trust gate fades compression work to zero below
# n_work_ref/2 = 0.125 and to full trust at n_work_ref = 0.25 of ambient.
# Aligning the overlay with that rule means we only PAINT cold where the
# engine considers the temperature meaningful. 0.25 = full-trust threshold.
COLD_N_MIN_FRAC = 0.25
# Ambient bulk N in the measured (gas_o2 + inert_n2) Q16.16 convention.
N_AMBIENT_RAW = 65536.0


def pack_cold_rgba(temperature: np.ndarray, max_alpha: int = 210,
                   n_bulk: Optional[np.ndarray] = None,
                   n_min_frac: float = COLD_N_MIN_FRAC) -> np.ndarray:
    """temperature: (H, W) Q16.16 int32 -- ΔT above ambient (D-7's frame).
    n_bulk: optional (H, W) Q16.16 bulk gas (gas_o2 + inert_n2); when given,
    cells below ``n_min_frac`` of ambient N are NOT painted (the trust-gate
    alignment above). When None, behaves as before (paint all cold cells).

    Returns a premultiplied-alpha uint8 (H, W, 4) RGBA array: T_rel >= 0
    packs fully transparent (0,0,0,0); T_rel < 0 interpolates linearly
    through ``COLD_STOPS`` (clamped at the deepest stop), matching
    ``PressureOverlay.update``'s stop-interpolation idiom. RENDER-ONLY: every
    field is read, never written.
    """
    t_game = np.asarray(temperature, dtype=np.float64) / TEMP_SCALE
    h, w = t_game.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    cold = t_game < 0.0
    if n_bulk is not None:
        has_gas = np.asarray(n_bulk, dtype=np.float64) \
            >= (n_min_frac * N_AMBIENT_RAW)
        cold &= has_gas
    if np.any(cold):
        # COLD_STOPS[:, 0] runs 0 -> -150 (decreasing); np.interp needs an
        # ASCENDING xp, so flip both the query sign and the table.
        tc = t_game[cold]
        xp = COLD_STOPS[::-1, 0]
        for ch in range(4):
            fp = COLD_STOPS[::-1, ch + 1]
            rgba[cold, ch] = np.interp(tc, xp, fp)

    # Pre-multiply alpha (Porter-Duff "over", the same fix FieldOverlay /
    # PressureOverlay apply — see renderer/overlays.py's header comment).
    a = rgba[..., 3:4] / 255.0
    rgba[..., 0:3] *= a
    return np.clip(rgba, 0.0, 255.0).astype(np.uint8)


class ColdFieldOverlay:
    """Owns the per-frame cold-tier texture — the WaterFieldOverlay pattern
    (a thin pyray-touching shell around the pure ``pack_cold_rgba``)."""

    def __init__(self, grid_h: int, grid_w: int, max_alpha: int = 210):
        from . import core
        self.h = grid_h
        self.w = grid_w
        self.max_alpha = int(max_alpha)
        self.tex = core.create_dynamic_rgba_texture(grid_w, grid_h)

    def update(self, temperature: np.ndarray,
               n_bulk: Optional[np.ndarray] = None) -> None:
        from . import core
        rgba = pack_cold_rgba(temperature, self.max_alpha, n_bulk=n_bulk)
        core.update_rgba_texture(self.tex, rgba)

    def draw(self, dst_x: int, dst_y: int, dst_w: int, dst_h: int) -> None:
        import pyray as rl
        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        src = rl.Rectangle(0, 0, float(self.w), float(self.h))
        dst = rl.Rectangle(float(dst_x), float(dst_y), float(dst_w), float(dst_h))
        rl.draw_texture_pro(self.tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_blend_mode()


__all__ = ["COLD_STOPS", "TEMP_SCALE", "pack_cold_rgba", "ColdFieldOverlay"]
