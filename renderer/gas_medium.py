"""The physical gas-medium render pass — ONE premultiplied layer (Fire & Heat
Beauty arc, B2 P2). Replaces the flat-grey ``smoke_overlay`` + additive
``glow_overlay`` pair with a single premultiplied-over layer built per frame:
alpha = the medium's OCCLUSION (Beer-Lambert extinction over all five trace
gases), RGB = the medium's LIT half (the ray march's inscatter, tone-mapped).

Design: docs/fire_b2_smoke_honesty_design_2026-07-21.md §3. RENDER-ONLY,
determinism-EXEMPT: pure float math on read-only copies of sim fields
(``gmap.gas``, ``gmap.smoke_glow``); it never writes any sim state and never
moves a golden. The core packing is pyray-free (numpy only), so it is
headless-testable without a GL context — the overlay class only owns the GPU
texture + the premult-over draw.

Why premultiplied-over is the whole model
-----------------------------------------
Premult-over IS the volume compositing operator ``C = inscatter + T·bg`` (with
transmittance ``T = 1 - alpha``): raylib's BLEND_ALPHA_PREMULTIPLY computes
``out.rgb = src.rgb + dst.rgb*(1 - src.a)`` and ``out.a = src.a + dst.a*(1 -
src.a)``. Packing ``src.rgb = inscatter`` (the additive lit half, NOT rescaled
by alpha) and ``src.a = alpha`` (occlusion only) makes three properties true by
construction:
  * glow-through-soot double-counting is impossible (there is one inscatter
    buffer, added once);
  * smoke cannot dim its own glow (the glow is added, not attenuated by alpha);
  * unlit smoke in a dark room is a black occluder (inscatter 0, alpha > 0);
    steam in darkness is invisible (alpha ~ 0 because steam barely absorbs, and
    inscatter 0 until a beam lights it) — the beacon/flashlight reveals it.

Single source of scale (critique finding)
------------------------------------------
The ray march already scales absorption by ``smoke_absorb_scale`` (the
Raycaster's shared beam-reach dial, default 1.4). This pass reads THAT as the
shared base scale and applies ``plume_k_scale`` (default 1.0) as a RELATIVE
multiplier on top, so the plume body and its god-rays track by construction
instead of via two independent dials agreeing. The per-gas extinction ``k_s``
is ``mean(absorption[s])`` read from the SAME GasTable optics columns the march
sums (``simulation/gases.py``) — the panchromatic collapse ``beam_absorb_q16``
already uses — so plume body and beam reach can never disagree about what a gas
is.

Credit (repo rule — cite what a file implements):
  - Beer-Lambert transmittance / volume emission-absorption:
    Pharr, Jakob & Humphreys, "Physically Based Rendering" (pbr-book.org),
    ch. "Volume Scattering".
  - Frostbite unified volumetrics (scatter/transmittance factoring):
    Sebastien Hillaire, "Physically Based and Unified Volumetric Rendering in
    Frostbite", SIGGRAPH 2015.
  - Premultiplied-alpha "over" compositing:
    Inigo Quilez, "Premultiplied alpha" (iquilezles.org/articles/premultipliedalpha).
  - ACES filmic tone-map (Narkowicz approximation): reused from
    renderer.blackbody.aces_tonemap (the SAME curve shaders/lighting.fs and the
    B1 HeatFieldOverlay apply — one tone-map across every render layer).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pyray as rl

from . import core
from .blackbody import aces_tonemap
from simulation.gases import N_TRACE_GASES

# Gameplay ("effect") gases that receive the non-physical emissive legibility
# floor when it is raised (design §3): poison must read sickly-green and teargas
# pale even in the dark, so a player can tell a lethal cloud from fire-smoke.
# Keyed off the GasTable ``effect`` tag so it stays data-driven (poison ->
# "damage_over_time", teargas -> "area_denial").
_EFFECT_GAS_TAGS = frozenset(("damage_over_time", "area_denial"))


def gas_medium_layer(
    gas_density: np.ndarray,
    k_s: np.ndarray,
    *,
    base_absorb_scale: float,
    plume_k_scale: float,
    tau_curve_a: float,
    tau_curve_b: float,
    smoke_glow: np.ndarray,
    glow_gain: float,
    effect_gas_floor: float = 0.0,
    effect_gas_mask: Optional[np.ndarray] = None,
    scatter_albedo: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Core of the pass (pyray-free): -> (rgb, alpha), both float32.

    Parameters
    ----------
    gas_density : (S, H, W) float — the S trace-gas density planes (dequantized
        real densities). Only the trace gases carry optics; the bulk o2/n2 pair
        is excluded by the caller (slice ``gmap.gas[:N_TRACE_GASES]``).
    k_s : (S,) float — per-gas mean extinction = ``mean`` over the absorption
        RGB triple (the panchromatic collapse the beam optics use). Read from
        the GasTable, so plume body and ray march share ONE optical identity.
    base_absorb_scale : the Raycaster's ``smoke_absorb_scale`` — the SHARED base
        scale the ray march already applies (single source of scale).
    plume_k_scale : RELATIVE multiplier on the base (design default 1.0).
    tau_curve_a, tau_curve_b : artistic remap IN TAU-SPACE (never on alpha):
        ``tau' = a * tau**b`` (defaults 1/1 = honest). ``b > 1`` steepens edges.
    smoke_glow : (H, W, 3) float — the ray march's inscatter buffer. It ALREADY
        carries the per-gas ``scatter_albedo`` colour summed density-weighted and
        multiplied by the local light, i.e. the species' LIT identity (steam
        near-white, soot barely). Read, never written.
    glow_gain : render exposure on the inscatter before the tone-map.
    effect_gas_floor : NON-PHYSICAL emissive legibility floor for gameplay gases
        (design override; default 0.0 = honest/off). When > 0 the HUD flags it.
    effect_gas_mask : (S,) bool — which gases receive the floor.
    scatter_albedo : (S, 3) float — per-gas albedo (the floor's hue).

    Returns
    -------
    rgb : (H, W, 3) float32 in [0,1] — the premultiplied additive lit half.
    alpha : (H, W) float32 in [0,1] — the occlusion (1 - transmittance).
    """
    gas_density = np.asarray(gas_density, dtype=np.float32)
    k_s = np.asarray(k_s, dtype=np.float32)

    # --- optical depth (panchromatic), sharing the ray march's base scale ----
    # tau = base_absorb_scale * plume_k_scale * Σ_s ( k_s · ρ_s )
    # tensordot over the gas axis is the density-weighted sum the march does per
    # channel, collapsed to the mean-extinction single channel. Done in float64
    # so `1 - exp(-tau)` below has no catastrophic cancellation at tiny tau (the
    # thin-smoke limit); the uint8 pack is byte-identical to float32 for any
    # visible smoke, this only keeps the linearization accurate.
    weighted = np.tensordot(k_s.astype(np.float64),
                            gas_density.astype(np.float64), axes=([0], [0]))
    tau = (float(base_absorb_scale) * float(plume_k_scale)) * weighted
    tau = np.maximum(tau, 0.0)

    # --- artistic remap in TAU-space, then Beer-Lambert to alpha -------------
    # Doing the curve on tau (not on alpha) keeps the thin limit linear and the
    # thick limit saturating; b=1,a=1 is fully honest. This REPLACES the retired
    # smoke_render_gamma (which curved the alpha directly).
    tau_p = float(tau_curve_a) * np.power(tau, float(tau_curve_b))
    alpha = 1.0 - np.exp(-tau_p)

    # --- the lit half: tone-mapped inscatter (premultiplied additive RGB) ----
    glow = np.array(smoke_glow, dtype=np.float32) * float(glow_gain)
    if (effect_gas_floor > 0.0 and effect_gas_mask is not None
            and scatter_albedo is not None):
        # Add a faint self-emission for gameplay gases, tinted by their own
        # scatter_albedo and present where the gas is (density-weighted, bounded
        # to [0,1] so a dense cloud cannot blow the floor up). Added BEFORE the
        # tone-map so it stays ACES-consistent with the rest of the layer.
        emask = np.asarray(effect_gas_mask, dtype=bool)
        alb = np.asarray(scatter_albedo, dtype=np.float32)
        for s in range(gas_density.shape[0]):
            if emask[s]:
                d = np.clip(gas_density[s], 0.0, 1.0)
                glow = glow + (float(effect_gas_floor)
                               * d[..., None] * alb[s][None, None, :])

    rgb = aces_tonemap(glow)                                         # [0,1]
    return rgb.astype(np.float32), alpha.astype(np.float32)


def pack_premult_rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """(rgb (H,W,3) [0,1], alpha (H,W) [0,1]) -> (H,W,4) uint8 premultiplied.

    RGB carries the inscatter (the ADDITIVE lit half) DIRECTLY — it is NOT
    multiplied by alpha, because the premult-over blend composites
    ``C = inscatter + (1-alpha)·bg``. Alpha is the occlusion only. This is why
    the additive case (alpha ~ 0 with RGB > 0 — steam glowing in a beam) is
    representable: RGB is deliberately NOT bounded by alpha. Do not "fix" that.
    """
    h, w = alpha.shape
    packed = np.empty((h, w, 4), dtype=np.uint8)
    rgb = np.clip(rgb, 0.0, 1.0)
    packed[..., 0] = (rgb[..., 0] * 255.0).astype(np.uint8)
    packed[..., 1] = (rgb[..., 1] * 255.0).astype(np.uint8)
    packed[..., 2] = (rgb[..., 2] * 255.0).astype(np.uint8)
    packed[..., 3] = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    return packed


def pack_gas_medium_rgba(
    gas_density: np.ndarray,
    k_s: np.ndarray,
    *,
    base_absorb_scale: float,
    plume_k_scale: float,
    tau_curve_a: float,
    tau_curve_b: float,
    smoke_glow: np.ndarray,
    glow_gain: float,
    effect_gas_floor: float = 0.0,
    effect_gas_mask: Optional[np.ndarray] = None,
    scatter_albedo: Optional[np.ndarray] = None,
) -> np.ndarray:
    """gas_medium_layer + pack -> (H, W, 4) uint8 premultiplied RGBA (pyray-free)."""
    rgb, alpha = gas_medium_layer(
        gas_density, k_s, base_absorb_scale=base_absorb_scale,
        plume_k_scale=plume_k_scale, tau_curve_a=tau_curve_a,
        tau_curve_b=tau_curve_b, smoke_glow=smoke_glow, glow_gain=glow_gain,
        effect_gas_floor=effect_gas_floor, effect_gas_mask=effect_gas_mask,
        scatter_albedo=scatter_albedo)
    return pack_premult_rgba(rgb, alpha)


class GasMediumOverlay:
    """Owns the dynamic RGBA texture + the premult-over draw for the P2 layer.

    Tunables (``plume_k_scale`` / ``tau_curve_a`` / ``tau_curve_b`` /
    ``glow_gain`` / ``effect_gas_floor``) are plain attributes bound from
    ``[render.gas_medium]`` and mutated live by the harness sliders — exactly
    like ``GlowOverlay.gain``. The per-gas optics (``k_s`` / albedo / effect
    mask) are bound from the live GasTable (``gmap.gases``) so the plume reads
    the SAME optical data as the ray march; rebound automatically if the table
    is rebuilt (config reload). ``base_absorb_scale`` is passed per-frame from
    the Raycaster (single source of scale), never cached here.
    """

    def __init__(self, grid_h: int, grid_w: int, *,
                 plume_k_scale: float = 1.0, tau_curve_a: float = 1.0,
                 tau_curve_b: float = 1.0, glow_gain: float = 1.0,
                 effect_gas_floor: float = 0.0):
        self.h = grid_h
        self.w = grid_w
        self.plume_k_scale = float(plume_k_scale)
        self.tau_curve_a = float(tau_curve_a)
        self.tau_curve_b = float(tau_curve_b)
        self.glow_gain = float(glow_gain)
        self.effect_gas_floor = float(effect_gas_floor)
        self.tex = core.create_dynamic_rgba_texture(grid_w, grid_h)
        self.packed = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
        # Per-gas optics, bound lazily from the GasTable (see _bind_table).
        self._table_id: Optional[int] = None
        self._k_s: Optional[np.ndarray] = None
        self._scatter: Optional[np.ndarray] = None
        self._effect_mask: Optional[np.ndarray] = None

    @classmethod
    def from_config(cls, grid_h: int, grid_w: int, cfg) -> "GasMediumOverlay":
        """Build with the ``[render.gas_medium]`` defaults (getattr-guarded —
        a config without the block still gets the honest design defaults)."""
        render = getattr(cfg, "render", None)
        gm = getattr(render, "gas_medium", None)
        g = lambda name, default: float(getattr(gm, name, default))
        return cls(
            grid_h, grid_w,
            plume_k_scale=g("plume_k_scale", 1.0),
            tau_curve_a=g("tau_curve_a", 1.0),
            tau_curve_b=g("tau_curve_b", 1.0),
            glow_gain=g("glow_gain", 1.0),
            effect_gas_floor=g("effect_gas_floor", 0.0),
        )

    def _bind_table(self, gas_table) -> None:
        """Cache k_s / albedo / effect-mask from the GasTable's trace-gas rows.

        ``k_s = mean(absorption[s])`` — the panchromatic collapse; the SAME data
        that drives the ray march (single optical identity, design §3)."""
        n = N_TRACE_GASES
        self._k_s = np.asarray(gas_table.absorption[:n],
                               dtype=np.float32).mean(axis=1)
        self._scatter = np.asarray(gas_table.scatter_albedo[:n],
                                   dtype=np.float32)
        self._effect_mask = np.array(
            [str(e) in _EFFECT_GAS_TAGS for e in gas_table.effect[:n]],
            dtype=bool)
        self._table_id = id(gas_table)

    def update(self, gas_density_trace: np.ndarray, smoke_glow: np.ndarray,
               gas_table, *, base_absorb_scale: float) -> None:
        """Rebuild the premultiplied layer texture for this frame.

        ``gas_density_trace`` : (N_TRACE_GASES, H, W) float — dequantized trace
        densities. ``smoke_glow`` : (H, W, 3) float — the ray march inscatter.
        ``gas_table`` : the live GasTable (``gmap.gases``). ``base_absorb_scale``
        : the Raycaster's ``smoke_absorb_scale`` (shared base scale)."""
        if self._table_id != id(gas_table):
            self._bind_table(gas_table)
        self.packed = pack_gas_medium_rgba(
            gas_density_trace, self._k_s,
            base_absorb_scale=base_absorb_scale,
            plume_k_scale=self.plume_k_scale,
            tau_curve_a=self.tau_curve_a, tau_curve_b=self.tau_curve_b,
            smoke_glow=smoke_glow, glow_gain=self.glow_gain,
            effect_gas_floor=self.effect_gas_floor,
            effect_gas_mask=self._effect_mask, scatter_albedo=self._scatter)
        core.update_rgba_texture(self.tex, self.packed)

    def draw(self, dst_x: int, dst_y: int, dst_w: int, dst_h: int) -> None:
        """ONE premult-over draw (C = inscatter + T·bg). BLEND_ALPHA_PREMULTIPLY
        is ``out = src.rgb + dst*(1-src.a)`` — the volume compositing operator;
        it preserves destination alpha correctly (opaque ship stays opaque, and
        because src.a is the REAL occlusion, vacuum tiles with no gas leave dstA
        untouched — no glow-overlay alpha-saturation hazard)."""
        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        src = rl.Rectangle(0, 0, float(self.w), float(self.h))
        dst = rl.Rectangle(float(dst_x), float(dst_y),
                           float(dst_w), float(dst_h))
        rl.draw_texture_pro(self.tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_blend_mode()


__all__ = [
    "gas_medium_layer", "pack_premult_rgba", "pack_gas_medium_rgba",
    "GasMediumOverlay",
]
