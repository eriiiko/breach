"""The sub-tile gas DETAIL pass (Fire & Heat Beauty arc, B2 P3) — owns
``shaders/gas_medium.fs`` + its textures, and draws the P2 gas-medium layer
THROUGH the shader to add advected-noise wisps without touching the sim.

RENDER-ONLY, determinism-EXEMPT. It reads two sim fields (the wind field, and —
via ``GasMediumOverlay.density_proxy`` — the trace densities), converts them to
render textures, and writes only its own GPU textures + the one premultiplied
draw. It never writes any sim state and cannot move a golden.

Plumbing mirrors ``renderer/water.py``'s ``WaterPass`` (the proven multi-texture
shader seam in this renderer): load the shader, look up uniforms (warn-but-
continue on -1), bake the static noise textures ONCE, upload a per-frame
dynamics texture, and bind everything by name INSIDE begin_shader_mode.

Texture units bound (4; ``WaterPass`` proves 5 works here):
  0 u_layer     - the P2 premultiplied RGBA8 layer (the draw texture; BILINEAR)
    u_dynamics  - per-frame RGBA16F: RG = wind tiles/tick, B = density solidity
    u_fbm       - baked tiling fBm (REPEAT+BILINEAR): R coverage, GB warp
    u_jitter    - baked white noise (REPEAT+POINT): R phase jitter, G dither

WIND UNITS — the critique's premise was EMPIRICALLY WRONG, corrected here.
``gmap.wind_x`` / ``wind_y`` are int32 Q16.16 planes of raw ``-grad(P)``. The
design said to scale them by ``advection_rate * dt`` (900/24) to get tiles/tick,
on the theory that this reproduces the plume drift. MEASURED in a burning
fire_studio (docs + probe): the raw ``-grad(P)`` is FIRE-SPIKED (plume-cell
dequant p50 ~ 12, up to ~1000+), so ``dequant * rate * dt`` gives ~1e4 tiles/tick
— five orders of magnitude past what the smoke actually does. The sim's
semi-Lagrangian backtrace clamps to walls/bounds and the field is diffusion-
dominated, so the SMOKE CENTROID only drifts ~0.1-0.3 tiles/tick and is quasi-
STATIONARY (it wobbles in place, it does not translate across the map). So a
literal velocity match is impossible; the raw wind is unusable as a velocity.

Instead this pass TAMES the wind into a bounded visual velocity: smooth the
spiky field (coherent flow direction), take the DIRECTION, and give it a small
SATURATING speed ``v = v_ref * (1 - exp(-|w| / v_sens))`` tiles/tick (default
v_ref ~ 0.08). The dominant "alive" motion is then the crossfade BOILING (which
happens even at zero wind — the plume is nearly stationary anyway); the tamed
wind is a gentle directional bias. ``adv_gain`` (config, the design's tuning
dial) multiplies the drift in the shader, so Erik dials flow-vs-boil in the
studio. This is a render-side CALIBRATION (P3's mandate), flagged for the
human-test. See docs/fire_b2_smoke_honesty_design_2026-07-21.md §4 + the P3
build notes.

Design: docs/fire_b2_smoke_honesty_design_2026-07-21.md §4. Credit for the
technique lives in shaders/gas_medium.fs + renderer/advected_noise.py headers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pyray as rl

from . import core
from .advected_noise import bake_fbm_rgba, bake_jitter_rgba, advection_phase

# Q16.16 scale for the wind dequantize (matches simulation.atmosphere_fixed —
# the renderer-boundary dequantize; kept as a local constant so this render
# module needs no sim import just for one number).
_FP_ONE_F = float(1 << 16)

# Bake sizes (design §4): fBm ~256^2 tiling, jitter small + static.
_FBM_SIZE = 256
_JITTER_SIZE = 64
_FBM_PERSISTENCE = 0.56           # Kolmogorov-flavoured roll-off (design §6)
_DITHER_AMP = 1.5 / 255.0         # ~half an 8-bit step, breaks banding

# Wind TAMING (see module header): the raw -grad(P) is fire-spiked + unusable as
# a velocity, so we smooth it, take the direction, and give a small saturating
# speed. Calibrated by eye against a real burning-studio wind snapshot so the
# drift reads as gentle smoke flow (the plume itself is near-stationary). Erik
# tunes flow-vs-boil live via adv_gain (config), which multiplies in the shader.
_WIND_VREF = 0.08                 # saturating max drift speed, tiles/tick
_WIND_VSENS = 0.5                 # dequant |wind| for ~63% of v_ref
_WIND_SMOOTH_PASSES = 2           # 3x3 box passes -> coherent flow direction

SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"


class GasDetailPass:
    """Owns the detail shader, the baked noise textures, and the per-frame
    wind/density dynamics texture. ``enabled`` gates the whole pass: when False
    the caller draws the plain P2 layer instead (byte-for-byte the P2 look)."""

    def __init__(self, grid_h: int, grid_w: int, *,
                 enabled: bool = True, noise_octaves: int = 4,
                 noise_wavelength_tiles: float = 3.0, adv_gain: float = 1.0,
                 cycle_seconds: float = 2.5, erode_strength: float = 0.6,
                 warp_px: float = 3.0, dither_on: bool = True):
        self.h = grid_h
        self.w = grid_w
        # Live-tunable dials (plain attributes, mutated by the harness sliders —
        # the GasMediumOverlay / WaterPass precedent). Read into uniforms each
        # draw(); noise_octaves is the one BAKE-time dial (lazy-rebaked below).
        self.enabled = bool(enabled)
        self.noise_octaves = int(round(noise_octaves))
        self.noise_wavelength_tiles = float(noise_wavelength_tiles)
        self.adv_gain = float(adv_gain)
        self.cycle_seconds = float(cycle_seconds)
        self.erode_strength = float(erode_strength)
        self.warp_px = float(warp_px)
        self.dither_on = bool(dither_on)
        # Wind-taming calibration (not a config dial — the user dial is adv_gain,
        # which multiplies the drift in the shader). See the module header.
        self.wind_v_ref = _WIND_VREF
        self.wind_v_sens = _WIND_VSENS

        # Domain-warp is authored in "screen px at the reference 24 px/tile"
        # (design), converted to tiles here so the shader stays resolution-clean.
        self._px_per_tile_ref = 24.0
        # Dither granularity: a few cells per tile after the bicubic upscale.
        self._dither_scale = float(max(grid_w, grid_h)) * 3.0
        # Phase-jitter sample wavelength (tiles): coarse -> spatially smooth, so
        # it desyncs the crossfade across the screen without adding shimmer.
        self._jitter_wl = 2.0

        # --- shader + uniform locations (mirror WaterPass) -------------------
        self.shader = core.load_shader_with_fallback(
            str(SHADERS_DIR / "lighting.vs"),      # reuse the passthrough VS
            str(SHADERS_DIR / "gas_medium.fs"),
        )
        self._loc_layer     = self._lookup("u_layer")
        self._loc_dynamics  = self._lookup("u_dynamics")
        self._loc_fbm       = self._lookup("u_fbm")
        self._loc_jitter    = self._lookup("u_jitter")
        self._loc_grid      = self._lookup("u_grid")
        self._loc_noise_wl  = self._lookup("u_noise_wl")
        self._loc_adv_gain  = self._lookup("u_adv_gain")
        self._loc_phase     = self._lookup("u_phase")
        self._loc_tau_ticks = self._lookup("u_tau_ticks")
        self._loc_erode     = self._lookup("u_erode")
        self._loc_warp      = self._lookup("u_warp_tiles")
        self._loc_dither_on = self._lookup("u_dither_on")
        self._loc_dither_amp = self._lookup("u_dither_amp")
        self._loc_jitter_wl = self._lookup("u_jitter_wl")
        self._loc_dither_scale = self._lookup("u_dither_scale")

        # --- per-frame dynamics texture (RG wind tiles/tick, B density) ------
        self.dyn_packed = np.zeros((grid_h, grid_w, 4), dtype=np.float16)
        self.dyn_tex = core.create_dynamic_rgba16f_texture(grid_w, grid_h)
        # Bilinear + clamp (smooth wind/density between tiles; the bicubic taps
        # rely on bilinear filtering for the density channel).

        # --- static baked noise textures (ONCE) -----------------------------
        self._baked_octaves = self.noise_octaves
        self.fbm_tex = self._make_static_rgba(
            bake_fbm_rgba(_FBM_SIZE, self.noise_octaves, _FBM_PERSISTENCE),
            wrap_repeat=True, point=False)
        self.jitter_tex = self._make_static_rgba(
            bake_jitter_rgba(_JITTER_SIZE), wrap_repeat=True, point=True)

        # Cached phase for draw() (computed per-frame in update()).
        self._phase = 0.0
        self._tau_ticks = float(cycle_seconds) * 24.0

    @classmethod
    def from_config(cls, grid_h: int, grid_w: int, cfg) -> "GasDetailPass":
        """Build from ``[render.gas_detail]`` (getattr-guarded honest defaults)."""
        render = getattr(cfg, "render", None)
        gd = getattr(render, "gas_detail", None)
        gi = lambda name, default: getattr(gd, name, default)
        return cls(
            grid_h, grid_w,
            enabled=bool(gi("enabled", True)),
            noise_octaves=int(gi("noise_octaves", 4)),
            noise_wavelength_tiles=float(gi("noise_wavelength_tiles", 3.0)),
            adv_gain=float(gi("adv_gain", 1.0)),
            cycle_seconds=float(gi("cycle_seconds", 2.5)),
            erode_strength=float(gi("erode_strength", 0.6)),
            warp_px=float(gi("warp_px", 3.0)),
            dither_on=bool(gi("dither_on", True)),
        )

    # ---- texture helpers ------------------------------------------------

    def _make_static_rgba(self, img_rgba: np.ndarray, *, wrap_repeat: bool,
                          point: bool) -> rl.Texture:
        """Upload a baked (H,W,4) uint8 array as a static texture with explicit
        wrap/filter (deliberate per-texture choice, design §4)."""
        h, w = img_rgba.shape[:2]
        tex = core.create_dynamic_rgba_texture(w, h)
        core.update_rgba_texture(tex, np.ascontiguousarray(img_rgba,
                                                            dtype=np.uint8))
        rl.set_texture_filter(
            tex, rl.TextureFilter.TEXTURE_FILTER_POINT if point
            else rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        rl.set_texture_wrap(
            tex, rl.TextureWrap.TEXTURE_WRAP_REPEAT if wrap_repeat
            else rl.TextureWrap.TEXTURE_WRAP_CLAMP)
        return tex

    def _lookup(self, name: str) -> int:
        loc = rl.get_shader_location(self.shader, name)
        if loc == -1:
            print(f"[gas_detail] WARN: shader uniform '{name}' not found (loc=-1)")
        return loc

    def _set_f(self, loc: int, v: float) -> None:
        val = rl.ffi.new("float[1]", [float(v)])
        rl.set_shader_value(self.shader, loc, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_FLOAT)

    def _set_i(self, loc: int, v: int) -> None:
        val = rl.ffi.new("int[1]", [int(v)])
        rl.set_shader_value(self.shader, loc, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_INT)

    def _set_vec2(self, loc: int, x: float, y: float) -> None:
        val = rl.ffi.new("float[2]", [float(x), float(y)])
        rl.set_shader_value(self.shader, loc, val,
                            rl.ShaderUniformDataType.SHADER_UNIFORM_VEC2)

    # ---- per-frame upload ----------------------------------------------

    @staticmethod
    def _box3(a: np.ndarray) -> np.ndarray:
        """One separable 3x3 box-blur pass with edge replication (pyray-free)."""
        p = np.pad(a, 1, mode="edge")               # (h+2, w+2)
        hz = (p[:, :-2] + p[:, 1:-1] + p[:, 2:]) / 3.0    # (h+2, w)
        return (hz[:-2, :] + hz[1:-1, :] + hz[2:, :]) / 3.0  # (h, w)

    @classmethod
    def pack_dynamics(cls, wind_x: np.ndarray, wind_y: np.ndarray,
                      density_proxy: np.ndarray, *,
                      v_ref: float = _WIND_VREF,
                      v_sens: float = _WIND_VSENS) -> np.ndarray:
        """(pyray-free) pack the RGBA16F dynamics array from the Q16.16 wind
        planes + the density solidity.

        R,G = the TAMED wind velocity in TILES/TICK: dequantize the raw
              ``-grad(P)`` (÷2^16), smooth it (coherent flow direction), keep the
              DIRECTION, and give it a small SATURATING speed
              ``v = v_ref * (1 - exp(-|w| / v_sens))``. The raw field is fire-
              spiked and NOT a usable velocity (module header), so this bounds it
              to a gentle drift; ``adv_gain`` in the shader scales it further.
        B   = density_proxy (saturate of the pre-curve optical depth), the
              erosion weight. A = 0.
        """
        h, w = density_proxy.shape
        wx = wind_x.astype(np.float64) / _FP_ONE_F
        wy = wind_y.astype(np.float64) / _FP_ONE_F
        for _ in range(_WIND_SMOOTH_PASSES):
            wx = cls._box3(wx)
            wy = cls._box3(wy)
        speed = np.hypot(wx, wy)
        vmag = float(v_ref) * (1.0 - np.exp(-speed / max(float(v_sens), 1e-6)))
        inv = np.where(speed > 1e-9, vmag / np.maximum(speed, 1e-30), 0.0)
        out = np.empty((h, w, 4), dtype=np.float16)
        out[..., 0] = (wx * inv).astype(np.float16)          # tiles/tick
        out[..., 1] = (wy * inv).astype(np.float16)
        out[..., 2] = density_proxy.astype(np.float16)
        out[..., 3] = np.float16(0.0)
        return out

    def update(self, density_proxy: np.ndarray, wind_x: np.ndarray,
               wind_y: np.ndarray, *, sim_tick: int, sim_dt: float) -> None:
        """Refresh the dynamics texture + the crossfade phase for this frame.

        ``sim_tick`` is the deterministic (monotonic) sim tick — the smoke clock
        (never wall time), so replays render identical smoke. RENDER-ONLY: every
        input is read, never written."""
        # Lazy rebake if the octaves dial moved (bake-time param; a few ms, only
        # on change — gives the harness live feel without per-frame cost).
        oct_now = int(round(self.noise_octaves))
        if oct_now != self._baked_octaves and oct_now >= 1:
            rl.unload_texture(self.fbm_tex)
            self.fbm_tex = self._make_static_rgba(
                bake_fbm_rgba(_FBM_SIZE, oct_now, _FBM_PERSISTENCE),
                wrap_repeat=True, point=False)
            self._baked_octaves = oct_now

        self.dyn_packed = self.pack_dynamics(
            wind_x, wind_y, density_proxy,
            v_ref=self.wind_v_ref, v_sens=self.wind_v_sens)
        core.update_rgba16f_texture(self.dyn_tex, self.dyn_packed)

        ph = advection_phase(int(sim_tick), self.cycle_seconds,
                             1.0 / float(sim_dt) if sim_dt > 0 else 0.0)
        self._phase = ph.phase
        self._tau_ticks = ph.tau_ticks

    # ---- draw ----------------------------------------------------------

    def draw(self, layer_tex: rl.Texture, dst_x: int, dst_y: int,
             dst_w: int, dst_h: int) -> None:
        """Draw the P2 layer THROUGH the detail shader, premultiplied.

        Same blend as the P2 plain draw (BLEND_ALPHA_PREMULTIPLY, out = src.rgb +
        dst.rgb*(1-src.a)) so the volume compositing is unchanged; the shader
        only reshapes the layer (bicubic + noise erosion). Sampler binds are
        issued INSIDE begin_shader_mode (the raylib gotcha WaterPass documents:
        a bind before begin_shader_mode can target the wrong shader)."""
        # Live dials -> uniforms (cheap; safe every frame — mirrors WaterPass).
        self._set_vec2(self._loc_grid, float(self.w), float(self.h))
        self._set_f(self._loc_noise_wl, max(self.noise_wavelength_tiles, 1e-3))
        self._set_f(self._loc_adv_gain, self.adv_gain)
        self._set_f(self._loc_phase, self._phase)
        self._set_f(self._loc_tau_ticks, self._tau_ticks)
        self._set_f(self._loc_erode, self.erode_strength)
        self._set_f(self._loc_warp, self.warp_px / self._px_per_tile_ref)
        self._set_i(self._loc_dither_on, 1 if self.dither_on else 0)
        self._set_f(self._loc_dither_amp, _DITHER_AMP)
        self._set_f(self._loc_jitter_wl, self._jitter_wl)
        self._set_f(self._loc_dither_scale, self._dither_scale)

        rl.begin_blend_mode(rl.BlendMode.BLEND_ALPHA_PREMULTIPLY)
        rl.begin_shader_mode(self.shader)
        # texture0 is the draw texture; bind every sampler by name (WaterPass).
        rl.set_shader_value_texture(self.shader, self._loc_layer, layer_tex)
        rl.set_shader_value_texture(self.shader, self._loc_dynamics, self.dyn_tex)
        rl.set_shader_value_texture(self.shader, self._loc_fbm, self.fbm_tex)
        rl.set_shader_value_texture(self.shader, self._loc_jitter, self.jitter_tex)
        src = rl.Rectangle(0, 0, float(self.w), float(self.h))
        dst = rl.Rectangle(float(dst_x), float(dst_y),
                           float(dst_w), float(dst_h))
        rl.draw_texture_pro(layer_tex, src, dst, rl.Vector2(0, 0), 0.0, rl.WHITE)
        rl.end_shader_mode()
        rl.end_blend_mode()

    def unload(self) -> None:
        rl.unload_shader(self.shader)
        rl.unload_texture(self.dyn_tex)
        rl.unload_texture(self.fbm_tex)
        rl.unload_texture(self.jitter_tex)


__all__ = ["GasDetailPass"]
