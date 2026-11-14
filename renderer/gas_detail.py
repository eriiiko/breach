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
  0 u_layer     - the P2 premultiplied RGBA16F layer (the draw texture; BILINEAR)
    u_dynamics  - per-frame RGBA16F: RG = wind tiles/tick, B = density solidity
    u_fbm       - baked tiling fBm (REPEAT+BILINEAR): R coverage, GB warp
    u_jitter    - baked white noise (REPEAT+POINT): R phase jitter (G unused --
                  the dither moved to the world resolve, graphics G0 P1)

WIND UNITS (issue #51, graphics G0 P2). ``gmap.wind_x`` / ``wind_y`` are the
EOS solver's own velocity state u, int32 Q16.16 in METRES PER SECOND
(cpp/src/eos_solver.h; per-cell capped at the local sound speed). They are
NOT a raw ``-grad(P)`` to be scaled by the pre-EOS ``advection_rate`` (900):
that legacy reading made the field look "five orders of magnitude past what
the smoke does" and the taming below was first calibrated on it. The honest
conversion is :func:`wind_tiles_per_tick`:

    tiles / tick = v [m/s] / tile_size_m / ticks_per_second

with ``tile_size_m`` from the level (``gmap.tile_size_m``) and the tick rate
from ``[clock] ticks_per_second`` -- no constant of this module knows either.

On that honest scale the field is still a poor VISUAL velocity: blast and fire
cells spike to the sound-speed cap while the plume as a whole barely moves.
So this pass TAMES it into a bounded visual drift: smooth the spiky field
(coherent flow direction), keep the DIRECTION, and give it a SATURATING speed
``v = v_ref * (1 - exp(-|u| / v_sens))``, both dials in tiles/tick. The
"alive" motion is then mostly the crossfade BOILING (which happens even at
zero wind); the tamed wind is a directional drift calibrated against the
plume's own measured centroid drift (``tools/wind_drift_probe.py``, numbers
at :data:`_WIND_VREF`). ``adv_gain`` (config) multiplies the drift in the
shader, so Erik dials flow-vs-boil in the studio. See
docs/fire_b2_smoke_honesty_design_2026-07-21.md §4 and
docs/graphics_direction_2026-10-18.md §10.

The taming lives in the module-level :func:`tame_wind` — THE render-side wind
seam, usable without a GL context and without this pass being enabled. Its
second consumer is prop sway (`renderer/static_props.py`, props & vegetation
arc #60 P4, design §4.3 F3). Anything that needs "the wind" as a velocity calls
it; a parallel smoothing/limiting implementation is the bug.

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
from simulation import atmosphere_fixed   # THE wind's Q16 boundary module

# Bake sizes (design §4): fBm ~256^2 tiling, jitter small + static.
_FBM_SIZE = 256
_JITTER_SIZE = 64
_FBM_PERSISTENCE = 0.56           # Kolmogorov-flavoured roll-off (design §6)

# Wind TAMING (see module header): the honest velocity (tiles/tick) is spiked
# at blast/fire cells, so we smooth it, take the direction, and give it a
# saturating speed. Both dials are in TILES PER TICK on the honest scale (#51).
# Erik tunes flow-vs-boil live via adv_gain (config), which multiplies in the
# shader. The values are the P2 RETUNE (tools/wind_drift_probe.py; the
# measurement is written down in docs/graphics_direction_2026-10-18.md §10).
# MEASURED (tools/wind_drift_probe.py + a dial sweep, 2026-10-19): on the
# smoke-light studio (tick 0-168) and a fire_studio frag blast (tick 24-120)
# the plume's tau-weighted centroid drifts 0.12-0.78 tiles/tick while a blast
# is moving the air and ~0.01 once it settles; the plume-weighted mean of the
# honest velocity is 0.5-4.6 tiles/tick in speed (blast) and 0.07-0.4 (settled
# sloshing). The pre-#51 dials (0.08, and v_sens 0.5 "dequant" = 0.0625
# tiles/tick) capped the visual drift at 0.08: ~10x too slow in a blast,
# median |log(visual / centroid)| 0.81 over 215 source-free windows. These
# dials minimise that error (median 0.25, mean 0.40; a broad valley along
# v_ref / v_sens ~ 0.35-0.5, i.e. ~0.47 x the smoothed |u| until it saturates).
_WIND_VREF = 1.4                  # saturating max drift speed, tiles/tick
_WIND_VSENS = 3.0                 # |u| (tiles/tick) for ~63% of v_ref
_WIND_SMOOTH_PASSES = 2           # 3x3 box passes -> coherent flow direction

# PUBLIC aliases of the taming calibration. A second consumer of the tamed wind
# (props & vegetation arc #60 P4: `renderer/static_props.py`'s sway) normalizes
# against the SAME saturation ceiling, so "full sway" means "the wind the smoke
# already treats as full drift". Import these — never re-type 0.08.
WIND_V_REF = _WIND_VREF
WIND_V_SENS = _WIND_VSENS
WIND_SMOOTH_PASSES = _WIND_SMOOTH_PASSES

SHADERS_DIR = Path(__file__).resolve().parent.parent / "shaders"


# ---------------------------------------------------------------------------
# THE wind-taming seam (module-level so it is usable WITHOUT a GL context and
# without the detail pass being enabled — props & vegetation arc #60 P4).
# ---------------------------------------------------------------------------

def _box3(a: np.ndarray) -> np.ndarray:
    """One separable 3x3 box-blur pass with edge replication (pyray-free)."""
    p = np.pad(a, 1, mode="edge")                     # (h+2, w+2)
    hz = (p[:, :-2] + p[:, 1:-1] + p[:, 2:]) / 3.0    # (h+2, w)
    return (hz[:-2, :] + hz[1:-1, :] + hz[2:, :]) / 3.0  # (h, w)


def wind_tiles_per_tick(wind_x: np.ndarray, wind_y: np.ndarray, *,
                        tile_size_m: float, ticks_per_second: float):
    """The HONEST wind conversion (#51): the EOS velocity planes (int32 Q16.16
    metres per second, dequantized through ``simulation.atmosphere_fixed``)
    -> ``(ux, uy)`` float64 in TILES PER SIM TICK:
    ``v / tile_size_m / ticks_per_second``. Both scales come from the caller
    (the level's ``tile_size_m``, ``[clock] ticks_per_second``) -- never a
    constant here. RENDER-ONLY: reads the planes, writes nothing."""
    k = 1.0 / (float(tile_size_m) * float(ticks_per_second))
    return (atmosphere_fixed.dequantize(wind_x) * k,
            atmosphere_fixed.dequantize(wind_y) * k)


def tame_wind(wind_x: np.ndarray, wind_y: np.ndarray, *,
              tile_size_m: float, ticks_per_second: float,
              v_ref: float = _WIND_VREF, v_sens: float = _WIND_VSENS,
              smooth_passes: int = _WIND_SMOOTH_PASSES) -> np.ndarray:
    """Turn the Q16.16 m/s velocity planes into a USABLE visual velocity.

    THE canonical wind-taming helper (module header): convert to tiles/tick
    (:func:`wind_tiles_per_tick`, the honest scale), box-smooth
    (``smooth_passes`` 3x3 passes → a coherent flow DIRECTION out of a field
    spiked at blast/fire cells), keep that direction, and give it a
    SATURATING speed ``v = v_ref * (1 - exp(-|u| / v_sens))``.

    Returns ``(h, w, 2)`` float32 in TILES/TICK, ``[..., 0] = x``,
    ``[..., 1] = y`` — bounded by ``v_ref`` everywhere, by construction.

    Every render consumer of "the wind" calls THIS (``pack_dynamics`` for the
    smoke detail pass; ``StaticPropRenderer`` for prop sway). A second smoothing
    /limiting implementation elsewhere is the bug: the raw planes are spiked and
    each ad-hoc taming would disagree about which way the air is moving in the
    same room.

    RENDER-ONLY: reads sim planes, writes nothing (the dequantize-at-the-render-
    read convention — the float result never goes back into sim state).
    """
    wx, wy = wind_tiles_per_tick(wind_x, wind_y, tile_size_m=tile_size_m,
                                 ticks_per_second=ticks_per_second)
    for _ in range(int(smooth_passes)):
        wx = _box3(wx)
        wy = _box3(wy)
    speed = np.hypot(wx, wy)
    vmag = float(v_ref) * (1.0 - np.exp(-speed / max(float(v_sens), 1e-6)))
    inv = np.where(speed > 1e-9, vmag / np.maximum(speed, 1e-30), 0.0)
    out = np.empty(wx.shape + (2,), dtype=np.float32)
    out[..., 0] = (wx * inv).astype(np.float32)
    out[..., 1] = (wy * inv).astype(np.float32)
    return out


class GasDetailPass:
    """Owns the detail shader, the baked noise textures, and the per-frame
    wind/density dynamics texture. ``enabled`` gates the whole pass: when False
    the caller draws the plain P2 layer instead (byte-for-byte the P2 look)."""

    def __init__(self, grid_h: int, grid_w: int, *,
                 enabled: bool = True, noise_octaves: int = 4,
                 noise_wavelength_tiles: float = 3.0, adv_gain: float = 1.0,
                 cycle_seconds: float = 2.5, erode_strength: float = 0.6,
                 warp_px: float = 3.0):
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
        # Wind-taming calibration (not a config dial — the user dial is adv_gain,
        # which multiplies the drift in the shader). See the module header.
        self.wind_v_ref = _WIND_VREF
        self.wind_v_sens = _WIND_VSENS

        # Domain-warp is authored in "screen px at the reference 24 px/tile"
        # (design), converted to tiles here so the shader stays resolution-clean.
        self._px_per_tile_ref = 24.0
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
        self._loc_jitter_wl = self._lookup("u_jitter_wl")

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

    _box3 = staticmethod(_box3)     # module-level; kept as an attribute so the
                                    # historical GasDetailPass._box3 call site
                                    # (and its tests) still resolve.

    @classmethod
    def pack_dynamics(cls, wind_x: np.ndarray, wind_y: np.ndarray,
                      density_proxy: np.ndarray, *,
                      tile_size_m: float, ticks_per_second: float,
                      v_ref: float = _WIND_VREF,
                      v_sens: float = _WIND_VSENS) -> np.ndarray:
        """(pyray-free) pack the RGBA16F dynamics array from the Q16.16 m/s
        wind planes + the density solidity.

        R,G = the TAMED wind velocity in TILES/TICK — :func:`tame_wind`, the ONE
              taming helper (this method is a consumer of it, not a second copy
              of the math). The honest field is spiked at blast/fire cells
              (module header), so taming bounds it to the plume's drift;
              ``adv_gain`` in the shader scales it further.
        B   = density_proxy (saturate of the pre-curve optical depth), the
              erosion weight. A = 0.
        """
        h, w = density_proxy.shape
        tamed = tame_wind(wind_x, wind_y, tile_size_m=tile_size_m,
                          ticks_per_second=ticks_per_second,
                          v_ref=v_ref, v_sens=v_sens)
        out = np.empty((h, w, 4), dtype=np.float16)
        out[..., 0] = tamed[..., 0].astype(np.float16)       # tiles/tick
        out[..., 1] = tamed[..., 1].astype(np.float16)
        out[..., 2] = density_proxy.astype(np.float16)
        out[..., 3] = np.float16(0.0)
        return out

    def update(self, density_proxy: np.ndarray, wind_x: np.ndarray,
               wind_y: np.ndarray, *, sim_tick: int, sim_dt: float,
               tile_size_m: float) -> None:
        """Refresh the dynamics texture + the crossfade phase for this frame.

        ``sim_tick`` is the deterministic (monotonic) sim tick — the smoke clock
        (never wall time), so replays render identical smoke; ``sim_dt`` its
        length and ``tile_size_m`` the level's tile, the two scales of the
        honest wind conversion (#51). RENDER-ONLY: every input is read, never
        written."""
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
            wind_x, wind_y, density_proxy, tile_size_m=tile_size_m,
            ticks_per_second=(1.0 / float(sim_dt)) if sim_dt > 0 else 0.0,
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
        self._set_f(self._loc_jitter_wl, self._jitter_wl)

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


__all__ = ["GasDetailPass", "tame_wind", "wind_tiles_per_tick",
           "WIND_V_REF", "WIND_V_SENS",
           "WIND_SMOOTH_PASSES"]
