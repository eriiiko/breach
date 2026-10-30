"""Heat shimmer -- the first term of the RESOLVE'S DISTORTION BLOCK (graphics
arc G3 "Fire", patch F4a; docs/g3_fire_design_2026-10-23.md §6.2, §10(b),
§13.12).

WHAT IT IS. Hot air bends light: the floor seen through the air over a fire
wobbles. Here that is a small UV offset applied INSIDE the one resolve
(``shaders/world_resolve.fs``, the ``distortion_px`` block) when it reads the
scene RT -- so there is no read-while-write and no second full-screen pass,
the HUD / UI (drawn after the resolve) are never distorted, and the tone-map
stays the one tone-map. EVERY later screen-space distortion (the G4
shockwave next) is another term in that same block, never a second pass.

THE OFFSET (its spec is :func:`shimmer_offset_px`, the shader mirrors it):

    offset = n(p, t) * smooth(S(p)) * amp_tiles * px_per_tile * zoom_fade

  * ``S`` = the hot-gas STRENGTH, per tile (:func:`shimmer_strength`): 0 at
    or below ``t0_k``, 1 at ``t1_k``, linear between, on gas tiles only
    (solid and vacuum tiles are 0 -- nothing to shimmer through). Uploaded
    per tile and upsampled BILINEARLY by the GPU, then smoothstepped -- so a
    pixel whose four nearest tile centres are all cold gets EXACTLY zero
    offset, and the resolve's plain point fetch, bit for bit.
  * ``n`` = a 2D noise vector in [-1, 1]^2: the decorrelated fBm pair of
    ``advected_noise.bake_fbm_rgba`` (G, B), ADVECTED along the frame's one
    tamed wind plus a slow boil drift, on a two-phase flow map
    (``fire_vfx.noise_phases``: Vlachos' crossfade, a fresh grain offset per
    cycle) on the VFX CLOCK (``SimClock.t_noise`` -- the flame's clock, so it
    idles with the flame while paused and a look shot is reproducible).
  * ``zoom_fade`` = 0 below ``zoom_zero`` screen px per tile, 1 above
    ``zoom_full``: zoomed out, a sub-pixel wobble only reads as the whole
    screen swimming, so it fades away.
  * |offset| <= ``amp_tiles * px_per_tile`` always (clamped).

THE TAP (§13.12). The scene RT is point-filtered (the resolve is a 1:1
copy), so a shifted read does its OWN bilinear tap (four ``texelFetch``) --
never a filter change on the scene RT.

RENDER-ONLY. Reads read-only views (temperature through the canonical
game-T -> Kelvin map, the solid / vacuum masks, the tamed wind); writes no
synced state; pyray only in the GL class.

Credits: Alex Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010 (the two-
phase flow-map crossfade); Fabrice Neyret, "Advected Textures", SCA 2003;
Ken Perlin & Fabrice Neyret, "Flow Noise", SIGGRAPH 2001 sketch. (The same
references as ``renderer/advected_noise.py``.)
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, fields
from typing import Callable, Optional, Tuple

import numpy as np

NOISE_SIZE = 128
SPEED_CAP_TILES_S = 3.0      # the advecting wind's speed cap in the texture
_UNIT_HEAT, _UNIT_NOISE = 5, 6
_GL_TEXTURE0 = 0x84C0
_GL_TEXTURE_2D = 0x0DE1


@dataclass(frozen=True)
class ShimmerSettings:
    """Every ``[render.shimmer]`` dial (the config is the live value)."""
    enabled: bool = True
    t0_k: float = 320.0          # gas at or below this does not shimmer at all
    t1_k: float = 520.0          # full strength from this gas temperature up
    amp_tiles: float = 0.10      # the largest offset, tiles (x px_per_tile)
    grain_tiles: float = 0.6     # tiles per repeat of the noise grain
    cycle_s: float = 0.45        # the flow map's phase cycle, s (VFX clock)
    boil_tiles_s: float = 0.8    # the grain's own drift, tiles/s (hot air rising)
    zoom_zero: float = 14.0      # screen px per tile at and below which it is off
    zoom_full: float = 32.0      # ... and from which it is at full strength
    seed: int = 2718


def config_keys() -> Tuple[str, ...]:
    return tuple(f.name for f in fields(ShimmerSettings))


def settings_from_config(cfg) -> ShimmerSettings:
    sec = getattr(getattr(cfg, "render", None), "shimmer", None)
    base = ShimmerSettings()
    kw = {}
    for f in fields(ShimmerSettings):
        d = getattr(base, f.name)
        kw[f.name] = (d if sec is None or not hasattr(sec, f.name)
                      else type(d)(getattr(sec, f.name)))
    return ShimmerSettings(**kw)


# ---------------------------------------------------------------------------
# The law (pure)
# ---------------------------------------------------------------------------

def shimmer_strength(t_kelvin: np.ndarray, gas: np.ndarray, t0_k: float,
                     t1_k: float) -> np.ndarray:
    """Per-tile hot-gas strength in [0, 1]: ``(T - t0) / (t1 - t0)`` clipped,
    on GAS tiles only (``gas`` = not solid, not vacuum); exactly 0 at or
    below ``t0_k``. Monotone in T."""
    tk = np.asarray(t_kelvin, dtype=np.float64)
    span = max(float(t1_k) - float(t0_k), 1e-6)
    s = np.clip((tk - float(t0_k)) / span, 0.0, 1.0)
    return np.where(np.asarray(gas, dtype=bool), s, 0.0)


def zoom_fade(zoom_px_per_tile: float, zoom_zero: float,
              zoom_full: float) -> float:
    """0 at or below ``zoom_zero`` screen px per tile, 1 at or above
    ``zoom_full``, smoothstep between."""
    span = max(float(zoom_full) - float(zoom_zero), 1e-6)
    u = min(max((float(zoom_px_per_tile) - float(zoom_zero)) / span, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def bilinear_tiles(strength: np.ndarray, x_tiles: np.ndarray,
                   y_tiles: np.ndarray) -> np.ndarray:
    """The GPU's bilinear upsampling of a per-tile texture (CLAMP), at
    positions in tiles (tile (i, j)'s centre at (j + .5, i + .5)) -- the
    spec of the resolve's ``texture(u_shimmer_heat, ...)``."""
    h, w = strength.shape
    fx = np.asarray(x_tiles, np.float64) - 0.5
    fy = np.asarray(y_tiles, np.float64) - 0.5
    x0 = np.floor(fx)
    y0 = np.floor(fy)
    ax, ay = fx - x0, fy - y0
    x0i = np.clip(x0.astype(np.int64), 0, w - 1)
    x1i = np.clip(x0.astype(np.int64) + 1, 0, w - 1)
    y0i = np.clip(y0.astype(np.int64), 0, h - 1)
    y1i = np.clip(y0.astype(np.int64) + 1, 0, h - 1)
    top = strength[y0i, x0i] * (1 - ax) + strength[y0i, x1i] * ax
    bot = strength[y1i, x0i] * (1 - ax) + strength[y1i, x1i] * ax
    return top * (1 - ay) + bot * ay


def shimmer_offset_px(s_sampled: np.ndarray, noise_vec: np.ndarray,
                      amp_px: float) -> np.ndarray:
    """The offset in world px from the SAMPLED strength (bilinear) and the
    noise vector in [-1, 1]^2: ``noise * smoothstep(s) * amp_px``, its
    length clamped to ``amp_px``. Zero wherever ``s_sampled`` is 0."""
    s = np.clip(np.asarray(s_sampled, np.float64), 0.0, 1.0)
    k = s * s * (3.0 - 2.0 * s)
    d = np.asarray(noise_vec, np.float64) * (k * float(amp_px))[..., None]
    ln = np.linalg.norm(d, axis=-1, keepdims=True)
    lim = max(float(amp_px), 0.0)
    return np.where(ln > lim, d * (lim / np.maximum(ln, 1e-12)), d)


def flow_phases(t: float, settings: ShimmerSettings):
    """The two-phase flow map at VFX noise time ``t``: ``(ph0, ph1, w0,
    offsets (4,), boil dirs (4,))`` -- the phases and fresh per-cycle grain
    offsets of ``fire_vfx.noise_phases`` (in noise-texture units), layer 0's
    crossfade weight (Vlachos: 0 when layer 0 resets), and per-cycle boil
    directions (unit vectors) from the same hash."""
    from .fire_vfx import _hash01, noise_phases
    ph0, ph1, offs = noise_phases(t, settings.cycle_s, 1.0, settings.seed)
    w0 = 0.5 - 0.5 * math.cos(2.0 * math.pi * ph0)
    c = max(float(settings.cycle_s), 1e-3)
    dirs = []
    for k in (0, 1):
        idx = (int(math.floor(float(t) / c + 0.5 * k)) * 2 + k) & 0x7FFFFFFF
        a = 2.0 * math.pi * _hash01(idx, int(settings.seed) + 131)
        dirs += [math.cos(a), math.sin(a)]
    return ph0, ph1, w0, offs, tuple(dirs)


# ---------------------------------------------------------------------------
# The GL face (the resolve binds it)
# ---------------------------------------------------------------------------

class HeatShimmer:
    """The shimmer's GPU inputs: the per-tile heat texture (r = strength,
    gb = the tamed wind in tiles/s, capped) and the noise texture, refreshed
    by :meth:`update` once per changed sim state; the resolve reads them
    through :meth:`bind` / :meth:`unbind` (``WorldComposite.resolve``)."""

    UNIFORMS = ("u_shimmer_on", "u_shimmer_heat", "u_shimmer_noise",
                "u_shimmer_tiles", "u_shimmer_a", "u_shimmer_ph",
                "u_shimmer_off", "u_shimmer_boil", "u_shimmer_box")

    def __init__(self, grid_h: int, grid_w: int, settings: ShimmerSettings):
        import pyray as rl
        from . import core
        from .advected_noise import bake_fbm_rgba
        self._rl, self._core = rl, core
        self._gl = core.GLDirect.get()
        self.h, self.w = int(grid_h), int(grid_w)
        self.settings = settings
        self.enabled_override: Optional[bool] = None
        self.heat_tex = core.create_dynamic_rgba32f_texture(self.w, self.h)
        rl.set_texture_filter(self.heat_tex,
                              rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        self.noise_tex = core.create_dynamic_rgba_texture(NOISE_SIZE,
                                                          NOISE_SIZE)
        core.update_rgba_texture(self.noise_tex,
                                 bake_fbm_rgba(NOISE_SIZE, octaves=4))
        rl.set_texture_filter(self.noise_tex,
                              rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        rl.set_texture_wrap(self.noise_tex,
                            rl.TextureWrap.TEXTURE_WRAP_REPEAT)
        self._key = None
        self.hot = False             # any tile with strength > 0 this frame
        self.box = (0.0, 0.0, 0.0, 0.0)  # their bbox + 1 tile (x0, y0, x1, y1)
        self.t_noise = 0.0
        self.zoom_px_per_tile = 0.0
        self.last_update_ms = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_override if self.enabled_override is not None
                    else self.settings.enabled)

    def toggle(self) -> bool:
        """The live on/off key (G3 F4a, HUMAN-TEST 2)."""
        self.enabled_override = not self.enabled
        return self.enabled

    def refresh_settings(self, settings: ShimmerSettings) -> None:
        if settings != self.settings:
            self._key = None
        self.settings = settings

    def update(self, *, t_kelvin_fn: Callable[[], np.ndarray], t_game,
               solid, vacuum, wind_fn: Callable[[], np.ndarray], tps: float,
               t_noise: float, zoom_px_per_tile: float,
               sim_tick: Optional[int] = None, paused: bool = True) -> None:
        """Per frame: keep the VFX clock and the zoom; re-pack the heat
        texture only when the sim state changed -- once per sim tick while
        running (``sim_tick``), CRC-keyed on the temperature and masks while
        paused or without a tick (so a paused dev edit shows too).
        ``t_kelvin_fn`` maps the read-only game T to Kelvin through the
        canonical scale; ``wind_fn`` is the frame's ONE tamed wind (tiles per
        tick), asked only when some gas is hot."""
        import time
        t0 = time.perf_counter()
        self.t_noise = float(t_noise)
        self.zoom_px_per_tile = float(zoom_px_per_tile)
        if not self.enabled:
            self.last_update_ms = (time.perf_counter() - t0) * 1000.0
            return
        if sim_tick is not None and not paused:
            key = ("tick", int(sim_tick))
        else:
            key = ("crc",
                   zlib.crc32(np.ascontiguousarray(t_game).view(np.uint8)),
                   zlib.crc32(np.ascontiguousarray(solid).view(np.uint8)),
                   zlib.crc32(np.ascontiguousarray(vacuum).view(np.uint8)))
        if key != self._key:
            self._key = key
            s = self.settings
            gas = ~np.asarray(solid, bool) & ~np.asarray(vacuum, bool)
            st = shimmer_strength(t_kelvin_fn(), gas, s.t0_k, s.t1_k)
            hot = st > 0.0
            self.hot = bool(hot.any())
            # the hot tiles' bounding box, dilated by the bilinear footprint
            # (one tile): the resolve skips every pixel outside it before
            # any fetch, so the cost scales with the hot area, not the map
            self.box = (0.0, 0.0, 0.0, 0.0)
            if self.hot:
                rows = np.flatnonzero(hot.any(axis=1))
                cols = np.flatnonzero(hot.any(axis=0))
                self.box = (float(max(cols[0] - 1, 0)),
                            float(max(rows[0] - 1, 0)),
                            float(min(cols[-1] + 2, self.w)),
                            float(min(rows[-1] + 2, self.h)))
            pack = np.zeros((self.h, self.w, 4), np.float32)
            pack[..., 0] = st
            if self.hot:
                wd = np.asarray(wind_fn(), np.float64) * float(tps)
                sp = np.linalg.norm(wd, axis=-1, keepdims=True)
                wd = np.where(sp > SPEED_CAP_TILES_S,
                              wd * (SPEED_CAP_TILES_S / np.maximum(sp, 1e-12)),
                              wd)
                pack[..., 1:3] = wd
            self._core.update_rgba32f_texture(self.heat_tex, pack)
        self.last_update_ms = (time.perf_counter() - t0) * 1000.0

    def active(self, px_per_tile: float) -> bool:
        """Whether the resolve's shimmer term runs this frame (on, some gas
        hot, zoomed in enough)."""
        s = self.settings
        return (self.enabled and self.hot and s.amp_tiles > 0.0
                and zoom_fade(self.zoom_px_per_tile, s.zoom_zero,
                              s.zoom_full) > 0.0)

    def uniforms(self, px_per_tile: float):
        """``{name: values}`` of the resolve's shimmer uniforms this frame
        (samplers excluded)."""
        s = self.settings
        on = self.active(px_per_tile)
        amp = (s.amp_tiles * float(px_per_tile)
               * zoom_fade(self.zoom_px_per_tile, s.zoom_zero, s.zoom_full))
        ph0, ph1, w0, offs, dirs = flow_phases(self.t_noise, s)
        return {"u_shimmer_on": (int(on),),
                "u_shimmer_tiles": (float(self.w), float(self.h)),
                "u_shimmer_a": (amp, s.grain_tiles, float(px_per_tile),
                                s.cycle_s),
                "u_shimmer_ph": (ph0, ph1, w0, s.boil_tiles_s),
                "u_shimmer_off": tuple(float(o) for o in offs),
                "u_shimmer_boil": tuple(float(d) for d in dirs),
                "u_shimmer_box": tuple(float(b) for b in self.box)}

    def bind(self) -> None:
        gl = self._gl
        gl.ActiveTexture(_GL_TEXTURE0 + _UNIT_HEAT)
        gl.BindTexture(_GL_TEXTURE_2D, int(self.heat_tex.id))
        gl.ActiveTexture(_GL_TEXTURE0 + _UNIT_NOISE)
        gl.BindTexture(_GL_TEXTURE_2D, int(self.noise_tex.id))
        gl.ActiveTexture(_GL_TEXTURE0)

    def unbind(self) -> None:
        gl = self._gl
        for unit in (_UNIT_HEAT, _UNIT_NOISE):
            gl.ActiveTexture(_GL_TEXTURE0 + unit)
            gl.BindTexture(_GL_TEXTURE_2D, 0)
        gl.ActiveTexture(_GL_TEXTURE0)

    def unload(self) -> None:
        self._rl.unload_texture(self.heat_tex)
        self._rl.unload_texture(self.noise_tex)


SAMPLER_UNITS = {"u_shimmer_heat": _UNIT_HEAT, "u_shimmer_noise": _UNIT_NOISE}

__all__ = [
    "ShimmerSettings", "settings_from_config", "config_keys",
    "shimmer_strength", "zoom_fade", "bilinear_tiles", "shimmer_offset_px",
    "flow_phases", "HeatShimmer", "SAMPLER_UNITS", "NOISE_SIZE",
]
