"""THE fine water surface (water look arc #97, patch W1;
docs/water_look_design_2026-11-19.md W1 + its critique §9): the law's numpy
TWIN, the ``[graphics.water]`` settings door and the per-tick packing. Pure
numpy, pyray-free: ``shaders/water.fs`` mirrors :func:`detail_slope` and is
GL-checked against it (``tests/gl_water_flow_check.py``); the properties are
asserted here (``tests/test_water_flow_detail.py``).

WHAT IT IS. One small repeating texture of fine wave SLOPES
(``noise_bake.bake_ripple_rgba``: a sum of plane waves, RMS slope 1), laid on
the water at ``detail_wavelength_m`` metres per dominant wave -- a size in
metres, independent of the tile (the answer to #97: the old ``ambientSine``
waves were sized in tiles and read "far too big") -- and CARRIED BY THE
SIMULATED FLOW (``gmap.flow_vx`` / ``flow_vy``, m/s, dequantized at the
render read).

    slope(p, t) = k(p) * mix(n1, n0, w0)          (a slope vector, y south)
    n_i         = hex(q_i)                         (hex-tiled lookup)
    q_i         = (p - slide(v, age_i)) / P - drift(t)
    slide       = v * age_i,  |slide| <= flow_max_offset_m
    v           = flow(p) * smoothstep(0, flow_edge_depth_m, depth)
    (age_i, w0) = advected_noise.layer_ages_weight(phase(t), off(p))
    k           = (still + (flow - still) * sat) * zoom_fade
    sat         = clamp(max(|v| / flow_speed_ref, |ripple_v| / agitation_ref))

  * THE FLOW (Vlachos 2010): two copies of the texture slid along the local
    flow, half a cycle apart, cross-faded so each one's periodic reset sits
    under zero weight. The crossfade law is ``renderer/advected_noise.py``'s
    (the one shared with the gas detail, the gas relief and the speckle),
    extended there with the sub-tick ``alpha`` and the per-point phase
    offset ``phase_offset_field`` -- a low-frequency function of position,
    so the reset never shows as a pulse of the whole surface. Its GLSL face
    is spliced into the shader (``advected_noise.ADVECTION_GLSL``).
  * THE CAP (critique M7): a layer's slide is clamped to
    ``flow_max_offset_m``, so a fast flood stretches the texture at most that
    far before its layer resets; the flow fades to 0 over the first
    ``flow_edge_depth_m`` of depth so the wet/dry edge does not tear.
  * STILL WATER: no flow, no slide -- both layers read the same point and
    the surface is time-steady but for a slow uniform drift
    (``detail_drift_mps`` along ``DRIFT_DIR``) at the faint
    ``detail_strength_still``.
  * NO REPETITION (Erik, 2026-11-20): every lookup is HEX-TILED (Mikkelsen
    2022): three taps on a hexagonal lattice, each with a hashed offset and
    rotation (the sampled slope VECTOR is rotated back with its tap), blended
    by the paper's contrast-preserving weights ``|s|``-luminance x ``w^7``.
    The lattice lives in texture space, so it travels with the flow and the
    drift; the hash is an integer one (``hex_hash``), the same bits on the
    GPU and here.
  * ZOOM FADE (critique M8): ``k`` fades to 0 below ``detail_zoom_zero``
    screen px per tile (``heat_shimmer.zoom_fade``, the one fade); the
    texture is mipmapped, so minified slopes average toward flat.

THE CLOCK. The SIM clock ``(tick + alpha) / tps`` through a
``fire_vfx.SimClock`` the pass owns (idle when paused, re-anchored when the
tick goes back), never wall time (critique B2). :func:`flow_frame` turns it
into the frame's bounded uniforms in float64.

Credits (repo rule -- cite what a file implements):
  - Alex Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010 course "Advances
    in Real-Time Rendering in 3D Graphics and Games" -- the two-phase flow
    map (archived: docs/papers/vlachos2010_water_flow_portal2.pdf).
  - Morten S. Mikkelsen, "Practical Real-Time Hex-Tiling", Journal of
    Computer Graphics Techniques (JCGT) 11(3), 77-94, 2022 -- the hex-tiled
    lookup and its weights (reference: docs/papers/README_water_2026-11.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import NamedTuple, Optional, Tuple

import numpy as np

from .advected_noise import (advection_phase, layer_ages_weight_field,
                             phase_offset_field)
from .noise_bake import RIPPLE_CELLS

# The drift's direction (a unit vector, y south): still water creeps this way.
DRIFT_DIR = (0.6, 0.8)

# Mikkelsen 2022's constants (his reference shader's defaults): the lattice
# scale (a triangle edge is 1 / (2 sqrt 3) ~ 0.29 texture periods), the
# weight exponent, the share of the luminance weight, the rotation strength.
HEX_SCALE = 2.0 * math.sqrt(3.0)
HEX_EXP = 7.0
HEX_FALLOFF_CONTRAST = 0.6
HEX_ROT_STRENGTH = 1.0
_SKEW = 0.57735027                     # the reference's literals, as written
_SKEW_Y = 1.15470054
_SALT_U, _SALT_V, _SALT_ROT = 1, 2, 3
_M32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Settings: ONE refusing door, read every frame (critique M6)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaterSettings:
    """Every ``[graphics.water]`` dial. Values come from config.toml only:
    there are no defaults here (a missing key is refused by name)."""
    roughness_base: float
    roughness_agitation: float
    fog_density: float
    refract_strength: float
    r0: float
    ripple_scale: float
    water_color: Tuple[float, float, float]
    glint_strength: float
    alpha_scale: float
    alpha_min: float
    alpha_max: float
    caustic_strength: float
    caustic_scale: float
    foam_threshold: float
    foam_intensity: float
    ca_amount: float
    height_scale: float
    height_edge: float
    height_floor: float
    # W1: the fine surface that flows
    detail_wavelength_m: float
    detail_strength_still: float
    detail_strength_flow: float
    detail_drift_mps: float
    detail_zoom_zero: float
    detail_zoom_full: float
    flow_cycle_s: float
    flow_speed_ref: float
    flow_max_offset_m: float
    flow_edge_depth_m: float
    flow_phase_noise_m: float
    agitation_ref: float
    debug_view: int

    @property
    def period_m(self) -> float:
        """The ripple texture's period in metres."""
        return self.detail_wavelength_m * RIPPLE_CELLS


_INT_KEYS = ("debug_view",)
_VEC3_KEYS = ("water_color",)
DEBUG_VIEWS = (0, 1)        # 0 = the picture; 1 = (slope x, slope y, k, 1)


def config_keys() -> Tuple[str, ...]:
    return tuple(f.name for f in fields(WaterSettings))


def water_settings(cfg) -> WaterSettings:
    """The ``[graphics.water]`` settings of a loaded config (``CFG``), read
    EVERY FRAME (F5 and the lighting demo's sliders retune live). A REFUSING
    door: a missing section or key, a mistyped value or one out of range
    raises ValueError naming it."""
    sec = getattr(getattr(cfg, "graphics", None), "water", None)
    if sec is None:
        raise ValueError("config.toml has no [graphics.water] section")
    vals = {}
    for name in config_keys():
        if not hasattr(sec, name):
            raise ValueError(f"[graphics.water] is missing {name!r}")
        v = getattr(sec, name)
        if name in _INT_KEYS:
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif name in _VEC3_KEYS:
            ok = (isinstance(v, (list, tuple)) and len(v) == 3
                  and all(isinstance(c, (int, float))
                          and not isinstance(c, bool) for c in v))
            v = tuple(float(c) for c in v) if ok else v
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
            v = float(v) if ok else v
        if not ok:
            raise ValueError(f"[graphics.water] {name} has the wrong type: "
                             f"{v!r}")
        vals[name] = v
    s = WaterSettings(**vals)
    for name in ("detail_wavelength_m", "flow_cycle_s", "flow_speed_ref",
                 "flow_edge_depth_m", "agitation_ref"):
        if getattr(s, name) <= 0.0:
            raise ValueError(f"[graphics.water] {name} must be > 0")
    for name in ("detail_strength_still", "detail_strength_flow",
                 "detail_drift_mps", "flow_max_offset_m",
                 "flow_phase_noise_m"):
        if getattr(s, name) < 0.0:
            raise ValueError(f"[graphics.water] {name} must be >= 0")
    if s.detail_zoom_full <= s.detail_zoom_zero:
        raise ValueError("[graphics.water] detail_zoom_full must exceed "
                         "detail_zoom_zero")
    if s.debug_view not in DEBUG_VIEWS:
        raise ValueError(f"[graphics.water] debug_view must be one of "
                         f"{DEBUG_VIEWS}: {s.debug_view}")
    return s


# ---------------------------------------------------------------------------
# The frame's uniforms (float64 on the CPU; the GPU sees bounded numbers)
# ---------------------------------------------------------------------------

class FlowFrame(NamedTuple):
    phase: float        # the crossfade phase in [0, 1)
    cycle_s: float      # one cycle, seconds (a layer's age is age_frac x it)
    drift_u: float      # the drift, texture periods (unbounded, see below)
    drift_v: float
    fade: float         # the zoom fade in [0, 1]


def flow_frame(t_s: float, tps: float, zoom_px_per_tile: float,
               s: WaterSettings) -> FlowFrame:
    """The frame's flow uniforms at sim-clock time ``t_s`` (seconds,
    ``SimClock.t_noise``). The phase is ``advected_noise.advection_phase``
    with the clock split into its tick and its sub-tick alpha. The drift is
    NOT wrapped: the hex lattice is not periodic in texture space, so a wrap
    would pop it; in float32 a drift of 0.03 m/s over a 1.9 m period keeps
    sub-texel precision for days of play."""
    from .heat_shimmer import zoom_fade
    ticks = float(t_s) * float(tps)
    k = math.floor(ticks)
    ph = advection_phase(k, s.flow_cycle_s, tps, alpha=ticks - k)
    cycle_s = ph.tau_ticks / float(tps)
    d = float(t_s) * s.detail_drift_mps / s.period_m
    return FlowFrame(ph.phase, cycle_s, d * DRIFT_DIR[0], d * DRIFT_DIR[1],
                     zoom_fade(zoom_px_per_tile, s.detail_zoom_zero,
                               s.detail_zoom_full))


# ---------------------------------------------------------------------------
# The hex-tiled lookup (Mikkelsen 2022)
# ---------------------------------------------------------------------------

def hex_hash(vx, vy, salt: int) -> np.ndarray:
    """A 32-bit integer hash of a lattice vertex (two's-complement ints) and
    a salt -> uint32: the key ``vx * 0x8da6b343 ^ vy * 0xd8163841 ^ salt *
    0xcb1ab31f`` through the lowbias32 finalizer (C. Wellons). The shader's
    ``hexHash`` is the same bits."""
    x = ((np.asarray(vx, np.int64) & _M32) * 0x8DA6B343) & _M32
    y = ((np.asarray(vy, np.int64) & _M32) * 0xD8163841) & _M32
    z = x ^ y ^ ((int(salt) * 0xCB1AB31F) & _M32)
    z ^= z >> 16
    z = (z * 0x7FEB352D) & _M32
    z ^= z >> 15
    z = (z * 0x846CA68B) & _M32
    z ^= z >> 16
    return z


def hex_hash01(vx, vy, salt: int) -> np.ndarray:
    """:func:`hex_hash` -> [0, 1) on a 2^-24 grid (exact in float32)."""
    return (hex_hash(vx, vy, salt) >> 8).astype(np.float64) / 16777216.0


def triangle_grid(st: np.ndarray):
    """Mikkelsen's ``TriangleGrid``: the barycentric weights (..., 3) and
    the three lattice vertices (..., 2) int of the triangle holding ``st``
    (..., 2) texture units."""
    st = np.asarray(st, np.float64) * HEX_SCALE
    sx = st[..., 0] - _SKEW * st[..., 1]
    sy = _SKEW_Y * st[..., 1]
    bx, by = np.floor(sx), np.floor(sy)
    fx, fy = sx - bx, sy - by
    fz = 1.0 - fx - fy
    s = (-fz >= 0.0).astype(np.float64)
    s2 = 2.0 * s - 1.0
    w = np.stack([-fz * s2, s - fy * s2, s - fx * s2], axis=-1)
    bx, by, si = bx.astype(np.int64), by.astype(np.int64), s.astype(np.int64)
    v1 = np.stack([bx + si, by + si], axis=-1)
    v2 = np.stack([bx + si, by + 1 - si], axis=-1)
    v3 = np.stack([bx + 1 - si, by + si], axis=-1)
    return w, (v1, v2, v3)


def hex_center(v: np.ndarray) -> np.ndarray:
    """A lattice vertex's position in texture units (``MakeCenST``)."""
    vx = v[..., 0].astype(np.float64)
    vy = v[..., 1].astype(np.float64)
    return np.stack([vx + 0.5 * vy, vy / _SKEW_Y], axis=-1) / HEX_SCALE


def sample_bilinear_repeat(tex: np.ndarray, st: np.ndarray) -> np.ndarray:
    """The GPU's bilinear lookup of a REPEAT texture at level 0 (texel
    centres at (i + .5) / n), ``tex`` (n, n, c) row = v."""
    n_v, n_u = tex.shape[:2]
    fx = np.asarray(st, np.float64)[..., 0] * n_u - 0.5
    fy = np.asarray(st, np.float64)[..., 1] * n_v - 0.5
    x0, y0 = np.floor(fx), np.floor(fy)
    ax, ay = (fx - x0)[..., None], (fy - y0)[..., None]
    x0i = np.mod(x0.astype(np.int64), n_u)
    y0i = np.mod(y0.astype(np.int64), n_v)
    x1i, y1i = np.mod(x0i + 1, n_u), np.mod(y0i + 1, n_v)
    t = np.asarray(tex, np.float64)
    top = t[y0i, x0i] * (1 - ax) + t[y0i, x1i] * ax
    bot = t[y1i, x0i] * (1 - ax) + t[y1i, x1i] * ax
    return top * (1 - ay) + bot * ay


def hex_slope(tex: np.ndarray, st: np.ndarray) -> np.ndarray:
    """The hex-tiled slope at ``st`` (..., 2) texture units -> (..., 2): each
    of the three taps reads the texture at its vertex's rotation about the
    vertex centre plus a hashed offset, rotates the sampled slope back
    (the gradient of a rotated field), and the taps are blended by
    ``(1 - c + c |s_i|) w_i^7``, normalised (Mikkelsen's ``hex2colTex`` with
    the slope length as its luminance)."""
    st = np.asarray(st, np.float64)
    w, verts = triangle_grid(st)
    acc = np.zeros(st.shape)
    wsum = np.zeros(st.shape[:-1])
    for i, v in enumerate(verts):
        cen = hex_center(v)
        th = (2.0 * np.pi * HEX_ROT_STRENGTH
              * (hex_hash01(v[..., 0], v[..., 1], _SALT_ROT) - 0.5))
        c, s = np.cos(th), np.sin(th)
        d = st - cen
        rx = c * d[..., 0] - s * d[..., 1]
        ry = s * d[..., 0] + c * d[..., 1]
        q = np.stack([rx + cen[..., 0]
                      + hex_hash01(v[..., 0], v[..., 1], _SALT_U),
                      ry + cen[..., 1]
                      + hex_hash01(v[..., 0], v[..., 1], _SALT_V)], axis=-1)
        sm = sample_bilinear_repeat(tex, q)
        # d/dst f(R st) = R^T grad f: rotate the sampled slope back
        gx = c * sm[..., 0] + s * sm[..., 1]
        gy = -s * sm[..., 0] + c * sm[..., 1]
        lum = np.sqrt(sm[..., 0] ** 2 + sm[..., 1] ** 2)
        wi = ((1.0 - HEX_FALLOFF_CONTRAST + HEX_FALLOFF_CONTRAST * lum)
              * w[..., i] ** HEX_EXP)
        acc[..., 0] += wi * gx
        acc[..., 1] += wi * gy
        wsum += wi
    return acc / wsum[..., None]


def plain_slope(tex: np.ndarray, st: np.ndarray) -> np.ndarray:
    """The texture's slope at ``st`` WITHOUT hex-tiling (one bilinear tap):
    the repeating reference the anti-tiling property is measured against."""
    return sample_bilinear_repeat(tex, st)[..., :2]


# ---------------------------------------------------------------------------
# The law
# ---------------------------------------------------------------------------

def smoothstep(e0: float, e1: float, x) -> np.ndarray:
    t = np.clip((np.asarray(x, np.float64) - e0) / max(e1 - e0, 1e-12),
                0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def flow_slide(vel: np.ndarray, age_s, max_offset_m: float) -> np.ndarray:
    """A layer's slide in metres: ``vel * age_s`` (vel (..., 2) m/s), its
    length clamped to ``max_offset_m`` (critique M7). Odd in ``vel``."""
    d = np.asarray(vel, np.float64) * np.asarray(age_s, np.float64)[..., None]
    ln = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2)
    cap = float(max_offset_m)
    k = np.where(ln > cap, cap / np.maximum(ln, 1e-30), 1.0)
    return d * k[..., None]


def layer_coords(x_m, y_m, vel, depth, frame: FlowFrame, s: WaterSettings):
    """The two layers' texture coordinates (..., 2), their crossfade weight
    ``w0`` and the effective flow ``v`` (depth-faded) -- the geometry half of
    :func:`detail_slope`, exposed for the properties."""
    x = np.asarray(x_m, np.float64)
    y = np.asarray(y_m, np.float64)
    edge = smoothstep(0.0, s.flow_edge_depth_m, depth)
    v = np.asarray(vel, np.float64) * edge[..., None]
    off = phase_offset_field(x, y, s.flow_phase_noise_m)
    a0, a1, w0 = layer_ages_weight_field(frame.phase, off)
    P = s.period_m
    out = []
    for age in (a0, a1):
        d = flow_slide(v, age * frame.cycle_s, s.flow_max_offset_m)
        out.append(np.stack([(x - d[..., 0]) / P - frame.drift_u,
                             (y - d[..., 1]) / P - frame.drift_v], axis=-1))
    return out[0], out[1], w0, v


def strength(v: np.ndarray, agitation, frame: FlowFrame,
             s: WaterSettings) -> np.ndarray:
    """``k``: the RMS slope of the fine surface here."""
    speed = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)
    sat = np.clip(np.maximum(speed / s.flow_speed_ref,
                             np.abs(np.asarray(agitation, np.float64))
                             / s.agitation_ref), 0.0, 1.0)
    return ((s.detail_strength_still
             + (s.detail_strength_flow - s.detail_strength_still) * sat)
            * frame.fade)


def detail_slope(x_m, y_m, vel, depth, agitation, frame: FlowFrame,
                 tex: np.ndarray, s: WaterSettings,
                 lookup=hex_slope) -> np.ndarray:
    """THE fine surface's slope (dh/dx, dh/dy; y south) at world positions
    in metres: the module docstring's law. ``vel`` (..., 2) the bilinear flow
    in m/s, ``depth`` m, ``agitation`` = |ripple_v| m/s, ``tex`` the ripple
    bake. ``lookup`` is the hex-tiled lookup (``plain_slope`` only for the
    anti-tiling property's reference)."""
    q0, q1, w0, v = layer_coords(x_m, y_m, vel, depth, frame, s)
    n0 = lookup(tex, q0)
    n1 = lookup(tex, q1)
    n = n1 * (1.0 - w0)[..., None] + n0 * w0[..., None]   # mix(n1, n0, w0)
    return n * strength(v, agitation, frame, s)[..., None]


# ---------------------------------------------------------------------------
# The per-tick packing (critique M4, M11)
# ---------------------------------------------------------------------------

def wet_box(depth_q: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """The wet tiles' bounding box dilated by one tile (the bilinear
    footprint), ``(x0, y0, x1, y1)`` end-exclusive and clamped, or None when
    dry. Reads the INT plane (no dequantize)."""
    wet = np.asarray(depth_q) > 0
    rows = np.flatnonzero(wet.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(wet.any(axis=0))
    h, w = wet.shape
    return (max(int(cols[0]) - 1, 0), max(int(rows[0]) - 1, 0),
            min(int(cols[-1]) + 2, w), min(int(rows[-1]) + 2, h))


def pack_flow(vx: np.ndarray, vy: np.ndarray, solid: np.ndarray,
              prev_depth: np.ndarray, out: np.ndarray) -> np.ndarray:
    """The second grid texture, RGBA16F (critique M11): ``(vx, vy, solid,
    previous depth)`` -- the flow in m/s (dequantized by the caller through
    ``water_fixed.dequantize_f32``), 1 on solid tiles, the depth (m) at the
    previous packed tick (W2's advancing-edge source). Written into ``out``
    (h, w, 4) float16, returned."""
    out[..., 0] = vx
    out[..., 1] = vy
    out[..., 2] = np.asarray(solid, bool)
    out[..., 3] = prev_depth
    return out


__all__ = [
    "WaterSettings", "water_settings", "config_keys", "DEBUG_VIEWS",
    "FlowFrame", "flow_frame", "DRIFT_DIR",
    "hex_hash", "hex_hash01", "triangle_grid", "hex_center",
    "sample_bilinear_repeat", "hex_slope", "plain_slope",
    "smoothstep", "flow_slide", "layer_coords", "strength", "detail_slope",
    "wet_box", "pack_flow",
]
