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
    sat         = clamp(max(|v| / flow_speed_ref, a / agitation_ref))
    a           = relative_agitation (ripple_v against its wet neighbours')

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

THE CAUSTIC (W4). The same surface's CURVATURE (a companion bake of the
same waves' exact second derivatives, read through the same hex-tiled,
flow-mapped taps: :func:`hex_surface`, :func:`detail_surface`) gives the
Jacobian of the refraction map ``x -> x + depth (1 - 1/n) grad h``;
the floor's source light is multiplied by ``min(1/|det J|, caustic_max)``
(signed: :func:`caustic_det`, :func:`caustic_gain`, :func:`caustic_term`;
``shaders/water.fs`` mirrors it, ``tests/gl_water_caustic_check.py`` holds
the two equal).

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
  - Evan Wallace, "WebGL Water" (2011, MIT; madebyevan.com/webgl-water) --
    the caustic's area method (a patch of light's area before and after the
    surface bends it), here as its analytic limit, the Jacobian determinant
    (archived: docs/papers/wallace2011_webgl_water_renderer.js).
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
    caustic_max: float
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
    # W2: foam with a life (renderer/water_foam.py)
    foam_cells_per_tile: int
    foam_step_s: float
    foam_decay_s: float
    foam_froude: float
    foam_gain_froude: float
    foam_gain_steep: float
    foam_gain_edge: float
    foam_gain_wall: float
    foam_bubble_m: Tuple[float, float]
    foam_cover: float
    foam_shadow: float
    # W3: things in the water (the waterline: :func:`waterline_tint` and
    # lit3d._WATERLINE_GLSL; the ring, the wake and the splash dab: the foam
    # splats, water_foam.splat_rate / renderer/water_wake.py)
    waterline_strength: float
    wake_body_m: float
    wake_band_m: float
    wake_ring_rate: float
    wake_gain: float
    splash_foam: float
    splash_radius_m: float
    debug_view: int

    @property
    def period_m(self) -> float:
        """The ripple texture's period in metres."""
        return self.detail_wavelength_m * RIPPLE_CELLS


_INT_KEYS = ("debug_view", "foam_cells_per_tile")
_VEC_KEYS = {"water_color": 3, "foam_bubble_m": 2}
# 0 = the picture; 1 = (slope x, slope y, k, 1); 2 = (foam amount, foam
# cover, 0, 1); 3 = (hxx, hxy, hyy, caustic gain) -- the caustic (W4)
DEBUG_VIEWS = (0, 1, 2, 3)
FOAM_CELLS_MAX = 8          # foam cells per tile (per axis), the door's top
FOAM_STEP_MAX_S = 0.1       # a foam step longer than this is refused


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
        elif name in _VEC_KEYS:
            ok = (isinstance(v, (list, tuple)) and len(v) == _VEC_KEYS[name]
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
    for name in ("foam_decay_s", "foam_cover"):
        if getattr(s, name) <= 0.0:
            raise ValueError(f"[graphics.water] {name} must be > 0")
    for name in ("foam_froude", "foam_gain_froude", "foam_gain_steep",
                 "foam_gain_edge", "foam_gain_wall"):
        if getattr(s, name) < 0.0:
            raise ValueError(f"[graphics.water] {name} must be >= 0")
    if not 1 <= s.foam_cells_per_tile <= FOAM_CELLS_MAX:
        raise ValueError(f"[graphics.water] foam_cells_per_tile must be in "
                         f"1..{FOAM_CELLS_MAX}: {s.foam_cells_per_tile}")
    if not 0.0 < s.foam_step_s <= FOAM_STEP_MAX_S:
        raise ValueError(f"[graphics.water] foam_step_s must be in "
                         f"(0, {FOAM_STEP_MAX_S}]: {s.foam_step_s}")
    if min(s.foam_bubble_m) <= 0.0:
        raise ValueError("[graphics.water] foam_bubble_m sizes must be > 0")
    if not 0.0 <= s.foam_shadow <= 1.0:
        raise ValueError(f"[graphics.water] foam_shadow must be in [0, 1]: "
                         f"{s.foam_shadow}")
    if s.caustic_strength < 0.0:
        raise ValueError("[graphics.water] caustic_strength must be >= 0")
    if s.caustic_max <= 1.0:
        raise ValueError(f"[graphics.water] caustic_max must be > 1 (the "
                         f"most a focus multiplies the light): "
                         f"{s.caustic_max}")
    if not 0.0 <= s.waterline_strength <= 1.0:
        raise ValueError(f"[graphics.water] waterline_strength must be in "
                         f"[0, 1]: {s.waterline_strength}")
    for name in ("wake_body_m", "wake_band_m", "splash_radius_m"):
        if getattr(s, name) <= 0.0:
            raise ValueError(f"[graphics.water] {name} must be > 0")
    for name in ("wake_ring_rate", "wake_gain", "splash_foam"):
        if getattr(s, name) < 0.0:
            raise ValueError(f"[graphics.water] {name} must be >= 0")
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
    return hex_surface(tex, None, st)[0]


def hex_surface(tex: np.ndarray, curv: Optional[np.ndarray],
                st: np.ndarray):
    """:func:`hex_slope` and, when ``curv`` (the curvature bake,
    ``noise_bake.bake_ripple_curvature_rgba``) is given, the surface's
    HESSIAN in texture units -> ``(slope (..., 2), hess (..., 3) = (hxx,
    hxy, hyy) or None)`` (W4).

    THE HEX BLEND AND THE ROTATION ON THE DERIVATIVES (derived for W4). A
    tap i reads the height field ``f`` at ``q' = R (st - c) + c + o``, so
    its own height field is ``h_i(st) = f(q')`` with gradient
    ``R^T grad f(q')`` (the slope rotated back, as W1) and Hessian
    ``H_i = R^T Hf(q') R`` -- the baked tensor ROTATED BACK on both indices.
    The blend applies the SAME normalised weights ``W_i`` (the slope taps'
    ``(1 - c + c |s_i|) w_i^7``) to the three Hessians: ``H = sum W_i H_i``.
    That is the derivative of the blended slope with the weights held
    fixed; the weights' own gradient (``grad W_i x (s_i - s)``, non-
    symmetric and confined to the narrow ``w^7`` seams) is dropped -- the
    blended slope is itself not the gradient of one height field, so the
    caustic treats the hex surface as a weighted mix of three honest
    surfaces, never as one field differentiated across its seams."""
    st = np.asarray(st, np.float64)
    w, verts = triangle_grid(st)
    acc = np.zeros(st.shape)
    hacc = None if curv is None else np.zeros(st.shape[:-1] + (3,))
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
        if hacc is not None:
            cm = sample_bilinear_repeat(curv, q)
            fuu, fuv, fvv = cm[..., 0], cm[..., 1], cm[..., 2]
            cc, cs, ss = c * c, c * s, s * s
            # R^T Hf R, R = [[c, -s], [s, c]]
            hacc[..., 0] += wi * (cc * fuu + 2.0 * cs * fuv + ss * fvv)
            hacc[..., 1] += wi * (-cs * fuu + (cc - ss) * fuv + cs * fvv)
            hacc[..., 2] += wi * (ss * fuu - 2.0 * cs * fuv + cc * fvv)
        wsum += wi
    return (acc / wsum[..., None],
            None if hacc is None else hacc / wsum[..., None])


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


def edge_flow(vel, depth, s: WaterSettings) -> np.ndarray:
    """The EFFECTIVE flow (..., 2) m/s: the bilinear flow faded to 0 over
    the first ``flow_edge_depth_m`` of depth (critique M7) -- the one flow
    the fine surface slides by and the foam (W2) is carried and made by."""
    edge = smoothstep(0.0, s.flow_edge_depth_m, depth)
    return np.asarray(vel, np.float64) * edge[..., None]


def layer_slides(x_m, y_m, vel, depth, frame: FlowFrame, s: WaterSettings):
    """THE two layers' slides ``(d0, d1)`` (..., 2) metres, their crossfade
    weight ``w0`` and the effective flow ``v`` -- the one flow map every
    surface term rides (the fine surface; W2's foam bubbles): never a second
    advector."""
    x = np.asarray(x_m, np.float64)
    y = np.asarray(y_m, np.float64)
    v = edge_flow(vel, depth, s)
    off = phase_offset_field(x, y, s.flow_phase_noise_m)
    a0, a1, w0 = layer_ages_weight_field(frame.phase, off)
    d0 = flow_slide(v, a0 * frame.cycle_s, s.flow_max_offset_m)
    d1 = flow_slide(v, a1 * frame.cycle_s, s.flow_max_offset_m)
    return d0, d1, w0, v


def drift_m(frame: FlowFrame, s: WaterSettings) -> Tuple[float, float]:
    """The still-water drift in METRES (the frame carries it in ripple
    periods)."""
    return frame.drift_u * s.period_m, frame.drift_v * s.period_m


def layer_coords(x_m, y_m, vel, depth, frame: FlowFrame, s: WaterSettings):
    """The two layers' texture coordinates (..., 2), their crossfade weight
    ``w0`` and the effective flow ``v`` (depth-faded) -- the geometry half of
    :func:`detail_slope`, exposed for the properties."""
    x = np.asarray(x_m, np.float64)
    y = np.asarray(y_m, np.float64)
    d0, d1, w0, v = layer_slides(x, y, vel, depth, frame, s)
    P = s.period_m
    out = []
    for d in (d0, d1):
        out.append(np.stack([(x - d[..., 0]) / P - frame.drift_u,
                             (y - d[..., 1]) / P - frame.drift_v], axis=-1))
    return out[0], out[1], w0, v


def relative_agitation(ripple_v: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """THE water's agitation, per tile (#97 W2; W1's fine-surface strength and
    W2's foam both read it, packed per tick in ``u_water``'s A): ``|ripple_v - mean of the
    wet 4-neighbours' ripple_v|`` (m/s) on a wet tile, 0 elsewhere and on a
    tile with no wet neighbour. Waves break where the surface moves against
    its neighbours, not where it heaves as one; a wall or a dry tile is no
    neighbour, so the water's edge against a wall makes none (the old
    gradient foam's wall bands). Per tick, at pack time."""
    rv = np.asarray(ripple_v, np.float64)
    wet = np.asarray(wet, bool)
    h, w = rv.shape
    acc = np.zeros((h, w))
    n = np.zeros((h, w))
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ys = slice(max(dy, 0), h + min(dy, 0))
        yd = slice(max(-dy, 0), h + min(-dy, 0))
        xs = slice(max(dx, 0), w + min(dx, 0))
        xd = slice(max(-dx, 0), w + min(-dx, 0))
        m = wet[ys, xs]
        acc[yd, xd] += np.where(m, rv[ys, xs], 0.0)
        n[yd, xd] += m
    out = np.abs(rv - acc / np.maximum(n, 1.0))
    return np.where(wet & (n > 0), out, 0.0)



def strength_ratio(v: np.ndarray, agitation, s: WaterSettings
                   ) -> np.ndarray:
    """The fine surface's DRIVE, unclamped: the flow speed over
    ``flow_speed_ref`` or the agitation (the RELATIVE one,
    :func:`relative_agitation`, bilinear) over ``agitation_ref``, whichever
    is more (1 = full strength). The sim's ripple field heaves whole rooms
    as one (after a blast anywhere every room's ``ripple_v`` jumps, then
    sits frozen near 0.04 m/s): read raw, still rooms ran at ~0.75 of full
    strength (W2's finding, 2026-11-20). W2's
    steepness foam source reads THIS term past 1 -- water rougher than the
    fine surface can show breaks -- never a slope of its own."""
    speed = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2)
    return np.maximum(speed / s.flow_speed_ref,
                      np.abs(np.asarray(agitation, np.float64))
                      / s.agitation_ref)


def strength_sat(v: np.ndarray, agitation, s: WaterSettings) -> np.ndarray:
    """``sat`` in [0, 1]: how far the fine surface is from still toward its
    full strength (:func:`strength_ratio`, clamped)."""
    return np.clip(strength_ratio(v, agitation, s), 0.0, 1.0)


def strength(v: np.ndarray, agitation, frame: FlowFrame,
             s: WaterSettings) -> np.ndarray:
    """``k``: the RMS slope of the fine surface here."""
    sat = strength_sat(v, agitation, s)
    return ((s.detail_strength_still
             + (s.detail_strength_flow - s.detail_strength_still) * sat)
            * frame.fade)


def detail_slope(x_m, y_m, vel, depth, agitation, frame: FlowFrame,
                 tex: np.ndarray, s: WaterSettings,
                 lookup=hex_slope) -> np.ndarray:
    """THE fine surface's slope (dh/dx, dh/dy; y south) at world positions
    in metres: the module docstring's law. ``vel`` (..., 2) the bilinear flow
    in m/s, ``depth`` m, ``agitation`` = the bilinear relative agitation
    (:func:`relative_agitation`, ``u_water``'s A) m/s, ``tex`` the ripple
    bake. ``lookup`` is the hex-tiled lookup (``plain_slope`` only for the
    anti-tiling property's reference)."""
    q0, q1, w0, v = layer_coords(x_m, y_m, vel, depth, frame, s)
    n0 = lookup(tex, q0)
    n1 = lookup(tex, q1)
    n = n1 * (1.0 - w0)[..., None] + n0 * w0[..., None]   # mix(n1, n0, w0)
    return n * strength(v, agitation, frame, s)[..., None]


def detail_surface(x_m, y_m, vel, depth, agitation, frame: FlowFrame,
                   tex: np.ndarray, curv: np.ndarray, s: WaterSettings):
    """THE fine surface's slope (as :func:`detail_slope`) AND its Hessian
    ``(hxx, hxy, hyy)`` in 1/m (W4's caustic) -> ``(slope (..., 2), hess
    (..., 3))``.

    THE FLOW ON THE DERIVATIVES (derived for W4). A layer reads the texture
    at ``q = (p - d(p)) / P - drift``; the slide ``d``, the strength ``k``
    and the crossfade weight ``w0`` vary over a tile or more (the bilinear
    tile flow, the phase noise, the agitation) while the waves vary over
    centimetres, so ``dq/dp = I / P`` and ``d slope / dp = k / P x (the
    texture-space Hessian)``: each layer's Hessian is its hex blend
    (:func:`hex_surface`) scaled by ``k / P``, and the two layers are
    cross-faded with the slope's own weight ``w0``. The dropped terms
    (``grad d``, ``grad k``, ``grad w0``) are smaller by about a wavelength
    over a tile."""
    q0, q1, w0, v = layer_coords(x_m, y_m, vel, depth, frame, s)
    n0, h0 = hex_surface(tex, curv, q0)
    n1, h1 = hex_surface(tex, curv, q1)
    k = strength(v, agitation, frame, s)[..., None]
    w = w0[..., None]
    slope = (n1 * (1.0 - w) + n0 * w) * k
    hess = (h1 * (1.0 - w) + h0 * w) * (k / s.period_m)
    return slope, hess


# ---------------------------------------------------------------------------
# The caustic (W4): the area method, analytic, in gather form
# ---------------------------------------------------------------------------

# Water's index of refraction (a constant of the medium, not a look dial)
# and the small-slope refraction law it gives: a vertical ray through a
# surface of slope ``g`` leaves at the horizontal rate ``(1 - 1/n) g``
# (Snell, to first order in the slope), so it lands on the floor ``depth``
# below, displaced by ``depth x CAUSTIC_K x g``.
WATER_IOR = 1.333
CAUSTIC_K = 1.0 - 1.0 / WATER_IOR


def caustic_det(hess: np.ndarray, depth) -> np.ndarray:
    """``det J`` of the refraction map ``X(x) = x + a grad h(x)``, ``a =
    depth x CAUSTIC_K``: the ratio of the floor area a patch of light lands
    on to the surface area it passed (Wallace's area method -- his ratio of
    a projected triangle's areas -- as the analytic Jacobian of the map).
    ``J = I + a H`` (H the Hessian), so ``det J = 1 + a (hxx + hyy) + a^2
    (hxx hyy - hxy^2)``, exact for that map; the only approximations are the
    small-slope refraction law and a vertical light (the light field's one
    elevation does not shear the pattern: accepted, v1)."""
    hs = np.asarray(hess, np.float64)
    a = np.asarray(depth, np.float64) * CAUSTIC_K
    hxx, hxy, hyy = hs[..., 0], hs[..., 1], hs[..., 2]
    return 1.0 + a * (hxx + hyy) + a * a * (hxx * hyy - hxy * hxy)


def caustic_gain(det, s: WaterSettings) -> np.ndarray:
    """The SIGNED caustic gain ``(min(1 / |det|, caustic_max) - 1) x
    caustic_strength``: negative where the light spreads (|det| > 1),
    positive where it focuses, never above ``(caustic_max - 1) x strength``,
    never below ``-strength``. ``|det|``: past a focus (det < 0) the beam is
    inverted, not dark, and carries ``E / |det|`` -- so the bright lines are
    the folds (det = 0) and the floor between them keeps its light (the
    brief's ``clamp(1/det, 0, max)`` would black out every folded patch).
    Exactly 0 on a flat surface (det == 1)."""
    d = np.abs(np.asarray(det, np.float64))
    g = 1.0 / np.maximum(d, 1.0 / s.caustic_max) - 1.0
    return g * s.caustic_strength


def caustic_term(light, det, s: WaterSettings) -> np.ndarray:
    """THE caustic: the light ADDED to (signed: taken from) the floor's
    source light where the fine surface focuses (spreads) it -- ``light``
    (..., 3) the light at the floor as the floor takes it (exposed,
    smoke-shadowed, curved) times :func:`caustic_gain`. No light, no
    caustic; the ambient is not refracted light and is never modulated.
    The picture is the floor's albedo times it, like any light."""
    return (np.asarray(light, np.float64)
            * caustic_gain(det, s)[..., None])


# ---------------------------------------------------------------------------
# The waterline (W3, critique M12): a body seen through the water above it
# ---------------------------------------------------------------------------

def waterline_column(depth, height_m) -> np.ndarray:
    """The water column (m) above a fragment at world ``height_m`` metres
    over the floor, where the water stands ``depth`` m deep: ``depth -
    height``, 0 at and above the surface (and on a dry tile)."""
    return np.maximum(np.asarray(depth, np.float64)
                      - np.asarray(height_m, np.float64), 0.0)


def waterline_weight(column, s: WaterSettings) -> np.ndarray:
    """How much of the water's volume colour a fragment under ``column`` m
    of water takes: the water pass's OWN alpha ramp over its own depth
    (``clamp(col x alpha_scale, alpha_min, alpha_max)``, softened over the
    first ``height_edge`` m as its lap edge) x ``waterline_strength``.
    Exactly 0 where the column is 0."""
    c = np.asarray(column, np.float64)
    a = np.clip(c * s.alpha_scale, s.alpha_min, s.alpha_max)
    a = a * smoothstep(0.0, max(s.height_edge, 1e-4), c)
    return np.where(c > 0.0, a * s.waterline_strength, 0.0)


def waterline_tint(lit, depth, height_m, s: WaterSettings) -> np.ndarray:
    """THE waterline (W3; ``lit3d._WATERLINE_GLSL`` mirrors it, per
    fragment, in WORLD height, so it holds under a tilted camera too): a
    fragment of a unit (or a prop) at ``height_m`` metres over the floor,
    lit ``lit`` (..., 3) linear, seen through the water column above it, by
    the water pass's own law for the floor -- the Beer-Lambert fade toward
    ``water_color`` at ``fog_density`` (``2^(-fog_density x col)``) and its
    alpha ramp (:func:`waterline_weight`):

        out = mix(lit, mix(water_color, lit, 2^(-fog x col)), weight(col))

    Above the surface (or on a dry tile) ``out`` IS ``lit``, bit for bit."""
    lit = np.asarray(lit, np.float64)
    col = waterline_column(depth, height_m)
    fog = np.exp2(-s.fog_density * col)[..., None]
    wc = np.asarray(s.water_color, np.float64)
    vol = wc + (lit - wc) * fog
    w = waterline_weight(col, s)[..., None]
    return np.where(col[..., None] > 0.0, lit + (vol - lit) * w, lit)


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
    "smoothstep", "flow_slide", "edge_flow", "layer_slides", "drift_m",
    "layer_coords", "relative_agitation", "strength_ratio", "strength_sat",
    "strength", "detail_slope", "hex_surface", "detail_surface",
    "WATER_IOR", "CAUSTIC_K", "caustic_det", "caustic_gain", "caustic_term",
    "waterline_column", "waterline_weight", "waterline_tint",
    "wet_box", "pack_flow",
]
