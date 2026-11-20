"""THE fine water surface (water look arc #97, patches W1 and W1b;
docs/water_look_design_2026-11-19.md W1, W1b + the critique §9): the law's
numpy TWIN, the ``[graphics.water]`` settings door and the per-tick
packing. Pure numpy, pyray-free: ``shaders/water.fs`` mirrors
:func:`detail_surface` and is GL-checked against it
(``tests/gl_water_flow_check.py``, ``tests/gl_water_caustic_check.py``);
the properties are asserted here (``tests/test_water_flow_detail.py``,
``tests/test_water_caustic.py``).

WHAT IT IS (W1b, 2026-11-20: Erik -- "those texture waves just don't look
as good as sines do, sines in x and y morph differently all the time"). A
PROCEDURAL sum of travelling, sharpened sines, in two layers through ONE
law: the SWELL (``swell_wavelength_m``, a dominant direction with a
spread, following the flow where the water flows) and the fine RIPPLES
(``detail_wavelength_m``, every direction). W1's baked ripple texture only
slid; travelling sines MORPH as they interfere, and that motion is what the
texture lost.

    slope(p, t) = sum over layers L of k_L(p) * mix(S_L(p - d1), S_L(p - d0), w0)
    S_L(q)      = hex blend over the 3 lattice taps v of the layer's wave set
                  turned by theta_v about the tap's centre c_v, shifted by o_v
    wave i      = h_i(psi) = (a / |k_i|) g_K(psi),
                  psi = (R_v k_i) . (q - c_v) + k_i . o_v + phi_i - omega_i t
    g_K         = 2 ((sin psi + 1) / 2)^K - 1        (Finch's sharpening)
    (d0, d1, w0)= advected_noise's two-phase crossfade of the flow's slide
    k_L         = (still_L + (flow_L - still_L) * sat) * zoom_fade
    K           = wave_sharpness + (wave_sharpness_flow - wave_sharpness) * sat
    sat         = clamp(max(|v| / flow_speed_ref, a / agitation_ref))
    a           = relative_agitation (ripple_v against its wet neighbours')

  * THE WAVE SET (:func:`wave_set`): ``wave_count`` sines per layer, each its
    own direction, wavelength and phase. The wavelengths spread over the
    layer's band at golden-ratio steps of the log wavelength, so any two are
    in an IRRATIONAL ratio and the sum never repeats exactly; each wave runs
    at its own speed from the dispersion of water at its wave number,
    ``omega = sqrt(g k + (sigma / rho) k^3)`` (capillary-gravity, deep
    water; ``wave_speed`` scales it, 1 = physical) -- longer waves faster,
    so the interference pattern keeps changing. The time enters only as
    each wave's phase ``phi_i - omega_i t`` wrapped to [0, 2 pi) in float64
    (:func:`wave_tables`): the GPU never sees absolute time.
  * THE SHAPE (Finch 2004): each sine is sharpened, ``g_K``, narrow crests
    and broad flat troughs; ``K = 1`` is the plain sine. Its derivatives are
    exact: with ``u = (sin psi + 1) / 2``, ``g' = K u^(K-1) cos psi`` and
    ``g'' = K u^(K-1) (2 (K - 1) (1 - u) - sin psi)`` (``cos^2 = 4 u (1-u)``
    folds the trough's ``u^(K-2)`` away). Agitation raises K toward
    ``wave_sharpness_flow``. The slope amplitude ``a`` makes the layer's
    RMS slope 1 at ``wave_sharpness`` (:func:`finch_rms`), so a strength IS
    the RMS slope of still water.
  * NO REPETITION (Mikkelsen 2022 on the wave set): a large hex lattice (a
    vertex every ``wave_cell_m``, one for both layers) in which each vertex
    turns each layer's whole set by its hashed angle about its centre and
    shifts it by a hashed offset (the layer's own salts);
    the three taps are blended by Mikkelsen's contrast-preserving weights
    ``(1 - c + c |s_i|) w_i^7`` (a tap under ``HEX_SKIP`` of the largest is
    not evaluated), with the integer ``hex_hash`` the GPU shares. The swell's
    vertices turn the set to the swell's DIRECTION at the vertex centre
    (``swell_dir_deg`` on still water, the flow's direction where it flows:
    :func:`swell_direction`), jittered by +-``swell_spread_deg / 2``.
  * THE FLOW (Vlachos 2010): the waves are evaluated at the ADVECTED point,
    ``p`` minus each of two slides along the local flow half a cycle apart,
    cross-faded so each one's reset sits under zero weight; the crossfade is
    ``renderer/advected_noise.py``'s, its GLSL face spliced into the shader.
    The slide is capped at ``flow_max_offset_m`` and the flow fades in over
    ``flow_edge_depth_m`` of depth (critique M7). On still water both slides
    are 0 and the sum is evaluated once.
  * ZOOM FADE (critique M8): ``k`` fades to 0 below ``detail_zoom_zero``
    screen px per tile (``heat_shimmer.zoom_fade``, the one fade).

THE CAUSTIC (W4). The Hessian of the same sum -- EXACT, the second
derivatives of the same sines (each tap's ``a g'' k' k'^T``), blended with
the slope's weights, cross-faded with ``w0``, x ``k_L``, summed over the
layers -- gives the Jacobian of the refraction map ``x -> x + depth (1 -
1/n) grad h``; the floor's source light is multiplied by ``min(1/|det J|,
caustic_max)`` (signed: :func:`caustic_det`, :func:`caustic_gain`,
:func:`caustic_term`). The weights' and the slide's own gradients (a tile or
more in scale, the seams of ``w^7``) are dropped: the surface is a blend of
honest wave sums, each differentiated exactly.

THE CLOCK. The SIM clock ``(tick + alpha) / tps`` through a
``fire_vfx.SimClock`` the pass owns (idle when paused, re-anchored when the
tick goes back), never wall time (critique B2). :func:`flow_frame` turns it
into the frame's bounded uniforms in float64.

Credits (repo rule -- cite what a file implements):
  - Mark Finch, "Effective Water Simulation from Physical Models", GPU Gems
    ch. 1 (2004) -- the sum of travelling sines, the sharpening exponent and
    the dispersion relation (reference: docs/papers/README_water_2026-11.md).
  - Alex Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010 course "Advances
    in Real-Time Rendering in 3D Graphics and Games" -- the two-phase flow
    map (archived: docs/papers/vlachos2010_water_flow_portal2.pdf).
  - Morten S. Mikkelsen, "Practical Real-Time Hex-Tiling", Journal of
    Computer Graphics Techniques (JCGT) 11(3), 77-94, 2022 -- the hex
    lattice and its weights, here over a wave set instead of a texture
    (reference: docs/papers/README_water_2026-11.md).
  - Evan Wallace, "WebGL Water" (2011, MIT; madebyevan.com/webgl-water) --
    the caustic's area method (a patch of light's area before and after the
    surface bends it), here as its analytic limit, the Jacobian determinant
    (archived: docs/papers/wallace2011_webgl_water_renderer.js).
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, fields
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np

from .advected_noise import (advection_phase, layer_ages_weight_field,
                             phase_offset_field)

# The drift's direction (a unit vector, y south): the foam's bubbles on still
# water creep this way (the waves travel by their own dispersion).
DRIFT_DIR = (0.6, 0.8)

# Mikkelsen 2022's constants (his reference shader's defaults): the lattice
# scale, the weight exponent, the share of the luminance weight.
HEX_SCALE = 2.0 * math.sqrt(3.0)
HEX_EXP = 7.0
HEX_FALLOFF_CONTRAST = 0.6
# A tap whose w^7 is under this share of the largest is not evaluated (its
# weight is 0): most points take one or two taps, never a visible step.
HEX_SKIP = 1e-2
_SKEW = 0.57735027                     # the reference's literals, as written
_SKEW_Y = 1.15470054
_SALT_U, _SALT_V, _SALT_ROT = 1, 2, 3
_M32 = 0xFFFFFFFF

# The physics of the waves (constants of water, not look dials).
G_ACCEL = 9.81                         # m/s^2
SIGMA_OVER_RHO = 7.28e-5               # surface tension / density, m^3/s^2
MAX_WAVES = 16                         # per layer: the shader's arrays
LAYER_DETAIL, LAYER_SWELL = 0, 1
LAYERS = (LAYER_DETAIL, LAYER_SWELL)
# The bands of wavelengths, as multiples of the layer's dial: ripples cover
# a factor ~2.7, the swell a narrow factor ~1.6.
DETAIL_BAND = (0.6, 1.6)
SWELL_BAND = (0.8, 1.25)
# Irrational steps (golden ratio, sqrt 2, sqrt 3): incommensurate
# wavelengths, spread directions and phases, deterministic.
_STEP_LAMBDA = (math.sqrt(5.0) - 1.0) / 2.0
_STEP_DIR = math.sqrt(2.0) - 1.0
_STEP_PHASE = math.sqrt(3.0) - 1.0
_LAYER_SALT = (0.0, 0.5)               # decorrelate the two layers' sets
SHARPNESS_MAX = 8.0
# A wave's SLOPE amplitude grows with its wavelength as (lambda /
# lambda0)^SLOPE_EXP (its height as lambda^(1 + SLOPE_EXP)): at 1 every wave
# of a layer carries the same curvature, so the caustic is a net of every
# wave, not a weave of the shortest (equal slopes -- Finch's equal
# steepness -- drew a regular cross-hatch; W1b, 2026-11-20).
SLOPE_EXP = 1.0


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
    # W1: the fine surface that flows (W1b: procedural travelling sines)
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
    # W1b: the wave law and the swell
    flow_still_mps: float
    wave_count: int
    wave_sharpness: float
    wave_sharpness_flow: float
    wave_speed: float
    wave_cell_m: float
    swell_wavelength_m: float
    swell_wave_count: int
    swell_strength_still: float
    swell_strength_flow: float
    swell_dir_deg: float
    swell_spread_deg: float
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


_INT_KEYS = ("debug_view", "foam_cells_per_tile", "wave_count",
             "swell_wave_count")
_VEC_KEYS = {"water_color": 3, "foam_bubble_m": 2}
# 0 = the picture; 1 = (slope x, slope y, RMS strength, 1); 2 = (foam
# amount, foam cover, 0, 1); 3 = (hxx, hxy, hyy, caustic gain) -- the
# caustic (W4)
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
                 "flow_edge_depth_m", "agitation_ref", "wave_cell_m",
                 "swell_wavelength_m", "flow_still_mps"):
        if getattr(s, name) <= 0.0:
            raise ValueError(f"[graphics.water] {name} must be > 0")
    for name in ("detail_strength_still", "detail_strength_flow",
                 "detail_drift_mps", "flow_max_offset_m",
                 "flow_phase_noise_m", "wave_speed", "swell_strength_still",
                 "swell_strength_flow"):
        if getattr(s, name) < 0.0:
            raise ValueError(f"[graphics.water] {name} must be >= 0")
    for name in ("wave_count", "swell_wave_count"):
        if not 1 <= getattr(s, name) <= MAX_WAVES:
            raise ValueError(f"[graphics.water] {name} must be in "
                             f"1..{MAX_WAVES}: {getattr(s, name)}")
    for name in ("wave_sharpness", "wave_sharpness_flow"):
        if not 1.0 <= getattr(s, name) <= SHARPNESS_MAX:
            raise ValueError(f"[graphics.water] {name} must be in "
                             f"[1, {SHARPNESS_MAX}]: {getattr(s, name)}")
    if not 0.0 <= s.swell_spread_deg <= 180.0:
        raise ValueError(f"[graphics.water] swell_spread_deg must be in "
                         f"[0, 180]: {s.swell_spread_deg}")
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
# The wave set (W1b): travelling sharpened sines, by dispersion
# ---------------------------------------------------------------------------

class WaveSet(NamedTuple):
    """One layer's waves: base wave vectors ``k`` (N, 2) rad/m (the swell's
    around +x), their phases at t = 0 and angular speeds (rad/s), and their
    SLOPE amplitudes ``amp`` (N,) (a wave's height amplitude is ``amp_i /
    |k_i|``)."""
    k: np.ndarray
    phase0: np.ndarray
    omega: np.ndarray
    amp: np.ndarray


class WaveTable(NamedTuple):
    """One layer's waves at a time t: ``k`` (N, 2) rad/m, ``phase`` (N,)
    = ``phase0 - omega t`` wrapped to [0, 2 pi) in float64, ``amp`` (N,)
    the slope amplitudes."""
    k: np.ndarray
    phase: np.ndarray
    amp: np.ndarray


def finch_rms(sharpness: float) -> float:
    """The RMS of ``g_K'`` over a period relative to the plain sine's
    (``1/sqrt 2``): exactly ``sqrt(8 K^2 B(2K - 1/2, 3/2) / pi)``, since ``u
    = (sin psi + 1) / 2`` is arcsine-distributed (``E[u^a (1-u)^b] =
    B(a + 1/2, b + 1/2) / pi``). 1 at K = 1; ~1.12 at 2, ~1.29 at 4."""
    k = float(sharpness)
    x = 2.0 * k - 0.5
    lb = math.lgamma(x) + math.lgamma(1.5) - math.lgamma(x + 1.5)
    return math.sqrt(8.0 * k * k * math.exp(lb) / math.pi)


def dispersion_omega(kmag, speed: float = 1.0) -> np.ndarray:
    """Capillary-gravity waves on deep water: ``omega = speed x sqrt(g k +
    (sigma / rho) k^3)`` (rad/s). A 0.8 m swell runs at ~1.1 m/s, a 12 cm
    ripple at ~0.44 m/s. Deep water: at 0.3 m depth ``tanh(k h)`` is over
    0.98 for every wave of the bands, and a depth term would shear the
    phases apart over time where the depth varies."""
    k = np.asarray(kmag, np.float64)
    return float(speed) * np.sqrt(G_ACCEL * k + SIGMA_OVER_RHO * k ** 3)


@functools.lru_cache(maxsize=16)
def wave_set(layer: int, s: WaterSettings) -> WaveSet:
    """THE layer's wave set (deterministic, no random stream; cached per
    settings: :func:`wave_tables` asks every frame): wave i of N
    has its wavelength ``lambda0 x lo x (hi / lo)^frac((i + 1/2) phi)``
    (golden-ratio steps of the log wavelength across the band: every ratio
    irrational), its direction ``2 pi frac((i + 1/2)(sqrt 2 - 1))`` (the
    ripples; the swell's ``+-spread`` about +x), its phase ``2 pi
    frac((i + 1/2)(sqrt 3 - 1))`` -- each offset by the layer's salt --, its
    dispersion speed and its slope amplitude ``c (lambda_i /
    lambda0)^SLOPE_EXP``, ``c`` making the layer's RMS slope exactly 1 at
    ``wave_sharpness`` (``sum amp_i^2 finch_rms^2 / 2 = 1``)."""
    n = int(s.wave_count if layer == LAYER_DETAIL else s.swell_wave_count)
    i = np.arange(n, dtype=np.float64) + 0.5
    salt = _LAYER_SALT[layer]

    def frac(step):
        return np.mod(i * step + salt, 1.0)
    if layer == LAYER_DETAIL:
        lam0, (lo, hi) = s.detail_wavelength_m, DETAIL_BAND
        th = 2.0 * np.pi * frac(_STEP_DIR)
    else:
        lam0, (lo, hi) = s.swell_wavelength_m, SWELL_BAND
        th = math.radians(s.swell_spread_deg) * (2.0 * frac(_STEP_DIR) - 1.0)
    lam = lam0 * lo * (hi / lo) ** frac(_STEP_LAMBDA)
    kmag = 2.0 * np.pi / lam
    k = np.stack([kmag * np.cos(th), kmag * np.sin(th)], axis=-1)
    r = (lam / lam0) ** SLOPE_EXP
    amp = r * (math.sqrt(2.0 / float(np.sum(r * r)))
               / finch_rms(s.wave_sharpness))
    return WaveSet(k, 2.0 * np.pi * frac(_STEP_PHASE),
                   dispersion_omega(kmag, s.wave_speed), amp)


def wave_tables(s: WaterSettings, t_s: float) -> Tuple[WaveTable, ...]:
    """Both layers' waves at sim time ``t_s``: each wave's phase ``phi_i -
    omega_i t`` wrapped to [0, 2 pi) in float64 -- what the shader gets, so
    it never sees absolute time."""
    out = []
    for layer in LAYERS:
        w = wave_set(layer, s)
        ph = np.mod(w.phase0 - w.omega * float(t_s), 2.0 * np.pi)
        out.append(WaveTable(w.k, ph, w.amp))
    return tuple(out)


# ---------------------------------------------------------------------------
# The frame's uniforms (float64 on the CPU; the GPU sees bounded numbers)
# ---------------------------------------------------------------------------

class FlowFrame(NamedTuple):
    phase: float        # the crossfade phase in [0, 1)
    cycle_s: float      # one cycle, seconds (a layer's age is age_frac x it)
    drift_x: float      # the foam bubbles' still-water drift, METRES
    drift_y: float      # (unbounded, see flow_frame)
    fade: float         # the zoom fade in [0, 1]
    waves: Optional[Tuple[WaveTable, ...]] = None   # W1b: the layers' waves


def flow_frame(t_s: float, tps: float, zoom_px_per_tile: float,
               s: WaterSettings) -> FlowFrame:
    """The frame's flow uniforms at sim-clock time ``t_s`` (seconds,
    ``SimClock.t_noise``). The phase is ``advected_noise.advection_phase``
    with the clock split into its tick and its sub-tick alpha. The waves'
    phases are :func:`wave_tables`. The drift (W2's foam bubbles) is NOT
    wrapped: their lookup is not periodic in the hex lattice's sense; in
    float32 0.03 m/s keeps sub-millimetre precision for days of play."""
    from .heat_shimmer import zoom_fade
    ticks = float(t_s) * float(tps)
    k = math.floor(ticks)
    ph = advection_phase(k, s.flow_cycle_s, tps, alpha=ticks - k)
    cycle_s = ph.tau_ticks / float(tps)
    d = float(t_s) * s.detail_drift_mps
    return FlowFrame(ph.phase, cycle_s, d * DRIFT_DIR[0], d * DRIFT_DIR[1],
                     zoom_fade(zoom_px_per_tile, s.detail_zoom_zero,
                               s.detail_zoom_full),
                     wave_tables(s, t_s))


# ---------------------------------------------------------------------------
# The hex lattice (Mikkelsen 2022), over the wave set
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
    (..., 2) lattice units."""
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
    """A lattice vertex's position in lattice units (``MakeCenST``)."""
    vx = v[..., 0].astype(np.float64)
    vy = v[..., 1].astype(np.float64)
    return np.stack([vx + 0.5 * vy, vy / _SKEW_Y], axis=-1) / HEX_SCALE


def sample_bilinear_repeat(tex: np.ndarray, st: np.ndarray) -> np.ndarray:
    """The GPU's bilinear lookup of a REPEAT texture at level 0 (texel
    centres at (i + .5) / n), ``tex`` (n, n, c) row = v (W2's bubble bake;
    the waves have no texture since W1b)."""
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
    surface term rides (the waves; W2's foam bubbles): never a second
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
    """The foam bubbles' still-water drift in METRES."""
    return frame.drift_x, frame.drift_y


def layer_points(x_m, y_m, vel, depth, frame: FlowFrame, s: WaterSettings):
    """The two flow layers' ADVECTED points (..., 2) metres -- where the
    waves are evaluated, ``p - d0`` and ``p - d1`` --, their crossfade
    weight ``w0`` and the effective flow ``v``: the geometry half of
    :func:`detail_surface`, exposed for the properties."""
    x = np.asarray(x_m, np.float64)
    y = np.asarray(y_m, np.float64)
    d0, d1, w0, v = layer_slides(x, y, vel, depth, frame, s)
    p = np.stack([x, y], axis=-1)
    return p - d0, p - d1, w0, v


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
             s: WaterSettings, layer: int = LAYER_DETAIL) -> np.ndarray:
    """``k_L``: the RMS slope of a layer of still water here (the ripples by
    default, ``LAYER_SWELL`` the swell): its still strength toward its flow
    strength with the drive, x the zoom fade."""
    sat = strength_sat(v, agitation, s)
    if layer == LAYER_DETAIL:
        lo, hi = s.detail_strength_still, s.detail_strength_flow
    else:
        lo, hi = s.swell_strength_still, s.swell_strength_flow
    return (lo + (hi - lo) * sat) * frame.fade


def sharpness(v: np.ndarray, agitation, s: WaterSettings) -> np.ndarray:
    """``K`` here: ``wave_sharpness`` toward ``wave_sharpness_flow`` with the
    drive -- agitated water has sharper crests."""
    sat = strength_sat(v, agitation, s)
    return s.wave_sharpness + (s.wave_sharpness_flow - s.wave_sharpness) * sat


def layer_on(layer: int, s: WaterSettings) -> bool:
    """Whether a layer exists at all (both its strengths 0 = it is not
    evaluated: the twin and the shader skip it)."""
    if layer == LAYER_DETAIL:
        return s.detail_strength_still > 0.0 or s.detail_strength_flow > 0.0
    return s.swell_strength_still > 0.0 or s.swell_strength_flow > 0.0


def swell_dir0(s: WaterSettings) -> Tuple[float, float]:
    """The swell's still-water direction, a unit vector (x east, y south:
    ``swell_dir_deg`` turns clockwise on screen from east)."""
    a = math.radians(s.swell_dir_deg)
    return math.cos(a), math.sin(a)


def swell_direction(v: np.ndarray, s: WaterSettings) -> np.ndarray:
    """The swell's direction for an effective flow ``v`` (..., 2), an angle
    in radians (x east, y south): ``swell_dir_deg`` on still water, turning
    toward the flow's direction as its speed rises to ``flow_speed_ref``,
    ``D = D0 (1 - f) + (v / |v|) f``, ``f = clamp(|v| / flow_speed_ref)``,
    and ``atan2(D)`` -- ``swell_dir_deg`` where ``D`` vanishes."""
    vv = np.asarray(v, np.float64)
    sp = np.sqrt(vv[..., 0] ** 2 + vv[..., 1] ** 2)
    f = np.clip(sp / s.flow_speed_ref, 0.0, 1.0)
    inv = np.where(sp > 0.0, 1.0 / np.maximum(sp, 1e-30), 0.0)
    cx, cy = swell_dir0(s)
    dx = cx * (1.0 - f) + vv[..., 0] * inv * f
    dy = cy * (1.0 - f) + vv[..., 1] * inv * f
    a0 = math.radians(s.swell_dir_deg)
    return np.where(dx * dx + dy * dy > 1e-12, np.arctan2(dy, dx), a0)


FlowFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def grid_flow_fn(flow_packed: np.ndarray, water_packed: np.ndarray,
                 tile_m: float, s: WaterSettings) -> FlowFn:
    """The effective flow at any point in metres from the two per-tile
    textures (``pack_flow``'s RG and ``u_water``'s depth), as the shader
    reads it at a swell vertex (bilinear, CLAMP to the edge texels)."""
    from .heat_shimmer import bilinear_tiles
    fx = np.asarray(flow_packed[..., 0], np.float64)
    fy = np.asarray(flow_packed[..., 1], np.float64)
    dp = np.asarray(water_packed[..., 2], np.float64)

    def fn(x_m, y_m):
        xt = np.asarray(x_m, np.float64) / tile_m
        yt = np.asarray(y_m, np.float64) / tile_m
        v = np.stack([bilinear_tiles(fx, xt, yt),
                      bilinear_tiles(fy, xt, yt)], axis=-1)
        return edge_flow(v, bilinear_tiles(dp, xt, yt), s)
    return fn


def wave_sum(table: WaveTable, q: np.ndarray, k_sharp,
             hessian: bool = True):
    """ONE wave set's sum, exact, at ``q`` (..., 2) metres in the set's own
    frame, with Finch's sharpness ``k_sharp`` (...,) -> ``(h, slope (...,
    2), hess (..., 3) or None)`` in metres and 1/m. ``h`` is the analytic
    height whose gradient and Hessian the other two ARE
    (``tests/test_water_flow_detail.py`` holds them to finite
    differences)."""
    q = np.asarray(q, np.float64)
    K = np.asarray(k_sharp, np.float64)
    plain = K - 1.0 < 1e-6
    h = np.zeros(q.shape[:-1])
    sl = np.zeros(q.shape)
    hs = np.zeros(q.shape[:-1] + (3,)) if hessian else None
    for (kx, ky), ph, am in zip(table.k, table.phase, table.amp):
        psi = kx * q[..., 0] + ky * q[..., 1] + ph
        sn = np.sin(psi)
        cs = np.cos(psi)
        u = (sn + 1.0) * 0.5
        upk = np.where(plain, 1.0, np.maximum(u, 0.0) ** (K - 1.0))
        a = am / math.hypot(kx, ky)                 # the height amplitude
        g1 = a * K * upk * cs
        h += a * (2.0 * u * upk - 1.0)
        sl[..., 0] += g1 * kx
        sl[..., 1] += g1 * ky
        if hessian:
            g2 = a * K * upk * (2.0 * (K - 1.0) * (1.0 - u) - sn)
            hs[..., 0] += g2 * (kx * kx)
            hs[..., 1] += g2 * (kx * ky)
            hs[..., 2] += g2 * (ky * ky)
    return h, sl, hs


def turn_slope(sl: np.ndarray, c, sn) -> np.ndarray:
    """A slope from a tap's own frame to the world: ``R s``, ``R = [[c,
    -s], [s, c]]``."""
    return np.stack([c * sl[..., 0] - sn * sl[..., 1],
                     sn * sl[..., 0] + c * sl[..., 1]], axis=-1)


def turn_hessian(hs: np.ndarray, c, sn) -> np.ndarray:
    """A Hessian ``(hxx, hxy, hyy)`` from a tap's own frame to the world:
    ``R H R^T``."""
    a, b, d = hs[..., 0], hs[..., 1], hs[..., 2]
    cc, cs, ss = c * c, c * sn, sn * sn
    return np.stack([cc * a - 2.0 * cs * b + ss * d,
                     cs * a + (cc - ss) * b - cs * d,
                     ss * a + 2.0 * cs * b + cc * d], axis=-1)


def hex_waves(p: np.ndarray, tables, k_sharp, s: WaterSettings,
              flow_fn: Optional[FlowFn] = None, hessian: bool = True,
              want_height: bool = False, layers=LAYERS):
    """The hex-tiled wave sums of ``layers`` at the (advected) points ``p``
    (..., 2) metres, ``tables`` the frame's per-layer :class:`WaveTable`s ->
    ``{layer: (slope (..., 2), hess or None, h or None)}``.

    ONE lattice for both layers (a vertex every ``wave_cell_m``: the
    triangle and its weights are found once per point): at each of its
    three taps every layer's set is read in its own frame ``R^T (p - c_v) +
    o_v`` (turned by the layer's angle about the vertex centre, shifted by
    the layer's offset -- hashed with the layer's own salts; the slope and
    the Hessian turned back, ``R s`` and ``R H R^T``), and each layer's taps
    are blended by Mikkelsen's ``(1 - c + c |s_i|) w_i^7`` with its own
    ``|s_i|``. A tap whose ``w^7`` is under ``HEX_SKIP`` of the largest
    carries weight 0 and is not evaluated. The ripples' angle is hashed (any
    turn); the swell's is its direction at the vertex centre
    (:func:`swell_direction` of ``flow_fn(c_v)``; still water where
    ``flow_fn`` is None) jittered by +-``swell_spread_deg / 2``. The weights
    are held fixed in the Hessian (their gradient lives on the narrow
    ``w^7`` seams): a blend of honest sums."""
    p = np.asarray(p, np.float64)
    cell = float(s.wave_cell_m)
    w, verts = triangle_grid(p / (cell * HEX_SCALE))
    w7 = np.maximum(w, 0.0) ** HEX_EXP
    wmax = w7.max(axis=-1)
    acc = {L: np.zeros(p.shape) for L in layers}
    hacc = {L: np.zeros(p.shape[:-1] + (3,)) for L in layers}
    hh = {L: np.zeros(p.shape[:-1]) for L in layers}
    wsum = {L: np.zeros(p.shape[:-1]) for L in layers}
    for i, v in enumerate(verts):
        cen = hex_center(v) * (cell * HEX_SCALE)
        live = w7[..., i] >= HEX_SKIP * wmax
        d = p - cen
        for L in layers:
            salt = 10 * L
            hr = hex_hash01(v[..., 0], v[..., 1], _SALT_ROT + salt)
            if L == LAYER_DETAIL:
                th = 2.0 * np.pi * (hr - 0.5)
            else:
                vc = (np.zeros(cen.shape) if flow_fn is None
                      else flow_fn(cen[..., 0], cen[..., 1]))
                th = (swell_direction(vc, s)
                      + math.radians(s.swell_spread_deg) * (hr - 0.5))
            o = np.stack([hex_hash01(v[..., 0], v[..., 1], _SALT_U + salt),
                          hex_hash01(v[..., 0], v[..., 1], _SALT_V + salt)],
                         axis=-1) * cell
            c, sn = np.cos(th), np.sin(th)
            q = np.stack([c * d[..., 0] + sn * d[..., 1] + o[..., 0],
                          -sn * d[..., 0] + c * d[..., 1] + o[..., 1]],
                         axis=-1)
            h, sl, hs = wave_sum(tables[L], q, k_sharp, hessian)
            lum = np.sqrt(sl[..., 0] ** 2 + sl[..., 1] ** 2)
            wi = np.where(live, (1.0 - HEX_FALLOFF_CONTRAST
                                 + HEX_FALLOFF_CONTRAST * lum)
                          * w7[..., i], 0.0)
            acc[L] += wi[..., None] * turn_slope(sl, c, sn)
            if hessian:
                hacc[L] += wi[..., None] * turn_hessian(hs, c, sn)
            hh[L] += wi * h
            wsum[L] += wi
    return {L: (acc[L] / wsum[L][..., None],
                hacc[L] / wsum[L][..., None] if hessian else None,
                hh[L] / wsum[L] if want_height else None)
            for L in layers}


def wave_slides(d0, d1, v, s: WaterSettings):
    """The waves' own slides: the flow map's ``d0``, ``d1`` x a fade from 0
    below ``flow_still_mps / 2`` to 1 at ``flow_still_mps`` of effective
    flow. A slow drift current (the sim's still rooms keep ~1 cm/s
    everywhere) is no sight against waves running at 0.3-1 m/s; with no
    slide both layers read the same point and the sum is evaluated ONCE
    (half the cost of a still room). W2's foam bubbles keep the plain
    slides."""
    vv = np.asarray(v, np.float64)
    sp = np.sqrt(vv[..., 0] ** 2 + vv[..., 1] ** 2)
    f = smoothstep(0.5 * s.flow_still_mps, s.flow_still_mps, sp)[..., None]
    return d0 * f, d1 * f


def detail_surface(x_m, y_m, vel, depth, agitation, frame: FlowFrame,
                   s: WaterSettings, flow_fn: Optional[FlowFn] = None,
                   hessian: bool = True):
    """THE fine surface's slope (dh/dx, dh/dy; y south) and, with
    ``hessian``, its Hessian ``(hxx, hxy, hyy)`` in 1/m (W4's caustic) at
    world positions in metres -> ``(slope (..., 2), hess (..., 3) or
    None)``: the module docstring's law. ``vel`` (..., 2) the bilinear flow
    in m/s, ``depth`` m, ``agitation`` the bilinear relative agitation
    (:func:`relative_agitation`, ``u_water``'s A) m/s, ``flow_fn`` the
    effective flow anywhere (:func:`grid_flow_fn`; the swell's vertices read
    it; None = still water).

    The layers are evaluated at the two flow layers' advected points (the
    waves' slides, :func:`wave_slides`) and cross-faded with ``w0`` -- or
    ONCE where both slides are equal (still water: both points are the
    same), the result taken as it is. Their Hessian is the exact one of each
    tap's sum, blended and cross-faded with the same weights, x ``k_L``, the
    layers summed. A layer whose strengths are both 0 is not evaluated."""
    x = np.asarray(x_m, np.float64)
    y = np.asarray(y_m, np.float64)
    d0, d1, w0, v = layer_slides(x, y, vel, depth, frame, s)
    d0, d1 = wave_slides(d0, d1, v, s)
    p = np.stack([x, y], axis=-1)
    slope = np.zeros(p.shape)
    hess = np.zeros(p.shape[:-1] + (3,)) if hessian else None
    on = tuple(L for L in LAYERS if layer_on(L, s))
    if not on:
        return slope, hess
    same = np.all(d0 == d1, axis=-1)[..., None]
    K = sharpness(v, agitation, s)
    w = w0[..., None]
    r0 = hex_waves(p - d0, frame.waves, K, s, flow_fn, hessian, layers=on)
    r1 = hex_waves(p - d1, frame.waves, K, s, flow_fn, hessian, layers=on)
    for L in on:
        (s0, h0, _), (s1, h1, _) = r0[L], r1[L]
        k = strength(v, agitation, frame, s, L)[..., None]
        slope = slope + np.where(same, s0, s1 * (1.0 - w) + s0 * w) * k
        if hessian:
            hess = hess + np.where(same, h0, h1 * (1.0 - w) + h0 * w) * k
    return slope, hess


def detail_slope(x_m, y_m, vel, depth, agitation, frame: FlowFrame,
                 s: WaterSettings, flow_fn: Optional[FlowFn] = None
                 ) -> np.ndarray:
    """THE fine surface's slope alone (:func:`detail_surface` without the
    Hessian; the same numbers)."""
    return detail_surface(x_m, y_m, vel, depth, agitation, frame, s,
                          flow_fn, hessian=False)[0]


def wave_uniforms(frame: FlowFrame) -> np.ndarray:
    """The shader's ``u_waves`` (2 x MAX_WAVES, 4) float32: per layer and
    wave ``(kx, ky, phase now, height amplitude)`` (unused rows 0)."""
    out = np.zeros((len(LAYERS) * MAX_WAVES, 4), np.float32)
    for layer, tab in zip(LAYERS, frame.waves):
        n = len(tab.phase)
        j = layer * MAX_WAVES
        kk = np.hypot(tab.k[:, 0], tab.k[:, 1])
        out[j:j + n, 0] = tab.k[:, 0]
        out[j:j + n, 1] = tab.k[:, 1]
        out[j:j + n, 2] = tab.phase
        out[j:j + n, 3] = tab.amp / kk
    return out


def rms_strength(v: np.ndarray, agitation, frame: FlowFrame,
                 s: WaterSettings) -> np.ndarray:
    """``sqrt(k_ripple^2 + k_swell^2)``: the surface's RMS slope here (the
    shader's debug view 1 writes it beside the slope)."""
    out = np.zeros(np.shape(v)[:-1])
    for layer in LAYERS:
        if layer_on(layer, s):
            out = out + strength(v, agitation, frame, s, layer) ** 2
    return np.sqrt(out)


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
    "FlowFrame", "flow_frame", "DRIFT_DIR", "MAX_WAVES", "LAYERS",
    "LAYER_DETAIL", "LAYER_SWELL", "HEX_SKIP", "G_ACCEL", "SIGMA_OVER_RHO",
    "WaveSet", "WaveTable", "finch_rms", "dispersion_omega", "wave_set",
    "wave_tables", "hex_hash", "hex_hash01", "triangle_grid", "hex_center",
    "sample_bilinear_repeat",
    "smoothstep", "flow_slide", "edge_flow", "layer_slides", "drift_m",
    "layer_points", "relative_agitation", "strength_ratio", "strength_sat",
    "strength", "sharpness", "layer_on", "swell_dir0", "swell_direction",
    "grid_flow_fn", "wave_sum", "turn_slope", "turn_hessian", "hex_waves",
    "wave_slides", "detail_surface", "detail_slope", "wave_uniforms",
    "rms_strength",
    "WATER_IOR", "CAUSTIC_K", "caustic_det", "caustic_gain", "caustic_term",
    "waterline_column", "waterline_weight", "waterline_tint",
    "wet_box", "pack_flow",
]
