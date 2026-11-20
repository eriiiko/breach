"""Advected-noise primitives for the sub-tile gas detail (Fire & Heat Beauty
arc, B2 P3) — the deterministic noise BAKES and the two-layer crossfade PHASE
math, all pyray-free (numpy only) so they are headless-testable without a GL
context and byte-for-byte reproducible on every machine.

This is the SHARED module the design (§4/§5) asks for: P3's ``shaders/
gas_medium.fs`` consumes the baked textures + the per-frame phase uniforms, and
**P4's CPU-side speckle** (``HeatFieldOverlay``, which has no fragment shader)
evaluates the SAME two-layer advected-phase recipe at grid resolution by calling
``advection_phase`` / ``layer_ages_weight`` here. Keep the crossfade math in ONE
place so the shader smoke and the CPU speckle can never drift apart.

The clock is the SIM TICK, never wall time: ``phase = frac(tick / tau_ticks)``
with ``tau_ticks = cycle_seconds * ticks_per_second``. Replays and spectators
feed the same integer tick, so they render identical smoke. The decomposition is
done in float64 from the integer tick and only a BOUNDED ``phase in [0,1)`` is
handed to the shader, so a long session's large tick never hits the float32
precision cliff on the GPU.

Credit (repo rule — cite what a file implements):
  - Alex Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010 course (Advances in
    Real-Time Rendering) — the two-layer ping-pong flow-map crossfade that hides
    the periodic reset of a single advected layer.
  - Fabrice Neyret, "Advected Textures", ACM SIGGRAPH/Eurographics SCA 2003 —
    advecting a noise field by a velocity field with periodic regeneration.
  - Ken Perlin & Fabrice Neyret, "Flow Noise" (SIGGRAPH 2001 sketch) — animated
    procedural noise for flowing media (the "breathing" a zero-wind crossfade
    gives here).
  Links live in docs/research/smoke_render_litsearch_2026-07-21.md §4.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np

# Fixed seeds so the bakes are byte-identical every run / machine (the gate's
# "noise-bake determinism"). Distinct seeds keep the fBm channels + the jitter
# field decorrelated.
_FBM_SEED = 0xB2F1  # "B2 fbm"
_JITTER_SEED = 0xB2 * 0x100 + 0x17  # "B2 jitter"

# The fBm bake's coarsest octave lattice period (in texels). Each octave doubles
# the frequency; period_o = _FBM_BASE_PERIOD * 2**o. Chosen so that for the
# design's texture size (256) and octave counts (<=6) every octave period
# divides the size EXACTLY -> the lattice wraps -> the texture tiles seamlessly.
_FBM_BASE_PERIOD = 4


# ---------------------------------------------------------------------------
# Tiling fBm bake (baked ONCE at startup)
# ---------------------------------------------------------------------------

def _periodic_value_noise(size: int, period: int,
                          rng: np.random.RandomState) -> np.ndarray:
    """One octave of TILING value noise, shape (size, size), in [0, 1).

    A random value lattice of shape (period, period) is smooth-step interpolated
    up to (size, size). The lattice index wraps modulo ``period`` on BOTH axes,
    so the octave — and therefore the summed fBm — tiles seamlessly. ``period``
    must divide ``size`` for the wrap to land on a texel boundary (asserted by
    the caller's octave choice)."""
    lattice = rng.rand(period, period).astype(np.float64)
    # Sample coordinates in lattice units [0, period).
    coord = np.arange(size, dtype=np.float64) * (period / size)
    i0 = np.floor(coord).astype(np.int64) % period
    i1 = (i0 + 1) % period
    frac = coord - np.floor(coord)
    # Smoothstep weights (C1-continuous -> no lattice-aligned creases).
    w = frac * frac * (3.0 - 2.0 * frac)
    wx = w[None, :]
    wy = w[:, None]
    ix0, ix1 = i0[None, :], i1[None, :]
    iy0, iy1 = i0[:, None], i1[:, None]
    # Bilinear blend of the four wrapped lattice corners.
    top = lattice[iy0, ix0] * (1.0 - wx) + lattice[iy0, ix1] * wx
    bot = lattice[iy1, ix0] * (1.0 - wx) + lattice[iy1, ix1] * wx
    return top * (1.0 - wy) + bot * wy


def _fbm_channel(size: int, octaves: int, persistence: float,
                 rng: np.random.RandomState) -> np.ndarray:
    """Sum ``octaves`` of tiling value noise (amplitude ``persistence**o``) and
    normalise to [0, 1]. Kolmogorov-flavoured: persistence ~0.56 gives the
    "-5/3" roll-off the design asks for (fine detail present but subordinate)."""
    acc = np.zeros((size, size), dtype=np.float64)
    amp_sum = 0.0
    for o in range(int(octaves)):
        period = _FBM_BASE_PERIOD * (2 ** o)
        if period > size:
            break
        assert size % period == 0, (
            f"fBm octave period {period} must divide texture size {size} "
            f"for seamless tiling")
        amp = float(persistence) ** o
        acc += amp * _periodic_value_noise(size, period, rng)
        amp_sum += amp
    if amp_sum > 0.0:
        acc /= amp_sum
    # Normalise the realised range to [0,1] so the erosion remap is stable
    # regardless of octave count / persistence.
    lo, hi = float(acc.min()), float(acc.max())
    if hi > lo:
        acc = (acc - lo) / (hi - lo)
    return acc


def bake_fbm_rgba(size: int = 256, octaves: int = 4,
                  persistence: float = 0.56, seed: int = _FBM_SEED
                  ) -> np.ndarray:
    """Bake the tiling fBm noise texture -> (size, size, 4) uint8.

    R  = primary fBm (the coverage/erosion noise, sampled at the advected UVs).
    GB = a second, decorrelated fBm pair used as the domain-WARP vector (hides
         the tile lattice).
    A  = 255 (opaque; unused).

    Deterministic: a fixed seed + the tiling construction make this byte-
    identical on every run and machine. Wrap the texture REPEAT + filter
    BILINEAR when uploading (it must tile and read smoothly)."""
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    r = _fbm_channel(size, octaves, persistence, rng)
    g = _fbm_channel(size, octaves, persistence, rng)
    b = _fbm_channel(size, octaves, persistence, rng)
    out = np.empty((size, size, 4), dtype=np.uint8)
    out[..., 0] = np.clip(r * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(g * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(b * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def bake_jitter_rgba(size: int = 64, seed: int = _JITTER_SEED) -> np.ndarray:
    """Bake the small static white-noise jitter texture -> (size, size, 4) uint8.

    R = per-pixel PHASE jitter (sampled at a COARSE scale in the shader so it is
        spatially smooth — it desynchronises the crossfade across the screen,
        killing the whole-screen pulse, without adding high-frequency shimmer).
    G = the DITHER source (sampled per-pixel/fine) that breaks 8-bit banding in
        the thin-gradient range.
    BA = further decorrelated noise (reserved).

    White noise (not fBm): the jitter wants to be per-pixel independent. Wrap
    REPEAT + filter POINT so no bilinear smoothing correlates neighbours."""
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    out = (rng.rand(size, size, 4) * 255.0 + 0.5).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Two-layer advected-phase crossfade math (SHARED with P4)
# ---------------------------------------------------------------------------

class AdvectionPhase(NamedTuple):
    """The per-frame crossfade state derived from the sim tick.

    phase      : float in [0, 1) — the global crossfade phase (bounded, so the
                 shader never sees a large tick). Layer 0 resets at phase==0,
                 layer 1 at phase==0.5.
    tau_ticks  : float — the cycle length in ticks (cycle_seconds *
                 ticks_per_second); the shader multiplies a layer's phase by
                 this to get its ADVECTION AGE in ticks, matching the sim's
                 per-tick wind displacement.
    """
    phase: float
    tau_ticks: float


def advection_phase(tick: int, cycle_seconds: float,
                    ticks_per_second: float,
                    alpha: float = 0.0) -> AdvectionPhase:
    """Integer sim tick (+ the sub-tick ``alpha``) -> the bounded crossfade
    phase + the cycle length.

    ``phase = frac((tick + alpha) / tau_ticks)`` with ``tau_ticks =
    cycle_seconds * ticks_per_second``. Computed in float64 from the integer
    tick, then only the bounded ``phase`` (and the small ``tau_ticks``) leave
    for the GPU — a long game's large tick never loses precision as a float32
    shader clock. This is the SINGLE definition of the smoke clock; P4's CPU
    speckle calls it too.

    ``alpha`` (water W1, #97) is the sub-tick fraction of a clock that runs
    between ticks (``fire_vfx.SimClock``); the caller hands it already
    clamped. The default 0.0 adds exactly nothing (``float(tick) + 0.0`` is
    ``float(tick)`` in IEEE 754), so every tick-stepped caller (gas detail,
    gas relief, speckle) is bit-identical to the law before W1
    (``tests/test_water_flow_detail.py`` pins it against the frozen formula)."""
    tau_ticks = float(cycle_seconds) * float(ticks_per_second)
    if tau_ticks <= 0.0:
        return AdvectionPhase(0.0, 0.0)
    phase = float(np.mod((float(tick) + float(alpha)) / tau_ticks, 1.0))
    return AdvectionPhase(phase, tau_ticks)


def layer_ages_weight(phase: float,
                      offset: float = 0.0) -> Tuple[float, float, float]:
    """(phase in [0,1)) -> (age_frac0, age_frac1, w0) — the two layers' age
    fractions and layer-0's crossfade weight, the CPU mirror of the shader math.

    age_frac_i in [0,1) is the layer's age as a fraction of the cycle (multiply
    by ``tau_ticks`` for ticks). ``w0 = 1/2 - 1/2 cos(2 pi phase)`` is layer-0's
    weight: it is 0 exactly when layer 0 resets (phase==0) and 1 when layer 1
    resets (phase==0.5), so each layer's periodic UV pop is hidden under zero
    weight (Vlachos ping-pong). The final value is ``mix(n1, n0, w0)``.

    ``offset`` (water W1) shifts this point's phase (the per-pixel low-
    frequency offset, :func:`phase_offset_field`): every point still resets
    under zero weight, just not all at once, so no whole-surface pulse. The
    default 0.0 is the pre-W1 law bit for bit."""
    p = (float(phase) + float(offset)) % 1.0
    age0 = p
    age1 = (p + 0.5) % 1.0
    w0 = 0.5 - 0.5 * np.cos(2.0 * np.pi * p)
    return age0, age1, float(w0)


def layer_ages_weight_field(phase: float, offset: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`layer_ages_weight` over an array of per-point phase offsets
    (water W1): ``(age_frac0, age_frac1, w0)`` arrays shaped like
    ``offset``. The same law (equal to the scalar form at every point); the
    shader's ``advLayers`` (:data:`ADVECTION_GLSL`) mirrors it, ``fract`` for
    the modulo."""
    p = np.mod(float(phase) + np.asarray(offset, dtype=np.float64), 1.0)
    age1 = np.mod(p + 0.5, 1.0)
    w0 = 0.5 - 0.5 * np.cos(2.0 * np.pi * p)
    return p, age1, w0


# The two directions and phases of the low-frequency offset (water W1). Two
# plane waves at incommensurate angles and lengths: smooth (so neighbouring
# points reset at nearly the same moment and the advected texture is not torn
# between them) yet spread over a whole cycle across a few lengths of
# ``scale_m``, so the reset of a cycle never shows as a pulse of the whole
# surface. The constants are mirrored literally in ADVECTION_GLSL.
_PH_K1 = (0.83, 0.56)
_PH_K2 = (-0.47 * 1.37, 0.88 * 1.37)
_PH_C1, _PH_C2 = 1.7, 4.1


def phase_offset_field(x_m, y_m, scale_m: float) -> np.ndarray:
    """The per-point phase offset in [-0.5, 0.5] at positions in metres
    (water W1): ``0.25 sin(a k1.p + c1) + 0.25 sin(a k2.p + c2)``, ``a =
    2 pi / scale_m``. A pure function of world position, so it never moves
    with time and is the same on every machine; ``scale_m <= 0`` is 0 (every
    point in phase, the plain Vlachos crossfade)."""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if float(scale_m) <= 0.0:
        return np.zeros(np.broadcast(x, y).shape)
    a = 2.0 * np.pi / float(scale_m)
    return (0.25 * np.sin(a * (_PH_K1[0] * x + _PH_K1[1] * y) + _PH_C1)
            + 0.25 * np.sin(a * (_PH_K2[0] * x + _PH_K2[1] * y) + _PH_C2))


# The GLSL face of the two functions above, spliced into a shader at its
# ``// <ADVECTION>`` marker (the water pass, W1). ONE copy: a consumer that
# needs the per-point crossfade splices this, never writes its own.
ADVECTION_GLSL = """
// --- two-layer advection crossfade (renderer/advected_noise.py) ---
float advPhaseOffset(vec2 p_m, float scale_m) {
    if (scale_m <= 0.0) return 0.0;
    float a = 6.28318530718 / scale_m;
    return 0.25 * sin(a * dot(vec2(%r, %r), p_m) + %r)
         + 0.25 * sin(a * dot(vec2(%r, %r), p_m) + %r);
}
// (age_frac0, age_frac1, w0) of a point whose phase is shifted by offset;
// the value is mix(layer1, layer0, w0).
vec3 advLayers(float phase, float offset) {
    float p = fract(phase + offset);
    return vec3(p, fract(p + 0.5), 0.5 - 0.5 * cos(6.28318530718 * p));
}
""" % (_PH_K1[0], _PH_K1[1], _PH_C1, _PH_K2[0], _PH_K2[1], _PH_C2)


__all__ = [
    "bake_fbm_rgba", "bake_jitter_rgba",
    "AdvectionPhase", "advection_phase", "layer_ages_weight",
    "layer_ages_weight_field", "phase_offset_field", "ADVECTION_GLSL",
]
