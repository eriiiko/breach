"""The smoke PHASE FIELD (issue #90, Erik's idea 2026-10-15) — pyray-free.

The baked volume is one looping movie. Instead of every pixel reading it at
the same global clock, each TILE gets its own phase, and the GPU's bilinear
lookup interpolates it between tile centres, so the phase is continuous over
the floor (the border between two tiles reads the mean of its neighbours,
every pixel in between a little ahead of or behind the one before it). The
movie is periodic in time, so a continuous phase has no seams. Two places
one period apart then show the same spatial texture at DIFFERENT moments of
the loop: once the offset differs by a decent fraction of the loop the
frames are decorrelated and the asset's repeat disappears — the number-one
complaint on the first look.

Two modes, both driven by the same per-tile sources:

* ``offset`` — phase = clock + offset(sources). Bounded, stable, seamless.
  The static world-space noise term alone already de-tiles a quiet room; the
  sim terms make hot / pressurised / windy tiles sit elsewhere in the loop.
* ``speed`` — each tile integrates its own rate deviation, so the LOCAL
  clock runs faster where the source is high and the phase difference to a
  quiet neighbour grows (a temporal shear across the tile). A leak toward
  zero (``relax_loops``) bounds the drift so the shear never diverges.

Sources (each normalised to roughly [-1, 1] before its amplitude):
``noise`` (static value noise over the tile grid, wavelength ``noise_tiles``),
``temp`` (dequantised game ΔT / ``temp_ref``, softened), ``press``
(dequantised atmosphere minus its median, / ``press_ref``, softened), ``wind``
(the TAMED wind speed / its saturation speed, already in [0, 1)).

Encoding: the shader needs the phase bilinearly interpolated ACROSS the wrap
(0.98 next to 0.02 must read as 0.00, not 0.50), so the field goes up as the
unit vector (cos 2πφ, sin 2πφ) in two half-float channels and the shader
takes atan2 — interpolation of unit vectors stays on the short arc.

RENDER-ONLY, determinism-exempt: reads dequantised sim fields, writes only
its own float state. The clock input is the sim tick, never wall time, so a
replay renders the same phase field.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

TWO_PI = 2.0 * np.pi
PHASE_MODES = ("off", "offset", "speed")


def value_noise_tiles(h: int, w: int, wavelength_tiles: float,
                      seed: int = 11) -> np.ndarray:
    """Static smooth value noise over the tile grid, in [-1, 1], with ~one
    bump per ``wavelength_tiles`` tiles. A random lattice at that spacing,
    cosine-interpolated onto tile centres — pure numpy, seeded, so the same
    level renders the same field every launch."""
    L = max(float(wavelength_tiles), 1.0)
    rng = np.random.default_rng(int(seed))
    gh = int(np.ceil(h / L)) + 2
    gw = int(np.ceil(w / L)) + 2
    lattice = rng.uniform(-1.0, 1.0, size=(gh, gw))
    ys = (np.arange(h) + 0.5) / L
    xs = (np.arange(w) + 0.5) / L
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = ys - y0
    fx = xs - x0
    # cosine ease: C1 across lattice cells (no visible creases)
    fy = 0.5 - 0.5 * np.cos(np.pi * fy)
    fx = 0.5 - 0.5 * np.cos(np.pi * fx)
    y1 = np.minimum(y0 + 1, gh - 1)
    x1 = np.minimum(x0 + 1, gw - 1)
    a = lattice[np.ix_(y0, x0)]
    b = lattice[np.ix_(y0, x1)]
    c = lattice[np.ix_(y1, x0)]
    d = lattice[np.ix_(y1, x1)]
    top = a + (b - a) * fx[None, :]
    bot = c + (d - c) * fx[None, :]
    out = top + (bot - top) * fy[:, None]
    return out.astype(np.float32)


def normalise_sources(*, temperature: Optional[np.ndarray] = None,
                      atmosphere: Optional[np.ndarray] = None,
                      wind_speed: Optional[np.ndarray] = None,
                      temp_ref: float = 200.0, press_ref: float = 0.05,
                      wind_v_ref: float = 0.08) -> Dict[str, np.ndarray]:
    """Dequantised sim fields -> the normalised source dict. Temperature and
    pressure are softened with tanh so a fire core or a blast cannot throw a
    tile a hundred loops away from its neighbour; the tamed wind is already
    bounded by its saturation speed."""
    out: Dict[str, np.ndarray] = {}
    if temperature is not None:
        t = np.asarray(temperature, dtype=np.float32)
        out["temp"] = np.tanh(t / max(float(temp_ref), 1e-6)).astype(np.float32)
    if atmosphere is not None:
        p = np.asarray(atmosphere, dtype=np.float32)
        dp = p - np.float32(np.median(p))
        out["press"] = np.tanh(dp / max(float(press_ref), 1e-9)).astype(np.float32)
    if wind_speed is not None:
        s = np.asarray(wind_speed, dtype=np.float32)
        out["wind"] = np.clip(s / max(float(wind_v_ref), 1e-9), 0.0, 1.0).astype(np.float32)
    return out


def encode_phase_rg(offset_loops: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Write (cos 2πφ, sin 2πφ) of the phase offset into channels 0,1 of an
    (H, W, 4) float16 buffer (channel 2 = 0, 3 = 1). Returns ``out``."""
    ang = (np.asarray(offset_loops, dtype=np.float64) * TWO_PI)
    out[..., 0] = np.cos(ang).astype(np.float16)
    out[..., 1] = np.sin(ang).astype(np.float16)
    out[..., 2] = np.float16(0.0)
    out[..., 3] = np.float16(1.0)
    return out


def decode_phase_rg(rg: np.ndarray) -> np.ndarray:
    """The shader's inverse: (cos, sin) -> phase in [0, 1). CPU mirror for
    the tests and the viewer."""
    rg = np.asarray(rg, dtype=np.float64)
    return np.mod(np.arctan2(rg[..., 1], rg[..., 0]) / TWO_PI, 1.0)


class PhaseField:
    """Per-tile phase offset state. ``step`` returns the offset (in LOOPS,
    unbounded in ``speed`` mode only by the leak) for this frame."""

    def __init__(self, h: int, w: int, *, seed: int = 11):
        self.h = int(h)
        self.w = int(w)
        self.seed = int(seed)
        self.acc = np.zeros((self.h, self.w), dtype=np.float32)
        self._noise: Optional[np.ndarray] = None
        self._noise_key = None
        self.offset = np.zeros((self.h, self.w), dtype=np.float32)

    def noise(self, wavelength_tiles: float) -> np.ndarray:
        key = round(float(wavelength_tiles), 4)
        if self._noise is None or self._noise_key != key:
            self._noise = value_noise_tiles(self.h, self.w, key, self.seed)
            self._noise_key = key
        return self._noise

    def reset(self) -> None:
        self.acc[...] = 0.0

    def step(self, *, mode: str, dt_loops: float, sources: Dict[str, np.ndarray],
             amps: Dict[str, float], noise_amp: float, noise_tiles: float,
             relax_loops: float) -> np.ndarray:
        """Advance one frame. ``dt_loops`` = sim time advanced since the last
        call, in loops (0 while paused — nothing drifts on a frozen frame).
        ``amps`` = per-source amplitude: in ``offset`` mode LOOPS per unit
        source; in ``speed`` mode the RATE deviation (loops per loop) per
        unit source."""
        if mode not in PHASE_MODES:
            raise ValueError(f"phase mode {mode!r} not in {PHASE_MODES}")
        if mode == "off":
            self.offset[...] = 0.0
            return self.offset
        drive = np.zeros((self.h, self.w), dtype=np.float32)
        for name, src in sources.items():
            a = float(amps.get(name, 0.0))
            if a != 0.0:
                drive += np.float32(a) * src
        static = np.float32(noise_amp) * self.noise(noise_tiles) if noise_amp else 0.0
        if mode == "offset":
            self.offset = (drive + static).astype(np.float32)
            return self.offset
        # speed: integrate the rate deviation, leak toward zero so the drift
        # between neighbours is bounded by ~ amp * relax_loops.
        dt = max(float(dt_loops), 0.0)
        if dt > 0.0:
            self.acc += drive * np.float32(dt)
            if relax_loops > 0.0:
                self.acc *= np.float32(np.exp(-dt / float(relax_loops)))
        self.offset = (self.acc + static).astype(np.float32)
        return self.offset


__all__ = ["PhaseField", "PHASE_MODES", "value_noise_tiles", "normalise_sources",
           "encode_phase_rg", "decode_phase_rg"]
