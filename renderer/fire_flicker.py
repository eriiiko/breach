"""The fire's ROOM-LIGHT FLICKER -- graphics arc G3 "Fire", patch F4b.

Design: ``docs/g3_fire_design_2026-10-23.md`` §1 item 6 and §6.3 (RULED D,
Erik 2026-10-23): the fire's own room light flickers through a LIGHT-ONLY
multiplier on the sweep's thermal emission of each burning cell. No heat, no
digest, no golden; lamps are untouched.

WHAT IT COMPUTES. Per tick, per BURNING tile (the flame's own source law,
``fire_vfx_ref.flame_sources``'s ``burning`` -- I above the flame floor, at or
above the ramp's glow floor, not vacuum), one float multiplier

    m = 1 + amplitude * clip(z / CLIP_SIGMA, -1, 1)
    z = sqrt(puff_share) * z_puff + sqrt(1 - puff_share) * z_tile

  * ``z_puff = sqrt(2) sin(theta)`` -- the tile's CLUSTER puffing, the very
    phase the flame injects with (``fire_vfx_ref.puff_theta``: Cetegen &
    Ahmed's f = 1.5 / sqrt(D), the cluster's hashed phase and jitter, the slow
    wobble). So the room light pulses WITH the flame the eye sees. It is
    shared by a fire's tiles, so it gets only ``puff_share`` of the variance:
    two tiles of one fire then correlate at about ``puff_share`` (the §6.3
    no-sync gate is |rho| < 0.3), and two separate fires not at all (their
    puffing has its own phase, frequency and wobble).
  * ``z_tile`` -- the tile's OWN turbulence: ``N_OSC`` sinusoids per tile at
    frequencies hashed log-uniformly from the band ``[band_lo_hz,
    band_hi_hz]`` (one per ``1/N_OSC`` of the band, so every tile spans it),
    with hashed phases and a gently pink weighting (amplitude ~ f^-1/4: a
    little more slow breathing than fast flutter), normalised to unit
    variance. Keyed by the tile's MAP index through a full splitmix64 mix
    (:func:`_mix01`), so it is a fixed function of (tile, time): neighbours
    decorrelate by construction, and nothing is shared but the clock.

    WHY EIGHT, AND ONLY A LITTLE PINK (F4b's measurement): two independent
    narrow-band signals still correlate BY CHANCE over a short window --
    with three strongly pink sinusoids per tile ~10 % of tile pairs passed
    |rho| = 0.3 over 10 s. Eight flatter ones spread each tile over more of
    the band, and over a 30 s window the chance correlation stays under the
    gate with margin (tests/test_fire_flicker.py measures it).

Both terms are zero-mean and symmetric, and the clip is symmetric, so the
multiplier is MEAN-PRESERVING: over a second the fire's average room light is
the physics' light (Erik's 1300 K exposure anchor and his 2026-10-22
brightness tuning stay true). ``z`` has unit variance, so the multiplier's RMS
swing is about ``amplitude / CLIP_SIGMA`` and it never leaves ``[1 -
amplitude, 1 + amplitude]``. The band tops out well under the 24 Hz tick's
Nyquist rate (12 Hz), so it breathes and flutters but never strobes.

THE CLOCK. The flame's VFX clock (``SimClock.t_noise``: sim time + alpha, plus
the idle counter while paused with ``[render.fire_vfx] idle_when_paused``):
the room light flickers exactly when the flame moves. The sweep reads the
multiplier once per sim tick (24 Hz); while PAUSED the game relights at the
TICK rate whenever the flicker moved (main.py), so a paused, idling fire
flickers as it does running -- and costs what a running tick's sweep costs.
With idle off the clock stops, the rows stop changing, and the paused relight
falls back to its 4 Hz. Without the flame VFX (``[render.fire_vfx] enabled =
false``) there is no clock and no burning-tile list: no flicker (rows None).

HOW IT CROSSES. This module is pure numpy floats (render layer, iron rule 1's
exemption). :func:`renderer.frame_lights.emit_mult_rows` is THE DOOR: the
floats become int64 rows ``(y, x, mult_q)`` in Q16 there, once, and those rows
enter the sweep as render state exactly like the cone rows
(``Simulation.set_light`` -> ``PhysicsRunner.set_light_emit_mult`` ->
``LightChannels.emit_mult``), where the C++ scales the cell's thermal light
emission ``(em * mult_q) >> 16`` in every ordinate (isotropic, design v1 of
D; the directional D2 is not built). ``light_q`` is not digested; heat never
reads the rows.

Credits: B.M. Cetegen, T.A. Ahmed, "Experiments on the periodic instability
of buoyant plumes and pool fires", Combustion and Flame 93 (1993) -- the
puffing frequency the shared term follows (archived with the flame's
references, ``docs/papers/fire_vfx_references.md``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import numpy as np

from . import fire_vfx_ref as ref
from . import frame_lights

#: |z| up to this many standard deviations maps linearly onto the amplitude;
#: rarer peaks saturate at +-amplitude (symmetric, so the mean is kept).
CLIP_SIGMA = 2.0
#: sinusoids per tile in the tile's own term (one per 1/N_OSC of the band)
N_OSC = 8
#: their weighting: amplitude ~ (band_lo / f) ** PINK_EXP (0 = flat, 0.5 = pink)
PINK_EXP = 0.25


@dataclass(frozen=True)
class FlickerSettings:
    """``[render.flicker]`` -- the schema and its defaults (config.toml holds
    the live values; Ctrl+R / F5 re-read them every frame)."""
    enabled: bool = True
    amplitude: float = 0.3        # the most the light leaves 1: m in [1-a, 1+a]
    puff_share: float = 0.1       # the cluster puffing's share of the variance
    band_lo_hz: float = 0.8       # the tile's own flicker band, Hz
    band_hi_hz: float = 5.0
    seed: int = 8086


def settings_from_config(cfg) -> FlickerSettings:
    """``[render.flicker]`` -> settings (schema types kept; a missing key
    falls back to the schema default). Raises ValueError naming the key for
    a value outside its range: an amplitude in [0, 1] (above 1 the light
    would go negative -- and 1 + a is the sweep door's 2 ONE), a share in
    [0, 1], and 0 < band_lo_hz < band_hi_hz <= 10 (the 24 Hz tick's Nyquist
    rate is 12 Hz: a higher band would alias into a strobe)."""
    sec = getattr(getattr(cfg, "render", None), "flicker", None)
    base = FlickerSettings()
    kw = {}
    for f in fields(FlickerSettings):
        d = getattr(base, f.name)
        kw[f.name] = (d if sec is None or not hasattr(sec, f.name)
                      else type(d)(getattr(sec, f.name)))
    s = FlickerSettings(**kw)
    if not (0.0 <= s.amplitude <= 1.0):
        raise ValueError(f"[render.flicker] amplitude must be in [0, 1], "
                         f"got {s.amplitude!r}")
    if not (0.0 <= s.puff_share <= 1.0):
        raise ValueError(f"[render.flicker] puff_share must be in [0, 1], "
                         f"got {s.puff_share!r}")
    if not (0.0 < s.band_lo_hz < s.band_hi_hz <= 10.0):
        raise ValueError(f"[render.flicker] band must satisfy 0 < band_lo_hz < "
                         f"band_hi_hz <= 10, got {s.band_lo_hz!r}, "
                         f"{s.band_hi_hz!r}")
    return s


def burning_tiles(sources, window) -> Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray]:
    """The burning tiles of a flame pack, in MAP coordinates: ``(ys, xs, hz,
    phase)`` -- each tile's cluster puffing frequency and phase. ``sources``
    is the pack's ``FlameSources`` (window-local), ``window`` its (x0, y0,
    x1, y1)."""
    by, bx = np.nonzero(sources.burning)
    hz = sources.burn_hz[by, bx]
    ph = sources.burn_phase[by, bx]
    return (by + int(window[1])).astype(np.int64), \
        (bx + int(window[0])).astype(np.int64), hz, ph


def _mix01(keys, salt: int) -> np.ndarray:
    """Deterministic hash -> [0, 1): the FULL splitmix64 finalizer (Steele,
    Lea & Flood 2014) on ``key * phi + salt * c``. Not ``fire_vfx_ref.
    _hash01``: that one skips the first xor-shift, and nearby keys come out
    nearly equal (F4b measured tiles 572 / 574 at 0.3282 / 0.3276) -- which
    here would hand two tiles one frequency."""
    with np.errstate(over="ignore"):         # wrap-around IS the hash
        z = (np.asarray(keys, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             + np.uint64(int(salt) & 0xFFFFFFFFFFFFFFFF)
             * np.uint64(0xD1B54A32D192ED03))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


_OSC_CACHE: dict = {}


def _tile_oscillators(keys: np.ndarray, s: FlickerSettings):
    """Per tile: (N, N_OSC) frequencies (Hz), phases (rad) and weights. A
    pure function of (keys, seed, band); the last answer is cached, since the
    burning set changes far more rarely than the tick."""
    keys = np.asarray(keys, dtype=np.int64)
    ck = (keys.tobytes(), int(s.seed), float(s.band_lo_hz), float(s.band_hi_hz))
    hit = _OSC_CACHE.get("last")
    if hit is not None and hit[0] == ck:
        return hit[1]
    out = _tile_oscillators_uncached(keys, s)
    _OSC_CACHE["last"] = (ck, out)
    return out


def _tile_oscillators_uncached(keys: np.ndarray, s: FlickerSettings):
    lo, hi = float(s.band_lo_hz), float(s.band_hi_hz)
    j = np.arange(N_OSC, dtype=np.float64)
    u = np.stack([_mix01(keys, s.seed + 101 * k) for k in range(N_OSC)], axis=-1)
    f = lo * (hi / lo) ** ((j + u) / N_OSC)
    ph = 2.0 * np.pi * np.stack(
        [_mix01(keys, s.seed + 101 * k + 53) for k in range(N_OSC)], axis=-1)
    a = (lo / f) ** PINK_EXP
    return f, ph, a


def tile_term(keys: np.ndarray, t: float, s: FlickerSettings) -> np.ndarray:
    """``z_tile`` at VFX time ``t`` for tiles with map linear indices
    ``keys``: unit variance, zero mean."""
    f, ph, a = _tile_oscillators(keys, s)
    cyc = np.mod(f * float(t), 1.0)                    # long play: reduce first
    z = (a * np.sin(2.0 * np.pi * cyc + ph)).sum(axis=-1)
    return z / np.sqrt(0.5 * (a * a).sum(axis=-1))


def puff_term(hz: np.ndarray, phase: np.ndarray, t: float,
              dials) -> np.ndarray:
    """``z_puff`` at ``t``: the cluster's puffing (the flame's own theta),
    unit variance; 0 where a tile has no puffing."""
    hz = np.asarray(hz, dtype=np.float64)
    th = ref.puff_theta(hz, np.asarray(phase, dtype=np.float64), float(t), dials)
    return np.where(hz > 0.0, np.sqrt(2.0) * np.sin(th), 0.0)


def multipliers(ys, xs, hz, phase, t: float, s: FlickerSettings, dials,
                map_w: int) -> np.ndarray:
    """THE FLICKER: one float multiplier per burning tile at VFX time ``t``
    (the module docstring's law). ``ys``/``xs`` map tile coordinates, ``hz``/
    ``phase`` the tile's cluster puffing, ``dials`` the flame's
    ``FireVFXDials`` (its puffing wobble)."""
    ys = np.asarray(ys, dtype=np.int64)
    xs = np.asarray(xs, dtype=np.int64)
    if ys.size == 0:
        return np.zeros(0, dtype=np.float64)
    keys = ys * int(map_w) + xs
    share = float(s.puff_share)
    z = (np.sqrt(share) * puff_term(hz, phase, t, dials)
         + np.sqrt(1.0 - share) * tile_term(keys, t, s))
    return 1.0 + float(s.amplitude) * np.clip(z / CLIP_SIGMA, -1.0, 1.0)


class FireFlicker:
    """The renderer's flicker state: the settings, the live key's override
    (Alt+F3, HUMAN-TEST 2) and the last computation's cost. :meth:`rows` is
    what the game's per-tick light assembly reads."""

    def __init__(self, settings: FlickerSettings = FlickerSettings()):
        self.settings = settings
        self.enabled_override: Optional[bool] = None
        self.last_ms = 0.0
        self.last_n = 0

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_override if self.enabled_override is not None
                    else self.settings.enabled)

    def toggle(self) -> bool:
        """The live on/off key (G3 F4b, HUMAN-TEST 2)."""
        self.enabled_override = not self.enabled
        return self.enabled

    def refresh_settings(self, settings: FlickerSettings) -> None:
        self.settings = settings

    def rows(self, pack, t: float, dials, grid_w: int,
             grid_h: int) -> Optional[np.ndarray]:
        """The sweep's emission-multiplier rows for this tick -- (n, 3) int64
        ``(y, x, mult_q)`` through THE DOOR (``frame_lights.emit_mult_rows``)
        -- or None: disabled, no flame pack, or nothing burning (the sweep then
        runs exactly as before F4b)."""
        t0 = time.perf_counter()
        out = None
        n = 0
        if self.enabled and pack is not None and pack.sources is not None \
                and pack.sources.burn_hz is not None:
            ys, xs, hz, ph = burning_tiles(pack.sources, pack.window)
            n = int(ys.size)
            if n:
                m = multipliers(ys, xs, hz, ph, t, self.settings, dials, grid_w)
                out = frame_lights.emit_mult_rows(ys, xs, m, grid_w, grid_h)
        self.last_n = n
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return out


__all__ = [
    "CLIP_SIGMA",
    "N_OSC",
    "PINK_EXP",
    "FlickerSettings",
    "settings_from_config",
    "burning_tiles",
    "tile_term",
    "puff_term",
    "multipliers",
    "FireFlicker",
]
