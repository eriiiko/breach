"""Game-temperature -> (linear-RGB chroma, HDR intensity) — the blackbody primitive.

Fire & Heat Beauty arc, beat B1 (design:
docs/fire_b1_blackbody_fire_lights_design_2026-07-21.md §1; research base:
docs/fire_rendering_research.md Q1 + pick (1)). RENDER-ONLY, determinism-EXEMPT:
this is float math on a copy of the (read-only) temperature field and never
writes any sim state. Importable without ``breach_physics`` (pure numpy), so it
is headless-testable.

One module, one job: map the dequantized game temperature ΔT (``gmap.temperature``
in Q16.16 above ambient) to a *normalized chroma* (max channel = 1) plus a
*separate HDR intensity* that carries the T⁴ brightness the chromaticity fits
deliberately drop. The overlay and the fire light sources both read this ONE code
path, so their colour agrees by construction.

Two moves, from the research:

- **Chromaticity** — Tanner Helland's piecewise Kelvin→RGB fit (log/power pieces,
  valid ~1000–40000 K), as refined by Neil Bartlett. These are white-balance fits:
  they give hue/chroma only, NOT brightness — the header note in every source.
  We normalize each entry so the max channel is 1, i.e. pure chroma.
- **Intensity** — the T⁴ half the fits omit, shaped as Macklin's blackbody-render
  exposure move: a reference temperature (~3000 K) sets the unit level, a power
  ``p`` (Stefan-Boltzmann flavour) drives the rise, and an HDR ceiling ``i_max``
  keeps the 16F light pipeline + ACES tone-map (no bloom, per Erik) in range.

Citations (repo rule — credit the source):
  - Tanner Helland, "How to Convert Temperature (K) to RGB" (2012):
    https://tannerhelland.com/2012/09/18/convert-temperature-rgb-algorithm-code.html
  - Neil Bartlett, "color-temperature" refinement:
    https://github.com/neilbartlett/color-temperature
  - Miles Macklin, "Blackbody Rendering" (2010) — reference-temperature exposure:
    https://blog.mmacklin.com/2010/12/29/blackbody-rendering/
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from simulation.fire_fixed import FP_ONE_F as TEMP_SCALE

# Q16.16 scale of the temperature field (materials.TEMP_SCALE == HEAT_SCALE).
# gmap.temperature is int32 ΔT above ambient in this fixed-point domain.
# ONE shared constant (cleanup #15, was independently re-declared as a float
# literal in blackbody/cold_overlay/fire_lights/hover_readout with a
# "MUST match" comment): sourced from simulation.fire_fixed.FP_ONE_F, the
# pure-numpy leaf module (no breach_physics, no pyray) that already proves
# this exact value across the shared Q16.16 domain (fire/water/wave/
# atmosphere/gas/heat all == 65536). cold_overlay.py / fire_lights.py /
# hover_readout.py import this SAME name from the SAME place.


def aces_tonemap(x: np.ndarray) -> np.ndarray:
    """ACES filmic tone-map (Narkowicz approximation), per-channel.

    Compresses HDR emissive (chroma * intensity, which runs past 1) toward
    [0, 1] while staying punchy/saturated, instead of per-channel clipping that
    would hue-shift a bright warm glow toward white. Same constants as
    shaders/lighting.fs :func:`aces_tonemap`, so the CPU heat overlay and the
    GPU-lit scene share one tone-map curve. Render-layer, determinism-exempt.
    """
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = np.asarray(x, dtype=np.float32)
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def pack_emissive_rgba(ramp: "BlackbodyRamp", temperature: np.ndarray,
                       max_alpha: int = 220,
                       intensity_mod: "Optional[np.ndarray]" = None
                       ) -> np.ndarray:
    """Q16.16 temperature field -> additive RGBA8 texture bytes (H, W, 4).

    The pure-numpy packing behind :class:`renderer.overlays.HeatFieldOverlay`,
    factored out here (pyray-free) so it is headless-testable without a GL
    context. Maps temperature through the black-body ramp, ACES tone-maps the
    HDR emissive (``chroma * intensity``), then splits the tone-mapped colour
    into a peak-normalized hue (RGB) and a brightness weight (alpha) so the
    RGB-only additive draw's ``rgb * alpha`` reconstructs the tone-mapped
    emissive. Cold tiles -> brightness ~0 -> alpha 0 -> invisible. RENDER-ONLY:
    ``temperature`` is read, never written.

    ``intensity_mod`` (Fire & Heat Beauty B2 P4, the dirty-Planck speckle seam):
    an optional (H, W) multiplicative field applied to the black-body INTENSITY
    BEFORE the tone-map — the honest spot for a sooty flame (a dirtier flame
    radiates less; ACES then compresses the mottle more in the white-hot core,
    more visible at the cooler edges). ``None`` (the B1 default) is a strict
    no-op: the pack is byte-for-byte identical to before. Because it scales the
    intensity, a cold tile (intensity 0) stays invisible regardless of the mod.
    See :mod:`renderer.speckle`.
    """
    rgb, inten = ramp.chroma_intensity(temperature)
    if intensity_mod is not None:
        inten = inten * np.asarray(intensity_mod, dtype=inten.dtype)
    emissive = rgb * inten[..., None]              # HDR linear (can exceed 1)
    tm = aces_tonemap(emissive)                    # (h,w,3) 0..1 — the add colour
    bright = tm.max(axis=-1)                        # (h,w) 0..1 brightness
    denom = np.where(bright > 1e-6, bright, 1.0)
    rgb_disp = tm / denom[..., None]                # hue at peak = 1
    h, w = bright.shape
    packed = np.empty((h, w, 4), dtype=np.uint8)
    packed[..., 0] = (rgb_disp[..., 0] * 255.0).astype(np.uint8)
    packed[..., 1] = (rgb_disp[..., 1] * 255.0).astype(np.uint8)
    packed[..., 2] = (rgb_disp[..., 2] * 255.0).astype(np.uint8)
    packed[..., 3] = (bright * float(max_alpha)).astype(np.uint8)
    return packed


def _tanner_helland_chroma(kelvin: np.ndarray) -> np.ndarray:
    """Vectorized Tanner Helland / Bartlett Kelvin -> normalized RGB chroma.

    Returns an (N, 3) float32 array; each row's max channel == 1 (pure chroma,
    brightness carried separately). The classic piecewise fit works on
    ``kelvin / 100`` and yields 0..255; we rescale to 0..1 then normalize.
    Determinism-exempt (render-layer float / libm allowed here).
    """
    t = np.asarray(kelvin, dtype=np.float64) / 100.0

    # --- Red ---
    red = np.where(
        t <= 66.0,
        255.0,
        329.698727446 * np.power(np.clip(t - 60.0, 1e-9, None), -0.1332047592),
    )

    # --- Green ---
    green = np.where(
        t <= 66.0,
        99.4708025861 * np.log(np.clip(t, 1e-9, None)) - 161.1195681661,
        288.1221695283 * np.power(np.clip(t - 60.0, 1e-9, None), -0.0755148492),
    )

    # --- Blue ---
    blue = np.where(
        t >= 66.0,
        255.0,
        np.where(
            t <= 19.0,
            0.0,
            138.5177312231 * np.log(np.clip(t - 10.0, 1e-9, None)) - 305.0447927307,
        ),
    )

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.clip(rgb, 0.0, 255.0) / 255.0
    # Normalize so the max channel is 1.0 -> pure chroma (brightness is the
    # separate intensity ramp). Guard the all-zero degenerate row (never happens
    # for kelvin > 0, but keep the divide safe).
    peak = np.max(rgb, axis=-1, keepdims=True)
    peak = np.where(peak > 1e-6, peak, 1.0)
    return (rgb / peak).astype(np.float32)


class BlackbodyRamp:
    """Baked LUT: game temperature ΔT -> (normalized chroma, HDR intensity).

    Build once at renderer startup; every per-frame lookup is an integer index
    into the two LUTs. The vectorized (overlay) and scalar (light-source) APIs
    index the SAME LUTs by the SAME nearest-index rule, so ``light_color`` and a
    single cell of ``chroma_intensity`` return bit-identical values (the "two
    wiring points agree by construction" invariant).

    All parameters are render-side config (``[render.blackbody]``), not sim
    constants — they are tuned by eye with Erik, never gated.
    """

    def __init__(
        self,
        kelvin_floor: float = 800.0,
        kelvin_ceil: float = 10000.0,
        lut_size: int = 256,
        kelvin_ambient: float = 293.0,
        k_temp_to_kelvin: float = 3.0,
        kelvin_glow_min: float = 800.0,
        kelvin_ref: float = 3000.0,
        intensity_exponent: float = 4.0,
        intensity_max: float = 8.0,
    ):
        self.kelvin_floor = float(kelvin_floor)
        self.kelvin_ceil = float(kelvin_ceil)
        self.lut_size = int(lut_size)
        self.kelvin_ambient = float(kelvin_ambient)
        self.k_temp_to_kelvin = float(k_temp_to_kelvin)
        self.kelvin_glow_min = float(kelvin_glow_min)
        self.kelvin_ref = float(kelvin_ref)
        self.intensity_exponent = float(intensity_exponent)
        self.intensity_max = float(intensity_max)

        if self.lut_size < 2:
            raise ValueError("lut_size must be >= 2")
        if self.kelvin_ceil <= self.kelvin_floor:
            raise ValueError("kelvin_ceil must exceed kelvin_floor")
        if self.kelvin_ref <= self.kelvin_glow_min:
            raise ValueError("kelvin_ref must exceed kelvin_glow_min")

        kelvins = np.linspace(self.kelvin_floor, self.kelvin_ceil,
                              self.lut_size)
        self._chroma_lut = _tanner_helland_chroma(kelvins)          # (N, 3)
        self._inten_lut = self._intensity_curve(kelvins).astype(    # (N,)
            np.float32)
        self._kelvins = kelvins.astype(np.float32)                  # for tests

    # ---- core curves -----------------------------------------------------

    def _intensity_curve(self, kelvin: np.ndarray) -> np.ndarray:
        """Kelvin -> HDR intensity (the T⁴ ramp the chromaticity fits omit).

        ``intensity = clip((max(kelvin - glow_min, 0) / (ref - glow_min))^p,
        0, i_max)``. The base ratio is floored at 0 BEFORE the power so an even
        exponent cannot resurrect sub-glow-min temperatures into a false glow
        (Macklin's reference-temperature exposure, Stefan-Boltzmann ``p``).
        """
        ratio = np.clip(
            (np.asarray(kelvin, dtype=np.float64) - self.kelvin_glow_min)
            / (self.kelvin_ref - self.kelvin_glow_min),
            0.0, None)
        return np.clip(np.power(ratio, self.intensity_exponent),
                       0.0, self.intensity_max)

    def _kelvin_from_tgame(self, t_game):
        """Dequantized game ΔT -> pseudo-Kelvin (the honest 'white for extremes'
        dial): ``kelvin = kelvin_ambient + k_temp_to_kelvin * T_game``."""
        return self.kelvin_ambient + self.k_temp_to_kelvin * t_game

    def _index(self, kelvin):
        """Kelvin -> clamped nearest LUT index (same rule for both APIs)."""
        frac = ((kelvin - self.kelvin_floor)
                / (self.kelvin_ceil - self.kelvin_floor))
        idx = np.rint(frac * (self.lut_size - 1))
        return np.clip(idx, 0, self.lut_size - 1).astype(np.int64)

    # ---- public API ------------------------------------------------------

    def chroma_intensity(
        self, temp_field: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Q16.16 temperature field (H, W) -> (chroma RGB (H, W, 3), intensity
        (H, W)). RENDER-ONLY: ``temp_field`` is read, never written.

        Dequantized in float64 (not float32) so the per-cell index math matches
        the scalar :meth:`light_color` path bit-for-bit — the two wiring points
        must agree exactly, not merely closely."""
        t_game = np.asarray(temp_field, dtype=np.float64) / TEMP_SCALE
        idx = self._index(self._kelvin_from_tgame(t_game))
        return self._chroma_lut[idx], self._inten_lut[idx]

    def light_color(self, t_game: float) -> Tuple[Tuple[float, float, float],
                                                  float]:
        """Scalar dequantized game ΔT -> ((r, g, b) chroma, intensity).

        The per-source colour path for fire lights. Shares ``_index`` with the
        vectorized overlay so a light and the overlay tile beneath it agree.
        """
        idx = int(self._index(self._kelvin_from_tgame(float(t_game))))
        r, g, b = self._chroma_lut[idx]
        return (float(r), float(g), float(b)), float(self._inten_lut[idx])

    # ---- config binding --------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "BlackbodyRamp":
        """Build from a config object, reading render-only dials from
        ``[render.blackbody]`` (getattr defaults — the renderer's standard
        optional-section idiom, a level/config without the section still
        gets the design defaults) and the game-T -> Kelvin map itself from
        ``[physics.temperature_scale]`` (the ONE canonical table shared with
        radiation + the tuning tools — design
        docs/temperature_scale_unification_design_2026-08-13.md §2/§3d)."""
        render = getattr(cfg, "render", None)
        bb = getattr(render, "blackbody", None)
        g = lambda name, default: float(getattr(bb, name, default))
        physics = getattr(cfg, "physics", None)
        ts = getattr(physics, "temperature_scale", None)
        gt = lambda name, default: float(getattr(ts, name, default))
        return cls(
            kelvin_floor=g("kelvin_floor", 800.0),
            kelvin_ceil=g("kelvin_ceil", 10000.0),
            lut_size=int(getattr(bb, "lut_size", 256)),
            kelvin_ambient=gt("kelvin_ambient", 293.0),
            k_temp_to_kelvin=gt("k_temp_to_kelvin", 3.0),
            kelvin_glow_min=g("kelvin_glow_min", 800.0),
            kelvin_ref=g("kelvin_ref", 3000.0),
            intensity_exponent=g("intensity_exponent", 4.0),
            intensity_max=g("intensity_max", 8.0),
        )
