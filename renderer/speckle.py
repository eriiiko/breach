"""Dirty-Planck speckle — the CPU-side blackbody mottle (Fire & Heat Beauty arc,
B2 P4). Erik's *spräcklig* ("speckled") idea: a real flame is not a smooth
black-body wash, it is soot-dirty and boils — luminous grains where incandescent
soot glows, dark grains where denser soot self-absorbs. This module modulates the
B1 :class:`renderer.overlays.HeatFieldOverlay` emissive at PACK TIME with a moving
two-layer advected-noise field, so the flame reads as sooty, alive, and — in the
``soot`` mode — visibly dirtier exactly where the chemistry put more soot.

RENDER-ONLY, determinism-EXEMPT and pyray-free (numpy only): it reads copies of
read-only sim fields (the soot/steam density planes) and returns a multiplicative
intensity field; it never writes sim state, owns no GPU resource, and cannot move
a golden. Being pyray-free it is headless-testable without a GL context (the
``advected_noise`` / ``blackbody`` pattern).

The cross-layer seam (critique resolution, design §5): ``HeatFieldOverlay`` is a
CPU-packed additive texture with NO fragment shader, so the speckle does NOT
sample P3's ``shaders/gas_medium.fs``. Instead the SAME two-layer advected-phase
recipe is evaluated CPU-side at grid resolution (256² numpy is cheap, and per-tile
flame mottle is grid-scale anyway) via the shared :mod:`renderer.advected_noise`
module, and modulates the overlay's colour/intensity in :func:`renderer.blackbody.
pack_emissive_rgba` (the ``intensity_mod`` seam).

Two variants behind one ``mode`` toggle (design §5), the A/B Erik picks by eye:
  * ``off``   — identity; the overlay packs byte-for-byte as B1 (no-op).
  * ``noise`` — pure render noise modulating the black-body intensity everywhere
                the plume glows (the naive baseline for the A/B).
  * ``soot``  — the DIRTY PLANCK: the mottle AMPLITUDE is seeded by the real local
                soot density (``gmap.gas[SMOKE]``). Chemistry decides where the
                flame is dirty, so a starving, sooting fire shows its soot in the
                flame colour for free. Steam (``gmap.gas[STEAM]``) is a clean gas —
                it is allowed at most ``_STEAM_MOTTLE_FRAC`` of soot's weight so a
                steam puff never sparkles like dirty flame (design's steam bound).

HARD RULE (lit-search): the speckle MUST MOVE WITH THE FLOW — a static speckle
reads as a screen overlay and is wrong. Motion comes from the shared two-layer
crossfade clocked on the SIM TICK (never wall time, so replays/spectators render
identical flame): the two phase-offset layers cross-dissolve (the Flow-Noise
"boil" that happens even at zero wind, since the plume is near-stationary — P3's
finding) and drift with a gentle buoyant rise (flame licks upward). The pattern is
a pure function of the integer tick, so consecutive ticks differ → it moves.

Credit (repo rule — cite what a file implements):
  - "Dirty Planck" / speckled black-body emission: the soot-loaded flame
    luminance model, docs/research/smoke_render_litsearch_2026-07-21.md §4
    ("spräcklig blackbody") + docs/blackbody_smoke_and_rendering_brainstorm.md.
  - Two-layer advected-phase crossfade: reused from renderer.advected_noise
    (Alex Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010; Fabrice Neyret,
    "Advected Textures", SCA 2003; Perlin–Neyret "Flow Noise", SIGGRAPH 2001).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .advected_noise import bake_fbm_rgba, advection_phase, layer_ages_weight

# The A/B modes, indexed by the harness's stepped 0/1/2 slider + the game's
# F10 cycle. Order is load-bearing: idx 0->off, 1->noise, 2->soot (design §5).
MODES: Tuple[str, str, str] = ("off", "noise", "soot")

# --- render tuning constants (by-eye dials; the live sliders are `amp` + mode) --
# Soot density that saturates the `soot`-mode dirtiness to full amplitude. Real
# densities are dequantized reals; a moderate plume soot (~0.5) reads fully dirty.
# Render-only normalisation — Erik trims the visible depth with the `amp` slider.
_SOOT_REF = 0.5
# Steam's weight in the dirtiness seed, as a fraction of soot's. The design's
# "keep steam-side mottle <= ~10%": a clean steam puff may carry at most a faint
# low-frequency mottle, never the full dirty-flame sparkle (steam == id 0).
_STEAM_MOTTLE_FRAC = 0.1
# Speckle spatial scale. `wavelength_tiles` is the tunable; this maps it to fBm
# texels so the DOMINANT (amplitude-leading, coarsest-octave) feature spans a few
# grid tiles -> a LOW-FREQUENCY boil, not a per-pixel sparkle (design §5).
_SPECKLE_FREQ_TEXELS = 32.0
_DEFAULT_WAVELENGTH_TILES = 3.0
# Buoyant drift: how far (grid tiles) the mottle rises per crossfade cycle. Small
# -> the crossfade "boil" dominates and the drift is a gentle upward bias (heat
# rises). (dx, dy) in grid space; +dy makes the pattern rise on screen (row 0 top).
_DRIFT_TILES_PER_CYCLE = 1.0
_DRIFT_DIR = (0.0, 1.0)


# ---------------------------------------------------------------------------
# mode <-> index helpers (shared by the game F10 cycle + the harness slider)
# ---------------------------------------------------------------------------

def clamp_mode_idx(idx) -> int:
    """Any slider float / cycle int -> a valid mode index in {0, 1, 2}."""
    return int(np.clip(int(round(float(idx))), 0, len(MODES) - 1))


def mode_name(idx) -> str:
    """Mode index -> its name ('off' | 'noise' | 'soot')."""
    return MODES[clamp_mode_idx(idx)]


def mode_index(name: str) -> int:
    """Mode name -> its index (unknown -> 'soot', the shipped default)."""
    try:
        return MODES.index(str(name))
    except ValueError:
        return MODES.index("soot")


# ---------------------------------------------------------------------------
# amplitude field (WHERE + HOW HARD the flame speckles) — the chemistry seam
# ---------------------------------------------------------------------------

def dirtiness(soot: np.ndarray, steam: np.ndarray, *,
              soot_ref: float = _SOOT_REF,
              steam_frac: float = _STEAM_MOTTLE_FRAC) -> np.ndarray:
    """(soot, steam density planes) -> local dirtiness in [0, 1] (float32).

    ``clip((soot + steam_frac·steam) / soot_ref, 0, 1)`` — the DIRTY-PLANCK seed:
    soot drives the mottle, steam is admitted only at ``steam_frac`` (<= 0.1) of
    soot's weight so a pure-steam cell can never sparkle like dirty flame (the
    design's steam bound; steam == gas id 0). RENDER-ONLY: inputs are read."""
    soot = np.asarray(soot, dtype=np.float64)
    steam = np.asarray(steam, dtype=np.float64)
    d = (soot + float(steam_frac) * steam) / max(float(soot_ref), 1e-9)
    return np.clip(d, 0.0, 1.0).astype(np.float32)


def amplitude_field(mode: str, amp: float,
                    soot: np.ndarray, steam: np.ndarray, *,
                    soot_ref: float = _SOOT_REF,
                    steam_frac: float = _STEAM_MOTTLE_FRAC) -> np.ndarray:
    """Per-cell speckle amplitude (H, W) float32, in [0, amp].

    * ``off``   -> zeros (identity).
    * ``noise`` -> uniform ``amp`` (pure render noise — the naive A/B baseline).
    * ``soot``  -> ``amp · dirtiness(soot, steam)`` (chemistry-seeded).
    The pattern (the moving noise) is applied on top of this field elsewhere; here
    we only decide the local strength."""
    soot = np.asarray(soot, dtype=np.float32)
    shape = soot.shape
    if mode == "off" or amp <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    if mode == "noise":
        return np.full(shape, float(amp), dtype=np.float32)
    # soot (dirty Planck)
    return (float(amp) * dirtiness(soot, steam, soot_ref=soot_ref,
                                   steam_frac=steam_frac)).astype(np.float32)


# ---------------------------------------------------------------------------
# the moving pattern (HOW it boils/drifts) — the shared advected-phase recipe
# ---------------------------------------------------------------------------

def _sample_bilinear_wrap(tex: np.ndarray, u: np.ndarray,
                          v: np.ndarray) -> np.ndarray:
    """Bilinear sample of a tiling (Hf, Wf) field at float texel coords (u, v),
    wrapping on both axes (the fBm tiles seamlessly). Fully vectorised."""
    hf, wf = tex.shape
    u0 = np.floor(u); v0 = np.floor(v)
    fu = u - u0; fv = v - v0
    u0i = u0.astype(np.int64) % wf
    v0i = v0.astype(np.int64) % hf
    u1i = (u0i + 1) % wf
    v1i = (v0i + 1) % hf
    c00 = tex[v0i, u0i]; c10 = tex[v0i, u1i]
    c01 = tex[v1i, u0i]; c11 = tex[v1i, u1i]
    top = c00 * (1.0 - fu) + c10 * fu
    bot = c01 * (1.0 - fu) + c11 * fu
    return top * (1.0 - fv) + bot * fv


def advected_speckle(fbm_r: np.ndarray, grid_h: int, grid_w: int, *,
                     sim_tick: int, cycle_seconds: float, tps: float,
                     wavelength_tiles: float = _DEFAULT_WAVELENGTH_TILES,
                     drift_tiles_per_cycle: float = _DRIFT_TILES_PER_CYCLE,
                     drift_dir: Tuple[float, float] = _DRIFT_DIR) -> np.ndarray:
    """The moving speckle pattern, SIGNED in [-1, 1], shape (H, W) float32.

    Two fBm layers are sampled at a low-frequency scale, each advected by a gentle
    buoyant drift proportional to its crossfade AGE, and cross-dissolved by the
    shared ``layer_ages_weight`` (Vlachos ping-pong). The layers carry a
    half-texture UV offset so they are decorrelated -> the crossfade genuinely
    boils rather than pulsing. The clock is the SIM TICK (via
    ``advection_phase``), so the field is a pure function of the integer tick:
    replays render identical flame and consecutive ticks DIFFER (it moves)."""
    ph = advection_phase(int(sim_tick), float(cycle_seconds), float(tps))
    age0, age1, w0 = layer_ages_weight(ph.phase)

    hf, wf = fbm_r.shape
    texels_per_tile = _SPECKLE_FREQ_TEXELS / max(float(wavelength_tiles), 0.1)
    ys = np.arange(grid_h, dtype=np.float64)[:, None]
    xs = np.arange(grid_w, dtype=np.float64)[None, :]
    u_base = xs * texels_per_tile
    v_base = ys * texels_per_tile
    drift = float(drift_tiles_per_cycle) * texels_per_tile
    dux, duy = float(drift_dir[0]) * drift, float(drift_dir[1]) * drift

    # Layer 0 and layer 1 (half-texture offset + tau/2 phase offset via ages).
    n0 = _sample_bilinear_wrap(fbm_r,
                               u_base + dux * age0,
                               v_base + duy * age0)
    n1 = _sample_bilinear_wrap(fbm_r,
                               u_base + dux * age1 + 0.5 * wf,
                               v_base + duy * age1 + 0.5 * hf)
    noise01 = w0 * n0 + (1.0 - w0) * n1          # [0, 1]
    return (2.0 * noise01 - 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# the field owner (renderer-side; still pyray-free — no GPU resource)
# ---------------------------------------------------------------------------

class SpeckleField:
    """Owns the baked fBm + the speckle dials; produces the per-frame intensity
    modulation the :class:`~renderer.overlays.HeatFieldOverlay` applies at pack
    time. CPU-only (no GPU texture) — the modulation is a numpy field handed to
    ``pack_emissive_rgba(..., intensity_mod=...)``.

    Tunables are plain attributes mutated live by the harness sliders / the game
    F10 cycle (the ``GasMediumOverlay`` / ``GasDetailPass`` precedent):
      * ``mode_idx`` — stepped 0/1/2 (off/noise/soot); ``mode`` is the name view.
      * ``amp``      — mottle depth (design default 0.25).
    """

    def __init__(self, grid_h: int, grid_w: int, *,
                 mode: str = "soot", amp: float = 0.25,
                 cycle_seconds: float = 2.5, ticks_per_second: float = 24.0,
                 wavelength_tiles: float = _DEFAULT_WAVELENGTH_TILES,
                 fbm_size: int = 256, fbm_octaves: int = 4,
                 fbm_persistence: float = 0.56):
        self.h = int(grid_h)
        self.w = int(grid_w)
        self.mode_idx = mode_index(mode)
        self.amp = float(amp)
        self.cycle_seconds = float(cycle_seconds)
        self.tps = float(ticks_per_second)
        self.wavelength_tiles = float(wavelength_tiles)
        # Bake the tiling fBm ONCE (deterministic; same recipe as the P3 shader
        # texture). Keep only the R channel (coverage) as a [0,1] float field —
        # this is a CPU array, NOT a GPU texture (the speckle never touches GL).
        self._fbm_r = (bake_fbm_rgba(fbm_size, fbm_octaves,
                                     fbm_persistence)[..., 0].astype(np.float64)
                       / 255.0)

    @property
    def mode(self) -> str:
        return mode_name(self.mode_idx)

    def cycle_mode(self) -> str:
        """Advance off -> noise -> soot -> off (the F10 live A/B). Returns the
        new mode name (so the caller can print/HUD it)."""
        self.mode_idx = (clamp_mode_idx(self.mode_idx) + 1) % len(MODES)
        return self.mode

    def modulation(self, soot: np.ndarray, steam: np.ndarray, *,
                   sim_tick: int) -> Optional[np.ndarray]:
        """(soot, steam density planes) -> the (H, W) float32 multiplicative
        intensity field for ``pack_emissive_rgba``, or ``None`` when the speckle
        is a no-op (``off`` mode or zero amplitude) so the overlay packs
        byte-for-byte as B1.

        ``mod = clip(1 + amplitude · signed_noise, 0, inf)`` — a signed mottle
        around 1.0: bright grains where incandescent soot glows, dark grains where
        it self-absorbs. Multiplied onto the black-body intensity BEFORE the ACES
        tone-map (the honest spot: a dirtier flame radiates less), so cold tiles
        (intensity 0) stay invisible regardless of the speckle."""
        if self.mode == "off" or self.amp <= 0.0:
            return None
        amp_f = amplitude_field(self.mode, self.amp, soot, steam)
        if not np.any(amp_f > 0.0):
            return None
        signed = advected_speckle(
            self._fbm_r, self.h, self.w,
            sim_tick=int(sim_tick), cycle_seconds=self.cycle_seconds,
            tps=self.tps, wavelength_tiles=self.wavelength_tiles)
        mod = 1.0 + amp_f * signed
        return np.maximum(mod, 0.0).astype(np.float32)

    @classmethod
    def from_config(cls, grid_h: int, grid_w: int, cfg) -> "SpeckleField":
        """Build from ``[render.speckle]`` (+ the sim clock for the tick rate),
        getattr-guarded so a config without the block still gets the design
        defaults (mode='soot', amp=0.25)."""
        render = getattr(cfg, "render", None)
        sp = getattr(render, "speckle", None)
        gd = getattr(render, "gas_detail", None)     # share the crossfade cycle
        clock = getattr(cfg, "clock", None)
        return cls(
            grid_h, grid_w,
            mode=str(getattr(sp, "mode", "soot")),
            amp=float(getattr(sp, "amp", 0.25)),
            cycle_seconds=float(getattr(gd, "cycle_seconds", 2.5)),
            ticks_per_second=float(getattr(clock, "ticks_per_second", 24)),
        )


__all__ = [
    "MODES", "clamp_mode_idx", "mode_name", "mode_index",
    "dirtiness", "amplitude_field", "advected_speckle", "SpeckleField",
]
