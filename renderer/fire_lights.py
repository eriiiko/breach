"""Fire tiles -> brightest-K render light sources (tier 2, ray-tracing glow).

Fire & Heat Beauty arc, beat B1 (design
docs/fire_b1_blackbody_fire_lights_design_2026-07-21.md §3). RENDER-ONLY: reads
``gmap.temperature`` and emits ``bp.LightSource`` PARAMETER DICTS (main.py maps
them onto the compiled struct with the same setattr loop it uses for level
lights). Pure numpy + a duck-typed ``ramp`` (anything with ``light_color`` —
:class:`renderer.blackbody.BlackbodyRamp`); imports nothing from the renderer
package, so it is headless-testable in isolation.

The pipeline (design §3):
  1. candidates  — tiles with game-ΔT >= ``t_light_min``,
  2. NMS         — keep only tiles that are the max of their NxN neighbourhood
                   (``nms_window`` = 3 or 5) so a blaze doesn't oversubscribe,
  3. brightest-K — take the ``max_lights`` hottest peaks (row-major tie-break),
  4. per light   — omni source at the tile centre, ``max_range`` = ``light_range``,
                   colour + intensity from the shared black-body ramp.

STRUCTURAL invariants (determinism safety, mirrors src/level_lights.py):
``heat = 0.0`` and ``jitter = 0.0`` are HARD-PINNED. Render fire lights must
NEVER write the synced ``heat`` channel — the sim already casts fire heat
separately (physics_runner.cast_fire_heat); a render-side heat deposit would
double-count AND diverge interactive sessions from their headless replays
(``heat`` is the only synced ray output). Flicker (jitter) is a later dial.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from simulation.fire_fixed import FP_ONE_F as TEMP_SCALE

# Q16.16 scale of the temperature field — matches renderer.blackbody.TEMP_SCALE
# and materials.TEMP_SCALE (the shared temperature/heat fixed-point domain).
# ONE shared constant (cleanup #15): sourced from simulation.fire_fixed.FP_ONE_F
# — see renderer/blackbody.py's header for why that module.
TAU = 2.0 * math.pi


def _local_max_mask(field: np.ndarray, window: int) -> np.ndarray:
    """(H, W) bool: True where ``field`` equals the max of its window x window
    neighbourhood (a non-maximum-suppression peak test).

    Pure numpy via a sliding-window view over an edge-replicated pad, so border
    tiles compare only against in-bounds neighbours. Since a tile is inside its
    own window, ``field == local_max`` exactly at peaks. On an exact-tie plateau
    every plateau tile passes — bounded downstream by the brightest-K cap.
    """
    if window <= 1:
        return np.ones(field.shape, dtype=bool)
    r = window // 2
    padded = np.pad(field, r, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    local_max = view.max(axis=(-1, -2))
    return field >= local_max


def select_fire_light_params(
    temp_field: np.ndarray,
    ramp,
    *,
    t_light_min: float = 250.0,
    nms_window: int = 3,
    max_lights: int = 16,
    light_range: float = 18.0,
    light_gain: float = 1.0,
) -> Tuple[List[dict], int]:
    """Temperature field -> (list of LightSource param dicts, n_peaks).

    ``n_peaks`` is the NMS peak count BEFORE the cap, so the caller can show
    "K / n" on the HUD and know when the cap truncated (no silent caps). RENDER-
    ONLY: ``temp_field`` is read, never written.
    """
    temp_field = np.asarray(temp_field)
    threshold = int(round(t_light_min * TEMP_SCALE))   # compare in Q16.16 int
    candidate = temp_field >= threshold
    if not candidate.any():
        return [], 0

    peaks = _local_max_mask(temp_field, int(nms_window)) & candidate
    ys, xs = np.nonzero(peaks)          # row-major (C order)
    n_peaks = int(ys.size)
    if n_peaks == 0:
        return [], 0

    temps = temp_field[ys, xs]
    # Brightest-K by temperature, descending; stable sort keeps row-major order
    # on exact ties (deterministic selection).
    order = np.argsort(-temps, kind="stable")[:int(max_lights)]

    lights: List[dict] = []
    for k in order:
        row = int(ys[k])
        col = int(xs[k])
        t_game = float(temp_field[row, col]) / TEMP_SCALE
        (r, g, b), intensity = ramp.light_color(t_game)
        lights.append({
            "x": col + 0.5,                 # tile centre (matches cast_fire_heat)
            "y": row + 0.5,
            "max_range": float(light_range),
            "intensity": float(intensity) * float(light_gain),
            "color": (float(r), float(g), float(b)),
            "angle_center": 0.0,
            "angle_spread": TAU,            # omni
            # STRUCTURAL zeroes — see module docstring. Render lights never write
            # the synced heat channel and never pull C++ RNG jitter.
            "heat": 0.0,
            "jitter": 0.0,
        })
    return lights, n_peaks


class FireLightSelector:
    """Config-bound wrapper around :func:`select_fire_light_params`.

    Built once from ``[render.fire_lights]``; call :meth:`select` each frame
    with the current temperature field + the renderer's black-body ramp. Emits
    param dicts (bp-free) — main.py builds the compiled ``bp.LightSource`` structs
    with its existing setattr loop.
    """

    def __init__(self, enabled: bool = True, t_light_min: float = 250.0,
                 nms_window: int = 3, max_lights: int = 16,
                 light_range: float = 18.0, light_gain: float = 1.0):
        self.enabled = bool(enabled)
        self.t_light_min = float(t_light_min)
        self.nms_window = int(nms_window)
        self.max_lights = int(max_lights)
        self.light_range = float(light_range)
        self.light_gain = float(light_gain)
        if self.nms_window < 1 or self.nms_window % 2 == 0:
            raise ValueError("nms_window must be a positive odd integer (3 or 5)")
        if self.max_lights < 0:
            raise ValueError("max_lights must be >= 0")

    def select(self, temp_field: np.ndarray, ramp) -> Tuple[List[dict], int]:
        if not self.enabled or self.max_lights == 0:
            return [], 0
        return select_fire_light_params(
            temp_field, ramp,
            t_light_min=self.t_light_min, nms_window=self.nms_window,
            max_lights=self.max_lights, light_range=self.light_range,
            light_gain=self.light_gain)

    @classmethod
    def from_config(cls, cfg) -> "FireLightSelector":
        render = getattr(cfg, "render", None)
        fl = getattr(render, "fire_lights", None)
        return cls(
            enabled=bool(getattr(fl, "enabled", True)),
            t_light_min=float(getattr(fl, "t_light_min", 250.0)),
            nms_window=int(getattr(fl, "nms_window", 3)),
            max_lights=int(getattr(fl, "max_lights", 16)),
            light_range=float(getattr(fl, "light_range", 18.0)),
            light_gain=float(getattr(fl, "light_gain", 1.0)),
        )
