"""THE LIGHT READ -- the one accessor of the light field (ray-engine-v2 P6b).

Design v3 §4.3 ("the one accessor"), §7.1 and §8.5;
docs/ray_engine_v2_p6b_lit_world_brief_2026-09-25.md §1.1.

The light field is produced by the radiation sweep's light channels (P6a) --
three int64 planes on the GameMap in L°'s currency (``light_q``,
``light_flux_q``, ``light_glow``), written by the engine whenever the game
REQUESTS light (``Simulation.set_light``) or relights a paused frame
(``Simulation.relight``). NOTHING OUTSIDE THIS MODULE MAY KNOW HOW THE FIELD WAS
PRODUCED: the renderer, the rules (P7) and RL read light only through the two
functions below, never a sweep plane and never a producer. That is the seam
the CUDA-OpenGL interop (S8 residency) will later replace the host read behind
without anything outside this file changing (design §7.1).

  * :func:`read_light` -- for the RENDERER: dequantized float32 views (the
    dequantize convention: fresh copies at the render read, never written back)
    of the irradiance and the glow in LIGHT UNITS (1.0 = the blackbody ramp's
    intensity 1.0), plus the NORMALIZED net flux vector. Every view is a
    monotone function of the integer field per channel -- what looks dark to
    the player is dark to the rules (§8.5; tests/test_light_field.py).
  * :func:`light_at` -- for RULES and RL (P7 consumes it): one cell's integer
    RGB irradiance, exactly the engine's.

THE FLUX VECTOR points the way light TRAVELS (P6a: Σ_m I_m ŝ_m). The old render
march stored the opposite -- the direction the light comes FROM -- so a
consumer that shades by "toward the light" negates it (renderer/lighting.py
does, packing the same texture contract the 3D units read). Zero flux (a
uniform sky, a cell between two equal lamps, darkness) normalizes to (0, 0).

Not sim-pure yet: the cone emitters feeding ``light_q`` come partly from render
state (the cursor lamp, the selected unit's flashlight mode, the renderer's
effect queue) until P7 separates them (CLAUDE.md); ``light_q`` is not digested.
Render-layer float math only; ``np.sqrt`` is the one non-algebraic operation,
correctly rounded (door 3), and nothing here writes a synced field.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation import optics_fixed


@dataclass(frozen=True)
class LightView:
    """One read of the light field, in light units, float32, fresh copies.

    rgb       (h, w, 3) -- the irradiance per channel (dequantized ``light_q``)
    flux_dir  (h, w, 2) -- the net flux vector, NORMALIZED (x then y, +y the
                           increasing row), pointing the way light travels;
                           (0, 0) where the flux is zero
    glow      (h, w, 3) -- the in-scattered glow at gas cells (dequantized
                           ``light_glow``), 0 elsewhere
    """
    rgb: np.ndarray
    flux_dir: np.ndarray
    glow: np.ndarray

    @property
    def scalar(self) -> np.ndarray:
        """The legacy SCALAR light field -- the brightest channel per cell,
        (h, w) float32. What ``GameMap.light_map`` derives to (deleted at P6c)."""
        return self.rgb.max(axis=2)


def read_light(gmap) -> LightView:
    """THE RENDERER'S READ: the light field as float32 views in light units
    (see the module docstring). Monotone in ``light_q`` per channel."""
    fb = optics_fixed.light_fine_bits(gmap)
    rgb = optics_fixed.dequantize_light(gmap.light_q, fb).astype(np.float32)
    glow = optics_fixed.dequantize_light(gmap.light_glow, fb).astype(np.float32)
    flux = np.asarray(gmap.light_flux_q, dtype=np.float64)
    norm = np.sqrt(flux[..., 0] * flux[..., 0] + flux[..., 1] * flux[..., 1])
    safe = np.where(norm > 0.0, norm, 1.0)
    flux_dir = np.where(norm[..., None] > 0.0, flux / safe[..., None], 0.0)
    return LightView(rgb=rgb, flux_dir=flux_dir.astype(np.float32), glow=glow)


def light_at(gmap, y: int, x: int) -> tuple:
    """THE RULES' READ (P7): cell (y, x)'s integer RGB irradiance, in L°'s
    currency -- exactly ``light_q[y, x]``, no dequantize, no float."""
    q = gmap.light_q[int(y), int(x)]
    return int(q[0]), int(q[1]), int(q[2])


__all__ = ["LightView", "read_light", "light_at"]
