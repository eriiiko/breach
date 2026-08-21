"""Hover-tile "microscope" readout — pyray-free value packing (B2 P1).

The tuning harness (tools/lighting_demo.py) shows, for the tile under the
cursor: temperature (game units AND pseudo-Kelvin), fire intensity, material
name, the five trace-gas densities (steam, smoke, poison, teargas, fuel_gas),
and O2. This module does the VALUE PACKING only — gmap reads -> display
values/strings — factored out (numpy + fixed-point + name tables, NO pyray) so
it is headless-unit-testable in isolation, exactly like B1's
:func:`renderer.blackbody.pack_emissive_rgba`. The pyray draw stays in the demo.

READ-ONLY: every field is read from ``gmap``; nothing is written (renderer +
tools contract — the RENDERER never writes sim fields, and this helper writes
nothing at all). Renderer-side module importing ``simulation`` is fine
(renderer -> simulation is the allowed direction).

The T -> pseudo-Kelvin conversion is REUSED, not reinvented: the caller passes
``kelvin_fn`` = the black-body ramp's own ``_kelvin_from_tgame``
(``kelvin = kelvin_ambient + k_temp_to_kelvin * T_game``, config
[physics.temperature_scale]), so the readout and the emissive overlay agree by
construction. Passing a callable keeps this module ramp-agnostic (and trivially
testable with a plain lambda).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from simulation.fire_fixed import FP_ONE_F as _FIRE_FP_ONE_F
from simulation.gas_fixed import FP_ONE_F as _GAS_FP_ONE_F
from simulation.gases import (FUEL_GAS, GAS_NAMES, O2, POISON, SMOKE, STEAM,
                              TEARGAS)
from simulation.materials import MATERIAL_NAMES

# Q16.16 temperature scale — MUST match materials.TEMP_SCALE /
# renderer.blackbody.TEMP_SCALE (the shared temperature/heat fixed-point
# domain). Defined locally so this module pulls in no pyray-touching import.
TEMP_SCALE = 65536.0

# The gases shown, in readout order: the five trace species then O2 (the
# O2-starvation story). inert_n2 is invisible bulk air — omitted deliberately.
_READOUT_GAS_IDS: Tuple[int, ...] = (STEAM, SMOKE, POISON, TEARGAS, FUEL_GAS, O2)


@dataclass
class HoverReadout:
    """Packed, display-ready readout for one tile (all values dequantized)."""
    tx: int
    ty: int
    material: str
    t_game: float                 # dequantized game-ΔT (temperature units)
    kelvin: float                 # pseudo-Kelvin via the reused ramp conversion
    fire: float                   # [0,1] fire intensity
    gases: dict                   # gas-name -> dequantized [0,1] density
    lines: List[str] = field(default_factory=list)   # panel-ready text rows


def _material_name(gmap, ty: int, tx: int) -> str:
    """Tile material label: vacuum (SPACE) overrides; else the material id."""
    if bool(gmap.is_vacuum[ty, tx]):
        return "vacuum"
    mid = int(gmap.material[ty, tx])
    return MATERIAL_NAMES.get(mid, f"mat{mid}")


def pack_hover_readout(gmap, tx: int, ty: int,
                       kelvin_fn: Callable[[float], float]
                       ) -> Optional[HoverReadout]:
    """Pack the tile under (tx, ty) into a :class:`HoverReadout`, or None.

    ``tx`` / ``ty`` are physics-tile column/row (the ``mouse_to_tile`` order).
    Returns None when the tile is outside the grid — the caller shows a
    "outside map" line. ``kelvin_fn`` maps dequantized game-ΔT to pseudo-Kelvin
    (pass the black-body ramp's ``_kelvin_from_tgame``).
    """
    h, w = gmap.material.shape
    if not (0 <= tx < w and 0 <= ty < h):
        return None

    material = _material_name(gmap, ty, tx)
    t_game = float(gmap.temperature[ty, tx]) / TEMP_SCALE
    # Kelvin display, two frames (T_abs arc, design D-7/C12 + HUMAN-TEST
    # 2026-08-21: Erik read "-574 K" off a 1.1 K cell and rightly asked if
    # the T_MIN floor had failed). The canonical render map
    # (kelvin_fn: K = 293 + 3*T_game) matches the visible glow and stays
    # the display for T >= ambient — but it is INVALID below ambient (goes
    # negative from T_game < -97.67). Sub-ambient gas therefore displays
    # the EOS's own absolute frame (K = T_game + 290, [physics.eos]
    # eos_t_amb_k — always positive, floor exactly 1 K), labeled "K_eos"
    # so the two frames cannot be confused.
    if t_game >= 0.0:
        kelvin = float(kelvin_fn(t_game))
        kelvin_label = "K"
    else:
        kelvin = 290.0 + t_game
        kelvin_label = "K_eos"
    fire = float(gmap.fire[ty, tx]) / _FIRE_FP_ONE_F
    gases = {
        GAS_NAMES[g]: float(gmap.gas[g][ty, tx]) / _GAS_FP_ONE_F
        for g in _READOUT_GAS_IDS
    }

    lines = [
        f"tile ({tx}, {ty})  {material}",
        f"T: {t_game:8.1f} u   ({kelvin:6.0f} {kelvin_label})",
        f"fire: {fire:5.3f}    O2: {gases['o2']:5.3f}",
        f"steam {gases['steam']:5.3f}   smoke {gases['smoke']:5.3f}",
        f"poison {gases['poison']:5.3f}  teargas {gases['teargas']:5.3f}",
        f"fuel_gas {gases['fuel_gas']:5.3f}",
    ]
    return HoverReadout(tx=int(tx), ty=int(ty), material=material,
                        t_game=t_game, kelvin=kelvin, fire=fire, gases=gases,
                        lines=lines)


__all__ = ["HoverReadout", "pack_hover_readout", "TEMP_SCALE"]
