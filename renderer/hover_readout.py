"""Hover-tile "microscope" readout — pyray-free value packing (B2 P1, extended
fire-12 phase 2 — docs/fire_phase2_hud_and_level_2026-09-01.md part A).

**THE PER-TILE DEBUG PROBE SEAM.** :func:`pack_hover_readout` is the ONLY
place tile values get packed for display, full stop — *all* tile-value
display (the in-game F6 HUD, tools/lighting_demo.py, any future probe UI)
goes through this one function; callers never read gmap fields directly for
display. This matters beyond tidiness: when the resident tick's once-per-tick
D2H sync goes away (Erik, on record 2026-08-31) and tile inspection becomes a
device-side one-tile gather instead of a full-grid host mirror read, that
swap happens INSIDE this seam and no caller changes — every caller already
only sees the packed :class:`HoverReadout`, never the raw arrays.

Shows, for the tile under the cursor: temperature (game units AND
pseudo-Kelvin), fire intensity, material name, the five trace-gas densities
(steam, smoke, poison, teargas, fuel_gas), O2, atmosphere pressure, bulk N
(o2 + inert_n2), wind (m/s), water depth, wall_hp + the fuel fraction F, and
gas_energy. This module does the VALUE PACKING only — gmap reads -> display
values/strings — factored out (numpy + fixed-point + name tables, NO pyray) so
it is headless-unit-testable in isolation, exactly like B1's
:func:`renderer.blackbody.pack_emissive_rgba`. The pyray draw stays in the demo
/ ``game_renderer.draw_debug_hud``.

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

Every field is dequantized through ITS OWN ``simulation/*_fixed.py`` boundary
module (CLAUDE.md rule — never an inline ``/65536`` or a hardcoded scale):
pressure + wind via ``atmosphere_fixed`` (the atmosphere module's own doc says
wind shares its scale/helpers — one boundary, two consumers), bulk N via
``gas_fixed`` (same scale as the trace gases already read here), water via
``water_fixed``, wall_hp via ``wall_fixed``. ``gas_energy`` has no dedicated
``*_fixed`` module (design `docs/gas_energy_conservation_design_2026-08-29.md`
§2.2: it is the RAW product ``N_raw * T_abs_raw`` of two already-Q16.16
quantities, "Q32 raw; no >>16") — displayed here by dividing by
``gas_fixed.FP_ONE_F ** 2`` (the same named Q16.16 unit, reused twice rather
than a second hardcoded constant) into the design doc's own unit label,
"N·K" (§2.1: "energy == N·T, atm-equivalent x game-deg"; T_abs is Kelvin under
the G12 one-frame map, so N (real, dimensionless bulk-gas count) times K is
the natural readable unit — a documented DECISION, not a canonical scale, since
none exists for this derived field).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from simulation import atmosphere_fixed as _atmo_fx
from simulation import wall_fixed as _wall_fx
from simulation import water_fixed as _water_fx
from simulation.fire_fixed import FP_ONE_F as _FIRE_FP_ONE_F
from simulation.gas_fixed import FP_ONE_F as _GAS_FP_ONE_F
from simulation.gases import (FUEL_GAS, GAS_NAMES, INERT_N2, O2, POISON,
                              SMOKE, STEAM, TEARGAS)
from simulation.materials import MATERIAL_NAMES

# Q16.16 temperature scale — matches materials.TEMP_SCALE / renderer.blackbody
# .TEMP_SCALE (the shared temperature/heat fixed-point domain). ONE shared
# constant (cleanup #15): the SAME simulation.fire_fixed.FP_ONE_F already
# imported above for the fire field, so this module pulls in no new import
# and still no pyray-touching dependency.
TEMP_SCALE = _FIRE_FP_ONE_F

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
    pressure: float = 0.0         # dequantized atmosphere (bulk air pressure)
    bulk_n: float = 0.0           # dequantized o2 + inert_n2 (the Dalton N_total)
    wind_vx: float = 0.0          # dequantized wind_x, m/s (true velocity, #51)
    wind_vy: float = 0.0          # dequantized wind_y, m/s
    water_depth: float = 0.0      # dequantized standing-water depth, metres
    wall_hp: float = 0.0          # dequantized structural HP (the fuel source)
    fuel_frac: float = 0.0        # F = clamp01(wall_hp / this tile's full hp)
    gas_energy: float = 0.0       # gas_energy raw / FP_ONE_F**2, unit "N.K"
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
    mat_id = int(gmap.material[ty, tx])
    t_game = float(gmap.temperature[ty, tx]) / TEMP_SCALE
    # Kelvin display, ONE frame (G12, issue #12,
    # docs/fire_g12_one_map_patch_2026-08-31.md): the canonical map
    # (kelvin_fn: K = 293 + T_game) is now the SAME frame the EOS
    # thermodynamics already uses, and it is valid all the way to the T_MIN
    # floor (T_game = -292 -> K = 1, never negative) — so there is no longer
    # a sub-ambient regime to special-case. This dissolves the dual-frame
    # "K_eos" patch that used to paper over the old ×3 map going unphysical
    # below T_game ~ -97.67 (the T_abs arc's "-574 K" symptom, design
    # D-7/C12 + HUMAN-TEST 2026-08-21) — that was a symptom of the ×3 map,
    # not a real sub-ambient case.
    kelvin = float(kelvin_fn(t_game))
    kelvin_label = "K"
    fire = float(gmap.fire[ty, tx]) / _FIRE_FP_ONE_F
    gases = {
        GAS_NAMES[g]: float(gmap.gas[g][ty, tx]) / _GAS_FP_ONE_F
        for g in _READOUT_GAS_IDS
    }

    # Atmosphere pressure + wind — dequantized through atmosphere_fixed, the
    # shared boundary the module's own docstring names for both fields (one
    # helper, two consumers: "no separate wind_fixed module").
    pressure = _atmo_fx.dequantize(gmap.atmosphere[ty, tx]).item()
    wind_vx = _atmo_fx.dequantize(gmap.wind_x[ty, tx]).item()
    wind_vy = _atmo_fx.dequantize(gmap.wind_y[ty, tx]).item()

    # Bulk N — the Dalton N_total the EOS derives pressure from: o2 +
    # inert_n2, the two CONSERVATIVE bulk gas planes (gamemap._gas_bulk_n_raw
    # sums the same pair; not re-derived here, just the two named ids read
    # through the same gas_fixed scale already used for the trace gases
    # above).
    bulk_n = (float(gmap.gas[O2][ty, tx]) + float(gmap.gas[INERT_N2][ty, tx])) \
        / _GAS_FP_ONE_F

    # Water depth, metres — its own boundary module.
    water_depth = _water_fx.dequantize(gmap.water_depth[ty, tx]).item()

    # Fuel: wall_hp (the fire's fuel source, its own boundary module) AND the
    # fire logistic's fuel-availability fraction F = clamp01(wall_hp /
    # this-tile's-full-hp) (materials.py's fuel_recip docstring; G3: fuel is
    # what kills a fire). hp_mat comes from the SAME MaterialTable the sim
    # quantized wall_hp's initial value from (gmap.materials.hp), so numerator
    # and denominator can never disagree.
    wall_hp = _wall_fx.dequantize(gmap.wall_hp[ty, tx]).item()
    hp_mat = float(gmap.materials.hp[mat_id])
    fuel_frac = min(1.0, max(0.0, wall_hp / hp_mat)) if hp_mat > 0.0 else 0.0

    # gas_energy — raw N_raw * T_abs_raw (Q32 raw, no dedicated *_fixed
    # module; see the module docstring for the unit choice). Divide by the
    # SAME named Q16.16 unit twice rather than a second hardcoded 65536.
    gas_energy = float(gmap.gas_energy[ty, tx]) / (_GAS_FP_ONE_F ** 2)

    lines = [
        f"tile ({tx}, {ty})  {material}",
        f"T: {t_game:8.1f} u   ({kelvin:6.0f} {kelvin_label})",
        f"fire: {fire:5.3f}    O2: {gases['o2']:5.3f}",
        f"steam {gases['steam']:5.3f}   smoke {gases['smoke']:5.3f}",
        f"poison {gases['poison']:5.3f}  teargas {gases['teargas']:5.3f}",
        f"fuel_gas {gases['fuel_gas']:5.3f}",
        f"P: {pressure:8.3f}    N: {bulk_n:6.3f}",
        f"wind: {wind_vx:6.2f}, {wind_vy:6.2f} m/s",
        f"water: {water_depth:6.3f} m",
        f"wall_hp: {wall_hp:7.2f}  F: {fuel_frac:5.3f}",
        f"gas_energy: {gas_energy:10.3f} N.K",
    ]
    return HoverReadout(tx=int(tx), ty=int(ty), material=material,
                        t_game=t_game, kelvin=kelvin, fire=fire, gases=gases,
                        pressure=pressure, bulk_n=bulk_n,
                        wind_vx=wind_vx, wind_vy=wind_vy,
                        water_depth=water_depth, wall_hp=wall_hp,
                        fuel_frac=fuel_frac, gas_energy=gas_energy,
                        lines=lines)


__all__ = ["HoverReadout", "pack_hover_readout", "TEMP_SCALE"]
