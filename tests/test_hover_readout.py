"""Headless unit tests for renderer/hover_readout.py (B2 P1).

The hover-tile "microscope" VALUE PACKING (gmap reads -> display values) is
pyray-free by construction, so it is loaded in ISOLATION (importlib from file,
the B1 pack_emissive_rgba pattern) — no GL, no renderer/__init__. The gmap is a
tiny stub of numpy fields; we assert the dequantized values + the material-name
mapping the panel will draw.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_hover_readout.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

_RENDERER = ROOT / "renderer"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_isolated_{name}", _RENDERER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass field-type introspection (which does
    # sys.modules.get(cls.__module__)) resolves under the isolated load.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


hover_readout = _load("hover_readout")
pack_hover_readout = hover_readout.pack_hover_readout
TEMP_SCALE = hover_readout.TEMP_SCALE

from simulation.gases import (FUEL_GAS, INERT_N2, N_GASES, O2, POISON,  # noqa: E402
                              SMOKE, STEAM, TEARGAS)
from simulation.materials import MAT_AIR, MAT_WOOD, MATERIAL_NAMES  # noqa: E402
from simulation.fire_fixed import FP_ONE_F as FIRE_FP_ONE_F  # noqa: E402
from simulation.gas_fixed import FP_ONE_F as GAS_FP_ONE_F  # noqa: E402
from simulation.atmosphere_fixed import FP_ONE_F as ATMO_FP_ONE_F  # noqa: E402
from simulation.water_fixed import FP_ONE_F as WATER_FP_ONE_F  # noqa: E402
from simulation.wall_fixed import FP_ONE_F as WALL_FP_ONE_F  # noqa: E402
from temperature_scale import load as _load_temperature_scale  # noqa: E402

H, W = 4, 5
# The config-live T->Kelvin conversion (kelvin_ambient + slope * T_game),
# re-derived from the SAME accessor the game uses (src/temperature_scale.py,
# [physics.temperature_scale]) rather than a hardcoded lambda — the demo
# passes the real ramp's _kelvin_from_tgame, which reads the same section.
_TS = _load_temperature_scale()
KELVIN_FN = _TS.to_kelvin


# Stub per-material hp table (materials.py's `table.hp[material_id]`, the
# fuel-fraction denominator) — real config values for the two ids the tests
# touch (air 0, wood 60); the rest are unused padding.
_N_MATERIALS = max(MATERIAL_NAMES) + 1
_STUB_HP = np.zeros(_N_MATERIALS, dtype=np.float32)
_STUB_HP[MAT_AIR] = 0.0                                  # air: massless (hp 0)
_STUB_HP[MAT_WOOD] = 60.0


def _stub_gmap():
    """A minimal gmap: the fields hover_readout reads, all zero to start."""
    return SimpleNamespace(
        material=np.zeros((H, W), dtype=np.int32),
        is_vacuum=np.zeros((H, W), dtype=bool),
        temperature=np.zeros((H, W), dtype=np.int32),
        fire=np.zeros((H, W), dtype=np.int32),
        gas=np.zeros((N_GASES, H, W), dtype=np.int32),
        atmosphere=np.zeros((H, W), dtype=np.int32),
        wind_x=np.zeros((H, W), dtype=np.int32),
        wind_y=np.zeros((H, W), dtype=np.int32),
        water_depth=np.zeros((H, W), dtype=np.int32),
        wall_hp=np.zeros((H, W), dtype=np.int32),
        gas_energy=np.zeros((H, W), dtype=np.int64),
        materials=SimpleNamespace(hp=_STUB_HP),
    )


def test_out_of_bounds_returns_none():
    g = _stub_gmap()
    assert pack_hover_readout(g, -1, 0, KELVIN_FN) is None
    assert pack_hover_readout(g, 0, H, KELVIN_FN) is None
    assert pack_hover_readout(g, W, 0, KELVIN_FN) is None


def test_packs_all_fields_dequantized():
    g = _stub_gmap()
    tx, ty = 2, 1
    g.material[ty, tx] = MAT_WOOD
    g.temperature[ty, tx] = int(round(300.0 * TEMP_SCALE))
    g.fire[ty, tx] = int(round(0.5 * FIRE_FP_ONE_F))
    g.gas[STEAM, ty, tx] = int(round(0.25 * GAS_FP_ONE_F))
    g.gas[SMOKE, ty, tx] = int(round(0.50 * GAS_FP_ONE_F))
    g.gas[POISON, ty, tx] = int(round(0.10 * GAS_FP_ONE_F))
    g.gas[TEARGAS, ty, tx] = int(round(0.05 * GAS_FP_ONE_F))
    g.gas[FUEL_GAS, ty, tx] = int(round(0.15 * GAS_FP_ONE_F))
    g.gas[O2, ty, tx] = int(round(0.21 * GAS_FP_ONE_F))
    g.gas[INERT_N2, ty, tx] = int(round(0.70 * GAS_FP_ONE_F))
    g.atmosphere[ty, tx] = int(round(1.05 * ATMO_FP_ONE_F))
    g.wind_x[ty, tx] = int(round(2.5 * ATMO_FP_ONE_F))
    g.wind_y[ty, tx] = int(round(-1.25 * ATMO_FP_ONE_F))
    g.water_depth[ty, tx] = int(round(0.4 * WATER_FP_ONE_F))
    g.wall_hp[ty, tx] = int(round(30.0 * WALL_FP_ONE_F))    # wood, half hp
    g.gas_energy[ty, tx] = int(123456789)

    r = pack_hover_readout(g, tx, ty, KELVIN_FN)
    assert r is not None
    assert (r.tx, r.ty) == (tx, ty)
    assert r.material == "wood"
    assert r.t_game == pytest.approx(300.0, abs=1e-3)
    assert r.kelvin == pytest.approx(_TS.to_kelvin(300.0), abs=1e-2)  # G12: 293 + 1*300
    assert r.fire == pytest.approx(0.5, abs=1e-4)
    assert r.gases["steam"] == pytest.approx(0.25, abs=1e-4)
    assert r.gases["smoke"] == pytest.approx(0.50, abs=1e-4)
    assert r.gases["poison"] == pytest.approx(0.10, abs=1e-4)
    assert r.gases["teargas"] == pytest.approx(0.05, abs=1e-4)
    assert r.gases["fuel_gas"] == pytest.approx(0.15, abs=1e-4)
    assert r.gases["o2"] == pytest.approx(0.21, abs=1e-4)
    # inert_n2 is invisible bulk air — deliberately NOT in the readout.
    assert "inert_n2" not in r.gases

    # Phase-2 fields: pressure/wind/bulk-N via atmosphere_fixed/gas_fixed,
    # water via water_fixed, wall_hp/fuel-fraction via wall_fixed + the
    # material's own hp (wood 60 -> half-hp wall_hp 30 -> F == 0.5), and
    # gas_energy as raw/FP_ONE_F**2.
    assert r.pressure == pytest.approx(1.05, abs=1e-4)
    assert r.bulk_n == pytest.approx(0.21 + 0.70, abs=1e-4)
    assert r.wind_vx == pytest.approx(2.5, abs=1e-4)
    assert r.wind_vy == pytest.approx(-1.25, abs=1e-4)
    assert r.water_depth == pytest.approx(0.4, abs=1e-4)
    assert r.wall_hp == pytest.approx(30.0, abs=1e-3)
    assert r.fuel_frac == pytest.approx(0.5, abs=1e-4)
    assert r.gas_energy == pytest.approx(123456789 / (GAS_FP_ONE_F ** 2), rel=1e-9)

    # Panel-ready lines carry the tile + a couple of the numbers.
    assert r.lines[0] == "tile (2, 1)  wood"
    assert len(r.lines) == 11


def test_vacuum_tile_labelled_vacuum():
    g = _stub_gmap()
    g.is_vacuum[0, 0] = True
    r = pack_hover_readout(g, 0, 0, KELVIN_FN)
    assert r is not None and r.material == "vacuum"


def test_cold_empty_tile_reads_zero():
    g = _stub_gmap()
    r = pack_hover_readout(g, 1, 1, KELVIN_FN)
    assert r.t_game == 0.0 and r.fire == 0.0
    assert r.kelvin == pytest.approx(_TS.kelvin_ambient)  # ambient at T_game 0
    assert all(v == 0.0 for v in r.gases.values())
    assert r.material == "air"                  # MAT_AIR == 0
    # Phase-2 fields all read zero on a cold, empty, air tile; air's hp is 0
    # so fuel_frac takes the guarded "no substance here" zero, not a div/0.
    assert r.pressure == 0.0 and r.bulk_n == 0.0
    assert r.wind_vx == 0.0 and r.wind_vy == 0.0
    assert r.water_depth == 0.0
    assert r.wall_hp == 0.0 and r.fuel_frac == 0.0
    assert r.gas_energy == 0.0


if __name__ == "__main__":
    test_out_of_bounds_returns_none()
    test_packs_all_fields_dequantized()
    test_vacuum_tile_labelled_vacuum()
    test_cold_empty_tile_reads_zero()
    print("OK — hover_readout packs T/Kelvin/fire/material/gases/O2/pressure/"
          "wind/water/fuel/gas_energy headless")
