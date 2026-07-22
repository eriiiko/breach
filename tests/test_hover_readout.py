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

from simulation.gases import (FUEL_GAS, N_GASES, O2, POISON, SMOKE,  # noqa: E402
                              STEAM, TEARGAS)
from simulation.materials import MAT_WOOD  # noqa: E402
from simulation.fire_fixed import FP_ONE_F as FIRE_FP_ONE_F  # noqa: E402
from simulation.gas_fixed import FP_ONE_F as GAS_FP_ONE_F  # noqa: E402

H, W = 4, 5
# The config-default T->Kelvin conversion (kelvin_ambient + slope * T_game);
# the demo passes the real ramp's _kelvin_from_tgame — here a plain lambda.
KELVIN_FN = lambda t: 293.0 + 2.0 * t   # noqa: E731


def _stub_gmap():
    """A minimal gmap: the fields hover_readout reads, all zero to start."""
    return SimpleNamespace(
        material=np.zeros((H, W), dtype=np.int32),
        is_vacuum=np.zeros((H, W), dtype=bool),
        temperature=np.zeros((H, W), dtype=np.int32),
        fire=np.zeros((H, W), dtype=np.int32),
        gas=np.zeros((N_GASES, H, W), dtype=np.int32),
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

    r = pack_hover_readout(g, tx, ty, KELVIN_FN)
    assert r is not None
    assert (r.tx, r.ty) == (tx, ty)
    assert r.material == "wood"
    assert r.t_game == pytest.approx(300.0, abs=1e-3)
    assert r.kelvin == pytest.approx(893.0, abs=1e-2)   # 293 + 2*300
    assert r.fire == pytest.approx(0.5, abs=1e-4)
    assert r.gases["steam"] == pytest.approx(0.25, abs=1e-4)
    assert r.gases["smoke"] == pytest.approx(0.50, abs=1e-4)
    assert r.gases["poison"] == pytest.approx(0.10, abs=1e-4)
    assert r.gases["teargas"] == pytest.approx(0.05, abs=1e-4)
    assert r.gases["fuel_gas"] == pytest.approx(0.15, abs=1e-4)
    assert r.gases["o2"] == pytest.approx(0.21, abs=1e-4)
    # inert_n2 is invisible bulk air — deliberately NOT in the readout.
    assert "inert_n2" not in r.gases
    # Panel-ready lines carry the tile + a couple of the numbers.
    assert r.lines[0] == "tile (2, 1)  wood"
    assert len(r.lines) == 6


def test_vacuum_tile_labelled_vacuum():
    g = _stub_gmap()
    g.is_vacuum[0, 0] = True
    r = pack_hover_readout(g, 0, 0, KELVIN_FN)
    assert r is not None and r.material == "vacuum"


def test_cold_empty_tile_reads_zero():
    g = _stub_gmap()
    r = pack_hover_readout(g, 1, 1, KELVIN_FN)
    assert r.t_game == 0.0 and r.fire == 0.0
    assert r.kelvin == pytest.approx(293.0)     # ambient at T_game 0
    assert all(v == 0.0 for v in r.gases.values())
    assert r.material == "air"                  # MAT_AIR == 0


if __name__ == "__main__":
    test_out_of_bounds_returns_none()
    test_packs_all_fields_dequantized()
    test_vacuum_tile_labelled_vacuum()
    test_cold_empty_tile_reads_zero()
    print("OK — hover_readout packs T/Kelvin/fire/material/gases/O2 headless")
