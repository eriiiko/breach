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
import math
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
        # Ray-engine-v2 P1: the shadow sweep's planes the readout reads.
        rad_fluence=np.zeros((H, W), dtype=np.int64),
        heat_atten_q=np.zeros((H, W), dtype=np.int32),
        dyn_heat_atten_q=np.zeros((H, W), dtype=np.int32),
        heat_inv_shift=np.zeros((H, W), dtype=np.int32),
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

    # Ray-engine-v2 P1: Phi / a / d dequantized through the temperature and
    # optics scales; f and E_inv(Phi) are nan on a stub with no engine bound.
    g.rad_fluence[ty, tx] = int(round(12.5 * TEMP_SCALE))
    g.heat_atten_q[ty, tx] = 65536
    g.dyn_heat_atten_q[ty, tx] = 65536
    r = pack_hover_readout(g, tx, ty, KELVIN_FN)
    assert r.phi == pytest.approx(12.5, abs=1e-6)
    assert r.atten_a == 1.0 and r.atten_d == 1.0
    assert np.isnan(r.fleck_f) and np.isnan(r.t_cap)

    # Panel-ready lines carry the tile + a couple of the numbers.
    assert r.lines[0] == "tile (2, 1)  wood"
    assert len(r.lines) == 13
    assert r.lines[11].startswith("Phi:") and r.lines[12].startswith("f:")


def test_sweep_rows_come_from_the_engine_when_one_is_bound():
    """PROPERTY: with a PhysicsEngine bound (as Simulation binds it), the f row
    is the engine's own Fleck factor — exactly 1.0 at ambient, below 1.0 on a
    hot opaque wood tile — and E_inv(Phi) inverts the engine's E° table (Phi =
    E°[b] -> 4b game), so the readout and the sweep share ONE implementation.

    BREAKS IF: the readout re-derives f or E_inv locally (a third copy that
    can drift), or reads them off the wrong engine member.
    """
    sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
    import breach_physics as bp
    eng = bp.PhysicsEngine()
    # T6 (issue #12): this used to read [physics.fire] rad_scale (the old
    # cast's fitted key, deleted with it). [physics.radiation]
    # rad_scale_derived is the key that actually feeds PhysicsEngine.emissive
    # in production, but it does NOT serve this test's purpose: at the
    # DERIVED (physically real) scale the Fleck damping this test exercises
    # NEVER engages for any shipped row (config.toml's own documented
    # consequence -- a physically heavy tile cannot radiate an appreciable
    # share of its own heat in 1/24 s), so `r_hot.fleck_f` would sit at 1.0
    # and the `0.0 < fleck_f < 1.0` assertion below would fail non-vacuously
    # -- as it did in the first cut of this fix. The property under test is
    # "the readout matches whatever the ENGINE independently computes", not
    # "matches production's specific scale", so a big-enough LITERAL scale
    # that reliably exercises the damping branch is the right fixture value
    # here, same as tests/test_emissive_table.py's own R.RAD_SCALE constant
    # (also no longer tied to any config key). Numerically the old fitted
    # 5.1427e-5, kept as a bare constant so this test does not silently go
    # vacuous again if a future patch changes either config key's value.
    RAD_SCALE_FOR_DAMPING_TEST = 5.1427e-5
    eng.emissive.rad_scale = RAD_SCALE_FOR_DAMPING_TEST
    eng.emissive.kelvin_ambient = float(_TS.kelvin_ambient)
    eng.emissive.k_temp_to_kelvin = float(_TS.k_temp_to_kelvin)
    eng.emissive.bake()
    table = np.asarray(eng.emissive.table(), dtype=np.int64)
    g = _stub_gmap()
    g._physics_engine = eng
    g._gas_energy_t_amb_raw = lambda: int(round(_TS.kelvin_ambient * TEMP_SCALE))
    tx, ty = 1, 2
    g.material[ty, tx] = MAT_WOOD
    g.heat_atten_q[ty, tx] = 65536
    g.dyn_heat_atten_q[ty, tx] = 65536
    g.heat_inv_shift[ty, tx] = 3
    r_amb = pack_hover_readout(g, tx, ty, KELVIN_FN)
    assert r_amb.fleck_f == 1.0
    assert r_amb.t_cap == 0.0                      # Phi = 0 < E°[0] -> E_inv = 0
    g.temperature[ty, tx] = int(round(1263.0 * TEMP_SCALE))
    b = 700
    g.rad_fluence[ty, tx] = int(table[b])
    r_hot = pack_hover_readout(g, tx, ty, KELVIN_FN)
    assert 0.0 < r_hot.fleck_f < 1.0
    assert r_hot.t_cap == pytest.approx(4 * b, abs=1e-9)
    assert r_hot.phi == pytest.approx(table[b] / TEMP_SCALE, rel=1e-12)


def test_sweep_rows_are_still_readable_after_a_whole_simulation_step():
    """PROPERTY (ray-engine-v2 P2a, design row 38 — the WHOLE POINT of moving
    the wipe): after a complete ``Simulation.step`` on a level with a burning,
    1263-game wood tile, the readout at a NEIGHBOURING cell still reports a
    positive Phi and a finite E_inv(Phi). While the four shadow planes were
    wiped by the conductor at end of tick, a render-time read saw Phi = 0 and
    E_inv(0) = 0 on every tile and the three sweep rows were decoration.

    This is the end-user-visible half of the patch, so it is asserted on the
    REAL engine through the real conductor, not on a stub.

    BREAKS IF: a ``fill(0)`` for the sweep's planes returns to Simulation.step,
    or ``RadiationSweep::run`` stops zeroing and writing them itself.
    """
    sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
    import breach_physics as bp
    from level_loader import load as load_level
    from simulation import Simulation, fire_fixed
    from simulation.materials import MAT_WOOD

    sim = Simulation(load_level("playground"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    ys, xs = np.where((g.material == MAT_WOOD) & g.thermal_solid)
    pick = None
    for y, x in zip(ys, xs):
        if 0 < y < g.material.shape[0] - 1 and 0 < x < g.material.shape[1] - 1:
            if not g.thermal_solid[y, x + 1]:
                pick = (int(y), int(x))
                break
    assert pick is not None, "the level carries no wood tile with an air neighbour"
    y, x = pick
    # A fire is started by DELIVERING HEAT (CLAUDE.md "Starting a fire"): the
    # tile is both hot and lit, or it would radiate nothing at all.
    g.temperature[y, x] = 1263 << 16
    g.fire[y, x] = fire_fixed.quantize_scalar(0.8)
    sim.set_paused(False)
    sim.step()                                   # a WHOLE tick, wipe included

    r = pack_hover_readout(g, x + 1, y, KELVIN_FN)   # the air tile beside the fire
    assert r is not None
    assert r.phi > 0.0, "Phi is zero after the tick — the readout is blind again"
    assert math.isfinite(r.t_cap) and r.t_cap >= 0.0
    assert math.isfinite(r.fleck_f) and 0.0 < r.fleck_f <= 1.0
    # non-vacuous: the neighbour of a 1263-game fire sees MORE than the bare
    # ambient ring every cell gets, so this is the fire's own radiation.
    e0 = int(np.asarray(sim.physics_runner.engine.emissive.table())[0])
    ambient_phi = 16 * ((e0 * 4096) >> 16) / TEMP_SCALE
    assert r.phi > ambient_phi, (r.phi, ambient_phi)
    print(f"\nafter one Simulation.step: neighbour of the fire reads Phi = {r.phi:.1f} u/t "
          f"(ambient ring alone would be {ambient_phi:.1f}), E_inv(Phi) = {r.t_cap:.1f} u, "
          f"f = {r.fleck_f:.6f}")


def test_a_gas_cell_shows_the_extinction_and_f_the_sweep_actually_uses():
    """PROPERTY (P5c): on a GAS cell the inspector shows what the sweep READS --
    `a` is the smoke term's extinction as a MAX with the material's (air's is
    0), `d` the stamped total as a MAX with it, and `f` the pre-pass's GAS arm
    priced in the fold's own currency -- not the material's a = 0 and the solid
    arm's f = 1 it showed before. Checked against the integer reference's own
    law (gas_extinction_q, fleck_f_gas_q) on a live engine with the SHIPPED
    smoke coefficient, on a hot smoky cell; a smoke-free gas cell still reads a
    = 0 and f = 1; a thermal solid keeps its material's a and the solid arm.

    BREAKS IF: the readout shows the material planes on a gas cell, forms the
    smoke term or f with a second copy of the law, or prices the gas arm in a
    currency other than PhysicsEngine.gas_capacity_q().
    """
    sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))
    sys.path.insert(0, str(ROOT / "tests"))
    import breach_physics as bp
    from _radiation_sweep_harness import R
    from level_loader import load as load_level
    from simulation import Simulation, gas_fixed
    sim = Simulation(load_level("playground"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    air = g._gas_energy_accountable()
    ys, xs = np.nonzero(air)
    y, x = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    y2, x2 = int(ys[len(ys) // 3]), int(xs[len(xs) // 3])
    g.gas[SMOKE][y, x] = gas_fixed.quantize_scalar(0.02)
    sel = np.zeros_like(air)
    sel[y, x] = True
    g.seed_gas_temperature(sel, 3000 << 16)                  # hot: the arm damps it
    hq = [int(v) for v in g.gases.heat_absorb_q16]
    assert hq[SMOKE] > 0, "the shipped smoke absorbs nothing"
    n_bulk = int(g.gas[O2][y, x]) + int(g.gas[INERT_N2][y, x])
    a_gas_q = R.gas_extinction_q([int(v) for v in g.gas[:, y, x]], hq, n_bulk)
    assert 0 < a_gas_q < 65536
    r = pack_hover_readout(g, x, y, KELVIN_FN)
    assert r.atten_a == pytest.approx(a_gas_q / 65536.0, abs=1e-12)
    assert r.a_gas == r.atten_a and r.atten_d >= r.atten_a
    f_want = R.fleck_f_gas_q(3000 << 16, a_gas_q, n_bulk, table=R.E_LIVE)[0]
    assert r.fleck_f == f_want / float(1 << 24) and r.fleck_f < 1.0
    assert "smoke" in r.lines[11]
    r0 = pack_hover_readout(g, x2, y2, KELVIN_FN)             # smoke-free air
    assert r0.atten_a == 0.0 and r0.fleck_f == 1.0 and r0.a_gas == 0.0
    ts_y, ts_x = (int(v[0]) for v in np.nonzero(g.thermal_solid))
    rs = pack_hover_readout(g, ts_x, ts_y, KELVIN_FN)
    assert rs.atten_a == pytest.approx(int(g.heat_atten_q[ts_y, ts_x]) / 65536.0)


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
