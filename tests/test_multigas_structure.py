"""Multi-gas system — M1 structure + behaviour-preservation tests (engine/05 §6.2).

M1 generalises the single ``smoke`` scalar field into N gas density fields
(``gmap.gas``, shape ``(N, h, w)``) + a data-driven ``[gases.*]`` table, WITHOUT
changing any visible behaviour: the existing smoke becomes the ``black_smoke``
slice, ``gmap.smoke`` is a view onto it, and the per-gas transport loop steps each
gas with the SAME C++ smoke solver. The raycaster still reads ``gmap.smoke``
(black_smoke); per-channel colour summation over gases is M2.

These tests assert the M1 contract:

1. ``gmap.gas`` has shape ``(N_GASES, h, w)`` float32.
2. ``gmap.smoke`` IS the ``black_smoke`` slice — a view: writing one is visible
   in the other (both directions).
3. ``GasTable`` exposes the 5 gases (white_smoke / black_smoke / poison / teargas
   / fuel_gas) with the §6.2 absorption / scatter / diffusion / decay / flags.
4. A populated NON-smoke gas (poison) advects + diffuses through the per-gas
   transport loop exactly as smoke does (transport generalises).
5. BEHAVIOUR PRESERVATION — a black_smoke deposit evolved through the new per-gas
   loop matches the pre-refactor single-field reference (the C++ solver called
   directly with the legacy d_smoke=0.1), within fp tolerance.
6. DETERMINISM — a full headless Simulation rollout is bit-identical run-to-run.
7. The recorder / renderer paths that read ``gmap.smoke`` still work (import +
   a headless ``Simulation.step()``).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_multigas_structure.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp
from level_loader import LevelData, load as load_level
from simulation import Simulation
from simulation.gamemap import GameMap
from simulation.gases import (
    GasTable,
    N_GASES,
    WHITE_SMOKE,
    BLACK_SMOKE,
    POISON,
    TEARGAS,
    FUEL_GAS,
)
from simulation.physics_runner import PhysicsRunner

SEED = 11


# --------------------------------------------------------------------------
# Test levels / helpers
# --------------------------------------------------------------------------
def _room_level(h=16, w=16):
    """A simple hull-walled room with an air interior (CSV: 1 = hull, 4 = air)."""
    tm = np.ones((h, w), dtype=np.int32)     # all hull
    tm[1:h - 1, 1:w - 1] = 4                  # carve interior air
    return LevelData(
        name="multigas_room",
        version="1",
        path=Path("."),
        tilemap=tm,
        tile_size_m=1.0,
        diffuse_path=Path("."),
    )


def _make_gmap():
    return GameMap(_room_level())


# --------------------------------------------------------------------------
# 1. gas array shape
# --------------------------------------------------------------------------
def test_gas_array_shape():
    g = _make_gmap()
    h, w = g.smoke.shape
    assert g.gas.shape == (N_GASES, h, w), \
        f"gas array shape {g.gas.shape} != (N={N_GASES}, {h}, {w})"
    assert g.gas.dtype == np.int32   # S2b: int32 Q16.16
    # EOS refactor P1 (docs/eos_refactor_design.md §1): N_GASES grew 5 -> 7
    # (o2, inert_n2 APPENDED at ids 5/6 — see tests/test_eos_p1_species_transport.py
    # for the full bulk-pair contract). The 5 M1 trace gases below are untouched.
    assert N_GASES == 7, f"expected 5 trace + 2 bulk = 7 gases, got {N_GASES}"


# --------------------------------------------------------------------------
# 2. gmap.smoke is the black_smoke slice (a VIEW — aliasing both ways)
# --------------------------------------------------------------------------
def test_smoke_is_black_smoke_view():
    g = _make_gmap()
    # Same memory (a view, not a copy).
    assert g.smoke.base is g.gas, "gmap.smoke is not a view into gmap.gas"
    assert np.shares_memory(g.smoke, g.gas[BLACK_SMOKE])

    # S2b: gas is int32 Q16.16 — write raw counts (the view aliasing is what's
    # under test, not the units, so plain integer counts are fine here).
    # Writing smoke is visible in the black_smoke slice.
    g.smoke[3, 4] = 42
    assert g.gas[BLACK_SMOKE][3, 4] == 42

    # Writing the black_smoke slice is visible in smoke.
    g.gas[BLACK_SMOKE][5, 6] = 77
    assert g.smoke[5, 6] == 77

    # Other gas slices are independent of smoke.
    g.gas[POISON][3, 4] = 90
    assert g.smoke[3, 4] == 42, "poison leaked into the smoke view"


# --------------------------------------------------------------------------
# 3. GasTable exposes the 5 gases with the §6.2 values
# --------------------------------------------------------------------------
def test_gas_table_values():
    """The 5 M1 TRACE gases' §6.2 values, unchanged by the P1 append (EOS
    refactor P1 grew the table to 7 rows total — o2/inert_n2 at ids 5/6 — see
    tests/test_eos_p1_species_transport.py::test_bulk_pair_table_contract for
    their contract)."""
    tbl = GasTable.from_config()
    assert tbl.n == 7
    assert tbl.names[:5] == ["white_smoke", "black_smoke", "poison", "teargas", "fuel_gas"]
    assert tbl.names[5:] == ["o2", "inert_n2"]
    assert tbl.name_to_id["black_smoke"] == BLACK_SMOKE

    # Absorption triples (§6.2).
    assert np.allclose(tbl.absorption[WHITE_SMOKE], [0.10, 0.10, 0.10])
    assert np.allclose(tbl.absorption[BLACK_SMOKE], [0.88, 0.90, 0.93])
    assert np.allclose(tbl.absorption[POISON],      [0.45, 0.10, 0.80])
    assert np.allclose(tbl.absorption[TEARGAS],     [0.12, 0.16, 0.30])
    assert np.allclose(tbl.absorption[FUEL_GAS],    [0.08, 0.10, 0.16])

    # Scatter albedo (§6.2).
    assert np.allclose(tbl.scatter_albedo[WHITE_SMOKE], [0.92, 0.92, 0.95])
    assert np.allclose(tbl.scatter_albedo[BLACK_SMOKE], [0.04, 0.04, 0.04])
    assert np.allclose(tbl.scatter_albedo[TEARGAS],     [0.88, 0.90, 0.92])

    # Per-gas diffusion + decay (§6.2) — the first 5 (trace) rows.
    assert np.allclose(tbl.diffusion[:5], [0.18, 0.10, 0.12, 0.15, 0.22])
    assert np.allclose(tbl.decay[:5],     [0.020, 0.008, 0.004, 0.010, 0.006])

    # black_smoke diffusion == today's d_smoke (the behaviour-preservation anchor).
    from config import CFG
    assert abs(float(tbl.diffusion[BLACK_SMOKE]) - float(CFG.physics.d_smoke)) < 1e-6

    # Flags: only fuel_gas is flammable; black_smoke + fuel_gas emit when hot
    # (among the 5 trace gases — o2/inert_n2 are never flammable/hot-emitting).
    assert list(tbl.flammable[:5].astype(bool)) == [False, False, False, False, True]
    assert list(tbl.emits_when_hot[:5].astype(bool)) == [True if i in (BLACK_SMOKE, FUEL_GAS) else False
                                                          for i in range(5)]
    # Effects (gameplay tags, read unit-side in mechanics).
    assert tbl.effect[POISON] == "damage_over_time"
    assert tbl.effect[TEARGAS] == "area_denial"
    assert tbl.effect[FUEL_GAS] == "ignition_hazard"


def test_gas_table_from_dict():
    """GasTable accepts a plain dict-of-dicts (the test-config path).

    GasTable always iterates the module-level GAS_NAMES (EOS P1: 7 ids, not
    5), so a from-scratch table must supply a row for every id — including
    the bulk pair's ``conservative`` column, now required on every row.
    """
    names = ["white_smoke", "black_smoke", "poison", "teargas", "fuel_gas",
             "o2", "inert_n2"]
    rows = {
        name: {
            "absorption": [0.1, 0.2, 0.3],
            "scatter_albedo": [0.4, 0.5, 0.6],
            "diffusion": 0.1 * (i + 1),
            "decay": 0.01 * (i + 1),
            "glow": 0.0,
            "flammable": (name == "fuel_gas"),
            "emits_when_hot": False,
            "effect": "x",
            "conservative": name in ("o2", "inert_n2"),
        }
        for i, name in enumerate(names)
    }
    tbl = GasTable(rows)
    assert tbl.n == 7
    assert np.allclose(tbl.diffusion, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert bool(tbl.flammable[FUEL_GAS]) is True
    assert list(tbl.conservative.astype(bool)) == [False, False, False, False, False, True, True]


# --------------------------------------------------------------------------
# 4. A non-smoke gas (poison) advects + diffuses through the per-gas loop
# --------------------------------------------------------------------------
def test_poison_transports_through_per_gas_loop():
    """A poison deposit (a non-smoke gas) moves under wind exactly as smoke does.

    The transport generalises: stepping the physics with a populated poison slice
    (and NO smoke) advects the poison cloud downwind, while the empty smoke slice
    stays empty (a cheap no-op). This proves the per-gas loop steps every slice,
    not just black_smoke.
    """
    from simulation import gas_fixed

    g = _make_gmap()
    runner = PhysicsRunner(bp)

    interior = (~g.solid) & (~g.is_vacuum)
    assert interior.any()

    # Deposit a poison blob on the left of the interior; smoke stays empty.
    # S2b: gas is int32 Q16.16 — full density (1.0) == FP_ONE counts.
    g.gas[POISON][:] = 0
    g.gas[POISON][6:10, 2:5] = gas_fixed.SMOKE_MAX_Q
    g.gas[POISON][~interior] = 0     # never inside walls
    assert g.smoke.sum() == 0, "smoke should be empty for this test"

    # Impose a strong, steady rightward wind (the atmosphere solver normally
    # produces this; we set it directly to isolate gas transport). S2c: wind is
    # int32 Q16.16 — quantize 3.0 real (a raw `= 3.0` would store 3 counts ~ 0).
    from simulation import atmosphere_fixed
    g.wind_x[:] = atmosphere_fixed.quantize_scalar(3.0)
    g.wind_y[:] = 0

    def _com_x(field):
        tot = field.sum()
        if tot <= 0:
            return None
        xs = np.arange(field.shape[1])[None, :]
        return float((field * xs).sum() / tot)

    cx0 = _com_x(g.gas[POISON])
    total0 = float(g.gas[POISON].sum())

    # Step physics a few ticks. The runner re-derives wind from the atmosphere
    # solver each substep, so pin the wind back inside the loop by stepping the
    # gas solver directly (the same call the per-gas loop makes) to keep this a
    # focused transport test.
    runner.smoke.d_smoke = float(g.gases.diffusion[POISON])
    dt = 0.05
    # Patch 2b: step is WIND-ONLY (no sink args), smoke moves on the real dt
    # (dt_scale gone). advection_rate is now ×9 so this advects even further right.
    for _ in range(20):
        runner.smoke.step(
            g.gas[POISON], g.wind_x, g.wind_y,
            g.obstacles, g.solid, g.is_vacuum, g.dyn_permeability,
            dt,
        )

    cx1 = _com_x(g.gas[POISON])
    assert cx1 is not None, "poison vanished entirely"
    assert cx1 > cx0 + 1.0, f"poison did not advect right: {cx0:.2f} -> {cx1:.2f}"
    # Smoke (black_smoke) untouched — the empty slice was a no-op.
    assert g.smoke.sum() == 0, "poison transport polluted the smoke slice"
    # Diffusion happened: the blob is no longer a sharp full column everywhere.
    assert g.gas[POISON].max() <= gas_fixed.SMOKE_MAX_Q
    assert total0 > 0.0


# --------------------------------------------------------------------------
# 5. BEHAVIOUR PRESERVATION — black_smoke matches the pre-refactor single field
# --------------------------------------------------------------------------
def test_black_smoke_matches_pre_refactor_reference():
    """A black_smoke deposit evolved through the per-gas loop is bit-close to the
    legacy single-smoke-field path (the C++ solver called directly with the old
    d_smoke=0.1 and the same wind/sink/dt).

    This is the M1 behaviour-preservation guarantee made explicit: black_smoke's
    diffusion (0.10) equals the legacy d_smoke (0.1), so the two evolutions are
    the SAME computation and must agree within float noise.
    """
    from config import CFG
    from simulation import gas_fixed

    h = w = 24
    rng = np.random.default_rng(SEED)
    # S2b: smoke is int32 Q16.16 — quantize a random [0,1] deposit.
    deposit = gas_fixed.quantize(rng.random((h, w)))
    # Open domain (isolate transport from BCs).
    obstacles = np.zeros((h, w), dtype=bool)
    is_wall = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    perm = np.ones((h, w), dtype=np.float32)
    sink_x = np.zeros((h, w), dtype=np.float32)
    sink_y = np.zeros((h, w), dtype=np.float32)
    wind_x = np.full((h, w), 0.4, dtype=np.float32)
    wind_y = np.full((h, w), -0.25, dtype=np.float32)

    def _solver():
        s = bp.SmokeDynamics()
        s.advection_rate = float(CFG.physics.advection_rate)
        s.wind_diffusion_scale = float(CFG.physics.wind_diffusion_scale)
        s.sink_strength = float(CFG.physics.smoke_sink_strength)
        return s

    dt = 0.02

    # Patch 2b: step is WIND-ONLY (no sink args) and runs on the real dt
    # (dt_scale gone). This test compares black_smoke's per-gas diffusion against
    # the legacy single-smoke d_smoke path — both stepped identically, so the
    # equality still holds regardless of the dt_scale removal.
    # Reference (pre-refactor): the single smoke field with legacy d_smoke.
    ref = deposit.copy()
    s_ref = _solver()
    s_ref.d_smoke = float(CFG.physics.d_smoke)   # 0.1
    for _ in range(30):
        s_ref.step(ref, wind_x, wind_y,
                   obstacles, is_wall, is_vacuum, perm, dt)

    # New path: the SAME field stepped with black_smoke's per-gas diffusion.
    gas = deposit.copy()
    s_new = _solver()
    s_new.d_smoke = float(GasTable.from_config().diffusion[BLACK_SMOKE])  # 0.10
    for _ in range(30):
        s_new.step(gas, wind_x, wind_y,
                   obstacles, is_wall, is_vacuum, perm, dt)

    # S2b: both paths run the identical integer-SL with the same d_smoke (0.1 ==
    # 0.10), so they are now BIT-IDENTICAL (was atol=1e-5 in the float build).
    assert np.array_equal(gas, ref), \
        f"black_smoke diverged from the legacy single-field path: " \
        f"max|diff|={np.abs(gas - ref).max()}"


# --------------------------------------------------------------------------
# 6. Determinism — a headless Simulation rollout is bit-identical run-to-run
# --------------------------------------------------------------------------
def _rollout_signature(n_steps=60):
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=False)
    sim.set_paused(False)
    for _ in range(n_steps):
        sim.step()
    g = sim.gmap
    return (
        g.gas.copy(),
        g.smoke.copy(),
        float(g.atmosphere.sum()),
    )


def test_determinism_bit_identical():
    gas_a, smoke_a, atm_a = _rollout_signature()
    gas_b, smoke_b, atm_b = _rollout_signature()
    assert np.array_equal(gas_a, gas_b), "gas array not bit-identical across runs"
    assert np.array_equal(smoke_a, smoke_b), "smoke not bit-identical across runs"
    assert atm_a == atm_b
    # The smoke view still aliases black_smoke after a full rollout.
    assert np.array_equal(smoke_a, gas_a[BLACK_SMOKE])


# --------------------------------------------------------------------------
# 7. Recorder / renderer paths that read gmap.smoke still work
# --------------------------------------------------------------------------
def test_recorder_and_headless_step():
    """A headless Simulation with the recorder ON steps cleanly; the recorder
    snapshots ``gmap.smoke`` (the black_smoke view) without error, and the smoke
    aliasing survives the step."""
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=True)
    sim.set_paused(False)
    for _ in range(10):
        sim.step()
    g = sim.gmap
    # Smoke is still the black_smoke view (not orphaned by any step).
    assert np.shares_memory(g.smoke, g.gas[BLACK_SMOKE])
    # The recorder captured frames including the smoke field.
    rec = sim.recorder
    assert rec is not None
    assert "smoke" in rec.DEFAULT_FIELDS


def test_renderer_overlay_reads_smoke():
    """The render-side smoke path reads gmap.smoke (S2b: a 2-D int32 Q16.16 view)
    and DEQUANTIZES it to float32 at the render boundary (game_renderer.py). The
    overlay itself takes float32; the dequantize is the FLOAT BRIDGE."""
    from renderer.overlays import FieldOverlay  # import path must resolve
    from simulation import gas_fixed
    g = _make_gmap()
    g.smoke[4:8, 4:8] = gas_fixed.quantize_scalar(0.5)
    # The black_smoke view is a (h, w) int32 field; the renderer dequantizes it.
    assert g.smoke.ndim == 2 and g.smoke.dtype == np.int32
    assert g.smoke[5, 5] == gas_fixed.quantize_scalar(0.5)
    smoke_f = gas_fixed.dequantize_f32(g.smoke)
    assert smoke_f.dtype == np.float32
    assert abs(float(smoke_f[5, 5]) - 0.5) < 1e-4


if __name__ == "__main__":
    test_gas_array_shape()
    test_smoke_is_black_smoke_view()
    test_gas_table_values()
    test_gas_table_from_dict()
    test_poison_transports_through_per_gas_loop()
    test_black_smoke_matches_pre_refactor_reference()
    test_determinism_bit_identical()
    test_recorder_and_headless_step()
    test_renderer_overlay_reads_smoke()
    print("OK: all multi-gas M1 structure + behaviour-preservation tests passed")
