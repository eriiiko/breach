"""Multi-gas system — M1 structure + behaviour-preservation tests (engine/05 §6.2).

M1 generalises the single ``smoke`` scalar field into N gas density fields
(``gmap.gas``, shape ``(N, h, w)``) + a data-driven ``[gases.*]`` table, WITHOUT
changing any visible behaviour: the existing smoke becomes the ``smoke``
slice, ``gmap.smoke`` is a view onto it, and the per-gas transport loop steps each
gas with the SAME C++ smoke solver. (At M1 the render march still read
``gmap.smoke``; per-channel colour summation over gases came at M2 and is the
radiation sweep's light channels' since ray-engine-v2 P6a.)

These tests assert the M1 contract:

1. ``gmap.gas`` has shape ``(N_GASES, h, w)`` float32.
2. ``gmap.smoke`` IS the ``smoke`` slice — a view: writing one is visible
   in the other (both directions).
3. ``GasTable`` exposes the 5 gases (steam / smoke / poison / teargas
   / fuel_gas) with the §6.2 absorption / scatter / diffusion / decay / flags.
4. A populated NON-smoke gas (poison) rides the air through the full physics
   tick exactly as smoke does (transport generalises).
5. ONE LAW FOR EVERY TRACE PLANE — two trace gases with equal dials evolve
   bit-identically through a blast (smoke transport v2, #12: the transport is
   the table's dials, never a per-gas branch). (Was: smoke matched the legacy
   single-field SmokeDynamics reference; that solver is deleted.)
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
    STEAM,
    SMOKE,
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
# 2. gmap.smoke is the smoke slice (a VIEW — aliasing both ways)
# --------------------------------------------------------------------------
def test_smoke_is_black_smoke_view():
    g = _make_gmap()
    # Same memory (a view, not a copy).
    assert g.smoke.base is g.gas, "gmap.smoke is not a view into gmap.gas"
    assert np.shares_memory(g.smoke, g.gas[SMOKE])

    # S2b: gas is int32 Q16.16 — write raw counts (the view aliasing is what's
    # under test, not the units, so plain integer counts are fine here).
    # Writing smoke is visible in the smoke slice.
    g.smoke[3, 4] = 42
    assert g.gas[SMOKE][3, 4] == 42

    # Writing the smoke slice is visible in smoke.
    g.gas[SMOKE][5, 6] = 77
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
    assert tbl.names[:5] == ["steam", "smoke", "poison", "teargas", "fuel_gas"]
    assert tbl.names[5:] == ["o2", "inert_n2"]
    assert tbl.name_to_id["smoke"] == SMOKE

    # Absorption triples (§6.2).
    assert np.allclose(tbl.absorption[STEAM], [0.10, 0.10, 0.10])
    assert np.allclose(tbl.absorption[SMOKE], [0.88, 0.90, 0.93])
    assert np.allclose(tbl.absorption[POISON],      [0.45, 0.10, 0.80])
    assert np.allclose(tbl.absorption[TEARGAS],     [0.12, 0.16, 0.30])
    assert np.allclose(tbl.absorption[FUEL_GAS],    [0.08, 0.10, 0.16])

    # Scatter albedo (§6.2).
    assert np.allclose(tbl.scatter_albedo[STEAM], [0.92, 0.92, 0.95])
    assert np.allclose(tbl.scatter_albedo[SMOKE], [0.04, 0.04, 0.04])
    assert np.allclose(tbl.scatter_albedo[TEARGAS],     [0.88, 0.90, 0.92])

    # Per-gas diffusion + decay (§6.2) — the first 5 (trace) rows.
    assert np.allclose(tbl.diffusion[:5], [0.18, 0.10, 0.12, 0.15, 0.22])
    assert np.allclose(tbl.decay[:5],     [0.020, 0.008, 0.004, 0.010, 0.006])

    # (The "smoke diffusion == [physics] d_smoke" anchor is gone with the key:
    # smoke transport v2, #12, retired d_smoke with the SL smoke step.)

    # Flags: only fuel_gas is flammable; smoke + fuel_gas emit when hot
    # (among the 5 trace gases — o2/inert_n2 are never flammable/hot-emitting).
    assert list(tbl.flammable[:5].astype(bool)) == [False, False, False, False, True]
    assert list(tbl.emits_when_hot[:5].astype(bool)) == [True if i in (SMOKE, FUEL_GAS) else False
                                                          for i in range(5)]
    # Effects (gameplay tags, read unit-side in mechanics).
    assert tbl.effect[POISON] == "damage_over_time"
    assert tbl.effect[TEARGAS] == "area_denial"
    assert tbl.effect[FUEL_GAS] == "ignition_hazard"


def test_gas_table_from_dict():
    """GasTable accepts a plain dict-of-dicts (the test-config path).

    GasTable always iterates the module-level GAS_NAMES (EOS P1: 7 ids, not
    5), so a from-scratch table must supply a row for every id — including
    the bulk pair's ``conservative`` column, now required on every row, and
    (ray-engine-v2 P5a) the ``heat_absorb`` column, required the same way.
    """
    names = ["steam", "smoke", "poison", "teargas", "fuel_gas",
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
            "heat_absorb": 0.0,
        }
        for i, name in enumerate(names)
    }
    tbl = GasTable(rows)
    assert tbl.n == 7
    assert np.allclose(tbl.diffusion, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert bool(tbl.flammable[FUEL_GAS]) is True
    assert list(tbl.conservative.astype(bool)) == [False, False, False, False, False, True, True]


# --------------------------------------------------------------------------
# 4. A non-smoke gas (poison) rides the air through the full physics tick
# --------------------------------------------------------------------------
def _heat_push(g, y, x, amount=3200.0, radius=3.0):
    """A heat-only blast through the FieldEdit queue (the deposit_heat shape):
    the air it heats expands and pushes everything it carries outward."""
    from simulation.field_edit import (
        EditMode, EditQueue, Falloff, FieldEdit, Region)
    q = EditQueue()
    q.enqueue(FieldEdit(field="heat", region=Region.DISC,
                        coords=(y, x, float(radius)), amount=float(amount),
                        mode=EditMode.ADD, falloff=Falloff.LINEAR, source_id=991))
    q.flush(g, np.random.default_rng(SEED))


def test_poison_transports_through_per_gas_loop():
    """PROPERTY: a poison deposit (a non-smoke gas) rides the air through the
    full physics tick -- a blast on its left pushes the cloud right -- while
    the empty smoke slice stays empty. Every trace slice rides, not just smoke.

    BREAKS IF: the trace ride (bulk_transport.cpp stage 3b) is restricted to
    one gas, or an empty slice gets written.
    """
    g = _make_gmap()
    runner = PhysicsRunner(bp)
    interior = (~g.solid) & (~g.is_vacuum)
    g.gas[POISON][:] = 0
    g.gas[POISON][5:11, 6:10] = 65536
    g.gas[POISON][~interior] = 0
    assert g.smoke.sum() == 0

    def _com_x(field):
        f = field.astype(np.float64)
        return float((f * np.arange(field.shape[1])[None, :]).sum() / f.sum())

    cx0 = _com_x(g.gas[POISON])
    _heat_push(g, 8, 2)
    for _ in range(12):
        runner.step(g, 1.0 / 24.0)
    assert g.gas[POISON].sum() > 0, "poison vanished entirely"
    assert _com_x(g.gas[POISON]) > cx0 + 0.25, "poison did not ride the blast right"
    assert g.smoke.sum() == 0, "poison transport polluted the smoke slice"


# --------------------------------------------------------------------------
# 5. One law for every trace plane
# --------------------------------------------------------------------------
def test_black_smoke_matches_pre_refactor_reference():
    """PROPERTY: two trace planes with EQUAL table dials (diffusion, decay)
    and the same deposit evolve bit-identically through a blast: the trace
    transport is the table's dials, never a per-gas branch (smoke transport
    v2, #12; the M1 behaviour-preservation anchor this test held -- smoke ==
    the legacy single-field SmokeDynamics path -- is retired with that
    solver).

    BREAKS IF: any trace plane gets its own transport arithmetic (a per-gas
    if, a smoke-only clamp, a hardcoded id in the ride or the tail).
    """
    g = _make_gmap()
    runner = PhysicsRunner(bp)
    g.gases.diffusion[POISON] = g.gases.diffusion[SMOKE]
    g.gases.decay[POISON] = g.gases.decay[SMOKE]
    rng = np.random.default_rng(SEED)
    interior = (~g.solid) & (~g.is_vacuum)
    deposit = np.where(interior, (rng.random(g.smoke.shape) * 65536 * 1.5), 0)
    g.gas[SMOKE][:] = deposit.astype(np.int32)
    g.gas[POISON][:] = deposit.astype(np.int32)
    _heat_push(g, 7, 4)
    for _ in range(24):
        runner.step(g, 1.0 / 24.0)
    assert g.gas[SMOKE].any()
    assert np.array_equal(g.gas[SMOKE], g.gas[POISON]),         f"max|diff| = {np.abs(g.gas[SMOKE] - g.gas[POISON]).max()}"


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
    # The smoke view still aliases smoke after a full rollout.
    assert np.array_equal(smoke_a, gas_a[SMOKE])


# --------------------------------------------------------------------------
# 7. Recorder / renderer paths that read gmap.smoke still work
# --------------------------------------------------------------------------
def test_recorder_and_headless_step():
    """A headless Simulation with the recorder ON steps cleanly; the recorder
    snapshots ``gmap.smoke`` (the smoke view) without error, and the smoke
    aliasing survives the step."""
    level = load_level("unhcr_vessel")
    sim = Simulation(level, seed=SEED, breach_physics=bp, enable_recorder=True)
    sim.set_paused(False)
    for _ in range(10):
        sim.step()
    g = sim.gmap
    # Smoke is still the smoke view (not orphaned by any step).
    assert np.shares_memory(g.smoke, g.gas[SMOKE])
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
    # The smoke view is a (h, w) int32 field; the renderer dequantizes it.
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
