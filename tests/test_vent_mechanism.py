"""Vent system PATCH 1 (issue #48, docs/vent_system_design_2026-08-23.md) —
the circulation mechanism gates: (a) bulk-pair + energy conservation exact
over a multi-tick vented run; (c) poison recirculation end-to-end (scrubbed
smoke into the counted sink, poison passing straight through).

Fixtures are programmatic LevelData / EntityInstance + real GameMaps (the
B1-B4 idiom, test_b4_pump.py) — no repo level is mutated, no golden moves.
Runs WITHOUT the compiled physics (``breach_physics=None``): the vent
mechanism edits ``gmap.gas``/``gmap.temperature`` directly through the
extended primitives — it needs no EOS transport to be exercised or to
conserve, exactly like the B4 pump gates.

Run:
    conda run -n data python -m pytest tests/test_vent_mechanism.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.gases import (  # noqa: E402
    FUEL_GAS, INERT_N2, N_GASES, O2, POISON, SMOKE, STEAM, TEARGAS,
)

TPS = 24   # config.toml [clock].ticks_per_second


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_b4_pump.py's _level/_inst helpers)
# ---------------------------------------------------------------------------

def _tm(h=10, w=10):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4              # interior air
    return tm


def _level(tm, entities=(), name="vent_fix", tile_size_m=1.0, **kw):
    return LevelData(name=name, version="1", path=Path("."), tilemap=tm,
                     tile_size_m=tile_size_m, diffuse_path=Path("."),
                     entities=list(entities), wires=[], **kw)


def _inst(cls_name, eid, ordinal, **overrides):
    cls = REGISTRY[cls_name]
    fields = {f.name: f.default for f in cls.FIELDS}
    fields.update(overrides)
    return EntityInstance(id=eid, class_name=cls_name, ordinal=ordinal,
                          tags=(), fields=fields)


def _duct_vent_pair(ret_xy, sup_xy, q_circ=5.0, filter_name="derelict"):
    """One duct + a (return, supply) floor-mounted vent pair, wired."""
    duct = _inst("duct", "d1", 0, filter=filter_name)
    rx, ry = ret_xy
    sx, sy = sup_xy
    ret = _inst("vent", "vret", 1, x=rx, y=ry, mount="floor", role="return",
               duct="d1", q_circ=q_circ)
    sup = _inst("vent", "vsup", 2, x=sx, y=sy, mount="floor", role="supply",
               duct="d1", q_circ=q_circ)
    return [duct, ret, sup]


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def _grid_total_all(gmap) -> int:
    return int(gmap.gas.astype(np.int64).sum())


def _ledger_total(duct) -> int:
    return duct.o2_raw + duct.n2_raw + sum(duct.trace_raw) + sum(duct.sink)


def _bulk_field_total(gmap) -> int:
    return int(gmap.gas[O2].astype(np.int64).sum()) + \
        int(gmap.gas[INERT_N2].astype(np.int64).sum())


def _bulk_energy_total(gmap) -> int:
    """Sum(N_bulk_tile * T_tile) over the whole grid — the EOS's own N
    convention (P-T0: n_total == n_bulk, cpp/src/eos_solver.cpp:719-729)."""
    n_bulk = gmap.gas[O2].astype(np.int64) + gmap.gas[INERT_N2].astype(np.int64)
    return int((n_bulk * gmap.temperature.astype(np.int64)).sum())


# ===========================================================================
# (a) Conservation — bulk mass + trace mass + energy, exact to the LSB
# ===========================================================================

def test_bulk_and_trace_mass_conserved_exactly_every_tick():
    """Grid total N (ALL slices) + the plenum's own bulk/trace/sink holdings
    is EXACTLY constant every tick — nothing manufactured or destroyed
    outside the counted sink (§4's conservation invariant)."""
    ents = _duct_vent_pair((2, 5), (7, 5), q_circ=8.0)
    sim = Simulation(_level(_tm(), ents), seed=1, breach_physics=None,
                     enable_recorder=False)
    ry, rx = sim._vents[0].aperture_y, sim._vents[0].aperture_x
    sim.gmap.gas[SMOKE][ry, rx] = 30000
    sim.gmap.gas[POISON][ry, rx] = 15000
    sim.gmap.gas[STEAM][ry, rx] = 7000

    total0 = _grid_total_all(sim.gmap)
    assert total0 > 0
    for _ in range(150):
        _step(sim)
        total = _grid_total_all(sim.gmap) + _ledger_total(sim._ducts[0])
        assert total == total0, "vent circulation leaked/fabricated mass"


def test_bulk_field_plus_plenum_bulk_conserved_exactly():
    """The BULK-ONLY sub-invariant (§4: "bulk-pair field totals + plenum
    pairs... = const") holds independently of trace/filter noise: bulk is
    NEVER filtered/sunk, so field bulk + plenum bulk is exactly constant."""
    ents = _duct_vent_pair((2, 5), (7, 5), q_circ=8.0)
    sim = Simulation(_level(_tm(), ents), seed=2, breach_physics=None,
                     enable_recorder=False)
    bulk0 = _bulk_field_total(sim.gmap)
    for _ in range(150):
        _step(sim)
        duct = sim._ducts[0]
        bulk = _bulk_field_total(sim.gmap) + duct.o2_raw + duct.n2_raw
        assert bulk == bulk0, "bulk-only sub-invariant violated"


def test_energy_conserved_exactly_when_no_trace_present():
    """The energy ledger (§4: "the energy books likewise"), isolated from
    the DELIBERATE trace-carried-heat asymmetry ("scrubbed smoke keeps its
    heat" — a trace gas contributes to E_plenum's credit at intake but not
    to the field's own N_bulk*T tally, by design): with ZERO trace gas ever
    present, Sum(N_bulk*T over the grid) + E_plenum + e_wipe is EXACTLY
    constant, proving the floor-division remainder banking (§4) and the
    T-rail hit accounting introduce no drift."""
    ents = _duct_vent_pair((2, 5), (7, 5), q_circ=6.0)
    sim = Simulation(_level(_tm(), ents), seed=3, breach_physics=None,
                     enable_recorder=False)
    ry, rx = sim._vents[0].aperture_y, sim._vents[0].aperture_x
    sim.gmap.temperature[:, :] = 5000        # uniform warm field, no trace anywhere
    e0 = _bulk_energy_total(sim.gmap)
    for _ in range(200):
        _step(sim)
        duct = sim._ducts[0]
        e = _bulk_energy_total(sim.gmap) + duct.e_plenum + duct.e_wipe
        assert e == e0, "energy ledger drifted with zero trace in play"
    # Sanity: the mechanism actually MOVED something (vacuous-gate guard).
    assert duct.o2_raw != 0 or duct.n2_raw != 0 or \
        int(sim.gmap.gas[O2][sim._vents[1].aperture_y,
                             sim._vents[1].aperture_x]) > 0


def test_deposit_runs_warm_under_smoky_intake_scrubbed_or_not():
    """§4 decision: scrubbed smoke KEEPS its heat — a hot, smoky return
    tile raises the plenum's T_dep (and hence the supply deposit's T) even
    though the smoke's MASS is scrubbed into the counted sink. Qualitative/
    directional check (the exact-energy case above already proves the
    ledger arithmetic to the LSB)."""
    ents = _duct_vent_pair((2, 5), (7, 5), q_circ=10.0, filter_name="hepa_basic")
    sim = Simulation(_level(_tm(), ents), seed=4, breach_physics=None,
                     enable_recorder=False)
    ry, rx = sim._vents[0].aperture_y, sim._vents[0].aperture_x
    sy, sx = sim._vents[1].aperture_y, sim._vents[1].aperture_x
    sim.gmap.temperature[:, :] = 0
    sim.gmap.temperature[ry, rx] = 20000     # hot return tile
    sim.gmap.gas[SMOKE][ry, rx] = 40000      # heavy smoke — fully scrubbed by hepa_basic
    for _ in range(60):
        _step(sim)
    duct = sim._ducts[0]
    assert duct.sink[1] > 0, "smoke was never scrubbed — test is vacuous"  # SMOKE index 1
    assert int(sim.gmap.temperature[sy, sx]) > 0, \
        "supply deposit never warmed despite hot+smoky return intake"


# ===========================================================================
# (c) Poison recirculation E2E — the mission-mechanic gate (§4)
# ===========================================================================

def test_poison_recirculates_smoke_scrubbed_hepa_filter():
    """A chemical grenade fed into a return re-emerges at the supply on the
    SAME duct (poison passes a filter untouched); smoke fed the same way
    is fully scrubbed into the counted sink and NEVER reaches the supply
    tile (§4's headline filter behaviours)."""
    ents = _duct_vent_pair((2, 5), (7, 5), q_circ=12.0, filter_name="hepa_basic")
    sim = Simulation(_level(_tm(), ents), seed=5, breach_physics=None,
                     enable_recorder=False)
    ry, rx = sim._vents[0].aperture_y, sim._vents[0].aperture_x
    sy, sx = sim._vents[1].aperture_y, sim._vents[1].aperture_x
    sim.gmap.gas[POISON][ry, rx] = 50000
    sim.gmap.gas[SMOKE][ry, rx] = 50000

    for _ in range(300):
        _step(sim)

    duct = sim._ducts[0]
    poison_at_supply = int(sim.gmap.gas[POISON][sy, sx])
    smoke_at_supply = int(sim.gmap.gas[SMOKE][sy, sx])
    assert poison_at_supply > 0, "poison never recirculated to the supply vent"
    assert smoke_at_supply == 0, "scrubbed smoke leaked past the hepa filter"
    assert duct.sink[1] > 0                # SMOKE index 1 — the counted mass sink
    assert duct.sink[2] == 0               # POISON index 2 — a hepa filter passes gas


def test_derelict_filter_scrubs_nothing_ducts_merely_redistribute():
    """The `derelict` filter row (all-zero efficiency, §4: "derelict
    all-zeros — ducts merely redistribute") passes EVERY trace species
    through untouched — the counted sink stays exactly zero."""
    ents = _duct_vent_pair((2, 5), (7, 5), q_circ=12.0, filter_name="derelict")
    sim = Simulation(_level(_tm(), ents), seed=6, breach_physics=None,
                     enable_recorder=False)
    ry, rx = sim._vents[0].aperture_y, sim._vents[0].aperture_x
    for gid in (STEAM, SMOKE, POISON, TEARGAS, FUEL_GAS):
        sim.gmap.gas[gid][ry, rx] = 10000
    for _ in range(300):
        _step(sim)
    duct = sim._ducts[0]
    assert duct.sink == [0] * 5, "the derelict (all-zero) filter scrubbed something"


# ===========================================================================
# The N_EPS near-empty-plenum wipe (§4) — a dedicated unit exercise
# ===========================================================================

def test_near_empty_plenum_wipes_residual_energy_no_crash_no_leak():
    """Below the N_EPS floor, the E/N divide is untrustworthy — the
    residual energy is wiped into the counted `e_wipe` channel (not left to
    silently distort a future T_dep once the plenum refills) and NOTHING
    is distributed that tick. Drives ``_duct_sweep`` directly (a tiny,
    controlled plenum state — the field-level scenario doesn't reliably
    hit this corner)."""
    from simulation.vent_system import N_EPS, _duct_sweep, _t_rails_q

    ents = _duct_vent_pair((2, 5), (7, 5), q_circ=1.0)
    sim = Simulation(_level(_tm(), ents), seed=8, breach_physics=None,
                     enable_recorder=False)
    duct = sim._ducts[0]
    ret, sup = sim._vents
    duct.o2_raw = N_EPS - 1                 # BELOW the floor
    duct.n2_raw = 0
    duct.e_plenum = 5_000_000               # stale residual energy
    t_min_q, t_max_q = _t_rails_q()

    sy, sx = sup.aperture_y, sup.aperture_x
    supply_before = int(sim.gmap.gas[O2][sy, sx]) + int(sim.gmap.gas[INERT_N2][sy, sx])

    _duct_sweep(sim.gmap, duct, [], [sup], t_min_q, t_max_q)

    assert duct.e_plenum == 0, "residual energy was not wiped"
    assert duct.e_wipe == 5_000_000, "the wipe channel did not receive the residual"
    supply_after = int(sim.gmap.gas[O2][sy, sx]) + int(sim.gmap.gas[INERT_N2][sy, sx])
    assert supply_after == supply_before, "a near-empty plenum still deposited"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
