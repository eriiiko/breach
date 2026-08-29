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
    convention (P-T0: n_total == n_bulk, cpp/src/eos_solver.cpp:719-729).

    arc #54 P-G1b: read off the STORED `gas_energy`, in the plenum's own
    RELATIVE currency (`E - N*T_AMB`), not off `temperature`. The mirror is a
    FLOOR read of the stored truth — `T = floordiv(E, N) - T_AMB` — so a
    mirror-based sum under-reports the books by up to `N-1` raw counts per
    cell, which on a 200-tick run is a drift this exact-to-the-LSB ledger
    would blame on the vent system. The field IS the ledger now."""
    acct = gmap._gas_energy_accountable()
    n_bulk = gmap.gas[O2].astype(np.int64) + gmap.gas[INERT_N2].astype(np.int64)
    t_amb = gmap._gas_energy_t_amb_raw()
    return int(np.where(acct, gmap.gas_energy - n_bulk * t_amb, 0)
               .sum(dtype=np.int64))


def _rail_destroyed(gmap) -> int:
    """arc #54 P-G1b: the energy the deposit-site T-RAIL destroyed, signed.

    The vent sweep runs at slot 9e — AFTER the EOS's once-per-tick recovery —
    so `inject_gas_n_vec` carries the rails itself, and a clamp is a COUNTED
    DESTRUCTION at the tile (GameMap books it to `pump_rail`). It is no longer
    banked back into the plenum: charging the duct less because the tile
    railed would quietly re-create the destroyed energy inside the duct, which
    is exactly the class of leak this arc closes. So the ledger below has to
    name it as its own term rather than expect it to come back."""
    return int(gmap.gas_energy_books.get("pump_rail", 0))


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


def test_energy_conserved_exactly_with_trace_and_filtering_present():
    """The energy ledger (§4: "the energy books likewise") — REVIEW FIX
    (2026-08-23): intake credits `E_plenum` BULK-ONLY (per the engine's own
    P-T0 convention, `n_total == n_bulk` — trace carries no engine-side
    energy anywhere else), so the bulk-only invariant
    Sum(N_bulk*T over the grid) + E_plenum + e_wipe is EXACTLY constant EVEN
    WITH heavy trace gas present and filtered at intake (the "scrubbed smoke
    keeps its heat" framing that needed a zero-trace carve-out is retired —
    crediting the trace share was itself the bug: an uncounted energy
    source). Hot, smoky, poisoned intake through a scrubbing filter,
    multi-tick."""
    ents = _duct_vent_pair((2, 5), (7, 5), q_circ=6.0, filter_name="hepa_basic")
    sim = Simulation(_level(_tm(), ents), seed=3, breach_physics=None,
                     enable_recorder=False)
    ry, rx = sim._vents[0].aperture_y, sim._vents[0].aperture_x
    # arc #54 P-G1b: seed through the ONE sanctioned seam. A bare
    # `temperature[...] = ` write leaves `gas_energy` (the stored truth) behind,
    # and the pump primitives price their withdrawal off that field.
    sim.gmap.seed_gas_temperature(np.s_[:, :], 5000)
    sim.gmap.seed_gas_temperature((ry, rx), 20000)   # hot return tile
    sim.gmap.gas[SMOKE][ry, rx] = 40000       # heavy smoke — scrubbed by hepa_basic
    sim.gmap.gas[POISON][ry, rx] = 25000      # poison — passes hepa_basic untouched
    e0 = _bulk_energy_total(sim.gmap)
    for _ in range(200):
        _step(sim)
        duct = sim._ducts[0]
        e = (_bulk_energy_total(sim.gmap) + duct.e_plenum + duct.e_wipe
             - _rail_destroyed(sim.gmap))
        assert e == e0, "energy ledger drifted with trace present and filtered"
    # Sanity: the mechanism actually MOVED something AND actually scrubbed
    # something (vacuous-gate guard).
    assert duct.o2_raw != 0 or duct.n2_raw != 0 or \
        int(sim.gmap.gas[O2][sim._vents[1].aperture_y,
                             sim._vents[1].aperture_x]) > 0
    assert duct.sink[1] > 0, "smoke was never scrubbed — test is vacuous"  # SMOKE index 1


def test_multi_supply_bulk_split_never_goes_negative_adversarial_ratio():
    """REVIEW FIX (MAJOR): two supply vents on ONE duct, an adversarial
    skewed plenum ratio (99:1 o2:n2) — the ORIGINAL per-vent "floor +
    exact-complement" split inflated the summed MINOR species past its
    actual holdings across several vents (concrete repro: o2=99, n2=1, two
    shares of 50 -> n2 goes to -1). The fixed `gas_proportional_split`-
    against-the-REMAINING-pool approach must hold per-species (not just
    summed) conservation and NEVER drive a holding negative, tick after
    tick, as the pre-loaded plenum drains through the N_EPS floor."""
    duct = _inst("duct", "d1", 0, filter="derelict")
    sup_a = _inst("vent", "vsup_a", 1, x=3, y=5, mount="floor", role="supply",
                 duct="d1", q_circ=9.0)
    sup_b = _inst("vent", "vsup_b", 2, x=6, y=5, mount="floor", role="supply",
                 duct="d1", q_circ=4.0)      # DIFFERENT weight from sup_a
    sim = Simulation(_level(_tm(), [duct, sup_a, sup_b]), seed=9,
                     breach_physics=None, enable_recorder=False)
    duct_rt = sim._ducts[0]
    duct_rt.o2_raw = 9900
    duct_rt.n2_raw = 100                     # the adversarial 99:1 ratio
    o2_total0 = 9900 + int(sim.gmap.gas[O2].astype(np.int64).sum())
    n2_total0 = 100 + int(sim.gmap.gas[INERT_N2].astype(np.int64).sum())

    for _ in range(400):
        _step(sim)
        assert duct_rt.o2_raw >= 0, "o2_raw went negative"
        assert duct_rt.n2_raw >= 0, "n2_raw went negative"
        o2_now = duct_rt.o2_raw + int(sim.gmap.gas[O2].astype(np.int64).sum())
        n2_now = duct_rt.n2_raw + int(sim.gmap.gas[INERT_N2].astype(np.int64).sum())
        assert o2_now == o2_total0, "O2 not conserved per-species (not just summed)"
        assert n2_now == n2_total0, "N2 not conserved per-species (not just summed)"

    # Vacuous-gate guard: both supply vents must have actually received
    # SOMETHING over the run (the multi-vent split path was really exercised).
    ay, ax = sup_a.fields["y"], sup_a.fields["x"]
    by, bx = sup_b.fields["y"], sup_b.fields["x"]
    got_a = int(sim.gmap.gas[O2][ay, ax]) + int(sim.gmap.gas[INERT_N2][ay, ax])
    got_b = int(sim.gmap.gas[O2][by, bx]) + int(sim.gmap.gas[INERT_N2][by, bx])
    assert got_a > 0 and got_b > 0, \
        "one of the two supply vents never received a deposit — test is vacuous"


def test_t_rail_clamp_hi_and_lo_bank_exact_energy_and_count_hits():
    """REVIEW FIX (minor): engineer `e_plenum`/`n_bulk` so `t_dep` forces
    each T-rail on deposit — the MEASURED-delta debit (§4) must still bank
    the clamp exactly (no energy leak), and `rail_lo_hits`/`rail_hi_hits`
    must increment. Drives `_duct_sweep` directly (needs precise control
    over e_plenum relative to n_bulk that a field scenario can't reliably
    hit)."""
    from simulation.vent_system import _duct_sweep, _t_rails_q

    ents = _duct_vent_pair((2, 5), (7, 5), q_circ=50.0)
    sim = Simulation(_level(_tm(), ents), seed=10, breach_physics=None,
                     enable_recorder=False)
    duct = sim._ducts[0]
    ret, sup = sim._vents
    t_min_q, t_max_q = _t_rails_q()
    sy, sx = sup.aperture_y, sup.aperture_x

    # The tile's PRE-EXISTING ambient bulk N is a WEIGHT in the mass-weighted
    # mix (T_new = (N_old*T_old + dN*T_dep)/(N_old+dN)) — an ambient-filled
    # tile would dilute even an astronomical t_dep back under the rail
    # (delta_n_bulk is a small fraction of the mix). Zero the aperture's bulk
    # gas first so N_old == 0 and T_new == t_dep exactly — a clean, robust
    # way to force the rail regardless of the vent's per-tick quantum size.
    def _zero_bulk_at(y, x):
        sim.gmap.gas[O2][y, x] = 0
        sim.gmap.gas[INERT_N2][y, x] = 0
        # arc #54 P-G1b: a direct bulk-N write needs its stored energy
        # re-derived at the cells it authored, or `gas_energy` keeps holding
        # the energy of mass that is no longer there — and the deposit below
        # would then read a tile whose E and N disagree.
        sim.gmap.reseed_gas_energy((y, x))

    # --- HI rail ---------------------------------------------------------
    _zero_bulk_at(sy, sx)
    duct.o2_raw, duct.n2_raw = 50000, 50000
    duct.e_plenum = (t_max_q * 10) * (duct.o2_raw + duct.n2_raw)  # forces t_dep >> t_max_q
    e_before = (_bulk_energy_total(sim.gmap) + duct.e_plenum
                + duct.e_wipe - _rail_destroyed(sim.gmap))
    hits_before = duct.rail_hi_hits
    _duct_sweep(sim.gmap, duct, [], [sup], t_min_q, t_max_q)
    assert duct.rail_hi_hits == hits_before + 1, "the hi rail never fired — vacuous"
    assert int(sim.gmap.temperature[sy, sx]) == t_max_q
    e_after = (_bulk_energy_total(sim.gmap) + duct.e_plenum
               + duct.e_wipe - _rail_destroyed(sim.gmap))
    assert e_after == e_before, "the hi-rail clamp leaked energy out of the ledger"

    # --- LO rail (fresh plenum + a cold, zeroed aperture) -----------------
    _zero_bulk_at(sy, sx)
    duct.o2_raw, duct.n2_raw = 50000, 50000
    duct.e_plenum = (t_min_q * 10) * (duct.o2_raw + duct.n2_raw)  # forces t_dep << t_min_q (t_min_q < 0)
    duct.e_wipe = 0
    e_before = (_bulk_energy_total(sim.gmap) + duct.e_plenum
                + duct.e_wipe - _rail_destroyed(sim.gmap))
    hits_before = duct.rail_lo_hits
    _duct_sweep(sim.gmap, duct, [], [sup], t_min_q, t_max_q)
    assert duct.rail_lo_hits == hits_before + 1, "the lo rail never fired — vacuous"
    assert int(sim.gmap.temperature[sy, sx]) == t_min_q
    e_after = (_bulk_energy_total(sim.gmap) + duct.e_plenum
               + duct.e_wipe - _rail_destroyed(sim.gmap))
    assert e_after == e_before, "the lo-rail clamp leaked energy out of the ledger"


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
