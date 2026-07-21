"""A5 — EOS evacuation rule: ``seal_tiles`` / ``unseal_tiles``
(docs/a5_evacuation_impl_2026-07-18.md v2, test plan §12).

The door-close half of the eos_refactor_design.md §2.2 occupancy-transition
rule, plus its conservative open half. THE gate: cycling the primitives in a
sealed room conserves every gas slice's grid total EXACTLY (integer
equality), pure-primitive and through real ``PhysicsRunner.step`` ticks.
Dormancy is structural (no sim call sites), so no golden is touched.

Run:
    conda run -n data python -m pytest tests/test_a5_seal_evacuation.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import atmosphere_fixed, gas_fixed  # noqa: E402
from simulation.gamemap import (  # noqa: E402
    GameMap, SealBlocked,
    MAT_AIR, MAT_HULL, MAT_DOOR, MAT_FURNITURE,
)
from simulation.gases import (  # noqa: E402
    N_GASES, BLACK_SMOKE, POISON, O2, INERT_N2,
)
from simulation.physics_runner import PhysicsRunner  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures + helpers (the test_eos_p1_species_transport.py idiom)
# ---------------------------------------------------------------------------
def _sealed_room_level(h=12, w=12) -> LevelData:
    """A hull-walled sealed box, interior air, NO vacuum/breach anywhere."""
    tm = np.ones((h, w), dtype=np.int32)      # all hull
    tm[1:h - 1, 1:w - 1] = 4                   # carve interior air
    return LevelData(name="a5_sealed_room", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _make_gmap(h=12, w=12) -> GameMap:
    return GameMap(_sealed_room_level(h, w))


def _all_hull_level(h=12, w=12) -> LevelData:
    tm = np.ones((h, w), dtype=np.int32)
    return LevelData(name="a5_all_hull", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _slice_totals(g):
    return [int(g.gas[i].astype(np.int64).sum()) for i in range(N_GASES)]


def _make_solid(g, y, x, mat=MAT_HULL):
    """Turn an open tile into a legit solid tile (as if authored that way):
    material + cache patch + the solid-steady-state fields (solids carry no
    gas/P — the solver invariant the primitives assume)."""
    g.material[y, x] = mat
    g.on_tile_changed(y, x)
    for i in range(N_GASES):
        g.gas[i][y, x] = 0
    g.atmosphere[y, x] = 0
    g.wave_p[y, x] = 0


def _make_exposed_vacuum(g, y, x):
    """Turn a tile into exposed vacuum (an open breach side)."""
    g.material[y, x] = MAT_AIR
    g.on_tile_changed(y, x)
    g.is_vacuum[y, x] = True
    for i in range(N_GASES):
        g.gas[i][y, x] = 0
    g.atmosphere[y, x] = 0
    g.wave_p[y, x] = 0


def _snapshot(g):
    return {
        "gas": g.gas.copy(), "material": g.material.copy(),
        "atmosphere": g.atmosphere.copy(), "wave_p": g.wave_p.copy(),
        "solid": g.solid.copy(), "is_vacuum": g.is_vacuum.copy(),
        "water_depth": g.water_depth.copy(), "wall_hp": g.wall_hp.copy(),
        "temperature": g.temperature.copy(),
    }


def _assert_unchanged(g, snap):
    for name, arr in snap.items():
        assert np.array_equal(getattr(g, name), arr), (
            f"atomicity violated: field '{name}' mutated on a refused call")


# ---------------------------------------------------------------------------
# §12.1 — sealed-room door-cycle EXACT N conservation (the A5 gate)
# ---------------------------------------------------------------------------
def test_cycle_exact_conservation_all_slices():
    g = _make_gmap()
    t = (6, 6)
    # Nonzero trace gas in and around the doorway: all 7 slices must move
    # by the same exact rule (pure-primitive cycling — no physics step, so
    # per-slice exactness holds for traces too).
    g.gas[BLACK_SMOKE][6, 6] = 12345
    g.gas[POISON][6, 6] = 7777
    g.gas[POISON][5, 6] = 999
    totals0 = _slice_totals(g)

    for cycle in range(100):
        g.seal_tiles([t], MAT_DOOR)
        assert _slice_totals(g) == totals0, f"seal leaked on cycle {cycle}"
        if cycle == 0:
            # Solid steady state (design §6 table): no haunted-door values.
            assert g.material[t] == MAT_DOOR and bool(g.solid[t])
            assert all(int(g.gas[i][t]) == 0 for i in range(N_GASES))
            assert int(g.atmosphere[t]) == 0 and int(g.wave_p[t]) == 0
            assert int(g.wind_x[t]) == 0 and int(g.wind_y[t]) == 0
            assert int(g.flow_vx[t]) == 0 and int(g.flow_vy[t]) == 0
            assert float(g.ripple[t]) == 0.0 and float(g.ripple_v[t]) == 0.0
        g.unseal_tiles([t])
        assert _slice_totals(g) == totals0, f"unseal leaked on cycle {cycle}"
        assert g.material[t] == MAT_AIR and not bool(g.solid[t])


# ---------------------------------------------------------------------------
# §12.2 — full-tick cycle (repaired per critique S1: bulk-only fixture +
# stamp_units after each flip, mirroring the real slot-6 contract)
# ---------------------------------------------------------------------------
def test_full_tick_cycle_conservation_bulk_only():
    g = _make_gmap(h=16, w=16)
    runner = PhysicsRunner(bp)
    g.stamp_units([])
    assert not g.is_vacuum.any()

    # Seed real wind: a localized hot patch raises p* = C*N*T there (the P1
    # gate's idiom — bumping atmosphere directly no longer does anything).
    interior = (~g.solid) & (~g.is_vacuum)
    ys, xs = np.where(interior)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    bump = (np.abs(ys - cy) < 3) & (np.abs(xs - cx) < 3)
    g.temperature[ys[bump], xs[bump]] += atmosphere_fixed.quantize_scalar(200.0)

    # Bulk-only fixture: zero trace gas anywhere (trace decay is credited to
    # inert_N2, so per-slice totals would NOT be tick-invariant with smoke).
    assert all(int(g.gas[i].astype(np.int64).sum()) == 0 for i in range(5))

    t = (8, 8)
    o2_0 = int(g.gas[O2].astype(np.int64).sum())
    n2_0 = int(g.gas[INERT_N2].astype(np.int64).sum())
    sim_time = 1.0 / float(CFG.clock.ticks_per_second)

    for cycle in range(6):
        g.seal_tiles([t], MAT_DOOR)
        g.stamp_units([])            # slot-6 restamp: dyn_permeability fresh
        for _ in range(3):
            runner.step(g, sim_time)
            assert int(g.gas[O2].astype(np.int64).sum()) == o2_0, (
                f"O2 total drifted under ticks after seal (cycle {cycle})")
            assert int(g.gas[INERT_N2].astype(np.int64).sum()) == n2_0, (
                f"inert_N2 total drifted under ticks after seal (cycle {cycle})")
        g.unseal_tiles([t])
        g.stamp_units([])
        for _ in range(3):
            runner.step(g, sim_time)
            assert int(g.gas[O2].astype(np.int64).sum()) == o2_0, (
                f"O2 total drifted under ticks after unseal (cycle {cycle})")
            assert int(g.gas[INERT_N2].astype(np.int64).sum()) == n2_0, (
                f"inert_N2 total drifted under ticks after unseal (cycle {cycle})")


# ---------------------------------------------------------------------------
# §12.3 — remainder placement (N,S,E,W receiver order)
# ---------------------------------------------------------------------------
def test_remainder_placement_nsew():
    g = _make_gmap()
    _make_solid(g, 6, 5)                    # kill the W receiver -> k=3
    g.gas[POISON][6, 6] = 7                 # n=7, k=3 -> q=2, r=1
    g.seal_tiles([(6, 6)], MAT_DOOR)
    # Shares [3, 2, 2] in receiver order N (5,6), S (7,6), E (6,7).
    assert int(g.gas[POISON][5, 6]) == 3
    assert int(g.gas[POISON][7, 6]) == 2
    assert int(g.gas[POISON][6, 7]) == 2
    assert int(g.gas[POISON][6, 6]) == 0


# ---------------------------------------------------------------------------
# §12.4 — multi-tile span: no intra-span evacuation; permutation invariance
# ---------------------------------------------------------------------------
def test_multi_tile_span_permutation_invariance():
    span = [(6, 6), (6, 7), (6, 8)]
    results = []
    for order in (span, list(reversed(span)), [span[1], span[2], span[0]]):
        g = _make_gmap()
        g.gas[POISON][6, 6] = 101
        g.gas[POISON][6, 7] = 202
        g.gas[POISON][6, 8] = 303
        totals0 = _slice_totals(g)
        g.seal_tiles(order, MAT_DOOR)
        assert _slice_totals(g) == totals0
        # Span members hold zero gas; every receiver is outside the span.
        for t in span:
            assert all(int(g.gas[i][t]) == 0 for i in range(N_GASES))
            assert bool(g.solid[t]) and g.material[t] == MAT_DOOR
        results.append(_snapshot(g))
    for snap in results[1:]:
        for name in results[0]:
            assert np.array_equal(results[0][name], snap[name]), (
                f"span-order permutation changed field '{name}'")


# ---------------------------------------------------------------------------
# §12.5 — sealed pocket: refuse (never delete), atomically
# ---------------------------------------------------------------------------
def test_sealed_pocket_refusal_and_gasfree_pocket():
    g = _make_gmap()
    for (dy, dx) in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        _make_solid(g, 6 + dy, 6 + dx)
    snap = _snapshot(g)
    assert not g.can_seal_tiles([(6, 6)])
    with pytest.raises(SealBlocked):
        g.seal_tiles([(6, 6)], MAT_DOOR)
    _assert_unchanged(g, snap)

    # Gas-free pocket seals fine (nothing to move; no receivers needed).
    for i in range(N_GASES):
        g.gas[i][6, 6] = 0
    assert g.can_seal_tiles([(6, 6)])
    g.seal_tiles([(6, 6)], MAT_DOOR)
    assert bool(g.solid[6, 6])


# ---------------------------------------------------------------------------
# §12.6 — vacuum: breach receiver, sealing a breach (hull patch), vacuum join
# ---------------------------------------------------------------------------
def test_vacuum_neighbor_receives_share():
    g = _make_gmap()
    _make_exposed_vacuum(g, 6, 7)
    totals0 = _slice_totals(g)
    g.seal_tiles([(6, 6)], MAT_DOOR)
    # The breach counts as a receiver: it holds its share until the next
    # flux pass vents it (the sanctioned sink) — the primitive itself is
    # exactly conservative.
    assert int(g.gas[O2][6, 7]) > 0
    assert _slice_totals(g) == totals0


def test_seal_breach_tile_hull_patch_e2e():
    g = _make_gmap(h=14, w=14)
    runner = PhysicsRunner(bp)
    g.stamp_units([])
    g.destroy_wall(0, 5)                       # edge hull -> true breach
    assert bool(g.is_vacuum[0, 5]) and not bool(g.solid[0, 5])
    g.stamp_units([])

    sim_time = 1.0 / float(CFG.clock.ticks_per_second)
    o2_start = int(g.gas[O2].astype(np.int64).sum())
    for _ in range(10):
        runner.step(g, sim_time)
    o2_vented = int(g.gas[O2].astype(np.int64).sum())
    assert o2_vented < o2_start, "breach did not vent — hull-patch E2E vacuous"

    # Patch the hull: seal the breach tile. is_vacuum stays True -> the
    # sealed-hull state of engine/04 §2.3; the room holds pressure again.
    g.seal_tiles([(0, 5)], MAT_HULL)
    assert bool(g.is_vacuum[0, 5]) and bool(g.solid[0, 5])
    g.stamp_units([])
    o2_sealed = int(g.gas[O2].astype(np.int64).sum())
    n2_sealed = int(g.gas[INERT_N2].astype(np.int64).sum())
    for _ in range(10):
        runner.step(g, sim_time)
        assert int(g.gas[O2].astype(np.int64).sum()) == o2_sealed
        assert int(g.gas[INERT_N2].astype(np.int64).sum()) == n2_sealed


def test_unseal_adjacent_to_vacuum_joins_no_seed():
    g = _make_gmap()
    _make_exposed_vacuum(g, 6, 7)
    _make_solid(g, 6, 6)
    donor_vals = {d: [int(g.gas[i][d]) for i in range(N_GASES)]
                  for d in ((5, 6), (7, 6), (6, 5))}
    g.unseal_tiles([(6, 6)])
    assert bool(g.is_vacuum[6, 6]) and not bool(g.solid[6, 6])
    assert all(int(g.gas[i][6, 6]) == 0 for i in range(N_GASES))
    for d, vals in donor_vals.items():
        assert [int(g.gas[i][d]) for i in range(N_GASES)] == vals, (
            f"vacuum join must not touch donor {d}")


# ---------------------------------------------------------------------------
# §12.7 — water rule v1: span refused; flooded receiver allowed
# ---------------------------------------------------------------------------
def test_water_refusal_and_flooded_receiver():
    g = _make_gmap()
    g.water_depth[6, 6] = 65536                # 1 m, Q16.16
    snap = _snapshot(g)
    assert not g.can_seal_tiles([(6, 6)])
    with pytest.raises(SealBlocked):
        g.seal_tiles([(6, 6)], MAT_DOOR)
    _assert_unchanged(g, snap)
    g.water_depth[6, 6] = 0                    # drained -> seals

    # A flooded RECEIVER does not block (guard covers the span only): the
    # gas parks under the water column, exactly conserved (design §8, N3).
    g.water_depth[5, 6] = 65536
    totals0 = _slice_totals(g)
    assert g.can_seal_tiles([(6, 6)])
    g.seal_tiles([(6, 6)], MAT_DOOR)
    assert _slice_totals(g) == totals0


# ---------------------------------------------------------------------------
# §12.8 — two-tile room compression: exact doubling, then P rises
# ---------------------------------------------------------------------------
def test_two_tile_room_compression():
    g = GameMap(_all_hull_level())
    for x in (6, 7):
        g.material[6, x] = MAT_AIR
        g.on_tile_changed(6, x)
    g._update_caches()                          # reseed ambient in the 2 tiles
    o2_one = int(g.gas[O2][6, 6])
    n2_one = int(g.gas[INERT_N2][6, 6])
    assert o2_one > 0

    g.seal_tiles([(6, 6)], MAT_DOOR)
    assert int(g.gas[O2][6, 7]) == 2 * o2_one
    assert int(g.gas[INERT_N2][6, 7]) == 2 * n2_one

    # Physics smoke check (not a golden): next materialized P rises above
    # ambient in the compressed cell.
    runner = PhysicsRunner(bp)
    g.stamp_units([])
    runner.step(g, 1.0 / float(CFG.clock.ticks_per_second))
    assert int(g.atmosphere[6, 7]) > atmosphere_fixed.FP_ONE


# ---------------------------------------------------------------------------
# §12.9 — conservative unseal: the k+1 seed shape (critique B1) + cascade
# ---------------------------------------------------------------------------
def test_unseal_k1_alcove_halves_single_donor():
    g = GameMap(_all_hull_level())
    g.material[6, 6] = MAT_AIR                  # the single donor
    g.on_tile_changed(6, 6)
    g._update_caches()
    m_o2 = int(g.gas[O2][6, 6])
    assert m_o2 > 0
    g.unseal_tiles([(6, 7)])                    # alcove door: k=1
    # k+1 divisor: donor is HALVED, never drained to 0 (the v1 blocker).
    assert int(g.gas[O2][6, 7]) == m_o2 // 2
    assert int(g.gas[O2][6, 6]) == m_o2 - m_o2 // 2
    assert int(g.gas[O2][6, 6]) > 0


def test_unseal_k2_equalizes_over_three():
    g = _make_gmap()
    _make_solid(g, 6, 6)
    _make_solid(g, 5, 6)                        # kill N donor -> donors E, W
    _make_solid(g, 7, 6)                        # kill S donor
    m = 60000
    g.gas[O2][6, 7] = m
    g.gas[O2][6, 5] = m
    g.unseal_tiles([(6, 6)])
    # target = 2m // (k+1) = 2m // 3 = 40000: doorway and both donors all
    # end equalized (the v1 //k rule gave doorway m with donors at m/2 —
    # a 2x spike between two dips).
    assert int(g.gas[O2][6, 6]) == (2 * m) // 3
    assert int(g.gas[O2][6, 7]) == (2 * m) // 3
    assert int(g.gas[O2][6, 5]) == (2 * m) // 3
    total = int(g.gas[O2][6, 5]) + int(g.gas[O2][6, 6]) + int(g.gas[O2][6, 7])
    assert total == 2 * m                       # exact conservation


def test_unseal_cascade_with_zero_donor():
    g = _make_gmap()
    _make_solid(g, 6, 6)
    _make_solid(g, 5, 6)
    _make_solid(g, 7, 6)                        # donors: E (6,7), W (6,5)
    g.gas[POISON][6, 7] = 0
    g.gas[POISON][6, 5] = 10
    totals0 = _slice_totals(g)
    g.unseal_tiles([(6, 6)])
    # target = 10 // 3 = 3; balanced take clamps at the zero donor and the
    # cascade pulls the shortfall from the holder: donors end [0, 7].
    assert int(g.gas[POISON][6, 6]) == 3
    assert int(g.gas[POISON][6, 7]) == 0
    assert int(g.gas[POISON][6, 5]) == 7
    assert _slice_totals(g) == totals0
    assert int(g.gas.min()) >= 0                # clamping never goes negative


# ---------------------------------------------------------------------------
# §12.10 — round trip: totals forever; structural caches exactly restored
# ---------------------------------------------------------------------------
def test_round_trip_totals_and_caches():
    g = _make_gmap()
    pristine = _make_gmap()
    totals0 = _slice_totals(g)
    for _ in range(3):
        g.seal_tiles([(6, 6), (6, 7)], MAT_DOOR)
        g.unseal_tiles([(6, 6), (6, 7)])
    assert _slice_totals(g) == totals0
    for name in ("material", "solid", "permeability", "wall_hp", "flammable",
                 "face_shift", "heat_inv_shift", "conductivity",
                 "light_atten", "heat_atten", "wave_absorb", "is_vacuum"):
        assert np.array_equal(getattr(g, name), getattr(pristine, name)), (
            f"structural cache '{name}' did not round-trip")


# ---------------------------------------------------------------------------
# §12.11 — freed-refill interaction: stamp_units touches only atmosphere
# ---------------------------------------------------------------------------
def test_freed_refill_touches_no_gas():
    g = _make_gmap()
    _make_solid(g, 6, 6)
    g.stamp_units([])                           # obstacles include the pillar
    g.unseal_tiles([(6, 6)])
    totals0 = _slice_totals(g)
    g.stamp_units([])                           # freed-tile refill fires
    assert _slice_totals(g) == totals0


# ---------------------------------------------------------------------------
# §12.12 — determinism: identical script, two fresh maps, byte-identical
# ---------------------------------------------------------------------------
def test_determinism_identical_script_two_maps():
    def script(g):
        g.gas[POISON][5, 5] = 4321
        g.seal_tiles([(5, 5), (5, 6)], MAT_DOOR)
        g.unseal_tiles([(5, 5)])
        g.seal_tiles([(7, 7)], MAT_DOOR)
        g.unseal_tiles([(5, 6), (7, 7)])
    a, b = _make_gmap(), _make_gmap()
    script(a)
    script(b)
    snap_a, snap_b = _snapshot(a), _snapshot(b)
    for name in snap_a:
        assert np.array_equal(snap_a[name], snap_b[name]), (
            f"determinism: field '{name}' diverged between identical runs")


# ---------------------------------------------------------------------------
# §12.13 — strictness: caller bugs raise atomically; overflow guard is loud
# ---------------------------------------------------------------------------
def test_strictness_caller_bugs_and_overflow():
    g = _make_gmap()
    snap = _snapshot(g)

    with pytest.raises(ValueError):
        g.seal_tiles([(6, 6), (6, 6)], MAT_DOOR)          # duplicates
    with pytest.raises(ValueError):
        g.can_seal_tiles([(6, 6), (6, 6)])                # duplicates (query)
    with pytest.raises(ValueError):
        g.seal_tiles([(6, 6), (99, 99)], MAT_DOOR)        # OOB
    with pytest.raises(ValueError):
        g.seal_tiles([(0, 0)], MAT_DOOR)                  # already solid
    with pytest.raises(ValueError):
        g.seal_tiles([(6, 6)], MAT_FURNITURE)             # non-solid material
    with pytest.raises(ValueError):
        g.seal_tiles([(6, 6)], MAT_AIR)                   # non-solid material
    with pytest.raises(ValueError):
        g.unseal_tiles([(6, 6)])                          # not solid
    with pytest.raises(ValueError):
        g.seal_tiles([(6, 6), (0, 0)], MAT_DOOR)          # mixed span: atomic
    _assert_unchanged(g, snap)

    # Overflow pre-check: loud, pre-mutation; can_seal_tiles covers it (S4).
    g.gas[O2][6, 6] = 2 ** 31 - 100
    g.gas[O2][5, 6] = 1000
    snap2 = _snapshot(g)
    assert not g.can_seal_tiles([(6, 6)])
    with pytest.raises(OverflowError):
        g.seal_tiles([(6, 6)], MAT_DOOR)
    _assert_unchanged(g, snap2)

    # can_seal True really means the seal completes (composition contract).
    g.gas[O2][6, 6] = 13763
    assert g.can_seal_tiles([(6, 6)])
    g.seal_tiles([(6, 6)], MAT_DOOR)
    assert bool(g.solid[6, 6])


# ---------------------------------------------------------------------------
# §12.14 — burst-after-seal pin (critique S2), gamemap level
# ---------------------------------------------------------------------------
def test_burst_after_seal_sees_differential():
    g = _make_gmap()
    for y in range(1, 11):                      # wall line x=6, doorway (6,6)
        if y != 6:
            _make_solid(g, y, 6)
    # Rooms at >burst_threshold differential (MAT_DOOR: 2.0 atm).
    g.atmosphere[6, 7] = atmosphere_fixed.quantize_scalar(3.5)
    assert (6, 6) not in g.find_burst_walls()   # open tile: not scanned
    g.seal_tiles([(6, 6)], MAT_DOOR)            # the seal itself SUCCEEDS
    # The next slot-9b scan would destroy the fresh door via the minting
    # destroy_wall path + DoorDestroyedEvent — the A6 rider's factual basis.
    assert (6, 6) in g.find_burst_walls()
    # Hull never bursts (burst_threshold 0.0): the door is the only pop.
    assert all(g.material[t] == MAT_DOOR for t in g.find_burst_walls())


# ---------------------------------------------------------------------------
# §12.15 — in-span vacuum-join chaining pin (critique N2): live-solid,
# chains DOWN the row-major span order, not up it
# ---------------------------------------------------------------------------
def test_inspan_vacuum_join_chains_down_span():
    g = _make_gmap()
    _make_exposed_vacuum(g, 5, 6)               # vacuum touches FIRST tile
    _make_solid(g, 6, 6)
    _make_solid(g, 7, 6)
    donors = ((6, 5), (6, 7), (7, 5), (7, 7), (8, 6))
    donor_vals = {d: [int(g.gas[i][d]) for i in range(N_GASES)] for d in donors}
    g.unseal_tiles([(6, 6), (7, 6)])
    # (6,6) joins the vacuum; (7,6) then sees (6,6) exposed (LIVE solid
    # mask) and chains — one connected breach, no seed anywhere.
    assert bool(g.is_vacuum[6, 6]) and bool(g.is_vacuum[7, 6])
    assert all(int(g.gas[i][6, 6]) == 0 for i in range(N_GASES))
    assert all(int(g.gas[i][7, 6]) == 0 for i in range(N_GASES))
    for d, vals in donor_vals.items():
        assert [int(g.gas[i][d]) for i in range(N_GASES)] == vals


def test_inspan_vacuum_join_does_not_chain_up_span():
    g = _make_gmap()
    _make_exposed_vacuum(g, 8, 6)               # vacuum touches LAST tile
    _make_solid(g, 6, 6)
    _make_solid(g, 7, 6)
    totals0 = _slice_totals(g)
    g.unseal_tiles([(6, 6), (7, 6)])
    # (6,6) is processed first, while (7,6) is still solid: no vacuum in
    # sight -> it seeds from its donors. (7,6) then joins the vacuum. The
    # asymmetry is the PINNED behavior (deterministic, conservative).
    assert not bool(g.is_vacuum[6, 6])
    assert int(g.gas[O2][6, 6]) > 0
    assert bool(g.is_vacuum[7, 6])
    assert all(int(g.gas[i][7, 6]) == 0 for i in range(N_GASES))
    assert _slice_totals(g) == totals0          # joins never mint or destroy


# ---------------------------------------------------------------------------
# Close-T amendment (Erik ruling 4, 2026-07-19; design §4a): on close, the
# door takes the wall assembly's temperature, not the displaced air's
# ---------------------------------------------------------------------------
def _wall_line_with_doorway(g, x=6, doorway_ys=(6,)):
    """A vertical wall at column ``x`` spanning the interior, with open
    doorway tiles at the given rows."""
    for y in range(1, 11):
        if y not in doorway_ys:
            _make_solid(g, y, x)


def test_close_t_hot_room_takes_wall_mean_not_air():
    g = _make_gmap()
    _wall_line_with_doorway(g, x=6, doorway_ys=(6,))
    t_hot = atmosphere_fixed.quantize_scalar(1000.0)   # post-grenade air
    t_n = atmosphere_fixed.quantize_scalar(293.0)
    t_s = atmosphere_fixed.quantize_scalar(311.0)
    g.temperature[6, 6] = t_hot
    g.temperature[5, 6] = t_n                          # N wall
    g.temperature[7, 6] = t_s                          # S wall
    totals0 = _slice_totals(g)

    g.seal_tiles([(6, 6)], MAT_DOOR)
    # Wall-assembly mean (floor), NOT the displaced air's T: no instant
    # hot door from 1000 K doorway air.
    assert int(g.temperature[6, 6]) == (int(t_n) + int(t_s)) // 2
    assert int(g.temperature[6, 6]) != int(t_hot)
    # Wall donors are read-only: the mean is taken, nothing is moved.
    assert int(g.temperature[5, 6]) == int(t_n)
    assert int(g.temperature[7, 6]) == int(t_s)
    # The T write carries no conservation weight — N totals exact.
    assert _slice_totals(g) == totals0

    # unseal_tiles' temperature behavior is UNCHANGED (design §4a): the
    # opened tile's solid T becomes the gas T — unseal writes nothing.
    door_t = int(g.temperature[6, 6])
    g.unseal_tiles([(6, 6)])
    assert int(g.temperature[6, 6]) == door_t


def test_close_t_floor_division_three_walls():
    g = _make_gmap()
    # Three solid neighbors (N, S, W) around (6, 6); E stays the receiver.
    for (y, x) in ((5, 6), (7, 6), (6, 5)):
        _make_solid(g, y, x)
    g.temperature[5, 6] = 7
    g.temperature[7, 6] = 8
    g.temperature[6, 5] = 10
    g.seal_tiles([(6, 6)], MAT_DOOR)
    # Integer mean = floor division: (7 + 8 + 10) // 3 = 8 (not 8.33...).
    assert int(g.temperature[6, 6]) == 8


def test_close_t_no_solid_neighbor_keeps_air_t():
    g = _make_gmap()
    t_hot = atmosphere_fixed.quantize_scalar(1000.0)
    g.temperature[6, 6] = t_hot                 # free-standing seal: all 4
    g.seal_tiles([(6, 6)], MAT_DOOR)            # neighbors are open air
    assert int(g.temperature[6, 6]) == int(t_hot)   # fallback: T stays


def test_close_t_span_members_never_donate():
    g = _make_gmap()
    # Wall at x=6 with a 2-tile doorway (5,6)+(6,6). Pre-existing solid
    # neighbors: (4,6) for the first span tile, (7,6) for the second — the
    # adjacent span member is NOT solid before the call and never donates.
    _wall_line_with_doorway(g, x=6, doorway_ys=(5, 6))
    t_a = atmosphere_fixed.quantize_scalar(280.0)
    t_b = atmosphere_fixed.quantize_scalar(400.0)
    t_hot = atmosphere_fixed.quantize_scalar(900.0)
    g.temperature[4, 6] = t_a
    g.temperature[7, 6] = t_b
    g.temperature[5, 6] = t_hot
    g.temperature[6, 6] = t_hot

    g.seal_tiles([(5, 6), (6, 6)], MAT_DOOR)
    # Each tile means over ITS pre-existing walls only (a single donor
    # each) — no mixing through the other span member, no air T.
    assert int(g.temperature[5, 6]) == int(t_a)
    assert int(g.temperature[6, 6]) == int(t_b)


def test_close_t_determinism_two_maps():
    def build_and_seal(g):
        _wall_line_with_doorway(g, x=6, doorway_ys=(5, 6))
        g.temperature[4, 6] = 1_000_001
        g.temperature[7, 6] = 2_000_003
        g.temperature[5, 6] = 65_536_000
        g.temperature[6, 6] = 65_536_000
        g.seal_tiles([(5, 6), (6, 6)], MAT_DOOR)
    a, b = _make_gmap(), _make_gmap()
    build_and_seal(a)
    build_and_seal(b)
    assert np.array_equal(a.temperature, b.temperature)
    for name, arr in _snapshot(a).items():
        assert np.array_equal(arr, _snapshot(b)[name]), (
            f"close-T determinism: field '{name}' diverged")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
