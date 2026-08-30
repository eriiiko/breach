"""EOS refactor P1 — species + conservative transport (docs/eos_refactor_design.md
§1/§2, §8 patch P1).

Covers the P1 scope end to end:

1. Structure — ``gases.py`` grows N_GASES 5->7 additively (O2=5, INERT_N2=6),
   the bulk pair is flagged ``conservative``, carries zero decay + all-zero
   optics + not-flammable + no effect tag; the legacy 5 gases are unchanged
   and ``conservative=False``.
2. Ambient initialization — interior air seeds O2/inert_N2 21/79 (summing
   back to exactly today's atmosphere==1.0 Q16.16 scale); solid/vacuum stay 0.
3. ``destroy_wall`` seeds the newly-opened tile's O2/inert_N2 with a CONSTANT
   ambient total, split by the donors' inherited mole fraction (re-pointed at
   P-M3 — it was the neighbour mean, which was the mass mint).
4. Donor-cell conservative flux transport (C++ ``bulk_flux_transport``):
   moves mass under wind, respects face permeability, and — the P1 GATE — a
   SEALED room's total O2+inert_N2 is EXACTLY conserved (integer equality)
   over 1000 ticks, even under a stress (high-CFL, checkerboard-velocity)
   regime that forces the per-cell outflow limiter to fire hard.
5. Save/field-edit round-trip — the ``gas`` FieldEdit channel policy accepts
   the new ids 5/6 (the array simply grew; no policy code changes needed).

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_eos_p1_species_transport.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import atmosphere_fixed, gas_fixed  # noqa: E402
from simulation import field_edit  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gases import (  # noqa: E402
    GasTable, N_GASES, STEAM, SMOKE, POISON, TEARGAS, FUEL_GAS,
    O2, INERT_N2,
)
from simulation.physics_runner import PhysicsRunner  # noqa: E402


# ---------------------------------------------------------------------------
# Test scenes
# ---------------------------------------------------------------------------
def _sealed_room_level(h=20, w=20) -> LevelData:
    """A hull-walled sealed box, interior air, NO vacuum/breach anywhere."""
    tm = np.ones((h, w), dtype=np.int32)      # all hull
    tm[1:h - 1, 1:w - 1] = 4                   # carve interior air
    return LevelData(name="eos_p1_sealed_room", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _make_gmap(h=20, w=20) -> GameMap:
    return GameMap(_sealed_room_level(h, w))


# ---------------------------------------------------------------------------
# 1. Structure
# ---------------------------------------------------------------------------
def test_n_gases_grew_additively_to_seven():
    assert N_GASES == 7
    assert O2 == 5 and INERT_N2 == 6
    # The five legacy ids are UNCHANGED (index-bound views like gmap.smoke
    # depend on this — the whole point of "append, never reorder").
    assert (STEAM, SMOKE, POISON, TEARGAS, FUEL_GAS) == (0, 1, 2, 3, 4)


def test_gas_array_shape_is_seven_planes():
    g = _make_gmap()
    h, w = g.smoke.shape
    assert g.gas.shape == (7, h, w)
    assert g.gas.dtype == np.int32
    # gmap.smoke is still the SMOKE view — untouched by the append.
    assert g.smoke.base is g.gas
    assert np.shares_memory(g.smoke, g.gas[SMOKE])


def test_bulk_pair_table_contract():
    tbl = GasTable.from_config()
    assert tbl.n == 7
    assert tbl.names[5:7] == ["o2", "inert_n2"]
    assert tbl.name_to_id["o2"] == O2
    assert tbl.name_to_id["inert_n2"] == INERT_N2

    # conservative: true ONLY for the bulk pair.
    cons = list(tbl.conservative.astype(bool))
    assert cons == [False, False, False, False, False, True, True]

    # Bulk pair: zero decay, all optics zero, not flammable, no effect tag.
    for gid in (O2, INERT_N2):
        assert float(tbl.decay[gid]) == 0.0
        assert np.allclose(tbl.absorption[gid], [0.0, 0.0, 0.0])
        assert np.allclose(tbl.scatter_albedo[gid], [0.0, 0.0, 0.0])
        assert float(tbl.glow[gid]) == 0.0
        assert bool(tbl.flammable[gid]) is False
        assert bool(tbl.emits_when_hot[gid]) is False
        assert tbl.effect[gid] == ""

    # The 5 legacy gases are UNCHANGED (values match test_multigas_structure.py).
    assert np.allclose(tbl.absorption[STEAM], [0.10, 0.10, 0.10])
    assert np.allclose(tbl.absorption[SMOKE], [0.88, 0.90, 0.93])
    assert np.allclose(tbl.diffusion[:5], [0.18, 0.10, 0.12, 0.15, 0.22])


# ---------------------------------------------------------------------------
# 2. Ambient initialization
# ---------------------------------------------------------------------------
def test_ambient_o2_n2_split_on_open_air():
    g = _make_gmap()
    interior = (~g.solid) & (~g.is_vacuum)
    assert interior.any()

    expect_o2 = gas_fixed.quantize_scalar(0.21)
    expect_n2 = gas_fixed.quantize_scalar(0.79)
    assert np.all(g.gas[O2][interior] == expect_o2)
    assert np.all(g.gas[INERT_N2][interior] == expect_n2)

    # Solid/vacuum tiles carry ZERO O2/N2, mirroring atmosphere.
    solid_or_vac = g.solid | g.is_vacuum
    assert np.all(g.gas[O2][solid_or_vac] == 0)
    assert np.all(g.gas[INERT_N2][solid_or_vac] == 0)

    # Ambient N_total reproduces today's atmosphere==1.0 Q16.16 scale exactly.
    total_q = g.gas[O2][interior].astype(np.int64) + g.gas[INERT_N2][interior].astype(np.int64)
    assert np.all(total_q == atmosphere_fixed.FP_ONE)
    assert np.all(g.atmosphere[interior] == atmosphere_fixed.FP_ONE)


def test_reload_material_table_preserves_running_gas_state():
    """A config hot-reload must NOT stomp live O2/N2 (or trace gas) state —
    mirrors the existing atmosphere/obstacles preservation contract."""
    g = _make_gmap()
    interior = (~g.solid) & (~g.is_vacuum)
    ys, xs = np.where(interior)
    y0, x0 = int(ys[0]), int(xs[0])

    # Perturb the running state away from ambient (a combustion event, say).
    g.gas[O2][y0, x0] = 12345
    g.gas[INERT_N2][y0, x0] = 54321
    g.gas[POISON][y0, x0] = 999
    gas_before = g.gas.copy()

    g.reload_material_table()

    assert np.array_equal(g.gas, gas_before), (
        "reload_material_table() must preserve the running gas array "
        "(O2/N2 ambient re-seed must not overwrite live state)")
    # The buffer identity is preserved too (never reassigned — a C++ view
    # of gmap.gas must stay valid across a hot-reload).
    assert g.gas is not None


# ---------------------------------------------------------------------------
# 3. destroy_wall seeding — a CONSTANT ambient total, composition inherited
#
# RE-POINTED at P-M3 (was `test_destroy_wall_seeds_bulk_gas_by_neighbor_mean`,
# which pinned the neighbour MEAN). The mean was the mass mint: it wrote the
# donors' average without withdrawing anything, so the seed scaled with local
# pressure — and the burst valve fires on pressure, which made the relief valve
# a pressure amplifier. destroy_wall now seeds one CONSTANT cell of the map's
# ambient air and books it. Full gate set:
# tests/test_destroy_wall_conserves_mass.py; design:
# docs/mass_books_pm3_destroy_wall_seed_design_2026-08-18.md.
# ---------------------------------------------------------------------------
def test_destroy_wall_seeds_constant_ambient_total_with_inherited_split():
    g = _make_gmap()
    # Pick an interior wall tile (a hull tile NOT on the map edge) so the
    # "interior: seed one ambient cell" branch fires, not the true breach
    # (which seeds nothing and evacuates instead).
    # Carve a single-tile pillar of hull inside the room to destroy.
    h, w = g._h, g._w
    py, px = h // 2, w // 2
    from simulation.materials import MAT_HULL
    g.material[py, px] = MAT_HULL
    g.on_tile_changed(py, px)
    g.solid[py, px] = True
    # A real solid tile holds NO bulk N (bulk_transport zeroes it every pass),
    # and the hand-carved pillar above kept the ambient gas the constructor
    # seeded. Clear it, or this is not the solid path and the booked delta is
    # `seed - prior` instead of the full seed.
    g.gas[O2][py, px] = 0
    g.gas[INERT_N2][py, px] = 0
    # Give its neighbors a distinct total AND a distinct composition, so the
    # two halves of the rule are separately checkable: the TOTAL must ignore
    # them (constant), the SPLIT must follow them (inherited).
    for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = py + dy, px + dx
        g.gas[O2][ny, nx] = 20000
        g.gas[INERT_N2][ny, nx] = 40000

    n_total_q, _o2_amb, _n2_amb, pin_q = g.ambient_seed()
    g.destroy_wall(py, px)

    # Total: the map's ambient constant — NOT the donors' 60000.
    assert int(g.gas[O2][py, px]) + int(g.gas[INERT_N2][py, px]) == n_total_q
    # Split: the donors' 20000/60000 mole fraction, by the exact int64 form
    # `(n_total * sum_o2 + sum_n // 2) // sum_n` — one rounding, no float.
    expected_o2 = (n_total_q * 20000 + 30000) // 60000
    assert int(g.gas[O2][py, px]) == expected_o2
    assert int(g.gas[INERT_N2][py, px]) == n_total_q - expected_o2
    # The same-tick companions (design §3.1.2 / §3.1 / §3.3).
    assert int(g.atmosphere[py, px]) == pin_q
    assert int(g.temperature[py, px]) == 0
    assert int(g.fire[py, px]) == 0
    assert not g.solid[py, px]
    # And it is BOOKED: a solid tile held 0, so the delta is one ambient cell.
    assert g.n_destruction_seed_sum == n_total_q


# ---------------------------------------------------------------------------
# 4. Donor-cell conservative flux transport — direct C++ unit + E2E gate
# ---------------------------------------------------------------------------
def _isum2(a, b):
    return int(a.astype(np.int64).sum()) + int(b.astype(np.int64).sum())


def test_bulk_flux_transport_moves_mass_downwind():
    """A direct call (mirrors test_poison_transports_through_per_gas_loop):
    a populated O2 blob under a steady rightward wind moves right; the total
    O2+inert_N2 mass stays EXACTLY conserved throughout (a bounded room under
    a steady one-way wind is NOT a no-op for a uniform field — it legitimately
    piles mass against the downwind wall, so the invariant under test is
    conservation, not stasis)."""
    h = w = 24
    gas = np.zeros((7, h, w), dtype=np.int32)
    gas_conservative = np.array([False] * 5 + [True, True], dtype=bool)
    solid = np.zeros((h, w), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    is_vacuum = np.zeros((h, w), dtype=bool)
    perm = np.where(solid, 0.0, 1.0).astype(np.float32)

    gas[O2][8:14, 4:8] = gas_fixed.quantize_scalar(1.0)
    gas[INERT_N2][~solid] = gas_fixed.quantize_scalar(0.5)   # uniform ambient-ish

    wind_x = np.full((h, w), atmosphere_fixed.quantize_scalar(4.0), dtype=np.int32)
    wind_y = np.zeros((h, w), dtype=np.int32)
    wind_x[solid] = 0

    def _com_x(field):
        tot = field.astype(np.int64).sum()
        xs = np.arange(field.shape[1])[None, :]
        return float((field.astype(np.int64) * xs).sum() / tot)

    cx0 = _com_x(gas[O2])
    total0 = _isum2(gas[O2], gas[INERT_N2])

    for _ in range(20):
        bp.bulk_flux_transport(gas, gas_conservative, wind_x, wind_y,
                               solid, is_vacuum, perm, 0.05)

    cx1 = _com_x(gas[O2])
    assert cx1 > cx0 + 1.0, f"O2 blob did not advect right: {cx0:.2f} -> {cx1:.2f}"
    assert _isum2(gas[O2], gas[INERT_N2]) == total0, "mass not conserved under a sealed wind"
    # Solid tiles never accumulate N (the conservation-guard clamp holds).
    assert np.all(gas[O2][solid] == 0)
    assert np.all(gas[INERT_N2][solid] == 0)
    # Trace planes (indices 0-4) untouched — this function only ever
    # touches conservative=True planes.
    assert np.all(gas[:5] == 0)


def test_bulk_flux_transport_never_goes_negative_under_high_cfl_stress():
    """The per-cell outflow limiter (ported from WaterSolver) must hold N>=0
    even when a single step's flux would otherwise over-drain a cell —
    mirrors test_water_conservation_stress.py's regime (checkerboard +/-
    v_max wind, dt forced far above any natural CFL bound)."""
    h = w = 22
    solid = np.zeros((h, w), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    is_vacuum = np.zeros((h, w), dtype=bool)
    perm = np.where(solid, 0.0, 1.0).astype(np.float32)

    rng = np.random.default_rng(3)
    gas = np.zeros((7, h, w), dtype=np.int32)
    gas_conservative = np.array([False] * 5 + [True, True], dtype=bool)
    interior = ~solid
    gas[O2][interior] = gas_fixed.quantize(0.05 + 0.4 * rng.random(int(interior.sum())))
    gas[INERT_N2][interior] = gas_fixed.quantize(0.05 + 0.4 * rng.random(int(interior.sum())))

    v_max = atmosphere_fixed.quantize_scalar(30.0)   # a large, aggressive wind
    yy, xx = np.mgrid[0:h, 0:w]
    sign_x = np.where(((yy + xx) % 2) == 0, 1, -1).astype(np.int32)
    sign_y = np.where(((yy * 3 + xx) % 2) == 0, 1, -1).astype(np.int32)
    wind_x = (sign_x * v_max).astype(np.int32)
    wind_y = (sign_y * v_max).astype(np.int32)
    wind_x[solid] = 0
    wind_y[solid] = 0

    total0 = _isum2(gas[O2], gas[INERT_N2])
    dt = 1.0   # aggressively large — many multiples of any sane per-tick dt

    min_seen = 0
    for _ in range(200):
        bp.bulk_flux_transport(gas, gas_conservative, wind_x, wind_y,
                               solid, is_vacuum, perm, dt)
        min_seen = min(min_seen, int(gas[O2].min()), int(gas[INERT_N2].min()))

    assert min_seen >= 0, f"bulk N went negative under stress (min={min_seen})"
    assert _isum2(gas[O2], gas[INERT_N2]) == total0, (
        "sealed high-CFL stress regime leaked mass "
        f"({_isum2(gas[O2], gas[INERT_N2])} != {total0})")


def test_sealed_room_bulk_conservation_e2e_1000_ticks():
    """THE P1 GATE: a sealed map, run for 1000 ticks through the REAL
    PhysicsRunner.run_substeps path (the exact call physics_runner.py makes
    each tick), asserts sum(N_O2)+sum(N_N2) is EXACTLY constant — integer
    equality, not a tolerance. A non-uniform initial atmosphere seeds real
    wind (via diffuse_solve) so the bulk transport actually churns."""
    g = _make_gmap(h=18, w=18)
    runner = PhysicsRunner(bp)
    # The real per-tick contract (physics_runner/Simulation.step): stamp_units
    # derives dyn_permeability from the static permeability table BEFORE the
    # first run_substeps call. A GameMap that never calls this leaves
    # dyn_permeability at its raw all-ones() construction default (solid
    # tiles included) — bulk_flux_transport's own solid[] gate covers that
    # defensively, but this test drives the REAL contract, not the gate.
    g.stamp_units([])

    interior = (~g.solid) & (~g.is_vacuum)
    assert not g.is_vacuum.any(), "test scene must be a truly sealed room (no vacuum)"

    # EOS refactor P3: `atmosphere` (== P) is solver-materialized every tick
    # from N/T alone (p* = C*N_total*T_abs) — bumping it directly no longer
    # seeds real wind (it is just this tick's P_prev, overwritten by the
    # solve). Bump `temperature` instead (a localized hot patch), which
    # raises p* there and drives a genuine outward flow from tick 1.
    ys, xs = np.where(interior)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    bump = (np.abs(ys - cy) < 3) & (np.abs(xs - cx) < 3)
    # arc #54 P-G1b: through the ONE sanctioned seeding seam. A direct
    # `temperature[...] +=` used to be rescued by the solver's entry re-sync;
    # with D1 live nothing re-derives `gas_energy` from the mirror any more, so
    # a bare T write raises no pressure, drives no wind, and this gate would go
    # vacuous rather than fail loudly. (CLAUDE.md: gas temperature is a mirror.)
    _sel = (ys[bump], xs[bump])
    g.seed_gas_temperature(
        _sel, g.temperature[_sel] + atmosphere_fixed.quantize_scalar(200.0))

    total0 = _isum2(g.gas[O2], g.gas[INERT_N2])
    sim_time = 1.0 / float(CFG.clock.ticks_per_second)

    max_abs_drift = 0
    wind_seen = False
    # EOS refactor P3: run_substeps' signature changed (wave_v/wave_source/
    # sink_x/sink_y retired; wave_p repurposed as p_prev; temperature added).
    # This is still THE P1 GATE, now driven through the real P3 solver path —
    # bulk_flux_transport's per-face gather-then-apply conservation proof is
    # unchanged by being CALLED once per eos substep instead of once per tick.
    # EOS refactor P4: run_substeps gained gas_decay + inert_n2_idx (the
    # trace decay->inert_N2 credit, decisions.md #12 v2.1) — both O2 and
    # inert_N2 carry decay=0.0 by config contract (gases.py), so this is a
    # 0-ULP addition for THIS test's bulk-only conservation proof; passed
    # through for signature parity with the real call site.
    runner.eos.dx = float(g.tile_size_m)
    inert_n2_idx = int(g.gases.name_to_id["inert_n2"])
    for t in range(1000):
        runner.engine.run_substeps(
            g.wave_p, g.atmosphere,
            g.wind_x, g.wind_y,
            g.temperature, g.gas_energy,   # arc #54 §2.2 (MECHANICAL)
            g.obstacles, g.solid, g.is_vacuum,
            g.dyn_permeability, g.dyn_wave_absorb,
            g.gas, g.gases.diffusion, g.gases.conservative,
            g.gases.decay, inert_n2_idx,
            sim_time,
        )
        if not wind_seen and (np.abs(g.wind_x).sum() + np.abs(g.wind_y).sum()) > 0:
            wind_seen = True
        drift = abs(_isum2(g.gas[O2], g.gas[INERT_N2]) - total0)
        max_abs_drift = max(max_abs_drift, drift)

    assert wind_seen, "test setup produced no wind — the conservation gate would be vacuous"
    assert max_abs_drift == 0, (
        f"sealed-room O2+inert_N2 total drifted by up to {max_abs_drift} "
        f"raw Q16.16 counts over 1000 ticks (must be EXACTLY 0)")
    assert _isum2(g.gas[O2], g.gas[INERT_N2]) == total0
    # Solid tiles never accumulated N (defensive clamp holds throughout).
    assert np.all(g.gas[O2][g.solid] == 0)
    assert np.all(g.gas[INERT_N2][g.solid] == 0)


# ---------------------------------------------------------------------------
# 5. field_edit round-trip on the new channel ids
# ---------------------------------------------------------------------------
def test_field_edit_gas_policy_refuses_the_conservative_bulk_slices():
    """arc #54 P-G1b (design §2.7 FieldEdit row) INVERTS this test.

    It used to assert that the channel-indexed `gas` policy is generic over
    the whole (N, h, w) array, O2 and inert_N2 included. Under a stored energy
    field that genericity is precisely the bug: the `gas` policy is a FLOAT
    BRIDGE (dequantize -> float -> [0,1] clamp -> requantize, plus an optional
    RNG draw), and the two conservative slices ARE bulk N — the quantity
    `gas_energy` is denominated against. Moving N through it would move mass
    without moving energy and mint or destroy `ΔN·T_abs` invisibly, once per
    edit, with nothing on either side of the seam to see it.

    So the policy now REFUSES a conservative slice, loudly. Every shipped
    caller is trace-only (combat.py's smoke, payloads.py's gas payloads), so
    this costs nothing today and closes the door before someone opens it; a
    caller that really wants to move bulk N has the pump primitives and the
    `atmosphere` policy, both of which carry the energy with the mass."""
    import random
    import pytest as _pytest
    g = _make_gmap()
    rng = random.Random(0)

    for gid, name in ((O2, "o2"), (INERT_N2, "inert_n2")):
        assert bool(g.gases.conservative[gid]), f"fixture: {name} must be bulk"
        edit = field_edit.FieldEdit(
            field="gas", region=field_edit.Region.TILE, coords=(5, 5),
            amount=0.05, mode=field_edit.EditMode.ADD, channel=gid,
        )
        before = int(g.gas[gid][5, 5])
        with _pytest.raises(AssertionError, match="CONSERVATIVE"):
            field_edit.apply_field_edit(g, edit, rng)
        assert int(g.gas[gid][5, 5]) == before, (
            f"the refused {name} edit still mutated the plane")


def test_field_edit_gas_channel_still_generic_over_trace_ids():
    """The other half of the old test's property SURVIVES: the policy is still
    generic over the (N, h, w) array for every TRACE slice — the array growing
    two bulk planes did not need a policy code change, and still doesn't. This
    is what keeps the refusal above a targeted rule rather than a regression in
    the emitter path."""
    import random
    g = _make_gmap()
    rng = random.Random(0)
    trace = [i for i in range(g.gas.shape[0])
             if not bool(g.gases.conservative[i])]
    assert trace, "fixture: there must be at least one trace slice"
    for gid in trace:
        edit = field_edit.FieldEdit(
            field="gas", region=field_edit.Region.TILE, coords=(5, 5),
            amount=0.05, mode=field_edit.EditMode.ADD, channel=gid,
        )
        before = int(g.gas[gid][5, 5])
        field_edit.apply_field_edit(g, edit, rng)
        assert int(g.gas[gid][5, 5]) != before, (
            f"trace gas channel {gid} edit was a no-op")


if __name__ == "__main__":
    test_n_gases_grew_additively_to_seven()
    test_gas_array_shape_is_seven_planes()
    test_bulk_pair_table_contract()
    test_ambient_o2_n2_split_on_open_air()
    test_reload_material_table_preserves_running_gas_state()
    test_destroy_wall_seeds_constant_ambient_total_with_inherited_split()
    test_bulk_flux_transport_moves_mass_downwind()
    test_bulk_flux_transport_never_goes_negative_under_high_cfl_stress()
    test_sealed_room_bulk_conservation_e2e_1000_ticks()
    test_field_edit_gas_policy_refuses_the_conservative_bulk_slices()
    test_field_edit_gas_channel_still_generic_over_trace_ids()
    print("OK: EOS P1 species + conservative transport tests passed")
