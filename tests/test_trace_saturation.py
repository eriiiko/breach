"""#88 P2 -- V6: the trace books close, and the counted int32 saturation holds
at every site that adds to a trace cell (CPU; the CUDA sites are
``test_cuda_trace_saturation.py``).

Design: ``docs/smoke_unit_rescale_design_2026-10-20.md`` §5.1 (the identity,
closed with no residual), §5.3 (the counted saturation and its sites), §10 V6;
critique A3 (the conversion never narrows before the saturation), A4 (from above
only, once per cell on the net; a negative is asserted, never clamped), A10 (the
saturation binds in reachable abuse, so the composite is a primary case), B1
(the mint and deposit books, written at the ATTEMPTED amount), B7 (the books'
per-tick window), C10 (every site, per backend).

THE IDENTITY, per tick and per trace plane, exactly in int64, in scenarios
without ACCEPTED GAP G6's unbooked destroyers (pump extraction, FieldEdit
REMOVE, the explosion's smoke clear):

    Σ S(after) − Σ S(before) == mint + deposit
                                − vent − wipe − sink − decay − clip

``mint`` = ``EOSSolver.trace_mint_sum`` (soot, W5); ``deposit`` =
``GameMap.trace_deposit_books`` (FieldEdit); ``clip`` = ``EOSSolver.
trace_clip_sum`` (3b, soot, W5, W3) + ``GameMap.trace_clip_books`` (FieldEdit,
seal_tiles, the vent supply's ``_inject_gas_n_core``).

THE AMOUNTS are written relative to the cell ceiling, so a case binds at any
currency: ``OVER`` is 200 real units at the shipped ``TRACE_FINE_BITS = 8``
(1.5625 × the 128-unit ceiling, A3's case) and the same ratio over the ceiling
at k = 0 (51200 units). Nothing pins a count, a set of gases or a tick count.
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

bp = pytest.importorskip("breach_physics")

from level_loader import LevelData  # noqa: E402
from simulation import Simulation, gas_fixed  # noqa: E402
from simulation.field_edit import (  # noqa: E402
    EditMode, Falloff, FieldEdit, Region, apply_field_edit,
)
from simulation.gases import (  # noqa: E402
    GasTable, INERT_N2, N_GASES, O2, SMOKE, STEAM,
)
from simulation.payloads import ignite_ring  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from config import CFG  # noqa: E402

TMAX = gas_fixed.TRACE_CELL_MAX
T_ONE = gas_fixed.TRACE_ONE
CEIL_UNITS = TMAX / T_ONE                       # the cell ceiling, real units
OVER = 200.0 * 2.0 ** (8 - gas_fixed.TRACE_FINE_BITS)   # 1.5625 × the ceiling
HULL, AIR, FURNITURE = 1, 0, 6                  # v2 codes == material ids


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _box(h, w):
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = HULL
    return tm


def _sim(tm, tile_m=0.333):
    lvl = LevelData(name="trace_sat", version="2", path=Path("."), tilemap=tm,
                    tile_size_m=tile_m, diffuse_path=Path("."))
    sim = Simulation(lvl, seed=7, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _trace_ids(g):
    return [i for i in range(g.gases.n) if not bool(g.gases.conservative[i])]


def _tot(g, gid):
    return int(g.gas[gid].astype(np.int64).sum())


def _net_books(sim, gid):
    """mint + deposit − vent − wipe − sink − decay − clip of the LAST tick."""
    eos = sim.physics_runner.engine.eos
    g = sim.gmap
    gain = eos.trace_mint_sum()[gid] + g.trace_deposit_books[gid]
    loss = (eos.boundary_flux()[gid] + eos.trace_wipe_sum()[gid]
            + eos.trace_sink_sum()[gid] + eos.trace_decay_sum()[gid]
            + eos.trace_clip_sum()[gid] + g.trace_clip_books[gid])
    return gain - loss


def _clip_total(sim, gid):
    return (sim.physics_runner.engine.eos.trace_clip_sum()[gid]
            + sim.gmap.trace_clip_books[gid])


def _assert_tick_closes(sim, before, tag):
    """The §5.1 identity on every trace plane for the tick just stepped, plus
    the saturation's range and the negative counter."""
    g = sim.gmap
    eos = sim.physics_runner.engine.eos
    for gid in _trace_ids(g):
        plane = g.gas[gid]
        assert int(plane.max()) <= TMAX and int(plane.min()) >= 0, (tag, gid)
        assert eos.trace_neg_hits()[gid] == 0, (tag, gid)
        d = _tot(g, gid) - before[gid]
        assert d == _net_books(sim, gid), (
            f"{tag}: gas {gid} Δ={d} books={_net_books(sim, gid)}")
        for book in (eos.trace_mint_sum()[gid], g.trace_deposit_books[gid],
                     eos.trace_clip_sum()[gid], g.trace_clip_books[gid]):
            assert book >= 0, (tag, gid, book)


def _step_checked(sim, tag):
    before = {gid: _tot(sim.gmap, gid) for gid in _trace_ids(sim.gmap)}
    sim.step()
    _assert_tick_closes(sim, before, tag)


def _heat_blast(sim, y, x, amount=3200.0, radius=5.0, sid=990):
    sim.edit_queue.enqueue(FieldEdit(
        field="heat", region=Region.DISC, coords=(y, x, float(radius)),
        amount=float(amount), mode=EditMode.ADD, falloff=Falloff.LINEAR,
        source_id=sid))


# ---------------------------------------------------------------------------
# the kit: the Python twin == the C++ kit
# ---------------------------------------------------------------------------
def test_the_saturation_twins_agree_and_saturate_from_above_only():
    """PROPERTY: ``gas_fixed.trace_add_sat`` and ``fixed_point.h
    trace_add_sat`` (bound as ``fp_trace_add_sat``) return the same stored
    value and the same clip on every input; a sum above TRACE_CELL_MAX stores
    the ceiling and returns EXACTLY the excess; a representable sum is stored
    unchanged (never cut, D1); a NEGATIVE sum is counted by the C++ kit and
    stored unclamped, and raises in Python (A4: clamping would mint mass).

    BREAKS IF: either twin clamps from below, cuts a representable sum, books
    a wrong excess, wraps, or the two drift apart.
    """
    cases = [(0, 0), (0, 1), (0, TMAX), (0, TMAX + 1), (1, TMAX),
             (TMAX, 0), (TMAX, 1), (TMAX - 7, 7), (TMAX - 7, 8),
             (12345, -12345), (12345, -1), (0, 1 << 40), (TMAX, 1 << 62)]
    for s, add in cases:
        stored_c, clip_c, neg_c = bp.fp_trace_add_sat(s, add)
        stored_p, clip_p = gas_fixed.trace_add_sat(s, add)
        t = s + add
        assert (stored_c, clip_c, neg_c) == (stored_p, clip_p, 0), (s, add)
        assert stored_p == min(t, TMAX) and clip_p == max(0, t - TMAX), (s, add)
        if add >= 0:
            assert stored_p >= s, (s, add)            # never cuts the cell
    stored_c, clip_c, neg_c = bp.fp_trace_add_sat(5, -9)
    assert (stored_c, clip_c, neg_c) == (-4, 0, 1)
    with pytest.raises(ValueError):
        gas_fixed.trace_add_sat(5, -9)


# ---------------------------------------------------------------------------
# V6 per site, CPU
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", [EditMode.ADD, EditMode.MAX])
def test_fieldedit_books_its_attempt_and_clips_the_excess(mode):
    """PROPERTY (site: FieldEdit trace ADD / MAX): an edit whose result cannot
    be represented stores TRACE_CELL_MAX, books the ATTEMPTED positive change
    in ``trace_deposit_books`` and EXACTLY the excess in ``trace_clip_books``,
    so ``Δ S == deposit − clip``; an OVER-ceiling deposit into an EMPTY cell
    (A3: 200 units at k = 8) lands at the ceiling too, never wraps or raises.

    BREAKS IF: the combine narrows before the saturation, the attempt is booked
    after it (or not at all), the excess is dropped, or the tile is cut.
    """
    sim = _sim(_box(12, 12))
    g = sim.gmap
    g.reset_trace_books()
    near = TMAX - 1000
    g.gas[SMOKE][4, 4] = near
    # ADD: a quarter-ceiling on a nearly full cell; MAX: a target above it.
    amount = 0.25 * CEIL_UNITS if mode == EditMode.ADD else 1.25 * CEIL_UNITS
    apply_field_edit(g, FieldEdit(field="gas", channel=SMOKE, region=Region.TILE,
                                  coords=(4, 4), amount=amount, mode=mode), sim.rng)
    want = gas_fixed.quantize_trace_scalar(
        (amount if mode == EditMode.MAX else gas_fixed.dequantize_trace(near) + amount))
    attempt = want - near
    assert attempt > 1000
    assert int(g.gas[SMOKE][4, 4]) == TMAX
    assert g.trace_deposit_books[SMOKE] == attempt
    assert g.trace_clip_books[SMOKE] == near + attempt - TMAX

    # A3: an over-ceiling amount into an EMPTY cell.
    g.reset_trace_books()
    apply_field_edit(g, FieldEdit(field="gas", channel=SMOKE, region=Region.TILE,
                                  coords=(6, 6), amount=OVER, mode=mode), sim.rng)
    q = gas_fixed.quantize_trace_scalar(OVER)
    assert q > TMAX
    assert int(g.gas[SMOKE][6, 6]) == TMAX
    assert g.trace_deposit_books[SMOKE] == q
    assert g.trace_clip_books[SMOKE] == q - TMAX


def test_fieldedit_remove_books_nothing_and_a_representable_add_is_exact():
    """PROPERTY: a representable ADD books its whole change as a deposit and
    clips nothing; a REMOVE books nothing at all (ACCEPTED GAP G6, #98).

    BREAKS IF: a REMOVE is booked as a negative deposit (the identity tests
    would then bend to a gap they do not cover), or a fitting ADD is clipped.
    """
    sim = _sim(_box(12, 12))
    g = sim.gmap
    g.reset_trace_books()
    apply_field_edit(g, FieldEdit(field="gas", channel=SMOKE, region=Region.TILE,
                                  coords=(4, 4), amount=0.5, mode=EditMode.ADD), sim.rng)
    assert g.trace_deposit_books[SMOKE] == int(g.gas[SMOKE][4, 4]) > 0
    assert g.trace_clip_books[SMOKE] == 0
    g.reset_trace_books()
    apply_field_edit(g, FieldEdit(field="gas", channel=SMOKE, region=Region.TILE,
                                  coords=(4, 4), amount=0.25, mode=EditMode.REMOVE), sim.rng)
    assert g.trace_deposit_books[SMOKE] == 0 and g.trace_clip_books[SMOKE] == 0


def test_seal_tiles_saturates_a_full_receiver_instead_of_refusing():
    """PROPERTY (site: ``seal_tiles``): sealing a smoky tile beside receivers
    already at the ceiling COMPLETES (no OverflowError: the trace slices take
    the counted saturation inside the raise-free mutation pass), every receiver
    stays ≤ TRACE_CELL_MAX, and what could not land is EXACTLY the clip book
    (a move: no deposit); the bulk slices still move to the count.

    BREAKS IF: the trace overflow check returns to the pre-mutation refusal,
    the redistribution wraps, or the excess is dropped unbooked.
    """
    sim = _sim(_box(10, 10))
    g = sim.gmap
    g.reset_trace_books()
    t = (5, 5)
    g.gas[SMOKE][t] = 5_000_000
    for r in ((4, 5), (6, 5), (5, 4), (5, 6)):
        g.gas[SMOKE][r] = TMAX - 3
    s_before = _tot(g, SMOKE)
    bulk_before = _tot(g, O2) + _tot(g, INERT_N2)
    g.seal_tiles([t], HULL)
    assert int(g.gas[SMOKE].max()) <= TMAX
    assert int(g.gas[SMOKE][t]) == 0
    assert g.trace_deposit_books[SMOKE] == 0
    assert g.trace_clip_books[SMOKE] == 5_000_000 - 4 * 3
    assert _tot(g, SMOKE) - s_before == -g.trace_clip_books[SMOKE]
    assert _tot(g, O2) + _tot(g, INERT_N2) == bulk_before


def test_the_vent_supply_inject_saturates_its_trace_slices():
    """PROPERTY (site: ``_inject_gas_n_core``, reached through
    ``inject_gas_n_vec``, the vent supply): a trace slice that cannot hold the
    supplied amount stores TRACE_CELL_MAX and books EXACTLY the excess in the
    clip book, with no OverflowError; the bulk slices are added exactly.

    BREAKS IF: the inject raises on a trace slice, wraps it, or drops the
    excess unbooked.
    """
    sim = _sim(_box(10, 10))
    g = sim.gmap
    g.reset_trace_books()
    g.gas[SMOKE][4, 4] = TMAX - 10
    add = [0] * N_GASES
    add[SMOKE] = 1_000_000
    add[O2] = 100
    add[INERT_N2] = 300
    o2_0, n2_0 = int(g.gas[O2][4, 4]), int(g.gas[INERT_N2][4, 4])
    g.inject_gas_n_vec(4, 4, add, 0, -(1 << 30), 1 << 30)
    assert int(g.gas[SMOKE][4, 4]) == TMAX
    assert g.trace_clip_books[SMOKE] == 1_000_000 - 10
    assert g.trace_deposit_books[SMOKE] == 0
    assert int(g.gas[O2][4, 4]) == o2_0 + 100
    assert int(g.gas[INERT_N2][4, 4]) == n2_0 + 300


def test_stage_3b_saturates_once_per_cell_under_blast_compression():
    """PROPERTY (site: stage 3b, CPU): a room full to the ceiling, compressed
    by heat blasts, keeps every trace cell in [0, TRACE_CELL_MAX] and closes
    the §5.1 identity every tick with no residual: the inflow a compressed
    cell cannot store is ``EOSSolver.trace_clip_sum`` (non-vacuous: it binds),
    the negative-net counter stays 0.

    BREAKS IF: 3b narrows with a wrap or an unbooked min, applies the
    saturation per face instead of on the net (a donor's outflow then lets a
    full cell overflow silently), or its clip escapes the books' window.
    """
    sim = _sim(_box(20, 28))
    g = sim.gmap
    g.gas[SMOKE][1:-1, 1:-1] = TMAX
    clipped = 0
    for tick in range(24):
        if tick % 6 == 0:
            _heat_blast(sim, 10, 8 + (tick % 12), sid=990 + tick)
        _step_checked(sim, f"3b tick {tick}")
        clipped += sim.physics_runner.engine.eos.trace_clip_sum()[SMOKE]
        assert sim.gmap.trace_clip_books[SMOKE] == 0
    assert clipped > 0, "the scenario never compressed a full cell: vacuous"


def _combustion_scene(soot_q):
    """A 5x5 grid: one lit, hot, fuelled wood source at (2, 2); air around it
    with O2 to burn; the smoke plane at ``soot_q`` on every air cell."""
    h = w = 5
    gas = np.zeros((N_GASES, h, w), dtype=np.int32)
    gas[INERT_N2][:] = gas_fixed.quantize_bulk_scalar(0.79)
    gas[O2][:] = gas_fixed.quantize_bulk_scalar(1.0)
    solid = np.zeros((h, w), dtype=bool)
    solid[2, 2] = True
    flammable = solid.copy()
    is_vacuum = np.zeros((h, w), dtype=bool)
    wall_hp = np.zeros((h, w), dtype=np.int32)
    wall_hp[2, 2] = 60 * 65536
    fire = np.zeros((h, w), dtype=np.int32)
    fire[2, 2] = 65536
    ign = np.zeros((h, w), dtype=np.int32)
    ign[2, 2] = 500 * 65536
    temperature = np.zeros((h, w), dtype=np.int32)
    temperature[2, 2] = 900 * 65536
    gas[SMOKE][~solid] = soot_q
    return dict(gas=gas, temperature=temperature, wall_hp=wall_hp, fire=fire,
                flammable=flammable, solid=solid, is_vacuum=is_vacuum,
                ignition_temp_q16=ign)


def _burn(st):
    comb = bp.CombustionSolver()
    # The dials of tests/cuda_combustion_check.py (one burning source draws
    # O2 and deposits soot; the exact values are not load-bearing).
    comb.burn_rate, comb.o2_thresh_burn = 1.0, 0.03
    comb.H_FUEL_M, comb.H_FUEL_SHIFT = 4.0, 0
    comb.soot_yield, comb.fuel_per_o2 = 0.3, 0.7
    comb.o2_frac_ext, comb.o2_frac_full = 0.13, 0.21
    comb.T_MAX_PHYS = 16000.0
    eos = bp.EOSSolver()
    comb.step(st["gas"], O2, INERT_N2, SMOKE, st["temperature"], st["wall_hp"],
              st["fire"], st["flammable"], st["solid"], st["is_vacuum"],
              st["ignition_temp_q16"], 0.25, 0.0076849, 0.05, eos=eos)
    return eos


def test_the_soot_mint_books_its_attempt_and_clips_the_excess():
    """PROPERTY (site: the soot mint, CPU; V4's mint half): into an empty
    smoke plane the mint book equals the smoke gained, with no clip; into a
    plane at the ceiling the SAME attempt is booked, every cell stays ≤
    TRACE_CELL_MAX and ``Δ S == mint − clip`` exactly; and the bulk planes
    (the Dalton split's N2, O2) are bit-identical between the two, because the
    saturation touches only the trace-currency add.

    BREAKS IF: the mint is booked after the saturation (or not at all), the
    clip is dropped, the narrow wraps, or the saturation leaks into the bulk
    split.
    """
    free = _combustion_scene(0)
    s0 = int(free["gas"][SMOKE].astype(np.int64).sum())
    eos_free = _burn(free)
    mint = eos_free.trace_mint_sum()[SMOKE]
    assert mint > 0, "the source never burned: vacuous"
    assert int(free["gas"][SMOKE].astype(np.int64).sum()) - s0 == mint
    assert eos_free.trace_clip_sum()[SMOKE] == 0

    full = _combustion_scene(TMAX - 1)
    s0 = int(full["gas"][SMOKE].astype(np.int64).sum())
    eos_full = _burn(full)
    assert eos_full.trace_mint_sum()[SMOKE] == mint
    assert int(full["gas"][SMOKE].max()) <= TMAX
    d = int(full["gas"][SMOKE].astype(np.int64).sum()) - s0
    assert d == mint - eos_full.trace_clip_sum()[SMOKE]
    assert eos_full.trace_clip_sum()[SMOKE] > 0
    assert eos_full.trace_neg_hits()[SMOKE] == 0
    for gid in (O2, INERT_N2):
        assert np.array_equal(free["gas"][gid], full["gas"][gid])


def _water_scene(h=3, w=3):
    solid = np.ones((h, w), dtype=bool)
    solid[1, 1] = False
    return dict(
        water_depth=np.zeros((h, w), dtype=np.int32),
        atmosphere=np.full((h, w), 65536, dtype=np.int32),
        solid=solid,
        gas=np.zeros((N_GASES, h, w), dtype=np.int32),
        before=np.zeros((h, w), dtype=np.int32),
        dyn_permeability=np.where(solid, 0.0, 1.0).astype(np.float32),
    )


def _water_tail(engine, st, steam_yield, ceiling_h=2.0):
    cons = np.asarray(GasTable.from_config(CFG).conservative, dtype=bool)
    h, w = st["solid"].shape
    engine.step_water_tail(
        st["water_depth"], st["atmosphere"], st["solid"], st["gas"],
        st["before"], st["dyn_permeability"], STEAM, 1.0,
        ceiling_h, 1e-3, 4.0, 1.0, 0.3, steam_yield,
        gas_energy=np.zeros((h, w), dtype=np.int64),
        gas_conservative=cons,
        thermal_solid=st["solid"].copy(),
        is_vacuum=np.zeros((h, w), dtype=bool),
        is_ambient=np.zeros((h, w), dtype=bool),
        t_amb_raw=0)


def test_an_over_ceiling_w5_puff_lands_at_the_ceiling_and_books_the_rest():
    """PROPERTY (site: the W5 steam puff; A3's conversion case): a puff of
    OVER units (200 at k = 8) into an EMPTY cell is quantized in int64, booked
    whole in ``trace_mint_sum``, and narrowed by the saturation: the cell
    holds TRACE_CELL_MAX and ``trace_clip_sum`` is EXACTLY the excess.

    BREAKS IF: the puff is narrowed (or wrapped) before the saturation, the
    mint is booked after it, or the excess is dropped.
    """
    engine = bp.PhysicsEngine()
    st = _water_scene()
    st["water_depth"][1, 1] = 65536          # 1.0 m, boiled off whole in 1 s
    st["before"][1, 1] = 65536
    st["atmosphere"][1, 1] = 0               # below boil_p_thresh
    _water_tail(engine, st, steam_yield=OVER)
    puff_q = bp.fp_quantize_trace(OVER)
    assert puff_q > TMAX
    assert int(st["gas"][STEAM][1, 1]) == TMAX
    assert engine.eos.trace_mint_sum()[STEAM] == puff_q
    assert engine.eos.trace_clip_sum()[STEAM] == puff_q - TMAX
    assert engine.eos.trace_neg_hits()[STEAM] == 0


def test_the_w3_receiver_saturates_and_the_donor_loses_its_whole_share():
    """PROPERTY (site: the W3 evacuation's receiver add): a flooding cell
    evacuating trace into a receiver at the ceiling leaves the receiver at
    TRACE_CELL_MAX, books EXACTLY what could not land in ``trace_clip_sum``,
    and the plane total falls by exactly that clip (the donor's debit is the
    whole evacuated share).

    BREAKS IF: the receiver add wraps or is clamped unbooked, or the donor is
    debited only what landed (the clip would then be invented mass).
    """
    engine = bp.PhysicsEngine()
    st = _water_scene(3, 4)
    st["solid"][1, 2] = False                # the one open receiver
    st["dyn_permeability"][1, 2] = 1.0
    st["water_depth"][1, 1] = 65536          # 1.0 m rising from 0: ratio 2
    st["gas"][SMOKE][1, 1] = 1 << 30
    st["gas"][SMOKE][1, 2] = TMAX - 5
    s0 = int(st["gas"][SMOKE].astype(np.int64).sum())
    _water_tail(engine, st, steam_yield=0.0)
    clip = engine.eos.trace_clip_sum()[SMOKE]
    assert int(st["gas"][SMOKE][1, 2]) == TMAX
    assert clip > 0
    assert int(st["gas"][SMOKE].astype(np.int64).sum()) - s0 == -clip
    assert engine.eos.trace_neg_hits()[SMOKE] == 0


# ---------------------------------------------------------------------------
# V6: the identity on live scenarios
# ---------------------------------------------------------------------------
def test_the_composite_abuse_grenade_spam_under_blast_compression_closes():
    """PROPERTY (A10, the primary case): smoke "grenades" spammed every tick
    into a room already near the ceiling, while heat blasts compress it, keep
    every trace cell in [0, TRACE_CELL_MAX], the negative counter at 0, and
    the §5.1 identity exact every tick with NO residual -- with the
    saturation binding at BOTH a Python site (FieldEdit) and an engine site
    (3b), so the books are proven to see each.

    BREAKS IF: any site wraps, raises, narrows before the saturation, clamps a
    negative, or drops an excess unbooked; or a book falls outside the window
    sim.step() brackets.
    """
    sim = _sim(_box(22, 30))
    g = sim.gmap
    g.gas[SMOKE][1:-1, 1:-1] = int(0.9 * TMAX)
    fe_clip = eng_clip = 0
    for tick in range(30):
        sim.edit_queue.enqueue(FieldEdit(
            field="gas", channel=SMOKE, region=Region.DISC,
            coords=(11, 6 + (tick % 18), 3.0), amount=0.3 * CEIL_UNITS,
            mode=EditMode.ADD, falloff=Falloff.LINEAR, source_id=500 + tick))
        if tick % 5 == 0:
            _heat_blast(sim, 11, 15, sid=990 + tick)
        _step_checked(sim, f"composite tick {tick}")
        fe_clip += g.trace_clip_books[SMOKE]
        eng_clip += sim.physics_runner.engine.eos.trace_clip_sum()[SMOKE]
    assert fe_clip > 0, "no FieldEdit ever saturated: vacuous"
    assert eng_clip > 0, "stage 3b never saturated: vacuous"


def test_a_soot_minting_fire_closes_the_books_with_no_residual():
    """PROPERTY (V6 identity, B1): a real crate fire, in a sealed room, mints
    soot every tick; ``Δ Σ S == mint + deposit − vent − wipe − sink − decay −
    clip`` holds exactly every tick on every trace plane -- the mint is read
    from its book, not inferred from the residual.

    BREAKS IF: the soot mint goes unbooked, is booked into the wrong gas or
    the wrong window (combustion runs after the EOS step), or the identity
    needs a residual again.
    """
    tm = _box(16, 20)
    tm[6:10, 8:12] = FURNITURE
    sim = _sim(tm)
    ignite_ring(sim.gmap, sim.edit_queue, 8, 10, 2.5, 1.0)
    minted = 0
    for tick in range(48):
        _step_checked(sim, f"fire tick {tick}")
        minted += sim.physics_runner.engine.eos.trace_mint_sum()[SMOKE]
    assert minted > 0, "the fire never minted soot: vacuous"
