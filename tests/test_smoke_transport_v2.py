"""Smoke transport v2 (#12, docs/smoke_transport_design_2026-10-04.md §7):
the trace gases ride the air.

The five trace planes (steam, smoke, poison, teargas, fuel_gas) move on the
bulk planes' APPLIED face flux, priced at the donor's trace-per-air ratio
(``bulk_transport.cpp`` stage 3b), lose what no air is left to carry (3c),
and once a tick get stranded zeroing, a conservative magnitude-truncated
Jacobi diffusion and ceil-rounded decay (``trace_tail``). Every loss is
booked: vent in the trace slots of ``EOSSolver.boundary_flux()``, then
``trace_wipe_sum`` / ``trace_sink_sum`` / ``trace_decay_sum``.

The V-numbers are the design's (§7). V6 (CPU == CUDA) is the CUDA patches'.
Every test names the PROPERTY it protects and the change that must BREAK it.
Nothing here pins a count of gases, a tick count of a scenario, or a feel
number; feel-adjacent readings are made in dequantized units.
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

from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import gas_fixed, unit_fixed  # noqa: E402
from simulation.exchange import apply_poison_dose, apply_teargas_blind  # noqa: E402
from simulation.field_edit import (  # noqa: E402
    EditMode, EditQueue, Falloff, FieldEdit, Region,
)
from simulation.gases import (  # noqa: E402
    GasTable, POISON, SMOKE, TEARGAS, check_trace_diffusion_stable,
    trace_diffusion_dd_q,
)
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.status import BLINDED  # noqa: E402
from simulation.unit import Unit  # noqa: E402

Q = 65536
HULL, AIR, FURNITURE, SPACE = 1, 0, 6, 9     # v2 codes == material ids


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _box(h, w, fill=AIR):
    """A sealed hull box of open floor."""
    tm = np.full((h, w), fill, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = HULL
    return tm


def _sim(tm, tile_m=0.333):
    lvl = LevelData(name="smoke_v2", version="2", path=Path("."), tilemap=tm,
                    tile_size_m=tile_m, diffuse_path=Path("."))
    sim = Simulation(lvl, seed=7, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)        # a Simulation starts paused
    return sim


def _trace_ids(g):
    return [i for i in range(g.gases.n) if not bool(g.gases.conservative[i])]


def _bulk_ids(g):
    return [i for i in range(g.gases.n) if bool(g.gases.conservative[i])]


def _tot(g, gid):
    return int(g.gas[gid].astype(np.int64).sum())


def _books(sim, gid):
    """(vent, wipe, sink, decay) of the LAST tick, raw counts."""
    eos = sim.physics_runner.engine.eos
    return (eos.boundary_flux()[gid], eos.trace_wipe_sum()[gid],
            eos.trace_sink_sum()[gid], eos.trace_decay_sum()[gid])


def _heat_blast(sim, y, x, amount=3200.0, radius=5.0):
    """A heat-only blast: one DISC heat deposit through the sim's own queue
    (the payload executor's deposit_heat shape). No soot, no pressure edit:
    the air expands because it was heated."""
    sim.edit_queue.enqueue(FieldEdit(
        field="heat", region=Region.DISC, coords=(y, x, float(radius)),
        amount=float(amount), mode=EditMode.ADD, falloff=Falloff.LINEAR,
        source_id=990))


# ---------------------------------------------------------------------------
# V1 — exact conservation
# ---------------------------------------------------------------------------
def test_v1_a_sealed_blast_conserves_the_trace_to_the_count():
    """PROPERTY (V1): in a sealed room with decay off, ``Σ S + wipe`` is
    constant TO THE COUNT across full Simulation.steps while a heat-only
    blast drives the air hard (multi-substep EOS ticks, metres-per-second
    wind) — and a tile seeded at 2.0 (above the old [0, 1] ceiling) keeps
    its smoke instead of being cut to 1.

    BREAKS IF: any stage floors asymmetrically (a face debited and credited
    different amounts), a face is skipped on one side, the diffusion floors,
    or any clamp (the SL step's, the fire step's, a deposit's) returns.
    """
    sim = _sim(_box(24, 32))
    g = sim.gmap
    g.gases.decay[:] = 0.0
    g.gas[SMOKE][12, 20] = 2 * Q
    _heat_blast(sim, 12, 10)
    total0 = _tot(g, SMOKE)
    wipe = 0
    max_wind = 0
    max_nsub = 0
    for _ in range(72):
        sim.step()
        wipe += _books(sim, SMOKE)[1]
        assert _tot(g, SMOKE) + wipe == total0
        max_wind = max(max_wind, int(np.abs(g.wind_x).max()), int(np.abs(g.wind_y).max()))
        max_nsub = max(max_nsub, sim.physics_runner.engine.eos.dbg_last_n_sub)
    # Non-vacuous: the blast really moved the air, and the seed really moved.
    assert max_wind > 2 * Q, "the blast never drove a wind of 2 m/s"
    assert max_nsub > 1, "the EOS never needed more than one substep"
    assert int(g.gas[SMOKE][12, 20]) < 2 * Q
    assert int(np.count_nonzero(g.gas[SMOKE])) > 20


def test_v1_a_tile_above_one_survives_a_burning_fire_step():
    """PROPERTY (V1, the fire clamp): the fire step no longer touches a
    trace plane — a tile holding 2.0 of smoke keeps more than 1.0 across
    ticks in which the fire step runs (a fire lit elsewhere in the room).

    BREAKS IF: the fire step's smoke [0, 1] clamp returns (it cut every
    compressed tile back to 1, every tick, silently undoing conservation).
    """
    sim = _sim(_box(16, 24))
    g = sim.gmap
    g.gases.decay[:] = 0.0
    g.gas[SMOKE][8, 18] = 2 * Q
    # The fire only has to EXIST so the fire step does not early-out; it is
    # not an igniter (no gameplay reads it), so a bare fire value is enough.
    g.fire[8, 4] = Q // 2
    sim.step()
    assert int(g.fire.max()) > 0, "the fire step must have run with a fire"
    assert int(g.gas[SMOKE].max()) > Q, \
        "a compressed tile was cut back to 1 (a [0, 1] clamp is back)"


# ---------------------------------------------------------------------------
# V2 — no expansion mint
# ---------------------------------------------------------------------------
def test_v2_expanding_air_thins_its_smoke_instead_of_multiplying_it():
    """PROPERTY (V2): with the trace at a uniform ratio to the air, S = r·N,
    and diffusion off, a heat-only blast leaves every cell's ratio within
    ``faces · substeps / N_min`` of r: the trace is priced BY AIR, so a
    parcel that expands carries the same smoke per unit of air. The bound
    is the floordiv truncation, at most one count per face per substep,
    mixed convexly with the neighbours' ratios.

    BREAKS IF: the trace is moved as a per-tile value (the H1 mint — the
    old semi-Lagrangian scheme kept a tile's value as its air expanded,
    and the studio's 8.9 deposited units became 100).
    """
    sim = _sim(_box(20, 28))
    g = sim.gmap
    eos = sim.physics_runner.engine.eos
    bulk = _bulk_ids(g)
    for t in _trace_ids(g):
        g.gases.diffusion[t] = 0.0
    g.gases.decay[:] = 0.0
    open_ = ~g.solid
    n_bulk = sum(g.gas[b].astype(np.int64) for b in bulk)
    g.gas[SMOKE][open_] = (n_bulk[open_] >> 2).astype(np.int32)   # r = 1/4
    r = 0.25
    _heat_blast(sim, 10, 8)
    sub_total = 0
    n_min = int(n_bulk[open_].min())
    max_dev = 0.0
    for _ in range(48):
        sim.step()
        sub_total += eos.dbg_last_n_sub
        n_bulk = sum(g.gas[b].astype(np.int64) for b in bulk)
        n_min = min(n_min, int(n_bulk[open_].min()))
        ratio = g.gas[SMOKE][open_].astype(np.float64) / n_bulk[open_]
        dev = float(np.abs(ratio - r).max())
        max_dev = max(max_dev, dev)
        # initial rounding (N mod 4)/4 counts, then 4 faces per substep
        bound = (4 * sub_total + 1) / n_min
        assert dev <= bound, (dev, bound, sub_total, n_min)
    assert sub_total > 48, "the blast never needed a second substep"


# ---------------------------------------------------------------------------
# V3 — venting closes the books
# ---------------------------------------------------------------------------
def test_v3_seal_then_breach_closes_the_identity_every_tick():
    """PROPERTY (V3): for every trace plane and every tick, sealed and then
    breached to space, ``Σ S(after) − Σ S(before) == −(vent + wipe + sink +
    decay)`` EXACTLY in int64 (there are no deposits in this room, so the
    transport's books must account for every count). Smoke is seeded at
    2.0 near the hull and trace is also written onto a vacuum tile after
    the breach (the stranded case), so vent, sink and decay are all live.

    BREAKS IF: a vent, diffusion, decay or sink path destroys trace without
    booking it, books a different amount than it removes, or a vacuum / ring
    cell is left holding trace.
    """
    tm = np.full((22, 30), SPACE, dtype=np.int32)
    tm[2:20, 2:28] = HULL
    tm[3:19, 3:27] = AIR
    sim = _sim(tm)
    g = sim.gmap
    traces = _trace_ids(g)
    g.gas[SMOKE][8:12, 20:25] = 2 * Q
    g.gas[POISON][5:8, 5:8] = Q // 3
    seen = {k: 0 for k in ("vent", "sink", "decay")}

    def tick():
        before = {t: _tot(g, t) for t in traces}
        sim.step()
        for t in traces:
            vent, wipe, sink, decay = _books(sim, t)
            assert _tot(g, t) - before[t] == -(vent + wipe + sink + decay), t
            assert vent >= 0 and wipe >= 0 and sink >= 0 and decay >= 0
            seen["vent"] += vent
            seen["sink"] += sink
            seen["decay"] += decay

    for _ in range(12):               # sealed
        tick()
    assert seen["vent"] == 0, "a sealed room vented trace"
    g.destroy_wall(10, 27)            # breach the east hull to space
    for _ in range(36):
        tick()
    vac = np.argwhere(g.is_vacuum & ~g.solid)
    y, x = (int(v) for v in vac[0])
    g.gas[SMOKE][y, x] += Q // 4       # stranded: trace on a vacuum tile
    for _ in range(4):
        tick()
    assert seen["vent"] > 0, "the breach never vented any trace"
    assert seen["sink"] >= Q // 4, "stranded trace was not sunk and booked"
    assert seen["decay"] > 0
    assert not g.gas[SMOKE][g.is_vacuum].any()


# ---------------------------------------------------------------------------
# V4 — non-negativity
# ---------------------------------------------------------------------------
def test_v4_no_trace_cell_is_ever_negative_across_the_blasts():
    """PROPERTY (V4): across a real blast scenario (the smoke-light studio's
    three charges, their soot and their winds) and a heat blast through a
    room of crates, no trace cell of any plane is ever negative.

    BREAKS IF: a donor is over-priced (a ceil where the price floors, a
    limiter bypassed), or the diffusion floors toward −inf instead of
    truncating the magnitude.
    """
    from level_loader import load as load_level
    lvl = load_level("smoke_light_studio", levels_dir=str(ROOT / "levels"))
    sim = Simulation(lvl, seed=42, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    traces = _trace_ids(sim.gmap)
    for _ in range(int(3.5 * sim._tps)):
        sim.step()
        for t in traces:
            assert int(sim.gmap.gas[t].min()) >= 0, t
    assert _tot(sim.gmap, SMOKE) > 0, "the studio made no smoke"

    tm = _box(18, 26)
    tm[4:14, 15] = FURNITURE
    sim = _sim(tm)
    g = sim.gmap
    g.gas[SMOKE][3:15, 3:12] = Q // 2
    g.gas[TEARGAS][3:15, 17:24] = Q // 5
    _heat_blast(sim, 9, 7, radius=4.0)
    for _ in range(48):
        sim.step()
        for t in traces:
            assert int(g.gas[t].min()) >= 0, t


# ---------------------------------------------------------------------------
# V5 — the air does not see the smoke
# ---------------------------------------------------------------------------
def test_v5_the_air_is_bit_identical_with_or_without_trace():
    """PROPERTY (V5): bulk N, ``gas_energy``, wind and temperature are
    bit-identical whether the trace planes are populated or empty, through
    a heat blast in a room with crate (thermal-solid) tiles in the flow —
    the trace rides the air and never pushes back. The radiative smoke term
    (a READER of smoke, not transport) is switched off in both runs so this
    isolates transport.

    BREAKS IF: trace leaks into bulk membership, into p*, or into the energy
    books (e.g. the crate tiles' extended pre-flux N reaching stage 3's
    energy pricing).
    """
    def run(populate):
        tm = _box(18, 26)
        tm[3:15, 16] = FURNITURE
        sim = _sim(tm)
        g = sim.gmap
        g.gases.heat_absorb_q16[:] = 0
        if populate:
            open_ = ~g.solid
            for k, t in enumerate(_trace_ids(g)):
                g.gas[t][open_] = (k + 1) * (Q // 7)
        _heat_blast(sim, 9, 6, radius=4.0)
        for _ in range(36):
            sim.step()
        return g, sim

    ga, sa = run(False)
    gb, sb = run(True)
    for b in _bulk_ids(ga):
        assert np.array_equal(ga.gas[b], gb.gas[b]), f"bulk plane {b}"
    assert np.array_equal(ga.gas_energy, gb.gas_energy)
    assert np.array_equal(ga.wind_x, gb.wind_x)
    assert np.array_equal(ga.wind_y, gb.wind_y)
    assert np.array_equal(ga.temperature, gb.temperature)
    # Non-vacuous: the trace did ride (a uniform seed is no longer uniform),
    # and the crate pores hold trace.
    vals = gb.gas[SMOKE][~gb.solid]
    assert int(vals.min()) != int(vals.max())
    assert int(gb.gas[SMOKE][3:15, 16].min()) > 0


def test_v5_trace_rides_through_crate_pores():
    """PROPERTY (D3): a crate (thermal-solid, permeable) tile carries trace
    in its pores as it carries air — smoke on one side of a full-height
    crate wall reaches the far side by riding the flow through it.

    BREAKS IF: thermal-solid tiles stop participating in the trace ride
    (the stage-3b predicate narrowed to the energy one).
    """
    tm = _box(12, 24)
    tm[1:-1, 12] = FURNITURE
    sim = _sim(tm)
    g = sim.gmap
    for t in _trace_ids(g):
        g.gases.diffusion[t] = 0.0       # the ride alone, no diffusion
    g.gases.heat_absorb_q16[:] = 0
    g.gas[SMOKE][1:-1, 1:12] = Q // 2
    _heat_blast(sim, 6, 4, radius=3.0)
    for _ in range(36):
        sim.step()
    assert int(g.gas[SMOKE][1:-1, 13:-1].sum()) > 0, \
        "no smoke crossed the crate wall"


# ---------------------------------------------------------------------------
# V7 — diffusion conserves, spreads, and does not drift
# ---------------------------------------------------------------------------
def test_v7_still_air_diffusion_conserves_spreads_and_stays_symmetric():
    """PROPERTY (V7): in still air (decay off) the diffusion alone keeps
    ``Σ S`` exact, grows a blob's second moment, and — seeded symmetric in a
    symmetric room — keeps the field EXACTLY mirror-symmetric (no drift in
    any direction). A uniform thin field stays exactly uniform.

    BREAKS IF: diffusion is removed (no spread), loses mass (the old
    ``mul_q16(coeff, lap)`` floor lost ~½ count per cell per tick), or
    floors toward −inf (which pumps thin smoke north-west and breaks the
    mirror symmetry).
    """
    sim = _sim(_box(17, 19))
    g = sim.gmap
    g.gases.heat_absorb_q16[:] = 0        # still air stays still
    g.gases.decay[:] = 0.0
    g.gas[SMOKE][7:10, 8:11] = 3 * Q
    g.gas[POISON][1:-1, 1:-1] = 37        # a uniform thin field (raw counts)
    total0 = _tot(g, SMOKE)
    yy, xx = np.mgrid[0:17, 0:19]

    def m2(S):
        w = S.astype(np.float64)
        return float((w * ((yy - 8) ** 2 + (xx - 9) ** 2)).sum() / w.sum())

    m2_0 = m2(g.gas[SMOKE])
    for _ in range(48):
        sim.step()
        assert _tot(g, SMOKE) == total0
    assert int(np.abs(g.wind_x).max()) == 0 and int(np.abs(g.wind_y).max()) == 0
    S = g.gas[SMOKE][1:-1, 1:-1]
    assert np.array_equal(S, S[:, ::-1]) and np.array_equal(S, S[::-1, :])
    assert m2(g.gas[SMOKE]) > 1.5 * m2_0, "the blob did not spread"
    P = g.gas[POISON][1:-1, 1:-1]
    assert int(P.min()) == int(P.max()) == 37


# ---------------------------------------------------------------------------
# V8 — decay reaches zero
# ---------------------------------------------------------------------------
def test_v8_thin_poison_decays_to_zero_in_bounded_time():
    """PROPERTY (V8): a thin uniform poison field (0.06 — below the old
    decay floor, where ``mul_q16(v, frac)`` rounded every tick's loss to 0,
    yet above the lethal ``poison_min_density``) decays to exactly 0 within
    ``v`` ticks (ceil rounding removes at least one count a tick), every
    lost count booked as decay.

    BREAKS IF: the decay floor returns (the field would sit at 0.06 forever:
    permanent haze, here permanent lethal poison).
    """
    sim = _sim(_box(6, 6))
    g = sim.gmap
    v = gas_fixed.quantize_scalar(0.06)
    assert v >= gas_fixed.quantize_scalar(float(CFG.exchange.poison_min_density))
    g.gas[POISON][1:-1, 1:-1] = v
    total0 = _tot(g, POISON)
    booked = 0
    for k in range(v + 1):
        if not g.gas[POISON].any():
            break
        sim.step()
        booked += _books(sim, POISON)[3]
    assert not g.gas[POISON].any(), "thin poison never decayed to zero"
    assert booked == total0


# ---------------------------------------------------------------------------
# V9 — readers at density 2
# ---------------------------------------------------------------------------
def test_v9_poison_dose_and_teargas_blind_are_defined_at_density_two():
    """PROPERTY (V9): conserved trace can exceed 1 in a compressed pocket;
    the gameplay readers stay well defined there — the poison dose is
    linear (density 2 doses twice density 1, to the HP grid) and teargas
    BLINDED fires at 2 as at any density above its threshold.

    BREAKS IF: a reader starts assuming density <= 1 (a clamp, a lookup
    table indexed by density, a saturating curve).
    """
    tps = int(CFG.clock.ticks_per_second)
    dose = {}
    for density in (1.0, 2.0):
        tm = _box(24, 24)
        lvl = LevelData(name="v9", version="2", path=Path("."), tilemap=tm,
                        tile_size_m=1.0, diffuse_path=Path("."))
        from simulation.gamemap import GameMap
        g = GameMap(lvl)
        u = Unit("M", x=8, y=8, team=0)
        u.id = 1
        u.current_hp = 1000.0
        q = gas_fixed.quantize_scalar(density)
        for (tx, ty) in u.occupied_tiles():
            g.gas[POISON][ty, tx] = q
            g.gas[TEARGAS][ty, tx] = q
        apply_poison_dose([u], g, tps, events=[])
        dose[density] = 1000.0 - u.current_hp
        apply_teargas_blind([u], g)
        assert any(s.kind == BLINDED for s in u.statuses), density
    assert dose[2.0] == unit_fixed.quantize_hp_delta(2.0 * dose[1.0])
    assert dose[2.0] > dose[1.0] > 0.0


# ---------------------------------------------------------------------------
# D1 — deposits are additive
# ---------------------------------------------------------------------------
def test_d1_a_trace_deposit_adds_exactly_and_never_cuts_the_tile():
    """PROPERTY (D1, Erik's ruling 2026-10-04): an ADD edit onto a trace
    plane adds exactly its amount — a tile already above 1 keeps all of it
    plus the deposit — and a REMOVE edit bottoms out at 0.

    BREAKS IF: the FieldEdit trace policy (or a caller's explicit clamp)
    goes back to clamping the whole tile to [0, 1].
    """
    tm = _box(10, 10)
    lvl = LevelData(name="d1", version="2", path=Path("."), tilemap=tm,
                    tile_size_m=1.0, diffuse_path=Path("."))
    from simulation.gamemap import GameMap
    g = GameMap(lvl)
    g.gas[SMOKE][5, 5] = int(1.7 * Q)
    q = EditQueue()
    q.enqueue(FieldEdit(field="gas", channel=SMOKE, region=Region.TILE,
                        coords=(5, 5), amount=0.75, mode=EditMode.ADD))
    q.enqueue(FieldEdit(field="smoke", region=Region.TILE, coords=(4, 4),
                        amount=1.5, mode=EditMode.ADD))
    q.enqueue(FieldEdit(field="smoke", region=Region.TILE, coords=(3, 3),
                        amount=5.0, mode=EditMode.REMOVE))
    g.gas[SMOKE][3, 3] = Q // 3
    q.flush(g, np.random.default_rng(0))
    assert int(g.gas[SMOKE][5, 5]) == int(1.7 * Q) + gas_fixed.quantize_scalar(0.75)
    assert int(g.gas[SMOKE][4, 4]) == gas_fixed.quantize_scalar(1.5)
    assert int(g.gas[SMOKE][3, 3]) == 0


# ---------------------------------------------------------------------------
# The stability door
# ---------------------------------------------------------------------------
def test_the_diffusion_stability_door_refuses_an_unstable_dial():
    """PROPERTY (design §2.2): the trace diffusion's integer stability
    condition ``4 · quantize(d · dt) <= ONE`` is refused at the gases door
    and again where PhysicsRunner binds dt, naming the gas; the shipped
    dials pass at the shipped clock; the check's dd_q is the engine's fold.

    BREAKS IF: either door stops checking, or the check drifts from the
    engine's own quantize (float32 dial x float32 dt, round-half-away).
    """
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    GasTable.from_config()                          # shipped dials pass
    assert trace_diffusion_dd_q(0.1, dt) == int(np.floor(
        float(np.float32(0.1)) * float(np.float32(dt)) * Q + 0.5))
    names = ["smoke", "o2"]
    with pytest.raises(ValueError, match="smoke"):
        check_trace_diffusion_stable(names, [0.25 / dt * 1.01, 0.0],
                                     [False, True], dt, "test")
    # a bulk plane is not this law's business
    check_trace_diffusion_stable(names, [0.1, 1e9], [False, True], dt, "test")
    with pytest.raises(ValueError, match="diffusion"):
        GasTable(CFG.gases, getattr(CFG, "smoke", None), tick_dt=10.0)
    # the dt-binding re-check: a tick long enough to destabilise the
    # shipped smoke dial is refused by the runner, by name
    sim = _sim(_box(6, 6))
    with pytest.raises(ValueError, match="PhysicsRunner"):
        sim.physics_runner.step(sim.gmap, 10.0)
