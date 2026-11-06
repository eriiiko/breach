"""The standard explosion (#31, docs/explosion_design_2026-11-06.md section 4): a
payload row with ``strength`` is one blast, expansion, heat and smoke derived from
[blast] and landed as fixed totals on tiles walked along open floor. Each test
names the property it protects and the change that must break it. No test reads a
strength or a [blast] value as a target; those are Erik's placeholders. A test may
fire a shipped row, because what it asserts holds whatever the row's number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")

from config import CFG  # noqa: E402
from field_ab_harness import capture_snapshot  # noqa: E402
from field_digest import field_digest, tick_digest  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation, gas_fixed  # noqa: E402
from simulation.blast_expansion import (S_MAX_Q, heat_units_per_mj,  # noqa: E402
                                        open_mask, reach_tiles, take_source_plane)
from simulation.damage import CREDIT_HEAT, tick_credit  # noqa: E402
from simulation.field_edit import EditQueue  # noqa: E402
from simulation.gases import INERT_N2, O2, SMOKE, smoke_units_of_soot  # noqa: E402
from simulation.materials import (MAT_AIR, MAT_DOOR_CLOSED,  # noqa: E402
                                  MAT_FURNITURE, MAT_HULL)
from simulation.payloads import execute_payload  # noqa: E402
from simulation.status import KNOCKED_DOWN  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import PayloadTable, get_tables  # noqa: E402
from test_heat_source import _closure_bad, _step  # noqa: E402

TILE = 0.333
ROW = dict(radius=2, wall_damage=0, unit_damage=0, explosive="comp_b",
           emit_blast_smoke=True, clear_smoke=True)


def _level(tm):
    return LevelData(name="explosion", version="2", path=Path("."), tilemap=tm,
                     tile_size_m=TILE, diffuse_path=Path("."), entities=[],
                     wires=[])


def _sim(tm, seed=3):
    return Simulation(_level(tm), seed=seed, breach_physics=bp,
                      enable_recorder=False)


def _room(h=30, w=30):
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    return tm


def _corridor(n=30):
    """A one-tile-wide corridor along row 3, sealed at both ends."""
    tm = np.full((7, n), MAT_HULL, dtype=np.int32)
    tm[3, 1:n - 1] = MAT_AIR
    return tm


def _row(strength, **over):
    return PayloadTable({"t": dict(ROW, strength=strength, **over)}).by_name["t"]


def _shipped(name):
    return get_tables().payloads.by_name[name]


def _fire(s, y, x, row, **kw):
    execute_payload(s.gmap, s.edit_queue, s.units, y, x, row, s.rng, **kw)


def _flushed(s):
    """Land the queue on the fields, with no physics after it."""
    s.edit_queue.flush(s.gmap, s.rng)


def _bulk(g):
    return int(g.gas[O2].astype(np.int64).sum() + g.gas[INERT_N2].astype(np.int64).sum())


def _heat_total(g):
    return int(g.heat.astype(np.int64).sum())


def _smoke_total(g):
    return float(gas_fixed.dequantize_trace(int(g.gas[SMOKE].astype(np.int64).sum())))


# --- delivery ---------------------------------------------------------------

@pytest.mark.parametrize("name", ["frag_standard", "demolition_c4"])
def test_a_generic_blast_adds_no_gas(name):
    """PROPERTY: a generic explosion moves air and creates none: in a sealed room
    the bulk N total is exact to the bit through the blast.

    BREAKS IF: a strength row regains `pressure`, or anything else that mints
    bulk N, the failure the source exists to replace.
    """
    s = _sim(_room())
    _step(s)
    n0 = _bulk(s.gmap)
    _fire(s, 15, 15, _shipped(name))
    for _ in range(12):
        _step(s)
    assert _bulk(s.gmap) == n0


def test_the_books_close_through_a_generic_blast():
    """PROPERTY: the closure identity is exact in int64 on every tick of a generic
    blast, walls destroyed included.

    BREAKS IF: the deposit reaches energy outside the seam, or the source reaches
    the expansion-work term.
    """
    s = _sim(_room())

    def blast(i, sim):
        if i == 2:
            _fire(sim, 15, 15, _shipped("frag_standard"))
            _fire(sim, 2, 15, _shipped("demolition_c4"))

    bad = _closure_bad(s, 12, before_step=blast)
    assert not bad, f"closure broken on (step, residual) {bad[:5]}"


def _divided():
    """A room split by a wall on column 15, the charge on the west side."""
    tm = _room()
    tm[:, 15] = MAT_HULL
    return tm


def test_a_wall_stops_every_delivery_and_a_gap_passes_them():
    """PROPERTY: heat, smoke, source and the air read all land only on tiles
    reached along open floor: nothing crosses a wall a disc would have crossed,
    and a gap in it passes all of them.

    BREAKS IF: any of the four falls back to a disc.
    """
    row = _row(9.0)           # r_h 6: well past the wall
    s = _sim(_divided())
    g = s.gmap
    g.gas[O2][:, 16:] //= 4   # thin O2 behind the wall: a disc would read it
    g.gas[INERT_N2][:, 16:] //= 4
    _fire(s, 15, 14, row)
    source = g.div_source.copy()
    _flushed(s)
    behind = (slice(None), slice(16, None))
    assert source[behind].sum() == 0 and g.heat[behind].sum() == 0
    assert g.gas[SMOKE][behind].sum() == 0
    assert source.sum() > 0 and g.heat.sum() > 0 and g.gas[SMOKE].sum() > 0

    tm = _divided()
    tm[15, 15] = MAT_AIR
    s = _sim(tm)
    g = s.gmap
    _fire(s, 15, 14, row)
    source = g.div_source.copy()
    _flushed(s)
    assert source[behind].sum() > 0 and g.heat[behind].sum() > 0
    assert g.gas[SMOKE][behind].sum() > 0


def test_a_charge_on_a_wall_delivers_on_its_approach_side():
    """PROPERTY: a charge whose tile is still solid after its wall damage delivers
    on the side it came from, and nothing lands behind the wall.

    BREAKS IF: the walk returns nothing for a solid centre (a round on a wall
    would be silent), or seeds both sides.
    """
    row = _row(1.0)           # no wall damage: the tile holds
    for approach, west in (((15, 12), True), ((15, 18), False)):
        s = _sim(_divided())
        g = s.gmap
        _fire(s, 15, 15, row, approach=approach)
        source = g.div_source.copy()
        _flushed(s)
        near = (slice(None), slice(0, 15) if west else slice(16, None))
        far = (slice(None), slice(16, None) if west else slice(0, 15))
        assert source[near].sum() > 0 and g.heat[near].sum() > 0
        assert source[far].sum() == 0 and g.heat[far].sum() == 0


def test_a_charge_with_no_side_delivers_on_every_open_side():
    """PROPERTY: with no known approach (a timed charge authored onto a wall) the
    walk seeds every open 4-neighbour, the total unchanged.

    BREAKS IF: an unknown side loses the blast.
    """
    s = _sim(_divided())
    g = s.gmap
    _fire(s, 15, 15, _row(1.0))
    _flushed(s)
    assert g.heat[:, :15].sum() > 0 and g.heat[:, 16:].sum() > 0


def test_a_blast_on_a_door_it_opens_delivers_on_both_sides():
    """PROPERTY: the walk runs after the wall damage, so a tile the blast opened
    is open and the blast reaches both rooms through it.

    BREAKS IF: the walk runs before `apply_explosion`.
    """
    tm = _divided()
    tm[15, 15] = MAT_DOOR_CLOSED
    s = _sim(tm)
    g = s.gmap
    _fire(s, 15, 15, _shipped("breach_focus"), approach=(15, 13))
    _flushed(s)
    assert not g.solid[15, 15], "the door held (the scenario is moot)"
    assert g.heat[:, :15].sum() > 0 and g.heat[:, 16:].sum() > 0


@pytest.mark.parametrize("strength", [1.0, 9.0])
def test_the_heat_total_is_the_same_in_a_room_and_a_corridor(strength):
    """PROPERTY: the enqueued heat is the row's fixed total wherever it goes off,
    to within half an LSB per tile, before the rail: a corridor concentrates it,
    it does not lose it. It also lands the rows's own MJ.

    BREAKS IF: the deposit stops normalising by the tiles the walk reached.
    """
    row = _row(strength)
    totals = []
    for tm, y, x in ((_room(), 15, 15), (_corridor(), 3, 15)):
        s = _sim(tm)
        _fire(s, y, x, row)
        _flushed(s)
        got = _heat_total(s.gmap)
        n = int(np.count_nonzero(s.gmap.heat))
        want = row.heat_mj * heat_units_per_mj(s.gmap) * 65536.0
        assert abs(got - want) <= n / 2 + 1, (got, want, n)
        totals.append(got)
    assert totals[0] != 0


def test_a_corridor_heats_each_tile_hotter_than_a_room():
    """PROPERTY: fixed total means a blast in a corridor is hotter per tile than the
    same blast in the open.

    BREAKS IF: the deposit is per tile, not a share of a total.
    """
    row = _row(1.0)
    peaks = []
    for tm, y, x in ((_room(), 15, 15), (_corridor(), 3, 15)):
        s = _sim(tm)
        _fire(s, y, x, row)
        _flushed(s)
        peaks.append(int(s.gmap.heat.max()))
    assert peaks[1] > peaks[0]


def test_the_smoke_mean_is_the_same_in_a_room_and_a_corridor():
    """PROPERTY: smoke is the charge's soot in expectation over seeds, room or
    corridor, the noise kept.

    BREAKS IF: the mean of the noise is left out of the deposit, or the smoke
    goes back to its own disc.
    """
    row = _row(4.0)
    want = smoke_units_of_soot(
        float(CFG.blast_air.smoke_scale) * row.blast_soot_g)
    for tm, y, x in ((_room(), 15, 15), (_corridor(), 3, 15)):
        means = []
        for seed in range(30):
            s = _sim(tm, seed=seed)
            before = _smoke_total(s.gmap)
            _fire(s, y, x, row)
            _flushed(s)
            means.append(_smoke_total(s.gmap) - before)
        assert abs(np.mean(means) - want) < 0.08 * want, (np.mean(means), want)


def test_the_smoke_noise_is_drawn_at_the_flush_in_row_major_order():
    """PROPERTY: the same seed gives the same cloud, and the executor itself draws
    nothing: the flush stays the one RNG consumer.

    BREAKS IF: the deposit draws at enqueue.
    """
    clouds = []
    for _ in range(2):
        s = _sim(_room(), seed=11)
        state = s.rng.bit_generator.state
        _fire(s, 15, 15, _row(1.0))
        assert s.rng.bit_generator.state == state
        _flushed(s)
        clouds.append(s.gmap.gas[SMOKE].copy())
    assert np.array_equal(clouds[0], clouds[1])


def test_double_strength_doubles_heat_and_smoke():
    """PROPERTY: strength is a linear dial on the delivered totals.

    BREAKS IF: a total stops being `per_strength * strength`.
    """
    (s1, s2) = (_sim(_room(), seed=5), _sim(_room(), seed=5))
    a, b = _row(9.0), _row(18.0)
    for s, row in ((s1, a), (s2, b)):
        _fire(s, 15, 15, row)
        _flushed(s)
    assert abs(_heat_total(s2.gmap) / _heat_total(s1.gmap) - 2.0) < 1e-3
    assert abs(b.blast_soot_g / a.blast_soot_g - 2.0) < 1e-9
    assert abs(_smoke_total(s2.gmap) / _smoke_total(s1.gmap) - 2.0) < 0.4


def _wind_at(y, x, row, ticks=20):
    s = _sim(_room(30, 60))
    _step(s)
    _fire(s, 15, 12, row)
    peak = 0.0
    for _ in range(ticks):
        _step(s)
        peak = max(peak, float(np.hypot(int(s.gmap.wind_x[y, x]),
                                        int(s.gmap.wind_y[y, x]))) / 65536.0)
    return peak


def test_wind_at_a_fixed_range_grows_with_strength():
    """PROPERTY: beyond the rail, at 8 tiles, more strength is more wind, so the
    number stays a real dial. (The peak near the blast rails; this does not.)

    BREAKS IF: the source saturates at open floor, or stops reaching the solve.
    """
    assert _wind_at(15, 20, _row(2.0)) > 1.3 * _wind_at(15, 20, _row(1.0))


def test_strength_under_a_step_stays_on_one_tile():
    """PROPERTY: the source's reach moves in integer steps, so a small change of
    strength never moves S 5-fold.

    BREAKS IF: the reach radius follows strength continuously.
    """
    s = _sim(_room())
    _fire(s, 15, 15, _row(1.0))
    a = s.gmap.div_source.copy()
    s = _sim(_room())
    _fire(s, 15, 15, _row(3.9))
    b = s.gmap.div_source.copy()
    assert np.count_nonzero(a) == np.count_nonzero(b)
    assert b.max() == pytest.approx(3.9 * a.max(), rel=1e-3)
    s = _sim(_room())
    _fire(s, 15, 15, _row(4.1))
    assert s.gmap.div_source.max() < b.max()


def test_a_source_past_s_max_saturates_counted_and_never_wraps():
    """PROPERTY: strength 12 in a one-tile pocket is a source no int32 Q16 holds;
    it saturates at S_MAX, the cut is counted, stacking a second blast stays at
    S_MAX, and the solve runs on it.

    BREAKS IF: the source is added in int32 (a crash, or a wrap to a sink), or the
    cut goes uncounted.
    """
    tm = np.full((9, 9), MAT_HULL, dtype=np.int32)
    tm[4, 4] = MAT_AIR
    s = _sim(tm)
    g = s.gmap
    row = _row(12.0)
    _fire(s, 4, 4, row)
    assert int(g.div_source[4, 4]) == S_MAX_Q
    assert g.div_source_sat_hits == 1 and g.div_source_cut_q > 0
    _fire(s, 4, 4, row)
    assert int(g.div_source[4, 4]) == S_MAX_Q
    assert g.div_source_sat_hits == 2
    _step(s)
    assert not g.div_source.any()


def test_the_source_lasts_one_tick():
    """PROPERTY: a strength row's source is consumed by one solve.

    BREAKS IF: `take_source_plane` stops zeroing the plane, so a blast pushes
    forever.
    """
    s = _sim(_room())
    _fire(s, 15, 15, _row(1.0))
    assert s.gmap.div_source.any()
    plane = take_source_plane(s.gmap)
    assert plane is not None and plane.dtype == np.int32
    assert take_source_plane(s.gmap) is None


def test_the_walk_never_enters_the_ambient_ring_or_vacuum():
    """PROPERTY: the walk stands on the solve's own mask, so no share of a total
    lands where the solve ignores it, and a passable fuel tile stays in it.

    BREAKS IF: the mask goes back to `solid` alone.
    """
    tm = _room()
    tm[15, 16] = MAT_FURNITURE
    s = _sim(tm)
    g = s.gmap
    mask = open_mask(g)
    assert mask[15, 16], "passable fuel left the walk"
    g.is_vacuum[15, 14] = True
    g.is_ambient[14, 15] = True
    mask = open_mask(g)
    tiles = reach_tiles(mask, 15, 15, 25)
    assert (15, 14) not in tiles and (14, 15) not in tiles
    assert (15, 16) in tiles


def test_the_walk_is_row_major():
    """PROPERTY: the tile list is sorted before any edit, so the noise draw order
    does not depend on the walk's order.

    BREAKS IF: `reach_tiles` returns its discovery order.
    """
    s = _sim(_room())
    tiles = reach_tiles(open_mask(s.gmap), 15, 15, 16)
    assert tiles == sorted(tiles)


def test_the_heat_claim_covers_the_walked_tiles():
    """PROPERTY: `deposit_heat`'s credit survives: the owner claims exactly the
    tiles the blast's heat lands on, so a unit burned there is credited to it.

    BREAKS IF: the claim stays on a disc, crediting tiles behind a wall.
    """
    s = _sim(_divided())
    g = s.gmap
    _fire(s, 15, 14, _row(1.0), owner_id=7)
    credit = tick_credit(g)
    assert credit.attacker_at(CREDIT_HEAT, 15, 13) == 7
    assert credit.attacker_at(CREDIT_HEAT, 15, 16) == -1


def test_a_row_without_strength_writes_no_source():
    """PROPERTY: a legacy row behaves as it did: no source on the plane.

    BREAKS IF: the generic path takes over rows it does not own.
    """
    s = _sim(_room())
    _fire(s, 15, 15, _shipped("frag_hot_mass"))
    assert not s.gmap.div_source.any()


# --- digest -----------------------------------------------------------------

def test_a_digest_covers_a_pending_source():
    """PROPERTY: two worlds that differ only in a pending source hash differently,
    in both digests: a slot-9e blast's source is synced state between ticks.

    BREAKS IF: `div_source` leaves SIM_FIELDS or the digest's section.
    """
    s = _sim(_room())
    clean = capture_snapshot(s)
    a = (field_digest(clean), tick_digest(clean))
    s.gmap.div_source[15, 15] = 1000
    dirty = capture_snapshot(s)
    assert field_digest(dirty) != a[0] and tick_digest(dirty) != a[1]


def test_a_zero_source_hashes_nothing():
    """PROPERTY: the section is absence-transparent: a snapshot with an all-zero
    plane hashes exactly what a snapshot without the plane does, which is what
    spares every golden without a blast.

    BREAKS IF: the plane joins DIGEST_FIELDS unconditionally.
    """
    s = _sim(_room())
    snap = capture_snapshot(s)
    without = {k: v for k, v in snap.items() if k != "div_source"}
    assert field_digest(snap) == field_digest(without)


def test_a_replay_of_a_match_with_grenades_clears_every_checkpoint():
    """PROPERTY: the generic blast is deterministic through re-simulation: a bot
    match that throws grenades replays to every checkpoint and to its end.

    BREAKS IF: the walk or the deposit depends on anything outside the seeded,
    synced state.
    """
    from dataclasses import replace

    import level_loader
    from run_match import run_match
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    spec = replace(level.match, time_limit_s=90.0)
    blasts = []

    def hook(sim):
        blasts.extend(e for e in sim.tick_events
                      if type(e).__name__ == "ExplosionEvent")

    run = run_match(level, spec, record=True, checkpoint_every_ticks=120,
                    on_tick=hook, out=lambda *_: None)
    assert blasts, "no grenade in the recorded match (the scenario is moot)"
    again = run_match(level, replay=run.recording, out=lambda *_: None)
    assert again.checkpoints_verified == len(run.recording.checkpoints)
    assert again.tick_digest == run.tick_digest


# --- the row door -----------------------------------------------------------

@pytest.mark.parametrize("key,value", [
    ("pressure", 0.0), ("heat_amount", 0.0), ("heat_radius", 0.0),
    ("explosive_kg", 0.1), ("ignite_intensity", 0.0), ("expansion_m3", 0.0),
    ("expansion_radius", 0.0), ("expansion_ticks", 1)])
def test_the_door_refuses_what_a_strength_row_may_not_author(key, value):
    """PROPERTY: the door refuses by the KEY's presence, by name, even at a value
    equal to its default: a strength row cannot say two things about one blast.

    BREAKS IF: the door reads `col()`, which cannot tell absent from default.
    """
    with pytest.raises(ValueError, match=key):
        PayloadTable({"t": dict(ROW, strength=1.0, **{key: value})})


@pytest.mark.parametrize("drop,needle", [
    ("explosive", "explosive"), ("emit_blast_smoke", "emit_blast_smoke"),
    ("radius", "radius")])
def test_the_door_requires_what_a_strength_row_needs(drop, needle):
    """PROPERTY: a strength row needs its explosive, its smoke and a damage disc.

    BREAKS IF: a row can skip `blast_log` or the soot it owes.
    """
    row = {k: v for k, v in ROW.items() if k != drop}
    with pytest.raises(ValueError, match=needle):
        PayloadTable({"t": dict(row, strength=1.0)})


def test_the_door_refuses_a_negative_strength():
    """PROPERTY: strength is >= 0, 0 meaning no generic blast.

    BREAKS IF: a negative strength reaches the walk as a sink.
    """
    with pytest.raises(ValueError, match="strength"):
        PayloadTable({"t": dict(ROW, strength=-1.0)})
    assert PayloadTable({"t": dict(ROW, strength=0.0, explosive_kg=0.1)})


def test_the_blast_door_refuses_a_missing_key():
    """PROPERTY: the [blast] door refuses by name, never defaulting.

    BREAKS IF: a missing key silently means zero.
    """
    from simulation.blast_expansion import blast_settings
    with pytest.raises(ValueError, match="grenade_kg"):
        blast_settings({"expansion_m3_per_strength": 1.0,
                        "heat_mj_per_strength": 1.0, "ignite_margin": 1.0})


def test_every_shipped_generic_row_derives_its_charge():
    """PROPERTY: a strength row's charge is derived before the blast-smoke check,
    so the shipped table loads without authoring `explosive_kg`.

    BREAKS IF: the door checks the mass before deriving it.
    """
    rows = [r for r in get_tables().payloads.by_name.values() if r.strength > 0]
    assert rows
    assert all(r.explosive_kg > 0 and r.expansion_m3 > 0 and r.heat_mj > 0
               for r in rows)


# --- ignition ---------------------------------------------------------------

def _fuel_room(fuel_at, wall=None):
    tm = _room()
    if wall is not None:
        tm[:, wall] = MAT_HULL
    tm[fuel_at] = MAT_FURNITURE
    return tm


def _burns(s, ticks=6):
    for _ in range(ticks):
        _step(s)
    return int(s.gmap.fire.astype(np.int64).sum())


@pytest.mark.parametrize("name,off", [("frag_standard", 1),
                                      ("engine_death", 2),
                                      ("magazine_cookoff", 3)])
def test_a_blast_lights_fuel_within_its_ignite_radius(name, off):
    """PROPERTY: a row with `ignite_radius` lights fuel it reaches, in air, through
    delivered heat: the temperature crosses and `apply_temperature_ignition` does
    the lighting.

    BREAKS IF: the ignition heat is sized off a stale constant, or lands on gas.
    """
    row = _shipped(name)
    assert row.ignite_r2 > off * off, "the row's reach moved (see the design)"
    s = _sim(_fuel_room((15, 15 + off)))
    _step(s)
    _fire(s, 15, 15, row)
    assert int(s.gmap.fire[15, 15 + off]) == 0, "a fire was written, not lit"
    assert _burns(s) > 0


def test_fuel_behind_a_wall_does_not_light():
    """PROPERTY: ignition stops at walls: the fuel tile is inside the radius but
    no walk reaches it.

    BREAKS IF: the ignition takes a disc.
    """
    row = _row(1.0, ignite_radius=4.0)
    s = _sim(_fuel_room((15, 17), wall=16))
    _step(s)
    _fire(s, 15, 15, row)
    assert _burns(s) == 0


def test_no_ignite_radius_lights_nothing():
    """PROPERTY: with the key absent a blast's own heat does not light fuel one
    tile off, so one number per row turns its fire-starting off.

    BREAKS IF: the blast's heat is enough on its own, and the ruling's reason for
    the rule is gone (surface it to Erik, do not bend this test).
    """
    row = _row(1.0)
    s = _sim(_fuel_room((15, 16)))
    _step(s)
    _fire(s, 15, 15, row)
    assert _burns(s) == 0


def test_the_ignition_heat_comes_from_the_tile_material():
    """PROPERTY: the heat is the tile's own capacity times what it lacks, so two
    fuels of different capacity both land at the same temperature margin.

    BREAKS IF: one heat is used for every fuel.
    """
    from simulation.materials import MAT_WOOD  # noqa: F401
    from simulation.payloads import heat_fuel
    s = _sim(_fuel_room((15, 16)))
    g = s.gmap
    q = EditQueue()
    heat_fuel(g, q, [(15, 15), (15, 16)])
    q.flush(g, s.rng)
    assert int(g.heat[15, 15]) == 0, "the ignition heat landed on gas"
    tbl = g.materials
    lands_at = (float(g.heat[15, 16]) / 65536.0
                / float(tbl.thermal_mass[MAT_FURNITURE]))
    assert lands_at > float(tbl.ignition_temp[MAT_FURNITURE])


# --- the breach charge ------------------------------------------------------

def test_a_breach_charge_breaches_a_door_and_spares_its_planter():
    """PROPERTY: the door charge still opens a door, and a marine standing one
    clear tile back from it loses no HP and is not knocked down. No rule fixes a
    placement distance (the order takes any tile), so this is the marine at his
    nearest that keeps a tile of air: the row's heat reaches two tiles, and a
    footprint touching the door stands in it (measured: 29 HP, knocked down).

    BREAKS IF: the row's strength or unit damage is raised past the planter's
    limit, or the wall damage stops reaching the door.
    """
    tm = _divided()
    tm[14:17, 15] = MAT_DOOR_CLOSED
    s = _sim(tm)
    g = s.gmap
    u = Unit("P", x=11.0, y=14.0, team=0)
    s.add_unit(u)
    hp0 = u.current_hp
    _step(s)
    _fire(s, 15, 15, _shipped("breach_focus"), approach=(15, 13))
    assert not g.solid[15, 15], "the door held"
    down = False
    for _ in range(36):
        _step(s)
        down = down or any(st.kind == KNOCKED_DOWN for st in u.statuses)
    assert u.current_hp == hp0 and not down
