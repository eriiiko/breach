"""A6 — doors v0 (docs/a6_doors_v0_impl_2026-07-19.md v2, test plan §13).

The door entity: MAT_DOOR_CLOSED stamp, load order, the slot-9e structural
sweep (want_open latch, occupancy/water retry, per-tile HP ledger,
whole-door external destruction), path-hold, capture carriage, determinism
— plus the critique's added regressions: B1 (blast destroys a closed
door), S2 (the one-boundary-tick lag), N7 (vacuum-adjacent cycle).

Fixture levels are programmatic ``LevelData`` (the A5 /
test_eos_p1_species_transport idiom) — no repo level is touched, no golden
moves. Sims run ``breach_physics=None`` where physics is irrelevant.

Run:
    conda run -n data python -m pytest tests/test_a6_doors.py -q
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
import level_loader  # noqa: E402
from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import door as door_mod  # noqa: E402
from simulation.entities.serialize import serialize_entity_state  # noqa: E402
from simulation.events import DoorDestroyedEvent  # noqa: E402
from simulation.field_edit import EditQueue  # noqa: E402
from simulation.gamemap import (  # noqa: E402
    GameMap, MAT_AIR, MAT_DOOR_CLOSED, MAT_HULL, N_GASES,
)
from simulation.orders import DET_BETWEEN_PHASES, ORDER_EXPLOSIVE, Order  # noqa: E402
from simulation.physics import apply_explosion  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation import wall_fixed  # noqa: E402

FP_ONE = 65536
CLOSED, OPEN, DESTROYED = (door_mod.DOOR_CLOSED, door_mod.DOOR_OPEN,
                           door_mod.DOOR_DESTROYED)

# Synced arrays for the §4.3 field-identity and §13.10 determinism checks.
SYNCED_ARRAYS = ("material", "is_vacuum", "gas", "atmosphere", "temperature",
                 "water_depth", "wall_hp", "solid", "permeability",
                 "flammable", "obstacles", "wind_x", "wind_y",
                 "flow_vx", "flow_vy")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _door_inst(eid, ordinal, x, y, orientation="h", length_m=1.0,
               initial_state="closed"):
    fields = {f.name: f.default for f in door_mod.door.FIELDS}
    fields.update(x=x, y=y, orientation=orientation, length_m=length_m,
                  initial_state=initial_state)
    return EntityInstance(id=eid, class_name="door", ordinal=ordinal,
                          fields=fields)


def _level(tm, doors=(), name="a6_fix", version="1", tile_size_m=1.0, **kw):
    return LevelData(name=name, version=version, path=Path("."), tilemap=tm,
                     tile_size_m=tile_size_m, diffuse_path=Path("."),
                     entities=list(doors), **kw)


def _box_tm(h=12, w=12):
    """v1 vocabulary: hull ring (1), interior air (4). tile_size_m=1 -> 1
    tile per meter, so length_m N == N tiles."""
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return tm


def _split_box_tm(h=12, w=12, wall_x=6, gap_rows=()):
    """Hull box with a vertical interior hull wall at ``wall_x``; the rows
    in ``gap_rows`` stay air (the doorway)."""
    tm = _box_tm(h, w)
    tm[1:h - 1, wall_x] = 1
    for r in gap_rows:
        tm[r, wall_x] = 4
    return tm


def _sim(level, seed=1, physics=None, recorder=False):
    s = Simulation(level, seed=seed, breach_physics=physics,
                   enable_recorder=recorder)
    return s


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def _slice_totals(g):
    return [int(g.gas[i].astype(np.int64).sum()) for i in range(N_GASES)]


def _the_door(sim):
    (d,) = sim._doors
    return d


# ---------------------------------------------------------------------------
# §13.1 — span derivation
# ---------------------------------------------------------------------------

def test_span_quantization_rules():
    q = door_mod.quantize_span_tiles
    assert q(1.0, 0.333) == 3                     # default door = 1 m = 3 tiles
    assert q(0.5, 0.333) == 2                     # 1.5 rounds HALF-UP -> 2
    assert q(0.34, 0.333) == 1                    # ~1.02 -> 1
    assert q(0.333, 0.333) == 1                   # the 1-tile door spelling
    assert door_mod.tiles_per_m(0.333) == 3       # the pinned 0.333 -> 3 map
    assert door_mod.tiles_per_m(1.0) == 1
    assert door_mod.tiles_per_m(0.25) == 4
    with pytest.raises(ValueError, match="non-integral"):
        door_mod.tiles_per_m(0.3)                 # 10/3 tiles/m: undefined
    with pytest.raises(ValueError, match="length_m"):
        q(0.0, 0.333)                             # N10: strictly positive
    with pytest.raises(ValueError, match="length_m"):
        q(-1.0, 0.333)


def test_span_orientation_and_res_replication():
    f = dict(x=4, y=2, orientation="h", length_m=1.0, initial_state="closed")
    assert door_mod.base_span(f, 0.333) == [(2, 4), (2, 5), (2, 6)]
    fv = dict(f, orientation="v")
    assert door_mod.base_span(fv, 0.333) == [(2, 4), (3, 4), (4, 4)]
    # --res 2: each base tile -> its 2x2 block; row-major sorted (S4 pin);
    # never re-quantized from meters at the scaled resolution.
    f1 = dict(x=3, y=1, orientation="h", length_m=0.34, initial_state="closed")
    assert door_mod.runtime_span(f1, 0.333, 2) == [(2, 6), (2, 7),
                                                   (3, 6), (3, 7)]
    # base 1x3 at --res 2 == the 2x6 rectangle the painted grid would give.
    assert door_mod.runtime_span(f, 0.333, 2) == sorted(
        (2 * 2 + dy, 2 * x + dx)
        for x in (4, 5, 6) for dy in (0, 1) for dx in (0, 1))


def test_res_recovery_uses_base_tile_size():
    """S1: a --res level carries tile_size_m_base + res_factor; the span
    rule reads the BASE size (the live 0.1665 would hard-error)."""
    tm = np.repeat(np.repeat(_box_tm(), 2, axis=0), 2, axis=1)
    d = _door_inst("d", 0, x=6, y=5, orientation="v", length_m=2.0)
    lvl = _level(tm, [d], tile_size_m=0.5, res_factor=2, tile_size_m_base=1.0)
    from simulation.door_system import door_spans
    (_, span), = door_spans(lvl)
    assert span == sorted((2 * fy + dy, 2 * 6 + dx)
                          for fy in (5, 6) for dy in (0, 1) for dx in (0, 1))


# ---------------------------------------------------------------------------
# §13.2 — load stamp + validation
# ---------------------------------------------------------------------------

def test_load_stamp_closed_preseed():
    tm = _split_box_tm(gap_rows=(5, 6))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=5, orientation="v",
                                 length_m=2.0)])
    g = GameMap(lvl)
    hp_q = int(wall_fixed.quantize_scalar(
        float(g.materials.hp[MAT_DOOR_CLOSED])))
    for t in ((5, 6), (6, 6)):
        assert int(g.material[t]) == MAT_DOOR_CLOSED
        assert bool(g.solid[t]) and not g.is_passable(*t)
        assert int(g.atmosphere[t]) == 0            # seeded post-stamp solid
        assert all(int(g.gas[i][t]) == 0 for i in range(N_GASES))
        assert int(g.wall_hp[t]) == hp_q


def test_load_stamp_open_is_air():
    tm = _split_box_tm(gap_rows=(5, 6))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=5, orientation="v",
                                 length_m=2.0, initial_state="open")])
    g = GameMap(lvl)
    for t in ((5, 6), (6, 6)):
        assert int(g.material[t]) == MAT_AIR and not g.solid[t]
        assert int(g.atmosphere[t]) == FP_ONE       # ambient (open air)


def test_load_validation_hard_errors():
    tm = _box_tm()
    # OOB span (anchor past the grid — bounds is checked before material).
    with pytest.raises(ValueError, match="out of bounds"):
        GameMap(_level(tm, [_door_inst("d", 0, x=50, y=5)]))
    # Span over hull.
    with pytest.raises(ValueError, match="CSV material"):
        GameMap(_level(tm, [_door_inst("d", 0, x=0, y=5, orientation="v")]))
    # Span over vacuum.
    tmv = _box_tm()
    tmv[5, 5] = 0                                    # v1 code 0 = vacuum
    with pytest.raises(ValueError, match="vacuum"):
        GameMap(_level(tmv, [_door_inst("d", 0, x=5, y=5)]))
    # Overlapping spans.
    with pytest.raises(ValueError, match="overlaps"):
        GameMap(_level(tm, [_door_inst("a", 0, x=4, y=5, length_m=2.0),
                            _door_inst("b", 1, x=5, y=5, length_m=2.0)]))


# ---------------------------------------------------------------------------
# §13.3 — round-trip FIELD identity (§4.3)
# ---------------------------------------------------------------------------

def _assert_fields_identical(ga, gb):
    for name in SYNCED_ARRAYS:
        assert np.array_equal(getattr(ga, name), getattr(gb, name)), \
            f"field '{name}' differs"


def test_authored_open_field_identical_to_authored_air():
    tm = _split_box_tm(gap_rows=(5, 6))
    ga = GameMap(_level(tm, [_door_inst("d", 0, x=6, y=5, orientation="v",
                                        length_m=2.0,
                                        initial_state="open")]))
    gb = GameMap(_level(tm))
    _assert_fields_identical(ga, gb)


def test_authored_closed_field_identical_to_painted_csv():
    # v2 vocabulary paints MAT_DOOR_CLOSED (7) literally.
    tm = np.full((12, 12), MAT_HULL, dtype=np.int32)
    tm[1:11, 1:11] = MAT_AIR
    tm[1:11, 6] = MAT_HULL
    tm[5, 6] = MAT_AIR
    tm[6, 6] = MAT_AIR
    ga = GameMap(_level(tm, [_door_inst("d", 0, x=6, y=5, orientation="v",
                                        length_m=2.0)], version="2"))
    tm_painted = tm.copy()
    tm_painted[5, 6] = MAT_DOOR_CLOSED
    tm_painted[6, 6] = MAT_DOOR_CLOSED
    gb = GameMap(_level(tm_painted, version="2"))
    _assert_fields_identical(ga, gb)


# ---------------------------------------------------------------------------
# §13.4 — flip cycle under the sweep
# ---------------------------------------------------------------------------

def test_flip_cycle_sweep_conserves_exactly():
    tm = _split_box_tm(gap_rows=(6,))
    sim = _sim(_level(tm, [_door_inst("d", 0, x=6, y=6, orientation="v",
                                      length_m=1.0)]))
    g = sim.gmap
    d = _the_door(sim)
    totals0 = _slice_totals(g)

    d.want_open = True
    _step(sim)
    assert d.state == OPEN
    assert int(g.material[6, 6]) == MAT_AIR and not g.solid[6, 6]
    assert _slice_totals(g) == totals0

    d.want_open = False
    _step(sim)
    assert d.state == CLOSED
    assert int(g.material[6, 6]) == MAT_DOOR_CLOSED and bool(g.solid[6, 6])
    assert _slice_totals(g) == totals0

    for _ in range(100):
        d.want_open = not d.want_open
        _step(sim)
    assert _slice_totals(g) == totals0


# ---------------------------------------------------------------------------
# §13.5 — occupancy retry (living blocks, corpse doesn't; latch retained)
# ---------------------------------------------------------------------------

def test_occupancy_blocks_close_and_retries():
    tm = _split_box_tm(gap_rows=(6,))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=6, orientation="v",
                                 length_m=1.0, initial_state="open")])
    sim = _sim(lvl)
    d = _the_door(sim)
    u = Unit("m", x=6, y=6, team=0, footprint=1)   # standing IN the doorway
    sim.add_unit(u)

    d.want_open = False
    _step(sim, 3)
    assert d.state == OPEN and d.want_open is False   # blocked, latch kept
    assert int(sim.gmap.material[6, 6]) == MAT_AIR

    u.x, u.y = 3.0, 3.0                               # step off
    _step(sim)
    assert d.state == CLOSED                          # closes next sweep
    assert u.alive                                    # never crushed


def test_corpse_never_blocks_close():
    tm = _split_box_tm(gap_rows=(6,))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=6, orientation="v",
                                 length_m=1.0, initial_state="open")])
    sim = _sim(lvl)
    d = _the_door(sim)
    u = Unit("m", x=6, y=6, team=0, footprint=1)
    sim.add_unit(u)
    u.alive = False                                   # a corpse in the doorway
    d.want_open = False
    _step(sim)
    assert d.state == CLOSED                          # sealed over (§15.2)


# ---------------------------------------------------------------------------
# §13.6 — water retry (via can_seal_tiles)
# ---------------------------------------------------------------------------

def test_water_blocks_close_until_drained():
    tm = _split_box_tm(gap_rows=(6,))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=6, orientation="v",
                                 length_m=1.0, initial_state="open")])
    sim = _sim(lvl)
    g = sim.gmap
    d = _the_door(sim)
    g.water_depth[6, 6] = int(0.2 * FP_ONE)           # standing water on span
    d.want_open = False
    _step(sim, 2)
    assert d.state == OPEN                            # refused, retrying
    g.water_depth[6, 6] = 0                           # drain
    _step(sim)
    assert d.state == CLOSED


# ---------------------------------------------------------------------------
# §13.7 — path-hold (the door-closes-across-path test)
# ---------------------------------------------------------------------------

def test_path_hold_across_closing_door():
    tm = _split_box_tm(gap_rows=(6,))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=6, orientation="v",
                                 length_m=1.0, initial_state="open")])
    sim = _sim(lvl)
    d = _the_door(sim)
    u = Unit("m", x=3, y=6, team=0, footprint=1)
    sim.add_unit(u)
    # Precomputed per-tick path straight through the doorway (col 3 -> 9).
    u.move_path = [(float(x), 6.0) for x in range(4, 10)]

    _step(sim)                                        # walks to x=4
    assert (u.x, u.y) == (4.0, 6.0)
    d.want_open = False                               # slam it (closes at 9e)
    _step(sim)                                        # slot 3 walked to x=5
    assert (u.x, u.y) == (5.0, 6.0) and d.state == CLOSED
    off0 = u.path_tick_offset
    _step(sim, 3)                                     # held at the door
    assert (u.x, u.y) == (5.0, 6.0)
    assert u.path_tick_offset == off0 + 3             # ticks burned, no skip
    d.want_open = True                                # reopen
    _step(sim)                                        # 9e opens AFTER slot 3
    _step(sim, 4)                                     # resumes, walks the tail
    assert (u.x, u.y) == (9.0, 6.0)                   # next un-walked index on


# ---------------------------------------------------------------------------
# §13.8 — capture carriage (recorder + get_state, ONE serializer)
# ---------------------------------------------------------------------------

def test_recorder_and_get_state_carry_door_rows():
    tm = _split_box_tm(gap_rows=(6,))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=6, orientation="v",
                                 length_m=1.0)])
    sim = _sim(lvl, recorder=True)
    _step(sim)
    blob0 = sim.recorder.entity_snapshots[0]
    assert blob0 is not None
    for token in (b"state|", b"want_open|", b"hp_0|"):
        assert token in blob0
    assert blob0 == serialize_entity_state(sim.entities)   # one serializer
    st = sim.get_state()
    assert st.entity_state["n_entities"] == 1
    from simulation.entities.serialize import entity_section_bytes
    assert entity_section_bytes(st.entity_state) \
        == serialize_entity_state(sim.entities)
    # Rows change across a flip.
    _the_door(sim).want_open = True
    _step(sim)
    assert sim.recorder.entity_snapshots[1] != blob0


def test_door_free_level_builds_identical_entity_list():
    lvl = _level(_box_tm())
    sim = _sim(lvl)
    assert sim._doors == []
    assert sim.entities == list(lvl.entities)


# ---------------------------------------------------------------------------
# §13.10 — determinism: identical toggle script, bit-identical trajectories
# ---------------------------------------------------------------------------

def test_toggle_script_deterministic():
    def run():
        tm = _split_box_tm(gap_rows=(5, 6))
        lvl = _level(tm, [_door_inst("d", 0, x=6, y=5, orientation="v",
                                     length_m=2.0)])
        sim = _sim(lvl, seed=7)
        d = _the_door(sim)
        snaps = []
        for t in range(12):
            if t in (2, 5, 9):
                d.want_open = not d.want_open
            _step(sim)
            snaps.append((
                {n: getattr(sim.gmap, n).copy() for n in SYNCED_ARRAYS},
                serialize_entity_state(sim.entities),
            ))
        return snaps

    a, b = run(), run()
    for t, ((fa, ea), (fb, eb)) in enumerate(zip(a, b)):
        assert ea == eb, f"entity bytes diverge at tick {t}"
        for name in SYNCED_ARRAYS:
            assert np.array_equal(fa[name], fb[name]), \
                f"field '{name}' diverges at tick {t}"


# ---------------------------------------------------------------------------
# §13.11 — HP ledger: no heal, no smear
# ---------------------------------------------------------------------------

def test_hp_ledger_no_heal_no_smear():
    tm = _split_box_tm(gap_rows=(5, 6))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=5, orientation="v",
                                 length_m=2.0)])
    sim = _sim(lvl)
    g = sim.gmap
    d = _the_door(sim)
    full = int(g.wall_hp[5, 6])
    damaged = full - int(wall_fixed.quantize_scalar(12.5))
    g.wall_hp[5, 6] = damaged                     # shoot tile 0 (span[0])

    d.want_open = True
    _step(sim)                                    # fold on open
    assert d.hp == [damaged, full]                # S4 order: span row-major
    d.want_open = False
    _step(sim)                                    # restamp on close
    assert int(g.wall_hp[5, 6]) == damaged        # no heal
    assert int(g.wall_hp[6, 6]) == full           # no smear
    # Digest rows carry the ledger.
    rows = dict(door_mod.door.runtime_digest_rows(d))
    assert rows["hp_0"] == damaged and rows["hp_1"] == full


# ---------------------------------------------------------------------------
# §13.12 — external destruction / whole-door (observables only, N3)
# ---------------------------------------------------------------------------

def test_burst_differential_kills_whole_door():
    tm = _split_box_tm(gap_rows=(5, 6))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=5, orientation="v",
                                 length_m=2.0)])
    sim = _sim(lvl)
    g = sim.gmap
    d = _the_door(sim)
    # Left room to 4 atm: spread across the closed door = 3.0 > 2.0.
    left = np.zeros_like(g.atmosphere, dtype=bool)
    left[1:11, 1:6] = True
    open_left = left & ~g.solid & ~g.is_vacuum
    g.atmosphere[open_left] = 4 * FP_ONE
    _step(sim)
    # Observables only (which tile 9b popped vs 9e completed: unasserted).
    assert d.alive is False and d.state == DESTROYED
    assert d.hp == [0, 0]
    assert all(int(g.material[t]) == MAT_AIR for t in ((5, 6), (6, 6)))
    door_events = [e for e in sim.tick_events
                   if isinstance(e, DoorDestroyedEvent)]
    assert len(door_events) == 2
    # Dead latch: a later toggle does nothing.
    d.want_open = True
    _step(sim)
    assert d.state == DESTROYED and not d.alive
    assert all(int(g.material[t]) == MAT_AIR for t in ((5, 6), (6, 6)))


# ---------------------------------------------------------------------------
# §13.13 — B1 regression: blast destroys a closed door
# ---------------------------------------------------------------------------

def test_blast_destroys_closed_door_tile():
    tm = _split_box_tm(gap_rows=(6,))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=6, orientation="v",
                                 length_m=1.0)])
    g = GameMap(lvl)
    assert int(g.material[6, 6]) == MAT_DOOR_CLOSED
    apply_explosion(g, EditQueue(), 6, 6, radius=2, pressure=1.0,
                    wall_damage=100.0)
    assert int(g.material[6, 6]) == MAT_AIR       # NOT blast-proof (B1)


def test_blast_partial_hit_completes_whole_door_at_9e():
    tm = _split_box_tm(gap_rows=(5, 6))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=5, orientation="v",
                                 length_m=2.0)])
    sim = _sim(lvl)
    g = sim.gmap
    d = _the_door(sim)
    # wall_damage 45: center tile (falloff 1) dies (hp 40); the other span
    # tile at dist 1 takes 22.5 < 40 and survives the blast itself.
    apply_explosion(g, sim.edit_queue, 5, 6, radius=2, pressure=0.0,
                    wall_damage=45.0)
    assert int(g.material[5, 6]) == MAT_AIR
    assert int(g.material[6, 6]) == MAT_DOOR_CLOSED
    _step(sim)                                    # 9e: whole-door rule
    assert d.alive is False and d.state == DESTROYED and d.hp == [0, 0]
    assert int(g.material[6, 6]) == MAT_AIR
    door_events = [e for e in sim.tick_events
                   if isinstance(e, DoorDestroyedEvent)]
    assert door_events and door_events[-1].pos == (6, 6)  # 9e completion


# ---------------------------------------------------------------------------
# §13.14 — S2 timing: the one-boundary-tick lag, pinned end to end
# ---------------------------------------------------------------------------

def test_boundary_explosive_lag_is_exactly_one_tick():
    """The BETWEEN-PHASES explosive volley runs at the tick BOTTOM
    (simulation.py:868-886) — after 9e and after the recorder snapshot —
    so a closed door it destroys stays CLOSED in entity rows for exactly
    one boundary tick; the next tick's 9e reconciles. The real mechanism:
    a marine's breach charge on the door, det slot BETWEEN_PHASES."""
    tpp = int(CFG.clock.ticks_per_phase)
    tm = _split_box_tm(gap_rows=(5, 6))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=5, orientation="v",
                                 length_m=2.0)])
    sim = _sim(lvl)
    d = _the_door(sim)
    u = Unit("m", x=3, y=3, team=0, footprint=1)
    sim.add_unit(u)
    assert sim.apply_action(u.id, Order(ORDER_EXPLOSIVE, 6, 5, 0,
                                        det_slot=DET_BETWEEN_PHASES))
    _step(sim, tpp)          # the volley fires at the bottom of step #tpp
    # The lag: the grid says destroyed, the entity rows still say CLOSED.
    assert int(sim.gmap.material[5, 6]) == MAT_AIR
    assert d.alive and d.state == CLOSED
    _step(sim)               # next tick's 9e reconciles
    assert d.alive is False and d.state == DESTROYED and d.hp == [0, 0]


# ---------------------------------------------------------------------------
# §13.15 — N7: vacuum-adjacent door cycle
# ---------------------------------------------------------------------------

def test_vacuum_adjacent_door_cycle():
    tm = _split_box_tm(gap_rows=(6,))
    tm[5, 6] = 0                                  # exposed vacuum above door
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=6, orientation="v",
                                 length_m=1.0)])
    sim = _sim(lvl)
    g = sim.gmap
    d = _the_door(sim)
    totals0 = _slice_totals(g)
    t = (6, 6)

    d.want_open = True
    _step(sim)                                    # open: joins vacuum, no seed
    assert d.state == OPEN
    assert bool(g.is_vacuum[t]) and not g.solid[t]
    assert all(int(g.gas[i][t]) == 0 for i in range(N_GASES))
    assert _slice_totals(g) == totals0            # nothing minted

    d.want_open = False
    _step(sim)                                    # re-close: sealed-hull state
    assert d.state == CLOSED
    assert bool(g.solid[t]) and bool(g.is_vacuum[t])
    assert _slice_totals(g) == totals0

    d.want_open = True
    _step(sim)                                    # re-open: vacuum again
    assert d.state == OPEN and bool(g.is_vacuum[t])
    assert _slice_totals(g) == totals0


# ---------------------------------------------------------------------------
# door_at — the O-key hit test (geometry, not material)
# ---------------------------------------------------------------------------

def test_door_at_matches_span_geometry_any_state():
    tm = _split_box_tm(gap_rows=(5, 6))
    lvl = _level(tm, [_door_inst("d", 0, x=6, y=5, orientation="v",
                                 length_m=2.0, initial_state="open")])
    sim = _sim(lvl)
    d = _the_door(sim)
    assert sim.door_at(5, 6) is d                 # OPEN span still matches
    assert sim.door_at(6, 6) is d
    assert sim.door_at(7, 6) is None
    assert sim.door_at(5, 5) is None


# ---------------------------------------------------------------------------
# The committed human-test level loads + steps (with real physics)
# ---------------------------------------------------------------------------

def test_door_test_level_loads_and_steps():
    lvl = level_loader.load("door_test")
    sim = Simulation(lvl, seed=42, breach_physics=bp, enable_recorder=False)
    states = {d.id: (d.state, d.want_open) for d in sim._doors}
    assert states == {"door_a": (CLOSED, False), "door_b": (CLOSED, False),
                      "door_c": (CLOSED, False), "door_d": (OPEN, True)}
    assert int((sim.gmap.material == MAT_DOOR_CLOSED).sum()) == 6
    _step(sim, 3)
    # Toggle the alcove door through real physics ticks.
    sim.door_at(14, 8).want_open = True
    _step(sim)
    assert sim.door_at(14, 8).state == OPEN


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
