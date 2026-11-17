"""#29 P7c.1 / P7c.2 -- the CPU side of the device payload function: the refusal
codes, the device edit door's DISC and TILES rows and their replay (noise and
scale included), the blast log's and the blast-source table's record
functions, the payload table rows. The device function itself is gated by
tests/cuda_unit_payloads_check.py (wrapper tests/test_cuda_unit_payloads.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simulation import applier, edit_door, object_tables, unit_schema, unit_tables
from simulation import field_edit as fe
from simulation.weapons import get_tables

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def box_sim():
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.materials import MAT_AIR, MAT_HULL
    from simulation.ruleset import ContinuousRealtime
    tm = np.full((20, 30), MAT_HULL, dtype=np.int32)
    tm[1:19, 1:29] = MAT_AIR
    level = LevelData(name="p7c1_box", version="3", path=Path("."), tilemap=tm,
                      tile_size_m=1.0, diffuse_path=Path("."))
    return Simulation(level, seed=3, enable_recorder=False, ruleset=ContinuousRealtime())


def test_every_payload_row_has_a_device_form():
    """Property (#29 P7c.2): the PAYLOAD function applies EVERY row of the
    payload table -- the strength path, bulk gas, the explosive rows' air read
    and the multi-tick source included -- so ``applier.payload_refusal`` is 0
    for each, and every code the refusal list still names (none) has its
    reason. Breaks if a row is refused again (a branch of execute_payload
    loses its device form) or a code is listed without its reason."""
    for name, p in get_tables().payloads.by_name.items():
        assert applier.payload_refusal(p) == 0, name
    for c, n, why in applier.PAYLOAD_REFUSALS:
        assert applier.payload_refusal_reason(c).startswith(n) and why


def test_the_payload_rows_are_read_from_the_sims_own_rows(box_sim):
    """Property: every per-payload Device tables row has one value per row of
    the payload table, read from the sim's own PayloadDef (the refusal row is
    the refusal code of each). Breaks if a row reads another table, drops a
    row, or the refusal row is computed apart from ``payload_refusal``."""
    vals = unit_tables.values(box_sim)
    names = list(box_sim.weapons_tables.payloads.by_name)
    got = unit_tables.read(vals, "payload_refuse")
    assert len(got) == len(names)
    assert tuple(got) == tuple(applier.payload_refusal(box_sim.weapons_tables.payloads.by_name[n])
                               for n in names)
    assert tuple(unit_tables.read(vals, "payload_radius")) == tuple(
        int(box_sim.weapons_tables.payloads.by_name[n].radius) for n in names)


def _disc(field, r, c, radius, falloff, amount, src, channel=None, clamp=None,
          mode=fe.EditMode.ADD):
    return fe.FieldEdit(field=field, region=fe.Region.DISC, coords=(r, c, float(radius)),
                        amount=float(amount), mode=mode, falloff=falloff, channel=channel,
                        clamp=clamp, source_id=src)


def test_a_disc_edits_door_rows_are_the_pack_plus_the_sort_key_and_the_region_arguments(
        box_sim):
    """Property: ``door_rows`` of a DISC edit is ``pack_for_device``'s EDIT and
    TILE rows, then the source id, the region and the region's arguments (the
    centre, the radius as the region walk quantizes it, the falloff). Breaks if
    an argument is dropped or quantized otherwise than ``_iter_region``."""
    pairs = [(_disc("gas", 9, 12, 2.5, fe.Falloff.LINEAR, 1.25, 3, channel=1), 40),
             (_disc("heat", -1, 3, 1.0, fe.Falloff.FLAT, 7.5, 4), 41)]
    pack, _r = fe.pack_for_device(box_sim.gmap, pairs, None)
    e, t = edit_door.door_rows(box_sim.gmap, pairs)
    assert np.array_equal(e[:, :edit_door.PACKED_COLS], pack.edits)
    assert np.array_equal(t, pack.tiles)
    a = edit_door.ARG_COL
    assert e[0, a:a + 4].tolist() == [9, 12, 2 * 65536 + 32768, int(fe.Falloff.LINEAR)]
    assert e[1, a:a + 4].tolist() == [-1, 3, 65536, int(fe.Falloff.FLAT)]
    assert e[:, edit_door.REGION_COL].tolist() == [int(fe.Region.DISC)] * 2


def test_a_disc_edit_replays_to_the_rows_it_came_from(box_sim):
    """Property: the chained landing's inverse (``edit_door.edit_of``) turns a
    DISC row back into a FieldEdit whose door rows are the same rows (the
    replay reproduces the device's rows), including a clamp equal to the
    field's policy clamp (ignite_ring's). Breaks if the centre, the radius,
    the falloff, the slice or the clamp is lost on the way back."""
    from simulation import payloads
    pairs = [(_disc("fire", 4, 5, 2.5, fe.Falloff.LINEAR, 0.5, 4,
                    clamp=payloads.IGNITE_CLAMP, mode=fe.EditMode.MAX), 7),
             (_disc("gas", 0, 29, 3.0, fe.Falloff.LINEAR, 1.5, 3, channel=2), 8),
             (_disc("smoke", 10, 10, 0.8, fe.Falloff.FLAT, 1e6, 1,
                    mode=fe.EditMode.REMOVE), 9)]
    e, t = edit_door.door_rows(box_sim.gmap, pairs)
    back = []
    for k in range(e.shape[0]):
        edit, key = edit_door.edit_of(box_sim.gmap, e[k], t[t[:, 0] == k])
        back.append((edit, key))
    e2, t2 = edit_door.door_rows(box_sim.gmap, back)
    assert np.array_equal(e, e2) and np.array_equal(t, t2)


def test_the_blast_log_is_an_object_table_whose_records_round_trip(box_sim):
    """Property: the blast log's record function and the table's decode give
    the host list back (every logged disc live, in call order). Breaks if a
    column is dropped, reordered or decoded otherwise."""
    from simulation import unit_kernels
    log = [(3, 4, 2, 25 * 65536), (-1, 7, 3, 0)]
    box_sim.gmap.blast_log[:] = log
    recs = unit_kernels.blast_log_of(box_sim)
    t = unit_schema.BLAST_LOG_TABLE
    back = object_tables.unpack(t, recs, recs.shape[0])
    assert back == log
    assert all(int(r[object_tables.table_layout(t).live]) == 1 for r in recs)
    box_sim.gmap.blast_log.clear()


def _tiles_edit(field, tiles, weights, total_q, noise, scale, src):
    """The edit payloads.deposit_over_tiles enqueues (one TILES FieldEdit)."""
    q = fe.EditQueue()
    from simulation import payloads
    payloads.deposit_over_tiles(q, field, tiles, weights, total_q, noise=noise,
                                scale=scale, source_id=src)
    return q._edits[0][0]


def test_a_tiles_edits_door_rows_carry_its_count_and_scale_and_replay(box_sim):
    """Property (#29 P7c.2): ``door_rows`` of the fixed-total deposit (a
    TILES edit: integer weights over their sum, a noise, an air-read scale) is
    ``pack_for_device``'s rows plus the source, the region and the region's
    arguments (the tile count, whether it is scaled), and ``edit_of`` turns
    the rows back into an edit with the same rows (the chained landing).
    Breaks if the TILES region, its count, its scale flag, the noise's low
    bound or a tile's scale factor is lost on the way back."""
    h, w = box_sim.gmap.material.shape
    scale = np.full((h, w), 65536, dtype=np.int64)
    scale[5, 6] = 40000
    scale[5, 7] = 131072
    tiles = [(5, 6), (5, 7), (6, 6)]
    weights = [65536, 30000, 12345]
    pairs = [(_tiles_edit("smoke", tiles, weights, 98765, 0.85, scale, 2), 11),
             (_tiles_edit("heat", tiles, weights, 4321 * 65536, 0.0, None, 7), 12),
             (_tiles_edit("smoke", tiles, weights, 5555, 0.3, None, 2), 13)]
    from simulation.draws import MatchDraws
    pack, _r = fe.pack_for_device(box_sim.gmap, pairs, MatchDraws(5))
    e, t = edit_door.door_rows(box_sim.gmap, pairs)
    assert np.array_equal(e[:, :edit_door.PACKED_COLS], pack.edits)
    a = edit_door.ARG_COL
    assert e[:, edit_door.REGION_COL].tolist() == [int(fe.Region.TILES)] * 3
    assert [e[k, a:a + 2].tolist() for k in range(3)] == [[3, 1], [3, 0], [3, 0]]
    back = [edit_door.edit_of(box_sim.gmap, e[k], t[t[:, 0] == k]) for k in range(3)]
    e2, t2 = edit_door.door_rows(box_sim.gmap, back)
    assert np.array_equal(e, e2) and np.array_equal(t, t2)


def test_a_noisy_scaled_disc_edit_replays_to_its_rows(box_sim):
    """Property (#29 P7c.2): add_explosion_smoke's DISC edit -- noisy and
    scaled by the disc's air read -- replays through ``edit_of`` to the same
    door rows (the noise from the row's low bound, the scale from the TILE
    rows' factors). Breaks if either is dropped or the denominator's ONE
    factors are misread."""
    from simulation import physics
    h, w = box_sim.gmap.material.shape
    scale = np.full((h, w), 65536, dtype=np.int64)
    scale[8, 8] = 12345
    q = fe.EditQueue()
    physics.add_explosion_smoke(box_sim.gmap, q, 8, 9, 3, amount=0.5, amount_q=777777,
                                scale=scale)
    physics.add_explosion_smoke(box_sim.gmap, q, 2, 2, 2, amount=0.5, amount_q=4444)
    pairs = list(q._edits)
    e, t = edit_door.door_rows(box_sim.gmap, pairs)
    back = [edit_door.edit_of(box_sim.gmap, e[k], t[t[:, 0] == k]) for k in range(2)]
    e2, t2 = edit_door.door_rows(box_sim.gmap, back)
    assert np.array_equal(e, e2) and np.array_equal(t, t2)
    assert e[0, fe.E_NOISY] == 1 and e[0, fe.E_DEN] == 65536 ** 3
    assert e[1, fe.E_DEN] == 65536 ** 2


def test_the_blast_source_table_round_trips_with_its_tile_side_table(box_sim):
    """Property (#29 P7c.2): ``GameMap.expansions`` through the blast-source
    table's record functions (``sources_of``, ``source_tiles_of``) and back
    (``sources_to_host``) is the same list -- each source's tiles in walk
    order, its per-tile source and its ticks still to deliver -- and every
    host source reads live. Breaks if a column, the span or the side table's
    order is lost."""
    from simulation import unit_kernels
    g = box_sim.gmap
    g.expansions[:] = [[[(3, 4), (3, 5), (4, 4)], 123456, 3],
                       [[(9, 9)], 7, 1],
                       [[(1, 1), (1, 2)], 2 ** 40, 2]]
    want = [[list(e[0]), e[1], e[2]] for e in g.expansions]
    recs = unit_kernels.sources_of(box_sim)
    tiles = unit_kernels.source_tiles_of(box_sim)
    lay = object_tables.table_layout(unit_schema.SOURCE_TABLE)
    assert all(int(r[lay.live]) == 1 for r in recs)
    assert tiles.shape[0] == 6
    unit_kernels.sources_to_host(box_sim, recs, tiles)
    assert [[list(e[0]), e[1], e[2]] for e in g.expansions] == want
    g.expansions.clear()


def test_the_strength_rows_are_the_laws_own_doors(box_sim):
    """Property (#29 P7c.2): each strength-path Device tables row is the
    reference law's own door function on the sim's own row and map -- the
    source total (``blast_expansion.source_total_q``, one tick for a strength
    row), the heat (``payloads.heat_base_q``), the soot (``soot_units_q``),
    the walks' r2 -- and a row that is not a strength row reads 0 there.
    Breaks if a row re-derives its value or reads another row's."""
    from simulation import blast_expansion as bx, payloads
    vals = unit_tables.values(box_sim)
    P = box_sim.weapons_tables.payloads.by_name
    names = list(P)
    gen = unit_tables.read(vals, "payload_generic")
    tot = unit_tables.read(vals, "payload_source_total_q")
    hb = unit_tables.read(vals, "payload_heat_base_q")
    soot = unit_tables.read(vals, "payload_soot_units_q")
    hr2 = unit_tables.read(vals, "payload_heat_r2")
    for k, n in enumerate(names):
        p = P[n]
        assert soot[k] == payloads.soot_units_q(p), n
        if p.strength > 0:
            assert gen[k] == 1 and hr2[k] == p.heat_r2, n
            assert tot[k] == bx.source_total_q(box_sim.gmap, p.expansion_m3, 1), n
            assert hb[k] == payloads.heat_base_q(box_sim.gmap, p), n
        else:
            assert gen[k] == 0 and hb[k] == 0 and hr2[k] == 0, n
    assert sum(gen) >= 1
