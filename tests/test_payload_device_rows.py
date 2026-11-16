"""#29 P7c.1 -- the CPU side of the device payload function: the refusal codes,
the device edit door's DISC rows and their replay, the blast log's record
function, the payload table rows. The device function itself is gated by
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


def test_a_refusal_code_names_the_first_branch_execute_payload_takes_off_the_disc_path():
    """Property: ``applier.payload_refusal`` is 0 exactly for a row the disc
    path applies (no strength, no bulk gas, no explosive, no expansion) and
    otherwise names the first of those branches the row takes, in
    execute_payload's own precedence (strength wins over everything). Breaks
    if the precedence changes, a branch is dropped, or a disc-path row is
    refused."""
    for name, p in get_tables().payloads.by_name.items():
        code = applier.payload_refusal(p)
        if p.strength > 0:
            assert code == 1, name
        elif p.bulk_gas:
            assert code == 2, name
        elif p.explosive:
            assert code == 3, name
        elif p.expansion_m3 > 0:
            assert code == 4, name
        else:
            assert code == 0, name
    codes = {c for c, _n, _w in applier.PAYLOAD_REFUSALS}
    assert codes == set(range(1, len(applier.PAYLOAD_REFUSALS) + 1))
    for c in codes:
        assert applier.payload_refusal_reason(c).split(":")[0] in (
            "strength", "bulk_gas", "explosive", "expansion")


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
