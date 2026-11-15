"""#29 P7a.3 -- the HOST halves of the device sprays (CPU, no GPU): the
SPRAY_CELLS record and the cell buffer (``simulation.applier.record_of`` /
``records_and_cells_of``), the device edit door's host reference and replay
(``simulation.edit_door``), the claims list's fold and replay
(``simulation.attack_lists``) and the P7a.3 device-table rows.

Each test names the property it protects and the change that must break it.
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

import breach_physics  # noqa: E402,F401  (the kit the trig calls need)
from level_loader import LevelData  # noqa: E402
import level_loader
from simulation import applier, attack_lists, combat, edit_door, simultaneous
from simulation import field_edit as fe
from simulation.damage import TickCredit


def _level():
    from simulation.materials import MAT_AIR, MAT_HULL, MAT_WOOD
    tm = np.full((24, 32), level_loader.SPACE_CODE, dtype=np.int32)
    tm[1:23, 1:31] = MAT_HULL
    tm[2:22, 2:30] = MAT_AIR
    tm[8:14, 14] = MAT_WOOD
    return LevelData(name="p7a3_cpu", version="3", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."), boundary="space")


def _sim_with_sprayers():
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    from simulation.turning import place
    from simulation.unit import Unit
    sim = Simulation(_level(), seed=5, breach_physics=None, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    out = []
    for name, x, y, wpn, face in (("A", 4, 8, "dragon_7", 0),
                                  ("B", 20, 8, "miasma_vent", 205887)):
        u = Unit(name, x=float(x), y=float(y), team=0)
        sim.add_unit(u)
        place(u, face)
        u.weapon_id = wpn
        out.append(u)
    return sim, out


def _capture_resolve(sim, ticks=2):
    """The attack slot's pending list at the conductor's resolve, per tick."""
    got = []
    orig = simultaneous.SimultaneousEffects.resolve

    def res(fx):
        got.append(list(fx._pending))
        return orig(fx)
    simultaneous.SimultaneousEffects.resolve = res
    try:
        for _ in range(ticks):
            for u in sim.units:
                sim.set_trigger(u.id, True)
            sim.step()
            sim.set_paused(False)
    finally:
        simultaneous.SimultaneousEffects.resolve = orig
    return got


def test_spray_cells_record_carries_the_cone_as_a_span_in_record_order():
    """PROPERTY: each spray world call is ONE SPRAY_CELLS record whose span in
    the cell buffer holds exactly that call's cells, in its order, the spans
    consecutive in record order; the call's scalars ride the record as given.
    BREAKS IF the span's start / count, the cells' order, the gas id's -1 for
    None or a scalar of the record changes."""
    sim, (a, b) = _sim_with_sprayers()
    pend = [p for p in _capture_resolve(sim) if p][-1]
    recs, cells = applier.records_and_cells_of(sim, pend)
    sprays = [(r, e) for r, e in zip(recs, pend) if r[0] == applier.KIND_SPRAY_CELLS]
    assert len(sprays) == 2
    off = 0
    for r, e in sprays:
        _gmap, _q, owner, heat_q, gas_id, gas_q, cl = e[2]
        f = {n: int(r[applier.kind_field("SPRAY_CELLS", n)])
             for n in dict(applier.KIND_FIELDS)["SPRAY_CELLS"]}
        assert f["start"] == off and f["count"] == len(cl) > 0
        assert [tuple(c) for c in cells[off:off + len(cl)].tolist()] == [tuple(c) for c in cl]
        assert (f["owner"], f["heat_q"], f["gas_q"]) == (owner, heat_q, gas_q)
        assert f["gas_id"] == (-1 if gas_id is None else gas_id)
        off += len(cl)
    assert off == len(cells)
    assert attack_lists.cells_of(sim, pend).tolist() == cells.tolist()


def test_spray_cells_record_is_refused_without_a_cell_buffer():
    """PROPERTY: a spray's world call becomes a record only together with its
    cells (record_of refuses it by name without the buffer). BREAKS IF
    record_of returns a span into a buffer nobody holds."""
    sim, _ = _sim_with_sprayers()
    pend = [p for p in _capture_resolve(sim, 1) if p][-1]
    spray = [e for e in pend if e[0] == simultaneous._WORLD][0]
    with pytest.raises(applier.EffectRefused, match="cell buffer"):
        applier.record_of(sim, spray)


def test_door_rows_are_pack_for_device_plus_the_sort_key():
    """PROPERTY: the door's host reference of a producer's edits is
    pack_for_device of them, its 16 columns unchanged, with source_id and
    region appended, the edit indices local to the producer. BREAKS IF a
    packed column, the key, or the appended sort key differs."""
    sim, _ = _sim_with_sprayers()
    pend = [p for p in _capture_resolve(sim, 1) if p][-1]
    q = sim.edit_queue
    q.clear()
    q._seq = 37                                   # a window that began earlier
    for e in pend:
        if e[0] == simultaneous._WORLD:
            e[1](*e[2], **e[3])
    pairs = list(q._edits)
    edits, tiles = edit_door.door_rows(sim.gmap, pairs, {5, 6})
    pack, reason = fe.pack_for_device(sim.gmap, pairs, None)
    assert reason is None
    assert np.array_equal(edits[:, :edit_door.PACKED_COLS], pack.edits)
    assert np.array_equal(tiles, pack.tiles)
    assert edits[:, fe.E_KEY].tolist() == [s for _e, s in pairs]
    assert edits[0, fe.E_KEY] == 37
    assert set(edits[:, edit_door.SRC_COL].tolist()) == {5, 6}
    assert set(edits[:, edit_door.REGION_COL].tolist()) == {int(fe.Region.TILE)}


def test_door_refuses_an_interleaved_producer_by_name():
    """PROPERTY: an edit of another source inside a gated walk is refused,
    never compared loosely (decision 7). BREAKS IF door_rows accepts it."""
    sim, _ = _sim_with_sprayers()
    e = fe.FieldEdit(field="heat", region=fe.Region.TILE, coords=(3, 3), amount=1.0,
                     amount_q=65536, source_id=1)
    with pytest.raises(edit_door.DoorInterleave):
        edit_door.door_rows(sim.gmap, [(e, 0)], {5, 6})


def test_replay_through_the_host_enqueue_reproduces_the_rows():
    """PROPERTY: the chained landing (edit_door.replay) enqueues one host
    FieldEdit per device row at the row's key and the replayed edits pack to
    the same rows; a row keyed off the queue's seq is refused. BREAKS IF the
    replay rebuilds a field, slice, amount, mode or source differently."""
    sim, _ = _sim_with_sprayers()
    pend = [p for p in _capture_resolve(sim, 1) if p][-1]
    q = sim.edit_queue
    q.clear()
    for e in pend:
        if e[0] == simultaneous._WORLD:
            e[1](*e[2], **e[3])
    edits, tiles = edit_door.door_rows(sim.gmap, list(q._edits), {5, 6})
    host = list(q._edits)
    q.clear()
    edit_door.replay(sim, edits, tiles)
    assert [(e.field, e.coords, e.amount_q, e.channel, e.source_id, s)
            for e, s in q._edits] == [(e.field, e.coords, e.amount_q, e.channel,
                                       e.source_id, s) for e, s in host]
    q.clear()
    q._seq = 5
    with pytest.raises(edit_door.DoorInterleave, match="seq"):
        edit_door.replay(sim, edits, tiles)


def test_claims_list_folds_to_the_tickcredit_and_replays_onto_it():
    """PROPERTY: the claims list holds every claim in order; folded to the
    first claim per key (after the keys already held) it is what a
    TickCredit gains from the same calls, in its insertion order, and
    replay_claims makes the same dict. BREAKS IF the fold keeps a later
    claim, drops the order or the heat channel's code differs."""
    from simulation.damage import CREDIT_HEAT
    from simulation.unit_events import CREDIT_HEAT as DEV_HEAT
    calls = [(CREDIT_HEAT, 3, 4, 7), (4, 3, 4, 7), (CREDIT_HEAT, 3, 4, 9),
             (CREDIT_HEAT, 5, 5, 9), (4, 3, 4, 1)]
    tc = TickCredit()
    tc.claim(CREDIT_HEAT, ((9, 9),), 2)            # a key held before
    n0 = len(tc)
    for ch, y, x, a in calls:
        tc.claim(ch, ((y, x),), a)
    dev = np.array([((DEV_HEAT if ch == CREDIT_HEAT else ch), y, x, a)
                    for ch, y, x, a in calls] + [(DEV_HEAT, 9, 9, 5)], dtype=np.int64)
    prior = attack_lists.claim_rows(tc)[:n0]
    folded = attack_lists.first_claims(dev, prior)
    assert folded.tolist() == [list(r) for r in attack_lists.claim_rows(tc)[n0:]]

    class G:
        pass
    g = G()
    attack_lists.replay_claims(g, dev[:-1])
    assert attack_lists.claim_rows(g.damage_credit) == attack_lists.claim_rows(tc)[n0:]


def test_spray_table_rows_read_the_sims_own_rows():
    """PROPERTY: the P7a.3 device rows are read per weapons-table row from the
    sim's own WeaponDef / round, quantized by the law's own doors; the range
    row moves with the map's tile size. BREAKS IF a row reads the config
    directly, another quantization, or a process-wide table."""
    from simulation import unit_tables
    sim, _ = _sim_with_sprayers()
    v = unit_tables.values(sim)
    names = list(sim.weapons_tables.weapons.by_name)
    d7 = names.index("dragon_7")
    w = sim.weapons_tables.weapons.by_name["dragon_7"]
    a = sim.weapons_tables.ammo_for_weapon(w)
    assert unit_tables.read(v, "spray_range_tiles")[d7] == w.range_tiles
    assert unit_tables.read(v, "spray_cone_half_q")[d7] == combat.cone_q16_of_deg(
        w.cone_half_angle_degrees)
    from simulation import gas_fixed, unit_fixed
    assert unit_tables.read(v, "spray_heat_q")[d7] == unit_fixed.quantize_scalar(
        float(a.heat_deposit))
    assert unit_tables.read(v, "spray_gas_q")[d7] == gas_fixed.quantize_trace_scalar(
        float(a.gas_amount))
    assert unit_tables.read(v, "spray_gas_id")[d7] == sim.gmap.gases.name_to_id[a.gas_species]
    knife = names.index("combat_knife")
    assert unit_tables.read(v, "spray_has_ammo")[knife] == 0
    flags = unit_tables.read(v, "edit_clamp_flags")
    ords = fe.DEVICE_FIELD_ORDINALS
    assert flags[ords["gas"]] == 1 and flags[ords["heat"]] == 0
