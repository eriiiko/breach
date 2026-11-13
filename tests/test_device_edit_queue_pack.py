"""#29 P4c -- the device edit queue's HOST half (``field_edit.pack_for_device``,
``EditQueue.take_ordered``, ``apply_ordered``), on the CPU build.

The device applies slot 6b's flush from the records this pack writes
(cpp/src/cuda_edit_queue.cu; its tol-0 gate is tests/cuda_edit_queue_check.py,
wrapper tests/test_cuda_edit_queue.py). These tests hold the pack's own
contract, which no CUDA build is needed to see. Each names the property it
protects and the change that must break it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import breach_physics as bp  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import field_edit as fe  # noqa: E402
from simulation.gases import O2  # noqa: E402
from simulation.materials import MAT_AIR, MAT_FURNITURE, MAT_HULL  # noqa: E402

F, R, M, L = fe.FieldEdit, fe.Region, fe.EditMode, fe.Falloff


def _sim(seed=5):
    tm = np.full((20, 20), MAT_HULL, dtype=np.int32)
    tm[1:19, 1:19] = MAT_AIR
    tm[12:14, 12:14] = MAT_FURNITURE
    lev = LevelData(name="pack", version="3", path=Path("."), tilemap=tm,
                    tile_size_m=1.0 / 3.0, diffuse_path=Path("."), entities=[], wires=[])
    return Simulation(lev, seed=seed, breach_physics=bp, enable_recorder=False)


def _edits():
    return [
        F("smoke", R.DISC, (6, 6, 3.0), 0.5, M.ADD, L.LINEAR, noise=0.3, source_id=4),
        F("heat", R.RECT, (2, 2, 4, 9), 1.0, M.ADD, source_id=1),
        F("atmosphere", R.TILE, (9, 9), 0.2, M.REMOVE, source_id=2),
        F("smoke", R.TILE, (6, 6), 0.1, M.REMOVE, source_id=4),
        F("fire", R.BEAM, (3, 3, 10, 12, 1.5), 0.6, M.MAX, L.LINEAR, source_id=3),
        F("gas_energy", R.TILES, ((5, 5, 2), (5, 6, 1), (0, 0, 3)), 0.0, M.ADD,
          amount_q=10 ** 6, weight_den=6, source_id=0),
    ]


def _queue(edits):
    q = fe.EditQueue()
    for e in edits:
        q.enqueue(e)
    return q


def test_the_device_field_ordinals_are_the_policies_lexicographic_ranks():
    """PROPERTY: a field's device ordinal is the rank of its name in the
    lexicographic order of FIELD_POLICY, and every policy has one (the design's
    "the field-ordinal table keeps the string fields' lexicographic order").
    BREAKS IF the table is hand-ordered, or a policy is added without its
    ordinal (the device would apply it as another field)."""
    names = sorted(fe.FIELD_POLICY)
    assert fe.DEVICE_FIELD_ORDINALS == {n: k for k, n in enumerate(names)}
    if getattr(bp, "EDIT_FIELD_ORDINALS", None) is not None:   # a CUDA build
        assert dict(bp.EDIT_FIELD_ORDINALS) == fe.DEVICE_FIELD_ORDINALS


def test_take_ordered_then_apply_ordered_is_the_flush():
    """PROPERTY: the host flush is take_ordered + apply_ordered -- the ONE
    applied order both backends read -- so splitting the flush changed nothing:
    two identical maps, one flushed, one through the split, end equal.
    BREAKS IF take_ordered sorts by another key than the flush, or drops or
    duplicates an edit."""
    a, b = _sim(), _sim()
    _queue(_edits()).flush(a.gmap, a.draws)
    q = _queue(_edits())
    ordered = q.take_ordered()
    assert len(q) == 0
    fe.apply_ordered(b.gmap, ordered, b.draws)
    for name in ("gas", "heat", "fire", "temperature", "gas_energy"):
        assert np.array_equal(getattr(a.gmap, name), getattr(b.gmap, name)), name
    assert a.gmap.trace_deposit_books == b.gmap.trace_deposit_books
    assert a.gmap.gas_energy_books == b.gmap.gas_energy_books


def test_the_pack_writes_the_flush_order_and_the_one_region_walk():
    """PROPERTY: the pack's tile rows are, in order, each edit of the flush's
    applied order (take_ordered) walked by _iter_region -- every tile the host
    loop visits, in its order, with its Q16 weight; each edit row carries its
    seq as the noise key. BREAKS IF the pack re-sorts, re-derives a region,
    drops the masked tiles on the host (the masks are the device's), or keys
    the noise by anything but the seq."""
    sim = _sim()
    g = sim.gmap
    ordered = _queue(_edits()).take_ordered()
    pack, reason = fe.pack_for_device(g, ordered, sim.draws)
    assert reason is None and pack is not None
    want = []
    for k, (e, seq) in enumerate(ordered):
        assert int(pack.edits[k, fe.E_KEY]) == seq
        assert int(pack.edits[k, fe.E_FIELD]) == fe.DEVICE_FIELD_ORDINALS[e.field]
        for (r, c, w) in fe._iter_region(e.region, e.coords, e.falloff, g.solid.shape):
            want.append((k, r * g._w + c, int(w)))
    got = [tuple(int(v) for v in row[:3]) for row in pack.tiles]
    assert got == want


def test_a_water_edit_falls_back_to_the_host_flush():
    """PROPERTY: a tick with a water_depth edit is not packed: the pack returns
    a reason naming water_depth and no records, so the runner flushes on the
    host (the resident pre-step reads water_depth before slot 7's upload).
    BREAKS IF water_depth is packed (the dormancy decision would read a plane
    the edit has not reached)."""
    sim = _sim()
    ordered = _queue(_edits() + [F("water_depth", R.TILE, (5, 5), 0.1)]).take_ordered()
    pack, reason = fe.pack_for_device(sim.gmap, ordered, sim.draws)
    assert pack is None and "water_depth" in reason


def test_an_edit_past_the_devices_integer_range_falls_back():
    """PROPERTY: an edit whose integer product or contribution would leave the
    device's 128 / 64-bit arithmetic is not packed (a reason, the host flush),
    never truncated on the device. BREAKS IF the pack's range check is dropped."""
    sim = _sim()
    big = F("gas_energy", R.TILE, (5, 5), 0.0, M.ADD, amount_q=1 << 60)
    pack, reason = fe.pack_for_device(sim.gmap, _queue([big]).take_ordered(), sim.draws)
    assert pack is None and "range" in reason


def test_the_pack_raises_the_host_refusals():
    """PROPERTY: a malformed edit is refused by the pack exactly as the host
    flush refuses it (a CONSERVATIVE slice through the gas policy: the
    AssertionError; a gas edit without its channel: the ValueError), so a
    refusal stays a host refusal on the resident path. BREAKS IF the pack
    skips the door's checks."""
    sim = _sim()
    cons = F("gas", R.TILE, (5, 5), 0.1, channel=int(O2))
    with pytest.raises(AssertionError, match="CONSERVATIVE"):
        fe.pack_for_device(sim.gmap, _queue([cons]).take_ordered(), sim.draws)
    with pytest.raises(AssertionError, match="CONSERVATIVE"):
        _queue([cons]).flush(sim.gmap, sim.draws)
    nochan = F("gas", R.TILE, (5, 5), 0.1)
    with pytest.raises(ValueError, match="channel"):
        fe.pack_for_device(sim.gmap, _queue([nochan]).take_ordered(), sim.draws)
    with pytest.raises(ValueError, match="channel"):
        _queue([nochan]).flush(sim.gmap, sim.draws)


def test_off_the_resident_path_slot_6b_flushes_on_the_host():
    """PROPERTY: the device queue serves 6b only when slot 7 runs resident
    (residency on AND a CUDA build): on the CPU build, residency on or off, the
    runner does not claim the queue and the step flushes it on the host.
    BREAKS IF serves_edit_queue ignores the backend (a CPU tick would pack and
    never apply)."""
    from simulation import physics_runner as pr
    sim = _sim()
    if getattr(bp, "HAS_CUDA", False):
        pytest.skip("the CPU build's contract")
    for on in (False, True):
        pr.set_residency(on)
        try:
            assert not sim.physics_runner.serves_edit_queue()
        finally:
            pr.set_residency(False)
    sim.edit(F("heat", R.TILE, (5, 5), 1.0))
    sim.set_paused(False)
    sim.step()
    assert len(sim.edit_queue) == 0
    assert sim.physics_runner._edit_pack is None
