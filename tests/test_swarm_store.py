"""Swarm store lifecycle — arc #63 P2 (docs/swarm_P2_impl.md §11, T12-T21).

The store (per-species SoA arrays on GameMap), spawn/reclaim (the k-th
request takes the k-th lowest free slot, atomic refusal on overflow), and
the invariant checker. Every test names the property it protects and the
change that must break it (the binding test standard) — none pins the
roster's size/order or the species set's size.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_swarm_store.py -q
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402

from config import CFG, Namespace  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import swarm, swarm_fixed  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402


def _mini_level() -> LevelData:
    """An 8x8 hull-walled room, interior air carved out — a minimal,
    synthetic level (no asset files) for store/lifecycle tests that need no
    physics build."""
    h = w = 8
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:7, 1:7] = 4
    return LevelData(name="swarm_store_test", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _swarm_cfg_namespace(**overrides) -> Namespace:
    """A ``[swarm.<sp>]`` Namespace (the shape ``CFG.reload()`` produces)
    covering EVERY current ``SWARM_SPECIES`` with the same values — never
    pins the species set's size."""
    base = {"max_units": overrides.pop("max_units", 8192),
            "hp_max": overrides.pop("hp_max", 1.0)}
    return Namespace({sp: dict(base) for sp in swarm.SWARM_SPECIES})


def _snapshot(g, sp: str):
    """Every byte of one species' store + the shared scalars — for
    before/after equality checks around an operation that must refuse."""
    cols = {c.name: swarm.swarm_column(g, sp, c.name).copy()
            for c in swarm.ROSTER}
    return (cols, dict(g.swarm_host_dirty), g.swarm_next_unit_id.copy(),
            int(getattr(g, swarm.high_water_attr(sp))[0]))


def _assert_unchanged(before, after):
    cols_a, dirty_a, nid_a, hw_a = before
    cols_b, dirty_b, nid_b, hw_b = after
    for name in cols_a:
        assert np.array_equal(cols_a[name], cols_b[name]), (
            f"column '{name}' changed despite a refused operation")
    assert dirty_a == dirty_b, "host_dirty changed despite a refused operation"
    assert np.array_equal(nid_a, nid_b), "next_unit_id changed despite a refusal"
    assert hw_a == hw_b, "high_water changed despite a refusal"


# ---------------------------------------------------------------------------
# T12 — the store is resident and starts empty
# ---------------------------------------------------------------------------

def test_store_arrays_are_resident_and_start_empty():
    """Every ROSTER column and each per-env scalar exists on GameMap with
    shape (N_ENV, max_units) / (N_ENV,), the roster dtype, C-contiguous and
    little-endian, starts in the empty form, and is born through residency
    (GameMap._RESIDENT_FIELD_NAMES). gmap.swarm_host_dirty has a key per
    species, all False on a fresh map, and is not itself resident.

    Must break if a column is mirror-only, has the wrong dtype, or the
    dirty flag becomes a synced array."""
    g = GameMap(_mini_level())

    for sp in swarm.SWARM_SPECIES:
        row = g.swarm_species.row(sp)
        hw_attr = swarm.high_water_attr(sp)
        hw_arr = getattr(g, hw_attr)
        assert hw_arr.shape == (swarm.N_ENV,)
        assert hw_arr.dtype == np.int32
        assert np.all(hw_arr == 0)
        assert hw_attr in GameMap._RESIDENT_FIELD_NAMES

        for col in swarm.ROSTER:
            arr = swarm.swarm_column(g, sp, col.name)
            assert arr.shape == (swarm.N_ENV, row.max_units)
            assert arr.dtype == col.dtype
            assert arr.flags["C_CONTIGUOUS"]
            assert arr.dtype.byteorder in ("=", "<", "|"), (
                f"column '{col.name}' is not native/little-endian")
            assert np.all(arr == 0)
            assert swarm.store_attr(sp, col.name) in GameMap._RESIDENT_FIELD_NAMES

    nid_arr = g.swarm_next_unit_id
    assert nid_arr.shape == (swarm.N_ENV,)
    assert nid_arr.dtype == np.int32
    assert np.all(nid_arr == swarm.FIRST_UNIT_ID)
    assert swarm.NEXT_UNIT_ID_ATTR in GameMap._RESIDENT_FIELD_NAMES

    assert set(g.swarm_host_dirty) == set(swarm.SWARM_SPECIES)
    assert all(v is False for v in g.swarm_host_dirty.values())
    assert swarm.HOST_DIRTY_ATTR not in GameMap._RESIDENT_FIELD_NAMES


# ---------------------------------------------------------------------------
# T13 — the k-th request takes the k-th lowest free slot
# ---------------------------------------------------------------------------

def test_spawn_takes_lowest_free_slots_in_request_order():
    """Fresh store -> 0..k-1; after reclaims, a batch of m gets
    sorted(free_before)[:m]. Must break under append-at-high_water, LIFO,
    or compaction."""
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)
    sp = swarm.SWARM_SPECIES[0]

    slots = su.spawn(sp, [0] * 5, [0] * 5, [0] * 5)
    assert list(slots) == [0, 1, 2, 3, 4]

    su.reclaim(sp, np.array([1, 3]))
    free_before = sorted(np.flatnonzero(
        su.column(sp, "valid")[0] == 0).tolist())
    slots2 = su.spawn(sp, [0, 0, 0], [0, 0, 0], [0, 0, 0])
    assert list(slots2) == free_before[:3]


# ---------------------------------------------------------------------------
# T14 — model-based: a seeded random interleaving vs. a reference model
# ---------------------------------------------------------------------------

def test_lifecycle_matches_a_reference_model_and_invariants_hold(monkeypatch):
    """After a seeded random interleaving of spawn/reclaim batches, the
    store equals a reference model, and check_invariants(gmap) == [] after
    EVERY operation. Also: high_water never decreases and == 1 + max slot
    ever occupied; every assigned unit_id is unique and strictly
    increasing; next_unit_id == FIRST_UNIT_ID + total spawned; a reused
    slot gets a new id; every non-empty spawn/reclaim sets host_dirty[sp].

    Must break under id==slot, id reuse, high_water recomputed from live
    slots, a lifecycle op that violates an invariant, or a write that skips
    the dirty flag."""
    max_units = 24
    monkeypatch.setattr(CFG, "swarm", _swarm_cfg_namespace(max_units=max_units))
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)
    sp = swarm.SWARM_SPECIES[0]

    rng = np.random.default_rng(20260924)
    slot_to_uid: dict = {}      # the reference model
    next_id = swarm.FIRST_UNIT_ID
    max_slot_ever = -1
    total_spawned = 0
    prev_hw = 0

    assert swarm.check_invariants(g) == []
    for _ in range(200):
        g.swarm_host_dirty[sp] = False   # so this op's write is observable
        free = sorted(set(range(max_units)) - set(slot_to_uid))
        do_spawn = (not slot_to_uid) or (rng.random() < 0.6 and free)

        if do_spawn and free:
            k = int(rng.integers(1, min(4, len(free)) + 1))
            xs = rng.integers(0, 8 << swarm_fixed.FP_SHIFT, size=k).tolist()
            ys = rng.integers(0, 8 << swarm_fixed.FP_SHIFT, size=k).tolist()
            heads = rng.integers(-(1 << 20), 1 << 20, size=k).tolist()
            expect_slots = free[:k]
            slots = su.spawn(sp, xs, ys, heads)
            assert list(slots) == expect_slots
            for i, s in enumerate(slots):
                uid = next_id + i
                assert uid not in slot_to_uid.values(), "unit_id reused"
                slot_to_uid[int(s)] = uid
            next_id += k
            total_spawned += k
            max_slot_ever = max(max_slot_ever, int(max(slots)))
        elif slot_to_uid:
            live_slots = sorted(slot_to_uid)
            m = int(rng.integers(1, min(3, len(live_slots)) + 1))
            reclaim_slots = rng.choice(live_slots, size=m, replace=False)
            su.reclaim(sp, reclaim_slots)
            for s in reclaim_slots:
                del slot_to_uid[int(s)]
        else:
            continue   # nothing free and nothing live -- unreachable given max_units > 0

        assert g.swarm_host_dirty[sp] is True, (
            "a non-empty spawn/reclaim must set host_dirty[sp]")

        expect_hw = max_slot_ever + 1 if max_slot_ever >= 0 else 0
        hw = su.high_water(sp)
        assert hw == expect_hw
        assert hw >= prev_hw, "high_water decreased"
        prev_hw = hw
        assert int(g.swarm_next_unit_id[0]) == next_id
        assert next_id == swarm.FIRST_UNIT_ID + total_spawned

        assert swarm.check_invariants(g) == []

    # Final cross-check: the live store matches the reference model exactly.
    valid = su.column(sp, "valid")[0]
    uid_col = su.column(sp, "unit_id")[0]
    live_slots_actual = set(np.flatnonzero(valid == 1).tolist())
    assert live_slots_actual == set(slot_to_uid)
    for s, uid in slot_to_uid.items():
        assert int(uid_col[s]) == uid


# ---------------------------------------------------------------------------
# T15 — reclaim writes the all-zero empty-slot form
# ---------------------------------------------------------------------------

def test_reclaim_writes_the_all_zero_empty_slot_form():
    """A reclaimed slot's every column is byte-equal to a never-spawned
    slot's (valid == 0, ai_state == AI_DEAD). Must break under a reclaim
    that only clears `valid`."""
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)
    sp = swarm.SWARM_SPECIES[0]
    never_spawned_slot = 6

    slots = su.spawn(sp, [100], [200], [300], faction=7)
    s = int(slots[0])
    assert s != never_spawned_slot
    su.reclaim(sp, np.array([s]))

    for col in swarm.ROSTER:
        arr = su.column(sp, col.name)[0]
        assert int(arr[s]) == 0, f"column '{col.name}' not zeroed by reclaim"
        assert int(arr[s]) == int(arr[never_spawned_slot])
    assert int(su.column(sp, "ai_state")[0, s]) == swarm.AI_DEAD


# ---------------------------------------------------------------------------
# T16 — atomic refusal + bad-input validation
# ---------------------------------------------------------------------------

def test_atomic_refusal_and_bad_inputs(monkeypatch):
    """Overfilling raises SwarmFull; bad inputs raise (out-of-grid pos, bad
    env, unknown species, reclaim of an empty/duplicate/>=hw slot). R7:
    with next_unit_id == INT32_MAX - 1, spawning 1 succeeds and the NEXT
    spawn of 1 raises; faction of 2**31 / -2**31-1 raises; a heading
    outside int32 raises; bool/float/uint64 arrays raise TypeError. In
    every refusal case all arrays, scalars and host_dirty are
    byte-identical to before.

    Must break under partial fill/write, a counter that can reach
    INT32_MAX + 1, or silent int32 truncation of faction."""
    monkeypatch.setattr(CFG, "swarm", _swarm_cfg_namespace(max_units=3))
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)
    sp = swarm.SWARM_SPECIES[0]
    w_bound = int(g._w) << swarm_fixed.FP_SHIFT

    def expect_no_write(exc_type, fn):
        before = _snapshot(g, sp)
        with pytest.raises(exc_type):
            fn()
        _assert_unchanged(before, _snapshot(g, sp))

    # capacity: 3 slots -- requesting 4 refuses atomically.
    expect_no_write(swarm.SwarmFull, lambda: su.spawn(sp, [0] * 4, [0] * 4, [0] * 4))

    # out-of-grid position.
    expect_no_write(ValueError, lambda: su.spawn(sp, [w_bound], [0], [0]))
    expect_no_write(ValueError, lambda: su.spawn(sp, [-1], [0], [0]))

    # bad env.
    expect_no_write(ValueError, lambda: su.spawn(sp, [0], [0], [0], env=1))
    expect_no_write(ValueError, lambda: su.spawn(sp, [0], [0], [0], env=-1))

    # unknown species.
    expect_no_write(ValueError, lambda: su.spawn("no_such_species", [0], [0], [0]))

    # reclaim of an empty slot (nothing spawned yet -> a caller bug).
    expect_no_write(ValueError, lambda: su.reclaim(sp, np.array([0])))

    # faction outside int32.
    expect_no_write(ValueError, lambda: su.spawn(sp, [0], [0], [0], faction=2 ** 31))
    expect_no_write(ValueError,
                    lambda: su.spawn(sp, [0], [0], [0], faction=-(2 ** 31) - 1))

    # heading outside int32.
    expect_no_write(ValueError, lambda: su.spawn(sp, [0], [0], [2 ** 31]))

    # bool / float / uint64 inputs raise TypeError (R7) -- raw Q16 only.
    expect_no_write(TypeError, lambda: su.spawn(sp, np.array([True]), [0], [0]))
    expect_no_write(TypeError, lambda: su.spawn(sp, [0.5], [0], [0]))
    expect_no_write(TypeError,
                    lambda: su.spawn(sp, np.array([1], dtype=np.uint64), [0], [0]))
    expect_no_write(TypeError, lambda: su.spawn(sp, [0], [0], [0], faction=True))

    # A real spawn now, to exercise reclaim's duplicate/>=hw refusals.
    slots = su.spawn(sp, [0], [0], [0])
    s = int(slots[0])
    expect_no_write(ValueError, lambda: su.reclaim(sp, np.array([s, s])))       # duplicate
    expect_no_write(ValueError, lambda: su.reclaim(sp, np.array([99])))         # >= hw
    expect_no_write(TypeError, lambda: su.reclaim(sp, np.array([True])))        # bool
    expect_no_write(TypeError, lambda: su.reclaim(sp, np.array([0.0])))         # float

    # R7: the id guard is `next_unit_id + k <= INT32_MAX`, checked before
    # any write -- spawning at the boundary succeeds once, then refuses.
    g.swarm_next_unit_id[0] = swarm.INT32_MAX - 1
    boundary_slots = su.spawn(sp, [0], [0], [0])
    assert int(su.column(sp, "unit_id")[0, boundary_slots[0]]) == swarm.INT32_MAX - 1
    assert int(g.swarm_next_unit_id[0]) == swarm.INT32_MAX
    expect_no_write(ValueError, lambda: su.spawn(sp, [0], [0], [0]))


# ---------------------------------------------------------------------------
# T17 — heading is canonical at every write
# ---------------------------------------------------------------------------

def test_heading_is_canonical_and_wrap_is_idempotent():
    """The stored heading lies in (-PI_Q16, PI_Q16] and is congruent to the
    input mod HEADING_PERIOD_Q16; wrap is idempotent; scalar and array
    forms agree. Inputs include the Philox draw-range endpoints (411774,
    411775) and INT32_MIN/MAX.

    Must break if a raw heading is stored, the period is quantize(2*pi)
    (411775), the wrap runs in int32, or the scalar/array forms diverge."""
    P = swarm_fixed.PI_Q16
    PERIOD = swarm_fixed.HEADING_PERIOD_Q16
    draw_range = 411775   # philox32.TWO_PI_Q16 -- distinct from PERIOD on purpose
    inputs = [P, -P, 3 * P, -3 * P, PERIOD, draw_range, 0, 1, -1,
             swarm.INT32_MIN, swarm.INT32_MAX, 0 - P, draw_range - P]

    for h in inputs:
        wrapped_scalar = swarm_fixed.wrap_heading_q16(h)
        wrapped_array = int(
            swarm_fixed.wrap_heading_q16(np.array([h], dtype=np.int64))[0])
        assert wrapped_scalar == wrapped_array, (
            f"scalar/array forms disagree for h={h}")
        assert -P < wrapped_scalar <= P
        assert (wrapped_scalar - h) % PERIOD == 0
        assert swarm_fixed.wrap_heading_q16(wrapped_scalar) == wrapped_scalar

    assert swarm_fixed.wrap_heading_q16(PERIOD) == 0
    assert swarm_fixed.wrap_heading_q16(draw_range) == 1
    assert swarm_fixed.wrap_heading_q16(swarm.INT32_MAX) == 82237
    assert swarm_fixed.wrap_heading_q16(swarm.INT32_MIN) == -82238

    # End-to-end through spawn: every stored heading is the wrapped value.
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)
    sp = swarm.SWARM_SPECIES[0]
    slots = su.spawn(sp, [0] * len(inputs), [0] * len(inputs), inputs)
    stored = su.column(sp, "heading")[0]
    for slot, h in zip(slots, inputs):
        assert int(stored[slot]) == swarm_fixed.wrap_heading_q16(h)


# ---------------------------------------------------------------------------
# T18 — the lifecycle draws no randomness
# ---------------------------------------------------------------------------

def test_lifecycle_draws_no_randomness():
    """spawn/reclaim leave sim.rng.bit_generator.state unchanged (the
    test_w6_armory.py canary idiom). Must break if spawn draws from
    sim.rng."""
    sim = Simulation(_mini_level(), seed=12345, breach_physics=None,
                     enable_recorder=False)
    fresh = np.random.default_rng(12345)
    assert sim.rng.bit_generator.state == fresh.bit_generator.state

    sp = swarm.SWARM_SPECIES[0]
    slots = sim.swarm.spawn(sp, [0, 1 << 16], [0, 1 << 16], [0, 100])
    sim.swarm.reclaim(sp, slots[:1])
    assert sim.rng.bit_generator.state == fresh.bit_generator.state


# ---------------------------------------------------------------------------
# T19 — picklability
# ---------------------------------------------------------------------------

def test_store_and_species_table_are_picklable():
    """A pickle round-trip of every array in SWARM_RESIDENT_NAMES, and of
    gmap.swarm_species, returns an equal object. Must break under a
    runtime-generated row class, a lambda, or another unpicklable table
    member."""
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)
    sp = swarm.SWARM_SPECIES[0]
    su.spawn(sp, [10], [20], [30])

    for name in swarm.SWARM_RESIDENT_NAMES:
        arr = getattr(g, name)
        restored = pickle.loads(pickle.dumps(arr))
        assert np.array_equal(arr, restored)
        assert arr.dtype == restored.dtype

    table = g.swarm_species
    restored_table = pickle.loads(pickle.dumps(table))
    assert restored_table == table
    assert restored_table.rows == table.rows
    for row, restored_row in zip(table.rows, restored_table.rows):
        assert row == restored_row


# ---------------------------------------------------------------------------
# T20 — check_invariants is not vacuous
# ---------------------------------------------------------------------------

def test_check_invariants_is_not_vacuous(monkeypatch):
    """On a freshly spawned scenario it returns []; each single deliberate
    corruption below is reported (non-empty). Must break if any one of the
    listed checks is dropped from check_invariants."""
    monkeypatch.setattr(CFG, "swarm", _swarm_cfg_namespace(max_units=8))
    sp = swarm.SWARM_SPECIES[0]

    def fresh_store():
        g = GameMap(_mini_level())
        su = swarm.SwarmUnits(g)
        su.spawn(sp, [0, 1 << 16], [0, 1 << 16], [0, 0])
        return g

    g = fresh_store()
    assert swarm.check_invariants(g) == [], "the uncorrupted fixture must be clean"

    def col(gmap, name):
        return swarm.swarm_column(gmap, sp, name)[0]

    # next_unit_id != FIRST with every high_water == 0.
    g2 = fresh_store()
    for c in swarm.ROSTER:
        col(g2, c.name)[:] = 0
    getattr(g2, swarm.high_water_attr(sp))[0] = 0
    g2.swarm_next_unit_id[0] = 5
    assert swarm.check_invariants(g2) != []

    # next_unit_id == FIRST with high_water > 0.
    g3 = fresh_store()
    g3.swarm_next_unit_id[0] = swarm.FIRST_UNIT_ID
    assert swarm.check_invariants(g3) != []

    # a duplicate live unit_id.
    g4 = fresh_store()
    col(g4, "unit_id")[1] = int(col(g4, "unit_id")[0])
    assert swarm.check_invariants(g4) != []

    # a live unit_id >= next_unit_id.
    g5 = fresh_store()
    col(g5, "unit_id")[0] = int(g5.swarm_next_unit_id[0]) + 5
    assert swarm.check_invariants(g5) != []

    # a non-zero freed/never-spawned slot.
    g6 = fresh_store()
    col(g6, "pos_x")[5] = 42
    assert swarm.check_invariants(g6) != []

    # a non-zero (valid) slot >= high_water.
    g7 = fresh_store()
    col(g7, "valid")[5] = 1
    assert swarm.check_invariants(g7) != []

    # valid == 2 (outside {0, 1}).
    g8 = fresh_store()
    col(g8, "valid")[0] = 2
    assert swarm.check_invariants(g8) != []

    # a live heading == -PI_Q16 (outside (-PI_Q16, PI_Q16]).
    g9 = fresh_store()
    col(g9, "heading")[0] = -swarm_fixed.PI_Q16
    assert swarm.check_invariants(g9) != []

    # a live slot with ai_state == AI_DEAD.
    g10 = fresh_store()
    col(g10, "ai_state")[0] = swarm.AI_DEAD
    assert swarm.check_invariants(g10) != []

    # high_water > capacity (max_units).
    g11 = fresh_store()
    getattr(g11, swarm.high_water_attr(sp))[0] = 999
    assert swarm.check_invariants(g11) != []


# ---------------------------------------------------------------------------
# T21 — name hygiene
# ---------------------------------------------------------------------------

def test_check_store_names_rejects_bad_names_and_collisions():
    """Every SWARM_SPECIES name matches ^[a-z][a-z0-9]*$; every generated
    store attribute name (== every .npz key) and the fixed names are
    pairwise distinct. Must break if the regex or the uniqueness check is
    removed."""
    swarm.check_store_names(swarm.SWARM_SPECIES, swarm.ROSTER)   # the real tuples pass

    with pytest.raises(ValueError):
        swarm.check_store_names(("bad_name",), swarm.ROSTER)     # underscore

    with pytest.raises(ValueError):
        swarm.check_store_names(("Larva",), swarm.ROSTER)        # uppercase

    # a hypothetical species "next" collides with swarm_next_unit_id.
    with pytest.raises(ValueError):
        swarm.check_store_names(swarm.SWARM_SPECIES + ("next",), swarm.ROSTER)

    # a column named "high_water" would alias swarm_<sp>_high_water.
    bad_roster = swarm.ROSTER + (swarm.Column("high_water", np.int32),)
    with pytest.raises(ValueError):
        swarm.check_store_names(swarm.SWARM_SPECIES, bad_roster)
