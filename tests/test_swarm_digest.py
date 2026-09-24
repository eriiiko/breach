"""Swarm digest section — arc #63 P2 (docs/swarm_P2_impl.md §11, T1-T11).

The presence-gated ``__swarm__`` fold into ``tick_digest``: absence
transparency (a swarm-free run hashes zero extra bytes, so every existing
golden stays byte-identical), the section is not vacuous (every column and
scalar moves the digest), capacity moves nothing, presence is monotone, the
strict carrier, the A/B harness locator, R1's pre-P2 compatibility, R2's
integer rendering, and R6's env-0-only guard.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_swarm_digest.py -q
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402

import field_ab_harness as fab  # noqa: E402
from config import CFG, Namespace  # noqa: E402
from field_digest import DIGEST_FIELDS, field_digest, tick_digest  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import swarm, swarm_fixed  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.swarm import SWARM_DIGEST_KEY  # noqa: E402


def _mini_level() -> LevelData:
    h = w = 8
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:7, 1:7] = 4
    return LevelData(name="swarm_digest_test", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _swarm_cfg_namespace(**overrides) -> Namespace:
    base = {"max_units": overrides.pop("max_units", 8192),
            "hp_max": overrides.pop("hp_max", 1.0)}
    return Namespace({sp: dict(base) for sp in swarm.SWARM_SPECIES})


def _mini_snapshot(h: int = 4, w: int = 4) -> dict:
    """A deterministic synthetic snapshot carrying every DIGEST_FIELDS array
    (field_digest checks names + dtypes, not shapes) plus a fixed unit hash
    -- the ``test_entity_digest.py::_mini_snapshot`` idiom, reproduced here
    so this file stays self-contained."""
    snap = {}
    for j, (name, dtype) in enumerate(DIGEST_FIELDS):
        shape = (2, h, w) if name == "gas" else (h, w)
        n = int(np.prod(shape))
        arr = (np.arange(n, dtype=np.int64) * 7 + j).reshape(shape)
        snap[name] = (arr % 2).astype(bool) if dtype == "bool" \
            else arr.astype(dtype)
    snap["__unit_state__"] = {"units": [], "events": [], "hash": "f" * 64}
    return snap


def _fake_entity_carrier(n: int = 1) -> dict:
    """A structurally-valid ``__entity__`` carrier stand-in — tick_digest
    only folds its bytes/counts, never validates entity content."""
    return {"n_entities": n, "records": (b"0|e1|Foo\nalive|1\n\n",) * n,
            "signals": (), "registry_hash": "deadbeef"}


def _pre_p2_tick_digest(snapshot: dict) -> str:
    """The pre-P2 ``tick_digest``, frozen VERBATIM (field fold + unit hash
    + the A4 entity/signal fold, NO swarm fold) as the dormancy reference —
    if the swarm fold ever perturbs a swarm-free hash, this catches it
    without any committed golden moving (the ``_pre_a4_tick_digest``
    idiom, ``test_entity_digest.py``)."""
    from field_ab_harness import UNIT_DIGEST_KEY
    from simulation.entities.serialize import (
        ENTITY_DIGEST_KEY, entity_section_bytes, signal_section_bytes)
    fd = field_digest(snapshot)
    unit_hash = ""
    if UNIT_DIGEST_KEY in snapshot:
        unit_hash = snapshot[UNIT_DIGEST_KEY]["hash"]
    h = hashlib.blake2b(digest_size=32)
    h.update(fd.encode("ascii"))
    h.update(b"|")
    h.update(unit_hash.encode("ascii"))
    carrier = snapshot.get(ENTITY_DIGEST_KEY)
    if carrier is not None and carrier["n_entities"] > 0:
        eh = hashlib.blake2b(entity_section_bytes(carrier),
                             digest_size=32).hexdigest()
        sh = hashlib.blake2b(signal_section_bytes(carrier),
                             digest_size=32).hexdigest()
        h.update(b"|__entity__|")
        h.update(eh.encode("ascii"))
        h.update(b"|__signals__|")
        h.update(sh.encode("ascii"))
    return h.hexdigest()


def _swarm_present_carrier():
    """A real swarm-present carrier with hw > live count (a reclaimed slot
    inside [0, high_water)), for the section's byte sensitivity tests."""
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)
    sp = swarm.SWARM_SPECIES[0]
    su.spawn(sp, [0, 1 << 16, 2 << 16], [0, 0, 0], [0, 100, -100])
    su.reclaim(sp, np.array([1]))   # a reclaimed valid=0 slot inside [0, hw)
    return swarm.swarm_carrier(g), sp


# ---------------------------------------------------------------------------
# T1 — every existing golden stays byte-identical
# ---------------------------------------------------------------------------

def test_canonical_scenario_still_matches_the_sanctioned_golden():
    """The canonical (swarm-free) A/B scenario, captured through
    capture_trajectory (which NOW always writes the __swarm__ carrier,
    present == False), still reproduces GOLDEN_AGGREGATE. The full existing
    suite (test_w6_armory's own golden + RNG canary, B6, vent/B1/B2) is the
    rest of this oracle -- checked by the gate's full-suite run, not
    re-derived here.

    Must break if the fold is unconditional, capacity is hashed, or store
    arrays join DIGEST_FIELDS/SIM_FIELDS."""
    from _xarch_perfield_digest import GOLDEN_AGGREGATE
    from field_digest import trajectory_digest
    traj = fab.capture_trajectory(make_sim=fab.default_scenario_sim, n_steps=30)
    assert all(not snap[SWARM_DIGEST_KEY]["present"] for snap in traj)
    assert trajectory_digest(traj) == GOLDEN_AGGREGATE


# ---------------------------------------------------------------------------
# T2 — absence transparency
# ---------------------------------------------------------------------------

def test_absence_transparency_matches_the_pre_p2_reference():
    """tick_digest of (a) a snapshot with no __swarm__ key, (b) one with a
    present=False carrier, and (c) the frozen pre-P2 tick_digest, are all
    equal -- on a synthetic snapshot both without and with an (unrelated)
    entity carrier. Must break if the marker or bytes are hashed when
    absent."""
    for with_entity in (False, True):
        base = _mini_snapshot()
        if with_entity:
            from simulation.entities.serialize import ENTITY_DIGEST_KEY
            base[ENTITY_DIGEST_KEY] = _fake_entity_carrier()

        d_nokey = tick_digest(dict(base))
        d_pre = _pre_p2_tick_digest(dict(base))

        absent = dict(base)
        absent[SWARM_DIGEST_KEY] = {"present": False,
                                    "next_unit_id": swarm.FIRST_UNIT_ID,
                                    "blocks": {}, "species_hash": ""}
        d_absent = tick_digest(absent)

        assert d_nokey == d_pre == d_absent, (
            f"absence transparency broken (with_entity={with_entity})")


# ---------------------------------------------------------------------------
# T3 — the section is not vacuous
# ---------------------------------------------------------------------------

def test_section_is_not_vacuous_every_column_and_scalar_moves_the_digest():
    """For every ROSTER column, perturbing one slot inside [0, hw)
    (including the reclaimed valid=0 slot) changes tick_digest, and
    restoring it restores the digest; the same holds for next_unit_id and
    each species' high_water; field_digest never changes.

    Must break if a column is dropped from the encoder, only live slots
    are hashed, or a scalar is omitted."""
    carrier, sp = _swarm_present_carrier()
    base_snap = _mini_snapshot()
    base_snap[SWARM_DIGEST_KEY] = carrier
    base_field = field_digest(base_snap)
    base_tick = tick_digest(base_snap)

    reclaimed_slot = 1   # inside [0, hw); all-zero (reclaimed)
    for col in swarm.ROSTER:
        arr = carrier["blocks"][sp]["columns"][col.name]
        orig = int(arr[reclaimed_slot])
        assert orig == 0, "fixture assumption: the reclaimed slot is zero"
        arr[reclaimed_slot] = 1

        assert tick_digest(base_snap) != base_tick, (
            f"perturbing column '{col.name}' did not move tick_digest")
        assert field_digest(base_snap) == base_field, (
            "field_digest must never fold swarm-section data")

        arr[reclaimed_slot] = orig
        assert tick_digest(base_snap) == base_tick

    orig_nid = carrier["next_unit_id"]
    carrier["next_unit_id"] = orig_nid + 1
    assert tick_digest(base_snap) != base_tick
    carrier["next_unit_id"] = orig_nid
    assert tick_digest(base_snap) == base_tick

    orig_hw = carrier["blocks"][sp]["high_water"]
    carrier["blocks"][sp]["high_water"] = orig_hw + 1
    assert tick_digest(base_snap) != base_tick
    carrier["blocks"][sp]["high_water"] = orig_hw
    assert tick_digest(base_snap) == base_tick


# ---------------------------------------------------------------------------
# T4 — capacity moves nothing
# ---------------------------------------------------------------------------

def test_capacity_moves_nothing(monkeypatch):
    """Identical spawns with max_units 8192 vs 64 give identical section
    bytes and tick_digest; a value written at a slot >= high_water leaves
    the digest unchanged. Must break if full capacity is serialized."""
    def build(max_units):
        monkeypatch.setattr(CFG, "swarm", _swarm_cfg_namespace(max_units=max_units))
        g = GameMap(_mini_level())
        su = swarm.SwarmUnits(g)
        sp = swarm.SWARM_SPECIES[0]
        su.spawn(sp, [0, 1 << 16], [0, 0], [0, 12345])
        return g, su, sp

    g_big, su_big, sp = build(8192)
    g_small, _, _ = build(64)

    bytes_big = swarm.swarm_section_bytes(swarm.swarm_carrier(g_big))
    bytes_small = swarm.swarm_section_bytes(swarm.swarm_carrier(g_small))
    assert bytes_big == bytes_small

    snap_big = _mini_snapshot()
    snap_big[SWARM_DIGEST_KEY] = swarm.swarm_carrier(g_big)
    snap_small = _mini_snapshot()
    snap_small[SWARM_DIGEST_KEY] = swarm.swarm_carrier(g_small)
    assert tick_digest(snap_big) == tick_digest(snap_small)

    baseline = tick_digest(snap_big)
    hw = su_big.high_water(sp)
    su_big.column(sp, "pos_x")[0, hw + 5] = 999999
    snap_big2 = _mini_snapshot()
    snap_big2[SWARM_DIGEST_KEY] = swarm.swarm_carrier(g_big)
    assert tick_digest(snap_big2) == baseline


# ---------------------------------------------------------------------------
# T5 — presence is monotone
# ---------------------------------------------------------------------------

def test_presence_is_monotone_after_reclaim_to_zero_live():
    """Spawn 1 and reclaim it -> live_count == 0 yet present, and the
    digest != the swarm-free digest. Must break if presence is keyed on
    live count."""
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)
    sp = swarm.SWARM_SPECIES[0]
    slots = su.spawn(sp, [0], [0], [0])
    su.reclaim(sp, slots)
    assert su.live_count(sp) == 0
    assert su.present() is True

    snap = _mini_snapshot()
    snap[SWARM_DIGEST_KEY] = swarm.swarm_carrier(g)
    snap_free = _mini_snapshot()
    snap_free[SWARM_DIGEST_KEY] = swarm.swarm_carrier(GameMap(_mini_level()))
    assert tick_digest(snap) != tick_digest(snap_free)


# ---------------------------------------------------------------------------
# T6 — strict carrier
# ---------------------------------------------------------------------------

def test_strict_carrier_presence_rule():
    """A present gmap plus a snapshot without __swarm__ raises KeyError; a
    swarm-free gmap accepts; capture_trajectory writes a carrier on every
    tick. Must break if a capture path skips the carrier."""
    g_present = GameMap(_mini_level())
    swarm.SwarmUnits(g_present).spawn(swarm.SWARM_SPECIES[0], [0], [0], [0])
    with pytest.raises(KeyError):
        swarm.require_swarm_carrier(g_present, {})

    g_free = GameMap(_mini_level())
    swarm.require_swarm_carrier(g_free, {})   # accepts -- no raise

    traj = fab.capture_trajectory(make_sim=fab.default_scenario_sim, n_steps=3)
    assert all(SWARM_DIGEST_KEY in snap for snap in traj)


# ---------------------------------------------------------------------------
# T7 — the A/B harness locator
# ---------------------------------------------------------------------------

def test_harness_self_match_and_swarm_locator():
    """capture_trajectory(make_sim=swarm_scenario_sim) twice ->
    assert_trajectories_match at tol 0. A one-slot perturbation is reported
    with species/column/slot/both unit_ids. Comparing two present carriers
    never raises. Must break if the locator treats the carrier as opaque,
    ignores it, or copies the entity `==` fast path."""
    traj_a = fab.capture_trajectory(make_sim=fab.swarm_scenario_sim, n_steps=5)
    traj_b = fab.capture_trajectory(make_sim=fab.swarm_scenario_sim, n_steps=5)
    assert traj_a[-1][SWARM_DIGEST_KEY]["present"]
    fab.assert_trajectories_match(traj_a, traj_b, tol=0.0)

    # comparing two present carriers never raises (no ambiguous truth value).
    assert fab._diff_swarm_state(
        0, traj_a[-1][SWARM_DIGEST_KEY], traj_b[-1][SWARM_DIGEST_KEY]) == []

    perturbed = [dict(s) for s in traj_b]
    carrier = dict(perturbed[-1][SWARM_DIGEST_KEY])
    carrier["blocks"] = {sp: dict(block) for sp, block in carrier["blocks"].items()}
    sp0 = next(iter(carrier["blocks"]))
    carrier["blocks"][sp0]["columns"] = dict(carrier["blocks"][sp0]["columns"])
    pos_x = carrier["blocks"][sp0]["columns"]["pos_x"].copy()
    pos_x[0] = pos_x[0] + 12345
    carrier["blocks"][sp0]["columns"]["pos_x"] = pos_x
    perturbed[-1][SWARM_DIGEST_KEY] = carrier

    diffs = fab.diff_trajectories(traj_a, perturbed, tol=0.0)
    located = [d for d in diffs if "pos_x" in d and "unit_id" in d and sp0 in d]
    assert located, f"perturbation not located precisely: {diffs}"


# ---------------------------------------------------------------------------
# T8 — the swarm is inert on fields (P2 scope)
# ---------------------------------------------------------------------------

def test_swarm_is_inert_on_fields():
    """Per tick, field_digest(swarm_scenario) == field_digest(default_scenario)
    while the tick digests differ (only the swarm section moved). EXPECTED
    TO BREAK AT P6 (food depletion writes fields) -- retire this test
    there."""
    traj_default = fab.capture_trajectory(make_sim=fab.default_scenario_sim, n_steps=5)
    traj_swarm = fab.capture_trajectory(make_sim=fab.swarm_scenario_sim, n_steps=5)
    for sa, sb in zip(traj_default, traj_swarm):
        assert field_digest(sa) == field_digest(sb)
        assert tick_digest(sa) != tick_digest(sb)


# ---------------------------------------------------------------------------
# T9 — (R1) pre-P2 compatibility, self-contained
# ---------------------------------------------------------------------------

def test_pre_p2_stripped_trajectory_diffs_clean_in_both_orders():
    """A fresh default-scenario capture with __swarm__ deleted from every
    snapshot (exactly the shape a pre-P2 build pickled) diffs clean against
    a fresh capture, in both argument orders. The same stripped trajectory
    against a swarm_scenario_sim capture reports 'present in only one run'
    on every tick. Self-contained: does not load the 45050f3 pickle.

    Must break if the one-sided-key check runs before the swarm case, or a
    missing key is compared as a mismatch against an absent carrier."""
    traj = fab.capture_trajectory(make_sim=fab.default_scenario_sim, n_steps=5)
    stripped = [dict(s) for s in traj]
    for s in stripped:
        del s[SWARM_DIGEST_KEY]

    assert fab.diff_trajectories(stripped, traj) == []
    assert fab.diff_trajectories(traj, stripped) == []

    traj_swarm = fab.capture_trajectory(make_sim=fab.swarm_scenario_sim, n_steps=5)
    diffs = fab.diff_trajectories(stripped, traj_swarm)
    assert len(diffs) == len(stripped)
    assert all("present in only one run" in d for d in diffs)


# ---------------------------------------------------------------------------
# T10 — (R2) integer rendering
# ---------------------------------------------------------------------------

def test_integer_rendering_numpy_scalars_vs_python_ints_encode_identically():
    """Two carriers equal except that next_unit_id/high_water are
    np.int32/np.int64 scalars in one and Python ints in the other encode to
    identical bytes; b"np." never appears; header integers are ASCII
    decimal. Must break if a header or scalar token is built through
    str()/repr() of a numpy scalar."""
    g = GameMap(_mini_level())
    su = swarm.SwarmUnits(g)
    sp = swarm.SWARM_SPECIES[0]
    su.spawn(sp, [0, 1 << 16], [0, 0], [0, 100])
    carrier = swarm.swarm_carrier(g)

    carrier_numpy = dict(carrier)
    carrier_numpy["next_unit_id"] = np.int64(carrier["next_unit_id"])
    carrier_numpy["blocks"] = {
        s: {"high_water": np.int32(b["high_water"]), "columns": b["columns"]}
        for s, b in carrier["blocks"].items()
    }

    bytes_python = swarm.swarm_section_bytes(carrier)
    bytes_numpy = swarm.swarm_section_bytes(carrier_numpy)
    assert bytes_python == bytes_numpy
    assert b"np." not in bytes_python
    assert b"np." not in bytes_numpy
    assert (f"next_unit_id|{int(carrier['next_unit_id'])}\n".encode("ascii")
           in bytes_python)


# ---------------------------------------------------------------------------
# T11 — (R6) env-0 only
# ---------------------------------------------------------------------------

def test_multi_env_raises_instead_of_silently_folding(monkeypatch):
    """With swarm.N_ENV monkeypatched to 2, swarm_carrier raises
    NotImplementedError instead of encoding. Must break if the carrier or
    encoder silently loops or folds envs (B1's decision)."""
    g = GameMap(_mini_level())
    monkeypatch.setattr(swarm, "N_ENV", 2)
    with pytest.raises(NotImplementedError):
        swarm.swarm_carrier(g)
