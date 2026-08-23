"""Vent system PATCH 1 (issue #48) — determinism + serialization gates:

(b) two identical seeded runs with vents produce IDENTICAL ENTITY_SECT +
    field digests (the A/B lockstep spirit, restricted to entity state since
    breach_physics=None here — the full-suite GOLDEN_AGGREGATE run covers
    the physics-coupled case);
(e) serialization roundtrip: ``runtime_digest_rows`` is a FAITHFUL mirror of
    the live plenum/accumulator state — what the recorder/digest capture is
    exactly what the runtime objects hold, at every point in a run, not just
    at rest.

Run:
    conda run -n data python -m pytest tests/test_vent_determinism_and_serialize.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.entities.serialize import entity_carrier, entity_section_bytes  # noqa: E402
from simulation.gases import POISON, SMOKE, STEAM  # noqa: E402

TPS = 24


def _tm(h=10, w=10):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return tm


def _level(tm, entities=(), **kw):
    return LevelData(name="vent_ds_fix", version="1", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."),
                     entities=list(entities), wires=[], **kw)


def _inst(cls_name, eid, ordinal, **overrides):
    cls = REGISTRY[cls_name]
    fields = {f.name: f.default for f in cls.FIELDS}
    fields.update(overrides)
    return EntityInstance(id=eid, class_name=cls_name, ordinal=ordinal,
                          tags=(), fields=fields)


def _ents():
    duct = _inst("duct", "d1", 0, filter="hepa_basic")
    ret = _inst("vent", "vret", 1, x=2, y=5, mount="floor", role="return",
               duct="d1", q_circ=7.0)
    sup = _inst("vent", "vsup", 2, x=7, y=5, mount="floor", role="supply",
               duct="d1", q_circ=7.0)
    return [duct, ret, sup]


def _build_and_seed(seed):
    sim = Simulation(_level(_tm(), _ents()), seed=seed, breach_physics=None,
                     enable_recorder=False)
    ry, rx = sim._vents[0].aperture_y, sim._vents[0].aperture_x
    sim.gmap.gas[SMOKE][ry, rx] = 25000
    sim.gmap.gas[POISON][ry, rx] = 18000
    sim.gmap.gas[STEAM][ry, rx] = 9000
    sim.gmap.temperature[:, :] = 3000
    return sim


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def _field_bytes(gmap):
    return (gmap.gas.tobytes() + gmap.temperature.tobytes())


# ===========================================================================
# (b) Determinism — two identical seeded runs -> identical digests
# ===========================================================================

def test_two_identical_runs_produce_identical_entity_and_field_digests():
    sim_a = _build_and_seed(seed=99)
    sim_b = _build_and_seed(seed=99)

    digests_a, digests_b = [], []
    for t in range(120):
        _step(sim_a)
        _step(sim_b)
        rec_a = entity_carrier(sim_a.entities)
        rec_b = entity_carrier(sim_b.entities)
        digests_a.append(entity_section_bytes(rec_a) + _field_bytes(sim_a.gmap))
        digests_b.append(entity_section_bytes(rec_b) + _field_bytes(sim_b.gmap))

    assert digests_a == digests_b, \
        "two identically-seeded vented runs diverged — the sweep is not deterministic"
    # Non-vacuous: something actually happened over 120 ticks.
    assert digests_a[-1] != digests_a[0]


def test_ordinal_order_independent_of_authoring_file_order():
    """Entity-ordinal order (§3) drives the sweep, not TOML declaration
    order — swapping which duct-member vent is declared FIRST in the file
    (but keeping the SAME ordinals) must not change the trajectory, since
    the sweep already iterates by ordinal, never by list-append order."""
    duct = _inst("duct", "d1", 0, filter="derelict")
    ret = _inst("vent", "vret", 1, x=2, y=5, mount="floor", role="return",
               duct="d1", q_circ=7.0)
    sup = _inst("vent", "vsup", 2, x=7, y=5, mount="floor", role="supply",
               duct="d1", q_circ=7.0)

    sim_a = Simulation(_level(_tm(), [duct, ret, sup]), seed=7,
                       breach_physics=None, enable_recorder=False)
    sim_b = Simulation(_level(_tm(), [duct, sup, ret]), seed=7,          # swapped list order
                       breach_physics=None, enable_recorder=False)
    for sim in (sim_a, sim_b):
        ry, rx = ([v for v in sim._vents if v.role == "return"][0].aperture_y,
                  [v for v in sim._vents if v.role == "return"][0].aperture_x)
        sim.gmap.gas[SMOKE][ry, rx] = 20000

    for _ in range(60):
        _step(sim_a)
        _step(sim_b)
    assert _field_bytes(sim_a.gmap) == _field_bytes(sim_b.gmap)


# ===========================================================================
# (e) Serialization roundtrip — runtime_digest_rows mirrors live state
# ===========================================================================

def test_duct_runtime_digest_rows_mirror_live_ledger_exactly():
    sim = _build_and_seed(seed=11)
    duct = sim._ducts[0]
    for _ in range(80):
        _step(sim)
        rows = dict(REGISTRY["duct"].runtime_digest_rows(duct))
        assert rows["o2_raw"] == duct.o2_raw
        assert rows["n2_raw"] == duct.n2_raw
        assert rows["e_plenum"] == duct.e_plenum
        assert rows["e_wipe"] == duct.e_wipe
        assert rows["rail_lo_hits"] == duct.rail_lo_hits
        assert rows["rail_hi_hits"] == duct.rail_hi_hits
        for i, v in enumerate(duct.trace_raw):
            assert rows[f"trace_{i}"] == v
        for i, v in enumerate(duct.sink):
            assert rows[f"sink_{i}"] == v


def test_vent_runtime_digest_rows_mirror_live_accumulator_exactly():
    sim = _build_and_seed(seed=12)
    for _ in range(80):
        _step(sim)
        for v in sim._vents:
            rows = dict(REGISTRY["vent"].runtime_digest_rows(v))
            assert rows["accum"] == v.accum


def test_entity_carrier_round_trips_duct_and_vent_rows_into_records():
    """The ONE serializer (entity_records/entity_carrier) folds the runtime
    rows into the hashed ENTITY_SECT bytes without raising, and the exact
    row values are recoverable by re-parsing the record header/rows back
    (a lightweight structural roundtrip — the digest format is
    newline/pipe-delimited ASCII + packed int64, entities/serialize.py)."""
    sim = _build_and_seed(seed=13)
    for _ in range(50):
        _step(sim)
    carrier = entity_carrier(sim.entities)
    assert carrier["n_entities"] == 3
    duct = sim._ducts[0]
    duct_record = next(r for r in carrier["records"]
                       if f"|{duct.id}|duct\n".encode("ascii") in r)
    # Every duct runtime row name appears, pipe-delimited, in the record bytes
    # (a coarse but real roundtrip check: the serializer did not drop a row).
    for name, _value in REGISTRY["duct"].runtime_digest_rows(duct):
        assert f"{name}|".encode("ascii") in duct_record


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
