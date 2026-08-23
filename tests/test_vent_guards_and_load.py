"""Vent system PATCH 1 (issue #48) — the runtime-guard + load-validation
gates:

(d) each RUNTIME aperture guard (solid / thermal_solid / vacuum / ambient /
    flooded) is a counted no-op — no ledger corruption, accumulator frozen,
    exactly ONE frozen tick's worth of mass/energy sits where it always did;
(f) a dangling ``vent.duct`` reference is a LOADER WARNING (the generic A3
    KIND_ENTITY_REF path, not a hard error) — the vent still builds, just
    never resolves a plenum (inert, fail-safe, the bus-free-pump precedent).

Run:
    conda run -n data python -m pytest tests/test_vent_guards_and_load.py -q
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.gases import N_GASES, O2, SMOKE  # noqa: E402


def _tm(h=10, w=10):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return tm


def _level(tm, entities=(), **kw):
    return LevelData(name="vent_guard_fix", version="1", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."),
                     entities=list(entities), wires=[], **kw)


def _inst(cls_name, eid, ordinal, **overrides):
    cls = REGISTRY[cls_name]
    fields = {f.name: f.default for f in cls.FIELDS}
    fields.update(overrides)
    return EntityInstance(id=eid, class_name=cls_name, ordinal=ordinal,
                          tags=(), fields=fields)


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def _ents(ret_xy=(2, 5), sup_xy=(7, 5), q_circ=8.0):
    duct = _inst("duct", "d1", 0, filter="derelict")
    rx, ry = ret_xy
    sx, sy = sup_xy
    ret = _inst("vent", "vret", 1, x=rx, y=ry, mount="floor", role="return",
               duct="d1", q_circ=q_circ)
    sup = _inst("vent", "vsup", 2, x=sx, y=sy, mount="floor", role="supply",
               duct="d1", q_circ=q_circ)
    return [duct, ret, sup]


def _snapshot(gmap):
    return (gmap.gas.copy(), gmap.temperature.copy())


def _unchanged(before, gmap):
    gas0, t0 = before
    return np.array_equal(gas0, gmap.gas) and np.array_equal(t0, gmap.temperature)


# ===========================================================================
# (d) Runtime aperture guards — counted no-op, zero ledger corruption
# ===========================================================================

@pytest.mark.parametrize("guard_flag", ["solid", "thermal_solid", "is_vacuum",
                                        "is_ambient", "water_depth"])
def test_return_vent_guard_is_a_counted_noop(guard_flag):
    sim = Simulation(_level(_tm(), _ents()), seed=1, breach_physics=None,
                     enable_recorder=False)
    v = sim._vents[0]                      # return
    fy, fx = v.aperture_y, v.aperture_x
    sim.gmap.gas[SMOKE][fy, fx] = 20000
    if guard_flag == "water_depth":
        sim.gmap.water_depth[fy, fx] = 100
    else:
        getattr(sim.gmap, guard_flag)[fy, fx] = True

    before = _snapshot(sim.gmap)
    accum0 = v.accum
    skips0 = v.guard_skips
    for _ in range(30):
        _step(sim)
    assert v.guard_skips > skips0, "the guard never fired — the gate is vacuous"
    assert v.accum == accum0, "a blocked vent's accumulator was NOT frozen"
    assert _unchanged(before, sim.gmap), "a blocked return vent still edited the field"
    duct = sim._ducts[0]
    assert duct.o2_raw == 0 and duct.n2_raw == 0 and sum(duct.trace_raw) == 0, \
        "a blocked return vent still credited the plenum"


@pytest.mark.parametrize("guard_flag", ["solid", "thermal_solid", "is_vacuum",
                                        "is_ambient", "water_depth"])
def test_supply_vent_guard_is_a_counted_noop_no_ledger_corruption(guard_flag):
    sim = Simulation(_level(_tm(), _ents()), seed=2, breach_physics=None,
                     enable_recorder=False)
    ret, sup = sim._vents
    duct = sim._ducts[0]
    # Prime the plenum directly (avoid depending on intake this test isn't
    # about) so the supply side has real mass ready to give.
    duct.o2_raw = 40000
    duct.n2_raw = 40000

    fy, fx = sup.aperture_y, sup.aperture_x
    if guard_flag == "water_depth":
        sim.gmap.water_depth[fy, fx] = 100
    else:
        getattr(sim.gmap, guard_flag)[fy, fx] = True

    supply_before = [int(sim.gmap.gas[g][fy, fx]) for g in range(N_GASES)]
    t_before = int(sim.gmap.temperature[fy, fx])
    accum0 = sup.accum
    skips0 = sup.guard_skips
    o2_before, n2_before = duct.o2_raw, duct.n2_raw
    for _ in range(30):
        _step(sim)
    assert sup.guard_skips > skips0, "the supply guard never fired — vacuous gate"
    assert sup.accum == accum0, "a blocked supply vent's accumulator was NOT frozen"
    # The return vent is unblocked in this scenario and may still have
    # credited the plenum from ambient content at its own aperture — but the
    # SUPPLY side, being fully blocked, must never have withdrawn anything:
    # o2_raw/n2_raw can only have gone UP (from intake), never down. And the
    # SUPPLY TILE itself (not the whole grid — the return tile legitimately
    # changes) must be untouched: zero deposit landed there.
    assert duct.o2_raw >= o2_before and duct.n2_raw >= n2_before
    supply_after = [int(sim.gmap.gas[g][fy, fx]) for g in range(N_GASES)]
    assert supply_after == supply_before, "a blocked supply vent still deposited"
    assert int(sim.gmap.temperature[fy, fx]) == t_before, \
        "a blocked supply vent still wrote temperature"


def test_flooded_guard_threshold_matches_seal_tiles_convention():
    """The flood guard is EXACTLY `water_depth != 0` (gamemap.seal_tiles'
    own invariant, no separate threshold dial) — the boundary itself, not
    just "some large depth", blocks the aperture."""
    sim = Simulation(_level(_tm(), _ents()), seed=3, breach_physics=None,
                     enable_recorder=False)
    v = sim._vents[0]
    fy, fx = v.aperture_y, v.aperture_x
    sim.gmap.water_depth[fy, fx] = 1        # the smallest possible non-zero depth
    sim.gmap.gas[SMOKE][fy, fx] = 5000
    before = _snapshot(sim.gmap)
    for _ in range(20):
        _step(sim)
    assert _unchanged(before, sim.gmap)
    assert v.guard_skips > 0


# ===========================================================================
# (f) Dangling duct ref -> loader warning (the generic A3 KIND_ENTITY_REF path)
# ===========================================================================

def test_dangling_duct_ref_warns_not_hard_errors():
    raw = {"entity": [
        {"id": "v1", "class": "vent", "x": 3, "y": 3, "mount": "floor",
         "role": "supply", "duct": "nonexistent_duct", "q_circ": 1.0},
    ]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        entities = level_loader._parse_entities(  # the A3 loader, per test_b1's precedent
            raw, toml_path=Path("<test>"), spawns=[])
    assert any("dangling ref" in str(w.message) for w in caught), \
        "a dangling vent.duct ref did not warn via the generic entity-ref path"
    assert entities[0].class_name == "vent"      # parsed anyway — not fatal


def test_vent_with_dangling_duct_builds_but_never_sweeps():
    """§2/§3: an unwired/dangling `duct` ref builds an inert vent — it never
    touches the field, exactly like a bus-free pump."""
    v_only = _inst("vent", "vorphan", 0, x=4, y=4, mount="floor",
                   role="return", duct="ghost_duct", q_circ=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(_level(_tm(), [v_only]), seed=4, breach_physics=None,
                         enable_recorder=False)
    assert len(sim._vents) == 1
    assert sim._vents[0].duct is None
    assert sim._ducts == []
    fy, fx = sim._vents[0].aperture_y, sim._vents[0].aperture_x
    sim.gmap.gas[SMOKE][fy, fx] = 12345
    before = _snapshot(sim.gmap)
    for _ in range(20):
        _step(sim)
    assert _unchanged(before, sim.gmap)


def test_orphan_vent_alongside_a_real_duct_stays_inert():
    """An unwired/dangling vent sitting on a level that DOES have a live
    duct (so `sweep_vents` genuinely runs) exercises the per-vent
    ``v.duct is None`` skip inside the sweep itself, not just the outer
    ``if self._ducts`` gate."""
    ents = _ents()
    orphan = _inst("vent", "vorphan", 3, x=1, y=1, mount="floor",
                   role="return", duct="ghost_duct", q_circ=9.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(_level(_tm(), ents + [orphan]), seed=5,
                         breach_physics=None, enable_recorder=False)
    assert sim._ducts != []                 # the real duct DID build — sweep runs
    orphan_rt = next(v for v in sim._vents if v.inst.id == "vorphan")
    assert orphan_rt.duct is None
    fy, fx = orphan_rt.aperture_y, orphan_rt.aperture_x
    sim.gmap.gas[SMOKE][fy, fx] = 7777
    before_orphan = int(sim.gmap.gas[SMOKE][fy, fx])
    for _ in range(30):
        _step(sim)
    assert int(sim.gmap.gas[SMOKE][fy, fx]) == before_orphan, \
        "an orphan vent on a live-duct level still touched its own aperture"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
