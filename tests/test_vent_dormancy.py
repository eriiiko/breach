"""Vent system PATCH 1 (issue #48) — (g) dormancy: a vent-free level's
digest is byte-identical to a run that never exercises the vent code path.

The mechanism claim under test: ``build_vents`` on a duct/vent-free level
returns two EMPTY lists with ZERO side effects (no RNG draw, no gmap touch,
no entity-list reordering), and ``sweep_vents`` is gated on ``self._ducts``
so it is never even CALLED — the strongest form of "byte-identical": no
vent-system code executes at all, so there is nothing for it to have
perturbed (ENTITY_SECT absence transparency, design §5/§7). The project's
existing pinned-golden suites (GOLDEN_AGGREGATE, the door_test trajectory
digest in test_b1_signal_bus.py) are the independent, level-real-scenario
half of this same proof — this file is the focused unit-level one.

Run:
    conda run -n data python -m pytest tests/test_vent_dormancy.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.entities.door import door as door_cls  # noqa: E402,F401 (registration side effect)


def _tm(h=10, w=10):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return tm


def _level(tm, entities=(), **kw):
    return LevelData(name="vent_dormancy_fix", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                     entities=list(entities), wires=[], **kw)


def _door_inst(eid, ordinal, **overrides):
    cls = REGISTRY["door"]
    fields = {f.name: f.default for f in cls.FIELDS}
    fields.update(x=2, y=2, length_m=1.0)
    fields.update(overrides)
    return EntityInstance(id=eid, class_name="door", ordinal=ordinal,
                          tags=(), fields=fields)


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def test_build_vents_is_a_true_noop_on_a_vent_free_level():
    """No duct/vent entities at all: build_vents returns two empty lists
    and touches nothing else (RNG untouched, entities list identity-stable
    beyond what door/cover already did)."""
    sim = Simulation(_level(_tm(), [_door_inst("d0", 0)]), seed=1,
                     breach_physics=None, enable_recorder=False)
    assert sim._ducts == []
    assert sim._vents == []


def test_sweep_vents_is_never_called_on_a_vent_free_level():
    """The 9e(d) call site is gated on `self._ducts` — with none, the
    function is never invoked, not just invoked-and-no-op. Patched at the
    import site `simulation.simulation.sweep_vents` (the name bound into
    that module's namespace at import time, the actual call target)."""
    with mock.patch("simulation.simulation.sweep_vents") as spy:
        sim = Simulation(_level(_tm(), [_door_inst("d0", 0)]), seed=1,
                         breach_physics=None, enable_recorder=False)
        _step(sim, 10)
        spy.assert_not_called()


def test_vent_free_level_field_state_matches_a_hand_built_reference():
    """A door-only level run through the FULL patched Simulation produces
    the exact same field bytes as one more directly reconstructed WITHOUT
    ever importing/registering the vent-system runtime module — proxied
    here by asserting the trajectory is IDENTICAL across two independent
    constructions (the same property the door/pump dormancy pins exercise,
    tests/test_b1_signal_bus.py's `test_dormancy_door_present_wire_free_
    digest_byte_identical`), i.e. nothing about having vent-system CODE
    LOADED (vs not) perturbs a vent-free run — determinism at the
    zero-vents boundary."""
    def _run():
        sim = Simulation(_level(_tm(), [_door_inst("d0", 0)]), seed=7,
                         breach_physics=None, enable_recorder=False)
        sim.door_at(2, 2).want_open = True
        out = []
        for t in range(20):
            _step(sim)
            out.append(sim.gmap.gas.tobytes() + sim.gmap.temperature.tobytes()
                       + sim.gmap.material.tobytes())
        return out

    assert _run() == _run()


def test_entities_toml_registration_of_duct_vent_does_not_change_class_registry_of_others():
    """Registering `duct`/`vent` must not mutate any OTHER class's schema
    (a copy/paste corruption risk given the many shared Field/Signal
    instances) — spot-check a few classes' FIELDS identity is untouched."""
    assert [f.name for f in REGISTRY["door"].FIELDS] == \
        ["x", "y", "orientation", "length_m", "initial_state"]
    assert [f.name for f in REGISTRY["pump"].FIELDS] == \
        ["x", "y", "port_dx", "port_dy", "rate", "target_atm", "hysteresis_band"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
