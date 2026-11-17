"""Shared pytest fixtures for the breach suite.

THE WEAPON-TABLE RESET (W6 merge finding, 2026-07-21). Since W6, weapon
ranges are meter-authored and the shared tables bind a LEVEL'S tile size at
Simulation construction (``rebuild_tables(tile_size_m=gmap.tile_size_m)``,
engine/14 door 2). That rebinding is a MODULE-GLOBAL side effect that
outlives the sim — so a test constructing a Simulation on a 0.333 m/tile
level (door_test, test_level, playground) would silently leak
playground-scale ranges into any LATER test that reads ``get_tables()``
bare (the documented bare convention is the 1.0 m/tile pinned-test world).
First observed as test_a6_doors (door_test, 0.333) poisoning
test_ammo_economy's GL-6 airburst position (range cap 40 -> 120 tiles):
green alone, red in sequence.

The autouse fixture below re-establishes the 1.0 binding BEFORE EVERY
TEST — cheap (an in-memory CFG projection, no I/O) and exactly the bare
convention. Tests that build a Simulation rebind inside their own body and
are unaffected; tests that drive combat helpers directly get the pinned
world they were written against, whatever ran before them.

(The deeper seam — combat consumers reading the module-global
``weapon_tables()`` instead of a sim-owned bundle — is noted in the W6
findings; it becomes load-bearing only when two Simulations with different
tile sizes must coexist in one process, which nothing does today.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# THE TWO TIERS AND THE DEVICE GROUP (#205, docs/test_suite_timing_2026-11-15.md)
#
# ``slow`` marks every test in ``tests/slow_tests.txt`` -- a MEASURED list
# (``tools/suite_timing.py --write-slow`` over a ``--durations=0`` run), never
# a hand-set mark. ``pytest tests -q -m "not slow"`` is the fast tier for work
# in progress; the full suite stays the gate before a merge. A test missing
# from the list is fast until the next measurement.
#
# Under pytest-xdist (``-n N --dist loadgroup``) every test that drives the GPU
# or opens a window (``_suite_kinds.is_device``) joins ONE group, so no two run
# at once, and the run starts with the longest measured tests so none of them
# is left to the end on its own. Both only with ``-n``: a plain run keeps its
# order and pays no source reading at collection.
# ---------------------------------------------------------------------------
SLOW_LIST = Path(__file__).with_name("slow_tests.txt")


def read_slow_list(path: Path = SLOW_LIST) -> dict[str, float]:
    """nodeid -> measured seconds, from ``<seconds> <nodeid>`` lines."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        secs, nodeid = line.split(maxsplit=1)
        out[nodeid] = float(secs)
    return out


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: measured slow (tests/slow_tests.txt); the fast tier "
        "is -m 'not slow'")
    config.addinivalue_line(
        "markers", "xdist_group(name): pytest-xdist's group (loadgroup)")


def _xdist_workers(config) -> int:
    n = getattr(config.option, "numprocesses", None)
    try:
        return int(n or 0)
    except (TypeError, ValueError):     # "auto" / "logical"
        return 2


def pytest_collection_modifyitems(config, items):
    slow = read_slow_list()
    for item in items:
        if item.nodeid in slow:
            item.add_marker(pytest.mark.slow)
    if not (_xdist_workers(config) or hasattr(config, "workerinput")):
        return
    import _suite_kinds
    for item in items:
        path, name = _suite_kinds.split_nodeid(item.nodeid)
        if _suite_kinds.is_device(str(ROOT / path), name):
            item.add_marker(pytest.mark.xdist_group("device"))
    # longest first; a stable sort keeps collection order among the rest
    items.sort(key=lambda it: -slow.get(it.nodeid, 0.0))


@pytest.fixture(autouse=True)
def _reset_weapon_tables_to_pinned_world():
    """Re-bind the shared weapon tables to the 1.0 m/tile test convention
    before every test (see module docstring)."""
    from simulation.weapons import rebuild_tables
    rebuild_tables(tile_size_m=1.0)
    yield
