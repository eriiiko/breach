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


@pytest.fixture(autouse=True)
def _reset_weapon_tables_to_pinned_world():
    """Re-bind the shared weapon tables to the 1.0 m/tile test convention
    before every test (see module docstring)."""
    from simulation.weapons import rebuild_tables
    rebuild_tables(tile_size_m=1.0)
    yield
