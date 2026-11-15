"""#29 P6.4b -- the per-env serial applier's one table (``simulation.applier``):
its generated C++ header, its planes against the GameMap they point into, its
books against the host quantities they twin, and the effect list's packing.

Pure Python (no CUDA): the device side is held to these by
``tests/cuda_unit_topology_check.py``.
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

bp = pytest.importorskip("breach_physics")

from simulation import applier  # noqa: E402


@pytest.fixture(scope="module")
def ship_gmap():
    import level_loader
    from simulation import Simulation
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    return Simulation(level, seed=1, breach_physics=bp,
                      enable_recorder=False).gmap


def test_the_checked_in_header_is_the_applier_table():
    """Property: ``cpp/src/applier_layout.h`` is exactly the text
    ``applier.header_text()`` generates (the record, the kinds, the books, the
    planes' order, the clears). Breaks when a kind, a book or a plane is added
    or reordered without regenerating: the device would read the launch's
    pointer table or write the books in another order than the host packs."""
    path = ROOT / applier.HEADER_REL
    have = path.read_text(encoding="utf-8") if path.is_file() else ""
    assert have == applier.header_text(), (
        f"{applier.HEADER_REL} is stale: run `C:/Users/steen/anaconda3/envs/"
        f"data/python.exe {applier.GENERATOR}` and commit the header")


def test_every_plane_is_the_gamemap_field_it_names(ship_gmap):
    """Property: every ``applier.PLANES`` row is a GameMap array attribute in
    the row's dtype with the row's per-cell layout (``(h, w)``, ``(h, w, c)``
    or the gas planes ``(G, h, w)``), C-contiguous -- so the device's typed
    rebase of the launch's pointer table reads the host's bytes. Breaks if a
    plane's dtype or shape moves on the GameMap (e.g. ``fuel_recip`` to
    int32) without the table."""
    g = ship_gmap
    h, w = g.material.shape
    for name, dt, cw in applier.PLANES:
        a = getattr(g, name)
        assert isinstance(a, np.ndarray), name
        assert a.dtype == np.dtype(dt), (name, a.dtype, dt)
        want = ((g.gas.shape[0], h, w) if cw == "G" else
                (h, w) if cw == 1 else (h, w, int(cw)))
        assert a.shape == want, (name, a.shape, want)
        assert a.flags.c_contiguous, name
    for name, dt in applier.CLEAR_PLANES:
        a = getattr(g, name)
        assert a.dtype == np.dtype(dt) and a.shape == (h, w), name


def test_every_book_twins_a_host_quantity(ship_gmap):
    """Property: each book with a host twin names a GameMap integer attribute
    or a key of a GameMap dict of integers (absent = 0), so the shadow harness
    can take the host's delta and the chained harness land the device's.
    Breaks if a twin is renamed on the GameMap (``n_destruction_seed_sum``,
    the seam channels, ``_tile_serial``)."""
    g = ship_gmap
    for b in applier.BOOKS:
        if b.host is None:
            continue
        attr = b.host.split(":", 1)[0]
        assert hasattr(g, attr), (b.name, b.host)
        assert isinstance(applier.host_book(g, b.host), int), b.name


def test_the_non_resident_planes_are_exactly_those_outside_the_resident_set():
    """Property: ``non_resident_planes`` names a plane iff it is not a
    resident field (``GameMap._RESIDENT_FIELD_NAMES``) -- the list P8b must
    bring onto the device. Breaks if a plane joins or leaves the resident set
    and the list does not follow."""
    from simulation.gamemap import GameMap
    res = set(GameMap._RESIDENT_FIELD_NAMES)
    nr = set(applier.non_resident_planes())
    for n in applier.PLANE_NAMES + applier.CLEAR_NAMES:
        assert (n in nr) == (n not in res), n


def test_a_list_past_its_capacity_keeps_its_first_entries_and_its_count():
    """Property: ``pack`` keeps each env's first ``cap`` records in order and
    its REQUESTED count (which the device compares with ``cap`` to count the
    dropped), padding the rest with no-record rows. Breaks if the packing
    truncates the count (the overflow would go uncounted) or reorders."""
    recs = applier.destroy_records([(i, i + 1) for i in range(10)])
    out, counts = applier.pack([recs, recs[:3], recs[:0]], cap=4)
    assert out.shape == (3, 4, applier.F)
    assert counts.tolist() == [10, 3, 0]
    assert np.array_equal(out[0], recs[:4])
    assert np.array_equal(out[1, :3], recs[:3]) and not out[1, 3].any()
    assert not out[2].any()
    assert (recs[:, 0] == applier.KIND_DESTROY_WALL).all()
