"""#29 P6.2 -- the device tables of the unit kernels (``simulation.unit_tables``):
the row schema, its generated C++ header, and the values a sim's reference
objects give.

Pure Python (no CUDA): the device side is held equal to these values by
``tests/cuda_unit_kernels_check.py`` part TABLES.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

bp = pytest.importorskip("breach_physics")

from simulation import unit_tables  # noqa: E402


@pytest.fixture(scope="module")
def ship_sim():
    import level_loader
    from simulation.match_session import MatchSession
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    return MatchSession(level, seed=1, breach_physics=bp).sim


def test_the_checked_in_header_is_the_tables_layout():
    """Property: ``cpp/src/unit_tables_layout.h`` is exactly the text
    ``unit_tables.header_text()`` generates from the rows, the hooks and the
    status kinds. Breaks when a table row, a hook or a status kind is added,
    removed or re-widened (an env profile or a species added included)
    without regenerating the header -- the kernels would read the wrong
    offsets or the wrong hook."""
    path = ROOT / unit_tables.HEADER_REL
    have = path.read_text(encoding="utf-8") if path.is_file() else ""
    assert have == unit_tables.header_text(), (
        f"{unit_tables.HEADER_REL} is stale: run "
        f"`C:/Users/steen/anaconda3/envs/data/python.exe "
        f"{unit_tables.GENERATOR}` and commit the header")


def test_rows_tile_the_vector_in_order_and_the_fingerprint_sees_a_width():
    """Property: the rows sit back to back in declaration order, each at its
    width, and the fingerprint moves when one row's width moves (the device
    store refuses a compiled module on it). Breaks if an offset is computed
    from anything but the running width, or the fingerprint ignores widths."""
    lay = unit_tables.layout()
    off = 0
    for r, c in zip(unit_tables.ROWS, lay.columns):
        assert (c.name, c.offset, c.width) == (r.name, off, r.w())
        off += c.width
    assert lay.width == off
    row = unit_tables.ROWS[0]
    widened = (replace(row, width=row.w() + 1),) + unit_tables.ROWS[1:]
    saved = unit_tables.ROWS
    try:
        unit_tables.ROWS = widened
        unit_tables._LAYOUT_CACHE.clear()
        assert unit_tables.layout().fingerprint != lay.fingerprint
    finally:
        unit_tables.ROWS = saved
        unit_tables._LAYOUT_CACHE.clear()
    assert unit_tables.layout().fingerprint == lay.fingerprint


def test_values_are_the_reference_objects_own(ship_sim):
    """Property: every value is read from the object the reference law reads
    -- the sim's impulse table, its turning dials, the status registry, the
    exposure settings, the class hp. Breaks if a row re-derives its value
    (from config, or another quantization) instead of reading that object:
    the device law would run on other numbers than the host's."""
    from simulation import exposure
    from simulation.status import STATUS_REGISTRY
    from simulation.unit_traits import class_hp_q
    v = unit_tables.values(ship_sim)
    t = ship_sim.impulse_table
    assert unit_tables.read(v, "impulse_cap_q") == t.cap_q
    assert unit_tables.read(v, "impulse_v_fall_q") == tuple(t.v_fall_q)
    assert unit_tables.read(v, "impulse_wind_k_prone_q") == t.wind_k_prone_q
    assert unit_tables.read(v, "turn_rate_class_q16") == \
        ship_sim._marine_turn_rate_q16
    assert unit_tables.read(v, "exposure_decay_q") == \
        exposure.get_settings().decay_q
    assert unit_tables.read(v, "hp_class_zombie_q") == class_hp_q(True)
    flags = unit_tables.read(v, "status_flags")
    flags = flags if isinstance(flags, tuple) else (flags,)
    for r, f in zip(STATUS_REGISTRY, flags):
        assert bool(f & unit_tables.FLAG_PRONE) == r.is_prone
        assert bool(f & unit_tables.FLAG_NO_MOVE) == (not r.can_move)


def test_a_value_outside_its_proved_range_is_refused_by_name(ship_sim):
    """Property: a table value outside the range the kernels' arithmetic was
    proved for is refused at :func:`values`, naming the row (an exposure
    decay outside [0, 65536] among them -- P6.1's runtime refusal moved to
    this door). Breaks if the door lets a value through, so a kernel could
    overflow int64 silently."""
    saved = unit_tables.ROWS
    i = [r.name for r in saved].index("exposure_decay_q")
    for bad in (-1, 65537):
        rows = list(saved)
        rows[i] = replace(saved[i], value=lambda src, b=bad: b)
        try:
            unit_tables.ROWS = tuple(rows)
            with pytest.raises(ValueError, match="exposure_decay_q"):
                unit_tables.values(ship_sim)
        finally:
            unit_tables.ROWS = saved


def test_a_batch_row_is_its_env_alone(ship_sim):
    """Property: ``values_batch`` row e is ``values`` of env e alone (per-map
    tables differ per env). Breaks if a batch shares one env's tables."""
    a = unit_tables.values(ship_sim)
    b = unit_tables.values_batch([ship_sim, ship_sim])
    assert b.shape == (2, a.shape[0]) and np.array_equal(b[1], a)
