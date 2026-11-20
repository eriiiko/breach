"""#143 -- ``GameMap.find_burst_walls`` (vectorised) is the pre-#143 loop, bit
for bit.

The scan runs every tick (slot 9b) and was a Python loop over every solid
tile; it is now numpy. What it returns decides which walls the valve destroys
and in which ORDER (``destroy_wall`` of the first can change the donor
composition of the next), so the oracle is the frozen pre-#143 loop in
``tests/_reference_scans.py::find_burst_walls_reference`` and every property
here is "the same list, in the same order" on seeded random maps built through
the real ``GameMap``: random materials (walls of every thickness, air, vacuum
or the ambient ring, materials that never burst), pressures from a coarse
discrete set (so equal spreads, spreads exactly AT a threshold and ties are
everywhere) and from a continuous one, with ``max_pops`` None / 0 / small /
large / negative.

Properties, never snapshots: the maps are random, nothing here pins a count or
a set of tiles; ``test_destroy_order_pins.py`` keeps the hand-built scenes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "tests", ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _reference_scans as ref  # noqa: E402
from level_loader import (  # noqa: E402
    LEVEL_FORMAT_VERSION, SPACE_CODE, LevelData,
)
from simulation import atmosphere_fixed as afx  # noqa: E402
from simulation.ambient import derive_ambient  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402

# air 0, hull 1 (never bursts), wood 2 (2.0), steel 4 (10.0), glass 5 (1.0),
# furniture 6 (2.0), door_closed 7 (2.0), kindling 8 (2.0), foliage 9 (0.0),
# grille 10 (0.0); SPACE_CODE = vacuum (or the ambient ring on a planetside map)
_CODES = [0, 0, 0, 0, 1, 2, 2, 2, 4, 5, 5, 6, 7, 8, 9, 10, SPACE_CODE]
_DISCRETE = [0.0, 0.5, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 7.5, 12.0]


def _random_map(rng, h, w, *, ambient: bool, wall_bias: float):
    """A random GameMap: random material codes (the thin and thick wall runs
    that make a tile's open sides 0..4 all occur), random pressures."""
    codes = np.array(_CODES, dtype=np.int32)
    tm = rng.choice(codes, size=(h, w))
    # long wall runs: whole thin partitions and a thick slab, so 1-tile
    # membranes with exactly two open sides exist in quantity
    for _ in range(int(rng.integers(1, 4))):
        mat = int(rng.choice([1, 2, 2, 4, 5, 7]))
        if rng.random() < 0.5:
            tm[int(rng.integers(0, h)), :] = mat
        else:
            tm[:, int(rng.integers(0, w))] = mat
    thick = int(rng.integers(0, w - 2))
    tm[:, thick:thick + 2] = np.where(rng.random((h, 2)) < wall_bias, 2, tm[:, thick:thick + 2])
    level = LevelData(
        name="burst_parity", version=LEVEL_FORMAT_VERSION, path=Path("."),
        tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
        boundary="ambient" if ambient else "space",
        ambient=derive_ambient() if ambient else None)
    g = GameMap(level)
    # pressures, written the way the sibling tests write them (quantize of a
    # float32 into the int32 Q16.16 plane): a coarse discrete set (ties, and
    # spreads exactly at a threshold) on most tiles, continuous on the rest
    coarse = rng.choice(np.array(_DISCRETE), size=(h, w))
    fine = rng.random((h, w)) * 9.0
    pick = rng.random((h, w)) < 0.75
    vals = np.where(pick, coarse, fine).astype(np.float32)
    g.atmosphere[:] = afx.quantize(vals)
    return g


def _burst_cases(seed, n):
    rng = np.random.default_rng(seed)
    for k in range(n):
        h, w = int(rng.integers(6, 40)), int(rng.integers(6, 40))
        yield _random_map(rng, h, w, ambient=bool(k % 3 == 2),
                          wall_bias=float(rng.random()))


@pytest.mark.parametrize("seed", range(5))
def test_burst_list_equals_the_pre143_loop_on_random_maps(seed):
    """PROPERTY: for any map the vectorised scan returns the pre-#143 loop's
    list -- the same tiles, in the same (descending spread, scan-order-stable)
    order, as plain Python int pairs -- with and without a cap. Breaks if the
    side rules (out of bounds / solid neighbours are not sides, vacuum 0.0,
    ambient ``p_amb_real``), the strict ``> t``, the ``t <= 0`` skip, the
    float64 arithmetic, the stable tie order or the cap-after-sort change."""
    nonempty = 0
    ambient_seen = vacuum_seen = 0
    for g in _burst_cases(100 + seed, 12):
        want = ref.find_burst_walls_reference(g)
        got = g.find_burst_walls()
        assert got == want
        assert all(type(a) is int and type(b) is int for a, b in got)
        for cap in (1, 2, 5, 40):
            assert g.find_burst_walls(max_pops=cap) == ref.find_burst_walls_reference(g, cap)
        nonempty += bool(want)
        ambient_seen += bool(g.is_ambient.any())
        vacuum_seen += bool(g.is_vacuum.any())
    assert nonempty, "the sweep must contain maps that actually burst walls"
    assert vacuum_seen or ambient_seen


def test_cap_edge_values_follow_list_slicing():
    """PROPERTY: ``max_pops`` is the old ``failing[:max_pops]`` after the sort:
    0 returns nothing, a cap above the count returns everything, and a
    negative cap slices from the end exactly as a list does. Breaks if the cap
    is applied before the sort or by another rule."""
    checked = 0
    for g in _burst_cases(7, 10):
        full = ref.find_burst_walls_reference(g)
        if len(full) < 3:
            continue
        for cap in (0, 1, len(full) - 1, len(full), len(full) + 5, -1, -2):
            assert g.find_burst_walls(max_pops=cap) == ref.find_burst_walls_reference(g, cap)
        checked += 1
    assert checked


def test_equal_spreads_keep_scan_order():
    """PROPERTY: tiles with exactly equal spreads come out in row-major scan
    order (the old stable ``sort(reverse=True)``). Breaks if the sort is
    unstable or the scan is column-major. A map of many identical thin
    partitions, every spread equal, is all ties."""
    h, w = 14, 17
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1
    for x in (4, 8, 12):
        tm[1:h - 1, x] = 2
    g = GameMap(LevelData(name="ties", version=LEVEL_FORMAT_VERSION, path=Path("."),
                          tilemap=tm, tile_size_m=1.0, diffuse_path=Path(".")))
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            p = 5.0 if (x // 4) % 2 == 0 else 1.0
            g.atmosphere[y, x] = afx.quantize(np.float32(p))
    got = g.find_burst_walls()
    assert got == ref.find_burst_walls_reference(g)
    assert len(got) > 20 and got == sorted(got), "all spreads equal -> pure scan order"


def test_thresholds_at_zero_negative_and_fractional(monkeypatch):
    """PROPERTY: a material with ``burst_threshold <= 0`` never bursts, a
    fractional float32 threshold compares in float64 exactly as
    ``float(thresh[id])`` did, and a spread exactly equal to a threshold does
    NOT burst (strict ``>``). Breaks if the threshold is compared in float32,
    ``<=`` replaces ``<``, or the ``t <= 0`` skip goes."""
    checked = 0
    for k, g in enumerate(_burst_cases(31, 12)):
        thr = np.array(g.materials.burst_threshold, copy=True)
        thr[2] = np.float32(0.1)          # wood: fractional, below most spreads
        thr[4] = np.float32(-3.0)         # steel: negative -> never
        thr[5] = np.float32(1.0)          # glass: equals the 1.0 spreads exactly
        thr[6] = np.float32(0.3)          # furniture: not a float32-exact value
        with monkeypatch.context() as m:
            m.setattr(g.materials, "burst_threshold", thr)
            want = ref.find_burst_walls_reference(g)
            assert g.find_burst_walls() == want
            checked += bool(want)
    assert checked


def test_no_wall_to_scan_returns_empty():
    """PROPERTY: a map with no solid tile, and a map whose solid tiles are all
    materials that never burst, return ``[]`` (never an error, never a
    tile). Breaks if an empty candidate set is not handled."""
    for tm in (np.zeros((8, 8), dtype=np.int32),
               np.pad(np.zeros((6, 6), dtype=np.int32), 1, constant_values=1)):
        g = GameMap(LevelData(name="empty", version=LEVEL_FORMAT_VERSION, path=Path("."),
                              tilemap=tm, tile_size_m=1.0, diffuse_path=Path(".")))
        assert g.find_burst_walls() == ref.find_burst_walls_reference(g) == []
        assert g.find_burst_walls(max_pops=3) == []
