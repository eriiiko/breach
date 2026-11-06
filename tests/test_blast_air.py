"""The blast's read of the air (#31, src/simulation/blast_air.py): a charge reads
O2 and pressure on each open tile at detonation, and the air scales its smoke per
tile and its heat for the whole blast. Each test names the property it protects
and the change that must break it. No test reads a [blast_air] value; those are
placeholders Erik tunes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from level_loader import LevelData  # noqa: E402
from simulation import atmosphere_fixed  # noqa: E402
from simulation.blast_air import ONE, disc_tiles, read_blast_air  # noqa: E402
from simulation.field_edit import EditQueue  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gases import INERT_N2, O2, SMOKE  # noqa: E402
from simulation.payloads import execute_payload  # noqa: E402
from simulation.weapons import PayloadDef  # noqa: E402

CH = (12, 12)
R = 4


def _map():
    tm = np.zeros((24, 24), dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1
    g = GameMap(LevelData(name="blast_air", version="2", path=Path("."),
                          tilemap=tm, tile_size_m=0.333, diffuse_path=Path(".")))
    g.atmosphere[...] = atmosphere_fixed.quantize_scalar(1.0)
    return g


def _thin_o2(g, frac, mask=None):
    """Swap O2 for inert on ``mask``, keeping the tile's bulk N."""
    mask = ~g.solid if mask is None else mask
    bulk = g.gas[O2].astype(np.int64) + g.gas[INERT_N2]
    new_o2 = (bulk * int(frac * 1000)) // 1000
    g.gas[O2][mask] = new_o2[mask]
    g.gas[INERT_N2][mask] = (bulk - new_o2)[mask]


def _read(g):
    return read_blast_air(g, disc_tiles(g, CH[0], CH[1], R))


def _soot_after(g, seed=7):
    row = PayloadDef("t", radius=R, emit_blast_smoke=True, explosive="comp_b",
                     explosive_kg=0.184, blast_soot_g=14.0)
    q = EditQueue()
    rng = np.random.default_rng(seed)
    execute_payload(g, q, [], CH[0], CH[1], row, rng)
    q.flush(g, rng)
    return g.gas[SMOKE].astype(np.int64)


def test_normal_air_changes_nothing():
    """PROPERTY: in ordinary air every factor is exactly 1, so a blast deposits
    what it did before the read existed.

    BREAKS IF: the ramps stop landing exactly on 1 at ambient, which would move
    every digest with a grenade in it.
    """
    read = _read(_map())
    assert read.smoke_scale_q16 is None
    assert read.heat_scale_q16 == ONE


def test_o2_poor_air_leaves_more_smoke_and_less_heat():
    """PROPERTY: at normal pressure, nothing burns the carbon in O2-poor air:
    more smoke, less heat, and the lower the O2 the stronger both are.

    BREAKS IF: the O2 read is dropped or its direction flips.
    """
    totals, heats = [], []
    for frac in (0.21, 0.15, 0.05):
        g = _map()
        _thin_o2(g, frac)
        heats.append(_read(g).heat_scale_q16)
        totals.append(int(_soot_after(g).sum()))
    assert totals[0] < totals[1] < totals[2]
    assert heats[0] > heats[1] > heats[2]


def test_thin_air_leaves_less_smoke_and_vacuum_none():
    """PROPERTY: smoke falls with pressure and vacuum holds none, even with the
    O2 fraction untouched (venting keeps the fraction).

    BREAKS IF: the read uses the O2 fraction alone, the failure #7 fixed for
    fire: a near-empty room reads as normal air.
    """
    totals = []
    for atm in (1.0, 0.3, 0.05):
        g = _map()
        g.atmosphere[...] = atmosphere_fixed.quantize_scalar(atm)
        totals.append(int(_soot_after(g).sum()))
    assert totals[0] > totals[1] > 0
    assert totals[2] == 0


def test_hot_thin_air_at_ambient_pressure_is_normal_air():
    """PROPERTY: density is not the read. Air thinned by heat at 1 atm counts as
    normal, so a second grenade in a hot room is not treated as a thin-air blast.

    BREAKS IF: pressure is replaced by bulk N (the July density trap).
    """
    g = _map()
    g.gas[O2] //= 8
    g.gas[INERT_N2] //= 8
    read = _read(g)
    assert read.smoke_scale_q16 is None and read.heat_scale_q16 == ONE


def test_smoke_follows_the_air_tile_by_tile():
    """PROPERTY: the read is per tile: with the west half of the disc O2-poor and
    the noise equal, the west half gets more smoke than its mirror.

    BREAKS IF: the smoke read is averaged over the disc, which would make every
    grenade's cloud the same shape.
    """
    g = _map()
    west = np.zeros(g.solid.shape, dtype=bool)
    west[:, :CH[1]] = True
    _thin_o2(g, 0.05, west & ~g.solid)
    smoke = _soot_after(g)
    assert smoke[:, :CH[1]].sum() > 1.5 * smoke[:, CH[1] + 1:].sum()


def test_a_blast_in_walls_reads_no_air():
    """PROPERTY: solid tiles are never read as air: a disc with no open tile, like
    a walk with no tile, returns the neutral read.

    BREAKS IF: the disc drops `solid`, so a wall's zero gas reads as vacuum, or an
    empty list divides by zero.
    """
    g = _map()
    g.solid[...] = True
    read = _read(g)
    assert read.smoke_scale_q16 is None and read.heat_scale_q16 == ONE


def test_the_read_takes_the_tiles_it_is_given():
    """PROPERTY: the read is over the tile list, so air the walk did not reach
    changes nothing: O2-poor air on tiles off the list moves no factor.

    BREAKS IF: the read goes back to a disc of its own.
    """
    g = _map()
    off_list = np.zeros(g.solid.shape, dtype=bool)
    off_list[:, CH[1] + 1:] = True
    _thin_o2(g, 0.05, off_list & ~g.solid)
    tiles = [(CH[0], CH[1]), (CH[0], CH[1] - 1)]
    read = read_blast_air(g, tiles)
    assert read.smoke_scale_q16 is None and read.heat_scale_q16 == ONE
