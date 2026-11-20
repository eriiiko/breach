"""#29 P4b — the stamp rows: what the host hands the device stamp kernel.

``GameMap.stamp_units`` (the CPU oracle) records the rows it flattened the
living units into; ``GameMap.take_stamp_rows`` hands them to the resident tick
ONCE, packed for the device (``(R, 9)`` int32: env, y, x, perm bits, wabsorb
bits, heat_q, l0, l1, l2). The device kernel's law is written out below in
numpy (``_device_law``) so the rows' sufficiency is a property of the ordinary
suite, with no GPU:

* ``test_the_rows_alone_rebuild_the_five_planes`` -- PROPERTY: applying the
  device law (reset to the static baselines, then the rows in order, the
  permeability a WRITE of ``min(up, sp)`` so the last row wins, the other
  three planes MAXes) to the rows reproduces the CPU stamp's five planes byte
  for byte, with overlapping units of different permeability on walls and off
  the map edge. BREAKS IF the rows lose a column the CPU stamp reads (a unit's
  attribute, the in-bounds filter), their order stops being the stamp's, or
  the CPU stamp's combine direction changes without the device law.
* ``test_the_rows_are_handed_over_once`` -- PROPERTY: a stamp's rows are
  consumed by one take; the second take returns None, so a tick that never
  re-stamped cannot replay the last tick's rows. BREAKS IF a stale row set
  survives a take.
* ``test_a_topology_change_after_the_stamp_withholds_the_rows`` -- PROPERTY:
  when a tile changes between the stamp and the take (slot 6b's payload walls)
  the static baselines the device would stamp from are not the ones the CPU
  stamp used, so no rows are handed over and the resident tick uploads the
  host's planes instead (which is what the CPU physics saw). BREAKS IF the
  device stamp were allowed to run on baselines newer than the oracle's.
* ``test_without_the_cpp_stamp_there_are_no_rows`` -- PROPERTY: the Python
  reference stamp (no engine bound, or ``use_cpp_stamp`` off) records no rows,
  so the resident tick falls back to uploading its planes. BREAKS IF the
  device stamp could run from a stamp that never built the rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pytest

import breach_physics as bp
from level_loader import LevelData
from simulation.gamemap import GameMap
from simulation.physics_runner import PhysicsRunner
from simulation.unit import Unit

H, W = 24, 32
PLANES = ("obstacles", "dyn_permeability", "dyn_wave_absorb",
          "dyn_heat_atten_q", "dyn_light_atten_q")


def _map(bind=True) -> GameMap:
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:H - 2, 2:W - 2] = 1
    tm[3:H - 3, 3:W - 3] = 4
    tm[10:16, 15] = 2                      # a wood wall segment
    g = GameMap(LevelData(name="stamp_rows", version="1", path=Path("."),
                          tilemap=tm, tile_size_m=1.0 / 3.0,
                          diffuse_path=Path(".")))
    if bind:
        g.bind_physics_engine(PhysicsRunner(bp).engine)
    return g


def _units():
    def mk(name, x, y, fp, **attrs):
        u = Unit(name, x, y, team=0, footprint=fp)
        for k, v in attrs.items():
            setattr(u, k, v)
        return u
    dead = mk("dead", 8.0, 8.0, 3)
    dead.alive = False
    return [
        mk("a", 8.0, 8.0, 3, permeability=0.2, wave_absorb=0.9, heat_atten=0.4,
           light_atten=(0.9, 0.2, 0.5)),
        mk("b", 9.0, 8.0, 3, permeability=0.9, wave_absorb=0.1, heat_atten=0.8,
           light_atten=(0.1, 0.8, 0.3)),
        mk("c", 15.0, 12.0, 5, permeability=0.3),          # over the wood
        mk("d", 1.0, 1.0, 5, wave_absorb=0.7),              # off the hull corner
        mk("e", 30.0, 22.0, 3),                              # off the far edge
        dead,
    ]


def _device_law(g: GameMap, rows: np.ndarray) -> dict:
    """The kernel's law in numpy (cuda_stamp.cu's header comment, line for
    line): reset to the baselines, then the rows in order."""
    out = {
        "obstacles": g.permeability <= 0.0,
        "dyn_permeability": g.permeability.copy(),
        "dyn_wave_absorb": g.wave_absorb.copy(),
        "dyn_heat_atten_q": g.heat_atten_q.copy(),
        "dyn_light_atten_q": g.light_atten_q.copy(),
    }
    for env, y, x, perm_bits, wabs_bits, heat_q, l0, l1, l2 in rows.tolist():
        assert env == 0 and 0 <= y < H and 0 <= x < W
        up = np.array([perm_bits], dtype=np.int32).view(np.float32)[0]
        uw = np.array([wabs_bits], dtype=np.int32).view(np.float32)[0]
        sp = g.permeability[y, x]
        out["dyn_permeability"][y, x] = up if up < sp else sp     # the last row wins
        cur = out["dyn_wave_absorb"][y, x]
        out["dyn_wave_absorb"][y, x] = cur if cur >= uw else uw
        out["dyn_heat_atten_q"][y, x] = max(int(out["dyn_heat_atten_q"][y, x]),
                                            heat_q)
        for c, lq in enumerate((l0, l1, l2)):
            out["dyn_light_atten_q"][y, x, c] = max(
                int(out["dyn_light_atten_q"][y, x, c]), lq)
    return out


def test_the_rows_alone_rebuild_the_five_planes():
    g = _map()
    units = _units()
    for t in range(6):
        for i, u in enumerate(units):
            u.set_pos(u.x + ((i + t) % 3) - 1, u.y + ((2 * i + t) % 3) - 1)
        g.stamp_units(units)
        rows = g.take_stamp_rows()
        assert rows is not None and rows.dtype == np.int32 and rows.shape[1] == 9
        assert rows.shape[0] > 0
        law = _device_law(g, rows)
        for name in PLANES:
            assert np.array_equal(law[name], getattr(g, name)), (t, name)


def test_the_rows_are_handed_over_once():
    g = _map()
    g.stamp_units(_units())
    assert g.take_stamp_rows() is not None
    assert g.take_stamp_rows() is None
    g.stamp_units([])
    empty = g.take_stamp_rows()
    assert empty is not None and empty.shape == (0, 9)


def test_a_topology_change_after_the_stamp_withholds_the_rows():
    g = _map()
    g.stamp_units(_units())
    g.destroy_wall(12, 15)                 # slot 6b's payload wall, say
    assert g.take_stamp_rows() is None
    g.stamp_units(_units())                # the next tick's stamp is current
    assert g.take_stamp_rows() is not None


def test_without_the_cpp_stamp_there_are_no_rows():
    unbound = _map(bind=False)
    unbound.stamp_units(_units())
    assert unbound.take_stamp_rows() is None
    g = _map()
    g.use_cpp_stamp = False
    g.stamp_units(_units())
    assert g.take_stamp_rows() is None
