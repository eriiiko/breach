"""The LIGHT extinction's doors and planes (ray-engine-v2 P6a; the P6a brief,
decision 6 and section 4; design v3 §2.3, §4.1, §4.4, §5).

Two doors feed the sweep's light channels, each quantizing ONCE:
  * the MATERIALS door -- `light_atten` [R, G, B] per row, validated in [0, 1]
    on every channel and quantized to `MaterialTable.light_atten_q16` through
    optics_fixed; projected per tile as `GameMap.light_atten_q`, patched on a
    tile change, and MAX-stamped by units into `GameMap.dyn_light_atten_q`;
  * the GASES door -- `GasTable.light_absorb_q16` / `light_glow_q16`, the RGB
    absorption and scatter albedo with the [smoke] dials folded in (one owner,
    the dials the render medium reads), inside the heat door's [0, 4096].

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_light_extinction_doors.py -q
"""
from __future__ import annotations

import copy
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release",
           ROOT / "docs" / "ray_engine_v2_scheme_study_2026-09-13"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import optics_fixed, unit_fixed  # noqa: E402
from simulation.gases import GasTable, HEAT_ABSORB_MAX  # noqa: E402
from simulation.materials import MaterialTable  # noqa: E402


def _cfg():
    with open(ROOT / "config.toml", "rb") as fh:
        return tomllib.load(fh)


def _mat_table_with(row_name, **overrides):
    cfg = _cfg()
    mats = copy.deepcopy(cfg["materials"])
    mats[row_name].update(overrides)
    return MaterialTable(mats, cfg["physics"]["thermal"], cfg["physics"]["fire"],
                         cfg["physics"]["combustion"])


# ---------------------------------------------------------------------------
# the materials door
# ---------------------------------------------------------------------------
def test_the_materials_door_quantizes_light_atten_once_per_channel():
    """PROPERTY: the shipped table's `light_atten_q16` is (N, 3) int32 and equals
    optics_fixed.quantize of the real column, channel for channel -- glass
    0.1, a crate 0.55, walls opaque, air and foliage clear.

    BREAKS IF: the column is quantized anywhere but the optics boundary module,
    a channel is dropped or reordered, or the door stops producing it.
    """
    cfg = _cfg()
    tbl = MaterialTable(cfg["materials"], cfg["physics"]["thermal"],
                        cfg["physics"]["fire"], cfg["physics"]["combustion"])
    assert tbl.light_atten_q16.dtype == np.int32
    assert tbl.light_atten_q16.shape == (tbl.n, 3)
    assert np.array_equal(tbl.light_atten_q16, optics_fixed.quantize(tbl.light_atten))
    vals = {int(v) for v in tbl.light_atten_q16.ravel()}
    assert 0 in vals and optics_fixed.FP_ONE in vals and len(vals) >= 3


@pytest.mark.parametrize("bad", [[-0.01, 0.0, 0.0], [0.5, 1.2, 0.5], [0.1, 0.1, float("nan")]])
def test_light_atten_outside_the_unit_interval_is_rejected_by_name(bad):
    """PROPERTY: a `light_atten` channel outside [0, 1] (or not finite) is
    REFUSED at the materials door, by the row's name -- the sweep's light
    positivity rests on 0 <= a_c <= d_c <= ONE exactly as heat's does.

    BREAKS IF: the door clamps instead of raising, or checks only one channel.
    """
    with pytest.raises(ValueError, match="glass"):
        _mat_table_with("glass", light_atten=bad)


def test_gamemap_projects_patches_and_stamps_the_light_twin():
    """PROPERTY: GameMap.light_atten_q is the material column projected per
    tile (h, w, 3); a tile whose material changes is patched in place and the
    stamped plane raised to it by MAX (a <= d holds before the next stamp); the
    resting stamped plane equals the static one.

    BREAKS IF: the projection reads the float column, the tile patch forgets
    the light twin, or the patch LOWERS a stamped value.
    """
    import breach_physics as bp  # noqa: F401  (the engine the map binds)
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.materials import MAT_GLASS, MAT_HULL
    h = w = 10
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:9, 1:9] = 4
    sim = Simulation(LevelData(name="doors", version="1", path=Path("."), tilemap=tm,
                               tile_size_m=1.0, diffuse_path=Path(".")),
                     seed=1, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    tbl = g.materials
    assert np.array_equal(g.light_atten_q, tbl.light_atten_q16[g.material])
    assert np.array_equal(g.dyn_light_atten_q, g.light_atten_q)
    assert np.all(g.light_atten_q[0, 0] == optics_fixed.FP_ONE)       # hull
    g.material[4, 4] = MAT_GLASS
    g.on_tile_changed(4, 4)
    assert list(g.light_atten_q[4, 4]) == list(tbl.light_atten_q16[MAT_GLASS])
    assert np.all(g.dyn_light_atten_q[4, 4] >= g.light_atten_q[4, 4])
    g.dyn_light_atten_q[5, 5] = optics_fixed.FP_ONE                   # a body stood here
    g.material[5, 5] = MAT_GLASS
    g.on_tile_changed(5, 5)
    assert np.all(g.dyn_light_atten_q[5, 5] == optics_fixed.FP_ONE)   # never lowered
    assert MAT_HULL != MAT_GLASS


# ---------------------------------------------------------------------------
# the gases door
# ---------------------------------------------------------------------------
def test_the_gases_door_folds_the_smoke_dials_into_the_light_columns():
    """PROPERTY (decision 6): GasTable.light_absorb_q16[g][c] ==
    quantize(absorption[g][c] * smoke_absorption[c] * smoke_absorb_scale) and
    light_glow_q16[g][c] == quantize(scatter_albedo[g][c] *
    smoke_scatter_albedo[c]) for every shipped gas and channel -- the [smoke]
    dials folded in HERE, once, the same dials the render medium reads; the
    bulk pair carries no optics (0). A table built from rows alone still
    carries the shipped dials (it reads CFG.smoke).

    BREAKS IF: a dial is dropped, applied twice, or the columns are quantized
    anywhere else; the absorb scale (1.4 shipped) moves the light column
    without moving this, or vice versa.
    """
    cfg = _cfg()
    sm = cfg["smoke"]
    scale = float(sm["smoke_absorb_scale"])
    for tbl in (GasTable.from_config(), GasTable(cfg["gases"])):
        assert tbl.light_absorb_q16.shape == (tbl.n, 3) == tbl.light_glow_q16.shape
        for i in range(tbl.n):
            for c in range(3):
                va = float(tbl.absorption[i, c]) * float(sm["smoke_absorption"][c]) * scale
                vg = float(tbl.scatter_albedo[i, c]) * float(sm["smoke_scatter_albedo"][c])
                assert tbl.light_absorb_q16[i, c] == unit_fixed.quantize_scalar(va)
                assert tbl.light_glow_q16[i, c] == unit_fixed.quantize_scalar(vg)
    tbl = GasTable.from_config()
    from simulation.gases import O2, INERT_N2, SMOKE
    assert not np.any(tbl.light_absorb_q16[[O2, INERT_N2]])
    assert not np.any(tbl.light_glow_q16[[O2, INERT_N2]])
    assert np.all(tbl.light_absorb_q16[SMOKE] > 0)
    # a moved dial moves the column: double the absorb scale
    tbl2 = GasTable(cfg["gases"], dict(sm, smoke_absorb_scale=2 * scale))
    assert np.all(np.abs(tbl2.light_absorb_q16[SMOKE] - 2 * tbl.light_absorb_q16[SMOKE]) <= 1)


def test_the_reference_gate_uses_the_shipped_light_columns():
    """PROPERTY: the integer reference's G18 light columns (soot and steam) are
    the shipped table's, to the count -- the spec measures the game's numbers.

    BREAKS IF: [gases.smoke] / [gases.steam] optics or the [smoke] dials move
    without the gate's constants (re-derive them), or vice versa.
    """
    import sweep_ref_q_gates as G
    from simulation.gases import SMOKE, STEAM
    tbl = GasTable.from_config()
    assert G.LIGHT_ABSORB[0] == [int(v) for v in tbl.light_absorb_q16[SMOKE]]
    assert G.LIGHT_ABSORB[1] == [int(v) for v in tbl.light_absorb_q16[STEAM]]
    assert G.LIGHT_GLOW[0] == [int(v) for v in tbl.light_glow_q16[SMOKE]]
    assert G.LIGHT_GLOW[1] == [int(v) for v in tbl.light_glow_q16[STEAM]]


@pytest.mark.parametrize("field,bad", [("absorption", [-0.1, 0.5, 0.5]),
                                       ("scatter_albedo", [0.5, 5000.0, 0.5])])
def test_the_gases_door_rejects_a_light_source_or_an_unbounded_coefficient(field, bad):
    """PROPERTY: a negative light coefficient (a gas that would be a light
    SOURCE) or one above HEAT_ABSORB_MAX (the density sum's int64 bound) is
    refused at the gases door, by the gas's name.

    BREAKS IF: the door clamps, or checks only the absorption column.
    """
    cfg = _cfg()
    gases = copy.deepcopy(cfg["gases"])
    gases["steam"][field] = bad
    with pytest.raises(ValueError, match="steam"):
        GasTable(gases, cfg["smoke"])
    assert HEAT_ABSORB_MAX == 4096.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
