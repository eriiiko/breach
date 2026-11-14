"""The extinction planes' INGRESS invariants (ray-engine-v2 P1, design v3 §2.3):
the materials door and the optics boundary module reject what the sweep's
positivity cannot survive, each by NAME.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_optics_ingress.py -q
"""
from __future__ import annotations

import copy
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import optics_fixed  # noqa: E402
from simulation.materials import MaterialTable  # noqa: E402


def _cfg_dicts():
    with open(ROOT / "config.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    return (cfg["materials"], cfg["physics"]["thermal"], cfg["physics"]["fire"],
            cfg["physics"]["combustion"])


def _table_with(row_name, **overrides):
    mats, thermal, fire, comb = _cfg_dicts()
    mats = copy.deepcopy(mats)
    mats[row_name].update(overrides)
    return MaterialTable(mats, thermal, fire, comb)


def test_shipped_config_passes_both_invariants_and_carries_a_q16_column():
    """PROPERTY: the shipped material table builds, every heat_atten is in
    [0, 1], every absorbing row is a thermal solid, and the pre-quantized Q16
    column equals optics_fixed.quantize of the real column.

    BREAKS IF: a shipped row violates an invariant (that is a FINDING to
    report, not a config edit), or the column stops being quantized at the door.
    """
    mats, thermal, fire, comb = _cfg_dicts()
    tbl = MaterialTable(mats, thermal, fire, comb)
    assert np.all(tbl.heat_atten >= 0.0) and np.all(tbl.heat_atten <= 1.0)
    assert np.all((tbl.heat_atten == 0.0) | tbl.thermal_solid)
    assert tbl.heat_atten_q16.dtype == np.int32
    assert np.array_equal(tbl.heat_atten_q16, optics_fixed.quantize(tbl.heat_atten))
    assert tbl.heat_atten_q16.min() == 0                        # air is transparent
    # T5b: the column's max used to be pinned at FP_ONE ("a wall is opaque").
    # That was true only while every structural row carried the placeholder 1.0.
    # `heat_atten` is an EMISSIVITY, and no real surface is a perfect black
    # body: steel is 0.85 (Incropera A.11), wood 0.90. The property that
    # survives is that it is a real emissivity -- strictly inside (0, 1) on
    # every absorbing row.
    absorbing = tbl.heat_atten > 0.0
    assert absorbing.any()
    assert np.all(tbl.heat_atten[absorbing] < 1.0), (
        "a heat_atten of exactly 1.0 is a perfect black body -- if a row really "
        "wants that, say why in its comment")


def test_flammable_thermal_solid_without_a_loss_channel_is_rejected_by_name():
    """PROPERTY (thermal model v2 §6 item 1, R12): the material door REFUSES
    ``flammable && thermal_solid && heat_atten == 0``.

    WHY IT IS A DOOR AND NOT A WARNING: after R1 deletes `cool_shift` and R10
    puts conduction at real physical rates, in-plane radiation is the ONLY
    meaningful loss channel a solid has. Such a row is an ENERGY RATCHET --
    combustion heats it, nothing cools it, it climbs to T_MAX_PHYS and re-lights
    its neighbours forever, with every channel involved correctly booked, so
    nothing else in the engine can notice.

    BREAKS IF: a row is authored that could ratchet (design v2 §6 item 1).
    """
    with pytest.raises(ValueError, match=r"materials\.foliage.*FLAMMABLE"):
        _table_with("foliage", heat_atten=0.0)
    with pytest.raises(ValueError, match=r"materials\.furniture.*FLAMMABLE"):
        _table_with("furniture", heat_atten=0.0)
    # ...and the three legs of the conjunction each disarm it on their own, so
    # the door is not a blanket "heat_atten must be positive":
    assert _table_with("glass", heat_atten=0.0) is not None      # not flammable
    assert _table_with("air", heat_atten=0.0) is not None        # not a solid
    assert _table_with("foliage", heat_atten=0.01) is not None   # has a channel


def test_the_shipped_table_has_no_energy_ratchet():
    """PROPERTY: EVERY shipped flammable thermal solid has a positive
    heat_atten -- the invariant above, asserted on what actually ships rather
    than on a synthetic row.

    BREAKS IF: R12's foliage change is reverted (it was the only violator).
    """
    mats, thermal, fire, comb = _cfg_dicts()
    tbl = MaterialTable(mats, thermal, fire, comb)
    ratchets = [n for n, f, ts, a in zip(tbl.names, tbl.flammable,
                                         tbl.thermal_solid, tbl.heat_atten)
                if f and ts and a <= 0.0]
    assert not ratchets, f"flammable thermal solids with no loss channel: {ratchets}"
    # non-vacuous: there ARE flammable thermal solids to check
    assert sum(1 for f, ts in zip(tbl.flammable, tbl.thermal_solid) if f and ts) >= 3


@pytest.mark.parametrize("bad", [1.5, -0.1])
def test_heat_atten_outside_unit_interval_is_rejected_by_name(bad):
    """PROPERTY: a row with heat_atten outside [0, 1] raises ValueError naming
    the material and the key.

    BREAKS IF: the door clamps instead of raising (an a > ONE would make
    abs_mat exceed the stream and the sweep's positivity would fail).
    """
    with pytest.raises(ValueError, match=r"materials\.wood\.heat_atten"):
        _table_with("wood", heat_atten=bad)


def test_absorbing_row_in_the_gas_regime_is_rejected_by_name():
    """PROPERTY: heat_atten > 0 with thermal_mass == 0 raises ValueError naming
    the material (an absorbing cell the fold ignores is an uncounted sink).

    BREAKS IF: the implication check is dropped.
    """
    with pytest.raises(ValueError, match=r"materials\.air"):
        _table_with("air", heat_atten=0.5)
    # the converse is legal: a thermal solid that neither absorbs nor emits
    tbl = _table_with("glass", heat_atten=0.0)
    assert tbl.thermal_solid[list(tbl.names).index("glass")]


def test_optics_fixed_is_the_boundary_and_raises_outside_unit_interval():
    """PROPERTY: quantize/quantize_scalar round half away from zero to Q16 on
    [0, 1] (0 -> 0, 1 -> 65536, 0.5 -> 32768), RAISE outside it, and
    dequantize is the exact inverse of the scale.

    BREAKS IF: the door starts clamping, or the scale drifts from 2^16.
    """
    assert optics_fixed.quantize_scalar(0.0) == 0
    assert optics_fixed.quantize_scalar(1.0) == 65536
    assert optics_fixed.quantize_scalar(0.5) == 32768
    assert optics_fixed.quantize_scalar(0.3) == 19661          # round(19660.8)
    assert np.array_equal(optics_fixed.quantize([0.0, 0.25, 1.0]),
                          np.array([0, 16384, 65536], dtype=np.int32))
    assert optics_fixed.dequantize(np.int32(32768)) == 0.5
    assert optics_fixed.dequantize_f32(np.array([65536], np.int32)).dtype == np.float32
    for bad in (1.0000001, -1e-9, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            optics_fixed.quantize_scalar(bad)
    with pytest.raises(ValueError):
        optics_fixed.quantize(np.array([0.5, 2.0]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
