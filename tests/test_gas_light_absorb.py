"""[gases.smoke] light_absorb -- soot's VISIBLE extinction per unit smoke density,
DERIVED (#12 smoke x light, handle 1, 2026-10-04; the derivation is the config
comment beside the value). How much light a given smoke density stops is physics,
never a dial: these tests hold the shipped value to its derivation and hold the
[smoke] dials away from it.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import gases, unit_fixed  # noqa: E402
from simulation.gases import GasTable  # noqa: E402

# The cited constants (config.toml [gases.smoke] carries the prose).
SIGMA_MC_633 = 8.7                   # m2/g at 632.8 nm (Mulholland & Croarkin 2000)
LAMBDA_MC = 632.8                    # nm
LAMBDA_RGB = (611.0, 549.0, 464.0)   # nm, dominant wavelengths of the sRGB primaries
Y_SOOT_WOOD = 0.015                  # g soot / g wood (Tewarson, SFPE Handbook)


def _derived_smoke_light_absorb():
    """tau_c per unit smoke density = sigma_c * M / (tile_w * ceiling_h): soot's
    visible extinction carried to each channel by the 1/lambda law, times the
    soot mass one unit of smoke density holds through the engine's own combustion
    bookkeeping, over the reference tile's face -- at the LIVE dials."""
    from config import CFG
    from simulation.materials import KG_FUEL_PER_N_O2
    m_soot = Y_SOOT_WOOD * KG_FUEL_PER_N_O2 * 1000.0 / float(
        CFG.physics.combustion.soot_yield)                                  # g per unit
    face = float(CFG.physics.thermal.tile_size_ref_m) * float(CFG.physics.water.ceiling_h)
    return np.array([SIGMA_MC_633 * LAMBDA_MC / lam * m_soot / face for lam in LAMBDA_RGB])


def test_smoke_light_absorb_is_the_cited_derivation_at_the_live_dials():
    """PROPERTY: the shipped [gases.smoke] light_absorb IS its derivation --
    Mulholland & Croarkin's 8.7 m2/g at 632.8 nm, per channel by 1/lambda at the
    sRGB primaries, times the engine's soot mass per unit density (11.15 g), over
    the reference tile face (tile_size_ref_m x ceiling_h) -- to 0.1 %, blue
    stopped hardest; and the light channels get exactly that value, quantized
    once.

    BREAKS IF: soot_yield, KG_FUEL_PER_N_O2 or the tile geometry moves without
    the value being re-derived, or the value is hand-tuned away from it.
    """
    derived = _derived_smoke_light_absorb()
    tbl = GasTable.from_config()
    shipped = tbl.light_absorb[gases.SMOKE]
    assert np.allclose(shipped, derived, rtol=1e-3), (shipped, derived)
    assert shipped[0] < shipped[1] < shipped[2]
    assert [int(v) for v in tbl.light_absorb_q16[gases.SMOKE]] == [
        unit_fixed.quantize_scalar(float(v)) for v in shipped]


def test_smoke_dials_never_scale_a_physical_light_absorb():
    """PROPERTY: a gas row that carries light_absorb is physics -- moving the
    [smoke] dials (smoke_absorption, smoke_absorb_scale) leaves its light
    coefficient untouched, while a gas WITHOUT the key still follows them.

    BREAKS IF: the door multiplies a physical light_absorb by a dial again, or
    stops scaling the gases that have no derived value.
    """
    from config import CFG
    base = dict(CFG.smoke.__dict__) if hasattr(CFG.smoke, "__dict__") else dict(CFG.smoke)
    moved = copy.deepcopy(base)
    moved["smoke_absorb_scale"] = float(base.get("smoke_absorb_scale", 1.0)) * 3.0
    moved["smoke_absorption"] = [0.5, 0.5, 0.5]
    a = GasTable(CFG.gases, base)
    b = GasTable(CFG.gases, moved)
    assert np.array_equal(a.light_absorb_q16[gases.SMOKE], b.light_absorb_q16[gases.SMOKE])
    dial_rows = [i for i in range(a.n) if np.isnan(a.light_absorb[i, 0])
                 and a.absorption[i].max() > 0.0]
    assert dial_rows, "no gas left on the dials to check against"
    for i in dial_rows:
        assert not np.array_equal(a.light_absorb_q16[i], b.light_absorb_q16[i]), a.names[i]
