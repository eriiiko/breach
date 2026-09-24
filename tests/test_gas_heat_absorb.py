"""The `[gases.*] heat_absorb` column at the gas-table door (ray-engine-v2 P5a,
design v3 §6.3).

`heat_absorb` is a gas's HEAT extinction per unit of its own density: on a GAS
cell the radiation sweep reads a_gas = min(ONE, Σ_g heat_absorb_q16[g] · N_g >> 16),
0 where the bulk count is below gas_energy.h's N_EPS_RAW (the sweep side is gated
bit for bit against the integer reference in test_radiation_sweep_reference.py).
This file owns the DOOR: validation, the one quantization, the bound's single
value across its three homes -- and, since P5c, the SHIPPED values: only smoke
absorbs, and its coefficient is the cited derivation at the engine's own units.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_gas_heat_absorb.py -q
"""
from __future__ import annotations

import copy
import math
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import gases, unit_fixed  # noqa: E402
from simulation.gases import GasTable, HEAT_ABSORB_MAX  # noqa: E402


def _config_rows():
    """The shipped [gases] rows as a plain dict-of-dicts, deep-copied so a test
    can edit a value without touching another test's table."""
    with open(ROOT / "config.toml", "rb") as fh:
        return copy.deepcopy(tomllib.load(fh)["gases"])


def test_the_door_quantizes_heat_absorb_once_in_the_beam_idiom():
    """PROPERTY: `GasTable.heat_absorb_q16[g]` is the config value quantized ONCE
    by the standard round-half-away-from-zero door (unit_fixed.quantize_scalar,
    the beam_absorb_q16 idiom, door 2) -- including values ABOVE 1, because
    heat_absorb is a coefficient per unit density, not a fraction -- held as a
    contiguous int32 array, one entry per gas id, ready to hand to the engine.

    BREAKS IF: the column is quantized by a different rule (truncation, a float
    kept past the door), clamped to [0, 1], or stored in a form the engine
    binding cannot take by pointer.
    """
    rows = _config_rows()
    values = {"steam": 0.3, "smoke": 17.5, "poison": 0.000011, "fuel_gas": 4096.0}
    for name, v in values.items():
        rows[name]["heat_absorb"] = v
    tbl = GasTable(rows)
    q = tbl.heat_absorb_q16
    assert q.dtype == np.int32 and q.ndim == 1 and q.flags.c_contiguous
    assert len(q) == tbl.n == gases.N_GASES
    for name, v in values.items():
        gid = tbl.name_to_id[name]
        # the float32 column is what the table holds; the door quantizes it
        assert int(q[gid]) == unit_fixed.quantize_scalar(float(np.float32(v))), name
    assert int(q[tbl.name_to_id["smoke"]]) == round(17.5 * 65536)        # not clamped to 1
    assert int(q[tbl.name_to_id["fuel_gas"]]) == 4096 * 65536             # the bound itself


@pytest.mark.parametrize("bad", [-1e-6, -1.0, 4096.5, 4097.0, 1e9, math.inf, -math.inf,
                                 math.nan])
def test_the_door_rejects_a_negative_non_finite_or_oversized_heat_absorb(bad):
    """PROPERTY: the gas door REFUSES (ValueError, naming the gas) a heat_absorb
    that is negative -- a radiation SOURCE, which would break the sweep's
    positivity (a >= 0) -- non-finite, or above HEAT_ABSORB_MAX = 4096, the
    arithmetic's own limit (the sweep's int64 density sum is only provably exact
    below it). Both edges, 0 and 4096, are accepted (the next test's control).
    The door judges the float32 value the table HOLDS, as the materials door
    does for heat_atten: 4096.0001 is 4096.0 there, legal, hence 4096.5 here.

    BREAKS IF: the range check is dropped or loosened, or turned into a silent
    clamp.
    """
    rows = _config_rows()
    rows["smoke"]["heat_absorb"] = bad
    with pytest.raises(ValueError, match=r"gases\.smoke\.heat_absorb"):
        GasTable(rows)


def test_the_door_accepts_both_edges_of_its_range():
    """PROPERTY (the rejections' control): 0.0 and exactly HEAT_ABSORB_MAX pass
    the door on every gas row at once, so the refusals above are about range and
    not about some other broken field.

    BREAKS IF: a bound becomes exclusive, or the table refuses a legal value.
    """
    for edge in (0.0, HEAT_ABSORB_MAX):
        rows = _config_rows()
        for row in rows.values():
            row["heat_absorb"] = edge
        tbl = GasTable(rows)
        assert np.all(tbl.heat_absorb_q16 == unit_fixed.quantize_scalar(edge))


def test_a_row_without_heat_absorb_is_a_load_error():
    """PROPERTY: `heat_absorb` is a REQUIRED column like every other gas column
    (a gas type is a data row; a missing field is a hard load error, never a
    silent default) -- so a new gas cannot ship without an explicit heat
    extinction, even the dormant 0.0.

    BREAKS IF: the column gains a code-side default.
    """
    rows = _config_rows()
    del rows["poison"]["heat_absorb"]
    with pytest.raises(KeyError, match="heat_absorb"):
        GasTable(rows)


def test_the_bound_is_one_number_in_its_three_homes():
    """PROPERTY: the door's bound (gases.HEAT_ABSORB_MAX), the integer
    reference's (sweep_ref_q.HEAT_ABSORB_MAX -> HEAT_ABSORB_Q_MAX) and the C++
    sweep's own re-check (RadiationSweep.HEAT_ABSORB_Q_MAX) are the SAME number,
    and so are the reference's and the sweep's gas-plane limit -- the headroom
    argument is made once and every home enforces exactly it.

    BREAKS IF: any one of them is changed without the others (a door looser than
    the sweep would let a config crash the engine; a sweep looser than the
    reference would measure what the spec cannot).
    """
    import breach_physics as bp
    from _radiation_sweep_harness import R
    assert R.HEAT_ABSORB_MAX == HEAT_ABSORB_MAX
    assert R.HEAT_ABSORB_Q_MAX == unit_fixed.quantize_scalar(HEAT_ABSORB_MAX)
    assert bp.RadiationSweep.HEAT_ABSORB_Q_MAX == R.HEAT_ABSORB_Q_MAX
    assert bp.RadiationSweep.N_GAS_PLANES_MAX == R.N_GAS_PLANES_MAX
    assert gases.N_GASES <= R.N_GAS_PLANES_MAX


# ---- P5c: the shipped values ----------------------------------------------
# The cited constants of [gases.smoke] heat_absorb's derivation (config.toml
# carries the prose). Each is a published number, not a dial:
K_E_IR = 8.5            # Widmann et al. 2003 (NIST), K_e = 8.0-9.0 over 2.8-4.1 um
RHO_SOOT = 1.86e6       # g/m3 -- the soot density Widmann et al. convert K_e with
PLANCK_MEAN_1_OVER_LAMBDA = 360.0 * 1.0369277551433699 / math.pi ** 4   # 3.8322
C2 = 1.438777e-2        # m.K, the second radiation constant
T_FIRE_GAME = 1263      # the arc's fire plateau (design 2.8's 1263-game source)
Y_SOOT_WOOD = 0.015     # g soot / g wood, well ventilated (Tewarson, SFPE Handbook)
SIGMA_MC_633 = 8.7      # m2/g at 632.8 nm (Mulholland & Croarkin 2000) -- the cross-check


def _derived_smoke_heat_absorb():
    """[gases.smoke] heat_absorb from the cited constants and the LIVE dials:
    tau per unit smoke density = sigma_P * M / (tile_w * ceiling_h), with
    sigma_P the Planck mean at the fire plateau and M the soot one unit of smoke
    density holds through the engine's own combustion bookkeeping."""
    from config import CFG
    from simulation.materials import KG_FUEL_PER_N_O2
    t_k = float(CFG.physics.temperature_scale.kelvin_ambient) + T_FIRE_GAME
    sigma_p = K_E_IR * PLANCK_MEAN_1_OVER_LAMBDA * t_k / (C2 * RHO_SOOT)   # m2/g
    m_soot = Y_SOOT_WOOD * KG_FUEL_PER_N_O2 * 1000.0 / float(
        CFG.physics.combustion.soot_yield)                                  # g per unit
    face = float(CFG.physics.thermal.tile_size_ref_m) * float(CFG.physics.water.ceiling_h)
    return sigma_p * m_soot / face, sigma_p, m_soot, face


def test_smoke_heat_absorb_is_the_cited_derivation_at_the_live_dials():
    """PROPERTY (P5c): the shipped [gases.smoke] heat_absorb IS the derivation
    its config comment states -- soot's IR mass-specific extinction (Widmann et
    al. 2003, K_e = 8.5), Planck-meaned at the 1263-game fire plateau (1.894
    m2/g), times the soot mass one unit of smoke density holds through the
    engine's OWN bookkeeping (soot_yield units of smoke per unit N_O2 burned,
    KG_FUEL_PER_N_O2 kg of wood per unit N_O2, 0.015 g of soot per g of wood:
    11.15 g), over the tile's cross-section (tile_size_ref_m x ceiling_h) --
    to the four figures the config carries. And the visible cross-check holds:
    Mulholland & Croarkin's 8.7 m2/g at 632.8 nm, carried to the same Planck
    mean by the same 1/lambda law, lands within 25 % above it (the visible
    scattering the IR measurement does not carry), never below.

    BREAKS IF: soot_yield, KG_FUEL_PER_N_O2, the tile geometry or the ambient
    moves without the value being re-derived (the value is then silently the
    old soot mass), or the value is hand-tuned away from its derivation.
    """
    derived, sigma_p, m_soot, face = _derived_smoke_heat_absorb()
    tbl = GasTable.from_config()
    shipped = float(tbl.heat_absorb[gases.SMOKE])
    assert abs(shipped - derived) <= 0.005 * derived, (shipped, derived, sigma_p, m_soot)
    assert tbl.heat_absorb_q16[gases.SMOKE] == unit_fixed.quantize_scalar(shipped)
    mc = SIGMA_MC_633 * 0.6328e-6 * PLANCK_MEAN_1_OVER_LAMBDA * (
        T_FIRE_GAME + 293.0) / C2
    assert sigma_p < mc <= 1.25 * sigma_p, (sigma_p, mc)
    print(f"\nsmoke heat_absorb: shipped {shipped}, derived {derived:.4f} "
          f"(sigma_P {sigma_p:.4f} m2/g, M {m_soot:.3f} g, face {face:.4f} m2; "
          f"M&C cross-check sigma_P {mc:.4f} m2/g)")


def test_only_smoke_absorbs_heat_and_every_other_gas_is_a_stated_gap():
    """PROPERTY (P5c, replacing P5a's dormancy gate, whose docstring named this
    patch as the one that retires it): the temperature fold's gas branch now
    lands what a gas cell absorbs, so a gas may carry a non-zero heat_absorb --
    and exactly ONE does: smoke, the combustion soot, a grey absorber. Every
    other shipped row stays 0.0, each for a STATED reason in its config comment
    -- the bulk pair physically (homonuclear diatomics do not absorb in the IR;
    N2's lumped burnt products an ACCEPTED GAP), the trace gases as ACCEPTED
    GAPS (steam and fuel gas are band absorbers that are not modelled).

    BREAKS IF: a second gas starts absorbing without its own derivation and
    gate, smoke goes back to 0.0 (the branch would be dormant again), or a
    zero row loses the comment that says why it is zero.
    """
    tbl = GasTable.from_config()
    absorbing = [tbl.names[g] for g in range(tbl.n) if tbl.heat_absorb_q16[g] != 0]
    assert absorbing == ["smoke"], absorbing
    for gid in np.flatnonzero(tbl.conservative):
        assert float(tbl.heat_absorb[gid]) == 0.0
    text = (ROOT / "config.toml").read_text(encoding="utf-8")
    for name in tbl.names:
        if name == "smoke":
            continue
        block = text.split(f"[gases.{name}]", 1)[1].split("\n[", 1)[0]
        line = next(ln for ln in block.splitlines() if ln.startswith("heat_absorb"))
        assert ("ACCEPTED GAP" in line) or ("PHYSICAL" in line), (name, line)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
