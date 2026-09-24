"""The `[gases.*] heat_absorb` column at the gas-table door (ray-engine-v2 P5a,
design v3 §6.3).

`heat_absorb` is a gas's HEAT extinction per unit of its own density: on a GAS
cell the radiation sweep reads a_gas = min(ONE, Σ_g heat_absorb_q16[g] · N_g >> 16),
0 where the bulk count is below gas_energy.h's N_EPS_RAW (the sweep side is gated
bit for bit against the integer reference in test_radiation_sweep_reference.py).
This file owns the DOOR: validation, the one quantization, the bound's single
value across its three homes, and the dormancy the golden rests on.

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


def test_no_shipped_gas_absorbs_heat_while_nothing_consumes_gas_rad_net():
    """PROPERTY (P5a's dormancy): every shipped gas has heat_absorb == 0, because
    until P5b opens the temperature fold's gas branch NOTHING consumes a gas
    cell's rad_net -- a non-zero value would take radiation out of the stream
    (smoke shadowing a crate) with no book for it to land in: an uncounted sink,
    exactly what the materials door refuses for `heat_atten > 0` on a gas-regime
    row. This is also why GOLDEN_AGGREGATE does not move at P5a.

    WHEN THIS MUST CHANGE: P5b, in the patch that opens the fold's `ts` mask to
    gas cells, replaces this with the books gate (the #54 closure identity with
    gas absorbing). Setting a value > 0 BEFORE that is the change it exists to
    catch.
    """
    tbl = GasTable.from_config()
    assert np.all(tbl.heat_absorb_q16 == 0), {
        tbl.names[g]: int(tbl.heat_absorb_q16[g])
        for g in range(tbl.n) if tbl.heat_absorb_q16[g] != 0}
    # ...and the physically-zero rows are zero for their OWN reason: the bulk
    # pair (homonuclear diatomics) carries no optics at all
    for gid in np.flatnonzero(tbl.conservative):
        assert float(tbl.heat_absorb[gid]) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
