"""EOS refactor P1 — calibration + range witnesses (docs/eos_refactor_design.md §2.1).

Pure-arithmetic unit tests, no engine/GameMap involved: pin the ``[physics.eos]``
constants (``C``, ``t_amb_k``) against the §2.1 calibration contract BEFORE
anything downstream (P2's unified temperature / P3's compressible solver)
consumes them. Nothing physically consumes ``C`` yet — this is the calibration
WITNESS the spec calls for, not a live solver path.

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_eos_p1_calibration.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import CFG  # noqa: E402
from simulation import gas_fixed  # noqa: E402


def test_eos_C_calibration_reproduces_ambient_one_atm():
    """quantize(C * N_amb * T_amb_k) == quantize(1.0) within 1 count (§2.1).

    N_amb (the ambient TOTAL bulk density, O2+inert_N2) is 1.0 by construction
    — the 21/79 split (gamemap._update_caches) sums back to exactly today's
    atmosphere==1.0 Q16.16 scale (13763 + 51773 == 65536). The calibration
    witness pins C against that same real-valued scale directly, exactly as
    §2.1 specifies: "choose C such that quantize(C * N_amb * T_amb) ==
    quantize(1.0)".
    """
    C = float(CFG.physics.eos.C)
    t_amb_k = float(CFG.physics.eos.t_amb_k)
    N_amb = 1.0

    lhs_q = gas_fixed.quantize_scalar(C * N_amb * t_amb_k)
    rhs_q = gas_fixed.quantize_scalar(1.0)
    assert abs(lhs_q - rhs_q) <= 1, (
        f"C*N_amb*T_amb_k calibration off by {abs(lhs_q - rhs_q)} counts "
        f"(C={C!r}, t_amb_k={t_amb_k!r}, lhs_q={lhs_q}, rhs_q={rhs_q})")


def test_eos_C_is_reciprocal_of_ambient_temperature():
    """C is defined as 1/t_amb_k (config.toml's derivation comment) — pin the
    relationship directly so a future edit to one constant without the other
    silently breaks the calibration only at the OTHER test's tolerance edge."""
    C = float(CFG.physics.eos.C)
    t_amb_k = float(CFG.physics.eos.t_amb_k)
    assert abs(C - 1.0 / t_amb_k) < 1e-12


def test_ambient_o2_plus_n2_equals_legacy_atmosphere_scale():
    """The 21/79 split's quantized counts sum to EXACTLY FP_ONE (65536) — not
    just close. This is the concrete Q16.16-level twin of the C-calibration
    witness above: quantize(0.21) + quantize(0.79) == quantize(1.0) exactly,
    because the two roundings' fractional remainders (.56 and .44) sum to 1.0."""
    o2_q = gas_fixed.quantize_scalar(0.21)
    n2_q = gas_fixed.quantize_scalar(0.79)
    assert o2_q + n2_q == gas_fixed.FP_ONE == gas_fixed.quantize_scalar(1.0)


def test_o2_tank_spike_fits_q16_16_with_headroom():
    """A 200x-ambient O2 spike (a ruptured tank — §5's emergent "O2-tank
    rupture -> fireball" payoff) must fit the Q16.16 range (+-32768 real
    units) with real headroom — the P1 range check the spec calls for.
    Ambient O2 is 0.21; a 200x spike is 42.0, tiny against the +-32768
    ceiling (comfortably inside even a 100x-headroom bound)."""
    ambient_o2 = 0.21
    spike = 200.0 * ambient_o2   # == 42.0

    Q16_16_MAX_REAL = 32768.0    # the format's representable real-value ceiling
    HEADROOM_FACTOR = 100.0      # "comfortable headroom", not just "doesn't overflow"

    assert spike * HEADROOM_FACTOR < Q16_16_MAX_REAL, (
        f"O2-tank spike {spike} leaves < {HEADROOM_FACTOR}x headroom under the "
        f"Q16.16 ceiling {Q16_16_MAX_REAL}")

    spike_q = gas_fixed.quantize_scalar(spike)
    # Representable as int32 (the storage type) ...
    assert -(1 << 31) < spike_q < (1 << 31)
    # ... and nowhere near even the FORMAT's own +-32768 overflow edge.
    assert abs(spike_q) < gas_fixed.FP_ONE * Q16_16_MAX_REAL / HEADROOM_FACTOR


if __name__ == "__main__":
    test_eos_C_calibration_reproduces_ambient_one_atm()
    test_eos_C_is_reciprocal_of_ambient_temperature()
    test_ambient_o2_plus_n2_equals_legacy_atmosphere_scale()
    test_o2_tank_spike_fits_q16_16_with_headroom()
    print("OK: EOS P1 calibration + range witnesses passed")
