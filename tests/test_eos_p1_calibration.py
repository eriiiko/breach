"""EOS refactor P1 — calibration + range witnesses (docs/eos_refactor_design.md §2.1).

Pure-arithmetic unit tests, no engine/GameMap involved: pin the EOS ambient-
pressure calibration constants (``C``, ``eos_t_amb_k``) against the §2.1
calibration contract BEFORE anything downstream (P2's unified temperature /
P3's compressible solver) consumes them. Nothing physically consumes ``C``
yet — this is the calibration WITNESS the spec calls for, not a live solver
path.

P-K3 (temperature_scale_unification_design_2026-08-13.md §2/§3c, ruling 6):
``[physics.eos]`` no longer carries ``t_amb_k``/``C`` — both moved to
``[physics.temperature_scale]`` (``eos_t_amb_k`` / the derived ``C``
property), read here via the canonical accessor
(:mod:`temperature_scale`), NOT via bare ``CFG.physics.eos`` attribute
access (that would now hard-red — the keys are gone). EOS ambient stays 290
K exactly (a deliberate exception to the unified kelvin_ambient map — T_game
is a ΔT, not an absolute temperature), so every numeric assertion below is
unchanged from pre-P-K3. This file additionally pins the QUANTIZED sim
chain (:func:`ambient.effective_pin`), not just the real-number identity —
the point of the P-K3 rewrite (design §3c): the real-valued
``C * N_amb * T_amb_k == 1.0`` identity is IEEE-exact, but what the EOS
actually materializes each tick is the truncating Q16.16 chain, whose
result (65540, not 65536) is the number that matters operationally.

Run:
    conda run -n data python -m pytest tests/test_eos_p1_calibration.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import temperature_scale  # noqa: E402
from simulation import ambient  # noqa: E402
from simulation import gas_fixed  # noqa: E402


def _load_ts():
    return temperature_scale.load()


def test_eos_t_amb_k_stays_290_under_ruling_6():
    """EOS ambient is a DELIBERATE exception to the unified kelvin_ambient
    map (ruling 6): T_game is a ΔT, and 290 is Q16.16-near-optimal (ambient
    pin 65540, +4 counts) where 293 would be near-worst-case (+96 counts).
    Byte-identity for this whole arc depends on this NOT having moved."""
    ts = _load_ts()
    assert ts.eos_t_amb_k == 290.0


def test_eos_C_calibration_reproduces_ambient_one_atm():
    """quantize(C * N_amb * T_amb_k) == quantize(1.0) within 1 count (§2.1).

    N_amb (the ambient TOTAL bulk density, O2+inert_N2) is 1.0 by construction
    — the 21/79 split (gamemap._update_caches) sums back to exactly today's
    atmosphere==1.0 Q16.16 scale (13763 + 51773 == 65536). The calibration
    witness pins C against that same real-valued scale directly, exactly as
    §2.1 specifies: "choose C such that quantize(C * N_amb * T_amb) ==
    quantize(1.0)".
    """
    ts = _load_ts()
    C = float(ts.C)
    t_amb_k = float(ts.eos_t_amb_k)
    N_amb = 1.0

    lhs_q = gas_fixed.quantize_scalar(C * N_amb * t_amb_k)
    rhs_q = gas_fixed.quantize_scalar(1.0)
    assert abs(lhs_q - rhs_q) <= 1, (
        f"C*N_amb*T_amb_k calibration off by {abs(lhs_q - rhs_q)} counts "
        f"(C={C!r}, t_amb_k={t_amb_k!r}, lhs_q={lhs_q}, rhs_q={rhs_q})")


def test_eos_C_is_reciprocal_of_ambient_temperature():
    """C is defined as 1/eos_t_amb_k (temperature_scale.py's TemperatureScale.C
    property) — pin the relationship directly so a future edit to one
    constant without the other silently breaks the calibration only at the
    OTHER test's tolerance edge."""
    ts = _load_ts()
    C = float(ts.C)
    t_amb_k = float(ts.eos_t_amb_k)
    assert abs(C - 1.0 / t_amb_k) < 1e-12


def test_eos_effective_pin_is_65540_not_65536():
    """The QUANTIZED sim chain, not just the real-number identity (design
    §3c's point). ``ambient.effective_pin`` replays the EOS's own truncating
    Q16.16 chain (``p* = C * N_total * T_abs``, ΔT=0) — at Earth-normal
    ambient (N_total == quantize(1.0) == 65536) this lands on 65540 raw
    (1.000061 atm), NOT 65536: the 4-count offset is the quantization
    lattice image of 1/290, not an error (ambient.py's module docstring;
    tests/test_air_boundary.py:548 pins the same constant independently)."""
    ts = _load_ts()
    n_total_q = gas_fixed.quantize_scalar(1.0)
    assert n_total_q == 65536
    pin = ambient.effective_pin(n_total_q, c=ts.C, t_amb_k=ts.eos_t_amb_k)
    assert pin == 65540


def test_ambient_default_C_and_T_AMB_K_match_the_accessor():
    """ambient.py's DEFAULT_C / DEFAULT_T_AMB_K (pump_system.py's import
    surface) are accessor-derived, not hand-kept copies (P-K3, §3c) — they
    must agree with temperature_scale.load() bit-for-bit."""
    ts = _load_ts()
    assert ambient.DEFAULT_C == ts.C
    assert ambient.DEFAULT_T_AMB_K == ts.eos_t_amb_k


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
    test_eos_t_amb_k_stays_290_under_ruling_6()
    test_eos_C_calibration_reproduces_ambient_one_atm()
    test_eos_C_is_reciprocal_of_ambient_temperature()
    test_eos_effective_pin_is_65540_not_65536()
    test_ambient_default_C_and_T_AMB_K_match_the_accessor()
    test_ambient_o2_plus_n2_equals_legacy_atmosphere_scale()
    test_o2_tank_spike_fits_q16_16_with_headroom()
    print("OK: EOS P1 calibration + range witnesses passed")
