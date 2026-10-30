"""#106 P6b -- the arm chain's checked-in ACTIVATION TABLE (cpp/src/activation_table.inc).

Design docs/octopus_p6_sim_port_design_2026-10-30.md section 6.5: Yekutieli's
stiffening-wave sigmoid S(x) = (1 + tanh x) / 2 is a table written OFFLINE by
tools/gen_activation_table.py (the L° precedent), compiled into the engine and
parsed by the integer reference from the same file. These tests hold the file to
its generator, to the creature row's sample count, and to the shape a sigmoid
must have.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_activation_table.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_activation_table as GEN  # noqa: E402

_REF = ROOT / "docs" / "octopus_integer_reference"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))


def test_the_checked_in_table_is_what_the_generator_makes():
    """Property: activation_table.inc is EXACTLY what the generator computes,
    every entry within one count (a libm tanh may differ in its last ULP), on
    the grid its header declares. Breaks if the .inc is hand-edited, or the
    generator's grid / currency / rounding moves without --write."""
    got = GEN.parse_inc()
    want = GEN.compute_table()
    assert (got["x_max"], got["step_shift"], got["samples"]) == (
        GEN.ACT_X_MAX, GEN.ACT_STEP_SHIFT, GEN.ACT_SAMPLES)
    assert max(abs(a - b) for a, b in zip(got["data"], want)) <= 1


def test_the_row_dial_names_the_tables_size():
    """Property: the creature row's ``[solver] activation_table_samples`` is
    the table's sample count (the row documents what the engine compiles).
    Breaks if either moves alone."""
    from simulation.creatures import get_creature
    assert get_creature("octopus").solver.activation_table_samples == GEN.parse_inc()["samples"]


def test_the_table_is_a_monotone_sigmoid_centred_on_one_half():
    """Property: the samples never decrease, start at 0, end at exactly 1.0
    (65536), sit at 0.5 at x = 0 and are odd-symmetric about it
    (S(-x) = 1 - S(x), within the one-count rounding). Breaks if the grid is
    shifted, a sample is corrupted, or the rounding becomes one-sided."""
    d = GEN.parse_inc()["data"]
    assert all(b >= a for a, b in zip(d, d[1:]))
    mid = (len(d) - 1) // 2
    assert d[0] == 0 and d[-1] == 1 << 16 and d[mid] == 1 << 15
    assert all(abs(d[mid - i] + d[mid + i] - (1 << 16)) <= 1 for i in range(mid + 1))


def test_the_lookup_law_tracks_tanh_within_two_q16_counts():
    """Property: the reference's lookup (clamp, shift index, linear
    interpolation on the low 10 bits, symmetric rounding) evaluates S within
    2 Q16 counts of (1 + tanh x)/2 everywhere on [-10, 10] (the table's
    interpolation error, 1.2e-5 = 0.8 count, plus the samples' and the
    interpolation's roundings, half a count each; measured 1.52), and clamps
    outside [-8, 8]. Breaks if the index shift, the interpolation weight or the
    clamp of the lookup is changed."""
    import chain_ref_q as R
    tab = R.load_activation_table()
    worst = 0.0
    for xq in range(-10 << 16, (10 << 16) + 1, 37):
        s = R.activation_s_q16(xq, tab)
        x = xq / 65536.0
        worst = max(worst, abs(s - 0.5 * (1.0 + math.tanh(x)) * 65536.0))
    assert worst <= 2.0, worst
    assert R.activation_s_q16(-(1 << 30), tab) == 0
    assert R.activation_s_q16(1 << 30, tab) == 1 << 16
