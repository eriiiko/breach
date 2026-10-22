"""#88 P1: the trace currency is ONE constant with held-equal twins, its kit
agrees value for value across languages, the tree speaks it (the lints), the
gases door keeps the bulk planes out of every optical density sum, and the two
engine-side writers mint in it.

Design: docs/smoke_unit_rescale_design_2026-10-20.md §4 (where the scale lives),
§10 V1 (the currency is closed), V4 (writer scale), V10 (the door); critiques
A2, A3, A12, C1, C2, C4, C5, C13. The reader side (V3) is
tests/test_trace_reader_invariance.py.

Every test names its property and the change that must break it. Nothing here
pins the value of TRACE_FINE_BITS: each property holds at any currency, which
is what lets the same file gate P1 at 0 bits and the flip at 8.
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import _trace_currency_lint as lint  # noqa: E402
import cuda_harness  # noqa: E402
from _radiation_sweep_harness import R  # noqa: E402
from simulation import fire_fixed, gas_fixed  # noqa: E402
from simulation.gases import (  # noqa: E402
    GAS_NAMES, INERT_N2, N_TRACE_GASES, O2, SMOKE, GasTable,
)

SPEC = ROOT / "tests" / "field_digest_spec.toml"


# ---------------------------------------------------------------------------
# V1 -- the twins agree
# ---------------------------------------------------------------------------
def test_the_four_twins_of_the_trace_currency_agree():
    """PROPERTY (V1, C5): the engine's TRACE_FINE_BITS (the CPU module),
    gas_fixed's LITERAL twin, the integer reference's and the digest spec's
    ``trace_fine_bits`` are one value, and every derived constant (the shift,
    one unit, the saturation limit) follows from it the same way on each side.

    BREAKS IF: any twin is moved alone (the flip touches four places), the
    Python twin becomes a read-through of the bound engine (which reads 0 bits
    on every engine-less GameMap), or the CPU module is stale.
    """
    spec = tomllib.loads(SPEC.read_text(encoding="utf-8"))
    k = int(bp.TRACE_FINE_BITS)
    assert gas_fixed.TRACE_FINE_BITS == k
    assert R.TRACE_FINE_BITS == k
    assert int(spec["trace_fine_bits"]) == k
    assert int(bp.TRACE_SHIFT) == gas_fixed.TRACE_SHIFT == R.TRACE_SHIFT == 16 + k
    assert gas_fixed.TRACE_ONE == R.TRACE_ONE == 1 << (16 + k)
    assert int(bp.TRACE_CELL_MAX) == gas_fixed.TRACE_CELL_MAX == 2 ** 31 - 1
    # the bulk and Q16 families never move with the trace currency
    assert gas_fixed.BULK_ONE == gas_fixed.Q16_ONE == 1 << 16


@pytest.mark.skipif(not cuda_harness.cuda_available(),
                    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present")
def test_the_cuda_module_carries_the_same_trace_currency():
    """PROPERTY (V1 / C4): the CUDA build's breach_physics carries the SAME
    TRACE_FINE_BITS as gas_fixed and its kit agrees with the Python twins value
    for value -- so a stale cpp/build_cuda fails here at the flip instead of
    comparing stale with stale under GPU == CPU.

    BREAKS IF: build_cuda is not rebuilt after the constant moves, or the CUDA
    module's kit differs from the CPU's.
    """
    proc = cuda_harness.run_cuda_script(
        "import breach_physics as bp\n"
        "from simulation import gas_fixed as g\n"
        "assert getattr(bp, 'HAS_CUDA', False), 'not the CUDA module'\n"
        "assert int(bp.TRACE_FINE_BITS) == g.TRACE_FINE_BITS, (bp.TRACE_FINE_BITS, g.TRACE_FINE_BITS)\n"
        "assert int(bp.TRACE_SHIFT) == g.TRACE_SHIFT\n"
        "for v in (0.0, 1e-7, 0.06, 0.15, 0.6, 1.0, 1.5, 38.85, 127.9, -0.4):\n"
        "    assert int(bp.fp_quantize_trace(v)) == g.quantize_trace_scalar(v), v\n"
        "for q in (0, 1, 255, 39322, 65536, 2**31 - 1):\n"
        "    assert int(bp.fp_trace_from_bulk(q)) == g.trace_from_bulk(q), q\n"
        "for s in (0, 1, 2**24 - 1, 2**24, 2**40 + 12345, 2**62):\n"
        "    assert int(bp.fp_trace_density_shr(s)) == g.trace_density_shr(s), s\n"
        "print('TRACE_CURRENCY_CUDA: PASS')\n",
        timeout=120)
    out = proc.stdout + "\n" + proc.stderr
    assert "TRACE_CURRENCY_CUDA: PASS" in out, f"returncode={proc.returncode}\n{out}"


def _quant_inputs():
    """Real densities across the ranges that matter: zero, sub-count, the
    gameplay thresholds, ties at the 2^-17 grid, one unit, the measured abuse,
    just under and over the int32 limit, far beyond it, and negatives."""
    lim = gas_fixed.TRACE_CELL_MAX / gas_fixed.TRACE_ONE
    out = [0.0, 1e-9, 0.06, 0.05, 0.15, 0.6, 1.0, 1.5, 2.0, 10.95, 38.85,
           lim * 0.999, lim, lim * 1.001, 200.0, 1e6, -0.4, -1.0]
    # exact half-count ties (round-half-away must agree on both sides)
    out += [(n + 0.5) / gas_fixed.TRACE_ONE for n in (0, 1, 2, 12345)]
    out += [-(n + 0.5) / gas_fixed.TRACE_ONE for n in (0, 7)]
    return out


def test_the_trace_kit_twins_agree_value_for_value():
    """PROPERTY (V1, A3): quantize_trace, trace_from_bulk and trace_density_shr
    give the same integer in C++ (the binding), in gas_fixed and in the
    integer reference, over thresholds, exact ties, one unit and amounts BEYOND
    the int32 plane -- which the quantize returns unnarrowed (int64), so only
    the counted saturation can ever narrow it.

    BREAKS IF: a twin's rounding drifts (half-even vs half-away), the C++
    quantize narrows to int32 or wraps past the plane's limit, or a shift is
    re-derived with another width on one side.
    """
    for v in _quant_inputs():
        want = gas_fixed.quantize_trace_scalar(v)
        assert int(bp.fp_quantize_trace(v)) == want, v
        assert R.quant_trace(v) == want, v
        assert int(gas_fixed.quantize_trace(np.array([v]))[0]) == want, v
    beyond = 4.0 * gas_fixed.TRACE_CELL_MAX / gas_fixed.TRACE_ONE   # 4x the int32 plane
    big = gas_fixed.quantize_trace_scalar(beyond)
    assert big > gas_fixed.TRACE_CELL_MAX                     # never narrowed
    assert gas_fixed.quantize_trace(np.array([beyond])).dtype == np.int64
    assert int(gas_fixed.quantize_trace(np.array([beyond]))[0]) == big == int(bp.fp_quantize_trace(beyond))
    for q in (0, 1, 255, 13763, 39322, 65536, 2 ** 31 - 1):
        want = q * (1 << gas_fixed.TRACE_FINE_BITS)
        assert int(bp.fp_trace_from_bulk(q)) == gas_fixed.trace_from_bulk(q) \
            == R.trace_from_bulk(q) == want
    with pytest.raises(ValueError):
        gas_fixed.trace_from_bulk(-1)
    for s in (0, 1, 65535, 65536, 2 ** 24 - 1, 2 ** 24, 2 ** 40 + 12345, 2 ** 62):
        want = s >> gas_fixed.TRACE_SHIFT
        assert int(bp.fp_trace_density_shr(s)) == gas_fixed.trace_density_shr(s) \
            == R.trace_density_shr(s) == want
    # the Q16 sensor conversion floors and is exact on S_old << k
    for s_old in (0, 1, 4242, 9830, -5, 2 ** 22):
        lifted = s_old * (1 << gas_fixed.TRACE_FINE_BITS)
        assert gas_fixed.trace_to_q16(lifted) == s_old


# ---------------------------------------------------------------------------
# V1 -- the lints: the tree speaks the currency
# ---------------------------------------------------------------------------
def test_no_deleted_ambiguous_gas_fixed_name_is_referenced():
    """PROPERTY (V1, §4): FP_ONE, FP_ONE_F, SMOKE_MAX_Q, quantize,
    quantize_scalar, dequantize, dequantize_f32 (and FP_SHIFT) of gas_fixed are
    referenced nowhere in src/, renderer/, tools/ or tests/ -- every caller
    names its own currency (trace, bulk or q16).

    BREAKS IF: an ambiguous name is reintroduced (it would silently pick one
    currency for all three kinds of user).
    """
    hits = lint.deleted_name_hits()
    assert not hits, "deleted gas_fixed names in use:\n" + "\n".join(hits)
    for name in lint.DELETED:
        assert not hasattr(gas_fixed, name), name


def test_every_write_into_a_trace_slice_names_the_trace_currency():
    """PROPERTY (V1 write lint, C1): every assignment or += into a trace slice
    (gas[<trace id>], .smoke, gas[:N_TRACE_GASES]) in src/, renderer/, tools/
    and tests/ builds its value from a trace-currency name, is a literal zero /
    fresh zero plane, copies another trace slice, or is MARKED scale-free (a
    raw-count fixture whose property holds at any currency) or real-units (a
    float plane).

    BREAKS IF: a fixture seeds a trace plane from a Q16 literal (``Q = 65536``),
    another boundary module (``fire_fixed``) or a bulk name -- the silent class
    that goes 2^k times thinner at the flip and keeps passing.
    """
    hits = lint.write_lint_hits()
    assert not hits, "trace writes outside the trace currency:\n" + "\n".join(hits)


def test_no_bulk_name_takes_a_trace_slice():
    """PROPERTY (V1, C2): no bulk_* / quantize_bulk* / dequantize_bulk* call
    takes a trace slice as an argument.

    BREAKS IF: a trace call site is renamed to the bulk family (e.g. the
    renderer's dequantize moved to dequantize_bulk_f32) -- identical at 0 bits,
    2^k off after the flip.
    """
    hits = lint.bulk_name_on_trace_hits()
    assert not hits, "bulk names on trace slices:\n" + "\n".join(hits)


def test_every_sum_over_a_trace_slice_accumulates_wide():
    """PROPERTY (V1, A2): a sum over a trace slice runs in an explicit int64
    (or exact float64) accumulator, never the array's own int32.

    BREAKS IF: a bare ``.sum()`` over a trace plane returns -- numpy < 2 on
    Windows (the Lenovo's 1.26.4) reduces int32 in 32 bits, and a plane past
    128 units total wraps at 8 extra bits.
    """
    hits = lint.narrow_sum_hits()
    assert not hits, "narrow sums over trace slices:\n" + "\n".join(hits)


def test_the_filter_validator_bounds_against_q16_one():
    """PROPERTY (V1, A12): a duct filter efficiency is a Q16 FRACTION of
    removed mass, quantized by quantize_q16_scalar and bounded by Q16_ONE --
    never a trace density.

    BREAKS IF: the filter validator moves to the trace family (eff_q up to
    2^24 would pass a TRACE_ONE bound and scrub 2^k times the mass).
    """
    src = (ROOT / "src" / "simulation" / "vent_system.py").read_text(encoding="utf-8")
    m = re.search(r"\n    eff_q = \[\]\n(?P<body>.*?)\n    return eff_q", src, re.S)
    assert m, "the filter validator block moved -- re-point this lint"
    body = m.group("body")
    assert "quantize_q16_scalar" in body and "Q16_ONE" in body
    assert "TRACE_ONE" not in body and "quantize_trace" not in body


def _cpp_function_body(text, name):
    m = re.search(r"FP_HD inline int32_t " + name + r"\s*\([^)]*\)\s*\{", text)
    assert m, f"{name} not found"
    depth, j = 1, m.end()
    while depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[m.end():j]


def test_the_sweep_density_finishes_shift_through_the_kit():
    """PROPERTY (V1, C13): the two radiation_sweep.h finishes -- through which
    every CPU, CUDA and binding trace density sum is finished -- contain no
    FP_SHIFT and call trace_density_shr.

    BREAKS IF: a finish shifts a trace density sum by the Q16 shift (or any
    inline shift) instead of the trace currency's own.
    """
    text = (ROOT / "cpp" / "src" / "radiation_sweep.h").read_text(encoding="utf-8")
    for fn in ("gas_extinction_finish", "gas_extinction_exp_finish"):
        body = _cpp_function_body(text, fn)
        assert "FP_SHIFT" not in body, fn
        assert "trace_density_shr" in body, fn


# ---------------------------------------------------------------------------
# V10 -- the door
# ---------------------------------------------------------------------------
def _rows(**over):
    """A legal from-scratch gas table config; `over` maps gas name -> column
    overrides."""
    rows = {}
    for i in range(len(GAS_NAMES)):
        name = GAS_NAMES[i]
        bulk = i >= N_TRACE_GASES
        rows[name] = {
            "absorption": [0.0] * 3 if bulk else [0.1, 0.2, 0.3],
            "scatter_albedo": [0.0] * 3 if bulk else [0.4, 0.5, 0.6],
            "diffusion": 0.0 if bulk else 0.1, "decay": 0.0 if bulk else 0.01,
            "glow": 0.0, "flammable": False, "emits_when_hot": False,
            "effect": "", "conservative": bulk, "heat_absorb": 0.0,
        }
    for name, cols in over.items():
        rows[name].update(cols)
    return rows


@pytest.mark.parametrize("gas", ["o2", "inert_n2"])
@pytest.mark.parametrize("column, value", [
    ("heat_absorb", 0.5),
    ("absorption", [0.0, 0.2, 0.0]),
    ("scatter_albedo", [0.0, 0.0, 0.1]),
    ("light_absorb", [0.0, 0.3, 0.0]),
])
def test_the_gases_door_refuses_an_optical_coefficient_on_a_bulk_row(gas, column, value):
    """PROPERTY (V10, §2.2): a config giving a conservative (bulk) row a
    non-zero heat_absorb, absorption, scatter_albedo or light_absorb is
    refused at the gases door, with a message naming #88 -- so every optical
    density sum (sweep heat and light terms, glow, laser) only ever sees trace
    planes and finishes in the trace currency.

    BREAKS IF: the door lets a bulk plane carry an optical coefficient, which
    would enter a trace-currency sum 2^k times too strong after the flip.
    """
    with pytest.raises(ValueError, match="#88"):
        GasTable(_rows(**{gas: {column: value}}))


def test_the_gases_door_keeps_the_bulk_rows_exactly_the_bulk_ids():
    """PROPERTY (V10): the conservative rows are exactly the ids >=
    N_TRACE_GASES (the layout every per-plane currency choice keys on); a
    trace row marked conservative, or a bulk row not, is refused naming #88;
    the shipped config and a legal from-scratch table load.

    BREAKS IF: the trace/bulk id layout can drift away from the conservative
    flag, so a reader keyed on ``g < N_TRACE_GASES`` picks the wrong currency.
    """
    GasTable(_rows())
    GasTable.from_config()
    with pytest.raises(ValueError, match="#88"):
        GasTable(_rows(smoke={"conservative": True}))
    with pytest.raises(ValueError, match="#88"):
        GasTable(_rows(o2={"conservative": False}))


# ---------------------------------------------------------------------------
# V4 -- writer scale
# ---------------------------------------------------------------------------
def test_the_soot_mint_enters_smoke_at_trace_from_bulk_of_the_bulk_soot():
    """PROPERTY (V4, G5): CombustionSolver.step's soot mint takes ``soot``
    BULK counts out of the burn (N2 receives burn - soot) and writes exactly
    trace_from_bulk(soot) into the smoke plane -- per cell a multiple of
    2^TRACE_FINE_BITS, and in total
    ``Σ ΔN2 + Σ ΔSMOKE / 2^k == −Σ ΔO2`` exactly (the Dalton split in bulk
    counts), with soot actually minted.

    BREAKS IF: the mint writes bulk counts into the trace plane (or shifts
    the bulk side), or the Dalton split stops being exact in bulk counts.
    """
    h = w = 7
    gas = np.zeros((7, h, w), dtype=np.int32)
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    flammable = np.zeros((h, w), dtype=bool)
    wall_hp = np.zeros((h, w), dtype=np.int32)
    fire = np.zeros((h, w), dtype=np.int32)
    ign = np.zeros((h, w), dtype=np.int32)
    temperature = np.zeros((h, w), dtype=np.int32)
    from simulation.materials import MAT_WOOD, MaterialTable
    ign_wood = int(MaterialTable.from_config().ignition_temp_q16[MAT_WOOD])
    cy = cx = 3
    solid[cy, cx] = flammable[cy, cx] = True
    wall_hp[cy, cx] = fire_fixed.quantize_scalar(60.0)
    ign[cy, cx] = ign_wood
    temperature[cy, cx] = ign_wood * 2
    fire[cy, cx] = fire_fixed.quantize_scalar(0.6)
    gas[O2][~solid] = gas_fixed.quantize_bulk_scalar(0.21)
    gas[INERT_N2][~solid] = gas_fixed.quantize_bulk_scalar(0.79)
    comb = bp.CombustionSolver()
    comb.burn_rate, comb.o2_thresh_burn, comb.H_FUEL_M, comb.soot_yield = 2.0, 0.02, 4.0, 0.3
    minted = 0
    for _ in range(10):
        before = gas.astype(np.int64).copy()
        comb.step(gas, O2, INERT_N2, SMOKE, temperature, wall_hp, fire,
                  flammable, solid, is_vacuum, ign, 1.0 / 24.0, 1.0, 0.05)
        d = gas.astype(np.int64) - before
        dsmoke = d[SMOKE]
        assert np.all(dsmoke % (1 << gas_fixed.TRACE_FINE_BITS) == 0)
        soot_bulk = int(dsmoke.sum()) >> gas_fixed.TRACE_FINE_BITS
        assert int(d[INERT_N2].sum()) + soot_bulk == -int(d[O2].sum())
        assert int(gas_fixed.trace_from_bulk(soot_bulk)) == int(dsmoke.sum())
        minted += soot_bulk
    assert minted > 0, "no soot was minted -- the gate is vacuous"


def test_a_fieldedit_add_moves_a_trace_tile_by_the_quantized_amount():
    """PROPERTY (V4, A11): a FieldEdit ADD of amount ``a`` onto a trace tile
    holding ``v >= 0`` changes it by quantize_trace_scalar(a) to within one
    count (float64 ties near .5 may differ by one); a MAX edit lands at
    max(v, quantize_trace_scalar(a)) within one count; the deposit is in the
    trace currency whatever the tile held.

    BREAKS IF: the FieldEdit trace combine quantizes in another currency, or
    a narrow drops the deposit.
    """
    from level_loader import LevelData
    from simulation.field_edit import EditMode, EditQueue, FieldEdit, Region
    from simulation.gamemap import GameMap
    tm = np.zeros((10, 10), dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 1
    g = GameMap(LevelData(name="v4", version="2", path=Path("."), tilemap=tm,
                          tile_size_m=1.0, diffuse_path=Path(".")))
    rng = np.random.default_rng(88)
    for v_real, a in ((0.0, 0.75), (1.7, 0.3), (0.013, 1e-4), (5.0, 2.5), (0.2, 0.06)):
        v_trace = gas_fixed.quantize_trace_scalar(v_real)
        g.gas[SMOKE][5, 5] = v_trace
        q = EditQueue()
        q.enqueue(FieldEdit(field="gas", channel=SMOKE, region=Region.TILE,
                            coords=(5, 5), amount=a, mode=EditMode.ADD))
        q.flush(g, rng)
        moved = int(g.gas[SMOKE][5, 5]) - v_trace
        assert abs(moved - gas_fixed.quantize_trace_scalar(a)) <= 1, (v_real, a, moved)
        g.gas[SMOKE][5, 5] = v_trace
        q.enqueue(FieldEdit(field="gas", channel=SMOKE, region=Region.TILE,
                            coords=(5, 5), amount=a, mode=EditMode.MAX))
        q.flush(g, rng)
        want = max(v_trace, gas_fixed.quantize_trace_scalar(a))
        assert abs(int(g.gas[SMOKE][5, 5]) - want) <= 1, (v_real, a)
