"""P-E3 — interior momentum drag with a heat counterparty (energy-books arc).

Design: ``docs/energy_transport_design_2026-08-16.md`` v2.2 §2.8. As-built:
``docs/e1_p_e3_asbuilt_2026-08-17.md``.

The mechanism: per TICK, in the step-4 kick loop, after the |u| cap and
before the store, velocity is shrunk component-wise magnitude-first
(``u *= 1 - kd_q``) and the removed kinetic energy is deposited as heat into
the SAME cell's T. Four new oracle counters (``ke_drag_removed``,
``e_drag_deposit``, ``e_drag_drop_sum``, ``e_drag_rail_clipped``) must obey
the identity ``ke_removed == 2*c_v*(e_deposit + e_drop + e_rail_clipped)``
within a small per-cell LSB slack.

This module drives ``eos_kick_compression_ref`` (the CPU P6.4 reference —
the SAME loop the live ``EOSSolver::step`` runs) directly, on synthetic
fields, rather than the full engine — the fast, precise way to gate a
per-cell arithmetic identity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402

FP_ONE = 65536

CONSTS = dict(
    c_max=300.0, dx=1.0 / 3.0, adiabatic_index=1.4, absorb_strength=8.0,
    n_floor_solver=1e-3, t_min=-289.0, t_work_clamp=0.5,
    t_max_phys=16000.0, u_max=1000.0,
)


def _q(x):
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _make_case(h, w, wmag, tlo, thi, nsc, seed, thin_frac=0.1):
    rng = np.random.default_rng(seed)
    wx = _q((rng.random((h, w)) * 2 - 1) * wmag).astype(np.int32)
    wy = _q((rng.random((h, w)) * 2 - 1) * wmag).astype(np.int32)
    t = _q(rng.random((h, w)) * (thi - tlo) + tlo).astype(np.int32)
    p_new = _q(np.ones((h, w))).astype(np.int32)
    gas = np.zeros((3, h, w), dtype=np.int32)
    gas[0] = _q(rng.random((h, w)) * 0.30 * nsc)
    gas[1] = _q(rng.random((h, w)) * 0.80 * nsc)
    thin = rng.random((h, w)) < thin_frac   # phantom-T guard coverage
    gas[0][thin] = 0
    gas[1][thin] = 0
    gas_conservative = np.array([True, True, False])
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    absorb = np.zeros((h, w), dtype=np.float32)
    return dict(wind_x=wx, wind_y=wy, temperature=t, p_new=p_new, gas=gas,
                gas_conservative=gas_conservative, solid=solid,
                is_vacuum=is_vacuum, dyn_wave_absorb=absorb)


def _run(inp, k_drag, k_drag_heat_frac, c_v, c_local=2300.0, dt=1.0 / 24.0,
         thermal_solid=None):
    res = bp.eos_kick_compression_ref(
        inp["wind_x"].copy(), inp["wind_y"].copy(), inp["temperature"].copy(),
        inp["p_new"], inp["gas"], inp["gas_conservative"],
        inp["solid"], inp["is_vacuum"], inp["dyn_wave_absorb"],
        dt, int(_q(c_local)), k_drag=k_drag, k_drag_heat_frac=k_drag_heat_frac,
        c_v=c_v, thermal_solid=thermal_solid, **CONSTS)
    names = ("dig_vel", "dig_comp", "u_clamp", "u_max", "work_clamp",
             "energy_floor", "t_max_phys", "ke_drag_removed", "e_drag_deposit",
             "e_drag_drop_sum", "e_drag_rail_clipped")
    return dict(zip(names, res))


# ---------------------------------------------------------------------------
# 1. Dormancy — k_drag=0.0 (the shipped default) must be a byte-for-byte no-op
# ---------------------------------------------------------------------------
def test_dormant_at_default_k_drag_is_byte_identical_and_counters_zero():
    inp = _make_case(24, 24, 900.0, -100.0, 15990.0, 1.0, seed=1)
    wx0, wy0, t0 = inp["wind_x"].copy(), inp["wind_y"].copy(), inp["temperature"].copy()

    r_off = _run(inp, k_drag=0.0, k_drag_heat_frac=1.0, c_v=1.0)
    for k in ("ke_drag_removed", "e_drag_deposit", "e_drag_drop_sum",
              "e_drag_rail_clipped"):
        assert r_off[k] == 0, f"{k} nonzero at k_drag=0.0"

    # A tiny-but-nonzero k_drag must ALSO be dormant: it quantizes to kd_q==0
    # (design's explicit warning — dormancy branches on the QUANTIZED fold).
    r_tiny = _run(inp, k_drag=1e-6, k_drag_heat_frac=1.0, c_v=1.0)
    for k in ("ke_drag_removed", "e_drag_deposit", "e_drag_drop_sum",
              "e_drag_rail_clipped"):
        assert r_tiny[k] == 0, f"{k} nonzero at k_drag=1e-6 (should quantize to kd_q=0)"

    # Fields must be untouched by the reference call itself (sanity: the
    # in/out arrays were copies, originals unmodified).
    assert np.array_equal(inp["wind_x"], wx0)
    assert np.array_equal(inp["wind_y"], wy0)
    assert np.array_equal(inp["temperature"], t0)


# ---------------------------------------------------------------------------
# 2. THE DRAG IDENTITY — the headline oracle, measured with slack reported.
# ---------------------------------------------------------------------------
CASES = [
    # (h, w, wmag, tlo, thi, nsc, heat_frac, c_v)
    (24, 24, 900.0, -50.0, 200.0, 1.0, 1.0, 1.0),
    (24, 24, 900.0, 15000.0, 15999.0, 1.0, 1.0, 1.0),   # near T_MAX_PHYS -> clip
    (24, 24, 500.0, -289.0, 300.0, 0.5, 0.5, 1.0),       # heat_frac < 1 -> drop
    (24, 24, 999.0, 15900.0, 15999.0, 2.0, 0.3, 1.0),    # both drop + clip
    (16, 40, 700.0, -100.0, 500.0, 1.0, 1.0, 0.5),       # c_v != 1
    (16, 40, 300.0, 0.0, 100.0, 0.02, 1.0, 1.0),         # thin gas everywhere
]
K_DRAGS = (0.02, 0.05, 1.0)


@pytest.mark.parametrize("case_idx", range(len(CASES)))
def test_drag_identity_holds_within_lsb_slack(case_idx):
    h, w, wmag, tlo, thi, nsc, hf, cv = CASES[case_idx]
    worst_frac = 0.0
    for k_drag in K_DRAGS:
        for seed in range(3):
            inp = _make_case(h, w, wmag, tlo, thi, nsc,
                             seed=1000 * seed + case_idx)
            r = _run(inp, k_drag=k_drag, k_drag_heat_frac=hf, c_v=cv)
            lhs = r["ke_drag_removed"]
            rhs = 2 * cv * (r["e_drag_deposit"] + r["e_drag_drop_sum"]
                            + r["e_drag_rail_clipped"])
            diff = lhs - round(rhs)
            # One-way: the identity is proven exact in reals; the ONLY
            # integer slack is the accumulated floor-division truncation,
            # which is structurally >= 0 (never a mint the other way).
            assert diff >= -1, (
                f"identity violated the wrong direction: case={case_idx} "
                f"k_drag={k_drag} seed={seed} lhs={lhs} rhs={rhs} diff={diff}")
            if lhs != 0:
                worst_frac = max(worst_frac, abs(diff) / abs(lhs))
    # Measured bound (see the P-E3 as-built for the run across the full
    # battery): relative slack stays many orders of magnitude below 1e-3.
    assert worst_frac < 1e-3, (
        f"case {case_idx}: relative identity slack {worst_frac:.3e} exceeds "
        "the expected LSB-scale bound")


# ---------------------------------------------------------------------------
# 3. ts cells skip BOTH the drag and the deposit (ruling A1).
# ---------------------------------------------------------------------------
def test_ts_cells_skip_drag_entirely():
    h, w = 8, 8
    inp = _make_case(h, w, 900.0, 100.0, 200.0, 1.0, seed=7, thin_frac=0.0)
    ts = np.zeros((h, w), dtype=bool)
    ts[3, 3] = True
    ts[3, 4] = True
    t0 = inp["temperature"][3, 3]

    # eos_kick_compression_ref mutates its wind_x/wind_y/temperature args IN
    # PLACE and returns (digest_velocity, digest_compression, *counters) —
    # inspect the mutated local arrays, not the return value. Compare a
    # k_drag=0 (no drag anywhere) run against a k_drag>0 run WITH the ts
    # mask: at a ts cell the two must be IDENTICAL (drag never touches it —
    # only the ordinary kick/absorb/cap chain does, which is k_drag-blind);
    # at an ordinary cell they must DIFFER (proving k_drag is actually live
    # elsewhere, so the ts-identity above isn't a vacuous "nothing moved").
    wx_off, wy_off, t_off = (inp["wind_x"].copy(), inp["wind_y"].copy(),
                             inp["temperature"].copy())
    bp.eos_kick_compression_ref(
        wx_off, wy_off, t_off,
        inp["p_new"], inp["gas"], inp["gas_conservative"],
        inp["solid"], inp["is_vacuum"], inp["dyn_wave_absorb"],
        1.0 / 24.0, int(_q(2300.0)),
        k_drag=0.0, k_drag_heat_frac=1.0, c_v=1.0,
        thermal_solid=ts, **CONSTS)

    wx_on, wy_on, t_on = (inp["wind_x"].copy(), inp["wind_y"].copy(),
                          inp["temperature"].copy())
    bp.eos_kick_compression_ref(
        wx_on, wy_on, t_on,
        inp["p_new"], inp["gas"], inp["gas_conservative"],
        inp["solid"], inp["is_vacuum"], inp["dyn_wave_absorb"],
        1.0 / 24.0, int(_q(2300.0)),
        k_drag=1.0, k_drag_heat_frac=1.0, c_v=1.0,
        thermal_solid=ts, **CONSTS)

    assert wx_on[3, 3] == wx_off[3, 3] and wy_on[3, 3] == wy_off[3, 3], (
        "a ts cell's velocity moved when k_drag > 0 — the ts skip is not "
        "engaging for the velocity shrink")
    assert t_on[3, 3] == t0 == t_off[3, 3], (
        "a ts cell's temperature was written by the drag deposit")
    # Sanity: an ordinary (non-ts) cell DID change with k_drag on — proves
    # the mechanism is live and the ts-cell identity above is meaningful.
    assert (wx_on[1, 1] != wx_off[1, 1]) or (wy_on[1, 1] != wy_off[1, 1]), (
        "sanity: k_drag=1.0 had no effect anywhere — the mechanism is not live")


# ---------------------------------------------------------------------------
# 4. Phantom-T guard: a near-vacuum cell's T is not written, but the oracle
#    still prices it (n-weighted to ~0 — the identity above already proves
#    this doesn't break the books; this test isolates the T-write guard).
# ---------------------------------------------------------------------------
def test_phantom_cell_temperature_not_written():
    # SPATIALLY UNIFORM wind + pressure so step 4c's compression work
    # (div(u_new) == 0 everywhere for a uniform field) contributes NOTHING
    # to any cell's T — isolating the drag deposit as the T write's only
    # possible source at all (a prerequisite for cleanly testing the guard).
    h, w = 6, 6
    wx = np.full((h, w), _q(900.0), dtype=np.int32)
    wy = np.full((h, w), _q(-400.0), dtype=np.int32)
    t = np.full((h, w), _q(150.0), dtype=np.int32)
    p_new = np.full((h, w), _q(1.0), dtype=np.int32)
    gas = np.zeros((3, h, w), dtype=np.int32)
    gas[0, :, :] = _q(0.21)
    gas[1, :, :] = _q(0.79)
    gas[0, 2, 2] = 0
    gas[1, 2, 2] = 0   # n_bulk == 0 < N_EPS (1 raw count) — the phantom cell
    gas_conservative = np.array([True, True, False])
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    absorb = np.zeros((h, w), dtype=np.float32)
    t0 = t[2, 2]

    bp.eos_kick_compression_ref(
        wx, wy, t, p_new, gas, gas_conservative, solid, is_vacuum, absorb,
        1.0 / 24.0, int(_q(2300.0)),
        k_drag=1.0, k_drag_heat_frac=1.0, c_v=1.0, **CONSTS)

    # A neighbouring, non-phantom cell DID get a deposit (proves the drag
    # mechanism itself is live and this isn't a vacuous "nothing happened").
    assert t[1, 1] != _q(150.0), "sanity: a normal cell's T was not touched by the drag deposit at all"
    assert t[2, 2] == t0, "phantom-T guard failed: a near-vacuum cell's T was written"
    # But velocity WAS still dragged there — same shrink as every other cell
    # (the guard is deposit-only; kd_q = quantize(k_drag*dt) < FP_ONE, so
    # this is a partial per-tick shrink, not a full zeroing).
    assert wx[2, 2] == wx[1, 1] and wy[2, 2] == wy[1, 1], (
        "the phantom cell's velocity shrink differs from an ordinary "
        "cell's — the T-write guard must not also gate the velocity drag")
