"""P-E3 — interior momentum drag with a heat counterparty.

Design: ``docs/energy_transport_design_2026-08-16.md`` v2.2 §2.8 (the
mechanism), REWRITTEN for the gas-energy conservation arc
(``docs/gas_energy_conservation_design_2026-08-29.md`` §2.3, D5, P-G1a).

The mechanism: per TICK, in the step-4 kick loop, after the |u| cap and
before the store, velocity is shrunk (stage L component-wise magnitude-first
``u *= 1 - kd_q``, then stage Q's implicit ``u /= 1 + k2*dt*|u|``) and the
removed kinetic energy is deposited as heat into the SAME cell.

WHAT CHANGED AT P-G1a. The deposit used to be
``ΔT = k_drag_heat_frac · (Δ|u|²/2) / c_v`` written into ``temperature`` and
railed at T_MAX_PHYS, priced by three counters (``e_drag_deposit``,
``e_drag_drop_sum``, ``e_drag_rail_clipped``) against the identity
``ke_removed == 2·c_v·(deposit + drop + rail)``. Every term of that is gone:

  * there is no heat FRACTION (design D5 — the dial was a hand-rolled stand-in
    for ``1/c_v_phys ≈ 0.0018``; the constant is now DERIVED);
  * there is no ``c_v`` divide (that convention dial belongs to the radiation
    deposit — §2.1);
  * there is no rail AT the deposit site (the once-per-tick §2.6 recovery owns
    T_MIN / T_MAX_PHYS now), so a deposit can no longer be silently dropped;
  * the deposit lands in ``gas_energy``, not in ``temperature``.

So the identity is restated in the field's own ABSOLUTE currency:

    e_drag_heat_sum == Σ_cells  N_i · trunc_{k_ke}( Δ|u|²_i )

with the SAME two-stage truncation the kick applies (§2.3's pinned order):

    t  = mul128_shr(k_ke_q32, du2_raw, 48)     // Q32·Q32>>48 = a Q16 ΔT
    dE = mul128_shr(N_raw,    t,       0)      // Q16·Q16     = Q32 energy

and ``k_ke_q32 = round(2^32 · γ(γ−1)·T_AMB_K / (2·c_max²))``. Because that is
the arithmetic the counter itself books, the identity is EXACT — not
"within an LSB slack" — which is the point of a derived constant.

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
    # arc #54 P-G1a: `t_work_clamp` left the signature with step 4c (D11).
    n_floor_solver=1e-3, t_min=-292.0,
    t_max_phys=16000.0, u_max=1000.0,
)
# The ambient K the derived k_ke is folded from — explicit test parameter fed
# to BOTH the Python oracle (_k_ke_q32 below) and the actual solver call, so
# it need not track EOSSolver::T_AMB_K's compiled-in struct default (still
# 290.0f, unchanged — C++ default values are comment-only under G12, issue
# #12, docs/fire_g12_one_map_patch_2026-08-31.md); kept at the config's live
# eos_t_amb_k (293.0) for realism. t_min likewise mirrors config's T_MIN
# (-292.0), though this kernel does not rail T_MIN at the deposit site (see
# the module docstring) so neither value is load-bearing for the identity
# under test.
T_AMB_K = 293.0


def _q(x):
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


def _k_ke_q32(c_max=300.0, gamma=1.4, t_amb_k=T_AMB_K):
    """``make_recip(1 / k_ke)`` — the SAME fold eos_solver.cpp performs.

    ``fixedpoint::make_recip(v) == round(2^32 / v)``, so the Q.32
    representation of k_ke itself is ``make_recip(1/k_ke)``. Re-derived here
    from the dials rather than hardcoded, so a c_max / γ / T_AMB_K change
    moves the oracle and the engine together.
    """
    k_ke = gamma * (gamma - 1.0) * t_amb_k / (2.0 * c_max * c_max)
    return int((2.0 ** 32) / (1.0 / k_ke) + 0.5)


def _dE_of(n_raw, du2_raw, k_ke_q32):
    """The pinned two-stage ΔKE→energy chain, in Python ints (exact)."""
    t = (k_ke_q32 * int(du2_raw)) >> 48          # arithmetic shift == floor
    return (int(n_raw) * t)                       # shift 0


def _make_case(h, w, wmag, tlo, thi, nsc, seed, thin_frac=0.1):
    rng = np.random.default_rng(seed)
    wx = _q((rng.random((h, w)) * 2 - 1) * wmag).astype(np.int32)
    wy = _q((rng.random((h, w)) * 2 - 1) * wmag).astype(np.int32)
    t = _q(rng.random((h, w)) * (thi - tlo) + tlo).astype(np.int32)
    p_new = _q(np.ones((h, w))).astype(np.int32)
    gas = np.zeros((3, h, w), dtype=np.int32)
    gas[0] = _q(rng.random((h, w)) * 0.30 * nsc)
    gas[1] = _q(rng.random((h, w)) * 0.80 * nsc)
    thin = rng.random((h, w)) < thin_frac   # thin-N coverage
    gas[0][thin] = 0
    gas[1][thin] = 0
    gas_conservative = np.array([True, True, False])
    solid = np.zeros((h, w), dtype=bool)
    is_vacuum = np.zeros((h, w), dtype=bool)
    absorb = np.zeros((h, w), dtype=np.float32)
    return dict(wind_x=wx, wind_y=wy, temperature=t, p_new=p_new, gas=gas,
                gas_conservative=gas_conservative, solid=solid,
                is_vacuum=is_vacuum, dyn_wave_absorb=absorb)


def _run(inp, k_drag, k_drag2=0.0, c_v=1.0, c_local=2300.0, dt=1.0 / 24.0,
         thermal_solid=None, gas_energy=None):
    # VELOCITY-CLAMP (P-V1, D2v2/D5): a UNIFORM plane at c_local² — the kick
    # trusts it VERBATIM (no re-min against U_MAX), so this genuinely keeps
    # the |u| clamp disengaged for this module's velocities. The drag
    # identity then runs on the UNCLAMPED velocity.
    h, w = inp["wind_x"].shape
    cap2 = np.full((h, w), int(_q(c_local)) ** 2, dtype=np.int64)
    res = bp.eos_kick_compression_ref(
        inp["wind_x"].copy(), inp["wind_y"].copy(), inp["temperature"].copy(),
        inp["p_new"], inp["gas"], inp["gas_conservative"],
        inp["solid"], inp["is_vacuum"], inp["dyn_wave_absorb"],
        dt, cap2, k_drag=k_drag, k_drag2=k_drag2,
        c_v=c_v, t_amb_k=T_AMB_K, gas_energy=gas_energy,
        thermal_solid=thermal_solid, **CONSTS)
    # arc #54 D10: the counters_out[9] LAYOUT is kept, with slots 2/3/4/7/8
    # retired-and-zero. Slot 5 is the raw KE oracle, slot 6 the one drag
    # energy counter (ex `e_drag_deposit`).
    names = ("dig_vel", "dig_comp", "u_clamp", "u_max", "work_clamp_retired",
             "energy_floor_retired", "t_max_phys_retired", "ke_drag_removed",
             "e_drag_heat_sum", "drop_retired", "rail_retired")
    return dict(zip(names, res))


# ---------------------------------------------------------------------------
# 1. Dormancy — k_drag=0.0 (the shipped default) must be a byte-for-byte no-op
# ---------------------------------------------------------------------------
def test_dormant_at_default_k_drag_is_byte_identical_and_counters_zero():
    inp = _make_case(24, 24, 900.0, -100.0, 15990.0, 1.0, seed=1)
    wx0, wy0, t0 = (inp["wind_x"].copy(), inp["wind_y"].copy(),
                    inp["temperature"].copy())

    r_off = _run(inp, k_drag=0.0)
    assert r_off["ke_drag_removed"] == 0, "ke_drag_removed nonzero at k_drag=0"
    assert r_off["e_drag_heat_sum"] == 0, "e_drag_heat_sum nonzero at k_drag=0"

    # A tiny-but-nonzero k_drag must ALSO be dormant: it quantizes to kd_q==0
    # (design's explicit warning — dormancy branches on the QUANTIZED fold).
    r_tiny = _run(inp, k_drag=1e-6)
    assert r_tiny["ke_drag_removed"] == 0, "k_drag=1e-6 should quantize to kd_q=0"
    assert r_tiny["e_drag_heat_sum"] == 0, "k_drag=1e-6 should quantize to kd_q=0"

    # arc #54 D10: the retired slots stay zero, so every positional unpack
    # downstream keeps working and nothing silently re-appears.
    for k in ("work_clamp_retired", "energy_floor_retired",
              "t_max_phys_retired", "drop_retired", "rail_retired"):
        assert r_off[k] == 0, f"retired counter slot {k} is not zero"

    # Fields must be untouched by the reference call itself (sanity: the
    # in/out arrays were copies, originals unmodified).
    assert np.array_equal(inp["wind_x"], wx0)
    assert np.array_equal(inp["wind_y"], wy0)
    assert np.array_equal(inp["temperature"], t0)


# ---------------------------------------------------------------------------
# 2. THE DRAG IDENTITY, in absolute currency — EXACT, not slack-bounded.
#
#    e_drag_heat_sum == Σ_cells N_i · trunc_{k_ke}(Δ|u|²_i)
#
#    We cannot see the per-cell Δ|u|² from outside, but we CAN see the raw
#    n-weighted oracle `ke_drag_removed == Σ_i mul128_shr(N_i, du2_i, 16)`.
#    So the identity is gated two ways:
#      (a) EXACTLY, per cell, by re-running the same call on a ONE-CELL-live
#          grid where the whole counter IS that cell's term;
#      (b) in AGGREGATE, by bounding e_drag_heat_sum against the raw oracle
#          through the derived constant (k_ke·2^16·ke_removed, to within the
#          per-cell truncation, which is structurally one-way).
# ---------------------------------------------------------------------------
CASES = [
    # (h, w, wmag, tlo, thi, nsc, k_drag2, c_v)
    (24, 24, 900.0, -50.0, 200.0, 1.0, 0.0, 1.0),
    (24, 24, 900.0, 15000.0, 15999.0, 1.0, 0.0, 1.0),
    (24, 24, 500.0, -289.0, 300.0, 0.5, 0.0, 1.0),
    (24, 24, 999.0, 15900.0, 15999.0, 2.0, 0.25, 1.0),   # stage Q live too
    (16, 40, 700.0, -100.0, 500.0, 1.0, 0.0, 0.5),       # c_v must be INERT
    (16, 40, 300.0, 0.0, 100.0, 0.02, 0.0, 1.0),         # thin gas everywhere
]
K_DRAGS = (0.02, 0.05, 1.0)


@pytest.mark.parametrize("case_idx", range(len(CASES)))
def test_drag_identity_holds_in_absolute_currency(case_idx):
    h, w, wmag, tlo, thi, nsc, kd2, cv = CASES[case_idx]
    kq = _k_ke_q32()
    saw_live = False
    for k_drag in K_DRAGS:
        for seed in range(3):
            inp = _make_case(h, w, wmag, tlo, thi, nsc,
                             seed=1000 * seed + case_idx)
            r = _run(inp, k_drag=k_drag, k_drag2=kd2, c_v=cv)
            ke = int(r["ke_drag_removed"])
            heat = int(r["e_drag_heat_sum"])
            if ke == 0:
                assert heat == 0, "heat booked with no KE removed"
                continue
            saw_live = True
            # The deposit is STRICTLY POSITIVE and strictly one-way: the
            # magnitude-first shrink makes Δ|u|² >= 0 structurally, and both
            # mul128_shr stages floor, so the booked heat can only ever be
            # <= the real value — never a mint.
            assert heat > 0, "KE was removed but no heat was deposited"
            # ke_drag_removed == Σ mul128_shr(N, du2, 16) == Σ (N·du2)>>16, so
            # (k_ke · 2^16) · ke_removed reconstructs the deposit in Q32.
            ideal = (kq * ke) >> 32          # == k_ke * 2^16 * ke, floored
            # The two chains floor at DIFFERENT places, so they differ by a
            # bounded, ONE-WAY amount:
            #   heat = Σ N_i·floor(k_ke·du2_i·2^16)   loses <= 1 ΔT count per
            #                                          cell, i.e. <= N_i each;
            #   ideal reconstructs from an already-floored ke, so it loses
            #                                          <= k_ke·2^16 ≈ 59 each.
            n_sum = int((inp["gas"][0].astype(np.int64)
                         + inp["gas"][1].astype(np.int64)).sum())
            assert heat <= ideal + 128 * h * w, (
                f"case={case_idx} k_drag={k_drag} seed={seed}: booked heat "
                f"{heat} exceeds the reconstruction {ideal} by more than the "
                f"per-cell floor of `ke_drag_removed` can explain — minting")
            assert heat >= ideal - n_sum - 2 * h * w, (
                f"case={case_idx} k_drag={k_drag} seed={seed}: booked heat "
                f"{heat} is {ideal - heat} below the reconstruction {ideal} — "
                f"more than the per-cell ΔT floor (<= Σ N = {n_sum}) explains")
    assert saw_live, f"case {case_idx} never engaged the drag — vacuous"


def test_drag_deposit_is_exact_on_a_single_live_cell():
    """The identity with NO aggregation: one cell carries all the mass, so
    ``e_drag_heat_sum`` IS that cell's term and can be reproduced digit for
    digit from the pinned §2.3 chain."""
    h, w = 5, 5
    kq = _k_ke_q32()
    wx = np.zeros((h, w), dtype=np.int32)
    wy = np.zeros((h, w), dtype=np.int32)
    wx[2, 2] = _q(180.0)
    wy[2, 2] = _q(-95.0)
    t = np.zeros((h, w), dtype=np.int32)
    p_new = np.full((h, w), _q(1.0), dtype=np.int32)
    gas = np.zeros((3, h, w), dtype=np.int32)
    gas[0, 2, 2] = _q(0.21)
    gas[1, 2, 2] = _q(0.79)
    cons = np.array([True, True, False])
    solid = np.zeros((h, w), dtype=bool)
    vac = np.zeros((h, w), dtype=bool)
    absorb = np.zeros((h, w), dtype=np.float32)
    cap2 = np.full((h, w), int(_q(2300.0)) ** 2, dtype=np.int64)

    wxc, wyc, tc = wx.copy(), wy.copy(), t.copy()
    res = bp.eos_kick_compression_ref(
        wxc, wyc, tc, p_new, gas, cons, solid, vac, absorb,
        1.0 / 24.0, cap2, k_drag=0.5, t_amb_k=T_AMB_K, **CONSTS)
    heat = int(res[8])

    # Reproduce Δ|u|² from the mutated velocity: only stage L ran (k_drag2=0),
    # and the cell's pre-drag (ux, uy) is its post-kick value. With a uniform
    # p_new the kick is a no-op and absorb is off, so pre-drag == the seeded
    # velocity exactly.
    du2 = (int(wx[2, 2]) ** 2 + int(wy[2, 2]) ** 2
           - int(wxc[2, 2]) ** 2 - int(wyc[2, 2]) ** 2)
    assert du2 > 0, "stage L did not shrink the velocity"
    n_raw = int(gas[0, 2, 2]) + int(gas[1, 2, 2])
    assert heat == _dE_of(n_raw, du2, kq), (
        "the booked drag heat is not the pinned §2.3 chain "
        f"(got {heat}, expected {_dE_of(n_raw, du2, kq)})")


def test_c_v_is_inert_in_the_drag_deposit():
    """D5: the deposit constant is DERIVED, so the convention dial `c_v` --
    which used to divide it -- must not move it at all any more."""
    inp = _make_case(20, 20, 800.0, -50.0, 400.0, 1.0, seed=42)
    a = _run(inp, k_drag=0.5, c_v=1.0)
    b = _run(inp, k_drag=0.5, c_v=0.01)
    assert a["e_drag_heat_sum"] == b["e_drag_heat_sum"] != 0, (
        "c_v still moves the drag deposit — the derived-constant replacement "
        "is not complete (design D5)")


# ---------------------------------------------------------------------------
# 3. The deposit lands in gas_energy, cell-local, and ts cells are skipped.
# ---------------------------------------------------------------------------
def test_deposit_lands_in_gas_energy_not_temperature():
    h, w = 6, 6
    inp = _make_case(h, w, 900.0, 100.0, 200.0, 1.0, seed=11, thin_frac=0.0)
    cap2 = np.full((h, w), int(_q(2300.0)) ** 2, dtype=np.int64)
    ge = np.zeros((h, w), dtype=np.int64)
    wx, wy, t = (inp["wind_x"].copy(), inp["wind_y"].copy(),
                 inp["temperature"].copy())
    t_in = t.copy()
    res = bp.eos_kick_compression_ref(
        wx, wy, t, inp["p_new"], inp["gas"], inp["gas_conservative"],
        inp["solid"], inp["is_vacuum"], inp["dyn_wave_absorb"],
        1.0 / 24.0, cap2, k_drag=1.0, t_amb_k=T_AMB_K,
        gas_energy=ge, **CONSTS)
    # arc #54 P-G1a: step 4c is gone from this reference, so `temperature`
    # is NEVER written by it -- the mirror is refreshed by step()'s §2.6
    # recovery instead.
    assert np.array_equal(t, t_in), (
        "the kick reference wrote `temperature` — step 4c should be gone and "
        "the drag deposit should land in gas_energy")
    assert (ge > 0).any(), "no drag heat reached gas_energy"
    # The whole field delta IS the counter: cell-local, nothing lost.
    assert int(ge.astype(object).sum()) == int(res[8]), (
        "Σ gas_energy delta != e_drag_heat_sum — the deposit is not "
        "cell-local or the counter does not book what the field got")


def test_ts_cells_skip_drag_entirely():
    h, w = 8, 8
    inp = _make_case(h, w, 900.0, 100.0, 200.0, 1.0, seed=7, thin_frac=0.0)
    ts = np.zeros((h, w), dtype=bool)
    ts[3, 3] = True
    ts[3, 4] = True

    cap2 = np.full((h, w), int(_q(2300.0)) ** 2, dtype=np.int64)
    wx_off, wy_off, t_off = (inp["wind_x"].copy(), inp["wind_y"].copy(),
                             inp["temperature"].copy())
    ge_off = np.zeros((h, w), dtype=np.int64)
    bp.eos_kick_compression_ref(
        wx_off, wy_off, t_off,
        inp["p_new"], inp["gas"], inp["gas_conservative"],
        inp["solid"], inp["is_vacuum"], inp["dyn_wave_absorb"],
        1.0 / 24.0, cap2, k_drag=0.0, t_amb_k=T_AMB_K,
        gas_energy=ge_off, thermal_solid=ts, **CONSTS)

    wx_on, wy_on, t_on = (inp["wind_x"].copy(), inp["wind_y"].copy(),
                          inp["temperature"].copy())
    ge_on = np.zeros((h, w), dtype=np.int64)
    bp.eos_kick_compression_ref(
        wx_on, wy_on, t_on,
        inp["p_new"], inp["gas"], inp["gas_conservative"],
        inp["solid"], inp["is_vacuum"], inp["dyn_wave_absorb"],
        1.0 / 24.0, cap2, k_drag=1.0, t_amb_k=T_AMB_K,
        gas_energy=ge_on, thermal_solid=ts, **CONSTS)

    assert wx_on[3, 3] == wx_off[3, 3] and wy_on[3, 3] == wy_off[3, 3], (
        "a ts cell's velocity moved when k_drag > 0 — the ts skip is not "
        "engaging for the velocity shrink")
    # arc #54 F5: a ts cell carries N and u but NO gas_energy — every bracket
    # it opens is exported to the counter, never stored.
    assert ge_on[3, 3] == 0 and ge_on[3, 4] == 0, (
        "a thermal_solid cell's gas_energy was written (F5: ts cells export "
        "their brackets to e_ts_ke_sum, they never store)")
    # Sanity: an ordinary (non-ts) cell DID change with k_drag on — proves
    # the mechanism is live and the ts-cell identity above is meaningful.
    assert (wx_on[1, 1] != wx_off[1, 1]) or (wy_on[1, 1] != wy_off[1, 1]), (
        "sanity: k_drag=1.0 had no effect anywhere — the mechanism is not live")
    assert ge_on[1, 1] != 0, "sanity: an ordinary cell got no deposit"


# ---------------------------------------------------------------------------
# 4. Thin-N cells: the deposit is n-WEIGHTED, so a cell with no bulk mass
#    receives nothing — but its velocity is still dragged. (The old
#    "phantom-T guard" test, restated: there is no T write to guard any more,
#    and the guard's job is done structurally by the N weight.)
# ---------------------------------------------------------------------------
def test_thin_cell_gets_no_deposit_but_is_still_dragged():
    h, w = 6, 6
    wx = np.full((h, w), _q(900.0), dtype=np.int32)
    wy = np.full((h, w), _q(-400.0), dtype=np.int32)
    t = np.full((h, w), _q(150.0), dtype=np.int32)
    p_new = np.full((h, w), _q(1.0), dtype=np.int32)
    gas = np.zeros((3, h, w), dtype=np.int32)
    gas[0, :, :] = _q(0.21)
    gas[1, :, :] = _q(0.79)
    gas[0, 2, 2] = 0
    gas[1, 2, 2] = 0   # n_bulk == 0 — the thin cell
    cons = np.array([True, True, False])
    solid = np.zeros((h, w), dtype=bool)
    vac = np.zeros((h, w), dtype=bool)
    absorb = np.zeros((h, w), dtype=np.float32)
    cap2 = np.full((h, w), int(_q(2300.0)) ** 2, dtype=np.int64)
    ge = np.zeros((h, w), dtype=np.int64)

    bp.eos_kick_compression_ref(
        wx, wy, t, p_new, gas, cons, solid, vac, absorb,
        1.0 / 24.0, cap2, k_drag=1.0, t_amb_k=T_AMB_K,
        gas_energy=ge, **CONSTS)

    assert ge[1, 1] != 0, "sanity: a normal cell got no deposit at all"
    assert ge[2, 2] == 0, (
        "a zero-N cell received a deposit — the deposit must be n-weighted")
    # But velocity WAS still dragged there — same shrink as every other cell
    # (kd_q = quantize(k_drag*dt) < FP_ONE, so this is a partial per-tick
    # shrink, not a full zeroing).
    assert wx[2, 2] == wx[1, 1] and wy[2, 2] == wy[1, 1], (
        "the thin cell's velocity shrink differs from an ordinary cell's — "
        "the N weight must not also gate the velocity drag")
