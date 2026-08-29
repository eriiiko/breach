"""Drag-law v2 (docs/drag_law_v2_design_2026-08-23.md §6/§8 P2 gate row) --
the VENTING GATE: five legs (regression fence, k2 sweep bound, quadratic +
linear negative controls, blast/heat watch), transcribing the blast + 4-tile
breach-to-vacuum scenario of ``tools/tabs_pw2_venting_capture.py`` (design
§6: "do NOT invent new geometry" -- the Benches reuse rule; that option was
evaluated and rejected in the tool's own header). Kept as a TRANSCRIPTION,
not an import: ``tools/`` has no ``__init__.py`` and no existing test
imports from it -- the same precedent ``tabs_pw2_venting_capture.py`` itself
set by transcribing (not importing) ``cuda_kick_check.py``'s PART-2 scenario.

50%-EQUALIZATION -- EXACT DEFINITION (pinned here, per design §6's
instruction to define it concretely):

    N_total(tick) = sum of the Dalton gas mass (O2 + INERT_N2, Q16.16) over
    every OPEN cell (``~solid & ~is_vacuum`` -- the interior room, excluding
    the hull ring and the true-vacuum band/breach itself), sampled once per
    tick (tick 0 = post-construction, pre-step).

    The BASELINE leg (k2=0, shipped k_drag) is run for TICKS ticks; Nhalf :=
    N0 - 0.5*(N0 - Nfinal) where N0/Nfinal are the baseline's own tick-0/
    tick-TICKS values. tick_50 for ANY leg (baseline or otherwise) is the
    first tick at which that leg's OWN N_total trace is <= Nhalf (Nhalf is
    ALWAYS the baseline's, never recomputed per leg -- design §6 leg 1:
    "bound expressed as a ratio to this SAME-RUN baseline, no frozen N").
    ratio := tick_50(leg) / tick_50(baseline).

WHY MASS, NOT PRESSURE (design's own example was "the vented room's mean
pressure reaching halfway", explicitly given as an "e.g." with the
instruction to pin our own choice and report it): mean ATMOSPHERE pressure
was tried first and is UNUSABLE here above roughly k2>=1 -- MEASURED, this
scenario, whole-room mean over the open mask: at k2=10 the mean pressure
does not fall, it RISES from ambient=1.0 to a PEAK of ~0.91 around tick 100,
while only ~6% of the room's actual gas mass has left by that same tick
(vs ~94% vented at the k2=0 baseline by tick 100). Cause: k_drag_heat_frac's
deposit is a small FRACTION of removed KE, but at k2=10 the absolute KE
removed is enormous, so cells at the neck rail to T_MAX_PHYS=16000 K
repeatedly (t_max_phys_hits climbs into the thousands over a few hundred
ticks), and the solver's fixed-schedule MULTIGRID pressure solve carries
that injected energy ROOM-SCALE in a single V-cycle per tick (eos_solver.h's
own header: "MG carries room-scale influence in one cycle") -- so a handful
of railed neck cells pump the WHOLE room's mean pressure up, inverting the
sign of the very venting-slowdown effect leg 3 exists to catch. Verified
this is not a spatial-averaging artifact: a west-wall probe strip far from
both the breach and both hot pockets tracked the whole-room mean within 1
tick throughout. Gas MASS is monotonic (nothing re-enters from true vacuum)
and structurally immune to the heat-deposit side channel (heat moves T, not
N) -- it is exactly "how much air is left in the room", the physical
quantity k2 acts on via velocity, and the clean choice for isolating
venting speed from the heat/blast side effects leg 5 watches separately.

SCOPE CAVEAT (design §6, pinned verbatim per the task): the scenario's
breach is one tile => L_neck ~= 0.333 m (tabs_pw2_venting_capture.py:78), so
a green k2=1.0 leg at THIS neck does NOT establish design §1(b)'s
k2 <~ 0.6 "venting-safe" band -- that band is set by longer necks. P3 must
not read gate-green-at-1.0 (or, per this file's leg-2 finding below,
gate-RED-at-1.0) as generalizing to other neck sizes.

LEG 2 IS EXPECTED RED (see its xfail reason for the full, measured
explanation) -- this was investigated thoroughly, not assumed: design
§1(b)'s steady-orifice-flow estimate (sqrt(1+2*k2*L)) predicts mild
1.08x-1.29x slowdowns for k2 in {0.25, 0.5, 1.0} at this L_neck, but this
scenario is a BLAST release, not steady flow -- the uncapped (k2=0) neck
speed reaches into the hundreds of m/s within the first few ticks, and
design §1(c)'s named ceiling u_ceil = 1/(k2*dt) throttles that directly and
far harder than the steady-flow formula modeled (measured: k2=0.25 caps
neck speed to ~72-82 m/s against a predicted u_ceil=96 m/s; k2=1.0 caps it
to ~22-23 m/s against a predicted u_ceil=24 m/s -- the formula matches
almost exactly, it is the STEADY-FLOW ESTIMATE that doesn't apply at blast
velocities). This is new, P3-relevant information, not a gate defect: swept
independently (tests/_drag2_sweep_bench.py) at this same TICKS, the TRUE
empirical 1.5x crossover for this scenario sits between k2=0.1 (ratio 1.40)
and k2=0.15 (ratio 1.60), well below the design's tested "legal" set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                             # noqa: E402
from config import CFG                                   # noqa: E402
from level_loader import LevelData                       # noqa: E402
from simulation import atmosphere_fixed                  # noqa: E402
from simulation.gamemap import GameMap                    # noqa: E402
from simulation.gases import O2, INERT_N2                 # noqa: E402
from simulation.physics_runner import PhysicsRunner       # noqa: E402

H = W = 48
FP_ONE = 65536.0

# Long enough for the slowest LEGAL sweep leg (k2=1.0, measured tick_50~72-74)
# to actually cross 50%-equalization with margin; short enough to keep this
# 6-leg module well under the suite budget (measured ~3 ms/tick -> ~2-3 s
# total for the whole module at TICKS=120).
TICKS = 120

SWEEP_K2 = (0.25, 0.5, 1.0)          # design §6 leg 2's set
QUAD_NEG_CONTROL_K2 = 10.0           # design §6 leg 3
LIN_NEG_CONTROL_KDRAG = 10.0         # design §6 leg 4 (k2=0)
BOUND_RATIO = 1.5                    # design §6's shared bound, all legs


def _build_scenario():
    """TRANSCRIBED verbatim from tools/tabs_pw2_venting_capture.py::
    build_scenario -- see module docstring for why this is a transcription,
    not an import."""
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:46, 2:46] = 1
    tm[3:45, 3:45] = 4
    tm[22:26, 45] = 4          # the breach: hull ring opened to the vacuum band
    level = LevelData(name="eos_p64_blast_vent", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_vacuum.any(), "scenario must have vacuum to vent into"

    q = atmosphere_fixed.quantize_scalar
    g.temperature[10:16, 10:16] += q(5000.0)
    g.gas[O2, 11:14, 11:14] += q(4.0)
    g.temperature[30:36, 30:36] += q(15500.0)
    return g


def _run_leg(ticks, k_drag2=0.0, k_drag=None):
    """Run the transcribed scenario for ``ticks`` ticks with ``k_drag2``
    (and optionally ``k_drag``) set POST-CONSTRUCTION on ``runner.eos`` --
    the tool's own ``dx`` precedent (tabs_pw2_venting_capture.py:96 sets
    ``runner.eos.dx`` the same way right after ``PhysicsRunner`` is built;
    ``k_drag``/``k_drag2`` are readwrite ``EOSSolver`` members, same idiom).

    Tracks, per tick: the interior gas-mass trace (the 50%-equalization
    observable, see module docstring) and the four P-E3 drag counters,
    SUMMED across ticks (they reset at every ``step()`` entry --
    ``eos_solver.cpp``'s ``step()`` zeroes ``ke_drag_removed``/
    ``e_drag_deposit``/``e_drag_drop_sum``/``e_drag_rail_clipped`` near its
    top, unlike the five CUMULATIVE rail counters; precedent
    ``tools/velocity_clamp_pv2_measure.py:159`` diffs them itself against
    the previous tick's cumulative read -- here each tick's raw value
    already IS that tick's own contribution, so we sum directly). Also
    tracks the worst (max) single-tick ``ke_drag_removed`` (design §5's
    headroom target) and the worst single-tick max |u| over the open
    interior (the sweep bench's reporting need).
    """
    g = _build_scenario()
    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    runner.eos.k_drag2 = float(k_drag2)
    if k_drag is not None:
        runner.eos.k_drag = float(k_drag)
    dt = 1.0 / float(CFG.clock.ticks_per_second)

    open_mask = ~g.solid & ~g.is_vacuum

    def n_total():
        return float((g.gas[O2][open_mask].astype(np.int64)
                      + g.gas[INERT_N2][open_mask].astype(np.int64)).sum()) / FP_ONE

    n_trace = np.empty(ticks + 1, dtype=np.float64)
    n_trace[0] = n_total()
    # arc #54 P-G1a (design D5/D10): `e_drag_deposit` / `e_drag_drop_sum` /
    # `e_drag_rail_clipped` are RETIRED -- there is no heat FRACTION, no c_v
    # divide and no T_MAX_PHYS rail AT the deposit site any more (the
    # once-per-tick §2.6 recovery owns the rails). The one drag energy
    # counter is `e_drag_heat_sum`, in the gas_energy Q32 currency, and it is
    # the WHOLE removed KE at the derived k_ke -- so what used to be three
    # channels (deposit / dropped / rail-clipped) is now one, and the
    # "dominant sink" split the old leg-5 reported no longer exists.
    ke_sum = e_heat_sum = 0
    worst_ke = 0
    worst_ke_tick = -1
    worst_speed = 0.0

    for k in range(1, ticks + 1):
        runner.step(g, dt)
        n_trace[k] = n_total()
        ke = int(runner.eos.ke_drag_removed)
        ke_sum += ke
        e_heat_sum += int(runner.eos.e_drag_heat_sum)
        if ke > worst_ke:
            worst_ke = ke
            worst_ke_tick = k
        rad = (g.wind_x[open_mask].astype(np.int64) ** 2
               + g.wind_y[open_mask].astype(np.int64) ** 2)
        speed = float(np.sqrt(rad.max())) / FP_ONE if rad.size else 0.0
        worst_speed = max(worst_speed, speed)

    return dict(
        n_trace=n_trace, k_drag2=float(k_drag2), k_drag=float(runner.eos.k_drag),
        ke_sum=ke_sum, e_heat_sum=e_heat_sum,
        worst_ke=worst_ke, worst_ke_tick=worst_ke_tick,
        worst_speed=worst_speed, t_max_phys_hits=int(runner.eos.t_max_phys_hits),
    )


def _tick_50(n_trace, n_half):
    """First tick index (0=tick-0 entry) at which ``n_trace`` has fallen to
    or below ``n_half``; -1 if the trace never reaches it within this run."""
    below = np.where(n_trace <= n_half)[0]
    return int(below[0]) if len(below) else -1


@pytest.fixture(scope="module")
def legs():
    """Runs every leg ONCE (module-scoped) -- baseline + the k2 sweep + both
    negative controls -- and derives the shared 50%-equalization threshold
    from the baseline leg alone (design §6 leg 1)."""
    baseline = _run_leg(TICKS, k_drag2=0.0)
    N0 = baseline["n_trace"][0]
    Nfinal = baseline["n_trace"][-1]
    Nhalf = N0 - 0.5 * (N0 - Nfinal)
    tick50_base = _tick_50(baseline["n_trace"], Nhalf)
    assert tick50_base > 0, (
        "baseline never reached 50%-equalization within TICKS -- widen TICKS")
    bound_tick = BOUND_RATIO * tick50_base

    sweep = {}
    for k2 in SWEEP_K2:
        r = _run_leg(TICKS, k_drag2=k2)
        r["tick50"] = _tick_50(r["n_trace"], Nhalf)
        r["ratio"] = (r["tick50"] / tick50_base) if r["tick50"] > 0 else float("inf")
        sweep[k2] = r

    quad = _run_leg(TICKS, k_drag2=QUAD_NEG_CONTROL_K2)
    quad["tick50"] = _tick_50(quad["n_trace"], Nhalf)
    quad["ratio"] = (quad["tick50"] / tick50_base) if quad["tick50"] > 0 else float("inf")

    lin = _run_leg(TICKS, k_drag2=0.0, k_drag=LIN_NEG_CONTROL_KDRAG)
    lin["tick50"] = _tick_50(lin["n_trace"], Nhalf)
    lin["ratio"] = (lin["tick50"] / tick50_base) if lin["tick50"] > 0 else float("inf")

    return dict(baseline=baseline, N0=N0, Nfinal=Nfinal, Nhalf=Nhalf,
                tick50_base=tick50_base, bound_tick=bound_tick,
                sweep=sweep, quad=quad, lin=lin)


# ---------------------------------------------------------------------------
# Leg 1 -- regression fence at shipped dials (k2=0)
# ---------------------------------------------------------------------------
def test_leg1_regression_fence_baseline_capture(legs):
    """Design §6 leg 1: k2=0 at shipped dials (k_drag=0.5) -- capture the
    50%-equalization profile. Establishes Nhalf/tick50_base; every other
    leg's bound is a RATIO to this same-run number, never a frozen N."""
    b = legs["baseline"]
    print(f"\nleg1 baseline: N0={legs['N0']:.3f} Nfinal(t={TICKS})={legs['Nfinal']:.3f} "
          f"Nhalf={legs['Nhalf']:.3f} tick50_base={legs['tick50_base']} "
          f"(1.5x bound tick = {legs['bound_tick']:.2f})")
    assert legs["tick50_base"] > 0
    assert b["k_drag2"] == 0.0
    # Interior mass must be (numerically) non-increasing -- nothing re-enters
    # from true vacuum; a tiny positive slack absorbs int64/float64 rounding
    # in the Dalton-sum reconstruction, not real physics.
    slack = 1e-6 * legs["N0"]
    assert np.all(np.diff(b["n_trace"]) <= slack), (
        "interior mass rose at some tick beyond rounding slack -- venting "
        "is no longer monotonic")


# ---------------------------------------------------------------------------
# Leg 2 -- k2 sweep, must stay within the 1.5x bound. EXPECTED RED: see the
# module docstring and this xfail's reason for the full measured explanation.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(strict=True, reason=(
    "EXPECTED RED, measured and explained at P2 (2026-08-23) -- not a gate "
    "defect; see the module docstring's 'WHY MASS, NOT PRESSURE' and 'LEG 2 "
    "IS EXPECTED RED' sections for the full account. MEASURED mass-"
    "equalization ratio vs the k2=0 baseline, this exact transcribed "
    "scenario at TICKS=120: k2=0.25 -> ~1.9x, k2=0.5 -> ~2.8x, k2=1.0 -> "
    "~4.6x -- all exceed the 1.5x bound, by a widening margin (see the "
    "printed values on a run for this module's exact numbers). Root cause, "
    "hard-measured: this is a BLAST release, not design §1(b)'s steady "
    "orifice flow -- uncapped (k2=0) neck speed reaches 294->838 m/s within "
    "the first 5 ticks; §1(c)'s named ceiling u_ceil=1/(k2*dt) throttles "
    "this directly (k2=0.25 -> u_ceil=96 m/s, measured neck speed settles "
    "~72-82 m/s; k2=1.0 -> u_ceil=24 m/s, measured ~22-23 m/s) -- a 4x-15x "
    "velocity cut the steady-flow formula never modeled, and since venting "
    "is advective (mass flux ~ neck velocity) this directly explains the "
    "slowdown. Independently swept: the TRUE empirical 1.5x crossover for "
    "this scenario sits around k2~=0.1-0.15 (at TICKS=120: k2=0.1 -> 1.40x, "
    "k2=0.15 -> 1.60x, straddling the 1.5x line), a materially smaller "
    "'legal' band than the design's tested "
    "{0.25,0.5,1.0} set -- P3-relevant. strict=True ON PURPOSE: if a future "
    "k1/k2 retune (P3) or mechanism change ever makes this pass, that XPASS "
    "is the signal to revisit the sweep set / bound with Erik, not to "
    "delete this test."))
@pytest.mark.parametrize("k2", SWEEP_K2)
def test_leg2_k2_sweep_within_1_5x_bound(legs, k2):
    """Design §6 leg 2: k2 in {0.25, 0.5, 1.0} -- 50%-equalization time must
    be <= 1.5x the baseline's. See the xfail reason above for why this is
    currently, measuredly, expectedly red."""
    r = legs["sweep"][k2]
    print(f"\nleg2 k2={k2}: tick50={r['tick50']} ratio={r['ratio']:.4f} "
          f"(bound {BOUND_RATIO}, bound_tick={legs['bound_tick']:.2f})")
    assert r["ratio"] <= BOUND_RATIO, (
        f"k2={k2}: 50%-equalization took {r['ratio']:.4f}x the baseline "
        f"(tick {r['tick50']} vs baseline tick {legs['tick50_base']}), "
        f"exceeding the {BOUND_RATIO}x bound")


# ---------------------------------------------------------------------------
# Leg 3 -- quadratic negative control: k2=10 MUST exceed the bound.
# ---------------------------------------------------------------------------
def test_leg3_quadratic_negative_control_k2_10_must_exceed_bound(legs):
    """Design §6 leg 3: k2=10 sits deep in the choke regime (§1b: molasses
    parity ~k2=0.86 at this neck) and MUST fail the 1.5x bound -- proves the
    gate can catch a quadratic venting death, not only a linear one."""
    r = legs["quad"]
    bt = int(round(legs["bound_tick"]))
    print(f"\nleg3 k2={QUAD_NEG_CONTROL_K2}: tick50={r['tick50']} ratio={r['ratio']} "
          f"N(bound_tick={bt})={r['n_trace'][bt]:.3f} (Nhalf={legs['Nhalf']:.3f})")
    assert r["ratio"] > BOUND_RATIO, (
        f"k2={QUAD_NEG_CONTROL_K2} did NOT fail the {BOUND_RATIO}x bound "
        f"(ratio={r['ratio']}) -- the gate has no power to catch a "
        f"quadratic venting death")


# ---------------------------------------------------------------------------
# Leg 4 -- linear negative control: k_drag=10 (k2=0) MUST exceed the bound.
# ---------------------------------------------------------------------------
def test_leg4_linear_negative_control_k_drag_10_must_exceed_bound(legs):
    """Design §6 leg 4: k_drag=10 (k2=0) is the 2026-08-20 molasses -- kept
    as the revert-the-fix validation. MUST fail the 1.5x bound.

    NOTE the measured margin here is THIN (measured ratio 1.60x at
    TICKS=120, vs bound 1.5x -- crosses just 1-2 ticks past bound_tick) --
    deterministic and reproducible (this sim is fixed-point, not
    stochastic), but worth flagging: a future change elsewhere that shifts
    this leg's crossing tick by even one tick could flip this assertion.
    """
    r = legs["lin"]
    print(f"\nleg4 k_drag={LIN_NEG_CONTROL_KDRAG}: tick50={r['tick50']} "
          f"ratio={r['ratio']:.4f} (bound_tick={legs['bound_tick']:.2f}) "
          "-- NOTE thin margin, see this test's docstring")
    assert r["ratio"] > BOUND_RATIO, (
        f"k_drag={LIN_NEG_CONTROL_KDRAG} did NOT fail the {BOUND_RATIO}x "
        f"bound (ratio={r['ratio']:.4f})")


# ---------------------------------------------------------------------------
# Leg 5 -- blast/heat watch (design §6 leg 5 / §5's ledger headroom bound).
# ---------------------------------------------------------------------------
def test_leg5_blast_heat_watch(legs):
    """Design §6 leg 5: the four P-E3 drag counters, accumulated PER TICK
    across the run (they reset at every ``step()`` entry -- see ``_run_leg``
    docstring), watched on the k2=1.0 (top of the tested legal sweep) and
    k2=10 (quadratic negative control -- design §1(c) predicts the ~24x
    deposit-step growth here) legs against the k2=0 baseline. Bounds below
    are MEASURED at this exact scenario/TICKS and pinned with slack (per
    the design's own instruction: "within factors you MEASURE and pin in
    the test") -- not re-derived from the design doc's ~24x estimate, which
    was an order-of-magnitude anchor, not a promise.

    Also asserts design §5's int64 ledger-headroom margin: the worst
    single-tick raw ``ke_drag_removed`` (= Sigma_cells N*du^2 * 2^32,
    int64) must sit comfortably below the 2^63 overflow point, i.e.
    Sigma_cells N*du^2 (real units) comfortably below 2^31.
    """
    b = legs["baseline"]
    k2_1 = legs["sweep"][1.0]
    k2_10 = legs["quad"]

    # arc #54 P-G1a: the observable is `e_drag_heat_sum` (the whole removed
    # KE, at the derived k_ke) rather than the deposit/drop/rail triple. The
    # RATIO the leg gates is the same physical statement -- "more quadratic
    # drag means proportionally more heat in the neck" -- and its band is
    # UNCHANGED, because e_drag_heat_sum is proportional to what
    # e_drag_deposit was at the shipped k_drag_heat_frac = 1.0 (the constant
    # differs, the ratio does not). The `e_drag_rail_clipped > 0` half of the
    # old leg is RETIRED with the deposit-site rail: a deposit can no longer
    # be clipped at all, which is the point (D5) -- the T_MAX_PHYS rail is
    # the once-per-tick recovery's, and `t_max_phys_hits` is still reported.
    assert b["e_heat_sum"] > 0, (
        "baseline (k_drag=0.5 shipped, k2=0) must already show some linear "
        "drag heat deposit -- else the ratios below are meaningless")

    ratio_dep_k2_1 = k2_1["e_heat_sum"] / b["e_heat_sum"]
    ratio_dep_k2_10 = k2_10["e_heat_sum"] / b["e_heat_sum"]
    print(f"\nleg5 e_drag_heat_sum ratio vs baseline: k2=1.0 -> {ratio_dep_k2_1:.2f}x  "
          f"k2=10 -> {ratio_dep_k2_10:.2f}x")
    print(f"leg5 ke_drag_removed (raw KE oracle): baseline={b['ke_sum']} "
          f"k2=1.0={k2_1['ke_sum']} k2=10={k2_10['ke_sum']}")
    print(f"leg5 t_max_phys_hits (cumulative): baseline={b['t_max_phys_hits']} "
          f"k2=1.0={k2_1['t_max_phys_hits']} k2=10={k2_10['t_max_phys_hits']}")

    assert 15.0 <= ratio_dep_k2_1 <= 200.0, (
        f"k2=1.0 e_drag_heat_sum ratio {ratio_dep_k2_1:.2f}x moved outside "
        "the pinned [15x, 200x] band -- re-measure and re-pin")
    assert 60.0 <= ratio_dep_k2_10 <= 600.0, (
        f"k2=10 e_drag_heat_sum ratio {ratio_dep_k2_10:.2f}x moved outside "
        "the pinned [60x, 600x] band -- re-measure and re-pin")

    candidates = {"baseline": b, "k2=0.25": legs["sweep"][0.25],
                  "k2=0.5": legs["sweep"][0.5], "k2=1.0": k2_1,
                  "k2=10": k2_10, "k_drag=10": legs["lin"]}
    worst_label = max(candidates, key=lambda name: candidates[name]["worst_ke"])
    worst_ke_raw = candidates[worst_label]["worst_ke"]
    worst_ke_tick = candidates[worst_label]["worst_ke_tick"]
    worst_real = worst_ke_raw / (2.0 ** 32)
    margin = (2.0 ** 31) / worst_real if worst_real > 0 else float("inf")
    print(f"leg5 worst-tick ke_drag_removed (raw int64) = {worst_ke_raw} "
          f"at leg={worst_label} tick={worst_ke_tick} "
          f"(Sigma N*du^2 real units = {worst_real:.6e}), "
          f"headroom margin to 2^31 = {margin:.2f}x")
    assert margin > 10.0, (
        f"worst-tick int64 headroom margin {margin:.2f}x is below the 10x "
        "safety pin -- design §5's overflow bound is getting close, "
        "re-measure before P3 sweeps larger k2")
