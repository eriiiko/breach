"""P-E0 hot-rail repro + pinned N~0.15 pocket variant (energy-books arc).

The committed E2E reproduction of the in-game HOT-RAIL blowup anatomy
(storm audit 2026-08-14 SS4.4): a small sealed room with an oversized fire
load; the burning block exhausts its O2, saturates smoke, evacuates (the
plume wind blows bulk N out, min N -> 0.000), and step-4c compression work
then multiplies T geometrically (the x~1.5/tick = 1+T_WORK_CLAMP rate-rail
signature) up to the T_MAX_PHYS ceiling, with multi-atm |dP| spikes when
gas slams back into the ceiling-hot pocket.

HEAD-ANATOMY NUMBERS BELOW ARE PRE-T_ABS LINEAGE (P-E0 as-built,
docs/e1_p_e0_asbuilt_2026-08-17.md, measured against the relative-T
compression law that predates docs/tabs_compression_work_design_2026-08-20.md).
Kept for provenance, not as current gate targets: eos.t_max_phys_hits = 2130
over the 2000-tick run, first hit tick 1761; a x1.4972/tick geometric climb
sustained 19 consecutive ticks (the audit's x1.4957 signature); peak T at
the 15984.5 ceiling; |dP| spike 97.5 atm (tick 1904); and the P-E0
eth_transport_delta bracket shows the SL T-copy transport pass MINTING
+3.72e16 raw book-energy over the run, beating the SS7 truncation
allowance on 901 ticks (worst tick +3.80e15 vs an allowance ~5e7). The
CURRENT (T_abs compression-work law) measured anatomy — one 13-tick
variable-rate climb, 4 ceiling hits, sharp collapse, oscillating band,
equilibrium ~5341 — is documented in each gate test's own docstring below
and in docs/tabs_compression_work_design_2026-08-20.md SS0b (R-1/R-2).

Gate idiom (design energy_transport_design_2026-08-16.md v2.1 SS6): the two
healthy-property tests below assert what a CLOSED energy book will satisfy,
so they are RED on HEAD by construction and carried as
xfail-with-owning-patch until their owning rung lands:
  - test_no_transport_mint  -> owned by P-E1 (energy-conservative transport);
                               RE-DERIVED to the exact closure identity by
                               P-W1b (design SS0b R-1) once e_ts_residual
                               (a counted signed channel) went live.
  - test_no_rail_hits       -> owned by P-E4 (compression-work trust gate;
                               flips strict at P-E4); RE-DERIVED from
                               `== 0` to a bounded-transient gate by P-W1b
                               (design SS0b R-2) once T_abs compression work
                               gave ambient air a genuine hot-rail entry
                               point.
The determinism tests pin the scenarios themselves (P-E0 oracle row:
"all scenarios deterministic + committed").
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                      # noqa: E402
from config import CFG                           # noqa: E402
from level_loader import LevelData               # noqa: E402
from simulation import Simulation                # noqa: E402
from simulation import fire_fixed                # noqa: E402

import storm_probe as sp                         # noqa: E402
from fire_timing_harness import (                # noqa: E402
    FP_ONE, AIR, HULL, FURN, apply_overrides, restore_overrides,
)

# The synced planes the determinism digests cover (bench_two_room idiom).
DIGEST_FIELDS = ("atmosphere", "temperature", "wind_x", "wind_y",
                 "gas", "fire", "smoke")

# --- the committed scenarios (parameters are PINNED; see the as-built) ----
# Hot rail: 8x8 interior at the SHIPPED tile scale (0.333 m) almost filled
# by a 6x6 furniture block, every fuel tile ignited at once (the oversized
# fire load). Rail first trips ~tick 1761 on HEAD -> 2000 ticks.
HOT = dict(interior=8, tile=0.333, fuel_block=6)
HOT_TICKS = 2000
# Pinned pocket variant (design SS2.4: the mid-band trust-gate residual):
# 10x10 interior at tile 0.5 with a 4x4 block — the burning block's hot
# cells sit at n_bulk ~0.145-0.156 (the n_work_ref half-band) for the whole
# plateau on HEAD, with NO rail engagement (measured; see the as-built).
POCKET = dict(interior=10, tile=0.5, fuel_block=4)


def build_fuel_room(interior, tile, fuel_block):
    """Sealed room (1-tile HULL ring, space boundary) with a centered
    fuel_block x fuel_block FURN block."""
    h = w = interior + 2
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = HULL
    tm[-1, :] = HULL
    tm[:, 0] = HULL
    tm[:, -1] = HULL
    f0 = (h - fuel_block) // 2
    tm[f0:f0 + fuel_block, f0:f0 + fuel_block] = FURN
    return LevelData(name="e0_fuel_room", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=float(tile),
                     diffuse_path=Path("."), boundary="space"), f0


def _plane_bytes(gmap, name):
    return np.ascontiguousarray(getattr(gmap, name)).tobytes()


def run_scenario(interior, tile, fuel_block, ticks, collect=False):
    """Run the scenario under the P-F1b dials; return telemetry + digests."""
    restore = apply_overrides(dict(sp.PF1B))
    try:
        level, f0 = build_fuel_room(interior, tile, fuel_block)
        sim = Simulation(level, seed=12345, breach_physics=bp,
                         enable_recorder=False)
        gmap = sim.gmap
        seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
        slf = (slice(f0, f0 + fuel_block), slice(f0, f0 + fuel_block))
        gmap.fire[slf] = fire_fixed.quantize_scalar(seed_i)
        gmap.temperature[slf] = fire_fixed.quantize_scalar(280.0)

        eos = sim.physics_runner.eos
        gas = gmap.gas
        o2i = int(gmap.gases.name_to_id["o2"])
        n2i = int(gmap.gases.name_to_id["inert_n2"])
        om = ~gmap.solid

        traj = {name: hashlib.sha256() for name in DIGEST_FIELDS}
        eth_ticks = []      # (eth_transport_delta, allowance) per tick
        # P-W1b (design SS0b R-1): the exact per-tick closure identity's
        # four counted terms, mirroring cuda_bulk_flux_check.py:454-473's
        # PART-3 identity check (trunc = eth - (-e_ts_residual - e_wipe_sum
        # + e_floor_sum); trunc must be in (-n_bulk_active_sum, 0]).
        identity_ticks = []  # (eth, e_ts_residual, e_wipe_sum, e_floor_sum, n_bulk_active_sum)
        tick_peak_T = []      # P-W1b R-2: per-tick max cell T (not running max)
        # P-E1 counter accumulators (design SS2.5): the run totals the
        # as-built reports and the active-flux fraction SS7's bound scales by.
        e1 = dict(n_active_flux=0, n_bulk_active_sum=0, n_cell_substeps=0,
                  e_ts_residual=0, e_wipe_sum=0, e_floor_sum=0)
        peak_T = 0.0
        o2_start = int(gas[o2i][om].astype(np.int64).sum())

        # ==============================================================
        # arc #54 (gas-energy conservation) §2.8 — THE CLOSURE IDENTITY,
        # ACROSS WHOLE TICKS (design §6's REWRITTEN row, P-G1b).
        #
        # P-G1a could only bracket the EOS step: it was transitional, so
        # combustion, the thermal solver and the seam writers still wrote
        # `temperature` outside the EOS and were absorbed by the solver's
        # entry re-sync. P-G1b lands every one of those writers on the gas
        # energy seam and DELETES the re-sync — D1 is live — so the honest
        # bracket is the WHOLE `Simulation.step`, and the identity has to
        # account for all four groups of counters that may move the field:
        #
        #   EOS        design §2.8's seven terms  (RESET every step)
        #   tail       the thermal solver's gas side (ACCUMULATING)
        #   combustion the two-hop energy ledger  (ACCUMULATING)
        #   seam       every Python writer's net  (ACCUMULATING)
        #
        # Anything that changes `Σ_accountable gas_energy` without landing
        # in one of them is exactly the class of silent mint this arc
        # exists to make impossible, and this gate is where it shows up.
        # ==============================================================
        closure = {"ticks": 0, "bad": 0, "worst": 0}
        tsolver = sim.physics_runner.engine.temperature
        comb = sim.physics_runner.combustion

        def _e_acct():
            acct = gmap._gas_energy_accountable()
            return int(gmap.gas_energy[acct].astype(object).sum())

        def _terms():
            return (
                int(eos.e_entry_resync_sum) + int(eos.e_transport_net_sum)
                - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
                + int(eos.e_drag_heat_sum) - int(eos.e_work_export_sum)
                + int(eos.e_rail_sum),
                int(tsolver.e_gas_deposit_sum) + int(tsolver.e_gas_cond_sum)
                + int(tsolver.e_gas_rail_sum),
                -int(comb.e_comb_draw_sum) + int(comb.e_comb_deliver_sum)
                + int(comb.e_comb_heat_sum) + int(comb.e_comb_rail_sum),
                int(gmap.gas_energy_seam_net()),
                # the water-displacement evacuation's export (design §2.7 row
                # 2, R3-#10): `step_water_tail` moves `gas_energy` with the
                # bulk shares a flooding cell pushes out, conservatively
                # inside the accountable set, so the only term is what LEFT
                # it. Reset per call, like the EOS group. Zero on this dry
                # scenario -- named so a future wet one cannot open a hole.
                -int(sim.physics_runner.engine.e_water_evac_export_sum),
            )

        # The combustion PARCEL identity (combustion.h identity (B)): nothing
        # drawn from a donor is lost on the way to the flame. Checked once at
        # the end of the run — it is a cumulative statement.
        parcel = {"resid": 0}
        prev_e = _e_acct()
        prev_terms = _terms()

        for _ in range(ticks):
            sim.set_paused(False)
            sim.step()
            if collect:
                e_now = _e_acct()
                terms = _terms()
                expected = (terms[0]                       # EOS: reset/step
                            + (terms[1] - prev_terms[1])   # tail
                            + (terms[2] - prev_terms[2])   # combustion
                            + (terms[3] - prev_terms[3])   # seams
                            + terms[4])                    # water tail
                resid = (e_now - prev_e) - expected
                closure["ticks"] += 1
                if resid:
                    closure["bad"] += 1
                    closure["worst"] = max(closure["worst"], abs(resid))
                prev_e, prev_terms = e_now, terms
            if collect:
                # SS7 truncation allowance, deliberately GENEROUS and
                # law-independent: n_sub x (total bulk N raw over the whole
                # map) — one raw-T LSB per bulk count per substep. The
                # active-flux scaling (P-E1's n_active_flux counter) can only
                # SHRINK it, so a mint that beats this bound beats SS7 too.
                nb_tot = int((gas[o2i].astype(np.int64)
                              + gas[n2i].astype(np.int64)).sum())
                allowance = int(eos.dbg_last_n_sub) * nb_tot
                eth_ticks.append((int(eos.eth_transport_delta), allowance))
                identity_ticks.append((
                    int(eos.eth_transport_delta),
                    int(eos.e_ts_residual),
                    int(eos.e_wipe_sum),
                    int(eos.e_floor_sum),
                    int(eos.n_bulk_active_sum),
                ))
                # P-E1 (design SS2.5/SS7): the ACTIVE-FLUX fraction the SS7
                # bound is really scaled by, plus the counted one-way terms.
                n_cells = int(om.sum())
                e1["n_active_flux"] += int(eos.n_active_flux)
                e1["n_bulk_active_sum"] += int(eos.n_bulk_active_sum)
                e1["n_cell_substeps"] += int(eos.dbg_last_n_sub) * n_cells
                e1["e_ts_residual"] += int(eos.e_ts_residual)
                e1["e_wipe_sum"] += int(eos.e_wipe_sum)
                e1["e_floor_sum"] += int(eos.e_floor_sum)
                tick_max_T = float(
                    gmap.temperature[om].astype(np.int64).max()) / FP_ONE
                tick_peak_T.append(tick_max_T)
                peak_T = max(peak_T, tick_max_T)
            for name in DIGEST_FIELDS:
                traj[name].update(_plane_bytes(gmap, name))
        o2_end = int(gas[o2i][om].astype(np.int64).sum())
        # combustion.h identity (B) — the PARCEL identity, cumulative.
        parcel["resid"] = (
            int(comb.e_comb_draw_sum) + int(comb.e_comb_mint_sum)
            - int(comb.e_comb_deliver_sum) - int(comb.e_soot_shed_sum)
            - int(comb.e_ts_products_sum) - int(comb.e_comb_export_sum))
        parcel["drawn"] = int(comb.e_comb_draw_sum)
        parcel["shed"] = int(comb.e_soot_shed_sum)
        return dict(
            parcel=parcel,
            digests={n: h.hexdigest() for n, h in traj.items()},
            t_max_phys_hits=int(eos.t_max_phys_hits),
            work_clamp_hits=int(eos.work_clamp_hits),
            eth_ticks=eth_ticks, identity_ticks=identity_ticks,
            closure=closure,                       # arc #54 §2.8
            tick_peak_T=tick_peak_T, peak_T=peak_T, e1=e1,
            o2_burned_frac=1.0 - o2_end / max(o2_start, 1))
    finally:
        restore_overrides(restore)


# The full hot-rail run is expensive (~2000 ticks) — run it ONCE per session
# and let both healthy-property tests read the same telemetry.
@pytest.fixture(scope="module")
def hot_run():
    return run_scenario(ticks=HOT_TICKS, collect=True, **HOT)


def test_hot_scenario_reaches_the_audit_anatomy(hot_run):
    """Non-vacuousness (audit rule R1): the committed scenario really is an
    oversized-load starved fire, not a fizzle — it burns most of the room's
    O2 and reaches flame-grade temperatures. (Stays green post-arc: closing
    the energy books must not put the fire out.)"""
    # Measured on HEAD: 31.5% of the ROOM total is burned by tick 2000 (the
    # burning block's own cells hit O2 = 0.000 long before — starvation is
    # local). 15% is the fizzle line, generous to post-arc retuning.
    assert hot_run["o2_burned_frac"] > 0.15, (
        f"fire only burned {hot_run['o2_burned_frac']:.1%} of the room O2 — "
        "the oversized-load premise is gone")
    assert hot_run["peak_T"] > 1000.0, (
        f"peak gas T {hot_run['peak_T']:.0f} never reached flame grade")


def test_no_transport_mint(hot_run):
    """HEALTHY property, RE-DERIVED AGAIN by the gas-energy conservation arc
    (#54 P-G1b, design §2.8) onto the ABSOLUTE closure identity, now holding
    ACROSS WHOLE TICKS:

        d(Sum_accountable gas_energy) over one Simulation.step ==
              [ e_entry_resync_sum + e_transport_net_sum - e_wipe_sum
                - e_kick_ke_sum + e_drag_heat_sum - e_work_export_sum
                + e_rail_sum ]                                    (EOS)
            + [ e_gas_deposit_sum + e_gas_cond_sum + e_gas_rail_sum ]
                                                       (thermal solver, gas)
            + [ -e_comb_draw_sum + e_comb_deliver_sum + e_comb_heat_sum
                + e_comb_rail_sum ]                        (combustion)
            + gas_energy_seam_net()                        (Python seams)

    exact in int64, EVERY TICK. P-G1a could only pin the EOS step, because
    the writers outside it still wrote `temperature` and were swept up by
    the solver's entry re-sync. With D1 live that re-sync is gone and this
    is the real statement: `gas_energy` is the cross-tick truth and NOTHING
    may change it except a named, counted channel.

    That makes this the strongest form this gate has ever had — the earlier
    ones bounded the TRANSPORT pass alone, and only in the "books never
    open" direction. This pins the whole engine: transport, the five KE
    brackets, the face-flux step, the recovery rails, conduction, the heat
    deposits, combustion's two hops, and every structural / pump / FieldEdit
    seam, in both directions at once. The face-flux term contributes exactly
    0 to it by per-face cancellation: that IS the arc.

    (`e_entry_resync_sum` is RETIRED and structurally 0; it stays named here
    so this transcription still lists all seven EOS terms.)

    --- the superseded P-W1b statement, kept for the history it carries ---
    HEALTHY property, RE-DERIVED to the exact closure identity by P-W1b
    (design SS0b R-1): the old allowance (`n_sub x whole-map bulk N`)
    predates any counted creation/destruction channel being live, and under
    the T_abs compression-work law it is provably too loose — the entire
    excess on the 290 ticks that used to beat it is `e_ts_residual`, rule
    (d)'s air->thermal_solid debit charging a genuinely sub-ambient donor
    (min T -> -248 game-deg), a COUNTED signed channel that could not exist
    before the T_abs law made sub-ambient donors reachable again
    (corr(excess, -e_ts_residual) = 0.999999992 on the hot run;
    e_floor_sum = e_wipe_sum = 0 the whole run).

    The gate is now the exact per-tick closure identity (mirroring
    cuda_bulk_flux_check.py:454-473's PART-3 check):
        trunc = eth_transport_delta - (-e_ts_residual - e_wipe_sum + e_floor_sum)
        assert trunc <= 0                        # books never open
    This is STRICTER than the old bound in the honest dimension (zero
    uncounted energy, EVER) while being correct in the new two-signed-
    channel regime, where the old `total <= 0` assert on the raw
    eth_transport_delta SUM is false-by-design (e_ts_residual can now be
    net-positive across the run without any book ever opening — see
    test_transport_delta_is_one_way_negative for the truncation-bound
    half of the same identity)."""
    c = hot_run["closure"]
    assert c["ticks"] > 0, "vacuous: the closure bracket never ran"
    assert c["bad"] == 0, (
        f"the arc #54 §2.8 closure identity FAILED on {c['bad']} of "
        f"{c['ticks']} TICKS (worst |residual| {c['worst']} raw Q32) — "
        "energy appeared in or vanished from `gas_energy` beyond the four "
        "counted groups (EOS / thermal solver / combustion / seams)")


def test_combustion_parcel_identity(hot_run):
    """arc #54 P-G1b (combustion.h identity (B)): nothing a burn DRAWS from
    its donors is lost on the two hops to the flame.

        e_comb_draw_sum + e_comb_mint_sum ==
            e_comb_deliver_sum + e_soot_shed_sum
          + e_ts_products_sum + e_comb_export_sum

    The books identity above already pins what combustion did to the FIELD;
    this pins where the parcel WENT, which is the half that catches the
    R3-#9 failure mode specifically — delivering the soot's share as well as
    the bulk's would satisfy neither side of this equation.

    Non-vacuous by construction: the hot run burns >15% of the room's O2
    (test_hot_scenario_reaches_the_audit_anatomy), so `e_comb_draw_sum` is
    large and `e_soot_shed_sum` is a real fraction of it."""
    p = hot_run["parcel"]
    assert p["drawn"] > 0, (
        "vacuous: the hot run drew no combustion energy at all")
    assert p["shed"] > 0, (
        "vacuous: soot_yield > 0 but no parcel energy was ever shed — the "
        "R3-#9 compounding guard is not being exercised")
    assert p["resid"] == 0, (
        f"the combustion PARCEL identity failed by {p['resid']} raw Q32 — "
        "energy drawn from the donors did not all arrive somewhere named")


def test_transport_delta_is_one_way_negative(hot_run):
    """SUPERSEDED by the R-1 closure identity (design SS0b): with the two
    signed counted channels (e_ts_residual, the vac/ring wipes) now live,
    the SUM of eth_transport_delta over the run is no longer the invariant
    — a legitimately two-signed counted series can net either way without
    any book opening. What remains invariant is the OTHER half of the same
    per-tick identity: truncation loss is bounded below by the SS7
    active-flux allowance, per tick:
        trunc = eth_transport_delta - (-e_ts_residual - e_wipe_sum + e_floor_sum)
        assert trunc > -n_bulk_active_sum   (mirrors cuda_bulk_flux_check.py
                                              :454-473's inclusive bound;
                                              the tick-1 edge where
                                              n_bulk_active_sum == 0 with
                                              trunc == 0 is legal under this
                                              inclusive form)
    Both named invariants — books never open (test_no_transport_mint) and
    truncation bounded (here) — hold on all 2000/2000 ticks of the hot
    run."""
    n_bad_bound = 0
    worst_slack = 0
    for eth, e_ts_residual, e_wipe_sum, e_floor_sum, n_bulk in hot_run["identity_ticks"]:
        counted = -e_ts_residual - e_wipe_sum + e_floor_sum
        trunc = eth - counted
        worst_slack = min(worst_slack, trunc)
        if trunc < -n_bulk:
            n_bad_bound += 1
    assert n_bad_bound == 0, (
        f"truncation loss beat the SS7 active-flux bound on {n_bad_bound} "
        f"ticks (worst slack {worst_slack} raw Q16.16^2)")


def test_no_rail_hits(hot_run):
    """HEALTHY property (design SS7 rails row), RE-DERIVED from `== 0` to a
    bounded-transient gate by P-W1b (design SS0b R-2): T_abs compression
    work gives ambient air a genuine hot-rail entry point for the first
    time (T=0 stops being a fixed point of step 4c), so the old strict
    `t_max_phys_hits == 0` — honest under the relative law, where the
    trust gate's fade starved the climb of any seed to compound from — is
    no longer the right invariant. What the rail is FOR (bounding the
    compounding's VALUE, counted, never silent) still needs a gate that
    can fail.

    Measured on the current hot run: ONE episode — a 13-tick variable-rate
    climb (ratios 1.01-1.29/tick, NOT the old law's sustained x1.4972/tick
    signature), 4 counted ceiling hits (ticks 1680-83), a sharp collapse
    (x0.44) via evacuation, then a 3,600-10,000 oscillating band; only 7
    ticks all-run with any cell above 15000; late-run equilibrium (mean of
    the last 1000 ticks' peak) 5,341 game-deg, close to the OLD law's
    all-run peak of 5,553 (docstring-stale pre-T_abs number, kept above for
    provenance). The old runaway (2130 hits, 19+ sustained ceiling ticks)
    fails both bounds below by orders of magnitude, so this gate still
    catches it.

    New gate (both 2x measured headroom): t_max_phys_hits <= 8 AND ticks
    with any cell's T > 15000 <= 14."""
    n_hits = hot_run["t_max_phys_hits"]
    n_ceiling_ticks = sum(1 for t in hot_run["tick_peak_T"] if t > 15000.0)
    assert n_hits <= 8, (
        f"T_MAX_PHYS engaged {n_hits} times — beyond the 2x-headroom "
        "bounded-transient budget (measured 4 on the current hot run)")
    assert n_ceiling_ticks <= 14, (
        f"{n_ceiling_ticks} ticks had a cell above 15000 game-deg — beyond "
        "the 2x-headroom bounded-transient budget (measured 7 on the "
        "current hot run)")


def test_hot_scenario_prefix_is_deterministic():
    """Scenario pin: two short prefix runs are digest-identical, all planes."""
    a = run_scenario(ticks=240, **HOT)
    b = run_scenario(ticks=240, **HOT)
    assert a["digests"] == b["digests"]


def test_pocket_variant_is_deterministic():
    """The pinned N~0.15 pocket variant (design SS2.4 measurement scenario):
    determinism assert only — the healthy-property gate values for this
    scenario are frozen later, per SS7 (P-E4 row). Measured HEAD anatomy is
    recorded in the P-E0 as-built."""
    a = run_scenario(ticks=240, **POCKET)
    b = run_scenario(ticks=240, **POCKET)
    assert a["digests"] == b["digests"]
