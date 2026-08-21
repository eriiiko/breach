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
        for _ in range(ticks):
            sim.set_paused(False)
            sim.step()
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
        return dict(
            digests={n: h.hexdigest() for n, h in traj.items()},
            t_max_phys_hits=int(eos.t_max_phys_hits),
            work_clamp_hits=int(eos.work_clamp_hits),
            eth_ticks=eth_ticks, identity_ticks=identity_ticks,
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
    """HEALTHY property, RE-DERIVED to the exact closure identity by P-W1b
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
    n_bad_open = 0
    worst_open = 0
    for eth, e_ts_residual, e_wipe_sum, e_floor_sum, _n_bulk in hot_run["identity_ticks"]:
        counted = -e_ts_residual - e_wipe_sum + e_floor_sum
        trunc = eth - counted
        if trunc > 0:
            n_bad_open += 1
            worst_open = max(worst_open, trunc)
    assert n_bad_open == 0, (
        f"books OPEN on {n_bad_open} ticks (worst {worst_open} raw "
        "Q16.16^2 of book-energy appeared beyond the counted terms) — "
        "the T_abs closure identity does not hold")


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
