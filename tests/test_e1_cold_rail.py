"""P-E0 cold-rail window scenario pin (energy-books arc).

The storm audit's SS4.3 supercooling window — two-room bench, air damping
0.005, k_wind_strip restored to 0.5 (the 08-03 dial the P-K0 promotion
turned off) — is the COLD-rail counterpart of the hot-rail repro in
test_e1_hot_rail.py: wind-strip kills the fire, the over-damped expansion
pocket supercools to the T_MIN floor, and the ambient reservoir back-feeds
it through conduction while step-4c work compounds.

This file pins the SCENARIO only (P-E0 oracle row: "all scenarios
deterministic + committed"): a short prefix run, digest-identical twice.
The healthy-property gate values for the window are frozen later, per
design SS7 — and per SS2.3 the decision depends on the measured window
pocket N, which P-E0 recovered from the regenerated
`_fire_tuning_artifacts/ledger_window_e0.npz` (regenerable, deliberately
untracked; command in the as-built): during the T-floor episode the
pocket cell holds n_bulk ~1.7-9.3 — ABOVE the n_work_ref trust band
(0.125-0.25), so the P-E4 trust gate does NOT bound this loop and the
cold-rail residual returns to Erik as a measured accepted-gap decision
(docs/e1_p_e0_asbuilt_2026-08-17.md).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_two_room as bench  # noqa: E402

# The audit SS7 window dials: P-F1b + the strip restored; damp 0.005 is
# stamped on the mirrored planes by run_bench (never written to config).
WINDOW_DIALS = dict(bench.sp.PF1B, **{"k_wind_strip": "0.5"})
WINDOW_DAMP = 0.005


def test_window_scenario_prefix_is_deterministic():
    n_ticks = 300  # 12.5 s: ignition + the first damped door transient
    a = bench.run_bench(ticks=n_ticks, damp=WINDOW_DAMP,
                        dials=dict(WINDOW_DIALS))
    b = bench.run_bench(ticks=n_ticks, damp=WINDOW_DAMP,
                        dials=dict(WINDOW_DIALS))
    # Non-vacuous: the fire lit and the atmosphere moved under damping.
    assert a["summary"]["ke_peak"] > 0.01
    assert max(r["I"] for r in a["res"]["rows"]) > 0.05, "crate never burned"
    # The scenario itself is pinned: bit-identical trajectories, both layers.
    assert a["digests"]["trajectory"] == b["digests"]["trajectory"]
    assert a["digests"]["final"] == b["digests"]["final"]
