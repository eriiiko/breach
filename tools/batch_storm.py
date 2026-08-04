"""batch_storm.py — run the storming battery and print summary tables.
ANALYSIS ONLY: drives the shipped engine through storm_probe; changes nothing.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import storm_probe as sp  # noqa: E402

OUT = HERE / "results"
OUT.mkdir(exist_ok=True)


def row(label, res, summ, keys):
    vals = []
    for k in keys:
        v = summ.get(k)
        vals.append("n/a" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)))
    print(f"  {label:34s} " + "  ".join(f"{v:>12s}" for v in vals))


def table(title, keys, runs):
    print(f"\n=== {title} ===")
    print(f"  {'case':34s} " + "  ".join(f"{k:>12s}" for k in keys))
    for label, res, summ in runs:
        row(label, res, summ, keys)


def do(label, **kw):
    res = sp.run(**kw)
    summ = sp.analyse(res)
    c = res["counters"]
    if any(v for k, v in c.items() if k != "dbg_last_n_sub"):
        print(f"  !! rails engaged in {label}: {c}")
    return (label, res, summ)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    KEYS_RING = ["ke_peak", "ke_efold_s", "ke_retention", "umax_peak", "dom_period_s"]
    KEYS_STEADY = ["ke_last_decile_mean", "umax_final", "dp_mean_tail", "p_std_note"]

    if which in ("all", "jet"):
        runs = [do(f"jet room12 damp={d}", mode="jet", geom="room", interior=12,
                   tile=0.5, seconds=30.0, damp=d, jet=5.0, patch=3)
                for d in (0.0, 0.005, 0.01, 0.02, 0.05)]
        table("T2a — PURE MOMENTUM ring-down (jet impulse, sealed 12x12, no fire)",
              KEYS_RING, runs)

    if which in ("all", "thermal"):
        runs = [do(f"thermal room12 damp={d}", mode="thermal", geom="room",
                   interior=12, tile=0.5, seconds=30.0, damp=d, dT=300.0, patch=2)
                for d in (0.0, 0.01, 0.05)]
        runs += [do("thermal room24 damp=0", mode="thermal", geom="room",
                    interior=24, tile=0.5, seconds=30.0, damp=0.0, dT=300.0, patch=2)]
        runs += [do("thermal tworoom12 damp=0", mode="thermal", geom="tworoom",
                    interior=12, tile=0.5, seconds=30.0, damp=0.0, dT=300.0, patch=2)]
        runs += [do("thermal tworoom12 damp=0.02", mode="thermal", geom="tworoom",
                    interior=12, tile=0.5, seconds=30.0, damp=0.02, dT=300.0, patch=2)]
        table("T2b/c — THERMAL impulse ring-down (+ the B2 two-room Helmholtz case)",
              KEYS_RING, runs)

    if which in ("all", "hotplate"):
        runs = [do(f"hotplate dT={dT}", mode="hotplate", geom="room", interior=12,
                   tile=0.5, seconds=40.0, damp=0.0, dT=dT)
                for dT in (50.0, 100.0, 200.0, 300.0, 400.0, 600.0)]
        table("T1 — STEADY DRIVE: one tile pinned at dT, no fire (sealed 12x12)",
              ["ke_last_decile_mean", "ke_peak", "umax_peak", "umax_final",
               "dp_mean_tail"], runs)

    if which in ("all", "hotdamp"):
        runs = [do(f"hotplate dT=300 damp={d}", mode="hotplate", geom="room",
                   interior=12, tile=0.5, seconds=40.0, damp=d, dT=300.0)
                for d in (0.0, 0.005, 0.01, 0.02, 0.05)]
        table("T5 — DAMPING SWEEP at full drive (hotplate dT=300)",
              ["ke_last_decile_mean", "umax_peak", "umax_final", "dp_mean_tail"],
              runs)

    print("\ndone.")


if __name__ == "__main__":
    main()
