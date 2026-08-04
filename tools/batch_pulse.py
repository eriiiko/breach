"""batch_pulse.py — the frequency-response battery.

Question: at which drive frequency does a flickering hot tile pump the most
sustained air motion, and does the answer depend on geometry (one sealed room
vs two rooms joined by a door) or on the shipped-but-inert air damping lever?
ANALYSIS ONLY.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import storm_probe as sp  # noqa: E402

KEYS = ["ke_last_decile_mean", "ke_peak", "umax_peak", "umax_final",
        "dp_mean_tail", "dom_period_s"]


def do(label, **kw):
    res = sp.run(**kw)
    return (label, res, sp.analyse(res))


def table(title, runs):
    print(f"\n=== {title} ===")
    print(f"  {'case':30s} " + "  ".join(f"{k:>12s}" for k in KEYS))
    for label, res, s in runs:
        cells = []
        for k in KEYS:
            v = s.get(k)
            cells.append("n/a" if v is None else f"{v:.4g}")
        print(f"  {label:30s} " + "  ".join(f"{c:>12s}" for c in cells))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    # P-F1b furniture plateau: T swings ~210-331 about ~285 -> mid 285, amp 45.
    MID, AMP = 285.0, 45.0
    PERIODS = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)

    if which in ("all", "freq"):
        runs = [do(f"room12 period={p}s", mode="pulse", geom="room", interior=12,
                   tile=0.5, seconds=max(120.0, 4 * p), damp=0.0, dT=MID,
                   amp=AMP, period=p) for p in PERIODS]
        table("F1 — frequency response, ONE sealed room (flicker amp +-45 game)", runs)

        runs = [do(f"tworoom period={p}s", mode="pulse", geom="tworoom",
                   interior=12, tile=0.5, seconds=max(120.0, 4 * p), damp=0.0,
                   dT=MID, amp=AMP, period=p) for p in PERIODS]
        table("F2 — frequency response, TWO rooms + door (the B2 geometry)", runs)

    if which in ("all", "amp"):
        runs = [do(f"tworoom amp={a}", mode="pulse", geom="tworoom", interior=12,
                   tile=0.5, seconds=120.0, damp=0.0, dT=MID, amp=a, period=10.0)
                for a in (0.0, 15.0, 45.0, 90.0)]
        table("F3 — flicker AMPLITUDE scaling (two rooms, period 10 s)", runs)

    if which in ("all", "damp"):
        runs = [do(f"tworoom damp={d}", mode="pulse", geom="tworoom", interior=12,
                   tile=0.5, seconds=120.0, damp=d, dT=MID, amp=AMP, period=10.0)
                for d in (0.0, 0.005, 0.01, 0.02, 0.05)]
        table("F4 — DAMPING at full flicker drive (two rooms, period 10 s)", runs)

        runs = [do(f"room12 damp={d}", mode="pulse", geom="room", interior=12,
                   tile=0.5, seconds=120.0, damp=d, dT=MID, amp=AMP, period=10.0)
                for d in (0.0, 0.01, 0.05)]
        table("F5 — DAMPING at full flicker drive (one room, period 10 s)", runs)

    if which in ("all", "phi"):
        # Erik's phi: scale the EXPANSION-felt excess, leaving the flicker shape.
        # Emulated by scaling both mean and swing about ambient.
        runs = []
        for phi in (1.0, 0.5, 0.33, 0.2):
            runs.append(do(f"tworoom phi={phi}", mode="pulse", geom="tworoom",
                           interior=12, tile=0.5, seconds=120.0, damp=0.0,
                           dT=MID * phi, amp=AMP * phi, period=10.0))
        table("F6 — Erik's phi: scaling the expansion-felt temperature (two rooms)",
              runs)

    print("\ndone.")


if __name__ == "__main__":
    main()
