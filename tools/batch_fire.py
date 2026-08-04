"""batch_fire.py — the real-fire storming battery (P-F1b dials).
Does the shipped-but-inert air-damping lever calm a real fire's atmosphere,
and how does the effect depend on connected geometry? ANALYSIS ONLY.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import storm_probe as sp  # noqa: E402

KEYS = ["ke_last_decile_mean", "ke_peak", "ke_retention", "umax_peak",
        "umax_final", "dp_mean_tail"]


def do(label, **kw):
    kw.setdefault("dials", dict(sp.PF1B))
    res = sp.run(**kw)
    return (label, res, sp.analyse(res))


def table(title, runs):
    print(f"\n=== {title} ===")
    print(f"  {'case':28s} " + "  ".join(f"{k:>12s}" for k in KEYS))
    for label, res, s in runs:
        cells = ["n/a" if s.get(k) is None else f"{s[k]:.4g}" for k in KEYS]
        print(f"  {label:28s} " + "  ".join(f"{c:>12s}" for c in cells))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    SEC = 200.0

    if which in ("all", "damp"):
        runs = [do(f"fire tworoom damp={d}", mode="fire", geom="tworoom",
                   interior=12, tile=0.5, seconds=SEC, damp=d)
                for d in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)]
        table("R1 — REAL FIRE, two rooms: the air-damping lever", runs)

    if which in ("all", "door"):
        runs = [do(f"fire door={w}", mode="fire", geom="tworoom", interior=12,
                   tile=0.5, seconds=SEC, damp=0.0, door=w)
                for w in (1, 2, 3, 6)]
        table("R2 — REAL FIRE: door width (undamped)", runs)

    print("\ndone.")


if __name__ == "__main__":
    main()
