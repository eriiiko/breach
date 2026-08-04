"""inspect_fire.py — compare fire traces between runs (ANALYSIS ONLY)."""
import csv
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results"


def load(p):
    rows = list(csv.DictReader(l for l in open(p, encoding="utf-8")
                               if not l.startswith("#")))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    for name in sys.argv[1:]:
        p = R / name
        if not p.exists():
            print(f"missing {p}")
            continue
        d = load(p)
        t = d["t"]
        print(f"\n--- {name} ---")
        if "I" in d:
            I, T = d["I"], d["T_fire"]
            alive = I > 0.02
            print(f"  I    peak {I.max():.3f} @ {t[I.argmax()]:6.1f}s | "
                  f"final {I[-1]:.3f} | alive {alive.mean()*100:4.1f}% of run")
            print(f"  Tfire peak {T.max():6.0f} @ {t[T.argmax()]:6.1f}s | "
                  f"final {T[-1]:6.0f} | mean-while-alive "
                  f"{T[alive].mean() if alive.any() else float('nan'):6.0f}")
        ke, um = d["ke"], d["umax"]
        print(f"  ke   peak {ke.max():9.1f} @ {t[ke.argmax()]:6.1f}s | "
              f"final {ke[-1]:9.1f}")
        print(f"  umax peak {um.max():9.2f} @ {t[um.argmax()]:6.1f}s | "
              f"final {um[-1]:6.2f}")
        print("     t(s)      I    Tfire         ke     umax")
        for frac in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
            i = int(frac * (len(t) - 1))
            Iv = d["I"][i] if "I" in d else float("nan")
            Tv = d["T_fire"][i] if "T_fire" in d else float("nan")
            print(f"  {t[i]:7.1f}  {Iv:5.3f}  {Tv:7.0f}  {ke[i]:9.1f}  {um[i]:7.2f}")


if __name__ == "__main__":
    main()
