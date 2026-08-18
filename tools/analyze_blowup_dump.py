"""Analyze a recorder blowup dump — is it THERMAL or MASS/MOMENTUM driven?

Written during the energy-books arc's HUMAN-TEST (2026-08-17) to separate two
in-game blowups that looked identical from the outside and had different causes:

  * `debug_blowup_20260817_051730` (k_drag 0.5, k_drag_heat_frac 1.0) —
    THERMAL. T slammed into the 16000 ceiling across 739 cells while pressure
    was still a normal 1.16 atm, and pressure followed it up to 66 atm one
    snapshot later. Cause: the drag heat deposit scales with u^2, so at blast
    velocities an explosion's own wind self-immolates. Fixed by shipping
    k_drag_heat_frac = 0.0014.
  * `debug_blowup_20260817_051006` (k_drag 0.0) — NOT thermal. T never exceeded
    741 (normal fire range, zero cells near the ceiling) yet pressure hit 98
    atm and P_min went NEGATIVE (-0.98). At a normal ~700 game-T, 98 atm means
    p* = C*N*T_abs is being driven by N: ~29x ambient density in one cell.
    That is the open pressure/momentum question the next arc owns.

The discriminator is WHICH FIELD MOVES FIRST. Temperature leading pressure is a
thermal runaway; pressure moving at flat temperature is a mass/momentum event.

Usage:
    conda run -n data python tools/analyze_blowup_dump.py <dump.npz> [--tail N]

Dumps recorded from 2026-08-17 (commit df088f1) onward also carry `wind_x`,
`wind_y` and `inert_n2`, and this tool reports the momentum/density picture when
they are present. Older dumps carry neither: wind is NOT recoverable from the
pressure field (the gradient is the per-tick ACCELERATION while u is its
accumulated history — the two run ~90 deg out of phase in the Helmholtz mode),
which is exactly why those planes were added to Recorder.DEFAULT_FIELDS.
"""
from __future__ import annotations

import argparse

import numpy as np

T_MAX_PHYS = 16000.0   # eos_solver.h — the counted physical-maximum rail on T
FP_ONE = 65536.0       # Q16.16 scale (the recorder's dequantize divisor)
N_AMBIENT = 1.0        # ambient bulk N: the o2+n2 split sums to exactly 1.0
                       # (config.toml [physics.eos]: 0.21 + 0.79 in Q16.16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", help="path to a debug_blowup_*.npz")
    ap.add_argument("--tail", type=int, default=30,
                    help="snapshots of context to print around the event")
    args = ap.parse_args()

    d = np.load(args.dump)
    have = set(d.keys())
    P = d["atmosphere"].astype(np.float64)
    T = d["temperature"].astype(np.float64)
    n = P.shape[0]

    flat = lambda a: a.reshape(n, -1)          # noqa: E731
    pmax, pmin = flat(P).max(axis=1), flat(P).min(axis=1)
    tmax = flat(T).max(axis=1)

    print(f"{args.dump}: {n} snapshots, grid {P.shape[1]}x{P.shape[2]}")
    print(f"fields: {sorted(have)}\n")

    # --- the discriminator -------------------------------------------------
    # First crossing of a "clearly abnormal" level on each axis. Whichever
    # crosses FIRST names the driver.
    t_trip = int(np.argmax(tmax > 0.5 * T_MAX_PHYS)) if (tmax > 0.5 * T_MAX_PHYS).any() else None
    p_trip = int(np.argmax(pmax > 5.0)) if (pmax > 5.0).any() else None

    print(f"global T_max = {tmax.max():10.1f}   (ceiling {T_MAX_PHYS:.0f})")
    print(f"global P_max = {pmax.max():10.3f} atm")
    print(f"global P_min = {pmin.min():10.3f} atm"
          f"{'   <-- NEGATIVE (unphysical)' if pmin.min() < -1e-6 else ''}")
    print(f"first snap T_max > {0.5 * T_MAX_PHYS:.0f}: {t_trip}")
    print(f"first snap P_max > 5 atm       : {p_trip}")

    if t_trip is not None and (p_trip is None or t_trip < p_trip):
        verdict = ("THERMAL-DRIVEN — temperature leads pressure. Suspect a heat "
                   "source: drag deposit (u^2-scaled), combustion, or compression work.")
    elif p_trip is not None and (t_trip is None or p_trip < t_trip):
        verdict = ("MASS/MOMENTUM-DRIVEN — pressure moves while temperature is "
                   "normal, so p* = C*N*T_abs is being driven by N (density).")
    else:
        verdict = "INCONCLUSIVE — neither axis crossed its abnormal threshold."
    print(f"\nVERDICT: {verdict}\n")

    # --- density reconstruction at the worst cell --------------------------
    worst = int(np.argmax(pmax))
    yx = np.unravel_index(np.argmax(P[worst]), P[worst].shape)
    p_w, t_w = P[worst][yx], T[worst][yx]
    if "gas_o2" in have and "inert_n2" in have:
        # UNITS (fixed 2026-08-18): Recorder.record() dequantizes a NAMED list
        # of planes by /65536 and `gas_o2` is on it while `inert_n2` is NOT
        # (recorder.py:172-176). Summing them raw therefore mixes physical
        # units with Q16.16 counts and overstates N by ~65536x — this printed
        # "36973768 x ambient" for a cell that is really ~714x. Normalise the
        # raw plane before summing, and state the ambient reference explicitly.
        o2_w = float(d["gas_o2"][worst][yx])            # already physical
        n2_w = float(d["inert_n2"][worst][yx]) / FP_ONE  # raw Q16.16 -> physical
        n_bulk = o2_w + n2_w
        src = "measured (gas_o2 + inert_n2/65536)"
    else:
        # p* = C*N*T_abs with C = 1/290 => N = P * 290 / T_abs
        n_bulk = p_w * 290.0 / (t_w + 290.0)
        src = "RECONSTRUCTED from p*/T (dump predates the inert_n2 field)"
    print(f"peak-pressure cell {yx} at snap {worst}: "
          f"P={p_w:.3f} atm  T={t_w:.1f}  N~={n_bulk / N_AMBIENT:.3f} x ambient "
          f"(N={n_bulk:.4f}, ambient={N_AMBIENT})  [{src}]")

    if "wind_x" in have and "wind_y" in have:
        wx, wy = d["wind_x"].astype(np.float64), d["wind_y"].astype(np.float64)
        speed = np.hypot(wx, wy)
        print(f"peak |u| over run: {speed.max():.2f} m/s "
              f"(at snap {int(np.argmax(speed.reshape(n, -1).max(axis=1)))})")
        print(f"|u| at that cell : {speed[worst][yx]:.2f} m/s")
    else:
        print("wind planes ABSENT — momentum analysis impossible for this dump. "
              "Re-record: wind_x/wind_y are in Recorder.DEFAULT_FIELDS since df088f1.")

    # --- context trace -----------------------------------------------------
    lo = max(0, n - args.tail)
    print(f"\nsnap |    P_max     P_min |     T_max")
    for i in range(lo, n):
        flag = ""
        if tmax[i] >= T_MAX_PHYS - 1:
            flag += "  T@CEILING"
        if pmin[i] < -1e-6:
            flag += "  P<0"
        print(f"{i:4d} | {pmax[i]:9.3f} {pmin[i]:9.3f} | {tmax[i]:9.2f}{flag}")


if __name__ == "__main__":
    main()
