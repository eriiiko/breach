"""Analyze a recorder blowup dump — is it THERMAL or MASS/MOMENTUM driven?

Written during the energy-books arc's HUMAN-TEST (2026-08-17) to separate two
in-game blowups that looked identical from the outside and had different causes:

  * `debug_blowup_20260817_051730` (k_drag 0.5, k_drag_heat_frac 1.0 --
    that dial is RETIRED at arc #54 P-G1a, design D5; the dump predates it) --
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


def bulk_n_planes(d):
    """Total bulk N per snapshot, in PHYSICAL cell-equivalents (ambient = 1.0).

    THE UNITS RULE, in one place so it cannot be got wrong again (it has been,
    twice — see this module's header and the mass-books arc kickoff):
    ``Recorder.record()`` dequantizes a NAMED list of planes by /65536.
    ``gas_o2`` is on that list; ``inert_n2`` is NOT. Summing them raw mixes
    physical values with Q16.16 counts and overstates N by ~65536x.
    """
    return (d["gas_o2"].astype(np.float64)
            + d["inert_n2"].astype(np.float64) / FP_ONE)


def mass_books(d, have) -> None:
    """Are the MASS books closed? Totals, and the catalogue of events that move
    them (mass-books arc P-M0, docs/mass_books_arc_kickoff_2026-08-18.md §1).

    A total alone hides the structure that actually names a culprit: whether the
    mint is diffuse or arrives in discrete events, whether payloads repeat (a
    fired constant) or never do (proportional to local state), and whether they
    coincide with the obstacle grid changing.
    """
    if not {"gas_o2", "inert_n2"} <= have:
        print("\nMASS BOOKS: dump predates the inert_n2 field — cannot audit N.")
        return
    N = bulk_n_planes(d)
    s = N.shape[0]
    tot = N.reshape(s, -1).sum(axis=1)
    dn = np.diff(tot)
    excess = tot[-1] - tot[0]

    print(f"\n{'=' * 62}\nMASS BOOKS (physical cell-equivalents, ambient = 1.0)")
    print(f"  total N   start {tot[0]:12.1f}   final {tot[-1]:12.1f}   "
          f"ratio {tot[-1] / tot[0]:6.3f}")
    print(f"  net change      {excess:+12.1f} cell-equivalents")
    print(f"  worst cell      {N.max():12.1f} x ambient")
    print(f"  snaps where N falls: {int((dn < 0).sum())} of {len(dn)}"
          f"   (largest single fall {-dn.min() if dn.min() < 0 else 0:.2f})")

    ob = d["obstacles"] if "obstacles" in have else None
    dsolid = np.diff(ob.reshape(s, -1).sum(axis=1)) if ob is not None else None

    ev = np.nonzero(dn > 1.0)[0]
    if not len(ev):
        print("  no discrete deposit events > 1 cell-eq.")
        return
    print(f"\n  {len(ev)} events > 1 cell-eq deliver {dn[ev].sum():.1f} "
          f"({100 * dn[ev].sum() / excess:.1f}% of the net change); "
          f"all other snaps net {dn[dn <= 1.0].sum():+.1f}")

    print(f"\n  {'snap':>6} {'payload':>10} {'cells':>6} {'peak':>9}  walls")
    rows = []
    for i in ev:
        delta = N[i + 1] - N[i]
        hot = delta > max(delta.max() * 0.01, 1e-3)
        walls = int(-dsolid[i]) if dsolid is not None and dsolid[i] < 0 else 0
        rows.append((int(i), float(dn[i]), int(hot.sum()), float(delta.max()), walls))
        print(f"  {i:6d} {dn[i]:10.2f} {int(hot.sum()):6d} {delta.max():9.2f}"
              f"  {'-' + str(walls) if walls else '.':>5}")

    pay = np.array([r[1] for r in rows])
    rep = {v: c for v, c in zip(*np.unique(np.round(pay, 1), return_counts=True)) if c > 1}
    print(f"\n  payloads: {len(np.unique(np.round(pay, 1)))} distinct in {len(rows)} events")
    for v, c in sorted(rep.items(), key=lambda t: -t[0] * t[1])[:5]:
        print(f"    {v:9.2f} cell-eq recurs x{c}  <- a FIXED payload (a fired constant)")
    if dsolid is not None:
        w = np.array([r[4] for r in rows]) > 0
        print(f"  coinciding with a wall break: {int(w.sum())}/{len(rows)} events, "
              f"{100 * pay[w].sum() / pay.sum():.1f}% of the deposited mass")
        print("  (a payload that does NOT repeat, riding wall breaks, is "
              "proportional to\n   local state — see destroy_wall's neighbour-mean "
              "seed, gamemap.py:1752-1754)")


def mach_census(d, have) -> None:
    """--mach-census (P-W0, docs/tabs_compression_work_design_2026-08-20.md
    §6): per-snapshot census for the T_abs compression-work arc's D-1/B-F7
    data needs — sub-ambient (bulk N < N_AMBIENT) open-cell count, the
    |u|/c_own percentile distribution (c_own = 300*sqrt(t_abs/290), t_abs =
    T_game + 290 — floats fine, this is an analysis tool, not the sim path),
    min P over open cells, and a max|grad P| proxy (max abs neighbour diff
    of the atmosphere plane). Follows this module's existing unit convention
    (see the header docstring + ``bulk_n_planes``): ``atmosphere`` and
    ``temperature`` are ALREADY dequantized to physical units by
    ``Recorder.record()`` — no /65536 on those two planes here."""
    P = d["atmosphere"].astype(np.float64)
    T = d["temperature"].astype(np.float64)
    n = P.shape[0]
    flat = lambda a: a.reshape(n, -1)          # noqa: E731

    if "obstacles" in have:
        open_mask = ~d["obstacles"].astype(bool)
    else:
        open_mask = np.ones_like(P, dtype=bool)
        print("mach-census: no 'obstacles' field -- treating ALL cells as "
              "open (census includes walls).")
    open_flat = flat(open_mask)

    if {"gas_o2", "inert_n2"} <= have:
        N = bulk_n_planes(d)
        n_src = "measured (gas_o2 + inert_n2/65536)"
    else:
        N = P * 290.0 / (T + 290.0)             # p* = C*N*T_abs, C = 1/290
        n_src = "RECONSTRUCTED from p*/T_abs (dump predates inert_n2)"

    sub_amb_count = flat((N < N_AMBIENT) & open_mask).sum(axis=1)

    t_abs = T + 290.0
    c_own = 300.0 * np.sqrt(t_abs / 290.0)

    have_wind = {"wind_x", "wind_y"} <= have
    if have_wind:
        wx, wy = d["wind_x"].astype(np.float64), d["wind_y"].astype(np.float64)
        ratio_flat = flat(np.hypot(wx, wy) / np.maximum(c_own, 1e-9))
        open_ratio = [ratio_flat[i][open_flat[i]] for i in range(n)]
    else:
        print("mach-census: no wind_x/wind_y -- |u|/c_own percentiles "
              "unavailable for this dump.")

    Pf = flat(P)
    min_P = np.array([Pf[i][open_flat[i]].min() if open_flat[i].any()
                      else np.nan for i in range(n)])

    dPy = np.abs(np.diff(P, axis=1)).reshape(n, -1)
    dPx = np.abs(np.diff(P, axis=2)).reshape(n, -1)
    max_grad_P = np.maximum(dPy.max(axis=1) if dPy.size else np.zeros(n),
                            dPx.max(axis=1) if dPx.size else np.zeros(n))

    print(f"\n{'=' * 62}\nMACH CENSUS (N source: {n_src})")
    print(f"  sub-ambient (N < {N_AMBIENT}) open cells: peak "
          f"{int(sub_amb_count.max())} (snap {int(np.argmax(sub_amb_count))}), "
          f"total snap-cells {int(sub_amb_count.sum())}")
    print(f"  min P over open cells: {np.nanmin(min_P):.4f} atm "
          f"(snap {int(np.nanargmin(min_P))})")
    print(f"  max|grad P| proxy (neighbour diff): {max_grad_P.max():.4f} atm "
          f"(snap {int(np.argmax(max_grad_P))})")

    header = f"  {'snap':>6} {'sub_amb':>8} {'min_P':>9} {'max|dP|':>9}"
    if have_wind:
        header += f" {'u/c_p50':>8} {'u/c_p90':>8} {'u/c_p99':>8} {'u/c_max':>8}"
    print(header)
    for i in range(n):
        row = (f"  {i:6d} {int(sub_amb_count[i]):8d} {min_P[i]:9.4f} "
               f"{max_grad_P[i]:9.4f}")
        if have_wind:
            arr = open_ratio[i]
            if arr.size:
                row += (f" {np.percentile(arr, 50):8.3f} "
                        f"{np.percentile(arr, 90):8.3f} "
                        f"{np.percentile(arr, 99):8.3f} {arr.max():8.3f}")
            else:
                row += f" {'--':>8} {'--':>8} {'--':>8} {'--':>8}"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", help="path to a debug_blowup_*.npz")
    ap.add_argument("--tail", type=int, default=30,
                    help="snapshots of context to print around the event")
    ap.add_argument("--mass-books", action="store_true",
                    help="audit bulk N: totals + the catalogue of events that "
                         "move them (mass-books arc)")
    ap.add_argument("--mach-census", action="store_true",
                    help="sub-ambient open-cell census + |u|/c_own "
                         "percentiles + min P + max|grad P| proxy per "
                         "snapshot (T_abs compression-work arc, P-W0)")
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

    if args.mass_books:
        mass_books(d, have)

    if args.mach_census:
        mach_census(d, have)

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
