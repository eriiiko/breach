"""The two figures of `report_p0.md`, measured with the integer reference.

    C:/Users/steen/anaconda3/python.exe p0_figures.py

Writes `p0_isotropy.png` and `p0_stability.png` beside this file and prints every
number it plots, so the report quotes the run rather than the picture.

Colour is by ROLE, one mapping across every panel (validated CVD-safe pair):
    blue   #0072B2  the scheme AS DESIGNED (shear / Fleck+clamp / damped source)
    orange #D55E00  the scheme WITHOUT the protection (step / Fleck alone / undamped)
    grey   #6E7781  the truth (analytic solution / exact equilibrium) -- dashed,
                    a reference line, never a category
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sweep_ref_q as R                  # noqa: E402
import sweep_ref_q_gates as G            # noqa: E402

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#6E7781"
INK, MUTED = "#24292f", "#57606a"
N_GRID = 81
PHI_IGN = R.E[R.e_bucket_of(G.T_IGN_GAME << 16)]


def _style(ax):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_color("#d8dade")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color="#e6e8ea", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
def figure_isotropy():
    """The 64-direction ignition footprint, shear vs step, three source sizes."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4),
                             subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")
    print("\n=== figure 1: the ignition footprint "
          f"(grid {N_GRID}x{N_GRID}, source {G.T_SRC_GAME} game, leak k=0.10, "
          f"ambient ring ON, threshold E°[{G.T_IGN_GAME}] = {PHI_IGN}) ===")
    print(f"{'source':>8}{'transport':>11}{'mean r':>9}{'min r':>8}{'max r':>8}"
          f"{'spread':>9}")
    for ax, (half, nm) in zip(axes, ((0, "1 tile"), (1, "3x3"), (2, "5x5"))):
        rmax = 0.0
        for transport, colour in (("shear", BLUE), ("step", ORANGE)):
            fl, c = G._point_fluence(transport, half, N_GRID)
            mean_r, spread, radii = G.footprint(fl, c, PHI_IGN)
            th = np.array([2 * math.pi * k / len(radii) for k in range(len(radii))])
            rr = np.array(radii)
            print(f"{nm:>8}{transport:>11}{mean_r:9.2f}{rr.min():8.2f}{rr.max():8.2f}"
                  f"{spread:9.3f}")
            ax.plot(np.append(th, th[0]), np.append(rr, rr[0]), color=colour,
                    linewidth=2.0, label=f"{transport}  spread {spread:.2f}")
            rmax = max(rmax, rr.max())
        ax.set_title(f"{nm} source", color=INK, fontsize=11, pad=14)
        ax.set_ylim(0, rmax * 1.12)
        ax.set_facecolor("white")
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.grid(color="#e6e8ea", linewidth=0.6)
        ax.set_xticks(np.linspace(0, 2 * math.pi, 8, endpoint=False))
        ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
        ax.set_rlabel_position(202.5)
        ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.14), frameon=False,
                  fontsize=8, labelcolor=INK)
    fig.suptitle("Ignition-footprint radius over 64 directions — the isotropy the "
                 "transport step buys (P0, integer reference)",
                 color=INK, fontsize=12, y=0.99)
    fig.text(0.5, 0.012, "radius in tiles at which the fluence crosses "
             "E°[ignition_temp]; a perfect circle is spread 1.00",
             ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    out = HERE / "p0_isotropy.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# --------------------------------------------------------------------------- #
def figure_stability():
    """Cooling curve, equilibrium table, and the damped source."""
    A, HIS = G.A_FURNITURE, G.HIS_FURNITURE
    phi_amb = R.E0
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))
    fig.patch.set_facecolor("white")

    # ---- panel A: cooling a 1263-game cell for 0.5 s ----------------------- #
    ax = axes[0]
    _style(ax)
    n = int(0.5 * R.TICK_HZ)
    tr_e, _ = R.cell_march(G.T_SRC_GAME << 16, phi_amb, A, HIS, n, fleck=False,
                           clamp_enabled=False, trace=True)
    tr_f, _ = R.cell_march(G.T_SRC_GAME << 16, phi_amb, A, HIS, n, fleck=True,
                           clamp_enabled=False, trace=True)
    t = np.arange(n + 1) / R.TICK_HZ
    an = [R.cool_exact_bath(float(G.T_SRC_GAME), float(s), 0.5, HIS) for s in t]
    ye = [v / 65536 for v in tr_e]
    yf = [v / 65536 for v in tr_f]
    ax.plot(t, an, color=GREY, linewidth=4.0, alpha=0.55, solid_capstyle="round",
            label="analytic", zorder=1)
    ax.plot(t, ye, color=ORANGE, linewidth=1.8, label="explicit (no Fleck)", zorder=2)
    ax.plot(t, yf, color=BLUE, linewidth=1.8, label="Fleck, excess form", zorder=3)
    ax.set_xlabel("seconds", color=MUTED, fontsize=9)
    ax.set_ylabel("cell temperature (game units)", color=MUTED, fontsize=9)
    ax.set_title(f"cooling a {G.T_SRC_GAME}-game cell in an ambient bath",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    err_e = (ye[-1] - an[-1]) / an[-1] * 100
    err_f = (yf[-1] - an[-1]) / an[-1] * 100
    ax.annotate(f"{err_e:+.2f}%", (t[-1], ye[-1]), color=ORANGE, fontsize=8,
                xytext=(-4, -12), textcoords="offset points", ha="right")
    ax.annotate(f"{err_f:+.2f}%", (t[-1], yf[-1]), color=BLUE, fontsize=8,
                xytext=(-4, 6), textcoords="offset points", ha="right")
    print("\n=== figure 2A: cooling 1263 game for 0.5 s (a = 0.5, his = 3) ===")
    print(f"  explicit {ye[-1]:8.1f} ({err_e:+.2f}%)   Fleck {yf[-1]:8.1f} "
          f"({err_f:+.2f}%)   analytic {an[-1]:8.1f}")

    # ---- panel B: the equilibrium table ------------------------------------ #
    ax = axes[1]
    _style(ax)
    srcs = (1263, 5000, 16000)
    iters = 400 * 24
    true_v, fleck_v, clamp_v = [], [], []
    print("\n=== figure 2B: equilibrium under a held Phi = 0.25 x E°[T_src] ===")
    print(f"  {'T_src':>7}{'true':>10}{'Fleck only':>16}{'Fleck+clamp':>13}")
    for ts in srcs:
        phi = int(0.25 * R.E[R.e_bucket_of(ts << 16)])
        tv = R.e_inv_q(phi) >> 16
        fv, _ = R.cell_march(0, phi, A, HIS, iters, clamp_enabled=False,
                             rails_enabled=False, int32_sat=False)
        cv, _ = R.cell_march(0, phi, A, HIS, iters, clamp_enabled=True)
        true_v.append(tv)
        fleck_v.append(fv / 65536)
        clamp_v.append(cv / 65536)
        print(f"  {ts:>7}{tv:>10}{fv / 65536:>16,.1f}{cv / 65536:>13.1f}")
    # a DOT plot, not bars: the values span four decades, and on a log axis only a
    # position encoding is honest (a bar's length would encode the logarithm).
    xs = np.arange(len(srcs), dtype=float)
    for x, tv, fv, cv in zip(xs, true_v, fleck_v, clamp_v):
        ax.plot([x, x], [min(tv, cv), fv], color="#d8dade", linewidth=1.4, zorder=1)
    # the truth is an open ring; the clamped value is a square drawn inside it, so
    # "the clamp lands exactly on the truth" is visible, not just asserted
    ax.plot(xs, true_v, "o", markerfacecolor="none", markeredgecolor=GREY,
            markersize=15, markeredgewidth=2.2, linestyle="none",
            label="true (E°⁻¹ of Φ)", zorder=2)
    ax.plot(xs, fleck_v, "^", color=ORANGE, markersize=9, linestyle="none",
            markeredgecolor="white", markeredgewidth=1.0, label="Fleck alone", zorder=3)
    ax.plot(xs, clamp_v, "s", color=BLUE, markersize=7, linestyle="none",
            markeredgecolor="white", markeredgewidth=1.0, label="Fleck + clamp",
            zorder=4)
    for x, fv, cv in zip(xs, fleck_v, clamp_v):
        ax.annotate(f"{fv:,.0f}", (x, fv), xytext=(11, 5), textcoords="offset points",
                    ha="left", color=ORANGE, fontsize=7.5)
        ax.annotate(f"{cv:,.0f}", (x, cv), xytext=(11, -11), textcoords="offset points",
                    ha="left", color=BLUE, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(300, 4e7)
    ax.set_xlim(-0.45, len(srcs) - 0.25)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s} game" for s in srcs])
    ax.set_xlabel("held source temperature", color=MUTED, fontsize=9)
    ax.set_ylabel("receiver equilibrium (game units, log)", color=MUTED, fontsize=9)
    ax.set_title("the maximum-principle clamp restores the equilibrium",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")

    # ---- panel C: what the Fleck factor costs in emission ------------------ #
    ax = axes[2]
    _style(ax)
    # the last sample IS the table top, so the "xN at the table top" annotation
    # quotes the same number the gates and report_p0.md do (P0b)
    temps = sorted({4 * b for b in range(0, R.E_TABLE_SIZE, 8)}
                   | {R.T_TABLE_TOP_GAME})
    undamped = [R.E[R.e_bucket_of(t << 16)] for t in temps]
    damped = [R.damped_source_q(t << 16, A, HIS) for t in temps]
    ax.plot(temps, undamped, color=ORANGE, linewidth=2.0, label="E°[T] (black body)")
    ax.plot(temps, damped, color=BLUE, linewidth=2.0,
            label="E°[0] + f·(E°[T] − E°[0]) (what the cell emits)")
    ax.set_yscale("log")
    ax.set_xlabel("cell temperature (game units)", color=MUTED, fontsize=9)
    ax.set_ylabel("emissive power (counts, log)", color=MUTED, fontsize=9)
    ax.set_title("the rate the stability fix costs", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    ratio = undamped[-1] / damped[-1]
    ax.annotate(f"×{ratio:.0f} at the table top", (temps[-1], damped[-1]),
                xytext=(-8, 10), textcoords="offset points", ha="right",
                color=INK, fontsize=8.5)
    print("\n=== figure 2C: the damped source (a = 0.5, his = 3) ===")
    for t in (280, 1263, 1800, 5000, 15996):
        u = R.E[R.e_bucket_of(t << 16)]
        d = R.damped_source_q(t << 16, A, HIS)
        print(f"  T={t:6d}: E°={u:>16,}  damped={d:>16,}  ratio x{u / d:8.1f}  "
              f"f={R.fleck_f_solid_q(t << 16, A, HIS)[0] / R.F_ONE:.7f}")

    fig.suptitle("Stability on the NEW forms: the excess-form Fleck factor and the "
                 "corrected clamp (P0, integer reference)", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = HERE / "p0_stability.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


def extras():
    """The characterisation tables report_p0.md quotes beside the gates."""
    print("\n=== extra 1: the footprint in the FLOAT STUDY's configuration "
          "(no ambient inflow, threshold relative to the source) ===")
    print("  reproduces reach_and_leak_study.py section 5b, which the design quotes")
    g_ign = (573.0 / 1556.0) ** 4
    e_src = float(R.E[R.e_bucket_of(G.T_SRC_GAME << 16)])
    print(f"{'source':>8}{'transport':>11}{'mean r':>9}{'spread':>9}")
    for transport in ("shear", "step"):
        for half, nm in ((0, "1 tile"), (1, "3x3"), (2, "5x5")):
            fl, c = G._point_fluence(transport, half, N_GRID, e_ref=0)
            mean_r, spread, _ = G.footprint(fl, c, g_ign * e_src)
            print(f"{nm:>8}{transport:>11}{mean_r:9.2f}{spread:9.3f}")

    print("\n=== extra 2: the sweep's own geometric factor G = (Phi - E°[0]) / E°[T_src] "
          "===")
    print("  (the study's 0-D tables assume G = 0.25 for 'an adjacent tile')")
    n, c = 21, 10
    e_src_i = R.E[R.e_bucket_of(G.T_SRC_GAME << 16)]
    for transport in ("shear", "step"):
        a = R.plane(n, n, 0)
        d = R.plane(n, n, 0)
        T = R.plane(n, n, 0)
        a[c][c] = R.ONE
        d[c][c] = R.ONE
        T[c][c] = G.T_SRC_GAME << 16
        fl = R.sweep_q(a, d, R.plane(n, n, 0), T, transport=transport).rad_fluence
        axis = [(fl[c][c + i] - R.E0) / e_src_i for i in (1, 2, 3, 4)]
        diag = (fl[c + 1][c + 1] - R.E0) / e_src_i
        print(f"  {transport:5s}: G on the axis at 1,2,3,4 tiles = "
              + ", ".join(f"{v:.4f}" for v in axis) + f";  diagonal (1,1) = {diag:.4f}")
    a = R.plane(n, n, 0)
    d = R.plane(n, n, 0)
    T = R.plane(n, n, 0)
    for y in range(c - 1, c + 2):
        for x in range(c - 1, c + 2):
            if (y, x) != (c, c):
                a[y][x] = R.ONE
                d[y][x] = R.ONE
                T[y][x] = G.T_SRC_GAME << 16
    fl = R.sweep_q(a, d, R.plane(n, n, 0), T).rad_fluence
    print(f"  a cell fully ringed by {G.T_SRC_GAME}-game walls: "
          f"G = {(fl[c][c] - R.E0) / e_src_i:.4f}, E°⁻¹(Phi) = "
          f"{R.e_inv_q(fl[c][c]) >> 16} game -- a cavity reaches its wall temperature")

    print("\n=== extra 3: is the 1.73-million-game runaway even representable? ===")
    phi = int(0.25 * R.E[R.e_bucket_of(16000 << 16)])
    t_un, _ = R.cell_march(0, phi, G.A_FURNITURE, G.HIS_FURNITURE, 9600,
                           clamp_enabled=False, rails_enabled=False, int32_sat=False)
    t_i32, c_i32 = R.cell_march(0, phi, G.A_FURNITURE, G.HIS_FURNITURE, 9600,
                                clamp_enabled=False, rails_enabled=False,
                                int32_sat=True)
    t_rail, c_rail = R.cell_march(0, phi, G.A_FURNITURE, G.HIS_FURNITURE, 9600,
                                  clamp_enabled=False, rails_enabled=True,
                                  int32_sat=True)
    print(f"  unbounded ints        : {t_un / 65536:>14,.1f} game")
    print(f"  int32 Q16.16 sat_add  : {t_i32 / 65536:>14,.1f} game "
          f"(INT32_MAX / 65536 = {R.INT32_MAX / 65536:,.1f})")
    print(f"  + the T_MAX_PHYS rail : {t_rail / 65536:>14,.1f} game, "
          f"t_max_phys_hits = {c_rail.t_max_phys_hits}")


if __name__ == "__main__":
    figure_isotropy()
    figure_stability()
    extras()
