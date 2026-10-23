"""P0b: THE FLECK ALPHA FLOOR, MEASURED FOR ERIK (design v3 section 12 item 4).

    C:/Users/steen/anaconda3/python.exe p0b_alpha_floor.py

A MEASUREMENT, NOT A CHANGE. It runs the same integer reference at both floors
side by side and writes `p0b_alpha_floor.png`, so Erik could settle section 12
item 4 on numbers instead of on the algebra. **He did, on 2026-09-16: FLOOR 0**
(design row 39), and P2a made it the reference's default -- this file names both
floors explicitly on every call, so it is unaffected by that default and still
prints the same comparison.

    floor 0 (RULED, row 39)  D = max(T_abs,      4L)   alpha = max(0, 1 - 1/g)
    floor 1/2 (superseded)   D = max(T_abs + 2L, 4L)   Fleck's own IMC bound

WHAT THE THREE MEASUREMENTS MEAN

  A DRIVEN SOURCE is a cell whose temperature is HELD by something other than
  radiation -- a burning tile refilled by combustion every tick. The Fleck factor
  assumes a cell will cool during the tick and pre-pays for that cooling by
  damping the emission; a driven cell never cools, so the damping is pure loss of
  radiated power. Column `emitted/black body` below is exactly how much of its
  black-body power such a tile actually delivers to its neighbours.

  FREE COOLING is the opposite case: nothing holds the cell, it starts hot and
  radiates its way down. There the damping is what keeps the explicit update from
  overshooting, and the error against the analytic solution is what it costs.

  THE STABILITY MULTIPLIER is the factor by which a small temperature error is
  multiplied each tick by the fixed-point map; the update is monotone and
  non-oscillatory exactly while it stays in (0, 1]. For the damped update it is
  `x = g(1 + alpha*g/4)/(1 + alpha*g)^2`, and floor 0 keeps it there for every g
  (section 12 item 4), which is why the choice is legitimate at all.
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

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#6E7781"
INK, MUTED = "#24292f", "#57606a"

FLOORS = ((R.ALPHA_FLOOR_HALF, "alpha floor 1/2 (superseded)", BLUE),
          (R.ALPHA_FLOOR_ZERO, "alpha floor 0 (RULED, row 39)", ORANGE))

# the two rows section 12 item 4 argues over: a burning crate and bare wood
ROWS = (("furniture / kindling", R.quant(0.5), 3),
        ("wood / door", R.ONE, 3))
DRIVEN_T = (300, 600, 900, 1263, 1800, 2500, 5000, 16000)
T_SRC_GAME = 1263
A_FURNITURE, HIS_FURNITURE = R.quant(0.5), 3


def _style(ax):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_color("#d8dade")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color="#e6e8ea", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def _emitted_fraction(T_game, a_q, his, floor):
    """(f, emitted excess / black-body excess) for a cell HELD at T_game."""
    T_q = T_game << 16
    f_q, L_q = R.fleck_f_solid_q(T_q, a_q, his, alpha_floor=floor)
    ex_bb = R.E[R.e_bucket_of(T_q)] - R.E0
    ex_emit = R.damped_source_q(T_q, a_q, his, alpha_floor=floor) - R.E0
    frac = (ex_emit / ex_bb) if ex_bb > 0 else 1.0
    g = 4.0 * (L_q / 65536.0) / (T_game + R.K_AMB)
    return f_q / R.F_ONE, frac, g


# --------------------------------------------------------------------------- #
def table_a_driven():
    """(a) What a cell HELD at T radiates, relative to the black body."""
    print("\n=== (a) THE DRIVEN SOURCE: a cell held at T by a deposit (combustion) ===")
    print("    'emitted/bb' is the excess a driven tile actually radiates, as a")
    print("    fraction of the black-body excess -- i.e. the Fleck factor itself.")
    for nm, a_q, his in ROWS:
        print(f"\n  {nm}  (heat_atten = {a_q / R.ONE:.2f}, thermal_mass = {1 << his})")
        print(f"    {'T game':>8}{'g = 4L/T_abs':>14}"
              f"{'f (floor 1/2)':>16}{'emitted/bb':>13}"
              f"{'f (floor 0)':>16}{'emitted/bb':>13}{'ratio 0 : 1/2':>15}")
        for T in DRIVEN_T:
            fh, frh, g = _emitted_fraction(T, a_q, his, R.ALPHA_FLOOR_HALF)
            fz, frz, _ = _emitted_fraction(T, a_q, his, R.ALPHA_FLOOR_ZERO)
            note = "  (table top)" if T > R.T_TABLE_TOP_GAME else ""
            print(f"    {T:>8}{g:>14.3f}{fh:>16.6f}{frh:>13.4f}"
                  f"{fz:>16.6f}{frz:>13.4f}{frz / frh:>15.2f}{note}")


def table_b_free_cooling():
    """(b) The free-cooling error at 0.5 s from 1263 game, both floors."""
    print("\n=== (b) FREE COOLING: 1263 game for 0.5 s, a = 0.5, thermal_mass = 8 ===")
    n = int(0.5 * R.TICK_HZ)
    phi_amb = R.E0
    analytic = R.cool_exact_bath(float(T_SRC_GAME), 0.5, 0.5, HIS_FURNITURE)
    e_new, _ = R.cell_march(T_SRC_GAME << 16, phi_amb, A_FURNITURE, HIS_FURNITURE, n,
                            fleck=False, clamp_enabled=False)
    print(f"    {'law':<40}{'T after 0.5 s':>15}{'error':>10}")
    print(f"    {'explicit (no Fleck)':<40}{e_new / 65536:>15.1f}"
          f"{(e_new / 65536 - analytic) / analytic * 100:>9.2f}%")
    out = {}
    for floor, label, _c in FLOORS:
        t, _ = R.cell_march(T_SRC_GAME << 16, phi_amb, A_FURNITURE, HIS_FURNITURE, n,
                            fleck=True, clamp_enabled=False, alpha_floor=floor)
        out[floor] = t / 65536
        print(f"    {'Fleck, ' + label:<40}{t / 65536:>15.1f}"
              f"{(t / 65536 - analytic) / analytic * 100:>+9.2f}%")
    print(f"    {'analytic dT/dt = -c[(T+K)^4 - K^4]':<40}{analytic:>15.1f}{'--':>10}")
    print("    (at this start g = 0.74 < 1, so floor 0 sets alpha = 0 and f = 1 "
          "exactly: floor 0 IS the explicit update here, to the last count)")
    return out


def table_c_stability_under_floor_zero():
    """(c) Is floor 0 safe? The same sweep G10 runs, plus its equilibrium table."""
    print("\n=== (c) STABILITY UNDER FLOOR 0 ===")
    phi_amb = R.E0
    for floor, label, _c in FLOORS:
        bad, worst_rise = [], 0
        for b in range(0, R.E_TABLE_SIZE, 100):
            T0 = 4 * b
            tr, cc = R.cell_march(T0 << 16, phi_amb, A_FURNITURE, HIS_FURNITURE, 48,
                                  fleck=True, trace=True, alpha_floor=floor)
            if any(v < 0 for v in tr) or cc.t_low_rail_hits:
                bad.append(("negative/railed", T0))
            rise = max((tr[i + 1] - tr[i] for i in range(len(tr) - 1)), default=0)
            if rise > 0:
                bad.append(("non-monotone", T0))
                worst_rise = max(worst_rise, rise)
        print(f"    {label}: 48 ticks from every start in 0..{R.T_TABLE_TOP_GAME} game "
              f"(stride 400) -> "
              + ("monotone, positive, finite -- all clean"
                 if not bad else f"VIOLATIONS {bad[:6]} worst rise {worst_rise}"))
    print("\n    the 0-D equilibrium table under a held Phi = 0.25 x E°[T_src], "
          "9600 ticks:")
    print(f"      {'T_src':>8}{'true (E_inv)':>14}"
          f"{'Fleck only, 1/2':>18}{'Fleck+clamp':>13}"
          f"{'Fleck only, 0':>18}{'Fleck+clamp':>13}")
    for T_src in (1263, 5000, 16000):
        phi = int(0.25 * R.E[R.e_bucket_of(T_src << 16)])
        t_true = R.e_inv_q(phi) >> 16
        cells = []
        for floor, _label, _c in FLOORS:
            t_f, _ = R.cell_march(0, phi, A_FURNITURE, HIS_FURNITURE, 9600,
                                  clamp_enabled=False, rails_enabled=False,
                                  int32_sat=False, alpha_floor=floor)
            t_c, _cc = R.cell_march(0, phi, A_FURNITURE, HIS_FURNITURE, 9600,
                                    clamp_enabled=True, alpha_floor=floor)
            cells += [t_f / 65536, t_c / 65536]
        print(f"      {T_src:>8}{t_true:>14}{cells[0]:>18,.1f}{cells[1]:>13,.1f}"
              f"{cells[2]:>18,.1f}{cells[3]:>13,.1f}")


# --------------------------------------------------------------------------- #
def figure():
    """(d) p0b_alpha_floor.png -- what a driven cell radiates, and what free
    cooling costs, at both floors."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    fig.patch.set_facecolor("white")

    # ---- left: emitted excess / black-body excess (= f) vs T --------------- #
    # LINEAR in both axes and cut at 3000 game: the whole question lives in the
    # fire range, where floor 0 sits at 1.0 and floor 1/2 does not. Above the
    # crossing the two floors are IDENTICAL (alpha = 1 - 1/g on both), so the
    # table top is a number in the corner, not four more decades of overlap.
    ax = axes[0]
    _style(ax)
    temps = [4 * b for b in range(0, 760)]           # 0 .. 3036 game
    ax.axvspan(280, 2000, color="#f2a900", alpha=0.10, zorder=0)
    ax.annotate("the fire range: ignition 280 — flame ~2000 game", (1140, 0.035),
                color=MUTED, fontsize=8, ha="center")
    for (nm, a_q, his), dash in zip(ROWS, ("-", "--")):
        for floor, label, colour in FLOORS:
            fr = [_emitted_fraction(t, a_q, his, floor)[1] for t in temps]
            ax.plot(temps, fr, color=colour, linewidth=1.9, linestyle=dash,
                    label=f"{nm.split(' / ')[0]}, {label.split(' (')[0]}")
    for (a_q, his, dy) in ((A_FURNITURE, HIS_FURNITURE, 9), (R.ONE, 3, -15)):
        fr = _emitted_fraction(T_SRC_GAME, a_q, his, R.ALPHA_FLOOR_HALF)[1]
        ax.plot([T_SRC_GAME], [fr], "o", color=BLUE, markersize=5, zorder=5)
        ax.annotate(f"{fr * 100:.0f} %", (T_SRC_GAME, fr), xytext=(-6, dy),
                    textcoords="offset points", color=BLUE, fontsize=8, ha="right")
    ax.set_xlim(0, 3036)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("cell temperature held by a deposit (game units)",
                  color=MUTED, fontsize=9)
    ax.set_ylabel("emitted excess / black-body excess  (= f)", color=MUTED, fontsize=9)
    ax.set_title("what a DRIVEN cell radiates", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper right")
    f_top = _emitted_fraction(R.T_TABLE_TOP_GAME, A_FURNITURE, HIS_FURNITURE,
                              R.ALPHA_FLOOR_HALF)[1]

    # ---- right: free cooling from 1263 game -------------------------------- #
    ax = axes[1]
    _style(ax)
    n = int(0.5 * R.TICK_HZ)
    t = np.arange(n + 1) / R.TICK_HZ
    an = [R.cool_exact_bath(float(T_SRC_GAME), float(s), 0.5, HIS_FURNITURE) for s in t]
    ax.plot(t, an, color=GREY, linewidth=4.0, alpha=0.55, solid_capstyle="round",
            label="analytic", zorder=1)
    tr_e, _ = R.cell_march(T_SRC_GAME << 16, R.E0, A_FURNITURE, HIS_FURNITURE, n,
                           fleck=False, clamp_enabled=False, trace=True)
    # drawn WIDE and pale under floor 0, which lands on it to the last count
    ax.plot(t, [v / 65536 for v in tr_e], color="#c9ccd1", linewidth=5.0,
            solid_capstyle="round", label="explicit (no Fleck)", zorder=2)
    for floor, label, colour in FLOORS:
        tr, _ = R.cell_march(T_SRC_GAME << 16, R.E0, A_FURNITURE, HIS_FURNITURE, n,
                             fleck=True, clamp_enabled=False, trace=True,
                             alpha_floor=floor)
        y = [v / 65536 for v in tr]
        ax.plot(t, y, color=colour, linewidth=1.9, label=label, zorder=3)
        err = (y[-1] - an[-1]) / an[-1] * 100
        ax.annotate(f"{err:+.2f}%", (t[-1], y[-1]), color=colour, fontsize=8,
                    xytext=(-4, 6 if err > 0 else -13), textcoords="offset points",
                    ha="right")
    ax.annotate("floor 0 lies exactly on the explicit curve:\n"
                "below g = 1 it sets alpha = 0, so f = 1", (0.20, 1150),
                color=MUTED, fontsize=7.5)
    ax.set_xlabel("seconds", color=MUTED, fontsize=9)
    ax.set_ylabel("cell temperature (game units)", color=MUTED, fontsize=9)
    ax.set_title(f"what FREE COOLING costs (from {T_SRC_GAME} game, 0.5 s)",
                 color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)

    fig.suptitle("The Fleck alpha floor, measured at both settings — a driven fire "
                 "radiates, or free cooling is exact (P0b, integer reference)",
                 color=INK, fontsize=12)
    fig.text(0.5, 0.055, "past each row's crossing (g = 1) the two floors are "
             f"IDENTICAL — at the table top both emit {f_top * 100:.2f} % of black body "
             "(furniture); the whole difference lives in the fire range",
             ha="center", color=MUTED, fontsize=8.5)
    fig.text(0.5, 0.013, "nothing was changed: the reference's default is and stays "
             "the ruled floor of 1/2 (design section 2.8); this is the measurement "
             "section 12 item 4 asks for",
             ha="center", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.085, 1, 0.92))
    out = HERE / "p0b_alpha_floor.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    print(__doc__.split("WHAT THE THREE MEASUREMENTS MEAN")[0].strip())
    table_a_driven()
    table_b_free_cooling()
    table_c_stability_under_floor_zero()
    figure()
