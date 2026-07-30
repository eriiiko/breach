#!/usr/bin/env python
"""Plot the fire_tune_loop time-series (Fable, 2026-07-25).

Three stacked panels, shared time axis:
    1. fire intensity I           (+ the peak target band)
    2. crate-tile temperature     (+ the flame-T realism band, plotted in K)
    3. room-mean O2 mole fraction (+ ambient 0.21 / extinction 0.13 lines)

The shaded bands come from the CALLER: fire_tune_loop.py passes its own `T`
target dict, so the shading and the scorecard's PASS/MISS verdicts can never
disagree. Standalone runs fall back to DEFAULT_TARGETS below.

Standalone:   python tools/fire_tune_plot.py [csv_path] [--show]
From the loop: imported and called automatically after each sim (and the loop
survives this file failing — the scorecard is the deliverable, not the plot).
Always writes <csv_stem>.png next to the CSV.

Columns read (written by tools/fire_timing_harness.write_timeseries_csv):
    t_min, I, T_game, O2room_X      — verified present 2026-07-30.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

INK = "#374151"        # primary text
MUTED = "#9ca3af"      # secondary text / grid
C_I = "#d97706"        # intensity — amber
C_T = "#dc2626"        # temperature — red
C_X = "#0d9488"        # oxygen — teal
BAND = "#6b7280"       # neutral target shading


def load(csv_path):
    rows, hdr = [], None
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "t_s":
                hdr = row
                continue
            rows.append({h: float(x) for h, x in zip(hdr, row)})
    if not rows:
        raise SystemExit(f"no data rows in {csv_path}")
    return rows


# Fallback target bands, used when the caller passes none (standalone runs).
# fire_tune_loop.py passes its own `T` dict so the shading here can never drift
# out of step with the scorecard's verdicts.
DEFAULT_TARGETS = {"peak_lo": 0.40, "peak_aim": 0.50, "peak_hi": 0.60,
                   "flameT_lo": 400.0, "flameT_hi": 500.0}

# The CSV columns this plot reads. fire_timing_harness.write_timeseries_csv is
# the writer; checked up front so a column rename is a clear message, not a
# KeyError halfway through drawing.
NEEDED = ("t_min", "I", "T_game", "O2room_X")


def make_plot(csv_path, show=False, targets=None):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tg = dict(DEFAULT_TARGETS)
    tg.update({k: v for k, v in (targets or {}).items() if k in DEFAULT_TARGETS})

    s = load(csv_path)
    missing = [c for c in NEEDED if c not in s[0]]
    if missing:
        raise SystemExit(
            f"{csv_path}: harness CSV is missing {', '.join(missing)} — the "
            f"columns this plot reads are {', '.join(NEEDED)}")
    t = [r["t_min"] for r in s]
    png = Path(csv_path).with_suffix(".png")

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    fig.suptitle("fire_tune_loop — last run", color=INK, fontsize=12)

    for ax in (ax1, ax2, ax3):
        ax.grid(True, color=MUTED, alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK, labelsize=9)

    # 1 — intensity
    ax1.axhspan(tg["peak_lo"], tg["peak_hi"], color=BAND, alpha=0.08)
    ax1.axhline(tg["peak_aim"], color=BAND, alpha=0.35, lw=0.8, ls="--")
    ax1.plot(t, [r["I"] for r in s], color=C_I, lw=2)
    ax1.set_ylabel("intensity I", color=INK)
    ax1.set_ylim(0, 1)

    # 2 — temperature in KELVIN (sim stores game units; K = 293 + 2*T_game).
    # Target band 400-500 game == 1093-1293 K (a real wood flame).
    ax2.axhspan(293 + 2 * tg["flameT_lo"], 293 + 2 * tg["flameT_hi"],
                color=BAND, alpha=0.08)
    ax2.axhline(293, color=BAND, alpha=0.35, lw=0.8, ls="--")
    ax2.plot(t, [293 + 2 * r["T_game"] for r in s], color=C_T, lw=2)
    ax2.set_ylabel("crate T (K)", color=INK)
    ax2.annotate("ambient 293 K", xy=(t[-1], 293), fontsize=8,
                 color=MUTED, ha="right", va="bottom")

    # 3 — room-mean O2 mole fraction
    ax3.axhline(0.21, color=BAND, alpha=0.5, lw=0.8, ls="--")
    ax3.axhline(0.13, color=BAND, alpha=0.5, lw=0.8, ls=":")
    ax3.plot(t, [r["O2room_X"] for r in s], color=C_X, lw=2)
    ax3.set_ylabel("room-mean O₂\nmole fraction X", color=INK)
    ax3.set_ylim(0, 0.25)
    ax3.set_xlabel("time (min)", color=INK)
    ax3.annotate("ambient 0.21", xy=(t[-1], 0.21), fontsize=8,
                 color=MUTED, ha="right", va="bottom")
    ax3.annotate("extinction 0.13", xy=(t[-1], 0.13), fontsize=8,
                 color=MUTED, ha="right", va="bottom")

    fig.savefig(png, dpi=130)
    print(f"[plot] {png}")
    if show:
        plt.show()
    plt.close(fig)
    return png


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--show"]
    default = (Path(__file__).resolve().parent.parent
               / "_fire_tuning_artifacts" / "tune_loop_last.csv")
    make_plot(Path(args[0]) if args else default, show="--show" in sys.argv)
