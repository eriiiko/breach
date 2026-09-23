"""Issue #70 — figures from the sealed-box bench's traces.

The measurement lives in ``tests/_sealedbox_bisect_bench.py`` (``--trace DIR``
writes one ``<variant>.npz`` per variant); this tool only DRAWS what it wrote,
so there is one instrument and one presentation layer. Nothing here touches
the simulation.

Expected layout (what the #70 runs produce)::

    DIR/trace18/   18 s  : nofire, nofire_nocond, nofire_nopush, and the
                           knockout variants (VARIANTS_70)
    DIR/trace180/  180 s : optional; used for the long run if trace600 is absent
    DIR/trace600/  600 s : nofire, nofire_nopush, nofire_nocond

Figures are written to DIR. Temperatures are ΔT from the 293 K ambient in
kelvin (one game degree is one kelvin at the shipped temperature scale);
air curves are energy-weighted over their region (ΣE/ΣN, the arc #54
books), wall curves heat-capacity-weighted (Σ cap·T / Σ cap).

Run:
    python tools/plot_sealedbox_70.py C:/tmp/breach_70
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# The Q16.16 unit comes from the canonical boundary module, never a literal
# (CLAUDE.md: Q16 boundary modules). gas_fixed is an import-light leaf.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from simulation.gas_fixed import FP_ONE_F as Q  # noqa: E402

TPS = 24
T_AMB_K = 293.0

# ---- palette: the dataviz reference instance, light mode ------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# one colour per ENTITY, fixed across every figure (slots 1..8 in order)
C = {"box_air": "#2a78d6", "arena_air": "#eb6834", "pen_air": "#1baf7a",
     "box_layer": "#eda100", "box_core": "#e87ba4", "halo_air": "#008300",
     "box_wall": "#4a3aa7", "arena_wall": "#e34948"}
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                 "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#0d366b", "#256abf", "#86b6ef", "#f0efec",
                      "#f2a3a2", "#e34948", "#8c1d1d"])


def _style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 9.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def load(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def seconds(d):
    return d["t"] / float(d["meta"].get("tps", TPS))


def region_dT(d, k):
    """ΔT of a region in K: gas ΣE_rel/ΣN, solid Σcap·T/Σcap (both raw /2^16)."""
    return d[k + "_E"].astype(np.float64) / d[k + "_C"].astype(np.float64) / Q


def with_t0(t, y, y0=0.0):
    """Prepend the t = 0 state (every trace starts from rest at ambient)."""
    return np.concatenate(([0.0], t)), np.concatenate(([y0], y))


def end_labels(ax, items, x_pad=0.012, min_gap=0.075):
    """Direct end-labels in ink beside each line's last point; labels that
    would collide are spread vertically and tied back with a thin leader."""
    if not items:
        return
    log = ax.get_yscale() == "log"
    fwd = (lambda v: np.log10(v)) if log else (lambda v: v)
    inv = (lambda v: 10.0 ** v) if log else (lambda v: v)
    y0, y1 = (fwd(v) for v in ax.get_ylim())
    x0, x1 = ax.get_xlim()
    gap = (y1 - y0) * min_gap
    items = sorted(((x, fwd(y), t) for x, y, t in items), key=lambda it: it[1])
    placed = []
    for x, y, text in items:
        yy = y if not placed else max(y, placed[-1][2] + gap)
        placed.append((x, y, yy, text))
    over = placed[-1][2] - (y1 - gap * 0.3)
    if over > 0:
        placed = [(x, y, yy - over, t) for x, y, yy, t in placed]
    xl = x1 + (x1 - x0) * x_pad
    for x, y, yy, text in placed:
        if abs(yy - y) > gap * 0.15:
            ax.plot([x, xl], [inv(y), inv(yy)], color=MUTED, lw=0.6, clip_on=False)
        ax.text(xl + (x1 - x0) * 0.006, inv(yy), text, color=INK2, fontsize=8.3,
                va="center", ha="left", clip_on=False)


def note(fig, text, y=0.005):
    """A muted footnote, wrapped to the figure's width (~7.8 pt glyphs)."""
    import textwrap
    width = max(60, int(fig.get_figwidth() * 72 / 4.6))
    fig.text(0.01, y, chr(10).join(textwrap.wrap(text, width)), color=MUTED,
             fontsize=7.8, ha="left", va="bottom", linespacing=1.35)


# ---------------------------------------------------------------------------
def fig_picture(d, out, title, controls=(), legend_b=None):
    """Fig 1/2: the picture — air, the wall layers, the walls."""
    t = seconds(d)
    fig, axs = plt.subplots(3, 1, figsize=(8.6, 9.2), sharex=True,
                            gridspec_kw={"hspace": 0.32, "left": 0.09,
                                         "right": 0.78, "top": 0.93,
                                         "bottom": 0.08})
    ax = axs[0]
    labs = []
    for k, lab, z in (("box_air", "box air", 3), ("arena_air", "arena air", 2),
                      ("pen_air", "sealed pen (built at load)", 3)):
        tt, y = with_t0(t, region_dT(d, k))
        ax.plot(tt, y, color=C[k], label=lab, zorder=z)
        labs.append((tt[-1], y[-1], lab))
    for cd, lab, col in controls:
        tt, y = with_t0(seconds(cd), region_dT(cd, "box_air"))
        ax.plot(tt, y, color=col, lw=1.2, label=lab, zorder=4)
        labs.append((tt[-1], y[-1], lab))
    ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
    ax.set_ylabel("ΔT from ambient (K)")
    ax.set_title("(a) Air, energy-weighted over each region (ΣE/ΣN)")
    end_labels(ax, labs)

    ax = axs[1]
    labs = []
    for k, lab in (("box_layer", "box air touching the walls (24 cells)"),
                   ("box_core", "box core air (25 cells)"),
                   ("halo_air", "outside air touching the walls (36 cells)")):
        tt, y = with_t0(t, region_dT(d, k))
        ax.plot(tt, y, color=C[k], label=lab)
        labs.append((tt[-1], y[-1], lab.split(" (")[0]))
    ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
    ax.set_ylabel("ΔT from ambient (K)")
    ax.set_title("(b) Where the box's heat goes in: the air layers on both "
                 "faces of the glass")
    ax.legend(**(legend_b or {"loc": "upper center", "bbox_to_anchor": (0.5, 0.80)}))
    end_labels(ax, labs)

    ax = axs[2]
    labs = []
    for k, lab in (("box_wall", "box walls (32 glass tiles)"),
                   ("arena_wall", "arena walls and furniture (69 tiles)")):
        tt, y = with_t0(t, region_dT(d, k) * 1e3)
        ax.plot(tt, y, color=C[k], label=lab)
        labs.append((tt[-1], y[-1], lab.split(" (")[0]))
    # the floordiv bias: exactly one raw count (2^-16 K) per tick, downward
    ax.plot(t, -t * TPS / Q * 1e3, color=MUTED, lw=1.0, zorder=1)
    labs.append((t[-1], -t[-1] * TPS / Q * 1e3, "−1 raw count per tick"))
    ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
    ax.set_ylabel("ΔT from ambient (mK)")
    ax.set_xlabel("time since the seal (s)")
    ax.set_title("(c) Walls, heat-capacity-weighted (note: millikelvin)")
    ax.legend(loc="lower left")
    end_labels(ax, labs)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="semibold",
                 color=INK)
    note(fig, "Ambient 293 K. Box: 7×7 air cells inside a 1-tile glass ring "
         "on open arena floor, sealed at t = 0 by GameMap.seal_tiles. "
         "Fireless; conduction, drag, k_leak and the radiation sweep live.")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_budget(d, out):
    """Fig 3: the box air's and the box walls' energy budgets, cumulative."""
    m = d["meta"]
    Jb, Jh = m["J_books"], m["J_heat"]
    t = seconds(d)
    box_JK = m["first"]["box_air"][1] * Q * Jb
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.6),
                            gridspec_kw={"wspace": 0.55, "left": 0.07,
                                         "right": 0.84, "top": 0.84,
                                         "bottom": 0.18})
    cum = lambda k, J: np.concatenate(([0.0], np.cumsum(d[k].astype(np.float64)) * J / 1e3))
    tt = np.concatenate(([0.0], t))

    ax = axs[0]
    meas = np.concatenate(([0.0], (d["box_air_E"] - m["first"]["box_air"][0]) * Jb / 1e3))
    cond = cum("q_wall_to_box", Jh)
    eos = cum("box_air_d_eos", Jb)
    rest = sum(cum(f"box_air_d_{p}", Jb) for p in ("py_pre", "comb_sky", "py_post"))
    ax.plot(tt, meas, color=INK, lw=2.0, label="measured ΔE")
    step = TPS
    ax.plot(tt[::step], cond[::step], ls="none", marker="o", ms=5.5, mfc=SURFACE,
            mec=C["box_air"], mew=1.4,
            label="heat in through its walls (replay)")
    ax.plot(tt, eos, color=C["arena_air"], label="EOS phase")
    ax.plot(tt, rest, color=C["pen_air"], label="all other phases")
    ax.set_title("(a) Box air: where its energy comes from")
    ax.set_ylabel("cumulative energy (kJ)")
    ax.set_xlabel("time since the seal (s)")
    end_labels(ax, [(tt[-1], meas[-1], f"measured {meas[-1]:+.1f} kJ"),
                    (tt[-1], eos[-1], f"EOS {eos[-1]:+.2f} kJ"),
                    (tt[-1], rest[-1] - 0.001, f"other {rest[-1]:+.2f} kJ")])
    ax.legend(loc="upper left", fontsize=7.8)

    ax = axs[1]
    measw = np.concatenate(([0.0], (d["box_wall_E"] - m["first"]["box_wall"][0]) * Jh / 1e3))
    inner = -cum("q_wall_to_box", Jh)
    outer = -cum("q_wall_to_halo", Jh)
    trunc = cum("wall_trunc", Jh)
    ax.plot(tt, measw, color=INK, lw=2.0, label="measured ΔH of the box walls")
    ax.plot(tt, inner, color=C["box_air"], label="out to the box air (inner faces)")
    ax.plot(tt, outer, color=C["halo_air"], label="out to the outside air (outer faces)")
    ax.plot(tt, trunc, color=C["box_wall"], label="destroyed by the solid endpoint floordiv")
    ax.set_title("(b) Box walls: where their energy goes")
    ax.set_ylabel("cumulative energy (kJ)")
    ax.set_xlabel("time since the seal (s)")
    end_labels(ax, [(tt[-1], measw[-1], f"measured {measw[-1]:+.0f} kJ"),
                    (tt[-1], inner[-1], f"to box air {inner[-1]:+.0f}"),
                    (tt[-1], outer[-1], f"to outside air {outer[-1]:+.0f}"),
                    (tt[-1], trunc[-1], f"destroyed {trunc[-1]:+.0f}")])
    ax.legend(loc="lower left", fontsize=7.8)
    fig.suptitle("Energy budget of the fireless sealed box, 0–18 s",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    b = m["bad"]
    exact = not any(v for k, v in b.items() if k != "ticks")
    note(fig, f"1 box-K = {box_JK / 1e3:.2f} kJ (the box air's heat capacity). "
         f"Radiation fold on the walls: identically 0 (rad_net = 0 on every cell, "
         f"every tick). Replay check over {b['ticks']} tails: "
         f"{'exact, cell by cell and against the engine counters' if exact else 'NOT exact'}.")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _crop(a, r0=48, r1=61, c0=22, c1=35):
    return a[r0:r1, c0:c1]


def fig_maps(d_on, d_off, out):
    """Fig 4: the mechanism on the grid — the push, tick 1, and 18 s."""
    mat = np.array(d_on["meta"]["material"])
    wall = _crop(mat) == 5            # MAT_GLASS
    N0 = _crop(d_on["snap0_N"]) / Q
    T1 = _crop(d_on["snap1_T"]) / Q
    end_on = d_on["meta"]["snap_ticks"][-1]
    end_off = d_off["meta"]["snap_ticks"][-1]
    Ton = _crop(d_on[f"snap{end_on}_T"]) / Q
    Toff = _crop(d_off[f"snap{end_off}_T"]) / Q
    fig, axs = plt.subplots(1, 4, figsize=(13.2, 4.1),
                            gridspec_kw={"wspace": 0.08, "left": 0.02,
                                         "right": 0.93, "top": 0.8,
                                         "bottom": 0.22})
    lim = 50.0
    panels = (
        (N0, SEQ_BLUE, (1.0, 2.0), "(a) N at t = 0, just after the seal"),
        (T1, DIVERGING, (-lim, lim), "(b) T after the first tick (1/24 s)"),
        (Toff, DIVERGING, (-lim, lim), f"(c) T at {end_off / TPS:.0f} s, conduction OFF"),
        (Ton, DIVERGING, (-lim, lim), f"(d) T at {end_on / TPS:.0f} s, conduction ON"),
    )
    ims = []
    for ax, (a, cmap, (vmin, vmax), title) in zip(axs, panels):
        a = np.ma.masked_where(wall, a)
        im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ims.append(im)
        for (r, c) in zip(*np.nonzero(wall)):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=INK2,
                                   edgecolor=SURFACE, lw=0.6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=9.5)
    cb1 = fig.colorbar(ims[0], ax=axs[0], orientation="horizontal", fraction=0.05, pad=0.03)
    cb1.set_label("N (atm per cell)", color=INK2)
    cb2 = fig.colorbar(ims[3], ax=list(axs[1:]), orientation="horizontal",
                       fraction=0.05, pad=0.03, aspect=60)
    cb2.set_label("air ΔT from ambient (K); dark squares are the glass ring", color=INK2)
    fig.suptitle("The mechanism on the grid: the seal pushes air in, the air "
                 "expands and cools against the walls, the walls warm it",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    note(fig, "Box footprint with a 2-cell margin (rows 48–60, cols 22–34). "
         "seal_tiles splits each ring tile's air between its two open neighbours, "
         "so both faces of the new wall start at 1.5 atm (2.0 in the inner corners).",
         y=0.01)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_knockouts(runs, out):
    """Fig 5: box ΔT at 18 s for each knockout."""
    rows = [
        ("nofire", "shipped: the seal pushes air both ways"),
        (None, None),
        ("nofire_nocond", "conduction off"),
        ("nofire_nodrag", "drag off (k_drag = 0)"),
        ("nofire_noleak", "k_leak = 0"),
        ("nofire_ambuniform", "per-cell ambient forced uniform"),
        ("nofire_norad", "radiation off (extinction planes zeroed)"),
        ("nofire_mg6", "pressure solve: 6 V-cycles instead of 2"),
        (None, None),
        ("nofire_push_box", "push kept on the box side only"),
        ("nofire_push_arena", "push kept on the arena side only"),
        ("nofire_nopush", "push undone (uniform N and T at t = 0)"),
        ("nofire_nopush_nocond", "push undone and conduction off"),
        (None, None),
        ("wall2_nofire", "2-tile glass wall (5×5 box)"),
        ("wall3_nofire", "3-tile glass wall (3×3 box)"),
    ]
    vals, labels, ys = [], [], []
    y = 0
    for key, lab in rows:
        if key is None:
            y += 0.6
            continue
        if key not in runs:
            continue
        d = runs[key]
        m = d["meta"]
        dT = (d["box_air_E"][-1] / d["box_air_C"][-1]
              - m["first"]["box_air"][0] / m["first"]["box_air"][1]) / Q
        vals.append(dT); labels.append(lab); ys.append(y)
        y += 1
    fig, ax = plt.subplots(figsize=(8.4, 0.36 * len(ys) + 1.6))
    fig.subplots_adjust(left=0.40, right=0.93, top=0.86, bottom=0.10)
    ax.barh(ys, vals, height=0.62, color=C["box_air"])
    for yy, v in zip(ys, vals):
        ax.text(max(v, 0) + 0.08, yy, f"{v:+.3f} K", va="center", ha="left",
                color=INK, fontsize=8.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(min(0, min(vals)) - 0.2, max(vals) * 1.22)
    ax.set_xlabel("box air ΔT after 18 s, ΣE/ΣN (K)")
    ax.set_axisbelow(True)
    fig.suptitle("Knock out one path at a time: only the seal's air push and "
                 "conduction move the result", x=0.01, ha="left", fontsize=12,
                 fontweight="semibold", color=INK)
    note(fig, "Every run fireless, 432 ticks. k_leak, the per-cell ambient and "
         "radiation give results bit-identical to the shipped run.")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_second_law(d_off, d_on, out):
    """Fig 6: the EOS's second-law violation in the adiabatic box."""
    m = d_off["meta"]
    gam = float(m["gamma"])
    box = d_off["mask_box_air"]
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.9),
                            gridspec_kw={"wspace": 0.72, "left": 0.07,
                                         "right": 0.86, "top": 0.84,
                                         "bottom": 0.16})
    ax = axs[0]
    nn = np.linspace(0.95, 2.05, 200)
    floor = T_AMB_K * (nn / 2.0) ** (gam - 1.0) - T_AMB_K
    ax.fill_between(nn, floor, -200, color="#f0efec", zorder=0, lw=0)
    ax.text(1.97, -68, "impossible for an" + chr(10) + "adiabatic box",
            color=INK2, fontsize=8, ha="right", va="center")
    iso = []
    for n0, lab in ((2.0, "isentrope, 2.0 atm"),
                    (1.5, "isentrope, 1.5 atm"),
                    (1.0, "isentrope, 1.0 atm")):
        yv = T_AMB_K * (nn / n0) ** (gam - 1.0) - T_AMB_K
        ax.plot(nn, yv, color=MUTED, lw=1.0, zorder=1)
        iso.append((nn[-1], yv[-1], lab))
    for tick, col, lab in ((0, INK, "t = 0"), (6, C["box_air"], "t = 0.25 s"),
                           (m["snap_ticks"][-1], C["arena_air"],
                            f"t = {m['snap_ticks'][-1] / TPS:.0f} s")):
        N = d_off[f"snap{tick}_N"][box] / Q
        T = d_off[f"snap{tick}_T"][box] / Q
        if tick == 0:
            ax.plot(N, T, ls="none", marker="o", ms=7, mfc=SURFACE, mec=col,
                    mew=1.3, label=lab, zorder=3)
        else:
            ax.plot(N, T, ls="none", marker="o", ms=5.5, color=col,
                    mec=SURFACE, mew=1.0, label=lab, zorder=3)
    ax.set_xlabel("N in the cell (atm)")
    ax.set_ylabel("air ΔT from ambient (K)")
    ax.set_title("(a) Box cells in the (N, T) plane, conduction OFF")
    ax.legend(loc="upper left")
    ax.set_xlim(0.95, 2.05)
    ax.set_ylim(-80, 105)
    end_labels(ax, iso)

    ax = axs[1]
    labs = []
    for d, col, lab in ((d_off, C["box_air"], "conduction off (adiabatic box)"),
                        (d_on, C["arena_air"], "conduction on (shipped)")):
        t = seconds(d)
        y = d["box_s_min_excess"]
        ax.plot(t, y, color=col, label=lab)
        labs.append((t[-1], y[-1], lab.split(" (")[0]))
    ax.axhline(0, color=INK2, lw=0.9)
    ax.text(0.02, 0.004, "second-law floor: an adiabatic box cannot go below 0",
            color=INK2, fontsize=8, transform=ax.get_yaxis_transform(), va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("time since the seal (s, log)")
    ax.set_ylabel("lowest box s − lowest s at t = 0  (c_v)")
    ax.set_title("(b) The box's lowest specific entropy")
    end_labels(ax, labs)
    fig.suptitle("The EOS cools the expanding wall layer below its isentrope",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    note(fig, f"s = ln T_abs − (γ−1) ln N, γ = {gam:.1f}. Isentropes: "
         "T = 293 K·(N/N0)^(γ−1). A cell below the lowest isentrope has lost "
         "entropy with no heat leaving it, which no adiabatic process allows.")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_motion(d, out):
    """Fig 7: air motion — where the 14–17 m/s is — and the pressure lag."""
    t = seconds(d)
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.5),
                            gridspec_kw={"wspace": 0.5, "left": 0.07,
                                         "right": 0.84, "top": 0.84,
                                         "bottom": 0.14})
    ax = axs[0]
    labs = []
    for key, col, lab in (("neck_u", C["arena_air"], "SW room's one-tile doorway (54, 3)"),
                          ("box_air_umax", C["box_air"], "sealed box (max over its air)"),
                          ("pen_air_umax", C["pen_air"], "sealed pen, built at load")):
        keep = d[key] > 0
        y = d[key][keep]
        ax.plot(t[keep], y, color=col, label=lab, lw=1.3)
        labs.append((t[keep][-1], y[-1], lab.split(" (")[0].split(",")[0]))
    ax.set_yscale("log")
    ax.set_ylim(0.05, 400)
    ax.set_ylabel("|u| max (m/s, log)")
    ax.set_xlabel("time since the seal (s)")
    ax.set_title("(a) Air speed: the 14–17 m/s is in a doorway, not the box")
    ax.legend(loc="upper left", fontsize=7.8)
    end_labels(ax, labs)

    ax = axs[1]
    k = t <= 5.0
    ax.plot(t[k], d["box_air_P"][k], color=C["box_air"], label="solved pressure P (atmosphere field)")
    ax.plot(t[k], d["box_air_pstar"][k], color=C["arena_air"],
            label="the box air's own state pressure N·T_abs/T_amb")
    ax.set_ylabel("box-mean pressure (atm)")
    ax.set_xlabel("time since the seal (s)")
    ax.set_title("(b) The pressure solve lags the sealed box's state")
    ax.legend(loc="lower right", fontsize=7.8)
    fig.suptitle("Air motion after the seal", x=0.01, ha="left", fontsize=12,
                 fontweight="semibold", color=INK)
    note(fig, "With the seal's push undone, every cell on the map stays at "
         "u = 0.00 for the whole run (not drawable on a log axis).")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_thickness(runs, out):
    """Fig 8: box ΔT(t) for 1-, 2- and 3-tile glass walls."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    fig.subplots_adjust(left=0.09, right=0.78, top=0.86, bottom=0.14)
    labs = []
    for key, col, lab in (("nofire", C["box_air"], "1-tile wall, 7×7 box, mean 1.29 atm"),
                          ("wall2_nofire", C["arena_air"], "2-tile wall, 5×5 box, mean 1.40 atm"),
                          ("wall3_nofire", C["pen_air"], "3-tile wall, 3×3 box, mean 1.67 atm")):
        if key not in runs:
            continue
        d = runs[key]
        tt, y = with_t0(seconds(d), region_dT(d, "box_air"))
        ax.plot(tt, y, color=col, label=lab)
        labs.append((tt[-1], y[-1], lab.split(",")[0]))
    ax.axhline(0, color=AXIS, lw=0.8, zorder=0)
    ax.set_xlabel("time since the seal (s)")
    ax.set_ylabel("box air ΔT, ΣE/ΣN (K)")
    ax.set_title("Wall thickness changes the box, not the heat path", fontsize=11)
    ax.legend(loc="upper left")
    end_labels(ax, labs)
    note(fig, "Every added layer is sealed outward, so the box always receives "
         "+0.5 atm per adjacent ring tile; a smaller box sits at a higher mean, "
         "and in the 3×3 box only the corners end up colder than the walls.")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/tmp/breach_70")
    _style()
    runs18 = {p.stem: load(p) for p in sorted((root / "trace18").glob("*.npz"))}
    runs180 = {p.stem: load(p) for p in sorted((root / "trace180").glob("*.npz"))}
    runs600 = {p.stem: load(p) for p in sorted((root / "trace600").glob("*.npz"))}
    made = []
    if "nofire" in runs18:
        ctl = []
        if "nofire_nopush" in runs18:
            ctl.append((runs18["nofire_nopush"], "box air, push undone", MUTED))
        fig_picture(runs18["nofire"], root / "fig1_picture_18s.png",
                    "Fireless sealed box: temperatures over the first 18 s", ctl)
        made.append("fig1_picture_18s.png")
        fig_budget(runs18["nofire"], root / "fig3_energy_budget_18s.png")
        made.append("fig3_energy_budget_18s.png")
        fig_motion(runs18["nofire"], root / "fig7_air_motion_18s.png")
        made.append("fig7_air_motion_18s.png")
    long_runs = runs600 if "nofire" in runs600 else runs180
    long = long_runs.get("nofire")
    if long is not None:
        ctl = []
        for key, lab, col in (("nofire_nopush", "box air, push undone", MUTED),
                              ("nofire_nocond", "box air, conduction off", INK2)):
            if key in long_runs:
                ctl.append((long_runs[key], lab, col))
        span = seconds(long)[-1]
        fig_picture(long, root / f"fig2_picture_{span:.0f}s.png",
                    f"The long run: rise, peak and decay over {span:.0f} s", ctl,
                    legend_b={"loc": "upper right"})
        made.append(f"fig2_picture_{span:.0f}s.png")
    if "nofire" in runs18 and "nofire_nocond" in runs18:
        fig_maps(runs18["nofire"], runs18["nofire_nocond"], root / "fig4_mechanism_maps.png")
        made.append("fig4_mechanism_maps.png")
        fig_second_law(runs18["nofire_nocond"], runs18["nofire"], root / "fig6_second_law.png")
        made.append("fig6_second_law.png")
    fig_knockouts(runs18, root / "fig5_knockouts_18s.png")
    made.append("fig5_knockouts_18s.png")
    fig_thickness(runs18, root / "fig8_wall_thickness_18s.png")
    made.append("fig8_wall_thickness_18s.png")
    for f in made:
        print(root / f)


if __name__ == "__main__":
    main()
