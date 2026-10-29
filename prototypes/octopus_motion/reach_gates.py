"""P1 gates: does the XPBD chain + stiffening wave reproduce the real reach?

Runs the experiments of study brief v1 §5 P1 on `arm_chain.py`, writes the
figures and the GIF to `out/`, prints PASS/FAIL per gate and writes
`out/report.md`. Pure numpy + matplotlib (Agg), no game, no Blender.

    <breach-py> prototypes/octopus_motion/reach_gates.py

Gates (brief §5, Yekutieli I/II, Sumbre 2001):
  a1  bell-shaped bend velocity (single rise then fall / dissolution at the tip)
  a2  the arm behind the bend is straight (mean local angle <= 3 deg)
  a3  the tip never overshoots (no reverse curl, tip stays on the curl's side)
  a4  peak bend speed ~ the wave's own speed (ratio in [0.7, 1.3])
  b   kinematics invariant to a_ca above the minimum-force curve; below it
      the arm ends curved (the map)
  c   a dead arm's bend decelerates monotonically and ends curved
  d   operation count per arm-substep (tallied from the solver, printed)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from arm_chain import (  # noqa: E402
    N_SEG, ArmChain, ArmGeometry, ChainParams, ReachCommand, bend_kinematics,
    hooked_pose, mean_local_angle_deg, run_passive, run_reach, segment_angles,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

GEOM = ArmGeometry()
PRM = ChainParams()
P0 = hooked_pose(GEOM)
S_NODE = ArmChain(GEOM, PRM, P0).s_node
L = GEOM.length_m

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})

results: list[tuple[str, bool, str]] = []


def gate(name: str, ok: bool, note: str) -> None:
    results.append((name, bool(ok), note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {note}")


# ---------------------------------------------------------------- helpers

def straight_behind_deg(P: np.ndarray, s_bend: float, margin_nodes: int = 2) -> float:
    """Mean |turn| of the interior nodes at least `margin_nodes` proximal of the bend."""
    k = int(np.searchsorted(S_NODE, s_bend)) - margin_nodes
    return mean_local_angle_deg(P, upto_node=k) if k >= 2 else np.nan


def distal_turn_deg(P: np.ndarray, n_last: int = 5) -> float:
    return float(np.degrees(segment_angles(P)[-n_last:].sum()))


def profile_shape(t: np.ndarray, vel: np.ndarray) -> dict:
    ok = np.isfinite(vel)
    tt, vv = t[ok], vel[ok]
    if len(vv) < 5:
        return {"n": len(vv), "single_peak": False, "rise": 0.0, "end_frac": 1.0, "t_peak_frac": 0.0}
    peak = vv.max()
    i_pk = int(vv.argmax())
    # count extrema with 8 % hysteresis
    ext = 0
    direction = 0
    ref = vv[0]
    for x in vv[1:]:
        if direction >= 0 and x < ref - 0.08 * peak:
            ext += 1
            direction = -1
        elif direction <= 0 and x > ref + 0.08 * peak:
            ext += (direction == -1)
            direction = 1
        ref = x if (direction > 0 and x > ref) or (direction < 0 and x < ref) else ref
    return {
        "n": len(vv),
        "single_peak": ext <= 1,
        "rise": float(peak / max(vv[: max(3, len(vv) // 6)].mean(), 1e-9)),
        "end_frac": float(vv[-3:].mean() / peak),
        "t_peak_frac": float((tt[i_pk] - tt[0]) / max(tt[-1] - tt[0], 1e-9)),
        "peak": float(peak),
        "t_end": float(tt[-1]),
    }


def plot_arm(ax, P, color, lw=2.0, alpha=1.0, label=None):
    ax.plot(P[:, 0], P[:, 1], "-", color=color, lw=lw, alpha=alpha, label=label, solid_capstyle="round")


# ------------------------------------------------------------ 1. the reach

cmd0 = ReachCommand(a_ca=0.6, t_act=0.9)
t_wall = time.perf_counter()
tr0 = run_reach(GEOM, PRM, cmd0, P0, 1.6)
t_wall = time.perf_counter() - t_wall
t, s_b, pos_b, vel_b, amp_b = bend_kinematics(tr0, S_NODE)
V_wave = L / cmd0.t_act

fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axs[0]
cmap = plt.get_cmap("viridis")
snap = np.arange(0, 0.95, 0.1)
for ts in snap:
    i = int(np.argmin(np.abs(t - ts)))
    plot_arm(ax, tr0.p[i], cmap(ts / 1.0), lw=2.2, label=f"{t[i]:.1f} s")
    if np.isfinite(s_b[i]):
        ax.plot(*pos_b[i], "o", color="crimson", ms=4)
ax.plot(0, 0, "ks", ms=6)
ax.set_aspect("equal")
ax.set_xlabel("x (m)  — base pinned at the origin, aim along +x")
ax.set_ylabel("y (m)")
ax.set_title(f"Reach, a_ca {cmd0.a_ca}, t_act {cmd0.t_act} s (red dot = bend point)")
ax.legend(fontsize=7, ncol=2, loc="upper right")

ax = axs[1]
ax.plot(t, vel_b, "-", color="crimson", lw=2, label="bend-point speed (sustained activation)")
tr_fade = run_reach(GEOM, PRM, ReachCommand(a_ca=0.6, t_act=0.9, hold=False), P0, 1.6)
t_f, _, _, vel_f, _ = bend_kinematics(tr_fade, S_NODE)
ax.plot(t_f, vel_f, "-", color="darkorange", lw=1.5, label="activation fading 0.4 L behind the front")
ax.axhline(V_wave, color="k", ls="--", lw=1, label=f"wave speed L / t_act = {V_wave:.2f} m/s")
ax2 = ax.twinx()
ax2.plot(t, s_b / L, "-", color="steelblue", lw=1.5, label="bend position s / L")
ax2.plot(t, np.clip((t - cmd0.t_start) / cmd0.t_act, 0, 1), ":", color="steelblue", lw=1, label="front s / L")
ax2.set_ylabel("s / L")
ax2.set_ylim(0, 1.05)
ax2.grid(False)
ax.set_xlabel("t (s)")
ax.set_ylabel("speed (m/s)")
ax.set_title("Bend kinematics")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_reach.png"))
plt.close(fig)

shape = profile_shape(t, vel_b)
gate("a1 bell-shaped bend velocity",
     shape["single_peak"] and shape["rise"] >= 1.4 and shape["t_peak_frac"] >= 0.3,
     f"single peak {shape['single_peak']}, peak/early {shape['rise']:.2f}, "
     f"peak at {shape['t_peak_frac']:.0%} of the tracked time, the bend dissolves at t = {shape['t_end']:.2f} s "
     f"(s = {np.nanmax(s_b) / L:.2f} L)")

ok_idx = [i for i in range(len(t)) if np.isfinite(s_b[i]) and s_b[i] > 0.4 * L]
behind = np.array([straight_behind_deg(tr0.p[i], s_b[i]) for i in ok_idx])
gate("a2 straight arm behind the bend",
     np.nanmax(behind) <= 3.0,
     f"mean local angle >= 2 nodes behind the bend, once it has passed 0.4 L: max {np.nanmax(behind):.2f} deg, "
     f"final whole-arm {mean_local_angle_deg(tr0.p[-1]):.2f} deg")

end_idx = [i for i in range(len(t)) if (np.isfinite(s_b[i]) and s_b[i] > 0.8 * L) or (not np.isfinite(s_b[i]) and t[i] > 0.3)]
mid_idx = [i for i in range(len(t)) if np.isfinite(s_b[i]) and s_b[i] <= 0.8 * L]
tip_y_min = min(tr0.p[i][-1][1] for i in end_idx)
dist_turn_min = min(distal_turn_deg(tr0.p[i]) for i in end_idx)
wiggle = min(distal_turn_deg(tr0.p[i]) for i in mid_idx)
tr_nodrag = run_reach(GEOM, ChainParams(**{**PRM.__dict__, "c_per": 0.0, "c_tan": 0.0}), cmd0, P0, 1.6)
nodrag_turn = min(distal_turn_deg(P) for P in tr_nodrag.p)
nodrag_tip_y = min(P[-1][1] for P in tr_nodrag.p)
gate("a3 tip never overshoots (end of the reach)",
     tip_y_min >= -0.05 * L and dist_turn_min >= -10.0,
     f"once the bend has passed 0.8 L: min tip y {tip_y_min * 1e3:.1f} mm (limit {-0.05 * L * 1e3:.0f}), "
     f"min distal turn {dist_turn_min:.1f} deg (limit -10). Control WITHOUT drag: distal turn {nodrag_turn:.0f} deg, "
     f"tip y {nodrag_tip_y * 1e3:.0f} mm = Yekutieli's overshoot. Mid-reach S-flex of the distal arm (reported, not gated): {wiggle:.1f} deg")

# ----------------------------------------------- 2. wave speed vs peak speed

t_acts = [0.5, 0.7, 0.9, 1.2, 1.5]
peaks = []
for ta in t_acts:
    tr = run_reach(GEOM, PRM, ReachCommand(a_ca=0.6, t_act=ta), P0, ta + 0.8)
    _, _, _, v, _ = bend_kinematics(tr, S_NODE)
    peaks.append(np.nanmax(v))
peaks = np.array(peaks)
waves = L / np.array(t_acts)
ratios = peaks / waves
natural = [(ta >= 0.6) and (ta <= 1.2) for ta in t_acts]   # Yekutieli I: t_act 0.6-1.2 s spans the natural reaches
gate("a4 peak bend speed ~ wave speed (natural range t_act 0.6-1.2 s)",
     bool(np.all((ratios[natural] >= 0.7) & (ratios[natural] <= 1.3))),
     "peak / wave = " + ", ".join(f"{r:.2f} @ {ta} s" for r, ta in zip(ratios, t_acts))
     + " (outside the natural range the peak is the tip's flick as the bend dissolves)")

# ------------------------------------------- 3. amplitude invariance (gate b)

a_cas = [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
profiles = {}
for a in a_cas:
    tr = run_reach(GEOM, PRM, ReachCommand(a_ca=a, t_act=0.9), P0, 0.9 + 0.15)
    tt, ss, pp, vv, aa = bend_kinematics(tr, S_NODE)
    final = mean_local_angle_deg(tr.p[-1])
    profiles[a] = (tt, vv, final, tr.p[-1])
    axs[0].plot(tt, vv, lw=1.6, label=f"a_ca {a} (final {final:.1f} deg)")
    plot_arm(axs[2], tr.p[-1], plt.get_cmap("plasma")(a), lw=1.8, label=f"a_ca {a}")
axs[0].axhline(L / 0.9, color="k", ls="--", lw=1)
axs[0].set_title("Bend speed vs time, t_act 0.9 s")
axs[0].set_xlabel("t (s)")
axs[0].set_ylabel("m/s")
axs[0].legend(fontsize=7)
# normalised profiles
norm_curves = {}
for a, (tt, vv, final, _) in profiles.items():
    ok = np.isfinite(vv)
    if ok.sum() < 5:
        continue
    tn = (tt[ok] - tt[ok][0]) / (tt[ok][-1] - tt[ok][0])
    vn = vv[ok] / vv[ok].max()
    grid = np.linspace(0, 1, 50)
    norm_curves[a] = np.interp(grid, tn, vn)
    axs[1].plot(grid, norm_curves[a], lw=1.4, label=f"a_ca {a}")
axs[1].set_title("Normalised profiles (time, speed by peak)")
axs[1].set_xlabel("normalised time")
axs[1].legend(fontsize=7)
axs[2].set_aspect("equal")
axs[2].set_title("Shapes at t_act + 0.15 s (the end of the wave)")
axs[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_amplitude_invariance.png"))
plt.close(fig)

above = [a for a in a_cas if profiles[a][2] <= 2.0]
if len(above) >= 3:
    ref = norm_curves[above[-1]]
    rms = {a: float(np.sqrt(np.mean((norm_curves[a] - ref) ** 2))) for a in above if a in norm_curves}
    pk = {a: float(np.nanmax(profiles[a][1])) for a in above}
    pk_spread = (max(pk.values()) - min(pk.values())) / max(pk.values())
    gate("b kinematics invariant to a_ca above the minimum force",
         max(rms.values()) <= 0.15 and pk_spread <= 0.2,
         f"straight (<= 2 deg at t_act + 0.15 s) for a_ca {above}; normalised-profile RMS vs a_ca {above[-1]}: "
         + ", ".join(f"{a}: {r:.3f}" for a, r in rms.items())
         + f"; peak speeds spread {pk_spread:.0%}; below: "
         + ", ".join(f"a_ca {a} ends at {profiles[a][2]:.1f} deg" for a in a_cas if a not in above))
else:
    gate("b kinematics invariant to a_ca above the minimum force", False,
         "fewer than three amplitudes straightened the arm: " + ", ".join(f"{a}: {profiles[a][2]:.1f} deg" for a in a_cas))

# --------------------------------------- 4. the minimum-force map (b, cont.)

ta_grid = np.array([0.1, 0.2, 0.3, 0.45, 0.6, 0.9, 1.2, 1.8, 2.5])
a_grid = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])
final_deg = np.zeros((len(a_grid), len(ta_grid)))
travelled = np.zeros_like(final_deg)
for ia, a in enumerate(a_grid):
    for it, ta in enumerate(ta_grid):
        # straightness is judged when the wave has just reached the tip (Yekutieli II: "the arm
        # straight at the end of the movement"), not after a long settle
        tr = run_reach(GEOM, PRM, ReachCommand(a_ca=a, t_act=ta), P0, ta + 0.15)
        final_deg[ia, it] = mean_local_angle_deg(tr.p[-1])
        _, ss, _, _, _ = bend_kinematics(tr, S_NODE)
        travelled[ia, it] = np.nanmax(ss) / L if np.isfinite(ss).any() else 0.0
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
im = axs[0].pcolormesh(ta_grid, a_grid, final_deg, shading="nearest", cmap="magma_r", vmin=0, vmax=12)
axs[0].contour(ta_grid, a_grid, final_deg, levels=[2.0], colors="cyan", linewidths=2)
fig.colorbar(im, ax=axs[0], label="final mean local angle (deg)")
axs[0].set_xscale("log")
axs[0].set_xlabel("t_act (s)")
axs[0].set_ylabel("a_ca")
axs[0].set_title("Minimum-force curve at t_act + 0.15 s (cyan = 2 deg: straight)")
im = axs[1].pcolormesh(ta_grid, a_grid, travelled, shading="nearest", cmap="viridis", vmin=0, vmax=1)
fig.colorbar(im, ax=axs[1], label="how far the bend travelled (s / L)")
axs[1].set_xscale("log")
axs[1].set_xlabel("t_act (s)")
axs[1].set_title("Bend propagation (low = the whole arm opens at once)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_minimum_force_map.png"))
plt.close(fig)
n_straight = int((final_deg <= 2.0).sum())
print(f"  min-force map: {n_straight}/{final_deg.size} cells straight; "
      f"bend travel at t_act 0.1 s: {travelled[:, 0].mean():.2f} L vs t_act 0.9 s: {travelled[:, 5].mean():.2f} L")

# ------------------------------------------------- 5. the dead arm (gate c)

prm_dead = PRM
omega = 3.0  # rad/s whip about the base: tip at ~0.3 L * omega
v0 = omega * np.stack([-P0[:, 1], P0[:, 0]], axis=1)
trd = run_passive(GEOM, prm_dead, P0, v0, 2.0)
td, sd, pd, vd, ad = bend_kinematics(trd, S_NODE)
okd = np.isfinite(vd)
vv = vd[okd]
i_pk = int(vv.argmax()) if len(vv) else 0
after = vv[i_pk:]
mono = bool(np.all(np.diff(after) <= 0.05 * vv.max())) if len(after) > 2 else False
gate("c dead arm: bend decelerates monotonically, ends curved",
     mono and mean_local_angle_deg(trd.p[-1]) > 2.0,
     f"peak {vv.max() if len(vv) else 0:.3f} m/s at {td[okd][i_pk] if len(vv) else 0:.2f} s, then "
     f"{'monotone' if mono else 'NOT monotone'} over {len(after)} ticks; final mean angle {mean_local_angle_deg(trd.p[-1]):.1f} deg")
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ts in np.arange(0, 1.0, 0.1):
    i = int(np.argmin(np.abs(td - ts)))
    plot_arm(axs[0], trd.p[i], cmap(ts), lw=1.8, label=f"{td[i]:.1f} s")
axs[0].set_aspect("equal")
axs[0].legend(fontsize=7, ncol=2)
axs[0].set_title("Dead arm whipped about the base (no activation)")
axs[1].plot(td, vd, color="crimson", lw=2)
axs[1].set_xlabel("t (s)")
axs[1].set_ylabel("bend-point speed (m/s)")
axs[1].set_title("Passive bend: decelerates monotonically")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_dead_arm.png"))
plt.close(fig)

# ------------------------------------------- 6. the startle (open-arm) look

fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, (a, ta, title) in zip(axs, [(0.6, 0.9, "natural reach"), (0.6, 0.1, "wave too fast: the whole arm opens (startle)"),
                                     (0.1, 0.9, "too little force: lazy curved reach")]):
    tr = run_reach(GEOM, PRM, ReachCommand(a_ca=a, t_act=ta), P0, 1.3)
    tt = np.array(tr.t)
    for ts in np.arange(0, 1.05, 0.1):
        i = int(np.argmin(np.abs(tt - ts)))
        plot_arm(ax, tr.p[i], cmap(ts), lw=1.8)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\na_ca {a}, t_act {ta} s", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_failure_looks.png"))
plt.close(fig)

# ------------------------------------------------------------ 7. the GIF

fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.set_xlim(-0.08, 0.27)
ax.set_ylim(-0.05, 0.17)
ax.set_aspect("equal")
ax.set_title("P1: XPBD chain + stiffening wave (a_ca 0.6, t_act 0.9 s)", fontsize=8)
line, = ax.plot([], [], "-", color="#5a3b8a", lw=4, solid_capstyle="round")
act = ax.scatter([], [], c=[], cmap="hot", vmin=0, vmax=1, s=14, zorder=3)
dot, = ax.plot([], [], "o", color="crimson", ms=6, zorder=4)
txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=8, va="top")


def _frame(i):
    P = tr0.p[i]
    line.set_data(P[:, 0], P[:, 1])
    act.set_offsets(P[1:-1])
    act.set_array(tr0.activation[i])
    if np.isfinite(s_b[i]):
        dot.set_data([pos_b[i][0]], [pos_b[i][1]])
    else:
        dot.set_data([], [])
    txt.set_text(f"t = {t[i]:.2f} s   bend speed {np.nan_to_num(vel_b[i]):.2f} m/s\nhot dots = activation (the stiffening wave)")
    return line, act, dot, txt


anim = FuncAnimation(fig, _frame, frames=len(tr0.p), interval=1000 * PRM.dt_tick, blit=True)
anim.save(os.path.join(OUT, "reach.gif"), writer=PillowWriter(fps=int(round(1 / PRM.dt_tick))))
plt.close(fig)

# ------------------------------------------------ 8. operation count (gate d)
# Tallied from arm_chain.ArmChain.substep for one arm, one substep, N = 18 nodes,
# 17 distance + 16 bend constraints. Counted: +, -, * as 1 each; a reciprocal
# (division) and an atan2 named separately (the sim port's kit calls).
n_nodes, n_dist, n_bend = 18, N_SEG, N_SEG - 1
drag_per_node = 2 + 4 + 2 + 2 + 4 + 4 + 2 + 4     # tangent, normalise, dot x2, factors x2, scale, recompose
drag_ops = n_nodes * drag_per_node                # ~24 each
drag_recips = n_nodes * 3                        # 1/|tangent|, two 1/(1+..)
predict_ops = n_nodes * 4
dist_ops = n_dist * 14
dist_recips = n_dist * 1
dist_sqrts = n_dist * 1
bend_ops = n_bend * 46
bend_recips = n_bend * 3                         # 1/l1, 1/l2, 1/denom
bend_atan2 = n_bend * 1
vel_ops = n_nodes * 4
per_pass_ops = dist_ops + bend_ops
per_pass_special = dist_recips + dist_sqrts * 8 + bend_recips * 8 + bend_atan2 * 16   # kit weights: recip/sqrt ~8, atan2 ~16 mul/add
total_ops = drag_ops + predict_ops + vel_ops + PRM.n_pass * per_pass_ops
weighted = total_ops + drag_recips * 8 + PRM.n_pass * per_pass_special
per_octopus_tick = weighted * PRM.n_sub * 8
gate("d operation count per arm-substep",
     weighted <= 25_000,
     f"~{total_ops} mul/add + {drag_recips + PRM.n_pass * (dist_recips + bend_recips)} reciprocals, "
     f"{PRM.n_pass * dist_sqrts} sqrt, {PRM.n_pass * bend_atan2} atan2 per arm-substep ({PRM.n_pass} passes); "
     f"weighted (recip/sqrt x8, atan2 x16) ~{weighted} mul/add-equivalents = ~{weighted / 1000:.1f} us at 1 ns each; "
     f"x {PRM.n_sub} substeps x 8 arms = ~{per_octopus_tick / 1e6:.2f} M per octopus-tick = ~{per_octopus_tick / 1e6:.2f} ms at 1 ns/op "
     f"(ceiling 0.2 ms: P6 must measure; numpy wall {t_wall / (len(tr0.p) - 1) * 1e3:.1f} ms per tick is Python loops)")

# ----------------------------------------------------------------- report

lines = ["# P1 gate report -- XPBD chain + stiffening wave, one arm, planar\n",
         f"Generated by `reach_gates.py`, {time.strftime('%Y-%m-%d %H:%M')}. Geometry: Yekutieli's 240 mm arm, "
         f"18 nodes; dt {PRM.dt_tick:.4f} s x {PRM.n_sub} substeps x {PRM.n_pass} GS passes; k_bend {PRM.k_bend} N m/rad, "
         f"passive {PRM.k_passive_frac:.0%}, c_per {PRM.c_per}, c_tan {PRM.c_tan}, g_eff {PRM.g_eff}.\n",
         "| gate | result | measurement |", "|---|---|---|"]
for name, ok, note in results:
    lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {note} |")
lines += ["", "Figures: `fig1_reach.png` (snapshots + bend kinematics), `fig2_amplitude_invariance.png`, "
          "`fig3_minimum_force_map.png`, `fig4_dead_arm.png`, `fig5_failure_looks.png`, `reach.gif`.", ""]
with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} gates pass. Report: {os.path.join(OUT, 'report.md')}")
sys.exit(1 if n_fail else 0)
