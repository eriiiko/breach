"""P3 gates: swimming (study brief v1 Âsection 5 P3).

    <breach-py> prototypes/octopus_motion/swim_gates.py      (~3 min, writes out/)

Gates:
  1  the umbrella stroke travels from drag alone: steady speed > 0 with the
     real (quadratic, anisotropic) drag, and the CONTROL with linear
     isotropic drag on arms and body nets < 15 % of it
  2  swimming is >= 5x the crawl's mean speed (0.05 m/s from P2): measured
     for the umbrella alone, the jet alone and the escape swim (a jet on
     every close); the gate is on the escape swim, the others are reported
  3  a jet carries the body >= 3 body lengths (0.1 m mantle) and the speed
     decays to < 10 % of its peak within 5 s
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
from matplotlib.patches import Circle  # noqa: E402

from swim_sim import SwimParams, make_swimmer  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
results: list[tuple[str, bool, str]] = []
CRAWL_SPEED = 0.05      # m/s, P2's mean body speed
BODY_LEN = 0.1          # m, the mantle (2 x crown radius)


def gate(name: str, ok, note: str) -> None:
    results.append((name, bool(ok), note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {note}", flush=True)


def kin(s):
    T = np.array(s.log_t)
    X = np.array(s.log_x)
    V = np.gradient(X, T, axis=0)
    return T, X, V, np.linalg.norm(V, axis=1)


# ------------------------------------------------ 1. umbrella, real vs control
t0 = time.perf_counter()
sw_q = make_swimmer(linear=False)
sw_q.start_umbrella()
sw_q.run(10.0)
sw_l = make_swimmer(linear=True)
sw_l.start_umbrella()
sw_l.run(10.0)
Tq, Xq, Vq, spq = kin(sw_q)
Tl, Xl, Vl, spl = kin(sw_l)
steady = Tq > 5.0
v_umb = float(Vq[steady, 0].mean())
v_ctl = float(Vl[steady, 0].mean())
period = sw_q.swim.open_s + sw_q.swim.close_s
gate("1 the umbrella travels from drag alone",
     v_umb > 0.0 and abs(v_ctl) < 0.15 * v_umb,
     f"steady speed {v_umb:.3f} m/s ({v_umb / BODY_LEN:.2f} mantle lengths/s, {v_umb * period * 1e3:.0f} mm per stroke) with the real drag; "
     f"the linear isotropic control {v_ctl:.3f} m/s ({abs(v_ctl) / max(v_umb, 1e-9):.0%}); open:close {sw_q.swim.open_s}:{sw_q.swim.close_s} s, web factor {sw_q.swim.web_factor}"
     f" ({time.perf_counter() - t0:.0f} s)")

# ------------------------------------------------ 3. the jet
sw_j = make_swimmer()
sw_j.start_jet()
sw_j.run(6.0)
Tj, Xj, Vj, spj = kin(sw_j)
i_pk = int(spj.argmax())
after = np.where((Tj > Tj[i_pk]) & (spj < 0.1 * spj[i_pk]))[0]
t_decay = float(Tj[after[0]] - Tj[i_pk]) if len(after) else np.inf
dist_jet = float(Xj[-1, 0] - Xj[0, 0])
gate("3 a jet carries a few body lengths, then decays",
     dist_jet >= 3 * BODY_LEN and t_decay <= 5.0,
     f"peak {spj[i_pk]:.2f} m/s, {dist_jet:.2f} m = {dist_jet / BODY_LEN:.1f} mantle lengths, speed < 10 % of peak after {t_decay:.1f} s "
     f"(jet dv {sw_j.swim.jet_dv} m/s over {sw_j.swim.jet_s} s, body c_d {sw_j.swim.body_c_drag})")

# ------------------------------------------------ 2. the escape swim: a jet on every close
sw_e = make_swimmer()
sw_e.start_umbrella()
jets = 0
for _ in range(int(round(10.0 / sw_e.chain.dt_tick))):
    if sw_e.mode == "close" and sw_e.t_mode == 0.0:
        pass
    sw_e.tick()
    # fire the jet with the close stroke: at the moment the stroke turns from open to close
    if sw_e.mode == "close" and sw_e.t_mode <= sw_e.chain.dt_tick * 1.01 and sw_e.jet_force == 0.0:
        h = np.array([np.cos(sw_e.body.theta), np.sin(sw_e.body.theta)])
        sw_e.body.v += sw_e.swim.jet_dv * 0.5 * h       # half a full jet's impulse per stroke
        jets += 1
Te, Xe, Ve, spe = kin(sw_e)
v_esc = float(Ve[Te > 5.0, 0].mean())
# the real escape (Wells 1987; motor-biology digest): a TRAIN of jets, arms bundled behind, the mantle
# refilling between -- one jet per second
sw_t = make_swimmer()
sw_t.start_jet()
JET_PERIOD = 1.0
next_jet = JET_PERIOD
for _ in range(int(round(10.0 / sw_t.chain.dt_tick))):
    sw_t.tick()
    if sw_t.time >= next_jet:
        sw_t.start_jet()
        next_jet += JET_PERIOD
Tt, Xt, Vt, spt = kin(sw_t)
v_train = float(Vt[Tt > 5.0, 0].mean())
gate("2 swimming >= 5x the crawl",
     v_train >= 5 * CRAWL_SPEED,
     f"jet train (one jet per second, arms bundled) {v_train:.3f} m/s = {v_train / CRAWL_SPEED:.1f}x crawl; "
     f"umbrella alone {v_umb:.3f} m/s = {v_umb / CRAWL_SPEED:.1f}x (the cruise, slow in the animal too); "
     f"umbrella + half-jet on each close ({jets} in 10 s) {v_esc:.3f} m/s = {v_esc / CRAWL_SPEED:.1f}x (the open umbrella brakes the jet); "
     f"a single jet peaks at {spj[i_pk]:.2f} m/s = {spj[i_pk] / CRAWL_SPEED:.0f}x. Gameplay speeds stay Erik's dial (brief section 6)")

# ------------------------------------------------ figures
fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
ax = axs[0, 0]
ax.plot(Tq, Xq[:, 0], lw=1.8, label=f"real drag: {v_umb:.3f} m/s steady")
ax.plot(Tl, Xl[:, 0], lw=1.8, label=f"linear isotropic control: {v_ctl:.3f} m/s")
ax.plot(Tq, np.degrees(sw_q.log_phi) / 400.0, lw=0.8, color="gray", label="umbrella angle / 400")
ax.set_xlabel("t (s)")
ax.set_ylabel("x (m)")
ax.set_title("Umbrella stroke: travel per stroke (slow open, fast close)")
ax.legend(fontsize=7)
ax = axs[0, 1]
ax.plot(Tq, Vq[:, 0], lw=1, label="real drag")
ax.plot(Tl, Vl[:, 0], lw=1, label="control", alpha=0.7)
ax.set_xlabel("t (s)")
ax.set_ylabel("body vx (m/s)")
ax.set_title("Body velocity: the fast close pushes more than the slow open pulls back")
ax.legend(fontsize=7)
ax = axs[1, 0]
ax.plot(Tj, spj, lw=1.5, color="C3", label="jet alone")
ax.plot(Te, spe, lw=1, color="C2", alpha=0.8, label="umbrella + half-jets")
ax.plot(Tt, spt, lw=1, color="C0", alpha=0.8, label=f"jet train: {v_train:.2f} m/s mean")
ax.axhline(5 * CRAWL_SPEED, color="k", ls="--", lw=1, label="5x crawl")
ax.set_xlabel("t (s)")
ax.set_ylabel("m/s")
ax.set_title(f"Jet: {dist_jet / BODY_LEN:.1f} mantle lengths, decays in {t_decay:.1f} s")
ax.legend(fontsize=7)
ax = axs[1, 1]
P = np.array(sw_q.log_p)
cmap = plt.get_cmap("viridis")
i0 = int(np.argmin(np.abs(Tq - 7.0)))
n_per = int(round(period / sw_q.chain.dt_tick))
for k in range(0, n_per, max(1, n_per // 7)):
    i = i0 + k
    col = cmap(k / n_per)
    for a in range(2):
        ax.plot(P[i, a, :, 0] - Xq[i, 0], P[i, a, :, 1], "-", color=col, lw=1.5)
    ax.add_patch(Circle((0, 0), sw_q.body.radius, color=col, fill=False, lw=1))
ax.set_aspect("equal")
ax.set_title("One stroke in the body frame (side view; swims to +x, mantle first)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_swim.png"))
plt.close(fig)

# GIF: umbrella then a jet, world frame
sw_g = make_swimmer()
sw_g.start_umbrella()
sw_g.run(4.2)
sw_g.start_jet()
sw_g.run(3.0)
Tg, Xg, Vg, spg = kin(sw_g)
Pg = np.array(sw_g.log_p)
fig, ax = plt.subplots(figsize=(7.0, 3.6))
ax.set_aspect("equal")
ax.set_xlim(-0.35, 1.0)
ax.set_ylim(-0.35, 0.35)
lines = [ax.plot([], [], "-", color="#5a3b8a", lw=3, solid_capstyle="round")[0] for _ in range(2)]
body = Circle((0, 0), sw_g.body.radius, color="#5a3b8a", alpha=0.7)
ax.add_patch(body)
txt = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=8, va="top")
ax.set_title("P3: umbrella strokes, then a jet (side view, mantle first)", fontsize=8)


def _frame(i):
    for a in range(2):
        lines[a].set_data(Pg[i, a, :, 0], Pg[i, a, :, 1])
    body.center = (Xg[i, 0], Xg[i, 1])
    txt.set_text(f"t = {Tg[i]:.2f} s   {sw_g.log_phase[i]}   v = {spg[i]:.2f} m/s")
    return lines + [body, txt]


anim = FuncAnimation(fig, _frame, frames=len(Tg), interval=1000 * sw_g.chain.dt_tick)
anim.save(os.path.join(OUT, "swim.gif"), writer=PillowWriter(fps=int(round(1 / sw_g.chain.dt_tick))))
plt.close(fig)

lines_md = ["# P3 gate report -- swimming\n",
            f"Generated by `swim_gates.py`, {time.strftime('%Y-%m-%d %H:%M')}. Side view, two bank arms (up / down groups of 4, "
            f"mass and drag x 4, web factor {SwimParams().web_factor} on broadside drag), body 0.3 kg / 0.05 m, 16 x 2 at 24 Hz. "
            f"Umbrella {SwimParams().closed_deg}-{SwimParams().open_deg} deg, open {SwimParams().open_s} s / close {SwimParams().close_s} s; "
            f"jet dv {SwimParams().jet_dv} m/s over {SwimParams().jet_s} s.\n",
            "| gate | result | measurement |", "|---|---|---|"]
for name, ok, note in results:
    lines_md.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {note} |")
lines_md += ["", "Figures: `fig8_swim.png`, `swim.gif`.", ""]
with open(os.path.join(OUT, "report_p3.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines_md))
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} gates pass. Report: {os.path.join(OUT, 'report_p3.md')}")
sys.exit(1 if n_fail else 0)
