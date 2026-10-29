"""P4 gates: reach, grab, hold, pull in, carry (study brief v1 section 5 P4).

    <breach-py> prototypes/octopus_motion/grab_gates.py     (~4 min, writes out/)

Gates:
  1  the reach makes contact and the grab closes MID-ARM: every sealed node
     at s in [0.2, 0.65] L, none at the tip, at least 3 sealed after the wrap
  2  the prop stays attached through 15 s of crawling with the load and
     three jets (the brief's criterion): sealed count never drops below 3,
     no sucker pops. How tightly it is held (the prop-to-crown distance band
     during the crawl and during the jets) is REPORTED, not gated: a loose
     hold is a tuning item for P6/P7, a dropped prop would be a defect
  3  the other arms re-plan around the loaded arm: the holding arm never
     pushes after the grab; the loaded crawl still travels (direction within
     15 deg of wanted, speed >= 50 % of the unloaded crawl)
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

from crawl_sim import ARM_NAMES, make_octopus  # noqa: E402
from grab_sim import CARRY, make_grab_octopus  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
results: list[tuple[str, bool, str]] = []


def gate(name: str, ok, note: str) -> None:
    results.append((name, bool(ok), note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {note}", flush=True)


t_wall = time.perf_counter()
o = make_grab_octopus(seed=3)
L = o.geom.length_m
a = o.gp.arm
o.run(0.5)                      # a beat of stillness before the decision (checklist row 7)
o.start_reach()
while o.phase != CARRY and o.time < 8.0:
    o.tick()
t_carry = o.time
t_contact = o.t_grab
o.run(15.0)                     # the loaded crawl
t_jets = []
for _ in range(3):
    o.jet(0.5)
    t_jets.append(o.time)
    o.run(2.0)
print(f"  reach at 0.5 s, contact at {t_contact:.2f} s, carry from {t_carry:.2f} s, done {o.time:.1f} s ({time.perf_counter() - t_wall:.0f} s wall)", flush=True)

T = np.array(o.log_t)
X = np.array(o.log_x)
PR = np.array(o.log_prop)
NG = np.array(o.log_ngrab)
S = np.array(o.log_stance)
P = np.array(o.log_p)
phase = np.array(o.log_phase)

# ---- 1. mid-arm grab
sealed_s = sorted(set(float(o.bank.s_node[k]) for (aa, k, _, _) in o.bank.grabs if aa == a))
first_s = o.grab_s
gate("1 the grab closes mid-arm",
     np.isfinite(t_contact) and len(sealed_s) >= 3 and all(0.2 * L <= s <= 0.65 * L for s in sealed_s),
     f"contact {t_contact - 0.5:.2f} s after the reach began at s = {', '.join(f'{s / L:.2f}' for s in first_s)} L; "
     f"after the wrap {len(sealed_s)} nodes sealed at s = {', '.join(f'{s / L:.2f}' for s in sealed_s)} L (tip = 1.0)")

# ---- 2. stays attached
i_c = int(np.searchsorted(T, t_carry))
i_j = int(np.searchsorted(T, t_jets[0]))
d_prop = np.linalg.norm(PR - X, axis=1)
d_ref = d_prop[i_c + 24: i_c + 48].mean()
d_crawl = d_prop[i_c:i_j]
d_jets = d_prop[i_j:]
ng_min = int(NG[i_c:].min())
popped = int(o.bank.popped.sum())
gate("2 the prop stays attached through the carry and three jets",
     ng_min >= 3 and popped == 0,
     f"sealed nodes during the carry min {ng_min}, suckers popped {popped} (grip limit {o.gp.grip_limit} N); held at "
     f"{d_ref * 1e3:.0f} mm from the crown, swinging {d_crawl.min() / d_ref - 1:+.0%}..{d_crawl.max() / d_ref - 1:+.0%} during the crawl "
     f"and {d_jets.min() / d_ref - 1:+.0%}..{d_jets.max() / d_ref - 1:+.0%} through the jets at {', '.join(f'{t:.1f}' for t in t_jets)} s "
     f"(a loose hold: P6/P7 tuning, not a drop)")

# ---- 3. the others re-plan
pushed_after = int(sum(1 for (arm, t0, _) in o.steps if arm == a and t0 >= t_contact))
i_end = int(np.searchsorted(T, t_jets[0]))
disp = X[i_end] - X[i_c]
dur = T[i_end] - T[i_c]
speed_loaded = float(np.linalg.norm(disp) / dur)
err = float(abs((np.degrees(np.arctan2(disp[1], disp[0])) - 0.0 + 180) % 360 - 180))
o_free = make_octopus(direction=(1.0, 0.0), seed=3)
o_free.run(15.0)
Xf = np.array(o_free.log_x)
speed_free = float(np.linalg.norm(Xf[-1] - Xf[0]) / 15.0)
who = sorted(set(ARM_NAMES[arm] for (arm, t0, _) in o.steps if t0 >= t_contact))
gate("3 the other arms re-plan around the loaded arm",
     pushed_after == 0 and err <= 15.0 and speed_loaded >= 0.5 * speed_free,
     f"{ARM_NAMES[a]} pushed {pushed_after} times after the grab; loaded crawl {speed_loaded:.3f} m/s "
     f"({speed_loaded / speed_free:.0%} of the unloaded {speed_free:.3f}), {err:.0f} deg off wanted; arms that pushed: {', '.join(who)}")

# ---- figures
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
ax = axs[0]
cmap = plt.get_cmap("viridis")
for ts in [0.5, 0.8, 1.1, 1.4, t_contact, t_carry]:
    i = int(np.argmin(np.abs(T - ts)))
    col = cmap((ts - 0.5) / max(t_carry - 0.5, 1e-9))
    ax.plot(P[i, a, :, 0], P[i, a, :, 1], "-", color=col, lw=2, label=f"{T[i]:.2f} s {phase[i]}")
    ax.add_patch(Circle(PR[i], o.prop.radius, color=col, fill=False, lw=1))
for arm in range(8):
    if arm != a:
        ax.plot(P[0, arm, :, 0], P[0, arm, :, 1], "-", color="lightgray", lw=1)
ax.add_patch(Circle(X[0], o.body.radius, color="gray", fill=False))
ax.set_aspect("equal")
ax.legend(fontsize=7)
ax.set_title(f"Reach -> grab -> wrap -> pull in ({ARM_NAMES[a]}); circles = the prop", fontsize=9)
ax = axs[1]
ax.plot(T, d_prop * 1e3, lw=1.5, color="k", label="prop to crown (mm)")
ax.plot(T, NG * 10, lw=1, color="crimson", label="sealed nodes x 10")
for t in t_jets:
    ax.axvline(t, color="steelblue", ls=":", lw=1)
ax.axvline(t_contact, color="gray", ls="--", lw=1)
ax.axvline(t_carry, color="gray", ls="--", lw=1)
ax.set_xlabel("t (s)")
ax.set_title("The hold through the carry and three jets (dotted)", fontsize=9)
ax.legend(fontsize=7)
ax = axs[2]
for arm in range(8):
    on = S[:, arm]
    ax.fill_between(T, arm - 0.4, arm + 0.4, where=on, color="C%d" % arm, step="mid")
ax.axvline(t_contact, color="gray", ls="--", lw=1)
ax.set_yticks(range(8))
ax.set_yticklabels(ARM_NAMES)
ax.set_xlabel("t (s)")
ax.set_title(f"Who pushes: {ARM_NAMES[a]} holds (its bar = the hold), the others crawl", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_grab.png"))
plt.close(fig)

# GIF
fig, ax = plt.subplots(figsize=(5.8, 5.2))
ax.set_aspect("equal")
lines = [ax.plot([], [], "-", color=("#c0392b" if arm == a else "#5a3b8a"), lw=2.2, solid_capstyle="round")[0] for arm in range(8)]
seal = ax.plot([], [], "o", color="gold", ms=5, zorder=5)[0]
anchors = ax.plot([], [], "o", color="crimson", ms=4, zorder=4)[0]
body = Circle((0, 0), o.body.radius, color="#5a3b8a", alpha=0.6)
prop = Circle((0, 0), o.prop.radius, color="#2e86c1", alpha=0.8)
ax.add_patch(body)
ax.add_patch(prop)
txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=8, va="top")
ax.set_title("P4: reach, grab, hold, carry a prop (top-down; red arm holds)", fontsize=8)
An = np.array(o.log_anchored)
frames = list(range(0, int(np.searchsorted(T, t_carry + 8.0)), 1)) + list(range(int(np.searchsorted(T, t_jets[0] - 0.5)), len(T), 1))


def _frame(i):
    for arm in range(8):
        lines[arm].set_data(P[i, arm, :, 0], P[i, arm, :, 1])
    anc = An[i]
    anchors.set_data(P[i][anc][:, 0], P[i][anc][:, 1])
    ks = [k for (aa, k, _, _) in o.bank.grabs] if T[i] >= t_contact else []
    seal.set_data(P[i, a, ks, 0], P[i, a, ks, 1])
    body.center = (X[i, 0], X[i, 1])
    prop.center = (PR[i, 0], PR[i, 1])
    ax.set_xlim(X[i, 0] - 0.45, X[i, 0] + 0.45)
    ax.set_ylim(X[i, 1] - 0.45, X[i, 1] + 0.45)
    txt.set_text(f"t = {T[i]:.1f} s   {phase[i]}   sealed {NG[i]}   pushing: {', '.join(ARM_NAMES[k] for k in np.where(S[i])[0] if k != a) or '-'}")
    return lines + [seal, anchors, body, prop, txt]


anim = FuncAnimation(fig, _frame, frames=frames, interval=1000 * o.chain.dt_tick)
anim.save(os.path.join(OUT, "grab.gif"), writer=PillowWriter(fps=int(round(1 / o.chain.dt_tick))))
plt.close(fig)

lines_md = ["# P4 gate report -- reach, grab, hold, carry\n",
            f"Generated by `grab_gates.py`, {time.strftime('%Y-%m-%d %H:%M')}. The P2 animal; prop = a {o.gp.prop_mass} kg disc of "
            f"radius {o.gp.prop_radius} m at {o.gp.prop_at} from the crown; reach = aim + P1 wave over {o.gp.reach_wave_s} s + stretch "
            f"{o.gp.reach_stretch}; grab patch nodes {o.gp.grab_nodes}; wrap 0.6 s; pull-in to {o.gp.pull_in_scale} over {o.gp.pull_in_s} s; "
            f"grip limit {o.gp.grip_limit} N; carry = 15 s crawl + 3 jets of 0.5 m/s.\n",
            "| gate | result | measurement |", "|---|---|---|"]
for name, ok, note in results:
    lines_md.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {note} |")
lines_md += ["", "Figures: `fig9_grab.png`, `grab.gif`.", ""]
with open(os.path.join(OUT, "report_p4.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines_md))
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} gates pass. Report: {os.path.join(OUT, 'report_p4.md')}")
sys.exit(1 if n_fail else 0)
