"""P5: idle, scored by the measurable rows of the craft checklist
(games/film digest, synthesis section 2) over 60 s of a seated animal.

    <breach-py> prototypes/octopus_motion/idle_gates.py     (~2 min, writes out/)

Rows:
  1  few actors: arms whose distal half moves (max node speed > 0.03 m/s)
     at once: max <= 3, 95th percentile <= 2 (the proximal segments of a
     seated arm jiggle when the body turns under a reach: secondary motion)
  3  willed motion starts at the base: within a reach, the time each node's
     speed peaks increases with its index (Spearman rho >= 0.8), for >= 80 %
     of the gestures
  7  holds and rests: >= 25 % of the time nothing moves; the longest still
     interval >= 1.5 s; no rhythm: the intervals between gesture starts have
     CV >= 0.5 (a clock ~0); the spectral line test is reported only (too
     few events)
  8  de-synchrony: no two arms start a gesture within 0.3 s of each other
     more than 10 % of the time; the arms' gesture counts are spread
     (front > back on average, but every live arm moves at least once)
 13  stable over time: at the end every segment is within 3 % of its rest
     length, no node has strayed beyond 1.5 L from its root, and the body
     has not drifted more than 0.05 m (nothing pushes it)
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
from scipy.stats import spearmanr  # noqa: E402

from crawl_sim import ARM_NAMES  # noqa: E402
from idle_sim import make_idle_octopus  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
results: list[tuple[str, bool, str]] = []
MOVE = 0.03   # m/s


def gate(name: str, ok, note: str) -> None:
    results.append((name, bool(ok), note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {note}", flush=True)


t_wall = time.perf_counter()
o = make_idle_octopus(seed=5)
o.run(60.0)
print(f"  60 s idle: {len(o.gestures)} gestures ({time.perf_counter() - t_wall:.0f} s wall)", flush=True)
T = np.array(o.log_t)
P = np.array(o.log_p)                 # (t, 8, N, 2)
X = np.array(o.log_x)
L = o.geom.length_m
dt = o.chain.dt_tick
V = np.gradient(P, T, axis=0)
node_speed = np.linalg.norm(V, axis=3)        # (t, 8, N)
# an ACTOR moves its distal half: a reaching arm's reaction turns the body a little and jiggles every
# arm's proximal segments at ~3 cm/s between root and anchor -- secondary motion, not deliberate work
arm_speed = node_speed[:, :, 9:].max(axis=2)  # (t, 8)
moving = arm_speed > MOVE
moving[T < 1.0] = False          # the first second is the seated animal settling onto its anchors
n_moving = moving.sum(axis=1)

# ---- 1 few actors
gate("1 few actors", n_moving.max() <= 3 and np.percentile(n_moving, 95) <= 2,
     f"arms moving at once: max {n_moving.max()}, 95th percentile {np.percentile(n_moving, 95):.0f}, mean {n_moving.mean():.2f}")

# ---- 3 willed motion from the base
rhos = []
for arm, t0, t1 in o.gestures:
    if not np.isfinite(t1):
        continue
    i0 = int(np.searchsorted(T, t0))
    i1 = int(np.searchsorted(T, t0 + o.ex_wave[arm] if False else t0 + 1.3))   # the reach + touch window
    i1 = min(i1, len(T) - 1)
    if i1 - i0 < 6:
        continue
    seg = node_speed[i0:i1, arm, 2:]            # nodes 2.. (0, 1 ride on the body)
    t_peak = seg.argmax(axis=0)
    rho = float(spearmanr(np.arange(seg.shape[1]), t_peak)[0])
    rhos.append(rho)
rhos = np.array(rhos)
frac = float((rhos >= 0.8).mean()) if len(rhos) else 0.0
gate("3 willed motion starts at the base", len(rhos) >= 5 and frac >= 0.8,
     f"{len(rhos)} gestures; Spearman(node index, time of peak speed) median {np.median(rhos):.2f}, {frac:.0%} >= 0.8")

# ---- 7 holds and rests
still = n_moving == 0
runs = []
cur = 0
for s in still:
    if s:
        cur += 1
    else:
        if cur:
            runs.append(cur)
        cur = 0
if cur:
    runs.append(cur)
longest_still = max(runs) * dt if runs else 0.0
sig = n_moving.astype(float) - n_moving.mean()
freqs = np.fft.rfftfreq(len(sig), d=dt)
power = np.abs(np.fft.rfft(sig * np.hanning(len(sig)))) ** 2
k = 15
env = np.array([np.median(power[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(power))])
white = power / np.maximum(env, 1e-30)
band = (freqs >= 0.1) & (freqs <= 2.0)
line = float(white[band].max())
# 24 gestures in 60 s are too few events for P2's spectral-line test (reported anyway); the rhythm
# test here is the spread of the intervals between gesture starts: a clock has CV ~ 0, no rhythm >= 0.5
onsets = np.sort([t0 for _, t0, _ in o.gestures])
gaps = np.diff(onsets)
cv = float(gaps.std() / gaps.mean()) if len(gaps) > 2 else np.nan
gate("7 holds and rests", still.mean() >= 0.25 and longest_still >= 1.5 and cv >= 0.5,
     f"still {still.mean():.0%} of the time, longest still {longest_still:.1f} s; gaps between gesture starts "
     f"{gaps.mean():.2f} +- {gaps.std():.2f} s (CV {cv:.2f}); whitened spectral peak of the activity count {line:.1f}x "
     f"(at {freqs[band][white[band].argmax()]:.2f} Hz; too few events for that test to bind)")

# ---- 8 de-synchrony
starts = sorted((t0, arm) for arm, t0, _ in o.gestures)
close = 0
for i in range(1, len(starts)):
    if starts[i][0] - starts[i - 1][0] < 0.3 and starts[i][1] != starts[i - 1][1]:
        close += 1
counts = np.bincount([arm for arm, _, _ in o.gestures], minlength=8)
front = counts[[0, 7]].mean()
back = counts[[3, 4]].mean()
gate("8 de-synchrony and roles", close / max(len(starts), 1) <= 0.10 and counts.min() >= 1 and front >= back,
     f"gesture starts within 0.3 s of another arm's: {close}/{len(starts)}; gestures per arm "
     + ", ".join(f"{ARM_NAMES[a]} {counts[a]}" for a in range(8)) + f"; front pair {front:.1f} vs back pair {back:.1f}")

# ---- 13 stable over time
p_end = P[-1]
seg = np.linalg.norm(p_end[:, 1:] - p_end[:, :-1], axis=2)
rest = o.bank.rest
still_arms = [a for a in range(8) if o.ex[a] == "still"]     # an arm mid-gesture is being re-shaped on purpose
seg_err = float(np.abs(seg[still_arms] / rest[still_arms] - 1).max()) if still_arms else np.nan
reach = np.linalg.norm(p_end - p_end[:, :1], axis=2).max() / L
body_drift = float(np.linalg.norm(X[-1] - X[0]))
gate("13 stable over time", seg_err <= 0.03 and reach <= 1.5 and body_drift <= 0.05,
     f"segment length error max {seg_err:.1%} over the {len(still_arms)} still arms, farthest node {reach:.2f} L from its root, body drift {body_drift * 1e3:.0f} mm in 60 s")

# ---- figures
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axs[0]
for arm in range(8):
    ax.fill_between(T, arm - 0.4, arm + 0.4, where=moving[:, arm], color="C%d" % arm, step="mid")
for arm, t0, t1 in o.gestures:
    ax.plot([t0], [arm], "k|", ms=8)
ax.set_yticks(range(8))
ax.set_yticklabels(ARM_NAMES)
ax.set_xlabel("t (s)")
ax.set_title("Who moves (colour) and when a gesture is decided (tick): holds between, never all at once")
ax = axs[1]
ax.plot(T, n_moving, lw=0.8, color="k")
ax.set_xlabel("t (s)")
ax.set_ylabel("arms moving")
ax.set_title(f"Arms moving at once: max {n_moving.max()}, still {still.mean():.0%} of the time")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig10_idle.png"))
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.set_aspect("equal")
ax.set_xlim(X[0, 0] - 0.42, X[0, 0] + 0.42)
ax.set_ylim(X[0, 1] - 0.42, X[0, 1] + 0.42)
lines = [ax.plot([], [], "-", color="#5a3b8a", lw=2.4, solid_capstyle="round")[0] for _ in range(8)]
body = Circle((X[0, 0], X[0, 1]), o.body.radius, color="#5a3b8a", alpha=0.6)
ax.add_patch(body)
txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=8, va="top")
ax.set_title("P5: idle -- explore, touch, go limp; holds between (top-down)", fontsize=8)
n_frames = int(24.0 / dt)


def _frame(i):
    for arm in range(8):
        lines[arm].set_data(P[i, arm, :, 0], P[i, arm, :, 1])
        lines[arm].set_color("#c0392b" if o.log_ex[i][arm] in ("reach", "touch") else ("#8e6bbf" if o.log_ex[i][arm] == "limp" else "#5a3b8a"))
    body.center = (X[i, 0], X[i, 1])
    acts = [f"{ARM_NAMES[a]}:{o.log_ex[i][a]}" for a in range(8) if o.log_ex[i][a] != "still"]
    txt.set_text(f"t = {T[i]:.1f} s   {'  '.join(acts) or 'still'}")
    return lines + [body, txt]


anim = FuncAnimation(fig, _frame, frames=n_frames, interval=1000 * dt)
anim.save(os.path.join(OUT, "idle.gif"), writer=PillowWriter(fps=int(round(1 / dt))))
plt.close(fig)

lines_md = ["# P5 report -- idle, scored by the craft checklist\n",
            f"Generated by `idle_gates.py`, {time.strftime('%Y-%m-%d %H:%M')}. 60 s, seed 5, the P2 animal seated; explore = decide "
            "0.4-1.6 s, reach (P1 wave 0.7-1.3 s, stretch 1.15-1.35, aim +-45 deg), touch 0.3-1.1 s, limp 1-2.5 s; at most 2 arms "
            "exploring, p_start 0.15 per 0.1 s, front bias 1.0. Breathing and tip curls are render controls, not scored here.\n",
            "| checklist row | result | measurement |", "|---|---|---|"]
for name, ok, note in results:
    lines_md.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {note} |")
lines_md += ["", "Rows 2, 4, 5, 6, 9, 10, 11, 12, 14 are judged by eye on `idle.gif` and the P8 clips. Figures: `fig10_idle.png`, `idle.gif`.", ""]
with open(os.path.join(OUT, "report_p5.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines_md))
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} rows pass. Report: {os.path.join(OUT, 'report_p5.md')}")
sys.exit(1 if n_fail else 0)
