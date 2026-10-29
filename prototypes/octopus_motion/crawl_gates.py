"""P2 gates: the crawl (study brief v1 §5 P2).

    <breach-py> prototypes/octopus_motion/crawl_gates.py      (~6 min, writes out/)

Gates:
  0  the arm bank reproduces the P1 chain bit for bit (one arm)
  1  travels where wanted: r >= 0.9 between the predicted direction
     (sum of the pushing arms' push directions, Levy 2015) and the body's
     velocity direction; final displacement within 20 deg of the wanted
     direction, for four wanted directions relative to the heading
  2  1-3 arms push at any time (never 0 for long, never > 3)
  3  the body-speed spectrum is flat (no rhythm): no peak above 4x the
     median power in 0.1-2 Hz
  4  still travels with 7 and 6 arms (>= 50 % of the 8-arm displacement,
     direction within 25 deg)
  5  body heading is independent of travel direction (heading drift < 20 deg
     over 25 s of crawling 70 deg off the heading)
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
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from arm_bank import ArmBank  # noqa: E402
from arm_chain import ArmChain, ArmGeometry, ChainParams, ReachCommand, hooked_pose  # noqa: E402
from crawl_sim import ARM_ANGLES, ARM_NAMES, make_octopus  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130})
results: list[tuple[str, bool, str]] = []


def gate(name: str, ok, note: str) -> None:
    results.append((name, bool(ok), note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {note}", flush=True)


def ang_deg(v) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def ang_err(a, b) -> float:
    return float(abs((ang_deg(a) - ang_deg(b) + 180) % 360 - 180))


# ------------------------------------------------ 0. bank == chain (one arm)
geom, prm, p0 = ArmGeometry(), ChainParams(), hooked_pose(ArmGeometry())
chain, bank = ArmChain(geom, prm, p0), ArmBank(geom, prm, p0[None])
bank.w[:, :2] = 0.0
cmd = ReachCommand()
dt = prm.dt_tick / prm.n_sub
for _ in range(24 * prm.n_sub):
    chain.set_activation_wave(cmd, chain.time)
    bank.activation[0] = chain.activation
    chain.substep(dt)
    bank.substep(dt)
gate("0 arm bank == P1 chain", np.abs(bank.p[0] - chain.p).max() == 0.0,
     f"max |dp| {np.abs(bank.p[0] - chain.p).max():.1e} after 1 s")


# ------------------------------------------------ 1. four directions
def analyse(o):
    T = np.array(o.log_t)
    X = np.array(o.log_x)
    S = np.array(o.log_stance)
    P = np.array(o.log_pred)
    V = np.gradient(X, T, axis=0)
    sp = np.linalg.norm(V, axis=1)
    ok = (S.sum(1) > 0) & (sp > 0.01) & (np.linalg.norm(P, axis=1) > 0)
    cosines = np.array([(V[i] @ P[i]) / (sp[i] * np.linalg.norm(P[i])) for i in np.where(ok)[0]])
    # Levy 2015's statistic: one point per push -- the body's displacement direction over the push
    # against the direction of the pushing arms' vector sum at its midpoint; r = circular correlation
    pred_ang, act_ang = [], []
    for arm, t0, t1 in o.steps:
        if not np.isfinite(t1):
            continue
        i0, i1 = int(np.searchsorted(T, t0)), int(np.searchsorted(T, t1))
        if i1 - i0 < 2 or i1 >= len(T):
            continue
        d = X[i1] - X[i0]
        pm = P[(i0 + i1) // 2]
        if np.linalg.norm(d) < 0.005 or np.linalg.norm(pm) == 0:
            continue
        act_ang.append(np.arctan2(d[1], d[0]))
        pred_ang.append(np.arctan2(pm[1], pm[0]))
    pa, aa = np.array(pred_ang), np.array(act_ang)
    if len(pa) > 3:   # Jammalamadaka-Sarma circular correlation
        num = np.sum(np.sin(pa - np.angle(np.exp(1j * pa).mean())) * np.sin(aa - np.angle(np.exp(1j * aa).mean())))
        den = np.sqrt(np.sum(np.sin(pa - np.angle(np.exp(1j * pa).mean())) ** 2) * np.sum(np.sin(aa - np.angle(np.exp(1j * aa).mean())) ** 2))
        r_circ = float(num / den) if den > 0 else np.nan
        mean_abs_err = float(np.degrees(np.abs((pa - aa + np.pi) % (2 * np.pi) - np.pi)).mean())
    else:
        r_circ, mean_abs_err = np.nan, np.nan
    disp = X[-1] - X[0]
    return dict(T=T, X=X, S=S, V=V, sp=sp, r=r_circ, cos_tick=float(cosines.mean()) if len(cosines) else np.nan,
                n_push=len(pa), push_err=mean_abs_err,
                n_ok=int(ok.sum()), disp=disp, err=ang_err(disp, o.travel),
                heading_drift=float(np.degrees(o.log_theta[-1] - o.log_theta[0])),
                stance_mean=float(S.sum(1).mean()), stance_max=int(S.sum(1).max()),
                idle_frac=float((S.sum(1) == 0).mean()))


dirs_deg = [0.0, 70.0, 135.0, -90.0]
runs = {}
t0 = time.perf_counter()
for d in dirs_deg:
    o = make_octopus(direction=(np.cos(np.radians(d)), np.sin(np.radians(d))), seed=1)
    o.run(25.0)
    runs[d] = (o, analyse(o))
    a = runs[d][1]
    print(f"  dir {d:+.0f} deg: disp {np.linalg.norm(a['disp']):.2f} m, err {a['err']:.1f} deg, r {a['r']:.3f} over {a['n_push']} pushes "
          f"(per-push |err| {a['push_err']:.0f} deg, per-tick cos {a['cos_tick']:.2f}), "
          f"stance mean {a['stance_mean']:.2f} max {a['stance_max']}, heading drift {a['heading_drift']:+.1f} deg "
          f"({time.perf_counter() - t0:.0f} s)", flush=True)
rs = [runs[d][1]["r"] for d in dirs_deg]
errs = [runs[d][1]["err"] for d in dirs_deg]
gate("1 travels where the pushing arms say", min(rs) >= 0.9 and max(errs) <= 20.0,
     "circular r (per push: direction of the pushing arms' sum vs the body's displacement) = " + ", ".join(f"{r:.2f}" for r in rs)
     + "; per-push |angle error| " + ", ".join(f"{runs[d][1]['push_err']:.0f}" for d in dirs_deg) + " deg"
     + "; per-tick mean cos " + ", ".join(f"{runs[d][1]['cos_tick']:.2f}" for d in dirs_deg)
     + "; direction error = " + ", ".join(f"{e:.0f}" for e in errs) + f" deg for wanted {dirs_deg} deg off the heading; "
     + "speed mean " + ", ".join(f"{runs[d][1]['sp'].mean():.3f}" for d in dirs_deg) + " m/s")
gate("2 one to three arms push", all(runs[d][1]["stance_max"] <= 3 for d in dirs_deg)
     and all(runs[d][1]["stance_mean"] >= 0.7 for d in dirs_deg),
     "stance mean " + ", ".join(f"{runs[d][1]['stance_mean']:.2f}" for d in dirs_deg)
     + ", max " + ", ".join(str(runs[d][1]["stance_max"]) for d in dirs_deg)
     + ", fraction of time with no arm pushing " + ", ".join(f"{runs[d][1]['idle_frac']:.0%}" for d in dirs_deg))

# ------------------------------------------------ 3. spectrum
def speed_spectrum(o):
    a = analyse(o)
    sp = a["sp"] - a["sp"].mean()
    fs = 1.0 / prm.dt_tick
    freqs = np.fft.rfftfreq(len(sp), d=1.0 / fs)
    power = np.abs(np.fft.rfft(sp * np.hanning(len(sp)))) ** 2
    # whiten by the local envelope (a push pulse + coast has a sloped spectrum that is not a rhythm):
    # a rhythm is a LINE standing above its neighbours
    k = 15
    env = np.array([np.median(power[max(0, i - k // 2): i + k // 2 + 1]) for i in range(len(power))])
    white = power / np.maximum(env, 1e-30)
    band = (freqs >= 0.1) & (freqs <= 2.0)
    return a, freqs, power, white, band


o_long = make_octopus(direction=(1.0, 0.0), seed=7)
o_long.run(40.0)
al, freqs, power, white, band = speed_spectrum(o_long)
line_ratio = float(white[band].max())
f_peak = float(freqs[band][white[band].argmax()])
step_durs = np.array([t1 - t0 for _, t0, t1 in o_long.steps if np.isfinite(t1)])
starts = np.array([t0 for _, t0, _ in o_long.steps])
gaps = np.diff(np.sort(starts))
# the CONTROL: the same animal on a clock (no jitter, recruit the moment an arm is ready) must show a line
from crawl_sim import CrawlParams  # noqa: E402
o_clock = make_octopus(direction=(1.0, 0.0), seed=7,
                       crawl=CrawlParams(push_jitter=0.0, recover_jitter=0.0, p_recruit=1.0))
o_clock.run(40.0)
ac, freqs_c, power_c, white_c, band_c = speed_spectrum(o_clock)
line_ratio_clock = float(white_c[band_c].max())
f_peak_clock = float(freqs_c[band_c][white_c[band_c].argmax()])
# For a line-free band the whitened power is ~exponential around 1: the max over ~76 bins is ~6 by
# chance; a clock stands tens of times above its neighbours. Threshold 12, and the clocked control
# must exceed it or the statistic cannot tell.
gate("3 no rhythm: no line in the speed spectrum", line_ratio <= 12.0 and line_ratio_clock > 12.0,
     f"whitened peak {line_ratio:.1f}x its neighbours (at {f_peak:.2f} Hz); the CLOCKED control {line_ratio_clock:.1f}x "
     f"(at {f_peak_clock:.2f} Hz = 1 / {1 / f_peak_clock:.2f} s); {len(o_long.steps)} steps in 40 s, "
     f"push {step_durs.mean():.2f} +- {step_durs.std():.2f} s, gap between recruits {gaps.mean():.2f} +- {gaps.std():.2f} s (CV {gaps.std() / gaps.mean():.2f})")

# ------------------------------------------------ 4. seven and six arms
base_disp = np.linalg.norm(runs[0.0][1]["disp"])
loss = {}
for label, dead in [("7 arms (L4 dead)", (3,)), ("6 arms (L4, R3 dead)", (3, 5))]:
    o = make_octopus(direction=(1.0, 0.0), seed=1, dead=dead)
    o.run(25.0)
    a = analyse(o)
    loss[label] = (o, a)
    print(f"  {label}: disp {np.linalg.norm(a['disp']):.2f} m ({np.linalg.norm(a['disp']) / base_disp:.0%}), err {a['err']:.0f} deg, r {a['r']:.2f}, per-push |err| {a['push_err']:.0f} deg", flush=True)
gate("4 still travels with 7 and 6 arms",
     all(np.linalg.norm(a["disp"]) >= 0.5 * base_disp and a["err"] <= 25.0 for _, a in loss.values()),
     "; ".join(f"{k}: {np.linalg.norm(a['disp']) / base_disp:.0%} of 8-arm displacement, {a['err']:.0f} deg off" for k, (_, a) in loss.items()))

# ------------------------------------------------ 5. heading independent
drifts = [runs[d][1]["heading_drift"] for d in dirs_deg]
# Levy 2015 SI: the real heading wanders (SD ~25 deg per 2.8 s) and is uncorrelated with the travel
# direction (r ~ 0.12). The gate: the body never turns to face where it goes -- heading stays within
# 30 deg of its start over 25 s while the travel direction is up to 135 deg off it.
gate("5 heading independent of travel direction", max(abs(x) for x in drifts) < 30.0,
     "heading drift over 25 s = " + ", ".join(f"{x:+.1f}" for x in drifts) + f" deg while travelling {dirs_deg} deg off it "
     "(the real animal wanders ~25 deg SD per 2.8 s)")

# ------------------------------------------------ figures
fig, axs = plt.subplots(2, 2, figsize=(11, 9))
ax = axs[0, 0]
for d in dirs_deg:
    X = runs[d][1]["X"]
    ax.plot(X[:, 0], X[:, 1], lw=1.8, label=f"wanted {d:+.0f} deg (err {runs[d][1]['err']:.0f})")
    ax.annotate("", xy=0.3 * np.array([np.cos(np.radians(d)), np.sin(np.radians(d))]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))
ax.set_aspect("equal")
ax.set_title("Body path for four wanted directions (heading stays +x; grey = wanted)")
ax.legend(fontsize=7)
ax = axs[0, 1]
o, a = runs[0.0]
for arm in range(8):
    on = a["S"][:, arm]
    ax.fill_between(a["T"], arm - 0.4, arm + 0.4, where=on, color="C%d" % arm, step="mid")
ax.set_yticks(range(8))
ax.set_yticklabels(ARM_NAMES)
ax.set_xlabel("t (s)")
ax.set_title("Which arm pushes (wanted = heading): no order, no clock")
ax = axs[1, 0]
ax.plot(a["T"], a["sp"], lw=1, color="k", label="body speed")
ax.plot(a["T"], a["S"].sum(1) * 0.03, lw=0.8, color="C3", alpha=0.6, label="arms pushing x 0.03")
ax.set_xlabel("t (s)")
ax.set_ylabel("m/s")
ax.set_title("Body speed: a push is a pulse, the body coasts between")
ax.legend(fontsize=7)
ax = axs[1, 1]
ax.semilogy(freqs[1:], white[1:], lw=0.8, color="k", label=f"lottery: peak {line_ratio:.1f}x")
ax.semilogy(freqs_c[1:], white_c[1:], lw=0.8, color="C3", alpha=0.7, label=f"clocked control: peak {line_ratio_clock:.1f}x")
ax.axhline(12.0, color="gray", ls="--", lw=1)
ax.axvspan(0.1, 2.0, color="y", alpha=0.15)
ax.set_xlim(0, 3)
ax.set_xlabel("Hz")
ax.legend(fontsize=7)
ax.set_title("Whitened speed spectrum over 40 s: a clock is a line, the lottery is not")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_crawl.png"))
plt.close(fig)

# snapshots top-down + the loss runs
fig, axs = plt.subplots(1, 3, figsize=(14, 4.8))
for ax, (title, (o, a)) in zip(axs, [("8 arms", runs[0.0]), *loss.items()]):
    P = np.array(o.log_p)
    An = np.array(o.log_anchored)
    for ts in np.arange(0, 25.0, 5.0):
        i = int(np.argmin(np.abs(a["T"] - ts)))
        col = plt.get_cmap("viridis")(ts / 25.0)
        for arm in range(8):
            ax.plot(P[i, arm, :, 0], P[i, arm, :, 1], "-", color=col, lw=1.0 if o.bank.alive[arm] else 0.5,
                    alpha=0.9 if o.bank.alive[arm] else 0.4)
            anc = An[i, arm]
            if anc.any():
                ax.plot(P[i, arm, anc, 0], P[i, arm, anc, 1], "o", color="crimson", ms=3)
        circ = Circle(a["X"][i], o.body.radius, color=col, fill=False, lw=1.5)
        ax.add_patch(circ)
    ax.plot(a["X"][:, 0], a["X"][:, 1], "k-", lw=0.8)
    ax.set_aspect("equal")
    ax.set_title(f"{title}: every 5 s, red = anchored suckers\n{np.linalg.norm(a['disp']):.2f} m in 25 s, {a['err']:.0f} deg off", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_crawl_snapshots.png"))
plt.close(fig)

# GIF of the first 12 s of the 70-degree run, top-down
o, a = runs[70.0]
P = np.array(o.log_p)
An = np.array(o.log_anchored)
X = a["X"]
fig, ax = plt.subplots(figsize=(5.6, 5.0))
ax.set_aspect("equal")
lines = [ax.plot([], [], "-", color="#5a3b8a", lw=2.2, solid_capstyle="round")[0] for _ in range(8)]
anchors = ax.plot([], [], "o", color="crimson", ms=5, zorder=4)[0]
body = Circle((0, 0), o.body.radius, color="#5a3b8a", alpha=0.6)
ax.add_patch(body)
head, = ax.plot([], [], "-", color="white", lw=2)
trail, = ax.plot([], [], "-", color="gray", lw=0.8)
txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=8, va="top")
ax.set_title("P2 crawl, wanted direction 70 deg off the heading (top-down)", fontsize=8)
n_frames = int(12.0 / prm.dt_tick)


def _frame(i):
    for arm in range(8):
        lines[arm].set_data(P[i, arm, :, 0], P[i, arm, :, 1])
    anc = An[i]
    anchors.set_data(P[i][anc][:, 0], P[i][anc][:, 1])
    body.center = (X[i, 0], X[i, 1])
    th = o.log_theta[i]
    head.set_data([X[i, 0], X[i, 0] + o.body.radius * np.cos(th)], [X[i, 1], X[i, 1] + o.body.radius * np.sin(th)])
    trail.set_data(X[: i + 1, 0], X[: i + 1, 1])
    ax.set_xlim(X[i, 0] - 0.45, X[i, 0] + 0.45)
    ax.set_ylim(X[i, 1] - 0.45, X[i, 1] + 0.45)
    txt.set_text(f"t = {a['T'][i]:.1f} s   pushing: {', '.join(ARM_NAMES[k] for k in np.where(a['S'][i])[0]) or '-'}")
    return lines + [anchors, body, head, trail, txt]


anim = FuncAnimation(fig, _frame, frames=n_frames, interval=1000 * prm.dt_tick)
anim.save(os.path.join(OUT, "crawl.gif"), writer=PillowWriter(fps=int(round(1 / prm.dt_tick))))
plt.close(fig)

# ------------------------------------------------ report
lines_md = ["# P2 gate report -- the crawl\n",
            f"Generated by `crawl_gates.py`, {time.strftime('%Y-%m-%d %H:%M')}. Yekutieli's arm x 8 on a 0.05 m crown, "
            f"body 0.3 kg, in water on a deck (mu_k 0.1 on the net weight), 16 substeps x 2 passes at 24 Hz. "
            "PUSH_STEP: mid-arm anchor (nodes 5-7), proximal elongation to 1.6 over 0.7 s, release on saturation, "
            "recover 1.2 + U(0,1) s; hub lottery every 0.1 s, p 0.5, max 3 in stance, back bias 0.5, neighbour bonus 0.3.\n",
            "| gate | result | measurement |", "|---|---|---|"]
for name, ok, note in results:
    lines_md.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {note} |")
lines_md += ["", "Figures: `fig6_crawl.png` (paths, stance raster, speed, spectrum), `fig7_crawl_snapshots.png` "
             "(8 / 7 / 6 arms), `crawl.gif`.", ""]
with open(os.path.join(OUT, "report_p2.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines_md))
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"\n{len(results) - n_fail}/{len(results)} gates pass. Report: {os.path.join(OUT, 'report_p2.md')}")
sys.exit(1 if n_fail else 0)
