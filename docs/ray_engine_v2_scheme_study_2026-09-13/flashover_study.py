"""Task 2 -- the flashover study case.

Erik's worry (handoff section 4b), which he guessed himself: in a compartment near
equilibrium every surface emits a lot and absorbs nearly as much, so the NET is a small
difference between large numbers.  A transport loss of fraction f shrinks absorbed but
not emitted, producing a systematic cooling proportional to emission -- i.e. to T^4, the
same shape as real radiative loss, therefore invisible and untunable-around.  It would
impose a hidden temperature ceiling and could silently cap flashover.

This makes it a number.  A sealed compartment lined with furniture, one sustained fire
held at the measured plateau, every surface radiating by its own temperature (R-K).
Sweep -> dE -> temperature -> ambient relaxation, at the real 24 Hz tick and the real
constants (rad_scale, heat_inv_shift 3, cool_shift 13, ignition 280 game).

THE SWEEP IS EXACTLY LINEAR IN THE EMISSION FIELD, so instead of running a sweep per
tick we build the transfer matrix ONCE per geometry (one sweep per emitting cell) and
the 15-minute run becomes a matrix-vector product per tick.  Exact, not an approximation.

Two knobs:
    k_leak  -- the ACCOUNTED out-of-plane export (the proposed reach lever), which
               carries the ceiling's return radiation so ambient stays a fixed point
    f_lost  -- an UNACCOUNTED transport loss with no return term: section 4b's fear
"""
import math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_ref import sweep_blend

RAD_SCALE = 5.1427e-5
K_AMB = 293.0
IGNITION = 280.0
HIS = 3
COOL_SHIFT = 13
ONE = 65536.0             # temperature is Q16.16: the fold is (dE >> his) / 65536
TICK_HZ = 24.0
PLATEAU = 1556.0 - K_AMB

def E_deg(T):
    return RAD_SCALE * (K_AMB + np.maximum(T, 0.0))**4

IH, IW = 12, 24
H, W = IH + 2, IW + 2

def build():
    a = np.zeros((H, W)); wall = np.zeros((H, W), bool)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 1.0
    wall[0, :] = wall[-1, :] = wall[:, 0] = wall[:, -1] = True
    fire = np.zeros((H, W), bool)
    fire[IH//2:IH//2+2, 2:4] = True
    a[fire] = 1.0
    return a, wall, fire

def transfer(a, k_leak, theta, with_return):
    """Phi = M @ E + c.  One sweep per emitting cell, once per geometry."""
    idx = np.argwhere(a > 0)
    n = len(idx)
    E_amb = E_deg(0.0)
    amb_for_return = E_amb if with_return else 0.0
    # constant term: boundary inflow (and ceiling return) with no emission anywhere
    _d, c, *_ = sweep_blend(a, np.zeros((H, W)), kleak=k_leak, E_amb=amb_for_return,
                            n=16, theta=theta)
    M = np.zeros((n, n))
    for j, (jy, jx) in enumerate(idx):
        E = np.zeros((H, W)); E[jy, jx] = 1.0
        _d, fl, *_ = sweep_blend(a, E, kleak=k_leak, E_amb=0.0, n=16, theta=theta)
        M[:, j] = fl[idx[:, 0], idx[:, 1]]
    return idx, M, c[idx[:, 0], idx[:, 1]]

def run(k_leak=0.0, f_lost=0.0, theta=1.0, seconds=900.0):
    a, wall, fire = build()
    # an unaccounted loss is an extinction with NO return term
    kl = k_leak + f_lost
    idx, M, c = transfer(a, kl, theta, with_return=(f_lost == 0.0))
    av = a[idx[:, 0], idx[:, 1]]
    is_fire = fire[idx[:, 0], idx[:, 1]]
    is_wall = wall[idx[:, 0], idx[:, 1]]
    T = np.zeros(len(idx)); T[is_fire] = PLATEAU
    hist = []
    for t in range(int(seconds * TICK_HZ)):
        E = E_deg(T)
        dE = av * (M @ E + c - E)
        T = T + dE / (2.0**HIS) / ONE
        T = T - T / (2.0**COOL_SHIFT)
        T[is_fire] = PLATEAU
        T = np.maximum(T, 0.0)
        if t % 240 == 0:
            hist.append((t/TICK_HZ, T[is_wall].mean()))
    s = T[is_wall]
    return s.mean(), s.min(), s.max(), (s >= IGNITION).mean(), hist

print("Task 2 -- FLASHOVER.  A 2x2 fire held at the measured 1556 K plateau in a sealed")
print("24x12 compartment lined with furniture; every surface radiates by its own")
print(f"temperature (R-K).  Ignition = {IGNITION:.0f} game units.  15 minutes of sim per row.\n")
print(f"{'k_leak':>7}{'f_lost':>8} | {'mean surf T':>12}{'min':>8}{'max':>8} | {'surfaces lit':>13}  verdict")
for k_leak, f in ((0.0, 0.0), (0.10, 0.0), (0.20, 0.0), (0.30, 0.0),
                  (0.0, 0.01), (0.0, 0.05), (0.0, 0.10), (0.0, 0.26)):
    m, lo, hi, frac, h = run(k_leak=k_leak, f_lost=f)
    v = "FLASHOVER" if frac > 0.9 else ("partial" if frac > 0.05 else "no flashover")
    print(f"{k_leak:7.2f}{f:8.2f} | {m:12.1f}{lo:8.1f}{hi:8.1f} | {100*frac:11.0f}%   {v}")

print("\ntime course at the derived leak (k=0.10), mean surface temperature, game units:")
_m, _lo, _hi, _f, h = run(k_leak=0.10, f_lost=0.0)
print("  " + "  ".join(f"{t:.0f}s:{v:.0f}" for t, v in h[::3]))

print("\n--- section 4b made quantitative: how much UNACCOUNTED transport loss kills flashover?")
print(f"{'f_lost':>8} {'mean surf T':>12} {'vs f=0':>9} {'surfaces lit':>14}")
base = None
for f in (0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05):
    m, lo, hi, frac, _ = run(k_leak=0.0, f_lost=f)
    if base is None: base = m
    print(f"{f:8.3f} {m:12.1f} {m/base:8.2f}x {100*frac:13.0f}%")

print("\n--- is the conclusion scheme-dependent?  (k_leak = 0.10, the derived value)")
print(f"{'scheme':>12} {'mean surf T':>12}{'min':>8}{'max':>8} {'surfaces lit':>14}")
for th, nm in ((0.0, "step"), (0.5, "blend 0.50"), (1.0, "shear")):
    m, lo, hi, frac, _ = run(k_leak=0.10, f_lost=0.0, theta=th)
    print(f"{nm:>12} {m:12.1f}{lo:8.1f}{hi:8.1f} {100*frac:13.0f}%")
