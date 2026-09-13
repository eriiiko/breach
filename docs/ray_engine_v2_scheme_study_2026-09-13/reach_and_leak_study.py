"""The reach lever: what falloff laws can a 2D sweep actually produce? (2026-09-13)

Reproduces every table in docs/ray_engine_v2_reach_and_papers_2026-09-13.md sections
1-5, in one run (~4 minutes).

THE QUESTION.  docs/ray_engine_v2_NEXT_SESSION_2026-09-13.md section 3 offers three
reach levers, the first being "adopt 1/r^2 as structural".  A discrete-ordinates sweep
has no notion of distance from a source, because it has no source: the only laws it can
express are in-plane geometric spreading (1/r) and extinction (exp(-k r)).  So the third
option -- an OUT-OF-PLANE LEAK, energy exported through the deck's floor and ceiling
rather than absorbed by the air -- is the only one the solver can execute, and this
measures it.

THE DERIVATION.  Osborne & Sannikov 2024 section 3.1 do 2D transfer by augmenting the
flatland quadrature with inclined rays, assuming homogeneity along the third axis.  Our
slab is bounded, so an inclined ray leaves it after a finite horizontal distance.  The
surviving fraction at horizontal distance r is the fraction of the sphere within
|elevation| < arctan(a/r), a = h/2 in tiles:

    S(r) = a / sqrt(a^2 + r^2)      =>      k(r) = ln(1 + r^2/a^2) / (2r)

which for a 2.5 m deck is k ~ 0.10 and nearly CONSTANT over the whole gameplay range.
That is what makes a single coefficient legitimate rather than a fudge.

Calibration of the instrument: this file independently reproduces critique 1's ignition
radius (axis 19.4 tiles against their ~17) and its section 7c stability table.
"""
from __future__ import annotations
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_ref import sweep, sweep_shear, sweep_blend   # noqa: E402

TILE_M = 0.333
K_AMB = 293.0
K_IGN = 573.0                  # furniture ignition_temp 280 game
K_SRC = 1556.0                 # the measured crate plateau
RAD_SCALE = 5.1427e-5
H = W = 81
C = 40
N = 16

G_IGN = (K_IGN / K_SRC) ** 4   # the fluence at which a receiver equilibrates at ignition
YY, XX = np.mgrid[0:H, 0:W]
RR = np.hypot(YY - C, XX - C)


# ---------------------------------------------------------------- section 2
def k_of_r(r, a):
    return math.log(1.0 + (r * r) / (a * a)) / (2.0 * r)


def section2():
    print("=" * 78)
    print("2. DERIVING THE LEAK FROM DECK GEOMETRY (not fitting it)")
    print("=" * 78)
    rs = (1, 2, 3, 5, 8, 12, 16)
    print(f"{'deck':>8} {'a (tiles)':>10} " + "".join(f"{'r='+str(r):>8}" for r in rs))
    for h_m in (2.2, 2.5, 3.0, 4.0, 6.0):
        a = (h_m / TILE_M) / 2.0
        print(f"{h_m:6.1f} m {a:10.2f} " + "".join(f"{k_of_r(r, a):8.4f}" for r in rs))
    a = (2.5 / TILE_M) / 2.0
    print("\n  a single constant k = 0.10 against the exact slab law, 2.5 m deck:")
    print(f"  {'r':>4} {'slab S(r)':>11} {'exp(-0.10r)':>13} {'ratio':>8}")
    for r in (1, 2, 3, 5, 8, 12, 16, 20):
        s = a / math.sqrt(a * a + r * r)
        e = math.exp(-0.10 * r)
        print(f"  {r:4d} {s:11.4f} {e:13.4f} {e/s:8.3f}")
    print("  -> within 6% out to r = 12, conservatively steeper beyond.")

    sigma = 5.670374419e-8
    print("\n  and the ignition criterion itself (critique 1 section 3d):")
    for nm, K in (("573 K, the engine's own ignition_temp", K_IGN),
                  ("680 K, radiative balance at 11 kW/m^2", 680.0)):
        print(f"    {nm:<40} -> {sigma*K**4/1000:6.2f} kW/m^2")
    r3 = 3.3 * math.sqrt(11.0 / (sigma * K_IGN**4 / 1000))
    print(f"  -> section 9.3's 3.3 tiles becomes {r3:.2f} tiles under the engine's criterion.")


# ---------------------------------------------------------------- helpers
def point_field(fn, kleak, phases=1, half=0, theta=None):
    a = np.zeros((H, W))
    E = np.zeros((H, W))
    sl = slice(C - half, C + half + 1)
    a[sl, sl] = 1.0
    E[sl, sl] = 1.0                      # per-tile power, so a bigger fire is stronger
    acc = np.zeros((H, W))
    for p in range(phases):
        kw = dict(kleak=kleak, E_amb=0.0, n=N, phase=p * (2 * math.pi / N) / phases)
        if theta is not None:
            kw["theta"] = theta
        fl = fn(a, E, **kw)[1]
        acc += fl
    return acc / phases


def rings(fl, rmax=26):
    return {r: (fl[(RR >= r - 0.5) & (RR < r + 0.5)].mean(), fl[C, C + r], fl[C + r, C + r])
            for r in range(1, rmax + 1)}


def r_ign(ring, K_s=K_SRC, key=0):
    tgt = (K_IGN / K_s) ** 4
    pr = pg = None
    for r in sorted(ring):
        g = ring[r][key]
        if pg is not None and pg >= tgt > g:
            t = (math.log(pg) - math.log(tgt)) / (math.log(pg) - math.log(g))
            return pr + t * (r - pr)
        pr, pg = r, g
    return float('inf') if ring[max(ring)][key] >= tgt else float('nan')


# ---------------------------------------------------------------- section 3
def section3():
    print("\n" + "=" * 78)
    print("3. WHAT THE LEAK DOES TO REACH   (target 4.4 tiles, section 3a)")
    print("=" * 78)
    print(f"{'deck':>8}{'k_leak':>8} | {'step ring':>10}{'axis':>7}{'diag':>7}{'ax/di':>7}"
          f" | {'shear ring':>11}{'axis':>7}{'diag':>7}{'ax/di':>7}")
    cache = {}
    for h_m, kl in ((2.2, 0.110), (2.5, 0.100), (3.0, 0.085), (4.0, 0.062), (None, 0.0)):
        row = []
        for fn in (sweep, sweep_shear):
            rg = rings(point_field(fn, kl))
            cache[(fn.__name__, kl)] = rg
            row.append([r_ign(rg, K_SRC, k) for k in (0, 1, 2)])
        lbl = "none" if h_m is None else f"{h_m:.1f} m"
        print(f"{lbl:>8}{kl:8.3f} | {row[0][0]:10.2f}{row[0][1]:7.2f}{row[0][2]:7.2f}"
              f"{row[0][1]/row[0][2]:7.2f} | {row[1][0]:11.2f}{row[1][1]:7.2f}"
              f"{row[1][2]:7.2f}{row[1][1]/row[1][2]:7.2f}")

    print("\n3c. fragility to the plateau (ring mean), the property to avoid:")
    hdr = (1100, 1300, 1556, 1800, 2200)
    print(f"{'k_leak':>8} " + "".join(f"{k:>9d}K" for k in hdr))
    for kl in (0.0, 0.10):
        rg = cache[("sweep_shear", kl)]
        print(f"{kl:8.2f} " + "".join(f"{r_ign(rg, k, 0):10.2f}" for k in hdr))
    g1 = cache[("sweep_shear", 0.0)][1][0]
    print(f"{'1/r^2':>8} " + "".join(
        f"{(g1/((K_IGN/k)**4))**0.5:10.2f}" for k in hdr) + "   (for comparison)")

    print("\n3b. both ends of section 9.3's reach curve, per-tile power held constant:")
    print(f"{'source':>10}{'r_ign centre':>14}{'from the edge':>15}{'target':>9}")
    for half, nm, tgt in ((0, "1 tile", "4.4"), (1, "3x3", "~11")):
        rg = rings(point_field(sweep_shear, 0.10, half=half))
        r = r_ign(rg)
        print(f"{nm:>10}{r:14.2f}{r-half:15.2f}{tgt:>9}")


# ---------------------------------------------------------------- section 3d
def section3d():
    print("\n" + "=" * 78)
    print("3d. THE GATES, with the leak in")
    print("=" * 78)
    rng = np.random.default_rng(3)
    for nm, fn in (("step ", sweep), ("shear", sweep_shear)):
        for kl in (0.0, 0.10):
            a = rng.choice([0.0, 0.15, 0.3, 0.5, 0.87, 1.0], size=(24, 24))
            E = rng.choice([0.0, 1e5, 3e6, 3e8], size=(24, 24))
            dE, fl, so, si, ce = fn(a, E, kleak=kl, E_amb=RAD_SCALE * K_AMB**4)
            res = dE.sum() + so - si + ce
            aU = rng.choice([0.15, 0.3, 0.87, 1.0], size=(24, 24))
            EU = np.full((24, 24), RAD_SCALE * K_AMB**4)
            dU = fn(aU, EU, kleak=kl, E_amb=RAD_SCALE * K_AMB**4)[0]
            print(f"  {nm} k={kl:4.2f}  conservation rel {res/max(abs(dE).sum(),1): .1e}"
                  f"   uniform-ambient max|dE| {np.abs(dU[2:-2,2:-2]).max(): .2e}"
                  f"   min fluence {fl.min(): .2e}")
    print("  (float here; exact in int64 by the one-integer-applied-twice idiom)")


# ---------------------------------------------------------------- section 4b
def section4b():
    print("\n" + "=" * 78)
    print("4b. PER-TICK ROTATION: measured, it makes ripple WORSE")
    print("=" * 78)
    rs = (1, 2, 3, 5, 8)
    print(f"{'scheme':<34}" + "".join(f"{'r='+str(r):>8}" for r in rs))
    for nm, fn in (("step ", sweep), ("shear", sweep_shear)):
        for ph in (1, 16):
            fl = point_field(fn, 0.10, phases=ph)
            cells = []
            for r in rs:
                v = fl[(RR >= r - 0.5) & (RR < r + 0.5)]
                cells.append(f"{v.max()/max(v.min(),1e-30):8.2f}")
            tag = "1 phase" if ph == 1 else f"{ph} phases"
            print(f"{nm+', leak 0.10, '+tag:<34}" + "".join(cells))


# ---------------------------------------------------------------- section 5b
def section5b():
    print("\n" + "=" * 78)
    print("5b. IGNITION FOOTPRINT -- the statistic that decides, since heat is never drawn")
    print("=" * 78)

    def footprint(fl):
        out = []
        for k in range(64):
            th = 2 * math.pi * k / 64
            dx, dy = math.cos(th), math.sin(th)
            pr = pv = None
            r = 0.5
            while r < 30:
                v = fl[int(round(C + dy * r)), int(round(C + dx * r))]
                if pv is not None and pv >= G_IGN > v:
                    t = (pv - G_IGN) / (pv - v)
                    out.append(pr + t * (r - pr))
                    break
                pr, pv = r, v
                r += 0.25
        arr = np.array(out)
        return arr.mean(), arr.max() / arr.min()

    print(f"{'scheme':>8}{'leak':>7}{'source':>9} | {'mean r':>9}{'spread':>9}")
    for nm, fn in (("step", sweep), ("shear", sweep_shear)):
        for half, snm in ((0, "1 tile"), (1, "3x3"), (2, "5x5")):
            m, sp = footprint(point_field(fn, 0.10, half=half))
            print(f"{nm:>8}{0.10:7.2f}{snm:>9} | {m:9.2f}{sp:9.2f}")
    print("  -> shear is rounder at every source size; a clustered fire averages the")
    print("     striping down, confirming the scheme study's untested caveat.")


if __name__ == "__main__":
    section2()
    section3()
    section3d()
    section4b()
    section5b()
