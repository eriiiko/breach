"""Task 1 -- the solvers on a REAL heatmap from the running sim (2026-09-13).

Erik: "I'd like to look at these solvers one more time, but using an actual heatmap
from a game sim as input... we would need to start a few fires, or even fake such a
heatmap. Also have some flashlights perhaps."  And: "I could accept the artifacts, but
I'd like to see them with more light sources first."

Input is heatmap_<level>.npz, dumped from the real Simulation after real fires burned
for a real minute (four ignition sites on levels/playground, 117 burning tiles at the
dump, peak 977 game = 1270 K).  No synthetic point source anywhere in this file.

WHAT THE COLUMNS ARE
    exact       analytic E/(2*pi*r) with visibility -- the yardstick, no discretisation
    step S16    the transport step the design specifies (section 2.4)
    shear S16   critique 1's proposed fix (required change 6)
    + leak      the same, with the derived out-of-plane leak k = 0.10 for a 2.5 m deck

Run:
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/heatmap_render.py
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE))

RAD_SCALE = 5.1427e-5
K_AMB = 293.0
K_LEAK = 0.10                 # derived: ln(1 + r^2/a^2)/(2r) for a 2.5 m deck
LEVEL = "playground"

PALETTE = ["#05050c", "#2a1146", "#7b1d3f", "#d2492a", "#f4a93a", "#fff3c4"]
WALL_COLOUR = "#3a3f4a"
GAMMA = 0.42
CLIP_PCT = 99.5

# --- the reference sweep, float, paired bookkeeping -------------------------
def ordinates(n=16, phase=0.0):
    ang = (np.arange(n) + 0.5) * (2*math.pi/n) + phase
    return np.cos(ang), np.sin(ang), 1.0/n

def sweep(a, E, kleak=0.0, E_amb=0.0, n=16, phase=0.0, theta=1.0):
    """theta = 0 -> step (two-face cosine split); theta = 1 -> shear."""
    h, w = a.shape
    kl = np.broadcast_to(np.asarray(kleak, float), (h, w))
    mus, etas, wt = ordinates(n, phase)
    fluence = np.zeros((h, w))
    for mu, eta in zip(mus, etas):
        am, ae = abs(mu), abs(eta)
        sx, sy = (1 if mu > 0 else -1), (1 if eta > 0 else -1)
        x_major = am >= ae
        f = (ae/am) if x_major else (am/ae)
        fx_w = am/(am+ae)
        inflow = np.zeros((h, w))
        src = E * wt
        xs = range(w) if mu > 0 else range(w-1, -1, -1)
        ys = range(h) if eta > 0 else range(h-1, -1, -1)
        order = (((y, x) for x in xs for y in ys) if x_major
                 else ((y, x) for y in ys for x in xs))
        for y, x in order:
            i_in = inflow[y, x]
            stream = i_in * (1.0 - kl[y, x]) + E_amb * wt * kl[y, x]
            ai = a[y, x]
            i_out = stream - stream*ai + src[y, x]*ai
            fluence[y, x] += stream
            nx, ny = x+sx, y+sy
            def push(ty, tx, amt):
                if 0 <= tx < w and 0 <= ty < h:
                    inflow[ty, tx] += amt
            i_step = i_out * (1.0 - theta)
            i_sh = i_out - i_step
            sfx = i_step * fx_w
            push(y, nx, sfx); push(ny, x, i_step - sfx)
            ha = i_sh * (1.0 - f)
            if x_major:
                push(y, nx, ha); push(ny, nx, i_sh - ha)
            else:
                push(ny, x, ha); push(ny, nx, i_sh - ha)
    return fluence

def solve_exact(E, opaque, chunk=64):
    """Analytic 2D yardstick: sum over emitters of E/(2*pi*r), with visibility.

    Vectorised over the whole grid per source -- the naive triple loop is ~10^8
    Python steps on a 70x100 map with 500 emitters and does not finish.
    """
    h, w = E.shape
    out = np.zeros((h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    op = opaque.astype(np.uint8)
    srcs = np.argwhere(E > 0)
    K = 96                                     # samples along each segment
    ts = (np.arange(1, K) / K)[:, None, None]  # (K-1, 1, 1)
    for n, (sy, sx) in enumerate(srcs):
        e = float(E[sy, sx])
        dy, dx = yy - sy, xx - sx
        r = np.hypot(dy, dx)
        vis = r >= 0.75
        py = np.clip(np.rint(sy + dy[None] * ts), 0, h - 1).astype(np.intp)
        px = np.clip(np.rint(sx + dx[None] * ts), 0, w - 1).astype(np.intp)
        blocked = op[py, px]
        # the source tile itself never blocks
        blocked &= ~((py == sy) & (px == sx))
        vis &= ~(blocked.any(axis=0).astype(bool))
        vis &= ~opaque
        out[vis] += e / (2 * np.pi * r[vis])
    return out


def tonemap(f, opaque, gamma=GAMMA, hi=None):
    v = np.array(f, float)
    if hi is None:
        hi = np.percentile(v[~opaque], CLIP_PCT)
    v = np.clip(v/max(hi, 1e-12), 0, 1) ** gamma
    return np.ma.array(v, mask=opaque)

def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    d = np.load(HERE / f"heatmap_{LEVEL}.npz", allow_pickle=True)
    T = d["temperature"].astype(float)
    a = d["heat_atten"].astype(float)
    solid = d["solid"]
    E = np.where(a > 0, RAD_SCALE * (K_AMB + np.maximum(T, 0.0))**4, 0.0)
    E_amb = RAD_SCALE * K_AMB**4
    opaque = solid & (a >= 0.99)
    print(f"{LEVEL}: {T.shape}, {int((T>100).sum())} tiles over 100 game, "
          f"peak {T.max():.0f} game ({K_AMB+T.max():.0f} K)")

    # R-K: EVERY cell with a > 0 radiates by its own temperature, so the yardstick
    # must include all of them, not just the fires -- otherwise it is not the same
    # problem the sweeps are solving and the comparison is meaningless.
    E_src = np.where(a > 0, E * a, 0.0)
    print(f"  {int((E_src>0).sum())} emitting tiles for the analytic yardstick "
          f"(every cell with a>0, per R-K)")

    ex = solve_exact(E_src, opaque)
    ex += E_amb * ~opaque      # the ambient sky floor the sweeps also carry
    cols = [("exact (analytic yardstick)", ex),
            ("step S16  (design section 2.4)", sweep(a, E, E_amb=E_amb, theta=0.0)),
            ("shear S16  (critique 1 fix 6)", sweep(a, E, E_amb=E_amb, theta=1.0)),
            (f"shear S16 + out-of-plane leak {K_LEAK}",
             sweep(a, E, kleak=K_LEAK, E_amb=E_amb, theta=1.0))]

    fire = LinearSegmentedColormap.from_list("fire", PALETTE)
    fire.set_bad(WALL_COLOUR)
    fig, axes = plt.subplots(2, 4, figsize=(21, 11))
    ref_hi = np.percentile(cols[0][1][~opaque], CLIP_PCT)
    for c, (name, f) in enumerate(cols):
        axes[0, c].imshow(tonemap(f, opaque, hi=np.percentile(f[~opaque], CLIP_PCT)),
                          cmap=fire, vmin=0, vmax=1, interpolation="nearest")
        axes[0, c].set_title(name, fontsize=11)
        axes[0, c].set_xticks([]); axes[0, c].set_yticks([])
        hi = np.percentile(f[~opaque], CLIP_PCT)
        lg = np.where(opaque, np.nan, np.log10(np.maximum(f/max(hi, 1e-12), 1e-6)))
        im = axes[1, c].imshow(lg, cmap="magma", vmin=-4.5, vmax=0,
                               interpolation="nearest")
        axes[1, c].set_xticks([]); axes[1, c].set_yticks([])
        if c == 3:
            fig.colorbar(im, ax=axes[1, c], fraction=0.046, label="log10 (normalised)")
    axes[0, 0].set_ylabel("tone-mapped", fontsize=10)
    axes[1, 0].set_ylabel("log10 -- where the artifacts live", fontsize=10)
    fig.suptitle(
        f"The HEAT solvers on a real sim heatmap -- levels/{LEVEL}, four fires, 60 s of "
        f"real simulation\n"
        f"117 burning tiles, peak {K_AMB+T.max():.0f} K. Each panel normalised on its own "
        f"99.5th percentile: this is about SHAPE, not scale.", fontsize=13)
    fig.tight_layout()
    fig.savefig(HERE / "heatmap_render.png", dpi=110)
    print("wrote", HERE / "heatmap_render.png")
    # Agreement with the yardstick, for the two schemes solving the SAME problem.
    # The leak column deliberately solves a different one (it exports out of plane),
    # so scoring it against a no-leak yardstick would be a category error.
    print("")
    print(f"{'scheme':<40}{'mean |log10 ratio| vs exact':>30}")
    m = (~opaque) & (cols[0][1] > 0)
    for name, f in cols[1:3]:
        r = np.log10(np.maximum(f[m], 1e-12) / np.maximum(cols[0][1][m], 1e-12))
        print(f"{name:<40}{np.abs(r).mean():>30.3f}")
    print("    (the leak column solves a different problem by design -- not scored here)")

if __name__ == "__main__":
    main()
