"""Ray engine v2 — spatial-scheme bake-off (design study, 2026-09-13).

Erik's question: "is it possible to test the alternatives in a simple python
render, only rendering one frame, to see how the different methods compare?"

This is that. One frame, no engine, numpy + matplotlib only.

WHAT IS BEING DECIDED
    docs/ray_engine_v2_design_2026-09-12.md §2.4 picks STEP (upwind) differencing
    for the heat sweep, on the grounds of positivity and exact conservation, and
    claims its numerical diffusion is "comfortably below the tile quantisation we
    already accept". I doubted my own claim: transverse spread in a step scheme
    grows like sqrt(distance), which at our ~3-tile ignition reach is ~1.7 tiles
    of smear. This measures it instead of arguing about it.

THE THREE-WAY, chosen so the two error sources SEPARATE
    exact  — the analytic solution of the continuous problem. In a transparent
             2D medium each emitter contributes E/(2*pi*r) at an unoccluded
             receiver and nothing at an occluded one. NO angular discretisation,
             NO spatial discretisation. The yardstick.
    long   — LONG CHARACTERISTICS at S16. Each cell integrates backwards along
             each of the 16 ordinates to the grid edge, accumulating emission.
             Carries the ANGULAR error (the ray effect) and essentially NO
             spatial error. Isolates "how much does S16 alone cost us".
    step   — first-order upwind finite volume at S16, what the design specifies.
             Carries the angular error AND the spatial diffusion. The gap
             between `long` and `step` is exactly the price of the spatial
             scheme, which is the number the decision turns on.

    Reading the result: if step ~ long, the spatial scheme is free and the design
    stands. If step is much worse than long, the design needs a sharper spatial
    scheme (short characteristics, or the lumped Linear Characteristic family
    which satisfies corner balance and so is conservative AND low-diffusion).

NORMALISATION (got this wrong on the first pass; stated explicitly now)
    RTE:      s.grad(I) + kappa*I = q      q = emission per unit area per steradian
    2D angle: full circle is 2*pi, so ordinate weights are w_m = 2*pi/N.
    A cell emitting total power E is isotropic, so q_m = E / (2*pi) for every m.
    Fluence:  Phi = sum_m I_m * w_m.
    Free-space check: Phi(r) must equal E/(2*pi*r), and every scheme must land on
    the SAME constant. Disagreement there means a normalisation bug, not a
    scheme property — that is how the first version of this file was caught.

Run:
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/scheme_study.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
N_ORD = 16
TILE_M = 0.333
TWO_PI = 2.0 * math.pi


def ordinates(n=N_ORD, phase=0.0):
    """n evenly spaced directions, cell-centred in angle. Weights sum to 2*pi."""
    ang = (np.arange(n) + 0.5) * (TWO_PI / n) + phase
    return np.cos(ang), np.sin(ang), TWO_PI / n, ang


# ---------------------------------------------------------------------------
def solve_step(kappa, source, n=N_ORD, phase=0.0):
    """Upwind finite volume. Per-ordinate balance over a unit cell:

        |mu|*(I - I_xup) + |eta|*(I - I_yup) + kappa*I = q_m
        =>  I = (|mu|*I_xup + |eta|*I_yup + q_m) / (|mu| + |eta| + kappa)

    Positive whenever its inputs are; conservative by construction.
    """
    h, w = kappa.shape
    mus, etas, wt, _ = ordinates(n, phase)
    q = source / TWO_PI                      # per steradian, isotropic
    fluence = np.zeros((h, w))
    absorbed = np.zeros((h, w))
    escaped = 0.0
    for mu, eta in zip(mus, etas):
        I = np.zeros((h, w))
        am, ae = abs(mu), abs(eta)
        xs = range(w) if mu > 0 else range(w - 1, -1, -1)
        ys = range(h) if eta > 0 else range(h - 1, -1, -1)
        dx = -1 if mu > 0 else 1
        dy = -1 if eta > 0 else 1
        for y in ys:
            for x in xs:
                xu, yu = x + dx, y + dy
                i_xu = I[y, xu] if 0 <= xu < w else 0.0
                i_yu = I[yu, x] if 0 <= yu < h else 0.0
                I[y, x] = (am * i_xu + ae * i_yu + q[y, x]) / (am + ae + kappa[y, x])
        fluence += I * wt
        absorbed += kappa * I * wt
        col = I[:, -1] if mu > 0 else I[:, 0]
        row = I[-1, :] if eta > 0 else I[0, :]
        escaped += (col.sum() * am + row.sum() * ae) * wt
    return fluence, absorbed, escaped



# ---------------------------------------------------------------------------
def solve_shear(kappa, source, n=N_ORD, phase=0.0):
    """SHEAR / characteristic transport step (critique 1's proposed fix).

    Instead of splitting the upwind value between the x and y neighbours, advance
    ONE FULL CELL along the dominant axis and interpolate the transverse
    fraction f = |minor/major| between the two cells there. Every cell is still
    visited exactly once per ordinate, so the cost is identical to step.

    Why it is sharper: for an axis-aligned ordinate f = 0 and for a 45-degree
    ordinate f = 1, and BOTH are exact shifts with no interpolation at all. Only
    intermediate angles diffuse, and only transversally, once per cell. Step
    differencing by contrast mixes the x and y upwind values at every angle.

    Conservation: the two interpolation weights are non-negative and sum to 1, so
    transport preserves the stream exactly; absorption and emission are the same
    paired per-cell terms as in solve_step.
    """
    h, w = kappa.shape
    mus, etas, wt, _ = ordinates(n, phase)
    q = source / TWO_PI
    fluence = np.zeros((h, w))
    absorbed = np.zeros((h, w))
    escaped = 0.0
    for mu, eta in zip(mus, etas):
        I = np.zeros((h, w))
        am, ae = abs(mu), abs(eta)
        x_major = am >= ae
        f = (ae / am) if x_major else (am / ae)      # transverse fraction, 0..1
        sx = 1 if mu > 0 else -1
        sy = 1 if eta > 0 else -1
        xs = range(w) if mu > 0 else range(w - 1, -1, -1)
        ys = range(h) if eta > 0 else range(h - 1, -1, -1)
        for y in ys:
            for x in xs:
                if x_major:
                    xu = x - sx
                    if 0 <= xu < w:
                        ya, yb = y, y - sy
                        a = I[ya, xu]
                        b = I[yb, xu] if 0 <= yb < h else 0.0
                        i_up = a * (1.0 - f) + b * f
                    else:
                        i_up = 0.0
                    path = 1.0 / am
                else:
                    yu = y - sy
                    if 0 <= yu < h:
                        xa, xb = x, x - sx
                        a = I[yu, xa]
                        b = I[yu, xb] if 0 <= xb < w else 0.0
                        i_up = a * (1.0 - f) + b * f
                    else:
                        i_up = 0.0
                    path = 1.0 / ae
                k = kappa[y, x]
                if k > 1e-12:
                    t = math.exp(-k * path)
                    I[y, x] = i_up * t + (q[y, x] / k) * (1.0 - t)
                else:
                    I[y, x] = i_up + q[y, x] * path
        fluence += I * wt
        absorbed += kappa * I * wt
        col = I[:, -1] if mu > 0 else I[:, 0]
        row = I[-1, :] if eta > 0 else I[0, :]
        escaped += (col.sum() * am + row.sum() * ae) * wt
    return fluence, absorbed, escaped


# ---------------------------------------------------------------------------
def solve_long(kappa, source, n=N_ORD, phase=0.0, max_steps=200):
    """LONG CHARACTERISTICS. For each cell and each ordinate, walk backwards
    along the ray to the grid edge, accumulating emission attenuated by whatever
    it passed through. No spatial discretisation error beyond the tile itself —
    what remains is purely the angular (ray-effect) error of S16.

    Deliberately the expensive, obviously-correct method: this is a yardstick,
    not a candidate implementation.
    """
    h, w = kappa.shape
    mus, etas, wt, _ = ordinates(n, phase)
    q = source / TWO_PI
    fluence = np.zeros((h, w))
    step = 0.25                               # sub-tile march, plenty fine
    for mu, eta in zip(mus, etas):
        I = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                acc = 0.0
                trans = 1.0
                px, py = x + 0.5, y + 0.5
                for _ in range(max_steps):
                    px -= mu * step
                    py -= eta * step
                    ix, iy = int(px), int(py)
                    if not (0 <= ix < w and 0 <= iy < h):
                        break
                    k = kappa[iy, ix]
                    acc += q[iy, ix] * trans * step
                    trans *= math.exp(-k * step)
                    if trans < 1e-6:
                        break
                I[y, x] = acc
        fluence += I * wt
    return fluence


# ---------------------------------------------------------------------------
def solve_exact(source, opaque):
    """Analytic: E/(2*pi*r) from every unoccluded emitter. No discretisation."""
    h, w = source.shape
    out = np.zeros((h, w))
    for (ey, ex) in np.argwhere(source > 0):
        e = source[ey, ex]
        for y in range(h):
            for x in range(w):
                if (y, x) == (ey, ex) or not _visible(opaque, ey, ex, y, x):
                    continue
                out[y, x] += e / (TWO_PI * math.hypot(y - ey, x - ex))
    return out


def _visible(opaque, y0, x0, y1, x1):
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if (x, y) != (x0, y0) and (x, y) != (x1, y1) and opaque[y, x]:
            return False
        if x == x1 and y == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


# ---------------------------------------------------------------------------
def scene_point(size=49):
    kappa = np.zeros((size, size))
    source = np.zeros((size, size))
    opaque = np.zeros((size, size), dtype=bool)
    c = size // 2
    source[c, c] = 1.0
    return kappa, source, opaque, (c, c)


def scene_shadow(size=49):
    """Emitter, then an opaque wall with a 7-tile gap. Measures shadow edges."""
    kappa = np.zeros((size, size))
    source = np.zeros((size, size))
    opaque = np.zeros((size, size), dtype=bool)
    c = size // 2
    source[c, 8] = 1.0
    wx = 18
    opaque[:, wx] = True
    opaque[c - 3:c + 4, wx] = False
    kappa[opaque] = 50.0                      # optically thick over one tile
    return kappa, source, opaque, (c, 8), wx


# ---------------------------------------------------------------------------
def ring(field, cy, cx, r, n=180):
    out = []
    for k in range(n):
        a = TWO_PI * k / n
        y, x = cy + r * math.sin(a), cx + r * math.cos(a)
        y0, x0 = int(math.floor(y)), int(math.floor(x))
        fy, fx = y - y0, x - x0
        v = 0.0
        for dy in (0, 1):
            for dx in (0, 1):
                wgt = (fy if dy else 1 - fy) * (fx if dx else 1 - fx)
                yy, xx = y0 + dy, x0 + dx
                if 0 <= yy < field.shape[0] and 0 <= xx < field.shape[1]:
                    v += wgt * field[yy, xx]
        out.append(v)
    return np.array(out)


def ripple(p):
    m = p.mean()
    return 0.0 if m <= 0 else 100.0 * (p.max() - p.min()) / m


def edge_width(prof, axis):
    lo, hi = prof.min(), prof.max()
    if hi <= lo:
        return float("nan")
    t90, t10 = lo + 0.9 * (hi - lo), lo + 0.1 * (hi - lo)
    a90 = a10 = None
    for v, a in zip(prof, axis):
        if a90 is None and v <= t90:
            a90 = a
        if a10 is None and v <= t10:
            a10 = a
    return float("nan") if a90 is None or a10 is None else abs(a10 - a90)


# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 74)
    print("RAY ENGINE v2 - SPATIAL SCHEME BAKE-OFF  (S16, 49x49, tile 0.333 m)")
    print("=" * 74)

    kappa, source, opaque, (cy, cx) = scene_point()
    f_exact = solve_exact(source, opaque)
    f_long = solve_long(kappa, source)
    f_step, _, _ = solve_step(kappa, source)
    f_shear, _, _ = solve_shear(kappa, source)
    nrot = N_ORD
    f_long_rot = np.zeros_like(f_long)
    f_step_rot = np.zeros_like(f_step)
    f_shear_rot = np.zeros_like(f_shear)
    for k in range(nrot):
        ph = k * (TWO_PI / (N_ORD * N_ORD))
        f_long_rot += solve_long(kappa, source, phase=ph) / nrot
        f_step_rot += solve_step(kappa, source, phase=ph)[0] / nrot
        f_shear_rot += solve_shear(kappa, source, phase=ph)[0] / nrot

    rows = [("exact (no discretisation)", f_exact),
            ("long char S16 (angular only)", f_long),
            ("long char S16 + rotation", f_long_rot),
            ("step S16 (angular+spatial)", f_step),
            ("step S16 + rotation", f_step_rot),
            ("shear S16 (angular+spatial)", f_shear),
            ("shear S16 + rotation", f_shear_rot)]

    print("\nNORMALISATION CHECK - mean ring fluence x r; all schemes must agree")
    print("  (flat across r == 1/r falloff; equal across schemes == same scale)")
    print(f"  {'scheme':<30}{'r=2':>10}{'r=4':>10}{'r=8':>10}")
    for name, f in rows:
        v = [ring(f, cy, cx, r).mean() * r for r in (2, 4, 8)]
        print(f"  {name:<30}" + "".join(f"{x:>10.4f}" for x in v))
    print(f"  {'analytic value E/(2pi)':<30}{1/TWO_PI:>10.4f}"
          f"{1/TWO_PI:>10.4f}{1/TWO_PI:>10.4f}")

    print("\nTEST 1 - ISOTROPY (single emitter, empty space)")
    print("  ring ripple, peak-to-peak as % of mean. 0% = perfectly isotropic.")
    print(f"  {'scheme':<30}{'r=3':>10}{'r=8':>10}")
    for name, f in rows:
        print(f"  {name:<30}{ripple(ring(f, cy, cx, 3)):>9.1f}%"
              f"{ripple(ring(f, cy, cx, 8)):>9.1f}%")

    k2, s2, o2, (ey, ex), wx = scene_shadow()
    g_exact = solve_exact(s2, o2)
    g_long = solve_long(k2, s2)
    g_step, ab_step, es_step = solve_step(k2, s2)
    g_shear, ab_shear, es_shear = solve_shear(k2, s2)
    probe = 30
    ys = np.arange(k2.shape[0])

    print(f"\nTEST 2 - SHADOW SHARPNESS (cut at x={probe}, {probe-wx} tiles past"
          f" the wall)")
    print("  90%->10% edge width. Lower is sharper.")
    print(f"  {'scheme':<30}{'tiles':>10}{'metres':>10}")
    for name, g in (("exact (no discretisation)", g_exact),
                    ("long char S16", g_long),
                    ("step S16", g_step),
                    ("shear S16", g_shear)):
        ew = edge_width(g[ey:, probe], ys[ey:])
        print(f"  {name:<30}{ew:>10.2f}{ew*TILE_M:>10.2f}")

    print("\nTEST 3 - CONSERVATION (shadow scene)")
    emitted = s2.sum()
    for nm, ab, es in (("step S16", ab_step, es_step),
                       ("shear S16", ab_shear, es_shear)):
        tot = ab.sum() + es
        print(f"  {nm:<12} emitted={emitted:.5f}  absorbed+escaped={tot:.5f}  "
              f"residual={100*(emitted-tot)/emitted:+.4f}%")

    print("\nTEST 4 - DOES ORDINATE COUNT RESCUE THE WINNER?")
    print("  shear, no rotation. ring ripple (% of mean) and shadow edge (tiles)")
    print(f"  {'quadrature':<18}{'r=3':>9}{'r=8':>9}{'shadow':>9}")
    for nn in (16, 32, 64):
        fp, _, _ = solve_shear(kappa, source, n=nn)
        gp, _, _ = solve_shear(k2, s2, n=nn)
        ew = edge_width(gp[ey:, probe], ys[ey:])
        print(f"  shear S{nn:<11}{ripple(ring(fp, cy, cx, 3)):>8.1f}%"
              f"{ripple(ring(fp, cy, cx, 8)):>8.1f}%{ew:>9.2f}")
    print(f"  {'step S16 (ref)':<18}{ripple(ring(f_step, cy, cx, 3)):>8.1f}%"
          f"{ripple(ring(f_step, cy, cx, 8)):>8.1f}%"
          f"{edge_width(g_step[ey:, probe], ys[ey:]):>9.2f}")

    # ---------------- figures ----------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (nm, f) in zip(axes[0], (("exact", f_exact), ("step S16", f_step),
                                     ("shear S16", f_shear))):
        im = ax.imshow(np.log10(np.maximum(f, 1e-6)), cmap="inferno",
                       vmin=-4, vmax=-1)
        ax.set_title(f"point source - {nm}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="log10 fluence")
    for ax, (nm, g) in zip(axes[1], (("exact", g_exact), ("step S16", g_step),
                                     ("shear S16", g_shear))):
        im = ax.imshow(np.log10(np.maximum(g, 1e-6)), cmap="inferno",
                       vmin=-4, vmax=-1)
        ax.axvline(wx, color="cyan", lw=0.8, ls="--")
        ax.axvline(probe, color="white", lw=0.8)
        ax.set_title(f"wall + gap - {nm}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="log10 fluence")
    fig.suptitle("Ray engine v2 - spatial scheme bake-off (S16, 49x49, tile "
                 "0.333 m)\ntop: isotropy    bottom: shadow  "
                 "(dashed = wall, solid = profile cut)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "scheme_fields.png", dpi=130)

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    degs = np.degrees(np.linspace(0, TWO_PI, 180, endpoint=False))
    for nm, f in rows:
        p = ring(f, cy, cx, 3)
        a1.plot(degs, p / p.mean(), lw=1.3, label=nm)
    a1.set_title("isotropy: fluence around a ring at r = 3 tiles (normalised)")
    a1.set_xlabel("angle (deg)"); a1.set_ylabel("fluence / mean")
    a1.legend(fontsize=7); a1.grid(alpha=0.3)
    for nm, g in (("exact", g_exact), ("long char S16", g_long),
                  ("step S16", g_step), ("shear S16", g_shear)):
        p = g[:, probe]
        a2.plot(ys - ey, p / max(p.max(), 1e-12), lw=1.6, label=nm)
    a2.set_title(f"shadow edge: normalised cut at x={probe}")
    a2.set_xlabel("tiles from the gap centre"); a2.set_ylabel("normalised fluence")
    a2.set_xlim(-16, 16); a2.legend(fontsize=8); a2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT / "scheme_profiles.png", dpi=130)
    print(f"\nwrote {OUT/'scheme_fields.png'}\nwrote {OUT/'scheme_profiles.png'}")


if __name__ == "__main__":
    main()
