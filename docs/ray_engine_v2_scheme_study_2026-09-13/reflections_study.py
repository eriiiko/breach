"""Diffuse reflections on both methods, and what they cost (2026-09-13).

Erik: "please i'd like to see the reflections on both methods, and i'd like to know how
expensive they each are computationally."  And: "it seems to me that sharp edges from
flashlights etc are easier with the step and shear."

THE SCENE is built so that bounce is the ONLY thing that can light most of it: a
flashlight in room A pointed at a blank wall.  Without reflections everything outside
the beam is black; with them the room fills.  A second lamp sits behind the partition
so there is also an around-the-corner case.

HOW A BOUNCE WORKS, in both solvers.  A diffuse reflection is an isotropic re-emission
of what a surface received, scaled by its albedo.  That makes it a SCATTERING term, and
a scattering term couples the directions to each other -- which is exactly what design
section 2.2 relies on NOT having when it says one sweep is exact.  So reflections are
computed by source iteration:

    solve with the direct emitters          -> bounce 0
    surfaces re-emit albedo x what they got -> solve again -> bounce 1
    repeat

Each pass is one full solve and one more bounce, and the series converges like the
albedo, so albedo 0.5 is within a few per cent after three.  THE COST OF REFLECTIONS IS
THEREFORE (number of bounces + 1) TIMES THE COST OF THE SOLVER -- for either method.
That is the headline, and it is the same headline for both.

THE BOUNCE SOURCE is deliberately built identically for both solvers so the comparison
is apples to apples:

    bounce[cell] = albedo * fluence[cell] * (solid 4-neighbours / 4)

i.e. light passing next to a wall is partly scattered back into the room.  This is a
lumped surface-scattering model on a tile grid, not a radiosity solve, and it is a
DEMO approximation.  A real sweep implementation can do better and cheaper: the wall
cell already absorbs a known integer, so re-emitting albedo x that integer is native,
exactly conservative, and needs no neighbour gather.

Run (~6 minutes):
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/reflections_study.py
"""
from __future__ import annotations
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from room_render import H, W, tonemap                  # noqa: E402
import cascade_fix_study as CF                         # noqa: E402

N_ORD = 16
ALBEDO = 0.8        # a pale bulkhead; high on purpose so the effect is legible
GAMMA = 0.35
PALETTE = ["#05050c", "#141a30", "#2f3d63", "#6d7fa8", "#b9c4dc", "#ffffff"]
WALL_COLOUR = "#2b2f38"

# (row, col, beam direction deg, half-angle deg, power).  half=180 is a bare lamp.
LIGHTS = [(45, 8, 0.0, 14.0, 1.0),        # flashlight, pointed at the blank wall
          (10, 14, 180.0, 180.0, 0.25)]   # a small lamp behind the partition


def scene():
    opaque = np.zeros((H, W), dtype=bool)
    opaque[0, :] = opaque[-1, :] = True
    opaque[:, 0] = opaque[:, -1] = True
    opaque[:, 26] = True
    opaque[28:34, 26] = False
    opaque[:, 40] = True
    opaque[14:20, 40] = False
    opaque[20, 1:26] = True
    opaque[20, 6:11] = False
    kappa = np.where(opaque, 50.0, 0.0)
    return kappa, opaque


def wall_fraction(opaque):
    """How much of each open cell's boundary is wall -- the lumped bounce weight."""
    n = np.zeros((H, W))
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n += np.roll(np.roll(opaque.astype(float), dy, 0), dx, 1)
    return np.where(opaque, 0.0, n / 4.0)


# --------------------------------------------------------------------- sweep
def sweep_light(kappa, iso_src, lights, theta=1.0, n=N_ORD):
    """One directional sweep.  theta=0 is STEP, theta=1 is SHEAR.

    `iso_src` is isotropic emitted power per cell (the bounce source lives here);
    `lights` are cone emitters, which a directional solve handles natively by
    emitting only into the ordinates inside the beam.
    """
    a = 1.0 - np.exp(-kappa)                   # per-tile absorptivity
    ang = (np.arange(n) + 0.5) * (2 * math.pi / n)
    mus, etas, wt = np.cos(ang), np.sin(ang), 1.0 / n
    fluence = np.zeros((H, W))
    absorbed = np.zeros((H, W))
    for m in range(n):
        mu, eta = mus[m], etas[m]
        # per-ordinate emission: the isotropic part plus any cone that covers it
        src = iso_src * wt
        deg = math.degrees(ang[m])
        for (ly, lx, ldeg, half, amp) in lights:
            d = (deg - ldeg + 180.0) % 360.0 - 180.0
            if abs(d) <= half:
                src[ly, lx] += amp * wt * (180.0 / half)   # power kept constant
        am, ae = abs(mu), abs(eta)
        sx, sy = (1 if mu > 0 else -1), (1 if eta > 0 else -1)
        x_major = am >= ae
        f = (ae / am) if x_major else (am / ae)
        fx_w = am / (am + ae)
        inflow = np.zeros((H, W))
        xs = range(W) if mu > 0 else range(W - 1, -1, -1)
        ys = range(H) if eta > 0 else range(H - 1, -1, -1)
        order = (((y, x) for x in xs for y in ys) if x_major
                 else ((y, x) for y in ys for x in xs))
        for y, x in order:
            stream = inflow[y, x]
            ai = a[y, x]
            ab = stream * ai
            i_out = stream - ab + src[y, x]
            fluence[y, x] += stream
            absorbed[y, x] += ab
            nx, ny = x + sx, y + sy
            i_step = i_out * (1.0 - theta)
            i_sh = i_out - i_step
            sfx = i_step * fx_w
            for (ty, tx, amt) in ((y, nx, sfx), (ny, x, i_step - sfx)):
                if 0 <= tx < W and 0 <= ty < H:
                    inflow[ty, tx] += amt
            ha = i_sh * (1.0 - f)
            pair = ((y, nx, ha), (ny, nx, i_sh - ha)) if x_major else \
                   ((ny, x, ha), (ny, nx, i_sh - ha))
            for (ty, tx, amt) in pair:
                if 0 <= tx < W and 0 <= ty < H:
                    inflow[ty, tx] += amt
    return fluence, absorbed


# ------------------------------------------------------------------ cascades
def cascade_light(kappa, iso_src, lights, bounce_only=False):
    """Radiance cascades with the bilinear fix, with cone emitters."""
    em = iso_src / (2.0 * math.pi)
    CF.LIGHTS = lights
    old_trace = CF.trace_seg

    def trace_seg(p0, p1, em_, kappa_):
        """Same march, but a cone emitter contributes only if the segment's direction
        lies inside its beam (light travels from the cell toward the probe)."""
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return 0.0, 1.0
        nstep = max(1, int(length / CF.MARCH))
        ds = length / nstep
        ux, uy = dx / length, dy / length
        back = math.degrees(math.atan2(-uy, -ux))
        rad, trans = 0.0, 1.0
        for k in range(nstep):
            t = (k + 0.5) * ds
            ix, iy = int(x0 + ux * t), int(y0 + uy * t)
            if not (0 <= ix < W and 0 <= iy < H):
                break
            e = em_[iy, ix]
            for (ly, lx, ldeg, half, amp) in lights:
                if iy == ly and ix == lx:
                    d = (back - ldeg + 180.0) % 360.0 - 180.0
                    if abs(d) <= half:
                        e += amp / (2.0 * math.pi) * (180.0 / half)
            rad += e * trans * ds
            trans *= math.exp(-kappa_[iy, ix] * ds)
            if trans < 1e-5:
                break
        return rad, trans

    CF.trace_seg = trace_seg
    try:
        return CF.cascades(em, kappa, bilinear_fix=True)
    finally:
        CF.trace_seg = old_trace


# ---------------------------------------------------------------------- main
def bounce_series(solver, kappa, opaque, wf, nbounce):
    """Returns (list of cumulative fields, list of per-pass seconds)."""
    fields, secs = [], []
    total = np.zeros((H, W))
    src = np.zeros((H, W))
    lights = LIGHTS
    for b in range(nbounce + 1):
        t0 = time.perf_counter()
        f = solver(kappa, src, lights)
        secs.append(time.perf_counter() - t0)
        total = total + f
        fields.append(total.copy())
        src = ALBEDO * f * wf          # what the surfaces send back
        lights = []                    # the lamps only shine once
    return fields, secs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    kappa, opaque = scene()
    wf = wall_fraction(opaque)
    NB = 3

    print("shear sweep S16 ...")
    shear_f, shear_t = bounce_series(
        lambda k, s, l: sweep_light(k, s, l, theta=1.0)[0], kappa, opaque, wf, NB)
    print("step sweep S16 ...")
    step_f, step_t = bounce_series(
        lambda k, s, l: sweep_light(k, s, l, theta=0.0)[0], kappa, opaque, wf, NB)
    print("cascades + bilinear fix ...")
    casc_f, casc_t = bounce_series(
        lambda k, s, l: cascade_light(k, s, l), kappa, opaque, wf, 2)

    rows = [("shear sweep S16", shear_f, shear_t),
            ("step sweep S16", step_f, step_t),
            ("cascades + bilinear fix", casc_f, casc_t)]

    cmap = LinearSegmentedColormap.from_list("cool", PALETTE)
    cmap.set_bad(WALL_COLOUR)
    fig, axes = plt.subplots(3, 3, figsize=(16, 15.5))
    for r, (name, fields, secs) in enumerate(rows):
        direct = fields[0]
        full = fields[-1]
        bounce = full - direct
        hi = np.percentile(full[~opaque], 99.0)
        hib = max(np.percentile(bounce[~opaque], 99.0), 1e-12)
        nb = len(fields) - 1
        panels = [("direct light only (line of sight)", direct, hi),
                  (f"with {nb} bounce{'s' if nb > 1 else ''}", full, hi),
                  ("THE BOUNCED LIGHT ALONE (own scale)", bounce, hib)]
        for c, (sub, f, scale) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(tonemap(f, opaque, gamma=GAMMA, hi=scale), cmap=cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            lit = (f[~opaque] > 0.02 * hi).mean()
            extra = "" if c == 2 else f"  --  {100*lit:.0f}% of the deck lit"
            ax.set_title(f"{name}\n{sub}{extra}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "DIFFUSE REFLECTIONS on both methods.  A flashlight aimed at a blank wall, plus "
        "a lamp behind the partition.\n"
        "Column 1 is line of sight only.  Column 2 adds the bounces.  Column 3 is the "
        f"bounced light by itself.\nAlbedo {ALBEDO}.  Columns 1 and 2 share a scale "
        "within each row; column 3 has its own.", fontsize=14)
    fig.tight_layout()
    fig.savefig(HERE / "reflections_study.png", dpi=100)
    print("wrote", HERE / "reflections_study.png")

    print(f"\n{'solver':<26}{'per solve (s)':>15}{'passes for 3 bounces':>23}"
          f"{'total':>10}")
    for name, fields, secs in rows:
        per = sum(secs) / len(secs)
        print(f"{name:<26}{per:15.2f}{len(secs):>17} passes{per*4:10.1f}")
    print("\n  (Python prototype timings -- the RATIO between rows is the meaningful")
    print("   part, not the absolute seconds.)")
    print(f"\n  cascades cost {sum(casc_t)/len(casc_t) / (sum(shear_t)/len(shear_t)):.0f}x "
          f"a shear sweep per pass in this prototype.")


if __name__ == "__main__":
    main()
