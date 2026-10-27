"""Why does cascade_render.png's middle panel look bad? (2026-09-13)

Erik: "the middle plot in cascade_render.png looks really bad ... is it because something
is wrong here?  Or because point lights are hard?  because in real_scene_light_cascades.png it
looks awesome, and this is also cascade right?"

Two candidate explanations, and they make DIFFERENT predictions, so the experiment
separates them:

  (a) OUR PROTOTYPE IS MISSING A PUBLISHED FIX.  Osborne & Sannikov 2024 section 2.5
      name the artifact: parallax between a cascade-i probe and the four cascade-(i+1)
      probes it interpolates, which they call "non-conservation of energy at locations
      where cascades overlap a light source".  Their remedy is the BILINEAR FIX: trace
      each interval from its own start position to the start position of the associated
      child cone ON EACH of the four coarse probes, merge each with that probe's own
      sample, and only then average with the bilinear weights.  Costs 4x the rays.
      Prediction: implementing it cleans the picture up with ONE source.

  (b) A SINGLE BRIGHT POINT SOURCE IN AN EMPTY ROOM IS THE WORST CASE.  The same paper
      calls its ringing test "the worst-case scenario ... a small, very high opacity
      source embedded in a completely transparent medium", which is exactly this scene.
      Prediction: adding more sources cleans the picture up WITHOUT any fix.

So: the same scene, 1 source versus 4, each with and without the fix.  Whichever axis
moves the picture is the answer -- and if both do, we learn how much each is worth.

Run (~4 minutes):
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/cascade_fix_study.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from room_render import H, W, deck, tonemap            # noqa: E402
from scheme_study import solve_exact                   # noqa: E402

BASE_DIRS = 16
N_CASCADES = 4
BASE_LEN = 1.0
MARCH = 0.5            # <= 0.5 so a one-tile wall can never be stepped over
GAMMA = 0.42
PALETTE = ["#05050c", "#2a1146", "#7b1d3f", "#d2492a", "#f4a93a", "#fff3c4"]
WALL_COLOUR = "#3a3f4a"

EXTRA_SOURCES = [(40, 12), (52, 20), (10, 8), (32, 33)]


def scene(n_sources):
    kappa, source, opaque = deck()
    source = np.zeros((H, W))
    for (y, x) in EXTRA_SOURCES[:n_sources]:
        if not opaque[y, x]:
            source[y, x] = 1.0
    return kappa, source, opaque


def trace_seg(p0, p1, em, kappa):
    """March the straight segment p0 -> p1.  Returns (radiance, transmittance)."""
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return 0.0, 1.0
    n = max(1, int(length / MARCH))
    ds = length / n
    ux, uy = dx / length, dy / length
    rad, trans = 0.0, 1.0
    for k in range(n):
        t = (k + 0.5) * ds
        ix, iy = int(x0 + ux * t), int(y0 + uy * t)
        if not (0 <= ix < W and 0 <= iy < H):
            break
        rad += em[iy, ix] * trans * ds
        trans *= math.exp(-kappa[iy, ix] * ds)
        if trans < 1e-5:
            break
    return rad, trans


def cascades(em, kappa, ncasc=N_CASCADES, bilinear_fix=False):
    """Radiance cascades, branching factor 4, optionally with the bilinear fix."""
    far_edge = math.hypot(H, W)

    def geom(i):
        sp = 2 ** i
        nd = BASE_DIRS * (4 ** i)
        r0 = BASE_LEN * (4 ** i - 1) / 3.0
        r1 = BASE_LEN * (4 ** (i + 1) - 1) / 3.0
        if i == ncasc - 1:
            r1 = far_edge
        return sp, nd, r0, r1

    # --- top cascade: traced normally, it has no parent to be parallaxed against
    sp, nd, r0, r1 = geom(ncasc - 1)
    ph, pw = H // sp, W // sp
    top = np.zeros((ph, pw, nd))
    for j in range(ph):
        for k in range(pw):
            px, py = (k + 0.5) * sp, (j + 0.5) * sp
            for d in range(nd):
                ang = (d + 0.5) * (2.0 * math.pi / nd)
                ca, sa = math.cos(ang), math.sin(ang)
                top[j, k, d] = trace_seg((px + ca * r0, py + sa * r0),
                                         (px + ca * r1, py + sa * r1), em, kappa)[0]
    print(f"    cascade {ncasc-1}: {ph}x{pw} x {nd} dirs, {r0:.1f}..{r1:.1f}")
    far = top

    # --- downwards: build and merge in one step
    for i in range(ncasc - 2, -1, -1):
        sp, nd, r0, r1 = geom(i)
        fsp, fnd, _fr0, _fr1 = geom(i + 1)
        ph, pw = H // sp, W // sp
        fph, fpw = H // fsp, W // fsp
        cur = np.zeros((ph, pw, nd))
        for j in range(ph):
            for k in range(pw):
                px, py = (k + 0.5) * sp, (j + 0.5) * sp
                # the four coarse probes this one interpolates, and their weights
                cx = px / fsp - 0.5
                cy = py / fsp - 0.5
                k0, j0 = int(math.floor(cx)), int(math.floor(cy))
                tx, ty = cx - k0, cy - j0
                quads = []
                for dj in (0, 1):
                    for dk in (0, 1):
                        wgt = (ty if dj else 1 - ty) * (tx if dk else 1 - tx)
                        jj, kk = j0 + dj, k0 + dk
                        if wgt > 0 and 0 <= jj < fph and 0 <= kk < fpw:
                            quads.append((wgt, jj, kk,
                                          (kk + 0.5) * fsp, (jj + 0.5) * fsp))
                wsum = sum(q[0] for q in quads) or 1.0
                for d in range(nd):
                    ang = (d + 0.5) * (2.0 * math.pi / nd)
                    ca, sa = math.cos(ang), math.sin(ang)
                    if not bilinear_fix:
                        # one interval from this probe, then bilinearly blended far light
                        rad, tr = trace_seg((px + ca * r0, py + sa * r0),
                                            (px + ca * r1, py + sa * r1), em, kappa)
                        acc = 0.0
                        for (wgt, jj, kk, _qx, _qy) in quads:
                            acc += wgt * sum(far[jj, kk, (d * 4 + c) % fnd]
                                             for c in range(4)) / 4.0
                        cur[j, k, d] = rad + tr * (acc / wsum)
                    else:
                        # THE BILINEAR FIX: one interval per coarse probe, ending where
                        # that probe's child cone starts, merged with that probe's own
                        # sample, and only then averaged with the bilinear weights.
                        tot = 0.0
                        for (wgt, jj, kk, qx, qy) in quads:
                            rad, tr = trace_seg((px + ca * r0, py + sa * r0),
                                                (qx + ca * r1, qy + sa * r1), em, kappa)
                            fq = sum(far[jj, kk, (d * 4 + c) % fnd]
                                     for c in range(4)) / 4.0
                            tot += wgt * (rad + tr * fq)
                        cur[j, k, d] = tot / wsum
        print(f"    cascade {i}: {ph}x{pw} x {nd} dirs, {r0:.1f}..{r1:.1f}"
              f"{'  [bilinear fix]' if bilinear_fix else ''}")
        far = cur
    return far.mean(axis=2) * 2.0 * math.pi


def dark_room_leak(f, opaque):
    """Mean light in the far room (right of column 40), which only one doorway feeds,
    and in the sealed strip behind the partition -- relative to the source room."""
    src_room = np.zeros((H, W), bool)
    src_room[21:63, 1:26] = True
    src_room &= ~opaque
    far_room = np.zeros((H, W), bool)
    far_room[41:63, 41:63] = True          # below doorway 2, no line of sight
    far_room &= ~opaque
    return f[far_room].mean() / max(f[src_room].mean(), 1e-12)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    fire = LinearSegmentedColormap.from_list("fire", PALETTE)
    fire.set_bad(WALL_COLOUR)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11.2))

    rows = []
    for r, nsrc in enumerate((1, 4)):
        kappa, source, opaque = scene(nsrc)
        em = source / (2.0 * math.pi)
        print(f"{nsrc} source(s):")
        ex = solve_exact(source, opaque)
        print("  cascades, as the prototype stands")
        c_plain = cascades(em, kappa, bilinear_fix=False)
        print("  cascades, with the bilinear fix")
        c_fix = cascades(em, kappa, bilinear_fix=True)
        for f in (ex, c_plain, c_fix):
            f[opaque] = 0.0
        rows.append((nsrc, ex, c_plain, c_fix, opaque))

        names = ["exact (ground truth)", "cascades -- prototype as it stands",
                 "cascades + BILINEAR FIX"]
        hi = np.percentile(ex[~opaque], 99.5)
        for c, (nm, f) in enumerate(zip(names, (ex, c_plain, c_fix))):
            axes[r, c].imshow(tonemap(f, opaque, gamma=GAMMA, hi=hi), cmap=fire,
                              vmin=0, vmax=1, interpolation="nearest")
            leak = dark_room_leak(f, opaque)
            axes[r, c].set_title(f"{nm}\n{nsrc} source(s) -- light in the dark room: "
                                 f"{100*leak:.2f}% of the lit room", fontsize=10)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle(
        "Is the cascade artifact OURS or the algorithm's?\n"
        "Left to right: the missing published fix.  Top to bottom: one source versus "
        "four.  All six panels on the same scene and the same brightness scale.",
        fontsize=13)
    fig.tight_layout()
    fig.savefig(HERE / "cascade_fix_study.png", dpi=115)
    print("\nwrote", HERE / "cascade_fix_study.png")

    print(f"\n{'sources':>9}{'exact':>12}{'prototype':>12}{'+ fix':>12}"
          "     (light in the dark room, % of the lit room)")
    for (nsrc, ex, cp, cf, opq) in rows:
        print(f"{nsrc:9d}{100*dark_room_leak(ex, opq):11.2f}%"
              f"{100*dark_room_leak(cp, opq):11.2f}%{100*dark_room_leak(cf, opq):11.2f}%")


if __name__ == "__main__":
    main()
