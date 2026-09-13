"""Radiance cascades — a working prototype, rendered (2026-09-13).

Erik asked for "a picture of the actual light rendering". The room render shows
the HEAT schemes, which are grid sweeps and carry visible cross/star artifacts.
Light is a different solver in the design (R-P: radiance cascades in GLSL), and
its whole selling point is that it does NOT have those artifacts, because the
angular resolution grows with distance instead of staying fixed.

This is a minimal but honest 2D radiance-cascades implementation, so that claim
can be looked at rather than taken on faith.

THE ALGORITHM (Sannikov 2023; branching factor 4)
    Cascade i has probe spacing s*2^i and d*4^i directions, and covers the radial
    interval [r_i, r_{i+1}) with r_i = L*(4^i - 1)/3. So as you go up: four times
    the angular resolution, a quarter of the probes, four times the distance.
    Each probe-direction stores the radiance gathered over its own interval and
    the transmittance across it.

    Merging runs top down. A cascade-i interval is completed by the four
    cascade-(i+1) intervals that subdivide its direction, sampled bilinearly
    over the four nearest coarse probes:

        L_total = L_near + T_near * L_far

    which is exactly the interval-composition identity the design quotes in §4.1.

    The penumbra hypothesis is why this works: near the receiver you need spatial
    detail and little angular detail, far away you need the opposite, and the
    cascade structure spends the budget accordingly.

Run:
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/cascade_render.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from room_render import H, W, deck, tonemap                 # noqa: E402
from scheme_study import solve_exact, solve_step            # noqa: E402

OUT = Path(__file__).resolve().parent

# ===========================================================================
#  DIALS — the whole tuning surface. Edit, re-run, look. ~3 s per run.
#  (Same panel-at-the-top idea as tools/fire_tuning_lab.py.)
# ===========================================================================

# --- quality -------------------------------------------------------------
BASE_DIRS = 8      # directions in cascade 0. THE quality knob. Measured ring
                    # ripple at r=3: 4 dirs -> 76%, 8 -> 19%, 16 -> 15%.
                    # 4->8 is the big jump; after 16 it barely moves.
                    # Cost is linear in this, at every cascade.
N_CASCADES = 5      # how many levels. Each covers 4x the distance of the one
                    # below, so 5 reaches 341 tiles — more than the map. Drop
                    # to 4 and the far field goes dark; raise it and nothing
                    # changes except wasted work.
BASE_LEN = 1.0      # length of cascade 0's radial interval, in tiles. Smaller
                    # = finer near-field detail and more cascades needed to
                    # cover the map. This is the penumbra-hypothesis dial.
MARCH = 0.25/2        # sub-tile step when tracing an interval. Smaller is more
                    # accurate and slower; 0.25 is well past the point of
                    # visible change.

# --- look ----------------------------------------------------------------
GAMMA = 0.42        # tone-map curve. LOWER lifts the dark end (more of the
                    # room becomes visible, flatter); HIGHER crushes it (more
                    # contrast, deeper blacks). 0.42 is roughly sRGB-ish.
                    # Try 0.30 to see what is hiding in the shadows, 0.6 for
                    # the high-contrast prototype look Erik liked (issue #64).
CLIP_PCT = 99.5     # what counts as "full brightness". Lower = more blown-out
                    # highlights; raise toward 100 to keep the fire's core from
                    # clipping.
PALETTE = ["#05050c", "#2a1146", "#7b1d3f", "#d2492a", "#f4a93a", "#fff3c4"]
WALL_COLOUR = "#3a3f4a"

# --- what to compare -----------------------------------------------------
SHOW_STEP = True    # include the S16 heat sweep as a contrast column
LOG_FLOOR = -4.5    # bottom of the log10 row. Raise toward -3 to zoom in on
                    # the bright half; drop to -6 to see the faintest leakage.
# ===========================================================================


def trace(px, py, ang, r0, r1, emission, kappa):
    """March one interval [r0, r1) from (px, py) along ang.

    Returns (radiance gathered, transmittance across the interval)."""
    rad, trans = 0.0, 1.0
    dx, dy = math.cos(ang), math.sin(ang)
    n = max(1, int((r1 - r0) / MARCH))
    ds = (r1 - r0) / n
    for k in range(n):
        t = r0 + (k + 0.5) * ds
        ix, iy = int(px + dx * t), int(py + dy * t)
        if not (0 <= ix < W and 0 <= iy < H):
            break
        rad += emission[iy, ix] * trans * ds
        trans *= math.exp(-kappa[iy, ix] * ds)
        if trans < 1e-5:
            break
    return rad, trans


def build(emission, kappa):
    """Trace every cascade's own interval. Returns per-level (rad, trans)."""
    levels = []
    for i in range(N_CASCADES):
        spacing = 2 ** i
        ndirs = BASE_DIRS * (4 ** i)
        r0 = BASE_LEN * (4 ** i - 1) / 3.0
        r1 = BASE_LEN * (4 ** (i + 1) - 1) / 3.0
        ph, pw = H // spacing, W // spacing
        rad = np.zeros((ph, pw, ndirs))
        tr = np.ones((ph, pw, ndirs))
        for j in range(ph):
            for k in range(pw):
                px = (k + 0.5) * spacing
                py = (j + 0.5) * spacing
                for d in range(ndirs):
                    ang = (d + 0.5) * (2.0 * math.pi / ndirs)
                    rad[j, k, d], tr[j, k, d] = trace(px, py, ang, r0, r1,
                                                      emission, kappa)
        levels.append([rad, tr])
        print(f"  cascade {i}: {ph}x{pw} probes x {ndirs} dirs, "
              f"interval {r0:.2f}..{r1:.2f} tiles")
    return levels


def merge(levels):
    """Top-down merge. L_near += T_near * (bilinear, direction-averaged L_far)."""
    for i in range(N_CASCADES - 2, -1, -1):
        rad, tr = levels[i]
        far_rad, _ = levels[i + 1]
        ph, pw, nd = rad.shape
        fph, fpw, fnd = far_rad.shape
        sp, fsp = 2 ** i, 2 ** (i + 1)
        for j in range(ph):
            for k in range(pw):
                # this probe's position in the coarse grid's probe coordinates
                px = ((k + 0.5) * sp) / fsp - 0.5
                py = ((j + 0.5) * sp) / fsp - 0.5
                k0, j0 = int(math.floor(px)), int(math.floor(py))
                fx, fy = px - k0, py - j0
                for d in range(nd):
                    acc = 0.0
                    for c in range(4):                 # the 4 finer directions
                        fd = (d * 4 + c) % fnd
                        v = 0.0
                        for dj in (0, 1):
                            for dk in (0, 1):
                                wgt = ((fy if dj else 1 - fy)
                                       * (fx if dk else 1 - fx))
                                jj, kk = j0 + dj, k0 + dk
                                if 0 <= jj < fph and 0 <= kk < fpw:
                                    v += wgt * far_rad[jj, kk, fd]
                        acc += v
                    rad[j, k, d] += tr[j, k, d] * (acc / 4.0)
    return levels


def fluence(levels, opaque):
    """Cascade 0's directions, averaged, lifted back to the tile grid."""
    rad0 = levels[0][0]
    f = rad0.mean(axis=2) * (2.0 * math.pi)
    return np.where(opaque, 0.0, f)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    kappa, source, opaque = deck()
    emission = source / (2.0 * math.pi)

    print("building cascades ...")
    levels = build(emission, kappa)
    print("merging ...")
    merge(levels)
    f_casc = fluence(levels, opaque)

    print("exact ...")
    f_exact = solve_exact(source, opaque)
    print("step S16 (the heat sweep, for contrast) ...")
    f_step, _, _ = solve_step(kappa, source)

    fire = LinearSegmentedColormap.from_list("fire", PALETTE)
    fire.set_bad(WALL_COLOUR)

    # normalise each field on its own 99.5th percentile: this is a question
    # about SHAPE and artifacts, not about absolute scale
    fields = [("exact (ground truth)", f_exact),
              (f"radiance cascades ({BASE_DIRS} base dirs)", f_casc)]
    if SHOW_STEP:
        fields.append(("step S16 sweep (the HEAT solver)", f_step))

    fig, axes = plt.subplots(2, len(fields),
                             figsize=(5.2 * len(fields), 9.2))
    for col, (name, f) in enumerate(fields):
        hi = np.percentile(f[~opaque], CLIP_PCT)
        axes[0, col].imshow(tonemap(f, opaque, gamma=GAMMA, hi=hi), cmap=fire, vmin=0, vmax=1,
                            interpolation="nearest")
        axes[0, col].set_title(name, fontsize=11)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        lg = np.where(opaque, np.nan, np.log10(np.maximum(f / max(hi, 1e-12),
                                                          1e-6)))
        im = axes[1, col].imshow(lg, cmap="magma", vmin=LOG_FLOOR, vmax=0,
                                 interpolation="nearest")
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
        if col == len(fields) - 1:
            fig.colorbar(im, ax=axes[1, col], fraction=0.046,
                         label="log10 (normalised)")
    axes[0, 0].set_ylabel("tone-mapped", fontsize=10)
    axes[1, 0].set_ylabel("log10", fontsize=10)
    fig.suptitle("Radiance cascades vs the grid sweep — one fire, ship deck, "
                 f"64x64, {N_CASCADES} cascades\n"
                 "Cascades are the design's LIGHT solver; the S16 sweep is its "
                 "HEAT solver. Same scene, same source.", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "cascade_render.png", dpi=125)
    print(f"wrote {OUT / 'cascade_render.png'}")

    # isotropy of each, as a number, around the source
    from scheme_study import ring, ripple
    cy, cx = 40, 12
    print(f"\n{'scheme':<38}{'ripple r=3':>12}{'ripple r=8':>12}")
    for name, f in fields:
        print(f"{name:<38}{ripple(ring(f, cy, cx, 3)):>11.1f}%"
              f"{ripple(ring(f, cy, cx, 8)):>11.1f}%")


if __name__ == "__main__":
    main()
