"""Task 1, light half -- radiance cascades on the real map, with flashlights (2026-09-13).

Erik: "Also have some flashlights perhaps." and "I could accept the artifacts, but I would
like to see them with more light sources first."

Same real scene as real_scene_heat_sweeps.py (levels/playground after 60 s of real fires), lit by
every burning tile PLUS three flashlights.  A flashlight is a CONE emitter: it radiates
only into directions within its beam, which is natural in a directional solve and is what
design section 4.2 specifies.

THE TWO KNOBS THAT ACTUALLY CHANGE THE PICTURE.  The last session's dials note records
that Erik changed BASE_DIRS 16->8 and MARCH 0.25->0.125 and saw essentially nothing.
These are the ones that do something:
    N_CASCADES -- reach versus leak.  cascade_diagnose.py found coarse probes landing
                  INSIDE walls at our map sizes: that is the parallax/bilinear leak that
                  Osborne & Sannikov 2024 section 2.5 names, with a published fix.
    GAMMA      -- the transfer curve.  This, not field resolution, is what produces the
                  crisp two-state look of issue #64.

Run:
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/real_scene_light_cascades.py
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
LEVEL = "playground"

BASE_DIRS = 16
N_CASCADES = 4
BASE_LEN = 1.0
MARCH = 0.34
GAMMA = 0.42
CLIP_PCT = 99.7
PALETTE = ["#05050c", "#141a30", "#2f3d63", "#6d7fa8", "#b9c4dc", "#ffffff"]
WALL_COLOUR = "#2b2f38"

# (row, col, beam direction in degrees, half-angle, brightness)
# Amplitudes are chosen so a flashlight and the hottest fire tile read at comparable
# brightness.  A first pass had the lamps 65x the brightest fire and the fires vanished
# from the picture, which is a tone-mapping accident, not a property of the solver.
FLASHLIGHTS = [(20, 20, 0.0, 22.0, 6.0),
               (33, 62, 135.0, 22.0, 6.0),
               (40, 50, 270.0, 30.0, 4.5)]

FL_MASK = None
FL_LIVE = [False]        # flashlights off for the fires-only panel


def load():
    d = np.load(HERE / f"heatmap_{LEVEL}.npz", allow_pickle=True)
    T = d["temperature"].astype(float)
    la = d["light_atten"].astype(float)
    kappa = la.mean(axis=2) * 4.0          # scalar extinction, generous so walls block
    solid = d["solid"]
    opaque = solid & (kappa > 0.5)
    # visible emission: hot tiles glow, steeply with temperature
    em = np.where(T > 200.0, ((T - 200.0) / 800.0) ** 3, 0.0)
    em *= 1.0 / max(em.max(), 1e-9)        # hottest fire tile = 1.0
    return kappa, em, opaque, T


def cone_gain(iy, ix, ang):
    """A flashlight at (iy,ix) emits toward -ang only inside its beam."""
    g = 0.0
    for (fy, fx, fdeg, half, amp) in FLASHLIGHTS:
        if iy == fy and ix == fx:
            to_probe = math.degrees(math.atan2(-math.sin(ang), -math.cos(ang)))
            d = (to_probe - fdeg + 180.0) % 360.0 - 180.0
            if abs(d) <= half:
                g += amp * (0.5 + 0.5 * math.cos(math.pi * d / half))
    return g


def trace(px, py, ang, r0, r1, em, kappa, H, W, march):
    rad, trans = 0.0, 1.0
    dx, dy = math.cos(ang), math.sin(ang)
    n = max(1, int((r1 - r0) / march))
    ds = (r1 - r0) / n
    for k in range(n):
        t = r0 + (k + 0.5) * ds
        ix, iy = int(px + dx * t), int(py + dy * t)
        if not (0 <= ix < W and 0 <= iy < H):
            break
        e = em[iy, ix]
        if FL_LIVE[0] and FL_MASK[iy, ix]:
            e = e + cone_gain(iy, ix, ang)
        rad += e * trans * ds
        trans *= math.exp(-kappa[iy, ix] * ds)
        if trans < 1e-5:
            break
    return rad, trans


def build(em, kappa, H, W, ncasc):
    far = math.hypot(H, W)
    levels = []
    for i in range(ncasc):
        sp = 2 ** i
        nd = BASE_DIRS * (4 ** i)
        r0 = BASE_LEN * (4 ** i - 1) / 3.0
        r1 = BASE_LEN * (4 ** (i + 1) - 1) / 3.0
        if i == ncasc - 1:
            r1 = far                 # top cascade runs to the domain edge (paper 2.3)
        # The march must stay fine enough to see a one-tile wall, whatever the interval
        # length -- otherwise the top cascade steps straight over occluders and what looks
        # like the bilinear leak is really just an undersampled march.
        march = min(MARCH, 0.5)
        ph, pw = H // sp, W // sp
        rad = np.zeros((ph, pw, nd))
        tr = np.ones((ph, pw, nd))
        for j in range(ph):
            for k in range(pw):
                px, py = (k + 0.5) * sp, (j + 0.5) * sp
                for d in range(nd):
                    ang = (d + 0.5) * (2.0 * math.pi / nd)
                    rad[j, k, d], tr[j, k, d] = trace(px, py, ang, r0, r1,
                                                      em, kappa, H, W, march)
        levels.append([rad, tr])
        print(f"    cascade {i}: {ph}x{pw} probes x {nd} dirs, {r0:.1f}..{r1:.1f} tiles")
    return levels


def merge(levels):
    for i in range(len(levels) - 2, -1, -1):
        rad, tr = levels[i]
        far_rad = levels[i + 1][0]
        ph, pw, nd = rad.shape
        fph, fpw, fnd = far_rad.shape
        sp, fsp = 2 ** i, 2 ** (i + 1)
        for j in range(ph):
            for k in range(pw):
                px = ((k + 0.5) * sp) / fsp - 0.5
                py = ((j + 0.5) * sp) / fsp - 0.5
                k0, j0 = int(math.floor(px)), int(math.floor(py))
                fx, fy = px - k0, py - j0
                for d in range(nd):
                    acc = 0.0
                    for c in range(4):
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


def tonemap(f, opaque, gamma, hi):
    v = np.clip(np.asarray(f, float) / max(hi, 1e-12), 0, 1) ** gamma
    return np.ma.array(v, mask=opaque)


def _label(mask):
    """4-connected component labelling, numpy only (scipy in this env is built against
    numpy 1.x and will not import under numpy 2)."""
    lab = np.zeros(mask.shape, np.int32)
    cur = 0
    H, W = mask.shape
    for sy in range(H):
        for sx in range(W):
            if not mask[sy, sx] or lab[sy, sx]:
                continue
            cur += 1
            stack = [(sy, sx)]
            lab[sy, sx] = cur
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = cur
                        stack.append((ny, nx))
    return lab, cur


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    global FL_MASK

    kappa, em, opaque, T = load()
    H, W = em.shape
    FL_MASK = np.zeros((H, W), bool)
    for fl in FLASHLIGHTS:
        FL_MASK[fl[0], fl[1]] = True
    print(f"{LEVEL} {H}x{W}: {int((em>0).sum())} glowing tiles + "
          f"{len(FLASHLIGHTS)} flashlights")

    print("  fires only")
    lv = merge(build(em, kappa, H, W, N_CASCADES))
    f_fire = np.where(opaque, 0.0, lv[0][0].mean(axis=2) * 2.0 * math.pi)
    print("  fires + flashlights")
    FL_LIVE[0] = True
    lv = merge(build(em, kappa, H, W, N_CASCADES))
    f_all = np.where(opaque, 0.0, lv[0][0].mean(axis=2) * 2.0 * math.pi)
    panels = [("fires only -- 326 emitting tiles, no lamps", f_fire),
              ("fires + three flashlights", f_all)]

    hi = np.percentile(f_all[~opaque], CLIP_PCT)
    cmap = LinearSegmentedColormap.from_list("cool", PALETTE)
    cmap.set_bad(WALL_COLOUR)

    cols = [(f"{n}  (gamma {GAMMA})", f, GAMMA) for n, f in panels]
    cols.append(("fires + flashlights  (gamma 0.85 -- the #64 two-state look)",
                 f_all, 0.85))
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    for c, (name, f, g) in enumerate(cols):
        axes[c].imshow(tonemap(f, opaque, g, hi), cmap=cmap, vmin=0, vmax=1,
                       interpolation="nearest")
        axes[c].set_title(name, fontsize=11)
        axes[c].set_xticks([])
        axes[c].set_yticks([])
        for fl in FLASHLIGHTS:
            axes[c].plot(fl[1], fl[0], marker="o", ms=5, mfc="none",
                         mec="#ff4444", mew=1.4)
    fig.suptitle(
        "LIGHT -- radiance cascades on the same real scene.  Flashlights are CONE "
        "emitters (red rings)\n"
        f"levels/{LEVEL}, {int((em>0).sum())} glowing tiles after 60 s of real fire.  "
        "No spokes at any setting.  Middle vs right is the TRANSFER CURVE alone -- "
        "same field, same resolution.", fontsize=12)
    fig.tight_layout()
    fig.savefig(HERE / "real_scene_light_cascades.png", dpi=115)
    print("wrote", HERE / "real_scene_light_cascades.png")

    # how much light lands in sealed rooms that should be dark?  (the bilinear leak)
    lab, n = _label(~opaque)
    print(f"\n  {n} connected open regions; mean light in each, brightest first,")
    print("  normalised to the brightest region (a sealed dark room should be ~0):")
    for (name, f) in panels:
        vals = sorted((f[lab == i].mean() for i in range(1, n + 1)), reverse=True)
        print(f"    {name:<32} " + "  ".join(f"{v/max(vals[0],1e-12):.4f}"
                                             for v in vals[:8]))


if __name__ == "__main__":
    main()
