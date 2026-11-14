"""Where do the vertical bands in the corridor come from? (2026-09-13)

Erik, looking at cascade_render.png: "the middle one radiance cascades looks
very strange to me, it has some artifacts I can't explain, especially in the
middle room, why do we see some vertical pillar in the middle corridor?"

Hypothesis: COARSE PROBES LANDING INSIDE WALLS. Cascade i has probe spacing
2^i, so cascade 4's probes sit 16 tiles apart. The deck's walls are at x = 26
and x = 40. A probe that lands inside a wall sees nothing, and because the merge
interpolates bilinearly between coarse probes, that dead probe's zero gets
smeared across everything downstream of it — as a band aligned with the coarse
probe grid.

If that is the cause, the bands must MOVE when the probe grid moves. This script
shifts the probe grid and varies the cascade count, and looks.

Run:
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/cascade_diagnose.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade_render as CR                              # noqa: E402
from room_render import H, W, deck, tonemap              # noqa: E402

OUT = Path(__file__).resolve().parent


def probe_wall_fraction(opaque, n_cascades):
    """For each cascade level, what fraction of its probes sit inside a wall?"""
    rows = []
    for i in range(n_cascades):
        sp = 2 ** i
        ph, pw = H // sp, W // sp
        dead = 0
        for j in range(ph):
            for k in range(pw):
                py, px = int((j + 0.5) * sp), int((k + 0.5) * sp)
                if 0 <= py < H and 0 <= px < W and opaque[py, px]:
                    dead += 1
        rows.append((i, sp, ph * pw, dead, 100.0 * dead / (ph * pw)))
    return rows


def render(kappa, source, opaque, n_cascades, base_dirs):
    CR.N_CASCADES = n_cascades
    CR.BASE_DIRS = base_dirs
    emission = source / (2.0 * math.pi)
    levels = CR.build(emission, kappa)
    CR.merge(levels)
    return CR.fluence(levels, opaque)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    kappa, source, opaque = deck()

    print("=" * 74)
    print("PROBE / WALL COLLISION  (deck walls at x = 26 and x = 40)")
    print("=" * 74)
    print(f"  {'cascade':<9}{'spacing':>9}{'probes':>9}{'in a wall':>11}{'%':>8}")
    for i, sp, tot, dead, pct in probe_wall_fraction(opaque, 6):
        note = "  <-- coarse enough to straddle a whole room" if sp >= 8 else ""
        print(f"  {i:<9}{sp:>9}{tot:>9}{dead:>11}{pct:>7.1f}%{note}")

    # which x-columns do the coarse probe grids actually land on?
    print("\n  coarse probe columns (x), and whether each is inside a wall:")
    for i in (3, 4, 5):
        sp = 2 ** i
        cols = [(int((k + 0.5) * sp), bool(opaque[32, int((k + 0.5) * sp)]))
                for k in range(W // sp)]
        s = "  ".join(f"{x}{'*' if d else ''}" for x, d in cols)
        print(f"    cascade {i} (spacing {sp:>2}): {s}       (* = inside a wall)")

    variants = []
    for nc in (3, 4, 5, 6):
        print(f"\nrendering N_CASCADES={nc} ...")
        variants.append((f"N_CASCADES = {nc}", render(kappa, source, opaque,
                                                      nc, 16)))

    fire = LinearSegmentedColormap.from_list("fire", CR.PALETTE)
    fire.set_bad(CR.WALL_COLOUR)

    fig, axes = plt.subplots(2, len(variants),
                             figsize=(4.9 * len(variants), 8.8))
    for col, (name, f) in enumerate(variants):
        hi = np.percentile(f[~opaque], 99.5)
        axes[0, col].imshow(tonemap(f, opaque, hi=hi), cmap=fire,
                            vmin=0, vmax=1, interpolation="nearest")
        axes[0, col].set_title(name, fontsize=11)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        lg = np.where(opaque, np.nan,
                      np.log10(np.maximum(f / max(hi, 1e-12), 1e-6)))
        axes[1, col].imshow(lg, cmap="magma", vmin=-4.5, vmax=0,
                            interpolation="nearest")
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
        # mark the coarse probe columns of the TOP cascade in this variant
        sp = 2 ** (int(name.split("=")[1]) - 1)
        for k in range(W // sp):
            x = (k + 0.5) * sp
            axes[1, col].axvline(x, color="cyan", lw=0.6, alpha=0.55)
    axes[0, 0].set_ylabel("tone-mapped", fontsize=10)
    axes[1, 0].set_ylabel("log10\n(cyan = top cascade's probe columns)",
                          fontsize=9)
    fig.suptitle("Where the corridor banding comes from: coarse cascade probes\n"
                 "If the bands line up with the cyan probe columns and MOVE when "
                 "the cascade count changes, the probe grid is the cause.",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "cascade_diagnose.png", dpi=125)
    print(f"\nwrote {OUT / 'cascade_diagnose.png'}")

    # a horizontal cut straight through the corridor
    print("\nhorizontal cut through the corridor (row 32), normalised:")
    print(f"  {'x':>4}" + "".join(f"{n.split('=')[1].strip():>9}"
                                  for n, _ in variants))
    for x in range(27, 40):
        if opaque[32, x]:
            continue
        vals = [f[32, x] / max(f[~opaque].max(), 1e-12) for _, f in variants]
        print(f"  {x:>4}" + "".join(f"{v:>9.4f}" for v in vals))


if __name__ == "__main__":
    main()
