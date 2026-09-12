"""Room render — what the schemes actually LOOK like (2026-09-13).

Erik: "do u have a picture of the actual light rendering in some way? could be
just intensity, how it reacts to edges or whatever"

The bake-off measured profiles and ripple percentages. This renders a ship-deck
floorplan instead — rooms, a corridor, two doorways, one source — and tone-maps
the fluence the way a frame would be shown, so the artifacts can be judged by
eye rather than by statistic.

Two views per scheme:
  tone-mapped  — gamma-corrected, walls drawn in. What it would look like.
  log10        — the same field on a log scale. What is actually happening in
                 the dark parts, which the tone-map hides.

Run:
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/room_render.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scheme_study import (N_ORD, TWO_PI, solve_exact, solve_shear,
                          solve_step)          # noqa: E402

OUT = Path(__file__).resolve().parent
H = W = 64


def deck():
    """A ship-deck section: hull, two rooms, a corridor, two doorways.

    Chosen because it exercises every edge case the heat field has to get right:
    a doorway (aperture), a long corridor (channelling), an inside corner
    (shadow terminator), and a far room the source can only reach through one
    opening.
    """
    opaque = np.zeros((H, W), dtype=bool)
    opaque[0, :] = opaque[-1, :] = True            # hull
    opaque[:, 0] = opaque[:, -1] = True
    opaque[:, 26] = True                           # room A | corridor
    opaque[28:34, 26] = False                      #   doorway 1
    opaque[:, 40] = True                           # corridor | room B
    opaque[14:20, 40] = False                      #   doorway 2
    opaque[20, 1:26] = True                        # a partition inside room A
    opaque[20, 6:11] = False                       #   gap in the partition
    kappa = np.where(opaque, 50.0, 0.0)
    source = np.zeros((H, W))
    source[40, 12] = 1.0                           # the fire, in room A
    return kappa, source, opaque


def rotated(fn, kappa, source, nrot=8):
    acc = None
    for k in range(nrot):
        r = fn(kappa, source, phase=k * (TWO_PI / (N_ORD * nrot)))
        f = r[0] if isinstance(r, tuple) else r
        acc = f.copy() if acc is None else acc + f
    return acc / nrot


def tonemap(f, opaque, gamma=0.42, hi=None):
    """Gamma-curve the fluence into 0..1 and punch the walls out to dark grey."""
    hi = hi if hi is not None else np.percentile(f[~opaque], 99.5)
    v = np.clip(f / max(hi, 1e-12), 0.0, 1.0) ** gamma
    return np.where(opaque, np.nan, v)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    kappa, source, opaque = deck()
    print("solving exact ...")
    f_exact = solve_exact(source, opaque)
    print("solving step ...")
    f_step, _, _ = solve_step(kappa, source)
    print("solving step + rotation ...")
    f_steprot = rotated(solve_step, kappa, source)
    print("solving shear ...")
    f_shear, _, _ = solve_shear(kappa, source)
    print("solving shear + rotation ...")
    f_shearrot = rotated(solve_shear, kappa, source)

    fields = [("exact (ground truth)", f_exact),
              ("step S16", f_step),
              ("step S16 + rotation", f_steprot),
              ("shear S16", f_shear),
              ("shear S16 + rotation", f_shearrot)]

    # a warm fire palette; NaN (walls) renders as dark slate
    fire = LinearSegmentedColormap.from_list(
        "fire", ["#05050c", "#2a1146", "#7b1d3f", "#d2492a",
                 "#f4a93a", "#fff3c4"])
    fire.set_bad("#3a3f4a")

    hi = np.percentile(f_exact[~opaque], 99.5)

    fig, axes = plt.subplots(2, 5, figsize=(21, 8.6))
    for col, (name, f) in enumerate(fields):
        axes[0, col].imshow(tonemap(f, opaque, hi=hi), cmap=fire, vmin=0, vmax=1,
                            interpolation="nearest")
        axes[0, col].set_title(name, fontsize=11)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        lg = np.where(opaque, np.nan, np.log10(np.maximum(f, 1e-7)))
        im = axes[1, col].imshow(lg, cmap="magma", vmin=-5, vmax=-1.2,
                                 interpolation="nearest")
        axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
        if col == 4:
            fig.colorbar(im, ax=axes[1, col], fraction=0.046, label="log10")
    axes[0, 0].set_ylabel("tone-mapped\n(what you would see)", fontsize=10)
    axes[1, 0].set_ylabel("log10 fluence\n(what is really there)", fontsize=10)
    fig.suptitle("Ray engine v2 — one fire in a ship deck, 64x64, S16.  "
                 "Doorways, a corridor, a partition gap, an inside corner.\n"
                 "Top row is the frame you would see; bottom row is the same "
                 "field on a log scale, where the artifacts live.", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "room_render.png", dpi=125)
    print(f"wrote {OUT / 'room_render.png'}")

    # a close-up on the first doorway, where edges are decided
    fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4.4))
    sl = (slice(18, 46), slice(20, 50))
    for col, (name, f) in enumerate(fields):
        axes2[col].imshow(tonemap(f, opaque, hi=hi)[sl], cmap=fire,
                          vmin=0, vmax=1, interpolation="nearest")
        axes2[col].set_title(name, fontsize=11)
        axes2[col].set_xticks([]); axes2[col].set_yticks([])
    fig2.suptitle("Close-up: the doorway into the corridor — how each scheme "
                  "handles an aperture and the wall edges beside it", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(OUT / "room_doorway.png", dpi=130)
    print(f"wrote {OUT / 'room_doorway.png'}")

    # how much light reaches the far room, which only one opening feeds
    far = (~opaque) & (np.arange(W)[None, :] > 40)
    print(f"\n{'scheme':<26}{'far-room mean':>16}{'vs exact':>10}")
    ref = f_exact[far].mean()
    for name, f in fields:
        print(f"{name:<26}{f[far].mean():>16.3e}{f[far].mean()/ref:>10.2f}")


if __name__ == "__main__":
    main()
