"""Light in smoke: does it favour cascades? (2026-09-13)

Erik: "The one question that perhaps cascades excelled at, is if light hits our smoke -
would that look better with cascade?  Can we even simulate it and have a look at what we
can expect when we introduce it?"

WHY THIS IS THE HARDEST TEST OF THE THREE SOLVERS, and the prediction going in.
In empty air you only ever see light that has landed on a SURFACE.  The field in
between is invisible, so a solver's directional artifacts are invisible with it.  Smoke
changes that: a smoke cell scatters some of the light passing through it out of the
plane, toward the camera, so its brightness is proportional to the LOCAL FLUENCE.  The
field itself becomes visible.  So the prediction is that every artifact the schemes
have -- step's axis cross, shear's 16 spokes -- stops being a curiosity about a field
nobody sees and becomes a visible streak in the smoke.

WHAT IS MODELLED
  - Extinction: smoke attenuates light per tile, on top of the walls.  Both solver
    families do this natively; cascades carry transmittance through the interval merge,
    the sweep carries it through the per-cell absorptivity.  Nothing to choose here.
  - Single scattering, as the RENDER term: visible smoke brightness = density x fluence.
    This is a post-process on the field, so it is identical arithmetic for all three
    solvers -- which is exactly what makes it a fair test of the FIELD.
  - Multiple scattering (light bouncing inside the smoke, which is what makes thick
    smoke glow softly) is a scattering term like a diffuse reflection, so it costs the
    same thing: one extra solve per bounce.  Not included here; the cost rule is the
    one already measured in reflections_study.py.

Run (~1 minute):
    C:/Users/steen/anaconda3/python.exe docs/ray_engine_v2_scheme_study_2026-09-13/smoke_study.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from room_render import H, W, tonemap                  # noqa: E402
import reflections_study as RS                         # noqa: E402

SMOKE_EXT = 0.16       # extinction per tile at density 1
SMOKE_SCAT = 0.38      # how brightly a unit of smoke scatters toward the camera
FLOOR = 0.22           # how much of the fluence a bare floor shows back
GAMMA = 0.42
PALETTE = ["#05050c", "#141a30", "#2f3d63", "#6d7fa8", "#b9c4dc", "#ffffff"]
WALL_COLOUR = "#2b2f38"

LIGHTS = [(45, 8, 0.0, 14.0, 1.0),        # flashlight, straight down the room
          (10, 14, 180.0, 180.0, 0.25)]   # lamp behind the partition


def smoke_field(opaque):
    """A drifting bank across the flashlight beam, plus a plume by the lamp."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    d = np.zeros((H, W))
    for (cy, cx, sy, sx, amp) in ((45.0, 17.0, 9.0, 5.0, 1.0),
                                  (38.0, 21.0, 6.0, 4.0, 0.55),
                                  (11.0, 18.0, 5.0, 6.0, 0.7)):
        d += amp * np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2))
    d[opaque] = 0.0
    return np.clip(d, 0.0, 1.4)


def composite(fluence, dens, opaque):
    """What a frame would show: the lit floor, dimmed by the smoke above it, plus the
    smoke's own single-scattered glow."""
    trans = np.exp(-SMOKE_EXT * dens * 2.0)
    img = fluence * (FLOOR * trans + SMOKE_SCAT * dens)
    return np.where(opaque, 0.0, img)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    kappa_walls, opaque = RS.scene()
    dens = smoke_field(opaque)
    kappa = kappa_walls + SMOKE_EXT * dens
    print(f"smoke covers {(dens > 0.05).sum()} tiles, peak density {dens.max():.2f}")

    solvers = [
        ("step sweep S16", lambda k: RS.sweep_light(k, np.zeros((H, W)), LIGHTS,
                                                    theta=0.0)[0]),
        ("shear sweep S16", lambda k: RS.sweep_light(k, np.zeros((H, W)), LIGHTS,
                                                     theta=1.0)[0]),
        ("cascades + bilinear fix", lambda k: RS.cascade_light(k, np.zeros((H, W)),
                                                               LIGHTS)),
    ]

    cmap = LinearSegmentedColormap.from_list("cool", PALETTE)
    cmap.set_bad(WALL_COLOUR)
    fig, axes = plt.subplots(3, 3, figsize=(16, 15.5))
    for r, (name, solve) in enumerate(solvers):
        print(f"  {name} ...")
        clear = solve(kappa_walls)
        smoky = solve(kappa)
        img_clear = composite(clear, np.zeros_like(dens), opaque)
        img_smoky = composite(smoky, dens, opaque)
        glow = np.where(opaque, 0.0, smoky * dens)
        hi = np.percentile(img_smoky[~opaque], 99.0)
        panels = [("no smoke -- only surfaces are visible", img_clear, hi),
                  ("WITH SMOKE -- the light field becomes visible", img_smoky, hi),
                  ("the smoke's glow alone (own scale)", glow,
                   max(glow[~opaque].max(), 1e-12))]
        for c, (sub, f, scale) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(tonemap(f, opaque, gamma=GAMMA, hi=scale), cmap=cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"{name}\n{sub}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "LIGHT IN SMOKE.  A flashlight beam crossing a smoke bank, and a lamp in a "
        "plume.\n"
        "Smoke makes the field itself visible, so a solver's directional artifacts stop "
        "being invisible.\nSame smoke, same lights, same render maths in every row.",
        fontsize=14)
    fig.tight_layout()
    fig.savefig(HERE / "smoke_study.png", dpi=100)
    print("wrote", HERE / "smoke_study.png")

    # The number that matters here is not the glow's roughness -- the plume's own
    # density varies along an arc, so that statistic is contaminated.  What smoke does
    # is make the SOLVER's isotropy error visible, so measure that directly: ring ripple
    # of the clear-air fluence around the BARE LAMP, which is isotropic, so the true
    # answer is 1.00 and anything above it is pure artifact that smoke will reveal.
    ly, lx = LIGHTS[1][0], LIGHTS[1][1]
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.hypot(yy - ly, xx - lx)
    lamp_only = [LIGHTS[1]]
    print("")
    print(f"{'solver':<26}{'ring ripple around the bare lamp':>34}")
    for name, _solve in solvers:
        if name.startswith("cascades"):
            f = RS.cascade_light(kappa_walls, np.zeros((H, W)), lamp_only)
        else:
            th = 0.0 if name.startswith("step") else 1.0
            f = RS.sweep_light(kappa_walls, np.zeros((H, W)), lamp_only, theta=th)[0]
        vals = []
        for r0 in (3, 4, 5, 6, 7):
            m = (rr >= r0 - 0.5) & (rr < r0 + 0.5) & (~opaque)
            v = f[m]
            if v.size > 6 and v.min() > 0:
                vals.append(v.max() / v.min())
        print(f"{name:<26}{np.mean(vals):>34.2f}x")
    print("  (1.00x would be perfectly round.  In clear air this error is invisible")
    print("   because nothing between the surfaces is drawn; smoke draws it.)")


if __name__ == "__main__":
    main()
