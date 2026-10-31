"""Octopus clips from the SIM (#106 P6d2): the prototype GIF harness
(``prototypes/octopus_motion/gates3d.py``'s ``_gif``) re-pointed at the shared
``Simulation`` -- the ported animal, driven by its unit's intents, drawn from
the ``__creature__`` record the digest hashes, so Erik sees what the sim does
before P7's render.

    <breach-py> tools/octopus_clips.py                    # all four
    <breach-py> tools/octopus_clips.py --which grab idle  # some

writes ``prototypes/octopus_motion/out/sim_{crawl,swim,grab,idle}.gif``.

Pure half / thin shell (the map-editor pattern): :func:`clip_frames` turns a
run (``tests/_creature_scenarios.Run``: per tick the chain accessor's view,
``creature_system.read_chain`` -- THE read of a creature's chain since #106
P7 -- and the decoded record for the hub's gait and seals) into plain
per-frame drawing data -- the crown, the eight arm polylines and the colour
of each arm's primitive state, the prop, a caption -- with no matplotlib;
:func:`write_gif` draws them, top-down (screen frame: North up) and, for the
swim and the grab, a side view (x, z). Everything float here is the
accessor's (dequantized once at the read); nothing is written back.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "prototypes" / "octopus_motion" / "out"

#: Arm colour by the hub's primitive state (creature_primitives.ARM_*).
STATE_COLOURS = {
    "limp": "#9a9a9a", "ready": "#5a3b8a", "stance": "#d1495b", "recover": "#edae49",
    "still": "#5a3b8a", "decide": "#7d6bb0", "explore": "#00798c", "touch": "#30a0b0",
    "relax": "#a3c4bc", "reach": "#00798c", "swim": "#2e86ab", "grab_reach": "#00798c",
    "grab_wrap": "#e36414", "grab_hold": "#6a040f", "carry": "#6a040f",
}


# ---------------------------------------------------------------------------
# The pure half
# ---------------------------------------------------------------------------
def clip_frames(run, every: int = 1, label: str = "") -> list[dict]:
    """Per drawn frame: ``t``, ``crown`` (x, y, z in the screen frame, m),
    ``arms`` (A, n, 3) screen-frame nodes, ``colours`` (A,), ``prop`` (x, y,
    z) or None, ``caption``. The screen frame flips the grid's y (North
    up); z is height above the deck."""
    from simulation import creature_primitives as P
    out = []
    for f in range(0, len(run.records), max(1, int(every))):
        rec = run.records[f]
        view = run.views[f]                    # THE chain accessor (#106 P7)
        Pn = view.nodes.astype(float)
        Pn[..., 1] = -Pn[..., 1]
        X = view.crown.astype(float)
        X[1] = -X[1]
        names = [P.ARM_STATE_NAMES[int(s)] for s in view.arm_state]
        prop = None
        if view.prop is not None:
            prop = view.prop[:3].astype(float)
            prop[1] = -prop[1]
        counts = {}
        for n in names:
            counts[n] = counts.get(n, 0) + 1
        states = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        gait = P.GAIT_NAMES[int(rec["gait"])]
        extra = f", seals {len(rec['grabs'])}" if rec["grabs"] else ""
        out.append(dict(t=float(run.t[f]), crown=X, arms=Pn,
                        colours=[STATE_COLOURS[n] for n in names], prop=prop,
                        caption=f"{label}  t = {run.t[f]:5.2f} s   gait {gait}: {states}{extra}"))
    return out


def view_limits(frames, axes, pad: float) -> tuple:
    """Fixed axis limits enclosing every frame's nodes, crown and prop."""
    pts = [f["arms"].reshape(-1, 3) for f in frames] + [f["crown"][None] for f in frames]
    pts += [f["prop"][None] for f in frames if f["prop"] is not None]
    allp = np.concatenate(pts)
    lo, hi = allp.min(0) - pad, allp.max(0) + pad
    return [(lo[i], hi[i]) for i in axes]


# ---------------------------------------------------------------------------
# The thin shell
# ---------------------------------------------------------------------------
def write_gif(path, frames, *, title: str, crown_r: float, prop_r: float | None = None,
              side: bool = False, follow: float | None = None, fps: int = 12) -> str:
    """Draw ``frames`` (``clip_frames``) as a GIF: top-down, plus a side view
    (x, z) when ``side``; ``follow`` (m) keeps the crown centred in a window
    of that half-width instead of the whole run's extent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle

    views = [(0, 1)] + ([(0, 2)] if side else [])
    fig, axs = plt.subplots(1, len(views), figsize=(5.0 * len(views), 4.6), squeeze=False, dpi=80)
    axs = axs[0]
    A = frames[0]["arms"].shape[0]
    lines, crowns, props = [], [], []
    for ax, (i, j) in zip(axs, views):
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("xyz"[i] + " (m)")
        ax.set_ylabel(("x", "y (North up)", "z (m, deck = 0)")[j])
        if j == 2:
            ax.axhline(0.0, color="#7a6a55", lw=2)
        lines.append([ax.plot([], [], "-", lw=2.2, solid_capstyle="round")[0] for _ in range(A)])
        c = Circle((0, 0), crown_r, color="#5a3b8a", alpha=0.45)
        ax.add_patch(c)
        crowns.append(c)
        if prop_r is not None:
            pc = Circle((0, 0), prop_r, color="#6b8f71", alpha=0.6)
            ax.add_patch(pc)
            props.append(pc)
    txt = fig.text(0.01, 0.01, "", fontsize=7, family="monospace")
    fig.suptitle(title, fontsize=9)
    lim = None if follow is not None else view_limits(frames, (0, 1, 2), 0.5)

    def draw(k):
        f = frames[k]
        for ax, (i, j), ls, c in zip(axs, views, lines, crowns):
            for a, ln in enumerate(ls):
                ln.set_data(f["arms"][a, :, i], f["arms"][a, :, j])
                ln.set_color(f["colours"][a])
            c.center = (f["crown"][i], f["crown"][j])
            if follow is not None:
                cx = f["crown"][i]
                # the side view keeps the deck in sight for an animal on it
                cy = f["crown"][j] if j != 2 else max(f["crown"][2], follow * 0.4)
                ax.set_xlim(cx - follow, cx + follow)
                ax.set_ylim(cy - follow * 0.8, cy + follow * 0.8)
            else:
                ax.set_xlim(*lim[i])
                ax.set_ylim(*lim[j])
        for (i, j), pc in zip(views, props):
            if f["prop"] is not None:
                pc.center = (f["prop"][i], f["prop"][j])
        txt.set_text(f["caption"])
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000.0 / fps)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# The four clips
# ---------------------------------------------------------------------------
def make_clip(which: str, out_dir: Path = OUT) -> str:
    import _creature_scenarios as SC
    from simulation.creatures import get_creature
    row = get_creature("octopus")
    crown_r = row.body.body_radius_m
    if which == "crawl":
        run = SC.crawl(70.0, 12.0, seed=1, name="clip_crawl")
        frames = clip_frames(run, every=2, label="crawl toward 70 deg")
        return write_gif(out_dir / "sim_crawl.gif", frames, crown_r=crown_r, follow=4.5,
                         title="SIM crawl (P6d2): MOVE_DIR 70 deg off the heading\n"
                               "red = pushing (anchored, elongating), amber = recovering")
    if which == "swim":
        sc = SC.Scenario("clip_swim", w=150, h=40, at=(12.0, 18.5))
        sc.lift(8.0)
        from simulation.orders import ORDER_MOVE_COVER, ORDER_SPRINT
        sc.seconds(4.3, each=lambda s: s.move(0.0, ORDER_MOVE_COVER))
        sc.seconds(3.0, each=lambda s: s.move(0.0, ORDER_SPRINT))
        frames = clip_frames(sc.finish(), every=2, label="umbrella, then the jet train")
        return write_gif(out_dir / "sim_swim.gif", frames, crown_r=crown_r, side=True, follow=4.5,
                         title="SIM swim (P6d2) from mid water: 3 umbrella strokes (MOVE_DIR at "
                               "move-with-cover),\nthen the jet train (sprint) -- top-down | side")
    if which == "grab":
        run = SC.grab(crawl_s=6.0, escape_s=3.0, size=110, seed=3)
        frames = clip_frames(run, every=2, label="reach, grab, lift, carry, jets")
        return write_gif(out_dir / "sim_grab.gif", frames, crown_r=crown_r,
                         prop_r=row.grab.prop_radius, side=True, follow=4.5,
                         title="SIM grab (P6d2): USE -> reach, wrap (orange), hold + LIFT (dark red),\n"
                               "carry while crawling, then the carry jets -- top-down | side")
    if which == "idle":
        run = SC.idle(24.0, seed=5)
        frames = clip_frames(run, every=1, label="idle")
        return write_gif(out_dir / "sim_idle.gif", frames, crown_r=crown_r, follow=4.0,
                         title="SIM idle (P6d2): no intent -- seated (purple), at most two arms\n"
                               "explore (teal), relax (pale), stick down")
    raise ValueError(f"unknown clip {which!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--which", nargs="*", default=["crawl", "swim", "grab", "idle"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    for w in args.which:
        print(make_clip(w, Path(args.out)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
