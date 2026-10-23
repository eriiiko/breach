"""Build the space marine from scratch, compare it with the artwork, render, save.

    blender -b --factory-startup -P scripts/build.py -- [options]

    --draft          coarse grid + few samples (fast iteration)
    --sheet          orthographic front/side/back in the reference sheet's framing,
                     plus a silhouette overlay and a per-height mismatch table
    --beauty a,b,..  perspective studio renders (see scene.BEAUTY; `all` for every one)
    --scale S        beauty resolution scale (default 1.0)
    --samples N      Cycles samples (default 96, draft 24)
    --save           write source/space_marine.blend and source/mesh_stats.json
    --tag NAME       file prefix for the sheet outputs (default `sheet`)
"""
import argparse
import json
import os
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import kit  # noqa: E402
import marine  # noqa: E402
import scene  # noqa: E402
import studio  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, ".."))
PREVIEWS, SOURCE = os.path.join(ROOT, "previews"), os.path.join(ROOT, "source")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--beauty", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tag", default="sheet")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    kit.RES = 0.008 if args.draft else 0.004
    M = scene.make_materials()
    marine.build(M)
    rig, cam, floor = scene.setup(samples=args.samples or (24 if args.draft else 96))
    print("built in %.1fs" % (time.time() - t0))
    os.makedirs(PREVIEWS, exist_ok=True)
    os.makedirs(SOURCE, exist_ok=True)

    if args.save:
        stats = studio.mesh_stats(os.path.join(SOURCE, "mesh_stats.json"))
        print("mesh stats:", json.dumps(stats["TOTAL"]))
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(SOURCE, "space_marine.blend"))
    if args.sheet:
        panels = scene.render_sheet(rig, cam, floor, PREVIEWS, args.tag)
        scene.compare_sheet(panels, PREVIEWS, args.tag, verbose=not args.quiet)
    names = list(scene.BEAUTY) if args.beauty == "all" else [n for n in args.beauty.split(",") if n]
    for n in names:
        scene.render_beauty(rig, cam, n, PREVIEWS, args.scale)
    print("done in %.1fs" % (time.time() - t0))


main()
