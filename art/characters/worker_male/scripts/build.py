"""Build the male worker from scratch, compare with the sheet, render, save.

    blender -b --factory-startup -P scripts/build.py -- [options]
    (or, from art/characters:  bash charkit/run.sh worker_male [options])

    --draft          coarse grid + few samples (fast iteration)
    --sheet          orthographic front/side/back in the reference sheet's framing, the
                     artwork above the model, a silhouette overlay, IoU per view and a
                     per-height mismatch table
    --turn           orthographic turnaround (front, side, back, three-quarter)
    --beauty a,b,..  perspective studio renders (worker.BEAUTY; `all` for every one)
    --variant NAME   a palette/dirt variant (worker.VARIANTS); renders get a NAME_ prefix
    --scale S        beauty resolution scale (default 1.0)
    --samples N      Cycles samples (default 96, draft 24)
    --save           write source/worker_male.blend and source/mesh_stats.json
"""
import argparse
import json
import os
import sys
import time

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import garment  # noqa: E402
import kit  # noqa: E402
import sheet_ref  # noqa: E402
import studio  # noqa: E402
import worker  # noqa: E402
import workwear  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, ".."))
PREVIEWS, SOURCE = os.path.join(ROOT, "previews"), os.path.join(ROOT, "source")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--turn", action="store_true")
    ap.add_argument("--beauty", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    t0 = time.time()
    os.makedirs(PREVIEWS, exist_ok=True)
    os.makedirs(SOURCE, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    kit.RES = 0.008 if args.draft else 0.004
    pal, dirt = worker.palette(args.variant or None)
    M = workwear.materials(pal, dirt)
    workwear.build(M, worker.DIMS, worker.HEAD, worker.HAIR)
    report = garment.orient_outward()
    rig, cam, floor = studio.setup(samples=args.samples or (24 if args.draft else 96), backdrop=studio.BACKDROP)
    print("built in %.1fs" % (time.time() - t0))

    if args.save:
        stats = studio.mesh_stats(os.path.join(SOURCE, "mesh_stats.json"))
        with open(os.path.join(SOURCE, "mesh_stats.json")) as f:
            stats = json.load(f)
        stats["NORMALS"] = dict(flipped=report["flipped"], closed=report["closed"], open=report["open"], undecided=report["undecided"])
        with open(os.path.join(SOURCE, "mesh_stats.json"), "w") as f:
            json.dump(stats, f, indent=1)
        print("mesh stats:", json.dumps(stats["TOTAL"]))
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(SOURCE, "worker_male.blend"))
    prefix = (args.variant + "_") if args.variant else ""
    if args.sheet:
        panels = sheet_ref.SHEET.render(rig, cam, floor, PREVIEWS, prefix + "sheet")
        sheet_ref.SHEET.compare(panels, PREVIEWS, prefix + "sheet", verbose=not args.quiet)
    if args.turn:
        studio.turnaround(rig, cam, floor, PREVIEWS, worker.HEIGHT, tag=prefix + "turnaround")
    names = list(worker.BEAUTY) if args.beauty == "all" else [n for n in args.beauty.split(",") if n]
    for n in names:
        studio.render_beauty(rig, cam, worker.BEAUTY[n], prefix + n, PREVIEWS, args.scale)
    print("done in %.1fs" % (time.time() - t0))


main()
