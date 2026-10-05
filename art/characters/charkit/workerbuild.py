"""The build driver shared by every worker character (Blender 4.5).

A worker's `scripts/build.py` is three lines: put `charkit` on the path, import its own
`worker` (tables) and `sheet_ref` (its reference sheet), and call `run(...)`:

    blender -b --factory-startup -P <worker>/scripts/build.py -- [options]
    (or, from art/characters:  bash charkit/run.sh <worker> [options])

    --draft          coarse grid + few samples (fast iteration)
    --sheet          orthographic front/side/back in the reference sheet's framing, the
                     artwork above the model, a silhouette overlay, IoU per view and a
                     per-height mismatch table
    --turn           orthographic turnaround (front, side, back, three-quarter)
    --beauty a,b,..  perspective studio renders (worker.BEAUTY; `all` for every one), plus
                     the close-up working views in DEV_VIEWS (never part of `all`)
    --variant NAME   a palette/dirt variant (worker.VARIANTS); renders get a NAME_ prefix
    --scale S        beauty resolution scale (default 1.0)
    --samples N      Cycles samples (default 96, draft 24)
    --save           write source/<name>.blend and source/mesh_stats.json
"""
import argparse
import json
import os
import sys
import time

import bpy

import garment
import kit
import studio
import workwear

# Close-ups for working on the face; rendered on request only (`--beauty face,face_side`).
DEV_VIEWS = {
    "face": (0.0, 0.0, 2.0, 0.75, None, 85.0, (900, 900)),
    "face_side": (0.0, 90.0, 2.0, 0.75, None, 85.0, (900, 900)),
    "face_quarter": (0.0, 35.0, 4.0, 0.75, None, 85.0, (900, 900)),
    "eyes": (0.0, 15.0, 3.0, 0.32, None, 85.0, (900, 600)),
}


def run(root, name, worker, sheet_ref, builder=None):
    """Build `worker`'s figure from its tables, then compare/render/save as asked.

    `builder` is the module (or any object) with `materials(palette, dirt)` and
    `build(M, dims, head_spec, hair_spec)` that makes the figure; None is `workwear`, the
    coverall worker. Another family (the French guard officer) passes its own."""
    builder = builder or workwear
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
    previews, source = os.path.join(root, "previews"), os.path.join(root, "source")
    os.makedirs(previews, exist_ok=True)
    os.makedirs(source, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    kit.RES = 0.008 if args.draft else 0.004
    pal, dirt = worker.palette(args.variant or None)
    M = builder.materials(pal, dirt)
    builder.build(M, worker.DIMS, worker.HEAD, worker.HAIR)
    report = garment.orient_outward()
    rig, cam, floor = studio.setup(samples=args.samples or (24 if args.draft else 96), backdrop=studio.BACKDROP)
    print("built in %.1fs" % (time.time() - t0))

    if args.save:
        path = os.path.join(source, "mesh_stats.json")
        stats = studio.mesh_stats(path)
        stats["NORMALS"] = dict(flipped=report["flipped"], closed=report["closed"], open=report["open"], undecided=report["undecided"])
        with open(path, "w") as f:
            json.dump(stats, f, indent=1)
        print("mesh stats:", json.dumps(stats["TOTAL"]))
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(source, name + ".blend"))
    prefix = (args.variant + "_") if args.variant else ""
    if args.sheet:
        panels = sheet_ref.SHEET.render(rig, cam, floor, previews, prefix + "sheet")
        sheet_ref.SHEET.compare(panels, previews, prefix + "sheet", verbose=not args.quiet)
    if args.turn:
        studio.turnaround(rig, cam, floor, previews, worker.HEIGHT, tag=prefix + "turnaround")
    names = list(worker.BEAUTY) if args.beauty == "all" else [n for n in args.beauty.split(",") if n]
    for n in names:
        view = worker.BEAUTY.get(n)
        if view is None:  # a close-up working view, aimed at this worker's eyes
            v = DEV_VIEWS[n]
            view = v[:4] + (worker.HEAD["eye_z"] - (0.0 if n == "eyes" else 0.03),) + v[5:]
        studio.render_beauty(rig, cam, view, prefix + n, previews, args.scale)
    print("done in %.1fs" % (time.time() - t0))
