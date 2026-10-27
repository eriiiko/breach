"""Common build driver for a scripted character.

A character's `scripts/build.py` puts `charkit` on the path and calls `run(...)`:

    blender -b --factory-startup -P <character>/scripts/build.py -- [options]

    --draft          coarse grid + few samples (fast iteration)
    --turn           orthographic turnaround: front, side, back, three-quarter
    --beauty a,b,..  perspective studio renders (`all` for every one)
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

import kit
import studio


def run(root, name, build, make_materials, beauty, height, backdrop=studio.BACKDROP):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--turn", action="store_true")
    ap.add_argument("--beauty", default="")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    t0 = time.time()
    previews, source = os.path.join(root, "previews"), os.path.join(root, "source")
    os.makedirs(previews, exist_ok=True)
    os.makedirs(source, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    kit.RES = 0.008 if args.draft else 0.004
    build(make_materials())
    rig, cam, floor = studio.setup(samples=args.samples or (24 if args.draft else 96), backdrop=backdrop)
    print("built in %.1fs" % (time.time() - t0))
    if args.save:
        stats = studio.mesh_stats(os.path.join(source, "mesh_stats.json"))
        print("mesh stats:", json.dumps(stats["TOTAL"]))
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(source, name + ".blend"))
    if args.turn:
        studio.turnaround(rig, cam, floor, previews, height, backdrop=backdrop)
    for n in (list(beauty) if args.beauty == "all" else [n for n in args.beauty.split(",") if n]):
        studio.render_beauty(rig, cam, beauty[n], n, previews, args.scale)
    print("done in %.1fs" % (time.time() - t0))
