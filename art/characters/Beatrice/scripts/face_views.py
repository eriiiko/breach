"""Beatrice's face pictures (B2a): the saved build (`source/Beatrice.blend`) opened and its head rendered, never saved.

    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/face_views.py -- [options]
    (from art/characters; then `python Beatrice/scripts/face_sheets.py` lays out face_vs_art.jpg, face_turn.jpg, face_closeup.jpg)

    --blend PATH     another saved build
    --head NAME      rebuild the head on the opened build from `beatrice.head(NAME)` first (e.g. `apose`, B2a's A-pose-size head)
    --look NAME      B2b: the look (`beatrice.LOOKS`; default `beatrice.LOOK`, the toon look; `pbr` the physically-lit one)
    --only a,b       art, turn, close (default all)
    --samples N      Cycles samples (default 96)
    --out DIR        where the raw RGBA renders go (default $TEMP/Beatrice_face)

The shots are `beatrice.FACE_SHOTS`, in HEAD space (B2b: placed and sized by the head's scale, so the art frames lie on the
close-ups at any head size). In the pbr look the turn and the close-ups use the optional portrait light
(`POSE["lights"]["portrait"]`) and the art the default studio rig; the toon look brings its own light rig. The raw renders of a
look other than the default get a `<look>_` prefix.
"""
import argparse
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import bpy  # noqa: E402
import beatrice  # noqa: E402
import studio  # noqa: E402
import suitpose  # noqa: E402
import suitlook  # noqa: E402
import animehead  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--blend", default="")
ap.add_argument("--head", default="")
ap.add_argument("--only", default="art,turn,close")
ap.add_argument("--samples", type=int, default=96)
ap.add_argument("--out", default="")
ap.add_argument("--look", default="")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
root = os.path.normpath(os.path.join(HERE, ".."))
bpy.ops.wm.open_mainfile(filepath=args.blend or os.path.join(root, "source", "Beatrice.blend"))
sc = bpy.context.scene
sc.cycles.samples = args.samples
print("render device:", suitpose.use_gpu())
spec = beatrice.head(args.head or None)
if args.head:
    spec = beatrice.head(args.head)
    for ob in list(bpy.data.collections["Head"].objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    mats = animehead.materials("beatrice", beatrice.palette(), spec["eyes"], spec["face"])
    animehead.build("Beatrice", spec["face"], spec["eyes"], mats, "Head", h=spec.get("res", 0.0007), hair=spec.get("hair"),
                    scale=spec.get("scale", 1.0), hair_face=spec.get("hair_face"))
head = animehead.Head(spec["face"], spec["eyes"], spec.get("scale", 1.0))
W = head.world
out = args.out or os.path.join(tempfile.gettempdir(), "Beatrice_face")
os.makedirs(out, exist_ok=True)
lname = suitlook.name_of(beatrice, args.look)
tag = (args.head + "_" if args.head else "") + ("" if lname == suitlook.name_of(beatrice) else lname + "_")
cam, rig = bpy.data.objects["Cam"], bpy.data.objects["StudioRig"]
look = suitlook.prepare(beatrice, lname, args.head or None)
suitlook.apply(beatrice, look, rig, cam)
floor = bpy.data.objects.get("Floor")
if floor:
    floor.hide_render = True
hair = [o for o in bpy.data.objects if o.name.startswith("Beatrice_Hair")]
S = beatrice.FACE_SHOTS
only = args.only.split(",")

if "art" in only:
    cam.data.type = "ORTHO"
    cam.data.clip_start, cam.data.clip_end = 0.01, 40.0
    for name, s in S["art"].items():
        cam.data.ortho_scale = s["width"] * head.s
        u, z = s["centre"]
        if s["side"] == "front":
            c = W([u, 0.0, z])
            cam.location, cam.rotation_euler = (c[0], -3.0, c[2]), (math.radians(90.0), 0.0, 0.0)
        else:   # from her right: image right is her front
            c = W([0.0, u, z])
            cam.location, cam.rotation_euler = (-3.0, c[1], c[2]), (math.radians(90.0), 0.0, math.radians(-90.0))
        suitlook.width_mpp(look, cam.data.ortho_scale / 1254, cam)
        for with_hair in (True, False):
            for o in hair:
                o.hide_render = not with_hair
            studio.render(os.path.join(out, "%sart_%s%s.png" % (tag, name, "" if with_hair else "_nohair")), (1254, 1254), transparent=True)
    for o in hair:
        o.hide_render = False

if not look:
    suitpose.add_lights(beatrice.POSE["lights"]["portrait"], rig)
if "turn" in only:
    T = S["turn"]
    cam.data.type = "PERSP"
    cam.data.lens = T["lens"]
    tx, ty, tz = W(T["target"])
    d = T["dist"] * head.s
    for az in T["azimuths"]:
        a, e = math.radians(az), math.radians(T["elev"])
        cam.location = (tx + d * math.sin(a) * math.cos(e), ty - d * math.cos(a) * math.cos(e), tz + d * math.sin(e))
        studio._look_at(cam, (tx, ty, tz))
        suitlook.width(look, cam, (tx, ty, tz), T["res"])
        studio.render(os.path.join(out, "%sturn_%03d.png" % (tag, az)), T["res"])
if "close" in only:
    cam.data.type = "PERSP"
    for name, s in S["close"].items():
        cam.data.lens = s["lens"]
        tgt = W(s["target"])
        cam.location = tuple(W(s["cam"]))
        studio._look_at(cam, tuple(tgt))
        suitlook.width(look, cam, tgt, s["res"])
        studio.render(os.path.join(out, "%sclose_%s.png" % (tag, name)), s["res"])
print("face views ->", out)
