"""Beatrice on a VRoid base (the VRoid spike, 2026-10-22): build and render from the committed CC0 VRM files.

    "C:/Program Files/Blender Foundation/Blender 4.5/blender.exe" -b --factory-startup -P Beatrice/scripts/vroid.py -- [mode]
    (from art/characters)

    --candidates         step 1: every candidate VRM (vroid_tables.CANDIDATES, read from the download folder) imported,
                         its head rendered front / three-quarter as imported (Cycles, the studio rig) and in the toon look
    --out DIR            raw renders (default $TEMP/Beatrice_vroid)

Then `python Beatrice/scripts/vroid_sheets.py` lays out the pictures in `previews/`.
"""
import argparse
import math
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import bpy  # noqa: E402
import numpy as np  # noqa: E402

import studio  # noqa: E402
import toonlook  # noqa: E402
import vrmbase  # noqa: E402
import vroid_tables as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--candidates", action="store_true")
ap.add_argument("--only", default="")
ap.add_argument("--out", default="")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
OUT = args.out or os.path.join(tempfile.gettempdir(), "Beatrice_vroid")
os.makedirs(OUT, exist_ok=True)


def fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    vrmbase.enable()
    rig, cam, floor = studio.setup(samples=48)
    floor.hide_render = True
    return rig, cam


def head_frame(vrm):
    """The eyes' midpoint (world) from the VRM's eye bones (fallback: the head bone + 6 cm)."""
    try:
        return (vrm.bone("J_Adj_L_FaceEye") + vrm.bone("J_Adj_R_FaceEye")) / 2.0
    except Exception:
        return vrm.bone("C_Head") + np.array([0.0, 0.0, 0.06])


def head_shots(cam, eye, tag, spec, look=None):
    tx, ty, tz = eye[0], eye[1] - spec["back"], eye[2] + spec["up"]
    for az in spec["azimuths"]:
        a, e = math.radians(az), math.radians(spec["elev"])
        d = spec["dist"]
        cam.data.type, cam.data.lens = "PERSP", spec["lens"]
        cam.location = (tx + d * math.sin(a) * math.cos(e), ty - d * math.cos(a) * math.cos(e), tz + d * math.sin(e))
        studio._look_at(cam, (tx, ty, tz))
        if look:
            bpy.context.view_layer.update()
            toonlook.width_for(cam, (tx, ty, tz), spec["res"], look)
        studio.render(os.path.join(OUT, "%s_%03d.png" % (tag, az)), spec["res"])


def candidates():
    only = [s for s in args.only.split(",") if s]
    for name, rel, cc0 in T.CANDIDATES:
        if only and name not in only:
            continue
        t0 = time.time()
        rig, cam = fresh()
        cam.parent = None
        vrm = vrmbase.load(os.path.join(T.DOWNLOADS, rel), "Cand")
        vrmbase.split_face(vrm)
        vrmbase.materials(vrm, T.PREFIX)
        eye = head_frame(vrm)
        print("candidate", name, "eyes at", eye.round(3), "height", vrmbase.rest_verts(vrm.parts["body"])[:, 2].max().round(3))
        head_shots(cam, eye, "cand_%s_neutral" % name, T.CANDIDATE_SHOTS)
        look = T.LOOK_VROID
        toonlook.apply(look, rig, cam, palette=T.PALETTE)
        head_shots(cam, eye, "cand_%s_toon" % name, T.CANDIDATE_SHOTS, look)
        print("candidate %s done in %.1fs" % (name, time.time() - t0))


if args.candidates:
    candidates()
