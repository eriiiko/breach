"""Render an existing, possibly hand-edited .blend without rebuilding or saving it.

blender --background source/space_marine.blend --python scripts/render_space_marine.py
Arguments after --: --views hero,front,side,back,closeup,elevated,details --draft
"""
import argparse
from pathlib import Path
import sys
import bpy

parser=argparse.ArgumentParser()
parser.add_argument("--views",default="hero,front,side,back,closeup,elevated,details")
parser.add_argument("--draft",action="store_true")
parser.add_argument("--samples",type=int,default=160)
parser.add_argument("--output",type=Path)
args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
root=Path(__file__).resolve().parents[1]
output=args.output or root/"previews"
output.mkdir(exist_ok=True,parents=True)
scene=bpy.context.scene
scene.render.engine="CYCLES"
scene.cycles.samples=40 if args.draft else args.samples
scene.cycles.use_denoising=True
try:
    preferences=bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type="OPTIX";preferences.get_devices()
    for device in preferences.devices:device.use=device.type=="OPTIX"
    if any(device.type=="OPTIX" for device in preferences.devices):scene.cycles.device="GPU"
except Exception as exc:
    print("Using configured Cycles device:",exc)
for name in args.views.split(","):
    camera=bpy.data.objects.get("Camera / "+name)
    if not camera or camera.type!="CAMERA":raise ValueError("Missing review camera: "+name)
    scene.camera=camera
    scene.render.resolution_x=900 if args.draft else (1800 if name=="closeup" else 1600)
    scene.render.resolution_y=1100 if args.draft else (1600 if name=="closeup" else 2000)
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    scene.render.filepath=str(output/(("draft_" if args.draft else "")+name+".png"))
    bpy.ops.render.render(write_still=True)
    print("RENDERED",scene.render.filepath,flush=True)
print("Existing .blend left unchanged.")
