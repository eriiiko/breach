"""Bounded one-sleeve study, using the committed source's materials and lighting.

blender --background --python THIS_FILE -- --candidate 1
The baseline .blend, seven renders and their manifest are hashed before and after.
Only experiment outputs are written. Simulation is applied before saving the study.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

HERE=Path(__file__).resolve().parent
ASSET=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ASSET/"scripts"))
import build_space_marine as author
from cloth_authoring import compressed_limb,bake_compressed_cloth,sleeve_profiles


def baseline_files():
    paths=[ASSET/"source/space_marine.blend",ASSET/"source/model_stats.json",ASSET/"source/validation.json",
           ASSET/"README.md",ASSET/"previews/index.html",ASSET/"previews/render_manifest.json"]
    paths.extend(ASSET/"previews"/(name+".png") for name in
                 ("front","side","back","hero","closeup","elevated","details","contact_sheet"))
    return {str(path.relative_to(ASSET)).replace("\\","/"):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def check_baseline():
    path=HERE/"baseline_files.json"
    current=baseline_files()
    if path.is_file():
        frozen=json.loads(path.read_text(encoding="utf-8"))["files"]
        if current!=frozen:raise RuntimeError("Committed baseline changed during bounded study")
    else:
        path.write_text(json.dumps({"baseline_commit":"a298b68","files":current},indent=2)+"\n",encoding="utf-8")
    return current


def arm_object(obj):
    return obj.name.startswith(("Upper arm R /","Sleeve R /","Elbow R /","Wrist R /",
                                "Glove R /","Pauldron R /","Shoulder R /"))


def skin_object(obj):
    return obj.name.startswith(("Upper arm R / flex","Sleeve R /","Elbow R / compressed","Wrist R / seal"))


def camera_setup(scene,name):
    camera=bpy.data.objects["Camera / "+("front" if name=="front" else "side")]
    camera.location=(.36,-25,1.245) if name=="front" else (25,0,1.245)
    author.aim(camera,(.36,0,1.195))
    camera.data.ortho_scale=.91
    scene.camera=camera


def render_pair(prefix):
    scene=bpy.context.scene
    scene.render.resolution_x=1000;scene.render.resolution_y=1400
    scene.render.resolution_percentage=100;scene.cycles.samples=96
    scene.cycles.use_denoising=True
    for name in ("front","side"):
        camera_setup(scene,name)
        scene.render.filepath=str(HERE/(prefix+"_"+name+".png"))
        bpy.ops.render.render(write_still=True)
        print("STUDY_RENDER",scene.render.filepath,flush=True)


def final_seams(obj,meta):
    """Trace the baked surface, including its simulated displacement."""
    pieces=[];sides=meta["sides"];zs=meta["zs"]
    for piece,(low,high) in enumerate(meta["cloth_ranges"]):
        rows=[row for row,z in enumerate(zs) if low+.013<z<high-.013]
        for side,angle in enumerate((-1.4,1.55)):
            for binding,offset in ((False,0),(True,.027)):
                a=((angle+offset)%math.tau)/math.tau*sides
                index=int(a);fraction=a-index
                points=[]
                for j,row in enumerate(rows):
                    left=obj.data.vertices[row*sides+index]
                    right=obj.data.vertices[row*sides+(index+1)%sides]
                    co=left.co.lerp(right.co,fraction)
                    normal=left.normal.lerp(right.normal,fraction).normalized()
                    edge=max(0,1-min(j,len(rows)-1-j)/5)
                    points.append(co+normal*(.001-.0024*edge*edge))
                seam=author.curve(f"STUDY / baked seam {piece} {side} {binding}",points,
                                  .00065 if binding else .00115,
                                  "Ivory binding" if binding else "Reinforced textile seams","01 / Pressure garment")
                seam.parent=obj.parent;pieces.append(seam)
    return pieces


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--candidate",type=int,choices=(1,2),required=True)
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    # Candidate 2 is specified only after inspecting the first, never a parameter grid.
    parameters={1:{"compression":.13,"bending":.045,"pressure":.6}}
    if args.candidate not in parameters:raise RuntimeError("Candidate 2 has not been specified from the first review")
    prefix=f"candidate_{args.candidate}"
    baseline=check_baseline()
    bpy.ops.wm.open_mainfile(filepath=str(ASSET/"source/space_marine.blend"))
    author.COLLECTIONS={col.name:col for col in bpy.data.collections}
    author.MATERIALS={mat.name:mat for mat in bpy.data.materials}
    scene=bpy.context.scene
    try:
        preferences=bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type="OPTIX";preferences.get_devices()
        for device in preferences.devices:device.use=device.type=="OPTIX"
        if any(device.type=="OPTIX" for device in preferences.devices):scene.cycles.device="GPU"
    except Exception as exc:print("Configured Cycles device retained:",exc)
    originals=[]
    for obj in scene.objects:
        is_character=any(col.name[:2].isdigit() and int(col.name[:2])<90 for col in obj.users_collection)
        if is_character and obj.type in {"MESH","CURVE","FONT"}:
            obj.hide_render=not arm_object(obj)
            obj.hide_set(not arm_object(obj))
            if skin_object(obj):originals.append(obj)
    if args.candidate==1:render_pair("baseline")
    for obj in originals:obj.hide_render=True;obj.hide_set(True)
    obj,meta=compressed_limb(author,"STUDY / candidate sleeve",sleeve_profiles(),[(1.008,1.137),(1.239,1.427)],**parameters[args.candidate])
    obj.parent=bpy.data.objects["SPACE MARINE / move the complete source"]
    bake_compressed_cloth([obj])
    deltas=[(obj.data.vertices[i].co-meta["final"][i]).length for i in meta["pin_indices"]]
    sub=obj.modifiers.new("Baked cloth surface interpolation","SUBSURF");sub.levels=1;sub.render_levels=1
    # Interpolation is applied: the saved study needs no cloth cache or simulation.
    seams=final_seams(obj,meta)
    bpy.context.view_layer.objects.active=obj
    bpy.ops.object.select_all(action="DESELECT");obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=sub.name)
    positions=[v.co for v in obj.data.vertices]
    obj.data.calc_loop_triangles()
    metrics={"candidate":args.candidate,"parameters":parameters[args.candidate],
             "simulation_frames":42,"solver_quality":10,"pressure_factor":.5,
             "source_vertices":len(obj.data.vertices),"source_triangles":len(obj.data.loop_triangles),
             "max_pinned_endpoint_error_metres":max(deltas),
             "finite_geometry":all(math.isfinite(value) for co in positions for value in co),
             "bounds_min_local_metres":[min(p[i] for p in positions) for i in range(3)],
             "bounds_max_local_metres":[max(p[i] for p in positions) for i in range(3)],
             "live_cloth_modifiers":sum(mod.type=="CLOTH" for item in scene.objects for mod in item.modifiers),
             "shape_keys_on_baked_candidate":bool(obj.data.shape_keys),
             "materials":[mat.name for mat in obj.data.materials],
             "baseline_source_sha256":baseline["source/space_marine.blend"],
             "seams_follow_applied_surface":True,"visual_verdict":"pending inspection"}
    (HERE/(prefix+"_metrics.json")).write_text(json.dumps(metrics,indent=2)+"\n",encoding="utf-8")
    camera_setup(scene,"front")
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=="VIEW_3D":
                area.spaces.active.region_3d.view_distance=1.3
                area.spaces.active.region_3d.view_location=(.36,0,1.20)
                area.spaces.active.region_3d.view_rotation=scene.camera.rotation_euler.to_quaternion()
    scene["study_only"]="Single sleeve experiment; baseline file is unchanged; no rollout approved"
    bpy.ops.wm.save_as_mainfile(filepath=str(HERE/(prefix+".blend")),compress=True)
    render_pair(prefix)
    check_baseline()
    print("CANDIDATE_COMPLETE",prefix,json.dumps(metrics),flush=True)


if __name__=="__main__":main()
