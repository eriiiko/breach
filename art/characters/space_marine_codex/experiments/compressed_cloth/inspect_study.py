"""Reopen both applied study files, audit them, and recheck the frozen baseline.

Run with Blender --background --python THIS_FILE. This does not save any .blend.
"""
import hashlib
import json
import math
from pathlib import Path

import bmesh
import bpy

HERE=Path(__file__).resolve().parent
ASSET=HERE.parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera_record(name):
    obj=bpy.data.objects["Camera / "+name]
    return {"location":list(obj.location),"rotation":list(obj.rotation_euler),
            "type":obj.data.type,"ortho_scale":obj.data.ortho_scale}


def audit_candidate(number):
    path=HERE/f"candidate_{number}.blend"
    before=sha256(path)
    bpy.ops.wm.open_mainfile(filepath=str(path))
    obj=bpy.data.objects["STUDY / candidate sleeve"]
    mesh=obj.data
    mesh.calc_loop_triangles()
    bm=bmesh.new();bm.from_mesh(mesh)
    non_manifold=sum(not edge.is_manifold for edge in bm.edges)
    bm.free()
    missing_images=[image.name for image in bpy.data.images
                    if image.source=="FILE" and not image.packed_file
                    and not Path(bpy.path.abspath(image.filepath)).is_file()]
    live_cloth=[item.name for item in bpy.context.scene.objects
                if any(mod.type=="CLOTH" for mod in item.modifiers)]
    materials={}
    for name in ("Warm woven pressure fabric","Charcoal flex textile","Graphite elastomer"):
        mat=bpy.data.materials[name]
        principled=next(node for node in mat.node_tree.nodes if node.type=="BSDF_PRINCIPLED")
        materials[name]={socket:list(principled.inputs[socket].default_value)
                         if socket=="Base Color" else principled.inputs[socket].default_value
                         for socket in ("Base Color","Roughness","Metallic","Coat Weight","Coat Roughness")}
    lights={item.name:{"type":item.data.type,"energy":item.data.energy,
                      "color":list(item.data.color),"location":list(item.location),
                      "rotation":list(item.rotation_euler),"size":getattr(item.data,"size",None)}
            for item in bpy.context.scene.objects if item.type=="LIGHT"}
    report={"file":path.name,"sha256":before,"source_vertices":len(mesh.vertices),
            "source_triangles":len(mesh.loop_triangles),"non_manifold_edges":non_manifold,
            "degenerate_triangles_below_1e_12_square_metres":sum(t.area<1e-12 for t in mesh.loop_triangles),
            "finite_geometry":all(math.isfinite(value) for v in mesh.vertices for value in v.co),
            "live_cloth_modifiers":live_cloth,"shape_keys_on_candidate":bool(mesh.shape_keys),
            "candidate_modifiers":[mod.type for mod in obj.modifiers],
            "candidate_material_slots":[mat.name for mat in mesh.materials],
            "saved_resolution":[bpy.context.scene.render.resolution_x,bpy.context.scene.render.resolution_y],
            "saved_cycles_samples":bpy.context.scene.cycles.samples,
            "external_cache_files":len(bpy.data.cache_files),"missing_images":missing_images,
            "packed_images":[image.name for image in bpy.data.images if image.packed_file],
            "front_camera":camera_record("front"),"side_camera":camera_record("side"),
            "lighting":lights,"material_controls":materials,
            "file_unchanged_by_audit":sha256(path)==before}
    assert not live_cloth and not mesh.shape_keys and not obj.modifiers
    assert not missing_images and not bpy.data.cache_files
    assert report["finite_geometry"] and not non_manifold
    assert report["file_unchanged_by_audit"]
    return report


frozen=json.loads((HERE/"baseline_files.json").read_text(encoding="utf-8"))["files"]
changed=[name for name,digest in frozen.items() if sha256(ASSET/name)!=digest]
assert not changed,changed
reports=[audit_candidate(number) for number in (1,2)]
comparison={key:reports[0][key]==reports[1][key]
            for key in ("front_camera","side_camera","lighting","material_controls")}
assert all(comparison.values()),comparison
result={"baseline_files_checked":len(frozen),"baseline_files_changed":changed,
        "same_comparison_conditions":comparison,"candidates":reports}
(HERE/"reopened_validation.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print("REOPENED_STUDY_AUDIT",json.dumps({"baseline_files_checked":len(frozen),
      "same_comparison_conditions":comparison,"candidate_counts":[(r["source_vertices"],r["source_triangles"]) for r in reports]}),flush=True)
