"""Read-only audit of the reopened authored .blend; write a small JSON report.

blender --background source/space_marine.blend --python scripts/inspect_space_marine.py
This is a source-asset audit, not an engine, rigging, UV or deformation test.
"""
import hashlib
import json
import math
from pathlib import Path
import bpy

root=Path(__file__).resolve().parents[1]
scene=bpy.context.scene
depsgraph=bpy.context.evaluated_depsgraph_get()
character=[]
for obj in scene.objects:
    if any(col.name[:2].isdigit() and int(col.name[:2])<90 for col in obj.users_collection):
        if obj.type in {"MESH","CURVE","FONT"}:character.append(obj)
errors=[];source_vertices=source_polygons=vertices=triangles=degenerate=0
minimum=[math.inf]*3;maximum=[-math.inf]*3
material_names=set()
for obj in character:
    if not obj.data.materials:errors.append("No material: "+obj.name)
    material_names.update(material.name for material in obj.data.materials if material)
    if obj.type=="MESH":
        source_vertices+=len(obj.data.vertices);source_polygons+=len(obj.data.polygons)
    evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh()
    vertices+=len(mesh.vertices);mesh.calc_loop_triangles();triangles+=len(mesh.loop_triangles)
    for vertex in mesh.vertices:
        co=evaluated.matrix_world@vertex.co
        if not all(math.isfinite(value) for value in co):errors.append("Nonfinite geometry: "+obj.name);break
        for axis in range(3):minimum[axis]=min(minimum[axis],co[axis]);maximum[axis]=max(maximum[axis],co[axis])
    degenerate+=sum(triangle.area<1e-12 for triangle in mesh.loop_triangles)
    evaluated.to_mesh_clear()
external_images=[]
for image in bpy.data.images:
    if image.source=="FILE" and not image.packed_file:
        path=Path(bpy.path.abspath(image.filepath));external_images.append(str(path))
        if not path.is_file():errors.append("Missing external image: "+str(path))
references=[image.name for image in bpy.data.images if image.packed_file]
reference=root.parent/"Space Marine Turnaround Sheet.png"
actual_hash=hashlib.sha256(reference.read_bytes()).hexdigest()
if scene.get("reference_sha256")!=actual_hash:errors.append("Reference hash mismatch")
if not references:errors.append("The supplied reference is not packed")
for view in ("front","side","back","hero","closeup","elevated","details"):
    if "Camera / "+view not in bpy.data.objects:errors.append("Missing camera: "+view)
armatures=[obj.name for obj in scene.objects if obj.type=="ARMATURE"]
if armatures:errors.append("Unexpected skeleton added to deferred rigging source")
live_cloth=[obj.name for obj in scene.objects if any(mod.type=="CLOTH" for mod in obj.modifiers)]
if live_cloth:errors.append("Live cloth/cache dependency remains in delivery source")
glass=bpy.data.objects.get("VISOR GLASS / convex optical shield")
glass_report={}
if not glass:
    errors.append("Missing independently selectable visor glass")
else:
    mat=glass.data.materials[0]
    bsdf=mat.node_tree.nodes.get("Principled BSDF")
    material_users=[obj.name for obj in character if mat.name in obj.data.materials]
    glass_report={"object":glass.name,"material":mat.name,"object_mask_id":glass.pass_index,
                  "material_mask_id":mat.pass_index,"roughness":bsdf.inputs["Roughness"].default_value,
                  "coat_weight":bsdf.inputs["Coat Weight"].default_value,
                  "coat_roughness":bsdf.inputs["Coat Roughness"].default_value,
                  "game_gloss_override":mat.get("game_gloss"),
                  "material_used_by":material_users,
                  "separate_from_frame_and_gasket":True}
    if glass.pass_index!=1 or mat.pass_index!=1:errors.append("Visor mask ID is not 1")
    if any(obj!=glass and obj.pass_index==1 for obj in scene.objects):errors.append("Glass object mask ID reused")
    if any(material!=mat and material.pass_index==1 for material in bpy.data.materials):errors.append("Glass material mask ID reused")
    if any(name!=glass.name for name in material_users):errors.append("Glass material assigned outside the visor")
report={
    "opened_file":Path(bpy.data.filepath).name,
    "blender_version":bpy.app.version_string,
    "reopened_successfully":True,
    "character_objects":len(character),
    "source_mesh_vertices":source_vertices,"source_mesh_polygons":source_polygons,
    "evaluated_vertices_including_curve_detail":vertices,"evaluated_triangles":triangles,
    "triangles_below_1e-12_square_metres":degenerate,
    "bounds_min_metres":minimum,"bounds_max_metres":maximum,
    "dimensions_metres":[b-a for a,b in zip(minimum,maximum)],
    "materials":sorted(material_names),"packed_images":references,
    "external_unpacked_images":external_images,"armatures":armatures,
    "live_cloth_modifiers":live_cloth,"visor_glass":glass_report,
    "reference_sha256":actual_hash,"errors":errors,
    "limitations":["No rig, skin weights or deformation test", "No authored UV atlas or baked textures",
                   "Dense parametric source and remeshed trousers; game retopology and LODs remain",
                   "Material appearance depends on Blender procedural shaders",
                   "Separate layered shells overlap intentionally; this is not one watertight printable mesh",
                   "No runtime export or engine compatibility validation"]
}
(root/"source"/"validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2))
if errors:raise RuntimeError("Source audit found: "+"; ".join(errors))
