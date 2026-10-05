"""Space Marine (Codex): bounded helmet-only edit of the approved source.

Blender --background --python THIS_FILE -- --build --render both --draft
Use --render both without --build to render the existing saved sources unchanged.
The approved source, generator and original gallery are always hash-checked.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import bmesh
from mathutils import Vector

VARIANT=Path(__file__).resolve().parents[1]
ASSET=VARIANT.parents[1]
APPROVED=ASSET/"versions/approved_v1"
SOURCE=VARIANT/"source/space_marine_codex_head_v2.blend"
sys.path.insert(0,str(ASSET/"scripts"))
import build_space_marine as author

GROUP="03 / Helmet and neck seal"
GLASS_GROUP="03 / Visor glass selection"
GLASS_MAT="VISOR GLASS / olive optical coating"
APERTURE=1.32
CENTRE_PHI=1.58


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserved_files():
    manifest=json.loads((APPROVED/"manifest.json").read_text(encoding="utf-8"))
    changed=[name for name,digest in manifest["preserved_files_relative_to_asset"].items()
             if sha256(ASSET/name)!=digest]
    frozen=APPROVED/manifest["snapshot"]
    if sha256(frozen)!=manifest["snapshot_sha256"]:changed.append("Approved snapshot")
    if changed:raise RuntimeError("Preserved original changed: "+str(changed))
    return frozen,manifest


def is_head(obj):
    return obj.name.startswith(("Helmet /","Helmet V2 /","Visor /","VISOR GLASS /"))


def is_character(obj):
    return obj.type in {"MESH","CURVE","FONT"} and any(
        col.name[:2].isdigit() and int(col.name[:2])<90 for col in obj.users_collection)


def simple(value):
    if isinstance(value,(str,int,float,bool)) or value is None:return value
    try:return list(value)
    except TypeError:return str(value)


def material_record(mat):
    nodes=[]
    for node in sorted(mat.node_tree.nodes,key=lambda item:item.name):
        record={"name":node.name,"type":node.bl_idname,
                "inputs":{str(i)+" / "+sock.name:simple(sock.default_value)
                          for i,sock in enumerate(node.inputs) if hasattr(sock,"default_value")}}
        for key in ("operation","blend_type","noise_dimensions","normalize","wave_type",
                    "bands_direction","wave_profile","interpolation","use_clamp"):
            if hasattr(node,key):record[key]=simple(getattr(node,key))
        if hasattr(node,"color_ramp"):
            record["ramp"]={"interpolation":node.color_ramp.interpolation,
                            "elements":[[e.position,list(e.color)] for e in node.color_ramp.elements]}
        nodes.append(record)
    links=sorted((link.from_node.name,link.from_socket.identifier,link.to_node.name,link.to_socket.identifier)
                 for link in mat.node_tree.links)
    return {"nodes":nodes,"links":links,"pass_index":mat.pass_index,
            "diffuse_color":list(mat.diffuse_color)}


def fingerprints():
    """Hash evaluated non-head geometry, world transforms and assigned shaders."""
    depsgraph=bpy.context.evaluated_depsgraph_get()
    geometry={};materials={}
    for obj in sorted(bpy.context.scene.objects,key=lambda item:item.name):
        if not is_character(obj) or is_head(obj):continue
        evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh()
        digest=hashlib.sha256()
        for collection,property_name,code,count in ((mesh.vertices,"co","f",3),
                                                   (mesh.loops,"vertex_index","i",1),
                                                   (mesh.polygons,"loop_total","i",1),
                                                   (mesh.polygons,"material_index","i",1)):
            values=array(code,[0])*(len(collection)*count)
            collection.foreach_get(property_name,values);digest.update(values.tobytes())
        digest.update(array("f",[value for row in obj.matrix_world for value in row]).tobytes())
        assignments=[mat.name if mat else None for mat in obj.data.materials]
        digest.update(json.dumps(assignments).encode())
        digest.update(str(obj.hide_render).encode())
        geometry[obj.name]=digest.hexdigest()
        evaluated.to_mesh_clear()
        for mat in obj.data.materials:
            if mat:materials[mat.name]=hashlib.sha256(json.dumps(material_record(mat),sort_keys=True).encode()).hexdigest()
    return {"objects":geometry,"materials":materials}


def shell_point(a,p):
    """Compact calotte, near-planar lower cheeks, and a projected facial envelope."""
    s=math.sin(p);c=math.cos(p)
    projection=.014*math.exp(-((p-1.54)/.70)**4)*max(0,math.cos(a))**2
    return Vector((.153*s*math.sin(a)*(1-.055*max(0,-c)),
                   .006-(.164+projection)*s*math.cos(a),
                   1.744+.147*c))


def normal_at(a,p):
    da=shell_point(a+.0001,p)-shell_point(a-.0001,p)
    dp=shell_point(a,p+.0001)-shell_point(a,p-.0001)
    n=dp.cross(da)
    return n.normalized() if n.length>1e-10 else Vector((0,0,1))


def surface(a,p,offset=0):
    return shell_point(a,p)+normal_at(a,p)*offset


def upper_boundary(a):
    u=min(1,abs(a)/APERTURE)
    return .995+.030*u*u+.10*u**8+.455*(1-math.sqrt(max(0,1-u*u)))**3


def lower_boundary(a):
    u=min(1,abs(a)/APERTURE)
    return 2.22-.12*u*u-.32*u**6-.20*u**16


def top_end(a):
    return upper_boundary(a) if abs(a)<APERTURE else CENTRE_PHI


def bottom_start(a):
    return lower_boundary(a) if abs(a)<APERTURE else CENTRE_PHI


def lower_end(a):
    return 2.35-.07*(1-max(0,math.cos(a)))


def mesh_part(name,vertices,faces,material="Ivory ceramic enamel",group=GROUP,thickness=0):
    obj=author.mesh_object(name,vertices,faces,material,group)
    # Open parametric patches must face outward before their shell thickness is added.
    centre=Vector((0,.006,1.744))
    if sum(face.normal.dot(face.center-centre) for face in obj.data.polygons)<0:
        bm=bmesh.new();bm.from_mesh(obj.data)
        bmesh.ops.reverse_faces(bm,faces=list(bm.faces));bm.to_mesh(obj.data);bm.free()
    if thickness:author.solidify(obj,thickness)
    return obj


def patch(name,amin,amax,pmin,pmax,offset=0,material="Ivory ceramic enamel",
          thickness=.005,columns=48,rows=28):
    vertices=[];faces=[]
    for j in range(rows+1):
        t=j/rows
        for k in range(columns+1):
            a=amin+(amax-amin)*k/columns
            lo=pmin(a) if callable(pmin) else pmin
            hi=pmax(a) if callable(pmax) else pmax
            vertices.append(surface(a,lo+(hi-lo)*t,offset))
    for j in range(rows):
        for k in range(columns):
            q=j*(columns+1)+k;faces.append((q,q+columns+1,q+columns+2,q+1))
    return mesh_part(name,vertices,faces,material,thickness=thickness)


def aperture_loop(samples=256):
    # Cosine sampling resolves the rounded temporal corners without polar aliasing.
    angles=[-APERTURE*math.cos(math.pi*j/samples) for j in range(samples+1)]
    return [(a,upper_boundary(a)) for a in angles]+[(a,lower_boundary(a)) for a in reversed(angles[1:-1])]


def aperture_band(name,profiles,material):
    contour=aperture_loop();count=len(contour);vertices=[];faces=[]
    for scale,offset in profiles:
        vertices.extend(surface(a*scale,CENTRE_PHI+(p-CENTRE_PHI)*scale,offset) for a,p in contour)
    for row in range(len(profiles)-1):
        for k in range(count):
            n=(k+1)%count
            faces.append((row*count+k,(row+1)*count+k,(row+1)*count+n,row*count+n))
    return mesh_part(name,vertices,faces,material,thickness=.003)


def curved_panel(name,outline,offset,material):
    # Boundary bevel is formed in the curved surface, not a floating flat plaque.
    centre=tuple(sum(point[i] for point in outline)/len(outline) for i in (0,1))
    outline=[(a+(b-a)*t/12,p+(q-p)*t/12)
             for (a,p),(b,q) in zip(outline,outline[1:]+outline[:1]) for t in range(12)]
    vertices=[];faces=[];count=len(outline)
    for radius,depth in ((0,offset),(.45,offset),(.88,offset),(.97,offset-.0006),(1,offset-.0018)):
        vertices.extend(surface(centre[0]+(a-centre[0])*radius,
                                centre[1]+(p-centre[1])*radius,depth) for a,p in outline)
    for row in range(4):
        for k in range(count):
            n=(k+1)%count
            faces.append((row*count+k,row*count+n,(row+1)*count+n,(row+1)*count+k))
    return mesh_part(name,vertices,faces,material,thickness=.004)


def build_head():
    # Under-shell is visible only within manufactured panel clearances.
    patch("Helmet V2 / continuous crown substrate",-math.pi,math.pi,.0001,top_end,
          -.001,"Recess",.003,192,44)
    patch("Helmet V2 / continuous jaw substrate",-math.pi,math.pi,bottom_start,lower_end,
          -.001,"Recess",.003,192,24)
    patch("Helmet V2 / removable crown cap",-math.pi,math.pi,.0001,.675,.001,
          columns=192,rows=30)
    divisions=(-math.pi,-2.18,-1.32,-.43,.43,1.32,2.18,math.pi)
    for index,(lo,hi) in enumerate(zip(divisions,divisions[1:])):
        patch(f"Helmet V2 / upper shell panel {index+1}",lo+.005,hi-.005,.686,
              lambda a:top_end(a)-.008,.0015)
    divisions=(-math.pi,-2.36,-1.32,-.53,.53,1.32,2.36,math.pi)
    for index,(lo,hi) in enumerate(zip(divisions,divisions[1:])):
        patch(f"Helmet V2 / cheek and nape panel {index+1}",lo+.005,hi-.005,
              lambda a:bottom_start(a)+.009,lambda a:lower_end(a)-.007,.002,
              columns=48,rows=30)
    # A broad optical pane with clipped lower corners, rather than a rectangle.
    contour=aperture_loop();count=len(contour);vertices=[];faces=[];rings=52
    for row in range(rings+1):
        r=row/rings
        for a,p in contour:
            vertices.append(surface(a*r*.995,CENTRE_PHI+(p-CENTRE_PHI)*r*.995,
                                    .0030+.0025*(1-r*r)))
    for row in range(rings):
        for k in range(count):
            n=(k+1)%count
            faces.append((row*count+k,row*count+n,(row+1)*count+n,(row+1)*count+k))
    glass=mesh_part("VISOR GLASS / convex optical shield",vertices,faces,GLASS_MAT,GLASS_GROUP,.0035)
    glass.pass_index=1
    glass["surface_role"]="Independent olive visor glass; head_v2 contour"
    glass["mask_id"]=1
    aperture_band("Helmet V2 / continuous black visor gasket",
                  ((.985,.0018),(.996,.0050),(1.025,.0053),(1.041,.0025)),"Graphite elastomer")
    aperture_band("Helmet V2 / shaped ivory retaining frame",
                  ((1.036,.0025),(1.047,.0045),(1.069,.0035),(1.079,.0007)),"Ivory ceramic enamel")
    # Curved temple carriers seat behind the optical edge and connect to the jaw.
    outline=[(1.37,1.29),(1.52,1.24),(1.77,1.35),(1.84,1.76),
             (1.76,2.12),(1.53,2.20),(1.40,2.06),(1.43,1.80),(1.35,1.67)]
    for side in (-1,1):
        points=[(side*a,p) for a,p in outline]
        curved_panel(f"Helmet V2 / temple carrier seal {side}",points,.003,"Graphite elastomer")
        centre=(side*1.58,1.74)
        face=[(centre[0]+(a-centre[0])*.90,centre[1]+(p-centre[1])*.94) for a,p in points]
        curved_panel(f"Helmet V2 / temple carrier armour {side}",face,.007,"Ivory ceramic enamel")
        latch=[(side*a,p) for a,p in ((1.398,1.46),(1.49,1.44),(1.52,1.73),(1.425,1.76))]
        curved_panel(f"Helmet V2 / visor latch recess {side}",latch,.008,"Recess")
        latch_face=[(side*a,p) for a,p in ((1.421,1.49),(1.478,1.48),(1.493,1.69),(1.442,1.70))]
        curved_panel(f"Helmet V2 / visor latch paddle {side}",latch_face,.010,"Ivory ceramic enamel")
        for a,p in ((side*1.66,1.38),(side*1.64,2.08)):
            author.screw(f"Helmet V2 / captive temple fastener {side} {p}",surface(a,p,.009),
                         normal_at(a,p),GROUP,.0021)
    # Few flush fasteners make the segmented construction legible without busy trim.
    for a in (-2.68,-1.85,-.90,.90,1.85,2.68):
        p=.78
        author.screw(f"Helmet V2 / crown captive fastener {a}",surface(a,p,.0028),normal_at(a,p),GROUP,.0016)
    mat=bpy.data.materials[GLASS_MAT]
    bsdf=mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value=.14
    bsdf.inputs["Coat Weight"].default_value=.52
    bsdf.inputs["Coat Roughness"].default_value=.065
    bsdf.inputs["Metallic"].default_value=.40
    ramp=mat.node_tree.nodes.get("Restrained tonal variation — edit these two colours")
    if ramp:
        color=(.0105,.028,.010)
        for element,factor in zip(ramp.color_ramp.elements,(.93,1.045)):
            element.color=(*(c*factor for c in color),1)
    return glass


def setup_cameras():
    positions={"front":(0,-25,1.76),"side":(25,0,1.76),
               "back":(0,25,1.76),"hero":(16,-25,9.0)}
    for name,position in positions.items():
        obj=bpy.data.objects.get("Head review / "+name)
        if obj is None:
            data=bpy.data.cameras.new("Head review / "+name)
            obj=bpy.data.objects.new(data.name,data);author.collection("91 / Review cameras").objects.link(obj)
        obj.location=position;author.aim(obj,(0,0,1.702))
        obj.data.type="ORTHO";obj.data.ortho_scale=.61;obj.data.clip_end=5000


def open_source(path):
    bpy.ops.wm.open_mainfile(filepath=str(path))
    author.COLLECTIONS={col.name:col for col in bpy.data.collections}
    author.MATERIALS={mat.name:mat for mat in bpy.data.materials}


def build():
    frozen,manifest=preserved_files();open_source(frozen)
    before=fingerprints()
    removed=[obj.name for obj in bpy.context.scene.objects if is_head(obj)]
    for name in removed:bpy.data.objects.remove(bpy.data.objects[name],do_unlink=True)
    glass=build_head()
    parent=bpy.data.objects["SPACE MARINE / move the complete source"]
    for obj in bpy.context.scene.objects:
        if is_head(obj):obj.parent=parent
    setup_cameras()
    scene=bpy.context.scene
    scene.camera=bpy.data.objects["Head review / hero"]
    scene.render.resolution_x=1600;scene.render.resolution_y=1600;scene.cycles.samples=144
    scene["asset_display_name"]="Space Marine (Codex) / head v2"
    scene["variant_scope"]="Helmet only. Approved original and all non-head geometry/materials preserved."
    scene["approved_source_sha256"]=manifest["snapshot_sha256"]
    bpy.ops.object.select_all(action="DESELECT");glass.hide_set(False);glass.select_set(True)
    bpy.context.view_layer.objects.active=glass
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=="VIEW_3D":
                area.spaces.active.region_3d.view_distance=1.15
                area.spaces.active.region_3d.view_location=(0,0,1.68)
                area.spaces.active.region_3d.view_rotation=scene.camera.rotation_euler.to_quaternion()
                area.spaces.active.shading.type="MATERIAL"
    after=fingerprints()
    if before!=after:raise RuntimeError("Non-head geometry/material fingerprint changed")
    SOURCE.parent.mkdir(parents=True,exist_ok=True)
    (SOURCE.parent/"preservation_fingerprints.json").write_text(json.dumps(
        {"approved_source_sha256":manifest["snapshot_sha256"],"removed_head_objects":removed,
         "unchanged_non_head":before},indent=2)+"\n",encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE),compress=True)
    preserved_files()
    print("HEAD_VARIANT_SAVED",str(SOURCE),"UNCHANGED_BODY_OBJECTS",len(before["objects"]),flush=True)


def configure_cycles(draft):
    scene=bpy.context.scene;scene.render.engine="CYCLES"
    scene.cycles.samples=64 if draft else 144;scene.cycles.use_denoising=True
    try:
        preferences=bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type="OPTIX";preferences.get_devices()
        for device in preferences.devices:device.use=device.type=="OPTIX"
        if any(device.type=="OPTIX" for device in preferences.devices):scene.cycles.device="GPU"
    except Exception as exc:print("Configured Cycles device retained:",exc)


def render(which,draft):
    frozen,manifest=preserved_files()
    output=VARIANT/"previews"/("review_1" if draft else "final")
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for label,path in (("approved_v1",frozen),("head_v2",SOURCE)):
        if which not in ("both",label):continue
        digest=sha256(path);open_source(path);setup_cameras();configure_cycles(draft)
        scene=bpy.context.scene
        for name in ("front","side","back","hero","full_hero"):
            if name=="full_hero" and label=="approved_v1":continue
            scene.camera=bpy.data.objects["Camera / hero" if name=="full_hero" else "Head review / "+name]
            scene.render.resolution_x=1100 if draft else 1600
            scene.render.resolution_y=(1375 if draft else 2000) if name=="full_hero" else scene.render.resolution_x
            scene.render.resolution_percentage=100
            scene.render.filepath=str(output/f"{label}_{name}.png")
            bpy.ops.render.render(write_still=True)
            records.append({"source":label,"source_sha256":digest,"view":name,
                            "image":Path(scene.render.filepath).name,"samples":scene.cycles.samples,
                            "camera_location":list(scene.camera.location),"camera_rotation":list(scene.camera.rotation_euler),
                            "ortho_scale":scene.camera.data.ortho_scale,
                            "resolution":[scene.render.resolution_x,scene.render.resolution_y]})
            print("HEAD_REVIEW_RENDER",scene.render.filepath,flush=True)
        if sha256(path)!=digest:raise RuntimeError("Saved source changed during render")
    (output/"render_manifest.json").write_text(json.dumps(records,indent=2)+"\n",encoding="utf-8")
    preserved_files()


def audit():
    frozen,manifest=preserved_files();open_source(SOURCE)
    expected=json.loads((SOURCE.parent/"preservation_fingerprints.json").read_text(encoding="utf-8"))["unchanged_non_head"]
    current=fingerprints()
    assert current==expected,"Reopened non-head geometry or shaders changed"
    glass=bpy.data.objects["VISOR GLASS / convex optical shield"];mat=glass.data.materials[0]
    users=[obj.name for obj in bpy.context.scene.objects if is_character(obj) and mat.name in obj.data.materials]
    assert users==[glass.name] and glass.pass_index==1 and mat.pass_index==1
    assert all(obj==glass or obj.pass_index!=1 for obj in bpy.context.scene.objects)
    missing=[image.name for image in bpy.data.images if image.source=="FILE" and not image.packed_file
             and not Path(bpy.path.abspath(image.filepath)).is_file()]
    cloth=[obj.name for obj in bpy.context.scene.objects if any(mod.type=="CLOTH" for mod in obj.modifiers)]
    assert not missing and not cloth
    depsgraph=bpy.context.evaluated_depsgraph_get();vertices=triangles=degenerate=0
    lower=[math.inf]*3;upper=[-math.inf]*3;objects=0
    for obj in bpy.context.scene.objects:
        if not is_character(obj):continue
        objects+=1;evaluated=obj.evaluated_get(depsgraph);mesh=evaluated.to_mesh();mesh.calc_loop_triangles()
        vertices+=len(mesh.vertices);triangles+=len(mesh.loop_triangles)
        degenerate+=sum(face.area<1e-12 for face in mesh.loop_triangles)
        for vertex in mesh.vertices:
            point=evaluated.matrix_world@vertex.co
            assert all(math.isfinite(v) for v in point)
            for axis in range(3):lower[axis]=min(lower[axis],point[axis]);upper[axis]=max(upper[axis],point[axis])
        evaluated.to_mesh_clear()
    bsdf=mat.node_tree.nodes.get("Principled BSDF")
    report={"source":SOURCE.name,"source_sha256":sha256(SOURCE),"approved_snapshot_sha256":sha256(frozen),
            "preserved_original_files":len(manifest["preserved_files_relative_to_asset"]),
            "unchanged_non_head_objects":len(current["objects"]),"unchanged_non_head_materials":len(current["materials"]),
            "evaluated_vertices":vertices,"evaluated_triangles":triangles,"character_objects":objects,
            "degenerate_triangles_below_1e_12_square_metres":degenerate,
            "bounds_min":lower,"bounds_max":upper,"missing_images":missing,"live_cloth":cloth,
            "visor":{"object":glass.name,"material":mat.name,"object_mask_id":glass.pass_index,
                     "material_mask_id":mat.pass_index,"material_users":users,
                     "roughness":bsdf.inputs["Roughness"].default_value,"coat_weight":bsdf.inputs["Coat Weight"].default_value,
                     "coat_roughness":bsdf.inputs["Coat Roughness"].default_value}}
    (SOURCE.parent/"validation.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("HEAD_REOPEN_AUDIT",json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--build",action="store_true")
    parser.add_argument("--render",choices=("both","approved_v1","head_v2"))
    parser.add_argument("--draft",action="store_true")
    parser.add_argument("--audit",action="store_true")
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if args.build:build()
    if args.render:render(args.render,args.draft)
    if args.audit:audit()
    if not any((args.build,args.render,args.audit)):parser.error("Choose --build, --render and/or --audit")


if __name__=="__main__":main()
