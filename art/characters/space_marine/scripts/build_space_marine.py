"""Author the reference-guided EVA marine. Blender 4.5 LTS, no external assets.

Run from any directory with Blender --background --python THIS_FILE -- [--draft].
Coordinates: metres, +Z up, character looks toward -Y. The garment is deliberately
a dense editable source surface, not a claim of deformation-ready game topology.
All shape fields and materials below are original work for this asset.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
PREVIEWS = ROOT / "previews"
REFERENCE = ROOT.parent / "Space Marine Turnaround Sheet.png"
TAU = math.tau
COLLECTIONS = {}
MATERIALS = {}


def collection(name):
    if name not in COLLECTIONS:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
        COLLECTIONS[name] = col
    return COLLECTIONS[name]


def relocate(obj, group):
    for col in list(obj.users_collection):
        col.objects.unlink(obj)
    collection(group).objects.link(obj)
    return obj


def mesh_object(name, verts, faces, material, group, smooth=True):
    mesh = bpy.data.meshes.new(name + " / surface")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection(group).objects.link(obj)
    if material:
        obj.data.materials.append(MATERIALS[material])
    for poly in mesh.polygons:
        poly.use_smooth = smooth
    return obj


def bevel(obj, amount=0.004, segments=3):
    mod = obj.modifiers.new("Manufactured edge radius", "BEVEL")
    mod.width = amount
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = 0.5
    return obj


def solidify(obj, thickness=0.008):
    mod = obj.modifiers.new("Physical shell thickness", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = -1
    return obj


def curve(name, points, radius, material, group, cyclic=False):
    data = bpy.data.curves.new(name + " / sweep", "CURVE")
    data.dimensions = "3D"
    data.resolution_u = 2
    data.bevel_depth = radius
    data.bevel_resolution = 3
    spl = data.splines.new("POLY")
    spl.points.add(len(points) - 1)
    for p, co in zip(spl.points, points):
        p.co = (*co, 1)
    spl.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, data)
    collection(group).objects.link(obj)
    data.materials.append(MATERIALS[material])
    return obj


def ellipsoid(name, center, scale, material, group, segments=48, rings=24):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(MATERIALS[material])
    for face in obj.data.polygons:
        face.use_smooth = True
    relocate(obj, group)
    return obj


def box(name, center, dimensions, material, group, rounding=0.004):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(MATERIALS[material])
    relocate(obj, group)
    bevel(obj, rounding, 4)
    normal = obj.modifiers.new("Weighted planar normals", "WEIGHTED_NORMAL")
    normal.keep_sharp = True
    return obj


def cylinder(name, center, radius, depth, material, group, axis=(0, 0, 1), vertices=48):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = Vector(axis).to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(MATERIALS[material])
    relocate(obj, group)
    bevel(obj, min(depth * 0.15, 0.0018), 3)
    for p in obj.data.polygons:
        p.use_smooth = len(p.vertices) == 4
    return obj


def screw(name, position, normal, group, radius=0.0026):
    cylinder(name, position, radius, 0.0016, "Fastener", group, normal, 20)
    n = Vector(normal).normalized()
    tangent = n.cross(Vector((0, 0, 1)))
    if tangent.length < 0.01:
        tangent = Vector((1, 0, 0))
    tangent.normalize()
    p = Vector(position) + n * 0.001
    curve(name + " / slot", [p - tangent * radius * 0.55, p + tangent * radius * 0.55],
          0.00032, "Recess", group)


def material(name, color, roughness, metallic=0.0, fabric=False, coat=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.19
    if fabric:
        bsdf.inputs["Sheen Weight"].default_value = 0.22
        bsdf.inputs["Sheen Roughness"].default_value = 0.65
    tex = nodes.new("ShaderNodeTexCoord")
    tex.location = (-750, 80)
    noise = nodes.new("ShaderNodeTexNoise")
    noise.name = "Subtle material variation"
    noise.location = (-540, 120)
    noise.inputs["Scale"].default_value = 27
    noise.inputs["Detail"].default_value = 3
    links.new(tex.outputs["Generated"], noise.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = "Restrained tonal variation — edit these two colours"
    ramp.location = (-300, 160)
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = (*(c * 0.93 for c in color), 1)
    ramp.color_ramp.elements[1].position = 0.84
    ramp.color_ramp.elements[1].color = (*(min(c * 1.045, 1) for c in color), 1)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    fine = nodes.new("ShaderNodeTexNoise")
    fine.name = "Microsurface only — medium folds are geometry"
    fine.location = (-510, -170)
    fine.inputs["Scale"].default_value = 580 if fabric else 290
    fine.inputs["Detail"].default_value = 2
    links.new(tex.outputs["Object"], fine.inputs["Vector"])
    bump = nodes.new("ShaderNodeBump")
    bump.location = (-90, -100)
    bump.inputs["Strength"].default_value = 0.21 if fabric else 0.11
    bump.inputs["Distance"].default_value = 0.00045 if fabric else 0.00016
    links.new(fine.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    MATERIALS[name] = mat
    return mat


def make_materials():
    material("Ivory ceramic enamel", (0.71, 0.666, 0.567), 0.42, 0.075, coat=0.16)
    material("Warm woven pressure fabric", (0.70, 0.659, 0.563), 0.82, fabric=True)
    material("Reinforced textile seams", (0.47, 0.435, 0.365), 0.88, fabric=True)
    material("Ivory binding", (0.64, 0.594, 0.497), 0.75, fabric=True)
    material("Charcoal flex textile", (0.032, 0.036, 0.033), 0.85, fabric=True)
    material("Glove rubber", (0.022, 0.027, 0.025), 0.58, fabric=True)
    material("Graphite elastomer", (0.026, 0.031, 0.029), 0.63)
    material("Recess", (0.011, 0.015, 0.014), 0.64)
    material("Steel blue enamel", (0.155, 0.239, 0.313), 0.46, 0.13, coat=0.1)
    material("Fastener", (0.26, 0.268, 0.251), 0.37, 0.68)
    material("Printed charcoal", (0.10, 0.119, 0.107), 0.64)
    material("Sole rubber", (0.040, 0.045, 0.041), 0.91)
    visor = material("Deep olive optical shield", (0.024, 0.049, 0.025), 0.12, 0.54, coat=0.8)
    bsdf = visor.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Coat Roughness"].default_value = 0.07
    bsdf.inputs["IOR"].default_value = 1.48
    bsdf.inputs["Transmission Weight"].default_value = 0.035
    bsdf.inputs["Specular IOR Level"].default_value = 0.52
    bsdf.inputs["Normal"].default_value = (0, 0, 0)
    for link in list(visor.node_tree.links):
        if link.to_socket == bsdf.inputs["Normal"]:
            visor.node_tree.links.remove(link)
    material("Studio warm grey", (0.24, 0.265, 0.276), 0.88)


def catmull(values, t):
    """Smoothly interpolate regularly spaced profile values."""
    t = max(0, min(1, t)) * (len(values) - 1)
    i = min(int(t), len(values) - 2)
    f = t - i
    a, b, c, d = [values[max(0, min(len(values) - 1, j))] for j in (i-1, i, i+1, i+2)]
    return 0.5 * ((2*b) + (-a+c)*f + (2*a-5*b+4*c-d)*f*f + (-a+3*b-3*c+d)*f*f*f)


def wrapped_angle(a):
    return (a + math.pi) % TAU - math.pi


def fold_field(t, theta, folds, seed):
    displacement = 0.0
    for pos, amplitude, width, slope, angle, spread in folds:
        centre = pos + slope * math.sin(theta - angle + 0.3)
        dt = (t - centre) / width
        local = math.exp(-0.5 * (wrapped_angle(theta - angle) / spread) ** 2)
        ridge = math.exp(-dt*dt) - 0.58 * math.exp(-((dt-1.55)/0.8)**2)
        displacement += amplitude * ridge * local
    displacement += 0.00125 * math.sin(theta*3 + t*9 + seed) * math.sin(math.pi*t)**2
    displacement += 0.00050 * math.sin(theta*7 - t*26 + seed) * math.sin(math.pi*t)
    # No periodic axial corrugation: every medium crease above has a location,
    # diagonal direction, angular envelope and compression-dependent amplitude.
    return displacement * min(1, t*20, (1-t)*20)


def garment(name, start, end, radii_u, radii_v, folds, group="01 / Pressure garment",
            material_name="Warm woven pressure fabric", seed=1, count=104, sides=80,
            cross_power=1.0, seam_angles=(), cap=True):
    a, b = Vector(start), Vector(end)
    axis = (b-a).normalized()
    v = Vector((0, 1, 0))
    u = v.cross(axis).normalized()
    v = axis.cross(u).normalized()

    def point(t, theta, offset=0):
        r1, r2 = catmull(radii_u, t), catmull(radii_v, t)
        c, s = math.cos(theta), math.sin(theta)
        c = math.copysign(abs(c)**cross_power, c)
        s = math.copysign(abs(s)**cross_power, s)
        fold = fold_field(t, theta, folds, seed)
        return a.lerp(b, t) + u*(c*(r1+fold+offset)) + v*(s*(r2+fold+offset))

    verts = [point(j/count, k/sides*TAU) for j in range(count+1) for k in range(sides)]
    faces = []
    for j in range(count):
        for k in range(sides):
            q=j*sides+k; r=j*sides+(k+1)%sides
            faces.append((q, r, r+sides, q+sides))
    if cap:
        faces.append(tuple(reversed(range(sides))))
        faces.append(tuple(count*sides+k for k in range(sides)))
    obj = mesh_object(name, verts, faces, material_name, group)
    for i, angle in enumerate(seam_angles):
        points = [point(0.025 + j/100*0.95, angle, 0.0010) for j in range(101)]
        curve(name + f" / tailored seam {i+1}", points, 0.00115,
              "Reinforced textile seams", group)
        points = [point(0.025 + j/100*0.95, angle+0.027, 0.0011) for j in range(101)]
        curve(name + f" / seam binding {i+1}", points, 0.00065, "Ivory binding", group)
    return obj, point


def folds(seed, amount=10, amplitude=0.009, edge_bias=False):
    rng = random.Random(seed)
    result = []
    for i in range(amount):
        t = rng.uniform(0.10, 0.90)
        if edge_bias:
            t = rng.choice((rng.uniform(.08,.28), rng.uniform(.66,.93)))
        result.append((t, rng.uniform(.48,1)*amplitude,
                       rng.uniform(.023,.055), rng.uniform(-.09,.09),
                       rng.uniform(-math.pi,math.pi), rng.uniform(.45,1.12)))
    return result


def chamfer_outline(width, height, chamfer=0.018):
    w,h=width/2,height/2
    c=min(chamfer,w*.45,h*.45)
    return [(-w+c,-h),(w-c,-h),(w,-h+c),(w,h-c),
            (w-c,h),(-w+c,h),(-w,h-c),(-w,-h+c)]


def panel(name, center, outline, material_name="Ivory ceramic enamel", group="02 / Armour",
          normal=(0,-1,0), u=(1,0,0), v=(0,0,1), thickness=.012, bulge=.006, edge=.003):
    c,n,bu,bv=map(Vector,(center,normal,u,v))
    verts=[]; faces=[]; N=len(outline)
    levels=(.0,.22,.45,.68,.84,1.0)
    for s in levels:
        for x,y in outline:
            verts.append(c+bu*(x*s)+bv*(y*s)+n*(thickness*.5+bulge*(1-s*s)))
    for j in range(len(levels)-1):
        for k in range(N):
            nxt=(k+1)%N
            faces.append((j*N+k,j*N+nxt,(j+1)*N+nxt,(j+1)*N+k))
    back=len(verts)
    verts.extend(c+bu*x+bv*y-n*thickness*.5 for x,y in outline)
    for k in range(N):
        nxt=(k+1)%N
        faces.append(((len(levels)-1)*N+k,(len(levels)-1)*N+nxt,back+nxt,back+k))
    faces.append(tuple(reversed(range(back,back+N))))
    obj=mesh_object(name,verts,faces,material_name,group)
    bevel(obj,edge,3)
    return obj


def ribbon(name, points, width, material_name, group, thickness=.004):
    verts=[]; faces=[]
    for p in points:
        verts.extend(((p[0]-width/2,p[1],p[2]),(p[0]+width/2,p[1],p[2])))
    for i in range(len(points)-1):
        faces.append((2*i,2*i+1,2*i+3,2*i+2))
    obj=mesh_object(name,verts,faces,material_name,group)
    solidify(obj,thickness)
    bevel(obj,.0015,3)
    return obj


def ring(name, center, profiles, material_name, group, front_dip=0, sides=128):
    """Elliptical swept cross-section; profiles are (x radius, y radius, z)."""
    verts=[]; faces=[]
    for rx,ry,z in profiles:
        for k in range(sides):
            a=k/sides*TAU
            dip=front_dip*max(0,-math.sin(a))**3
            verts.append((center[0]+rx*math.cos(a),center[1]+ry*math.sin(a),center[2]+z-dip))
    for j in range(len(profiles)):
        jn=(j+1)%len(profiles)
        for k in range(sides):
            kn=(k+1)%sides
            faces.append((j*sides+k,j*sides+kn,jn*sides+kn,jn*sides+k))
    return mesh_object(name,verts,faces,material_name,group)


def build_garment():
    torso_folds=[(.13,.007,.038,.06,-1.4,1.0),(.27,.009,.042,-.08,-1.4,1.0),
                 (.39,.008,.05,.05,-1.7,.8),(.50,.006,.045,-.10,1.4,.9),
                 (.71,.005,.035,.09,0,.8),(.62,.007,.04,-.06,math.pi,.8)]
    garment("Torso / tailored pressure jacket",(0,.016,1.095),(0,.007,1.604),
            [.177,.190,.217,.224,.187],[.116,.135,.145,.141,.103],torso_folds,
            cross_power=.88,seam_angles=(-1.12,-2.02,.96,2.18),seed=40,count=130)
    garment("Hip / continuous seat and abdominal section",(0,.018,.885),(0,.018,1.158),
            [.082,.163,.210,.202,.182],[.085,.121,.134,.131,.119],
            folds(47,15,.007),cross_power=.91,seam_angles=(-1.48,-1.66,1.15,1.99),seed=18)
    # Dark inner gusset is inset, almost completely covered by the leg roots.
    ellipsoid("Crotch / stretch gusset",(0,.008,.914),(.055,.082,.072),
              "Charcoal flex textile","01 / Pressure garment")
    for side, suffix in ((-1,"L"),(1,"R")):
        garment(f"Upper arm {suffix} / flex under pauldron",(side*.328,.0,1.337),(side*.260,.002,1.532),
                [.084,.098,.097,.087],[.081,.098,.108,.087],folds(10+side,8,.007),
                material_name="Charcoal flex textile",seed=side,count=64)
        garment(f"Sleeve {suffix} / upper woven section",(side*.382,-.012,1.224),(side*.303,0,1.444),
                [.071,.078,.093,.082],[.071,.088,.096,.080],folds(23+side,15,.0095,True),
                seed=23+side,seam_angles=(-1.38,1.57),count=90)
        garment(f"Elbow {suffix} / compressed flex",(side*.411,-.011,1.132),(side*.380,-.011,1.249),
                [.065,.075,.076,.070],[.066,.075,.076,.070],folds(33+side,11,.008),
                material_name="Charcoal flex textile",seed=33+side,count=76)
        garment(f"Sleeve {suffix} / forearm",(side*.473,-.013,.989),(side*.410,-.008,1.151),
                [.055,.067,.076,.067],[.052,.068,.073,.065],folds(50+side,14,.0085,True),
                seed=53+side,seam_angles=(-1.4,1.55),count=90)
        garment(f"Wrist {suffix} / seal",(side*.480,-.013,.955),(side*.470,-.013,1.011),
                [.051,.057,.056,.054],[.048,.053,.052,.050],[],material_name="Graphite elastomer",count=12)
        garment(f"Thigh {suffix} / woven pressure trouser",(side*.158,.008,.653),(side*.108,.019,1.051),
                [.084,.108,.119,.120,.108],[.079,.103,.115,.114,.104],folds(70+side,21,.010,True),
                seed=73+side,seam_angles=(-1.19,-2.08,1.46),count=126)
        garment(f"Knee {suffix} / articulated textile",(side*.177,.004,.493),(side*.157,.012,.679),
                [.077,.090,.097,.084],[.073,.085,.089,.078],folds(90+side,13,.009),
                material_name="Charcoal flex textile",seed=93+side,count=88)
        garment(f"Calf {suffix} / woven pressure trouser",(side*.194,.027,.225),(side*.177,.013,.527),
                [.061,.075,.082,.090,.077],[.063,.078,.082,.084,.072],folds(120+side,19,.0095,True),
                seed=121+side,seam_angles=(-.93,-2.24,1.52),count=118)


def helmet_point(azimuth, phi, offset=0):
    return Vector(((.158+offset)*math.sin(phi)*math.sin(azimuth),
                   .006-(.167+offset)*math.sin(phi)*math.cos(azimuth),
                   1.742+(.151+offset)*math.cos(phi)))


def build_helmet():
    group="03 / Helmet and neck seal"
    angular=192; vertical=42; aperture=1.18
    for top in (True,False):
        verts=[]; faces=[]
        for i in range(angular):
            a=-math.pi+i/angular*TAU
            fraction=max(0,1-(abs(a)/aperture)**4)**.25 if abs(a)<aperture else 0
            boundary=1.59+(-.555 if top else .598)*fraction
            for j in range(vertical+1):
                t=j/vertical
                phi=boundary*t if top else boundary+(math.pi-boundary)*t
                verts.append(helmet_point(a,phi))
        for i in range(angular):
            nxt=(i+1)%angular
            for j in range(vertical):
                faces.append((i*(vertical+1)+j,nxt*(vertical+1)+j,
                              nxt*(vertical+1)+j+1,i*(vertical+1)+j+1))
        obj=mesh_object("Helmet / "+("crown shell" if top else "mandible shell"),verts,faces,
                        "Ivory ceramic enamel",group)
        solidify(obj,.007)
    def shieldpoint(u,v,offset=.0028):
        return helmet_point(u*aperture,1.59+v*(.555 if v<0 else .598),offset)
    # Concentric superellipse contours produce the broad shield and rounded corners.
    verts=[]; faces=[]; steps=30; sides=160
    for j in range(steps+1):
        r=j/steps
        for k in range(sides):
            a=k/sides*TAU
            u=math.copysign(abs(math.cos(a))**.5,math.cos(a))*r
            v=math.copysign(abs(math.sin(a))**.5,math.sin(a))*r
            verts.append(shieldpoint(u,v))
    for j in range(steps):
        for k in range(sides):
            n=(k+1)%sides
            faces.append((j*sides+k,j*sides+n,(j+1)*sides+n,(j+1)*sides+k))
    shield=mesh_object("Visor / convex olive laminated shield",verts,faces,"Deep olive optical shield",group)
    solidify(shield,.004)
    for rad, offset, mat, label in ((.007,.003,"Recess","pressure gasket"),
                                     (.0028,.0085,"Ivory ceramic enamel","retaining lip")):
        points=[]
        for k in range(sides):
            a=k/sides*TAU
            u=math.copysign(abs(math.cos(a))**.5,math.cos(a))
            v=math.copysign(abs(math.sin(a))**.5,math.sin(a))
            points.append(shieldpoint(u,v,offset))
        curve("Visor / "+label,points,rad,mat,group,True)
    # Shell panel joints wrap over crown and back, with flush fasteners.
    for angle in (-2.52,-1.52,1.52,2.52):
        points=[helmet_point(angle,.13+j/100*2.66,.0004) for j in range(101)]
        curve(f"Helmet / meridian seam {angle}",points,.00085,"Reinforced textile seams",group)
    for phi in (.75,1.61,2.32):
        points=[helmet_point(1.22+j/140*(TAU-2.44),phi,.0008) for j in range(141)]
        curve(f"Helmet / rear panel seam {phi}",points,.0010,"Reinforced textile seams",group)
    for side in (-1,1):
        for phi in (.83,2.16):
            p=helmet_point(side*1.46,phi,.003)
            screw(f"Helmet / shell screw {side} {phi}",p,(side,0,.10),group,.0020)
        cylinder(f"Helmet / side hinge {side}",(side*.154,-.001,1.726),.024,.012,
                 "Graphite elastomer",group,(side,0,0))
        cylinder(f"Helmet / hinge cover {side}",(side*.161,-.001,1.726),.019,.010,
                 "Ivory ceramic enamel",group,(side,0,0))
        panel(f"Helmet / lower cheek service panel {side}",(side*.128,.056,1.656),
              chamfer_outline(.087,.057,.014),group=group,normal=(side*.92,.2,-.14),
              u=(0,1,0),v=(0,0,1),thickness=.009,bulge=.002)
    # Collar is a tall, layered physical pressure seal, sunk into the jacket.
    ring("Neck / flexible seated gasket",(0,.009,0),
         [(.174,.137,1.571),(.181,.145,1.584),(.180,.143,1.625),
          (.150,.119,1.628),(.148,.117,1.577)],"Graphite elastomer",group,.019)
    ring("Collar / armoured lower flange",(0,.009,0),
         [(.170,.131,1.544),(.197,.157,1.559),(.199,.161,1.579),
          (.191,.155,1.600),(.164,.129,1.598),(.158,.125,1.557)],
         "Ivory ceramic enamel",group,.025)
    ring("Collar / upper pressure lock",(0,.009,0),
         [(.177,.139,1.592),(.188,.152,1.601),(.185,.149,1.636),
          (.171,.138,1.647),(.155,.123,1.642),(.157,.124,1.599)],
         "Ivory ceramic enamel",group,.026)
    for angle in (-2.6,-2.0,-1.15,-.55,.25,.95,1.8,2.6):
        x=.19*math.cos(angle);y=.009+.155*math.sin(angle)
        z=1.607-.026*max(0,-math.sin(angle))**3
        n=Vector((math.cos(angle),math.sin(angle),0))
        obj=panel(f"Collar / lock segment {angle}",(x,y,z),chamfer_outline(.024,.028,.004),
                  "Graphite elastomer",group,normal=n,u=(-n.y,n.x,0),v=(0,0,1),
                  thickness=.005,bulge=0,edge=.002)


def shoulder_surface(side, theta, phi, offset=0):
    return Vector((side*(.236+(.128+offset)*math.sin(phi)*math.cos(theta)),
                   (.137+offset)*math.sin(phi)*math.sin(theta),
                   1.439+(.181+offset)*math.cos(phi)))


def shoulder_patch(name, side, theta_extent, phi_min, phi_max, offset, material_name):
    rows=42;cols=60;verts=[];faces=[]
    for j in range(rows+1):
        phi=phi_min+(phi_max-phi_min)*j/rows
        for k in range(cols+1):
            th=-theta_extent+2*theta_extent*k/cols
            verts.append(shoulder_surface(side,th,phi,offset))
    for j in range(rows):
        for k in range(cols):
            q=j*(cols+1)+k
            face=(q,q+1,q+cols+2,q+cols+1)
            faces.append(face if side>0 else tuple(reversed(face)))
    obj=mesh_object(name,verts,faces,material_name,"02 / Armour")
    solidify(obj,.006 if offset else .010)
    bevel(obj,.0028,3)
    return obj


def build_armour():
    group="02 / Armour"
    for side, label in ((-1,"L"),(1,"R")):
        # Broad sewn vest bindings connect chest, upper shoulders and pack harness.
        points=[]
        for j in range(61):
            t=j/60
            a=-math.pi*.48+t*math.pi*.96
            points.append((side*(.170+.014*math.cos(a)),.146*math.sin(a),1.403+.181*math.cos(a)))
        ribbon(f"Harness {label} / over-shoulder webbing",points,.038,"Ivory binding",group,.010)
        for edge in (-1,1):
            curve(f"Harness {label} / edge seam {edge}",[(x+edge*.016,y-.0002,z+.001) for x,y,z in points],
                  .0013,"Reinforced textile seams",group)
        shoulder_patch(f"Pauldron {label} / compound shell",side,1.48,.20,1.60,0,"Ivory ceramic enamel")
        shoulder_patch(f"Pauldron {label} / steel-blue inset",side,1.25,.57,1.365,.003,"Steel blue enamel")
        for phi in (.60,1.33):
            for theta in (-1.17,0,1.17):
                p=shoulder_surface(side,theta,phi,.008)
                normal=(side*math.sin(phi)*math.cos(theta),math.sin(phi)*math.sin(theta),math.cos(phi))
                screw(f"Pauldron {label} / insert fastener {phi} {theta}",p,normal,group,.0020)
        elbow_center=(side*.436,-.052,1.198)
        n=Vector((side*.61,-.79,.04)).normalized()
        u=Vector((.79,side*.61,0))
        panel(f"Elbow {label} / guard underlay",elbow_center,chamfer_outline(.085,.112,.024),
              "Graphite elastomer",group,normal=n,u=u,thickness=.012,bulge=.010,edge=.005)
        panel(f"Elbow {label} / ivory guard",Vector(elbow_center)+n*.009,chamfer_outline(.073,.099,.021),
              group=group,normal=n,u=u,thickness=.012,bulge=.013,edge=.004)
        # Knee is a curved two-stage protector bonded to the flex section.
        x=side*.173
        outline=[(-.040,-.076),(.039,-.076),(.063,-.041),(.061,.048),(.035,.082),(-.036,.082),(-.061,.046),(-.064,-.030)]
        panel(f"Knee {label} / rubber cradle",(x,-.068,.593),outline,"Graphite elastomer",group,
              thickness=.022,bulge=.014,edge=.006)
        panel(f"Knee {label} / armour plate",(x,-.087,.596),[(u*.84,v*.86) for u,v in outline],
              group=group,thickness=.014,bulge=.017,edge=.005)
        for dx,dz in ((-.034,.046),(.034,.046),(-.028,-.043),(.028,-.043)):
            screw(f"Knee {label} / fastener {dx} {dz}",(x+dx,-.111,.596+dz),(0,-1,0),group,.0020)
        panel(f"Calf {label} / recessed service tab",(side*.249,-.007,.399),chamfer_outline(.024,.061,.008),
              "Graphite elastomer",group,normal=(side*.96,-.27,0),u=(.27,side*.96,0),
              thickness=.007,bulge=.002,edge=.003)
    chest_outline=[(-.119,-.087),(.119,-.087),(.147,-.062),(.155,.055),(.13,.080),(-.13,.080),(-.155,.055),(-.147,-.062)]
    panel("Chest / dark compliant backing",(0,-.140,1.399),[(x*1.035,z*1.035) for x,z in chest_outline],
          "Graphite elastomer",group,thickness=.019,bulge=.006,edge=.008)
    panel("Chest / removable life-support interface",(0,-.154,1.399),chest_outline,
          group=group,thickness=.022,bulge=.010,edge=.006)
    for x in (-.106,.106):
        panel(f"Chest / upper socket bezel {x}",(x,-.144,1.526),chamfer_outline(.039,.047,.004),
              group=group,thickness=.009,bulge=.001,edge=.002)
        panel(f"Chest / upper black socket {x}",(x,-.152,1.526),chamfer_outline(.025,.033,.003),
              "Graphite elastomer",group,thickness=.006,bulge=0,edge=.0015)
        for dz in (-.006,0,.006):
            box(f"Chest / socket grille {x} {dz}",(x,-.156,1.526+dz),(.017,.0015,.0018),"Fastener",group,.0005)
    for x in (-.155,.155):
        box(f"Chest / side retention clip {x}",(x,-.149,1.409),(.011,.020,.050),"Graphite elastomer",group,.003)
    panel("Chest / lower inset vent",(0,-.174,1.341),chamfer_outline(.084,.020,.005),
          "Recess",group,thickness=.004,bulge=0,edge=.002)
    for dz in (-.003,.0025):
        box(f"Chest / lower vent blade {dz}",(0,-.177,1.341+dz),(.067,.002,.0015),"Fastener",group,.0006)
    for x,z in ((-.119,1.449),(.119,1.449),(-.110,1.348),(.110,1.348)):
        screw(f"Chest / plate screw {x} {z}",(x,-.170,z),(0,-1,0),group,.0020)


def build_belt_pockets():
    group="04 / Belt and cargo"
    ring("Belt / dark webbing foundation",(0,.013,0),
         [(.191,.129,1.101),(.196,.134,1.105),(.196,.134,1.172),
          (.192,.130,1.176),(.180,.117,1.167),(.180,.117,1.108)],"Charcoal flex textile",group)
    # Solid front belt with small articulated side/rear sections.
    for a in (-2.60,-1.98,-1.57,-1.15,-.54,.10,.72,1.35,1.98,2.60):
        n=Vector((math.cos(a),math.sin(a),0))
        center=(.197*math.cos(a),.013+.137*math.sin(a),1.139)
        panel(f"Belt / articulated segment {a}",center,chamfer_outline(.077,.064,.006),
              group=group,normal=n,u=(-n.y,n.x,0),v=(0,0,1),thickness=.009,bulge=.001,edge=.003)
    for side,label in ((-1,"L"),(1,"R")):
        x=side*.203
        normal=Vector((side*.36,-.93,0));u=Vector((.93,side*.36,0))
        panel(f"Waist pouch {label} / textile gusset",(x,-.093,1.139),chamfer_outline(.110,.135,.020),
              "Ivory binding",group,normal=normal,u=u,thickness=.062,bulge=.013,edge=.010)
        panel(f"Waist pouch {label} / lid",(x,-.129,1.144),chamfer_outline(.105,.127,.018),
              group=group,normal=normal,u=u,thickness=.011,bulge=.010,edge=.005)
        for dx in (-.031,.031):
            p=Vector((x,-.141,1.144))+u*dx
            curve(f"Waist pouch {label} / vertical mould seam {dx}",[p+Vector((0,0,-.043)),p+Vector((0,0,.043))],
                  .0009,"Reinforced textile seams",group)
        panel(f"Waist pouch {label} / dark side clasp",(side*.158,-.147,1.169),chamfer_outline(.014,.038,.002),
              "Graphite elastomer",group,thickness=.007,bulge=0,edge=.002)
        # The broad thigh pocket follows the outside/front quadrant, rather than a box on the centre thigh.
        n=Vector((side*.67,-.743,0));u=Vector((.743,side*.67,0))
        center=Vector((side*.226,-.051,.855))
        pocket_outline=[(-.063,-.094),(.059,-.094),(.073,-.070),(.069,.086),(.047,.105),(-.055,.103),(-.071,.081),(-.071,-.069)]
        panel(f"Thigh cargo {label} / padded pocket",center,pocket_outline,"Warm woven pressure fabric",group,
              normal=n,u=u,thickness=.031,bulge=.014,edge=.009)
        border=[center+u*x+Vector((0,0,z))+n*.021 for x,z in pocket_outline]
        curve(f"Thigh cargo {label} / stitched perimeter",border,.00125,"Reinforced textile seams",group,True)
        flap_center=center+Vector((0,0,.077))+n*.027
        panel(f"Thigh cargo {label} / overlapping flap",flap_center,chamfer_outline(.152,.053,.008),
              "Ivory binding",group,normal=n,u=u,thickness=.009,bulge=.002,edge=.003)
        for dx in (-.065,.065):
            p=flap_center+u*dx+n*.004
            panel(f"Thigh cargo {label} / flap latch {dx}",p,chamfer_outline(.011,.044,.003),
                  "Graphite elastomer",group,normal=n,u=u,thickness=.005,bulge=0,edge=.0015)
        for i in range(3):
            z=.797+i*.031
            points=[]
            for j in range(35):
                t=j/34
                points.append(center+u*(-.051+.104*t)+Vector((0,0,z-.855+.010*math.sin(t*4+i)))+n*(.029+.004*math.sin(t*math.pi)))
            curve(f"Thigh cargo {label} / soft gusset fold {i}",points,.0014,"Ivory binding",group)


def finger(name, points, radius, group):
    count=36;sides=20;verts=[];faces=[]
    xs=[p[0] for p in points];ys=[p[1] for p in points];zs=[p[2] for p in points]
    for j in range(count+1):
        t=j/count
        c=Vector((catmull(xs,t),catmull(ys,t),catmull(zs,t)))
        ta=max(0,t-.002);tb=min(1,t+.002)
        tangent=Vector((catmull(xs,tb)-catmull(xs,ta),catmull(ys,tb)-catmull(ys,ta),catmull(zs,tb)-catmull(zs,ta))).normalized()
        u=tangent.cross(Vector((0,1,0))).normalized();v=tangent.cross(u).normalized()
        r=radius*(.92+.10*math.sin(t*math.pi*3))
        r*=min(1,math.sqrt(max(.006,(1-t)*8)))
        for k in range(sides):
            a=k/sides*TAU
            verts.append(c+u*(r*math.cos(a))+v*(r*.86*math.sin(a)))
    for j in range(count):
        for k in range(sides):
            n=(k+1)%sides
            faces.append((j*sides+k,j*sides+n,(j+1)*sides+n,(j+1)*sides+k))
    faces.append(tuple(reversed(range(sides))))
    faces.append(tuple(count*sides+k for k in range(sides)))
    return mesh_object(name,verts,faces,"Glove rubber",group)


def build_hands():
    group="05 / Gloves"
    for side,label in ((-1,"L"),(1,"R")):
        palm=ellipsoid(f"Glove {label} / anatomical palm",(side*.489,-.012,.927),(.042,.026,.056),"Glove rubber",group)
        palm.rotation_euler[1]=side*-.16
        panel(f"Glove {label} / dorsal armour",(side*.488,-.039,.938),
              [(-.025,-.039),(.021,-.039),(.031,-.022),(.024,.034),(-.023,.037),(-.030,.018)],
              group=group,thickness=.007,bulge=.004,edge=.004)
        for i,(dx,length) in enumerate(((-.025,.074),(-.008,.083),(.010,.078),(.028,.061))):
            x=side*(.493+dx);z=.902-abs(dx)*.16
            pts=[(x,-.013,z),(x+side*.005,-.009,z-length*.37),
                 (x+side*.010,.004,z-length*.76),(x+side*.002,.018,z-length)]
            finger(f"Glove {label} / finger {i+1}",pts,.0117 if i<3 else .0105,group)
            ellipsoid(f"Glove {label} / knuckle reinforcement {i+1}",(x,-.034,z+.009),(.0102,.0048,.0125),
                      "Graphite elastomer",group,28,16)
            for t in (.39,.70):
                xx=catmull([p[0] for p in pts],t);yy=catmull([p[1] for p in pts],t);zz=catmull([p[2] for p in pts],t)
                points=[(xx+.010*math.cos(a),yy-.0088*math.sin(a),zz+.0015*math.cos(a)) for a in [j/20*math.pi for j in range(21)]]
                curve(f"Glove {label} / finger {i+1} crease {t}",points,.00065,"Recess",group)
        thumb=[(side*.461,-.013,.942),(side*.448,-.020,.922),(side*.440,-.014,.898),(side*.437,-.007,.886)]
        finger(f"Glove {label} / opposed thumb",thumb,.0144,group)
        garment(f"Glove {label} / cuff binding",(side*.478,-.013,.963),(side*.474,-.013,.987),
                [.053,.054,.054],[.050,.051,.051],[],group=group,material_name="Ivory binding",count=10)
        panel(f"Glove {label} / cuff closure",(side*.478,-.064,.977),chamfer_outline(.038,.011,.002),
              "Graphite elastomer",group,thickness=.006,bulge=0,edge=.0015)


def build_boots():
    group="06 / Boots"
    for side,label in ((-1,"L"),(1,"R")):
        x=side*.194
        # The boot is a foot-shaped horizontal loft: broad toe, narrow heel and a rising instep.
        def boot_loft(name, profiles, mat):
            sides=96;verts=[];faces=[]
            for z,rx,ry,cy in profiles:
                for k in range(sides):
                    a=k/sides*TAU;c=math.cos(a);s=math.sin(a)
                    xx=math.copysign(abs(c)**.65,c)*rx*(1-.18*(s+1)/2)
                    yy=math.copysign(abs(s)**.65,s)*ry
                    verts.append((x+xx,cy+yy,z))
            for j in range(len(profiles)-1):
                for k in range(sides):
                    n=(k+1)%sides
                    faces.append((j*sides+k,j*sides+n,(j+1)*sides+n,(j+1)*sides+k))
            faces.append(tuple(reversed(range(sides))))
            faces.append(tuple((len(profiles)-1)*sides+k for k in range(sides)))
            obj=mesh_object(name,verts,faces,mat,group)
            bevel(obj,.002,3)
            return obj
        boot_loft(f"Boot {label} / layered rubber outsole",[(.013,.079,.146,-.039),(.023,.087,.154,-.039),
                    (.051,.087,.153,-.039),(.060,.081,.149,-.039)],"Sole rubber")
        boot_loft(f"Boot {label} / welt",[(.054,.081,.150,-.039),(.064,.084,.149,-.037),
                    (.076,.082,.146,-.036)],"Graphite elastomer")
        boot_loft(f"Boot {label} / pressure upper",[(.070,.078,.143,-.035),(.096,.083,.141,-.035),
                    (.119,.082,.136,-.031),(.144,.077,.120,-.016),(.173,.071,.097,.008),
                    (.209,.067,.078,.030),(.257,.065,.073,.032)],"Ivory ceramic enamel")
        for a in range(7):
            yy=-.164+a*.043
            for sign in (-1,1):
                box(f"Boot {label} / lateral tread {a} {sign}",(x+sign*.078,yy,.034),(.023,.023,.034),
                    "Sole rubber",group,.004)
        for sign in (-1,1):
            box(f"Boot {label} / toe tread {sign}",(x+sign*.031,-.185,.033),(.045,.026,.035),"Sole rubber",group,.004)
            box(f"Boot {label} / heel lug {sign}",(x+sign*.038,.100,.030),(.040,.035,.033),"Sole rubber",group,.004)
        # Rubber heel cup and toe bumper are attached to the outsole.
        panel(f"Boot {label} / toe bumper",(x,-.178,.074),chamfer_outline(.082,.040,.012),
              "Graphite elastomer",group,thickness=.012,bulge=.003,edge=.004)
        for dz in (-.006,.002,.010):
            curve(f"Boot {label} / toe traction seam {dz}",[(x-.027,-.188,.074+dz),(x+.027,-.188,.074+dz)],
                  .0013,"Sole rubber",group)
        panel(f"Boot {label} / heel cup",(x,.103,.091),chamfer_outline(.105,.059,.013),
              "Graphite elastomer",group,normal=(0,1,0),u=(-1,0,0),thickness=.012,bulge=.002,edge=.004)
        # Curved toe-cap and instep stitch lines follow the shell.
        for i,(yy,zz) in enumerate(((-.133,.134),(-.088,.158),(-.040,.192))):
            points=[]
            for k in range(51):
                u=-1+2*k/50
                points.append((x+.068*u,yy+.023*u*u,zz-.027*u*u))
            curve(f"Boot {label} / upper panel joint {i}",points,.0012,"Reinforced textile seams",group)
            curve(f"Boot {label} / upper panel binding {i}",[(a,b+.002,c+.0003) for a,b,c in points],.0007,"Ivory binding",group)
        # Wide angled instep strap conforms to the top of the foot.
        panel(f"Boot {label} / instep strap",(x,-.048,.173),chamfer_outline(.145,.046,.008),
              "Ivory binding",group,normal=(0,-.69,.72),u=(1,0,0),v=(0,.72,.69),
              thickness=.008,bulge=.010,edge=.003)
        panel(f"Boot {label} / instep buckle",(x+side*.065,-.052,.166),chamfer_outline(.023,.045,.004),
              "Graphite elastomer",group,normal=(side*.40,-.58,.71),u=(.9,side*.4,0),v=(0,.72,.69),
              thickness=.010,bulge=.001,edge=.003)
        ring(f"Boot {label} / ankle strap",(x,.029,0),[(.067,.075,.217),(.071,.079,.220),
                    (.071,.079,.254),(.067,.075,.259),(.062,.070,.251),(.062,.070,.220)],
             "Graphite elastomer",group)
        panel(f"Boot {label} / ankle front plate",(x,-.046,.239),chamfer_outline(.087,.050,.008),
              group=group,thickness=.011,bulge=.005,edge=.003)
        for sign in (-1,1):
            panel(f"Boot {label} / ankle latch {sign}",(x+sign*.066,-.008,.239),chamfer_outline(.023,.039,.004),
                  "Graphite elastomer",group,normal=(sign,0,0),u=(0,1,0),thickness=.010,bulge=.001,edge=.002)


def build_pack():
    group="07 / Life support pack"
    outline=[(-.115,-.185),(.115,-.185),(.145,-.157),(.145,.139),(.097,.194),(-.097,.194),(-.145,.139),(-.145,-.157)]
    panel("Pack / compliant mount",(0,.145,1.401),[(x*.97,z*.97) for x,z in outline],
          "Graphite elastomer",group,normal=(0,1,0),u=(-1,0,0),thickness=.032,bulge=0,edge=.009)
    panel("Pack / pressure housing",(0,.212,1.401),outline,
          group=group,normal=(0,1,0),u=(-1,0,0),thickness=.125,bulge=.002,edge=.011)
    panel("Pack / rear removable cover",(0,.279,1.401),[(x*.90,z*.93) for x,z in outline],
          group=group,normal=(0,1,0),u=(-1,0,0),thickness=.014,bulge=.0015,edge=.004)
    border=[(-x*.88,.289,1.401+z*.91) for x,z in outline]
    curve("Pack / cover gasket",border,.0012,"Reinforced textile seams",group,True)
    panel("Pack / blue upper access plate",(0,.291,1.512),chamfer_outline(.110,.087,.008),
          "Steel blue enamel",group,normal=(0,1,0),u=(-1,0,0),thickness=.007,bulge=.001,edge=.003)
    for x in (-.043,.043):
        for z in (1.480,1.544):
            screw(f"Pack / blue-panel screw {x} {z}",(x,.297,z),(0,1,0),group,.0020)
    curve("Pack / horizontal service seam",[(-.126,.292,1.434),(.126,.292,1.434)],.0011,"Reinforced textile seams",group)
    for side in (-1,1):
        x=side*.073
        panel(f"Pack / lower filter bezel {side}",(x,.294,1.273),chamfer_outline(.058,.066,.009),
              "Graphite elastomer",group,normal=(0,1,0),u=(-1,0,0),thickness=.011,bulge=0,edge=.003)
        panel(f"Pack / recessed filter {side}",(x,.301,1.273),chamfer_outline(.042,.049,.003),
              "Recess",group,normal=(0,1,0),u=(-1,0,0),thickness=.003,bulge=0,edge=.001)
        for j in range(7):
            box(f"Pack / lower filter blade {side} {j}",(x,.304,1.254+j*.006),(.037,.003,.0025),"Fastener",group,.0007)
        for z in (1.274,1.385,1.494):
            box(f"Pack / side latch {side} {z}",(side*.145,.238,z),(.014,.036,.048),"Graphite elastomer",group,.004)
            box(f"Pack / latch tongue {side} {z}",(side*.153,.240,z),(.006,.021,.029),"Fastener",group,.002)
        # Side radiator modules give the profile real depth and intentional construction.
        for z in (1.365,1.520):
            panel(f"Pack / side radiator bezel {side} {z}",(side*.148,.203,z),chamfer_outline(.062,.072,.009),
                  "Graphite elastomer",group,normal=(side,0,0),u=(0,side,0),v=(0,0,1),
                  thickness=.006,bulge=0,edge=.003)
            for j in range(7):
                box(f"Pack / side radiator blade {side} {z} {j}",(side*.153,.203,z-.023+j*.0075),
                    (.003,.045,.0027),"Fastener",group,.0007)
        for z in (1.238,1.562):
            screw(f"Pack / corner screw {side} {z}",(side*.106,.292,z),(0,1,0),group,.0022)
    box("Pack / upper carrying grip",(0,.217,1.604),(.085,.028,.012),"Graphite elastomer",group,.004)
    box("Pack / bottom connector recess",(0,.225,1.205),(.101,.055,.014),"Graphite elastomer",group,.004)


def label(name, text, center, size, normal, group, material_name="Printed charcoal"):
    data=bpy.data.curves.new(name,"FONT")
    data.body=text;data.size=size;data.align_x="CENTER";data.align_y="CENTER"
    data.extrude=.00003;data.resolution_u=4
    obj=bpy.data.objects.new(name,data);collection(group).objects.link(obj)
    obj.location=center
    obj.rotation_euler=Vector(normal).to_track_quat("Z","Y").to_euler()
    data.materials.append(MATERIALS[material_name])
    return obj


def add_small_markings():
    label("Chest / equipment identifier","EVA  /  03",(0,-.177,1.457),.009,(0,-1,0),"08 / Markings")
    label("Chest / micro service label","PRESSURE SYSTEM",(0,-.177,1.363),.0045,(0,-1,0),"08 / Markings")
    label("Pack / maintenance label","LIFE SUPPORT",(0,.297,1.407),.006,(0,1,0),"08 / Markings")
    label("Pack / unit identifier","03",(0,.298,1.512),.024,(0,1,0),"08 / Markings")
    for x in (-.114,.114):
        for j in range(3):
            box(f"Chest / tiny index stroke {x} {j}",(x+j*.005,-.176,1.433),(.0023,.0005,.007),
                "Printed charcoal","08 / Markings",.0001)


def aim(obj,target):
    obj.rotation_euler=(Vector(target)-obj.location).to_track_quat("-Z","Y").to_euler()


def setup_studio():
    scene=bpy.context.scene
    scene.unit_settings.system="METRIC";scene.unit_settings.scale_length=1
    scene.render.engine="CYCLES"
    scene.cycles.samples=112
    scene.cycles.use_denoising=True
    scene.cycles.preview_samples=24
    scene.cycles.max_bounces=8
    scene.cycles.diffuse_bounces=3
    scene.cycles.glossy_bounces=4
    try:
        prefs=bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type="OPTIX"
        prefs.get_devices()
        for dev in prefs.devices:
            dev.use=dev.type=="OPTIX"
        if any(dev.type=="OPTIX" for dev in prefs.devices):
            scene.cycles.device="GPU"
    except Exception as exc:
        print("GPU setup fallback:",exc)
    scene.world.use_nodes=True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value=(.31,.36,.39,1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value=.33
    floor=box("Studio / seamless ground",(0,0,-.013),(200,200,.022),"Studio warm grey","90 / Studio",.001)
    for name,loc,power,size,color in (
        ("Key / broad softbox",(-3.0,-4.0,5.0),550,3.0,(1.0,.91,.79)),
        ("Fill / neutral softbox",(3.4,-1.7,2.8),340,2.7,(.82,.90,1.0)),
        ("Rim / upper strip",(1.8,3.3,4.0),650,2.5,(.86,.93,1.0)),
        ("Front / visor strip",(-.7,-3.5,2.8),70,1.0,(1.0,1.0,.95))):
        data=bpy.data.lights.new(name,"AREA");data.energy=power;data.shape="DISK";data.size=size;data.color=color
        obj=bpy.data.objects.new(name,data);collection("90 / Studio").objects.link(obj);obj.location=loc;aim(obj,(0,0,1))
    cameras={
        "front":((0,-5.5,1.02),(0,0,.97),2.13),
        "side":((5.5,0,1.02),(0,0,.97),2.13),
        "back":((0,5.5,1.02),(0,0,.97),2.13),
        "hero":((3.1,-5.5,2.75),(0,0,.99),2.17),
        "closeup":((1.65,-3.4,2.1),(0,-.02,1.55),.87),
        "elevated":((2.8,-3.8,5.6),(0,0,.96),2.23),
    }
    for name,(loc,target,scale) in cameras.items():
        data=bpy.data.cameras.new("Camera / "+name);data.type="ORTHO";data.ortho_scale=scale
        obj=bpy.data.objects.new("Camera / "+name,data);collection("91 / Review cameras").objects.link(obj)
        obj.location=loc;aim(obj,target)
    scene.camera=bpy.data.objects["Camera / hero"]
    scene.render.resolution_x=1400;scene.render.resolution_y=1600;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format="PNG"
    scene.render.image_settings.color_mode="RGBA"
    scene.view_settings.view_transform="AgX"
    scene.view_settings.look="AgX - Medium High Contrast"
    scene.view_settings.exposure=.1
    scene.render.film_transparent=False
    scene.render.image_settings.color_depth="8"
    # A packed non-rendering sheet makes the source useful away from this checkout.
    ref=bpy.data.images.load(str(REFERENCE));ref.name="REFERENCE / supplied turnaround";ref.pack()
    obj=bpy.data.objects.new("REFERENCE / Space Marine Turnaround Sheet",None)
    obj.empty_display_type="IMAGE";obj.data=ref;obj.empty_display_size=2.9
    obj.location=(2.4,.75,1.0);obj.rotation_euler=(math.pi/2,0,0);obj.hide_render=True
    collection("92 / Packed reference").objects.link(obj)
    collection("92 / Packed reference").hide_viewport=True
    scene["asset_status"]="Authored source study — unrigged, not retopologized, no baked game textures"
    scene["reference_sha256"]=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    scene["forward_axis"]="-Y"
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=="VIEW_3D":
                area.spaces.active.region_3d.view_distance=3.0
                area.spaces.active.region_3d.view_location=(0,0,1)
                area.spaces.active.region_3d.view_rotation=scene.camera.rotation_euler.to_quaternion()
                area.spaces.active.shading.type="MATERIAL"
    bpy.ops.object.select_all(action="DESELECT")
    shell=bpy.data.objects.get("Helmet / crown shell")
    shell.select_set(True);bpy.context.view_layer.objects.active=shell


def stats():
    deps=bpy.context.evaluated_depsgraph_get()
    source_vertices=source_faces=eval_vertices=eval_triangles=0
    bounds=[];objects=0
    for col_name,col in COLLECTIONS.items():
        if not col_name[:2].isdigit() or int(col_name[:2])>=90:
            continue
        for obj in col.objects:
            if obj.type not in {"MESH","CURVE","FONT"}:
                continue
            objects+=1
            if obj.type=="MESH":
                source_vertices+=len(obj.data.vertices);source_faces+=len(obj.data.polygons)
            evaluated=obj.evaluated_get(deps);mesh=evaluated.to_mesh()
            eval_vertices+=len(mesh.vertices);mesh.calc_loop_triangles();eval_triangles+=len(mesh.loop_triangles)
            bounds.extend(evaluated.matrix_world@v.co for v in mesh.vertices)
            evaluated.to_mesh_clear()
    minimum=[min(p[i] for p in bounds) for i in range(3)]
    maximum=[max(p[i] for p in bounds) for i in range(3)]
    result={"character_objects":objects,"source_mesh_vertices":source_vertices,"source_mesh_polygons":source_faces,
            "evaluated_vertices_including_curve_detail":eval_vertices,"evaluated_triangles":eval_triangles,
            "bounds_min_metres":minimum,"bounds_max_metres":maximum,
            "dimensions_metres":[b-a for a,b in zip(minimum,maximum)],
            "rigged":False,"uv_baked":False,"game_optimized":False,
            "reference_sha256":hashlib.sha256(REFERENCE.read_bytes()).hexdigest()}
    (SOURCE/"model_stats.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("MODEL_STATS",json.dumps(result))


def render_views(names, draft=False):
    scene=bpy.context.scene
    scene.cycles.samples=32 if draft else 112
    for name in names:
        scene.camera=bpy.data.objects["Camera / "+name]
        scene.render.resolution_x=800 if draft else (1600 if name=="closeup" else 1400)
        scene.render.resolution_y=1000 if draft else (1400 if name=="closeup" else 1700)
        scene.render.filepath=str(PREVIEWS/(("draft_" if draft else "")+name+".png"))
        bpy.ops.render.render(write_still=True)
        print("RENDERED",scene.render.filepath,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--draft",action="store_true")
    parser.add_argument("--no-render",action="store_true")
    parser.add_argument("--views",default="front,side,back,hero")
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    SOURCE.mkdir(exist_ok=True,parents=True);PREVIEWS.mkdir(exist_ok=True,parents=True)
    bpy.ops.object.select_all(action="SELECT");bpy.ops.object.delete(use_global=False)
    for data in list(bpy.data.collections):
        if data.users==0:bpy.data.collections.remove(data)
    make_materials()
    for fn in (build_garment,build_helmet,build_armour,build_belt_pockets,build_hands,build_boots,build_pack,add_small_markings):
        print("BUILD",fn.__name__,flush=True);fn()
    setup_studio();stats()
    bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/"space_marine.blend"),compress=True)
    print("CHECKPOINT_SAVED",flush=True)
    if not args.no_render:
        render_views(args.views.split(","),args.draft)
    scene=bpy.context.scene
    scene.camera=bpy.data.objects["Camera / hero"];scene.cycles.samples=112
    scene.render.resolution_x=1400;scene.render.resolution_y=1700
    scene.render.filepath="//../previews/hero.png"
    bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/"space_marine.blend"),compress=True)


if __name__=="__main__":
    main()
