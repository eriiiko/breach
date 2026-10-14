"""Offline compressed-cloth authoring experiment and baking helpers.

The source file keeps applied mesh geometry; the cloth cache is authoring-only.
No simulation here is part of Breach's deterministic runtime.
"""
import math
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def interpolate_profile(profiles,z):
    for i,(lo,hi) in enumerate(zip(profiles,profiles[1:])):
        if lo[0]-1e-8<=z<=hi[0]+1e-8:
            t=(z-lo[0])/(hi[0]-lo[0]);t=max(0,min(1,t))
            return [lo[j]+(hi[j]-lo[j])*t for j in range(1,len(lo))]
    return list(profiles[0 if z<profiles[0][0] else -1][1:])


def compressed_limb(author,name,profiles,cloth_ranges,side=1,compression=.13,sides=64,
                    bending=.045,pressure=.6,ring_spacing=.0042,localized=False,
                    solver_quality=10,elbow_surface=None):
    """Create one closed mesh with pinned black flex areas and cream cloth panels."""
    zmin,zmax=profiles[0][0],profiles[-1][0]
    heights={zmin,zmax}
    for lo,hi in cloth_ranges:
        heights.update((lo,hi))
    heights.update(p[0] for p in profiles)
    stops=sorted(heights);zs=[]
    for lo,hi in zip(stops,stops[1:]):
        count=max(2,math.ceil((hi-lo)/ring_spacing))
        zs.extend(lo+(hi-lo)*j/count for j in range(count))
    zs.append(zmax)
    verts=[];final=[];faces=[];face_mats=[];pin=[];soft_pins=[]
    elbow_bvh=None
    if elbow_surface is not None:
        elbow_bvh=BVHTree.FromObject(elbow_surface,bpy.context.evaluated_depsgraph_get())
    for row,z in enumerate(zs):
        cx,cy,rx,ry=interpolate_profile(profiles,z)
        material_index=0
        nearest=1e9
        cloth_t=0
        for lo,hi in cloth_ranges:
            if lo<z<hi:
                material_index=1;nearest=min(z-lo,hi-z);cloth_t=(z-lo)/(hi-lo)
        extra=compression*sum(max(0,hi-max(z,lo)) for lo,hi in cloth_ranges)
        zlo=max(zmin,z-.001);zhi=min(zmax,z+.001)
        l=interpolate_profile(profiles,zlo);h=interpolate_profile(profiles,zhi)
        tangent=Vector((side*(h[0]-l[0]),h[1]-l[1],zhi-zlo)).normalized()
        bu=Vector((0,1,0)).cross(tangent).normalized();bv=tangent.cross(bu).normalized()
        for k in range(sides):
            a=k/sides*math.tau
            noise=.00065*math.sin(a*5+z*76)*math.sin(a*2-z*35)
            if not material_index:noise=.0014*math.sin(z*110+a*1.8)*math.sin((z-zmin)/(zmax-zmin)*math.pi)
            if localized and material_index:
                # Millimetre rest-shape asymmetry seeds short buckles; the solver
                # supplies the visible folds rather than a final displacement field.
                envelope=.4+.6*abs(2*cloth_t-1)
                noise=.0011*envelope*(math.sin(a*5+z*151)+.45*math.sin(a*9-z*233))
            elif localized and z<cloth_ranges[0][0]:
                noise=0
            p=Vector((side*cx,cy,z))+bu*((rx+noise)*math.cos(a))+bv*((ry+noise)*math.sin(a))
            if elbow_bvh is not None and cloth_ranges[0][1]<z<cloth_ranges[1][0]:
                # Resample the accepted flex surface into this mesh. It remains
                # pinned, with a fade into the shared ivory/black boundary rings.
                centre=Vector((side*cx,cy,z))
                direction=(p-centre).normalized()
                hit,normal,face,distance=elbow_bvh.ray_cast(centre,direction,.20)
                if hit is not None:
                    edge=min(z-cloth_ranges[0][1],cloth_ranges[1][0]-z)
                    blend=min(1,edge/.012)
                    blend=blend*blend*(3-2*blend)
                    p=p.lerp(hit,blend)
            final.append(p)
            verts.append(p+Vector((side*extra*.34,0,-extra)))
            index=row*sides+k
            if not material_index or nearest<.009:pin.append(index)
            elif localized:
                # Two broad, weak sewing restraints along each existing seam stop
                # all excess length migrating into a single fold at the wrist.
                angular=min(abs((a-seam+math.pi)%math.tau-math.pi) for seam in (-1.4,1.55))
                axial=sum(math.exp(-((cloth_t-centre)/.12)**2) for centre in (.34,.71))
                weight=.42*math.exp(-(angular/.22)**2)*min(1,axial)
                if weight>.015:soft_pins.append((index,weight))
    for row in range(len(zs)-1):
        mid=(zs[row]+zs[row+1])/2
        m=1 if any(lo<mid<hi for lo,hi in cloth_ranges) else 0
        if localized and mid<cloth_ranges[0][0]:m=2
        for k in range(sides):
            kn=(k+1)%sides
            faces.append((row*sides+k,row*sides+kn,(row+1)*sides+kn,(row+1)*sides+k));face_mats.append(m)
    faces.append(tuple(reversed(range(sides))));face_mats.append(2 if localized else 0)
    faces.append(tuple((len(zs)-1)*sides+k for k in range(sides)));face_mats.append(0)
    obj=author.mesh_object(name,verts,faces,"Charcoal flex textile","01 / Pressure garment")
    obj.data.materials.append(author.MATERIALS["Warm woven pressure fabric"])
    if localized:obj.data.materials.append(author.MATERIALS["Graphite elastomer"])
    for polygon,index in zip(obj.data.polygons,face_mats):polygon.material_index=index
    basis=obj.shape_key_add(name="Uncompressed sewing rest shape")
    compressed=obj.shape_key_add(name="Fitted endpoint compression")
    for vertex,co in zip(compressed.data,final):vertex.co=co
    compressed.value=0;compressed.keyframe_insert(data_path="value",frame=1)
    compressed.value=1;compressed.keyframe_insert(data_path="value",frame=23)
    group=obj.vertex_groups.new(name="Pinned seals and sewn boundaries");group.add(pin,1,"REPLACE")
    for index,weight in soft_pins:group.add([index],weight,"REPLACE")
    mod=obj.modifiers.new("AUTHORING ONLY / pressure cloth","CLOTH")
    settings=mod.settings
    settings.quality=solver_quality;settings.mass=.20;settings.air_damping=4
    settings.tension_stiffness=25;settings.compression_stiffness=25;settings.shear_stiffness=12
    settings.bending_stiffness=bending;settings.bending_damping=.8
    settings.vertex_group_mass=group.name;settings.pin_stiffness=1
    settings.rest_shape_key=basis
    settings.use_pressure=True;settings.uniform_pressure_force=pressure;settings.pressure_factor=.5
    mod.collision_settings.use_collision=False;mod.collision_settings.use_self_collision=False
    mod.point_cache.frame_start=1;mod.point_cache.frame_end=42
    return obj,{"zs":zs,"sides":sides,"cloth_ranges":cloth_ranges,"final":final,
                "pin_indices":pin,"soft_pin_vertices":len(soft_pins),
                "compression":compression,"bending":bending,"pressure":pressure,
                "solver_quality":solver_quality}


def bake_compressed_cloth(objects,end=42):
    scene=bpy.context.scene;scene.gravity=(0,0,-.65)
    for frame in range(1,end+1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        for obj in objects:
            obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        if frame%5==0:print("CLOTH_FRAME",frame,flush=True)
    depsgraph=bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        mesh=bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph),preserve_all_data_layers=True,depsgraph=depsgraph)
        mesh.name=obj.name+" / applied cloth surface"
        obj.modifiers.clear();obj.data=mesh;obj.animation_data_clear()
        for face in mesh.polygons:face.use_smooth=True
        obj["authoring_method"]="Pinned pressure cloth, applied as editable mesh; parameters in study metrics"
    scene.frame_set(1);scene.gravity=(0,0,-9.81)


def sleeve_profiles(extended_cuff=False):
    profiles=[
        (.963,.478,-.013,.051,.048),(.992,.474,-.013,.058,.055),
        (1.008,.469,-.013,.054,.052),(1.038,.455,-.011,.064,.064),
        (1.080,.437,-.009,.073,.071),(1.126,.417,-.008,.071,.068),
        (1.144,.409,-.008,.070,.067),(1.188,.395,-.009,.075,.074),
        (1.221,.382,-.010,.073,.071),(1.242,.371,-.008,.074,.072),
        (1.290,.348,-.004,.083,.083),(1.357,.318,.000,.087,.086),
        (1.416,.289,.001,.074,.075),(1.447,.276,.002,.070,.079),
        (1.490,.260,.002,.074,.082),(1.531,.250,.002,.069,.072)]
    if extended_cuff:
        # Match the accepted seal's lower extent, covering the internal binding.
        profiles.insert(0,(.953,.480,-.013,.050,.047))
    return profiles
