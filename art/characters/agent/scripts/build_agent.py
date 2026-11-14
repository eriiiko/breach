"""Breach agent silhouette study, authored from Erik's concept (2026-10-03).

Blender 4.5 bundled Python. Editable ring surfaces; no rigging or runtime export.
Coordinates: metres, Blender Z-up, character faces -Y. Equipment is independent.
"""
import bpy
import json
import math
from pathlib import Path
from mathutils import Vector

HOME = Path(__file__).resolve().parents[1]
SOURCE = HOME / 'source'
PREVIEWS = HOME / 'previews'
SOURCE.mkdir(parents=True, exist_ok=True)
PREVIEWS.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'

def collection(name):
    c = bpy.data.collections.new(name)
    scene.collection.children.link(c)
    return c

BODY = collection('AGENT | unrigged silhouette surfaces')
BAG = collection('EQUIPMENT | removable soft duffel placeholder')
WEAPON = collection('EQUIPMENT | removable compact weapon placeholder')
STUDIO = collection('REVIEW | cameras lights floor')

def material(name, color, roughness=0.6, metallic=0.0):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*color, 1)
    p.inputs['Roughness'].default_value = roughness
    p.inputs['Metallic'].default_value = metallic
    return m

PALE = material('Armour | warm off-white | adjustable', (0.72,0.73,0.69), 0.48, 0.15)
SUIT = material('Suit | fitted charcoal | adjustable', (0.045,0.052,0.062), 0.83)
BLACK = material('Rubber and webbing | near black', (0.013,0.018,0.022),0.86)
VISOR = material('Visor | smoked blue black | adjustable tint', (0.016,0.029,0.036),0.19,0.35)
TEAL = material('Bag | muted teal canvas | adjustable', (0.035,0.16,0.20),0.84)
EDGE = material('Hardware | gunmetal', (0.1,0.12,0.13),0.44,0.5)

def mesh(name, verts, faces, mat, coll=BODY, smooth=False):
    data = bpy.data.meshes.new(name+' surface')
    data.from_pydata(verts, [], faces)
    data.update()
    o = bpy.data.objects.new(name,data)
    coll.objects.link(o)
    o.data.materials.append(mat)
    for p in data.polygons: p.use_smooth = smooth
    return o

def rings(name, rows, mat, coll=BODY, segments=16, smooth=True):
    # Each row: centre X/Y/Z, ellipse radius X/Y. Quad strips stay editable.
    verts=[]
    for x,y,z,rx,ry in rows:
        verts.extend((x+rx*math.cos(2*math.pi*i/segments),y+ry*math.sin(2*math.pi*i/segments),z) for i in range(segments))
    faces=[]
    for j in range(len(rows)-1):
        for i in range(segments):
            a=j*segments+i;b=j*segments+(i+1)%segments
            faces.append((a,b,b+segments,a+segments))
    faces += [tuple(reversed(range(segments))),tuple((len(rows)-1)*segments+i for i in range(segments))]
    return mesh(name,verts,faces,mat,coll,smooth)

def tube(name, a,b, radii, mat, coll=BODY, segments=12):
    axis=(Vector(b)-Vector(a)).normalized()
    u=axis.cross(Vector((0,1,0))).normalized()
    if u.length<0.1: u=axis.cross(Vector((1,0,0))).normalized()
    v=axis.cross(u).normalized()
    verts=[]
    for j,(fraction,r) in enumerate(radii):
        centre=Vector(a).lerp(Vector(b),fraction)
        verts.extend(tuple(centre+u*math.cos(i*2*math.pi/segments)*r+v*math.sin(i*2*math.pi/segments)*r*0.85) for i in range(segments))
    faces=[]
    for j in range(len(radii)-1):
        for i in range(segments):
            k=j*segments+i;n=j*segments+(i+1)%segments
            faces.append((k,n,n+segments,k+segments))
    faces += [tuple(reversed(range(segments))),tuple((len(radii)-1)*segments+i for i in range(segments))]
    return mesh(name,verts,faces,mat,coll,True)

def panel(name, outline, mat=PALE, thickness=0.028, coll=BODY):
    # Outline is on a tailored, sometimes curved front surface. Back displaced +Y.
    n=len(outline)
    verts=list(outline)+[(x,y+thickness,z) for x,y,z in outline]
    # Fan around a deliberately faceted centre, not a flat stretched cube.
    centre=tuple(sum(p[j] for p in outline)/n for j in range(3))
    verts.append((centre[0],centre[1]-0.012,centre[2]))
    faces=[(i,(i+1)%n,2*n) for i in range(n)]
    faces += [tuple(reversed(range(n,2*n)))]
    faces += [(i,i+n,(i+1)%n+n,(i+1)%n) for i in range(n)]
    o=mesh(name,verts,faces,mat,coll)
    bevel=o.modifiers.new('Soft manufacturing edge | editable','BEVEL')
    bevel.width=0.006;bevel.segments=2
    o.modifiers.new('Weighted armour normals','WEIGHTED_NORMAL')
    return o

def box(name, loc, scale, mat, coll=BODY, bevel=0.018):
    bpy.ops.mesh.primitive_cube_add(size=1,location=loc)
    o=bpy.context.object;o.name=name;o.scale=scale
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    for c in list(o.users_collection):c.objects.unlink(o)
    coll.objects.link(o);o.data.materials.append(mat)
    if bevel:
        m=o.modifiers.new('Rounded placeholder corners','BEVEL');m.width=bevel;m.segments=3
        o.modifiers.new('Weighted corner normals','WEIGHTED_NORMAL')
    return o

def strap(name, points, width, mat=BLACK, coll=BAG):
    verts=[]
    for x,y,z in points: verts.extend([(x-width/2,y,z),(x+width/2,y,z)])
    o=mesh(name,verts,[(i*2,i*2+1,i*2+3,i*2+2) for i in range(len(points)-1)],mat,coll)
    m=o.modifiers.new('Webbing thickness','SOLIDIFY');m.thickness=0.008
    m=o.modifiers.new('Webbing edge','BEVEL');m.width=0.004;m.segments=2
    return o

# Narrow athletic suit: waist below ribs, modest chest; no sculpted exposed anatomy.
rings('Suit torso | tailored waist and modest chest',[(0,.015,.865,.049,.062),(0,.014,.90,.112,.090),(0,.012,.96,.156,.106),(0,.008,1.04,.147,.10),(0,.004,1.12,.115,.087),(0,0,1.21,.121,.092),(0,-.004,1.30,.155,.108),(0,-.006,1.38,.177,.115),(0,0,1.45,.184,.10),(0,.006,1.49,.142,.077)],SUIT)
rings('Sealed neck gasket',[(0,0,1.44,.070,.064),(0,0,1.50,.069,.064),(0,0,1.55,.068,.065)],BLACK)
rings('Armoured sealed collar',[(0,0,1.475,.089,.079),(0,0,1.525,.086,.077)],PALE,segments=12,smooth=False)
for z in (1.485,1.501): rings('Collar flexible seal band',[(0,0,z,.073,.068),(0,0,z+.008,.073,.068)],BLACK)

# Helmet shell profile: narrow jaw, broad crown, enclosed rear; dark visor wraps sides.
rings('Helmet | enclosed faceted shell',[(0,-.003,1.54,.055,.061),(0,0,1.58,.082,.084),(0,0,1.65,.097,.096),(0,.005,1.73,.102,.101),(0,.006,1.80,.089,.085),(0,.009,1.845,.061,.056),(0,.010,1.86,.014,.013)],PALE,segments=20,smooth=False)
verts=[]
for z,rx,ry in [(1.635,.096,.098),(1.65,.102,.101),(1.715,.105,.104),(1.727,.099,.103)]:
    for i in range(13):
        a=math.radians(195+i*150/12)
        verts.append((rx*math.cos(a),ry*math.sin(a)-.010,z))
faces=[(j*13+i,j*13+i+1,(j+1)*13+i+1,(j+1)*13+i) for j in range(3) for i in range(12)]
o=mesh('Visor | wraparound dark sealed lens',verts,faces,VISOR,smooth=True)
m=o.modifiers.new('Visor lens thickness','SOLIDIFY');m.thickness=.005
panel('Respirator | original angular lower face', [(-.038,-.092,1.647),(0,-.113,1.655),(.038,-.092,1.647),(.031,-.108,1.578),(0,-.117,1.557),(-.031,-.108,1.578)],EDGE,.02)
for s in (-1,1):
    panel('Helmet cheek armour '+str(s),[(s*.036,-.090,1.64),(s*.073,-.065,1.66),(s*.082,-.065,1.60),(s*.049,-.086,1.563),(s*.032,-.10,1.578)])
    o=box('Helmet ear seal '+str(s),(s*.099,.005,1.67),(.025,.077,.082),EDGE,bevel=.016)
    tube('Respirator filter '+str(s),(s*.052,-.081,1.59),(s*.060,-.105,1.59),[(0,.016),(1,.016)],BLACK)

# Chest plates follow body taper, with a central modest pectoral volume and clear underarm gaps.
for s in (-1,1):
    outline=[(s*.013,-.117,1.435),(s*.119,-.112,1.447),(s*.167,-.113,1.383),(s*.150,-.136,1.304),(s*.093,-.118,1.275),(s*.018,-.110,1.30)]
    panel('Chest armour | tailored half '+str(s),outline)
    panel('Shoulder cap | faceted '+str(s),[(s*.177,-.075,1.47),(s*.228,-.075,1.457),(s*.274,-.070,1.419),(s*.276,-.086,1.35),(s*.238,-.103,1.326),(s*.204,-.091,1.37)],thickness=.11)
    panel('Upper chest harness '+str(s),[(s*.127,-.127,1.477),(s*.151,-.122,1.475),(s*.171,-.138,1.369),(s*.150,-.146,1.357)],BLACK,.012)
    panel('Back armour | provisional half '+str(s),[(s*.013,.145,1.44),(s*.127,.125,1.44),(s*.174,.106,1.378),(s*.141,.138,1.30),(s*.078,.139,1.28),(s*.018,.14,1.31)],thickness=-.025)
panel('Chest armour | connecting central breastplate',[(-.033,-.133,1.432),(.033,-.133,1.432),(.043,-.156,1.37),(.035,-.143,1.311),(0,-.127,1.283),(-.035,-.143,1.311),(-.043,-.156,1.37)],thickness=.028)
panel('Chest centre insert', [(-.012,-.122,1.436),(.012,-.122,1.436),(.013,-.143,1.397),(-.013,-.143,1.397)],EDGE,.01)
rings('Waist utility belt',[(0,.012,1.03,.160,.114),(0,.012,1.065,.154,.110)],BLACK,segments=16,smooth=False)
box('Belt clasp',(0,-.103,1.049),(.046,.022,.033),EDGE,bevel=.004)

for s,side in [(-1,'L'),(1,'R')]:
    # Relaxed review stance; arms curve outward rather than intersect chest/pelvis.
    shoulder=(s*.221,.012,1.419);elbow=(s*.289,.005,1.191);wrist=(s*.36,-.007,.957)
    tube(side+' upper arm fitted suit',shoulder,elbow,[(0,.062),(.25,.066),(.65,.052),(1,.047)],SUIT)
    tube(side+' elbow flexible seal',elbow,(s*.306,.003,1.14),[(0,.049),(1,.047)],BLACK)
    tube(side+' forearm fitted suit',(s*.30,.003,1.16),wrist,[(0,.05),(.38,.052),(1,.034)],SUIT)
    panel(side+' forearm vambrace',[(s*.277,-.050,1.151),(s*.324,-.050,1.143),(s*.382,-.043,.974),(s*.346,-.048,.959),(s*.318,-.067,1.01)],thickness=.049)
    tube(side+' wrist seal',(s*.356,-.007,.972),(s*.367,-.008,.932),[(0,.037),(1,.036)],BLACK)
    rings(side+' glove palm',[(s*.37,-.008,.942,.033,.027),(s*.382,-.014,.90,.035,.028),(s*.384,-.016,.855,.030,.024)],BLACK,segments=12)
    panel(side+' glove knuckle shield',[(s*.351,-.042,.931),(s*.397,-.042,.922),(s*.410,-.042,.886),(s*.362,-.043,.89)],thickness=.014)
    # Grouped fingers are intentionally rough, separated enough to read as a glove.
    for k in range(4):
        x=s*(.365+k*.013)
        tube(side+' glove finger '+str(k),(x,-.012,.863),(x+s*.004,-.023,.816+(abs(k-1.5)*.008)),[(0,.007),(1,.005)],BLACK,segments=8)
    tube(side+' glove thumb',(s*.358,-.010,.905),(s*.348,-.041,.865),[(0,.011),(1,.008)],BLACK,segments=10)
    # Long slim legs, narrow upper thighs, subtle knee and calf forms.
    x=s*.093
    rings(side+' fitted leg | thigh knee calf',[(s*.071,.014,.985,.053,.054),(s*.091,.014,.932,.072,.081),(s*.101,.014,.865,.084,.092),(s*.112,.012,.78,.078,.084),(s*.119,.009,.665,.060,.067),(s*.122,0,.575,.047,.052),(s*.124,.01,.52,.050,.055),(s*.130,.022,.435,.056,.064),(s*.136,.017,.34,.043,.051),(s*.14,.011,.205,.030,.036),(s*.14,.011,.15,.030,.034)],SUIT)
    panel(side+' knee plate',[(s*.080,-.055,.622),(s*.155,-.055,.622),(s*.173,-.066,.576),(s*.149,-.070,.531),(s*.095,-.070,.531),(s*.074,-.065,.568)],thickness=.026)
    panel(side+' shin armour | long tapered shell',[(s*.092,-.054,.514),(s*.165,-.054,.514),(s*.181,-.066,.432),(s*.165,-.054,.233),(s*.153,-.040,.194),(s*.113,-.040,.195),(s*.096,-.054,.30),(s*.077,-.060,.421)],thickness=.049)
    # The rear armour stops at calf; flexible ankle stays visible.
    panel(side+' rear calf protection',[(s*.100,.099,.476),(s*.159,.099,.476),(s*.167,.089,.382),(s*.154,.074,.263),(s*.117,.074,.263),(s*.096,.089,.382)],thickness=-.016)
    rings(side+' boot upper sealed cuff',[(s*.14,.008,.105,.039,.046),(s*.14,.011,.185,.037,.043)],BLACK,segments=12)
    # Authored boot mesh uses unequal front/back Y depths rather than a cube foot.
    outline=[(-.042,-.17),(.042,-.17),(.049,-.13),(.043,.048),(-.043,.048),(-.049,-.13)]
    verts=[(s*.14+dx,y,z) for z in (.026,.075,.119) for dx,y in outline]
    # toe upper lowered, heel upper taller
    for i in range(6):
        vx,vy,vz=verts[12+i];verts[12+i]=(vx,vy,.075 if i in (0,1) else (.10 if i in (2,5) else .14))
    faces=[(j*6+i,j*6+(i+1)%6,(j+1)*6+(i+1)%6,(j+1)*6+i) for j in range(2) for i in range(6)]
    faces += [tuple(reversed(range(6))),tuple(range(12,18))]
    mesh(side+' boot | shaped toe and heel',verts,faces,BLACK)
    panel(side+' boot instep armour',[(s*.14-.04,-.142,.085),(s*.14+.04,-.142,.085),(s*.14+.038,-.065,.116),(s*.14+.027,.012,.151),(s*.14-.027,.012,.151),(s*.14-.038,-.065,.116)],thickness=-.018)
    box(side+' lateral thigh utility pocket',(s*.183,.015,.833),(.044,.1,.117),BLACK,bevel=.012)

# Independent soft duffel: shaped rectangular canvas loft with slightly irregular ends.
bag= rings('Duffel | soft canvas loft',[(.545,0,.285,.108,.100),(.55,0,.31,.146,.121),(.55,.005,.40,.151,.127),(.55,.0,.51,.14,.116),(.55,0,.565,.104,.088)],TEAL,BAG,segments=16,smooth=False)
for x in (.478,.625):
    strap('Duffel load webbing',[(x,-.105,.30),(x,-.132,.38),(x,-.119,.51),(x,-.084,.562),(x,.071,.562),(x,.118,.49),(x,.129,.37),(x,.09,.30)],.025)
strap('Duffel carry loop front',[(.475,-.069,.55),(.448,-.044,.66),(.394,-.008,.815),(.426,-.009,.819),(.600,-.04,.665),(.625,-.068,.55)],.024)
strap('Duffel carry loop rear',[(.475,.061,.55),(.448,.035,.66),(.394,.008,.815),(.426,.008,.819),(.600,.038,.665),(.625,.06,.55)],.024)
strap('Duffel zip seam',[(.444,-.004,.568),(.50,-.004,.576),(.59,-.004,.576),(.655,-.004,.565)],.009,EDGE)
for x in (.478,.625): box('Duffel strap buckle',(x,-.123,.49),(.034,.01,.027),EDGE,BAG,bevel=.004)

# Compact original bullpup housing placeholder; true open grip apertures, no imported gun.
# Local weapon frame: X=length, Y=width, Z=height, later hung muzzle-down in left glove.
gunparts=[]
def gunbox(name,loc,scale,mat=BLACK):
    o=box(name,loc,scale,mat,WEAPON,bevel=.012);gunparts.append(o);return o
gunbox('Weapon | original compact receiver',(-.08,0,.02),(.31,.056,.075))
gunbox('Weapon | top magazine placeholder',(-.04,0,.067),(.26,.040,.029),EDGE)
gunbox('Weapon | butt housing',(-.235,0,-.005),(.063,.067,.12))
gunbox('Weapon | foregrip bridge',(.070,0,-.035),(.095,.056,.03))
gunbox('Weapon | rear grip',(-.138,0,-.070),(.031,.041,.095))
gunbox('Weapon | front grip',(.090,0,-.07),(.027,.041,.068))
gunbox('Weapon | lower grip bridge',(-.013,0,-.113),(.223,.037,.022))
gunbox('Weapon | short barrel',(.103,0,.006),(.071,.029,.029),EDGE)
gunbox('Weapon | sight rail',(-.05,0,.103),(.10,.029,.027),BLACK)
anchor=bpy.data.objects.new('Weapon asset pivot | independent',None);WEAPON.objects.link(anchor)
for o in gunparts:o.parent=anchor
anchor.rotation_euler[1]=math.radians(72)
anchor.location=(-.405,-.005,.715)

# Review stage is kept independent and excluded from character statistics.
floor=box('Studio floor',(0,0,.0085),(200,200,.035),material('Studio | graphite',(0.092,.108,.125),.95),STUDIO,bevel=0)
world=bpy.data.worlds.new('Studio neutral world');world.use_nodes=True
world.node_tree.nodes['Background'].inputs[0].default_value=(.25,.29,.34,1)
world.node_tree.nodes['Background'].inputs[1].default_value=.4;scene.world=world
def light(name,loc,power,size):
    data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size
    o=bpy.data.objects.new(name,data);STUDIO.objects.link(o);o.location=loc
    o.rotation_euler=(Vector((0,0,1))-o.location).to_track_quat('-Z','Y').to_euler()
light('Key softbox',(-3,-4,5),550,4)
light('Fill softbox',(3,-1,3),350,3)
light('Rear softbox',(0,3,4),650,3)
scene.render.engine='CYCLES'
scene.cycles.samples=32
scene.cycles.use_denoising=True
scene.render.resolution_x=900;scene.render.resolution_y=1100;scene.render.resolution_percentage=100
scene.view_settings.view_transform='AgX'
scene.render.image_settings.file_format='PNG'
scene.render.film_transparent=False
scene.render.use_file_extension=True

def camera(name,loc,target,scale=2.15,vertical=False):
    data=bpy.data.cameras.new(name);data.type='ORTHO';data.ortho_scale=scale
    o=bpy.data.objects.new(name,data);STUDIO.objects.link(o);o.location=loc
    o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()
    if vertical: o.rotation_euler=(0,0,0)  # optical -Z, image +Y upward; no tilt.
    return o
views=[('front',(0,-5,.96),(0,0,.96),2.10),('side',(5,0,.96),(0,0,.96),2.10),('back',(0,5,.96),(0,0,.96),2.10),('three_quarter',(3,-5,2.3),(0,0,.95),2.17),('top_down_vertical',(.06,0,6),(.06,0,0),1.68),('top_down_tilted_readability',(2,-3,5),(0,0,.90),1.70)]
cameras={}
for name,loc,target,scale in views:cameras[name]=camera('REVIEW '+name,loc,target,scale,name=='top_down_vertical')
scene.camera=cameras['three_quarter']
scene['milestone']='ROUGH SILHOUETTE: unrigged, UV/textures/details deferred; equipment placeholders'
scene['coordinate_mapping']='Blender X,-Y,Z => runtime X,forward-Z,height-Y at future glTF export; top camera has zero tilt.'

def stats(coll):
    result={'objects':0,'vertices':0,'triangles_base':0,'triangles_evaluated':0}
    graph=bpy.context.evaluated_depsgraph_get()
    for o in coll.objects:
        if o.type!='MESH':continue
        result['objects']+=1;result['vertices']+=len(o.data.vertices)
        o.data.calc_loop_triangles();result['triangles_base']+=len(o.data.loop_triangles)
        ev=o.evaluated_get(graph);m=ev.to_mesh();m.calc_loop_triangles()
        result['triangles_evaluated']+=len(m.loop_triangles);ev.to_mesh_clear()
    return result
(SOURCE/'mesh_statistics.json').write_text(json.dumps({'agent':stats(BODY),'duffel_placeholder':stats(BAG),'weapon_placeholder':stats(WEAPON),'unit':'metres','height':1.86,'rigged':False,'UVs_authored':False,'runtime_export':False},indent=2),encoding='utf8')
for o in bpy.context.selected_objects:o.select_set(False)
for o in BODY.objects:o.select_set(True)
bpy.context.view_layer.objects.active=next(o for o in BODY.objects if o.type=='MESH')
# Make the saved file comfortable to open in a standard solid/material viewport.
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_distance=3.2
            area.spaces.active.region_3d.view_location=(0,0,.95)
            area.spaces.active.shading.color_type='MATERIAL'
            area.spaces.active.overlay.show_floor=False
bpy.ops.wm.save_as_mainfile(filepath=str(SOURCE/'agent_silhouette.blend'))
for name,*_ in views:
    scene.camera=cameras[name]
    floor.hide_render=name in ('front','side','back')
    scene.render.filepath=str(PREVIEWS/(name+'.png'))
    bpy.ops.render.render(write_still=True)
print('AGENT_SILHOUETTE_COMPLETE',json.loads((SOURCE/'mesh_statistics.json').read_text()))
