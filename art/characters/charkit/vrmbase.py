"""A character started from a VRM base model (VRoid Studio's CC0 samples) -- Beatrice's VRoid spike, 2026-10-22.

The scratch characters build every surface from fields; this module starts from a finished anime base instead: a VRM 0.x
file (a glTF with a humanoid skeleton, expression shape keys and toon (MToon) materials) imported by the VRM add-on for
Blender (saturday06, extensions.blender.org, MIT / GPL-3.0; installed once with
`blender --command extension install-file -r user_default --no-prefs add-on-vrm-vX.zip`, enabled here per process).
Nothing is done by hand: every step is a function, the character's script holds the tables.

    vrm = vrmbase.load(path, "Vroid")                  # import; colliders and the add-on's outline modifiers removed
    vrmbase.split_face(vrm)                            # the face's painted layers as their own objects (eyes, lashes, ...)
    vrmbase.materials(vrm, "vroid", recolour)          # MToon -> a Principled BSDF on the lit texture (what toonlook converts),
                                                       # named `<prefix>_<kind>`, each optionally recoloured
    vrmbase.drop(vrm, kinds=("cloth",))                # the sample's clothes removed (the body is whole under them)
    shell = vrmbase.shell(vrm, "Vroid_Suit", mat, 0.0015, fields)   # a skin-tight layer over the body, cut by signed fields

The materials are rebuilt round the VRM's own LIT texture (its painted eyes, brows, lashes, mouth, blush and the hair's
strands stay -- that is the point of a base); the shade texture, the matcap, the rim and the outline of MToon are left to the
character's look (`charkit/toonlook.py`, which converts a Principled material).
"""
import fnmatch
import math
import re

import bpy
import numpy as np

ADDON = "bl_ext.user_default.vrm"

# VRoid's material names -> a kind (the first pattern that matches wins)
KINDS = (("*EyeHighlight*", "eye_hi"), ("*EyeIris*", "iris"), ("*EyeWhite*", "eye_white"), ("*EyeExtra*", "eye_extra"),
         ("*FaceEyeline*", "eyeline"), ("*FaceEyelash*", "lash"), ("*FaceBrow*", "brow"), ("*FaceMouth*", "mouth"),
         ("*_Face_*SKIN*", "face"), ("*Body_*SKIN*", "skin"), ("*HairBack*", "hair_back"), ("*_HAIR*", "hair"),
         ("*CLOTH*", "cloth"))
# the face's painted layers, split off the face mesh into their own objects (so an outline never rings a transparent card)
FACE_LAYERS = ("eye_hi", "iris", "eye_white", "eye_extra", "eyeline", "lash", "brow", "mouth")


def kind_of(name):
    for pat, k in KINDS:
        if fnmatch.fnmatch(name, pat):
            return k
    return None


def enable():
    import addon_utils
    mod = addon_utils.enable(ADDON, default_set=True, persistent=False)   # default_set: the add-on reads its preferences
    if mod is None:
        raise RuntimeError("the VRM add-on is not installed: blender --command extension install-file -r user_default "
                           "--no-prefs <add-on-vrm-vX.zip>")
    return getattr(mod, "bl_info", {}).get("version") or _version()


def _version():
    import os
    import tomllib
    import addon_utils
    for m in addon_utils.modules():
        if m.__name__ == ADDON:
            p = os.path.join(os.path.dirname(m.__file__), "blender_manifest.toml")
            with open(p, "rb") as f:
                return tomllib.load(f).get("version")
    return None


class Vrm:
    """An imported VRM: `arm` the armature, `parts` {role: object} (`face`, `body`, `hair`, and after `split_face` one
    object per painted layer)."""

    def __init__(self, arm, parts, name):
        self.arm, self.parts, self.name = arm, parts, name

    def bone(self, name):
        """A bone's rest head in world space (VRM humanoid names without the `J_Bip_` prefix work: `C_Head`, `L_Hand`)."""
        b = self.arm.data.bones.get(name) or self.arm.data.bones.get("J_Bip_" + name)
        return np.asarray(self.arm.matrix_world @ b.head_local)

    def meshes(self):
        return [o for o in self.parts.values() if o is not None and o.type == "MESH"]


def load(path, name, collection=None):
    """Import a VRM and tidy it: its spring-bone colliders (empties) removed, the add-on's outline modifiers removed, the
    objects renamed `<name>_Face` / `_Body` / `_Hair` / `_Armature`."""
    enable()
    before = set(bpy.data.objects)
    r = bpy.ops.import_scene.vrm(filepath=path, use_addon_preferences=False, extract_textures_into_folder=False,
                                 make_new_texture_folder=False, set_shading_type_to_material_on_import=False,
                                 set_view_transform_to_standard_on_import=False, set_armature_display_to_wire=False,
                                 set_armature_display_to_show_in_front=False, set_armature_bone_shape_to_default=False,
                                 enable_mtoon_outline_preview=False)
    if "FINISHED" not in r:
        raise RuntimeError("VRM import failed: %s %s" % (path, r))
    new = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new if o.type == "ARMATURE")
    parts = {}
    for o in new:
        if o.type == "EMPTY":
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        if o.type != "MESH":
            continue
        for md in list(o.modifiers):
            if md.type == "NODES":   # the add-on's MToon outline (the look draws its own)
                o.modifiers.remove(md)
        role = o.name.split(".")[0].lower()
        role = "hair" if role.startswith("hair") else role
        o.name = "%s_%s" % (name, role.capitalize())
        parts[role] = o
    arm.name = name + "_Armature"
    for m in list(bpy.data.materials):   # the add-on's outline materials, now unused
        if m.name.startswith("MToon Outline") and m.users == 0:
            bpy.data.materials.remove(m)
    if collection:
        coll = bpy.data.collections.get(collection) or bpy.data.collections.new(collection)
        if coll.name not in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.link(coll)
        for o in [arm] + list(parts.values()):
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
    return Vrm(arm, parts, name)


def _separate(ob, keep_kinds, new_name):
    """Move the faces of `ob` whose material is of `keep_kinds` into a new object (vertex groups and shape keys kept)."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    me = ob.data
    sel = np.array([kind_of(me.materials[p.material_index].name if me.materials[p.material_index] else "") in keep_kinds
                    for p in me.polygons], bool)
    if not sel.any():
        return None
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    me.polygons.foreach_set("select", sel)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    new = [o for o in bpy.context.selected_objects if o is not ob][0]
    new.name = new_name
    # an object holding only some materials keeps every slot; drop the unused ones
    _prune_slots(new)
    _prune_slots(ob)
    return new


def _prune_slots(ob):
    used = {p.material_index for p in ob.data.polygons}
    for i in reversed(range(len(ob.material_slots))):
        if i not in used:
            ob.active_material_index = i
            bpy.context.view_layer.objects.active = ob
            bpy.ops.object.material_slot_remove()


def split_face(vrm):
    """The face's painted layers (eyes, lashes, brows, the mouth card) as their own object `<name>_FaceLayers`."""
    face = vrm.parts.get("face")
    if face is None:
        return
    lay = _separate(face, FACE_LAYERS, vrm.name + "_FaceLayers")
    if lay is not None:
        vrm.parts["face_layers"] = lay


def drop(vrm, kinds=("cloth",)):
    """Delete every face whose material is of `kinds` (the sample's clothes) from the body and hair."""
    import bmesh
    for role in ("body", "hair"):
        ob = vrm.parts.get(role)
        if ob is None:
            continue
        me = ob.data
        bm = bmesh.new()
        bm.from_mesh(me)
        dead = [f for f in bm.faces if kind_of(me.materials[f.material_index].name if me.materials[f.material_index] else "") in kinds]
        bmesh.ops.delete(bm, geom=dead, context="FACES")
        bm.to_mesh(me)
        bm.free()
        me.update()
        bpy.context.view_layer.objects.active = ob
        _prune_slots(ob)


# ------------------------------------------------------------------------------------------------ materials
def _srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _hex(h):
    h = h.lstrip("#")
    return _srgb_to_lin([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def lit_texture(m):
    """The MToon material's lit (base colour) texture, its factor (linear RGBA) and alpha mode, from the add-on's properties
    (fallback: the largest image of the node tree that is not a shader helper)."""
    img, fac, mode, cut = None, (1.0, 1.0, 1.0, 1.0), "OPAQUE", 0.5
    try:
        mt = m.vrm_addon_extension.mtoon1
        img = mt.pbr_metallic_roughness.base_color_texture.index.source
        fac = tuple(mt.pbr_metallic_roughness.base_color_factor)
        mode, cut = mt.alpha_mode, mt.alpha_cutoff
    except Exception:
        pass
    if img is None and m.node_tree:
        ims = [n.image for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image
               and not re.search(r"Shader_|Matcap|_nml|_spe|_out", n.image.name)]
        img = max(ims, key=lambda i: i.size[0] * i.size[1]) if ims else None
    return img, fac, mode, cut


def image_stats(img, mask_alpha=0.5):
    """Linear RGB of the image's opaque pixels: mean, and luminance percentiles (5, 50, 95)."""
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    px = px[px[:, 3] > mask_alpha]
    if len(px) == 0:
        return dict(mean=np.ones(3), p=(0.0, 0.5, 1.0))
    rgb = px[:, :3]
    if img.colorspace_settings.name.lower().startswith("srgb"):
        rgb = _srgb_to_lin(rgb)
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    return dict(mean=rgb.mean(0), p=tuple(float(v) for v in np.percentile(lum, (5, 50, 95))))


def _recolour(t, tex_col, img, spec):
    """The lit texture's colour recoloured (returns a colour socket):
    `tint` (hex): multiplied so the opaque pixels' mean becomes this colour (the painting's detail kept);
    `ramp` ((pos, hex), ...): the luminance, normalised to the image's 5..95 percentile, mapped onto these colours;
    `keep` (0..1): how much of the original stays (mixed back)."""
    if not spec:
        return tex_col
    st = image_stats(img)
    out = tex_col
    if "tint" in spec:
        k = _hex(spec["tint"]) / np.maximum(st["mean"], 1e-4)
        out = t.mix(1.0, out, (float(k[0]), float(k[1]), float(k[2]), 1.0), "MULTIPLY")
    if "ramp" in spec:
        bw = t.n("ShaderNodeRGBToBW")
        t.put(bw.inputs[0], tex_col)
        lo, _, hi = st["p"]
        lum = t.ramp(bw.outputs[0], lo, max(hi, lo + 1e-3))
        node = t.n("ShaderNodeValToRGB")
        cr = node.color_ramp
        cr.interpolation = spec.get("interp", "LINEAR")
        stops = spec["ramp"]
        cr.elements[0].position, cr.elements[0].color = stops[0][0], (*_hex(stops[0][1]), 1.0)
        cr.elements[1].position, cr.elements[1].color = stops[-1][0], (*_hex(stops[-1][1]), 1.0)
        for pos, col in stops[1:-1]:
            e = cr.elements.new(pos)
            e.color = (*_hex(col), 1.0)
        t.put(node.inputs[0], lum)
        out = node.outputs[0]
    if spec.get("keep"):
        out = t.mix(spec["keep"], out, tex_col)
    return out


def principled(name, img, fac=(1.0, 1.0, 1.0, 1.0), alpha_mode="OPAQUE", cutoff=0.5, recolour=None):
    """A Principled material on the lit texture (what `toonlook.convert` rebuilds): Base Color = texture x factor
    (recoloured), Alpha = its alpha for MASK / BLEND."""
    import studio
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    t = studio.NT(m)
    tex = t.n("ShaderNodeTexImage")
    tex.image = img
    col = tex.outputs["Color"] if img is not None else (1.0, 1.0, 1.0, 1.0)
    if img is not None and tuple(fac[:3]) != (1.0, 1.0, 1.0):
        col = t.mix(1.0, col, (fac[0], fac[1], fac[2], 1.0), "MULTIPLY")
    if img is not None:
        col = _recolour(t, col, img, recolour)
    t.put(t.bsdf.inputs["Base Color"], col)
    t.bsdf.inputs["Roughness"].default_value = 0.6
    if alpha_mode in ("MASK", "BLEND") and img is not None:
        a = tex.outputs["Alpha"]
        if fac[3] != 1.0:
            a = t.math("MULTIPLY", a, fac[3])
        if alpha_mode == "MASK":
            a = t.math("GREATER_THAN", a, cutoff)
        t.put(t.bsdf.inputs["Alpha"], a)
        m.surface_render_method = "DITHERED"
    m["vroid_alpha"] = alpha_mode
    return m


def materials(vrm, prefix, recolour=None):
    """Every MToon material of the VRM rebuilt as `principled` and named `<prefix>_<kind>` (toonlook finds its spec by the
    kind). `recolour` = {kind: spec} (see `_recolour`). Returns {kind: [materials]}."""
    made, done = {}, {}
    for ob in vrm.meshes():
        for slot in ob.material_slots:
            m = slot.material
            if m is None:
                continue
            if m.name in done:
                slot.material = done[m.name]
                continue
            k = kind_of(m.name) or "other"
            img, fac, mode, cut = lit_texture(m)
            new = principled("%s_%s" % (prefix, k), img, fac, mode, cut, (recolour or {}).get(k))
            new["vroid_source"] = m.name
            done[m.name] = new
            slot.material = new
            made.setdefault(k, []).append(new)
    return made


# ------------------------------------------------------------------------------------------------ proportions and pose
def pose_bones(vrm, rot=None, scale=None, isolate=True):
    """Pose the armature: `rot` = {bone: (x, y, z) degrees, XYZ Euler in the bone's own rest frame}, `scale` = {bone:
    (x, y, z)} (y along the bone). With `isolate`, a scaled bone's children do not inherit its scale (they move with its
    tail but keep their own size), so a longer thigh does not make a bigger foot."""
    pb = vrm.arm.pose.bones
    for name, s in (scale or {}).items():
        b = pb.get(name) or pb.get("J_Bip_" + name)
        b.scale = s
        if isolate:
            for c in b.children:
                c.bone.inherit_scale = "NONE"
    for name, r in (rot or {}).items():
        b = pb.get(name) or pb.get("J_Bip_" + name)
        b.rotation_mode = "XYZ"
        b.rotation_euler = tuple(math.radians(v) for v in r)
    bpy.context.view_layer.update()


def world_pose(vrm, rot):
    """Pose bones by WORLD-axis rotations (degrees about world X, Y, Z, applied in that order, as `suitpose` reads a pose):
    each bone's rotation is converted into its own frame, parents first."""
    from mathutils import Euler, Matrix
    arm = vrm.arm
    pb = arm.pose.bones
    order = [b for b in pb if any(b.name in (k, "J_Bip_" + k) for k in rot)]
    order.sort(key=lambda b: len(b.parent_recursive))
    for b in order:
        key = next(k for k in rot if b.name in (k, "J_Bip_" + k))
        rx, ry, rz = (math.radians(v) for v in rot[key])
        R = (Euler((rx, ry, rz), "XYZ").to_matrix()).to_4x4()
        bpy.context.view_layer.update()
        M = arm.matrix_world @ b.matrix          # the bone's current world matrix
        head = M.translation.copy()
        Mn = Matrix.Translation(head) @ R @ Matrix.Translation(-head) @ M
        b.matrix = arm.matrix_world.inverted() @ Mn
    bpy.context.view_layer.update()


def evaluated_verts(ob):
    """World-space vertices of the evaluated (posed) object."""
    dg = bpy.context.evaluated_depsgraph_get()
    e = ob.evaluated_get(dg)
    me = e.to_mesh()
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3) @ np.asarray(ob.matrix_world)[:3, :3].T + np.asarray(ob.matrix_world)[:3, 3]
    e.to_mesh_clear()
    return V


def rest_verts(ob):
    V = np.empty(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", V)
    M = np.asarray(ob.matrix_world)
    return V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]


def set_attr(ob, name, values, kind="FLOAT"):
    me = ob.data
    a = me.attributes.get(name) or me.attributes.new(name, kind, "POINT")
    a.data.foreach_set("vector" if kind == "FLOAT_VECTOR" else "value", np.asarray(values, np.float32).ravel())


def subdivide(ob, levels=1):
    """Apply a Catmull-Clark subdivision to the REST mesh (before the armature), vertex groups and attributes interpolated:
    a VRoid body is ~7 k vertices and reads faceted in a close shot."""
    md = ob.modifiers.new("Subdiv", "SUBSURF")
    md.levels = md.render_levels = levels
    md.uv_smooth = "PRESERVE_BOUNDARIES"
    bpy.context.view_layer.objects.active = ob
    while ob.modifiers[0].name != md.name:
        bpy.ops.object.modifier_move_up(modifier=md.name)
    bpy.ops.object.modifier_apply(modifier=md.name)


def shell(vrm, name, mat, offset, fields=None):
    """A skin-tight layer over the body: a copy of the body's skin faces pushed `offset` metres out along the normals,
    rigged as the body (its vertex groups and armature modifier), material `mat`. `fields` = {attribute: f(rest verts,
    rest normals) -> per-vertex values}, written as point attributes for the material to cut by (signed distances)."""
    body = vrm.parts["body"]
    me = body.data.copy()
    ob = bpy.data.objects.new(name, me)
    for c in body.users_collection:
        c.objects.link(ob)
    ob.matrix_world = body.matrix_world
    for md in body.modifiers:
        if md.type == "ARMATURE":
            a = ob.modifiers.new("Armature", "ARMATURE")
            a.object = md.object
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(me)
    dead = [f for f in bm.faces if not _is_skin(me, f.material_index)]
    bmesh.ops.delete(bm, geom=dead, context="FACES")
    bm.to_mesh(me)
    bm.free()
    me.materials.clear()
    me.materials.append(mat)
    for p in me.polygons:
        p.material_index = 0
    me.update()
    V = rest_verts(ob)
    N = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("normal", N)
    N = N.reshape(-1, 3)
    me.vertices.foreach_set("co", (V + N * offset - np.asarray(ob.matrix_world)[:3, 3]).ravel())
    me.update()
    for k, f in (fields or {}).items():
        set_attr(ob, k, f(V, N))
    vrm.parts["suit"] = ob
    return ob


def _is_skin(me, idx):
    m = me.materials[idx] if idx < len(me.materials) else None
    if m is None:
        return False
    src = m.get("vroid_source", m.name)
    return kind_of(src) == "skin" or m.name.split(".")[0].endswith("_skin")
