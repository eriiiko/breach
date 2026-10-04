"""Game-ready mesh from a scripted character (Blender 4.5, Cycles): one command, scripts -> outputs.

A character's `scripts/game.py` puts `charkit` on the path and calls `run(...)`:

    blender -b --factory-startup -P <character>/scripts/game.py -- [options]

    --tris N         triangle target of the skin (default 10000)
    --voxel M        voxel size of the remesh in metres (default 0.006)
    --close N        cavity closing radius in voxels: openings up to 2N voxels are sealed (default 2)
    --tex PX         texture size (default 1024)
    --angle DEG      Smart UV Project angle limit (default 66)
    --min-island N   UV islands under N faces merge into a neighbour (default 24)
    --cage M         bake cage extrusion in metres (default 0.012)
    --ray M          bake max ray distance in metres (default 0.04)
    --bake-samples N Cycles samples per bake (default 128)
    --samples N      Cycles samples of the preview renders (default 96)
    --no-previews    skip the preview renders (stats still measured)

Stages:
1. build the character at full resolution -- the bake source;
2. skin: apply + join copies (open sheets thickened inward first), voxel-remesh, fill the
   body's interior cavity in voxel space and remesh again -> ONE watertight outer surface;
   drop enclosed shells and crumbs; collapse-decimate (X-symmetric) to the target;
3. UVs: Smart UV Project, crumbs merged into neighbours, charts flattened angle-based,
   folded / crushed / self-overlapping charts re-cut, packed;
4. bakes (Cycles, selected-to-active, from ONE joined copy of the source whose closed parts
   have outward normals): tangent normal, ambient occlusion, base colour as emission, gloss (from
   the materials' roughness, `gloss_signal`) as emission; missed texels inpainted -> albedo.png
   (sRGB colour x AO in RGB, the gloss mask in ALPHA -- the unit shader's contract, #33) +
   normal.png (linear);
5. measure (stats.json: mesh, UVs, bake coverage, silhouette IoU against the source) and
   preview (textured turnaround, top view, source-vs-game side by side);
6. export a static .glb (Y-up, metres, feet at the origin) + a .blend of the skin;
7. rig (`rig.py`, when the character's game.py passes a rig spec): the skin bound to the game's
   53-bone skeleton with its 46 clips, exported as the game asset. Starts from the saved .blend:

    --rig-only       skip stages 1-6, rig the saved game/<name>_game.blend
    --no-rig         stop after stage 6
    --abduct DEG     extra upper-arm abduction for the clips (overrides the spec's ABDUCTION_DEG)

A change to the materials alone (a colour, the gloss mask) need not rebuild anything:

    --retexture      re-bake the albedo (colour x AO, gloss in alpha) onto the game asset's OWN
                     mesh and UVs and splice it into the .glb (`glbtex.py`); the mesh, skin and
                     clips stay byte for byte (checked)
    --variant NAME   a look variant's albedo on the same skin (the saved .blend, else the asset)

Outputs under <character>/game/ (the .glb, the textures and the .blend are
regenerated, so gitignored; stats.json is tracked) and <character>/previews/game_*.
"""
import argparse
import json
import math
import os
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

import kit
import studio

SILHOUETTE_VIEWS = (("front", 0.0, 560), ("side", -90.0, 440), ("back", 180.0, 560))


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tris", type=int, default=10000)
    ap.add_argument("--voxel", type=float, default=0.006)
    ap.add_argument("--close", type=int, default=2)
    ap.add_argument("--tex", type=int, default=1024)
    ap.add_argument("--angle", type=float, default=66.0)
    ap.add_argument("--min-island", type=int, default=24)
    ap.add_argument("--cage", type=float, default=0.012)
    ap.add_argument("--ray", type=float, default=0.04)
    ap.add_argument("--bake-samples", type=int, default=128)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--no-previews", action="store_true")
    ap.add_argument("--rig-only", action="store_true", help="skip stages 1-6: rig the saved game/<name>_game.blend")
    ap.add_argument("--no-rig", action="store_true", help="stop after the static export")
    ap.add_argument("--abduct", type=float, default=None, help="extra upper-arm abduction (deg), overrides the spec")
    ap.add_argument("--variant", default="", help="re-bake this look variant's albedo onto the saved skin (same mesh and "
                                                  "UVs) as game/albedo_<variant>.png; needs run(variants=...)")
    ap.add_argument("--retexture", action="store_true", help="re-bake the game asset's albedo (gloss in alpha) onto its "
                                                             "own mesh and UVs and splice it into the .glb; mesh, skin "
                                                             "and clips stay byte for byte")
    return ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


# ---------------------------------------------------------------------- helpers
def _select(objs, active):
    vl = bpy.context.view_layer
    for ob in bpy.context.scene.objects:
        ob.select_set(False)
    for ob in objs:
        ob.select_set(True)
    vl.objects.active = active


def _tris(ob):
    ob.data.calc_loop_triangles()
    return len(ob.data.loop_triangles)


def _apply_modifier(ob, mod):
    with bpy.context.temp_override(object=ob, active_object=ob):
        bpy.ops.object.modifier_apply(modifier=mod.name)


def _edit(ob, fn):
    _select([ob], ob)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    fn()
    bpy.ops.object.mode_set(mode="OBJECT")


def _pixels(img):
    w, h = img.size
    buf = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(buf)
    return buf.reshape(h, w, 4)  # row 0 = BOTTOM (Blender order)


def _save_png(arr, path, colorspace, alpha=False):
    """arr in Blender row order (row 0 = bottom), values already in the file's encoding. With `alpha`
    the PNG keeps arr's 4th channel as straight alpha (RGBA); else RGB."""
    h, w = arr.shape[:2]
    img = bpy.data.images.new("_save", w, h, alpha=alpha)
    img.colorspace_settings.name = colorspace
    if alpha:
        img.alpha_mode = "STRAIGHT"
    img.pixels.foreach_set(np.ascontiguousarray(arr, np.float32).ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)


# ------------------------------------------------------------------------- skin
def _thicken_thin_walls(sources, wall, log, walls=()):
    """Solidify walls thinner than `wall` (m) -- a cloth shell 2.5 mm thick is lost in a 6 mm level set
    -- made `wall` thick, still grown INWARD (`kit._finish`'s offset -1), so the silhouette keeps.
    `walls` = ((name prefix, metres), ...) sets a part's wall outright (a trouser leg thick enough that
    the gap to the boot inside it closes). Returns the (modifier, thickness) pairs to restore."""
    saved = []
    for ob in sources:
        w = next((x for pre, x in walls if ob.name.startswith(pre)), wall or 0.0)
        for m in ob.modifiers:
            if m.type == "SOLIDIFY" and m.offset <= -0.999 and 0.0 < m.thickness < w:
                saved.append((m, m.thickness))
                m.thickness = w
    log["thin_walls_thickened"] = len(saved)
    return saved


def make_skin(sources, voxel, tris_target, close_r, log, min_wall=None, inset=(), walls=(), fine=None):
    """Apply + join copies of the sources, voxel-remesh, drop enclosed shells, decimate. With `min_wall`
    (m), thinner inward solidified walls are thickened to it for the skin only (the bake source keeps them).
    `inset` = ((name prefix, metres), ...): those parts move inward along their normals by that much in the
    skin only (a palm sunk inside its own fine shell). `fine` = dict(prefixes, voxel, tris, keep=()):
    parts named so are remeshed apart at the finer voxel into their own shells, decimated to their own
    `tris` and joined to the skin (bare fingers 3-4 mm apart fuse into a mitten at 6 mm); of them,
    only the `keep` prefixes also enter the main skin (a palm that closes the sleeve's cuff)."""
    saved = _thicken_thin_walls(sources, min_wall, log, walls) if (min_wall or walls) else []
    dg = bpy.context.evaluated_depsgraph_get()
    parts, thickened, fine_parts = [], 0, []
    for ob in sources:
        if fine and ob.name.startswith(tuple(fine["prefixes"])):
            fm = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
            fm.transform(ob.matrix_world)
            fm.materials.clear()
            fp = bpy.data.objects.new("_fine_part", fm)
            bpy.context.scene.collection.objects.link(fp)
            fine_parts.append(fp)
            if not ob.name.startswith(tuple(fine.get("keep", ()))):
                continue
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
        me.transform(ob.matrix_world)
        me.materials.clear()
        d = next((m for pre, m in inset if ob.name.startswith(pre)), 0.0)
        if d:
            co = np.empty(len(me.vertices) * 3, np.float32)
            nr = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", co)
            me.vertices.foreach_get("normal", nr)
            me.vertices.foreach_set("co", co - d * nr)
            me.update()
            log["inset_parts"] = log.get("inset_parts", 0) + 1
        cp = bpy.data.objects.new("_skin_part", me)
        bpy.context.scene.collection.objects.link(cp)
        parts.append(cp)
        # An open sheet (a cloth loft, an open tube end) is thinner than a voxel and the level set
        # loses it: give it a closed wall 1.5 voxels thick, grown INWARD so the silhouette keeps.
        side = _open_sheet_side(me)
        if side:
            sm = cp.modifiers.new("Solidify", "SOLIDIFY")
            sm.thickness, sm.offset, sm.use_rim, sm.use_even_offset = 1.5 * voxel, -side, True, True
            _apply_modifier(cp, sm)
            thickened += 1
    for m, th in saved:
        m.thickness = th
    log["open_parts_thickened"] = thickened
    fine_skin = _fine_shells(fine_parts, fine, log) if fine_parts else None
    if fine_skin is not None:
        tris_target -= _tris(fine_skin)
    skin = parts[0]
    with bpy.context.temp_override(active_object=skin, selected_editable_objects=parts, selected_objects=parts):
        bpy.ops.object.join()
    skin.name = skin.data.name = "GameSkin"
    log["joined_tris"] = _tris(skin)

    _remesh(skin, voxel)
    log["remeshed_tris"] = _tris(skin)
    skin = _drop_shells(skin, log)
    # Cavities: walls grown inward still leave the inside of the body as a second, inner surface,
    # joined to the outer one wherever a tube end or a gap opens into it. Close the solid in voxel
    # space, stuff the cavity with a block filler, remesh again: one outer surface remains.
    filler = cavity_filler(skin, voxel, close_r, log)
    if filler is not None:
        both = [skin, filler]
        with bpy.context.temp_override(active_object=skin, selected_editable_objects=both, selected_objects=both):
            bpy.ops.object.join()
        _remesh(skin, voxel)
        skin = _drop_shells(skin, log)
    log["filled_tris"] = _tris(skin)

    # Collapse-decimate to the target (two passes: the first ratio is a guess on a quad mesh).
    for _ in range(3):
        n = _tris(skin)
        if n <= tris_target:
            break
        dm = skin.modifiers.new("Decimate", "DECIMATE")
        dm.decimate_type, dm.ratio = "COLLAPSE", tris_target / n
        dm.use_symmetry, dm.symmetry_axis, dm.use_collapse_triangulate = True, "X", True
        _apply_modifier(skin, dm)
    if fine_skin is not None:
        both = [skin, fine_skin]
        with bpy.context.temp_override(active_object=skin, selected_editable_objects=both, selected_objects=both):
            bpy.ops.object.join()
        skin.name = skin.data.name = "GameSkin"
    me = skin.data
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    if me.has_custom_normals:
        with bpy.context.temp_override(object=skin, active_object=skin):
            bpy.ops.mesh.customdata_custom_splitnormals_clear()
    me.update()
    log["tris"] = _tris(skin)
    return skin


def _fine_shells(fine_parts, fine, log):
    """The `fine` parts joined, remeshed at their own voxel, crumbs dropped, decimated to their budget."""
    ob = fine_parts[0]
    with bpy.context.temp_override(active_object=ob, selected_editable_objects=fine_parts, selected_objects=fine_parts):
        bpy.ops.object.join()
    rm = ob.modifiers.new("Remesh", "REMESH")
    rm.mode, rm.voxel_size, rm.adaptivity, rm.use_smooth_shade = "VOXEL", fine["voxel"], 0.0, True
    _apply_modifier(ob, rm)
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    shells, seen = [], set()
    for f in bm.faces:  # connected pieces; a crumb under 24 faces goes
        if f.index in seen:
            continue
        stack, piece = [f], []
        seen.add(f.index)
        while stack:
            g = stack.pop()
            piece.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index)
                        stack.append(h)
        shells.append(piece)
    crumbs = [g for p in shells if len(p) < 24 for g in p]
    bmesh.ops.delete(bm, geom=crumbs, context="FACES")
    bm.to_mesh(ob.data)
    bm.free()
    log["fine_shells"] = sum(len(p) >= 24 for p in shells)
    log["fine_remeshed_tris"] = _tris(ob)
    for _ in range(3):
        n = _tris(ob)
        if n <= fine["tris"]:
            break
        dm = ob.modifiers.new("Decimate", "DECIMATE")
        dm.decimate_type, dm.ratio = "COLLAPSE", fine["tris"] / n
        dm.use_symmetry, dm.symmetry_axis, dm.use_collapse_triangulate = True, "X", True
        _apply_modifier(ob, dm)
    ob.name = ob.data.name = "_FineSkin"
    log["fine_tris"] = _tris(ob)
    return ob


def _remesh(ob, voxel):
    rm = ob.modifiers.new("Remesh", "REMESH")
    rm.mode, rm.voxel_size, rm.adaptivity, rm.use_smooth_shade = "VOXEL", voxel, 0.0, True
    _apply_modifier(ob, rm)
    ob.data.name = "GameSkin"


def _drop_shells(skin, log):
    """Split into connected pieces; drop a piece inside the largest one (hidden interior) or a crumb."""
    _edit(skin, lambda: bpy.ops.mesh.separate(type="LOOSE"))
    pieces = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.data.name.startswith("GameSkin")]
    pieces.sort(key=lambda o: len(o.data.polygons), reverse=True)
    main = pieces[0]
    bvh = BVHTree.FromPolygons([v.co.copy() for v in main.data.vertices], [p.vertices[:] for p in main.data.polygons])
    kept, dropped = [main], []
    for p in pieces[1:]:
        if len(p.data.polygons) < 24 or _inside(bvh, p.data.vertices[0].co):
            dropped.append(len(p.data.polygons))
            bpy.data.objects.remove(p)
        else:
            kept.append(p)
    log.setdefault("pieces_kept", []).append([len(o.data.polygons) for o in kept])
    log.setdefault("pieces_dropped", []).append(dropped)
    if len(kept) > 1:
        with bpy.context.temp_override(active_object=main, selected_editable_objects=kept, selected_objects=kept):
            bpy.ops.object.join()
    main.name = "GameSkin"
    main.data.name = "GameSkin"
    return main


def _dilate(m, r=1):
    for _ in range(r):
        d = m.copy()
        d[1:] |= m[:-1]
        d[:-1] |= m[1:]
        d[:, 1:] |= m[:, :-1]
        d[:, :-1] |= m[:, 1:]
        d[:, :, 1:] |= m[:, :, :-1]
        d[:, :, :-1] |= m[:, :, 1:]
        m = d
    return m


def _axis_visible(wall):
    """Voxels that see the grid border along at least one of the six axis directions without crossing `wall`.
    (A flood fill from outside would leak into the body through any tube end or gap; a straight sight line
    rarely finds such an opening.)"""
    vis = np.zeros_like(wall)
    for ax in range(3):
        vis |= ~np.logical_or.accumulate(wall, axis=ax)
        vis |= ~np.flip(np.logical_or.accumulate(np.flip(wall, ax), axis=ax), ax)
    return vis


def cavity_filler(skin, voxel, close_r, log):
    """Block mesh filling the skin's interior cavities (the solid closed by `close_r` voxels), or None."""
    me = skin.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    pad = (close_r + 3) * voxel
    lo = co.min(0) - pad
    dims = np.ceil((co.max(0) + pad - lo) / voxel).astype(int)
    bvh = BVHTree.FromPolygons([Vector(v) for v in co], [p.vertices[:] for p in me.polygons])
    solid = np.zeros(dims, bool)
    zc = lo[2] + (np.arange(dims[2]) + 0.5) * voxel
    up = Vector((0.0, 0.0, 1.0))
    for i in range(dims[0]):
        x = lo[0] + (i + 0.5 + 1e-3) * voxel
        for j in range(dims[1]):
            o = Vector((x, lo[1] + (j + 0.5 + 2e-3) * voxel, lo[2]))
            hits = []
            for _ in range(256):
                loc, _n, _i, _d = bvh.ray_cast(o, up)
                if loc is None:
                    break
                hits.append(loc.z)
                o = loc + up * 1e-6
            for a, b in zip(hits[0::2], hits[1::2]):
                solid[i, j] |= (zc > a) & (zc < b)
    outside = _dilate(_axis_visible(_dilate(solid, close_r)), close_r)
    cavity = ~outside & ~solid
    log["cavity_voxels"] = int(cavity.sum())
    if not cavity.any():
        return None
    fill = _dilate(cavity, 1) & ~outside
    return _block_mesh(fill, lo, voxel)


def _block_mesh(m, lo, voxel):
    """Closed quad surface of a voxel set (the faces between set and unset voxels)."""
    p = np.pad(m, 1)
    corner = []
    # face offsets per axis: the 4 corners of the face between voxel v and v+e_ax, in voxel-corner units
    for ax in range(3):
        u, w = [a for a in range(3) if a != ax]
        diff = np.diff(p.astype(np.int8), axis=ax)  # +1: unset->set going +ax, -1: set->unset
        for sign in (1, -1):
            idx = np.argwhere(diff == sign)  # index in padded grid of the lower voxel
            if not len(idx):
                continue
            base = idx.astype(np.int64)
            base[:, ax] += 1  # the face plane sits at corner index (lower voxel + 1)
            c = np.zeros((len(idx), 4, 3), np.int64)
            du, dw = np.zeros(3, np.int64), np.zeros(3, np.int64)
            du[u], dw[w] = 1, 1
            order = (0, du, du + dw, dw) if sign == -1 else (0, dw, du + dw, du)
            for k, off in enumerate(order):
                c[:, k] = base + off
            corner.append(c)
    c = np.concatenate(corner).reshape(-1, 3)
    uniq, inv = np.unique(c, axis=0, return_inverse=True)
    v = lo + (uniq - 1) * voxel
    f = inv.reshape(-1, 4)
    me = bpy.data.meshes.new("GameSkinFill")
    me.from_pydata(v.tolist(), [], f.tolist())
    me.update()
    ob = bpy.data.objects.new("GameSkinFill", me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def _open_sheet_side(me):
    """0 for a closed mesh; else +1 if its normals point away from its centroid (outward), -1 if inward."""
    ne = len(me.edges)
    if ne == 0:
        return 0
    eidx = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("edge_index", eidx)
    if not (np.bincount(eidx, minlength=ne) == 1).any():
        return 0
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    nrm = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("normal", nrm)
    d = (nrm.reshape(-1, 3) * (co - co.mean(0))).sum(1).mean()
    return 1 if d >= 0 else -1


def _inside(bvh, co):
    """Ray-parity test against a closed surface, majority over three directions."""
    votes = 0
    for d in (Vector((0.0, 0.0, 1.0)), Vector((1.0, 0.0, 0.0)), Vector((0.0, -1.0, 0.0))):
        o, hits = Vector(co), 0
        for _ in range(64):
            loc, _n, _i, _d = bvh.ray_cast(o, d)
            if loc is None:
                break
            hits += 1
            o = loc + d * 1e-5
        votes += hits % 2
    return votes >= 2


# --------------------------------------------------------------------------- UVs
RECUT_ROUNDS = 2  # Smart UV angle halves each round: 66 -> 33 -> 16.5 degrees


def unwrap(skin, angle_deg, tex, min_faces=24, log=None):
    """UV charts for the skin, kept few because every chart border duplicates vertices in the export.

    Smart UV Project cuts the surface into islands -- on a decimated voxel surface, hundreds of crumbs --
    and every island under `min_faces` faces is merged into the neighbour it shares the longest border
    with. The charts are flattened angle-based along those seams. A chart that folds or overlaps itself is
    cut again by Smart UV at half the angle (its own crumbs merged back inside it), for RECUT_ROUNDS rounds;
    then everything is packed. What still folds is counted in the stats (uv_folded_charts_left)."""
    margin = 6.0 / tex
    me = skin.data
    _edit(skin, lambda: bpy.ops.uv.smart_project(angle_limit=math.radians(angle_deg), island_margin=margin,
                                                 area_weight=0.0, correct_aspect=True, scale_to_bounds=False))
    lab = _merge_crumbs(me, _mesh_uv_islands(me), min_faces)
    smart = len(set(_mesh_uv_islands(me).tolist()))
    recut, angle, bad = 0, angle_deg, set()
    for rnd in range(RECUT_ROUNDS + 1):
        _set_seams(me, lab)
        _edit(skin, lambda: bpy.ops.uv.unwrap(method="ANGLE_BASED", fill_holes=True, correct_aspect=True,
                                              margin=margin))
        bad = _folded_charts(me, lab)
        print("unwrap: %d charts, %d fold or overlap (Smart UV angle %.1f)" % (lab.max() + 1, len(bad), angle),
              flush=True)
        if not bad or rnd == RECUT_ROUNDS:
            break
        sel = np.isin(lab, list(bad))
        angle *= 0.5
        recut += len(bad)
        me.polygons.foreach_set("select", sel)
        _select([skin], skin)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(angle), island_margin=margin, area_weight=0.0,
                                 correct_aspect=True, scale_to_bounds=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        sub = _mesh_uv_islands(me)
        # new labels inside the cut charts (a sub-island never spans two charts)
        key = np.where(sel, lab.max() + 1 + sub * (lab.max() + 1) + lab, lab)
        _, lab = np.unique(key, return_inverse=True)
        lab = _merge_crumbs(me, lab, min_faces, only=sel)  # the cut's crumbs merge back inside it
    if bad:  # last resort: those charts become their bare Smart UV islands at this angle, unmerged
        sel = np.isin(lab, list(bad))
        me.polygons.foreach_set("select", sel)
        _select([skin], skin)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(angle), island_margin=margin, area_weight=0.0,
                                 correct_aspect=True, scale_to_bounds=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        sub = _mesh_uv_islands(me)
        key = np.where(sel, lab.max() + 1 + sub * (lab.max() + 1) + lab, lab)
        _, lab = np.unique(key, return_inverse=True)
        _set_seams(me, lab)
        _edit(skin, lambda: bpy.ops.uv.unwrap(method="ANGLE_BASED", fill_holes=True, correct_aspect=True,
                                              margin=margin))
        bad = _folded_charts(me, lab)
        print("unwrap: %d charts after the bare-island fallback, %d still fold, crush or overlap"
              % (lab.max() + 1, len(bad)), flush=True)
    if log is not None:
        log["smart_uv_islands"] = smart
        log["charts_recut_for_folding"] = recut
        log["folded_charts_left"] = len(bad)

    def pack():
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(rotate=True, margin_method="FRACTION", margin=margin, shape_method="CONCAVE")

    _edit(skin, pack)


def _mesh_uv_islands(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    lab = _uv_islands(bm)
    bm.free()
    return lab


def _merge_crumbs(me, lab, min_faces, only=None):
    """Merge each label under `min_faces` faces into the neighbour label it shares the most edges with.
    With `only` (a face mask), only labels inside the mask merge, and only with labels inside it."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    pairs = []
    for e in bm.edges:
        if len(e.link_faces) == 2:
            fa, fb = (f.index for f in e.link_faces)
            if only is None or (only[fa] and only[fb]):
                pairs.append((fa, fb))
    bm.free()
    pairs = np.array(pairs, np.int64).reshape(-1, 2)
    lab = lab.copy()
    for _ in range(12):
        size = np.bincount(lab)
        la, lb = lab[pairs[:, 0]], lab[pairs[:, 1]]
        cross = la != lb
        border = {}
        for a, b in zip(la[cross].tolist(), lb[cross].tolist()):
            border[(a, b)] = border.get((a, b), 0) + 1
            border[(b, a)] = border.get((b, a), 0) + 1
        best = {}
        for (a, b), n in border.items():
            if size[a] < min_faces and (a not in best or (n, size[b]) > (best[a][1], size[best[a][0]])):
                best[a] = (b, n)
        if not best:
            break
        remap = np.arange(lab.max() + 1)
        for a, (b, _n) in best.items():
            if not (b in best and best[b][0] == a and b < a):  # two crumbs choosing each other: merge once
                remap[a] = b
        _, lab = np.unique(remap[lab], return_inverse=True)
    return lab


def _set_seams(me, lab):
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    for e in bm.edges:
        e.seam = len(e.link_faces) == 2 and lab[e.link_faces[0].index] != lab[e.link_faces[1].index]
    bm.to_mesh(me)
    bm.free()


CRUSH_MIN_AREA = 1e-4  # m^2: one square centimetre


def _folded_charts(me, lab, res=1024):
    """Charts that fold (a UV face wound against the chart's majority), crush part of themselves, or overlap
    themselves (two of the chart's triangles strictly covering one texel centre at `res`)."""
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    areas, dens = {}, {}
    for p in me.polygons:
        q = uv[list(p.loop_indices)]
        a = 0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
        areas.setdefault(int(lab[p.index]), []).append(a)
        dens.setdefault(int(lab[p.index]), []).append((abs(a) / max(p.area, 1e-12), p.area))
    bad = {c for c, a in areas.items() if min(sum(x > 0 for x in a), sum(x < 0 for x in a)) > 0}
    # a chart that crushes part of itself (a sizeable face at under a tenth of the chart's texel density)
    for c, d in dens.items():
        d = np.array(d)
        if ((d[:, 0] < 0.1 * np.median(d[:, 0])) & (d[:, 1] > CRUSH_MIN_AREA)).any():
            bad.add(c)
    me.calc_loop_triangles()
    owner = -np.ones((res, res), np.int64)
    hits = {}
    for lt in me.loop_triangles:
        c = int(lab[lt.polygon_index])
        sub, inside = _raster_tri(uv[list(lt.loops)] * res - 0.5, owner, strict=True)
        if sub is None:
            continue
        taken = sub[inside]
        taken = taken[taken >= 0]
        if taken.size and (lab[taken] == c).any():
            hits[c] = hits.get(c, 0) + int((lab[taken] == c).sum())
        sub[inside & (sub < 0)] = lt.polygon_index
    return bad | {c for c, n in hits.items() if n >= 2}


def _raster_tri(t, grid, strict=False):
    """(view of `grid` over the triangle's box, mask of texel centres inside it) or (None, None)."""
    a, b, c = t
    res = grid.shape[0]
    x0, x1 = int(max(0, math.floor(min(a[0], b[0], c[0])))), int(min(res - 1, math.ceil(max(a[0], b[0], c[0]))))
    y0, y1 = int(max(0, math.floor(min(a[1], b[1], c[1])))), int(min(res - 1, math.ceil(max(a[1], b[1], c[1]))))
    d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if x1 < x0 or y1 < y0 or abs(d) < 1e-12:
        return None, None
    X, Y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    l1 = ((b[1] - c[1]) * (X - c[0]) + (c[0] - b[0]) * (Y - c[1])) / d
    l2 = ((c[1] - a[1]) * (X - c[0]) + (a[0] - c[0]) * (Y - c[1])) / d
    e = 1e-4 if strict else 0.0
    inside = (l1 > e) & (l2 > e) & (l1 + l2 < 1 - e) if strict else (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
    return grid[y0:y1 + 1, x0:x1 + 1], inside


def _uv_islands(bm):
    """Face -> UV island label (faces joined across an edge whose two UV ends agree)."""
    uv = bm.loops.layers.uv.active
    parent = list(range(len(bm.faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def uv_at(face, vert):
        for lp in face.loops:
            if lp.vert is vert:
                return lp[uv].uv
        return None

    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        fa, fb = e.link_faces
        if all((uv_at(fa, v) - uv_at(fb, v)).length < 1e-6 for v in e.verts):
            ra, rb = find(fa.index), find(fb.index)
            if ra != rb:
                parent[ra] = rb
    roots = [find(f.index) for f in bm.faces]
    _, lab = np.unique(roots, return_inverse=True)
    return lab


def mesh_measure(skin):
    """Triangles, vertices, UV islands, watertightness, UV range, and the vertex count after UV splitting."""
    bm = bmesh.new()
    bm.from_mesh(skin.data)
    uv = bm.loops.layers.uv.active
    bm.faces.ensure_lookup_table()
    boundary = sum(len(e.link_faces) == 1 for e in bm.edges)
    nonmanifold = sum(len(e.link_faces) > 2 for e in bm.edges)
    islands = len(set(_uv_islands(bm).tolist()))
    uvs = np.array([lp[uv].uv[:] for f in bm.faces for lp in f.loops])
    split = {(lp.vert.index, round(lp[uv].uv.x, 6), round(lp[uv].uv.y, 6)) for f in bm.faces for lp in f.loops}
    out = dict(triangles=sum(len(f.verts) - 2 for f in bm.faces), blender_vertices=len(bm.verts),
               vertices_after_uv_split=len(split), uv_islands=islands, boundary_edges=boundary,
               non_manifold_edges=nonmanifold, uv_min=[round(float(x), 5) for x in uvs.min(0)],
               uv_max=[round(float(x), 5) for x in uvs.max(0)],
               uv_outside_0_1=int(((uvs < 0) | (uvs > 1)).any(1).sum()),
               zero_area_uv_faces=int(sum(f.calc_area() > 0 and _uv_area(f, uv) < 1e-12 for f in bm.faces)))
    bm.free()
    return out


def hidden_area_fraction(skin):
    """Share of surface area that sees open sky in none of 14 directions (a conservative interior test:
    a deep crevice also counts)."""
    me = skin.data
    bvh = BVHTree.FromPolygons([v.co.copy() for v in me.vertices], [p.vertices[:] for p in me.polygons])
    dirs = [Vector(d).normalized() for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] +
            [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]]
    hidden = total = 0.0
    for p in me.polygons:
        total += p.area
        if not any(d.dot(p.normal) > 0.05 and bvh.ray_cast(p.center + p.normal * 1e-4, d)[0] is None for d in dirs):
            hidden += p.area
    return round(hidden / max(total, 1e-12), 4)


def _uv_area(face, uv):
    p = [lp[uv].uv for lp in face.loops]
    return abs(sum((p[i].x * p[(i + 1) % len(p)].y - p[(i + 1) % len(p)].x * p[i].y) for i in range(len(p)))) / 2


def uv_coverage(skin, tex):
    """(coverage mask, overlap mask) at `tex`, Blender row order: texel centres inside any UV triangle, and
    texel centres strictly inside two (shared edges do not count)."""
    me = skin.data
    me.calc_loop_triangles()
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2) * tex - 0.5
    cover = np.zeros((tex, tex), bool)
    count = np.zeros((tex, tex), np.int16)
    for lt in me.loop_triangles:
        t = uv[list(lt.loops)]
        sub, inside = _raster_tri(t, cover)
        if sub is not None:
            sub |= inside
        sub, inside = _raster_tri(t, count, strict=True)
        if sub is not None:
            sub += inside
    return cover, count > 1


# -------------------------------------------------------------------------- bake
# Gloss (#33): the game's unit shader draws a highlight scaled and tightened by a per-texel gloss
# value, baked into the albedo's ALPHA from the materials' own roughness: 1 at or below
# GLOSS_ROUGH_FULL, 0 at or above GLOSS_ROUGH_MATTE, falling as the cube between -- so a visor
# (0.045) bakes ~1, moulded armour (0.42-0.66) a little, cloth (0.84+) and rubber 0. A clear coat
# counts by its weight: gloss = max(g(Roughness), Coat Weight * g(Coat Roughness)).
GLOSS_ROUGH_FULL, GLOSS_ROUGH_MATTE, GLOSS_POWER = 0.05, 0.85, 3.0


def _value(t, sock):
    """The signal into `sock`: its link's source socket, or its own default value."""
    return sock.links[0].from_socket if sock.is_linked else sock.default_value


def gloss_signal(t, bsdf):
    """A float socket carrying the gloss of a Principled BSDF (see GLOSS_ROUGH_*)."""

    def g(rough):
        mr = t.nodes.new("ShaderNodeMapRange")
        mr.clamp = True
        mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = GLOSS_ROUGH_MATTE, GLOSS_ROUGH_FULL
        mr.inputs["To Min"].default_value, mr.inputs["To Max"].default_value = 0.0, 1.0
        v = _value(t, rough)
        if isinstance(v, bpy.types.NodeSocket):
            t.links.new(v, mr.inputs["Value"])
        else:
            mr.inputs["Value"].default_value = v
        pw = t.nodes.new("ShaderNodeMath")
        pw.operation = "POWER"
        t.links.new(mr.outputs[0], pw.inputs[0])
        pw.inputs[1].default_value = GLOSS_POWER
        return pw.outputs[0]

    coat = t.nodes.new("ShaderNodeMath")
    coat.operation = "MULTIPLY"
    w = _value(t, bsdf.inputs["Coat Weight"])
    if isinstance(w, bpy.types.NodeSocket):
        t.links.new(w, coat.inputs[0])
    else:
        coat.inputs[0].default_value = w
    t.links.new(g(bsdf.inputs["Coat Roughness"]), coat.inputs[1])
    mx = t.nodes.new("ShaderNodeMath")
    mx.operation = "MAXIMUM"
    t.links.new(g(bsdf.inputs["Roughness"]), mx.inputs[0])
    t.links.new(coat.outputs[0], mx.inputs[1])
    return mx.outputs[0]


class EmissionSwap:
    """Temporarily re-wire every source material to emit its Base Color signal, plain white, or (gloss)
    its gloss as grey (`gloss_signal`)."""

    def __init__(self, mats, white=False, gloss=False):
        self.saved = []
        for m in mats:
            t = m.node_tree
            out = next(n for n in t.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
            bsdf = next(n for n in t.nodes if n.type == "BSDF_PRINCIPLED")
            old = out.inputs["Surface"].links[0].from_socket if out.inputs["Surface"].is_linked else None
            before = set(t.nodes)
            em = t.nodes.new("ShaderNodeEmission")
            em.inputs["Strength"].default_value = 1.0
            bc = bsdf.inputs["Base Color"]
            if white:
                em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            elif gloss:
                t.links.new(gloss_signal(t, bsdf), em.inputs["Color"])
            elif bc.is_linked:
                t.links.new(bc.links[0].from_socket, em.inputs["Color"])
            else:
                em.inputs["Color"].default_value = bc.default_value
            t.links.new(em.outputs[0], out.inputs["Surface"])
            self.saved.append((t, out, old, [n for n in t.nodes if n not in before]))

    def restore(self):
        for t, out, old, added in self.saved:
            for n in added:
                t.nodes.remove(n)
            if old is not None:
                t.links.new(old, out.inputs["Surface"])


def bake_source(sources, log):
    """Sources applied and joined into one object, materials and attributes kept. A closed part whose
    faces point inward is turned outward: Cycles renders both sides alike, so the model never showed it,
    but a bake reads the normal's side -- inward faces bake black occlusion and an inverted normal map."""
    dg = bpy.context.evaluated_depsgraph_get()
    parts, flipped = [], []
    for ob in sources:
        me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        me.transform(ob.matrix_world)
        if _open_sheet_side(me) == 0 and len(me.polygons):
            bm = bmesh.new()
            bm.from_mesh(me)
            before = np.array([f.normal.copy() for f in bm.faces])
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            bm.normal_update()
            after = np.array([f.normal.copy() for f in bm.faces])
            n_flip = int(((before * after).sum(1) < 0).sum())
            if n_flip:
                bm.to_mesh(me)
                flipped.append("%s:%d/%d" % (ob.name, n_flip, len(before)))
            bm.free()
        if me.has_custom_normals:
            log.setdefault("bake_source_custom_normals", []).append(ob.name)
        cp = bpy.data.objects.new("_bake_src", me)
        bpy.context.scene.collection.objects.link(cp)
        parts.append(cp)
    with bpy.context.temp_override(active_object=parts[0], selected_editable_objects=parts, selected_objects=parts):
        bpy.ops.object.join()
    log["bake_source_faces_flipped"] = flipped
    me = parts[0].data
    log["bake_source_material_slots"] = len(me.materials)
    log["bake_source_attributes"] = sorted(a.name for a in me.attributes if not a.name.startswith((".", "sharp_")))
    return parts[0]


def bake_all(skin, sources, tex, cage, ray, samples, timing, log):
    sc = bpy.context.scene
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.world.light_settings.distance = 0.15  # AO bake reach (metres)
    # The skin is the bake target, not an occluder.
    for k in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter",
              "visible_shadow"):
        setattr(skin, k, False)

    mat = bpy.data.materials.new("_bake_target")
    mat.use_nodes = True
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    mat.node_tree.nodes.active = node
    skin.data.materials.clear()
    skin.data.materials.append(mat)
    # Bake from ONE joined copy: Cycles re-syncs the whole scene once per selected object (106 syncs
    # per bake otherwise), and the copy is where closed parts get their normals turned outward.
    src = bake_source(sources, log)
    for ob in sources:
        ob.hide_render = True
    _select([src, skin], skin)
    bk = sc.render.bake
    bk.use_selected_to_active, bk.cage_extrusion, bk.max_ray_distance = True, cage, ray
    bk.margin, bk.margin_type, bk.use_clear = 16, "EXTEND", True
    bk.normal_space = "TANGENT"
    src_mats = [m for m in src.data.materials if m]

    def bake(kind, noncolor):
        img = bpy.data.images.new("_bake_" + kind.lower(), tex, tex, alpha=False, float_buffer=True)
        if noncolor:
            img.colorspace_settings.name = "Non-Color"
        node.image = img
        t0 = time.time()
        bpy.ops.object.bake(type=kind, use_selected_to_active=True, cage_extrusion=cage, max_ray_distance=ray,
                            margin=16, margin_type="EXTEND", use_clear=True, target="IMAGE_TEXTURES")
        timing["bake_%s_s" % kind.lower()] = round(time.time() - t0, 1)
        px = _pixels(img)
        bpy.data.images.remove(img)
        return px

    normal = bake("NORMAL", True)  # original materials: the shader bump is in the shading normal
    ao = bake("AO", True)
    swap = EmissionSwap(src_mats)
    color = bake("EMIT", False)
    swap.restore()
    t_color = timing["bake_emit_s"]
    swap = EmissionSwap(src_mats, gloss=True)  # #33: the gloss mask, from the materials' roughness
    gloss = bake("EMIT", True)[..., :1]
    timing["bake_gloss_s"], timing["bake_emit_s"] = timing["bake_emit_s"], t_color
    swap.restore()
    # Coverage check: everything emits white; a texel inside a UV triangle that stays dark was missed.
    swap = EmissionSwap(src_mats, white=True)
    img = bpy.data.images.new("_bake_hit", tex, tex, alpha=False, float_buffer=True)
    node.image = img
    bpy.ops.object.bake(type="EMIT", use_selected_to_active=True, cage_extrusion=cage, max_ray_distance=ray, margin=0,
                        use_clear=True, target="IMAGE_TEXTURES")
    hit = _pixels(img)[..., 0]
    bpy.data.images.remove(img)
    swap.restore()
    skin.data.materials.clear()
    bpy.data.materials.remove(mat)
    bpy.data.objects.remove(src)
    for ob in sources:
        ob.hide_render = False
    for k in ("visible_camera", "visible_diffuse", "visible_glossy", "visible_transmission", "visible_volume_scatter",
              "visible_shadow"):
        setattr(skin, k, True)
    return normal, ao, color, gloss, hit


def albedo_rgba(color, ao, gloss):
    """The game albedo: sRGB colour x AO in RGB, the gloss mask (linear, 0 matte .. 1 glossiest) in
    ALPHA -- the unit shader's contract (#33): the model is drawn opaque, alpha is gloss."""
    albedo = np.clip(color[..., :3] * ao[..., :1], 0.0, 1.0)
    return np.dstack([studio.to_srgb(albedo), np.clip(gloss[..., :1], 0.0, 1.0)])


def _inpaint(img, valid, region, steps=64):
    """Fill texels of `region` that are not `valid` from their valid neighbours, ring by ring."""
    img = img.copy()
    todo = region & ~valid
    ok = valid.copy()
    for _ in range(steps):
        if not todo.any():
            break
        acc = np.zeros(img.shape, np.float32)
        n = np.zeros(img.shape[:2], np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh_ok = np.roll(ok, (dy, dx), (0, 1))
            acc += np.roll(img, (dy, dx), (0, 1)) * sh_ok[..., None]
            n += sh_ok
        grow = todo & (n > 0)
        img[grow] = acc[grow] / n[grow][:, None]
        ok |= grow
        todo &= ~grow
    return img


def _splice_albedo(glb, albedo_path):
    """Put the albedo PNG into the exported .glb byte for byte: the exporter's own encoding of an
    image whose alpha no socket reads is not ours to rely on, and the alpha is the gloss mask (#33)."""
    import glbtex
    with open(albedo_path, "rb") as f:
        glbtex.replace_image(glb, "baseColor", f.read())


def game_material(albedo_path, normal_path):
    m = bpy.data.materials.new("game_skin")
    m.use_nodes = True
    t = m.node_tree
    bsdf = t.nodes["Principled BSDF"]
    a = t.nodes.new("ShaderNodeTexImage")
    a.image = bpy.data.images.load(albedo_path)
    a.image.colorspace_settings.name = "sRGB"
    n = t.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(normal_path)
    n.image.colorspace_settings.name = "Non-Color"
    nm = t.nodes.new("ShaderNodeNormalMap")
    t.links.new(a.outputs["Color"], bsdf.inputs["Base Color"])
    t.links.new(n.outputs["Color"], nm.inputs["Color"])
    t.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.6
    return m


# ---------------------------------------------------------------- measure / show
def silhouettes(rig, cam, floor, sources, skin, out_dir, height):
    """IoU of the skin's orthographic mask against the source's, per view."""
    sc = bpy.context.scene
    keep = sc.cycles.samples, sc.cycles.use_denoising
    sc.cycles.samples, sc.cycles.use_denoising = 4, False
    m_per_px = height * 1.12 / 1100

    def masks():
        p = studio.ortho_panels(rig, cam, floor, out_dir, SILHOUETTE_VIEWS, m_per_px, 1100, height * 0.5, "_mask")
        return {k: v[..., 3] > 0.5 for k, v in p.items()}

    _show(sources, skin, source=True)
    src = masks()
    _show(sources, skin, source=False)
    game = masks()
    _show(sources, skin, source=True, both=True)
    sc.cycles.samples, sc.cycles.use_denoising = keep
    return {k: round(float((src[k] & game[k]).sum() / max(1, (src[k] | game[k]).sum())), 4) for k in src}


def _show(sources, skin, source, both=False):
    for ob in sources:
        ob.hide_render = not (source or both)
    skin.hide_render = source and not both


def _save_jpg(arr_top_first, path):
    img = bpy.data.images.new("_jpg", arr_top_first.shape[1], arr_top_first.shape[0], alpha=False)
    rgba = np.dstack([arr_top_first[..., :3], np.ones(arr_top_first.shape[:2], np.float32)])
    img.pixels.foreach_set(np.ascontiguousarray(rgba[::-1], np.float32).ravel())
    sc = bpy.context.scene
    fmt = sc.render.image_settings.file_format, sc.render.image_settings.color_mode
    sc.render.image_settings.file_format, sc.render.image_settings.color_mode = "JPEG", "RGB"
    sc.render.image_settings.quality = 92
    img.save_render(path, scene=sc)
    sc.render.image_settings.file_format, sc.render.image_settings.color_mode = fmt
    bpy.data.images.remove(img)


def previews(rig, cam, floor, sources, skin, out_dir, height, beauty, backdrop):
    _show(sources, skin, source=False)
    studio.turnaround(rig, cam, floor, out_dir, height, tag="game_turnaround", backdrop=backdrop)
    studio.render_beauty(rig, cam, beauty["top"], "game_top", out_dir)
    view = beauty["hero"]
    shots = []
    for src in (True, False):
        _show(sources, skin, source=src)
        studio.render_beauty(rig, cam, view, "_cmp", out_dir, 0.6)
        p = os.path.join(out_dir, "_cmp.jpg")
        shots.append(studio.load_rgba(p)[..., :3])
        os.remove(p)
    gap = np.ones((shots[0].shape[0], 12, 3), np.float32)
    _save_jpg(np.concatenate([shots[0], gap, shots[1]], axis=1), os.path.join(out_dir, "game_vs_source.jpg"))
    _show(sources, skin, source=True, both=True)


# ---------------------------------------------------------------------------- run
def run(root, name, build, make_materials, height, beauty, backdrop=studio.BACKDROP, rig_spec=None, rig_out=None,
        previews_dir=None, variants=None, skin_opts=None):
    """Stages 1-6 (the static skin), then stage 7 (`rig.py`) when the character supplies a rig spec.
    `previews_dir` replaces <character>/previews; `variants(name)` returns the make_materials of a look
    variant, for `--variant` (a re-bake of the albedo alone, nothing else runs). `skin_opts`: `min_wall_voxels`
    (inward solidified walls thinner than that many voxels thickened to it, for the skin) and `inset`
    (parts thinned for the skin), `walls` (per-part skin walls) and `fine` (parts remeshed apart at
    a finer voxel), see `make_skin` -- characters in thin cloth, with bare fingers."""
    args = _args()
    if args.variant:
        if variants is None:
            raise SystemExit("this character has no look variants")
        rebake_variant(args, root, name, build, variants(args.variant), args.variant, asset=rig_out)
        return
    if args.retexture:
        if rig_out is None:
            raise SystemExit("--retexture needs the character's game asset (run(rig_out=...))")
        retexture(args, root, name, build, make_materials, rig_out)
        return
    if not args.rig_only:
        static(args, root, name, build, make_materials, height, beauty, backdrop, previews_dir, skin_opts)
    if rig_spec is not None and not args.no_rig:
        import rig
        rig.run(args, root, name, build, make_materials, rig_spec, rig_out, previews_dir)


def _skin_from_asset(asset):
    """The game asset's own skin as a bare mesh object "GameSkin" (no armature, no modifiers, rest
    pose, in the character's frame -- `rig.py` exports it there): the mesh and UVs the asset's
    textures are drawn on. Starts from an empty scene."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=asset)
    # the skinned mesh (the importer also makes small bone-shape meshes)
    ob = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    me = ob.data.copy()
    me.transform(ob.matrix_world)
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    me.materials.clear()
    skin = bpy.data.objects.new("GameSkin", me)
    bpy.context.scene.collection.objects.link(skin)
    return skin


def _saved_skin(root, name, asset):
    """The skin a re-bake lands on, in the source's frame: the saved game/<name>_game.blend's when it
    exists (stage 6 wrote it), else the game asset's own (`_skin_from_asset`) -- the .blend is
    regenerated output, gitignored, so a fresh checkout has only the asset. Both carry the same mesh
    and UVs."""
    blend = os.path.join(root, "game", name + "_game.blend")
    if os.path.isfile(blend):
        bpy.ops.wm.open_mainfile(filepath=blend)
        skin = bpy.data.objects["GameSkin"]
        for ob in list(bpy.data.objects):
            if ob is not skin:
                bpy.data.objects.remove(ob)
        skin.data.transform(Matrix.Translation(-Vector(skin["export_offset"])))  # back into the source's frame
        return skin, blend
    if asset is None or not os.path.isfile(asset):
        raise SystemExit("no saved skin: neither %s nor the game asset %s exists" % (blend, asset))
    return _skin_from_asset(asset), asset


def _bake_albedo(args, skin, build, make_materials):
    """The source rebuilt with `make_materials` round `skin`, the colour, AO and gloss bakes of stage 4
    -> (RGBA albedo in Blender row order, timing, bake stats)."""
    kit.RES = 0.004
    build(make_materials())
    sources = [o for o in bpy.context.scene.objects if o.type == "MESH" and o is not skin]
    _rig, _cam, floor = studio.setup(samples=args.samples)
    floor.hide_render = True  # as in stage 4: the studio floor would occlude the soles' AO
    log, timing = {}, {}
    _normal, ao, color, gloss, hit = bake_all(skin, sources, args.tex, args.cage, args.ray, args.bake_samples,
                                              timing, log)
    cover, _overlap = uv_coverage(skin, args.tex)
    valid = hit >= 0.5
    ao, color, gloss = (_inpaint(x, valid, cover) for x in (ao, color, gloss))
    return albedo_rgba(color, ao, gloss), timing, dict(missed_fraction_of_covered=round(float(
        (cover & ~valid).sum() / max(1, cover.sum())), 5), **gloss_stats(gloss, cover))


def gloss_stats(gloss, cover):
    """How the gloss mask spreads over the covered texels."""
    g = gloss[..., 0][cover]
    return dict(gloss_mean=round(float(g.mean()), 4), gloss_over_0_5=round(float((g > 0.5).mean()), 4),
                gloss_0_05_to_0_5=round(float(((g > 0.05) & (g <= 0.5)).mean()), 4),
                gloss_matte_under_0_05=round(float((g <= 0.05).mean()), 4))


def rebake_variant(args, root, name, build, make_materials, variant, asset=None):
    """The albedo of a look variant on the SAME skin and UVs (`_saved_skin`), the source rebuilt with
    the variant's materials, the colour, AO and gloss bakes of stage 4 again -> game/albedo_<variant>.png
    (gloss in its alpha). Geometry, UVs, normal map and the .glb are untouched."""
    t0 = time.time()
    game = os.path.join(root, "game")
    skin, frm = _saved_skin(root, name, asset)
    rgba, timing, bstats = _bake_albedo(args, skin, build, make_materials)
    path = os.path.join(game, "albedo_%s.png" % variant)
    _save_png(rgba, path, "sRGB", alpha=True)
    print("variant %s: %s (skin from %s) in %.1fs %s %s" % (variant, path, frm, time.time() - t0, json.dumps(timing),
                                                           json.dumps(bstats)), flush=True)
    return path


def retexture(args, root, name, build, make_materials, asset):
    """Re-bake the game asset's albedo (colour x AO, gloss in alpha) onto the asset's OWN mesh and UVs
    and splice it into the .glb in place (`glbtex.replace_image`): the mesh, skin and clips -- every
    other buffer view -- stay byte for byte, checked here. The normal map is kept. For a change to
    the materials alone (the gloss mask, a colour) without re-running stages 1-7."""
    import glbtex
    t0 = time.time()
    before = glbtex.digest(asset)
    skin = _skin_from_asset(asset)
    rgba, timing, bstats = _bake_albedo(args, skin, build, make_materials)
    path = os.path.join(root, "game", "albedo.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _save_png(rgba, path, "sRGB", alpha=True)
    with open(path, "rb") as f:
        glbtex.replace_image(asset, "baseColor", f.read())
    after = glbtex.digest(asset)
    if after != before:
        raise SystemExit("retexture changed the mesh, skin or clips of %s: %s -> %s" % (asset, before, after))
    stats_path = os.path.join(root, "game", "stats.json")
    stats = {}
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    stats["retexture"] = dict(asset=os.path.relpath(asset, os.path.join(root, "..", "..", "..")).replace("\\", "/"),
                              mesh_skin_clips_unchanged=True, geometry_digest=before, bake=bstats,
                              gloss=dict(rough_full=GLOSS_ROUGH_FULL, rough_matte=GLOSS_ROUGH_MATTE, power=GLOSS_POWER),
                              timing_s=dict(timing, total_s=round(time.time() - t0, 1)))
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=1)
    print("retexture %s: %s in %.1fs %s" % (name, asset, time.time() - t0, json.dumps(stats["retexture"])), flush=True)
    return path


def static(args, root, name, build, make_materials, height, beauty, backdrop, previews_dir=None, skin_opts=None):
    t0 = time.time()
    log, timing = {}, {}
    game, prev = os.path.join(root, "game"), previews_dir or os.path.join(root, "previews")
    os.makedirs(game, exist_ok=True)
    os.makedirs(prev, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    kit.RES = 0.004
    build(make_materials())
    sources = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    rig, cam, floor = studio.setup(samples=args.samples, backdrop=backdrop)
    timing["build_s"] = round(time.time() - t0, 1)

    t = time.time()
    so = skin_opts or {}
    skin = make_skin(sources, args.voxel, args.tris, args.close, log,
                     min_wall=args.voxel * so["min_wall_voxels"] if so.get("min_wall_voxels") else None,
                     inset=so.get("inset", ()), walls=so.get("walls", ()), fine=so.get("fine"))
    timing["skin_s"] = round(time.time() - t, 1)
    print("skin:", json.dumps(log), flush=True)

    t = time.time()
    unwrap(skin, args.angle, args.tex, args.min_island, log)
    m = mesh_measure(skin)
    m["smart_uv_islands_before_merge"] = log["smart_uv_islands"]
    m["uv_charts_recut_for_folding"] = log["charts_recut_for_folding"]
    m["uv_folded_charts_left"] = log["folded_charts_left"]
    m["hidden_area_fraction"] = hidden_area_fraction(skin)
    timing["uv_s"] = round(time.time() - t, 1)
    print("mesh:", json.dumps(m), flush=True)

    t = time.time()
    floor.hide_render = True
    normal, ao, color, gloss, hit = bake_all(skin, sources, args.tex, args.cage, args.ray, args.bake_samples,
                                             timing, log)
    floor.hide_render = False
    cover, overlap = uv_coverage(skin, args.tex)
    missed = cover & (hit < 0.5)
    # A texel whose ray found no source (a sliver in a tight crease) takes its neighbours' values.
    valid = hit >= 0.5
    normal, ao, color, gloss = (_inpaint(x, valid, cover) for x in (normal, ao, color, gloss))
    albedo_path, normal_path = os.path.join(game, "albedo.png"), os.path.join(game, "normal.png")
    _save_png(albedo_rgba(color, ao, gloss), albedo_path, "sRGB", alpha=True)
    _save_png(np.dstack([normal[..., :3], np.ones(normal.shape[:2])]), normal_path, "Non-Color")
    skin.data.materials.append(game_material(albedo_path, normal_path))
    timing["bake_s"] = round(time.time() - t, 1)
    bake = dict(texture_px=args.tex, cage_extrusion_m=args.cage, max_ray_distance_m=args.ray,
                uv_coverage=round(float(cover.mean()), 4), missed_texels=int(missed.sum()),
                missed_fraction_of_covered=round(float(missed.sum() / max(1, cover.sum())), 5),
                uv_overlap_texels=int(overlap.sum()),
                source_faces_turned_outward=log.get("bake_source_faces_flipped", []),
                source_material_slots=log.get("bake_source_material_slots"),
                source_attributes=log.get("bake_source_attributes"),
                mean_ao_on_covered=round(float(ao[..., 0][cover].mean()), 3), **gloss_stats(gloss, cover))
    print("bake:", json.dumps(bake), json.dumps(timing), flush=True)

    t = time.time()
    iou = silhouettes(rig, cam, floor, sources, skin, prev, height)
    print("silhouette IoU:", json.dumps(iou), flush=True)
    if not args.no_previews:
        previews(rig, cam, floor, sources, skin, prev, height, beauty, backdrop)
    timing["measure_previews_s"] = round(time.time() - t, 1)

    # Export: the skin alone, feet on the ground at the origin (offset applied to the exported data only).
    t = time.time()
    co = np.empty(len(skin.data.vertices) * 3, np.float32)
    skin.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    feet = co[co[:, 2] < co[:, 2].min() + 0.05]
    offset = Vector((-float(feet[:, 0].mean()), -float(feet[:, 1].mean()), -float(co[:, 2].min())))
    for ob in sources + [floor]:
        bpy.data.objects.remove(ob)
    skin.data.transform(Matrix.Translation(offset))
    skin["export_offset"] = offset[:]  # the rig stage moves the skin back into the character's frame
    _select([skin], skin)
    glb = os.path.join(game, name + ".glb")
    bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_selection=True, export_yup=True,
                              export_apply=True, export_texcoords=True, export_normals=True, export_tangents=True,
                              export_materials="EXPORT", export_image_format="AUTO", export_animations=False,
                              export_skins=False, export_morph=False, export_cameras=False, export_lights=False)
    _splice_albedo(glb, albedo_path)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(game, name + "_game.blend"))
    timing["export_s"] = round(time.time() - t, 1)
    timing["total_s"] = round(time.time() - t0, 1)

    stats = dict(character=name,
                 params=dict(tris_target=args.tris, voxel_m=args.voxel, close_voxels=args.close,
                             uv_angle_deg=args.angle, min_island_faces=args.min_island, bake_samples=args.bake_samples),
                 source=dict(objects=len(sources), joined_tris_after_thickening=log["joined_tris"]),
                 skin=dict(open_parts_thickened=log["open_parts_thickened"], remeshed_tris=log["remeshed_tris"],
                           thin_walls_thickened=log.get("thin_walls_thickened", 0), inset_parts=log.get("inset_parts", 0),
                           fine_shells=log.get("fine_shells", 0), fine_remeshed_tris=log.get("fine_remeshed_tris", 0),
                           fine_tris=log.get("fine_tris", 0),
                           cavity_voxels_filled=log.get("cavity_voxels", 0), filled_tris=log["filled_tris"],
                           pieces_kept=log["pieces_kept"], enclosed_or_tiny_pieces_dropped=log["pieces_dropped"]),
                 mesh=m, bake=bake, silhouette_iou=iou,
                 export=dict(file=os.path.basename(glb), bytes=os.path.getsize(glb),
                             offset_m=[round(x, 4) for x in offset]),
                 timing_s=timing)
    with open(os.path.join(game, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print("stats:", json.dumps(stats))
    print("done in %.1fs" % (time.time() - t0))
