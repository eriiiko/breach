"""The pelvis judged on its own (Blender 4.5): outline measurements and the hip pictures.

    --outline   slice the built body (the evaluated `Suit_Body`: mirrored, solidified) at every 2 mm
                of height and print the outlines' worst direction change per cm, waist to knee
                (`silhouette.py`), per view; writes source/<prefix>outline.json
    --hips      the hip pictures (suitbuild.run calls `pictures`):
                  hips.jpg               the default, waist to knee, matte grey, pure form (no seams,
                                         panels, zips, pads, arms), front / 3/4 front / side / 3/4 back / back
                  hips_vs_drawings.jpg   front, side, back in matte grey with the drawing's outline
                                         (orange), the concept beside the front
                  hips_before_after.jpg  the outlines of tables.HIPS_BEFORE (the previous default) and of
                                         the default, overlaid, front / side / back
                  crotch.jpg             the crotch close up: matte front / back / 3/4 front / from below
                                         (front and back, 40 deg under the horizontal); glossy front, 3/4 front
    The measurements (source/<prefix>outline.json) also hold the half-gap between the legs every 5 mm,
    how square the surface crosses the mid-plane (`midplane_normals`) and a mesh check of the suit
    (`mesh_check`: degenerate faces, non-manifold and open edges).
"""
import json
import math
import os

import bpy
import numpy as np

import silhouette
import studio

DECOR = ("Suit_Piping", "Suit_Mesh", "Suit_Zip", "Suit_Knee_Pad")
ARMS = ("Suit_Sleeve", "Suit_Cuff")
BEFORE = (0.20, 0.45, 0.95)   # sRGB: the previous default's outline
AFTER = (1.0, 0.42, 0.05)     # sRGB: the new default's outline (orange, as the drawings' elsewhere)


def body_mesh(name="Suit_Body"):
    """(V, E) of the evaluated body mesh (world coordinates: objects sit at identity)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ob = bpy.data.objects[name]
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    E = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", E)
    ev.to_mesh_clear()
    return V.reshape(-1, 3), E.reshape(-1, 2)


def measure(source, prefix, lo=0.60, hi=1.14):
    V, E = body_mesh()
    zs = np.round(np.arange(0.56, 1.20 + 1e-9, 0.002), 4)
    o = silhouette.outlines(V, E, zs)
    res, tr = silhouette.report(zs, o, lo, hi)
    print("outline turning, worst deg/cm %.2f-%.2f m:" % (lo, hi))
    for k, (v, z) in res.items():
        print("  %-14s %6.2f at z %.3f" % (k, v, z))
    # the gap between the legs, every 5 mm (the left half's innermost x; 0 where the legs are joined)
    z5 = np.round(np.arange(0.70, 0.905 + 1e-9, 0.005), 4)
    g5 = silhouette.outlines(V, E, z5, gap_min=2e-4)["inner"]
    print("half-gap mm:", " ".join("%.3f:%s" % (z, "--" if not np.isfinite(g) else "%.1f" % (1000 * g)) for z, g in zip(z5, g5)))
    out = dict(worst={k: dict(deg_per_cm=v, z=z) for k, (v, z) in res.items()}, z=zs.tolist(),
               outline={k: [None if not np.isfinite(x) else float(x) for x in a] for k, a in o.items()},
               turning={k: [None if not np.isfinite(x) else float(x) for x in a] for k, a in tr.items()},
               half_gap=dict(z=z5.tolist(), g=[None if not np.isfinite(g) else float(g) for g in g5]),
               midplane=midplane_normals(), check=mesh_check())
    with open(os.path.join(source, prefix + "outline.json"), "w") as f:
        json.dump(out, f)
    return out


def _source_mesh(ob):
    me = ob.data
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    loops = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", loops)
    start = np.empty(len(me.polygons), dtype=np.int64)
    total = np.empty(len(me.polygons), dtype=np.int64)
    me.polygons.foreach_get("loop_start", start)
    me.polygons.foreach_get("loop_total", total)
    return V, loops, start, total


def midplane_normals(name="Suit_Body", z_lo=0.82, z_hi=1.14, eps=1e-6):
    """How square the surface crosses the mid-plane: at every vertex ON the plane (the half-body's weld
    line) the area-weighted normal of its own half's faces; its sideways component |n_x| is the sine of
    the crease's half-angle (0 = the halves meet square, no crease). Measured on the half-body's own mesh
    (before mirror and solidify), per 5 mm of height, front (y < the line's mid y) and back, worst."""
    V, loops, start, total = _source_mesh(bpy.data.objects[name])
    on = np.abs(V[:, 0]) < eps
    acc = np.zeros((len(V), 3))
    touch = np.add.reduceat(on[loops].astype(np.int64), start) > 0
    for s0, t0 in zip(start[touch], total[touch]):
        f = loops[s0:s0 + t0]
        P = V[f]
        n = np.zeros(3)
        for i in range(1, len(f) - 1):
            n += 0.5 * np.cross(P[i] - P[0], P[i + 1] - P[0])
        acc[f[on[f]]] += n
    idx = np.flatnonzero(on & (V[:, 2] >= z_lo) & (V[:, 2] <= z_hi) & (np.linalg.norm(acc, axis=1) > 0))
    if not len(idx):
        return dict(worst=None)
    n = acc[idx] / np.linalg.norm(acc[idx], axis=1, keepdims=True)
    Pz, Py, nx = V[idx, 2], V[idx, 1], np.abs(n[:, 0])
    rows = []
    for z0 in np.arange(z_lo, z_hi, 0.005):
        m = (Pz >= z0) & (Pz < z0 + 0.005)
        if not m.any():
            continue
        ymid = 0.5 * (Py[m].min() + Py[m].max())
        fr, bk = m & (Py <= ymid), m & (Py > ymid)
        rows.append((round(float(z0), 3), float(nx[fr].max()) if fr.any() else None, float(nx[bk].max()) if bk.any() else None))
    j = int(np.argmax(nx))
    print("mid-plane facet |n_x| per 5 mm (front, back):", " ".join("%.3f:%s/%s" % (z, "-" if a is None else "%.2f" % a, "-" if b is None else "%.2f" % b)
                                                             for z, a, b in rows))
    print("mid-plane facet |n_x| worst %.3f at z %.3f y %.3f" % (nx[j], Pz[j], Py[j]))
    # the SURFACE's own normal at the plane, free of the facet's width: the level section through each weld
    # vertex A (the mesh sliced there) fitted near A by y = y_A + b x + c x^2 (x < 5 mm, one branch), the
    # weld line's own slope e = dy/dz from its neighbours: n ~ (b, -1, e), |n_x| = |b| / sqrt(1 + b^2 + e^2)
    nxt = np.arange(len(loops)) + 1
    nxt[start + total - 1] = start
    Eall = np.column_stack([loops, loops[nxt]])
    Eall = Eall[(V[Eall[:, 0], 0] < 0.02) & (np.abs(V[Eall[:, 0], 2] - 0.5 * (z_lo + z_hi)) < 0.5 * (z_hi - z_lo) + 0.01)]
    fit = []
    for a in idx:
        A = V[a]
        z = A[2] + 1e-6
        p, q = V[Eall[:, 0]], V[Eall[:, 1]]
        m = (p[:, 2] - z) * (q[:, 2] - z) < 0.0
        w = (z - p[m, 2]) / (q[m, 2] - p[m, 2])
        S_ = p[m] + w[:, None] * (q[m] - p[m])
        S_ = S_[(S_[:, 0] > 2e-4) & (S_[:, 0] < 0.005) & (np.hypot(S_[:, 0], S_[:, 1] - A[1]) < 0.006)]
        if len(S_) < 2:
            continue
        X = np.column_stack([S_[:, 0], S_[:, 0] ** 2])
        b = np.linalg.lstsq(X, S_[:, 1] - A[1], rcond=None)[0][0]
        same = idx[(np.abs(V[idx, 2] - A[2]) < 0.004) & (np.abs(V[idx, 1] - A[1]) < 0.01)]
        e = np.polyfit(V[same, 2], V[same, 1], 1)[0] if len(same) >= 3 else 0.0
        fit.append((A[2], A[1], abs(b) / math.sqrt(1.0 + b * b + e * e)))
    if fit:
        F_ = np.array(fit)
        frow = []
        for z0 in np.arange(z_lo, z_hi, 0.005):
            m = (F_[:, 0] >= z0) & (F_[:, 0] < z0 + 0.005)
            if m.any():
                ymid = 0.5 * (F_[m, 1].min() + F_[m, 1].max())
                fr, bk = m & (F_[:, 1] <= ymid), m & (F_[:, 1] > ymid)
                frow.append((round(float(z0), 3), float(F_[fr, 2].max()) if fr.any() else None, float(F_[bk, 2].max()) if bk.any() else None))
        k = int(np.argmax(F_[:, 2]))
        print("mid-plane surface |n_x| per 5 mm (front, back):", " ".join("%.3f:%s/%s" % (z, "-" if a is None else "%.2f" % a, "-" if b is None else "%.2f" % b)
                                                                    for z, a, b in frow))
        print("mid-plane surface |n_x| worst %.3f at z %.3f y %.3f" % (F_[k, 2], F_[k, 0], F_[k, 1]))
        surf = dict(worst=float(F_[k, 2]), at=(float(F_[k, 0]), float(F_[k, 1])), rows=frow)
    else:
        surf = None
    return dict(worst=float(nx[j]), at=(float(Pz[j]), float(Py[j])), rows=rows, surface=surf)


def mesh_check(coll="Suit", area_min=1e-10, edge_min=1e-6):
    """Every evaluated mesh in the collection: degenerate faces (area under `area_min` m^2 or an edge
    under `edge_min` m), non-manifold edges (more than two faces), open edges (one face)."""
    dg = bpy.context.evaluated_depsgraph_get()
    tot = dict(degenerate=0, nonmanifold=0, open=0)
    per = {}
    for ob in bpy.data.objects:
        if ob.type != "MESH" or not ob.users_collection or ob.users_collection[0].name != coll:
            continue
        ev = ob.evaluated_get(dg)
        me = ev.to_mesh()
        nv, nf, ne = len(me.vertices), len(me.polygons), len(me.edges)
        V = np.empty(nv * 3)
        me.vertices.foreach_get("co", V)
        V = V.reshape(-1, 3)
        area = np.empty(nf)
        me.polygons.foreach_get("area", area)
        E = np.empty(ne * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", E)
        E = E.reshape(-1, 2)
        le = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("edge_index", le)
        start = np.empty(nf, dtype=np.int64)
        total = np.empty(nf, dtype=np.int64)
        me.polygons.foreach_get("loop_start", start)
        me.polygons.foreach_get("loop_total", total)
        ev.to_mesh_clear()
        elen = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        cnt = np.bincount(le, minlength=ne)
        short = elen < edge_min
        face_of_loop = np.repeat(np.arange(nf), total)
        bad = (area < area_min)
        bad[np.unique(face_of_loop[short[le]])] = True
        r = dict(degenerate=int(bad.sum()), nonmanifold=int((cnt > 2).sum()), open=int((cnt == 1).sum()))
        if any(r.values()):
            per[ob.name] = r
        for k in tot:
            tot[k] += r[k]
    print("mesh check (%s): %s %s" % (coll, json.dumps(tot), json.dumps(per) if per else ""))
    return dict(total=tot, objects=per)


def hide(prefixes, colls=()):
    for ob in bpy.data.objects:
        if ob.name.startswith(prefixes) or (ob.users_collection and ob.users_collection[0].name in colls):
            ob.hide_render = True


def _band_masks(sheet_ref, suitbuild, rig, previews, z0, z1, hw):
    """{view: alpha mask} cropped to the band, for the front, side and back sheets."""
    out = {}
    for S, view, w in ((sheet_ref.SHEET, "front", hw[0]), (sheet_ref.SIDE, "side", hw[1]), (sheet_ref.BACK, "back", hw[0])):
        p = S.render(*rig, previews, "hipsmask")[view]
        out[view] = suitbuild.band_crop(S, p[..., 3] > 0.5, z0, z1, w)
    return out


def pictures(tables, sheet_ref, previews, source, args, suitbuild):
    """The three hip pictures (see the module doc). Builds tables.HIPS_BEFORE, then the default."""
    z0, z1 = tables.COMPARE_BAND
    fw, sw = 0.27, 0.20
    F, Sd, Bk = sheet_ref.SHEET, sheet_ref.SIDE, sheet_ref.BACK
    # --- the previous default: its outlines only
    report, rig, cam, floor = suitbuild.build(tables, tables.HIPS_BEFORE, "", args.draft, args.samples)
    hide(DECOR + ARMS, ("Hands",))
    before = _band_masks(sheet_ref, suitbuild, (rig, cam, floor), previews, z0, z1, (fw, sw))
    measure(source, tables.HIPS_BEFORE + "_")
    # --- the default
    report, rig, cam, floor = suitbuild.build(tables, "", "", args.draft, args.samples)
    out = measure(source, "")
    glossy = crotch_cells(cam, previews, CROTCH_GLOSS, "glossy suit")
    suitbuild.matte_figure()
    hide(DECOR)
    matte = crotch_cells(cam, previews, CROTCH_VIEWS, "matte")
    w = sum(c.shape[1] for c, _ in matte) + 10 * (len(matte) - 1)
    rows = [suitbuild.assemble_row(matte), suitbuild.assemble_row(glossy)]
    rows[1] = np.concatenate([rows[1], np.ones((rows[1].shape[0], w - rows[1].shape[1], 4), np.float32)], axis=1)
    title = suitbuild.label_strip(tables.PREFIX.capitalize() + " - the crotch, close up (85 mm lens, 0.75 m): top row matte grey without "
                                  "seams, below the glossy suit", w, 44)
    suitbuild.save_image(np.concatenate([title, rows[0], np.ones((10, w, 4), np.float32), rows[1]], axis=0),
                         os.path.join(previews, "crotch.jpg"))
    # hips_vs_drawings.jpg: front (with the concept beside it), side, back; arms shown (the drawings have theirs)
    cells = []
    concept = suitbuild.concept_panel(sheet_ref, F)
    cells.append((suitbuild.band_crop(F, concept, z0, z1, fw), "the concept"))
    for S, view, hw in ((F, "front", fw), (Sd, "side", sw), (Bk, "back", fw)):
        line = suitbuild.band_crop(S, suitbuild.outline(S.mask()[:, S.panel_slice(view)[1]]), z0, z1, hw)
        p = suitbuild.band_crop(S, studio.compose(S.render(rig, cam, floor, previews, "hips"), [view]), z0, z1, hw).copy()
        p[line[:p.shape[0], :p.shape[1]]] = (*suitbuild.OUTLINE, 1.0)
        cells.append((p, view))
    suitbuild.save_image(suitbuild.assemble_row(cells, [tables.PREFIX.capitalize() + " - the hips against the drawings", "matte grey: the model, orange: the drawing's outline"]),
                         os.path.join(previews, "hips_vs_drawings.jpg"))
    # hips_before_after.jpg: outlines only, over the new default's matte form (arms hidden)
    hide(ARMS, ("Hands",))
    after = _band_masks(sheet_ref, suitbuild, (rig, cam, floor), previews, z0, z1, (fw, sw))
    cells = []
    for S, view, hw in ((F, "front", fw), (Sd, "side", sw), (Bk, "back", fw)):
        p = suitbuild.band_crop(S, studio.compose(S.render(rig, cam, floor, previews, "hips"), [view]), z0, z1, hw).copy()
        p[..., :3] = 0.55 * p[..., :3] + 0.45 * 0.93   # the form faded, so the lines read
        for m, col in ((before[view], BEFORE), (after[view], AFTER)):
            e = suitbuild.outline(m, 3)
            p[e[:p.shape[0], :p.shape[1]]] = (*col, 1.0)
        cells.append((p, view))
    w = out["worst"]
    title = [tables.PREFIX.capitalize() + " - hip outlines: blue = before (%s), orange = now" % tables.HIPS_BEFORE,
             "worst turn deg/cm now: front %.1f, side front %.1f, side back %.1f" % (
                 w["outer"]["deg_per_cm"], w["front"]["deg_per_cm"], w["back"]["deg_per_cm"])]
    suitbuild.save_image(suitbuild.assemble_row(cells, title), os.path.join(previews, "hips_before_after.jpg"))
    # hips.jpg: pure form, five views, orthographic, waist to knee
    hip_views(tables, suitbuild, rig, cam, floor, previews, z0, z1)


# crotch.jpg: (label, azimuth deg, elevation deg) round the crotch's tip; below = 40 deg under the horizontal
CROTCH_Z = 0.882
CROTCH_VIEWS = (("front", 0.0, 0.0), ("back", 180.0, 0.0), ("three-quarter front", 40.0, 0.0), ("from below, front", 0.0, -40.0),
                ("from below, back", 180.0, -40.0))
CROTCH_GLOSS = (("front", 0.0, 0.0), ("three-quarter front", 40.0, 0.0))


def crotch_cells(cam, previews, views, what, d=0.75, res=640):
    import math
    cells = []
    for name, az, el in views:
        a, e = math.radians(az), math.radians(el)
        cam.data.type, cam.data.lens, cam.data.clip_start = "PERSP", 85.0, 0.01
        cam.location = (d * math.sin(a) * math.cos(e), -d * math.cos(a) * math.cos(e), CROTCH_Z + d * math.sin(e))
        studio._look_at(cam, (0.0, 0.0, CROTCH_Z))
        path = os.path.join(previews, "_crotch.png")
        studio.render(path, (res, res))
        img = studio.load_rgba(path)
        os.remove(path)
        img[..., 3] = 1.0
        cells.append((img, "%s, %s" % (name, what)))
    return cells


HIP_VIEWS = (("front", 0.0), ("three-quarter front", 40.0), ("side", 90.0), ("three-quarter back", 140.0), ("back", 180.0))


def hip_views(tables, suitbuild, rig, cam, floor, previews, z0, z1, h_px=1300, half_w=0.24):
    mpp = (z1 - z0) / h_px
    w_px = int(round(2 * half_w / mpp))
    views = [("v%d" % i, az, w_px) for i, (_, az) in enumerate(HIP_VIEWS)]
    # the camera's centre at the band's middle: ortho_panels frames a square of max(w, h) round centre_z
    panels = studio.ortho_panels(rig, cam, floor, previews, views, mpp, h_px, 0.5 * (z0 + z1), "hipsv")
    cells = [(studio.compose(panels, [v[0]]), name) for (name, _), v in zip(HIP_VIEWS, views)]
    suitbuild.save_image(suitbuild.assemble_row(cells, tables.PREFIX.capitalize() + " - the pelvis, pure form (matte grey, no seams, panels or arms), waist to knee"),
                         os.path.join(previews, "hips.jpg"))
