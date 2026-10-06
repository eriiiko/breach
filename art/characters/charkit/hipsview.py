"""The pelvis judged on its own (Blender 4.5): outline measurements and the hip pictures.

    --outline   slice the built body (the evaluated `Suit_Body`: mirrored, solidified) at every 2 mm
                of height and print the outlines' worst direction change per cm, waist to knee
                (`silhouette.py`), per view; writes source/<prefix>outline.json
    --hips      the hip pictures (suitbuild.run calls `pictures`):
                  hips.jpg               the default, waist to knee, matte grey, pure form (no seams,
                                         panels, zips, pads, arms), front / 3/4 front / side / 3/4 back / back
                  hips_vs_drawings.jpg   front, side, back in matte grey with the drawing's outline
                                         (orange), the concept beside the front
                  hips_before_after.jpg  tables.HIPS_BEFORE (the previous default) above, the default below with
                                         the before's outline over it, matte, front / 3/4 front / side / back
                  shape_vs_silhouette.jpg  sheet_ref.SILHOUETTE (a frontal body reference) beside both
                                         defaults' matte front views at one scale
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
BEFORES = (BEFORE, (0.10, 0.68, 0.30), (0.80, 0.20, 0.70))   # one colour per earlier shape (tables.HIPS_BEFORE, in order)
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
    lens = lens_measure(V, E)
    chords = side_chords(zs, o)
    flare = flare_shares(zs, o)
    print("front flare: waist %.1f mm at z %.3f, hip line (z %.3f) %.1f, widest %.1f at z %.3f; shares of the gain at 0.2/0.4/0.6/0.8: %s" % (
        flare["waist_mm"], flare["waist_z"], flare["hip_z"], flare["hip_mm"], flare["widest_mm"], flare["widest_z"], flare["shares"]))
    print("first light through the leg gap at z %s; the tip's arch (d2 <= 0) %s mm long; then the inner outline convex (d2 > 0) down to z %s" % (
        lens["first_light"], lens.get("tip_arch_mm"), lens["convex_to"]))
    print("side chords: thigh front max %.1f mm from the 0.90-0.62 chord at z %.3f; seat %.1f mm from the lumbar(%.3f)-0.70 chord at z %.3f" % (
        chords["front"]["dev_mm"], chords["front"]["at"], chords["back"]["dev_mm"], chords["back"]["lumbar_z"], chords["back"]["at"]))
    out = dict(worst={k: dict(deg_per_cm=v, z=z) for k, (v, z) in res.items()}, z=zs.tolist(),
               outline={k: [None if not np.isfinite(x) else float(x) for x in a] for k, a in o.items()},
               turning={k: [None if not np.isfinite(x) else float(x) for x in a] for k, a in tr.items()},
               half_gap=dict(z=z5.tolist(), g=[None if not np.isfinite(g) else float(g) for g in g5]),
               lens=lens, chords=chords, flare=flare, midplane=midplane_normals(), check=mesh_check())
    with open(os.path.join(source, prefix + "outline.json"), "w") as f:
        json.dump(out, f)
    return out


def lens_measure(V, E, z_lo=0.70, z_hi=0.90, dz=0.001, win=0.006):
    """The leg gap per mm: the half-gap g(z) (the left half's innermost x, nan where the legs are joined), the
    first light (the highest open level), and the inner outline's curvature sign: d2 = g'' from a local
    quadratic fitted over +-win/2 of height; d2 > 0 = convex (the thigh bulging toward the centre).
    `convex_to` = how far down from the first light d2 stays positive (levels within 1 mm of the first
    light skipped: the bridge's tip)."""
    zf = np.round(np.arange(z_lo, z_hi + 1e-9, dz), 4)
    g = silhouette.outlines(V, E, zf, gap_min=2e-4)["inner"]
    ok = np.isfinite(g)
    if not ok.any():
        return dict(first_light=None, convex_to=None)
    first = float(zf[ok].max())
    d2 = np.full(len(zf), np.nan)
    for i, z in enumerate(zf):
        m = ok & (np.abs(zf - z) <= 0.5 * win + 1e-9)
        if m.sum() >= 5 and z <= first - 0.001:
            d2[i] = 2.0 * np.polyfit(zf[m] - z, g[m], 2)[0]
    # going down from the first light: the tip's own arch (d2 <= 0: the bridge's end, how long), then the convex run
    conv, arch_end = None, None
    for z, c in sorted(zip(zf, d2), key=lambda q: -q[0]):
        if not np.isfinite(c):
            continue
        if arch_end is None:
            if c > 0.0:
                arch_end = float(z)
            else:
                continue
        if c <= 0.0:
            break
        conv = float(z)
    return dict(first_light=first, tip_arch_mm=None if arch_end is None else round(1000.0 * (first - arch_end), 1), convex_to=conv,
                z=zf.tolist(), g=[None if not np.isfinite(x) else float(x) for x in g], d2=[None if not np.isfinite(x) else float(x) for x in d2])


def flare_shares(zs, o, hip_z=0.893, waist=(1.08, 1.20), widest=(0.78, 1.10), fr=(0.2, 0.4, 0.6, 0.8)):
    """The front view's waist-to-hip flare, measured as on the owner's frontal reference: the outer outline's share of
    its gain from the narrowest waist (min within `waist`) to the hip line at `hip_z` (just above the crotch), at the
    fractions `fr` of that height; the narrowest and widest half-widths (mm)."""
    zs, u = np.asarray(zs), np.asarray(o["outer"], float)
    mw = (zs >= waist[0]) & (zs <= waist[1]) & np.isfinite(u)
    zw, w = float(zs[mw][np.argmin(u[mw])]), float(u[mw].min())
    H = float(np.interp(hip_z, zs, u))
    mx = (zs >= widest[0]) & (zs <= widest[1]) & np.isfinite(u)
    sh = [round((float(np.interp(zw - f * (zw - hip_z), zs, u)) - w) / (H - w), 3) for f in fr]
    return dict(waist_mm=1000 * w, waist_z=zw, hip_z=hip_z, hip_mm=1000 * H, widest_mm=1000 * float(u[mx].max()),
                widest_z=float(zs[mx][np.argmax(u[mx])]), shares=sh)


def side_chords(zs, o, front=(0.90, 0.62), back_low=0.70, lumbar=(1.04, 1.16)):
    """How far the side outline bows away from a straight line: the front outline's largest distance from the chord
    between its points at front[0] and front[1] (the thigh's swell, + = in front of the chord), and the back
    outline's from the chord between its lumbar minimum (deepest point within `lumbar`) and its point at back_low
    (the seat's round, + = behind the chord). mm, measured horizontally (y)."""
    zs = np.asarray(zs)

    def dev(u, za, zb, sign):
        m = (zs <= max(za, zb)) & (zs >= min(za, zb)) & np.isfinite(u)
        ua, ub = np.interp(za, zs, u), np.interp(zb, zs, u)
        ch = ua + (ub - ua) * (zs[m] - za) / (zb - za)
        d = sign * (u[m] - ch)
        j = int(np.argmax(d))
        return 1000.0 * float(d[j]), float(zs[m][j])

    fy, by = np.asarray(o["front"], float), np.asarray(o["back"], float)
    f_mm, f_at = dev(fy, front[0], front[1], -1.0)
    lm = (zs >= lumbar[0]) & (zs <= lumbar[1]) & np.isfinite(by)
    lz = float(zs[lm][np.argmin(by[lm])])
    b_mm, b_at = dev(by, lz, back_low, 1.0)
    # the same on the thigh alone (groin to above the knee: the 0.62 end sits on the knee, whose taper bows the long chord)
    t_mm, t_at = dev(fy, 0.88, 0.70, -1.0)
    return dict(front=dict(dev_mm=f_mm, at=f_at, chord=list(front)), back=dict(dev_mm=b_mm, at=b_at, lumbar_z=lz, chord=[lz, back_low]),
                thigh_front=dict(dev_mm=t_mm, at=t_at, chord=[0.88, 0.70]))


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
    # --- the earlier shapes (tables.HIPS_BEFORE: one name or several, oldest first): outlines and measures only
    befores = tuple(tables.HIPS_BEFORE) if isinstance(tables.HIPS_BEFORE, (tuple, list)) else (tables.HIPS_BEFORE,)
    now_label = getattr(tables, "NOW_LABEL", "default")
    sil = getattr(sheet_ref, "SILHOUETTE", None)
    fronts, before, outs = [], {}, {}
    for shape in befores:
        report, rig, cam, floor = suitbuild.build(tables, shape, "", args.draft, args.samples)
        suitbuild.matte_figure()
        hide(DECOR)
        if sil:  # the matte front view, arms shown (the reference has them), for shape_vs_silhouette.jpg
            fronts.append((studio.compose(F.render(rig, cam, floor, previews, "silh"), ["front"]), shape))
        hide(DECOR + ARMS, ("Hands",))
        before[shape] = pure_panels(rig, cam, floor, previews, z0, z1)
        outs[shape] = measure(source, shape + "_")
    # --- the default
    report, rig, cam, floor = suitbuild.build(tables, "", "", args.draft, args.samples)
    out = measure(source, "")
    # the earlier shapes' half-gap tables and lens / chord / flare measures beside the default's, in source/outline.json
    out["before"] = {s: dict(half_gap=outs[s]["half_gap"], lens={k: outs[s]["lens"].get(k) for k in ("first_light", "tip_arch_mm", "convex_to")},
                             chords=outs[s]["chords"], flare=outs[s].get("flare")) for s in befores}
    with open(os.path.join(source, "outline.json"), "w") as f:
        json.dump(out, f)
    glossy = crotch_cells(cam, previews, CROTCH_GLOSS, "glossy suit")
    suitbuild.matte_figure()
    hide(DECOR)
    if sil:
        fronts.append((studio.compose(F.render(rig, cam, floor, previews, "silh"), ["front"]), now_label))
        m_ref = silhouette_picture(tables, sheet_ref, suitbuild, fronts, previews)
        lens_picture(tables, sheet_ref, suitbuild, rig, cam, floor, previews, out, m_ref)
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
    # hips_before_after.jpg: tables.HIPS_BEFORE above, the default below with the before's outline (blue) over
    # it, matte grey, pure form (arms hidden), front / three-quarter front / side / back, one scale
    hide(ARMS, ("Hands",))
    after = pure_panels(rig, cam, floor, previews, z0, z1)
    cols = {s: BEFORES[i % len(BEFORES)] for i, s in enumerate(befores)}
    names = {(0.20, 0.45, 0.95): "blue", (0.10, 0.68, 0.30): "green", (0.80, 0.20, 0.70): "magenta"}
    rows = []
    for shape in befores:
        rows.append([(studio.compose(before[shape], [name]), "%s, %s" % (shape, name)) for name, _ in BA_VIEWS])
    row_a = []
    for name, _ in BA_VIEWS:
        p = studio.compose(after, [name]).copy()
        for shape in befores:
            p[suitbuild.outline(before[shape][name][..., 3] > 0.5, 3)] = (*cols[shape], 1.0)
        row_a.append((p, "now, %s" % name))
    rows.append(row_a)
    w = out["worst"]
    key = ", ".join("%s: %s's outline" % (names.get(tuple(cols[s]), "line"), s) for s in befores)
    title = [tables.PREFIX.capitalize() + " - the pelvis: %s, then now (%s, bottom row), matte grey, waist to knee" % (", ".join(befores), now_label),
             "bottom row lines: %s. Worst turn deg/cm now (0.60-1.14 m, the waist included): front %.1f, side front %.1f, side back %.1f" % (
                 key, w["outer"]["deg_per_cm"], w["front"]["deg_per_cm"], w["back"]["deg_per_cm"])]
    imgs = [suitbuild.assemble_row(r, title if i == 0 else None) for i, r in enumerate(rows)]
    gap = np.ones((10, imgs[0].shape[1], 4), np.float32)
    suitbuild.save_image(np.concatenate(sum(([im, gap] for im in imgs[:-1]), []) + [imgs[-1]], axis=0),
                         os.path.join(previews, "hips_before_after.jpg"))
    # side_outlines.jpg: the side outlines of every shape and of the side drawing over the default's matte side view
    side_outlines_picture(tables, sheet_ref, suitbuild, after["side"], z0, z1, outs, out, cols, names, befores, now_label, previews)
    # hips.jpg: pure form, five views, orthographic, waist to knee
    hip_views(tables, suitbuild, rig, cam, floor, previews, z0, z1)


def _stamp(img, px, py, col, r=1.6):
    """Draw a polyline (pixel coordinates, float) onto img, `r` px thick."""
    h, w = img.shape[:2]
    pts = np.column_stack([px, py])
    pts = pts[np.isfinite(pts).all(axis=1)]
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        n = max(2, int(math.hypot(xb - xa, yb - ya) * 2) + 1)
        for s in np.linspace(0.0, 1.0, n):
            x, y = xa + s * (xb - xa), ya + s * (yb - ya)
            x0, x1, y0, y1 = int(max(0, x - r)), int(min(w, x + r + 1)), int(max(0, y - r)), int(min(h, y + r + 1))
            if x0 < x1 and y0 < y1:
                yy, xx = np.mgrid[y0:y1, x0:x1]
                m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
                img[y0:y1, x0:x1][m] = (*col, 1.0)


DRAWING = (0.15, 0.15, 0.15)  # the side drawing's outline (dark: the other colours are the shapes')


def side_outlines_picture(tables, sheet_ref, suitbuild, panel, z0, z1, outs, out, cols, names, befores, now_label, previews,
                          h_px=1000, half_w=0.24):
    """side_outlines.jpg: the default's matte side view (pure form, the band z0..z1) with the measured side outlines
    (front-most and back-most y every 2 mm, from the sliced mesh) of every earlier shape, of the default (orange) and
    the side drawing's outline (dark grey, its mask's front and back edge); the chord deviations in the title."""
    mpp = (z1 - z0) / h_px
    img = studio.compose({"s": panel}, ["s"]).copy()
    H, W = img.shape[:2]
    zc = 0.5 * (z0 + z1)
    alpha = panel[..., 3] > 0.5
    zs = np.asarray(out["z"])
    fy = np.array([np.nan if v is None else v for v in out["outline"]["front"]])
    by = np.array([np.nan if v is None else v for v in out["outline"]["back"]])
    row = lambda z: (H / 2.0) - (np.asarray(z) - zc) / mpp
    # which way the rig shows y: the sign that puts the default's measured front on its own silhouette's edge
    j = int(np.argmin(np.abs(zs - zc)))
    r = int(round(float(row(zs[j]))))
    cs = np.flatnonzero(alpha[r])
    sgn = 1.0
    if len(cs):
        left = (cs.min() - W / 2.0) * mpp
        sgn = 1.0 if abs(left - fy[j]) < abs(left + fy[j]) else -1.0
    colx = lambda y: W / 2.0 + sgn * np.asarray(y) / mpp
    band = (zs >= z0) & (zs <= z1)
    # the drawing: the side sheet's mask edges (front, back) per row, in metres, on the same axes
    Sd = sheet_ref.SIDE
    mk = Sd.mask()
    c0 = Sd.panel_slice("side")[0]
    dz, dyf, dyb = [], [], []
    for rr in range(int(Sd.foot_row - z1 / Sd.m_per_px), int(Sd.foot_row - z0 / Sd.m_per_px) + 1):
        cc = np.flatnonzero(mk[rr])
        if not len(cc):
            continue
        # the run that holds the figure's middle (the arm may stand off it in front or behind)
        runs = np.split(cc, np.flatnonzero(np.diff(cc) > 1) + 1)
        best = min(runs, key=lambda q: min(abs(q[0] - c0), abs(q[-1] - c0), 0 if q[0] <= c0 <= q[-1] else 1e9))
        dz.append((Sd.foot_row - rr) * Sd.m_per_px)
        dyf.append((best[0] - c0) * Sd.m_per_px)
        dyb.append((best[-1] + 1 - c0) * Sd.m_per_px)
    dz, dyf, dyb = np.array(dz), np.array(dyf), np.array(dyb)
    # the studio shows the side view facing image-left as the drawing does: screen-x = drawing's (col - centre)
    for yv in (dyf, dyb):
        _stamp(img, W / 2.0 + yv / mpp, row(dz), DRAWING, 1.4)
    for s in befores:
        o = outs[s]["outline"]
        for k in ("front", "back"):
            v = np.array([np.nan if q is None else q for q in o[k]])
            _stamp(img, colx(v[band]), row(zs[band]), cols[s], 1.4)
    for v in (fy, by):
        _stamp(img, colx(v[band]), row(zs[band]), AFTER, 1.6)
    # the largest separation of the default from the first earlier shape that is not stage 4 (else the last one)
    ref = [s for s in befores if s != "stage4"] or list(befores)
    o = outs[ref[-1]]["outline"]
    sep = []
    for k in ("front", "back"):
        v = np.array([np.nan if q is None else q for q in o[k]])
        d = np.abs((fy if k == "front" else by) - v) * 1000.0
        m = band & np.isfinite(d)
        i = int(np.nanargmax(np.where(m, d, np.nan)))
        sep.append((k, float(d[i]), float(zs[i])))
    ch, chb = out["chords"], {s: outs[s]["chords"] for s in befores}
    key = "; ".join("%s %s" % (names.get(tuple(cols[s]), "line"), s) for s in befores)
    title = [tables.PREFIX.capitalize() + " - the side outline, %.2f-%.2f m: %s; orange now (%s); dark grey the side drawing" % (z0, z1, key, now_label),
             "largest separation of now from %s: front %.1f mm at z %.3f, back %.1f mm at z %.3f. Thigh-front swell from the 0.90-0.62 chord: %s now %.1f mm;"
             " seat round from the lumbar-0.70 chord: %s now %.1f mm" % (
                 ref[-1], sep[0][1], sep[0][2], sep[1][1], sep[1][2],
                 " ".join("%s %.1f," % (s, chb[s]["front"]["dev_mm"]) for s in befores), ch["front"]["dev_mm"],
                 " ".join("%s %.1f," % (s, chb[s]["back"]["dev_mm"]) for s in befores), ch["back"]["dev_mm"])]
    suitbuild.save_image(suitbuild.assemble_row([(img, "side, matte (now)")], title), os.path.join(previews, "side_outlines.jpg"))
    print("side outline: largest separation from %s: %s" % (ref[-1], sep))


def lens_picture(tables, sheet_ref, suitbuild, rig, cam, floor, previews, out, m_ref, z0=0.76, z1=0.94, half_w=0.10, h_px=900):
    """lens.jpg: the leg gap from the front (matte, orthographic, the first 10 cm and more below the crotch) beside the
    owner's frontal reference (sheet_ref.SILHOUETTE) at ONE scale (its shoulder width = the model's, as in
    shape_vs_silhouette.jpg), the two aligned where light first shows through between the thighs."""
    si = sheet_ref.SILHOUETTE
    mpp = (z1 - z0) / h_px
    w_px = int(round(2 * half_w / mpp))
    p = studio.ortho_panels(rig, cam, floor, previews, [("lens", 0.0, w_px)], mpp, h_px, 0.5 * (z0 + z1), "lens")
    mine = studio.compose(p, ["lens"])
    fl = out["lens"]["first_light"] or tables.dims()["crotch_z"]
    k = m_ref / mpp
    ref = suitbuild.load_scaled(si["path"], k)
    pan = np.ones((h_px, w_px, 4), np.float32)
    pan[..., :3] = studio.to_srgb(studio.BACKDROP)
    oy = int(round(h_px / 2.0 - (fl - 0.5 * (z0 + z1)) / mpp - si["light_row"] * k))
    ox = int(round(w_px / 2.0 - si["centre"] * k))
    rh, rw = ref.shape[:2]
    y0, y1, x0, x1 = max(oy, 0), min(oy + rh, h_px), max(ox, 0), min(ox + rw, w_px)
    pan[y0:y1, x0:x1, :3] = ref[y0 - oy:y1 - oy, x0 - ox:x1 - ox, :3]
    cells = [(pan, "the owner's front reference (its frame ends 1 cm under its first light)"), (mine, "now, matte front")]
    title = [tables.PREFIX.capitalize() + " - the leg gap from the front, %.2f-%.2f m, one scale (shoulder widths equal)" % (z0, z1),
             "aligned where light first shows between the thighs: the model at z %.3f m" % fl]
    suitbuild.save_image(suitbuild.assemble_row(cells, title), os.path.join(previews, "lens.jpg"))


def silhouette_picture(tables, sheet_ref, suitbuild, fronts, previews, z0=0.80, z1=1.52):
    """shape_vs_silhouette.jpg: sheet_ref.SILHOUETTE (a frontal body reference, not a sheet) beside the model's
    matte front views (`fronts` = [(front sheet image, label)]) at ONE scale: the reference scaled so its
    shoulder width (outer deltoids, `shoulder_hw` px) equals the model's (measured on the first front view's
    mask, as suitbuild.proportions measures it), its waist row aligned with the model's waist height."""
    F, si = sheet_ref.SHEET, sheet_ref.SILHOUETTE
    mask = fronts[0][0]
    c = F.panel_slice("front")[0] - F.panel_slice("front")[1].start
    mine = np.abs(mask[..., :3] - studio.to_srgb(studio.BACKDROP)).max(axis=-1) > 0.02
    prop = suitbuild.proportions(mine, F, c)
    k = (0.5 * prop["shoulder"] / F.m_per_px) / si["shoulder_hw"]   # front-sheet px per reference px
    ref = suitbuild.load_scaled(si["path"], k)
    h_img, w_img = F.size[1], F.size[0]
    pan = np.ones((h_img, w_img, 4), np.float32)
    pan[..., :3] = studio.to_srgb(studio.BACKDROP)
    wrow = int(round(F.foot_row - si["waist_z"] / F.m_per_px))
    ox, oy = int(round(c - si["centre"] * k)), int(round(wrow - si["waist_row"] * k))
    rh, rw = ref.shape[:2]
    y0, y1, x0, x1 = max(oy, 0), min(oy + rh, h_img), max(ox, 0), min(ox + rw, w_img)
    pan[y0:y1, x0:x1, :3] = ref[y0 - oy:y1 - oy, x0 - ox:x1 - ox, :3]
    hw = 0.30
    cells = [(suitbuild.band_crop(F, pan, z0, z1, hw), "the owner's front reference (good-body-silhouette)")]
    cells += [(suitbuild.band_crop(F, img, z0, z1, hw), lab) for img, lab in fronts]
    title = [tables.PREFIX.capitalize() + " - body shape against the owner's frontal reference, matte grey, %.2f-%.2f m" % (z0, z1),
             "one scale: the reference's shoulder width = the model's (%.3f m), its narrowest waist on the model's (%.2f m)" % (prop["shoulder"], si["waist_z"])]
    suitbuild.save_image(suitbuild.assemble_row(cells, title), os.path.join(previews, "shape_vs_silhouette.jpg"))
    return 0.5 * prop["shoulder"] / si["shoulder_hw"]   # metres per reference pixel at this scale


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


BA_VIEWS = (("front", 0.0), ("three-quarter front", 40.0), ("side", 90.0), ("back", 180.0))


def pure_panels(rig, cam, floor, previews, z0, z1, h_px=1000, half_w=0.24):
    """{view name: RGBA} orthographic, BA_VIEWS, the band z0..z1 (for hips_before_after.jpg)."""
    mpp = (z1 - z0) / h_px
    w_px = int(round(2 * half_w / mpp))
    return studio.ortho_panels(rig, cam, floor, previews, [(n, az, w_px) for n, az in BA_VIEWS], mpp, h_px, 0.5 * (z0 + z1), "ba")


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
