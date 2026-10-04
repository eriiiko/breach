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
"""
import json
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
    out = dict(worst={k: dict(deg_per_cm=v, z=z) for k, (v, z) in res.items()}, z=zs.tolist(),
               outline={k: [None if not np.isfinite(x) else float(x) for x in a] for k, a in o.items()},
               turning={k: [None if not np.isfinite(x) else float(x) for x in a] for k, a in tr.items()})
    with open(os.path.join(source, prefix + "outline.json"), "w") as f:
        json.dump(out, f)
    return out


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
    suitbuild.matte_figure()
    hide(DECOR)
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
    title = [tables.PREFIX.capitalize() + " - hip outlines: blue = before (stage 3), orange = now",
             "worst turn deg/cm now: front %.1f, side front %.1f, side back %.1f" % (
                 w["outer"]["deg_per_cm"], w["front"]["deg_per_cm"], w["back"]["deg_per_cm"])]
    suitbuild.save_image(suitbuild.assemble_row(cells, title), os.path.join(previews, "hips_before_after.jpg"))
    # hips.jpg: pure form, five views, orthographic, waist to knee
    hip_views(tables, suitbuild, rig, cam, floor, previews, z0, z1)


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
