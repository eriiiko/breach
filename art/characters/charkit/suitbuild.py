"""The build driver for bodysuit characters (Blender 4.5): `charkit/bodysuit.py` from a table module.

A character's `scripts/build.py` is a few lines: put `charkit` on the path, import its tables
module and its `sheet_ref`, call `run(...)`:

    blender -b --factory-startup -P <name>/scripts/build.py -- [options]
    (or, from art/characters:  bash charkit/run.sh <name> [options])

    --draft          coarse grid + few samples (fast iteration)
    --sheet          orthographic view(s) in the reference's framing, the reference above the
                     model, a silhouette overlay, IoU per view AND per height band (tables.BANDS),
                     the per-height mismatch table, and the proportion ratios (shoulder, waist,
                     hip, thigh) measured the same way on the reference and on the model
    --turn           orthographic turnaround (front, side, back, three-quarter)
    --beauty a,b,..  perspective studio renders (tables.BEAUTY; `all` for every one)
    --variant NAME   a palette variant (tables.VARIANTS); renders get a NAME_ prefix
    --shape NAME     a shape variant (tables.SHAPES); outputs get a NAME_ prefix
    --compare        build the default shape AND every shape in tables.SHAPES, and write
                     shape_vs_concept.jpg: the concept, the reference and each build's front
                     view at the reference's scale (needs sheet_ref.CONCEPT_PATH / CONCEPT_FRAME);
                     with tables.COMPARE (shapes + labels) and sheet_ref.SIDE: the shapes in matte
                     grey, front and side, zoomed on tables.COMPARE_BAND, the drawings' outlines
                     over them and the concept beside the front row, every panel labelled
    --scale S        beauty resolution scale (default 1.0)
    --samples N      Cycles samples (default 96, draft 24)
    --save           write source/<name>.blend and source/mesh_stats.json
    --outline        slice the built body and print its outlines' worst direction change per cm, waist
                     to knee, per view (`hipsview.measure`; source/<prefix>outline.json)
    --hips           the hip pictures (hips.jpg, hips_vs_drawings.jpg, hips_before_after.jpg; needs
                     tables.HIPS_BEFORE, tables.COMPARE_BAND and sheet_ref.SIDE / BACK): `hipsview.pictures`
"""
import argparse
import json
import os
import sys
import time

import bpy
import numpy as np

import bodysuit
import garment
import kit
import sheetfit
import studio


def build(tables, shape, variant, draft, samples):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    kit.RES = 0.008 if draft else 0.004
    M = bodysuit.materials(tables.palette(variant or None), tables.GLOSS, tables.PREFIX)
    bodysuit.build(M, tables.dims(shape or None), tables.HEAD, tables.PREFIX.capitalize())
    report = garment.orient_outward()
    rig, cam, floor = studio.setup(samples=samples or (24 if draft else 96), backdrop=studio.BACKDROP)
    return report, rig, cam, floor


# ------------------------------------------------------------------ measuring
def _centre_runs(row, c):
    """The runs either side of column c (or the one covering it): (left run, right run)."""
    runs = sheetfit.row_runs(row, min_len=3, max_gap=2)
    cover = [r for r in runs if r[0] <= c < r[1]]
    if cover:
        return cover[0], cover[0]
    left = [r for r in runs if r[1] <= c]
    right = [r for r in runs if r[0] > c]
    return (left[-1] if left else None), (right[0] if right else None)


def proportions(mask, sheet, c):
    """Shoulder (max over 1.30-1.40 m), waist (min over 1.10-1.20), hip (max outer width over
    0.84-1.10, between the runs nearest the centre line: below the crotch the two legs), thigh
    (the right leg's run 0.035 m below the crotch) in metres, from a front-view mask in the
    sheet's frame, and their ratios. The same procedure runs on the reference and the model."""
    mpp = sheet.m_per_px
    row = lambda z: int(round(sheet.foot_row - z / mpp))

    def width(z):
        lr = _centre_runs(mask[row(z)], c)
        if lr[0] is None or lr[1] is None:
            return 0.0
        return (lr[1][1] - lr[0][0]) * mpp

    zs = lambda a, b: np.arange(a, b, 0.004)
    shoulder = max(width(z) for z in zs(1.30, 1.40))
    waist = min(width(z) for z in zs(1.10, 1.20))
    hip_z = max(zs(0.84, 1.10), key=width)
    hip = width(hip_z)
    crotch = next(z for z in zs(0.80, 1.05) if mask[row(z), c - 1:c + 2].any())  # lowest z where the centre is filled
    r = _centre_runs(mask[row(crotch - 0.035)], c)[1]
    thigh = (r[1] - r[0]) * mpp if r else 0.0
    return dict(shoulder=shoulder, waist=waist, hip=hip, hip_z=float(hip_z), crotch_z=float(crotch), thigh=thigh,
                hip_shoulder=hip / shoulder, hip_waist=hip / waist, thigh_hip=thigh / hip)


def band_iou(rm, mine, sheet, bands):
    out = {}
    for name, z0, z1 in bands:
        r0, r1 = int(round(sheet.foot_row - z1 / sheet.m_per_px)), int(round(sheet.foot_row - z0 / sheet.m_per_px))
        a, b = rm[max(r0, 0):r1], mine[max(r0, 0):r1]
        out[name] = float((a & b).sum() / max(1, (a | b).sum()))
    return out


def save_image(arr, path, quality=92):
    h, w = arr.shape[:2]
    img = bpy.data.images.new("_out", w, h, alpha=True)
    img.pixels.foreach_set(np.ascontiguousarray(arr[::-1]).astype(np.float32).ravel())
    img.filepath_raw = path
    img.file_format = "JPEG" if path.lower().endswith(".jpg") else "PNG"
    sc = bpy.context.scene
    sc.render.image_settings.quality = quality
    img.save()
    bpy.data.images.remove(img)


def load_scaled(path, k):
    """An image scaled by k, float RGBA, row 0 at the top."""
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    img.scale(max(1, int(round(w * k))), max(1, int(round(h * k))))
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)[::-1].copy()


def sheet_report(tables, sheet_ref, panels, previews, tag, verbose):
    S = sheet_ref.SHEET
    S.compare(panels, previews, tag, verbose=verbose)
    cx, sl = S.panel_slice("front")
    rm = S.mask()[:, sl]
    mine = panels["front"][..., 3] > 0.5
    bands = band_iou(rm, mine, S, tables.BANDS)
    c = cx - sl.start
    ref_p, mod_p = proportions(rm, S, c), proportions(mine, S, c)
    print("bands IoU:", json.dumps({k: round(v, 3) for k, v in bands.items()}))
    for k in ("shoulder", "waist", "hip", "hip_z", "crotch_z", "thigh", "hip_shoulder", "hip_waist", "thigh_hip"):
        print("  %-13s reference %.3f   model %.3f" % (k, ref_p[k], mod_p[k]))
    if hasattr(tables, "CONCEPT"):
        for k, v in tables.CONCEPT.items():
            print("  concept %-12s %.3f   model %.3f (%+.1f%%)" % (k, v, mod_p[k], 100.0 * (mod_p[k] / v - 1.0)))
    return dict(bands=bands, reference=ref_p, model=mod_p)


def view_report(tables, S, panels, previews, tag, verbose):
    """A further view (side, back) against its own drawing: overlay, `<tag>_vs_reference.png`, IoU
    whole and per height band (tables.BANDS)."""
    name = S.panels[0][0]
    scores = S.compare(panels, previews, tag, verbose=verbose)
    rm = S.mask()[:, S.panel_slice(name)[1]]
    bands = band_iou(rm, panels[name][..., 3] > 0.5, S, tables.BANDS)
    print("%s bands IoU:" % tag, json.dumps({k: round(v, 3) for k, v in bands.items()}))
    return dict(iou=scores[name], bands=bands)


def run(root, name, tables, sheet_ref):
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--turn", action="store_true")
    ap.add_argument("--beauty", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--shape", default="")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--outline", action="store_true")
    ap.add_argument("--hips", action="store_true")
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    t0 = time.time()
    previews, source = os.path.join(root, "previews"), os.path.join(root, "source")
    os.makedirs(previews, exist_ok=True)
    os.makedirs(source, exist_ok=True)
    prefix = (args.shape + "_" if args.shape else "") + (args.variant + "_" if args.variant else "")

    if args.hips:
        import hipsview
        hipsview.pictures(tables, sheet_ref, previews, source, args, sys.modules[__name__])
        print("done in %.1fs" % (time.time() - t0))
        return
    if args.compare:
        (compare_shapes if hasattr(tables, "COMPARE") else compare)(tables, sheet_ref, previews, args)
        print("done in %.1fs" % (time.time() - t0))
        return

    report, rig, cam, floor = build(tables, args.shape, args.variant, args.draft, args.samples)
    print("built in %.1fs" % (time.time() - t0))
    if args.outline:
        import hipsview
        hipsview.measure(source, prefix)
    if args.save:
        path = os.path.join(source, prefix + "mesh_stats.json")
        stats = studio.mesh_stats(path)
        stats["NORMALS"] = dict(flipped=report["flipped"], closed=report["closed"], open=report["open"], undecided=report["undecided"])
        with open(path, "w") as f:
            json.dump(stats, f, indent=1)
        print("mesh stats:", json.dumps(stats["TOTAL"]))
        if not prefix:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(source, name + ".blend"))
    if args.sheet:
        panels = sheet_ref.SHEET.render(rig, cam, floor, previews, prefix + "sheet")
        res = sheet_report(tables, sheet_ref, panels, previews, prefix + "sheet", not args.quiet)
        res["views"] = {}
        for tag, S in getattr(sheet_ref, "VIEWS", ()):  # further views, each its own drawing
            res["views"][tag] = view_report(tables, S, S.render(rig, cam, floor, previews, prefix + tag), previews, prefix + tag,
                                            not args.quiet)
        with open(os.path.join(source, prefix + "sheet_scores.json"), "w") as f:
            json.dump(res, f, indent=1)
    if args.turn:
        studio.turnaround(rig, cam, floor, previews, tables.HEIGHT, tag=prefix + "turnaround")
    names = list(tables.BEAUTY) if args.beauty == "all" else [n for n in args.beauty.split(",") if n]
    for n in names:
        view = tables.BEAUTY[n] if n in tables.BEAUTY else tables.DEV_VIEWS[n]  # DEV_VIEWS: close-ups on request, never in `all`
        if isinstance(view, dict):
            render_aimed(cam, view, prefix + n, previews, args.scale)
        else:
            studio.render_beauty(rig, cam, view, prefix + n, previews, args.scale)
    print("done in %.1fs" % (time.time() - t0))


def render_aimed(cam, view, name, out_dir, scale=1.0):
    """A close-up aimed at any point (studio.render_beauty always aims at the centre line):
    view = dict(cam=(x, y, z), target=(x, y, z), lens=mm, res=(w, h)); the studio's lights
    stay where they are."""
    cam.data.type = "PERSP"
    cam.data.lens = view.get("lens", 85.0)
    cam.data.clip_start, cam.data.clip_end = 0.01, 60.0
    cam.location = view["cam"]
    studio._look_at(cam, view["target"])
    res = view.get("res", (900, 900))
    studio.render(os.path.join(out_dir, name + ".jpg"), (int(res[0] * scale), int(res[1] * scale)))


def compare(tables, sheet_ref, previews, args):
    """shape_vs_concept.jpg: concept | reference | default build | each SHAPES build, one scale."""
    S = sheet_ref.SHEET
    h_img = S.size[1]
    cols = []
    cf = sheet_ref.CONCEPT_FRAME
    k = 1.0 / cf["scale"]
    con = load_scaled(sheet_ref.CONCEPT_PATH, k)
    panel = np.ones((h_img, S.size[0], 4), np.float32)
    panel[..., :3] = studio.to_srgb(studio.BACKDROP)
    ox = int(round(S.panel_slice("front")[0] - cf["waist"][0] * k))
    oy = int(round(cf["sheet_waist_row"] - cf["waist"][1] * k))
    ch, cw = con.shape[:2]
    y0, y1, x0, x1 = max(oy, 0), min(oy + ch, h_img), max(ox, 0), min(ox + cw, S.size[0])
    panel[y0:y1, x0:x1, :3] = con[y0 - oy:y1 - oy, x0 - ox:x1 - ox, :3]
    cols.append(panel)
    ref = S.image().copy()
    ref[..., 3] = 1.0
    cols.append(ref[:, S.panel_slice("front")[1]])
    for shape in [""] + list(getattr(tables, "SHAPES", {})):
        report, rig, cam, floor = build(tables, shape, args.variant, args.draft, args.samples)
        panels = S.render(rig, cam, floor, previews, "cmp")
        cols.append(studio.compose(panels, ["front"]))
        print("compare: built", shape or "default")
    gap = np.ones((h_img, 12, 4), np.float32)
    row = []
    for c in cols:
        row += [c, gap]
    save_image(np.concatenate(row[:-1], axis=1), os.path.join(previews, "shape_vs_concept.jpg"))


# ------------------------------------------------------------------ the shape comparison
MATTE = (0.30, 0.30, 0.31)          # linear: a neutral matte grey, so no gloss hides the form
OUTLINE = (1.0, 0.42, 0.05)         # sRGB: the drawing's outline, a contrasting orange
INK = (0.08, 0.08, 0.09)


def text_rgba(text, w, h, size=22):
    """A label as an RGBA array (white text, alpha), drawn with Blender's font into an image."""
    import blf
    import imbuf
    ib = imbuf.new((w, h))
    with blf.bind_imbuf(0, ib):
        blf.size(0, size)
        blf.color(0, 1.0, 1.0, 1.0, 1.0)
        blf.position(0, 8, int(h * 0.30), 0)
        blf.draw_buffer(0, text)
    path = os.path.join(bpy.app.tempdir or os.environ.get("TEMP", "."), "_label.png")
    imbuf.write(ib, filepath=path)
    a = studio.load_rgba(path)
    os.remove(path)
    return a


def label_strip(text, w, h=40, bg=(0.93, 0.93, 0.92)):
    t = text_rgba(text, w, h)
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = bg
    a = t[..., 3:4]
    out[..., :3] = out[..., :3] * (1 - a) + np.array(INK) * a
    return out


def outline(mask, width=2):
    """The edge pixels of a mask, thickened to `width`."""
    e = np.zeros_like(mask)
    e[1:] |= mask[1:] != mask[:-1]
    e[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    for _ in range(width - 1):
        g = e.copy()
        g[1:] |= e[:-1]
        g[:, 1:] |= e[:, :-1]
        e = g
    return e


def matte_figure():
    """Every part of the suit and boots in one neutral matte grey (the head and hands keep skin)."""
    m = bpy.data.materials.new("_compare_matte")
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*MATTE, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.62
    for ob in bpy.data.objects:
        if ob.type == "MESH" and ob.users_collection and ob.users_collection[0].name in ("Suit", "Boots"):
            for i in range(len(ob.data.materials)):
                ob.data.materials[i] = m


def band_crop(S, img, z0, z1, half_w):
    """Rows z0..z1 (metres) and columns +-half_w (metres) round the panel's centre, in the sheet's frame."""
    r0, r1 = int(round(S.foot_row - z1 / S.m_per_px)), int(round(S.foot_row - z0 / S.m_per_px))
    cx = S.panels[0][1] - S.panel_slice(S.panels[0][0])[1].start
    hw = int(round(half_w / S.m_per_px))
    return img[r0:r1, max(cx - hw, 0):cx + hw]


def concept_panel(sheet_ref, F):
    """The concept placed at the front view's scale, over the backdrop, in the front sheet's frame."""
    cf = sheet_ref.CONCEPT_FRAME
    k = 1.0 / cf["scale"]
    con = load_scaled(sheet_ref.CONCEPT_PATH, k)
    h_img = F.size[1]
    cpan = np.ones((h_img, F.size[0], 4), np.float32)
    cpan[..., :3] = studio.to_srgb(studio.BACKDROP)
    ox = int(round(F.panel_slice("front")[0] - cf["waist"][0] * k))
    oy = int(round(cf["sheet_waist_row"] - cf["waist"][1] * k))
    ch, cw = con.shape[:2]
    y0, y1, x0, x1 = max(oy, 0), min(oy + ch, h_img), max(ox, 0), min(ox + cw, F.size[0])
    cpan[y0:y1, x0:x1, :3] = con[y0 - oy:y1 - oy, x0 - ox:x1 - ox, :3]
    return cpan


def assemble_row(cells, title=None):
    """[(RGBA image, label), ...] side by side, each under its label, a title strip above."""
    h = max(p.shape[0] for p, _ in cells)
    row = []
    for p, lab in cells:
        cell = np.ones((h, p.shape[1], 4), np.float32)
        cell[..., :3] = studio.to_srgb(studio.BACKDROP)
        cell[:p.shape[0]] = p
        row += [np.concatenate([label_strip(lab, p.shape[1]), cell], axis=0), np.ones((h + 40, 10, 4), np.float32)]
    img = np.concatenate(row[:-1], axis=1)
    for t in reversed([title] if isinstance(title, str) else (title or [])):
        img = np.concatenate([label_strip(t, img.shape[1], 44), img], axis=0)
    return img


def compare_shapes(tables, sheet_ref, previews, args):
    """shape_vs_concept.jpg: each shape in tables.COMPARE built from its table rows, shown in a matte
    grey, front AND side, zoomed on waist to knee (tables.COMPARE_BAND), the drawing's outline over
    each (orange), the concept beside the front row, every panel labelled."""
    z0, z1 = tables.COMPARE_BAND
    F, Sd = sheet_ref.SHEET, sheet_ref.SIDE
    fw, sw = 0.27, 0.20
    f_out = band_crop(F, outline(F.mask()[:, F.panel_slice("front")[1]]), z0, z1, fw)
    s_out = band_crop(Sd, outline(Sd.mask()[:, Sd.panel_slice("side")[1]]), z0, z1, sw)
    # the concept, placed at the front view's scale (as in the stage-1 comparison)
    cf = sheet_ref.CONCEPT_FRAME
    k = 1.0 / cf["scale"]
    con = load_scaled(sheet_ref.CONCEPT_PATH, k)
    h_img = F.size[1]
    cpan = np.ones((h_img, F.size[0], 4), np.float32)
    cpan[..., :3] = studio.to_srgb(studio.BACKDROP)
    ox = int(round(F.panel_slice("front")[0] - cf["waist"][0] * k))
    oy = int(round(cf["sheet_waist_row"] - cf["waist"][1] * k))
    ch, cw = con.shape[:2]
    y0, y1, x0, x1 = max(oy, 0), min(oy + ch, h_img), max(ox, 0), min(ox + cw, F.size[0])
    cpan[y0:y1, x0:x1, :3] = con[y0 - oy:y1 - oy, x0 - ox:x1 - ox, :3]
    front_row = [(band_crop(F, cpan, z0, z1, fw), "the concept (original artwork)")]
    side_row = []
    for shape, label in tables.COMPARE:
        report, rig, cam, floor = build(tables, shape, args.variant, args.draft, args.samples)
        matte_figure()
        for S, view, row, out, hw in ((F, "front", front_row, f_out, fw), (Sd, "side", side_row, s_out, sw)):
            p = studio.compose(S.render(rig, cam, floor, previews, "cmp"), [view])
            p = band_crop(S, p, z0, z1, hw).copy()
            p[out[:p.shape[0], :p.shape[1]]] = (*OUTLINE, 1.0)
            row.append((p, "%s, %s" % (label, view)))
        print("compare: built", shape or "default")
    # the side row has no concept: a key in its place
    key = np.ones_like(side_row[0][0])
    key[..., :3] = (0.93, 0.93, 0.92)
    lines = ["matte grey: the model", "orange line: the drawing's", "outline (front / side view)", "waist to knee, %.2f-%.2f m" % (z0, z1)]
    for i, ln in enumerate(lines):
        t = text_rgba(ln, key.shape[1], 40, 24)
        r = 60 + 46 * i
        a = t[..., 3:4]
        key[r:r + 40, :, :3] = key[r:r + 40, :, :3] * (1 - a) + np.array(INK) * a
    side_row.insert(0, (key, "key"))

    def assemble(row):
        h = max(p.shape[0] for p, _ in row)
        cells = []
        for p, lab in row:
            cell = np.ones((h, p.shape[1], 4), np.float32)
            cell[..., :3] = studio.to_srgb(studio.BACKDROP)
            cell[:p.shape[0]] = p
            cells += [np.concatenate([label_strip(lab, p.shape[1]), cell], axis=0), np.ones((h + 40, 10, 4), np.float32)]
        return np.concatenate(cells[:-1], axis=1)

    fr, sr = assemble(front_row), assemble(side_row)
    w = max(fr.shape[1], sr.shape[1])
    pad = lambda a: np.concatenate([a, np.ones((a.shape[0], w - a.shape[1], 4), np.float32)], axis=1) if a.shape[1] < w else a
    title = label_strip(tables.PREFIX.capitalize() + " - hip shapes from table values (front row: front view; second row: side view)", w, 44)
    img = np.concatenate([title, pad(fr), np.ones((10, w, 4), np.float32), pad(sr)], axis=0)
    save_image(img, os.path.join(previews, "shape_vs_concept.jpg"))
