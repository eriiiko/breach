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
                     view at the reference's scale (needs sheet_ref.CONCEPT_PATH / CONCEPT_FRAME)
    --scale S        beauty resolution scale (default 1.0)
    --samples N      Cycles samples (default 96, draft 24)
    --save           write source/<name>.blend and source/mesh_stats.json
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
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

    t0 = time.time()
    previews, source = os.path.join(root, "previews"), os.path.join(root, "source")
    os.makedirs(previews, exist_ok=True)
    os.makedirs(source, exist_ok=True)
    prefix = (args.shape + "_" if args.shape else "") + (args.variant + "_" if args.variant else "")

    if args.compare:
        compare(tables, sheet_ref, previews, args)
        print("done in %.1fs" % (time.time() - t0))
        return

    report, rig, cam, floor = build(tables, args.shape, args.variant, args.draft, args.samples)
    print("built in %.1fs" % (time.time() - t0))
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
        with open(os.path.join(source, prefix + "sheet_scores.json"), "w") as f:
            json.dump(res, f, indent=1)
    if args.turn:
        studio.turnaround(rig, cam, floor, previews, tables.HEIGHT, tag=prefix + "turnaround")
    names = list(tables.BEAUTY) if args.beauty == "all" else [n for n in args.beauty.split(",") if n]
    for n in names:
        view = tables.BEAUTY[n] if n in tables.BEAUTY else tables.DEV_VIEWS[n]  # DEV_VIEWS: close-ups on request, never in `all`
        studio.render_beauty(rig, cam, view, prefix + n, previews, args.scale)
    print("done in %.1fs" % (time.time() - t0))


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
