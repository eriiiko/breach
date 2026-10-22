"""Lay out the VRoid spike's pictures (plain Python: numpy + PIL) from the raw renders of `vroid.py`.

    python Beatrice/scripts/vroid_sheets.py [raw dir] [--only candidates,compare,face,full]   (default $TEMP/Beatrice_vroid)

vroid_candidates.jpg   every candidate head, front and 30 deg, as imported (Cycles) and in the toon look, beside the drawings
vroid_vs_concept.jpg   the four references | B (the whole VRoid figure, posed) | A (if rendered) | the merged B2e build
vroid_face.jpg         the face front / 30 / 60 / profile: the drawings, B, the merged B2e build
vroid_full.jpg         the full figure front / side / back (B; A if rendered)

Only the four references the owner named are shown: concept.jfif, concept-face-front-v1.png, concept-face-side-v1.png,
concept-upper-body-side-v2.png.
"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
PREV = os.path.join(ROOT, "previews")
BACK = (161, 164, 171)
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]


def _font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def flat(path, size=None, crop=None):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, BACK + (255,))
    out = Image.alpha_composite(bg, im).convert("RGB")
    if crop:
        out = out.crop(crop)
    return fit(out, size) if size else out


def fit(im, size):
    """Scale to fit inside `size`, centred on a backdrop of exactly `size`."""
    im = im.copy()
    im.thumbnail(size, Image.LANCZOS)
    out = Image.new("RGB", size, BACK)
    out.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return out


def label(img, text, h=34, fs=19):
    strip = Image.new("RGB", (img.width, h), (238, 238, 235))
    ImageDraw.Draw(strip).text((8, 7), text, fill=(20, 20, 20), font=_font(fs))
    out = Image.new("RGB", (img.width, img.height + h), (255, 255, 255))
    out.paste(strip, (0, 0))
    out.paste(img, (0, h))
    return out


def row(cells, gap=6):
    h = max(c.height for c in cells)
    out = Image.new("RGB", (sum(c.width for c in cells) + gap * (len(cells) - 1), h), (255, 255, 255))
    x = 0
    for c in cells:
        out.paste(c, (x, 0))
        x += c.width + gap
    return out


def col(rows, gap=6):
    w = max(r.width for r in rows)
    out = Image.new("RGB", (w, sum(r.height for r in rows) + gap * (len(rows) - 1)), (255, 255, 255))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height + gap
    return out


def ref(name, size, crop=None):
    im = Image.open(os.path.join(ROOT, name)).convert("RGB")
    if crop:
        im = im.crop(crop)
    return fit(im, size)


def candidates(raw):
    import vroid_tables as T
    cell = (300, 375)
    head = [label(ref("concept-face-front-v1.png", cell, (130, 0, 1124, 1254)), "drawing: front"),
            label(ref("concept-face-side-v1.png", cell, (130, 0, 1124, 1254)), "drawing: profile"),
            label(ref("concept.jfif", cell, (180, 0, 860, 860)), "concept.jfif (original)")]
    rows = [row(head)]
    for name, _, cc0 in T.CANDIDATES:
        cells = []
        for kind in ("neutral", "toon"):
            for az in (0, 30):
                p = os.path.join(raw, "cand_%s_%s_%03d.png" % (name, kind, az))
                if os.path.exists(p):
                    cells.append(label(flat(p, cell), "%s %s %d deg" % (name if az == 0 and kind == "neutral" else "", kind, az)))
        if cells:
            rows.append(label(row(cells), "%s -- %s" % (name, T.LICENCE_NOTE[cc0]), 30, 18))
    col(rows).save(os.path.join(PREV, "vroid_candidates.jpg"), quality=88)


def cell_of(path, x0, y0, w, h):
    """A cell of a laid-out preview (the merged B2e build's sheets), its label strip cut off."""
    im = Image.open(path).convert("RGB").crop((x0, y0, x0 + w, y0 + h))
    a = np.asarray(im).astype(int)
    top = 0
    lab = np.array([238, 238, 235])
    while top < h // 4 and (np.abs(np.median(a[top], 0) - lab).max() < 6 or np.median(a[top]) > 245):
        top += 1   # the label strip (its text does not stop it: the row's median stays the strip's colour) and the gap
    return im.crop((0, top, w, h))


def outline_over(base, rgba_path, colour=(255, 40, 40)):
    """The render's silhouette edge drawn over `base` (same size)."""
    a = Image.open(rgba_path).convert("RGBA").split()[3].resize(base.size, Image.LANCZOS)
    m = (np.asarray(a) > 128).astype(np.uint8) * 255
    mi = Image.fromarray(m)
    edge = np.asarray(mi.filter(ImageFilter.MaxFilter(5))).astype(int) - np.asarray(mi.filter(ImageFilter.MinFilter(3))).astype(int)
    out = np.asarray(base.convert("RGB")).copy()
    out[edge > 0] = colour
    return Image.fromarray(out)


def compare(raw):
    """vroid_vs_concept.jpg: per reference, the drawing | B in its framing | B's outline over it | the merged B2e build."""
    cell = (512, 768)
    r1 = Image.open(os.path.join(ROOT, "concept.jfif")).convert("RGB")
    r2 = Image.open(os.path.join(ROOT, "concept-upper-body-side-v2.png")).convert("RGB")
    b1, b2 = os.path.join(raw, "P_concept.png"), os.path.join(raw, "S_side_v2.png")
    e1 = cell_of(os.path.join(PREV, "posed_vs_concept.jpg"), 1032, 0, 1024, 1580)
    side = cell_of(os.path.join(PREV, "posed_full.jpg"), 1816, 0, 900, 1544)
    e2 = side.crop((0, 0, side.width, int(side.height * 0.62)))
    rows = [row([label(fit(r1, cell), "concept.jfif (original)"), label(flat(b1, cell), "B: VRoid figure, concept pose, toon"),
                 label(fit(outline_over(r1, b1), cell), "B's outline over the concept"),
                 label(fit(e1, cell), "merged scratch build (B2e), same framing")]),
            row([label(fit(r2, cell), "concept-upper-body-side-v2.png"), label(flat(b2, cell), "B: from her right, standing"),
                 label(fit(outline_over(r2, b2), cell), "B's outline over the drawing"),
                 label(fit(e2, cell), "B2e: posed_full side (other framing)")])]
    col(rows).save(os.path.join(PREV, "vroid_vs_concept.jpg"), quality=88)


def face(raw):
    """vroid_face.jpg: the close-up drawings; B in their framings and at 0 / 30 / 60 / 90 deg; the merged B2e build."""
    cell = (400, 400)
    tall = (400, 500)
    df = Image.open(os.path.join(ROOT, "concept-face-front-v1.png")).convert("RGB")
    ds = Image.open(os.path.join(ROOT, "concept-face-side-v1.png")).convert("RGB")
    bf, bs = os.path.join(raw, "S_art_front.png"), os.path.join(raw, "S_art_side.png")
    rows = [row([label(fit(df, cell), "drawing: front"), label(flat(bf, cell), "B, the drawing's framing"),
                 label(fit(outline_over(df, bf), cell), "B's outline over it"),
                 label(fit(ds, cell), "drawing: profile"), label(flat(bs, cell), "B, the drawing's framing"),
                 label(fit(outline_over(ds, bs), cell), "B's outline over it")])]
    fva = os.path.join(PREV, "face_vs_art.jpg")
    rows.append(row([label(fit(df, cell), "drawing: front"), label(fit(cell_of(fva, 420, 0, 420, 420), cell), "B2e, front (face_vs_art)"),
                     label(fit(cell_of(fva, 840, 0, 420, 420), cell), "B2e's outline over it"),
                     label(fit(ds, cell), "drawing: profile"), label(fit(cell_of(fva, 1680, 0, 420, 420), cell), "B2e, profile"),
                     label(fit(cell_of(fva, 2100, 0, 420, 420), cell), "B2e's outline over it")]))
    import vroid_tables as T
    rows.append(label(row([label(flat(os.path.join(raw, "S_face_%03d.png" % a), tall), "B, %d deg" % a) for a in T.FACE_VIEWS]),
                      "B: the VRoid face (HairSample_Female), expression and lower face by script", 30, 18))
    ft = os.path.join(PREV, "face_turn.jpg")
    turn = [cell_of(ft, 0, 0, 540, 675), cell_of(ft, 540, 0, 540, 675), cell_of(ft, 1080, 0, 540, 675), cell_of(ft, 0, 675, 540, 675)]
    rows.append(label(row([label(fit(c, tall), "B2e, %d deg" % a) for c, a in zip(turn, (0, 30, 60, 90))]),
                      "merged scratch build (B2e), face_turn.jpg", 30, 18))
    col(rows).save(os.path.join(PREV, "vroid_face.jpg"), quality=88)


def full(raw):
    """vroid_full.jpg: B standing, front / three-quarter / side / back; the merged B2e build's posed_full row."""
    cell = (450, 750)
    names = (("p_front", "B, front"), ("p_34", "B, front three-quarter"), ("p_side", "B, side (her right)"), ("p_back", "B, back"))
    rows = [row([label(flat(os.path.join(raw, "S_%s.png" % n), cell), t) for n, t in names])]
    pf = os.path.join(PREV, "posed_full.jpg")
    rows.append(label(row([label(fit(cell_of(pf, i * 908, 0, 900, 1544), cell), t) for i, t in
                           enumerate(("B2e, front", "B2e, front three-quarter", "B2e, side (her right)", "B2e, back three-quarter"))]),
                      "merged scratch build (B2e), posed_full.jpg (concept pose)", 30, 18))
    col(rows).save(os.path.join(PREV, "vroid_full.jpg"), quality=88)


def whatif(raw):
    """vroid_whatif_hair.jpg: B as built (the CC0 HairSample_Female's hair, lengthened) against a WHAT-IF with Sendagaya
    Shino's long hair (licence 'Other', not committed, read from the download folder): the concept framing and 0 / 60 / 90 deg."""
    cell, tall = (512, 768), (400, 500)
    rows = []
    for tag, txt in (("P", "B as built: HairSample_Female's hair (CC0), lengthened by script"),
                     ("W", "WHAT-IF: Sendagaya Shino's hair (NOT CC0 -- not committed; shows a real long VRoid hairstyle)")):
        cells = [label(flat(os.path.join(raw, "%s_concept.png" % tag), cell), "concept framing")]
        ft = "S" if tag == "P" else "W"
        cells += [label(flat(os.path.join(raw, "%s_face_%03d.png" % (ft, a)), tall), "%d deg" % a) for a in (0, 60, 90)]
        rows.append(label(row(cells), txt, 30, 18))
    col(rows).save(os.path.join(PREV, "vroid_whatif_hair.jpg"), quality=88)


def main(raw, only):
    if "candidates" in only:
        candidates(raw)
    if "compare" in only:
        compare(raw)
    if "face" in only:
        face(raw)
    if "full" in only:
        full(raw)
    if "whatif" in only:
        whatif(raw)
    print("written to", PREV)


if __name__ == "__main__":
    argv = sys.argv[1:]
    o = ["candidates", "compare", "face", "full"]
    if "--only" in argv:
        i = argv.index("--only")
        o = argv[i + 1].split(",")
        argv = argv[:i] + argv[i + 2:]
    main(argv[0] if argv else os.path.join(tempfile.gettempdir(), "Beatrice_vroid"), o)
