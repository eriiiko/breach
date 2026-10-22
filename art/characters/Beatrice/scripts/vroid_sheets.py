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

from PIL import Image, ImageDraw, ImageFont

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


def main(raw, only):
    if "candidates" in only:
        candidates(raw)
    print("written to", PREV)


if __name__ == "__main__":
    argv = sys.argv[1:]
    o = ["candidates", "compare", "face", "full"]
    if "--only" in argv:
        i = argv.index("--only")
        o = argv[i + 1].split(",")
        argv = argv[:i] + argv[i + 2:]
    main(argv[0] if argv else os.path.join(tempfile.gettempdir(), "Beatrice_vroid"), o)
