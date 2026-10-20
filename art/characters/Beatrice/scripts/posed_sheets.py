"""Lay out the posed pictures (B1m) from the raw renders of `posed.py` (plain Python: numpy + PIL).

    python Beatrice/scripts/posed_sheets.py [raw dir]      (default $TEMP/Beatrice_posed)

posed_vs_concept.jpg  the concept | the render at the same framing | the render's outline over the concept (B2b: the toon look)
posed_full.jpg        the pose full figure: front, front three-quarter, side, back three-quarter (the toon look)
posed_hotlight.jpg    the concept | the concept-framed render in the pbr look under the studio rig | under the `hot` light
                      setup | the toon look (B2b; the pbr renders are `posed.py -- --look pbr [--lights hot]`, prefix `pbr_`)
"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, os.path.normpath(os.path.join(HERE, "..", "..", "charkit"))]
import beatrice  # noqa: E402

BACK = tuple(int(round(255 * v)) for v in (0.6327, 0.6440, 0.6713))  # studio.BACKDROP in sRGB (0.36, 0.375, 0.41 linear)


def _font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def flat(path, size=None):
    """A raw RGBA render over the studio backdrop (the floor's shadow kept)."""
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, BACK + (255,))
    out = Image.alpha_composite(bg, im).convert("RGB")
    return out.resize(size, Image.LANCZOS) if size else out


def alpha(path, size):
    a = Image.open(path).convert("RGBA").split()[3]
    return a.resize(size, Image.LANCZOS)


def label(img, text, h=44):
    w = img.width
    strip = Image.new("RGB", (w, h), (238, 238, 235))
    ImageDraw.Draw(strip).text((12, 9), text, fill=(20, 20, 20), font=_font(24))
    out = Image.new("RGB", (w, img.height + h), (255, 255, 255))
    out.paste(strip, (0, 0))
    out.paste(img, (0, h))
    return out


def row(cells, gap=8):
    h = max(c.height for c in cells)
    out = Image.new("RGB", (sum(c.width for c in cells) + gap * (len(cells) - 1), h), (255, 255, 255))
    x = 0
    for c in cells:
        out.paste(c, (x, 0))
        x += c.width + gap
    return out


def outline_over(base, a, colour=(255, 40, 40)):
    """The render's silhouette edge (alpha > 0.5, the floor's shadow excluded) drawn over `base`."""
    m = (np.asarray(a) > 200).astype(np.uint8) * 255
    mi = Image.fromarray(m)
    edge = np.asarray(mi.filter(ImageFilter.MaxFilter(5))).astype(int) - np.asarray(mi.filter(ImageFilter.MinFilter(3))).astype(int)
    out = np.asarray(base).copy()
    out[edge > 0] = colour
    return Image.fromarray(out)


def main(raw):
    T = beatrice.POSE
    pose = next(iter(T["poses"]))
    concept = Image.open(os.path.join(ROOT, "concept.jfif")).convert("RGB")
    size = concept.size
    prev = os.path.join(ROOT, "previews")
    r = os.path.join(raw, "%s_concept.png" % pose)
    render = flat(r, size)
    row([label(concept, "concept.jfif"), label(render, "B2c posed, the toon look (default)"),
         label(outline_over(concept, alpha(r, size)), "the render's outline over the concept")]
        ).save(os.path.join(prev, "posed_vs_concept.jpg"), quality=90)
    cells = []
    for v, txt in (("front", "front"), ("front34", "front three-quarter"), ("side", "side (her right)"), ("back34", "back three-quarter")):
        p = os.path.join(raw, "%s_%s.png" % (pose, v))
        if os.path.exists(p):
            cells.append(label(flat(p), txt))
    if cells:
        row(cells).save(os.path.join(prev, "posed_full.jpg"), quality=90)
    h = os.path.join(raw, "pbr_%s_hot_concept.png" % pose)
    p = os.path.join(raw, "pbr_%s_concept.png" % pose)
    if os.path.exists(h) and os.path.exists(p):
        row([label(concept, "concept.jfif"), label(flat(p, size), "pbr look, studio rig"),
             label(flat(h, size), "pbr look, the 'hot' light setup (B1m)"), label(render, "toon look (B2b, default)")]
            ).save(os.path.join(prev, "posed_hotlight.jpg"), quality=90)
    print("written to", prev)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "Beatrice_posed"))
