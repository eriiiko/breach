"""Cut the three figures out of the guard's reference sheet: writes the figure mask the sheet
frame reads (`sheet_mask.png`) and a picture of it over the sheet (`previews/sheet_mask.png`).

    python scripts/make_mask.py          (plain Python with numpy + PIL, not Blender)

The sheet is not a plain backdrop: a light grid, a frame, a text header, view labels, a
callout box whose leader line comes within a few pixels of the front figure, and a saber
section below. The figures are drawn with dark outlines, which is what the cut relies on:

1. figure ink = colour distance from the backdrop above `THR` (the grid lines sit at ~15,
   the glove's lit faces at ~10, every outline and shaded part far above 30);
2. everything outside the figures' band of rows (`ROWS`) and inside the `EXCLUDE`
   rectangles (the callout box and its leader line) is blanked;
3. a closing by `CLOSE` px joins the outlines, every enclosed background region smaller
   than `HOLE_MAX` px is filled (the white gloves inside their outlines, the grid squares
   seen through the gaps between the fingers stay open only if they are large), and the
   closing is undone;
4. per view only the component that holds the view's `SEED` is kept, so loose ink (a
   stray grid crossing, the label text) never counts.
"""
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SHEET = os.path.join(ROOT, "Gemini_Generated_Image_k4m4suk4m4suk4m4.jfif")
OUT = os.path.join(ROOT, "sheet_mask.png")
PICTURE = os.path.join(ROOT, "previews", "sheet_mask.png")

BACKDROP = (219, 223, 226)       # the sheet's backdrop, sRGB
THR = 30                         # colour distance (max over channels) that counts as ink
ROWS = (96, 834)                 # the figures' band: below the header box, above the view labels
EXCLUDE = ((266, 505, 291, 581),  # the callout's leader line (x0, y0, x1, y1, exclusive ends)
           (279, 556, 372, 700),  # the callout box
           (813, 96, 832, 834),   # the frame's right edge, which the back view's right glove nearly touches
           (0, 96, 19, 834))      # the frame's left edge
LINE_REACH, LINE_BG, LINE_BLUE = 3, 22, 4.5  # a thin line: backdrop 3 px away on both sides
CLOSE = 2
HOLE_MAX = 2400                  # a glove's inside is ~1700 px; a grid square ~680 is never enclosed
SEED = dict(front=(180, 400), side=(440, 400), back=(678, 400))


def shift_or(m, r):
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def dilate(m, r):
    return shift_or(m, r)


def erode(m, r):
    return ~shift_or(~m, r)


def flood(mask, seed):
    """The 4-connected component of True pixels in `mask` that holds `seed` (x, y)."""
    lab = label(mask)
    x, y = seed
    return (lab == lab[y, x]) & mask if mask[y, x] else np.zeros_like(mask)


def label(mask):
    """4-connected component labels of `mask` (0 = not in it): every pixel points at the
    smallest flat index in its component, by neighbour-min propagation plus pointer jumping."""
    h, w = mask.shape
    n = h * w
    idx = np.arange(n).reshape(h, w)
    lab = np.where(mask, idx, n)
    while True:
        prev = lab
        p = np.pad(lab, 1, constant_values=n)
        nb = np.minimum.reduce([lab, p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        nb = np.where(mask, nb, n)
        flat = np.append(nb.ravel(), n)
        for _ in range(4):  # jump: a pixel takes its representative's representative
            flat = flat[flat]
        lab = flat[:-1].reshape(h, w)
        if np.array_equal(lab, prev):
            return np.where(mask, lab + 1, 0)


def fill_small_holes(m, max_px):
    """Fill every enclosed background region of at most `max_px` pixels."""
    bg = ~m
    holes = bg & ~flood(bg, (0, 0))
    lab = label(holes)
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    small = np.isin(lab, ids[counts <= max_px])
    return m | small


def box_mean(a, r=1):
    p = np.pad(a.astype(float), r, mode="edge")
    h, w = a.shape
    return sum(p[r + dy:r + dy + h, r + dx:r + dx + w] for dy in range(-r, r + 1) for dx in range(-r, r + 1)) / (2 * r + 1) ** 2


def thin_lines(rgb, dist, reach=LINE_REACH, bg=LINE_BG):
    """Pixels on a thin line drawn over the backdrop (the grid, the floor line, the leader
    line): pixels whose two neighbourhoods `reach` px away on some axis (across, down,
    either diagonal) are both plain BACKDROP -- near its colour and as blue as it (3 x 3
    means, so JPEG noise does not decide). A figure's edge never qualifies, since one side
    of it is figure; nor does a glove's grey outline, whose inner side is the glove's
    neutral white, not the backdrop's blue-grey."""
    plain = (box_mean(dist) < bg) & (box_mean(rgb[..., 2] - rgb[..., 0]) >= LINE_BLUE)
    r = reach
    p = np.pad(plain, r, constant_values=False)
    h, w = dist.shape

    def at(dy, dx):
        return p[r + dy:r + dy + h, r + dx:r + dx + w]

    out = np.zeros(dist.shape, bool)
    for dy, dx in ((0, r), (r, 0), (r, r), (r, -r)):
        out |= at(dy, dx) & at(-dy, -dx)
    return out


def main():
    rgb = np.asarray(Image.open(SHEET).convert("RGB")).astype(int)
    h, w = rgb.shape[:2]
    dist = np.abs(rgb - np.array(BACKDROP)).max(axis=2)
    ink = (dist > THR) & ~thin_lines(rgb, dist)
    ink[:ROWS[0]] = False
    ink[ROWS[1]:] = False
    for x0, y0, x1, y1 in EXCLUDE:
        ink[y0:y1, x0:x1] = False
    m = dilate(ink, CLOSE)
    m = fill_small_holes(m, HOLE_MAX)
    m = erode(m, CLOSE) | ink
    m = fill_small_holes(m, HOLE_MAX // 4)
    keep = np.zeros_like(m)
    lab = label(m)
    for name, (x, y) in SEED.items():
        comp = lab == lab[y, x]
        print("%s: %d px" % (name, comp.sum()))
        keep |= comp
    Image.fromarray((keep * 255).astype(np.uint8)).save(OUT)
    # the picture: the sheet, the mask tinted over it, the excluded rectangles outlined
    pic = rgb.astype(float)
    pic[keep] = 0.45 * pic[keep] + 0.55 * np.array([40, 170, 90])
    pic[~keep] = 0.6 * pic[~keep] + 0.4 * 255
    im = Image.fromarray(pic.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for x0, y0, x1, y1 in EXCLUDE:
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(220, 40, 40))
    d.line([(0, ROWS[0]), (w, ROWS[0])], fill=(220, 40, 40))
    d.line([(0, ROWS[1]), (w, ROWS[1])], fill=(220, 40, 40))
    os.makedirs(os.path.dirname(PICTURE), exist_ok=True)
    im.crop((0, 0, w, 900)).save(PICTURE)
    print("wrote", OUT, PICTURE)


if __name__ == "__main__":
    main()
