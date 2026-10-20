"""Lay out Beatrice's face pictures (B2a, B2b) from `face_views.py`'s raw renders (plain Python + PIL, from art/characters):

    python Beatrice/scripts/face_sheets.py [--raw DIR] [--posed DIR]

B2b: the default look is the toon look; the raw renders of `face_views.py -- --look pbr` (prefix `pbr_`) give the comparison
rows, those of `-- --head apose` (prefix `apose_`) the head-size picture, and `posed.py`'s raw renders (default and
`-- --look pbr`) the posed cells of look_compare.jpg.

face_vs_art.jpg   the close-ups and the model in their frames: drawing | render | the render's outline (red) over the drawing,
                  front and profile; rows: toon with hair, toon without hair, pbr with hair (B2a's look, for comparison)
face_turn.jpg     the head at 0, 30, 60, 90, 135, 180 deg in the toon look, and a pbr row at 0, 30, 90
face_closeup.jpg  eyes, an eye in three-quarter, nose and mouth, the mouth from the side (toon); the drawing's eyes and mouth
                  beside; the pbr eyes and three-quarter eye
head_size.jpg     the default head (x 1.20, the concept's ratio) against B2a's A-pose size (`apose`), and the concept
look_compare.jpg  the same shots in the pbr and the toon look: face front, three-quarter, the posed concept framing, full figure
hair_vs_art.jpg   B2c: the close-ups' framing with the B2c hair and B2a's blockout (`face_views.py -- --head hair_b2a`), outlines
hair_back.jpg     B2c: the hair from behind and three-quarter behind (turn 180, 135, 225), the A-pose back drawing, B2a's back
"""
import argparse
import os
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=os.path.join(tempfile.gettempdir(), "Beatrice_face"))
ap.add_argument("--posed", default=os.path.join(tempfile.gettempdir(), "Beatrice_posed"))
args = ap.parse_args()
PREV = os.path.join(ROOT, "previews")
BG = (203, 205, 212)
BACK = (161, 164, 171)   # studio.BACKDROP in sRGB


def font(n=26):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, n)
        except OSError:
            pass
    return ImageFont.load_default()


def label(im, text, n=26):
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, n + 12], fill=(240, 240, 238))
    d.text((8, 4), text, fill=(20, 20, 20), font=font(n))
    return im


def has(name, d=None):
    return os.path.exists(os.path.join(d or args.raw, name))


def raw(name, d=None):
    return Image.open(os.path.join(d or args.raw, name)).convert("RGBA")


def on_bg(im, bg=BG):
    b = Image.new("RGBA", im.size, bg + (255,))
    b.alpha_composite(im)
    return b.convert("RGB")


def outline(im, width=3):
    a = np.asarray(im)[..., 3] > 127
    e = np.zeros_like(a)
    e[1:] |= a[1:] != a[:-1]
    e[:, 1:] |= a[:, 1:] != a[:, :-1]
    for _ in range(width - 1):
        e2 = e.copy()
        e2[1:] |= e[:-1]
        e2[:-1] |= e[1:]
        e2[:, 1:] |= e[:, :-1]
        e2[:, :-1] |= e[:, 1:]
        e = e2
    return e


def over(ref, im):
    r = np.asarray(ref.convert("RGB")).copy()
    r[outline(im)] = (230, 20, 20)
    return Image.fromarray(r)


def grid(cells, cols, w, h):
    sheet = Image.new("RGB", (w * cols, h * ((len(cells) + cols - 1) // cols)), (255, 255, 255))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % cols) * w, (i // cols) * h))
    return sheet


P = 420
# --- face_vs_art
rows = []
for pre, hair, what in (("", "", "toon, with hair"), ("", "_nohair", "toon, without hair"), ("pbr_", "", "pbr (B2a's look), with hair")):
    if not has("%sart_front%s.png" % (pre, hair)):
        continue
    cells = []
    for view, ref in (("front", "concept-face-front-v1.png"), ("profile", "concept-face-side-v1.png")):
        R = Image.open(os.path.join(ROOT, ref)).convert("RGB")
        im = raw("%sart_%s%s.png" % (pre, view, hair))
        cells += [label(R.resize((P, P), Image.LANCZOS), "drawing, %s" % view),
                  label(on_bg(im).resize((P, P), Image.LANCZOS), "model, %s, %s" % (view, what)),
                  label(over(R, im).resize((P, P), Image.LANCZOS), "model's outline over it")]
    rows.append(grid(cells, 6, P, P))
sheet = Image.new("RGB", (rows[0].width, P * len(rows)), (255, 255, 255))
for i, r in enumerate(rows):
    sheet.paste(r, (0, i * P))
sheet.save(os.path.join(PREV, "face_vs_art.jpg"), quality=90)

# --- face_turn
az = (0, 30, 60, 90, 135, 180)
cells = [label(raw("turn_%03d.png" % a).convert("RGB").resize((540, 675), Image.LANCZOS), "toon, %d deg" % a) for a in az]
cells += [label(raw("pbr_turn_%03d.png" % a).convert("RGB").resize((540, 675), Image.LANCZOS), "pbr (B2a's look), %d deg" % a)
          for a in (0, 30, 90) if has("pbr_turn_%03d.png" % a)]
grid(cells, 3, 540, 675).save(os.path.join(PREV, "face_turn.jpg"), quality=90)

# --- face_closeup
F = Image.open(os.path.join(ROOT, "concept-face-front-v1.png")).convert("RGB")
S = Image.open(os.path.join(ROOT, "concept-face-side-v1.png")).convert("RGB")
cells = [(raw("close_eyes.png").convert("RGB").resize((1000, 500), Image.LANCZOS), "eyes (toon)"),
         (F.crop((300, 450, 1000, 800)).resize((1000, 500), Image.LANCZOS), "drawing: eyes"),
         (raw("close_eye_q.png").convert("RGB").resize((667, 500), Image.LANCZOS), "her left eye, three-quarter (toon)"),
         (raw("close_nose_mouth.png").convert("RGB").resize((667, 500), Image.LANCZOS), "nose and mouth (toon)"),
         (raw("close_mouth_side.png").convert("RGB").resize((667, 500), Image.LANCZOS), "mouth, from her right (toon)"),
         (F.crop((480, 660, 830, 935)).resize((636, 500), Image.LANCZOS), "drawing: nose and mouth"),
         (S.crop((700, 600, 1000, 1000)).transpose(Image.FLIP_LEFT_RIGHT).resize((375, 500), Image.LANCZOS), "drawing: profile")]
if has("pbr_close_eyes.png"):
    cells += [(raw("pbr_close_eyes.png").convert("RGB").resize((1000, 500), Image.LANCZOS), "eyes (pbr, B2a's look)"),
              (raw("pbr_close_eye_q.png").convert("RGB").resize((667, 500), Image.LANCZOS), "three-quarter (pbr)")]
W = 2000
rowh = 500
x = y = 0
places = []
for im, t in cells:
    if x + im.width > W:
        x, y = 0, y + rowh
    places.append((im, t, x, y))
    x += im.width
sheet = Image.new("RGB", (W, y + rowh), (255, 255, 255))
for im, t, x, y in places:
    sheet.paste(label(im.copy(), t), (x, y))
sheet.save(os.path.join(PREV, "face_closeup.jpg"), quality=90)

# --- head_size: the default (x 1.20) against B2a's A-pose size
if has("apose_turn_000.png"):
    C = Image.open(os.path.join(ROOT, "concept.jfif")).convert("RGB")
    cells = [label(raw("apose_turn_000.png").convert("RGB").resize((540, 675), Image.LANCZOS), "A-pose size (apose), framed on the head"),
             label(raw("turn_000.png").convert("RGB").resize((540, 675), Image.LANCZOS), "x 1.20 (default), framed on the head"),
             label(raw("apose_turn_030.png").convert("RGB").resize((540, 675), Image.LANCZOS), "A-pose size, 30 deg"),
             label(raw("turn_030.png").convert("RGB").resize((540, 675), Image.LANCZOS), "x 1.20 (default), 30 deg"),
             label(C.resize((450, 675), Image.LANCZOS), "concept")]
    sheet = Image.new("RGB", (540 * 4 + 450, 675), (255, 255, 255))
    x = 0
    for c in cells:
        sheet.paste(c, (x, 0))
        x += c.width
    sheet.save(os.path.join(PREV, "head_size.jpg"), quality=90)


# --- look_compare: pbr | toon, the same shots
def flat(im):
    return on_bg(im, BACK)


H = 760
pairs = [("face, front", "pbr_turn_000.png", "turn_000.png", args.raw),
         ("face, 30 deg", "pbr_turn_030.png", "turn_030.png", args.raw),
         ("posed, the concept's framing", "pbr_concept_concept.png", "concept_concept.png", args.posed),
         ("posed, full figure", "pbr_concept_front.png", "concept_front.png", args.posed)]
cells = []
for what, a, b, d in pairs:
    if not (has(a, d) and has(b, d)):
        continue
    for name, look in ((a, "pbr"), (b, "toon")):
        im = flat(raw(name, d))
        im = im.resize((int(im.width * H / im.height), H), Image.LANCZOS)
        cells.append(label(im, "%s: %s" % (look, what), 22))
if cells:
    rows = [cells[i:i + 4] for i in range(0, len(cells), 4)]
    width = max(sum(c.width for c in r) + 8 * (len(r) - 1) for r in rows)
    sheet = Image.new("RGB", (width, (H + 8) * len(rows)), (255, 255, 255))
    y = 0
    for r in rows:
        x = 0
        for c in r:
            sheet.paste(c, (x, y))
            x += c.width + 8
        y += H + 8
    sheet.save(os.path.join(PREV, "look_compare.jpg"), quality=90)
# --- B2c: hair_vs_art -- the close-ups' framing, the B2c hair and B2a's blockout (`face_views.py -- --head hair_b2a --only art,turn`)
if has("hair_b2a_art_front.png"):
    rows = []
    for view, ref in (("front", "concept-face-front-v1.png"), ("profile", "concept-face-side-v1.png")):
        R = Image.open(os.path.join(ROOT, ref)).convert("RGB")
        now, b2a = raw("art_%s.png" % view), raw("hair_b2a_art_%s.png" % view)
        cells = [label(R.resize((P, P), Image.LANCZOS), "drawing, %s" % view),
                 label(on_bg(now).resize((P, P), Image.LANCZOS), "B2c hair, %s" % view),
                 label(over(R, now).resize((P, P), Image.LANCZOS), "B2c outline over the drawing"),
                 label(on_bg(b2a).resize((P, P), Image.LANCZOS), "B2a/B2b hair (before)"),
                 label(over(R, b2a).resize((P, P), Image.LANCZOS), "before: outline over the drawing")]
        rows.append(grid(cells, 5, P, P))
    sheet = Image.new("RGB", (rows[0].width, P * len(rows)), (255, 255, 255))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, i * P))
    sheet.save(os.path.join(PREV, "hair_vs_art.jpg"), quality=90)

# --- B2c: hair_back -- straight back and three-quarter back (both sides), the A-pose back drawing beside, B2a's before
if has("turn_180.png"):
    Bk = Image.open(os.path.join(ROOT, "with_hair_back.png")).convert("RGB").crop((262, 0, 762, 625))
    cells = [label(Bk.resize((540, 675), Image.LANCZOS), "drawing: with_hair_back.png")]
    for name, t in (("turn_180.png", "B2c, straight back"), ("turn_135.png", "B2c, three-quarter back (her left)"),
                    ("turn_225.png", "B2c, three-quarter back (her right)"), ("hair_b2a_turn_180.png", "B2a/B2b (before), back")):
        if has(name):
            cells.append(label(raw(name).convert("RGB").resize((540, 675), Image.LANCZOS), t))
    sheet = Image.new("RGB", (540 * len(cells), 675), (255, 255, 255))
    for i, c in enumerate(cells):
        sheet.paste(c, (540 * i, 0))
    sheet.save(os.path.join(PREV, "hair_back.jpg"), quality=90)

print("sheets ->", PREV)
