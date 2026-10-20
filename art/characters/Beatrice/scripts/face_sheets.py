"""Lay out Beatrice's face pictures (B2a) from `face_views.py`'s raw renders (plain Python + PIL, from art/characters):

    python Beatrice/scripts/face_sheets.py [--raw DIR]

face_vs_art.jpg   the close-ups and the model in their frames, with hair and without: drawing | render | the render's outline
                  (red) over the drawing, front and profile
face_turn.jpg     the head at 0, 30, 60, 90, 135, 180 deg under the portrait light
face_closeup.jpg  eyes, an eye in three-quarter, nose and mouth, the mouth from the side; the drawing's eyes and mouth beside
head_size.jpg     (when `--head concept` was rendered) the A-pose-size head against the concept-ratio variant, and the concept
"""
import argparse
import os
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=os.path.join(tempfile.gettempdir(), "Beatrice_face"))
args = ap.parse_args()
PREV = os.path.join(ROOT, "previews")
BG = (203, 205, 212)


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


def raw(name):
    return Image.open(os.path.join(args.raw, name)).convert("RGBA")


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


P = 420
# --- face_vs_art
rows = []
for hair in ("", "_nohair"):
    cells = []
    for view, ref in (("front", "concept-face-front-v1.png"), ("profile", "concept-face-side-v1.png")):
        R = Image.open(os.path.join(ROOT, ref)).convert("RGB")
        im = raw("art_%s%s.png" % (view, hair))
        what = "with hair" if not hair else "without hair"
        cells += [label(R.resize((P, P), Image.LANCZOS), "drawing, %s" % view),
                  label(on_bg(im).resize((P, P), Image.LANCZOS), "model, %s, %s" % (view, what)),
                  label(over(R, im).resize((P, P), Image.LANCZOS), "model's outline over it")]
    row = Image.new("RGB", (P * len(cells), P), (255, 255, 255))
    for i, c in enumerate(cells):
        row.paste(c, (i * P, 0))
    rows.append(row)
sheet = Image.new("RGB", (rows[0].width, P * 2), (255, 255, 255))
for i, r in enumerate(rows):
    sheet.paste(r, (0, i * P))
sheet.save(os.path.join(PREV, "face_vs_art.jpg"), quality=90)

# --- face_turn
az = (0, 30, 60, 90, 135, 180)
ims = [label(raw("turn_%03d.png" % a).convert("RGB").resize((540, 675), Image.LANCZOS), "%d deg" % a) for a in az]
sheet = Image.new("RGB", (540 * 3, 675 * 2), (255, 255, 255))
for i, im in enumerate(ims):
    sheet.paste(im, ((i % 3) * 540, (i // 3) * 675))
sheet.save(os.path.join(PREV, "face_turn.jpg"), quality=90)

# --- face_closeup
F = Image.open(os.path.join(ROOT, "concept-face-front-v1.png")).convert("RGB")
S = Image.open(os.path.join(ROOT, "concept-face-side-v1.png")).convert("RGB")
cells = [(raw("close_eyes.png").convert("RGB").resize((1000, 500), Image.LANCZOS), "eyes"),
         (F.crop((300, 450, 1000, 800)).resize((1000, 500), Image.LANCZOS), "drawing: eyes"),
         (raw("close_eye_q.png").convert("RGB").resize((667, 500), Image.LANCZOS), "her left eye, three-quarter"),
         (raw("close_nose_mouth.png").convert("RGB").resize((667, 500), Image.LANCZOS), "nose and mouth"),
         (raw("close_mouth_side.png").convert("RGB").resize((667, 500), Image.LANCZOS), "mouth, from her right"),
         (F.crop((480, 660, 830, 935)).resize((636, 500), Image.LANCZOS), "drawing: nose and mouth"),
         (S.crop((700, 600, 1000, 1000)).transpose(Image.FLIP_LEFT_RIGHT).resize((375, 500), Image.LANCZOS), "drawing: profile")]
W = 2000
sheet = Image.new("RGB", (W, 1500), (255, 255, 255))
x = y = 0
rowh = 500
for im, t in cells:
    if x + im.width > W:
        x, y = 0, y + rowh
    sheet.paste(label(im.copy(), t), (x, y))
    x += im.width
sheet.save(os.path.join(PREV, "face_closeup.jpg"), quality=90)

# --- head_size (the variant)
if os.path.exists(os.path.join(args.raw, "concept_turn_000.png")):
    C = Image.open(os.path.join(ROOT, "concept.jfif")).convert("RGB")
    cells = [label(raw("turn_000.png").convert("RGB").resize((540, 675), Image.LANCZOS), "A-pose size (built)"),
             label(raw("concept_turn_000.png").convert("RGB").resize((540, 675), Image.LANCZOS), "concept ratio, x 1.20"),
             label(raw("turn_030.png").convert("RGB").resize((540, 675), Image.LANCZOS), "A-pose size, 30 deg"),
             label(raw("concept_turn_030.png").convert("RGB").resize((540, 675), Image.LANCZOS), "concept ratio, 30 deg"),
             label(C.resize((450, 675), Image.LANCZOS), "concept")]
    sheet = Image.new("RGB", (540 * 4 + 450, 675), (255, 255, 255))
    x = 0
    for c in cells:
        sheet.paste(c, (x, 0))
        x += c.width
    sheet.save(os.path.join(PREV, "head_size.jpg"), quality=90)
print("sheets ->", PREV)
