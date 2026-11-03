"""Recolour char_04's albedo: green armour -> a light shell colour, the dark
glass on the helmet front -> a visor colour. Shading and scratches are kept by
scaling the target colour with each texel's own brightness.

    python recolor.py <Image_0.png> <head_tris.json> <out.png> <shell r,g,b> <visor r,g,b>
        [--gloves r,g,b] [--debug]

Colours are 0..1 sRGB. --debug paints the visor mask magenta instead.
"""
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

src, tris_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
shell = np.array([float(c) for c in sys.argv[4].split(",")])
visor = np.array([float(c) for c in sys.argv[5].split(",")])
debug = "--debug" in sys.argv
gloves = (np.array([float(c) for c in sys.argv[sys.argv.index("--gloves") + 1].split(",")])
          if "--gloves" in sys.argv else None)

im = np.asarray(Image.open(src).convert("RGB")).astype(np.float32) / 255
H, W, _ = im.shape
r, g, b = im[..., 0], im[..., 1], im[..., 2]
mx, mn = im.max(2), im.min(2)
sat = (mx - mn) / np.maximum(mx, 1e-4)
lum = im.mean(2)

# The armour: greenish (green channel leads red, not blue-dominant) and not grey.
green = (g > r * 1.04) & (g >= b * 0.97) & (sat > 0.07)

# The visor: dark, low-saturation texels on the helmet's front faces.
data = json.load(open(tris_json))
head_front = Image.new("L", (W, H), 0)
draw = ImageDraw.Draw(head_front)
for t in data["tris"]:
    if t["ny"] < -0.35 and t["h"] > 0.86 and abs(t["x"]) < 0.17:
        draw.polygon([(u * W, (1 - v) * H) for u, v in t["uv"]], fill=255)
head_front = np.asarray(head_front) > 0
visor_mask = head_front & (lum < 0.16) & (sat < 0.25)

# The gloves (and other leather): brown -- red leads green leads blue, saturated.
brown = (r > g * 1.15) & (g >= b * 0.95) & (sat > 0.2)

res = im.copy()
if gloves is not None:
    bref = np.median(lum[brown])
    res[brown] = np.clip(gloves[None, :] * (lum[brown] / bref)[:, None] ** 0.7, 0, 1)
ref = np.median(lum[green])
res[green] = np.clip(shell[None, :] * (lum[green] / ref)[:, None] ** 0.5, 0, 1)
if debug:
    res[visor_mask] = (1, 0, 1)
else:
    vref = max(np.median(lum[visor_mask]), 1e-3) if visor_mask.any() else 1
    res[visor_mask] = np.clip(visor[None, :] * (lum[visor_mask] / vref)[:, None] ** 0.5, 0, 1)
Image.fromarray((res * 255 + 0.5).astype(np.uint8)).save(out)
np.save(out.replace(".png", "_visor.npy"), visor_mask)
print(f"green texels {green.mean():.1%}, brown {brown.mean():.1%}, head-front {head_front.mean():.2%}, visor {visor_mask.mean():.2%}")
