"""Recolour the beak helmet's base colour: the cream paint -> a target colour
(or kept), the turquoise lenses -> the visor green char_04 wears. Shading,
scratches and grime are kept by scaling each target with the texel's own
brightness; grey metal and dark rings are left alone.

    python recolor_helmet.py <basecolor.png> <out.png> <lens r,g,b> [--paint r,g,b]
"""
import sys

import numpy as np
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
lens = np.array([float(c) for c in sys.argv[3].split(",")])
paint = (np.array([float(c) for c in sys.argv[sys.argv.index("--paint") + 1].split(",")])
         if "--paint" in sys.argv else None)

im = np.asarray(Image.open(src).convert("RGB")).astype(np.float32) / 255
r, g, b = im[..., 0], im[..., 1], im[..., 2]
mx, mn = im.max(2), im.min(2)
sat = (mx - mn) / np.maximum(mx, 1e-4)
lum = im.mean(2)
d = np.maximum(mx - mn, 1e-4)
hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60

def ramp(x, a, b):
    return np.clip((x - a) / (b - a), 0, 1)

# Soft paint weight: warm hue, clearly saturated, not dark (the brown rings
# round the lenses and the warm-tinted grey metal stay as they are).
w_cream = ((hue > 25) & (hue < 70)) * ramp(sat, 0.13, 0.22) * ramp(lum, 0.30, 0.42)
cream = w_cream > 0.5
teal = (hue > 160) & (hue < 215) & (sat > 0.30)

res = im.copy()
if paint is not None:
    ref = np.median(lum[cream])
    tgt = np.clip(paint[None, None, :] * (lum / ref)[..., None] ** 0.8, 0, 1)
    res = res * (1 - w_cream[..., None]) + tgt * w_cream[..., None]
lref = np.median(lum[teal])
res[teal] = np.clip(lens[None, :] * (lum[teal] / lref)[:, None] ** 0.5, 0, 1)
Image.fromarray((res * 255 + 0.5).astype(np.uint8)).save(out)
print(f"cream {cream.mean():.1%}  teal {teal.mean():.2%}")
