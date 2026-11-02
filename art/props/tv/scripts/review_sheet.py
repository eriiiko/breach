"""Contact sheet of tv_gameready.py's previews: screen mask, then the off and
on rows (front34, front, side, top).   python review_sheet.py <preview_dir> <name>"""
import sys
from pathlib import Path
from PIL import Image

d, n = Path(sys.argv[1]), sys.argv[2]
S = 400
sheet = Image.new("RGB", (S * 4, S * 3), "white")
sheet.paste(Image.open(d / f"{n}_screenmask.png").convert("RGB").resize((S, S)), (0, 0))
for r, tag in enumerate(("off", "on"), start=1):
    for c, v in enumerate(("front34", "front", "side", "top")):
        im = Image.open(d / f"{n}_{tag}_{v}.png").convert("RGB").resize((S, S))
        sheet.paste(im, (c * S, r * S))
sheet.save(d / f"{n}_sheet.jpg", quality=88)
