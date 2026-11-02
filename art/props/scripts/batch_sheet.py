"""Contact sheet of prop_gameready.py previews: one row per prop, columns
front34 / front / side / top, the name written on each row.
    python batch_sheet.py <preview_dir> <out.jpg> <name> [<name> ...]"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw

d, out, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
S, views = 300, ("front34", "front", "side", "top")
sheet = Image.new("RGB", (S * len(views), S * len(names)), "white")
draw = ImageDraw.Draw(sheet)
for r, n in enumerate(names):
    for c, v in enumerate(views):
        sheet.paste(Image.open(d / f"{n}_{v}.png").convert("RGB").resize((S, S)), (c * S, r * S))
    draw.text((6, r * S + 6), n, fill="black")
sheet.save(out, quality=85)
