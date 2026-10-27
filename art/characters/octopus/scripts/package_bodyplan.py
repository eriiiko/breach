"""Labelled body-plan sheets from build_bodyplan.py's renders (#106). Pillow only.

    python package_bodyplan.py

Writes previews/bodyplan/: per row <row>_top.jpg and <row>_34.jpg (rest | reach),
options_top.jpg and options_34.jpg (small | mid | big, rest over reach), stretch_strip.jpg.
Top views carry a scale bar of one marine (the game's space marine, 1.885 m) and
the game's tiles (0.333 m, the shipped tile size).
"""
from pathlib import Path
import json

from PIL import Image, ImageDraw, ImageFont

import octopus_math as om

HOME = Path(__file__).resolve().parent.parent
OUT = HOME / "previews" / "bodyplan"
RAW = OUT / "raw"
LABEL = {"small_preview": "small (preview)", "mid": "medium", "big": "big (preview)"}
ORDER = ["small_preview", "mid", "big"]
TILE_M = 0.333
INK, PAPER, ACCENT = (24, 24, 26), (238, 236, 232), (178, 52, 30)


def font(size, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def scale_bar(img, m_per_px, y_from_bottom=46):
    """One-marine bar plus a game-tile ruler, bottom left."""
    d = ImageDraw.Draw(img)
    px = om.MARINE_HEIGHT_M / m_per_px
    tile_px = TILE_M / m_per_px
    x0, y0 = 30, img.height - y_from_bottom
    label = "1 marine = 1.885 m; ticks = game tiles (0.333 m)"
    d.rectangle([x0 - 10, y0 - 34, x0 + max(px, d.textlength(label, font=font(17))) + 14, y0 + 26], fill=(255, 255, 255))
    d.rectangle([x0, y0, x0 + px, y0 + 7], fill=INK)
    i = 0
    while i * tile_px <= px + 0.5:
        x = x0 + i * tile_px
        d.line([x, y0 - 6, x, y0 + 7], fill=INK, width=2)
        i += 1
    d.text((x0, y0 - 30), label, fill=INK, font=font(17))
    d.text((x0 + px + 10, y0 - 6), "", fill=INK, font=font(17))


def caption(name, rec):
    s = rec["size"]
    default = "  (default)" if name == "mid" else ""
    return [f"{LABEL[name]}  {name}{default}",
            f"body {s['body_length_m']:.2f} m long, mantle {s['mantle_width_m']:.2f} x {s['mantle_length_m']:.2f} m",
            f"arm {s['arm_length_rest_m']:.2f} m (x{s['arm_body_ratio']:.2f} body), stretched {s['arm_length_stretched_m']:.2f} m;"
            f" base {s['arm_base_diameter_m']:.2f} m thick",
            f"span {s['span_rest_m']:.2f} m = {s['span_rest_marines']:.2f} marines = {s['span_rest_tiles_at_0333m']:.1f} tiles;"
            f" stretched {s['span_stretched_m']:.2f} m = {s['span_stretched_marines']:.2f} marines"]


def labelled(path, lines, sub=None, bar=None, w=None):
    img = Image.open(path).convert("RGB")
    if w and img.width != w:
        img = img.resize((w, round(img.height * w / img.width)), Image.LANCZOS)
        if bar:
            bar = bar * Image.open(path).width / w
    if bar:
        scale_bar(img, bar)
    head = 30 + 26 * len(lines)
    out = Image.new("RGB", (img.width, img.height + head), PAPER)
    out.paste(img, (0, head))
    d = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        d.text((14, 10 + 26 * i), line, fill=ACCENT if i == 0 else INK, font=font(24 if i == 0 else 19, i == 0))
    if sub:
        d.text((out.width - 14 - d.textlength(sub, font=font(22, True)), 12), sub, fill=INK, font=font(22, True))
    return out


def grid(tiles, cols):
    w = max(t.width for t in tiles)
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    hs = [max(t.height for t in r) for r in rows]
    sheet = Image.new("RGB", (cols * w + (cols - 1) * 8, sum(hs) + (len(rows) - 1) * 8), (255, 255, 255))
    y = 0
    for r, h in zip(rows, hs):
        for c, t in enumerate(r):
            sheet.paste(t, (c * (w + 8), y))
        y += h + 8
    return sheet


def main():
    rep = json.loads((OUT / "bodyplan.json").read_text())
    opts = rep["options"]
    for view in ("top", "34"):
        tiles = {}
        for name in ORDER:
            rec = opts[name]
            for pose in ("rest", "reach"):
                bar = rec[f"{pose}_top"]["m_per_px"] if view == "top" else None
                sub = {"rest": "at rest", "reach": "reaching: R2 + R3 stretched"}[pose]
                tiles[name, pose] = labelled(RAW / f"{name}_{pose}_{view}.png", caption(name, rec), sub, bar, w=1000)
            grid([tiles[name, "rest"], tiles[name, "reach"]], 2).save(OUT / f"{name}_{view}.jpg", quality=88)
        sheet = grid([tiles[n, "rest"] for n in ORDER] + [tiles[n, "reach"] for n in ORDER], 3)
        sheet.save(OUT / f"options_{view}.jpg", quality=86)
    if "strip" in rep:
        st = rep["strip"]
        panels = []
        for k in ("0.0", "0.5", "1.0"):
            r = st[k]
            lines = [f"stretch {float(k):.1f}",
                     f"arm L1 {r['arm_length_m']:.2f} m, tip radius x{r['thinning']:.2f}"]
            panels.append(labelled(RAW / f"strip_{float(k):.1f}.png", lines, None, r["m_per_px"]))
        grid(panels, 3).save(OUT / "stretch_strip.jpg", quality=88)
    print("BODYPLAN_SHEETS_OK", sorted(p.name for p in OUT.glob("*.jpg")))


if __name__ == "__main__":
    main()
