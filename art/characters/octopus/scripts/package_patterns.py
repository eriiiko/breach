"""Octopus skin patterns (#113): per-variant game GLBs + the labelled lineup sheets. No Blender.

python package_patterns.py --root <breach checkout>      (after build_patterns.py; Pillow)

- GLB: prototype/octopus/variants/octopus_game_<name>.glb = the committed octopus_game.glb with
  ONLY its baseColor PNG replaced by octopus_albedo_gloss_<name>.png (charkit.glbtex.replace_image:
  every other buffer view is copied byte for byte). Then glbtex's mesh/skin/animation digests are
  compared against the committed GLB: a variant whose mesh differs is a failure (exit 1).
  The GLBs are ~8 MB each and gitignored; this command rebuilds them in seconds.
- previews/patterns/lineup_top.jpg, lineup_34.jpg, game_lineup.jpg.
"""
from pathlib import Path
import argparse
import json
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pattern_math as pm

HOME = HERE.parent
OCT = HOME / "prototype" / "octopus"
VAR = OCT / "variants"
PRE = HOME / "previews" / "patterns"


def font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def lineup(suffix, out, crop, tile_w, cols):
    tiles = []
    for name in pm.ORDER:
        im = Image.open(PRE / suffix.format(name)).convert("RGB")
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
        im = im.resize((tile_w, int(tile_w * im.height / im.width)), Image.LANCZOS)
        tiles.append((name, im))
    th = tiles[0][1].height
    rows = (len(tiles) + cols - 1) // cols
    head = 54
    sheet = Image.new("RGB", (cols * tile_w, rows * (th + head)), (236, 234, 230))
    d = ImageDraw.Draw(sheet)
    f1, f2 = font(24), font(14)
    for i, (name, im) in enumerate(tiles):
        x, y = (i % cols) * tile_w, (i // cols) * (th + head)
        sheet.paste(im, (x, y + head))
        d.text((x + 12, y + 6), name, fill=(30, 30, 30), font=f1)
        text = pm.PATTERNS[name]["animal"]
        while d.textlength(text, font=f2) > tile_w - 24:
            text = text[:-2].rstrip(" ,(") + "…" if not text.endswith("…") else text[:-2] + "…"
        d.text((x + 12, y + 34), text, fill=(90, 90, 90), font=f2)
    sheet.save(PRE / out, quality=88)
    return PRE / out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    sys.path.insert(0, str(args.root.resolve() / "art" / "characters"))
    from charkit import glbtex
    base = OCT / "octopus_game.glb"
    g0, b0 = glbtex.read(base)
    report = {}
    ok = True
    for name in pm.ORDER:
        png = VAR / f"octopus_albedo_gloss_{name}.png"
        out = VAR / f"octopus_game_{name}.glb"
        glbtex.replace_image(base, "baseColor", png.read_bytes(), out=out)
        g1, b1 = glbtex.read(out)
        img_view = g1["images"][glbtex.texture_image(g1, "baseColor")]["bufferView"]
        same = [glbtex.view_bytes(g0, b0, i) == glbtex.view_bytes(g1, b1, i)
                for i in range(len(g0["bufferViews"])) if i != img_view]
        mesh_ok = all(same) and len(g0["bufferViews"]) == len(g1["bufferViews"])
        ok &= mesh_ok
        report[name] = dict(glb_bytes=out.stat().st_size, png_bytes=png.stat().st_size,
                            non_albedo_views_identical=mesh_ok, views_compared=len(same))
        print(name, report[name])
    rp = VAR / "patterns_report.json"
    rep = json.loads(rp.read_text()) if rp.exists() else {}
    rep["glb"] = report
    rp.write_text(json.dumps(rep, indent=2) + "\n")
    lineup("{}_top.jpg", "lineup_top.jpg", (.06, .22, .94, .86), 420, 4)
    lineup("{}_34.jpg", "lineup_34.jpg", (.02, .1, .98, .97), 460, 4)
    lineup("game_{}_top.jpg", "game_lineup.jpg", (.06, .22, .94, .86), 360, 4)
    print("PACKAGE_PATTERNS_OK" if ok else "PACKAGE_PATTERNS_MESH_DIFFERS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
