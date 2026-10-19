"""Offline material inspection sheets; never a replacement for game look shots.

Reads the delivered maps without modifying them. The neutral diffuse diagnostic
uses Lambert's cosine law (Johann Heinrich Lambert, Photometria, 1760), solely
to expose normal direction and alignment; it is not Breach's final lighting.
Run with this machine's breach Python. Outputs live in normal-review/.
"""
from pathlib import Path
import argparse
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "normal-review"
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
SMALL = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)


def panel(im, size=240):
    rgba = im.convert("RGBA")
    rgba.thumbnail((size, size), Image.Resampling.LANCZOS)
    bg = Image.new("RGB", (size, size), (48, 48, 48))
    bg.paste(rgba, ((size-rgba.width)//2, (size-rgba.height)//2), rgba)
    return bg


def light_image(normal, angle, colour=None):
    # Same image-row-down normal frame as make_tileset; incoming light is
    # expressed in that frame. Renormalize after sampling the encoded normals.
    n = np.asarray(normal.convert("RGB"), dtype=np.float32) / 127.5 - 1
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    light = np.array([math.cos(angle), math.sin(angle), 0.22], dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = 0.12 + 0.88 * np.maximum(n @ light, 0)
    if colour is None:
        rgb = np.repeat(shade[..., None], 3, axis=-1)
    else:
        # Decode before lighting, then encode for this standalone diagnostic.
        rgb = (np.asarray(colour.convert("RGB"), dtype=np.float32) / 255) ** 2.2
        rgb = rgb * shade[..., None]
    return Image.fromarray(np.uint8(np.clip(rgb ** (1/2.2) * 255 + 0.5, 0, 255)))


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", action="store_true")
    args = parser.parse_args()
    if args.v2:
        OUT = ROOT / "normal-v2" / "review"
    OUT.mkdir(exist_ok=True)
    records = json.loads((ROOT / "validation.json").read_text())["assets"]
    pages, metrics = [], []
    for k, record in enumerate(records):
        rel = Path(record["asset"])
        path = ROOT / rel
        colour = Image.open(path).convert("RGBA")
        maps = ROOT / "normal-v2" / rel if args.v2 else path
        normal = Image.open(maps.with_name(path.stem + "_n.png"))
        height = Image.open(maps.with_name(path.stem + "_height.png"))
        h = np.asarray(height, dtype=np.float32) / 65535
        n = np.asarray(normal.convert("RGB"), dtype=np.float32) / 127.5 - 1
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        tilt = np.degrees(np.arccos(np.clip(n[..., 2], -1, 1)))
        metrics.append({"asset": rel.as_posix(),
                        "tilt_degrees_p50_p95_p99": np.percentile(tilt, [50,95,99]).tolist(),
                        "height_unique_levels": int(np.unique(h).size),
                        "height_edge_jump_x_y": [float(np.abs(h[:,0]-h[:,-1]).mean()),
                                                 float(np.abs(h[0]-h[-1]).mean())]})
        if k % 6 == 0:
            page = Image.new("RGB", (1020, 6*284+50), (24,26,31))
            d = ImageDraw.Draw(page)
            for x, title in zip([10,265,520,775], ["Colour", "Height", "Normal", "Neutral / grazing light"]):
                d.text((x,12), title, font=FONT, fill="white")
            pages.append(page)
        y = 48 + (k%6)*284
        ImageDraw.Draw(page).text((10,y), rel.as_posix(), font=FONT, fill="white")
        hi = Image.fromarray(np.uint8(np.clip(h*255+.5,0,255)))
        shaded = light_image(normal, -2.3)
        for x, im in zip([10,265,520,775], [colour,hi,normal,shaded]):
            page.paste(panel(im), (x,y+26))
        # Per-material moving-light comparison: uniform surface on left makes
        # inferred geometry visible; coloured surface on right exposes alignment.
        size = (256,256)
        nc = normal.resize(size, Image.Resampling.BILINEAR)
        cc = colour.resize(size, Image.Resampling.LANCZOS)
        alpha = cc.getchannel("A")
        frames=[]
        for frame in range(24):
            angle = 2*math.pi*frame/24
            canvas=Image.new("RGB", (512,290), (24,26,31))
            draw=ImageDraw.Draw(canvas)
            draw.text((6,3), rel.as_posix(), font=SMALL, fill="white")
            for x, im in [(0,light_image(nc,angle)),(256,light_image(nc,angle,cc))]:
                canvas.paste(im,(x,30),alpha)
            frames.append(canvas)
        stem = "--".join(rel.with_suffix("").parts)
        frames[0].save(OUT/(stem+".gif"),save_all=True,append_images=frames[1:],duration=85,loop=0)
        # Repeated albedo beside repeated normal lighting. No seam repair is
        # implied: a wrapped derivative alone cannot make a texture seamless.
        tile = Image.new("RGB",(1024,540),(24,26,31))
        dt=ImageDraw.Draw(tile)
        dt.text((6,3),rel.as_posix()+" : 2x2 repeat",font=FONT,fill="white")
        for row in range(2):
            for col in range(2):
                tile.paste(panel(cc,256),(col*256,28+row*256))
                tile.paste(light_image(nc,-2.3),(512+col*256,28+row*256),alpha)
        tile.save(OUT/(stem+"--repeat.jpg"),quality=92)
    for index,page in enumerate(pages,1):
        page.save(OUT/f"sheet-{index:02d}.jpg",quality=94)
    (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(f"Reviewed data for {len(records)} material sets; {len(pages)} sheets + animations + repeat views in {OUT}")


if __name__ == "__main__":
    main()
