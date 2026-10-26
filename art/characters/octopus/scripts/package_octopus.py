"""Assemble the octopus review pictures: swim GIF, concept-vs-build sheets, pose strip.

python package_octopus.py   (after build_octopus.py; Pillow only)
"""
from pathlib import Path
from PIL import Image, ImageDraw

HOME = Path(__file__).resolve().parent.parent
P = HOME / "previews" / "octopus"
CONCEPT = HOME / "concepts" / "octopus_concept_v1.png"

frames = [Image.open(p).convert("RGB") for p in sorted((P / "swim_frames").glob("*.png"))]
if len(frames) < 2:
    raise SystemExit("run build_octopus.py --stage gif first")
frames[0].save(P / "swim_loop.gif", save_all=True, append_images=frames[1:], duration=80, loop=0)
for p, f in zip(sorted((P / "swim_frames").glob("*.png")), frames):
    f.save(p.with_suffix(".jpg"), quality=84)


def side_by_side(left, right, out, labels, h=720):
    a, b = left.copy(), right.copy()
    a.thumbnail((10000, h))
    b.thumbnail((10000, h))
    W = Image.new("RGB", (a.width + b.width + 30, h + 50), (236, 234, 230))
    W.paste(a, (10, 40))
    W.paste(b, (20 + a.width, 40))
    d = ImageDraw.Draw(W)
    d.text((14, 12), labels[0], fill=(40, 40, 40))
    d.text((24 + a.width, 12), labels[1], fill=(40, 40, 40))
    W.save(out, quality=88)


sheet = Image.open(CONCEPT).convert("RGB")
sx, sy = sheet.size[0] / 1536, sheet.size[1] / 1024
top_concept = sheet.crop((int(0 * sx), int(40 * sy), int(800 * sx), int(1000 * sy)))
side_concept = sheet.crop((int(1105 * sx), int(600 * sy), int(1536 * sx), int(940 * sy)))
top_build = Image.open(P / "top_down.png").convert("RGB")
w, h = top_build.size
top_build = top_build.crop((int(.08 * w), int(.12 * h), int(.92 * w), int(.95 * h)))
side_build = Image.open(P / "side_profile.png").convert("RGB")
side_by_side(top_concept, top_build, P / "concept_vs_build_top.jpg", ["CONCEPT top view", "BUILD top view (rest pose, Cycles)"])
side_by_side(side_concept, side_build, P / "concept_vs_build_side.jpg", ["CONCEPT side profile", "BUILD side profile (rest pose)"], h=520)
names = ["rest", "curled", "reach", "swim", "walk"]
tiles = [Image.open(P / f"pose_{n}_top.png").convert("RGB") for n in names]
strip = Image.new("RGB", (5 * 420, 460), (236, 234, 230))
d = ImageDraw.Draw(strip)
for i, (n, t) in enumerate(zip(names, tiles)):
    t.thumbnail((420, 420))
    strip.paste(t, (i * 420, 36))
    d.text((i * 420 + 10, 12), n, fill=(40, 40, 40))
strip.save(P / "poses_top_strip.jpg", quality=86)
print(f"Packaged {len(frames)} swim frames, comparison sheets and the pose strip")
