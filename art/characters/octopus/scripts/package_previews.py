"""Assemble the actual Blender frames for a small offline HTML player and GIF."""
from pathlib import Path
from PIL import Image

HOME = Path(__file__).resolve().parent.parent
PREVIEW = HOME / "previews" / "arm"
paths = sorted((PREVIEW / "frames").glob("*.png"))
if len(paths) < 2:
    raise SystemExit("Run build_arm.py --animation first")
frames = [Image.open(p).convert("RGB") for p in paths]
frames[0].save(PREVIEW / "arm_loop.gif", save_all=True, append_images=frames[1:],
               duration=round(9000 / len(frames)), loop=0, optimize=False)
# JPEG frame strips make a scrubber work offline without video codecs/CDNs.
for path, frame in zip(paths, frames):
    frame.save(path.with_suffix(".jpg"), quality=86)
print(f"Packaged {len(frames)} actual mesh frames, 9-second loop")
