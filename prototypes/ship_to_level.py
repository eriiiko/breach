"""
Convert chatGPT spaceship image to a game level.
Outputs a material map as PNG and optionally a numpy array.
"""
from PIL import Image
import numpy as np

# --- Config ---
INPUT = "C:/Users/steen/projects/breach/chatgptSpaceShip1.png"
CORRIDOR_WIDTH_TILES = 6
OUTPUT_PNG = "C:/Users/steen/projects/breach/prototypes/ship_level_preview.png"
OUTPUT_NPY = "C:/Users/steen/projects/breach/levels/ship1_materials.npy"

# Material IDs (matching game.py)
MAT_AIR  = 0
MAT_HULL = 1
MAT_WOOD = 2  # interior walls
MAT_DOOR = 3

# --- Load and rotate ---
img = Image.open(INPUT)
arr = np.array(img)
arr = np.rot90(arr, k=-1)  # front points right
print(f"Image after rotation: {arr.shape}")

# --- Compute scale ---
# Find corridor width at center of ship
mid_col = arr.shape[1] // 2
alpha_col = arr[:, mid_col, 3]
bright_col = arr[:, mid_col, :3].mean(axis=1)
corridor = (alpha_col > 128) & (bright_col > 80)

# Find widest run
runs = []
start = None
for i, v in enumerate(corridor):
    if v and start is None: start = i
    if not v and start is not None:
        runs.append((start, i, i - start))
        start = None
if start is not None:
    runs.append((start, len(corridor), len(corridor) - start))
runs.sort(key=lambda x: -x[2])
corridor_px = runs[0][2]
scale = corridor_px / CORRIDOR_WIDTH_TILES
print(f"Corridor: {corridor_px}px -> {CORRIDOR_WIDTH_TILES} tiles, scale={scale:.1f} px/tile")

# --- Downscale ---
target_h = int(arr.shape[0] / scale)
target_w = int(arr.shape[1] / scale)
print(f"Level size: {target_w} x {target_h} tiles")

# Downscale by block-averaging
materials = np.full((target_h, target_w), MAT_AIR, dtype=np.int8)

for ty in range(target_h):
    for tx in range(target_w):
        # Source block
        sy0 = int(ty * scale)
        sy1 = min(int((ty + 1) * scale), arr.shape[0])
        sx0 = int(tx * scale)
        sx1 = min(int((tx + 1) * scale), arr.shape[1])

        block = arr[sy0:sy1, sx0:sx1]
        if block.size == 0:
            continue

        avg_alpha = block[:, :, 3].mean()
        avg_brightness = block[:, :, :3].mean()
        avg_r = block[:, :, 0].mean()
        avg_g = block[:, :, 1].mean()
        avg_b = block[:, :, 2].mean()

        if avg_alpha < 100:
            # Transparent = vacuum (outside ship)
            materials[ty, tx] = MAT_AIR  # will be vacuum
        elif avg_brightness < 45:
            # Very dark = hull walls
            materials[ty, tx] = MAT_HULL
        elif avg_brightness < 75:
            # Medium dark = interior walls / hull
            # Check if it's more gray (hull) or colored (interior wall)
            color_variance = np.std([avg_r, avg_g, avg_b])
            if color_variance < 15:
                materials[ty, tx] = MAT_HULL
            else:
                materials[ty, tx] = MAT_WOOD
        else:
            # Bright = interior air
            materials[ty, tx] = MAT_AIR

# --- Mark vacuum: air tiles outside the ship ---
# Flood fill from corners to find exterior
from collections import deque

is_exterior = np.zeros_like(materials, dtype=bool)
queue = deque()

# Seed from all edges
for x in range(target_w):
    for y in [0, target_h - 1]:
        if materials[y, x] == MAT_AIR:
            queue.append((y, x))
            is_exterior[y, x] = True
for y in range(target_h):
    for x in [0, target_w - 1]:
        if materials[y, x] == MAT_AIR:
            queue.append((y, x))
            is_exterior[y, x] = True

while queue:
    cy, cx = queue.popleft()
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = cy+dy, cx+dx
        if 0 <= ny < target_h and 0 <= nx < target_w:
            if not is_exterior[ny, nx] and materials[ny, nx] == MAT_AIR:
                is_exterior[ny, nx] = True
                queue.append((ny, nx))

# Count
n_exterior = is_exterior.sum()
n_hull = (materials == MAT_HULL).sum()
n_wood = (materials == MAT_WOOD).sum()
n_interior = ((materials == MAT_AIR) & ~is_exterior).sum()
print(f"Exterior (vacuum): {n_exterior}, Hull: {n_hull}, Wood: {n_wood}, Interior air: {n_interior}")

# --- Generate preview image ---
preview = np.zeros((target_h, target_w, 3), dtype=np.uint8)
preview[is_exterior] = [0, 0, 0]           # vacuum = black
preview[materials == MAT_HULL] = [80, 85, 95]   # hull = gray
preview[materials == MAT_WOOD] = [140, 100, 60] # wood = brown
interior_air = (materials == MAT_AIR) & ~is_exterior
preview[interior_air] = [40, 45, 50]        # interior = dark

# Scale up for visibility
preview_img = Image.fromarray(preview)
preview_img = preview_img.resize((target_w * 4, target_h * 4), Image.NEAREST)
preview_img.save(OUTPUT_PNG)
print(f"Preview saved to {OUTPUT_PNG}")

# --- Save material array ---
import os
os.makedirs(os.path.dirname(OUTPUT_NPY), exist_ok=True)

# Save with vacuum info
level_data = {
    'materials': materials,
    'is_exterior': is_exterior,
    'width': target_w,
    'height': target_h,
}
np.save(OUTPUT_NPY, level_data)
print(f"Level data saved to {OUTPUT_NPY}")
print("Done!")
