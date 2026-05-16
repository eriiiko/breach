"""Step 2: Render tilemap CSV to ship_final.png (fixed uint8 overflow with np.clip)."""
import numpy as np
from PIL import Image

TILE = 32

grid = np.loadtxt("tilemap.csv", delimiter=",", dtype=int)
H, W = grid.shape


def make_noise(seed):
    rng = np.random.RandomState(seed)
    return rng.rand(TILE, TILE)


def make_tile(t):
    n = make_noise(int(t) * 13)
    img = np.zeros((TILE, TILE, 4), dtype=np.uint8)

    def base(color):
        img[:, :, 0] = color[0]
        img[:, :, 1] = color[1]
        img[:, :, 2] = color[2]
        img[:, :, 3] = 255

    if t == 0:
        return Image.fromarray(img)

    elif t == 1:  # WALL
        base([210, 210, 215])
        img[0:2, :, :3] = np.clip(img[0:2, :, :3].astype(int) - 40, 0, 255).astype(np.uint8)
        img[-2:, :, :3] = np.clip(img[-2:, :, :3].astype(int) - 40, 0, 255).astype(np.uint8)
        img[:, 0:2, :3] = np.clip(img[:, 0:2, :3].astype(int) - 40, 0, 255).astype(np.uint8)
        img[:, -2:, :3] = np.clip(img[:, -2:, :3].astype(int) - 40, 0, 255).astype(np.uint8)
        img[:, :, 0] = np.clip(img[:, :, 0].astype(int) - (n * 20).astype(int), 0, 255).astype(np.uint8)

    elif t == 2:  # FLOOR
        base([225, 225, 230])
        seams = (np.indices((TILE, TILE)).sum(axis=0) % 8 == 0)
        img[seams] = np.clip(img[seams].astype(int) - 15, 0, 255).astype(np.uint8)

    elif t == 3:  # DOOR
        base([180, 160, 120])
        img[:, TILE // 2 - 2:TILE // 2 + 2, :] = [220, 200, 140, 255]

    elif t == 4:  # LAB
        base([190, 220, 230])
        img[:, :, 2] = np.clip(img[:, :, 2].astype(int) + (n * 20).astype(int), 0, 255).astype(np.uint8)

    elif t == 5:  # PLANTS
        base([120, 180, 120])
        img[:, :, 1] = np.clip(img[:, :, 1].astype(int) + (n * 40).astype(int), 0, 255).astype(np.uint8)

    elif t == 6:  # STORAGE
        base([170, 140, 110])
        img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + (n * 30).astype(int), 0, 255).astype(np.uint8)

    elif t == 7:  # COCKPIT
        base([120, 140, 170])
        img[:, :, 2] = np.clip(img[:, :, 2].astype(int) + (n * 30).astype(int), 0, 255).astype(np.uint8)

    elif t == 8:  # SANITARY
        base([230, 230, 235])
        img[:, :, 1] = np.clip(img[:, :, 1].astype(int) + (n * 10).astype(int), 0, 255).astype(np.uint8)

    return Image.fromarray(img)


img = Image.new("RGBA", (W * TILE, H * TILE))

for r in range(H):
    for c in range(W):
        tile_img = make_tile(grid[r, c])
        img.paste(tile_img, (c * TILE, r * TILE))

img.save("ship_final.png")
print(f"Done! Saved ship_final.png ({W*TILE}x{H*TILE} pixels)")
