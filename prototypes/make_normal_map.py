"""
Quick Sobel-based normal map generator.

Usage:
    python make_normal_map.py input.png [output.png]

Reads a diffuse image, computes a height map from luminance, then derives
a normal map via Sobel filtering. Rough and fast — not as nice as Laigter
but enough to play with the lighting system.
"""

import sys
import os
import numpy as np
from PIL import Image


def make_normal_map(diffuse_path, output_path=None, strength=4.0):
    img = Image.open(diffuse_path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0

    # Luminance as height map (BT.709 weights)
    height = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2])

    # Sobel kernels
    h, w = height.shape
    # Pad with edge replication
    p = np.pad(height, 1, mode='edge')

    # dx (horizontal gradient)
    dx = (
        -1.0 * p[1:h+1, 0:w]   + 1.0 * p[1:h+1, 2:w+2]
        + -2.0 * p[1:h+1, 0:w] + 2.0 * p[1:h+1, 2:w+2]
        + -1.0 * p[1:h+1, 0:w] + 1.0 * p[1:h+1, 2:w+2]
    ) / 8.0

    # dy (vertical gradient)
    dy = (
        -1.0 * p[0:h, 1:w+1]   + 1.0 * p[2:h+2, 1:w+1]
        + -2.0 * p[0:h, 1:w+1] + 2.0 * p[2:h+2, 1:w+1]
        + -1.0 * p[0:h, 1:w+1] + 1.0 * p[2:h+2, 1:w+1]
    ) / 8.0

    # Build normals: (-dx, -dy, 1/strength), normalize
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(nx)

    length = np.sqrt(nx*nx + ny*ny + nz*nz)
    nx /= length
    ny /= length
    nz /= length

    # Encode to 0-255 (XYZ -> RGB, with X,Y in [-1,1] mapped to [0,255], Z [0,1] -> [128,255])
    normal_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    normal_rgb[..., 0] = np.clip((nx * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    normal_rgb[..., 1] = np.clip((ny * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    normal_rgb[..., 2] = np.clip((nz * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)

    if output_path is None:
        base, ext = os.path.splitext(diffuse_path)
        output_path = f"{base}_normal.png"

    Image.fromarray(normal_rgb).save(output_path)
    print(f"Normal map: {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_normal_map.py input.png [output.png]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    make_normal_map(inp, out)
