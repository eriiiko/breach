"""
Generate level art from a real Breach material map (1 pixel = 1 tile).

Loads a hand-painted material map (using the Breach palette), then:
1. Generates canny edges from material transitions
2. Tries ControlNet (canny) generation
3. Tries img2img generation with the material map as init image
4. (Optional) tries different ControlNet strengths

All variations saved to prototypes/level_variations/
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "level_variations")
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_MAP = "C:/Users/steen/projects/breach/levels/Ship_walls_mats_test_20260406.png"

# Breach material palette
MAT_AIR   = (0, 0, 0)
MAT_HULL  = (80, 80, 90)
MAT_WOOD  = (70, 50, 30)
MAT_DOOR  = (50, 120, 50)
MAT_STEEL = (140, 140, 140)
MAT_GLASS = (80, 180, 200)


def load_material_map(path):
    img = Image.open(path).convert('RGB')
    return np.array(img)


def crop_to_content(mat_rgb, margin=2):
    """Crop the material map to the bounding box of non-air pixels.

    Adds a small margin (in tiles) around the content. This dramatically
    reduces wasted resolution when the ship is small relative to the canvas.
    """
    # Find non-air pixels
    is_content = np.any(mat_rgb != np.array(MAT_AIR), axis=-1)
    if not is_content.any():
        return mat_rgb

    rows = np.any(is_content, axis=1)
    cols = np.any(is_content, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]

    # Add margin and clip
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(mat_rgb.shape[0] - 1, y1 + margin)
    x1 = min(mat_rgb.shape[1] - 1, x1 + margin)

    return mat_rgb[y0:y1+1, x0:x1+1]


def material_to_id(mat_rgb):
    """Convert RGB material map to int8 material ID array."""
    h, w, _ = mat_rgb.shape
    ids = np.zeros((h, w), dtype=np.int8)
    palette = [
        (MAT_AIR, 0),
        (MAT_HULL, 1),
        (MAT_WOOD, 2),
        (MAT_DOOR, 3),
        (MAT_STEEL, 4),
        (MAT_GLASS, 5),
    ]
    for color, mid in palette:
        mask = np.all(mat_rgb == color, axis=-1)
        ids[mask] = mid
    return ids


def material_to_canny(mat_rgb, scale=8):
    """Generate canny-style edge image from material transitions."""
    ids = material_to_id(mat_rgb)
    h, w = ids.shape
    edges = np.zeros((h, w), dtype=np.uint8)

    diff_h = (ids[1:, :] != ids[:-1, :]).astype(np.uint8) * 255
    edges[1:, :] = np.maximum(edges[1:, :], diff_h)
    edges[:-1, :] = np.maximum(edges[:-1, :], diff_h)

    diff_v = (ids[:, 1:] != ids[:, :-1]).astype(np.uint8) * 255
    edges[:, 1:] = np.maximum(edges[:, 1:], diff_v)
    edges[:, :-1] = np.maximum(edges[:, :-1], diff_v)

    edges_big = np.repeat(np.repeat(edges, scale, axis=0), scale, axis=1)
    return Image.fromarray(edges_big)


def upscale_material_map(mat_rgb, scale=8):
    """Nearest-neighbor upscale of the material map for SDXL input."""
    h, w, _ = mat_rgb.shape
    big = np.repeat(np.repeat(mat_rgb, scale, axis=0), scale, axis=1)
    return Image.fromarray(big)


def main():
    print("=== Breach Level Generation from Material Map ===\n")

    # Step 1: Load and process the material map
    print(f"Loading material map: {INPUT_MAP}")
    mat_rgb = load_material_map(INPUT_MAP)
    print(f"  Original size: {mat_rgb.shape}")

    # Crop to content (the ship is much smaller than the canvas)
    mat_rgb = crop_to_content(mat_rgb, margin=2)
    print(f"  Cropped size:  {mat_rgb.shape}")

    # Choose scale so output fits in ~1024x1024 (SDXL native).
    # Pick the largest scale where both dims stay <= 1024 and are multiples of 8.
    h, w = mat_rgb.shape[:2]
    max_dim = max(h, w)
    scale = max(1, 1024 // max_dim)
    # Ensure resulting dims are multiples of 8 (SDXL requirement)
    out_w = (w * scale // 8) * 8
    out_h = (h * scale // 8) * 8
    target_size = (out_w, out_h)
    print(f"  Scale factor: {scale}x")
    print(f"  Target output size: {target_size}")

    # Step 2: Generate canny edges (then resize to exact target_size)
    canny_img = material_to_canny(mat_rgb, scale=scale)
    if canny_img.size != target_size:
        canny_img = canny_img.resize(target_size, Image.NEAREST)
    canny_path = os.path.join(OUT_DIR, "00_canny.png")
    canny_img.save(canny_path)
    print(f"  Canny saved: {canny_path} ({canny_img.size})")

    # Save upscaled material map (for img2img and reference)
    color_img = upscale_material_map(mat_rgb, scale=scale)
    if color_img.size != target_size:
        color_img = color_img.resize(target_size, Image.NEAREST)
    color_path = os.path.join(OUT_DIR, "00_material_upscaled.png")
    color_img.save(color_path)
    print(f"  Color map saved: {color_path}")

    canny_rgb = canny_img.convert('RGB') if canny_img.mode != 'RGB' else canny_img

    # Step 3: Load models
    print("\nLoading SDXL + ControlNet (this takes ~30s)...")
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    controlnet = ControlNetModel.from_pretrained(
        "C:/Users/steen/projects/breach/tools/models/controlnet-canny-sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        "C:/Users/steen/projects/breach/tools/models/sd_xl_base_1.0.safetensors",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    print("  Models loaded.")

    # Step 4: Generate variations
    prompt = (
        "top-down 2D game map, dark spaceship interior, narrow corridors, "
        "metallic walls, pixel art game asset, sci-fi horror, "
        "dimly lit, atmospheric, dead space style, grimy floors"
    )
    negative = "blurry, low quality, text, watermark, 3d render, photo, characters, people"

    variations = [
        {"name": "01_canny_strength_1.0_seed42", "scale": 1.0, "seed": 42},
        {"name": "02_canny_strength_1.3_seed42", "scale": 1.3, "seed": 42},
        {"name": "03_canny_strength_0.7_seed42", "scale": 0.7, "seed": 42},
        {"name": "04_canny_strength_1.0_seed7",  "scale": 1.0, "seed": 7},
        {"name": "05_canny_strength_1.0_seed999","scale": 1.0, "seed": 999},
    ]

    print(f"\nGenerating {len(variations)} variations at {target_size}...\n")
    for i, var in enumerate(variations, 1):
        print(f"[{i}/{len(variations)}] {var['name']}")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=canny_rgb,
            num_inference_steps=25,
            controlnet_conditioning_scale=var["scale"],
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(var["seed"]),
        ).images[0]
        out = os.path.join(OUT_DIR, f"{var['name']}.png")
        result.save(out)
        print(f"  Saved: {out}\n")

    print("Done! See prototypes/level_variations/")


if __name__ == "__main__":
    main()
