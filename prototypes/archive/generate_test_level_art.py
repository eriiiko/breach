"""
Generate test level art for Breach using ControlNet + SDXL.

Creates a random ship material grid, renders it as a conditioning image,
and uses ControlNet to generate sci-fi ship interior art that respects
the wall layout.

Usage:
    python generate_test_level_art.py

Output:
    prototypes/test_level_material_map.png   -- the conditioning image
    prototypes/test_level_generated.png      -- the AI-generated art
"""

import numpy as np
from PIL import Image, ImageDraw
import sys
import os

# Material IDs (matching Breach)
MAT_AIR = 0
MAT_HULL = 1
MAT_WOOD = 2
MAT_DOOR = 3

# Colors for the conditioning image (matching _draw_map)
MAT_COLORS = {
    MAT_AIR:  (30, 35, 45),    # dark floor
    MAT_HULL: (80, 80, 90),    # gray hull
    MAT_WOOD: (70, 50, 30),    # brown wood
    MAT_DOOR: (50, 70, 50),    # green door
}


def generate_random_ship(width=40, height=25, coarse=3):
    """Generate a random ship material grid.

    Uses coarse tiles (like the game) then expands to fine tiles.
    Creates hull border, carves rooms, adds interior walls with doors.
    """
    fw, fh = width * coarse, height * coarse
    mat = np.full((fh, fw), MAT_AIR, dtype=np.int8)

    # Hull border (2 coarse tiles thick)
    hull_thickness = 2
    for t in range(hull_thickness * coarse):
        mat[t, :] = MAT_HULL
        mat[-(t+1), :] = MAT_HULL
        mat[:, t] = MAT_HULL
        mat[:, -(t+1)] = MAT_HULL

    # Mark exterior as vacuum (outside hull)
    # For the conditioning image we just leave it as air (dark)
    # but we want the hull border visible
    exterior = hull_thickness * coarse

    # Add some random vertical walls (wood)
    rng = np.random.default_rng(42)
    n_vwalls = rng.integers(3, 7)
    vwall_positions = sorted(rng.choice(
        range(exterior + 4*coarse, fw - exterior - 4*coarse, coarse),
        size=min(n_vwalls, 6), replace=False
    ))

    for wx in vwall_positions:
        # Wall spans most of height, with gaps for doors
        for fy in range(exterior, fh - exterior):
            mat[fy, wx] = MAT_WOOD

        # Add 1-2 doors (3 tiles wide for unit passage)
        n_doors = rng.integers(1, 3)
        door_positions = rng.choice(
            range(exterior + 2*coarse, fh - exterior - 2*coarse, coarse),
            size=n_doors, replace=False
        )
        for dy in door_positions:
            for d in range(coarse):
                mat[dy + d, wx] = MAT_DOOR

    # Add some horizontal walls
    n_hwalls = rng.integers(2, 5)
    hwall_positions = sorted(rng.choice(
        range(exterior + 3*coarse, fh - exterior - 3*coarse, coarse),
        size=min(n_hwalls, 4), replace=False
    ))

    for wy in hwall_positions:
        # Only span between some vertical walls (creates rooms)
        start_x = exterior
        end_x = rng.choice(vwall_positions) if len(vwall_positions) > 0 else fw - exterior
        for fx in range(start_x, min(end_x + 1, fw - exterior)):
            # Don't overwrite hull
            if mat[wy, fx] == MAT_AIR:
                mat[wy, fx] = MAT_WOOD

        # Door in horizontal wall
        door_x = rng.integers(start_x + coarse, min(end_x, fw - exterior - coarse))
        door_x = (door_x // coarse) * coarse  # align to grid
        for d in range(coarse):
            if door_x + d < fw:
                mat[wy, door_x + d] = MAT_DOOR

    return mat


def material_grid_to_image(mat, scale=8):
    """Render material grid as a color image.

    Each fine tile becomes scale x scale pixels.
    """
    fh, fw = mat.shape
    img = Image.new('RGB', (fw * scale, fh * scale))
    pixels = np.zeros((fh * scale, fw * scale, 3), dtype=np.uint8)

    for mat_id, color in MAT_COLORS.items():
        mask = mat == mat_id
        # Expand mask to pixel resolution
        mask_expanded = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
        pixels[mask_expanded] = color

    img = Image.fromarray(pixels)
    return img


def material_grid_to_canny(mat, scale=8):
    """Convert material grid to a canny-style edge image for ControlNet.

    White edges where material changes, black elsewhere.
    """
    fh, fw = mat.shape

    # Find edges: where adjacent tiles have different materials
    edges = np.zeros((fh, fw), dtype=np.uint8)

    # Horizontal edges
    diff_h = (mat[1:, :] != mat[:-1, :]).astype(np.uint8) * 255
    edges[1:, :] = np.maximum(edges[1:, :], diff_h)
    edges[:-1, :] = np.maximum(edges[:-1, :], diff_h)

    # Vertical edges
    diff_v = (mat[:, 1:] != mat[:, :-1]).astype(np.uint8) * 255
    edges[:, 1:] = np.maximum(edges[:, 1:], diff_v)
    edges[:, :-1] = np.maximum(edges[:, :-1], diff_v)

    # Scale up
    edges_big = np.repeat(np.repeat(edges, scale, axis=0), scale, axis=1)
    return Image.fromarray(edges_big)


def generate_with_controlnet(canny_image, prompt, output_path):
    """Run ControlNet + SDXL to generate ship interior art."""
    import torch
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    print("Loading ControlNet model...")
    controlnet = ControlNetModel.from_pretrained(
        "C:/Users/steen/projects/breach/tools/models/controlnet-canny-sdxl",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
        "C:/Users/steen/projects/breach/tools/models/sd_xl_base_1.0.safetensors",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.to("cuda")

    # Enable memory optimizations for 8GB VRAM
    pipe.enable_model_cpu_offload()

    print(f"Generating image ({canny_image.size})...")
    # Resize canny to something SDXL can handle (must be multiple of 8)
    w, h = canny_image.size
    target_w = min(1024, (w // 8) * 8)
    target_h = min(1024, (h // 8) * 8)
    # Keep aspect ratio
    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio) // 8 * 8
    new_h = int(h * ratio) // 8 * 8
    canny_resized = canny_image.resize((new_w, new_h), Image.NEAREST)

    # Convert to RGB (ControlNet expects 3 channels)
    if canny_resized.mode != 'RGB':
        canny_resized = canny_resized.convert('RGB')

    result = pipe(
        prompt=prompt,
        negative_prompt="blurry, low quality, text, watermark, 3d render, photo",
        image=canny_resized,
        num_inference_steps=30,
        controlnet_conditioning_scale=0.7,
        guidance_scale=7.5,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]

    result.save(output_path)
    print(f"Saved generated image to: {output_path}")
    return result


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=== Breach Test Level Art Generator ===\n")

    # Step 1: Generate random ship layout
    print("Step 1: Generating random ship layout...")
    mat = generate_random_ship(width=40, height=25, coarse=3)
    print(f"  Material grid: {mat.shape} (fine tiles)")
    print(f"  Hull tiles: {np.sum(mat == MAT_HULL)}")
    print(f"  Wood tiles: {np.sum(mat == MAT_WOOD)}")
    print(f"  Door tiles: {np.sum(mat == MAT_DOOR)}")
    print(f"  Air tiles:  {np.sum(mat == MAT_AIR)}")

    # Step 2: Render as images
    print("\nStep 2: Rendering conditioning images...")
    color_img = material_grid_to_image(mat, scale=8)
    color_path = os.path.join(out_dir, "test_level_material_map.png")
    color_img.save(color_path)
    print(f"  Color map: {color_path} ({color_img.size})")

    canny_img = material_grid_to_canny(mat, scale=8)
    canny_path = os.path.join(out_dir, "test_level_canny.png")
    canny_img.save(canny_path)
    print(f"  Canny edges: {canny_path} ({canny_img.size})")

    # Step 3: Generate with ControlNet (if available)
    print("\nStep 3: Attempting ControlNet generation...")
    controlnet_model = "C:/Users/steen/projects/breach/tools/models/controlnet-canny-sdxl"
    if os.path.exists(os.path.join(controlnet_model, "config.json")):
        prompt = (
            "top-down view of a derelict spaceship interior, dark sci-fi corridors, "
            "metal walls, emergency lighting, pixel art style, detailed, "
            "atmospheric, dark ambient, rust and decay, game asset"
        )
        output_path = os.path.join(out_dir, "test_level_generated.png")
        generate_with_controlnet(canny_img, prompt, output_path)
    else:
        print("  ControlNet model not yet downloaded. Skipping generation.")
        print(f"  (Expected at: {controlnet_model})")
        print("  Run this script again after the model downloads.")

    print("\nDone! Check the output files in prototypes/")


if __name__ == "__main__":
    main()
