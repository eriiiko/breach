"""
Generate multiple level art variations to explore the prompt/strength space.

Loads SDXL + ControlNet once, then generates several images with different
prompts, seeds, and ControlNet conditioning scales. Much faster than running
the script multiple times.

Output: prototypes/variations/ directory with named PNGs.
"""

import os
import sys
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_test_level_art import (
    generate_random_ship,
    material_grid_to_image,
    material_grid_to_canny,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variations")
os.makedirs(OUT_DIR, exist_ok=True)

# Variations to generate
VARIATIONS = [
    {
        "name": "01_horror_high_strength",
        "prompt": "top-down 2D game map, dark spaceship interior, cramped corridors, "
                  "exposed pipes, blood stains, sci-fi horror, dimly lit, "
                  "metallic floors, pixel art game asset, dead space style",
        "controlnet_scale": 1.0,
        "seed": 42,
    },
    {
        "name": "02_horror_low_strength",
        "prompt": "top-down 2D game map, dark spaceship interior, cramped corridors, "
                  "exposed pipes, blood stains, sci-fi horror, dimly lit, "
                  "metallic floors, pixel art game asset, dead space style",
        "controlnet_scale": 0.5,
        "seed": 42,
    },
    {
        "name": "03_blueprint_style",
        "prompt": "spaceship deck plan, top-down floor plan, technical blueprint, "
                  "corridors and rooms clearly visible, sci-fi schematic, "
                  "labelled compartments, dark background, thin lines",
        "controlnet_scale": 1.2,
        "seed": 7,
    },
    {
        "name": "04_pixel_art_tilemap",
        "prompt": "2D pixel art tilemap, top-down spaceship interior, tile based game, "
                  "rpg maker style, clearly visible walls and floors, retro game art, "
                  "16-bit aesthetic",
        "controlnet_scale": 1.0,
        "seed": 123,
    },
    {
        "name": "05_alien_derelict",
        "prompt": "top-down view of derelict alien spaceship interior, organic walls, "
                  "biomechanical, dark atmospheric, eggs in corners, dripping fluids, "
                  "horror game art",
        "controlnet_scale": 0.9,
        "seed": 999,
    },
]


def main():
    print("=== Breach Level Art Variations ===\n")

    # Generate the random ship layout (same for all variations)
    print("Step 1: Generating random ship layout...")
    mat = generate_random_ship(width=40, height=25, coarse=3)
    print(f"  Material grid: {mat.shape}")

    # Save canny image
    canny_img = material_grid_to_canny(mat, scale=8)
    canny_path = os.path.join(OUT_DIR, "00_canny_input.png")
    canny_img.save(canny_path)
    print(f"  Canny saved: {canny_path}")

    # Save color map for reference
    color_img = material_grid_to_image(mat, scale=8)
    color_img.save(os.path.join(OUT_DIR, "00_material_map.png"))

    # Resize canny to SDXL-friendly resolution
    # Keep aspect ratio close to 16:10 (our 40x25 grid) but use multiples of 8
    target_w, target_h = 1024, 640  # 1.6 ratio matches our 40:25
    canny_resized = canny_img.resize((target_w, target_h), Image.NEAREST)
    if canny_resized.mode != 'RGB':
        canny_resized = canny_resized.convert('RGB')

    # Step 2: Load models (once)
    print("\nStep 2: Loading models (one-time)...")
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
    print("  Models loaded.\n")

    # Step 3: Generate each variation
    print(f"Step 3: Generating {len(VARIATIONS)} variations at {target_w}x{target_h}...\n")
    for i, var in enumerate(VARIATIONS, 1):
        print(f"[{i}/{len(VARIATIONS)}] {var['name']}")
        print(f"  Prompt: {var['prompt'][:80]}...")
        print(f"  ControlNet scale: {var['controlnet_scale']}, seed: {var['seed']}")

        result = pipe(
            prompt=var["prompt"],
            negative_prompt="blurry, low quality, text, watermark, 3d render, photo, "
                           "human figures, characters",
            image=canny_resized,
            num_inference_steps=25,
            controlnet_conditioning_scale=var["controlnet_scale"],
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(var["seed"]),
        ).images[0]

        out_path = os.path.join(OUT_DIR, f"{var['name']}.png")
        result.save(out_path)
        print(f"  Saved: {out_path}\n")

    print("Done! Check prototypes/variations/")


if __name__ == "__main__":
    main()
