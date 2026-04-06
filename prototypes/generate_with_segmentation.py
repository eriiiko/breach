"""
Generate level art using SEGMENTATION ControlNet with the material map directly.

The Breach material map (colored regions for each material) is used as a
segmentation mask input to ControlNet. This should give much stronger spatial
constraints than canny edges, since each colored region tells the model
what kind of surface that area should be.

Output: prototypes/seg_variations/
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seg_variations")
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_MAP = "C:/Users/steen/projects/breach/levels/Ship_walls_mats_test_20260406.png"

MAT_AIR = (0, 0, 0)


def load_material_map(path):
    return np.array(Image.open(path).convert('RGB'))


def crop_to_content(mat_rgb, margin=2):
    is_content = np.any(mat_rgb != np.array(MAT_AIR), axis=-1)
    if not is_content.any():
        return mat_rgb
    rows = np.any(is_content, axis=1)
    cols = np.any(is_content, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(mat_rgb.shape[0] - 1, y1 + margin)
    x1 = min(mat_rgb.shape[1] - 1, x1 + margin)
    return mat_rgb[y0:y1+1, x0:x1+1]


def main():
    print("=== Breach Level Generation (Segmentation ControlNet) ===\n")

    # Load and crop the material map
    print(f"Loading material map: {INPUT_MAP}")
    mat_rgb = load_material_map(INPUT_MAP)
    print(f"  Original size: {mat_rgb.shape}")

    mat_rgb = crop_to_content(mat_rgb, margin=2)
    print(f"  Cropped size:  {mat_rgb.shape}")

    # Compute upscale factor
    h, w = mat_rgb.shape[:2]
    max_dim = max(h, w)
    scale = max(1, 1024 // max_dim)
    out_w = (w * scale // 8) * 8
    out_h = (h * scale // 8) * 8
    target_size = (out_w, out_h)
    print(f"  Scale factor: {scale}x")
    print(f"  Target output size: {target_size}")

    # Upscale the material map (NEAREST = preserve hard color edges)
    seg_img = Image.fromarray(
        np.repeat(np.repeat(mat_rgb, scale, axis=0), scale, axis=1)
    )
    if seg_img.size != target_size:
        seg_img = seg_img.resize(target_size, Image.NEAREST)
    seg_path = os.path.join(OUT_DIR, "00_segmentation_input.png")
    seg_img.save(seg_path)
    print(f"  Segmentation input saved: {seg_path}")

    # Load models
    print("\nLoading SDXL + Segmentation ControlNet (~30s)...")
    from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

    controlnet = ControlNetModel.from_pretrained(
        "C:/Users/steen/projects/breach/tools/models/controlnet-seg-sdxl",
        torch_dtype=torch.float16,
        use_safetensors=False,  # SargeZT model uses .bin not .safetensors
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

    # Generate variations
    prompt = (
        "top-down 2D game map, dark spaceship interior, narrow corridors, "
        "metallic walls, pixel art game asset, sci-fi horror, "
        "dimly lit, atmospheric, dead space style, grimy floors"
    )
    negative = "blurry, low quality, text, watermark, 3d render, photo, characters, people"

    variations = [
        {"name": "01_seg_strength_1.0_seed42",  "scale": 1.0, "seed": 42},
        {"name": "02_seg_strength_1.5_seed42",  "scale": 1.5, "seed": 42},
        {"name": "03_seg_strength_0.8_seed42",  "scale": 0.8, "seed": 42},
        {"name": "04_seg_strength_1.0_seed7",   "scale": 1.0, "seed": 7},
        {"name": "05_seg_strength_1.0_seed999", "scale": 1.0, "seed": 999},
    ]

    print(f"\nGenerating {len(variations)} variations at {target_size}...\n")
    for i, var in enumerate(variations, 1):
        print(f"[{i}/{len(variations)}] {var['name']}")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=seg_img,
            num_inference_steps=25,
            controlnet_conditioning_scale=var["scale"],
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(var["seed"]),
        ).images[0]
        out = os.path.join(OUT_DIR, f"{var['name']}.png")
        result.save(out)
        print(f"  Saved: {out}\n")

    print("Done! See prototypes/seg_variations/")


if __name__ == "__main__":
    main()
