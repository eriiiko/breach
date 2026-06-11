"""Register + tone-match the level art layers onto the bare layer's frame.

Pipeline §5 step 0 (docs/level_editor_and_format_v2_proposal.md): the three
ship layers are separate generations — different sizes by a few pixels and
slightly different exposure. Per-tile layer patching needs them pixel-aligned
and tone-consistent, so: resize to the reference frame, estimate the residual
translation by phase correlation, shift, then histogram-match to the
reference. Writes a QA difference image per layer so alignment is judged by
eye, not faith.

Usage:
    python tools/register_layers.py <ref.png> <layer.png> [...more layers]
Outputs (next to each input): <layer>_reg.png + <layer>_reg_qa.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import shift as nd_shift

try:
    from skimage.exposure import match_histograms
    from skimage.registration import phase_cross_correlation
    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False


def _gray(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=2)


def register(ref_path: str, layer_path: str) -> None:
    ref = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.float32) / 255
    img = Image.open(layer_path).convert("RGB")
    img = img.resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
    lay = np.asarray(img, dtype=np.float32) / 255

    if _HAVE_SKIMAGE:
        dyx, err, _ = phase_cross_correlation(
            _gray(ref), _gray(lay), upsample_factor=10)
        lay = np.stack(
            [nd_shift(lay[..., c], dyx, order=1, mode="nearest")
             for c in range(3)], axis=-1)
        lay = match_histograms(lay, ref, channel_axis=-1)
        note = f"shift (dy, dx) = ({dyx[0]:+.1f}, {dyx[1]:+.1f}) px, err {err:.3f}"
    else:  # scipy-only fallback: mean/std tone match, no sub-pixel shift
        lay = (lay - lay.mean((0, 1))) / (lay.std((0, 1)) + 1e-6)
        lay = lay * ref.std((0, 1)) + ref.mean((0, 1))
        note = "skimage missing — resized + mean/std tone match only"

    lay = np.clip(lay, 0.0, 1.0)
    stem = Path(layer_path).with_suffix("")
    Image.fromarray((lay * 255 + 0.5).astype(np.uint8)).save(f"{stem}_reg.png")
    qa = np.abs(_gray(ref) - _gray(lay))
    qa = (np.clip(qa * 4.0, 0, 1) * 255).astype(np.uint8)  # x4 gain: misalign glows
    Image.fromarray(qa).save(f"{stem}_reg_qa.png")
    print(f"{Path(layer_path).name}: {note} -> {stem.name}_reg.png (+_qa)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    for p in sys.argv[2:]:
        register(sys.argv[1], p)
