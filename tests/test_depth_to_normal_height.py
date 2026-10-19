"""Authored relief must retain the baker's direction and coverage contract."""
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


TOOL = Path(__file__).resolve().parents[1] / "tools" / "depth_to_normal.py"


def convert(tmp_path, height, *extra):
    source = tmp_path / "source.png"
    normal = tmp_path / "normal.png"
    Image.fromarray(height).save(source)
    subprocess.run([sys.executable, str(TOOL), str(source),
                    str(tmp_path / "height.png"), str(normal),
                    "--height-input", "--strength", "8", *extra],
                   check=True, capture_output=True)
    return np.asarray(Image.open(normal))


def test_flat_relief_encodes_surface_normal(tmp_path):
    normal = convert(tmp_path, np.full((16, 16), 100, dtype=np.uint8), "--tileable")
    assert np.all(normal == (128, 128, 255))


def test_rising_relief_points_towards_lower_columns_and_rows(tmp_path):
    yy, xx = np.indices((16, 16))
    normal = convert(tmp_path, ((xx + yy) * 7).astype(np.uint8))
    assert np.all(normal[1:-1, 1:-1, :2] < 128)
    decoded = normal.astype(np.float32) / 127.5 - 1
    assert np.max(np.abs(np.linalg.norm(decoded, axis=-1) - 1)) < 0.007


def test_periodic_relief_is_translation_equivariant(tmp_path):
    yy, xx = np.indices((16, 16))
    height = ((np.sin(xx * 2 * np.pi / 16) + np.cos(yy * 2 * np.pi / 16)) * 40 + 128).astype(np.uint8)
    normal = convert(tmp_path, height, "--tileable", "--blur", "0.8")
    shifted = convert(tmp_path, np.roll(height, 3, axis=1), "--tileable", "--blur", "0.8")
    assert np.array_equal(shifted, np.roll(normal, 3, axis=1))


def test_transparent_rug_keeps_albedo_coverage(tmp_path):
    rgba = np.full((16, 16, 4), 255, dtype=np.uint8)
    rgba[:3, :, 3] = 0
    albedo = tmp_path / "albedo.png"
    Image.fromarray(rgba).save(albedo)
    normal = convert(tmp_path, np.full((16, 16), 100, dtype=np.uint8),
                     "--alpha-from", str(albedo))
    assert np.array_equal(normal[..., 3], rgba[..., 3])
    assert np.all(normal[:3, :, :3] == (128, 128, 255))


def test_smoothing_preserves_relief_smaller_than_one_eight_bit_step(tmp_path):
    """A uint8 intermediate must fail: it erases this shallow 16-bit ramp."""
    height = np.tile(np.arange(32, dtype=np.uint16) + 30000, (16, 1))
    convert(tmp_path, height, "--blur", "0.8")
    saved = np.asarray(Image.open(tmp_path / "height.png"))
    assert np.all(np.diff(saved[8, 4:-4].astype(np.int64)) > 0)
    assert np.max(np.abs(saved[8, 4:-4].astype(np.int64) - height[8, 4:-4])) <= 1
