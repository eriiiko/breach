"""The G1 billow-noise baker (renderer/noise_bake.py, graphics arc G1,
docs/graphics_direction_2026-10-18.md §2.1, §8.2). Headless: the module is
pyray-free and its bakes are plain numpy arrays.

The relief shader samples the billow texture with REPEAT wrapping over the
whole ship, so a seam would draw a straight line through every plume; and
replays/spectators must see the same smoke, so the bake must not depend on
the run. Those two are what these tests hold.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from renderer import noise_bake as nb


def _random_coords(n: int, dim: int, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).random((n, dim)) * 3.0 - 1.0


@pytest.mark.parametrize("dim", [2, 3])
def test_billow_field_is_periodic_on_every_axis(dim):
    """Property: the billow field is EXACTLY periodic with period 1 on every
    axis -- f(x + e_a) == f(x) -- in 2D and in the 3D tilt variant, so any
    bake of it tiles. Breaks if any ingredient's lattice stops wrapping (a
    non-integer Perlin/Worley period, a warp that is not itself periodic, a
    feature-point table indexed without the modulo)."""
    x = _random_coords(400, dim)
    f0 = nb.billow_field(x)
    for a in range(dim):
        shifted = x.copy()
        shifted[:, a] += 1.0
        np.testing.assert_allclose(nb.billow_field(shifted), f0, atol=1e-9)


def _seam_and_interior(img: np.ndarray, axis: int):
    """(mean |step| across the wrap seam, mean |step| between interior
    neighbours) along ``axis``."""
    a = img.astype(np.int32)
    seam = np.abs(np.take(a, 0, axis=axis) - np.take(a, -1, axis=axis))
    return float(seam.mean()), float(np.abs(np.diff(a, axis=axis)).mean())


# A seam is invisible when stepping across it is like stepping between any
# two neighbours; a broken wrap jumps between UNcorrelated texels, several
# times the neighbour step. 1.5 leaves room for one slice's sampling noise.
_SEAM_TOL = 1.5


def test_baked_texture_tiles_without_a_seam():
    """Property: across the wrap seam (last column -> first, last row ->
    first) every channel of the 2D bake steps, on average, like interior
    neighbours do -- a tiled copy shows no line. Breaks
    if the bake grid stops sampling the period [0, 1) (an endpoint-inclusive
    grid repeats a texel) or a channel stops being periodic."""
    tex = nb.bake_billow_rgba(128)
    for c in range(4):
        for axis in (0, 1):
            seam, inner = _seam_and_interior(tex[..., c], axis)
            assert seam <= _SEAM_TOL * inner, (
                f"channel {c} axis {axis}: seam {seam:.1f} vs {inner:.1f}")


def test_3d_bake_tiles_on_all_three_axes():
    """Property: the 3D variant's wrap step on each axis is like its interior
    neighbour step (it tiles as a cube for the tilt integrator).
    Breaks if the cube grid or any axis's lattice stops wrapping."""
    cube = nb.bake_billow_3d(32)
    for axis in range(3):
        seam, inner = _seam_and_interior(cube, axis)
        assert seam <= _SEAM_TOL * inner, (
            f"axis {axis}: seam {seam:.1f} vs {inner:.1f}")


def test_bakes_are_deterministic():
    """Property: two bakes with the same arguments are byte-identical (a
    replay or a spectator draws the same smoke). Breaks if a bake draws
    from an unseeded or shared global random stream."""
    np.testing.assert_array_equal(nb.bake_billow_rgba(64),
                                  nb.bake_billow_rgba(64))
    np.testing.assert_array_equal(nb.bake_billow_3d(16), nb.bake_billow_3d(16))


def test_billow_is_lumpy_not_smooth():
    """Property: the billow channel carries cellular FORM -- at least one
    local maximum per coarse Worley cell (the lumps), and a full value
    range. Breaks if the recipe degrades to a smooth low-frequency field
    (e.g. the Perlin clustering swamping the domes), which reads as fog
    rather than billows."""
    r = nb.bake_billow_rgba(128)[..., 0].astype(np.int32)
    p = np.pad(r, 1, mode="wrap")
    centre = p[1:-1, 1:-1]
    is_max = np.ones_like(centre, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                is_max &= centre > p[1 + dy:p.shape[0] - 1 + dy,
                                     1 + dx:p.shape[1] - 1 + dx]
    assert int(is_max.sum()) >= nb.BASE_CELLS ** 2
    assert r.max() - r.min() >= 200


def test_module_is_pyray_free():
    """Property: the baker imports no pyray / breach_physics, so it bakes
    and tests without a window. Breaks if GL code moves into it."""
    src = Path(nb.__file__).read_text(encoding="utf-8")
    assert "import pyray" not in src and "breach_physics" not in src
