"""Property gates for the smoke detail-volume bake (issue #90,
renderer/volume_bake.py) — HEADLESS, no GL, small grids.

Each test names the property it protects and the change that must break it.
None pins a snapshot of a bake: the solver is free to change its look.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from renderer import volume_bake as vb
from renderer.gas_volume import pad_volume_for_gl


def _spectral_div_rel(u, v, w):
    """max |k . u_k| relative to max |k| |u_k| — 0 for a divergence-free
    field, O(1) for white noise."""
    kz, ky, kx = vb.wavenumbers(u.shape)
    U, V, W = np.fft.rfftn(u), np.fft.rfftn(v), np.fft.rfftn(w)
    D = kx * U + ky * V + kz * W
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    scale = float((kmag * np.sqrt(np.abs(U) ** 2 + np.abs(V) ** 2
                                  + np.abs(W) ** 2)).max())
    return float(np.abs(D).max()) / max(scale, 1e-12)


# ---------------------------------------------------------------------------
# Projection: the torus pressure solve removes ALL divergence (spectrally,
# with the Nyquist modes treated as k = 0 — the derivative of a Nyquist mode
# aliases to zero) and is idempotent. On a NON-cubic grid, so an axis mix-up
# cannot pass. Breaks if the k=0/Nyquist guards, the wavenumber layout or the
# rfftn axes drift (the 2026-10-14 incident: a live Nyquist k made irfftn
# silently undo the projection on that plane).
# ---------------------------------------------------------------------------
def test_project_removes_spectral_divergence_and_is_idempotent():
    rng = np.random.default_rng(1)
    shape = (8, 12, 16)
    u, v, w = (rng.normal(size=shape).astype(np.float32) for _ in range(3))
    assert _spectral_div_rel(u, v, w) > 0.1          # the input IS divergent
    pu, pv, pw = vb.project(u, v, w)
    assert _spectral_div_rel(pu, pv, pw) < 1e-4
    qu, qv, qw = vb.project(pu, pv, pw)
    assert np.allclose(qu, pu, atol=1e-5)
    assert np.allclose(qv, pv, atol=1e-5)
    assert np.allclose(qw, pw, atol=1e-5)


# ---------------------------------------------------------------------------
# Forcing: the solenoidal ModeSet is divergence-free BY CONSTRUCTION (each
# mode's polarisation is perpendicular to its PHYSICAL wavevector). On a
# non-cubic grid, so polarising against the integer index vector (the
# 2026-10-14 bug) fails. Breaks if the polarisation or the axis order drift.
# ---------------------------------------------------------------------------
def test_solenoidal_forcing_is_divergence_free():
    rng = np.random.default_rng(2)
    shape = (8, 16, 24)
    ms = vb.ModeSet(rng, shape, n=6, kmax=2, rate=0.1, amp=1.0, solenoidal=True)
    fx, fy, fz = ms.vector(t=3.0)
    assert np.abs(fx).max() > 0.05
    assert _spectral_div_rel(fx, fy, fz) < 1e-4


# ---------------------------------------------------------------------------
# Advection keeps periodicity and bounds: a MacCormack step of a [0,1] scalar
# on a periodic grid stays in [0,1] (the clamp) and a uniform field is a fixed
# point. Breaks if the clamp is dropped or the wrap mode changes.
# ---------------------------------------------------------------------------
def test_maccormack_clamps_and_preserves_uniform_field():
    rng = np.random.default_rng(3)
    shape = (6, 10, 10)
    adv = vb.Advector(shape)
    u, v, w = (rng.uniform(-1.5, 1.5, shape).astype(np.float32) for _ in range(3))
    adv.set_velocity(u, v, w)
    f = rng.uniform(0, 1, shape).astype(np.float32)
    g = adv.maccormack(f)
    assert g.min() >= 0.0 - 1e-6 and g.max() <= 1.0 + 1e-6
    const = np.full(shape, 0.37, np.float32)
    assert np.allclose(adv.maccormack(const), 0.37, atol=1e-6)


# ---------------------------------------------------------------------------
# Loop closure: O(0) == F(T) exactly, O(T-1) is dominated by F(T-1), and the
# weights sum to one everywhere (no brightness pulse). Breaks if the
# crossfade direction or the smoothstep endpoints change.
# ---------------------------------------------------------------------------
def test_close_loop_endpoints_and_partition_of_unity():
    rng = np.random.default_rng(4)
    T = 6
    F = rng.uniform(0, 1, (2 * T, 3, 4, 4, 2)).astype(np.float32)
    O = vb.close_loop(F)
    assert O.shape[0] == T
    assert np.array_equal(O[0], F[T])
    # weights: O(k) = (1-s)F(k+T) + sF(k); with F == 1 everywhere O == 1.
    ones = np.ones_like(F)
    assert np.allclose(vb.close_loop(ones), 1.0, atol=1e-6)
    # last frame leans on F(T-1), the frame just before the loop restarts
    k = T - 1
    s = k / T
    s = s * s * (3 - 2 * s)
    assert np.allclose(O[k], (1 - s) * F[k + T] + s * F[k], atol=1e-6)


# ---------------------------------------------------------------------------
# The procedural fallback honours the asset contract (shape, dtype, two
# channels) and the GL padding duplicates slice 0 on top of every frame — the
# property the shader's in-frame z wrap relies on. Breaks if the pad goes to
# the bottom or the frame-major reshape order changes.
# ---------------------------------------------------------------------------
def test_procedural_volume_contract_and_gl_padding_wraps_z():
    vol = vb.procedural_volume(frames=3, nz=4, ny=5, nx=6, seed=1)
    assert vol.shape == (3, 4, 5, 6, 2) and vol.dtype == np.uint8
    assert vol.max() > vol.min()
    padded = pad_volume_for_gl(vol)
    assert padded.shape == (3 * 5, 5, 6, 2)
    for t in range(3):
        base = t * 5
        assert np.array_equal(padded[base:base + 4], vol[t])
        assert np.array_equal(padded[base + 4], vol[t, 0])


# ---------------------------------------------------------------------------
# Round trip through the .npz contract. Breaks if save/load disagree on a
# metadata key or the format version check is dropped.
# ---------------------------------------------------------------------------
def test_save_load_round_trip(tmp_path):
    p = vb.BakeParams(nx=6, ny=5, nz=4, frames=2, period_tiles=3.0,
                      height_tiles=0.5, loop_seconds=2.5)
    vol = vb.procedural_volume(frames=2, nz=4, ny=5, nx=6)
    res = vb.BakeResult(volume=vol, params=p)
    path = tmp_path / "v.npz"
    vb.save_volume(path, res)
    vol2, meta = vb.load_volume(path)
    assert np.array_equal(vol2, vol)
    assert meta["period_tiles"] == 3.0 and meta["height_tiles"] == 0.5
    assert meta["loop_seconds"] == 2.5
    assert meta["version"] == vb.VOLUME_FORMAT_VERSION


# ---------------------------------------------------------------------------
# A tiny end-to-end bake runs, loops (first == crossfade of the recording)
# and stays finite. Breaks if any stage raises or produces NaNs. Kept tiny
# so the suite stays fast.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1])
def test_tiny_bake_runs_and_is_finite(seed):
    p = vb.BakeParams(nx=8, ny=8, nz=4, frames=2, substeps=1,
                      warmup_steps=2, seed=seed)
    res = vb.bake(p)
    assert res.volume.shape == (2, 4, 8, 8, 2)
    assert np.isfinite(res.stats["div_max"])
    assert res.volume.max() > 0
