"""Property gates for the smoke phase field (issue #90,
renderer/volume_phase.py) — pyray-free.

Each test names the property it protects and the change that must break it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from renderer import volume_phase as vp


def test_encode_decode_round_trip_across_the_wrap():
    """PROPERTY: the (cos, sin) encoding decodes to the phase mod 1 for any
    offset, including offsets far outside [0, 1) and either side of the wrap.
    BREAKS IF: the encoding stops being a unit vector of 2*pi*phase, or the
    decoder's division constant drifts from the shader's 1/(2 pi)."""
    off = np.array([[0.0, 0.25, 0.98], [1.02, -0.3, 7.5]], dtype=np.float32)
    buf = np.zeros(off.shape + (4,), dtype=np.float16)
    vp.encode_phase_rg(off, buf)
    got = vp.decode_phase_rg(buf[..., :2].astype(np.float32))
    assert np.allclose(got, np.mod(off, 1.0), atol=2e-3)
    assert np.all(buf[..., 3] == 1.0)


def test_unit_vector_interpolation_crosses_the_wrap_on_the_short_arc():
    """PROPERTY: the mean of the encodings of 0.98 and 0.02 decodes to ~0.00,
    not 0.50 — the reason the field goes up as cos/sin and not as a scalar
    (the GPU interpolates linearly between tile centres).
    BREAKS IF: someone 'simplifies' the texture to a scalar phase channel."""
    off = np.array([[0.98, 0.02]], dtype=np.float32)
    buf = np.zeros((1, 2, 4), dtype=np.float16)
    vp.encode_phase_rg(off, buf)
    mid = buf[0, :, :2].astype(np.float64).mean(axis=0)
    ph = vp.decode_phase_rg(mid[None, :])[0]
    assert min(ph, 1.0 - ph) < 1e-2


def test_value_noise_is_seeded_bounded_and_smooth():
    """PROPERTY: the static phase noise is deterministic for a seed (a replay
    renders the same field), bounded in [-1, 1], and smooth: the step between
    neighbouring tiles is small against the wavelength (no creases that would
    read as seams in the movie).
    BREAKS IF: the interpolation loses its cosine ease or the lattice spacing
    stops following `wavelength_tiles`."""
    a = vp.value_noise_tiles(40, 60, 6.0, seed=3)
    b = vp.value_noise_tiles(40, 60, 6.0, seed=3)
    c = vp.value_noise_tiles(40, 60, 6.0, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.min() >= -1.0 and a.max() <= 1.0
    step = max(np.abs(np.diff(a, axis=0)).max(), np.abs(np.diff(a, axis=1)).max())
    # a full swing of 2 over half a wavelength (3 tiles) is < 0.7/tile; the
    # cosine ease keeps the per-tile step well under that bound
    assert step < 0.7


def test_offset_mode_is_bounded_and_frozen_while_paused():
    """PROPERTY: in `offset` mode the phase offset is a pure function of the
    sources (no state): the same inputs give the same field at any dt, and its
    magnitude is bounded by the sum of amplitudes. A paused sim (dt 0) changes
    nothing.
    BREAKS IF: offset mode starts integrating, or the sources stop being
    normalised to [-1, 1]."""
    f = vp.PhaseField(8, 8, seed=1)
    temp = np.full((8, 8), 1e6, dtype=np.float32)      # a furnace: saturates
    src = vp.normalise_sources(temperature=temp, temp_ref=200.0)
    assert np.all(np.abs(src["temp"]) <= 1.0)
    kw = dict(mode="offset", sources=src, amps={"temp": 0.3}, noise_amp=0.5,
              noise_tiles=4.0, relax_loops=2.0)
    o1 = f.step(dt_loops=0.0, **kw).copy()
    o2 = f.step(dt_loops=5.0, **kw).copy()
    assert np.array_equal(o1, o2)
    assert np.all(np.abs(o1) <= 0.8 + 1e-6)


def test_speed_mode_drift_is_bounded_by_the_leak_and_stops_when_paused():
    """PROPERTY: in `speed` mode a constant rate deviation `a` integrates to a
    drift that saturates at ~a * relax_loops instead of growing without bound
    (the temporal shear between a hot tile and its quiet neighbour stays
    finite), and a frozen sim (dt 0) does not advance it.
    BREAKS IF: the leak is removed or applied before the increment in a way
    that changes the fixed point by more than the discretisation allows."""
    f = vp.PhaseField(2, 2, seed=1)
    src = {"temp": np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)}
    kw = dict(mode="speed", sources=src, amps={"temp": 0.5}, noise_amp=0.0,
              noise_tiles=4.0, relax_loops=2.0)
    for _ in range(400):
        off = f.step(dt_loops=0.05, **kw)
    hot = float(off[0, 0])
    assert 0.8 < hot < 1.05                 # a * relax = 1.0 fixed point
    assert off[0, 1] == 0.0 and off[1, 1] == 0.0
    frozen = f.step(dt_loops=0.0, **kw).copy()
    assert np.array_equal(frozen, off)


def test_off_mode_is_identically_zero_and_unknown_modes_fail_loudly():
    """PROPERTY: `off` reproduces the single global clock exactly (zero
    offset everywhere) and a misspelt mode raises rather than silently
    picking one.
    BREAKS IF: `off` keeps the static noise, or the mode check is dropped."""
    f = vp.PhaseField(3, 3)
    off = f.step(mode="off", dt_loops=0.1, sources={}, amps={}, noise_amp=1.0,
                 noise_tiles=4.0, relax_loops=1.0)
    assert not np.any(off)
    with pytest.raises(ValueError):
        f.step(mode="offsett", dt_loops=0.1, sources={}, amps={}, noise_amp=1.0,
               noise_tiles=4.0, relax_loops=1.0)
