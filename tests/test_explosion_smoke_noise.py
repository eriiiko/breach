"""Explosion-smoke noise dial (smoke v2 — explosion-smoke-noise knob).

Canon: ch.05 §4 "explosion smoke noise too subtle". The per-tile multiplier
applied by ``simulation.physics.add_explosion_smoke`` is drawn uniform in
``[1 - noise, 1.0]``. ``noise`` is the contrast knob:

- ``noise = 0``   -> multiplier == 1 everywhere: a FLAT blob, zero texture.
- ``noise = 0.6`` -> ``[0.4, 1.0]``: the old (too-subtle) shipped look.
- ``noise = 0.85``-> ``[0.15, 1.0]``: config default, ragged holes.
- ``noise = 1.0`` -> ``[0.0, 1.0]``: maximal contrast.

These tests assert, deterministically with a fixed seed, that a HIGHER noise
value produces visibly more per-tile texture in the deposited cloud (higher
variance and a lower floor among the deposited tiles) than a LOW value, and
that ``noise = 0`` is perfectly flat. The deposit must also stay deterministic
for a given seed.

Run:
    C:/Users/steen/anaconda3/python.exe tests/test_explosion_smoke_noise.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import numpy as np

import breach_physics as bp
from level_loader import LevelData
from simulation import Simulation
from simulation.physics import add_explosion_smoke
from simulation.field_edit import EditQueue

SEED = 11


def _open_level(h: int = 24, w: int = 24):
    """A fully-interior-air map (border hull) so the deposit disc lands on air.

    CSV codes: 1 = hull, 4 = interior air.
    """
    tm = np.ones((h, w), dtype=np.int32)   # border hull
    tm[1:h - 1, 1:w - 1] = 4               # interior air
    return LevelData(
        name="open_room_test",
        version="test",
        path=Path("."),
        tilemap=tm,
        tile_size_m=1.0,
        diffuse_path=Path("."),
    )


def _make_sim():
    sim = Simulation(_open_level(), seed=SEED, breach_physics=bp,
                     enable_recorder=False)
    return sim


def _deposit(noise: float, radius: int = 8):
    """Deposit one explosion-smoke disc at the map centre with the given noise,
    from a fresh RNG seeded identically. Returns ``(smoke, mult)`` where
    ``smoke`` is the full deposited field and ``mult`` is the per-tile NOISE
    multiplier (``deposit / base``) on the deposited tiles only — isolating the
    noise from the deterministic radial falloff ``base = 0.8*(1 - dist/r)`` so
    the texture can be measured directly.
    """
    sim = _make_sim()
    g = sim.gmap
    g.smoke[:] = 0.0
    rng = np.random.default_rng(SEED)
    h, w = g.material.shape
    cy, cx = h // 2, w // 2
    # engine/13: the deposit is now an enqueued FieldEdit; the per-tile noise
    # multiplier is drawn from `rng` at flush time, in the queue's deterministic
    # tile order — so a fixed seed gives the same deposit as the old inline draw.
    queue = EditQueue()
    add_explosion_smoke(g, queue, cy, cx, radius, noise=noise)
    queue.flush(g, rng)
    smoke = g.smoke.copy()

    # Reconstruct base per tile to recover the noise multiplier deposit/base.
    yy, xx = np.indices((h, w))
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    base = np.where(dist < radius, 0.8 * (1.0 - dist / radius), 0.0)
    dep_mask = smoke > 0.0
    mult = smoke[dep_mask] / base[dep_mask]
    return smoke, mult


# --------------------------------------------------------------------------
# 1. Higher noise -> more per-tile texture (variance + lower floor)
# --------------------------------------------------------------------------
def test_higher_noise_more_texture():
    low_smoke, low_mult = _deposit(noise=0.2)
    high_smoke, high_mult = _deposit(noise=0.95)

    assert low_mult.size > 0 and high_mult.size > 0
    # The disc footprint is identical for both noise values — noise only scales
    # the per-tile base, it never changes which tiles receive smoke.
    assert np.array_equal(low_smoke > 0.0, high_smoke > 0.0), (
        "noise changed which tiles received smoke (it must only scale them)")

    # Per-tile texture lives in the NOISE multiplier (deposit/base). Its spread
    # must rise with noise: low noise -> [0.8, 1.0], high -> [0.05, 1.0].
    low_var = float(np.var(low_mult))
    high_var = float(np.var(high_mult))
    assert high_var > low_var, (
        f"higher noise did not raise multiplier variance: "
        f"low={low_var:.5f} high={high_var:.5f}")

    # Lower floor: high noise drives some tiles much closer to empty (the
    # ragged holes / missing patches) than low noise.
    assert high_mult.min() < low_mult.min(), (
        f"higher noise did not lower the floor: "
        f"low_min={low_mult.min():.5f} high_min={high_mult.min():.5f}")
    # Sanity on the configured range: low-noise floor ~ 1-0.2 = 0.8, high ~ 0.05.
    assert low_mult.min() > 0.75, f"low-noise floor too low: {low_mult.min():.3f}"
    assert high_mult.min() < 0.30, f"high-noise floor too high: {high_mult.min():.3f}"

    print(f"OK: higher_noise_more_texture "
          f"(mult var {low_var:.4f}->{high_var:.4f}, "
          f"min {low_mult.min():.4f}->{high_mult.min():.4f})")


# --------------------------------------------------------------------------
# 2. noise = 0 is a perfectly flat deposit (no texture)
# --------------------------------------------------------------------------
def test_zero_noise_is_flat():
    _flat_smoke, flat_mult = _deposit(noise=0.0)
    assert flat_mult.size > 0
    # With noise=0 the multiplier is exactly 1 on every deposited tile — a flat
    # blob with zero per-tile texture.
    assert np.allclose(flat_mult, 1.0, atol=1e-6), (
        f"noise=0 multiplier not flat: var={np.var(flat_mult):.2e}")
    # And it is strictly flatter than a noisy deposit.
    _noisy_smoke, noisy_mult = _deposit(noise=0.85)
    assert float(np.var(flat_mult)) < float(np.var(noisy_mult))
    print(f"OK: zero_noise_is_flat "
          f"(mult var {np.var(flat_mult):.2e}, all == 1.0)")


# --------------------------------------------------------------------------
# 3. Determinism — same seed + noise -> identical deposit
# --------------------------------------------------------------------------
def test_deposit_is_deterministic():
    a, _ = _deposit(noise=0.85)
    b, _ = _deposit(noise=0.85)
    assert np.array_equal(a, b), "deposit not deterministic for a fixed seed"
    print("OK: deposit_is_deterministic")


if __name__ == "__main__":
    test_higher_noise_more_texture()
    test_zero_noise_is_flat()
    test_deposit_is_deterministic()
    print("\nAll explosion-smoke noise tests passed.")
