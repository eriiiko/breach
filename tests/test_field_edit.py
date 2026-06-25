"""Tests for the FieldEdit write primitive (engine/13).

Covers, per the engine/13 spec:

* each EditMode (ADD / REMOVE / MAX) on a float field;
* each Region (TILE / DISC / BEAM / RECT) — tile coverage + shape;
* LINEAR falloff (weight = 1 - dist/radius) and post-combine clamp;
* per-field skip-mask (a smoke edit never writes a solid tile; a fire edit
  never writes a non-flammable tile);
* the heat Q16.16 branch (quantize + saturating add, clamps at INT32_MAX);
* DETERMINISM: two overlapping edits applied through the queue give the same
  result regardless of enqueue order (the stable sort);
* the noise draw is deterministic for a fixed seed (one RNG consumer at flush);
* before/after equivalence of the migrated apply_explosion / add_explosion_smoke
  (the deposits land identically to a hand-rolled reference).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_field_edit.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import numpy as np

from simulation.field_edit import (
    FieldEdit, EditMode, Region, Falloff, EditQueue,
    apply_field_edit, heat_quantize, heat_saturating_add, HEAT_SCALE,
)
from simulation import wave_fixed   # S2a: wave_source Q16.16 helpers


# ---------------------------------------------------------------------------
# Minimal gmap stub: the field arrays the policy touches + the three masks the
# skip-policies read. apply_field_edit reads getattr(gmap, field) and the
# skip-mask callables read gmap.solid / gmap.is_vacuum / gmap.flammable.
# ---------------------------------------------------------------------------
class _GMapStub:
    def __init__(self, h=12, w=12):
        self.smoke = np.zeros((h, w), dtype=np.float32)
        self.atmosphere = np.zeros((h, w), dtype=np.float32)
        # S2a: wave_source is int32 Q16.16 (the "wave" field-edit dtype combines
        # in real units, stores quantized) — mirror the real gmap dtype here.
        self.wave_source = np.zeros((h, w), dtype=np.int32)
        self.fire = np.zeros((h, w), dtype=np.float32)
        self.heat = np.zeros((h, w), dtype=np.int32)
        # An (h, w, 3) channel field to exercise the channel path.
        self.light_rgb = np.zeros((h, w, 3), dtype=np.float32)
        self.solid = np.zeros((h, w), dtype=bool)
        self.is_vacuum = np.zeros((h, w), dtype=bool)
        self.flammable = np.ones((h, w), dtype=bool)


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def test_mode_add():
    g = _GMapStub()
    g.atmosphere[5, 5] = 1.0
    apply_field_edit(g, FieldEdit("atmosphere", Region.TILE, (5, 5), 0.25,
                                  EditMode.ADD), _rng())
    assert abs(float(g.atmosphere[5, 5]) - 1.25) < 1e-6


def test_mode_remove_clamps_to_floor():
    g = _GMapStub()
    g.smoke[5, 5] = 0.3
    # REMOVE a large amount -> clamped to the smoke policy floor (0).
    apply_field_edit(g, FieldEdit("smoke", Region.TILE, (5, 5), 10.0,
                                  EditMode.REMOVE, clamp=(0.0, 1.0)), _rng())
    assert float(g.smoke[5, 5]) == 0.0


def test_mode_max_never_lowers():
    g = _GMapStub()
    g.fire[5, 5] = 0.7
    # MAX with a smaller value leaves it; with a larger value raises it.
    apply_field_edit(g, FieldEdit("fire", Region.TILE, (5, 5), 0.4,
                                  EditMode.MAX, clamp=(0.0, 1.0)), _rng())
    assert abs(float(g.fire[5, 5]) - 0.7) < 1e-6
    apply_field_edit(g, FieldEdit("fire", Region.TILE, (5, 5), 0.9,
                                  EditMode.MAX, clamp=(0.0, 1.0)), _rng())
    assert abs(float(g.fire[5, 5]) - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
def test_region_tile():
    g = _GMapStub()
    apply_field_edit(g, FieldEdit("atmosphere", Region.TILE, (3, 4), 1.0,
                                  EditMode.ADD), _rng())
    assert float(g.atmosphere[3, 4]) == 1.0
    assert float(g.atmosphere.sum()) == 1.0  # only that one tile


def test_region_disc_flat():
    g = _GMapStub()
    apply_field_edit(g, FieldEdit("atmosphere", Region.DISC, (6, 6, 3.0), 1.0,
                                  EditMode.ADD, Falloff.FLAT), _rng())
    # Centre is filled; a tile just outside the radius is not.
    assert float(g.atmosphere[6, 6]) == 1.0
    assert float(g.atmosphere[6, 6 + 3]) == 0.0   # dist == 3 == radius -> excluded (strict <)
    assert float(g.atmosphere[6, 6 + 2]) == 1.0   # dist 2 < 3 -> filled, FLAT weight 1


def test_region_rect():
    g = _GMapStub()
    apply_field_edit(g, FieldEdit("atmosphere", Region.RECT, (2, 3, 4, 6), 1.0,
                                  EditMode.ADD), _rng())
    # The inclusive box [2..4] x [3..6] = 3 rows x 4 cols = 12 tiles.
    assert float(g.atmosphere.sum()) == 12.0
    assert float(g.atmosphere[2, 3]) == 1.0
    assert float(g.atmosphere[4, 6]) == 1.0
    assert float(g.atmosphere[1, 3]) == 0.0  # outside


def test_region_beam():
    g = _GMapStub()
    # A horizontal beam from (5,1) to (5,9), width 0 -> the single centre row.
    apply_field_edit(g, FieldEdit("atmosphere", Region.BEAM, (5, 1, 5, 9, 0.0),
                                  1.0, EditMode.ADD), _rng())
    row = g.atmosphere[5, 1:10]
    assert np.all(row == 1.0), f"beam centre row not fully covered: {row}"
    # Nothing off the beam line.
    assert float(g.atmosphere[4, 5]) == 0.0
    assert float(g.atmosphere[6, 5]) == 0.0


def test_region_beam_width():
    g = _GMapStub()
    # Width 1 -> the centre row plus the adjacent rows (perp dist <= 1).
    apply_field_edit(g, FieldEdit("atmosphere", Region.BEAM, (5, 2, 5, 8, 1.0),
                                  1.0, EditMode.ADD, Falloff.FLAT), _rng())
    assert float(g.atmosphere[5, 5]) == 1.0
    assert float(g.atmosphere[4, 5]) == 1.0   # 1 tile off-axis, within width
    assert float(g.atmosphere[6, 5]) == 1.0


# ---------------------------------------------------------------------------
# LINEAR falloff + clamp
# ---------------------------------------------------------------------------
def test_linear_falloff_weights():
    g = _GMapStub()
    cr, cc, radius = 6, 6, 4.0
    apply_field_edit(g, FieldEdit("atmosphere", Region.DISC, (cr, cc, radius),
                                  2.0, EditMode.ADD, Falloff.LINEAR), _rng())
    # weight = 1 - dist/radius; amount = 2.0.
    assert abs(float(g.atmosphere[cr, cc]) - 2.0) < 1e-5         # dist 0 -> w 1
    d2 = float(g.atmosphere[cr, cc + 2])                         # dist 2 -> w 0.5
    assert abs(d2 - 2.0 * (1.0 - 2.0 / radius)) < 1e-5


def test_clamp_ceiling():
    g = _GMapStub()
    g.smoke[5, 5] = 0.8
    apply_field_edit(g, FieldEdit("smoke", Region.TILE, (5, 5), 0.9,
                                  EditMode.ADD, clamp=(0.0, 1.0)), _rng())
    assert abs(float(g.smoke[5, 5]) - 1.0) < 1e-6  # 0.8 + 0.9 -> clamped to 1


def test_policy_default_clamp_applied():
    # A smoke edit with NO explicit clamp still gets the policy [0,1] ceiling.
    g = _GMapStub()
    g.smoke[5, 5] = 0.8
    apply_field_edit(g, FieldEdit("smoke", Region.TILE, (5, 5), 5.0,
                                  EditMode.ADD), _rng())
    assert abs(float(g.smoke[5, 5]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Per-field skip-mask
# ---------------------------------------------------------------------------
def test_smoke_edit_skips_solid():
    g = _GMapStub()
    g.solid[6, 6] = True  # centre tile is a wall
    apply_field_edit(g, FieldEdit("smoke", Region.DISC, (6, 6, 3.0), 1.0,
                                  EditMode.ADD, Falloff.FLAT, clamp=(0.0, 1.0)),
                     _rng())
    assert float(g.smoke[6, 6]) == 0.0, "smoke wrote a solid tile"
    assert float(g.smoke[6, 5]) == 1.0, "smoke did not write the open neighbour"


def test_fire_edit_skips_non_flammable():
    g = _GMapStub()
    g.flammable[:] = True
    g.flammable[6, 6] = False  # centre is non-flammable
    apply_field_edit(g, FieldEdit("fire", Region.DISC, (6, 6, 3.0), 1.0,
                                  EditMode.MAX, Falloff.FLAT, clamp=(0.0, 1.0)),
                     _rng())
    assert float(g.fire[6, 6]) == 0.0, "fire wrote a non-flammable tile"
    assert float(g.fire[6, 5]) == 1.0, "fire did not ignite the flammable neighbour"


def test_wave_source_skips_vacuum_and_solid():
    g = _GMapStub()
    g.is_vacuum[6, 7] = True
    g.solid[6, 5] = True
    apply_field_edit(g, FieldEdit("wave_source", Region.DISC, (6, 6, 3.0), 1.0,
                                  EditMode.ADD, Falloff.FLAT), _rng())
    # S2a: wave_source is Q16.16 int32 — dequantize for the value assertions.
    assert int(g.wave_source[6, 7]) == 0, "wave_source wrote a vacuum tile"
    assert int(g.wave_source[6, 5]) == 0, "wave_source wrote a solid tile"
    assert float(g.wave_source[6, 6]) / wave_fixed.FP_ONE_F == 1.0


# ---------------------------------------------------------------------------
# heat Q16.16 branch
# ---------------------------------------------------------------------------
def test_heat_quantize_round_and_saturate():
    assert heat_quantize(0.0) == 0
    assert heat_quantize(-5.0) == 0
    assert heat_quantize(1.0) == HEAT_SCALE
    # round-to-nearest
    assert heat_quantize(0.5) == HEAT_SCALE // 2
    # saturate
    assert heat_quantize(1e9) == np.iinfo(np.int32).max


def test_heat_saturating_add():
    imax = int(np.iinfo(np.int32).max)
    assert heat_saturating_add(10, 5) == 15
    assert heat_saturating_add(10, -5) == 10        # non-positive delta is a no-op
    assert heat_saturating_add(imax - 1, 100) == imax  # clamps, never wraps


def test_heat_field_edit_uses_fixed_point():
    g = _GMapStub()
    apply_field_edit(g, FieldEdit("heat", Region.TILE, (5, 5), 1.0,
                                  EditMode.ADD), _rng())
    assert int(g.heat[5, 5]) == HEAT_SCALE  # 1.0 energy -> HEAT_SCALE raw counts


def test_heat_field_edit_saturates_on_overlap():
    g = _GMapStub()
    g.heat[5, 5] = np.iinfo(np.int32).max - 10
    apply_field_edit(g, FieldEdit("heat", Region.TILE, (5, 5), 1.0,
                                  EditMode.ADD), _rng())
    assert int(g.heat[5, 5]) == int(np.iinfo(np.int32).max)  # saturates, no wrap


# ---------------------------------------------------------------------------
# Channel path (RGB field)
# ---------------------------------------------------------------------------
def test_channel_write():
    g = _GMapStub()
    # No policy entry for light_rgb -> would raise; use a registered float field
    # instead is not possible (all float policies are scalar). Register-free path:
    # the channel path is exercised by writing a known (h,w,3) field that HAS a
    # policy. None of the shipped policies are (h,w,3); so we assert the channel
    # arg is honoured on a manually-policied field via monkeypatch-free route:
    # skip if no (h,w,3) policy exists. (Kept as a guard for when one is added.)
    from simulation.field_edit import FIELD_POLICY, _FieldPolicy
    FIELD_POLICY["light_rgb"] = _FieldPolicy("float", None, None)
    try:
        apply_field_edit(g, FieldEdit("light_rgb", Region.TILE, (5, 5), 0.5,
                                      EditMode.ADD, channel=1), _rng())
        assert abs(float(g.light_rgb[5, 5, 1]) - 0.5) < 1e-6
        assert float(g.light_rgb[5, 5, 0]) == 0.0
        assert float(g.light_rgb[5, 5, 2]) == 0.0
    finally:
        del FIELD_POLICY["light_rgb"]


# ---------------------------------------------------------------------------
# Determinism — order independence via the stable sort
# ---------------------------------------------------------------------------
def test_queue_order_independence_max():
    # Two MAX edits overlapping a tile -> result independent of enqueue order.
    a = FieldEdit("fire", Region.TILE, (5, 5), 0.3, EditMode.MAX,
                  clamp=(0.0, 1.0), source_id=1)
    b = FieldEdit("fire", Region.TILE, (5, 5), 0.8, EditMode.MAX,
                  clamp=(0.0, 1.0), source_id=2)

    g1 = _GMapStub()
    q1 = EditQueue(); q1.enqueue(a); q1.enqueue(b); q1.flush(g1, _rng())

    g2 = _GMapStub()
    q2 = EditQueue(); q2.enqueue(b); q2.enqueue(a); q2.flush(g2, _rng())

    assert np.array_equal(g1.fire, g2.fire), "MAX result depended on enqueue order"
    assert abs(float(g1.fire[5, 5]) - 0.8) < 1e-6


def test_queue_order_independence_clamped_add():
    # Two clamped ADDs overlapping: clamp(clamp(s+a)+b) is order-sensitive in
    # general; the stable sort fixes the applied order so both orders agree.
    a = FieldEdit("smoke", Region.DISC, (5, 5, 3.0), 0.7, EditMode.ADD,
                  Falloff.FLAT, clamp=(0.0, 1.0), source_id=10)
    b = FieldEdit("smoke", Region.DISC, (6, 6, 3.0), 0.7, EditMode.ADD,
                  Falloff.FLAT, clamp=(0.0, 1.0), source_id=20)

    g1 = _GMapStub()
    q1 = EditQueue(); q1.enqueue(a); q1.enqueue(b); q1.flush(g1, _rng())

    g2 = _GMapStub()
    q2 = EditQueue(); q2.enqueue(b); q2.enqueue(a); q2.flush(g2, _rng())

    assert np.array_equal(g1.smoke, g2.smoke), \
        "clamped-ADD overlap depended on enqueue order"


def test_noise_draw_deterministic_for_fixed_seed():
    e = FieldEdit("smoke", Region.DISC, (6, 6, 4.0), 0.8, EditMode.ADD,
                  Falloff.LINEAR, clamp=(0.0, 1.0), noise=0.85, source_id=1)

    g1 = _GMapStub(); q1 = EditQueue(); q1.enqueue(e); q1.flush(g1, _rng(seed=7))
    g2 = _GMapStub(); q2 = EditQueue(); q2.enqueue(e); q2.flush(g2, _rng(seed=7))
    assert np.array_equal(g1.smoke, g2.smoke), "noise deposit not deterministic"

    # A different seed gives a different deposit (the noise is actually random).
    g3 = _GMapStub(); q3 = EditQueue(); q3.enqueue(e); q3.flush(g3, _rng(seed=99))
    assert not np.array_equal(g1.smoke, g3.smoke), "noise ignored the seed"


class _CountingRng:
    """Wraps a Generator, counting uniform() calls — to prove the skip-mask
    veto runs BEFORE the RNG draw (a skipped tile consumes no random value)."""
    def __init__(self, seed):
        self._g = np.random.default_rng(seed)
        self.draws = 0

    def uniform(self, lo, hi):
        self.draws += 1
        return self._g.uniform(lo, hi)


def test_skipped_tile_draws_no_noise():
    # A noisy smoke disc: the number of RNG draws must equal the number of
    # SURVIVING (non-skipped) tiles, not the full region tile count. So masking
    # one interior tile solid must drop the draw count by exactly one — proving
    # the skip-mask veto precedes the draw (skipped tiles consume no randomness).
    e = FieldEdit("smoke", Region.DISC, (6, 6, 4.0), 0.8, EditMode.ADD,
                  Falloff.LINEAR, clamp=(0.0, 1.0), noise=0.85, source_id=1)

    g_clean = _GMapStub()
    rng_clean = _CountingRng(seed=3)
    apply_field_edit(g_clean, e, rng_clean)
    n_survivors = int((g_clean.smoke > 0).sum())
    # Every surviving tile drew exactly once (LINEAR weight > 0 inside the disc).
    assert rng_clean.draws == n_survivors, \
        f"draws {rng_clean.draws} != surviving tiles {n_survivors}"

    g_solid = _GMapStub()
    g_solid.solid[5, 5] = True  # one interior disc tile masked out
    rng_solid = _CountingRng(seed=3)
    apply_field_edit(g_solid, e, rng_solid)
    assert float(g_solid.smoke[5, 5]) == 0.0
    # Exactly one fewer draw — the skipped tile consumed no RNG value.
    assert rng_solid.draws == rng_clean.draws - 1, \
        f"masking a solid tile changed draw count by {rng_clean.draws - rng_solid.draws}, expected 1"


# ---------------------------------------------------------------------------
# Migration equivalence — apply_explosion / add_explosion_smoke deposits land
# identically to a hand-rolled reference of the legacy inline behaviour.
# ---------------------------------------------------------------------------
def _open_gmap(h=24, w=24):
    """A real GameMap with a fully-interior-air room (border hull)."""
    from level_loader import LevelData
    from simulation.gamemap import GameMap
    tm = np.ones((h, w), dtype=np.int32)  # border hull
    tm[1:h - 1, 1:w - 1] = 4              # interior air
    level = LevelData(name="fe_test", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))
    return GameMap(level)


def test_add_explosion_smoke_equivalence():
    """The migrated add_explosion_smoke deposit == the legacy inline formula."""
    from simulation.physics import add_explosion_smoke
    import math

    radius, noise, seed = 8, 0.85, 11
    cy, cx = 12, 12

    # Reference: the exact legacy inline loop (row-major draw order).
    ref = _open_gmap()
    ref.smoke[:] = 0.0
    rng_ref = np.random.default_rng(seed)
    low = 1.0 - noise
    h, w = ref.material.shape
    for ddy in range(-radius, radius + 1):
        for ddx in range(-radius, radius + 1):
            ny, nx = cy + ddy, cx + ddx
            if 0 <= ny < h and 0 <= nx < w and not ref.solid[ny, nx]:
                dist = math.sqrt(ddy ** 2 + ddx ** 2)
                if dist < radius:
                    base = 0.8 * (1 - dist / radius)
                    mult = float(rng_ref.uniform(low, 1.0))
                    ref.smoke[ny, nx] = min(1.0, ref.smoke[ny, nx] + base * mult)

    # Migrated path through the queue, same seed.
    got = _open_gmap()
    got.smoke[:] = 0.0
    q = EditQueue()
    add_explosion_smoke(got, q, cy, cx, radius, noise=noise)
    q.flush(got, np.random.default_rng(seed))

    assert np.allclose(ref.smoke, got.smoke, atol=1e-6), \
        "migrated add_explosion_smoke diverged from the legacy deposit"


def test_apply_explosion_atmosphere_equivalence():
    """The migrated atmosphere deposit == the legacy ``+= pressure*falloff``."""
    from simulation.physics import apply_explosion
    import math

    radius, pressure, wall_damage = 6, 3.0, 0.0  # 0 wall damage: no topology edit
    cy, cx = 12, 12

    ref = _open_gmap()
    h, w = ref.material.shape
    atm0 = ref.atmosphere.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                dist = math.sqrt(dy * dy + dx * dx)
                if dist <= radius and not ref.solid[ny, nx] and not ref.is_vacuum[ny, nx]:
                    ref.atmosphere[ny, nx] += pressure * (1.0 - dist / radius)

    got = _open_gmap()
    q = EditQueue()
    apply_explosion(got, q, cy, cx, radius, pressure, wall_damage)
    q.flush(got, np.random.default_rng(0))

    assert np.allclose(ref.atmosphere, got.atmosphere, atol=1e-5), \
        "migrated apply_explosion atmosphere diverged from the legacy deposit"
    # Sanity: the atmosphere actually changed somewhere.
    assert not np.allclose(got.atmosphere, atm0)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
