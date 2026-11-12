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
import pytest

from simulation.field_edit import (
    FieldEdit, EditMode, Region, Falloff, EditQueue,
    apply_field_edit, heat_quantize, heat_saturating_add, HEAT_SCALE,
)
from simulation import wave_fixed   # S2a: wave_source Q16.16 helpers
from simulation import gas_fixed     # S2b: smoke/gas Q16.16 helpers
from simulation import atmosphere_fixed  # S2c: atmosphere Q16.16 helpers
from simulation import fire_fixed    # S3a: fire Q16.16 helpers


# ---------------------------------------------------------------------------
# Minimal gmap stub: the field arrays the policy touches + the three masks the
# skip-policies read. apply_field_edit reads getattr(gmap, field) and the
# skip-mask callables read gmap.solid / gmap.is_vacuum / gmap.flammable.
# ---------------------------------------------------------------------------
# S2b: smoke is int32 Q16.16 — the "gas" field-edit dtype combines in real
# density, stores quantized. Helpers to author/read smoke in density units.
def _smoke_trace_q(v):
    return gas_fixed.quantize_trace_scalar(v)      # the TRACE currency (#88)


def _smoke_d(q):
    return float(gas_fixed.dequantize_trace(q))


# S2c: atmosphere is int32 Q16.16 — the "atmosphere" field-edit dtype combines in
# real pressure, stores quantized. Helpers to author/read atmosphere in real units.
def _atm_q(v):
    return atmosphere_fixed.quantize_scalar(v)


def _atm_d(q):
    return float(q) / atmosphere_fixed.FP_ONE_F


# S3a: fire is int32 Q16.16 — the "fire" field-edit dtype combines in real
# intensity, stores quantized. Helpers to author/read fire in real [0,1] units.
def _fire_q(v):
    return fire_fixed.quantize_scalar(v)


def _fire_d(q):
    return float(q) / fire_fixed.FP_ONE_F


class _GasesStub:
    """Minimal `gmap.gases` stand-in: `apply_field_edit`'s "atmosphere" policy
    (EOS refactor P3) resolves `gmap.gas[gmap.gases.name_to_id[...]]` — the
    real GasTable's id map, name-keyed (simulation.gases.O2/INERT_N2)."""
    name_to_id = {"o2": 0, "inert_n2": 1}


class _GMapStub:
    def __init__(self, h=12, w=12):
        # S2b: smoke is int32 Q16.16 (the "gas" field-edit dtype) — mirror the
        # real gmap dtype here.
        self.smoke = np.zeros((h, w), dtype=np.int32)
        # S2c: atmosphere is int32 Q16.16 (the "atmosphere" field-edit dtype) —
        # mirror the real gmap dtype here (use _atm_q to author, _atm_d to read).
        # EOS refactor P3: "atmosphere" edits no longer WRITE this array (P is
        # solver-owned) — it is kept only so any generic-field test path that
        # merely reads gmap.atmosphere for shape/dtype still finds it.
        self.atmosphere = np.zeros((h, w), dtype=np.int32)
        # EOS refactor P3: `gas`/`gases` — the "atmosphere" edit's REAL target
        # (a bulk-N deposit split 21/79 across O2/inert_N2, design §6). Two
        # planes is enough for the stub (only ids 0/1 are ever resolved).
        self.gas = np.zeros((2, h, w), dtype=np.int32)
        self.gases = _GasesStub()
        # S2a: wave_source is int32 Q16.16 (the "wave" field-edit dtype combines
        # in real units, stores quantized) — RETAINED array, but EOS refactor
        # P3's "wave_source" edit now targets `temperature` instead (below).
        self.wave_source = np.zeros((h, w), dtype=np.int32)
        # EOS refactor P3: `temperature` — the "wave_source" edit's REAL target
        # (an energy deposit, design §6). Shares the Q16.16/HEAT_SCALE domain.
        self.temperature = np.zeros((h, w), dtype=np.int32)
        # S3a: fire is int32 Q16.16 (the "fire" field-edit dtype combines in real
        # intensity, stores quantized) — mirror the real gmap dtype here.
        self.fire = np.zeros((h, w), dtype=np.int32)
        self.heat = np.zeros((h, w), dtype=np.int32)
        # An (h, w, 3) channel field to exercise the channel path.
        self.light_rgb = np.zeros((h, w, 3), dtype=np.float32)
        self.solid = np.zeros((h, w), dtype=bool)
        self.is_vacuum = np.zeros((h, w), dtype=bool)
        self.flammable = np.ones((h, w), dtype=bool)


def _rng(seed=0):
    from simulation.draws import MatchDraws   # #29 P3b: the keyed draws
    return MatchDraws(seed)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def test_mode_add():
    # EOS refactor P3: "atmosphere" edits now target a bulk-N deposit (design
    # §6), not this generic ADD-mode mechanic — swapped the test vehicle to
    # "smoke" (an unaffected float-combine field; wide clamp to keep the
    # original unclamped-value assertion honest).
    g = _GMapStub()
    g.smoke[5, 5] = _smoke_trace_q(1.0)
    apply_field_edit(g, FieldEdit("smoke", Region.TILE, (5, 5), 0.25,
                                  EditMode.ADD, clamp=(0.0, 10.0)), _rng())
    assert abs(_smoke_d(g.smoke[5, 5]) - 1.25) < 1e-4


def test_mode_remove_clamps_to_floor():
    g = _GMapStub()
    g.smoke[5, 5] = _smoke_trace_q(0.3)
    # REMOVE a large amount -> clamped to the smoke policy floor (0).
    apply_field_edit(g, FieldEdit("smoke", Region.TILE, (5, 5), 10.0,
                                  EditMode.REMOVE, clamp=(0.0, 1.0)), _rng())
    assert int(g.smoke[5, 5]) == 0


def test_mode_max_never_lowers():
    g = _GMapStub()
    g.fire[5, 5] = _fire_q(0.7)
    # MAX with a smaller value leaves it; with a larger value raises it.
    apply_field_edit(g, FieldEdit("fire", Region.TILE, (5, 5), 0.4,
                                  EditMode.MAX, clamp=(0.0, 1.0)), _rng())
    assert abs(_fire_d(g.fire[5, 5]) - 0.7) < 1e-4
    apply_field_edit(g, FieldEdit("fire", Region.TILE, (5, 5), 0.9,
                                  EditMode.MAX, clamp=(0.0, 1.0)), _rng())
    assert abs(_fire_d(g.fire[5, 5]) - 0.9) < 1e-4


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
def test_region_tile():
    # EOS refactor P3: swapped "atmosphere" -> "smoke" as the region-mechanics
    # test vehicle (see test_mode_add's attribution).
    g = _GMapStub()
    apply_field_edit(g, FieldEdit("smoke", Region.TILE, (3, 4), 1.0,
                                  EditMode.ADD, clamp=(0.0, 10.0)), _rng())
    assert _smoke_d(g.smoke[3, 4]) == 1.0
    assert _smoke_d(g.smoke.astype(np.int64).sum()) == 1.0  # only that one tile


def test_region_disc_flat():
    g = _GMapStub()
    apply_field_edit(g, FieldEdit("smoke", Region.DISC, (6, 6, 3.0), 1.0,
                                  EditMode.ADD, Falloff.FLAT, clamp=(0.0, 10.0)),
                     _rng())
    # Centre is filled; a tile just outside the radius is not.
    assert _smoke_d(g.smoke[6, 6]) == 1.0
    assert _smoke_d(g.smoke[6, 6 + 3]) == 0.0   # dist == 3 == radius -> excluded (strict <)
    assert _smoke_d(g.smoke[6, 6 + 2]) == 1.0   # dist 2 < 3 -> filled, FLAT weight 1


def test_region_rect():
    g = _GMapStub()
    apply_field_edit(g, FieldEdit("smoke", Region.RECT, (2, 3, 4, 6), 1.0,
                                  EditMode.ADD, clamp=(0.0, 10.0)), _rng())
    # The inclusive box [2..4] x [3..6] = 3 rows x 4 cols = 12 tiles.
    assert _smoke_d(g.smoke.astype(np.int64).sum()) == 12.0
    assert _smoke_d(g.smoke[2, 3]) == 1.0
    assert _smoke_d(g.smoke[4, 6]) == 1.0
    assert _smoke_d(g.smoke[1, 3]) == 0.0  # outside


def test_region_beam():
    g = _GMapStub()
    # A horizontal beam from (5,1) to (5,9), width 0 -> the single centre row.
    apply_field_edit(g, FieldEdit("smoke", Region.BEAM, (5, 1, 5, 9, 0.0),
                                  1.0, EditMode.ADD, clamp=(0.0, 10.0)), _rng())
    row = g.smoke[5, 1:10]
    assert np.all(row == _smoke_trace_q(1.0)), f"beam centre row not fully covered: {row}"
    # Nothing off the beam line.
    assert _smoke_d(g.smoke[4, 5]) == 0.0
    assert _smoke_d(g.smoke[6, 5]) == 0.0


def test_region_beam_width():
    g = _GMapStub()
    # Width 1 -> the centre row plus the adjacent rows (perp dist <= 1).
    apply_field_edit(g, FieldEdit("smoke", Region.BEAM, (5, 2, 5, 8, 1.0),
                                  1.0, EditMode.ADD, Falloff.FLAT, clamp=(0.0, 10.0)),
                     _rng())
    assert _smoke_d(g.smoke[5, 5]) == 1.0
    assert _smoke_d(g.smoke[4, 5]) == 1.0   # 1 tile off-axis, within width
    assert _smoke_d(g.smoke[6, 5]) == 1.0


# ---------------------------------------------------------------------------
# LINEAR falloff + clamp
# ---------------------------------------------------------------------------
def test_linear_falloff_weights():
    g = _GMapStub()
    cr, cc, radius = 6, 6, 4.0
    apply_field_edit(g, FieldEdit("smoke", Region.DISC, (cr, cc, radius),
                                  2.0, EditMode.ADD, Falloff.LINEAR,
                                  clamp=(0.0, 10.0)), _rng())
    # weight = 1 - dist/radius; amount = 2.0.
    assert abs(_smoke_d(g.smoke[cr, cc]) - 2.0) < 1e-4         # dist 0 -> w 1
    d2 = _smoke_d(g.smoke[cr, cc + 2])                         # dist 2 -> w 0.5
    assert abs(d2 - 2.0 * (1.0 - 2.0 / radius)) < 1e-4


def test_clamp_ceiling():
    """PROPERTY: an edit's OWN clamp overrides the field policy's default --
    the clamp mechanism itself. Exercised on ``fire`` (a bounded field), not
    on a trace plane: since smoke transport v2 D1 no trace edit carries a
    ceiling.

    BREAKS IF: an explicit FieldEdit.clamp stops being applied."""
    g = _GMapStub()
    g.fire[5, 5] = _fire_q(0.3)
    apply_field_edit(g, FieldEdit("fire", Region.TILE, (5, 5), 0.4,
                                  EditMode.ADD, clamp=(0.0, 0.5)), _rng())
    assert abs(_fire_d(g.fire[5, 5]) - 0.5) < 1e-4    # 0.3 + 0.4 -> clamped to 0.5


def test_policy_default_clamp_applied():
    """PROPERTY (smoke transport v2, design §11 D1, Erik's ruling 2026-10-04:
    "I want smoke adding to be additive"): a trace edit with no explicit
    clamp gets the policy's TRACE_CLAMP -- an ADD adds exactly its amount on
    top of whatever the tile holds (no ceiling at 1), and a REMOVE bottoms
    out at 0 (the floor stays).

    BREAKS IF: the smoke / gas policy goes back to a [0, 1] ceiling (which
    cut a compressed tile's existing smoke on every deposit)."""
    g = _GMapStub()
    g.smoke[5, 5] = _smoke_trace_q(0.8)
    apply_field_edit(g, FieldEdit("smoke", Region.TILE, (5, 5), 5.0,
                                  EditMode.ADD), _rng())
    assert abs(_smoke_d(g.smoke[5, 5]) - 5.8) < 1e-4
    apply_field_edit(g, FieldEdit("smoke", Region.TILE, (5, 5), 100.0,
                                  EditMode.REMOVE), _rng())
    assert int(g.smoke[5, 5]) == 0


# ---------------------------------------------------------------------------
# Per-field skip-mask
# ---------------------------------------------------------------------------
def test_smoke_edit_skips_solid():
    g = _GMapStub()
    g.solid[6, 6] = True  # centre tile is a wall
    apply_field_edit(g, FieldEdit("smoke", Region.DISC, (6, 6, 3.0), 1.0,
                                  EditMode.ADD, Falloff.FLAT, clamp=(0.0, 1.0)),
                     _rng())
    assert int(g.smoke[6, 6]) == 0, "smoke wrote a solid tile"
    assert abs(_smoke_d(g.smoke[6, 5]) - 1.0) < 1e-4, "smoke did not write the open neighbour"


def test_fire_edit_skips_non_flammable():
    g = _GMapStub()
    g.flammable[:] = True
    g.flammable[6, 6] = False  # centre is non-flammable
    apply_field_edit(g, FieldEdit("fire", Region.DISC, (6, 6, 3.0), 1.0,
                                  EditMode.MAX, Falloff.FLAT, clamp=(0.0, 1.0)),
                     _rng())
    assert int(g.fire[6, 6]) == 0, "fire wrote a non-flammable tile"
    assert abs(_fire_d(g.fire[6, 5]) - 1.0) < 1e-4, "fire did not ignite the flammable neighbour"


def test_wave_source_skips_vacuum_and_solid():
    # EOS refactor P3 (design §6): the skip-mask is UNCHANGED (still solid +
    # vacuum), but the deposit lands in `temperature` now (an energy deposit),
    # not the retired `wave_source` array — assert on the real target.
    g = _GMapStub()
    g.is_vacuum[6, 7] = True
    g.solid[6, 5] = True
    apply_field_edit(g, FieldEdit("wave_source", Region.DISC, (6, 6, 3.0), 1.0,
                                  EditMode.ADD, Falloff.FLAT), _rng())
    assert int(g.temperature[6, 7]) == 0, "wave_source wrote a vacuum tile"
    assert int(g.temperature[6, 5]) == 0, "wave_source wrote a solid tile"
    assert float(g.temperature[6, 6]) / HEAT_SCALE == 1.0


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
def test_a_policy_without_an_integer_currency_is_refused():
    """#29 P3e: every write through FieldEdit is integer in its field's
    currency; a policy whose dtype has none (the old float fields -- no
    shipped field is one) is REFUSED, never combined in floats. Breaks if a
    float combine path returns."""
    import pytest
    g = _GMapStub()
    from simulation.field_edit import FIELD_POLICY, _FieldPolicy
    FIELD_POLICY["light_rgb"] = _FieldPolicy("float", None, None)
    try:
        with pytest.raises(ValueError, match="no currency"):
            apply_field_edit(g, FieldEdit("light_rgb", Region.TILE, (5, 5),
                                          0.5, EditMode.ADD, channel=1),
                             _rng())
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
    assert abs(_fire_d(g1.fire[5, 5]) - 0.8) < 1e-4


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


def test_skipped_tile_draws_no_noise():
    # A noisy smoke disc: the number of draws must equal the number of
    # SURVIVING (non-skipped) tiles, not the full region tile count. So masking
    # one interior tile solid must drop the draw count by exactly one — proving
    # the skip-mask veto precedes the draw (skipped tiles draw nothing). And,
    # since #29 P3b, every OTHER tile keeps its value: a tile's draw is keyed
    # by (tick, edit key, tile), not by its place in a stream.
    e = FieldEdit("smoke", Region.DISC, (6, 6, 4.0), 0.8, EditMode.ADD,
                  Falloff.LINEAR, clamp=(0.0, 1.0), noise=0.85, source_id=1)

    g_clean = _GMapStub()
    rng_clean = _rng(seed=3)
    apply_field_edit(g_clean, e, rng_clean, key=0)
    n_survivors = int(np.count_nonzero(g_clean.smoke > 0))
    # Every surviving tile drew exactly once (LINEAR weight > 0 inside the disc).
    assert rng_clean.drawn == n_survivors,         f"draws {rng_clean.drawn} != surviving tiles {n_survivors}"

    g_solid = _GMapStub()
    g_solid.solid[5, 5] = True  # one interior disc tile masked out
    rng_solid = _rng(seed=3)
    apply_field_edit(g_solid, e, rng_solid, key=0)
    assert float(g_solid.smoke[5, 5]) == 0.0
    # Exactly one fewer draw — the skipped tile drew nothing.
    assert rng_solid.drawn == rng_clean.drawn - 1,         f"masking a solid tile changed draw count by {rng_clean.drawn - rng_solid.drawn}, expected 1"
    keep = np.ones(g_clean.smoke.shape, dtype=bool)
    keep[5, 5] = False
    assert np.array_equal(g_clean.smoke[keep], g_solid.smoke[keep])


def test_a_noisy_edit_needs_its_edit_key():
    """apply_field_edit refuses a noisy edit without its key (#29 P3b)."""
    e = FieldEdit("smoke", Region.TILE, (2, 2), 0.5, noise=0.5, source_id=1)
    with pytest.raises(ValueError, match="edit key"):
        apply_field_edit(_GMapStub(), e, _rng(1))


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
    """The migrated add_explosion_smoke deposit == THE integer law written by
    hand (#29 P3e): the amount quantized once into the trace currency, the
    DISC's LINEAR weight ``falloff_round`` on the Q16 radius, one integer
    noise draw per surviving tile in row-major order, the product rounded
    once (half away) -- on the same seed, bit for bit."""
    from simulation.physics import add_explosion_smoke
    from simulation import draws, unit_fixed as ufx

    radius, noise, seed = 8, 0.85, 11
    cy, cx = 12, 12
    one = ufx.FP_ONE

    ref = _open_gmap()
    ref_q = np.zeros(ref.smoke.shape, dtype=np.int64)
    from simulation.draws import MatchDraws
    from simulation.rng_salts import Salt
    rng_ref = MatchDraws(seed)
    amount_q = gas_fixed.quantize_trace_scalar(0.8)
    low_q = one - ufx.quantize_scalar(noise)
    r_q = radius * one
    h, w = ref.material.shape
    for ddy in range(-radius, radius + 1):
        for ddx in range(-radius, radius + 1):
            ny, nx = cy + ddy, cx + ddx
            if 0 <= ny < h and 0 <= nx < w and not ref.solid[ny, nx]:
                d2 = (ddy * ddy + ddx * ddx) << 32
                if d2 < r_q * r_q:
                    wgt = ufx.falloff_round(one, d2, r_q * r_q)
                    # keyed: (tick 0, edit key = the edit's seq 0, tile)
                    mult = draws.noise_mult_q16(
                        rng_ref.at(Salt.FIELD_NOISE, ny * w + nx, 0), low_q)
                    ref_q[ny, nx] += ufx.div_round(amount_q * wgt * mult,
                                                   one * one)

    got = _open_gmap()
    got.smoke[:] = 0
    q = EditQueue()
    add_explosion_smoke(got, q, cy, cx, radius, noise=noise)
    q.flush(got, _rng(seed))

    assert np.array_equal(ref_q, got.smoke), \
        "migrated add_explosion_smoke diverged from the integer law"


def test_apply_explosion_atmosphere_equivalence():
    """EOS refactor P3 (design §6): the `atmosphere` FieldEdit policy no
    longer writes P directly (solver-owned, materialized once/tick) — it
    deposits a bulk-N split 21/79 across gas[O2]/gas[INERT_N2] instead
    (`_combine_atmosphere_to_N`). The old "== legacy direct-P += pressure*
    falloff" equivalence this test asserted is EXACTLY what the migration
    retires; re-pointed to assert the new N-split contract instead (per-tile
    deposit == pressure*falloff split 21/79, gas[O2]/gas[INERT_N2] the only
    fields apply_explosion's atmosphere edit touches — `gmap.atmosphere`
    itself is untouched by this FieldEdit, since it is solver-materialized)."""
    from simulation.physics import apply_explosion
    from simulation.gases import O2, INERT_N2
    import math

    radius, pressure, wall_damage = 6, 3.0, 0.0  # 0 wall damage: no topology edit
    cy, cx = 12, 12

    ref = _open_gmap()
    h, w = ref.material.shape
    o2_0 = ref.gas[O2].copy()
    n2_0 = ref.gas[INERT_N2].copy()
    ref_o2 = gas_fixed.dequantize_bulk(ref.gas[O2])
    ref_n2 = gas_fixed.dequantize_bulk(ref.gas[INERT_N2])
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w:
                dist = math.sqrt(dy * dy + dx * dx)
                if dist <= radius and not ref.solid[ny, nx] and not ref.is_vacuum[ny, nx]:
                    dep = pressure * (1.0 - dist / radius)
                    ref_o2[ny, nx] += dep * 0.21
                    ref_n2[ny, nx] += dep * 0.79

    got = _open_gmap()
    q = EditQueue()
    apply_explosion(got, q, cy, cx, radius, pressure, wall_damage)
    q.flush(got, _rng(0))

    got_o2 = gas_fixed.dequantize_bulk(got.gas[O2])
    got_n2 = gas_fixed.dequantize_bulk(got.gas[INERT_N2])
    assert np.allclose(ref_o2, got_o2, atol=1e-4), \
        "migrated apply_explosion O2 deposit diverged from the 21% split"
    assert np.allclose(ref_n2, got_n2, atol=1e-4), \
        "migrated apply_explosion inert_N2 deposit diverged from the 79% split"
    # Sanity: the bulk species actually changed somewhere; P (`atmosphere`)
    # itself is NOT touched by this FieldEdit (solver-materialized).
    assert not np.array_equal(got.gas[O2], o2_0)
    assert not np.array_equal(got.gas[INERT_N2], n2_0)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
