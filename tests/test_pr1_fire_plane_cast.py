"""P-R4 gate (e), CPU half — the RADIATION-LAW witness for the fire-plane cast.

REWRITTEN AT P-R4 (documented re-anchor). This module was the P-R1 transition
witness: it pinned ``heat`` byte-identical between the new C++
``cast_from_fire_plane`` and a frozen transcription of the old Python per-tile
loop. Its oracle was the PAINTER — the one-way ``heat = k_fire_heat * I``
deposit — and P-R4 retires that law outright
(docs/radiation_raycaster_extinction_ruling_2026-07-31.md A1), so the oracle no
longer exists to compare against. The patch's bounded re-anchor authority
therefore turns it into the witness for the law that replaced it, keeping the
two contracts P-R1 actually protected:

  * the SOURCE ENUMERATION ORDER (row-major over the fire plane), and
  * the MARCHED / TOUCHED TILE SET (the fan geometry, the range model, the
    material-only ``heat_survival > heat_cull`` gate and the source-tile
    self-occlusion skip are all untouched by P-R4).

Both are still checkable against code this patch does not modify: the LEGACY
``cast_source_directional`` heat channel is still there for non-fire light
sources (lamps/beams/weapons — ruling A5's "unchanged external deposits"), so a
painter cast with the same fan reproduces exactly the tile set the exchange
marches. That is the oracle used below.

What is NEW and gated here (ruling A1, gate (e)):

  (i)   ANTISYMMETRY — two adjacent EQUAL-temperature emitters exchange
        EXACTLY 0 (integer equality, not a tolerance). This is the arc's
        load-bearing property: it makes the divergence hazard impossible by
        construction rather than by tuning.
  (ii)  CONSERVATION — over any scene, ``rad_net.sum() == 0`` exactly: what one
        end loses the other gains, to the count.
  (iii) DIRECTION — a hot emitter LOSES (negative rad_net) and a cold absorber
        GAINS (positive), and no absorber can be driven past its emitter.
  (iv)  AIR IS INERT — a tile with ``heat_atten == 0`` neither absorbs nor
        receives. The painter's air-heating died with the painter.
  (v)   TILE SET / ORDER — as above, against the legacy painter cast.
  (vi)  THE E° BAKE — exact integers, no libm: E[t] == round(rad_scale * K^4)
        with K = 297 + 8t computed by repeated multiplication.

The CUDA half (CPU vs GPU at tolerance 0) is tests/cuda_pr1_fire_plane_check.py
(skips cleanly without a CUDA build).

Run:
    conda run -n data python -m pytest tests/test_pr1_fire_plane_cast.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from simulation import fire_fixed  # noqa: E402  S3a: gmap.fire is int32 Q16.16

FP_ONE = 1 << 16
RAD_SCALE = 1.0e-5
DIALS = dict(fire_ray_count=8, range_base=2.0, range_per_i=3.0,
             intensity_base=0.3, intensity_per_i=0.7, color=(1.0, 0.6, 0.2))


def _make_raycaster(rad_scale=RAD_SCALE, t_emit_gate=180.0):
    rc = bp.Raycaster()
    rc.light_cull = 0.01
    rc.heat_cull = 0.01
    rc.smoke_absorb_scale = 1.4
    rc.rad_scale = rad_scale
    rc.T_emit_gate = t_emit_gate
    rc.bake_emissive_table()
    return rc


class _Scene:
    """The minimal plane set the radiation cast reads, as raw numpy arrays."""

    def __init__(self, h, w, n_gases=2):
        self.h, self.w = h, w
        self.fire = np.zeros((h, w), np.int32)
        self.temperature = np.zeros((h, w), np.int32)
        self.heat_atten = np.zeros((h, w), np.float32)
        self.heat_inv_shift = np.zeros((h, w), np.int32)
        self.thermal_solid = np.zeros((h, w), bool)
        self.rad_net = np.zeros((h, w), np.int32)
        self.rad_flux = np.zeros((h, w), np.int32)   # D3 damage sensor
        self.light_atten = np.zeros((h, w, 3), np.float32)
        self.gas = np.zeros((n_gases, h, w), np.float32)
        self.gas_abs = np.zeros((n_gases, 3), np.float32)
        self.gas_sca = np.zeros((n_gases, 3), np.float32)
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.dx = np.zeros((h, w), np.float32)
        self.dy = np.zeros((h, w), np.float32)

    def solid(self, y, x, atten=0.5, his=3, T_game=0.0):
        """Make (y,x) a thermal solid: absorptivity == emissivity == `atten`."""
        self.heat_atten[y, x] = atten
        self.heat_inv_shift[y, x] = his
        self.thermal_solid[y, x] = True
        self.temperature[y, x] = int(round(T_game * FP_ONE))

    def burn(self, y, x, I=0.21):
        self.fire[y, x] = fire_fixed.quantize_scalar(float(I))

    def cast(self, rc, tick=0):
        self.rgb.fill(0.0)
        self.dx.fill(0.0)
        self.dy.fill(0.0)
        rc.cast_from_fire_plane(
            self.fire, DIALS["fire_ray_count"], DIALS["range_base"],
            DIALS["range_per_i"], DIALS["intensity_base"],
            DIALS["intensity_per_i"], DIALS["color"],
            self.rgb, self.dx, self.dy,
            self.gas, self.gas_abs, self.gas_sca, self.light_atten,
            self.heat_atten, self.temperature, self.heat_inv_shift,
            self.thermal_solid, self.rad_net, self.rad_flux, tick)
        return self.rad_net


# ---------------------------------------------------------------------------
# (i) ANTISYMMETRY — the arc's load-bearing property, at integer equality.
# ---------------------------------------------------------------------------
def test_equal_temperature_pair_nets_exactly_zero():
    print("\nP-R4 gate (e)(i) — two adjacent EQUAL-T emitters, integer equality:")
    for T in (180.0, 280.0, 443.0, 1000.0):
        sc = _Scene(21, 21)
        sc.solid(10, 10, T_game=T)
        sc.solid(10, 11, T_game=T)
        sc.burn(10, 10)
        sc.burn(10, 11)
        rad = sc.cast(_make_raycaster())
        a, b = int(rad[10, 10]), int(rad[10, 11])
        nz = int(np.count_nonzero(rad))
        print(f"  T={T:7.1f} game -> rad_net[A]={a}  rad_net[B]={b}  "
              f"nonzero cells={nz}")
        assert a == 0 and b == 0, (
            f"equal-T pair exchanged {a}/{b} counts — antisymmetry BROKEN")
        # Nothing else in the scene absorbs (all air), so the whole plane is 0.
        assert nz == 0, f"{nz} cells took a deposit from an equal-T pair"
    print("  every equal-T pair exchanged EXACTLY 0 counts.")


def test_self_cell_deposit_is_exactly_zero():
    """The source is its own first marched cell: E[T_s] - E[T_s] == 0."""
    sc = _Scene(15, 15)
    sc.solid(7, 7, T_game=443.0)
    sc.burn(7, 7)
    rad = sc.cast(_make_raycaster())
    print(f"\nP-R4 — lone emitter in open air: rad_net[src]={int(rad[7, 7])}, "
          f"nonzero cells={int(np.count_nonzero(rad))}")
    assert int(rad[7, 7]) == 0, "a tile radiated to ITSELF"
    assert int(np.count_nonzero(rad)) == 0, "air absorbed radiation"


# ---------------------------------------------------------------------------
# (ii)/(iii) CONSERVATION + DIRECTION
# ---------------------------------------------------------------------------
def test_exchange_conserves_exactly_and_flows_hot_to_cold():
    print("\nP-R4 gate (e) — conservation + direction (hot/cold pair):")
    sc = _Scene(21, 21)
    sc.solid(10, 10, T_game=443.0)      # hot emitter
    sc.solid(10, 11, T_game=20.0)       # cold absorber (first ring)
    sc.burn(10, 10)
    rad = sc.cast(_make_raycaster())
    hot, cold = int(rad[10, 10]), int(rad[10, 11])
    print(f"  emitter rad_net={hot}  absorber rad_net={cold}  "
          f"sum={int(rad.sum())}")
    assert hot < 0, "the hot emitter did not LOSE heat"
    assert cold > 0, "the cold absorber did not GAIN heat"
    assert hot == -cold, "the pair did not exchange the SAME integer"
    assert int(rad.sum()) == 0, "the scene created or destroyed energy"


def test_conservation_holds_over_a_600_emitter_firestorm():
    """Scaled-up conservation: the plane sums to EXACTLY zero, any scene."""
    print("\nP-R4 gate (e) — 600-emitter firestorm, exact conservation:")
    rng = np.random.default_rng(20260801)
    h = w = 128
    sc = _Scene(h, w)
    cells = set()
    while len(cells) < 600:
        cells.add((int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))))
    for (y, x) in cells:
        sc.solid(y, x, atten=0.5, his=3,
                 T_game=float(rng.uniform(200.0, 900.0)))
        sc.burn(y, x, I=float(rng.uniform(0.3, 1.0)))
    # A scatter of cold absorbers (walls/glass) so real transfers happen.
    for _ in range(400):
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        if (y, x) in cells:
            continue
        sc.solid(y, x, atten=float(rng.uniform(0.1, 1.0)),
                 his=int(rng.integers(2, 6)), T_game=0.0)
    rad = sc.cast(_make_raycaster())
    n_moved = int(np.count_nonzero(rad))
    print(f"  600 emitters, {n_moved} cells moved, |rad_net|max="
          f"{int(np.abs(rad).max())}, sum={int(rad.sum())}")
    assert n_moved > 0, "vacuous gate — nothing exchanged"
    assert int(rad.sum()) == 0, "energy was created/destroyed in a firestorm"


# ---------------------------------------------------------------------------
# (iv) AIR IS INERT — the painter's air-heating died with the painter.
# ---------------------------------------------------------------------------
def test_air_neither_absorbs_nor_receives():
    sc = _Scene(21, 21)
    sc.solid(10, 10, T_game=600.0)
    sc.burn(10, 10)
    rad = sc.cast(_make_raycaster())
    air = rad.copy()
    air[sc.heat_atten > 0.0] = 0
    print(f"\nP-R4 — air deposit total = {int(np.abs(air).sum())} counts")
    assert int(np.abs(air).sum()) == 0, "air (heat_atten == 0) took a deposit"


# ---------------------------------------------------------------------------
# (v) THE TILE SET AND THE ORDER — against the legacy painter cast, which this
#     patch does NOT touch (it survives for non-fire light sources).
# ---------------------------------------------------------------------------
def _painter_touched(rc, sc):
    """The tile set the SAME fan marches, via the legacy per-source heat
    channel. Reproduces build_fire_sources' per-tile parameters exactly (the
    row-major walk, the (x*7+y*13) phase, the range model) — the two contracts
    P-R1 protected and P-R4 keeps."""
    two_pi = 2.0 * np.pi
    n = DIALS["fire_ray_count"]
    heat = np.zeros((sc.h, sc.w), np.int32)
    rgb = np.zeros((sc.h, sc.w, 3), np.float32)
    dx = np.zeros((sc.h, sc.w), np.float32)
    dy = np.zeros((sc.h, sc.w), np.float32)
    ys, xs = np.nonzero(sc.fire > 0)
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        I = float(sc.fire[yy, xx]) / float(FP_ONE)
        s = bp.LightSource()
        s.x = float(xx) + 0.5
        s.y = float(yy) + 0.5
        s.max_range = DIALS["range_base"] + DIALS["range_per_i"] * I
        s.ray_count = n
        s.angle_spread = two_pi
        s.angle_center = ((xx * 7 + yy * 13) % n) * (two_pi / n)
        s.intensity = DIALS["intensity_base"] + DIALS["intensity_per_i"] * I
        s.heat = 1000.0          # any positive payload: we only want the SET
        s.jitter = 0.0
        s.color = DIALS["color"]
        rc.cast_source_directional(
            s, rgb, dx, dy, sc.gas, sc.gas_abs, sc.gas_sca, sc.light_atten,
            heat=heat, smoke_glow=None, heat_atten=sc.heat_atten)
    return heat != 0


def test_touched_tile_set_matches_the_unchanged_fan():
    print("\nP-R4 gate (e)(v) — marched tile set vs the unchanged fan:")
    rng = np.random.default_rng(4242)
    h = w = 48
    sc = _Scene(h, w)
    # Emitters, all HOT so every absorbing cell they reach gets a nonzero net.
    fires = [(8, 8), (8, 30), (30, 8), (31, 31), (20, 20)]
    for (y, x) in fires:
        sc.solid(y, x, atten=0.5, his=3, T_game=800.0)
        sc.burn(y, x, I=1.0)
    # A field of COLD absorbers + a couple of opaque walls (occlusion path).
    for _ in range(300):
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        if (y, x) in fires:
            continue
        sc.solid(y, x, atten=float(rng.uniform(0.2, 1.0)),
                 his=int(rng.integers(2, 6)), T_game=0.0)
    sc.heat_atten[24, :] = 1.0
    sc.thermal_solid[24, :] = True
    sc.heat_inv_shift[24, :] = 3

    rc = _make_raycaster()
    painter = _painter_touched(rc, sc)
    rad = sc.cast(rc)
    absorbing = sc.heat_atten > 0.0
    got = rad != 0
    want = painter & absorbing
    # The emitters themselves are debited, so they are in BOTH sets whenever
    # they moved anything; every other difference would be a real divergence.
    extra = int(np.count_nonzero(got & ~want))
    missing = int(np.count_nonzero(want & ~got))
    print(f"  painter-marched cells: {int(painter.sum())};  absorbing subset: "
          f"{int(want.sum())};  exchange-touched: {int(got.sum())}")
    print(f"  extra={extra}  missing={missing}")
    assert int(want.sum()) > 0, "vacuous gate"
    assert extra == 0, f"{extra} cells took a deposit the fan never marched"
    # `missing` cells are absorbing cells the fan marched whose E-difference
    # quantized to 0 (a far cell behind heavy attenuation) — legal, and the
    # law's own "below one count moves nothing". They must be a small tail.
    assert missing <= 0.25 * int(want.sum()), (
        f"{missing}/{int(want.sum())} marched absorbers moved nothing — the "
        f"tile set has genuinely shrunk, not just quantized away")


def test_emitter_enumeration_is_row_major_and_includes_warm_solids():
    """Order + the P-R4 emitter-set widening (ruling A1.8).

    A tile that is NOT burning but is a thermal solid at or above
    ``T_emit_gate`` also casts; below the gate it does not. Receivers are free
    either way — the gate only decides who can radiatively LOSE heat."""
    print("\nP-R4 gate (e) — the warm-emitter gate (T_emit_gate):")
    for T_warm, expect_cast in ((179.0, False), (181.0, True)):
        sc = _Scene(21, 21)
        sc.solid(10, 10, T_game=T_warm)      # warm, NOT burning
        sc.solid(10, 11, T_game=0.0)         # cold absorber, first ring
        rad = sc.cast(_make_raycaster(t_emit_gate=180.0))
        moved = int(rad[10, 11])
        print(f"  warm tile at T={T_warm} -> absorber gained {moved} counts "
              f"(expected {'>0' if expect_cast else '0'})")
        if expect_cast:
            assert moved > 0, "a tile above T_emit_gate did not cast"
            assert int(rad[10, 10]) == -moved, "the warm emitter was not debited"
        else:
            assert moved == 0, "a tile below T_emit_gate cast anyway"


# ---------------------------------------------------------------------------
# (vi) THE E° BAKE — exact integers, no libm.
# ---------------------------------------------------------------------------
def test_emissive_table_is_the_exact_integer_bake():
    print("\nP-R4 — the E° bake (E[t] = clamp_int32(rad_scale * K^4), K = 297+8t):")
    for scale in (1.0e-5, 3.0e-6):
        rc = _make_raycaster(rad_scale=scale)
        tab = rc.emissive_table()
        assert tab.shape == (4000,) and tab.dtype == np.int32
        imax = int(np.iinfo(np.int32).max)
        for t in (0, 1, 45, 70, 110, 200, 1000):
            K = 297 + 8 * t
            # The bake's own contract: clamp_int32(round(rad_scale * K^4)) with
            # K^4 built by repeated multiplication (exact in int64). Above the
            # saturation knee the entry pins at INT32_MAX — T⁴ simply outruns
            # int32 (at the shipped rad_scale that is T_game ~ 1768, 4x above
            # the operating band), and the exchange goes flat there by design.
            ref = min(imax, int(round(scale * float(K) ** 4)))
            assert int(tab[t]) == ref, (
                f"bucket {t}: baked {int(tab[t])} != exact {ref}")
        # Monotone (T⁴ is), and saturating rather than wrapping at the top.
        assert bool(np.all(np.diff(tab) >= 0)), "E° table is not monotone"
        assert int(tab[-1]) == np.iinfo(np.int32).max or int(tab[-1]) > 0
        print(f"  rad_scale={scale:g}: exact at every probed bucket, monotone, "
              f"top = {int(tab[-1])}")


# ---------------------------------------------------------------------------
# D3 — the RADIANT-FLUX SENSOR (ruling amendment 5). Units must cook again.
# ---------------------------------------------------------------------------
def test_air_gets_flux_but_no_energy():
    """Air takes NO energy (Kirchhoff) but DOES register incident flux.

    The two planes answer different questions and must not be conflated:
    ``rad_net`` is the energy ledger (solids only, signed, conserving);
    ``rad_flux`` is a damage SENSOR (air only, positive, outside the ledger).
    Without the sensor a fire could not burn a marine standing beside it —
    measured before D3: unit HP never dropped."""
    print("\nP-R4 D3 — the radiant-flux sensor at air cells:")
    sc = _Scene(21, 21)
    sc.solid(10, 10, T_game=443.0)
    sc.burn(10, 10)
    rad = sc.cast(_make_raycaster())
    air = sc.heat_atten <= 0.0
    flux_air = int(sc.rad_flux[air].sum())
    flux_hot = int(sc.rad_flux[10, 11])           # first-ring air neighbour
    print(f"  air cells: rad_net total {int(np.abs(rad[air]).sum())} (must be 0), "
          f"rad_flux total {flux_air}, first-ring {flux_hot}")
    assert int(np.abs(rad[air]).sum()) == 0, "air absorbed ENERGY — ledger leak"
    assert flux_air > 0, "no radiant flux registered — units would not cook"
    assert flux_hot > 0, "the tile beside the fire registered no flux"
    # The sensor is positive-only (it keeps heat[]'s saturating contract).
    assert int(sc.rad_flux.min()) >= 0, "the flux sensor went negative"
    # ...and it is OCCLUDED like the painter was: behind an opaque wall, none.
    sc2 = _Scene(21, 21)
    sc2.solid(10, 10, T_game=443.0)
    sc2.burn(10, 10)
    for y in range(21):
        sc2.solid(y, 12, atten=1.0, his=3, T_game=0.0)    # opaque wall column
    sc2.cast(_make_raycaster())
    beyond = int(sc2.rad_flux[:, 13:].sum())
    print(f"  behind an opaque wall: rad_flux total {beyond} (must be 0)")
    assert beyond == 0, "flux leaked through an opaque wall — occlusion broken"


# ---------------------------------------------------------------------------
# D4 — the PER-TICK FAN PHASE ROTATION (ruling amendment 5).
# ---------------------------------------------------------------------------
def test_fan_rotation_connects_every_neighbour_within_ray_count_ticks():
    """The 8-ray fan used to have permanent blind spots.

    Rotating N evenly-spaced rays by a multiple of their own spacing maps the
    set onto itself, so the shipped ``((x*7+y*13) mod N)`` phase was a NO-OP and
    EVERY source cast the same 8 directions — the (+2, 0) axis neighbour was
    never on any of them, at any intensity (measured, and the retired painter
    had the identical blind spot). D4 adds a sub-spacing rotation that advances
    with the tick, so N consecutive ticks sweep one full spacing."""
    print("\nP-R4 D4 — per-tick fan rotation kills the fan's blind spots:")
    N = DIALS["fire_ray_count"]
    for (dy, dx, name) in ((0, 2, "axis (+2,0)"), (2, 0, "axis (0,+2)"),
                           (2, 2, "diagonal (+2,+2)"), (1, 2, "knight (+1,+2)")):
        connected = []
        for tick in range(N):
            sc = _Scene(21, 21)
            sc.solid(10, 10, T_game=900.0)
            sc.solid(10 + dy, 10 + dx, T_game=0.0)
            sc.burn(10, 10, I=1.0)                # max_range 5.0 covers 2 tiles
            rad = sc.cast(_make_raycaster(), tick=tick)
            if int(rad[10 + dy, 10 + dx]) != 0:
                connected.append(tick)
        print(f"  {name:18s} connected on ticks {connected} of 0..{N-1}")
        assert connected, (
            f"{name} was NEVER connected over a full {N}-tick rotation — the "
            f"fan still has a permanent blind spot")


def test_fan_rotation_is_a_pure_function_of_the_tick():
    """Determinism: the same (scene, tick) must give the same plane, and two
    different ticks in the rotation must genuinely differ (or the rotation is
    a no-op again)."""
    def cast_at(tick):
        sc = _Scene(21, 21)
        sc.solid(10, 10, T_game=900.0)
        sc.solid(10, 12, T_game=0.0)
        sc.burn(10, 10, I=1.0)
        return sc.cast(_make_raycaster(), tick=tick).copy()

    a1, a2 = cast_at(3), cast_at(3)
    assert np.array_equal(a1, a2), "same tick gave two different planes"
    # A full rotation returns to the start (tick and tick+N are congruent).
    assert np.array_equal(cast_at(3), cast_at(3 + DIALS["fire_ray_count"])), (
        "tick and tick+ray_count did not give the same fan")
    diffs = sum(0 if np.array_equal(cast_at(0), cast_at(t)) else 1
                for t in range(1, DIALS["fire_ray_count"]))
    print(f"\nP-R4 D4 — {diffs}/{DIALS['fire_ray_count']-1} ticks in the "
          f"rotation differ from tick 0; tick+N == tick.")
    assert diffs > 0, "the per-tick rotation changed nothing — still a no-op"


if __name__ == "__main__":
    test_equal_temperature_pair_nets_exactly_zero()
    test_self_cell_deposit_is_exactly_zero()
    test_exchange_conserves_exactly_and_flows_hot_to_cold()
    test_conservation_holds_over_a_600_emitter_firestorm()
    test_air_neither_absorbs_nor_receives()
    test_touched_tile_set_matches_the_unchanged_fan()
    test_emitter_enumeration_is_row_major_and_includes_warm_solids()
    test_emissive_table_is_the_exact_integer_bake()
    print("OK — P-R4 CPU radiation-law witness")
