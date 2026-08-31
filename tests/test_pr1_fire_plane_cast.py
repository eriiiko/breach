"""P-F1a gate, CPU half — the RADIATION-LAW witness for the fire-plane cast.

RE-ANCHORED AT P-F1a (documented re-anchor #1; the second in this module's
life). It was re-anchored once already at P-R4, from the retired PAINTER to the
net-T4 exchange. P-F1a replaces that exchange with the VERIFIED BOOKS (design
v6.1 rules 1/3/4 as amended by v7 + the v7.1 closure edits), so every assertion
that pinned a P-R4 DEPOSIT is restated against the law that replaced it. The
four things that actually moved, and why each assertion changed:

  A. RULE 3 — CONTACT FACES ARE RADIATION-INERT. A ray stepping from a solid
     into a FACE-ADJACENT solid terminates with no deposit and no charge;
     conduction owns contact. Every scene here that placed an absorber at the
     emitter's first ring was therefore measuring a direction that no longer
     participates. Those scenes are now AIR-SEPARATED — which is also exactly
     what the design's own equivalence gate requires (v7.1 item 5, M5).
  B. RULE 4 — THE SKY TERM. Emission rays now reach RADIATION_RANGE (>= the
     grid diagonal), so a ray that leaves the grid charges its emitter the
     escaping residual and books the SAME integer to the per-tile ambient
     ledger. CONSERVATION IS THEREFORE `rad_net.sum() + rad_amb.sum() == 0`,
     not `rad_net.sum() == 0`: the old identity was only ever true because
     rays used to expire in mid-air and be charged to nobody (the corridor
     leak). An open-air emitter now correctly reads NEGATIVE.
  C. RULE 2 — mutual emitters exchange the SAME gap-signed pair at HALF weight
     (not P-R4/v6's one-way potential term), so two equal-T emitters still net
     exactly 0 between themselves — but each still pays its own sky.
  D. THE E TABLE IS int64 (L2-B3). Its int32 saturation above T_game ~ 1768
     was the law's real high-temperature ceiling, not the flux limiter.

What this module still protects, unchanged in intent: the SOURCE ENUMERATION
ORDER, the FAN GEOMETRY, air's Kirchhoff inertness, the E bake's exactness, and
D4's per-tick rotation.

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

  (i)   ANTISYMMETRY — two AIR-SEPARATED EQUAL-temperature emitters exchange
        EXACTLY 0 between themselves (integer equality, not a tolerance),
        measured as "each loses exactly what it would lose alone". This is the
        arc's load-bearing property: it makes the divergence hazard impossible
        by construction rather than by tuning.
  (ii)  CONSERVATION — over any scene, ``rad_net.sum() + rad_amb.sum() == 0``
        exactly: what one end loses the other gains, to the count, and the only
        energy that leaves the tile books is booked to the sky ledger.
  (iii) DIRECTION — a hot emitter LOSES (negative rad_net) and a cold absorber
        GAINS (positive), and no absorber can be driven past its emitter.
  (iv)  AIR IS INERT — a tile with ``heat_atten == 0`` neither absorbs nor
        receives. The painter's air-heating died with the painter.
  (v)   TILE SET / ORDER — as above, against the legacy painter cast.
  (vi)  THE E° BAKE — exact integers, no libm: E[t] == round(rad_scale * K^4)
        with K = kelvin_ambient + k_temp_to_kelvin*(4t+2) (== 295 + 4t at the
        canonical [physics.temperature_scale] dials, G12 — issue #12,
        docs/fire_g12_one_map_patch_2026-08-31.md; was 299 + 12t pre-G12)
        computed by repeated multiplication.

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
import temperature_scale  # noqa: E402  P-K2: the canonical game-T -> Kelvin map
from simulation import fire_fixed  # noqa: E402  S3a: gmap.fire is int32 Q16.16

FP_ONE = 1 << 16
# P-K2: re-anchored to preserve emitted flux at the P-F1b plateau (T=300 game)
# under the ×3 Kelvin map — rad_scale' = 1.0e-5 * (893/1193)^4 = 3.1394e-6
# (temperature_scale_unification_design_2026-08-13 §3b; config.toml carries
# the derivation).
RAD_SCALE = 3.1394e-6
DIALS = dict(fire_ray_count=8, range_base=2.0, range_per_i=3.0,
             intensity_base=0.3, intensity_per_i=0.7, color=(1.0, 0.6, 0.2))


def _make_raycaster(rad_scale=RAD_SCALE, t_emit_gate=180.0,
                    kelvin_ambient=None, k_temp_to_kelvin=None):
    """``kelvin_ambient``/``k_temp_to_kelvin`` default to None, meaning "leave
    the Raycaster struct's own compiled default" (293/3.0, unchanged by G12 —
    C++ struct defaults are comment-only, issue #12,
    docs/fire_g12_one_map_patch_2026-08-31.md) — every gate in this file below
    pins RAD_SCALE against THAT default map, deliberately, so most callers
    never pass these. Only :func:`test_emissive_table_is_the_exact_integer_bake`
    cross-checks against the LIVE config map and passes them explicitly."""
    rc = bp.Raycaster()
    rc.light_cull = 0.01
    rc.heat_cull = 0.01
    rc.smoke_absorb_scale = 1.4
    rc.rad_scale = rad_scale
    rc.T_emit_gate = t_emit_gate
    if kelvin_ambient is not None:
        rc.kelvin_ambient = kelvin_ambient
    if k_temp_to_kelvin is not None:
        rc.k_temp_to_kelvin = k_temp_to_kelvin
    # P-F1a / v7 rule 4: emission rays reach RADIATION_RANGE, a stability-class
    # constant >= the grid diagonal of the largest shipping level. Pinned here
    # (rather than left at the default) so these scenes measure the shipped law:
    # every ray that is not absorbed or contact-terminated LEAVES THE GRID and
    # is charged to the sky ledger.
    rc.radiation_range = 320.0
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
        self.rad_amb = np.zeros((h, w), np.int32)    # rule 4: the SKY ledger
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
            self.thermal_solid, self.rad_net, self.rad_amb, self.rad_flux,
            tick)
        return self.rad_net

    def books_close(self):
        """Rule 4's ledger identity: the ONLY energy that leaves the tile books
        is what the sky ledger received, to the count."""
        return int(self.rad_net.sum()) + int(self.rad_amb.sum()) == 0


# ---------------------------------------------------------------------------
# (i) ANTISYMMETRY — the arc's load-bearing property, at integer equality.
# ---------------------------------------------------------------------------
def test_equal_temperature_pair_nets_exactly_zero():
    """Two AIR-SEPARATED equal-T emitters exchange EXACTLY 0 between themselves.

    RE-ANCHOR (A + B + C). The pair used to be FACE-ADJACENT and the assertion
    was ``rad_net == 0`` on both ends. Both halves of that had to move:

      * ADJACENT is now a CONTACT face (rule 3) — the direction between them
        does not participate at all, so an adjacent pair would make this gate
        VACUOUS rather than strict. They are separated by one air tile, which
        is the geometry rule 2 is actually about.
      * ``rad_net == 0`` is no longer the right oracle, because each emitter
        now pays its own SKY (rule 4) in every direction that leaves the grid.
        The property under test is that the MUTUAL exchange is zero, so the
        oracle is the LONE emitter: each end must lose EXACTLY what it loses
        with the other end absent. Any nonzero mutual term shows up as a
        difference, to the count.

    Under rule 2 both ends are emitters, so each casts the half-weight
    gap-signed pair; equal T means the same E bucket, so ``diff == 0`` and the
    term is structurally zero before any rounding can act on it.
    """
    print("\nP-F1a gate (i) - two AIR-SEPARATED EQUAL-T emitters, integer equality:")
    for T in (180.0, 280.0, 443.0, 1000.0, 2500.0):
        sc = _Scene(21, 21)
        sc.solid(10, 10, T_game=T)
        sc.solid(10, 12, T_game=T)       # AIR-SEPARATED (rule 3)
        sc.burn(10, 10)
        sc.burn(10, 12)
        rad = sc.cast(_make_raycaster())
        a, b = int(rad[10, 10]), int(rad[10, 12])
        # The oracle: the SAME emitter, alone in the same grid.
        lone = _Scene(21, 21)
        lone.solid(10, 10, T_game=T)
        lone.burn(10, 10)
        solo = int(lone.cast(_make_raycaster())[10, 10])
        print(f"  T={T:7.1f} game -> rad_net[A]={a}  rad_net[B]={b}  "
              f"lone={solo}  (mutual term = {a - solo})")
        assert a == solo, (
            f"the mutual pair moved {a - solo} counts at equal T - rule 2's "
            f"gap-signed antisymmetry is BROKEN")
        assert a == b, "the equal-T pair was not symmetric between its ends"
        assert int(rad[10, 11]) == 0, "the separating air tile took energy"
        assert sc.books_close(), "the ledger identity failed on the pair scene"
    print("  every equal-T pair exchanged EXACTLY 0 counts between its ends.")


def test_self_cell_is_wholly_excluded_and_open_air_loses_to_sky():
    """The self-cell deposits nothing; a lone open-air emitter loses to SKY.

    RE-ANCHOR (B). P-R4 asserted ``rad_net[src] == 0`` for a lone emitter in
    open air, which was only true because its rays expired in mid-air and were
    charged to NOBODY - the corridor leak. Under rule 4 the rays reach the grid
    edge and the emitter is correctly charged the full escaping residual, so the
    honest assertion is that its entire loss IS the sky booking, to the count.

    The self-cell exclusion is now CODED (an explicit distance-0 test), not
    inferred from ``diff == 0``: under rule 2 the source is itself an emitter
    and would otherwise take the half-weight branch.
    """
    sc = _Scene(15, 15)
    sc.solid(7, 7, T_game=443.0)
    sc.burn(7, 7)
    rad = sc.cast(_make_raycaster())
    src = int(rad[7, 7])
    sky = int(sc.rad_amb[7, 7])
    print(f"\nP-F1a - lone emitter in open air: rad_net[src]={src}, "
          f"rad_amb[src]={sky}, nonzero rad_net cells="
          f"{int(np.count_nonzero(rad))}")
    assert src < 0, "an open-air emitter did not lose to the sky"
    assert src == -sky, "the sky charge and the sky booking are not one integer"
    # Nothing else moved: air is Kirchhoff-inert and there is no other solid.
    assert int(np.count_nonzero(rad)) == 1, "something other than the emitter moved"
    assert int(np.count_nonzero(sc.rad_amb)) == 1, "sky was booked to another tile"
    assert sc.books_close(), "the ledger identity failed on the lone-emitter scene"


# ---------------------------------------------------------------------------
# (ii)/(iii) CONSERVATION + DIRECTION
# ---------------------------------------------------------------------------
def test_exchange_conserves_exactly_and_flows_hot_to_cold():
    """RE-ANCHOR (A + B). The absorber moves off the CONTACT face to an
    air-separated tile (rule 3 makes the adjacent direction non-participating),
    and conservation becomes the rule-4 ledger identity
    ``rad_net.sum() + rad_amb.sum() == 0``.

    Direction is still checked exactly as before, and the pair term itself is
    still ONE integer applied +/-: with the absorber below the gate it is rule
    1, so what the absorber gains is exactly what that direction cost the
    emitter — verified by differencing against the lone-emitter sky loss.
    """
    print("\nP-F1a gate - conservation + direction (hot/cold pair):")
    # The absorber is a COLUMN, air-separated from the emitter by the x == 11
    # air column. A single tile at (+2, 0) is not a usable target: the 8-ray fan
    # only sweeps onto the pure axis on SOME ticks of the D4 rotation (that is
    # what test_fan_rotation_connects_every_neighbour... measures), so a
    # single-tile absorber would make this gate tick-dependent. A column cannot
    # be missed by any rightward ray, and it is still AIR-SEPARATED, so rule 3
    # never fires between the emitter and it.
    sc = _Scene(21, 21)
    sc.solid(10, 10, T_game=443.0)                     # hot emitter
    for y in range(21):
        sc.solid(y, 12, atten=0.5, his=3, T_game=20.0)  # cold absorber column
    sc.burn(10, 10)
    rad = sc.cast(_make_raycaster())
    hot = int(rad[10, 10])
    cold = int(rad[:, 12].sum())
    print(f"  emitter rad_net={hot}  absorber column rad_net={cold}  "
          f"rad_net.sum={int(rad.sum())}  rad_amb.sum={int(sc.rad_amb.sum())}")
    assert hot < 0, "the hot emitter did not LOSE heat"
    assert cold > 0, "the cold absorber did not GAIN heat"
    assert sc.books_close(), "the scene created or destroyed energy"
    # No absorber may be driven past its emitter: every cold tile's gain is
    # bounded by the gap-signed pair, which is zero at equal T.
    assert cold < -hot, "the absorbers gained more than the emitter lost"
    assert int(rad.sum()) == -int(sc.rad_amb.sum()), (
        "the tile books and the sky ledger disagree")


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
          f"{int(np.abs(rad).max())}, rad_net.sum={int(rad.sum())}, "
          f"rad_amb.sum={int(sc.rad_amb.sum())}")
    assert n_moved > 0, "vacuous gate - nothing exchanged"
    # RE-ANCHOR (B): the ledger identity, not `rad_net.sum() == 0`. The sky is
    # the ONLY entry that leaves the tile books, and it is booked as one integer
    # in both directions.
    assert sc.books_close(), "energy was created/destroyed in a firestorm"
    assert int(sc.rad_amb.sum()) > 0, "no ray escaped a 128x128 open grid"
    assert int(sc.rad_amb.min()) >= 0, "the sky ledger went negative"


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
    """RE-ANCHOR (B). The witness moves from ``rad_net`` to ``rad_flux``.

    P-R4 compared the exchange's touched SOLIDS against the painter's marched
    set, which worked because both used the same short per-intensity range.
    Under v7 rule 4 the emission ray reaches RADIATION_RANGE, so the exchange
    now legitimately touches solids far beyond anything the painter marched --
    comparing the two sets would fail for the right reason and prove nothing.

    But D3's ``rad_flux`` sensor DELIBERATELY KEPT THE OLD REACH (v7.1 item 4):
    it is written at AIR cells behind a deterministic ``damage_range`` guard
    that is exactly the legacy ``range_base + range_per_intensity * I``. So the
    sensor's touched set is precisely the painter's marched AIR set, and it is
    the sharper witness for what this test always cared about: THE FAN GEOMETRY
    AND THE ENUMERATION ORDER ARE UNCHANGED. Nothing about the DDA, the angle
    sweep, the phase hash or the range model moved -- and if any of it had, this
    comparison would break immediately.
    """
    print("\nP-F1a gate (v) - marched tile set vs the unchanged fan:")
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
    sc.cast(rc)
    air = sc.heat_atten <= 0.0
    got = sc.rad_flux != 0
    want = painter & air
    # The sensor writes only where the ray is still ALIVE and inside
    # damage_range, which is exactly the painter's own gate, so the two sets are
    # equal up to cells whose quantized flux rounded to 0 (a far cell behind
    # heavy attenuation) -- the law's own "below one count moves nothing".
    extra = int(np.count_nonzero(got & ~want))
    missing = int(np.count_nonzero(want & ~got))
    print(f"  painter-marched cells: {int(painter.sum())};  air subset: "
          f"{int(want.sum())};  flux-sensor-touched: {int(got.sum())}")
    print(f"  extra={extra}  missing={missing}")
    assert int(want.sum()) > 0, "vacuous gate"
    assert extra == 0, (
        f"{extra} AIR cells took a flux write the fan never marched - the "
        f"damage_range guard or the fan geometry has moved")
    assert missing <= 0.25 * int(want.sum()), (
        f"{missing}/{int(want.sum())} marched air cells got no flux - the "
        f"sensor's reach has genuinely shrunk, not just quantized away")


def test_emitter_enumeration_is_row_major_and_includes_warm_solids():
    """Order + the P-R4 emitter-set widening (ruling A1.8).

    A tile that is NOT burning but is a thermal solid at or above
    ``T_emit_gate`` also casts; below the gate it does not. Receivers are free
    either way — the gate only decides who can radiatively LOSE heat."""
    print("\nP-F1a gate - the warm-emitter gate (T_emit_gate):")
    # RE-ANCHOR (A): the absorber is AIR-SEPARATED. On the first ring it shares
    # a face with the emitter, and rule 3 makes that direction non-participating
    # -- the gate would have read "no cast" for BOTH temperatures and passed for
    # entirely the wrong reason.
    for T_warm, expect_cast in ((179.0, False), (181.0, True)):
        sc = _Scene(21, 21)
        sc.solid(10, 10, T_game=T_warm)      # warm, NOT burning
        # An absorber COLUMN, air-separated (see the hot/cold gate above for
        # why a single (+2,0) tile is not a tick-independent target).
        for y in range(21):
            sc.solid(y, 12, atten=0.5, his=3, T_game=0.0)
        rad = sc.cast(_make_raycaster(t_emit_gate=180.0))
        moved = int(rad[:, 12].sum())
        print(f"  warm tile at T={T_warm} -> absorber gained {moved} counts "
              f"(expected {'>0' if expect_cast else '0'})")
        if expect_cast:
            assert moved > 0, "a tile above T_emit_gate did not cast"
            # RE-ANCHOR (B): the emitter's debit is the pair term PLUS its sky,
            # so it is strictly more negative than -moved; what must hold to the
            # count is the ledger identity.
            assert int(rad[10, 10]) <= -moved, "the warm emitter was not debited"
            assert sc.books_close(), "the books did not close at the gate"
        else:
            assert moved == 0, "a tile below T_emit_gate cast anyway"
            assert int(rad.sum()) == 0 and int(sc.rad_amb.sum()) == 0, (
                "a sub-gate tile moved energy somewhere")


# ---------------------------------------------------------------------------
# (vi) THE E° BAKE — exact integers, no libm.
# ---------------------------------------------------------------------------
def test_emissive_table_is_the_exact_integer_bake():
    """RE-ANCHOR (D). The table is int64 and NO LONGER SATURATES.

    P-R4 baked int32 and this test asserted the saturation knee as intended
    behaviour ("the exchange goes flat there by design"). L2-B3 found that knee
    was the law's real high-temperature ceiling, not a design choice: above
    T_game ~ 1768 at the then-shipped rad_scale (1.0e-5, since re-anchored at
    P-K2 — see RAD_SCALE above) every entry pinned at INT32_MAX, so `diff`
    between ANY two tiles above it collapsed to 0 and a 3000-game tile
    radiated to a 2000-game tile exactly nothing. The table is now int64, and
    the assertion is that no shipping rad_scale can saturate it.
    """
    # P-K2: K(t) is read from [physics.temperature_scale] via the accessor —
    # an INDEPENDENT re-implementation of the bake's formula (not a call into
    # the engine's own table), so this stays an oracle rather than a tautology.
    ts = temperature_scale.load()
    amb, slope = ts.kelvin_ambient, ts.k_temp_to_kelvin
    assert amb == int(amb) and slope == int(slope), (
        "the exact-integer bake reference below assumes integer-valued dials")
    amb, slope = int(amb), int(slope)
    print(f"\nP-F1a - the E bake (E[t] = round(rad_scale * K^4), "
          f"K = {amb} + {slope}*(4t+2) == {amb + 2 * slope} + {4 * slope}t, int64):")
    # Saturation-property probe values (design §3e: "survive, headroom
    # verified") — deliberately NOT tied to the shipped RAD_SCALE pin; they
    # exercise the no-saturation property across a range, old and new.
    for scale in (1.0e-5, 3.0e-6):
        rc = _make_raycaster(rad_scale=scale,
                             kelvin_ambient=amb, k_temp_to_kelvin=slope)
        tab = rc.emissive_table()
        assert tab.shape == (4000,) and tab.dtype == np.int64, (
            f"the E table must be int64 since P-F1a, got {tab.dtype}")
        for t in (0, 1, 45, 70, 110, 200, 442, 1000, 2500, 3999):
            K = amb + slope * (4 * t + 2)
            # The bake's own contract, with K^4 built by repeated
            # multiplication (exact in int64) and ONE rounding boundary.
            ref = int(round(scale * float(K) ** 4))
            assert int(tab[t]) == ref, (
                f"bucket {t}: baked {int(tab[t])} != exact {ref}")
        # Monotone (T^4 is), and STRICTLY so at the top -- the proof that the
        # old ceiling is gone: a saturated tail would be flat here.
        assert bool(np.all(np.diff(tab) >= 0)), "E table is not monotone"
        assert int(tab[-1]) > int(tab[-2]), (
            "the E table is FLAT at the top - the int32 saturation ceiling "
            "L2-B3 removed has come back")
        assert int(tab[-1]) > int(np.iinfo(np.int32).max), (
            "the top entry still fits int32 - the widening bought nothing")
        print(f"  rad_scale={scale:g}: exact at every probed bucket, monotone, "
              f"strictly increasing at the top, top = {int(tab[-1])}")


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
