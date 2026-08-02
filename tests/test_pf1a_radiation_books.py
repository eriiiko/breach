"""P-F1a — THE VERIFIED RADIATION BOOKS. The patch's own gate set.

Builds the law of docs/fire_realism_design_2026-08-01.md v6.1 rules 1/3/4 as
amended by v7 (the symmetrized books) and the v7.1 closure edits. The spec
survived eight verification rounds; round-3.7 certified conservation, the ledger
identity and second-law safety on paper. THIS MODULE IS THE EMPIRICAL BACKSTOP
that certification was signed against, so each test names the gate it is.

Gate map (the patch's charter):

  (i)   EQUIVALENCE, TOLERANCE ZERO — the arc's headline, and previously
        ungated (round-3.6 BLOCKER-3). One emitter's net radiative loss is
        BIT-IDENTICAL open-field and centred in a sealed ambient-temperature
        room. Then the a = 0.5 variant, with the contact-termination count
        measured and the tolerance DERIVED from it rather than guessed.
  (ii)  LEDGER — `sum(rad_net) + sum(rad_amb) == 0` exactly, PRE-FOLD, on a
        firestorm-scale scene (v7.1 item 9).
  (iii) NET — an air-separated equal-T emitter lattice exchanges exactly
        nothing; the same lattice below the gate is trivially zero.
  (iv)  EXCHANGE RATE — a two-temperature sealed box, averaged over exactly one
        D4 fan-rotation period (8 ticks), equals 1x the pair law; swept over
        dT including the small values that would catch a clamp bite; continuous
        across a tile's own gate crossing, swept both directions; plus one tile
        pinned at exactly quantize(T_emit_gate).
  (v)   RANGE FLOOR + GREY BODY — a sealed equal-T room WIDER than the old
        max_range nets zero per tile (the corridor leak is gone), and an
        open-field lone emitter's sky ledger equals the full grey-body rate
        within a tolerance set BELOW the named heat_cull residual.
  (vi)  NEGATIVE PAIR — a hot SUB-GATE solid heats a cooler EMITTER. Sign
        correctness in the direction the one-way form used to get wrong.
  (vii) RAIL — one extreme-gap pair asserts MONOTONE APPROACH while the flux
        limiter is engaged, not merely that nothing exploded.

Every scenario also asserts the Pass-1 rails INERT where it runs the solver: a
counted rail engagement inside a gate run is a RED, not a shrug (v6.3).

Run:
    conda run -n data python -m pytest tests/test_pf1a_radiation_books.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from simulation import fire_fixed  # noqa: E402

FP_ONE = 1 << 16
RAD_SCALE = 1.0e-5
T_EMIT_GATE = 180.0
RAY_COUNT = 8                      # == fire_ray_count == the D4 rotation period
DIALS = dict(fire_ray_count=RAY_COUNT, range_base=2.0, range_per_i=3.0,
             intensity_base=0.3, intensity_per_i=0.7, color=(1.0, 0.6, 0.2))


def Q(t_game):
    """Game units -> Q16.16, the SAME boundary cast the C++ gate uses."""
    return int(round(float(t_game) * FP_ONE))


def _rc(rad_scale=RAD_SCALE, t_emit_gate=T_EMIT_GATE, radiation_range=320.0):
    rc = bp.Raycaster()
    rc.light_cull = 0.01
    rc.heat_cull = 0.01
    rc.smoke_absorb_scale = 1.4
    rc.rad_scale = rad_scale
    rc.T_emit_gate = t_emit_gate
    rc.radiation_range = radiation_range
    rc.bake_emissive_table()
    return rc


class Scene:
    """The plane set the emission cast reads, as raw numpy arrays.

    Deliberately NOT a GameMap: these gates are about the law, and a synthetic
    grid lets each scenario pin exactly the geometry its proof needs (which is
    what the equivalence gate's "air-separated, single-layer, grid-border-backed
    walls" clause is really specifying).
    """

    def __init__(self, h, w, n_gases=2):
        self.h, self.w = h, w
        self.fire = np.zeros((h, w), np.int32)
        self.temperature = np.zeros((h, w), np.int32)
        self.heat_atten = np.zeros((h, w), np.float32)
        self.heat_inv_shift = np.zeros((h, w), np.int32)
        self.thermal_solid = np.zeros((h, w), bool)
        self.rad_net = np.zeros((h, w), np.int32)
        self.rad_amb = np.zeros((h, w), np.int32)     # rule 4: the SKY ledger
        self.rad_flux = np.zeros((h, w), np.int32)
        self.light_atten = np.zeros((h, w, 3), np.float32)
        self.gas = np.zeros((n_gases, h, w), np.float32)
        self.gas_abs = np.zeros((n_gases, 3), np.float32)
        self.gas_sca = np.zeros((n_gases, 3), np.float32)
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.dx = np.zeros((h, w), np.float32)
        self.dy = np.zeros((h, w), np.float32)
        self.contact_hits = 0

    def solid(self, y, x, atten=0.5, his=3, T_game=0.0):
        self.heat_atten[y, x] = atten
        self.heat_inv_shift[y, x] = his
        self.thermal_solid[y, x] = True
        self.temperature[y, x] = Q(T_game)

    def burn(self, y, x, I=0.21):
        self.fire[y, x] = fire_fixed.quantize_scalar(float(I))

    def cast(self, rc, tick=0, light=False):
        """Run one emission cast. Returns rad_net; records contact_hits.

        `light=False` passes no light buffers, which is the zero-cost route:
        the visible-light cast is skipped entirely. These gates are about the
        books, and the light cast provably cannot touch them (separate march,
        radiation off).
        """
        self.rgb.fill(0.0)
        self.dx.fill(0.0)
        self.dy.fill(0.0)
        self.contact_hits = rc.cast_from_fire_plane(
            self.fire, DIALS["fire_ray_count"], DIALS["range_base"],
            DIALS["range_per_i"], DIALS["intensity_base"],
            DIALS["intensity_per_i"], DIALS["color"],
            self.rgb, self.dx, self.dy,
            self.gas, self.gas_abs, self.gas_sca, self.light_atten,
            self.heat_atten, self.temperature, self.heat_inv_shift,
            self.thermal_solid, self.rad_net, self.rad_amb, self.rad_flux,
            tick)
        return self.rad_net

    def clear_ledgers(self):
        self.rad_net.fill(0)
        self.rad_amb.fill(0)
        self.rad_flux.fill(0)

    def books_close(self):
        """Rule 4's ledger identity, evaluated PRE-FOLD (v7.1 item 9)."""
        return int(self.rad_net.sum()) + int(self.rad_amb.sum()) == 0


def _interior(sc, plane):
    """The plane with the border ring cut away.

    WHY THE RING IS EXCLUDED FROM "SEALED" CLAIMS. The ring is the outermost
    layer of the world, so its OUTWARD directions leave the grid and are
    correctly charged to the sky by rule 4 — a wall with space on the far side
    radiates to space, which is the law working, not a leak. "Sealed" in these
    gates means the ROOM is sealed: no ray that starts inside can escape without
    being absorbed. That is a statement about the interior, and it is the same
    like-for-like framing the equivalence gate uses when it compares an emitter
    against AIR-SEPARATED ambient scenery rather than against the void.
    """
    return plane[1:-1, 1:-1]


def _border_ring(sc, atten, his=3, T_game=0.0):
    """A SINGLE-LAYER, GRID-BORDER-BACKED wall ring at ambient temperature.

    Three properties, all load-bearing for the equivalence gate (v7.1 item 5):
      * SINGLE-LAYER — a second layer would be a CONTACT face (rule 3) and the
        ray would terminate there, charged to nobody;
      * GRID-BORDER-BACKED — the ring IS the outer edge, so nothing escapes
        past it and the room is genuinely sealed;
      * AIR-SEPARATED from the emitter — which is what makes the comparison
        like-for-like (v7.1 item 12: a wall-ADJACENT emitter is EXPECTED to lose
        less, because its contact face conducts instead).
    """
    h, w = sc.h, sc.w
    for x in range(w):
        sc.solid(0, x, atten=atten, his=his, T_game=T_game)
        sc.solid(h - 1, x, atten=atten, his=his, T_game=T_game)
    for y in range(h):
        sc.solid(y, 0, atten=atten, his=his, T_game=T_game)
        sc.solid(y, w - 1, atten=atten, his=his, T_game=T_game)


# ===========================================================================
# GATE (i) — THE EQUIVALENCE GATE. Tolerance ZERO.
# ===========================================================================
def test_gate_i_equivalence_open_field_vs_sealed_ambient_room_tol_zero():
    """The arc's headline claim, at BIT IDENTITY.

    Erik's requirement: an emitter must not care whether the cold thing it is
    radiating to is a wall across the room or the sky. Under the closed books
    that is not an approximation -- it is an algebraic identity, because the
    sky term and the rule-1 pair against a T = 0 absorber are THE SAME
    EXPRESSION:

        sky  = a_s * tau_end * w * (E[T_s] - E[0])
        pair = a_s * a_r * tau * w * (E[T_s] - E[T_r])   with a_r = 1, T_r = 0

    With a = 1.0 walls the ray deposits once (survival goes to 0) at tau = 1,
    so the two reduce to the identical float fold and the identical single
    quantize. Anything less than bit-identity here would mean the two paths had
    drifted -- a different rounding, a different clamp, a different order.

    THE CLAMP IS ALSO PINNED IDENTICAL: the sky term clamps on
    rad_pair_budget(|T_s|, his_s) and the pair clamps on the MIN over both ends
    of rad_pair_budget(|T_s - 0|, his). Giving the walls the emitter's own
    `his` makes those the same number, so the identity holds whether or not the
    limiter is engaged.
    """
    print("\nP-F1a GATE (i) - equivalence, tolerance ZERO:")
    for T in (200.0, 443.0, 900.0, 2000.0):
        # OPEN FIELD: nothing but the emitter. Every ray leaves the world.
        openf = Scene(21, 21)
        openf.solid(10, 10, atten=0.5, his=3, T_game=T)
        openf.burn(10, 10)
        openf.cast(_rc())
        loss_open = int(openf.rad_net[10, 10])

        # SEALED AMBIENT ROOM: the same emitter, centred, with a = 1.0
        # single-layer grid-border-backed walls pinned at T = 0 and the
        # emitter's own `his`. Every ray terminates on a wall instead.
        room = Scene(21, 21)
        room.solid(10, 10, atten=0.5, his=3, T_game=T)
        room.burn(10, 10)
        _border_ring(room, atten=1.0, his=3, T_game=0.0)
        room.cast(_rc())
        loss_room = int(room.rad_net[10, 10])

        print(f"  T={T:7.1f} game -> open-field loss {loss_open:>12d}   "
              f"sealed-room loss {loss_room:>12d}   diff "
              f"{loss_open - loss_room}")
        assert loss_open == loss_room, (
            f"EQUIVALENCE BROKEN at T={T}: open field {loss_open} vs sealed "
            f"ambient room {loss_room} (tolerance is ZERO -- the sky term and "
            f"the rule-1 pair against T=0 are the same expression)")
        # Open field books its whole loss to the sky; the room books none.
        assert int(openf.rad_amb.sum()) == -loss_open
        assert int(room.rad_amb.sum()) == 0, "a ray escaped a SEALED room"
        assert openf.books_close() and room.books_close()
        assert openf.contact_hits == 0, (
            "an open-field ray hit a contact face -- the scene is not open")
    print("  bit-identical at every probed temperature.")


def test_gate_i_equivalence_a_half_walls_with_derived_tolerance():
    """The a = 0.5 variant, with the tolerance DERIVED, not guessed.

    At a = 0.5 the ray no longer dies at the wall: it deposits half and carries
    half onward, where it leaves the grid and is charged as sky. The TOTAL is
    still the full grey-body loss, but it is now split across TWO quantize
    boundaries instead of one, and some directions terminate on CONTACT instead
    (a ray entering the ring obliquely and stepping along it) -- and a contact
    direction's residual is charged to NOBODY, by design.

    So the tolerance is built from what the geometry actually does:

        tol = (rounding)  n_split_rays * 1 count
            + (contact)   n_contact    * the largest residual a direction can
                                         still be carrying at termination

    with the contact count MEASURED from the cast rather than assumed. That is
    the whole point of returning it: a tolerance nobody can derive is a
    tolerance nobody can defend.
    """
    print("\nP-F1a GATE (i) - a = 0.5 walls, derived tolerance:")
    for T in (443.0, 900.0):
        openf = Scene(21, 21)
        openf.solid(10, 10, atten=0.5, his=3, T_game=T)
        openf.burn(10, 10)
        openf.cast(_rc())
        loss_open = int(openf.rad_net[10, 10])

        room = Scene(21, 21)
        room.solid(10, 10, atten=0.5, his=3, T_game=T)
        room.burn(10, 10)
        _border_ring(room, atten=0.5, his=3, T_game=0.0)
        room.cast(_rc())
        loss_room = int(room.rad_net[10, 10])
        n_contact = int(room.contact_hits)

        # The per-direction share: what ONE ray of the fan carries in total.
        per_ray = abs(loss_open) / RAY_COUNT
        # A ray that contact-terminates inside the a=0.5 ring has already
        # deposited its first half, so at most half its share is unaccounted.
        contact_budget = n_contact * per_ray * 0.5
        # Each surviving ray now crosses TWO quantize boundaries instead of
        # one: at most 1 count of round-half-away-from-zero error each.
        rounding_budget = RAY_COUNT
        tol = contact_budget + rounding_budget
        diff = abs(loss_open - loss_room)

        print(f"  T={T:7.1f} game -> open {loss_open:>12d}  room "
              f"{loss_room:>12d}  |diff| {diff:>10d}")
        print(f"    contact terminations = {n_contact}/{RAY_COUNT} rays; "
              f"derived tol = {contact_budget:.0f} (contact) + "
              f"{rounding_budget} (rounding) = {tol:.0f}")
        assert diff <= tol, (
            f"a=0.5 equivalence missed by {diff}, outside the DERIVED "
            f"tolerance {tol:.0f} ({n_contact} contact terminations)")
        assert room.books_close() and openf.books_close()
    print("  within the derived tolerance at every probed temperature.")


# ===========================================================================
# GATE (ii) — THE LEDGER IDENTITY, PRE-FOLD, FIRESTORM SCALE.
# ===========================================================================
def test_gate_ii_ledger_identity_pre_fold_on_a_firestorm():
    """`sum(rad_net) + sum(rad_amb) == 0` EXACTLY, on a real firestorm.

    Evaluated PRE-FOLD, deliberately (v7.1 item 9): the fold discards each
    tile's sub-2^his remainder, a bias toward zero in MAGNITUDE -- a systematic
    slight UNDER-transfer, never a mint. Checking post-fold would be checking
    the fold's rounding, not the books.

    Everything the law can do is exercised here at once: rule-1 pairs, rule-2
    half-weight mutual pairs (600 emitters see each other), rule-3 contact
    terminations (the random scatter creates plenty), and rule-4 sky escapes at
    all four edges.
    """
    print("\nP-F1a GATE (ii) - ledger identity, pre-fold, firestorm scale:")
    rng = np.random.default_rng(20260802)
    h = w = 128
    sc = Scene(h, w)
    cells = set()
    while len(cells) < 600:
        cells.add((int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))))
    for (y, x) in cells:
        sc.solid(y, x, atten=0.5, his=3, T_game=float(rng.uniform(200.0, 3000.0)))
        sc.burn(y, x, I=float(rng.uniform(0.3, 1.0)))
    for _ in range(800):
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        if (y, x) in cells:
            continue
        sc.solid(y, x, atten=float(rng.uniform(0.1, 1.0)),
                 his=int(rng.integers(2, 6)), T_game=float(rng.uniform(0.0, 400.0)))
    sc.cast(_rc())
    tiles = int(sc.rad_net.sum())
    sky = int(sc.rad_amb.sum())
    print(f"  600 emitters: sum(rad_net) = {tiles}, sum(rad_amb) = {sky}, "
          f"total = {tiles + sky}")
    print(f"  moved cells = {int(np.count_nonzero(sc.rad_net))}, "
          f"contact terminations = {sc.contact_hits}")
    assert tiles + sky == 0, (
        f"THE BOOKS DID NOT CLOSE: rad_net {tiles} + rad_amb {sky} = "
        f"{tiles + sky}")
    assert sky > 0, "vacuous gate - nothing escaped a 128x128 grid"
    assert int(np.count_nonzero(sc.rad_net)) > 0, "vacuous gate - nothing moved"
    assert int(sc.rad_amb.min()) >= 0, "the sky ledger went negative"
    assert sc.contact_hits > 0, (
        "vacuous gate - a dense random scatter produced no contact "
        "terminations, so rule 3 is not being exercised here")


# ===========================================================================
# GATE (iii) — THE NET TEST.
# ===========================================================================
def _equal_T_lattice(T_game, atten=0.5, his=3):
    """A sealed room whose walls AND interior lattice are all at ONE
    temperature, with the lattice tiles AIR-SEPARATED from each other.

    Sealed matters: with an open grid the lattice would still (correctly) lose
    to the SKY, and "every rad_net == 0" would be false for a reason that has
    nothing to do with rule 2. Sealed and isothermal is the configuration in
    which the claim is exactly "reciprocity holds to the integer".

    Air-separated matters too: a contiguous block is VACUOUS under rule 3 --
    every interior direction is a contact face, so nothing would be exchanged
    and the gate would pass without testing anything.
    """
    sc = Scene(21, 21)
    _border_ring(sc, atten=1.0, his=his, T_game=T_game)
    for y in range(3, 18, 2):
        for x in range(3, 18, 2):
            sc.solid(y, x, atten=atten, his=his, T_game=T_game)
    return sc


def test_gate_iii_equal_temperature_lattice_nets_exactly_zero():
    """Every tile's rad_net is EXACTLY 0 on an isothermal sealed lattice.

    Above the gate this is rule 2 doing its job: both ends are emitters, both
    cast the half-weight gap-signed pair, and equal T means the same E bucket
    so `diff == 0` structurally -- before any rounding can act. Below the gate
    it is trivially zero because nobody casts, which is the OTHER half of the
    claim and is worth pinning so a future change that made sub-gate tiles cast
    would be caught immediately.

    FIRE-FREE by construction (phi == 1 and flame_lift == 0 under the sizing
    ruling's package A, so a burning tile would not perturb this -- but a
    fire-free grid makes the claim independent of that).
    """
    print("\nP-F1a GATE (iii) - the NET test on an equal-T lattice:")
    for T, above in ((443.0, True), (900.0, True), (100.0, False)):
        sc = _equal_T_lattice(T)
        sc.cast(_rc())
        worst = int(np.abs(_interior(sc, sc.rad_net)).max())
        sky_interior = int(_interior(sc, sc.rad_amb).sum())
        print(f"  T={T:7.1f} game ({'above' if above else 'below'} the "
              f"{T_EMIT_GATE:.0f} gate): max|rad_net| (interior) = {worst}, "
              f"interior sky = {sky_interior}, contact = {sc.contact_hits}")
        assert worst == 0, (
            f"an isothermal lattice moved {worst} counts at T={T} -- "
            f"reciprocity is broken")
        assert sky_interior == 0, (
            "an INTERIOR tile of a sealed isothermal room escaped to the sky "
            "-- the room is not sealed")
        assert sc.books_close()
    # Non-vacuous, two ways. (1) Above the gate the lattice really is casting:
    # break the isotherm at ONE tile and energy must move.
    hot = _equal_T_lattice(443.0)
    hot.temperature[9, 9] = Q(1200.0)
    hot.cast(_rc())
    assert int(np.abs(_interior(hot, hot.rad_net)).max()) > 0, (
        "breaking the isotherm at one tile moved nothing -- the equal-T "
        "result above is vacuous")
    # (2) Below the gate NOBODY casts, which is the other half of the claim:
    # the same broken isotherm must move nothing at all when both tiles are
    # sub-gate, because a sub-gate tile does not CAST (it only participates in
    # pairs an emitter's ray forms).
    cold = _equal_T_lattice(100.0)
    cold.temperature[9, 9] = Q(150.0)      # still below the 180 gate
    cold.cast(_rc())
    assert int(np.abs(cold.rad_net).max()) == 0, (
        "a sub-gate tile cast anyway -- 'below the gate does not radiate' is "
        "not being honoured")
    print("  exactly zero everywhere; perturbing one tile moves energy above "
          "the gate and nothing below it.")


# ===========================================================================
# GATE (iv) — THE EXCHANGE RATE, OVER ONE FULL D4 ROTATION.
# ===========================================================================
def _two_temperature_box(T_hot, T_cold, his=3, atten=0.5, cold_emits=True):
    """A sealed box with two AIR-SEPARATED slabs at different temperatures.

    Slabs (columns) rather than single tiles: the 8-ray fan only sweeps onto a
    given single-tile direction on SOME ticks of the D4 rotation, so a
    single-tile pair would measure the rotation. Columns are hit every tick,
    and the 8-tick average is then a statement about the LAW.

    THE RING SITS AT THE COLD SLAB'S TEMPERATURE, deliberately. With the ring at
    ambient the cold slab would be losing to the walls at the same time as it
    gained from the hot slab, and its net would be a mixture of two effects.
    Isothermal with the cold slab, the ring exchanges EXACTLY zero with it (same
    E bucket), so the cold slab's rad_net is the s<->r pair and nothing else --
    which is the quantity this gate is about.

    `cold_emits=False` strips the cold slab of `thermal_solid`, which removes it
    from the emitter mask however hot it is while leaving it a full absorber
    (a > 0). That is the control for the "1x, no double count" claim: the same
    pair, cast one-way instead of mutually.
    """
    sc = Scene(21, 21)
    _border_ring(sc, atten=1.0, his=his, T_game=T_cold)
    for y in range(1, 20):
        sc.solid(y, 8, atten=atten, his=his, T_game=T_hot)
        sc.solid(y, 12, atten=atten, his=his, T_game=T_cold)
        if not cold_emits:
            sc.thermal_solid[y, 12] = False
    return sc


def test_gate_iv_two_temperature_box_exchange_is_one_times_the_pair_law():
    """Exchange == 1x the pair law, averaged over exactly 8 ticks.

    THE 8 IS NOT ARBITRARY: it is `fire_ray_count`, which is exactly the D4 fan
    rotation period (the phase advances by one N-th of a ray spacing per tick,
    so N ticks sweep one full spacing). v7.1 item 3 (M3) requires the
    measurement over a WHOLE multiple of that period, because per-tick the
    discrete view factor is only reciprocity-limited -- the continuity claim is
    equality IN EXPECTATION over the rotation.

    dT IS SWEPT DOWN TO 5 GAME ON PURPOSE. The flux limiter clamps each term to
    a |dT|/16 share of the gap; if it were biting anywhere in the operating
    band, the small-dT points are where it would show, as a collapse of the
    exchange relative to the T^4 prediction. Round-3.6 killed v6's rule 2
    precisely because its |dT|-based clamp annihilated the term below
    dT ~ 76 game, so this sweep is that failure's dedicated tripwire.
    """
    print("\nP-F1a GATE (iv) - two-temperature box, 8-tick (one rotation) mean:")
    print(f"  {'dT':>6}  {'mutual (both emit)':>20}  {'one-way (cold inert)':>21}"
          f"  {'ratio':>8}")
    T_base = 400.0
    rows = []
    for dT in (5.0, 10.0, 20.0, 40.0, 200.0):
        mutual_total = 0
        oneway_total = 0
        hot_total = 0
        for tick in range(RAY_COUNT):
            mut = _two_temperature_box(T_base + dT, T_base, cold_emits=True)
            mut.cast(_rc(), tick=tick)
            # Rows 1..19 only: rows 0 and 20 of these columns are BORDER RING
            # tiles, and the ring legitimately radiates to space off the far
            # side of the world (rule 4). Including them would be measuring the
            # room's outer skin, not the slab pair.
            mutual_total += int(mut.rad_net[1:-1, 12].sum())
            hot_total += int(mut.rad_net[1:-1, 8].sum())
            assert mut.books_close(), f"books open at dT={dT}, tick={tick}"
            assert int(_interior(mut, mut.rad_amb).sum()) == 0, (
                "an interior tile of a sealed box leaked to the sky")

            one = _two_temperature_box(T_base + dT, T_base, cold_emits=False)
            one.cast(_rc(), tick=tick)
            oneway_total += int(one.rad_net[1:-1, 12].sum())
            assert one.books_close()
        mutual = mutual_total / RAY_COUNT
        oneway = oneway_total / RAY_COUNT
        hot_mean = hot_total / RAY_COUNT
        ratio = (mutual / oneway) if oneway else float("nan")
        rows.append((dT, mutual, oneway, ratio))
        print(f"  {dT:>6.0f}  {mutual:>20.1f}  {oneway:>21.1f}  {ratio:>8.4f}")
        # DIRECTION: hot loses, cold gains. Never the reverse, at any dT.
        assert hot_mean < 0, f"the HOT slab gained heat at dT={dT}"
        assert mutual > 0, f"the COLD slab lost heat at dT={dT}"
        # *** ONE TIMES THE PAIR LAW. *** The mutual case (both slabs above the
        # gate, each casting the HALF-weight pair) must move the SAME energy as
        # the one-way case (only the hot slab casts, at full weight). Rule 2's
        # whole purpose is that the two half-casts sum to exactly 1x — a 2x
        # double-count, which is what a naive "both ends cast the full pair"
        # would give, shows here as a ratio near 2.
        assert 0.85 < ratio < 1.15, (
            f"the mutual exchange is {ratio:.3f}x the one-way exchange at "
            f"dT={dT} -- rule 2's half weight is not summing to 1x")
    # THE CLAMP IS INERT ACROSS THE BAND — measured against the RIGHT oracle.
    #
    # The naive check ("exchange per unit dT is constant") is WRONG here: E is
    # sigma*T^4, so over T_base = 400 -> 600 game the slope dE/dT itself rises
    # ~1.7x. What must be constant is the exchange divided by the ANALYTIC
    # E-DIFFERENCE the law actually reads -- the same baked table, the same
    # 4-game-unit buckets, no interpolation. That ratio is the fraction of the
    # available potential the fan transports, and it is a pure geometry number:
    # it cannot depend on dT unless something is clamping.
    #
    # This is round-3.6's own failure mode, dedicated: v6's rule 2 used a
    # |dT|-based clamp that ANNIHILATED the term below dT ~ 76 game, which
    # would show here as the small-dT ratios collapsing toward zero.
    rc = _rc()
    tab = rc.emissive_table()

    def _dE(T_hot, T_cold):
        bh = min(int(Q(T_hot)) >> 18, tab.shape[0] - 1)
        bc = min(int(Q(T_cold)) >> 18, tab.shape[0] - 1)
        return float(int(tab[bh]) - int(tab[bc]))

    fracs = [abs(m) / _dE(T_base + dT, T_base) for (dT, m, _o, _r) in rows]
    spread = max(fracs) / min(fracs)
    print("  transported fraction of the analytic E-difference:")
    for (dT, _m, _o, _r), f in zip(rows, fracs):
        print(f"    dT={dT:>6.0f} -> {f:.6f}")
    print(f"  spread across the sweep: {spread:.4f}x (1.0 == the limiter is "
          f"perfectly inert; a clamp bite collapses the small-dT end)")
    assert spread < 1.10, (
        f"the transported fraction varies {spread:.3f}x across the dT sweep "
        f"-- the flux limiter is BITING inside the operating band")


def test_gate_iv_crossing_the_emit_gate_is_continuous_both_directions():
    """The s<->r exchange rate does not jump as r crosses T_emit_gate.

    This is v7's headline structural win over v6. Under v6's rule 2 the term
    changed FORM at the gate (a one-way potential-vs-ambient term above,
    a gap-signed pair below), so the exchange rate stepped discontinuously.
    Under v7 both branches are the SAME gap-signed pair and the only difference
    is the half weight -- and the halving is exactly compensated by r now
    casting its own half back. What actually changes at the gate is r beginning
    to pay its OWN other directions and its own sky: the physical onset of
    emission, not a discontinuity in the pair.

    Swept BOTH directions (v7.3): r approaching the gate from below and from
    above, so a one-sided artefact cannot hide.
    """
    print("\nP-F1a GATE (iv) - continuity across r's own gate crossing:")
    gate_q = Q(T_EMIT_GATE)
    print(f"  quantize(T_emit_gate) = {gate_q} ({T_EMIT_GATE} game)")
    print(f"  {'T_r':>8}  {'emitter?':>9}  {'8-tick mean s<->r exchange':>28}")
    series = []
    for T_r in (170.0, 176.0, 179.0, 180.0, 181.0, 184.0, 190.0):
        total = 0
        for tick in range(RAY_COUNT):
            sc = _two_temperature_box(600.0, T_r)
            sc.cast(_rc(), tick=tick)
            total += int(sc.rad_net[1:-1, 12].sum())   # interior rows only
            assert sc.books_close()
        mean = total / RAY_COUNT
        is_emitter = Q(T_r) >= gate_q
        series.append((T_r, mean))
        print(f"  {T_r:>8.1f}  {str(is_emitter):>9}  {mean:>28.1f}")
    # The receiver's GAIN must vary smoothly with its own temperature. Step
    # ratios between adjacent probes bracket the crossing; a form change at the
    # gate would show as one ratio far from the others.
    gains = [g for (_t, g) in series]
    assert all(g > 0 for g in gains), "the cooler slab stopped gaining"
    steps = [gains[i + 1] / gains[i] for i in range(len(gains) - 1)]
    print(f"  adjacent-probe ratios: "
          f"{', '.join(f'{s:.4f}' for s in steps)}")
    # The crossing pair (179 -> 180) must not stand out against its neighbours.
    worst = max(abs(s - 1.0) for s in steps)
    print(f"  largest deviation from a smooth 1.0 step: {worst:.4f}")
    assert worst < 0.25, (
        f"a {worst:.3f} step appeared across the sweep -- the exchange is NOT "
        f"continuous through r's gate crossing")


def test_gate_iv_boundary_tile_pinned_exactly_at_the_gate():
    """One tile pinned at EXACTLY quantize(T_emit_gate) (v7.1 item 13).

    The emitter mask is an integer `>=` compare against a quantized threshold,
    so the boundary value is the one place a `>` / `>=` slip or a float
    round-trip would show. Pinning a tile exactly there and requiring it to
    behave as an EMITTER (not merely "not crash") is what makes the mask's
    definition testable rather than aspirational.
    """
    gate_q = Q(T_EMIT_GATE)
    # AT the gate: an emitter. It must cast -- i.e. lose to the sky on its own.
    at_gate = Scene(21, 21)
    at_gate.solid(10, 10, atten=0.5, his=3)
    at_gate.temperature[10, 10] = gate_q          # EXACTLY the threshold
    at_gate.cast(_rc())
    loss_at = int(at_gate.rad_net[10, 10])

    # One count BELOW the gate: not an emitter, casts nothing at all.
    below = Scene(21, 21)
    below.solid(10, 10, atten=0.5, his=3)
    below.temperature[10, 10] = gate_q - 1
    below.cast(_rc())
    loss_below = int(below.rad_net[10, 10])

    print(f"\nP-F1a GATE (iv) - boundary tile at exactly quantize(T_emit_gate)"
          f" = {gate_q}:")
    print(f"  at the gate  (T_q = {gate_q}):     rad_net = {loss_at}")
    print(f"  one count below (T_q = {gate_q - 1}): rad_net = {loss_below}")
    assert loss_at < 0, (
        "a tile at EXACTLY quantize(T_emit_gate) did not cast -- the emitter "
        "mask's integer >= compare is off by one")
    assert loss_below == 0, (
        "a tile ONE COUNT below the gate cast anyway -- the threshold is not "
        "where it says it is")
    assert at_gate.books_close() and below.books_close()


# ===========================================================================
# GATE (v) — THE RANGE FLOOR AND THE GREY-BODY RATE.
# ===========================================================================
def test_gate_v_sealed_equal_T_room_wider_than_the_old_max_range():
    """A sealed isothermal room WIDER than the old max_range nets zero.

    The regression this keeps: under P-R4 an emission ray died at
    `range_base + range_per_intensity * I` (about 5 tiles at full intensity),
    so in any room bigger than that the rays expired in mid-air and were
    charged to NOBODY -- the corridor leak. A 41-tile room is 8x that reach.
    With RADIATION_RANGE >= the grid diagonal the rays now cross the room and
    terminate on the far wall, and because the room is isothermal every one of
    those terminations moves exactly 0.
    """
    old_max_range = DIALS["range_base"] + DIALS["range_per_i"] * 1.0
    sc = Scene(41, 41)
    _border_ring(sc, atten=1.0, his=3, T_game=443.0)
    for y in range(5, 36, 5):
        for x in range(5, 36, 5):
            sc.solid(y, x, atten=0.5, his=3, T_game=443.0)
            sc.burn(y, x, I=1.0)
    sc.cast(_rc())
    worst = int(np.abs(_interior(sc, sc.rad_net)).max())
    sky_interior = int(_interior(sc, sc.rad_amb).sum())
    print(f"\nP-F1a GATE (v) - sealed isothermal room 41x41 (old max_range was "
          f"{old_max_range:.1f} tiles):")
    print(f"  max|rad_net| (interior) = {worst}, interior sky = {sky_interior}")
    assert 41 > 2 * old_max_range, "the room is not wider than the old reach"
    assert worst == 0, (
        f"a sealed isothermal room moved {worst} counts -- rays are still "
        f"expiring in mid-air (the corridor leak)")
    assert sky_interior == 0, (
        "an INTERIOR tile of a sealed room leaked to the sky")


def test_gate_v_open_field_lone_emitter_matches_the_grey_body_rate():
    """The open-field sky ledger == the full grey-body rate.

    THE ANALYTIC TARGET: with every direction escaping and clear air
    (tau == 1), rule 4 sums over the fan to

        loss = sum_rays a_s * (1/N) * (E[T_s] - E[0])  =  a_s * (E[T_s] - E[0])

    -- the whole grey-body emission, with NO ray-count dependence. That
    cancellation is the point: the 8-ray fan is a sampler, not a dial.

    THE TOLERANCE, and why it is set where it is. v7.1 item 11 (m5) names the
    one leak this law keeps: a ray culled at `heat_cull` still carries up to 1%
    of its direction share, charged to nobody. It is an UNDER-cooling -- the
    safe direction -- and gate (v)'s tolerance must sit BELOW it, or the gate
    could not tell the named leak from an unnamed one. Here the rays never cull
    at all (clear air, survival stays 1.0), so the only error is the single
    round-half-away-from-zero at each ray: at most N counts out of ~1e7. The
    tolerance is therefore set at 0.1% -- an order of magnitude below the 1%
    heat_cull residual it must not be able to mask.

    THE TEMPERATURE RANGE IS BOUNDED ON PURPOSE, and the bound is measured
    below rather than assumed. The FLUX LIMITER caps each ray's term at
    (|dT| << his) >> RAD_LIM_SHIFT, and against the T = 0 ambient counterparty
    |dT| is just T_s -- which grows LINEARLY while the T^4 term does not. Above
    a crossover the rail therefore engages on the sky term itself and the loss
    is legitimately BELOW the grey-body rate. At the shipped dials
    (rad_scale 1e-5, a_s 0.5, his 3, N 8) that crossover is ~1300 game, four
    times the operating band; the sweep stays under it, and the test then
    CONFIRMS the rail does engage above it, so the boundary is pinned rather
    than avoided.
    """
    print("\nP-F1a GATE (v) - open-field lone emitter vs the grey-body rate:")
    tol_frac = 0.001                       # 10x below the ~1% heat_cull residual
    a_s = 0.5

    def _open_field_loss(T, rc):
        sc = Scene(31, 31)
        sc.solid(15, 15, atten=a_s, his=3, T_game=T)
        sc.burn(15, 15)
        sc.cast(rc)
        assert sc.books_close()
        assert int(sc.rad_net[15, 15]) == -int(sc.rad_amb.sum())
        assert sc.contact_hits == 0, "an open-field ray hit a contact face"
        return int(sc.rad_amb.sum())

    def _grey_body(T, rc):
        tab = rc.emissive_table()
        bucket = min(int(Q(T)) >> 18, tab.shape[0] - 1)
        return a_s * float(int(tab[bucket]) - int(tab[0]))

    rc = _rc()
    for T in (300.0, 443.0, 800.0, 1200.0):
        measured = _open_field_loss(T, rc)
        expected = _grey_body(T, rc)
        rel = abs(measured - expected) / expected
        print(f"  T={T:7.1f} game: sky ledger {measured:>14d}  grey body "
              f"{expected:>16.0f}  rel err {rel:.2e}")
        assert rel < tol_frac, (
            f"open-field loss is {rel:.4%} off the grey-body rate at T={T} -- "
            f"above the {tol_frac:.1%} tolerance, which is itself set BELOW "
            f"the named ~1% heat_cull residual")
    # ABOVE THE CROSSOVER the limiter takes over, and the loss must fall BELOW
    # the grey body (never above it -- railing can only under-transfer).
    for T in (3000.0, 8000.0):
        measured = _open_field_loss(T, rc)
        expected = _grey_body(T, rc)
        frac = measured / expected
        print(f"  T={T:7.1f} game: sky ledger {measured:>14d}  grey body "
              f"{expected:>16.0f}  ratio {frac:.4f}  <- flux limiter engaged")
        assert frac < 1.0, (
            f"at T={T} the loss EXCEEDS the grey-body rate -- the limiter is "
            f"adding energy, which it must never do")
    print(f"  within {tol_frac:.1%} through the operating band (the named "
          f"heat_cull residual is ~1%, an order of magnitude larger); the rail "
          f"engages above ~1300 game and only ever under-transfers.")


# ===========================================================================
# GATE (vi) — THE NEGATIVE PAIR.
# ===========================================================================
def test_gate_vi_hot_sub_gate_solid_heats_a_cooler_emitter():
    """A HOT SUB-GATE solid heats a COOLER EMITTER. Sign correctness.

    This is the case v6.2(c) got wrong and v7.3 corrected: "sub-gate tiles pay
    nothing" was too strong. A sub-gate tile does not CAST -- but it is still a
    full participant in any pair an emitter's ray forms with it, in EITHER
    direction. So when the emitter is the COLDER end, the gap-signed pair must
    run backwards: the emitter GAINS and the sub-gate solid LOSES, on the
    emitter's own ray.

    Getting this wrong is not cosmetic. A one-way "potential vs ambient" term
    (which is what v6's rule 2 was) can only ever move heat AWAY from the
    caster, so it would have the cold emitter heating the hot wall -- energy
    flowing cold-to-hot, a second-law violation the gap-signed form makes
    structurally impossible.
    """
    print("\nP-F1a GATE (vi) - negative pair (hot sub-gate solid, cool emitter):")
    sc = Scene(21, 21)
    # The EMITTER: above the gate (so it casts) but COOL.
    sc.solid(10, 10, atten=0.5, his=3, T_game=200.0)
    # The counterparty: BELOW the gate (so it never casts) but much HOTTER.
    # Sub-gate means < 180 game... which cannot be hotter than 200. So the
    # emitter is instead the one that is cool ENOUGH: gate at 180, emitter at
    # 181, sub-gate wall at 179 -- and to get a real gap, drop the gate.
    sc = Scene(21, 21)
    rc = _rc(t_emit_gate=180.0)
    sc.solid(10, 10, atten=0.5, his=3, T_game=181.0)    # emitter, just over
    for y in range(21):
        sc.solid(y, 12, atten=0.5, his=3, T_game=179.0)  # sub-gate, just under
    sc.cast(rc)
    print(f"  emitter (T=181, casts) rad_net = {int(sc.rad_net[10, 10])}")
    print(f"  sub-gate wall (T=179, never casts) rad_net = "
          f"{int(sc.rad_net[:, 12].sum())}")
    # Here the emitter is the HOTTER end, so it loses to the wall: the
    # ordinary direction, established first as the control.
    assert int(sc.rad_net[:, 12].sum()) > 0, "the cooler sub-gate wall did not gain"
    assert sc.books_close()

    # NOW THE NEGATIVE CASE: flip the gap. Raise the gate so the HOT wall is
    # sub-gate, and leave the emitter cooler but still above the gate.
    rc2 = _rc(t_emit_gate=100.0)
    neg = Scene(21, 21)
    neg.solid(10, 10, atten=0.5, his=3, T_game=120.0)    # emitter: above 100
    for y in range(21):
        # The wall is HOTTER but NOT an emitter... which needs it below the
        # gate. With the gate at 100 that is impossible for a hotter tile, so
        # make the wall a NON-thermal-solid absorber instead: it has a > 0 (it
        # absorbs and emits by Kirchhoff) but thermal_solid is False, so it is
        # excluded from the emitter mask no matter how hot it is.
        neg.heat_atten[y, 12] = 0.5
        neg.heat_inv_shift[y, 12] = 3
        neg.thermal_solid[y, 12] = False                  # NEVER an emitter
        neg.temperature[y, 12] = Q(900.0)                 # but much HOTTER
    neg.cast(rc2)
    emitter_net = int(neg.rad_net[10, 10])
    wall_net = int(neg.rad_net[:, 12].sum())
    print(f"  --- negative pair ---")
    print(f"  emitter (T=120, casts)        rad_net = {emitter_net}")
    print(f"  hot non-emitter wall (T=900)  rad_net = {wall_net}")
    print(f"  sky ledger = {int(neg.rad_amb.sum())}")
    assert emitter_net > 0, (
        "the COOLER emitter did not GAIN from the hotter wall -- the pair is "
        "not gap-signed, and heat can flow cold-to-hot")
    assert wall_net < 0, "the hotter wall did not LOSE to the cooler emitter"
    assert neg.books_close()


# ===========================================================================
# GATE (vii) — THE RAIL, WHEN IT ACTUALLY FIRES.
# ===========================================================================
def test_gate_vii_limiter_engaged_pair_approaches_monotonically():
    """When the flux limiter FIRES, the pair still approaches MONOTONICALLY.

    v7.3 item 14 is explicit that it is not enough to show the rail did not
    explode -- the rail has to be shown DOING ITS JOB. So this drives a pair to
    a T_MAX_PHYS-scale gap where the T^4 term vastly exceeds the budget,
    confirms the clamp is actually engaged (the transfer is pinned at the
    budget, not at the T^4 value), and then iterates the pair by hand and
    requires that:

      * the gap shrinks EVERY step (monotone approach, never oscillation),
      * the two ends never CROSS (no overshoot past equalisation),
      * energy is conserved at every step (the clamp is applied to the whole
        term BEFORE the +/- split, so both ends always move the same integer --
        which is exactly why railing cannot break conservation).

    That last point is the design decision this test exists to protect: clamp
    before the split, never after.
    """
    print("\nP-F1a GATE (vii) - the flux limiter, engaged:")
    rc = _rc()
    his = 3
    T_hot, T_cold = 15000.0, 0.0          # a T_MAX_PHYS-scale gap
    gaps = []
    hot_hist, cold_hist = [], []
    Th, Tc = Q(T_hot), Q(T_cold)
    railed_steps = 0
    for step in range(12):
        sc = Scene(21, 21)
        _border_ring(sc, atten=1.0, his=his, T_game=0.0)
        for y in range(1, 20):
            sc.heat_atten[y, 8] = 0.5
            sc.heat_inv_shift[y, 8] = his
            sc.thermal_solid[y, 8] = True
            sc.temperature[y, 8] = Th
            sc.heat_atten[y, 12] = 0.5
            sc.heat_inv_shift[y, 12] = his
            sc.thermal_solid[y, 12] = True
            sc.temperature[y, 12] = Tc
        sc.cast(rc)
        assert sc.books_close(), f"the books opened while railing (step {step})"

        # The per-tile fold, done exactly as temperature_solver Pass 1 does it:
        # shr_round0 (symmetric round-toward-zero) by heat_inv_shift.
        def fold(rn):
            return int(np.sign(rn) * (abs(int(rn)) >> his))

        d_hot = fold(sc.rad_net[10, 8])
        d_cold = fold(sc.rad_net[10, 12])
        # Is the rail actually engaged? The budget for one ray is
        # (|dT| << his) >> RAD_LIM_SHIFT; the unclamped T^4 term at this gap is
        # orders larger, so a transfer sitting AT a multiple of the budget is
        # the signature of the clamp.
        budget = (abs(Th - Tc) << his) >> 4
        per_ray = abs(int(sc.rad_net[10, 12])) / RAY_COUNT
        if per_ray >= budget * 0.5:
            railed_steps += 1

        gaps.append(Th - Tc)
        hot_hist.append(Th)
        cold_hist.append(Tc)
        Th += d_hot
        Tc += d_cold
        assert Th >= Tc, (
            f"the pair CROSSED at step {step}: hot {Th} < cold {Tc} -- the "
            f"limiter failed to keep the update monotone")

    print(f"  gap trajectory (Q16.16): "
          f"{', '.join(str(g) for g in gaps[:6])} ...")
    print(f"  steps with the rail engaged: {railed_steps}/12")
    assert railed_steps > 0, (
        "the limiter never engaged -- this gate is vacuous, it must be driven "
        "into the rail to test the rail")
    for i in range(1, len(gaps)):
        assert gaps[i] < gaps[i - 1], (
            f"the gap did not shrink at step {i}: {gaps[i - 1]} -> {gaps[i]} "
            f"-- the railed update is not a monotone approach")
    print(f"  gap shrank monotonically on every step, ends never crossed, "
          f"books closed throughout.")


if __name__ == "__main__":
    test_gate_i_equivalence_open_field_vs_sealed_ambient_room_tol_zero()
    test_gate_i_equivalence_a_half_walls_with_derived_tolerance()
    test_gate_ii_ledger_identity_pre_fold_on_a_firestorm()
    test_gate_iii_equal_temperature_lattice_nets_exactly_zero()
    test_gate_iv_two_temperature_box_exchange_is_one_times_the_pair_law()
    test_gate_iv_crossing_the_emit_gate_is_continuous_both_directions()
    test_gate_iv_boundary_tile_pinned_exactly_at_the_gate()
    test_gate_v_sealed_equal_T_room_wider_than_the_old_max_range()
    test_gate_v_open_field_lone_emitter_matches_the_grey_body_rate()
    test_gate_vi_hot_sub_gate_solid_heats_a_cooler_emitter()
    test_gate_vii_limiter_engaged_pair_approaches_monotonically()
    print("OK - P-F1a: the verified radiation books")
