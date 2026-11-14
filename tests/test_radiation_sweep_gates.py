"""GATES 1-6 ON THE ENGINE'S OUTPUT (ray-engine-v2 P1, design v3 §2.9).

The eight property gates the shadow sweep must pass, run against the C++
sweep (cpp/src/radiation_sweep.cpp) and — for gates 4 and 5 — the C++ Pass-1
fold with the maximum-principle clamp (cpp/src/temperature_solver.cpp, via the
direct TemperatureSolver.step binding, the tests/test_temperature_convert.py
precedent). Each gate is written exactly as sweep_ref_q_gates.py does it and
is NON-VACUOUS the same way; where the reference models the fold, the C++
fold's trajectory and counters are ALSO compared to the reference's, tick for
tick, so gate 0's bit-for-bit claim extends across the clamp.

Gate 4 and 5 note: the fold's `rad_net` parameter is int64 since P3a-1 (the
widening, design rows 26/35), so the sweep's int64 rad_net goes to the fold AS
IT STANDS after the HELD source cell is zeroed (the reference's `held` mask: a
source pinned by another heater takes no radiative update). The P1 narrowing is
gone; the assertion that it WOULD have been exact stays, because it is the
proof that removing it moved no number in these gates.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_radiation_sweep_gates.py -q
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from _radiation_sweep_harness import (  # noqa: E402  (sets up sys.path: build dir + study)
    F_ONE, ONE, R, T_AMB_Q, cpp_sweep, ref_sweep, reference_table, ts_from_a)
import breach_physics as bp  # noqa: E402
import sweep_ref_q_gates as G  # noqa: E402  (the reference's own scene builders)

Q = R.quant
K_LEAK = Q(0.10)
INT32_MAX = 2 ** 31 - 1
T_SRC = G.T_SRC_GAME
A_FURN = G.A_FURNITURE
HIS_FURN = G.HIS_FURNITURE


# --------------------------------------------------------------------------- #
# G1 conservation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("k_q", [0, K_LEAK])
@pytest.mark.parametrize("transport", ["shear", "step"])
def test_g1_three_term_identity_is_exact_with_each_sum_nonzero(transport, k_q, n_ord):
    """G1: Σ rad_net + Σ rad_flux + Σ rad_amb == 0 exactly in int64 on a
    randomised scene with bodies (d > a), leak on and off, both transports,
    S16 and S12 — and each of the three sums is individually non-zero.

    BREAKS IF: any writer books one side of a transfer without the other (the
    remainder split, the body's re-emission, either of the ring's two books).
    """
    rng = random.Random(20260915 + n_ord + k_q)
    h, w = 9, 11
    a, d, T, _f = G._rand_scene(rng, h, w)
    rn, rf, ra, _rl, _fl, _sw = cpp_sweep(a, d, k_q, T, 3, ts_from_a(a),
                                          transport=transport, n_ord=n_ord)
    sn, sf, sa = int(rn.sum()), int(rf.sum()), int(ra.sum())
    assert sn + sf + sa == 0, (sn, sf, sa)
    assert sn != 0 and sf != 0 and sa != 0, "a term is vacuous"


# --------------------------------------------------------------------------- #
# G2a uniform ambient
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("transport", ["shear", "step"])
def test_g2a_uniform_ambient_is_a_per_cell_exact_fixed_point_with_bodies_and_leak(transport, n_ord):
    """G2a: a uniform ambient field (T == 0 everywhere) leaves ALL THREE ledger
    planes exactly zero at EVERY cell — with non-dyadic a, bodies present, the
    leak on, both transports, S16 and S12. Holds for any Fleck factor by
    construction (the excess is zero at ambient).

    BREAKS IF: the body stops re-emitting at ambient (row 25), the ceiling
    return loses its amb_m-first association, the ring stops returning amb_m,
    or the Fleck factor multiplies the whole emission instead of the excess.
    NON-VACUITY: the same scene with a hot cell is NOT a fixed point.
    """
    rng = random.Random(4)
    h, w = 8, 10
    a = [[rng.choice([0, Q(0.37), Q(0.91), ONE]) for _ in range(w)] for _ in range(h)]
    d = [[min(ONE, a[y][x] + rng.choice([0, Q(0.5), ONE - a[y][x]]))
          for x in range(w)] for y in range(h)]
    n_bodies = sum(1 for y in range(h) for x in range(w) if d[y][x] > a[y][x])
    assert n_bodies > 0
    T = R.plane(h, w, 0)
    rn, rf, ra, rl, _fl, _sw = cpp_sweep(a, d, K_LEAK, T, 3, ts_from_a(a),
                                         transport=transport, n_ord=n_ord)
    nz = int(np.count_nonzero(rn) + np.count_nonzero(rf) + np.count_nonzero(ra))
    assert nz == 0, f"{nz} non-zero ledger cells on a uniform ambient field"
    assert np.all(rl > 0), "the fluence plane must carry the ambient stream"
    # NON-VACUITY: one hot cell breaks the fixed point on all three planes.
    T[3][3] = T_SRC << 16
    rn2, rf2, ra2, _rl2, _fl2, _sw2 = cpp_sweep(a, d, K_LEAK, T, 3, ts_from_a(a),
                                                transport=transport, n_ord=n_ord)
    assert np.any(rn2 != 0) and np.any(rf2 != 0) and np.any(ra2 != 0)


# --------------------------------------------------------------------------- #
# G2b the enclosed isothermal box
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("transport", ["shear", "step"])
def test_g2b_enclosed_isothermal_box_inner_layer_is_exactly_zero_with_f_below_one(transport, n_ord):
    """G2b: 2-cell-thick opaque walls at T0 = 1263 game around a transparent
    interior — the INNER wall layer is a per-cell exact zero and the interior
    air is exactly zero, while the outer layer cools to the sky (min < 0). The
    engine's OWN Fleck factor is below 2^24 on every wall cell (asserted), so
    the property is exercised with f < 1, not with f == 1.

    BREAKS IF: emission and absorption stop being the same arithmetic (the
    (E·w)·a association), the split stops carrying the remainder, or f is
    applied per ordinate inconsistently.
    """
    n = 12
    a = R.plane(n, n, 0)
    d = R.plane(n, n, 0)
    T = R.plane(n, n, 0)
    for y in range(n):
        for x in range(n):
            if y < 2 or y >= n - 2 or x < 2 or x >= n - 2:
                a[y][x] = ONE
                d[y][x] = ONE
                T[y][x] = T_SRC << 16
    rn, _rf, _ra, _rl, fl, _sw = cpp_sweep(a, d, 0, T, 3, ts_from_a(a),
                                           transport=transport, n_ord=n_ord)
    wall = np.asarray(a) == ONE
    assert np.all(fl[wall] < F_ONE) and np.all(fl[wall] > 0), "f must be < 1 on the walls"
    inner = [rn[y, x] for y in range(n) for x in range(n)
             if (y in (1, n - 2) and 1 <= x <= n - 2) or (x in (1, n - 2) and 1 <= y <= n - 2)]
    outer = [rn[y, x] for y in range(n) for x in range(n) if y in (0, n - 1) or x in (0, n - 1)]
    interior = rn[2:n - 2, 2:n - 2]
    assert max(abs(int(v)) for v in inner) == 0, inner
    assert int(np.abs(interior).max()) == 0
    assert min(int(v) for v in outer) < 0, "the outer layer must lose to the sky"


# --------------------------------------------------------------------------- #
# G3 positivity + ingress rejections
# --------------------------------------------------------------------------- #
def test_g3_no_negative_stream_anywhere_and_illegal_extinction_is_rejected():
    """G3: the smallest per-ordinate stream over randomised hot scenes (bodies,
    leak on/off, both transports) is >= 0, the largest is > 0 (the scenes are
    not empty); and the materials door rejects heat_atten outside [0, 1] and
    heat_atten > 0 with thermal_mass == 0 (tests/test_optics_ingress.py owns
    the per-row rejections; the C++ door's rejections are in
    test_radiation_sweep_reference.py).

    BREAKS IF: absorption stops being bounded by a + b <= ONE (a negative
    stream), or the telemetry stops tracking the real minimum.
    """
    rng = random.Random(7)
    h, w = 9, 11
    worst_min = None
    biggest = 0
    for transport in ("shear", "step"):
        for k_q in (0, K_LEAK):
            a, d, T, _f = G._rand_scene(rng, h, w)
            _rn, _rf, _ra, rl, _fl, sw = cpp_sweep(a, d, k_q, T, 3, ts_from_a(a),
                                                   transport=transport)
            worst_min = sw.min_stream if worst_min is None else min(worst_min, sw.min_stream)
            biggest = max(biggest, sw.max_stream)
            assert np.all(rl >= 0)
    assert worst_min >= 0, worst_min
    assert biggest > 0


# --------------------------------------------------------------------------- #
# The fold, through the direct binding (gates 4 and 5)
# --------------------------------------------------------------------------- #
def _solver():
    """Conduction off (all-NO_FACE faces) and cooling off (shift 31 -> T >> 31
    == 0), so only Pass 1's radiation fold moves T — the
    tests/test_temperature_convert.py idiom."""
    from simulation.materials import MaterialTable
    tbl = MaterialTable.from_config()
    s = bp.TemperatureSolver()
    s.no_face = int(tbl.no_face)
    # T5b step 7 / R1: Pass 3 is deleted, so there is no cooling to disable.
    return s, int(tbl.no_face)


class _FoldScene:
    """A reference Scene mirrored into the C++ sweep + C++ fold, run tick by
    tick with the reference's Scene alongside so the two trajectories (T plane
    and counters) can be compared exactly."""

    def __init__(self, scene: "R.Scene"):
        self.ref = scene
        self.h, self.w = len(scene.a), len(scene.a[0])
        self.table = reference_table()
        self.sweep = bp.RadiationSweep()
        self.solver, self.no_face = _solver()
        self.T = np.ascontiguousarray(np.asarray(scene.T, dtype=np.int64).astype(np.int32))
        self.his = np.ascontiguousarray(np.asarray(R.plane(self.h, self.w, scene.his)
                                                   if isinstance(scene.his, int) else scene.his,
                                                   dtype=np.int32))
        self.ts = np.ascontiguousarray(np.asarray(scene.ts, dtype=np.int64) != 0)
        self.held = (np.asarray(scene.held, dtype=np.int64) != 0) if scene.held is not None \
            else np.zeros((self.h, self.w), dtype=bool)
        self.face = np.full((self.h, self.w, 4), self.no_face, dtype=np.int32)
        self.heat = np.zeros((self.h, self.w), dtype=np.int32)
        self.solid = np.ascontiguousarray(self.ts.copy())
        self.vac = np.zeros((self.h, self.w), dtype=bool)
        self.atm = np.full((self.h, self.w), 1 << 16, dtype=np.int32)
        self.max_abs_rad_net = 0
        self.last_rn = None
        self.last_rl = None

    def tick(self, *, clamp_enabled=True):
        sc = self.ref
        T_list = self.T.astype(np.int64).tolist()
        rn, _rf, _ra, rl, _fl, _sw = cpp_sweep(
            sc.a, sc.d, sc.k[0][0], T_list, self.his.tolist(), self.ts.tolist(),
            transport=sc.transport, table=self.table, sweep=self.sweep)
        rn = rn.copy()
        rn[self.held] = 0                          # a pinned source takes no update
        self.max_abs_rad_net = max(self.max_abs_rad_net, int(np.abs(rn).max()))
        # P3a-1: the fold takes int64 now, so `rn` goes in as it stands. The
        # assertion stays as the PROOF that the widening changed nothing here:
        # the P1 narrowing it guarded was exact, so removing it is a no-op on
        # every value these gates produce.
        assert int(np.abs(rn).max()) <= INT32_MAX, "the P1 int32 narrowing would have wrapped"
        rn64 = np.ascontiguousarray(rn)
        self.last_rn, self.last_rl = rn, rl
        before = self.T.copy()
        self.solver.step(self.T, self.heat, self.his, self.face, self.solid, self.vac,
                         self.atm, thermal_solid=self.ts, rad_net=rn64,
                         rad_fluence=np.ascontiguousarray(rl), e_table=self.table,
                         clamp_enabled=clamp_enabled)
        self.T[self.held] = before[self.held]      # belt and braces: held cells never move
        return before

    def counters(self):
        s = self.solver
        return (int(s.t_max_phys_hits), int(s.t_low_rail_hits), int(s.rad_clamp_hits))


def _run_both(scene_fn, ticks, *, clamp_enabled=True):
    """Run the reference Scene and its C++ mirror for `ticks`; assert the T
    plane and the three counters agree after EVERY tick; return both."""
    ref_scene, extra = scene_fn()
    cpp_scene = _FoldScene(ref_scene)
    for t in range(ticks):
        ref_scene.tick(clamp_enabled=clamp_enabled)
        cpp_scene.tick(clamp_enabled=clamp_enabled)
        T_ref = np.asarray(ref_scene.T, dtype=np.int64)
        assert np.array_equal(cpp_scene.T.astype(np.int64), T_ref), \
            f"tick {t}: the C++ fold's T plane differs from the reference's"
        c = ref_scene.counters
        assert cpp_scene.counters() == (c.t_max_phys_hits, c.t_low_rail_hits, c.rad_clamp_hits), \
            f"tick {t}: counters differ {cpp_scene.counters()} vs reference"
    return ref_scene, cpp_scene, extra


def _marine():
    return G._marine_scene("reemit"), None


def _overdriven():
    sc, crate = G._overdriven_scene()
    return sc, crate


# --------------------------------------------------------------------------- #
# G4 counters, per counter
# --------------------------------------------------------------------------- #
def test_g4_marine_beside_ambient_wall_leaves_every_counter_silent_for_24_ticks():
    """G4 (per counter): on the marine-beside-an-ambient-wall scene — a wall
    column at a = 0.9 and one opaque body one cell from it — t_low_rail_hits,
    rad_clamp_hits and t_max_phys_hits are ALL zero over 24 ticks through the
    C++ fold, the wall's rad_net is exactly zero, and the C++ trajectory
    equals the reference's tick for tick.

    BREAKS IF: the body stops re-emitting at ambient (a pure sink shadows the
    sky and trips the low rail on every wall cell every tick — the reference
    gate measures that variant), or the clamp engages on a cell it should not.
    NON-VACUITY: the same body one tile from a fire books a positive rad_flux.
    """
    ref_scene, cpp_scene, _ = _run_both(_marine, 24)
    assert cpp_scene.counters() == (0, 0, 0)
    wall = cpp_scene.last_rn[:, 6]
    assert np.all(wall == 0), wall.tolist()
    assert np.all(cpp_scene.T == 0)
    # positive: the re-emitting body still senses a fire
    h = w = 7
    a = R.plane(h, w, 0); d = R.plane(h, w, 0); T = R.plane(h, w, 0)
    a[3][1] = ONE; d[3][1] = ONE; T[3][1] = T_SRC << 16
    d[3][3] = ONE
    _rn, rf, _ra, _rl, _fl, _sw = cpp_sweep(a, d, 0, T, 3, ts_from_a(a))
    assert rf[3, 3] > 0


def test_g4_over_driven_scene_engages_the_clamp_and_not_the_rails():
    """G4 (per counter): a held 16000-game source one tile from a cold crate —
    rad_clamp_hits > 0 within 24 ticks, t_max_phys_hits == 0 (the clamp's
    ceiling saturates one LSB below 16000 game, below the rail) and
    t_low_rail_hits == 0, tick for tick equal to the reference -- and the crate
    is HELD AT its ceiling, e_ceiling_q(Φ) of the fluence it absorbs.

    P5d: the crate is the SHIPPED furniture row (sweep_ref_q_gates.
    HIS_OVERDRIVEN), which the source genuinely over-drives (Fleck-damped). The
    P0-era crate (thermal_mass 8) is undamped there since P2a's alpha floor 0:
    its clamp hits were the bucket-edge pinning P5d removes, and the headroom
    ceiling never clamps it (the reference's G4 measures it beside this scene).

    BREAKS IF: the clamp is dropped or placed after the rails (the rail would
    fire first), its ceiling saturates at or above T_MAX_PHYS, or the fold
    stops holding an over-driven cell at e_ceiling_q (e.g. e_inv_q again: the
    crate would sit two buckets lower).
    """
    ref_scene, cpp_scene, crate = _run_both(_overdriven, 24)
    t_max, low, clamp = cpp_scene.counters()
    assert clamp > 0 and t_max == 0 and low == 0, (t_max, low, clamp)
    cy, cx = crate
    t_crate = cpp_scene.T[cy, cx] / 65536.0
    ceiling = int(cpp_scene.table.e_ceiling_q(int(cpp_scene.last_rl[cy, cx])))
    print(f"\nG4 over-driven: clamp={clamp} t_max={t_max} low_rail={low} "
          f"crate T={t_crate:.3f} game after 24 ticks (its ceiling {ceiling / 65536:.3f}); "
          f"max|rad_net| narrowed = {cpp_scene.max_abs_rad_net}")
    assert int(cpp_scene.T[cy, cx]) == ceiling


# --------------------------------------------------------------------------- #
# G5 the maximum principle
# --------------------------------------------------------------------------- #
def test_g5_radiative_substep_never_exceeds_max_t_before_e_ceiling_phi():
    """G5: immediately after the Pass-1 radiative sub-step, on that sub-step
    alone, T_new <= max(T_before, e_ceiling_q(Φ)) for every non-held thermal
    solid over 24 ticks of the over-driven scene, evaluated from the direct
    binding with the sweep's own Φ; the clamp engages (hits > 0). P5d: the
    ceiling is the top of the first E° bucket out-emitting Φ (it was E°⁻¹(Φ),
    at most two buckets lower).

    BREAKS IF: the clamp is removed, written in the bare v2 form, evaluated
    after conduction/cooling (which would falsely fail), or its ceiling moves
    above the top of the first out-emitting bucket.
    """
    ref_scene, cpp_scene, crate = _run_both(_overdriven, 24)
    tbl = cpp_scene.table
    # re-run one more tick by hand to evaluate the property on the sub-step
    before = cpp_scene.tick()
    rl = cpp_scene.last_rl
    worst = None
    for y in range(cpp_scene.h):
        for x in range(cpp_scene.w):
            if not cpp_scene.ts[y, x] or cpp_scene.held[y, x]:
                continue
            cap = max(int(before[y, x]), int(tbl.e_ceiling_q(int(rl[y, x]))))
            ex = int(cpp_scene.T[y, x]) - cap
            worst = ex if worst is None else max(worst, ex)
    assert worst is not None and worst <= 0, worst
    assert cpp_scene.counters()[2] > 0


def test_g5_clamp_disabled_climbs_past_the_clamped_value_tick_for_tick_with_the_reference():
    """G5 non-vacuity (a): with `clamp_enabled=False` on the SAME over-driven
    2-D scene the crate climbs past the clamped value (the reference's own
    gate 5(a) assertion), rad_clamp_hits stays 0, and the C++ trajectory
    equals the reference's tick for tick over 96 ticks.

    P5d: the crate is the SHIPPED furniture row (Fleck-damped at its
    equilibrium), so this is a real over-drive: clamped, it is held at its
    ceiling (~1116 game); un-clamped, Fleck alone carries it to the T_MAX_PHYS
    rail within a few ticks. (Until P5d the crate was the P0-era thermal_mass-8
    row, which at the RULED alpha floor 0 is undamped there and SETTLED just
    above the clamped value -- 1108.0 vs 1113.6 game: the clamp was pinning it
    one bucket short of its own balance, which is what P5d removes.) The rail's
    reachability on the radiative branch with an INT32_MAX deposit is proved
    separately below. Both counters are printed.

    BREAKS IF: the binding keyword stops switching the clamp off, or the
    un-clamped fold drifts from the reference's.
    """
    ref_c, cpp_c, crate = _run_both(_overdriven, 24, clamp_enabled=True)
    ref_u, cpp_u, crate_u = _run_both(_overdriven, 96, clamp_enabled=False)
    cy, cx = crate
    t_clamped = int(cpp_c.T[cy, cx])
    t_unclamped = int(cpp_u.T[cy, cx])
    assert t_unclamped > t_clamped
    t_max, low, clamp = cpp_u.counters()
    assert clamp == 0, "clamp_enabled=False must not count clamp hits"
    assert low == 0
    print(f"\nG5 (a): clamped crate {t_clamped / 65536:.1f} game (24 ticks) vs un-clamped "
          f"{t_unclamped / 65536:.1f} game (96 ticks); t_max_phys_hits={t_max} on the "
          f"2-D scene (clamped it is held at its ceiling; Fleck alone rails it)")


def test_g5_t_max_phys_rail_is_reachable_on_the_radiative_branch_with_the_clamp_off():
    """G5 non-vacuity (a), the counter the design's gate 5 expects, where the P1
    fold CAN reach it: an INT32_MAX deposit (4096 game per tick through a wood
    tile's >> 3 — the largest the plane could carry before P3a-1 widened it,
    kept at that VALUE so the trajectory this gate pins does not move) on a cold
    crate with the clamp OFF rails the field at T_MAX_PHYS within four ticks and
    t_max_phys_hits counts every tick after; the SAME deposit with the clamp ON
    and a modest Φ (E°[100] = 400 game) is clipped at E°⁻¹(Φ) every tick,
    rad_clamp_hits counts it and the rail stays silent — the clamp binds first.
    Both runs equal the reference's fold_pass1_solid tick for tick. P5d: the
    clipped value is the clamp's ceiling e_ceiling_q(Φ) -- the top of the first
    bucket out-emitting Φ = E°[100], i.e. 408 game less one LSB (it was
    E°⁻¹(Φ) = 400 game).

    BREAKS IF: the clamp is placed after the rail, the rail stops counting, or
    the clamp's ceiling loses its fluence arm (or stops being e_ceiling_q).
    """
    from simulation.materials import MAT_WOOD, MaterialTable
    tbl = MaterialTable.from_config()
    table = reference_table()
    E = R.E
    for clamp_on, phi, expect in ((False, 0, "rail"), (True, int(E[100]), "clamp")):
        solver, no_face = _solver()
        h, w = 1, 3
        T = np.zeros((h, w), dtype=np.int32)
        heat = np.zeros((h, w), dtype=np.int32)
        his = np.full((h, w), int(tbl.heat_inv_shift[MAT_WOOD]), dtype=np.int32)
        face = np.full((h, w, 4), no_face, dtype=np.int32)
        ts = np.ones((h, w), dtype=bool)
        vac = np.zeros((h, w), dtype=bool)
        atm = np.full((h, w), 1 << 16, dtype=np.int32)
        rn = np.full((h, w), INT32_MAX, dtype=np.int64)   # P3a-1: same value, wider box
        rl = np.full((h, w), phi, dtype=np.int64)
        T_ref = [[0] * w for _ in range(h)]
        counters = R.FoldCounters()
        for tick in range(8):
            solver.step(T, heat, his, face, ts, vac, atm, thermal_solid=ts,
                        rad_net=rn, rad_fluence=rl, e_table=table, clamp_enabled=clamp_on)
            R.fold_pass1_solid(T_ref, [[INT32_MAX] * w], [[phi] * w],
                               int(tbl.heat_inv_shift[MAT_WOOD]), [[1] * w], counters,
                               clamp_enabled=clamp_on)
            assert T.astype(np.int64).tolist() == T_ref, tick
        t_max, low, clamp = (int(solver.t_max_phys_hits), int(solver.t_low_rail_hits),
                             int(solver.rad_clamp_hits))
        assert (t_max, low, clamp) == (counters.t_max_phys_hits, counters.t_low_rail_hits,
                                       counters.rad_clamp_hits)
        if expect == "rail":
            assert t_max > 0 and clamp == 0 and int(T[0, 0]) == 16000 << 16
        else:
            assert clamp > 0 and t_max == 0
            assert int(T[0, 0]) == int(table.e_ceiling_q(phi)) == (408 << 16) - 1
        print(f"\nG5 rail/clamp: clamp_on={clamp_on} Phi={phi}: t_max_phys_hits={t_max} "
              f"rad_clamp_hits={clamp} T={T[0, 0] / 65536:.0f} game after 8 ticks")


def test_g5_burning_crate_above_its_cap_is_not_clamped():
    """G5 non-vacuity (b): a crate held hot by COMBUSTION (T_before = 1263 game,
    far above the clamp's ceiling e_ceiling_q(Φ) of the ambient bath it sits
    in) is NOT clamped — it cools by one radiative step and rad_clamp_hits
    stays 0 — where the bare v2 form would have set it to that ceiling in one
    tick (reported).

    BREAKS IF: the clamp loses its max(T_before, ·) arm.
    """
    n = 7
    a = R.plane(n, n, A_FURN); d = [row[:] for row in a]; k = R.plane(n, n, 0)
    T = R.plane(n, n, 0); T[3][3] = T_SRC << 16
    sc = R.Scene(a=a, d=d, k=k, T=T, his=HIS_FURN)
    cpp_scene = _FoldScene(sc)
    before = cpp_scene.tick()
    cap_bare = cpp_scene.table.e_ceiling_q(int(cpp_scene.last_rl[3, 3]))
    assert cpp_scene.counters()[2] == 0
    assert int(cpp_scene.T[3, 3]) < int(before[3, 3])
    assert cap_bare < int(before[3, 3])
    print(f"\nG5 (b): burning crate {before[3, 3] / 65536:.1f} -> {cpp_scene.T[3, 3] / 65536:.1f} "
          f"game, not clamped (bare-form cap would be {cap_bare >> 16} game)")


# --------------------------------------------------------------------------- #
# G6 isotropy
# --------------------------------------------------------------------------- #
def _point_fluence_cpp(transport, half, n_grid, *, fleck):
    a = R.plane(n_grid, n_grid, 0); d = R.plane(n_grid, n_grid, 0)
    T = R.plane(n_grid, n_grid, 0)
    c = n_grid // 2
    for y in range(c - half, c + half + 1):
        for x in range(c - half, c + half + 1):
            a[y][x] = ONE; d[y][x] = ONE; T[y][x] = T_SRC << 16
    _rn, _rf, _ra, rl, _fl, _sw = cpp_sweep(a, d, K_LEAK, T, 3, ts_from_a(a),
                                            transport=transport, fleck=fleck)
    return rl, c


def _isotropy_table(n_grid, *, fleck):
    phi_ign = R.E[R.e_bucket_of(G.T_IGN_GAME << 16)]
    got, lines = {}, []
    for transport in ("shear", "step"):
        for half, nm in ((0, "1 tile"), (1, "3x3"), (2, "5x5")):
            rl, c = _point_fluence_cpp(transport, half, n_grid, fleck=fleck)
            fl = rl.tolist()
            mean_r, spread, _ = G.footprint(fl, c, phi_ign)
            rings = G.ring_stats(fl, c, (1, 2, 3, 5, 8))
            got[(transport, half)] = (mean_r, spread, rl)
            lines.append(f"  {transport:5s} {nm:>6}: mean r_ign = {mean_r:6.2f} tiles, "
                         f"spread = {spread:5.3f}, ring max/min "
                         + " ".join(f"r{r}={v:5.2f}" for r, v in rings.items()))
    return got, lines


def test_g6_isotropy_shear_rounder_than_step_at_one_tile_and_five_by_five():
    """G6: the 64-direction ignition footprint spread (max/min radius at the
    E°[280] crossing) and the ring max/min of the fluence, on an 81x81 grid
    with a 1-tile / 3x3 / 5x5 source at 1263 game, leak on, both transports,
    in the REFERENCE'S configuration (Fleck damping off — `f_plane=None` is
    what sweep_ref_q_gates.gate6_isotropy measures and what design §2.4
    quotes). The fluence planes are asserted EQUAL to the reference's own on
    the full 81x81 grid (gate 0 at scale), the numbers are printed, and the
    ORDERING shear < step is asserted at 1 tile and at 5x5 only (P0 found the
    3x3 ordering flips under the ambient ring).

    The SAME table with the engine's own Fleck factor ON is printed beside it
    and its ordering is NOT asserted — a FINDING for the design: the damped
    ordering is not stable under the alpha floor. At the superseded floor of ½
    (P1) the 1-tile case flipped, 1.716 shear vs 1.688 step; at the RULED floor
    0 (P2a) that flip is GONE (1.520 vs 1.762) and the 5x5 case flips instead
    (1.261 vs 1.249, ~1 %). So the "heat takes shear" table rests on the
    undamped configuration either way, which is where this gate asserts.

    BREAKS IF: the transport constants change, or the half-offset ordinate set
    is replaced by one with an ordinate on an axis.
    """
    n_grid = 81
    got, lines = _isotropy_table(n_grid, fleck=False)
    print("\nG6 isotropy, reference configuration (Fleck OFF), C++ sweep, 81x81, "
          "k=0.10, ambient ring on:\n" + "\n".join(lines))
    # gate 0 at scale: the reference's own fluence planes, cell for cell
    for transport in ("shear", "step"):
        for half in (0, 1, 2):
            ref_fl, _c = G._point_fluence(transport, half, n_grid)
            assert np.array_equal(got[(transport, half)][2],
                                  np.asarray(ref_fl, dtype=np.int64)), (transport, half)
    assert got[("shear", 0)][1] < got[("step", 0)][1]
    assert got[("shear", 2)][1] < got[("step", 2)][1]
    assert all(np.isfinite(v[1]) and v[1] >= 1.0 for v in got.values())
    got_f, lines_f = _isotropy_table(n_grid, fleck=True)
    print("G6 isotropy, ENGINE configuration (the engine's own Fleck factor ON), "
          "reported, not asserted:\n" + "\n".join(lines_f))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
