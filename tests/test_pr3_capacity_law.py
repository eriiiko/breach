"""P-R3 — the CAPACITY LAW at the solver boundary
(docs/radiation_raycaster_extinction_ruling_2026-07-31.md A3, on Erik's ruling R-b).

The fire logistic's growth term used to carry a HARDWIRED carrying capacity of 1
(the ``(1 - I)`` factor). It is now RESOURCE-PROPORTIONAL — ``I_cap = c*a`` with
``a = F*o2f*hot`` and ``c = I_cap_per_avail``::

    gap  = avail*hot - I/c                        (SIGNED — may go negative)
    grow = k_grow * I * gap * (1 + k_wind_fan*W)
    die  = k_die * (1 - avail*hot) * I + k_wind_strip * W * (1-I) * I   (UNCHANGED)

which is the logistic ``k_grow*a*I*(1 - I/(c*a))`` with ``a`` cancelled out of
the bracket — hence no division in the sim path. Its fixed point and sustain
threshold are::

    I_eq = c * (a - r*(1-a)),      r = k_die/k_grow
    sustain  <=>  a > r/(1+r)

WHY THIS IS THE PATCH THAT MATTERS. Under the old law ``r`` set BOTH the
equilibrium intensity AND the extinction wall (``I_eq = 1 - r(1-a)/a``), so
asking for a small fire forced ``r`` up against the operating point: measured
(ruling §5), the product ``F*o2f*hot`` could fall only to 80.5% of its ambient
value before the fire died at ANY temperature. Consequences, both measured: at
most 19.5% of a crate's hp could ever burn (fuel-governed death was unreachable
at every dial in the tuning loop), and the literature-anchored O2 extinction
limit ``o2_frac_ext = 0.13`` was DEAD CODE, because the logistic wall bit first
at X = 0.1944. Moving size into ``c`` severs that identity: each dial now has
exactly one job — ``c`` = size, ``k_grow`` = tempo, ``k_die`` = where the death
wall sits.

These tests drive the C++ ``FireSimulation.step`` in isolation with ``hot``
pinned to 1 and ``F`` pinned to 1 (the pattern ``test_fuel_fraction_axis.py``
established), so the ONLY moving part is the capacity law itself.

R1 O2f-RENORMALIZATION (fire session #12, 2026-09-01, docs/fire_3c_design_
2026-09-01.md "Ruling R1") moves what ``avail = F*o2f`` (i.e. ``a`` above, with
F pinned to 1) READS: ``o2f`` is renormalized to ``o2_frac_amb`` (0.21) instead
of ``o2_frac_full`` (1.0), and its clamp's upper edge is the NEW ``o2f_cap``
(5.0) instead of 1.0 — so ambient air (X = 0.21) now reads ``a == 1.0``
EXACTLY (was 0.092), and the whole [X_ext, X_amb] span (0.13..0.21, just 0.08
wide) carries the range ``a`` used to sweep out over [X_ext, X_full]
(0.13..1.0, 0.87 wide). This file's own ``_o2f`` helper and every scene below
are re-derived against that new span — this is a PROBE-LAW change, not a
capacity-law change: ``I_eq = c*(a - r*(1-a))`` and ``sustain <=> a >
r/(1+r)`` are UNCHANGED shapes, just fed a differently-scaled ``a``.

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_pr3_capacity_law.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                                     # noqa: E402
from simulation import fire_fixed                               # noqa: E402
from simulation.materials import (                              # noqa: E402
    MAT_FURNITURE, MAT_WOOD, MaterialTable, fire_T_ext_from_ignition,
    quantize_q16,
)

Q = fire_fixed.quantize_scalar
ONE = fire_fixed.FP_ONE
ONE_F = fire_fixed.FP_ONE_F

# The dials the gate is specified at (ruling §2 row P-R3). k_die 0.035 is P-R5's
# blessed-config move; here it is set explicitly on the solver, so this file
# does not depend on what config.toml currently ships.
K_GROW = 3.5
K_DIE = 0.035
C_CAP = 2.53
R = K_DIE / K_GROW                     # 0.010
X_EXT, X_FULL = 0.13, 1.0              # X_FULL: the DEMAND-side (combustion)
                                        # reference — unread by this file's
                                        # scenes (FireSimulation.step only)
# R1 (fire session #12, docs/fire_3c_design_2026-09-01.md "Ruling R1"): the
# SUSTAIN law's span upper reference, and the new clamp ceiling on it. Set
# explicitly on every `_make()` sim below (not left to the struct default) so
# this file states its own contract rather than riding a compiled default.
X_AMB = 0.21
O2F_CAP = 5.0

# Q16.16 truncation margin. Every multiply in the pinned left-fold truncates
# toward -inf, and near the fixed point the per-tick delta is a handful of
# counts, so the solver parks a hair BELOW the real-arithmetic root. Measured
# shortfall at these dials: 0.12% - 0.59%.
TOL_REL = 0.04


def _o2f(X, x_amb=X_AMB, cap=O2F_CAP):
    """The SUSTAIN continuous-O2 law in plain float (Peatross & Beyler linear
    form), R1-renormalized: anchored on x_amb (not x_full), capped at `cap`
    (not 1.0)."""
    return min(cap, max(0.0, (X - X_EXT) / (x_amb - X_EXT)))


def _I_eq(a, c=C_CAP, r=R):
    """THE CAPACITY LAW'S FIXED POINT: I_eq = c*(a - r*(1-a))."""
    return c * (a - r * (1.0 - a))


def _make(c=C_CAP, k_grow=K_GROW, k_die=K_DIE, I_min=0.0, o2_frac_amb=X_AMB,
          o2f_cap=O2F_CAP):
    """A FireSimulation with everything EXCEPT the capacity law pinned inert:
    no wind, no wall damage, no smoke, temp_scale identity."""
    sim = bp.FireSimulation()
    p = sim.params
    p.k_grow = float(k_grow)
    p.k_die = float(k_die)
    p.I_cap_per_avail = float(c)
    p.fire_T_ext = 350.0
    p.fire_T_span = 150.0
    p.fuel_ref = 60.0
    p.o2_frac_ext = X_EXT
    p.o2_frac_full = X_FULL             # DEMAND-side reference; unread here
    p.o2_frac_amb = float(o2_frac_amb)  # R1: the SUSTAIN law's live reference
    p.o2f_cap = float(o2f_cap)          # R1: the NEW sustain clamp ceiling
    p.I_min = float(I_min)
    p.k_wind_fan = 0.0
    p.k_wind_strip = 0.0
    p.wall_damage = 0.0         # F stays pinned at 1 (no fuel drain)
    # smoke_emission RETIRED at P-S1 — the field no longer exists on
    # FireParams (the scatter it drove is deleted); nothing to zero here now.
    p.temp_scale = ONE_F        # identity: T in game units IS the Q16.16 field
    return sim


def _scene(X, I0):
    """A 3x3 scene: a burning flammable wall at the centre with EXACTLY ONE open
    neighbour holding the chosen mole fraction (n_o2 = X, n_total = 1), so
    ``Sum n_o2 / Sum n_total`` is that single cell's own fraction. wall_hp 100
    against fuel_ref 60 -> F clamps to 1; T = 600 against T_ext 350 / span 150
    -> `hot` clamps to exactly FP_ONE. So ``a == o2f(X)``."""
    h = w = 3
    z = lambda: np.zeros((h, w), dtype=np.int32)          # noqa: E731
    st = dict(
        fire=z(), atmosphere=z(), n_o2=z(), n_total=z(), smoke=z(),
        wall_hp=z(), temperature=z(), wind_x=z(), wind_y=z(),
        is_wall=np.ones((h, w), dtype=bool),
        is_vacuum=np.zeros((h, w), dtype=bool),
        flammable=np.zeros((h, w), dtype=bool),
    )
    st["fire"][1, 1] = Q(float(I0))
    st["wall_hp"][1, 1] = Q(100.0)
    st["temperature"][1, 1] = Q(600.0)
    st["is_wall"][1, 2] = False                            # the one open cell
    st["flammable"][1, 1] = True
    st["n_o2"][1, 2] = Q(float(X))
    st["n_total"][1, 2] = Q(1.0)
    return st


def _step(sim, st, dt=1.0 / 24.0, fire_T_ext_plane=None):
    return sim.step(fire=st["fire"], atmosphere=st["atmosphere"],
                    n_o2=st["n_o2"], n_total=st["n_total"], smoke=st["smoke"],
                    wall_hp=st["wall_hp"], temperature=st["temperature"],
                    wind_x=st["wind_x"], wind_y=st["wind_y"],
                    is_wall=st["is_wall"], is_vacuum=st["is_vacuum"],
                    flammable=st["flammable"], dt=dt,
                    fire_T_ext_plane=fire_T_ext_plane)


def _drive(X, I0=0.12, ticks=6000, sim=None):
    """Run the solver to its fixed point and return the settled intensity."""
    sim = sim or _make()
    st = _scene(X, I0)
    for _ in range(ticks):
        _step(sim, st)
    return int(st["fire"][1, 1]) / ONE_F


# ---------------------------------------------------------------------------
# 1. THE FIXED POINT — I_eq = c*(a - r*(1-a)) at three availabilities
# ---------------------------------------------------------------------------
# The ruling's enrichment row (A3, Erik ruling R-a — the response is LINEAR in
# O2, so these are equally spaced in `a`, which the OLD hyperbolic law was not):
#   X = 0.21 (ambient) -> a = 0.09195 -> I_eq 0.2100
#   X = 0.25           -> a = 0.13793 -> I_eq 0.3275
#   X = 0.30           -> a = 0.19540 -> I_eq 0.4746
#
# R1 RE-DERIVATION (fire session #12, 2026-09-01, docs/fire_3c_design_2026-09-
# 01.md "Ruling R1"): `a` is renormalized to o2_frac_amb, so ambient (X=0.21)
# now reads a == 1.0 EXACTLY, and I_eq(a=1.0, c=2.53) = 2.53 — saturated at
# the [0,1] clamp, not a genuine (unclamped) fixed point to gate on. The whole
# unclamped range this test wants to sample now lives BELOW ambient (X in
# [0.13, 0.21), a in [0, 1)), not at/above it. Three equally-spaced `a` rows
# in that band, mirroring the ORIGINAL row shape (three points, increasing a,
# increasing I_eq, all comfortably unsaturated at c=2.53):
#   a = 0.15 -> X = 0.142 -> I_eq 0.35800
#   a = 0.25 -> X = 0.150 -> I_eq 0.61353
#   a = 0.35 -> X = 0.158 -> I_eq 0.86906
_ROWS = [
    pytest.param(0.142, 0.15, 0.35800, id="a-0.15"),
    pytest.param(0.150, 0.25, 0.61353, id="a-0.25"),
    pytest.param(0.158, 0.35, 0.86906, id="a-0.35"),
]


@pytest.mark.parametrize("X, a_want, I_eq_want", _ROWS)
def test_capacity_law_settles_at_its_derived_fixed_point(X, a_want, I_eq_want):
    """THE GATE. With `hot` pinned 1 and F pinned 1, the solver driven to
    equilibrium must land on ``I_eq = c*(a - r*(1-a))`` — the whole reason the
    patch exists, since it is `c` (not `r`) that now sets the size."""
    a = _o2f(X)
    assert abs(a - a_want) < 5e-4, f"scene mis-specified: a = {a}"
    pred = _I_eq(a)
    assert abs(pred - I_eq_want) < 2e-3, f"ruling row drift: pred = {pred}"
    got = _drive(X)
    rel = abs(got - pred) / pred
    assert rel <= TOL_REL, (
        f"X={X}: settled I = {got:.6f}, predicted I_eq = {pred:.6f} "
        f"({rel * 100:.2f}% off, tol {TOL_REL * 100:.0f}%)")
    # ...and it must SETTLE, not merely pass through: another 500 ticks moves it
    # by well under the tolerance.
    sim = _make()
    st = _scene(X, 0.12)
    for _ in range(6000):
        _step(sim, st)
    parked = int(st["fire"][1, 1])
    for _ in range(500):
        _step(sim, st)
    assert abs(int(st["fire"][1, 1]) - parked) <= max(2, parked // 200)


def test_the_fixed_point_is_approached_from_ABOVE_too():
    """A fire started ABOVE its capacity must SHRINK to it — the signed `gap`
    path (``gap < 0`` -> ``grow < 0``). This is the branch the ruling calls out
    as riding the existing signed-delta machinery: nothing special is needed,
    but if the subtract were done unsigned the fire would grow instead.

    R1 RE-ANCHOR (fire session #12, docs/fire_3c_design_2026-09-01.md "Ruling
    R1"): X = 0.21 (ambient) now reads a == 1.0 exactly, giving I_eq(c=2.53) =
    2.53 — an ALREADY-saturated target, so a fire started at I0=0.90 there
    would simply keep climbing to the [0,1] clamp (never approaching from
    above at all). Re-anchored at one of the new unclamped rows (X=0.150,
    a=0.25, I_eq=0.61353) with I0 started comfortably above that target."""
    X = 0.150
    a = _o2f(X)
    pred = _I_eq(a)
    got = _drive(X, I0=0.90)
    assert got < 0.90, "a fire above its capacity must shrink"
    assert abs(got - pred) / pred <= TOL_REL, (
        f"from above: settled {got:.6f} vs predicted {pred:.6f}")


def test_size_and_death_are_decoupled():
    """The POINT of the law, asserted directly: changing `c` moves the fire's
    SIZE and leaves the sustain threshold alone; changing `r` moves the
    threshold. Under the old law one number did both.

    R1 RE-ANCHOR (fire session #12, docs/fire_3c_design_2026-09-01.md "Ruling
    R1"): X = 0.21 (ambient) now reads a == 1.0, at which BOTH c=1.0 (I_eq=1.0)
    and c=5.0 (I_eq=5.0) are already saturated at the [0,1] clamp — the two
    sizes would read identically (both 1.0), which would prove nothing. One of
    the new unclamped rows (X=0.142, a=0.15) keeps both c values genuinely
    unsaturated (I_eq 0.1415 / 0.7075) so the size comparison is real."""
    X = 0.142
    a = _o2f(X)
    small = _drive(X, sim=_make(c=1.0))
    big = _drive(X, sim=_make(c=5.0))
    assert abs(small - _I_eq(a, c=1.0)) / _I_eq(a, c=1.0) <= TOL_REL
    assert abs(big - _I_eq(a, c=5.0)) / _I_eq(a, c=5.0) <= TOL_REL
    assert big > small * 4.5, "size must scale ~linearly with c"
    # Both still sustain: the threshold a > r/(1+r) = 0.0099 is untouched by c,
    # and this X's a = 0.15 clears it by 15x.
    assert small > 0.0 and big > 0.0


def test_sustain_threshold_sits_at_r_over_one_plus_r():
    """``a > r/(1+r)`` is the sustain condition — the SAME shape as the old law,
    but `r` is now free to sit at the physical limits (0.010) instead of being
    dragged up to the operating point to hold the fire small.

    R1 RE-DERIVATION (fire session #12, docs/fire_3c_design_2026-09-01.md
    "Ruling R1"): `a` is renormalized against o2_frac_amb (span 0.13..0.21,
    0.08 wide) instead of o2_frac_full (span 0.13..1.0, 0.87 wide) — same
    a_thresh, but a MUCH smaller X floor (the whole span the old ``+0.02`` /
    ``-0.004`` probe offsets were calibrated against shrank ~11x, so they are
    re-derived to ``+0.002`` / ``-0.0005``, proportionally similar probes into
    the new, much narrower span)."""
    a_thresh = R / (1.0 + R)                       # 0.009901
    x_thresh = X_EXT + a_thresh * (X_AMB - X_EXT)   # 0.130792 — the LIVE X floor
    assert abs(x_thresh - 0.130792) < 1e-5
    # Above the threshold -> a live fire that parks at its (small) capacity.
    above = _drive(x_thresh + 0.002, I0=0.05, ticks=6000)
    a_above = _o2f(x_thresh + 0.002)
    assert above > 0.0
    # NB an ABSOLUTE tolerance here, not the 4% relative one used at the three
    # gate availabilities: the truncation shortfall is a fixed handful of Q16
    # counts, so its RELATIVE share grows as I_eq shrinks, and this point sits
    # deliberately just above the wall at I_eq = 0.064 (vs 0.36-0.87 in the
    # main _ROWS gate).
    assert 0.0 < above <= _I_eq(a_above)
    assert _I_eq(a_above) - above < 0.008
    # Below it -> no fixed point exists and the fire decays away (here with the
    # snap-extinguish floor OFF, so the decay itself is what is asserted)...
    below = _drive(x_thresh - 0.0005, I0=0.05, ticks=3000)
    assert below < 0.05 * 0.05, f"below the threshold I must collapse, got {below}"
    # ...and with the shipped I_min it snaps out outright. THE HEADLINE: this
    # X floor is 0.1308, not the old (pre-R1) law's 0.1944 — so `o2_frac_ext`
    # = 0.13 (Peatross-Beyler) is REACHABLE again instead of being dead code,
    # now even closer to the physical limit than the pre-R1 capacity law
    # (P-R3) already put it.
    assert _drive(x_thresh - 0.0005, I0=0.05, ticks=3000,
                  sim=_make(I_min=0.02)) == 0.0
    assert x_thresh < 0.1944, "the logistic wall must no longer bite first"


# ---------------------------------------------------------------------------
# 2. THE Q16.16 GROWTH QUANTUM (the 95bdec0 trap)
# ---------------------------------------------------------------------------
def test_seed_vicinity_net_growth_clears_the_q16_quantum():
    """A fire whose per-tick net growth TRUNCATES TO ZERO cannot grow at all,
    however right the algebra is — that is the trap commit 95bdec0 fell into
    (largest dI/dt on the whole logistic = 0.969 counts/tick). At the seed the
    net must clear >= 2 counts/tick.

    Derived (ruling A3): dt*k_grow*seed*(a - seed/c)*65536 ~= 51 counts of
    growth, ~40 counts NET of the die term.

    R1 RE-DERIVATION (fire session #12, docs/fire_3c_design_2026-09-01.md
    "Ruling R1"): `a` at X=0.21 (ambient) is now 1.0 exactly (renormalized to
    o2_frac_amb, was 0.092 against o2_frac_full) — a much bigger `a`, so the
    net growth here is now ~1092 counts/tick, not ~40 (still comfortably clear
    of the Q16 quantum — if anything, less at risk of it than before). The die
    term is also re-derived with the R1 sign-fix floor `max(0, 1-a)` (at
    a==1.0 exactly this is 0 either way, but the formula now matches the
    solver's actual shape for any future a > 1 edit of this scene)."""
    sim = _make()
    st = _scene(0.21, 0.12)                    # seed 0.12, a = 1.0 (R1: ambient)
    before = int(st["fire"][1, 1])
    _step(sim, st)
    net = int(st["fire"][1, 1]) - before
    assert net >= 2, f"net growth {net} counts/tick is at/below the Q16 quantum"
    # And the derived value, to a couple of counts (this IS the ruling's ~1092).
    a, seed = _o2f(0.21), 0.12
    grow = K_GROW * seed * (a - seed / C_CAP)
    die = K_DIE * max(0.0, 1.0 - a) * seed      # R1 die-term sign-fix floor
    want = (grow - die) / 24.0 * ONE
    assert abs(net - want) <= 3, f"net {net} counts vs derived {want:.1f}"


# ---------------------------------------------------------------------------
# 3. THE CAPACITY CEILING IS REALLY THE DIAL
# ---------------------------------------------------------------------------
def test_capacity_ceiling_off_is_legal_and_means_unbounded():
    """``I_cap_per_avail <= 0`` is the documented "ceiling OFF" value (INV_C =
    0): the deliberate answer to a divide-by-zero misconfig, and the probe idiom
    the o2f-readout tests use. With it off, `gap == avail*hot` and the fire runs
    away to the clamp instead of parking at a fixed point."""
    for c_off in (0.0, -1.0):
        got = _drive(0.21, I0=0.12, ticks=3000, sim=_make(c=c_off))
        assert got > 0.99, f"c={c_off}: expected runaway to the clamp, got {got}"


def test_capacity_law_is_bit_deterministic():
    """Identical inputs -> identical output, twice (the standing Q16.16
    contract; the signed `gap` must not have introduced any float)."""
    a = _drive(0.25, ticks=400)
    b = _drive(0.25, ticks=400)
    assert a == b


# ---------------------------------------------------------------------------
# 4. PER-MATERIAL fire_T_ext — the plane contract (the fuel_ref precedent)
# ---------------------------------------------------------------------------
def _rand_fire_state(rng, h, w):
    """A randomised scene, mirroring test_fuel_fraction_axis's ``_fire_state``
    so the byte-identity claim is exercised over a genuinely mixed grid."""
    n = h * w
    _q = lambda x: np.ascontiguousarray(                       # noqa: E731
        (np.asarray(x, dtype=np.float64) * ONE_F).astype(np.int32))
    flammable = (rng.random(n) < 0.6).reshape(h, w)
    is_wall = np.ones((h, w), dtype=bool)
    is_wall[rng.random(n).reshape(h, w) < 0.5] = False
    is_vacuum = (rng.random(n) < 0.08).reshape(h, w)
    total = rng.random(n).reshape(h, w) * 1.0 + 0.3
    frac = rng.random(n).reshape(h, w) * 0.35
    return dict(
        fire=_q(rng.random(n).reshape(h, w)),
        atmosphere=_q(np.ones((h, w))),
        n_o2=_q(frac * total),
        n_total=_q(total),
        smoke=_q(np.zeros((h, w))),
        wall_hp=_q(rng.random(n).reshape(h, w) * 60.0),
        # T spans the whole ramp so `hot` is genuinely partial on many tiles —
        # otherwise a clamped gate would hide a wrong subtrahend.
        temperature=_q(rng.random(n).reshape(h, w) * 900.0),
        wind_x=_q(rng.random(n).reshape(h, w) * 0.4 - 0.2),
        wind_y=_q(rng.random(n).reshape(h, w) * 0.4 - 0.2),
        is_wall=np.ascontiguousarray(is_wall),
        is_vacuum=np.ascontiguousarray(is_vacuum),
        flammable=np.ascontiguousarray(flammable),
    )


def _run_plane(state, plane, fire_T_ext=350.0):
    sim = bp.FireSimulation()
    sim.params.fire_T_ext = float(fire_T_ext)
    sim.params.temp_scale = ONE_F
    c = {k: v.copy() for k, v in state.items()}
    destroyed = _step(sim, c, fire_T_ext_plane=plane)
    return c, sorted(tuple(t) for t in destroyed)


@pytest.mark.parametrize("seed", range(8))
def test_uniform_fire_T_ext_plane_equals_the_scalar_fallback(seed):
    """THE BACK-COMPAT CONTRACT, at the solver boundary — the same property the
    ``fuel_recip`` plane had to satisfy before it. A plane filled with
    ``quantize(fire_T_ext)`` must reproduce the no-plane (pre-derivation) result
    BYTE FOR BYTE, on every mutated field and on the destroyed list."""
    rng = np.random.default_rng(31000 + seed)
    h, w = 17, 23
    st = _rand_fire_state(rng, h, w)
    ref = np.full((h, w), quantize_q16(350.0), dtype=np.int32)
    a, da = _run_plane(st, None)
    b, db = _run_plane(st, np.ascontiguousarray(ref))
    for k in ("fire", "temperature", "smoke", "wall_hp"):
        assert np.array_equal(a[k], b[k]), f"{k} differs (seed {seed})"
    assert da == db


@pytest.mark.parametrize("seed", range(4))
def test_the_fire_T_ext_plane_is_actually_read(seed):
    """NON-VACUOUSNESS control for the test above: a plane that is NOT the
    scalar must diverge, or the equality would hold for the boring reason."""
    rng = np.random.default_rng(32000 + seed)
    h, w = 17, 23
    st = _rand_fire_state(rng, h, w)
    other = np.full((h, w), quantize_q16(180.0), dtype=np.int32)
    a, _ = _run_plane(st, None)
    b, _ = _run_plane(st, np.ascontiguousarray(other))
    assert not np.array_equal(a["fire"], b["fire"])


def test_a_non_uniform_plane_is_read_per_tile():
    """The point of the axis: two tiles with the SAME temperature but different
    material extinction floors must step differently in the SAME call."""
    h, w = 3, 6
    st = _rand_fire_state(np.random.default_rng(11), h, w)
    st["flammable"][:] = False
    st["flammable"][1, :] = True
    st["is_wall"][:] = False
    st["is_wall"][1, :] = True
    st["is_vacuum"][:] = False
    st["fire"][:] = 0
    st["fire"][1, :] = Q(0.5)
    st["wall_hp"][:] = Q(60.0)
    st["temperature"][:] = Q(260.0)      # BETWEEN the two floors' ramps
    st["wind_x"][:] = 0
    st["wind_y"][:] = 0
    st["n_total"][:] = Q(1.0)
    st["n_o2"][:] = Q(0.21)
    plane = np.full((h, w), quantize_q16(200.0), dtype=np.int32)   # wood
    plane[1, 3:] = quantize_q16(180.0)                             # furniture
    out, _ = _run_plane(st, np.ascontiguousarray(plane))
    lo = out["fire"][1, 1:3]      # interior tiles (edge columns see 3 nbrs)
    hi = out["fire"][1, 3:5]
    assert len(set(lo.tolist())) == 1 and len(set(hi.tolist())) == 1
    assert hi[0] > lo[0], "the tile with the LOWER extinction floor is hotter"


# ---------------------------------------------------------------------------
# 5. THE DERIVATION ITSELF (ruling A3 ride-along)
# ---------------------------------------------------------------------------
def test_fire_T_ext_is_derived_from_ignition_temp():
    """``fire_T_ext[mat] = ignition_temp[mat] - ignition_to_ext_delta`` — ONE
    new global, zero new per-material columns. The number that motivated the
    ORIGINAL delta of 100: furniture 280 - 100 = 180, EXACTLY the blessed
    bench value, so a derived run and the old ``--set fire_T_ext=180``
    override agreed.

    Re-derivation note (fire-family triage, 2026-08-30): P-K0 (9016cd7,
    2026-08-13) promoted ``ignition_to_ext_delta`` from that bench-derived
    100 to the shipped 200 ("knee geometry — the FOOT", config.toml). The
    literal 180.0/200.0 pins here were downstream of the OLD delta and would
    silently re-pin to whatever config.toml ships next time it moves again;
    read the shipped delta from config instead and derive both expected
    values from it, the same way the table itself derives fire_T_ext — this
    test is about the DERIVATION relationship holding, not about pinning any
    one delta's history."""
    tbl = MaterialTable.from_config()
    delta = float(tbl.ignition_to_ext_delta)
    assert delta == 200.0   # the shipped P-K0 value (9016cd7, 2026-08-13)
    assert float(tbl.fire_T_ext[MAT_FURNITURE]) == pytest.approx(
        float(tbl.ignition_temp[MAT_FURNITURE]) - delta)
    assert float(tbl.fire_T_ext[MAT_WOOD]) == pytest.approx(
        float(tbl.ignition_temp[MAT_WOOD]) - delta)
    for idx, name in enumerate(tbl.names):
        want = fire_T_ext_from_ignition(tbl.ignition_temp[idx],
                                        tbl.ignition_to_ext_delta)
        assert float(tbl.fire_T_ext[idx]) == pytest.approx(want), name
        assert int(tbl.fire_T_ext_q16[idx]) == quantize_q16(want), name


def test_the_invariant_fire_T_ext_below_ignition_temp_is_structural():
    """The defect the derivation closes: the shipped global (350) exceeded BOTH
    flammable materials' ignition temps, so a tile could ignite at 300 and sit
    permanently below its own extinction floor. Derived, it CANNOT — for any
    positive Δ."""
    tbl = MaterialTable.from_config()
    assert float(tbl.ignition_to_ext_delta) > 0.0
    for idx, name in enumerate(tbl.names):
        if not bool(tbl.flammable[idx]):
            continue
        assert float(tbl.fire_T_ext[idx]) < float(tbl.ignition_temp[idx]), name


def test_the_quantize_bake_matches_the_cpp_boundary_cast():
    """``quantize_q16`` must be bit-identical to C++ ``fixedpoint::quantize`` —
    that agreement is WHY a uniform plane equals the scalar fallback."""
    xs = [0.0, 1.0, -1.0, 180.0, 200.0, 350.0, -100.0, 0.5, -0.5,
          123.456789, -987.654321, 1e-5, 32767.5]
    for v in xs:
        assert quantize_q16(v) == bp.fp_quantize(float(v)), v


def test_the_plane_is_a_required_step_tail_argument():
    """Like ``fuel_recip`` / ``thermal_solid`` / ``cool_shift_grid``: a caller
    must not be able to silently put the LIVE engine back on the global."""
    doc = bp.PhysicsEngine.step_tail.__doc__ or ""
    assert "fire_T_ext_plane" in doc, (
        "step_tail's signature must name fire_T_ext_plane; got:\n" + doc)
