"""Gate a — continuous O2->combustion law unit tests
(docs/continuous_o2_law_design_2026-07-24.md §3.a).

The fire O2 sustain factor is now LINEAR in the local O2 MOLE FRACTION
X = Sn_o2/Sn_total over the open 4-neighbours, with an extinction limit:

    o2f = clamp01((X - o2_frac_ext) / (o2_frac_full - o2_frac_ext))

replacing the old smoothstep(P_min, P_full) on ABSOLUTE n_o2 density. These
tests drive the C++ FireSimulation.step in isolation (a 3x3 grid, one burning
flammable wall at the centre, its four open-air neighbours holding a chosen
O2 composition/density) and assert the LAW's shape via the one observable it
controls: the next-tick intensity.

FULL-RESPONSE REFERENCE SPLIT (2026-07-30). The denominator's upper end used to
be ``o2_frac_amb`` — the AMBIENT dial — so ambient air always yielded o2f == 1
and the clamp01 made ambient the ceiling: locally elevated O2 (reservoirs,
leaks, wind delivery) was invisible BY CONSTRUCTION. The span's top is now the
separate ``o2_frac_full`` (pure O2, 1.0, NOT map-overridden), so o2f is a true
physical fraction — "O2 above extinction, normalized to pure oxygen":

    X = 0.13 -> 0.000   X = 0.21 (ambient) -> 0.092
    X = 0.25 -> 0.138   X = 0.30           -> 0.195   X = 1.00 -> 1.000

Design gate a: X = X_full -> o2f = 1; X <= X_ext -> 0; midpoint linearity;
X_ext = 0 degenerates to X/X_full; the clamp guards both ends; a level's
[ambient] o2_frac no longer changes o2f's SHAPE; plus the headline property the
law exists for — INVARIANCE under thermal expansion (same composition, different
density -> same burn), the "density trap" fix.

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_continuous_o2_law.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                                    # noqa: E402
from simulation import fire_fixed                              # noqa: E402

Q = fire_fixed.quantize_scalar
ONE_F = fire_fixed.FP_ONE_F

# Law dials used by the tests (the shipped continuous-O2 defaults).
X_EXT = 0.13
X_AMB = 0.21     # what the ambient atmosphere IS — NOT the law's reference
X_FULL = 1.0     # the FULL-RESPONSE reference (pure O2) — the law's reference
# The span midpoint under the new normalization (o2f == 0.5 there).
X_MID = X_EXT + 0.5 * (X_FULL - X_EXT)     # 0.565


def _expected_o2f(X, x_ext=X_EXT, x_full=X_FULL):
    """The law in plain floating point, for the assertions to compare against."""
    return min(1.0, max(0.0, (X - x_ext) / (x_full - x_ext)))


def _make_sim(o2_frac_ext=X_EXT, o2_frac_full=X_FULL, o2_frac_amb=X_AMB):
    """A FireSimulation with the terms OTHER than o2f pinned so the O2 law is
    the ONLY moving part: hot == 1 (T well above the ramp), F == 1 (full fuel),
    W == 0 (no wind), the shipped k_grow/k_die."""
    sim = bp.FireSimulation()
    p = sim.params
    p.k_grow = 4.0
    p.k_die = 2.0
    p.fire_T_ext = 350.0
    p.fire_T_span = 150.0
    p.fuel_ref = 60.0
    p.I_min = 0.02
    p.k_wind_fan = 0.5
    p.k_wind_strip = 0.5
    p.o2_frac_ext = float(o2_frac_ext)
    p.o2_frac_full = float(o2_frac_full)
    # The ambient dial is SET on every sim in this file precisely to prove it is
    # inert in the law (see test_ambient_override_does_not_change_o2f_shape).
    p.o2_frac_amb = float(o2_frac_amb)
    p.temp_scale = float(ONE_F)          # identity (T in game units == Q16.16)
    return sim


def _step_once(sim, X, density=1.0, I0=0.5):
    """One fire step on a 3x3 grid: centre (1,1) is a burning flammable wall;
    its four open-air neighbours hold n_o2 = X*density, n_total = density.
    Returns the centre tile's next-tick intensity as a float in [0,1]."""
    h = w = 3
    z = lambda: np.zeros((h, w), dtype=np.int32)
    fire = z(); fire[1, 1] = Q(float(I0))
    atmosphere = z()                                  # unused by the O2 law
    n_o2 = z(); n_total = z()
    smoke = z()
    wall_hp = z(); wall_hp[1, 1] = Q(100.0)           # F = clamp01(100/60) = 1
    temperature = z(); temperature[1, 1] = Q(600.0)   # hot = 1
    wind_x = z(); wind_y = z()
    is_wall = np.zeros((h, w), dtype=bool); is_wall[1, 1] = True   # fire lives on a wall
    is_vacuum = np.zeros((h, w), dtype=bool)
    flammable = np.zeros((h, w), dtype=bool); flammable[1, 1] = True

    # The four open-air neighbours of (1,1) carry the chosen composition/density.
    o2_val = Q(float(X) * float(density))
    tot_val = Q(float(density))
    for (yy, xx) in ((0, 1), (2, 1), (1, 0), (1, 2)):
        n_o2[yy, xx] = o2_val
        n_total[yy, xx] = tot_val

    sim.step(fire=fire, atmosphere=atmosphere, n_o2=n_o2, n_total=n_total,
             smoke=smoke, wall_hp=wall_hp, temperature=temperature,
             wind_x=wind_x, wind_y=wind_y, is_wall=is_wall,
             is_vacuum=is_vacuum, flammable=flammable, dt=1.0 / 24.0)
    return float(fire[1, 1]) / ONE_F


# --- gate a: the law's endpoints ------------------------------------------
def test_o2f_full_at_pure_o2_grows():
    """X == X_full (pure O2) -> o2f == 1 -> avail == F -> the fire GROWS. This
    is the endpoint the split MOVED: it used to sit at ambient (0.21)."""
    assert _step_once(_make_sim(), X=X_FULL, I0=0.5) > 0.5


def test_o2f_zero_below_ext_decays():
    """X <= X_ext -> o2f == 0 -> avail == 0 -> grow == 0, only die -> DECAYS."""
    # Strictly below the limit, and AT the limit (o2f == 0 at X == X_ext).
    assert _step_once(_make_sim(), X=0.10, I0=0.5) < 0.5
    assert _step_once(_make_sim(), X=X_EXT, I0=0.5) < 0.5


def test_o2f_half_at_midpoint_is_equilibrium():
    """At the span midpoint o2f == 0.5; with I0 == 0.5 that is the logistic
    equilibrium (grow == die), so the intensity barely moves — a direct read
    of the LINEAR law's half value. The midpoint is now 0.565, not 0.17."""
    assert abs(_step_once(_make_sim(), X=X_MID, I0=0.5) - 0.5) < 0.02


def test_monotone_increasing_in_X():
    """Next-tick intensity is monotone non-decreasing in the O2 fraction —
    the linear law never inverts. Sampled across the WHOLE new span."""
    sim = _make_sim()
    vals = [_step_once(sim, X=x, I0=0.5) for x in (0.10, X_AMB, 0.50, X_FULL)]
    assert vals[0] <= vals[1] <= vals[2] <= vals[3]
    assert vals[0] < vals[3]                          # strictly separated ends


def test_ext_zero_degenerates_to_proportional():
    """o2_frac_ext == 0 -> o2f == X / X_full (Erik's pure proportional). At
    X == X_full/2 that is 0.5 -> equilibrium at I0 == 0.5."""
    sim0 = _make_sim(o2_frac_ext=0.0)
    assert abs(_step_once(sim0, X=X_FULL / 2.0, I0=0.5) - 0.5) < 0.02
    assert _step_once(sim0, X=X_FULL, I0=0.5) > 0.5    # full fraction still grows


# --- the headline property: density invariance (the trap fix) -------------
def test_density_invariance_bit_exact_in_clamped_region():
    """Same composition, wildly different density -> IDENTICAL burn. In the
    clamped region (X == 1.0 -> o2f == FP_ONE exactly) the next intensity is
    bit-exact across a 10x density change: the fraction is invariant under
    thermal expansion (the density trap the smoothstep-on-absolute-n_o2 law
    suffered is closed)."""
    dense = _step_once(_make_sim(), X=1.0, density=1.0, I0=0.5)
    thin = _step_once(_make_sim(), X=1.0, density=0.1, I0=0.5)
    assert dense == thin


def test_thin_hot_gas_at_ambient_composition_burns_the_same():
    """A thermally-expanded cell (ambient composition 21/79 but 1/10 the
    density) must burn essentially like dense ambient air — the OLD absolute
    law would have read n_o2 ~= 0.021 as near-starved. Both land within a hair
    of each other. (Since the full-response split, ambient is o2f = 0.092, so
    at the UNTUNED k_die/k_grow both DECAY — the invariance claim is about the
    two being equal, not about the sign; the k_die/k_grow rescale that puts a
    normal fire at I ~ 0.5 is Erik's tuning loop, deliberately not done here.)"""
    dense = _step_once(_make_sim(), X=X_AMB, density=1.0, I0=0.5)
    thin = _step_once(_make_sim(), X=X_AMB, density=0.1, I0=0.5)
    assert abs(dense - thin) < 5e-3


def test_vitiation_starves_at_constant_density():
    """The complement: at FIXED density, replacing O2 with inert product
    (dropping the fraction from ambient to below the extinction limit) DOES
    starve the fire harder — only true vitiation, not low density, kills it."""
    fresh = _step_once(_make_sim(), X=X_AMB, density=1.0, I0=0.5)
    vitiated = _step_once(_make_sim(), X=0.10, density=1.0, I0=0.5)
    assert vitiated < fresh < 0.5      # ambient alone no longer sustains I=0.5
    # ...and pure O2 at the same density does sustain growth.
    assert _step_once(_make_sim(), X=X_FULL, density=1.0, I0=0.5) > 0.5


def test_fully_enclosed_reads_zero_and_decays():
    """No open neighbour (all walls) -> both neighbour sums 0 -> X == 0 ->
    o2f == 0 -> the fire decays (matches the old count==0 guard)."""
    sim = _make_sim()
    h = w = 3
    z = lambda: np.zeros((h, w), dtype=np.int32)
    fire = z(); fire[1, 1] = Q(0.5)
    wall_hp = z(); wall_hp[1, 1] = Q(100.0)
    temperature = z(); temperature[1, 1] = Q(600.0)
    is_wall = np.ones((h, w), dtype=bool)             # everything solid: no open air
    flammable = np.zeros((h, w), dtype=bool); flammable[1, 1] = True
    sim.step(fire=fire, atmosphere=z(), n_o2=z(), n_total=z(), smoke=z(),
             wall_hp=wall_hp, temperature=temperature, wind_x=z(), wind_y=z(),
             is_wall=is_wall, is_vacuum=np.zeros((h, w), dtype=bool),
             flammable=flammable, dt=1.0 / 24.0)
    assert float(fire[1, 1]) / ONE_F < 0.5


def test_deterministic_repeat():
    """The law is bit-deterministic: identical inputs -> identical output."""
    a = _step_once(_make_sim(), X=X_AMB, I0=0.5)
    b = _step_once(_make_sim(), X=X_AMB, I0=0.5)
    assert a == b


# ===========================================================================
# FULL-RESPONSE REFERENCE SPLIT (2026-07-30) — o2f read DIRECTLY, in raw counts
# ===========================================================================
# The tests above read o2f indirectly, through the logistic's sign. These read
# the actual Q16.16 value out of BOTH laws, exactly, by arranging each so that
# its one observable equals o2f_q with no rounding in between:
#
#   FIRE       with k_grow = 4, k_die = 0, no wind, F = 1, hot = 1, I0 = 0.5,
#              dt = 1.0 the pinned left-fold collapses to  grow == o2f_q  and
#              I_next == 0.5 + o2f  exactly (valid while o2f <= 0.5).
#   COMBUSTION with burn_rate = 1.0, dt = 1.0 (burn_cap_q == FP_ONE) and a
#              source at I = 1.0, demand_k == o2f_q exactly; an uncontested
#              air cell therefore loses EXACTLY o2f_q counts of O2.
#
# Both scenes give the two laws the SAME mole-fraction inputs — a single open
# air neighbour holding n_o2 = X, n_total = 1 — so the fraction divide itself is
# bit-identical between them and the two o2f values must match to the LSB.

_LOW_XS = (X_EXT, X_AMB, 0.25, 0.30, 0.45, 0.55)   # the o2f <= 0.5 band


def _o2f_fire_q(X, x_ext=X_EXT, x_full=X_FULL, x_amb=X_AMB):
    """o2f in RAW Q16.16 counts, read out of FireSimulation::step. Exact for
    o2f <= 0.5 (above that I_next saturates at 1.0 and the read saturates)."""
    sim = bp.FireSimulation()
    p = sim.params
    p.k_grow = 4.0
    p.k_die = 0.0
    p.fire_T_ext = 350.0
    p.fire_T_span = 150.0
    p.fuel_ref = 60.0
    p.I_min = 0.0
    p.k_wind_fan = 0.0
    p.k_wind_strip = 0.0
    p.o2_frac_ext = float(x_ext)
    p.o2_frac_full = float(x_full)
    p.o2_frac_amb = float(x_amb)
    p.temp_scale = float(ONE_F)
    p.wall_damage = 0.0
    p.smoke_emission = 0.0
    p.fire_pressure_gain = 0.0

    h = w = 3
    z = lambda: np.zeros((h, w), dtype=np.int32)
    fire = z(); fire[1, 1] = Q(0.5)
    n_o2 = z(); n_total = z()
    wall_hp = z(); wall_hp[1, 1] = Q(100.0)           # F = clamp01(100/60) = 1
    temperature = z(); temperature[1, 1] = Q(600.0)   # hot = 1
    # EXACTLY ONE open neighbour, so Σn_o2/Σn_total is the single cell's own
    # fraction — the same divide the combustion law does at that cell.
    is_wall = np.ones((h, w), dtype=bool)
    is_wall[1, 2] = False
    flammable = np.zeros((h, w), dtype=bool); flammable[1, 1] = True
    n_o2[1, 2] = Q(float(X))
    n_total[1, 2] = Q(1.0)

    sim.step(fire=fire, atmosphere=z(), n_o2=n_o2, n_total=n_total, smoke=z(),
             wall_hp=wall_hp, temperature=temperature, wind_x=z(), wind_y=z(),
             is_wall=is_wall, is_vacuum=np.zeros((h, w), dtype=bool),
             flammable=flammable, dt=1.0)
    return int(fire[1, 1]) - Q(0.5)


def _o2f_combustion_q(X, x_ext=X_EXT, x_full=X_FULL, x_amb=X_AMB, density=1.0):
    """o2f in RAW Q16.16 counts, read out of CombustionSolver::step (exact over
    the WHOLE range) as the O2 the uncontested air cell loses in one step.

    ``density`` scales the ABSOLUTE gas the air cell holds without changing its
    mole fraction — needed wherever o2f approaches 1, because the read is only
    valid on the UNCONTESTED branch (demand <= O2[j]); the assert below refuses
    to return a contested (fully-drained) reading."""
    from simulation.gases import O2, INERT_N2, SMOKE, N_GASES

    c = bp.CombustionSolver()
    c.burn_rate = 1.0            # burn_cap_q == FP_ONE at dt = 1.0
    c.o2_thresh_burn = 0.0
    c.H_fuel = 0.0               # no heat deposit needed for this read
    c.soot_yield = 0.0
    c.fuel_per_o2 = 0.0          # do not deplete the source's fuel
    c.o2_frac_ext = float(x_ext)
    c.o2_frac_full = float(x_full)
    c.o2_frac_amb = float(x_amb)

    h = w = 3
    gas = np.zeros((N_GASES, h, w), dtype=np.int32)
    gas[O2][1, 2] = Q(float(X) * float(density))
    gas[INERT_N2][1, 2] = Q(float(density)) - Q(float(X) * float(density))
    solid = np.zeros((h, w), dtype=bool); solid[1, 1] = True
    flammable = np.zeros((h, w), dtype=bool); flammable[1, 1] = True
    wall_hp = np.zeros((h, w), dtype=np.int32); wall_hp[1, 1] = Q(1000.0)
    fire = np.zeros((h, w), dtype=np.int32); fire[1, 1] = Q(1.0)
    ign = np.zeros((h, w), dtype=np.int32); ign[1, 1] = Q(100.0)
    temperature = np.zeros((h, w), dtype=np.int32); temperature[1, 1] = Q(500.0)
    is_vacuum = np.zeros((h, w), dtype=bool)

    before = int(gas[O2][1, 2])
    c.step(gas, O2, INERT_N2, SMOKE, temperature, wall_hp, fire, flammable,
           solid, is_vacuum, ign, 1.0, 1.0, 0.05)
    drawn = before - int(gas[O2][1, 2])
    assert not (drawn > 0 and drawn == before and before < ONE_F), (
        f"contested read at X={X} density={density}: the cell fully drained, so "
        f"{drawn} is O2[j], not o2f — raise `density`")
    return drawn


def test_o2f_table_matches_the_new_normalization():
    """The headline numbers: o2f is now "O2 above extinction, normalized to PURE
    oxygen". X = 0.13 -> 0; ambient 0.21 -> 0.092; 0.25 -> 0.138; 0.30 -> 0.195;
    pure O2 -> 1. Read exactly out of the combustion law (valid at every X)."""
    for X, want in ((0.13, 0.0), (0.21, 0.092), (0.25, 0.138),
                    (0.30, 0.195), (1.0, 1.0)):
        got = _o2f_combustion_q(X) / ONE_F
        assert abs(got - want) < 1e-3, f"X={X}: o2f={got:.6f}, want ~{want}"
        assert abs(got - _expected_o2f(X)) < 1e-4


def test_o2f_clamp_guards_both_ends():
    """clamp01 still guards BELOW extinction and ABOVE the full reference."""
    for X in (0.0, 0.05, 0.10, X_EXT):
        assert _o2f_combustion_q(X) == 0            # nothing drawn at all
    # X can exceed X_full only if the reference is lowered; do that explicitly
    # so the upper clamp is genuinely exercised rather than unreachable.
    assert _o2f_combustion_q(0.9, x_full=0.5, density=8.0) == ONE_F
    assert _o2f_combustion_q(1.0, x_full=0.5, density=8.0) == ONE_F
    # At the SHIPPED reference, X == X_full is the endpoint rather than past it,
    # and reciprocal_q16(N_total) is 1 LSB shy of exact, so the mole fraction
    # reads 65535 counts and o2f follows it — 0.99998, not a clamp failure.
    assert _o2f_combustion_q(1.0) >= ONE_F - 2


def test_ambient_override_does_not_change_o2f_shape():
    """A level's ``[ambient] o2_frac`` writes o2_frac_amb (physics_runner
    _ambient_args). Since the split that dial is NOT the law's reference, so it
    cannot move o2f by a single count — the whole point of the change."""
    for X in (0.15, X_AMB, 0.30, 0.60):
        base = _o2f_combustion_q(X, x_amb=X_AMB)
        for amb in (0.10, 0.15, 0.21, 0.35, 1.0):
            assert _o2f_combustion_q(X, x_amb=amb) == base
            assert _o2f_fire_q(X, x_amb=amb) == _o2f_fire_q(X, x_amb=X_AMB)


def test_full_reference_is_the_dial_that_moves_o2f():
    """The complement: o2_frac_full IS live. Pinning it back to o2_frac_amb
    reproduces the PRE-SPLIT law exactly (o2f == 1 at ambient) — which is what
    gate (a)'s byte-identity capture relies on."""
    # Above ambient, the pinned law saturates EXACTLY (the pre-split ceiling).
    assert _o2f_combustion_q(0.30, x_full=X_AMB, density=8.0) == ONE_F
    # AT ambient it lands on the endpoint itself — within a rounding hair of 1.0
    # (the mole-fraction divide at finite density cannot hit 0.21 exactly).
    assert _o2f_combustion_q(X_AMB, x_full=X_AMB, density=8.0) >= ONE_F - 64
    # ...whereas at the SHIPPED reference ambient is a tenth of that.
    assert _o2f_combustion_q(X_AMB, x_full=X_FULL) < ONE_F // 5


def test_two_o2_laws_are_bit_identical():
    """The fire logistic and the combustion draw must read the SAME law to the
    LSB (they are deliberately bit-identical twins). Same mole-fraction inputs,
    same o2f counts."""
    for X in _LOW_XS:
        assert _o2f_fire_q(X) == _o2f_combustion_q(X), f"laws differ at X={X}"


def test_o2f_is_linear_between_the_endpoints():
    """Equal steps in X give equal steps in o2f (to <=1 LSB) — the law is a
    straight line from (X_ext, 0) to (X_full, 1), not a curve."""
    xs = [0.20, 0.30, 0.40, 0.50, 0.60]
    vals = [_o2f_combustion_q(x) for x in xs]
    deltas = [b - a for a, b in zip(vals, vals[1:])]
    assert max(deltas) - min(deltas) <= 2, deltas


# --- §2.4: ignition reads the SAME mole-fraction law (+ P1b fuel gate) -----
from simulation.combat import apply_temperature_ignition          # noqa: E402
from simulation.materials import MAT_AIR, MAT_WOOD, MaterialTable  # noqa: E402

_TBL = MaterialTable.from_config()


class _GasStub:
    def __init__(self):
        from simulation.gases import O2, INERT_N2
        self.name_to_id = {"o2": O2, "inert_n2": INERT_N2}


class _IgnGMap:
    """A 3x3 gmap stand-in for apply_temperature_ignition: a flammable centre
    surrounded by an air ring at a chosen O2 composition X and density."""

    def __init__(self, X, density=1.0, wall_hp=100.0, temp=350.0, fire0=0.0):
        from simulation.gases import O2, INERT_N2, N_GASES
        self.materials = _TBL
        self.gases = _GasStub()
        m = np.full((3, 3), MAT_AIR, dtype=np.int8)
        m[1, 1] = MAT_WOOD
        self.material = m
        self.flammable = _TBL.flammable[m]
        self.solid = (_TBL.permeability[m] <= 0.0)
        ring = ~self.solid                                    # the open-air ring
        self.gas = np.zeros((N_GASES, 3, 3), dtype=np.int32)
        self.gas[O2] = np.where(ring, Q(float(X) * float(density)), 0)
        self.gas[INERT_N2] = np.where(ring, Q((1.0 - float(X)) * float(density)), 0)
        self.is_vacuum = np.zeros((3, 3), dtype=bool)
        self.temperature = np.zeros((3, 3), dtype=np.int32)
        self.temperature[1, 1] = Q(float(temp))               # >= wood ignition (300)
        self.wall_hp = np.zeros((3, 3), dtype=np.int32)
        self.wall_hp[1, 1] = int(round(float(wall_hp) * ONE_F))
        self.fire = np.zeros((3, 3), dtype=np.int32)
        self.fire[1, 1] = Q(float(fire0))
        # Edge-trigger arm (Fable 2026-07-24): all tiles start ARMED, so the
        # unlit centre seeds on its first hot+O2 tick exactly as before.
        self.ignition_armed = np.ones((3, 3), dtype=bool)


def _ignited(**kw):
    g = _IgnGMap(**kw)
    apply_temperature_ignition(g, o2_frac_ext=X_EXT, ignition_seed=0.1)
    return g.fire[1, 1] > 0


def test_ignition_fires_above_ext():
    """Ambient air (X == 0.21 > X_ext) + hot + fuel -> ignites (one law with
    sustain)."""
    assert _ignited(X=X_AMB)


def test_ignition_blocked_at_or_below_ext():
    """X <= X_ext -> no ignition (a tile cannot ignite into a state the fire
    step would immediately suffocate)."""
    assert not _ignited(X=0.10)
    assert not _ignited(X=X_EXT)


def test_ignition_density_invariant():
    """Thin ambient-composition gas (X == 0.21 at 1/10 density) STILL ignites —
    the trap fix carries to the ignition gate too."""
    assert _ignited(X=X_AMB, density=0.1)


def test_ignition_p1b_no_fuel_no_reignite():
    """P1b: a burnt-out tile (wall_hp <= 0) does NOT re-ignite, even hot and
    oxygenated; a tile with fuel left does."""
    assert not _ignited(X=X_AMB, wall_hp=0.0)      # destroyed -> stays out
    assert _ignited(X=X_AMB, wall_hp=100.0)        # fuel remaining -> lights
