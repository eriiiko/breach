"""Gate a — continuous O2->combustion law unit tests
(docs/continuous_o2_law_design_2026-07-24.md §3.a).

The fire O2 sustain factor is now LINEAR in the local O2 MOLE FRACTION
X = Sn_o2/Sn_total over the open 4-neighbours, with an extinction limit:

    o2f = clamp01((X - o2_frac_ext) / (o2_frac_amb - o2_frac_ext))

replacing the old smoothstep(P_min, P_full) on ABSOLUTE n_o2 density. These
tests drive the C++ FireSimulation.step in isolation (a 3x3 grid, one burning
flammable wall at the centre, its four open-air neighbours holding a chosen
O2 composition/density) and assert the LAW's shape via the one observable it
controls: the next-tick intensity.

Design gate a: X = X_amb -> o2f = 1; X <= X_ext -> 0; midpoint linearity;
X_ext = 0 degenerates to X/X_amb; plus the headline property the law exists
for — INVARIANCE under thermal expansion (same composition, different density
-> same burn), the "density trap" fix.

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
X_AMB = 0.21


def _make_sim(o2_frac_ext=X_EXT, o2_frac_amb=X_AMB):
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
def test_o2f_full_at_ambient_grows():
    """X == X_amb -> o2f == 1 -> avail == F -> the (hot, fuelled) fire GROWS."""
    assert _step_once(_make_sim(), X=X_AMB, I0=0.5) > 0.5


def test_o2f_zero_below_ext_decays():
    """X <= X_ext -> o2f == 0 -> avail == 0 -> grow == 0, only die -> DECAYS."""
    # Strictly below the limit, and AT the limit (o2f == 0 at X == X_ext).
    assert _step_once(_make_sim(), X=0.10, I0=0.5) < 0.5
    assert _step_once(_make_sim(), X=X_EXT, I0=0.5) < 0.5


def test_o2f_half_at_midpoint_is_equilibrium():
    """At the span midpoint o2f == 0.5; with I0 == 0.5 that is the logistic
    equilibrium (grow == die), so the intensity barely moves — a direct read
    of the LINEAR law's half value."""
    X_mid = X_EXT + 0.5 * (X_AMB - X_EXT)             # 0.17
    assert abs(_step_once(_make_sim(), X=X_mid, I0=0.5) - 0.5) < 0.02


def test_monotone_increasing_in_X():
    """Next-tick intensity is monotone non-decreasing in the O2 fraction —
    the linear law never inverts."""
    sim = _make_sim()
    vals = [_step_once(sim, X=x, I0=0.5) for x in (0.10, 0.14, 0.17, 0.21)]
    assert vals[0] <= vals[1] <= vals[2] <= vals[3]
    assert vals[0] < vals[3]                          # strictly separated ends


def test_ext_zero_degenerates_to_proportional():
    """o2_frac_ext == 0 -> o2f == X / X_amb (Erik's pure proportional). At
    X == X_amb/2 that is 0.5 -> equilibrium at I0 == 0.5."""
    sim0 = _make_sim(o2_frac_ext=0.0)
    assert abs(_step_once(sim0, X=X_AMB / 2.0, I0=0.5) - 0.5) < 0.02
    assert _step_once(sim0, X=X_AMB, I0=0.5) > 0.5     # full fraction still grows


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


def test_thin_hot_gas_at_ambient_composition_still_burns():
    """A thermally-expanded cell (ambient composition 21/79 but 1/10 the
    density) must burn essentially like dense ambient air — the OLD absolute
    law would have read n_o2 ~= 0.021 as near-starved. Here both GROW and land
    within a hair of each other."""
    dense = _step_once(_make_sim(), X=X_AMB, density=1.0, I0=0.5)
    thin = _step_once(_make_sim(), X=X_AMB, density=0.1, I0=0.5)
    assert dense > 0.5 and thin > 0.5
    assert abs(dense - thin) < 5e-3


def test_vitiation_starves_at_constant_density():
    """The complement: at FIXED density, replacing O2 with inert product
    (dropping the fraction from ambient to below the extinction limit) DOES
    starve the fire — only true vitiation, not low density, kills it."""
    fresh = _step_once(_make_sim(), X=X_AMB, density=1.0, I0=0.5)
    vitiated = _step_once(_make_sim(), X=0.10, density=1.0, I0=0.5)
    assert fresh > 0.5
    assert vitiated < 0.5


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
