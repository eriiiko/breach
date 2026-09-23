"""Issue #7 — vacuum kills fire: the O2 law's PRESSURE factor.

Erik's ruling of 2026-09-23 (docs/fire_vacuum_pressure_factor_brief_2026-09-23.md
§2): the O2 mole-fraction law stays exactly as it was and is MULTIPLIED, at both
of its reads — the fire intensity ODE's `o2f` (cpp/src/fire_simulation.cpp) and
combustion's claim gate `o2f_j` (cpp/src/combustion.cpp) — by

    g = clamp01((p - p_ext) / (p_full - p_ext)),   p_ext 0.1 atm, p_full 0.5 atm,

where p is the engine's MATERIALIZED pressure `atmosphere` over the same air the
fraction reads (the fire's open 4-neighbours; the claim gate's own air cell).
The law lives once, FP_HD, in cpp/src/o2_pressure_factor.h; its edges are
[physics.fire] p_ext_atm / p_full_atm, quantized at load through
atmosphere_fixed.

Every test names the PROPERTY it protects and the CHANGE that must break it
(brief §4). The CPU == CUDA half of the gate (tol 0 on both law sites, plus the
live fire backend) is tests/cuda_o2_pressure_check.py via
tests/test_cuda_o2_pressure.py.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_o2_pressure_factor.py -q
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

import breach_physics as bp                                    # noqa: E402
from config import CFG                                         # noqa: E402
from level_loader import LevelData                             # noqa: E402
from simulation import atmosphere_fixed, fire_fixed, gas_fixed, wall_fixed  # noqa: E402
from simulation.gamemap import GameMap                         # noqa: E402
from simulation.gases import INERT_N2, N_GASES, O2, SMOKE      # noqa: E402
from simulation.materials import (MAT_AIR, MAT_HULL, MAT_WOOD,  # noqa: E402
                                  MaterialTable, quantize_q16)
from simulation.physics_runner import PhysicsRunner            # noqa: E402

DT = 1.0 / 24.0
FP_ONE = atmosphere_fixed.FP_ONE
_TBL = MaterialTable.from_config()
IGN_WOOD_Q16 = int(_TBL.ignition_temp_q16[MAT_WOOD])
FIRE = (4, 4)
NBRS = ((3, 4), (5, 4), (4, 3), (4, 5))


def _live_edges():
    """The two edges exactly as the live engine binds them (Q16.16 atm)."""
    pr = PhysicsRunner(bp)
    return int(pr.fire.params.p_ext_q), int(pr.fire.params.p_full_q)


P_EXT_Q, P_FULL_Q = _live_edges()


def _short_ramp_edges():
    """An edge pair whose BARE ramp stops one count short of FP_ONE at p_full —
    the case o2_pressure_factor.h's explicit `p >= p_full` branch exists for.
    (At the ruled 0.1 / 0.5 atm the span's integer reciprocal happens to round
    UP and the bare ramp lands on FP_ONE exactly, so the ruled edges alone
    cannot tell whether the branch is there.) Found with the kit's own integer
    reciprocal — bp.fp_reciprocal_q16, the function o2_pressure::bake calls —
    scanning p_full over 0.30..0.95 atm at the live p_ext."""
    for hundredths in range(30, 96):
        p_full = atmosphere_fixed.quantize_scalar(hundredths / 100.0)
        span = p_full - P_EXT_Q
        if span > 0 and (span * int(bp.fp_reciprocal_q16(span))) >> 16 < FP_ONE:
            return P_EXT_Q, p_full
    raise AssertionError("no span with a short bare ramp: the kit's reciprocal "
                         "changed — re-derive this fixture")


EDGE_SETS = {"ruled": (P_EXT_Q, P_FULL_Q), "short_ramp": _short_ramp_edges()}


# ---------------------------------------------------------------------------
# Scenario helpers (the live engine, stepped the way the game steps it)
# ---------------------------------------------------------------------------
def _room(hh=9):
    """A hull-walled square room, air inside, one wood fuel tile at FIRE."""
    tm = np.full((hh, hh), MAT_HULL, dtype=np.int32)
    tm[1:hh - 1, 1:hh - 1] = MAT_AIR
    tm[FIRE] = MAT_WOOD
    ld = LevelData(name="o2_pressure_test", version="2", path=Path("."),
                   tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    return GameMap(ld)


def _ignite(gmap, intensity=0.6):
    """Light the wood tile the way CLAUDE.md's ignition rule demands: the flame
    AND heat above the tile's own ignition temperature (a thermal solid, so its
    temperature is its own truth — not a gas-mirror write)."""
    gmap.fire[FIRE] = fire_fixed.quantize_scalar(intensity)
    gmap.temperature[FIRE] = int(IGN_WOOD_Q16 * 1.5)


def _runner(dormant=False):
    """The live PhysicsRunner (config-bound edges), or the same runner with the
    pressure factor switched off at BOTH sites (the pre-#7 law)."""
    pr = PhysicsRunner(bp)
    if dormant:
        pr.fire.params.p_ext_q = 0
        pr.fire.params.p_full_q = 0
        pr.combustion.p_ext_q = 0
        pr.combustion.p_full_q = 0
    return pr


def _step(pr, gmap):
    """PhysicsRunner.step, returning this tick's heat plane BEFORE the per-tick
    clear that Simulation.step performs last (tests/test_eos_p4_combustion.py's
    game-faithful tick): `heat` at the fire tile is then combustion's fuel-bed
    deposit, the claim gate's own output."""
    pr.step(gmap, DT)
    heat = gmap.heat.copy()
    gmap.heat.fill(0)
    gmap.rad_net.fill(0)
    gmap.rad_flux.fill(0)
    return heat


_SYNCED = ("fire", "temperature", "wall_hp", "atmosphere", "wind_x", "wind_y",
           "gas", "gas_energy", "dem_acc", "smoke")


def _assert_same_world(a, b, tick):
    for name in _SYNCED:
        fa, fb = getattr(a, name), getattr(b, name)
        if not np.array_equal(fa, fb):
            idx = int(np.argmax(fa != fb))
            raise AssertionError(
                f"tick {tick}: `{name}` diverged between the live law and the "
                f"pre-#7 law (first flat index {idx}: {fa.flat[idx]} vs "
                f"{fb.flat[idx]})")


def _nbr_mean(field):
    return float(np.mean([int(field[c]) for c in NBRS]))


# ---------------------------------------------------------------------------
# The door: config -> both law sites
# ---------------------------------------------------------------------------
def test_edges_reach_both_law_sites_through_the_pressure_door():
    """PROPERTY: the two [physics.fire] edges reach BOTH reads of the O2 law as
    the SAME integers, quantized once by the pressure field's own Q16 door
    (atmosphere_fixed) — so the fire ODE and the claim gate agree on where a
    fire dies, in the unit of the `atmosphere` plane they are compared against.
    BREAKS IF: an edge is bound onto only one solver, left dormant (0 / 0), or
    not carried onto the atmosphere Q16 scale."""
    pr = PhysicsRunner(bp)
    want_ext = atmosphere_fixed.quantize_scalar(float(CFG.physics.fire.p_ext_atm))
    want_full = atmosphere_fixed.quantize_scalar(float(CFG.physics.fire.p_full_atm))
    assert pr.fire.params.p_ext_q == want_ext == pr.combustion.p_ext_q
    assert pr.fire.params.p_full_q == want_full == pr.combustion.p_full_q
    assert 0 < want_ext < want_full, "the live law must have a real ramp"


def test_an_empty_span_is_refused_at_load(monkeypatch):
    """PROPERTY: a span with p_full_atm <= p_ext_atm fails LOUDLY at the door
    instead of silently turning Erik's linear ramp into a step.
    BREAKS IF: PhysicsRunner's validation of the two keys is removed."""
    monkeypatch.setattr(CFG.physics.fire, "p_full_atm",
                        float(CFG.physics.fire.p_ext_atm))
    with pytest.raises(ValueError, match="p_ext_atm < p_full_atm"):
        PhysicsRunner(bp)


# ---------------------------------------------------------------------------
# The law's shape, read directly off the claim gate
# ---------------------------------------------------------------------------
def _claim_gate_g(p_q, p_ext_q=P_EXT_Q, p_full_q=P_FULL_Q):
    """Read g(p) at the claim gate as an O2 debit.

    One lit wood source beside ONE open air cell. Every other factor of the
    demand is pinned to exactly FP_ONE — burn_rate*dt = 1, I = 1, the fraction
    factor clamps at 1 (pure O2 with the span top lowered to 0.5), hotf clamps
    at hotf_cap = 1 — so on the pre-D1 chained-truncation path (dem_acc None)
    the demand IS g, and with O2 to spare the air cell is debited exactly g."""
    h, w = 1, 3
    gas = np.zeros((N_GASES, h, w), dtype=np.int32)
    gas[O2][0, 1] = gas_fixed.quantize_scalar(2.0)
    solid = np.array([[True, False, True]])
    flammable = np.array([[True, False, False]])
    is_vacuum = np.zeros((h, w), dtype=bool)
    wall_hp = np.zeros((h, w), dtype=np.int32)
    wall_hp[0, 0] = wall_fixed.quantize_scalar(60.0)
    fire = np.zeros((h, w), dtype=np.int32)
    fire[0, 0] = FP_ONE
    ign = np.zeros((h, w), dtype=np.int32)
    ign[0, 0] = IGN_WOOD_Q16
    temperature = np.zeros((h, w), dtype=np.int32)
    temperature[0, 0] = quantize_q16(2000.0)
    atmosphere = np.full((h, w), int(p_q), dtype=np.int32)
    c = bp.CombustionSolver()
    c.burn_rate = 1.0
    c.o2_frac_ext = 0.13
    c.o2_frac_full = 0.5
    c.hotf_cap = 1.0
    c.p_ext_q = int(p_ext_q)
    c.p_full_q = int(p_full_q)
    o2_before = int(gas[O2][0, 1])
    c.step(gas, O2, INERT_N2, SMOKE, temperature, wall_hp, fire, flammable,
           solid, is_vacuum, ign, 1.0, 1.0, 0.01, atmosphere=atmosphere)
    return o2_before - int(gas[O2][0, 1])


@pytest.mark.parametrize("edges", list(EDGE_SETS.values()), ids=list(EDGE_SETS))
def test_ramp_is_linear_between_the_edges_and_exact_at_them(edges):
    """PROPERTY: g is Erik's LINEAR ramp in pressure — exactly 0 at and below
    p_ext, exactly FP_ONE at and above p_full, within 2 counts of the exact
    rational (p - p_ext) / (p_full - p_ext) between, and never decreasing —
    for the ruled edges AND for a span whose bare ramp would stop short.
    BREAKS IF: the ramp is replaced by a step or smoothstep, the span
    reciprocal is wrong (e.g. 1/p_full instead of 1/(p_full - p_ext)), or the
    explicit p >= p_full branch is dropped (the short-ramp case then reads
    65535 at p == p_full)."""
    p_ext, p_full = edges
    span = p_full - p_ext
    for p in (0, 1, p_ext - 1, p_ext):
        assert _claim_gate_g(p, p_ext, p_full) == 0, f"g({p}) must be 0 at/below p_ext"
    for p in (p_full, p_full + 1, FP_ONE, 2 * FP_ONE):
        assert _claim_gate_g(p, p_ext, p_full) == FP_ONE, (
            f"g({p}) must be exactly FP_ONE at/above p_full")
    prev = 0
    for p in range(p_ext + 1, p_full, max(1, span // 97)):
        g = _claim_gate_g(p, p_ext, p_full)
        exact = (p - p_ext) * FP_ONE / span
        assert abs(g - exact) <= 2, f"g({p}) = {g}, linear ramp says {exact:.2f}"
        assert g >= prev, f"g not monotone at p = {p}"
        prev = g
    mid = (p_ext + p_full) // 2
    assert abs(_claim_gate_g(mid, p_ext, p_full) - FP_ONE // 2) <= 2


# ---------------------------------------------------------------------------
# Property 1 — ambient is untouched (brief §4.1)
# ---------------------------------------------------------------------------
def _fire_state(rng, h, w, p_lo_q, p_hi_q):
    """A random fire field: lit fuel, mixed composition incl. starved and
    enriched air, walls, vacuum, wind — every branch of the ODE — with the
    `atmosphere` plane drawn from [p_lo_q, p_hi_q]."""
    flammable = rng.random((h, w)) < 0.55
    is_wall = flammable | (rng.random((h, w)) < 0.15)
    is_vacuum = (~is_wall) & (rng.random((h, w)) < 0.08)
    fire = np.where(flammable & (rng.random((h, w)) < 0.8),
                    fire_fixed.quantize(rng.random((h, w))), 0).astype(np.int32)
    total = rng.random((h, w)) + 0.3
    total[rng.random((h, w)) < 0.08] = 0.0005
    frac = rng.random((h, w)) * 0.35
    frac[rng.random((h, w)) < 0.15] = 0.0
    temps = rng.random((h, w)) * 900.0 + 150.0
    return dict(
        fire=fire,
        atmosphere=rng.integers(p_lo_q, p_hi_q + 1, size=(h, w), dtype=np.int32),
        n_o2=gas_fixed.quantize(frac * total).astype(np.int32),
        n_total=gas_fixed.quantize(total).astype(np.int32),
        smoke=gas_fixed.quantize(rng.random((h, w)) * 0.3).astype(np.int32),
        wall_hp=wall_fixed.quantize(rng.random((h, w)) * 60.0).astype(np.int32),
        temperature=np.array([[quantize_q16(v) for v in row] for row in temps],
                             dtype=np.int32),
        wind_x=atmosphere_fixed.quantize((rng.random((h, w)) * 2 - 1) * 2.0),
        wind_y=atmosphere_fixed.quantize((rng.random((h, w)) * 2 - 1) * 2.0),
        is_wall=is_wall, is_vacuum=is_vacuum, flammable=flammable)


def _fire_step(state, p_ext_q, p_full_q):
    sim = bp.FireSimulation()
    params = sim.params
    params.p_ext_q = int(p_ext_q)
    params.p_full_q = int(p_full_q)
    sim.params = params
    s = {k: np.ascontiguousarray(v.copy()) for k, v in state.items()}
    destroyed = sim.step(s["fire"], s["atmosphere"], s["n_o2"], s["n_total"],
                         s["smoke"], s["wall_hp"], s["temperature"], s["wind_x"],
                         s["wind_y"], s["is_wall"], s["is_vacuum"],
                         s["flammable"], DT)
    return s, [tuple(t) for t in destroyed]


def _comb_state(rng, h, w, p_lo_q, p_hi_q):
    """A random combustion field (lit, cold, choked and fuel-less sources;
    starved to rich air cells) with `atmosphere` drawn from [p_lo_q, p_hi_q]."""
    solid = rng.random((h, w)) < 0.35
    flammable = solid & (rng.random((h, w)) < 0.6)
    is_vacuum = (~solid) & (rng.random((h, w)) < 0.1)
    air = (~solid) & (~is_vacuum)
    gas = np.zeros((N_GASES, h, w), dtype=np.int32)
    gas[O2] = np.where(air, gas_fixed.quantize(rng.random((h, w)) * 0.5), 0)
    gas[INERT_N2] = np.where(air, gas_fixed.quantize_scalar(0.79), 0)
    temps = rng.random((h, w)) * 1400.0 - 100.0
    return dict(
        gas=gas,
        temperature=np.where(flammable, np.array(
            [[quantize_q16(v) for v in row] for row in temps]), 0).astype(np.int32),
        wall_hp=np.where(flammable, wall_fixed.quantize(rng.random((h, w)) * 80.0),
                         0).astype(np.int32),
        fire=np.where(flammable & (rng.random((h, w)) < 0.8),
                      fire_fixed.quantize(rng.random((h, w))), 0).astype(np.int32),
        flammable=flammable, solid=solid, is_vacuum=is_vacuum,
        ignition_temp_q16=np.where(flammable, IGN_WOOD_Q16, 0).astype(np.int32),
        heat=np.zeros((h, w), dtype=np.int32),
        dem_acc=np.zeros((12, h, w), dtype=np.int32),
        atmosphere=rng.integers(p_lo_q, p_hi_q + 1, size=(h, w), dtype=np.int32))


def _comb_step(state, with_atmosphere, edges=(P_EXT_Q, P_FULL_Q)):
    c = bp.CombustionSolver()
    c.p_ext_q, c.p_full_q = (int(e) for e in edges)
    s = {k: np.ascontiguousarray(v.copy()) for k, v in state.items()}
    c.step(s["gas"], O2, INERT_N2, SMOKE, s["temperature"], s["wall_hp"],
           s["fire"], s["flammable"], s["solid"], s["is_vacuum"],
           s["ignition_temp_q16"], DT, 1.0, 0.01,
           heat=s["heat"], dem_acc=s["dem_acc"], draw_r=2, max_claimants=12,
           atmosphere=s["atmosphere"] if with_atmosphere else None)
    return s


@pytest.mark.parametrize("edges", list(EDGE_SETS.values()), ids=list(EDGE_SETS))
def test_at_or_above_p_full_both_reads_are_the_fraction_law_bit_for_bit(edges):
    """PROPERTY: wherever the air is at or above p_full, the pressure factor is
    the EXACT identity at both reads of the law — the fire ODE and the claim
    gate reproduce the pre-#7 fraction law bit for bit (brief §4.1: g == 1
    exactly in Q16), for the ruled edges and for a span whose bare ramp would
    stop short. Covered: random fields at [p_full, 1.5 atm] and a field
    sitting EXACTLY on p_full everywhere.
    BREAKS IF: g at p >= p_full is anything but FP_ONE (e.g. the explicit
    p >= p_full branch is dropped: the short-ramp span then reads 65535 at
    p == p_full), or g is folded in through a rounding that is not the
    identity at FP_ONE."""
    p_ext, p_full = edges
    rng = np.random.default_rng(20260923)
    hi = atmosphere_fixed.quantize_scalar(1.5)
    for trial in range(6):
        st = _fire_state(rng, 17, 19, p_full, hi)
        if trial == 0:
            st["atmosphere"][:] = p_full           # exactly on the edge
        live, d_live = _fire_step(st, p_ext, p_full)
        old, d_old = _fire_step(st, 0, 0)
        for k in ("fire", "smoke", "wall_hp", "temperature"):
            assert np.array_equal(live[k], old[k]), (
                f"fire site, trial {trial}: `{k}` moved above p_full")
        assert d_live == d_old
        cs = _comb_state(rng, 15, 16, p_full, hi)
        if trial == 0:
            cs["atmosphere"][:] = p_full
        c_live = _comb_step(cs, True, edges)
        c_old = _comb_step(cs, False, edges)
        for k in ("gas", "temperature", "wall_hp", "heat", "dem_acc"):
            assert np.array_equal(c_live[k], c_old[k]), (
                f"claim gate, trial {trial}: `{k}` moved above p_full")
    # Non-vacuity: the same fields BELOW p_ext must differ at both sites, or
    # the equalities above proved nothing about the factor being wired.
    st = _fire_state(rng, 17, 19, 0, p_ext)
    assert not np.array_equal(_fire_step(st, p_ext, p_full)[0]["fire"],
                              _fire_step(st, 0, 0)[0]["fire"])
    cs = _comb_state(rng, 15, 16, 0, p_ext)
    assert not np.array_equal(_comb_step(cs, True, edges)["gas"],
                              _comb_step(cs, False, edges)["gas"])


def test_live_fire_at_ambient_pressure_is_untouched():
    """PROPERTY: on the LIVE path (PhysicsRunner, combustion + fire step, the
    config-bound edges) a fire in a sealed room at ambient pressure runs the
    pre-#7 law bit for bit — every synced field, every tick.
    BREAKS IF: the live wiring feeds either site anything but the
    materialized pressure in atm (e.g. the edges bound in another unit), or the
    factor alters the law above p_full."""
    live, old = _room(), _room()
    _ignite(live)
    _ignite(old)
    pr_live, pr_old = _runner(), _runner(dormant=True)
    min_p = None
    o2_start = int(live.gas[O2].sum())
    for t in range(1, 241):
        _step(pr_live, live)
        _step(pr_old, old)
        _assert_same_world(live, old, t)
        p = min(int(live.atmosphere[c]) for c in NBRS)
        min_p = p if min_p is None else min(min_p, p)
    # Non-vacuity: the fire burned (combustion drew O2) the whole way, at a
    # pressure the factor had to evaluate, not skip.
    assert int(live.fire[FIRE]) > fire_fixed.quantize_scalar(0.1)
    assert int(live.gas[O2].sum()) < o2_start
    assert min_p >= P_FULL_Q, f"the room left the full-effect band ({min_p})"


# ---------------------------------------------------------------------------
# Property 2 — hot air keeps its fire: the density-trap guard (brief §4.2)
# ---------------------------------------------------------------------------
def _hot_thin_room(pr):
    """A lit wood tile whose four open neighbours hold THIN, HOT air at ambient
    composition and ambient PRESSURE: a quarter of the ambient moles at four
    times the ambient absolute temperature, so N_total·T_abs — and therefore
    p = C·N·T_abs — equals the ambient cell's. This is thermal expansion at
    constant pressure, built through the sanctioned seams (bulk `gas` written
    by a scenario builder, then `seed_gas_temperature` re-derives E = N·T_abs
    for exactly those cells)."""
    gmap = _room()
    _ignite(gmap)
    n_amb, o2_amb, n2_amb, _pin = gmap.ambient_seed()
    t_amb_raw = pr._eos_t_amb_raw()
    mask = np.zeros(gmap.fire.shape, dtype=bool)
    for c in NBRS:
        mask[c] = True
    gmap.gas[O2][mask] = o2_amb // 4
    gmap.gas[INERT_N2][mask] = n2_amb // 4
    gmap.seed_gas_temperature(mask, 3 * t_amb_raw)   # T_abs = 4·T_amb
    return gmap, n_amb


def test_hot_thin_air_at_ambient_pressure_keeps_its_fire():
    """PROPERTY (the July density-trap fix survives): the factor reads
    PRESSURE, which thermal expansion does not lower. A fire whose neighbour
    air is hot and thin — under half the ambient moles, a density any
    density-based factor would punish — but at ambient pressure burns
    EXACTLY as the fraction law alone has it: every synced field identical to
    the pre-#7 run, every tick, while the thin air is still there.
    BREAKS IF: g's input is switched from the materialized pressure to a
    density (N_total, the absolute n_o2) — or to anything else that thermal
    expansion lowers."""
    pr_live, pr_old = _runner(), _runner(dormant=True)
    live, n_amb = _hot_thin_room(pr_live)
    old, _ = _hot_thin_room(pr_old)
    thin_ticks = 0
    for t in range(1, 49):
        _step(pr_live, live)
        _step(pr_old, old)
        _assert_same_world(live, old, t)
        n_nbr = _nbr_mean(live.gas[O2]) + _nbr_mean(live.gas[INERT_N2])
        if n_nbr < 0.5 * n_amb:                 # thinner than p_full's density
            thin_ticks += 1
            assert _nbr_mean(live.atmosphere) >= P_FULL_Q, (
                f"tick {t}: thin air at {_nbr_mean(live.atmosphere) / FP_ONE:.3f} "
                f"atm — the scenario is not constant-pressure expansion")
    assert thin_ticks >= 1, "the neighbour air never read as thin — vacuous"
    assert int(live.fire[FIRE]) > 0, "the fire died — nothing was sustained"


# ---------------------------------------------------------------------------
# Property 3 — a vented room loses its fire (brief §4.3)
# ---------------------------------------------------------------------------
def test_vented_room_loses_its_fire():
    """PROPERTY: once a breach vents a room below p_ext, a fire in it cannot
    live. At the claim gate combustion stops drawing — the fuel-bed deposit
    (`heat` at the fire tile, combustion's own output) is exactly zero every
    tick from then on; in the ODE the intensity falls strictly every tick until
    it snaps out; and the fire is OUT while the same fire in the same room,
    sealed, is still burning.
    BREAKS IF: the factor is removed (on fire-12 the vented fire GREW to
    I = 1.0 in vacuum), or wired at only one of the two sites (the claim gate
    keeps depositing heat, or the ODE holds the fire at its fraction-law
    equilibrium)."""
    vented, sealed = _room(), _room()
    _ignite(vented)
    _ignite(sealed)
    vented.destroy_wall(0, 4)        # the canonical breach: an edge hull tile
    assert vented.is_vacuum[0, 4]    # joins the space boundary
    pr_v, pr_s = _runner(), _runner()
    room = ~vented.solid & ~vented.is_vacuum
    t_below = None
    prev_i = int(vented.fire[FIRE])
    out_at = None
    for t in range(1, 2401):
        heat_v = _step(pr_v, vented)
        _step(pr_s, sealed)
        i_now = int(vented.fire[FIRE])
        if t_below is None and int(vented.atmosphere[room].max()) <= P_EXT_Q:
            t_below = t
        if t_below is not None and out_at is None:
            assert int(vented.atmosphere[room].max()) <= P_EXT_Q, (
                f"tick {t}: the vented room re-pressurised")
            assert int(heat_v[FIRE]) == 0, (
                f"tick {t}: combustion still feeds the fire tile below p_ext "
                f"(heat {int(heat_v[FIRE])}) — the claim gate is not reading g")
            assert i_now < prev_i, (
                f"tick {t}: intensity did not fall below p_ext "
                f"({prev_i} -> {i_now}) — the ODE is not reading g")
        if i_now == 0 and out_at is None:
            out_at = t
            assert int(sealed.fire[FIRE]) > fire_fixed.quantize_scalar(0.02), (
                "the sealed control died too — this is not the vacuum's doing")
        prev_i = i_now
    assert t_below is not None and t_below <= 5 * 24, (
        f"the breach did not vent the room below p_ext in 5 s (t_below {t_below})")
    assert out_at is not None, "the vented fire was never extinguished"


# ---------------------------------------------------------------------------
# The source and the gather: the factor reads `atmosphere`, over the same air
# ---------------------------------------------------------------------------
def _one_fire(atm_nbr, n_total_nbr, x_nbr=0.21, solid_nbrs=(), vacuum_nbrs=()):
    """A single lit fuel tile in a 5x5 field; its four neighbours carry the
    given pressure / density / composition, some optionally solid or vacuum
    (holding P == 0, as the EOS stores there)."""
    h = w = 5
    z = lambda: np.zeros((h, w), dtype=np.int32)          # noqa: E731
    st = dict(fire=z(), atmosphere=z(), n_o2=z(), n_total=z(), smoke=z(),
              wall_hp=z(), temperature=z(), wind_x=z(), wind_y=z(),
              is_wall=np.zeros((h, w), dtype=bool),
              is_vacuum=np.zeros((h, w), dtype=bool),
              flammable=np.zeros((h, w), dtype=bool))
    st["flammable"][2, 2] = st["is_wall"][2, 2] = True
    st["fire"][2, 2] = fire_fixed.quantize_scalar(0.5)
    st["temperature"][2, 2] = quantize_q16(700.0)
    st["wall_hp"][2, 2] = wall_fixed.quantize_scalar(60.0)
    st["atmosphere"][:] = atmosphere_fixed.quantize_scalar(atm_nbr)
    st["n_total"][:] = gas_fixed.quantize_scalar(n_total_nbr)
    st["n_o2"][:] = gas_fixed.quantize_scalar(x_nbr * n_total_nbr)
    for c in solid_nbrs:
        st["is_wall"][c] = True
        st["atmosphere"][c] = 0
        st["n_total"][c] = st["n_o2"][c] = 0
    for c in vacuum_nbrs:
        st["is_vacuum"][c] = True
        st["atmosphere"][c] = 0
        st["n_total"][c] = st["n_o2"][c] = 0
    st["atmosphere"][2, 2] = 0                               # a solid tile
    return st


def _i_after(st, live):
    out, _ = _fire_step(st, *((P_EXT_Q, P_FULL_Q) if live else (0, 0)))
    return int(out["fire"][2, 2])


def test_the_factor_reads_the_materialized_pressure_over_the_same_air():
    """PROPERTY: g reads the `atmosphere` plane — the engine's one materialized
    pressure — and nothing else, over EXACTLY the air the fraction reads: the
    fire's open 4-neighbours (solid and vacuum neighbours excluded, as in X),
    and the claim gate's own air cell.
    BREAKS IF: g is computed from N_total / n_o2 or a re-derived p* instead of
    the plane, or the fire's pressure gather includes solid/vacuum cells or
    uses another stencil, or the claim gate reads pressure at another cell."""
    # Ambient density and composition but a vacuum's pressure -> the factor
    # bites (the plane decides, not N) ...
    assert _i_after(_one_fire(0.05, 1.0), True) < _i_after(_one_fire(0.05, 1.0), False)
    # ... and thin air at ambient pressure is untouched (N does not decide).
    st = _one_fire(1.0, 0.2)
    assert _i_after(st, True) == _i_after(st, False)
    # The gather: three solid neighbours (P == 0 in the engine) and one open
    # neighbour at 0.6 atm -> the open mean is 0.6 >= p_full, g == 1. A gather
    # that counted the solids would average 0.15 atm and bite.
    solids = ((1, 2), (3, 2), (2, 1))
    st = _one_fire(0.6, 1.0, solid_nbrs=solids)
    assert _i_after(st, True) == _i_after(st, False)
    vac = _one_fire(0.6, 1.0, vacuum_nbrs=((1, 2),))
    assert _i_after(vac, True) == _i_after(vac, False)
    # Claim gate: the pressure read is the air cell's OWN. Everything but the
    # air cell at vacuum pressure -> it still draws exactly g(its own p) = 1.
    assert _claim_gate_g_with_far_vacuum() == FP_ONE


def _claim_gate_g_with_far_vacuum():
    """_claim_gate_g's field, with every cell except the air cell itself at a
    vacuum pressure: only the air cell's own reading may count."""
    h, w = 1, 3
    gas = np.zeros((N_GASES, h, w), dtype=np.int32)
    gas[O2][0, 1] = gas_fixed.quantize_scalar(2.0)
    solid = np.array([[True, False, True]])
    flammable = np.array([[True, False, False]])
    wall_hp = np.zeros((h, w), dtype=np.int32)
    wall_hp[0, 0] = wall_fixed.quantize_scalar(60.0)
    fire = np.zeros((h, w), dtype=np.int32)
    fire[0, 0] = FP_ONE
    ign = np.zeros((h, w), dtype=np.int32)
    ign[0, 0] = IGN_WOOD_Q16
    temperature = np.zeros((h, w), dtype=np.int32)
    temperature[0, 0] = quantize_q16(2000.0)
    atmosphere = np.zeros((h, w), dtype=np.int32)
    atmosphere[0, 1] = FP_ONE
    c = bp.CombustionSolver()
    c.burn_rate = 1.0
    c.o2_frac_ext = 0.13
    c.o2_frac_full = 0.5
    c.hotf_cap = 1.0
    c.p_ext_q = P_EXT_Q
    c.p_full_q = P_FULL_Q
    o2_before = int(gas[O2][0, 1])
    c.step(gas, O2, INERT_N2, SMOKE, temperature, wall_hp, fire, flammable,
           solid, np.zeros((h, w), dtype=bool), ign, 1.0, 1.0, 0.01,
           atmosphere=atmosphere)
    return o2_before - int(gas[O2][0, 1])
