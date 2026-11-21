"""#217 §8 -- the EOS updates pressure at least once per tile of air motion.

docs/eos_checkerboard_design_2026-11-20.md §8 (Erik, 2026-11-21: "5 decisions:
yes to all"). Steps 1-7 of ``EOSSolver::step`` run k in {1, 2, 4, 8} times a
tick at dt_k = dt/k, k chosen once at entry from the k-set's wind, divergence
and blast source and capped at K_MAX; past K_MAX the cap stage's budget
ceiling K_MAX*dx/dt holds; the cap returns as heat the part of its removal
the same sub-tick's kick supplied. The tests read three solver test seams
(``dbg_last_k``, ``dbg_solves_last_step``, ``dbg_subtick_umax``) and the
three switches (``subtick_loop``, ``subtick_ceiling``, ``cap_heat``; all off
== HEAD). They pin no count and no trajectory: every bound is the design's
own (C_P = 1 tile per update, CFL_ADV per substep, the [0.25, 4] pressure
band, the 5 % stored-pressure lag) or the energy identity, exact.

T1  each pressure update sees bounded mass motion        (breaks: loop around
    1a/1d only; the n_sub floor dropped; the ceiling dropped)
T2  no pile and no sink (a frag, a frag beside a body, the studio's legacy
    charges, a hull breach)                              (breaks: k forced to 1)
T3  the stored pressure tracks the air                   (breaks: k forced to 1)
T4  the cap returns only what the kick supplied          (breaks: the cap
    exports everything again; the cap heats a speed that came from outside)
T5  the closure identity is exact in int64 with k > 1    (breaks: a counter
    reset per sub-tick)
T6  legacy switches == HEAD; k = 1 with the defaults == legacy
                                                         (breaks: any edit of
    the k = 1 path; a switch that fails to restore HEAD)
The CUDA twins (T7) are the cuda_*_check gates (cuda_s8a_check,
cuda_batch_invariance_check, cuda_conductor_batch_check, ...).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from level_loader import LevelData  # noqa: E402
from level_loader import load as load_level  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.payloads import execute_payload  # noqa: E402
from simulation.physics_runner import PhysicsRunner  # noqa: E402

Q = 65536
TPS = 24
DT = 1.0 / TPS
C_P = 1.0                  # tiles per pressure update (§8.6.2)
QUANTUM = 1e-3             # one Q16 rounding of the Courant reads, generous
FRAG = (24, 16, 24)        # fire_studio frag: x, y, tick (§8.1.1, §8.9)
#: The old GOLDEN_AGGREGATE (tests/_xarch_perfield_digest.py before §8):
#: with the three legacy switches the scenario must still hash to it (T6(a),
#: test_b6_logic_golden's "sole cause" standard: the old setting reproduces
#: the previous value on the same build).
PRE_S8_GOLDEN = "d5d03433c4894d79922664e14b6a24801d36406ed7c321ad14904c649d2f487b"
LEGACY = dict(subtick_loop=False, subtick_ceiling=False, cap_heat=False)


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------
def _sim(level: str) -> Simulation:
    if level == "range_hall":
        import gen_training_rooms as gtr
        d = Path(tempfile.gettempdir()) / "breach_s8_rooms"
        d.mkdir(exist_ok=True)
        gtr.write(gtr.range_hall(), d)
        lvl = load_level("range_hall", str(d))
    else:
        lvl = load_level(level, str(ROOT / "levels"))
    sim = Simulation(lvl, seed=42, breach_physics=bp, enable_recorder=False)
    sim.set_paused(False)
    return sim


def _detonate(sim, x, y, ammo):
    pay = sim.weapons_tables.payload_for_ammo(ammo)
    execute_payload(sim.gmap, sim.edit_queue, sim.units, y, x, pay, sim.draws,
                    events=[], kind="grenade")


def _kset(g, skip_partial=False):
    """The k-set of §8.6.2 in numpy (the C++ eos_kset_scan's membership):
    gas cells, N >= 5 %, no vacuum / ring cell within 2 tiles (4-neighbour);
    with ``skip_partial`` (#217 §8.17 lever a, the solver's
    ``kset_skip_partial``) no partial-permeability cell (perm < 1)."""
    from scipy.ndimage import binary_dilation
    bnd = g.is_vacuum.copy()
    if getattr(g, "is_ambient", None) is not None:
        bnd |= g.is_ambient
    diamond = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], bool)
    near = binary_dilation(bnd, structure=diamond)
    ks = (~g.solid) & (~bnd) & (~near) & (g._gas_bulk_n_raw() >= 3277)
    if skip_partial:
        ks &= ~(g.dyn_permeability < 1.0)
    return ks


def _identity_terms(sim):
    """test_e1_hot_rail's transcription of the closure identity (§2.8 +
    #217 §8.6.5's e_cap_heat_sum in the EOS group)."""
    eos = sim.physics_runner.eos
    ts = sim.physics_runner.engine.temperature
    comb = sim.physics_runner.combustion
    g = sim.gmap
    return (
        int(eos.e_entry_resync_sum) + int(eos.e_transport_net_sum)
        - int(eos.e_wipe_sum) - int(eos.e_kick_ke_sum)
        + int(eos.e_drag_heat_sum) + int(eos.e_cap_heat_sum)
        - int(eos.e_work_export_sum) + int(eos.e_rail_sum),
        int(ts.e_gas_deposit_sum) + int(ts.e_gas_cond_sum) + int(ts.e_gas_rail_sum),
        -int(comb.e_comb_draw_sum) + int(comb.e_comb_deliver_sum)
        + int(comb.e_comb_heat_sum) + int(comb.e_comb_rail_sum),
        int(g.gas_energy_seam_net()),
        -int(sim.physics_runner.engine.e_water_evac_export_sum),
    )


def _e_acct(g):
    acct = g._gas_energy_accountable()
    return int(g.gas_energy[acct].astype(object).sum())


def _run(level, ticks, det=None, ammo="grenade_frag", eos=None, ring=None, units=()):
    """Run a level, recording per tick what every test below reads.
    ``units``: (x, y) tiles where an idle marine stands from tick 0 (#217
    §8.17: a body in the blast's path)."""
    from simulation.unit import Unit
    sim = _sim(level)
    for i, (ux, uy) in enumerate(units):
        sim.add_unit(Unit(f"s8_body_{i}", x=float(ux), y=float(uy), team=0))
    e = sim.physics_runner.eos
    for k, v in (eos or {}).items():
        setattr(e, k, v)
    e.dbg_record_subtick_umax = True
    g = sim.gmap
    dx = float(g.tile_size_m)
    acct = g._gas_energy_accountable()
    n0 = float(np.median(g._gas_bulk_n_raw()[acct]))
    rows = []
    prev_e, prev_t = _e_acct(g), _identity_terms(sim)
    # detonations: the scripted one, and every timed charge's schedule (its
    # payload lands in slot 9e, so the EOS meets it on the NEXT tick)
    blasts = [det[2]] if det else []
    for ent in getattr(sim.level, "entities", ()) or ():
        if ent.class_name == "timed_charge" and ent.fields.get("enabled", True):
            blasts.append(int(round(float(ent.fields["first_at_s"]) * TPS)) + 1)
    blasts.sort()
    for t in range(ticks):
        if det and t == det[2]:
            _detonate(sim, det[0], det[1], ammo)
        if sim.is_paused():
            sim.set_paused(False)
        sim.step()
        acct = g._gas_energy_accountable()
        n = g._gas_bulk_n_raw().astype(np.float64)
        E = g.gas_energy.astype(np.float64)
        e_now, terms = _e_acct(g), _identity_terms(sim)
        expected = (terms[0] + (terms[1] - prev_t[1]) + (terms[2] - prev_t[2])
                    + (terms[3] - prev_t[3]) + terms[4])
        resid = (e_now - prev_e) - expected
        prev_e, prev_t = e_now, terms
        ks = _kset(g, bool(e.kset_skip_partial))
        body = (g.dyn_permeability < 1.0) & ~g.solid
        r = dict(
            t=t, k=int(e.dbg_last_k), solves=int(e.dbg_solves_last_step),
            n_sub=int(e.dbg_last_n_sub),
            umax_sub=[v / Q for v in e.dbg_subtick_umax],
            nmin=float(n[acct].min() / n0), nmax=float(n[acct].max() / n0),
            band=(float(E[acct].min() / E[acct].mean()),
                  float(E[acct].max() / E[acct].mean())),
            kband=((float(E[ks].min() / E[ks].mean()),
                    float(E[ks].max() / E[ks].mean())) if ks.any() else None),
            p_ratio=float(g.atmosphere[acct].astype(np.float64).mean() / Q
                          / (E[acct].mean() / Q / Q / float(e.T_AMB_K))),
            resid=resid, tmax=int(e.t_max_phys_hits), rail=int(e.e_rail_sum),
            umed=float(np.median(np.hypot(g.wind_x, g.wind_y)[acct]) / Q),
            # #217 §8.17: the partial-permeability (body) cells
            body_n=((float(n[body].min() / n0), float(n[body].max() / n0))
                    if body.any() else None),
            body_cupd=(float(np.hypot(g.wind_x, g.wind_y)[body].max() / Q * DT / dx)
                       if body.any() else 0.0),
            body_in_kset=bool((body & ks).any()),
        )
        if ring:
            x, y = ring
            cells = [(y, x), (y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)]
            r["ring"] = float(min(n[c] for c in cells) / n0)
        rows.append(r)
    return dict(rows=rows, dx=dx, k_max=int(e.K_MAX), cfl=float(e.CFL_ADV),
                blasts=blasts)


@pytest.fixture(scope="module")
def frag_run():
    return _run("fire_studio", 72, det=FRAG, ring=FRAG[:2])


@pytest.fixture(scope="module")
def studio_run():
    return _run("smoke_light_studio", 72)


@pytest.fixture(scope="module")
def breach_run():
    # a C4 against range_hall's west hull: the room vents through the breach
    return _run("range_hall", 240, det=(1, 13, 24), ammo="demo_c4")


def _wind_room(u_ms=60.0, tile=1.0 / 3.0, eos=None):
    """A sealed 0.333 m hull box, uniform wind ``u_ms`` along +x."""
    w, h = 40, 14
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    lvl = LevelData(name="s8_wind_room", version="3", path=Path("."),
                    tilemap=tm, tile_size_m=tile, diffuse_path=Path("."))
    g = GameMap(lvl)
    g.stamp_units([])
    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    for k, v in (eos or {}).items():
        setattr(runner.eos, k, v)
    live = ~(g.solid | g.is_vacuum)
    g.wind_x[live] = int(round(u_ms * Q))
    return g, runner


# ---------------------------------------------------------------------------
# T1 -- each pressure update sees bounded mass motion
# ---------------------------------------------------------------------------
def _t1_violations(rows, dx, k_max, cfl):
    out = []
    for r in rows:
        k = r["k"]
        if r["solves"] != k:
            out.append(f"tick {r['t']}: {r['solves']} solves for k = {k}")
        if len(r["umax_sub"]) != k:
            out.append(f"tick {r['t']}: {len(r['umax_sub'])} sub-tick reads for k = {k}")
        dtk = DT / k
        for j, u in enumerate(r["umax_sub"]):
            c_upd = u * dtk / dx
            c_sub = c_upd / r["n_sub"]
            if c_upd > max(C_P, k_max / k) + QUANTUM:
                out.append(f"tick {r['t']} sub-tick {j}: {c_upd:.3f} tiles per "
                           f"update > {max(C_P, k_max / k):.3f}")
            if c_sub > cfl + QUANTUM:
                out.append(f"tick {r['t']} sub-tick {j}: {c_sub:.3f} tiles per "
                           f"substep > CFL {cfl}")
    return out


def test_each_pressure_update_sees_bounded_mass_motion_after_a_frag(frag_run):
    """PROPERTY (T1, the frag's detonation tick: entry u ~ 0, L1-3's case):
    every tick runs exactly k solves, and the air each sub-tick transports
    (the k-set's max|u| snapshotted at that sub-tick's transport entry, not
    k's own estimate) moves at most max(C_P, K_MAX/k) tiles between two
    pressure updates and at most CFL_ADV tiles per substep.
    BREAKS IF: the loop wraps 1a/1d only (solves 1 != k); the n_sub floor is
    dropped (sub-ticks >= 2 of the detonation tick reach Csub 1.0); the
    ceiling is dropped (a sub-tick above K_MAX/k)."""
    bad = _t1_violations(frag_run["rows"], frag_run["dx"], frag_run["k_max"],
                         frag_run["cfl"])
    assert not bad, "\n".join(bad[:10])
    det = frag_run["rows"][FRAG[2]]
    assert det["k"] > 1, (
        f"the frag's detonation tick ran k = {det['k']}: the scene no longer "
        "exercises the sub-tick loop (the test would be vacuous)")


def test_each_pressure_update_sees_bounded_mass_motion_in_a_fast_wind():
    """PROPERTY (T1, a uniform 60 m/s wind in a sealed 0.333 m room: 7.5
    tiles a tick, so k = K_MAX at entry): as above -- solves == k, at most
    max(C_P, K_MAX/k) tiles per update and CFL_ADV per substep, every
    sub-tick. BREAKS IF: k ignores the wind term, or the sub-tick loop
    wraps transport only."""
    g, runner = _wind_room()
    e = runner.eos
    e.dbg_record_subtick_umax = True
    rows = []
    for t in range(4):
        runner.step(g, DT)
        rows.append(dict(t=t, k=int(e.dbg_last_k), solves=int(e.dbg_solves_last_step),
                         n_sub=int(e.dbg_last_n_sub),
                         umax_sub=[v / Q for v in e.dbg_subtick_umax]))
    bad = _t1_violations(rows, float(g.tile_size_m), int(e.K_MAX), float(e.CFL_ADV))
    assert not bad, "\n".join(bad[:10])
    assert rows[0]["k"] == int(e.K_MAX), f"60 m/s at 0.333 m ran k = {rows[0]['k']}"


def test_the_t1_bound_fails_without_the_ceiling(frag_run):
    """NON-VACUITY of T1: with the ceiling (and so the n_sub floor) switched
    off, the same frag breaks the per-substep bound on its detonation tick --
    the bound is a property the design holds, not one any run meets.
    BREAKS IF: T1's reads stop seeing what the transport moves."""
    run = _run("fire_studio", FRAG[2] + 3, det=FRAG, eos=dict(subtick_ceiling=False))
    bad = _t1_violations(run["rows"], run["dx"], run["k_max"], run["cfl"])
    assert bad, "T1 held with the ceiling off: its reads cannot fail"


# ---------------------------------------------------------------------------
# T2 -- no pile and no sink
# ---------------------------------------------------------------------------
BAND = (0.25, 4.0)
SETTLE = int(round(0.25 * TPS))   # 0.25 s after a detonation


def _outside_settle(t, blasts):
    return all(not (b <= t < b + SETTLE) for b in blasts)


def test_a_frag_leaves_no_sink_and_no_pile(frag_run):
    """PROPERTY (T2, the frag leg, L1-4): the frag's source tile and its
    4-ring keep N >= 0.05 N0 on every tick (emptied cells are the sinks that
    pin the solved P), and no cell piles above 4 N0.
    BREAKS IF: k is forced to 1 (measured before the build: the ring falls to
    0.035 with the ceiling kept, §8.1.1); see the next test."""
    low = [(r["t"], r["ring"]) for r in frag_run["rows"] if r["ring"] < 0.05]
    pile = [(r["t"], r["nmax"]) for r in frag_run["rows"] if r["nmax"] > 4.0]
    assert not low, f"source ring emptied below 0.05 N0: {low[:5]}"
    assert not pile, f"a pile above 4 N0: {pile[:5]}"


def test_the_frag_leg_breaks_with_one_update_per_tick():
    """NON-VACUITY of T2's frag leg (L1-1's mutation: k forced to 1, the
    ceiling kept): the source ring empties below 0.05 N0.
    BREAKS IF: the frag leg's criterion stops discriminating the regime."""
    run = _run("fire_studio", FRAG[2] + 12, det=FRAG, ring=FRAG[:2],
               eos=dict(subtick_loop=False))
    assert min(r["ring"] for r in run["rows"]) < 0.05, (
        "the source ring stayed >= 0.05 N0 with k forced to 1")


#: #217 §8.17 lever a: a marine standing three tiles from the frag, in the
#: blast's path (fire_studio is open air there).
BODY = (FRAG[0] + 3, FRAG[1])


@pytest.fixture(scope="module")
def frag_body_run():
    return _run("fire_studio", 72, det=FRAG, ring=FRAG[:2], units=[BODY])


def test_a_frag_beside_a_body_leaves_no_sink_and_no_pile(frag_body_run):
    """PROPERTY (T2, the body leg; #217 §8.17 lever a, 2026-11-21): with a
    marine standing three tiles from the frag, its footprint (permeability
    0.5) is OUT of the k-set, yet the blast stays as stable as T2's frag leg:
    the source tile and its 4-ring keep N >= 0.05 N0, no cell piles above
    4 N0, the body's own cells stay within [0.05, 4] N0 on every tick, and
    the detonation tick runs k = K_MAX. Non-vacuous: the body's cells are out
    of the k-set and the wind in them crosses more than one tile per tick on
    some tick of the blast (the exclusion hides a fast wind from k, which the
    free air around the body must still carry).
    BREAKS IF: the exclusion takes the blast's own free air out of the k-set
    (a lost k: the ring empties, as with k forced to 1), or the body's cells
    are emptied or piled by air the k-rule no longer reads."""
    rows = frag_body_run["rows"]
    assert all(not r["body_in_kset"] for r in rows if r["body_n"] is not None),         "a partial-permeability cell is in the k-set (lever a is off)"
    assert any(r["body_cupd"] > 1.0 for r in rows),         "the body's cells never carried a wind past one tile per tick (vacuous)"
    low = [(r["t"], r["ring"]) for r in rows if r["ring"] < 0.05]
    pile = [(r["t"], r["nmax"]) for r in rows if r["nmax"] > 4.0]
    body = [(r["t"], r["body_n"]) for r in rows if r["body_n"] is not None
            and not (0.05 <= r["body_n"][0] and r["body_n"][1] <= 4.0)]
    assert not low, f"source ring emptied below 0.05 N0: {low[:5]}"
    assert not pile, f"a pile above 4 N0: {pile[:5]}"
    assert not body, f"the body's cells left [0.05, 4] N0: {body[:5]}"
    assert rows[FRAG[2]]["k"] == frag_body_run["k_max"],         f"the detonation tick ran k = {rows[FRAG[2]]['k']}"


def test_the_body_leg_breaks_with_one_update_per_tick():
    """NON-VACUITY of the body leg (L1-1's mutation, k forced to 1 with the
    ceiling kept): the source ring empties below 0.05 N0 with the body there.
    BREAKS IF: the body leg's criterion stops discriminating the regime."""
    run = _run("fire_studio", FRAG[2] + 12, det=FRAG, ring=FRAG[:2], units=[BODY],
               eos=dict(subtick_loop=False))
    assert min(r["ring"] for r in run["rows"]) < 0.05, (
        "the source ring stayed >= 0.05 N0 with k forced to 1")


def test_the_studio_charges_leave_no_pile_and_no_sink(studio_run):
    """PROPERTY (T2, smoke_light_studio's legacy charges, 3 s): outside the
    0.25 s after each detonation every accountable cell's p* (C*E) stays
    within [0.25, 4] x the room's mean p*, and no cell falls below 0.02 N0
    on any tick. Pressure, not N: the legacy rows' heat makes an isobaric N
    vary 10x (§8.1.1). BREAKS IF: k is forced to 1 (before the build: piles
    of 113 N0 and empty cells, §8.1)."""
    blasts = studio_run["blasts"]
    assert len(blasts) >= 3, f"the studio's three charges did not all fire: {blasts}"
    bad = [(r["t"], r["band"]) for r in studio_run["rows"]
           if r["t"] > blasts[0] and _outside_settle(r["t"], blasts)
           and not (BAND[0] <= r["band"][0] and r["band"][1] <= BAND[1])]
    assert not bad, f"p* outside {BAND} x mean: {bad[:5]}"
    low = [(r["t"], r["nmin"]) for r in studio_run["rows"] if r["nmin"] < 0.02]
    assert not low, f"cells emptied below 0.02 N0: {low[:5]}"


def test_a_hull_breach_leaves_no_pile_in_the_venting_room(breach_run):
    """PROPERTY (T2, the breach leg, L2-2): a C4 against range_hall's hull;
    from 0.25 s after the detonation every k-set cell (N >= 5 %, more than 2
    tiles from the vacuum) keeps p* within [0.25, 4] x the k-set's mean p*
    while the room drains (the band near the mouth is not in the k-set).
    BREAKS IF: one pressure update per tick against a sonic outflow (k
    forced to 1) piles the air against the walls."""
    det = 24
    bad = [(r["t"], r["kband"]) for r in breach_run["rows"]
           if r["t"] >= det + SETTLE and r["kband"] is not None
           and not (BAND[0] <= r["kband"][0] and r["kband"][1] <= BAND[1])]
    assert not bad, f"k-set p* outside {BAND} x mean: {bad[:5]}"
    assert breach_run["rows"][det]["k"] > 1, "the breach tick ran k = 1"


# ---------------------------------------------------------------------------
# T3 -- the stored pressure tracks the air
# ---------------------------------------------------------------------------
def test_the_stored_pressure_tracks_the_air(studio_run):
    """PROPERTY (T3): |mean P - mean p*| / mean p* < 5 % on every tick
    outside the 0.25 s after each detonation; inside that window the ratio
    rises monotonically until it is within 5 %, and stays there. (The legacy
    rows dump their heat into p* before the solve; the solve catches up at
    the ceiling's pace -- a lag, not a sink.) BREAKS IF: k is forced to 1
    with the ceiling kept (measured: P / p* 0.03, staying)."""
    rows = {r["t"]: r for r in studio_run["rows"]}
    blasts = studio_run["blasts"]
    bad = [(t, r["p_ratio"]) for t, r in rows.items()
           if _outside_settle(t, blasts) and abs(r["p_ratio"] - 1.0) >= 0.05]
    assert not bad, f"stored P off the air's p* by >= 5 %: {bad[:5]}"
    for b in blasts:
        ratios = [rows[t]["p_ratio"] for t in range(b, b + SETTLE) if t in rows]
        reached = False
        for a, c in zip(ratios, ratios[1:]):
            if not reached and abs(a - 1.0) >= 0.05:
                assert c >= a - 1e-9, f"blast at {b}: P / p* fell {a:.3f} -> {c:.3f}"
            reached = reached or abs(c - 1.0) < 0.05
            if reached:
                assert abs(c - 1.0) < 0.05, f"blast at {b}: left the 5 % band again"
        assert reached, f"blast at {b}: P / p* never came within 5 %: {ratios}"


# ---------------------------------------------------------------------------
# T4 -- the cap returns only what the kick supplied
# ---------------------------------------------------------------------------
def _kick_scene(u0_ms, p_step, cap_ms, cap_heat):
    """A 9x9 sealed patch at ambient N, one cell's neighbours kicked by a P
    step (or a cell with an entry speed and no gradient), a uniform cap."""
    h = w = 9
    wx = np.zeros((h, w), np.int32)
    wy = np.zeros((h, w), np.int32)
    wx[4, 2] = int(round(u0_ms * Q))
    temp = np.zeros((h, w), np.int32)
    p = np.full((h, w), Q, np.int32)
    p[4, 5] += int(round(p_step * Q))
    gas = np.zeros((3, h, w), np.int32)
    gas[0], gas[1] = int(0.21 * Q), int(0.79 * Q)
    cons = np.array([True, True, False])
    solid = np.zeros((h, w), bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    vac = np.zeros((h, w), bool)
    absorb = np.zeros((h, w), np.float32)
    cap_q = int(round(cap_ms * Q))
    cap2 = np.full((h, w), cap_q * cap_q, np.int64)
    ge = np.full((h, w), int(Q) * (293 * Q), np.int64)
    ge0 = ge.copy()
    res = bp.eos_kick_compression_ref(
        wx, wy, temp, p, gas, cons, solid, vac, absorb, DT, cap2,
        300.0, 1.0 / 3.0, 1.4, 0.0, 1e-3, -292.0, 16000.0, 1000.0,
        k_drag=0.0, k_drag2=0.0, t_amb_k=293.0, gas_energy=ge, cap_heat=cap_heat)
    dE = int((ge - ge0)[~solid].astype(object).sum())
    return dict(dE=dE, drag=int(res[8]), heat=int(res[11]), destroyed=int(res[12]),
                kick=int(res[13]), clamps=int(res[2]))


def test_the_cap_returns_only_what_the_kick_supplied():
    """PROPERTY (T4, ruling 3): over the kick loop, for cells kicked from
    rest past their cap,
      d(sum gas_energy) == -kick + drag + cap_heat,  exactly;
      cap_heat + destroyed == the whole removal (the destroy-all run's);
      cap_heat <= the kick's own debit, and destroyed ~ 0 (the kick supplied
      it all: <= one raw count per kicked cell of rounding).
    BREAKS IF: the cap exports everything again (cap_heat 0, destroyed ==
    removal), or the heat exceeds the kick's own debit (a mint)."""
    on = _kick_scene(0.0, 3.0, 20.0, cap_heat=True)
    off = _kick_scene(0.0, 3.0, 20.0, cap_heat=False)
    assert on["clamps"] > 0, "the scene's kick never reached the cap"
    assert on["dE"] == -on["kick"] + on["drag"] + on["heat"]
    assert off["dE"] == -off["kick"] + off["drag"] and off["heat"] == 0
    assert on["heat"] + on["destroyed"] == off["destroyed"] > 0
    assert 0 < on["heat"] <= on["kick"]
    assert on["destroyed"] <= on["clamps"] * 65536 * 2, on


def test_the_cap_destroys_a_speed_that_came_from_outside_the_kick():
    """PROPERTY (T4, the tripwire): a cell whose ENTRY speed is above the cap
    (a FieldEdit or level-load wind) and that no gradient kicks has nothing
    the kick supplied: its whole removal stays destroyed and counted
    (e_clamp_destroyed_sum > 0, cap_heat == 0). BREAKS IF: the cap heats its
    whole removal (the entry-speed leg of the design's formula is dropped)."""
    r = _kick_scene(80.0, 0.0, 20.0, cap_heat=True)
    assert r["clamps"] > 0
    assert r["heat"] == 0 and r["destroyed"] > 0, r


# ---------------------------------------------------------------------------
# T5 -- the closure identity, exact, with k > 1; the cap heat's rail gate
# ---------------------------------------------------------------------------
def test_the_closure_identity_is_exact_across_sub_ticks(frag_run):
    """PROPERTY (T5): d(sum_accountable gas_energy) over every whole tick ==
    EOS + thermal-solver gas side + combustion + seams (+ the water export),
    exactly in int64, on the frag run whose ticks reach k = 8.
    BREAKS IF: a counter resets per sub-tick instead of per tick, or a
    sub-tick writes gas_energy past every counter."""
    bad = [(r["t"], r["k"], r["resid"]) for r in frag_run["rows"] if r["resid"]]
    assert not bad, f"identity residual on ticks (t, k, resid): {bad[:5]}"
    assert max(r["k"] for r in frag_run["rows"]) == frag_run["k_max"]


def test_the_cap_heat_does_not_ride_a_rail(frag_run):
    """PROPERTY (L1-6's gate, the frag -- no legacy rows): the heat the cap
    returns never drives a cell onto the T_MAX_PHYS rail, and the run's rails
    move no more energy than the same run with the cap heat off.
    BREAKS IF: the cap heats more than the kick took from that cell (a hot
    spot that rails)."""
    off = _run("fire_studio", 72, det=FRAG, eos=dict(cap_heat=False))
    assert frag_run["rows"][-1]["tmax"] == off["rows"][-1]["tmax"] == \
        frag_run["rows"][0]["tmax"], "a T_MAX_PHYS hit on the frag run"
    rails_on = sum(abs(r["rail"]) for r in frag_run["rows"])
    rails_off = sum(abs(r["rail"]) for r in off["rows"])
    assert rails_on <= rails_off, (rails_on, rails_off)


# ---------------------------------------------------------------------------
# T6 -- k = 1 / legacy is HEAD
# ---------------------------------------------------------------------------
def test_the_legacy_switches_reproduce_the_pre_s8_golden():
    """PROPERTY (T6(a), the standing k = 1 guard, L2-6): with the three
    legacy switches off the GOLDEN_AGGREGATE scenario on this build hashes to
    the pre-§8 golden -- the sub-tick loop, the ceiling and the cap heat are
    the SOLE cause of its re-baseline. BREAKS IF: any edit of the k = 1 path
    (a fold, an order, a reset), or a switch that fails to restore HEAD."""
    from field_ab_harness import capture_trajectory, default_scenario_sim
    from field_digest import trajectory_digest
    from _xarch_perfield_digest import N_STEPS

    def legacy():
        sim = default_scenario_sim()
        for k, v in LEGACY.items():
            setattr(sim.physics_runner.eos, k, v)
        return sim
    assert trajectory_digest(capture_trajectory(make_sim=legacy, n_steps=N_STEPS)) \
        == PRE_S8_GOLDEN


def test_the_defaults_are_the_legacy_path_while_k_is_one():
    """PROPERTY (T6(c), the exact identity claim, L1-1): on idle range_hall
    (k = 1 on every tick, winds far below the ceiling, no cap binding) the
    defaults equal the legacy switches bit for bit, every field, every tick.
    BREAKS IF: the default path differs from HEAD where k = 1 and nothing
    binds (e.g. dt_k folded through a second rounding, the floor applied at
    k = 1, the cap heat writing when the cap does not bind)."""
    from field_ab_harness import capture_trajectory, diff_trajectories

    def mk(legacy):
        def f():
            sim = _sim("range_hall")
            if legacy:
                for k, v in LEGACY.items():
                    setattr(sim.physics_runner.eos, k, v)
            return sim
        return f
    a = capture_trajectory(make_sim=mk(False), n_steps=96)
    b = capture_trajectory(make_sim=mk(True), n_steps=96)
    assert not diff_trajectories(a, b, tol=0.0)
