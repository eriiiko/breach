"""#106 P6b -- the arm chain's INTEGER REFERENCE, kept running in CI.

The reference (docs/octopus_integer_reference/chain_ref_q.py) and its gate runner
(chain_ref_q_gates.py) live in docs/ because they are a specification, not engine
code: nothing in src/ or cpp/ imports them, and P6c's C++ solver is held to them
at tol 0. This file keeps them from rotting while the design they specify is still
being built (the precedent: tests/test_ray_engine_v2_integer_reference.py). The
sys.path insertion is the only way to import from a docs folder and is confined
here.

Fast (always): G1's reach (P1 a1-a3), G5 determinism, G6 headroom on short runs,
G8 the static-topology contacts (#106 P6c), the reference's wiring to the kit
twins and the currencies, and two exact properties of the arithmetic (mirror
symmetry, the ballistic identity).

Long (opt-in, ``BREACH_OCTOPUS_LONG_GATES=1``, ~10 min): G2 the dead arm (20 s),
G3 no slide (60 s, plus its floor-rounding pair) and G4 float vs int -- G3 and
G4 also need the P5z traces (``prototypes/octopus_motion/gates3d.py`` writes
out/traces_p5z.npz, gitignored) and skip without them.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_octopus_chain_reference.py -q
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
_REF = ROOT / "docs" / "octopus_integer_reference"
for _p in (ROOT, ROOT / "src", _REF):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import chain_ref_q as R  # noqa: E402
import chain_ref_q_gates as G  # noqa: E402
from simulation import creature_fixed as CF  # noqa: E402
from simulation import fixedpoint_twins as K  # noqa: E402

LONG = os.environ.get("BREACH_OCTOPUS_LONG_GATES") == "1"
TRACES = ROOT / "prototypes" / "octopus_motion" / "out" / "traces_p5z.npz"


def _run(gate, fast=True):
    ok, lines, _ = gate(fast)
    report = "\n".join(lines)
    assert ok, f"\n{gate.__name__}\n{report}"
    return report


def test_the_reference_computes_with_the_kit_twins_and_the_doors_currencies():
    """Property: the reference's rounding, square root and trig ARE the twins
    held equal to fixed_point.h (tests/test_fixedpoint_twins.py), and its
    currencies are creature_fixed's -- one copy of each constant. Breaks if
    the reference grows a private rounding helper or a second copy of a shift
    the door also uses."""
    assert R.K is K
    assert R.sra is K.shr_round_away and R.dra is K.div_round_away
    assert R.msa is K.mul_shr_away and R.isqrt is K.sqrt_q16
    assert R.CHAIN_SHIFT == CF.CHAIN_SHIFT
    assert R.KW_PRE_SHIFT == CF.KW_PRE_SHIFT
    row = R.chain_row(G.creature())
    cdef = G.creature()
    assert row.inv_dt2 == cdef.chain.inv_dt2 and row.atan2_shift == cdef.chain.atan2_shift
    assert row.w_q32 == tuple(cdef.chain.node_inv_mass_q32)


def test_g1_the_reach_passes_p1_a1_to_a3_on_ints():
    """G1 (fast): the mid row's canonical reach (a_ca 0.6, t_act 0.9 s, through
    the checked-in activation table) run in integers passes P1's a1 (bell
    velocity), a2 (straight behind the bend, <= 3 deg) and a3 (no tip
    overshoot), measured by the P5z gates' own code. Breaks if the integer
    chain's bends, frame or activation lookup stop reproducing the reach --
    e.g. a dead band in the passive tip or a wrong atan2 narrowing."""
    print(_run(G.g1_p1))


def test_g5_two_runs_are_byte_identical_and_check_mode_changes_nothing():
    """G5 (fast): the same reach twice -- and once with CHECK on -- leaves the
    same state bytes. Breaks if anything non-integer or order-dependent
    enters the arithmetic, or the headroom checks perturb what they measure."""
    print(_run(G.g5_determinism))


def test_g6_no_int64_or_int32_site_overflows_at_the_three_art_sizes():
    """G6 (fast): the stiffest P1 wave (a_ca 1, t_act 0.5 s) at the 1.80 / 2.52
    / 3.39 m arms runs in CHECK mode with every int64 intermediate below 2^63
    and the int32 state below 2^31. Breaks if a currency or pre-shift change
    pushes a product past the width the C++ will carry it in."""
    print(_run(G.g6_headroom))


def test_g8_the_static_topology_contacts_hold():
    """G8 (fast, #106 P6c): the wall contacts are inert on an open map (state
    bytes identical with and without a topology), an arm thrown flat at a wall
    stops at its face without penetrating or sliding along it, oblique and
    corner throws never end a tick inside a blocked tile, crowns pushed into a
    wall or a footprint stay out, and the new int64 sites fit at 3.39 m.
    Breaks if a contact leaks, slides, or perturbs a run that touches nothing."""
    print(_run(G.g8_walls))


def _reach_world(mirror: bool, ticks: int):
    cdef = G.creature()
    row = R.chain_row(cdef)
    w = R.World(row)
    P0 = G.hooked3d(cdef, 1.5)
    if mirror:
        P0 = P0 * np.array([1.0, -1.0, 1.0])
    w.add_arm(P0, pinned=(0, 1))
    w.set_wave(0, 0.6, 0.9)
    for _ in range(ticks):
        w.tick()
    return w


def test_a_mirrored_reach_is_the_exact_mirror_in_integers():
    """Property (design 6.3, the no-slide rule's root): every signed rounding
    is symmetric, so the reach mirrored through the x-z plane is, count for
    count, the mirror of the reach -- positions and velocities, after 12
    ticks of constraints, frames, atan2 and drag. Breaks if any signed
    correction or narrow rounds toward -inf (a bare >> or floordiv_q) -- the
    bias that would let an arm push its body on rounding alone. (Validated:
    with the reference's three roundings swapped for floors the two runs
    differ within the first tick.)"""
    a, b = _reach_world(False, 12), _reach_world(True, 12)
    for qa, qb, va, vb in zip(a.arms[0].p, b.arms[0].p, a.arms[0].v, b.arms[0].v):
        assert (qa[0], -qa[1], qa[2]) == tuple(qb)
        assert (va[0], -va[1], va[2]) == tuple(vb)


def test_a_free_arm_with_no_force_flies_ballistically_and_keeps_its_velocity():
    """Property: with gravity and drag off and no constraint violated, a tick
    moves every node by its velocity to within one count per substep and
    leaves the velocity EXACTLY unchanged -- the predict's rounding remainder
    is carried, never erased (the reason sub-count accelerations survive).
    Breaks if the velocity rebuild drops the remainder or the drag / gravity
    path perturbs a node they should not touch."""
    cdef = G.creature()
    row = replace(R.chain_row(cdef), g_q24=0, g_dv_q24=0)
    w = R.World(row, floor_on=False)
    n = row.n_nodes
    seg = row.rest0_q24[0] / CF.CHAIN_ONE           # exactly one rest length
    p0 = [[i * seg, 0.0, 1.0] for i in range(n)]
    v0 = [[0.0123, -0.0071, 0.0049]] * n
    w.add_arm(p0, v0=v0, drag_mode=R.DRAG_UNIFORM, tau=1e9)
    w.arms[0].u_q30 = 0
    before = [list(q) for q in w.arms[0].p]
    vq = [list(v) for v in w.arms[0].v]
    w.tick()
    for q, q0, v, vv in zip(w.arms[0].p, before, w.arms[0].v, vq):
        assert v == vv
        for c in range(3):
            assert abs(q[c] - q0[c] - vv[c] * row.n_sub / row.inv_dt) <= row.n_sub


@pytest.mark.skipif(not LONG, reason="long gate: set BREACH_OCTOPUS_LONG_GATES=1")
def test_g2_a_dead_arm_sinks_to_the_deck_and_rests():
    """G2 (long, 20 s): released 0.2 m above the deck, the dead arm touches it,
    rests (|v| < 1 mm/s over the last 5 s) and is never below z = 0 by more
    than a count. Breaks if g_eff is erased by the velocity rebuild (critique
    finding 4) or the floor contact leaks."""
    print(_run(G.g2_dead_arm, fast=False))


@pytest.mark.skipif(not (LONG and TRACES.exists()), reason="long gate + P5z traces")
def test_g3_a_free_animal_does_not_slide():
    """G3 (long, 60 s): the free animal's centre of mass moves <= 1 mm, and the
    same run on FLOOR rounding slides > 10x further (the gate can fail).
    Breaks if any signed correction rounds with a direction."""
    print(_run(G.g3_no_slide, fast=False))


@pytest.mark.skipif(not (LONG and TRACES.exists()), reason="long gate + P5z traces")
def test_g4_the_integer_reach_follows_the_float_within_tolerance():
    """G4 (long): P5z's reach trace replayed through the reference stays
    within 1 cm node RMS of the float (dead arm and crawl reported). Breaks
    if the integer chain stops being the float chain's transcription."""
    print(_run(G.g4_float_vs_int, fast=False))
