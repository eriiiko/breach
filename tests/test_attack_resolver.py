"""The attack-resolution layer (mechanics/06 §5, weapons W2) — unit tests.

Covers the resolver primitives in isolation (the end-to-end march wiring is
tests/test_weapons_march.py):

  - arc classification: front / flank / behind vs the target's synced facing,
    the exact boundary (<= front half-width is front), the arcs-are-data rule
    (a 360/0 slime-blob profile has no flank/behind — front wins first), and
    the standard multipliers from [combat] (1 / 2 / 4);
  - scale_half_away: exact-int crit amounts, round HALF AWAY FROM ZERO
    (4.5 -> 5, 10.5 -> 11, -4.5 -> -5), integer products pass through;
  - cover_exposure_at: reads the materials table's Q16 column through the
    material grid (furniture = the float32-cast 0.55 quantized; air/walls
    ONE); a bare stub gmap without a materials table = ONE (no concealment);
  - roll_exposure / roll_crit: THE DRAW FORM OF RECORD (#29 P3e) —
    ``draws.roll_q16``: one raw step's high 16 bits < a Q16 p — bit-equal to
    a parallel generator, stream advanced by exactly one step;
  - arc_multiplier consumes NO randomness (it feeds the caller's lazy draw).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_attack_resolver.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from simulation import attack_resolver as ar  # noqa: E402
from simulation import unit_fixed as ufx  # noqa: E402
from simulation.environment import EnvironmentProfile, HUMAN_ENVIRONMENT  # noqa: E402
from simulation.materials import MAT_AIR, MAT_FURNITURE, MaterialTable  # noqa: E402


Q = ufx.quantize_scalar
ONE = ufx.FP_ONE


def _target(facing, profile=HUMAN_ENVIRONMENT):
    """A stub target whose synced facing is ``facing`` radians (Q16)."""
    return SimpleNamespace(facing_q=Q(facing), environment=profile)


# ---------------------------------------------------------------------------
# Arc classification (shot_angle is SCREEN convention: y down; facing y up),
# Q16 in and Q16 out since #29 P3e.
# ---------------------------------------------------------------------------
def test_arc_front_flank_behind_standard_values():
    # Shot flying EAST on screen (angle 0) => incoming reversed bearing (y-up)
    # is pi (the shooter is west of the target).
    east_shot = 0
    # Target faces WEST (pi, toward the shooter): delta 0 -> FRONT x1.
    assert ar.arc_multiplier(east_shot, _target(math.pi)) ==         Q(CFG.combat.crit_front_mult) == ONE
    # Target faces EAST (0, away): delta pi -> BEHIND x4.
    assert ar.arc_multiplier(east_shot, _target(0.0)) ==         Q(CFG.combat.crit_behind_mult) == 4 * ONE
    # Target faces NORTH (pi/2): |delta| = 90 deg — outside the 60 deg front
    # half-width, inside the 135 deg behind threshold -> FLANK x2.
    assert ar.arc_multiplier(east_shot, _target(math.pi / 2)) ==         Q(CFG.combat.crit_flank_mult) == 2 * ONE


def test_arc_front_boundary_is_inclusive():
    # |delta| == front_arc/2 (60 deg, Q16) exactly -> still FRONT (<=).
    pi_q, half = Q(math.pi), Q(0.5 * math.radians(120.0))
    t = SimpleNamespace(facing_q=pi_q - half, environment=HUMAN_ENVIRONMENT)
    assert ar.arc_multiplier(0, t) == ONE
    # One count beyond -> flank.
    t.facing_q -= 1
    assert ar.arc_multiplier(0, t) == 2 * ONE


def test_arc_behind_boundary():
    # |delta| == pi - behind_arc/2 (135 deg) exactly -> BEHIND (>=).
    pi_q, half = Q(math.pi), Q(0.5 * math.radians(90.0))
    t = SimpleNamespace(facing_q=pi_q - (pi_q - half),
                        environment=HUMAN_ENVIRONMENT)
    assert ar.arc_multiplier(0, t) == 4 * ONE
    t.facing_q += 1
    assert ar.arc_multiplier(0, t) == 2 * ONE


def test_arc_screen_vs_facing_convention():
    # Shot flying NORTH on screen = angle -pi/2 (y down). In the y-up facing
    # convention the flight bearing is +pi/2; incoming reversed = -pi/2.
    north_shot = Q(-math.pi / 2)
    # Target faces SOUTH (y-up: -pi/2) straight at the shooter -> FRONT.
    assert ar.arc_multiplier(north_shot, _target(-math.pi / 2)) == ONE
    # Target faces NORTH (pi/2) — shot in the back -> BEHIND.
    assert ar.arc_multiplier(north_shot, _target(math.pi / 2)) == 4 * ONE
    # Target faces EAST (0) — side-on -> FLANK.
    assert ar.arc_multiplier(north_shot, _target(0.0)) == 2 * ONE


def test_arcs_are_data_slime_blob_profile():
    """A radially symmetric species ships 360/0: everything is FRONT (the
    front check wins first — no behind to stab), zero special-case code."""
    blob = EnvironmentProfile(front_arc_deg=360.0, behind_arc_deg=0.0)
    for facing in (0.0, math.pi, math.pi / 2, -math.pi / 2, 1.234):
        assert ar.arc_multiplier(0, _target(facing, blob)) == ONE


def test_arc_defaults_for_bare_stub():
    # No environment profile -> human 120/90 defaults.
    bare = SimpleNamespace(facing_q=Q(math.pi))   # faces the shooter
    assert ar.arc_multiplier(0, bare) == ONE


def test_arc_multiplier_consumes_no_rng():
    rng = np.random.default_rng(3)
    state_before = rng.bit_generator.state
    ar.arc_multiplier(0, _target(math.pi))
    assert rng.bit_generator.state == state_before   # pure classification


# ---------------------------------------------------------------------------
# Crit amount: exact ints, round half away from zero
# ---------------------------------------------------------------------------
def test_scale_half_away():
    assert ar.scale_half_away(10, 2.0) == 20      # integer product: exact
    assert ar.scale_half_away(5, 1.5) == 8        # 7.5  -> 8
    assert ar.scale_half_away(7, 1.5) == 11       # 10.5 -> 11 (away from 0)
    assert ar.scale_half_away(3, 1.5) == 5        # 4.5  -> 5
    assert ar.scale_half_away(-5, 1.5) == -8      # -7.5 -> -8 (sign-symmetric)
    assert ar.scale_half_away(-3, 1.5) == -5      # -4.5 -> -5
    assert ar.scale_half_away(2, 3.0) == 6
    assert ar.scale_half_away(0, 4.0) == 0


# ---------------------------------------------------------------------------
# cover_exposure_at
# ---------------------------------------------------------------------------
def test_cover_exposure_reads_the_materials_column():
    table = MaterialTable.from_config()
    gmap = SimpleNamespace(
        materials=table,
        material=np.array([[MAT_AIR, MAT_FURNITURE]], dtype=np.int32),
    )
    assert ar.cover_exposure_at(gmap, 0, 0) == ONE                    # air
    exp = ar.cover_exposure_at(gmap, 0, 1)                            # crate
    # the once-at-load float32 cast, quantized once to Q16 (#29 P3e)
    assert exp == Q(float(np.float32(0.55)))
    assert exp < ONE


def test_cover_exposure_stub_without_table_is_no_concealment():
    gmap = SimpleNamespace(material=np.zeros((2, 2), dtype=np.int32))
    assert ar.cover_exposure_at(gmap, 1, 1) == ONE


# ---------------------------------------------------------------------------
# The draw form of record (#29 P3e: one raw step, an integer compare)
# ---------------------------------------------------------------------------
def test_roll_exposure_draw_form_and_stream():
    from simulation import draws
    p_q = Q(0.55)
    for seed in range(8):
        rng = np.random.default_rng(seed)
        parallel = np.random.default_rng(seed)
        expected = (int(parallel.bit_generator.random_raw()) >> 48) < p_q
        assert ar.roll_exposure(p_q, rng) == expected
        # Exactly ONE raw step consumed — streams line up bit-for-bit.
        assert rng.bit_generator.state == parallel.bit_generator.state
        assert draws.roll_q16(np.random.default_rng(seed), p_q) == expected


def test_roll_crit_draw_form():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        parallel = np.random.default_rng(seed)
        # 0.15 x behind(4.0) = 0.6 in Q16, the product rounded half away.
        p_q = ufx.mul_q16_round(Q(0.15), 4 * ONE)
        expected = (int(parallel.bit_generator.random_raw()) >> 48) < p_q
        assert ar.roll_crit(0.15, 4 * ONE, rng) == expected
        assert rng.bit_generator.state == parallel.bit_generator.state


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
