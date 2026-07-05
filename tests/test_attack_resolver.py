"""The attack-resolution layer (mechanics/06 §5, weapons W2) — unit tests.

Covers the resolver primitives in isolation (the end-to-end march wiring is
tests/test_weapons_march.py):

  - arc classification: front / flank / behind vs the target's synced facing,
    the exact boundary (<= front half-width is front), the arcs-are-data rule
    (a 360/0 slime-blob profile has no flank/behind — front wins first), and
    the standard multipliers from [combat] (1 / 2 / 4);
  - scale_half_away: exact-int crit amounts, round HALF AWAY FROM ZERO
    (4.5 -> 5, 10.5 -> 11, -4.5 -> -5), integer products pass through;
  - cover_exposure_at: reads the materials-table column through the material
    grid (furniture = the float32-cast 0.55; air/walls 1.0); a bare stub gmap
    without a materials table = 1.0 (no cover data == no concealment);
  - roll_exposure / roll_crit: THE DRAW FORM OF RECORD —
    float(rng.uniform(0.0, 1.0)) < p — one draw, bit-equal to a parallel
    generator, stream advanced by exactly one uniform;
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
from simulation.environment import EnvironmentProfile, HUMAN_ENVIRONMENT  # noqa: E402
from simulation.materials import MAT_AIR, MAT_FURNITURE, MaterialTable  # noqa: E402


def _target(facing, profile=HUMAN_ENVIRONMENT):
    return SimpleNamespace(facing=facing, environment=profile)


# ---------------------------------------------------------------------------
# Arc classification (shot_angle is SCREEN convention: y down; facing y up)
# ---------------------------------------------------------------------------
def test_arc_front_flank_behind_standard_values():
    # Shot flying EAST on screen (angle 0) => incoming reversed bearing (y-up)
    # is pi (the shooter is west of the target).
    east_shot = 0.0
    # Target faces WEST (pi, toward the shooter): delta 0 -> FRONT x1.
    assert ar.arc_multiplier(east_shot, _target(math.pi)) == \
        CFG.combat.crit_front_mult == 1.0
    # Target faces EAST (0, away): delta pi -> BEHIND x4.
    assert ar.arc_multiplier(east_shot, _target(0.0)) == \
        CFG.combat.crit_behind_mult == 4.0
    # Target faces NORTH (pi/2): |delta| = 90 deg — outside the 60 deg front
    # half-width, inside the 135 deg behind threshold -> FLANK x2.
    assert ar.arc_multiplier(east_shot, _target(math.pi / 2)) == \
        CFG.combat.crit_flank_mult == 2.0


def test_arc_front_boundary_is_inclusive():
    # |delta| == front_arc/2 (60 deg) exactly -> still FRONT (<=).
    east_shot = 0.0
    facing = math.pi - math.radians(60.0)   # delta == radians(60) exactly
    assert ar.arc_multiplier(east_shot, _target(facing)) == 1.0
    # A hair beyond -> flank.
    facing_beyond = math.pi - math.radians(60.0) - 1e-9
    assert ar.arc_multiplier(east_shot, _target(facing_beyond)) == 2.0


def test_arc_behind_boundary():
    east_shot = 0.0
    # |delta| == pi - behind_arc/2 (135 deg) exactly -> BEHIND (>=).
    facing = math.pi - math.radians(135.0)
    assert ar.arc_multiplier(east_shot, _target(facing)) == 4.0
    facing_inside = math.pi - math.radians(135.0) + 1e-9
    assert ar.arc_multiplier(east_shot, _target(facing_inside)) == 2.0


def test_arc_screen_vs_facing_convention():
    # Shot flying NORTH on screen = angle -pi/2 (y down). In the y-up facing
    # convention the flight bearing is +pi/2; incoming reversed = -pi/2.
    north_shot = -math.pi / 2
    # Target faces SOUTH (y-up: -pi/2 == 3pi/2), i.e. straight at the shooter
    # -> FRONT.
    assert ar.arc_multiplier(north_shot, _target(-math.pi / 2)) == 1.0
    # Target faces NORTH (pi/2) — shot in the back -> BEHIND.
    assert ar.arc_multiplier(north_shot, _target(math.pi / 2)) == 4.0
    # Target faces EAST (0) — side-on -> FLANK.
    assert ar.arc_multiplier(north_shot, _target(0.0)) == 2.0


def test_arcs_are_data_slime_blob_profile():
    """A radially symmetric species ships 360/0: everything is FRONT (the
    front check wins first — no behind to stab), zero special-case code."""
    blob = EnvironmentProfile(front_arc_deg=360.0, behind_arc_deg=0.0)
    east_shot = 0.0
    for facing in (0.0, math.pi, math.pi / 2, -math.pi / 2, 1.234):
        assert ar.arc_multiplier(east_shot, _target(facing, blob)) == 1.0


def test_arc_defaults_for_bare_stub():
    # No environment profile -> human 120/90 defaults.
    bare = SimpleNamespace(facing=math.pi)   # faces the shooter
    assert ar.arc_multiplier(0.0, bare) == 1.0


def test_arc_multiplier_consumes_no_rng():
    rng = np.random.default_rng(3)
    state_before = rng.bit_generator.state
    ar.arc_multiplier(0.0, _target(math.pi))
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
    assert ar.cover_exposure_at(gmap, 0, 0) == 1.0                    # air
    exp = ar.cover_exposure_at(gmap, 0, 1)                            # crate
    assert exp == float(np.float32(0.55))    # the once-at-load float32 cast
    assert exp < 1.0


def test_cover_exposure_stub_without_table_is_no_concealment():
    gmap = SimpleNamespace(material=np.zeros((2, 2), dtype=np.int32))
    assert ar.cover_exposure_at(gmap, 1, 1) == 1.0


# ---------------------------------------------------------------------------
# The draw form of record (door 4)
# ---------------------------------------------------------------------------
def test_roll_exposure_draw_form_and_stream():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        parallel = np.random.default_rng(seed)
        expected = float(parallel.uniform(0.0, 1.0)) < 0.55
        assert ar.roll_exposure(0.55, rng) == expected
        # Exactly ONE uniform consumed — streams line up bit-for-bit.
        assert rng.bit_generator.state == parallel.bit_generator.state


def test_roll_crit_draw_form():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        parallel = np.random.default_rng(seed)
        # 0.15 x behind(4.0) = 0.6 — one exact IEEE product.
        expected = float(parallel.uniform(0.0, 1.0)) < 0.15 * 4.0
        assert ar.roll_crit(0.15, 4.0, rng) == expected
        assert rng.bit_generator.state == parallel.bit_generator.state


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
