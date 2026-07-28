"""P4 — vision v1 (onephase_wego design §8).

Pins the model exactly as the design states it: a facing cone of UNLIMITED
range limited only by walls, a short 360 deg awareness radius, team vision as
the union of member cones, fog of war as pure visibility gating, and the two
predicates (discovered / flanked) that fall out of the same cones — plus §9's
"facing determines defense" and §7's computable "largest exposed profile".
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import vision as V  # noqa: E402
from simulation.ruleset import OnePhaseWEGO, TwoPhaseWEGO  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402

EAST = 0.0
NORTH = math.pi / 2
WEST = math.pi
SOUTH = -math.pi / 2


def _level(h=64, w=64):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    return LevelData(name="onephase_vision", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim():
    return Simulation(_level(), seed=11, breach_physics=None,
                      enable_recorder=False, ruleset=OnePhaseWEGO())


def _unit(sim, x, y, team=0, facing=EAST, name=None):
    u = Unit(name or f"u{len(sim.units)}", x=x, y=y, team=team)
    sim.add_unit(u)
    u.facing = facing
    return u


def _wall_column(sim, x, y0, y1):
    sim.gmap.material[y0:y1, x] = 1
    sim.gmap.solid[y0:y1, x] = True
    sim.gmap.obstacles[y0:y1, x] = True


# ---------------------------------------------------------------------------
# The cone: unlimited range, walls the only limit (§8)
# ---------------------------------------------------------------------------
def test_vision_range_is_unlimited():
    """Erik's ruling: a max vision range is unrealistic and disliked. A target
    40 tiles down an empty corridor is seen."""
    sim = _sim()
    m = _unit(sim, 4.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 50.0, 30.0, team=1)
    assert V.can_see(sim, m, z) is True


def test_a_wall_is_the_only_limit():
    sim = _sim()
    m = _unit(sim, 4.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 50.0, 30.0, team=1)
    _wall_column(sim, 20, 0, 64)
    assert V.can_see(sim, m, z) is False


def test_a_target_behind_the_cone_is_not_seen():
    sim = _sim()
    m = _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 6.0, 30.0, team=1)          # due west, well outside the arc
    assert V.can_see(sim, m, z) is False


def test_turning_around_acquires_the_target():
    sim = _sim()
    m = _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 6.0, 30.0, team=1)
    assert V.can_see(sim, m, z) is False
    m.facing = WEST
    assert V.can_see(sim, m, z) is True


def test_the_cone_half_angle_is_the_dial():
    half = CFG.onephase.vision_cone_half_deg
    inside = math.radians(half - 5.0)
    outside = math.radians(half + 5.0)
    # Offsets rotated off due-east by just under / just over the half-angle.
    assert V.within_cone(EAST, half, math.cos(inside), -math.sin(inside))
    assert not V.within_cone(EAST, half, math.cos(outside), -math.sin(outside))


# ---------------------------------------------------------------------------
# The 360 deg awareness radius (§8)
# ---------------------------------------------------------------------------
def test_awareness_radius_sees_behind_you():
    sim = _sim()
    m = _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 28.0, 30.0, team=1)          # just behind, within the radius
    assert V.can_see(sim, m, z) is True


def test_awareness_radius_is_short():
    sim = _sim()
    m = _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    far = CFG.onephase.awareness_radius_tiles + 8
    z = _unit(sim, 30.0 - far, 30.0, team=1)
    assert V.can_see(sim, m, z) is False


def test_awareness_still_needs_a_clear_line():
    sim = _sim()
    m = _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 27.0, 30.0, team=1)
    assert V.can_see(sim, m, z) is True
    _wall_column(sim, 29, 0, 64)
    assert V.can_see(sim, m, z) is False


# ---------------------------------------------------------------------------
# Team vision = union of member cones (§8)
# ---------------------------------------------------------------------------
def test_team_vision_is_the_union():
    sim = _sim()
    blind = _unit(sim, 30.0, 10.0, team=0, facing=NORTH, name="blind")
    spotter = _unit(sim, 30.0, 40.0, team=0, facing=SOUTH, name="spotter")
    z = _unit(sim, 30.0, 50.0, team=1)
    assert V.can_see(sim, blind, z) is False
    assert V.can_see(sim, spotter, z) is True
    assert z.id in sim.visible_enemy_ids(0), "team vision is not the union"


def test_an_unseen_enemy_is_absent_from_team_vision():
    sim = _sim()
    _unit(sim, 30.0, 30.0, team=0, facing=NORTH)
    z = _unit(sim, 30.0, 50.0, team=1)          # behind the only marine
    assert z.id not in sim.visible_enemy_ids(0)


def test_a_dead_spotter_stops_spotting():
    sim = _sim()
    spotter = _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 45.0, 30.0, team=1)
    assert z.id in sim.visible_enemy_ids(0)
    spotter.alive = False
    sim._vision_cache = None
    assert z.id not in sim.visible_enemy_ids(0)


def test_vision_is_symmetric_in_machinery_but_not_in_facing():
    """Both sides run the same model — a zombie facing away is equally blind."""
    sim = _sim()
    m = _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 45.0, 30.0, team=1, facing=EAST)
    assert V.can_see(sim, m, z) is True
    assert V.can_see(sim, z, m) is False, "the zombie is facing away"


# ---------------------------------------------------------------------------
# The per-tick cache
# ---------------------------------------------------------------------------
def test_the_state_is_cached_within_a_tick_and_rebuilt_across_ticks():
    sim = _sim()
    _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    _unit(sim, 45.0, 30.0, team=1)
    first = V.state_for(sim)
    assert V.state_for(sim) is first, "recomputed within one tick"
    sim.set_paused(False)
    sim.step()
    assert V.state_for(sim) is not first
    assert V.state_for(sim).tick == sim.tick


def test_observer_lists_are_deterministically_ordered():
    sim = _sim()
    for i in range(4):
        _unit(sim, 30.0 + i, 30.0, team=0, facing=EAST)
    z = _unit(sim, 48.0, 30.0, team=1)
    obs = V.state_for(sim).observers_of(z.id)
    assert obs == tuple(sorted(obs))


# ---------------------------------------------------------------------------
# Predicates (§8) and defense (§9)
# ---------------------------------------------------------------------------
def test_discovered_is_entering_an_enemys_vision():
    sim = _sim()
    m = _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 45.0, 30.0, team=1, facing=WEST)
    assert V.is_discovered(sim, m) is True      # the zombie is looking at him
    z.facing = EAST
    sim._vision_cache = None
    assert V.is_discovered(sim, m) is False


def test_flanked_is_attacked_from_outside_your_own_arc():
    sim = _sim()
    m = _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    assert V.is_flanked(m, 50.0, 30.0) is False   # shot from the front
    assert V.is_flanked(m, 10.0, 30.0) is True    # shot from behind


def test_facing_determines_defense():
    sim = _sim()
    m = _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    assert V.defense_multiplier(m, 50.0, 30.0) == 1.0
    assert V.defense_multiplier(m, 10.0, 30.0) == CFG.onephase.flank_damage_mult


def test_overwatch_carries_a_harsher_rear_penalty():
    """§9: overwatch buys target control with side blindness — priced here."""
    sim = _sim()
    m = _unit(sim, 30.0, 30.0, team=0, facing=EAST)
    m.overwatch_facing = EAST
    mult = V.defense_multiplier(m, 10.0, 30.0)
    assert mult == CFG.onephase.overwatch_rear_damage_mult
    assert mult > CFG.onephase.flank_damage_mult


# ---------------------------------------------------------------------------
# Largest exposed profile (§7/§9) — "easiest to hit" made computable
# ---------------------------------------------------------------------------
def test_a_target_in_the_open_is_fully_exposed():
    sim = _sim()
    m = _unit(sim, 10.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 40.0, 30.0, team=1)
    assert V.exposed_profile(sim, m, z) == 1.0


def test_a_target_behind_a_wall_is_not_exposed():
    sim = _sim()
    m = _unit(sim, 10.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 40.0, 30.0, team=1)
    _wall_column(sim, 25, 0, 64)
    assert V.exposed_profile(sim, m, z) == 0.0


def test_partial_occlusion_gives_a_partial_profile():
    """A wall stub covering part of the silhouette leaves part of it shootable
    — which is exactly what makes "the largest exposed profile" a usable
    target-priority rule rather than a boolean."""
    sim = _sim()
    m = _unit(sim, 10.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 40.0, 30.0, team=1)
    # A stub right in front of the target, tall enough to cover the upper
    # part of its silhouette. Close to the TARGET is what gives the rays
    # enough angular separation to disagree — a stub out at mid-range would
    # occlude all of them or none.
    _wall_column(sim, 38, 0, 31)
    p = V.exposed_profile(sim, m, z)
    assert 0.0 < p < 1.0
    assert p == pytest.approx(0.6)


def test_a_deeper_stub_exposes_less():
    """The metric is monotone in how much geometry is in the way — which is
    what lets §9 rank "easiest to hit" instead of just answering yes/no."""
    sim = _sim()
    m = _unit(sim, 10.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 40.0, 30.0, team=1)
    _wall_column(sim, 38, 0, 32)
    assert V.exposed_profile(sim, m, z) < 0.6


def test_profile_is_deterministic():
    sim = _sim()
    m = _unit(sim, 10.0, 30.0, team=0, facing=EAST)
    z = _unit(sim, 40.0, 30.0, team=1)
    _wall_column(sim, 38, 0, 31)
    assert V.exposed_profile(sim, m, z) == V.exposed_profile(sim, m, z)


# ---------------------------------------------------------------------------
# Fog of war is gating only, and only under this ruleset
# ---------------------------------------------------------------------------
def test_fog_of_war_is_off_for_the_shipped_rulesets():
    two = Simulation(_level(), seed=11, breach_physics=None,
                     enable_recorder=False, ruleset=TwoPhaseWEGO())
    assert two.ruleset.fog_of_war is False
    assert OnePhaseWEGO().fog_of_war is True


def test_fog_hides_nothing_from_the_simulation_itself():
    """Visibility gating is a RENDER concern (§8): the sim keeps simulating an
    unseen enemy in full, it is simply not drawn."""
    sim = _sim()
    _unit(sim, 30.0, 30.0, team=0, facing=NORTH)
    z = _unit(sim, 30.0, 50.0, team=1)
    assert z.id not in sim.visible_enemy_ids(0)
    assert z in sim.units and z.alive
