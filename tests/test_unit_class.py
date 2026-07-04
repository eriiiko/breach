"""Tests for the Unit class foundation pass.

Covers:
- BaseStats fields exist and are constructible from human species
- predefined_unit_attributes is deterministic, Q16.16-grid-aligned, clamped
  (the draft MVN sampler was ingress-banned 2026-07-04 — lenovo_dev_setup §8b)
- Unit("test", 5, 5) constructs cleanly with all new fields populated
- occupied_tiles() returns 9 tiles for a default human (3x3 footprint)
- occupied_tiles() returns 16 for footprint=4
- facing_compass() returns expected string for each cardinal radian
- effective_vitality(unit) returns unit.base_stats.vitality (identity stub)
- is_stat_player_visible(IMAGINATION, unit) = False unless unit.awakened
- is_stat_player_visible(WILL_ORIENTATION, unit) = always False

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_unit_class.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from simulation.generation import predefined_unit_attributes
from simulation.species import HUMAN, get_species, GENERATED_STAT_NAMES, N_GENERATED_STATS
from simulation.stats import (
    BaseStats, StatId,
    effective_vitality, effective_strength, effective_agility,
    is_stat_player_visible,
)
from simulation.unit import Unit, LifeState


# ---------------------------------------------------------------------------
# Species / stat distribution tests
# ---------------------------------------------------------------------------

def test_human_species_registered():
    """get_species("human") returns the HUMAN SpeciesDef."""
    species = get_species("human")
    assert species.id == "human"
    assert species.name == "Human"


def test_base_stats_fields():
    """BaseStats dataclass has all expected fields."""
    bs = BaseStats(
        strength=5.0, agility=5.0, endurance=5.0, vitality=100.0,
        intelligence=5.0, will_strength=5.0,
        imagination=5.0, will_orientation=0.0,
    )
    assert bs.strength == 5.0
    assert bs.vitality == 100.0
    assert bs.will_orientation == 0.0


def _stat_vec(base, mass, base_speed):
    return [
        base.strength, base.agility, base.endurance, base.vitality,
        base.intelligence, base.will_strength,
        base.imagination, base.will_orientation,
        mass, base_speed,
    ]


def test_predefined_unit_attributes_returns_correct_types():
    """predefined_unit_attributes returns (BaseStats, float, float) — no RNG."""
    base, mass, base_speed = predefined_unit_attributes(HUMAN)
    assert isinstance(base, BaseStats)
    assert isinstance(mass, float)
    assert isinstance(base_speed, float)


def test_predefined_unit_attributes_deterministic():
    """Two calls are bit-identical — spawn stats are synced state and must be
    reproducible cross-machine (ingress rule; the MVN draft was not)."""
    a = _stat_vec(*predefined_unit_attributes(HUMAN))
    b = _stat_vec(*predefined_unit_attributes(HUMAN))
    assert a == b


def test_predefined_unit_attributes_on_q16_grid():
    """Every stat is an exact multiple of 1/65536 (ingress door 2: config
    constants quantize onto the Q16.16 grid at the boundary)."""
    vec = _stat_vec(*predefined_unit_attributes(HUMAN))
    for name, v in zip(GENERATED_STAT_NAMES, vec):
        assert v * 65536 == int(v * 65536), (
            f"stat '{name}' = {v!r} is not on the Q16.16 grid"
        )


def test_predefined_unit_attributes_within_clamps():
    """The predefined vector lands within [clamp_min, clamp_max]."""
    sd = HUMAN.stat_dist
    vec = _stat_vec(*predefined_unit_attributes(HUMAN))
    for i, (v, lo, hi) in enumerate(
        zip(vec, sd.clamp_min.tolist(), sd.clamp_max.tolist())
    ):
        name = GENERATED_STAT_NAMES[i]
        assert lo <= v <= hi, (
            f"stat '{name}' out of clamp: {v:.3f} not in [{lo}, {hi}]"
        )


def test_human_vitality_exactly_100():
    """Predefined vitality == the species mean == CFG.marine.hp baseline,
    exactly (100.0 is dyadic — the Q16.16 snap is an identity on it)."""
    base, _, _ = predefined_unit_attributes(HUMAN)
    assert base.vitality == 100.0


def test_predefined_overrides_applied_and_snapped():
    """Named-character overrides land, Q16.16-snapped (door 2)."""
    base, mass, _ = predefined_unit_attributes(
        HUMAN, overrides={"vitality": 87.3, "mass": 90.0})
    assert base.vitality == round(87.3 * 65536) / 65536
    assert base.vitality != 87.3          # 87.3 is not dyadic — snap moved it
    assert mass == 90.0


# ---------------------------------------------------------------------------
# Unit construction
# ---------------------------------------------------------------------------

def test_unit_constructs_cleanly():
    """Unit('test', 5, 5) should build without raising, populating new fields."""
    u = Unit("test", 5, 5)

    # Position
    assert u.x == 5.0
    assert u.y == 5.0
    assert u.tile_x == 5
    assert u.tile_y == 5

    # New foundation fields
    assert u.species_id == "human"
    assert isinstance(u.base_stats, BaseStats)
    assert isinstance(u.mass, float)
    assert isinstance(u.base_speed, float)
    assert u.mass > 0
    assert u.base_speed > 0
    assert u.current_hp > 0
    assert u.life_state is LifeState.ALIVE
    assert u.alive is True
    assert u.awakened is False
    assert u.faction_id == 0
    assert u.environment is not None
    assert u.inventory is not None
    assert isinstance(u.offsets, list)

    # Legacy fields still present
    assert u.team == 0
    assert u.footprint == 3
    assert isinstance(u.orders, list)


def test_unit_has_no_hp_attribute():
    """The old 'hp' field must not exist — it is now 'current_hp'."""
    u = Unit("test", 5, 5)
    assert not hasattr(u, "hp"), "hp field should have been renamed to current_hp"


def test_unit_has_no_max_hp_attribute():
    """max_hp was removed — effective_vitality() replaces it."""
    u = Unit("test", 5, 5)
    assert not hasattr(u, "max_hp"), "max_hp should not exist; use effective_vitality()"


def test_unit_facing_is_float():
    """Facing must be a float (radians), not a string."""
    u = Unit("test", 5, 5)
    assert isinstance(u.facing, float)
    # Default spawn = π/2 (North).
    assert abs(u.facing - math.pi / 2) < 1e-9


# ---------------------------------------------------------------------------
# occupied_tiles and occupies
# ---------------------------------------------------------------------------

def test_occupied_tiles_default_footprint():
    """Default human (footprint=3) occupies exactly 9 tiles."""
    u = Unit("test", 5, 5)
    tiles = u.occupied_tiles()
    assert len(tiles) == 9


def test_occupied_tiles_default_content():
    """Tiles for Unit at (5, 5) with footprint=3 cover the 3×3 block."""
    u = Unit("test", 5, 5)
    tiles = set(u.occupied_tiles())
    expected = {(5 + dx, 5 + dy) for dy in range(3) for dx in range(3)}
    assert tiles == expected


def test_occupied_tiles_footprint4():
    """footprint=4 → 16 tiles."""
    u = Unit("test", 5, 5, footprint=4)
    tiles = u.occupied_tiles()
    assert len(tiles) == 16


def test_occupied_tiles_footprint1():
    """footprint=1 → 1 tile."""
    u = Unit("test", 10, 20, footprint=1)
    tiles = u.occupied_tiles()
    assert len(tiles) == 1
    assert tiles[0] == (10, 20)


def test_occupies_true():
    """occupies() returns True for tiles within the footprint."""
    u = Unit("test", 5, 5)
    assert u.occupies((5, 5))
    assert u.occupies((7, 7))   # (5+2, 5+2)
    assert u.occupies((6, 5))


def test_occupies_false():
    """occupies() returns False for tiles outside the footprint."""
    u = Unit("test", 5, 5)
    assert not u.occupies((4, 5))   # one tile to the left
    assert not u.occupies((8, 5))   # one tile past right edge
    assert not u.occupies((5, 8))


# ---------------------------------------------------------------------------
# facing_compass
# ---------------------------------------------------------------------------

_CARDINAL_CASES = [
    (math.pi / 2,       "N"),   # default spawn
    (0.0,               "E"),
    (math.pi,           "W"),   # or could snap to W/NW — check sector
    (-math.pi / 2,      "S"),
    (math.pi / 4,       "NE"),
    (3 * math.pi / 4,   "NW"),
    (-math.pi / 4,      "SE"),
    (-3 * math.pi / 4,  "SW"),
]


@pytest.mark.parametrize("radians,expected", _CARDINAL_CASES)
def test_facing_compass_cardinals(radians, expected):
    """facing_compass() returns the correct compass label for cardinal angles."""
    u = Unit("test", 5, 5)
    u.facing = radians
    assert u.facing_compass() == expected, (
        f"facing={radians:.4f} rad → expected '{expected}', "
        f"got '{u.facing_compass()}'"
    )


def test_facing_compass_default():
    """Default spawn facing (π/2) maps to 'N'."""
    u = Unit("test", 5, 5)
    assert u.facing_compass() == "N"


def test_facing_compass_full_rotation():
    """Rotating 360° should cycle through all 8 directions."""
    u = Unit("test", 5, 5)
    seen = set()
    for deg in range(0, 360, 10):
        u.facing = math.radians(deg)
        seen.add(u.facing_compass())
    assert seen == {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}, (
        f"Not all 8 directions seen: {seen}"
    )


# ---------------------------------------------------------------------------
# Stat accessors
# ---------------------------------------------------------------------------

def test_effective_vitality_identity():
    """effective_vitality returns base_stats.vitality (modifier-stub identity)."""
    u = Unit("test", 5, 5)
    assert effective_vitality(u) == u.base_stats.vitality


def test_effective_strength_identity():
    u = Unit("test", 5, 5)
    assert effective_strength(u) == u.base_stats.strength


def test_effective_agility_identity():
    u = Unit("test", 5, 5)
    assert effective_agility(u) == u.base_stats.agility


# ---------------------------------------------------------------------------
# Stat visibility rules
# ---------------------------------------------------------------------------

def test_imagination_hidden_by_default():
    """IMAGINATION is not player-visible unless unit.awakened."""
    u = Unit("test", 5, 5)
    assert u.awakened is False
    assert not is_stat_player_visible(StatId.IMAGINATION, u)


def test_imagination_visible_when_awakened():
    """IMAGINATION becomes visible once unit.awakened = True."""
    u = Unit("test", 5, 5)
    u.awakened = True
    assert is_stat_player_visible(StatId.IMAGINATION, u)


def test_will_orientation_always_hidden():
    """WILL_ORIENTATION is never player-visible."""
    u = Unit("test", 5, 5)
    assert not is_stat_player_visible(StatId.WILL_ORIENTATION, u)
    u.awakened = True
    assert not is_stat_player_visible(StatId.WILL_ORIENTATION, u)


def test_player_visible_stats():
    """Standard stats are always visible."""
    u = Unit("test", 5, 5)
    for stat in (StatId.STRENGTH, StatId.AGILITY, StatId.ENDURANCE,
                 StatId.VITALITY, StatId.INTELLIGENCE, StatId.WILL_STRENGTH):
        assert is_stat_player_visible(stat, u), f"{stat} should be visible"


# ---------------------------------------------------------------------------
# LifeState and alive property
# ---------------------------------------------------------------------------

def test_alive_property_true_by_default():
    u = Unit("test", 5, 5)
    assert u.alive is True
    assert u.life_state is LifeState.ALIVE


def test_alive_setter():
    """Setting unit.alive = False transitions life_state to DEAD."""
    u = Unit("test", 5, 5)
    u.alive = False
    assert u.life_state is LifeState.DEAD
    assert u.alive is False


def test_alive_setter_restore():
    """Setting unit.alive = True after death transitions back to ALIVE."""
    u = Unit("test", 5, 5)
    u.alive = False
    u.alive = True
    assert u.life_state is LifeState.ALIVE
    assert u.alive is True


# ---------------------------------------------------------------------------
# Faction id
# ---------------------------------------------------------------------------

def test_faction_id_matches_team():
    """faction_id should equal the team argument at construction."""
    marine = Unit("M", 5, 5, team=0)
    zombie = Unit("Z", 5, 5, team=1)
    assert marine.faction_id == 0
    assert zombie.faction_id == 1


# ---------------------------------------------------------------------------
# Inventory stub
# ---------------------------------------------------------------------------

def test_inventory_stub_present():
    """Every unit has an inventory with current_load() == 0."""
    u = Unit("test", 5, 5)
    assert u.inventory is not None
    assert u.inventory.current_load() == 0.0
    assert isinstance(u.inventory.equipped, list)
    assert isinstance(u.inventory.carried, list)


def test_has_grenade_and_explosive_intact():
    """Legacy has_grenade / has_explosive booleans still exist on Unit."""
    marine = Unit("M", 5, 5, team=0)
    zombie = Unit("Z", 5, 5, team=1)
    # Marines get starting loadout; zombies start with none.
    assert marine.has_grenade >= 0
    assert marine.has_explosive >= 0
    assert zombie.has_grenade == 0
    assert zombie.has_explosive == 0


if __name__ == "__main__":
    import pytest as pt
    pt.main([__file__, "-v"])
