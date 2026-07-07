"""The weapon/ammo/payload data tables (mechanics/03 §4) — the W1 gate.

W1 is a pure re-home: the three shipped weapons move onto table rows with the
SAME NUMBERS and zero behavior change (the golden digest is the empirical
backstop; these tests lock the data layer itself):

  - the three tables load from the real config.toml (rows present, columns
    typed as authored);
  - re-home equivalence LITERALS: k5_carbine == the old [weapons.rifle]
    numbers (10 / 5 / 3.0 / 90 / 1 + the cadence derivation), frag_standard ==
    the old grenade blast (5 / 10.0 / 200 / 60), breach_focus == the old door
    charge (3 / 5.0 / 500 / 60), hand_grenade == the old fuse/throw knobs;
  - the grenade travel-speed representation: the Projectile-consumed
    travel_speed_tiles_per_second is EXACTLY the old 30.0, and the W2
    data-of-record speed_tiles_per_tick is exactly consistent with it at the
    live clock (1.25 * 24 == 30.0, all exact binary floats);
  - WEAPON_ARCHETYPES is the CLOSED six (mechanics/03 §1);
  - validation is loud: bad archetype, orphaned ammo family (both
    directions, with ammo_family="none" exempt), unresolved payload ref,
    unknown dtype, missing required columns;
  - plain dict-of-dicts construction works (the GasTable test-config path);
  - unit.weapon_id: marines carry "k5_carbine", zombies "" (melee stays on
    the ai_zombie path).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_weapons_tables.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import CFG, ticks_from_seconds  # noqa: E402
from simulation.weapons import (  # noqa: E402
    WEAPON_ARCHETYPES, WeaponsTables, get_tables,
)
from simulation.unit import Unit  # noqa: E402


# ---------------------------------------------------------------------------
# Loading from the real config
# ---------------------------------------------------------------------------
def test_loads_from_real_config():
    t = WeaponsTables.from_config()
    assert {"k5_carbine", "hand_grenade", "breach_charge"} <= set(t.weapons.by_name)
    assert {"rifle_556_standard", "grenade_frag", "demo_breach"} <= set(t.ammo.by_name)
    assert {"frag_standard", "breach_focus"} <= set(t.payloads.by_name)
    # The module-level shared tables load the same rows.
    shared = get_tables()
    assert set(shared.weapons.by_name) == set(t.weapons.by_name)


def test_archetypes_are_the_closed_six():
    assert WEAPON_ARCHETYPES == frozenset(
        {"hitscan", "projectile", "lobbed", "placed", "spray", "melee"})


# ---------------------------------------------------------------------------
# Re-home equivalence: the literal old numbers, bit-for-bit
# ---------------------------------------------------------------------------
def test_k5_carbine_equals_the_old_rifle():
    t = get_tables()
    k5 = t.weapons.by_name["k5_carbine"]
    round_556 = t.ammo.by_name["rifle_556_standard"]
    assert k5.archetype == "projectile"
    assert k5.ammo_family == "rifle_556" == round_556.family
    assert round_556.damage == 10                  # was damage_per_bullet
    assert round_556.dtype == "kinetic"
    assert k5.shots_per_trigger == 5               # was bullets_per_burst
    assert k5.spread_deg == 3.0                    # was cone_half_angle_degrees
    assert k5.spread_snap_deg == 3.0               # single cone until W2
    assert k5.range_tiles == 90
    assert k5.ap_cost == 1
    # The cadence derivation moved onto the row — same helper, same input
    # (0.16666667 s, the old burst_interval_seconds), same integer out.
    tps = CFG.clock.ticks_per_second
    assert k5.rof_interval_seconds == 0.16666667
    assert k5.rof_interval_ticks == ticks_from_seconds(0.16666667, tps)


def test_frag_standard_equals_the_old_grenade_blast():
    t = get_tables()
    frag = t.payloads.by_name["frag_standard"]
    assert frag.radius == 5                        # was blast_radius
    assert frag.pressure == 10.0
    assert frag.wall_damage == 200
    assert frag.unit_damage == 60
    # W3 smoke boolean split (the W1 finding of record): BOTH true — the
    # inner clear (data-of-record, inside apply_explosion in v1) AND the
    # live add_explosion_smoke gate.
    assert frag.clear_smoke is True
    assert frag.emit_blast_smoke is True
    assert frag.gas_species == "" and frag.gas_amount == 0.0
    # ...and the grenade round resolves to it.
    assert t.payload_for_ammo("grenade_frag") is frag


def test_breach_focus_equals_the_old_door_explosive():
    t = get_tables()
    breach = t.payloads.by_name["breach_focus"]
    assert breach.radius == 3
    assert breach.pressure == 5.0
    assert breach.wall_damage == 500
    assert breach.unit_damage == 60
    # "rest zero/empty" — the unlisted columns default.
    assert breach.gas_species == "" and breach.gas_amount == 0.0
    assert breach.gas_radius == 0
    assert breach.ignite_radius == 0.0 and breach.ignite_intensity == 0.0
    # W3 smoke boolean split (the W1 finding of record): the door charge
    # carries BOTH booleans true, or it silently loses its smoke the day the
    # executor takes over (mechanics/03 §8 — this was the whole warning).
    assert breach.clear_smoke is True
    assert breach.emit_blast_smoke is True
    assert t.payload_for_ammo("demo_breach") is breach


def test_hand_grenade_row_and_defaults():
    hand_grenade = get_tables().weapons.by_name["hand_grenade"]
    assert hand_grenade.archetype == "lobbed"
    assert hand_grenade.ap_cost == 1
    assert hand_grenade.max_throw_range == 30
    assert hand_grenade.fuse_min_seconds == 0.0
    assert hand_grenade.fuse_max_seconds == 10.0
    assert hand_grenade.fuse_default_seconds == 1.0
    # Unlisted columns take the loader defaults.
    assert hand_grenade.spread_deg == 0.0
    assert hand_grenade.shots_per_trigger == 1
    assert hand_grenade.rof_interval_seconds == 0.0
    assert hand_grenade.rof_interval_ticks == 0    # 0 s = no cadence gate
    assert hand_grenade.mag_size == 0
    assert hand_grenade.crit_chance == 0.0


def test_grenade_travel_speed_representation():
    """The Projectile consumes tiles-per-SECOND; the row keeps that field at
    EXACTLY the old 30.0 (bit-identical update_position arithmetic), and the
    W2 march's tiles-per-TICK twin is exactly consistent at the live clock:
    1.25 * 24 == 30.0 (all exact binary floats — an exact product)."""
    frag_round = get_tables().ammo.by_name["grenade_frag"]
    assert frag_round.travel_speed_tiles_per_second == 30.0
    tps = CFG.clock.ticks_per_second
    assert (frag_round.speed_tiles_per_tick * tps
            == frag_round.travel_speed_tiles_per_second)


def test_breach_charge_row():
    breach_charge = get_tables().weapons.by_name["breach_charge"]
    assert breach_charge.archetype == "placed"
    assert breach_charge.ammo_family == "demo_charge"
    assert breach_charge.ap_cost == 1


# ---------------------------------------------------------------------------
# Validation is loud (dict-built minimal configs — the GasTable test path)
# ---------------------------------------------------------------------------
_WEAPON_OK = {"archetype": "projectile", "ammo_family": "f"}
_AMMO_OK = {"family": "f", "dtype": "kinetic"}


def _build(weapons, ammo, payloads, tps=24):
    return WeaponsTables(weapons, ammo, payloads, tps)


def test_dict_construction_works():
    t = _build({"w": dict(_WEAPON_OK)}, {"a": dict(_AMMO_OK)}, {})
    assert t.weapons.by_name["w"].archetype == "projectile"
    assert t.ammo.by_name["a"].family == "f"
    assert t.ammo.by_name["a"].damage == 0         # defaulted
    # rof derivation on the dict path too.
    t2 = _build({"w": dict(_WEAPON_OK, rof_interval_seconds=0.16666667)},
                {"a": dict(_AMMO_OK)}, {}, tps=24)
    assert t2.weapons.by_name["w"].rof_interval_ticks == 4


def test_bad_archetype_rejected():
    with pytest.raises(ValueError, match="archetype"):
        _build({"w": {"archetype": "beam", "ammo_family": "none"}}, {}, {})


def test_orphan_ammo_family_rejected():
    # No weapon accepts family "g".
    with pytest.raises(ValueError, match="not accepted by any weapon"):
        _build({"w": dict(_WEAPON_OK)},
               {"a": dict(_AMMO_OK), "b": {"family": "g", "dtype": "blast"}},
               {})


def test_weapon_family_without_ammo_rejected():
    with pytest.raises(ValueError, match="has no ammo rows"):
        _build({"w": dict(_WEAPON_OK)}, {}, {})


def test_ammo_family_none_needs_no_ammo():
    # W5: the melee archetype is LIVE — a melee row must now author its
    # strike packet (melee_damage/melee_dtype), loud at load otherwise.
    t = _build({"knife": {"archetype": "melee", "ammo_family": "none",
                          "melee_damage": 35, "melee_dtype": "kinetic"}},
               {}, {})
    assert t.weapons.by_name["knife"].ammo_family == "none"


def test_unresolved_payload_ref_rejected():
    with pytest.raises(ValueError, match="does not resolve"):
        _build({"w": dict(_WEAPON_OK)},
               {"a": dict(_AMMO_OK, payload="ghost")},
               {})


def test_bad_dtype_rejected():
    with pytest.raises(ValueError, match="damage type"):
        _build({"w": dict(_WEAPON_OK)},
               {"a": {"family": "f", "dtype": "sonic"}},
               {})


def test_missing_required_columns_are_loud():
    with pytest.raises(KeyError, match="archetype"):
        _build({"w": {"ammo_family": "none"}}, {}, {})
    with pytest.raises(KeyError, match="family"):
        _build({"w": dict(_WEAPON_OK)}, {"a": {"dtype": "kinetic"}}, {})


# ---------------------------------------------------------------------------
# unit.weapon_id (W1: config-static binding, not in the synced digest)
# ---------------------------------------------------------------------------
def test_unit_weapon_id():
    marine = Unit("M1", x=5, y=5, team=0)
    zombie = Unit("Z1", x=8, y=8, team=1)
    assert marine.weapon_id == "k5_carbine"
    assert zombie.weapon_id == ""


# ---------------------------------------------------------------------------
# W2 data-of-record: the Lance-3 row, the new columns, the derivations
# ---------------------------------------------------------------------------
def test_lance_3_row_equals_the_armory():
    """mechanics/03 §6: hitscan, cell_laser, 0.1°/0.1°, range 60, 1 @ 0.5 s,
    crit 0 (HITSCAN does not crit in v1), mass 4.1, loudness 0.1."""
    t = get_tables()
    lance = t.weapons.by_name["lance_3"]
    cell = t.ammo.by_name["cell_laser_standard"]
    assert lance.archetype == "hitscan"
    assert lance.ammo_family == "cell_laser" == cell.family
    assert lance.spread_deg == 0.1 and lance.spread_snap_deg == 0.1
    assert lance.range_tiles == 60
    assert lance.shots_per_trigger == 1
    assert lance.rof_interval_seconds == 0.5
    assert lance.rof_interval_ticks == ticks_from_seconds(
        0.5, CFG.clock.ticks_per_second)
    assert lance.crit_chance == 0.0
    assert lance.mass_kg == 4.1
    assert lance.loudness == 0.1
    assert cell.damage == 25
    assert cell.dtype == "energy"
    assert cell.ap == 0
    assert cell.wall_damage == 15
    assert t.ammo_for_weapon(lance) is cell


def test_w2_march_columns_on_the_rifle_round():
    """wall_damage (bullet chew) + speed_q16 (the door-2 quantized march
    budget: 96.0 -> exactly 96 << 16)."""
    r556 = get_tables().ammo.by_name["rifle_556_standard"]
    assert r556.wall_damage == 2
    assert r556.speed_q16 == 96 * 65536
    # The march's same-tick guarantee: speed >= range (mechanics/03 §2).
    k5 = get_tables().weapons.by_name["k5_carbine"]
    assert r556.speed_tiles_per_tick >= k5.range_tiles


def test_ammo_for_weapon_resolution():
    t = get_tables()
    assert t.ammo_for_weapon("k5_carbine").name == "rifle_556_standard"
    assert t.ammo_for_weapon("lance_3").name == "cell_laser_standard"
    with pytest.raises(KeyError, match="none"):
        # A melee row feeds on nothing (W5: live melee rows author their
        # strike packet — the load-time validation).
        melee = WeaponsTables(
            {"knife": {"archetype": "melee", "ammo_family": "none",
                       "melee_damage": 35, "melee_dtype": "kinetic"}},
            {}, {}, 24)
        melee.ammo_for_weapon("knife")


def test_cover_exposure_materials_column():
    """Every row authored explicitly: 1.0 everywhere except the
    furniture-class soft cover (0.55, float32-cast once at load)."""
    import numpy as np
    from simulation.materials import MaterialTable
    t = MaterialTable.from_config()
    by_name = dict(zip(t.names, t.cover_exposure.tolist()))
    assert by_name["furniture"] == float(np.float32(0.55))
    for name in ("air", "hull", "wood", "door", "steel", "glass"):
        assert by_name[name] == 1.0


def test_beam_absorb_q16_derivation_twin():
    """The gases-table beam coefficient == quantize(mean of the RGB
    absorption triple), per gas — the documented derivation of record."""
    from simulation import unit_fixed
    from simulation.gases import GasTable
    g = GasTable.from_config()
    for i in range(g.n):
        mean = (float(g.absorption[i, 0]) + float(g.absorption[i, 1])
                + float(g.absorption[i, 2])) / 3.0
        assert g.beam_absorb_q16[i] == unit_fixed.quantize_scalar(mean)


def test_crit_arc_multipliers_config():
    assert CFG.combat.crit_front_mult == 1.0
    assert CFG.combat.crit_flank_mult == 2.0
    assert CFG.combat.crit_behind_mult == 4.0
    # Arc widths are species-profile data (mechanics/06 §5).
    from simulation.environment import HUMAN_ENVIRONMENT
    assert HUMAN_ENVIRONMENT.front_arc_deg == 120.0
    assert HUMAN_ENVIRONMENT.behind_arc_deg == 90.0
