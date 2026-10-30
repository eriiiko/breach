"""#106 P6a -- the creature row, its door, the species join, CREATURE_TEAM
(properties, never snapshots).

Design: docs/octopus_p6_sim_port_design_2026-10-30.md sections 3, 4.1, 4.3,
4.6, 6.1, 6.6. Each test names the property it protects and the change that
must break it. No test pins the row's VALUES (they are data Erik and the lead
will move); a refusal is provoked by mutating a copy of the shipped row.
"""
from __future__ import annotations

import copy
import dataclasses
import math
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

from config import CFG, ticks_from_seconds  # noqa: E402
from simulation import creature_fixed as cf  # noqa: E402
from simulation.ai_zombie import convert_marines_to_zombies  # noqa: E402
from simulation.creatures import (  # noqa: E402
    JITTER, WAVE_DUR, DUR, CreatureTable, build_creature, get_creature,
    scale_row_by_law,
)
from simulation.damage import NEUTRAL_MITIGATION  # noqa: E402
from simulation.environment import HUMAN_ENVIRONMENT  # noqa: E402
from simulation.factions import CREATURE_TEAM, ZOMBIE_TEAM  # noqa: E402
from simulation.ruleset import MatchRules  # noqa: E402
from simulation.species import (  # noqa: E402
    GENERATED_STAT_NAMES, HUMAN, SpeciesDef, _default_3x3_offsets,
    _human_stat_distribution, get_species,
)
from simulation.unit import Unit  # noqa: E402

TPS = int(CFG.clock.ticks_per_second)


def _row() -> dict:
    """A fresh, mutable copy of the SHIPPED [creatures.octopus] row."""
    with open(ROOT / "config.toml", "rb") as f:
        return copy.deepcopy(tomllib.load(f)["creatures"]["octopus"])


def _scaled_row(arm_length_m: float) -> dict:
    """The shipped row rescaled to another arm length by P5z's law
    (creatures.scale_row_by_law, the one place the law is written)."""
    return scale_row_by_law(_row(), arm_length_m)


def _build(row):
    return build_creature("octopus", row, TPS)


# ---------------------------------------------------------------------------
# The door accepts the shipped row and the sizes the art table names
# ---------------------------------------------------------------------------
def test_the_shipped_row_passes_the_door_and_every_bound_has_margin():
    """Property: the shipped octopus row is accepted, and every headroom
    bound it is checked against passes (margin > 1). Breaks if a row value or
    a creature_fixed currency moves the mid octopus over its integer headroom
    -- which is exactly the case the door exists to refuse loudly."""
    c = get_creature("octopus")
    assert c.name == "octopus" and c.version >= 1
    assert len(c.chain.bounds) >= 10
    for b in c.chain.bounds:
        assert b.ok and b.margin > 1.0, (b.name, b.value, b.limit)


@pytest.mark.parametrize("arm_length_m", [1.80, 2.52, 3.39])
def test_the_art_tables_sizes_are_accepted(arm_length_m):
    """Property: the SIZES table's small_preview / mid / big arms (CONTROLS.md,
    #112), scaled from the shipped row by the P5z law, all pass the door.
    Breaks if a headroom bound or currency is tightened until a size the art
    already plans for is refused."""
    _build(_scaled_row(arm_length_m))


@pytest.mark.parametrize("arm_length_m, bound", [
    (0.05, "mass_floor"),            # the tip node is < 1 Q16 count
    (0.24, "W_fit"),                 # the real animal: the bend's W ~ L^-5
    (50.0, "inv_mass_precision"),    # a 4000 t body loses w's precision
    (50.0, "chain_range"),           # ... and the animal outgrows +/-128 m
])
def test_an_over_headroom_size_is_refused_by_the_bound_that_binds(
        arm_length_m, bound):
    """Property: a size the integer chain cannot hold is REFUSED, by name,
    with the arm length in the message -- never accepted by luck. Breaks if
    the headroom check is skipped, or a bound's formula stops scaling the way
    its docstring says (mass L^3, bend W L^-5, inverse mass L^-3, range L)."""
    with pytest.raises(ValueError, match=bound) as exc:
        _build(_scaled_row(arm_length_m))
    assert "creatures.octopus" in str(exc.value)
    assert f"{arm_length_m:.6g} m" in str(exc.value)


def test_a_stiffness_typo_is_refused_by_the_distance_impulse_bound():
    """Property: a stiffness typo (k_dist x1e6) is refused by the bound on the
    distance impulse's numerator k C d/|d| (the integer reference's dist.k_Cd
    site, #106 P6c), by name. Breaks if that bound is dropped or stops
    reading k_dist."""
    r = _row()
    r["tissue"]["k_dist"] *= 1e6
    with pytest.raises(ValueError, match="dist_impulse_fit"):
        _build(r)


def test_the_distance_impulse_bound_refuses_a_big_animal():
    """Property: dist_impulse_fit scales L^4 under the law (k_dist L^3 x the
    longest segment L), so it refuses BIG sizes: the three art sizes pass,
    a 5 m arm is refused by it, by name (#106 P6c). Breaks if the bound stops
    scaling with the segment length or the stiffness."""
    with pytest.raises(ValueError, match="dist_impulse_fit"):
        _build(_scaled_row(5.0))


# ---------------------------------------------------------------------------
# The door's refusals, one by one
# ---------------------------------------------------------------------------
def _del(group, key):
    def f(r):
        del r[group][key]
    return f


def _set(group, key, value):
    def f(r):
        if group is None:
            r[key] = value
        else:
            r[group][key] = value
    return f


REFUSALS = [
    # missing
    (_del("tissue", "k_dist"), r"creatures\.octopus\.tissue\.k_dist: missing"),
    (lambda r: r.pop("idle"), r"creatures\.octopus\.idle: missing"),
    (lambda r: r.pop("version"), r"creatures\.octopus\.version: missing"),
    # unknown (a typo can never silently default)
    (_set("body", "arm_lenght_m", 2.5), r"body: unknown key.*arm_lenght_m"),
    (_set(None, "limbs", {}), r"creatures\.octopus: unknown key.*limbs"),
    # mistyped
    (_set("tissue", "k_dist", True), r"tissue\.k_dist must be a number"),
    (_set("body", "arms", 8.0), r"body\.arms must be an integer"),
    (_set("rules", "armed", 0), r"rules\.armed must be a bool"),
    (_set("grab", "prop_at_m", [1.0]), r"grab\.prop_at_m must be a list of 2"),
    (_set("rules", "blocks", "arms"), r"rules\.blocks must be one of"),
    # negative / zero where the physics cannot take it
    (_set("tissue", "k_bend", -1.0), r"tissue\.k_bend must be >= 0"),
    (_set("body", "arm_length_m", -2.52), r"body\.arm_length_m must be > 0"),
    (_set("body", "body_mass_kg", 0.0), r"body\.body_mass_kg must be > 0"),
    (_set("crawl", "push_s", -0.5), r"crawl\.push_s must be > 0"),
    (_set("idle", "hold_jitter", -0.1), r"idle\.hold_jitter must be >= 0"),
    (_set("tissue", "bend_damp_s", -0.001), r"tissue\.bend_damp_s must be >= 0"),
    # fractions and cosines
    (_set("crawl", "p_recruit", 1.5), r"crawl\.p_recruit is a fraction"),
    (_set("tissue", "k_passive_frac", -0.01), r"k_passive_frac is a fraction"),
    (_set("crawl", "s_min", 1.2), r"crawl\.s_min is a cosine"),
    # structural minima
    (_set("body", "nodes_per_arm", 2), r"body\.nodes_per_arm must be >= 3"),
    (_set("body", "arms", 0), r"body\.arms must be >= 1"),
    (_set("solver", "substeps", 0), r"solver\.substeps must be >= 1"),
    (_set("solver", "passes", 0), r"solver\.passes must be >= 1"),
    (_set("body", "radius_tip_m", 0.1), r"radius_tip_m .* above radius_base_m"),
    # indices
    (_set("crawl", "anchor_nodes", [5, 6, 18]),
     r"crawl\.anchor_nodes\[2\] = 18 is outside the node range"),
    (_set("grab", "grab_nodes", [-1, 3]),
     r"grab\.grab_nodes\[0\] = -1 is outside the node range"),
    (_set("grab", "grab_nodes", [11, 3]), r"grab\.grab_nodes must be strictly"),
    (_set("body", "arm_angles_deg", [0.0] * 7),
     r"arm_angles_deg has 7 angles for 8 arms"),
    (_set("body", "front_pairs", [0, 8]),
     r"body\.front_pairs\[1\] = 8 is not an arm index"),
    (_set("body", "back_pairs", [-1, 4]),
     r"body\.back_pairs\[0\] = -1 is not an arm index"),
    (_set("grab", "arm", 8), r"grab\.arm = 8 is not an arm index"),
    (_set("crawl", "max_stance", 9), r"crawl\.max_stance = 9 is above"),
]


@pytest.mark.parametrize("mutate, match", REFUSALS,
                         ids=[m for _, m in REFUSALS])
def test_the_door_refuses(mutate, match):
    """Property: each malformed row is refused with a ValueError naming the
    key path. Breaks if that check is removed or its message stops naming the
    key (a refusal that does not say WHERE is a bug hunt)."""
    r = _row()
    mutate(r)
    with pytest.raises(ValueError, match=match):
        _build(r)


def test_an_empty_creatures_section_builds_an_empty_table():
    """Property: no [creatures.*] rows = an empty table, not an error (a
    config without creatures stays legal). Breaks if the table demands a
    row."""
    assert CreatureTable(None, TPS).by_name == {}
    with pytest.raises(KeyError, match="no \\[creatures.kraken\\] row"):
        CreatureTable(None, TPS).get("kraken")


# ---------------------------------------------------------------------------
# Quantized once
# ---------------------------------------------------------------------------
def test_every_duration_is_quantized_once_beside_its_seconds():
    """Property: every duration key has an int ``*_ticks`` twin equal to
    ticks_from_seconds(seconds, tps) (a wave travel time also ``*_substeps``
    at tps x substeps), the seconds are kept, and a ZERO jitter stays 0 ticks.
    Breaks if a consumer is left to re-quantize, a duration slips through
    unquantized, or a zero jitter becomes one tick of jitter."""
    c = get_creature("octopus")
    seen = 0
    for gname in ("reach", "crawl", "swim", "grab", "idle"):
        grp = getattr(c, gname)
        for f in dataclasses.fields(grp):
            kind = f.metadata.get("kind")
            if kind not in (DUR, WAVE_DUR, JITTER):
                continue
            sec = getattr(grp, f.name)
            stem = f.name if kind == JITTER else f.name[:-2]
            ticks = getattr(grp, f"{stem}_ticks")
            assert isinstance(ticks, int) and not isinstance(ticks, bool)
            assert ticks == ticks_from_seconds(sec, TPS), (gname, f.name)
            if kind == WAVE_DUR:
                sub = getattr(grp, f"{stem}_substeps")
                assert sub == ticks_from_seconds(
                    sec, TPS * c.solver.substeps)
            seen += 1
    assert seen >= 20
    r = _row()
    r["crawl"]["push_jitter"] = 0.0
    assert _build(r).crawl.push_jitter_ticks == 0
    # the damping TIME CONSTANT is never ticks: it is gamma = tau / dt_sub
    assert c.tissue.bend_damp_s == pytest.approx(_row()["tissue"]["bend_damp_s"])
    assert c.chain.bend_gamma_q16 == round(
        c.tissue.bend_damp_s * TPS * c.solver.substeps * 65536)
    assert c.chain.inv_dt2 == (TPS * c.solver.substeps) ** 2


def test_the_chain_masses_are_the_quantized_prototype_masses():
    """Property: node masses are the prototype's (P1-P5z) mass law quantized
    to Q16 once, and each w is the exact integer inverse of its quantized
    mass. Breaks if the door and the integer reference derive masses two ways,
    or w is computed from a float."""
    c = get_creature("octopus")
    b = c.body
    ms = cf.node_masses_kg(b.nodes_per_arm, b.arm_length_m, b.radius_base_m,
                           b.radius_tip_m, b.tissue_density)
    assert c.chain.node_mass_q16 == tuple(cf.q16_from_kg(m) for m in ms)
    for m_q, w_q in zip(c.chain.node_mass_q16, c.chain.node_inv_mass_q32):
        assert w_q == (1 << 48) // m_q
    # an arm's total mass is its volume x density (half-segments sum back)
    assert sum(ms) == pytest.approx(sum(
        b.tissue_density * math.pi * (0.5 * (r0 + r1)) ** 2
        * b.arm_length_m / (b.nodes_per_arm - 1)
        for r0, r1 in zip(cf.node_radii_m(b.nodes_per_arm, b.radius_base_m,
                                          b.radius_tip_m)[:-1],
                          cf.node_radii_m(b.nodes_per_arm, b.radius_base_m,
                                          b.radius_tip_m)[1:])))


def test_the_row_hash_moves_with_any_value_and_only_then():
    """Property: row_hash is a function of the row's content -- equal rows
    hash equal, a changed value changes it. Breaks if the hash ignores a
    group or depends on something outside the row."""
    a, b = _build(_row()), _build(_row())
    assert a.row_hash == b.row_hash
    r = _row()
    r["idle"]["p_start"] = 0.16
    assert _build(r).row_hash != a.row_hash


# ---------------------------------------------------------------------------
# Species join
# ---------------------------------------------------------------------------
def test_the_octopus_species_joins_its_creature_row():
    """Property: the octopus species is joined to its row by
    SpeciesDef.creature and READS its mass, footprint and rules from it.
    Breaks if the species restates a creature number instead of reading it."""
    sp = get_species("octopus")
    row = get_creature(sp.creature)
    i = GENERATED_STAT_NAMES.index("mass")
    assert sp.stat_dist.mean[i] == row.body.body_mass_kg
    assert sp.stat_dist.clamp_min[i] <= row.body.body_mass_kg \
        <= sp.stat_dist.clamp_max[i]
    assert len(sp.default_offsets) == row.body.footprint ** 2
    assert sp.can_become_zombie == row.rules.can_become_zombie
    assert sp.armed == row.rules.armed
    u = Unit("O", x=5, y=5, team=CREATURE_TEAM, species_id="octopus",
             footprint=row.body.footprint)
    assert u.mass == pytest.approx(row.body.body_mass_kg, abs=1e-4)


def test_the_human_species_is_unchanged_by_the_join():
    """Property: the human row is exactly what its pre-P6a constructor call
    builds -- creature None, armed, convertible. Breaks if a new SpeciesDef
    field's default differs from what the human had, or the human row is
    edited by the join."""
    pre_p6a = SpeciesDef(
        id="human", name="Human", stat_dist=HUMAN.stat_dist,
        default_offsets=_default_3x3_offsets(),
        environment=HUMAN_ENVIRONMENT, mitigation=NEUTRAL_MITIGATION,
        can_become_zombie=True, nn_intelligence_tier=2)
    assert pre_p6a == HUMAN
    assert HUMAN.creature is None and HUMAN.armed and HUMAN.can_become_zombie
    ref = _human_stat_distribution()
    for name in ("mean", "stddev", "correlation", "clamp_min", "clamp_max"):
        assert (getattr(HUMAN.stat_dist, name) == getattr(ref, name)).all()


def test_a_bite_never_converts_a_species_that_cannot_become_zombie():
    """Property: convert_marines_to_zombies honours
    species.can_become_zombie -- a dead-by-zombie octopus stays a dead
    creature on its team; a dead-by-zombie human still converts. Breaks if
    the conversion goes back to checking killed_by_zombie alone (the bug), or
    the fix stops humans converting."""
    octo = Unit("O", x=5, y=5, team=CREATURE_TEAM, species_id="octopus")
    man = Unit("M", x=5, y=5, team=0)
    for u in (octo, man):
        u.alive = False
        u.killed_by_zombie = True
    convert_marines_to_zombies([octo, man])
    assert not octo.is_zombie and not octo.alive and octo.team == CREATURE_TEAM
    assert man.is_zombie and man.alive and man.team == ZOMBIE_TEAM


def test_an_unarmed_species_binds_no_kit_even_when_offered_one():
    """Property: a species whose data says armed = false binds NO loadout
    (the zombie's no-kit path: weapon_id "", no grenades, empty loadout and
    belt), even when the caller names a loadout row. Breaks if Unit.__init__
    arms every non-zombie again."""
    u = Unit("O", x=5, y=5, team=CREATURE_TEAM, species_id="octopus",
             loadout_name=CFG.marine.loadout)
    assert get_species("octopus").armed is False
    assert u.loadout_name is None and u.weapon_id == ""
    assert u.has_grenade == 0 and u.has_explosive == 0
    assert u.loadout == [] and u.belt == []
    assert not u.is_zombie


def test_an_unarmed_creature_never_shoots_and_a_tick_with_it_runs():
    """Property: slot 4 tolerates a unit with no kit -- an octopus given a
    move-attack order next to a marine fires nothing and the ticks run; the
    marine, a different team, does fire on it (CREATURE_TEAM is hostile).
    Breaks if the shooting path stops skipping weapon_id "" for a
    non-zombie, or CREATURE_TEAM stops being an enemy of marine teams."""
    import breach_physics as bp
    from level_loader import load
    from simulation import Simulation
    from simulation.events import ShotFiredEvent
    from simulation.orders import ORDER_MOVE_ATTACK, Order

    sim = Simulation(load("playground"), seed=7, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    level = load("playground")
    s = next(sp for sp in level.spawns if sp.team != ZOMBIE_TEAM)
    ax, ay = int(s.x), int(s.y)
    found = None
    for d in range(5, 10):
        for bx, by in ((ax + d, ay), (ax - d, ay), (ax, ay + d), (ax, ay - d)):
            if g.is_passable_block(by, bx, 3) and g.has_los(
                    ay + 1, ax + 1, by + 1, bx + 1):
                found = (bx, by)
                break
        if found:
            break
    assert found, "no open footprint in sight of the marine spawn"
    bx, by = found
    marine = Unit("M", x=ax, y=ay, team=0)
    octo = Unit("O", x=bx, y=by, team=CREATURE_TEAM, species_id="octopus")
    sim.add_unit(marine)
    sim.add_unit(octo)
    marine.face_towards(float(bx), float(by))
    octo.face_towards(float(ax), float(ay))
    for u, (tx, ty) in ((marine, (ax, ay)), (octo, (bx, by))):
        sim.apply_action(u.id, Order(ORDER_MOVE_ATTACK, target_fx=tx,
                                     target_fy=ty, phase=0))
    sim.set_paused(False)
    shooters = set()
    for _ in range(int(CFG.clock.ticks_per_phase)):
        sim.step()
        shooters |= {e.unit_id for e in sim.tick_events
                     if isinstance(e, ShotFiredEvent)}
    assert octo.id not in shooters
    assert marine.id in shooters


# ---------------------------------------------------------------------------
# CREATURE_TEAM
# ---------------------------------------------------------------------------
def test_creature_team_is_its_own_team_and_not_counted_by_the_match():
    """Property: CREATURE_TEAM is neither the zombies' team nor a marine
    team, and MatchRules.living_teams does not count it -- a match with one
    marine team and a live octopus is over. Breaks if living_teams counts
    every non-zombie team again (an octopus would keep a match open)."""
    assert CREATURE_TEAM != ZOMBIE_TEAM and CREATURE_TEAM not in range(0, 16)
    octo = Unit("O", x=5, y=5, team=CREATURE_TEAM, species_id="octopus")
    red, blue = Unit("R", x=5, y=5, team=0), Unit("B", x=5, y=5, team=1)
    sim = SimpleNamespace(units=[red, blue, octo], tick=0, _tps=TPS)
    assert MatchRules.living_teams(sim) == [0, 1]
    blue.alive = False
    assert MatchRules.living_teams(sim) == [0]
    assert MatchRules(time_limit_s=300.0).is_terminal(sim)


# ---------------------------------------------------------------------------
# creature_fixed: the chain currency and the pose convention
# ---------------------------------------------------------------------------
def test_chain_rounding_is_symmetric_about_zero():
    """Property: chain_from_m rounds half AWAY from zero, so it is odd:
    q(-x) == -q(x) for every x, half-counts included. Breaks if a floor (or
    a bare int()) sneaks in -- the bias design 6.3 forbids (an animal must not
    crawl on rounding)."""
    one = 1.0 / cf.CHAIN_ONE
    for counts in (0.5, 1.5, 2.5, 0.49, 0.51, 1234.5, 7.0):
        x = counts * one
        assert cf.chain_from_m(-x) == -cf.chain_from_m(x)
    assert cf.chain_from_m(0.5 * one) == 1 and cf.chain_from_m(-0.5 * one) == -1
    assert cf.chain_from_m(1.5 * one) == 2 and cf.chain_from_m(-2.5 * one) == -3
    xs = [-3.25, -0.5 * one, 0.0, 1e-9, 2.52]
    arr = cf.chain_from_m_array(xs)
    assert arr.dtype.name == "int32"
    assert list(arr) == [cf.chain_from_m(x) for x in xs]
    assert cf.chain_to_f32(arr)[-1] == pytest.approx(2.52, abs=1e-7)
    assert cf.CHAIN_SHIFT == 16 + cf.CHAIN_FINE_BITS
    with pytest.raises(ValueError, match="chain range"):
        cf.chain_from_m(cf.CHAIN_RANGE_M)


@pytest.mark.parametrize("footprint", [1, 3, 4])
def test_pose_and_unit_anchor_round_trip_with_pinned_handedness(footprint):
    """Property: pose (crown centre, metres, grid axes) <-> Unit anchor
    (top-left, float tiles) round-trip exactly; +x metres is +x tiles (East),
    +y metres is +y tiles (SOUTH, the grid is y-down); the crown centre lies
    on the footprint's centre; heading pi/2 is the Unit's North. Breaks if
    the anchor/centre offset, an axis sign, or the heading convention
    changes on one side only."""
    tile = 0.333
    for x_m, y_m in ((4.0, 7.5), (0.1665, 0.1665), (12.3456, 0.0)):
        ux, uy = cf.pose_to_unit_anchor(x_m, y_m, footprint, tile)
        bx, by = cf.unit_anchor_to_pose(ux, uy, footprint, tile)
        assert (bx, by) == pytest.approx((x_m, y_m), abs=1e-12)
    ux0, uy0 = cf.pose_to_unit_anchor(4.0, 7.5, footprint, tile)
    ux1, uy1 = cf.pose_to_unit_anchor(5.0, 7.5, footprint, tile)
    ux2, uy2 = cf.pose_to_unit_anchor(4.0, 8.5, footprint, tile)
    assert (ux1 - ux0, uy1 - uy0) == pytest.approx((1.0 / tile, 0.0))
    assert (ux2 - ux0, uy2 - uy0) == pytest.approx((0.0, 1.0 / tile))
    # a footprint anchored at (10, 20): its centre is (10 + F/2, 20 + F/2)
    x_m, y_m = cf.unit_anchor_to_pose(10.0, 20.0, footprint, tile)
    assert (x_m / tile, y_m / tile) == pytest.approx(
        (10.0 + footprint / 2, 20.0 + footprint / 2))
    u = Unit("O", x=10, y=20, team=CREATURE_TEAM, footprint=footprint)
    centre = u.tile_x + footprint // 2
    assert int(x_m / tile + 1e-9) == centre == u.center_tile_x()
    u.facing = cf.heading_rad_to_facing(math.pi / 2)
    assert u.facing_compass() == "N"
    assert cf.facing_to_heading_rad(u.facing) == math.pi / 2


def test_the_ingress_lint_scans_the_creature_modules_clean():
    """Property: creature_fixed.py and creatures.py sit under src/simulation/
    (the ingress lint's scan set) and have no banned number ingress. Breaks if
    either moves out of the scanned tree or grows a libm transcendental."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_ingress_lint", ROOT / "tests" / "test_ingress_lint.py")
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)
    for name in ("creature_fixed.py", "creatures.py"):
        path = lint.SIM_DIR / name
        assert path in set(lint.SIM_DIR.rglob("*.py"))
        assert lint._scan_file(path) == []
