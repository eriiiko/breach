"""#29 U1 -- the unit's HOOKS at neutral (``simulation.unit_traits``;
``docs/unit_design_2026-11-14.md`` sections 2, 3, 8).

A hook is one unit's Q16 factor on a class value a table holds; every hooked
value has ONE accessor, and every reader -- the law, the bots' predictors,
the observation, the panel -- calls it.

PROPERTIES PROTECTED (each test names its own and the change that breaks it)
  1. Every hook obeys the table's rule: a column of the block, neutral by
     default and then absent from the digest, inside the door, with its
     accessor named (iterated over ``HOOKS``, never a pinned list).
  2. The door refuses out-of-range, non-integer and unknown hooks by name,
     and a TURN hook exactly when the turning law would need more trials
     than its device loop holds.
  3. Each hook is LIVE and EXACT at every site of its row: at 2 ONE and
     ONE/2 the value changes by the hook's ratio (one integer rounding of
     the class value) -- HP at the spawn, the conversion, the panel and the
     fall-back; WALK / SPRINT on the marine walk, the zombie walk, the WEGO
     path and the onephase cadence; TURN in the law and its predictor; HOLD
     in the knockdown; STRIKE in the blow and the bite; PUSH in the blow and
     the shove; SPREAD in the aimed, snap and beam cones; HANDLING in the
     reload, the wind-up and the plant -- and the bots' predictors and the
     observation read the same value as the law.
  4. A zombie keeps its hooks: a converted unit holds the same block and its
     values are the zombie class's times them.
  5. The scan: no reader of a hooked class value in ``src/`` outside its
     accessor (a reader is a line naming the class value; an exception says
     ``traits-class:`` with its reason).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation, combat, unit_impulse, unit_schema  # noqa: E402
from simulation import ai_zombie, unit_traits as ut  # noqa: E402
from simulation import obs_spec_v1 as O  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.movement import (MOBILITY_ONE, FootprintSamples,  # noqa: E402
                                 default_speed, half_up)
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.status import BLINDED, KNOCKED_DOWN, apply_status  # noqa: E402
from simulation.turning import PI_Q16, place  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.unit_fixed import mul_q16_round, quantize_scalar  # noqa: E402

ONE = ut.ONE
TILE = 1.0 / 3.0
EAST = 0
#: The two non-neutral values every live test runs at.
FACTORS = (2 * ONE, ONE // 2)


def _arena(h=64, w=64):
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    return LevelData(name="traits", version="1", path=Path("."), tilemap=tm,
                     tile_size_m=TILE, diffuse_path=Path("."))


def _sim(ruleset=None):
    sim = Simulation(_arena(), seed=1, breach_physics=bp,
                     enable_recorder=False,
                     ruleset=ruleset or ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _add(sim, name, x, y, team=0, facing=EAST, hooks=None):
    u = sim.get_unit(sim.add_unit(Unit(name, x=x, y=y, team=team),
                                  hooks=hooks))
    place(u, facing)
    return u


def _expect(class_value, h):
    """The rule every accessor follows: one integer rounding of the class
    value, the identity at ONE."""
    return class_value if h == ONE else mul_q16_round(class_value, h)


# ---------------------------------------------------------------------------
# 1, 2: the table's rule and the door
# ---------------------------------------------------------------------------
def test_every_hook_obeys_the_tables_rule():
    """Property 1: every row of ``HOOKS`` is a slot of the unit's block
    (the schema column's width is the table's length), neutral on a fresh
    unit, absent from the digest record while neutral and present once
    not, inside the door at its bounds, and names an accessor. Iterates the
    table, so a new hook is held to it with no edit here. Breaks if the
    block's width leaves the table, a neutral block reaches the digest, or
    a hook is added without an accessor."""
    col = unit_schema.COLUMN_BY_NAME["traits"]
    assert unit_schema.column_width(col) == ut.N_HOOKS == len(ut.HOOKS)
    assert col.digest == unit_schema.DIGEST_NONDEFAULT
    sim = _sim()
    u = _add(sim, "u", 20, 20)
    assert list(u.traits) == [ONE] * ut.N_HOOKS
    assert "traits" not in unit_schema.digest_record(u)
    for i, h in enumerate(ut.HOOKS):
        assert ut.HOOK_INDEX[h.name] == i and h.accessor and h.scales
        for v in (ut.HOOK_MIN, ut.HOOK_MAX):
            w = _add(sim, f"{h.name}{v}", 30, 30, hooks={h.name: v})
            assert w.traits[i] == v
            assert unit_schema.digest_record(w)["traits"][i] == v


def test_the_door_refuses_by_name():
    """Property 2: a hook outside ``[ONE/8, 8 ONE]``, a non-integer (a float,
    a bool) and an unknown hook name are refused at the compile step, the
    message naming the hook. Breaks if the door lets one through or stops
    naming it."""
    sim = _sim()
    for h in ut.HOOKS:
        for bad in (ut.HOOK_MIN - 1, ut.HOOK_MAX + 1, float(ONE), True):
            with pytest.raises(ValueError, match=h.name):
                sim.add_unit(Unit("x", x=20, y=20), hooks={h.name: bad})
    with pytest.raises(ValueError, match="SPEEDY"):
        sim.add_unit(Unit("x", x=20, y=20), hooks={"SPEEDY": ONE})


def test_the_turn_door_is_the_laws_trial_bound():
    """Property 2: with the class dials set so the law sits near its trial
    bound, the door refuses a TURN hook EXACTLY when ``ceil(min(w', pi) /
    a')`` (the hooked dials) exceeds ``TURN_TRIALS_MAX`` or a dial rounds to
    0, and accepts every other. Breaks if the door checks the class dials
    only, or not at all."""
    from simulation import turning
    sim = _sim()
    w = sim._marine_turn_rate_q16
    sim._marine_turn_accel_q16 = -(-w // turning.TURN_TRIALS_MAX)
    refused = accepted = 0
    for h in range(ut.HOOK_MIN, ut.HOOK_MAX + 1, ONE // 64):
        wq = _expect(w, h)
        aq = _expect(sim._marine_turn_accel_q16, h)
        bad = (wq < 1 or aq < 1
               or -(-min(wq, PI_Q16) // aq) > turning.TURN_TRIALS_MAX)
        try:
            ut.check_traits(sim, [h if i == ut.TURN else ONE
                                  for i in range(ut.N_HOOKS)])
            ok = True
        except ValueError as e:
            assert "TURN" in str(e)
            ok = False
        assert ok == (not bad), (h, wq, aq)
        refused += not ok
        accepted += ok
    assert refused and accepted


# ---------------------------------------------------------------------------
# 3: each hook live and exact at every site of its row
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h", FACTORS)
def test_hp_at_every_site(h):
    """Property 3, HP: the spawn (a marine and a zombie), the conversion,
    the panel and ``bots_v1``'s fall-back threshold all read the class hp
    times the hook. Breaks if one site reads the class row (or the species'
    vitality) instead of ``max_hp_q``."""
    import control_bots_v1
    from ui.model import panel_state_lines
    sim = _sim()
    m = _add(sim, "m", 20, 20, hooks={"HP": h})
    z = _add(sim, "z", 40, 40, team=ZOMBIE_TEAM, hooks={"HP": h})
    marine_q = ut.class_hp_q(False)
    zombie_q = ut.class_hp_q(True)
    assert m.hp_q == ut.max_hp_q(m) == _expect(marine_q, h)
    assert z.hp_q == ut.max_hp_q(z) == _expect(zombie_q, h)
    # the panel's "HP: now/full"
    txt = " ".join(l.text for l in panel_state_lines(sim, m))
    assert f"/{_expect(marine_q, h) // ONE}" in txt
    # bots_v1's fall-back: wounded below frac x the hooked full health
    bot = control_bots_v1.BotTeamV1(0)
    frac = bot.settings.fallback_hp_frac
    full = _expect(marine_q, h)
    m.hp_q = int(full * frac) - ONE
    assert bot._falling_back(m, [m])
    m.hp_q = int(full * frac) + ONE
    assert not bot._falling_back(m, [m])
    # the conversion: the zombie class times the KEPT hook
    m.alive = False
    m.killed_by_zombie = True
    ai_zombie.convert_marines_to_zombies(sim.units)
    assert m.is_zombie and m.hp_q == _expect(zombie_q, h)


def _walk_step(sim, u, throttle, ticks=1):
    """``u``'s displacement (Q16 tiles, along +x) over ``ticks`` ticks of a
    forward walk east at ``throttle``."""
    x0 = u.x_q
    for _ in range(ticks):
        sim.set_move_dir(u.id, ONE, 0, throttle)
        sim.step()
    return u.x_q - x0


@pytest.mark.parametrize("h", FACTORS)
def test_walk_and_sprint_on_the_marine_walk(h):
    """Property 3, WALK / SPRINT on the continuous walk: a unit walking
    forward at throttle 0 steps ``v = ONE / cost'`` with ``cost'`` the walk
    cost reduced with its WALK hook inside ``default_speed``'s one rounding;
    at full throttle the full-speed cost with its SPRINT hook; each hook
    leaves the other gait alone. Breaks if the walk site drops a hook,
    swaps them, or applies one to both costs."""
    from simulation.movement import move_speed_q16
    sim = _sim()
    a = _add(sim, "a", 10, 10, hooks={"WALK": h})
    b = _add(sim, "b", 10, 30, hooks={"SPRINT": h})
    c = _add(sim, "c", 10, 50)
    t = sim.speed_table
    mob = [MOBILITY_ONE] * 9

    def v(walk_h, sprint_h, throttle):
        cw = default_speed(FootprintSamples(mob, speed_hook_q16=walk_h),
                           t.walk_cost_q)
        cf = default_speed(FootprintSamples(mob, speed_hook_q16=sprint_h),
                           t.full_cost_q)
        return move_speed_q16(cw, cf, t.side_q, t.back_q, throttle, ONE)

    for u in (a, b, c):
        sim.set_move_dir(u.id, ONE, 0, 0)
    sim.step()                     # one tick to settle the facing / effort
    da, db, dc = (_walk_step(sim, u, 0) for u in (a, b, c))
    assert dc == v(ONE, ONE, 0) and da == v(h, ONE, 0) and db == dc
    assert abs(da * ONE - dc * h) <= h      # the ratio, to one count
    fa, fb, fc = (_walk_step(sim, u, ONE) for u in (a, b, c))
    assert fc == v(ONE, ONE, ONE) and fb == v(ONE, h, ONE) and fa == fc
    assert abs(fb * ONE - fc * h) <= h


def test_the_speed_hook_composes_inside_default_speeds_rounding():
    """Property 3, WALK / SPRINT in the law: ``default_speed`` with a hook
    is ``half_up(class * n * 1000 * ONE, total * hook)`` and exactly the
    old cost at ``ONE`` for every mobility mix. Breaks if the hook is
    applied as a second rounding or at the wrong side of the division."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        mob = [int(m) for m in rng.integers(300, 1001, size=9)]
        c = int(rng.integers(1, 30 * ONE))
        h = int(rng.integers(ut.HOOK_MIN, ut.HOOK_MAX + 1))
        base = half_up(c * 9 * MOBILITY_ONE, sum(mob))
        assert default_speed(FootprintSamples(mob), c) == max(1, base)
        assert default_speed(FootprintSamples(mob, speed_hook_q16=h), c) == \
            max(1, half_up(c * 9 * MOBILITY_ONE * ONE, sum(mob) * h))


def _zombie_scene(gait_hook):
    sim = _sim()
    _add(sim, "prey", 50, 20)
    zs = []
    for k, h in enumerate((ONE, gait_hook)):
        z = _add(sim, f"z{k}", 8, 12 + 16 * k, team=ZOMBIE_TEAM,
                 hooks={"WALK": h, "SPRINT": h})
        z.zombie_activated = True
        zs.append(z)
    return sim, zs


@pytest.mark.parametrize("h", FACTORS)
def test_walk_and_sprint_on_the_zombie_walk(h):
    """Property 3, WALK / SPRINT on the zombie walk: an activated zombie's
    stride (ticks per step, read from its own steps) is the shamble cost
    reduced with its WALK hook -- and, sprinting at prey in sight, the
    sprint cost with its SPRINT hook. Breaks if ``_walk`` drops the hook or
    the sprint takes the walk's."""
    from simulation.horde import horde_settings
    for sprint in (False, True):
        settings = horde_settings(CFG)
        sim, (z0, z1) = _zombie_scene(h)
        sim.horde_settings = replace(settings, sprint=sprint,
                                     sprint_r2=10 ** 6 if sprint else 0)
        t = sim.speed_table
        base = t.zombie_sprint_ticks if sprint else t.zombie_shamble_ticks
        mob = [MOBILITY_ONE] * 9
        want = [default_speed(FootprintSamples(mob, speed_hook_q16=x), base)
                for x in (ONE, h)]
        steps = {z0.id: [], z1.id: []}
        last = {z.id: (z.x_q, z.y_q) for z in (z0, z1)}
        for _ in range(80):
            sim.step()
            for z in (z0, z1):
                if (z.x_q, z.y_q) != last[z.id]:
                    steps[z.id].append(sim.tick)
                    last[z.id] = (z.x_q, z.y_q)
        for z, w in zip((z0, z1), want):
            gaps = set(np.diff(steps[z.id]).tolist())
            assert gaps == {w}, (sprint, z.name, gaps, w)


@pytest.mark.parametrize("h", FACTORS)
def test_walk_and_sprint_on_the_legacy_cadences(h):
    """Property 3, WALK / SPRINT where WEGO and onephase share the law:
    onephase's tile cadence (its full speed) takes the SPRINT hook, and a
    TwoPhaseWEGO move order walks the precomputed path at the WALK-hooked
    cadence. Breaks if a legacy site builds its samples without the hook."""
    from simulation import orders as Ord
    from simulation.ruleset import TwoPhaseWEGO
    from simulation.timeline import base_ticks_per_tile, tile_cadence
    sim = _sim()
    a = _add(sim, "a", 10, 10, hooks={"SPRINT": h})
    b = _add(sim, "b", 10, 30)
    base = base_ticks_per_tile(sim.gmap)
    mob = [MOBILITY_ONE] * 9
    assert tile_cadence(sim.gmap, b, 10, 12, Ord.ORDER_MOVE, False) == base
    assert tile_cadence(sim.gmap, a, 10, 12, Ord.ORDER_MOVE, False) == \
        default_speed(FootprintSamples(mob, speed_hook_q16=h), base)
    # TwoPhaseWEGO: a move-attack order's path, per tile
    wsim = _sim(TwoPhaseWEGO())
    w0 = _add(wsim, "w0", 4, 4)
    w1 = _add(wsim, "w1", 4, 30, hooks={"WALK": h})
    for u, ty in ((w0, 4), (w1, 30)):
        wsim.apply_action(u.id, Ord.Order(Ord.ORDER_MOVE_ATTACK, 40, ty,
                                          phase=0))
    wsim.set_paused(False)
    wsim.step()
    # the ticks of the first tile step: its last interpolated position is
    # the next tile exactly (index step_ticks - 1)
    per_tile = [next(i for i, (x, _) in enumerate(u.move_path) if x == 5.0)
                + 1 for u in (w0, w1)]
    speed = int(CFG.movement.marine_attack_ticks_per_tile)
    assert per_tile[0] == default_speed(FootprintSamples(mob), speed)
    assert per_tile[1] == default_speed(
        FootprintSamples(mob, speed_hook_q16=h), speed)


@pytest.mark.parametrize("h", FACTORS)
def test_turn_in_the_law_its_predictor_and_the_observation(h):
    """Property 3, TURN: both dials are the class value times the hook;
    a unit turning from rest takes the hooked acceleration on its first
    tick; ``facing_after_turn_q16`` (the bots' predictor) equals the facing
    the step then gives; and the observation's aimer reads the same dials.
    Breaks if the law, the predictor or the observation reads the cached
    class dials."""
    sim = _sim()
    u = _add(sim, "u", 20, 20, hooks={"TURN": h})
    n = _add(sim, "n", 40, 40)
    w, a = sim._marine_turn_rate_q16, sim._marine_turn_accel_q16
    assert sim.unit_turn_rate_q16(u) == _expect(w, h)
    assert sim.unit_turn_accel_q16(u) == _expect(a, h)
    assert sim.unit_turn_rate_q16(n) == w
    for unit in (u, n):
        sim.set_aim(unit.id, -ONE, 0)                 # a half turn west
    want = sim.facing_after_turn_q16(u, -ONE, 0)
    sim.step()
    assert u.facing_q == want
    assert u.omega_q == _expect(a, h) and n.omega_q == a
    aimer = O._Aimer.build(SimpleNamespace(
        weapons_tables=sim.weapons_tables, impulse_table=sim.impulse_table,
        unit_turn_accel_q16=sim.unit_turn_accel_q16,
        unit_turn_rate_q16=sim.unit_turn_rate_q16),
        u, SimpleNamespace(max_throw_range=1, frag_radius=1,
                           half_turn_ticks=17))
    assert (aimer.a, aimer.w) == (_expect(a, h), _expect(w, h))


def _knocked(h, j_factor):
    sim = _sim()
    u = _add(sim, "u", 20, 20, hooks={"HOLD": h})
    table = sim.impulse_table
    jx = int(j_factor * table.v_fall_q[0]) * u.mass_g
    unit_impulse.add(table, u, jx, 0)
    for _ in range(3):
        sim.step()
    return any(st.kind == KNOCKED_DOWN for st in u.statuses)


@pytest.mark.parametrize("h", FACTORS)
def test_hold_scales_the_knockdown(h):
    """Property 3, HOLD: ``stability_q16`` is the class stability times the
    hook (its one reader is the law), and the law shows it: some push along
    the facing knocks one of a neutral and a hooked unit down and not the
    other -- the stronger one standing. Breaks if the accessor or the law
    ignores the hook (then every push treats both alike)."""
    sim = _sim()
    u = _add(sim, "u", 20, 20, hooks={"HOLD": h})
    assert unit_impulse.stability_q16(u) == _expect(ONE, h)
    split = [f / 10 for f in range(5, 40)
             if _knocked(ONE, f / 10) != _knocked(h, f / 10)]
    assert split, "no push tells the hooked unit from the neutral one"
    f = split[0]
    assert _knocked(min(ONE, h), f) and not _knocked(max(ONE, h), f)


class _Record:
    """A stand-in attack slot that records what phase A would apply."""

    def __init__(self):
        self.hits, self.pushes = [], []

    def hit(self, target, packet, events=None, source=None, **kw):
        self.hits.append(packet)

    def push(self, table, target, jx, jy, *a, **kw):
        self.pushes.append((jx, jy))

    def status(self, *a, **kw):
        pass

    def face(self, *a, **kw):
        pass


def _melee_pair(h_strike=ONE, h_push=ONE):
    sim = _sim()
    s = _add(sim, "s", 20, 20, hooks={"STRIKE": h_strike, "PUSH": h_push})
    e = _add(sim, "e", 23, 20, team=1, facing=PI_Q16)
    return sim, s, e


@pytest.mark.parametrize("h", FACTORS)
def test_strike_in_the_blow_and_the_bite(h):
    """Property 3, STRIKE: the melee verb's blow carries the row's
    ``melee_damage`` times the striker's hook (Q16), and a zombie's bite
    ``[zombie] melee_damage`` times the biter's. Breaks if either site
    builds its packet from the class value."""
    sim, s, e = _melee_pair(h_strike=h)
    w = sim.weapons_tables.weapons.by_name[s.melee_id]
    rec = _Record()
    s.pending_melee = True
    combat.process_melee_intents(sim.units, sim.tick, sim.weapons_tables,
                                 sim.impulse_table, effects=rec)
    assert [p.q() for p in rec.hits] == [_expect(int(w.melee_damage) << 16, h)]
    # the bite
    bsim = _sim()
    prey = _add(bsim, "prey", 20, 20)
    z = _add(bsim, "z", 23, 20, team=ZOMBIE_TEAM, facing=PI_Q16,
             hooks={"STRIKE": h})
    z.zombie_activated = True
    got = []
    real = ai_zombie.apply_packet

    def spy(target, packet, **kw):
        got.append(packet.q())
        return real(target, packet, **kw)
    ai_zombie.apply_packet = spy
    try:
        bsim.step()
    finally:
        ai_zombie.apply_packet = real
    assert got == [_expect(quantize_scalar(float(CFG.zombie.melee_damage)),
                           h)], prey.hp_q


@pytest.mark.parametrize("h", FACTORS)
def test_push_in_the_blow_and_the_shove(h):
    """Property 3, PUSH: the melee blow's recorded impulse and the shove's
    impulse into its target's accumulator are each the class impulse
    (``melee_push_q`` of the row, ``shove_q``) times the pusher's hook.
    Breaks if either reads the impulse table directly."""
    sim, s, e = _melee_pair(h_push=h)
    table = sim.impulse_table
    w = sim.weapons_tables.weapons.by_name[s.melee_id]
    rec = _Record()
    s.pending_melee = True
    combat.process_melee_intents(sim.units, sim.tick, sim.weapons_tables,
                                 table, effects=rec)
    (jx, jy), = rec.pushes
    assert (jx, jy) == unit_impulse.along_q(
        _expect(table.melee_push_q[w.name], h), ONE, 0)
    x0 = e.imp_x
    assert unit_impulse.shove(table, sim.units, s, sim.tick)
    assert e.imp_x - x0 == unit_impulse.along_q(
        _expect(table.shove_q, h), ONE, 0)[0]


def _cones(sim, shooter, weapon_id, blinded=False, ticks=1):
    """The cone half-angles (Q16) the draws are handed for ``shooter``'s
    held trigger over ``ticks`` ticks."""
    from simulation import draws
    shooter.weapon_id = weapon_id
    if blinded:
        apply_status(shooter, BLINDED, 0, 50)
    seen = []
    real = draws.cone_offset_q16

    def spy(d, cone_q):
        seen.append(int(cone_q))
        return real(d, cone_q)
    combat._draws.cone_offset_q16 = spy
    try:
        for _ in range(ticks):
            sim.set_aim(shooter.id, ONE, 0)
            sim.set_trigger(shooter.id, True)
            sim.step()
    finally:
        combat._draws.cone_offset_q16 = real
    return set(seen)


@pytest.mark.parametrize("h", FACTORS)
def test_spread_in_every_cone_and_its_predictors(h):
    """Property 3, SPREAD: the aimed cone, the snap cone (a blinded shooter)
    and a beam's cone are each the class cone through its door times the
    shooter's hook, and the bots' trigger discipline (both bot layers) and
    the observation's aim judgements read the aimed one. Breaks if a draw
    site, the selection or a predictor reads the row's ``spread_deg``."""
    import control_bots
    import control_bots_v1
    tables = combat.weapon_tables()
    for wid, blinded in (("k5_carbine", False), ("k5_carbine", True),
                         ("lance_3", False)):
        sim = _sim()
        s = _add(sim, "s", 20, 20, hooks={"SPREAD": h})
        w = tables.weapons.by_name[wid]
        deg = w.spread_snap_deg if blinded else w.spread_deg
        assert _cones(sim, s, wid, blinded) == {
            _expect(combat.cone_q16_of_deg(deg), h)}, (wid, blinded)
    sim = _sim()
    s = _add(sim, "s", 20, 20, hooks={"SPREAD": h})
    w = sim.weapons_tables.weapons.by_name[s.weapon_id]
    aimed = _expect(combat.cone_q16_of_deg(w.spread_deg), h)
    assert ut.shot_cone_q16(s, w) == aimed
    assert control_bots.BotTeam(0)._weapon_spread_q(s) == aimed
    assert control_bots_v1.BotTeamV1(0)._weapon_spread_q(s) == aimed
    assert O._cone_of(sim.weapons_tables.weapons.by_name, None, s) == aimed


@pytest.mark.parametrize("h", FACTORS)
def test_handling_in_the_reload_the_windup_and_the_plant(h):
    """Property 3, HANDLING: a reload's stall (and its ``ReloadEvent``), a
    throw's wind-up (and its ``ThrowEvent``) and a plant (and its
    ``PlantEvent``) each run the class ticks times the unit's hook, and the
    observation's time-left fractions are over the unit's OWN duration (so
    they start at (d - 1) / d whatever the hook). Breaks if a law or the
    observation reads the row's ticks."""
    from simulation.events import PlantEvent, ReloadEvent, ThrowEvent
    sim = _sim()
    u = _add(sim, "u", 20, 20, hooks={"HANDLING": h})
    rows = sim.weapons_tables.weapons.by_name
    w = rows[u.weapon_id]
    hand, c4 = rows["hand_grenade"], rows["c4_satchel"]
    ev = []
    combat.mag_bind(u, w)
    assert combat.reload_begin(u, w, sim.tick, events=ev)
    reload_t = _expect(int(w.reload_ticks), h)
    assert u.reload_done_tick - sim.tick == reload_t
    assert [e.ticks for e in ev if isinstance(e, ReloadEvent)] == [reload_t]
    t0 = sim.tick
    sim.throw_grenade_intent(u.id, 0, ONE, 48, 5 * ONE, 0)
    sim.step()
    wind = max(1, _expect(int(hand.throw_windup_ticks), h))
    assert u.throw_release_tick - t0 == wind
    assert [e.ticks for e in sim.tick_events if isinstance(e, ThrowEvent)] \
        == [wind]
    p = _add(sim, "p", 40, 12, hooks={"HANDLING": h})
    t1 = sim.tick
    sim.plant_charge_intent(p.id, 0)
    sim.step()
    plant = max(1, _expect(int(c4.plant_ticks), h))
    assert p.plant_done_tick - t1 == plant
    assert [e.ticks for e in sim.tick_events if isinstance(e, PlantEvent)] \
        == [plant]
    # the observation's denominators: the unit's own durations
    ctx = SimpleNamespace(hand_rows=tuple(sim.weapons_tables.hand_grenade_rows),
                          shove_cooldown_ticks=1, x_max_q=ONE,
                          c4_fuse_max_ticks=1)
    tick = sim.tick
    kit = O._kit_values(u, w, None, None, tick, ctx)
    assert kit[3] == 1.0 and kit[4] == pytest.approx(
        (u.reload_done_tick - tick) / reload_t)
    for unit in (u, p):
        e = np.zeros(O.F, np.float32)
        O._unit_columns(e, unit, ctx, tick, rows)
        if unit is u:
            assert e[16] == pytest.approx((u.throw_release_tick - tick) / wind)
        else:
            assert e[19] == pytest.approx((p.plant_done_tick - tick) / plant)


def test_handling_never_rounds_a_duration_to_zero():
    """Property 3, HANDLING's floor: a class duration of one tick or more
    stays at least one tick at the smallest hook (a 0-tick wind-up or plant
    is another code path), and a 0-tick class duration stays 0. Breaks if
    the floor is dropped or applied to a 0-tick row."""
    u = SimpleNamespace(traits=[ut.HOOK_MIN] * ut.N_HOOKS)
    for base in range(0, 10):
        got = ut.reload_ticks(u, SimpleNamespace(reload_ticks=base))
        assert got == (0 if base == 0 else max(1, _expect(base, ut.HOOK_MIN)))


# ---------------------------------------------------------------------------
# 4: a zombie keeps its hooks
# ---------------------------------------------------------------------------
def test_a_converted_unit_keeps_its_block_and_takes_the_zombie_class():
    """Property 4: a marine with every hook non-neutral, killed by a zombie
    and converted, holds the SAME block (never re-compiled), and its hooked
    values are the zombie CLASS's times those hooks (the full health; the
    stability, whose zombie overlay is a class value). Breaks if the
    conversion re-runs the compile step or resets the block."""
    from simulation.exchange import _stability_for
    sim = _sim()
    hooks = {h.name: ONE + (k + 1) * (ONE // 16)
             for k, h in enumerate(ut.HOOKS)}
    m = _add(sim, "m", 20, 20, hooks=hooks)
    before = list(m.traits)
    m.alive = False
    m.killed_by_zombie = True
    ai_zombie.convert_marines_to_zombies(sim.units)
    assert m.is_zombie and list(m.traits) == before
    assert m.hp_q == _expect(ut.class_hp_q(True), before[ut.HP])
    assert unit_impulse.stability_q16(m) == _expect(
        quantize_scalar(_stability_for(m)), before[ut.HOLD])
    assert sim.unit_turn_rate_q16(m) == _expect(
        sim._marine_turn_rate_q16, before[ut.TURN])


# ---------------------------------------------------------------------------
# 5: the scan
# ---------------------------------------------------------------------------
#: Per hook, the CLASS values it scales, as the source text that reads them,
#: and the files allowed to (where the class value is defined, quantized or
#: is the accessor itself). Every other line naming one must carry
#: ``traits-class: <reason>``.
CLASS_READERS = {
    "HP": (r"\b(zombie|marine)\.hp\b|\.vitality\b", ("unit_traits.py",)),
    "WALK": (r"\.walk_cost_q\b|\.zombie_shamble_ticks\b",
             ("speeds.py",)),
    "SPRINT": (r"\.full_cost_q\b|\.zombie_sprint_ticks\b", ("speeds.py",)),
    "TURN": (r"_marine_turn_(rate|accel)_q16|\bturn_(rate|accel)_q16\(",
             ("turning.py", "unit_traits.py")),
    "HOLD": (r"\b_stability_for\(", ("exchange.py",)),
    "STRIKE": (r"\.melee_damage\b", ("unit_traits.py", "weapons.py")),
    "PUSH": (r"\.melee_push_q\b|\.shove_q\b", ("unit_traits.py",
                                               "unit_impulse.py")),
    "SPREAD": (r"\.spread_(snap_)?deg\b|\bcone_q16_of_deg\(",
               ("unit_traits.py", "weapons.py")),
    "HANDLING": (r"(?<!unit_traits)\.(reload|throw_windup|plant)_ticks\b",
                 ("unit_traits.py", "weapons.py")),
}
#: Lines of an allowed-file exception that ARE the accessor (the accessor
#: lives beside its class value): ``def`` lines and the accessor bodies.
_ACCESSORS = {
    "TURN": ("simulation.py", ("unit_turn_rate_q16", "unit_turn_accel_q16")),
    "HOLD": ("unit_impulse.py", ("stability_q16",)),
    "SPREAD": ("combat.py", ("cone_q16_of_deg",)),       # the cone's door
}


def _src_files():
    for path in sorted((ROOT / "src").rglob("*.py")):
        yield path
    for name in ("ui/model.py", "main.py"):
        p = ROOT / name
        if p.exists():
            yield p


def _function_spans(tree, names):
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            spans.append((node.lineno, node.end_lineno))
    return spans


def test_no_reader_of_a_hooked_class_value_outside_its_accessor():
    """Property 5 (the scan): every line in ``src/`` (and the panel's
    ``ui/model.py``) that names a hooked CLASS value -- the hp rows, the
    speed table's costs, the turning dials, the stability read, a row's
    melee damage / push / spread / durations -- is in the file that defines
    it, inside the hook's accessor, or marked ``traits-class:`` with its
    reason (a reference denominator, a pass-through to the accessor, a
    spray jet's cone). Comments and docstrings are not readers. Breaks when
    a new reader takes the class value raw, so a non-neutral hook would
    escape it."""
    bad = []
    for path in _src_files():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        doc_lines = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(
                    getattr(node, "value", None), ast.Constant) and isinstance(
                    node.value.value, str):
                doc_lines.update(range(node.lineno, node.end_lineno + 1))
        lines = text.splitlines()
        for hook, (pattern, allowed) in CLASS_READERS.items():
            if path.name in allowed:
                continue
            spans = []
            acc = _ACCESSORS.get(hook)
            if acc and path.name == acc[0]:
                spans = _function_spans(tree, set(acc[1]))
            rx = re.compile(pattern)
            for i, line in enumerate(lines, 1):
                code = line.split("#", 1)[0]
                if i in doc_lines or not rx.search(code):
                    continue
                if any(a <= i <= b for a, b in spans):
                    continue
                if "traits-class:" in line:
                    continue
                if path.name == "simulation.py" and hook == "TURN" and \
                        "= turning.turn_" in code:
                    continue          # the class dials' door, cached once
                bad.append(f"{hook}: {path.relative_to(ROOT)}:{i}: "
                           f"{line.strip()}")
    assert not bad, "\n".join(bad)


def test_every_footprint_samples_call_names_the_walkers_speed_hook():
    """Property 5, WALK / SPRINT: every ``footprint_samples`` call in
    ``src/`` passes ``speed_hook_q16`` (the walker's own hook for the cost
    it reduces). Breaks if a new walker builds its samples without it (found
    by parsing, never a pinned list)."""
    missing = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id",
                                getattr(node.func, "attr", "")) ==
                    "footprint_samples"
                    and not any(k.arg == "speed_hook_q16"
                                for k in node.keywords)):
                missing.append(f"{path.name}:{node.lineno}")
    assert not missing, missing
