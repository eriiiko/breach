"""Gameplay pack G3p -- grenade kinds, the flashbang's blinding, the throw
wind-up (#183).

``[loadouts.*] grenades`` (a table by hand-grenade ``[ammo.*]`` row, or a bare
int = frags), the unit's count block, ``Simulation.throw_grenade_intent``'s
``kind`` and integer ``fuse_ticks``, the wind-up (``[weapons.hand_grenade]
throw_windup_seconds``, ``Simulation._start_throw`` / ``_advance_throw``), the
render-only ``ThrowEvent``, and ``simulation/flash_blind.py`` over
``[payloads.flashbang] blind_*``. Design:
``docs/gameplay_pack_v1_design_2026-11-10.md`` §3 G3p.

Properties (each test's docstring names the change that must break it): a unit
behind a wall is never blinded; the blinding is monotone non-increasing in
distance and in the angle away from the flash, is ``blind_seconds`` facing the
flash and ``blind_away_frac`` times that facing away; nobody beyond the radius
is blinded and the thrower's own team is blinded like any other; a payload row
without the blind columns blinds nobody; each kind's count falls only by a
throw of that kind; the grenade leaves exactly the wind-up after the verb,
along the facing it has then; a unit that dies or is stunned in the wind-up
throws nothing and keeps its count; a second throw intent in a wind-up is
ignored; with no wind-up the throw is the pre-G3p same-tick throw (the pinned
match, tick for tick); the loadout door refuses by name; ``ThrowEvent`` moves
no digest; the direct control's selection and the HUD line.
"""
from __future__ import annotations

import sys
from pathlib import Path
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
from simulation import Simulation  # noqa: E402
from simulation import flash_blind as fb  # noqa: E402
from simulation import simulation as sim_mod  # noqa: E402
from simulation import unit_fixed  # noqa: E402
from simulation.events import ThrowEvent  # noqa: E402
from simulation.intents import FP_ONE  # noqa: E402
from simulation.loadouts import LoadoutTable  # noqa: E402
from simulation.payloads import execute_payload  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.status import (BLINDED, STUNNED, apply_status,  # noqa: E402
                               composed_flags)
from simulation.turning import PI_Q16  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import FRAG_ROW, PayloadTable, get_tables  # noqa: E402
from simulation.turning import place as _place_facing  # noqa: E402
from simulation.unit_fixed import quantize_scalar as _q_rad  # noqa: E402

SEED = 7
TABLES = get_tables()
ROWS = TABLES.hand_grenade_rows
FRAG, SMOKE, FLASH = (ROWS.index(r) for r in
                      ("grenade_frag", "grenade_smoke", "grenade_flash"))
HAND = TABLES.weapons.by_name["hand_grenade"]
WINDUP = int(HAND.throw_windup_ticks)
FLASHBANG = TABLES.payloads.by_name["flashbang"]


# ---------------------------------------------------------------------------
# An arena, optionally with a full-height wall across it
# ---------------------------------------------------------------------------
def _arena(h=24, w=48, wall_x=None) -> LevelData:
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    if wall_x is not None:
        tm[1:h - 1, wall_x] = 1
    return LevelData(name="g3p_arena", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim(wall_x=None) -> Simulation:
    sim = Simulation(_arena(wall_x=wall_x), seed=SEED, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _add(sim, name, x, y, team=0, facing_q=0):
    uid = sim.add_unit(Unit(name, x=x, y=y, team=team))
    u = sim.get_unit(uid)
    _place_facing(u, facing_q)
    return u


def _flash_at(sim, fy, fx, owner=-1):
    execute_payload(sim.gmap, sim.edit_queue, sim.units, fy, fx, FLASHBANG,
                    sim.rng, events=sim.tick_events, kind="grenade",
                    owner_id=owner)


def _blind_ticks(u) -> int:
    return max((s.remaining_ticks for s in u.statuses if s.kind == BLINDED),
               default=0)


def _unit_at(sim, cx, cy, **kw):
    """A marine whose CENTRE tile is (cx, cy) (a footprint-3 block)."""
    return _add(sim, kw.pop("name", "V"), cx - 1, cy - 1, **kw)


def _facing_away_q(dx, dy) -> int:
    """The facing (Q16 rad) that looks straight AWAY from a flash at the
    offset (dx, dy) from the unit."""
    to = unit_fixed.atan2_q16(-dy << 16, dx << 16)
    return to - PI_Q16 if to > 0 else to + PI_Q16


# ---------------------------------------------------------------------------
# A. the rows, the loadout door, the unit's block
# ---------------------------------------------------------------------------
def test_a_bare_int_grenades_means_that_many_frags():
    """Property: a loadout row's bare-int `grenades` arms exactly that many
    FRAGS and nothing of any other kind (the pre-G3p form stays legal).
    Breaks if the bare int is read as a total spread over kinds, as another
    kind, or refused."""
    t = LoadoutTable({"t": dict(weapon="k5_carbine", grenades=3,
                                explosives=0)}, ["k5_carbine"], ROWS)
    row = t.by_name["t"]
    assert row.grenades == {FRAG_ROW: 3} and row.frags == 3


def test_the_loadout_door_refuses_by_name():
    """Property: the `grenades` table is refused, naming the loadout and the
    key, for an unknown row, a row that is not of the hand-grenade family, a
    negative count and a non-integer one (a bool included). Breaks if a
    typo'd or mistyped grenade count is silently accepted or defaulted."""
    def build(grenades):
        return LoadoutTable({"t": dict(weapon="k5_carbine",
                                       grenades=grenades, explosives=0)},
                            ["k5_carbine"], ROWS)

    with pytest.raises(ValueError, match=r"loadouts\.t\.grenades\.no_such"):
        build({"no_such": 1})
    forty = next(n for n, a in TABLES.ammo.by_name.items()
                 if a.family != "hand_grenade")
    with pytest.raises(ValueError, match=rf"grenades\.{forty}"):
        build({forty: 1})
    with pytest.raises(ValueError, match="non-negative integer"):
        build({FRAG_ROW: -1})
    with pytest.raises(ValueError, match="non-negative integer"):
        build({FRAG_ROW: 1.5})
    with pytest.raises(ValueError, match="non-negative integer"):
        build({FRAG_ROW: True})
    with pytest.raises(ValueError, match="non-negative integer"):
        build(-2)


def test_a_unit_carries_one_count_per_hand_grenade_row():
    """Property: every unit has exactly one count per hand-grenade row of the
    table (the rows are the capacity), each the loadout's count for that row
    or 0, and `has_grenade` -- the legacy single count every old reader uses
    -- IS the frag slot, read and written. Breaks if a kind's count is
    dropped or shifted, the block's width is not the family's, or the legacy
    count drifts from the frag slot."""
    from simulation.loadouts import get_tables as loadout_tables
    for name, row in loadout_tables().loadouts.by_name.items():
        u = Unit("M", x=5, y=5, loadout_name=name)
        assert len(u.grenade_counts) == len(ROWS)
        assert u.grenade_counts == [int(row.grenades.get(r, 0)) for r in ROWS]
        assert u.has_grenade == u.grenade_counts[FRAG]
        u.has_grenade = 7
        assert u.grenade_counts[FRAG] == 7
        assert sum(u.grenade_counts) - 7 == sum(row.grenades.values()) \
            - row.grenades.get(FRAG_ROW, 0)
    z = Unit("Z", x=5, y=5, team=255)
    assert z.grenade_counts == [0] * len(ROWS) and z.has_grenade == 0


def test_the_hand_grenade_kinds_include_frag_first_and_the_flashbang():
    """Property: kind 0 is the frag (a zero action row and every legacy
    caller throw the frag), and the flashbang is a hand-grenade row whose
    payload is the one that blinds. Breaks if the frag is not the default
    kind, or the flash row is missing or carries another payload."""
    assert ROWS[0] == FRAG_ROW
    assert TABLES.ammo.by_name["grenade_flash"].family == "hand_grenade"
    assert TABLES.payload_for_ammo("grenade_flash") is FLASHBANG
    assert FLASHBANG.blind_radius_q > 0


def test_the_throw_verb_refuses_an_unknown_kind_and_a_float_fuse_by_name():
    """Property: `throw_grenade_intent` refuses a kind that names no
    hand-grenade row (naming the kinds) and a fuse that is not an integer
    number of ticks, before anything is recorded or latched. Breaks if a bad
    kind is clamped or latched, or a float fuse is truncated silently."""
    sim = _sim()
    u = _unit_at(sim, 10, 10)
    for bad in (len(ROWS), -1, 1.0, True):
        with pytest.raises(ValueError, match="grenade kind"):
            sim.throw_grenade_intent(u.id, FP_ONE, 0, 24, kind=bad)
    with pytest.raises(TypeError, match="fuse_ticks"):
        sim.throw_grenade_intent(u.id, FP_ONE, 0, 1.0, kind=FRAG)
    assert getattr(u, "pending_throw", None) is None


def test_the_payload_door_refuses_blind_columns_by_value_and_name():
    """Property: the blind columns pass the payload door like the flash
    columns: an unknown `blind_*` key, a seconds / fraction without a radius,
    a radius without both, a non-positive duration and a fraction outside
    [0, 1] are each refused by name; a row without them blinds nobody.
    Breaks if a misspelt or half-set column is a silent default."""
    def table(**cols):
        return PayloadTable({"x": dict(flash=1.0, **cols)}, None, None, 24)

    ok = table(blind_radius=5.0, blind_seconds=2.0, blind_away_frac=0.5)
    row = ok.by_name["x"]
    assert row.blind_radius_q > 0 and row.blind_ticks == 48
    assert table().by_name["x"].blind_radius_q == 0
    for cols, needle in (
            (dict(blind_radius=5.0, blind_secs=2.0), "blind_secs"),
            (dict(blind_seconds=2.0), "blind_radius"),
            (dict(blind_radius=5.0, blind_seconds=2.0), "blind_away_frac"),
            (dict(blind_radius=5.0, blind_seconds=0.0,
                  blind_away_frac=0.5), "blind_seconds"),
            (dict(blind_radius=5.0, blind_seconds=2.0,
                  blind_away_frac=1.5), "blind_away_frac"),
            (dict(blind_radius=-1.0), "blind_radius")):
        with pytest.raises(ValueError, match=needle):
            table(**cols)


def test_the_wind_up_is_derived_to_ticks_once_and_refused_when_invalid():
    """Property: `throw_windup_seconds` becomes integer ticks at the weapon
    table (0 stays 0, a positive time at least 1) and a negative or
    non-number value is refused by name. Breaks if the wind-up is read as
    float seconds at the sim or a typo'd value is accepted."""
    from simulation.weapons import WeaponTable
    def build(v):
        return WeaponTable({"g": dict(archetype="lobbed",
                                      ammo_family="hand_grenade",
                                      throw_windup_seconds=v)}, 24)
    assert build(0.0).by_name["g"].throw_windup_ticks == 0
    assert build(0.3).by_name["g"].throw_windup_ticks == 7
    assert build(0.01).by_name["g"].throw_windup_ticks == 1
    for bad in (-0.1, "0.3", True, float("nan")):
        with pytest.raises(ValueError, match="throw_windup_seconds"):
            build(bad)


# ---------------------------------------------------------------------------
# B. the blinding law
# ---------------------------------------------------------------------------
def test_a_unit_behind_a_wall_is_never_blinded():
    """Property: with a wall between the unit and the flash it gets no
    BLINDED status at any distance inside the radius and at any facing,
    including straight at the flash. Breaks if the blinding ignores line of
    sight (a distance-only or angle-only rule)."""
    radius = FLASHBANG.blind_radius_q >> 16
    for dist in range(2, radius):
        for k in range(8):
            facing = (k * PI_Q16) // 4 - PI_Q16 // 1
            sim = _sim(wall_x=15)
            _unit_at(sim, 12, 12, name="V", facing_q=facing)   # not used
            sim2 = _sim(wall_x=15)
            flash_x = 13
            ux = flash_x + dist
            if ux <= 15 or ux >= 46:
                continue
            u = _unit_at(sim2, ux, 12, facing_q=facing)
            _flash_at(sim2, 12, flash_x)
            assert _blind_ticks(u) == 0, (dist, facing)
    # the control: the same unit with the wall gone IS blinded at 3 tiles
    sim3 = _sim()
    u3 = _unit_at(sim3, 16, 12, facing_q=PI_Q16)             # faces west
    _flash_at(sim3, 12, 13)
    assert _blind_ticks(u3) > 0


def test_blinding_is_monotone_in_distance_and_in_the_angle_away():
    """Property: the duration never increases with distance from the flash
    (facing it) and never increases with the angle between the unit's facing
    and the direction to the flash (at a fixed distance). Breaks if either
    falloff is not monotone (a wrapped angle, a sign error, a rounding that
    steps back up)."""
    sim = _sim()
    row = FLASHBANG
    prev = None
    for d in range(0, 14):                     # tiles, facing the flash
        facing = PI_Q16                        # unit east of the flash, west-facing
        t = fb.blind_ticks_for(row, d * d, -d, 0, facing)
        if prev is not None:
            assert t <= prev, d
        prev = t
    prev = None
    for k in range(0, 129):                    # 0 .. half a turn
        ang = (k * PI_Q16) // 128
        facing = PI_Q16 + ang                  # rotated away from the flash
        t = fb.blind_ticks_for(row, 16, -4, 0, facing)
        if prev is not None:
            assert t <= prev, k
        prev = t
    assert sim is not None


def test_the_blinding_is_blind_seconds_facing_and_away_frac_of_it_away():
    """Property: at the flash's tile and facing it a unit is blinded exactly
    `blind_seconds` (in ticks); with its back to it (same distance) it is
    blinded `blind_away_frac` of that (within one tick of rounding); the
    cone of +-60 degrees gets the whole of it. Breaks if the facing factor's
    ends are not 1 and `blind_away_frac`, or the front cone is narrower."""
    full = FLASHBANG.blind_ticks
    sim = _sim()
    u = _unit_at(sim, 20, 12, facing_q=PI_Q16)               # faces the flash
    _flash_at(sim, 12, 20)                                    # on its own tile
    assert _blind_ticks(u) == full
    # facing the flash from 2 tiles: scaled by g only; away: by g * away
    g = fb.distance_factor_q16(4, FLASHBANG.blind_radius_q)
    facing = _facing_away_q(-2, 0)
    t_front = fb.blind_ticks_for(FLASHBANG, 4, -2, 0, PI_Q16)
    t_back = fb.blind_ticks_for(FLASHBANG, 4, -2, 0, facing)
    assert t_front == (full * g) >> 16
    want = full * g / 65536.0 * FLASHBANG.blind_away_frac
    assert abs(t_back - want) <= 1.0
    # the front cone: 59 degrees off still gets all of it
    off = (59 * PI_Q16) // 180
    assert fb.blind_ticks_for(FLASHBANG, 4, -2, 0, PI_Q16 + off) == t_front
    assert fb.blind_ticks_for(FLASHBANG, 4, -2, 0, PI_Q16 - off) == t_front
    assert t_back < t_front


def test_nobody_beyond_the_radius_is_blinded_and_the_thrower_team_is():
    """Property: a unit at or beyond `blind_radius` is not blinded at any
    facing; units of the thrower's own team and of the other team inside it
    are blinded alike (any team). Breaks if the radius is not a hard edge, or
    the rule skips the thrower's team or the thrower."""
    radius = FLASHBANG.blind_radius_q >> 16
    sim = _sim()
    far = _unit_at(sim, 14 + radius + 1, 12, team=1, facing_q=PI_Q16)
    edge = _unit_at(sim, 14 + radius, 12, team=1, facing_q=PI_Q16)
    mate = _unit_at(sim, 18, 6, team=0, facing_q=PI_Q16, name="mate")
    foe = _unit_at(sim, 18, 18, team=1, facing_q=PI_Q16, name="foe")
    thrower = _unit_at(sim, 16, 12, team=0, facing_q=PI_Q16, name="thrower")
    _flash_at(sim, 12, 14, owner=thrower.id)
    assert _blind_ticks(far) == 0 and _blind_ticks(edge) == 0
    assert _blind_ticks(mate) > 0 and _blind_ticks(foe) > 0
    assert _blind_ticks(thrower) > 0
    # the status says who cast it
    st = next(s for s in thrower.statuses if s.kind == BLINDED)
    assert st.source_id == thrower.id


def test_a_payload_without_blind_columns_blinds_nobody():
    """Property: executing a frag or a smoke payload next to a unit applies
    no BLINDED status (no row but the flashbang blinds; the teargas coupling
    is the gas's). Breaks if blinding is keyed on anything but the row's own
    blind columns (a per-grenade check, a flash-only test)."""
    for name in ("frag_standard", "smoke_screen"):
        sim = _sim()
        u = _unit_at(sim, 16, 12, team=1, facing_q=PI_Q16)
        execute_payload(sim.gmap, sim.edit_queue, sim.units, 12, 14,
                        TABLES.payloads.by_name[name], sim.rng,
                        events=sim.tick_events, kind="grenade", owner_id=-1)
        assert all(s.kind != BLINDED for s in u.statuses), name
        assert TABLES.payloads.by_name[name].blind_radius_q == 0


def test_a_second_weaker_flash_does_not_shorten_a_longer_blinding():
    """Property: a unit already blinded for longer keeps it when a weaker
    flash lands on it (BLINDED is a refresh status; the flash raises, never
    lowers); a stronger one does raise it. Breaks if the second flash
    overwrites with a shorter timer."""
    sim = _sim()
    u = _unit_at(sim, 20, 12, facing_q=PI_Q16)
    _flash_at(sim, 12, 20)                       # the whole of it
    full = _blind_ticks(u)
    v = _unit_at(sim, 24, 12, facing_q=PI_Q16, name="v")
    _flash_at(sim, 12, 20)
    assert _blind_ticks(u) == full
    # a weak one (far, facing the flash) cannot lower u's remaining time
    u2 = _unit_at(sim, 20, 18, facing_q=PI_Q16, name="u2")
    _flash_at(sim, 18, 20)                       # strong for u2
    strong = _blind_ticks(u2)
    _flash_at(sim, 18, 28)                       # 8 tiles off: weaker
    assert _blind_ticks(u2) == strong
    assert v is not None


def test_a_blinded_unit_cannot_aim():
    """Property: the flashbang applies the SAME BLINDED status the teargas
    does -- the composed flags lose `can_aim` and nothing else -- so a
    blinded unit's fire falls to the snap cone. Breaks if the blinding is a
    new status kind or removes more than the aim."""
    sim = _sim()
    u = _unit_at(sim, 20, 12, facing_q=PI_Q16)
    assert composed_flags(u).can_aim
    _flash_at(sim, 12, 20)
    f = composed_flags(u)
    assert not f.can_aim and f.can_move and f.can_act


# ---------------------------------------------------------------------------
# C. the throw: kinds, counts, the wind-up
# ---------------------------------------------------------------------------
def _thrower(sim, **kw):
    u = _unit_at(sim, 10, 12, **kw)
    u.grenade_counts[:] = [0] * len(ROWS)
    return u


def _run(sim, n):
    events = []
    for _ in range(n):
        t = sim.tick
        sim.step()
        events += [(t, e) for e in sim.tick_events
                   if isinstance(e, ThrowEvent)]
    return events


def test_each_kinds_count_falls_only_by_a_throw_of_that_kind():
    """Property: throwing kind k spends one of k and leaves every other
    count alone; the counts never go below 0; a throw of a kind the unit
    does not carry spends nothing, starts nothing and emits no ThrowEvent.
    Breaks if the count pool is shared, a kind is spent by another's throw,
    or a throw with none left still lobs or emits."""
    sim = _sim()
    u = _thrower(sim)
    u.grenade_counts[FRAG], u.grenade_counts[SMOKE] = 2, 1
    before = list(u.grenade_counts)
    sim.throw_grenade_intent(u.id, FP_ONE, 0, 24, kind=SMOKE)
    ev = _run(sim, WINDUP + 2)
    assert len(ev) == 1 and ev[0][1].kind == SMOKE
    want = list(before)
    want[SMOKE] -= 1
    assert u.grenade_counts == want
    assert len(sim.projectiles) == 1
    assert sim.projectiles[0].ammo_name == ROWS[SMOKE]
    # a kind it lacks: nothing, no event
    n_proj = len(sim.projectiles)
    for kind in (SMOKE, FLASH):                  # smoke now 0; flash never had
        sim.throw_grenade_intent(u.id, FP_ONE, 0, 24, kind=kind)
        assert _run(sim, WINDUP + 2) == []
        assert u.grenade_counts == want
        assert len(sim.projectiles) == n_proj
    # the frags go one at a time to zero and no further
    for _ in range(4):
        sim.throw_grenade_intent(u.id, FP_ONE, 0, 24, kind=FRAG)
        _run(sim, WINDUP + 2)
        assert min(u.grenade_counts) >= 0
    assert u.grenade_counts[FRAG] == 0
    assert u.grenade_counts[SMOKE] == 0


def test_the_grenade_leaves_exactly_the_wind_up_after_the_verb_along_the_facing():
    """Property: the ThrowEvent is emitted on the tick the verb is consumed
    with `ticks` = the wind-up; the grenade leaves the hand exactly `ticks`
    ticks later (thrown_tick, and the count falls then, not before) along
    the facing the unit has THEN -- turned meanwhile at the turn cap, not the
    direction the verb named; the fuse counts from the release. Breaks if the
    grenade leaves early or late, is aimed with the verb's vector or the
    facing at the verb, or the fuse counts from the verb."""
    assert WINDUP >= 2
    sim = _sim()
    u = _thrower(sim, facing_q=0)                    # east
    u.grenade_counts[FRAG] = 1
    sim.set_aim(u.id, FP_ONE, 0)
    t0 = sim.tick
    sim.throw_grenade_intent(u.id, FP_ONE, 0, 30, reach_q=8 * FP_ONE,
                             kind=FRAG)
    sim.set_aim(u.id, 0, -FP_ONE)                    # turn to north (90 deg)
    ev = []
    for _ in range(WINDUP + 1):
        before = u.grenade_counts[FRAG]
        t = sim.tick
        sim.set_aim(u.id, 0, -FP_ONE)
        sim.step()
        ev += [(t, e) for e in sim.tick_events if isinstance(e, ThrowEvent)]
        if sim.tick - 1 < t0 + WINDUP:
            assert u.grenade_counts[FRAG] == before == 1
            assert not sim.projectiles
    assert [(t, e.ticks, e.kind, e.unit_id) for t, e in ev] == \
        [(t0, WINDUP, FRAG, u.id)]
    assert u.grenade_counts[FRAG] == 0
    (proj,) = sim.projectiles
    assert proj.thrown_tick == t0 + WINDUP
    assert proj.fuse_ticks == 30
    assert proj.get_detonate_tick() == t0 + WINDUP + 30
    # north: the target is straight above the thrower's centre, 8 tiles
    cx, cy = u.center_tile_x(), u.center_tile_y()
    assert abs(proj.target_fx - 0.5 - cx) < 0.1
    assert abs(proj.target_fy - 0.5 - (cy - 8)) < 0.1


def test_a_second_throw_intent_during_a_wind_up_is_ignored():
    """Property: a throw intent consumed while the unit is winding up does
    nothing -- no second event, no second grenade, no second count spent, and
    the first throw's kind and reach stand. Breaks if the second verb
    restarts the wind-up, queues, or spends a count."""
    sim = _sim()
    u = _thrower(sim)
    u.grenade_counts[FRAG], u.grenade_counts[SMOKE] = 2, 2
    sim.throw_grenade_intent(u.id, FP_ONE, 0, 24, kind=FRAG)
    ev = []
    for i in range(WINDUP + 2):
        if i in (1, 3):
            sim.throw_grenade_intent(u.id, FP_ONE, 0, 24, kind=SMOKE)
        t = sim.tick
        sim.step()
        ev += [e for e in sim.tick_events if isinstance(e, ThrowEvent)]
    assert len(ev) == 1 and ev[0].kind == FRAG
    assert u.grenade_counts[FRAG] == 1 and u.grenade_counts[SMOKE] == 2
    assert [p.ammo_name for p in sim.projectiles] == [ROWS[FRAG]]


def test_a_unit_stunned_or_killed_in_the_wind_up_throws_nothing_and_keeps_its_count():
    """Property: a unit that is stunned (cannot act) or dies between the verb
    and the release throws no grenade and keeps its count; the wind-up is
    gone, so after the stun it can throw again. Breaks if the grenade leaves
    anyway, the count is spent at the verb, or the dropped wind-up blocks
    every later throw."""
    for how in ("stun", "kill"):
        sim = _sim()
        u = _thrower(sim)
        u.grenade_counts[FRAG] = 2
        sim.throw_grenade_intent(u.id, FP_ONE, 0, 24, kind=FRAG)
        sim.step()                                   # consumed: winding up
        assert u.throw_release_tick >= 0
        if how == "stun":
            apply_status(u, STUNNED, magnitude=0, duration_ticks=3 * WINDUP)
        else:
            u.alive = False
        _run(sim, WINDUP + 3)
        assert u.grenade_counts[FRAG] == 2, how
        assert not sim.projectiles, how
        assert u.throw_release_tick == -1, how
    # a stunned unit's fresh verb is ignored too (no event), and after the
    # stun it throws again
    sim = _sim()
    u = _thrower(sim)
    u.grenade_counts[FRAG] = 1
    apply_status(u, STUNNED, magnitude=0, duration_ticks=4)
    sim.throw_grenade_intent(u.id, FP_ONE, 0, 24, kind=FRAG)
    assert _run(sim, 2) == []
    _run(sim, 6)
    sim.throw_grenade_intent(u.id, FP_ONE, 0, 24, kind=FRAG)
    assert len(_run(sim, WINDUP + 2)) == 1
    assert u.grenade_counts[FRAG] == 0


def test_with_no_wind_up_the_throw_is_the_pre_g3p_same_tick_throw(monkeypatch):
    """Property: with `throw_windup_seconds = 0` the grenade leaves on the
    tick the verb is consumed, along the verb's own vector at the verb's
    reach (the pre-G3p throw), the ThrowEvent says 0 ticks, and the fuse
    counts from that tick. Breaks if the zero wind-up still defers the
    release, or uses the facing instead of the verb's vector."""
    monkeypatch.setattr(CFG.weapons.hand_grenade, "throw_windup_seconds", 0.0)
    sim = _sim()
    assert sim.weapons_tables.weapons.by_name["hand_grenade"] \
        .throw_windup_ticks == 0
    u = _thrower(sim, facing_q=PI_Q16)               # faces WEST ...
    u.grenade_counts[FRAG] = 1
    t0 = sim.tick
    sim.throw_grenade_intent(u.id, 0, FP_ONE, 24, reach_q=5 * FP_ONE,
                             kind=FRAG)              # ... verb says SOUTH
    ev = _run(sim, 1)
    assert [(t, e.ticks) for t, e in ev] == [(t0, 0)]
    (proj,) = sim.projectiles
    assert proj.thrown_tick == t0 and proj.get_detonate_tick() == t0 + 24
    cx, cy = u.center_tile_x(), u.center_tile_y()
    assert abs(proj.target_fx - 0.5 - cx) < 1e-9
    assert abs(proj.target_fy - 0.5 - (cy + 5)) < 1e-9
    assert u.grenade_counts[FRAG] == 0


#: ship_demo seed 1 as it played before G3p's wind-up (the pin of G1p-c / G2p
#: at main 0f0225bf: win 1 at 240). With the wind-up at 0 it must be again.
#: RE-MEASURED at #183 G4p (2026-11-10, the focus-fire law, another cause --
#: the test_turn_rate precedent): the same wind-up-0 run with exposure live
#: (the shipped [damage.exposure] dials). SOLE CAUSE, verified on the same
#: tree: with `k_expose = 0` the wind-up-0 run gives exactly the values this
#: gate held at G3p (win 1 at 240, field 4a6ae8a34011fc3cc832c20326df631cd17
#: 4b27bf64fc359d6249717ae6c9d89, tick 6eb3105b24e4debf47cdcb6366caea9df2d871
#: 7bbc2ca6387aba968a3002939e, units 816192b5309273533dff345e4300080b980e5323
#: c43fb4c46241c92af38ae4ad).
#: RE-MEASURED at #183 G5p-a1 commit 1 (2026-11-10, the turning law's
#: representation and routing at unlimited acceleration): the same run on
#: the new tree. Outcome, ticks and field_digest UNCHANGED; tick_digest and
#: the unit stream move. SOLE CAUSES, verified on the same tree: with every
#: non-aim facing writer placing the facing at once again and the old float
#: representation hashed, the run gives exactly the previous values; with
#: the representation alone undone only the unit stream moves (the zombie
#: walk's facing lands a tick later, through slot 1a).
#: (was tick 0fe41371d6d3c881305628ad25114c8c631b69075483cb58f1175bfefba157f3,
#: units cdbc400cffeb98d9e0514afe4310d64685621f19fb1f2205b8fc66398c9de81b)
_PRE_G3P_SHIP_DEMO = dict(
    outcome="win", team=1, ticks=240,
    field_digest="3086f0bfa2047def383cbe22d2e0d964"
                 "f5e681d584a75b8063ce35dbe879ef44",
    tick_digest="3c33cb848ddb0c0319fb40d214e21ff0"
                "7c046d9a31ae945290cc806ae9076271",
    unit_stream="0cec524a21865e061416a1e1708eb00f"
                "e0b2218a1c1f8214af3a4e9aa243b617")


def test_with_no_wind_up_the_pinned_match_is_the_pre_g3p_match(monkeypatch):
    """Property (the cause of the G3p re-pin): with the wind-up at 0, ship_demo
    seed 1 plays exactly the match it played before G3p, tick for tick --
    the new loadout table (smoke and a flashbang the bots never throw), the
    integer fuse, the kind column, the format bump and the ThrowEvent change
    nothing in it, so the WIND-UP is the only cause of the pin's move. Breaks
    if any of those is not neutral at wind-up 0 (the frag's kind, its fuse
    tick, its reach and direction). The next DELIBERATE re-pin of this match
    by another cause retires this gate's values with it."""
    import level_loader
    from run_match import run_match
    from test_zombie_control import _UnitStream
    monkeypatch.setattr(CFG.weapons.hand_grenade, "throw_windup_seconds", 0.0)
    level = level_loader.load("ship_demo", str(ROOT / "levels"))
    stream = _UnitStream()
    o = run_match(level, on_tick=stream)
    got = dict(outcome=o.result.outcome, team=o.result.team,
               ticks=o.result.ticks, field_digest=o.field_digest,
               tick_digest=o.tick_digest, unit_stream=stream.h.hexdigest())
    assert got == _PRE_G3P_SHIP_DEMO


# ---------------------------------------------------------------------------
# D. ThrowEvent is render-only
# ---------------------------------------------------------------------------
def test_the_throw_event_is_once_per_started_throw_and_moves_no_digest(
        monkeypatch):
    """Property: one ThrowEvent per STARTED throw (a refused verb emits
    none), outside the synced event set, and a match's digests are identical
    with the event emitted and with it removed from every tick's events.
    Breaks if the event joins the digest, is emitted per tick of the wind-up
    or for an ignored verb."""
    import field_ab_harness as H
    assert "ThrowEvent" not in H._SYNCED_EVENT_TYPES
    from run_match import run_match
    from test_match_replay import (BOX_SPEC, _ThrowerControl, _box)
    import control_source
    from simulation.match import MatchTeam
    monkeypatch.setitem(control_source._MATCH_CONTROLLERS, "test_thrower",
                        _ThrowerControl)
    import dataclasses
    t = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                  controller="test_thrower")
    spec = dataclasses.replace(BOX_SPEC, teams=(t, t))
    import tempfile
    lvl = _box(Path(tempfile.mkdtemp()), "g3pbox", spec)

    seen = []
    orig_start = sim_mod.Simulation._start_throw

    def counting(self, u, throw):
        n0 = sum(isinstance(e, ThrowEvent) for e in self.tick_events)
        orig_start(self, u, throw)
        seen.append(sum(isinstance(e, ThrowEvent) for e in self.tick_events)
                    - n0)

    monkeypatch.setattr(sim_mod.Simulation, "_start_throw", counting)
    a = run_match(lvl)
    assert seen and set(seen) <= {0, 1} and 1 in seen

    def silent(self, u, throw):
        orig_start(self, u, throw)
        self.tick_events[:] = [e for e in self.tick_events
                               if not isinstance(e, ThrowEvent)]

    monkeypatch.setattr(sim_mod.Simulation, "_start_throw", silent)
    b = run_match(lvl)
    assert (a.result, a.field_digest, a.tick_digest) == \
        (b.result, b.field_digest, b.tick_digest)


def test_a_recorded_match_with_two_kinds_of_throw_replays_every_checkpoint(
        monkeypatch, tmp_path):
    """Property: a recorded match whose units throw two different kinds
    carries both kind indices in the recording and replays through every
    checkpoint to the original digests. Breaks if the kind column is not
    recorded or fed back, or the wind-up makes a replay diverge."""
    import dataclasses
    import control_source
    import match_replay as mr
    from run_match import run_match
    from simulation.match import MatchTeam
    from test_match_replay import BOX_SPEC, _ThrowerControl, _box, _copy
    monkeypatch.setitem(control_source._MATCH_CONTROLLERS, "test_thrower",
                        _ThrowerControl)
    t = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                  controller="test_thrower")
    lvl = _box(tmp_path, "g3pkinds",
               dataclasses.replace(BOX_SPEC, teams=(t, t)))
    a = run_match(lvl, record=True, checkpoint_every_ticks=5)
    kinds = {args[5] for v in a.recording.intents.values()
             for k, args in v if k == "throw_grenade_intent"}
    assert {FRAG, SMOKE} <= kinds
    b = run_match(lvl, replay=_copy(a.recording))
    assert b.checkpoints_verified == len(a.recording.checkpoints)
    assert (b.result, b.field_digest, b.tick_digest) == \
        (a.result, a.field_digest, a.tick_digest)
    assert mr.INTENT_KINDS["throw_grenade_intent"][3] == "int"


# ---------------------------------------------------------------------------
# E. direct control: the selection, the HUD line
# ---------------------------------------------------------------------------
def test_next_carried_kind_only_ever_names_a_carried_kind_and_wraps():
    """Property: the selection steps to the next kind (wrapping) of which a
    count is left, never to an empty one, stays put when it is the only kind
    carried and is None when none is. Breaks if the cycle can land on a kind
    with 0, fails to wrap, or loses the selection when nothing else is
    carried."""
    from control_direct import next_carried_kind as nxt
    rng = np.random.default_rng(5)
    for _ in range(200):
        counts = [int(c) for c in rng.integers(0, 3, size=6)]
        cur = int(rng.integers(0, 6)) if rng.random() < 0.8 else None
        k = nxt(counts, cur)
        if not any(counts):
            assert k is None
        else:
            assert counts[k] > 0
            if cur is not None and sum(c > 0 for c in counts) == 1 \
                    and counts[cur] > 0:
                assert k == cur
    assert nxt([1, 0, 0], 0) == 0 and nxt([1, 1, 0], 1) == 0
    assert nxt([0, 2, 0], None) == 1 and nxt([0, 0, 0], 2) is None


def test_the_direct_control_cycles_throws_the_selected_kind_and_moves_on():
    """Property: the cycle key steps the selected kind through the kinds the
    unit carries, a throw goes out with the SELECTED kind, and when that kind
    is spent the selection moves to the next carried one by itself (a throw
    is never issued with a kind the unit lacks, and with none carried none is
    issued). Breaks if the throw ignores the selection, the cycle lands on an
    empty kind, or the selection sticks to a spent kind."""
    from config import CFG as cfg
    from control_direct import DeviceFrame, DirectControl, key_table
    dc = DirectControl(keys=key_table(cfg))

    class _Sim:
        def __init__(self, u):
            self.u, self.calls = u, []

        def get_unit(self, uid):
            return self.u

        def __getattr__(self, name):
            def call(*a, **k):
                self.calls.append((name, a, k))
            return call

    u = SimpleNamespace(id=1, grenade_counts=[2, 1, 0, 0, 1, 0])
    sim = _Sim(u)

    def frame(**kw):
        d = dict(move=None, aim=None, trigger=False, throw=None, use=False)
        d.update(kw)
        return DeviceFrame(**d)

    dc._sync_grenade_kind(u)
    assert dc.grenade_kind == 0                    # frag first
    dc._emit(sim, 1, frame(cycle_grenade=True))
    assert dc.grenade_kind == 1
    dc._emit(sim, 1, frame(cycle_grenade=True))
    assert dc.grenade_kind == 4                    # skips the empty kinds
    dc._emit(sim, 1, frame(cycle_grenade=True))
    assert dc.grenade_kind == 0                    # wraps
    sim.calls.clear()
    dc.grenade_kind = 4
    dc._emit(sim, 1, frame(throw=(FP_ONE, 0, None)))
    (call,) = [c for c in sim.calls if c[0] == "throw_grenade_intent"]
    assert call[2]["kind"] == 4
    u.grenade_counts[4] = 0                        # the flash is spent
    dc._sync_grenade_kind(u)
    assert u.grenade_counts[dc.grenade_kind] > 0
    sim.calls.clear()
    u.grenade_counts[:] = [0] * 6
    dc._sync_grenade_kind(u)
    dc._emit(sim, 1, frame(cycle_grenade=True))
    assert not [c for c in sim.calls if c[0] == "throw_grenade_intent"]


def test_the_cycle_key_is_bound_and_no_longer_inert():
    """Property: `cycle_grenade` is bound in the key table and is not in the
    inert list. Breaks if the key is left inert after the verb exists."""
    import control_direct as cd
    assert "cycle_grenade" in cd.KEY_ACTIONS
    assert "cycle_grenade" not in cd.INERT_ACTIONS


def test_the_hud_line_shows_the_selected_kind_and_each_count():
    """Property: the grenade line names every kind the unit carries with its
    count and marks the selected kind (even at 0), says `none` for an empty
    belt, and the continuous panel carries it. Breaks if a count is missing,
    the selection is not marked, or the panel drops the line."""
    from ui.model import grenade_text, panel_state_lines
    rows = ["grenade_frag", "grenade_smoke", "grenade_flash"]
    text = grenade_text(rows, [2, 1, 1], 0)
    assert text == "[frag 2]  smoke 1  flash 1"
    assert grenade_text(rows, [2, 0, 1], 1) == "frag 2  [smoke 0]  flash 1"
    assert grenade_text(rows, [0, 0, 0], None) == "none"
    sim = _sim()
    u = _unit_at(sim, 10, 12)
    lines = [ln.text for ln in panel_state_lines(sim, u, grenade_kind=FLASH)]
    joined = "\n".join(lines)
    assert "Grenades (4): " in joined and "Charges: " in joined
    carried = {r: n for r, n in zip(ROWS, u.grenade_counts)}
    assert f"[flash {carried['grenade_flash']}]" in joined
    assert f"frag {carried['grenade_frag']}" in joined


def test_the_policy_row_carries_the_kind_and_an_integer_fuse():
    """Property: the generated action row has `kind` and integer
    `fuse_ticks` columns for the throw (and no float fuse), and the random
    rows draw the kind inside the hand-grenade rows. Breaks if a column is
    missing, the kind domain exceeds the family, or the fuse is a float."""
    import control_policy as cp
    schema = cp.action_schema()
    names = schema.names
    assert "throw_grenade_intent.kind" in names
    assert "throw_grenade_intent.fuse_ticks" in names
    assert not any("fuse_seconds" in n for n in names)
    rows = cp.random_action_rows(np.random.default_rng(11), 600)
    k = rows[:, schema.index("throw_grenade_intent.kind")]
    f = rows[:, schema.index("throw_grenade_intent.fuse_ticks")]
    assert k.min() >= 0 and k.max() < len(ROWS)
    assert len(set(k.tolist())) > 1
    assert f.min() >= 0


# ---------------------------------------------------------------------------
# F. End to end: the real main.py in a hidden window, direct control
# ---------------------------------------------------------------------------
def test_e2e_four_cycles_w_throws_each_kind_after_the_wind_up():
    """Property (the real window, direct control on door_test): W throws the
    selected kind, FOUR steps the selection to the next carried kind, a spent
    kind hands the selection to the next carried one by itself -- so frag,
    FOUR, smoke, W (the flashbang is selected now), flashbang, W (the frag
    left) throw four grenades of the kinds frag, smoke, flash, frag; each
    ThrowEvent names the wind-up, each grenade leaves exactly that many ticks
    after it and is of that kind's round, each count falls by one when its
    grenade leaves, and the flashbang blinds. Breaks if a key is not wired to
    the verb, the selection is not the throw's kind, a grenade leaves at the
    verb's tick, or the flash does not blind."""
    import json
    import os
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [sys.executable, "tests/_e2e_throw_probe.py", "--level", "door_test",
         "--control", "direct", "--mouse", "900,508@1", "--frames", "480",
         "--press", "W@20", "--press", "FOUR@200", "--press", "W@260",
         "--press", "W@330", "--press", "W@420"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    line = next(ln for ln in r.stdout.splitlines()
                if ln.startswith("THROWPROBE "))
    log = json.loads(line[len("THROWPROBE "):])
    events = [(t, tuple(ev)) for t, counts, evs, *_ in log for ev in evs]
    assert [k for _, (k, _n) in events] == [FRAG, SMOKE, FLASH, FRAG]
    assert all(n == WINDUP for _, (_k, n) in events)
    # each grenade is in flight WINDUP ticks after its event, of its row
    flights = {}
    for t, counts, evs, flying, blind, *_ in log:
        for thrown, ammo in flying:
            flights[thrown] = ammo
    for t, (k, _n) in events:
        assert flights[t + WINDUP] == ROWS[k]
    # the counts: one less of that kind when it leaves, not at the verb
    by_tick = {t: counts for t, counts, *_ in log}
    start = log[0][1]
    for i, (t, (k, _n)) in enumerate(events):
        spent = sum(1 for _, (k2, _n2) in events[:i] if k2 == k)
        assert by_tick[t][k] == start[k] - spent            # still in hand
        assert by_tick[t + WINDUP][k] == start[k] - spent - 1
    assert min(min(c) for _, c, *_ in log) >= 0
    # the flashbang blinds somebody (the units of door_test stand in the room)
    flash_release = next(t for t, (k, _n) in events if k == FLASH) + WINDUP
    after = [b for t, _c, _e, _f, b, *_ in log if t > flash_release]
    assert any(after), "the flashbang blinded nobody"
