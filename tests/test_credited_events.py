"""#123 -- credited events: every UnitHit / UnitKilled carries a cause from a
closed vocabulary (``events.Cause``) and the unit credited with it
(``attacker_id``, -1 = the environment).

Properties gated here (each test names the change that must break it):

(a) a scripted match on a small lane level -- a bullet, a frag grenade, a
    zombie bite and a heat death nobody caused -- ends with EVERY death
    carrying a kill event whose cause is in the vocabulary, credited to the
    shooter, the thrower, the biter and the environment respectively;
(b) a team-kill is credited to the team-mate who fired, and the bench's
    attacker table calls it ``own``;
(c) an unknown cause string is refused at the packet door (and by the
    events' own constructors) before the unit is touched;
(d) the vocabulary's values ARE the strings the producers emitted before
    #123 (consumers comparing literals, and the hashed string, unchanged);
(e) the bench's by-attacker-team table sums to its by-cause table;
(f) determinism: one seed, one per-tick digest stream with credited events
    -- and the attacker IS in the hash;
(g) indirect credit: a flame jet's own heat is credited to its sprayer this
    tick; an indirect credit never enters the idle stance's
    ``recent_attackers``; the per-tick claims are gone after the tick.

The level is synthetic (``LevelData``, the test_p3_direct_e2e precedent):
four lanes walled off from each other so the scenes cannot interact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tests", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from bench_bot_matches import DeathLog  # noqa: E402
from field_ab_harness import capture_snapshot, _capture_unit_state  # noqa: E402
from field_digest import tick_digest  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.damage import (  # noqa: E402
    CREDIT_HEAT, HEAT, KINETIC, DamagePacket, TickCredit, apply_packet,
    tick_credit)
from simulation.events import (  # noqa: E402
    CAUSE_NAMES, NO_ATTACKER, Cause, UnitHitEvent, UnitKilledEvent)
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.field_edit import EditMode, FieldEdit, Region  # noqa: E402
from simulation.intents import FP_ONE, FP_SHIFT  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.status import STATUS_REGISTRY  # noqa: E402
from simulation.unit import Unit  # noqa: E402

SEED = 20261029
HULL, AIR = 1, 4


# ---------------------------------------------------------------------------
# The lane level + the scripted match
# ---------------------------------------------------------------------------
LANE_TOPS = (1, 13, 25, 37)          # 7-row lanes, 5 hull rows between


def _lanes(h=45, w=30) -> LevelData:
    """Four 7-row air lanes between 5-row hull bands: no line of sight, no
    path and no blast reach between lanes (a frag's hot cloud still warms
    the next lane a little through the hull -- victims carry enough HP for
    that)."""
    tm = np.full((h, w), HULL, dtype=np.int32)
    for top in LANE_TOPS:
        tm[top:top + 7, 1:w - 1] = AIR
    return LevelData(name="credit_lanes", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim() -> Simulation:
    sim = Simulation(_lanes(), seed=SEED, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _add(sim, name, x, y, team, hp=None):
    u = sim.get_unit(sim.add_unit(Unit(name, x=x, y=y, team=team)))
    if hp is not None:
        u.current_hp = float(hp)
    return u


def _heat_on(sim, u, amount=4.0e6):
    """An UNOWNED heat deposit on ``u``'s footprint through the canonical
    ingress (FieldEdit into ``heat``): the environment's fire."""
    for (tx, ty) in u.occupied_tiles():
        sim.edit(FieldEdit(field="heat", region=Region.TILE, coords=(ty, tx),
                           amount=amount, mode=EditMode.ADD, source_id=990))


def _scripted_match(n_ticks=60, on_tick=None):
    """Lane A: S shoots VA. Lane B: T lobs a frag onto VB. Lane C: zombie Z
    bites VC. Lane D: VD stands in an unowned heat deposit. Victims are team
    1 marines with no orders (they neither move nor fire), each at HP its
    own killer clears in one hit (bullet 10, frag 60 at the centre, bite 60)
    and the others' stray warmth does not."""
    a, b, c, d = (t + 2 for t in LANE_TOPS)
    sim = _sim()
    who = {}
    who["S"] = _add(sim, "S", 3, a, team=0)
    who["VA"] = _add(sim, "VA", 12, a, team=1, hp=5)
    who["T"] = _add(sim, "T", 3, b, team=0)
    who["VB"] = _add(sim, "VB", 12, b, team=1, hp=30)
    # Z starts one tile out of reach, so the bite lands after tick 1 (the
    # bench's watcher first sees the match AFTER tick 1).
    who["Z"] = _add(sim, "Z", 2, c, team=ZOMBIE_TEAM)
    who["VC"] = _add(sim, "VC", 7, c, team=1, hp=30)
    who["VD"] = _add(sim, "VD", 12, d, team=1, hp=1)

    sim.set_aim(who["S"].id, FP_ONE, 0)               # east, down lane A
    sim.set_trigger(who["S"].id, True)
    # The lob: from T's centre (4, 12) nine tiles east -> VB's centre tile.
    who["T"].has_grenade = 1
    sim.throw_grenade_intent(who["T"].id, FP_ONE, 0, fuse_seconds=0.5,
                             reach_q=9 << FP_SHIFT)
    events, digests = [], []
    for _ in range(n_ticks):
        if who["VD"].alive:
            _heat_on(sim, who["VD"])
        sim.step()
        sim.set_paused(False)
        events.extend(e for e in sim.tick_events
                      if isinstance(e, (UnitHitEvent, UnitKilledEvent)))
        digests.append(tick_digest(capture_snapshot(sim)))
        if on_tick is not None:
            on_tick(sim)
    return sim, who, events, digests


@pytest.fixture(scope="module")
def match():
    log = DeathLog()
    sim, who, events, digests = _scripted_match(on_tick=log)
    return sim, who, events, digests, log


# ---------------------------------------------------------------------------
# (a) every death credited: shooter, thrower, biter, environment
# ---------------------------------------------------------------------------
def _kill_of(events, unit):
    kills = [e for e in events
             if isinstance(e, UnitKilledEvent) and e.unit_id == unit.id]
    assert len(kills) == 1, f"{unit.name}: expected one kill event, {kills}"
    return kills[0]


def test_every_death_of_the_scripted_match_is_credited(match):
    """Property: every unit that died has exactly one kill event, its cause
    in the closed vocabulary, credited to the unit that caused it (-1 for
    the environment). Breaks if a producer drops or mis-names its attacker
    (the shooter, the grenade's thrower, the biting zombie), credits the
    environment's heat to somebody, or a death goes without a kill event."""
    sim, who, events, _digests, _log = match
    # Dead, or bitten to death and turned (killed marines convert, #122).
    dead = [u for u in sim.units
            if not u.alive or (u.is_zombie and u is not who["Z"])]
    assert {u.id for u in dead} == {who[k].id for k in ("VA", "VB", "VC",
                                                         "VD")}
    assert who["VC"].is_zombie                         # the bite converted
    for u in dead:
        assert _kill_of(events, u).killed_by in CAUSE_NAMES
    for e in events:
        src = e.source if isinstance(e, UnitHitEvent) else e.killed_by
        assert src in CAUSE_NAMES

    ka = _kill_of(events, who["VA"])
    assert (ka.killed_by, ka.attacker_id) == ("bullet", who["S"].id)
    kb = _kill_of(events, who["VB"])
    assert (kb.killed_by, kb.attacker_id) == ("explosion", who["T"].id)
    kc = _kill_of(events, who["VC"])
    assert (kc.killed_by, kc.attacker_id) == ("melee", who["Z"].id)
    kd = _kill_of(events, who["VD"])
    assert (kd.killed_by, kd.attacker_id) == ("heat", NO_ATTACKER)


def test_a_hit_and_its_kill_carry_one_attacker(match):
    """Property: the killing UnitHitEvent and the UnitKilledEvent of one
    death name the same attacker and cause. Breaks if apply_packet credits
    the two events differently."""
    _sim_, who, events, _d, _l = match
    for name in ("VA", "VB", "VC", "VD"):
        u = who[name]
        k = _kill_of(events, u)
        i = events.index(k)
        hit = events[i - 1]
        assert isinstance(hit, UnitHitEvent) and hit.unit_id == u.id
        assert (hit.source, hit.attacker_id) == (k.killed_by, k.attacker_id)


# ---------------------------------------------------------------------------
# (b) a team-kill is credited to the team-mate
# ---------------------------------------------------------------------------
def test_a_team_kill_is_credited_to_the_team_mate():
    """Property: friendly fire stays ON and is VISIBLE -- a marine shot by a
    team-mate dies credited to that team-mate, and the bench files the kill
    under ``own``. Breaks if friendly fire is filtered out, or the credit or
    the bench's team split goes to the wrong side."""
    sim = _sim()
    shooter = _add(sim, "A", 3, 3, team=0)
    mate = _add(sim, "B", 12, 3, team=0, hp=1)
    log = DeathLog()
    sim.set_aim(shooter.id, FP_ONE, 0)
    sim.set_trigger(shooter.id, True)
    kills = []
    for _ in range(30):
        sim.step()
        sim.set_paused(False)
        kills += [e for e in sim.tick_events if isinstance(e, UnitKilledEvent)]
        log(sim)
    assert [(k.unit_id, k.killed_by, k.attacker_id) for k in kills] == [
        (mate.id, "bullet", shooter.id)]
    assert log.marine_by["own"] == 1 and sum(log.marine_by.values()) == 1


# ---------------------------------------------------------------------------
# (c) the packet door refuses an unknown cause
# ---------------------------------------------------------------------------
def test_an_unknown_cause_is_refused_at_the_packet_door():
    """Property: a cause outside the vocabulary raises before any HP moves
    or any event is emitted. Breaks if the door validates after applying,
    or accepts a free string again."""
    u = Unit("U", x=2, y=2, team=0)
    hp0 = u.current_hp
    events = []
    with pytest.raises(ValueError, match="unknown damage cause"):
        apply_packet(u, DamagePacket(5, KINETIC, 3), events, source="lava")
    assert u.current_hp == hp0 and events == []
    with pytest.raises(ValueError):
        UnitHitEvent(unit_id=1, damage=1, source="lava")
    with pytest.raises(ValueError):
        UnitKilledEvent(unit_id=1, killed_by="fall")   # no producer yet


# ---------------------------------------------------------------------------
# (d) the values are the old strings
# ---------------------------------------------------------------------------
# The literal ``source=`` strings every producer emitted before #123 (main
# de1f461): combat (bullet, laser, melee), ai_zombie (melee), exchange (heat,
# explosion, poison_gas), and status rows by name.
_PRE_123_SOURCES = ("bullet", "laser", "melee", "heat", "explosion",
                    "poison_gas")


def test_the_cause_values_are_the_pre_123_strings():
    """Property: every pre-#123 source string is a cause whose value is that
    string, a member compares equal to its string, and an event stores the
    PLAIN string (so a consumer's literal and the digest's repr are
    unchanged). Breaks if a value is renamed or the event stores the
    member."""
    for s in _PRE_123_SOURCES:
        assert Cause(s).value == s and Cause(s) == s
    for row in STATUS_REGISTRY:
        if row.dtype is not None:
            assert Cause(row.name).value == row.name
    ev = UnitHitEvent(unit_id=1, damage=1, source=Cause.HEAT)
    assert type(ev.source) is str and ev.source == "heat"
    assert repr(ev.source) == "'heat'"


# ---------------------------------------------------------------------------
# (e) the bench's two tables agree
# ---------------------------------------------------------------------------
def test_the_bench_attacker_table_sums_to_its_cause_table(match):
    """Property: every death the bench counts lands in one cause bucket and
    one attacker-team bucket, per victim group; on the scripted match the
    marine victims split enemy (bullet, frag, bite) / environment (heat).
    Breaks if the bench drops or double-counts a death in either table."""
    _sim_, _who, _events, _d, log = match
    assert sum(log.marine.values()) == sum(log.marine_by.values())
    assert sum(log.zombie.values()) == sum(log.zombie_by.values())
    assert log.marine_by["enemy"] >= 3 and log.marine_by["environment"] >= 1
    assert log.marine_by["uncredited"] == 0


# ---------------------------------------------------------------------------
# (f) determinism, with the attacker in the hash
# ---------------------------------------------------------------------------
def test_one_seed_one_credited_digest_stream(match):
    """Property: the scripted match replays to the same per-tick digest
    stream and the same credited event stream. Breaks on any
    nondeterminism in who is credited (e.g. a claim order that depends on
    dict or set iteration)."""
    _sim_, _who, events, digests, _log = match
    _s2, _w2, events2, digests2 = _scripted_match()
    assert digests2 == digests
    assert ([(type(e).__name__, e.unit_id, e.attacker_id) for e in events2]
            == [(type(e).__name__, e.unit_id, e.attacker_id) for e in events])


def test_the_attacker_is_hashed():
    """Property: the per-tick unit digest sees WHO is credited -- two ticks
    identical but for a hit's attacker hash differently. Breaks if the
    attacker is dropped from the hashed event record."""
    sim = _sim()
    u = _add(sim, "U", 3, 3, team=0)

    def h(attacker):
        sim.tick_events[:] = [UnitHitEvent(unit_id=u.id, damage=1,
                                           source="bullet",
                                           attacker_id=attacker)]
        return _capture_unit_state(sim)["hash"]
    assert h(5) != h(6)


# ---------------------------------------------------------------------------
# (g) indirect credit
# ---------------------------------------------------------------------------
def test_a_flame_jet_credits_its_own_heat_to_the_sprayer():
    """Property: a unit burned by a flamethrower's jet this tick is credited
    to the sprayer (cause heat), and the per-tick claims are cleared at the
    end of every tick. Breaks if the spray stops claiming its cone, the heat
    row stops reading the claim, or the claims outlive the tick."""
    sim = _sim()
    sprayer = _add(sim, "F", 3, 3, team=0)
    sprayer.weapon_id = "dragon_7"
    victim = _add(sim, "V", 8, 3, team=1)
    sim.set_aim(sprayer.id, FP_ONE, 0)
    sim.set_trigger(sprayer.id, True)
    heat_hits = []
    for _ in range(24):
        sim.step()
        sim.set_paused(False)
        heat_hits += [e for e in sim.tick_events
                      if isinstance(e, UnitHitEvent) and e.source == "heat"
                      and e.unit_id == victim.id]
        assert len(tick_credit(sim.gmap)) == 0      # cleared every tick
    assert heat_hits, "the jet must burn the victim"
    assert {e.attacker_id for e in heat_hits} == {sprayer.id}
    assert sprayer.id not in victim.recent_attackers   # indirect


def test_an_indirect_credit_is_not_an_attacker_to_return_fire_at():
    """Property: the idle stance's ``recent_attackers`` takes direct hits
    only -- an indirect (world-borne) credit rides the events but leaves the
    list as it was before #123. Breaks if apply_packet records indirect
    credit there (a behaviour change of the stance)."""
    u = Unit("U", x=2, y=2, team=0)
    u.id = 1
    events = []
    apply_packet(u, DamagePacket(1, HEAT, 7, indirect=True), events,
                 source="heat")
    assert u.recent_attackers == {}
    assert events[0].attacker_id == 7
    apply_packet(u, DamagePacket(1, KINETIC, 7), events, source="bullet")
    assert u.recent_attackers == {7: 1}


def test_tick_credit_first_claim_wins_and_the_environment_claims_nothing():
    """Property: a (channel, tile) keeps its FIRST claimant this tick (claims
    arrive in the sim's fixed order), -1 / None claim nothing, and clear()
    empties it. Breaks if a later claim overwrites, or the environment
    becomes a claimant."""
    tc = TickCredit()
    tc.claim(CREDIT_HEAT, [(1, 2)], 4)
    tc.claim(CREDIT_HEAT, [(1, 2), (1, 3)], 9)
    tc.claim(CREDIT_HEAT, [(5, 5)], NO_ATTACKER)
    tc.claim(CREDIT_HEAT, [(6, 6)], None)
    assert tc.attacker_at(CREDIT_HEAT, 1, 2) == 4
    assert tc.attacker_at(CREDIT_HEAT, 1, 3) == 9
    assert tc.attacker_at(CREDIT_HEAT, 5, 5) == NO_ATTACKER
    assert tc.attacker_at(2, 1, 2) == NO_ATTACKER     # another channel
    tc.clear()
    assert len(tc) == 0
