"""#195 1i -- reward profile v1 (``simulation/reward_v1.py``, ``rl_env``'s
``[reward] profile``; the contract ``docs/rl_agent_rung1_note_1i_2026-11-13.md``,
design ``docs/rl_agent_design_2026-11-11.md`` 6.1-6.9).

THE GATE -- the books close (design 6.8), three identities in float64 to
1e-9 on every decision of every scene:

1. per team, the steps' ``d_balance`` add up to ``B(end) - B(start)``, both
   read from state by THIS file's own balance (:func:`_balance_ref`, never
   ``reward_v1.standing``);
2. per team, the own group over its rows plus its uncredited rest equals
   the step's ``d_balance`` (before the records-ended mask);
3. per team, the team group over its rows equals the step's ``d_balance``.

Each scene names what it must have met (an overkill hit, a hit on a
corpse, a turning, a zombie's bite on a marine, an environment hit,
friendly fire, an elimination, a win) or FAILS as vacuous; the counts are
printed. Each test names the property it protects and the change that must
break it. No test names a hit-point figure (the pending zombie balance:
health enters only as hp over the unit's own bar) and none pins the
counter tuple's length or order.
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import replace
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

import control_policy as cp  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import rl_env  # noqa: E402
from simulation import reward_v1 as R  # noqa: E402
from simulation.events import UnitHitEvent, UnitKilledEvent  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.match_session import MatchSession  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402

TOL = 1e-9
FIRETEAM = R.PROFILES["fireteam"]
G_BOOK = R.group_weights(FIRETEAM, R.BOOK_COUNTERS)
OUT, OWN, TEAM, FF, ORDER = range(5)
assert R.REWARD_GROUPS == ("outcome", "own", "team", "friendly_fire",
                           "order")
CI = {n: i for i, n in enumerate(R.BOOK_COUNTERS)}
Q = 1 << 16


# ===========================================================================
# Part 1 -- the books on hand-built events (a stand-in sim: units, rosters,
# the picture's bars, the tick's events; every hp change as apply_packet
# makes it: hp_q -= d_q, one UnitHitEvent, a UnitKilledEvent on alive -> dead)
# ===========================================================================
FAKE_BAR = 64.0          # a stand-in unit's bar (not a game figure)


class _U:
    def __init__(self, uid, team, slot, bar=FAKE_BAR, zombie=False):
        self.id, self.team, self.slot = uid, team, slot
        self.hp_q = int(bar * Q)
        self.alive, self.is_zombie = True, zombie

    @property
    def current_hp(self):
        return self.hp_q / Q


class _Sim:
    """What the books read: ``units``, ``roster(t)``, ``team_picture.bars``,
    ``tick``, ``tick_events``."""

    def __init__(self, teams, zombies=0):
        self.units, self._rosters, uid = [], {}, 0
        for t, n in enumerate(teams):
            for s in range(n):
                u = _U(uid, t, s)
                self.units.append(u)
                self._rosters.setdefault(t, []).append(u)
                uid += 1
        for s in range(zombies):
            self.units.append(_U(uid, ZOMBIE_TEAM, s, zombie=True))
            uid += 1
        self.team_picture = SimpleNamespace(
            bars={u.id: u.current_hp for u in self.units})
        self.tick, self.tick_events = 0, []
        self.spec = SimpleNamespace(teams=tuple(teams))

    def roster(self, t):
        return list(self._rosters.get(t, ()))

    def u(self, uid):
        return self.units[uid]

    # -- a tick's writes, as apply_packet makes them ------------------------
    def hit(self, v, frac, a=-1, kill_mark=False):
        """Damage ``frac`` of ``v``'s bar (negative: a heal), credited to
        ``a``."""
        u = self.u(v)
        d_q = int(round(frac * self.team_picture.bars[v] * Q))
        was = u.alive
        u.hp_q -= d_q
        self.tick_events.append(UnitHitEvent(v, d_q / Q, "bullet", a))
        if was and u.hp_q <= 0:
            u.alive = False
            u.killed_by_zombie = kill_mark
            self.tick_events.append(UnitKilledEvent(v, "bullet", a))

    def turn(self, v, frac=1.0):
        """The conversion (on_tick_end) and the picture's new bar (slot
        11)."""
        u = self.u(v)
        u.team, u.is_zombie, u.alive = ZOMBIE_TEAM, True, True
        u.hp_q = int(frac * FAKE_BAR * Q)
        self.team_picture.bars[v] = u.current_hp


def _books(sim, terms=FIRETEAM, S=None):
    S = S or max(sim.spec.teams)
    return R.RewardBooks(sim, sim.spec, S, terms)


def _tick(sim, books, fn=None):
    """One tick: begin, the tick's writes, end; returns the books' step."""
    books.start_step()
    books.begin(sim)
    sim.tick_events = []
    if fn is not None:
        fn(sim)
    sim.tick += 1
    books.end(sim)
    return books


def _groups(counters, terms=FIRETEAM):
    G = R.group_weights(terms, R.BOOK_COUNTERS)
    return (counters[:, :, None] * G[None]).sum(axis=1)


def _check_identities_2_3(books, terms=FIRETEAM):
    """Identities 2 and 3 on the books' last step; returns the per-row
    groups (unmasked)."""
    g = _groups(books.raw, terms)
    for t, rows in books.rows.items():
        own = g[rows, OWN].sum()
        team = g[rows, TEAM].sum()
        assert abs(own + books.rest[t] - books.d_balance[t]) < TOL, \
            (t, own, books.rest[t], books.d_balance[t])
        assert abs(team - books.d_balance[t]) < TOL, (t, team,
                                                       books.d_balance[t])
    return g


def test_bar_ratio_counts_health_in_bars_clipped():
    """Property: health counts in bars -- hp over the unit's own bar,
    clipped at one above and at zero below; no bar reads 0. Breaks if
    health above the bar earns more than one bar or a corpse's negative hp
    counts."""
    b = R.bar_q_of(FAKE_BAR)
    assert R.bar_ratio(b // 2, b) == 0.5
    assert R.bar_ratio(3 * b, b) == 1.0
    assert R.bar_ratio(-b, b) == 0.0
    assert R.bar_ratio(b, 0) == 0.0


def test_an_overkill_pays_only_the_health_that_was_there():
    """Property (6.8, identity 2): a hit far stronger than the health left
    pays the attacker ``bar_ratio(h)`` and nothing more, the victim loses
    exactly that, its death once, and own + rest == the balance change.
    Breaks if the full force is credited (the note's break (i))."""
    sim = _Sim((1, 1))
    books = _books(sim)
    _tick(sim, books, lambda s: s.hit(1, 0.875, a=0))    # 1/8 bar left
    _check_identities_2_3(books)
    _tick(sim, books, lambda s: s.hit(1, 5.0, a=0))      # an overkill
    _check_identities_2_3(books)
    c = books.raw
    assert abs(c[0, CI["bars_enemy"]] - 0.125) < TOL
    assert abs(c[1, CI["bars_lost"]] - 0.125) < TOL
    assert c[0, CI["kills_enemy"]] == 1 and c[1, CI["died"]] == 1
    assert books.stats["overkill_hits"] == 1


def test_a_hit_on_a_corpse_pays_nothing_and_blames_nobody():
    """Property (6.8): a credited hit -- an enemy's and a team-mate's -- on
    a dead unit leaves every counter of every row unchanged and the balance
    where it was. Breaks if the corpse's negative hp is not clipped, or the
    alive term is paid twice."""
    sim = _Sim((2, 1))
    books = _books(sim)

    def kill_then_hit(s):
        s.hit(2, 2.0, a=0)          # kills
        s.hit(2, 1.0, a=1)          # an enemy's hit on the corpse
    _tick(sim, books, kill_then_hit)
    _check_identities_2_3(books)
    _tick(sim, books, lambda s: (s.hit(2, 1.0, a=0), s.hit(0, 0.0, a=1)))
    _check_identities_2_3(books)
    assert not books.raw.any() and not books.rest.any()
    assert not books.d_balance.any()
    sim2 = _Sim((2, 1))
    books2 = _books(sim2)
    _tick(sim2, books2, lambda s: s.hit(1, 2.0, a=2))   # team 0's 1 dies
    _tick(sim2, books2, lambda s: s.hit(1, 1.0, a=0))   # friendly, corpse
    assert not books2.raw.any() and not books2.d_balance.any()
    assert books.stats["corpse_hits"] >= 2


def test_healing_an_enemy_takes_back_what_hurting_it_paid():
    """Property (6.8): hurt an enemy by x bars, give it back -- inside a
    tick (a HEAL event credited to a unit) or between ticks (an hp write the
    next ``begin`` settles) -- and the hurting team's team part over the
    two is 0; the hurter's own part is unchanged by the heal; the healer's
    own part gets nothing; the healed soldier's own ``bars_lost`` goes back
    down. Breaks if a heal is ignored (begin skipped, or ``d <= 0`` dropped
    from the walk) or credited to the healer's own part."""
    for between in (False, True):
        sim = _Sim((1, 1))
        books = _books(sim)
        _tick(sim, books, lambda s: s.hit(1, 0.25, a=0))
        g1 = _check_identities_2_3(books).copy()
        own0 = books.raw[0].copy()
        if between:
            sim.u(1).hp_q += int(0.25 * FAKE_BAR * Q)     # outside a tick
            _tick(sim, books)
            assert books.outside_changes == 1
        else:
            _tick(sim, books, lambda s: s.hit(1, -0.25, a=0))  # 0 heals 1
        g2 = _check_identities_2_3(books)
        assert abs(g1[0, TEAM] + g2[0, TEAM]) < TOL
        assert g1[0, TEAM] > 0
        assert not books.raw[0, :CI["share_bars"]].any()   # nothing own
        assert own0[CI["bars_enemy"]] > 0
        assert abs(books.raw[1, CI["bars_lost"]] + 0.25) < TOL


def test_friendly_fire_blames_the_credited_unit_and_no_other():
    """Property (6.8): a credited hit on a team-mate puts ``bars_team`` on
    the shooter's row only (and ordinary lost health on the victim's), never
    on the balance; a self-hit is lost health only; a turned soldier's
    bites on its old squad blame nobody. Breaks if blame is shared out, or a
    turned unit is still mapped to its old row."""
    sim = _Sim((3, 1), zombies=1)
    books = _books(sim)
    _tick(sim, books, lambda s: s.hit(1, 0.5, a=0))
    _check_identities_2_3(books)
    assert books.raw[0, CI["bars_team"]] == 0.5
    assert books.raw[1, CI["bars_lost"]] == 0.5
    assert not books.raw[2, :CI["share_bars"]].any()   # no own, no blame
    assert not books.raw[1, CI["bars_team"]]
    _tick(sim, books, lambda s: s.hit(0, 0.25, a=0))    # self-hit
    assert books.raw[0, CI["bars_lost"]] == 0.25
    assert not books.raw[0, CI["bars_team"]]
    # unit 2 dies to the zombie (4), turns, then bites 0 and 1
    _tick(sim, books, lambda s: (s.hit(2, 2.0, a=4, kill_mark=True),
                                 s.turn(2)))
    _check_identities_2_3(books)
    _tick(sim, books, lambda s: (s.hit(0, 0.1, a=2), s.hit(1, 0.1, a=2)))
    _check_identities_2_3(books)
    assert not books.raw[:, CI["bars_team"]].any()
    assert not books.raw[2, :CI["share_bars"]].any()
    assert books.stats["zombie_bites"] >= 2


def test_the_turning_is_uncredited_rest_for_every_team():
    """Property (6.2, 6.8): a turning adds a zombie at its new bar; the
    change (``-zb`` per team) is in every marine team's uncredited rest and
    in no row's own part, and identity 2 holds on that tick. Breaks if a
    turning is dropped (the note's break (ii)) or credited to a row."""
    sim = _Sim((1, 1), zombies=1)
    books = _books(sim)
    _tick(sim, books, lambda s: (s.hit(0, 2.0, a=2, kill_mark=True),
                                 s.turn(0)))
    _check_identities_2_3(books)
    mb, ma, zb = (FIRETEAM["marine_bar"], FIRETEAM["marine_alive"],
                  FIRETEAM["zombie_bar"])
    # team 0's own row holds its loss; its rest is the turning alone
    assert abs(books.rest[0] - (-zb)) < TOL
    # team 1: the horde's (uncredited) kill of its enemy, then the turning
    assert abs(books.rest[1] - (mb + ma - zb)) < TOL
    assert books.stats["turnings"] == 1


def test_the_zombie_term_is_not_a_mirror():
    """Property (6.2): a zombie's lost health raises EVERY marine team's
    balance by the same amount, whoever took it. Breaks if the zombie term
    is credited to one team only."""
    sim = _Sim((1, 1, 1), zombies=1)
    books = _books(sim)
    _tick(sim, books, lambda s: s.hit(3, 0.5, a=0))
    _check_identities_2_3(books)
    zb = FIRETEAM["zombie_bar"]
    assert np.allclose(books.d_balance, zb * 0.5, atol=TOL)
    assert books.raw[0, CI["bars_zombie"]] == 0.5


def test_outcomes_are_paid_once_to_every_row_dead_included():
    """Property (6.2, 6.5): a win, an elimination and the clock are paid to
    every row of the team exactly once, the dead rows included; two last
    teams wiped in one tick are two eliminations (and nothing more on a
    draw). Breaks if an outcome is paid twice, skipped on a dead row, or a
    simultaneous wipe is read from ``result()`` as a draw."""
    from simulation.ruleset import MatchResult
    # a win: team 1 eliminated (both its rows), team 0 wins (its dead row
    # included)
    sim = _Sim((2, 2))
    books = _books(sim)
    _tick(sim, books, lambda s: s.hit(1, 2.0, a=2))         # 0's 1 dies
    _tick(sim, books, lambda s: (s.hit(2, 2.0, a=0), s.hit(3, 2.0, a=0)))
    books.finish(MatchResult("win", 0, sim.tick))
    c = books.raw
    assert list(c[:, CI["eliminated"]]) == [0, 0, 1, 1]
    assert list(c[:, CI["won"]]) == [1, 1, 0, 0]
    assert not c[:, CI["clock"]].any()
    assert books.rows_ended.all()
    # a simultaneous wipe
    sim = _Sim((1, 1))
    books = _books(sim)
    _tick(sim, books, lambda s: (s.hit(0, 2.0, a=-1), s.hit(1, 2.0, a=-1)))
    books.finish(MatchResult("draw", None, sim.tick))
    assert list(books.raw[:, CI["eliminated"]]) == [1, 1]
    assert not books.raw[:, CI["won"]].any()
    assert not books.raw[:, CI["clock"]].any()
    # the clock: every row of each standing team, the dead included
    sim = _Sim((2, 1))
    books = _books(sim)
    _tick(sim, books, lambda s: s.hit(0, 2.0, a=2))
    books.start_step()
    books.finish(MatchResult("draw", None, sim.tick))
    assert list(books.raw[:3, CI["clock"]]) == [1, 1, 1]
    assert np.allclose(books.outcome[:2], FIRETEAM["clock"])
    with pytest.raises(RuntimeError, match="finish"):
        books.finish(MatchResult("draw", None, sim.tick))


def test_records_end_at_an_elimination_in_a_match_that_goes_on():
    """Property (6.6): in a three-team match, a team's rows earn nothing
    after its elimination (posthumous credit included), are marked
    ``rows_ended`` on that step only, while the books keep the team's
    numbers (identities 2 and 3 still close). Breaks if records do not end,
    or the payment of the elimination step itself is masked."""
    sim = _Sim((1, 1, 1))
    books = _books(sim)
    _tick(sim, books, lambda s: s.hit(2, 2.0, a=0))
    assert books.counters[2, CI["eliminated"]] == 1
    assert books.rows_ended[2] and not books.rows_ended[:2].any()
    # posthumous: a hit credited to the eliminated unit (a late grenade)
    _tick(sim, books, lambda s: s.hit(1, 0.5, a=2))
    _check_identities_2_3(books)
    assert not books.counters[2].any() and books.raw[2].any()
    assert not books.rows_ended.any()


def test_the_check_raises_on_an_hp_write_the_events_do_not_explain():
    """Property: an hp written inside a tick with no event raises
    ``RewardBooksMismatch`` naming the unit and the tick. Breaks if the
    books absorb an unexplained change into the rest."""
    sim = _Sim((1, 1))
    books = _books(sim)

    def silent(s):
        s.u(1).hp_q -= 3
    with pytest.raises(R.RewardBooksMismatch, match=r"tick 1: unit 1"):
        _tick(sim, books, silent)


def test_sizes_come_from_the_profile_and_the_mirror_holds():
    """Property (6.5, 6.9): overriding a term (``zombie_bar``, ``clock``)
    moves exactly the counters' weights it names in the group matrix; every
    balance term is ONE weight used with its fixed sign on the own and the
    team counters alike; identities 2 and 3 hold under the override. Breaks
    if a size is a literal in the walk or the own and team signs drift
    apart."""
    names = R.BOOK_COUNTERS + ("team_kills",)
    base = R.group_weights(FIRETEAM, names)
    for term, moved in (("zombie_bar", {"bars_zombie",
                                        "share_zombie_bars"}),
                        ("clock", {"clock"})):
        terms = R.profile_terms("fireteam", {term: 0.125})
        diff = np.flatnonzero((R.group_weights(terms, names)
                               != base).any(axis=1))
        assert {names[i] for i in diff} == moved
    G = base
    ci = {n: i for i, n in enumerate(names)}
    for own, team, sign in (("bars_enemy", "share_bars", 1),
                            ("kills_enemy", "share_alive", 1),
                            ("bars_lost", "share_enemy_bars", 1),
                            ("bars_zombie", "share_zombie_bars", -1)):
        assert G[ci[own], OWN] == sign * G[ci[team], TEAM] != 0
    assert not G[:, ORDER].any()
    terms = R.profile_terms("fireteam", {"zombie_bar": 0.1,
                                         "marine_alive": 2.0})
    sim = _Sim((2, 1), zombies=1)
    books = _books(sim, terms)
    _tick(sim, books, lambda s: (s.hit(3, 0.5, a=0), s.hit(2, 2.0, a=1)))
    _check_identities_2_3(books, terms)
    with pytest.raises(ValueError, match="'hits_dealt' is not a term"):
        R.profile_terms("fireteam", {"hits_dealt": 1.0})


# ===========================================================================
# Part 2 -- the gate on real matches (in-process EnvRunners)
# ===========================================================================
POL = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                controller="policy")
POL1 = replace(POL, n=1)
COLS = {c.name: i for i, c in enumerate(cp.action_schema().columns)}


def _level(root, name, spec, rooms, w=40, h=24):
    """A hull box with ``rooms`` and ONE room over its floor as its room
    graph, written through ``level_lib`` (the match.toml on idle
    controllers; the run file sets them)."""
    lvl = root / name
    lvl.mkdir(parents=True)
    level_lib.write_level_header(lvl, name=name, tile_size_m=1.0)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", rooms)
    level_lib.write_match(lvl, replace(spec, teams=tuple(
        replace(t, controller="idle") for t in spec.teams)))
    grid = np.full((h, w), -1, dtype=np.int32)
    grid[1:h - 1, 1:w - 1] = 0
    level_lib.write_room_graph(
        lvl, [dict(id=0, x=1, y=1, w=w - 2, h=h - 2, kind="room")], [], grid)
    return level_loader.load(name, str(root))


ZOMBIE_ROOMS = [   # tests/test_rl_env.py's box: the horde eats team 0
    SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=4),
    SpawnRoom(team=1, x=30, y=14, w=8, h=8, max_per_team=4),
    SpawnRoom(team=ZOMBIE_TEAM, x=12, y=2, w=8, h=8, max_per_team=4)]
THREE_ROOMS = [    # the horde eats team 2, then team 0; team 1 is left
    SpawnRoom(team=0, x=2, y=14, w=8, h=8, max_per_team=4),
    SpawnRoom(team=1, x=30, y=2, w=8, h=8, max_per_team=4),
    SpawnRoom(team=2, x=2, y=2, w=8, h=8, max_per_team=4),
    SpawnRoom(team=ZOMBIE_TEAM, x=12, y=2, w=8, h=8, max_per_team=4)]


def _pair_rooms(y1):
    return [SpawnRoom(team=0, x=10, y=10, w=10, h=4, max_per_team=2),
            SpawnRoom(team=1, x=10, y=y1, w=10, h=4, max_per_team=2)]


def _scene_level(root, scene):
    if scene == "zombie_box":
        return _level(root, "zbox", MatchSpec(
            teams=(POL, POL), seed=1, zombies=3, time_limit_s=20.0),
            ZOMBIE_ROOMS)
    if scene == "three_teams":
        return _level(root, "tbox", MatchSpec(
            teams=(POL, POL, POL), seed=1, zombies=3, time_limit_s=60.0),
            THREE_ROOMS)
    if scene == "charge_ff":       # a C4 among a pair and an enemy
        return _level(root, "cbox", MatchSpec(
            teams=(POL, POL1), seed=1, zombies=0, time_limit_s=20.0),
            _pair_rooms(14))
    if scene == "charge_wipe":     # one a side, both in the blast
        return _level(root, "wbox", MatchSpec(
            teams=(POL1, POL1), seed=1, zombies=0, time_limit_s=20.0),
            _pair_rooms(12))
    if scene == "melee_corpse":    # two blows land on one sliver of health
        return _level(root, "mbox", MatchSpec(
            teams=(POL, POL1), seed=1, zombies=0, time_limit_s=5.0),
            [SpawnRoom(team=0, x=2, y=2, w=10, h=4, max_per_team=2),
             SpawnRoom(team=1, x=20, y=14, w=10, h=4, max_per_team=2)])
    if scene == "clock_box":
        return _level(root, "kbox", MatchSpec(
            teams=(POL, POL), seed=1, zombies=0, time_limit_s=2.0),
            ZOMBIE_ROOMS[:2])
    raise ValueError(scene)


def _ship(root, time_limit_s=None):
    dst = root / "ship_demo"
    shutil.copytree(ROOT / "levels" / "ship_demo", dst)
    if time_limit_s is not None:
        src = level_loader.load("ship_demo", str(ROOT / "levels"))
        level_lib.write_match(dst, replace(src.match,
                                           time_limit_s=time_limit_s))
    return level_loader.load("ship_demo", str(root))


def _run_text(level, root, *, seed=7, decimation=4, policy="policy",
              action="raw", reward='profile = "fireteam"\n',
              full_view=False, n_envs=1):
    return ("[run]\n"
            f"seed = {seed}\nn_envs = {n_envs}\ndecimation = {decimation}\n"
            f"levels = [\"{level}\"]\npolicy = \"{policy}\"\n"
            f"levels_dir = '{Path(root).as_posix()}'\n"
            "[obs]\nspec = \"v1\"\n"
            f"full_view = {'true' if full_view else 'false'}\n"
            f"[action]\nspec = \"{action}\"\n"
            f"[reward]\n{reward}")


def _runner(root, level, **kw):
    p = root / f"run_{level}.toml"
    p.write_text(_run_text(level, root, **kw), encoding="utf-8")
    run = rl_env.load_run(p)
    levels = rl_env.load_levels(run)
    layout = rl_env.make_layout(run, levels)
    return rl_env.EnvRunner(0, run, layout, levels, breach_physics=bp), run


def _balance_ref(sim, t, terms):
    """THE TEST'S OWN balance of team ``t`` from state (identity 1's second
    implementation; never ``reward_v1``'s)."""
    bars = sim.team_picture.bars
    mb, ma, zb = (terms["marine_bar"], terms["marine_alive"],
                  terms["zombie_bar"])
    total = 0.0
    for u in sim.units:
        if not u.alive:
            continue
        bar = int(round(bars[u.id] * Q))
        h = min(max(int(u.hp_q), 0), bar) / bar
        if u.is_zombie:
            total -= zb * h
        elif int(u.team) == t:
            total += mb * h + ma
        else:
            total -= mb * h + ma
    return total


def _raw_rows(n_rows, **cols):
    rows = cp.zero_rows(n_rows)
    for name, (slots, value) in cols.items():
        rows[list(slots), COLS[name]] = value
    return rows


def _drive(r, run, actions, *, setup=None, max_steps=100_000):
    """One whole episode of runner ``r`` with the gate's three identities
    checked on every step. ``actions(r, step, out)`` -> the step's rows
    (``out`` the observation, built only under ``"v1"`` actions);
    ``setup(sim)`` writes the scene's state after the reset (between ticks:
    the first ``begin`` settles it). Returns the books' stats, the steps'
    outputs and the per-step books' numbers."""
    terms = run.terms()
    r.reset()
    sim = r.session.sim
    n_match = len(r.session.spec.teams)
    b0 = [_balance_ref(sim, t, terms) for t in range(n_match)]
    if setup is not None:
        setup(sim)
    out = None
    if r.layout.action_spec == "v1":
        out = {k: np.zeros(s[1:], d)
               for k, (s, d) in r.layout.obs_shapes().items()}
    sum_db = np.zeros(n_match)
    steps = []
    for k in range(max_steps):
        if out is not None:
            r.observe(out)
        res = r.step(actions(r, k, out))
        books = r.books
        _check_identities_2_3(books, terms)
        sum_db += books.d_balance[:n_match]
        steps.append((res, books.raw.copy(), books.rest.copy(),
                      books.d_balance.copy()))
        if res["ended"]:
            break
    assert res["ended"], "the episode did not end"
    for t in range(n_match):                         # identity 1
        b1 = _balance_ref(sim, t, terms)
        assert abs(sum_db[t] - (b1 - b0[t])) < TOL, (t, sum_db[t], b1, b0)
    return dict(r.books.stats, outside=r.books.outside_changes), steps


def _zero(r, k, out):
    return cp.zero_rows(r.layout.n_rows)


def _legal_v1(seed):
    import rl_smoke
    rng = np.random.default_rng(seed)

    def fn(r, k, out):
        o = {key: v[None] for key, v in out.items()}
        return rl_smoke.legal_v1_rows(rng, o)[0]
    return fn


def _first_step(rows):
    def fn(r, k, out):
        return rows if k == 0 else cp.zero_rows(r.layout.n_rows)
    return fn


def _melee_setup(sim):
    from simulation.turning import place
    a, b, v = sim.units
    a.set_pos(10, 10)
    b.set_pos(10, 13)
    v.set_pos(13, 11.5)
    place(a, 0)
    place(b, 0)
    v.hp_q = v.hp_q // 64                 # a sliver of its bar


#: scene -> (what the scene must have met, its runner arguments, actions,
#: setup)
SCENES = {
    "zombie_box": (("turnings", "zombie_bites", "eliminations", "wins",
                    "overkill_hits"), {}, _zero, None),
    "three_teams": (("eliminations", "wins", "turnings"),
                    {"policy": "idle"}, _zero, None),
    "charge_ff": (("friendly_fire", "overkill_hits", "eliminations",
                   "wins"), {},
                  _first_step(_raw_rows(4, **{
                      "plant_charge_intent.fuse_ticks": ([0], 120)})),
                  None),
    "charge_wipe": (("eliminations",), {},
                    _first_step(_raw_rows(4, **{
                        "plant_charge_intent.fuse_ticks": ([0], 120)})),
                    None),
    "melee_corpse": (("corpse_hits", "overkill_hits", "wins"), {},
                     _first_step(_raw_rows(4, melee_intent=([0, 1], 1))),
                     _melee_setup),
}


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_the_books_close_on_the_boxes(tmp_path, scene, capsys):
    """Property (the gate, identities 1-3): on every decision of a whole
    match the books close -- the zombie box (bites, turnings, a converted
    zombie's bites on its old team, an elimination and a win), three teams
    (one eliminated mid-match), a C4 among a pair and an enemy (friendly
    fire), a C4 wiping one soldier a side, two blows on a sliver (a hit on
    a corpse) -- and the scene met what it is there for (else it is
    vacuous). Breaks if a change of state is missed (the note's breaks
    (i)-(iii)), or ``d_balance`` is computed from events instead of
    state."""
    need, kw, actions, setup = SCENES[scene]
    _scene_level(tmp_path, scene)
    level = {"zombie_box": "zbox", "three_teams": "tbox", "charge_ff": "cbox",
             "charge_wipe": "wbox", "melee_corpse": "mbox"}[scene]
    r, run = _runner(tmp_path, level, **kw)
    stats, steps = _drive(r, run, actions, setup=setup)
    print(f"{scene}: {stats}")
    for k in need:
        assert stats[k] > 0, f"{scene} is vacuous: no {k} ({stats})"


@pytest.mark.parametrize("seed", [3, 8])
def test_the_books_close_on_ship_demo_policy_play(tmp_path, seed):
    """Property (the gate on ship_demo): every team on ``policy`` with the
    1h legal-uniform sampler, ``"v1"`` actions, the books close on every
    decision of a (shortened) match. Breaks as the boxes' gate does."""
    _ship(tmp_path, time_limit_s=40.0)
    r, run = _runner(tmp_path, "ship_demo", seed=seed, action="v1")
    stats, _ = _drive(r, run, _legal_v1(seed))
    print(f"ship_demo policy seed {seed}: {stats}")


@pytest.mark.parametrize("seed", [3, 8])
def test_the_books_close_on_ship_demo_bots_matches(tmp_path, seed):
    """Property (the gate on ship_demo): both teams on ``bots`` (the bots
    kill; random rows rarely do), whole matches; the books close on every
    decision and the match met overkill hits and environment hits. Breaks
    as the boxes' gate does."""
    _ship(tmp_path)
    r, run = _runner(tmp_path, "ship_demo", seed=seed, policy="bots")
    stats, _ = _drive(r, run, _zero)
    print(f"ship_demo bots seed {seed}: {stats}")
    for k in ("overkill_hits", "uncredited_hits"):
        assert stats[k] > 0, f"ship_demo bots seed {seed}: no {k} ({stats})"


def test_a_heal_between_steps_is_settled_by_begin(tmp_path):
    """Property (6.8, the books' begin): a blow hurts an enemy, its health
    is written back between the steps (the HEAL a test writes), and the
    next step's team part takes back exactly what the blow paid, the
    striker's own part is unchanged by the heal, and the identities close
    on both steps. Breaks if ``begin`` loses the out-of-tick residual (the
    note's break (iii))."""
    _scene_level(tmp_path, "melee_corpse")
    r, run = _runner(tmp_path, "mbox")
    r.reset()
    sim = r.session.sim
    a, b, v = sim.units
    from simulation.turning import place
    a.set_pos(10, 10)
    v.set_pos(13, 10)
    place(a, 0)
    full = v.hp_q
    r.step(_raw_rows(4, melee_intent=([0], 1)))
    g1 = _check_identities_2_3(r.books).copy()
    assert v.hp_q < full and g1[0, TEAM] > 0 and g1[0, OWN] > 0
    v.hp_q = full                                         # given back
    r.step(cp.zero_rows(4))
    g2 = _check_identities_2_3(r.books)
    assert r.books.outside_changes == 1
    assert abs(g1[0, TEAM] + g2[0, TEAM]) < TOL
    assert g2[0, OWN] == 0
    S = r.layout.n_slots
    assert r.books.raw[S, CI["bars_lost"]] < 0


def test_the_books_check_raises_on_a_silent_hp_write(tmp_path,
                                                     monkeypatch):
    """Property: on a real match, an hp written inside a tick outside
    ``apply_packet`` raises ``RewardBooksMismatch`` naming the unit and the
    tick. Breaks if the books absorb an unexplained change."""
    _scene_level(tmp_path, "zombie_box")
    r, run = _runner(tmp_path, "zbox")
    r.reset()
    sim = r.session.sim
    real = sim.step
    victim = sim.roster(1)[0]

    def step():
        real()
        victim.hp_q -= 1
    monkeypatch.setattr(sim, "step", step)
    with pytest.raises(R.RewardBooksMismatch,
                       match=rf"tick 1: unit {victim.id}'s"):
        r.step(cp.zero_rows(r.layout.n_rows))


def test_three_sides(tmp_path):
    """Property (6.6): in a three-team box the eliminated team gets
    ``eliminated`` at its elimination, ``rows_ended`` there and zeros
    after; the last team wins; a team's balance rises on a step it dealt
    nothing while the other two are hurt. Breaks if records do not end, or
    enemies of different teams are weighted differently."""
    _scene_level(tmp_path, "three_teams")
    r, run = _runner(tmp_path, "tbox", policy="idle")
    S = r.layout.n_slots
    _, steps = _drive(r, run, _zero)
    rows2 = slice(2 * S, 2 * S + 2)
    k_end = next(k for k, s in enumerate(steps)
                 if s[0]["rows_ended"][rows2].all())
    res = steps[k_end][0]
    assert np.allclose(res["reward"][rows2, OUT], FIRETEAM["eliminated"])
    for later, *_ in steps[k_end + 1:]:
        assert not later["reward"][rows2].any()
        assert not later["counters"][rows2].any()
        assert not later["rows_ended"][rows2].any()
    last = steps[-1][0]
    assert last["result"] == {"outcome": "win", "team": 1,
                              "ticks": last["result"]["ticks"]}
    assert np.allclose(last["reward"][S:S + 2, OUT], FIRETEAM["win"])
    rose = [s for s in steps if s[0]["team_balance"][1] > 0
            and not s[0]["reward"][S:S + 2, OWN].any()]
    assert rose, "team 1's balance never rose on a step it dealt nothing"


def test_the_same_match_gives_the_same_rewards(tmp_path):
    """Property (6.8): two runners on one seed and one row stream give
    equal counters and rewards every step; a recorded match re-run through
    ``MatchSession(level, replay=rec)`` with fresh books gives the same
    per-tick numbers. Breaks if the books read anything but state, events
    and bars (a clock, a dict order, a draw)."""
    lv = _scene_level(tmp_path, "zombie_box")
    rng = np.random.default_rng(5)
    r1, run = _runner(tmp_path, "zbox")
    r2, _ = _runner(tmp_path, "zbox")
    r1.reset()
    r2.reset()
    U = r1.layout.n_rows
    while True:
        rows = cp.random_action_rows(rng, U)
        o1, o2 = r1.step(rows), r2.step(rows)
        for key in ("counters", "reward", "team_balance", "team_uncredited",
                    "team_outcome", "rows_ended"):
            assert np.array_equal(o1[key], o2[key]), key
        if o1["ended"]:
            break
    spec = replace(lv.match, teams=tuple(replace(t, controller="idle")
                                         for t in lv.match.teams))

    def per_tick(ses):
        books = R.RewardBooks(ses.sim, ses.spec, 4, FIRETEAM)
        got = []
        while not ses.is_terminal():
            books.start_step()
            books.begin(ses.sim)
            ses.tick()
            books.end(ses.sim)
            got.append((books.raw.copy(), books.rest.copy(),
                        books.d_balance.copy(), books.outcome.copy()))
        books.start_step()
        books.finish(ses.result())
        got.append((books.raw.copy(), books.outcome.copy()))
        return got
    ses = MatchSession(lv, spec, seed=11, breach_physics=bp)
    ses.record(60)
    first = per_tick(ses)
    rec = ses.finish()
    again = per_tick(MatchSession(lv, replay=rec, breach_physics=bp))
    assert len(first) == len(again)
    for a_, b_ in zip(first, again):
        for x, y in zip(a_, b_):
            assert np.array_equal(x, y)


def _env_run(tmp_path, level, **kw):
    p = tmp_path / f"env_{level}.toml"
    p.write_text(_run_text(level, tmp_path, n_envs=1, **kw),
                 encoding="utf-8")
    return rl_env.load_run(p)


def test_the_five_numbers_and_the_clock_as_an_ending(tmp_path):
    """Property (6.3, 6.5, 7.2): under ``"fireteam"`` ``reward`` is ``(N,
    U, 5)`` in ``REWARD_GROUPS`` order; a row with ``in_match`` 0 is all
    zeros; a dead and a turned row keep the team and outcome groups (not
    zero) and nothing in own / friendly fire after it fell; ``order`` is 0;
    the per-team blocks and ``rows_ended`` are served; a time-limit end is
    ``terminated`` and not ``truncated`` and pays the clock to every row.
    Breaks if a dead row is skipped (the share divided among the living),
    the env sums the groups, or the clock is reported as a cut under the
    profile."""
    _scene_level(tmp_path, "zombie_box")
    run = _env_run(tmp_path, "zbox")
    with rl_env.BatchedEnv(run, n_workers=1) as env:
        N, U, S, T = (env.n_envs, env.layout.n_rows, env.layout.n_slots,
                      env.layout.n_teams)
        obs, _ = env.reset()
        fell_team, fell_own, end = [], [], None
        dead = np.zeros(U, bool)
        for _ in range(10_000):
            acts = np.broadcast_to(cp.zero_rows(U), (N, U, env.layout
                                                    .n_actions)).copy()
            obs2, rew, term, trunc, info = env.step(acts)
            assert rew.shape == (N, U, len(R.REWARD_GROUPS))
            assert rew.dtype == np.float32
            for key in ("team_balance", "team_uncredited", "team_outcome"):
                assert info[key].shape == (N, T)
            assert info["rows_ended"].shape == (N, U)
            in_match = (info["final_obs"]["in_match"] if info["ended"][0]
                        else obs2["in_match"])
            assert not rew[0][in_match[0] == 0].any()
            assert not rew[..., ORDER].any()
            fell_team.extend(rew[0, dead, TEAM].tolist())
            fell_own.extend(rew[0, dead, OWN].tolist())
            if info["ended"][0]:
                end = (rew[0].copy(), bool(term[0]), bool(trunc[0]),
                       info["final_obs"]["in_match"][0].copy())
                break
            dead |= (obs2["in_match"][0] == 1) & (obs2["valid"][0] == 0)
        assert dead[:S].any()                      # team 0 fell and turned
        assert any(v != 0 for v in fell_team)
        assert not any(fell_own)
        rew_end, term_, trunc_, im = end
        assert term_ and not trunc_
        assert (rew_end[:S][im[:S] == 1, OUT] == FIRETEAM["eliminated"]).all()
    _scene_level(tmp_path, "clock_box")
    run = _env_run(tmp_path, "kbox")
    with rl_env.BatchedEnv(run, n_workers=1) as env:
        N, U = env.n_envs, env.layout.n_rows
        env.reset()
        for _ in range(1000):
            acts = np.broadcast_to(cp.zero_rows(U), (N, U, env.layout
                                                    .n_actions)).copy()
            _, rew, term, trunc, info = env.step(acts)
            if info["ended"][0]:
                break
        assert info["result"][0]["outcome"] == "draw"
        assert term[0] and not trunc[0]
        im = info["final_obs"]["in_match"][0] == 1
        assert np.allclose(rew[0][im, OUT], FIRETEAM["clock"])
        assert info["rows_ended"][0][im].all()


def test_a_policy_never_reads_a_reward(tmp_path):
    """Property (obs spec 4.1, "1i's trainer test"): ``obs`` holds no
    reward, counter or team key and ``team_rows(obs, t)`` hands a policy
    exactly ``obs``'s keys. Breaks if a reward key is added to ``obs``."""
    _scene_level(tmp_path, "zombie_box")
    run = _env_run(tmp_path, "zbox")
    with rl_env.BatchedEnv(run, n_workers=1) as env:
        obs, _ = env.reset()
        U = env.layout.n_rows
        acts = cp.zero_rows(U)[None].copy()
        obs, _, _, _, info = env.step(acts)
        banned = ("reward", "counter", "team_", "rows_ended", "won",
                  "eliminated", "clock")
        for k in obs:
            assert not any(b in k for b in banned), k
        for t in range(env.layout.n_teams):
            assert set(env.team_rows(obs, t)) == set(obs)


@pytest.mark.parametrize("edit,needle", [
    ('profile = "duel"\n', "profile must be one of"),
    ('profile = "fireteam"\nkills = 1.0\n', "'kills' is not a term"),
    ('profile = "fireteam"\nzombie_bars = 0.1\n', "'zombie_bars' is not a"),
    ('profile = "fireteam"\nclock = "-10"\n', "clock must be a number"),
    ('profile = 3\n', "profile must be one of"),
])
def test_the_run_file_door_refuses_by_name(tmp_path, edit, needle):
    """Property: an unknown profile, a counter name or an unknown term
    under a profile and a non-number are refused by name; a profile with
    ``[obs] spec = "crop"`` is refused; no ``profile`` key is today's
    ``"counters"``; an override lands in the terms. Breaks if the door
    accepts what the books cannot pay."""
    p = tmp_path / "r.toml"
    p.write_text(_run_text("x", tmp_path, reward=edit), encoding="utf-8")
    with pytest.raises(ValueError, match=needle):
        rl_env.load_run(p)
    p.write_text("[run]\nseed = 1\nn_envs = 1\ndecimation = 4\n"
                 "levels = [\"x\"]\npolicy = \"policy\"\n[obs]\n"
                 "channels = [\"smoke\"]\ncrop_radius = 2\nvector = []\n"
                 "[reward]\nprofile = \"fireteam\"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs \\[obs\\] spec 'v1'"):
        rl_env.load_run(p)
    p.write_text(_run_text("x", tmp_path, reward="kills = 1.0\n"),
                 encoding="utf-8")
    run = rl_env.load_run(p)
    assert run.reward_profile == "counters" and run.reward_terms == ()
    p.write_text(_run_text("x", tmp_path, reward='profile = "fireteam"\n'
                           'zombie_bar = 0.1\n'), encoding="utf-8")
    run = rl_env.load_run(p)
    assert run.terms()["zombie_bar"] == 0.1
    assert run.terms()["win"] == FIRETEAM["win"]


def test_the_full_view_balance_is_the_reward_standing(tmp_path):
    """Property (obs spec 5.4): the full view's ``balance[t]`` column is
    ``squash(standing().h(t) / n_t)`` -- the reward's one implementation of
    a team's health in bars -- and equals the old unclipped sum where no
    unit stands above its bar. Breaks if the full view keeps its own
    builder."""
    from simulation.obs_spec_v1 import squash
    lv = _scene_level(tmp_path, "zombie_box")
    r, run = _runner(tmp_path, "zbox", full_view=True)
    r.reset()
    lay = r.layout
    out = {k: np.zeros(s[1:], d) for k, (s, d) in lay.obs_shapes().items()}
    out.update({k: np.zeros(s[1:], d)
                for k, (s, d) in lay.full_shapes().items()})
    seen_hurt = False
    for k in range(400):
        r.observe(out)
        sim = r.session.sim
        bars = sim.team_picture.bars
        parts = R.standing(sim.units, bars)
        for t in range(2):
            n_t = lv.match.teams[t].n
            want = squash(parts.h(t) / n_t)
            old = sum(u.current_hp / bars[u.id] for u in sim.units
                      if u.alive and not u.is_zombie and u.team == t) / n_t
            row = t * lay.n_slots
            assert out["full_match"][row, 2 * t + 1] == np.float32(want)
            assert abs(want - squash(old)) < 1e-12
            seen_hurt |= 0 < parts.h(t) < n_t
        res = r.step(cp.zero_rows(lay.n_rows))
        if res["ended"]:
            break
    assert seen_hurt
