"""#195 1g (a), (b) -- observation spec v1: the soldier's vector, the list,
the row flags and the two windows (``simulation/obs_spec_v1.py``; the
contract ``docs/rl_agent_obs_spec_v1_2026-11-12.md`` v2, section 7's
properties).

Patch (a)'s gates: the reconstruction (vector, list) against the plainer
second implementation ``tests/_obs_reference.py``; honesty -- the direct
checks and the counterfactual family's cases (a)-(e) over ``vector`` /
``list`` / ``list_mask`` / ``list_uid`` / the row flags; the chokepoint
(``TeamKnowledge`` / ``ObsWorld``), the AST scan and ``team_rows``; a
record outlives an unseen death; the graph refusal; the row flags' truth
table; the squash; no raw number; the new functions; the trackers in scope;
replay unmoved; the farm. Patch (b)'s: the reconstruction of every plane
(and the vector's ``air`` / ``felt_temp``), the family (a)-(e) extended to
``near_u`` / ``far_u`` and the non-physics far planes, the far reduce, the
sight planes against ``has_los``, the lanes plane and the lane pair,
``felt_temperature`` against the heat row, the scan's new callees. The full
view is patch (c)'s.
"""
from __future__ import annotations

import ast
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import _obs_reference as R  # noqa: E402
import control_policy as cp  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import rl_env  # noqa: E402
from config import CFG  # noqa: E402
from control_bots import BotTeam  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation, vision  # noqa: E402
from simulation import charges as ch  # noqa: E402
from simulation import line_of_fire, obs_spec_v1 as O  # noqa: E402
from simulation import team_picture as tp  # noqa: E402
from simulation.damage import KINETIC, DamagePacket, apply_packet  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import MatchSpec, MatchTeam  # noqa: E402
from simulation.match_session import MatchSession  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.sensor_accessor import EntityFieldAccessor  # noqa: E402
from simulation.turning import PI_Q16  # noqa: E402
from simulation.turning import place as place_facing  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from test_bots_v1 import _level  # noqa: E402
from test_rl_env import _box  # noqa: E402

ONE = 1 << 16
IDLE = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                 controller="idle")
BOTS = replace(IDLE, controller="bots")
PAIR_SPEC = MatchSpec(teams=(IDLE, IDLE), seed=1, zombies=0,
                      time_limit_s=60.0)
BANNED = ("units", "roster", "get_unit", "projectiles", "planted_charges",
          "team_picture", "tick_events", "_units_by_id", "state_for")
OBS_KEYS = O.OBS_KEYS
#: The arrays the counterfactual family compares byte for byte (spec 7):
#: everything but the physics planes and the vector's air / felt_temp
#: columns (the bodies are stamped into the air, spec 3.2 PENDING ERIK).
FAMILY_KEYS = ("vector", "list", "list_mask", "list_uid", "near_u", "far_u",
               "far_f", "valid", "in_match")
AIR_COLS = list(range(58, 66)) + [96]
FAR_NON_PHYSICS = (0, 9)            # wall_share, route


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _sim(level, units, *, sight="cone", memory_ticks=40, seed=1):
    """A running sim on ``level`` holding ``units`` (team, x, y), the
    picture started for teams 0 and 1."""
    sim = Simulation(level, seed=seed, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    sim.sight = sight
    out = []
    for i, (team, x, y) in enumerate(units):
        u = Unit(f"u{i}", x=x, y=y, team=team, footprint=3)
        sim.add_unit(u)
        out.append(u)
    sim.start_team_picture((0, 1), memory_ticks=memory_ticks)
    return sim, out


def _rows(sim, spec, S):
    units = [None] * (len(spec.teams) * S)
    for t in range(len(spec.teams)):
        for u in sim.roster(t):
            if u.alive and u.team == t:
                units[t * S + u.slot] = u
    return units


def _flags(sim, spec, S):
    n = len(spec.teams) * S
    valid = np.zeros(n, np.uint8)
    in_match = np.zeros(n, np.uint8)
    for t in range(len(spec.teams)):
        for u in sim.roster(t):
            in_match[t * S + u.slot] = 1
            valid[t * S + u.slot] = u.alive and u.team == t
    return valid, in_match


def _observe(sim, ctx, spec, S):
    """The accessor's door over every row: the eight arrays plus the
    flags."""
    U = len(spec.teams) * S
    out = {k: np.zeros(shape[1:], dtype)
           for k, (shape, dtype) in O.shapes(1, U, S).items()
           if k in OBS_KEYS}
    out["list_uid"][...] = -1
    EntityFieldAccessor(sim.gmap).observe("v1", _rows(sim, spec, S), out,
                                          sim=sim, context=ctx)
    out["valid"], out["in_match"] = _flags(sim, spec, S)
    return out


def _team_bytes(out, team, S):
    """Team ``team``'s rows of every array the family compares, as bytes:
    the vector without its air / felt_temp columns, ``far_f``'s wall share
    and route planes only (spec 7: the physics planes are (b)'s imprint
    measurement's, not the family's)."""
    rows = slice(team * S, (team + 1) * S)
    got = []
    for k in FAMILY_KEYS:
        a = out[k][rows]
        if k == "vector":
            a = np.delete(a, AIR_COLS, axis=1)
        elif k == "far_f":
            a = a[:, list(FAR_NON_PHYSICS)]
        got.append(np.ascontiguousarray(a).tobytes())
    return tuple(got)


def _hook_step(monkeypatch, sim, fn):
    """Run ``fn(sim)`` INSIDE the next step, before slot 11 (the picture)."""
    rules = sim.ruleset
    real = type(rules).on_tick_end
    done = []

    def hooked(sim_):
        if not done:
            done.append(True)
            fn(sim_)
        return real(rules, sim_)
    monkeypatch.setattr(rules, "on_tick_end", hooked)


def _kill(victim, by_zombie=False, source="burning"):
    def fn(sim):
        apply_packet(victim, DamagePacket(10_000, KINETIC, -1),
                     sim.tick_events, source=source,
                     mark_killed_by_zombie=by_zombie)
    return fn


def _graph_box(root, name, spec, w=40, h=24):
    """``test_rl_env._box`` with ONE room over its floor as its room graph
    (``level_lib.write_room_graph``): a level a ``"v1"`` run accepts."""
    lvl = _box(root, name, spec, w, h)
    grid = np.full((h, w), -1, dtype=np.int32)
    grid[1:h - 1, 1:w - 1] = 0
    level_lib.write_room_graph(
        lvl.path, [dict(id=0, x=1, y=1, w=w - 2, h=h - 2, kind="room")],
        [], grid)
    return level_loader.load(name, str(root))


def _run_v1(path, *, levels, levels_dir, seed=11, n_envs=2, decimation=4,
            policy='"policy"'):
    lst = "[" + ", ".join(f'"{x}"' for x in levels) + "]"
    path.write_text(
        "[run]\n"
        f"seed = {seed}\nn_envs = {n_envs}\ndecimation = {decimation}\n"
        f"levels = {lst}\npolicy = {policy}\n"
        f"levels_dir = '{Path(levels_dir).as_posix()}'\n"
        "[obs]\nspec = \"v1\"\n[reward]\nkills = 1.0\n", encoding="utf-8")
    return path


def _zero_actions(env):
    rows = cp.zero_rows(env.layout.n_rows)
    return np.broadcast_to(rows, (env.n_envs,) + rows.shape).copy()


# ---------------------------------------------------------------------------
# the reconstruction and the direct checks (a ship_demo bots match)
# ---------------------------------------------------------------------------
def _ship_match():
    lv = level_loader.load("ship_demo", str(ROOT / "levels"))
    spec = replace(lv.match, teams=tuple(replace(t, controller="bots")
                                         for t in lv.match.teams))
    ses = MatchSession(lv, spec, seed=1, breach_physics=bp)
    S = max(rl_env._team_slots(lv))
    return ses, spec, S


def test_equals_an_independent_reconstruction():
    """Property (spec 7): over a ship_demo bots-vs-bots match observed
    every 4 ticks for 600 ticks, every vector column and every list entry,
    mask and uid equals ``tests/_obs_reference.py``'s -- a second, plainer
    implementation written from the document (loops, the sight oracle per
    line, the bots' own ``_teammate_on_line``, ``flood_reference``), never
    importing ``obs_spec_v1``. Breaks if a builder reads the wrong seam,
    denominator, order or frame. A CORRECTNESS gate; honesty is below."""
    ses, spec, S = _ship_match()
    sim = ses.sim
    ctx = O.ObsContext.build(sim, spec, S)
    K = R.constants(sim, spec, S)
    bots = {t: BotTeam(t) for t in range(len(spec.teams))}
    names_v = [f.name for f in O.VECTOR_COLUMNS]
    names_e = [f.name for f in O.ENTRY_COLUMNS]
    types = np.zeros(5, dtype=np.int64)
    rows_checked = 0
    for _ in range(600):
        ses.tick()
        if sim.tick % 4 or ses.is_terminal():
            continue
        out = _observe(sim, ctx, spec, S)
        for i, u in enumerate(_rows(sim, spec, S)):
            if u is None:
                assert not out["vector"][i].any() and not out["list_mask"][i].any()
                continue
            want = R.vector(sim, K, u)
            bad = [names_v[j] for j in range(O.V)
                   if not math.isclose(out["vector"][i, j], want[j],
                                       rel_tol=1e-6, abs_tol=1e-6)]
            assert not bad, (sim.tick, u.name, bad, out["vector"][i][
                [names_v.index(b) for b in bad]], want[[names_v.index(b)
                                                       for b in bad]])
            ents, mask, uid = R.entry_list(sim, K, u, bots[int(u.team)])
            assert np.array_equal(out["list_mask"][i], mask), (sim.tick, u.name)
            assert np.array_equal(out["list_uid"][i], uid), (sim.tick, u.name)
            for k in np.flatnonzero(mask):
                bad = [names_e[j] for j in range(O.F)
                       if not math.isclose(out["list"][i, k, j], ents[k, j],
                                           rel_tol=1e-6, abs_tol=1e-6)]
                assert not bad, (sim.tick, u.name, k, bad,
                                 out["list"][i, k][[names_e.index(b)
                                                    for b in bad]],
                                 ents[k][[names_e.index(b) for b in bad]])
                types[int(ents[k, 0])] += 1
            rows_checked += 1
        if ses.is_terminal():
            break
    assert rows_checked > 100
    assert types[0] > 0 and types[1] > 0 and types[2] > 0, types


def test_shows_nothing_the_team_cannot_know_the_direct_checks():
    """Property (spec 7): through the same match, every ``list_uid`` of
    types 0-2 is a living team-mate's id or a key of the team's records;
    every type-3 uid is the team's object tracker's id for an object in
    ``seen_now`` whose tile is ``rays_clear(physical=False)`` from a living
    team-mate's centre; every type-4 uid a living door's ordinal plus the
    base. Breaks if an entry leaks a unit or object the team has no record
    or sight of."""
    ses, spec, S = _ship_match()
    sim = ses.sim
    ctx = O.ObsContext.build(sim, spec, S)
    checked = 0
    for _ in range(600):
        ses.tick()
        if sim.tick % 4 or ses.is_terminal():
            continue
        out = _observe(sim, ctx, spec, S)
        for i, u in enumerate(_rows(sim, spec, S)):
            if u is None:
                continue
            team = int(u.team)
            recs = sim.team_picture.records(team)
            mates = {m.id for m in sim.units
                     if m.alive and not m.is_zombie and m.team == team}
            ids = sim.team_picture.object_ids(team)
            seen_now = {ouid: o for o, ouid in sim.team_picture.seen_now[team]}
            for k in np.flatnonzero(out["list_mask"][i]):
                t, uid = int(out["list"][i, k, 0]), int(out["list_uid"][i, k])
                if t == 0:
                    assert uid in mates and uid != u.id
                elif t in (1, 2):
                    assert uid in recs
                    assert (t == 1) == (uid in vision.state_for(sim)
                                        .visible_by_team.get(team, ()))
                elif t == 3:
                    assert uid in seen_now and ids[seen_now[uid]] == uid
                    oy, ox = tp.object_tile(seen_now[uid])
                    assert any(vision.rays_clear(
                        sim, [(m.center_tile_y(), m.center_tile_x(), oy, ox)],
                        physical=False)[0]
                        for m in sim.units if m.id in mates)
                else:
                    assert t == 4
                    assert uid - O.DOOR_UID_BASE in {
                        d.ordinal for d in sim._doors if d.alive}
                checked += 1
        if ses.is_terminal():
            break
    assert checked > 0


# ---------------------------------------------------------------------------
# the counterfactual family (a)-(e): two sims, one hidden difference
# ---------------------------------------------------------------------------
def _pair(tmp_path, name, units, *, wall_x=None, gap=None, sight="cone",
          memory_ticks=40):
    lvl = _level(tmp_path, name, wall_x=wall_x, gap=gap)
    a, ua = _sim(lvl, units, sight=sight, memory_ticks=memory_ticks)
    b, ub = _sim(lvl, units, sight=sight, memory_ticks=memory_ticks)
    S = 2
    ctx = (O.ObsContext.build(a, PAIR_SPEC, S), O.ObsContext.build(b, PAIR_SPEC, S))
    return (a, b), (ua, ub), ctx, S


def _step_both(sims, n=1):
    for _ in range(n):
        for s in sims:
            s.step()


def _compare_team0(sims, ctxs, S, ticks, every=4):
    """Step both sims ``ticks`` ticks; at every ``every``-th tick team 0's
    rows of every (a)-array must be byte-equal."""
    n = 0
    for _ in range(ticks):
        _step_both(sims)
        if sims[0].tick % every:
            continue
        a = _team_bytes(_observe(sims[0], ctxs[0], PAIR_SPEC, S), 0, S)
        b = _team_bytes(_observe(sims[1], ctxs[1], PAIR_SPEC, S), 0, S)
        for k, (x, y) in zip(FAMILY_KEYS, zip(a, b)):
            assert x == y, (k, sims[0].tick)
        n += 1
    assert n > 0


def _seen_then_unseen(tmp_path, name, memory_ticks=40):
    """The cone scene: me (team 0) facing east sees the foe (team 1) once,
    turns north and loses it; returns the pair with the foe remembered."""
    sims, (ua, ub), ctxs, S = _pair(tmp_path, name,
                                    [(0, 10, 20), (1, 30, 20)],
                                    memory_ticks=memory_ticks)
    for me in (ua[0], ub[0]):
        place_facing(me, 0)
    _step_both(sims)
    for s, u in zip(sims, (ua, ub)):
        assert u[1].id in s.team_picture.records(0)
        s.set_aim(u[0].id, 0, -ONE)             # turn north, away
    _step_both(sims, 16)
    for s, u in zip(sims, (ua, ub)):
        vis = vision.state_for(s)
        assert u[1].id not in vis.visible_by_team.get(0, ())
        assert s.team_picture.records(0)[u[1].id].tick < s.tick
    return sims, (ua, ub), ctxs, S


def test_family_a_an_unseen_death_changes_no_row(tmp_path, monkeypatch):
    """Property (spec 7, case a): a remembered enemy killed unseen by a
    status (a packet through the door inside the step) vs. left alive:
    team 0's rows of vector, list, list_mask, list_uid, valid and in_match
    are byte-equal at every decision for the memory time. Breaks if any
    builder reads liveness, a count or a tracker of a unit the team cannot
    see."""
    sims, (ua, ub), ctxs, S = _seen_then_unseen(tmp_path, "fam_a")
    _hook_step(monkeypatch, sims[1], _kill(ub[1]))
    _compare_team0(sims, ctxs, S, 40)
    assert not ub[1].alive and ua[1].alive


def test_family_b_an_unseen_move_changes_no_row(tmp_path):
    """Property (spec 7, case b): a remembered enemy walking out of sight
    (a move intent through the door) vs. standing: team 0's rows are
    byte-equal. Breaks if a builder reads a live position of an unseen
    unit."""
    sims, (ua, ub), ctxs, S = _seen_then_unseen(tmp_path, "fam_b")
    sims[1].set_move_dir(ub[1].id, -ONE, 0, 0)
    _compare_team0(sims, ctxs, S, 40)
    assert ub[1].x < ua[1].x


def test_family_c_an_unseen_conversion_changes_no_row(tmp_path, monkeypatch):
    """Property (spec 7, case c): a remembered enemy converted out of sight
    vs. not: team 0's rows are byte-equal. Breaks if class, side or a
    count is read live."""
    sims, (ua, ub), ctxs, S = _seen_then_unseen(tmp_path, "fam_c")
    _hook_step(monkeypatch, sims[1], _kill(ub[1], by_zombie=True))
    _compare_team0(sims, ctxs, S, 40)
    assert ub[1].is_zombie and not ua[1].is_zombie


def test_family_d_an_unseen_grenade_moves_no_object_id(tmp_path):
    """Property (spec 7, case d): an extra grenade thrown by an unseen unit
    BEFORE a grenade the team sees (first in ``sim.projectiles``) changes
    nothing in team 0's rows -- the seen grenade's ``list_uid`` included.
    Breaks if an object id is a list or sim index, or the tracker counts
    unseen objects."""
    sims, (ua, ub), ctxs, S = _pair(
        tmp_path, "fam_d", [(0, 10, 20), (1, 45, 20), (1, 20, 8)],
        wall_x=30, sight="all_round")
    _step_both(sims)
    for s, u in zip(sims, (ua, ub)):
        assert u[2].id in s.team_picture.records(0)
        assert u[1].id not in s.team_picture.records(0)
        s.throw_grenade_intent(u[2].id, ONE, 0, 24, 3 << 16, 1)   # seen
    sims[1].throw_grenade_intent(ub[1].id, ONE, 0, 24, 3 << 16, 1)  # unseen
    _compare_team0(sims, ctxs, S, 40)
    out = _observe(sims[1], ctxs[1], PAIR_SPEC, S)
    assert (out["list_uid"][0] == O.OBJECT_UID_BASE).sum() <= 1


def test_family_e_an_unseen_owners_conversion_changes_no_object_row(
        tmp_path, monkeypatch):
    """Property (spec 7, case e): the owner of a charge the team sees is
    converted out of sight vs. not: the charge's ``side`` (-1) and
    ``whose`` (by the roster) and every other row stay byte-equal. Breaks
    if an object's side or owner is read off the live owner."""
    sims, (ua, ub), ctxs, S = _pair(tmp_path, "fam_e",
                                    [(0, 10, 20), (1, 45, 20)],
                                    wall_x=30, sight="all_round")
    for s, u in zip(sims, (ua, ub)):
        assert ch.place(s, ch.PlantedCharge(20, 20, "demo_c4", u[1].id,
                                            det_tick=None))
    _step_both(sims)
    out = _observe(sims[0], ctxs[0], PAIR_SPEC, S)
    k = int(np.flatnonzero(out["list"][0, :, 0] == O.TYPE_OBJECT)[0])
    assert out["list_mask"][0, k] and out["list"][0, k, 8] == -1
    assert out["list"][0, k, 56] == 2                 # another owner
    _hook_step(monkeypatch, sims[1], _kill(ub[1], by_zombie=True))
    _compare_team0(sims, ctxs, S, 40)
    assert ub[1].is_zombie


# ---------------------------------------------------------------------------
# a record outlives its unit's unseen death
# ---------------------------------------------------------------------------
def test_a_record_outlives_its_units_unseen_death(tmp_path, monkeypatch):
    """Property (spec 7): a seen enemy walks out of sight and is killed by
    a status unseen: its type-2 entry stays for ``memory_ticks`` with
    ``age`` rising and ``dx``, ``class``, ``vel``, ``cos``, ``sin`` frozen
    at the sighting; a SEEN death drops the entry on the next decision.
    Breaks if the list reads liveness or the live unit."""
    sims, (ua, ub), ctxs, S = _seen_then_unseen(tmp_path, "outlive")
    sim, (me, foe), ctx = sims[1], ub, ctxs[1]
    rec = sim.team_picture.records(0)[foe.id]
    _hook_step(monkeypatch, sim, _kill(foe))
    sim.step()
    assert not foe.alive
    ages = []
    while sim.tick - rec.tick <= 40:
        out = _observe(sim, ctx, PAIR_SPEC, S)
        k = np.flatnonzero((out["list"][0, :, 0] == O.TYPE_RECORD)
                           & (out["list_uid"][0] == foe.id))
        assert len(k) == 1, sim.tick
        e = out["list"][0, int(k[0])]
        assert e[1] == O.squash(O.ratio(rec.centre[0] - me.center_tile_x(),
                                        ctx.l_ref))
        assert e[9] == float(O.ratio(1, 1) * R.row("loadouts", rec.loadout))
        assert e[6] == O.squash(O.ratio(rec.vel_q[0], ctx.v_ref_q))
        assert e[4] == R.cos_q16(rec.facing_q) / ONE
        ages.append(float(e[54]))
        sim.step()
    assert ages == sorted(ages) and ages[-1] > ages[0]
    out = _observe(sim, ctx, PAIR_SPEC, S)
    assert foe.id not in out["list_uid"][0]
    # a SEEN death: gone at the next decision
    lvl = _level(tmp_path, "seen_death")
    sim2, (me2, foe2) = _sim(lvl, [(0, 10, 20), (1, 30, 20)],
                             sight="all_round")
    ctx2 = O.ObsContext.build(sim2, PAIR_SPEC, S)
    sim2.step()
    assert foe2.id in _observe(sim2, ctx2, PAIR_SPEC, S)["list_uid"][0]
    _hook_step(monkeypatch, sim2, _kill(foe2, source="bullet"))
    sim2.step()
    assert foe2.id not in _observe(sim2, ctx2, PAIR_SPEC, S)["list_uid"][0]


# ---------------------------------------------------------------------------
# the doors: the graph refusal, the row flags, the squash, no raw number
# ---------------------------------------------------------------------------
def test_a_level_without_a_room_graph_is_refused(tmp_path):
    """Property (spec 7): ``load_levels`` on a ``"v1"`` run naming
    ``playground`` (no rooms.npy) raises naming the level and the spec; the
    same run under ``"crop"`` loads. Breaks if the refusal is missing or on
    the wrong path."""
    p = _run_v1(tmp_path / "run.toml", levels=["playground"],
                levels_dir=ROOT / "levels")
    run = rl_env.load_run(p)
    with pytest.raises(ValueError, match="playground.*room graph.*v1"):
        rl_env.load_levels(run)
    from test_rl_env import _run_toml
    crop = rl_env.load_run(_run_toml(tmp_path / "crop.toml",
                                     levels=["playground"],
                                     levels_dir=ROOT / "levels"))
    assert rl_env.load_levels(crop)[0].name.lower() == "playground"


def test_the_run_file_door_refuses_the_crop_keys_under_v1(tmp_path):
    """Property: under ``spec = "v1"`` a crop key (``channels``,
    ``crop_radius``, ``vector``) is refused by name; an unknown spec is
    refused naming the known ones. Breaks if a stale crop key silently
    rides along."""
    p = _run_v1(tmp_path / "run.toml", levels=["x"], levels_dir=tmp_path)
    s = p.read_text(encoding="utf-8")
    p.write_text(s.replace('spec = "v1"\n', 'spec = "v1"\ncrop_radius = 4\n'),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="crop_radius"):
        rl_env.load_run(p)
    p.write_text(s.replace('"v1"', '"v9"'), encoding="utf-8")
    with pytest.raises(ValueError, match="v9"):
        rl_env.load_run(p)


def test_the_two_row_flags_truth_table(tmp_path):
    """Property (spec 4.2, 7): on the horde box (idle marines; the horde
    kills and converts team 0) a living soldier reads ``in_match`` 1,
    ``valid`` 1; one killed or converted 1, 0 with a zero vector and an
    all-zero mask; a slot beyond the team's ``n`` 0, 0 -- in ``obs`` and
    in ``final_obs``. Breaks if a dead row reads as an empty place, or is
    observed."""
    from test_rl_env import BOX_SPEC
    _graph_box(tmp_path, "hbox", BOX_SPEC)
    run = rl_env.load_run(_run_v1(tmp_path / "run.toml", levels=["hbox"],
                                  levels_dir=tmp_path, n_envs=1))
    saw = {"dead": False, "living": False}
    with rl_env.BatchedEnv(run, n_workers=1) as env:
        S = env.layout.n_slots
        obs, _ = env.reset()
        for _ in range(400):
            obs, _, _, _, info = env.step(_zero_actions(env))
            final = info["final_obs"] if info["ended"][0] else None
            for o in (obs, final):
                if o is None:
                    continue
                for t in range(2):
                    for s in range(S):
                        r = t * S + s
                        im, va = int(o["in_match"][0, r]), int(o["valid"][0, r])
                        if s >= 2:
                            assert (im, va) == (0, 0)
                        else:
                            assert im == 1
                        if va:
                            saw["living"] = True
                        else:
                            saw["dead"] |= im == 1
                            assert not o["vector"][0, r].any()
                            assert not o["list_mask"][0, r].any()
                            assert (o["list_uid"][0, r] == -1).all()
                            for k in ("near_f", "near_u", "far_f", "far_u"):
                                assert not o[k][0, r].any(), k
                for k in OBS_KEYS + ("valid", "in_match"):
                    assert o[k].dtype == env.layout.shapes()[k][1]
            if info["ended"][0]:
                assert final["in_match"][0, :2].all()
                assert not final["valid"][0, :2].any()      # team 0 gone
                assert final["valid"][0, S:S + 2].all()      # team 1 lives
                break
        else:
            raise AssertionError("the box match did not end")
    assert saw["dead"] and saw["living"]


def test_the_squash():
    """Property (spec 0.2, 7): the identity on [-1, 1], monotone,
    continuous at +-1, capped at +-4, ``squash(e) == 2``. Breaks if the
    curve changes."""
    xs = np.linspace(-60, 60, 20001)
    ys = [O.squash(x) for x in xs]
    assert all(y == x for x, y in zip(xs, ys) if abs(x) <= 1)
    assert all(b >= a for a, b in zip(ys, ys[1:]))
    assert abs(O.squash(1 + 1e-9) - 1) < 1e-6 and abs(O.squash(-1 - 1e-9) + 1) < 1e-6
    assert O.squash(math.e) == pytest.approx(2.0)
    assert O.squash(1e9) == 4.0 and O.squash(-1e9) == -4.0
    assert max(abs(y) for y in ys) == 4.0


def test_no_raw_number():
    """Property (spec 7): every vector and entry column is declared with a
    denominator of the closed vocabulary, or as a flag, a kind, an angle
    or a unit-interval value; the positions cover 0..V-1 and 0..F-1 once
    each. Breaks if a builder stores a game number under no table
    value."""
    kinds = {O.FLAG, O.KIND, O.ANGLE, O.UNIT}
    for cols, n in ((O.VECTOR_COLUMNS, O.V), (O.ENTRY_COLUMNS, O.F)):
        assert [f.pos for f in cols] == list(range(n))
        assert len({f.name for f in cols}) == n
        for f in cols:
            assert f.denom and (f.denom in kinds or f.denom in O.DENOMINATORS), f
            if f.denom in kinds:
                assert not f.squash, f
    assert O.V == 97 and O.F == 67
    # (b): every plane of both windows is declared too (a plane may be a
    # COUNT, the far unit planes; a vector or entry column never is)
    planes = set(O.NEAR_F_PLANES) | set(O.NEAR_U_PLANES) | set(O.FAR_F_PLANES) \
        | set(O.FAR_U_PLANES)
    assert planes == set(O.PLANE_COLUMNS)
    for name, (denom, sq) in O.PLANE_COLUMNS.items():
        assert denom in kinds | {O.COUNT} or denom in O.DENOMINATORS, name
        if denom in kinds | {O.COUNT}:
            assert not sq, name
    assert not any(f.denom == O.COUNT for f in O.VECTOR_COLUMNS + O.ENTRY_COLUMNS)
    # the vector's air block reads the planes' denominators
    for p, name in enumerate(O.PHYSICS_PLANES):
        assert O.VECTOR_COLUMNS[58 + p].denom == O.PLANE_COLUMNS[name][0]


# ---------------------------------------------------------------------------
# the new functions
# ---------------------------------------------------------------------------
def test_mate_in_way_is_the_bots_line_geometry():
    """Property (spec 1.7, 7): ``line_of_fire.mate_in_way`` equals
    ``BotTeam._teammate_on_line`` on 1 000 random triples of shooter,
    target and three mates. Breaks if the module function drifts from the
    bots' geometry."""
    bots = BotTeam(0)
    rng = np.random.default_rng(7)
    for _ in range(1000):
        pts = rng.integers(0, 30, size=(5, 2))
        us = [Unit(f"u{i}", x=int(p[0]), y=int(p[1]), team=0, footprint=3)
              for i, p in enumerate(pts)]
        u, target, mine = us[0], us[1], [us[0]] + us[2:]
        want = bots._teammate_on_line(u, target, mine)
        got = line_of_fire.mate_in_way(
            R.centre(u), R.centre(target), [R.centre(m) for m in us[2:]],
            bots.s.line_clear_tiles, 3)
        assert got == want


def test_solid_in_way_is_the_physical_walk_and_the_live_cover(tmp_path):
    """Property (spec 1.7): ``solid_in_way`` is 1 across a wall and 0 in
    the open, judged by the physical walk (a round's path), not sight.
    Breaks if the flag reads the sight screen."""
    lvl = _level(tmp_path, "lof", wall_x=30)
    sim, (a, b, c) = _sim(lvl, [(0, 10, 20), (1, 40, 20), (0, 20, 20)],
                          sight="all_round")
    world = O.ObsWorld(sim)
    assert line_of_fire.solid_in_way(world, R.centre(a), R.centre(b))
    assert not line_of_fire.solid_in_way(world, R.centre(a), R.centre(c))


def test_aim_of_on_a_hand_built_scene(tmp_path):
    """Property (spec 1.7, 7): ``aim_of`` answers me / a mate / another
    side / nobody with the right uid; two candidates at equal distance
    answer the lower uid; an unknown unit (no record) is never answered;
    a wall between answers nobody. Breaks if the rule reads a live enemy,
    the wrong ray (a mate's facing instead of its target) or no sight."""
    lvl = _level(tmp_path, "aim", wall_x=40)
    sim, (me, mate, foe, hidden) = _sim(
        lvl, [(0, 10, 20), (0, 20, 20), (1, 30, 20), (1, 50, 20)],
        sight="all_round")
    ctx = O.ObsContext.build(sim, PAIR_SPEC, 2)
    place_facing(me, 0)                     # east: the mate is nearest
    place_facing(mate, PI_Q16)              # west: me
    place_facing(foe, PI_Q16)               # west: the mate
    sim.step()
    know = O.TeamKnowledge.build(sim, sim.team_picture, 0)
    assert O.aim_of(know, ctx, me, me.id) == (2, mate.id)
    assert O.aim_of(know, ctx, mate, me.id) == (1, me.id)
    assert O.aim_of(know, ctx, foe, me.id) == (2, mate.id)
    # the mate's ray is its TARGET: aim east and it names the foe at once
    sim.set_aim(mate.id, ONE, 0)
    sim.step()
    know = O.TeamKnowledge.build(sim, sim.team_picture, 0)
    assert O.aim_of(know, ctx, mate, me.id) == (3, foe.id)
    # nobody: me facing north
    place_facing(me, PI_Q16 // 2)
    assert O.aim_of(know, ctx, me, me.id) == (0, -1)
    # an unknown unit is never answered: the hidden foe faces the mate's
    # row through the wall -- but from a mate's side, a unit with no
    # record in my cone stands unanswered
    sim2, (me2, foe2) = _sim(_level(tmp_path, "aim2"), [(0, 10, 20), (1, 30, 20)],
                             sight="cone")
    ctx2 = O.ObsContext.build(sim2, PAIR_SPEC, 2)
    place_facing(me2, 0)
    know2 = O.TeamKnowledge.build(sim2, sim2.team_picture, 0)
    assert foe2.id not in know2.records          # faced north at the start
    assert O.aim_of(know2, ctx2, me2, me2.id) == (0, -1)
    # the tie: two records at one centre answer the lower uid
    sim3, (me3, f1, f2) = _sim(_level(tmp_path, "aim3"),
                               [(0, 10, 20), (1, 30, 20), (1, 30, 20)],
                               sight="all_round")
    ctx3 = O.ObsContext.build(sim3, PAIR_SPEC, 2)
    place_facing(me3, 0)
    sim3.step()
    know3 = O.TeamKnowledge.build(sim3, sim3.team_picture, 0)
    assert O.aim_of(know3, ctx3, me3, me3.id) == (3, min(f1.id, f2.id))
    # a wall between: nobody (a record written by hand behind it)
    sim4, (me4, foe4) = _sim(_level(tmp_path, "aim4", wall_x=20),
                             [(0, 10, 20), (1, 30, 20)], sight="all_round")
    ctx4 = O.ObsContext.build(sim4, PAIR_SPEC, 2)
    place_facing(me4, 0)
    sim4.team_picture._write(sim4, 0, foe4)
    know4 = O.TeamKnowledge.build(sim4, sim4.team_picture, 0)
    assert foe4.id in know4.records
    assert O.aim_of(know4, ctx4, me4, me4.id) == (0, -1)


def test_routes_field_is_zero_at_the_place_and_its_first_step_descends():
    """Property (spec 1.7, 7): ``route``'s field reads 0 on every anchor
    whose block centre lies in the place and the first step from a unit
    in the other room is strictly downhill. Breaks if the seeds, the
    walkable set or the descent rule move."""
    from test_orders import _sim as orders_sim, _marine
    sim = orders_sim(graph=True)
    sim.add_unit(Unit("m", x=5, y=5, team=0))
    world = O.ObsWorld(sim)
    graph = world.graph
    field = O.route(world, graph, 1, 3)
    from simulation import horde, room_graph
    passable = horde.block_passable(horde.walkable_tiles(sim.gmap), 3)
    h, w = field.shape
    zeros = 0
    for y in range(h):
        for x in range(w):
            if passable[y, x] and room_graph.place_at(graph, y + 1, x + 1) == 1:
                assert field[y, x] == 0
                zeros += 1
    assert zeros > 0
    u = sim.units[0]
    c = horde.candidates(field, u.tile_y, u.tile_x)
    assert c and c[0][0] < field[u.tile_y, u.tile_x] < horde.UNREACHED


def test_the_trackers_answer_only_in_scope(tmp_path):
    """Property (spec 0.5, 7): the view's trackers refuse ``motion`` of a
    non-mate and ``bar`` / ``damage_last_s`` of a uid the team does not see
    now. Breaks if the view hands out memory of a unit outside its scope."""
    lvl = _level(tmp_path, "scope", wall_x=30)
    sim, (me, foe) = _sim(lvl, [(0, 10, 20), (1, 40, 20)], sight="all_round")
    sim.team_picture._write(sim, 0, foe)        # remembered, unseen
    know = O.TeamKnowledge.build(sim, sim.team_picture, 0)
    assert foe.id in know.records and foe.id not in know.seen
    assert know.trackers.motion(me.id) == (0, 0)
    with pytest.raises(KeyError, match="not a living mate"):
        know.trackers.motion(foe.id)
    with pytest.raises(KeyError, match="neither a mate nor seen"):
        know.trackers.bar(foe.id)
    with pytest.raises(KeyError, match="neither a mate nor seen"):
        know.trackers.damage_last_s(foe.id)
    with pytest.raises(KeyError, match="not a living mate"):
        know.seen_by(foe.id, me.id)


# ---------------------------------------------------------------------------
# the policy never sees it: the chokepoint, the scan, team_rows
# ---------------------------------------------------------------------------
def test_the_chokepoint_raises_on_every_banned_name(tmp_path):
    """Property (spec 5.7): ``know.world`` asked for each banned sim
    member raises ``AttributeError`` (the wall is enforced at run time,
    not only grepped). Breaks if the facade grows a fallback or one of
    those members."""
    lvl = _level(tmp_path, "wall")
    sim, _ = _sim(lvl, [(0, 10, 20), (1, 30, 20)])
    know = O.TeamKnowledge.build(sim, sim.team_picture, 0)
    for name in BANNED + ("_vision_cache",):
        with pytest.raises(AttributeError):
            getattr(know.world, name)
    assert not hasattr(type(know.world), "__getattr__")


def _function(tree, name):
    """The FunctionDef ``name`` (``Class.method`` allowed) of a module."""
    cls, _, meth = name.rpartition(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == meth:
                    return sub
        if not cls and isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function {name}")


def _banned_uses(node, exclude=()):
    skip = set()
    for ex in exclude:
        for sub in ast.walk(ex):
            skip.add(id(sub))
    hits = []
    for sub in ast.walk(node):
        if id(sub) in skip:
            continue
        if isinstance(sub, ast.Attribute) and sub.attr in BANNED:
            hits.append((sub.lineno, sub.attr))
        elif isinstance(sub, ast.Name) and sub.id in BANNED:
            hits.append((sub.lineno, sub.id))
    return hits


def test_the_scan_finds_no_banned_name_in_the_builders():
    """Property (spec 5.7): an AST scan of ``obs_spec_v1.py`` (bar
    ``TeamKnowledge.build`` and ``observe_rows``, the two doors that take
    the sim), ``line_of_fire.py``, ``turning.ticks_to_within``,
    ``verb_gates.door_in_reach`` / ``_reach_key``, ``route``'s callees in
    ``horde.py`` and the vision routines finds none of the banned names.
    A new callee joins the list in the patch that adds it. Breaks if a
    builder or a callee reaches the sim's units, roster, objects, picture
    or events."""
    src = ROOT / "src" / "simulation"

    def tree(name):
        return ast.parse((src / name).read_text(encoding="utf-8"))

    t = tree("obs_spec_v1.py")
    doors = [_function(t, "TeamKnowledge.build"), _function(t, "observe_rows")]
    assert not _banned_uses(t, exclude=doors)
    assert not _banned_uses(tree("line_of_fire.py"))
    assert not _banned_uses(_function(tree("turning.py"), "ticks_to_within"))
    vg = tree("verb_gates.py")
    for fn in ("door_in_reach", "_reach_key"):
        assert not _banned_uses(_function(vg, fn))
    hd = tree("horde.py")
    for fn in ("flood", "flood_reference", "walkable_tiles", "block_passable",
               "candidates", "field_at"):
        assert not _banned_uses(_function(hd, fn))
    vs = tree("vision.py")
    for fn in ("ray_clear", "rays_clear", "window_clear", "lines_clear",
               "_apply_cover", "_blocking_cover"):
        assert not _banned_uses(_function(vs, fn))
    # patch (b)'s callees: the wind plane, the felt temperature, the lanes
    assert not _banned_uses(_function(tree("gamemap.py"), "GameMap.wind_load_plane"))
    ex = tree("exchange.py")
    for fn in ("felt_temperature", "heat_dials"):
        assert not _banned_uses(_function(ex, fn))
    tpm = tree("team_picture.py")
    for fn in ("fire_lanes_of", "draw_lanes"):
        assert not _banned_uses(_function(tpm, fn))
    assert _banned_uses(doors[0])            # the door itself does read it


def test_obs_holds_the_spec_keys_and_team_rows_are_views(tmp_path):
    """Property (spec 4.1, 5.7, 7): ``obs`` holds exactly the v1 keys (no
    ``full_*`` key, nothing of info); ``BatchedEnv.team_rows(obs, t)``
    returns every key as the VIEW of team ``t``'s rows, equal to the slice
    and sharing its memory; a team outside the run is refused. Breaks if a
    full-view array or another team's row reaches the policy, or the rows
    are copied."""
    from test_rl_env import BOX_SPEC
    _graph_box(tmp_path, "tbox", BOX_SPEC)
    run = rl_env.load_run(_run_v1(tmp_path / "run.toml", levels=["tbox"],
                                  levels_dir=tmp_path, n_envs=2))
    with rl_env.BatchedEnv(run, n_workers=1) as env:
        obs, _ = env.reset()
        obs, _, _, _, info = env.step(_zero_actions(env))
        assert set(obs) == set(OBS_KEYS) | {"valid", "in_match"}
        assert set(OBS_KEYS) == {"vector", "list", "list_mask", "list_uid",
                                 "near_f", "near_u", "far_f", "far_u"}
        assert not any(k.startswith("full") for k in list(obs) + list(info))
        S, T = env.layout.n_slots, env.layout.n_teams
        for t in range(T):
            rows = env.team_rows(obs, t)
            assert set(rows) == set(obs)
            for k, v in rows.items():
                assert v.shape[1] == S and np.shares_memory(v, obs[k])
                assert np.array_equal(v, obs[k][:, t * S:(t + 1) * S])
        with pytest.raises(ValueError, match="not a row block"):
            env.team_rows(obs, T)
        assert obs["in_match"][:, :2].all() and not obs["in_match"][:, 2:S].any()


# ---------------------------------------------------------------------------
# replay unmoved, the farm
# ---------------------------------------------------------------------------
def test_replay_is_unmoved_by_the_observation(tmp_path):
    """Property (spec 7): a recorded ``policy`` match (zero rows) observed
    under ``"v1"`` every step replays through ``run_match(replay=)`` with
    every checkpoint equal -- the observation (its trackers included)
    writes no sim state. Breaks if a builder or a tracker writes the
    sim."""
    spec = MatchSpec(teams=(IDLE, IDLE), seed=1, zombies=3, time_limit_s=6.0)
    lv = _graph_box(tmp_path, "rbox", spec)
    run = rl_env.load_run(_run_v1(tmp_path / "run.toml", levels=["rbox"],
                                  levels_dir=tmp_path, n_envs=1,
                                  decimation=1))
    levels = rl_env.load_levels(run)
    layout = rl_env.make_layout(run, levels)
    r = rl_env.EnvRunner(0, run, layout, levels, breach_physics=bp,
                         record=True, checkpoint_every_ticks=12)
    r.reset()
    U = layout.n_rows
    out = {k: np.zeros(shape[1:], dtype) for k, (shape, dtype)
           in layout.obs_shapes().items()}
    zero = cp.zero_rows(U)
    for _ in range(2000):
        step = r.step(zero)
        r.observe(out)                        # every step, every row
        if step["ended"]:
            break
    assert step["ended"]
    rec = r.last_recording
    assert len(rec.checkpoints) > 3
    o = run_match(lv, replay=rec)             # verifies every checkpoint
    assert o.result.ticks == max(rec.checkpoints)


def test_the_farm_serves_v1_and_fills_the_final_arrays(tmp_path, capsys):
    """Property (spec 7, the farm): ``tools/rl_smoke.py`` on a ``"v1"``
    run (2 envs, 2 workers) runs without an exception; ``BatchedEnv``
    returns every array of 4.1 at its shape and dtype, env-owned (the same
    object every step), and ``final_*`` filled for an env that ended.
    Breaks if a shape, a dtype or the env-owned copy is wrong."""
    import rl_smoke
    spec = MatchSpec(teams=(IDLE, IDLE), seed=1, zombies=0, time_limit_s=1.0)
    _graph_box(tmp_path, "fbox", spec)
    p = _run_v1(tmp_path / "run.toml", levels=["fbox"], levels_dir=tmp_path,
                n_envs=2, decimation=4)
    assert rl_smoke.main([str(p), "--workers", "2", "--episodes", "1",
                          "--warmup-steps", "2"]) == 0
    assert "env-ticks/s:" in capsys.readouterr().out
    run = rl_env.load_run(p)
    with rl_env.BatchedEnv(run, n_workers=2) as env:
        N, U, S = env.n_envs, env.layout.n_rows, env.layout.n_slots
        E = S + 33
        want = {"vector": ((N, U, 97), np.float32),
                "list": ((N, U, E, 67), np.float32),
                "list_mask": ((N, U, E), np.uint8),
                "list_uid": ((N, U, E), np.int32),
                "near_f": ((N, U, 8, 25, 25), np.float16),
                "near_u": ((N, U, 5, 25, 25), np.uint8),
                "far_f": ((N, U, 10, 25, 25), np.float16),
                "far_u": ((N, U, 5, 25, 25), np.uint8),
                "valid": ((N, U), np.uint8), "in_match": ((N, U), np.uint8)}
        obs, _ = env.reset()
        first = {k: id(v) for k, v in obs.items()}
        ended = False
        for _ in range(10):
            obs, rew, term, trunc, info = env.step(_zero_actions(env))
            assert {k: id(v) for k, v in obs.items()} == first
            for k, (shape, dtype) in want.items():
                assert obs[k].shape == shape and obs[k].dtype == dtype, k
                assert info["final_obs"][k].shape == shape
                assert info["final_obs"][k].dtype == dtype, k
            if info["ended"].any():
                ended = True
                e = int(np.flatnonzero(info["ended"])[0])
                assert info["final_obs"]["in_match"][e].any()
                assert info["final_obs"]["valid"][e].any()
                assert info["final_obs"]["vector"][e].any()
                assert info["final_obs"]["list_mask"][e].any()
                assert info["final_obs"]["near_u"][e].any()
                assert np.abs(obs["vector"]).max() <= 4.0
                assert np.abs(obs["list"]).max() <= max(4.0, 255.0)
                for k in ("near_f", "far_f"):
                    assert np.abs(obs[k].astype(np.float32)).max() <= 4.0, k
        assert ended


# ---------------------------------------------------------------------------
# patch (b): the two windows
# ---------------------------------------------------------------------------
def _planes_close(got, want, name, where):
    """A float16 plane against its float64 reference: within float16's
    own precision on [-4, 4]."""
    g = np.asarray(got, dtype=np.float64)
    bad = np.abs(g - want) > 2e-3 + 2e-3 * np.abs(want)
    assert not bad.any(), (name, where, np.argwhere(bad)[:5],
                           g[bad][:5], want[bad][:5])


def _windows_of(sim, ctx, u, know=None):
    """One soldier's four window arrays through ``build_windows``."""
    know = O.TeamKnowledge.build(sim, sim.team_picture, int(u.team)) \
        if know is None else know
    nf = np.zeros((8, 25, 25), np.float16)
    nu = np.zeros((5, 25, 25), np.uint8)
    ff = np.zeros((10, 25, 25), np.float16)
    fu = np.zeros((5, 25, 25), np.uint8)
    O.build_windows(know, u, ctx, nf, nu, ff, fu)
    return nf, nu, ff, fu


def test_the_planes_equal_an_independent_reconstruction():
    """Property (spec 7, (b)): over the ship_demo bots-vs-bots match,
    observed every 24 ticks for 600 ticks, every plane of ``near_f`` /
    ``near_u`` / ``far_f`` / ``far_u`` equals ``_obs_reference.py``'s tile
    by tile and block by block (``has_los`` per cell, the loops' maxima,
    the counts from the picture) -- and the vector's ``air`` block is the
    near window's centre cell. Breaks if a plane reads the wrong seam,
    denominator, frame, fill or reduction. The reference draws ``watched``
    from the RECORDED centres and the far unit planes from the whole
    picture, so this is also spec 7's direct honesty check for them."""
    ses, spec, S = _ship_match()
    sim = ses.sim
    ctx = O.ObsContext.build(sim, spec, S)
    K = R.constants(sim, spec, S)
    # (ship_demo's doors are tiles, not entities, and its air is still: the
    # door and wind planes are covered by the random-map reduce test)
    seen = {"watched": 0, "lanes": 0, "enemies": 0, "records": 0}
    rows_checked = 0
    for _ in range(600):
        ses.tick()
        if sim.tick % 24 or ses.is_terminal():
            continue
        out = _observe(sim, ctx, spec, S)
        for i, u in enumerate(_rows(sim, spec, S)):
            if u is None:
                for k in ("near_f", "near_u", "far_f", "far_u"):
                    assert not out[k][i].any()
                continue
            nf, nu = R.near_planes(sim, K, u)
            ff, fu = R.far_planes(sim, K, u)
            for k, name in enumerate(O.NEAR_F_PLANES):
                _planes_close(out["near_f"][i, k], nf[k], name, (sim.tick, u.name))
            for k, name in enumerate(O.NEAR_U_PLANES):
                assert np.array_equal(out["near_u"][i, k], nu[k]), (name, sim.tick, u.name)
            for k, name in enumerate(O.FAR_F_PLANES):
                _planes_close(out["far_f"][i, k], ff[k], name, (sim.tick, u.name))
            for k, name in enumerate(O.FAR_U_PLANES):
                assert np.array_equal(out["far_u"][i, k], fu[k]), (name, sim.tick, u.name)
            # the air block is the near window's centre cell (float32 vs 16)
            centre = out["near_f"][i, :, O.HALF, O.HALF].astype(np.float64)
            assert np.allclose(out["vector"][i, 58:66], centre, atol=4e-3)
            seen["watched"] += int(out["near_u"][i, 2].any())
            seen["lanes"] += int(out["near_u"][i, 4].any())
            seen["enemies"] += int(out["far_u"][i, 2].any())
            seen["records"] += int(out["far_u"][i, 3].any())
            rows_checked += 1
        if ses.is_terminal():
            break
    assert rows_checked > 40
    assert all(v > 0 for v in seen.values()), seen


def test_the_route_and_place_planes_follow_the_order(tmp_path):
    """Property (spec 3.8, 7): a soldier ordered to the other room reads
    the far ``route`` plane as the flood's block minimum over ``L_ref``
    (-1 unreached) and ``place`` 1 on the blocks holding that room's
    tiles; without an order both are -1 / 0 everywhere; an opening's
    place is its rectangle. Breaks if the seeds, the reduce or the place
    mask move."""
    from test_orders import _sim as orders_sim
    from simulation import unit_orders as UO
    sim = orders_sim(graph=True)
    a = Unit("a", x=4, y=4, team=0)
    b = Unit("b", x=4, y=12, team=0)
    sim.add_unit(a)
    sim.add_unit(b)
    sim.start_team_picture((0,), memory_ticks=40)
    sim.order_intent(a.id, UO.ORDER_GO_TO, 1, 0, -1)
    sim.step()
    assert a.order_what == UO.ORDER_GO_TO and not b.order_what
    ctx = O.ObsContext.build(sim, PAIR_SPEC, 2)
    K = R.constants(sim, PAIR_SPEC, 2)
    know = O.TeamKnowledge.build(sim, sim.team_picture, 0)
    for u, has in ((a, True), (b, False)):
        _, _, ff, fu = _windows_of(sim, ctx, u, know)
        rff, rfu = R.far_planes(sim, K, u)
        _planes_close(ff[9], rff[9], "route", u.name)
        assert np.array_equal(fu[4], rfu[4])
        assert (ff[9] > -1).any() == has and fu[4].any() == has
        if has:
            # the block under the unit (the other room) is no place and its
            # route cell is reached and positive; at the place the route is 0
            assert fu[4, 12, 12] == 0 and 0 < ff[9, 12, 12] <= 4
            assert (ff[9][fu[4] == 1] == 0).all()
    sim.order_intent(a.id, UO.ORDER_GO_TO, UO.ORDER_OPENING_BASE + 0, 0, -1)
    sim.step()
    _, _, _, fu = _windows_of(sim, ctx, a)
    o = know.world.graph.openings[0]
    by0, bx0 = a.center_tile_y() // 4 - 12, a.center_tile_x() // 4 - 12
    want = np.zeros((25, 25), np.uint8)
    for y in range(o.y, o.y + o.h):
        for x in range(o.x, o.x + o.w):
            want[y // 4 - by0, x // 4 - bx0] = 1
    assert np.array_equal(fu[4], want)


def test_the_far_reduce_on_a_random_map():
    """Property (spec 3.3, 7): on a random room with random planes the
    reduced planes equal a per-tile reference -- the max over each 4 x 4
    block (padded with 0), the wall share (outside tiles solid), the door
    code's max, and the wind's direction from the block's largest load,
    first in row-major on a tie. Breaks if the reshape or the padding is
    off by one or the wind's tile is not the load's argmax."""
    from types import SimpleNamespace
    from simulation import unit_impulse as ui
    from simulation.gases import INERT_N2, O2, POISON as GP, SMOKE as GS, \
        TEARGAS as GT
    from test_wave_push import _room_map
    rng = np.random.default_rng(3)
    g = _room_map(18, 30)                       # 18 x 30: ragged blocks
    h, w = g.solid.shape
    open_ = ~g.solid
    g.temperature[...] = rng.integers(-5 << 16, 400 << 16, size=(h, w))
    g.fire[...] = rng.integers(0, 1 << 16, size=(h, w))
    for k in (GP, GS, GT):
        g.gas[k][...] = rng.integers(0, 1 << 24, size=(h, w))
    g.atmosphere[...] = rng.integers(0, 3 << 16, size=(h, w))
    n = rng.integers(13_000, 200_000, size=(h, w))
    g.gas[O2][open_] = (n // 5)[open_]
    g.gas[INERT_N2][open_] = (n - n // 5)[open_]
    g.wind_x[open_] = rng.integers(-20 << 16, 20 << 16, size=(h, w))[open_]
    g.wind_y[open_] = rng.integers(-20 << 16, 20 << 16, size=(h, w))[open_]
    ys, xs = np.nonzero(open_)
    pick = rng.choice(len(ys), size=len(ys) // 8, replace=False)
    g.solid[ys[pick], xs[pick]] = True
    doors = (SimpleNamespace(alive=True, state=1, span=[(5, 7), (5, 8)]),
             SimpleNamespace(alive=True, state=0, span=[(6, 7)]),
             SimpleNamespace(alive=False, state=0, span=[(2, 2)]))
    P = O.FieldPlanes(g, doors, ui.table_for(g))
    hb, wb = -(-h // 4), -(-w // 4)
    assert P.wall_share.shape == (hb, wb)
    load, vx, vy = g.wind_load_plane()
    for by in range(hb):
        for bx in range(wb):
            tiles = [(by * 4 + a, bx * 4 + b) for a in range(4) for b in range(4)]
            inb = [(y, x) for y, x in tiles if y < h and x < w]
            for name in O.PHYSICS_PLANES[:6]:
                want = max([0.0] + [P.raw[name][y, x] for y, x in inb])
                assert P.far_raw[name][by, bx] == want, (name, by, bx)
            best, at = 0.0, None
            for y, x in inb:
                if load[y, x] > best:
                    best, at = load[y, x], (y, x)
            for name in ("wind_x", "wind_y"):
                want = P.raw[name][at] if at is not None else 0.0
                assert P.far_raw[name][by, bx] == want, (name, by, bx)
            solid = sum(1 for y, x in tiles
                        if y >= h or x >= w or g.solid[y, x])
            assert P.wall_share[by, bx] == solid / 16.0
            assert P.far_door[by, bx] == max(
                [0] + [P.door[y, x] for y, x in inb])
    assert P.door[5, 7] == 1 and P.door[5, 8] == 1 and P.door[6, 7] == 2
    assert P.door[2, 2] == 0 and P.far_door[1, 1] == 2
    assert (P.raw["wind_x"][g.solid] == 0).all()


def test_the_sight_planes_equal_the_one_sight_test(tmp_path):
    """Property (spec 3.5, 7): ``watched`` cell for cell equals
    ``gmap.has_los(ry, rx, y, x)`` OR-ed over the team's records at their
    RECORDED centres (a remembered enemy that moved unseen is still watched
    from where it was seen; cells outside the map follow ``has_los``);
    ``sees_target`` is all 0 in 1g. Breaks if a window argument
    (``to_window``) is swapped, a live position is read, or the outside
    rule differs."""
    lvl = _level(tmp_path, "sight", wall_x=30, gap=(18, 23))
    sim, (me, foe, far) = _sim(lvl, [(0, 10, 20), (1, 45, 20), (1, 45, 5)],
                               sight="all_round")
    ctx = O.ObsContext.build(sim, PAIR_SPEC, 2)
    sim.step()
    recs = sim.team_picture.records(0)
    assert foe.id in recs and far.id not in recs      # far: behind the wall
    sim.team_picture._write(sim, 0, far)              # remembered by hand
    sim.set_aim(me.id, 0, -ONE)                       # look away: the foe
    for _ in range(12):                               # walks south unseen
        sim.step()
    sim.set_move_dir(foe.id, 0, ONE, 0)
    for _ in range(12):
        sim.step()
    recs = sim.team_picture.records(0)
    assert foe.id in recs and far.id in recs
    cx, cy = me.center_tile_x(), me.center_tile_y()
    nf, nu, _, _ = _windows_of(sim, ctx, me)
    want = np.zeros((25, 25), np.uint8)
    for i in range(25):
        for j in range(25):
            y, x = cy - 12 + i, cx - 12 + j
            want[i, j] = any(sim.gmap.has_los(r.centre[1], r.centre[0], y, x)
                             for r in recs.values())
    assert np.array_equal(nu[2], want)
    assert want.any() and not want.all()
    assert not nu[3].any()
    assert nu[2][12, 12] == 1                          # me, from the foe
    assert nu[0][12, 12 - cx] == 1                     # the hull at x = 0
    assert nu[0][12, 12 - cx - 1] == 1                 # outside the map: wall


def test_watched_is_drawn_from_the_recorded_centre(tmp_path):
    """Property (spec 3.5, 7: family case (b) with an occluder, and the
    direct check): me behind a wall with a gap sees an enemy through the
    gap, turns away and loses it; in one of two identical sims the enemy
    then walks out of the gap's line, unseen. Team 0's rows -- ``near_u``'s
    ``watched`` and the far ``records`` plane included -- stay byte-equal at
    every decision, and ``watched`` is the lines from where the team LAST
    SAW it (through the gap), not from where it stands (behind the wall:
    a different plane). Breaks if the plane reads a live position."""
    sims, (ua, ub), ctxs, S = _pair(tmp_path, "watched",
                                    [(0, 10, 20), (1, 40, 20)],
                                    wall_x=30, gap=(18, 23), memory_ticks=200)
    for me in (ua[0], ub[0]):
        place_facing(me, 0)
    _step_both(sims)
    for s, u in zip(sims, (ua, ub)):
        assert u[1].id in s.team_picture.records(0)     # seen through the gap
        s.set_aim(u[0].id, 0, -ONE)                     # turn north, away
    _step_both(sims, 16)
    for s, u in zip(sims, (ua, ub)):
        assert u[1].id not in vision.state_for(s).visible_by_team.get(0, ())
    rec = sims[1].team_picture.records(0)[ub[1].id]
    sims[1].set_move_dir(ub[1].id, 0, ONE, ONE)         # south, unseen, fast
    _compare_team0(sims, ctxs, S, 48)
    sim, (me, foe), ctx = sims[1], ub, ctxs[1]
    assert sim.team_picture.records(0)[foe.id].centre == rec.centre
    assert foe.center_tile_y() >= 23 > rec.centre[1]    # behind the wall now
    cx, cy = me.center_tile_x(), me.center_tile_y()
    _, nu, _, fu = _windows_of(sim, ctx, me)
    recorded = np.zeros((25, 25), np.uint8)
    live = np.zeros((25, 25), np.uint8)
    for i in range(25):
        for j in range(25):
            y, x = cy - 12 + i, cx - 12 + j
            recorded[i, j] = sim.gmap.has_los(rec.centre[1], rec.centre[0], y, x)
            live[i, j] = sim.gmap.has_los(foe.center_tile_y(),
                                          foe.center_tile_x(), y, x)
    assert np.array_equal(nu[2], recorded)
    assert recorded.any() and not np.array_equal(live, recorded)
    by0, bx0 = cy // 4 - 12, cx // 4 - 12
    assert fu[3, rec.centre[1] // 4 - by0, rec.centre[0] // 4 - bx0] == 1
    assert fu[3].sum() == 1 and not fu[2].any()


def test_the_lanes_plane_and_the_lane_pair(tmp_path):
    """Property (spec 3.6, 2.2 cols 52-53, 7): a mate holding its trigger
    facing east marks the near cells along its facing to its range,
    widened by the dials ``[bots] lane_cell`` / ``lane_halfwidth``, and no
    others; released, none; my own lane marks nothing in my plane; a mate
    standing in my lane reads ``it_in_my_lane`` and one whose lane holds
    me ``me_in_its_lane``. Breaks if the plane reads the wrong owner or the
    pair the shared grid instead of each owner's own."""
    lvl = _level(tmp_path, "lanes")
    sim, (me, mate) = _sim(lvl, [(0, 10, 20), (0, 20, 20)], sight="all_round")
    ctx = O.ObsContext.build(sim, PAIR_SPEC, 2)
    place_facing(me, 0)
    place_facing(mate, 0)
    sim.step()
    cell, hw = int(CFG.bots.lane_cell), int(CFG.bots.lane_halfwidth)

    def plane(u):
        return _windows_of(sim, ctx, u)[1][4]

    def pair(u):
        out = _observe(sim, ctx, PAIR_SPEC, 2)
        row = 0 if u is me else 1
        i = int(np.flatnonzero(out["list_mask"][row])[0])
        e = out["list"][row, i]
        assert e[0] == O.TYPE_MATE
        return int(e[52]), int(e[53])

    assert not plane(me).any() and pair(me) == (0, 0)
    sim.set_trigger(mate.id, True)
    p = plane(me)
    grid = tp.fire_lanes(sim, 0, cell, hw)
    cx, cy = me.center_tile_x(), me.center_tile_y()
    h, w = sim.gmap.solid.shape
    want = np.zeros((25, 25), np.uint8)
    for i in range(25):
        for j in range(25):
            y, x = cy - 12 + i, cx - 12 + j
            if 0 <= y < h and 0 <= x < w:
                want[i, j] = grid[y // cell, x // cell] == mate.id
    assert np.array_equal(p, want) and want.any()
    mx = mate.center_tile_x()
    # east of the mate along its row: marked; far north of it: not
    assert p[12, 12 + (mx + 2 - cx)] == 1
    assert p[12 - (6 + hw * cell), 12 + (mx - cx)] == 0
    # I am west of the mate: not in its lane; it is not in mine (no trigger)
    assert pair(me) == (0, 0)
    sim.set_trigger(me.id, True)                 # my lane runs east over it
    assert not plane(mate)[12, 12] or pair(mate) == (1, 0)
    assert pair(me) == (0, 1)
    assert pair(mate) == (1, 0)
    assert not plane(me).any() or np.array_equal(plane(me), want)  # mine never
    place_facing(mate, PI_Q16)                   # it turns west: onto me
    assert pair(me) == (1, 1)
    sim.set_trigger(mate.id, False)
    sim.set_trigger(me.id, False)
    assert not plane(me).any() and pair(me) == (0, 0)


def test_felt_temperature_is_the_heat_rows(tmp_path):
    """Property (spec 1.3, 7): ``exchange.felt_temperature`` equals the heat
    row's ``t_felt`` written by hand (the reference) on random bodies over
    random ``heat`` and radiation-sensor planes, bit for bit; it answers
    ``heat_ambient_ref`` with no flux; and the vector's col 96 is that
    temperature over the profile's ``temperature_max``, squashed. The
    row's damage is held by ``tools/match_pin_check.py`` and
    ``test_unit_heat_damage.py``. Breaks if the row or the observation
    carries a copy of the law."""
    from simulation import exchange
    lvl = _level(tmp_path, "felt")
    sim, (me, _) = _sim(lvl, [(0, 10, 20), (0, 30, 20)], sight="all_round")
    ctx = O.ObsContext.build(sim, PAIR_SPEC, 2)
    g = sim.gmap
    rng = np.random.default_rng(5)
    dials = exchange.heat_dials(CFG)
    assert exchange.felt_temperature(g, me, dials)[0] == float(
        CFG.combat.heat_ambient_ref)
    g.heat[...] = rng.integers(0, 3 << 16, size=g.heat.shape)
    g.heat[g.heat < (1 << 15)] = 0
    rad = getattr(g, "rad_flux_sweep", None)
    if rad is not None:
        rad[...] = rng.integers(0, 5 << 20, size=rad.shape)
    hits = 0
    for _ in range(200):
        me.x, me.y = float(rng.integers(1, 55)), float(rng.integers(1, 35))
        got = exchange.felt_temperature(g, me, dials)
        want = R.felt_temperature(sim, me)
        assert got[0] == want
        assert (got[1] is None) == (want == float(CFG.combat.heat_ambient_ref))
        hits += got[1] is not None
    assert hits > 100
    know = O.TeamKnowledge.build(sim, sim.team_picture, 0)
    v = np.zeros(O.V, np.float32)
    O.build_vector(know, me, ctx, v)
    assert v[96] == np.float32(O.squash(O.ratio(
        exchange.felt_temperature(g, me, dials)[0],
        me.environment.temperature_max)))
