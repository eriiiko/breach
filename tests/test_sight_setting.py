"""#195 patch 1a -- sight as a SETTING OF THE MATCH (RL agent design 3.6).

``sight = "cone" | "all_round"`` is a key of match.toml (default ``"cone"``,
written only when it is not the default), held by the sim
(``Simulation.sight``, set by ``spawn_match``) and read by ``vision.can_see``
from the sim it is handed. Under ``"all_round"`` every living non-zombie unit
sees every living unit of another team along a clear ``vision.ray_clear``
line -- no cone, no radius -- and a zombie observes nothing; under ``"cone"``
the path is the game's, untouched (the pre-G8p pinned bots_v1 match and
``tools/match_pin_check.py`` hold it bit for bit). A run file's optional
``[run] sight`` replaces the level's value and so rides in the recorded spec;
a recording without the field is a cone match (no format version moves).

Scenes reuse ``test_bots_v1``'s open hull box (a unit spawns facing NORTH,
its cone 55 deg, its awareness radius 4 tiles).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402,F401

import level_lib  # noqa: E402
import level_loader  # noqa: E402
import match_replay  # noqa: E402
import rl_env  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import vision  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import (DEFAULT_SIGHT, MatchSpec, MatchTeam,  # noqa: E402
                              SIGHT_RULES, parse_match)
from simulation.match_session import MatchSession  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from test_bots_v1 import BOTS, ROOMS, _level, _sim  # noqa: E402
from test_rl_env import _box, _run_toml  # noqa: E402

ALL_ROUND = "all_round"


def _all_round(sim):
    sim.sight = ALL_ROUND
    sim._vision_cache = None
    return sim


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------
def test_all_round_sees_a_unit_directly_behind_with_a_clear_line(tmp_path):
    """Property: under all_round an observer sees an enemy directly BEHIND it,
    far outside its 4-tile awareness radius, when the line is clear; under
    cone it does not (the facing is north, the enemy due south). Breaks if
    all_round still applies the cone / radius, or if the cone path starts
    seeing behind."""
    lvl = _level(tmp_path, "behind")
    sim, (obs, foe) = _sim(lvl, [(0, 10, 10), (1, 10, 32)])   # foe 22 south
    assert not vision.can_see(sim, obs, foe)
    _all_round(sim)
    assert vision.can_see(sim, obs, foe)
    assert vision.can_see(sim, foe, obs)
    # and the per-tick cache and the team union follow the same rule
    st = vision.state_for(sim)
    assert foe.id in vision.visible_enemy_ids(sim, 0)
    assert st.observers_of(foe.id) == (obs.id,)


def test_all_round_stops_at_a_wall(tmp_path):
    """Property: under all_round a wall between observer and enemy blocks
    sight (walls are the only limit): the enemy behind a full-height wall is
    unseen, the same enemy with the wall pierced by a gap on the line is
    seen. Breaks if all_round skips `vision.ray_clear` (x-ray sight)."""
    lvl = _level(tmp_path, "walled", wall_x=29)
    sim, (obs, foe) = _sim(lvl, [(0, 10, 20), (1, 45, 20)])
    _all_round(sim)
    assert not vision.can_see(sim, obs, foe)
    assert foe.id not in vision.visible_enemy_ids(sim, 0)

    lvl2 = _level(tmp_path, "gapped", wall_x=29, gap=(10, 30))
    sim2, (obs2, foe2) = _sim(lvl2, [(0, 10, 20), (1, 45, 20)])
    _all_round(sim2)
    assert vision.can_see(sim2, obs2, foe2)


def test_a_zombie_observes_nothing_under_all_round(tmp_path):
    """Property: under all_round a ZOMBIE is no observer (nothing reads what a
    zombie sees; its waking is ai_zombie's own law) -- not even a marine at
    its feet -- while a marine does see the zombie, and under cone the
    zombie's cone path is the old one (it sees a marine in front of it).
    Breaks if all_round lets zombies observe (all-round sight would triple
    its per-tick cost for nothing), or if it hides zombies from marines."""
    lvl = _level(tmp_path, "zed")
    # the zombie (id order: u0) faces north with the marine 5 tiles ahead
    sim, (zed, mar) = _sim(lvl, [(ZOMBIE_TEAM, 10, 20), (0, 10, 15)])
    assert zed.is_zombie and not mar.is_zombie
    assert vision.can_see(sim, zed, mar)            # cone: unchanged law
    _all_round(sim)
    assert not vision.can_see(sim, zed, mar)
    assert vision.can_see(sim, mar, zed)
    st = vision.compute(sim)
    assert st.seen_by.get(mar.id, ()) == ()
    assert st.seen_by[zed.id] == (mar.id,)
    assert ZOMBIE_TEAM not in st.visible_by_team


def test_all_round_never_sees_a_teammate_or_a_corpse(tmp_path):
    """Property: all_round is about ENEMIES that are alive -- no observer sees
    its own team through `can_see`'s pair test, and a dead unit is neither
    observer nor target. Breaks if the team / alive guards are lost."""
    lvl = _level(tmp_path, "mates")
    sim, (a, b, c) = _sim(lvl, [(0, 10, 10), (0, 10, 30), (1, 30, 30)])
    _all_round(sim)
    assert not vision.can_see(sim, a, b)
    c.alive = False
    assert not vision.can_see(sim, a, c)
    assert not vision.can_see(sim, c, a)


def test_the_sim_holds_cone_everywhere_but_a_match(tmp_path):
    """Property: a sim that is not a match holds "cone"; `spawn_match` sets
    the sim's rule from the spec (both ways, so an all_round match followed
    by a reset sim spawned at cone is cone again). Breaks if the rule is a
    global, or read by a caller, or not set at the one spawn."""
    spec = MatchSpec(teams=(BOTS, BOTS), seed=1, time_limit_s=3.0)
    lvl = _level(tmp_path, "held", spec=spec)
    from simulation import Simulation
    from simulation.ruleset import MatchRules
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    assert sim.sight == "cone"
    sim2 = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False,
                      ruleset=MatchRules.from_spec(spec))
    sim2.spawn_match(replace(spec, sight=ALL_ROUND))
    assert sim2.sight == ALL_ROUND and sim.sight == "cone"
    sim3 = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False,
                      ruleset=MatchRules.from_spec(spec))
    sim3.spawn_match(spec)
    assert sim3.sight == "cone"


def test_a_match_under_all_round_differs_from_the_cone_match(tmp_path):
    """Property: the setting is live -- the same match and seed under
    all_round ends in another state than under cone (the bots see, and so
    shoot, behind them and across the map), and each is itself a pure
    function of spec + seed. Breaks if the key never reaches the sim."""
    cone = MatchSpec(teams=(BOTS, BOTS), seed=1, time_limit_s=6.0)
    lvl = _level(tmp_path, "live", spec=cone)
    a = run_match(lvl, cone)
    b = run_match(lvl, replace(cone, sight=ALL_ROUND))
    b2 = run_match(lvl, replace(cone, sight=ALL_ROUND))
    assert a.tick_digest != b.tick_digest
    assert b.tick_digest == b2.tick_digest


# ---------------------------------------------------------------------------
# The match file: door and writer
# ---------------------------------------------------------------------------
_TEAM = {"n": 2, "loadout": "rifleman", "env_profile": "default",
         "controller": "idle"}


def test_the_match_door_refuses_an_unknown_sight_by_name():
    """Property: parse_match refuses a sight that is not "cone" / "all_round"
    (a typo, a number, a bool), naming the key and the value, and accepts
    both. Breaks if a typo in match.toml silently plays as cone."""
    for bad in ("fisheye", "ALL_ROUND", 1, True, ""):
        raw = {"seed": 1, "sight": bad, "team": [dict(_TEAM), dict(_TEAM)]}
        with pytest.raises(ValueError, match=r"sight.*is not a sight rule"):
            parse_match(raw, ROOMS, "match.toml")
    with pytest.raises(ValueError, match="fisheye"):
        parse_match({"seed": 1, "sight": "fisheye",
                     "team": [dict(_TEAM), dict(_TEAM)]}, ROOMS, "match.toml")
    for ok in SIGHT_RULES:
        raw = {"seed": 1, "sight": ok, "team": [dict(_TEAM), dict(_TEAM)]}
        assert parse_match(raw, ROOMS, "match.toml").sight == ok
    raw = {"seed": 1, "team": [dict(_TEAM), dict(_TEAM)]}
    assert parse_match(raw, ROOMS, "match.toml").sight == DEFAULT_SIGHT


def test_a_default_spec_writes_no_sight_line_and_a_match_file_round_trips(
        tmp_path):
    """Property: the writer omits `sight` at the default, so every existing
    match.toml is byte for byte what the writer makes of its own parse
    (ship_demo's, checked); all_round writes ONE `sight` line and reads back
    as all_round. Breaks if the writer emits the default (every match file
    moves) or drops a non-default value."""
    src = ROOT / "levels" / "ship_demo" / "match.toml"
    lvl = level_loader.load("ship_demo", str(ROOT / "levels"))
    assert lvl.match.sight == "cone"
    assert level_lib.format_match_text(lvl.match) == \
        src.read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert "sight" not in src.read_text(encoding="utf-8")

    spec = MatchSpec(teams=(BOTS, BOTS), seed=1, time_limit_s=3.0)
    text = level_lib.format_match_text(spec)
    assert "sight" not in text
    text2 = level_lib.format_match_text(replace(spec, sight=ALL_ROUND))
    assert text2.count("sight") == 1 and 'sight = "all_round"\n' in text2
    lvl2 = _level(tmp_path, "wr", spec=replace(spec, sight=ALL_ROUND))
    assert lvl2.match.sight == ALL_ROUND


# ---------------------------------------------------------------------------
# The run file
# ---------------------------------------------------------------------------
def test_the_run_file_sight_key_overrides_and_is_refused_by_name(tmp_path):
    """Property: `[run] sight` replaces the level's value in the spec every
    episode plays (and so in the spec the recorder stores); absent, the
    level's own value stands; a bad value is refused naming the key.
    Breaks if the override is dropped before the spawn, applied to some
    episodes only, or a typo is accepted."""
    _box(tmp_path, "abox", spec=MatchSpec(
        teams=(BOTS, BOTS), seed=1, zombies=0, time_limit_s=4.0))
    p = _run_toml(tmp_path / "run.toml", levels=["abox"],
                  levels_dir=tmp_path, n_envs=1,
                  extra_run='sight = "all_round"\n')
    run = rl_env.load_run(p)
    assert run.sight == ALL_ROUND
    levels = rl_env.load_levels(run)
    layout = rl_env.make_layout(run, levels)
    r = rl_env.EnvRunner(0, run, layout, levels, breach_physics=bp,
                         record=True)
    r.reset()
    assert r.session.spec.sight == ALL_ROUND
    assert r.session.sim.sight == ALL_ROUND
    r.reset()                                   # a second episode, too
    assert r.session.spec.sight == ALL_ROUND

    plain = rl_env.load_run(_run_toml(tmp_path / "plain.toml",
                                      levels=["abox"], levels_dir=tmp_path,
                                      n_envs=1))
    assert plain.sight is None
    r2 = rl_env.EnvRunner(0, plain, layout, levels, breach_physics=bp)
    r2.reset()
    assert r2.session.spec.sight == "cone"

    bad = _run_toml(tmp_path / "bad.toml", levels=["abox"],
                    levels_dir=tmp_path, n_envs=1,
                    extra_run='sight = "fisheye"\n')
    with pytest.raises(ValueError, match=r"\[run\] sight.*fisheye"):
        rl_env.load_run(bad)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------
def test_a_recorded_all_round_match_replays_with_every_checkpoint_equal(
        tmp_path):
    """Property: an all_round match recorded through the tap, written to disk
    and read back, replays as an all_round match (the spec in the header
    carries the rule) with every checkpoint equal and the same end; the
    replay of a header that says cone against this recording is refused.
    Breaks if the rule is not part of the recorded spec (the replay would
    play cone and diverge) or the header check ignores it."""
    spec = MatchSpec(teams=(BOTS, BOTS), seed=2, time_limit_s=6.0,
                     sight=ALL_ROUND)
    lvl = _level(tmp_path, "rec", spec=spec)
    out = run_match(lvl, spec, record=True, checkpoint_every_ticks=5)
    rec = out.recording
    assert rec.spec.sight == ALL_ROUND and len(rec.checkpoints) >= 3
    path = tmp_path / "all_round.replay.gz"
    match_replay.write_recording(rec, path)
    rec2 = match_replay.read_recording(path)
    assert rec2.spec.sight == ALL_ROUND
    back = run_match(lvl, replay=rec2)
    assert back.checkpoints_verified == len(rec.checkpoints)
    assert (back.field_digest, back.tick_digest) == \
        (out.field_digest, out.tick_digest)
    with pytest.raises(match_replay.ReplayRefused):
        MatchSession(lvl, replace(spec, sight="cone"), replay=rec2,
                     breach_physics=bp)


def test_an_old_spec_without_the_field_reads_as_cone_and_cone_writes_none():
    """Property: a recorded-spec dict without `sight` (every recording made
    before this patch, FORMAT_VERSION unchanged) reads as a cone spec;
    `spec_to_dict` writes the field only for a non-default rule, so a cone
    recording's header is byte for byte the old one. Breaks if the field is
    required, or always written (every old header would differ from a new
    cone recording of the same match)."""
    spec = MatchSpec(teams=(BOTS, BOTS), seed=1, zombies=1)
    d = match_replay.spec_to_dict(spec)
    assert "sight" not in d
    assert match_replay.spec_from_dict(d).sight == "cone"
    assert match_replay.spec_from_dict(d) == spec
    d2 = match_replay.spec_to_dict(replace(spec, sight=ALL_ROUND))
    assert d2["sight"] == ALL_ROUND
    assert match_replay.spec_from_dict(d2).sight == ALL_ROUND
    # 1a moved no format version (v7 then); later patches may (1f: v8)
    assert match_replay.FORMAT_VERSION >= 7
