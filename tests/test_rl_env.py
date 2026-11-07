"""#29 P1a -- the batched environment's ``cpu_farm`` backend (``src/rl_env.py``),
the steppable match under it (``simulation.match_session.MatchSession``) and
the ``policy`` controller driving it.

The gates (residency design §9 row P1a, the brief's G-a..G-c):

- G-a REFERENCE EQUALITY: the farm, its policy issuing zero rows (move
  cleared, no aim, no trigger), plays exactly ``run_match`` with idle teams
  and the zombie horde: every checkpoint's ``(field_digest, tick_digest)``
  equal, for three episode seeds on ship_demo and on a box where the horde
  kills and converts.
- G-b REPLAYABILITY: a random-action episode on the farm equals the same
  actions in-process, is recorded through the sim's one intent tap, and
  replays under ``tools/run_match.py --replay`` through every checkpoint.
- G-c EPISODE PURITY: episode k of env e has the same level and seed (and so
  the same tick-0 state) whatever the farm's size and whenever other envs
  reset.

Plus the run-file door, the backend door, the worker's import isolation,
the reward / terminal semantics, the counters' crediting and the smoke
driver. Levels are written through ``level_lib``; references run through
``run_match``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
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

import control_policy as cp  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import match_replay  # noqa: E402
import rl_env  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation.events import UnitHitEvent, UnitKilledEvent  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.match import MatchSpec, MatchTeam, SpawnRoom  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402

EVERY = 60          # [replay] checkpoint_every_ticks, as shipped
IDLE = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                 controller="idle")
# The zombie room right next to team 0 (tests/test_zombie_control.py's box):
# with idle marines the horde kills team 0 and converts it, team 1 wins.
ROOMS = [
    SpawnRoom(team=0, x=2, y=2, w=8, h=8, max_per_team=4),
    SpawnRoom(team=1, x=30, y=14, w=8, h=8, max_per_team=4),
    SpawnRoom(team=ZOMBIE_TEAM, x=12, y=2, w=8, h=8, max_per_team=4),
]
BOX_SPEC = MatchSpec(teams=(IDLE, IDLE), seed=1, zombies=3,
                     time_limit_s=20.0)
ALL_CHANNELS = ("pressure", "smoke", "water_depth", "temperature", "fire",
                "o2", "solid", "material", "units_own", "units_enemy",
                "units_zombie")


def _box(root: Path, name: str, spec=BOX_SPEC, w=40, h=24):
    lvl = root / name
    lvl.mkdir(parents=True)
    level_lib.write_level_header(lvl, name=name, tile_size_m=1.0)
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    level_lib.write_tilemap_csv(lvl, tm, csv_bak=False)
    from PIL import Image
    Image.new("RGB", (w, h)).save(lvl / "diffuse.png")
    level_lib.write_spawn_rooms(lvl / "level.toml", ROOMS)
    level_lib.write_match(lvl, spec)
    return level_loader.load(name, str(root))


def _ship_copy(root: Path, time_limit_s: float):
    """ship_demo copied verbatim into ``root``, its match.toml rewritten (the
    writer) with idle teams and a shorter time limit."""
    dst = root / "ship_demo"
    shutil.copytree(ROOT / "levels" / "ship_demo", dst)
    src = level_loader.load("ship_demo", str(ROOT / "levels"))
    spec = replace(src.match, time_limit_s=time_limit_s,
                   teams=tuple(replace(t, controller="idle")
                               for t in src.match.teams))
    level_lib.write_match(dst, spec)
    return level_loader.load("ship_demo", str(root))


def _run_toml(path: Path, *, levels, levels_dir, seed=11, n_envs=2,
              decimation=4, policy='"policy"', channels=ALL_CHANNELS,
              crop_radius=4, vector=("hp", "team", "alive", "facing", "x",
                                     "y"), reward='kills = 1.0\n'
              'team_kills = -1.0\ndeaths_melee = -0.5\n', extra_run=""):
    def lst(xs):
        return "[" + ", ".join(f'"{x}"' for x in xs) + "]"
    path.write_text(
        "[run]\n"
        f"seed = {seed}\nn_envs = {n_envs}\ndecimation = {decimation}\n"
        f"levels = {lst(levels)}\npolicy = {policy}\n"
        f"levels_dir = '{Path(levels_dir).as_posix()}'\n{extra_run}"
        "[obs]\n"
        f"channels = {lst(channels)}\ncrop_radius = {crop_radius}\n"
        f"vector = {lst(vector)}\n"
        f"[reward]\n{reward}", encoding="utf-8")
    return path


def _zero_actions(env):
    rows = cp.zero_rows(env.layout.n_rows)
    return np.broadcast_to(rows, (env.n_envs,) + rows.shape).copy()


def _episode0_checkpoints(env, actions_fn, max_steps=10_000):
    """Reset, then step until every env's episode 0 ended; per env the
    checkpoints of episode 0 as {tick: (field, tick)} and its result."""
    obs, info = env.reset()
    cps = [dict() for _ in range(env.n_envs)]
    results = [None] * env.n_envs
    done = np.zeros(env.n_envs, dtype=bool)

    def take(info_cps):
        for e, lst in enumerate(info_cps):
            for k, t, f, d in lst:
                if k == 0:
                    cps[e][t] = (f, d)
    take(info["checkpoints"])
    for _ in range(max_steps):
        if done.all():
            break
        _, rew, term, trunc, info = env.step(actions_fn())
        take(info["checkpoints"])
        for e in np.flatnonzero(info["ended"] & ~done):
            results[e] = (info["result"][e], bool(term[e]), bool(trunc[e]),
                          rew[e].copy(), info["counters"][e].copy())
            done[e] = True
    assert done.all(), "an episode did not end"
    return cps, results, info


# ---------------------------------------------------------------------------
# G-a: reference equality
# ---------------------------------------------------------------------------
def test_g_a_zero_rows_play_run_match_s_idle_match_on_ship_demo(tmp_path):
    """Property (G-a, ship_demo): three farm envs whose policy issues zero
    rows reproduce ``run_match`` with both teams ``idle`` and the horde on
    ``zombies``, at their drawn episode seeds, checkpoint for checkpoint
    (tick 0, every 60 ticks, the last tick). Breaks if issuing an empty row
    changes sim state, the farm's spec or seed differs from the episode
    draw, MatchSession steps differently from run_match, or a worker's
    observation / reward pass writes the sim."""
    lv = _ship_copy(tmp_path, time_limit_s=20.0)
    run = rl_env.load_run(_run_toml(tmp_path / "run.toml",
                                    levels=["ship_demo"],
                                    levels_dir=tmp_path, n_envs=3))
    with rl_env.BatchedEnv(run, n_workers=3) as env:
        cps, results, info = _episode0_checkpoints(
            env, lambda: _zero_actions(env))
    seeds = [rl_env.episode_draw(run.seed, e, 0, 1)[0] for e in range(3)]
    assert len(set(seeds)) == 3
    for e, s in enumerate(seeds):
        ref = run_match(lv, lv.match, seed=s, record=True,
                        checkpoint_every_ticks=EVERY)
        assert cps[e] == ref.recording.checkpoints, f"env {e} seed {s}"
        res, term, trunc, _, _ = results[e]
        assert res == {"outcome": ref.result.outcome, "team": ref.result.team,
                       "ticks": ref.result.ticks}


def test_g_a_zero_rows_play_the_idle_match_where_the_horde_kills(tmp_path):
    """Property (G-a with kills): on a box where the horde eats team 0 and
    converts it, the zero-row farm env plays ``run_match``'s idle match
    checkpoint for checkpoint and ends with the same result; its terminal is
    ``terminated`` (one marine team left), not ``truncated``; team 0's rows
    carry the melee deaths and the reward is the weighted counters. Breaks
    if the policy drives a converted unit, a death is not credited to its
    row, or terminated / truncated swap."""
    lv = _box(tmp_path, "zbox")
    run = rl_env.load_run(_run_toml(tmp_path / "run.toml", levels=["zbox"],
                                    levels_dir=tmp_path, n_envs=1,
                                    decimation=3))
    with rl_env.BatchedEnv(run, n_workers=1) as env:
        S = env.layout.n_slots
        cps, results, _ = _episode0_checkpoints(
            env, lambda: _zero_actions(env))
    s = rl_env.episode_draw(run.seed, 0, 0, 1)[0]
    ref = run_match(lv, lv.match, seed=s, record=True,
                    checkpoint_every_ticks=EVERY)
    assert cps[0] == ref.recording.checkpoints
    res, term, trunc, rew, counters = results[0]
    assert ref.result.outcome == "win" and res["team"] == ref.result.team
    assert term and not trunc
    weights = run.reward_weights()
    assert np.allclose(rew, (counters * weights).sum(axis=1))


def test_truncation_is_the_time_limit_and_not_a_termination(tmp_path):
    """Property: a match that runs to its time limit with both marine teams
    alive ends ``truncated`` and not ``terminated``, its result a draw, and
    the env comes back reset to episode 1 at tick 0 inside the same step
    (the terminal observation in ``final_obs``). Breaks if the time limit
    is reported as a win/loss terminal or the in-step reset is skipped."""
    spec = MatchSpec(teams=(IDLE, IDLE), seed=1, zombies=0,
                     time_limit_s=2.0)
    _box(tmp_path, "tbox", spec)
    run = rl_env.load_run(_run_toml(tmp_path / "run.toml", levels=["tbox"],
                                    levels_dir=tmp_path, n_envs=1,
                                    decimation=5))
    with rl_env.BatchedEnv(run, n_workers=1) as env:
        env.reset()
        for _ in range(100):
            obs, _, term, trunc, info = env.step(_zero_actions(env))
            if info["ended"][0]:
                break
    assert trunc[0] and not term[0]
    assert info["result"][0]["outcome"] == "draw"
    assert info["episode"][0] == 1 and info["tick"][0] == 0
    assert info["final_obs"]["valid"][0].any() and obs["valid"][0].any()


# ---------------------------------------------------------------------------
# G-b: a random episode records and replays
# ---------------------------------------------------------------------------
def test_g_b_a_random_farm_episode_records_and_replays(tmp_path):
    """Property (G-b): a seeded random-action episode on the farm (one
    worker process) has the same checkpoints as the same action rows driven
    in-process through ``EnvRunner`` with the recording tap on; that
    recording, written to disk, replays under ``tools/run_match.py --replay``
    in a fresh process through every checkpoint (exit 0). Breaks if the tap
    misses a policy intent or its order, an argument is stored inexactly
    (the float fuse), the worker's shared-memory path changes what the sim
    sees, or a farm episode depends on per-process state."""
    _ship_copy(tmp_path, time_limit_s=20.0)
    run = rl_env.load_run(_run_toml(tmp_path / "run.toml",
                                    levels=["ship_demo"],
                                    levels_dir=tmp_path, n_envs=1,
                                    decimation=2, seed=5))
    rng = np.random.default_rng(1234)
    sent = []
    with rl_env.BatchedEnv(run, n_workers=1) as env:
        U = env.layout.n_rows

        def acts():
            a = cp.random_action_rows(rng, U)[None]
            sent.append(a[0].copy())
            return a
        farm_cps, results, _ = _episode0_checkpoints(env, acts)
        layout = env.layout
    levels = rl_env.load_levels(run)
    r = rl_env.EnvRunner(0, run, layout, levels, breach_physics=bp,
                         record=True)
    local = {0: r.reset()[0][2:]}
    for a in sent:
        out = r.step(a)
        local.update({t: (f, d) for _, t, f, d in out["checkpoints"]})
        if out["ended"]:
            break
    assert out["ended"] and local == farm_cps[0]
    rec = r.last_recording
    assert rec.n_intents() > 0 and rec.checkpoints == farm_cps[0]
    kinds = {k for v in rec.intents.values() for k, _ in v}
    assert {"set_aim", "set_trigger"} <= kinds
    path = tmp_path / "farm.replay.gz"
    match_replay.write_recording(rec, path)
    env_ = dict(os.environ)
    env_.pop("PYTHONPATH", None)
    p = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "run_match.py"),
         "--replay", str(path), "--levels-dir", str(tmp_path)],
        cwd=str(ROOT), env=env_, capture_output=True, text=True,
        timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (f"replay: every checkpoint matched ({len(rec.checkpoints)})"
            in p.stdout)


# ---------------------------------------------------------------------------
# G-c: episode purity
# ---------------------------------------------------------------------------
def test_episode_draw_is_a_pure_function_of_seed_env_and_k():
    """Property: ``episode_draw`` returns the same (seed, level) for the same
    arguments and distinct seeds for distinct (run seed, env, k), with every
    level of a run drawn. Breaks if the draw reads hidden state, collides
    across envs or episodes, or never reaches a level."""
    grid = [(rs, e, k) for rs in (0, 1, 2 ** 40 + 3) for e in range(4)
            for k in range(4)]
    a = [rl_env.episode_draw(rs, e, k, 3) for rs, e, k in grid]
    b = [rl_env.episode_draw(rs, e, k, 3) for rs, e, k in grid]
    assert a == b
    assert len({s for s, _ in a}) == len(grid)
    assert {li for _, li in a} == {0, 1, 2}
    assert all(0 <= s < 2 ** 64 for s, _ in a)


def test_g_c_episodes_do_not_depend_on_farm_size_or_reset_timing(tmp_path):
    """Property (G-c): over two farms of different N and worker counts, and
    different reset patterns, every (env, episode) that both ran has the
    same level, the same match seed and the same tick-0 digests. Breaks if
    an env's episode sequence depends on N, on which envs reset before it,
    or on a counter shared between envs."""
    _box(tmp_path, "abox")
    _box(tmp_path, "bbox", replace(BOX_SPEC, zombies=1))
    seen = []
    plans = [(2, 1, [[True, False], [True, False], [False, True]]),
             (3, 2, [[False, True, False], [False, True, False],
                     [True, False, True], [False, True, False]])]
    for n, workers, masks in plans:
        run = rl_env.load_run(_run_toml(
            tmp_path / f"run{n}.toml", levels=["abox", "bbox"],
            levels_dir=tmp_path, n_envs=n, seed=99))
        got = {}
        with rl_env.BatchedEnv(run, n_workers=workers) as env:
            _, info = env.reset()

            def take(info):
                for e in range(n):
                    for k, t, f, d in info["checkpoints"][e]:
                        if t == 0:
                            got[(e, k)] = (int(info["level"][e]),
                                           int(info["episode_seed"][e]), f, d)
            take(info)
            for m in masks:
                _, info = env.reset(np.array(m))
                take(info)
        seen.append(got)
    a, b = seen
    common = set(a) & set(b)
    assert len(common) >= 4 and any(k > 0 for _, k in common)
    for key in common:
        assert a[key] == b[key], key
    assert len({v[0] for v in list(a.values()) + list(b.values())}) == 2
    for (e, k), v in a.items():
        assert (v[1], v[0]) == rl_env.episode_draw(99, e, k, 2)


# ---------------------------------------------------------------------------
# Doors, isolation, counters, the smoke driver
# ---------------------------------------------------------------------------
_OK = dict(levels=["abox"])


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s + "[extra]\nx = 1\n", "unknown table"),
    (lambda s: s.replace("n_envs = 2\n", "n_envs = 2\nnenvs = 2\n"),
     "nenvs"),
    (lambda s: s.replace("decimation = 4\n", ""), "decimation"),
    (lambda s: s.replace("seed = 11", "seed = true"), "seed"),
    (lambda s: s.replace("seed = 11", "seed = -1"), "seed"),
    (lambda s: s.replace("n_envs = 2", "n_envs = 0"), "n_envs"),
    (lambda s: s.replace('levels = ["abox"]', "levels = []"), "levels"),
    (lambda s: s.replace('policy = "policy"', 'policy = "robots"'),
     "robots"),
    (lambda s: s.replace('channels = ["pressure"', 'channels = ["light"'),
     "light"),
    (lambda s: s.replace('channels = ["pressure"', 'channels = ["presure"'),
     "presure"),
    (lambda s: s.replace('vector = ["hp"', 'vector = ["mana"'), "mana"),
    (lambda s: s.replace("kills = 1.0", "frags = 1.0"), "frags"),
    (lambda s: s.replace("kills = 1.0", 'kills = "yes"'), "kills"),
    (lambda s: s.replace("crop_radius = 4", "crop_radius = 2.5"),
     "crop_radius"),
])
def test_the_run_file_door_refuses_by_name(tmp_path, mutate, needle):
    """Property: the run-file door refuses an unknown table or key, a missing
    key, a mistyped or out-of-range value, an unknown controller, a refused
    or unknown crop channel, an unknown vector field or reward counter --
    naming it. Breaks if a typo in a run file silently changes a training
    run."""
    p = _run_toml(tmp_path / "run.toml", levels_dir=tmp_path, **_OK)
    p.write_text(mutate(p.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(ValueError, match=needle):
        rl_env.load_run(p)


def test_the_run_file_hash_and_backend_door(tmp_path):
    """Property: ``run_sha256`` is the sha256 of the run file's exact bytes,
    and a backend other than ``cpu_farm`` is refused by name before any
    worker starts. Breaks if the manifest hash describes something else or
    an unbuilt backend is silently replaced by the farm."""
    import hashlib
    _box(tmp_path, "abox")
    p = _run_toml(tmp_path / "run.toml", levels_dir=tmp_path, **_OK)
    run = rl_env.load_run(p)
    assert run.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="device"):
        rl_env.BatchedEnv(p, backend="device")


def test_a_per_team_policy_list_must_match_the_level(tmp_path):
    """Property: ``policy = [..]`` names one controller per team; a list of
    another length is refused naming the level. Breaks if a team would be
    left without a driver or a name would land on the wrong team."""
    _box(tmp_path, "abox")
    p = _run_toml(tmp_path / "run.toml", levels_dir=tmp_path,
                  policy='["policy", "bots", "idle"]', **_OK)
    with pytest.raises(ValueError, match="abox"):
        rl_env.load_levels(rl_env.load_run(p))


def test_workers_load_the_cpu_build_and_no_renderer(tmp_path):
    """Property: every farm worker imported ``breach_physics`` from the CPU
    build and no renderer, raylib or CUDA module, after reset and a step, and
    runs with one BLAS thread (``WORKER_ENV``) while the driver's own
    environment is left as it was. Breaks if a worker could create a CUDA
    context or pull in a window, or commits a BLAS buffer per hardware
    thread."""
    _box(tmp_path, "abox")
    p = _run_toml(tmp_path / "run.toml", levels_dir=tmp_path, **_OK)
    before = {k: os.environ.get(k) for k in rl_env.WORKER_ENV}
    with rl_env.BatchedEnv(p, n_workers=2) as env:
        env.reset()
        env.step(_zero_actions(env))
        probes = env.probe()
    cpu = (ROOT / "cpp" / "build" / "Release").resolve()
    assert len(probes) == 2
    for pr in probes:
        assert Path(pr["breach_physics"]).resolve().parent == cpu
        assert pr["modules"] == []
        assert pr["env"] == rl_env.WORKER_ENV       # one BLAS thread each
    for k in rl_env.WORKER_ENV:                     # the driver's restored
        assert os.environ.get(k) == before.get(k)


def test_counters_credit_the_row_of_the_tick_start_team(tmp_path):
    """Property: a kill of an enemy credits the attacker's ``kills`` and the
    victim's ``deaths_<cause>``; a teammate's death is a ``team_kills``; a
    self-hit counts only as taken; an attacker that was no longer on its
    team at the tick's start (converted) and the environment are credited
    with nothing; a heal is not a hit. Breaks if credit follows the unit's
    team AFTER conversion, or friendly fire is paid as kills."""
    lv = _box(tmp_path, "abox")
    run = rl_env.load_run(_run_toml(tmp_path / "run.toml", levels_dir=tmp_path,
                                    **_OK))
    layout = rl_env.make_layout(run, [lv])
    r = rl_env.EnvRunner(0, run, layout, [lv], breach_physics=bp)
    S = layout.n_slots
    r._id_row = {1: (0, 0), 2: (0, 1), 3: (1, S), 4: (1, S + 1)}
    before = {1: 0, 2: 0, 3: 1, 4: ZOMBIE_TEAM, 9: ZOMBIE_TEAM}
    c = np.zeros((layout.n_rows, len(layout.counters)), dtype=np.float32)
    ev = [UnitHitEvent(3, 10.0, "bullet", 1), UnitKilledEvent(3, "bullet", 1),
          UnitHitEvent(2, 4.0, "bullet", 1), UnitKilledEvent(2, "bullet", 1),
          UnitHitEvent(1, 3.0, "explosion", 1),
          UnitHitEvent(1, 5.0, "melee", 4), UnitHitEvent(1, 2.0, "heat", -1),
          UnitHitEvent(3, -1.0, "regen", -1)]
    r._count(ev, before, c)
    col = {n: i for i, n in enumerate(layout.counters)}
    assert c[0, col["kills"]] == 1 and c[0, col["team_kills"]] == 1
    assert c[0, col["hits_dealt"]] == 1 and c[0, col["team_hits"]] == 1
    assert c[0, col["damage_dealt"]] == 10.0
    assert c[S, col["deaths_bullet"]] == 1 and c[1, col["deaths_bullet"]] == 1
    assert c[0, col["hits_taken"]] == 3 and c[0, col["damage_taken"]] == 10.0
    assert c[S, col["hits_taken"]] == 1      # the heal is not a hit
    assert not c[S + 1].any()                # converted attacker: nothing


def test_the_smoke_driver_runs_and_reports(tmp_path, capsys):
    """Property: ``tools/rl_smoke.py`` drives a farm through at least one
    in-step reset per env and prints the farm's measurement lines (env-
    ticks/s, step p95, RSS per worker, the run's sha256). Breaks if the
    yardstick instrument rots."""
    import rl_smoke
    spec = MatchSpec(teams=(IDLE, IDLE), seed=1, zombies=0,
                     time_limit_s=1.0)
    _box(tmp_path, "sbox", spec)
    p = _run_toml(tmp_path / "run.toml", levels=["sbox"],
                  levels_dir=tmp_path, n_envs=2, decimation=4)
    assert rl_smoke.main([str(p), "--workers", "2", "--episodes", "1",
                          "--warmup-steps", "2"]) == 0
    out = capsys.readouterr().out
    for needle in ("env-ticks/s:", "step ms: p50", "p95", "RSS per worker",
                   rl_env.load_run(p).sha256):
        assert needle in out, out
