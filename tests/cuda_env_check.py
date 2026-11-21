"""#29 P8.10a -- THE DEVICE BACKEND OF THE BATCHED ENVIRONMENT
(``rl_env.BatchedEnv(run, backend="device")``, ``src/rl_env_device.py``)
held EQUAL to ``cpu_farm``, THE reference, step by step.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_env.py). The farm's workers import the CPU build
(``rl_env._worker_main``), this process the CUDA one.

PART EQUALITY -- BACKEND EQUALITY (:data:`EQUALITY_RUNS`): the same run file
(every team ``policy``, observation and action spec v1, "fireteam", ``[run]
commander = "none"``, the replay checkpoint cadence), run seed and action
stream on both backends: ``rl_env_device.compare_backends`` resets both,
then per step shows the device's served masks raw-bit equal to the farm's,
draws seeded-random actions under them, steps both and compares EVERY
returned field as raw bits (dtype and shape included): the observation
(every key; the planes / columns no device patch has built yet --
``device_obs.BUILT``, ``tests/obs_column_owners.toml``: the order, route and
place columns of P8.6b part 1 -- named and their differing cells counted,
not gated), reward, terminated, truncated, and info's tick, episode,
episode_seed, level, ticks, ended, counters, team blocks, rows_ended,
masked_choices, final_obs, final_full_view, full_view, result and the
per-env checkpoints (digests). Each env must end ``episodes`` episodes
(auto-reset on both sides) and the envs must end at different steps; a
training room and ship_demo (zombies), N = 1, 4 and one larger N.

PART REFUSALS -- each by name: a team not ``policy``; ``[run] commander``
not "none"; a level set that cannot share one batch (a training room with
ship_demo); the "counters" reward profile; an action out of range (raised
at the step as the host's ``ValueError`` with ``check_actions``' message and
``.env``, and every later call refused).

PART SYNCS -- a steady step (no env ends, no checkpoint due) makes exactly
the conductor's two named host syncs per tick (``max_n_sub``, the books
read) and nothing else, at N = 1 and 2, counted after a warm-up with the
cycle collector run first and held off while counting (the robust form of
the count P8.8b found fragile).

PART HANDOVER -- the tensors a step returns for the observation, the full
view and the final_ sets share their pointers with the env's device buffers
and are the same tensors on every step.

PART BREAKS -- each must turn PART EQUALITY's comparison red (``DeviceEnv.brk``,
``rl_env_device.BREAKS``): ``reset_late`` (the auto-reset one step late),
``final_kept`` (the final_ rows of envs that did not end not zeroed),
``episode_env`` (the episode drawn from the wrong env index), ``decide_late``
(the decisions handed over after the step's first tick).

PART COST -- a reset's host build and restage, and a checkpoint's
download + digest, at N = 4 / 64 (ship_demo).

PROPERTY: the device backend IS the farm, field for field, step for step,
across episodes and resets, in a batch as alone. BREAKS IF a reset lands a
step late or on another env's episode, a final_ row leaks, a decision is
handed over late, a served array is read after a reset re-seeded it, or a
checkpoint is taken off the farm's ticks.
"""
from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src", _ROOT / "tools", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import rl_env  # noqa: E402
import rl_env_device as rd  # noqa: E402

FAILS: list = []
ROOM = "range_door"
_ROOMS_DIR = None
#: (label, level, N, episodes per env, run seed, action seed)
EQUALITY_RUNS = (("room N=1", "room", 1, 3, 11, 1),
                 ("room N=4", "room", 4, 3, 12, 2),
                 ("ship_demo N=1", "ship", 1, 2, 13, 3),
                 ("ship_demo N=4", "ship", 4, 2, 14, 4),
                 ("room N=8", "room", 8, 2, 15, 5))


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", str(msg).encode("ascii", "replace").decode("ascii"))


def rooms_dir() -> str:
    global _ROOMS_DIR
    import gen_training_rooms as TR
    if _ROOMS_DIR is None:
        _ROOMS_DIR = tempfile.mkdtemp(prefix="env_rooms_")
        for d in TR.drafts():
            if d.name == ROOM:
                TR.write(d, _ROOMS_DIR)
    return _ROOMS_DIR


def run_of(level: str, n: int, seed: int, **over):
    """The gate's run (parsed from a dict, as the run file door does)."""
    levels = {"room": [ROOM], "ship": ["ship_demo"], "both": [ROOM, "ship_demo"]}[level]
    ldir = rooms_dir() if level == "room" else str(_ROOT / "levels")
    if level == "both":                      # the room copied beside ship_demo
        import shutil
        ldir = tempfile.mkdtemp(prefix="env_both_")
        shutil.copytree(Path(rooms_dir()) / ROOM, Path(ldir) / ROOM)
        shutil.copytree(_ROOT / "levels" / "ship_demo", Path(ldir) / "ship_demo")
    raw = {"run": {"seed": int(seed), "n_envs": int(n), "decimation": 4, "levels": levels,
                   "policy": "policy", "levels_dir": ldir, "commander": "none"},
           "obs": {"spec": "v1", "full_view": True},
           "reward": {"profile": "fireteam"}, "action": {"spec": "v1"}}
    for k, v in over.items():
        tab, key = k.split("__")
        raw[tab][key] = v
    return rl_env.parse_run(raw, f"env check ({level})")


def pair(run):
    farm = rl_env.BatchedEnv(run, "cpu_farm", n_workers=min(run.n_envs, 8))
    try:
        dev = rl_env.BatchedEnv(run, "device")
    except BaseException:
        farm.close()
        raise
    return farm, dev


# ---------------------------------------------------------------------------
def equality_run(label, level, n, episodes, seed, aseed, brk="") -> dict:
    run = run_of(level, n, seed)
    farm, dev = pair(run)
    dev._dev.brk = brk
    t0 = time.perf_counter()
    try:
        st = rd.compare_backends(farm, dev, episodes=episodes, seed=aseed,
                                 max_steps=4000)
        cov_f, cov_d = farm.coverage(), dev.coverage()
    finally:
        farm.close()
        dev.close()
    st["wall"] = time.perf_counter() - t0
    st["coverage_equal"] = all(np.array_equal(cov_f[k], cov_d[k]) for k in cov_f)
    st["coverage"] = cov_f
    return st


def part_equality():
    tot = {"steps": 0, "endings": 0, "ticks": 0}
    unbuilt = {}
    for label, level, n, eps, seed, aseed in EQUALITY_RUNS:
        st = equality_run(label, level, n, eps, seed, aseed)
        if st["difference"] is not None:
            fail(f"{label}: {st['difference']}")
            continue
        if st["endings"].min() < eps:
            fail(f"{label}: endings per env {st['endings'].tolist()}, fewer than {eps} (vacuous)")
        steps = sorted({s for _e, s in st["end_steps"]})
        if n > 1 and len(steps) < 2:
            fail(f"{label}: every env ended at the same steps {steps} (not a batch property)")
        if not st["coverage_equal"]:
            fail(f"{label}: the coverage counters differ between the backends")
        for k, v in st["unbuilt_cells"].items():
            unbuilt[k] = unbuilt.get(k, 0) + v
        tot["steps"] += st["steps"]
        tot["endings"] += int(st["endings"].sum())
        tot["ticks"] += st["ticks"]
        print(f"  {label}: EQUAL over {st['steps']} steps ({st['ticks']} env-ticks), episodes "
              f"ended per env {st['endings'].tolist()} (auto-resets on both sides), endings "
              f"at {len(steps)} distinct steps, coverage equal ({st['wall']:.0f} s)")
    names = ", ".join(f"{k}/{c}" for k, c in sorted(unbuilt))
    print(f"  equality: {tot['steps']} steps, {tot['endings']} episodes ended, {tot['ticks']} "
          f"env-ticks; NOT YET BUILT on the device (P8.6b part 1, served as zeros, named, "
          f"not gated): {names}; cells where the farm's differ: "
          + ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in sorted(unbuilt.items()) if v))


# ---------------------------------------------------------------------------
def _refused(label, fn, exc, needle):
    try:
        fn()
    except exc as e:
        if needle not in str(e):
            fail(f"refusal {label}: raised {type(e).__name__} without {needle!r}: {e}")
            return None
        print(f"  refusal {label}: {type(e).__name__}: {str(e)[:160]}")
        return e
    except Exception as e:                    # the wrong exception
        fail(f"refusal {label}: {type(e).__name__}: {e}")
        return None
    fail(f"refusal {label}: not refused")
    return None


def part_refusals():
    R = rd.DeviceBackendRefused
    _refused("a team not policy", lambda: rl_env.BatchedEnv(
        run_of("room", 1, 1, run__policy=["policy", "bots"]), "device"), R, "'bots'")
    _refused("commander", lambda: rl_env.BatchedEnv(
        run_of("room", 1, 1, run__commander="bots"), "device"), R, "commander")
    _refused("heterogeneous levels", lambda: rl_env.BatchedEnv(
        run_of("both", 2, 1), "device"), R, "cannot share one batch")
    _refused("counters profile", lambda: rl_env.BatchedEnv(
        run_of("room", 1, 1, reward__profile="counters"), "device"), R, "counters")
    env = rl_env.BatchedEnv(run_of("room", 2, 1), "device")
    try:
        obs, _ = env.reset()
        acts = rl_env.random_legal_actions(np.random.default_rng(0),
                                           {k: rd.host(v) for k, v in obs.items()})
        acts[1, 1, 3] = 999
        e = _refused("an action out of range", lambda: env.step(acts), ValueError,
                     "action out of range: env 1, row 1, head 'hand'")
        if e is not None and getattr(e, "env", None) != 1:
            fail(f"refusal out of range: .env {getattr(e, 'env', None)}, not 1")
        _refused("a step after the fault", lambda: env.step(acts), rd.DeviceEnvFailed,
                 "earlier step refused the batch")
    finally:
        env.close()


# ---------------------------------------------------------------------------
def part_syncs():
    import cuda_conductor_check as cc
    counts = {}
    for n in (1, 2):
        env = rl_env.BatchedEnv(run_of("room", n, 31, run__checkpoint_every_ticks=0), "device")
        try:
            rng = np.random.default_rng(5)
            obs, _ = env.reset()
            for _ in range(6):                          # warm-up: first allocations
                obs = env.step(rl_env.random_legal_actions(
                    rng, {k: rd.host(v) for k, v in obs.items()}))[0]
            per = []
            for _ in range(10):
                acts = rl_env.random_legal_actions(rng, {k: rd.host(v) for k, v in obs.items()})
                acts_d = env._dev.cp.asarray(acts)     # the trainer's tensor: on the card
                gc.collect()
                gc.disable()
                sc = cc.SyncCount()
                try:
                    sc.py = []
                    bp.host_sync_log_begin()
                    out = env.step(acts_d)
                    log = list(bp.host_sync_log_end()) + sc.py
                finally:
                    sc.close()
                    gc.enable()
                obs = out[0]
                if not out[4]["ended"].any():
                    per.append(log)
            if not per:
                fail(f"syncs N={n}: every counted step ended an episode (vacuous)")
                continue
            names = sorted({x.partition(" | ")[2] or x for lg in per for x in lg})
            counts[n] = [len(lg) for lg in per]
            want = 2 * env.run.decimation
            print(f"  N={n}: host syncs per steady step {counts[n]} (decimation "
                  f"{env.run.decimation}: want {want}), named {names}")
            if any(c != want for c in counts[n]):
                fail(f"syncs N={n}: {counts[n]} syncs per steady step, want {want}")
            extra = [x for x in names if x not in ("max_n_sub", "books_read")]
            if extra:
                fail(f"syncs N={n}: a sync outside the conductor's named reads: {extra}")
        finally:
            env.close()


# ---------------------------------------------------------------------------
def part_handover():
    env = rl_env.BatchedEnv(run_of("room", 2, 41), "device")
    try:
        d = env._dev
        obs, info = env.reset()
        rng = np.random.default_rng(1)
        ptr0 = {k: v.data_ptr() for k, v in obs.items()}
        for k, v in obs.items():
            if v.data_ptr() != d.obs.buffers[k].data.ptr:
                fail(f"handover: obs[{k!r}] is not the env's buffer")
        for k, v in info["full_view"].items():
            if v.data_ptr() != d.obs.buffers[k].data.ptr:
                fail(f"handover: full_view[{k!r}] is not the env's buffer")
        for _ in range(3):
            out = env.step(rl_env.random_legal_actions(
                rng, {k: rd.host(v) for k, v in obs.items()}))
            obs, info = out[0], out[4]
            for k, v in obs.items():
                if v.data_ptr() != ptr0[k] or not v.is_cuda:
                    fail(f"handover: obs[{k!r}] moved or left the card at a step")
            for k, v in info["final_obs"].items():
                if v.data_ptr() != d._final[k].data.ptr:
                    fail(f"handover: final_obs[{k!r}] is not the env's final_ buffer")
            for name in ("reward", "terminated", "truncated"):
                t = out[{"reward": 1, "terminated": 2, "truncated": 3}[name]]
                if not t.is_cuda:
                    fail(f"handover: {name} is not on the card")
        tr = env.team_rows(obs, 1)
        S = env.layout.n_slots
        if tr["vector"].data_ptr() != obs["vector"][:, S:2 * S].data_ptr():
            fail("handover: team_rows is not a view of the env's buffer")
        print(f"  handover: {len(obs)} obs keys, {len(info['full_view'])} full-view keys and the "
              f"final_ sets are the env's device buffers (pointer-identical, every step); "
              f"reward / terminated / truncated on the card; team_rows a view")
    finally:
        env.close()


# ---------------------------------------------------------------------------
def part_breaks():
    for brk, label, level, n, eps, seed in (("reset_late", "room N=2", "room", 2, 2, 51),
                                            ("final_kept", "room N=2", "room", 2, 3, 52),
                                            ("episode_env", "room N=2", "room", 2, 1, 53),
                                            ("decide_late", "room N=1", "room", 1, 1, 54)):
        st = equality_run(label, level, n, eps, seed, 7, brk=brk)
        if st["difference"] is None:
            fail(f"break {brk}: stayed green over {st['steps']} steps")
        else:
            print(f"  break {brk}: RED at {st['difference'][:200]}")


# ---------------------------------------------------------------------------
def part_cost(n_list=(4, 64)):
    for n in n_list:
        env = rl_env.BatchedEnv(run_of("ship", n, 61), "device")
        try:
            d = env._dev
            obs, _ = env.reset()
            rng = np.random.default_rng(2)
            for _ in range(2):
                obs = env.step(rl_env.random_legal_actions(
                    rng, {k: rd.host(v) for k, v in obs.items()}))[0]
            builds, restage = [], []
            for j in range(3):
                e = (5 * j + 1) % n
                t0 = time.perf_counter()
                built = d._build(e)
                t1 = time.perf_counter()
                d.c.reset_env(e, built[3].sim, built[3].spec)
                d._take(e, built)
                t2 = time.perf_counter()
                builds.append(1e3 * (t1 - t0))
                restage.append(1e3 * (t2 - t1))
            cps = []
            for j in range(3):
                e = (3 * j) % n
                t0 = time.perf_counter()
                d.c.download([e])
                d.sessions[e].digests()
                cps.append(1e3 * (time.perf_counter() - t0))
            print(f"  cost N={n} (ship_demo): host build of an episode "
                  f"{np.mean(builds):.1f} ms, restage (reset_env) {np.mean(restage):.1f} ms, a "
                  f"checkpoint (download + digest of one env) {np.mean(cps):.1f} ms")
        finally:
            env.close()
        gc.collect()


PARTS = {"equality": part_equality, "refusals": part_refusals, "syncs": part_syncs,
         "handover": part_handover, "breaks": part_breaks, "cost": part_cost}


def main(parts=None) -> int:
    from simulation import physics_runner as pr
    pr.set_residency(False)
    t0 = time.perf_counter()
    print("ENV check at P8.10a")
    for name in (parts or list(PARTS)):
        print(f"PART {name}")
        try:
            PARTS[name]()
        except Exception as e:                 # a part that raised is a failure, named
            import traceback
            traceback.print_exc()
            fail(f"part {name} raised {type(e).__name__}: {e}")
    print(f"{time.perf_counter() - t0:.1f} s")
    print("ENV_RESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
