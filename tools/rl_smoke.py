"""tools/rl_smoke.py -- the batched environment's random-action driver (#29
P1a) and THE farm measurement instrument (residency design §9.1).

Steps ``rl_env.BatchedEnv`` with uniformly random action rows
(``control_policy.random_action_rows``, a seeded generator: the trainer stub;
under ``[action] spec = "v1"`` (#195 1h) each head drawn uniformly among the
options the served masks leave legal, the argument among ``arg_mask``'s of
the drawn hand and aim -- :func:`legal_v1_rows`) until every env has finished ``--episodes`` episodes through the in-step
reset, then reports:

- env-ticks/s: ticks simulated by all envs / wall time, over the window after
  the first ``--warmup-steps`` steps (the workers' start, imports and first
  episode setup excluded), the driver's own action generation INCLUDED;
- step ms: p50 / p95 / max of ``BatchedEnv.step`` wall time in that window;
- per worker at the end (psutil): RSS (on Windows the working set), the
  peak working set and the private bytes (the commit charge); episodes ended;
- the run file's sha256 and every knob, so a later reading repeats it.

    C:/Users/steen/anaconda3/envs/data/python.exe tools/rl_smoke.py RUN.toml \\
        --workers 8 --episodes 3

``--backend device`` (#29 P8.10a): the same driver over
``BatchedEnv(run, "device")`` in this process (the CUDA build imported
through ``tests/cuda_harness``'s paths), the actions drawn ON THE CARD by
torch from the served masks (:func:`legal_v1_rows_torch`: no host round trip
per step), and the step's host time split by part (``DeviceEnv.timing``:
conductor ticks, decisions, served copies, resets, checkpoints, coverage,
the rest the env's Python) plus, over the last ``--profile-steps`` steps,
the conductor ticks' GPU time by call line grouped (the sim, the decoder's
``0 decode``, the observation + full view + books ``11' observation``).

Every later farm reading uses this driver with the same run files (their
exact text is in ``docs/residency_benchmark_2026-11-07.md``).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import rl_env  # noqa: E402
from control_policy import random_action_rows  # noqa: E402


#: The v1 trainer stub (the env's one copy, ``rl_env.random_legal_actions``).
legal_v1_rows = rl_env.random_legal_actions


def legal_v1_rows_torch(gen, obs):
    """The same draw ON THE CARD (#29 P8.10a, the device backend): per row
    each head uniform among the served masks' legal options, the argument
    among ``arg_mask``'s (its law restated in torch from action_spec_v1's
    constants), ``(N, U, 5)`` int32 on the masks' device; no sync."""
    import torch
    from simulation import action_spec_v1 as A

    def pick(mask):
        r = torch.rand(mask.shape, generator=gen, device=mask.device)
        return torch.where(mask != 0, r, torch.full_like(r, -1.0)).argmax(dim=-1)
    cols = [pick(obs[k]) for k in A.MASK_KEYS]
    hand, aim = cols[3], cols[1]
    throw = hand >= A.HAND_THROW0
    four = throw | (hand == A.HAND_PLANT)
    am = torch.zeros(hand.shape + (A.N_ARG,), dtype=torch.uint8, device=hand.device)
    am[..., :A.N_FRACTIONS] = four[..., None].to(torch.uint8)
    am[..., A.ARG_ONTO] = (throw & (aim >= A.AIM_TOKENS)).to(torch.uint8)
    am[..., 0] |= (~four).to(torch.uint8)
    cols.append(pick(am))
    return torch.stack(cols, dim=-1).to(torch.int32)


def drive(env, *, episodes: int, seed: int, warmup_steps: int,
          max_steps: int | None = None, out=print, profile_steps: int = 0) -> dict:
    """Step ``env`` with random rows until every env ended ``episodes``
    episodes (or ``max_steps``); returns the measurement."""
    rng = np.random.default_rng(seed)
    n, u, a = env.layout.n_envs, env.layout.n_rows, env.layout.n_actions
    v1 = env.layout.action_spec == "v1"
    dev = env.backend == "device"
    gen = None
    if dev:
        import torch
        gen = torch.Generator(device="cuda")
        gen.manual_seed(int(seed))
    obs, _ = env.reset()
    t_dev0 = None
    ended = np.zeros(n, dtype=np.int64)
    step_s, ticks_win = [], 0
    t_win = None
    steps = 0
    while ended.min() < episodes:
        if max_steps is not None and steps >= max_steps:
            break
        if steps == warmup_steps:
            t_win = time.perf_counter()
            if dev:
                t_dev0 = dict(env._dev.timing)
        if dev and profile_steps and max_steps is not None \
                and steps == max_steps - profile_steps:
            env._dev.profile = True
        if dev:
            acts = legal_v1_rows_torch(gen, obs)
        elif v1:
            acts = legal_v1_rows(rng, obs)
        else:
            acts = random_action_rows(rng, n * u).reshape(n, u, a)
        t0 = time.perf_counter()
        obs, rew, _, _, info = env.step(acts)
        t1 = time.perf_counter()
        # #195 1i: a named profile's reward is (N, U, 5); every reward finite
        # (the device's checked once at the end: a per-step read is a sync)
        if (tuple(rew.shape) != (n, u) + env.layout.reward_shape_tail
                or (not dev and not np.isfinite(rew).all())):
            raise AssertionError(f"reward {rew.shape} (finite: "
                                 f"{bool(np.isfinite(rew).all())}) does not "
                                 f"fit the layout's profile "
                                 f"{env.layout.reward_profile!r}")
        steps += 1
        ended += info["ended"]
        if t_win is not None and steps > warmup_steps:
            step_s.append(t1 - t0)
            ticks_win += int(info["ticks"].sum())
    if dev:
        import torch
        torch.cuda.synchronize()
        if not bool(torch.isfinite(rew).all()):
            raise AssertionError("a device reward is not finite")
    wall = (time.perf_counter() - t_win) if t_win is not None else 0.0
    parts, gpu = {}, {}
    if dev and t_dev0 is not None:
        parts = {k: v - t_dev0.get(k, 0.0) for k, v in env._dev.timing.items()}
        gpu = dict(env._dev.prof)
    ms = np.asarray(step_s) * 1e3
    rss = []
    try:
        import psutil
        for pid in env.worker_pids():
            mi = psutil.Process(pid).memory_info()
            rss.append((pid, mi.rss, getattr(mi, "peak_wset", mi.rss),
                        getattr(mi, "private", mi.rss)))
    except Exception as e:      # psutil missing or a worker gone
        out(f"(no RSS: {e})")
    return {
        "steps": steps, "measured_steps": len(step_s),
        "env_ticks": ticks_win, "wall_s": wall,
        "env_ticks_per_s": ticks_win / wall if wall > 0 else 0.0,
        "step_ms_p50": float(np.percentile(ms, 50)) if ms.size else 0.0,
        "step_ms_p95": float(np.percentile(ms, 95)) if ms.size else 0.0,
        "step_ms_max": float(ms.max()) if ms.size else 0.0,
        "episodes_ended": ended.tolist(),
        "rss": rss, "parts_s": parts, "gpu_ms": gpu,
        "env_steps": len(step_s) * n,
        "env_steps_per_s": len(step_s) * n / wall if wall > 0 else 0.0,
        "decisions_per_s": len(step_s) * n * u / wall if wall > 0 else 0.0,
    }


def report_lines(env, res: dict, args) -> list:
    r = env.run
    mb = [x[1] / 2 ** 20 for x in res["rss"]]
    pk = [x[2] / 2 ** 20 for x in res["rss"]]
    pv = [x[3] / 2 ** 20 for x in res["rss"]]
    lines = [
        f"backend: {env.backend}",
        f"run: {r.path or '(in memory)'} sha256={env.run_sha256}",
        f"levels={list(r.levels)} n_envs={r.n_envs} "
        f"decimation={r.decimation} workers={env.n_workers} "
        f"policy={r.policy!r} channels={len(r.channels)} "
        f"crop_radius={r.crop_radius} vector={len(r.vector)} "
        f"rows={env.layout.n_rows} actions={env.layout.n_actions} "
        f"obs={r.spec} action_spec={r.action_spec} "
        f"full_view={env.layout.full_view} "
        f"reward_profile={r.reward_profile}",
        f"driver: seed={args.seed} episodes>={args.episodes} per env, "
        f"warmup={args.warmup_steps} steps",
        f"steps: {res['steps']} ({res['measured_steps']} measured), "
        f"episodes ended per env: {res['episodes_ended']}",
        f"env-ticks/s: {res['env_ticks_per_s']:.1f} "
        f"({res['env_ticks']} ticks in {res['wall_s']:.1f} s)",
        f"step ms: p50 {res['step_ms_p50']:.1f} p95 {res['step_ms_p95']:.1f} "
        f"max {res['step_ms_max']:.1f}",
        f"env-steps/s: {res['env_steps_per_s']:.1f}; soldier-decisions/s (every row "
        f"of every env-step): {res['decisions_per_s']:.1f}",
        ("RSS per worker MB: "
         + " ".join(f"{m:.0f}" for m in mb)
         + (f" (mean {np.mean(mb):.0f}); peak "
            + " ".join(f"{m:.0f}" for m in pk)
            + "; private " + " ".join(f"{m:.0f}" for m in pv)
            if mb else "")),
    ]
    if res.get("parts_s"):
        ms = res["measured_steps"] or 1
        lines.append("host ms per step by part: " + ", ".join(
            f"{k} {1e3 * v / ms:.2f}" for k, v in sorted(res["parts_s"].items())))
    if res.get("gpu_ms"):
        g = res["gpu_ms"]
        dec = g.get("0 decode", 0.0)
        obs = g.get("11' observation", 0.0)
        sim = sum(v for k, v in g.items() if k not in ("0 decode", "11' observation"))
        k = args.profile_steps or 1
        lines.append(f"GPU ms per step (last {k} steps, profiled): sim {sim / k:.2f}, "
                     f"observation + full view + books {obs / k:.2f}, decoder {dec / k:.2f}")
    return lines


def _parse(argv):
    ap = argparse.ArgumentParser(
        prog="rl_smoke", description="Random-action farm driver (#29 P1a).")
    ap.add_argument("run", help="the run file (run.toml, outside the repo)")
    ap.add_argument("--workers", type=int, default=None,
                    help="farm worker processes (default min(n_envs, "
                         "cpu_count - 1))")
    ap.add_argument("--episodes", type=int, default=3,
                    help="episodes every env finishes (default 3)")
    ap.add_argument("--seed", type=int, default=0,
                    help="the random-action generator's seed (default 0)")
    ap.add_argument("--warmup-steps", type=int, default=20,
                    help="steps excluded from the timing (default 20)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after this many steps (default: none)")
    ap.add_argument("--backend", default="cpu_farm", choices=("cpu_farm", "device"),
                    help="the env's backend (#29 P8.10a; default cpu_farm)")
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="device: profile the conductor's GPU time over the last "
                         "this many of --max-steps steps (default 0)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    kw = {}
    if args.backend == "device":
        import cuda_harness             # the CUDA build in THIS process
        cuda_harness.register_cuda_runtime()
        sys.path.insert(0, str(cuda_harness.CUDA_BUILD_DIR))
    else:
        kw["n_workers"] = args.workers
    with rl_env.BatchedEnv(args.run, args.backend, **kw) as env:
        res = drive(env, episodes=args.episodes, seed=args.seed,
                    warmup_steps=args.warmup_steps, max_steps=args.max_steps,
                    profile_steps=args.profile_steps)
        for line in report_lines(env, res, args):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
