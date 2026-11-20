"""tools/bench_conductor.py -- what does a device conductor tick cost, per env, as N grows? (#29 P7e.5)

    C:/Users/steen/anaconda3/envs/data/python.exe tools/bench_conductor.py \\
        --levels ship_demo,range_hall --ns 1,4,16,64 \\
        --out <scratch>/conductor_profile.json

THE PROFILE of the device conductor (``simulation.conductor.DeviceConductor``,
Class T): N homogeneous envs of one level (seeds 1..N) driven by ONE recorded
match's intents (``tools/run_match.run_match(record=True)``, the level's own
match, recorded once on the host, the same intent rows latched into every
env), every slot of the reference tick on the device. The shape of
``tools/bench_resident_scaling.py --batch-sweep``: every (level, N) runs in a
FRESH subprocess (``--run``), its JSON line collected by the sweep.

Per (level, N):
- ms per tick and per env-tick: wall time of ``conductor.tick()`` (the latch
  scatter, the walk, the one books read and the deferred physics fold) over
  ``--ticks`` timed ticks after ``--warmup``. The tick ends in the books
  read's ``cudaStreamSynchronize`` on the tick stream (every launch, copy and
  memset of the tick is on that stream), so the wall clock read after it is
  GPU-synchronised: the tick's GPU work is complete when it is read.
- the share of each sub-slot: ``--profile-ticks`` more ticks with the
  conductor's profile open (``tick(profile=True)``): a CUDA event on the tick
  stream at every call line of the walk (``bp.conductor_label``), at the
  stages of slot 7's chain (``physics_runner._PROFILE_MARK``: the EOS
  substeps, the trace tail, combustion) and inside the resident tail (bulk sum
  + fire, the sweep, temperature, the swarm, the books D2H); a segment's GPU
  ms is ``cudaEventElapsedTime`` between its event and the next (the stream's
  clock: kernel time plus any time the stream sat idle waiting for the host),
  its host ms the steady clock between the two marks (the Python and launch
  cost). The books read's own segment is host time (it waits for the GPU),
  and so is the host fold after it (``books fold (host)``: the physics
  books' fold, the per-env refusal checks, the edit books' fold).
- host syncs per tick (``bp.host_sync_log_begin`` / ``_end`` around one tick,
  every counted sync and the tick sync points by name).
- device memory per env: ``cudaMemGetInfo``'s used bytes before the
  conductor is built and after the timed ticks (CuPy's pool not freed),
  divided by N.

- ``--obs`` (#29 P8.0): the device observation attached
  (``simulation.device_obs``, ``--decimation`` ticks per decision): its
  blocks are the walk's ``11' observation`` segment, and ``obs_build_ms`` is
  the GPU time of one decision's blocks for all N envs (``--obs-builds``
  launches after the timed ticks, CUDA events around them on the tick
  stream).

Levels: ``ship_demo`` and a training room (``tools/gen_training_rooms.py``
writes it into ``<--levels-dir>``, a scratch folder, the default a temp dir).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TRAINING_ROOMS = ("range_hall", "range_door", "range_corner")

#: Profile labels grouped for the table: slot 7's stages (the chain), the
#: books read, and every other call line (the unit side, per face).
SLOT7 = ("7 physics", "7 EOS substeps", "7 trace tail", "7 combustion")


def _load_level(name: str, levels_dir: str):
    import level_loader
    if name in TRAINING_ROOMS or name.split("_swapped")[0] in TRAINING_ROOMS:
        import gen_training_rooms as gtr
        p = Path(levels_dir) / name
        if not (p / "level.toml").exists():
            for d in gtr.drafts():
                if d.name == name:
                    gtr.write(d, levels_dir)
        return level_loader.load(name, str(levels_dir))
    return level_loader.load(name, str(ROOT / "levels"))


def _device_used() -> int:
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    return int(total - free)


def run_one(level_name: str, levels_dir: str, n: int, warmup: int, ticks: int,
            profile_ticks: int, obs: bool = False, decimation: int = 4,
            obs_builds: int = 50) -> dict:
    """One (level, N) in THIS process; returns the result row."""
    import numpy as np
    import breach_physics as bp
    from dataclasses import replace
    from run_match import run_match
    from simulation import physics_runner as pr
    from simulation.conductor import DeviceConductor
    from simulation.match_session import MatchSession
    pr.set_residency(False)
    level = _load_level(level_name, levels_dir)
    total = warmup + ticks + profile_ticks
    spec = replace(level.match, time_limit_s=(total + 24) / 24.0)
    t0 = time.perf_counter()
    rec = run_match(level, spec, seed=1, record=True).recording
    t_rec = time.perf_counter() - t0
    end = max(1, int(rec.end_tick))
    used0 = _device_used()
    sims = [MatchSession(level, rec.spec, seed=e + 1, breach_physics=bp).sim for e in range(n)]
    t0 = time.perf_counter()
    c = DeviceConductor(sims, bp=bp)
    t_build = time.perf_counter() - t0
    dev_obs = None
    if obs:
        import rl_env
        from simulation.device_obs import DeviceObservation
        dev_obs = DeviceObservation(c, [rec.spec] * n, max(rl_env._team_slots(level)),
                                    decimation)

    def latch():
        for e, s in enumerate(sims):
            c.latch(rec.intents.get(int(s.tick) % end, ()), env=e)

    for _ in range(warmup):
        latch()
        c.tick()
    wall = []
    syncs = []
    for _ in range(ticks):
        latch()
        bp.host_sync_log_begin()
        ts = time.perf_counter()
        c.tick()
        wall.append(time.perf_counter() - ts)
        syncs.append(list(bp.host_sync_log_end()))
    used1 = _device_used()
    obs_ms = None
    if dev_obs is not None:
        import cupy as cp
        st = cp.cuda.ExternalStream(int(bp.tick_stream_ptr()))
        a, b = cp.cuda.Event(), cp.cuda.Event()
        dev_obs.build()                             # warm
        a.record(st)
        for _ in range(obs_builds):
            dev_obs.build()
        b.record(st)
        b.synchronize()
        obs_ms = cp.cuda.get_elapsed_time(a, b) / obs_builds
    seg_gpu: dict = {}
    seg_host: dict = {}
    order: list = []
    for _ in range(profile_ticks):
        latch()
        c.tick(profile=True)
        for label, g, h in c.last_profile:
            if label not in seg_host:
                order.append(label)
                seg_gpu[label] = []
                seg_host[label] = []
            seg_gpu[label].append(0.0 if g is None else float(g))
            seg_host[label].append(float(h))
    ms = 1000.0 * float(np.mean(wall))
    names = sorted({x.partition(" | ")[2] for lg in syncs for x in lg})
    return {
        "level": level_name, "n": n, "h": int(c.h), "w": int(c.w), "units": int(c.U),
        "ticks": ticks, "warmup": warmup, "record_s": round(t_rec, 1),
        "build_s": round(t_build, 2),
        "ms_per_tick": round(ms, 3), "ms_per_tick_median": round(1000 * float(np.median(wall)), 3),
        "ms_per_env_tick": round(ms / n, 3),
        "syncs_per_tick": round(float(np.mean([len(s) for s in syncs])), 2),
        "sync_names": names,
        "mem_used_mb": round((used1 - used0) / 2 ** 20, 1),
        "mem_per_env_mb": round((used1 - used0) / 2 ** 20 / n, 2),
        "obs_build_ms": None if obs_ms is None else round(float(obs_ms), 4),
        "segments": [{"label": k, "gpu_ms": round(float(np.mean(seg_gpu[k])), 4),
                      "host_ms": round(float(np.mean(seg_host[k])), 4)} for k in order],
    }


def _group(seg) -> str:
    lab = seg["label"]
    if lab == "books read":
        return "books read"
    if lab == "books fold (host)":
        return "fold"
    if lab in SLOT7 or lab.startswith("7 tail"):
        return "slot 7"
    return "unit side"


def format_rows(rows) -> str:
    out = []
    out.append(f"{'level':<12} {'N':>3} {'ms/tick':>8} {'ms/env':>7} {'syncs':>5} "
               f"{'MB/env':>7} {'GPU unit':>8} {'GPU s7':>7} {'host unit':>9} {'host s7':>7} "
               f"{'read':>6} {'fold':>6}  top GPU sub-slots")
    for r in rows:
        if "error" in r:
            out.append(f"{r['level']:<12} {r['n']:>3}  {r['error']}")
            continue
        g = {"unit side": 0.0, "slot 7": 0.0, "books read": 0.0, "fold": 0.0}
        hh = dict(g)
        for s in r["segments"]:
            g[_group(s)] += s["gpu_ms"]
            hh[_group(s)] += s["host_ms"]
        tot = sum(g.values()) or 1.0
        top = sorted(r["segments"], key=lambda s: -s["gpu_ms"])[:4]
        tops = ", ".join(f"{s['label']} {100 * s['gpu_ms'] / tot:.0f}%" for s in top)
        out.append(f"{r['level']:<12} {r['n']:>3} {r['ms_per_tick']:>8.2f} "
                   f"{r['ms_per_env_tick']:>7.3f} {r['syncs_per_tick']:>5.1f} "
                   f"{r['mem_per_env_mb']:>7.1f} {g['unit side']:>8.2f} {g['slot 7']:>7.2f} "
                   f"{hh['unit side']:>9.2f} {hh['slot 7']:>7.2f} {hh['books read']:>6.2f} "
                   f"{hh['fold']:>6.2f}  {tops}"
                   + ("" if r.get("obs_build_ms") is None
                      else f"  | obs blocks {r['obs_build_ms']:.3f} ms per decision"))
    return "\n".join(out)


def sweep(args) -> int:
    rows = []
    for level in args.levels.split(","):
        for n in [int(x) for x in args.ns.split(",")]:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run", "--level", level,
                   "--n", str(n), "--levels-dir", args.levels_dir, "--warmup", str(args.warmup),
                   "--ticks", str(args.ticks), "--profile-ticks", str(args.profile_ticks),
                   "--build", args.build, "--decimation", str(args.decimation)]
            if args.obs:
                cmd.append("--obs")
            t0 = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT),
                               env=dict(os.environ, PYTHONPATH=os.environ.get("PYTHONPATH", "")))
            row = None
            for line in p.stdout.splitlines():
                if line.startswith("CONDUCTOR_BENCH_JSON "):
                    row = json.loads(line[len("CONDUCTOR_BENCH_JSON "):])
            if row is None:
                tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
                row = {"level": level, "n": n, "error": " | ".join(tail)[-400:]}
            print(f"[{level} N={n}] {time.perf_counter() - t0:.0f} s", flush=True)
            rows.append(row)
    print(format_rows(rows))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench_conductor")
    ap.add_argument("--levels", default="ship_demo,range_hall")
    ap.add_argument("--ns", default="1,4,16,64")
    ap.add_argument("--levels-dir", default=str(Path(tempfile.gettempdir()) / "bench_conductor_levels"))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--ticks", type=int, default=60)
    ap.add_argument("--profile-ticks", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--table", default=None, metavar="JSON",
                    help="print the table of a sweep's --out file and exit")
    ap.add_argument("--run", action="store_true", help="one (level, N) in this process")
    ap.add_argument("--level", default="ship_demo")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--obs", action="store_true",
                    help="attach the device observation (#29 P8.0) and time its blocks")
    ap.add_argument("--decimation", type=int, default=4,
                    help="the device observation's ticks per decision (with --obs)")
    ap.add_argument("--build", default=str(ROOT / "cpp" / "build_cuda"),
                    help="the CUDA build of breach_physics (every backend starts OFF: the "
                         "recording runs the CPU path, the conductor brackets slot 7's)")
    args = ap.parse_args(argv)
    if args.table:
        print(format_rows(json.loads(Path(args.table).read_text())))
        return 0
    Path(args.levels_dir).mkdir(parents=True, exist_ok=True)
    if args.run:
        from bench_bot_matches import import_build
        import_build(Path(args.build))
        try:
            row = run_one(args.level, args.levels_dir, args.n, args.warmup, args.ticks,
                          args.profile_ticks, obs=args.obs, decimation=args.decimation)
        except Exception as ex:   # an out-of-memory N is a row of the table, by name
            row = {"level": args.level, "n": args.n, "error": f"{type(ex).__name__}: {ex}"[:400]}
        print("CONDUCTOR_BENCH_JSON " + json.dumps(row), flush=True)
        return 0
    return sweep(args)


if __name__ == "__main__":
    sys.exit(main())
