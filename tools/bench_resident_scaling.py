"""tools/bench_resident_scaling.py -- how does one tick's cost scale with cells, per backend? (#29)

    C:/Users/steen/anaconda3/envs/data/python.exe tools/bench_resident_scaling.py \\
        --levels-dir <scratch>/levels --src ship_s1_small \\
        --build C:/Users/steen/projects/breach/cpp/build_cuda \\
        --sweep --ns 1,2,4,8,16,32,64 --backends cpu,cuda,resident \\
        --out <scratch>/scaling_small.json

The STACKED-GRID probe of the residency + batching arc: a level that is N
vertical copies of one generated ship (``--src``, written by
``tools/gen_ship.py``) in ONE grid of shape ``(N*h, w)`` -- the ship's own
space margin separates the copies -- stepped physics-only (no match, no
units, no controllers: ``Simulation`` under ``ContinuousRealtime``), so the
cost of a tick can be read against the number of cells on each backend.

It is a COST proxy only: the copies share one multigrid hierarchy, one
radiation sweep and one EOS substep count, so the physics is not N
independent worlds; what it gives is the per-cell cost and where it bends.

Modes:
    --make N         write ``<src>_x<N>`` into ``--levels-dir`` through
                     ``level_lib`` (tilemap, boundary, entities per copy with
                     ids suffixed ``_c<k>`` and rows shifted; ``air_init.npy``
                     / ``temp_init.npy`` tiled when the source has them; the
                     diffuse picture tiled -- the loader requires one)
    --run            one (level, backend) in THIS process: ``--warmup`` ticks,
                     ``--ticks`` plain ticks (the timers in, no device sync:
                     the truth), then ``--ticks`` more with a device sync
                     after each device stage (the attribution,
                     ``tools/tick_stage_profile.py``); prints SCALING_JSON
    --run --nvtx     the same under ``nsys profile --capture-range=
                     cudaProfilerApi``: NVTX ranges per stage, the capture
                     window over the plain ticks only (read it with
                     ``tools/nsys_stage_bus.py``)
    --batch-run      #29 P5: N independent envs of ``--level`` (seeds 1..N, no
                     units, entities on) in ONE ``ResidentBatch`` -- the
                     batched resident engine, every env its own grid: the
                     whole tick, the slot-7 call and the host slots per env
    --batch-sweep    ``--batch-run`` for every N in ``--ns``, fresh processes
    --sweep          for every backend and N: make the level if missing, then
                     ``--run`` it in a FRESH subprocess (a second CUDA run in
                     one process is slower, #125); collects the JSON lines
                     into ``--out`` and prints the table

Backends, all from ONE module (``--build``): ``cpu`` = every backend off,
``cuda`` = every live backend on (``tools/run_on_cuda.py``'s list, the
per-call path), ``resident`` = those plus ``physics_runner.set_residency``.
Device memory (``resident`` / ``cuda``): ``cudaMemGetInfo``'s used bytes
before the ``Simulation`` is built and after the timed ticks, and CuPy's pool.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BACKENDS = ("cpu", "cuda", "resident")


# ---------------------------------------------------------------------------
# the stacked level
# ---------------------------------------------------------------------------
def stacked_name(src: str, n: int) -> str:
    return f"{src}_x{n}"


def stack_entities(entities, n: int, h: int):
    """``n`` copies of ``entities``: copy k > 0 has every id suffixed
    ``_c<k>``, every string field naming another entity renamed with it (a
    vent's ``duct``), and its ``y`` moved down ``k*h`` rows; ordinals in file
    order. Copy 0 is the source unchanged."""
    ids = {e.id for e in entities}
    out = []
    for k in range(n):
        suf = "" if k == 0 else f"_c{k}"
        for e in entities:
            fields = dict(e.fields)
            for key, v in fields.items():
                if isinstance(v, str) and v in ids:
                    fields[key] = v + suf
            if "y" in fields:
                fields["y"] = fields["y"] + k * h
            out.append(dataclasses.replace(e, id=e.id + suf,
                                           ordinal=len(out), fields=fields))
    return out


def make_stacked(src: str, levels_dir, n: int) -> Path:
    """Write ``<src>_x<n>`` (``n`` vertical copies of ``src``) into
    ``levels_dir`` through ``level_lib``; returns the folder."""
    import numpy as np
    from PIL import Image
    import level_lib
    import level_loader
    lvl = level_loader.load(src, str(levels_dir))
    if lvl.wires:
        raise SystemExit(f"{src} has wires; stacking them is not built")
    h, w = lvl.tilemap.shape
    out = Path(levels_dir) / stacked_name(src, n)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    toml = level_lib.write_level_header(
        out, name=stacked_name(src, n), tile_size_m=lvl.tile_size_m,
        comment_lines=(f"STACKED {n} x {src} ({w}x{h}) -> {w}x{n * h}: the #29 "
                       f"scaling probe, written by",
                       "tools/bench_resident_scaling.py --make through "
                       "level_lib. A cost proxy, not a level."))
    level_lib.write_boundary_field(toml, lvl.boundary)
    level_lib.write_tilemap_csv(out, np.tile(lvl.tilemap, (n, 1)),
                                csv_bak=False)
    pic = np.asarray(Image.open(lvl.diffuse_path))
    Image.fromarray(np.concatenate([pic] * n, axis=0)).save(out / "diffuse.png")
    if lvl.entities:
        ents = stack_entities(lvl.entities, n, h)
        level_lib.write_managed_blocks(
            toml, {"entity": lambda nl: level_lib.format_entity_lines(ents, nl)},
            write_bak=False)
    if lvl.air_init_q is not None:
        level_lib.write_air_init_npy(out, np.tile(lvl.air_init_q, (n, 1)),
                                     npy_bak=False)
    if lvl.temp_init_q is not None:
        level_lib.write_temp_init_npy(out, np.tile(lvl.temp_init_q, (n, 1)),
                                      npy_bak=False)
    return out


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def _device_used():
    import cupy
    free, total = cupy.cuda.runtime.memGetInfo()
    return int(total - free)


def run_one(level_name: str, levels_dir, backend: str, build: Path,
            warmup: int, ticks: int, nvtx: bool = False) -> dict:
    """One (level, backend) in this process. Returns the SCALING_JSON dict."""
    from bench_bot_matches import import_build, setup_cuda
    if backend == "cpu":
        import_build(build)
    else:
        setup_cuda(build, backend == "resident")
    import breach_physics as bp
    import level_loader
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    from tick_stage_profile import TickStageProfiler
    import match_replay
    device = backend != "cpu"
    lvl = level_loader.load(level_name, str(levels_dir))
    used0 = _device_used() if device else None
    prof = TickStageProfiler(skip=warmup, sync=False, nvtx=nvtx and device,
                             nvtx_ticks=ticks)
    prof.install(bp, device=device)
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    plain = []
    for i in range(warmup + ticks):
        t0 = time.perf_counter()
        sim.step()
        if i >= warmup:
            plain.append(time.perf_counter() - t0)
    s_plain = prof.summary()
    # The attribution: the same number of ticks again, each device stage
    # synchronised (a no-op on the cpu backend, which never imports CuPy).
    prof.reset()
    prof.skip = 0
    prof.set_sync(True)
    for _ in range(ticks):
        sim.step()
    s_sync = prof.summary()
    prof.uninstall()
    used1 = _device_used() if device else None
    pool = None
    if device:
        import cupy
        pool = int(cupy.get_default_memory_pool().total_bytes())
    _fd, td = match_replay.match_digests(sim)
    h, w = sim.gmap.material.shape
    import numpy as np
    a = np.asarray(plain) * 1e3
    return {"level": level_name, "backend": backend, "h": int(h), "w": int(w),
            "cells": int(h * w), "warmup": warmup, "ticks": ticks,
            "tick_ms": {"mean": float(a.mean()), "p50": float(np.median(a)),
                        "p95": float(np.percentile(a, 95)),
                        "max": float(a.max())},
            "us_per_cell": float(a.mean() * 1e3 / (h * w)),
            "stages_plain": s_plain["stages"], "stages_sync": s_sync["stages"],
            "bus": s_sync["bus"],
            "device_used_before": used0, "device_used_after": used1,
            "cupy_pool_bytes": pool, "tick_digest": td,
            "final_tick": int(sim.tick)}


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
STAGE_COLS = (
    ("H2D", "  7 H2D GameMap.from_host"),
    ("EOS", None),        # run_substeps_resident or run_substeps
    ("trace", "  7 trace tail (resident)"),
    ("D2H", "  7 D2H GameMap.to_host"),
    ("comb", "  7 combustion (host)"),
    ("tail", "  7 step_tail (ripple+fire+temp+sweep)"),
    ("rest7", "  (rest of 7: host pre-physics)"),
    ("6 stamp", "6 stamp_units"),
    ("9b", "9b over-pressure walls"),
    ("other", None),      # step - slot 7 - stamp - 9b
)


def stage_row(r: dict) -> dict:
    """The sweep table's per-stage split (ms per tick, synced)."""
    st = r["stages_sync"]

    def m(label):
        return st.get(label, {}).get("mean", 0.0)
    row = {}
    for col, label in STAGE_COLS:
        if col == "EOS":
            row[col] = (m("  7 EOS run_substeps_resident")
                        + m("  7 EOS run_substeps (cpu/per-call)"))
        elif col == "other":
            row[col] = m("step") - m("7 physics") - m("6 stamp_units") \
                - m("9b over-pressure walls")
        else:
            row[col] = m(label)
    return row


def format_table(results) -> str:
    cols = [c for c, _ in STAGE_COLS]
    head = (f"{'backend':9s} {'N':>4s} {'cells':>8s} {'ms/tick':>9s} "
            f"{'us/cell':>8s} {'dev MiB':>8s} " + " ".join(f"{c:>7s}" for c in cols)
            + "  digest")
    lines = [head]
    for r in results:
        dev = ""
        if r.get("device_used_after") is not None:
            dev = f"{(r['device_used_after'] - r['device_used_before']) / 2**20:8.0f}"
        row = stage_row(r)
        lines.append(f"{r['backend']:9s} {r['n']:4d} {r['cells']:8d} "
                     f"{r['tick_ms']['mean']:9.2f} {r['us_per_cell']:8.3f} "
                     f"{dev:>8s} " + " ".join(f"{row[c]:7.2f}" for c in cols)
                     + f"  {r['tick_digest'][:12]}")
    return "\n".join(lines)


def sweep(args) -> int:
    ns = [int(x) for x in args.ns.split(",")]
    backends = [b for b in args.backends.split(",") if b]
    for b in backends:
        if b not in BACKENDS:
            raise SystemExit(f"unknown backend {b!r}; one of {BACKENDS}")
    results = []
    out = Path(args.out) if args.out else None
    t_all = time.perf_counter()
    for n in ns:
        name = stacked_name(args.src, n)
        if not (Path(args.levels_dir) / name / "level.toml").is_file():
            make_stacked(args.src, args.levels_dir, n)
        for b in backends:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--run",
                   "--level", name, "--levels-dir", str(args.levels_dir),
                   "--backend", b, "--build", str(args.build),
                   "--warmup", str(args.warmup), "--ticks", str(args.ticks)]
            t0 = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True)
            wall = time.perf_counter() - t0
            line = next((ln for ln in p.stdout.splitlines()
                         if ln.startswith("SCALING_JSON ")), None)
            if p.returncode != 0 or line is None:
                print(f"[sweep] {b} N={n} FAILED (exit {p.returncode}):\n"
                      f"{p.stdout[-2000:]}\n{p.stderr[-3000:]}", flush=True)
                results.append({"backend": b, "n": n, "failed": True,
                                "stderr": p.stderr[-3000:]})
                continue
            r = json.loads(line[len("SCALING_JSON "):])
            r["n"] = n
            r["process_wall_s"] = wall
            results.append(r)
            print(f"[sweep] {b:9s} N={n:4d} cells={r['cells']:8d} "
                  f"{r['tick_ms']['mean']:9.2f} ms/tick "
                  f"{r['us_per_cell']:7.3f} us/cell  ({wall:.0f} s)", flush=True)
            if out is not None:
                out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    ok = [r for r in results if not r.get("failed")]
    print(format_table(ok))
    print(f"sweep wall: {time.perf_counter() - t_all:.0f} s")
    return 0


# ---------------------------------------------------------------------------
# #29 P5: THE BATCHED RESIDENT ENGINE (simulation/resident_batch.py) -- N
# independent copies of one level stepped as ONE batch (not stacked in one
# grid): every env its own (h, w) grid, its own multigrid, sweep and n_sub.
# The scene G1 measures (design §3): NO units, the level's entities ON (vents,
# ducts, doors, charges, heat sources), ContinuousRealtime.
# ---------------------------------------------------------------------------
def run_batch(level_name: str, levels_dir, n: int, build: Path,
              warmup: int, ticks: int) -> dict:
    """``n`` envs of ``level_name`` (seeds 1..n) in one ResidentBatch, in THIS
    process. Times the whole batched tick, the slot-7 call (the device chain
    plus every env's entry / exit transfers: ``run_resident_batch``) and the
    rest (every env's host slots, 1..6b and 9..10). Returns BATCH_JSON."""
    from bench_bot_matches import setup_cuda
    setup_cuda(build, True)
    import breach_physics as bp
    import level_loader
    import numpy as np
    from simulation import Simulation
    from simulation import physics_runner as pr
    from simulation.resident_batch import ResidentBatch
    from simulation.ruleset import ContinuousRealtime
    lvl = level_loader.load(level_name, str(levels_dir))
    used0 = _device_used()
    sims = []
    for e in range(n):
        s = Simulation(lvl, seed=1 + e, breach_physics=bp, enable_recorder=False,
                       ruleset=ContinuousRealtime())
        s.physics_runner.headless = True
        s.set_paused(False)
        sims.append(s)
    batch = ResidentBatch(sims)
    slot7 = []
    real = pr.run_resident_batch

    def timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return real(*a, **k)
        finally:
            slot7.append(time.perf_counter() - t0)
    pr.run_resident_batch = timed
    nsub0 = int(bp.eos_nsub_reads())
    totals = []
    try:
        for i in range(warmup + ticks):
            t0 = time.perf_counter()
            batch.step()
            if i >= warmup:
                totals.append(time.perf_counter() - t0)
    finally:
        pr.run_resident_batch = real
    used1 = _device_used()
    tot = np.asarray(totals) * 1e3
    s7 = np.asarray(slot7[warmup:]) * 1e3
    host = tot - s7
    h, w = sims[0].gmap.material.shape
    cells = int(h * w)
    mean = float(tot.mean())
    return {"level": level_name, "n": n, "h": int(h), "w": int(w), "cells": cells,
            "warmup": warmup, "ticks": ticks,
            "tick_ms": {"mean": mean, "p50": float(np.median(tot)),
                        "p95": float(np.percentile(tot, 95))},
            "slot7_ms": float(s7.mean()),
            "host_ms_per_env": float(host.mean() / n),
            "env_ticks_per_s": float(n * 1e3 / mean),
            "slot7_ns_per_cell": float(s7.mean() * 1e6 / (n * cells)),
            "tick_ns_per_cell": float(mean * 1e6 / (n * cells)),
            "nsub_reads": int(bp.eos_nsub_reads()) - nsub0,
            "n_sub_last": [int(s.physics_runner.eos.dbg_last_n_sub) for s in sims[:8]],
            "device_used_before": used0, "device_used_after": used1,
            "entities": int(sum(1 for _ in getattr(sims[0], "entities", []) or []))}


def batch_sweep(args) -> int:
    ns = [int(x) for x in args.ns.split(",")]
    results = []
    out = Path(args.out) if args.out else None
    for n in ns:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--batch-run",
               "--level", args.level, "--levels-dir", str(args.levels_dir),
               "--n", str(n), "--build", str(args.build),
               "--warmup", str(args.warmup), "--ticks", str(args.ticks)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        line = next((ln for ln in p.stdout.splitlines()
                     if ln.startswith("BATCH_JSON ")), None)
        if p.returncode != 0 or line is None:
            print(f"[batch] N={n} FAILED (exit {p.returncode}):\n{p.stdout[-2000:]}\n"
                  f"{p.stderr[-3000:]}", flush=True)
            results.append({"n": n, "failed": True})
            continue
        r = json.loads(line[len("BATCH_JSON "):])
        r["process_wall_s"] = wall
        results.append(r)
        print(f"[batch] {args.level} N={n:3d}: {r['tick_ms']['mean']:8.2f} ms/tick "
              f"(slot 7 {r['slot7_ms']:7.2f}, host {r['host_ms_per_env']:.3f} ms/env) "
              f"{r['env_ticks_per_s']:8.0f} env-ticks/s, slot 7 "
              f"{r['slot7_ns_per_cell']:6.1f} ns/cell, n_sub {r['n_sub_last']}  "
              f"({wall:.0f} s)", flush=True)
        if out is not None:
            out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench_resident_scaling")
    ap.add_argument("--levels-dir", required=True)
    ap.add_argument("--src", default="ship_s1_small",
                    help="the generated ship to stack")
    ap.add_argument("--build", default=str(ROOT / "cpp" / "build_cuda"),
                    help="the breach_physics build (a CUDA build serves all "
                         "three backends)")
    ap.add_argument("--make", type=int, default=None, metavar="N")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--batch-run", action="store_true",
                    help="#29 P5: N envs of --level in ONE ResidentBatch, this "
                         "process; prints BATCH_JSON")
    ap.add_argument("--batch-sweep", action="store_true",
                    help="#29 P5: --batch-run for every N in --ns, each in a "
                         "fresh subprocess")
    ap.add_argument("--n", type=int, default=1, help="with --batch-run")
    ap.add_argument("--level", default=None, help="with --run")
    ap.add_argument("--backend", default="resident", choices=BACKENDS)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=10)
    ap.add_argument("--ns", default="1,2,4,8,16,32,64")
    ap.add_argument("--backends", default="cpu,cuda,resident")
    ap.add_argument("--out", default=None, help="with --sweep: results JSON")
    ap.add_argument("--nvtx", action="store_true",
                    help="with --run on the GPU: NVTX ranges + a "
                         "cudaProfilerStart/Stop window over the plain ticks "
                         "(for nsys --capture-range=cudaProfilerApi)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if sum(bool(x) for x in (args.make is not None, args.run, args.sweep,
                                args.batch_run, args.batch_sweep)) != 1:
        ap.error("exactly one of --make N / --run / --sweep / --batch-run / "
                 "--batch-sweep")
    if args.batch_run or args.batch_sweep:
        if args.level is None:
            ap.error("--batch-run / --batch-sweep need --level")
        if args.batch_sweep:
            return batch_sweep(args)
        r = run_batch(args.level, args.levels_dir, args.n, Path(args.build),
                      args.warmup, args.ticks)
        print("BATCH_JSON " + json.dumps(r), flush=True)
        return 0
    if args.make is not None:
        print(f"wrote {make_stacked(args.src, args.levels_dir, args.make)}")
        return 0
    if args.run:
        if args.level is None:
            ap.error("--run needs --level")
        r = run_one(args.level, args.levels_dir, args.backend,
                    Path(args.build), args.warmup, args.ticks, nvtx=args.nvtx)
        print("SCALING_JSON " + json.dumps(r), flush=True)
        return 0
    return sweep(args)


if __name__ == "__main__":
    sys.exit(main())
