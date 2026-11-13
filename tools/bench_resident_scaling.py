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
class GpuSampler:
    """``nvidia-smi`` beside a timed window: the P-state, the clocks, the
    utilisation, the temperature and the power draw every ``period_ms``.

    The P2 finding (``docs/residency_benchmark_2026-11-07.md``): on a lightly
    loaded GPU the driver holds a CUDA context at lower clocks after a few
    seconds, so a device number without its clocks says little."""

    QUERY = "pstate,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu,power.draw"

    def __init__(self, period_ms: int = 500):
        self.period_ms = int(period_ms)
        self._proc = None
        self._rows = []
        self._thread = None

    def start(self):
        import threading
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={self.QUERY}",
                 "--format=csv,noheader,nounits", "-lms", str(self.period_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self._proc = None
            return self

        def read():
            for ln in self._proc.stdout:
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) != 6:
                    continue
                try:
                    self._rows.append((parts[0],) + tuple(float(x) for x in parts[1:]))
                except ValueError:
                    continue
        self._thread = threading.Thread(target=read, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        import statistics
        if self._proc is None:
            return {"samples": 0}
        self._proc.terminate()
        self._thread.join(timeout=5)
        rows = list(self._rows)
        if not rows:
            return {"samples": 0}
        pst = {}
        for r in rows:
            pst[r[0]] = pst.get(r[0], 0) + 1

        def col(i):
            xs = [r[i] for r in rows]
            return {"median": statistics.median(xs), "min": min(xs), "max": max(xs)}
        return {"samples": len(rows), "period_ms": self.period_ms, "pstate": pst,
                "sm_mhz": col(1), "mem_mhz": col(2), "util_pct": col(3),
                "temp_c": col(4), "power_w": col(5)}


# #29 G1: the attributed split of one batched tick. Labels for the callables
# the batch driver adds around the stages ``tick_stage_profile`` already knows.
BATCH_TICK = "batch tick"
BATCH_WRAPS = (
    ("simulation.simulation", "Simulation._step_pre_physics", "host pre (slots 1..6b)"),
    ("simulation.simulation", "Simulation._step_post_physics", "host post (slots 9..10)"),
    ("simulation.physics_runner", "PhysicsRunner._physics_head", "7 physics head (host)"),
    ("simulation.physics_runner", "PhysicsRunner._resident_prepare", "7 prepare (host)"),
    ("simulation.physics_runner", "run_resident_batch", "7 slot (the batch call)"),
)
# The device chain's stages (each synced in the attributed pass).
CHAIN_LABELS = (
    "  7 light inputs (resident)",
    "  7 stamp (resident)",
    "  7 water (resident: substeps + W5/W3)",
    "  7 EOS run_substeps_resident",
    "  7 trace tail (resident)",
    "  7 combustion (resident)",
    "  7 tail (resident: fire+sweep+temp+books)",
)
BUS_STAGE_LABELS = ("  7 H2D GameMap.from_host", "  7 D2H GameMap.to_host")


def _batch_split(batch, bp, ticks: int) -> dict:
    """``ticks`` more batched ticks with a device sync after every device
    stage (the attribution: ``tick_stage_profile``'s timers, plus the batch
    driver's own callables). Returns ms per BATCHED tick per label and the
    derived lines (device chain, bus, host)."""
    import importlib
    from simulation.resident_batch import ResidentBatch
    from tick_stage_profile import TickStageProfiler, _resolve
    prof = TickStageProfiler(skip=0, sync=True)
    prof.install(bp, device=True)
    prof._patch(ResidentBatch, "step", BATCH_TICK, device=True, tick=True)
    for mod, qual, label in BATCH_WRAPS:
        cont, attr = _resolve(importlib.import_module(mod), qual)
        prof._patch(cont, attr, label)
    try:
        for _ in range(ticks):
            batch.step()
    finally:
        prof.uninstall()
    n = max(1, len(prof.records))
    labels = set()
    for r in prof.records:
        labels.update(r)
    ms = {lb: 1e3 * sum(r.get(lb, 0.0) for r in prof.records) / n
          for lb in sorted(labels)}
    calls = {lb: sum(c.get(lb, 0) for c in prof.calls) / n for lb in sorted(labels)}
    bus_bytes = {}
    for t in prof.bus:
        for (d, _f), (_c, b) in t.items():
            bus_bytes[d] = bus_bytes.get(d, 0) + b
    n_env = batch.n_env
    chain = sum(ms.get(lb, 0.0) for lb in CHAIN_LABELS)
    bus = sum(ms.get(lb, 0.0) for lb in BUS_STAGE_LABELS)
    slot7 = ms.get("7 slot (the batch call)", 0.0)
    pre = ms.get("host pre (slots 1..6b)", 0.0)
    post = ms.get("host post (slots 9..10)", 0.0)
    head = ms.get("7 physics head (host)", 0.0)
    return {
        "ticks": len(prof.records), "ms_per_batch_tick": ms, "calls_per_batch_tick": calls,
        "tick_ms_synced": ms.get(BATCH_TICK, 0.0),
        "chain_ms": chain, "chain_ms_per_env": chain / n_env,
        "bus_ms_per_env": bus / n_env,
        "bus_kib_per_env": {d: b / n / n_env / 1024.0 for d, b in bus_bytes.items()},
        "slot7_host_rest_ms_per_env": (slot7 - chain - bus) / n_env,
        "host_pre_ms_per_env": pre / n_env,
        "host_post_ms_per_env": post / n_env,
        "physics_head_ms_per_env": head / n_env,
        "entities_9e_ms_per_env": ms.get("9e entities", 0.0) / n_env,
        "stamp_host_ms_per_env": ms.get("6 stamp_units", 0.0) / n_env,
        "walls_9b_ms_per_env": ms.get("9b over-pressure walls", 0.0) / n_env,
    }


def _parse_window(text):
    """``"A:B"`` -> ``(A, B)``: the scene's ticks ``A+1 .. B`` are timed."""
    if not text:
        return None
    a, b = (int(x) for x in str(text).split(":"))
    if not 0 <= a < b:
        raise SystemExit(f"--window {text!r}: need 0 <= A < B")
    return a, b


def _bench_edits(sim, e, tick):
    """#29 P4c ``--edits``: env ``e``'s slot-6b FieldEdits on ``tick`` -- a
    blast-sized load at an open tile of its own (a noisy smoke disc, a heat
    disc, a noisy atmosphere disc, a wave_source tile; ~110 tiles), so the
    batched tick carries every env's flush."""
    from simulation.field_edit import EditMode, Falloff, FieldEdit, Region
    g = sim.gmap
    cache = g.__dict__.setdefault("_bench_open", None)
    if cache is None:
        import numpy as np
        cache = np.argwhere(~g.solid & ~g.is_vacuum)
        g._bench_open = cache
    y, x = (int(v) for v in cache[(e * 997 + tick * 37) % len(cache)])
    sim.edit(FieldEdit("smoke", Region.DISC, (y, x, 4.0), 0.4, EditMode.ADD,
                       Falloff.LINEAR, noise=0.3, source_id=1))
    sim.edit(FieldEdit("heat", Region.DISC, (y, x, 3.0), 2.0, EditMode.ADD,
                       Falloff.LINEAR, source_id=2))
    sim.edit(FieldEdit("atmosphere", Region.DISC, (y, x, 3.0), 0.3, EditMode.ADD,
                       Falloff.LINEAR, noise=0.3, source_id=3))
    sim.edit(FieldEdit("wave_source", Region.TILE, (y, x), 50.0, EditMode.ADD,
                       source_id=4))


def run_batch(level_name: str, levels_dir, n: int, build: Path,
              warmup: int, ticks: int, *, warmup_s: float = 0.0,
              seconds: float = 0.0, split_ticks: int = 0,
              digest_tick: int = 0, window=None, edits: bool = False,
              host_edits: bool = False) -> dict:
    """``n`` envs of ``level_name`` (seeds 1..n) in one ResidentBatch, in THIS
    process. Times the whole batched tick, the slot-7 call (the device chain
    plus every env's entry / exit transfers: ``run_resident_batch``) and the
    rest (every env's host slots, 1..6b and 9..10). Returns BATCH_JSON.

    Without ``window`` (P5's form): ``warmup`` ticks, then ``ticks`` timed.

    #29 G1, ``window = (A, B)``: the scene is NOT stationary (an empty ship's
    substep count falls from the cap to 1 as its vents settle), so both sides
    time the SAME ticks of it, ``A+1 .. B`` after a fresh start. First the
    context is loaded for ``warmup_s`` seconds (and ``warmup`` ticks) on a
    throwaway state -- the driver settles a context's clocks only after
    several seconds of load --, then every env is replaced by a fresh
    simulation (``ResidentBatch.reset_env``), stepped ``A`` ticks untimed and
    ``B - A`` ticks timed; that repeats until ``seconds`` of timed ticks are
    pooled. The GPU's clocks are sampled over the timed ticks.
    ``split_ticks`` > 0 adds one more fresh start whose ticks ``A+1 ..
    A+split_ticks`` run the attributed pass (:func:`_batch_split`);
    ``digest_tick`` > 0 records the first envs' tick digests after that many
    ticks of the first fresh start (the same-scene check against
    ``--farm-run``)."""
    from bench_bot_matches import setup_cuda
    setup_cuda(build, True)
    import breach_physics as bp
    import level_loader
    import match_replay
    import numpy as np
    from simulation import Simulation
    from simulation import physics_runner as pr
    from simulation.resident_batch import ResidentBatch
    from simulation.ruleset import ContinuousRealtime
    lvl = level_loader.load(level_name, str(levels_dir))
    used0 = _device_used()

    def fresh(e):
        s = Simulation(lvl, seed=1 + e, breach_physics=bp, enable_recorder=False,
                       ruleset=ContinuousRealtime())
        s.physics_runner.headless = True
        s.set_paused(False)
        return s
    sims = [fresh(e) for e in range(n)]
    batch = ResidentBatch(sims)
    if host_edits:
        pr.set_device_edit_queue(False)   # #29 P4c: the A/B (host flush at 6b)
    if edits:
        _orig_step = batch.step
        _tick = [0]

        def _step_with_edits():
            for e, s in enumerate(batch.sims):
                _bench_edits(s, e, _tick[0])
            _tick[0] += 1
            return _orig_step()
        batch.step = _step_with_edits
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
    totals, s7, nsub_seen = [], [], []
    digests = None
    reps = 0
    gpu = {"samples": 0}
    sampler = None
    try:
        # the warm-up: P5's tick count, and G1's seconds of load
        t_w = time.perf_counter()
        n_warm = 0
        while n_warm < warmup or time.perf_counter() - t_w < warmup_s:
            batch.step()
            n_warm += 1
        if window is None:
            sampler = GpuSampler().start()
            for _ in range(ticks):
                k0 = len(slot7)
                t0 = time.perf_counter()
                batch.step()
                totals.append(time.perf_counter() - t0)
                s7.append(slot7[k0])
                nsub_seen.append(int(batch.sims[0].physics_runner.eos.dbg_last_n_sub))
            gpu = sampler.stop()
            sampler = None
        else:
            a_tick, b_tick = window
            sampler = GpuSampler().start()
            while reps == 0 or sum(totals) < seconds:
                for e in range(n):
                    batch.reset_env(e, fresh(e))
                for i in range(1, b_tick + 1):
                    k0 = len(slot7)
                    t0 = time.perf_counter()
                    batch.step()
                    dt = time.perf_counter() - t0
                    if i > a_tick:
                        totals.append(dt)
                        s7.append(slot7[k0])
                        nsub_seen.append(
                            int(batch.sims[0].physics_runner.eos.dbg_last_n_sub))
                    if reps == 0 and i == digest_tick:
                        digests = [match_replay.match_digests(x)[1]
                                   for x in batch.sims[:8]]
                reps += 1
            gpu = sampler.stop()
            sampler = None
    finally:
        pr.run_resident_batch = real
        if sampler is not None:
            sampler.stop()
    used1 = _device_used()
    tot = np.asarray(totals) * 1e3
    s7 = np.asarray(s7) * 1e3
    host = tot - s7
    h, w = batch.sims[0].gmap.material.shape
    cells = int(h * w)
    mean = float(tot.mean())
    rss = None
    try:
        import psutil
        rss = int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    out = {"level": level_name, "n": n, "h": int(h), "w": int(w), "cells": cells,
           "warmup": n_warm, "ticks": len(totals),
           "window": list(window) if window else None, "reps": reps,
           "window_s": float(tot.sum() / 1e3),
           "tick_ms": {"mean": mean, "p50": float(np.median(tot)),
                       "p95": float(np.percentile(tot, 95))},
           "slot7_ms": float(s7.mean()),
           "host_ms_per_env": float(host.mean() / n),
           "env_ticks_per_s": float(n * 1e3 / mean),
           "slot7_ns_per_cell": float(s7.mean() * 1e6 / (n * cells)),
           "tick_ns_per_cell": float(mean * 1e6 / (n * cells)),
           "nsub_reads": int(bp.eos_nsub_reads()) - nsub0,
           "n_sub_last": [int(x.physics_runner.eos.dbg_last_n_sub)
                          for x in batch.sims[:8]],
           "n_sub_mean_env0": float(np.mean(nsub_seen)),
           "device_used_before": used0, "device_used_after": used1,
           "host_rss_bytes": rss, "gpu": gpu,
           "digest_tick": digest_tick, "tick_digests": digests,
           "build_id": str(bp.build_id()),
           "edits": bool(edits), "host_edits": bool(host_edits),
           "entities": int(sum(1 for _ in getattr(batch.sims[0], "entities", []) or []))}
    if split_ticks > 0:
        if window is not None:
            for e in range(n):
                batch.reset_env(e, fresh(e))
            for _ in range(window[0]):
                batch.step()
        sampler = GpuSampler().start()
        try:
            out["split"] = _batch_split(batch, bp, split_ticks)
        finally:
            out["split_gpu"] = sampler.stop()
    return out


# ---------------------------------------------------------------------------
# #29 G1: THE SAME SCENE ON A CPU FARM -- W processes, one env each, the CPU
# build, the scene ``--batch-run`` steps (NO units, the level's entities ON,
# ContinuousRealtime, headless: no ripple), seeds 1..W. No driver and no IPC
# per step: every worker steps flat out over one common window, so this is the
# farm's best case for the scene (``rl_env``'s farm adds its observation, the
# pipes and the driver: 1 to 8 % of a step, P1a).
# ---------------------------------------------------------------------------
def _farm_worker(idx, level_name, levels_dir, build, window, seconds, digest_tick,
                 barrier, stop, q):
    try:
        from bench_bot_matches import import_build
        bp = import_build(Path(build))
        import level_loader
        import match_replay
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        lvl = level_loader.load(level_name, str(levels_dir))
        a_tick, b_tick = window

        def fresh():
            s = Simulation(lvl, seed=1 + idx, breach_physics=bp,
                           enable_recorder=False, ruleset=ContinuousRealtime())
            s.physics_runner.headless = True
            s.set_paused(False)
            return s
        sim = fresh()
        for _ in range(20):          # imports, first-touch scratch
            sim.step()
        barrier.wait(timeout=600)
        digest = None
        n, dt, reps, nsub_sum = 0, 0.0, 0, 0
        reported = False
        while not stop.is_set():
            sim = fresh()
            for i in range(1, b_tick + 1):
                t0 = time.perf_counter()
                sim.step()
                t1 = time.perf_counter()
                if i > a_tick and not reported:
                    n += 1
                    dt += t1 - t0
                    nsub_sum += int(sim.physics_runner.eos.dbg_last_n_sub)
                if reps == 0 and i == digest_tick:
                    digest = match_replay.match_digests(sim)[1]
            reps += 1
            if not reported and dt >= seconds:
                # report, then KEEP the core busy until every worker has
                # (a finished worker would hand its core to the others)
                rss = None
                try:
                    import psutil
                    rss = int(psutil.Process().memory_info().rss)
                except Exception:
                    pass
                mods = sorted(m for m in sys.modules
                              if m.split(".")[0] in ("cupy", "pyray"))
                q.put({"idx": idx, "ticks": n, "seconds": dt, "reps": reps,
                       "digest": digest, "rss": rss, "n_sub_mean": nsub_sum / max(1, n),
                       "build_id": str(bp.build_id()), "gpu_modules": mods})
                reported = True
    except Exception:
        import traceback
        q.put({"idx": idx, "error": traceback.format_exc()})
        try:
            barrier.abort()
        except Exception:
            pass


def run_farm(level_name: str, levels_dir, workers: int, build: Path,
             window, seconds: float, digest_tick: int = 0) -> dict:
    """``workers`` processes, each stepping its own no-units env of
    ``level_name`` on the CPU build. Every worker times the scene's ticks
    ``A+1 .. B`` (``window``) of fresh start after fresh start until it has
    ``seconds`` of timed ticks, and keeps cycling until all have (constant
    load). Returns FARM_JSON: env-ticks/s = the sum over workers of timed
    ticks / timed seconds (a worker's fresh starts and its untimed ticks
    ``1 .. A`` are outside its own clock, inside the others' load)."""
    import multiprocessing as mp
    import os
    from rl_env import WORKER_ENV
    saved = {k: os.environ.get(k) for k in WORKER_ENV}
    os.environ.update(WORKER_ENV)      # spawned workers inherit: one BLAS thread
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(workers)
    stop = ctx.Event()
    q = ctx.Queue()
    procs = [ctx.Process(target=_farm_worker,
                         args=(i, level_name, str(levels_dir), str(build),
                               tuple(window), seconds, digest_tick, barrier, stop, q))
             for i in range(workers)]
    t0 = time.perf_counter()
    try:
        for p in procs:
            p.start()
        res = [q.get(timeout=900 + 20 * seconds) for _ in procs]
        stop.set()
        for p in procs:
            p.join(timeout=120)
    finally:
        stop.set()
        for p in procs:
            if p.is_alive():
                p.terminate()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    errs = [r for r in res if "error" in r]
    if errs:
        raise SystemExit("farm worker failed:\n" + errs[0]["error"])
    res.sort(key=lambda r: r["idx"])
    rates = [r["ticks"] / r["seconds"] for r in res]
    return {"level": level_name, "workers": workers, "window": list(window),
            "seconds": seconds, "wall_s": time.perf_counter() - t0,
            "env_ticks_per_s": float(sum(rates)),
            "per_worker": {"mean": float(sum(rates) / workers),
                           "min": float(min(rates)), "max": float(max(rates))},
            "ticks": [r["ticks"] for r in res],
            "reps": [r["reps"] for r in res],
            "n_sub_mean": float(sum(r["n_sub_mean"] for r in res) / workers),
            "rss_mb": [None if r["rss"] is None else r["rss"] / 2 ** 20 for r in res],
            "digest_tick": digest_tick,
            "tick_digests": [r["digest"] for r in res][:8],
            "build_id": res[0]["build_id"],
            "gpu_modules": sorted({m for r in res for m in r["gpu_modules"]})}


def batch_sweep(args) -> int:
    ns = [int(x) for x in args.ns.split(",")]
    results = []
    out = Path(args.out) if args.out else None
    for n in ns:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--batch-run",
               "--level", args.level, "--levels-dir", str(args.levels_dir),
               "--n", str(n), "--build", str(args.build),
               "--warmup", str(args.warmup), "--ticks", str(args.ticks),
               "--warmup-s", str(args.warmup_s), "--seconds", str(args.seconds),
               "--split-ticks", str(args.split_ticks),
               "--digest-tick", str(args.digest_tick)]
        if args.window:
            cmd += ["--window", args.window]
        if args.edits:
            cmd += ["--edits"]
        if args.host_edits:
            cmd += ["--host-edits"]
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
    ap.add_argument("--farm-run", action="store_true",
                    help="#29 G1: --workers CPU processes, one no-units env of "
                         "--level each (the --batch-run scene) on --cpu-build, "
                         "for --seconds; prints FARM_JSON")
    ap.add_argument("--workers", type=int, default=8, help="with --farm-run")
    ap.add_argument("--cpu-build", default=str(ROOT / "cpp" / "build" / "Release"),
                    help="with --farm-run: the CPU breach_physics build")
    ap.add_argument("--window", default=None, metavar="A:B",
                    help="#29 G1, --batch-run / --farm-run: time the scene's "
                         "ticks A+1 .. B of fresh start after fresh start "
                         "(--farm-run requires it)")
    ap.add_argument("--warmup-s", type=float, default=0.0,
                    help="with --batch-run: load the context at least this "
                         "long before the timed ticks")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="with --window: pool at least this many seconds of "
                         "timed ticks (per worker on the farm)")
    ap.add_argument("--split-ticks", type=int, default=0,
                    help="with --batch-run: that many more ticks with a device "
                         "sync per stage (the attributed split)")
    ap.add_argument("--digest-tick", type=int, default=0,
                    help="--batch-run / --farm-run: record the tick digest of "
                         "the first 8 envs after this many ticks of the first "
                         "fresh start")
    ap.add_argument("--n", type=int, default=1, help="with --batch-run")
    ap.add_argument("--edits", action="store_true",
                    help="#29 P4c: every env enqueues a blast-sized FieldEdit load "
                         "each tick (with --batch-run / --batch-sweep)")
    ap.add_argument("--host-edits", action="store_true",
                    help="#29 P4c: the device edit queue OFF (slot 6b flushes on "
                         "the host on the resident path): the A/B")
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
                                args.batch_run, args.batch_sweep,
                                args.farm_run)) != 1:
        ap.error("exactly one of --make N / --run / --sweep / --batch-run / "
                 "--batch-sweep / --farm-run")
    if args.farm_run:
        if args.level is None:
            ap.error("--farm-run needs --level")
        if not args.window:
            ap.error("--farm-run needs --window A:B")
        r = run_farm(args.level, args.levels_dir, args.workers, Path(args.cpu_build),
                     _parse_window(args.window), args.seconds or 20.0,
                     args.digest_tick)
        print("FARM_JSON " + json.dumps(r), flush=True)
        return 0
    if args.batch_run or args.batch_sweep:
        if args.level is None:
            ap.error("--batch-run / --batch-sweep need --level")
        if args.batch_sweep:
            return batch_sweep(args)
        r = run_batch(args.level, args.levels_dir, args.n, Path(args.build),
                      args.warmup, args.ticks, warmup_s=args.warmup_s,
                      seconds=args.seconds, split_ticks=args.split_ticks,
                      digest_tick=args.digest_tick,
                      window=_parse_window(args.window),
                      edits=args.edits, host_edits=args.host_edits)
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
