"""Timing bench for the radiation sweep (ray-engine-v2 P1 CPU, P4 CUDA twin;
design v3 §10).

Design §10 estimated ~10 ms per tick at 128x256 on one core, from a GUESSED
~5 ns per cell-update ("the number that is not known is a single
cell-update's cost"). This bench MEASURES it: the C++ sweep
(cpp/src/radiation_sweep.cpp, the binding's RadiationSweep.run) on a random
scene at three grid sizes, S16, shear AND step, and prints ms per tick and
ns per cell-update (one cell-update = one cell in one ordinate).

The scene is a bordered room with random opaque solids (about 12 % of the
interior) at fire temperatures and a few bodies — a busy scene, not an empty
one, so the timing is not flattered by a uniform field.

--cuda (P4) runs the SAME scenes through the CUDA build (cpp/build_cuda, which
carries the CPU sweep too, so both are timed in one process on one binary) and
adds §10's two unknowns, measured rather than assumed:
  * gpu/call   — breach_cuda::radiation_sweep_step, the per-call path
                 PhysicsEngine.step_tail dispatches (arena malloc, H2D, the
                 launches, sync, D2H, free), wall clock;
  * core       — the launch core alone on resident CuPy buffers (launch +
                 sync): the kernels and their launch overhead, no transfer;
  * launches   — kernel launches per sweep (3 bookkeeping + one per
                 wavefront index) and core time per launch;
  * H2D / D2H  — what the per-call path moves, bytes and ms, measured with
                 CuPy on the same pageable numpy planes;
  * alloc      — one cudaMalloc + cudaFree of the per-call arena's size.
Every GPU result is checked bit for bit against the CPU sweep first.

Run:
    C:/Users/steen/anaconda3/python.exe tools/bench_radiation_sweep.py [--iters N]
    C:/Users/steen/anaconda3/python.exe tools/bench_radiation_sweep.py --cuda [--iters N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIZES = ((72, 46), (128, 256), (256, 512))
ONE = 65536

bp = None           # the breach_physics module, imported per mode in main()


def _import_bp(cuda: bool):
    """The CPU build by default; with --cuda the CUDA build, through
    tools/run_on_cuda.py's one path setup (never a second copy of it)."""
    global bp
    if cuda:
        sys.path.insert(0, str(ROOT / "tools"))
        from run_on_cuda import setup_cuda_import
        setup_cuda_import()
    else:
        for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
    import breach_physics
    bp = breach_physics
    if cuda and not getattr(bp, "HAS_CUDA", False):
        raise SystemExit("--cuda: the imported breach_physics is the CPU build")
    return bp


def _table():
    from config import CFG
    import temperature_scale
    ts = temperature_scale.load(CFG)
    tbl = bp.EmissiveTable()
    # T6 (issue #12): [physics.fire] rad_scale (the old cast's fitted key) is
    # deleted; [physics.radiation] rad_scale_derived is the sweep's own live
    # scale, and this bench is timing the sweep, so it is the right key too.
    tbl.rad_scale = float(CFG.physics.radiation.rad_scale_derived)
    tbl.kelvin_ambient = float(ts.kelvin_ambient)
    tbl.k_temp_to_kelvin = float(ts.k_temp_to_kelvin)
    tbl.bake()
    return tbl, int(round(float(ts.kelvin_ambient) * ONE))


def _scene(h, w, rng):
    a = np.zeros((h, w), dtype=np.int32)
    ts = np.zeros((h, w), dtype=bool)
    T = np.zeros((h, w), dtype=np.int32)
    a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = ONE
    inner = rng.random((h, w)) < 0.12
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    a[inner] = rng.choice([ONE, 32768, 19661], size=int(inner.sum()))
    ts[:] = a > 0
    hot = inner & (rng.random((h, w)) < 0.3)
    T[hot] = rng.choice([300 << 16, 1263 << 16, 5000 << 16], size=int(hot.sum()))
    d = a.copy()
    bodies = (~ts) & (rng.random((h, w)) < 0.01)
    d[bodies] = ONE
    his = np.full((h, w), 3, dtype=np.int32)
    return (np.ascontiguousarray(T), np.ascontiguousarray(a), np.ascontiguousarray(d),
            his, np.ascontiguousarray(ts))


def _currency():
    """The gas arm's currency (P5b) exactly as the live engine derives it:
    PhysicsEngine.gas_capacity_q() on a temperature solver bound to config's
    [physics.thermal] c_v / n_floor_heat (PhysicsRunner's own binding)."""
    from config import CFG
    eng = bp.PhysicsEngine()
    thermal = CFG.physics.thermal
    eng.temperature.c_v = float(thermal.c_v)
    eng.temperature.n_floor_heat = float(thermal.n_floor_heat)
    n_floor_q, _c_v_q, recip_cv = eng.gas_capacity_q()
    return int(n_floor_q), int(recip_cv)


def _smoke_group(h, w, rng, ts, T):
    """--smoke (P5b): a TEST-FIXTURE smoke group, to time the gas arm of the Fleck
    pre-pass where it engages -- the engine's 7-plane layout (steam, smoke,
    poison, teargas, fuel_gas, o2, inert_n2), smoke absorbing at heat_absorb 5.0
    (every SHIPPED row is 0.0, so the live game never pays this) on every air
    cell, the bulk pair at ambient, and the air HOT (a fire's plume), so the arm
    prices every one of them and damps the hot ones. Mutates T; returns
    (gas, hq, n_bulk)."""
    gas = np.zeros((7, h, w), dtype=np.int32)
    air = ~ts
    gas[1][air] = rng.choice([1966, 6554, 17695, ONE], size=int(air.sum()))
    gas[5][:] = 13763
    gas[6][:] = 51773
    hq = np.zeros(7, dtype=np.int32)
    hq[1] = 5 * ONE
    n_bulk = (gas[5].astype(np.int64) + gas[6]).astype(np.int32)
    T[air] = rng.choice(np.asarray([0, 300 << 16, 1263 << 16, 3000 << 16], dtype=np.int32),
                        size=int(air.sum()))
    return np.ascontiguousarray(gas), hq, np.ascontiguousarray(n_bulk)


def _best_of(fn, iters, reps=3):
    """min over `reps` of the mean over `iters` calls, in seconds."""
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        dt = (time.perf_counter() - t0) / iters
        best = dt if best is None else min(best, dt)
    return best


def _cpu_rows(args, tbl, t_amb_q, rng):
    print(f"radiation sweep CPU timing, S16, {args.iters} iterations per point "
          f"(one cell-update = one cell in one ordinate)"
          + ("; SMOKE: the gas arm priced on every air cell (a test fixture)"
             if args.smoke else ""))
    print(f"{'grid':>10} {'transport':>10} {'ms/tick':>10} {'ns/cell-update':>16} {'cells':>10}")
    cur = _currency()
    for (h, w) in SIZES:
        T, a, d, his, ts = _scene(h, w, rng)
        gkw = {}
        if args.smoke:
            g, hq, nb = _smoke_group(h, w, rng, ts, T)
            gkw = dict(gas=g, heat_absorb_q16=hq, n_bulk=nb, n_floor_q=cur[0],
                       recip_cv=cur[1])
        for name, transport in (("shear", bp.RadiationSweep.SHEAR), ("step", bp.RadiationSweep.STEP)):
            sweep = bp.RadiationSweep()
            planes = [np.zeros((h, w), dtype=np.int64) for _ in range(4)]
            # warm-up (scratch allocation) outside the timed loop
            sweep.run(T, a, d, his, ts, tbl, None, t_amb_q, 0, transport, 16, *planes, **gkw)
            best = _best_of(lambda: sweep.run(T, a, d, his, ts, tbl, None, t_amb_q, 0,
                                              transport, 16, *planes, **gkw), args.iters)
            ident = int(planes[0].sum()) + int(planes[1].sum()) + int(planes[2].sum())
            assert ident == 0, "the identity failed inside the bench"
            n_updates = h * w * 16
            print(f"{h:>4}x{w:<5} {name:>10} {best * 1e3:10.2f} {best * 1e9 / n_updates:16.1f} {h * w:10d}")


def _cuda_rows(args, tbl, t_amb_q, rng):
    import cupy as cp
    SLOTS = int(bp.RADIATION_SWEEP_CNT_SLOTS)
    k_leak_q = 6554                     # [physics.radiation] k_leak = 0.10, Q16
    print(f"radiation sweep CPU vs CUDA twin, S16, k_leak 0.10, {args.iters} "
          f"iterations per point; device: {bp.cuda_device_info()}"
          + ("; SMOKE: the gas arm priced on every air cell (a test fixture)"
             if args.smoke else ""))
    print(f"{'grid':>10} {'transport':>9} {'cpu ms':>8} {'gpu/call':>9} {'core ms':>8} "
          f"{'launches':>9} {'us/launch':>10} {'H2D KB':>7} {'H2D ms':>7} "
          f"{'D2H KB':>7} {'D2H ms':>7} {'alloc ms':>9}")
    cur = _currency()
    for (h, w) in SIZES:
        T, a, d, his, ts = _scene(h, w, rng)
        vac = np.zeros((h, w), dtype=bool)
        gkw = {}
        if args.smoke:
            g_sm, hq_sm, nb_sm = _smoke_group(h, w, rng, ts, T)
            gkw = dict(gas=g_sm, heat_absorb_q16=hq_sm, n_bulk=nb_sm, n_floor_q=cur[0],
                       recip_cv=cur[1])
        for name, transport in (("shear", bp.RadiationSweep.SHEAR), ("step", bp.RadiationSweep.STEP)):
            # --- the CPU sweep, the engine's own derive + run ---
            sweep = bp.RadiationSweep()
            cpu = [np.zeros((h, w), dtype=np.int64) for _ in range(4)]
            amb = sweep.derive_ambient(vac, tbl, -1)

            def cpu_once():
                sweep.run(T, a, d, his, ts, tbl, amb, t_amb_q, k_leak_q,
                          transport, 16, *cpu, **gkw)
            cpu_once()
            t_cpu = _best_of(cpu_once, args.iters)
            # --- the per-call GPU path (what step_tail dispatches) ---
            gpu = [np.zeros((h, w), dtype=np.int64) for _ in range(4)]

            def gpu_once():
                return bp.cuda_radiation_sweep_run(
                    T, a, d, his, ts, tbl, None, t_amb_q, k_leak_q, transport, 16,
                    *gpu, is_vacuum=vac, vac_level=-1, **gkw)
            _mn, _mx, launches = gpu_once()
            for c_, g_ in zip(cpu, gpu):
                assert np.array_equal(c_, g_), "the GPU sweep is not the CPU sweep"
            t_call = _best_of(gpu_once, args.iters)
            # --- the launch core on resident buffers (launch + sync only) ---
            dv = {k: cp.asarray(v) for k, v in (("T", T), ("a", a), ("d", d),
                                               ("his", his), ("ts", ts), ("vac", vac))}
            d_etab = cp.asarray(np.asarray(tbl.table(), dtype=np.int64))
            d_vl = cp.asarray(np.asarray([-1], dtype=np.int64))
            d_kl = cp.asarray(np.asarray([k_leak_q], dtype=np.int32))
            d_ta = cp.asarray(np.asarray([t_amb_q], dtype=np.int32))
            d_out = cp.empty((1, 16, h, w), dtype=cp.int64)
            scratch = [cp.empty((1, h, w), dtype=cp.int64) for _ in range(2)]
            d_f = cp.empty((1, h, w), dtype=cp.int32)
            # P5a: the effective extinction planes the wavefronts read (int32
            # scratch); no gas group here — the shipped table is all zero, so
            # the live per-call path uploads no gas plane either.
            eff = [cp.empty((1, h, w), dtype=cp.int32) for _ in range(2)]
            d_rad = [cp.empty((1, h, w), dtype=cp.int64) for _ in range(4)]
            d_cnt = cp.empty((1, SLOTS), dtype=cp.int64)
            # the gas group on the device: none by default (the shipped table
            # is all zero, so the live per-call path uploads no gas plane
            # either); --smoke uploads the fixture's ACTIVE plane, as the
            # per-call path does (P5a), with the currency by value (P5b)
            core_gas = (0, 0, 0, 0, 0, 0)
            if args.smoke:
                d_g = cp.asarray(g_sm[1:2][None])                 # (1, 1, h, w): active
                d_hq = cp.asarray(hq_sm[1:2])
                d_nb = cp.asarray(nb_sm[None])
                core_gas = (d_g.data.ptr, 1, d_hq.data.ptr, d_nb.data.ptr,
                            cur[0], cur[1])

            def core_once():
                bp.cuda_radiation_sweep_resident(
                    1, h, w, dv["T"].data.ptr, dv["a"].data.ptr, dv["d"].data.ptr,
                    dv["his"].data.ptr, dv["ts"].data.ptr, 0, dv["vac"].data.ptr,
                    d_etab.data.ptr, d_vl.data.ptr, d_kl.data.ptr, d_ta.data.ptr,
                    *core_gas,                           # P5a gas group + P5b currency
                    transport, 16, True, d_out.data.ptr, scratch[0].data.ptr,
                    scratch[1].data.ptr, d_f.data.ptr,
                    eff[0].data.ptr, eff[1].data.ptr,
                    *[p.data.ptr for p in d_rad], d_cnt.data.ptr)
                cp.cuda.Device().synchronize()
            core_once()
            for c_, g_ in zip(cpu, d_rad):
                assert np.array_equal(c_, g_.get()[0]), "the core is not the CPU sweep"
            t_core = _best_of(core_once, args.iters)
            # --- what the per-call path moves, on the same pageable planes ---
            ins = (T, a, d, his, ts, vac, np.asarray(tbl.table(), dtype=np.int64))
            if args.smoke:                    # the ACTIVE plane, its table, the bulk
                ins = ins + (g_sm[1], hq_sm[1:2], nb_sm)
            h2d_bytes = sum(int(x.nbytes) for x in ins)

            def h2d_once():
                for x in ins:
                    cp.asarray(x)
                cp.cuda.Device().synchronize()
            t_h2d = _best_of(h2d_once, args.iters)
            d2h_bytes = sum(int(x.nbytes) for x in cpu)

            def d2h_once():
                for p in d_rad:
                    p.get()
            t_d2h = _best_of(d2h_once, args.iters)
            # --- one cudaMalloc + cudaFree of the arena's size ---
            arena = (h2d_bytes + 16 * h * w * 8 + 2 * h * w * 8 + h * w * 4
                     + 2 * h * w * 4                   # P5a: the effective planes
                     + 4 * h * w * 8)

            def alloc_once():
                ptr = cp.cuda.runtime.malloc(arena)
                cp.cuda.runtime.free(ptr)
            t_alloc = _best_of(alloc_once, args.iters)
            print(f"{h:>4}x{w:<5} {name:>9} {t_cpu * 1e3:8.2f} {t_call * 1e3:9.2f} "
                  f"{t_core * 1e3:8.2f} {launches:>9d} {t_core * 1e6 / launches:10.2f} "
                  f"{h2d_bytes / 1024:7.0f} {t_h2d * 1e3:7.2f} {d2h_bytes / 1024:7.0f} "
                  f"{t_d2h * 1e3:7.2f} {t_alloc * 1e3:9.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cuda", action="store_true",
                    help="time the CUDA twin against the CPU sweep (design §10)")
    ap.add_argument("--smoke", action="store_true",
                    help="P5b: add a test-fixture absorbing smoke on every air cell "
                         "(hot), so the gas arm of the Fleck pre-pass is priced "
                         "everywhere -- its cost where it engages")
    args = ap.parse_args(argv)
    _import_bp(args.cuda)
    tbl, t_amb_q = _table()
    rng = np.random.default_rng(20260916)
    if args.cuda:
        _cuda_rows(args, tbl, t_amb_q, rng)
    else:
        _cpu_rows(args, tbl, t_amb_q, rng)
    return 0


if __name__ == "__main__":
    sys.exit(main())
