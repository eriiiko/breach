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
    (a fixture; the SHIPPED smoke absorbs at its derived 25.36 since P5c, so the
    live game pays this wherever there is smoke) on every air
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
            # scratch); no gas group without --smoke (a smoke-free scene: the
            # live per-call path uploads no gas plane there either).
            eff = [cp.empty((1, h, w), dtype=cp.int32) for _ in range(2)]
            d_rad = [cp.empty((1, h, w), dtype=cp.int64) for _ in range(4)]
            d_cnt = cp.empty((1, SLOTS), dtype=cp.int64)
            # the gas group on the device: none by default (a smoke-free
            # scene); --smoke uploads the fixture's ACTIVE plane, as the
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
                    d_etab.data.ptr, int(tbl.fine_bits),  # #78: the device table's currency
                    d_vl.data.ptr, d_kl.data.ptr, d_ta.data.ptr,
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


def _playground_planes():
    """The PLAYGROUND's own planes, as the live step_tail hands them to the
    sweep: a wood tile heated to the 1263-game plateau and lit (CLAUDE.md
    "Starting a fire"), a marine stamped beside it, a few ticks run with light
    requested so the smoke, the plume and the stamps are the game's. Returns a
    dict of the sweep's inputs (heat, gas and light groups)."""
    from level_loader import load as load_level
    from simulation import Simulation, fire_fixed
    from simulation.materials import MAT_WOOD
    from simulation.unit import Unit
    sim = Simulation(load_level("playground"), seed=1, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    ys, xs = np.where((g.material == MAT_WOOD) & g.thermal_solid)
    for y, x in zip(ys, xs):
        if 0 < y < g.material.shape[0] - 1 and 0 < x < g.material.shape[1] - 1 \
                and not g.thermal_solid[y, x + 1]:
            g.temperature[y, x] = 1263 << 16
            g.fire[y, x] = fire_fixed.quantize_scalar(0.8)
            sim.add_unit(Unit("M1", x=int(x) + 2, y=int(y), team=0))
            break
    sim.physics_runner.engine.light_requested = True
    for _ in range(12):
        sim.set_paused(False)
        sim.step()
    pr = sim.physics_runner
    nb = (g.gas[pr._o2_idx].astype(np.int64)
          + g.gas[int(g.gases.name_to_id["inert_n2"])]).astype(np.int32)
    return dict(sim=sim, T=g.temperature, a=g.heat_atten_q, d=g.dyn_heat_atten_q,
                his=g.heat_inv_shift, ts=g.thermal_solid, vac=g.is_vacuum,
                k_leak_q=int(pr.k_leak_q), vac_level=int(pr.rad_amb_vacuum_q),
                gas=np.ascontiguousarray(g.gas), hq=g.gases.heat_absorb_q16,
                n_bulk=np.ascontiguousarray(nb),
                la=g.light_atten_q, ld=g.dyn_light_atten_q,
                lab=g.gases.light_absorb_q16, lgl=g.gases.light_glow_q16)


def _synthetic_planes(h, w, rng):
    """The bench's busy random scene at (h, w), with light planes on the same
    solids (a per channel, a tint on a third of them) and bodies stamped."""
    T, a, d, his, ts = _scene(h, w, rng)
    la = np.repeat(a[..., None], 3, axis=-1).astype(np.int32)
    tint = (rng.random((h, w)) < 0.33) & ts
    la[tint, 1] //= 2
    ld = np.maximum(la, np.repeat(d[..., None], 3, axis=-1)).astype(np.int32)
    return dict(T=T, a=a, d=d, his=his, ts=ts, vac=np.zeros((h, w), dtype=bool),
                k_leak_q=6554, vac_level=-1, gas=None,
                la=np.ascontiguousarray(la), ld=np.ascontiguousarray(ld))


def _light_rows(args, tbl, t_amb_q, rng):
    """P6a (design §10's gate; the brief's timing STOP): the sweep's ms per
    tick with the light channels OFF and ON, CPU and (with --cuda) the CUDA
    per-call path, on the playground's own planes and on 128x256. Every light
    run is checked bit for bit CPU vs GPU, and heat with light on against heat
    with light off, before it is timed."""
    cur = _currency()
    rows = [("playground", _playground_planes())]
    rows.append(("128x256", _synthetic_planes(128, 256, rng)))
    print(f"radiation sweep, light OFF vs ON (P6a), shear heat + step light, S16, "
          f"{args.iters} iterations per point"
          + (f"; device: {bp.cuda_device_info()}" if args.cuda else ""))
    hdr = f"{'scene':>11} {'cells':>7} {'cpu off':>8} {'cpu on':>8} {'light +':>8}"
    if args.cuda:
        hdr += f" {'gpu off':>8} {'gpu on':>8} {'launch off':>11} {'launch on':>10}"
    print(hdr + "   (ms per tick)")
    for name, P in rows:
        h, w = P["T"].shape
        gkw = {}
        lkw_extra = {}
        if P.get("gas") is not None:
            gkw = dict(gas=P["gas"], heat_absorb_q16=P["hq"], n_bulk=P["n_bulk"],
                       n_floor_q=cur[0], recip_cv=cur[1])
            lkw_extra = dict(light_absorb_q16=P["lab"], light_glow_q16=P["lgl"])
        sweep = bp.RadiationSweep()
        amb = sweep.derive_ambient(np.ascontiguousarray(P["vac"]), tbl, P["vac_level"])
        heat = [np.zeros((h, w), dtype=np.int64) for _ in range(4)]
        lq = np.zeros((h, w, 3), dtype=np.int64)
        lf = np.zeros((h, w, 2), dtype=np.int64)
        lg = np.zeros((h, w, 3), dtype=np.int64)
        lkw = dict(light_atten_q=P["la"], dyn_light_atten_q=P["ld"], light_q=lq,
                   light_flux_q=lf, light_glow=lg, **lkw_extra)
        base = (P["T"], P["a"], P["d"], P["his"], P["ts"], tbl, amb, t_amb_q,
                P["k_leak_q"], bp.RadiationSweep.SHEAR, 16)

        def cpu_off():
            sweep.run(*base, *heat, **gkw)

        def cpu_on():
            sweep.run(*base, *heat, **gkw, **lkw)
        cpu_off()
        heat_off = [p.copy() for p in heat]
        cpu_on()
        assert all(np.array_equal(x, y) for x, y in zip(heat_off, heat)), \
            "light moved a heat plane"
        assert np.any(lq), f"{name}: a light run lit nothing"
        light_cpu = (lq.copy(), lf.copy(), lg.copy())
        t_off = _best_of(cpu_off, args.iters)
        t_on = _best_of(cpu_on, args.iters)
        line = (f"{name:>11} {h * w:7d} {t_off * 1e3:8.2f} {t_on * 1e3:8.2f} "
                f"{(t_on - t_off) * 1e3:8.2f}")
        if args.cuda:
            gheat = [np.zeros((h, w), dtype=np.int64) for _ in range(4)]
            glq, glf, glg = (np.zeros_like(lq), np.zeros_like(lf), np.zeros_like(lg))
            glkw = dict(light_atten_q=P["la"], dyn_light_atten_q=P["ld"], light_q=glq,
                        light_flux_q=glf, light_glow=glg, **lkw_extra)
            gbase = (P["T"], P["a"], P["d"], P["his"], P["ts"], tbl, None, t_amb_q,
                     P["k_leak_q"], bp.RadiationSweep.SHEAR, 16)
            vkw = dict(is_vacuum=np.ascontiguousarray(P["vac"]), vac_level=P["vac_level"])

            def gpu_off():
                return bp.cuda_radiation_sweep_run(*gbase, *gheat, **vkw, **gkw)

            def gpu_on():
                return bp.cuda_radiation_sweep_run(*gbase, *gheat, **vkw, **gkw, **glkw)
            n_off = gpu_off()[2]
            n_on = gpu_on()[2]
            assert all(np.array_equal(x, y) for x, y in zip(heat, gheat)), "GPU heat != CPU"
            assert all(np.array_equal(x, y) for x, y in zip(light_cpu, (glq, glf, glg))), \
                "GPU light != CPU"
            g_off = _best_of(gpu_off, args.iters)
            g_on = _best_of(gpu_on, args.iters)
            line += f" {g_off * 1e3:8.2f} {g_on * 1e3:8.2f} {n_off:11d} {n_on:10d}"
        print(line)
    budget = 41.67
    print(f"  the brief's STOP: CPU heat + light on the playground must stay under "
          f"{budget / 2:.2f} ms (half the {budget} ms tick)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--cuda", action="store_true",
                    help="time the CUDA twin against the CPU sweep (design §10)")
    ap.add_argument("--smoke", action="store_true",
                    help="P5b: add a test-fixture absorbing smoke on every air cell "
                         "(hot), so the gas arm of the Fleck pre-pass is priced "
                         "everywhere -- its cost where it engages")
    ap.add_argument("--light", action="store_true",
                    help="P6a: the sweep with the light channels OFF vs ON, on the "
                         "playground's own planes and on 128x256 (with --cuda: the "
                         "per-call GPU path too)")
    args = ap.parse_args(argv)
    _import_bp(args.cuda)
    tbl, t_amb_q = _table()
    rng = np.random.default_rng(20260916)
    if args.light:
        _light_rows(args, tbl, t_amb_q, rng)
    elif args.cuda:
        _cuda_rows(args, tbl, t_amb_q, rng)
    else:
        _cpu_rows(args, tbl, t_amb_q, rng)
    return 0


if __name__ == "__main__":
    sys.exit(main())
