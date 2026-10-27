"""P6a — THE LIGHT CHANNELS' CUDA TWIN, held to the CPU sweep at tol 0
(ray-engine-v2; docs/ray_engine_v2_p6a_light_channels_brief_2026-09-25.md
section 3 items 2 and 3). Runs INSIDE the GPU subprocess; the pytest wrapper is
tests/test_cuda_radiation_sweep_light.py.

PROPERTY: with the light group riding the sweep, `breach_cuda::radiation_sweep_
step` (the per-call path PhysicsEngine.step_tail dispatches) and its (N, h, w)
launch core compute the SAME integers as `RadiationSweep::run`: the three light
planes (light_q, light_flux_q, light_glow), the twelve light books and the
light-stream telemetry -- AND the four heat planes, the Fleck plane and the heat
telemetry, which must moreover be BIT-IDENTICAL to the same sweep with light
OFF, on both backends (the heat-untouched gate: with light on the twin walks
every ordinate along the step anti-diagonals instead of shear's columns, and the
gather form must make that invisible to heat).

BREAKS IF: the .cu transcribes a light term differently (the split and its
remainder, the dark ring, the stamped absorption, the per-ordinate emission, the
symmetric flux shift, the 128-bit glow), an anti-diagonal launch stops being a
topological order for either transport, a light book stops being an order-free
integer sum (the warp reduction, the ring atomics), the union packing of the
gas planes feeds heat or light a wrong coefficient, or the (N, h, w) indexing of
a light plane leaks one env into another.

  PART L1 the matrix: random heat + light scenes (tints, glass, bodies, the
          temperature range to the table top) x S16/S12 x heat shear/step x
          light step/shear x smoke on/off, on the LIVE fine table and the
          resolving one; heat also compared against the light-OFF run.
  PART L2 shapes: degenerate strips and squares, odd sizes, 128x256, 256x512.
  PART L3 overwrite + ingress: garbage outputs come back exact; an illegal
          light scene (a > d on one channel, d > ONE, a partial group, a table
          that glows at bucket 0, smoke columns without the gas group) is
          rejected by both backends, the caller's planes untouched.
  PART L4 the launch core at N = 3 with light: three envs, each equal to its
          own CPU run; one illegal light env masked alone.
  PART L5 the live conductor: Simulation.step on the playground with a burning
          tile and a marine, light REQUESTED, only the radiation backend
          flipping -- every GameMap array (the light planes included), the
          engine's light books, per tick, tol 0.

Prints ``LS_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import random
import sys

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics`.
import breach_physics as bp

from _radiation_sweep_harness import (  # noqa: E402
    ONE, R, T_AMB_Q, TRANSPORTS, as_i32, engine_currency, gas_arrays,
    light_arrays, live_table, random_gas, random_scene, reference_table)

Q = R.quant
K_LEAK = Q(0.10)
GARBAGE = -0x5A5A5A5A5A5A5A5
_FAILS: list[str] = []
HEAT = ("rad_net", "rad_flux", "rad_amb", "rad_fluence")
LIGHT = ("light_q", "light_flux_q", "light_glow")


def _fail(msg: str) -> None:
    _FAILS.append(msg)
    print("  FAIL: " + msg)


def light_scene(rng, h, w):
    """Random light extinction, channel-first lists: clear air, glass, a crate,
    opaque, two asymmetric tints; bodies (d = ONE) on some cells."""
    tints = [(0, 0, 0), (0, 0, 0), (Q(0.1),) * 3, (Q(0.55),) * 3, (ONE,) * 3,
             (Q(0.2), Q(0.6), Q(0.9)), (Q(0.9), Q(0.1), Q(0.4))]
    la = [[[0] * w for _ in range(h)] for _ in range(3)]
    ld = [[[0] * w for _ in range(h)] for _ in range(3)]
    for y in range(h):
        for x in range(w):
            t = rng.choice(tints)
            body = rng.random() < 0.15
            for c in range(3):
                la[c][y][x] = t[c]
                ld[c][y][x] = ONE if body else t[c]
    return la, ld


def light_cols(rng, n):
    """Random light smoke columns for n gas planes (zeros included)."""
    lab = [[rng.choice([0, Q(0.14), Q(1.26), Q(2.5)]) for _ in range(3)] for _ in range(n)]
    lgl = [[rng.choice([0, Q(0.04), Q(0.92)]) for _ in range(3)] for _ in range(n)]
    return lab, lgl


def run(backend, sc, table, *, transport="shear", n_ord=16, k_q=0, fleck=True,
        gas=None, light=None, garbage=False):
    """One sweep on `backend` ("cpu" | "gpu"). `sc` = (a, d, T, his, ts) as
    lists; `gas` the reference-format (gas, hq, n_bulk); `light` a harness
    light dict (see light_arrays). Returns a dict of every output."""
    a, d, T, his, ts = sc
    h, w = len(a), len(a[0])
    heat = [np.full((h, w), GARBAGE if garbage else 0, dtype=np.int64) for _ in range(4)]
    g_a, hq_a, nb_a = gas_arrays(*gas) if gas is not None else (None, None, None)
    cur = {}
    if gas is not None:
        nf, rcv = engine_currency()
        cur = dict(n_floor_q=nf, recip_cv=rcv)
    lkw = light_arrays(light, h, w) if light is not None else {}
    if light is not None and garbage:
        for k in LIGHT:
            lkw[k][...] = GARBAGE
    args = (as_i32(T), as_i32(a), as_i32(d), as_i32(his),
            np.ascontiguousarray(np.asarray(ts) != 0), table)
    out = {}
    if backend == "cpu":
        sw = bp.RadiationSweep()
        sw.run(*args, None, int(T_AMB_Q), int(k_q), TRANSPORTS[transport], int(n_ord),
               *heat, fleck_enabled=bool(fleck), gas=g_a, heat_absorb_q16=hq_a,
               n_bulk=nb_a, **cur, **lkw)
        out["fleck"] = np.asarray(sw.fleck_plane(), dtype=np.int32)
        out["tele"] = (int(sw.min_stream), int(sw.max_stream))
        if light is not None:
            out["books"] = dict(sw.light_books)
    else:
        fo = np.full((h, w), -7, dtype=np.int32)
        res = bp.cuda_radiation_sweep_run(
            *args, None, int(T_AMB_Q), int(k_q), TRANSPORTS[transport], int(n_ord),
            *heat, fleck_enabled=bool(fleck), fleck_out=fo, gas=g_a,
            heat_absorb_q16=hq_a, n_bulk=nb_a, **cur, **lkw)
        out["fleck"] = fo
        out["tele"] = (int(res[0]), int(res[1]))
        want = bp.cuda_radiation_sweep_launch_count(TRANSPORTS[transport], int(n_ord),
                                                    h, w, light is not None)
        if int(res[2]) != want:
            _fail(f"launch count {res[2]} != the shape's {want}")
        if light is not None:
            out["books"] = dict(res[3])
    for name, p in zip(HEAT, heat):
        out[name] = p
    for k in LIGHT:
        if light is not None:
            out[k] = lkw[k]
    return out


def same(tag, c, g, keys) -> bool:
    ok = True
    for k in keys:
        if k in ("tele", "books"):
            if c[k] != g[k]:
                _fail(f"{tag}: {k} cpu={c[k]} gpu={g[k]}")
                ok = False
            continue
        if not np.array_equal(c[k], g[k]):
            bad = np.argwhere(c[k] != g[k])
            _fail(f"{tag}: {k} differs at {len(bad)} cells; first {tuple(bad[0])} "
                  f"cpu={int(c[k][tuple(bad[0])])} gpu={int(g[k][tuple(bad[0])])}")
            ok = False
    return ok


ALL_HEAT = HEAT + ("fleck", "tele")
ALL = ALL_HEAT + LIGHT + ("books",)


# ---------------------------------------------------------------------------
def part1_matrix() -> None:
    print("PART L1 — the light matrix, GPU vs CPU, tol 0; heat vs light OFF")
    n = n_glow = 0
    for tname, table, tbl_ref in (("live", live_table(), R.E_LIVE),
                                  ("resolving", reference_table(), R.E)):
        for seed in (1, 2, 3):
            rng = random.Random(20260925 + seed)
            h, w = rng.choice([(7, 9), (9, 11), (12, 8)])
            a, d, T, his, ts = random_scene(rng, h, w)
            if tname == "live":
                his = [[rng.choice([-3, -2, 3, 5]) for _ in range(w)] for _ in range(h)]
            la, ld = light_scene(rng, h, w)
            gas = random_gas(rng, h, w)
            lab, lgl = light_cols(rng, len(gas[0]))
            sc = (a, d, T, his, ts)
            for n_ord in (16, 12):
                for htr in ("shear", "step"):
                    for ltr in ("step", "shear"):
                        for smoke in (False, True):
                            light = dict(la=la, ld=ld, transport=ltr)
                            gkw = {}
                            if smoke:
                                light.update(light_absorb_q=lab, light_glow_q=lgl)
                                gkw = dict(gas=gas)
                            tag = f"L1 {tname} s{seed} S{n_ord} {htr}/{ltr} smoke={smoke}"
                            c = run("cpu", sc, table, transport=htr, n_ord=n_ord,
                                    k_q=K_LEAK, light=light, **gkw)
                            g = run("gpu", sc, table, transport=htr, n_ord=n_ord,
                                    k_q=K_LEAK, light=light, garbage=True, **gkw)
                            same(tag, c, g, ALL)
                            # HEAT UNTOUCHED: light on == light off, both backends
                            c0 = run("cpu", sc, table, transport=htr, n_ord=n_ord,
                                     k_q=K_LEAK, **gkw)
                            g0 = run("gpu", sc, table, transport=htr, n_ord=n_ord,
                                     k_q=K_LEAK, **gkw)
                            same(tag + " HEAT cpu on/off", c0, c, ALL_HEAT)
                            same(tag + " HEAT gpu on/off", g0, g, ALL_HEAT)
                            for k in LIGHT[:2]:
                                if not np.any(c[k] != 0):
                                    _fail(f"{tag}: {k} identically zero -- vacuous")
                            if smoke and np.any(c["light_glow"] != 0):
                                n_glow += 1
                            bk = c["books"]
                            for ch in range(3):
                                if bk["emit"][ch] + bk["ring_in"][ch] != \
                                        bk["absorb"][ch] + bk["ring_out"][ch]:
                                    _fail(f"{tag}: the light books do not close, ch {ch}")
                            n += 1
    if n_glow == 0:
        _fail("L1: no smoke case produced a glow -- the glow comparison is vacuous")
    print(f"  {n} configurations: every heat + light plane, the books and both "
          f"telemetries tol 0; heat identical with light off on both backends; "
          f"{n_glow} smoke cases with glow")


def part2_shapes() -> None:
    print("PART L2 — shapes (degenerate, odd, 128x256, 256x512)")
    table = live_table()
    rng = random.Random(62)
    n = 0
    for (h, w) in ((1, 1), (1, 9), (9, 1), (2, 2), (17, 33), (128, 256), (256, 512)):
        ts = [[1 if rng.random() < 0.15 else 0 for _ in range(w)] for _ in range(h)]
        a = [[Q(0.9) if ts[y][x] else 0 for x in range(w)] for y in range(h)]
        d = [[ONE if (not ts[y][x] and rng.random() < 0.01) else a[y][x]
              for x in range(w)] for y in range(h)]
        T = [[rng.choice([0, 1263 << 16, 5000 << 16]) if ts[y][x] else 0
              for x in range(w)] for y in range(h)]
        his = [[3] * w for _ in range(h)]
        la = [[[a[y][x] if c != 1 else a[y][x] // 2 for x in range(w)] for y in range(h)]
              for c in range(3)]
        ld = [[[max(la[c][y][x], d[y][x]) for x in range(w)] for y in range(h)]
              for c in range(3)]
        sc = (a, d, T, his, ts)
        light = dict(la=la, ld=ld)
        c = run("cpu", sc, table, light=light)
        g = run("gpu", sc, table, light=light, garbage=True)
        same(f"L2 {h}x{w}", c, g, ALL)
        c0 = run("cpu", sc, table)
        same(f"L2 {h}x{w} HEAT on/off", c0, c, ALL_HEAT)
        n += 1
    print(f"  {n} shapes tol 0, heat identical with light off")


def part3_ingress() -> None:
    print("PART L3 — overwrite + ingress rejection (light)")
    table = live_table()
    rng = random.Random(63)
    h, w = 6, 7
    a, d, T, his, ts = random_scene(rng, h, w)
    la, ld = light_scene(rng, h, w)
    sc = (a, d, T, his, ts)

    def raises(tag, light, **kw):
        for be in ("cpu", "gpu"):
            lkw = light_arrays(light, h, w)
            for k in LIGHT:
                lkw[k][...] = GARBAGE
            heat = [np.full((h, w), GARBAGE, dtype=np.int64) for _ in range(4)]
            args = (as_i32(T), as_i32(a), as_i32(d), as_i32(his),
                    np.ascontiguousarray(np.asarray(ts) != 0), table)
            fn = (bp.RadiationSweep().run if be == "cpu" else bp.cuda_radiation_sweep_run)
            try:
                fn(*args, None, int(T_AMB_Q), 0, TRANSPORTS["shear"], 16, *heat,
                   **kw, **lkw)
                _fail(f"L3 {tag}: {be} accepted an illegal light scene")
            except (ValueError, TypeError):
                pass
            if any(np.any(p != GARBAGE) for p in heat) or \
                    any(np.any(lkw[k] != GARBAGE) for k in LIGHT):
                _fail(f"L3 {tag}: {be} touched the caller's planes on a rejection")

    bad = [[row[:] for row in p] for p in la], [[row[:] for row in p] for p in ld]
    bad[0][1][2][3], bad[1][1][2][3] = Q(0.9), Q(0.4)          # a > d on channel G
    raises("a > d on one channel", dict(la=bad[0], ld=bad[1]))
    bad2 = [[row[:] for row in p] for p in ld]
    bad2[2][0][0] = ONE + 1                                      # d > ONE
    raises("d > ONE", dict(la=la, ld=bad2))
    glowing = np.asarray(bp.LightEmissionTable().table()).copy()
    glowing[0, 0] = 1                                            # bucket 0 glows
    raises("a table that glows at room temperature", dict(la=la, ld=ld, table=glowing))
    raises("smoke columns without the gas group",
           dict(la=la, ld=ld, light_absorb_q=[[0, 0, 0]], light_glow_q=[[0, 0, 0]]))
    # overwrite: garbage outputs come back exact
    light = dict(la=la, ld=ld)
    c = run("cpu", sc, table, light=light)
    g = run("gpu", sc, table, light=light, garbage=True)
    same("L3 overwrite", c, g, ALL)
    print("  4 illegal light scenes rejected by both, planes untouched; overwrite exact")


def part4_batch() -> None:
    print("PART L4 — the launch core, N = 3 envs with light")
    import cupy as cp
    SLOTS = int(bp.RADIATION_SWEEP_CNT_SLOTS)
    table = live_table()
    rng = random.Random(64)
    h, w, N, n_ord = 11, 17, 3, 16
    envs = []
    for e in range(N):
        a, d, T, his, ts = random_scene(rng, h, w)
        la, ld = light_scene(rng, h, w)
        envs.append(((a, d, T, his, ts), dict(la=la, ld=ld)))
    # env 1 carries an ILLEGAL light cell in the second run
    runs = [("legal", None), ("env 1 illegal", 1)]
    lt = np.asarray(bp.LightEmissionTable().table(), dtype=np.int64)
    for tag, bad_env in runs:
        stack = lambda f, dt: cp.asarray(np.stack([f(s) for s in envs]).astype(dt))  # noqa: E731
        lplanes = []
        for e, (_sc, lg) in enumerate(envs):
            kw = light_arrays(lg, h, w)
            if bad_env == e:
                kw["light_atten_q"][4, 4, 0] = Q(0.9)
                kw["dyn_light_atten_q"][4, 4, 0] = Q(0.3)
            lplanes.append(kw)
        d_T = stack(lambda s: as_i32(s[0][2]), np.int32)
        d_a = stack(lambda s: as_i32(s[0][0]), np.int32)
        d_d = stack(lambda s: as_i32(s[0][1]), np.int32)
        d_his = stack(lambda s: as_i32(s[0][3]), np.int32)
        d_ts = stack(lambda s: np.asarray(s[0][4]) != 0, np.bool_)
        d_vac = cp.zeros((N, h, w), dtype=cp.bool_)
        d_etab = cp.asarray(np.asarray(table.table(), dtype=np.int64))
        d_vl = cp.asarray(np.full(N, -1, dtype=np.int64))
        d_kl = cp.asarray(np.asarray([0, K_LEAK, Q(0.3)], dtype=np.int32))
        d_ta = cp.asarray(np.full(N, int(T_AMB_Q), dtype=np.int32))
        d_out = cp.empty((N, n_ord, h, w), dtype=cp.int64)
        d_ambm, d_ex = cp.empty((N, h, w), dtype=cp.int64), cp.empty((N, h, w), dtype=cp.int64)
        d_f = cp.empty((N, h, w), dtype=cp.int32)
        d_ae, d_de = cp.empty((N, h, w), dtype=cp.int32), cp.empty((N, h, w), dtype=cp.int32)
        d_rad = [cp.full((N, h, w), GARBAGE, dtype=cp.int64) for _ in range(4)]
        d_cnt = cp.zeros((N, SLOTS), dtype=cp.int64)
        d_la = cp.asarray(np.stack([k["light_atten_q"] for k in lplanes]))
        d_ld = cp.asarray(np.stack([k["dyn_light_atten_q"] for k in lplanes]))
        d_lt = cp.asarray(lt)
        d_lout = cp.empty((N, n_ord, h, w, 3), dtype=cp.int64)
        d_lem = cp.empty((N, h, w, 3), dtype=cp.int64)
        d_ldd = cp.empty((N, h, w, 3), dtype=cp.int32)
        d_lg = cp.empty((N, h, w, 3), dtype=cp.int32)
        d_lq = cp.full((N, h, w, 3), GARBAGE, dtype=cp.int64)
        d_lf = cp.full((N, h, w, 2), GARBAGE, dtype=cp.int64)
        d_lgl = cp.full((N, h, w, 3), GARBAGE, dtype=cp.int64)
        light = dict(d_light_atten_q=d_la.data.ptr, d_dyn_light_atten_q=d_ld.data.ptr,
                     d_l_table=d_lt.data.ptr, transport=int(bp.RadiationSweep.STEP),
                     d_l_outflow=d_lout.data.ptr, d_l_emit=d_lem.data.ptr,
                     d_l_d=d_ldd.data.ptr, d_l_g=d_lg.data.ptr,
                     d_light_q=d_lq.data.ptr, d_light_flux_q=d_lf.data.ptr,
                     d_light_glow=d_lgl.data.ptr)
        launches = bp.cuda_radiation_sweep_resident(
            N, h, w, d_T.data.ptr, d_a.data.ptr, d_d.data.ptr, d_his.data.ptr,
            d_ts.data.ptr, 0, d_vac.data.ptr, d_etab.data.ptr, int(table.fine_bits),
            d_vl.data.ptr, d_kl.data.ptr, d_ta.data.ptr, 0, 0, 0, 0, 0, 0,
            TRANSPORTS["shear"], n_ord, True,
            d_out.data.ptr, d_ambm.data.ptr, d_ex.data.ptr, d_f.data.ptr,
            d_ae.data.ptr, d_de.data.ptr, *[p.data.ptr for p in d_rad], d_cnt.data.ptr,
            light=light)
        cp.cuda.Device().synchronize()
        if launches != bp.cuda_radiation_sweep_launch_count(
                TRANSPORTS["shear"], n_ord, h, w, True):
            _fail(f"L4 {tag}: launch count {launches}")
        cnt = d_cnt.get()
        lq, lf, lgl = d_lq.get(), d_lf.get(), d_lgl.get()
        for e in range(N):
            if bad_env == e:
                if int(cnt[e][int(bp.RS_SLOT_BAD_LIGHT_EXTINCTION)]) != 1:
                    _fail(f"L4 {tag}: the illegal env counted "
                          f"{int(cnt[e][int(bp.RS_SLOT_BAD_LIGHT_EXTINCTION)])} bad cells")
                if np.any(lq[e] != GARBAGE) or any(np.any(p.get()[e] != GARBAGE) for p in d_rad):
                    _fail(f"L4 {tag}: the rejected env's planes were touched")
                continue
            if int(cnt[e][int(bp.RS_SLOT_BAD_LIGHT_EXTINCTION)]):
                _fail(f"L4 {tag}: a LEGAL env {e} was charged a light violation")
            sc, lg = envs[e]
            c = run("cpu", sc, table, k_q=[0, K_LEAK, Q(0.3)][e], light=lg)
            books = {
                "emit": tuple(int(v) for v in cnt[e][int(bp.RS_SLOT_LIGHT_EMIT):][:3]),
                "absorb": tuple(int(v) for v in cnt[e][int(bp.RS_SLOT_LIGHT_ABSORB):][:3]),
                "ring_in": tuple(int(v) for v in cnt[e][int(bp.RS_SLOT_LIGHT_RING_IN):][:3]),
                "ring_out": tuple(int(v) for v in cnt[e][int(bp.RS_SLOT_LIGHT_RING_OUT):][:3]),
                "min_stream": int(cnt[e][int(bp.RS_SLOT_LIGHT_MIN_STREAM)]),
                "max_stream": int(cnt[e][int(bp.RS_SLOT_LIGHT_MAX_STREAM)])}
            g = {"light_q": lq[e], "light_flux_q": lf[e], "light_glow": lgl[e],
                 "books": books, "rad_net": d_rad[0].get()[e],
                 "rad_fluence": d_rad[3].get()[e]}
            same(f"L4 {tag} env {e}", c, g,
                 ("light_q", "light_flux_q", "light_glow", "books", "rad_net",
                  "rad_fluence"))
        if bad_env is None and np.array_equal(lq[0], lq[1]):
            _fail("L4: two envs produced the same light_q -- the batch is vacuous")
    print("  3 envs each equal to its own CPU run; an illegal light env masked alone")


def part5_live() -> None:
    print("PART L5 — the live conductor, light REQUESTED, radiation backend flipping")
    from cuda_radiation_sweep_check import _ALL_BACKENDS, _burning_playground
    from simulation import physics_runner

    def only_radiation(on):
        physics_runner.set_residency(False)
        for name in _ALL_BACKENDS:
            getattr(bp, name)(False)
        bp.set_radiation_backend(bool(on))

    s_cpu, pick = _burning_playground()
    s_gpu, _ = _burning_playground()
    for s in (s_cpu, s_gpu):
        s.physics_runner.engine.light_requested = True
    n_ticks, lit, calls = 20, 0, 0
    for t in range(n_ticks):
        only_radiation(False)
        s_cpu.set_paused(False)
        s_cpu.step()
        c0 = int(bp.radiation_sweep_cuda_calls())
        only_radiation(True)
        s_gpu.set_paused(False)
        s_gpu.step()
        calls += int(bp.radiation_sweep_cuda_calls()) - c0
        only_radiation(False)
        gc, gg = s_cpu.gmap, s_gpu.gmap
        bad = 0
        for k, v in vars(gc).items():
            if isinstance(v, np.ndarray):
                eq = (np.array_equal(v, getattr(gg, k), equal_nan=True)
                      if v.dtype.kind == "f" else np.array_equal(v, getattr(gg, k)))
                if not eq:
                    bad += 1
                    _fail(f"L5 tick {t}: gmap.{k} differs")
        bc = s_cpu.physics_runner.engine.radiation.light_books
        bg = s_gpu.physics_runner.engine.radiation.light_books
        if dict(bc) != dict(bg):
            bad += 1
            _fail(f"L5 tick {t}: the engine's light books cpu={dict(bc)} gpu={dict(bg)}")
        lit += int(np.any(gc.light_q != 0))
        if bad >= 4:
            break
    if calls != n_ticks:
        _fail(f"L5: the GPU sweep ran {calls} times in {n_ticks} ticks")
    if lit == 0:
        _fail("L5: light_q never non-zero -- the live light comparison is vacuous")
    print(f"  {n_ticks} ticks, every GameMap array and the light books tol 0; "
          f"{lit} lit ticks; GPU sweep calls {calls}; fire at {pick}")


def main() -> int:
    try:
        dev = bp.cuda_device_info() if hasattr(bp, "cuda_device_info") else ""
        print(f"device: {dev}")
    except Exception:                                   # noqa: BLE001
        pass
    for part in (part1_matrix, part2_shapes, part3_ingress, part4_batch, part5_live):
        try:
            part()
        except Exception as exc:                        # noqa: BLE001
            import traceback
            traceback.print_exc()
            _fail(f"{part.__name__} raised {type(exc).__name__}: {exc}")
    print(f"\nLS_RESULT: {'PASS' if not _FAILS else 'FAIL'} ({len(_FAILS)} failures)")
    return 0 if not _FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
