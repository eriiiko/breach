"""GPU == CPU at tol 0 for THE FIRE'S LIGHT FLICKER (graphics G3 F4b; design
docs/g3_fire_design_2026-10-23.md section 6.3, RULED D). Run by
tests/test_cuda_radiation_sweep_emit_mult.py through tests/cuda_harness.py (a
CUDA subprocess: the CUDA .pyd is never imported into pytest).

PROPERTY: with emission-multiplier rows riding the light group (beside cones and
a sky), the CUDA twin -- the per-call path `cuda_radiation_sweep_run`, the
(N, h, w) launch core `cuda_radiation_sweep_resident` with a per-env multiplier
plane, and the live conductor with light REQUESTED and the runner's flicker
rows set -- computes the SAME integers as `RadiationSweep::run` on every light
plane, the twelve light books and the light telemetry; heat stays bit-identical
to the flicker-off sweep on both backends; the door refuses an illegal row on
both without touching a plane.

BREAKS IF: the .cu scales the emission at a different point of the chain than
run() (a different floor), scales the cone injection, books the unscaled
emission, indexes an env's multiplier plane wrong, or builds the plane with a
second scatter (the host builder RadiationSweep::build_emit_mult is the one).

  PART M1 the matrix: random scenes x rows (every factor kind) x S16/S12 x heat
          shear/step x light step/shear x smoke on/off, GPU vs CPU tol 0, heat
          vs the flicker off.
  PART M2 ingress: an illegal row is refused by both, planes untouched.
  PART M3 the launch core at N = 3: per-env multiplier planes, each env equal
          to its own CPU run.
  PART M4 the live conductor, light REQUESTED with flicker rows set on the
          runner, only the radiation backend flipping: every GameMap array and
          the light books tol 0 per tick, and the flicker moving light_q but no
          other array against a flicker-off twin.

Prints ``LM_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import random
import sys

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness puts it first)

from _radiation_sweep_harness import (  # noqa: E402
    ONE, R, T_AMB_Q, TRANSPORTS, light_arrays, live_table, random_gas, random_scene)
import cuda_radiation_sweep_light_check as L  # noqa: E402
from cuda_radiation_sweep_cones_check import random_cones, random_sky  # noqa: E402

Q = R.quant
K_LEAK = Q(0.10)
MAX = R.LIGHT_EMIT_MULT_MAX
_FAILS = L._FAILS


def _fail(msg):
    L._fail(msg)


def hot(rng, sc, h, w, la, ld):
    """Raise five cells into the visible range with an emitting material, so
    the multiplier has thermal light to scale. Returns their cells."""
    T = sc[2]
    cells = [(rng.randrange(h), rng.randrange(w)) for _ in range(5)]
    for (y, x) in cells:
        T[y][x] = rng.choice([1100, 1263, 1500, 2200]) << 16
        for c in range(3):
            la[c][y][x] = ld[c][y][x] = Q(0.8)
    return cells


def rows_for(rng, cells, h, w, extra=4):
    seen, out = set(), []
    vals = [0, ONE, MAX, Q(0.7), Q(1.3)]
    pool = list(cells) + [(rng.randrange(h), rng.randrange(w)) for _ in range(extra)]
    for i, (y, x) in enumerate(pool):
        if (y, x) in seen:
            continue
        seen.add((y, x))
        out.append((y, x, vals[i] if i < len(vals) else rng.randrange(MAX + 1)))
    return out


def part1_matrix():
    print("PART M1 — flicker matrix, GPU vs CPU, tol 0; heat vs flicker OFF")
    table = live_table()
    n = 0
    for seed in (1, 2):
        rng = random.Random(20261023 + seed)
        h, w = rng.choice([(7, 9), (9, 11), (12, 8)])
        a, d, T, his, ts = random_scene(rng, h, w)
        la, ld = L.light_scene(rng, h, w)
        sc = (a, d, T, his, ts)
        cells = hot(rng, sc, h, w, la, ld)
        gas = random_gas(rng, h, w)
        lab, lgl = L.light_cols(rng, len(gas[0]))
        cones = random_cones(rng, h, w, 3)
        rows = rows_for(rng, cells, h, w)
        for n_ord in (16, 12):
            sky = random_sky(rng, n_ord, "random")
            for htr in ("shear", "step"):
                for ltr in ("step", "shear"):
                    for smoke in (False, True):
                        base = dict(la=la, ld=ld, transport=ltr, cones=cones, sky=sky)
                        gkw = {}
                        if smoke:
                            base.update(light_absorb_q=lab, light_glow_q=lgl)
                            gkw = dict(gas=gas)
                        light = dict(base, emit_mult=rows)
                        tag = f"M1 s{seed} S{n_ord} {htr}/{ltr} smoke={smoke}"
                        c = L.run("cpu", sc, table, transport=htr, n_ord=n_ord,
                                  k_q=K_LEAK, light=light, **gkw)
                        g = L.run("gpu", sc, table, transport=htr, n_ord=n_ord,
                                  k_q=K_LEAK, light=light, garbage=True, **gkw)
                        L.same(tag, c, g, L.ALL)
                        c0 = L.run("cpu", sc, table, transport=htr, n_ord=n_ord,
                                   k_q=K_LEAK, light=base, **gkw)
                        g0 = L.run("gpu", sc, table, transport=htr, n_ord=n_ord,
                                   k_q=K_LEAK, light=base, **gkw)
                        L.same(tag + " HEAT cpu flicker on/off", c0, c, L.ALL_HEAT)
                        L.same(tag + " HEAT gpu flicker on/off", g0, g, L.ALL_HEAT)
                        if np.array_equal(c0["light_q"], c["light_q"]):
                            _fail(f"{tag}: the rows moved no light -- vacuous")
                        bk = c["books"]
                        for ch in range(3):
                            if bk["emit"][ch] + bk["ring_in"][ch] != \
                                    bk["absorb"][ch] + bk["ring_out"][ch]:
                                _fail(f"{tag}: the light books do not close, ch {ch}")
                        n += 1
    print(f"  {n} configurations: every heat + light plane, the books and both "
          f"telemetries tol 0; heat identical with the flicker off; light moved")


def part2_ingress():
    print("PART M2 — the emit_mult door on both backends")
    import _radiation_sweep_harness as H
    table = live_table()
    rng = random.Random(77)
    h, w = 6, 7
    a, d, T, his, ts = random_scene(rng, h, w)
    la, ld = L.light_scene(rng, h, w)
    bad = [("off-grid row", [(h, 0, ONE)]), ("negative x", [(0, -1, ONE)]),
           ("negative factor", [(0, 0, -1)]), ("above max", [(0, 0, MAX + 1)]),
           ("two rows, one cell", [(1, 1, ONE), (1, 1, Q(0.5))])]
    for tag, rows in bad:
        for be in ("cpu", "gpu"):
            lkw = light_arrays(dict(la=la, ld=ld, emit_mult=rows), h, w)
            for k in L.LIGHT:
                lkw[k][...] = L.GARBAGE
            heat = [np.full((h, w), L.GARBAGE, dtype=np.int64) for _ in range(4)]
            args = (H.as_i32(T), H.as_i32(a), H.as_i32(d), H.as_i32(his),
                    np.ascontiguousarray(np.asarray(ts) != 0), table)
            fn = bp.RadiationSweep().run if be == "cpu" else bp.cuda_radiation_sweep_run
            try:
                fn(*args, None, int(T_AMB_Q), 0, TRANSPORTS["shear"], 16, *heat, **lkw)
                _fail(f"M2 {tag}: {be} accepted it")
            except (ValueError, TypeError):
                pass
            if any(np.any(p != L.GARBAGE) for p in heat) or \
                    any(np.any(lkw[k] != L.GARBAGE) for k in L.LIGHT):
                _fail(f"M2 {tag}: {be} touched the caller's planes on a rejection")
    print(f"  {len(bad)} illegal inputs refused by both, planes untouched")


def part3_batch():
    print("PART M3 — the launch core, N = 3 envs, per-env multiplier planes")
    try:
        import cupy as cp
    except ImportError:
        _fail("M3: cupy is not importable in the CUDA interpreter -- the launch "
              "core cannot be driven (put cupy on its path)")
        return
    import _radiation_sweep_harness as H
    SLOTS = int(bp.RADIATION_SWEEP_CNT_SLOTS)
    table = live_table()
    rng = random.Random(78)
    h, w, N, n_ord = 11, 17, 3, 16
    envs = []
    for e in range(N):
        sc = random_scene(rng, h, w)
        la, ld = L.light_scene(rng, h, w)
        cells = hot(rng, sc, h, w, la, ld)
        envs.append((sc, dict(la=la, ld=ld, emit_mult=rows_for(rng, cells, h, w, 2 + e))))
    plane = np.full((N, h, w), ONE, dtype=np.int32)       # the host builder's shape
    for e, (_sc, lg) in enumerate(envs):
        for (y, x, m) in lg["emit_mult"]:
            plane[e, y, x] = m
    stack = lambda f, dt: cp.asarray(np.stack([f(s) for s in envs]).astype(dt))  # noqa: E731
    lplanes = [light_arrays(dict(la=lg["la"], ld=lg["ld"]), h, w) for _s, lg in envs]
    d_T = stack(lambda s: H.as_i32(s[0][2]), np.int32)
    d_a = stack(lambda s: H.as_i32(s[0][0]), np.int32)
    d_d = stack(lambda s: H.as_i32(s[0][1]), np.int32)
    d_his = stack(lambda s: H.as_i32(s[0][3]), np.int32)
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
    d_rad = [cp.full((N, h, w), L.GARBAGE, dtype=cp.int64) for _ in range(4)]
    d_cnt = cp.zeros((N, SLOTS), dtype=cp.int64)
    d_la = cp.asarray(np.stack([k["light_atten_q"] for k in lplanes]))
    d_ld = cp.asarray(np.stack([k["dyn_light_atten_q"] for k in lplanes]))
    d_lt = cp.asarray(np.asarray(bp.LightEmissionTable().table(), dtype=np.int64))
    d_lout = cp.empty((N, n_ord, h, w, 3), dtype=cp.int64)
    d_lem = cp.empty((N, h, w, 3), dtype=cp.int64)
    d_ldd = cp.empty((N, h, w, 3), dtype=cp.int32)
    d_lg = cp.empty((N, h, w, 3), dtype=cp.int32)
    d_lq = cp.full((N, h, w, 3), L.GARBAGE, dtype=cp.int64)
    d_lf = cp.full((N, h, w, 2), L.GARBAGE, dtype=cp.int64)
    d_lgl = cp.full((N, h, w, 3), L.GARBAGE, dtype=cp.int64)
    d_em = cp.asarray(plane)
    light = dict(d_light_atten_q=d_la.data.ptr, d_dyn_light_atten_q=d_ld.data.ptr,
                 d_l_table=d_lt.data.ptr, transport=int(bp.RadiationSweep.STEP),
                 d_l_outflow=d_lout.data.ptr, d_l_emit=d_lem.data.ptr,
                 d_l_d=d_ldd.data.ptr, d_l_g=d_lg.data.ptr,
                 d_light_q=d_lq.data.ptr, d_light_flux_q=d_lf.data.ptr,
                 d_light_glow=d_lgl.data.ptr, d_emit_mult=d_em.data.ptr)
    bp.cuda_radiation_sweep_resident(
        N, h, w, d_T.data.ptr, d_a.data.ptr, d_d.data.ptr, d_his.data.ptr,
        d_ts.data.ptr, 0, d_vac.data.ptr, d_etab.data.ptr, int(table.fine_bits),
        d_vl.data.ptr, d_kl.data.ptr, d_ta.data.ptr, 0, 0, 0, 0, 0, 0,
        TRANSPORTS["shear"], n_ord, True,
        d_out.data.ptr, d_ambm.data.ptr, d_ex.data.ptr, d_f.data.ptr,
        d_ae.data.ptr, d_de.data.ptr, *[p.data.ptr for p in d_rad], d_cnt.data.ptr,
        light=light)
    cp.cuda.Device().synchronize()
    cnt = d_cnt.get()
    lq, lf, lgl = d_lq.get(), d_lf.get(), d_lgl.get()
    for e in range(N):
        sc, lg = envs[e]
        c = L.run("cpu", sc, table, k_q=[0, K_LEAK, Q(0.3)][e], light=lg)
        books = {
            "emit": tuple(int(v) for v in cnt[e][int(bp.RS_SLOT_LIGHT_EMIT):][:3]),
            "absorb": tuple(int(v) for v in cnt[e][int(bp.RS_SLOT_LIGHT_ABSORB):][:3]),
            "ring_in": tuple(int(v) for v in cnt[e][int(bp.RS_SLOT_LIGHT_RING_IN):][:3]),
            "ring_out": tuple(int(v) for v in cnt[e][int(bp.RS_SLOT_LIGHT_RING_OUT):][:3]),
            "min_stream": int(cnt[e][int(bp.RS_SLOT_LIGHT_MIN_STREAM)]),
            "max_stream": int(cnt[e][int(bp.RS_SLOT_LIGHT_MAX_STREAM)])}
        g = {"light_q": lq[e], "light_flux_q": lf[e], "light_glow": lgl[e], "books": books,
             "rad_net": d_rad[0].get()[e], "rad_fluence": d_rad[3].get()[e]}
        L.same(f"M3 env {e}", c, g, ("light_q", "light_flux_q", "light_glow", "books",
                                      "rad_net", "rad_fluence"))
    print("  3 envs, each with its own multiplier plane, each equal to its own CPU run")


def part4_live():
    print("PART M4 — the live conductor with flicker rows, light REQUESTED, backend flipping")
    from cuda_radiation_sweep_check import _ALL_BACKENDS, _burning_playground
    from simulation import physics_runner

    def only_radiation(on):
        physics_runner.set_residency(False)
        for name in _ALL_BACKENDS:
            getattr(bp, name)(False)
        bp.set_radiation_backend(bool(on))

    s_cpu, pick = _burning_playground()
    s_gpu, _ = _burning_playground()
    s_off, _ = _burning_playground()            # the flicker-off twin (CPU)
    rng = random.Random(79)
    for s in (s_cpu, s_gpu, s_off):
        s.physics_runner.engine.light_requested = True
    n_ticks, moved, calls = 12, 0, 0
    for t in range(n_ticks):
        g = s_cpu.gmap
        hot_cells = np.argwhere(np.asarray(g.fire) > 0)
        if len(hot_cells) == 0:
            hot_cells = np.argwhere(np.asarray(g.temperature) == np.asarray(g.temperature).max())
        rows = np.asarray([[int(y), int(x), rng.randrange(Q(0.6), Q(1.4))]
                           for (y, x) in hot_cells[:64]], dtype=np.int64)
        for s in (s_cpu, s_gpu):
            s.physics_runner.set_light_emit_mult(rows)
        only_radiation(False)
        for s in (s_cpu, s_off):
            s.set_paused(False)
            s.step()
        c0 = int(bp.radiation_sweep_cuda_calls())
        only_radiation(True)
        s_gpu.set_paused(False)
        s_gpu.step()
        calls += int(bp.radiation_sweep_cuda_calls()) - c0
        only_radiation(False)
        gc, gg, go = s_cpu.gmap, s_gpu.gmap, s_off.gmap
        bad = 0
        for k, v in vars(gc).items():
            if not isinstance(v, np.ndarray):
                continue
            eq = (np.array_equal(v, getattr(gg, k), equal_nan=True)
                  if v.dtype.kind == "f" else np.array_equal(v, getattr(gg, k)))
            if not eq:
                bad += 1
                _fail(f"M4 tick {t}: gmap.{k} differs cpu/gpu")
            if k in ("light_q", "light_flux_q", "light_glow"):
                continue
            eq = (np.array_equal(v, getattr(go, k), equal_nan=True)
                  if v.dtype.kind == "f" else np.array_equal(v, getattr(go, k)))
            if not eq:
                bad += 1
                _fail(f"M4 tick {t}: the flicker moved gmap.{k}")
        moved += int(not np.array_equal(gc.light_q, go.light_q))
        bc = s_cpu.physics_runner.engine.radiation.light_books
        bg = s_gpu.physics_runner.engine.radiation.light_books
        if dict(bc) != dict(bg):
            bad += 1
            _fail(f"M4 tick {t}: light books cpu={dict(bc)} gpu={dict(bg)}")
        if bad >= 4:
            break
    if calls != n_ticks:
        _fail(f"M4: the GPU sweep ran {calls} times in {n_ticks} ticks")
    if moved == 0:
        _fail("M4: the flicker never moved light_q -- vacuous")
    print(f"  {n_ticks} ticks: every GameMap array and the light books tol 0 cpu/gpu; "
          f"the flicker moved light_q on {moved} ticks and no other array; fire at {pick}")


def main() -> int:
    for part in (part1_matrix, part2_ingress, part3_batch, part4_live):
        try:
            part()
        except Exception as exc:                        # noqa: BLE001
            import traceback
            traceback.print_exc()
            _fail(f"{part.__name__} raised {type(exc).__name__}: {exc}")
    print(f"\nLM_RESULT: {'PASS' if not _FAILS else 'FAIL'} ({len(_FAILS)} failures)")
    return 0 if not _FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
