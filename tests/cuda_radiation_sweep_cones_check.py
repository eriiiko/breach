"""GPU == CPU at tol 0 for the CONE EMITTERS and THE SKY (ray-engine-v2 P6b;
docs/ray_engine_v2_p6b_lit_world_brief_2026-09-25.md section 3.2). Run by
tests/test_cuda_radiation_sweep_cones.py through tests/cuda_harness.py (a CUDA
subprocess: the CUDA .pyd is never imported into pytest).

PROPERTY: with cone emitters and a sky riding the light group, the CUDA twin --
the per-call path `cuda_radiation_sweep_run`, the (N, h, w) launch core
`cuda_radiation_sweep_resident`, and the live conductor with light REQUESTED --
computes the SAME integers as `RadiationSweep::run` on every light plane, the
twelve light books and the light telemetry, and leaves heat bit-identical to the
light-off sweep on both backends.

BREAKS IF: the .cu reads the sky for one upwind share only, injects a cone after
the cell's absorption, books the injection outside the emission slot, indexes a
slot's ordinate or an env's sky wrong, or the host-built injection disagrees with
the CPU's (the projection has ONE implementation, RadiationSweep::cone_emission;
a second one in the .cu would be the bug).

  PART C1 the matrix: random scenes x cones (every beam kind, two on one cell) x
          sky (overcast / sun / random / none) x S16/S12 x heat shear/step x
          light step/shear x smoke on/off, GPU vs CPU tol 0, heat vs light off.
  PART C2 ingress: an illegal cone or sky is refused by both, planes untouched.
  PART C3 the launch core at N = 3: per-env sky and cones (slots numbered
          across envs), each env equal to its own CPU run.
  PART C4 the live conductor, light REQUESTED with cones and a sky set on the
          runner, only the radiation backend flipping: every GameMap array and
          the engine's light books tol 0, per tick; and relight() equal too.

Prints ``LC_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import random
import sys

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness puts it first)

from _radiation_sweep_harness import (  # noqa: E402
    ONE, R, T_AMB_Q, TRANSPORTS, light_arrays, live_table, random_gas, random_scene)
import cuda_radiation_sweep_light_check as L  # noqa: E402

Q = R.quant
K_LEAK = Q(0.10)
TURN = R.CONE_TURN
_FAILS = L._FAILS          # one failure list: L's helpers append to it


def _fail(msg):
    L._fail(msg)


def random_cones(rng, h, w, n):
    beams = [(rng.randrange(TURN), 0), (rng.randrange(TURN), rng.randrange(1, 3000)),
             (rng.randrange(TURN), rng.randrange(3000, 40000)), (65000, 3000),
             (rng.randrange(TURN), TURN), (0, 10923)]
    out = []
    for i in range(n):
        cq, sq = beams[i % len(beams)]
        y, x = (rng.randrange(h), rng.randrange(w)) if i != 1 else (out[0][0], out[0][1])
        out.append((y, x, tuple(rng.randrange(1 << 39) for _ in range(3)), cq, sq))
    return out


def random_sky(rng, n_ord, kind):
    if kind is None:
        return None
    if kind == "overcast":
        return [[5 << 30, 5 << 30, 6 << 30]] * n_ord
    if kind == "sun":
        return R.cone_emission([3 << 34] * 3, rng.randrange(TURN), 4000, n_ord)
    return [[rng.randrange(1 << 36) for _ in range(3)] for _ in range(n_ord)]


def part1_matrix():
    print("PART C1 — cones + sky matrix, GPU vs CPU, tol 0; heat vs light OFF")
    table = live_table()
    n = 0
    for seed in (1, 2):
        rng = random.Random(20260929 + seed)
        h, w = rng.choice([(7, 9), (9, 11), (12, 8)])
        a, d, T, his, ts = random_scene(rng, h, w)
        la, ld = L.light_scene(rng, h, w)
        gas = random_gas(rng, h, w)
        lab, lgl = L.light_cols(rng, len(gas[0]))
        cones = random_cones(rng, h, w, 6)
        sc = (a, d, T, his, ts)
        for n_ord in (16, 12):
            for sky_kind in ("overcast", "sun", "random", None):
                sky = random_sky(rng, n_ord, sky_kind)
                for htr in ("shear", "step"):
                    for ltr in ("step", "shear"):
                        for smoke in (False, True):
                            light = dict(la=la, ld=ld, transport=ltr, cones=cones, sky=sky)
                            gkw = {}
                            if smoke:
                                light.update(light_absorb_q=lab, light_glow_q=lgl)
                                gkw = dict(gas=gas)
                            tag = f"C1 s{seed} S{n_ord} sky={sky_kind} {htr}/{ltr} smoke={smoke}"
                            c = L.run("cpu", sc, table, transport=htr, n_ord=n_ord,
                                      k_q=K_LEAK, light=light, **gkw)
                            g = L.run("gpu", sc, table, transport=htr, n_ord=n_ord,
                                      k_q=K_LEAK, light=light, garbage=True, **gkw)
                            L.same(tag, c, g, L.ALL)
                            if htr == "shear" and ltr == "step":
                                c0 = L.run("cpu", sc, table, transport=htr, n_ord=n_ord,
                                           k_q=K_LEAK, **gkw)
                                g0 = L.run("gpu", sc, table, transport=htr, n_ord=n_ord,
                                           k_q=K_LEAK, **gkw)
                                L.same(tag + " HEAT cpu on/off", c0, c, L.ALL_HEAT)
                                L.same(tag + " HEAT gpu on/off", g0, g, L.ALL_HEAT)
                            bk = c["books"]
                            for ch in range(3):
                                if bk["emit"][ch] + bk["ring_in"][ch] != \
                                        bk["absorb"][ch] + bk["ring_out"][ch]:
                                    _fail(f"{tag}: the light books do not close, ch {ch}")
                            if sky_kind in ("overcast", "random") and \
                                    not all(v > 0 for v in bk["ring_in"]):
                                _fail(f"{tag}: a sky with no ring_in -- vacuous")
                            n += 1
    print(f"  {n} configurations: every heat + light plane, the books and both "
          f"telemetries tol 0; heat identical with light off")


def part2_ingress():
    print("PART C2 — the cone / sky door on both backends")
    table = live_table()
    rng = random.Random(65)
    h, w = 6, 7
    a, d, T, his, ts = random_scene(rng, h, w)
    la, ld = L.light_scene(rng, h, w)
    B = R.LIGHT_CONE_BUDGET
    bad = [("off-grid cone", dict(cones=[(h, 0, (1, 1, 1), 0, 0)])),
           ("negative rgb", dict(cones=[(0, 0, (-1, 1, 1), 0, 0)])),
           ("centre out of range", dict(cones=[(0, 0, (1, 1, 1), TURN, 0)])),
           ("spread out of range", dict(cones=[(0, 0, (1, 1, 1), 0, TURN + 1)])),
           ("budget + 1", dict(cones=[(0, 0, (B, 0, 0), 0, 0), (1, 1, (1, 0, 0), 0, 0)])),
           ("sky above max", dict(sky=[[R.LIGHT_SKY_MAX + 1, 0, 0]] + [[0] * 3] * 15)),
           ("negative sky", dict(sky=[[-1, 0, 0]] + [[0] * 3] * 15))]
    import _radiation_sweep_harness as H
    for tag, kw in bad:
        for be in ("cpu", "gpu"):
            lkw = light_arrays(dict(la=la, ld=ld, **kw), h, w)
            for k in L.LIGHT:
                lkw[k][...] = L.GARBAGE
            heat = [np.full((h, w), L.GARBAGE, dtype=np.int64) for _ in range(4)]
            args = (H.as_i32(T), H.as_i32(a), H.as_i32(d), H.as_i32(his),
                    np.ascontiguousarray(np.asarray(ts) != 0), table)
            fn = bp.RadiationSweep().run if be == "cpu" else bp.cuda_radiation_sweep_run
            try:
                fn(*args, None, int(T_AMB_Q), 0, TRANSPORTS["shear"], 16, *heat, **lkw)
                _fail(f"C2 {tag}: {be} accepted it")
            except (ValueError, TypeError):
                pass
            if any(np.any(p != L.GARBAGE) for p in heat) or \
                    any(np.any(lkw[k] != L.GARBAGE) for k in L.LIGHT):
                _fail(f"C2 {tag}: {be} touched the caller's planes on a rejection")
    print(f"  {len(bad)} illegal inputs refused by both, planes untouched")


def _injection(env_cones, h, w, n_ord):
    """(index (N, h, w) int32, slots (n_slots, n_ord, 3) int64): the per-cell
    sums of each env's cones, slots numbered across envs -- built from the
    ENGINE's own projection (bp.RadiationSweep.cone_emission), never a copy."""
    N = len(env_cones)
    idx = np.full((N, h, w), -1, dtype=np.int32)
    slots = []
    for e, cones in enumerate(env_cones):
        for (y, x, rgb, cq, sq) in cones:
            em = np.asarray(bp.RadiationSweep.cone_emission(list(rgb), cq, sq, n_ord),
                            dtype=np.int64)
            if idx[e, y, x] < 0:
                idx[e, y, x] = len(slots)
                slots.append(np.zeros((n_ord, 3), dtype=np.int64))
            slots[idx[e, y, x]] += em
    return idx, np.ascontiguousarray(np.stack(slots)) if slots else np.zeros((1, n_ord, 3), np.int64)


def part3_batch():
    print("PART C3 — the launch core, N = 3 envs, per-env cones and sky")
    import cupy as cp
    import _radiation_sweep_harness as H
    SLOTS = int(bp.RADIATION_SWEEP_CNT_SLOTS)
    table = live_table()
    rng = random.Random(66)
    h, w, N, n_ord = 11, 17, 3, 16
    envs = []
    for e in range(N):
        sc = random_scene(rng, h, w)
        la, ld = L.light_scene(rng, h, w)
        cones = random_cones(rng, h, w, 4 + e)
        sky = random_sky(rng, n_ord, ("overcast", "sun", "random")[e])
        envs.append((sc, dict(la=la, ld=ld, cones=cones, sky=sky)))
    stack = lambda f, dt: cp.asarray(np.stack([f(s) for s in envs]).astype(dt))  # noqa: E731
    idx, inj = _injection([lg["cones"] for _s, lg in envs], h, w, n_ord)
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
    d_sky = cp.asarray(np.stack([np.asarray(lg["sky"], dtype=np.int64) for _s, lg in envs]))
    d_idx = cp.asarray(idx)
    d_inj = cp.asarray(inj)
    light = dict(d_light_atten_q=d_la.data.ptr, d_dyn_light_atten_q=d_ld.data.ptr,
                 d_l_table=d_lt.data.ptr, transport=int(bp.RadiationSweep.STEP),
                 d_l_outflow=d_lout.data.ptr, d_l_emit=d_lem.data.ptr,
                 d_l_d=d_ldd.data.ptr, d_l_g=d_lg.data.ptr,
                 d_light_q=d_lq.data.ptr, d_light_flux_q=d_lf.data.ptr,
                 d_light_glow=d_lgl.data.ptr,
                 d_sky=d_sky.data.ptr, d_cone_index=d_idx.data.ptr,
                 d_cone_inj=d_inj.data.ptr)
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
        L.same(f"C3 env {e}", c, g, ("light_q", "light_flux_q", "light_glow", "books",
                                      "rad_net", "rad_fluence"))
    if np.array_equal(lq[0], lq[1]):
        _fail("C3: two envs produced the same light_q -- the batch is vacuous")
    print("  3 envs (overcast / sun / random sky, 4-6 cones each) each equal to its own CPU run")


def part4_live():
    print("PART C4 — the live conductor with cones + sky, light REQUESTED, backend flipping")
    from cuda_radiation_sweep_check import _ALL_BACKENDS, _burning_playground
    from simulation import physics_runner

    def only_radiation(on):
        physics_runner.set_residency(False)
        for name in _ALL_BACKENDS:
            getattr(bp, name)(False)
        bp.set_radiation_backend(bool(on))

    s_cpu, pick = _burning_playground()
    s_gpu, _ = _burning_playground()
    h, w = s_cpu.gmap.solid.shape
    rng = random.Random(67)
    sky = np.asarray(random_sky(rng, 16, "sun"), dtype=np.int64) + (1 << 30)
    for s in (s_cpu, s_gpu):
        s.physics_runner.engine.light_requested = True
        s.physics_runner.set_light_sky(sky)
    n_ticks, lit, calls = 12, 0, 0
    for t in range(n_ticks):
        cones = np.asarray([[rng.randrange(h), rng.randrange(w), 1 << 37, 1 << 36, 1 << 35,
                             rng.randrange(TURN), rng.choice([0, 12000, TURN])]
                            for _ in range(4)], dtype=np.int64)
        for s in (s_cpu, s_gpu):
            s.physics_runner.set_light_cones(cones)
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
                    _fail(f"C4 tick {t}: gmap.{k} differs")
        bc = s_cpu.physics_runner.engine.radiation.light_books
        bg = s_gpu.physics_runner.engine.radiation.light_books
        if dict(bc) != dict(bg):
            bad += 1
            _fail(f"C4 tick {t}: light books cpu={dict(bc)} gpu={dict(bg)}")
        lit += int(np.any(gc.light_q != 0))
        if bad >= 4:
            break
    # relight on both backends: the same planes, and no synced array moves
    snap = {k: v.copy() for k, v in vars(s_cpu.gmap).items() if isinstance(v, np.ndarray)}
    only_radiation(False)
    s_cpu.physics_runner.relight(s_cpu.gmap)
    only_radiation(True)
    s_gpu.physics_runner.relight(s_gpu.gmap)
    only_radiation(False)
    for k in ("light_q", "light_flux_q", "light_glow"):
        if not np.array_equal(getattr(s_cpu.gmap, k), getattr(s_gpu.gmap, k)):
            _fail(f"C4 relight: {k} cpu != gpu")
    for k, v in snap.items():
        if k in ("light_q", "light_flux_q", "light_glow"):
            continue
        if not (np.array_equal(v, getattr(s_cpu.gmap, k), equal_nan=True)
                if v.dtype.kind == "f" else np.array_equal(v, getattr(s_cpu.gmap, k))):
            _fail(f"C4 relight moved gmap.{k}")
    if calls != n_ticks:
        _fail(f"C4: the GPU sweep ran {calls} times in {n_ticks} ticks")
    if lit == 0:
        _fail("C4: light_q never non-zero -- vacuous")
    print(f"  {n_ticks} ticks, every GameMap array and the light books tol 0 ({lit} lit); "
          f"relight equal on both backends and moved no other array; fire at {pick}")


def main() -> int:
    for part in (part1_matrix, part2_ingress, part3_batch, part4_live):
        try:
            part()
        except Exception as exc:                        # noqa: BLE001
            import traceback
            traceback.print_exc()
            _fail(f"{part.__name__} raised {type(exc).__name__}: {exc}")
    print(f"\nLC_RESULT: {'PASS' if not _FAILS else 'FAIL'} ({len(_FAILS)} failures)")
    return 0 if not _FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
