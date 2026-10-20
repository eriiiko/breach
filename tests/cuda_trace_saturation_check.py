"""#88 P2 -- V6 on CUDA: the counted int32 saturation at every device site, with
its books, CPU == GPU at tolerance 0 -- runs inside the GPU subprocess
(``test_cuda_trace_saturation.py`` is the pytest wrapper).

Design: ``docs/smoke_unit_rescale_design_2026-10-20.md`` §5.3, §8, §10 V6 / V9;
critique A4 (once per cell on the net), A10 (the composite abuse is primary),
B7 (CUDA slots APPENDED, >= 0), C10 (every site per backend: 3b chained, 3b
resident, the soot mint).

  PART 1 -- 3b CHAINED: grenade spam under blast compression in a room near
            the ceiling (the CPU suite's composite case), CPU world vs the
            chained GPU EOS world, per tick: every gas plane, E, wind, T, P
            and EVERY trace book (vent, wipe, sink, decay, and P2's clip, mint,
            neg hits; the Python deposit / clip books) bit-identical; the §5.1
            identity closes on the GPU world with no residual; the saturation
            binds on the device (clip > 0); neg hits == 0.
  PART 2 -- 3b RESIDENT: the same on the device-resident tick.
  PART 3 -- the MINT: the CPU CombustionSolver.step vs cuda_combustion_step on
            a smoke plane at the ceiling: planes and the mint / clip / neg
            books identical, the mint attempt == the free-plane mint, clip > 0;
            and a real crate fire with the combustion backend on the GPU, in a
            room near the ceiling: the GPU world alone (the live GPU combustion
            carries no gas_energy, so CPU/GPU E/T differ outside #88), per tick
            the identity closes, the mint mints and clips, neg hits == 0.
  PART 4 -- the CUDA build's CPU path still reproduces the committed golden.

PROPERTY: GPU-int == CPU-int on the saturation and its books, at every device
site. BREAKS IF: a device site narrows with a wrap or an unbooked clamp,
saturates per face instead of on the net, books into a slot that is not reset
per call or not ADDED into the solver's books, the slot layout is inserted
rather than appended, or the runner stops handing ``eos=`` to the GPU
combustion.

Prints ``TRACE_SAT_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics`.
import breach_physics as bp

from simulation import gas_fixed
from simulation.gases import INERT_N2, N_GASES, O2, SMOKE

TRACE_MAX = gas_fixed.TRACE_CELL_MAX
CEIL_UNITS = TRACE_MAX / gas_fixed.TRACE_ONE
HULL, AIR, FURNITURE = 1, 0, 6

_EOS_SETTERS = ("set_sl_advection_backend", "set_bulk_flux_backend",
                "set_mg_solve_backend", "set_kick_compression_backend")


def _set_eos_backends(on: bool) -> None:
    for name in _EOS_SETTERS:
        getattr(bp, name)(bool(on))


def _residency(on: bool) -> None:
    from simulation import physics_runner
    physics_runner.set_residency(bool(on))


def _box(h, w):
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = HULL
    return tm


def _sim(tm, tile_m=0.333):
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    lvl = LevelData(name="trace_sat_cuda", version="2", path=Path("."),
                    tilemap=tm, tile_size_m=tile_m, diffuse_path=Path("."))
    sim = Simulation(lvl, seed=7, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _trace_ids(g):
    return [i for i in range(g.gases.n) if not bool(g.gases.conservative[i])]


def _books(sim):
    eos = sim.physics_runner.engine.eos
    g = sim.gmap
    return {
        "vent": list(eos.boundary_flux()),
        "wipe": list(eos.trace_wipe_sum()),
        "sink": list(eos.trace_sink_sum()),
        "decay": list(eos.trace_decay_sum()),
        "clip": list(eos.trace_clip_sum()),
        "mint": list(eos.trace_mint_sum()),
        "neg": list(eos.trace_neg_hits()),
        "py_deposit": list(g.trace_deposit_books),
        "py_clip": list(g.trace_clip_books),
    }


def _identity_residual(sim, before):
    """{gid: Δ S − (mint + deposit − vent − wipe − sink − decay − clip)}."""
    b = _books(sim)
    out = {}
    for gid in _trace_ids(sim.gmap):
        d = int(sim.gmap.gas[gid].astype(np.int64).sum()) - before[gid]
        net = (b["mint"][gid] + b["py_deposit"][gid] - b["vent"][gid]
               - b["wipe"][gid] - b["sink"][gid] - b["decay"][gid]
               - b["clip"][gid] - b["py_clip"][gid])
        out[gid] = d - net
    return out


def _fe_smoke(sim, y, x, amount, sid):
    from simulation.field_edit import EditMode, Falloff, FieldEdit, Region
    sim.edit_queue.enqueue(FieldEdit(
        field="gas", channel=SMOKE, region=Region.DISC, coords=(y, x, 3.0),
        amount=float(amount), mode=EditMode.ADD, falloff=Falloff.LINEAR,
        source_id=sid))


def _heat_blast(sim, y, x, sid, amount=3200.0, radius=5.0):
    from simulation.field_edit import EditMode, Falloff, FieldEdit, Region
    sim.edit_queue.enqueue(FieldEdit(
        field="heat", region=Region.DISC, coords=(y, x, float(radius)),
        amount=float(amount), mode=EditMode.ADD, falloff=Falloff.LINEAR,
        source_id=sid))


# ---------------------------------------------------------------------------
# scenarios: (tilemap, setup(sim), action(tick, sim), n_ticks, gpu_combustion)
# ---------------------------------------------------------------------------
def _scn_composite():
    tm = _box(22, 30)

    def setup(sim):
        sim.gmap.gas[SMOKE][1:-1, 1:-1] = int(0.9 * TRACE_MAX)

    def action(tick, sim):
        _fe_smoke(sim, 11, 6 + (tick % 18), 0.3 * CEIL_UNITS, 500 + tick)
        if tick % 5 == 0:
            _heat_blast(sim, 11, 15, 990 + tick)

    return tm, setup, action, 30, False


# The live GPU combustion (cuda_combustion_step) carries NO gas_energy -- it
# predates arc #54's ledger -- so a CPU-vs-GPU world with combustion on the GPU
# diverges on E/T by design, outside #88. The FIRE leg therefore checks the GPU
# world ALONE: the runner's eos= wiring on the CUDA branch, the identity, the
# range and the non-vacuity. CPU == GPU on the mint is PART 3's isolated pair.
def _scn_fire():
    from simulation.payloads import ignite_ring
    tm = _box(16, 20)
    tm[6:10, 8:12] = FURNITURE

    def setup(sim):
        open_ = ~sim.gmap.solid
        sim.gmap.gas[SMOKE][open_] = TRACE_MAX - 64
        ignite_ring(sim.gmap, sim.edit_queue, 8, 10, 2.5, 1.0)

    def action(tick, sim):
        return None

    return tm, setup, action, 36, True


def _run(name, builder, resident, parity=True):
    tag = f"{name}/{'resident' if resident else 'chained'}"
    tm, setup, action, n_ticks, gpu_comb = builder()
    cpu, gpu = _sim(tm), _sim(tm)
    setup(cpu)
    setup(gpu)
    gc, gg = cpu.gmap, gpu.gmap
    fields = ("gas", "gas_energy", "wind_x", "wind_y", "temperature", "atmosphere")
    for f in fields:
        assert np.array_equal(getattr(gc, f), getattr(gg, f)), f"{tag}: setup {f}"
    calls0 = bp.eos_resident_calls() if resident else bp.eos_step_cuda_calls()
    bad = 0
    tot = {"clip": 0, "mint": 0, "py_clip": 0}
    for tick in range(n_ticks):
        action(tick, cpu)
        action(tick, gpu)
        _residency(False)
        _set_eos_backends(False)
        bp.set_combustion_backend(False)
        cpu.step()
        before = {gid: int(gg.gas[gid].astype(np.int64).sum())
                  for gid in _trace_ids(gg)}
        _set_eos_backends(True)
        bp.set_combustion_backend(bool(gpu_comb))
        _residency(resident)
        gpu.step()
        _residency(False)
        _set_eos_backends(False)
        bp.set_combustion_backend(False)

        for f in (fields if parity else ()):
            a, b = getattr(gc, f), getattr(gg, f)
            if not np.array_equal(a, b):
                bad += 1
                idx = int(np.argmax(a != b))
                print(f"  [{tag}] tick {tick}: {f} {int(np.count_nonzero(a != b))} "
                      f"MISMATCH (first @ {idx}: cpu={a.flat[idx]} gpu={b.flat[idx]})")
        bc, bg = _books(cpu), _books(gpu)
        for key in (bc if parity else ()):
            if bc[key] != bg[key]:
                bad += 1
                print(f"  [{tag}] tick {tick}: books[{key}] cpu={bc[key]} gpu={bg[key]}")
        res = _identity_residual(gpu, before)
        if any(v != 0 for v in res.values()):
            bad += 1
            print(f"  [{tag}] tick {tick}: GPU identity residual {res}")
        for gid in _trace_ids(gg):
            p = gg.gas[gid]
            if int(p.max()) > TRACE_MAX or int(p.min()) < 0:
                bad += 1
                print(f"  [{tag}] tick {tick}: gas {gid} out of [0, MAX]")
            if bg["neg"][gid] != 0:
                bad += 1
                print(f"  [{tag}] tick {tick}: neg hits {bg['neg'][gid]} on gas {gid}")
            for key in ("clip", "mint", "py_clip", "py_deposit"):
                if bg[key][gid] < 0:
                    bad += 1
                    print(f"  [{tag}] tick {tick}: books[{key}] negative on gas {gid}")
        for key in tot:
            tot[key] += bg[key][SMOKE]
        if bad >= 10:
            print("  aborting after 10 divergences")
            break
    ok = bad == 0
    fired = (bp.eos_resident_calls() if resident else bp.eos_step_cuda_calls()) - calls0
    if fired < n_ticks:
        ok = False
        print(f"  [{tag}] the GPU dispatch fired {fired}/{n_ticks} ticks")
    if name == "COMPOSITE" and not (tot["clip"] > 0 and tot["py_clip"] > 0):
        ok = False
        print(f"  [{tag}] VACUOUS: the saturation never bound on device 3b "
              f"and FieldEdit ({tot})")
    if name == "FIRE" and not (tot["mint"] > 0 and tot["clip"] > 0):
        ok = False
        print(f"  [{tag}] VACUOUS: the GPU soot mint never minted and clipped ({tot})")
    if ok:
        what = ("bit-identical incl. every trace book; " if parity
                else "GPU world alone (see _scn_fire); ")
        print(f"  [{tag}] {n_ticks} ticks {what}GPU identity exact; "
              f"smoke totals {tot}")
    return ok


def part1_chained() -> bool:
    print("PART 1 -- 3b CHAINED: CPU == chained GPU, the saturation and its books:")
    return _run("COMPOSITE", _scn_composite, resident=False)


def part2_resident() -> bool:
    print("PART 2 -- 3b RESIDENT: CPU == the resident tick, the saturation and its books:")
    try:
        import cupy  # noqa: F401
    except Exception as exc:      # reported, never skipped
        print(f"  cupy unavailable ({exc}) -- the resident leg CANNOT be gated")
        return False
    return _run("COMPOSITE", _scn_composite, resident=True)


def _comb_scene(soot_trace_q):
    h = w = 5
    gas = np.zeros((N_GASES, h, w), dtype=np.int32)
    gas[INERT_N2][:] = gas_fixed.quantize_bulk_scalar(0.79)
    gas[O2][:] = gas_fixed.quantize_bulk_scalar(1.0)
    solid = np.zeros((h, w), dtype=bool)
    solid[2, 2] = True
    st = dict(gas=gas, solid=solid, flammable=solid.copy(),
              is_vacuum=np.zeros((h, w), dtype=bool),
              wall_hp=np.where(solid, 60 * 65536, 0).astype(np.int32),
              fire=np.where(solid, 65536, 0).astype(np.int32),
              ignition_temp_q16=np.where(solid, 500 * 65536, 0).astype(np.int32),
              temperature=np.where(solid, 900 * 65536, 0).astype(np.int32))
    st["gas"][SMOKE][~solid] = soot_trace_q
    return st


DIALS = dict(burn_rate=1.0, o2_thresh_burn=0.03, H_FUEL_M=4.0, H_FUEL_SHIFT=0,
             soot_yield=0.3, fuel_per_o2=0.7, o2_frac_ext=0.13,
             o2_frac_full=0.21, T_MAX_PHYS=16000.0)


def _burn_pair(soot_trace_q):
    d = DIALS
    c = _comb_scene(soot_trace_q)
    comb = bp.CombustionSolver()
    for k, v in d.items():
        setattr(comb, k, v)
    eos_c = bp.EOSSolver()
    comb.step(c["gas"], O2, INERT_N2, SMOKE, c["temperature"], c["wall_hp"],
              c["fire"], c["flammable"], c["solid"], c["is_vacuum"],
              c["ignition_temp_q16"], 0.25, 0.0076849, 0.05, eos=eos_c)
    g = _comb_scene(soot_trace_q)
    eos_g = bp.EOSSolver()
    bp.cuda_combustion_step(
        g["gas"], O2, INERT_N2, SMOKE, g["temperature"], g["wall_hp"], g["fire"],
        g["flammable"], g["solid"], g["is_vacuum"], g["ignition_temp_q16"],
        0.25, 0.0076849, 0.05,
        d["burn_rate"], d["o2_thresh_burn"], d["H_FUEL_M"], int(d["H_FUEL_SHIFT"]),
        d["soot_yield"], d["fuel_per_o2"], d["o2_frac_ext"], d["o2_frac_full"],
        d["T_MAX_PHYS"],
        fire_T_ext=float(comb.fire_T_ext), fire_T_span=float(comb.fire_T_span),
        hotf_cap=float(comb.hotf_cap), eos=eos_g)
    return c, eos_c, g, eos_g


def part3_mint() -> bool:
    print("PART 3 -- the SOOT MINT: CPU == GPU, the saturation and its books:")
    ok = True
    books = lambda e: (e.trace_mint_sum()[SMOKE], e.trace_clip_sum()[SMOKE],
                       e.trace_neg_hits()[SMOKE])
    c0, ec0, g0, eg0 = _burn_pair(0)
    c1, ec1, g1, eg1 = _burn_pair(TRACE_MAX - 1)
    for tag, c, ec, g, eg in (("free", c0, ec0, g0, eg0), ("full", c1, ec1, g1, eg1)):
        for k in ("gas", "temperature", "wall_hp"):
            if not np.array_equal(c[k], g[k]):
                ok = False
                print(f"  [{tag}] {k} MISMATCH CPU/GPU")
        if books(ec) != books(eg):
            ok = False
            print(f"  [{tag}] books MISMATCH cpu={books(ec)} gpu={books(eg)}")
    mint0, clip0, _ = books(eg0)
    mint1, clip1, neg1 = books(eg1)
    if not (mint0 > 0 and clip0 == 0):
        ok = False
        print(f"  free plane: mint {mint0} clip {clip0} (want mint > 0, clip 0)")
    if not (mint1 == mint0 and clip1 > 0 and neg1 == 0):
        ok = False
        print(f"  full plane: mint {mint1} clip {clip1} neg {neg1} "
              f"(want mint == {mint0}, clip > 0, neg 0)")
    if int(g1["gas"][SMOKE].max()) > TRACE_MAX:
        ok = False
        print("  full plane: a cell above TRACE_CELL_MAX")
    d = int(g1["gas"][SMOKE].astype(np.int64).sum()) - int(
        _comb_scene(TRACE_MAX - 1)["gas"][SMOKE].astype(np.int64).sum())
    if d != mint1 - clip1:
        ok = False
        print(f"  full plane: Δ S {d} != mint − clip {mint1 - clip1}")
    if ok:
        print(f"  isolated: CPU == GPU; mint {mint1}, clip {clip1} at the ceiling")
    ok = _run("FIRE", _scn_fire, resident=False, parity=False) and ok
    return ok


def part4_golden() -> bool:
    print("PART 4 -- CUDA build's CPU path (flags off) vs the committed golden:")
    _set_eos_backends(False)
    bp.set_combustion_backend(False)
    from field_ab_harness import capture_trajectory
    from field_digest import trajectory_digest
    from _xarch_perfield_digest import GOLDEN_AGGREGATE as GOLDEN
    dig = trajectory_digest(capture_trajectory(n_steps=30))
    if dig != GOLDEN:
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
        return False
    print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return True


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("TRACE_SAT_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    results = [part1_chained(), part2_resident(), part3_mint(), part4_golden()]
    if all(results):
        print("TRACE_SAT_RESULT: PASS")
        return 0
    print("TRACE_SAT_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
