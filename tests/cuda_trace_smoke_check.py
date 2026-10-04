"""Smoke transport v2 (#12) V6-CHAINED: CPU == chained GPU EOS at tolerance 0 on
every trace plane and every trace counter (runs inside the GPU subprocess).

Rewritten at P2a. The checker this file used to be tested the semi-Lagrangian
SmokeDynamics step against ``bp.cuda_smoke_step`` (both DELETED with the old
law). The trace planes now ride the bulk face flux INSIDE the EOS substep loop
(``bulk_transport.cpp`` stages 3b/3c — THE oracle — and their device twins in
``cuda_bulk_transport.cu``), plus a once-per-tick trace tail that runs in shared
host code after the dispatch. This check is the parity gate on that law.

PROPERTY: the chained GPU orchestration (cuda_eos_step.cu: the trace planes
H2D'd before the substep loop, stages 3b/3c once per substep in
bulk_flux_energy_transport_device, D2H'd before digest_bulk_flux) leaves EVERY
gas plane, wind, temperature, the stored gas energy AND every trace book
(``boundary_flux()`` trace slots = vent, ``trace_wipe_sum``, ``trace_sink_sum``,
``trace_decay_sum``) bit-identical to the CPU run of the same scenario, every
tick. BREAKS IF: a device stage-3b/3c arithmetic differs from the CPU's (a
price, a face order, a participation predicate), the 3b/3c order around stage
4's d_nb re-accumulation is changed, the D2H lands after the digest, a counter
is not reset per tick on the chained path, or the boundary_flux slots disagree
on a space map.

Driven through the REAL Simulation (full step: payload edits, EOS, trace tail,
fire, combustion, ...) on two independently built worlds per scenario — flags
OFF for the CPU world, the four EOS kernel-surface flags ON for the GPU world
(dispatch-fired proven via ``eos_step_cuda_calls``, so a silently-CPU run can
never pass). Three scenarios, each with its own non-vacuity guards:

  BLAST   a sealed two-room hull, a heat-only blast, five trace planes seeded
          (one tile at 2.0, above the old ceiling), an AIR-LESS enclosed pocket
          holding trace (stage 3c's N_EPS wipe) — decay, wipe live;
  BREACH  a hull inside a space band, breached mid-run, trace deposited on a
          vacuum tile (the stranded case) — vent, sink, decay live; on a SPACE
          map, where boundary_flux() must read zeros of size n_gases;
  CRATES  a crate (thermal-solid, permeable) wall in the flow of a blast —
          the thermal-solid participation of stage 3b (D3).

Part 2: the CUDA build's CPU path (flags off) still reproduces the committed
default-scenario golden.

Prints ``TRACE_SMOKE_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Import the CUDA build FIRST so it is the cached `breach_physics` before
# field_ab_harness (which inserts cpp/build/Release on sys.path) imports it.
import breach_physics as bp

Q = 65536
HULL, AIR, FURNITURE, SPACE = 1, 0, 6, 9     # v2 codes == material ids

# The four EOS kernel-surface flags the chained dispatch ANDs.
_EOS_SETTERS = ("set_sl_advection_backend", "set_bulk_flux_backend",
                "set_mg_solve_backend", "set_kick_compression_backend")


def _set_eos_backends(on: bool) -> None:
    for name in _EOS_SETTERS:
        getattr(bp, name)(bool(on))


def _sim(tm, tile_m=0.333):
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    lvl = LevelData(name="trace_v6", version="2", path=Path("."), tilemap=tm,
                    tile_size_m=tile_m, diffuse_path=Path("."))
    sim = Simulation(lvl, seed=7, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _trace_ids(g):
    return [i for i in range(g.gases.n) if not bool(g.gases.conservative[i])]


def _heat_blast(sim, y, x, amount=3200.0, radius=5.0):
    from simulation.field_edit import EditMode, FieldEdit, Falloff, Region
    sim.edit_queue.enqueue(FieldEdit(
        field="heat", region=Region.DISC, coords=(y, x, float(radius)),
        amount=float(amount), mode=EditMode.ADD, falloff=Falloff.LINEAR,
        source_id=990))


def _books(sim):
    eos = sim.physics_runner.engine.eos
    return {
        "vent": list(eos.boundary_flux()),
        "wipe": list(eos.trace_wipe_sum()),
        "sink": list(eos.trace_sink_sum()),
        "decay": list(eos.trace_decay_sum()),
    }


# ---------------------------------------------------------------------------
# scenario builders: (tilemap, setup(sim), action(tick, sim), n_ticks)
# ---------------------------------------------------------------------------
def _scn_blast():
    from simulation.gases import POISON, SMOKE, STEAM, TEARGAS, FUEL_GAS
    tm = np.full((26, 38), AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = HULL
    tm[1:-1, 19] = HULL
    tm[11:15, 19] = AIR                      # a doorway between the rooms
    pocket = (20, 33)                        # the enclosed AIR-LESS pocket's tile
    tm[19, 33] = tm[21, 33] = tm[20, 32] = tm[20, 34] = HULL

    def setup(sim):
        g = sim.gmap
        g.gas[SMOKE][8:12, 4:10] = Q // 2
        g.gas[SMOKE][12, 6] = 2 * Q          # above the old [0, 1] ceiling
        g.gas[POISON][5:9, 22:28] = Q // 3
        g.gas[TEARGAS][14:18, 6:12] = Q // 5
        g.gas[STEAM][2:5, 2:6] = Q // 4
        g.gas[FUEL_GAS][17:22, 24:30] = Q // 6
        # The pocket: open (participating) but with NO air, holding trace ->
        # stage 3c's N_EPS wipe is live from the first substep.
        py, px = pocket
        for b in range(g.gases.n):
            if bool(g.gases.conservative[b]):
                g.gas[b][py, px] = 0
        g.gas_energy[py, px] = 0
        g.temperature[py, px] = 0
        g.gas[SMOKE][py, px] = Q // 2
        g.gas[POISON][py, px] = Q // 8
        _heat_blast(sim, 12, 8, amount=3600.0, radius=5.0)

    def action(tick, sim):
        if tick == 20:
            _heat_blast(sim, 12, 28, amount=2400.0, radius=4.0)

    return tm, setup, action, 60


def _scn_breach():
    from simulation.gases import POISON, SMOKE, STEAM
    tm = np.full((24, 34), SPACE, dtype=np.int32)
    tm[2:22, 2:32] = HULL
    tm[3:21, 3:31] = AIR

    def setup(sim):
        g = sim.gmap
        g.gas[SMOKE][8:13, 22:28] = 2 * Q
        g.gas[POISON][5:9, 5:9] = Q // 3
        g.gas[STEAM][14:18, 20:26] = Q // 3
        _heat_blast(sim, 10, 12, amount=3000.0, radius=5.0)

    def action(tick, sim):
        g = sim.gmap
        if tick == 8:
            g.destroy_wall(10, 31)           # breach the east hull to space
            g.destroy_wall(11, 31)
        if tick == 30:
            vac = np.argwhere(g.is_vacuum & ~g.solid)
            y, x = (int(v) for v in vac[0])
            g.gas[SMOKE][y, x] += Q // 4     # stranded: trace on a vacuum tile
            g.gas[POISON][y, x] += Q // 8

    return tm, setup, action, 60


def _scn_crates():
    from simulation.gases import POISON, SMOKE, TEARGAS
    tm = np.full((18, 34), AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = HULL
    tm[1:-1, 17] = FURNITURE                 # a full-height crate wall
    tm[4:7, 24:27] = FURNITURE               # a free-standing crate

    def setup(sim):
        g = sim.gmap
        g.gas[SMOKE][1:-1, 1:17] = Q // 2
        g.gas[TEARGAS][3:15, 19:23] = Q // 5
        g.gas[POISON][3:9, 3:8] = Q // 4
        _heat_blast(sim, 9, 6, amount=3200.0, radius=4.0)

    def action(tick, sim):
        return None

    return tm, setup, action, 48


def _run_scenario(name, builder) -> bool:
    tm, setup, action, n_ticks = builder()
    cpu, gpu = _sim(tm), _sim(tm)
    setup(cpu)
    setup(gpu)
    gc, gg = cpu.gmap, gpu.gmap
    traces = _trace_ids(gc)
    fields = ("gas", "gas_energy", "wind_x", "wind_y", "temperature", "atmosphere")
    for f in fields:
        assert np.array_equal(getattr(gc, f), getattr(gg, f)), \
            f"{name}: scenario construction not deterministic on {f}"

    calls0 = bp.eos_step_cuda_calls()
    tot = {k: 0 for k in ("vent", "wipe", "sink", "decay")}
    max_nsub = 0
    bad = 0
    for tick in range(n_ticks):
        action(tick, cpu)
        action(tick, gpu)
        _set_eos_backends(False)
        cpu.step()
        _set_eos_backends(True)
        gpu.step()
        _set_eos_backends(False)

        for f in fields:
            a, b = getattr(gc, f), getattr(gg, f)
            if not np.array_equal(a, b):
                bad += 1
                mism = int(np.count_nonzero(a != b))
                idx = int(np.argmax(a != b))
                print(f"  [{name}] tick {tick}: field {f}: {mism} MISMATCH(es) "
                      f"(first flat @ {idx}: cpu={a.flat[idx]} gpu={b.flat[idx]})")
        # The EOS's own bulk-flux digest hashes EVERY gas plane (trace included)
        # from host memory: the trace D2H must have landed before it.
        dc = cpu.physics_runner.engine.eos.digest_bulk_flux
        dg = gpu.physics_runner.engine.eos.digest_bulk_flux
        if dc != dg:
            bad += 1
            print(f"  [{name}] tick {tick}: digest_bulk_flux cpu={dc} gpu={dg}")
        bc, bg = _books(cpu), _books(gpu)
        for key in bc:
            if bc[key] != bg[key]:
                bad += 1
                print(f"  [{name}] tick {tick}: books[{key}] cpu={bc[key]} gpu={bg[key]}")
        for t in traces:
            tot["vent"] += bc["vent"][t]
            tot["wipe"] += bc["wipe"][t]
            tot["sink"] += bc["sink"][t]
            tot["decay"] += bc["decay"][t]
        max_nsub = max(max_nsub, cpu.physics_runner.engine.eos.dbg_last_n_sub)
        if bad >= 10:
            print("  aborting after 10 divergences")
            break

    ok = (bad == 0)
    fired = bp.eos_step_cuda_calls() - calls0
    if fired < n_ticks:
        ok = False
        print(f"  [{name}] the chained dispatch fired {fired}/{n_ticks} ticks "
              f"(a silently-CPU run cannot pass)")
    # boundary_flux() must agree on map kind: n_gases zeros-or-values on BOTH.
    if len(_books(cpu)["vent"]) != len(_books(gpu)["vent"]):
        ok = False
        print(f"  [{name}] boundary_flux size differs CPU/GPU")

    # ---- non-vacuity guards: the gate must exercise what it claims ----------
    need = {"BLAST": ("wipe", "decay"), "BREACH": ("vent", "sink", "decay"),
            "CRATES": ("decay",)}[name]
    for k in need:
        if tot[k] <= 0:
            ok = False
            print(f"  [{name}] scenario too tame: no '{k}' was ever booked")
    if max_nsub <= 1:
        ok = False
        print(f"  [{name}] scenario too tame: the EOS never needed a 2nd substep")
    if name == "CRATES":
        from simulation.gases import SMOKE
        if not int(gg.gas[SMOKE][1:-1, 18:-1].sum()) > 0:
            ok = False
            print("  [CRATES] no smoke crossed the crate wall")
    if ok:
        print(f"  [{name}] {n_ticks} ticks bit-identical (gas, E, wind, T, P, "
              f"all trace books); booked totals {tot}, max n_sub {max_nsub}.")
    return ok


def part1_chained_parity() -> bool:
    print("PART 1 — V6-chained: CPU == chained GPU EOS on every trace plane and "
          "every trace book, per tick:")
    ok = True
    for name, builder in (("BLAST", _scn_blast), ("BREACH", _scn_breach),
                          ("CRATES", _scn_crates)):
        ok = _run_scenario(name, builder) and ok
    return ok


# ---------------------------------------------------------------------------
# PART 2 — the CUDA build's CPU path still reproduces the committed golden
# ---------------------------------------------------------------------------
def part2_golden() -> bool:
    print("PART 2 — CUDA build's CPU path (flags off) vs the committed golden:")
    _set_eos_backends(False)
    from field_ab_harness import capture_trajectory
    from field_digest import trajectory_digest

    # The sanctioned golden is OWNED by tests/_xarch_perfield_digest.py (its
    # lineage block carries every rebase + rationale); import it.
    from _xarch_perfield_digest import GOLDEN_AGGREGATE as GOLDEN
    base = capture_trajectory(n_steps=30)
    dig = trajectory_digest(base)
    if dig != GOLDEN:
        print(f"  GOLDEN MISMATCH: {dig[:16]}... != {GOLDEN[:16]}...")
        return False
    print(f"  CUDA build CPU path reproduces the golden ({dig[:12]}...).")
    return True


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("TRACE_SMOKE_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    p1 = part1_chained_parity()
    p2 = part2_golden()
    if p1 and p2:
        print("TRACE_SMOKE_RESULT: PASS")
        return 0
    print("TRACE_SMOKE_RESULT: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
