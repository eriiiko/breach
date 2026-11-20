"""arc #63 -- the swarm store rides ResidentBatch's ONE batch axis.

Erik's P6 ruling (docs/unit_design_2026-11-14.md §6.3, 2026-11-14): the swarm
store declares PER-MAP arrays (a column is ``(max_units,)``, ``high_water``
and ``next_unit_id`` are 0-d) and ``ResidentBatch`` supplies N: it stacks
every ``GameMap._RESIDENT_FIELD_NAMES`` entry to ``(N,) + shape`` and binds
slice e to env e. There is no second N.

PROPERTIES (each with the change that must break it):

1. STACKED ON THE ONE AXIS: every swarm resident buffer of a batch of N is
   ``(N,) + the map's own array shape``, and the map's array carries no
   batch axis of its own (no leading length-1 env axis). Breaks if the store
   grows back a leading ``N_ENV`` axis (the buffer would be
   ``(N, 1, max_units)``: two Ns).
2. SLICE e IS ENV e: right after binding, every env's device slice equals
   that env's host array, and only the env that spawned holds a present
   swarm. Breaks on a bind that seeds slices from the wrong map.
3. INDEPENDENT SLICES: a spawn into env 1 after binding, uploaded with env
   1's own ``from_host``, leaves every other env's device slice byte-equal
   and every other env's ``__swarm__`` carrier absent. Breaks if a swarm
   upload writes through another env's slice (an env-stride bug).
4. INDEPENDENT DIGESTS: env 0's ``tick_digest`` after every batched tick is
   the same whether or not env 1 carries a swarm. Breaks if swarm state
   leaks across envs into hashed state.

Run through tests/test_cuda_swarm_batch.py (the cuda_harness subprocess).
"""
from __future__ import annotations

import sys

import numpy as np

import breach_physics as bp  # noqa: F401  (the CUDA build, loaded first)

import cuda_s8a_check as s8a
from cuda_batch_invariance_check import _ship_sim

N_BATCH = 3
TICKS = 8
SPECIES = "larva"


def _swarm_names():
    from simulation.swarm import SWARM_RESIDENT_NAMES
    return SWARM_RESIDENT_NAMES


def _open_tile_centres(sim, k, offset):
    from simulation import swarm_fixed
    g = sim.gmap
    ys, xs = np.nonzero(~g.solid & ~g.is_vacuum)
    idx = [(offset + 7 * i) % len(ys) for i in range(k)]
    half = swarm_fixed.FP_ONE >> 1
    px = [(int(xs[i]) << swarm_fixed.FP_SHIFT) + half for i in idx]
    py = [(int(ys[i]) << swarm_fixed.FP_SHIFT) + half for i in idx]
    return px, py


def _digest(sim):
    from field_ab_harness import capture_snapshot
    from field_digest import tick_digest
    return tick_digest(capture_snapshot(sim))


def _run(spawn_in_env1: bool):
    """One batch of N_BATCH ship_demo sims; returns (ok, env-0 digests)."""
    import cupy as cp
    from simulation.resident_batch import ResidentBatch
    from simulation.swarm import swarm_carrier, swarm_section_bytes
    ok = True
    sims = [_ship_sim(e + 1) for e in range(N_BATCH)]
    if spawn_in_env1:
        px, py = _open_tile_centres(sims[1], 6, 11)
        sims[1].swarm.spawn(SPECIES, px, py, [0, 100, -100, 2000, -2000, 50000])
    s8a._residency(True); s8a._set_backends(True)
    digests = []
    try:
        batch = ResidentBatch(sims)
        names = _swarm_names()
        # 1. stacked on the one axis
        for name in names:
            host = getattr(sims[0].gmap, name)
            buf = batch.buffers[name]
            if tuple(buf.shape) != (N_BATCH,) + tuple(host.shape):
                ok = False
                print(f"  FAIL 1: {name} buffer {tuple(buf.shape)} != "
                      f"(N,) + map shape {tuple(host.shape)}")
            if host.ndim >= 1 and host.shape[0] == 1:
                ok = False
                print(f"  FAIL 1: {name} map array {host.shape} has a "
                      f"leading length-1 (env) axis")
        # 2. slice e is env e
        for e, sim in enumerate(sims):
            for name in names:
                if not np.array_equal(batch.buffers[name][e].get(),
                                      getattr(sim.gmap, name)):
                    ok = False
                    print(f"  FAIL 2: env {e} slice of {name} != its host array")
            present = swarm_carrier(sim.gmap)["present"]
            if present != (spawn_in_env1 and e == 1):
                ok = False
                print(f"  FAIL 2: env {e} present={present}")
        if spawn_in_env1:
            # 3. independent slices
            before = {(e, n): batch.buffers[n][e].get().tobytes()
                      for e in range(N_BATCH) for n in names if e != 1}
            carriers = {e: swarm_section_bytes(swarm_carrier(sims[e].gmap))
                        for e in range(N_BATCH) if e != 1}
            px, py = _open_tile_centres(sims[1], 3, 5)
            sims[1].swarm.spawn(SPECIES, px, py, [7, 8, 9])
            sims[1].gmap.from_host(list(names))
            for (e, n), b in before.items():
                if batch.buffers[n][e].get().tobytes() != b:
                    ok = False
                    print(f"  FAIL 3: env 1's upload moved env {e}'s {n}")
            for e, b in carriers.items():
                c = swarm_carrier(sims[e].gmap)
                if c["present"] or swarm_section_bytes(c) != b:
                    ok = False
                    print(f"  FAIL 3: env {e}'s carrier moved")
            for n in names:
                if not np.array_equal(batch.buffers[n][1].get(),
                                      getattr(sims[1].gmap, n)):
                    ok = False
                    print(f"  FAIL 3: env 1's slice of {n} != its host after upload")
        for _t in range(TICKS):
            batch.step()
            digests.append(_digest(sims[0]))
    finally:
        s8a._residency(False); s8a._set_backends(False)
        for sim in sims:
            sim.gmap.__dict__.pop("_dev", None)
        del sims
        cp.get_default_memory_pool().free_all_blocks()
    return ok, digests


def main() -> int:
    print(f"swarm batch axis (arc #63, P6 ruling), N = {N_BATCH}, {TICKS} ticks")
    ok_a, d_without = _run(spawn_in_env1=False)
    ok_b, d_with = _run(spawn_in_env1=True)
    ok = ok_a and ok_b
    # 4. independent digests
    if d_without != d_with:
        ok = False
        first = next(i for i, (a, b) in enumerate(zip(d_without, d_with)) if a != b)
        print(f"  FAIL 4: env 0's tick_digest differs at tick {first} when env 1 "
              f"carries a swarm")
    else:
        print(f"  env 0's tick_digest equal over {TICKS} ticks with and without "
              f"a swarm in env 1")
    print("SWARM_BATCH_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
