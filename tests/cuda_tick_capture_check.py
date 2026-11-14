"""#29 P5 -- the resident tick's device chain CAPTURES in global mode.

docs/residency_batching_design_2026-11-07.md §4.2 "Streams": one stream is
threaded through every launch core, every in-tick copy and memset is
asynchronous on it, and "a capture-mode test asserts the whole tick captures
in global mode (any legacy-stream operation fails it)". This is that test.

The device chain of a tick (physics_runner.run_resident_batch, from after the
entry uploads to before the exit downloads: the stamp, water, the EOS stage,
the trace tail, combustion, the tail) runs under cudaStreamBeginCapture(
tick stream, cudaStreamCaptureModeGlobal) through the TEST-ONLY hook
physics_runner._TICK_CAPTURE. In global mode any legacy-stream operation, any
synchronous copy, any cudaMalloc, any pageable copy issued while capturing
fails the CUDA call that made it (the tick raises). The tick's host syncs go
through tick_sync_point, which -- with the probe on -- ends the segment,
launches it as a graph (so the tick still runs), and restarts the capture.

THE NAMED EXCEPTIONS, exactly two per tick (the test asserts this list, in
order, and nothing else):
  "max_n_sub" -- the EOS stage's counted 8-byte read of the batch's max n_sub,
                 which drives the masked substep loop (P9 replaces it with a
                 WHILE graph node);
  "books"     -- the tail's one books D2H (every stage, every env), folded
                 into the engines on the host while the host gameplay still
                 runs per env (P5..P8).

PROPERTY: the device chain is legacy-stream-free and allocation-free, and its
only host syncs are the two named reads; capturing it changes nothing (the
captured tick equals the uncaptured twin, bit for bit, every array and book).
A change that must break it: a kernel launched on the default stream, a
cudaMemcpy / cudaMemset without the stream, a cudaMalloc in a steady-state
tick, a new host sync in the chain. Validated by breaking it once (K_store_atm
launched on the default stream: the capture fails).

Scenes: the s8a blast room at N = 1 (water live, the device stamp, combustion)
and a batch of N = 3 (blast / calm / pool), each after warm-up ticks (the
one-time table uploads happen outside steady state).
"""
from __future__ import annotations

import sys

import breach_physics as bp

import cuda_batch_invariance_check as inv
import cuda_s8a_check as s8a

EXPECTED_POINTS = ["max_n_sub", "books"]


class _Probe:
    def __init__(self):
        self.summary = None

    def begin(self):
        bp.resident_capture_begin()

    def end(self):
        self.summary = bp.resident_capture_end()


def _captured_tick(step_fn):
    from simulation import physics_runner as pr
    probe = _Probe()
    pr._TICK_CAPTURE = probe
    try:
        step_fn()
    except Exception:
        bp.resident_capture_abort()
        raise
    finally:
        pr._TICK_CAPTURE = None
    return probe.summary


def _scene(templates, warm=3, ticks=3) -> bool:
    from simulation.resident_batch import ResidentBatch
    n = len(templates)
    print(f"  scene {templates} (N = {n}): {warm} warm-up ticks, then {ticks} "
          f"captured ticks beside an uncaptured twin batch")
    worlds_c = [inv.make_world(t, 77 + i) for i, t in enumerate(templates)]
    worlds_r = [inv.make_world(t, 77 + i) for i, t in enumerate(templates)]
    s8a._residency(True); s8a._set_backends(True)
    ok = True
    try:
        bc, br = ResidentBatch(worlds_c), ResidentBatch(worlds_r)
        for _ in range(warm):
            bc.step(); br.step()
        for t in range(ticks):
            try:
                summ = _captured_tick(bc.step)
            except Exception as e:   # an op the capture refused
                print(f"    tick {t}: the capture FAILED: {type(e).__name__}: {e}")
                return False
            br.step()
            pts = list(summ["sync_points"])
            print(f"    tick {t}: captured in global mode -- {summ['segments']} segments, "
                  f"{summ['nodes']} graph nodes, host syncs {pts}")
            if pts != EXPECTED_POINTS:
                print(f"    FAIL: the host syncs must be exactly {EXPECTED_POINTS}")
                ok = False
            for i, (wc, wr) in enumerate(zip(worlds_c, worlds_r)):
                fc, fr = inv._fingerprint(wc), inv._fingerprint(wr)
                if fc[0] != fr[0] or fc[1] != fr[1]:
                    print("    FAIL " + inv._diff(f"env {i} tick {t} captured vs uncaptured",
                                                   fc, fr))
                    ok = False
    finally:
        s8a._residency(False); s8a._set_backends(False)
        for w in worlds_c + worlds_r:
            inv._free(w)
    return ok


def main() -> int:
    print(f"tick capture (#29 P5), HAS_CUDA={getattr(bp, 'HAS_CUDA', False)}")
    ok = _scene(["blast"])
    ok = _scene(["blast", "calm", "pool"]) and ok
    print("TICK_CAPTURE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
