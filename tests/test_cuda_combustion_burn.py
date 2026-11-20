"""#71 / residency arc #29 P2 — the CUDA combustion twin on a BURNING scene.

Property: with ONLY the combustion backend flipped, a burning scene stepped
through the real tick (``Simulation.step`` -> ``PhysicsRunner.step``) is
bit-identical on the GPU twin and the CPU ``CombustionSolver``: every GameMap
array and every integer book (rail and floor hits, every ``e_comb_*`` channel,
the EOS / temperature / engine books, the trace books, the seam net) at tol 0
on every tick, the across-tick energy closure residual equal on both worlds,
and in isolation the energy-on pass (draw radius 1-3, ring donors and sites,
furniture sites, thin cells, the T_MAX_PHYS rail) equal plane for plane and
book for book. The scenes are the #71 repro (the playground, a lit 1263-game
wood tile, a marine beside it) and the fire + smoke studio.

Breaks if the CUDA twin drifts from the CPU law anywhere on the energy seam —
a donor's energy not debited or its mirror not refreshed, a parcel split or
routed differently, the fire heat landed in ``temperature`` instead of
``gas_energy`` (the pre-#54 twin, issue #71), a book not booked or booked into
another channel — or if the runner stops handing the twin the energy field or
stops booking its counters into the solver. Validated 2026-11-07 by breaking
the twin (an off-by-one in Pass B's remainder energy) and watching it fail.

Runs ``cuda_combustion_burn_check`` in an isolated ``data`` subprocess against
this checkout's ``cpp/build_cuda`` (tests/cuda_harness.py); skips cleanly
without a CUDA build.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_cuda_combustion_twin_is_bit_identical_on_burning_scenes():
    proc = cuda_harness.run_cuda_script(
        "import cuda_combustion_burn_check, sys; "
        "sys.exit(cuda_combustion_burn_check.main())",
        timeout=1800,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "COMB_BURN_RESULT: PASS" in out, (
        f"the combustion twin is not level on a burning scene.\n"
        f"returncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
