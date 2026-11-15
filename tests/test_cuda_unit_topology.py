"""#29 P6.4b -- the topology gate (pytest wrapper of
tests/cuda_unit_topology_check.py).

Property: the per-env serial applier's topology seam on the device
(``GameMap.on_tile_changed`` and ``destroy_wall`` as device functions,
``cpp/src/cuda_topology.cuh``) equals the Python on every plane it writes and
every book, bit for bit, on fixture maps that reach every branch (the ambient
fallback, donors of unequal composition and the seed's rounding, edge hulls,
a tile exposing the boundary through a neighbour, a chain whose two orders
give different maps, furniture in a compressed hot pocket and beside a
breach, a hot gas tile breaching, a burning tile, both doors, every material
row, out-of-bounds and air no-ops, the membership retire and the born at
ambient; space and planetside maps at 1 / 0.5 / 0.333 m), with the energy and
mass books closing on the DEVICE's planes as on the host's; slot 9 (the
fire's burn-through walls) and the tick-end clears equal their Python slots
in the shadow lockstep on a fire scene where walls burn through on many
ticks (a breach to space, adjacent tiles in one tick, both doors) and on
ship_demo's bots match, the write-set gate seeing no host write that is
neither an output nor host-only; env i of a batch of N = 2 / 8 different envs
equals env i alone; an empty env beside full ones is untouched; a list past
its capacity applies its first entries, counts the rest and is refused by
name; and slot 9 and the clears chained in place of the host's keep a twin
fire match on the reference's path every tick (digests, planes, books).
Breaks if the seed's rounding, the donor predicate, the breach rule's read of
an earlier destroy, the atmosphere pin, T or fire's clear, a neighbour's
facing face, the membership retire, a booking's sign, any cache patch, the
list's order, the env rebase, a table row or a cleared plane differs from the
reference, or a plane the host writes is left undeclared.

SKIPS cleanly without a CUDA build / device; runs the check in an isolated
CUDA subprocess (cuda_harness).
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def test_cuda_unit_topology_seam_slot9_clears_and_chained():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_topology_check, sys; sys.exit(cuda_unit_topology_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_TOPOLOGY_RESULT: PASS" in out, (
        f"the topology gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
