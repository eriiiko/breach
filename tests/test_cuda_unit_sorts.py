"""#29 P7e.2 -- the device sorts and 9b (pytest wrapper of
tests/cuda_unit_sorts_check.py). One test per device input the patch moved
off the host; each runs its parts in an isolated CUDA subprocess
(cuda_harness) and SKIPS cleanly without a CUDA build / device.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def _run(parts, timeout=1800):
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_sorts_check, sys; "
        f"sys.exit(cuda_unit_sorts_check.main({list(parts)!r}))",
        timeout=timeout,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_SORTS_RESULT: PASS" in out, out[-6000:]


def test_slot9_reads_a_burn_through_list_sorted_on_the_device():
    """PROPERTY: slot 9's DESTROY_WALL records come from the burn-through
    list as the device holds it (a scrambled order in the shadow, the
    resident fire's own atomic arrival order in place), ranked into the
    host's row-major order ON THE DEVICE: every plane and book equals the
    host slot on fire scenes where several walls fall in one tick and the
    order decides the map. BREAKS IF the device sort is skipped or unstable
    (the walk sees the arrival order), the face is fed a host-built list, or
    the resident address is not the fire's."""
    _run(["burn", "resident"])


def test_slot9b_scan_and_top_k_run_on_the_device():
    """PROPERTY: slot 9b's destroys come from the device's own scan of the
    planes (no host list is handed to it): the candidates beyond the cap,
    the cut inside a tie and the descending order equal the host's, every
    plane and book at tol 0. BREAKS IF the scan falls back to a host list,
    the tie order or the descending order changes, or a threshold row is not
    the material's."""
    _run(["burst"])


def test_slot6_stamp_rows_are_built_on_the_device_from_the_unit_store():
    """PROPERTY: slot 6's stamp rows are built on the device from the unit
    store (equal to the host C++ stamp's rows on every tick while the units
    walk, so the rows change) and the five planes and the refilled
    atmosphere equal the host's. BREAKS IF the rows come from anywhere but
    the store (a stale or host-built set diverges as the units move), the
    footprint order or a default value changes, or the refill's order or
    mean differs."""
    _run(["stamp"])
