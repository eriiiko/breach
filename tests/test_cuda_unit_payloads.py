"""#29 P7c.1 / P7c.2 -- the payloads gate, every row (pytest wrapper of
tests/cuda_unit_payloads_check.py).

Property: a PAYLOAD record of ANY [payloads.*] row is
``payloads.execute_payload`` integer for integer on the device
(cpp/src/cuda_unit_payload.cuh over cpp/src/cuda_unit_blast.cuh): the disc
path (the disc, its wall damage, the stencil, the DISC edits, the immediate
blast packets, the flash, emit_gas / ignite_ring / deposit_heat, the noise,
the blast log) and -- #29 P7c.2 -- the strength path (the reach walk after
the wall damage with its seed rule, the air read, start_source, the
fixed-total TILES smoke and heat with their claims), the explosive rows' disc
air read and noisy scaled smoke, the comparison rows' multi-tick source on the
blast-source table and its take at slot 7's head (handed to the resident EOS
on the card), bulk gas through the gas-N seam; equal on probes, on scenes that
detonate every row through every call site, in batches, with padding
untouched, every list refused by name at its capacity, no row refused, a bulk
deposit past int32 refused by name writing nothing, and in chained twins
whose host execute_payload never runs (ship_demo's bots' frag included).
Breaks if a payload row, a walk's tile set or order, a seed, the air read's
moment or arithmetic, a weight, a TILES row, the source's sum, saturation,
count-down or compaction, a bulk tile, book or refusal, the call order or the
batch indexing differs from the law.

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


def test_cuda_payloads_of_every_row_equal_the_reference():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_payloads_check, sys; sys.exit(cuda_unit_payloads_check.main())",
        timeout=7200,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_PAYLOADS_RESULT: PASS" in out, (
        f"the payloads gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
