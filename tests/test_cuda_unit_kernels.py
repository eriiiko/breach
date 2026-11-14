"""#29 P6.2 -- the unit kernels gate (pytest wrapper of
tests/cuda_unit_kernels_check.py).

Property: the device tables equal the reference objects' values; the device
unit library (the composed flags, the status apply with the stack cap's
refusal, the sweep, the hook accessor) equals status.py / unit_traits.py on
synthetic rows over their range; and every ported slot kernel -- 1a (the
impulse integration and the turn), 9c2 (the wind's impulse), 1b' and 10' (the
orders), the tick-end conversion, 1c -- equals its Python law row for row at
tol 0 on every tick of live scenes that make each law act (non-neutral TURN /
HOLD / HP hooks, an order schedule, the doorway jet's knockdowns and prone
bodies in wind, a horde's conversions), in batches of N = 2 and 8 with padding
rows (env i == env i alone), its coverage counter equal to the rows the host
changed; hooking a match leaves it equal to an unhooked twin; and the two
chained pairs (1b' + 1c, 10' + the conversion) replacing the host's slots keep
a twin match on the reference's path every tick. Breaks if any kernel's
arithmetic differs from its law, a hook is ignored, a padding row or another
env is touched, a table value is not the reference's, or a chain diverges.

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


def test_cuda_unit_kernels_shadow_lockstep_and_chained():
    proc = cuda_harness.run_cuda_script(
        "import cuda_unit_kernels_check, sys; sys.exit(cuda_unit_kernels_check.main())",
        timeout=3600,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "UNIT_KERNELS_RESULT: PASS" in out, (
        f"the unit kernels gate did not pass.\nreturncode={proc.returncode}\n{out}")
    assert proc.returncode == 0, f"subprocess exit {proc.returncode}\n{out}"
