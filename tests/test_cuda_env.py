"""#29 P8.10a -- the device backend of the batched environment (pytest
wrapper of tests/cuda_env_check.py). One test per property; each runs its
part in an isolated CUDA subprocess (cuda_harness) and SKIPS cleanly without
a CUDA build / device.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def _run(parts, timeout=5400):
    proc = cuda_harness.run_cuda_script(
        "import cuda_env_check, sys; "
        f"sys.exit(cuda_env_check.main({list(parts)!r}))",
        timeout=timeout,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "ENV_RESULT: PASS" in out, out[-6000:]


def test_the_device_backend_equals_the_farm_step_by_step():
    """PROPERTY: BACKEND EQUALITY -- the same run file, run seed and action
    stream (seeded-random under the served masks, the two sides' masks shown
    raw-bit identical first) give on ``backend="device"`` and on
    ``cpu_farm``, at EVERY step, the same observation (every built key, raw
    bits; the columns no device patch has built yet named, not gated),
    reward groups, terminated, truncated, counters, team blocks,
    rows_ended, masked choices, final_obs / final_full_view (zeros on the
    rows that did not end), full view, result and per-env checkpoints, over
    several episodes per env with the auto-resets on both sides, envs ending
    at different steps: a training room at N = 1, 4 and 8 and ship_demo
    (zombies) at N = 1 and 4; the coverage counters equal too. BREAKS IF a
    reset lands late or on another env's draw, a final_ row leaks, the
    decision is handed over late, a served array is read after a reset, or
    a checkpoint is taken off the farm's ticks."""
    _run(["equality"])


def test_the_device_backend_refuses_by_name():
    """PROPERTY: the device backend refuses, naming the reason, a team not
    'policy', ``[run] commander`` other than "none", a level set that cannot
    share one batch, the "counters" reward profile, and an out-of-range
    action (the host's ValueError with check_actions' message and the env;
    every later call refused). BREAKS IF a run the device cannot serve is
    silently played differently from the farm."""
    _run(["refusals"])


def test_a_steady_step_makes_only_the_conductors_syncs():
    """PROPERTY: a step that ends no episode and takes no checkpoint makes
    exactly two host syncs per tick, both the conductor's named reads
    (max_n_sub, the books read), at N = 1 and 2, the decision handed over as
    a device tensor; counted after a warm-up with the cycle collector off.
    BREAKS IF the env adds a download (a host copy of a served array, a
    per-step terminated read) or the count picks up a destructor."""
    _run(["syncs"])


def test_the_returned_observation_is_the_envs_device_buffers():
    """PROPERTY: the observation, full view and final_ tensors a step
    returns share their pointers with the env's device buffers, on every
    step; reward / terminated / truncated are on the card; team_rows is a
    view. BREAKS IF a step copies an observation (to the host or into a
    fresh tensor)."""
    _run(["handover"])


def test_each_deliberate_break_turns_the_equality_red():
    """PROPERTY: the backend-equality comparison sees each deliberate
    defect (the auto-reset one step late, the final_ rows of envs that did
    not end kept, the episode drawn from the wrong env index, the decision
    handed over after the step's first tick). BREAKS IF the gate cannot see
    a defect of the env loop."""
    _run(["breaks"])
