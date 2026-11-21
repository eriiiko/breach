"""#29 P8 -- the device observation on the card (pytest wrapper of
tests/cuda_obs_check.py). One test per property; each runs its part in an
isolated CUDA subprocess (cuda_harness) and SKIPS cleanly without a CUDA
build / device.
"""
from __future__ import annotations

import pytest

import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


def _run(parts, timeout=3600):
    proc = cuda_harness.run_cuda_script(
        "import cuda_obs_check, sys; "
        f"sys.exit(cuda_obs_check.main({list(parts)!r}))",
        timeout=timeout,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "OBS_RESULT: PASS" in out, out[-6000:]


def test_the_device_observation_equals_the_host_one_on_the_corpus():
    """PROPERTY: on the pinned ship_demo matches, a training room and the
    hand match (corpus (d): the cases the others never reach), at every
    decision tick, every column / plane owned so far (the column-to-patch
    table; from P8.1 the float32 vector columns of the store and the
    tables) equals the host observation as raw bits, and no owned column is
    constant over the corpus. BREAKS IF a block reads the wrong anchor
    (OBS_BREAK=anchor), is not rebuilt on a decision tick (OBS_BREAK=stale),
    the gate compares a column that never moves (OBS_BREAK=constant), or a
    column divides by a neighbouring table row (OBS_BREAK=divisor)."""
    _run(["gate"], timeout=7200)


def test_each_env_of_a_batch_equals_its_own_reference():
    """PROPERTY: in batches of N = 2 and 4 (ship_demo and the training
    room, different seeds per env, envs ending apart) every env's owned
    observation equals its own host reference at each of its decision
    ticks. BREAKS IF a block reads or writes another env's slice, or the
    row map or cadence of one env leaks into another."""
    _run(["batch"], timeout=7200)


def test_torch_gets_the_buffers_themselves_in_stream_order():
    """PROPERTY: the torch tensors of the observation (and team t's row
    views) share the CuPy buffers' memory, and the ordering the hand-over
    rests on holds: the tick stream is blocking and CuPy / torch run on the
    legacy default stream, so torch's read after a tick sees the block.
    BREAKS IF the hand-over copies, or the tick stream becomes non-blocking
    (then a torch read could race the block)."""
    _run(["handover"])


def test_the_observation_adds_no_host_sync_to_a_tick():
    """PROPERTY: with the observation attached a conductor tick still makes
    only its two named syncs (max_n_sub, the books read), at N = 1 and 2,
    while blocks are built. BREAKS IF the observation downloads or syncs
    inside the tick."""
    _run(["syncs"])


def test_an_ended_env_keeps_its_observation():
    """PROPERTY: an env whose ended latch is set is skipped by the blocks
    (its buffers unchanged) while its neighbour is rebuilt and equal to its
    reference. BREAKS IF a block ignores the latch."""
    _run(["ended"])


def test_the_observation_float_kit_rounds_like_the_host():
    """PROPERTY: the device observation's float kit (cpp/src/cuda_obs_float.cuh)
    converts double -> float16 exactly as numpy does on EVERY float16
    rounding case (each half, each midpoint and its float64 neighbours, the
    overflow and underflow edges, signed zeros, NaNs), double -> float32 as
    numpy does, never contracts a*b + c into an FMA, and serves the squash
    (CUDA's log) as the host's float32 on a dense sweep. BREAKS IF the half
    conversion rounds twice through float32 (OBS_BREAK=half_f32), the kit's
    arithmetic is fused (OBS_BREAK=fused), or CUDA's log moves a served
    squashed value (then the shared log module is due)."""
    _run(["floatkit"])
