"""#29 P7e.5 -- the device conductor over N envs (pytest wrapper of
tests/cuda_conductor_batch_check.py). One test per property; each runs its
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


def _run(parts, timeout=3600):
    proc = cuda_harness.run_cuda_script(
        "import cuda_conductor_batch_check, sys; "
        f"sys.exit(cuda_conductor_batch_check.main({list(parts)!r}))",
        timeout=timeout,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "CONDUCTOR_BATCH_RESULT: PASS" in out, out[-6000:]


def test_each_env_of_a_batch_equals_itself_alone_on_the_hand_scenes():
    """PROPERTY: env i of a batch of N (2, 4) of the hand scenes -- the envs'
    intents different (a per-env delay of the scene's schedule; scenes that
    share a level side by side) -- equals env i run alone by the conductor:
    (field_digest, tick_digest) and the state in no digest at every tick.
    BREAKS IF a face reads or writes another env's rows, planes, tables or
    lists, or a latch lands on the wrong env."""
    _run(["scenes"])


def test_each_env_of_a_batch_equals_itself_alone_on_recorded_matches():
    """PROPERTY: ship_demo matches recorded with different seeds per env (two
    specs), replayed together to their ends (envs end at different ticks; an
    ended env ticks on with no intents), each env equal to itself alone at
    every tick. BREAKS IF an env's end, length or intents leak into another,
    or a range is crossed in a batch that a lone env does not cross."""
    _run(["matches"], timeout=7200)


def test_each_env_of_a_swarm_batch_equals_itself_alone():
    """PROPERTY: larva_studio envs with different seeds (the broods differ),
    swarm units alive, each env of N = 2, 4 equal to itself alone every tick;
    and an env alone equal to the CPU reference (the swarm's launch words
    from the device tick row). BREAKS IF the swarm reads another env's tick
    or key, or the host clock instead of the device row."""
    _run(["swarm"])


def test_an_envs_state_never_depends_on_its_batch_index():
    """PROPERTY: the same envs in another order give every env the same
    fingerprints at every tick. BREAKS IF an env's index (or a neighbour)
    enters its state (shown red with CONDUCTOR_BATCH_BREAK=index: the index
    leaking into its draw key)."""
    _run(["swap"])


def test_the_sync_count_of_a_batch_tick_does_not_depend_on_n():
    """PROPERTY: a steady conductor tick makes the same host syncs at N = 1,
    2 and 4, and only the named reads (the EOS stage's max_n_sub, the one
    books read). BREAKS IF a per-env download or sync enters the walk (shown
    red with CONDUCTOR_BATCH_BREAK=sync) or the physics books get their own
    sync again."""
    _run(["syncs"])


def test_a_fault_in_env_k_names_env_k():
    """PROPERTY: a refusal met in one env of a batch is raised by name with
    that env (``.env`` and the message) and refuses the batch. BREAKS IF the
    books read maps a count to the wrong env (shown red with
    CONDUCTOR_BATCH_BREAK=layout: the count block read env-major)."""
    _run(["fault"])


def test_a_quiet_env_and_a_blasting_env_each_equal_themselves_alone():
    """#217 §8.9 T7 -- PART BLAST: a quiet env and a blasting env of one level
    in one batch, each env == itself alone at every tick, and their sub-tick
    counts k differ on some tick. Breaks if an env past its own k is not
    masked (a tick_off missing on any EOS kernel), or reads another env's
    (k, n_sub) row."""
    _run(["blast"])
