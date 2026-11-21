"""#29 P7e.5 -- the device conductor over N envs (pytest wrapper of
tests/cuda_conductor_batch_check.py). One test per property; the selected
tests share ONE isolated CUDA subprocess (cuda_harness; the session fixture
``batch_run``), each judged on its own parts, and SKIP cleanly without a CUDA
build / device.
"""
from __future__ import annotations

import pytest

import _check_parts
import cuda_harness

pytestmark = pytest.mark.skipif(
    not cuda_harness.cuda_available(),
    reason="no CUDA build (cpp/build_cuda) or CUDA runtime DLLs present",
)


#: test -> (its parts, the harness timeout its parts had alone, s).
PARTS_OF = {
    "test_each_env_of_a_batch_equals_itself_alone_on_the_hand_scenes": (['scenes'], 3600),
    "test_each_env_of_a_batch_equals_itself_alone_on_recorded_matches": (['matches'], 7200),
    "test_each_env_of_a_swarm_batch_equals_itself_alone": (['swarm'], 3600),
    "test_an_envs_state_never_depends_on_its_batch_index": (['swap'], 3600),
    "test_the_sync_count_of_a_batch_tick_does_not_depend_on_n": (['syncs'], 3600),
    "test_an_env_reset_mid_batch_equals_its_episode_built_alone": (['reset', 'reset_refusals'], 7200),
    "test_a_fault_in_env_k_names_env_k": (['fault'], 3600),
}


@pytest.fixture(scope="session")
def batch_run(request):
    """ONE subprocess of tests/cuda_conductor_batch_check.py for every selected test of this
    module (DEVICE CHECK PLAYS): the host references the parts share are
    played once; each part is judged alone and each test reads only its own
    parts' verdicts (``_check_parts``). Under xdist every test here is in the
    one device group (``_suite_kinds.is_device``): one worker, one run."""
    return _check_parts.shared_run(
        request, __file__, PARTS_OF, "cuda_conductor_batch_check",
        lambda body, timeout: cuda_harness.run_cuda_script(body, timeout=timeout))


def _run(run, test_name):
    _check_parts.assert_parts(run, PARTS_OF[test_name][0])


def test_each_env_of_a_batch_equals_itself_alone_on_the_hand_scenes(batch_run):
    """PROPERTY: env i of a batch of N (2, 4) of the hand scenes -- the envs'
    intents different (a per-env delay of the scene's schedule; scenes that
    share a level side by side) -- equals env i run alone by the conductor:
    (field_digest, tick_digest) and the state in no digest at every tick.
    BREAKS IF a face reads or writes another env's rows, planes, tables or
    lists, or a latch lands on the wrong env."""
    _run(batch_run, "test_each_env_of_a_batch_equals_itself_alone_on_the_hand_scenes")


def test_each_env_of_a_batch_equals_itself_alone_on_recorded_matches(batch_run):
    """PROPERTY: ship_demo matches recorded with different seeds per env (two
    specs), replayed together to their ends (envs end at different ticks; an
    ended env ticks on with no intents), each env equal to itself alone at
    every tick. BREAKS IF an env's end, length or intents leak into another,
    or a range is crossed in a batch that a lone env does not cross."""
    _run(batch_run, "test_each_env_of_a_batch_equals_itself_alone_on_recorded_matches")


def test_each_env_of_a_swarm_batch_equals_itself_alone(batch_run):
    """PROPERTY: larva_studio envs with different seeds (the broods differ),
    swarm units alive, each env of N = 2, 4 equal to itself alone every tick;
    and an env alone equal to the CPU reference (the swarm's launch words
    from the device tick row). BREAKS IF the swarm reads another env's tick
    or key, or the host clock instead of the device row."""
    _run(batch_run, "test_each_env_of_a_swarm_batch_equals_itself_alone")


def test_an_envs_state_never_depends_on_its_batch_index(batch_run):
    """PROPERTY: the same envs in another order give every env the same
    fingerprints at every tick. BREAKS IF an env's index (or a neighbour)
    enters its state (shown red with CONDUCTOR_BATCH_BREAK=index: the index
    leaking into its draw key)."""
    _run(batch_run, "test_an_envs_state_never_depends_on_its_batch_index")


def test_the_sync_count_of_a_batch_tick_does_not_depend_on_n(batch_run):
    """PROPERTY: a steady conductor tick makes the same host syncs at N = 1,
    2 and 4, and only the named reads (the EOS stage's max_n_sub, the one
    books read). BREAKS IF a per-env download or sync enters the walk (shown
    red with CONDUCTOR_BATCH_BREAK=sync) or the physics books get their own
    sync again."""
    _run(batch_run, "test_the_sync_count_of_a_batch_tick_does_not_depend_on_n")


def test_an_env_reset_mid_batch_equals_its_episode_built_alone(batch_run):
    """PROPERTY (#29 P8.9a): env e reset mid-batch (``DeviceConductor.
    reset_env``) to a new episode (ship_demo with bots and zombies;
    larva_studio with swarm units alive), resets at different ticks for
    different envs, the same env twice and a reset on the tick after another
    env's, at N = 2 and 4: the reset env equals a fresh N = 1 conductor of the
    same (level, seed) at every tick, and every other env equals the same
    batch run without the reset at the reset tick and every tick after
    (digests, the state in no digest, swarm deaths). BREAKS IF a reset
    leaves a table unstaged (CONDUCTOR_BATCH_BREAK=reset_table), keeps the
    old episode's draw key (reset_key), writes a row of the neighbouring env
    (reset_neighbour), or touches or pauses another env. And a sim that does
    not fit the batch (another grid, more units than U) is refused by name
    with nothing written."""
    _run(batch_run, "test_an_env_reset_mid_batch_equals_its_episode_built_alone")


def test_a_fault_in_env_k_names_env_k(batch_run):
    """PROPERTY: a refusal met in one env of a batch is raised by name with
    that env (``.env`` and the message) and refuses the batch. BREAKS IF the
    books read maps a count to the wrong env (shown red with
    CONDUCTOR_BATCH_BREAK=layout: the count block read env-major)."""
    _run(batch_run, "test_a_fault_in_env_k_names_env_k")
