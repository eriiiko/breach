"""#29 P7e.4 -- the device conductor at N = 1 (pytest wrapper of
tests/cuda_conductor_check.py). One test per stage of the patch; the selected
tests share ONE isolated CUDA subprocess (cuda_harness; the session fixture
``conductor_run``), each judged on its own parts, and SKIP cleanly without a
CUDA build / device.
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
    "test_the_conductor_plays_the_hand_scenes_equal_every_tick": (['scenes'], 3600),
    "test_the_conductor_restages_nothing_from_the_host_between_ticks": (['resident'], 3600),
    "test_the_conductor_replays_a_recorded_match_equal_every_tick": (['match'], 3600),
    "test_the_conductor_replays_the_pinned_matches_to_their_end": (['pinned'], 7200),
    "test_a_conductor_tick_syncs_only_at_its_named_reads": (['syncs'], 3600),
}


@pytest.fixture(scope="session")
def conductor_run(request):
    """ONE subprocess of tests/cuda_conductor_check.py for every selected test of this
    module (DEVICE CHECK PLAYS): the host references the parts share are
    played once; each part is judged alone and each test reads only its own
    parts' verdicts (``_check_parts``). Under xdist every test here is in the
    one device group (``_suite_kinds.is_device``): one worker, one run."""
    return _check_parts.shared_run(
        request, __file__, PARTS_OF, "cuda_conductor_check",
        lambda body, timeout: cuda_harness.run_cuda_script(body, timeout=timeout))


def _run(run, test_name):
    _check_parts.assert_parts(run, PARTS_OF[test_name][0])


def test_the_conductor_plays_the_hand_scenes_equal_every_tick(conductor_run):
    """PROPERTY (S1): the hand scenes of the device checks (payloads, hazards,
    sprays, the horde with bashing, signals, fire, walking), driven by
    intents alone, played by the device conductor beside a host
    re-simulation: (field_digest, tick_digest) and the cross-tick state in no
    digest equal at every tick. BREAKS IF a slot runs out of the reference's
    order, a face reads a host input instead of a device one, a latch writes
    a column its intent does not, or a face's law differs."""
    _run(conductor_run, "test_the_conductor_plays_the_hand_scenes_equal_every_tick")


def test_the_conductor_restages_nothing_from_the_host_between_ticks(conductor_run):
    """PROPERTY (S1, residency): once built, no conductor tick uploads the
    store, its tables, an object table, the horde cache or a map plane from
    the host. BREAKS IF a tick restages any of them (shown red once with
    CONDUCTOR_BREAK=restage)."""
    _run(conductor_run, "test_the_conductor_restages_nothing_from_the_host_between_ticks")


def test_the_conductor_replays_a_recorded_match_equal_every_tick(conductor_run):
    """PROPERTY (S2, the latch writer): a ship_demo match recorded through the
    game's recorder path and replayed -- the reference by the replay
    controller, the conductor from the recording's intent rows through its
    column scatter -- equal at every tick. BREAKS IF a recorded intent
    touches state the latch writer does not write, or writes it otherwise."""
    _run(conductor_run, "test_the_conductor_replays_a_recorded_match_equal_every_tick")


def test_the_conductor_replays_the_pinned_matches_to_their_end(conductor_run):
    """PROPERTY (S3): the two pinned ship_demo matches (bots, bots_v1),
    recorded through the game's recorder and fed to the conductor from the
    recording's intent rows to their end (whatever tick that is; the
    recording play itself the reference, played once), equal at every tick
    and ended at the reference's ending tick, or refused BY NAME at a form
    the device does not have (never compared past it). BREAKS IF any slot
    diverges over a full match, or a range or capacity is crossed silently."""
    _run(conductor_run, "test_the_conductor_replays_the_pinned_matches_to_their_end")


def test_a_conductor_tick_syncs_only_at_its_named_reads(conductor_run):
    """PROPERTY (S4): a conductor tick outside the gate's download makes no
    device-to-host sync but its named ones (the books read, the physics'
    max_n_sub and books). BREAKS IF a face downloads its counts at its own
    launch again, or a new host read enters the walk."""
    _run(conductor_run, "test_a_conductor_tick_syncs_only_at_its_named_reads")
