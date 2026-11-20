"""#29 P7e.4 -- the device conductor at N = 1 (pytest wrapper of
tests/cuda_conductor_check.py). One test per stage of the patch; each runs
its part in an isolated CUDA subprocess (cuda_harness) and SKIPS cleanly
without a CUDA build / device.
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
        "import cuda_conductor_check, sys; "
        f"sys.exit(cuda_conductor_check.main({list(parts)!r}))",
        timeout=timeout,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "CONDUCTOR_RESULT: PASS" in out, out[-6000:]


def test_the_conductor_plays_the_hand_scenes_equal_every_tick():
    """PROPERTY (S1): the hand scenes of the device checks (payloads, hazards,
    sprays, the horde with bashing, signals, fire, walking), driven by
    intents alone, played by the device conductor beside a host
    re-simulation: (field_digest, tick_digest) and the cross-tick state in no
    digest equal at every tick. BREAKS IF a slot runs out of the reference's
    order, a face reads a host input instead of a device one, a latch writes
    a column its intent does not, or a face's law differs."""
    _run(["scenes"])


def test_the_conductor_restages_nothing_from_the_host_between_ticks():
    """PROPERTY (S1, residency): once built, no conductor tick uploads the
    store, its tables, an object table, the horde cache or a map plane from
    the host. BREAKS IF a tick restages any of them (shown red once with
    CONDUCTOR_BREAK=restage)."""
    _run(["resident"])


def test_the_conductor_replays_a_recorded_match_equal_every_tick():
    """PROPERTY (S2, the latch writer): a ship_demo match recorded through the
    game's recorder path and replayed -- the reference by the replay
    controller, the conductor from the recording's intent rows through its
    column scatter -- equal at every tick. BREAKS IF a recorded intent
    touches state the latch writer does not write, or writes it otherwise."""
    _run(["match"])


def test_the_conductor_replays_the_pinned_matches_to_their_end():
    """PROPERTY (S3): the two pinned ship_demo matches (bots, bots_v1),
    recorded and replayed through the conductor to their last tick, equal at
    every tick, or refused BY NAME at a form the device does not have (never
    compared past it). BREAKS IF any slot diverges over a full match, or a
    range or capacity is crossed silently."""
    _run(["pinned"], timeout=7200)


def test_a_conductor_tick_syncs_only_at_its_named_reads():
    """PROPERTY (S4): a conductor tick outside the gate's download makes no
    device-to-host sync but its named ones (the books read, the physics'
    max_n_sub and books). BREAKS IF a face downloads its counts at its own
    launch again, or a new host read enters the walk."""
    _run(["syncs"])
