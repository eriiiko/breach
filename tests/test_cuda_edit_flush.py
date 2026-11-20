"""#29 P7e.3 -- the edit door, step (c) (pytest wrapper of
tests/cuda_edit_flush_check.py). One test per property the patch moved onto
the device; each runs its part in an isolated CUDA subprocess (cuda_harness)
and SKIPS cleanly without a CUDA build / device.
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
        "import cuda_edit_flush_check, sys; "
        f"sys.exit(cuda_edit_flush_check.main({list(parts)!r}))",
        timeout=timeout,
    )
    out = proc.stdout + "\n" + proc.stderr
    assert "EDIT_FLUSH_RESULT: PASS" in out, out[-6000:]


def test_the_flush_sorts_the_door_window_on_the_device():
    """PROPERTY (item 1): the buffers K1 / K2 read are built on the device
    from the door's rows -- each env's edits in the host's stable order
    (field, source, region, seq), the tiles carried with their edit, the edit
    index and offsets batch-global, the scalars from the tick row and the
    tables -- equal to the host pack's layout, and the flush equal to the
    host-pack flush, at N = 1 and 2, for windows that arrive out of order with
    tied keys. BREAKS IF the rank ignores a key or the tie, a tile leaves its
    edit, a scalar is read elsewhere, or the device path calls the host's
    sort, pack or buffer (they raise during the run)."""
    _run(["sort"])


def test_the_flush_refuses_by_name_what_the_host_pack_refuses():
    """PROPERTY (item 2): every row the host pack refuses, raises on or falls
    back on (water_depth, a slice past G, a conservative slice, den 2^62, a
    contribution of 2^55, a product past 2^125) refuses its env by name with
    the other env applied; the row just inside each numeric bound is applied
    equal to the host pack. BREAKS IF a range check is dropped, moved past its
    bound, clamps instead of refusing, or refuses the whole batch."""
    _run(["range"])


def test_the_door_window_crosses_the_tick_boundary():
    """PROPERTY (item 3): rows written after a flush survive the tick row's
    advance in the door and are applied with the next tick's rows at the next
    flush, at the new tick's draws. BREAKS IF anything but the flush empties
    the door, or the flush applies only part of the window."""
    _run(["window"])


def test_seq_resets_only_at_the_flush_and_is_absolute():
    """PROPERTY (item 4): the flush resets every env's seq to 0 (the host
    queue's next seq), a noisy edit draws at its ABSOLUTE seq, and an env
    reset resets that env alone. BREAKS IF the flush leaves the seq, the key
    column is rebased, or reset_env touches another env."""
    _run(["seq"])


def test_played_scenes_flush_equal_from_the_device_door():
    """PROPERTY (items 1-5 on played scenes): on the spray, payload and hazard
    scenes at N = 1 and 2, every tick, the device door holds the host queue's
    window (across the tick boundary) and its device flush equals the
    host-pack flush in planes and books, the door empty after it; the scenes
    reach out-of-order windows with ties, noisy edits and windows spanning
    the boundary. BREAKS IF any of items 1-4 breaks, or the books differ from
    the host fold's."""
    _run(["scenes"])
