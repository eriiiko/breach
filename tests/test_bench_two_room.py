"""Gate for the committed two-room storming bench (storm audit 2026-08-14).

Two assertions, per the house audit rules (docs/audit_lessons_and_rules_2026-08-04.md):

R1 (non-vacuousness) — the fixture actually CONTAINS the phenomenon geometry:
two air rooms connected ONLY through a 1-tile door in a hull partition, with a
flammable crate. A gate on a scenario that lacks the door would be green and
blind — exactly the failure that let the storming reach Erik's eyes.

Determinism — the bench run is bit-reproducible: two runs from the same state
produce identical per-field trajectory AND final digests. This is the property
every storm-audit comparison rests on.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bench_two_room as bench  # noqa: E402


def test_fixture_is_two_room_with_door():
    level = bench.load_fixture()   # asserts byte-equality with build_tworoom
    tm = level.tilemap
    h, w = tm.shape
    assert (h, w) == (14, 27)
    # Hull ring sealed.
    assert (tm[0, :] == 1).all() and (tm[-1, :] == 1).all()
    assert (tm[:, 0] == 1).all() and (tm[:, -1] == 1).all()
    # Partition column carries EXACTLY one door (air) tile.
    mid = 13
    door_rows = [r for r in range(h) if tm[r, mid] == 0]
    assert door_rows == [6], f"expected a 1-tile door at row 6, got {door_rows}"
    assert ((tm[:, mid] == 1) | (tm[:, mid] == 0)).all()
    # One flammable FURNITURE crate, in the left room.
    crates = list(zip(*np.where(tm == 6)))
    assert crates == [(7, 7)]
    # The two rooms are otherwise disjoint: flood-fill from the left room
    # with the door SEALED must not reach the right room.
    blocked = tm.copy()
    blocked[6, mid] = 1
    lab, n = _label4(blocked != 1)
    assert n >= 2, "sealing the door must disconnect the rooms"
    assert lab[7, 3] != lab[7, 20], "rooms connect around the partition?!"


def _label4(mask):
    """4-connected component labelling, numpy only.

    This used scipy.ndimage.label. breach has exactly one scipy call in its whole
    test suite, and the shared conda env ships a scipy built against numpy 1.x that
    raises ImportError under numpy 2 -- so the one call was turning a green suite
    red for a reason that has nothing to do with breach. Fifteen lines here make the
    project scipy-free instead, which is also one fewer thing to keep in step across
    machines.
    """
    lab = np.zeros(mask.shape, np.int32)
    cur = 0
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or lab[sy, sx]:
                continue
            cur += 1
            stack = [(sy, sx)]
            lab[sy, sx] = cur
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                        lab[ny, nx] = cur
                        stack.append((ny, nx))
    return lab, cur


def test_bench_runs_and_is_deterministic():
    n_ticks = 120  # 5 s — enough for ignition + the first door transient
    a = bench.run_bench(ticks=n_ticks, dials=dict(bench.sp.PF1B))
    b = bench.run_bench(ticks=n_ticks, dials=dict(bench.sp.PF1B))
    # Non-vacuous: the fire burned and the atmosphere moved.
    ke_peak = a["summary"]["ke_peak"]
    assert ke_peak > 0.01, f"bench fire produced no wind (ke_peak={ke_peak})"
    assert max(r["I"] for r in a["res"]["rows"]) > 0.05, "crate never burned"
    # Deterministic: bit-identical trajectories, both digest layers.
    assert a["digests"]["trajectory"] == b["digests"]["trajectory"]
    assert a["digests"]["final"] == b["digests"]["final"]
