"""tools/bench_octopus.py (#106 P6e): the octopus cost report's SHAPE and the
timer's inertness. Never asserts a time -- cost is reported, never gated."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests",
           ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(bp, "arm_chain_tick"),
                                reason="this build has no arm chain solver")


def test_part_laps_charge_each_interval_to_its_part_and_ignore_laps_outside_a_tick():
    """The part timer charges the clock's advance since the previous lap to
    the named part, opens a tick on "enter", ignores laps before it (a
    spawn's write_unit) and resets on take(). Breaks if a lap is charged to
    the wrong part or an interval is counted twice."""
    import bench_octopus as B
    t = iter([5, 10, 13, 20, 26, 40, 41, 50])
    laps = B.PartLaps(clock=lambda: next(t))
    laps.lap("hub")                      # 5: no open tick -> ignored
    laps.lap("enter")                    # 10
    laps.lap("setup")                    # 13 -> setup 3
    laps.lap("hub")                      # 20 -> hub 7
    laps.lap("chain")                    # 26 -> chain 6
    laps.lap("hub")                      # 40 -> hub 14
    row = laps.take()
    assert set(row) == set(B.PARTS)
    assert row["setup"] == 3 and row["hub"] == 21 and row["chain"] == 6
    laps.lap("chain")                    # 41: closed by take() -> ignored
    assert sum(laps.take().values()) == 0


def test_the_slot5b_timer_is_detached_by_default():
    """Slot 5b's timer is OFF unless a bench attaches it; set_tick_laps
    returns what was attached and restores the no-op. Breaks if the sim
    ever runs with a live timer nobody asked for."""
    from simulation import creature_system as CS
    assert CS.set_tick_laps(None) is None
    class Marker:
        def lap(self, part):
            pass
    marker = Marker()
    assert CS.set_tick_laps(marker) is None
    assert CS.set_tick_laps(None) is marker
    assert CS.set_tick_laps(None) is None


def test_the_report_has_every_part_and_the_timer_moves_no_digest(tmp_path):
    """A few crawl ticks in the gate room with and without the part timer:
    the final tick_digest is IDENTICAL (the timer is inert: no state, no
    order change); the report carries every part, the slot total, record,
    step, rest, the per-octopus means, the share and the ns per weighted op;
    the parts sum to the slot total, and the slot fits inside the step it
    was measured in. Breaks if the hook changes what slot 5b does, a part
    goes missing, or the split stops adding up."""
    import bench_octopus as B
    on = B.run_room("crawl", ticks=6, warmup=2, tmp=tmp_path / "on")
    off = B.run_room("crawl", ticks=6, warmup=2, timing=False, tmp=tmp_path / "off")
    base = B.run_room("crawl", ticks=6, warmup=2, octopus=False, tmp=tmp_path / "base")
    assert on.digest == off.digest
    assert len(on.ticks) == 6
    for row in on.ticks:
        slot = sum(row[k] for k in B.PARTS)
        assert row["chain"] > 0 and row["hub"] > 0 and row["hit_shape"] > 0
        assert 0 < slot <= row["step"]
    assert all(sum(r[k] for k in B.PARTS) == 0 for r in off.ticks)
    rep = B.summarize([on], [base], tps=24, arms=8, substeps=16)
    for k in B.PARTS + ("slot5b", "record", "step", "rest", "ctrl", "total"):
        assert set(rep["cols"][k]) == {"mean", "p50", "p95", "max"}, k
    assert set(rep["per_octopus_ms"]) == set(B.PARTS) | {"slot5b", "record"}
    parts = sum(rep["per_octopus_ms"][k] for k in B.PARTS)
    assert parts == pytest.approx(rep["per_octopus_ms"]["slot5b"], rel=1e-9)
    assert rep["cols"]["rest"]["mean"] == pytest.approx(
        rep["cols"]["step"]["mean"] - rep["cols"]["slot5b"]["mean"], rel=1e-9)
    assert rep["weighted_ops_per_octopus_tick"] == B.WEIGHTED_OPS_PER_ARM_SUBSTEP * 8 * 16
    assert rep["chain_ns_per_op"] > 0 and rep["chain_ms_per_arm_substep"] > 0
    assert "baseline" in rep and rep["realtime"] > 0
    md = B.markdown([rep], "cpu")
    for k in B.PARTS + ("slot5b",):
        assert k in md
