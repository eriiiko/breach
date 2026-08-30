"""B6 — the cross-machine LOGIC golden + attestation (impl doc §10 B6 / §8).

Arc B impl doc (docs/arc_b_impl_2026-07-21.md v2), the arc's closing patch. This
is the analogue of the Arc-A field/xarch golden, extended to the logic layer: a
small deterministic sim exercising a **sensor -> filter -> decider -> door
feedback loop closed through physics**, trajectory-hashed against a committed
machine-independent constant. Because every synced field + the __entity__ /
__signals__ sections are integer-only (Q16.16), the trajectory digest is a pure
integer function of seed+content, so a second machine reproduces it bit-for-bit
(the cross-GPU/cross-arch attestation the S-series proved for the field path,
now covering the SignalBus too).

The loop (a REAL closed loop through the atmosphere solver, §10 B6):

    pressure sensor (chamber air) --value--> filter (EMA) --out--> decider
      (gt 0.75 atm) --out--> door.open

  · The chamber starts sealed at ~1 atm (door CLOSED), so the probe reads a
    stable high pressure while the filter EMA charges from 0.
  · When the FILTERED pressure crosses the decider threshold, the decider fires
    and (one node hop later) the door is commanded open; (one more tick) it
    flips OPEN.
  · The open door VENTS the chamber to the vacuum beyond it -> the pressure the
    sensor reads collapses. The door's own state has changed the field its
    sensor samples: the loop is closed THROUGH physics, not through a wire only.

What this pins (the four B6 deliverables):
  1. TRAJECTORY DIGEST — the whole 30-tick run (all synced fields + the
     __entity__ /__signals__ sections) hashes to LOOP_GOLDEN_TRAJ_DIGEST, a
     committed constant. This is the cross-machine attestation.
  2. __signals__ CAPTURE — the wired golden captures a NON-EMPTY __signals__
     section every tick (3 slots: the sensor value, the filter out, the decider
     out). field_ab_harness.capture_trajectory folds it via sim._digest_signals()
     (B1 left it () for the wire-free goldens; this is where it is wired up).
  3. LATENCY — the exact tick the loop first actuates, and the 1-tick-per-hop
     contract (§2c): filter-crosses-threshold -> decider-high -> door-flips, one
     tick each.
  4. DORMANCY / no re-baseline (escalation-trigger 3) — the B1 door-present,
     wire-free digest is byte-identical, and a wire-free sim captured through the
     SAME (B6-extended) harness still yields __signals__ == () (no bus => no
     signals => Arc-A bytes).

Provenance: entity-present digests are only comparable at equal
registry_content_hash (the A4 rule — match-setup material, like the seed); the
carrier records it and this module asserts it, so a cross-machine mismatch is
attributable to a registry drift vs a real logic desync.

Run:
    conda run -n data python -m pytest tests/test_b6_logic_golden.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from level_loader import EntityInstance, LevelData, Wire  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.entities.door import DOOR_CLOSED, DOOR_OPEN  # noqa: E402
from simulation.entities.registry import registry_content_hash  # noqa: E402


# ---------------------------------------------------------------------------
# The golden scenario — a synthetic in-memory level (the A7 pattern: goldens
# derive from code-built scenarios, never a committed level file, so the
# constant cannot drift when a level is re-authored).
# ---------------------------------------------------------------------------

# The committed, machine-INDEPENDENT trajectory digest of the loop over
# LOOP_GOLDEN_STEPS ticks (seed=1, physics=breach_physics). Integer-only ⇒ a
# second machine reproduces it bit-for-bit (the cross-machine attestation). This
# is a NEW golden constant (expected, §11) — NOT a re-baseline of an existing one.
# T_ABS COMPRESSION-WORK GOLDEN REBASE (2026-08-21, arc `tabs-compression-work`,
# HUMAN-TEST PASS 2026-08-21, docs/tabs_compression_work_rebaseline_2026-08-21.md).
# The loop closes THROUGH the atmosphere solver (module docstring), so step 4c's
# move onto absolute T (docs/tabs_compression_work_design_2026-08-20.md §1/§2)
# lawfully moves this trajectory too. Re-run twice, independently: identical.
# (was ed42914ebe44d355ab311e0346ce8d9602dd9728887f1fe35fe7a377dc5cb189)
# GAS-ENERGY CONSERVATION ARC, P-G0 GOLDEN REBASE (2026-08-29, digest spec
# bump event 1: v4 -> v5, +gas_energy int64). A SCHEMA move: gas_energy
# joins DIGEST_FIELDS and folds into every tick_digest regardless of
# whether the physics changed (field_ab_harness's canonical-scenario A/B
# diff over 30 ticks proved every pre-existing field byte-identical to
# HEAD; this loop scenario is a different level but exercises the SAME
# unmodified EOS/thermal/pump code paths). (was
# a631c182c5669ebefd390dd321868874bbe17db1cd1f3e3195be1c276ede05dd)
# GAS-ENERGY CONSERVATION ARC, P-G3 GOLDEN REBASE (2026-08-30, value-move
# event 2). Unlike P-G0's schema-only move above, this loop scenario runs
# LIVE physics (breach_physics), so P-G1a's kick-loop KE brackets + sub-cycled
# face-flux energy step (replacing step 4c), P-G1b's live writer seam
# (pumps in particular -- this scenario is a pump/vent loop), and P-G1d's
# face-form wall divergence all move the atmosphere/pump trajectory this
# golden pins. DIGEST_SPEC_VERSION unchanged. (was
# 4fa67f37383c9c3abeedef73699f480e2d7f30d35d37397b34719ad653778769)
LOOP_GOLDEN_TRAJ_DIGEST = \
    "38a47454a12b09b7815c9b95b672e815f9291bb0b3e42c30386fdb2577b3b6b3"
LOOP_GOLDEN_STEPS = 30
LOOP_GOLDEN_SEED = 1

# Latency pins (§2c), in sim-tick terms (sim.tick starts at 1 on the first step).
# The chain has TWO node hops (filter, decider): once the FILTERED pressure
# crosses the threshold (visible in pub at tick 11), the decider is high one hop
# later (tick 12), and the door flips one more tick later (tick 13) — exactly one
# tick per hop, the 2-tick/per-hop contract golden-locked.
# T_ABS COMPRESSION-WORK GOLDEN REBASE (2026-08-21): re-measured against the new
# law and UNMOVED (11/12/13, same as before) — the chamber's pressure trajectory
# up to first threshold-cross is dominated by the sealed-room fill, not by the
# ambient-air 4c term this arc changes; `test_logic_loop_latency_pins_the_per_hop_contract`
# passed green on this arc's own build without edits, confirming no re-pin was needed.
LOOP_FILTER_CROSS_TICK = 11
LOOP_DECIDER_HIGH_TICK = 12
LOOP_DOOR_OPEN_TICK = 13

_DECIDER_THRESHOLD_Q16 = 49152     # 0.75 atm in Q16.16 (65536 == 1.0 atm)


def _inst(class_name, eid, ordinal, **overrides):
    """An EntityInstance with schema defaults + overrides (the B1..B5 idiom)."""
    cls = REGISTRY[class_name]
    fields = {f.name: f.default for f in cls.FIELDS}
    fields.update(overrides)
    return EntityInstance(id=eid, class_name=class_name, ordinal=ordinal,
                          tags=(), fields=fields)


def _loop_tilemap():
    """A 9x14 v2 tilemap (codes: 0 = air, 1 = hull, 9 = SPACE/vacuum).

        rows 3..5, cols 1..6 : the sealed AIR chamber (col 6 is the door gap)
        rows 3..5, cols 7..12: open SPACE (vacuum) — the vent sink
        everything else      : hull

    The pressure probe mounts on the chamber ceiling (a hull tile at row 2) and
    faces DOWN into the chamber air (sample_dy = +1); the vent door sits in the
    chamber's outer wall (col 6). Closed, it seals ~1 atm; open, the chamber
    vents to the vacuum beyond."""
    h, w = 9, 14
    tm = np.full((h, w), 1, dtype=np.int32)   # all hull
    tm[3:6, 1:7] = 0                           # chamber air (incl. door gap col 6)
    tm[3:6, 7:13] = 9                          # vacuum sink (SPACE)
    return tm


def _loop_golden_sim():
    """Build the sensor->filter->decider->door feedback-loop sim (seed +
    physics fixed). Ordinals pin the sweep/serialization order (§9)."""
    probe = _inst("pressure", "chamber_probe", 0, x=3, y=2,
                  sample_dx=0, sample_dy=1)          # faces air (3,3)
    smoother = _inst("filter", "smoother", 1, tau_s=0.35)
    trigger = _inst("decider", "vent_trigger", 2,
                    comparator="gt", threshold=_DECIDER_THRESHOLD_Q16)
    door = _inst("door", "vent_door", 3, x=6, y=3, orientation="v",
                 length_m=3.0, initial_state="closed")
    wires = [
        Wire(0, "value", 1, "in", "single"),    # probe.value  -> filter.in
        Wire(1, "out", 2, "in", "single"),       # filter.out   -> decider.in
        Wire(2, "out", 3, "open", "held"),       # decider.out  -> door.open
    ]
    level = LevelData(name="logic_loop_golden", version="2", path=Path("."),
                      tilemap=_loop_tilemap(), tile_size_m=1.0,
                      diffuse_path=Path("."), entities=[probe, smoother,
                                                        trigger, door],
                      wires=wires)
    return Simulation(level, seed=LOOP_GOLDEN_SEED, breach_physics=bp,
                      enable_recorder=False)


# ---------------------------------------------------------------------------
# 1 + 2. The trajectory digest golden + the wired __signals__ capture
# ---------------------------------------------------------------------------

def test_logic_loop_trajectory_digest_matches_committed_golden():
    from field_ab_harness import capture_trajectory
    from field_digest import trajectory_digest

    traj = capture_trajectory(make_sim=_loop_golden_sim, n_steps=LOOP_GOLDEN_STEPS)
    assert len(traj) == LOOP_GOLDEN_STEPS

    # The WIRED golden captures a NON-EMPTY __signals__ every tick — the three
    # bus slots (probe.value, filter.out, decider.out); `alive` is never a
    # __signals__ row (A4 c7). This is the capture B1 left as () for the
    # wire-free goldens, now exercised for real.
    for i, snap in enumerate(traj):
        carrier = snap["__entity__"]
        assert carrier["n_entities"] == 4, i
        sig_names = sorted(name for (_o, name, _v) in carrier["signals"])
        assert sig_names == ["out", "out", "value"], (i, carrier["signals"])

    assert trajectory_digest(traj) == LOOP_GOLDEN_TRAJ_DIGEST


def test_logic_loop_digest_is_reproducible_and_2x_bit_identical():
    # Determinism (§9): two independent runs are per-cell / per-signal identical
    # AND hash to the same trajectory digest — the property that makes the golden
    # a cross-machine proof (seed + content are the only inputs).
    from field_ab_harness import capture_trajectory, assert_trajectories_match
    from field_digest import trajectory_digest

    a = capture_trajectory(make_sim=_loop_golden_sim, n_steps=LOOP_GOLDEN_STEPS)
    b = capture_trajectory(make_sim=_loop_golden_sim, n_steps=LOOP_GOLDEN_STEPS)
    assert_trajectories_match(a, b)                       # per-cell / per-signal
    assert trajectory_digest(a) == trajectory_digest(b)
    assert trajectory_digest(a) == LOOP_GOLDEN_TRAJ_DIGEST


# ---------------------------------------------------------------------------
# 3. Latency — the exact first-actuation tick + the 1-tick-per-hop contract
# ---------------------------------------------------------------------------

def test_logic_loop_latency_pins_the_per_hop_contract():
    sim = _loop_golden_sim()
    door = sim.door_at(3, 6)                              # (row, col) = (y, x)
    assert door.state == DOOR_CLOSED

    bus = sim._signal_bus
    filt_cross = dec_high = door_open = None
    for _ in range(LOOP_DOOR_OPEN_TICK + 2):
        sim.set_paused(False)
        sim.step()
        t = sim.tick
        # All reads are the POST-swap pub (§2b): filter/decider outputs settle
        # at end-of-tick, the door state is this tick's flip.
        if filt_cross is None and int(bus.read(1, "out")) >= _DECIDER_THRESHOLD_Q16:
            filt_cross = t
        if dec_high is None and int(bus.read(2, "out")) == 1:
            dec_high = t
        if door_open is None and door.state == DOOR_OPEN:
            door_open = t

    assert filt_cross == LOOP_FILTER_CROSS_TICK
    assert dec_high == LOOP_DECIDER_HIGH_TICK
    assert door_open == LOOP_DOOR_OPEN_TICK
    # One tick per node hop (§2c): filter->decider and decider->door are each
    # exactly one tick (the actuator/next node reads the previous swap, never stg).
    assert dec_high - filt_cross == 1
    assert door_open - dec_high == 1


def test_logic_loop_closes_through_physics():
    # The feedback is REAL, not wire-only: the chamber holds a stable high
    # pressure while the door is shut, then the pressure the SENSOR reads
    # collapses within a couple of ticks of the door opening — the door's own
    # state changed the field its sensor samples.
    sim = _loop_golden_sim()
    door = sim.door_at(3, 6)
    chamber = (3, 3)                                      # the sampled air tile

    p_before = None
    for _ in range(LOOP_DOOR_OPEN_TICK):
        sim.set_paused(False)
        sim.step()
        if door.state == DOOR_CLOSED:
            p_before = int(sim.gmap.atmosphere[chamber])
    # Sealed chamber ~ 1 atm (Q16.16) right up to the tick before it opens.
    assert p_before is not None and p_before > 60000
    assert door.state == DOOR_OPEN                        # opened on this tick

    for _ in range(3):                                   # let the vent bite
        sim.set_paused(False)
        sim.step()
    p_after = int(sim.gmap.atmosphere[chamber])
    assert p_after < p_before // 4                        # collapsed toward vacuum


# ---------------------------------------------------------------------------
# Provenance — the xarch attestation line (A4 rule)
# ---------------------------------------------------------------------------

def test_logic_loop_registry_provenance_is_recorded_and_stable():
    from field_ab_harness import capture_trajectory

    traj = capture_trajectory(make_sim=_loop_golden_sim, n_steps=4)
    reg_hash = registry_content_hash()
    # Every entity-present snapshot carries the registry content hash — the
    # match-setup material a cross-machine comparison keys on (digests are only
    # comparable at equal registry_hash). It is constant across the run.
    for snap in traj:
        assert snap["__entity__"]["registry_hash"] == reg_hash
    assert len(reg_hash) == 64                            # blake2b-256 hex


# ---------------------------------------------------------------------------
# 4. Dormancy — no existing golden re-baselined (escalation-trigger 3)
# ---------------------------------------------------------------------------

def test_b1_dormancy_door_present_wire_free_still_byte_identical():
    # The fragile case (§8, D-nit N4): a door-present, WIRE-FREE level must hash
    # byte-identically to its pre-Arc-B baseline. Re-run B1's own gate verbatim
    # (it owns the frozen constant) so B6 cannot silently move it.
    import test_b1_signal_bus as b1
    b1.test_dormancy_door_present_wire_free_digest_byte_identical()


def test_wire_free_capture_through_b6_harness_still_has_empty_signals():
    # The B6 harness change (capture_trajectory now folds sim._digest_signals())
    # must be INERT for a bus-free sim: no wires/sensors/nodes => no SignalBus
    # => _digest_signals() == () => carrier signals () => Arc-A bytes. This is
    # the dormancy of the harness edit itself (the change that could regress a
    # door-only golden, but does not).
    from field_ab_harness import capture_trajectory

    def _door_only_sim():
        h, w = 9, 12
        tm = np.full((h, w), 1, dtype=np.int32)
        tm[3:6, 1:11] = 0                                # air corridor
        door = _inst("door", "d0", 0, x=6, y=3, orientation="v",
                     length_m=3.0, initial_state="closed")
        level = LevelData(name="door_only", version="2", path=Path("."),
                          tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                          entities=[door], wires=[])       # NO wires
        return Simulation(level, seed=1, breach_physics=bp,
                          enable_recorder=False)

    sim = _door_only_sim()
    assert sim._signal_bus is None                        # dormant: no bus
    traj = capture_trajectory(make_sim=_door_only_sim, n_steps=3)
    for snap in traj:
        assert snap["__entity__"]["signals"] == ()        # () despite the door


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
