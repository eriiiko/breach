"""The status/condition system (mechanics/06 §4) — P3 gate.

Covers the system core (this file grows with the P3 stages):

  - the registry: kinds are distinct table-ordered ints; CC rows carry flag
    suppressions and no damage type; DoT/HoT rows carry their §2 dtype;
  - apply_status: door-2 magnitude quantization, integer-duration loudness,
    and the three stacking rules (refresh / stack / max);
  - tick_statuses: expiry, P0 (unit-id, list-order) processing, DoT/HoT
    emission THROUGH the DamagePacket pipeline — so mitigation composes for
    free (the zombie-BURNING-at-4x-a-marine test), heals are negative-
    direction and unresisted, DoT deaths never convert, statuses freeze on
    corpses;
  - composed_flags: the AND-of-suppressions (+ OR prone) the consumers read;
  - determinism: identical runs produce identical status lists + hp traces;
  - (stage 2) can_move / can_act suppression end-to-end on a tiny sim;
  - (stage 3) digest inclusion: the status list is hashed synced state.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_status_system.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

from config import CFG  # noqa: E402
from simulation import unit_fixed  # noqa: E402
from simulation.damage import ASPHYX, HEAL, HEAT, POISON  # noqa: E402
from simulation.events import UnitHitEvent, UnitKilledEvent  # noqa: E402
from simulation.status import (  # noqa: E402
    BLINDED, BURNING, IMMOBILIZED, KNOCKED_DOWN, N_STATUS_KINDS, PARALYZED,
    POISONED, REGEN, STACK_MAX, STACK_REFRESH, STACK_STACK, STATUS_REGISTRY,
    STUNNED, SUFFOCATING, ComposedFlags, apply_status, composed_flags,
    serialize_statuses, tick_statuses,
)
from simulation.unit import LifeState, Unit  # noqa: E402


def _marine(name="M", x=2, y=2):
    return Unit(name, x=x, y=y, team=0)


def _zombie(name="Z", x=6, y=6):
    return Unit(name, x=x, y=y, team=1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_kinds_distinct_table_ordered_and_complete():
    kinds = (KNOCKED_DOWN, IMMOBILIZED, STUNNED, PARALYZED,
             BURNING, POISONED, SUFFOCATING, REGEN, BLINDED)
    assert len(set(kinds)) == len(kinds) == N_STATUS_KINDS
    assert sorted(kinds) == list(range(N_STATUS_KINDS))
    assert len(STATUS_REGISTRY) == N_STATUS_KINDS
    for i, row in enumerate(STATUS_REGISTRY):
        assert row.kind == i, "registry must be indexed by kind"
        assert row.stacking in (STACK_REFRESH, STACK_STACK, STACK_MAX)


def test_registry_cc_rows_suppress_flags_and_carry_no_dtype():
    """The spec's CC roster: knocked_down = prone + no move/act; immobilized /
    stunned / paralyzed are flag variants of the same machinery."""
    kd = STATUS_REGISTRY[KNOCKED_DOWN]
    assert (kd.can_move, kd.can_act, kd.is_prone) == (False, False, True)
    im = STATUS_REGISTRY[IMMOBILIZED]
    assert (im.can_move, im.can_act) == (False, True)
    st = STATUS_REGISTRY[STUNNED]
    assert (st.can_move, st.can_act, st.can_aim) == (True, False, False)
    pa = STATUS_REGISTRY[PARALYZED]
    assert (pa.can_move, pa.can_act, pa.can_aim) == (False, False, False)
    for k in (KNOCKED_DOWN, IMMOBILIZED, STUNNED, PARALYZED):
        assert STATUS_REGISTRY[k].dtype is None, "CC kinds emit nothing"


def test_registry_dot_hot_rows_declare_pipeline_dtypes():
    """DoT/HoT kinds declare the §2 damage type they emit — composition with
    mitigation is structural, never coded per-status."""
    assert STATUS_REGISTRY[BURNING].dtype == HEAT
    assert STATUS_REGISTRY[POISONED].dtype == POISON
    assert STATUS_REGISTRY[SUFFOCATING].dtype == ASPHYX
    assert STATUS_REGISTRY[REGEN].dtype == HEAL
    for k in (BURNING, POISONED, SUFFOCATING, REGEN):
        row = STATUS_REGISTRY[k]
        assert (row.can_move, row.can_act, row.can_aim, row.is_prone) == \
            (True, True, True, False), "v1 DoT/HoT rows suppress no flags"


# ---------------------------------------------------------------------------
# apply_status — doors + stacking
# ---------------------------------------------------------------------------
def test_apply_status_quantizes_magnitude_door_2():
    """The authored magnitude snaps onto the Q16.16 grid ONCE at application;
    the instance stores the INTEGER count (L2 representation)."""
    u = _marine()
    st = apply_status(u, BURNING, magnitude=0.1, duration_ticks=3)
    assert isinstance(st.magnitude_q16, int)
    assert st.magnitude_q16 == unit_fixed.quantize_scalar(0.1) == 6554
    # Exact dyadics pass through exactly.
    st2 = apply_status(_marine(), POISONED, magnitude=6.25, duration_ticks=2)
    assert st2.magnitude_q16 == 6.25 * 65536


def test_apply_status_duration_must_be_positive_integer_ticks():
    u = _marine()
    with pytest.raises(ValueError):
        apply_status(u, BURNING, magnitude=1.0, duration_ticks=0)
    with pytest.raises(ValueError):
        apply_status(u, BURNING, magnitude=1.0, duration_ticks=-3)
    st = apply_status(u, BURNING, magnitude=1.0, duration_ticks=5)
    assert st.remaining_ticks == 5 and isinstance(st.remaining_ticks, int)


def test_apply_status_creates_list_on_bare_stub_units():
    stub = SimpleNamespace(id=0, alive=True)
    apply_status(stub, IMMOBILIZED, magnitude=0.0, duration_ticks=2)
    assert len(stub.statuses) == 1


def test_stacking_refresh_one_instance_timer_reset():
    u = _marine()
    apply_status(u, KNOCKED_DOWN, magnitude=0.0, duration_ticks=4, source_id=7)
    tick_statuses([u])          # 4 -> 3
    st = apply_status(u, KNOCKED_DOWN, magnitude=1.0, duration_ticks=6,
                      source_id=9)
    assert len(u.statuses) == 1, "refresh keeps ONE instance"
    assert st is u.statuses[0]
    assert st.remaining_ticks == 6
    assert st.magnitude_q16 == 65536
    assert st.source_id == 9


def test_stacking_stack_instances_coexist_and_expire_independently():
    u = _marine()
    apply_status(u, POISONED, magnitude=1.0, duration_ticks=1, source_id=1)
    apply_status(u, POISONED, magnitude=2.0, duration_ticks=3, source_id=2)
    assert len(u.statuses) == 2, "stack appends a NEW instance"
    hp0 = u.current_hp
    tick_statuses([u])          # both emit; the 1-tick dose hits 0
    assert u.current_hp == hp0 - 3.0
    assert [st.remaining_ticks for st in u.statuses] == [0, 2]
    tick_statuses([u])          # dose-1 swept; the survivor ticks alone
    assert u.current_hp == hp0 - 5.0
    assert len(u.statuses) == 1 and u.statuses[0].source_id == 2


def test_stacking_max_keeps_strongest_magnitude_and_longest_timer():
    u = _marine()
    apply_status(u, BURNING, magnitude=8.0, duration_ticks=2, source_id=1)
    # A weaker, longer re-ignition: magnitude stays, timer extends, source stays.
    st = apply_status(u, BURNING, magnitude=3.0, duration_ticks=5, source_id=2)
    assert len(u.statuses) == 1, "max keeps ONE instance"
    assert st.magnitude_q16 == 8 * 65536
    assert st.remaining_ticks == 5
    assert st.source_id == 1, "source updates only on strictly greater magnitude"
    # A stronger burn upgrades magnitude AND takes the attribution.
    apply_status(u, BURNING, magnitude=12.0, duration_ticks=3, source_id=3)
    assert st.magnitude_q16 == 12 * 65536
    assert st.remaining_ticks == 5, "timer never shortens under max"
    assert st.source_id == 3


# ---------------------------------------------------------------------------
# tick_statuses — expiry, order, emission
# ---------------------------------------------------------------------------
def test_expiry_duration_n_suppresses_exactly_n_ticks_lazy_sweep():
    """The duration contract: N = exactly N ticks of suppression. The pass
    runs BEFORE each tick's flag consumers, so expiry is LAZY — the status
    hits 0 and keeps suppressing that tick; the NEXT pass sweeps it."""
    u = _marine()
    apply_status(u, STUNNED, magnitude=0.0, duration_ticks=2)
    tick_statuses([u])          # tick 1: 2 -> 1 (suppressed tick #1)
    assert len(u.statuses) == 1 and u.statuses[0].remaining_ticks == 1
    assert composed_flags(u).can_act is False
    tick_statuses([u])          # tick 2: 1 -> 0 (suppressed tick #2, the last)
    assert len(u.statuses) == 1 and u.statuses[0].remaining_ticks == 0
    assert composed_flags(u).can_act is False, \
        "duration 2 means TWO suppressed ticks — 0-count still holds this one"
    tick_statuses([u])          # tick 3: swept at the top — released
    assert u.statuses == []
    assert composed_flags(u).can_act is True
    tick_statuses([u])          # empty list is a clean no-op
    assert u.statuses == []


def test_expiry_compacts_in_place_held_reference_stays_valid():
    u = _marine()
    apply_status(u, STUNNED, magnitude=0.0, duration_ticks=1)
    held = u.statuses
    tick_statuses([u])          # 1 -> 0 (its one suppressed tick)
    tick_statuses([u])          # swept
    assert held is u.statuses and held == []


def test_dot_emits_exactly_duration_packets():
    u = _marine()
    apply_status(u, BURNING, magnitude=2.0, duration_ticks=3)
    hp0 = u.current_hp
    events = []
    for _ in range(6):          # over-run past expiry
        tick_statuses([u], events=events)
    assert u.current_hp == hp0 - 6.0, "exactly N=3 emissions of 2.0"
    hits = [e for e in events if isinstance(e, UnitHitEvent)]
    assert len(hits) == 3
    assert all(e.source == "burning" and e.damage == 2.0 for e in hits)


def test_units_processed_in_id_order_not_list_order():
    """P0: the pass sorts by unit id — a scrambled list emits in id order."""
    u1, u2, u3 = _marine("A"), _marine("B"), _marine("C")
    u1.id, u2.id, u3.id = 1, 2, 3
    for u in (u1, u2, u3):
        apply_status(u, POISONED, magnitude=1.0, duration_ticks=1)
    events = []
    tick_statuses([u3, u1, u2], events=events)      # scrambled input order
    hit_ids = [e.unit_id for e in events if isinstance(e, UnitHitEvent)]
    assert hit_ids == [1, 2, 3]


def test_statuses_freeze_on_dead_units():
    u = _marine()
    u.alive = False
    apply_status(u, BURNING, magnitude=5.0, duration_ticks=4)
    hp0 = u.current_hp
    events = []
    tick_statuses([u], events=events)
    assert u.current_hp == hp0, "no emission on a corpse"
    assert u.statuses[0].remaining_ticks == 4, "no decrement on a corpse"
    assert events == []


def test_zero_magnitude_dot_emits_no_events_but_still_expires():
    u = _marine()
    apply_status(u, POISONED, magnitude=0.0, duration_ticks=2)
    events = []
    tick_statuses([u], events=events)
    tick_statuses([u], events=events)
    tick_statuses([u], events=events)   # the lazy sweep
    assert events == [] and u.statuses == []


# ---------------------------------------------------------------------------
# DoT/HoT through the pipeline — mitigation composes for free
# ---------------------------------------------------------------------------
def test_zombie_burning_ticks_at_4x_a_marines():
    """THE composition proof (mechanics/06 §4): BURNING emits HEAT packets
    through damage.apply_packet, so the zombie's resist_mult[HEAT] = 4.0
    mitigates it with ZERO status-side code — the same status burns a zombie
    at exactly 4x a marine, bit-for-bit (x4.0 is an exact binary scale)."""
    m, z = _marine(), _zombie()
    assert m.current_hp == z.current_hp == 100.0
    for u in (m, z):
        apply_status(u, BURNING, magnitude=6.25, duration_ticks=3)
    tick_statuses([m, z])
    marine_delta = 100.0 - m.current_hp
    zombie_delta = 100.0 - z.current_hp
    assert marine_delta == 6.25
    assert zombie_delta == 25.0
    assert zombie_delta == 4.0 * marine_delta   # exact, not approximate
    # ... and per tick, all the way down.
    tick_statuses([m, z])
    assert (100.0 - z.current_hp) == 4.0 * (100.0 - m.current_hp) == 50.0


def test_regen_heals_negative_direction_unresisted():
    """REGEN emits HEAL packets: negative-direction, mitigation = identity in
    v1 — a zombie regenerates exactly as fast as a marine."""
    m, z = _marine(), _zombie()
    m.current_hp = z.current_hp = 50.0
    for u in (m, z):
        apply_status(u, REGEN, magnitude=2.5, duration_ticks=4)
    tick_statuses([m, z])
    tick_statuses([m, z])
    assert m.current_hp == 55.0
    assert z.current_hp == 55.0, "HEAL is unresisted (zombie x4 is HEAT-only)"


def test_dot_kill_emits_kill_event_and_never_converts():
    """A burning death is a kill through the same life transition as every
    other packet — and NEVER zombifies (conversion is melee-kill-only)."""
    u = _marine()
    u.current_hp = 5.0
    apply_status(u, BURNING, magnitude=10.0, duration_ticks=2, source_id=42)
    events = []
    tick_statuses([u], events=events)
    assert not u.alive and u.life_state is LifeState.DEAD
    assert u.killed_by_zombie is False, "DoT deaths never convert"
    kills = [e for e in events if isinstance(e, UnitKilledEvent)]
    assert len(kills) == 1 and kills[0].killed_by == "burning"
    # The corpse freezes: the status stopped at its post-kill count.
    assert u.statuses[0].remaining_ticks == 1
    tick_statuses([u], events=events)
    assert u.statuses[0].remaining_ticks == 1 and len(events) == 2


def test_death_mid_list_stops_later_emissions_same_pass():
    """A unit killed by an earlier status in its own list receives no further
    emissions that pass (the alive gate every damage site honors)."""
    u = _marine()
    u.current_hp = 5.0
    apply_status(u, BURNING, magnitude=10.0, duration_ticks=3)
    apply_status(u, POISONED, magnitude=1.0, duration_ticks=3)
    events = []
    tick_statuses([u], events=events)
    hits = [e for e in events if isinstance(e, UnitHitEvent)]
    assert len(hits) == 1 and hits[0].source == "burning"
    assert u.current_hp == -5.0, "poison never landed"
    # Both statuses still decremented this (dying) pass — then freeze.
    assert [st.remaining_ticks for st in u.statuses] == [2, 2]


# ---------------------------------------------------------------------------
# composed_flags
# ---------------------------------------------------------------------------
def test_composed_flags_default_all_true_standing():
    u = _marine()
    f = composed_flags(u)
    assert f == ComposedFlags(True, True, True, False)
    # Bare stubs without the attribute compose to the same default.
    assert composed_flags(SimpleNamespace()) == f


def test_composed_flags_and_of_suppressions_or_of_prone():
    u = _marine()
    apply_status(u, IMMOBILIZED, magnitude=0.0, duration_ticks=3)
    f = composed_flags(u)
    assert (f.can_move, f.can_act, f.can_aim, f.is_prone) == \
        (False, True, True, False)
    apply_status(u, STUNNED, magnitude=0.0, duration_ticks=3)
    f = composed_flags(u)
    assert (f.can_move, f.can_act, f.can_aim, f.is_prone) == \
        (False, False, False, False), "IMMOBILIZED+STUNNED compose by AND"
    apply_status(u, KNOCKED_DOWN, magnitude=0.0, duration_ticks=3)
    assert composed_flags(u).is_prone is True, "prone composes by OR"


def test_composed_flags_restore_when_statuses_expire():
    u = _marine()
    apply_status(u, PARALYZED, magnitude=0.0, duration_ticks=1)
    assert composed_flags(u) == ComposedFlags(False, False, False, False)
    tick_statuses([u])          # its ONE suppressed tick (0-count, lazy)
    assert composed_flags(u) == ComposedFlags(False, False, False, False)
    tick_statuses([u])          # swept — released
    assert composed_flags(u) == ComposedFlags(True, True, True, False)


def test_dots_suppress_no_flags():
    u = _marine()
    for kind in (BURNING, POISONED, SUFFOCATING, REGEN):
        apply_status(u, kind, magnitude=0.5, duration_ticks=3)
    assert composed_flags(u) == ComposedFlags(True, True, True, False)


# ---------------------------------------------------------------------------
# Determinism + serialization
# ---------------------------------------------------------------------------
def test_serialize_statuses_canonical_ints_in_list_order():
    u = _marine()
    apply_status(u, BURNING, magnitude=1.5, duration_ticks=4, source_id=None)
    apply_status(u, POISONED, magnitude=0.25, duration_ticks=2, source_id=6)
    rec = serialize_statuses(u)
    assert rec == [
        [BURNING, 98304, 4, -1],       # None source -> -1 (int-typed record)
        [POISONED, 16384, 2, 6],
    ]
    assert all(isinstance(v, int) for row in rec for v in row)
    assert serialize_statuses(SimpleNamespace()) == []


# ---------------------------------------------------------------------------
# Flag consumption END-TO-END on a tiny sim (P3 stage 2 — the wired gates:
# movement / shooting / zombie AI read composed_flags through real step()s).
# No trigger applies statuses in-game yet (P4+), so tests apply directly and
# observe the suppression — proving the dead paths are wired correctly.
# ---------------------------------------------------------------------------
def _tiny_sim(seed=20260705):
    """A 16x16 hull-walled room, interior air, NO physics module (the unit-
    simulation section runs fully without it) — same synthetic-level idiom as
    field_ab_harness._scenario_level."""
    import numpy as np
    from level_loader import LevelData
    from simulation import Simulation

    h = w = 16
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:15, 1:15] = 4
    level = LevelData(name="status_e2e", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))
    sim = Simulation(level, seed=seed, breach_physics=None,
                     enable_recorder=False)
    return sim


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def test_e2e_can_move_suppression_pauses_marine_path_then_resumes():
    """can_move == False: the marine holds position and its precomputed path
    PAUSES (offset shift) — on release it resumes at the next un-walked
    index, not a catch-up teleport."""
    sim = _tiny_sim()
    u = Unit("M1", x=2, y=2, team=0)
    sim.add_unit(u)
    u.move_path = [(3.0, 2.0), (4.0, 2.0), (5.0, 2.0)]
    u.path_tick_offset = 0
    apply_status(u, KNOCKED_DOWN, magnitude=0.0, duration_ticks=2)

    _step(sim)                          # tick 0: suppressed tick #1
    assert (u.x, u.y) == (2.0, 2.0)
    _step(sim)                          # tick 1: 0-count lazy — tick #2
    assert (u.x, u.y) == (2.0, 2.0)
    assert u.path_tick_offset == 2, "the path paused, was not consumed"
    _step(sim)                          # tick 2: swept — resumes at path[0]
    assert (u.x, u.y) == (3.0, 2.0), "resumed at the FIRST un-walked index"
    _step(sim)
    assert (u.x, u.y) == (4.0, 2.0)


def test_e2e_control_marine_walks_without_statuses():
    """The dead-path guarantee's control: no statuses -> pre-P3 movement."""
    sim = _tiny_sim()
    u = Unit("M1", x=2, y=2, team=0)
    sim.add_unit(u)
    u.move_path = [(3.0, 2.0), (4.0, 2.0)]
    u.path_tick_offset = 0
    _step(sim)
    assert (u.x, u.y) == (3.0, 2.0)
    _step(sim)
    assert (u.x, u.y) == (4.0, 2.0)


def test_e2e_can_act_suppression_blocks_marine_fire_order():
    """can_act == False: the fire order is NOT executed while stunned; the
    order stays queued and executes once the stun releases."""
    from simulation.orders import ORDER_FIRE, Order

    def build():
        sim = _tiny_sim()
        m = Unit("M1", x=2, y=6, team=0)
        z = Unit("Z1", x=10, y=6, team=1)
        sim.add_unit(m)
        sim.add_unit(z)
        ok = sim.apply_action(m.id, Order(ORDER_FIRE, z.tile_x, z.tile_y,
                                          phase=0))
        assert ok, "fire order must validate (AP + inventory)"
        return sim, m

    # Control: the order fires on the very first tick.
    sim, m = build()
    _step(sim)
    assert m.last_fire_tick == 0, "control marine fires immediately"

    # Stunned: no attack while suppressed; fires on the release tick.
    sim, m = build()
    apply_status(m, STUNNED, magnitude=0.0, duration_ticks=2)
    _step(sim, 2)                       # ticks 0-1: both suppressed
    assert m.last_fire_tick == -999, "stunned marine executed no attack"
    _step(sim)                          # tick 2: swept — order still queued
    assert m.last_fire_tick == 2, "suppression delays, never cancels"


def test_e2e_can_act_suppression_blocks_zombie_melee():
    """can_act == False on the zombie: no bite while paralyzed; the marine
    takes the (config) melee damage only in the control run."""
    def build():
        sim = _tiny_sim()
        m = Unit("M1", x=5, y=5, team=0)
        z = Unit("Z1", x=7, y=5, team=1)    # centers 2 apart -> adjacent
        sim.add_unit(m)
        sim.add_unit(z)
        return sim, m, z

    sim, m, z = build()
    _step(sim)
    assert m.current_hp == 100.0 - CFG.zombie.melee_damage, \
        "control zombie bites on tick 0"

    sim, m, z = build()
    apply_status(z, PARALYZED, magnitude=0.0, duration_ticks=3)
    _step(sim, 3)
    assert m.current_hp == 100.0, "paralyzed zombie never bit"


def test_e2e_can_move_suppression_freezes_zombie_walk():
    """can_move == False on the zombie: it stands (stride clock paused)
    while the control zombie closes distance over the same ticks."""
    def build(immobilize):
        sim = _tiny_sim()
        m = Unit("M1", x=2, y=5, team=0)
        z = Unit("Z1", x=11, y=5, team=1)   # far, but LOS + trigger radius
        sim.add_unit(m)
        sim.add_unit(z)
        if immobilize:
            apply_status(z, IMMOBILIZED, magnitude=0.0, duration_ticks=25)
        return sim, z, (z.x, z.y)

    sim, z, start = build(immobilize=False)
    _step(sim, 20)
    assert (z.x, z.y) != start, \
        "control zombie must walk (validates pathfinding is live)"

    sim, z, start = build(immobilize=True)
    _step(sim, 20)
    assert (z.x, z.y) == start, "immobilized zombie never moved"
    assert z.zombie_activated, "perception is deliberately ungated"


def test_e2e_dot_drains_through_real_steps_and_events():
    """The tick slot is LIVE: a BURNING applied directly drains hp through
    real Simulation.step()s, and the hit events land in sim.tick_events
    (synced, in emission order) on each burning tick."""
    sim = _tiny_sim()
    u = Unit("M1", x=2, y=2, team=0)
    sim.add_unit(u)
    apply_status(u, BURNING, magnitude=1.5, duration_ticks=3, source_id=None)
    hp = [u.current_hp]
    burn_events = 0
    for _ in range(5):
        _step(sim)
        burn_events += sum(1 for e in sim.tick_events
                           if isinstance(e, UnitHitEvent)
                           and e.source == "burning")
        hp.append(u.current_hp)
    assert hp == [100.0, 98.5, 97.0, 95.5, 95.5, 95.5], \
        "exactly duration=3 emissions of 1.5, through real steps"
    assert burn_events == 3
    assert u.statuses == [], "expired + swept inside the loop"


# ---------------------------------------------------------------------------
# Digest inclusion (P3 stage 3): statuses are SYNCED state — the unit record
# carries them and the unit-state hash is sensitive to them.
# ---------------------------------------------------------------------------
def test_unit_record_carries_canonical_status_serialization():
    from field_ab_harness import _unit_record

    u = _marine()
    u.id = 0
    assert _unit_record(u)["statuses"] == []
    apply_status(u, BURNING, magnitude=1.5, duration_ticks=4, source_id=2)
    apply_status(u, KNOCKED_DOWN, magnitude=0.0, duration_ticks=2)
    rec = _unit_record(u)
    assert rec["statuses"] == serialize_statuses(u) == [
        [BURNING, 98304, 4, 2],
        [KNOCKED_DOWN, 0, 2, -1],
    ]


def test_unit_state_hash_sensitive_to_statuses():
    """A status changes the synced unit hash the moment it is applied, tracks
    its count-down, and — for a pure CC that touches nothing else — the hash
    RESTORES exactly once the status expires and sweeps (the record is back
    to statuses=[], all other fields untouched)."""
    from field_ab_harness import _capture_unit_state

    def h(u):
        return _capture_unit_state(
            SimpleNamespace(units=[u], tick_events=[]))["hash"]

    u = _marine()
    u.id = 0
    h0 = h(u)
    apply_status(u, IMMOBILIZED, magnitude=0.0, duration_ticks=1)
    h1 = h(u)
    assert h1 != h0, "applying a status must move the unit hash"
    tick_statuses([u])                  # 1 -> 0 (still present, lazy)
    h2 = h(u)
    assert h2 not in (h0, h1), "the count-down itself is hashed state"
    tick_statuses([u])                  # swept
    h3 = h(u)
    assert h3 == h0, "pure-CC expiry restores the record exactly"


def test_two_identical_runs_identical_status_lists_and_hp():
    """Determinism: the same applications on the same ticks produce identical
    serialized status lists AND hp traces, tick for tick."""
    def run():
        m, z = _marine(), _zombie()
        m.id, z.id = 0, 1
        units = [m, z]
        trace = []
        for t in range(12):
            if t == 1:
                apply_status(m, BURNING, magnitude=1.25, duration_ticks=5,
                             source_id=1)
                apply_status(z, KNOCKED_DOWN, magnitude=0.0, duration_ticks=3)
            if t == 4:
                apply_status(m, POISONED, magnitude=0.5, duration_ticks=4)
                apply_status(m, POISONED, magnitude=0.5, duration_ticks=2)
            if t == 7:
                apply_status(z, REGEN, magnitude=2.0, duration_ticks=3,
                             source_id=0)
            events = []
            tick_statuses(units, events=events)
            trace.append((
                [serialize_statuses(u) for u in units],
                [u.current_hp for u in units],
                [(type(e).__name__, e.unit_id, getattr(e, "damage", None))
                 for e in events],
            ))
        return trace

    assert run() == run()


if __name__ == "__main__":
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-q"]))
