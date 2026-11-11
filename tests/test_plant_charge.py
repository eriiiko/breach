"""#193 -- marines plant ONE timed C4 charge on the continuous path.

``[loadouts.rifleman] explosives = 1`` + ``charge = "demo_c4"``, the verb
``Simulation.plant_charge_intent(unit_id, fuse_ticks)`` (a latch: clamped to
``[weapons.c4_satchel] fuse_min_seconds .. fuse_max_seconds``, a plant of
``plant_seconds`` interrupted by death or a stun), the conductor's slot-9e
``charges.tick`` (the continuous path's ONE driver; OnePhaseWEGO's timeline
keeps its own), the absence-transparent digest of planted charges and of a
plant under way, the replay kind (format v7), the policy's ``nonzero`` latch,
and the direct control's timer.

Every test's docstring names the property it protects and the change that
must break it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import charges as C  # noqa: E402
from simulation.events import (PlantEvent, UnitHitEvent,  # noqa: E402
                               UnitKilledEvent)
from simulation.loadouts import LoadoutTable, get_tables as loadouts  # noqa: E402
from simulation.ruleset import ContinuousRealtime, OnePhaseWEGO  # noqa: E402
from simulation.status import STUNNED, apply_status  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import (PLANT_WEAPON, fuse_ticks_from_seconds,  # noqa: E402
                                get_tables)

SEED = 11
TPS = int(CFG.clock.ticks_per_second)
ROW = get_tables().weapons.by_name[PLANT_WEAPON]
PLANT = int(ROW.plant_ticks)
LO = fuse_ticks_from_seconds(ROW.fuse_min_seconds, TPS)
HI = fuse_ticks_from_seconds(ROW.fuse_max_seconds, TPS)


def _arena(h=30, w=60) -> LevelData:
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="c4_arena", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim(ruleset=None, physics=True) -> Simulation:
    sim = Simulation(_arena(), seed=SEED,
                     breach_physics=bp if physics else None,
                     enable_recorder=False,
                     ruleset=ruleset or ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _marine(sim, cx, cy, name="P", team=0):
    """A rifleman whose CENTRE tile is (cx, cy) (a footprint-3 block)."""
    uid = sim.add_unit(Unit(name, x=cx - 1, y=cy - 1, team=team))
    return sim.get_unit(uid)


def _run(sim, n):
    events = []
    for _ in range(n):
        t = sim.tick
        sim.step()
        sim.set_paused(False)
        events += [(t, e) for e in sim.tick_events]
    return events


# ---------------------------------------------------------------------------
# A. the loadout and the rows
# ---------------------------------------------------------------------------
def test_the_rifleman_carries_one_c4_and_the_plant_row_is_five_to_sixty_s():
    """Property: the shipped rifleman carries exactly one charge, of the C4
    row, and the plant row's timer range is Erik's 5 .. 60 s. Breaks if the
    loadout count or charge row moves, or the fuse range is edited."""
    row = loadouts().loadout("rifleman")
    assert (row.explosives, row.charge) == (1, "demo_c4")
    assert get_tables().ammo.by_name["demo_c4"].payload == "demolition_c4"
    assert (LO, HI) == (5 * TPS, 60 * TPS)
    sim = _sim(physics=False)
    u = _marine(sim, 10, 10)
    assert (u.has_explosive, u.charge_ammo) == (1, "demo_c4")


def test_the_loadout_door_refuses_a_count_without_a_charge_and_a_bad_row():
    """Property: a loadout that carries explosives names WHICH charge, a
    demo-charge [ammo.*] row, or is refused by name. Breaks if the door lets
    a count without a charge through, or accepts a non-charge row."""
    rows = get_tables().charge_rows
    base = dict(weapon="k5_carbine", grenades=0)
    LoadoutTable({"ok": dict(base, explosives=1, charge="demo_c4")},
                 ["k5_carbine"], charge_rows=rows)
    with pytest.raises(ValueError, match="names no charge"):
        LoadoutTable({"x": dict(base, explosives=1)}, ["k5_carbine"],
                     charge_rows=rows)
    with pytest.raises(ValueError, match="demo-charge"):
        LoadoutTable({"x": dict(base, explosives=1, charge="grenade_frag")},
                     ["k5_carbine"], charge_rows=rows)


# ---------------------------------------------------------------------------
# B. the verb
# ---------------------------------------------------------------------------
def test_the_fuse_is_clamped_to_the_rows_range_and_the_tap_records_it():
    """Property: the verb clamps its fuse into [5 s, 60 s] in ticks and the
    intent tap records the CLAMPED value (so a replay plants the same
    charge); a float fuse is refused. Breaks if the clamp is dropped, moved
    after the tap, or the type check removed."""
    sim = _sim(physics=False)
    u = _marine(sim, 10, 10)
    seen = []

    class _Log:
        def record(self, tick, kind, args):
            seen.append((kind, args))
    sim.attach_intent_log(_Log())
    sim.plant_charge_intent(u.id, 1)
    assert u.pending_plant == LO and seen[-1] == ("plant_charge_intent",
                                                  (u.id, LO))
    sim.plant_charge_intent(u.id, 10 ** 6)
    assert u.pending_plant == HI and seen[-1][1] == (u.id, HI)
    sim.plant_charge_intent(u.id, 7 * TPS)
    assert u.pending_plant == 7 * TPS
    with pytest.raises(TypeError):
        sim.plant_charge_intent(u.id, 5.0)


def test_the_charge_fires_exactly_at_completion_plus_fuse_at_the_feet():
    """Property: a plant consumed at tick t0 lies down at t0 + plant ticks
    on the planter's centre tile as it stands then, owned by the planter,
    and detonates on exactly the tick completion + fuse (not one before);
    one PlantEvent at t0. Breaks if the fuse counts from the verb, the
    charge is placed at the verb, the firing slot is skipped or doubled."""
    sim = _sim()
    u = _marine(sim, 20, 15)
    fuse = 6 * TPS
    t0 = sim.tick
    sim.plant_charge_intent(u.id, fuse)
    ev = _run(sim, PLANT)
    assert [(t, e.unit_id, e.ticks) for t, e in ev
            if isinstance(e, PlantEvent)] == [(t0, u.id, PLANT)]
    assert sim.planted_charges == [] and u.has_explosive == 1
    _run(sim, 1)                                     # tick t0 + PLANT
    (c,) = sim.planted_charges
    assert (c.x, c.y, c.owner_id, c.ammo_name) == (20, 15, u.id, "demo_c4")
    assert c.det_tick == t0 + PLANT + fuse
    assert u.has_explosive == 0
    # walk away; the charge stays where it was laid
    while sim.tick < c.det_tick:
        _run(sim, 1)
        if sim.tick < c.det_tick:
            assert c.live, "fired early"
    assert sim.tick == c.det_tick and c.live
    _run(sim, 1)                                     # the det tick itself
    assert not c.live and sim.planted_charges == []


def test_an_interrupted_plant_plants_nothing_and_keeps_the_count():
    """Property: a unit stunned or killed during the plant plants nothing,
    keeps its charge, and (stunned) can plant again afterwards. Breaks if
    the charge appears anyway, the count is spent at the verb, or the
    dropped plant blocks later plants."""
    for how in ("stun", "kill"):
        sim = _sim(physics=False)
        u = _marine(sim, 10, 10)
        sim.plant_charge_intent(u.id, LO)
        _run(sim, 1)
        assert u.plant_done_tick >= 0
        if how == "stun":
            apply_status(u, STUNNED, magnitude=0, duration_ticks=PLANT + 2)
        else:
            u.alive = False
        _run(sim, PLANT + 3)
        assert sim.planted_charges == [], how
        assert u.has_explosive == 1, how
        assert u.plant_done_tick == -1, how
    sim = _sim(physics=False)
    u = _marine(sim, 10, 10)
    apply_status(u, STUNNED, magnitude=0, duration_ticks=3)
    sim.plant_charge_intent(u.id, LO)
    assert not [e for _, e in _run(sim, 2) if isinstance(e, PlantEvent)]
    _run(sim, 4)
    sim.plant_charge_intent(u.id, LO)
    _run(sim, PLANT + 1)
    assert len(sim.planted_charges) == 1 and u.has_explosive == 0


def test_the_count_is_spent_once_and_a_unit_with_none_plants_nothing():
    """Property: one charge plants once; a second verb during the plant and
    every verb after the count is spent are ignored with no event; a throw
    cannot start during a plant. Breaks if the count goes negative, a second
    charge appears, or the hands-busy gate is removed."""
    sim = _sim(physics=False)
    u = _marine(sim, 10, 10)
    sim.plant_charge_intent(u.id, LO)
    _run(sim, 1)
    sim.plant_charge_intent(u.id, LO)                 # during the plant
    sim.throw_grenade_intent(u.id, 65536, 0, 24, kind=0)
    ev = _run(sim, PLANT + 1)
    assert not sim.projectiles and u.throw_release_tick == -1
    assert len(sim.planted_charges) == 1 and u.has_explosive == 0
    sim.plant_charge_intent(u.id, LO)                 # none left
    ev = _run(sim, PLANT + 2)
    assert not [e for _, e in ev if isinstance(e, PlantEvent)]
    assert len(sim.planted_charges) == 1 and u.has_explosive == 0


def test_the_blast_is_credited_to_the_planter_and_hurts_the_planter_too():
    """Property: the detonation's unit damage is credited to the planter
    (attacker_id), to an enemy and to a teammate alike, and the planter
    standing on its own charge is hurt by it (no special case). Breaks if
    the owner is not handed to the payload executor, or the planter is
    exempted."""
    sim = _sim()
    p = _marine(sim, 20, 15, "P", team=0)
    mate = _marine(sim, 24, 15, "M", team=0)
    foe = _marine(sim, 20, 19, "F", team=1)
    sim.plant_charge_intent(p.id, LO)
    _run(sim, PLANT + 1)
    det = sim.planted_charges[0].det_tick
    ev = _run(sim, det - sim.tick + 1)
    hits = [e for t, e in ev if t == det
            and isinstance(e, (UnitHitEvent, UnitKilledEvent))]
    hit_ids = {e.unit_id for e in hits}
    assert {p.id, mate.id, foe.id} <= hit_ids
    assert all(e.attacker_id == p.id for e in hits)


# ---------------------------------------------------------------------------
# C. one driver per path
# ---------------------------------------------------------------------------
def _count_ticks(monkeypatch):
    calls = []
    real = C.tick

    def counting(sim):
        calls.append(sim.tick)
        return real(sim)
    monkeypatch.setattr(C, "tick", counting)
    return calls


def test_continuous_ticks_a_charge_once_per_tick(monkeypatch):
    """Property: on the continuous path the conductor calls charges.tick
    exactly once per sim tick while a charge is live (and not at all with
    none). Breaks if the slot-9e call is removed, duplicated, or runs on
    every tick of a charge-free match."""
    calls = _count_ticks(monkeypatch)
    sim = _sim(physics=False)
    u = _marine(sim, 10, 10)
    _run(sim, 3)
    assert calls == []
    sim.plant_charge_intent(u.id, LO)
    _run(sim, PLANT + 10)
    live_from = sim.planted_charges[0].det_tick - LO
    assert calls == list(range(live_from, sim.tick))


def test_onephase_ticks_a_charge_once_per_tick(monkeypatch):
    """Property: under OnePhaseWEGO (a ruleset that drives the unit slots)
    the timeline is the only caller of charges.tick -- once per tick, never
    also the conductor -- even for a charge the continuous verb planted.
    Breaks if the conductor's ``drives_units`` guard is dropped."""
    calls = _count_ticks(monkeypatch)
    sim = _sim(ruleset=OnePhaseWEGO(), physics=False)
    u = _marine(sim, 10, 10)
    sim.planted_charges.append(C.PlantedCharge(10, 10, "demo_c4", u.id,
                                               det_tick=sim.tick + 50))
    t0 = sim.tick
    _run(sim, 10)
    assert calls == list(range(t0, t0 + 10))


# ---------------------------------------------------------------------------
# D. digest, replay, policy
# ---------------------------------------------------------------------------
def test_charges_and_a_plant_are_hashed_only_while_they_exist():
    """Property: the unit-state digest carries a ``charges`` key only while a
    charge is live and a unit's ``plant`` key only while it plants, so a
    match without a plant hashes exactly as before #193, and a live charge
    moves the hash. Breaks if either is hashed unconditionally (every pinned
    digest would move) or not at all (a charge desync would go unseen)."""
    from field_ab_harness import _capture_unit_state, _unit_record
    sim = _sim(physics=False)
    u = _marine(sim, 10, 10)
    assert "plant" not in _unit_record(u)
    before = _capture_unit_state(sim)
    sim.plant_charge_intent(u.id, LO)
    _run(sim, 1)
    assert _unit_record(u)["plant"] == [u.plant_done_tick, LO]
    _run(sim, PLANT)
    assert "plant" not in _unit_record(u)
    snap = _capture_unit_state(sim)
    c = sim.planted_charges[0]
    c.det_tick += 1
    assert _capture_unit_state(sim)["hash"] != snap["hash"]
    c.live = False
    sim.planted_charges.clear()
    u.has_explosive = 1                  # (not hashed)
    sim.tick_events.clear()
    assert set(_capture_unit_state(sim)) == set(before)


def test_a_recorded_match_with_a_plant_replays_every_checkpoint(
        monkeypatch, tmp_path):
    """Property: a recorded match in which units plant (and a charge goes
    off) carries the verb in the recording at the clamped fuse and replays
    through every checkpoint to the original digests. Breaks if the kind is
    not in INTENT_KINDS, the fuse is recorded unclamped, or the conductor's
    firing is not deterministic."""
    import dataclasses
    import control_source
    import match_replay as mr
    from control_source import ControlSource
    from run_match import run_match
    from simulation.match import MatchTeam
    from test_match_replay import BOX_SPEC, _box, _copy

    class _Planter(ControlSource):
        def __init__(self, team):
            self.team = team

        def handle_frame(self, sim, renderer=None):
            for u in sim.units:
                if u.alive and u.team == self.team and sim.tick == 4:
                    sim.plant_charge_intent(u.id, 1)     # clamped to 5 s
    monkeypatch.setitem(control_source._MATCH_CONTROLLERS, "test_planter",
                        _Planter)
    t = MatchTeam(n=2, loadout="rifleman", env_profile="default",
                  controller="test_planter")
    lvl = _box(tmp_path, "c4plant",
               dataclasses.replace(BOX_SPEC, teams=(t, t), zombies=0))
    a = run_match(lvl, record=True, checkpoint_every_ticks=5)
    plants = [args for v in a.recording.intents.values()
              for k, args in v if k == "plant_charge_intent"]
    assert plants and all(args[1] == LO for args in plants)
    assert a.result.ticks >= 4 + PLANT + LO, "the charges never went off"
    b = run_match(lvl, replay=_copy(a.recording))
    assert b.checkpoints_verified == len(a.recording.checkpoints)
    assert (b.result, b.field_digest, b.tick_digest) == \
        (a.result, a.field_digest, a.tick_digest)
    assert mr.FORMAT_VERSION == 7


def test_the_policy_zero_row_plants_nothing_and_a_fuse_plants_once():
    """Property: a policy's zero row never plants; a row whose
    ``plant_charge_intent.fuse_ticks`` is non-zero issues the verb on the
    first tick of its decimation step only, at that fuse. Breaks if the
    plant's gate becomes ``always`` (a zero row would plant at 5 s) or the
    class becomes ``hold``."""
    from control_policy import AgentPolicy, action_schema, zero_rows
    issued = []
    sim = _sim(physics=False)
    u = _marine(sim, 10, 10)
    real = sim.plant_charge_intent
    sim.plant_charge_intent = lambda uid, f: (issued.append((uid, f)),
                                              real(uid, f))
    pol = AgentPolicy(0)
    pol.set_rows(zero_rows(1))
    for _ in range(3):
        pol.handle_frame(sim)
        _run(sim, 1)
    assert issued == []
    rows = zero_rows(1)
    rows[0, action_schema().index("plant_charge_intent.fuse_ticks")] = 8 * TPS
    pol.set_rows(rows)
    for _ in range(3):
        pol.handle_frame(sim)
        _run(sim, 1)
    assert issued == [(u.id, 8 * TPS)]


# ---------------------------------------------------------------------------
# E. direct control: the timer, the keys, the HUD
# ---------------------------------------------------------------------------
def test_the_timer_steps_five_to_sixty_and_wraps():
    """Property: the plant timer steps 5 s at a time through 5 .. 60 and
    wraps to 5; every value is inside the verb's clamp. Breaks if the step
    list or the wrap changes."""
    from control_direct import (CHARGE_TIMER_DEFAULT_S, CHARGE_TIMER_STEPS_S,
                                next_charge_timer)
    assert CHARGE_TIMER_STEPS_S == tuple(range(5, 61, 5))
    assert CHARGE_TIMER_DEFAULT_S == 10
    seen = [5]
    for _ in range(12):
        seen.append(next_charge_timer(seen[-1]))
    assert seen == [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 5]


def test_the_direct_control_plants_at_the_selected_timer(monkeypatch):
    """Property: the direct control's frame with ``cycle_charge_timer``
    steps its timer and one with ``plant_charge`` issues the verb at that
    timer in ticks; both keys are bound through the door and not inert.
    Breaks if _emit drops either field or converts the timer elsewhere."""
    from control_direct import (DeviceFrame, DirectControl, INERT_ACTIONS,
                                KEY_ACTIONS, key_table)
    keys = key_table(CFG)
    assert {"plant_charge", "cycle_charge_timer"} <= set(KEY_ACTIONS)
    assert keys["plant_charge"].name == "X"
    assert keys["cycle_charge_timer"].name == "5"
    assert not set(INERT_ACTIONS) & {"plant_charge", "cycle_charge_timer"}
    dc = DirectControl(keys)
    sim = _sim(physics=False)
    u = _marine(sim, 10, 10)
    issued = []
    monkeypatch.setattr(sim, "plant_charge_intent",
                        lambda uid, f: issued.append((uid, f)))
    frame = dict(move=None, aim=None, trigger=False, throw=None, use=False)
    dc._emit(sim, u.id, DeviceFrame(**frame, cycle_charge_timer=True))
    assert dc.charge_timer_s == 15 and issued == []
    dc._emit(sim, u.id, DeviceFrame(**frame, plant_charge=True))
    assert issued == [(u.id, 15 * TPS)]


def test_the_hud_line_shows_the_count_and_the_timer():
    """Property: the side panel names the charges carried and, under direct
    control, the selected timer. Breaks if the line drops either."""
    from ui.model import charge_text, panel_state_lines
    sim = _sim(physics=False)
    u = _marine(sim, 10, 10)
    texts = [ln.text for ln in panel_state_lines(sim, u, charge_timer_s=25)]
    assert "C4 (X): 1   timer 25 s (5)" in texts
    assert charge_text(u) == "Charges: 1"
