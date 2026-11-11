"""Gameplay pack G2p -- reload and magazines (#183).

``combat.mag_gate`` / ``mag_spend`` / ``reload_begin`` / ``mag_settle`` and
``Simulation.reload_intent``, over ``[loadouts.*] magazines`` and
``[weapons.*] mag_size`` / ``reload_seconds``. Design:
``docs/gameplay_pack_v1_design_2026-11-10.md`` §3 G2p.

Properties (each test's docstring names the change that must break it):
ammunition is magazines, and a unit never fires more rounds than
``mag_size x magazines``; a manual reload discards the magazine in the gun
with every round in it; no shot leaves inside the stall and the stall lasts
exactly the weapon's reload ticks; a unit out of magazines with an empty gun
does nothing, for good; the verb does nothing when the magazine is full, when
a reload is running, and when no spare exists; a weapon with ``mag_size`` 0
and a loadout whose spares are untracked behave as before G2p; the verb goes
through the one tap and a recorded match with reloads replays through every
checkpoint; the HUD says rounds and spares and nothing turn-based; hand-
written bots never call the verb.

The sim here is a real ``ContinuousRealtime`` simulation in an open arena:
one marine holding its trigger along its facing (a shot is a
``ShotFiredEvent`` with ``launch`` set).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import match_replay as mr  # noqa: E402
from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.combat import (SPARES_UNTRACKED, mag_status,  # noqa: E402
                               mag_gate, mag_spend)
from simulation.events import ShotFiredEvent  # noqa: E402
from simulation.intents import FP_ONE  # noqa: E402
from simulation.loadouts import LoadoutTable  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables  # noqa: E402

SEED = 7
K5 = get_tables().weapons.by_name["k5_carbine"]
MAG = int(K5.mag_size)
RELOAD_TICKS = int(K5.reload_ticks)
ROF = int(K5.rof_interval_ticks)


# ---------------------------------------------------------------------------
# An open arena, one marine, the trigger held along +x
# ---------------------------------------------------------------------------
def _arena(h=16, w=40) -> LevelData:
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="g2p_arena", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


class _Range:
    """One marine of the shipped rifleman loadout in an arena, with the
    trigger held. ``step`` runs one tick and returns how many rounds left the
    barrel in it; ``shot_ticks`` collects the ``sim.tick`` of every shot."""

    def __init__(self, magazines=None):
        self.sim = Simulation(_arena(), seed=SEED, breach_physics=bp,
                              enable_recorder=False,
                              ruleset=ContinuousRealtime())
        self.sim.set_paused(False)
        self.uid = self.sim.add_unit(Unit("M", x=4, y=6, team=0))
        self.u = self.sim.get_unit(self.uid)
        if magazines is not None:
            self.u.magazines = magazines
        self.sim.set_aim(self.uid, FP_ONE, 0)
        self.shot_ticks: list = []

    def hold(self, held=True):
        self.sim.set_trigger(self.uid, held)

    def step(self) -> int:
        t = self.sim.tick
        self.sim.step()
        n = sum(1 for e in self.sim.tick_events
                if isinstance(e, ShotFiredEvent) and e.launch
                and e.unit_id == self.uid)
        self.shot_ticks += [t] * n
        return n

    def run(self, ticks: int) -> int:
        return sum(self.step() for _ in range(ticks))

    def run_until_shots(self, n: int, limit: int = 4000) -> None:
        """Step until ``n`` rounds have left in all (or ``limit`` ticks)."""
        for _ in range(limit):
            if len(self.shot_ticks) >= n:
                return
            self.step()
        raise AssertionError(f"only {len(self.shot_ticks)} of {n} rounds "
                             f"fired in {limit} ticks")

    def status(self):
        weapon = self.sim.weapons_tables.weapons.by_name[self.u.weapon_id]
        return mag_status(self.u, weapon, self.sim.tick)


# ---------------------------------------------------------------------------
# The shipped rows
# ---------------------------------------------------------------------------
def test_the_carbine_fires_twelve_rounds_a_second_one_round_a_pull():
    """Property (Erik's ruling, 2026-11-10): at 24 ticks a second the
    carbine's cadence quantizes to exactly 2 ticks and one pull is one round
    -- 12 rounds a second. Breaks if the row's seconds drift off the 2-tick
    quantization or the burst size is not 1 again."""
    tps = CFG.clock.ticks_per_second
    assert K5.shots_per_trigger == 1
    if tps == 24:
        assert ROF == 2
    r = _Range()
    r.hold()
    r.run(2 * ROF * 5)
    gaps = np.diff(r.shot_ticks)
    assert len(r.shot_ticks) >= 5 and set(gaps.tolist()) == {ROF}


def test_the_rifleman_carries_the_carbine_and_three_magazines():
    """Property: the shipped rifleman is a tracked-magazine carbine bearer
    with `magazines` counting the one in the gun (so two spares on the
    belt); a unit built from it binds exactly that. Breaks if the loadout
    row or the unit's bind loses the count."""
    r = _Range()
    total = int(r.u.magazines)
    assert total >= 2 and K5.mag_size > 0
    assert r.u.weapon_id == "k5_carbine"
    assert r.status() == (MAG, total - 1, False)       # read-only, unbound
    assert r.u.current_mag is None                      # reading bound nothing
    r.hold()
    r.step()
    assert r.u.current_mag == MAG - 1 and r.u.spare_mags == total - 1


# ---------------------------------------------------------------------------
# A. Ammunition is magazines
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("magazines", [1, 2, 3])
def test_a_unit_never_fires_more_rounds_than_its_magazines_hold(magazines):
    """Property: with the trigger held for far longer than the ammunition
    lasts, the rounds that leave are exactly `mag_size x magazines` (the
    cadence and the reload stalls are the only limits, and the last magazine
    ends the gun). Breaks if a reload ever refills past the spares, if an
    emptied last magazine is topped up, or if `magazines` stops being the
    TOTAL carried (an off-by-one on the one in the gun)."""
    r = _Range(magazines=magazines)
    r.hold()
    fired = r.run(MAG * magazines * ROF + magazines * RELOAD_TICKS + 400)
    assert fired <= MAG * magazines
    assert fired == MAG * magazines
    assert r.status()[0] == 0


def test_a_manual_reload_at_k_rounds_left_loses_exactly_those_k_rounds():
    """Property: a reload discards the magazine in the gun with every round
    still in it -- rounds are never moved to another magazine. Fire n rounds,
    reload by choice, fire everything that is left: the total is the
    magazines' capacity minus the k = MAG - n rounds thrown away. Breaks if
    the discarded magazine's rounds are carried over, or lost twice."""
    n = 11
    r = _Range(magazines=3)
    r.hold()
    r.run_until_shots(n)
    r.hold(False)
    r.step()
    assert len(r.shot_ticks) == n
    k = MAG - n
    assert r.status()[0] == k
    r.sim.reload_intent(r.uid)
    r.step()                       # consumed here
    assert r.u.current_mag == 0                # the old magazine is gone
    r.hold()
    total = r.run(RELOAD_TICKS + 3 * MAG * ROF + 400)
    assert n + total == 3 * MAG - k


def test_no_shot_leaves_inside_the_stall_and_it_lasts_exactly_reload_ticks():
    """Property: after a manual reload is consumed at tick T the first round
    leaves at T + reload_ticks, not one tick earlier or later, and none
    leaves in between -- with the trigger held throughout and the cadence
    long past. Breaks if the stall is shorter or longer than the row's
    reload ticks, or the gate lets a round through while it runs."""
    r = _Range(magazines=3)
    r.hold()
    r.run_until_shots(5)
    t_before = len(r.shot_ticks)
    r.sim.reload_intent(r.uid)
    t_consumed = r.sim.tick            # consumed by the step that starts here
    r.run(RELOAD_TICKS + 20)
    after = [t for t in r.shot_ticks[t_before:]]
    assert after, "the gun never resumed"
    assert min(after) == t_consumed + RELOAD_TICKS
    assert not [t for t in after if t < t_consumed + RELOAD_TICKS]
    assert r.status()[1] == r.u.magazines - 1 - 1     # one spare used


def test_emptying_the_magazine_reloads_by_itself_for_exactly_reload_ticks():
    """Property: the emptied magazine starts the same reload automatically
    when a spare exists -- the gap between the last round of a magazine and
    the next is exactly reload_ticks (the stall dominates the cadence),
    twice in a row. Breaks if the auto reload needs the verb, runs a
    different length, or is skipped."""
    r = _Range(magazines=3)
    r.hold()
    r.run_until_shots(2 * MAG + 1)
    s = r.shot_ticks
    assert s[MAG] - s[MAG - 1] == RELOAD_TICKS
    assert s[2 * MAG] - s[2 * MAG - 1] == RELOAD_TICKS
    inside = [b - a for a, b in zip(s[:MAG - 1], s[1:MAG])]
    assert set(inside) == {ROF}


def test_with_no_spare_and_an_empty_magazine_the_trigger_does_nothing_forever():
    """Property: a dry unit (empty gun, no spare) fires nothing and changes
    no state, however long the trigger is held, and the verb does nothing
    either. Breaks if a dry trigger refills, shoots a ghost round, raises,
    or lets the verb conjure a magazine."""
    r = _Range(magazines=1)
    r.hold()
    assert r.run(MAG * ROF + 100) == MAG
    frozen = (r.u.current_mag, r.u.spare_mags, r.u.reload_done_tick)
    assert frozen[0] == 0 and frozen[1] == 0
    r.sim.reload_intent(r.uid)
    assert r.run(600) == 0
    assert (r.u.current_mag, r.u.spare_mags, r.u.reload_done_tick) == frozen
    r.hold(False)
    r.hold()
    assert r.run(100) == 0


# ---------------------------------------------------------------------------
# B. The verb's no-ops
# ---------------------------------------------------------------------------
def test_the_verb_does_nothing_with_a_full_magazine():
    """Property: a reload with a full magazine in the gun changes nothing
    (no spare spent, no stall). Breaks if the full-magazine check is lost
    (a free spare burn)."""
    r = _Range(magazines=3)
    r.sim.reload_intent(r.uid)
    r.run(5)
    assert (r.u.current_mag, r.u.spare_mags, r.u.reload_done_tick) == \
        (MAG, r.u.magazines - 1, -1)
    # also after a stall has just completed (the gun is full again)
    r.hold()
    r.run_until_shots(2)
    r.hold(False)
    r.sim.reload_intent(r.uid)
    r.run(RELOAD_TICKS + 2)
    spares = r.u.spare_mags
    r.sim.reload_intent(r.uid)
    r.run(RELOAD_TICKS + 2)
    assert r.status() == (MAG, spares, False) and r.u.spare_mags == spares


def test_the_verb_does_nothing_while_a_reload_is_running():
    """Property: a second reload during a stall neither spends another spare
    nor restarts or extends the stall. Breaks if the verb stacks reloads
    (the stall would end later than reload_ticks after the first)."""
    r = _Range(magazines=3)
    r.hold()
    r.run_until_shots(4)
    r.hold(False)
    r.sim.reload_intent(r.uid)
    t0 = r.sim.tick
    r.step()
    spares = r.u.spare_mags
    done = r.u.reload_done_tick
    assert done == t0 + RELOAD_TICKS
    for _ in range(RELOAD_TICKS // 2):
        r.sim.reload_intent(r.uid)
        r.step()
    assert r.u.spare_mags == spares and r.u.reload_done_tick == done


def test_the_verb_does_nothing_when_no_spare_exists():
    """Property: with `magazines` 1 (the one in the gun, no spare) a
    reload keeps the half-empty magazine -- the verb never discards rounds it
    cannot replace. Breaks if a reload may empty the gun with nothing to
    load."""
    r = _Range(magazines=1)
    r.hold()
    r.run_until_shots(7)
    r.hold(False)
    r.step()
    left = r.u.current_mag
    r.sim.reload_intent(r.uid)
    r.run(RELOAD_TICKS + 5)
    assert r.u.current_mag == left == MAG - 7
    assert r.u.reload_done_tick == -1 and r.u.spare_mags == 0


def test_a_reload_does_not_stop_movement_and_is_never_interrupted():
    """Property (v1 rule): a unit keeps walking through its stall, and
    releasing or pulling the trigger and re-issuing aim does not cut the
    stall short. Breaks if the reload clears the move intent or if any
    other intent shortens it."""
    r = _Range(magazines=3)
    r.hold()
    r.run_until_shots(3)
    r.sim.reload_intent(r.uid)
    r.sim.set_move_dir(r.uid, FP_ONE, 0, 0)
    x0 = r.u.x
    t0 = r.sim.tick
    r.step()
    done = r.u.reload_done_tick
    for _ in range(RELOAD_TICKS - 2):
        r.hold(False)
        r.hold(True)
        r.sim.set_aim(r.uid, FP_ONE, 0)
        r.step()
    assert r.u.x > x0
    assert r.u.reload_done_tick == done == t0 + RELOAD_TICKS


# ---------------------------------------------------------------------------
# C. As before: untracked weapons and untracked spares
# ---------------------------------------------------------------------------
def test_a_weapon_with_mag_size_zero_never_touches_the_state_or_the_verb():
    """Property: a unit whose weapon row says `mag_size` 0 (lance_3, a row
    G2p did not change) keeps virgin magazine state through a trigger, a
    verb and a stall that never comes. Breaks if the reload law or the bind
    writes anything for an untracked weapon."""
    r = _Range(magazines=3)
    r.u.weapon_id = "lance_3"
    assert get_tables().weapons.by_name["lance_3"].mag_size == 0
    r.hold()
    r.sim.reload_intent(r.uid)
    r.run(200)
    assert r.u.current_mag is None
    assert r.u.spare_mags == SPARES_UNTRACKED
    assert r.u.reload_done_tick == -1
    assert mag_status(r.u, get_tables().weapons.by_name["lance_3"], 0) is None


def test_a_swapped_in_weapon_keeps_untracked_spares():
    """Property: the loadout's `magazines` belong to the loadout's own
    weapon. A tracked weapon the unit swaps to (the debug cycle key, a
    secondary) binds with untracked spares, so nothing but the weapon the
    loadout names is limited. Breaks if the rifleman's three magazines leak
    onto whatever else it holds (the p12_whisper would run dry after 36)."""
    other = "p12_whisper"
    row = get_tables().weapons.by_name[other]
    assert row.mag_size > 0
    r = _Range(magazines=3)
    r.u.weapon_id = other
    r.u.current_mag = None
    assert r.status() == (int(row.mag_size), None, False)
    r.hold()
    r.step()
    assert r.u.spare_mags == SPARES_UNTRACKED
    # and back to the loadout's own weapon: counted again
    r.u.weapon_id = "k5_carbine"
    r.u.current_mag = None
    r.run(6)
    assert r.u.spare_mags == 2


def test_untracked_spares_are_todays_infinite_auto_reload():
    """Property: a loadout with `magazines` 0 (or absent) reloads forever
    by the old law -- the stall after each emptied magazine, the same
    rounds, no spare ever counted and the unit never dry. The rounds' ticks
    are predicted here from first principles (cadence, magazine, stall), not
    from the code under test. Breaks if untracked spares are counted, or
    the law moves for a loadout nobody gave magazines."""
    r = _Range(magazines=0)
    r.hold()
    r.run(3000)
    expected = []
    t = r.shot_ticks[0]
    while len(expected) < len(r.shot_ticks):
        for i in range(MAG):
            expected.append(t + i * ROF)
        t = expected[-1] + RELOAD_TICKS
    assert r.shot_ticks == expected[:len(r.shot_ticks)]
    assert len(r.shot_ticks) > 3 * MAG
    assert r.u.spare_mags == SPARES_UNTRACKED


def test_a_manual_reload_with_untracked_spares_discards_and_stalls():
    """Property: with untracked spares the verb still discards the gun's
    magazine and stalls reload_ticks, spending nothing. Breaks if the verb
    needs a counted spare, or decrements the untracked sentinel."""
    r = _Range(magazines=0)
    r.hold()
    r.run_until_shots(6)
    r.hold(False)
    r.sim.reload_intent(r.uid)
    t0 = r.sim.tick
    r.step()
    assert r.u.spare_mags == SPARES_UNTRACKED
    assert r.u.reload_done_tick == t0 + RELOAD_TICKS
    r.run(RELOAD_TICKS)
    assert r.status()[0] == MAG


def test_the_gate_and_spend_stand_alone_for_a_tracked_weapon():
    """Property: `mag_gate` / `mag_spend` carry the whole law with no
    simulation: spending the last round of a magazine with a spare starts
    the stall, with none leaves the gate shut for good. Breaks if the law
    lives anywhere else than these functions (a bot, a test, the timeline
    would each need their own copy)."""
    u = SimpleNamespace(magazines=2, current_mag=None, spare_mags=-1,
                        reload_done_tick=-1)
    assert mag_gate(u, K5, 0) and u.current_mag == MAG and u.spare_mags == 1
    for t in range(MAG):
        assert mag_gate(u, K5, t)
        mag_spend(u, K5, t)
    assert u.spare_mags == 0 and u.reload_done_tick == MAG - 1 + RELOAD_TICKS
    assert not mag_gate(u, K5, MAG - 1 + RELOAD_TICKS - 1)
    assert mag_gate(u, K5, MAG - 1 + RELOAD_TICKS) and u.current_mag == MAG
    for t in range(MAG):
        mag_spend(u, K5, 1000 + t)
    assert u.current_mag == 0 and u.reload_done_tick == -1
    assert not mag_gate(u, K5, 10 ** 6)


# ---------------------------------------------------------------------------
# D. The loadout door
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [-1, 1.5, "3", True, None])
def test_the_loadout_door_refuses_a_bad_magazine_count_by_name(bad):
    """Property: `magazines` must be a non-negative integer, refused by the
    row's name; absent means 0 (untracked). Breaks if the door lets a
    negative, fractional, boolean or text count into a unit."""
    row = dict(weapon="k5_carbine", grenades=0, explosives=0, magazines=bad)
    with pytest.raises(ValueError, match=r"loadouts\.t\.magazines"):
        LoadoutTable({"t": row}, ["k5_carbine"])
    ok = LoadoutTable({"t": dict(weapon="k5_carbine", grenades=0,
                                 explosives=0)}, ["k5_carbine"])
    assert ok.get("t").magazines == 0


# ---------------------------------------------------------------------------
# E. The verb through the one tap, the action row and the direct control
# ---------------------------------------------------------------------------
def test_reload_is_a_recorded_kind_and_the_format_was_bumped():
    """Property: `reload_intent` is a row of INTENT_KINDS (one unit
    argument), the tap hands it to the attached log BEFORE the call acts, and
    a recording of the previous format is refused by its header. Breaks if
    the verb bypasses the tap, or the format stays at v2 with a changed
    carbine."""
    assert mr.INTENT_KINDS["reload_intent"] == ("int",)
    assert mr.FORMAT_VERSION >= 3
    r = _Range()
    seen = []
    r.sim.attach_intent_log(SimpleNamespace(
        record=lambda tick, kind, args: seen.append(
            (tick, kind, args, r.u.pending_reload if hasattr(
                r.u, "pending_reload") else False))))
    r.sim.reload_intent(r.uid)
    assert seen == [(0, "reload_intent", (r.uid,), False)]   # before it acted


def test_the_policy_row_carries_the_verb_as_a_latch_flag():
    """Property: the generated action row has a `reload_intent` flag column
    with a latch rule, so an agent reloads when it chooses -- once per
    decimation step. Breaks if the verb has no column or no rule."""
    import control_policy as cp
    assert "reload_intent" in cp.action_schema().names
    assert cp.VERB_RULES["reload_intent"] == ("latch", "flag")


def test_the_direct_control_reload_action_is_live_and_emits_the_verb():
    """Property: the `reload` action is no longer inert, and a device frame
    with `reload` set becomes exactly one reload_intent through the scheme's
    one emit path. Breaks if the key stays bound-but-dead or the pad/keyboard
    half reaches the facade another way."""
    import control_direct as cd
    assert "reload" not in cd.INERT_ACTIONS
    calls = []
    sim = SimpleNamespace(
        reload_intent=lambda uid: calls.append(("reload_intent", uid)),
        clear_move_dir=lambda uid: calls.append(("clear", uid)),
        set_trigger=lambda uid, held: calls.append(("trigger", uid, held)),
        # G7p: the scheme holds the flashlight bit every frame
        set_flashlight=lambda uid, off: None)
    dc = cd.DirectControl(keys=cd.key_table(CFG))
    dc._emit(sim, 7, cd.DeviceFrame(move=None, aim=None, trigger=False,
                                    throw=None, use=False, reload=True))
    assert ("reload_intent", 7) in calls
    calls.clear()
    dc._emit(sim, 7, cd.DeviceFrame(move=None, aim=None, trigger=False,
                                    throw=None, use=False))
    assert ("reload_intent", 7) not in calls


# ---------------------------------------------------------------------------
# F. Recorded matches with reloads replay; the bots never press the verb
# ---------------------------------------------------------------------------
def _reload_match(tmp_path, monkeypatch):
    import control_source
    import test_match_replay as tmr
    from control_source import ControlSource, quantize_stick_direction

    class _Reloader(ControlSource):
        """Aim at the nearest enemy, hold the trigger and reload by choice
        every 37 ticks (mid-magazine at the 12 rounds a second)."""

        def __init__(self, team):
            self.team = team

        def handle_frame(self, sim, renderer=None):
            foes = [u for u in sim.units if u.alive and u.team != self.team]
            for u in sim.units:
                if not (u.alive and u.team == self.team) or not foes:
                    continue
                f = min(foes, key=lambda f: ((f.x - u.x) ** 2
                                             + (f.y - u.y) ** 2, f.id))
                dx, dy, _ = quantize_stick_direction(f.x - u.x, f.y - u.y,
                                                     0.0)
                sim.set_aim(u.id, dx, dy)
                sim.set_trigger(u.id, True)
                if sim.tick % 37 == 20:
                    sim.reload_intent(u.id)

    monkeypatch.setitem(control_source._MATCH_CONTROLLERS, "g2p_reloader",
                        _Reloader)
    # The focus-fire law dormant (#183 G4p, k_expose = 0): with it live the
    # duel ends before anyone empties a magazine or reaches a choice reload,
    # and this match is about reloads, not about how fast marines die.
    from simulation import exposure
    monkeypatch.setattr(exposure, "_SETTINGS", exposure.get_settings())
    monkeypatch.setattr(CFG.damage.exposure, "k_expose", 0.0)
    t = tmr.MatchTeam(n=2, loadout="rifleman", env_profile="default",
                      controller="g2p_reloader")
    spec = replace(tmr.BOX_SPEC, teams=(t, t), zombies=0, time_limit_s=12.0)
    return tmr._box(tmp_path, "reloadbox", spec)


def test_a_recorded_match_with_reloads_replays_through_every_checkpoint(
        tmp_path, monkeypatch):
    """Property: a match whose units reload by choice, recorded, replays to
    every checkpoint and the original digests -- and the reloads are what the
    replay needs: the same recording with its reload intents dropped
    diverges. Breaks if the verb is not recorded (or not fed back in order),
    or if the reload law reads anything the recording does not carry."""
    lvl = _reload_match(tmp_path, monkeypatch)
    a = run_match(lvl, record=True, checkpoint_every_ticks=20)
    kinds = [k for v in a.recording.intents.values() for k, _ in v]
    assert "reload_intent" in kinds
    back = mr.decode_recording(mr.encode_recording(a.recording))
    o = run_match(lvl, replay=back)
    assert o.checkpoints_verified == len(a.recording.checkpoints)
    assert (o.result, o.field_digest, o.tick_digest) == \
        (a.result, a.field_digest, a.tick_digest)
    stripped = mr.decode_recording(mr.encode_recording(a.recording))
    for t in list(stripped.intents):
        stripped.intents[t] = [(k, args) for k, args in stripped.intents[t]
                               if k != "reload_intent"]
    with pytest.raises(mr.ReplayDivergence):
        run_match(lvl, replay=stripped)


def test_hand_written_bots_never_press_the_reload_verb(tmp_path):
    """Property (Erik's ruling): the bots rely on the automatic reload; a
    bots-vs-bots match records no reload_intent. Breaks if a bot starts
    issuing the verb (a change of the bots' behaviour belongs to G8p)."""
    import test_match_replay as tmr
    lvl = tmr._box(tmp_path, "botbox",
                   replace(tmr.BOX_SPEC, time_limit_s=10.0))
    a = run_match(lvl, record=True, checkpoint_every_ticks=60)
    kinds = {k for v in a.recording.intents.values() for k, _ in v}
    assert kinds and "reload_intent" not in kinds


# ---------------------------------------------------------------------------
# G. The HUD says rounds and spares, and nothing turn-based
# ---------------------------------------------------------------------------
def test_the_continuous_panel_shows_the_magazine_and_nothing_turn_based():
    """Property: under ContinuousRealtime the side panel's game-state lines
    carry the selected unit's rounds and spare magazines (and RELOADING / DRY
    when so) and none of the turn-based text (round, phase, AP, queued
    orders, planning phase); a turn-based ruleset still shows those. Breaks
    if the panel goes back to round / phase text under direct control, or
    drops the magazine."""
    from ui.model import ammo_text, panel_state_lines
    r = _Range(magazines=2)
    lines = [ln.text for ln in panel_state_lines(r.sim, r.u)]
    joined = "\n".join(lines)
    assert f"Ammo: {MAG}/{MAG}   spare 1" in joined
    for turn_based in ("Round", "Phase", "AP:", "Queued orders", "Planning"):
        assert turn_based not in joined
    r.hold()
    r.run(ROF * 3)
    assert ammo_text(r.u, r.sim.tick, r.sim.weapons_tables.weapons.by_name) \
        == f"Ammo: {MAG - 3}/{MAG}   spare 1"
    r.sim.reload_intent(r.uid)
    r.step()
    assert "RELOADING" in ammo_text(
        r.u, r.sim.tick, r.sim.weapons_tables.weapons.by_name)
    r.run(MAG * ROF + 3 * RELOAD_TICKS)
    assert ammo_text(r.u, r.sim.tick,
                     r.sim.weapons_tables.weapons.by_name).endswith("DRY")
    # a turn-based ruleset keeps its text
    from simulation.ruleset import TwoPhaseWEGO
    wego = Simulation(_arena(), seed=SEED, breach_physics=bp,
                      enable_recorder=False, ruleset=TwoPhaseWEGO())
    uid = wego.add_unit(Unit("W", x=4, y=6, team=0))
    text = "\n".join(ln.text for ln in panel_state_lines(
        wego, wego.get_unit(uid)))
    assert "Round" in text and "AP:" in text and "Ammo" not in text


# ---------------------------------------------------------------------------
# H. End to end: the real main.py in a hidden window, direct control
# ---------------------------------------------------------------------------
def _drive_probe(*args):
    """``tests/_e2e_mag_probe.py`` over ``tools/e2e_drive.py`` on door_test
    (the first [[spawn]] marine is possessed); returns the probe's log rows
    ``[tick, rounds, current_mag, spare_mags, reload_done_tick]``."""
    import json
    import os
    import subprocess
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(
        [sys.executable, "tests/_e2e_mag_probe.py", "--level", "door_test",
         "--control", "direct", "--mouse", "900,508@1", *args],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    line = next(ln for ln in r.stdout.splitlines()
                if ln.startswith("MAGPROBE "))
    return json.loads(line[len("MAGPROBE "):])


def test_e2e_holding_fire_empties_reloads_by_itself_and_goes_dry():
    """Property (the real window, the left mouse button held): the marine
    fires its magazine, the automatic reload stalls exactly reload_ticks
    with no round in it, fire resumes from a full magazine, and after
    `mag_size x magazines` rounds in all the trigger does nothing for the
    rest of the run. Breaks if the auto reload needs another input, the
    stall is another length, fire resumes early, or a dry gun fires."""
    log = _drive_probe("--frames", "720", "--hold", "MOUSE_LEFT@10-710")
    shots = [(t, n) for t, n, *_ in log if n]
    from simulation.loadouts import get_tables as loadout_tables
    carried = loadout_tables().loadout("rifleman").magazines
    assert carried >= 2
    assert sum(n for _, n in shots) <= MAG * carried
    assert sum(n for _, n in shots) == MAG * carried
    ticks = [t for t, n in shots for _ in range(n)]
    gaps = np.diff(ticks)
    stalls = sorted(set(int(g) for g in gaps if g > ROF))
    assert stalls == [RELOAD_TICKS]
    assert (gaps > ROF).sum() == carried - 1       # one reload per spare
    assert log[-1][2] == 0 and log[-1][3] == 0     # dry at the end


def test_e2e_the_reload_key_mid_magazine_discards_the_rest():
    """Property (the real window, Z pressed with the trigger held): the
    rounds still in the gun are thrown away, a spare is used, no round
    leaves for exactly reload_ticks from the tick the key was consumed, and
    the next magazine is full. Breaks if the key is not wired to the verb,
    keeps the old rounds, or lets a shot through the stall."""
    log = _drive_probe("--frames", "330", "--hold", "MOUSE_LEFT@10-320",
                       "--press", "Z@90")
    # the discard: current_mag drops to 0 with a round count still unspent
    i = next(i for i, (t, n, mag, sp, done) in enumerate(log)
             if mag == 0 and done >= 0)
    t0, n0, mag0, spare0, done0 = log[i]
    before = log[i - 1]
    assert 0 < before[2] < MAG and n0 == 0         # discarded mid-magazine
    assert before[3] - spare0 == 1                  # exactly one spare used
    assert done0 == t0 + RELOAD_TICKS
    after = [row for row in log[i + 1:] if row[1]]
    assert after[0][0] == done0 and after[0][2] == MAG - 1
