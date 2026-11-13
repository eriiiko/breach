"""#184 -- the SOMA marine in play: locomotion from the drawn speed
(``renderer/pose_locomotion.py``), directional hits, the reload / throw
one-shots, and the carbine on a hand socket (``renderer/held_props.py``).
Properties only; every test names the property it protects and the change
that must break it.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_pose_locomotion.py -q
"""
from __future__ import annotations

import dataclasses
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from renderer import pose_layer as pl_mod  # noqa: E402
from renderer.pose_layer import (NO_CLIP, PoseConfigError, PoseLayer,  # noqa: E402
                                 TimedShotSettings, forward_kinematics, pose_settings)
from renderer.pose_locomotion import measure_gait  # noqa: E402
from renderer.unit_looks import all_look_names, look_path, looks_table  # noqa: E402

from _pose_looks import quaternius_looks  # noqa: E402

MARINE = "soma_space_marine"
SETTINGS = pose_settings(CFG)
TPS = 24
DT = 1.0 / 60.0
# drawn speeds (model m/s) the tests sweep: the shipped walk / full speed
# (1.333 / 3 tiles/s) and the gameplay pack's (#183: walk 1.4, full 4.5 m/s,
# sideways 0.8 x and backward 0.6 x the walk), at 0.333 m sim tiles and the
# renderer's 0.3142 model m per drawn tile
_M_PER_TILE = 1.885 / 6.0
SPEEDS = sorted({round(v, 3) for v in (
    1.333 * _M_PER_TILE, 3.0 * _M_PER_TILE,
    *(v / 0.333 * _M_PER_TILE for v in (1.4, 4.5, 0.8 * 1.4, 0.6 * 1.4)))})


@pytest.fixture(scope="module")
def layer() -> PoseLayer:
    pl = PoseLayer(SETTINGS)
    for name in quaternius_looks() + [MARINE]:
        pl.add_look(name, look_path(name))
    return pl


# per PoseState (by id; the tests keep their states alive): its wall clock and
# the time of its next sim tick, so successive _drive calls tick at 24 tps
_CLOCKS = {}


def _drive(pl, ps, speed, rel_deg, seconds, look=MARINE, role="player", record=None,
           fps=60.0):
    """Move a unit at ``speed`` (model m/s) along ``rel_deg`` (relative to
    its facing) for ``seconds``, the position changing once per SIM tick as
    in the game (most frames see no delta). Returns the states."""
    dt = 1.0 / fps
    clock = _CLOCKS.setdefault(id(ps), [0.0, 0.0])    # the unit's (wall, next tick) clock
    states = []
    for _ in range(int(round(seconds * fps))):
        clock[0] += dt
        moved = 0.0
        while clock[1] <= clock[0]:
            moved += speed / TPS
            clock[1] += 1.0 / TPS
        travel = math.radians(rel_deg) if moved > 0.0 else None
        st = pl.locomote(ps, "walk", moved, travel, True, dt, role, 1.0 / TPS)
        pl.drive(ps, st, dt, role)
        states.append(st)
        if record is not None:
            record.append(pl.compose(look, [ps]).global_trs[0].astype(np.float64))
    return states


def _feet(pl, look=MARINE):
    sk = pl.rig(look).bank.skeleton
    return [sk.index(f) for f in pl.settings.playback["soma"].locomotion.feet]


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------

def test_every_gait_state_of_the_marine_strides_and_walks_its_sector(layer):
    """PROPERTY: every gait and sector state's legs clip on the shipped
    marine is a measured gait clip (it strides), and each sector's clip
    travels within 25 deg of the sector's centre -- so the strafe clips are
    on the side their sector names (Rifle_Aim_Left walks to the unit's left).

    BREAKS IF: a sector row names the wrong clip (Left / Right swapped, the
    back clip in a front sector), or a clip is re-exported travelling
    another way, or the measurement's travel sign flips.
    """
    rig = layer.rig(MARINE)
    loco = rig.locomotion
    role = loco.roles["player"]
    assert role.gaits and role.gaits[0].sectors
    for g in role.gaits:
        assert g.legs_clip in loco.measures
        for s in g.sectors:
            m = loco.measures[s.legs_clip]
            off = abs((m.travel - s.center + math.pi) % (2 * math.pi) - math.pi)
            assert math.degrees(off) < 25.0, (s.state, math.degrees(m.travel))


# ---------------------------------------------------------------------------
# the feet do not skate
# ---------------------------------------------------------------------------

def _skate(pl, speed, rel_deg):
    """The planted foot's speed against the GROUND, as a share of the unit's
    speed: the composed pose's lower foot (when within 3 cm of the lowest a
    foot gets) moving in the model frame, plus the body's own travel."""
    rec = []
    ps = pl.new_state(MARINE, "idle", "player")
    _drive(pl, ps, speed, rel_deg, 3.5, record=rec)
    p = np.stack(rec)[90:][:, _feet(pl), 0:3]
    a = math.radians(rel_deg)
    v = np.gradient(p, DT, axis=0) + speed * np.array([math.sin(a), 0.0, math.cos(a)])
    low = np.argmin(p[:, :, 1], axis=1)
    idx = np.arange(len(p))
    planted = p[idx, low, 1] < p[:, :, 1].min() + 0.03
    vl = v[idx, low][planted][:, [0, 2]]
    return float(np.median(np.linalg.norm(vl, axis=1))) / speed, ps


@pytest.mark.parametrize("rel_deg", [0.0, 45.0, 90.0, 135.0, 180.0, -90.0])
@pytest.mark.parametrize("speed", SPEEDS)
def test_the_planted_foot_holds_the_ground_at_any_speed_and_direction(layer, speed, rel_deg):
    """PROPERTY: at every speed the shipped and the #183 sim moves a marine
    (drawn speeds 0.42 .. 4.25 m/s) and in every direction against its aim,
    the planted foot of the composed pose moves against the ground at under
    12 % of the unit's speed on a walk gait and under 20 % on a run (whose
    flight phase blurs the planted test) -- the rate is derived, the gait
    and the sector fit, and the twist covers the rest.

    BREAKS IF: a gait clip's rate becomes a literal (or the stride
    measurement drifts), the residual twist turns the wrong way or about
    the swaying hips (it slid the feet 25 % at a 58 deg residual), the drawn
    speed is mis-estimated, or a sector plays the wrong clip.
    """
    ratio, ps = _skate(layer, speed, rel_deg)
    running = ps.loco_state in ("run", "sprint")
    assert ratio < (0.20 if running else 0.12), (speed, rel_deg, ps.loco_state, ratio)


def test_the_frames_between_sim_ticks_never_read_as_stopping(layer):
    """PROPERTY: a unit moving at a steady speed -- its position changing
    only on the frames a sim tick lands, 2 or 3 frames apart at 60 fps --
    stays on its gait every frame (never the idle state), its drawn speed
    averages the true one within 3 %; once its position stops changing it
    is idle within ``stop_after_s`` plus the crossfade.

    BREAKS IF: a frame without a delta is taken as a stop (the legs would
    flicker between idle and walk), or the speed estimate is biased by the
    uneven frames per tick.
    """
    ps = layer.new_state(MARINE, "idle", "player")
    speed = 1.4 / 0.333 * _M_PER_TILE
    _drive(layer, ps, speed, 0.0, 1.0)
    est = []
    t = nt = 0.0
    for _ in range(240):
        t += DT
        moved = 0.0
        while nt <= t:
            moved += speed / TPS
            nt += 1.0 / TPS
        st = layer.locomote(ps, "walk", moved, 0.0 if moved else None, True, DT, "player",
                            1.0 / TPS)
        layer.drive(ps, st, DT, "player")
        assert st == "walk", st
        est.append(ps.speed)
    assert abs(np.mean(est) / speed - 1.0) < 0.03
    stop = layer.settings.playback["soma"].locomotion.roles["player"].stop_after_s
    states = []
    for _ in range(int((stop + 0.1) / DT)):
        st = layer.locomote(ps, "walk", 0.0, None, True, DT, "player", 1.0 / TPS)
        layer.drive(ps, st, DT, "player")
        states.append(st)
    assert states[-1] == "idle" and ps.speed == 0.0


# ---------------------------------------------------------------------------
# the gait choice
# ---------------------------------------------------------------------------

def test_walk_run_sprint_follow_the_clip_rates_with_hysteresis(layer):
    """PROPERTY: ramping the drawn speed up and back down, the gait steps
    walk -> run -> sprint and back, once each way; each step up happens
    where the slower gait's clip would play faster than its ``max_rate`` x
    (1 + hysteresis), each step down where it would play under its
    ``max_rate`` x (1 - hysteresis) -- choices made on the CLIPS' rates, so
    they move with a re-exported clip, never with a speed in m/s; and a
    speed held at a threshold with 3 % jitter does not switch back and forth.

    BREAKS IF: the gait is chosen on a fixed speed, the hysteresis is lost
    (the legs would flicker between walk and run at the boundary), or a gait
    is skipped.
    """
    rig = layer.rig(MARINE)
    loco = rig.locomotion
    role = loco.roles["player"]
    h = role.hysteresis
    fwd = next(s for s in role.gaits[0].sectors if s.center == 0.0)
    walk_stride = loco.measures[fwd.legs_clip].stride
    run_stride = loco.measures[role.gaits[1].legs_clip].stride
    ps = layer.new_state(MARINE, "idle", "player")
    _drive(layer, ps, 0.5, 0.0, 1.0)
    seq, at = [], {}
    for v in list(np.linspace(0.5, 5.0, 300)) + list(np.linspace(5.0, 0.5, 300)):
        st = _drive(layer, ps, float(v), 0.0, 0.1)[-1]
        if not seq or seq[-1] != st:
            seq.append(st)
            at[(len(seq), st)] = float(v)
    assert seq == ["walk", "run", "sprint", "run", "walk"], seq
    up_run = at[(2, "run")] / walk_stride
    up_sprint = at[(3, "sprint")] / run_stride
    down_run = at[(4, "run")] / run_stride
    down_walk = at[(5, "walk")] / walk_stride
    tol = 0.08                                  # the estimator's lag on the ramp
    assert abs(up_run / (role.gaits[0].max_rate * (1 + h)) - 1) < tol, up_run
    assert abs(up_sprint / (role.gaits[1].max_rate * (1 + h)) - 1) < tol, up_sprint
    assert abs(down_run / (role.gaits[1].max_rate * (1 - h)) - 1) < tol, down_run
    assert abs(down_walk / (role.gaits[0].max_rate * (1 - h)) - 1) < tol, down_walk
    # held at the walk -> run threshold, jittered
    ps = layer.new_state(MARINE, "idle", "player")
    edge = role.gaits[0].max_rate * walk_stride
    switches, last = 0, None
    rng = np.random.default_rng(184)
    for _ in range(60):
        st = _drive(layer, ps, float(edge * (1 + rng.uniform(-0.03, 0.03))), 0.0, 0.1)[-1]
        switches += last is not None and st != last
        last = st
    assert switches == 0


# ---------------------------------------------------------------------------
# the direction choice, the residual twist, the phase over a switch
# ---------------------------------------------------------------------------

def test_the_sector_follows_the_travel_with_hysteresis_and_the_twist_is_the_residual(layer):
    """PROPERTY: sweeping the travel slowly once round the unit, the walk
    visits its sectors in order (forward, left, back, right, forward), each
    switch beyond the half-way line by up to ``sector_hysteresis_deg``;
    jittering the travel 5 deg about a sector edge does not flicker; and at
    every frame the legs' twist differs from the residual it aims for (the
    travel against the playing clip's own travel) only while slewing, and is
    bounded by the sector's half-width plus the hysteresis plus the clip's
    own offset from the sector's centre.

    BREAKS IF: the sector choice loses its hysteresis, the sectors are
    unordered, or the twist covers the full angle (+-90 deg) instead of the
    residual.
    """
    loco = layer.rig(MARINE).locomotion
    role = loco.roles["player"]
    walk = role.gaits[0]
    hy = role.sector_hysteresis
    speed = 1.0
    ps = layer.new_state(MARINE, "idle", "player")
    _drive(layer, ps, speed, 0.0, 1.0)
    visited = []
    for a in np.linspace(0.0, 360.0, 721):
        st = _drive(layer, ps, speed, float(a), 1.0 / 60.0)[-1]
        if not visited or visited[-1] != st:
            visited.append(st)
            if len(visited) > 1:
                s = walk.sectors[ps.sector]
                edge = math.degrees(s.center - s.lo)
                over = (a - edge + 180.0) % 360.0 - 180.0
                # past the edge by the hysteresis, plus the travel filter's lag
                # (smooth_s at this sweep's 30 deg/s) and one sim tick's
                lag = 30.0 * (SETTINGS.aim_travel.smooth_s + 1.0 / TPS) + 1.0
                assert 0.0 <= over <= math.degrees(hy) + lag, (st, a, edge)
        s = walk.sectors[ps.sector]
        clip_off = abs((loco.measures[s.legs_clip].travel - s.center + math.pi)
                       % (2 * math.pi) - math.pi)
        assert abs(ps.twist) <= max(s.lo, s.hi) + hy + clip_off + 1e-6
    assert visited == ["walk", "walk_left", "walk_back", "walk_right", "walk"], visited
    # jitter about the forward / left edge
    edge = math.degrees(walk.sectors[1].center - walk.sectors[1].lo)
    ps = layer.new_state(MARINE, "idle", "player")
    _drive(layer, ps, speed, edge - 3.0, 1.0)
    rng = np.random.default_rng(7)
    seen = {_drive(layer, ps, speed, float(edge + rng.uniform(-5, 5)), 0.05)[-1]
            for _ in range(80)}
    assert len(seen) == 1, seen
    # the twist settles on the residual
    ps = layer.new_state(MARINE, "idle", "player")
    _drive(layer, ps, speed, 120.0, 1.5)
    s = walk.sectors[ps.sector]
    want = (math.radians(120.0) - loco.measures[s.legs_clip].travel + math.pi) \
        % (2 * math.pi) - math.pi
    assert abs(ps.twist - want) < math.radians(0.5), (math.degrees(ps.twist),
                                                       math.degrees(want))


def test_a_switch_carries_the_gait_phase_over(layer):
    """PROPERTY: every gait clip on every layer -- and the crossfade's
    source -- shows the unit's ONE gait phase at its own measured offset,
    every frame; across a forward -> strafe switch the phase moves on by no
    more than the distance walked in the frame over the cycle, so the feet
    keep their step (the planted foot does not jump: it moves no more on the
    switch frames than the largest step elsewhere in the walk, plus 2 cm).

    BREAKS IF: a switch restarts the new clip at its start (the feet pop),
    or a layer's clip runs on its own clock out of step with the legs.
    """
    rig = layer.rig(MARINE)
    loco, bank = rig.locomotion, rig.bank
    ps = layer.new_state(MARINE, "idle", "player")
    rec = []
    _drive(layer, ps, 1.0, 0.0, 1.5, record=rec)
    n0 = len(rec)
    phases = []
    for k in range(90):
        _drive(layer, ps, 1.0, 90.0 if k >= 30 else 0.0, 1.0 / 60.0, record=rec)
        phases.append(ps.gait_phase)
        for clips, times in ((ps.clip, ps.time), (ps.prev_clip, ps.prev_time)):
            for layer_i in range(len(clips)):
                m = loco.measures.get(int(clips[layer_i]))
                if m is not None:
                    want = ((ps.gait_phase + m.offset) % 1.0) * bank.duration[int(clips[layer_i])]
                    assert abs(times[layer_i] - want) < 1e-9
    d = np.abs(np.diff(phases))
    d = np.minimum(d, 1.0 - d)
    # one frame's walk (at the estimate's speed: within 10 % of the true 1 m/s)
    assert d.max() <= 1.1 * DT / min(m.cycle_m for m in loco.measures.values())
    feet = _feet(layer)
    p = np.stack(rec)[:, feet, 0:3]
    step = np.linalg.norm(np.diff(p, axis=0), axis=-1).max(axis=1)
    calm = step[:n0 + 25].max()
    switch = step[n0 + 28:n0 + 45].max()
    assert switch <= calm + 0.02, (switch, calm)


def test_a_look_without_locomotion_steers_exactly_as_before(layer):
    """PROPERTY: on a Quaternius look (its skeleton has no locomotion
    table) ``PoseLayer.locomote`` returns the renderer's state unchanged and
    leaves the pose bit for bit what ``steer`` leaves -- the shipped zombies
    and the old marine move exactly as before #184.

    BREAKS IF: the locomotion leaks onto a skeleton that does not ask for
    it (a rate, a sector, a phase lock on the Quaternius clips).
    """
    look = quaternius_looks()[0]
    a = layer.new_state(look, "walk", "player")
    b = layer.new_state(look, "walk", "player")
    for k in range(120):
        rel = math.radians(40.0 + k)
        st = layer.locomote(a, "walk", 0.01, rel, True, DT, "player")
        assert st == "walk"
        layer.drive(a, st, DT, "player")
        layer.steer(b, rel, True, DT)
        layer.drive(b, "walk", DT, "player")
    assert np.array_equal(layer.compose(look, [a]).global_trs,
                          layer.compose(look, [b]).global_trs)


# ---------------------------------------------------------------------------
# the locomotion door
# ---------------------------------------------------------------------------

def _raw() -> dict:
    import copy
    return {"render": {"pose": copy.deepcopy(pl_mod._as_dict(CFG.render.pose))}}


@pytest.mark.parametrize("edit, match", [
    (lambda lo: lo.pop("feet"), "feet"),
    (lambda lo: lo.update(stride=1), "stride"),
    (lambda lo: lo["roles"]["player"].pop("stop_after_s"), "stop_after_s"),
    (lambda lo: lo["roles"]["player"].update(hysteresis="0.1"), "hysteresis"),
    (lambda lo: lo["roles"]["player"]["gaits"][0].pop("min_rate"), "min_rate"),
    (lambda lo: lo["roles"]["player"]["gaits"][-1].update(max_rate=2.0), "max_rate"),
    (lambda lo: lo["roles"]["player"]["gaits"][1].update(state="jog"), "jog"),
    (lambda lo: lo["roles"].update(ghost=lo["roles"]["player"]), "ghost"),
    (lambda lo: lo["roles"]["player"]["gaits"][0]["sectors"][1].update(center_deg=0.0),
     "centre"),
])
def test_the_locomotion_door_refuses_by_name(edit, match):
    """PROPERTY: a missing, unknown, mistyped or misplaced locomotion key is
    refused when the settings load, naming it (and a state or role the
    tables do not have, and two sectors at one centre).

    BREAKS IF: the door goes lenient (a typo'd dial would silently default).
    """
    d = _raw()
    edit(d["render"]["pose"]["skeletons"]["soma"]["locomotion"])
    with pytest.raises(PoseConfigError, match=match):
        pose_settings(d)


# ---------------------------------------------------------------------------
# directional hits (renderer hand-off)
# ---------------------------------------------------------------------------

class _Rl:
    def __getattr__(self, name):
        return lambda *a, **k: None


class _Skin:
    def prepare(self, *a):
        pass

    def apply(self, *a):
        pass

    def release(self):
        pass


@dataclass
class _Unit:
    id: int
    x: float
    y: float
    facing: float = math.pi / 2.0
    is_zombie: bool = False
    alive: bool = True
    footprint: int = 3
    move_path: list = None
    weapon_id: str = ""


def _renderer(monkeypatch, settings=None):
    from renderer import unit_model_renderer as umr
    from renderer.unit_looks import LookAssigner
    monkeypatch.setattr(umr, "rl", _Rl())
    monkeypatch.setattr(CFG.render.pose, "enabled", True)
    if settings is not None:
        monkeypatch.setattr(pl_mod, "pose_settings", lambda cfg: settings)
    r = umr.UnitModelRenderer()
    for name in all_look_names(looks_table(CFG)):
        model = SimpleNamespace(boneCount=30, name=name, meshCount=0)
        anims = [SimpleNamespace(tag=(name, k), frameCount=40) for k in range(2)]
        r._looks[name] = umr._Look(name=name, path=look_path(name), model=model,
                                   anims=anims, n_anims=2, clip_index={}, native_height=1.9)
        r._resolve[name] = r._looks[name]
    r._assigner = LookAssigner(looks_table(CFG))
    r._loaded = True
    r._setup_pose(skinner=_Skin())
    return r


def _draw(r, units, k):
    r.draw_units(units, 8.0, 1.0 + k * DT, camera3d=None, light_view=None)


# the victim (id 1) at tile (20, 20) facing north (+pi/2: screen up, world -y).
# #185: a hit carries the direction it TRAVELLED (Q16, x right, y down); the
# side it came from is minus that. Rows: travel -> the flinch of the source
# side (travelling west = from the unit's right, east = from its left, north
# (-y) = from behind, south (+y) = from the front).
_Q = 1 << 16
_TRAVEL = {"west": (-_Q, 0), "east": (_Q, 0), "north": (0, -_Q), "south": (0, _Q)}
_FROM = {"west": "Hit_Right", "east": "Hit_Left", "north": "Hit_Back",
         "south": "Hit_Chest"}


def _react_clip(monkeypatch, evs, units=None):
    """The flinch clip unit 1 plays after ``evs`` are queued and drawn."""
    r = _renderer(monkeypatch)
    units = units or [_Unit(1, 20.0, 20.0)]
    _draw(r, units, 0)
    r.queue_events(evs)
    _draw(r, units, 1)
    ps = r._anim[1].pose
    return r._pose.rig(ps.look).bank.names[ps.flinch_clip]


def _hit_clip(monkeypatch, travel, source="bullet", attacker=-1, extra=(), units=None):
    from simulation.events import UnitHitEvent
    dx, dy = travel
    evs = [UnitHitEvent(unit_id=1, damage=20, source=source, attacker_id=attacker,
                        dir_x_q=dx, dir_y_q=dy)] + list(extra)
    return _react_clip(monkeypatch, evs, units)


@pytest.mark.parametrize("travel", sorted(_TRAVEL))
def test_a_hit_flinches_from_the_side_its_travel_came_from(monkeypatch, travel):
    """PROPERTY (#185): a hit on a SOMA marine plays the flinch clip of the
    side the hit CAME FROM -- minus the event's own travel vector, against
    the victim's facing: front, left, behind, right.

    BREAKS IF: the source side is taken as the travel instead of minus it
    (every flinch reversed), the angle's sign or zero is wrong (a hit from
    the left plays the right flinch), or the vector is dropped on the way
    to the pose layer.
    """
    assert _hit_clip(monkeypatch, _TRAVEL[travel]) == _FROM[travel]


def test_a_hit_without_a_direction_flinches_from_the_front(monkeypatch):
    """PROPERTY: a hit carrying (0, 0) (no direction) plays the front
    flinch, whoever is credited.

    BREAKS IF: an unknown direction picks an arbitrary row, or a guess
    (the attacker's position) comes back as a fallback."""
    units = [_Unit(1, 20.0, 20.0), _Unit(3, 10.0, 20.0)]     # attacker on its left
    assert _hit_clip(monkeypatch, (0, 0), attacker=3, units=units) == "Hit_Chest"


def test_the_direction_is_the_events_not_a_guess(monkeypatch):
    """PROPERTY (#185, Erik's option c): the flinch reads ONLY the event's
    vector -- an explosion hit travelling east (blast west of the victim)
    plays the LEFT flinch although its thrower stands east and an
    ``ExplosionEvent`` lies east in the same frame.

    BREAKS IF: the deleted guesses return (the attacker's drawn position,
    the nearest ``ExplosionEvent``), as a source or as a fallback."""
    from simulation.events import ExplosionEvent
    units = [_Unit(1, 20.0, 20.0), _Unit(2, 30.0, 20.0)]    # thrower on its right
    blast = ExplosionEvent(pos=(28, 21), radius=3, kind="grenade")
    assert _hit_clip(monkeypatch, _TRAVEL["east"], source="explosion", attacker=2,
                     extra=[blast], units=units) == "Hit_Left"


@pytest.mark.parametrize("travel", sorted(_TRAVEL))
def test_a_stagger_plays_along_its_push(monkeypatch, travel):
    """PROPERTY (#185): a ``StaggerEvent`` plays the stagger reaction from
    the side its PUSH came from (minus the push vector, against the facing),
    at ``[render.pose.stagger] weight``, over that row's masks.

    BREAKS IF: the stagger is dropped (no reaction), reads the push's
    direction instead of its source side, or plays with the hit's weight
    or bones instead of its own row's."""
    from simulation.events import StaggerEvent
    dx, dy = _TRAVEL[travel]
    r = _renderer(monkeypatch)
    units = [_Unit(1, 20.0, 20.0)]
    _draw(r, units, 0)
    r.queue_events([StaggerEvent(unit_id=1, ticks=6, dir_x_q=dx, dir_y_q=dy)])
    _draw(r, units, 1)
    ps = r._anim[1].pose
    rig = r._pose.rig(ps.look)
    assert rig.bank.names[ps.flinch_clip] == _FROM[travel]
    assert ps.flinch_peak == SETTINGS.stagger.weight
    assert ps.flinch_ramps == (SETTINGS.stagger.ramp_in_s, SETTINGS.stagger.ramp_out_s)
    assert np.array_equal(ps.flinch_bones, rig.stagger_bones)


def test_a_role_without_the_stagger_reaction_does_not_stagger(monkeypatch):
    """PROPERTY: the stagger reaction follows the role's ``reactions`` list
    (the shipped player and zombie both carry it; a role without it plays
    nothing).

    BREAKS IF: the renderer plays the stagger regardless of the role."""
    from simulation.events import StaggerEvent
    roles = dict(SETTINGS.roles)
    roles["player"] = dataclasses.replace(
        roles["player"], reactions=tuple(x for x in roles["player"].reactions
                                         if x != "stagger"))
    r = _renderer(monkeypatch, settings=dataclasses.replace(SETTINGS, roles=roles))
    units = [_Unit(1, 20.0, 20.0)]
    _draw(r, units, 0)
    r.queue_events([StaggerEvent(unit_id=1, ticks=6, dir_x_q=_Q, dir_y_q=0)])
    _draw(r, units, 1)
    assert r._anim[1].pose.flinch_t < 0.0
    assert "stagger" in SETTINGS.roles["player"].reactions
    assert "stagger" in SETTINGS.roles["zombie"].reactions


def test_a_skeleton_without_hit_directions_keeps_its_single_flinch(layer):
    """PROPERTY: on a Quaternius look (no ``hit_directions``) every hit,
    from any side, plays the one ``[render.pose.hit] clip``.

    BREAKS IF: the directional rows leak onto a skeleton without them."""
    look = quaternius_looks()[0]
    rig = layer.rig(look)
    for src in (None, 0.0, 1.5, -1.5, 3.1):
        ps = layer.new_state(look, "idle", "player")
        layer.hit(ps, 30.0, "bullet", src)
        assert ps.flinch_clip == rig.hit_clip


# ---------------------------------------------------------------------------
# reload / throw (stand-in events; the sim adds the real ones in #183)
# ---------------------------------------------------------------------------

@dataclass
class _StandinReload:
    unit_id: int
    ticks: int


@dataclass
class _StandinThrow:
    unit_id: int
    kind: str
    ticks: int


def _timed_settings(release_s=1.0):
    return dataclasses.replace(
        SETTINGS, reload=TimedShotSettings(clip="Get_Up", layers=("upper", "head")),
        throw=TimedShotSettings(clip="Get_Up", layers=("upper", "head"), release_s=release_s))


def _with_standins(monkeypatch):
    from simulation import events as sev
    monkeypatch.setattr(sev, "ReloadEvent", _StandinReload, raising=False)
    monkeypatch.setattr(sev, "ThrowEvent", _StandinThrow, raising=False)


def _upper_time_after(r, units, seconds):
    """Draw ``seconds`` more of wall time (frames numbered on from the
    renderer's last one); the unit's upper layer clip time then."""
    k0 = getattr(r, "_test_frame", 1) + 1
    for k in range(k0, k0 + int(round(seconds / DT))):
        _draw(r, units, k)
        r._test_frame = k
    ps = r._anim[1].pose
    return ps, ps.time[list(SETTINGS.layers).index("upper")]


def test_a_reload_lasts_exactly_its_ticks(monkeypatch):
    """PROPERTY: a ``ReloadEvent(unit_id, ticks)`` plays the reload clip on
    its layers over exactly ``ticks / ticks_per_second`` seconds: half-way
    through, the clip is half-way; at the end, at its end (within a
    frame), and the layers then return to the state's clips.

    BREAKS IF: the clip plays at its own speed, the tick rate is read
    wrong, or the one-shot does not let go of its layers.
    """
    _with_standins(monkeypatch)
    r = _renderer(monkeypatch, settings=_timed_settings())
    units = [_Unit(1, 20.0, 20.0)]
    _draw(r, units, 0)
    ticks = 36                                   # 1.5 s at 24 tps
    r.queue_events([_StandinReload(unit_id=1, ticks=ticks)])
    _draw(r, units, 1)
    rig = r._pose.rig(r._anim[1].pose.look)
    dur = rig.bank.duration[rig.reload_clip]
    secs = ticks / CFG.clock.ticks_per_second
    ps, t_half = _upper_time_after(r, units, secs / 2 - DT)
    speed = dur / secs
    assert ps.clip[1] == rig.reload_clip
    assert abs(t_half - dur / 2) <= speed * DT + 1e-9, (t_half, dur)
    ps, t_end = _upper_time_after(r, units, secs / 2)
    assert abs(t_end - dur) <= speed * DT + 1e-9 or ps.clip[1] != rig.reload_clip
    _upper_time_after(r, units, 0.3)
    assert r._anim[1].pose.clip[1] != rig.reload_clip


@pytest.mark.parametrize("ticks", [7, 48])          # a wind-up shorter / longer than the clip's
def test_a_throw_releases_at_the_end_of_its_wind_up(monkeypatch, ticks):
    """PROPERTY: a ``ThrowEvent(unit_id, kind, ticks)`` at the start of the
    wind-up plays the throw clip so that its ``release_s`` is reached
    exactly when the wind-up ends (``ticks / ticks_per_second`` later,
    within a frame) -- for a wind-up shorter than the clip's own (it starts
    part-way) and longer (it plays slowed); after it, at the authored speed.

    BREAKS IF: the clip starts at 0 at its authored speed whatever the
    wind-up (the grenade would leave before or after the hand opens), or
    the release time is ignored.
    """
    _with_standins(monkeypatch)
    rel = 1.0
    r = _renderer(monkeypatch, settings=_timed_settings(rel))
    units = [_Unit(1, 20.0, 20.0)]
    _draw(r, units, 0)
    r.queue_events([_StandinThrow(unit_id=1, kind="frag", ticks=ticks)])
    _draw(r, units, 1)
    w = ticks / CFG.clock.ticks_per_second
    ps, t = _upper_time_after(r, units, w - DT)
    rig = r._pose.rig(ps.look)
    assert ps.clip[1] == rig.throw_clip
    speed = max(rel / w, rig.rate * rig.bank.rate[rig.throw_clip])
    assert abs(t - rel) <= speed * DT + 1e-6, (ticks, t, rel)


def test_blank_reload_and_throw_clips_change_nothing(monkeypatch):
    """PROPERTY: with blank rows (clip "" -- a game without the clips, as
    shipped until #188 filled them), reload and throw events change no pose,
    bit for bit; and the event classes are looked up only where the sim has
    them.

    BREAKS IF: a blank clip plays something, or a missing event class
    raises.
    """
    from renderer.unit_model_renderer import RELOAD_EVENT, THROW_EVENT
    from simulation import events as sev
    r0 = _renderer(monkeypatch)
    assert (r0._reload_event is None) == (not hasattr(sev, RELOAD_EVENT))
    assert (r0._throw_event is None) == (not hasattr(sev, THROW_EVENT))
    _with_standins(monkeypatch)
    blank = dataclasses.replace(
        SETTINGS, reload=TimedShotSettings(clip="", layers=("upper", "head")),
        throw=TimedShotSettings(clip="", layers=("upper", "head")))
    rows = []
    for events in (False, True):
        r = _renderer(monkeypatch, settings=blank)
        units = [_Unit(1, 20.0, 20.0)]
        out = []
        for k in range(40):
            if events and k == 3:
                r.queue_events([_StandinReload(unit_id=1, ticks=24),
                                _StandinThrow(unit_id=1, kind="frag", ticks=7)])
            _draw(r, units, k)
            ps = r._anim[1].pose
            out.append(r._pose.compose(ps.look, [ps]).global_trs.copy())
        rows.append(np.stack(out))
    assert np.array_equal(rows[0], rows[1])


# ---------------------------------------------------------------------------
# the held prop
# ---------------------------------------------------------------------------

def test_the_held_prop_rides_its_socket_bone_exactly(layer):
    """PROPERTY: the carbine's world transform is the body's model
    transform x the socket bone's COMPOSED global pose x the grip: its grip
    point lands exactly on the grip's place in the hand (to 1e-6 m), in
    every frame of a walk, whatever the hand does; the pose layer's TRS row
    of a bone is its FK matrix; and the body transform is raylib's own
    (``draw_model_ex``: translate x rotate-y x scale).

    BREAKS IF: the prop is placed from anything but the composed hand (the
    clip's raw key, the wrong bone, a stale frame), the row -> matrix
    conversion is wrong, or the body transform disagrees with raylib's.
    """
    import pyray as rl
    from renderer.held_props import (body_matrix, grip_matrix, held_prop_settings,
                                     model_extent, prop_world, trs_matrix)
    hp = held_prop_settings(CFG)
    prop = hp.prop_for("k5_carbine")
    assert prop is not None
    rig = layer.rig(MARINE)
    bone = rig.sockets[prop.socket]
    grip = grip_matrix(prop, prop.socket, model_extent(look_path(prop.model)))
    body = body_matrix(123.0, 45.0, 37.0, 21.5)
    rlm = rl.matrix_multiply(rl.matrix_multiply(rl.matrix_scale(21.5, 21.5, 21.5),
                                                rl.matrix_rotate_y(math.radians(37.0))),
                             rl.matrix_translate(123.0, 0.0, 45.0))
    want = np.array([[getattr(rlm, f"m{4 * c + r_}") for c in range(4)] for r_ in range(4)])
    assert np.allclose(body, want, atol=1e-4)
    ps = layer.new_state(MARINE, "idle", "player")
    offset = np.array(prop.grips[prop.socket].offset_m + (1.0,))
    rec = []
    _drive(layer, ps, 1.0, 30.0, 1.0, record=rec)
    sk = rig.bank.skeleton
    for row in rec[::7]:
        w = prop_world(body, row[bone], grip)
        got = w @ np.array(prop.grip + (1.0,))
        assert np.allclose(got, body @ trs_matrix(row[bone]) @ offset, atol=1e-6 * 21.5)
    c = rig.state_clips["walk"][0]
    o = rig.bank.offset[c]
    g = forward_kinematics(sk, rig.bank.t[o:o + 1], rig.bank.r[o:o + 1], None)[0]
    from renderer.pose_layer import global_trs
    assert np.allclose(trs_matrix(global_trs(g[None])[0][bone]), g[bone], atol=1e-5)


def test_only_a_look_with_the_socket_holds_the_prop(monkeypatch):
    """PROPERTY: a marine with the carbine's weapon id drawn with the SOMA
    marine holds it on the socket's bone; the same unit on a Quaternius
    look (no sockets table), a zombie (no weapon) and a unit whose weapon
    has no prop row hold nothing.

    BREAKS IF: the prop is drawn on a skeleton that lacks the socket, or is
    keyed by anything but the unit's weapon id.
    """
    from renderer import unit_model_renderer as umr
    from renderer.held_props import held_prop_settings
    r = _renderer(monkeypatch)
    hm = umr._HeldModel(name="cqb_carbine", path=look_path("cqb_carbine"), model=None,
                        extent=1.0)
    armed = _Unit(1, 20.0, 20.0, weapon_id="k5_carbine")
    plain = _Unit(2, 30.0, 20.0, weapon_id="laser_rifle_x")
    zombie = _Unit(3, 40.0, 20.0, is_zombie=True)
    worker_armed = _Unit(4, 50.0, 20.0, weapon_id="k5_carbine")
    quat = next(n for n in quaternius_looks() if n in r._looks)
    r._assigner._look[4] = quat                  # an armed unit drawn with a Quaternius look
    _draw(r, [armed, plain, worker_armed], 0)
    _draw(r, [zombie], 0)
    r._held = umr._HeldProps(settings=held_prop_settings(CFG), models={"cqb_carbine": hm})
    row = np.zeros((30, 10), dtype=np.float32)
    got = r._held_prop_of(armed, r._looks[MARINE], row)
    assert got is not None and got[2] == r._pose.rig(MARINE).sockets["right_hand"]
    assert r._held_prop_of(plain, r._looks[MARINE], row) is None
    zlook = r._looks[r._assigner.looks_for([zombie])[0]]
    assert r._held_prop_of(zombie, zlook, row) is None
    assert r._held_prop_of(worker_armed, r._looks[quat], np.zeros((53, 10))) is None


@pytest.mark.parametrize("edit, match", [
    (lambda h: h.pop("weapons"), "weapons"),
    (lambda h: h["weapons"].update(k9_blaster="carbine"), "k9_blaster"),
    (lambda h: h["weapons"].update(k5_carbine="musket"), "musket"),
    (lambda h: h["props"]["carbine"].pop("length_m"), "length_m"),
    (lambda h: h["props"]["carbine"].update(length_m=-1), "length_m"),
    (lambda h: h["props"]["carbine"].update(colour="red"), "colour"),
    (lambda h: h["props"]["carbine"].update(socket="left_hand"), "left_hand"),
    (lambda h: h["props"]["carbine"]["grips"]["right_hand"].pop("offset_m"), "offset_m"),
    (lambda h: h["props"]["carbine"].update(gloss=2.0), "gloss"),
])
def test_the_held_prop_door_refuses_by_name(edit, match):
    """PROPERTY: ``[render.held_props]`` refuses a missing, unknown or
    mistyped key, a weapon id that is no ``[weapons.*]`` row, a prop name
    that is no row and a default socket without a grip row -- naming it.

    BREAKS IF: the door goes lenient (a typo would draw no gun, silently).
    """
    import copy
    from renderer.held_props import HeldPropConfigError, _as_dict, held_prop_settings
    d = {"render": {"held_props": copy.deepcopy(_as_dict(CFG.render.held_props))},
         "weapons": _as_dict(CFG.weapons)}
    edit(d["render"]["held_props"])
    with pytest.raises(HeldPropConfigError, match=match):
        held_prop_settings(d)


def test_no_bone_is_named_in_the_locomotion_module():
    """PROPERTY (the skeleton-agnostic rule): no SOMA bone name is a string
    literal in ``renderer/pose_locomotion.py`` or ``renderer/held_props.py``
    -- feet and sockets are table data.

    BREAKS IF: a bone is hard-coded there."""
    import re
    from renderer.pose_layer import _glb_skeleton
    names = set(_glb_skeleton(look_path(MARINE)).names)
    for mod in ("pose_locomotion", "held_props"):
        src = (ROOT / "renderer" / f"{mod}.py").read_text(encoding="utf-8")
        literals = set(re.findall(r"""["']([^"'\n]+)["']""", src))
        assert not (literals & names), (mod, sorted(literals & names))
