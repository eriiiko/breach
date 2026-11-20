"""The sound layer's pure half (audio/model.py, #202 S1). Headless: no
raylib, no simulation, fake events and a fake world.

Properties held (each names the change that must break it):
  - a sound's gain falls with distance and is exactly 0 at its reach, so a
    source out of reach starts no voice at all;
  - `reach = "loudness"` reads the weapon / payload row's own loudness: the
    player hears a shot exactly as far as the horde does (the stopgap, #201);
  - the choke scheme: a unit's new shot stops ITS previous voice and nobody
    else's, so automatic fire never stacks tails and two shooters never cut
    each other off;
  - an unbound weapon or payload is silent, not an error;
  - the voice pool never exceeds its cap and gives way quietest-first;
  - the door refuses a bad row by name;
  - the shipped config passes its own door and every file it names exists.
LOOPS (#202 S2):
  - a vent sounds only within its cue's reach and at most `max_emitters`
    vents sound, the loudest at the listener (breaks if the cap is dropped
    or the choice is not by gain);
  - a loop that keeps sounding is never restarted (LoopSet, no new start),
    and one that leaves earshot is stopped with the ramp, not cut;
  - paused, every loop goes to 0 and none is stopped or started, so the
    same voices come back on resume;
  - the whole fire is ONE voice per fire cue however many tiles burn, the
    crackle grows with the burning area heard and saturates, the roar is
    silent below its threshold and full above its top; nothing burning in
    earshot = no fire voice;
  - the fire pans to the side it burns on.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest

from audio import model as am

TILE_M = 0.5


class ShotFiredEvent:
    def __init__(self, unit_id, from_tile, launch=True):
        self.unit_id, self.from_tile, self.launch = unit_id, from_tile, launch


class ExplosionEvent:
    def __init__(self, pos, payload):
        self.pos, self.payload = pos, payload


class ReloadEvent:
    def __init__(self, unit_id):
        self.unit_id = unit_id


class World:
    tile_size_m = TILE_M

    def __init__(self, units, loud=None):
        self.units = units
        self.loud = loud or {"carbine": 24.0, "frag": 40.0, "pistol": 10.0}

    def unit(self, uid):
        return self.units.get(uid)

    def weapon_loudness_m(self, w):
        return self.loud.get(w, 0.0)

    def payload_loudness_m(self, p):
        return self.loud.get(p, 0.0)


LOOPS = {"vent": "fan", "fire_small": "crackle", "fire_large": "roar",
         "ramp_ms": 150.0, "fire_small_full_m2": 0.5, "fire_large_from_m2": 1.5,
         "fire_large_full_m2": 10.0}


def _cfg(**over):
    sound = {
        "enabled": True, "master": 0.8, "max_voices": 4, "pan_width_m": 12.0,
        "cues": {
            "shot": {"files": ["a.wav", "b.wav"], "reach": "loudness",
                     "choke": "emitter", "choke_fade_ms": 12.0},
            "boom": {"files": ["boom.wav"], "reach": "loudness"},
            "click": {"files": ["click.wav"], "reach": 5.0},
            "fan": {"files": ["fan.wav"], "reach": 8.0, "loop": True, "max_emitters": 2},
            "crackle": {"files": ["crackle.wav"], "reach": 15.0, "loop": True,
                        "max_emitters": 1},
            "roar": {"files": ["roar.wav"], "reach": 30.0, "loop": True,
                     "max_emitters": 1},
        },
        "loops": dict(LOOPS),
        "bindings": {"shot": {"carbine": "shot"}, "reload": {"carbine": "click"},
                     "explosion": {"frag": "boom"}},
    }
    sound.update(over)
    return SimpleNamespace(sound=sound)


def _model(**over):
    s = am.sound_settings(_cfg(**over))
    return am.SoundModel(s, {"a.wav": 3.0, "b.wav": 3.0, "boom.wav": 10.0, "click.wav": 1.0})


def _starts(cmds):
    return [c for c in cmds if isinstance(c, am.Start)]


def _stops(cmds):
    return [c for c in cmds if isinstance(c, am.Stop)]


# -- the gain law ------------------------------------------------------------

def test_gain_falls_with_distance_and_is_zero_at_the_reach():
    reach = 24.0
    gains = [am.acoustics(d, reach) for d in (0.0, 3.0, 6.0, 12.0, 18.0, 23.9)]
    assert gains[0] == 1.0
    assert all(b < a for a, b in zip(gains, gains[1:]))
    assert all(g > 0.0 for g in gains)
    assert am.acoustics(reach, reach) == 0.0
    assert am.acoustics(reach * 3, reach) == 0.0
    assert am.acoustics(1.0, 0.0) == 0.0


@pytest.mark.parametrize("loud_m", [10.0, 24.0, 100.0])
def test_a_shot_is_heard_exactly_as_far_as_its_rows_loudness(loud_m):
    w = World({1: (0.0, 0.0, "carbine")}, loud={"carbine": loud_m})
    inside = (loud_m * 0.95 / TILE_M, 0.0)
    outside = (loud_m * 1.05 / TILE_M, 0.0)
    assert _starts(_model().on_events([ShotFiredEvent(1, inside)], w, (0.0, 0.0), 0.0))
    assert not _starts(_model().on_events([ShotFiredEvent(1, outside)], w, (0.0, 0.0), 0.0))


def test_a_cue_with_its_own_reach_ignores_the_rows_loudness():
    w = World({1: (0.0, 0.0, "carbine")})
    m = _model()
    near = m.on_events([ReloadEvent(1)], w, (4.0 / TILE_M, 0.0), 0.0)      # 4 m < 5 m
    far = m.on_events([ReloadEvent(1)], w, (6.0 / TILE_M, 0.0), 0.0)       # 6 m < 24 m loudness
    assert _starts(near) and not _starts(far)


def test_a_source_to_the_right_pans_right_and_saturates():
    assert am.pan_of(0.0, 12.0) == 0.0
    assert 0.0 < am.pan_of(3.0, 12.0) < am.pan_of(6.0, 12.0) < 1.0
    assert am.pan_of(50.0, 12.0) == 1.0 and am.pan_of(-50.0, 12.0) == -1.0
    w = World({1: (0.0, 0.0, "carbine")})
    s = _starts(_model().on_events([ShotFiredEvent(1, (8.0, 0.0))], w, (0.0, 0.0), 0.0))[0]
    assert s.pan > 0.0


# -- the choke scheme --------------------------------------------------------

def test_a_units_new_shot_chokes_its_previous_voice_and_only_that():
    w = World({1: (0.0, 0.0, "carbine"), 2: (2.0, 0.0, "carbine")})
    m = _model()
    first = _starts(m.on_events([ShotFiredEvent(1, (0.0, 0.0))], w, (0.0, 0.0), 0.00))[0]
    other = _starts(m.on_events([ShotFiredEvent(2, (2.0, 0.0))], w, (0.0, 0.0), 0.04))[0]
    cmds = m.on_events([ShotFiredEvent(1, (0.0, 0.0))], w, (0.0, 0.0), 0.083)
    assert [c.voice for c in _stops(cmds)] == [first.voice]
    assert other.voice not in [c.voice for c in _stops(cmds)]
    assert len(_starts(cmds)) == 1


def test_a_burst_never_holds_more_than_one_voice_per_shooter():
    w = World({1: (0.0, 0.0, "carbine")})
    m = _model()
    for k in range(30):
        m.on_events([ShotFiredEvent(1, (0.0, 0.0))], w, (0.0, 0.0), k / 12.0)
        assert len(m.voices) == 1


def test_the_last_shot_is_never_stopped():
    w = World({1: (0.0, 0.0, "carbine")})
    m = _model()
    stopped, started = set(), []
    for k in range(5):
        cmds = m.on_events([ShotFiredEvent(1, (0.0, 0.0))], w, (0.0, 0.0), k / 12.0)
        stopped |= {c.voice for c in _stops(cmds)}
        started += [c.voice for c in _starts(cmds)]
    assert started[-1] not in stopped and set(started[:-1]) == stopped


def test_a_later_segment_of_a_slow_round_makes_no_sound():
    w = World({1: (0.0, 0.0, "carbine")})
    assert not _model().on_events([ShotFiredEvent(1, (0.0, 0.0), launch=False)],
                                  w, (0.0, 0.0), 0.0)


# -- bindings and the pool ---------------------------------------------------

def test_an_unbound_weapon_or_payload_is_silent():
    w = World({1: (0.0, 0.0, "pistol")})
    m = _model()
    assert m.on_events([ShotFiredEvent(1, (0.0, 0.0)), ReloadEvent(1),
                        ExplosionEvent((0.0, 0.0), "smoke"),
                        ExplosionEvent((0.0, 0.0), "")], w, (0.0, 0.0), 0.0) == []
    assert m.on_events([ShotFiredEvent(99, (0.0, 0.0))], w, (0.0, 0.0), 0.0) == []


def test_an_explosion_plays_its_payload_rows_cue():
    w = World({})
    s = _starts(_model().on_events([ExplosionEvent((4.0, 2.0), "frag")], w, (0.0, 0.0), 0.0))
    assert [c.file for c in s] == ["boom.wav"]


def test_the_pool_holds_its_cap_and_the_quietest_gives_way():
    w = World({})
    m = _model()                       # max_voices = 4
    # four distant explosions, then a near one
    for k in range(4):
        m.on_events([ExplosionEvent((60.0 + k, 0.0), "frag")], w, (0.0, 0.0), 0.0)
    assert len(m.voices) == 4
    quiet = min(m.voices, key=lambda v: v.gain).voice
    cmds = m.on_events([ExplosionEvent((1.0, 0.0), "frag")], w, (0.0, 0.0), 0.1)
    assert [c.voice for c in _stops(cmds)] == [quiet] and len(_starts(cmds)) == 1
    assert len(m.voices) == 4
    # a newcomer quieter than everything playing is dropped, nothing stopped
    assert m.on_events([ExplosionEvent((75.0, 0.0), "frag")], w, (0.0, 0.0), 0.2) == []


def test_finished_voices_leave_the_pool():
    w = World({})
    m = _model()
    m.on_events([ExplosionEvent((1.0, 0.0), "frag")], w, (0.0, 0.0), 0.0)
    m.on_events([], w, (0.0, 0.0), 11.0)          # boom.wav is 10 s
    assert m.voices == []


def test_several_files_never_repeat_twice_running():
    w = World({1: (0.0, 0.0, "carbine")})
    m = _model()
    files = [_starts(m.on_events([ShotFiredEvent(1, (0.0, 0.0))], w, (0.0, 0.0), k * 0.1))[0].file
             for k in range(20)]
    assert all(a != b for a, b in zip(files, files[1:]))


def test_the_model_is_repeatable():
    w = World({1: (0.0, 0.0, "carbine")})
    runs = []
    for _ in range(2):
        m = _model()
        runs.append([m.on_events([ShotFiredEvent(1, (3.0, 1.0))], w, (0.0, 0.0), k * 0.1)
                     for k in range(10)])
    assert runs[0] == runs[1]


# -- the door ----------------------------------------------------------------

@pytest.mark.parametrize("mutate, word", [
    (lambda s: s.pop("master"), "master"),
    (lambda s: s.update(volume=1.0), "volume"),
    (lambda s: s.update(max_voices=0), "max_voices"),
    (lambda s: s["cues"]["shot"].pop("files"), "files"),
    (lambda s: s["cues"]["shot"].update(reach="far"), "reach"),
    (lambda s: s["cues"]["shot"].update(choke="all"), "choke"),
    (lambda s: s["cues"]["shot"].update(loop=True), "loop"),
    (lambda s: s["cues"]["fan"].update(reach="loudness"), "loop"),
    (lambda s: s["cues"]["fan"].update(choke="emitter"), "choke"),
    (lambda s: s["cues"]["click"].update(max_emitters=2), "max_emitters"),
    (lambda s: s["cues"]["fan"].update(max_emitters=0), "max_emitters"),
    (lambda s: s.pop("loops"), "loops"),
    (lambda s: s["loops"].update(vent="click"), "not a loop cue"),
    (lambda s: s["loops"].update(fire_small="nope"), "nope"),
    (lambda s: s["loops"].pop("ramp_ms"), "ramp_ms"),
    (lambda s: s["loops"].update(fire_large_full_m2=1.0), "fire_large_full_m2"),
    (lambda s: s["bindings"]["reload"].update(carbine="fan"), "loop cue"),
    (lambda s: s["bindings"]["shot"].update(carbine="nope"), "nope"),
    (lambda s: s["bindings"].update(footstep={}), "footstep"),
])
def test_the_door_refuses_by_name(mutate, word):
    cfg = _cfg()
    mutate(cfg.sound)
    with pytest.raises(am.SoundConfigError, match=word):
        am.sound_settings(cfg)


def test_the_door_refuses_an_unknown_weapon_payload_or_file(tmp_path):
    with pytest.raises(am.SoundConfigError, match="carbine"):
        am.sound_settings(_cfg(), weapon_ids=["rifle"], payload_ids=["frag"])
    with pytest.raises(am.SoundConfigError, match="frag"):
        am.sound_settings(_cfg(), weapon_ids=["carbine"], payload_ids=[])
    with pytest.raises(am.SoundConfigError, match="a.wav"):
        am.sound_settings(_cfg(), root=tmp_path)


def test_the_shipped_config_passes_its_door_against_the_real_tables():
    from config import CFG
    weapons = set(vars(CFG.weapons)) if hasattr(CFG, "weapons") else None
    payloads = set(vars(CFG.payloads)) if hasattr(CFG, "payloads") else None
    assert weapons and payloads
    s = am.sound_settings(CFG, root=ROOT / "assets" / "sounds",
                          weapon_ids=weapons, payload_ids=payloads)
    for f in s.files():
        assert am.sample_seconds(ROOT / "assets" / "sounds" / f) > 0.05


# -- loops (#202 S2) -----------------------------------------------------------

class LoopWorld(World):
    def __init__(self, vents=(), fire=()):
        super().__init__({})
        self._vents = list(vents)            # (key, x, y) tiles
        self._fire = list(fire)              # (x, y, level) tiles

    def vents(self):
        return self._vents

    def fire_near(self, x, y, r):
        f = [t for t in self._fire if abs(t[0] - x) <= r and abs(t[1] - y) <= r]
        return ([t[0] for t in f], [t[1] for t in f], [t[2] for t in f])


def _loop_model(**over):
    s = am.sound_settings(_cfg(**over))
    return am.SoundModel(s, {"fan.wav": 20.0, "crackle.wav": 20.0, "roar.wav": 10.0})


def _kinds(cmds, kind):
    return [c for c in cmds if isinstance(c, kind)]


def _m(metres):
    return metres / TILE_M


def test_a_vent_sounds_only_within_its_reach():
    m = _loop_model()
    far = m.update_loops(LoopWorld([("v", _m(8.5), 0.0)]), (0.0, 0.0), False)
    assert far == []
    near = m.update_loops(LoopWorld([("v", _m(7.5), 0.0)]), (0.0, 0.0), False)
    (st,) = _kinds(near, am.LoopStart)
    assert st.file == "fan.wav" and 0.0 < st.gain < 1.0 and st.pan > 0.0


def test_only_the_loudest_vents_sound_up_to_the_cap():
    m = _loop_model()                                   # fan max_emitters = 2
    w = LoopWorld([("a", _m(6.0), 0.0), ("b", _m(1.0), 0.0), ("c", _m(3.0), 0.0)])
    cmds = m.update_loops(w, (0.0, 0.0), False)
    assert len(_kinds(cmds, am.LoopStart)) == 2
    assert {k[1] for k in m.loops} == {"b", "c"}
    # "a" walks right up to the listener: it takes the place of the quieter "c"
    w2 = LoopWorld([("a", _m(0.5), 0.0), ("b", _m(1.0), 0.0), ("c", _m(3.0), 0.0)])
    voice_c = m.loops[("fan", "c")]
    cmds = m.update_loops(w2, (0.0, 0.0), False)
    stops = _kinds(cmds, am.Stop)
    assert [c.voice for c in stops] == [voice_c] and stops[0].fade_ms == LOOPS["ramp_ms"]
    assert {k[1] for k in m.loops} == {"a", "b"}


def test_a_sounding_loop_is_never_restarted():
    m = _loop_model()
    w = LoopWorld([("v", _m(2.0), 0.0)], fire=[(_m(1.0), 0.0, 1.0)])
    first = m.update_loops(w, (0.0, 0.0), False)
    voices = {c.voice for c in _kinds(first, am.LoopStart)}
    assert len(voices) == 2                        # the vent and the crackle
    for _ in range(10):
        cmds = m.update_loops(w, (0.0, 0.0), False)
        assert not _kinds(cmds, am.LoopStart) and not _kinds(cmds, am.Stop)
        assert {c.voice for c in _kinds(cmds, am.LoopSet)} == voices


def test_two_vents_never_start_in_phase():
    m = _loop_model()
    cmds = m.update_loops(LoopWorld([("a", 1.0, 0.0), ("b", -1.0, 0.0)]), (0.0, 0.0), False)
    a, b = _kinds(cmds, am.LoopStart)
    assert a.offset_s != b.offset_s and 0.0 <= a.offset_s <= 20.0


def test_paused_loops_fall_silent_and_come_back_as_the_same_voices():
    m = _loop_model()
    w = LoopWorld([("v", _m(2.0), 0.0)], fire=[(_m(1.0), 0.0, 1.0)])
    m.update_loops(w, (0.0, 0.0), False)
    before = dict(m.loops)
    paused = m.update_loops(w, (0.0, 0.0), True)
    assert paused and all(isinstance(c, am.LoopSet) and c.gain == 0.0 for c in paused)
    resumed = m.update_loops(w, (0.0, 0.0), False)
    assert m.loops == before and not _kinds(resumed, am.LoopStart)
    assert all(c.gain > 0.0 for c in _kinds(resumed, am.LoopSet))


def _fire_gains(m, tiles):
    cmds = m.update_loops(LoopWorld(fire=tiles), (0.0, 0.0), False)
    g = {"crackle.wav": 0.0, "roar.wav": 0.0}
    for c in _kinds(cmds, am.LoopStart):
        g[c.file] = c.gain
    by_voice = {v: k[0] for k, v in m.loops.items()}
    for c in _kinds(cmds, am.LoopSet):
        g[by_voice[c.voice] + ".wav"] = c.gain
    return g, cmds


def test_the_whole_fire_is_one_voice_per_fire_cue():
    m = _loop_model()
    tiles = [(x * 0.5, y * 0.5, 1.0) for x in range(-20, 21) for y in range(-20, 21)]
    _, cmds = _fire_gains(m, tiles)
    assert sorted(c.file for c in _kinds(cmds, am.LoopStart)) == ["crackle.wav", "roar.wav"]
    assert len(m.loops) == 2


def test_the_fire_pair_follows_the_burning_area():
    area_tile = TILE_M * TILE_M
    crackle, roar = [], []
    for n in (1, 2, 4, 8, 40, 400, 4000):          # tiles burning right at the listener
        g, _ = _fire_gains(_loop_model(), [(0.0, 0.0, 1.0)] * n)
        crackle.append(g["crackle.wav"])
        roar.append(g["roar.wav"])
        if n * area_tile < LOOPS["fire_large_from_m2"]:
            assert g["roar.wav"] == 0.0
        if n * area_tile >= LOOPS["fire_large_full_m2"]:
            assert g["roar.wav"] == 1.0
    assert all(b >= a for a, b in zip(crackle, crackle[1:])) and crackle[0] > 0.0
    assert crackle[-1] == 1.0
    assert all(b >= a for a, b in zip(roar, roar[1:]))
    # a weaker fire (a lower fire level) is heard as less of one
    weak, _ = _fire_gains(_loop_model(), [(0.0, 0.0, 0.25)] * 2)
    full, _ = _fire_gains(_loop_model(), [(0.0, 0.0, 1.0)] * 2)
    assert 0.0 < weak["crackle.wav"] < full["crackle.wav"]


def test_a_fire_out_of_earshot_or_out_makes_no_sound():
    m = _loop_model()
    assert m.update_loops(LoopWorld(fire=[(_m(40.0), 0.0, 1.0)]), (0.0, 0.0), False) == []
    m.update_loops(LoopWorld(fire=[(0.0, 0.0, 1.0)]), (0.0, 0.0), False)
    assert len(m.loops) == 1                        # the crackle; one tile is no roar
    cmds = m.update_loops(LoopWorld(), (0.0, 0.0), False)
    assert len(_kinds(cmds, am.Stop)) == 1 and m.loops == {}


def test_the_fire_pans_to_where_it_burns():
    m = _loop_model()
    cmds = m.update_loops(LoopWorld(fire=[(-_m(3.0), 0.0, 1.0)] * 4), (0.0, 0.0), False)
    assert _kinds(cmds, am.LoopStart)[0].pan < 0.0
    m = _loop_model()
    cmds = m.update_loops(LoopWorld(fire=[(_m(3.0), 0.0, 1.0)] * 4), (0.0, 0.0), False)
    assert _kinds(cmds, am.LoopStart)[0].pan > 0.0


def test_a_silent_slot_plays_nothing():
    m = _loop_model(loops=dict(LOOPS, vent="", fire_small="", fire_large=""))
    w = LoopWorld([("v", 0.0, 0.0)], fire=[(0.0, 0.0, 1.0)] * 100)
    assert m.update_loops(w, (0.0, 0.0), False) == []
