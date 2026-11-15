"""The sound layer's pure half: what plays, how loud, where (#202 S1).

Design: docs/sound/playback_layer_design_2026-11-12.md. No raylib and no
simulation import here -- the model turns the sim's per-tick events, a small
read-only view of the world and the listener's place into COMMANDS
(`Start`, `Stop`) that the dumb half (`audio/device.py`) executes. It
writes nothing to the sim and draws from its own seeded generator, so no
digest can move and a headless run never needs it.

* A sound is a CUE row (`[sound.cues.<name>]`): its sample files, gain,
  variation, choke rule and reach. A BINDING (`[sound.bindings.<kind>]`)
  maps a weapon id or payload row to a cue; an unbound one is silent.
* `sound_settings` is the refusing door of every `[sound]` key.
* `acoustics` is the ONE place a voice's gain comes from. v1: distance
  only, to zero at the cue's reach. `reach = "loudness"` reads the weapon /
  payload row's own `loudness` (metres; the radius zombies hear it in) --
  a STOPGAP (Erik, 2026-11-12; #201 replaces it with a propagation law).
* The choke scheme (`choke = "emitter"`): a new voice of a cue fades the
  previous voice of that cue from the same emitter, so an automatic
  weapon's tails never stack and its last shot rings out.
* LOOPS (S2): a cue with `loop = true` is played from STATE, read once a
  frame (`update_loops`), never from an event. `[sound.loops]` names which
  loop cue each emitter kind plays: every running vent (only the cue's
  `max_emitters` loudest at the listener play), and the FIRE as ONE pair of
  voices for the whole fire -- `fire_small` and `fire_large`, their gains
  following how much burns within earshot (`fire_gains`), never a voice
  per tile. A loop's gain changes are ramped by the device over `ramp_ms`;
  paused, every loop goes to silence and comes back on resume.
"""
from __future__ import annotations

import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np

BINDING_KINDS = ("shot", "reload", "explosion")
CHOKE_RULES = ("none", "emitter")
REACH_LOUDNESS = "loudness"
STEAL_FADE_MS = 8.0

_TOP_KEYS = ("enabled", "master", "max_voices", "pan_width_m", "cues", "bindings",
             "loops")
_CUE_REQUIRED = ("files", "reach")
_CUE_OPTIONAL = {"gain": 1.0, "pitch_jitter": 0.0, "gain_jitter": 0.0,
                 "choke": "none", "choke_fade_ms": 12.0,
                 "loop": False, "max_emitters": 4}
_ONE_SHOT_ONLY = ("choke", "choke_fade_ms", "pitch_jitter", "gain_jitter")
LOOP_SLOTS = ("vent", "fire_small", "fire_large")
_LOOPS_KEYS = LOOP_SLOTS + ("ramp_ms", "fire_small_full_m2", "fire_large_from_m2",
                            "fire_large_full_m2")
FIRE_KEY = ("fire",)          # the fire pair's one emitter


class SoundConfigError(ValueError):
    """A `[sound]` setting refused (names the key)."""


@dataclass(frozen=True)
class Cue:
    name: str
    files: Tuple[str, ...]            # relative to the sounds root
    reach: Optional[float]            # metres; None = the row's loudness
    gain: float
    pitch_jitter: float
    gain_jitter: float
    choke: str
    choke_fade_ms: float
    loop: bool = False
    max_emitters: int = 4


@dataclass(frozen=True)
class LoopSettings:
    """`[sound.loops]`: which loop cue each emitter kind plays ("" = none)
    and the fire pair's law (`fire_gains`)."""
    vent: str
    fire_small: str
    fire_large: str
    ramp_ms: float
    fire_small_full_m2: float
    fire_large_from_m2: float
    fire_large_full_m2: float


@dataclass(frozen=True)
class SoundSettings:
    enabled: bool
    master: float
    max_voices: int
    pan_width_m: float
    cues: Dict[str, Cue]
    bindings: Dict[str, Dict[str, str]]     # kind -> {weapon / payload id -> cue}
    loops: LoopSettings

    def files(self) -> Tuple[str, ...]:
        return tuple(sorted({f for c in self.cues.values() for f in c.files}))


def _as_dict(obj):
    if isinstance(obj, Mapping):
        return {k: _as_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, bytes, int, float, list)):
        return {k: _as_dict(v) for k, v in vars(obj).items()}
    return obj


def _number(v, where: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SoundConfigError(f"{where} must be a number, got {v!r}")
    if not (lo <= float(v) <= hi):
        raise SoundConfigError(f"{where} must be in [{lo:g}, {hi:g}], got {v!r}")
    return float(v)


def _cue(name: str, row, root: Optional[Path]) -> Cue:
    where = f"[sound.cues.{name}]"
    if not isinstance(row, dict):
        raise SoundConfigError(f"{where} must be a table")
    for k in _CUE_REQUIRED:
        if k not in row:
            raise SoundConfigError(f"{where} is missing {k!r}")
    bad = sorted(set(row) - set(_CUE_REQUIRED) - set(_CUE_OPTIONAL))
    if bad:
        raise SoundConfigError(f"{where} has unknown keys {bad}")
    files = row["files"]
    if (not isinstance(files, list) or not files
            or not all(isinstance(f, str) and f for f in files)):
        raise SoundConfigError(f"{where} files must be a non-empty list of paths")
    if root is not None:
        for f in files:
            if not (root / f).is_file():
                raise SoundConfigError(f"{where} files: no such file {root / f}")
    reach = row["reach"]
    if reach == REACH_LOUDNESS:
        reach_m: Optional[float] = None
    else:
        if isinstance(reach, str):
            raise SoundConfigError(
                f"{where} reach must be metres or \"{REACH_LOUDNESS}\", got {reach!r}")
        reach_m = _number(reach, f"{where} reach", 0.1, 10000.0)
    v = {k: row.get(k, d) for k, d in _CUE_OPTIONAL.items()}
    if v["choke"] not in CHOKE_RULES:
        raise SoundConfigError(f"{where} choke must be one of {CHOKE_RULES}, got {v['choke']!r}")
    if not isinstance(v["loop"], bool):
        raise SoundConfigError(f"{where} loop must be true or false, got {v['loop']!r}")
    me = v["max_emitters"]
    if isinstance(me, bool) or not isinstance(me, int) or not (1 <= me <= 16):
        raise SoundConfigError(f"{where} max_emitters must be an integer in [1, 16], got {me!r}")
    if v["loop"]:
        if reach_m is None:
            raise SoundConfigError(
                f"{where} is a loop: its reach must be metres (a loop has no "
                f"weapon or payload row to read a loudness from)")
        for k in _ONE_SHOT_ONLY:
            if k in row:
                raise SoundConfigError(f"{where} is a loop: {k!r} is a one-shot's key")
    elif "max_emitters" in row:
        raise SoundConfigError(f"{where} max_emitters is a loop's key (set loop = true)")
    return Cue(name=name, files=tuple(files), reach=reach_m,
               gain=_number(v["gain"], f"{where} gain", 0.0, 4.0),
               pitch_jitter=_number(v["pitch_jitter"], f"{where} pitch_jitter", 0.0, 0.5),
               gain_jitter=_number(v["gain_jitter"], f"{where} gain_jitter", 0.0, 1.0),
               choke=v["choke"],
               choke_fade_ms=_number(v["choke_fade_ms"], f"{where} choke_fade_ms", 0.0, 500.0),
               loop=v["loop"], max_emitters=me)


def _loops(raw, cues: Dict[str, Cue]) -> LoopSettings:
    where = "[sound.loops]"
    if not isinstance(raw, dict):
        raise SoundConfigError(f"{where} must be a table")
    for k in _LOOPS_KEYS:
        if k not in raw:
            raise SoundConfigError(f"{where} is missing {k!r}")
    bad = sorted(set(raw) - set(_LOOPS_KEYS))
    if bad:
        raise SoundConfigError(f"{where} has unknown keys {bad}")
    for k in LOOP_SLOTS:
        name = raw[k]
        if not isinstance(name, str):
            raise SoundConfigError(f"{where} {k} must be a cue name or \"\", got {name!r}")
        if name and name not in cues:
            raise SoundConfigError(f"{where} {k} names an unknown cue {name!r}")
        if name and not cues[name].loop:
            raise SoundConfigError(f"{where} {k} names {name!r}, which is not a loop cue")
    lo = _number(raw["fire_large_from_m2"], f"{where} fire_large_from_m2", 0.0, 1e4)
    hi = _number(raw["fire_large_full_m2"], f"{where} fire_large_full_m2", 0.0, 1e4)
    if hi <= lo:
        raise SoundConfigError(
            f"{where} fire_large_full_m2 ({hi:g}) must be above fire_large_from_m2 ({lo:g})")
    return LoopSettings(
        vent=raw["vent"], fire_small=raw["fire_small"], fire_large=raw["fire_large"],
        ramp_ms=_number(raw["ramp_ms"], f"{where} ramp_ms", 0.0, 5000.0),
        fire_small_full_m2=_number(raw["fire_small_full_m2"],
                                   f"{where} fire_small_full_m2", 1e-3, 1e4),
        fire_large_from_m2=lo, fire_large_full_m2=hi)


def sound_settings(cfg, *, root: Optional[Path] = None,
                   weapon_ids: Optional[Iterable[str]] = None,
                   payload_ids: Optional[Iterable[str]] = None) -> SoundSettings:
    """THE door of `[sound]`. `root` (the sounds folder) makes it check that
    every file exists; `weapon_ids` / `payload_ids` make it refuse a binding
    that names a row which does not exist."""
    raw = _as_dict(getattr(cfg, "sound", None))
    if not isinstance(raw, dict):
        raise SoundConfigError("config.toml has no [sound] table")
    for k in _TOP_KEYS:
        if k not in raw:
            raise SoundConfigError(f"[sound] is missing {k!r}")
    bad = sorted(set(raw) - set(_TOP_KEYS))
    if bad:
        raise SoundConfigError(f"[sound] has unknown keys {bad}")
    if not isinstance(raw["enabled"], bool):
        raise SoundConfigError(f"[sound] enabled must be true or false, got {raw['enabled']!r}")
    mv = raw["max_voices"]
    if isinstance(mv, bool) or not isinstance(mv, int) or not (1 <= mv <= 64):
        raise SoundConfigError(f"[sound] max_voices must be an integer in [1, 64], got {mv!r}")
    if not isinstance(raw["cues"], dict):
        raise SoundConfigError("[sound.cues] must be a table of cue rows")
    cues = {name: _cue(name, row, root) for name, row in raw["cues"].items()}
    if not isinstance(raw["bindings"], dict):
        raise SoundConfigError("[sound.bindings] must be a table")
    bad = sorted(set(raw["bindings"]) - set(BINDING_KINDS))
    if bad:
        raise SoundConfigError(f"[sound.bindings] has unknown kinds {bad}; known: {BINDING_KINDS}")
    w_ids = None if weapon_ids is None else set(weapon_ids)
    p_ids = None if payload_ids is None else set(payload_ids)
    known = {"shot": w_ids, "reload": w_ids, "explosion": p_ids}
    bindings: Dict[str, Dict[str, str]] = {k: {} for k in BINDING_KINDS}
    for kind, table in raw["bindings"].items():
        if not isinstance(table, dict):
            raise SoundConfigError(f"[sound.bindings.{kind}] must be a table")
        ids = known[kind]
        for row_id, cue_name in table.items():
            if cue_name not in cues:
                raise SoundConfigError(
                    f"[sound.bindings.{kind}] {row_id} names an unknown cue {cue_name!r}")
            if cues[cue_name].loop:
                raise SoundConfigError(
                    f"[sound.bindings.{kind}] {row_id} names the loop cue {cue_name!r}; "
                    f"an event plays a one-shot")
            if ids is not None and row_id not in ids:
                what = "payload" if kind == "explosion" else "weapon"
                raise SoundConfigError(
                    f"[sound.bindings.{kind}] names an unknown {what} row {row_id!r}")
            bindings[kind][row_id] = cue_name
    return SoundSettings(
        enabled=raw["enabled"],
        master=_number(raw["master"], "[sound] master", 0.0, 1.0),
        max_voices=mv,
        pan_width_m=_number(raw["pan_width_m"], "[sound] pan_width_m", 0.1, 1000.0),
        cues=cues, bindings=bindings, loops=_loops(raw["loops"], cues))


def sample_seconds(path: Path) -> float:
    """A WAV's length, from its header."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def acoustics(dist_m: float, reach_m: float) -> float:
    """THE gain of a source `dist_m` from the listener: 1 at the listener,
    falling smoothly to exactly 0 at `reach_m`. A new propagation effect
    (walls, muffle, air density) is a term here, never a second path."""
    if reach_m <= 0.0 or dist_m >= reach_m:
        return 0.0
    k = 1.0 - max(0.0, dist_m) / reach_m
    return k * k


def fire_gains(area_small_m2: float, area_large_m2: float,
               loops: LoopSettings) -> Tuple[float, float]:
    """The fire pair's law (S2). Each input is the burning AREA heard at the
    listener: every burning tile's area times its fire level (0..1) times
    `acoustics` at that tile under its cue's reach. The small loop (crackle)
    rises linearly to full at `fire_small_full_m2`; the large one (roar)
    comes in from `fire_large_from_m2` and is full at `fire_large_full_m2`.
    Both 0 with nothing burning in earshot."""
    small = min(1.0, max(0.0, area_small_m2) / loops.fire_small_full_m2)
    span = loops.fire_large_full_m2 - loops.fire_large_from_m2
    large = min(1.0, max(0.0, (area_large_m2 - loops.fire_large_from_m2) / span))
    return small, large


def pan_of(dx_m: float, pan_width_m: float) -> float:
    """-1 = fully left, +1 = fully right, by the source's offset to the
    right of the listener on screen."""
    return max(-1.0, min(1.0, dx_m / pan_width_m))


class World(Protocol):
    """What the model may read of the game. main.py builds it over the sim."""
    tile_size_m: float

    def unit(self, unit_id: int) -> Optional[Tuple[float, float, str]]:
        """(centre x, centre y) in tiles and the weapon id, or None."""
        ...

    def weapon_loudness_m(self, weapon_id: str) -> float: ...

    def payload_loudness_m(self, payload: str) -> float: ...

    def vents(self) -> Sequence[Tuple[object, float, float]]:
        """Every RUNNING vent: (a stable key, centre x, centre y) in tiles."""
        ...

    def fire_near(self, x: float, y: float, r_tiles: float):
        """The burning tiles within `r_tiles` of (x, y): arrays of tile
        centre x, centre y and fire level in (0, 1]."""
        ...


@dataclass(frozen=True)
class Start:
    voice: int
    file: str
    gain: float          # 0..1, before the master
    pan: float           # -1 left .. +1 right
    pitch: float


@dataclass(frozen=True)
class Stop:
    voice: int
    fade_ms: float


@dataclass(frozen=True)
class LoopStart:
    """Start a looping voice at gain 0, `offset_s` into its file (so two
    vents never play the same loop in phase); the device ramps it to
    `gain`."""
    voice: int
    file: str
    gain: float
    pan: float
    offset_s: float


@dataclass(frozen=True)
class LoopSet:
    """A loop's new target gain (ramped by the device) and its pan."""
    voice: int
    gain: float
    pan: float


Command = Union[Start, Stop, LoopStart, LoopSet]


@dataclass
class _Voice:
    voice: int
    cue: str
    emitter: object
    gain: float
    ends_at: float


class SoundModel:
    """Events in, commands out. Holds only which voices it believes alive."""

    def __init__(self, settings: SoundSettings, durations: Mapping[str, float],
                 seed: int = 1):
        self.settings = settings
        self.durations = dict(durations)
        self.rng = random.Random(seed)
        self.voices: List[_Voice] = []
        self._next_voice = 1
        self._last_file: Dict[str, str] = {}
        self.loops: Dict[Tuple[str, object], int] = {}     # (cue, emitter) -> voice

    # -- loops -------------------------------------------------------------

    def update_loops(self, world: World, listener: Tuple[float, float],
                     paused: bool) -> List[Command]:
        """The loop commands for one frame: which emitters of each loop cue
        play, at what gain and pan. Paused: every loop to 0, none stopped or
        started, so the same voices come back on resume. Loops do not use
        the one-shots' voice pool: each loop cue holds at most its
        `max_emitters` voices."""
        out: List[Command] = []
        if paused:
            for voice in self.loops.values():
                out.append(LoopSet(voice, 0.0, 0.0))
            return out
        L = self.settings.loops
        want: Dict[str, List[Tuple[object, float, float]]] = {}
        if L.vent:
            want[L.vent] = self._vent_emitters(world, listener, self.settings.cues[L.vent])
        if L.fire_small or L.fire_large:
            for cue_name, g, pan in self._fire_emitters(world, listener):
                want.setdefault(cue_name, []).append((FIRE_KEY, g, pan))
        playing = dict(self.loops)
        for cue_name, emitters in want.items():
            cue = self.settings.cues[cue_name]
            heard = sorted((e for e in emitters if e[1] > 0.0), key=lambda e: -e[1])
            for key, g, pan in heard[: cue.max_emitters]:
                gain = min(1.0, cue.gain * g)
                k = (cue_name, key)
                voice = playing.pop(k, None)
                if voice is None:
                    voice = self._next_voice
                    self._next_voice += 1
                    self.loops[k] = voice
                    file = self._pick(cue)
                    offset = self.rng.uniform(0.0, max(0.0, self.durations.get(file, 0.0)))
                    out.append(LoopStart(voice, file, gain, pan, offset))
                else:
                    out.append(LoopSet(voice, gain, pan))
        for k, voice in playing.items():    # out of reach, out of the loudest, gone
            out.append(Stop(voice, L.ramp_ms))
            del self.loops[k]
        return out

    def _vent_emitters(self, world: World, listener, cue: Cue):
        ts = float(world.tile_size_m)
        out = []
        for key, x, y in world.vents():
            dx_m = (x - listener[0]) * ts
            dy_m = (y - listener[1]) * ts
            g = acoustics((dx_m * dx_m + dy_m * dy_m) ** 0.5, cue.reach)
            out.append((key, g, pan_of(dx_m, self.settings.pan_width_m)))
        return out

    def _fire_emitters(self, world: World, listener):
        """[(cue, gain 0..1, pan)] of the fire pair: ONE emitter for the
        whole fire, panned to the heard-weighted mean of where it burns."""
        L = self.settings.loops
        cues = self.settings.cues
        names = [n for n in (L.fire_small, L.fire_large) if n]
        ts = float(world.tile_size_m)
        r_m = max(cues[n].reach for n in names)
        xs, ys, level = world.fire_near(listener[0], listener[1], r_m / ts)
        xs = np.asarray(xs, dtype=np.float64)
        area = {n: 0.0 for n in names}
        pan = {n: 0.0 for n in names}
        if xs.size:
            dx_m = (xs - listener[0]) * ts
            dy_m = (np.asarray(ys, dtype=np.float64) - listener[1]) * ts
            d = np.sqrt(dx_m * dx_m + dy_m * dy_m)
            level = np.asarray(level, dtype=np.float64)
            for n in names:
                k = np.clip(1.0 - d / cues[n].reach, 0.0, None) ** 2   # acoustics, per tile
                w = level * k
                tot = float(w.sum())
                area[n] = tot * ts * ts
                if tot > 0.0:
                    pan[n] = pan_of(float((w * dx_m).sum()) / tot, self.settings.pan_width_m)
        small, large = fire_gains(area.get(L.fire_small, 0.0),
                                  area.get(L.fire_large, 0.0), L)
        out = []
        if L.fire_small:
            out.append((L.fire_small, small, pan[L.fire_small]))
        if L.fire_large:
            out.append((L.fire_large, large, pan[L.fire_large]))
        return out

    # -- one-shots ---------------------------------------------------------

    def on_events(self, events: Sequence, world: World,
                  listener: Tuple[float, float], now: float) -> List[Command]:
        """The commands for one sim step's events. `listener` is in tiles,
        `now` in wall seconds."""
        out: List[Command] = []
        self._expire(now)
        b = self.settings.bindings
        for ev in events:
            kind = type(ev).__name__
            if kind == "ShotFiredEvent":
                if not getattr(ev, "launch", True):
                    continue
                u = world.unit(ev.unit_id)
                if u is None:
                    continue
                cue = b["shot"].get(u[2])
                if cue is not None:
                    x, y = ev.from_tile
                    self._start(out, cue, ("unit", ev.unit_id), (float(x), float(y)),
                                world.weapon_loudness_m(u[2]), world, listener, now)
            elif kind == "ReloadEvent":
                u = world.unit(ev.unit_id)
                if u is None:
                    continue
                cue = b["reload"].get(u[2])
                if cue is not None:
                    self._start(out, cue, ("unit", ev.unit_id), (u[0], u[1]),
                                world.weapon_loudness_m(u[2]), world, listener, now)
            elif kind == "ExplosionEvent":
                name = getattr(ev, "payload", "")
                cue = b["explosion"].get(name)
                if cue is not None:
                    x, y = ev.pos
                    self._start(out, cue, ("pos", float(x), float(y)), (float(x), float(y)),
                                world.payload_loudness_m(name), world, listener, now)
        return out

    def _start(self, out: List[Command], cue_name: str, emitter: object,
               pos: Tuple[float, float], loudness_m: float, world: World,
               listener: Tuple[float, float], now: float) -> None:
        cue = self.settings.cues[cue_name]
        ts = float(world.tile_size_m)
        dx_m = (pos[0] - listener[0]) * ts
        dy_m = (pos[1] - listener[1]) * ts
        reach = loudness_m if cue.reach is None else cue.reach
        g = acoustics((dx_m * dx_m + dy_m * dy_m) ** 0.5, reach)
        if cue.choke == "emitter":
            # the previous voice is choked whether or not the new one is
            # audible: the emitter's sound moved on
            for v in [v for v in self.voices if v.cue == cue_name and v.emitter == emitter]:
                out.append(Stop(v.voice, cue.choke_fade_ms))
                self.voices.remove(v)
        if g <= 0.0:
            return
        gain = min(1.0, cue.gain * g * (1.0 - self.rng.uniform(0.0, cue.gain_jitter)))
        pitch = 1.0 + self.rng.uniform(-cue.pitch_jitter, cue.pitch_jitter)
        file = self._pick(cue)
        if len(self.voices) >= self.settings.max_voices:
            quietest = min(self.voices, key=lambda v: v.gain)
            if quietest.gain >= gain:
                return                      # the newcomer is the quietest: drop it
            out.append(Stop(quietest.voice, STEAL_FADE_MS))
            self.voices.remove(quietest)
        voice = self._next_voice
        self._next_voice += 1
        self.voices.append(_Voice(voice, cue_name, emitter, gain,
                                  now + self.durations.get(file, 0.0) / pitch))
        out.append(Start(voice, file, gain, pan_of(dx_m, self.settings.pan_width_m), pitch))

    def _pick(self, cue: Cue) -> str:
        files = cue.files
        if len(files) > 1:
            last = self._last_file.get(cue.name)
            files = tuple(f for f in files if f != last)
        f = self.rng.choice(files)
        self._last_file[cue.name] = f
        return f

    def _expire(self, now: float) -> None:
        self.voices = [v for v in self.voices if v.ends_at > now]
