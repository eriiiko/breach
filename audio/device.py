"""The sound layer's dumb half: raylib's audio device (#202 S1).

Executes the model's commands (`audio/model.py`) and decides nothing. Each
sample file is loaded once and played through a small pool of aliases; a
`Stop` fades its voice out over the command's milliseconds, stepped once
per `update` call (per frame -- raylib's simple playback has no per-sample
ramp; design doc section 9).

raylib 5.5's `SetSoundPan` takes 0..1 with 1.0 = LEFT; the model's pan is
-1 left .. +1 right, converted here and nowhere else.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pyray as rl

from audio.model import Command, SoundSettings, Start, Stop, sample_seconds

ALIASES_PER_FILE = 8


class _Playing:
    __slots__ = ("sound", "base", "fade_from", "fade_s")

    def __init__(self, sound, base: float):
        self.sound = sound
        self.base = base
        self.fade_from: Optional[float] = None
        self.fade_s = 0.0


class SoundDevice:
    def __init__(self, settings: SoundSettings, root: Path, *, muted: bool = False):
        self.settings = settings
        self.ready = False
        self.durations: Dict[str, float] = {}
        self._pools: Dict[str, List] = {}
        self._turn: Dict[str, int] = {}
        self._playing: Dict[int, _Playing] = {}
        rl.init_audio_device()
        if not rl.is_audio_device_ready():
            print("  sound: no audio device -- the game runs silent")
            return
        self.ready = True
        rl.set_master_volume(0.0 if muted else settings.master)
        for f in settings.files():
            path = root / f
            first = rl.load_sound(str(path))
            self._pools[f] = [first] + [rl.load_sound_alias(first)
                                        for _ in range(ALIASES_PER_FILE - 1)]
            self._turn[f] = 0
            self.durations[f] = sample_seconds(path)

    def execute(self, commands: Sequence[Command], now: float) -> None:
        if not self.ready:
            return
        for c in commands:
            if isinstance(c, Start):
                pool = self._pools[c.file]
                snd = next((s for s in pool if not rl.is_sound_playing(s)), None)
                if snd is None:                 # all busy: take them in turn
                    snd = pool[self._turn[c.file] % len(pool)]
                    self._turn[c.file] += 1
                    rl.stop_sound(snd)
                    for k in [k for k, p in self._playing.items() if p.sound is snd]:
                        del self._playing[k]
                rl.set_sound_volume(snd, c.gain)
                rl.set_sound_pitch(snd, c.pitch)
                rl.set_sound_pan(snd, 0.5 - 0.5 * c.pan)
                rl.play_sound(snd)
                self._playing[c.voice] = _Playing(snd, c.gain)
            elif isinstance(c, Stop):
                p = self._playing.get(c.voice)
                if p is None:
                    continue
                if c.fade_ms <= 0.0:
                    rl.stop_sound(p.sound)
                    del self._playing[c.voice]
                elif p.fade_from is None:
                    p.fade_from = now
                    p.fade_s = c.fade_ms / 1000.0

    def update(self, now: float) -> None:
        """Advance the fades; forget what finished. Once per frame."""
        if not self.ready:
            return
        for k in list(self._playing):
            p = self._playing[k]
            if not rl.is_sound_playing(p.sound):
                del self._playing[k]
            elif p.fade_from is not None:
                g = 1.0 - (now - p.fade_from) / p.fade_s
                if g <= 0.0:
                    rl.stop_sound(p.sound)
                    del self._playing[k]
                else:
                    rl.set_sound_volume(p.sound, p.base * g)

    def playing(self) -> int:
        return len(self._playing)

    def close(self) -> None:
        if self.ready:
            for pool in self._pools.values():
                for alias in pool[1:]:
                    rl.unload_sound_alias(alias)
                rl.unload_sound(pool[0])
            rl.close_audio_device()
            self.ready = False
