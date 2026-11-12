"""Sound audition: hear generated clips the way the game would play them.

    <breach-py> tools/sound_audition.py [--dir DIR] [--rate 12] [--prepare]

DIR (default docs/sound/prototypes/seed_audio) holds generated clips
(.mp3 / .wav). Every clip whose name starts with `carbine_` and does not
contain `burst` or `reload` is a SHOT source: it is cut into one sample per
shot (tools/sound_clips.py: leading silence gone, normalized) and written to
DIR/prepared/ as WAV. `--prepare` does only that, prints what it measured
and exits (no window, no audio device).

The window plays the carbine by the CHOKE scheme: every shot plays its whole
sample from the crack; a new shot fades the previous one out over a few
milliseconds, so tails never stack; the last shot is never choked, so its
reverb rings out. Single shots and bursts are the same code.

  SPACE (hold)   automatic fire at the rate        F / left mouse   one shot
  1..9           which sample set (one per source, then `layered`, then `all`)
  C              choke on / off (off = every tail stacks: the comparison)
  V              variation on / off (random sample, pitch and volume +-3 %)
  UP / DOWN      choke fade, ms                    LEFT / RIGHT     rounds per second
  F1..F12        the other clips in DIR (a name with `loop` toggles a loop)

This is a listening tool, not the game's audio layer.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import sound_clips as sc  # noqa: E402

DEFAULT_DIR = ROOT / "docs" / "sound" / "prototypes" / "seed_audio"
VOICES_PER_SAMPLE = 4
VARIATION = 0.03


def decode(path: Path) -> tuple[np.ndarray, int]:
    """Any format raylib reads (mp3, wav, ogg, flac) -> float32 array, rate."""
    import pyray as rl
    w = rl.load_wave(str(path))
    if w.frameCount == 0:
        raise ValueError(f"{path}: could not be decoded")
    n = int(w.frameCount) * int(w.channels)
    ptr = rl.load_wave_samples(w)
    x = np.frombuffer(rl.ffi.buffer(ptr, n * 4), dtype=np.float32).copy()
    sr, ch = int(w.sampleRate), int(w.channels)
    rl.unload_wave_samples(ptr)
    rl.unload_wave(w)
    return (x if ch == 1 else x.reshape(-1, ch)), sr


def is_shot_source(p: Path) -> bool:
    n = p.stem.lower()
    return n.startswith("carbine_") and "burst" not in n and "reload" not in n


def prepare(folder: Path, report=print) -> dict[str, list[Path]]:
    """Cut every shot source into samples under folder/prepared/. Returns
    the sample sets by name: one per source, plus `layered` when a `dry`
    and another source both exist."""
    out = folder / "prepared"
    out.mkdir(exist_ok=True)
    sets: dict[str, list[Path]] = {}
    cut: dict[str, tuple[list[np.ndarray], int]] = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in (".mp3", ".wav", ".ogg", ".flac") or not is_shot_source(p):
            continue
        x, sr = decode(p)
        d = sc.describe(x, sr)
        shots = [sc.normalize(s) for s in sc.split_shots(x, sr)]
        if not shots:                      # no clear transient: keep it whole
            shots = [sc.normalize(sc.trim(x, sr))]
        name = p.stem.replace("carbine_", "")
        cut[name] = (shots, sr)
        paths = []
        for k, s in enumerate(shots):
            q = out / f"{p.stem}_{k + 1}.wav"
            sc.write_wav(q, s, sr)
            paths.append(q)
        sets[name] = paths
        report(f"{p.name}: {d['duration_ms']:.0f} ms, {d['sample_rate']} Hz, "
               f"{d['channels']} ch, onsets at "
               f"{[round(o) for o in d['onsets_ms']]} ms, last tail "
               f"{d['tail_ms_to_minus_40db']:.0f} ms to -40 dB -> {len(shots)} sample(s) "
               f"of {[round(len(s) * 1000 / sr) for s in shots]} ms")
    dry = next((n for n in cut if "dry" in n), None)
    body = next((n for n in cut if n != dry), None)
    if dry and body and cut[dry][1] == cut[body][1]:
        sr = cut[body][1]
        q = out / "carbine_layered_1.wav"
        sc.write_wav(q, sc.layer(cut[body][0][-1], cut[dry][0][0], sr), sr)
        sets["layered"] = [q]
        report(f"layered: the crack of '{dry}' over '{body}' -> {q.name}")
    return sets


class Voice:
    __slots__ = ("sound", "base", "choked_at")

    def __init__(self, sound, base: float):
        self.sound = sound
        self.base = base
        self.choked_at: float | None = None


class Carbine:
    """The choke scheme over a pool of raylib sounds."""

    def __init__(self, rl, sets: dict[str, list[Path]]):
        self.rl = rl
        self.pools: dict[Path, list] = {}
        for paths in sets.values():
            for p in paths:
                if p not in self.pools:
                    first = rl.load_sound(str(p))
                    self.pools[p] = [first] + [rl.load_sound_alias(first)
                                               for _ in range(VOICES_PER_SAMPLE - 1)]
        self.sets = dict(sets)
        self.sets["all"] = [p for ps in sets.values() for p in ps]
        self.names = list(self.sets)
        self.current = self.names[0]
        self.choke = True
        self.vary = True
        self.fade_ms = 12.0
        self.live: list[Voice] = []
        self.last_path = None
        self.rng = random.Random(1)
        self.shots = 0

    def fire(self, now: float) -> None:
        rl = self.rl
        paths = self.sets[self.current]
        p = paths[0]
        if self.vary and len(paths) > 1:
            p = self.rng.choice([q for q in paths if q != self.last_path] or paths)
        self.last_path = p
        pool = self.pools[p]
        snd = next((s for s in pool if not rl.is_sound_playing(s)), pool[self.shots % len(pool)])
        k = VARIATION if self.vary else 0.0
        base = 1.0 - self.rng.uniform(0.0, 2.0 * k)
        rl.set_sound_pitch(snd, 1.0 + self.rng.uniform(-k, k))
        rl.set_sound_volume(snd, base)
        if self.choke:
            for v in self.live:
                if v.choked_at is None:
                    v.choked_at = now
        self.live = [v for v in self.live if v.sound is not snd]
        rl.stop_sound(snd)
        rl.play_sound(snd)
        self.live.append(Voice(snd, base))
        self.shots += 1

    def update(self, now: float) -> None:
        rl = self.rl
        keep = []
        for v in self.live:
            if not rl.is_sound_playing(v.sound):
                continue
            if v.choked_at is not None:
                g = sc.choke_gain((now - v.choked_at) * 1000.0, self.fade_ms)
                if g <= 0.0:
                    rl.stop_sound(v.sound)
                    continue
                rl.set_sound_volume(v.sound, v.base * g)
            keep.append(v)
        self.live = keep


def run(folder: Path, rate: float) -> None:
    import pyray as rl
    rl.set_trace_log_level(rl.LOG_WARNING)
    sets = prepare(folder)
    if not sets:
        raise SystemExit(f"no carbine_* shot clip in {folder}")
    rl.init_window(760, 420, "Breach sound audition")
    rl.init_audio_device()
    rl.set_target_fps(240)
    gun = Carbine(rl, sets)
    others = [p for p in sorted(folder.iterdir())
              if p.suffix.lower() in (".mp3", ".wav", ".ogg", ".flac") and not is_shot_source(p)][:12]
    one_shots = {p: rl.load_sound(str(p)) for p in others if "loop" not in p.stem}
    loops = {p: rl.load_music_stream(str(p)) for p in others if "loop" in p.stem}
    looping: set[Path] = set()
    next_shot = 0.0
    held = False
    while not rl.window_should_close():
        now = rl.get_time()
        for i, name in enumerate(gun.names[:9]):
            if rl.is_key_pressed(rl.KEY_ONE + i):
                gun.current = name
        if rl.is_key_pressed(rl.KEY_C):
            gun.choke = not gun.choke
        if rl.is_key_pressed(rl.KEY_V):
            gun.vary = not gun.vary
        if rl.is_key_pressed(rl.KEY_UP):
            gun.fade_ms = min(80.0, gun.fade_ms + 2.0)
        if rl.is_key_pressed(rl.KEY_DOWN):
            gun.fade_ms = max(0.0, gun.fade_ms - 2.0)
        if rl.is_key_pressed(rl.KEY_RIGHT):
            rate = min(30.0, rate + 1.0)
        if rl.is_key_pressed(rl.KEY_LEFT):
            rate = max(1.0, rate - 1.0)
        if rl.is_key_pressed(rl.KEY_F) or rl.is_mouse_button_pressed(rl.MOUSE_BUTTON_LEFT):
            gun.fire(now)
        if rl.is_key_down(rl.KEY_SPACE):
            if not held:
                next_shot = now
            held = True
            while now >= next_shot:
                gun.fire(now)
                next_shot += 1.0 / rate
        else:
            held = False
        for i, p in enumerate(others):
            if rl.is_key_pressed(rl.KEY_F1 + i):
                if p in loops:
                    if p in looping:
                        rl.stop_music_stream(loops[p])
                        looping.discard(p)
                    else:
                        rl.play_music_stream(loops[p])
                        looping.add(p)
                else:
                    rl.play_sound(one_shots[p])
        for p in looping:
            rl.update_music_stream(loops[p])
        gun.update(now)

        rl.begin_drawing()
        rl.clear_background(rl.Color(18, 20, 24, 255))
        y = 14
        def line(text, color=rl.RAYWHITE):
            nonlocal y
            rl.draw_text(text, 16, y, 18, color)
            y += 24
        line("SPACE hold = automatic   F / click = one shot", rl.GRAY)
        for i, name in enumerate(gun.names[:9]):
            on = name == gun.current
            line(f"{'>' if on else ' '} {i + 1}  {name}  ({len(gun.sets[name])} sample(s))",
                 rl.YELLOW if on else rl.LIGHTGRAY)
        line(f"C  choke: {'ON' if gun.choke else 'OFF (tails stack)'}    "
             f"UP/DOWN fade {gun.fade_ms:.0f} ms")
        line(f"V  variation: {'ON' if gun.vary else 'OFF'}    LEFT/RIGHT rate {rate:.0f} / s"
             f"    shots {gun.shots}")
        for i, p in enumerate(others):
            line(f"F{i + 1}  {p.stem}{'  [looping]' if p in looping else ''}",
                 rl.GREEN if p in looping else rl.GRAY)
        rl.end_drawing()
    rl.close_audio_device()
    rl.close_window()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--rate", type=float, default=12.0, help="rounds per second (the carbine: 12)")
    ap.add_argument("--prepare", action="store_true", help="cut the samples, print, exit")
    a = ap.parse_args()
    if not a.dir.is_dir():
        raise SystemExit(f"no such folder: {a.dir}")
    if a.prepare:
        prepare(a.dir)
    else:
        run(a.dir, a.rate)


if __name__ == "__main__":
    main()
