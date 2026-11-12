"""Sound clip preparation: the pure half (numpy only, no raylib).

A generated clip (docs/sound/prototypes/) is not a game sample yet: it has
silence before the sound, it may hold several shots, and its level is
whatever the generator gave. This module turns a decoded clip into game
samples:

  onsets(x, sr)        where each transient (shot) starts
  split_shots(x, sr)   one sample per shot that keeps a usable tail
  trim(x, sr)          leading silence and dead tail removed, tail faded
  squeeze_silence(x)   every silence shortened (a reload spread too long)
  normalize(x)         peak to a fixed level
  layer(body, head)    a dry crack laid over a body-and-tail sample
  write_wav / read_wav 16-bit PCM through the stdlib

and it holds the choke law the playback uses (`choke_gain`): every shot
plays its whole sample, a new shot fades the previous one out over a few
milliseconds, and the last shot is never choked so its tail rings out.

Arrays are float32, shape (n,) mono or (n, channels), in [-1, 1].
The shell that decodes MP3 and plays the result is tools/sound_audition.py.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

PEAK = 0.89            # normalize target (about -1 dBFS)
PRE_ROLL_MS = 1.0      # kept before an onset so the attack is not clipped
MIN_SHOT_MS = 250.0    # an onset sooner than this after another is its echo
MIN_GAP_MS = 40.0      # two onsets closer than this are one transient


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.abs(x) if x.ndim == 1 else np.abs(x).max(axis=1)


def envelope(x: np.ndarray, sr: int, hop_ms: float = 1.0) -> np.ndarray:
    """Peak envelope, one value per hop."""
    m = _mono(x)
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    n = len(m) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return m[: n * hop].reshape(n, hop).max(axis=1)


def onsets(x: np.ndarray, sr: int, *, level: float = 0.3, rise: float = 3.0) -> list[int]:
    """Sample index of each transient: the envelope reaches `level` of the
    clip's peak AND stands `rise` times above the 20 ms before it (so a loud
    reverb tail is not an onset). Each index is backed off by the pre-roll."""
    env = envelope(x, sr)
    if len(env) == 0 or float(env.max()) <= 0.0:
        return []
    pk = float(env.max())
    hop = max(1, int(round(sr / 1000.0)))
    gap = int(MIN_GAP_MS)
    pre = int(round(sr * PRE_ROLL_MS / 1000.0))
    out: list[int] = []
    last = -gap
    for i, v in enumerate(env):
        if v < level * pk or i - last < gap:
            continue
        before = env[max(0, i - 22): max(0, i - 2)]
        floor = float(before.mean()) if len(before) else 0.0
        if v >= rise * floor:
            # walk back to where this rise began (10 % of its level)
            j = i
            while j > 0 and env[j - 1] > 0.1 * v and i - j < 10:
                j -= 1
            out.append(max(0, j * hop - pre))
            last = i
    return out


def fade_out(x: np.ndarray, sr: int, ms: float) -> np.ndarray:
    x = np.array(x, dtype=np.float32, copy=True)
    n = min(len(x), int(round(sr * ms / 1000.0)))
    if n > 0:
        ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
        x[-n:] *= ramp if x.ndim == 1 else ramp[:, None]
    return x


def trim(x: np.ndarray, sr: int, *, tail_db: float = -60.0, fade_ms: float = 15.0) -> np.ndarray:
    """From the first onset to the last moment above `tail_db` (re the peak),
    the end faded. A clip with no onset comes back whole."""
    x = np.asarray(x, dtype=np.float32)
    ons = onsets(x, sr)
    start = ons[0] if ons else 0
    m = _mono(x)
    pk = float(m.max()) if len(m) else 0.0
    if pk <= 0.0:
        return x[start:]
    alive = np.nonzero(m > pk * 10.0 ** (tail_db / 20.0))[0]
    end = int(alive[-1]) + 1 if len(alive) else len(x)
    end = max(end, start + 1)
    return fade_out(x[start:end], sr, fade_ms)


def split_shots(x: np.ndarray, sr: int) -> list[np.ndarray]:
    """One trimmed sample per shot: the stretch from an onset to the next
    onset (or the clip's end). An onset less than MIN_SHOT_MS after the one
    before it starts no sample of its own -- it is an early echo (or a shot
    that cannot be separated with a tail) and stays inside that sample, so
    nothing of the clip is dropped."""
    x = np.asarray(x, dtype=np.float32)
    ons: list[int] = []
    for o in onsets(x, sr):
        if not ons or (o - ons[-1]) * 1000.0 / sr >= MIN_SHOT_MS:
            ons.append(o)
    shots: list[np.ndarray] = []
    for k, a in enumerate(ons):
        b = ons[k + 1] if k + 1 < len(ons) else len(x)
        seg = x[a:b]
        cut = k + 1 < len(ons)
        seg = fade_out(seg, sr, 5.0) if cut else trim(seg, sr)
        shots.append(seg)
    return shots


def squeeze_silence(x: np.ndarray, sr: int, *, floor_db: float = -50.0,
                    gap_ms: float = 60.0, pad_ms: float = 30.0) -> np.ndarray:
    """The clip with every silence shortened to `gap_ms`: the stretches
    louder than `floor_db` (re the peak), each kept with `pad_ms` around it
    and faded at both ends, joined in order. For a sound made of several
    separate actions (a reload) that the generator spread over more time
    than the game gives it. Nothing audible is dropped or reordered."""
    x = np.asarray(x, dtype=np.float32)
    env = envelope(x, sr)
    if len(env) == 0 or float(env.max()) <= 0.0:
        return x
    hop = max(1, int(round(sr / 1000.0)))
    loud = env > float(env.max()) * 10.0 ** (floor_db / 20.0)
    pad = int(round(pad_ms))
    spans: list[list[int]] = []
    for i in np.nonzero(loud)[0]:
        a, b = max(0, int(i) - pad), min(len(env), int(i) + pad + 1)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    gap = np.zeros((int(round(sr * gap_ms / 1000.0)),) + x.shape[1:], dtype=np.float32)
    fade = min(pad_ms / 2.0, 10.0)
    parts: list[np.ndarray] = []
    for k, (a, b) in enumerate(spans):
        seg = fade_out(x[a * hop:b * hop], sr, fade)
        seg = fade_out(seg[::-1], sr, fade)[::-1]          # and a fade in
        if k:
            parts.append(gap)
        parts.append(seg)
    return np.concatenate(parts) if parts else x


def normalize(x: np.ndarray, peak: float = PEAK) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    pk = float(np.abs(x).max()) if x.size else 0.0
    return x if pk <= 0.0 else (x * (peak / pk)).astype(np.float32)


def _match_channels(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.ndim == b.ndim:
        return a, b
    if a.ndim == 1:
        a = np.repeat(a[:, None], b.shape[1], axis=1)
    else:
        b = np.repeat(b[:, None], a.shape[1], axis=1)
    return a, b


def layer(body: np.ndarray, head: np.ndarray, sr: int, *, head_ms: float = 90.0,
          head_gain: float = 0.8) -> np.ndarray:
    """The first `head_ms` of `head` (its crack, faded out over its second
    half) added onto `body`; both must already start at their onsets."""
    body, head = _match_channels(np.asarray(body, np.float32), np.asarray(head, np.float32))
    n = min(len(head), int(round(sr * head_ms / 1000.0)))
    piece = fade_out(head[:n], sr, head_ms / 2.0) * head_gain
    out = np.array(body, copy=True)
    if len(out) < n:
        pad = np.zeros((n - len(out),) + out.shape[1:], dtype=np.float32)
        out = np.concatenate([out, pad])
    out[:n] += piece
    return normalize(out)


def choke_gain(age_ms: float, fade_ms: float) -> float:
    """Gain of a voice `age_ms` after a newer shot choked it: linear to 0
    over `fade_ms`. 0 means the voice may be stopped."""
    if fade_ms <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - age_ms / fade_ms)))


def describe(x: np.ndarray, sr: int) -> dict:
    """The numbers a person wants about a clip: length, onsets, tail."""
    m = _mono(x)
    pk = float(m.max()) if len(m) else 0.0
    ons = onsets(x, sr)
    tail_ms = 0.0
    if pk > 0.0 and ons:
        alive = np.nonzero(m > pk * 10.0 ** (-40.0 / 20.0))[0]
        tail_ms = (int(alive[-1]) - ons[-1]) * 1000.0 / sr
    return {
        "duration_ms": len(m) * 1000.0 / sr if sr else 0.0,
        "onsets_ms": [o * 1000.0 / sr for o in ons],
        "tail_ms_to_minus_40db": tail_ms,
        "channels": 1 if np.asarray(x).ndim == 1 else np.asarray(x).shape[1],
        "sample_rate": sr,
    }


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(x * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1 if pcm.ndim == 1 else pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        ch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit WAV is read here")
        pcm = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32767.0
    return (pcm if ch == 1 else pcm.reshape(-1, ch)), sr
