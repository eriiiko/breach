"""The sound clip preparation (tools/sound_clips.py): the pure half of the
audition tool. Headless, numpy only, synthetic clips.

Properties held (each names the change that must break it):
  - a prepared sample starts AT its transient: leading silence left in
    would delay every shot the game plays;
  - nothing of a clip is dropped: an early echo stays inside its shot, and
    separate shots become separate samples;
  - the choke law is a monotone fade that reaches exactly 0, and a voice
    that was never choked is never faded.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import pytest

import sound_clips as sc

SR = 44100


def _shot(ms: float, decay_ms: float = 120.0, seed: int = 0) -> np.ndarray:
    n = int(SR * ms / 1000.0)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * np.exp(-t / (decay_ms / 1000.0))).astype(np.float32) * 0.5


def _clip(parts: list[tuple[float, np.ndarray]], total_ms: float) -> np.ndarray:
    x = np.zeros(int(SR * total_ms / 1000.0), dtype=np.float32)
    for at_ms, s in parts:
        a = int(SR * at_ms / 1000.0)
        x[a:a + len(s)] += s[: len(x) - a]
    return x


@pytest.mark.parametrize("lead_ms", [0.0, 37.0, 170.0, 1542.0])
def test_a_trimmed_sample_starts_at_its_transient(lead_ms):
    x = _clip([(lead_ms, _shot(800.0))], lead_ms + 1500.0)
    y = sc.trim(x, SR)
    loud = np.nonzero(np.abs(y) > 0.1 * np.abs(y).max())[0][0]
    assert loud * 1000.0 / SR <= 3.0          # within the pre-roll, whatever the lead
    assert len(y) < len(x) or lead_ms == 0.0


def test_separate_shots_become_separate_samples_each_from_its_crack():
    x = _clip([(100.0, _shot(500.0, seed=1)), (900.0, _shot(500.0, seed=2)),
               (1700.0, _shot(900.0, seed=3))], 3000.0)
    shots = sc.split_shots(x, SR)
    assert len(shots) == 3
    for s in shots:
        loud = np.nonzero(np.abs(s) > 0.1 * np.abs(s).max())[0][0]
        assert loud * 1000.0 / SR <= 3.0


def test_an_early_echo_stays_inside_its_shot():
    echo = _shot(600.0, seed=5) * 0.9
    x = _clip([(170.0, _shot(60.0, decay_ms=15.0, seed=4)), (261.0, echo)], 2000.0)
    shots = sc.split_shots(x, SR)
    assert len(shots) == 1
    # the sample spans both transients: it is longer than the gap between them
    assert len(shots[0]) * 1000.0 / SR > 300.0
    first = np.nonzero(np.abs(shots[0]) > 0.1 * np.abs(shots[0]).max())[0][0]
    assert first * 1000.0 / SR <= 3.0


def test_normalize_sets_the_peak_and_keeps_silence():
    x = _shot(200.0) * 0.1
    assert np.abs(sc.normalize(x)).max() == pytest.approx(sc.PEAK, rel=1e-5)
    z = np.zeros(100, dtype=np.float32)
    assert not sc.normalize(z).any()


def test_layer_keeps_the_body_after_the_head_window():
    body = sc.normalize(_shot(1000.0, decay_ms=300.0, seed=6))
    head = sc.normalize(_shot(300.0, decay_ms=10.0, seed=7))
    mix = sc.layer(body, head, SR, head_ms=90.0)
    a = int(SR * 0.1)
    # past the head window the mix is the body times one constant gain
    ratio = mix[a:a + 2000] / body[a:a + 2000]
    assert np.allclose(ratio, ratio[0], rtol=1e-3)
    assert np.abs(mix).max() == pytest.approx(sc.PEAK, rel=1e-5)


def test_layer_mixes_mono_with_stereo():
    body = np.stack([_shot(500.0, seed=8), _shot(500.0, seed=9)], axis=1)
    assert sc.layer(body, _shot(200.0, seed=10), SR).shape[1] == 2


def test_choke_gain_is_a_monotone_fade_to_exactly_zero():
    fade = 12.0
    gains = [sc.choke_gain(t, fade) for t in np.linspace(0.0, 30.0, 61)]
    assert gains[0] == 1.0
    assert all(b <= a for a, b in zip(gains, gains[1:]))
    assert all(g == 0.0 for g, t in zip(gains, np.linspace(0.0, 30.0, 61)) if t >= fade)
    assert sc.choke_gain(0.0, 0.0) == 0.0     # no fade = a hard cut


def test_wav_round_trip(tmp_path):
    x = np.stack([_shot(100.0, seed=11), _shot(100.0, seed=12)], axis=1)
    sc.write_wav(tmp_path / "a.wav", x, SR)
    y, sr = sc.read_wav(tmp_path / "a.wav")
    assert sr == SR and y.shape == x.shape
    assert np.abs(y - np.clip(x, -1, 1)).max() < 1.0 / 16000.0
