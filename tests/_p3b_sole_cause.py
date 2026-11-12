"""#29 P3b's SOLE-CAUSE instrument (a pytest plugin, never collected).

P3b changes WHICH random numbers the sim draws (keyed Philox in place of one
numpy stream) and nothing else. This plugin proves it: it replaces every
keyed draw's word by the word P3e's code drew at that point -- the high 32
bits of the next raw step of ``np.random.default_rng(seed)``, one generator
per ``MatchDraws`` (per sim), consumed in call order -- so the branch, run
through it, must reproduce main's numbers bit for bit:

    <breach py> -m pytest -p _p3b_sole_cause <the pinned tests at main's pins>

Green there (on the branch, with main's pins) = the draws are the only
behavioural change. A sim the shim cannot map (an unseeded one) fails loudly.
"""
from __future__ import annotations

import numpy as np

from simulation import draws as _draws

_streams: dict = {}


def _legacy_word(self) -> int:
    owner = self.owner
    gen = _streams.get(id(owner))
    if gen is None or gen[0] is not owner:
        gen = (owner, np.random.default_rng(owner.seed))
        _streams[id(owner)] = gen
    return int(gen[1].bit_generator.random_raw()) >> 32


def pytest_configure(config):
    _draws.Draw.word = _legacy_word
