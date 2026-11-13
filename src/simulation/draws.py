"""The game's integer DRAWS (#29 P3e, P3b) -- the ONE module every random
number the sim draws goes through.

KEYED DRAWS (#29 P3b, design §4.7). Every draw is one block of
Philox-4x32-10 (Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As
Easy as 1, 2, 3", SC'11, DOI 10.1145/2063384.2063405; the vendored twin
``simulation/philox32.py``, its C++ header ``cpp/src/philox32.h``) on the
kit's layout:

* key     = ``(seed lo, seed hi)`` -- the sim's seed (a match's episode
  seed, ``Simulation(seed=)``), 64 bits, and nothing else;
* counter = ``(tick, draw_index, salt, unit_or_site_id)`` -- ``tick`` the
  sim's tick (plus the legacy WEGO round base, :attr:`MatchDraws.tick_base`),
  ``salt`` the site's member of :class:`simulation.rng_salts.Salt`,
  ``unit_or_site_id`` the shooter / striker id (``-1``, "no unit", is the
  word ``2**32 - 1``) or the tile's linear index, ``draw_index`` LOCAL to
  the site: the k-th draw that (salt, id) takes this tick
  (:meth:`MatchDraws.next`), an in-flight round's ``(age << 16) | ordinal``
  (:func:`round_index`), or a FieldEdit's edit key.

A draw is therefore a pure function of (seed, tick, site, id, index): no
stream position, nothing another unit, site or env draws can move it. The
word used is the block's first (``out[0]``); one block per draw.

The maps from a word to an outcome (``u`` the word in ``[0, 2**32)``; P3e's,
unchanged):

* :func:`cone_offset_q16` -- ``((u * (2c + 1)) >> 32) - c``: an integer in
  ``[-c, c]`` (Q16.16 radians), each value with probability ``1 / (2c + 1)``
  to within ``2**-32`` (the shot and beam cones);
* :func:`roll_q16` -- ``(u >> 16) < p_q``: True with probability exactly
  ``p_q / 65536`` (the exposure and crit rolls; ``p_q`` a Q16 probability);
* :func:`noise_mult_q16` -- ``lo + ((u * (ONE - lo + 1)) >> 32)``: an integer
  in ``[lo, ONE]`` (the FieldEdit noise multiplier, Q16).

``sim.rng`` (a numpy Generator) is no draw source of this module: a helper
handed one refuses it by name.
"""
from __future__ import annotations

import operator

from simulation.philox32 import philox32_4x32_10
from simulation.rng_salts import Salt

ONE = 1 << 16
_WORD = 1 << 32
_M32 = _WORD - 1
#: An in-flight round's draw index packs its age (ticks since it was fired)
#: over its ordinal (the k-th round its shooter fired that tick), 16 bits each.
ROUND_FIELD_BITS = 16
_ROUND_FIELD_MAX = (1 << ROUND_FIELD_BITS) - 1


def _seed64(seed) -> int:
    if isinstance(seed, bool):
        raise ValueError(f"draws: the seed must be an integer, got {seed!r}")
    s = operator.index(seed)
    if not 0 <= s < 1 << 64:
        raise ValueError(f"draws: the seed must be in [0, 2**64), got {s}")
    return s


def _id_word(ident) -> int:
    """A unit / site id as a counter word: ``-1`` (no unit) is ``2**32 - 1``;
    anything else must be in ``[0, 2**32 - 1)``."""
    i = int(ident)
    if i == -1:
        return _M32
    if not 0 <= i < _M32:
        raise ValueError(f"draws: id {i} is outside [-1, 2**32 - 1)")
    return i


def round_index(age: int, ordinal: int) -> int:
    """The draw index of an in-flight round ``age`` ticks after it was fired
    as its shooter's ``ordinal``-th round of that tick: ``(age << 16) |
    ordinal``. Unique per round per tick (the tick and the age give the
    firing tick, the shooter's id and the ordinal the round); refused past
    16 bits each."""
    a, k = int(age), int(ordinal)
    if not (0 <= a <= _ROUND_FIELD_MAX and 0 <= k <= _ROUND_FIELD_MAX):
        raise ValueError(f"draws.round_index: age {a} / ordinal {k} outside "
                         f"[0, {_ROUND_FIELD_MAX}]")
    return (a << ROUND_FIELD_BITS) | k


class Draw:
    """ONE named draw: (tick, index, salt, id) under its owner's key. Its
    :meth:`word` is the Philox block's first word."""

    __slots__ = ("owner", "tick", "index", "salt", "ident")

    def __init__(self, owner, tick, index, salt, ident):
        self.owner = owner
        self.tick = int(tick)
        self.index = int(index)
        self.salt = int(salt)
        self.ident = _id_word(ident)

    def counter(self) -> tuple:
        return (self.tick & _M32, self.index, self.salt, self.ident)

    def word(self) -> int:
        self.owner.drawn += 1
        return philox32_4x32_10(self.counter(), self.owner.key)[0]


class MatchDraws:
    """The keyed draws of ONE env (one ``Simulation``): the key is its seed,
    the tick its clock's (``clock()`` + :attr:`tick_base`). Holds no stream:
    only the per-tick, per-(salt, id) count :meth:`next` hands out, cleared
    when the tick changes."""

    def __init__(self, seed: int, clock=None):
        s = _seed64(seed)
        self.seed = s
        self.key = (s & _M32, s >> 32)
        self._clock = clock
        self._fixed_tick = 0
        #: Legacy TwoPhaseWEGO rewinds ``sim.tick`` every round; the round
        #: end adds the round's length here so its draws do not repeat.
        self.tick_base = 0
        self._count_tick = None
        self._counts: dict = {}
        #: How many words were drawn (a diagnostic: "this draws nothing" is
        #: ``drawn`` unchanged). Never synced, never hashed.
        self.drawn = 0

    @property
    def tick(self) -> int:
        t = self._clock() if self._clock is not None else self._fixed_tick
        return int(t) + self.tick_base

    def set_tick(self, tick: int) -> None:
        """The tick of a clock-less instance (tests, tools)."""
        if self._clock is not None:
            raise ValueError("MatchDraws.set_tick: this instance reads its "
                             "sim's clock")
        self._fixed_tick = int(tick)

    def at(self, salt, ident, index) -> Draw:
        """The draw ``(this tick, index, salt, ident)``."""
        return Draw(self, self.tick, index, Salt(salt), ident)

    def next(self, salt, ident) -> Draw:
        """The NEXT draw of site (salt, ident) this tick: index 0, 1, ... in
        the order that site asks (local to the site)."""
        t = self.tick
        if t != self._count_tick:
            self._count_tick = t
            self._counts.clear()
        key = (int(salt), int(ident))
        k = self._counts.get(key, 0)
        self._counts[key] = k + 1
        return Draw(self, t, k, Salt(salt), ident)


class RefusingRng:
    """The ``sim.rng`` a match carries (#29 P3b, design §4.7): ANY use raises,
    naming it -- the match path draws only through :class:`MatchDraws`
    (``sim.draws``), so a new ``rng.uniform`` on it fails in the ordinary
    suite instead of drawing from a stream no device can reproduce."""

    __slots__ = ()

    def __getattr__(self, name):
        raise RuntimeError(
            f"sim.rng.{name}: a match draws only through simulation.draws "
            f"(sim.draws, keyed Philox; #29 P3b) -- name the site's Salt and "
            f"id instead of drawing from sim.rng")

    def __repr__(self):
        return "RefusingRng()"


def _word(d) -> int:
    if not isinstance(d, Draw):
        raise TypeError(
            f"draws: a draw is a keyed simulation.draws.Draw (sim.draws.next "
            f"/ .at), got {type(d).__name__} -- the numpy stream is no draw "
            f"source since #29 P3b")
    return d.word()


def cone_offset_q16(d: Draw, cone_q: int) -> int:
    """A cone offset in ``[-cone_q, cone_q]`` Q16.16 radians, one draw."""
    c = int(cone_q)
    if c < 0:
        raise ValueError(f"cone_offset_q16: cone {c} < 0")
    return ((_word(d) * (2 * c + 1)) >> 32) - c


def roll_q16(d: Draw, p_q: int) -> bool:
    """One Bernoulli roll: True with probability ``p_q / 65536``."""
    return (_word(d) >> 16) < int(p_q)


def noise_mult_q16(d: Draw, lo_q: int) -> int:
    """A noise multiplier in ``[lo_q, ONE]`` (Q16), one draw."""
    lo = int(lo_q)
    if not 0 <= lo <= ONE:
        raise ValueError(f"noise_mult_q16: low {lo} outside [0, {ONE}]")
    return lo + ((_word(d) * (ONE - lo + 1)) >> 32)


__all__ = ["MatchDraws", "Draw", "RefusingRng", "round_index",
           "ROUND_FIELD_BITS", "cone_offset_q16", "roll_q16",
           "noise_mult_q16", "ONE", "Salt"]
