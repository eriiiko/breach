"""The match path's integer DRAWS (#29 P3e) -- the ONE module every
match-path random number goes through.

Each helper takes exactly ONE raw word from the generator
(``rng.bit_generator.random_raw()``, one step of the bit generator, of which
the high 32 bits are used), where the float path made ONE ``rng.uniform``
call (also one step), in the same order -- so the number of draws per shot, per roll and per noisy
tile is what it was. The word is mapped to the outcome in integers: no float,
no division, no rejection loop (a fixed one word per draw, which is what a
counter-based generator hands out). P3b (Philox at the match draw sites)
swaps the generator here and nowhere else: each helper's ``rng`` argument
becomes the keyed stream.

The maps (``u`` the word in ``[0, 2**32)``):

* :func:`cone_offset_q16` -- ``((u * (2c + 1)) >> 32) - c``: an integer in
  ``[-c, c]`` (Q16.16 radians), each value with probability ``1 / (2c + 1)``
  to within ``2**-32`` (the shot cone; ``c = 0`` still draws, as
  ``uniform(-0, 0)`` did);
* :func:`roll_q16` -- ``(u >> 16) < p_q``: True with probability exactly
  ``p_q / 65536`` (the exposure and crit rolls; ``p_q`` a Q16 probability);
* :func:`noise_mult_q16` -- ``lo + ((u * (ONE - lo + 1)) >> 32)``: an integer
  in ``[lo, ONE]`` (the FieldEdit noise multiplier, Q16).
"""
from __future__ import annotations

ONE = 1 << 16
_WORD = 1 << 32


def word(rng) -> int:
    """ONE uniform 32-bit word from ``rng``: the high half of one raw step
    of its bit generator (no rejection, no buffering)."""
    return int(rng.bit_generator.random_raw()) >> 32


def cone_offset_q16(rng, cone_q: int) -> int:
    """A cone offset in ``[-cone_q, cone_q]`` Q16.16 radians, one draw."""
    c = int(cone_q)
    if c < 0:
        raise ValueError(f"cone_offset_q16: cone {c} < 0")
    return ((word(rng) * (2 * c + 1)) >> 32) - c


def roll_q16(rng, p_q: int) -> bool:
    """One Bernoulli roll: True with probability ``p_q / 65536``."""
    return (word(rng) >> 16) < int(p_q)


def noise_mult_q16(rng, lo_q: int) -> int:
    """A noise multiplier in ``[lo_q, ONE]`` (Q16), one draw."""
    lo = int(lo_q)
    if not 0 <= lo <= ONE:
        raise ValueError(f"noise_mult_q16: low {lo} outside [0, {ONE}]")
    return lo + ((word(rng) * (ONE - lo + 1)) >> 32)


__all__ = ["word", "cone_offset_q16", "roll_q16", "noise_mult_q16", "ONE"]
