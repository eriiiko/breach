"""The turn-rate cap (gameplay pack v1, G1p; ``docs/gameplay_pack_v1_design_2026-11-10.md`` §3).

``Simulation.set_aim`` names the facing a unit WANTS; each tick the unit's
facing turns toward it by at most its turn rate, the shorter way round, and
reaches it exactly. This module is that law, in PURE INTEGER angle arithmetic
in the trig kit's own currency: Q16.16 RADIANS, the range ``atan2_q16``
returns (``(-pi, pi]`` as ``[-PI_Q16, PI_Q16]``, ``fixed_point.h``). The
angle a unit wants is ``atan2_q16`` of its aim vector (through
``unit_fixed.atan2_q16``, the kit's binding); nothing here calls a float
transcendental, and the only float is the door's quantization of the config
dial, once, at sim construction.

The wrap period is ``TWO_PI_Q16 = 2 * PI_Q16``, so the kit's two ends of the
range, ``-PI_Q16`` and ``+PI_Q16``, name the same angle (as atan2's do): the
period is 0.8 LSB short of the true 2 pi in Q16, far below the kit's own
9e-6 rad accuracy floor.

A cap of half a turn or more per tick (``>= PI_Q16``) reaches every aim in
one tick: the law is then exactly the pre-G1p instant facing (the gate that
proves the cap is the only thing moving the pinned matches,
``tests/test_turn_rate.py``).

Settings: ``[marine] turn_rate_deg_per_s`` (PLACEHOLDER, Erik tunes), read by
:func:`turn_rate_q16` -- a refusing door. It is a marine row today and a
per-unit stat later (dexterity, armour weight), so the sim reads it per unit
(``Simulation.unit_turn_rate_q16``).
"""
from __future__ import annotations

import math

from simulation.unit_fixed import quantize_scalar

#: quantize(pi) -- the kit's pinned ``atan2_q16(0, x < 0)`` (fixed_point.h).
PI_Q16 = 205887
#: The wrap period: both range ends name the same angle (see the module doc).
TWO_PI_Q16 = 2 * PI_Q16


def wrap_q16(d: int) -> int:
    """An angle difference folded into ``(-PI_Q16, PI_Q16]`` -- the SHORTER
    way round; an exact half turn goes positive (counter-clockwise), so the
    choice is deterministic."""
    d = int(d) % TWO_PI_Q16          # Python: [0, TWO_PI_Q16)
    return d - TWO_PI_Q16 if d > PI_Q16 else d


def normalize_q16(a: int) -> int:
    """An angle folded into ``[-PI_Q16, PI_Q16]`` (the atan2 range)."""
    return wrap_q16(a)


def turn_toward_q16(cur_q: int, target_q: int, cap_q: int) -> int:
    """The facing after one tick: ``cur_q`` turned toward ``target_q`` by at
    most ``cap_q`` (all Q16.16 radians), the shorter way round. Within the
    cap it returns ``target_q`` ITSELF (no overshoot, no residual); beyond it
    it steps exactly ``cap_q``. ``cap_q`` must be > 0."""
    d = wrap_q16(int(target_q) - int(cur_q))
    if -cap_q <= d <= cap_q:
        return int(target_q)
    return normalize_q16(int(cur_q) + (cap_q if d > 0 else -cap_q))


def turn_rate_q16(cfg) -> int:
    """``[marine] turn_rate_deg_per_s`` of a loaded config (``CFG``) as the
    per-TICK cap in Q16.16 radians: ``quantize(radians(rate) / tps)``,
    quantized once. Refused (ValueError naming the key) when the section or
    key is missing, not a number, not > 0, or so small it quantizes to 0."""
    section = getattr(cfg, "marine", None)
    if section is None or not hasattr(section, "turn_rate_deg_per_s"):
        raise ValueError("[marine] is missing 'turn_rate_deg_per_s'")
    v = section.turn_rate_deg_per_s
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v > 0:
        raise ValueError(f"[marine] turn_rate_deg_per_s must be a number > 0, "
                         f"got {v!r}")
    tps = int(cfg.clock.ticks_per_second)
    cap = quantize_scalar(math.radians(float(v)) / tps)
    if cap < 1:
        raise ValueError(f"[marine] turn_rate_deg_per_s = {v!r} is below one "
                         f"Q16 radian count per tick at {tps} tps")
    return cap


__all__ = ["PI_Q16", "TWO_PI_Q16", "wrap_q16", "normalize_q16",
           "turn_toward_q16", "turn_rate_q16"]
