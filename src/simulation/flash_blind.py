"""Flash blinding (gameplay pack v1, G3p; ``docs/gameplay_pack_v1_design_2026-11-10.md``
§3 G3p): the flashbang's effect on the units that see it.

A payload row with ``blind_radius > 0`` (``[payloads.*] blind_radius`` /
``blind_seconds`` / ``blind_away_frac``, the door ``weapons._blind_columns``)
BLINDS at detonation: every LIVING unit of ANY team (the thrower's own
included -- Erik's ruling: looking away is what a team does on a called
throw) within the radius of the detonation tile and with a line of sight to
it (``GameMap.has_los``: a wall protects fully, at any distance and facing)
gets the BLINDED status (:data:`simulation.status.BLINDED`: no aim, its fire
falls to the snap cone) for::

    ticks = floor(blind_ticks * f * g)

``blind_ticks`` is ``blind_seconds`` quantized once at the payload door.
``g`` falls linearly with distance, from 1 at the flash to 0 at the radius.
``f`` is 1 while the unit faces within +-60 degrees of the direction to the
flash and falls linearly to ``blind_away_frac`` when it faces straight away.
Zero ticks applies nothing.

PURE INTEGER (design §2 rules 1 and 2): positions are centre tiles, the
distance is ``isqrt`` of the squared tile distance in Q16, the direction to
the flash is the kit's ``atan2_q16`` and the unit's facing is its Q16
radians, both in the angle currency of ``turning.py`` (``wrap_q16``). No
float, no draw. ``f`` and ``g`` are Q16 and each step is a floor of a
monotone expression, so the duration is monotone non-increasing in the
distance and in the angle away from the flash.

A unit already blinded for longer keeps its longer blinding (BLINDED is a
REFRESH status, and a second, weaker flash must not shorten the first).

Called by :func:`simulation.payloads.execute_payload`; inside an attack slot
that records effects the whole detonation is the recorded world effect
(``SimultaneousEffects.world``), so the statuses land at its turn.
"""
from __future__ import annotations

import math

from simulation import unit_fixed
from simulation.status import BLINDED, apply_status
from simulation.turning import PI_Q16, wrap_q16

ONE = 1 << 16
#: A unit facing within this angle of the flash gets the whole blinding: 60
#: degrees, ``round(pi / 3 * 2^16)``, the kit's pi over three.
FRONT_Q16 = PI_Q16 // 3


def distance_factor_q16(d2_tiles: int, radius_q: int) -> int:
    """``g`` in Q16: 1 at the flash, linear down to 0 at ``radius_q`` (Q16
    tiles); 0 at or beyond it. ``d2_tiles`` is the squared integer tile
    distance."""
    dist_q = math.isqrt(int(d2_tiles) << 32)          # Q16 tiles
    if dist_q >= radius_q:
        return 0
    return ONE - (dist_q * ONE) // radius_q


def facing_factor_q16(dx: int, dy: int, facing_q: int, away_q: int) -> int:
    """``f`` in Q16 for a unit whose facing is ``facing_q`` (Q16 radians,
    math-style, y up) and for which the flash lies at the integer tile
    offset ``(dx, dy)`` (world, y down) from the unit. 1 within
    :data:`FRONT_Q16` of the flash's direction, linear to ``away_q`` at half
    a turn. A flash on the unit's own tile names no direction: factor 1."""
    if dx == 0 and dy == 0:
        return ONE
    to_flash = unit_fixed.atan2_q16(-int(dy) << 16, int(dx) << 16)
    diff = abs(wrap_q16(to_flash - int(facing_q)))
    if diff <= FRONT_Q16:
        return ONE
    return ONE - ((ONE - away_q) * (diff - FRONT_Q16)) // (PI_Q16 - FRONT_Q16)


def blind_ticks_for(payload, d2_tiles: int, dx: int, dy: int,
                    facing_q: int) -> int:
    """The BLINDED duration (integer ticks, possibly 0) a unit at squared
    tile distance ``d2_tiles`` and offset ``(dx, dy)`` from the flash,
    facing ``facing_q``, gets from ``payload``."""
    g = distance_factor_q16(d2_tiles, payload.blind_radius_q)
    if g == 0:
        return 0
    f = facing_factor_q16(dx, dy, facing_q, payload.blind_away_q)
    return (payload.blind_ticks * f * g) >> 32


def apply_flash_blind(gmap, units, fy: int, fx: int, payload,
                      owner_id: int = -1) -> None:
    """Blind every living unit that sees the detonation at tile ``(fy, fx)``
    (the law in the module doc). ``owner_id`` (-1 = nobody) is the status's
    source. Units in list order; each reads only the map and its own state."""
    if payload.blind_radius_q <= 0:
        return
    r_tiles = (payload.blind_radius_q >> 16) + 1       # integer bound
    r2_max = r_tiles * r_tiles
    source = None if owner_id is None or owner_id < 0 else int(owner_id)
    for u in units:
        if not u.alive:
            continue
        ux = u.center_tile_x()
        uy = u.center_tile_y()
        dx = int(fx) - ux
        dy = int(fy) - uy
        d2 = dx * dx + dy * dy
        if d2 > r2_max:
            continue
        if not gmap.has_los(uy, ux, int(fy), int(fx)):
            continue
        ticks = blind_ticks_for(payload, d2, dx, dy,
                                unit_fixed.quantize_scalar(u.facing))
        if ticks < 1:
            continue
        for st in getattr(u, "statuses", None) or ():
            if st.kind == BLINDED and st.remaining_ticks >= ticks:
                break                                  # already blinded longer
        else:
            apply_status(u, BLINDED, magnitude=0, duration_ticks=ticks,
                         source_id=source)


__all__ = ["apply_flash_blind", "blind_ticks_for", "distance_factor_q16",
           "facing_factor_q16", "FRONT_Q16"]
