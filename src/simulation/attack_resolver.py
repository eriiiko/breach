"""The attack-resolution layer — exposure-vs-cover + crit-vs-facing
(mechanics/06 §5, wired live by weapons W2).

Whether a :class:`~simulation.damage.DamagePacket` is emitted at all is
decided HERE, in front of the pipeline. The doctrine (DECIDED 2026-07-04):

    Physics decides what is possible; probability models what 2D cannot see.

- **The physical layer stays absolute** — walls block bullets, the spread
  cone and bodies in the way are geometry (the combat.py march). No roll ever
  hits what physics rules out; there is no dodge stat.
- **The exposure roll answers cover** (:func:`cover_exposure_at` +
  :func:`roll_exposure`): a top-down ray cannot see a marine crouched behind
  a crate, so when the march would enter a target footprint having just
  crossed a tile whose material carries ``cover_exposure < 1.0``, the shot
  connects with that probability — else it is absorbed by the cover tile
  (the caller deposits the round's wall damage there: suppressive fire chews
  the crate until it stops *being* cover). Cover is directional BY
  CONSTRUCTION: approach through a non-cover tile and no roll exists.
- **The crit roll answers facing** (:func:`arc_multiplier` + the caller's
  lazy draw): ``crit% = weapon.crit_chance × arc multiplier`` (×1 front,
  ×2 flank, ×4 behind — ``[combat] crit_*_mult``); a crit multiplies the
  packet amount by ``weapon.crit_mult`` before mitigation
  (:func:`scale_half_away`). Arc widths are species-profile DATA
  (``EnvironmentProfile.front_arc_deg`` / ``behind_arc_deg``): a slime blob
  ships 360/0 and has no back to stab — zero special-case code.

Determinism (engine/14 — every number through a door; integer since #29
P3e):

- **THE LAZY-ROLL RULE (mechanics/03 §3, canon):** a roll that cannot matter
  is never drawn. No cover on the approach → no exposure draw;
  ``crit_chance == 0`` → no crit draw. Draw count therefore depends only on
  synced state, so replay/cross-machine identity hold and a weapon with the
  feature dialed to zero leaves the RNG stream — and the golden digest —
  untouched (the dormant-seam pattern).
- **THE DRAW FORM OF RECORD (#29 P3e):** every roll here is
  ``draws.roll_q16(d, p_q)`` -- one keyed draw ``d`` (#29 P3b: a
  ``simulation.draws.Draw``, its Philox word's high 16 bits) compared below
  a Q16 probability ``p_q`` (exactly ``p_q / 65536``).
  ``p_q`` is quantized at a door: the materials table's
  ``cover_exposure_q16``, the weapon's ``crit_chance`` and the
  ``[combat] crit_*_mult`` constants (Q16, once per value).
- **No trig, no float:** arc classification compares the Q16.16 wrapped
  difference between the impact bearing and the target's synced
  ``facing_q`` (``turning.wrap_q16``) against Q16 half-arcs quantized from
  the profile's degrees at the door.
"""
from __future__ import annotations

import math

from config import CFG
from simulation import draws as _draws
from simulation import unit_fixed as _ufx


# ---------------------------------------------------------------------------
# Exposure vs cover (mechanics/06 §5; mechanics/03 §3 "the exposure roll")
# ---------------------------------------------------------------------------
def cover_exposure_at(gmap, iy, ix):
    """The ``cover_exposure`` of the material at (iy, ix) as a Q16
    probability (#29 P3e) — ``ONE`` = no concealment (never roll), below
    ``ONE`` = soft cover (roll on entry-through).

    Reads the materials table's ``cover_exposure_q16`` column (quantized once
    at its build -- the door) through the material grid. Bare test stubs
    without a materials table carry no cover data → ``ONE`` (no
    concealment), which keeps the lazy-roll rule intact: a stub map draws
    nothing."""
    materials = getattr(gmap, "materials", None)
    col = getattr(materials, "cover_exposure_q16", None)
    if col is None:
        return _ufx.FP_ONE
    return int(col[int(gmap.material[iy, ix])])


def roll_exposure(exposure_q, d):
    """One exposure roll — True = the shot connects, False = the cover
    absorbs it: ``draws.roll_q16(d, exposure_q)`` (one keyed draw ``d``,
    True with probability ``exposure_q / 65536``).

    Callers must invoke this LAZILY (only when ``exposure_q < ONE``): the
    lazy-roll rule is what keeps dormant cover off the RNG stream."""
    return _draws.roll_q16(d, exposure_q)


# ---------------------------------------------------------------------------
# Crit vs facing (mechanics/06 §5 "the crit roll")
# ---------------------------------------------------------------------------
_HALF_ARC_Q = {}


def _half_arc_q(deg) -> int:
    """Half of an arc in degrees as Q16.16 radians -- the door of a profile
    constant, cached per value (#29 P3e)."""
    key = float(deg)
    q = _HALF_ARC_Q.get(key)
    if q is None:
        q = _ufx.quantize_scalar(0.5 * math.radians(key))   # p3e-door
        _HALF_ARC_Q[key] = q
    return q


_PI_Q = _ufx.quantize_scalar(math.pi)                         # p3e-door


def arc_multiplier(shot_angle_q, target) -> int:
    """Classify the attack arc of a connecting hit and return its crit
    multiplier as a Q16 int (``[combat] crit_front/flank/behind_mult``,
    quantized at the door).

    ``shot_angle_q`` is the hit's travel bearing in SCREEN convention, Q16.16
    radians (y down — the combat.py march: direction (cos a, sin a) on the
    grid). The unit facing convention is math-style (y up, 0 = East, CCW —
    unit.py), so the flight direction maps to ``-a`` and the IMPACT
    DIRECTION REVERSED (target → shooter bearing) is ``-a + π``. The arc is
    the wrapped difference between that bearing and the target's synced
    ``facing_q`` (``turning.wrap_q16``):

        |Δ| ≤ front_arc/2         → front  (×1)
        |Δ| ≥ π − behind_arc/2    → behind (×4)
        else                      → flank  (×2)

    Integer compares only (#29 P3e); consumes NO randomness (the multiplier
    feeds the caller's lazy crit draw). Arc widths come from the target's
    ``environment`` profile (species data — mechanics/06 §5 "arcs are
    data"); bare stubs fall back to the human 120/90."""
    from simulation.turning import wrap_q16
    profile = getattr(target, "environment", None)
    front_half = _half_arc_q(getattr(profile, "front_arc_deg", 120.0))
    behind_half = _half_arc_q(getattr(profile, "behind_arc_deg", 90.0))
    incoming = -int(shot_angle_q) + _PI_Q     # target→shooter bearing (y-up)
    facing_q = int(getattr(target, "facing_q", 0))
    ad = abs(wrap_q16(incoming - facing_q))
    combat = CFG.combat
    if ad <= front_half:
        return _mult_q(getattr(combat, "crit_front_mult", 1.0))
    if ad >= _PI_Q - behind_half:
        return _mult_q(getattr(combat, "crit_behind_mult", 4.0))
    return _mult_q(getattr(combat, "crit_flank_mult", 2.0))


def _mult_q(v) -> int:
    """A config multiplier as Q16 (the door; exact for the grid values)."""
    return _ufx.quantize_scalar(float(v))                     # p3e-door


def roll_crit(crit_chance, arc_mult_q, d):
    """One crit roll: ``draws.roll_q16`` against ``crit_chance × arc_mult``
    in Q16 (``crit_chance`` quantized at the door, the product rounded half
    away). Callers must gate LAZILY on ``crit_chance > 0`` BEFORE calling
    (all shipped ranged weapons carry 0 → the stream never moves)."""
    chance_q = _ufx.quantize_scalar(float(crit_chance))  # p3e-door: the row's
    p_q = _ufx.mul_q16_round(chance_q, int(arc_mult_q))
    return _draws.roll_q16(d, p_q)


def scale_half_away(amount, mult):
    """``amount × mult`` in EXACT INTS, rounding half away from zero — the
    crit amount rule (mechanics/06 §5: a crit multiplies the packet amount
    before mitigation; W2 pins the rounding).

    ``amount`` is the integer pre-mitigation packet amount; ``mult`` the
    multiplier -- a Q16 int (#29 P3e: the arc / crit path), or a config
    float quantized to Q16 here (the door: ``crit_mult``, the legacy facing
    defense). ``amount × mult_q / 65536`` rounded half away from zero, the
    ``fixedpoint::quantize`` idiom: 4.5 → 5, 10.5 → 11, −4.5 → −5; integer
    products pass through exactly."""
    m = mult if isinstance(mult, int) else _ufx.quantize_scalar(float(mult))
    return _ufx.mul_q16_round(int(amount), m)


__all__ = [
    "cover_exposure_at", "roll_exposure",
    "arc_multiplier", "roll_crit", "scale_half_away",
]
