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

Determinism (engine/14 — every number through a door):

- **THE LAZY-ROLL RULE (mechanics/03 §3, canon):** a roll that cannot matter
  is never drawn. No cover on the approach → no exposure draw;
  ``crit_chance == 0`` → no crit draw. Draw count therefore depends only on
  synced state, so replay/cross-machine identity hold and a weapon with the
  feature dialed to zero leaves the RNG stream — and the golden digest —
  untouched (the dormant-seam pattern).
- **THE DRAW FORM OF RECORD (door 4):** every roll here is
  ``float(rng.uniform(0.0, 1.0)) < p`` — one uniform dyadic from the seeded
  PCG64 stream (``uniform`` is affine on the bitstream: exact) compared
  against a load-time constant ``p``. The compare is exact for ANY fixed
  ``p``; the door-2 ingress step is the once-at-load float32 cast /
  config-float read that produced ``p`` (cover_exposure column, crit
  columns). No distribution methods, no re-quantize per roll.
- **No trig:** arc classification is pure ``+ − ×`` on angles that are
  already deterministic (kit-atan2 facings, kit-based march angles,
  ``math.radians`` on config constants — the same affine transform the
  shipped cone code uses) plus exact float compares — door 3 throughout.
"""
from __future__ import annotations

import math

from config import CFG


# ---------------------------------------------------------------------------
# Exposure vs cover (mechanics/06 §5; mechanics/03 §3 "the exposure roll")
# ---------------------------------------------------------------------------
def cover_exposure_at(gmap, iy, ix):
    """The ``cover_exposure`` of the material at (iy, ix) — 1.0 = no
    concealment (never roll), < 1.0 = soft cover (roll on entry-through).

    Reads the materials-table column projected through the material grid.
    Bare test stubs without a materials table carry no cover data → 1.0 (no
    concealment), which keeps the lazy-roll rule intact: a stub map draws
    nothing. Returned as a plain float — a load-time constant (the float32
    cast happened once at table build; ingress door 2)."""
    materials = getattr(gmap, "materials", None)
    if materials is None or not hasattr(materials, "cover_exposure"):
        return 1.0
    return float(materials.cover_exposure[int(gmap.material[iy, ix])])


def roll_exposure(exposure, rng):
    """One exposure roll — True = the shot connects, False = the cover
    absorbs it. THE DRAW FORM OF RECORD (module docstring):
    ``float(rng.uniform(0.0, 1.0)) < exposure`` — door 4, exact.

    Callers must invoke this LAZILY (only when ``exposure < 1.0``): the
    lazy-roll rule is what keeps dormant cover off the RNG stream."""
    return float(rng.uniform(0.0, 1.0)) < exposure


# ---------------------------------------------------------------------------
# Crit vs facing (mechanics/06 §5 "the crit roll")
# ---------------------------------------------------------------------------
_TWO_PI = 2.0 * math.pi


def arc_multiplier(shot_angle_screen, target):
    """Classify the attack arc of a connecting hit and return its crit
    multiplier (``[combat] crit_front/flank/behind_mult``).

    ``shot_angle_screen`` is the bullet's march angle in SCREEN convention
    (y down — the combat.py march: direction (cos a, sin a) on the grid).
    The unit facing convention is math-style (y up, 0 = East, CCW —
    unit.py), so the flight direction maps to ``-a`` and the IMPACT
    DIRECTION REVERSED (target → shooter bearing) is ``-a + π``. The arc is
    the wrapped difference between that bearing and the target's synced
    facing:

        |Δ| ≤ front_arc/2         → front  (×1)
        |Δ| ≥ π − behind_arc/2    → behind (×4)
        else                      → flank  (×2)

    Pure ``+ − ×`` + exact compares on deterministic inputs (kit facings,
    kit-derived march angles, ``math.radians`` of profile constants) —
    door 3; consumes NO randomness (the multiplier feeds the caller's lazy
    crit draw). Arc widths come from the target's ``environment`` profile
    (species data — mechanics/06 §5 "arcs are data"); bare stubs fall back
    to the human 120/90."""
    profile = getattr(target, "environment", None)
    front_half = 0.5 * math.radians(
        float(getattr(profile, "front_arc_deg", 120.0)))
    behind_half = 0.5 * math.radians(
        float(getattr(profile, "behind_arc_deg", 90.0)))

    incoming = -float(shot_angle_screen) + math.pi   # target→shooter bearing (y-up)
    delta = incoming - float(getattr(target, "facing", 0.0))
    while delta > math.pi:
        delta -= _TWO_PI
    while delta < -math.pi:
        delta += _TWO_PI
    ad = abs(delta)

    combat = CFG.combat
    if ad <= front_half:
        return float(getattr(combat, "crit_front_mult", 1.0))
    if ad >= math.pi - behind_half:
        return float(getattr(combat, "crit_behind_mult", 4.0))
    return float(getattr(combat, "crit_flank_mult", 2.0))


def roll_crit(crit_chance, arc_mult, rng):
    """One crit roll — the draw form of record against
    ``crit_chance × arc_mult`` (an exact IEEE product of two load-time
    constants). Callers must gate LAZILY on ``crit_chance > 0`` BEFORE
    calling (all shipped weapons carry 0 → the stream never moves)."""
    return float(rng.uniform(0.0, 1.0)) < float(crit_chance) * float(arc_mult)


def scale_half_away(amount, mult):
    """``amount × mult`` in EXACT INTS, rounding half away from zero — the
    crit amount rule (mechanics/06 §5: a crit multiplies the packet amount
    before mitigation; W2 pins the rounding).

    ``amount`` is the integer pre-mitigation packet amount; ``mult`` the
    weapon's ``crit_mult`` (config float, door 2). The product is one
    correctly-rounded IEEE multiply (exact for in-range game numbers); the
    round is the ``fixedpoint::quantize`` idiom without the ×65536 —
    ``floor(v + 0.5)`` for v ≥ 0, ``ceil(v − 0.5)`` below (sign-symmetric,
    no toward-−∞ bias — the fixed-point-arc lesson). 4.5 → 5, 10.5 → 11,
    −4.5 → −5; integer products pass through exactly."""
    v = float(amount) * float(mult)
    return int(math.floor(v + 0.5) if v >= 0.0 else math.ceil(v - 0.5))


__all__ = [
    "cover_exposure_at", "roll_exposure",
    "arc_multiplier", "roll_crit", "scale_half_away",
]
