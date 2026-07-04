"""The DamagePacket pipeline — mechanics/06 §2 (packet + apply) + §3 (mitigation).

Every damage source — coupling responses (heat, blast), bullets, melee, and
the future DoT statuses — emits :class:`DamagePacket`\\ s; ONE pipeline owns
everything after::

    DamagePacket(amount, dtype, source_id)
       │
       ▼ mitigation   amount' = max(0, amount − max(0, armor[dtype] − ap))
       │                        × resist_mult[dtype]
       ▼ quantize     unit_fixed.quantize_hp_delta (the Q2-lift snap)
       ▼ apply        unit.current_hp -= applied ;  events carry APPLIED amounts
       ▼ life         hp <= 0 → DEAD

P2 scope (behaviour-preserving wiring, 2026-07-05): ``amount`` lives in the
SAME float64 domain the shipped damage chains use (ingress door 3 — pure
``+ − × ÷`` on deterministic inputs, quantized at the HP write boundary by
:func:`unit_fixed.quantize_hp_delta`). The chapter's integer-packet
(``amount_q16``) representation and the per-unit/per-phase batching stage are
LATER patches; nothing here reorders or merges damage applications. With the
neutral default tables (armor 0, resist 1.0) every mitigation is an IEEE-exact
no-op (``x − 0`` and ``x × 1.0`` are identity ops on every finite float), so
routing the shipped sites through this pipeline is bit-identical to the
inline code it replaces — proven bitwise in ``tests/test_damage_pipeline.py``.

Mitigation tables (mechanics/06 §3) are door-2 species-profile data
(:mod:`simulation.species`): flat ``armor[dtype]`` (mainly KINETIC/ENERGY,
mainly equipment later; weapons carry AP, subtracted from flat armor) and
``resist_mult[dtype]`` (0 immune, 1 neutral, >1 vulnerable — the zombie's
``resist_mult[HEAT] = 4.0`` replaces the old ``fire_damage_multiplier``
special case; a vulnerability is just a resistance above 1). Authored values
snap onto the Q16.16 grid once at definition (door 2); the composition rule
(species + equipment armors ADD, multipliers MULTIPLY) arrives with the
equipment/status patches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from simulation import unit_fixed
from simulation.events import UnitHitEvent, UnitKilledEvent


# ---------------------------------------------------------------------------
# Damage types (mechanics/06 §2) — v1 vocabulary + reserved slots.
# Plain int constants in table order (the orders.py style); the mitigation
# tables below are tuples indexed by these.
# ---------------------------------------------------------------------------
KINETIC  = 0   # bullets, melee
BLAST    = 1   # overpressure
HEAT     = 2   # radiant + fire
ENERGY   = 3   # beams / lasers
POISON   = 4   # gas dose
ASPHYX   = 5   # O2 / water
HEAL     = 6   # negative-direction, unresisted in v1 (mitigation = identity)
ELECTRIC = 7   # RESERVED (engine/11) — defined, no emitter yet
PSY      = 8   # RESERVED (the Gray / will-stats) — defined, no emitter yet

N_DAMAGE_TYPES = 9

DAMAGE_TYPE_NAMES = {
    KINETIC:  "kinetic",
    BLAST:    "blast",
    HEAT:     "heat",
    ENERGY:   "energy",
    POISON:   "poison",
    ASPHYX:   "asphyx",
    HEAL:     "heal",
    ELECTRIC: "electric",
    PSY:      "psy",
}


# ---------------------------------------------------------------------------
# Mitigation tables (mechanics/06 §3) — the profile structure + builder.
# The authored per-species INSTANCES live in simulation.species (door-2 data);
# this module owns the vocabulary so species.py can import it one-way.
# ---------------------------------------------------------------------------

def _q16_snap(v: float) -> float:
    """Snap an authored table value onto the Q16.16 grid (ingress door 2) —
    the exact dyadic float n/65536, via the unit_fixed quantize twins
    (round half away from zero, matching cpp/src/fixed_point.h). The v1
    table values (0.0 armor, 1.0 / 4.0 resists) are already exact powers of
    two, so they pass through unchanged; the snap is the standing pattern
    for every future authored value (generation._q16_snap's twin).
    """
    return unit_fixed.dequantize_scalar(unit_fixed.quantize_scalar(float(v)))


@dataclass(frozen=True)
class MitigationProfile:
    """Per-profile mitigation tables (mechanics/06 §3), indexed by dtype.

    ``armor``       — flat points subtracted pre-multiplier (floor 0; weapon
                      AP is subtracted from armor first, floored at 0 so AP
                      can cancel armor but never grant bonus damage).
    ``resist_mult`` — multiplier: 0 immune, 1 neutral, >1 vulnerable.

    Both are length-``N_DAMAGE_TYPES`` tuples of Q16.16-snapped floats
    (door 2). Build instances with :func:`build_mitigation`.
    """
    armor:       tuple
    resist_mult: tuple


def build_mitigation(armor: Optional[dict] = None,
                     resist_mult: Optional[dict] = None) -> MitigationProfile:
    """Build a :class:`MitigationProfile` from sparse ``{dtype: value}``
    overrides on the neutral baseline (armor 0.0 everywhere, resist 1.0
    everywhere). Every authored value is Q16.16-snapped (door 2). Slots are
    independent, so dict iteration order cannot matter.
    """
    a = [0.0] * N_DAMAGE_TYPES
    r = [1.0] * N_DAMAGE_TYPES
    for dtype, value in (armor or {}).items():
        a[dtype] = _q16_snap(value)
    for dtype, value in (resist_mult or {}).items():
        r[dtype] = _q16_snap(value)
    return MitigationProfile(armor=tuple(a), resist_mult=tuple(r))


#: The neutral profile — mitigation is an IEEE-exact no-op through it.
#: Fallback for units that carry no profile (bare stubs in tests).
NEUTRAL_MITIGATION = build_mitigation()


# ---------------------------------------------------------------------------
# The packet
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DamagePacket:
    """One quantum of damage entering the pipeline (mechanics/06 §2).

    ``amount``    — pre-mitigation damage in the float64 domain the shipped
                    door-3 chains use (int amounts pass through exactly; the
                    integer ``amount_q16`` form is a later patch).
    ``dtype``     — damage type (KINETIC … PSY above).
    ``source_id`` — causing entity's unit id (shooter, melee attacker), or
                    ``None`` for environmental sources (heat, blast field).
                    Carried for future attribution (kill credit, AI blame);
                    the renderer-facing event ``source`` STRING is a separate
                    vocabulary passed to :func:`apply_packet`.
    ``ap``        — armor penetration, subtracted from flat armor before
                    mitigation (mechanics/06 §3). 0 for every shipped source.
    """
    amount:    float
    dtype:     int
    source_id: Optional[int] = None
    ap:        float = 0


# ---------------------------------------------------------------------------
# Mitigation (mechanics/06 §3): flat armor first, then multiplier — DECIDED.
# ---------------------------------------------------------------------------
def mitigate(amount, dtype: int, profile: MitigationProfile, ap=0):
    """``amount' = max(0, amount − max(0, armor[dtype] − ap)) × resist_mult[dtype]``

    Runs PRE-quantize, inside the existing float64 door-3 chain shape: pure
    ``+ − ×`` and comparisons on deterministic inputs — cross-machine exact.
    The floor keeps over-armored hits at 0 (small arms chip harmlessly off
    heavy plate) and keeps them 0 through the multiplier.

    Neutral identity (proven bitwise in tests): with armor 0 / resist 1.0,
    ``x − 0`` and ``x × 1.0`` are IEEE-exact no-ops, so ``mitigate(x) == x``
    to the bit for every finite non-negative amount.

    HEAL is unresisted in v1 (mechanics/06 §2) — mitigation is the identity:
    the damage floor must not zero a negative-direction (healing) amount, and
    armor never blocks a heal.
    """
    if dtype == HEAL:
        return amount
    effective_armor = max(0, profile.armor[dtype] - ap)
    return max(0.0, amount - effective_armor) * profile.resist_mult[dtype]


# Lazily-bound simulation.species (the unit_fixed._kit pattern). species.py
# imports THIS module's vocabulary at load; we only need its authored tables
# at damage time — the lazy bind keeps the import graph one-way.
_species_mod = None


def _species():
    global _species_mod
    if _species_mod is None:
        from simulation import species as _m
        _species_mod = _m
    return _species_mod


def mitigation_for(unit) -> MitigationProfile:
    """Resolve the unit's mitigation tables (mechanics/06 §3).

    Zombie-ness is runtime STATE, not a species (the one-species foundation
    decision, species.py) — so the zombie profile is selected by state at
    damage time, on exactly the predicate the dissolved ``fire_damage_
    multiplier`` special case used (``u.is_zombie``): construction with
    ``team=1``, end-of-round conversion, and direct flag flips all resolve
    identically to the old inline branch. When zombification becomes a
    status (mechanics/06 §4) this collapses into status multiplier
    composition. Non-zombies read the species profile pointer stamped on
    the unit at construction (``unit.mitigation``, mirroring
    ``unit.environment``); bare objects fall back to neutral.
    """
    if getattr(unit, "is_zombie", False):
        return _species().ZOMBIE_MITIGATION
    profile = getattr(unit, "mitigation", None)
    return NEUTRAL_MITIGATION if profile is None else profile


# ---------------------------------------------------------------------------
# Apply (mechanics/06 §2): mitigate → quantize → hp → events → life.
# ---------------------------------------------------------------------------
def apply_packet(unit, packet: DamagePacket, events=None, *,
                 source: str, mark_killed_by_zombie: bool = False):
    """Apply one packet to one unit — the single owner of the post-source
    damage chain. Replicates the shipped inline sequence EXACTLY (this is
    the load-bearing behaviour-preservation contract; the lockstep digest
    hashes the applied deltas, the emitted event stream, and hp):

    1. mitigate (§3) — at the position the old per-site multipliers sat
       (pre-quantize, inside the float64 chain);
    2. ``unit_fixed.quantize_hp_delta`` — the Q2-lift snap; the APPLIED value;
    3. ``unit.current_hp -= applied``;
    4. ``events``: :class:`UnitHitEvent` with the applied damage + ``source``
       (the renderer vocabulary: "heat" / "explosion" / "bullet" / "melee");
       ``events=None`` emits nothing (the melee site's shipped shape);
    5. life transition: ``hp <= 0`` → ``alive = False``; then
       ``mark_killed_by_zombie`` sets the conversion flag (melee kills ONLY —
       heat / blast / bullet deaths never convert); then
       :class:`UnitKilledEvent` (``killed_by = source``).

    Callers gate on ``unit.alive`` before applying, exactly as the shipped
    sites do. Returns the applied (post-quantize) delta.
    """
    amount = mitigate(packet.amount, packet.dtype, mitigation_for(unit),
                      ap=packet.ap)
    dmg = unit_fixed.quantize_hp_delta(amount)
    unit.current_hp -= dmg
    if events is not None:
        uid = getattr(unit, "id", -1)
        events.append(UnitHitEvent(unit_id=uid, damage=dmg, source=source))
    if unit.current_hp <= 0:
        unit.alive = False
        if mark_killed_by_zombie:
            unit.killed_by_zombie = True
        if events is not None:
            uid = getattr(unit, "id", -1)
            events.append(UnitKilledEvent(unit_id=uid, killed_by=source))
    return dmg


__all__ = [
    "KINETIC", "BLAST", "HEAT", "ENERGY", "POISON", "ASPHYX", "HEAL",
    "ELECTRIC", "PSY", "N_DAMAGE_TYPES", "DAMAGE_TYPE_NAMES",
    "MitigationProfile", "build_mitigation", "NEUTRAL_MITIGATION",
    "DamagePacket", "mitigate", "mitigation_for", "apply_packet",
]
