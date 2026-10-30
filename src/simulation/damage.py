"""The DamagePacket pipeline — mechanics/06 §2 (packet + apply) + §3 (mitigation).

Every damage source — coupling responses (heat, blast), bullets, melee, and
the future DoT statuses — emits :class:`DamagePacket`\\ s; ONE pipeline owns
everything after::

    DamagePacket(amount, dtype, attacker_id)      + its cause (#123)
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
from simulation.events import (UnitHitEvent, UnitKilledEvent, NO_ATTACKER,
                               attacker_value, cause_value)


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
#: NOT a tuning knob: armor 0 / resist 1.0 is the structural identity the
#: bit-identity contract of P2 rests on (a marine's damage must equal the
#: pre-pipeline inline chain). The tunable STANDARD VALUES live in the
#: per-species tables (species.py, sourced from config.toml — e.g.
#: ``[zombie] fire_damage_multiplier``); tune THERE, never here.
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
    ``attacker_id`` — the unit CREDITED with the hit (#123): the shooter,
                    the striker, the biting zombie, the thrower / firer /
                    planter of a payload, the sprayer whose jet was on the
                    tile this tick, a status's applier — or ``NO_ATTACKER``
                    (-1, also ``None``) for the environment. It rides the
                    hit / kill events; the CAUSE (``events.Cause``) is passed
                    to :func:`apply_packet` as ``source``.
    ``ap``        — armor penetration, subtracted from flat armor before
                    mitigation (mechanics/06 §3). 0 for every shipped source.
    ``indirect``  — True when the credit came THROUGH THE WORLD (a payload's
                    blast, a jet's or a splash's heat or gas this tick —
                    :class:`TickCredit`), not a strike aimed at this unit.
                    Credit only: the events carry the attacker either way;
                    the idle stance's ``recent_attackers`` list ignores an
                    indirect hit, exactly as it ignored those (then
                    unattributed) packets before #123 — no behaviour moved.
    """
    amount:      float
    dtype:       int
    attacker_id: Optional[int] = NO_ATTACKER
    ap:          float = 0
    indirect:    bool = False


# ---------------------------------------------------------------------------
# Indirect credit (#123): who put THIS TICK's heat / gas deposit on a tile.
# ---------------------------------------------------------------------------
#: Channel key of the ``heat`` ingress buffer in a :class:`TickCredit`; a gas
#: deposit is keyed by its ``gmap.gas`` slice id (an int).
CREDIT_HEAT = "heat"


class TickCredit:
    """The owners of the per-tick world deposits a unit can be hurt by (#123).

    The MINIMAL honest mechanism for environmental credit: a deposit site
    that knows its owner (a flame jet's sprayer, a payload's thrower / firer
    / planter) CLAIMS the tiles its ``heat`` or gas edit covers, and the
    exchange rows (``apply_environmental_damage``, ``apply_poison_dose``)
    credit a hit to the claimant of the footprint tile that hurt the unit —
    only where the hurt came from that deposit's own plane (the ``heat``
    buffer, not the radiation sensor).

    It is NOT a provenance plane: no per-tile owner survives the tick. The
    store is cleared once per tick together with the ``heat`` buffer it
    describes (``Simulation.step``'s end-of-tick clear), so a fire the jet
    lit, a lingering gas cloud, or a hot room after the splash is credited
    to the ENVIRONMENT — the accepted gap (#123), since the fire / gas
    planes carry no per-tile provenance and one is not added for this.

    A plain dict, never digested (the ``gmap.render_dirty`` precedent); its
    effect is — the credited ``attacker_id`` rides the hashed hit events.
    Deterministic: claims arrive in the sim's fixed order and the FIRST
    claimant of a (channel, tile) keeps it this tick.
    """

    __slots__ = ("_owner",)

    def __init__(self):
        self._owner = {}

    def claim(self, channel, tiles, attacker_id) -> None:
        """Claim ``tiles`` (``(y, x)`` pairs) on ``channel`` for
        ``attacker_id``; a no-op for the environment (-1 / None)."""
        a = attacker_value(attacker_id)
        if a == NO_ATTACKER:
            return
        own = self._owner
        for (y, x) in tiles:
            own.setdefault((channel, int(y), int(x)), a)

    def attacker_at(self, channel, y, x) -> int:
        """The claimant of ``(y, x)`` on ``channel`` this tick, else -1."""
        return self._owner.get((channel, int(y), int(x)), NO_ATTACKER)

    def clear(self) -> None:
        self._owner.clear()

    def __len__(self) -> int:
        return len(self._owner)


def tick_credit(gmap) -> TickCredit:
    """``gmap``'s :class:`TickCredit` (created on first use, so a stand-in
    map in a test needs nothing)."""
    tc = getattr(gmap, "damage_credit", None)
    if tc is None:
        tc = TickCredit()
        gmap.damage_credit = tc
    return tc


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
    ``team=ZOMBIE_TEAM``, conversion, and direct flag flips all resolve
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
    4. ``events``: :class:`UnitHitEvent` with the applied damage, the cause
       ``source`` and the packet's ``attacker_id``; ``events=None`` emits
       nothing (since #122 the melee site passes the tick's events too);
    5. life transition: ``hp <= 0`` → ``alive = False``; then
       ``mark_killed_by_zombie`` sets the conversion flag (melee kills ONLY —
       heat / blast / bullet deaths never convert); then
       :class:`UnitKilledEvent` (``killed_by = source``, the same attacker).

    THE PACKET DOOR for credit (#123): ``source`` must be a
    :class:`simulation.events.Cause` (member or its string) — anything else
    raises ``ValueError`` BEFORE the unit is touched.

    Callers gate on ``unit.alive`` before applying, exactly as the shipped
    sites do. Returns the applied (post-quantize) delta.
    """
    source = cause_value(source)                  # the door: refuse first
    attacker = attacker_value(getattr(packet, "attacker_id", NO_ATTACKER))
    amount = mitigate(packet.amount, packet.dtype, mitigation_for(unit),
                      ap=packet.ap)
    dmg = unit_fixed.quantize_hp_delta(amount)
    unit.current_hp -= dmg
    # Who is shooting at me (onephase_wego design §13, the idle stance): a unit
    # with no orders "returns fire at attackers — preferring marked targets —
    # but does not free-fire at everything it sees", so it needs to know who
    # attacked it and when. Recorded here because this is the ONE place every
    # damage source converges. Attribution only — no behaviour, no events, no
    # digest surface — so the shipped rulesets are unaffected; an attacker of
    # -1 (environment: heat, blast with no shooter) is not an attacker.
    # Deliberately UNAGED: the list is cleared at the round boundary, so it
    # means "who has shot at me this round" — which is the right window for a
    # 4 s round and needs no clock threaded through every damage call site.
    # An INDIRECT credit (#123: a blast, a jet's heat or gas, credited through
    # the world) is not "shooting at me": those packets were unattributed
    # before #123 and stay out of this list, so the stance is unchanged.
    if attacker != NO_ATTACKER and not getattr(packet, "indirect", False):
        recent = getattr(unit, "recent_attackers", None)
        if (recent is not None
                and attacker != int(getattr(unit, "id", -1))):
            recent[attacker] = recent.get(attacker, 0) + 1
    if events is not None:
        uid = getattr(unit, "id", -1)
        events.append(UnitHitEvent(unit_id=uid, damage=dmg, source=source,
                                   attacker_id=attacker))
    if unit.current_hp <= 0:
        unit.alive = False
        if mark_killed_by_zombie:
            unit.killed_by_zombie = True
        if events is not None:
            uid = getattr(unit, "id", -1)
            events.append(UnitKilledEvent(unit_id=uid, killed_by=source,
                                          attacker_id=attacker))
    return dmg


__all__ = [
    "KINETIC", "BLAST", "HEAT", "ENERGY", "POISON", "ASPHYX", "HEAL",
    "ELECTRIC", "PSY", "N_DAMAGE_TYPES", "DAMAGE_TYPE_NAMES",
    "MitigationProfile", "build_mitigation", "NEUTRAL_MITIGATION",
    "DamagePacket", "mitigate", "mitigation_for", "apply_packet",
    "CREDIT_HEAT", "TickCredit", "tick_credit",
]
