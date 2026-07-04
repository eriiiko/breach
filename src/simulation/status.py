"""The status/condition system — mechanics/06 §4 (one system, many triggers).

The two-axis model (mechanics/06 §1): LIFE (``ALIVE | DEAD``, on the unit) is
minimal on purpose; everything temporarily true of a unit — crowd control
(knocked down, immobilized, stunned, paralyzed), afflictions (burning,
poisoned, suffocating), buffs (regen) — is a **status**: applied by anyone
(a coupling row, a weapon, a collision, terrain), interpreted uniformly
(behavior flags). No meaning is baked into the state machine; kinds are
config rows and adding one is O(row).

Determinism (engine/14 + mechanics/05 §4/§5) — statuses are SYNCED unit
state, digest-hashed (the ``__unit_status__`` sub-hash):

- **Durations are integer ticks** (door 1). ``remaining_ticks`` counts the
  status-tick passes the effect still participates in.
- **Magnitudes are Q16.16-snapped at application** (door 2): ``apply_status``
  quantizes the authored float once; :class:`StatusEffect` stores the INTEGER
  count (``magnitude_q16``) — L2 representation, a nondeterministic float
  physically cannot live in it. Emission dequantizes to the exact dyadic
  ``n/65536`` float the packet pipeline consumes.
- **Order is P0**: units in id order, each unit's status list in list order,
  events in emission order. Nothing iterates a dict/set.

The tick contract (:func:`tick_statuses` — called at the TOP of the
unit-simulation section of ``Simulation.step``, ch. 05 §4 phase 3):

1. Dead units are skipped whole — statuses FREEZE on a corpse (no decrement,
   no emission; kept for forensics/looting rules later).
2. Per status, in list order: (a) a DoT/HoT kind emits ONE
   :class:`~simulation.damage.DamagePacket` of ``magnitude`` per tick through
   :func:`simulation.damage.apply_packet` — so mitigation composes for free
   (a zombie's BURNING ticks at ``resist_mult[HEAT] = 4.0``x a marine's,
   never coded per-status); HEAL-typed kinds (REGEN) emit the NEGATIVE
   amount (heal is negative-direction damage, unresisted in v1). A unit
   killed mid-list by an earlier status stops receiving emissions the same
   pass. (b) ``remaining_ticks`` decrements. (c) At ``<= 0`` the status
   expires (removed after the unit's list is processed, in place).
3. Duration semantics for triggers: a status applied with duration ``N``
   BEFORE this tick's status pass / flag consumers (the projectile-blast and
   exchange-read positions P4's triggers use) suppresses for ``N``
   consecutive ticks INCLUDING the application tick, and a DoT emits exactly
   ``N`` packets. A trigger firing AFTER the pass (shooting, zombie melee)
   starts the same contract on the NEXT tick.

Flag composition (:func:`composed_flags`): each kind's row declares the
behavior booleans it suppresses (``can_move`` / ``can_act`` / ``can_aim`` —
``False`` in the row means "this status takes it away"; composition is AND)
and whether it lays the unit prone (``is_prone`` — composition is OR). Unit
logic consults the COMPOSED flags, never individual statuses — that is what
makes conditions crowd-control without special-casing.

No max-HP clamp on heals in v1 (accepted gap — the overheal rule is a
ruleset decision owed to the standard-values pass, mechanics/06 §8).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from simulation import unit_fixed
from simulation.damage import (
    ASPHYX, HEAL, HEAT, POISON, DamagePacket, apply_packet,
)


# ---------------------------------------------------------------------------
# Status kinds (mechanics/06 §4, the v1 roster) — plain int constants in
# table order (the damage.py / orders.py style); STATUS_REGISTRY below is
# indexed by these.
# ---------------------------------------------------------------------------
KNOCKED_DOWN = 0   # CC: prone, no move/act (blast Δv trigger — P4)
IMMOBILIZED  = 1   # CC: no move
STUNNED      = 2   # CC: no act, no aim
PARALYZED    = 3   # CC: no move, no act, no aim
BURNING      = 4   # DoT: HEAT packets per tick (fire coupling row — later)
POISONED     = 5   # DoT: POISON packets per tick (gas dose row — later)
SUFFOCATING  = 6   # DoT: ASPHYX packets per tick (O2/water rows — later)
REGEN        = 7   # HoT: HEAL packets per tick (heal/stabilize mechanism)

N_STATUS_KINDS = 8

# Stacking rules (mechanics/06 §4): what a re-application of the SAME kind
# does. Genuine tie rules are explicit data, not code accidents (ch. 05 P4).
STACK_REFRESH = "refresh"   # one instance; new application resets it
STACK_STACK   = "stack"     # instances coexist; each ticks independently
STACK_MAX     = "max"       # one instance; keeps max(magnitude), max(ticks)


@dataclass(frozen=True)
class StatusKindDef:
    """One registry row — the config-style declaration of a status kind.

    ``can_move`` / ``can_act`` / ``can_aim``: ``False`` = this kind
    SUPPRESSES the flag (AND-composed across active statuses).
    ``is_prone``: ``True`` = this kind lays the unit prone (OR-composed;
    render + future hitbox/stamp/exposure implications, mechanics/06 §5).
    ``dtype``: the DoT/HoT damage type (``damage.KINETIC..PSY``) — the kind
    emits one packet of ``magnitude`` per tick through the §2 pipeline;
    ``None`` = pure CC/buff, no emission. HEAL-typed kinds emit the negative
    amount. Speed/accuracy modifier columns arrive with the stat-modifier
    statuses (mechanics/06 §4 roster tail) — not in the v1 rows.
    """
    kind:     int
    name:     str            # also the UnitHitEvent ``source`` vocabulary
    stacking: str
    can_move: bool = True
    can_act:  bool = True
    can_aim:  bool = True
    is_prone: bool = False
    dtype:    Optional[int] = None


# The v1 roster (mechanics/06 §4) — rows in kind order; magnitudes/durations
# arrive from the applying trigger (door-2 config), never from these rows.
STATUS_REGISTRY: tuple[StatusKindDef, ...] = (
    StatusKindDef(KNOCKED_DOWN, "knocked_down", STACK_REFRESH,
                  can_move=False, can_act=False, is_prone=True),
    StatusKindDef(IMMOBILIZED,  "immobilized",  STACK_REFRESH,
                  can_move=False),
    StatusKindDef(STUNNED,      "stunned",      STACK_REFRESH,
                  can_act=False, can_aim=False),
    StatusKindDef(PARALYZED,    "paralyzed",    STACK_REFRESH,
                  can_move=False, can_act=False, can_aim=False),
    StatusKindDef(BURNING,      "burning",      STACK_MAX,   dtype=HEAT),
    StatusKindDef(POISONED,     "poisoned",     STACK_STACK, dtype=POISON),
    StatusKindDef(SUFFOCATING,  "suffocating",  STACK_REFRESH, dtype=ASPHYX),
    StatusKindDef(REGEN,        "regen",        STACK_REFRESH, dtype=HEAL),
)

assert all(row.kind == i for i, row in enumerate(STATUS_REGISTRY)), \
    "STATUS_REGISTRY must be indexed by kind (table order)"


# ---------------------------------------------------------------------------
# The per-unit status instance
# ---------------------------------------------------------------------------
@dataclass
class StatusEffect:
    """One active status on one unit (mechanics/06 §4) — SYNCED state.

    ``magnitude_q16``   — Q16.16 INTEGER counts (door 2, snapped once in
                          :func:`apply_status`); for DoT/HoT kinds this is
                          the per-tick amount, for CC it is spare data
                          (0 unless a trigger wants to carry intensity).
    ``remaining_ticks`` — integer ticks left (door 1); see the module
                          docstring for the exact duration contract.
    ``source_id``       — causing entity's unit id, or ``None`` for
                          environmental sources (matches DamagePacket's
                          convention; forwarded into emitted packets for
                          future kill attribution).
    """
    kind:            int
    magnitude_q16:   int
    remaining_ticks: int
    source_id:       Optional[int] = None


# ---------------------------------------------------------------------------
# Composed behavior flags (mechanics/06 §4)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComposedFlags:
    """The AND-of-suppressions (+ OR for prone) over a unit's active
    statuses. Unit logic consults THIS, never individual statuses."""
    can_move: bool = True
    can_act:  bool = True
    can_aim:  bool = True
    is_prone: bool = False


#: The no-statuses fast path — everything permitted, standing.
FLAGS_DEFAULT = ComposedFlags()


def composed_flags(unit) -> ComposedFlags:
    """Compose the behavior flags over ``unit``'s active statuses.

    AND across suppressions (any active status that declares
    ``can_move=False`` takes movement away; likewise act/aim), OR for
    ``is_prone``. With no statuses this is the all-True default — the
    consumers' checks are then dead paths, bit-identical to pre-status
    behavior (the P3 wiring contract).
    """
    statuses = getattr(unit, "statuses", None)
    if not statuses:
        return FLAGS_DEFAULT
    can_move = can_act = can_aim = True
    is_prone = False
    for st in statuses:
        row = STATUS_REGISTRY[st.kind]
        can_move = can_move and row.can_move
        can_act = can_act and row.can_act
        can_aim = can_aim and row.can_aim
        is_prone = is_prone or row.is_prone
    return ComposedFlags(can_move, can_act, can_aim, is_prone)


# ---------------------------------------------------------------------------
# Application (with the per-kind stacking rule)
# ---------------------------------------------------------------------------
def apply_status(unit, kind: int, magnitude: float, duration_ticks: int,
                 source_id: Optional[int] = None) -> StatusEffect:
    """Apply ``kind`` to ``unit`` under the kind's stacking rule.

    ``magnitude`` is an authored/computed REAL value (e.g. damage per tick);
    it snaps onto the Q16.16 grid HERE (ingress door 2) and the instance
    stores the integer count. ``duration_ticks`` must be an integer >= 1
    (door 1) — a non-positive duration is a trigger/config bug and fails
    LOUDLY rather than deterministically doing nothing.

    Stacking (mechanics/06 §4, per-kind data):
      - ``refresh``: one instance — re-application overwrites its magnitude,
        duration, and source (timer reset).
      - ``stack``:   instances coexist — always appends; each ticks and
        expires independently (dose accumulation).
      - ``max``:     one instance — keeps ``max(magnitude)`` and
        ``max(remaining_ticks)`` component-wise; the source updates only
        when the new magnitude is strictly greater (the stronger burn's
        cause gets the attribution).

    Returns the created/updated :class:`StatusEffect`. List positions are
    stable (an existing instance refreshes IN PLACE), so P0's list order is
    the application order of first contact — deterministic.
    """
    row = STATUS_REGISTRY[kind]
    duration = int(duration_ticks)
    if duration < 1:
        raise ValueError(
            f"apply_status({row.name}): duration_ticks must be >= 1 integer "
            f"ticks (got {duration_ticks!r})")
    mag_q16 = unit_fixed.quantize_scalar(float(magnitude))

    statuses = getattr(unit, "statuses", None)
    if statuses is None:                # bare stub units in tests
        statuses = unit.statuses = []

    if row.stacking != STACK_STACK:
        for st in statuses:             # list order — first instance owns
            if st.kind != kind:
                continue
            if row.stacking == STACK_REFRESH:
                st.magnitude_q16 = mag_q16
                st.remaining_ticks = duration
                st.source_id = source_id
            else:                       # STACK_MAX
                if mag_q16 > st.magnitude_q16:
                    st.magnitude_q16 = mag_q16
                    st.source_id = source_id
                st.remaining_ticks = max(st.remaining_ticks, duration)
            return st

    st = StatusEffect(kind=kind, magnitude_q16=mag_q16,
                      remaining_ticks=duration, source_id=source_id)
    statuses.append(st)
    return st


# ---------------------------------------------------------------------------
# The per-tick pass (ch. 05 §4 — top of tick phase 3, P0 order)
# ---------------------------------------------------------------------------
def tick_statuses(units, events=None) -> None:
    """One status tick for every unit — durations count down, expired
    statuses drop, DoT/HoT kinds emit DamagePackets through the §2 pipeline.

    P0 order: units in id order (explicitly sorted — never list order),
    each unit's status list in list order. Dead units are skipped whole
    (statuses freeze on corpses). ``events`` is the sim's tick-event list
    (hit/kill events emitted by ``apply_packet`` land there, in emission
    order — synced, digest-hashed); ``None`` emits nothing.

    DoT deaths never convert (``mark_killed_by_zombie`` stays False —
    zombie conversion is melee-kill-only, mechanics/06 §6). Zero-magnitude
    DoTs emit nothing (no zero-damage event spam). Expiry compacts the
    list IN PLACE so any held reference to ``unit.statuses`` stays valid
    (the in-place-write discipline).
    """
    for u in sorted(units, key=lambda u: int(getattr(u, "id", -1))):
        if not u.alive:
            continue                    # frozen on corpses
        statuses = getattr(u, "statuses", None)
        if not statuses:
            continue
        for st in statuses:
            row = STATUS_REGISTRY[st.kind]
            if row.dtype is not None and st.magnitude_q16 != 0 and u.alive:
                amount = unit_fixed.dequantize_scalar(st.magnitude_q16)
                if row.dtype == HEAL:
                    amount = -amount    # heal = negative-direction damage
                apply_packet(
                    u,
                    DamagePacket(amount=amount, dtype=row.dtype,
                                 source_id=st.source_id),
                    events, source=row.name)
            st.remaining_ticks -= 1
        statuses[:] = [st for st in statuses if st.remaining_ticks > 0]


# ---------------------------------------------------------------------------
# Canonical digest serialization (engine/14 §5 L3 — the __unit_status__ sub-hash)
# ---------------------------------------------------------------------------
def serialize_statuses(unit) -> list:
    """The canonical, byte-stable serialization of a unit's status list for
    the synced unit-state digest: ``[[kind, magnitude_q16, remaining_ticks,
    source_id], ...]`` in LIST ORDER (list order IS the P0 sync order — do
    not sort). All entries are plain ints (``None`` source -> ``-1``), so
    ``repr`` is byte-stable across runs and machines.
    """
    return [
        [int(st.kind), int(st.magnitude_q16), int(st.remaining_ticks),
         -1 if st.source_id is None else int(st.source_id)]
        for st in getattr(unit, "statuses", ())
    ]


__all__ = [
    "KNOCKED_DOWN", "IMMOBILIZED", "STUNNED", "PARALYZED",
    "BURNING", "POISONED", "SUFFOCATING", "REGEN", "N_STATUS_KINDS",
    "STACK_REFRESH", "STACK_STACK", "STACK_MAX",
    "StatusKindDef", "STATUS_REGISTRY", "StatusEffect",
    "ComposedFlags", "FLAGS_DEFAULT", "composed_flags",
    "apply_status", "tick_statuses", "serialize_statuses",
]
