"""Poison -- the two-level poison law (#29 PX; ``docs/poison_law_design_2026-11-14.md``,
the frame ``docs/unit_design_2026-11-14.md`` section 6). The law and its first
numbers are Erik's rulings of 2026-11-14.

A unit holds two amounts of poison, Q16 ints (one unit of poison = 65 536
counts; one larva bite, Erik): the DEPOT ``poison_x0`` (entered the body, not
yet in the blood: a bite site, the lungs) and the BLOOD ``poison_x1`` (what
hurts). Poison moves depot -> blood -> out and HURTS AS IT LEAVES the blood::

    dx0/dt = -x0 / T_onset + input
    dx1/dt = +x0 / T_onset - x1 / T_decay
    damage rate = H x (x1 / T_decay)

so the total harm of a dose is ``H x dose`` whatever the two times are (the
dose and ``H`` set HOW MUCH, the times set WHEN). One dose rises, peaks and
decays (the Bateman curve). There is no ceiling on the harm (Erik: a unit
cannot be "fully poisoned"); the body's store is bounded only by the integer
range (:data:`X_MAX`, never reached in play: 32 767 units).

THE INTEGER LAW, per tick, for each living unit holding poison (the dose of
this tick already added by :func:`dose`, saturating at :data:`X_MAX` in all
and counted)::

    m0 = step(x0, a_onset, flush=False);       x0 -= m0;  x1 += m0
    m1 = step(x1, a_decay, flush=(x0 == 0));   x1 -= m1
    damage_q = mul_q16_round(m1, H_q)          one POISON DamagePacket (amount_q)

    step(x, a, flush) = 0                      if x == 0
                      = x                      if flush and x <= FLUSH_Q
                      = max(1, (x * a) >> 16)  otherwise

``m0`` and ``m1`` are each computed ONCE and applied to both sides, so no
poison is created or lost: over any schedule ``sum(doses) == x0 + x1 +
sum(m1)``. ``a_onset = round(ONE / (T_onset x tps))`` (an exact rational of
the authored float, rounded once at the door; no exponential is evaluated),
``a_decay`` likewise; at 24 tps the first values give 683 and 137. THE FLUSH:
once the depot is empty, a blood level at or below ``FLUSH_Q`` (1/64 of a
unit) leaves in one tick, so the levels always return to EXACTLY 0 -- the
unit goes dormant, ``poison_src`` returns to -1 and its digest rows disappear
(each column hashed only while non-zero). The depot drains at least one
count a tick and is never flushed. (AS BUILT, against the design's letter,
which flushed every level at or below ``FLUSH_Q``: a level still being FED
sits below 1/64 of a unit at the start of every dose -- one tick of full gas
is 0.025 units -- so that flush emptied the blood every tick and deleted the
decay stage, and emptied the depot of every fringe-gas dose; the flush is
for the tail only.)

INT64 BOUND (the door holds it): ``x0 + x1 <= X_MAX = 2^31 - 1``, so ``x * a
<= 2^31 x 2^16 = 2^47``; ``m1 <= X_MAX < 2^31`` and the hooked harm ``H_q x
8 < 2^31`` (``H < 4096`` points, the traits' largest hook ``8 ONE``), so
``m1 x harm < 2^62`` and the half-away rounding's doubling stays below 2^63;
the gas dose ``gas_q x density_q < 2^31 x 2^31 = 2^62``
(``gas_units_per_second < 32 768``, ``density_q <= TRACE_CELL_MAX``).

``damage_q`` goes through ``damage.apply_packet`` as ONE packet of type POISON
per tick (credited to ``poison_src``, ``indirect=True``, cause
``poison_gas``): mitigation (a zombie's resist 0 would zero it, though a
zombie never holds poison), exposure and the events are the pipeline's own. A
tick with ``m1 == 0`` emits nothing.

THE ONE DOOR IN: :func:`dose` (a coupling row calls it: today the
``gas[poison]`` row, ``exchange.apply_poison_dose``; after P6 the larvae's
bite field). A ZOMBIE takes no dose (it does not breathe and is immune to
bites, Erik): the door returns without writing. The latest dose's credited
unit is ``poison_src`` (ACCEPTED GAP: it takes the credit for all the poison
in the body). Zeroed at death and at conversion (:func:`clear`). Nothing but
this module writes the three columns, and nothing but :func:`tick` emits a
POISON packet from them (``tests/test_poison.py`` scans ``src/``).

PER-UNIT RATES: the three accessors :func:`onset_q`, :func:`decay_q`,
:func:`harm_q` are the only readers of the class values (the law, the panel,
the bench; ``tests/test_unit_traits.py`` scans for another): the class value
times the unit traits' hook ``TOX_ONSET`` / ``TOX_DECAY`` / ``TOX_HARM``
(``unit_traits.scaled``, the identity at ONE; design section 3; all ``ONE``
until #199), each fraction kept at most ONE.

Integer only, no random draw. The dials are construction-bound
(:func:`rebuild_settings`, called by ``Simulation.__init__``; the exposure
law's precedent). Device twin: P6.2 (a per-unit kernel, no order dependence).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from simulation import unit_fixed, unit_traits
from simulation.damage import POISON, DamagePacket, apply_packet
from simulation.events import NO_ATTACKER, attacker_value

#: The ``[damage.poison]`` keys, every one required.
SETTING_KEYS = ("onset_seconds", "decay_seconds", "hp_per_unit",
                "gas_units_per_second")

ONE = unit_fixed.FP_ONE
FP_SHIFT = unit_fixed.FP_SHIFT
#: The most poison a body holds in all (``x0 + x1``), in counts: the int32
#: range, 32 767 units -- the bound the int64 argument above rests on. A dose
#: past it is clipped and counted (:data:`SATURATION`); never reached in play.
X_MAX = (1 << 31) - 1
#: Below this many counts a level moves in one tick (1/64 of a unit; 0.16 HP
#: at the first values): the flush that brings every level to exactly 0.
FLUSH_Q = ONE // 64
#: The largest ``hp_per_unit`` (points) the int64 bound admits at the traits'
#: largest hook (8 ONE): ``H_q x 8 < 2^31``.
HP_PER_UNIT_MAX = 4096.0
#: The largest ``gas_units_per_second``: ``gas_q < 2^31``.
GAS_UNITS_MAX = 32768.0

#: Doses clipped at :data:`X_MAX`: ``{"hits": n, "dropped_q": counts}`` -- a
#: diagnostic, never hashed (the counted refusal of the bound).
SATURATION = {"hits": 0, "dropped_q": 0}


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------
def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"{where} must be a number (got {value!r})")
    return float(value)


def _items(value):
    return value.items() if isinstance(value, dict) else vars(value).items()


def fraction_q(seconds: float, tps: int) -> int:
    """The per-tick Q16 fraction of a time constant: ``round(ONE / (seconds
    x tps))``, half away from zero -- the authored float taken as its EXACT
    rational (``Fraction``), the quotient rounded once (no libm, no float
    division)."""
    t = Fraction(float(seconds))        # p3e-door: the authored time, exact
    return unit_fixed.div_round(ONE * t.denominator, t.numerator * int(tps))


@dataclass(frozen=True)
class PoisonSettings:
    """The ``[damage.poison]`` dials, quantized once."""
    tps: int
    class_onset_q: int  # a_onset, Q16 fraction of the depot moved per tick
    class_decay_q: int  # a_decay, Q16 fraction of the blood leaving per tick
    class_harm_q: int   # H, Q16 hit points per unit of poison leaving the blood
    gas_q: int          # Q16 units of poison per second at full gas density


def settings_from_config(cfg) -> PoisonSettings:
    """THE refusing door of ``[damage.poison]``: a missing or unknown key, a
    value that is not a number, a time constant below one tick, ``hp_per_unit``
    not in (0, 4096) or ``gas_units_per_second`` not in [0, 32768) (the int64
    bound of the module docstring) is refused naming the key."""
    sec = getattr(getattr(cfg, "damage", None), "poison", None)
    if sec is None:
        raise ValueError("config has no [damage.poison] table "
                         f"(keys {list(SETTING_KEYS)})")
    cols = dict(_items(sec))
    for key in SETTING_KEYS:
        if key not in cols:
            raise ValueError(f"[damage.poison] is missing '{key}'")
    unknown = sorted(set(cols) - set(SETTING_KEYS))
    if unknown:
        raise ValueError(f"[damage.poison]: unknown key(s) {unknown} "
                         f"(it carries {list(SETTING_KEYS)})")
    v = {k: _number(cols[k], f"[damage.poison] {k}") for k in SETTING_KEYS}
    tps = int(cfg.clock.ticks_per_second)
    for key in ("onset_seconds", "decay_seconds"):
        if not v[key] * tps >= 1.0:
            raise ValueError(
                f"[damage.poison] {key} must be at least one tick "
                f"(1/{tps} s; got {cols[key]!r})")
    if not 0.0 < v["hp_per_unit"] < HP_PER_UNIT_MAX:
        raise ValueError(
            f"[damage.poison] hp_per_unit must be in (0, {HP_PER_UNIT_MAX:g}) "
            f"(the int64 bound; got {cols['hp_per_unit']!r})")
    if not 0.0 <= v["gas_units_per_second"] < GAS_UNITS_MAX:
        raise ValueError(
            f"[damage.poison] gas_units_per_second must be in "
            f"[0, {GAS_UNITS_MAX:g}) (the int64 bound; got "
            f"{cols['gas_units_per_second']!r})")
    return PoisonSettings(
        tps=tps,
        class_onset_q=fraction_q(v["onset_seconds"], tps),
        class_decay_q=fraction_q(v["decay_seconds"], tps),
        class_harm_q=unit_fixed.quantize_scalar(v["hp_per_unit"]),
        gas_q=unit_fixed.quantize_scalar(v["gas_units_per_second"]))


_SETTINGS: PoisonSettings | None = None


def get_settings() -> PoisonSettings:
    """The shared settings, built from CFG on first use."""
    global _SETTINGS
    if _SETTINGS is None:
        rebuild_settings()
    return _SETTINGS


def rebuild_settings(cfg=None) -> PoisonSettings:
    """Rebuild from the live CFG (``Simulation`` construction)."""
    global _SETTINGS
    if cfg is None:
        from config import CFG as cfg
    _SETTINGS = settings_from_config(cfg)
    return _SETTINGS


# ---------------------------------------------------------------------------
# The accessors (one per hooked value; design section 3)
# ---------------------------------------------------------------------------
def _fraction(q: int) -> int:
    """A per-tick fraction kept at most ONE (a fraction above one would move
    more than the level holds; reached only at a hook above ``ONE / a``)."""
    return ONE if q > ONE else q


def onset_q(u, s: PoisonSettings | None = None) -> int:
    """THE per-tick fraction of ``u``'s depot that reaches the blood (Q16):
    ``a_onset`` times its ``TOX_ONSET`` hook (``unit_traits.scaled``: one
    rounding, the identity at ONE), at most ONE. int64: ONE x 8 ONE."""
    s = s or get_settings()
    return _fraction(unit_traits.scaled(u, unit_traits.TOX_ONSET,
                                        s.class_onset_q))


def decay_q(u, s: PoisonSettings | None = None) -> int:
    """THE per-tick fraction of ``u``'s blood that leaves it (Q16):
    ``a_decay`` times its ``TOX_DECAY`` hook, at most ONE."""
    s = s or get_settings()
    return _fraction(unit_traits.scaled(u, unit_traits.TOX_DECAY,
                                        s.class_decay_q))


def harm_q(u, s: PoisonSettings | None = None) -> int:
    """THE hit points ``u`` loses per unit of poison leaving its blood (Q16):
    ``hp_per_unit`` times its ``TOX_HARM`` hook. int64: ``H_q < 2^28`` (the
    door) x ``HOOK_MAX = 2^19`` < 2^47."""
    s = s or get_settings()
    return unit_traits.scaled(u, unit_traits.TOX_HARM, s.class_harm_q)


# ---------------------------------------------------------------------------
# The law
# ---------------------------------------------------------------------------
def step(x: int, a: int, flush: bool = True) -> int:
    """What leaves a level ``x`` this tick at the Q16 fraction ``a``: 0 at 0;
    with ``flush``, all of it at or below :data:`FLUSH_Q`; else ``max(1, (x
    * a) >> 16)``. The law flushes only the BLOOD and only once the depot is
    empty (:func:`advance`): a level still being fed is never flushed."""
    if x <= 0:
        return 0
    if flush and x <= FLUSH_Q:
        return x
    m = (x * a) >> FP_SHIFT
    return m if m >= 1 else 1


def holds(u) -> bool:
    """Whether ``u`` holds any poison."""
    return bool(getattr(u, "poison_x0", 0) or getattr(u, "poison_x1", 0))


def dose(u, amount_q: int, source_id=NO_ATTACKER) -> int:
    """THE one way poison enters a unit: ``amount_q`` (Q16 units) onto its
    depot, ``source_id`` (the credited unit; -1 = the environment) recorded
    as ``poison_src`` (the latest dose wins). A zombie or a corpse takes
    nothing, and a dose of 0 writes nothing. The body holds at most
    :data:`X_MAX` in all: the excess is clipped and counted
    (:data:`SATURATION`). Returns the counts taken."""
    a = int(amount_q)
    if a <= 0 or getattr(u, "is_zombie", False) \
            or not getattr(u, "alive", True):
        return 0
    x0 = int(u.poison_x0)
    room = X_MAX - x0 - int(u.poison_x1)
    take = a if a <= room else max(0, room)
    if take < a:
        SATURATION["hits"] += 1
        SATURATION["dropped_q"] += a - take
    if take <= 0:
        return 0
    u.poison_x0 = x0 + take
    u.poison_src = attacker_value(source_id)
    return take


def advance(u, s: PoisonSettings | None = None):
    """One tick of the law on ``u``'s two levels (no packet): returns ``(m0,
    m1)``, what moved depot -> blood and what left the blood. Writes the
    levels, and ``poison_src`` back to -1 when both reach 0."""
    s = s or get_settings()
    x0 = int(u.poison_x0)
    x1 = int(u.poison_x1)
    m0 = step(x0, onset_q(u, s), flush=False)
    x0 -= m0
    x1 += m0
    m1 = step(x1, decay_q(u, s), flush=(x0 == 0))
    x1 -= m1
    u.poison_x0 = x0
    u.poison_x1 = x1
    if x0 == 0 and x1 == 0:
        u.poison_src = NO_ATTACKER
    return m0, m1


def tick(units, tick_no: int = 0, events=None) -> None:
    """Slot 9c3, after the gas row's doses: every living unit holding poison
    advances one tick (:func:`advance`), and what left its blood hurts as ONE
    POISON packet of ``mul_q16_round(m1, harm)`` Q16 points through
    ``damage.apply_packet`` (credited to the dose's ``poison_src``,
    indirectly; cause ``poison_gas``). Order-free: each unit reads only its
    own levels. ``tick_no`` is the sim tick (unused by the law; the device
    kernel's signature)."""
    s = get_settings()
    for u in units:
        if not (getattr(u, "poison_x0", 0) or getattr(u, "poison_x1", 0)):
            continue
        if not u.alive or u.is_zombie:
            clear(u)                    # never reached: death / turning clear
            continue
        src = int(u.poison_src)
        _m0, m1 = advance(u, s)
        if m1 <= 0:
            continue
        dmg_q = unit_fixed.mul_q16_round(m1, harm_q(u, s))
        if dmg_q <= 0:
            continue
        apply_packet(u, DamagePacket(                    # p3e-view: amount
            amount=unit_fixed.dequantize_scalar(dmg_q), dtype=POISON,
            attacker_id=src, indirect=True, amount_q=dmg_q),
            events, source="poison_gas")


def pending_harm_q(u, s: PoisonSettings | None = None) -> int:
    """The hit points (Q16, before mitigation) still to come from the poison
    ``u`` holds: ``(x0 + x1) x harm``. A read (the panel, the bench)."""
    s = s or get_settings()
    return unit_fixed.mul_q16_round(
        int(getattr(u, "poison_x0", 0)) + int(getattr(u, "poison_x1", 0)),
        harm_q(u, s))


def clear(u) -> None:
    """Zero both levels and the credit (death, conversion)."""
    if hasattr(u, "poison_x0"):
        u.poison_x0 = 0
        u.poison_x1 = 0
        u.poison_src = NO_ATTACKER


__all__ = ["SETTING_KEYS", "ONE", "X_MAX", "FLUSH_Q", "HP_PER_UNIT_MAX",
           "GAS_UNITS_MAX", "SATURATION", "fraction_q", "PoisonSettings",
           "settings_from_config", "get_settings", "rebuild_settings",
           "onset_q", "decay_q", "harm_q", "step", "holds", "dose",
           "advance", "tick", "pending_harm_q", "clear"]
