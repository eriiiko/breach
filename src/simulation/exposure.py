"""Exposure -- the focus-fire law (gameplay pack v1 G4p, issue #181).

Hits lower a target's defences, the lowering decays, and several shooters
lower it further with no record of who shot
(``docs/gameplay_pack_v1_design_2026-11-10.md`` §3 G4p). Every unit carries
ONE decaying level per damage type, ``unit.exposure`` -- a fixed block of
``N_DAMAGE_TYPES`` Q16 ints, all 0 on a fresh unit -- and the law is three
integer lines:

    on each hit of type T that reaches a living unit:
        x[T] = min(x_max, x[T] + (expose * (ONE - expose_resist[T]) >> 16))
    each tick (slot 1c, before any attack):
        x[T] = max(0, x[T] - max(1, (x[T] * decay) >> 16))
    damage of type T taken (in ``damage.apply_packet``, after mitigation):
        dmg_q = dmg_q * (ONE + (k_expose * x[T] >> 16)) >> 16

``expose`` is the round's ``[ammo.*] expose`` column (``DamagePacket.expose``;
0 = the round exposes nothing), ``expose_resist`` the target's
(``[env_profiles.*] expose_resist``, the zombie's ``[zombie] expose_resist``:
1 = immune), ``decay`` / ``k_expose`` / ``x_max`` the ``[damage.exposure]``
dials. ``x`` is dimensionless: with ``k_expose = 1`` an exposure of 1.0 doubles
the damage of its type. HEAL is outside the law (mitigation leaves it alone
too): it is never scaled. ``k_expose = 0`` makes the law DORMANT: nothing is
raised (a level that scales nothing is not kept), so the damage and every
digest are exactly what they were before G4p.

THE START-OF-SLOT RULE (rule 5 of the pack): inside an attack slot
(``SimultaneousEffects.resolve``) every hit reads the exposure its target had
when the slot began, and the raises of all the slot's hits land TOGETHER at
the end of the resolve -- an :class:`ExposureLanding` collects them (and the
deaths, whose zeroing waits for the same moment), so two shooters in one tick
take the same damage per round whatever their ids. A packet applied outside a
slot (the zombie bite, the exchange rows, a status's DoT) raises and zeroes at
once; in v1 none of those carries an ``expose``.

Integer only, no random draw. The block is zeroed at death and at conversion
to a zombie, and hashed in the per-tick unit digest only while non-zero
(``tests/field_ab_harness._unit_record``), so a match in which nothing exposes
hashes as before. Not a status row: a status has a duration and a fixed
magnitude, this is one decaying level.

The dials are construction-bound (:func:`rebuild_settings`, called by
``Simulation.__init__``; the weapon tables' precedent): a config edit applies
to the next Simulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config import ticks_from_seconds
from simulation.damage import DAMAGE_TYPE_NAMES, HEAL, N_DAMAGE_TYPES
from simulation.unit_fixed import FP_ONE, FP_SHIFT, quantize_scalar

#: The ``[damage.exposure]`` keys, every one required.
SETTING_KEYS = ("half_life_seconds", "k_expose", "x_max")

_TYPE_BY_NAME = {name: t for t, name in DAMAGE_TYPE_NAMES.items()}


# ---------------------------------------------------------------------------
# The doors
# ---------------------------------------------------------------------------
def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"{where} must be a number (got {value!r})")
    return float(value)


def _items(value):
    return value.items() if isinstance(value, dict) else vars(value).items()


def resist_block(value, where: str) -> tuple:
    """An ``expose_resist`` column as the fixed block of Q16 ints, one per
    damage type: a scalar applies to every type, a table names types by their
    ``damage.DAMAGE_TYPE_NAMES`` name (an absent type is 0). Refused by name
    outside [0, 1], on an unknown type, or on a value that is not a number."""
    if isinstance(value, dict) or hasattr(value, "__dict__"):
        block = [0] * N_DAMAGE_TYPES
        for name, v in _items(value):
            if name not in _TYPE_BY_NAME:
                raise ValueError(
                    f"{where}.{name} names no damage type "
                    f"(known: {sorted(_TYPE_BY_NAME)})")
            block[_TYPE_BY_NAME[name]] = _resist_q(v, f"{where}.{name}")
        return tuple(block)
    return (_resist_q(value, where),) * N_DAMAGE_TYPES


def _resist_q(value, where: str) -> int:
    v = _number(value, where)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{where} must be in [0, 1] (got {value!r}; "
                         f"0 = fully exposed, 1 = immune)")
    return quantize_scalar(v)


def expose_q(value, where: str) -> int:
    """An ``[ammo.*] expose`` value as its Q16 raise per hit; refused by name
    when negative or not a number."""
    v = _number(value, where)
    if v < 0.0:
        raise ValueError(f"{where} must be >= 0 (got {value!r}; 0 = the "
                         f"round exposes nothing)")
    return quantize_scalar(v)


NO_RESIST = (0,) * N_DAMAGE_TYPES


def _decay_q(half_life_ticks: int) -> int:
    """The per-tick Q16 decay fraction: the SMALLEST ``d`` for which the decay
    line takes an exposure of ONE to at most ONE/2 within ``half_life_ticks``
    ticks -- found by integer bisection over the law itself (monotone in
    ``d``), so the dial means exactly what the integer step does and no libm
    root is taken."""
    def after(d):
        x = FP_ONE
        for _ in range(half_life_ticks):
            x = decay_step(x, d)
        return x
    lo, hi = 1, FP_ONE
    while lo < hi:
        mid = (lo + hi) // 2
        if after(mid) <= FP_ONE // 2:
            hi = mid
        else:
            lo = mid + 1
    return lo


@dataclass(frozen=True)
class ExposureSettings:
    """The ``[damage.exposure]`` dials, quantized once (and the zombie's
    resistance, which is the zombie STATE's profile column)."""
    half_life_ticks: int
    decay_q: int                 # Q16 fraction of x lost per tick
    k_q: int                     # Q16 extra damage per unit of exposure
    x_max_q: int                 # Q16 cap
    zombie_resist: tuple = NO_RESIST


def settings_from_config(cfg) -> ExposureSettings:
    """THE refusing door of ``[damage.exposure]`` (+ ``[zombie]
    expose_resist``, optional, default 0): a missing or unknown key, a value
    that is not a number, ``half_life_seconds <= 0``, ``k_expose < 0`` or
    ``x_max <= 0`` is refused naming the key."""
    sec = getattr(getattr(cfg, "damage", None), "exposure", None)
    if sec is None:
        raise ValueError("config has no [damage.exposure] table "
                         f"(keys {list(SETTING_KEYS)})")
    cols = dict(_items(sec))
    for key in SETTING_KEYS:
        if key not in cols:
            raise ValueError(f"[damage.exposure] is missing '{key}'")
    unknown = sorted(set(cols) - set(SETTING_KEYS))
    if unknown:
        raise ValueError(f"[damage.exposure]: unknown key(s) {unknown} "
                         f"(it carries {list(SETTING_KEYS)})")
    half = _number(cols["half_life_seconds"],
                   "[damage.exposure] half_life_seconds")
    k = _number(cols["k_expose"], "[damage.exposure] k_expose")
    x_max = _number(cols["x_max"], "[damage.exposure] x_max")
    if not half > 0:
        raise ValueError(f"[damage.exposure] half_life_seconds must be > 0 "
                         f"(got {half!r})")
    if k < 0:
        raise ValueError(f"[damage.exposure] k_expose must be >= 0 "
                         f"(got {k!r})")
    if not x_max > 0:
        raise ValueError(f"[damage.exposure] x_max must be > 0 "
                         f"(got {x_max!r})")
    tps = int(cfg.clock.ticks_per_second)
    n_half = ticks_from_seconds(half, tps)
    zombie = getattr(cfg, "zombie", None)
    zr = getattr(zombie, "expose_resist", None) if zombie is not None else None
    if isinstance(zombie, dict):
        zr = zombie.get("expose_resist")
    return ExposureSettings(
        half_life_ticks=n_half, decay_q=_decay_q(n_half),
        k_q=quantize_scalar(k), x_max_q=quantize_scalar(x_max),
        zombie_resist=(NO_RESIST if zr is None
                       else resist_block(zr, "[zombie] expose_resist")))


_SETTINGS: ExposureSettings | None = None


def get_settings() -> ExposureSettings:
    """The shared settings, built from CFG on first use."""
    global _SETTINGS
    if _SETTINGS is None:
        rebuild_settings()
    return _SETTINGS


def rebuild_settings(cfg=None) -> ExposureSettings:
    """Rebuild from the live CFG (``Simulation`` construction)."""
    global _SETTINGS
    if cfg is None:
        from config import CFG as cfg
    _SETTINGS = settings_from_config(cfg)
    return _SETTINGS


# ---------------------------------------------------------------------------
# The law
# ---------------------------------------------------------------------------
def new_block() -> list:
    """A fresh unit's block: every type at 0."""
    return [0] * N_DAMAGE_TYPES


def decay_step(x: int, decay_q: int) -> int:
    """One tick of decay: ``max(0, x - max(1, (x * decay_q) >> 16))``."""
    if x <= 0:
        return 0
    return max(0, x - max(1, (x * decay_q) >> FP_SHIFT))


def decay_units(units) -> None:
    """Slot 1c: every unit's block decays one tick (a zero block is skipped)."""
    d = get_settings().decay_q
    for u in units:
        block = getattr(u, "exposure", None)
        if block and any(block):
            for t in range(N_DAMAGE_TYPES):
                if block[t]:
                    block[t] = decay_step(block[t], d)


def resist_for(unit) -> tuple:
    """The target's ``expose_resist`` block: the zombie STATE's
    (``[zombie] expose_resist``, selected on ``is_zombie`` at hit time like
    ``damage.mitigation_for``), else its environment profile's."""
    if getattr(unit, "is_zombie", False):
        return get_settings().zombie_resist
    env = getattr(unit, "environment", None)
    return getattr(env, "expose_resist", NO_RESIST) or NO_RESIST


def scale_q(dmg_q: int, unit, dtype: int) -> int:
    """The mitigated Q16 damage ``dmg_q`` of type ``dtype`` scaled by the
    target's exposure: ``dmg_q * (ONE + (k * x >> 16)) >> 16`` -- the
    identity, exactly, when the exposure or ``k_expose`` is 0 (and for HEAL)."""
    if dtype == HEAL or dmg_q <= 0:
        return dmg_q
    block = getattr(unit, "exposure", None)
    x = block[dtype] if block else 0
    if x <= 0:
        return dmg_q
    gain = (get_settings().k_q * x) >> FP_SHIFT
    if gain <= 0:
        return dmg_q
    return (dmg_q * (FP_ONE + gain)) >> FP_SHIFT


def raise_amount(unit, packet) -> int:
    """The Q16 exposure ``packet`` adds to its type on ``unit`` (before the
    cap): ``expose * (ONE - resist[T]) >> 16``. 0 while ``k_expose`` is 0:
    a level that scales nothing is not kept (the law is then dormant, and
    every digest is what it was with no exposure at all)."""
    e = int(getattr(packet, "expose", 0) or 0)
    if e <= 0 or packet.dtype == HEAL or get_settings().k_q == 0:
        return 0
    return (e * (FP_ONE - resist_for(unit)[packet.dtype])) >> FP_SHIFT


def _raise(unit, dtype: int, add: int) -> None:
    block = getattr(unit, "exposure", None)
    if block is None or add <= 0:
        return
    block[dtype] = min(get_settings().x_max_q, block[dtype] + add)


def clear(unit) -> None:
    """Zero the block (death, conversion)."""
    block = getattr(unit, "exposure", None)
    if block:
        for t in range(len(block)):
            block[t] = 0


class ExposureLanding:
    """The raises and deaths of ONE attack slot's phase B, landed together by
    :meth:`land` (the start-of-slot rule): until then every hit of the slot
    reads the exposure its target had when the slot began. Additions commute
    (a sum, then the cap) and a death zeroes, so the order they were recorded
    in cannot matter."""

    __slots__ = ("_raises", "_dead")

    def __init__(self):
        self._raises = []
        self._dead = []

    def on_hit(self, unit, packet) -> None:
        add = raise_amount(unit, packet)
        if add > 0:
            self._raises.append((unit, packet.dtype, add))

    def on_death(self, unit) -> None:
        self._dead.append(unit)

    def land(self) -> None:
        raises, self._raises = self._raises, []
        dead, self._dead = self._dead, []
        for unit, dtype, add in raises:
            if getattr(unit, "alive", True):
                _raise(unit, dtype, add)
        for unit in dead:
            clear(unit)


def on_hit(unit, packet, landing=None) -> None:
    """A packet reached ``unit`` alive: raise its exposure now, or at the
    slot's landing."""
    if landing is not None:
        landing.on_hit(unit, packet)
    elif getattr(unit, "alive", True):
        _raise(unit, packet.dtype, raise_amount(unit, packet))


def on_death(unit, landing=None) -> None:
    """``unit`` died: its block is zeroed now, or at the slot's landing."""
    if landing is not None:
        landing.on_death(unit)
    else:
        clear(unit)


__all__ = ["SETTING_KEYS", "ExposureSettings", "settings_from_config",
           "get_settings", "rebuild_settings", "resist_block", "expose_q",
           "NO_RESIST", "new_block", "decay_step", "decay_units",
           "resist_for", "scale_q", "raise_amount", "clear",
           "ExposureLanding", "on_hit", "on_death"]
