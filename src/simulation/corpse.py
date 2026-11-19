"""The corpse law -- infestation and integrity (arc #63 P5b;
``docs/swarm_P5b_impl.md``; Erik's rulings of 2026-11-14,
``docs/unit_design_2026-11-14.md`` section 7.4: "larvae on a corpse").

A dead unit carries two numbers, Q16 unit-schema columns, each hashed only
while it differs from its default (so a match without larvae hashes exactly as
before):

- ``infest_q`` -- its INFESTATION LEVEL (default 0; 1.0 = ``ONE``). Raised by
  the swarm's per-tile COUNT plane (``swarm.count_plane``: the living,
  non-coma swarm units on each tile this tick) summed over the corpse's
  footprint, decaying by a fraction of itself::

      level += n x rise_q - drain(level, decay_a)      (explicit Euler)

  ``rise_q`` = round(``infest_rate`` x ONE / tps): the level one swarm unit
  adds per tick, an exact rational rounded once; ``decay_a`` the per-tick
  fraction ``infest_decay`` / tps at ``clearing.RATE_SHIFT`` (2^-24, through
  :func:`clearing.rate_per_tick`), drained by THE clearing step
  (``clearing.drain``, no flush: the floor of one count brings a level back to
  exactly 0). The drain reads the level BEFORE this tick's rise. At the first
  numbers one larva settles near 0.4 (never raises a corpse), three near 1.2,
  ten near 4.

- ``corpse_integrity_q`` -- what is left of the body, in Q16 hit points
  (default, and set at every death: ``[corpse] integrity``). At or below 0 the
  corpse is DESTROYED for rising: its level is held at 0 and it never converts
  by either trigger. ACCEPTED GAP (docs/swarm_P5b_impl.md, the census): NO
  damage lowers it yet. Blasts, the heat row, statuses and poison stop at
  death (``alive`` gates), so a frag beside a corpse leaves it whole; the one
  reach to a corpse today (an attack slot's or the zombie slot's packet
  decided against its start state, landing after an earlier one killed)
  still writes the hp, because routing it to the integrity moves five of
  main's pinned matches (their unit streams) and no re-pin is approved.

RISING: :func:`rises` is the conversion door's second trigger
(``ai_zombie.convert_marines_to_zombies``, the ONE conversion path): a dead
unit whose level reached ``infest_threshold``, a zombie's corpse included, any
number of times. :func:`clear` at conversion resets both columns (level 0,
integrity full); :func:`on_death` sets them at death (``damage.apply_packet``).
A LIVING unit's level never moves (the law runs on corpses only, and both the
death and the conversion reset it).

THE ONE WRITER: nothing but this module writes the two columns
(``tests/test_corpse.py`` scans ``src/``). The swarm never reads them (the
boundary: the swarm writes the count plane, the unit side reads it through the
``swarm_count`` coupling row, ``exchange.apply_corpse_infestation``).

Integer only, no random draw. The dials are construction-bound
(:func:`rebuild_settings`, called by ``Simulation.__init__``; the poison
law's precedent). Device twin: slot 9c3's kernel (the infestation row, fused
into ``cpp/src/cuda_unit_exchange.cu``), the device ``apply_packet``'s death
branch and the device conversion (``cpp/src/cuda_unit_conversion.cu``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from simulation import clearing, unit_fixed

ONE = unit_fixed.FP_ONE
#: The ``[corpse]`` keys, every one required.
SETTING_KEYS = ("infest_rate", "infest_decay", "infest_threshold", "integrity")
#: The level's proved range (Q16): the law refuses a level or a rise ``n x
#: rise_q`` past 2^38, so the drain's product stays below 2^38 x 2^24 = 2^62
#: (``decay_a`` is at most ``clearing.RATE_ONE``). Never reached in play: a
#: level converts at ``infest_threshold`` <= 1024 (2^26).
LEVEL_MAX = 1 << 38
#: The door's bounds on the authored values.
RATE_MAX = 1024.0            # level per swarm unit per second
DECAY_MAX = 64.0             # per second (the per-tick fraction is also held <= 1)
THRESHOLD_MAX = 1024.0
INTEGRITY_MAX = 32767.0      # hit points


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"{where} must be a number (got {value!r})")
    return float(value)


def _items(value):
    return value.items() if isinstance(value, dict) else vars(value).items()


@dataclass(frozen=True)
class CorpseSettings:
    """The ``[corpse]`` dials, quantized once."""
    tps: int
    rise_q: int        # Q16 level one swarm unit adds per tick
    decay_a: int       # the level's per-tick fraction, 2^-24 (clearing.RATE_SHIFT)
    threshold_q: int   # Q16 level at which a corpse rises
    integrity_q: int   # Q16 hit points a fresh corpse holds


def settings_from_config(cfg) -> CorpseSettings:
    """THE refusing door of ``[corpse]``: a missing or unknown key, a value
    that is not a number or outside its range (``infest_rate`` in [0, 1024),
    ``infest_decay`` in [0, 64) with a per-tick fraction at most one,
    ``infest_threshold`` in (0, 1024], ``integrity`` in (0, 32767]) is refused
    naming the key."""
    sec = getattr(cfg, "corpse", None)
    if sec is None:
        raise ValueError(f"config has no [corpse] table (keys {list(SETTING_KEYS)})")
    cols = dict(_items(sec))
    for key in SETTING_KEYS:
        if key not in cols:
            raise ValueError(f"[corpse] is missing '{key}'")
    unknown = sorted(set(cols) - set(SETTING_KEYS))
    if unknown:
        raise ValueError(f"[corpse]: unknown key(s) {unknown} "
                         f"(it carries {list(SETTING_KEYS)})")
    v = {k: _number(cols[k], f"[corpse] {k}") for k in SETTING_KEYS}
    tps = int(cfg.clock.ticks_per_second)
    if not 0.0 <= v["infest_rate"] < RATE_MAX:
        raise ValueError(f"[corpse] infest_rate must be in [0, {RATE_MAX:g}) "
                         f"(got {cols['infest_rate']!r})")
    if not 0.0 <= v["infest_decay"] < DECAY_MAX:
        raise ValueError(f"[corpse] infest_decay must be in [0, {DECAY_MAX:g}) "
                         f"(got {cols['infest_decay']!r})")
    if not 0.0 < v["infest_threshold"] <= THRESHOLD_MAX:
        raise ValueError(f"[corpse] infest_threshold must be in (0, "
                         f"{THRESHOLD_MAX:g}] (got {cols['infest_threshold']!r})")
    if not 0.0 < v["integrity"] <= INTEGRITY_MAX:
        raise ValueError(f"[corpse] integrity must be in (0, {INTEGRITY_MAX:g}] "
                         f"(got {cols['integrity']!r})")
    decay_a = clearing.rate_per_tick(v["infest_decay"], tps)
    if decay_a > clearing.RATE_ONE:
        raise ValueError(f"[corpse] infest_decay {cols['infest_decay']!r} drains "
                         f"more than the whole level a tick at {tps} tps")
    r = Fraction(v["infest_rate"]) * ONE / tps          # p3e-door: exact rational
    return CorpseSettings(
        tps=tps,
        rise_q=unit_fixed.div_round(r.numerator, r.denominator),
        decay_a=decay_a,
        threshold_q=unit_fixed.quantize_scalar(v["infest_threshold"]),
        integrity_q=unit_fixed.quantize_scalar(v["integrity"]))


_SETTINGS: CorpseSettings | None = None


def get_settings() -> CorpseSettings:
    """The shared settings, built from CFG on first use."""
    global _SETTINGS
    if _SETTINGS is None:
        rebuild_settings()
    return _SETTINGS


def rebuild_settings(cfg=None) -> CorpseSettings:
    """Rebuild from the live CFG (``Simulation`` construction)."""
    global _SETTINGS
    if cfg is None:
        from config import CFG as cfg
    _SETTINGS = settings_from_config(cfg)
    return _SETTINGS


def full_integrity_q() -> int:
    """A fresh corpse's integrity (the column's default)."""
    return get_settings().integrity_q


# ---------------------------------------------------------------------------
# The law
# ---------------------------------------------------------------------------
class CorpseLevelRange(ValueError):
    """A level or a rise past :data:`LEVEL_MAX` (the int64 bound): refused by
    name, never clamped."""


def destroyed(u) -> bool:
    """A corpse whose integrity is gone: it never rises again."""
    return int(u.corpse_integrity_q) <= 0


def infest(u, n: int, s: CorpseSettings | None = None) -> None:
    """One tick of the level of DEAD unit ``u`` with ``n`` swarm units under
    its footprint: ``level += n x rise_q - drain(level, decay_a)`` (the drain
    on the level before the rise). A destroyed corpse's level is held at 0."""
    s = s or get_settings()
    if destroyed(u):
        if u.infest_q:
            u.infest_q = 0
        return
    x = int(u.infest_q)
    gain = int(n) * s.rise_q
    if not 0 <= x <= LEVEL_MAX or not 0 <= gain <= LEVEL_MAX:
        raise CorpseLevelRange(f"corpse {getattr(u, 'id', '?')}: level {x} or "
                               f"rise {gain} outside [0, 2^38]")
    nx = x - clearing.drain(x, s.decay_a, clearing.RATE_SHIFT) + gain
    if nx != x:
        u.infest_q = nx


def rises(u, s: CorpseSettings | None = None) -> bool:
    """The conversion door's infestation trigger: a dead, not destroyed unit
    whose level reached ``infest_threshold`` (a zombie's corpse included)."""
    s = s or get_settings()
    return (not u.alive and not destroyed(u)
            and int(u.infest_q) >= s.threshold_q)


def on_death(u) -> None:
    """At death (``damage.apply_packet``): the level at 0 and the integrity
    full -- both the defaults, so a death hashes exactly as before."""
    if hasattr(u, "infest_q"):
        u.infest_q = 0
        u.corpse_integrity_q = full_integrity_q()


def clear(u) -> None:
    """At conversion: the level back to 0 (the rise is spent) and the
    integrity full for the next death."""
    on_death(u)


__all__ = ["ONE", "SETTING_KEYS", "LEVEL_MAX", "CorpseSettings",
           "settings_from_config", "get_settings", "rebuild_settings",
           "full_integrity_q", "CorpseLevelRange", "destroyed", "infest",
           "rises", "on_death", "clear"]
