"""Stress -- a unit's one-level excitement and the heart rate read off it
(#215; ``docs/body_temperature_design_2026-11-17.md`` section 2b, version 4,
AGREED by Erik 2026-11-17).

One number per unit, ``stress_q`` (Q16 stress points; starts at 0, hashed
only while non-zero)::

    stress += what the producers add this tick
    stress -= drain(stress, rate)          rate from a half-life (2 minutes)

THE PRODUCERS are named rows (:data:`PRODUCERS`), each with its weight, and
every point of stress enters through ONE door, :func:`produce`. The first and
only producer is DAMAGE TAKEN: ``stress_per_hp`` points per hit point a
living unit actually loses, fed by ``damage.apply_packet`` (the one place hit
points are written) through :func:`on_damage`. A later producer (a near miss,
a team-mate dying in sight, a drug) is one more row and its own ruling. A
ZOMBIE takes none (like poison: the simplest honest form, Erik's to
overturn) and neither does a corpse; death and conversion clear the level
(:func:`clear`).

THE CLEARING is the shared step ``clearing.drain`` (the poison law's step,
lifted) at a rate in the finer :data:`clearing.RATE_SHIFT` currency
(``2^-24`` per tick: a 120 s half-life at 24 tps is 4038 counts, where Q16
would hold 16), with a flush of :data:`FLUSH_Q` (1/16 of a point) so the
level returns to EXACTLY 0. Cleared ONLY by its own tick (:func:`tick`, slot
9c3', after the poison law) -- nothing else lowers it.

THE HEART RATE is DERIVED, never stored: ONE Hill curve, ``rest + (max -
rest) x s^n / (half^n + s^n)``, an integer table built once at load from the
config's dials in exact rationals (:class:`HillCurve`; the sim path has no
powers and no floats), read through :func:`heart_rate_q` -- the one function
any later effect of stress reads, never the raw level. NOTHING in the game
reads the stress or the heart rate in this patch (Erik: "readout in first
patch is good, we wait with effects"): the tile inspector and the side panel
show them (``tests/test_stress.py`` scans ``src/`` for another reader).

INT64 BOUND: the level is held at most :data:`STRESS_MAX` (2^38 counts, four
million points; past it a production is clipped and counted in
:data:`SATURATION`), so ``x * rate < 2^38 x 2^24 = 2^62``. Device twin:
``cuda_unit_damage.cuh``'s ``apply_packet`` (the producer) and slot 9c3's
kernel (the tick), integer for integer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from simulation import clearing, unit_fixed

ONE = unit_fixed.FP_ONE
#: The section's keys, every one required.
SETTING_KEYS = ("stress_half_life_s", "stress_per_hp", "heart_rate_rest",
                "heart_rate_max", "heart_half_stress", "heart_hill_n")
#: The producer rows: name -> the ``[damage.stress]`` key of its weight. The
#: one door refuses any other name.
PRODUCERS = {"damage": "stress_per_hp"}
#: The most stress a unit holds (counts): the int64 bound of the law.
STRESS_MAX = 1 << 38
#: A level at or below this leaves in one tick (1/16 of a point).
FLUSH_Q = ONE // 16
#: The Hill table's resolution: entries per ``heart_half_stress``, and its
#: reach in halves (past it the last entry holds).
HILL_PER_HALF = 64
HILL_HALVES = 128
#: The largest weight (points per hit point) and Hill exponent the door takes.
WEIGHT_MAX = 1024.0
HILL_N_MAX = 8

#: Productions clipped at :data:`STRESS_MAX`: a diagnostic, never hashed.
SATURATION = {"hits": 0, "dropped_q": 0}


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        raise ValueError(f"{where} must be a number (got {value!r})")
    return float(value)


def _items(value):
    return value.items() if isinstance(value, dict) else vars(value).items()


@dataclass(frozen=True)
class HillCurve:
    """THE heart-rate curve as an integer table: entry ``i`` is the rate (Q16
    beats a minute) at stress ``i x half / HILL_PER_HALF``, ``rest + (max -
    rest) x i^n / (HILL_PER_HALF^n + i^n)`` rounded once (the stress cancels
    into the index: the table is exact rationals of the dials)."""
    half_q: int
    rest_q: int
    max_q: int
    n: int
    table: tuple

    @staticmethod
    def build(rest: float, top: float, half: float, n: int) -> "HillCurve":
        rest_q = unit_fixed.quantize_scalar(rest)               # p3e-door
        max_q = unit_fixed.quantize_scalar(top)                 # p3e-door
        half_q = unit_fixed.quantize_scalar(half)               # p3e-door
        span = max_q - rest_q
        base = HILL_PER_HALF ** n
        table = tuple(rest_q + unit_fixed.div_round(span * i ** n, base + i ** n)
                      for i in range(HILL_PER_HALF * HILL_HALVES + 1))
        return HillCurve(half_q=half_q, rest_q=rest_q, max_q=max_q, n=n,
                         table=table)

    def at(self, stress_q: int) -> int:
        """The rate at ``stress_q`` (Q16 points): the table interpolated
        linearly between its entries (floored), the last entry past its
        reach. Monotone, ``rest`` at 0, never above ``max``."""
        s = int(stress_q)
        if s <= 0:
            return self.table[0]
        pos = s * HILL_PER_HALF
        i, frac = divmod(pos, self.half_q)
        last = len(self.table) - 1
        if i >= last:
            return self.table[last]
        lo, hi = self.table[i], self.table[i + 1]
        return lo + ((hi - lo) * frac) // self.half_q


@dataclass(frozen=True)
class StressSettings:
    """The ``[damage.stress]`` dials, quantized once."""
    tps: int
    rate_q: int          # per-tick fraction, 2^-24 (clearing.RATE_SHIFT)
    weights_q: dict      # producer -> Q16 points per Q16 unit of its amount
    hill: HillCurve


def settings_from_config(cfg) -> StressSettings:
    """THE refusing door of ``[damage.stress]``: a missing or unknown key, a
    value that is not a number, a half-life below one tick, a weight outside
    [0, 1024), a curve whose rest is not below its max, a half stress not
    above 0 or an exponent that is not a whole number in [1, 8] is refused
    naming the key."""
    sec = getattr(getattr(cfg, "damage", None), "stress", None)
    if sec is None:
        raise ValueError("config has no [damage.stress] table "
                         f"(keys {list(SETTING_KEYS)})")
    cols = dict(_items(sec))
    for key in SETTING_KEYS:
        if key not in cols:
            raise ValueError(f"[damage.stress] is missing '{key}'")
    unknown = sorted(set(cols) - set(SETTING_KEYS))
    if unknown:
        raise ValueError(f"[damage.stress]: unknown key(s) {unknown} "
                         f"(it carries {list(SETTING_KEYS)})")
    v = {k: _number(cols[k], f"[damage.stress] {k}") for k in SETTING_KEYS}
    tps = int(cfg.clock.ticks_per_second)
    try:
        rate_q = clearing.half_life_rate(v["stress_half_life_s"], tps)
    except ValueError as e:
        raise ValueError(f"[damage.stress] stress_half_life_s: {e}") from None
    weights = {}
    for name, key in PRODUCERS.items():
        if not 0.0 <= v[key] < WEIGHT_MAX:
            raise ValueError(f"[damage.stress] {key} must be in [0, "
                             f"{WEIGHT_MAX:g}) (got {cols[key]!r})")
        weights[name] = unit_fixed.quantize_scalar(v[key])     # p3e-door
    if not v["heart_rate_rest"] < v["heart_rate_max"]:
        raise ValueError("[damage.stress] heart_rate_rest must be below "
                         "heart_rate_max")
    if not v["heart_half_stress"] > 0.0:
        raise ValueError("[damage.stress] heart_half_stress must be above 0")
    n = v["heart_hill_n"]
    if n != int(n) or not 1 <= int(n) <= HILL_N_MAX:
        raise ValueError(f"[damage.stress] heart_hill_n must be a whole "
                         f"number in [1, {HILL_N_MAX}] (got {cols['heart_hill_n']!r})")
    return StressSettings(
        tps=tps, rate_q=rate_q, weights_q=weights,
        hill=HillCurve.build(v["heart_rate_rest"], v["heart_rate_max"],
                             v["heart_half_stress"], int(n)))


_SETTINGS: StressSettings | None = None


def get_settings() -> StressSettings:
    """The shared settings, built from CFG on first use."""
    global _SETTINGS
    if _SETTINGS is None:
        rebuild_settings()
    return _SETTINGS


def rebuild_settings(cfg=None) -> StressSettings:
    """Rebuild from the live CFG (``Simulation`` construction)."""
    global _SETTINGS
    if cfg is None:
        from config import CFG as cfg
    _SETTINGS = settings_from_config(cfg)
    return _SETTINGS


# ---------------------------------------------------------------------------
# The door in, the tick, the clear
# ---------------------------------------------------------------------------
def produce(u, producer: str, amount_q: int,
            s: StressSettings | None = None) -> int:
    """THE one way stress enters a unit: producer row ``producer`` adds
    ``round(amount_q x weight)`` (Q16) to ``u.stress_q``. A zombie or a
    corpse takes nothing; nothing is written for an amount that adds nothing.
    The level holds at most :data:`STRESS_MAX` (the excess clipped and
    counted). Returns the counts added. Refuses an unknown producer by name."""
    s = s or get_settings()
    try:
        w = s.weights_q[producer]
    except KeyError:
        raise ValueError(f"stress: no producer row {producer!r} "
                         f"(known: {list(PRODUCERS)})") from None
    if getattr(u, "is_zombie", False) or not getattr(u, "alive", True):
        return 0
    add = unit_fixed.mul_q16_round(int(amount_q), w)
    if add <= 0:
        return 0
    x = int(getattr(u, "stress_q", 0))
    room = STRESS_MAX - x
    take = add if add <= room else max(0, room)
    if take < add:
        SATURATION["hits"] += 1
        SATURATION["dropped_q"] += add - take
    if take <= 0:
        return 0
    u.stress_q = x + take
    return take


def on_damage(u, dmg_q: int) -> int:
    """The DAMAGE producer: ``dmg_q`` Q16 hit points a living unit actually
    lost (``damage.apply_packet``, before its death check; a heal or a hit
    that took nothing adds nothing)."""
    if dmg_q <= 0:
        return 0
    return produce(u, "damage", dmg_q)


def tick(units) -> None:
    """Slot 9c3', after the poison law: every living, non-zombie unit holding
    stress loses ``clearing.drain(stress, rate, RATE_SHIFT, FLUSH_Q)``.
    Order-free (each unit reads only its own level). Dormant (one int read
    per unit) while nobody holds stress."""
    s = get_settings()
    for u in units:
        x = int(getattr(u, "stress_q", 0))
        if x == 0:
            continue
        if not u.alive or u.is_zombie:
            clear(u)                    # never reached: death / turning clear
            continue
        u.stress_q = x - clearing.drain(x, s.rate_q, clearing.RATE_SHIFT,
                                        FLUSH_Q)


def clear(u) -> None:
    """Zero the level (death, conversion)."""
    if hasattr(u, "stress_q"):
        u.stress_q = 0


# ---------------------------------------------------------------------------
# The heart rate: the one curve every later effect reads
# ---------------------------------------------------------------------------
def heart_rate_q(u_or_stress_q, s: StressSettings | None = None) -> int:
    """THE heart rate (Q16 beats a minute) of a unit (or of a stress level in
    Q16 points), read off the one Hill curve. Derived, never stored."""
    s = s or get_settings()
    x = u_or_stress_q if isinstance(u_or_stress_q, int) \
        else int(getattr(u_or_stress_q, "stress_q", 0))
    return s.hill.at(x)


__all__ = ["ONE", "SETTING_KEYS", "PRODUCERS", "STRESS_MAX", "FLUSH_Q",
           "HILL_PER_HALF", "HILL_HALVES", "SATURATION", "HillCurve",
           "StressSettings", "settings_from_config", "get_settings",
           "rebuild_settings", "produce", "on_damage", "tick", "clear",
           "heart_rate_q"]
