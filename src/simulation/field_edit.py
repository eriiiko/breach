"""FieldEdit — the canonical *write* primitive for GameMap fields.

Engine canon: ``docs/architecture/engine/13_field_edit.md`` (Depends on: 01 grid,
02 state, 03 materials). This is the third leg of a pattern the engine already
names twice — ``wind = -grad(p)`` is the canonical *read* primitive ("one field,
many readers", engine/04); the DDA march is "one primitive, two consumers"
(engine/08); **FieldEdit is the canonical *write* primitive** — "many systems
write many fields through one operator."

Three call sites used to mutate physics fields ad hoc, each re-deriving the same
disc/line loop with a different sign and clamp (``apply_explosion``'s
``smoke=0`` / ``atmosphere +=`` / ``wave_source +=`` / ``fire = max(...)`` and
``add_explosion_smoke``'s noisy disc deposit). All of them are the same
operation: *take a field, a region, an amount; combine into the field with a
mode and a falloff.* This module is that operation, written once.

Three composing parts:

* :class:`FieldEdit` — a pure, frozen description of one edit (field, region,
  coords, amount, mode, falloff, channel, clamp, noise, source_id).
* :func:`apply_field_edit` — the ONLY code that writes a field through this
  path. :func:`_iter_region` yields ``(row, col, weight)`` (the disc/beam/rect
  loop, written once); :func:`_combine` does float ``+=`` / ``-=`` / ``max`` for
  float fields and a Q16.16 saturating branch for the ``heat`` field (never a
  float ``+=`` on ``heat``).
* :class:`EditQueue` — consumers ``enqueue`` during the tick; ``flush`` applies
  EVERY queued edit in a **stable sort** by ``(field, source_id, region, seq)``,
  so the applied order is identical on every machine regardless of enqueue
  order. For Level-2 lockstep this order-independence is the whole determinism
  story: two grenades overlapping a tile give ``clamp(clamp(s+a)+b)`` — the
  moment a clamp / MAX is involved the result is order-dependent, so the order
  must be fixed. ``noise > 0`` draws its per-tile multiplier from the seeded
  ``sim.rng`` at flush time, in sorted order, so the flush is the single RNG
  consumer (the seeded-rollout guarantee is structural, not a per-caller
  convention).

Per-field policy (dtype + default clamp + skip-mask) lives in
:data:`FIELD_POLICY`, so consumers stop needing to know these rules: a smoke
edit never writes a solid tile, a fire edit never writes a non-flammable tile,
and ``heat`` always goes through the fixed-point branch.

DEFERRED (designed, not built here): ``wall_hp`` damage as a REMOVE FieldEdit
plus the post-flush ``<= 0`` destruction sweep (lands with the fire phase); the
planned laser emitters; the ``SET`` and ``MIN`` modes (added when a consumer
needs them). See the canon chapter. (The gas emitter LANDED with weapons W3:
``field="gas"`` + ``channel=<slice id>`` targets one plane of the (N, h, w)
multi-gas array — simulation.payloads.emit_gas is the consumer.)
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np

from simulation import water_fixed   # S1: water_depth Q16.16 quantize helpers
from simulation import wave_fixed     # S2a: wave_source Q16.16 quantize helpers
from simulation import gas_fixed      # S2b: smoke/gas Q16.16 quantize helpers
from simulation import atmosphere_fixed  # S2c: atmosphere Q16.16 quantize helpers
from simulation import fire_fixed      # S3a: fire Q16.16 quantize helpers

# Q16.16 fixed-point scale for the `heat` field — MUST match the C++
# raycaster.h HEAT_SCALE (and materials.TEMP_SCALE). One unit of heat energy ==
# HEAT_SCALE raw int32 counts. Mirrored here (not imported from C++) so the
# Python write path is self-contained and testable headless.
HEAT_SCALE = 65536
_INT32_MAX = np.iinfo(np.int32).max


# ---------------------------------------------------------------------------
# Q16.16 helpers — Python mirrors of cpp/src/raycaster.h heat_quantize /
# heat_saturating_add. Same semantics: round-to-nearest, saturate at INT32_MAX,
# never wrap, never go negative on a positive accumulator. The `heat` field is
# written ONLY through these — never a float `+=`.
# ---------------------------------------------------------------------------
def heat_quantize(energy: float) -> int:
    """Saturating quantize float heat energy -> Q16.16 int32 (round-to-nearest).

    Mirrors the C++ ``heat_quantize``: ``<= 0`` returns 0; values whose scaled
    magnitude would exceed ``INT32_MAX`` clamp to ``INT32_MAX``; otherwise
    round-to-nearest (``+ 0.5`` then truncate, valid since input is positive).
    """
    if energy <= 0.0:
        return 0
    scaled = float(energy) * float(HEAT_SCALE)
    if scaled >= float(_INT32_MAX):
        return int(_INT32_MAX)
    return int(scaled + 0.5)


def heat_saturating_add(cell: int, delta: int) -> int:
    """Saturating add into a Q16.16 accumulator: clamp at INT32_MAX, never wrap.

    Mirrors the C++ ``heat_saturating_add``: a non-positive ``delta`` is a no-op
    (the heat deposit is additive-only); otherwise add, clamping at
    ``INT32_MAX``. Returns the new cell value (Python ints are unbounded, so the
    clamp is explicit, not a wrap guard).
    """
    cell = int(cell)
    delta = int(delta)
    if delta <= 0:
        return cell
    if cell > _INT32_MAX - delta:
        return int(_INT32_MAX)
    return cell + delta


# ---------------------------------------------------------------------------
# Enums. IntEnum so the stable-sort key on `region` / `mode` is a plain int
# compare (deterministic, no reliance on Enum member ordering quirks).
# ---------------------------------------------------------------------------
class EditMode(IntEnum):
    """How a contribution combines into the existing field value.

    Only the three modes a live consumer needs today. ``SET`` (lerp-to-value)
    and ``MIN`` are deferred until a consumer requires them (designed in the
    canon chapter).
    """
    ADD = 0       # field += contribution           (deposits: smoke, pressure, heat)
    REMOVE = 1    # field -= contribution           (burn-off, clearing)
    MAX = 2       # field = max(field, contribution) (ignite: never lower an existing fire)


class Region(IntEnum):
    """The set of tiles an edit touches, plus the per-tile falloff weight."""
    TILE = 0      # coords = (r, c)                       — a single cell, weight 1
    DISC = 1      # coords = (r, c, radius)               — a filled disc
    BEAM = 2      # coords = (r0, c0, r1, c1, width)      — a thick line (laser)
    RECT = 3      # coords = (r0, c0, r1, c1)             — an axis-aligned box


class Falloff(IntEnum):
    """Per-tile weight profile within a region (multiplies ``amount``)."""
    FLAT = 0      # weight = 1 everywhere
    LINEAR = 1    # weight = 1 - dist/radius  (== today's explosion falloff)


# ---------------------------------------------------------------------------
# FieldEdit — a frozen description of one edit. Pure data, no state.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldEdit:
    """One immutable edit: combine ``amount`` (× falloff weight × optional noise)
    into ``field`` over ``region`` using ``mode``.

    Frozen so an edit can be safely queued, hashed for debugging, and sorted
    without any consumer mutating it after enqueue.
    """
    field: str                                   # "smoke" / "atmosphere" / "wave_source" / "fire" / "heat" / ...
    region: Region
    coords: Tuple                                # TILE:(r,c) · DISC:(r,c,rad) · BEAM:(r0,c0,r1,c1,w) · RECT:(r0,c0,r1,c1)
    amount: float
    mode: EditMode = EditMode.ADD
    falloff: Falloff = Falloff.FLAT
    channel: Optional[int] = None                # None = scalar field; 0/1/2 = R/G/B of an (h,w,3) field
    clamp: Optional[Tuple[float, float]] = None  # post-combine clamp; None = use the field policy default
    noise: float = 0.0                           # >0 = per-tile multiplier in [1-noise, 1], drawn from sim.rng
    source_id: int = 0                           # bookkeeping + stable-sort grouping (one emitter = one id)


# ---------------------------------------------------------------------------
# Per-field policy table (engine/02 ownership). Declares, per field:
#   dtype        — "float" (float += / max) or "heat" (Q16.16 saturating branch)
#   clamp        — default post-combine clamp (None = unbounded) used when the
#                  FieldEdit itself sets no clamp
#   skip         — a callable (gmap) -> bool mask of tiles this field MUST NOT
#                  write (a per-cell veto), so consumers stop knowing these rules
#
# skip-mask semantics mirror the legacy ad-hoc sites exactly:
#   smoke       skip solid                       (gases don't enter walls)
#   atmosphere  skip solid                       (bulk field lives in open cells)
#   wave_source skip solid + is_vacuum           (no shockwave source in walls/space)
#   fire        skip non-flammable               (only flammable tiles can burn)
#   heat        (none — heat deposits everywhere the ray reaches)
# ---------------------------------------------------------------------------
def _skip_solid(gmap):
    return gmap.solid


def _skip_solid_or_vacuum(gmap):
    return gmap.solid | gmap.is_vacuum


def _skip_non_flammable(gmap):
    return ~gmap.flammable


@dataclass(frozen=True)
class _FieldPolicy:
    dtype: str
    clamp: Optional[Tuple[float, float]]
    skip: Optional[object]   # callable(gmap) -> bool mask, or None


FIELD_POLICY = {
    # S2b: smoke is int32 Q16.16. The "gas" dtype dequantizes the stored int to
    # real density, combines (the float path's exact +=/-=/max semantics + the
    # [0,1] clamp), then re-quantizes round-to-nearest — so smoke/gas edits (the
    # explosion deposit + clear) keep being authored in real density while the
    # stored field stays integer/deterministic (same idiom as "wave" / "water").
    "smoke":       _FieldPolicy("gas",   (0.0, 1.0), _skip_solid),
    # W3 (mechanics/03 §4 gas payloads): the full multi-gas array (engine/05
    # §6.2). ``gmap.gas`` is (N, h, w) int32 Q16.16 — one slice per gas type;
    # a "gas" edit names its slice via ``FieldEdit.channel`` (the GAS_* id /
    # ``gmap.gases.name_to_id``), resolved to a contiguous (h, w) view at
    # apply time. Same combine + policy as "smoke" (which IS the BLACK_SMOKE
    # slice): solid skip-mask, [0, 1] saturation clamp, quantize-once at the
    # write boundary.
    "gas":         _FieldPolicy("gas",   (0.0, 1.0), _skip_solid),
    # S2c: atmosphere is int32 Q16.16. The "atmosphere" dtype dequantizes the
    # stored int to real pressure, combines (the float path's exact +=/-=/max
    # semantics), then re-quantizes round-to-nearest — so atmosphere edits (the
    # explosion pressure boost) keep being authored in real units while the
    # stored field stays integer/deterministic (same idiom as "wave" / "gas").
    "atmosphere":  _FieldPolicy("atmosphere", None,  _skip_solid),
    # S2a: wave_source is int32 Q16.16. The "wave" dtype dequantizes the stored
    # int to real units, combines (the float path's exact +=/max semantics), then
    # re-quantizes round-to-nearest — so wave-source edits (the explosion deposit)
    # keep being authored in real units while the stored field stays integer.
    "wave_source": _FieldPolicy("wave", None,        _skip_solid_or_vacuum),
    # S3a: fire is int32 Q16.16. The "fire" dtype dequantizes the stored int to
    # real intensity, combines (the float path's exact +=/-=/max semantics + the
    # [0,1] clamp), then re-quantizes round-to-nearest — so the explosion's
    # `fire = max(...)` ignite deposit keeps being authored in real intensity
    # while the stored field stays integer/deterministic (same idiom as "gas").
    "fire":        _FieldPolicy("fire",  (0.0, 1.0), _skip_non_flammable),
    "heat":        _FieldPolicy("heat",  None,       None),
    # S1: water_depth is int32 Q16.16 metres. The "water" dtype dequantizes the
    # stored int to metres, combines in metres (with the metres clamp), then
    # re-quantizes round-to-nearest — so field edits keep being authored in
    # metres while the stored field stays integer/deterministic.
    "water_depth": _FieldPolicy("water", (0.0, float("inf")), _skip_solid),
}


def _policy(field: str) -> _FieldPolicy:
    try:
        return FIELD_POLICY[field]
    except KeyError as exc:  # pragma: no cover - guard against typos / new fields
        raise KeyError(
            f"FieldEdit: no policy registered for field {field!r}. "
            f"Add it to FIELD_POLICY (dtype, default clamp, skip-mask)."
        ) from exc


# ---------------------------------------------------------------------------
# Region iteration — the disc / beam / rect loop, written ONCE.
# Yields (row, col, weight) in a DETERMINISTIC row-major order. The weight is
# the falloff multiplier in [0, 1]; the caller multiplies amount × weight ×
# optional noise. In-bounds is the caller's job? No — we clip to the grid here
# so every consumer is bounds-safe.
# ---------------------------------------------------------------------------
def _iter_region(region: Region, coords: Tuple, falloff: Falloff, shape):
    """Yield ``(row, col, weight)`` for every in-bounds tile of ``region``.

    Order is row-major (ascending row, then ascending col) and identical on
    every machine — this is what makes the noise RNG draw sequence (one draw per
    surviving tile, at flush) deterministic. ``weight`` applies ``falloff``:
    FLAT -> 1; LINEAR -> ``1 - dist/radius`` (clamped to [0, 1]).

    DISC uses a STRICT ``dist < radius`` membership (matching the legacy
    ``add_explosion_smoke`` predicate); the ``dist == radius`` ring has weight 0
    under LINEAR anyway, so excluding it changes no additive/max result while
    keeping the deterministic per-tile RNG draw count exact.
    """
    h, w = shape

    if region == Region.TILE:
        r, c = int(coords[0]), int(coords[1])
        if 0 <= r < h and 0 <= c < w:
            yield r, c, 1.0
        return

    if region == Region.DISC:
        cr, cc, radius = int(coords[0]), int(coords[1]), float(coords[2])
        r0 = max(0, int(np.floor(cr - radius)))
        r1 = min(h - 1, int(np.ceil(cr + radius)))
        c0 = max(0, int(np.floor(cc - radius)))
        c1 = min(w - 1, int(np.ceil(cc + radius)))
        for r in range(r0, r1 + 1):
            dr = r - cr
            for c in range(c0, c1 + 1):
                dc = c - cc
                dist = float(np.sqrt(dr * dr + dc * dc))
                if radius > 0.0 and dist < radius:
                    yield r, c, _falloff_weight(falloff, dist, radius)
        return

    if region == Region.RECT:
        r0, c0, r1, c1 = (int(coords[0]), int(coords[1]),
                          int(coords[2]), int(coords[3]))
        if r1 < r0:
            r0, r1 = r1, r0
        if c1 < c0:
            c0, c1 = c1, c0
        r0 = max(0, r0); c0 = max(0, c0)
        r1 = min(h - 1, r1); c1 = min(w - 1, c1)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                yield r, c, 1.0
        return

    if region == Region.BEAM:
        # Thick line from (r0,c0) to (r1,c1) with half-width `width`. We rasterise
        # the bounding box and keep tiles whose perpendicular distance to the
        # segment is <= width. Row-major scan -> deterministic order. LINEAR
        # falloff fades with perpendicular distance across the beam's width.
        r0, c0, r1, c1, width = (int(coords[0]), int(coords[1]),
                                 int(coords[2]), int(coords[3]), float(coords[4]))
        seg_r = r1 - r0
        seg_c = c1 - c0
        seg_len2 = float(seg_r * seg_r + seg_c * seg_c)
        br0 = max(0, min(r0, r1) - int(np.ceil(width)))
        br1 = min(h - 1, max(r0, r1) + int(np.ceil(width)))
        bc0 = max(0, min(c0, c1) - int(np.ceil(width)))
        bc1 = min(w - 1, max(c0, c1) + int(np.ceil(width)))
        for r in range(br0, br1 + 1):
            for c in range(bc0, bc1 + 1):
                # Perpendicular distance from (r, c) to the segment.
                if seg_len2 <= 0.0:
                    t = 0.0
                else:
                    t = ((r - r0) * seg_r + (c - c0) * seg_c) / seg_len2
                    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                pr = r0 + t * seg_r
                pc = c0 + t * seg_c
                dist = float(np.sqrt((r - pr) ** 2 + (c - pc) ** 2))
                if dist <= width:
                    # LINEAR fades across the half-width; FLAT is uniform.
                    yield r, c, _falloff_weight(falloff, dist, max(width, 1.0))
        return

    raise ValueError(f"FieldEdit: unknown region {region!r}")  # pragma: no cover


def _falloff_weight(falloff: Falloff, dist: float, radius: float) -> float:
    """Per-tile falloff weight in [0, 1]."""
    if falloff == Falloff.FLAT:
        return 1.0
    if falloff == Falloff.LINEAR:
        if radius <= 0.0:
            return 1.0
        wgt = 1.0 - dist / radius
        return 0.0 if wgt < 0.0 else (1.0 if wgt > 1.0 else wgt)
    raise ValueError(f"FieldEdit: unknown falloff {falloff!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Combine — old value + contribution -> new value, per mode and dtype.
# Float fields: +=, -=, max. The `heat` (Q16.16) field: NEVER a float +=; the
# contribution is quantized and saturating-added.
# ---------------------------------------------------------------------------
def _combine_float(old: float, contribution: float, mode: EditMode,
                   clamp: Optional[Tuple[float, float]]) -> float:
    if mode == EditMode.ADD:
        new = old + contribution
    elif mode == EditMode.REMOVE:
        new = old - contribution
    elif mode == EditMode.MAX:
        new = old if old >= contribution else contribution
    else:  # pragma: no cover - deferred modes
        raise ValueError(f"FieldEdit: unsupported float mode {mode!r}")
    if clamp is not None:
        lo, hi = clamp
        if new < lo:
            new = lo
        elif new > hi:
            new = hi
    return new


def _combine_water(old_q: int, contribution: float, mode: EditMode,
                   clamp: Optional[Tuple[float, float]]) -> int:
    """Q16.16 combine for the `water_depth` field (S1). The stored value is an
    int32 in Q16.16 metres; dequantize to metres, combine in metres with the
    metres clamp (the float path's exact semantics), re-quantize round-to-
    nearest. Keeps edits authored in metres while the field stays integer."""
    old_m = float(old_q) / water_fixed.FP_ONE_F
    new_m = _combine_float(old_m, contribution, mode, clamp)
    return water_fixed.quantize_scalar(new_m)


def _combine_wave(old_q: int, contribution: float, mode: EditMode,
                  clamp: Optional[Tuple[float, float]]) -> int:
    """Q16.16 combine for the `wave_source` field (S2a). The stored value is an
    int32 in Q16.16; dequantize to real units, combine with the float path's
    exact +=/-=/max semantics (and any clamp), re-quantize round-to-nearest.
    Keeps the explosion's wave_source deposits authored in real units while the
    field stays integer/deterministic (the same idiom as `_combine_water`)."""
    old_v = float(old_q) / wave_fixed.FP_ONE_F
    new_v = _combine_float(old_v, contribution, mode, clamp)
    return wave_fixed.quantize_scalar(new_v)


def _combine_atmosphere(old_q: int, contribution: float, mode: EditMode,
                        clamp: Optional[Tuple[float, float]]) -> int:
    """Q16.16 combine for the `atmosphere` field (S2c). The stored value is an
    int32 in Q16.16 pressure; dequantize to real pressure, combine with the float
    path's exact +=/-=/max semantics (and any clamp), re-quantize round-to-
    nearest. Keeps the explosion pressure boost authored in real units while the
    field stays integer/deterministic (the same idiom as `_combine_wave`).

    RETAINED for any legacy direct caller, but no longer used by
    `apply_field_edit`'s "atmosphere" policy (EOS refactor P3 — see
    `_combine_atmosphere_to_N` below): P is solver-owned now (materialized
    once/tick by eos_solver), so writing it directly here would be silently
    clobbered next tick."""
    old_v = float(old_q) / atmosphere_fixed.FP_ONE_F
    new_v = _combine_float(old_v, contribution, mode, clamp)
    return atmosphere_fixed.quantize_scalar(new_v)


# EOS refactor P3 (design §6 "FieldEdit atmosphere/wave_source policies"):
# ambient composition split (P1 §2.1) — 21% O2, 79% inert_N2, matching the
# calibration that makes N_amb == 1.0 == P_amb at ambient (config
# [physics.eos] C = 1/t_amb_k). An "atmosphere" edit's `amount`/`contribution`
# keeps the SAME real-number scale the old direct-P edit used (both are
# calibrated to the same N_amb==P_amb==1.0 point), so shipped config values
# land in a similar ballpark without a forced re-tune — an explicit
# k_push/k_p-style tuning pass is a LATER patch (P3 reports deltas, does not
# secretly retune, per the design's own instruction).
_O2_FRACTION = 0.21
_N2_FRACTION = 0.79


def _combine_atmosphere_to_N(old_o2_q: int, old_n2_q: int, contribution: float,
                             mode: EditMode,
                             clamp: Optional[Tuple[float, float]]) -> Tuple[int, int]:
    """EOS refactor P3: the `atmosphere` FieldEdit policy's new target — a
    bulk-N deposit split 21/79 across `gas[O2]`/`gas[INERT_N2]` (design §6),
    REPLACING the direct P write `_combine_atmosphere` performed. `contribution`
    is split by the ambient fraction BEFORE combining, and any clamp is
    likewise fraction-scaled so a `(0, 2.0)` atmosphere clamp still bounds the
    TOTAL N the same way a total-pressure clamp would have."""
    old_o2 = float(old_o2_q) / gas_fixed.FP_ONE_F
    old_n2 = float(old_n2_q) / gas_fixed.FP_ONE_F
    o2_clamp = None
    n2_clamp = None
    if clamp is not None:
        lo, hi = clamp
        o2_clamp = (lo * _O2_FRACTION, hi * _O2_FRACTION)
        n2_clamp = (lo * _N2_FRACTION, hi * _N2_FRACTION)
    new_o2 = _combine_float(old_o2, contribution * _O2_FRACTION, mode, o2_clamp)
    new_n2 = _combine_float(old_n2, contribution * _N2_FRACTION, mode, n2_clamp)
    return gas_fixed.quantize_scalar(new_o2), gas_fixed.quantize_scalar(new_n2)


def _combine_temperature(old_q: int, contribution: float, mode: EditMode,
                         clamp: Optional[Tuple[float, float]]) -> int:
    """EOS refactor P3: the `wave_source` FieldEdit policy's new target —
    a `temperature` energy deposit (design §6: "wave_source -> T energy
    deposit"; explosions inject energy, not phantom acoustic staging).
    `temperature` shares the Q16.16 scale with `heat` (TEMP_SCALE ==
    HEAT_SCALE == 65536, materials.py); dequantize, combine with the float
    path's exact +=/-=/max semantics, re-quantize round-to-nearest (the same
    idiom as every other `_combine_*` helper). `contribution` keeps the SAME
    real-number scale the old `wave_source` edit used — a Kelvin re-tune of
    explosion `amount`s is a LATER pass, not done here."""
    old_v = float(old_q) / HEAT_SCALE
    new_v = _combine_float(old_v, contribution, mode, clamp)
    return int(new_v * HEAT_SCALE + (0.5 if new_v >= 0 else -0.5))


def _combine_gas(old_q: int, contribution: float, mode: EditMode,
                 clamp: Optional[Tuple[float, float]]) -> int:
    """Q16.16 combine for the `smoke` / gas fields (S2b). The stored value is an
    int32 in Q16.16 density; dequantize to real density, combine with the float
    path's exact +=/-=/max semantics (and the [0,1] clamp), re-quantize round-to-
    nearest. Keeps the explosion smoke deposit / clear authored in real density
    while the field stays integer/deterministic (the same idiom as
    `_combine_wave` / `_combine_water`)."""
    old_v = float(old_q) / gas_fixed.FP_ONE_F
    new_v = _combine_float(old_v, contribution, mode, clamp)
    return gas_fixed.quantize_scalar(new_v)


def _combine_fire(old_q: int, contribution: float, mode: EditMode,
                  clamp: Optional[Tuple[float, float]]) -> int:
    """Q16.16 combine for the `fire` field (S3a). The stored value is an int32 in
    Q16.16 intensity; dequantize to real [0,1] intensity, combine with the float
    path's exact +=/-=/max semantics (and the [0,1] clamp), re-quantize round-to-
    nearest. Keeps the explosion's `fire = max(...)` ignite deposit authored in
    real intensity while the field stays integer/deterministic (the same idiom as
    `_combine_gas`)."""
    old_v = float(old_q) / fire_fixed.FP_ONE_F
    new_v = _combine_float(old_v, contribution, mode, clamp)
    return fire_fixed.quantize_scalar(new_v)


def _combine_heat(old: int, contribution: float, mode: EditMode) -> int:
    """Q16.16 combine for the `heat` field. ADD only path uses the saturating
    integer add; REMOVE/MAX are defined for completeness (saturating subtract /
    integer max) but the live deposit path is ADD.
    """
    raw = heat_quantize(contribution)
    if mode == EditMode.ADD:
        return heat_saturating_add(int(old), raw)
    if mode == EditMode.REMOVE:
        new = int(old) - raw
        return new if new > 0 else 0
    if mode == EditMode.MAX:
        old_i = int(old)
        return old_i if old_i >= raw else raw
    raise ValueError(f"FieldEdit: unsupported heat mode {mode!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# The applier — the ONLY code that writes a field through this path.
# ---------------------------------------------------------------------------
def apply_field_edit(gmap, edit: FieldEdit, rng) -> None:
    """Apply one :class:`FieldEdit` to ``gmap`` in place.

    Resolves the field array + policy (dtype, default clamp, skip-mask), then
    iterates the region in deterministic row-major order. Per surviving tile
    (one that passes the skip-mask): if ``edit.noise > 0`` draw ONE multiplier
    from ``rng`` in ``[1 - noise, 1]`` (skipped tiles draw nothing — the RNG
    sequence depends only on the surviving-tile order, which is deterministic),
    compute ``contribution = amount × weight × noise_mult``, and combine.

    ``heat`` goes through the Q16.16 saturating branch; every other field is
    float ``+=`` / ``-=`` / ``max``. The clamp is the edit's own clamp if set,
    else the field-policy default.
    """
    pol = _policy(edit.field)
    # EOS refactor P3 (design §6): "atmosphere" and "wave_source" edits no
    # longer target their namesake array — P is solver-owned (a direct write
    # would be clobbered next tick) and wave_source is retired. Resolve the
    # REAL target array(s) here; the per-tile loop below special-cases both.
    o2_arr = n2_arr = None
    if edit.field == "atmosphere":
        o2_arr = gmap.gas[gmap.gases.name_to_id["o2"]]
        n2_arr = gmap.gas[gmap.gases.name_to_id["inert_n2"]]
        arr = o2_arr   # only used for `.shape` below; the loop writes o2/n2 directly
    elif edit.field == "wave_source":
        arr = gmap.temperature
    else:
        arr = getattr(gmap, edit.field)
    ch = edit.channel
    if pol.dtype == "gas" and arr.ndim == 3:
        # The (N, h, w) multi-gas array (W3): ``channel`` names the SLICE
        # (gas id — engine/05 §6.2), not an RGB channel. Resolve it to the
        # contiguous (h, w) int32 view once, then the edit proceeds exactly
        # like a scalar gas field ("smoke" is the same array's BLACK_SMOKE
        # slice). A gas edit without a channel is a caller bug — loud.
        if ch is None:
            raise ValueError(
                "FieldEdit: a 'gas' edit must set channel=<gas id> "
                "(the (N, h, w) slice index — gmap.gases.name_to_id)")
        arr = arr[int(ch)]
        ch = None
    shape = (arr.shape[0], arr.shape[1])

    skip_mask = pol.skip(gmap) if pol.skip is not None else None
    clamp = edit.clamp if edit.clamp is not None else pol.clamp

    noise = edit.noise
    if noise > 0.0:
        noise = 1.0 if noise > 1.0 else noise
    low = 1.0 - noise

    is_heat = (pol.dtype == "heat")
    is_water = (pol.dtype == "water")
    is_wave = (pol.dtype == "wave")
    is_gas = (pol.dtype == "gas")
    is_atmosphere = (pol.dtype == "atmosphere")
    is_fire = (pol.dtype == "fire")
    # ``ch`` was resolved above: None for scalar fields (incl. a gas SLICE,
    # already sliced out), 0/1/2 for the RGB channel of an (h, w, 3) field.

    for (r, c, weight) in _iter_region(edit.region, edit.coords,
                                       edit.falloff, shape):
        # Skip-mask veto FIRST — a skipped tile draws no RNG (order preserved).
        if skip_mask is not None and skip_mask[r, c]:
            continue

        if noise > 0.0:
            mult = float(rng.uniform(low, 1.0))
        else:
            mult = 1.0

        contribution = edit.amount * weight * mult

        if is_heat:
            arr[r, c] = _combine_heat(arr[r, c], contribution, edit.mode)
        elif is_water:
            arr[r, c] = _combine_water(int(arr[r, c]), contribution,
                                       edit.mode, clamp)
        elif is_wave:
            # EOS refactor P3: `arr` is `gmap.temperature` here (resolved
            # above) — the energy-deposit target, not the retired wave_source
            # array.
            arr[r, c] = _combine_temperature(int(arr[r, c]), contribution,
                                             edit.mode, clamp)
        elif is_gas:
            arr[r, c] = _combine_gas(int(arr[r, c]), contribution,
                                     edit.mode, clamp)
        elif is_atmosphere:
            # EOS refactor P3: bulk-N deposit split 21/79 across O2/inert_N2
            # (o2_arr/n2_arr resolved above), not a direct P write.
            new_o2, new_n2 = _combine_atmosphere_to_N(
                int(o2_arr[r, c]), int(n2_arr[r, c]), contribution,
                edit.mode, clamp)
            o2_arr[r, c] = new_o2
            n2_arr[r, c] = new_n2
        elif is_fire:
            arr[r, c] = _combine_fire(int(arr[r, c]), contribution,
                                      edit.mode, clamp)
        elif ch is None:
            arr[r, c] = _combine_float(float(arr[r, c]), contribution,
                                       edit.mode, clamp)
        else:
            arr[r, c, ch] = _combine_float(float(arr[r, c, ch]), contribution,
                                           edit.mode, clamp)


# ---------------------------------------------------------------------------
# EditQueue — per-tick deposit list with a deterministically-ordered flush.
# ---------------------------------------------------------------------------
class EditQueue:
    """Collects :class:`FieldEdit`s during a tick; applies them all at flush in a
    stable sort by ``(field, source_id, region, seq)``.

    ``seq`` is the monotonically-increasing enqueue index, the final tie-break:
    within one emitter (same field/source_id/region) the apply order equals the
    enqueue order, so e.g. the explosion's many ``wave_source`` ADD edits sum in
    exactly their original disc-iteration order (bit-identical float result).
    Across emitters the sort makes the order independent of projectile / AI /
    container iteration order — the determinism guarantee.

    The flush is the SINGLE RNG consumer for ``noise > 0`` edits: it draws in
    sorted order, so a seeded ``sim.rng`` gives the same rollout regardless of
    how the edits were enqueued.
    """

    def __init__(self):
        self._edits = []   # list[(FieldEdit, seq)]
        self._seq = 0

    def __len__(self):
        return len(self._edits)

    def enqueue(self, edit: FieldEdit) -> None:
        """Append ``edit`` to this tick's queue (no application yet)."""
        self._edits.append((edit, self._seq))
        self._seq += 1

    def clear(self) -> None:
        """Drop all queued edits and reset the sequence counter."""
        self._edits.clear()
        self._seq = 0

    def _sort_key(self, item):
        edit, seq = item
        return (edit.field, int(edit.source_id), int(edit.region), seq)

    def flush(self, gmap, rng) -> None:
        """Apply every queued edit in stable-sorted order, then clear the queue.

        Deterministic regardless of enqueue order: the sort key is
        ``(field, source_id, region, seq)``. The ``noise`` RNG draws happen here,
        in that order, so the flush is the one RNG consumer.
        """
        if not self._edits:
            return
        ordered = sorted(self._edits, key=self._sort_key)
        for edit, _seq in ordered:
            apply_field_edit(gmap, edit, rng)
        self.clear()


__all__ = [
    "FieldEdit", "EditMode", "Region", "Falloff", "EditQueue",
    "apply_field_edit", "FIELD_POLICY",
    "HEAT_SCALE", "heat_quantize", "heat_saturating_add",
]
