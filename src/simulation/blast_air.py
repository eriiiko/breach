"""THE blast's read of the air it goes off in (#31, Erik 2026-11-06).

A charge reads each open tile it reached once, at detonation, after its wall
damage and before its field edits land, and the air decides what the blast leaves
behind. A strength row reads the tiles its walk reached (blast_expansion.
reach_tiles), so air behind a wall is not its air; any other charge reads its disc
(``disc_tiles``). Air there is taken as two numbers:

  g  the O2 law's pressure factor (o2_pressure_factor.h, the same two edges,
     [physics.fire] p_ext_atm / p_full_atm): 0 at or below p_ext, 1 at or above
     p_full. Pressure, never density: hot air is thin, not low-pressure.
  b  how burnable the tile's O2 is: 0 at or below [blast_air] o2_low, 1 at or
     above o2_full, on the O2 MOLE FRACTION.

Per tile, the smoke deposit is scaled by ``g * (1 + starved_smoke * (1 - b))``:
normal air leaves the charge's soot as it is, O2-poor air leaves more (nothing
burned the carbon), thin air less, vacuum none. So each blast's smoke follows
the air it landed in.

For the whole blast, the heat is scaled by ``s + (1 - s) * mean(g * b)`` with
s = [blast_air] detonation_share: the detonation itself needs no air, the
afterburn does. Sources: docs/research/explosion_litsearch_2026-11-06.md
sections 3 and 7 (Dobratz & Crawford for the two heats; Sachs for thin air).

A blast in normal air reads exactly 1 everywhere, so it deposits what it always
did. No RNG. Integer law; the two results cross to the float FieldEdit amounts
once, in ``payloads.execute_payload``. Row values are PLACEHOLDERS.

A deliberate approximation: nothing burns O2 here, and no carbon is tracked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import CFG
from simulation import atmosphere_fixed, gas_fixed

ONE = gas_fixed.quantize_q16_scalar(1.0)


@dataclass(frozen=True)
class BlastAirSettings:
    o2_low_q: int
    o2_full_q: int
    starved_smoke_q: int
    detonation_share_q: int
    p_ext_q: int
    p_full_q: int


@dataclass(frozen=True)
class AirRead:
    """``smoke_scale_q16`` is an (h, w) plane of Q16 factors, or None when every
    tile reads exactly 1. ``heat_scale_q16`` is the one factor on the heat."""
    smoke_scale_q16: np.ndarray | None
    heat_scale_q16: int


def blast_air_settings() -> BlastAirSettings:
    """The door: refuses a missing key or an impossible span by name."""
    rows = getattr(CFG, "blast_air", None)
    if rows is None:
        raise ValueError("config.toml has no [blast_air] table")
    vals = {}
    for key in ("o2_low", "o2_full", "starved_smoke", "detonation_share"):
        v = getattr(rows, key, None)
        if v is None:
            raise ValueError(f"[blast_air] {key} is missing")
        vals[key] = float(v)
    if not 0.0 <= vals["o2_low"] < vals["o2_full"]:
        raise ValueError("[blast_air] needs 0 <= o2_low < o2_full")
    if vals["starved_smoke"] < 0.0 or not 0.0 <= vals["detonation_share"] <= 1.0:
        raise ValueError("[blast_air] needs starved_smoke >= 0 and "
                         "detonation_share in [0, 1]")
    fire = CFG.physics.fire
    return BlastAirSettings(
        o2_low_q=gas_fixed.quantize_q16_scalar(vals["o2_low"]),
        o2_full_q=gas_fixed.quantize_q16_scalar(vals["o2_full"]),
        starved_smoke_q=gas_fixed.quantize_q16_scalar(vals["starved_smoke"]),
        detonation_share_q=gas_fixed.quantize_q16_scalar(vals["detonation_share"]),
        p_ext_q=int(atmosphere_fixed.quantize_scalar(float(fire.p_ext_atm))),
        p_full_q=int(atmosphere_fixed.quantize_scalar(float(fire.p_full_atm))))


def _ramp_q(x, lo, hi):
    """Q16 ramp, exactly 0 at or below ``lo`` and exactly 1 at or above ``hi``."""
    r = np.where(x >= hi, ONE, (np.maximum(x - lo, 0) * ONE) // (hi - lo))
    return np.minimum(r, ONE)


def disc_tiles(gmap, fy, fx, radius):
    """The open tiles of the disc around (fy, fx): what a charge without
    ``strength`` reads."""
    h, w = gmap.solid.shape
    r = int(np.ceil(max(float(radius), 1.0)))
    y0, y1 = max(fy - r, 0), min(fy + r + 1, h)
    x0, x1 = max(fx - r, 0), min(fx + r + 1, w)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disc = (yy - fy) ** 2 + (xx - fx) ** 2 <= r * r
    ys, xs = np.nonzero(disc & ~gmap.solid[y0:y1, x0:x1])
    return list(zip((ys + y0).tolist(), (xs + x0).tolist()))


def read_blast_air(gmap, tiles, settings=None) -> AirRead:
    """Read ``tiles``, (y, x) pairs: the same list the blast delivers on."""
    s = blast_air_settings() if settings is None else settings
    from simulation.gases import INERT_N2, O2
    if not tiles:
        return AirRead(None, ONE)
    ys, xs = np.array(tiles, dtype=np.intp).T

    o2 = gmap.gas[O2][ys, xs].astype(np.int64)
    bulk = o2 + gmap.gas[INERT_N2][ys, xs].astype(np.int64)
    x_q = np.where(bulk > 0, (o2 * ONE) // np.maximum(bulk, 1), 0)
    p_q = gmap.atmosphere[ys, xs].astype(np.int64)

    b_q = _ramp_q(x_q, s.o2_low_q, s.o2_full_q)
    g_q = _ramp_q(p_q, s.p_ext_q, s.p_full_q)
    smoke_q = (g_q * (ONE + ((s.starved_smoke_q * (ONE - b_q)) >> 16))) >> 16
    burn_q = (g_q * b_q) >> 16

    mean_burn = int(burn_q.sum()) // len(tiles)
    heat_q = s.detonation_share_q + (((ONE - s.detonation_share_q) * mean_burn) >> 16)
    if (smoke_q == ONE).all():
        return AirRead(None, heat_q)
    plane = np.full(gmap.solid.shape, ONE, dtype=np.int64)
    plane[ys, xs] = smoke_q
    return AirRead(plane, heat_q)


def blast_base_soot_g(payload) -> float:
    """The soot a charge always leaves: its explosive's own, never below
    [blast_air] min_soot_yield (Erik, 2026-11-06: a little smoke always, as the
    stand-in for the charge's fuel and what lets the fireball radiate its heat
    away; clean gas does not radiate), all of it times [blast_air] smoke_scale,
    the game's look (Erik chose 0.25 by eye, 2026-11-06).
    No soot_yield row is touched: those stay the cited physics."""
    floor = float(CFG.blast_air.min_soot_yield)
    return float(CFG.blast_air.smoke_scale) * max(
        float(getattr(payload, "blast_soot_g", 0.0)),
        float(getattr(payload, "explosive_kg", 0.0)) * 1000.0 * floor)


__all__ = ["AirRead", "blast_base_soot_g", "BlastAirSettings", "blast_air_settings",
           "disc_tiles", "read_blast_air"]
