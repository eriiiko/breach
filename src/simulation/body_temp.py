"""Body temperature -- a unit's shell temperature, an energy balance with a heat
capacity (#215; ``docs/body_temperature_design_2026-11-17.md`` section 2,
version 4, AGREED by Erik 2026-11-17).

One number per unit, ``body_temp_q`` (Q16 degrees C; starts at the ambient
``[combat] heat_ambient_ref``, hashed only while it differs). It is written by
the heat coupling row ALONE (``exchange.apply_environmental_damage`` calls
:func:`advance`), and reset to ambient by :func:`clear` at death and
conversion. Anything that warms or cools a body changes the felt temperature
or the air, never the body.

THE LAW, per tick, every living unit (degrees as Q16; ``a`` the per-tick
fraction ``loss / (capacity x tps)``)::

    in    = a x (felt - ambient) + k_air x (T_air - ambient)   what reaches the body
    out   = (k_air + a_rad) x (body - ambient)                  what it loses
    body += round((round(h x in / ONE) - out) / ONE)
    k_air = min(a_air x p, ONE - a_rad)

``h`` is THE PROTECTION (#215, Erik 2026-11-18: "protect only the body"): the
env profile's ``heat_absorption``, the share of the heat that reaches the body
it takes up, 1.0 = the unprotected law. It scales both inflows alike (the
radiation and deposits the felt temperature reads, and the air's excess over
ambient), never the losses, so the balance stays one linear law and in still
ambient air the body settles at ``ambient + h x (felt - ambient)``. At ``h =
ONE`` the step is the unscaled law bit for bit (the air term ``k_air (T_air -
body)`` split at ambient into its inflow and its loss).

``felt`` is THE felt temperature (``exchange.felt_temperature_q``): the heat
row's reading of what the body absorbs, which the felt temperature already is
in disguise (``felt - ambient = gain x Phi`` with ``Phi`` the absorbed heat
units, so ``a x (felt - ambient) = Phi_abs x J_per_heat_unit / C``). THE LOSS
is derived from the same anchor, never tuned: ``loss = J_per_heat_unit x tps
/ heat_flux_to_temp`` (:func:`loss_w_per_k`; 159 W/K at 0.333 m tiles), so
``a = J_per_heat_unit / (heat_flux_to_temp x C)`` and at one atmosphere in 20
C air the body settles EXACTLY where the felt temperature stands (the
literature table's steady damage kept). ``C`` is the unit's environment
profile's ``body_heat_capacity_j_per_k`` (320 J/K: a time constant of 2 s).

THE AIR AND THE VACUUM: the loss splits into the share the shell radiates,
``a_rad = round(a x s)`` with ``s = [combat] body_radiated_share`` (0.15,
against ambient: the only cooling in vacuum), and the rest ``a_air = a -
a_rad`` exchanged with the air on the unit's tiles (:func:`air_read`): the
mean pressure ``p`` (atm, Q16) of the footprint's gas tiles scales it, and its
temperature is the pressure-weighted mean of theirs (each tile counts by how
much air it holds; a vacuum tile adds nothing), ``T_air = ambient + k x
T_game`` (the engine's ambient IS the body model's 20 C). At ``p == ONE`` and
``T_air == ambient`` the two parts sum to ``a`` exactly, so nothing of the
table moves. ``k_air`` is held at most ``ONE - a_rad`` (no tick closes more
than the whole gap: the explicit step stays stable at any pressure); that
bounds the step, never the energy -- nothing clips what a body absorbs.

Integer only, no random draw. Device twin: the heat row's kernel
(``cpp/src/cuda_unit_exchange.cu``) over ``unit_lib::body_step`` /
``body_air`` (``cpp/src/cuda_unit_damage.cuh``), integer for integer; it
FAULTS past :data:`BODY_MAX` (the range the int64 products are proved for).
"""
from __future__ import annotations

from fractions import Fraction

from simulation import unit_fixed

ONE = unit_fixed.FP_ONE
#: The proved range of the device's products: |body|, |felt - ambient| and
#: |T_air| at most 2^44 Q16 degrees (about 2.7e8 C), a tile's pressure at most
#: 2^25 (512 atm). Python's ints are unbounded; the device faults past it.
BODY_MAX = 1 << 44
PRESSURE_MAX = 1 << 25


def _cfg():
    from config import CFG
    return CFG


def ambient_q() -> int:
    """The body's ambient, ``[combat] heat_ambient_ref`` as Q16 degrees: the
    schema's default and :func:`clear`'s value."""
    return unit_fixed.quantize_scalar(
        float(_cfg().combat.heat_ambient_ref))                 # p3e-door


def j_per_heat_unit(gmap) -> Fraction:
    """One heat unit of the map in joules (``THERMAL_MASS_UNIT x V_tile``,
    ``blast_expansion.heat_units_per_mj``'s own reckoning), exact."""
    from simulation.blast_expansion import tile_volume_m3
    from simulation.materials import THERMAL_MASS_UNIT
    return Fraction(float(THERMAL_MASS_UNIT)) * Fraction(tile_volume_m3(gmap))


def loss_w_per_k(gmap, tps: int | None = None) -> float:
    """THE derived loss, W/K: ``J_per_heat_unit x tps / heat_flux_to_temp``
    (a read for the readout and the report; the law reads :func:`rates_q`)."""
    tps = int(tps if tps is not None else _cfg().clock.ticks_per_second)
    k = Fraction(float(_cfg().combat.heat_flux_to_temp))
    return float(j_per_heat_unit(gmap) * tps / k)


def capacity_of(env) -> float:
    """The unit's shell capacity, J/K: its environment's, else ``[combat]``'s."""
    c = getattr(env, "body_heat_capacity_j_per_k", None)
    if c is None:
        c = _cfg().combat.body_heat_capacity_j_per_k
    return float(c)


def rates_q(gmap, capacity_j_per_k: float) -> tuple:
    """``(a_q, a_rad_q)``: the per-tick fraction ``J_per_heat_unit /
    (heat_flux_to_temp x C)`` (Q16, the exact rational rounded once) and its
    radiated share ``round(a x s)``. The door: refuses a capacity that is not
    above 0 or so small the body would move more than the whole gap in a
    tick (``a > ONE``), and a share outside [0, 1]."""
    c = float(capacity_j_per_k)                               # p3e-door
    if not c > 0.0:
        raise ValueError(f"body_heat_capacity_j_per_k must be above 0 "
                         f"(got {capacity_j_per_k!r})")
    k = Fraction(float(_cfg().combat.heat_flux_to_temp))
    a = j_per_heat_unit(gmap) * ONE / (k * Fraction(c))
    a_q = unit_fixed.div_round(a.numerator, a.denominator)
    if not 0 < a_q <= ONE:
        raise ValueError(
            f"body_heat_capacity_j_per_k = {c:g} J/K gives a body time "
            f"constant below one tick (a = {a_q}/{ONE} per tick)")
    share = float(_cfg().combat.body_radiated_share)
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"[combat] body_radiated_share must be in [0, 1] "
                         f"(got {share!r})")
    a_rad_q = unit_fixed.mul_q16_round(a_q, unit_fixed.quantize_scalar(share))
    return a_q, a_rad_q


def k_temp_q() -> int:
    """The game-temperature -> degrees scale (``temperature_scale``'s
    ``k_temp_to_kelvin``), Q16: the air's excess over the engine's ambient in
    degrees."""
    from temperature_scale import load
    return unit_fixed.quantize_scalar(float(load(_cfg()).k_temp_to_kelvin))


def air_read(gmap, u, amb_q: int, k_q: int) -> tuple:
    """``(p_q, t_air_q)``: the air on ``u``'s footprint -- over its in-bounds
    GAS tiles (a solid tile, a door in the frame, holds no air), each tile's
    pressure ``p_i = max(0, atmosphere)`` (atm, Q16): ``p_q = mean_round(sum
    p_i, n)`` and ``t_air_q = ambient + round(k x round(sum p_i x T_i / sum
    p_i))`` (the pressure-weighted air temperature; game degrees to C).
    ``(0, ambient)`` when no air is there (vacuum: the air part vanishes)."""
    from simulation.exchange import _mean_round
    h, w = gmap.atmosphere.shape
    solid = gmap.solid
    atm = gmap.atmosphere
    temp = gmap.temperature
    n = 0
    p_sum = 0
    pt_sum = 0
    for (tx, ty) in u.occupied_tiles():
        if 0 <= ty < h and 0 <= tx < w and not solid[ty, tx]:
            p = int(atm[ty, tx])
            if p < 0:
                p = 0
            n += 1
            p_sum += p
            pt_sum += p * int(temp[ty, tx])
    if n == 0 or p_sum == 0:
        return 0, amb_q
    t_game = unit_fixed.div_round(pt_sum, p_sum)
    return (_mean_round(p_sum, n),
            amb_q + unit_fixed.mul_q16_round(k_q, t_game))


def absorption_q(env) -> int:
    """The unit's protection, its environment's ``heat_absorption`` as a Q16
    fraction (the profile door holds it in [0, 1])."""
    from simulation.environment import HUMAN_ENVIRONMENT
    h = getattr(env, "heat_absorption", None)
    if h is None:
        h = HUMAN_ENVIRONMENT.heat_absorption
    h = float(h)                                              # p3e-door
    if not 0.0 <= h <= 1.0:
        raise ValueError(f"heat_absorption must be in [0, 1] (got {h!r})")
    return unit_fixed.quantize_scalar(h)


def step_q(body_q: int, felt_q: int, p_q: int, t_air_q: int, a_q: int,
           a_rad_q: int, amb_q: int, h_q: int = ONE) -> int:
    """The law's one tick (module docstring): the new body temperature."""
    k_air = unit_fixed.mul_q16_round(a_q - a_rad_q, p_q)
    cap = ONE - a_rad_q
    if k_air > cap:
        k_air = cap
    inflow = a_q * (felt_q - amb_q) + k_air * (t_air_q - amb_q)
    if h_q != ONE:
        inflow = unit_fixed.mul_q16_round(h_q, inflow)
    num = inflow - (k_air + a_rad_q) * (body_q - amb_q)
    return body_q + unit_fixed.div_round(num, ONE)


def advance(u, gmap, felt_q: int, a_q: int, a_rad_q: int, amb_q: int,
            k_q: int, h_q: int = ONE) -> int:
    """One tick of the law on ``u`` (the heat row's call): reads the air on
    its footprint, writes and returns ``u.body_temp_q``."""
    body = int(getattr(u, "body_temp_q", amb_q))
    p_q, t_air_q = air_read(gmap, u, amb_q, k_q)
    new = step_q(body, felt_q, p_q, t_air_q, a_q, a_rad_q, amb_q, h_q)
    if new != body:
        u.body_temp_q = new
    return new


def clear(u) -> None:
    """Back to ambient (death, conversion)."""
    if hasattr(u, "body_temp_q"):
        u.body_temp_q = ambient_q()


__all__ = ["ONE", "BODY_MAX", "PRESSURE_MAX", "absorption_q", "ambient_q", "j_per_heat_unit",
           "loss_w_per_k", "capacity_of", "rates_q", "k_temp_q", "air_read",
           "step_q", "advance", "clear"]
