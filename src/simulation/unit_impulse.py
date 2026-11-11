"""The unit impulse law, its linear part (gameplay pack v1, G5p-a2;
``docs/unit_impulse_design_2026-11-10.md`` §3 to §6 and the binding build
sheet §13, from critique B).

THE WAY ANYTHING PUSHES A UNIT. Everything that pushes a unit hands it an
IMPULSE through :func:`add` -- a round that hits, the round's recoil on its
shooter, a shove, and (G5p-b) the wind, the blast's wind included
(:func:`add_wind`, the ``wind | max`` coupling row at slot 9c2) -- never a
position or a velocity write. Once per tick, in conductor slot 1a, :func:`integrate_all`
turns what each living unit received into a velocity change over its mass and
a spin change over its yaw inertia; its feet resist up to one tick of HOLD; a
sudden push is first absorbed IN PLACE as a SWAY (which never moves the unit)
and only what overflows it becomes a PUSHED VELOCITY ``pv``; how hard the feet
are working (the low-passed DEMAND, ``effort``) slows the walk; past a fall
limit, a saturated effort or an unrecoverable push the unit goes KNOCKED_DOWN.
In slot 3 :func:`move` moves a unit by its walk and its ``pv`` as ONE
displacement, in sub-tile substeps, a wall zeroing (and booking) the push
into it.

Integers only (pack §2 rule 1). Currencies (§13):

* velocities -- ``swx/swy``, ``pvx/pvy``, every limit -- Q16.16 TILES PER TICK;
* impulses -- the accumulators ``imp_x/imp_y`` -- Q16.16 GRAM-TILES PER TICK,
  ``imp_l`` Q16.16 gram-tile^2 per tick; mass is integer GRAMS
  (``Unit.mass_g``), the yaw inertia integer gram-tile^2;
* ``effort`` Q16 in ``[0, 2 ONE]``; ``react_ticks`` whole ticks;
* angles in the trig kit's Q16.16 radians (``turning.py``).

Every real-unit dial is quantized ONCE per map in the :class:`ImpulseTable`
(``impulse_table(cfg, tile_size_m, tps)``, built at Simulation construction
beside the speed table and cached on the map, :func:`table_for`). Division
truncates toward zero (``unit_fixed.div_trunc``, C's ``/``) and the REMAINDER
of ``J / m`` and ``L / I`` stays in the accumulator, so momentum closes
EXACTLY in integers, per unit per tick (the books, :func:`books_identity`)::

    sum of J added == m * (pv + sw + resisted + wall + clamped) + imp_residual

A velocity bound :data:`V_MAX_Q` (4 tiles a tick) clamps ``dv`` and ``pv``,
booked as ``clamped``; with it every square fits int64 and every argument of
the kit's int32 trig fits int32.

Order-free by construction: :func:`add` is an integer sum, so the result never
depends on the ids or the order of whoever pushed; the integration reads only
the unit's own state, the displacement only the map and the unit.

What the law gives (design §6) -- soft input never moves a unit: only ``pv``
moves it and ``pv`` is fed only by sway overflow; an input train whose
per-tick ``|dv|`` stays within the in-place capacity and whose mean is below
the hold is removed by the resist step every tick, the reaction delay
included. A force above the hold overflows the sway within ticks, pushes by
its excess and floors the unit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from simulation import turning, unit_fixed
from simulation.unit_fixed import (FP_ONE as ONE, div_trunc, effort_step,
                                   isqrt, isqrt_ceil, shr_trunc)

#: The velocity bound (critique B, M3): 4 tiles a tick (32 m/s at 0.333 m
#: tiles, 96 m/s at 1 m), in Q16 tiles per tick. ``dv`` and ``pv`` are
#: clamped to it, booked as ``clamped``.
V_MAX_Q = 4 * ONE
#: The largest part of a displacement per axis per substep: just under one
#: tile, so a footprint anchored at ``floor(x)`` can never skip a tile column
#: (the bound the Speed table's door already holds a walk step to).
SUBSTEP_Q = ONE - 1
#: The effort's ceiling: demand is clipped at twice the hold.
EFFORT_MAX = 2 * ONE
#: The direction blend's three slots (forward = a push along the facing,
#: i.e. from behind; side; backward).
FWD, SIDE, BACK = 0, 1, 2
#: The books' slots (velocity counts except ``IN_*``, which are impulse
#: counts): see :func:`books_identity`.
IN_X, IN_Y, RES_X, RES_Y, WALL_X, WALL_Y, CLAMP_X, CLAMP_Y = range(8)


# ---------------------------------------------------------------------------
# The door and the table
# ---------------------------------------------------------------------------
_REAL_KEYS = (
    "gravity_mps2", "k_hold", "k_slow", "mu_slide",
    "v_inplace_fwd_mps", "v_inplace_side_mps", "v_inplace_back_mps",
    "v_fall_fwd_mps", "v_fall_side_mps", "v_fall_back_mps",
    "t_react_seconds", "yaw_inertia_kgm2", "r_body_m",
    "drag_area_m2", "drag_area_prone_m2")
_MULT_KEYS = ("bullet_push_mult", "bullet_twist_mult", "recoil_mult",
              "wind_push_mult", "zombie_hold_mult")
_SHOVE_KEYS = ("impulse_ns", "cooldown_seconds", "cone_half_deg")


@dataclass(frozen=True)
class ImpulseSettings:
    """The ``[impulse]`` dials in real units, refused at :func:`impulse_settings`."""
    gravity_mps2: float
    k_hold: float
    k_slow: float
    mu_slide: float
    v_inplace_fwd_mps: float
    v_inplace_side_mps: float
    v_inplace_back_mps: float
    v_fall_fwd_mps: float
    v_fall_side_mps: float
    v_fall_back_mps: float
    t_react_seconds: float
    n_rec_ticks: int
    yaw_inertia_kgm2: float
    r_body_m: float
    drag_area_m2: float
    drag_area_prone_m2: float
    bullet_push_mult: float
    bullet_twist_mult: float
    recoil_mult: float
    wind_push_mult: float
    zombie_hold_mult: float
    shove_impulse_ns: float
    shove_cooldown_seconds: float
    shove_cone_half_deg: float
    getup_ticks: int


def _num(section, where, key, *, positive=True):
    if section is None or not hasattr(section, key):
        raise ValueError(f"{where} is missing {key!r} (the unit impulse law, "
                         f"G5p)")
    v = getattr(section, key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where} {key} must be a number, got {v!r}")
    v = float(v)
    if not math.isfinite(v) or (v <= 0.0 if positive else v < 0.0):
        raise ValueError(f"{where} {key} must be "
                         f"{'> 0' if positive else '>= 0'}, got {v!r}")
    return v


def impulse_settings(cfg) -> ImpulseSettings:
    """THE refusing door of ``[impulse]`` and ``[impulse.shove]`` (a loaded
    config, ``CFG``): every key required, a number, finite, > 0 (the four
    multipliers >= 0; ``n_rec_ticks`` a positive integer); ``k_slow`` below
    ``k_hold``. The knockdown's duration is the one the wind law uses,
    ``[exchange] knockdown_getup_ticks``. A missing or mistyped key is
    refused by name -- a silent default would hide a typo."""
    sec = getattr(cfg, "impulse", None)
    if sec is None:
        raise ValueError("config.toml has no [impulse] section (the unit "
                         "impulse law, G5p)")
    vals = {k: _num(sec, "[impulse]", k) for k in _REAL_KEYS}
    for k in _MULT_KEYS:
        vals[k] = _num(sec, "[impulse]", k, positive=False)
    if not hasattr(sec, "n_rec_ticks"):
        raise ValueError("[impulse] is missing 'n_rec_ticks'")
    n_rec = sec.n_rec_ticks
    if isinstance(n_rec, bool) or not isinstance(n_rec, int) or n_rec < 1:
        raise ValueError(f"[impulse] n_rec_ticks must be an integer >= 1, "
                         f"got {n_rec!r}")
    if not vals["k_slow"] < vals["k_hold"]:
        raise ValueError(f"[impulse] k_slow ({vals['k_slow']}) must be below "
                         f"k_hold ({vals['k_hold']})")
    shove = getattr(sec, "shove", None)
    if shove is None:
        raise ValueError("config.toml has no [impulse.shove] table")
    sv = {k: _num(shove, "[impulse.shove]", k) for k in _SHOVE_KEYS}
    if not sv["cone_half_deg"] <= 180.0:
        raise ValueError(f"[impulse.shove] cone_half_deg must be <= 180, got "
                         f"{sv['cone_half_deg']!r}")
    exch = getattr(cfg, "exchange", None)
    getup = getattr(exch, "knockdown_getup_ticks", None)
    if isinstance(getup, bool) or not isinstance(getup, (int, float)) \
            or int(getup) < 1:
        raise ValueError(f"[exchange] knockdown_getup_ticks must be >= 1 "
                         f"(the knockdown's duration), got {getup!r}")
    return ImpulseSettings(
        **vals, n_rec_ticks=int(n_rec),
        shove_impulse_ns=sv["impulse_ns"],
        shove_cooldown_seconds=sv["cooldown_seconds"],
        shove_cone_half_deg=sv["cone_half_deg"],
        getup_ticks=int(getup))


def _q(v: float) -> int:
    """A real value onto the Q16.16 grid, round half away from zero (the
    kit's ``quantize``), for any magnitude (an impulse in gram-tiles per tick
    exceeds the int32 range of the trig arguments; it is never one)."""
    v = float(v) * ONE
    return int(math.floor(v + 0.5) if v >= 0.0 else math.ceil(v - 0.5))


@dataclass(frozen=True)
class ImpulseTable:
    """The law's dials in GAME units, quantized ONCE for one map (its tile
    size) and the clock (§13). Velocities are Q16 tiles per tick, impulses
    Q16 gram-tiles per tick, the inertia gram-tile^2, angles Q16 radians."""
    tile_size_m: float
    tps: int
    cap_q: int                    # one tick of standing hold (a velocity)
    cap_prone_q: int              # ... of a prone body's friction
    zombie_hold_q16: int          # the zombie state's hold multiplier (Q16)
    v_inplace_q: tuple            # (fwd, side, back) in-place capacity
    v_fall_q: tuple               # (fwd, side, back) fall limit at stability 1
    react_ticks: int              # the reaction delay after an overflow
    n_rec: int                    # the recovery horizon, ticks
    effort_edge_q: int            # k_slow / k_hold: effort below it slows nothing
    inertia_g: int                # yaw inertia, gram-tile^2 (>= 1)
    r_body_q: int                 # every lever arm is clamped to it
    twist_q16: int                # bullet_twist_mult in Q16
    getup_ticks: int              # the knockdown's duration
    ammo_push_q: dict = field(default_factory=dict)    # ammo row -> J (hit)
    ammo_recoil_q: dict = field(default_factory=dict)  # ammo row -> J (recoil)
    shove_q: int = 0              # the shove's impulse
    shove_cooldown_ticks: int = 0
    shove_cone_q: int = 0         # the shove's half-angle, Q16 radians
    #: G6p: a melee row's blow push, ``[weapons.*] melee_push_ns`` as Q16
    #: gram-tiles per tick, by weapon row name (rows with a push only).
    melee_push_q: dict = field(default_factory=dict)
    #: G5p-b: the wind's impulse per unit of wind load, standing and prone:
    #: ``J_q = ((W >> 16) * wind_k_q) >> 16`` (:func:`wind_impulse_q`).
    wind_k_q: int = 0
    wind_k_prone_q: int = 0
    rho_air: float = 0.0          # the [materials.air] density it was built at
    settings: ImpulseSettings | None = None

    def hold_q(self, u, prone: bool = False) -> int:
        """One tick of hold for ``u`` (standing, or prone friction): the
        same for every mass (a velocity); the zombie STATE's multiplier
        applies on ``is_zombie`` (RULED 1.0: a zombie is a copy of the unit
        it infected)."""
        cap = self.cap_prone_q if prone else self.cap_q
        if getattr(u, "is_zombie", False) and self.zombie_hold_q16 != ONE:
            cap = shr_trunc(cap * self.zombie_hold_q16)
        return cap if cap >= 1 else 1


def _momentum_q(ns: float, mult: float, tile_m: float, tps: int) -> int:
    """An impulse in N s as Q16 gram-tiles per tick: ``J * 1000 g/kg /
    (tile_m * tps)`` (1 N s = 1 kg m/s), times its multiplier, quantized
    once."""
    return _q(float(ns) * float(mult) * 1000.0 / (float(tile_m) * int(tps)))


def _wind_k(rho: float, area_m2: float, mult: float, tile: float,
            tps: int) -> int:
    """The wind's impulse per unit of the pre-shifted wind load, Q16 (G5p-b).

    ``W = (N / N_amb) v^2 2^32`` (``GameMap.footprint_wind``), so ``q =
    rho W / 2^33`` Pa; the force ``q A`` held for one tick is ``q A / tps``
    N s, which is ``q A 1000 2^16 / (tile tps^2)`` Q16 gram-tiles per tick
    (:func:`_momentum_q`'s unit). With ``W' = W >> 16`` that is ``W' K``,
    ``K = rho A mult 1000 / (2 tile tps^2)``, quantized ONCE here as
    ``round(K 2^16)`` (2.83 -> 185 224 at 0.9 m2, 0.333 m tiles, 24 tps: a
    relative grain of 5e-6)."""
    k = float(rho) * float(area_m2) * float(mult) * 1000.0 \
        / (2.0 * float(tile) * int(tps) * int(tps))
    return int(math.floor(k * ONE + 0.5))


def impulse_table(cfg, tile_size_m: float, tps: int | None = None,
                  weapon_tables=None, rho_air: float | None = None
                  ) -> ImpulseTable:
    """The :class:`ImpulseTable` for a map of ``tile_size_m`` metres a tile
    at ``tps`` ticks a second (default ``[clock] ticks_per_second``), from
    the door :func:`impulse_settings` and the ammo rows' momentum
    (``[ammo.*] bullet_g``, ``muzzle_mps``). ``rho_air`` is the map's
    ``[materials.air]`` density (the materials table's row, the one owner;
    None reads the config's row) -- the wind's dynamic pressure is priced
    with it. Refuses (ValueError, naming what) a dial that quantizes to 0
    counts, a hold of one tick that is not below the smallest in-place
    capacity (critique A, m3 -- compared on the QUANTIZED values), an
    in-place capacity at or above a fall limit, a yaw inertia below one
    gram-tile^2 and an air density that is not > 0."""
    s = impulse_settings(cfg)
    tps = int(cfg.clock.ticks_per_second if tps is None else tps)
    tile = float(tile_size_m)
    if not tile > 0.0:
        raise ValueError(f"impulse table: tile_size_m must be > 0, got "
                         f"{tile_size_m!r}")
    if rho_air is None:
        rho_air = float(cfg.materials.air.density)
    rho_air = float(rho_air)
    if not (math.isfinite(rho_air) and rho_air > 0.0):
        raise ValueError(f"impulse table: the [materials.air] density must "
                         f"be > 0, got {rho_air!r}")
    v_unit = tile * tps                      # 1 tile/tick in m/s

    def vel(mps, key):
        q = _q(mps / v_unit)
        if q < 1:
            raise ValueError(f"[impulse] {key} = {mps} m/s is below one Q16 "
                             f"count of tiles per tick at {tile} m, {tps} tps")
        return q

    g = s.gravity_mps2
    cap = vel(s.k_hold * g / tps, "k_hold")
    cap_prone = vel(s.mu_slide * g / tps, "mu_slide")
    v_in = (vel(s.v_inplace_fwd_mps, "v_inplace_fwd_mps"),
            vel(s.v_inplace_side_mps, "v_inplace_side_mps"),
            vel(s.v_inplace_back_mps, "v_inplace_back_mps"))
    v_fall = (vel(s.v_fall_fwd_mps, "v_fall_fwd_mps"),
              vel(s.v_fall_side_mps, "v_fall_side_mps"),
              vel(s.v_fall_back_mps, "v_fall_back_mps"))
    if not cap < min(v_in):
        raise ValueError(
            f"[impulse] one tick of hold ({cap} counts, k_hold) must be below "
            f"the smallest in-place capacity ({min(v_in)} counts) at "
            f"{tps} tps, or a steady sub-hold force would stagger")
    for k in range(3):
        if not v_in[k] < v_fall[k]:
            raise ValueError(f"[impulse] the in-place capacity must be below "
                             f"the fall limit in every direction "
                             f"({v_in} vs {v_fall})")
    inertia = int(math.floor(s.yaw_inertia_kgm2 * 1000.0 / (tile * tile)
                             + 0.5))
    if inertia < 1:
        raise ValueError(f"[impulse] yaw_inertia_kgm2 = {s.yaw_inertia_kgm2} "
                         f"is below one gram-tile^2 at {tile} m tiles")
    r_body = _q(s.r_body_m / tile)
    if r_body < 1:
        raise ValueError(f"[impulse] r_body_m = {s.r_body_m} is below one "
                         f"count at {tile} m tiles")
    react = int(math.floor(s.t_react_seconds * tps + 0.5))   # half up: 7.2 -> 7
    if weapon_tables is None:
        from simulation.weapons import get_tables
        weapon_tables = get_tables()
    push, recoil = {}, {}
    for name, a in weapon_tables.ammo.by_name.items():
        ns = float(getattr(a, "momentum_ns", 0.0))
        if ns > 0.0:
            push[name] = _momentum_q(ns, s.bullet_push_mult, tile, tps)
            recoil[name] = _momentum_q(ns, s.recoil_mult, tile, tps)
    melee_push = {}
    for name, w in weapon_tables.weapons.by_name.items():
        ns = float(getattr(w, "melee_push_ns", 0.0))
        if ns > 0.0:
            melee_push[name] = _momentum_q(ns, 1.0, tile, tps)
    return ImpulseTable(
        tile_size_m=tile, tps=tps, cap_q=cap, cap_prone_q=cap_prone,
        zombie_hold_q16=_q(s.zombie_hold_mult),
        v_inplace_q=v_in, v_fall_q=v_fall, react_ticks=react,
        n_rec=s.n_rec_ticks,
        effort_edge_q=_q(s.k_slow / s.k_hold),
        inertia_g=inertia, r_body_q=r_body,
        twist_q16=_q(s.bullet_twist_mult), getup_ticks=s.getup_ticks,
        ammo_push_q=push, ammo_recoil_q=recoil,
        melee_push_q=melee_push,
        shove_q=_momentum_q(s.shove_impulse_ns, 1.0, tile, tps),
        shove_cooldown_ticks=max(1, int(math.floor(
            s.shove_cooldown_seconds * tps + 0.5))),
        shove_cone_q=_q(math.radians(s.shove_cone_half_deg)),
        wind_k_q=_wind_k(rho_air, s.drag_area_m2, s.wind_push_mult, tile,
                         tps),
        wind_k_prone_q=_wind_k(rho_air, s.drag_area_prone_m2,
                               s.wind_push_mult, tile, tps),
        rho_air=rho_air,
        settings=s)


def map_air_density(gmap) -> float | None:
    """``gmap``'s ``[materials.air]`` density row (None for a map without a
    materials table: the config's row is read instead)."""
    mats = getattr(gmap, "materials", None)
    dens = getattr(mats, "density", None)
    if dens is None:
        return None
    from simulation.materials import MAT_AIR
    return float(dens[MAT_AIR])


def table_for(gmap) -> ImpulseTable:
    """The map's :class:`ImpulseTable` (cached on it: the table is per map).
    ``Simulation`` builds it at construction; a bare map a test hands the
    combat helpers gets the live config's table on first use."""
    t = getattr(gmap, "_impulse_table", None)
    if t is None:
        from config import CFG
        t = impulse_table(CFG, float(getattr(gmap, "tile_size_m", 1.0)),
                          rho_air=map_air_density(gmap))
        gmap._impulse_table = t
    return t


# ---------------------------------------------------------------------------
# The per-unit state
# ---------------------------------------------------------------------------
def kit_kg(u) -> float:
    """What ``u`` carries on top of its body, kg: its loadout row's
    ``kit_kg`` (a zombie, or a unit with no loadout, carries 0)."""
    name = getattr(u, "loadout_name", None)
    if name is None:
        return 0.0
    from simulation.loadouts import get_tables
    return float(get_tables().loadout(name).kit_kg)


def bind_mass(u) -> None:
    """``u.mass_g``: the body (``Unit.mass``, the species' Q16-snapped kg)
    plus the kit, in integer GRAMS, quantized once (§13). Refused when not
    > 0. Called at construction and again where the sim re-assigns the
    species mass (``Simulation.add_unit``); a conversion keeps it (a zombie
    is a copy of the unit it infected)."""
    m = int(math.floor((float(u.mass) + kit_kg(u)) * 1000.0 + 0.5))
    if m <= 0:
        raise ValueError(f"unit {getattr(u, 'name', '?')}: mass + kit must be "
                         f"> 0 (got {m} g)")
    u.mass_g = m


def init_state(u) -> None:
    """Every impulse state of ``u`` at rest (construction). Integers, a
    fixed width (pack §2 rule 3)."""
    u.imp_x = u.imp_y = u.imp_l = 0
    u.stand_x = u.stand_y = 0
    u.swx = u.swy = 0
    u.pvx = u.pvy = 0
    u.effort = 0
    u.react_ticks = 0
    u.last_shove_tick = -1
    u.pending_shove = False
    u.impulse_books = [0] * 8
    bind_mass(u)


def clear(u) -> None:
    """Zero ``u``'s impulse state and books (at death and at conversion,
    §13): a revived corpse inherits no stale push. The facing and the turn
    are the turning law's (``turning.settle`` at conversion)."""
    u.imp_x = u.imp_y = u.imp_l = 0
    u.stand_x = u.stand_y = 0
    u.swx = u.swy = 0
    u.pvx = u.pvy = 0
    u.effort = 0
    u.react_ticks = 0
    u.impulse_books = [0] * 8


def at_rest(u) -> bool:
    """No impulse state at all (the integration's fast path, and what the
    digest hashes as absent)."""
    return not (u.imp_x or u.imp_y or u.imp_l or u.swx or u.swy or u.pvx
                or u.pvy or u.effort or u.react_ticks
                or getattr(u, "stand_x", 0) or getattr(u, "stand_y", 0))


def books_identity(u):
    """The momentum books of ``u`` per axis: ``(in, out)`` with ``in`` the
    impulse ADDED since the last :func:`clear` and ``out = m * (pv + sw +
    resisted + wall + clamped) + residual`` -- equal, exactly, after every
    integration and every move (gate G4)."""
    b, m = u.impulse_books, u.mass_g
    out_x = m * (u.pvx + u.swx + b[RES_X] + b[WALL_X] + b[CLAMP_X]) + u.imp_x
    out_y = m * (u.pvy + u.swy + b[RES_Y] + b[WALL_Y] + b[CLAMP_Y]) + u.imp_y
    return (b[IN_X], out_x), (b[IN_Y], out_y)


# ---------------------------------------------------------------------------
# THE ONE ENTRY
# ---------------------------------------------------------------------------
def angular_q(table: ImpulseTable, jx_q: int, jy_q: int, lever_q: int,
              twist_q16: int = ONE) -> int:
    """The angular impulse of a push ``(jx, jy)`` whose line passes
    ``lever_q`` (Q16 tiles, signed: positive turns the unit
    COUNTER-CLOCKWISE in the facing's frame) from the unit's centre: the
    lever CLAMPED to the body radius (per input, never on a sum), times the
    push's magnitude, times ``twist_q16``; truncated toward zero, so a
    mirrored push gives the mirrored spin to the count."""
    r = table.r_body_q
    lev = r if lever_q > r else (-r if lever_q < -r else int(lever_q))
    if lev == 0:
        return 0
    l_q = shr_trunc(isqrt(jx_q * jx_q + jy_q * jy_q) * lev)
    if twist_q16 != ONE:
        l_q = shr_trunc(l_q * int(twist_q16))
    return l_q


def lever_q(cx_q: int, cy_q: int, ox_q: int, oy_q: int, sx_q: int,
            sy_q: int) -> int:
    """The signed perpendicular distance (Q16 tiles) from the centre ``c``
    of a unit to the line through ``o`` along the unit vector ``s`` (world
    axes, y down): ``((c - o) x s)`` in the form whose positive sign turns
    the unit counter-clockwise in the facing's (y-up) frame -- a push along
    ``s`` passing on the RIGHT of the centre as seen along ``s`` (south of
    it when ``s`` points east) turns the unit counter-clockwise. Truncated
    toward zero (sign-symmetric)."""
    return shr_trunc((cx_q - ox_q) * sy_q - (cy_q - oy_q) * sx_q)


def add(table: ImpulseTable, unit, jx_q: int, jy_q: int, lever: int = 0,
        twist_q16: int = ONE) -> None:
    """THE way anything pushes a unit: the linear impulse ``(jx_q, jy_q)``
    (Q16 gram-tiles per tick, world axes, y down) is added to the unit's
    accumulators and books, and its angular part (:func:`angular_q`: the
    lever clamped to the body radius) to ``imp_l``. Sums of integers
    commute, so the result never depends on the order or the ids of whoever
    pushed. Integrated in the next slot 1a. No clamp on the running sum."""
    jx_q, jy_q = int(jx_q), int(jy_q)
    unit.imp_x += jx_q
    unit.imp_y += jy_q
    b = unit.impulse_books
    b[IN_X] += jx_q
    b[IN_Y] += jy_q
    if lever:
        unit.imp_l += angular_q(table, jx_q, jy_q, int(lever), twist_q16)


def along_q(j_q: int, sx_q: int, sy_q: int):
    """An impulse of magnitude ``j_q`` along the Q16 unit vector ``s``,
    per axis, truncated toward zero."""
    return shr_trunc(j_q * int(sx_q)), shr_trunc(j_q * int(sy_q))


# ---------------------------------------------------------------------------
# The wind (G5p-b): a force is an impulse every tick
# ---------------------------------------------------------------------------
def wind_impulse_q(table: ImpulseTable, w: int, prone: bool = False) -> int:
    """The magnitude of one tick of wind on a body, Q16 gram-tiles per tick:
    the dynamic pressure of the wind load ``w`` (``GameMap.footprint_wind``'s
    ``W``) times the body's drag area (standing, or ``prone``), held for one
    tick -- ``((w >> 16) * k) >> 16`` with the table's ``wind_k_q`` /
    ``wind_k_prone_q``. Non-negative, monotone in ``w``; 0 at ``w = 0``.
    Python ints: in int64 the product holds to ``W`` of about 3e18 (q of
    4e8 Pa)."""
    k = table.wind_k_prone_q if prone else table.wind_k_q
    w = int(w)
    if w <= 0 or k == 0:
        return 0
    return ((w >> 16) * k) >> 16


def add_wind(table: ImpulseTable, u, w: int, vx: int, vy: int,
             prone: bool) -> bool:
    """One tick of wind on the living unit ``u`` (the ``wind | max`` row,
    Simulation.step 9c2): the impulse :func:`wind_impulse_q` of the wind
    load ``w`` along the wind ``(vx, vy)`` of the tile that load was read on
    (Q16 m/s, world axes, y down), through :func:`add` with no lever (wind
    does not turn a unit in v1, design §10). A PRONE body takes its prone
    area's impulse; the surplus of the standing area's over it is kept in
    ``stand_x`` / ``stand_y`` -- never momentum, read once by the next
    integration's demand -- so a prone unit is judged by the wind it WOULD
    take standing (critique A, M7: nobody gets up into a gale). Returns
    whether anything was added. Integer only; truncated toward zero, so a
    mirrored wind pushes the mirrored way to the count."""
    vx, vy = int(vx), int(vy)
    n = isqrt(vx * vx + vy * vy)
    if n == 0:
        return False
    j = wind_impulse_q(table, w, prone)
    if prone:
        js = wind_impulse_q(table, w, False)
        if js != j:
            u.stand_x += div_trunc(js * vx, n) - div_trunc(j * vx, n)
            u.stand_y += div_trunc(js * vy, n) - div_trunc(j * vy, n)
    if j == 0:
        return False
    add(table, u, div_trunc(j * vx, n), div_trunc(j * vy, n))
    return True


# ---------------------------------------------------------------------------
# The integration (slot 1a)
# ---------------------------------------------------------------------------
def _blend(v3, c: int) -> int:
    """A direction limit from ``(fwd, side, back)`` at ``cos theta = c`` (Q16,
    theta from the facing): the Speed table's own linear blend in cos theta
    (``side + (fwd - side) c`` forward of the side, ``side + (side - back) c``
    behind it), floored."""
    fwd, side, back = v3
    if c >= 0:
        return side + (((fwd - side) * c) >> 16)
    return side + (((side - back) * c) >> 16)


def _limit(v3, vx: int, vy: int, facing_q: int) -> int:
    from simulation.movement import facing_cos_q16
    return _blend(v3, facing_cos_q16(vx, vy, facing_q))


def _shrink(x: int, y: int, b: int):
    """``(x, y)`` shortened toward zero by ``b`` in MAGNITUDE (integer root,
    components truncated, exactly zero when ``b`` covers it): returns the
    new vector and the magnitude used, ``min(|v|, b)``."""
    n = isqrt(x * x + y * y)
    if n <= b:
        return 0, 0, n
    return div_trunc(x * (n - b), n), div_trunc(y * (n - b), n), b


def _clamp_vmax(x: int, y: int):
    """``(x, y)`` scaled into :data:`V_MAX_Q` (unchanged within it)."""
    m2 = x * x + y * y
    if m2 <= V_MAX_Q * V_MAX_Q:
        return x, y
    n = isqrt_ceil(m2)
    return div_trunc(x * V_MAX_Q, n), div_trunc(y * V_MAX_Q, n)


def stability_q16(u) -> int:
    """``u``'s knockdown stability in Q16: the wind law's own read (the
    zombie state's overlay, else the env profile's), quantized."""
    from simulation.exchange import _stability_for
    return unit_fixed.quantize_scalar(_stability_for(u))


def pending_dw(table: ImpulseTable, u) -> int:
    """The spin change ``u``'s next slot 1a takes from its angular
    accumulator: ``imp_l / I`` truncated -- what the turning law adds to
    omega before it turns (the bots' predictor reads exactly this)."""
    l = getattr(u, "imp_l", 0)
    return div_trunc(l, table.inertia_g) if l else 0


def _knock_down(table, u) -> None:
    from simulation.status import KNOCKED_DOWN, apply_status
    apply_status(u, KNOCKED_DOWN, magnitude=0,
                 duration_ticks=table.getup_ticks, source_id=None)


def integrate(table: ImpulseTable, u, events=None) -> int:
    """One tick of the linear law for the living unit ``u`` (§5, §13 order):
    the accumulators into ``dv`` / ``dw`` (the remainders kept), the demand
    and the effort, the sway and its overflow into ``pv``, the resist, the
    knockdown checks. Returns ``dw`` (Q16 radians per tick) for the turning
    law. A unit with no impulse state does nothing and returns 0."""
    if at_rest(u):
        return 0
    from simulation.status import composed_flags
    m = u.mass_g
    b = u.impulse_books
    # 1. dv, dw -- the remainder stays in the accumulator (momentum exact).
    dvx = div_trunc(u.imp_x, m)
    dvy = div_trunc(u.imp_y, m)
    u.imp_x -= dvx * m
    u.imp_y -= dvy * m
    dw = 0
    if u.imp_l:
        dw = div_trunc(u.imp_l, table.inertia_g)
        u.imp_l -= dw * table.inertia_g
    cx, cy = _clamp_vmax(dvx, dvy)
    b[CLAMP_X] += dvx - cx
    b[CLAMP_Y] += dvy - cy
    dvx, dvy = cx, cy
    # 2. the demand on the STANDING hold (one denominator, prone or not) and
    #    the effort's low-pass toward it, rounded away from zero. A prone
    #    body's demand is the one it WOULD have standing (G5p-b, critique A
    #    M7): the wind's standing surplus joins dv here, never momentum.
    cap = table.hold_q(u)
    ddx, ddy = dvx, dvy
    sx_, sy_ = getattr(u, "stand_x", 0), getattr(u, "stand_y", 0)
    if sx_ or sy_:
        ddx += div_trunc(sx_, m)
        ddy += div_trunc(sy_, m)
        u.stand_x = u.stand_y = 0
    demand = 0
    if ddx or ddy:
        demand = div_trunc(isqrt(ddx * ddx + ddy * ddy) * ONE, cap)
        if demand > EFFORT_MAX:
            demand = EFFORT_MAX
    u.effort += effort_step(demand - u.effort)
    st = stability_q16(u)
    if composed_flags(u).is_prone:
        # PRONE: no sway, no reaction delay -- every input meets the prone
        # friction at once and what is left moves the body.
        u.pvx += u.swx + dvx
        u.pvy += u.swy + dvy
        u.swx = u.swy = 0
        u.react_ticks = 0
        px, py = _clamp_vmax(u.pvx, u.pvy)
        b[CLAMP_X] += u.pvx - px
        b[CLAMP_Y] += u.pvy - py
        nx, ny, _ = _shrink(px, py, table.hold_q(u, prone=True))
        b[RES_X] += px - nx
        b[RES_Y] += py - ny
        u.pvx, u.pvy = nx, ny
        # Nobody gets up into an over-hold input or while sliding.
        if u.effort * ONE >= st * st or nx or ny:
            _knock_down(table, u)
        return dw
    # 3. the in-place sway; its overflow becomes the pushed velocity.
    u.swx += dvx
    u.swy += dvy
    sx, sy = u.swx, u.swy
    if sx or sy:
        lim = _limit(table.v_inplace_q, sx, sy, u.facing_q)
        m2 = sx * sx + sy * sy
        if m2 > lim * lim:
            n = isqrt_ceil(m2)
            kx, ky = div_trunc(sx * lim, n), div_trunc(sy * lim, n)
            u.pvx += sx - kx
            u.pvy += sy - ky
            u.swx, u.swy = kx, ky
            u.react_ticks = table.react_ticks
            if events is not None:
                from simulation.events import StaggerEvent
                events.append(StaggerEvent(unit_id=int(getattr(u, "id", -1)),
                                           ticks=int(table.react_ticks)))
    px, py = _clamp_vmax(u.pvx, u.pvy)
    b[CLAMP_X] += u.pvx - px
    b[CLAMP_Y] += u.pvy - py
    u.pvx, u.pvy = px, py
    # 4. resist with one tick of hold: the sway first, EVERY tick; what is
    #    left of the budget works off pv once the reaction delay has run.
    nx, ny, used = _shrink(u.swx, u.swy, cap)
    b[RES_X] += u.swx - nx
    b[RES_Y] += u.swy - ny
    u.swx, u.swy = nx, ny
    margin = cap - used
    if u.react_ticks == 0:
        if (u.pvx or u.pvy) and margin > 0:
            nx, ny, _ = _shrink(u.pvx, u.pvy, margin)
            b[RES_X] += u.pvx - nx
            b[RES_Y] += u.pvy - ny
            u.pvx, u.pvy = nx, ny
    else:
        u.react_ticks -= 1
    # 6. the knockdown checks (5, the walking scale, is read in slot 3).
    down = False
    vx, vy = u.pvx + u.swx, u.pvy + u.swy
    if vx or vy:
        vf = (_limit(table.v_fall_q, vx, vy, u.facing_q) * st) >> 16
        down = vx * vx + vy * vy >= vf * vf
    if not down:
        down = u.effort * ONE >= st * st
    if not down and u.react_ticks == 0 and (u.pvx or u.pvy):
        r = table.n_rec * margin
        down = u.pvx * u.pvx + u.pvy * u.pvy > r * r
    if down:
        _knock_down(table, u)
    return dw


def integrate_all(sim) -> None:
    """Conductor slot 1a, "integrate + turn" (§13): for every living unit in
    id (list) order, one tick of the linear law (:func:`integrate`), then
    ``omega += dw`` and one tick of the turning law toward its target (the
    live aim while one is held). Each unit reads only its own state, so the
    order changes nothing; every impulse is integrated exactly one tick after
    it lands, whichever slot produced it."""
    table = sim.impulse_table
    events = sim.tick_events
    for u in sim.units:
        if u.alive:
            dw = integrate(table, u, events)
            turning.step(u, sim.unit_turn_accel_q16(u),
                         sim.unit_turn_rate_q16(u), dw)


# ---------------------------------------------------------------------------
# The walk's two readers (slot 3 and the zombie / WEGO walk sites)
# ---------------------------------------------------------------------------
def effort_scale_q16(table: ImpulseTable, u) -> int:
    """The walking scale from the effort (design §5 step 5, critique B M2):
    ``ONE`` up to the slow edge ``k_slow / k_hold``, linear down to half at
    an effort of ONE (the hold), half above; floored. The
    ``FootprintSamples.effort_scale_q16`` member every walker composes in
    ``movement.default_speed``."""
    e = int(getattr(u, "effort", 0))
    e0 = table.effort_edge_q
    if e <= e0:
        return ONE
    if e >= ONE:
        return ONE // 2
    return ONE - ((ONE // 2) * (e - e0)) // (ONE - e0)


def walk_allowed(u) -> bool:
    """THE walk ban (design §9, PLACEHOLDER): a unit whose reaction delay
    runs (it is catching itself after a push that overflowed its sway) does
    not walk -- it can still shoot. Read at the three walk sites (MOVE_DIR,
    the zombie walk, the WEGO path); never a scale (``default_speed``
    refuses 0)."""
    return int(getattr(u, "react_ticks", 0)) == 0


# ---------------------------------------------------------------------------
# The displacement (slot 3)
# ---------------------------------------------------------------------------
def move(gmap, u, walk_dx_q: int = 0, walk_dy_q: int = 0) -> bool:
    """Move ``u`` by its walk ``(walk_dx_q, walk_dy_q)`` (Q16 tiles this
    tick) and its pushed velocity as ONE displacement (§13), on the Q16 grid
    (the position enters as ``quantize(u.x)`` and leaves as an exact
    n/65536 double, so every position this writes is on the grid). In
    ``k = ceil(max(|dx|, |dy|) / SUBSTEP_Q)`` substeps of under one tile per
    axis (one, i.e. today's test, at a walk alone), each running the walk's
    own slide rule through the collision predicate walking uses
    (``GameMap.is_passable_block``): the full step, else the x part alone,
    else the y part alone, else stop. The axis not taken is INTO THE WALL:
    pv's component there is zeroed and booked (``wall``), the rest of the
    move on that axis dropped. A pushed unit therefore never ends inside a
    wall and never crosses one (no bouncing in v1). Returns True when the
    position changed."""
    dx = int(walk_dx_q) + u.pvx
    dy = int(walk_dy_q) + u.pvy
    if dx == 0 and dy == 0:
        return False
    fp = u.footprint
    x0 = unit_fixed.quantize_scalar(u.x)
    y0 = unit_fixed.quantize_scalar(u.y)
    x, y = x0, y0
    k = max(1, -(-max(abs(dx), abs(dy)) // SUBSTEP_Q))
    px, py = div_trunc(dx, k), div_trunc(dy, k)
    x_free = y_free = True
    b = u.impulse_books
    passable = gmap.is_passable_block
    for i in range(k):
        last = i == k - 1
        ex = (dx - px * (k - 1) if last else px) if x_free else 0
        ey = (dy - py * (k - 1) if last else py) if y_free else 0
        if ex == 0 and ey == 0:
            continue
        nx, ny = x + ex, y + ey
        if passable(ny >> 16, nx >> 16, fp):
            x, y = nx, ny
        elif passable(y >> 16, nx >> 16, fp):
            x = nx                               # slide along x: y is the wall
            if ey:
                b[WALL_Y] += u.pvy
                u.pvy = 0
                y_free = False
        elif passable(ny >> 16, x >> 16, fp):
            y = ny                               # slide along y: x is the wall
            if ex:
                b[WALL_X] += u.pvx
                u.pvx = 0
                x_free = False
        else:                                    # boxed in: the move ends
            if ex:
                b[WALL_X] += u.pvx
                u.pvx = 0
            if ey:
                b[WALL_Y] += u.pvy
                u.pvy = 0
            break
    if (x, y) == (x0, y0):
        return False
    u.x = x / 65536.0
    u.y = y / 65536.0
    return True


def walk_step_q(move_dir, v_q: int):
    """The walk part of a MOVE_DIR tick in Q16 tiles: the intent's Q16 unit
    vector times the commanded speed ``v_q`` (Q16 tiles per tick,
    ``movement.move_speed_q16``), each axis truncated toward zero."""
    return (shr_trunc(int(move_dir.dx_q) * int(v_q)),
            shr_trunc(int(move_dir.dy_q) * int(v_q)))


# ---------------------------------------------------------------------------
# The shove (slot 1b)
# ---------------------------------------------------------------------------
def centre_q(u):
    """``u``'s footprint centre, Q16 tiles (``quantize(x) + fp / 2``)."""
    half = int(u.footprint) << 15
    return (unit_fixed.quantize_scalar(u.x) + half,
            unit_fixed.quantize_scalar(u.y) + half)


def shove_target(table: ImpulseTable, units, u):
    """The unit ``u`` shoves (§13, critique B M10): among the OTHER living
    units of any team whose centre is ADJACENT (squared centre distance at
    most ``((fp_a + fp_b) / 2 + 1)^2`` tiles) and within the shove's cone of
    ``u``'s facing (``[impulse.shove] cone_half_deg``), the one whose
    bearing is nearest that facing, ties to the lower id. None when there is
    none. Integer only; reads the slot-start positions."""
    cx, cy = centre_q(u)
    best = None
    best_key = None
    for t in units:
        if t is u or not t.alive:
            continue
        tx, ty = centre_q(t)
        dx, dy = tx - cx, ty - cy
        reach = ((int(u.footprint) + int(t.footprint)) << 15) + ONE
        if dx * dx + dy * dy > reach * reach:
            continue
        if dx == 0 and dy == 0:
            continue
        off = abs(turning.wrap_q16(
            unit_fixed.atan2_q16(-dy, dx) - u.facing_q))
        if off > table.shove_cone_q:
            continue
        key = (off, int(t.id))
        if best_key is None or key < best_key:
            best, best_key = t, key
    return best


def shove(table: ImpulseTable, units, u, tick: int, events=None) -> bool:
    """One shove by ``u`` (slot 1b's consumption of ``shove_intent``): if its
    cooldown has run and a target exists (:func:`shove_target`), the shove's
    impulse along ``u``'s facing is added to the target (:func:`add`; the
    lever is the offset of ``u``'s centre line from the target's centre),
    the cooldown starts and a render-only ``ShoveEvent`` is emitted. With no
    target nothing happens and the cooldown is not spent. The shover feels
    no reaction in v1 (ACCEPTED GAP, design §1)."""
    last = int(getattr(u, "last_shove_tick", -1))
    if last >= 0 and tick - last < table.shove_cooldown_ticks:
        return False
    t = shove_target(table, units, u)
    if t is None:
        return False
    f = u.facing_q
    sx, sy = unit_fixed.cos_q16(f), -unit_fixed.sin_q16(f)   # world y down
    jx, jy = along_q(table.shove_q, sx, sy)
    ux, uy = centre_q(u)
    tx, ty = centre_q(t)
    add(table, t, jx, jy, lever_q(tx, ty, ux, uy, sx, sy))
    u.last_shove_tick = int(tick)
    if events is not None:
        from simulation.events import ShoveEvent
        events.append(ShoveEvent(unit_id=int(u.id), target_id=int(t.id)))
    return True


__all__ = [
    "V_MAX_Q", "SUBSTEP_Q", "EFFORT_MAX", "ImpulseSettings", "ImpulseTable",
    "impulse_settings", "impulse_table", "table_for", "kit_kg", "bind_mass",
    "init_state", "clear", "at_rest", "books_identity", "angular_q",
    "lever_q", "add", "along_q", "wind_impulse_q", "add_wind",
    "map_air_density", "stability_q16", "pending_dw", "integrate",
    "integrate_all", "effort_scale_q16", "walk_allowed", "move",
    "walk_step_q", "centre_q", "shove_target", "shove",
]
