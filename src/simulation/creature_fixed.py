"""Fixed-point boundary module for creatures (#106 P6a): the CHAIN currency,
the creature's mass and stiffness currencies, the integer HEADROOM bounds a
creature row must satisfy, and the ONE pose <-> Unit-anchor conversion.

Design: docs/octopus_p6_sim_port_design_2026-10-30.md sections 3, 4.1, 6.1,
6.2, 6.6. Every number entering a creature's synced state converts ONCE
through this module (or the C++ kit's twin, once ``CHAIN_FINE_BITS`` lands in
``cpp/src/fixed_point.h`` with P6b/P6c) -- never an inline shift, never a
hardcoded ``1 << 24``. The integer reference (P6b,
``docs/octopus_chain_ref_q.py``) imports its currencies and headroom
constants from HERE, so the door that accepts a row and the arithmetic that
runs it can never disagree about what fits.

Currencies
----------
* chain positions and velocities: int32, Q24 metres (m/s) --
  ``CHAIN_SHIFT = 16 + CHAIN_FINE_BITS``, 60 nm per count, +/-128 m
  (``CHAIN_RANGE_M``). The fine bits are the ``TRACE_FINE_BITS`` precedent:
  in Q16 every acceleration below one count per substep would be erased by
  the PBD velocity rebuild (design 6.1).
* masses: Q16 kg (``MASS_SHIFT``); inverse masses ``w``: Q32 1/kg in int64
  (``INV_MASS_SHIFT``), derived from the QUANTIZED mass as
  ``w = 2^48 // m_q16`` so the integer path never sees a float.
* stiffness (``k_dist`` N/m, ``k_bend`` N m/rad): Q16 (``STIFF_SHIFT``),
  int64.
* ``W = sum_i w_i |grad C_i|^2``: Q32 (``W_SHIFT``) in its own SI unit
  (1/kg for a distance constraint, 1/(kg m^2) for a bend).
* the constraint RATE ``r = k W (1 + gamma) + 1/dt^2``: Q0, integer 1/s^2
  (``RATE_SHIFT = 0``). Exact because ``1/dt^2 = (tps * substeps)^2`` is an
  integer (147456 at 24 tps x 16) and every shipped r is >= that, so Q0
  keeps >= 17 bits of relative resolution. ``k W`` is formed as ONE 128-bit
  product narrowed by ``KW_PRE_SHIFT = STIFF_SHIFT + W_SHIFT - RATE_SHIFT``
  (the kit's ``mul128_shr``).

Rounding: every conversion here rounds half AWAY from zero
(``fixedpoint::quantize``), so it is symmetric about 0 -- the "no slide"
discipline of design 6.3 starts at the door.

Import-light (numpy only): the level loader, tools and the reference read it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Currencies
# ---------------------------------------------------------------------------
#: The chain currency's extra fractional bits over Q16 (design 6.1). THE
#: Python twin of the coming ``fixed_point.h`` ``CHAIN_FINE_BITS`` (P6b/P6c
#: add the C++ literal and the twins test).
CHAIN_FINE_BITS = 8
CHAIN_SHIFT = 16 + CHAIN_FINE_BITS          # Q24
CHAIN_ONE = 1 << CHAIN_SHIFT                # counts per metre
_CHAIN_ONE_F = float(CHAIN_ONE)
#: int32 positions in Q24: the representable span, metres (and m/s).
CHAIN_RANGE_M = float(1 << (31 - CHAIN_SHIFT))   # 128.0
CHAIN_INT32_MAX = (1 << 31) - 1

MASS_SHIFT = 16                             # Q16 kg
STIFF_SHIFT = 16                            # Q16 N/m, N m/rad
INV_MASS_SHIFT = 32                         # Q32 1/kg (int64)
W_SHIFT = 32                                # Q32, the constraint's W
RATE_SHIFT = 0                              # r in integer 1/s^2
#: ``k_q16 * W_q32`` (a 128-bit product, Q48) >> this -> r's currency (Q0).
KW_PRE_SHIFT = STIFF_SHIFT + W_SHIFT - RATE_SHIFT   # 48
Q16_ONE = 1 << 16
_Q16_ONE_F = float(Q16_ONE)

# ---------------------------------------------------------------------------
# Headroom constants (the bounds; see chain_headroom for each formula)
# ---------------------------------------------------------------------------
#: Every int64 intermediate is held under 2^62: one bit of sign, one of carry
#: for the sums the reference adds to it (``+ 1/dt^2``, ``+ D``).
INT64_HEADROOM_BITS = 62
#: (i) the lightest node must be at least this many Q16 counts (design 6.1:
#: ">= 1 count"). The mid octopus's tip is ~870 counts.
MASS_MIN_COUNTS = 1
#: (i) the heaviest mass (a node or the body) keeps this many counts of its
#: Q32 inverse: w >= 2^16 counts <=> m <= 65536 kg.
INV_MASS_MIN_COUNTS = 1 << 16
#: (ii) the distance factor ``f = k w / r`` is a Q30 fraction formed as
#: ``floordiv((k w)_q0 << F_SHIFT, r_q0)``.
F_SHIFT = 30
#: (ii) ``(k w)_q0`` of the HEAVIEST arm node must resolve: >= 2^8 counts
#: (<= 0.4 % rounding), or its distance constraint quietly goes slack.
DIST_KW_MIN_COUNTS = 1 << 8
#: (ii) the bend factor's numerator ``k w |grad C|`` is formed in Q24 m/s^2
#: (its quotient by r_q0 is then a Q24 length).
BEND_NUM_SHIFT = CHAIN_SHIFT
#: Segment-length slack: the bounds assume a live segment stays within
#: [rest * stretch_min / SEGMENT_SLACK, rest * stretch_max * SEGMENT_SLACK]
#: (the distance constraint is ~rigid: kW >> 1/dt^2 on the mid row). The
#: solver guards the degenerate (near-zero) segment itself.
SEGMENT_SLACK = 2
#: (iii) ``atan2_q16`` takes int32: the Q48 cross and dot of two Q24 segment
#: vectors are narrowed by ONE row-derived shift to at most this many bits ...
ATAN2_IN_BITS = 30
#: ... and the SHORTEST segment's narrowed product must keep at least this
#: many bits (angle resolution <= 2^-16 rad, the Q16 output's own LSB).
ATAN2_MIN_BITS = 16


# ---------------------------------------------------------------------------
# Rounding + scalar conversions
# ---------------------------------------------------------------------------
def _round_half_away(v: float) -> int:
    """float -> int, round half away from zero (``fixedpoint::quantize``)."""
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"creature_fixed: non-finite value {v!r}")
    return int(math.floor(v + 0.5)) if v >= 0.0 else int(math.ceil(v - 0.5))


def chain_from_m(x: float) -> int:
    """Metres (or m/s) -> chain counts (Q24), round half away from zero, so
    ``chain_from_m(-x) == -chain_from_m(x)`` exactly. Refuses a value outside
    the int32 range (+/- ``CHAIN_RANGE_M``)."""
    q = _round_half_away(float(x) * _CHAIN_ONE_F)
    if not -CHAIN_INT32_MAX <= q <= CHAIN_INT32_MAX:
        raise ValueError(f"chain_from_m: {x!r} m is outside the chain range "
                         f"+/-{CHAIN_RANGE_M} m")
    return q


def chain_from_m_array(x) -> np.ndarray:
    """Array of metres -> int32 chain counts (round half away, range-checked)."""
    a = np.asarray(x, dtype=np.float64) * _CHAIN_ONE_F
    if not np.isfinite(a).all():
        raise ValueError("chain_from_m_array: non-finite value")
    q = np.where(a >= 0.0, np.floor(a + 0.5), np.ceil(a - 0.5))
    if (np.abs(q) > CHAIN_INT32_MAX).any():
        raise ValueError(f"chain_from_m_array: value outside the chain range "
                         f"+/-{CHAIN_RANGE_M} m")
    return q.astype(np.int32)


def chain_to_m(q) -> np.ndarray:
    """Chain counts (scalar or array) -> float64 metres, exact /2^24."""
    return np.asarray(q, dtype=np.float64) / _CHAIN_ONE_F


def chain_to_f32(q) -> np.ndarray:
    """Chain counts -> float32 metres (the render / tools bridge; never written
    back)."""
    return (np.asarray(q, dtype=np.float64) / _CHAIN_ONE_F).astype(np.float32)


def q16_from_kg(kg: float) -> int:
    """A mass in kg -> Q16 counts (round half away). Refuses a negative mass."""
    if float(kg) < 0.0:
        raise ValueError(f"q16_from_kg: negative mass {kg!r}")
    return _round_half_away(float(kg) * _Q16_ONE_F)


def q16_from_stiffness(k: float) -> int:
    """A stiffness (N/m or N m/rad) -> Q16 counts, a Python int (int64 in the
    engine: the mid row's k_dist is ~2^37 counts). Refuses a negative k."""
    if float(k) < 0.0:
        raise ValueError(f"q16_from_stiffness: negative stiffness {k!r}")
    return _round_half_away(float(k) * _Q16_ONE_F)


def q16_from_scalar(v: float) -> int:
    """A dimensionless Q16 value (gamma, a fraction) -> counts (round half
    away)."""
    return _round_half_away(float(v) * _Q16_ONE_F)


def inv_mass_q32(m_q16: int) -> int:
    """Q16 mass -> its Q32 inverse, ``2^48 // m_q16`` (exact integer floor).
    A zero mass is refused (a pinned node is w = 0 by the solver's mask, never
    by a mass)."""
    m_q16 = int(m_q16)
    if m_q16 <= 0:
        raise ValueError(f"inv_mass_q32: mass of {m_q16} counts")
    return (1 << (MASS_SHIFT + INV_MASS_SHIFT)) // m_q16


# ---------------------------------------------------------------------------
# Pose <-> Unit anchor (design 4.1)
# ---------------------------------------------------------------------------
# The creature body's POSE is the crown CENTRE in metres, in the GRID's axes
# (x East, y SOUTH -- the tile grid is y-down), with a heading in radians
# measured like ``Unit.facing``: 0 = East, +pi/2 = North (screen up, -y),
# counter-clockwise as seen on screen. ``Unit.x/y`` are float tile units of
# the footprint's TOP-LEFT anchor: tile i spans [i, i+1), so a footprint of
# side F anchored at u has its centre at u + F/2.
def pose_to_unit_anchor(x_m: float, y_m: float, footprint: int,
                        tile_size_m: float) -> tuple[float, float]:
    """Crown centre (metres, grid axes) -> ``(Unit.x, Unit.y)``, the
    footprint's top-left anchor in float tile units."""
    if not tile_size_m > 0.0:
        raise ValueError(f"tile_size_m must be > 0 (got {tile_size_m!r})")
    half = 0.5 * int(footprint)
    return (float(x_m) / tile_size_m - half, float(y_m) / tile_size_m - half)


def unit_anchor_to_pose(unit_x: float, unit_y: float, footprint: int,
                        tile_size_m: float) -> tuple[float, float]:
    """``(Unit.x, Unit.y)`` -> the crown centre in metres (grid axes); the
    exact inverse of :func:`pose_to_unit_anchor`."""
    if not tile_size_m > 0.0:
        raise ValueError(f"tile_size_m must be > 0 (got {tile_size_m!r})")
    half = 0.5 * int(footprint)
    return ((float(unit_x) + half) * tile_size_m,
            (float(unit_y) + half) * tile_size_m)


def heading_rad_to_facing(heading_rad: float) -> float:
    """The pose heading -> ``Unit.facing``. Identity today (both are radians,
    0 = East, CCW on screen); the ONE place a convention change would land."""
    return float(heading_rad)


def facing_to_heading_rad(facing: float) -> float:
    """``Unit.facing`` -> the pose heading (identity today, see above)."""
    return float(facing)


# ---------------------------------------------------------------------------
# Node masses (load-time geometry, door 2/3: + - * / and the constant pi)
# ---------------------------------------------------------------------------
def node_radii_m(nodes: int, radius_base_m: float,
                 radius_tip_m: float) -> list[float]:
    """Radius at each node, linear taper base -> tip (the prototype's
    ``ArmGeometry.radii``)."""
    n = int(nodes)
    return [radius_base_m + (radius_tip_m - radius_base_m) * i / (n - 1)
            for i in range(n)]


def node_masses_kg(nodes: int, arm_length_m: float, radius_base_m: float,
                   radius_tip_m: float, tissue_density: float) -> list[float]:
    """Each node owns half of each adjacent tapered segment, a segment being
    ``rho * pi * rbar^2 * l`` with ``rbar`` the mean of its end radii -- the
    prototype's ``ArmGeometry.masses`` exactly (P1-P5z measured with it), a
    mean-radius approximation of the truncated cone (it under-counts a cone's
    ``pi l (r1^2 + r1 r2 + r2^2) / 3`` by ``pi l (r1 - r2)^2 / 12``, 0.03 % on
    the mid row)."""
    n = int(nodes)
    r = node_radii_m(n, radius_base_m, radius_tip_m)
    seg = arm_length_m / (n - 1)
    m = [0.0] * n
    for s in range(n - 1):
        rbar = 0.5 * (r[s] + r[s + 1])
        half = 0.5 * tissue_density * math.pi * rbar * rbar * seg
        m[s] += half
        m[s + 1] += half
    return m


def node_bend_stiffness_q16(nodes: int, radius_base_m: float, radius_tip_m: float,
                            k_bend: float, k_passive_frac: float
                            ) -> tuple[tuple, tuple]:
    """The bend stiffness at each INTERIOR node (1 .. n-2), Q16 N m/rad, quantized
    once (round half away): ``(k_active, k_passive)`` with
    ``k_active_j = k_bend (r_j / r_base)^3`` (torque ~ area x lever) and
    ``k_passive_j = k_passive_frac k_bend (r_j / r_base)^3`` -- the prototype's
    ``Chain3D.k_active`` / ``k_passive`` exactly. The solver forms
    ``k_j = k_passive_j + activation_j k_active_j`` per substep in integers
    (#106 P6b; the integer reference and P6c both read these)."""
    r = node_radii_m(nodes, radius_base_m, radius_tip_m)
    act, pas = [], []
    for j in range(1, int(nodes) - 1):
        ratio = r[j] / radius_base_m
        c = ratio * ratio * ratio
        act.append(q16_from_stiffness(k_bend * c))
        pas.append(q16_from_stiffness(k_passive_frac * k_bend * c))
    return tuple(act), tuple(pas)


def node_drag_k_q16(nodes: int, arm_length_m: float, radius_base_m: float,
                    radius_tip_m: float, water_density: float, c_per: float,
                    c_tan: float) -> tuple[tuple, tuple]:
    """Each node's quadratic drag coefficients ``(K_per, K_tan)`` in Q16 kg/m,
    quantized once (round half away): the node owns half of each adjacent
    segment, ``l_i``; ``K_per = rho/2 (2 r_i l_i) c_per`` (the projected area)
    and ``K_tan = rho/2 (2 pi r_i l_i) c_tan`` (the wetted area) -- the
    prototype's ``Chain3D._K_per0`` / ``_K_tan0`` (Yekutieli I's measured
    coefficients). The drag RATE the solver uses is ``K w / (tps substeps)``,
    formed in integers from these and the node's ``w`` (#106 P6b)."""
    n = int(nodes)
    r = node_radii_m(n, radius_base_m, radius_tip_m)
    seg = arm_length_m / (n - 1)
    kp, kt = [], []
    for i in range(n):
        l_node = (0.5 * seg if i > 0 else 0.0) + (0.5 * seg if i < n - 1 else 0.0)
        kp.append(_round_half_away(0.5 * water_density * (2.0 * r[i] * l_node)
                                   * c_per * _Q16_ONE_F))
        kt.append(_round_half_away(0.5 * water_density * (2.0 * math.pi * r[i] * l_node)
                                   * c_tan * _Q16_ONE_F))
    return tuple(kp), tuple(kt)


# ---------------------------------------------------------------------------
# The headroom check
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeadroomBound:
    """One bound of the headroom check, for reports: ``value`` against
    ``limit`` (an upper bound unless ``floor``), ``margin`` = how many times
    the row could move toward the limit before it binds (> 1 passes)."""
    name: str
    formula: str
    value: float
    limit: float
    floor: bool = False

    @property
    def margin(self) -> float:
        if self.floor:
            return self.value / self.limit if self.limit else math.inf
        return self.limit / self.value if self.value else math.inf

    @property
    def ok(self) -> bool:
        return self.margin >= 1.0 if self.floor else self.value < self.limit


@dataclass(frozen=True)
class ChainHeadroom:
    """What the door quantized once for the chain, and the bounds it checked.

    ``node_mass_q16`` / ``node_inv_mass_q32`` run base -> tip; ``inv_dt2`` is
    ``(tps * substeps)^2`` (1/s^2, exact); ``bend_gamma_q16`` the XPBD
    damping ``bend_damp_s / dt_sub`` in Q16; ``atan2_shift`` the ONE narrowing
    shift taking the Q48 cross/dot of two segment vectors to <= 30 bits."""
    node_mass_q16: tuple
    node_inv_mass_q32: tuple
    body_mass_q16: int
    k_dist_q16: int
    k_bend_q16: int
    inv_dt2: int
    bend_gamma_q16: int
    stretch_min: float
    stretch_max: float
    atan2_shift: int
    bounds: tuple

    def bound(self, name: str) -> HeadroomBound:
        return next(b for b in self.bounds if b.name == name)


def chain_headroom(*, where: str, nodes_per_arm: int, arm_length_m: float,
                   radius_base_m: float, radius_tip_m: float,
                   tissue_density: float, body_mass_kg: float,
                   body_radius_m: float, k_bend: float, k_passive_frac: float,
                   k_dist: float, bend_damp_s: float, ticks_per_second: int,
                   substeps: int, stretch_min: float, stretch_max: float,
                   max_speed_m_s: float) -> ChainHeadroom:
    """Quantize the row's chain numbers once and check that the integer chain
    can hold them. Raises ``ValueError`` naming every failing bound, its value,
    its limit and the size (arm length) -- so a size is accepted or refused
    honestly, never by luck (design section 3).

    Geometry: ``n`` nodes, segments ``l = L / (n - 1)``, live segment length
    within ``[l_min, l_max] = [l * stretch_min / SLACK, l * stretch_max *
    SLACK]``. ``w_i = 2^48 // m_q16_i`` (Q32). Bend stiffness at interior node
    j: ``k_j = (1 + k_passive_frac) * k_bend * (r_j / r_base)^3`` (activation
    <= 1, the passive floor on top).

    The bounds (all magnitudes; an int64 bound is ``< 2^62``):

    (i)  mass_floor: ``min_i m_q16_i >= MASS_MIN_COUNTS`` -- the tip node
         (half a segment) is the lightest. Scales L^3: refuses SMALL animals.
         inv_mass_precision: ``2^32 / max(m_i, body) >= 2^16`` counts of w,
         i.e. the heaviest mass <= 65536 kg. Scales L^-3: refuses BIG ones.
         (w itself always fits: m_q16 >= 1 gives w <= 2^48.)
    (ii) W_fit: ``W_q32 < 2^62`` for the largest W -- distance
         ``W = w_i + w_{i+1}`` (``|grad C| = 1``, a unit vector), bend
         ``W_j <= (w_{j-1} + 4 w_j + w_{j+1}) / l_min^2`` (``|grad_end| =
         1/|d|``, the middle node's ``|g0 + g2| <= 2/l_min``). The bend term
         scales L^-5: refuses SMALL animals.
         rate_fit: ``r = max(k W (1 + gamma)) + 1/dt^2 < 2^62`` in Q0, the
         128-bit ``k_q16 * W_q32`` narrowed by ``KW_PRE_SHIFT`` (48). k W is
         SIZE-INVARIANT under the law (k_dist L^3 x w L^-3; k_bend L^5 x
         W L^-5): it refuses a stiffness typo, not a size.
         dist_factor_fit: ``(k_dist w_max)_q0 << F_SHIFT < 2^62``, the
         distance factor ``f = k w / r`` (Q30) numerator -- size-invariant.
         dist_resolution: ``(k_dist w_min)_q0 >= 2^8`` over the arm nodes --
         the heaviest node's constraint still resolves (size-invariant).
         bend_factor_fit: ``max k_j w_i |grad| (Q24 m/s^2) < 2^62`` with
         ``|grad| <= 2 / l_min`` -- the bend factor's numerator (its quotient
         by r_q0 is a Q24 length). Scales L^1.
    (iii) atan2_int32: ``l_max`` in Q24 < 2^31 (a segment vector is int32);
         the shift ``s = max(0, bitlen(l_max_q24^2) - 30)`` is DERIVED here
         and stored (``atan2_shift``): the angle is scale-free, so one shift
         per row fits every size. atan2_resolution: ``l_min_q24^2 >> s >=
         2^16`` (the shortest segment keeps a Q16-LSB angle).
    (iv) chain_range: the whole animal ``2 (body_radius + L stretch_max)`` and
         its fastest declared speed are inside the int32 Q24 range
         (``CHAIN_RANGE_M`` = 128 m, 128 m/s). Scales L^1: refuses BIG ones.
    """
    n = int(nodes_per_arm)
    tps = int(ticks_per_second)
    sub = int(substeps)
    inv_dt2 = (tps * sub) ** 2
    gamma = float(bend_damp_s) * tps * sub
    gamma_q16 = q16_from_scalar(gamma)

    masses = node_masses_kg(n, arm_length_m, radius_base_m, radius_tip_m,
                            tissue_density)
    m_q = tuple(q16_from_kg(m) for m in masses)
    body_q = q16_from_kg(body_mass_kg)
    bounds: list[HeadroomBound] = []

    # (i) masses -----------------------------------------------------------
    m_min = min(m_q)
    bounds.append(HeadroomBound(
        "mass_floor", "min node mass (Q16 counts) >= MASS_MIN_COUNTS",
        float(m_min), float(MASS_MIN_COUNTS), floor=True))
    heavy = max(max(m_q), body_q, 1)
    w_heavy = (1 << (MASS_SHIFT + INV_MASS_SHIFT)) // heavy
    bounds.append(HeadroomBound(
        "inv_mass_precision",
        "w of the heaviest mass (Q32 counts) >= INV_MASS_MIN_COUNTS",
        float(w_heavy), float(INV_MASS_MIN_COUNTS), floor=True))
    if m_min < MASS_MIN_COUNTS or m_min <= 0:
        # Nothing below can be evaluated on a zero mass.
        _refuse(where, arm_length_m, [b for b in bounds if not b.ok])
    w_q = tuple(inv_mass_q32(m) for m in m_q)
    w = [wq / float(1 << INV_MASS_SHIFT) for wq in w_q]   # 1/kg (bound math)

    # (ii) W and the rates -------------------------------------------------
    seg = arm_length_m / (n - 1)
    l_min = seg * stretch_min / SEGMENT_SLACK
    l_max = seg * stretch_max * SEGMENT_SLACK
    w_dist = max(w[i] + w[i + 1] for i in range(n - 1))
    r = node_radii_m(n, radius_base_m, radius_tip_m)
    k_j = [(1.0 + k_passive_frac) * k_bend
           * (r[j] / radius_base_m) * (r[j] / radius_base_m)
           * (r[j] / radius_base_m) for j in range(n)]
    inv_l2 = 1.0 / (l_min * l_min)
    w_bend = [(w[j - 1] + 4.0 * w[j] + w[j + 1]) * inv_l2
              for j in range(1, n - 1)]
    w_max = max([w_dist] + w_bend)
    bounds.append(HeadroomBound(
        "W_fit", "max W (Q32) < 2^62",
        w_max * float(1 << W_SHIFT), float(1 << INT64_HEADROOM_BITS)))
    kw_dist = k_dist * w_dist
    kw_bend = max(k_j[j] * w_bend[j - 1] for j in range(1, n - 1))
    rate = max(kw_dist, kw_bend * (1.0 + gamma)) + inv_dt2
    bounds.append(HeadroomBound(
        "rate_fit", "max r = k W (1 + gamma) + 1/dt^2 (Q0, 1/s^2) < 2^62",
        rate, float(1 << INT64_HEADROOM_BITS)))
    arm_w = w[1:]   # node 0 is the root (pinned to the crown)
    bounds.append(HeadroomBound(
        "dist_factor_fit", "(k_dist w_max)_q0 << F_SHIFT < 2^62",
        k_dist * max(arm_w) * float(1 << F_SHIFT),
        float(1 << INT64_HEADROOM_BITS)))
    bounds.append(HeadroomBound(
        "dist_resolution", "(k_dist w_min)_q0 >= DIST_KW_MIN_COUNTS",
        k_dist * min(arm_w), float(DIST_KW_MIN_COUNTS), floor=True))
    bend_num = max(k_j[j] * max(w[j - 1], w[j], w[j + 1]) * 2.0 / l_min
                   for j in range(1, n - 1))
    bounds.append(HeadroomBound(
        "bend_factor_fit", "max k_j w_i |grad| (Q24 m/s^2) < 2^62",
        bend_num * float(1 << BEND_NUM_SHIFT),
        float(1 << INT64_HEADROOM_BITS)))

    # (iii) atan2 ----------------------------------------------------------
    l_max_q = int(l_max * _CHAIN_ONE_F) + 1
    l_min_q = int(l_min * _CHAIN_ONE_F)
    bounds.append(HeadroomBound(
        "atan2_int32", "longest segment (Q24) < 2^31",
        float(l_max_q), float(1 << 31)))
    shift = max(0, (l_max_q * l_max_q).bit_length() - ATAN2_IN_BITS)
    bounds.append(HeadroomBound(
        "atan2_resolution",
        "shortest segment's (l_min_q24^2 >> atan2_shift) >= 2^16",
        float((l_min_q * l_min_q) >> shift), float(1 << ATAN2_MIN_BITS),
        floor=True))

    # (iv) range -----------------------------------------------------------
    bounds.append(HeadroomBound(
        "chain_range", "animal span 2 (body_r + L stretch_max) < 128 m",
        2.0 * (body_radius_m + arm_length_m * stretch_max), CHAIN_RANGE_M))
    bounds.append(HeadroomBound(
        "speed_range", "fastest declared speed < 128 m/s",
        float(max_speed_m_s), CHAIN_RANGE_M))

    failing = [b for b in bounds if not b.ok]
    if failing:
        _refuse(where, arm_length_m, failing)
    return ChainHeadroom(
        node_mass_q16=m_q, node_inv_mass_q32=w_q, body_mass_q16=body_q,
        k_dist_q16=q16_from_stiffness(k_dist),
        k_bend_q16=q16_from_stiffness(k_bend), inv_dt2=inv_dt2,
        bend_gamma_q16=gamma_q16, stretch_min=float(stretch_min),
        stretch_max=float(stretch_max), atan2_shift=shift,
        bounds=tuple(bounds))


def _refuse(where: str, arm_length_m: float, failing) -> None:
    lines = "; ".join(f"{b.name} [{b.formula}]: {b.value:.6g} vs limit "
                      f"{b.limit:.6g}" for b in failing)
    raise ValueError(f"{where}: over the integer chain's headroom at arm "
                     f"length {arm_length_m:.6g} m -- {lines}")


__all__ = [
    "CHAIN_FINE_BITS", "CHAIN_SHIFT", "CHAIN_ONE", "CHAIN_RANGE_M",
    "MASS_SHIFT", "STIFF_SHIFT", "INV_MASS_SHIFT", "W_SHIFT", "RATE_SHIFT",
    "KW_PRE_SHIFT", "F_SHIFT", "BEND_NUM_SHIFT", "INT64_HEADROOM_BITS",
    "MASS_MIN_COUNTS", "INV_MASS_MIN_COUNTS", "DIST_KW_MIN_COUNTS",
    "SEGMENT_SLACK", "ATAN2_IN_BITS", "ATAN2_MIN_BITS",
    "chain_from_m", "chain_from_m_array", "chain_to_m", "chain_to_f32",
    "q16_from_kg", "q16_from_stiffness", "q16_from_scalar", "inv_mass_q32",
    "pose_to_unit_anchor", "unit_anchor_to_pose", "heading_rad_to_facing",
    "facing_to_heading_rad", "node_radii_m", "node_masses_kg",
    "node_bend_stiffness_q16", "node_drag_k_q16",
    "HeadroomBound", "ChainHeadroom", "chain_headroom",
]
