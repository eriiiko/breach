"""THE INTEGER REFERENCE of the octopus arm chain (#106 P6b, 2026-10-30).

An exact-integer transcription of the float 3-D chain the P5z prototype
validated (`prototypes/octopus_motion/arm_chain3d.py` + `arm_bank3d.py`) in the
currencies of `src/simulation/creature_fixed.py`, following design section 6 of
`docs/octopus_p6_sim_port_design_2026-10-30.md`. It is the EXECUTABLE
SPECIFICATION the C++ solver (P6c, `cpp/src/arm_chain_solver.*`) is held to at
tol 0: every arithmetic change lands HERE first, its gates are re-run
(`chain_ref_q_gates.py`), and the C++ is written against it -- never the other
way round. Nothing in `src/` or `cpp/` imports it.

Pure Python ints in the arithmetic path: no float anywhere between the door and
the state. Floats appear only at the DOOR (a scenario's float set-up converted
once through `creature_fixed`) and in the gates' MEASUREMENTS (after
dequantizing). Python ints cannot overflow, so `CHECK = True` measures every
int64 intermediate at its narrow point and raises past 2^63 (int32 state past
2^31): the reference checks the headroom the door (`creature_fixed.chain_headroom`)
predicted, it never assumes it.

Technique credits (the float this transcribes): Macklin, Mueller, Chentanez 2016
"XPBD" (10.1145/2994258.2994272); Macklin et al. 2019 "Small steps"
(10.1145/3309486.3340247); Bishop 1975 "There is more than one way to frame a
curve" (Amer. Math. Monthly 82:246); Yekutieli et al. 2005 I/II
(10.1152/jn.00684.2004, 10.1152/jn.00685.2004) for the activation wave and the
drag law; Macklin et al. 2014 "Unified particle physics" (10.1145/2601097.2601152)
for the position-based friction. PDFs: docs/papers/octopus/.

CURRENCIES (int; Q<n> = 2^n counts per SI unit)
  positions, velocities      Q24 m, m/s (creature_fixed.CHAIN_SHIFT), int32 state
  body heading, yaw rate     Q24 rad, rad/s (int32 state; trig narrows to Q16)
  unit vectors (frame, n)    Q30 (U_SHIFT)
  masses / inverse masses    Q16 kg / Q32 1/kg (creature_fixed)
  stiffness k                Q16 N/m, N m/rad
  activation, rest scale,    Q16 (65536 = 1.0)
  bend target, angles        Q16 rad
  bend gradients g           Q16 1/m (G_SHIFT)
  bend W = sum w |g|^2       Q32 1/(kg m^2)
  constraint rate r          Q0 1/s^2  (r = k W (1 + gamma) + 1/dt^2, 1/dt^2 = 147456)
  bend dlam                  Q40 kg m^2 (DLAM_SHIFT; numerator pre-shift 8)
  distance impulse J         Q40 kg m (J = k C d/|d| / r; DIST_J_SHIFT)
  attachment impulse J       Q30 kg m (J = C / W; ATTACH_PRE)
  drag rate c = K w dt       Q32 s/m;  the dimensionless x = c |v| in Q30
  anchor force               Q8 N

ROUNDING. Every SIGNED correction and every signed narrow rounds half AWAY FROM
ZERO (`shr_round_away`, `div_round_away`, `mul_shr_away` -- fixedpoint_twins),
so +x and -x lose the same magnitude and no rounding has a direction: the "no
slide" rule (design 6.3). The only floors are `sqrt_q16` (floor of a square
root, a magnitude) and the anchor-force floor division (a non-negative
comparison). A two-node correction's shared quantity is computed ONCE (the
distance constraint's and the attachment's IMPULSE J, the bend's dlam) and each
participant moves by its own inverse mass times it, so momentum is conserved
to the 1e-10 of m w = 1. Every constraint correction is
formed FINE (Q32, FINE_BITS = 8 below a position count) and applied through
`_move`: the node moves by the rounded whole counts, the remainder (at most half
a count, per node, per substep -- scratch, not state) is carried into the
velocity rebuild.

ORDER OF ONE SUBSTEP (World.substep; dt = 1/(tps n_sub) = 1/384 s):
  1. activation   wave arms: a = a_ca S(x) from the checked-in table at the
                  arm's elapsed-SUBSTEP clock (activation_s_q16);
                  then k_j = k_pas_j + a_j k_act_j per interior node
  2. drag         per node in its own tangent frame, implicit per part
  3. gravity      v_z -= g_eff dt on every free node (w > 0)
  4. v = 0 on pinned and anchored nodes
  5. predict      disp = round(v / 384);  p_prev = p;  p += disp
  6. bodies       drag, gravity, yaw damping; x += round(v / 384), th += round(om / 384)
  7. n_pass x:    a. distance constraints, base -> tip
                  b. the Bishop frames from the current positions (frozen for c)
                  c. bends, node base -> tip, plane A then plane B
                  d. anchors (force, pop above the grip, lock)
                  e. root attachments (per body: arm 0 node 0, arm 0 node 1, ...)
                  f. grabs (sorted by arm, node)
                  g. floor (arms), floor (bodies)
                  h. static topology (#106 P6c, only with a Topology):
                     walls (arms), then the bodies' discs out of blocked
                     tiles and other units' footprints
  8. velocity     v = [384 (p - p_prev) + sra(384 r, 8)] (on the floor its z
                  part max(., 0); at a wall face its normal part clamped the
                  same way) + (v - 384 disp): the PBD rebuild of the
                  displacement INCLUDING the corrections' fine remainder r,
                  PLUS the predict's rounding remainder, so no sub-count
                  velocity is ever erased (`_rebuild`); pinned / anchored
                  v = 0; bodies likewise; heading wrapped into (-pi, pi].
                  The per-node contact report (floor / wall) is POSITIONAL,
                  read here at the end of every substep and OR-ed over the
                  tick.
The order is arm_chain3d.Chain3D.substep's with the bank's hooks
(arm_bank3d.Bank3D) -- the float P5z validated.

STATIC-TOPOLOGY CONTACTS (#106 P6c, design 4.4; no float twin -- P5z had no
walls). The topology is the map's STATIC blocked view (`GameMap.solid`:
material + door state, never `gmap.obstacles`), a tile grid of `tile_q24`
counts per tile with tile (0, 0)'s corner at `origin`; a tile outside the
grid is blocked. Tile of a coordinate: floor((x - origin) / tile)
(`Topology.tile`, the kit's floordiv_q). Walls are infinitely tall (the map
is 2-D).
  * A NODE (not pinned) inside a blocked tile is pushed out through ONE face
    whose neighbour tile is open: the face with the smallest penetration,
    ties broken in the fixed order -x, +x, -y, +y. The low face puts the node
    at `tile_low - 1` (one count inside the open neighbour), the high face at
    `tile_high` (the neighbour's first count). Its fine remainder on that axis
    is zeroed and the fine penetration feeds position-based Coulomb friction
    on the two TANGENTIAL axes (`_friction` on (the other horizontal, z)),
    the floor's law with the arm's own mu. No open neighbour: the node's x, y
    return to the substep's start (and the contact is reported).
  * At the rebuild a node AT a face (one count further toward it is in a
    blocked tile) has that velocity component clamped like the floor's
    v_z >= 0: never into the wall.
  * A BODY with a collision radius (`coll_r_q24 > 0`) is a disc in x, y: for
    every blocked tile under its bounding box (row-major, the box taken once
    per pass), then for every other unit's footprint rectangle (half-open
    tile rect, in the order given), a disc overlapping the closed rectangle
    is pushed out along (centre - closest point) by (r - d), fine, through
    the body's remainder; a centre inside the rectangle leaves through the
    nearest face (-x, +x, -y, +y on ties) by (that depth + r). No friction,
    no velocity clamp (the PBD rebuild takes the push).
  * Nodes never collide with units (Erik, 2026-10-30: arms vs marines is
    undesigned; arms pass through marines in v1). The report's unit bit
    exists for the bodies' footprint contacts only.
With no Topology every one of these is skipped: P6b's arithmetic, bit for bit.

OUTPUTS per tick (reset at the tick's start, never state): `Arm.contact` per
node (CONTACT_FLOOR | CONTACT_WALL), `Body.contact` (CONTACT_FLOOR |
CONTACT_WALL | CONTACT_UNIT), `Arm.popped`, `Arm.anchor_force` per node (the
largest anchor force seen, Q8 N: `((dist 1/dt^2) << 16) // w`, the very
comparison the grip makes; 0 on a pinned node), `World.grab_force` per grab
(the largest attachment force seen, Q8 N: each component of J / dt^2 narrowed
`sra(J 1/dt^2, ATTACH_J_SHIFT - 8)` and saturated at +-FORCE_SAT_Q8 before
the floor isqrt). `World.sub_count` is the chain clock; the C++ keeps it per
arm as the elapsed-substep clock `sub_count - t0_sub`.
"""
from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "src", _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from simulation import creature_fixed as CF          # noqa: E402
from simulation import fixedpoint_twins as K         # noqa: E402

sra = K.shr_round_away
dra = K.div_round_away
msa = K.mul_shr_away
isqrt = K.sqrt_q16

# ---------------------------------------------------------------------------
# Constants (each one a P6c literal; the currencies come from creature_fixed)
# ---------------------------------------------------------------------------
CHAIN_SHIFT = CF.CHAIN_SHIFT          # 24: positions / velocities
U_SHIFT = 30                          # unit vectors
#: Constraint corrections are computed FINE_BITS finer than a position count
#: (Q32 m); the position moves by the rounded whole counts and the fine
#: remainder (|r| <= half a count, per node, per substep: scratch, not state)
#: is carried into the velocity rebuild -- so a compliant constraint's
#: sub-count correction (the passive tip: ~1 count per 1e-4 rad per pass) and
#: a heavy body's share of an attachment (~0.3 % of the error) still
#: accelerate the node / body instead of falling into a dead band.
FINE_BITS = 8
FINE_SHIFT = CHAIN_SHIFT + FINE_BITS  # 32
U_ONE = 1 << U_SHIFT
Q16_ONE = 1 << 16
#: The distance constraint's IMPULSE J = k (C d/|d|) / r is formed in Q40 kg m
#: (k_q16 x Q24 = Q40, divided by r in Q0) and each node moves by w J: the
#: shared quantity is a MOMENTUM, so sum m dp = (m_a w_a - m_b w_b) J = 0 to
#: the 1e-10 of m w = 1 (P6a's per-node factor f = k w / r rounded each
#: node's share separately -- 1e-7 relative -- and measurably slid the free
#: animal: 1.1 mm in 60 s).
DIST_J_SHIFT = 40
#: The attachment's impulse J = C / W (kg m) is (C_q24 << ATTACH_PRE) / W_q32,
#: i.e. Q30; the node moves by -w_n J, the body by +w_b J and turns by
#: (r x J)_z / I -- momentum exact the same way. Requires |C| < 2^24 (1 m).
ATTACH_PRE = 38
ATTACH_J_SHIFT = CHAIN_SHIFT + ATTACH_PRE - 32      # 30
KW_PRE_SHIFT = CF.KW_PRE_SHIFT        # 48: (k_q16 w_q32) >> 48 -> Q0
G_SHIFT = 16                          # bend gradients, Q16 1/m
#: g = perp(a) / |a|^2 is formed as (perp(a)_q24 << G_PRE) / |a|^2_q32 -> Q16.
G_PRE = G_SHIFT + 32 - CHAIN_SHIFT    # 24
#: The bend's dlam = -k E / r is formed as ((k_q16 E_q16) << DLAM_PRE) / r_q0,
#: i.e. in Q(32 + DLAM_PRE): the PRE-SHIFT that keeps the passive tip's
#: correction resolved (design 6.2). At the mid row's last interior node
#: (k_passive 0.23 N m/rad, r ~ 1.5e5) an error E of 1e-3 rad gives dlam =
#: 1.5e-9 kg m^2 = 1700 counts (10.7 bits); the node's correction is then ~9
#: chain counts. 8 is the most the numerator allows: k E < 2^49 at |E| < 3 pi
#: on the mid row; the door's `bend_dlam_fit` holds it per size (#106 P6c).
DLAM_PRE = CF.BEND_DLAM_PRE
DLAM_SHIFT = 32 + DLAM_PRE            # 40
#: |D| (the bend's Rayleigh damping term, rad) is CLAMPED to pi: the bound that
#: keeps k (C + D) << DLAM_PRE inside int64 (|C| <= 2 pi). The float never
#: reaches it (|D| ~ gamma x the substep's turn, 1e-3 rad).
D_MAX_Q16 = K.PI_Q16
#: The Bishop transport's (1 + t_j . t_{j+1}) is floored at this many Q30
#: counts (a 180-degree fold; the float floors at 1e-9 ~ 1 count).
OPC_FLOOR = 16
#: The root normal's fallback (arm_chain3d.bishop_frames): |ref - (ref.t0) t0|
#: below 1e-3 -> ref := x (or y when |t0_x| >= 0.9).
N0_MIN = 1073742                      # round(1e-3 * 2^30)
ALT_X_LIMIT = 966367642               # round(0.9 * 2^30)
#: The body heading's wrap into (-pi, pi], Q24 rad.
PI_Q24 = 52707179                     # round(pi * 2^24)
TWO_PI_Q24 = 2 * PI_Q24
#: Drag laws (arm_chain3d): quadratic (the real one), linear (a control),
#: uniform (mass-proportional, the no-slide gate's symmetric drag).
DRAG_QUADRATIC, DRAG_LINEAR, DRAG_UNIFORM = 0, 1, 2
#: The wave clock saturates at +-WAVE_CLOCK_SPAN t_act: beyond it every node's
#: argument is past the table's +-8 (asserted at set_wave: w <= 3 L / 8).
WAVE_CLOCK_SPAN = 4
#: The fade of a released wave (hold = False) trails the front by 2/5 L.
FADE_BACK_NUM, FADE_BACK_DEN = 2, 5
#: The per-tick contact report's bits (#106 P6c).
CONTACT_FLOOR, CONTACT_WALL, CONTACT_UNIT = 1, 2, 4
#: The grab force's components are saturated here (Q8 N, 4.2 MN) before they
#: are squared, so the floor isqrt's radicand stays below 3 x 2^60.
FORCE_SAT_Q8 = 1 << 30

# ---------------------------------------------------------------------------
# Headroom checks (CHECK = True): each site's largest |value| in bits
# ---------------------------------------------------------------------------
CHECK = False
BITS: dict[str, int] = {}
_LIMIT = {}


def _chk(site: str, v: int, bits: int = 63) -> int:
    b = abs(v).bit_length()
    if b > BITS.get(site, 0):
        BITS[site] = b
        _LIMIT[site] = bits
    if b >= bits:
        raise OverflowError(f"chain_ref_q: {site} = {v} needs {b} bits (limit {bits})")
    return v


def reset_bits() -> None:
    BITS.clear()
    _LIMIT.clear()


def bits_report() -> list[tuple[str, int, int]]:
    """[(site, max bits seen, limit)] sorted by site."""
    return sorted((s, b, _LIMIT[s]) for s, b in BITS.items())


# ---------------------------------------------------------------------------
# The activation table (the ONE source: cpp/src/activation_table.inc)
# ---------------------------------------------------------------------------
def load_activation_table(path=None) -> tuple:
    """The checked-in S(x) samples (Q16), parsed from the .inc the engine
    compiles (tools/gen_activation_table.parse_inc)."""
    import gen_activation_table as GEN
    meta = GEN.parse_inc() if path is None else GEN.parse_inc(Path(path))
    if (meta["x_max"], meta["step_shift"]) != (GEN.ACT_X_MAX, GEN.ACT_STEP_SHIFT):
        raise ValueError("activation table: grid differs from the lookup law's")
    return meta["data"]


ACT_X_MAX_Q16 = 8 << 16
ACT_STEP_SHIFT = 10


def activation_s_q16(x_q16: int, tab) -> int:
    """S(x) = (1 + tanh x) / 2 through the table: clamp x to [-8, 8], index =
    (x + 8) >> 10 (step 1/64 = 2^10 Q16 counts), linear interpolation on the
    low 10 bits, round half away (the slope is >= 0, so half up)."""
    if x_q16 < -ACT_X_MAX_Q16:
        x_q16 = -ACT_X_MAX_Q16
    elif x_q16 > ACT_X_MAX_Q16:
        x_q16 = ACT_X_MAX_Q16
    u = x_q16 + ACT_X_MAX_Q16
    i = u >> ACT_STEP_SHIFT
    last = len(tab) - 2
    if i > last:
        i = last
    f = u - (i << ACT_STEP_SHIFT)
    t0 = tab[i]
    return t0 + sra((tab[i + 1] - t0) * f, ACT_STEP_SHIFT)


# ---------------------------------------------------------------------------
# The row, quantized once (from a CreatureDef: the door P6a built)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainRow:
    """Everything the chain needs from a `[creatures.*]` row, in integers.
    `chain_row(cdef)` builds it through creature_fixed (the door)."""
    n_nodes: int
    n_sub: int
    n_pass: int
    inv_dt: int                 # tps * n_sub (384)
    inv_dt2: int                # 147456
    L_q24: int
    rest0_q24: tuple            # (n_nodes - 1,)
    s_node_q24: tuple           # (n_nodes,) arc length at each node
    m_q16: tuple                # (n_nodes,)
    w_q32: tuple                # (n_nodes,) unpinned
    k_dist_q16: int
    k_act_q16: tuple            # (n_nodes - 2,) interior nodes 1 .. n-2
    k_pas_q16: tuple
    gamma_q16: int
    onepg_q16: int              # 1 + gamma
    atan2_shift: int
    plane_min_q32: int          # the bend's in-plane guard, Q32 m^2 (#106 P6c)
    K_per_q16: tuple            # (n_nodes,) kg/m
    K_tan_q16: tuple
    g_q24: int                  # g_eff, Q24 m/s^2
    g_dv_q24: int               # g_eff dt, the per-substep velocity step
    rho_water_q16: int          # kg/m^3 (the body's drag)
    front_w_q24: int            # the reach's default front width

    @property
    def n_seg(self) -> int:
        return self.n_nodes - 1

    @property
    def n_bend(self) -> int:
        return self.n_nodes - 2


def chain_row(cdef, ticks_per_second: int = 24) -> ChainRow:
    """A CreatureDef (creatures.get_creature / build_creature) -> ChainRow."""
    b, t, s, ch = cdef.body, cdef.tissue, cdef.solver, cdef.chain
    n = int(b.nodes_per_arm)
    inv_dt = int(ticks_per_second) * int(s.substeps)
    if inv_dt * inv_dt != ch.inv_dt2:
        raise ValueError("chain_row: the row's 1/dt^2 disagrees with tps x substeps")
    seg = CF.chain_from_m(b.arm_length_m / (n - 1))
    rest0 = (seg,) * (n - 1)
    s_node = tuple(seg * i for i in range(n))
    k_act, k_pas = CF.node_bend_stiffness_q16(n, b.radius_base_m, b.radius_tip_m,
                                              t.k_bend, t.k_passive_frac)
    kp, kt = CF.node_drag_k_q16(n, b.arm_length_m, b.radius_base_m, b.radius_tip_m,
                                b.water_density, t.c_drag_perp, t.c_drag_tan)
    g_q24 = CF.chain_from_m(t.g_eff)
    return ChainRow(
        n_nodes=n, n_sub=int(s.substeps), n_pass=int(s.passes), inv_dt=inv_dt,
        inv_dt2=ch.inv_dt2, L_q24=CF.chain_from_m(b.arm_length_m), rest0_q24=rest0,
        s_node_q24=s_node, m_q16=tuple(ch.node_mass_q16), w_q32=tuple(ch.node_inv_mass_q32),
        k_dist_q16=ch.k_dist_q16, k_act_q16=k_act, k_pas_q16=k_pas,
        gamma_q16=ch.bend_gamma_q16, onepg_q16=Q16_ONE + ch.bend_gamma_q16,
        atan2_shift=ch.atan2_shift, plane_min_q32=ch.bend_plane_min_q32, K_per_q16=kp, K_tan_q16=kt, g_q24=g_q24,
        g_dv_q24=dra(g_q24, inv_dt), rho_water_q16=CF.q16_from_scalar(b.water_density),
        front_w_q24=CF.chain_from_m(cdef.reach.front_width_m))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
@dataclass
class Topology:
    """The map's STATIC topology as the chain sees it (#106 P6c): a blocked
    grid (h rows of w bools; `GameMap.solid` = material + door state), the
    tile size and the grid origin in chain counts, and the OTHER units'
    footprint rectangles (half-open tile rects (tx0, ty0, tx1, ty1); the
    bodies collide with them, the nodes never)."""
    grid: list
    tile_q24: int
    origin: tuple = (0, 0)
    unit_rects: tuple = ()

    def blocked(self, ix: int, iy: int) -> bool:
        if ix < 0 or iy < 0 or iy >= len(self.grid) or ix >= len(self.grid[0]):
            return True
        return bool(self.grid[iy][ix])

    def tile(self, x: int, axis: int) -> int:
        """The tile index of a chain coordinate: floor((x - origin) / tile)."""
        return (x - self.origin[axis]) // self.tile_q24


def topology(blocked, tile_size_m: float, origin_m=(0.0, 0.0), unit_rects=()) -> Topology:
    """A Topology from a (h, w) bool array (the door: metres converted once)."""
    grid = [[bool(c) for c in row] for row in blocked]
    return Topology(grid=grid, tile_q24=CF.chain_from_m(tile_size_m),
                    origin=(CF.chain_from_m(origin_m[0]), CF.chain_from_m(origin_m[1])),
                    unit_rects=tuple(tuple(int(c) for c in r) for r in unit_rects))


@dataclass
class Wave:
    """A reach's stiffening wave (arm_chain.ReachCommand): amplitude Q16, the
    travel time in SUBSTEPS, the start in substeps of the world clock, the front
    width Q24 m, hold (False: the activation fades behind the front)."""
    a_ca_q16: int
    tact_sub: int
    t0_sub: int
    w_q24: int
    hold: bool = True


@dataclass
class Arm:
    p: list                      # n x [x, y, z] Q24
    v: list                      # n x [vx, vy, vz] Q24
    pinned: tuple                # n bools
    w: list                      # Q32, 0 on a pinned node
    r_dist: list                 # per segment r = k (w_a + w_b) + 1/dt^2, Q0
    c_per: list                  # Q32 s/m drag rates
    c_tan: list
    drag_mode: int = DRAG_QUADRATIC
    vref_q24: int = 0            # the linear law's reference speed
    u_q30: int = 0               # the uniform law's dt / tau
    mu_q16: int = 0              # deck friction
    root_ref: tuple = (0, 0, U_ONE)
    act: list = field(default_factory=list)       # Q16 per interior node
    rest: list = field(default_factory=list)      # Q24 per segment (rest0 x scale)
    rest_scale: list = field(default_factory=list)  # Q16 per segment (the INPUT rest is derived from)
    target: list = field(default_factory=list)    # per interior node [A, B] Q16 rad
    anchored: list = field(default_factory=list)  # bool per node
    anchor_pos: list = field(default_factory=list)
    popped: list = field(default_factory=list)
    grip_q8: int | None = None   # None = no limit
    wave: Wave | None = None
    body: int = -1
    root_local: list = field(default_factory=list)  # 2 x [x, y, z] Q24, body frame
    k_sub: list = field(default_factory=list)       # Q16 per interior node, this substep
    res: list = field(default_factory=list)         # n x 3 fine remainders, this substep
    contact: list = field(default_factory=list)     # OUT: CONTACT_* bits per node, this tick
    anchor_force: list = field(default_factory=list)  # OUT: Q8 N per node, this tick's largest


@dataclass
class Body:
    """A rigid disc with a 2-D pose plus height (arm_bank3d's bodies)."""
    x: list                      # Q24 m
    v: list                      # Q24 m/s
    th: int                      # Q24 rad
    om: int                      # Q24 rad/s
    w_q32: int                   # 1 / mass
    winvI_q32: int               # 1 / inertia, Q32 1/(kg m^2)
    cb_q32: int                  # quadratic drag rate, Q32 s/m
    drag_mode: int = DRAG_QUADRATIC
    vref_q24: int = 0
    u_q30: int = 0
    y_q30: int = 0               # yaw damping rate dt rot_damp / I (or dt / tau)
    mu_q16: int = 0
    gravity: bool = True
    floor: bool = True
    coll_r_q24: int = 0          # the collision disc's radius (0: no topology collision)
    res: list = field(default_factory=lambda: [0, 0, 0])   # fine remainders, this substep
    res_th: int = 0
    contact: int = 0             # OUT: CONTACT_* bits, this tick


def _q16(x: float) -> int:
    return CF.q16_from_scalar(x)


def _q24v(v) -> list:
    return [CF.chain_from_m(float(c)) for c in v]


def _unit_q30(v) -> tuple:
    """A float direction (door) -> a Q30 unit vector."""
    import math
    n = math.sqrt(sum(float(c) ** 2 for c in v))
    return tuple(CF._round_half_away(float(c) / n * U_ONE) for c in v)


class World:
    """The arms of every animal in a scenario, their bodies and grabs, stepped
    together (arm_bank3d.World's bank). Batching never changes a result: arms
    of different bodies never interact inside a substep."""

    def __init__(self, row: ChainRow, act_table=None, floor_on: bool = True,
                 topo: Topology | None = None):
        self.row = row
        self.tab = load_activation_table() if act_table is None else act_table
        self.arms: list[Arm] = []
        self.bodies: list[Body] = []
        self.grabs: list[tuple] = []      # (arm, node, body, local Q24 [3])
        self.grab_force: list[int] = []   # OUT: Q8 N per grab, this tick's largest
        self.floor_on = floor_on
        self.topo = topo                  # None: no static-topology contact (P6b)
        self.sub_count = 0                # the chain clock, in substeps

    # -- set-up (the door: floats converted ONCE) --------------------------------
    def add_body(self, x, theta=0.0, mass=1.0, radius=0.1, c_drag=1.0, area=0.0, rot_damp=0.0,
                 mu=0.0, inertia=None, gravity=True, floor=True, drag_mode=DRAG_QUADRATIC,
                 tau=1.0, vref=0.0, coll_radius=None) -> int:
        row = self.row
        m_q16 = CF.q16_from_kg(mass)
        I = 0.5 * mass * radius ** 2 if inertia is None else inertia
        w = CF.inv_mass_q32(m_q16)
        winvI = CF.inv_mass_q32(CF.q16_from_kg(I))
        kb_q16 = CF._round_half_away(0.5 * (row.rho_water_q16 / 65536.0) * c_drag * area * 65536.0)
        b = Body(x=_q24v(x), v=[0, 0, 0], th=CF.chain_from_m(theta), om=0, w_q32=w,
                 winvI_q32=winvI, cb_q32=dra(kb_q16 * w, row.inv_dt << 16), drag_mode=int(drag_mode),
                 vref_q24=CF.chain_from_m(vref), mu_q16=_q16(mu), gravity=bool(gravity),
                 floor=bool(floor),
                 coll_r_q24=0 if coll_radius is None else CF.chain_from_m(coll_radius))
        tau_q16 = _q16(tau)
        b.u_q30 = dra(1 << 46, row.inv_dt * tau_q16) if tau_q16 > 0 else 0
        if b.drag_mode == DRAG_UNIFORM:
            b.y_q30 = b.u_q30
        else:
            b.y_q30 = dra(_q16(rot_damp) * winvI, row.inv_dt << 18)
        self.bodies.append(b)
        return len(self.bodies) - 1

    def add_arm(self, p0, pinned=(), mu=0.0, drag_mode=DRAG_QUADRATIC, tau=1.0, vref=0.0,
                root_ref=(0.0, 0.0, 1.0), body=-1, root_local=None, grip=None, v0=None) -> int:
        row = self.row
        n = row.n_nodes
        pin = tuple(i in set(int(k) for k in pinned) for i in range(n))
        w = [0 if pin[i] else row.w_q32[i] for i in range(n)]
        r_dist = [msa(row.k_dist_q16, w[i] + w[i + 1], KW_PRE_SHIFT) + row.inv_dt2
                  for i in range(row.n_seg)]
        c_per = [dra(row.K_per_q16[i] * w[i], row.inv_dt << 16) for i in range(n)]
        c_tan = [dra(row.K_tan_q16[i] * w[i], row.inv_dt << 16) for i in range(n)]
        if CHECK:
            for i in range(n):
                _chk("arm.K_w", max(row.K_per_q16[i], row.K_tan_q16[i]) * w[i])
        tau_q16 = _q16(tau)
        a = Arm(p=[_q24v(q) for q in p0],
                v=[_q24v(q) for q in v0] if v0 is not None else [[0, 0, 0] for _ in range(n)],
                pinned=pin, w=w, r_dist=r_dist, c_per=c_per, c_tan=c_tan, drag_mode=int(drag_mode),
                vref_q24=CF.chain_from_m(vref),
                u_q30=dra(1 << 46, row.inv_dt * tau_q16) if tau_q16 > 0 else 0,
                mu_q16=_q16(mu), root_ref=_unit_q30(root_ref),
                act=[0] * row.n_bend, rest=list(row.rest0_q24), rest_scale=[Q16_ONE] * row.n_seg,
                target=[[0, 0] for _ in range(row.n_bend)], anchored=[False] * n,
                anchor_pos=[[0, 0, 0] for _ in range(n)], popped=[False] * n,
                grip_q8=None, body=int(body), k_sub=list(row.k_pas_q16),
                contact=[0] * n, anchor_force=[0] * n)
        if root_local is not None:
            a.root_local = [_q24v(q) for q in root_local]
        self.set_grip(a, grip)
        for i in range(n):
            if pin[i]:
                a.v[i] = [0, 0, 0]
        self.arms.append(a)
        return len(self.arms) - 1

    # -- per-tick inputs (what a hub writes) -------------------------------------
    @staticmethod
    def set_grip(arm: Arm, grip) -> None:
        import math
        arm.grip_q8 = None if grip is None or not math.isfinite(float(grip)) else \
            CF._round_half_away(float(grip) * 256.0)

    def set_wave(self, ai: int, a_ca: float, t_act_s: float, t_start_s: float = 0.0,
                 width_m: float | None = None, hold: bool = True) -> None:
        row = self.row
        w_q24 = row.front_w_q24 if width_m is None else CF.chain_from_m(width_m)
        if 8 * w_q24 > 3 * row.L_q24:
            raise ValueError("set_wave: front width > 3 L / 8 (the clock's saturation assumes it)")
        self.arms[ai].wave = Wave(_q16(a_ca), CF._round_half_away(t_act_s * row.inv_dt),
                                  CF._round_half_away(t_start_s * row.inv_dt), w_q24, bool(hold))

    def set_activation(self, ai: int, act) -> None:
        self.arms[ai].act = [_q16(a) for a in act]

    def set_rest_scale(self, ai: int, scale) -> None:
        r0 = self.row.rest0_q24
        a = self.arms[ai]
        a.rest_scale = [_q16(s) for s in scale]
        a.rest = [sra(r0[i] * q, 16) for i, q in enumerate(a.rest_scale)]

    def set_bend_target(self, ai: int, target) -> None:
        self.arms[ai].target = [[_q16(t[0]), _q16(t[1])] for t in target]

    def anchor(self, ai: int, nodes) -> None:
        """Anchor nodes where they ARE (the hub's `anchor`)."""
        a = self.arms[ai]
        for k in nodes:
            a.anchored[k] = True
            a.anchor_pos[k] = list(a.p[k])

    def release(self, ai: int) -> None:
        a = self.arms[ai]
        a.anchored = [False] * len(a.anchored)

    # -- one substep ---------------------------------------------------------------
    def substep(self) -> None:
        row = self.row
        for a in self.arms:
            self._activation(a)
            self._drag(a)
            gdv = row.g_dv_q24
            for i in range(row.n_nodes):
                if a.w[i] > 0 and gdv:
                    a.v[i][2] -= gdv
                if a.pinned[i] or a.anchored[i]:
                    a.v[i] = [0, 0, 0]
        p_prev, disp = [], []
        for a in self.arms:
            p_prev.append([list(q) for q in a.p])
            a.res = [[0, 0, 0] for _ in range(row.n_nodes)]
            d = []
            for i in range(row.n_nodes):
                v = a.v[i]
                di = [dra(v[0], row.inv_dt), dra(v[1], row.inv_dt), dra(v[2], row.inv_dt)]
                q = a.p[i]
                q[0] += di[0]
                q[1] += di[1]
                q[2] += di[2]
                d.append(di)
            disp.append(d)
        b_prev, b_disp = self._predict_bodies()
        for _ in range(row.n_pass):
            for ai, a in enumerate(self.arms):
                self._project_distance(a)
                frames = self._frames(a)
                self._project_bend(a, frames, p_prev[ai])
            self._project_extra()
            if self.floor_on:
                for ai, a in enumerate(self.arms):
                    self._project_floor(a, p_prev[ai])
                self._floor_bodies(b_prev)
            if self.topo is not None:
                for ai, a in enumerate(self.arms):
                    self._project_walls(a, p_prev[ai])
                self._collide_bodies()
        inv_dt = row.inv_dt
        topo = self.topo
        for ai, a in enumerate(self.arms):
            pp, dd = p_prev[ai], disp[ai]
            for i in range(row.n_nodes):
                q = a.p[i]
                on_floor = self.floor_on and q[2] <= 0
                touch = _wall_touch(topo, q) if topo is not None else _NO_TOUCH
                if on_floor:
                    a.contact[i] |= CONTACT_FLOOR
                if touch[4]:
                    a.contact[i] |= CONTACT_WALL
                if a.pinned[i] or a.anchored[i]:
                    a.v[i] = [0, 0, 0]
                    continue
                v, qp, di, rr = a.v[i], pp[i], dd[i], a.res[i]
                v[0] = _rebuild(v[0], q[0] - qp[0], rr[0], di[0], inv_dt, touch[0], touch[1])
                v[1] = _rebuild(v[1], q[1] - qp[1], rr[1], di[1], inv_dt, touch[2], touch[3])
                v[2] = _rebuild(v[2], q[2] - qp[2], rr[2], di[2], inv_dt, on_floor)
                if CHECK:
                    for c in range(3):
                        _chk("state.p", q[c], 31)
                        _chk("state.v", v[c], 31)
        for bi, b in enumerate(self.bodies):
            xp, (dx, dth) = b_prev[bi], b_disp[bi]
            if self.floor_on and b.floor and b.x[2] <= 0:
                b.contact |= CONTACT_FLOOR
            for c in range(3):
                b.v[c] = _rebuild(b.v[c], b.x[c] - xp[0][c], b.res[c], dx[c], inv_dt,
                                  c == 2 and self.floor_on and b.floor and b.x[2] <= 0)
            b.om = _rebuild(b.om, b.th - xp[1], b.res_th, dth, inv_dt, False)
            if b.th > PI_Q24:
                b.th -= TWO_PI_Q24
            elif b.th <= -PI_Q24:
                b.th += TWO_PI_Q24
            if CHECK:
                for c in range(3):
                    _chk("state.body_x", b.x[c], 31)
                    _chk("state.body_v", b.v[c], 31)
                _chk("state.body_om", b.om, 31)
        self.sub_count += 1

    def tick(self) -> None:
        for a in self.arms:
            n = len(a.popped)
            a.popped = [False] * n
            a.contact = [0] * n
            a.anchor_force = [0] * n
        for b in self.bodies:
            b.contact = 0
        self.grab_force = [0] * len(self.grabs)
        for _ in range(self.row.n_sub):
            self.substep()

    # -- 1. activation -----------------------------------------------------------------
    def _activation(self, a: Arm) -> None:
        row = self.row
        wv = a.wave
        if wv is not None:
            span = WAVE_CLOCK_SPAN * wv.tact_sub
            ne = self.sub_count - wv.t0_sub
            ne = -span if ne < -span else (span if ne > span else ne)
            den = wv.w_q24 * wv.tact_sub
            front = row.L_q24 * ne                         # s_front * tact_sub
            back = dra(FADE_BACK_NUM * row.L_q24, FADE_BACK_DEN) * wv.tact_sub
            act = []
            for j in range(row.n_bend):
                sj = row.s_node_q24[j + 1] * wv.tact_sub
                x = dra((front - sj) << 16, den)
                aj = sra(wv.a_ca_q16 * activation_s_q16(x, self.tab), 16)
                if not wv.hold:
                    x2 = dra((sj - front + back) << 16, den)
                    aj = sra(aj * activation_s_q16(x2, self.tab), 16)
                act.append(aj)
            a.act = act
        ka, kp = row.k_act_q16, row.k_pas_q16
        a.k_sub = [kp[j] + sra(a.act[j] * ka[j], 16) for j in range(row.n_bend)]

    # -- 2. drag -------------------------------------------------------------------------
    def _drag(self, a: Arm) -> None:
        n = self.row.n_nodes
        p, v = a.p, a.v
        mode = a.drag_mode
        for i in range(n):
            if a.w[i] == 0:
                continue                    # pinned: no drag (and v is zeroed)
            vi = v[i]
            if mode == DRAG_UNIFORM:
                den = U_ONE + a.u_q30
                vi[0] = dra(vi[0] << U_SHIFT, den)
                vi[1] = dra(vi[1] << U_SHIFT, den)
                vi[2] = dra(vi[2] << U_SHIFT, den)
                continue
            lo, hi = (i - 1 if i > 0 else 0), (i + 1 if i < n - 1 else n - 1)
            tx = p[hi][0] - p[lo][0]
            ty = p[hi][1] - p[lo][1]
            tz = p[hi][2] - p[lo][2]
            ln = isqrt(tx * tx + ty * ty + tz * tz)
            if ln:
                tx, ty, tz = dra(tx << U_SHIFT, ln), dra(ty << U_SHIFT, ln), dra(tz << U_SHIFT, ln)
            else:
                tx = ty = tz = 0
            vt = sra(vi[0] * tx + vi[1] * ty + vi[2] * tz, U_SHIFT)
            vtx, vty, vtz = sra(vt * tx, U_SHIFT), sra(vt * ty, U_SHIFT), sra(vt * tz, U_SHIFT)
            nx, ny, nz = vi[0] - vtx, vi[1] - vty, vi[2] - vtz
            if mode == DRAG_QUADRATIC:
                sn = isqrt(nx * nx + ny * ny + nz * nz)
                xt = sra(a.c_tan[i] * (vt if vt >= 0 else -vt), 26)
                xn = sra(a.c_per[i] * sn, 26)
            else:                           # DRAG_LINEAR: K_per v_ref on both parts
                xt = xn = sra(a.c_per[i] * a.vref_q24, 26)
            if CHECK:
                _chk("drag.v_dot_t", vi[0] * tx + vi[1] * ty + vi[2] * tz)
                _chk("drag.n_sq", nx * nx + ny * ny + nz * nz)
                _chk("drag.c_v", a.c_per[i] * (sn if mode == DRAG_QUADRATIC else a.vref_q24))
            if xt == 0 and xn == 0:
                continue
            vt2 = dra(vt << U_SHIFT, U_ONE + xt)
            dn = U_ONE + xn
            vi[0] = sra(vt2 * tx, U_SHIFT) + dra(nx << U_SHIFT, dn)
            vi[1] = sra(vt2 * ty, U_SHIFT) + dra(ny << U_SHIFT, dn)
            vi[2] = sra(vt2 * tz, U_SHIFT) + dra(nz << U_SHIFT, dn)

    # -- 6. bodies ------------------------------------------------------------------------
    def _predict_bodies(self):
        row = self.row
        prev, disp = [], []
        for b in self.bodies:
            prev.append((list(b.x), b.th))
            b.res = [0, 0, 0]
            b.res_th = 0
            v = b.v
            if b.drag_mode == DRAG_UNIFORM:
                x = b.u_q30
            elif b.drag_mode == DRAG_LINEAR:
                x = sra(b.cb_q32 * b.vref_q24, 26)
            else:
                x = sra(b.cb_q32 * isqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), 26)
            if x:
                den = U_ONE + x
                v[0], v[1], v[2] = (dra(v[0] << U_SHIFT, den), dra(v[1] << U_SHIFT, den),
                                    dra(v[2] << U_SHIFT, den))
            if b.gravity and row.g_dv_q24:
                v[2] -= row.g_dv_q24
            if b.y_q30:
                b.om = dra(b.om << U_SHIFT, U_ONE + b.y_q30)
            d = [dra(v[0], row.inv_dt), dra(v[1], row.inv_dt), dra(v[2], row.inv_dt)]
            dth = dra(b.om, row.inv_dt)
            b.x[0] += d[0]
            b.x[1] += d[1]
            b.x[2] += d[2]
            b.th += dth
            disp.append((d, dth))
        return prev, disp

    # -- 7a. distance -----------------------------------------------------------------------
    def _project_distance(self, a: Arm) -> None:
        p, rest, rd, res, w = a.p, a.rest, a.r_dist, a.res, a.w
        k = self.row.k_dist_q16
        js = DIST_J_SHIFT                  # Q32 x Q40 >> 40 -> Q32 fine
        for i in range(self.row.n_seg):
            P, Q = p[i], p[i + 1]
            dx, dy, dz = Q[0] - P[0], Q[1] - P[1], Q[2] - P[2]
            ln = isqrt(dx * dx + dy * dy + dz * dz)
            if ln == 0:
                continue
            C = ln - rest[i]
            if C == 0:
                continue
            # the shared IMPULSE J = k C (d/|d|) / r (Q40 kg m), computed ONCE,
            # each node moving by its w J (+ at the base end, - at the tip end)
            cx, cy, cz = dra(C * dx, ln), dra(C * dy, ln), dra(C * dz, ln)
            r = rd[i]
            jx, jy, jz = dra(k * cx, r), dra(k * cy, r), dra(k * cz, r)
            if CHECK:
                _chk("dist.len_sq", dx * dx + dy * dy + dz * dz)
                _chk("dist.C_d", C * (dx if abs(dx) >= abs(dy) else dy))
                _chk("dist.k_Cd", k * max(abs(cx), abs(cy), abs(cz)))
            wa, wb = w[i], w[i + 1]
            if wa:
                _move(P, res[i], msa(wa, jx, js), msa(wa, jy, js), msa(wa, jz, js))
            if wb:
                _move(Q, res[i + 1], -msa(wb, jx, js), -msa(wb, jy, js), -msa(wb, jz, js))

    # -- 7b. the Bishop frame ------------------------------------------------------------------
    def _frames(self, a: Arm):
        """(t, nA, nB) per segment 0 .. n_bend-1, Q30 unit vectors, from the
        CURRENT positions (arm_chain3d.bishop_frames, transcribed)."""
        nb = self.row.n_bend
        p = a.p
        t = []
        for j in range(nb):
            P, Q = p[j], p[j + 1]
            dx, dy, dz = Q[0] - P[0], Q[1] - P[1], Q[2] - P[2]
            ln = isqrt(dx * dx + dy * dy + dz * dz)
            t.append((dra(dx << U_SHIFT, ln), dra(dy << U_SHIFT, ln), dra(dz << U_SHIFT, ln))
                     if ln else (0, 0, 0))
        t0 = t[0]
        ref = a.root_ref
        n0 = _reject(ref, t0)
        nn = isqrt(n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2])
        if nn < N0_MIN:
            alt = (U_ONE, 0, 0) if abs(t0[0]) < ALT_X_LIMIT else (0, U_ONE, 0)
            n0 = _reject(alt, t0)
            nn = isqrt(n0[0] * n0[0] + n0[1] * n0[1] + n0[2] * n0[2])
        nA = [(dra(n0[0] << U_SHIFT, nn), dra(n0[1] << U_SHIFT, nn), dra(n0[2] << U_SHIFT, nn))]
        for j in range(nb - 1):
            tj, tk, n = t[j], t[j + 1], nA[j]
            c = sra(tj[0] * tk[0] + tj[1] * tk[1] + tj[2] * tk[2], U_SHIFT)
            opc = U_ONE + c
            if opc < OPC_FLOOR:
                opc = OPC_FLOOR
            nt = sra(n[0] * tk[0] + n[1] * tk[1] + n[2] * tk[2], U_SHIFT)
            f = dra(nt << U_SHIFT, opc)
            m = (n[0] - msa(f, tj[0] + tk[0], U_SHIFT), n[1] - msa(f, tj[1] + tk[1], U_SHIFT),
                 n[2] - msa(f, tj[2] + tk[2], U_SHIFT))
            if CHECK:
                _chk("frame.f", f)
                for c_ in m:
                    _chk("frame.n_transported", c_, 32)
            m = _reject(m, tk)
            mm = isqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2])
            nA.append((dra(m[0] << U_SHIFT, mm), dra(m[1] << U_SHIFT, mm), dra(m[2] << U_SHIFT, mm))
                      if mm else n)
        nB = [_cross_q30(nA[j], t[j]) for j in range(nb)]
        return t, nA, nB

    # -- 7c. bends -------------------------------------------------------------------------------
    def _project_bend(self, a: Arm, frames, pp) -> None:
        row = self.row
        t, nA, nB = frames
        p, w, k, tgt, res = a.p, a.w, a.k_sub, a.target, a.res
        ds = 32 - FINE_BITS
        sh = row.atan2_shift
        gam = row.gamma_q16
        onepg = row.onepg_q16
        inv_dt2 = row.inv_dt2
        pmin = row.plane_min_q32
        for j in range(row.n_bend):
            w0, w1, w2 = w[j], w[j + 1], w[j + 2]
            if w0 == 0 and w1 == 0 and w2 == 0:
                continue
            e1 = t[j]
            kj = k[j]
            for plane in (0, 1):
                eX = nA[j] if plane == 0 else nB[j]
                P0, P1, P2 = p[j], p[j + 1], p[j + 2]
                d1 = (P1[0] - P0[0], P1[1] - P0[1], P1[2] - P0[2])
                d2 = (P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2])
                a1x, a1y = _ip(d1, e1), _ip(d1, eX)
                a2x, a2y = _ip(d2, e1), _ip(d2, eX)
                n1 = sra(a1x * a1x + a1y * a1y, 16)          # |a1|^2, Q32
                n2 = sra(a2x * a2x + a2y * a2y, 16)
                # the in-plane guard (#106 P6c): a plane in which either
                # segment's projection is shorter than the door's l_min is
                # skipped -- the angle there is ill-conditioned and W ~ 1/|a|^2
                # would leave the bound the door checked (CHECK: 63 bits at
                # 3.39 m on a tip folded against a corner)
                if n1 < pmin or n2 < pmin:
                    continue
                cross = a1x * a2y - a1y * a2x                 # Q48
                dot = a1x * a2x + a1y * a2y
                yn, xn = sra(cross, sh), sra(dot, sh)
                if CHECK:
                    _chk("bend.atan2_y", yn, 31)
                    _chk("bend.atan2_x", xn, 31)
                C = K.atan2_q16(yn, xn) - tgt[j][plane]
                g0x, g0y = dra((-a1y) << G_PRE, n1), dra(a1x << G_PRE, n1)
                g2x, g2y = dra((-a2y) << G_PRE, n2), dra(a2x << G_PRE, n2)
                g1x, g1y = -(g0x + g2x), -(g0y + g2y)
                W = 0
                if w0:
                    W += msa(w0, g0x * g0x + g0y * g0y, 32)
                if w1:
                    W += msa(w1, g1x * g1x + g1y * g1y, 32)
                if w2:
                    W += msa(w2, g2x * g2x + g2y * g2y, 32)
                kW = msa(kj, W, 32)                           # Q16 1/s^2
                r = sra(kW * onepg, 32) + inv_dt2             # Q0
                Q0, Q1, Q2 = pp[j], pp[j + 1], pp[j + 2]
                q0 = (P0[0] - Q0[0], P0[1] - Q0[1], P0[2] - Q0[2])
                q1 = (P1[0] - Q1[0], P1[1] - Q1[1], P1[2] - Q1[2])
                q2 = (P2[0] - Q2[0], P2[1] - Q2[1], P2[2] - Q2[2])
                S = (g0x * _ip(q0, e1) + g0y * _ip(q0, eX) + g1x * _ip(q1, e1) + g1y * _ip(q1, eX)
                     + g2x * _ip(q2, e1) + g2y * _ip(q2, eX))   # Q40 rad
                D = sra(S * gam, 40)
                if D > D_MAX_Q16:
                    D = D_MAX_Q16
                elif D < -D_MAX_Q16:
                    D = -D_MAX_Q16
                E = C + D
                kE = kj * E                                   # Q32 N m
                dlam = -dra(kE << DLAM_PRE, r)                # Q40 kg m^2
                if CHECK:
                    _chk("bend.W", W)
                    _chk("bend.kW", kW)
                    _chk("bend.kW_onepg", kW * onepg)
                    _chk("bend.r", r)
                    _chk("bend.S", S)
                    _chk("bend.S_gamma", S * gam)
                    _chk("bend.kE_pre", kE << DLAM_PRE)
                    _chk("bend.dlam", dlam)
                if dlam == 0:
                    continue
                for wi, gx, gy, P, R in ((w0, g0x, g0y, P0, res[j]), (w1, g1x, g1y, P1, res[j + 1]),
                                         (w2, g2x, g2y, P2, res[j + 2])):
                    if wi == 0:
                        continue
                    wd = msa(wi, dlam, 32)                    # Q40 m^2
                    dax, day = msa(wd, gx, ds), msa(wd, gy, ds)   # Q32 m, in-plane
                    if CHECK:
                        _chk("bend.wd", wd)
                        _chk("bend.lift", dax * e1[0] + day * eX[0])
                    _move(P, R, sra(dax * e1[0] + day * eX[0], U_SHIFT),
                          sra(dax * e1[1] + day * eX[1], U_SHIFT),
                          sra(dax * e1[2] + day * eX[2], U_SHIFT))

    # -- 7d-f. anchors, roots, grabs ---------------------------------------------------------------
    def _project_extra(self) -> None:
        row = self.row
        for a in self.arms:
            if not any(a.anchored):
                continue
            for i in range(row.n_nodes):
                if not a.anchored[i]:
                    continue
                A, P = a.anchor_pos[i], a.p[i]
                dx, dy, dz = A[0] - P[0], A[1] - P[1], A[2] - P[2]
                dist = isqrt(dx * dx + dy * dy + dz * dz)
                force_q8 = 0
                if a.w[i]:
                    if CHECK:
                        _chk("anchor.force_num", (dist * row.inv_dt2) << 16)
                    force_q8 = ((dist * row.inv_dt2) << 16) // a.w[i]   # Q8 N, >= 0
                    if force_q8 > a.anchor_force[i]:
                        a.anchor_force[i] = force_q8
                if a.grip_q8 is not None:
                    if a.w[i] == 0:
                        pop = dist > 0
                    else:
                        pop = force_q8 > a.grip_q8
                    if pop:
                        a.anchored[i] = False
                        a.popped[i] = True
                        continue
                P[0], P[1], P[2] = A[0], A[1], A[2]
                a.res[i] = [0, 0, 0]
        for bi, b in enumerate(self.bodies):
            for a in self.arms:
                if a.body != bi:
                    continue
                for k in (0, 1):
                    self._attach(a, k, b, a.root_local[k])
        for gi, (ai, k, bi, local) in enumerate(self.grabs):
            J = self._attach(self.arms[ai], k, self.bodies[bi], local)
            if J is not None:
                f = _force_q8(J, row.inv_dt2)
                if f > self.grab_force[gi]:
                    self.grab_force[gi] = f

    def _attach(self, a: Arm, k: int, b: Body, L) -> None:
        """Rigid attachment of node k to the body point L (body frame, Q24):
        zero compliance, yaw-only rotation (arm_bank3d.Bank3D._attach). The
        shared IMPULSE J = C / W (Q30 kg m) is computed once: the node moves by
        -w_n J, the body by +w_b J and turns by (r x J)_z / I."""
        th16 = sra(b.th, CHAIN_SHIFT - 16)
        c, s = K.cos_q16(th16), K.sin_q16(th16)
        rx = sra(c * L[0] - s * L[1], 16)
        ry = sra(s * L[0] + c * L[1], 16)
        rz = L[2]
        P = a.p[k]
        Cx, Cy, Cz = P[0] - (b.x[0] + rx), P[1] - (b.x[1] + ry), P[2] - (b.x[2] + rz)
        cn = isqrt(Cx * Cx + Cy * Cy + Cz * Cz)
        if cn == 0:
            return None
        nx, ny = dra(Cx << U_SHIFT, cn), dra(Cy << U_SHIFT, cn)
        rxn = sra(rx * ny - ry * nx, U_SHIFT)                 # Q24 m
        wn = a.w[k]
        wrot = msa(msa(rxn, rxn, CHAIN_SHIFT), b.winvI_q32, CHAIN_SHIFT)   # Q32
        W = wn + b.w_q32 + wrot
        if CHECK:
            _chk("attach.C_sq", Cx * Cx + Cy * Cy + Cz * Cz)
            _chk("attach.W", W)
            _chk("attach.C_pre", max(abs(Cx), abs(Cy), abs(Cz)) << ATTACH_PRE)
        jx, jy, jz = dra(Cx << ATTACH_PRE, W), dra(Cy << ATTACH_PRE, W), dra(Cz << ATTACH_PRE, W)
        js = ATTACH_J_SHIFT                                 # Q32 x Q30 >> 30 -> Q32 fine
        if wn:
            _move(P, a.res[k], -msa(wn, jx, js), -msa(wn, jy, js), -msa(wn, jz, js))
        wb = b.w_q32
        _move(b.x, b.res, msa(wb, jx, js), msa(wb, jy, js), msa(wb, jz, js))
        tq = rx * jy - ry * jx                              # (r x J)_z, Q54 kg m^2
        if CHECK:
            _chk("attach.tq", tq)
        if tq:
            # Q54 x Q32 >> 54 -> Q32 rad: the yaw step, fine
            t = b.res_th + msa(tq, b.winvI_q32, CHAIN_SHIFT + ATTACH_J_SHIFT)
            i = sra(t, FINE_BITS)
            b.th += i
            b.res_th = t - (i << FINE_BITS)
        return jx, jy, jz

    # -- 7g. floor ----------------------------------------------------------------------------------
    def _project_floor(self, a: Arm, pp) -> None:
        mu = a.mu_q16
        for i in range(self.row.n_nodes):
            P = a.p[i]
            if P[2] >= 0 or a.pinned[i]:
                continue
            R = a.res[i]
            pen = -((P[2] << FINE_BITS) + R[2])                 # fine
            P[2] = 0
            R[2] = 0
            if mu > 0:
                _friction(P, R, pp[i], mu, pen)

    def _floor_bodies(self, b_prev) -> None:
        for bi, b in enumerate(self.bodies):
            if not b.floor or b.x[2] >= 0:
                continue
            pen = -((b.x[2] << FINE_BITS) + b.res[2])
            b.x[2] = 0
            b.res[2] = 0
            if b.mu_q16 > 0:
                _friction(b.x, b.res, b_prev[bi][0], b.mu_q16, pen)

    # -- 7h. static topology (#106 P6c) ----------------------------------------------------------------
    def _project_walls(self, a: Arm, pp) -> None:
        """Every unpinned node inside a blocked tile leaves through the face
        of least penetration whose neighbour is open (-x, +x, -y, +y on ties);
        position-based friction on the two tangential axes."""
        t = self.topo
        T = t.tile_q24
        ox, oy = t.origin
        mu = a.mu_q16
        for i in range(self.row.n_nodes):
            if a.pinned[i]:
                continue
            P = a.p[i]
            ix, iy = t.tile(P[0], 0), t.tile(P[1], 1)
            if not t.blocked(ix, iy):
                continue
            best_d, best_ax, best_face = 0, -1, 0
            for ax, nx, ny, face in ((0, ix - 1, iy, ox + ix * T - 1), (0, ix + 1, iy, ox + (ix + 1) * T),
                                     (1, ix, iy - 1, oy + iy * T - 1), (1, ix, iy + 1, oy + (iy + 1) * T)):
                if t.blocked(nx, ny):
                    continue
                d = P[ax] - face if face < P[ax] else face - P[ax]
                if best_ax < 0 or d < best_d:
                    best_d, best_ax, best_face = d, ax, face
            R = a.res[i]
            if best_ax < 0:
                P[0], P[1] = pp[i][0], pp[i][1]
                R[0] = R[1] = 0
                continue
            pen = (best_face << FINE_BITS) - ((P[best_ax] << FINE_BITS) + R[best_ax])
            if pen < 0:
                pen = -pen
            if CHECK:
                _chk("wall.pen", pen)
            P[best_ax] = best_face
            R[best_ax] = 0
            if mu > 0:
                _friction(P, R, pp[i], mu, pen, (1 - best_ax, 2))

    def _collide_bodies(self) -> None:
        """Each body's disc out of the blocked tiles under its bounding box
        (row-major), then out of the other units' footprints (in order)."""
        t = self.topo
        T = t.tile_q24
        ox, oy = t.origin
        for b in self.bodies:
            r = b.coll_r_q24
            if r <= 0:
                continue
            x0, x1 = t.tile(b.x[0] - r, 0), t.tile(b.x[0] + r, 0)
            y0, y1 = t.tile(b.x[1] - r, 1), t.tile(b.x[1] + r, 1)
            for iy in range(y0, y1 + 1):
                for ix in range(x0, x1 + 1):
                    if t.blocked(ix, iy) and _push_disc(b, r, ox + ix * T, oy + iy * T,
                                                        ox + (ix + 1) * T, oy + (iy + 1) * T):
                        b.contact |= CONTACT_WALL
            for tx0, ty0, tx1, ty1 in t.unit_rects:
                if tx1 <= tx0 or ty1 <= ty0:
                    continue
                if _push_disc(b, r, ox + tx0 * T, oy + ty0 * T, ox + tx1 * T, oy + ty1 * T):
                    b.contact |= CONTACT_UNIT

    # -- snapshots ------------------------------------------------------------------------------------
    def state_ints(self) -> list[int]:
        """Every synced integer of the world, in a fixed order."""
        out = [self.sub_count]
        for a in self.arms:
            for q in a.p:
                out += q
            for q in a.v:
                out += q
            out += [int(x) for x in a.anchored]
            for q in a.anchor_pos:
                out += q
        for b in self.bodies:
            out += b.x + b.v + [b.th, b.om]
        return out

    def state_bytes(self) -> bytes:
        v = self.state_ints()
        return struct.pack(f"<{len(v)}q", *v)

    def output_ints(self) -> list[int]:
        """This tick's outputs (#106 P6c), in a fixed order: per arm the
        contact bits, pops and anchor forces per node; per body its contact
        bits; per grab its force."""
        out = []
        for a in self.arms:
            out += a.contact
            out += [int(x) for x in a.popped]
            out += a.anchor_force
        for b in self.bodies:
            out.append(b.contact)
        out += self.grab_force
        return out


def _rebuild(v_pred: int, dp: int, res: int, disp: int, inv_dt: int, on_floor: bool,
             at_high_wall: bool = False) -> int:
    """The velocity after a substep, one component: the PBD rebuild
    ``inv_dt (p - p_prev)`` of the displacement INCLUDING the corrections' fine
    remainder ``res`` (``inv_dt dp + sra(inv_dt res, FINE_BITS)``) -- clamped
    at 0 from below on a floor contact (the float's ``v_z = max(v_z, 0)``) --
    PLUS the predict's rounding remainder ``v_pred - inv_dt disp``
    (|r| <= inv_dt / 2). Without the two remainders every velocity change
    below half a count of displacement per substep is erased (gravity rounds
    to whole counts per substep^2, the quadratic drag goes dead below
    ~1 cm/s, a compliant bend's and a heavy body's sub-count corrections
    vanish); with them the integer chain integrates what the float
    integrates, and with nothing acting ``v`` is returned unchanged.
    ``on_floor`` clamps the rebuilt part at 0 from BELOW (the floor, or a wall
    on the low side of this axis, #106 P6c), ``at_high_wall`` from ABOVE (a
    wall on the high side)."""
    pbd = inv_dt * dp + sra(inv_dt * res, FINE_BITS)
    if on_floor and pbd < 0:
        pbd = 0
    if at_high_wall and pbd > 0:
        pbd = 0
    return pbd + (v_pred - inv_dt * disp)


_NO_TOUCH = (False, False, False, False, False)


def _wall_touch(t: Topology, q) -> tuple:
    """(low-x, high-x, low-y, high-y, any) for a node at q: a side is TOUCHED
    when one count further toward it lies in a blocked tile (the node sits at
    that face); `any` also when the node itself is in a blocked tile."""
    ix, iy = t.tile(q[0], 0), t.tile(q[1], 1)
    inside = t.blocked(ix, iy)
    lx = t.blocked(t.tile(q[0] - 1, 0), iy)
    hx = t.blocked(t.tile(q[0] + 1, 0), iy)
    ly = t.blocked(ix, t.tile(q[1] - 1, 1))
    hy = t.blocked(ix, t.tile(q[1] + 1, 1))
    return (lx, hx, ly, hy, inside or lx or hx or ly or hy)


def _push_disc(b, r: int, X0: int, Y0: int, X1: int, Y1: int) -> bool:
    """Push a body's disc (radius r, Q24) out of the closed rectangle
    [X0, X1] x [Y0, Y1]; True when it overlapped. Fine, through `_move`."""
    cx, cy = b.x[0], b.x[1]
    qx = X0 if cx < X0 else (X1 if cx > X1 else cx)
    qy = Y0 if cy < Y0 else (Y1 if cy > Y1 else cy)
    dx, dy = cx - qx, cy - qy
    d2 = dx * dx + dy * dy
    if d2 >= r * r:
        return False
    if d2 > 0:
        d = isqrt(d2)
        push = r - d
        if CHECK:
            _chk("disc.push", (max(abs(dx), abs(dy)) * push) << FINE_BITS)
        fx, fy = dra((dx * push) << FINE_BITS, d), dra((dy * push) << FINE_BITS, d)
    else:
        best_d, best_s, best_ax = cx - X0, -1, 0
        for dd, sg, ax in ((X1 - cx, 1, 0), (cy - Y0, -1, 1), (Y1 - cy, 1, 1)):
            if dd < best_d:
                best_d, best_s, best_ax = dd, sg, ax
        mag = best_s * ((best_d + r) << FINE_BITS)
        fx, fy = (mag, 0) if best_ax == 0 else (0, mag)
    _move(b.x, b.res, fx, fy, 0)
    return True


def _force_q8(J, inv_dt2: int) -> int:
    """|J| / dt^2 in Q8 N for an attachment impulse J (Q30 kg m): each
    component narrowed to Q8 and saturated at +-FORCE_SAT_Q8, then the floor
    isqrt of the sum of squares."""
    s = 0
    for j in J:
        if CHECK:
            _chk("grab.force_c", j * inv_dt2)
        f = sra(j * inv_dt2, ATTACH_J_SHIFT - 8)
        if f > FORCE_SAT_Q8:
            f = FORCE_SAT_Q8
        elif f < -FORCE_SAT_Q8:
            f = -FORCE_SAT_Q8
        s += f * f
    return isqrt(s)


def _move(P, R, fx: int, fy: int, fz: int) -> None:
    """Apply a FINE (Q32) correction to a node: the remainder plus the
    correction, rounded half away to whole counts, moves the position; what
    is left (|.| <= half a count) stays in the remainder."""
    t = R[0] + fx
    i = sra(t, FINE_BITS)
    P[0] += i
    R[0] = t - (i << FINE_BITS)
    t = R[1] + fy
    i = sra(t, FINE_BITS)
    P[1] += i
    R[1] = t - (i << FINE_BITS)
    t = R[2] + fz
    i = sra(t, FINE_BITS)
    P[2] += i
    R[2] = t - (i << FINE_BITS)


def _ip(d, e) -> int:
    """The in-plane coordinate of a Q24 vector along a Q30 unit vector: Q24."""
    return sra(d[0] * e[0] + d[1] * e[1] + d[2] * e[2], U_SHIFT)


def _reject(v, t) -> tuple:
    """v - (v . t) t for a Q30 unit t (Gram-Schmidt), in v's currency."""
    s = sra(v[0] * t[0] + v[1] * t[1] + v[2] * t[2], U_SHIFT)
    return (v[0] - sra(s * t[0], U_SHIFT), v[1] - sra(s * t[1], U_SHIFT),
            v[2] - sra(s * t[2], U_SHIFT))


def _cross_q30(a, b) -> tuple:
    return (sra(a[1] * b[2] - a[2] * b[1], U_SHIFT), sra(a[2] * b[0] - a[0] * b[2], U_SHIFT),
            sra(a[0] * b[1] - a[1] * b[0], U_SHIFT))


def _friction(P, R, Pp, mu_q16: int, pen_f: int, ax=(0, 1)) -> None:
    """Coulomb friction on the substep's tangential displacement (position
    based): |dx_t| <- max(0, |dx_t| - mu pen), the FINE displacement (whole
    counts + remainder, Q32) scaled by keep in [0, 1] Q30. pen_f is fine.
    ``ax`` are the two tangential axes: (x, y) on the floor; (the other
    horizontal, z) on a wall (#106 P6c)."""
    a0, a1 = ax
    dx = ((P[a0] - Pp[a0]) << FINE_BITS) + R[a0]
    dy = ((P[a1] - Pp[a1]) << FINE_BITS) + R[a1]
    if CHECK:
        _chk("friction.s_sq", dx * dx + dy * dy)
        _chk("friction.mu_pen", (mu_q16 * pen_f) << 14)
    s = isqrt(dx * dx + dy * dy)
    if s == 0:
        return
    ratio = dra((mu_q16 * pen_f) << 14, s)          # mu pen / s, Q30
    keep = U_ONE - ratio
    if keep >= U_ONE:
        return
    if keep < 0:
        keep = 0
    for c, d in ((a0, dx), (a1, dy)):
        t = msa(d, keep, U_SHIFT) if keep else 0
        i = sra(t, FINE_BITS)
        P[c] = Pp[c] + i
        R[c] = t - (i << FINE_BITS)


# ---------------------------------------------------------------------------
# Dequantized views (the gates' measurements read only these)
# ---------------------------------------------------------------------------
def positions_m(world: World, arms=None):
    """(A, n, 3) float64 metres -- a MEASUREMENT, never written back."""
    import numpy as np
    idx = range(len(world.arms)) if arms is None else arms
    return np.array([[q for q in world.arms[i].p] for i in idx], dtype=np.float64) / CF.CHAIN_ONE


def velocities_m(world: World, arms=None):
    import numpy as np
    idx = range(len(world.arms)) if arms is None else arms
    return np.array([[q for q in world.arms[i].v] for i in idx], dtype=np.float64) / CF.CHAIN_ONE


# ---------------------------------------------------------------------------
# Replay: drive the reference with a P5z trace's recorded solver inputs
# ---------------------------------------------------------------------------
def load_traces(path=None) -> dict:
    """prototypes/octopus_motion/out/traces_p5z.npz -> {scenario: {field: array}}
    (regenerate with `prototypes/octopus_motion/gates3d.py`)."""
    import numpy as np
    path = _ROOT / "prototypes" / "octopus_motion" / "out" / "traces_p5z.npz" if path is None else Path(path)
    z = np.load(path, allow_pickle=False)
    out: dict = {}
    for k in z.files:
        if "/" in k:
            name, f = k.split("/", 1)
            out.setdefault(name, {})[f] = z[k]
    return out


def check_trace_params(row: ChainRow, cdef, tr: dict) -> list[str]:
    """The trace's solver params against the row the reference runs: a
    mismatch means the float and the int ran different animals."""
    import math
    want = dict(n_sub=row.n_sub, n_pass=row.n_pass, k_bend=cdef.tissue.k_bend,
                k_passive_frac=cdef.tissue.k_passive_frac, k_dist=cdef.tissue.k_dist,
                bend_damp_s=cdef.tissue.bend_damp_s, c_per=cdef.tissue.c_drag_perp,
                c_tan=cdef.tissue.c_drag_tan, g_eff=cdef.tissue.g_eff,
                arm_length_m=cdef.body.arm_length_m, r_base_m=cdef.body.radius_base_m,
                r_tip_m=cdef.body.radius_tip_m, rho_arm=cdef.body.tissue_density,
                rho_water=cdef.body.water_density, dt_tick=1.0 / (row.inv_dt // row.n_sub))
    bad = []
    for k, v in want.items():
        got = float(tr[k])
        if not math.isclose(got, float(v), rel_tol=1e-5, abs_tol=1e-12):
            bad.append(f"{k}: trace {got!r} vs row {v!r}")
    return bad


def replay(row: ChainRow, tr: dict, ticks: int | None = None, record_every: int = 1,
           on_tick=None, topo: Topology | None = None) -> dict:
    """Run the reference from a P5z trace: its initial state, its solver
    params (checked by `check_trace_params` beforehand), and -- for the bank
    scenarios -- its per-tick solver INPUTS (activation, rest scale, bend
    targets, anchors, root offsets, grip, grabs). A node newly anchored in the
    trace is anchored where the INT node is (the hub's `anchor` semantics:
    anchor_pos = the node's own position), so the int arm is never yanked to
    the float's position. The hubs' body-velocity writes (jets) are not
    replayed: the gated scenarios (reach, dead arm, crawl, free animal) have
    none. Returns {"P": (ticks, A, n, 3) m, "BX": (ticks, B, 3), "BTH": (ticks, B)}."""
    import numpy as np
    w = build_replay_world(row, tr, topo)
    bank = "in_activation" in tr
    n_ticks = tr["in_activation"].shape[0] if bank else tr["p"].shape[0]
    n_ticks = n_ticks if ticks is None else min(ticks, n_ticks)
    P, BX, BTH = [], [], []
    prev_anch = None
    for t in range(n_ticks):
        if bank:
            prev_anch = apply_trace_inputs(w, tr, t, prev_anch)
        w.tick()
        if on_tick is not None:
            on_tick(w, t)
        if t % record_every == 0 or t == n_ticks - 1:
            P.append(positions_m(w))
            BX.append([[c / CF.CHAIN_ONE for c in b.x] for b in w.bodies])
            BTH.append([b.th / CF.CHAIN_ONE for b in w.bodies])
    return dict(P=np.array(P), BX=np.array(BX), BTH=np.array(BTH), world=w)


def build_replay_world(row: ChainRow, tr: dict, topo: Topology | None = None) -> World:
    """The World `replay` runs: a P5z trace's initial state and solver params
    (its per-tick inputs are `apply_trace_inputs`)."""
    w = World(row, topo=topo)
    if "in_activation" in tr:
        for bi in range(tr["body_x0"].shape[0]):
            w.add_body(tr["body_x0"][bi], float(tr["body_th0"][bi]), mass=float(tr["body_mass"][bi]),
                       radius=float(tr["body_radius"][bi]), c_drag=float(tr["body_c_drag"][bi]),
                       area=float(tr["body_area"][bi]), rot_damp=float(tr["body_rot_damp"][bi]),
                       mu=float(tr["body_mu"][bi]), inertia=float(tr["body_inertia"][bi]),
                       drag_mode=int(tr["body_drag_mode"][bi]), tau=float(tr["body_tau"][bi]))
        for a in range(tr["p0"].shape[0]):
            w.add_arm(tr["p0"][a], mu=float(tr["mu"][a]), drag_mode=int(tr["drag_mode"][a]),
                      tau=float(tr["drag_tau"][a]), vref=float(tr["drag_vref"][a]),
                      root_ref=tuple(tr["root_ref"]), body=0, root_local=tr["in_root_local"][0][a],
                      grip=float(tr["in_grip"][0][a]))
    else:
        w.add_arm(tr["p0"], pinned=tuple(int(k) for k in tr["pinned"]), mu=float(tr["mu"]),
                  root_ref=tuple(tr["root_ref"]))
        if "wave_a" in tr:
            w.set_wave(0, float(tr["wave_a"]), float(tr["wave_tact"]), float(tr["wave_t0"]),
                       float(tr["wave_w"]), bool(tr["wave_hold"]))
        w.floor_on = bool(tr["floor"])
    return w


def apply_trace_inputs(w: World, tr: dict, t: int, prev_anch):
    """Write a bank trace's tick-t solver INPUTS into the World (what the hubs
    wrote before the step); a newly anchored node is anchored where the INT
    node is. Returns this tick's anchor mask (pass it back as `prev_anch`)."""
    row = w.row
    anch = tr["in_anchored"][t]
    for a in range(len(w.arms)):
        arm = w.arms[a]
        w.set_activation(a, tr["in_activation"][t][a])
        w.set_rest_scale(a, tr["in_rest_scale"][t][a])
        w.set_bend_target(a, tr["in_bend_target"][t][a])
        arm.root_local = [_q24v(q) for q in tr["in_root_local"][t][a]]
        w.set_grip(arm, float(tr["in_grip"][t][a]))
        for i in range(row.n_nodes):
            was = arm.anchored[i] if prev_anch is not None else False
            if anch[a][i] and not was:
                arm.anchored[i] = True
                arm.anchor_pos[i] = list(arm.p[i])
            elif not anch[a][i]:
                arm.anchored[i] = False
    gb, gl = tr["in_grab_body"][t], tr["in_grab_local"][t]
    w.grabs = [(a, k, int(gb[a][k]), _q24v(gl[a][k]))
               for a in range(len(w.arms)) for k in range(row.n_nodes) if gb[a][k] >= 0]
    return anch


__all__ = [
    "CHAIN_SHIFT", "U_SHIFT", "U_ONE", "DIST_J_SHIFT", "ATTACH_PRE", "KW_PRE_SHIFT", "G_SHIFT", "G_PRE",
    "DLAM_PRE", "DLAM_SHIFT", "D_MAX_Q16", "OPC_FLOOR", "N0_MIN", "ALT_X_LIMIT", "PI_Q24",
    "TWO_PI_Q24", "DRAG_QUADRATIC", "DRAG_LINEAR", "DRAG_UNIFORM", "WAVE_CLOCK_SPAN",
    "CONTACT_FLOOR", "CONTACT_WALL", "CONTACT_UNIT", "FORCE_SAT_Q8",
    "CHECK", "BITS", "reset_bits", "bits_report", "load_activation_table", "activation_s_q16",
    "ChainRow", "chain_row", "Topology", "topology", "Wave", "Arm", "Body", "World",
    "positions_m", "velocities_m",
    "load_traces", "check_trace_params", "replay", "build_replay_world", "apply_trace_inputs",
]
