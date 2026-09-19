"""THE INTEGER REFERENCE for the ray-engine-v2 sweep (P0, 2026-09-15).

An exact-integer transcription of `docs/ray_engine_v2_design_v3_2026-09-15.md`
section 2.3 in GATHER form, plus the excess-form Fleck factor (section 2.8), the
E-table and its inverse (section 2.6), the virtual ambient ring (section 2.7) and
a model of the temperature solver's Pass-1 fold for SOLIDS so that multi-tick
scenarios (cooling curves, equilibria, counter scenes) can be run.

Nothing here is engine code: this is the measuring instrument the design owes
before anything is built, and it is what P1's C++ must reproduce bit for bit.

WHAT TRANSCRIBES WHAT
  bake_e_table       <- cpp/src/raycaster.cpp:62-97 (the exact int64 chain)
  e_bucket_of        <- cpp/src/raycaster.h:203-213
  e_inv_q            <- design v3 section 2.6 ("E°⁻¹"), both edge cases of row 31
  shr_round0         <- cpp/src/fixed_point.h:410 (the int64 twin P1 owes)
  floordiv_q         <- cpp/src/fixed_point.h:562
  fleck_f_solid_q    <- design v3 section 2.8 (the excess form, row 22; Q24, row 32)
  sweep_q            <- design v3 section 2.3 (gather form; body re-emission, row 25)
  fold_pass1_solid   <- cpp/src/temperature_solver.cpp:247-299 + the clamp of
                        section 2.8 in its corrected form (row 21)

ARITHMETIC. Pure Python ints throughout, so no overflow is possible and every
headroom question is *measured* (`SweepResult.max_*`) rather than assumed. The
engine's int64 is wide enough exactly when those measurements stay below 2^63;
gate 11 asserts the design's own 2^46 bound on the per-cell sums.

The one place a width does bite is `temperature`, which is int32 Q16.16 in the
engine (max 32767.99 game): `fold_pass1_solid(int32_sat=True)`, the default,
models `sat_add_q16`. Pass `int32_sat=False` only to reproduce a float study's
unbounded number.

Erik's ruling of 2026-09-15 (design row 25) is implemented as `body_mode="reemit"`:
a body re-emits at the ambient level. `body_mode="sink"` is the pre-ruling variant,
kept ONLY so the gate that says the ruling was necessary cannot pass vacuously.

THE FLECK FACTOR IS Q24 (design row 32, P0b). `f_q` is `floordiv_q(T_abs << 24, D)`
and the damped source is `amb_m + ((ex_m * f_q24) >> 24)`; in Q16 that source was
not monotone in T above the fire range (P0 section 0.4, gate 12).

THE ALPHA FLOOR IS 0 (design row 39, RULED by Erik 2026-09-16; P2a). `alpha =
max(0, 1 - 1/g)`, so `D = max(T_abs, 4L)` and `f == 1` exactly wherever the
explicit material update is provably monotone (`g <= 1`: 1068 game for wood,
1424 for the a = 0.5 rows). Below that branch the reference IS forward Euler, to
the last count; above ~1800 game the two floors are bit-identical. Two knobs
exist so the gates can MEASURE the forms this file does not ship:
`fleck_f_q(shift=16)` is the rejected Q16 form (gate 12's non-vacuity), and
`alpha_floor="half"` is the superseded floor of one half (Fleck's own IMC bound),
which gates 8 and 10 measure beside the default so "equals explicit" and "within
one count" cannot pass vacuously (`p0b_alpha_floor.py` is P0b's measurement of
both, and it names its floors explicitly, so it is unaffected by the default).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Dials. The shipped values; `config_dials_match()` checks them against
# config.toml so this file cannot silently drift from the engine.
# --------------------------------------------------------------------------- #
ONE = 65536                      # Q16.16 unit
F_SHIFT = 24                     # the Fleck factor's fixed point (design row 32)
F_ONE = 1 << F_SHIFT             # f == F_ONE exactly when L_q == 0
ALPHA_FLOOR_HALF = "half"        # alpha = max(1/2, 1 - 1/g) -- Fleck's IMC bound
ALPHA_FLOOR_ZERO = "zero"        # alpha = max(0,   1 - 1/g) -- RULED (row 39), the default
ALPHA_FLOOR_DEFAULT = ALPHA_FLOOR_ZERO   # Erik, 2026-09-16: "Let's go floor 0 then."
RAD_SCALE = 5.1427e-5            # config.toml:526   [physics.fire] rad_scale
K_AMB = 293                      # config.toml:813   kelvin_ambient (integer-valued)
K_SLOPE = 1                      # config.toml:814   k_temp_to_kelvin (G12: the x1 map)
E_TABLE_SIZE = 4000              # raycaster.h:203   T_game in [0, 16000)
E_BUCKET_SHIFT = 2               # raycaster.h:204   4 game units per bucket
E_INDEX_SHIFT = 16 + E_BUCKET_SHIFT
T_MAX_PHYS_Q = 16000 << 16       # config.toml:186   the counted physical rail
T_TABLE_TOP_GAME = 4 * (E_TABLE_SIZE - 1)   # 15996 — where E°⁻¹ saturates
TICK_HZ = 24.0                   # the sim clock (for the analytic cooling reference)
INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)


def config_dials_match(config_path=None):
    """Return (ok, detail). Parses config.toml and compares the four dials above.

    A drift here silently invalidates every number this file prints, so the gate
    runner calls it and reports the result.
    """
    import pathlib
    import tomllib
    if config_path is None:
        config_path = pathlib.Path(__file__).resolve().parents[2] / "config.toml"
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)
    got = {
        "rad_scale": cfg["physics"]["fire"]["rad_scale"],
        "kelvin_ambient": cfg["physics"]["temperature_scale"]["kelvin_ambient"],
        "k_temp_to_kelvin": cfg["physics"]["temperature_scale"]["k_temp_to_kelvin"],
        "T_MAX_PHYS": cfg["physics"]["thermal"]["T_MAX_PHYS"],
    }
    want = {
        "rad_scale": RAD_SCALE,
        "kelvin_ambient": float(K_AMB),
        "k_temp_to_kelvin": float(K_SLOPE),
        "T_MAX_PHYS": float(T_MAX_PHYS_Q >> 16),
    }
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    return (not bad), (bad if bad else got)


def shipped_absorbing_rows(config_path=None):
    """Every SHIPPED material row that absorbs radiation, read from config.toml.

    Returns [(name, a_q, his, heat_atten, thermal_mass), ...] for the rows with
    `heat_atten > 0`; rows with `heat_atten == 0` (air, foliage) never emit or
    absorb and are skipped. `his = log2(thermal_mass)` is the engine's own
    `heat_inv_shift` (src/simulation/materials.py:343-357), so `thermal_mass`
    must be a power of two -- the same contract, asserted here.

    Read rather than hardcoded on purpose: gate 12's property is about the rows
    the game SHIPS, so a new material row must be able to fail it.
    """
    import pathlib
    import tomllib
    if config_path is None:
        config_path = pathlib.Path(__file__).resolve().parents[2] / "config.toml"
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)
    out = []
    for name, row in cfg["materials"].items():
        atten = float(row.get("heat_atten", 0.0))
        if atten <= 0.0:
            continue
        tm = int(round(float(row.get("thermal_mass", 0))))
        if tm <= 0 or (tm & (tm - 1)) != 0:
            raise ValueError(
                f"materials.{name}: heat_atten = {atten} with thermal_mass = {tm}; "
                f"an absorbing material must carry a power-of-two thermal mass "
                f"(it sits on the heat->temperature divide)")
        out.append((name, quant(atten), tm.bit_length() - 1, atten, tm))
    return out


# --------------------------------------------------------------------------- #
# The kit: the exact integer primitives, one transcription each.
# --------------------------------------------------------------------------- #
def shr_round0(x: int, s: int) -> int:
    """fixed_point.h:410, symmetric round-toward-zero shift (the int64 twin)."""
    return -((-x) >> s) if x < 0 else (x >> s)


def floordiv_q(n: int, d: int) -> int:
    """fixed_point.h:562. Python's // already floors; the C++ correction branch
    is unreachable for the Fleck operands (both strictly positive)."""
    return n // d


def quant(x: float) -> int:
    """The Q16 ingress door (round-half-away-from-zero), for scene setup only."""
    s = x * ONE
    return int(s + 0.5) if s >= 0 else int(s - 0.5)


def quant_f(x: float) -> int:
    """The same door in the Fleck factor's Q24, for a gate that FORCES an f."""
    s = x * F_ONE
    return int(s + 0.5) if s >= 0 else int(s - 0.5)


def sat_add_q16(a: int, b: int) -> int:
    """fixed_point.h:400, the signed saturating add the radiation fold uses."""
    v = a + b
    if v > INT32_MAX:
        return INT32_MAX
    if v < INT32_MIN:
        return INT32_MIN
    return v


# --------------------------------------------------------------------------- #
# The emissive table and its inverse (design section 2.6).
# --------------------------------------------------------------------------- #
def bake_e_table(rad_scale: float = RAD_SCALE, kelvin_ambient: int = K_AMB,
                 slope: int = K_SLOPE) -> list:
    """raycaster.cpp:62-97 verbatim: bucket midpoints, K^4 by repeated int64
    multiplication, ONE double boundary (`(double)k4 * scale + 0.5`).

    The integer-bake precondition (both Kelvin dials whole) is a hard invariant
    in the engine; it is asserted here for the same reason.
    """
    assert float(kelvin_ambient).is_integer() and float(slope).is_integer(), (
        "the exact int64 bake requires integer-valued Kelvin dials")
    amb_i, slope_i = int(kelvin_ambient), int(slope)
    tbl = []
    for t in range(E_TABLE_SIZE):
        T_mid = 4 * t + 2
        K = amb_i + slope_i * T_mid
        k2 = K * K
        k4 = k2 * k2
        v = float(k4) * rad_scale
        tbl.append(int(v + 0.5) if v > 0.0 else 0)
    return tbl


E = bake_e_table()
E0 = E[0]


def e_bucket_of(T_q: int) -> int:
    """raycaster.h:206-213. Sub-ambient indexes bucket 0 (the ambient floor)."""
    if T_q <= 0:
        return 0
    b = T_q >> E_INDEX_SHIFT
    return E_TABLE_SIZE - 1 if b >= E_TABLE_SIZE else b


def e_of_T(T_q: int, table=E) -> int:
    return table[e_bucket_of(T_q)]


def e_inv_q(phi: int, table=E) -> int:
    """E°⁻¹(Φ): the largest bucket b with E°[b] <= Φ, mapped to the bucket's LOW
    edge, returned as a Q16.16 game temperature (design section 2.6).

    A FIXED 12-TRIP binary lifting (2048, 1024, ... 1), in the `sqrt_q16_dev`
    idiom: one branch-uniform function for host and device, no data-dependent
    trip count. 2^12 = 4096 >= E_TABLE_SIZE = 4000.

    Both edge cases of design row 31:
      * Phi < E°[0]  -> 0 (no radiative warming above the ambient floor; such a
        cell's rad_net is <= 0 anyway).
      * Phi >= E°[3999] -> 15996 game, the last bucket's low edge, which is BELOW
        T_MAX_PHYS = 16000: on the radiative sub-step the clamp saturates first
        and the T_MAX_PHYS rail is unreachable.

    Idempotent by construction: e_bucket_of(4b << 16) == b, so E°[E°⁻¹(Φ)] <= Φ.
    """
    if phi < table[0]:
        return 0
    lo = 0
    span = 1 << 11
    while span:
        nxt = lo + span
        if nxt < E_TABLE_SIZE and table[nxt] <= phi:
            lo = nxt
        span >>= 1
    return (4 * lo) << 16


# --------------------------------------------------------------------------- #
# The Fleck factor, excess form (design section 2.8, row 22).
# --------------------------------------------------------------------------- #
def fleck_L_solid_q(T_q: int, a_q: int, his: int, table=E, e_ref: int = None) -> int:
    """The solid branch of L_q: the cell's FREE excess-emission loss this tick,
    in Q16.16 temperature.  L_q = shr_round0((a*(E°[T] - E°[0])) >> 16, his)."""
    ref = table[0] if e_ref is None else e_ref
    ex = table[e_bucket_of(T_q)] - ref
    if ex < 0:
        ex = 0
    return shr_round0((a_q * ex) >> 16, his)


def fleck_D_q(T_abs_q: int, L_q: int, alpha_floor: str = ALPHA_FLOOR_DEFAULT) -> int:
    """The Fleck denominator `T_abs * (1 + alpha*g)`, exactly, in shifts alone.

    Substituting g = 4L/T_abs, both floors collapse to one max() of two shifts:

        alpha = max(0,   1 - 1/g)  ->  D = max(T_abs,      4L)   (RULED row 39, DEFAULT)
        alpha = max(1/2, 1 - 1/g)  ->  D = max(T_abs + 2L, 4L)   (superseded; measured)

    Below the branch point alpha is its floor (1, or 1 + g/2); above it alpha is
    1 - 1/g and 1 + alpha*g == g on both, which is the shared `4L` arm. On the
    ruled floor the first arm is `T_abs` itself, so `f == 1` exactly whenever
    `4L <= T_abs` -- the whole explicitly-stable range is undamped (row 39).
    """
    if alpha_floor == ALPHA_FLOOR_HALF:
        return max(T_abs_q + 2 * L_q, 4 * L_q)
    if alpha_floor == ALPHA_FLOOR_ZERO:
        return max(T_abs_q, 4 * L_q)
    raise ValueError(f"unknown alpha_floor {alpha_floor!r} (want 'half' or 'zero')")


def fleck_f_q(T_q: int, L_q: int, *, shift: int = F_SHIFT,
              alpha_floor: str = ALPHA_FLOOR_DEFAULT) -> int:
    """f_q = floordiv_q((T_abs_q << 24), max(T_abs_q, 4L)), design 2.8 + rows 32, 39.

    `alpha = max(0, 1 - 1/g)` is never computed: 1 + alpha*g == max(1, g).
    T_abs_q > 0 always (T_MIN = -292 game keeps T_abs >= 1), D >= T_abs_q, so
    0 < f_q <= (1 << shift) and, on the RULED floor, f_q == (1 << shift) exactly
    when 4*L_q <= T_abs_q (on the superseded floor of one half, only when L_q == 0).

    `shift` is Q24 (row 32) and exists as a knob ONLY so the gates can measure the
    Q16 form the design rejected; `alpha_floor` likewise exists only so they can
    measure the superseded floor of one half. Neither is a scheme choice.
    """
    T_abs_q = T_q + (K_AMB << 16)
    assert T_abs_q > 0, "T_abs must be positive (T_MIN = -292 game guarantees it)"
    D = fleck_D_q(T_abs_q, L_q, alpha_floor)
    return floordiv_q(T_abs_q << shift, D)


def fleck_f_solid_q(T_q: int, a_q: int, his: int, table=E, e_ref: int = None, *,
                    shift: int = F_SHIFT, alpha_floor: str = ALPHA_FLOOR_DEFAULT):
    """Convenience: returns (f_q, L_q) for a solid cell."""
    L = fleck_L_solid_q(T_q, a_q, his, table, e_ref)
    return fleck_f_q(T_q, L, shift=shift, alpha_floor=alpha_floor), L


def fleck_f_float(T_game: float, L_game: float,
                  alpha_floor: str = ALPHA_FLOOR_DEFAULT) -> float:
    """The float form this is checked against: 1/(1 + alpha*g),
    g = 4*L/T_abs, alpha = max(floor, 1 - 1/g)."""
    T_abs = T_game + K_AMB
    g = 4.0 * L_game / T_abs
    if g <= 0.0:
        return 1.0
    if alpha_floor not in (ALPHA_FLOOR_HALF, ALPHA_FLOOR_ZERO):
        raise ValueError(f"unknown alpha_floor {alpha_floor!r}")
    floor = 0.5 if alpha_floor == ALPHA_FLOOR_HALF else 0.0
    alpha = max(floor, 1.0 - 1.0 / g)
    return 1.0 / (1.0 + alpha * g)


def damped_source_q(T_q: int, a_q: int, his: int, table=E, e_ref: int = None, *,
                    shift: int = F_SHIFT, alpha_floor: str = ALPHA_FLOOR_DEFAULT) -> int:
    """What a solid cell EMITS per ordinate-weight-1 this tick, exactly as the
    sweep forms it:  E_ref + ((E°[T] - E_ref) * f_q >> shift).

    This is the quantity design row 32 requires to be monotone in T (gate 12) and
    the quantity section 12 item 4 compares against the black body.
    """
    ref = table[0] if e_ref is None else e_ref
    ex = table[e_bucket_of(T_q)] - ref
    if ex < 0:
        ex = 0
    f_q = fleck_f_solid_q(T_q, a_q, his, table, e_ref,
                          shift=shift, alpha_floor=alpha_floor)[0]
    return ref + ((ex * f_q) >> shift)


# --------------------------------------------------------------------------- #
# Ordinates (design section 2.5) and their per-ordinate constants.
# --------------------------------------------------------------------------- #
def ordinates(n: int = 16, half_offset: bool = True):
    off = 0.5 if half_offset else 0.0
    return [((math.cos((m + off) * (2 * math.pi / n))),
             (math.sin((m + off) * (2 * math.pi / n)))) for m in range(n)]


def ordinate_constants(mu: float, eta: float, transport: str):
    """Returns (x_major, s_m, sx, sy). s_m is the MAJOR share of the split,
    quantized once at load (door 2); the minor share is the remainder."""
    am, ae = abs(mu), abs(eta)
    sx = 1 if mu > 0 else -1
    sy = 1 if eta > 0 else -1
    if transport == "shear":
        x_major = am >= ae
        f_m = quant((ae / am) if x_major else (am / ae))
        return x_major, ONE - f_m, sx, sy
    if transport == "step":
        return True, quant(am / (am + ae)), sx, sy
    raise ValueError(f"unknown transport {transport!r} (want 'shear' or 'step')")


# --------------------------------------------------------------------------- #
# The sweep (design section 2.3), gather form.
# --------------------------------------------------------------------------- #
@dataclass
class SweepResult:
    rad_net: list
    rad_flux: list
    rad_amb: list
    rad_fluence: list
    max_stream: int = 0           # the largest per-ordinate stream anywhere
    max_abs_net: int = 0          # max |rad_net[i]| (a per-cell 16-ordinate sum)
    max_fluence: int = 0          # max rad_fluence[i]
    max_product: int = 0          # the largest intermediate product (headroom)
    min_stream: int = 0           # positivity: must never be negative
    max_fleck_product: int = 0    # max (ex_m * f_q24), the widest Q24 product (G11)

    def sums(self):
        return (plane_sum(self.rad_net), plane_sum(self.rad_flux),
                plane_sum(self.rad_amb))

    def identity(self):
        return sum(self.sums())


def plane(h, w, v=0):
    return [[v] * w for _ in range(h)]


def plane_sum(p):
    return sum(sum(row) for row in p)


def plane_max_abs(p):
    return max(abs(v) for row in p for v in row)


def ambient_plane(e_ref, h, w, table=E):
    """Normalise the AMBIENT LEVEL argument to a per-cell plane (thermal v2 R3).

    The ambient the sweep radiates at is an EMISSIVE LEVEL, in the E° table's own
    units -- NOT a temperature, and this is forced, not a taste. The table's
    domain starts AT ambient (`e_bucket_of` returns bucket 0 for every T_q <= 0),
    so E°[T] >= E°[0] for every temperature there is: no temperature can
    express an ambient BELOW E°[0], and "0 K outside the hull" -- the thing
    R3 exists to make expressible -- is exactly such an ambient.

    `e_ref` accepts
      * None    -> E°[0] everywhere: the interior ambient, and the only value
                   the scalar era could express (thermal v2 R4 ships it
                   everywhere for now).
      * an int  -> that level everywhere. The scalar door, kept because a
                   uniform ambient is one number and a plane for it is noise;
                   0 is the scheme study's zero-sky configuration.
      * a plane -> used as given. In the engine this is DERIVED per cell from
                   vacuum/interior state (thermal v2 R3), never authored.
    There is ONE code path below this: the scalar is broadcast HERE, at the
    door, and the sweep reads a plane and nothing else.
    """
    if e_ref is None:
        e_ref = table[0]
    if isinstance(e_ref, int):
        return plane(h, w, e_ref)
    if len(e_ref) != h or len(e_ref[0]) != w:
        raise ValueError(f"ambient plane is {len(e_ref)}x{len(e_ref[0])}, "
                         f"scene is {h}x{w}")
    return e_ref


def validate_planes(a, d, k, amb=None, table=E):
    """The ingress invariants of design section 2.3, as raises (gate 3).

    The engine's site is src/simulation/materials.py; here they are the
    reference's contract, so a scene that violates one cannot be measured.

    `amb` (thermal v2 R3) is the per-cell ambient LEVEL plane, whose invariant
    is `0 <= amb[i] <= E°[0]`. The upper bound is what makes the per-cell
    excess `E°[T_i] - amb[i]` non-negative for EVERY temperature, since
    `e_bucket_of` floors at bucket 0. The excess form of the Fleck factor
    (design 2.8, row 22) and the sweep's positivity both rest on that
    non-negativity, so a hotter-than-room ambient is REFUSED here rather than
    clamped somewhere downstream -- a deck above at furnace temperature is
    design 7.4 territory and owes its own argument.
    """
    h, w = len(a), len(a[0])
    e0 = table[0]
    for y in range(h):
        for x in range(w):
            ai, di, ki = a[y][x], d[y][x], k[y][x]
            if ai < 0:
                raise ValueError(f"a < 0 at ({y},{x}): {ai}")
            if ai > di:
                raise ValueError(f"a > d at ({y},{x}): a={ai} d={di}")
            if di > ONE:
                raise ValueError(f"d > ONE at ({y},{x}): {di}")
            if ki < 0 or ki > ONE:
                raise ValueError(f"k outside [0, ONE] at ({y},{x}): {ki}")
            if amb is not None:
                mi = amb[y][x]
                if mi < 0 or mi > e0:
                    raise ValueError(
                        f"ambient level outside [0, E°[0]={e0}] at "
                        f"({y},{x}): {mi}")


def sweep_q(a, d, k, T, *, n_ord: int = 16, transport: str = "shear",
            f_plane=None, w_m: int = None, e_ref: int = None,
            body_mode: str = "reemit", table=E, validate: bool = True,
            half_offset: bool = True) -> SweepResult:
    """One tick of the sweep, exactly as design section 2.3 writes it.

    a, d, k : Q16 planes (material extinction, stamped extinction, leak)
    T       : Q16.16 temperature plane
    f_plane : the Fleck pre-pass output (Q24, row 32); None means f == F_ONE
    w_m     : the ordinate weight; default ONE // n_ord (4096 at S16)
    e_ref   : the ambient LEVEL -- a scalar (broadcast at the door) or a
              per-cell plane (thermal v2 R3). None is E°[0], the design's
              original single value; 0 reproduces the float scheme study's
              zero-sky configuration (the virtual ring then returns 0 and the
              whole emission is excess). See ambient_plane() for the
              denomination argument and validate_planes() for the invariant.

    WHICH CELL'S AMBIENT, AT EACH OF THE FOUR SITES -- the pin design v3 2.7
    owed and critique L2-B1 point 4 asked for. All four are cell i's OWN
    ambient, and the reason is the same every time: the per-cell fixed point of
    design 2.3 ("at exact ambient a body disturbs nothing") survives per-cell
    ambients only if every ambient-derived term at cell i is the SAME integer.
      * the cell's own emission floor, `src = amb_m_i + f*ex_m`: its excess is
        measured over ITS OWN ambient, so the two halves sum back to
        E°[T_i]*w_m. `ex_cell` is therefore per-cell-ref, and so is the Fleck
        pre-pass's L (fleck_prepass takes the same argument).
      * `ret`, the ceiling's ambient return: the ceiling above cell i is cell
        i's own out-of-plane boundary -- the same boundary `leaked` left
        through. At `i_in == amb_m_i`, `leaked == ret` as the same integer, so
        a cell at its own ambient stays an exact fixed point.
      * `emit_body`, the body standing on cell i: a grey body at the ambient of
        the cell it stands in (design row 25), so `abs_body == emit_body` by
        the same integers when the stream is at that cell's ambient.
        GAP, named: a marine on a vacuum tile then radiates at the vacuum's
        ambient, not at 310 K. Under thermal v2 R4 (space is at room
        temperature in v1) no shipped scene has a cold ambient anywhere, so
        this is a mechanism choice with no live consequence today; a body with
        an emission temperature of its own is a T5-or-later question.
      * the VIRTUAL AMBIENT RING (2.7): an out-of-grid upwind read returns the
        READING cell's own amb_m_i. The ring cell is outside the grid and has
        no cell of its own, and conservation survives any choice (whatever
        integer arrives is booked as -fa/-fb into the reading cell's rad_amb),
        so this is a PIN, not a correctness accident. Pinned this way because
        (a) it is the only choice that keeps a boundary cell at its own ambient
        an exact fixed point, (b) it needs no new global and no second ingress
        door, (c) it reduces to the scalar era exactly when the ambient is
        uniform, and (d) the CUDA twin (P4) reads a register it already holds
        instead of a neighbour outside the grid.
    body_mode : "reemit"  -> Erik's ruling (design row 25), emit_body = (amb_m*b)>>16
                "sink"    -> the pre-ruling variant, kept only so the gate that
                             says the ruling was needed cannot pass vacuously.

    Out-of-grid upwind reads return `amb_m` (the virtual ambient ring, 2.7).
    """
    if body_mode not in ("reemit", "sink"):
        raise ValueError(f"unknown body_mode {body_mode!r}")
    h, w = len(a), len(a[0])
    if w_m is None:
        w_m = ONE // n_ord
    amb_lvl = ambient_plane(e_ref, h, w, table)
    if validate:
        validate_planes(a, d, k, amb_lvl, table)
    # The per-cell ambient stream. NOT hoisted out of the cell loop -- that
    # hoist is precisely what thermal v2 section 6 item 3 exists to make
    # impossible, and it is what the first draft of this patch would have
    # shipped while passing every other gate.
    amb_m_cell = [[(amb_lvl[y][x] * w_m) >> 16 for x in range(w)]
                  for y in range(h)]
    if f_plane is None:
        f_plane = plane(h, w, F_ONE)

    res = SweepResult(plane(h, w), plane(h, w), plane(h, w), plane(h, w))
    rad_net, rad_flux, rad_amb, rad_flu = (res.rad_net, res.rad_flux,
                                           res.rad_amb, res.rad_fluence)
    # The per-cell excess emission over THAT CELL'S OWN ambient, independent of
    # ordinate. Non-negative by validate_planes' `amb <= E°[0]` invariant, so
    # the check below is provably dead on a validated scene and is kept as belt
    # and braces for a caller that passed validate=False.
    ex_cell = [[table[e_bucket_of(T[y][x])] - amb_lvl[y][x] for x in range(w)]
               for y in range(h)]
    for row in ex_cell:
        for v in row:
            if v < 0:
                raise ValueError("E°[T] < the cell's own ambient level: "
                                 "the excess must be >= 0")

    max_stream = 0
    min_stream = None
    max_product = 0
    max_fleck_product = 0
    for (mu, eta) in ordinates(n_ord, half_offset):
        x_major, s_m, sx, sy = ordinate_constants(mu, eta, transport)
        store = [[None] * w for _ in range(h)]
        xs = range(w) if mu > 0 else range(w - 1, -1, -1)
        ys = range(h) if eta > 0 else range(h - 1, -1, -1)
        if transport == "shear":
            order = ([(y, x) for x in xs for y in ys] if x_major
                     else [(y, x) for y in ys for x in xs])
        else:
            # any topological order of the DAG; the gather form makes the
            # result independent of which one (critique 3 section 2a)
            order = [(y, x) for y in ys for x in xs]

        for (y, x) in order:
            if transport == "shear":
                if x_major:
                    ua, ub = (y, x - sx), (y - sy, x - sx)
                else:
                    ua, ub = (y - sy, x), (y - sy, x - sx)
            else:
                ua, ub = (y, x - sx), (y - sy, x)
            in_a = 0 <= ua[0] < h and 0 <= ua[1] < w
            in_b = 0 <= ub[0] < h and 0 <= ub[1] < w
            amb_m = amb_m_cell[y][x]          # THIS cell's ambient stream
            io_a = store[ua[0]][ua[1]] if in_a else amb_m   # ring: cell i's own
            io_b = store[ub[0]][ub[1]] if in_b else amb_m
            if in_a:
                assert io_a is not None, "upwind not yet computed: wavefront order broken"
            if in_b:
                assert io_b is not None, "upwind not yet computed: wavefront order broken"
            # the split, recomputed from the stored outflow with the same shift
            fa = (io_a * s_m) >> 16
            fb = io_b - ((io_b * s_m) >> 16)       # the REMAINDER (critique 1 fix 3)
            i_in = fa + fb
            if not in_a:
                rad_amb[y][x] -= fa                 # taken from the virtual ring
            if not in_b:
                rad_amb[y][x] -= fb

            ki = k[y][x]
            leaked = (i_in * ki) >> 16
            ret = (amb_m * ki) >> 16          # cell i's own ceiling
            stream = i_in - leaked + ret
            ai = a[y][x]
            bi = d[y][x] - ai
            abs_mat = (stream * ai) >> 16
            abs_body = (stream * bi) >> 16
            ex_m = (ex_cell[y][x] * w_m) >> 16
            fleck_product = ex_m * f_plane[y][x]          # the widest Q24 product
            src = amb_m + (fleck_product >> F_SHIFT)
            emitted = (src * ai) >> 16
            emit_body = ((amb_m * bi) >> 16) if body_mode == "reemit" else 0
            i_out = stream - abs_mat - abs_body + emitted + emit_body
            if stream < 0 or i_out < 0:
                raise AssertionError(
                    f"positivity broken at ({y},{x}): stream={stream} i_out={i_out}")

            rad_net[y][x] += abs_mat - emitted          # the material ledger
            rad_flux[y][x] += abs_body - emit_body      # the body sensor, signed
            rad_amb[y][x] += leaked - ret               # the ceiling channel
            rad_flu[y][x] += stream
            store[y][x] = i_out

            # what this cell's own outflow sends into the virtual ring
            fa_o = (i_out * s_m) >> 16
            fb_o = i_out - fa_o
            if transport == "shear":
                if x_major:
                    ta, tb = (y, x + sx), (y + sy, x + sx)
                else:
                    ta, tb = (y + sy, x), (y + sy, x + sx)
            else:
                ta, tb = (y, x + sx), (y + sy, x)
            if not (0 <= ta[0] < h and 0 <= ta[1] < w):
                rad_amb[y][x] += fa_o
            if not (0 <= tb[0] < h and 0 <= tb[1] < w):
                rad_amb[y][x] += fb_o

            if stream > max_stream:
                max_stream = stream
            if min_stream is None or stream < min_stream:
                min_stream = stream
            p = max(stream * ai, stream * bi, ex_cell[y][x] * w_m,
                    fleck_product, src * ai)
            if p > max_product:
                max_product = p
            if fleck_product > max_fleck_product:
                max_fleck_product = fleck_product

    res.max_stream = max_stream
    res.min_stream = 0 if min_stream is None else min_stream
    res.max_abs_net = plane_max_abs(rad_net)
    res.max_fluence = max(v for row in rad_flu for v in row)
    res.max_product = max_product
    res.max_fleck_product = max_fleck_product
    return res


def fleck_prepass(T, a, his, *, table=E, e_ref=None, enabled: bool = True,
                  alpha_floor: str = ALPHA_FLOOR_DEFAULT):
    """The pre-sweep Fleck pass for SOLIDS (design section 2.8). Q24 (row 32).

    `his` may be an int (uniform) or a plane. `e_ref` is the ambient LEVEL, the
    same scalar-or-plane the sweep takes (ambient_plane): L_q is the cell's free
    EXCESS emission, and the excess is over THAT CELL'S OWN ambient -- a cell
    facing a cold sky has more excess to shed and is damped accordingly.
    Returns the f_plane.
    """
    h, w = len(T), len(T[0])
    if not enabled:
        return plane(h, w, F_ONE)
    amb_lvl = ambient_plane(e_ref, h, w, table)
    out = plane(h, w, F_ONE)
    for y in range(h):
        for x in range(w):
            s = his if isinstance(his, int) else his[y][x]
            out[y][x] = fleck_f_solid_q(T[y][x], a[y][x], s, table,
                                        amb_lvl[y][x],
                                        alpha_floor=alpha_floor)[0]
    return out


# --------------------------------------------------------------------------- #
# The Pass-1 fold for solids (temperature_solver.cpp:247-299 + the clamp).
# --------------------------------------------------------------------------- #
@dataclass
class FoldCounters:
    t_max_phys_hits: int = 0
    t_low_rail_hits: int = 0
    rad_clamp_hits: int = 0
    e_solid_deposit_sum: int = 0
    clamp_drop_sum: int = 0       # Q16.16 temperature removed by the clamp

    def zero(self):
        return (self.t_max_phys_hits == 0 and self.t_low_rail_hits == 0
                and self.rad_clamp_hits == 0)


def fold_pass1_solid(T, rad_net, rad_fluence, his, ts, counters: FoldCounters, *,
                     clamp_enabled: bool = True, rails_enabled: bool = True,
                     int32_sat: bool = True, cap_real=None, held=None, table=E):
    """The radiative sub-step of Pass 1, for thermal solids, IN ORDER.

        t_before = T[i]
        dTr      = shr_round0(rad_net[i], heat_inv_shift[i])
        T[i]     = sat_add_q16(T[i], dTr)                      # T_after
        T[i]     = min(T_after, max(t_before, E°⁻¹(Phi)))       # the clamp (row 21)
        rails: T_MAX_PHYS, then the low rail at 0
        books the ACTUAL applied dT x cap_real into e_solid_deposit_sum

    `clamp_enabled=False` models P1's `clamp_enabled` binding keyword (gate 5's
    non-vacuity). `rails_enabled=False` and `int32_sat=False` exist only to
    reproduce the float study's unbounded runaway number; the engine has both.
    `held` is an optional mask of cells whose temperature is pinned (a source).
    """
    h, w = len(T), len(T[0])
    for y in range(h):
        for x in range(w):
            if not ts[y][x]:
                continue
            if held is not None and held[y][x]:
                continue
            rn = rad_net[y][x]
            if rn == 0:
                continue
            s = his if isinstance(his, int) else his[y][x]
            t_before = T[y][x]
            dTr = shr_round0(rn, s)
            t_after = sat_add_q16(t_before, dTr) if int32_sat else t_before + dTr
            t_new = t_after
            if clamp_enabled:
                t_cap = e_inv_q(rad_fluence[y][x], table)
                ceiling = t_cap if t_cap > t_before else t_before
                if t_new > ceiling:
                    counters.rad_clamp_hits += 1
                    counters.clamp_drop_sum += t_new - ceiling
                    t_new = ceiling
            if rails_enabled:
                if t_new > T_MAX_PHYS_Q:
                    t_new = T_MAX_PHYS_Q
                    counters.t_max_phys_hits += 1
                if t_new < 0:
                    t_new = 0
                    counters.t_low_rail_hits += 1
            T[y][x] = t_new
            cap = 1 << s if cap_real is None else cap_real[y][x]
            counters.e_solid_deposit_sum += (t_new - t_before) * cap


# --------------------------------------------------------------------------- #
# A scene and a multi-tick runner.
# --------------------------------------------------------------------------- #
@dataclass
class Scene:
    """Everything one tick of (Fleck pre-pass -> sweep -> Pass-1 fold) needs."""
    a: list
    d: list
    k: list
    T: list
    his: object = 3                 # int or plane; log2(thermal_mass)
    ts: list = None                 # the thermal-solid mask; default a > 0
    held: list = None               # temperatures pinned by another heater
    n_ord: int = 16
    transport: str = "shear"
    body_mode: str = "reemit"
    e_ref: int = None
    fleck: bool = True
    alpha_floor: str = ALPHA_FLOOR_DEFAULT
    counters: FoldCounters = field(default_factory=FoldCounters)

    def __post_init__(self):
        h, w = len(self.a), len(self.a[0])
        if self.ts is None:
            self.ts = [[1 if self.a[y][x] > 0 else 0 for x in range(w)]
                       for y in range(h)]

    def tick(self, *, clamp_enabled=True, rails_enabled=True, int32_sat=True):
        f = fleck_prepass(self.T, self.a, self.his, e_ref=self.e_ref,
                          enabled=self.fleck, alpha_floor=self.alpha_floor)
        res = sweep_q(self.a, self.d, self.k, self.T, n_ord=self.n_ord,
                      transport=self.transport, f_plane=f, e_ref=self.e_ref,
                      body_mode=self.body_mode)
        fold_pass1_solid(self.T, res.rad_net, res.rad_fluence, self.his, self.ts,
                         self.counters, clamp_enabled=clamp_enabled,
                         rails_enabled=rails_enabled, int32_sat=int32_sat,
                         held=self.held)
        return res

    def run(self, ticks, **kw):
        last = None
        for _ in range(ticks):
            last = self.tick(**kw)
        return last


# --------------------------------------------------------------------------- #
# The 0-D cell model: the aggregate of the sweep's per-ordinate terms under a
# HELD fluence. This is what the stability and equilibrium tables run on
# (stability_study.py's scenario, in the new forms).
# --------------------------------------------------------------------------- #
def cell_rad_net_q(T_q: int, phi: int, a_q: int, his: int, *, table=E,
                   e_ref: int = None, fleck: bool = True,
                   alpha_floor: str = ALPHA_FLOOR_DEFAULT) -> int:
    """rad_net for one cell under a held total fluence Phi, in the excess form:

        rad_net = ((Phi*a) >> 16) - ((src*a) >> 16),
        src     = e_ref + ((ex * f) >> 24),  ex = E°[T] - e_ref

    This is section 2.3's per-ordinate pair summed over the ordinates with the
    per-ordinate truncations folded into one; it differs from a full sweep by at
    most N counts, which is why the sweep gates run on the sweep and only the
    stability tables run here.
    """
    ref = table[0] if e_ref is None else e_ref
    ex = table[e_bucket_of(T_q)] - ref
    if ex < 0:
        ex = 0
    f_q = F_ONE
    if fleck:
        f_q = fleck_f_q(T_q, shr_round0((a_q * ex) >> 16, his),
                        alpha_floor=alpha_floor)
    src = ref + ((ex * f_q) >> F_SHIFT)
    return ((phi * a_q) >> 16) - ((src * a_q) >> 16)


def cell_march(T0_q: int, phi: int, a_q: int, his: int, ticks: int, *,
               table=E, e_ref: int = None, fleck: bool = True,
               alpha_floor: str = ALPHA_FLOOR_DEFAULT,
               clamp_enabled: bool = True, rails_enabled: bool = True,
               int32_sat: bool = True, trace: bool = False):
    """March one cell `ticks` ticks under a held fluence. Returns (T_q, counters)
    or (trace list, counters) when trace=True."""
    T = [[T0_q]]
    ts = [[1]]
    counters = FoldCounters()
    out = [T0_q]
    for _ in range(ticks):
        rn = cell_rad_net_q(T[0][0], phi, a_q, his, table=table, e_ref=e_ref,
                            fleck=fleck, alpha_floor=alpha_floor)
        fold_pass1_solid(T, [[rn]], [[phi]], his, ts, counters,
                         clamp_enabled=clamp_enabled, rails_enabled=rails_enabled,
                         int32_sat=int32_sat, table=table)
        out.append(T[0][0])
    return (out if trace else T[0][0]), counters


# --------------------------------------------------------------------------- #
# Analytic references (float, for the cooling-curve comparison only).
# --------------------------------------------------------------------------- #
def e_deg_float(T_game: float, rad_scale: float = RAD_SCALE) -> float:
    return rad_scale * (K_AMB + K_SLOPE * max(T_game, 0.0)) ** 4


def cool_rate_const(a: float, his: int, rad_scale: float = RAD_SCALE) -> float:
    """c in dT/dt = -c * [(T+K)^4 - K_ref^4], per SECOND, in game units."""
    return a * rad_scale * TICK_HZ / (2.0 ** his) / ONE


def cool_exact_bath(T0: float, seconds: float, a: float, his: int) -> float:
    """The analytic solution of dT/dt = -c[(T+K)^4 - K^4] (cooling into an AMBIENT
    bath, which is the law the excess form discretises).

        F(u) = 1/(2K^2) * [ 1/(2K) ln((u-K)/(u+K)) - (1/K) atan(u/K) ],
        F(u) - F(u0) = -c t,   u = T + K

    Inverted by bisection (F is strictly decreasing in u for u > K).
    """
    c = cool_rate_const(a, his)
    K = float(K_AMB)

    def F(u):
        return (1.0 / (2.0 * K * K)) * ((1.0 / (2.0 * K)) * math.log((u - K) / (u + K))
                                        - (1.0 / K) * math.atan(u / K))

    target = F(T0 + K) - c * seconds
    lo, hi = K * (1.0 + 1e-12), T0 + K
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if F(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) - K


def cool_exact_zero_sky(T0: float, seconds: float, a: float, his: int) -> float:
    """stability_study.py's analytic reference: dT/dt = -c (T+K)^4, no bath.
    Kept so this file can reproduce the OLD table as an instrument check."""
    c = cool_rate_const(a, his)
    k0 = T0 + K_AMB
    return (k0 ** -3 + 3.0 * c * seconds) ** (-1.0 / 3.0) - K_AMB


def e_inv_float(phi: float, rad_scale: float = RAD_SCALE) -> float:
    """The continuous inverse of E°, for reporting the exact equilibrium beside
    the table's bucket-quantized one."""
    if phi <= 0.0:
        return 0.0
    return (phi / rad_scale) ** 0.25 - K_AMB


# --------------------------------------------------------------------------- #
# A float PUSH port of sweep_ref.py, EXTENDED with the body share and the body's
# ambient re-emission, so gate 7 can compare on a scene with bodies. On a scene
# without bodies the committed sweep_ref.py itself is used instead (stronger).
# --------------------------------------------------------------------------- #
def sweep_push_float(a, d, k, T, *, n_ord=16, transport="shear", f_plane=None,
                     e_ref=None, table=E, body_mode="reemit"):
    """sweep_ref.py's push form, in floats, with design v3's body terms."""
    h, w = len(a), len(a[0])
    ref = float(table[0] if e_ref is None else e_ref)
    wt = 1.0 / n_ord
    dE = [[0.0] * w for _ in range(h)]
    Ecell = [[float(table[e_bucket_of(T[y][x])]) for x in range(w)] for y in range(h)]
    if f_plane is None:
        f_plane = plane(h, w, F_ONE)
    for (mu, eta) in ordinates(n_ord):
        am, ae = abs(mu), abs(eta)
        sx = 1 if mu > 0 else -1
        sy = 1 if eta > 0 else -1
        inflow = [[0.0] * w for _ in range(h)]
        bnd = ref * wt
        if transport == "shear":
            x_major = am >= ae
            fm = (ae / am) if x_major else (am / ae)
            col = 0 if mu > 0 else w - 1
            row = 0 if eta > 0 else h - 1
            if x_major:
                for y in range(h):
                    inflow[y][col] += bnd
                for x in range(w):
                    inflow[row][x] += fm * bnd
                inflow[row][col] -= fm * bnd
            else:
                for x in range(w):
                    inflow[row][x] += bnd
                for y in range(h):
                    inflow[y][col] += fm * bnd
                inflow[row][col] -= fm * bnd
            xs = range(w) if mu > 0 else range(w - 1, -1, -1)
            ys = range(h) if eta > 0 else range(h - 1, -1, -1)
            order = ([(y, x) for x in xs for y in ys] if x_major
                     else [(y, x) for y in ys for x in xs])
        else:
            fx_w = am / (am + ae)
            col = 0 if mu > 0 else w - 1
            row = 0 if eta > 0 else h - 1
            for y in range(h):
                inflow[y][col] += bnd * fx_w
            for x in range(w):
                inflow[row][x] += bnd * (1.0 - fx_w)
            xs = range(w) if mu > 0 else range(w - 1, -1, -1)
            ys = range(h) if eta > 0 else range(h - 1, -1, -1)
            order = [(y, x) for y in ys for x in xs]
        for (y, x) in order:
            i_in = inflow[y][x]
            kl = k[y][x] / ONE
            leaked = i_in * kl
            ret = ref * wt * kl
            stream = i_in - leaked + ret
            ai = a[y][x] / ONE
            bi = (d[y][x] - a[y][x]) / ONE
            absorbed = stream * ai
            body = stream * bi
            ex = (Ecell[y][x] - ref) * (f_plane[y][x] / F_ONE)
            emitted = (ref + ex) * wt * ai
            emit_body = (ref * wt * bi) if body_mode == "reemit" else 0.0
            i_out = stream - absorbed - body + emitted + emit_body
            dE[y][x] += absorbed - emitted
            if transport == "shear":
                fa = i_out * (1.0 - fm)
                fb = i_out - fa
                if x_major:
                    if 0 <= x + sx < w:
                        inflow[y][x + sx] += fa
                        if 0 <= y + sy < h:
                            inflow[y + sy][x + sx] += fb
                else:
                    if 0 <= y + sy < h:
                        inflow[y + sy][x] += fa
                        if 0 <= x + sx < w:
                            inflow[y + sy][x + sx] += fb
            else:
                fx = i_out * fx_w
                fy = i_out - fx
                if 0 <= x + sx < w:
                    inflow[y][x + sx] += fx
                if 0 <= y + sy < h:
                    inflow[y + sy][x] += fy
    return dE
