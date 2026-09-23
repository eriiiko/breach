"""The heat law's END-TO-END properties, on the sweep (issue #12, T4).

These four re-home the properties design v3 §11.1 marks "rewrite" when the old
cast's two modules die:

  * `tests/test_fire_heat_source.py` — the chain across an air gap, and a body
    beside a fire taking radiation at all;
  * `tests/test_heat_attenuation.py` — the `air < glass < wall` ordering, and
    heat/light independence.

They are written against the SHADOW sweep, which is what exists today: the fold
still reads the old cast, so nothing here can assert HP or ignition through
`Simulation.step()`. Each therefore asserts the property one seam earlier — on
the planes the flip's fold will read (`rad_net`, `rad_flux`, `rad_fluence`), in
the units the flip's consumers use (`E°⁻¹(Φ)` against a material's own
`ignition_temp`). At the flip these become the natural end-to-end assertions
without changing what they claim.

The other two of §11.1's four re-homes need no new test; the property already
has a home that survives the flip, named in `report_t4.md` §1.1 A3:
determinism is `tests/test_w6_armory.py`'s GOLDEN_AGGREGATE plus gate 0, and
"the cast is wired into the tick" is
`tests/test_radiation_sweep_shadow_wiring.py`.

Scene vocabulary and drivers: `tests/_radiation_sweep_harness.py`, the same one
gates 0-6 use. Every optical and thermal number comes from the SHIPPED material
table, so a material re-tune moves these tests with the game rather than
stranding them.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_sweep_heat_law_properties.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from _radiation_sweep_harness import (  # noqa: E402  (sets up sys.path)
    ONE, cpp_sweep, reference_table)
import sweep_ref_q_gates as G  # noqa: E402  (the reference's scene constants)
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_FURNITURE, MAT_GLASS, MAT_HULL, MAT_WOOD, MaterialTable)

_TBL = MaterialTable.from_config()

A_AIR = int(_TBL.heat_atten_q16[MAT_AIR])        # 0     — transparent
A_GLASS = int(_TBL.heat_atten_q16[MAT_GLASS])    # 0.3
A_WALL = int(_TBL.heat_atten_q16[MAT_HULL])      # 0.85  — T5b: a real emissivity
A_OPAQUE = ONE                                   # 1.0   — the optical limit
#
# T5b, and it matters for every "shadow" claim below. `heat_atten` does DOUBLE
# DUTY on this plane: it is the cell's EMISSIVITY (what it radiates) and its
# EXTINCTION coefficient (what it stops), because the model has an absorb
# channel and a transmit channel but no REFLECT channel. Until T5b every
# structural row carried the placeholder 1.0, so the two readings coincided and
# a wall was perfectly opaque. T3 D6 §7.3 gives hull/steel their real
# emissivity, 0.85 (Incropera A.11), and the arithmetic consequence is that a
# single hull cell now TRANSMITS 15 % of the stream crossing it.
#
# Measured on this scene, fire-plateau source, `E°⁻¹(Φ)` at the target:
#     clear line       388 game   (ignites wood, ign 300)
#     one hull cell    148 game   (does NOT ignite)
#     two hull cells    36 game
#     a = ONE            0 game   (the ambient floor, bit for bit)
#
# So the GAMEPLAY property — a bulkhead stops a fire lighting the woodwork on
# the other side — survives, and is what the tests below assert of a hull wall.
# The OPTICAL property — a perfectly opaque column passes nothing at all — is
# asserted separately at `a = ONE`, which is the only value that actually means
# it. Whether an opaque STRUCTURAL row should carry 1.0 for extinction while
# emitting at 0.85 is Erik's (report_t5b.md §12, and T3 D6 §7.3 records the
# same tension for glass).
A_WOOD = int(_TBL.heat_atten_q16[MAT_WOOD])
HIS_WOOD = int(_TBL.heat_inv_shift[MAT_WOOD])
IGN_WOOD_Q16 = int(_TBL.ignition_temp_q16[MAT_WOOD])

T_FIRE_Q = G.T_SRC_GAME << 16      # the gates' own fire-plateau source, Q16.16
T_AMBIENT_Q = 0                    # temperature stores ΔT above ambient

H, W = 5, 13
SRC_X, TGT_X, OBST_X = 1, 5, 3     # source | air | OBST | air | target
MID_Y = 2


def _blank():
    """An empty H×W scene in the reference's list format: no absorbers, no
    bodies, everything at ambient, every cell log2(thermal_mass) == 3."""
    return ([[0] * W for _ in range(H)],        # a   — material extinction
            [[0] * W for _ in range(H)],        # d   — stamped extinction
            [[0] * W for _ in range(H)],        # T   — Q16.16 ΔT
            [[3] * W for _ in range(H)],        # his — log2 thermal mass
            [[0] * W for _ in range(H)])        # ts  — thermal-solid mask


def _emitter(a, d, T, ts, x, temp_q):
    a[MID_Y][x] = ONE
    d[MID_Y][x] = ONE
    ts[MID_Y][x] = 1
    T[MID_Y][x] = temp_q


def _column(a, d, ts, x, atten_q):
    """A full-height thermal-solid column at `x` — the obstacle. `atten_q == 0`
    is a real shipped case (`foliage`: a thermal solid that absorbs nothing)."""
    for y in range(H):
        a[y][x] = atten_q
        d[y][x] = atten_q
        ts[y][x] = 1


def _sweep(a, d, T, his, ts, *, table):
    """(rad_net, rad_flux, rad_amb, rad_fluence) for one scene."""
    rn, rf, ra, rl, _f, _sw = cpp_sweep(a, d, 0, T, his, ts, table=table)
    return rn, rf, ra, rl


# --------------------------------------------------------------------------- #
# 1. The chain across an air gap  (re-homes test_fire_heat_source.py::
#    test_full_chain_heat_ignites_air_separated_wood)
# --------------------------------------------------------------------------- #
def test_radiation_crosses_an_air_gap_and_clears_wood_s_own_ignition_temperature():
    """PROPERTY: a fire-temperature emitter separated from a WOOD tile by three
    cells of air delivers a fluence Φ whose `E°⁻¹(Φ)` clears wood's own
    `ignition_temp`, and the tile's `rad_net` is a strict gain. An opaque wall
    across the line drives both to the ambient-source floor, exactly.

    This is the pre-flip form of the ignition chain: `E°⁻¹(Φ)` is the same
    quantity the Pass-1 clamp and the tile inspector read, and `ignition_temp`
    is the engine's own criterion — so nothing here is a fitted number.

    BREAKS IF: air stops being transparent (`a = 0` no longer passes a stream),
    occlusion stops applying, `E°⁻¹` stops inverting the E table, or the derived
    scale (`[physics.radiation] rad_scale_derived`, P2b) collapses far enough
    that a fire no longer reaches an adjacent room's woodwork.
    """
    table = reference_table()

    def scene(*, wall, src_q):
        a, d, T, his, ts = _blank()
        _emitter(a, d, T, ts, SRC_X, src_q)
        a[MID_Y][TGT_X] = A_WOOD
        d[MID_Y][TGT_X] = A_WOOD
        ts[MID_Y][TGT_X] = 1
        his[MID_Y][TGT_X] = HIS_WOOD
        if wall is not None:
            _column(a, d, ts, OBST_X, wall)
        return _sweep(a, d, T, his, ts, table=table)

    rn_clear, _rfc, _rac, rl_clear = scene(wall=None, src_q=T_FIRE_Q)
    rn_wall, _rfw, _raw, rl_wall = scene(wall=A_WALL, src_q=T_FIRE_Q)
    rn_op, _rfo, _rao, rl_op = scene(wall=A_OPAQUE, src_q=T_FIRE_Q)
    _rnc, _rfk, _rak, rl_cold = scene(wall=None, src_q=T_AMBIENT_Q)

    reached_q = int(table.e_inv_q(int(rl_clear[MID_Y, TGT_X])))
    assert reached_q >= IGN_WOOD_Q16, (
        f"across three air cells the wood tile only sees "
        f"{reached_q / 65536.0:.1f} game, below its own ignition_temp "
        f"{IGN_WOOD_Q16 / 65536.0:.1f} — the chain reaches, but not far enough "
        f"to light anything")
    assert int(rn_clear[MID_Y, TGT_X]) > 0, "the target did not gain energy"

    # THE GAMEPLAY CONTROL — a HULL bulkhead stops the fire lighting the
    # woodwork on the other side. At the real emissivity 0.85 it transmits 15 %
    # per cell, so this is a threshold claim, not a zero claim (measured: 388
    # game clear, 148 behind one hull cell, ignition at 300).
    wall_q = int(table.e_inv_q(int(rl_wall[MID_Y, TGT_X])))
    assert wall_q < IGN_WOOD_Q16, (
        f"a hull bulkhead let through enough to light the wood behind it: "
        f"{wall_q / 65536.0:.1f} game vs ignition {IGN_WOOD_Q16 / 65536.0:.1f}")
    assert wall_q < reached_q, "the bulkhead attenuated nothing"

    # THE OPTICAL CONTROL — at `a = ONE`, and ONLY there, occlusion drives the
    # target to the SAME floor an ambient source leaves it at, bit for bit.
    assert int(rl_op[MID_Y, TGT_X]) == int(rl_cold[MID_Y, TGT_X]), (
        "a fully opaque column did not fully shadow the source")
    assert int(rn_op[MID_Y, TGT_X]) == 0
    # Non-vacuity: the ambient-source control is genuinely below ignition, so
    # the assertions above could not be satisfied by an all-hot scene.
    assert int(table.e_inv_q(int(rl_cold[MID_Y, TGT_X]))) < IGN_WOOD_Q16


# --------------------------------------------------------------------------- #
# 2. A body takes radiation  (re-homes the radiation half of
#    test_fire_heat_source.py::test_unit_next_to_fire_loses_hp_and_zombie_takes_4x;
#    the 4x multiplier itself lives in tests/test_unit_heat_damage.py::
#    test_zombie_takes_fire_multiplier_more and tests/test_damage_pipeline.py)
# --------------------------------------------------------------------------- #
def test_a_body_in_a_clear_line_absorbs_into_rad_flux_and_a_wall_shuts_it_off():
    """PROPERTY: a stamped body on an air tile (`d > a`, design v3 §6.2 — the
    marine's own extinction) in a clear line to a fire-temperature emitter books
    a strictly positive `rad_flux`; the SAME cell with an opaque wall between
    absorbs exactly nothing; and with no body stamped there the cell books
    nothing either.

    This is the channel §6.2 hands the flip: unit heat damage reads the flux a
    body absorbed. What a body does with that flux — the over-temperature ramp,
    and a zombie taking `zombie.fire_damage_multiplier` × a marine — is
    mechanism-independent and stays in `tests/test_unit_heat_damage.py`.

    BREAKS IF: the body share (`d − a`) stops absorbing, bodies stop being
    occluded by the geometry in front of them, or `rad_flux` stops being the
    plane bodies are booked into.
    """
    table = reference_table()

    def scene(*, body, wall):
        a, d, T, his, ts = _blank()
        _emitter(a, d, T, ts, SRC_X, T_FIRE_Q)
        if body:
            d[MID_Y][TGT_X] = ONE          # a marine stands on air: a stays 0
        if wall is not None:
            _column(a, d, ts, OBST_X, wall)
        return _sweep(a, d, T, his, ts, table=table)

    rf_clear = scene(body=True, wall=None)[1]
    rf_wall = scene(body=True, wall=A_WALL)[1]
    rf_opaque = scene(body=True, wall=A_OPAQUE)[1]
    rf_none = scene(body=False, wall=None)[1]

    assert int(rf_clear[MID_Y, TGT_X]) > 0, (
        "a body in a clear line to a fire absorbed nothing")
    # A HULL bulkhead at its real emissivity 0.85 transmits 15 % per cell, so
    # what it buys a marine is a large attenuation, not immunity. Measured on
    # this scene it is ~7.7x less flux; assert an order-of-magnitude floor so
    # the claim tracks the table instead of pinning today's ratio.
    assert 0 < int(rf_wall[MID_Y, TGT_X]) < int(rf_clear[MID_Y, TGT_X]) // 4, (
        "a hull bulkhead barely shielded the body behind it")
    # Only a FULLY OPAQUE column shuts the channel off entirely.
    assert int(rf_opaque[MID_Y, TGT_X]) == 0, (
        "a body behind a fully opaque wall still absorbed the fire's radiation")
    assert int(rf_none[MID_Y, TGT_X]) == 0, (
        "an air cell with no body stamped on it booked flux — rad_flux is no "
        "longer the body channel")


# --------------------------------------------------------------------------- #
# 3. The extinction ordering  (rewrites test_heat_attenuation.py::
#    test_heat_ordering_air_glass_wall as a scene on the sweep's planes)
# --------------------------------------------------------------------------- #
def test_downrange_fluence_orders_air_above_glass_above_wall():
    """PROPERTY: with one emitter and one full-height obstacle column, the
    fluence reaching a cell BEYOND the obstacle is strictly ordered by the
    obstacle's own shipped `heat_atten` — air (0) > glass (0.3) > hull (1.0) —
    and the opaque end is at or below what an ambient source leaves.

    The old march asserted the same ordering on its `heat` buffer; the sweep
    carries it on `rad_fluence`. The three coefficients are read from the
    shipped table, so this tracks a material re-tune instead of pinning 0.3.

    BREAKS IF: the extinction planes stop ordering transmission monotonically —
    an absorption term that is not monotone in `a`, a plane read at the wrong
    cell, or `a` and `d` swapped.
    """
    table = reference_table()
    assert A_AIR < A_GLASS < A_WALL, (
        "the shipped table no longer orders air < glass < hull in heat_atten; "
        "this test's premise is a FINDING, not a failure")

    def phi(atten_q, src_q=T_FIRE_Q):
        a, d, T, his, ts = _blank()
        _emitter(a, d, T, ts, SRC_X, src_q)
        _column(a, d, ts, OBST_X, atten_q)
        return int(_sweep(a, d, T, his, ts, table=table)[3][MID_Y, TGT_X])

    p_air, p_glass, p_wall = phi(A_AIR), phi(A_GLASS), phi(A_WALL)
    assert p_air > p_glass > p_wall, (
        f"downrange fluence is not ordered by heat_atten: air {p_air}, "
        f"glass {p_glass}, wall {p_wall}")
    # The ordering continues past the shipped table to the optical limit, and
    # THERE it meets the ambient floor. (T5b: hull is 0.85 now, not 1.0, so the
    # floor claim belongs at `a = ONE` -- see the A_OPAQUE note at the top.)
    p_opaque = phi(A_OPAQUE)
    assert p_wall > p_opaque, (
        f"a 0.85 bulkhead did not transmit more than a perfect absorber: "
        f"{p_wall} vs {p_opaque}")
    assert p_opaque <= phi(A_OPAQUE, src_q=T_AMBIENT_Q), (
        "a fully opaque column let the source through")


# --------------------------------------------------------------------------- #
# 4. Heat and light are separate columns  (rewrites test_heat_attenuation.py::
#    test_heat_transparent_light_opaque_tile and its converse, as one property)
# --------------------------------------------------------------------------- #
def test_the_sweep_s_extinction_is_heat_atten_alone_and_light_cannot_reach_it():
    """PROPERTY: the sweep's optical input is `heat_atten`, and nothing a
    material says about LIGHT can change it. Three legs:

      (a) the two columns are genuinely different numbers on shipped rows
          (glass 0.3 / 0.1, furniture 0.5 / 0.55) — so a mis-wire would move
          real values rather than being invisible;
      (b) the plane the live sweep reads, `GameMap.heat_atten_q`, is the
          `heat_atten_q16` projection cell for cell — never `light_atten`;
      (c) on the sweep itself, a thermal solid with `heat_atten == 0` — the
          shipped `foliage` case, and the old march's "light-opaque,
          heat-clear" tile — leaves ALL FOUR output planes bit-identical to no
          obstacle at all, while the same column at `heat_atten == 1` does not.

    The old march pinned this as "a light-opaque tile passes heat and a
    heat-opaque tile passes light". Light rejoins this same sweep at P6 on its
    own plane (`light_atten_q`); until then (a)+(b)+(c) is the whole of the
    independence claim that can be false.

    BREAKS IF: anyone feeds `light_atten` (or the RGB plane) into the sweep's
    extinction, or `heat_atten_q` stops being the per-material projection.
    """
    # (a) the columns differ on shipped rows
    light_max = _TBL.light_atten.max(axis=1)
    differing = [i for i in (MAT_GLASS, MAT_FURNITURE)
                 if float(_TBL.heat_atten[i]) != float(light_max[i])]
    assert differing, (
        "no shipped row separates heat_atten from light_atten any more — this "
        "test could not tell a mis-wire from the truth; that is a FINDING")

    # (b) the live plane is the heat projection
    from level_loader import LevelData
    from simulation.gamemap import GameMap
    tilemap = np.array([[MAT_AIR, MAT_GLASS, MAT_FURNITURE],
                        [MAT_HULL, MAT_FURNITURE, MAT_WOOD],
                        [MAT_AIR, MAT_GLASS, MAT_HULL]], dtype=np.int32)
    gmap = GameMap(LevelData(name="t4_optics", version="2", path=Path("."),
                             tilemap=tilemap, tile_size_m=1.0,
                             diffuse_path=Path(".")))
    assert np.array_equal(gmap.heat_atten_q,
                          _TBL.heat_atten_q16[gmap.material].astype(np.int32)), (
        "GameMap.heat_atten_q is not the heat_atten_q16 projection")
    # ...and it is NOT the light column, on a map that contains rows where the
    # two differ (the non-vacuity of (a), carried into the plane).
    light_q = np.rint(light_max[gmap.material] * 65536.0).astype(np.int32)
    assert not np.array_equal(gmap.heat_atten_q, light_q)

    # (c) the sweep responds to `a` and to nothing else about the cell
    table = reference_table()

    def planes(atten_q, *, obstacle):
        a, d, T, his, ts = _blank()
        _emitter(a, d, T, ts, SRC_X, T_FIRE_Q)
        if obstacle:
            _column(a, d, ts, OBST_X, atten_q)
        return _sweep(a, d, T, his, ts, table=table)

    bare = planes(0, obstacle=False)
    clear_solid = planes(A_AIR, obstacle=True)    # ts == 1, heat_atten == 0
    opaque = planes(A_WALL, obstacle=True)

    for i, name in enumerate(("rad_net", "rad_flux", "rad_amb", "rad_fluence")):
        assert np.array_equal(bare[i], clear_solid[i]), (
            f"a heat-transparent thermal solid changed {name} — something "
            f"other than heat_atten is attenuating the sweep")
    assert not np.array_equal(bare[3], opaque[3]), (
        "an opaque column changed nothing — the scene is vacuous")
