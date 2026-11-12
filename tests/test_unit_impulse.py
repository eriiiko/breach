"""G5p-a2 (#183) -- the unit impulse law, its linear part
(``simulation/unit_impulse.py``; ``docs/unit_impulse_design_2026-11-10.md``
§3-§6, §8 and the binding build sheet §13 with critique B's gates).

PROPERTIES PROTECTED (each test names its own and the change that breaks it)
  1. Soft input never moves a unit (§5, §8): an input train that never
     overflows the in-place sway leaves pv at exactly zero and the position
     unchanged, any direction, any phasing, with or without an earlier
     stagger.
  2. A steady force above the hold floors a unit of ANY stability within a
     bounded number of ticks.
  3. G4: the momentum books close exactly, every tick, every unit, wall,
     clamp and prone slide included.
  4. G5: the same rounds in phase and out of phase give the same position
     and the same mean effort over a period.
  5. G6: the torque's sign follows the physics and a mirrored scene gives
     the mirrored spin and facing to the count.
  6. G7: permuting the units' ids changes nothing.
  7. G8: a pushed unit never ends inside a wall nor crosses one at any speed
     the law can produce; at pv = 0 the move IS the walk's single-test
     slide.
  8. Displacement is monotone decreasing in mass.
  9. A prone unit under sustained fire comes to rest; nobody stands up while
     sliding or in an over-hold input.
 10. G12 / G13: a zombie pushed off its path never crosses a wall; a unit
     killed with pending impulses converts with none.
 11. The walk's two readers: the effort's walking scale and the walk ban.
 12. The shove: by mass and strength, from behind a stumble, from the front
     a fall, any team, the nearest unit in the facing's cone, its cooldown.
 13. The door refuses by name; the table is the design's numbers.
 14. The digest keys are absence-transparent; the events are render-only.
 15. G1 / G11: with every input off nothing is ever pushed in the pinned
     matches and they play re-pin 1; with the shipped dials every position
     stays on the Q16 grid.
"""
from __future__ import annotations

import ast
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import unit_impulse as ui  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.status import KNOCKED_DOWN, composed_flags  # noqa: E402
from simulation.turning import PI_Q16, place  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.unit_fixed import FP_ONE as ONE  # noqa: E402

#: The playground's tile (0.333 m): the design's numbers are at it.
TILE = 1.0 / 3.0
EAST, NORTH, WEST, SOUTH = 0, PI_Q16 // 2, PI_Q16, -(PI_Q16 // 2)


def _arena(h=48, w=48, tile=TILE, walls=()):
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    for (y0, y1, x0, x1) in walls:
        tm[y0:y1, x0:x1] = MAT_HULL
    return LevelData(name="impulse", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=tile, diffuse_path=Path("."))


@pytest.fixture
def live(monkeypatch):
    """The ruled real-momentum dials (1.0) whatever the shipped config says,
    for the gates that need a round's push or its recoil."""
    monkeypatch.setattr(CFG.impulse, "bullet_push_mult", 1.0)
    monkeypatch.setattr(CFG.impulse, "recoil_mult", 1.0)
    monkeypatch.setattr(CFG.impulse, "bullet_twist_mult", 1.0)


def _sim(**kw):
    sim = Simulation(_arena(**kw), seed=1, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _unit(sim, x, y, facing=EAST, team=0, name="M"):
    u = sim.get_unit(sim.add_unit(Unit(name, x=x, y=y, team=team)))
    place(u, facing)
    return u


def _kick(sim, u, dvx, dvy):
    """An impulse worth exactly ``(dvx, dvy)`` Q16 tiles per tick on ``u``."""
    ui.add(sim.impulse_table, u, dvx * u.mass_g, dvy * u.mass_g)


def _down(u):
    return composed_flags(u).is_prone


def _state(u):
    return (u.x, u.y, u.imp_x, u.imp_y, u.imp_l, u.swx, u.swy, u.pvx, u.pvy,
            u.effort, u.react_ticks, u.facing_q, u.omega_q, u.current_hp,
            tuple((s.kind, s.remaining_ticks) for s in u.statuses))


# ---------------------------------------------------------------------------
# 13 -- the door and the table
# ---------------------------------------------------------------------------
def test_the_table_quantizes_the_designs_numbers_once_per_map(live):
    """Property: at the playground's 0.333 m tiles and 24 tps one tick of
    hold is 753 counts (0.092 m/s), the smallest in-place capacity 1229
    (0.15 m/s), the reaction delay 0.30 s is 7 ticks (half up), the yaw
    inertia 3.0 kg m^2 is 27 000 g tile^2, the carbine round pushes with 3.6
    N s and the reference marine is 100 kg (80 + a 20 kg kit), a zombie 80.
    Breaks if a dial is read in the wrong unit, quantized twice or the kit
    is lost."""
    t = ui.impulse_table(CFG, TILE)
    assert t.cap_q == 753 and min(t.v_inplace_q) == 1229
    assert t.react_ticks == 7 and t.inertia_g == 27000
    j = 3.6 * 1000 / (TILE * 24)                       # g tile / tick
    assert abs(t.ammo_push_q["rifle_556_standard"] - j * ONE) <= 1
    assert t.ammo_recoil_q["rifle_556_standard"] == \
        t.ammo_push_q["rifle_556_standard"]
    sim = _sim()
    assert _unit(sim, 5, 5).mass_g == 100_000
    assert _unit(sim, 9, 5, team=ZOMBIE_TEAM, name="Z").mass_g == 80_000
    assert sim.gmap._impulse_table is sim.impulse_table


@pytest.mark.parametrize("key", ["k_hold", "v_fall_back_mps", "recoil_mult",
                                 "n_rec_ticks", "r_body_m"])
def test_the_door_refuses_a_missing_or_mistyped_key_by_name(monkeypatch,
                                                            key):
    """Property: every [impulse] key is required and typed; the refusal
    names it. Breaks if the door silently defaults a key."""
    monkeypatch.setattr(CFG.impulse, key, "fast")
    with pytest.raises(ValueError, match=key):
        ui.impulse_settings(CFG)
    monkeypatch.delattr(CFG.impulse, key)
    with pytest.raises(ValueError, match=key):
        ui.impulse_settings(CFG)


def test_the_door_refuses_a_hold_that_would_stagger_a_steady_sub_hold_force(
        monkeypatch):
    """Property (critique A, m3): one tick of hold must be below the smallest
    in-place capacity, compared on the quantized values. Breaks if the check
    is lost (a sub-hold force would then trip the stagger)."""
    monkeypatch.setattr(CFG.impulse, "k_hold", 0.5)
    with pytest.raises(ValueError, match="in-place capacity"):
        ui.impulse_table(CFG, TILE)


# ---------------------------------------------------------------------------
# 1 -- soft input never moves a unit
# ---------------------------------------------------------------------------
def _sub_overflow_train(rng, n, cap, lim):
    """A random pulse train (magnitudes, Q16 tiles/tick) that NEVER overflows
    a sway of capacity ``lim`` drained by ``cap`` a tick (the Lindley
    recursion of the collinear sway, with a 4-count margin for the integer
    root's floor) -- equivalently: any k consecutive ticks carry at most
    ``lim + (k - 1) cap``. Its mean is at most the hold."""
    out, s = [], 0
    for _ in range(n):
        p = int(rng.integers(0, lim + 1))
        if rng.random() < 0.3:
            p = 0
        p = min(p, lim - 4 - s)
        p = max(p, 0)
        out.append(p)
        s = max(0, s + p - cap)
    return out


@pytest.mark.parametrize("seed", range(6))
def test_soft_input_never_moves_a_unit_any_phasing_any_direction(seed):
    """Property 1 (§5 "why soft input can never move a unit", §8): an input
    train whose sway never overflows -- any k consecutive ticks carrying at
    most the smallest in-place capacity plus k - 1 ticks of hold, which
    bounds every tick by the capacity and the mean by the hold -- leaves pv
    exactly zero and the position exactly unchanged, in any direction to the
    facing, after an earlier stagger that has settled too. Breaks if the
    reaction delay switches the footing off (critique A, B1), the resist
    skips a tick, or anything but pv moves a unit. (The design's §8 wording,
    "per-tick |dv| within the capacity and mean below the hold", admits
    bursts such as two consecutive 1.6-hold ticks, which DO overflow; this
    window bound is the exact condition.)"""
    rng = np.random.default_rng(seed)
    sim = _sim()
    t = sim.impulse_table
    u = _unit(sim, 20, 20, facing=int(rng.integers(-PI_Q16, PI_Q16)))
    if seed % 2:
        _kick(sim, u, 8192, 0)               # a 1 m/s shove first ...
        for _ in range(80):
            sim.step()
        assert (u.pvx, u.pvy, u.swx, u.swy, u.react_ticks) == (0, 0, 0, 0, 0)
    x0, y0 = u.x, u.y
    theta = float(rng.uniform(-math.pi, math.pi))
    train = _sub_overflow_train(rng, 160, t.cap_q, min(t.v_inplace_q))
    assert max(train) > t.cap_q                       # bursts above the hold
    for p in train:
        _kick(sim, u, int(p * math.cos(theta)), int(p * math.sin(theta)))
        sim.step()
        assert (u.pvx, u.pvy) == (0, 0) and (u.x, u.y) == (x0, y0)
        assert u.react_ticks == 0 and not _down(u)


# ---------------------------------------------------------------------------
# 2 -- above the hold a unit falls, whatever its stability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stability, percent", [(0.5, 105), (0.9, 105),
                                                (1.0, 105), (1.5, 125)])
@pytest.mark.parametrize("facing", [EAST, NORTH, WEST])
def test_a_steady_force_above_the_hold_floors_any_stability(stability,
                                                            percent, facing):
    """Property 2 (§8, critique A B2): a steady force above the hold,
    pushing east, knocks a unit of ANY stability down within the law's own
    bound, whichever way it faces: through the effort's saturation while
    stability^2 is within reach of F / hold, else -- the sway full, every
    tick's input overflowing it, the reaction delay re-armed each tick --
    through its pushed velocity, which gains the excess every tick
    unresisted, reaching the fall limit (``v_fall x stability``). The bound
    is the reaction delay, the sway's fill and that climb at the excess, plus
    the effort filter's two seconds. Breaks if the effort filter cannot
    reach its input, a pushed unit is resisted with no margin left, or the
    fall limit is lost. (Above sqrt(F / hold) -- no shipped unit: the marine
    is 1.0, the zombie 0.9 -- that climb is a slide at the excess: a 1.5
    unit at 1.25 holds slides ~14 tiles before it falls; a finding for
    G5p-b, where wind reaches such units.)"""
    sim = _sim(w=64)
    t = sim.impulse_table
    u = _unit(sim, 4, 20, facing=facing)
    u.environment = dataclasses.replace(u.environment, stability=stability)
    f = (t.cap_q * percent) // 100
    ex = f - t.cap_q
    bound = (t.react_ticks + -(-max(t.v_inplace_q) // ex)
             + -(-int(max(t.v_fall_q) * stability) // ex) + 48)
    for tick in range(bound):
        _kick(sim, u, f, 0)
        sim.step()
        if _down(u):
            break
    assert _down(u), (stability, facing, bound)
    if stability * stability <= percent / 100:
        assert tick < 48            # the effort floors it within 2 s


# ---------------------------------------------------------------------------
# 3 -- G4, the momentum books
# ---------------------------------------------------------------------------
def test_g4_the_momentum_books_close_exactly_every_tick():
    """G4: for every unit, every tick, the impulse added equals mass x (pv +
    sway + resisted + wall + clamped) plus the accumulator's kept remainder,
    exactly in integers -- on a scene with a unit flung into a wall, one
    flung past the velocity bound and a prone slide, with odd impulses that
    leave remainders. Breaks if any term is unbooked (a wall's zeroing, the
    V_MAX clamp, a resist), or a division drops its remainder."""
    sim = _sim(walls=[(1, 47, 30, 32)])
    a = _unit(sim, 22, 10, name="wall")
    b = _unit(sim, 6, 20, name="vmax")
    c = _unit(sim, 6, 34, name="prone")
    for tick in range(60):
        if tick < 6:
            _kick(sim, a, 9000, 777)                    # into the wall
            ui.add(sim.impulse_table, a, 12345, -6789)  # remainders
        if tick == 2:
            _kick(sim, b, 9 * ui.V_MAX_Q, 3)            # past the bound
        if tick % 2 == 0 and tick < 40:
            _kick(sim, c, 1500, -400)                   # floors it, slides
        sim.step()
        for u in (a, b, c):
            (in_x, out_x), (in_y, out_y) = ui.books_identity(u)
            assert (in_x, in_y) == (out_x, out_y), (tick, u.name)
    assert a.impulse_books[ui.WALL_X] != 0
    assert b.impulse_books[ui.CLAMP_X] != 0
    assert c.impulse_books[ui.RES_X] != 0


# ---------------------------------------------------------------------------
# 4 -- G5, in phase and out of phase
# ---------------------------------------------------------------------------
def test_g5_rounds_in_phase_and_out_of_phase_give_one_position_and_band():
    """G5: four carbines on a 100 kg marine, all firing on even ticks (in
    phase) or two on each tick (out of phase): the position is EXACTLY the
    same (unmoved: no overflow either way) and the mean effort over a whole
    period of both trains agrees to within 4 counts (of ~51 000). Breaks if
    simultaneous rounds stagger where spread ones do not (critique A, B3).
    (Critique B asks for "one count per tick of period"; the effort step it
    fixes, d / 8 rounded AWAY from zero, carries a bias of less than 4
    counts on the mean of a 2-tick sawtooth -- solving its fixed point, the
    mean is D/2 + 4 (d1 - d2) with the two roundings d1, d2 in [0, 1) -- so
    the gate holds the filter's own bound.)"""
    j = ui.impulse_table(CFG, TILE)
    j_round = round(3.6 * 1000 / (TILE * 24) * ONE)   # one carbine round
    out = []
    for phased in (True, False):
        sim = _sim()
        u = _unit(sim, 20, 20, facing=WEST)           # pushed from the front
        x0 = (u.x, u.y)
        efforts = []
        for tick in range(240):
            n = (4 if tick % 2 == 0 else 0) if phased else 2
            ui.add(sim.impulse_table, u, -n * j_round, 0)
            sim.step()
            efforts.append(u.effort)
            assert (u.pvx, u.pvy) == (0, 0)
        # one whole period of BOTH trains: the pulse pattern (2 ticks) and
        # the kept remainder's cycle (J / m = 294.9 counts: 10 ticks)
        window = efforts[-120:]
        out.append(((u.x, u.y), sum(window), x0))
    assert out[0][0] == out[1][0] == out[0][2]
    assert abs(out[0][1] - out[1][1]) < 4 * 120  # < 4 counts on the mean
    assert out[0][1] > 120 * j.effort_edge_q  # the band is a slowed one


# ---------------------------------------------------------------------------
# 5 -- G6, the torque's sign and its mirror
# ---------------------------------------------------------------------------
def _hit_with_a_50(y_target):
    from simulation.combat import fire_burst
    sim = _sim()
    shooter = _unit(sim, 4, 19, facing=EAST, name="S")
    target = _unit(sim, 14, y_target, facing=EAST, name="T")
    target.current_hp = 10_000.0                  # the .50 must not kill it
    lr50 = sim.weapons_tables.weapons.by_name["lr50"]
    fire_burst(sim.gmap, sim.units, shooter, 5, 20, 30, 20, sim.tick, [],
               0.0, sim.rng, weapon=lr50, spread_deg=0.0, aim_q=0)
    l_after_hit = target.imp_l
    sim.step()
    return l_after_hit, target.facing_q, target.omega_q


def test_g6_a_round_right_of_centre_turns_the_target_counter_clockwise(live):
    """G6: a .50 round fired east along y = 20 past a target whose centre is
    half a tile NORTH of the line (the round passes on the right of the
    centre, as the shooter sees it) gives a positive angular impulse and
    turns the target COUNTER-clockwise in the facing's (y-up) frame -- the
    struck right side is driven back -- by what is left of the spin after
    the target's own torque has fought it for the tick (a carbine round's,
    or the .50's at a quarter tile, is cancelled within the tick: design
    §5.1); and the mirrored scene (the centre half a tile south) gives
    exactly the negated spin, facing and omega.
    Breaks if the cross product is taken in the y-down frame, the lever is
    unclamped or the rounding is not sign-symmetric. (Critique B's G6 says
    "clockwise" for this case; its own lever formula and sign convention,
    and the physics, give counter-clockwise -- built as the physics.)"""
    l_up, f_up, w_up = _hit_with_a_50(18.0)      # centre y 19.5: north
    l_dn, f_dn, w_dn = _hit_with_a_50(19.0)      # centre y 20.5: south
    assert l_up > 0 and l_dn == -l_up
    assert f_up > 0 and f_dn == -f_up and w_dn == -w_up


def test_the_lever_is_clamped_to_the_body_radius_on_each_input():
    """Property (critique A, M5): a push whose line passes far from the
    centre turns the unit as if it passed at the body radius; the clamp is
    per input (two half-impulses at two levers sum to their own clamps, not
    the clamp of a sum). Breaks if a lever is unclamped or the clamp moves
    onto the running sum."""
    t = ui.impulse_table(CFG, TILE)
    big = ui.angular_q(t, 10 ** 9, 0, 50 * ONE)
    edge = ui.angular_q(t, 10 ** 9, 0, t.r_body_q)
    assert big == edge > 0
    assert ui.angular_q(t, 10 ** 9, 0, -50 * ONE) == -edge


# ---------------------------------------------------------------------------
# 6 -- G7, the ids change nothing
# ---------------------------------------------------------------------------
def _permuted_scene(order, monkeypatch):
    sim = _sim()
    carbine = sim.weapons_tables.weapons.by_name["k5_carbine"]
    monkeypatch.setattr(carbine, "spread_deg", 0.0)   # no draw decides a hit
    spots = {"S1": (6, 14), "S2": (6, 20), "S3": (6, 26), "Z": (24, 20),
             "P": (21, 20)}
    units = {}
    for name in order:
        x, y = spots[name]
        team = ZOMBIE_TEAM if name == "Z" else 0
        units[name] = _unit(sim, x, y, facing=EAST, team=team, name=name)
        units[name].current_hp = 100_000.0       # nobody dies in the window
    units["Z"].zombie_activated = False
    for tick in range(36):
        for s in ("S1", "S2", "S3"):
            u, z = units[s], units["Z"]
            sim.set_aim(u.id, int((z.x - u.x) * 4096), int((z.y - u.y) * 4096))
            sim.set_trigger(u.id, True)
        if tick == 10:
            sim.shove_intent(units["P"].id)
        sim.step()
    return {n: _state(u) for n, u in units.items()}


def test_g7_permuting_the_units_ids_changes_nothing(monkeypatch, live):
    """G7: three carbines and a shove on one zombie, the units spawned in
    two different orders (so every id differs): every unit's position,
    impulse state, facing, hp and statuses, keyed by its identity, are
    equal. Breaks if a push depends on who pushed first (a clamp or a
    decision on the running sum, a slot reading another unit's
    mid-slot state)."""
    # The horde's own perception is not this law's: the zombie stays put
    # (its nearest-marine pick breaks ties by list order, a pre-existing id
    # dependence of the zombie AI, not of the impulse law).
    monkeypatch.setattr(CFG.zombie, "trigger_radius", 0)
    a = _permuted_scene(["S1", "S2", "S3", "Z", "P"], monkeypatch)
    b = _permuted_scene(["Z", "P", "S3", "S1", "S2"], monkeypatch)
    assert a == b
    assert a["Z"][:2] != (24.0, 20.0)          # the shove moved it
    assert all(a[s][9] > 0 for s in ("S1", "S2", "S3"))   # their recoil


# ---------------------------------------------------------------------------
# 7 -- G8, walls
# ---------------------------------------------------------------------------
def _solid_under(sim, u):
    fp = u.footprint
    return bool(sim.gmap.solid[u.tile_y:u.tile_y + fp,
                               u.tile_x:u.tile_x + fp].any())


@pytest.mark.parametrize("dx, dy", [(1, 0), (-1, 0), (0, 1), (0, -1),
                                    (1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_g8_at_the_velocity_bound_a_pushed_unit_never_enters_or_crosses_a_wall(
        dx, dy):
    """G8: a unit inside a closed box (walls one tile thick, a corner, a
    one-tile gap in its east wall narrower than its footprint) flung at the
    velocity bound in each of the 8 directions never ends a tick on a solid
    tile, never leaves the box, and comes to rest. Breaks if the move tests
    only its destination (tunnelling through a thin wall, M8), or a blocked
    axis keeps its push."""
    # box x 14..26, y 14..26 (walls), a one-tile gap at (20, 26)
    walls = [(14, 15, 14, 27), (26, 27, 14, 27), (14, 27, 14, 15),
             (14, 27, 26, 27)]
    sim = _sim(walls=walls)
    sim.gmap.material[20, 26] = MAT_AIR
    sim.gmap.solid[20, 26] = False
    u = _unit(sim, 19, 19)
    _kick(sim, u, dx * 20 * ui.V_MAX_Q, dy * 20 * ui.V_MAX_Q)
    for _ in range(40):
        sim.step()
        assert not _solid_under(sim, u)
        assert 15 <= u.x < 24 and 15 <= u.y < 24, (u.x, u.y)
    assert (u.pvx, u.pvy) == (0, 0)


def test_g8_at_no_push_the_move_is_the_walks_single_test_slide():
    """G8: with pv = 0 ``unit_impulse.move`` is exactly the pre-G5p-a2 walk
    -- the full step, else x alone, else y alone, else hold, each one
    ``is_passable_block`` test -- over a sweep of positions and steps near a
    wall, a corner and an open floor. Breaks if the substep rule splits a
    walk step or changes the slide's order."""
    sim = _sim(walls=[(10, 30, 20, 21), (20, 21, 10, 21)])
    g = sim.gmap
    rng = np.random.default_rng(7)
    u = _unit(sim, 5, 5)
    for _ in range(600):
        x = int(rng.integers(12 * ONE, 22 * ONE))
        y = int(rng.integers(12 * ONE, 26 * ONE))
        sx = int(rng.integers(-ONE // 2, ONE // 2))
        sy = int(rng.integers(-ONE // 2, ONE // 2))
        u.set_pos_q(x, y)
        if not g.is_passable_block(u.tile_y, u.tile_x, 3):
            continue
        nx, ny = (x + sx) / ONE, (y + sy) / ONE
        if g.is_passable_block(int(ny), int(nx), 3):
            want = (nx, ny)
        elif g.is_passable_block(int(u.y), int(nx), 3):
            want = (nx, u.y)
        elif g.is_passable_block(int(ny), int(u.x), 3):
            want = (u.x, ny)
        else:
            want = (u.x, u.y)
        ui.move(g, u, sx, sy)
        assert (u.x, u.y) == want


# ---------------------------------------------------------------------------
# 8 -- mass
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("impulse_ns", [60.0, 400.0])
def test_displacement_is_monotone_decreasing_in_mass(impulse_ns):
    """Property 8 (§8): one impulse from behind moves a heavier unit no
    farther than a lighter one, over 50 .. 200 kg, for a push that staggers
    every one of them (60 N s) and for one that floors every one (400 N s).
    Breaks if the impulse is not divided by the unit's own mass. (Across
    the fall boundary it is NOT monotone and is not asserted: a 100 N s
    shove floors a 50 kg body, whose prone friction stops it sooner than a
    70 kg one that staggers on its feet through the reaction delay -- 1.49
    against 1.90 tiles; a property of the specified law, reported.)"""
    dists, downs = [], []
    for kg in (50, 70, 80, 100, 140, 200):
        sim = _sim()
        u = _unit(sim, 6, 20)
        u.mass_g = kg * 1000
        x0 = u.x
        j = ui._momentum_q(impulse_ns, 1.0, TILE, 24)
        ui.add(sim.impulse_table, u, j, 0)
        fell = False
        for _ in range(60):
            sim.step()
            fell |= _down(u)
        dists.append(u.x - x0)
        downs.append(fell)
    assert len(set(downs)) == 1, downs               # one regime
    assert all(a >= b for a, b in zip(dists, dists[1:])), dists
    assert dists[0] > dists[-1] > 0


# ---------------------------------------------------------------------------
# 9 -- prone
# ---------------------------------------------------------------------------
def test_a_prone_unit_under_sustained_fire_comes_to_rest_and_stays_down():
    """Property 9 (§5 prone, critique A B4 / M7): a knocked-down unit under
    five in-phase carbines (above the standing in-place capacity, below the
    prone friction on average) slides a bounded distance and is at rest the
    tick after each burst is worked off; it stays down while it slides; and
    in a steady force above the STANDING hold it never gets up (the
    knockdown is refreshed on the standing-equivalent effort). Breaks if a
    prone body keeps a sudden-push branch (flung without end) or gets up
    into an over-hold input."""
    sim = _sim()
    u = _unit(sim, 10, 20, facing=WEST)
    j_round = round(3.6 * 1000 / (TILE * 24) * ONE)
    _kick(sim, u, 3 * 8192, 0)                  # 3 m/s: knocked down
    sim.step()
    assert _down(u)
    for _ in range(30):
        sim.step()
    x0 = u.x
    for tick in range(120):
        if tick % 2 == 0:
            ui.add(sim.impulse_table, u, 5 * j_round, 0)
        sim.step()
        assert _down(u)
        if u.pvx or u.pvy:
            assert any(s.kind == KNOCKED_DOWN for s in u.statuses)
    assert u.x - x0 < 1.0                       # tiles: bounded
    sim.step()
    sim.step()
    assert (u.pvx, u.pvy) == (0, 0)
    f = (sim.impulse_table.cap_q * 105) // 100
    for _ in range(200):
        _kick(sim, u, f, 0)
        sim.step()
        assert _down(u)


# ---------------------------------------------------------------------------
# 10 -- G12, G13
# ---------------------------------------------------------------------------
def test_g12_a_zombie_pushed_off_its_path_never_crosses_a_wall():
    """G12: a walking zombie shoved sideways off its A* path (its path is
    dropped) then walks on: no tick ends with its footprint on a solid tile
    and no walking tick moves it more than one tile per axis (a stale path
    would snap it back across what lies between). Breaks if a displaced
    zombie keeps its path. (A walking step still snaps a pushed zombie
    from its fractional position to the grid: one tile from the tile it
    stands on, an accepted gap until G9p's horde field.)"""
    sim = _sim(h=48, w=60, walls=[(1, 30, 30, 31)])   # a wall it walks round
    _unit(sim, 50, 8, name="M")
    z = _unit(sim, 8, 8, team=ZOMBIE_TEAM, name="Z")
    z.zombie_activated = True
    shoved = False
    for tick in range(400):
        if tick == 40:
            ui.add(sim.impulse_table, z, 0, 3 * sim.impulse_table.shove_q)
            shoved = True
        x0, y0, pushed = z.x, z.y, bool(z.pvx or z.pvy)
        sim.step()
        assert not _solid_under(sim, z), tick
        if shoved and not pushed and not (z.pvx or z.pvy):
            assert abs(z.x - math.floor(x0)) <= 1.0, tick
            assert abs(z.y - math.floor(y0)) <= 1.0, tick
    assert shoved


def test_g13_a_unit_killed_with_pending_impulses_converts_with_none():
    """G13: a marine carrying accumulated impulse, sway, push and effort,
    bitten to death and converted, is a zombie with every impulse state at
    rest. Breaks if death or conversion leaves a stale push."""
    from simulation.ai_zombie import convert_marines_to_zombies
    from simulation.damage import KINETIC, DamagePacket, apply_packet
    sim = _sim()
    u = _unit(sim, 10, 20)
    _kick(sim, u, 4000, 1000)
    sim.step()
    _kick(sim, u, 333, 77)
    assert not ui.at_rest(u)
    apply_packet(u, DamagePacket(amount=10_000, dtype=KINETIC), None,
                 source="melee", mark_killed_by_zombie=True)
    ui.add(sim.impulse_table, u, 999, 999)        # a push onto the corpse
    convert_marines_to_zombies(sim.units)
    assert u.is_zombie and u.alive and ui.at_rest(u)


# ---------------------------------------------------------------------------
# 11 -- the walk's readers
# ---------------------------------------------------------------------------
def test_the_effort_scale_slows_the_walk_from_the_slow_edge_to_half():
    """Property 11 (§5 step 5): the walking scale is ONE up to an effort of
    k_slow / k_hold, half at an effort of ONE (the hold) and above, linear
    and monotone between; ``default_speed`` composes it after the terrain,
    skipped at ONE (G5p-b: it is the only scale member -- the wind's slowing
    is the effort's). Breaks if the scale law or its place in the compose
    changes."""
    from types import SimpleNamespace
    from simulation.movement import FootprintSamples, default_speed, half_up
    t = ui.impulse_table(CFG, TILE)
    u = SimpleNamespace(effort=0)
    scales = []
    for e in range(0, 2 * ONE + 1, 997):
        u.effort = e
        scales.append(ui.effort_scale_q16(t, u))
        if e <= t.effort_edge_q:
            assert scales[-1] == ONE
        if e >= ONE:
            assert scales[-1] == ONE // 2
    assert all(a >= b for a, b in zip(scales, scales[1:]))
    s = FootprintSamples(mobility=[1000] * 9, effort_scale_q16=40000)
    assert default_speed(s, 18) == half_up(18 * ONE, 40000)
    assert not hasattr(s, "wind_scale_q16")
    assert default_speed(FootprintSamples(mobility=[1000] * 9), 18) == 18


def test_every_src_footprint_samples_call_passes_the_walkers_effort():
    """Property 11: every call of ``footprint_samples`` in ``src/`` names
    ``effort_scale_q16`` (the walker's own scale), so every walker -- the
    MOVE_DIR walk, the zombie walk, the WEGO order path, the onephase
    cadence -- is slowed by the same law. Breaks if a new caller forgets
    the effort (found by parsing, never a pinned list)."""
    missing = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and getattr(node.func, "id",
                                getattr(node.func, "attr", "")) ==
                    "footprint_samples"
                    and not any(k.arg == "effort_scale_q16"
                                for k in node.keywords)):
                missing.append(f"{path.name}:{node.lineno}")
    assert not missing, missing


def test_a_staggered_unit_cannot_walk_until_its_reaction_delay_has_run():
    """Property 11 (§9, PLACEHOLDER): a walker whose sway overflows stops
    walking (``walk_allowed`` False, only pv moves it) for the reaction
    delay, then walks again; it can still shoot meanwhile (``can_act``).
    Breaks if the walk ban is lost or never lifts."""
    sim = _sim()
    uid = sim.add_unit(Unit("M", x=10, y=20, team=0))
    u = sim.get_unit(uid)
    place(u, SOUTH)
    for _ in range(3):
        sim.set_move_dir(uid, 0, ONE, 0)
        sim.step()
    _kick(sim, u, 4000, 0)                        # sideways, overflows
    banned = 0
    for _ in range(20):
        sim.set_move_dir(uid, 0, ONE, 0)
        y0 = u.y
        sim.step()
        if not ui.walk_allowed(u):
            banned += 1
            assert u.y == y0 and composed_flags(u).can_act
    assert 1 <= banned <= sim.impulse_table.react_ticks
    y0 = u.y
    sim.set_move_dir(uid, 0, ONE, 0)
    sim.step()
    assert u.y > y0


# ---------------------------------------------------------------------------
# 12 -- the shove
# ---------------------------------------------------------------------------
def _shove_pair(behind=True, target_team=0, shover_team=0):
    """A target at (20, 20) facing east; the shover adjacent behind it
    (west, facing east) or in front of it (east, facing west)."""
    sim = _sim()
    t = _unit(sim, 20, 20, facing=EAST, team=target_team, name="T")
    if behind:
        s = _unit(sim, 17, 20, facing=EAST, team=shover_team, name="S")
    else:
        s = _unit(sim, 23, 20, facing=WEST, team=shover_team, name="S")
    return sim, s, t


def test_a_shove_from_behind_stumbles_a_marine_about_half_a_metre():
    """Property 12 (§6): a 100 N s shove from behind on a 100 kg marine
    staggers it forward (a render-only StaggerEvent), moves it between 0.2
    and 0.8 m over the next second without knocking it down, and its
    ShoveEvent names both units. Breaks if the shove is not an impulse over
    mass through the law, or not along the shover's facing."""
    from simulation.events import ShoveEvent, StaggerEvent
    sim, s, t = _shove_pair()
    x0 = t.x
    sim.shove_intent(s.id)
    sim.step()
    assert any(isinstance(e, ShoveEvent) and (e.unit_id, e.target_id) ==
               (s.id, t.id) for e in sim.tick_events)
    sim.step()
    assert any(isinstance(e, StaggerEvent) and e.unit_id == t.id
               for e in sim.tick_events)
    for _ in range(24):
        sim.step()
        assert not _down(t)
    metres = (t.x - x0) * TILE
    assert 0.2 <= metres <= 0.8, metres
    assert abs(t.y - 20.0) < 1e-9


def test_a_shove_from_the_front_knocks_a_marine_down():
    """Property 12 (§6): the same shove into a marine's face exceeds its
    0.8 m/s backward fall limit and knocks it down. Breaks if the fall
    limit's direction blend reads the wrong way."""
    sim, s, t = _shove_pair(behind=False)
    sim.shove_intent(s.id)
    sim.step()
    sim.step()
    assert _down(t)
    for _ in range(3):
        sim.step()
    assert t.x < 20.0              # pushed west, away from the shover


def test_a_shove_works_on_any_team_and_a_zombie_cannot_shove():
    """Property 12 (RULED: a shove works on any team): a marine shoves a
    zombie as it shoves a teammate; a zombie's shove intent does nothing.
    Breaks if the target filter reads the team, or zombies shove."""
    sim, s, t = _shove_pair(target_team=ZOMBIE_TEAM)
    sim.shove_intent(s.id)
    sim.step()
    assert t.imp_x > 0 or t.swx or t.pvx
    sim2, s2, t2 = _shove_pair(shover_team=ZOMBIE_TEAM)
    sim2.shove_intent(s2.id)
    sim2.step()
    sim2.step()
    assert ui.at_rest(t2) and s2.last_shove_tick == -1


def test_the_shove_picks_the_adjacent_unit_nearest_the_facing():
    """Property 12 (critique B M10): of several adjacent units the shove
    takes the one whose bearing is nearest the shover's facing; one outside
    the cone or beyond reach is never taken; equal bearings go to the lower
    id. Breaks if the selection is by distance, by list order alone, or
    ignores the cone."""
    sim = _sim()
    s = _unit(sim, 20, 20, facing=EAST, name="S")
    a = _unit(sim, 23, 21, name="A")          # 18 deg off the facing
    b = _unit(sim, 22, 23, name="B")          # 56 deg off
    c = _unit(sim, 17, 20, name="C")          # behind: outside the cone
    t = sim.impulse_table
    assert ui.shove_target(t, sim.units, s) is a
    sim.units.remove(a)
    assert ui.shove_target(t, sim.units, s) is b
    sim.units.remove(b)
    assert ui.shove_target(t, sim.units, s) is None
    sim.units[:] = [s, c]
    place(s, WEST)
    assert ui.shove_target(t, sim.units, s) is c
    d = _unit(sim, 17, 20, name="D")          # the same bearing, a higher id
    assert ui.shove_target(t, sim.units, s) is c
    assert d.id > c.id


def test_the_shove_cooldown_runs_and_a_missed_shove_spends_none():
    """Property 12: a second shove inside ``[impulse.shove]
    cooldown_seconds`` does nothing, one after it lands; a shove with no
    target in reach leaves the cooldown unspent. Breaks if the cooldown is
    lost or a miss spends it."""
    sim = _sim()
    s = _unit(sim, 6, 6, facing=EAST, name="S")
    sim.shove_intent(s.id)
    sim.step()
    assert s.last_shove_tick == -1                       # no one to shove
    t = _unit(sim, 9, 6, name="T")
    sim.shove_intent(s.id)
    sim.step()
    first = s.last_shove_tick
    assert first >= 0
    cd = sim.impulse_table.shove_cooldown_ticks
    assert cd == 24
    for _ in range(cd - 2):
        sim.shove_intent(s.id)
        sim.step()
        assert s.last_shove_tick == first
    while sim.tick - first < cd:
        sim.step()
    t.set_pos(s.x + 3.0, t.y)                           # still adjacent
    sim.shove_intent(s.id)
    sim.step()
    assert s.last_shove_tick > first


# ---------------------------------------------------------------------------
# 14 -- digest and events
# ---------------------------------------------------------------------------
def test_the_digest_keys_are_absence_transparent():
    """Property 14: a unit at rest records exactly as before G5p-a2 (no
    impulse key); a non-zero accumulator, sway, push, effort or reaction
    delay adds its own key and nothing else changes. Breaks if a key is
    hashed while zero (every pinned match would move) or a state escapes the
    digest."""
    from field_ab_harness import _unit_record
    sim = _sim()
    u = _unit(sim, 10, 10)
    rest = _unit_record(u)
    names = ("imp_x", "imp_y", "imp_l", "stand_x", "stand_y", "swx", "swy",
             "pvx", "pvy", "effort", "react_ticks")   # #29 P3a: the schema's
    assert not set(names) & set(rest)
    for attr in names:
        key = attr
        setattr(u, attr, 5)
        rec = _unit_record(u)
        assert key in rec
        assert {k: v for k, v in rec.items() if k != key} == rest
        setattr(u, attr, 0)


def test_the_shove_and_stagger_events_are_render_only():
    """Property 14: ``ShoveEvent`` and ``StaggerEvent`` are not synced event
    types (the ReloadEvent precedent). Breaks if either joins the digest."""
    from field_ab_harness import _SYNCED_EVENT_TYPES
    assert "ShoveEvent" not in _SYNCED_EVENT_TYPES
    assert "StaggerEvent" not in _SYNCED_EVENT_TYPES


# ---------------------------------------------------------------------------
# 15 -- G1 / G11, the pinned matches
# ---------------------------------------------------------------------------
#: ship_demo seed 1 and the box at G5p-a2's re-pin 1 (the law built with
#: every input at zero, the walk on the Q16 grid): what the pins held before
#: the hits and the recoil went live (tests/test_zombie_control.py).
#: RE-MEASURED at #183 G5p-b (2026-11-10, the wind and the blast onto the
#: unit impulse law: the dynamic pressure at each footprint is a force
#: q * drag_area through unit_impulse.add_wind; #140's cadence scale and
#: bespoke knockdown DELETED in the same commit).
#: Every input off is now the wind's too (the gate sets wind_push_mult
#: = 0), and #140's law is gone: ship_demo ends win 0 at 266 (was 265),
#: all three digests and the unit stream move; the box is UNCHANGED.
#: SOLE CAUSE, verified on the same build: the pre-G5p-b tree (0b57fc6e)
#: with the push and recoil off AND #140's wind law off (1e15 / 2e15 Pa)
#: gives exactly these values -- with every input off the impulse law
#: still moves nothing.
#: (was field 65fd96904dfa3cc25edc0fc344027876be1b6c11b824c92c6b5930bb3c
#: 9c3499, tick 2ca1d895b11749e6b16f281bdf976379e52077e1d29929284683648f
#: 8369c6a4, units ec0dad03ce0b0ef6389589f50b24728dc074a5f81a9340549b44d8
#: 6e3338fd35)
#: RE-MEASURED at #183 G9p STAGE 1 (2026-11-11, the horde field law,
#: residency P3d: the horde descends an integer BFS field, no A*; see
#: tests/test_zombie_control.py's pins for the law and its proof). This
#: gate's dial is untouched; the horde under it walks the field now, so
#: the match it pins is re-measured on this tree with the gate's own
#: settings -- the law is the whole move.
#: (was win 0 at 266; field
#: 81ba6bb1a51769f09988db7dd5aeb0f06e25e690dc15aade7f6adbe25dc65dc9; tick
#: 52f184d32ae6fa3f1e6a0d28eaacdda9215eba4d06f16358db023fdd91f349ef; units
#: 9c98dc5d7ab477d944d73452a9e2ca8dee2e9d4759cd102a967242319b65566f)
#: RE-MEASURED at #183 G9p STAGE 2 (2026-11-11, the horde behaviours
#: on the field: wander, hearing, fear of fire, door bashing and the
#: sprint, [zombie]; see tests/test_zombie_control.py's pins). This
#: gate's dial is untouched; the match it pins is re-measured on this
#: tree with the gate's own settings -- the behaviours are the whole
#: move (tests/test_horde_behaviours.py: with all of them off, the
#: stage-1 law exactly).
#: (was win 1 at 252; field
#: ceb234133f9f62f8d421b1b68baccac69836be1d99a141c2106592b4e79f87b8; tick
#: 72ae1c16c1615e9573dd8b05fa099cf3753cddbc09443d4fbe74bf620a0c50c0; units
#: b4ac3fd1c5d15dd6ad493cec70476fb1c7c40b0b90adf68c41fdf32fbb5274f7)
_RE_PIN_1_SHIP_DEMO = dict(
    outcome="win", team=0, ticks=240,
    field_digest="187e4a6f710671ff3644584fd51b6278"
                 "eaadb4bac5fbb1225bfeee7b5572de26",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was 46ff80270890878daa488b0c0f03165933ec3a7b6bbdd9055a23b45fb4f3a9fb)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was 46f9509e0b5e5db20e52fa4edb89bbec1bca1b4cf33c0d95a30f96e920ba3c1b)
    tick_digest="cf02f69aefb3d506c436070eebefa4c9"
                "2e7f00016398d2ba28c1668850ea1a6c",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was 93709823f7ef9829b77acc5f8215e5fd8a08721d278a3a9910544cf4c61377e6)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was 478e0b323a021b085799c215d6548a344a35f475aae2aa3d0be0e0d128822104)
    unit_stream="f6453368880f8b5a2cfc3eeb62385e20"
                "d457eeb9460a90ff9e64a59525abc75a")
#: RE-MEASURED at #183 G9p STAGE 1 (2026-11-11, the horde field law,
#: residency P3d: the horde descends an integer BFS field, no A*; see
#: tests/test_zombie_control.py's pins for the law and its proof). This
#: gate's dial is untouched; the horde under it walks the field now, so
#: the match it pins is re-measured on this tree with the gate's own
#: settings -- the law is the whole move.
#: (was win 1 at 130; field
#: a672c7249e18d0eebbb8e398b696bd9c3d34417445f66c3cab498ec7cd783ab1; units
#: f2ba375dd16868470f118df2bbbcab6c151ce28d4e2e108ab5f4433360406497)
#: RE-MEASURED at #183 G9p STAGE 2 (2026-11-11, the horde behaviours
#: on the field: wander, hearing, fear of fire, door bashing and the
#: sprint, [zombie]; see tests/test_zombie_control.py's pins). This
#: gate's dial is untouched; the match it pins is re-measured on this
#: tree with the gate's own settings -- the behaviours are the whole
#: move (tests/test_horde_behaviours.py: with all of them off, the
#: stage-1 law exactly).
#: (was win 1 at 130; field
#: 0fdc79f228cc2e64ca6936cf5a2ec41c52642adfca545cbe51324cfad2e21631; units
#: 05bdbac7c76250b949ef35aef81fdd403bd89af88232eebe061bf753e47b1e3d)
_RE_PIN_1_BOX = dict(
    outcome="win", team=1, ticks=92,
    field_digest="2a9339a53e67d28c29ff513f6942dab9"
                 "1586f5e85c6eb552885554c16ebe4c3b",
    # #29 P3a re-pin (2026-11-12, the unit schema): the unit record is now
    # the schema's (simulation.unit_schema.digest_record, as name|int64 rows
    # under UNIT_SECT_V2): ap / n_orders and the derived tile / alive /
    # faction / offsets left it, the hp hashes as its Q16 count, and every
    # synced column hashes while it differs from its default (the kit, the
    # magazine, the grenades, the cool-downs, the throw / plant in progress,
    # the intents). The GAME is unchanged: tests/_p3a_sole_cause.py gives
    # equal per-tick records and identical field digests on main edc0d2b1 and
    # on this tree. Only the hash's definition moved.
    # (was e082f7dfe103e2fdd594a952da81143ca4202e996238969b22c85eea05b0fd3d)
    # #29 P3c re-pin (2026-11-12, positions to Q16): the unit record hashes a
    # unit's position as its Q16 ints x_q / y_q instead of round(x * 1e9) under
    # x / y. The GAME is unchanged: tests/_p3c_sole_cause.py gives equal per-tick
    # records (x_q := round(x * 65536), every position already on the grid) and
    # identical field digests on main 229f7097 and on this tree; outcome, ticks
    # and field_digest here are unmoved. Only the hash's definition moved.
    # (was ac9bda6c239304078bef33858e2582f883e0684eddd14d541ec7f269e8a88c16)
    unit_stream="85d051afa1acda1b9bc1d3a680e63736"
                "32bfdb9f735b65bf51ec1a9bbf834238")

_IMPULSE_STATE = ("imp_x", "imp_y", "imp_l", "stand_x", "stand_y", "swx",
                  "swy", "pvx", "pvy", "effort", "react_ticks")


class _AtRest:
    """on_tick hook: every unit's impulse state is exactly zero."""

    def __init__(self, stream):
        self.stream = stream
        self.ticks = 0

    def __call__(self, sim):
        self.stream(sim)
        self.ticks += 1
        for u in sim.units:
            for k in _IMPULSE_STATE:
                assert getattr(u, k) == 0, (sim.tick, u.id, k)


def test_g1_with_every_input_off_nothing_is_pushed_and_the_pins_are_re_pin_1s(
        monkeypatch, tmp_path):
    """G1 (replaces the dormancy gate, critique B B4): with
    ``bullet_push_mult = recoil_mult = 0``, ``wind_push_mult = 0`` (G5p-b:
    the wind and the blast's wind) and no shove issued, every unit's
    accumulators, sway, push, effort and reaction delay -- and so the
    impulse part of omega -- are EXACTLY zero on every tick of ship_demo seed
    1 and of the box, asserted per tick (non-vacuous: the bots fire and walk
    throughout), and both matches play re-pin 1 exactly. Breaks if any input
    bypasses its multiplier, or the law moves anything its inputs do not
    control."""
    import level_loader
    from run_match import run_match
    from test_zombie_control import _UnitStream, _box
    monkeypatch.setattr(CFG.impulse, "bullet_push_mult", 0.0)
    monkeypatch.setattr(CFG.impulse, "recoil_mult", 0.0)
    monkeypatch.setattr(CFG.impulse, "wind_push_mult", 0.0)
    hook = _AtRest(_UnitStream())
    from test_bots_v1 import v1_spec
    ship = level_loader.load("ship_demo", str(ROOT / "levels"))
    # G8p: the first bots (bots_v1) played this pin; the layered bots
    # moved the shipped match, so this gate plays it with them.
    o = run_match(ship, v1_spec(ship), on_tick=hook)
    assert hook.ticks > 200
    assert dict(outcome=o.result.outcome, team=o.result.team,
                ticks=o.result.ticks, field_digest=o.field_digest,
                tick_digest=o.tick_digest,
                unit_stream=hook.stream.h.hexdigest()) == _RE_PIN_1_SHIP_DEMO
    hook = _AtRest(_UnitStream())
    o = run_match(_box(tmp_path, "box"), on_tick=hook)
    assert dict(outcome=o.result.outcome, team=o.result.team,
                ticks=o.result.ticks, field_digest=o.field_digest,
                unit_stream=hook.stream.h.hexdigest()) == _RE_PIN_1_BOX


def test_g11_every_position_stays_on_the_q16_grid_in_the_pinned_match():
    """G11: with the shipped dials (hits push, guns recoil, the wind and
    the frags' blast wind push), after every
    tick of ship_demo seed 1 every unit's x and y times 65536 are integers
    -- so P3c's integer positions are a change of storage only. Breaks if
    any position writer leaves the grid (a float walk step, a float push)."""
    import level_loader
    from run_match import run_match
    seen = {"n": 0}

    def hook(sim):
        for u in sim.units:
            assert (u.x * 65536).is_integer() and (u.y * 65536).is_integer()
        seen["n"] += 1
    run_match(level_loader.load("ship_demo", str(ROOT / "levels")),
              on_tick=hook)
    assert seen["n"] > 200
