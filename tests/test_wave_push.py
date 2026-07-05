"""The wave_p impulse-push row + KNOCKED_DOWN trigger (P4) — the gate.

Covers mechanics/05 §1 (the ``wave_p | grad`` coupling row, both outputs) and
mechanics/06 §4 (the knockdown trigger + stability):

  - the nudge: a synthetic wave_p gradient displaces a unit by the EXACT
    door-3 amount (k_push * (-grad/65536) / mass * dt), and the quiet-field
    path is a bit-identical no-op (dormancy — wave-free trajectories keep
    their digest);
  - mass is Newtonian: 2x mass -> exactly half the nudge (bitwise — binary
    scale through a correctly-rounded divide);
  - the sanity cap (push_max_tile_per_tick) clamps a violent gradient;
  - the wall clamp: a unit against a wall never enters it and SLIDES along
    it (per-axis x-then-y), and stays put in a dead-end corner;
  - the knockdown trigger: fires above threshold*stability, not below;
    compares SQUARES (behaviour asserted at the threshold edge); uses the
    UNCAPPED dv; re-knocks REFRESH the get-up timer (P3 stacking); zombies
    (stability 0.9 overlay) and high-stability profiles scale the edge;
  - determinism: the same grenade-blast scenario run twice is bitwise
    identical in positions and statuses;
  - THE RING REGRESSION (mechanics/06 §4's signature property): one blast's
    knockdown radius is measurably WIDER than its meaningful-damage radius
    ("the outer blast zone knocks marines sprawling without killing them").

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_wave_push.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

from config import CFG  # noqa: E402
from simulation.environment import EnvironmentProfile  # noqa: E402
from simulation.exchange import (  # noqa: E402
    COUPLING_TABLE, apply_blast_damage, apply_wave_push, reduce_grad,
)
from simulation.species import ZOMBIE_STABILITY  # noqa: E402
from simulation.status import KNOCKED_DOWN, composed_flags, tick_statuses  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation.weapons import get_tables as weapon_tables  # noqa: E402

Q16 = 65536
TPS = float(CFG.clock.ticks_per_second)
DT = 1.0 / TPS
K_PUSH = float(CFG.exchange.k_push)
CAP = float(CFG.exchange.push_max_tile_per_tick)
THRESHOLD = float(CFG.exchange.knockdown_dv_threshold)
GETUP = int(CFG.exchange.knockdown_getup_ticks)


# ---------------------------------------------------------------------------
# Direct-function fixtures: a stub gmap (wave_p + solid) and helper fields
# ---------------------------------------------------------------------------
def _gmap(h=16, w=16):
    """Minimal exchange-facing gmap stub: an int32 Q16.16 wave_p plane and a
    solid mask with a hull border (like every shipped level)."""
    solid = np.zeros((h, w), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    return SimpleNamespace(wave_p=np.zeros((h, w), dtype=np.int32),
                           solid=solid)


def _linear_x_field(gmap, slope_counts: int):
    """wave_p[y, x] = slope_counts * x — a uniform +x gradient: reduce_grad
    over any full 3x3 footprint is exactly (2 * slope_counts, 0)."""
    h, w = gmap.wave_p.shape
    xs = np.arange(w, dtype=np.int64) * int(slope_counts)
    gmap.wave_p[:] = np.broadcast_to(xs, (h, w)).astype(np.int32)


def _marine(x=6.0, y=6.0):
    u = Unit("M", x=x, y=y, team=0)
    u.id = 1
    return u


def _zombie(x=6.0, y=6.0):
    u = Unit("Z", x=x, y=y, team=1)
    u.id = 2
    return u


def _expected_dv(grad_counts: int, mass: float) -> float:
    """The row's exact door-3 chain for one axis (mirrors apply_wave_push)."""
    return K_PUSH * (-grad_counts / 65536.0) / mass


# ---------------------------------------------------------------------------
# The nudge — exactness, dormancy, mass, cap
# ---------------------------------------------------------------------------
def test_gradient_pushes_exact_amount():
    g = _gmap()
    slope = 30000                       # gx = 60000 counts -> dv ~ -4.58 t/s
    _linear_x_field(g, slope)
    u = _marine(x=6.0, y=6.0)
    apply_wave_push([u], g, TPS)

    gx, gy = 2 * slope, 0
    dvx = _expected_dv(gx, float(u.mass))
    assert abs(dvx) < THRESHOLD        # this test is about the nudge only
    assert u.x == 6.0 + dvx * DT       # EXACT: same pure +-*/ chain
    assert u.y == 6.0                  # no y gradient -> no y motion
    assert u.statuses == []            # below threshold: no knockdown


def test_quiet_field_is_bitwise_noop():
    """Dormancy: zero wave_p leaves position bitwise untouched and applies
    nothing — wave-free trajectories keep their digest (the P4 wiring
    contract for every existing non-wave test/scenario)."""
    g = _gmap()
    u = _marine(x=5.25, y=7.75)
    x0, y0 = u.x, u.y
    apply_wave_push([u], g, TPS)
    assert (u.x, u.y) == (x0, y0)
    assert u.statuses == []


def test_dead_units_are_not_pushed():
    g = _gmap()
    _linear_x_field(g, 30000)
    u = _marine()
    u.alive = False
    x0 = u.x
    apply_wave_push([u], g, TPS)
    assert u.x == x0 and u.statuses == []


def test_mass_dependence_exact_halving():
    """Newtonian response: 2x mass -> exactly half the nudge. Bitwise exact:
    the divide-by-mass is correctly rounded and 160 = 2 * 80 is a binary
    scale, so round(a/160) == round(a/80)/2 on every IEEE-754 machine."""
    g = _gmap()
    _linear_x_field(g, 30000)
    light = _marine(x=6.0, y=6.0)
    heavy = _marine(x=6.0, y=10.0)
    light.mass = 80.0
    heavy.mass = 160.0
    apply_wave_push([light, heavy], g, TPS)
    light_dx = light.x - 6.0
    heavy_dx = heavy.x - 6.0
    assert light_dx != 0.0
    assert heavy_dx == light_dx / 2.0   # exact halving, not approx


def test_push_cap_clamps_violent_gradient():
    g = _gmap()
    _linear_x_field(g, -600000)         # gx = -1.2e6 counts -> dv +91.6 t/s
    u = _marine(x=6.0, y=6.0)
    apply_wave_push([u], g, TPS)
    gx = 2 * -600000
    assert abs(_expected_dv(gx, float(u.mass)) * DT) > CAP  # cap engaged
    assert u.x == 6.0 + CAP             # clamped EXACTLY to the cap
    # a violent gradient is far past the knockdown edge too
    assert [s.kind for s in u.statuses] == [KNOCKED_DOWN]


# ---------------------------------------------------------------------------
# Wall clamp — never into solid, slide along, corner pin
# ---------------------------------------------------------------------------
def test_wall_clamp_blocks_axis_and_slides_along():
    """Unit anchored at x=11 (footprint 11..13) against the x=15... border of
    a 16-wide map (border col 15 solid): a diagonal +x/+y push must drop the
    x axis (footprint 12..14 would still be free; 13..15 hits the wall) and
    keep the y slide. Use a cap-exceeding gradient so the attempted x move
    crosses into the border column."""
    g = _gmap(h=16, w=16)
    h, w = g.wave_p.shape
    # wave_p = big * (x + y): gradient pushes toward -x/-y; use negative slope
    # to push +x/+y instead.
    xs = np.arange(w, dtype=np.int64)
    ys = np.arange(h, dtype=np.int64)
    plane = -(xs[None, :] + ys[:, None]) * 600000
    g.wave_p[:] = plane.astype(np.int32)

    u = _marine(x=11.6, y=6.0)          # footprint x 11..13; wall col at 15
    apply_wave_push([u], g, TPS)
    # x: 11.6 + 0.5 (cap) = 12.1 -> anchor 12, footprint 12..14: FREE, moves.
    assert u.x == 11.6 + CAP
    assert u.y == 6.0 + CAP             # y slides too (row 6.5+3 < border 15)

    # Second shove: x 12.1 + 0.5 = 12.6 -> anchor 12: free. Third: 13.1 ->
    # anchor 13, footprint 13..15 hits the border column -> x PINNED; y still
    # slides (until its own border).
    apply_wave_push([u], g, TPS)
    assert u.x == 12.1 + CAP
    x_pinned = u.x
    apply_wave_push([u], g, TPS)
    assert u.x == x_pinned              # never enters the wall
    assert u.y == 6.0 + 3 * CAP         # the free axis keeps sliding


def test_corner_pin_blocks_both_axes():
    """Sub-tile motion inside the current free tiles is legal (the clamp is
    tile-resolution, like the fields); pinning bites at the tile crossing:
    from x=12.6 a +0.5 push would land anchor 13 -> footprint 13..15 -> the
    border column/row -> BOTH axes drop, the unit stays put."""
    g = _gmap(h=16, w=16)
    xs = np.arange(16, dtype=np.int64)
    plane = -(xs[None, :] + xs[:, None]) * 600000
    g.wave_p[:] = plane.astype(np.int32)
    u = _marine(x=12.6, y=12.6)         # next +cap crosses into anchor 13
    apply_wave_push([u], g, TPS)
    assert (u.x, u.y) == (12.6, 12.6)   # pinned in the corner: nothing moves
    # ... but the knockdown STILL fired: dv is physical, walls don't shield
    assert [s.kind for s in u.statuses] == [KNOCKED_DOWN]


# ---------------------------------------------------------------------------
# Knockdown trigger — threshold edge, stability, refresh
# ---------------------------------------------------------------------------
def _slope_for_dv(dv_target: float, mass: float = 80.0) -> int:
    """Integer x-slope (counts/tile) whose 3x3 gradient produces ~dv_target
    tiles/s on `mass` (sign: positive dv pushes -x for a positive slope)."""
    #   dv = K_PUSH * (2*slope/65536) / mass
    return int(round(dv_target * mass * 65536.0 / (2.0 * K_PUSH)))


def test_knockdown_triggers_above_threshold_not_below():
    for factor, expect_down in ((0.9, False), (1.1, True)):
        g = _gmap()
        _linear_x_field(g, _slope_for_dv(THRESHOLD * factor))
        u = _marine()
        apply_wave_push([u], g, TPS)
        kinds = [s.kind for s in u.statuses]
        assert (KNOCKED_DOWN in kinds) == expect_down, \
            f"dv = {factor} x threshold should give knocked={expect_down}"
        if expect_down:
            st = u.statuses[0]
            assert st.remaining_ticks == GETUP
            assert st.magnitude_q16 == 0
            assert st.source_id is None
            assert composed_flags(u).is_prone
            assert not composed_flags(u).can_move


def test_knockdown_uses_uncapped_dv():
    """The displacement cap is a motion sanity rail, NOT part of the physics:
    a dv far past the cap still knocks down (already exercised by the corner
    test; here with the displacement visibly capped)."""
    g = _gmap()
    _linear_x_field(g, _slope_for_dv(THRESHOLD * 20.0))
    u = _marine(x=6.0, y=6.0)
    apply_wave_push([u], g, TPS)
    assert u.x == 6.0 - CAP             # motion capped...
    assert [s.kind for s in u.statuses] == [KNOCKED_DOWN]   # ...trigger not


def test_stability_scales_the_edge():
    """threshold * stability is the edge: a dv between the zombie's edge
    (x0.9) and the marine's (x1.0) fells only the zombie; a high-stability
    profile (1.5) stands through a dv that fells a marine."""
    dv_mid = THRESHOLD * 0.95           # between 0.9 and 1.0 x threshold
    g = _gmap()
    _linear_x_field(g, _slope_for_dv(dv_mid))
    m, z = _marine(x=4.0, y=4.0), _zombie(x=4.0, y=9.0)
    assert ZOMBIE_STABILITY < 0.95 < 1.0
    apply_wave_push([m, z], g, TPS)
    assert m.statuses == []                                # marine stands
    assert [s.kind for s in z.statuses] == [KNOCKED_DOWN]  # shambler topples

    g2 = _gmap()
    _linear_x_field(g2, _slope_for_dv(THRESHOLD * 1.2))
    braced = _marine(x=4.0, y=4.0)
    braced.environment = EnvironmentProfile(stability=1.5)  # door-2 profile
    normal = _marine(x=4.0, y=9.0)
    apply_wave_push([braced, normal], g2, TPS)
    assert braced.statuses == []                            # 1.2 < 1.5 edge
    assert [s.kind for s in normal.statuses] == [KNOCKED_DOWN]


def test_reknock_refreshes_getup_timer():
    """P3 refresh stacking through the trigger: a second blast while prone
    resets the timer on the SAME instance (no second entry)."""
    g = _gmap()
    _linear_x_field(g, _slope_for_dv(THRESHOLD * 1.5))
    u = _marine()
    apply_wave_push([u], g, TPS)
    assert len(u.statuses) == 1 and u.statuses[0].remaining_ticks == GETUP
    for _ in range(5):                  # time passes while prone
        tick_statuses([u])
    assert u.statuses[0].remaining_ticks == GETUP - 5
    apply_wave_push([u], g, TPS)        # re-knocked by a second wave
    assert len(u.statuses) == 1         # refresh, not stack
    assert u.statuses[0].remaining_ticks == GETUP


# ---------------------------------------------------------------------------
# Full-sim: determinism + THE RING regression
# ---------------------------------------------------------------------------
def _blast_room(h=48, w=48):
    from level_loader import LevelData
    tm = np.ones((h, w), dtype=np.int32)    # hull border
    tm[1:h - 1, 1:w - 1] = 4                # interior air
    return LevelData(name="push_ring", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _grenade_sim(unit_center_dist: int, n_ticks: int = 40, seed: int = 777):
    """A grenade-equivalent detonation at (24, 12) with one marine whose
    footprint CENTER sits `unit_center_dist` tiles along +x. Returns the sim
    + unit after `n_ticks` (the wave deposit fires on tick 1, exactly like a
    fuse-out); records whether the unit was ever knocked down."""
    import breach_physics as bp
    from simulation import Simulation
    from simulation.physics import apply_explosion

    sim = Simulation(_blast_room(), seed=seed, breach_physics=bp,
                     enable_recorder=False)
    cy, cx = 24, 12
    u = Unit("P", x=cx + unit_center_dist - 1, y=cy - 1, team=0)
    sim.add_unit(u)
    sim.set_paused(False)
    ever_down = False
    trace = []
    for t in range(n_ticks):
        if t == 1:
            # W1 re-home: the grenade blast numbers live on the frag_standard
            # payload row (same 5 / 10.0 / 200 literals as the old
            # CFG.weapons.grenade.* keys).
            frag = weapon_tables().payload_for_ammo("grenade_frag")
            apply_explosion(sim.gmap, sim.edit_queue, cy, cx,
                            frag.radius,
                            frag.pressure,
                            frag.wall_damage)
        sim.set_paused(False)
        sim.step()
        ever_down = ever_down or any(
            s.kind == KNOCKED_DOWN for s in u.statuses)
        trace.append((u.x, u.y, tuple((s.kind, s.remaining_ticks)
                                      for s in u.statuses)))
    return u, ever_down, trace


def test_full_sim_determinism_run_twice_identical():
    """The whole push+knockdown pipeline through a real blast, twice — every
    per-tick position and status bitwise identical."""
    _, down_a, trace_a = _grenade_sim(6, n_ticks=24)
    _, down_b, trace_b = _grenade_sim(6, n_ticks=24)
    assert down_a and down_b            # d=6 is well inside the knockdown ring
    assert trace_a == trace_b           # bitwise: floats compare exactly


def test_blast_visibly_shoves_the_unit():
    """The nudge is not cosmetic: over the wave passage the unit's position
    measurably moves (the buffet), and it never enters a wall."""
    u, _down, trace = _grenade_sim(6, n_ticks=24)
    xs = [x for (x, y, s) in trace]
    excursion = max(abs(x - xs[0]) for x in xs)
    assert excursion > 0.1              # visible at 48 px/tile (~5+ px)


def test_knockdown_ring_wider_than_damage_ring():
    """mechanics/06 §4, the signature property, as a regression test:

        knockdown_radius  >  meaningful-damage_radius

    Damage half (geometric — apply_blast_damage IS the shipped damage row):
    scan center distances for the farthest unit taking any damage (the row's
    own blast_damage_threshold already defines 'meaningful'). Knockdown half
    (field physics): a marine at d=7 — OUTSIDE the whole damage disc — is
    still bowled over by the wave; one at d=16 stands (the ring is finite).
    Margins at the calibrated values: dv(7) ~ 8.0 vs threshold 6.0 (+33%),
    dv(16) ~ 3.3 (-45%) — robust to tuning drift, loud on regression."""
    # --- damage radius: geometric, no sim needed --------------------------
    frag = weapon_tables().payload_for_ammo("grenade_frag")
    radius = frag.radius
    max_damage = frag.unit_damage
    fy = fx = 100.0                     # abstract plane, no walls involved
    probes = []
    for d in range(1, 2 * radius + 1):
        p = Unit(f"D{d}", x=fx + d - 1, y=fy - 1, team=0)
        p.id = 10 + d
        probes.append((d, p, p.current_hp))
    apply_blast_damage([p for (_d, p, _h) in probes], fx, fy,
                       radius, max_damage)
    damage_radius = max(
        (d for (d, p, hp0) in probes if p.current_hp < hp0), default=0)
    assert damage_radius > 0            # the blast does damage SOMEWHERE

    # --- knockdown radius: the wave does the work -------------------------
    _, down_at_7, _ = _grenade_sim(7)
    _, down_at_16, _ = _grenade_sim(16)
    assert down_at_7, "a marine 7 tiles out (outside the damage disc) " \
                      "must be knocked sprawling"
    assert not down_at_16, "the knockdown ring must be finite"
    knockdown_radius = 7                # the asserted-lower-bound radius

    print(f"\n[ring] meaningful-damage radius = {damage_radius} tiles; "
          f"knockdown radius >= {knockdown_radius} tiles "
          f"(calibrated edge ~9-10; d=16 stands)")
    assert knockdown_radius > damage_radius


# ---------------------------------------------------------------------------
# Table registration
# ---------------------------------------------------------------------------
def test_push_row_registered_in_coupling_table():
    rows = [(r.field, r.reduction, r.response) for r in COUPLING_TABLE]
    assert ("wave_p", "grad", apply_wave_push) in rows
    # table order: heat, blast, push (the chapter's row order — P0)
    assert rows[-1][2] is apply_wave_push


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
