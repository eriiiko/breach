"""The wind row at Simulation.step 9c2 -- the gate (#140's read, G5p-b's law).

The retired ``wave_p | grad`` impulse push lived here; #140 made the 9c2 row
the ``wind | max`` row of a momentum-flux threshold law (a cadence scale and a
bespoke knockdown, docs/wind_on_units_design_2026-10-31.md). Since #183 G5p-b
the row is ``exchange.apply_wind_force``: the dynamic pressure ``q = 1/2 rho
v^2`` at the footprint (THE read, ``GameMap.footprint_wind``) is a FORCE on
the unit impulse law -- one tick of impulse ``q * drag_area`` along the wind
of the tile it was read on, through ``unit_impulse.add_wind`` -- and the law's
own outcomes replace #140's (docs/unit_impulse_design_2026-11-10.md §4, §5,
§6, §7). A blast's wind is wind.

Each test names the property it protects and the change that breaks it.
(The walkers' cadence callers live in test_footprint_samples_builder.py; the
#140 doorway reproduction in test_wind_doorway_e2e.py; the law's own gates,
any input, in test_unit_impulse.py.)

A STEADY wind is set at the body by replacing the map's read on the
instance (``gmap.footprint_wind``) -- the conductor's 9c2 row then hands the
law exactly that load every tick, through the real Simulation.

Run:
    <breach py> -m pytest tests/test_wave_push.py -q
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from simulation import exchange  # noqa: E402
from simulation import unit_impulse as ui  # noqa: E402
from simulation.atmosphere_fixed import FP_ONE  # noqa: E402
from simulation.exchange import COUPLING_TABLE, apply_wind_force  # noqa: E402
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gas_fixed import BULK_ONE  # noqa: E402
from simulation.gases import INERT_N2, O2  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.status import KNOCKED_DOWN, composed_flags  # noqa: E402
from simulation.turning import PI_Q16, place  # noqa: E402
from simulation.unit import Unit  # noqa: E402

#: The playground's tile (0.333 m), the scale the shipped numbers are at.
TILE = 1.0 / 3.0
EAST, WEST = 0, PI_Q16
TPS = int(CFG.clock.ticks_per_second)


def _room_level(h=16, w=16, tile=1.0, walls=()):
    from level_loader import LevelData
    tm = np.full((h, w), MAT_AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = MAT_HULL
    for (y0, y1, x0, x1) in walls:
        tm[y0:y1, x0:x1] = MAT_HULL
    return LevelData(name="wind_room", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=tile, diffuse_path=Path("."))


def _room_map(h=16, w=16):
    return GameMap(_room_level(h, w))


def _set_air(g, n_raw, vx_raw, vy_raw=0):
    """Fixture write of the bulk N and the wind planes over the whole open
    room -- a pure-read test of the row; nothing is stepped afterwards."""
    open_ = ~g.solid
    g.gas[O2][open_] = n_raw // 5
    g.gas[INERT_N2][open_] = n_raw - n_raw // 5
    g.wind_x[open_] = vx_raw
    g.wind_y[open_] = vy_raw


def _sim(h=40, w=120, tile=TILE, walls=()):
    import breach_physics as bp
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    sim = Simulation(_room_level(h, w, tile, walls), seed=1,
                     breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _unit(sim, x, y, facing=EAST, team=0, name="M"):
    u = sim.get_unit(sim.add_unit(Unit(name, x=x, y=y, team=team)))
    place(u, facing)
    return u


def _hold_n(sim, u):
    """One standing hold of ``u`` in newtons (k_hold of its body weight)."""
    s = sim.impulse_table.settings
    return s.k_hold * (u.mass_g / 1000.0) * s.gravity_mps2


def _q_for(sim, newtons):
    """The dynamic pressure that puts ``newtons`` on a standing body."""
    return newtons / sim.impulse_table.settings.drag_area_m2


def _steady(sim, q_pa, vx=1, vy=0):
    """Replace the map's read on the instance: every footprint now reads a
    steady wind of dynamic pressure ``q_pa`` along ``(vx, vy)``. Returns a
    switch (call with False to calm the air)."""
    w = int(q_pa * float(1 << 33) / sim.impulse_table.rho_air)
    state = {"on": True}
    sim.gmap.footprint_wind = (
        lambda fy, fx, fp=3: (w, vx * 20 * FP_ONE, vy * 20 * FP_ONE)
        if state["on"] else (0, 0, 0))
    return lambda on: state.__setitem__("on", on)


def _down(u):
    return composed_flags(u).is_prone


# ---------------------------------------------------------------------------
# The row hands the law an impulse, and nothing else
# ---------------------------------------------------------------------------
def test_the_row_hands_an_impulse_and_writes_no_position_status_or_speed():
    """Property: in a wind far above the hold the 9c2 row changes only the
    unit's impulse accumulators (and their books) -- along the wind -- never
    its x, y, statuses or effort; whatever the unit does next is the law's
    (slot 1a). Breaks if the row grows a displacement, a status or a speed
    write of its own (the #140 knockdown or cadence scale coming back)."""
    g = _room_map()
    _set_air(g, BULK_ONE, 60 * FP_ONE, -20 * FP_ONE)
    u = Unit("M", x=5.25, y=7.75, team=0)
    u.id = 1
    apply_wind_force([u], g)
    assert (u.x, u.y) == (5.25, 7.75)
    assert u.statuses == [] and u.effort == 0
    assert (u.pvx, u.pvy, u.swx, u.swy) == (0, 0, 0, 0)
    assert u.imp_x > 0 and u.imp_y < 0               # along the wind
    assert abs(u.imp_x / u.imp_y + 3.0) < 1e-3
    (in_x, _), (in_y, _) = ui.books_identity(u)
    assert (in_x, in_y) == (u.imp_x, u.imp_y)


def test_dead_units_and_still_air_are_untouched():
    """A corpse in a gale gets nothing (the heat row killed it first); a
    living unit in still air gets nothing (W = 0). Breaks if the row reads a
    corpse or adds a zero-load impulse."""
    g = _room_map()
    _set_air(g, BULK_ONE, 60 * FP_ONE)
    u = Unit("M", x=6.0, y=6.0, team=0)
    u.id, u.alive = 1, False
    apply_wind_force([u], g)
    assert ui.at_rest(u)
    _set_air(g, BULK_ONE, 0)
    u.alive = True
    apply_wind_force([u], g)
    assert ui.at_rest(u)


def test_density_is_in_the_law():
    """Property: at the same wind, twice the tile's bulk N gives twice the
    wind load exactly and twice the impulse to within the one floor of its
    pricing. Breaks if density leaves the law (a speed threshold)."""
    g = _room_map()
    t = ui.table_for(g)
    _set_air(g, 2 * BULK_ONE, 12 * FP_ONE)
    w2 = g.footprint_wind_load(5, 5, 3)
    _set_air(g, BULK_ONE, 12 * FP_ONE)
    w1 = g.footprint_wind_load(5, 5, 3)
    assert w2 == 2 * w1 > 0
    j1, j2 = ui.wind_impulse_q(t, w1), ui.wind_impulse_q(t, w2)
    assert j1 > 0 and 0 <= j2 - 2 * j1 <= 1


def test_the_impulse_is_q_times_the_drag_area_for_one_tick():
    """Property: one tick of wind is ``q * drag_area / tps`` N s in the law's
    currency (Q16 gram-tiles per tick), standing and prone, to within the
    table's quantization of the price (half a count of its Q16 price, 5e-6
    relative standing, 2e-5 prone). Breaks if the price is read in another unit, quantized
    twice, or loses the area or the clock."""
    t = ui.impulse_table(CFG, TILE)
    s = t.settings
    for q in (39.0, 245.0, 2500.0):
        w = int(q * float(1 << 33) / t.rho_air)
        for prone, area in ((False, s.drag_area_m2),
                            (True, s.drag_area_prone_m2)):
            want = (q * area / TPS) * 1000.0 / (TILE * TPS) * FP_ONE
            got = ui.wind_impulse_q(t, w, prone)
            k = t.wind_k_prone_q if prone else t.wind_k_q
            assert abs(got - want) <= (0.5 / k + 1e-6) * want + 2, (q, prone)


def test_the_air_density_is_the_maps_materials_air_row():
    """Property: the table prices q with the map's [materials.air] density
    (one owner): the Simulation's table carries that row, and a map with an
    air twice as dense gets twice the price per wind load. Breaks if a
    second density dial appears or the row is not read."""
    sim = _sim(h=12, w=12)
    assert sim.impulse_table.rho_air == float(
        sim.gmap.materials.density[MAT_AIR])
    thin = ui.impulse_table(CFG, TILE, rho_air=sim.impulse_table.rho_air)
    dense = ui.impulse_table(CFG, TILE,
                             rho_air=2 * sim.impulse_table.rho_air)
    assert abs(dense.wind_k_q - 2 * thin.wind_k_q) <= 1


@pytest.mark.parametrize("key", ["drag_area_m2", "drag_area_prone_m2",
                                 "wind_push_mult"])
def test_the_door_refuses_a_missing_or_mistyped_wind_dial(monkeypatch, key):
    """Property: the wind's [impulse] dials are required and typed, refused
    by name (an area must be > 0; the multiplier >= 0). Breaks if the door
    silently defaults one."""
    monkeypatch.setattr(CFG.impulse, key, "big")
    with pytest.raises(ValueError, match=key):
        ui.impulse_settings(CFG)
    monkeypatch.setattr(CFG.impulse, key, -1.0)
    with pytest.raises(ValueError, match=key):
        ui.impulse_settings(CFG)
    monkeypatch.delattr(CFG.impulse, key)
    with pytest.raises(ValueError, match=key):
        ui.impulse_settings(CFG)


def test_the_table_refuses_a_non_positive_air_density():
    for rho in (0.0, -1.2, float("nan")):
        with pytest.raises(ValueError, match="air"):
            ui.impulse_table(CFG, TILE, rho_air=rho)


# ---------------------------------------------------------------------------
# The bands: #140's, restated as force over body weight
# ---------------------------------------------------------------------------
def _settled(sim, u, q_pa, ticks=72):
    _steady(sim, q_pa)
    x0, y0 = u.x, u.y
    for _ in range(ticks):
        sim.step()
        assert (u.x, u.y) == (x0, y0) and (u.pvx, u.pvy) == (0, 0)
        assert not _down(u)
    return ui.effort_scale_q16(sim.impulse_table, u)


def test_the_reference_marine_keeps_the_140_bands():
    """Property (§6, §7 gate "equal bands for the reference marine"): a
    100 kg marine standing in a steady wind below its hold is never moved
    and never floored, and its walking scale settles on #140's float law,
    ``s = 1 - 0.5 (q - 39) / (245 - 39)`` clamped at 1 (each edge as force
    over body weight: k_slow, k_hold), to within the effort's integer grain
    (0.4 %): exactly 1 below the slow edge, monotone in q. Breaks if the
    wind's price, the hold or the slow edge moves the bands, or the soft
    wind displaces."""
    sim = _sim(h=12, w=12)
    u = _unit(sim, 4, 4)
    assert u.mass_g == 100_000
    s = sim.impulse_table.settings
    q_slow = _q_for(sim, s.k_slow * 100.0 * s.gravity_mps2)
    q_fall = _q_for(sim, _hold_n(sim, u))
    assert abs(q_slow - 39.0) < 0.5 and abs(q_fall - 245.0) < 0.5
    prev = ONE = FP_ONE
    for q in (0.5 * q_slow, 0.99 * q_slow, 60.0, 120.0, 180.0, 240.0):
        sc = _settled(sim, u, q)
        want = 1.0 if q <= q_slow else \
            1.0 - 0.5 * (q - q_slow) / (q_fall - q_slow)
        assert abs(sc / ONE - want) <= 0.004, (q, sc / ONE, want)
        if q < q_slow:
            assert sc == ONE
        assert sc <= prev
        prev = sc


def test_a_lighter_body_is_slowed_sooner():
    """Property (§6): the same wind does more to a lighter body -- an 80 kg
    zombie is slowed at a q that leaves the 100 kg marine at full speed.
    Breaks if the wind's effect stops going through the body's mass."""
    sim = _sim(h=12, w=24)
    m = _unit(sim, 4, 4, name="M")
    z = _unit(sim, 14, 4, team=ZOMBIE_TEAM, name="Z")
    assert z.mass_g < m.mass_g
    _steady(sim, 37.0)                    # under the marine's slow edge
    for _ in range(72):
        sim.step()
    t = sim.impulse_table
    assert ui.effort_scale_q16(t, m) == FP_ONE
    assert ui.effort_scale_q16(t, z) < FP_ONE


@pytest.mark.parametrize("facing", [EAST, WEST])
def test_below_the_hold_a_steady_wind_never_moves_a_unit(facing):
    """Property (RULED: soft wind never displaces): 10 s of wind at 0.99 of
    the hold, with the wind or into it, leave the position and pv exactly
    unchanged and the unit standing. Breaks if the resist step leaves part
    of a sub-hold force over, or the row pushes outside the law."""
    sim = _sim(h=12, w=12)
    u = _unit(sim, 4, 4, facing=facing)
    _settled(sim, u, 0.99 * _q_for(sim, _hold_n(sim, u)), ticks=10 * TPS)


@pytest.mark.parametrize("team, frac, down", [
    (0, 0.99, False), (0, 1.01, True),
    (ZOMBIE_TEAM, 0.80, False), (ZOMBIE_TEAM, 0.82, True)])
def test_the_knockdown_edge_is_the_hold_times_stability_squared(team, frac,
                                                                down):
    """Property (#140's pinned edge, now the law's effort compare): a steady
    wind floors a marine (stability 1.0) at its hold and not below; a zombie
    (stability 0.9) at 0.81 of ITS hold and not at 0.80 -- within 2 s, and
    only by the excess (a few centimetres at most). Breaks if stability
    stops scaling the knockdown wind speed or the effort cannot reach its
    input."""
    sim = _sim(h=12, w=12)
    u = _unit(sim, 4, 4, team=team)
    _steady(sim, frac * _q_for(sim, _hold_n(sim, u)))
    x0 = u.x
    for _ in range(2 * TPS):
        sim.step()
    assert _down(u) == down, (team, frac)
    assert abs(u.x - x0) * TILE < 0.05


def test_a_braced_profile_is_floored_later_but_floored():
    """Property (§8: ANY stability falls in a bounded time): a stability-1.5
    profile stands through a wind that floors the marine at once, and a
    steady force over its hold still floors it within the law's bound --
    after sliding at the excess (the run-2 finding, design As built G5p-a2:
    no shipped unit is above 1.0). Breaks if stability stops protecting or
    an over-hold force can hold a unit up for ever."""
    sim = _sim(h=40, w=160)
    m = _unit(sim, 10, 4, name="M")
    b = _unit(sim, 10, 24, name="B")
    b.environment = dataclasses.replace(b.environment, stability=1.5)
    _steady(sim, 1.5 * _q_for(sim, _hold_n(sim, m)))
    for tick in range(20 * TPS):
        sim.step()
        if tick == 2 * TPS:
            assert _down(m) and not _down(b)
        if _down(b):
            break
    assert _down(b)


def test_a_prone_unit_stays_down_in_a_gale_and_gets_up_after_it():
    """Property (critique A M7; §5 prone): in a steady wind at 1.5 holds a
    marine is floored, and stays down -- never once up -- for three get-up
    times, though its prone body takes six times less wind (it is judged by
    the wind it WOULD take standing); it does not slide (the prone friction
    holds); once the wind drops it gets up. Breaks if the prone refresh
    reads the prone area (the 36-down / 13-up cycle), or the friction is
    lost."""
    sim = _sim(h=12, w=24)
    u = _unit(sim, 8, 4)
    switch = _steady(sim, 1.5 * _q_for(sim, _hold_n(sim, u)))
    getup = sim.impulse_table.getup_ticks
    for _ in range(TPS):
        sim.step()
    assert _down(u)
    x_rest = None
    for tick in range(3 * getup):
        sim.step()
        assert _down(u), tick
        if tick == 12:
            x_rest = u.x
        if x_rest is not None:
            assert u.x == x_rest and (u.pvx, u.pvy) == (0, 0)
    switch(False)
    for _ in range(getup + 2 * TPS):
        sim.step()
    assert not _down(u)


def test_a_prone_body_slides_only_under_a_much_larger_wind():
    """Property (§6, the design's intent): a prone marine is held by its
    prone friction until the wind's force on its prone area exceeds it --
    ``q * A_prone > mu_slide m g``, about ten times the wind that floors it
    standing; above that it slides by the excess. Breaks if the prone area
    or the prone friction stops being the prone law, or the books stop
    closing under a steady force (G4)."""
    s = CFG.impulse
    q_fall = s.k_hold * 100.0 * s.gravity_mps2 / s.drag_area_m2
    q_slide = s.mu_slide * 100.0 * s.gravity_mps2 / s.drag_area_prone_m2
    assert q_slide > 8 * q_fall
    sim = _sim(h=12, w=120)
    u = _unit(sim, 20, 4)
    _steady(sim, 1.5 * q_fall)               # floored, comes to rest
    for _ in range(TPS):
        sim.step()
    assert _down(u) and (u.pvx, u.pvy) == (0, 0)
    moved = {}
    for frac in (0.9, 1.2):
        _steady(sim, frac * q_slide)
        x1 = u.x
        for _ in range(2 * TPS):
            sim.step()
            assert _down(u)
            (ix, ox), (iy, oy) = ui.books_identity(u)
            assert (ix, iy) == (ox, oy)          # G4: a sliding body too
        moved[frac] = u.x - x1
    assert moved[0.9] == 0.0
    assert moved[1.2] > 0.5


def test_a_shove_in_a_near_hold_wind_does_not_slide_a_unit_away():
    """Property (critique A B1, the #140 failure in impulse form): a 100 N s
    shove from behind on a marine with its back to a wind at 0.99 of its
    hold moves it no farther than the same shove in still air (here it
    falls -- the margin left cannot work the push off -- and stops), and it
    comes to rest within two seconds. Breaks if the reaction delay or the
    near-hold wind switch the footing off (the 23 m slide of draft 1)."""
    out = {}
    for name, q_frac in (("calm", 0.0), ("wind", 0.99)):
        sim = _sim(h=12, w=60)
        u = _unit(sim, 10, 4)
        if q_frac:
            _steady(sim, q_frac * _q_for(sim, _hold_n(sim, u)))
        for _ in range(TPS):
            sim.step()
        x0 = u.x
        j = ui._momentum_q(100.0, 1.0, TILE, TPS)
        ui.add(sim.impulse_table, u, j, 0)
        for _ in range(2 * TPS):
            sim.step()
        x2 = u.x
        for _ in range(TPS):
            sim.step()
        assert u.x == x2, name                        # at rest
        out[name] = u.x - x0
    assert 0.0 < out["wind"] <= out["calm"]


# ---------------------------------------------------------------------------
# The blast is wind
# ---------------------------------------------------------------------------
def _frag(sim, cy, cx):
    from simulation.payloads import execute_payload
    from simulation.weapons import get_tables
    frag = get_tables().payload_for_ammo("grenade_frag")
    assert frag.strength > 0                     # THE generic explosion
    return execute_payload(sim.gmap, sim.edit_queue, sim.units, cy, cx,
                           frag, sim.rng)


def _blast_scene(mult, n_ticks=48):
    """A shipped frag at (20, 20) of an open room: a marine 3 tiles east of
    it (1 m), one 3 tiles west BEHIND a wall, one 30 tiles south (10 m)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CFG.impulse, "wind_push_mult", mult)
        sim = _sim(h=60, w=48, walls=[(10, 30, 16, 18)])
    near = _unit(sim, 21.5 - 1, 18.5, name="near")       # centre (23, 20)
    shielded = _unit(sim, 11.5, 18.5, name="behind")      # centre (13, 20)
    far = _unit(sim, 18.5, 48.5, name="far")
    for u in (near, shielded, far):
        u.current_hp = 10_000.0          # the push, not the kill
    starts = {u.name: (u.x, u.y) for u in sim.units}
    trace = []
    for t in range(n_ticks):
        if t == 1:
            _frag(sim, 20, 20)
        sim.step()
        for u in sim.units:
            (ix, ox), (iy, oy) = ui.books_identity(u)
            assert (ix, iy) == (ox, oy), (t, u.name)       # G4 through it
        trace.append(tuple((u.x, u.y, u.current_hp,
                            tuple((s.kind, s.remaining_ticks)
                                  for s in u.statuses))
                           for u in sim.units))
    return sim, starts, trace


def test_a_blast_pushes_the_units_its_wind_reaches_and_nothing_else():
    """Property (§4: a blast pushes through its own wind, nothing else): a
    shipped frag 1 m from a marine pushes it AWAY from the detonation and it
    comes to rest; a marine at the same distance behind a wall and one 10 m
    off are not moved; the momentum books close through the blast (G4); the
    same scene with the wind's input off moves nobody (the push is the
    wind's alone) and lands the same direct damage on the detonation tick.
    Breaks if the blast gains a second push, its wind stops at no wall, or
    the push leaks into the damage."""
    sim, starts, trace = _blast_scene(1.0)
    by = {u.name: u for u in sim.units}
    near = by["near"]
    assert near.x - starts["near"][0] > 0.1           # away: east, > 3 cm
    assert trace[-1][0][:2] == trace[-12][0][:2]      # at rest
    for name in ("behind", "far"):
        assert (by[name].x, by[name].y) == starts[name], name
    sim0, starts0, trace0 = _blast_scene(0.0)
    for u in sim0.units:
        assert (u.x, u.y) == starts0[u.name], u.name
    assert [r[2] for r in trace[1]] == [r[2] for r in trace0[1]]


def test_a_blast_runs_identically_twice():
    """Property (determinism): the shipped frag beside a marine, twice --
    every per-tick position, HP and status identical. Breaks if the row
    draws randomness or reads unordered state."""
    a = _blast_scene(1.0, n_ticks=30)[2]
    assert a == _blast_scene(1.0, n_ticks=30)[2]


# ---------------------------------------------------------------------------
# Dormancy on a real quiet level
# ---------------------------------------------------------------------------
def _quiet_ship_trace(monkeypatch, off):
    import breach_physics as bp
    from control_source import quantize_stick_direction
    from level_loader import load as load_level
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    if off:
        monkeypatch.setattr(CFG.impulse, "wind_push_mult", 0.0)
    sim = Simulation(load_level("ship_demo", levels_dir=str(ROOT / "levels")),
                     seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    standing = [Unit("s1", x=5.0, y=17.0, team=0),
                Unit("s2", x=88.0, y=20.0, team=1)]
    walker = Unit("w", x=5.0, y=22.0, team=0)
    zombie = Unit("z", x=45.0, y=22.0, team=ZOMBIE_TEAM)
    for u in standing + [walker, zombie]:
        sim.add_unit(u)
    sim.set_paused(False)
    dx, dy, _ = quantize_stick_direction(1.0, 0.0, 0.0)
    trace = []
    for _ in range(96):
        sim.set_move_dir(walker.id, dx, dy, 0)  # throttle 0: the walk
        sim.set_paused(False)
        sim.step()
        trace.append(tuple((u.x, u.y, tuple((s.kind, s.remaining_ticks)
                                            for s in u.statuses))
                           for u in sim.units))
    return trace


def test_a_quiet_real_level_runs_as_if_the_wind_were_off(monkeypatch):
    """Property (dormancy): ship_demo with its doors shut -- its EOS jitters
    a few counts (#116), its walkers stir the air -- plays every position
    and status bit-identically against the same run with the wind's input
    switched off (``wind_push_mult = 0``): standing units, a MOVE_DIR walker
    and a zombie. Breaks if a soft draft moves or slows anybody."""
    a = _quiet_ship_trace(monkeypatch, off=False)
    b = _quiet_ship_trace(monkeypatch, off=True)
    assert a == b
    assert a[0] != a[-1]          # non-vacuous: the walker and zombie moved


# ---------------------------------------------------------------------------
# The tile inspector shows the law's input
# ---------------------------------------------------------------------------
def test_the_tile_inspector_shows_q_and_the_force_through_the_one_read():
    """Property: the hover readout's wind row is the tile's W through
    GameMap.footprint_wind_load (footprint 1), q = rho_ref W / 2^33 in Pa
    and the force on a standing body, q * [impulse] drag_area_m2 in N --
    the unit impulse law's input. Breaks if the inspector re-derives the
    law or reads another density or area."""
    from renderer.hover_readout import pack_hover_readout
    g = _room_map()
    _set_air(g, BULK_ONE, 0)
    calm = pack_hover_readout(g, 5, 5, lambda t: t)
    assert calm.wind_w == 0 and calm.wind_force_n == 0.0
    _set_air(g, BULK_ONE, 30 * FP_ONE)
    r = pack_hover_readout(g, 5, 5, lambda t: t)
    assert r.wind_w == g.footprint_wind_load(5, 5, 1)
    assert r.wind_q_pa == exchange.wind_q_pa(g, r.wind_w)
    assert 500.0 < r.wind_q_pa < 560.0          # 1/2 * 1.2047 * 30^2 = 542 Pa
    assert r.wind_force_n == r.wind_q_pa * CFG.impulse.drag_area_m2
    assert any(line.startswith("wind q:") for line in r.lines)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
def test_the_wind_row_is_registered_and_the_old_laws_are_gone():
    """The 9c2 row is the table's ``wind | max`` row, ``apply_wind_force``;
    no row reads a pressure gradient, and #140's own outputs (the cadence
    scale, the bespoke knockdown) no longer exist -- the law's outcomes are
    the wind's (G5p-b: deleted in the commit that made the force live).
    (Its position in the table is not pinned.)"""
    rows = {(r.field, r.reduction): r.response for r in COUPLING_TABLE}
    assert rows[("wind", "max")] is apply_wind_force
    assert all(r.reduction != "grad" for r in COUPLING_TABLE)
    for gone in ("apply_wind_knockdown", "wind_scale_q16", "wind_knocks_down",
                 "wind_settings", "gmap_wind_settings"):
        assert not hasattr(exchange, gone), gone
    from simulation.movement import FootprintSamples
    assert "wind_scale_q16" not in {
        f.name for f in dataclasses.fields(FootprintSamples)}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
