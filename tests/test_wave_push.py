"""The wind row at Simulation.step 9c2 (#140) -- the gate.

The retired ``wave_p | grad`` impulse push lived here; since #140 the 9c2 row
is ``exchange.apply_wind_knockdown``, the ``wind | max`` row of the
momentum-flux threshold law (docs/wind_on_units_design_2026-10-31.md §7):
the dynamic pressure ``q = 1/2 rho v^2`` at the footprint as the int
``W = N_raw * (v^2 >> 16)``; below ``wind_q_slow_pa`` nothing, a cadence
scale down to ``wind_slow_min_scale`` at ``wind_q_fall_pa``, KNOCKED_DOWN at
``W_fall * stability^2``. Nothing in the law writes a position.

Each test names the property it protects and the change that breaks it.
(The cadence callers -- test 6 -- live in test_footprint_samples_builder.py;
the #140 doorway reproduction in test_wind_doorway_e2e.py.)

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
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from simulation import exchange  # noqa: E402
from simulation.atmosphere_fixed import FP_ONE  # noqa: E402
from simulation.environment import EnvironmentProfile  # noqa: E402
from simulation.exchange import (  # noqa: E402
    COUPLING_TABLE, WIND_ONE, apply_wind_knockdown, wind_knocks_down,
    wind_scale_q16, wind_settings,
)
from simulation.factions import ZOMBIE_TEAM  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.gas_fixed import BULK_ONE  # noqa: E402
from simulation.gases import INERT_N2, O2  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.status import KNOCKED_DOWN, composed_flags, tick_statuses  # noqa: E402
from simulation.unit import Unit  # noqa: E402
from simulation import unit_fixed  # noqa: E402

RHO_AIR = 1.2047          # only to build stub maps; the law reads the table


def _settings():
    return wind_settings(CFG.exchange, RHO_AIR)


def _marine(x=6.0, y=6.0):
    u = Unit("M", x=x, y=y, team=0)
    u.id = 1
    return u


def _zombie(x=6.0, y=9.0):
    u = Unit("Z", x=x, y=y, team=ZOMBIE_TEAM)
    u.id = 2
    return u


def _stub_map(w_load):
    """A gmap whose footprint wind load is ``w_load`` everywhere (a callable
    of (fy, fx) or an int): the law's input, set exactly."""
    f = w_load if callable(w_load) else (lambda fy, fx: w_load)
    return SimpleNamespace(
        materials=SimpleNamespace(density=np.array([RHO_AIR])),
        footprint_wind_load=lambda fy, fx, fp=3: f(fy, fx))


def _room_map(h=16, w=16):
    from level_loader import LevelData
    tm = np.full((h, w), MAT_AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = MAT_HULL
    return GameMap(LevelData(name="wind_room", version="3", path=Path("."),
                             tilemap=tm, tile_size_m=1.0,
                             diffuse_path=Path(".")))


def _set_air(g, n_raw, vx_raw, vy_raw=0):
    """Fixture write of the bulk N and the wind planes over the whole open
    room -- a pure-read test of the law; nothing is stepped afterwards."""
    open_ = ~g.solid
    g.gas[O2][open_] = n_raw // 5
    g.gas[INERT_N2][open_] = n_raw - n_raw // 5
    g.wind_x[open_] = vx_raw
    g.wind_y[open_] = vy_raw


# ---------------------------------------------------------------------------
# 2. Position is never written
# ---------------------------------------------------------------------------
def test_a_unit_in_a_gale_keeps_its_position():
    """Property: a unit in a wind far above q_fall is knocked down and its
    x, y are untouched -- on a real GameMap through the real read. Breaks
    if a displacement term returns to the wind row."""
    g = _room_map()
    _set_air(g, BULK_ONE, 60 * FP_ONE, -20 * FP_ONE)
    u = _marine(x=5.25, y=7.75)
    apply_wind_knockdown([u], g)
    assert (u.x, u.y) == (5.25, 7.75)
    assert [s.kind for s in u.statuses] == [KNOCKED_DOWN]
    assert composed_flags(u).is_prone and not composed_flags(u).can_move


def test_dead_units_are_untouched():
    """A corpse in a gale gets no status (the heat row killed it first)."""
    u = _marine()
    u.alive = False
    apply_wind_knockdown([u], _stub_map(10 ** 15))
    assert u.statuses == []


# ---------------------------------------------------------------------------
# 3. q, not v
# ---------------------------------------------------------------------------
def test_density_is_in_the_law():
    """Property: at the same wind, halving the tile's N halves W exactly,
    and a wind just under the calm edge at ambient density is in the slowed
    band at twice it. Breaks if density leaves the law (a speed threshold)."""
    g = _room_map()
    _set_air(g, 2 * BULK_ONE, 12 * FP_ONE)
    w2 = g.footprint_wind_load(5, 5, 3)
    _set_air(g, BULK_ONE, 12 * FP_ONE)
    w1 = g.footprint_wind_load(5, 5, 3)
    assert w2 == 2 * w1 > 0

    st = _settings()
    # the largest raw speed whose ambient-density load is still calm
    v = 1
    while BULK_ONE * (((2 * v) ** 2) >> 16) < st.w_slow:
        v *= 2
    lo, hi = v, 2 * v
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if BULK_ONE * ((mid * mid) >> 16) < st.w_slow:
            lo = mid
        else:
            hi = mid
    _set_air(g, BULK_ONE, lo)
    assert wind_scale_q16(g.footprint_wind_load(5, 5, 3), st) == WIND_ONE
    _set_air(g, 2 * BULK_ONE, lo)
    assert wind_scale_q16(g.footprint_wind_load(5, 5, 3), st) < WIND_ONE


# ---------------------------------------------------------------------------
# 4. Band edges, against the float law
# ---------------------------------------------------------------------------
def test_band_edges_and_linearity_in_q():
    """Property: ONE just below and AT W_slow, s_min at W_fall and above,
    the interior within one LSB of the FLOAT law s = 1 - (1 - s_min) *
    (q - q_slow) / (q_fall - q_slow) (the integer law floors once, so its
    error is below one LSB), and monotone non-increasing across a sweep.
    Breaks if the law is not linear in q or a clamp goes."""
    st = _settings()
    assert wind_scale_q16(st.w_slow - 1, st) == WIND_ONE
    assert wind_scale_q16(st.w_slow, st) == WIND_ONE
    assert wind_scale_q16(st.w_fall, st) == st.s_min_q16
    assert wind_scale_q16(10 * st.w_fall, st) == st.s_min_q16
    s_min = st.s_min_q16 / WIND_ONE
    prev = WIND_ONE
    for k in range(0, 101):
        w = st.w_slow + (st.w_fall - st.w_slow) * k // 100
        got = wind_scale_q16(w, st)
        q = st.rho_ref * w / float(2 ** 33)
        q_lo = st.rho_ref * st.w_slow / float(2 ** 33)
        q_hi = st.rho_ref * st.w_fall / float(2 ** 33)
        want = (1.0 - (1.0 - s_min) * (q - q_lo) / (q_hi - q_lo)) * WIND_ONE
        assert abs(got - want) <= 1.0, (k, got, want)
        assert got <= prev
        prev = got


# ---------------------------------------------------------------------------
# 5. Knockdown edge and stability
# ---------------------------------------------------------------------------
def _edge(stab_q16, st):
    """The smallest W the pinned integer compare knocks down."""
    num = st.w_fall * stab_q16 * stab_q16
    den = WIND_ONE * WIND_ONE
    return -(-num // den)


def test_knockdown_at_the_pinned_edge_not_one_count_below():
    """Property: a marine goes down at W_fall exactly and stands at
    W_fall - 1; a 0.9 zombie at the QUANTIZED edge, 58982^2 / 2^32 of W_fall
    (about 0.80999, not 0.81). Breaks if stability stops scaling the wind
    SPEED or the compare form moves."""
    st = _settings()
    m_edge = _edge(WIND_ONE, st)
    assert m_edge == st.w_fall
    for w, down in ((m_edge - 1, False), (m_edge, True)):
        u = _marine()
        apply_wind_knockdown([u], _stub_map(w))
        assert (KNOCKED_DOWN in [s.kind for s in u.statuses]) == down, w

    stab_z = unit_fixed.quantize_scalar(float(CFG.zombie.stability))
    z_edge = _edge(stab_z, st)
    ratio = z_edge / st.w_fall
    assert abs(ratio - stab_z * stab_z / 2 ** 32) < 1e-9
    if float(CFG.zombie.stability) == 0.9:
        assert stab_z == 58982 and 0.80998 < ratio < 0.81
    for w, down in ((z_edge - 1, False), (z_edge, True)):
        z = _zombie()
        apply_wind_knockdown([z], _stub_map(w))
        assert (KNOCKED_DOWN in [s.kind for s in z.statuses]) == down, w
    # a braced profile stands through the marine's edge
    braced = _marine()
    braced.environment = EnvironmentProfile(stability=1.5)
    apply_wind_knockdown([braced], _stub_map(m_edge * 2))
    assert braced.statuses == []
    assert wind_knocks_down(m_edge * 3, unit_fixed.quantize_scalar(1.5), st)


# ---------------------------------------------------------------------------
# 10. Re-knock refreshes
# ---------------------------------------------------------------------------
def test_sustained_wind_keeps_a_unit_down_past_one_getup():
    """Property: while W stays at or above W_fall the unit stays down past
    one get-up time -- each tick refreshes the SAME status (P3 stacking),
    and it gets up once the wind drops. Breaks if the row stops refreshing
    or stacks a second instance."""
    st = _settings()
    u = _marine()
    gale = _stub_map(st.w_fall)
    for _ in range(st.getup_ticks + 5):
        tick_statuses([u])
        apply_wind_knockdown([u], gale)
        assert len(u.statuses) == 1
        assert u.statuses[0].remaining_ticks == st.getup_ticks
    calm = _stub_map(0)
    for _ in range(st.getup_ticks + 1):
        tick_statuses([u])
        apply_wind_knockdown([u], calm)
    assert u.statuses == []


# ---------------------------------------------------------------------------
# 8. The config door
# ---------------------------------------------------------------------------
def _cfg(**over):
    base = dict(wind_q_slow_pa=39.0, wind_q_fall_pa=245.0,
                wind_slow_min_scale=0.5, knockdown_getup_ticks=36)
    base.update(over)
    return SimpleNamespace(**{k: v for k, v in base.items() if v is not None})


@pytest.mark.parametrize("over, name", [
    (dict(wind_q_fall_pa=None), "wind_q_fall_pa"),
    (dict(wind_q_slow_pa="39"), "wind_q_slow_pa"),
    (dict(wind_slow_min_scale=True), "wind_slow_min_scale"),
    (dict(wind_q_fall_pa=39.0), "wind_q_slow_pa"),
    (dict(wind_q_fall_pa=20.0), "wind_q_slow_pa"),
    (dict(wind_slow_min_scale=0.0), "wind_slow_min_scale"),
    (dict(wind_slow_min_scale=1.5), "wind_slow_min_scale"),
    (dict(wind_q_fall_pa=39.0 + 1e-12), "quantize"),
])
def test_the_door_refuses_by_name(over, name):
    """Property: a missing or mistyped key, q_fall <= q_slow, a scale
    outside (0, 1] and two thresholds quantizing to one load are refused,
    naming what is wrong. Breaks if a bad dial reaches the band's divide."""
    with pytest.raises((KeyError, TypeError, ValueError), match=name):
        wind_settings(_cfg(**over), RHO_AIR)


def test_the_door_refuses_a_non_positive_air_density():
    for rho in (0.0, -1.2):
        with pytest.raises(ValueError, match="rho_ref"):
            wind_settings(_cfg(), rho)


def test_the_density_is_the_materials_air_row():
    """The law's rho is the map's [materials.air] density (one owner): a
    map with a denser air row gets a proportionally lower W threshold."""
    st = exchange.gmap_wind_settings(_room_map())
    assert st.rho_ref == float(_room_map().materials.density[MAT_AIR])
    thin = wind_settings(CFG.exchange, st.rho_ref / 2)
    assert abs(thin.w_fall / st.w_fall - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# 1. Dormancy on a real quiet level; 9. determinism through a blast
# ---------------------------------------------------------------------------
def _quiet_ship_trace(monkeypatch, astronomical):
    import breach_physics as bp
    from control_source import quantize_stick_direction
    from level_loader import load as load_level
    from simulation import Simulation
    from simulation.orders import ORDER_MOVE_ATTACK
    from simulation.ruleset import ContinuousRealtime
    if astronomical:
        monkeypatch.setattr(CFG.exchange, "wind_q_slow_pa", 1e15)
        monkeypatch.setattr(CFG.exchange, "wind_q_fall_pa", 2e15)
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
        sim.set_move_dir(walker.id, dx, dy, ORDER_MOVE_ATTACK)
        sim.set_paused(False)
        sim.step()
        trace.append(tuple((u.x, u.y, tuple((s.kind, s.remaining_ticks)
                                            for s in u.statuses))
                           for u in sim.units))
    return trace


def test_a_quiet_real_level_runs_as_if_the_law_never_fired(monkeypatch):
    """Property (dormancy): ship_demo with its doors shut -- its EOS jitters
    a few counts (#116) -- runs bit-identically against the same run with
    the thresholds astronomically high: standing units, a MOVE_DIR walker
    and a zombie, positions and statuses every tick. Breaks if the law
    touches a quiet unit (the calm band no longer returns exactly ONE)."""
    a = _quiet_ship_trace(monkeypatch, astronomical=False)
    b = _quiet_ship_trace(monkeypatch, astronomical=True)
    assert a == b
    assert a[0] != a[-1]          # non-vacuous: the walker and zombie moved


def _grenade_trace(d=6, n_ticks=30):
    import breach_physics as bp
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.physics import apply_explosion
    tm = np.full((48, 48), MAT_AIR, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = MAT_HULL
    lvl = LevelData(name="wind_ring", version="3", path=Path("."),
                    tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))
    sim = Simulation(lvl, seed=777, breach_physics=bp, enable_recorder=False)
    u = Unit("P", x=12 + d - 1, y=23, team=0)
    sim.add_unit(u)
    trace = []
    for t in range(n_ticks):
        if t == 1:
            apply_explosion(sim.gmap, sim.edit_queue, 24, 12, 5, 10.0, 200)
        sim.set_paused(False)
        sim.step()
        trace.append((u.x, u.y, tuple((s.kind, s.remaining_ticks)
                                      for s in u.statuses)))
    return trace


def test_a_blast_runs_identically_twice():
    """Property (determinism): the calibration grenade (radius 5, pressure
    10.0) beside a marine, twice -- every per-tick position and status
    identical. Breaks if the row draws randomness or reads unordered state."""
    a = _grenade_trace()
    assert a == _grenade_trace()
    x0, y0 = a[0][0], a[0][1]
    assert all((x, y) == (x0, y0) for (x, y, _s) in a)   # never displaced


# ---------------------------------------------------------------------------
# The tile inspector shows the law's number
# ---------------------------------------------------------------------------
def test_the_tile_inspector_shows_q_through_the_one_read():
    """Property: the hover readout's wind row is the tile's W through
    GameMap.footprint_wind_load (footprint 1), q = rho_ref W / 2^33 in Pa,
    and the law's own cadence scale (1 calm, s_min in a gale). Breaks if the
    inspector re-derives the law or reads another density."""
    from renderer.hover_readout import pack_hover_readout
    g = _room_map()
    st = exchange.gmap_wind_settings(g)
    _set_air(g, BULK_ONE, 0)
    calm = pack_hover_readout(g, 5, 5, lambda t: t)
    assert calm.wind_w == 0 and calm.wind_scale == 1.0
    _set_air(g, BULK_ONE, 30 * FP_ONE)
    r = pack_hover_readout(g, 5, 5, lambda t: t)
    assert r.wind_w == g.footprint_wind_load(5, 5, 1)
    assert r.wind_q_pa == st.rho_ref * r.wind_w / float(2 ** 33)
    assert 500.0 < r.wind_q_pa < 560.0          # 1/2 * 1.2047 * 30^2 = 542 Pa
    assert r.wind_scale == st.s_min_q16 / WIND_ONE
    assert any(line.startswith("wind q:") for line in r.lines)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------
def test_the_wind_row_is_registered_and_the_push_row_is_gone():
    """The 9c2 row is the table's ``wind | max`` row; no row reads a
    pressure gradient any more. (Its position in the table is not pinned.)"""
    rows = {(r.field, r.reduction): r.response for r in COUPLING_TABLE}
    assert rows[("wind", "max")] is apply_wind_knockdown
    assert all(r.reduction != "grad" for r in COUPLING_TABLE)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
