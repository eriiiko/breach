"""B4 — the pump N-feed actuator + the integer zero-clamped gas-N primitive.

Arc B impl doc (docs/arc_b_impl_2026-07-21.md v2), patch B4. Gates:

- The NEW GameMap gas-N primitive (§6/D10): inject splits ΔN at the standard
  O2/N2 mix; extract removes ΔN proportional to composition, per-slice
  zero-clamped. Inject-then-extract of equal ΔN round-trips a tile's total N
  EXACTLY (per-slice exact on an empty tile); a near-empty tile clamps at 0 with
  no negative slice and no over-withdraw; the integer proportional split
  conserves the requested ΔN exactly; loud int32 overflow on inject.
- The pump ΔN quantum + the D11 load assert (ΔN_per_tick_atm < 2·band) fires.
- at_target hysteresis: a latched Schmitt band — reaches and latches, holds
  across the band (no chatter), releases past 2·band.
- port-tile-distinct-from-body: the edit lands on the open port tile, not the
  solid mount (which the physics solid mask would zero).
- The 2-tick story: the pump edit at 9e(d) lands on the CPU mirror this tick
  (the solver consumes it next tick).
- Dormancy: a pump-free level builds no pumps (the byte-identical door-only gate
  itself lives in tests/test_b1_signal_bus).

Fixtures are programmatic LevelData / EntityInstance + real GameMaps (the
B1/B2/B3 idiom) — no repo level is mutated, no golden moves.

Run:
    conda run -n data python -m pytest tests/test_b4_pump.py -q
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from level_loader import EntityInstance, LevelData, Wire  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation import gas_fixed  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402
from simulation.entities.nodes import PUMP_DEFAULT_BAND_Q16  # noqa: E402
from simulation.gamemap import GameMap, N_GASES  # noqa: E402
from simulation.gases import SMOKE, INERT_N2, O2  # noqa: E402
from simulation.pump_system import (  # noqa: E402
    PumpRuntime, build_pumps, is_pump,
)
from simulation.signal_bus import SignalBus  # noqa: E402

TPS = 24   # config.toml [clock].ticks_per_second


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _split_box_tm(h=12, w=12, wall_x=6):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4              # interior air
    tm[1:h - 1, wall_x] = 1               # a vertical solid wall
    return tm


def _level(tm, entities=(), wires=(), name="b4_fix", tile_size_m=1.0, **kw):
    return LevelData(name=name, version="1", path=Path("."), tilemap=tm,
                     tile_size_m=tile_size_m, diffuse_path=Path("."),
                     entities=list(entities), wires=list(wires), **kw)


def _pump_inst(eid, ordinal, **overrides):
    cls = REGISTRY["pump"]
    fields = {f.name: f.default for f in cls.FIELDS}
    fields.update(overrides)
    return EntityInstance(id=eid, class_name="pump", ordinal=ordinal,
                          tags=(), fields=fields)


def _clock_inst(eid, ordinal, period=1):
    cls = REGISTRY["clock"]
    fields = {f.name: f.default for f in cls.FIELDS}
    fields["period"] = period
    return EntityInstance(id=eid, class_name="clock", ordinal=ordinal,
                          tags=(), fields=fields)


def _bare_gmap():
    return GameMap(_level(_split_box_tm()))


def _zero_tile(gmap, fy, fx):
    for g in range(N_GASES):
        gmap.gas[g][fy, fx] = 0


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


# ===========================================================================
# 1. The integer gas-N primitive (GameMap) — N conservation (§6/D10)
# ===========================================================================

def test_inject_splits_standard_mix_exactly():
    g = _bare_gmap()
    _zero_tile(g, 5, 5)
    dn = 100_000
    g.inject_gas_n(5, 5, dn)
    o2 = int(g.gas[O2][5, 5])
    n2 = int(g.gas[INERT_N2][5, 5])
    # Exact injected mass == ΔN (the remainder rule); mix ≈ 0.21 O2.
    assert o2 + n2 == dn
    o2_frac_q = gas_fixed.quantize_scalar(0.21)
    assert o2 == (dn * o2_frac_q + (1 << 15)) >> 16       # the air-seed idiom
    assert abs(o2 / dn - 0.21) < 1e-3
    # Only the two bulk slices moved.
    assert int(g.gas[SMOKE][5, 5]) == 0


def test_inject_then_extract_roundtrips_total_n_exactly_empty_tile():
    # Empty tile → per-slice EXACT round-trip (extract removes what inject added
    # in the same 0.21/0.79 proportion).
    g = _bare_gmap()
    _zero_tile(g, 5, 5)
    dn = 65_536
    g.inject_gas_n(5, 5, dn)
    removed = g.extract_gas_n(5, 5, dn)
    assert removed == dn
    for gg in range(N_GASES):
        assert int(g.gas[gg][5, 5]) == 0                  # per-slice exact


def test_inject_then_extract_roundtrips_total_n_exactly_populated_tile():
    # A pre-populated tile (incl. a trace gas): TOTAL N is restored to the LSB.
    g = _bare_gmap()
    _zero_tile(g, 5, 5)
    g.gas[O2][5, 5] = 1000
    g.gas[INERT_N2][5, 5] = 3000
    g.gas[SMOKE][5, 5] = 500
    total0 = sum(int(g.gas[gg][5, 5]) for gg in range(N_GASES))
    dn = 20_000
    g.inject_gas_n(5, 5, dn)
    g.extract_gas_n(5, 5, dn)
    total1 = sum(int(g.gas[gg][5, 5]) for gg in range(N_GASES))
    assert total1 == total0
    assert all(int(g.gas[gg][5, 5]) >= 0 for gg in range(N_GASES))


def test_extract_near_empty_clamps_at_zero_no_negative_no_overwithdraw():
    g = _bare_gmap()
    _zero_tile(g, 5, 5)
    g.gas[O2][5, 5] = 30
    g.gas[INERT_N2][5, 5] = 20                             # total 50
    removed = g.extract_gas_n(5, 5, 1000)                  # ask far more than held
    assert removed == 50                                   # min(ΔN, available)
    assert sum(int(g.gas[gg][5, 5]) for gg in range(N_GASES)) == 0
    assert all(int(g.gas[gg][5, 5]) >= 0 for gg in range(N_GASES))
    # A wholly empty tile is a no-op, never negative.
    assert g.extract_gas_n(5, 5, 1000) == 0
    assert all(int(g.gas[gg][5, 5]) == 0 for gg in range(N_GASES))


def test_extract_proportional_split_conserves_requested_dn_exactly():
    # No-remainder case: 600 split by 150 → 25/75/50, sums to 150.
    g = _bare_gmap()
    _zero_tile(g, 5, 5)
    g.gas[O2][5, 5] = 100
    g.gas[INERT_N2][5, 5] = 300
    g.gas[SMOKE][5, 5] = 200                         # total 600
    before = [int(g.gas[gg][5, 5]) for gg in range(N_GASES)]
    removed = g.extract_gas_n(5, 5, 150)
    after = [int(g.gas[gg][5, 5]) for gg in range(N_GASES)]
    per_slice = [before[i] - after[i] for i in range(N_GASES)]
    assert removed == 150
    assert sum(per_slice) == 150                           # exact conservation
    assert per_slice[O2] == 25 and per_slice[INERT_N2] == 75
    assert per_slice[SMOKE] == 50
    assert all(x >= 0 for x in after)


def test_extract_proportional_split_with_remainder_cascades_pinned():
    # Remainder case: total 7, want 3 → floors 1/1 (O2/N2), short 1 → cascade to
    # the first slice with capacity (pinned slice-id order) → removed sums to 3.
    g = _bare_gmap()
    _zero_tile(g, 5, 5)
    g.gas[O2][5, 5] = 3
    g.gas[INERT_N2][5, 5] = 4                              # total 7
    removed = g.extract_gas_n(5, 5, 3)
    assert removed == 3
    assert int(g.gas[O2][5, 5]) + int(g.gas[INERT_N2][5, 5]) == 4   # 7 - 3


def test_inject_overflow_is_loud():
    g = _bare_gmap()
    _zero_tile(g, 5, 5)
    g.gas[O2][5, 5] = (1 << 31) - 10                       # near int32 max
    with pytest.raises(OverflowError):
        g.inject_gas_n(5, 5, 1000)


def test_primitive_bounds_are_strict():
    g = _bare_gmap()
    with pytest.raises(ValueError):
        g.inject_gas_n(-1, 0, 100)
    with pytest.raises(ValueError):
        g.extract_gas_n(0, 9999, 100)


def test_nonpositive_delta_is_noop():
    g = _bare_gmap()
    _zero_tile(g, 5, 5)
    g.inject_gas_n(5, 5, 0)
    g.inject_gas_n(5, 5, -50)
    assert g.extract_gas_n(5, 5, 0) == 0
    assert all(int(g.gas[gg][5, 5]) == 0 for gg in range(N_GASES))


# ===========================================================================
# 2. PumpRuntime.sweep — the edit + at_target hysteresis (§6)
# ===========================================================================

def _pump_rt(gmap, port_y, port_x, target_atm, band, delta_n_q,
             inject_slots=(), extract_slots=(), at_target_slot=None):
    inst = types.SimpleNamespace(ordinal=0, id="p0", class_name="pump",
                                 fields={})
    return PumpRuntime(inst, at_target_slot, list(inject_slots),
                       list(extract_slots), port_y, port_x, target_atm, band,
                       delta_n_q)


def test_sweep_inject_lands_on_mirror_when_held():
    g = _bare_gmap()
    _zero_tile(g, 5, 5)
    bus = SignalBus([(0, "value")])
    inj = bus.slot(0, "value")
    rt = _pump_rt(g, 5, 5, 65536, PUMP_DEFAULT_BAND_Q16, 5000,
                  inject_slots=[inj])
    sim = types.SimpleNamespace(_signal_bus=bus, gmap=g)
    bus.pub[inj] = 0
    rt.sweep(sim)                                          # inject NOT held → no edit
    assert int(g.gas[O2][5, 5]) == 0
    bus.pub[inj] = 1
    rt.sweep(sim)                                          # inject held → edit lands
    assert int(g.gas[O2][5, 5]) + int(g.gas[INERT_N2][5, 5]) == 5000


def test_sweep_extract_beats_inject():
    g = _bare_gmap()
    _zero_tile(g, 5, 5)
    g.gas[O2][5, 5] = 4000
    g.gas[INERT_N2][5, 5] = 6000                           # total 10000
    bus = SignalBus([(0, "inj"), (1, "ext")])
    inj, ext = bus.slot(0, "inj"), bus.slot(1, "ext")
    rt = _pump_rt(g, 5, 5, 65536, PUMP_DEFAULT_BAND_Q16, 2000,
                  inject_slots=[inj], extract_slots=[ext])
    sim = types.SimpleNamespace(_signal_bus=bus, gmap=g)
    bus.pub[inj] = 1
    bus.pub[ext] = 1                                       # both held → extract wins
    rt.sweep(sim)
    assert sum(int(g.gas[gg][5, 5]) for gg in range(N_GASES)) == 8000   # 10000-2000


def test_at_target_schmitt_enter_hold_release_reenter():
    g = _bare_gmap()
    target, band = 65536, PUMP_DEFAULT_BAND_Q16            # release = 2*band = 6554
    bus = SignalBus([(0, "at_target")])
    at = bus.slot(0, "at_target")
    rt = _pump_rt(g, 5, 5, target, band, 0, at_target_slot=at)
    sim = types.SimpleNamespace(_signal_bus=bus, gmap=g)

    def _sweep_with_P(p):
        g.atmosphere[5, 5] = p
        rt.sweep(sim)
        return rt.at_target, bus.read(0, "at_target")

    # Outside the band (band < err < release): stays OFF (never entered).
    assert _sweep_with_P(target - 4000) == (0, 0)
    # Enter the ±band window: latches ON.
    assert _sweep_with_P(target - 3000) == (1, 1)
    # Drift back into the release gap (band < err <= 2·band): HELD (no chatter).
    assert _sweep_with_P(target - 4000) == (1, 1)
    assert _sweep_with_P(target - 6000) == (1, 1)
    # Past the release band (err > 2·band): unlatches.
    assert _sweep_with_P(target - 7000) == (0, 0)
    # Re-enter from the other side of target: latches ON again.
    assert _sweep_with_P(target + 3000) == (1, 1)


def test_at_target_latch_reached_sweeping_with_dn_under_2band():
    # D11 entry guarantee: a per-tick step just under 2·band cannot skip the full
    # entry window, so sweeping P upward toward target ALWAYS lands at_target=1.
    g = _bare_gmap()
    target, band = 65536, PUMP_DEFAULT_BAND_Q16
    step = 2 * band - 1                                    # the max allowed ΔN step
    bus = SignalBus([(0, "at_target")])
    rt = _pump_rt(g, 5, 5, target, band, 0,
                  at_target_slot=bus.slot(0, "at_target"))
    sim = types.SimpleNamespace(_signal_bus=bus, gmap=g)
    latched = False
    p = target - 4 * band
    while p <= target + 2 * band:
        g.atmosphere[5, 5] = p
        rt.sweep(sim)
        latched = latched or rt.at_target == 1
        p += step
    assert latched                                         # entry was reachable


def test_dead_pump_freezes_latch_and_skips_edit():
    g = _bare_gmap()
    _zero_tile(g, 5, 5)
    bus = SignalBus([(0, "value")])
    inj = bus.slot(0, "value")
    rt = _pump_rt(g, 5, 5, 65536, PUMP_DEFAULT_BAND_Q16, 5000,
                  inject_slots=[inj])
    rt.alive = False
    sim = types.SimpleNamespace(_signal_bus=bus, gmap=g)
    bus.pub[inj] = 1
    rt.sweep(sim)
    assert int(g.gas[O2][5, 5]) == 0                       # no edit while dead
    assert rt.at_target == 0


# ===========================================================================
# 3. build_pumps — the ΔN quantum + the D11 load assert + port resolve
# ===========================================================================

def _sim_with_pump(pump, extra_entities=(), wires=None):
    """Construct a live Simulation with a pump wired for inject (so a bus exists
    and build_pumps runs)."""
    clk = _clock_inst("clk", 0, period=1)
    if wires is None:
        wires = [Wire(0, "value", pump.ordinal, "inject", "held")]
    ents = [clk, pump, *extra_entities]
    return Simulation(_level(_split_box_tm(), ents, wires), seed=1,
                      breach_physics=None, enable_recorder=False)


def test_build_pumps_ok_and_marker():
    assert is_pump("pump") and not is_pump("door") and not is_pump("clock")
    pump = _pump_inst("p", 1, x=6, y=5, port_dx=-1, port_dy=0, rate=1.0,
                      target_atm=65536)
    sim = _sim_with_pump(pump)
    assert len(sim._pumps) == 1
    rt = sim._pumps[0]
    # rate 1.0 atm/s / 24 tps = 0.041666 atm/tick → q = quantize(0.041666).
    assert rt.delta_n_q == gas_fixed.quantize_scalar(1.0 / TPS)
    # Port resolved to the OPEN tile left of the wall (mount is on the wall).
    assert (rt.port_y, rt.port_x) == (5, 5)
    # The pump instance was REPLACED by its runtime wrapper (§8 at_target row).
    assert isinstance(sim._entity_by_ordinal[1], PumpRuntime)


def test_d11_band_skip_assert_fires_at_load():
    # rate 3.0 atm/s / 24 tps = 0.125 atm/tick → q = 8192 >= 2·band (6554).
    pump = _pump_inst("p", 1, x=6, y=5, port_dx=-1, port_dy=0, rate=3.0)
    with pytest.raises(ValueError, match="never latches"):
        _sim_with_pump(pump)


def test_port_out_of_bounds_hard_errors_at_load():
    pump = _pump_inst("p", 1, x=1, y=1, port_dx=-5, port_dy=0, rate=1.0)
    with pytest.raises(ValueError, match="out of the"):
        _sim_with_pump(pump)


# ===========================================================================
# 4. E2E — the 2-tick story + port-distinct-from-body through a live sim
# ===========================================================================

def test_e2e_pump_edit_lands_on_mirror_port_not_body():
    # Mount ON the wall (5,6); port points to the open tile (5,5). A clock drives
    # inject held; after a step the PORT gas rose on the mirror (the solver reads
    # it next tick, the 2-tick field contract) while the SOLID body stays 0.
    pump = _pump_inst("p", 1, x=6, y=5, port_dx=-1, port_dy=0, rate=1.0,
                      target_atm=65536)
    sim = _sim_with_pump(pump)
    assert sim._signal_bus is not None
    assert sim.gmap.solid[5, 6] and not sim.gmap.solid[5, 5]

    o2_port_0 = int(sim.gmap.gas[O2][5, 5])
    body_gas_0 = sum(int(sim.gmap.gas[g][5, 6]) for g in range(N_GASES))
    _step(sim, 2)                                          # clock=1 by tick 1 → inject
    o2_port_1 = int(sim.gmap.gas[O2][5, 5])
    body_gas_1 = sum(int(sim.gmap.gas[g][5, 6]) for g in range(N_GASES))

    assert o2_port_1 > o2_port_0                           # edit landed on the port
    assert body_gas_1 == body_gas_0 == 0                  # solid body untouched


def test_e2e_at_target_latches_and_rides_entity_row():
    # Physics off → the open-air fill is a stable 1.0 atm (65536). A pump whose
    # target IS the fill pressure latches at_target on the first sweep, and the
    # latch rides the pump's __entity__ runtime row (§8) regardless of wiring.
    pump = _pump_inst("p", 1, x=6, y=5, port_dx=-1, port_dy=0, rate=1.0,
                      target_atm=65536)
    sim = _sim_with_pump(pump)
    assert int(sim.gmap.atmosphere[5, 5]) == 65536         # open-air fill
    _step(sim)
    assert sim._pumps[0].at_target == 1                    # err 0 ≤ band → latched
    # The latch is a synced runtime digest row on the pump.
    assert REGISTRY["pump"].runtime_digest_rows(sim._pumps[0]) == \
        (("at_target", 1),)


# ===========================================================================
# 5. Dormancy — a pump-free level builds no pumps
# ===========================================================================

def test_pump_free_level_builds_no_pumps():
    # A wire-free, pump-free level carries no bus and no pumps (the byte-
    # identical door-only gate itself lives in test_b1_signal_bus).
    sim = Simulation(_level(_split_box_tm()), seed=1, breach_physics=None,
                     enable_recorder=False)
    assert sim._signal_bus is None
    assert sim._pumps == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
