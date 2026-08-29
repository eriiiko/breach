"""P-G0 gate 2 — the gas_energy books identity (gas-energy conservation arc
#54, design §2.2/§2.8/§5, docs/gas_energy_conservation_design_2026-08-29.md).

P-G0 adds NO physics: `gas_energy` is a whole-grid MIRROR of the existing
`(N, T)` state, recomputed every tick by
:meth:`simulation.gamemap.GameMap.refresh_gas_energy` as
``N_raw * (temperature_raw + T_AMB_K_raw)`` on the accountable set (design
§2.2), zero elsewhere. By construction this makes

    Sum_accountable(gas_energy - N_raw * T_AMB_K_raw) == Sum_accountable(N_raw * T_raw)

which is EXACTLY the existing `eos_energy_books_sum` quantity
(`PhysicsRunner.energy_books_sum`, itself a thin wrapper over the C++
`eos_energy_books_sum` / `bulk_transport.cpp`'s `e_participates()`
skip-set). The identity is trivial today (both sides read the same mirror),
but it is the regression gate for every later patch: P-G1a+ makes
`gas_energy` a genuinely STORED, seam-written truth, and the day this
identity stops holding "for free" is the day a writer seam is missing or
wrong. Pinning it now, on three different scenarios, is what makes that
future regression visible.

Run on three scenarios (space-boundary sealed room, the `playground` level,
and a vacuum-breach synthetic level with a hot seed) so the gate is not
vacuous on any one topology (sealed / open / vented).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                              # noqa: E402
from config import CFG                                    # noqa: E402
from level_loader import LevelData, load as load_level    # noqa: E402
from simulation import Simulation                          # noqa: E402
from simulation import atmosphere_fixed                    # noqa: E402
from simulation.gamemap import GameMap                      # noqa: E402
from simulation.gases import O2, INERT_N2                   # noqa: E402
from simulation.physics_runner import PhysicsRunner          # noqa: E402

TICKS = 100


def _t_amb_k_raw() -> int:
    """The SAME canonical T_AMB_K -> raw Q16.16 fold `GameMap.
    _gas_energy_t_amb_raw` uses — re-derived here (not imported) so the test
    does not just check the production code against itself for THIS one
    constant; both must agree with the C++ EOSSolver's own `t_amb_q` fold."""
    import temperature_scale
    from simulation import gas_fixed as _gas_fx
    t_amb_k = temperature_scale.load(CFG).eos_t_amb_k
    return max(1, _gas_fx.quantize_scalar(t_amb_k))


def _books_identity_holds(gmap, runner) -> tuple[bool, int, int]:
    """Return (ok, lhs, rhs): lhs = Sum_accountable(gas_energy - N*T_AMB_raw),
    rhs = the existing eos_energy_books_sum (Sum_accountable(N*T_raw))."""
    accountable = gmap._gas_energy_accountable()
    n_bulk = gmap._gas_bulk_n_raw()
    t_amb_raw = _t_amb_k_raw()
    lhs = int(np.where(accountable, gmap.gas_energy - n_bulk * t_amb_raw, 0)
              .sum(dtype=np.int64))
    rhs = int(runner.energy_books_sum(gmap))
    return lhs == rhs, lhs, rhs


def _sealed_room_sim():
    """A small sealed HULL room (space boundary), ambient throughout — the
    trivial case: N is untouched from its P1-calibration default, T stays 0
    everywhere (no heat source), so gas_energy should equal N*T_AMB exactly
    on every accountable cell for the whole run."""
    h = w = 12
    tm = np.ones((h, w), dtype=np.int32)     # all hull
    tm[1:11, 1:11] = 4                        # carve interior air
    level = LevelData(name="gas_energy_sealed_room", version="1",
                     path=Path("."), tilemap=tm, tile_size_m=0.333,
                     diffuse_path=Path("."))
    return Simulation(level, seed=1, breach_physics=bp, enable_recorder=False)


def _playground_sim():
    """The `playground` level (the arc's own gate benches' level) — a real,
    complex map (doors, vents, multiple rooms) exercising combustion/pump/
    seal seams the sealed-room synthetic never touches."""
    lvl = load_level("playground", levels_dir=str(ROOT / "levels"))
    return Simulation(lvl, seed=2, breach_physics=bp, enable_recorder=False)


def _blast_vent_sim():
    """TRANSCRIBED from tests/_drag2_sweep_bench.py::build_scenario — a
    sealed room breached to vacuum, with two hot gas-cell patches seeded via
    `seed_gas_temperature` (design §2.7's seam primitive). The most
    stringent of the three: non-ambient N AND T, plus a live vacuum
    boundary venting mass out of the accountable set every tick."""
    H = W = 32
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:30, 2:30] = 1
    tm[3:29, 3:29] = 4
    tm[14:18, 29] = 4          # the breach: hull ring opened to vacuum
    level = LevelData(name="gas_energy_blast_vent", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_vacuum.any(), "scenario must have vacuum to vent into"
    q = atmosphere_fixed.quantize_scalar
    g.seed_gas_temperature((slice(6, 10), slice(6, 10)),
                           g.temperature[6:10, 6:10] + q(4000.0))
    g.gas[O2, 7:9, 7:9] += q(4.0)
    g.seed_gas_temperature((slice(18, 22), slice(18, 22)),
                           g.temperature[18:22, 18:22] + q(9000.0))

    class _Runner:
        """A bare Simulation-alike: build_scenario's GameMap has no
        Simulation wrapper (see the drag2 bench), so drive PhysicsRunner
        directly the same way that bench's run_leg() does."""
        def __init__(self, gmap):
            self.gmap = gmap
            self.physics_runner = PhysicsRunner(bp)
            self.physics_runner.eos.dx = float(gmap.tile_size_m)
            self.dt = 1.0 / float(CFG.clock.ticks_per_second)

        def step(self):
            self.physics_runner.step(self.gmap, self.dt, tick=0)

    return _Runner(g)


SCENARIOS = {
    "sealed_room": _sealed_room_sim,
    "playground": _playground_sim,
    "blast_vent": _blast_vent_sim,
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_gas_energy_books_identity_holds_over_100_ticks(name):
    """GATE 2 (P-G0): the books identity holds every tick, on all three
    scenarios, for the whole 100-tick run — not just at tick 100. A failure
    here means `refresh_gas_energy`'s accountable-set mask, N_bulk sum, or
    T_AMB_K fold has drifted from `eos_energy_books_sum`'s own C++
    transcription of the SAME quantities."""
    sim = SCENARIOS[name]()
    gmap = sim.gmap
    runner = sim.physics_runner
    for t in range(TICKS):
        if hasattr(sim, "set_paused"):
            sim.set_paused(False)
        sim.step()
        ok, lhs, rhs = _books_identity_holds(gmap, runner)
        assert ok, (
            f"[{name}] tick {t}: books identity broke -- "
            f"Sum(gas_energy - N*T_AMB) = {lhs} != eos_energy_books_sum = {rhs} "
            f"(delta {lhs - rhs})")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_gas_energy_zero_off_accountable_set(name):
    """`gas_energy` is 0 on every non-accountable cell (solid, thermal_solid,
    vacuum, ambient ring) after a 100-tick run -- design §2.2's "zero
    elsewhere, never read there" clause."""
    sim = SCENARIOS[name]()
    gmap = sim.gmap
    for _ in range(TICKS):
        if hasattr(sim, "set_paused"):
            sim.set_paused(False)
        sim.step()
    accountable = gmap._gas_energy_accountable()
    off = ~accountable
    n_nonzero = int(np.count_nonzero(gmap.gas_energy[off]))
    assert n_nonzero == 0, (
        f"[{name}] gas_energy is nonzero on {n_nonzero} non-accountable cell(s)")


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_gas_energy_dtype_and_shape(name):
    """`gas_energy` is int64, shaped (h, w) -- design §2.2/§5 (digest spec
    v5's `{name='gas_energy', dtype='int64'}` row depends on this)."""
    sim = SCENARIOS[name]()
    gmap = sim.gmap
    assert gmap.gas_energy.dtype == np.int64
    assert gmap.gas_energy.shape == gmap.temperature.shape
