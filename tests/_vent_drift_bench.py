"""Vent system PATCH 1 (issue #48, docs/vent_system_design_2026-08-23.md §7)
— the corridor drift bench, folded from #4 per ruling.

A synthetic single-row corridor: one RETURN vent at one end, one SUPPLY vent
at the other, sharing a duct, run under REAL physics (the compiled EOS
solver, not the ``breach_physics=None`` unit-test shortcut every other vent
test uses — this bench needs the actual wind field the circulation mass
delta induces) to STEADY STATE, then reports the corridor's steady |u|
against ``u_dead(k2)`` — the drag law's (docs/drag_law_v2_design_2026-08-23.
md §4) quantization dead-zone floor, DERIVED FROM THE LIVE INTEGER CHAIN
(never a printed number, R5 — the same ``_find_u0_raw`` search
``test_drag2_dead_zone_property.py`` uses), across a small ``q_circ`` sweep
and ``k2 in {0.01, 0.1, 1.0}``.

This is a HARNESS, not a pytest gate (the ``_`` prefix — tests/ convention):
it prints a table and exits 0. The #4 interlock it operationalizes: if honest
mass-flux drift dies in the dead zone at plausible vent rates, plan B (a
small directed momentum term at supply vents) is the fallback — this bench
is what would show that.

Run:
    conda run -n data python tests/_vent_drift_bench.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from level_loader import EntityInstance, LevelData  # noqa: E402
from simulation import Simulation  # noqa: E402
from simulation.entities import REGISTRY  # noqa: E402

TPS = 24                      # config.toml [clock].ticks_per_second
DT = 1.0 / TPS
CORRIDOR_LEN = 24             # interior tiles between the two vents
STEADY_TICKS = 400            # ticks run before sampling (empirically settles)
SAMPLE_TILE_FRAC = 0.5        # sample the corridor MIDPOINT


def _find_u0_raw(kd2_q: int) -> int:
    """THE live-integer-chain dead-zone floor (drag design §4, R5): the
    smallest positive ``umag_raw`` with ``(kd2_q*umag_raw) >> 16 > 0`` —
    verbatim from tests/test_drag2_dead_zone_property.py's
    ``_find_u0_raw`` (not re-derived by hand)."""
    assert kd2_q > 0
    umag = 1
    while (kd2_q * umag) >> 16 == 0:
        umag += 1
    return umag


def _kd2_q(k_drag2: float, dt: float) -> int:
    x = float(k_drag2) * float(dt) * 65536.0
    return int(np.floor(x + 0.5) if x >= 0 else np.ceil(x - 0.5))


def _corridor_level(h=3, w=CORRIDOR_LEN + 6, q_circ=5.0):
    tm = np.ones((h, w), dtype=np.int32)     # 1 == solid wall material
    tm[1, 1:w - 1] = 4                       # a single-row open corridor
    duct = _inst("duct", "d1", 0, filter="derelict")
    ret = _inst("vent", "vret", 1, x=2, y=1, mount="floor", role="return",
               duct="d1", q_circ=q_circ)
    sup = _inst("vent", "vsup", 2, x=w - 3, y=1, mount="floor", role="supply",
               duct="d1", q_circ=q_circ)
    lvl = LevelData(name="vent_drift_bench", version="1", path=Path("."),
                    tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
                    entities=[duct, ret, sup], wires=[])
    return lvl, (1, (2 + w - 3) // 2)         # (level, mid-corridor sample tile)


def _inst(cls_name, eid, ordinal, **overrides):
    cls = REGISTRY[cls_name]
    fields = {f.name: f.default for f in cls.FIELDS}
    fields.update(overrides)
    return EntityInstance(id=eid, class_name=cls_name, ordinal=ordinal,
                          tags=(), fields=fields)


def _run_one(q_circ: float, k_drag2: float) -> int:
    """Build a fresh corridor sim, arm k_drag2 (post-construction, per the
    drag design §7 P3 note — no config dial for it), run to steady state,
    return the steady |u| at the corridor midpoint (raw Q16.16)."""
    lvl, (sy, sx) = _corridor_level(q_circ=q_circ)
    sim = Simulation(lvl, seed=1, breach_physics=bp, enable_recorder=False)
    sim.physics_runner.eos.k_drag2 = float(k_drag2)
    for _ in range(STEADY_TICKS):
        sim.set_paused(False)
        sim.step()
    ux = int(sim.gmap.wind_x[sy, sx])
    uy = int(sim.gmap.wind_y[sy, sx])
    return int(round((ux * ux + uy * uy) ** 0.5))


def main() -> None:
    q_circ_sweep = (1.0, 5.0, 20.0)
    k2_sweep = (0.01, 0.1, 1.0)

    print(f"{'q_circ':>8} {'k2':>6} {'kd2_q':>8} {'U0_raw':>8} "
          f"{'u_dead_real':>12} {'|u|_steady_raw':>15} {'|u|_steady_real':>16} "
          f"{'above_floor':>12}")
    for q_circ in q_circ_sweep:
        for k2 in k2_sweep:
            kd2_q = _kd2_q(k2, DT)
            u0 = _find_u0_raw(kd2_q) if kd2_q > 0 else 0
            u_dead_real = u0 / 65536.0
            u_steady_raw = _run_one(q_circ, k2)
            u_steady_real = u_steady_raw / 65536.0
            above = u_steady_raw >= u0
            print(f"{q_circ:8.2f} {k2:6.2f} {kd2_q:8d} {u0:8d} "
                  f"{u_dead_real:12.6f} {u_steady_raw:15d} {u_steady_real:16.6f} "
                  f"{str(above):>12}")


if __name__ == "__main__":
    main()
