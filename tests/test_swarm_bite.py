"""The larvae's bite (arc #63 P5a, docs/swarm_P5a_impl.md): the swarm's
bite-dose plane and the ``swarm_bite_dose`` coupling row into the poison law.
The CPU build's host path; the device forms are gated by
``tests/cuda_swarm_check.py`` (PART 2c) and ``tests/cuda_unit_damage_check.py``
(the bite scene).

Each test names the property it protects and the change that must break it.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_swarm_bite.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import _swarm_testkit as tk  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import exchange, poison, swarm, swarm_fixed  # noqa: E402
from simulation.damage import POISON, build_mitigation  # noqa: E402
from simulation.events import (CAUSE_NAMES, Cause, UnitHitEvent,  # noqa: E402
                               UnitKilledEvent)
from simulation.simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402

ONE = swarm_fixed.FP_ONE
HALF = ONE >> 1
BITE = CAUSE_NAMES.index(Cause.LARVA_BITE.value)


def _room(h=16, w=16, tile=0.333):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="swarm_bite_test", version="1", path=Path("."), tilemap=tm,
                     tile_size_m=tile, diffuse_path=Path("."))


def _sim(seed=1):
    return Simulation(_room(), seed=seed, breach_physics=bp, enable_recorder=False)


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def _larvae(sim, tiles):
    """One larva per entry, at the tile's centre, heading 0."""
    xs = [(tx << 16) + HALF for tx, _ in tiles]
    ys = [(ty << 16) + HALF for _, ty in tiles]
    return sim.swarm.spawn("larva", xs, ys, [0] * len(tiles))


def _unit(sim, name, cx, cy, **kw):
    """A marine whose 3x3 footprint is centred on tile (cx, cy)."""
    u = Unit(name, x=float(cx - 1), y=float(cy - 1), team=0, **kw)
    sim.add_unit(u)
    return u


def _spy_doses(monkeypatch):
    """Every poison.dose call the bite row makes, as (unit name, amount,
    source, cause) -- the door's own arguments (the levels move on in the
    same tick, so the levels are not the dose)."""
    calls = []
    real = poison.dose

    def spy(u, amount_q, source_id=-1, cause=Cause.POISON_GAS):
        calls.append((u.name, int(amount_q), int(source_id), Cause(cause).value))
        return real(u, amount_q, source_id, cause)
    monkeypatch.setattr(exchange.poison, "dose", spy)
    return calls


# ---------------------------------------------------------------------------
# The plane (the swarm side)
# ---------------------------------------------------------------------------

def test_the_plane_counts_the_biting_larvae_per_tile_and_nothing_else():
    """Property: after a swarm step the plane holds, on every tile, exactly
    (live larvae not in coma on it) x bite_q -- a coma larva, a larva the step
    killed and a freed slot add nothing, and the garbage the plane held
    before is gone (it is rebuilt, never accumulated). Breaks if coma or dead
    slots bite, if the builder adds to last tick's plane, or if the dose per
    larva is not the table's row."""
    T = np.zeros((1, 6, 6), dtype=np.int32)
    T[0, 4, 4] = -30 * ONE                                   # a coma tile
    T[0, 1, 4] = tk.DEFAULT_LARVA["T_lethal_hot_q"] + 1      # a lethal tile
    wd = tk.larva_world(6, 6, 8, 1, T=T)
    q = int(tk.DEFAULT_LARVA["bite_q"])
    for s in range(3):                                       # three on (1, 1)
        tk.place(wd, 0, s, (1 << 16) + HALF, (1 << 16) + HALF, 0)
    tk.place(wd, 0, 3, (3 << 16) + HALF, (2 << 16) + HALF, 0)               # one on (3, 2)
    tk.place(wd, 0, 4, (4 << 16) + HALF, (4 << 16) + HALF, 0, state=swarm.AI_COMA)
    tk.place(wd, 0, 5, (4 << 16) + HALF, (1 << 16) + HALF, 0)               # dies
    wd.high_water[0] = 7                                     # slot 6: freed (all zero)
    rows = bp.PhysicsEngine().step_swarm(
        [swarm.EnginePack("larva", wd.columns(), wd.high_water, wd.tables)],
        wd.T, wd.solid, wd.env, {"larva": False}, wd.bite)
    assert rows.shape[0] == 1                                # the lethal one died
    assert int(wd.ai_state[0, 4]) == swarm.AI_COMA
    expect = np.zeros((6, 6), dtype=np.int64)
    for s in range(7):
        if wd.valid[0, s] and wd.ai_state[0, s] != swarm.AI_COMA:
            expect[wd.pos_y[0, s] >> 16, wd.pos_x[0, s] >> 16] += q
    assert np.array_equal(wd.bite[0], expect)
    assert int(wd.bite[0, 1, 1]) == 3 * q and int(wd.bite[0, 2, 3]) == q
    assert int(wd.bite[0, 4, 4]) == 0 and int(wd.bite[0, 1, 4]) == 0


def test_the_bite_row_is_the_expected_dose_rounded_once():
    """Property: bite_q is bite_dose x bite_rate x dt in Q16 poison units
    with ONE rounding: within half a count of the exact product of the three
    quantized values, and 218 at the repo's numbers (1 unit x 0.08/s at 24
    tps). Breaks if the derivation floors twice or reads another dt."""
    sim = _sim()
    row = sim.gmap.swarm_species.row("larva")
    ctx = swarm.species_context(sim.gmap)
    q = swarm.bite_tick_q(row, ctx)
    exact = row.bite_dose_q * row.bite_rate_q * ctx.dt_q / float(1 << 32)
    assert abs(q - exact) <= 0.5
    assert q == 218


def test_the_plane_is_rebuilt_every_tick_and_does_not_persist():
    """Property: larvae reclaimed between ticks t and t+1 leave the plane all
    zero after tick t+1. Breaks if the plane is accumulated or only cleared
    where a larva stands."""
    sim = _sim()
    slots = _larvae(sim, [(5, 5), (5, 5), (9, 9)])
    _step(sim)
    plane = swarm.bite_plane(sim.gmap)
    assert int(plane.sum()) == 3 * 218
    sim.swarm.reclaim("larva", slots)
    _step(sim)
    assert not swarm.bite_plane(sim.gmap).any()


def test_a_swarm_free_map_never_writes_the_plane():
    """Property: on a map that never held a swarm the plane stays all zero
    (the dormant step writes nothing) and no unit is dosed. Breaks if the
    dormant path touches the plane or the row doses from an empty plane."""
    sim = _sim()
    u = _unit(sim, "M", 7, 7)
    _step(sim, 10)
    assert not swarm.bite_plane(sim.gmap).any()
    assert u.poison_x0 == 0 and u.poison_x1 == 0


# ---------------------------------------------------------------------------
# The row (the unit side)
# ---------------------------------------------------------------------------

def test_a_unit_over_larvae_is_dosed_and_one_beside_them_is_not(monkeypatch):
    """Property: with larvae under one marine's footprint and none under
    another's, the first takes exactly the plane's sum under its footprint
    (cause larva_bite, uncredited) and the second nothing; the plane is
    written whether or not a body stands on it. Breaks if the row reads one
    tile, credits a unit, or doses from outside the footprint."""
    sim = _sim()
    a = _unit(sim, "A", 4, 4)
    b = _unit(sim, "B", 11, 11)
    _larvae(sim, [(4, 4), (3, 4), (5, 5), (8, 4)])           # 3 under A, 1 beside A
    calls = _spy_doses(monkeypatch)
    _step(sim)
    plane = swarm.bite_plane(sim.gmap)
    under_a = exchange.reduce_sum(plane, a.occupied_tiles())
    assert under_a == 3 * 218 and int(plane.sum()) == 4 * 218
    assert exchange.reduce_sum(plane, b.occupied_tiles()) == 0
    assert calls == [("A", under_a, -1, "larva_bite")]
    assert 0 < a.poison_x0 + a.poison_x1 <= under_a
    assert a.poison_cause == BITE and a.poison_src == -1
    assert b.poison_x0 == 0 and b.poison_x1 == 0
    assert b.poison_cause == poison.GAS_CAUSE


def test_the_dose_grows_with_the_count_and_with_time(monkeypatch):
    """Property: two marines over 6 and 2 larvae take doses in the ratio 3:1
    on the first tick, and each one's total grows tick over tick while the
    larvae stay. Breaks if the row takes a max (or a centre) instead of the
    sum, or doses once."""
    sim = _sim()
    a = _unit(sim, "A", 4, 4)
    b = _unit(sim, "B", 11, 11)
    _larvae(sim, [(4, 4)] * 3 + [(3, 3)] * 3 + [(11, 11), (12, 10)])
    calls = _spy_doses(monkeypatch)
    _step(sim)
    assert sorted(calls) == [("A", 6 * 218, -1, "larva_bite"),
                             ("B", 2 * 218, -1, "larva_bite")]
    da = a.poison_x0 + a.poison_x1
    held = []
    for _ in range(12):
        _step(sim)
        held.append(int(a.poison_x0) + int(a.poison_x1))
    assert all(x < y for x, y in zip([da] + held[:-1], held))   # rising
    assert held[-1] > 6 * da


def test_two_units_on_one_tile_each_take_the_tile_in_full(monkeypatch):
    """Property: a larva under two overlapping footprints doses both bodies
    with its whole dose (the plane is read, never split). Breaks if the row
    divides a tile's dose among the units standing on it."""
    sim = _sim()
    a = _unit(sim, "A", 6, 6)
    b = _unit(sim, "B", 8, 6)                                # shares column 7
    _larvae(sim, [(7, 6)])
    calls = _spy_doses(monkeypatch)
    _step(sim)
    assert sorted(calls) == [("A", 218, -1, "larva_bite"), ("B", 218, -1, "larva_bite")]


def test_the_zombie_and_the_poison_immune_take_nothing():
    """Property: a zombie over larvae, and a marine whose poison resistance
    is exactly 0, take no dose (the same immunity as the gas row:
    exchange.poison_immune, and the dose door's zombie refusal). Breaks if
    the bite row skips the immunity the gas row has."""
    sim = _sim()
    z = Unit("Z", x=3.0, y=3.0, team=255)
    sim.add_unit(z)
    r = _unit(sim, "R", 11, 11)
    r.mitigation = build_mitigation(resist_mult={POISON: 0.0})
    _larvae(sim, [(4, 4)] * 4 + [(11, 11)] * 4)
    _step(sim, 5)
    assert z.is_zombie
    assert z.poison_x0 == 0 and z.poison_x1 == 0
    assert r.poison_x0 == 0 and r.poison_x1 == 0


def test_a_suits_resistance_scales_the_harm_through_the_pipeline():
    """Property: two marines bitten by the same larvae for the same time take
    the same dose, and the one whose poison resistance is 0.5 loses about
    half the hit points (the law's packet through apply_packet's
    mitigation). Breaks if the bite hurts outside the poison law or around
    the mitigation."""
    sim = _sim()
    a = _unit(sim, "A", 4, 4)
    b = _unit(sim, "B", 11, 11)
    b.mitigation = build_mitigation(resist_mult={POISON: 0.5})
    _larvae(sim, [(4, 4)] * 4 + [(11, 11)] * 4)
    _step(sim, 24 * 8)
    lost_a, lost_b = 100.0 - a.current_hp, 100.0 - b.current_hp
    assert lost_a > 1.0
    assert abs(lost_b - 0.5 * lost_a) <= 0.05 * lost_a


def test_a_kill_by_bite_carries_the_bite_and_no_credited_unit():
    """Property: a marine killed by the larvae's poison dies with killed_by
    larva_bite and attacker_id -1, every hit on it says larva_bite, and its
    poison levels and cause return to the gas default at death. Breaks if
    the law's packet keeps the gas's cause, or the bite is credited."""
    sim = _sim()
    u = _unit(sim, "V", 6, 6)
    u.current_hp = 3.0
    _larvae(sim, [(6, 6)] * 6 + [(5, 5)] * 6)
    kills, hits = [], []
    for _ in range(24 * 30):
        _step(sim)
        for e in sim.tick_events:
            if isinstance(e, UnitKilledEvent) and e.unit_id == u.id:
                kills.append(e)
            if isinstance(e, UnitHitEvent) and e.unit_id == u.id:
                hits.append(e)
        if not u.alive:
            break
    assert not u.alive and len(kills) == 1
    assert kills[0].killed_by == "larva_bite" and kills[0].attacker_id == -1
    assert hits and all(h.source == "larva_bite" and h.attacker_id == -1 for h in hits)
    assert u.poison_x0 == 0 and u.poison_x1 == 0 and u.poison_cause == poison.GAS_CAUSE


def test_the_latest_dose_names_the_cause_and_a_drained_body_returns_to_gas():
    """Property: a body bitten, then dosed by gas, carries the gas's cause
    (the latest dose wins, the credit's accepted gap); bitten again it
    carries the bite's; once both levels reach 0 the cause is the gas
    default again (so a recovered unit hashes as never poisoned). Breaks if
    a dose does not write the cause or the drain leaves it."""
    u = Unit("P", x=2.0, y=2.0, team=0)
    poison.dose(u, 5000, -1, Cause.LARVA_BITE)
    assert u.poison_cause == BITE
    poison.dose(u, 5000, -1)
    assert u.poison_cause == poison.GAS_CAUSE
    poison.dose(u, 5000, -1, Cause.LARVA_BITE)
    assert u.poison_cause == BITE
    for _ in range(24 * 600):
        poison.advance(u)
        if u.poison_x0 == 0 and u.poison_x1 == 0:
            break
    assert u.poison_x0 == 0 and u.poison_x1 == 0
    assert u.poison_cause == poison.GAS_CAUSE


def test_same_seed_same_doses():
    """Property: two runs of one seed give the same plane and the same poison
    levels every tick. Breaks if the bite draws from an unkeyed source or
    depends on dict / thread order."""
    def run():
        sim = _sim(seed=7)
        u = _unit(sim, "A", 6, 6)
        _larvae(sim, [(6, 6), (5, 6), (7, 7), (6, 5), (10, 10)] * 2)
        out = []
        for _ in range(48):
            _step(sim)
            out.append((swarm.bite_plane(sim.gmap).tobytes(), int(u.poison_x0),
                        int(u.poison_x1), int(u.current_hp * 65536)))
        return out
    assert run() == run()


def test_the_coupling_row_is_registered():
    """Property: the bite is ONE row of the coupling table, a footprint sum
    of the swarm's plane answered by apply_bite_dose, after the gas poison
    row. Breaks if the bite is wired outside the table."""
    rows = exchange.COUPLING_TABLE
    names = [r.field for r in rows]
    i = names.index("swarm_bite_dose")
    assert rows[i].reduction == "sum" and rows[i].response is exchange.apply_bite_dose
    assert i > names.index("gas[poison]")
    assert rows[i].reduction in exchange.REDUCTIONS


@pytest.mark.parametrize("attr", ["swarm_bite_dose"])
def test_the_plane_is_resident_and_not_digested(attr):
    """Property: the plane is a resident field (the device step writes it)
    and enters no digest (a function of the store). Breaks if it joins the
    field digest (a membership change: a version bump and every golden)."""
    from simulation.gamemap import GameMap
    import field_ab_harness as fab
    assert attr in GameMap._RESIDENT_SYNCED
    assert attr not in getattr(fab, "SIM_FIELDS", ())
