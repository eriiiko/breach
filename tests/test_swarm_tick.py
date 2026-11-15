"""The swarm step on the tick (arc #63 P3, docs/swarm_P3_impl.md §11
T-P3-26..30, 32, 34..37; v4.4, v4.5). The CPU build's host path:
``PhysicsRunner._step_host`` -> ``PhysicsEngine.step_swarm``.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_swarm_tick.py -q
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
import field_ab_harness as fab  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import physics_runner as pr  # noqa: E402
from simulation import swarm, swarm_fixed, swarm_tables  # noqa: E402
from simulation.events import SwarmUnitDiedEvent  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402

ONE = swarm_fixed.FP_ONE
M32 = (1 << 32) - 1


def _room(h=10, w=10, tile=1.0):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="swarm_tick_test", version="1", path=Path("."), tilemap=tm,
                     tile_size_m=tile, diffuse_path=Path("."))


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def _deaths(sim):
    return [e for e in sim.tick_events if isinstance(e, SwarmUnitDiedEvent)]


# ---------------------------------------------------------------------------
# T-P3-26
# ---------------------------------------------------------------------------

def test_a_swarm_free_run_launches_nothing_and_moves_no_golden():
    """T-P3-26: the canonical scenario (no larva) runs 30 ticks with
    bp.swarm_cpu_calls() unchanged and (0, ROW_WORDS) death rows every tick,
    and its trajectory digest is GOLDEN_AGGREGATE.

    Must break on an unconditional launch, a field write by the swarm step,
    or a draw taken on a swarm-free map."""
    from _xarch_perfield_digest import GOLDEN_AGGREGATE, trajectory_digest
    c0 = bp.swarm_cpu_calls()
    sim = fab.default_scenario_sim()
    for _ in range(30):
        _step(sim)
        assert sim.physics_runner.swarm_death_rows.shape == (0, swarm.ROW_WORDS)
    assert bp.swarm_cpu_calls() == c0
    assert trajectory_digest(fab.capture_trajectory(n_steps=30)) == GOLDEN_AGGREGATE
    assert bp.swarm_cpu_calls() == c0


# ---------------------------------------------------------------------------
# T-P3-27
# ---------------------------------------------------------------------------

def test_the_draws_clock_survives_the_wego_round_rewind(monkeypatch):
    """T-P3-27: in a TwoPhaseWEGO sim run through two round boundaries, the
    packed ENV_TICK word rises by exactly 1 per step while sim.tick rewinds,
    and equals sim.draws.tick & M32 at slot 7; the key words equal
    sim.draws.key; the gather draws nothing (sim.draws.drawn unchanged by it).

    Must break if the counter is keyed on sim.tick, or the key is read from
    anywhere but sim.draws."""
    sim = fab.swarm_scenario_sim()
    from simulation.ruleset import TwoPhaseWEGO
    assert isinstance(sim.ruleset, TwoPhaseWEGO)
    seen = []
    real = swarm.engine_args

    def spy(gmaps, draws_list, sim_time, *, resident):
        d0 = sim.draws.drawn
        out = real(gmaps, draws_list, sim_time, resident=resident)
        assert sim.draws.drawn == d0, "the gather drew"
        seen.append((int(out["env"][0, swarm.ENV_TICK]), int(out["env"][0, swarm.ENV_SEED_LO]),
                     int(out["env"][0, swarm.ENV_SEED_HI]), sim.draws.tick & M32, sim.tick))
        return out
    monkeypatch.setattr(swarm, "engine_args", spy)
    n = 2 * int(sim.ruleset.ticks_per_round(sim) if callable(getattr(sim.ruleset, "ticks_per_round", None))
                else 240) + 20
    rewinds = 0
    prev_tick = None
    for _ in range(n):
        _step(sim)
        if prev_tick is not None and sim.tick < prev_tick:
            rewinds += 1
        prev_tick = sim.tick
    assert rewinds >= 2, "the run must cross two round boundaries"
    words = [s[0] for s in seen]
    assert all(b - a == 1 for a, b in zip(words, words[1:]))
    assert all(s[0] == s[3] for s in seen)
    assert all((s[1], s[2]) == sim.draws.key for s in seen)
    assert len({s[4] for s in seen}) < len(seen), "sim.tick must have rewound"


# ---------------------------------------------------------------------------
# T-P3-28, T-P3-30
# ---------------------------------------------------------------------------

def test_death_events_come_in_slot_order_from_the_pre_zero_values():
    """T-P3-28: larvae spawned into a lethal seeded patch emit
    SwarmUnitDiedEvents on the tick of death in ascending slot order, each
    with its slot's id, the heading it moved along, and a post-move position
    within one step of its pre-step position that the RAW reader finds
    lethal; the freed slot is reused by the next spawn with a NEW id.

    Must break on an unordered or atomic append, events built from the
    zeroed slot, or a reused id."""
    import temperature_scale
    sim = Simulation(_room(), seed=3, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    ts = temperature_scale.load()
    g.seed_gas_temperature((slice(2, 6), slice(2, 6)),
                           swarm_fixed.quantize_scalar(ts.from_kelvin(373.15)))
    xs = [(3 << 16) + 100 * k for k in range(6)]
    ys = [(4 << 16) + 7000 * k for k in range(6)]
    hs = [k * 20000 for k in range(6)]
    sim.swarm.spawn("larva", xs, ys, hs)
    sim.swarm.spawn("larva", [(7 << 16) + 5], [(7 << 16) + 5], [0])   # a survivor, slot 6
    pre = {c.name: swarm.swarm_column(g, "larva", c.name).copy() for c in swarm.ROSTER}
    _step(sim)
    ev = _deaths(sim)
    assert len(ev) == 6
    lethal = g.swarm_species.row("larva").T_lethal_hot_q
    slots = []
    for e in ev:
        s = int(np.flatnonzero(pre["unit_id"] == e.unit_id)[0])
        slots.append(s)
        assert e.species == "larva"
        assert e.heading_q == int(pre["heading"][s])
        assert abs(e.pos_x_q - int(pre["pos_x"][s])) < ONE
        assert abs(e.pos_y_q - int(pre["pos_y"][s])) < ONE
        assert int(g.temperature[e.pos_y_q >> 16, e.pos_x_q >> 16]) > lethal
    assert slots == sorted(slots)
    nid = int(g.swarm_next_unit_id)
    new = sim.swarm.spawn("larva", [(7 << 16) + 9], [(7 << 16) + 9], [0])
    assert int(new[0]) == 0
    assert int(swarm.swarm_column(g, "larva", "unit_id")[0]) == nid


def test_the_thermal_scenario_keeps_every_invariant():
    """T-P3-30: swarm.check_invariants(sim.gmap) == [] after every tick of the
    thermal scenario for 130 CPU ticks, and the scene reaches a death, a
    coma, a wall hit and a move (non-vacuity, G-d's tally).

    Must break on any lifecycle break of the law on the real tick."""
    bp.swarm_branch_tally(reset=True)
    sim = fab.swarm_thermal_scenario_sim()
    for t in range(130):
        _step(sim)
        assert swarm.check_invariants(sim.gmap) == [], t
    tally = bp.swarm_branch_tally()
    for name in ("died", "run_to_coma", "coma_held", "probe_solid", "moved", "alarm_on",
                 "tumble_taken", "pref_plus", "pref_minus"):
        assert tally[name] > 0, name


# ---------------------------------------------------------------------------
# T-P3-32
# ---------------------------------------------------------------------------

def test_the_invariant_position_rule_is_not_vacuous():
    """T-P3-32: one live larva at pos_x = w << 16 (and separately pos_y = -1)
    is reported naming the position; the uncorrupted fixture is clean.

    Must break if the position rule is dropped or bounded by the capacity
    instead of the grid."""
    wd = tk.larva_world(6, 7, 4)
    tk.place(wd, 0, 0, (2 << 16), (2 << 16), 0)
    assert swarm.check_invariants(tk.as_store_namespace(wd, 0)) == []
    bad = wd.copy()
    bad.pos_x[0, 0] = 7 << 16
    assert any("pos_x" in v for v in swarm.check_invariants(tk.as_store_namespace(bad, 0)))
    bad = wd.copy()
    bad.pos_y[0, 0] = -1
    assert any("pos_y" in v for v in swarm.check_invariants(tk.as_store_namespace(bad, 0)))


# ---------------------------------------------------------------------------
# T-P3-34, T-P3-35
# ---------------------------------------------------------------------------

def _packs_from(world):
    return [swarm.EnginePack(species="larva", columns=world.columns(),
                             high_water=world.high_water, tables=world.tables)]


def test_engine_presence_spans_every_env():
    """T-P3-34: a direct engine.step_swarm over hand-built n_env = 2 packs, env
    0 empty and env 1 one larva on a lethal tile, advances
    bp.swarm_cpu_calls() and returns one row with ROW_ENV == 1; with both
    envs empty it returns (0, ROW_WORDS) and advances nothing.

    Must break if presence is read from env 0 only."""
    eng = bp.PhysicsEngine()
    T = np.zeros((2, 6, 6), dtype=np.int32)
    T[1] = tk.DEFAULT_LARVA["T_lethal_hot_q"] + 1
    wd = tk.larva_world(6, 6, 4, 2, T=T)
    tk.place(wd, 1, 0, (2 << 16), (2 << 16), 0)
    dirty = {"larva": False}
    c0 = bp.swarm_cpu_calls()
    rows = eng.step_swarm(_packs_from(wd), wd.T, wd.solid, wd.env, dirty)
    assert bp.swarm_cpu_calls() == c0 + 1
    assert rows.shape == (1, swarm.ROW_WORDS) and int(rows[0, swarm.ROW_ENV]) == 1
    assert dirty["larva"] is True
    empty = tk.larva_world(6, 6, 4, 2, T=T)
    dirty = {"larva": False}
    rows = eng.step_swarm(_packs_from(empty), empty.T, empty.solid, empty.env, dirty)
    assert rows.shape == (0, swarm.ROW_WORDS)
    assert bp.swarm_cpu_calls() == c0 + 1
    assert dirty["larva"] is False


def test_the_host_path_marks_the_store_dirty_after_a_launch():
    """T-P3-35 (R9 / V7): after a CPU tick with a swarm present the species'
    host-dirty flag is True (the host store moved; a device copy is stale);
    after a swarm-free tick the flag is untouched (False on a fresh map).

    Must break if the host path leaves a stale device copy undetectable."""
    sim = fab.swarm_scenario_sim()
    sim.gmap.swarm_host_dirty["larva"] = False     # as if the device had just been fed
    _step(sim)
    assert sim.gmap.swarm_host_dirty["larva"] is True
    free = Simulation(_room(), seed=1, breach_physics=bp, enable_recorder=False)
    _step(free)
    assert free.gmap.swarm_host_dirty["larva"] is False


# ---------------------------------------------------------------------------
# T-P3-36
# ---------------------------------------------------------------------------

def test_the_bindings_refuse_mismatched_arrays_and_touch_nothing():
    """T-P3-36: for swarm_species_step and engine.step_swarm, a column of the
    wrong dtype, a non-contiguous column, a wrong shape, a table of the wrong
    width, or packs out of registry order each raise TypeError / ValueError
    and leave every input array byte-identical.

    Must break if forcecast silently copies an IN/OUT array (its writes
    would be lost) or a check runs after the law."""
    def fresh():
        wd = tk.larva_world(6, 6, 8, 1, p_base_q=ONE, p_alarm_q=ONE)
        for s in range(5):
            tk.place(wd, 0, s, (2 << 16) + 100 * s, (3 << 16), 1000 * s)
        return wd

    def snapshot(wd):
        return {k: v.copy() for k, v in vars(wd).items() if isinstance(v, np.ndarray)}

    def same(wd, snap):
        return all(np.array_equal(getattr(wd, k), v) for k, v in snap.items())

    breakers = []

    def wrong_dtype(wd):
        wd.pos_x = wd.pos_x.astype(np.int64)
    breakers.append(wrong_dtype)

    def non_contig(wd):
        big = np.zeros((1, 16), dtype=np.int32)
        big[:, ::2] = wd.heading
        wd.heading = big[:, ::2]
    breakers.append(non_contig)

    def wrong_shape(wd):
        wd.valid = np.zeros((1, 9), dtype=np.uint8)
    breakers.append(wrong_shape)

    def wrong_width(wd):
        wd.tables = wd.tables[:, :-1].copy()
    breakers.append(wrong_width)

    def solid_dtype(wd):
        wd.solid = wd.solid.astype(np.uint8)
    breakers.append(solid_dtype)

    for brk in breakers:
        wd = fresh()
        brk(wd)
        snap = snapshot(wd)
        with pytest.raises((TypeError, ValueError)):
            bp.swarm_species_step("larva", *wd.args())
        assert same(wd, snap), brk.__name__
        eng = bp.PhysicsEngine()
        with pytest.raises((TypeError, ValueError)):
            eng.step_swarm(_packs_from(wd), wd.T, wd.solid, wd.env, {"larva": False})
        assert same(wd, snap), brk.__name__
    wd = fresh()
    snap = snapshot(wd)
    bad = [swarm.EnginePack("notaspecies", wd.columns(), wd.high_water, wd.tables)]
    with pytest.raises(ValueError):
        bp.PhysicsEngine().step_swarm(bad, wd.T, wd.solid, wd.env, {"larva": False})
    with pytest.raises(ValueError):
        bp.swarm_species_step("notaspecies", *wd.args())
    assert same(wd, snap)


# ---------------------------------------------------------------------------
# T-P3-37 (v4.5): the write-set gate, borrowed from the shadow lockstep
# ---------------------------------------------------------------------------

def test_the_host_swarm_step_writes_only_the_swarm_store():
    """T-P3-37: over 60 CPU ticks of the thermal scenario, every GameMap
    attribute the host-path step_swarm changes is a present species' store
    column or its swarm_host_dirty key (cuda_unit_lockstep.gmap_state /
    changed_attrs, imported, never copied).

    Must break if the kernel writes any field, a counter it must not
    (high_water, swarm_next_unit_id: death leaves both), or another
    species' store."""
    from cuda_unit_lockstep import changed_attrs, gmap_state
    sim = fab.swarm_thermal_scenario_sim()
    g = sim.gmap
    runner = sim.physics_runner
    real = runner.engine
    changed_seen = set()

    class Proxy:
        def __getattr__(self, name):
            return getattr(real, name)

        def step_swarm(self, **kw):
            pre = gmap_state(g)
            rows = real.step_swarm(**kw)
            post = gmap_state(g)
            changed = changed_attrs(pre, post)
            allowed = {swarm.store_attr(sp, c.name) for sp in swarm.SWARM_SPECIES
                       if int(getattr(g, swarm.high_water_attr(sp))) > 0 for c in swarm.ROSTER}
            allowed |= {f"{swarm.HOST_DIRTY_ATTR}:{sp}" for sp in swarm.SWARM_SPECIES}
            bad = [c for c in changed if c not in allowed]
            assert not bad, f"the swarm step wrote {bad}"
            changed_seen.update(changed)
            return rows
    runner.engine = Proxy()
    for _ in range(60):
        g.swarm_host_dirty["larva"] = False
        _step(sim)
    assert swarm.store_attr("larva", "pos_x") in changed_seen
    assert f"{swarm.HOST_DIRTY_ATTR}:larva" in changed_seen
