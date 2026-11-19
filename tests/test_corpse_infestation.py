"""Corpse infestation (arc #63 P5b, docs/swarm_P5b_impl.md): the swarm's count
plane, the ``swarm_count`` coupling row into the corpse law
(``simulation.corpse``), the conversion door's second trigger and the corpse's
integrity. The CPU build's host path; the device forms are gated by
``tests/cuda_swarm_check.py`` (PART 2d), ``tests/cuda_unit_damage_check.py``
(the corpse scene) and ``tests/cuda_unit_kernels_check.py`` (the conversion).

Each test names the property it protects and the change that must break it.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_corpse_infestation.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import _swarm_testkit as tk  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import (ai_zombie, clearing, corpse, exchange, swarm,  # noqa: E402
                        swarm_fixed, unit_schema)
from simulation.damage import KINETIC, DamagePacket, apply_packet  # noqa: E402
from simulation.ruleset import ContinuousRealtime  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402

ONE = swarm_fixed.FP_ONE
HALF = ONE >> 1
TPS = 24


def _room(h, w, tile=0.333):
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    return LevelData(name="corpse_test", version="1", path=Path("."), tilemap=tm,
                     tile_size_m=tile, diffuse_path=Path("."))


def _sim(h=5, w=5, seed=1):
    """By default a 3 x 3 room: a 3 x 3 footprint at (1, 1) covers ALL of it,
    so every larva in the room is on the corpse whatever its walk. The
    continuous ruleset: the door runs at every tick's end."""
    return Simulation(_room(h, w), seed=seed, breach_physics=bp, enable_recorder=False,
                      ruleset=ContinuousRealtime())


def _step(sim, n=1):
    for _ in range(n):
        sim.set_paused(False)
        sim.step()


def _larvae(sim, tiles):
    xs = [(tx << 16) + HALF for tx, _ in tiles]
    ys = [(ty << 16) + HALF for _, ty in tiles]
    return sim.swarm.spawn("larva", xs, ys, [0] * len(tiles))


def _kill(u):
    """Kill ``u`` through the one damage door (an uncredited heat packet:
    never a converting death)."""
    apply_packet(u, DamagePacket(amount=10_000.0, dtype=KINETIC), None,
                 source="explosion")
    assert not u.alive


def _marine(sim, name="M", x=1.0, y=1.0):
    u = Unit(name, x=x, y=y, team=0)
    sim.add_unit(u)
    return u


def _zombie(sim, name="Z", x=1.0, y=1.0):
    z = Unit(name, x=x, y=y, team=255)
    sim.add_unit(z)
    assert z.is_zombie
    return z


def _ticks_to_rise(sim, u, limit):
    for t in range(1, limit + 1):
        _step(sim)
        if u.alive:
            return t
    return None


def _law_ticks(n, limit=20000):
    """The level of one corpse under ``n`` swarm units, tick by tick, through
    the law itself (a bare row); returns (ticks to the threshold or None,
    the level after ``limit`` ticks)."""
    s = corpse.get_settings()
    u = SimpleNamespace(infest_q=0, corpse_integrity_q=s.integrity_q, alive=False)
    hit = None
    for t in range(1, limit + 1):
        corpse.infest(u, n, s)
        if hit is None and u.infest_q >= s.threshold_q:
            hit = t
    return hit, u.infest_q


# ---------------------------------------------------------------------------
# The count plane (the swarm side)
# ---------------------------------------------------------------------------

def test_the_count_plane_counts_living_non_coma_units_per_tile():
    """Property: after a swarm step the count plane holds, on every tile,
    exactly the number of valid larvae not in coma on it (an integer sum,
    whatever the slot order), the garbage it held before is gone, and a coma
    larva, a larva the step killed and a freed slot count nothing. Breaks if
    coma or dead slots count, if the builder adds to last tick's plane, or if
    the count is skipped on a tile with larvae."""
    def world(order):
        T = np.zeros((1, 6, 6), dtype=np.int32)
        T[0, 4, 4] = -30 * ONE                                   # a coma tile
        T[0, 1, 4] = tk.DEFAULT_LARVA["T_lethal_hot_q"] + 1      # a lethal tile
        wd = tk.larva_world(6, 6, 8, 1, T=T)
        spots = [((1 << 16) + HALF, (1 << 16) + HALF, swarm.AI_RUN)] * 3 + [
            ((3 << 16) + HALF, (2 << 16) + HALF, swarm.AI_RUN),
            ((4 << 16) + HALF, (4 << 16) + HALF, swarm.AI_COMA),
            ((4 << 16) + HALF, (1 << 16) + HALF, swarm.AI_RUN)]
        for slot, k in enumerate(order):
            x, y, st = spots[k]
            tk.place(wd, 0, slot, x, y, 0, state=st, uid=100 + k)
        wd.high_water[0] = 7                                     # slot 6: freed
        bp.PhysicsEngine().step_swarm(
            [swarm.EnginePack("larva", wd.columns(), wd.high_water, wd.tables)],
            wd.T, wd.solid, wd.env, {"larva": False}, wd.bite, wd.count)
        return wd
    wd = world([0, 1, 2, 3, 4, 5])
    expect = np.zeros((6, 6), dtype=np.int64)
    for s in range(7):
        if wd.valid[0, s] and wd.ai_state[0, s] != swarm.AI_COMA:
            expect[wd.pos_y[0, s] >> 16, wd.pos_x[0, s] >> 16] += 1
    assert np.array_equal(wd.count[0], expect)
    assert int(wd.count[0].sum()) == 4                    # 3 + 1; coma, dead, freed: 0
    assert int(wd.count[0, 4, 4]) == 0 and int(wd.count[0, 1, 4]) == 0
    # the count plane counts exactly the larvae the bite plane bites with
    q = int(tk.DEFAULT_LARVA["bite_q"])
    assert np.array_equal(wd.bite[0], wd.count[0] * q)
    # order-free: the same larvae in other slots give the same plane
    wd2 = world([5, 3, 1, 4, 0, 2])
    assert np.array_equal(wd2.count[0], wd.count[0])


def test_the_count_plane_is_rebuilt_every_tick_and_a_swarm_free_map_never_writes_it():
    """Property: in a sim the plane equals the host's own scan of the store
    every tick (living, non-coma larvae per tile), and a map with no swarm
    never writes it (all zero, every tick). Breaks if the plane accumulates,
    if the host step does not zero it, or if a dormant step writes it."""
    sim = _sim(12, 12)
    _larvae(sim, [(3, 3)] * 5 + [(8, 6)] * 2)
    g = sim.gmap
    for _ in range(30):
        _step(sim)
        hw = int(getattr(g, swarm.high_water_attr("larva")))
        cols = {c: swarm.swarm_column(g, "larva", c)[:hw] for c in ("valid", "ai_state",
                                                                    "pos_x", "pos_y")}
        live = (cols["valid"] != 0) & (cols["ai_state"] != swarm.AI_COMA) & \
            (cols["ai_state"] != swarm.AI_DEAD)
        expect = np.zeros(swarm.count_plane(g).shape, dtype=np.int64)
        np.add.at(expect, (cols["pos_y"][live] >> 16, cols["pos_x"][live] >> 16), 1)
        assert np.array_equal(swarm.count_plane(g), expect)
        assert int(swarm.count_plane(g).sum()) == int(live.sum()) == 7
    bare = _sim(12, 12)
    for _ in range(5):
        _step(bare)
        assert not swarm.count_plane(bare.gmap).any()


@pytest.mark.parametrize("attr", ["swarm_count"])
def test_the_count_plane_is_resident_and_not_digested(attr):
    """Property: the plane is a resident synced field (the device step writes
    it, the exit download carries it while a swarm is present) and enters no
    digest. Breaks if it joins the field digest (a membership change: a
    version bump and every golden) or leaves the resident set."""
    from simulation.gamemap import GameMap
    import field_digest
    assert attr in swarm.SWARM_PLANE_NAMES
    assert attr in GameMap._RESIDENT_SYNCED
    assert attr not in {f for f in field_digest.DIGEST_FIELDS}


# ---------------------------------------------------------------------------
# The law
# ---------------------------------------------------------------------------

def test_the_settings_are_the_numbers_quantized_once():
    """Property: the four dials reach the law as ONE rounding each: the rise
    is round(rate x ONE / tps) (55 at 0.02 / s), the decay round(decay x 2^24
    / tps) (34953 at 0.05 / s), the threshold and the integrity Q16. Breaks if
    a dial is rounded twice or read in another currency."""
    s = corpse.get_settings()
    assert s.tps == TPS
    assert s.rise_q == round(0.02 * ONE / TPS) == 55
    assert s.decay_a == round(0.05 * clearing.RATE_ONE / TPS) == 34953
    assert s.threshold_q == ONE and s.integrity_q == 200 * ONE
    for rate in (0.0, 0.013, 0.05, 1.7, 9.0):
        exact = rate * clearing.RATE_ONE / TPS
        assert abs(clearing.rate_per_tick(rate, TPS) - exact) <= 0.5


def test_the_config_door_refuses_a_bad_table():
    """Property: a missing, unknown or out-of-range [corpse] key is refused
    naming the key. Breaks if the door clamps or ignores a typo."""
    good = dict(infest_rate=0.02, infest_decay=0.05, infest_threshold=1.0,
                integrity=200.0)
    clock = SimpleNamespace(ticks_per_second=TPS)

    def cfg(**over):
        d = dict(good)
        d.update(over)
        d = {k: v for k, v in d.items() if v is not None}
        return SimpleNamespace(corpse=SimpleNamespace(**d), clock=clock)
    assert corpse.settings_from_config(cfg()).rise_q == 55
    for over, word in ((dict(integrity=None), "integrity"),
                       (dict(infest_ratee=1.0), "infest_ratee"),
                       (dict(infest_rate=-1.0), "infest_rate"),
                       (dict(infest_threshold=0.0), "infest_threshold"),
                       (dict(integrity=0.0), "integrity"),
                       (dict(infest_decay=float("nan")), "infest_decay"),
                       (dict(infest_decay=30.0), "infest_decay")):
        with pytest.raises(ValueError, match=word):
            corpse.settings_from_config(cfg(**over))


def test_the_level_rises_with_n_and_decays_without():
    """Property: one tick under n swarm units raises an empty level by n x
    rise exactly; more units raise it faster; with nobody on it the level
    decays and returns to exactly 0. Breaks if the rise ignores n, if the
    drain reads the level after the rise, or if a level stalls above 0."""
    s = corpse.get_settings()
    u = SimpleNamespace(infest_q=0, corpse_integrity_q=s.integrity_q, alive=False)
    corpse.infest(u, 3, s)
    assert u.infest_q == 3 * s.rise_q
    # the drain is on the level BEFORE the rise
    x = u.infest_q
    corpse.infest(u, 3, s)
    assert u.infest_q == x - clearing.drain(x, s.decay_a, clearing.RATE_SHIFT) + 3 * s.rise_q
    one, three = _law_ticks(1, 240)[1], _law_ticks(3, 240)[1]
    assert three > one > 0
    for _ in range(240 * 60):
        corpse.infest(u, 0, s)
        if u.infest_q == 0:
            break
    assert u.infest_q == 0


def test_the_steady_states_and_the_three_times():
    """Property (the first numbers' promise): one larva settles near 0.4 and
    NEVER raises a corpse; three settle near 1.2 and raise it in about half a
    minute; ten raise it in about six seconds. Breaks if a dial, the rounding
    or the law moves those times out of their bands."""
    s = corpse.get_settings()
    hit1, level1 = _law_ticks(1, TPS * 600)
    assert hit1 is None
    assert 0.38 <= level1 / ONE <= 0.43
    _, level3 = _law_ticks(3, TPS * 600)
    assert 1.17 <= level3 / ONE <= 1.25
    hit3, _ = _law_ticks(3)
    hit10, _ = _law_ticks(10)
    assert 25 * TPS <= hit3 <= 40 * TPS            # continuous law: ln 6 / 0.05 = 35.8 s
    assert 5 * TPS <= hit10 <= 7 * TPS             # continuous law: ln(4/3) / 0.05 = 5.75 s
    assert s.threshold_q == ONE


def test_a_destroyed_corpse_holds_its_level_at_zero():
    """Property: a corpse whose integrity is gone gains no level under any
    swarm and a level it held drops to 0. Breaks if the law runs on a
    destroyed corpse."""
    s = corpse.get_settings()
    u = SimpleNamespace(infest_q=5 * ONE, corpse_integrity_q=0, alive=False)
    corpse.infest(u, 50, s)
    assert u.infest_q == 0
    corpse.infest(u, 50, s)
    assert u.infest_q == 0 and not corpse.rises(u, s)


def test_a_level_past_the_proved_range_is_refused():
    """Property: a level or a rise past corpse.LEVEL_MAX is refused by name,
    never clamped or wrapped. Breaks if the bound is dropped."""
    s = corpse.get_settings()
    u = SimpleNamespace(infest_q=corpse.LEVEL_MAX + 1, corpse_integrity_q=ONE, alive=False)
    with pytest.raises(corpse.CorpseLevelRange):
        corpse.infest(u, 0, s)
    u = SimpleNamespace(infest_q=0, corpse_integrity_q=ONE, alive=False)
    with pytest.raises(corpse.CorpseLevelRange):
        corpse.infest(u, corpse.LEVEL_MAX // s.rise_q + 1, s)


# ---------------------------------------------------------------------------
# In the sim: the row, the death, the door
# ---------------------------------------------------------------------------

def test_a_marine_corpse_rises_at_the_threshold_through_the_door():
    """Property: ten larvae on a marine's corpse raise its level to the
    threshold in about six seconds, and at the tick's end the ONE conversion
    door turns it into a zombie: alive, on the zombie team, the zombie class's
    health, its level back to 0 and its integrity full. Breaks if the row is
    not wired, if the door ignores the infestation trigger, or if the door
    does not reset the level."""
    sim = _sim()
    m = _marine(sim)
    _kill(m)
    assert m.infest_q == 0 and m.corpse_integrity_q == corpse.full_integrity_q()
    _larvae(sim, [(2, 2)] * 10)
    t = _ticks_to_rise(sim, m, 10 * TPS)
    assert t is not None and 5 * TPS <= t <= 7 * TPS, t
    assert m.alive and m.is_zombie and m.team == 255
    assert m.infest_q == 0 and m.corpse_integrity_q == corpse.full_integrity_q()
    assert m.name == "Z-M"


def test_one_larva_never_raises_a_corpse():
    """Property: one larva on a corpse for two minutes never raises it (its
    steady level is below the threshold). Breaks if the threshold or the law
    lets a lone larva raise the dead."""
    sim = _sim()
    m = _marine(sim)
    _kill(m)
    _larvae(sim, [(2, 2)])
    assert _ticks_to_rise(sim, m, 120 * TPS) is None
    assert 0.3 * ONE < m.infest_q < ONE


def test_a_zombie_corpse_rises_and_a_risen_zombie_can_die_and_rise_again():
    """Property: ANY corpse rises, a zombie's included, and any number of
    times: a dead zombie on larvae rises; killed again it rises again, each
    time with its level reset and nothing counting the rises. Breaks if the
    door keeps its old `not is_zombie` gate on the infestation trigger or if
    a rise leaves a mark that stops the next."""
    sim = _sim()
    z = _zombie(sim)
    _kill(z)
    _larvae(sim, [(2, 2)] * 10)
    for _ in range(3):
        t = _ticks_to_rise(sim, z, 10 * TPS)
        assert t is not None and z.alive and z.is_zombie
        assert z.infest_q == 0
        assert z.name == "Z"                     # a zombie is not renamed
        _kill(z)
        assert z.infest_q == 0


def test_a_destroyed_corpse_never_rises_by_either_trigger():
    """Property: a corpse whose integrity is gone rises neither when the
    larvae swarm it nor when a zombie killed it: its level stays 0 and the
    door passes it by. Breaks if the door skips the integrity check on
    either trigger."""
    sim = _sim(7, 9)
    a = _marine(sim, "A", 1.0, 1.0)
    b = _marine(sim, "B", 5.0, 1.0)
    _kill(a)
    _kill(b)
    a.corpse_integrity_q = 0                 # fixture: the corpse destroyed
    b.corpse_integrity_q = 0
    b.killed_by_zombie = True                # a zombie's kill
    _larvae(sim, [(2, 2)] * 12)
    _step(sim, 20 * TPS)
    assert not a.alive and not b.alive
    assert a.infest_q == 0
    ai_zombie.convert_marines_to_zombies([a, b])
    assert not a.alive and not b.alive and not a.is_zombie and not b.is_zombie
    # the same pair, whole: both rise
    a.corpse_integrity_q = b.corpse_integrity_q = corpse.full_integrity_q()
    a.infest_q = corpse.get_settings().threshold_q
    ai_zombie.convert_marines_to_zombies([a, b])
    assert a.alive and b.alive and a.is_zombie and b.is_zombie


def test_a_living_units_level_never_moves():
    """Property: a living marine standing among larvae for ten seconds keeps
    its level at 0 (the law runs on corpses only). Breaks if the row skips
    its `alive` gate."""
    sim = _sim(12, 12)
    m = _marine(sim, "M", 4.0, 4.0)
    _larvae(sim, [(5, 5)] * 20)
    for _ in range(10 * TPS):
        _step(sim)
        if not m.alive:
            break
        assert m.infest_q == 0


def test_death_sets_the_corpse_and_both_columns_hash_only_when_moved():
    """Property: a death sets the level to 0 and the integrity full (the
    columns' defaults, so a death without larvae hashes exactly as before),
    and a unit's digest record names neither column until the corpse law
    moves one. Breaks if a death writes a non-default value or a column is
    hashed at its default."""
    sim = _sim()
    m = _marine(sim)
    m.corpse_integrity_q = 5                 # fixture: a stale value before death
    m.infest_q = 3
    _kill(m)
    assert m.infest_q == 0 and m.corpse_integrity_q == corpse.full_integrity_q()
    rec = unit_schema.digest_record(m)
    assert "infest_q" not in rec and "corpse_integrity_q" not in rec
    _larvae(sim, [(2, 2)] * 3)
    _step(sim)
    assert m.infest_q > 0 and "infest_q" in unit_schema.digest_record(m)


def test_the_conversion_door_trigger():
    """Property: THE door's trigger is (a non-zombie a zombie killed) OR (an
    infested corpse at the threshold), never a destroyed corpse, never a
    living unit; a zombie kill rises at once whatever its level. Breaks if
    the trigger loses a branch or gains one."""
    s = corpse.get_settings()
    full = corpse.full_integrity_q()

    def u(**kw):
        d = dict(alive=False, is_zombie=False, killed_by_zombie=False, infest_q=0,
                 corpse_integrity_q=full)
        d.update(kw)
        return SimpleNamespace(**d)
    assert ai_zombie.rises_as_zombie(u(killed_by_zombie=True))
    assert ai_zombie.rises_as_zombie(u(infest_q=s.threshold_q))
    assert ai_zombie.rises_as_zombie(u(is_zombie=True, infest_q=s.threshold_q))
    assert not ai_zombie.rises_as_zombie(u(infest_q=s.threshold_q - 1))
    assert not ai_zombie.rises_as_zombie(u(is_zombie=True, killed_by_zombie=True))
    assert not ai_zombie.rises_as_zombie(u(alive=True, infest_q=s.threshold_q))
    assert not ai_zombie.rises_as_zombie(u(killed_by_zombie=True, corpse_integrity_q=0))
    assert not ai_zombie.rises_as_zombie(u(infest_q=s.threshold_q, corpse_integrity_q=0))


def test_same_seed_same_rise():
    """Property: two runs of the same scene raise the corpse on the same tick
    with the same level history. Breaks if anything in the row or the plane
    depends on the order or on state outside the sim."""
    def run():
        sim = _sim(7, 7)
        m = _marine(sim, "M", 2.0, 2.0)
        _kill(m)
        _larvae(sim, [(1, 1)] * 4 + [(5, 5)] * 4 + [(3, 3)] * 3)
        out = []
        for _ in range(40 * TPS):
            _step(sim)
            out.append((int(m.infest_q), m.alive))
            if m.alive:
                break
        return out
    assert run() == run()


def test_the_coupling_row_is_registered():
    """Property: the infestation is ONE row of the coupling table, a footprint
    sum of the swarm's count plane answered by apply_corpse_infestation,
    after the bite row. Breaks if the corpse is wired outside the table."""
    rows = exchange.COUPLING_TABLE
    names = [r.field for r in rows]
    i = names.index("swarm_count")
    assert rows[i].reduction == "sum"
    assert rows[i].response is exchange.apply_corpse_infestation
    assert i > names.index("swarm_bite_dose")


def test_only_the_corpse_module_writes_the_two_columns():
    """Property: nothing in src/ but simulation/corpse.py writes infest_q or
    corpse_integrity_q (the one writer). Breaks if a second writer appears."""
    pat = re.compile(r"\.(infest_q|corpse_integrity_q)\s*(=[^=]|\+=|-=)")
    bad = []
    for p in (ROOT / "src").rglob("*.py"):
        if p.name == "corpse.py":
            continue
        for k, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line):
                bad.append(f"{p.relative_to(ROOT)}:{k}: {line.strip()}")
    assert not bad, bad
