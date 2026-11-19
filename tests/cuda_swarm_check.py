"""arc #63 P3 -- THE 3-PART CUDA GATE of the larva kernel (docs/swarm_P3_impl.md
§11 "The 3-part CUDA gate", v4.7 G-a / G-b / G-d). Run through
tests/test_cuda_swarm.py (the cuda_harness subprocess); prints
``SWARM_P3_RESULT: PASS`` when every part holds.

PART 1 -- FORCED-BRANCH FUZZ, byte identity: every case runs through the CPU
entry (``swarm_species_step``, THE reference), the per-call CUDA wrapper
(``cuda_swarm_species_step``) and the RESIDENT launch core on gate-owned
device arrays (``cuda_swarm_resident_probe``); all 9 columns, the FULL record
arrays (pre-filled with identical garbage) and the returned counts must be
byte-identical. Every env's fields are drawn independently, including cases
where env e's lethal / cold tiles are env e+1's comfortable ones (G-a);
padding is LIVE-LOOKING (valid rows on lethal tiles past a short env's hw,
past hw up to capacity, garbage record slots, and invalid slots below hw
whose other columns are set) and must come out byte-unchanged with no death
(G-b). Breaks on a kernel or wrapper that differs from the CPU law, a bound
other than hw[e], a dropped valid check, a record copied beyond hw.

PART 2 -- 130-TICK LOCKSTEP on two GameMaps' stores driven by the facade, the
synthetic fields standalone arrays owned by this script (a cold band rising
through the hysteresis band, a hot front sweeping in, interior walls): store
A on the CPU entry, store B on the per-call wrapper, byte identity of the
store and the records every tick, every transition asserted non-vacuous, and
a freed slot reused by the next spawn with a new id.

PART 2b -- THE SIM PATHS, three-way: swarm_thermal_scenario_sim on the CPU
path, the per-call path (only set_swarm_backend on) and the resident path
(run_resident_batch at N = 1), with a spawn, a wall destroyed in front of a
wall larva and a reclaim scripted on all three: per tick the swarm section
bytes, the death events and the tick digest equal, every invariant clean;
the per-call and resident counters advance on every present tick; the swarm
store rides the entry upload exactly on the first tick and the tick after
each facade write. A swarm-free resident leg moves none of the swarm
counters.

PART 2c -- THE BITE, three-way (arc #63 P5a): the thermal scenario plus three
marines -- one over the spread larvae, one over the cold patch's coma larvae,
one at 0.25 HP over a pile of eight larvae -- on the CPU, per-call and resident
paths: per tick the bite-dose plane's bytes, every unit's poison levels and
cause, the hit / kill events and the tick digest equal. Non-vacuous: the plane
is non-zero, the first marine is dosed, the pile kills the third with cause
larva_bite and no credited unit, and the plane is zero under every coma larva
standing alone. Breaks on a device bite builder that differs from the host
one, a plane not rebuilt from zero, an exit download without the plane, or a
coma / dead slot that bites on one path.

PART 2d -- THE CORPSES, three-way (arc #63 P5b): the thermal scenario (the
continuous ruleset, so the door runs every tick) plus a marine's corpse, a
zombie's corpse and a destroyed corpse (integrity 0) under larvae piles, on
the CPU, per-call and resident paths: per tick the count plane's bytes, every
unit's (alive, zombie, level, integrity) and the tick digest equal; non-vacuous:
the plane is non-zero, the count plane equals a host scan of the store (living,
non-coma larvae per tile), the marine's and the zombie's corpses rise, the
destroyed one never does and its level stays 0. Breaks on a device count
builder that differs from the host (a coma or dead slot counted, the plane not
zeroed, not downloaded) or a path whose corpse level differs.

PART 3 -- the canonical golden on the CUDA build's CPU path.
"""
from __future__ import annotations

import sys

import numpy as np

import breach_physics as bp  # noqa: F401  (the CUDA build, loaded first)

import _swarm_testkit as tk
import cuda_s8a_check as s8a
from simulation import swarm, swarm_fixed

ONE = swarm_fixed.FP_ONE
PI = swarm_fixed.PI_Q16
I32MAX = (1 << 31) - 1
LETHAL = tk.DEFAULT_LARVA["T_lethal_hot_q"]
CTMIN = tk.DEFAULT_LARVA["T_ctmin_q"]
CEXIT = tk.DEFAULT_LARVA["T_coma_exit_q"]
COLD, COMFY, HOT = CTMIN - 5 * ONE, 0, LETHAL + 3 * ONE


def _run3(world):
    """The three loops on identical copies; returns (ok, cpu world)."""
    a, b, c = world.copy(), world.copy(), world.copy()
    na = int(bp.swarm_species_step("larva", *a.args()))
    nb = int(bp.cuda_swarm_species_step("larva", *b.args()))
    nc = int(bp.cuda_swarm_resident_probe("larva", *c.args()))
    ok = na == nb == nc
    for name in [col.name for col in swarm.ROSTER] + ["death"]:
        x, y, z = getattr(a, name), getattr(b, name), getattr(c, name)
        if not (np.array_equal(x, y) and np.array_equal(x, z)):
            ok = False
            print(f"    {name} differs: cpu/percall {int(np.count_nonzero(x != y))} "
                  f"cpu/resident {int(np.count_nonzero(x != z))}")
    if not ok:
        print(f"    counts cpu {na} percall {nb} resident {nc}")
    return ok, a


def _inv_ok(world, out):
    """check_invariants on each env's [0, hw) (padding past hw zeroed in a copy)."""
    for e in range(world.n_env):
        w = out.copy()
        hw = int(w.high_water[e])
        for col in swarm.ROSTER:
            getattr(w, col.name)[e, hw:] = 0
        v = swarm.check_invariants(tk.as_store_namespace(w, e))
        if v:
            print(f"    env {e}: {v[:3]}")
            return False
    return True


def _padding_ok(world, out):
    """Every byte past hw[e] (columns and records) is unchanged."""
    for e in range(world.n_env):
        hw = int(world.high_water[e])
        for name in [col.name for col in swarm.ROSTER] + ["death"]:
            if not np.array_equal(getattr(world, name)[e, hw:], getattr(out, name)[e, hw:]):
                print(f"    env {e}: {name} past hw={hw} changed")
                return False
    return True


def _rand_world(rng, forcer=None):
    n_env = int(rng.integers(1, 4))
    shape = rng.integers(0, 4)
    if shape == 0:
        h, w = 1, int(rng.integers(1, 30))
    elif shape == 1:
        h, w = int(rng.integers(1, 30)), 1
    else:
        h, w = int(rng.integers(2, 24)), int(rng.integers(2, 24))
    max_units = int(rng.integers(1, 301))
    # G-a: per-env fields; env e's class grid is env 0's rotated, so a lethal
    # tile in one env is a comfortable one in the next.
    cls = rng.integers(0, 3, size=(h, w))
    levels = np.array([COLD, COMFY, HOT], dtype=np.int64)
    T = np.zeros((n_env, h, w), dtype=np.int32)
    for e in range(n_env):
        base = levels[(cls + e) % 3]
        jitter = rng.integers(-3 * ONE, 3 * ONE, size=(h, w))
        straddle = rng.choice([CTMIN - 1, CTMIN, CEXIT, CEXIT + 1, LETHAL, LETHAL + 1],
                              size=(h, w))
        pick = rng.random((h, w))
        T[e] = np.where(pick < 0.3, straddle, base + jitter).astype(np.int32)
    smode = rng.integers(0, 4)
    solid = np.zeros((n_env, h, w), dtype=bool)
    for e in range(n_env):
        if smode == 1:
            solid[e] = True
        elif smode >= 2:
            solid[e] = rng.random((h, w)) < 0.25
    vals = {
        "step_q": [int(rng.choice([1, ONE - 1, int(rng.integers(1, ONE))])) for _ in range(n_env)],
        "turn_bias_q": [int(rng.choice([0, ONE, int(rng.integers(0, ONE))])) for _ in range(n_env)],
        "sweep_angle_q": [int(rng.choice([1, 102944, int(rng.integers(1, 102945))]))
                          for _ in range(n_env)],
    }
    tmin = [int(rng.integers(0, PI + 1)) for _ in range(n_env)]
    span = [int(rng.choice([0, PI - t, int(rng.integers(0, PI - t + 1))])) for t in tmin]
    full = rng.random() < 0.2
    vals["turn_min_q"] = [0 if full else t for t in tmin]
    vals["turn_max_q"] = [PI if full else t + s for t, s in zip(tmin, span)]
    pb = [int(rng.choice([0, ONE, int(rng.integers(0, ONE + 1))])) for _ in range(n_env)]
    vals["p_base_q"] = pb
    vals["p_alarm_q"] = [int(rng.integers(p, ONE + 1)) for p in pb]
    if forcer == "tumble":
        vals["p_base_q"] = vals["p_alarm_q"] = [ONE] * n_env
    if forcer == "blocked":
        solid[:] = True
    if forcer == "die":
        T[:] = LETHAL + ONE
    if forcer == "coma":
        T[:] = CTMIN - ONE
    tick = int(rng.choice([0, (1 << 32) - 1, int(rng.integers(0, 1 << 32))]))
    seed = (int(rng.choice([0, (1 << 32) - 1, int(rng.integers(0, 1 << 32))])),
            int(rng.choice([0, (1 << 32) - 1, int(rng.integers(0, 1 << 32))])))
    wd = tk.larva_world(h, w, max_units, n_env, T=T, solid=solid, tick=tick, seed=seed, **vals)
    hw = [int(rng.choice([0, max_units, int(rng.integers(0, max_units + 1))]))
          for _ in range(n_env)]
    if forcer == "hw0":
        hw = [0] * n_env
    for e in range(n_env):
        wd.next_unit_id[e] = int(rng.choice([1, I32MAX - 2 * max_units - 5]))
        for s in range(hw[e]):
            if rng.random() < 0.15:
                continue                                   # an empty slot below hw
            x = int(rng.choice([0, (w << 16) - 1, int(rng.integers(0, w << 16)),
                                int(rng.integers(0, w)) << 16]))
            y = int(rng.choice([0, (h << 16) - 1, int(rng.integers(0, h << 16)),
                                int(rng.integers(0, h)) << 16]))
            hd = int(rng.choice([-PI + 1, PI, int(rng.integers(-PI + 1, PI + 1))]))
            st = int(rng.choice([swarm.AI_RUN, swarm.AI_EAT, swarm.AI_COMA]))
            dist = int(rng.choice([0, (1 << 32) - 1 - int(rng.integers(0, 70000)),
                                   int(rng.integers(0, 1 << 32))]))
            tk.place(wd, e, s, x, y, hd, state=st, dist=dist)
        wd.high_water[e] = hw[e]
        if hw[e] == 0:
            wd.next_unit_id[e] = swarm.FIRST_UNIT_ID
    # G-b: LIVE-LOOKING padding past hw[e] (valid, on a lethal / cold tile)
    for e in range(n_env):
        for s in range(hw[e], max_units):
            if rng.random() < 0.7:
                wd.valid[e, s] = 1
                wd.ai_state[e, s] = swarm.AI_RUN
                wd.pos_x[e, s] = int(rng.integers(0, w << 16))
                wd.pos_y[e, s] = int(rng.integers(0, h << 16))
                wd.heading[e, s] = int(rng.integers(-PI + 1, PI + 1))
                wd.unit_id[e, s] = 1 + s
                wd.hp[e, s] = ONE
                wd.dist_walked[e, s] = int(rng.integers(0, 1000))
    if forcer == "garbage":
        for col in swarm.ROSTER:
            arr = getattr(wd, col.name)
            for e in range(n_env):
                arr[e, hw[e]:] = rng.integers(0, 200, size=arr[e, hw[e]:].shape).astype(arr.dtype)
    if forcer == "invalid_live":
        # valid = 0 below hw with every other column live: L0 must skip it
        for e in range(n_env):
            for s in range(0, hw[e], 3):
                wd.valid[e, s] = 0
                wd.pos_x[e, s] = 5
                wd.unit_id[e, s] = 99
                wd.ai_state[e, s] = swarm.AI_RUN
    return wd


def part1() -> bool:
    print("PART 1: forced-branch fuzz -- CPU entry vs per-call vs resident core, byte identity")
    rng = np.random.default_rng(20261115)
    bp.swarm_branch_tally(reset=True)
    c0 = bp.swarm_cuda_calls()
    ok = True
    cases = [None] * 160 + ["die"] * 6 + ["coma"] * 6 + ["blocked"] * 6 + ["tumble"] * 6 + \
        ["hw0"] * 4 + ["garbage"] * 6 + ["invalid_live"] * 6
    n_present = 0
    for i, f in enumerate(cases):
        wd = _rand_world(rng, f)
        same, out = _run3(wd)
        if not same:
            print(f"  case {i} ({f}): the three loops differ")
            ok = False
            continue
        if f not in ("garbage", "invalid_live") and not _inv_ok(wd, out):
            print(f"  case {i} ({f}): an invariant broke")
            ok = False
        if not _padding_ok(wd, out):
            print(f"  case {i} ({f}): padding was touched")
            ok = False
        if f == "invalid_live":
            for e in range(wd.n_env):
                for s in range(0, int(wd.high_water[e]), 3):
                    if int(out.pos_x[e, s]) != 5 or int(out.unit_id[e, s]) != 99:
                        print(f"  case {i}: an invalid slot was processed")
                        ok = False
        if f == "hw0":
            for name in [col.name for col in swarm.ROSTER] + ["death"]:
                if not np.array_equal(getattr(out, name), getattr(wd, name)):
                    print(f"  case {i}: hw = 0 touched {name}")
                    ok = False
        if f == "die":
            for e in range(wd.n_env):
                hw = int(wd.high_water[e])
                if (out.valid[e, :hw] != 0).any():
                    print(f"  case {i}: a larva survived the lethal field")
                    ok = False
        n_present += int(wd.high_water.max() > 0)
    calls = bp.swarm_cuda_calls() - c0
    if calls != n_present:
        print(f"  swarm_cuda_calls advanced {calls}, expected {n_present} (present cases)")
        ok = False
    print(f"  {len(cases)} cases, {n_present} with work: " + ("byte-identical" if ok else "FAILED"))
    return ok


def part2() -> bool:
    print("PART 2: 130-tick lockstep, two stores driven by the facade, script-owned fields")
    from level_loader import LevelData
    from pathlib import Path
    from simulation.gamemap import GameMap
    H, W = 20, 30
    tm = np.ones((H, W), dtype=np.int32)
    tm[1:H - 1, 1:W - 1] = 4
    lv = LevelData(name="swarm_p2", version="1", path=Path("."), tilemap=tm, tile_size_m=1.0,
                   diffuse_path=Path("."))
    gA, gB = GameMap(lv), GameMap(lv)
    fA, fB = swarm.SwarmUnits(gA), swarm.SwarmUnits(gB)
    solid = np.zeros((1, H, W), dtype=bool)
    solid[0, 0, :] = solid[0, -1, :] = solid[0, :, 0] = solid[0, :, -1] = True
    solid[0, 5:15, 15] = True
    solid[0, 8, 4:10] = True
    rng = np.random.default_rng(2)
    xs, ys, hs = [], [], []
    for _ in range(90):
        tx, ty = int(rng.integers(1, W - 1)), int(rng.integers(1, H - 1))
        if solid[0, ty, tx]:
            continue
        xs.append((tx << 16) + int(rng.integers(ONE))); ys.append((ty << 16) + int(rng.integers(ONE)))
        hs.append(int(rng.integers(-PI + 1, PI + 1)))
    fA.spawn("larva", xs, ys, hs)
    fB.spawn("larva", xs, ys, hs)
    tables = tk.larva_tables(1, step_q=3000, p_base_q=2000, p_alarm_q=9000)
    env = np.zeros((1, swarm.ENV_WORDS), dtype=np.uint32)
    env[0, 1], env[0, 2] = 0xC0FFEE, 0xBEEF
    death_a = np.full((1, gA.swarm_larva_valid.shape[0], 4), -3, np.int32)
    death_b = death_a.copy()
    counts = dict(tumble=0, wall=0, run_coma=0, coma_held=0, run_held=0, coma_run=0, kill=0,
                  reuse=0)
    ok = True
    bp.swarm_branch_tally(reset=False)
    for t in range(130):
        T = np.zeros((1, H, W), dtype=np.int32)
        T[0, :, 1:12] = int(-30 * ONE + t * 0.3 * ONE)          # the cold band rising
        T[0, :, max(1, 29 - t // 5):] = LETHAL + ONE             # the hot front
        T[0, 15:19, 12:22] = (CTMIN + CEXIT) // 2                # a band RUN larvae hold in
        env[0, 0] = t
        cols_a = tuple(swarm.swarm_column(gA, "larva", c.name).reshape(1, -1) for c in swarm.ROSTER)
        cols_b = tuple(swarm.swarm_column(gB, "larva", c.name).reshape(1, -1) for c in swarm.ROSTER)
        hw = np.array([int(gA.swarm_larva_high_water)], np.int32)
        pre = {c.name: swarm.swarm_column(gA, "larva", c.name).copy() for c in swarm.ROSTER}
        na = bp.swarm_species_step("larva", cols_a, hw, T, solid, env, tables, death_a)
        nb = bp.cuda_swarm_species_step("larva", cols_b, hw, T, solid, env, tables, death_b)
        same = na == nb and np.array_equal(death_a, death_b) and all(
            np.array_equal(a, b) for a, b in zip(cols_a, cols_b))
        if not same:
            print(f"  tick {t}: store / records differ")
            ok = False
            break
        if swarm.check_invariants(gA) or swarm.check_invariants(gB):
            print(f"  tick {t}: invariants {swarm.check_invariants(gA)[:2]}")
            ok = False
        post = {c.name: swarm.swarm_column(gA, "larva", c.name) for c in swarm.ROSTER}
        for s in range(int(hw[0])):
            if pre["valid"][s] == 0:
                continue
            tt = int(T[0, pre["pos_y"][s] >> 16, pre["pos_x"][s] >> 16])
            st0, st1 = int(pre["ai_state"][s]), int(post["ai_state"][s])
            if post["valid"][s] == 0:
                counts["kill"] += 1
                continue
            moved = (post["pos_x"][s], post["pos_y"][s]) != (pre["pos_x"][s], pre["pos_y"][s])
            turned = post["heading"][s] != pre["heading"][s]
            if st0 == swarm.AI_RUN and st1 == swarm.AI_RUN and turned and moved:
                counts["tumble"] += 1
            if st0 == swarm.AI_RUN and st1 == swarm.AI_RUN and turned and not moved:
                counts["wall"] += 1
            if st0 == swarm.AI_RUN and st1 == swarm.AI_COMA:
                counts["run_coma"] += 1
            if st0 == swarm.AI_COMA and st1 == swarm.AI_COMA and CTMIN <= tt <= CEXIT:
                counts["coma_held"] += 1
            if st0 == swarm.AI_RUN and st1 == swarm.AI_RUN and CTMIN <= tt <= CEXIT:
                counts["run_held"] += 1
            if st0 == swarm.AI_COMA and st1 == swarm.AI_RUN:
                counts["coma_run"] += 1
        if t == 100:
            free = np.flatnonzero(gA.swarm_larva_valid == 0)[:3]
            nid = int(gA.swarm_next_unit_id)
            sa = fA.spawn("larva", [(3 << 16)] * 3, [(3 << 16)] * 3, [0, 1, 2])
            sb = fB.spawn("larva", [(3 << 16)] * 3, [(3 << 16)] * 3, [0, 1, 2])
            if (list(sa) == list(free) == list(sb) and
                    list(gA.swarm_larva_unit_id[sa]) == [nid, nid + 1, nid + 2]):
                counts["reuse"] += 1
    print(f"  transitions: {counts}")
    for k, v in counts.items():
        if v <= 0:
            print(f"  PART 2 never saw '{k}'")
            ok = False
    tally = bp.swarm_branch_tally()
    missing = [k for k, v in tally.items() if v <= 0]
    if missing:
        print(f"  PART 1 + PART 2 never took the law branches {missing}")
        ok = False
    return ok


def _thermal_world(mode):
    import field_ab_harness as fab
    s8a._residency(False); s8a._set_backends(False); bp.set_swarm_backend(False)
    sim = fab.swarm_thermal_scenario_sim()
    sim.physics_runner.headless = True
    return sim


def _mode(mode):
    if mode == "cpu":
        s8a._residency(False); s8a._set_backends(False); bp.set_swarm_backend(False)
    elif mode == "percall":
        s8a._residency(False); s8a._set_backends(False); bp.set_swarm_backend(True)
    else:
        s8a._residency(True); s8a._set_backends(True); bp.set_swarm_backend(False)


def part2b() -> bool:
    print("PART 2b: the sim paths three-way -- CPU, per-call, resident (N = 1)")
    from field_ab_harness import capture_snapshot
    from field_digest import tick_digest
    from simulation.events import SwarmUnitDiedEvent
    from simulation.swarm import swarm_carrier, swarm_section_bytes
    modes = ("cpu", "percall", "resident")
    worlds = {m: _thermal_world(m) for m in modes}
    ok = True
    deaths = comas = 0
    walls = 0
    upload_ticks = []
    reclaimed = None
    for t in range(130):
        for m in modes:
            sim = worlds[m]
            g = sim.gmap
            if t == 40:
                sim.swarm.spawn("larva", [(7 << 16) + 9, (9 << 16) + 9], [(8 << 16) + 9, (6 << 16) + 9],
                                [1000, -1000])
            if t == 70:
                g.destroy_wall(8, 15)
            if t == 90:
                live = np.flatnonzero(g.swarm_larva_valid == 1)
                reclaimed = int(live[-1])
                sim.swarm.reclaim("larva", [reclaimed])
        st = {}
        for m in modes:
            sim = worlds[m]
            r = sim.physics_runner
            _mode(m)
            cc, rc, up = bp.swarm_cuda_calls(), dict(bp.resident_stage_counters()), r.swarm_store_uploads
            pre_state = sim.gmap.swarm_larva_ai_state.copy()
            pre_head = sim.gmap.swarm_larva_heading.copy()
            pre_x = sim.gmap.swarm_larva_pos_x.copy()
            sim.set_paused(False)
            sim.step()
            present = swarm.swarm_present(sim.gmap)
            if m == "percall" and present and bp.swarm_cuda_calls() <= cc:
                print(f"  tick {t}: the per-call path did not run on the device"); ok = False
            if m == "resident":
                rc2 = dict(bp.resident_stage_counters())
                if present and (rc2["swarm"] <= rc["swarm"] or rc2["swarm_present"] <= rc["swarm_present"]):
                    print(f"  tick {t}: the resident swarm stage did not run"); ok = False
                if r.swarm_store_uploads > up:
                    upload_ticks.append(t)
            if m == "cpu":
                post = sim.gmap.swarm_larva_ai_state
                v = sim.gmap.swarm_larva_valid == 1
                comas += int(((pre_state == swarm.AI_RUN) & (post == swarm.AI_COMA) & v).sum())
                walls += int(((pre_head != sim.gmap.swarm_larva_heading)
                              & (pre_x == sim.gmap.swarm_larva_pos_x) & v
                              & (post == swarm.AI_RUN)).sum())
            ev = [(e.species, e.unit_id, e.pos_x_q, e.pos_y_q, e.heading_q)
                  for e in sim.tick_events if isinstance(e, SwarmUnitDiedEvent)]
            st[m] = (swarm_section_bytes(swarm_carrier(sim.gmap)), ev,
                     tick_digest(capture_snapshot(sim)))
            if swarm.check_invariants(sim.gmap):
                print(f"  tick {t} {m}: {swarm.check_invariants(sim.gmap)[:2]}"); ok = False
            if t == 91:
                for c in swarm.ROSTER:
                    if int(swarm.swarm_column(sim.gmap, "larva", c.name)[reclaimed]) != 0:
                        print(f"  tick {t} {m}: the reclaimed slot resurrected"); ok = False
        _mode("cpu")
        deaths += len(st["cpu"][1])
        for m in ("percall", "resident"):
            for k, name in enumerate(("swarm section", "death events", "tick digest")):
                if st[m][k] != st["cpu"][k]:
                    print(f"  tick {t}: {m} {name} differs from the CPU path"); ok = False
        if not ok:
            break
    dist = int(worlds["cpu"].gmap.swarm_larva_dist_walked.max())
    print(f"  deaths {deaths}, comas {comas}, wall hits {walls}, max dist {dist}, "
          f"store uploads at ticks {upload_ticks}")
    if deaths <= 0 or comas <= 0 or walls <= 0 or dist <= 0:
        print("  PART 2b non-vacuity failed"); ok = False
    if upload_ticks != [0, 40, 90]:
        print("  the store must ride the entry upload on the first tick and after each facade write")
        ok = False

    # the swarm-free resident leg
    import field_ab_harness as fab
    _mode("cpu")
    sim = fab.default_scenario_sim()
    sim.physics_runner.headless = True
    _mode("resident")
    rc = dict(bp.resident_stage_counters())
    try:
        for t in range(10):
            sim.set_paused(False)
            sim.step()
            if sim.physics_runner.swarm_death_rows.shape[0] != 0:
                print("  swarm-free resident leg: death rows"); ok = False
    finally:
        _mode("cpu")
    rc2 = dict(bp.resident_stage_counters())
    if rc2["swarm"] != rc["swarm"] or rc2["swarm_present"] != rc["swarm_present"] \
            or sim.physics_runner.swarm_store_uploads != 0:
        print("  swarm-free resident leg moved the swarm counters"); ok = False
    print("  " + ("three paths equal every tick" if ok else "FAILED"))
    return ok


def _bite_world():
    from simulation.unit import Unit
    sim = _thermal_world("cpu")
    half = ONE >> 1
    pile = [((10 << 16) + half + 300 * k, (11 << 16) + half) for k in range(8)]
    sim.swarm.spawn("larva", [x for x, _ in pile], [y for _, y in pile], [0] * 8)
    marines = (("A", 7.0, 8.0, None), ("C", 2.0, 10.0, None), ("V", 9.0, 10.0, 0.25))
    for name, x, y, hp in marines:
        u = Unit(name, x=x, y=y, team=0)
        sim.add_unit(u)
        if hp is not None:
            u.current_hp = hp
    return sim


def part2c() -> bool:
    print("PART 2c: the bite three-way -- the plane and the poison on CPU, per-call, resident")
    from field_ab_harness import capture_snapshot
    from field_digest import tick_digest
    from simulation.events import UnitHitEvent, UnitKilledEvent
    modes = ("cpu", "percall", "resident")
    worlds = {m: _bite_world() for m in modes}
    ok = True
    plane_ticks = dosed = 0
    bite_kill = None
    coma_alone = coma_zero = 0
    for t in range(200):
        st = {}
        for m in modes:
            sim = worlds[m]
            _mode(m)
            sim.set_paused(False)
            sim.step()
            g = sim.gmap
            plane = swarm.bite_plane(g)
            units = tuple((u.name, u.alive, int(u.poison_x0), int(u.poison_x1),
                           int(u.poison_cause), int(u.poison_src)) for u in sim.units)
            ev = tuple((type(e).__name__, getattr(e, "unit_id", None),
                        getattr(e, "source", getattr(e, "killed_by", None)),
                        getattr(e, "attacker_id", None))
                       for e in sim.tick_events
                       if isinstance(e, (UnitHitEvent, UnitKilledEvent)))
            st[m] = (plane.tobytes(), units, ev, tick_digest(capture_snapshot(sim)))
            if m == "cpu":
                if plane.any():
                    plane_ticks += 1
                a = next(u for u in sim.units if u.name == "A")
                if a.poison_x0 > 0:
                    dosed += 1
                for e in sim.tick_events:
                    if isinstance(e, UnitKilledEvent):
                        bite_kill = (e.killed_by, e.attacker_id)
                v = g.swarm_larva_valid == 1
                tx = g.swarm_larva_pos_x >> 16
                ty = g.swarm_larva_pos_y >> 16
                biting = v & (g.swarm_larva_ai_state != swarm.AI_COMA)
                for k in np.flatnonzero(v & (g.swarm_larva_ai_state == swarm.AI_COMA)):
                    same = biting & (tx == tx[k]) & (ty == ty[k])
                    if not same.any():
                        coma_alone += 1
                        coma_zero += int(plane[ty[k], tx[k]] == 0)
        _mode("cpu")
        for m in ("percall", "resident"):
            for k, name in enumerate(("bite plane", "unit poison", "hit/kill events",
                                      "tick digest")):
                if st[m][k] != st["cpu"][k]:
                    print(f"  tick {t}: {m} {name} differs from the CPU path"); ok = False
        if not ok:
            break
    print(f"  plane non-zero on {plane_ticks} ticks, marine A dosed on {dosed}, "
          f"bite kill {bite_kill}, coma larvae alone {coma_alone} (plane 0 at {coma_zero})")
    if plane_ticks <= 0 or dosed <= 0 or coma_alone <= 0:
        print("  PART 2c non-vacuity failed"); ok = False
    if coma_zero != coma_alone:
        print("  a coma larva alone on its tile bit"); ok = False
    if bite_kill != ("larva_bite", -1):
        print(f"  the pile did not kill by bite (got {bite_kill})"); ok = False
    print("  " + ("three paths equal every tick" if ok else "FAILED"))
    return ok


def _corpse_world():
    from simulation import corpse
    from simulation.damage import KINETIC, DamagePacket, apply_packet
    from simulation.ruleset import ContinuousRealtime
    from simulation.unit import Unit
    sim = _thermal_world("cpu")
    sim.ruleset = ContinuousRealtime()
    half = ONE >> 1
    units = (("KM", 11.0, 9.0, 0), ("KZ", 11.0, 5.0, 255), ("KX", 6.0, 12.0, 0))
    for name, x, y, team in units:
        u = Unit(name, x=x, y=y, team=team)
        sim.add_unit(u)
        apply_packet(u, DamagePacket(amount=u.current_hp, dtype=KINETIC), None,
                     source="explosion")
        if name == "KX":
            u.hp_q = -corpse.body_q(u)               # fixture: destroyed (P5c)
    piles = [(12, 10)] * 16 + [(12, 6)] * 14 + [(7, 13)] * 10
    sim.swarm.spawn("larva", [(tx << 16) + half for tx, _ in piles],
                    [(ty << 16) + half for _, ty in piles], [0] * len(piles))
    return sim


def part2d() -> bool:
    print("PART 2d: the corpses three-way -- the count plane and the corpse levels")
    from field_ab_harness import capture_snapshot
    from field_digest import tick_digest
    modes = ("cpu", "percall", "resident")
    worlds = {m: _corpse_world() for m in modes}
    ok = True
    plane_ticks = 0
    rose = {}
    kx_level = 0
    for t in range(400):
        st = {}
        for m in modes:
            sim = worlds[m]
            _mode(m)
            sim.set_paused(False)
            sim.step()
            g = sim.gmap
            plane = swarm.count_plane(g)
            units = tuple((u.name, u.alive, u.is_zombie, int(u.infest_q),
                           int(u.hp_q)) for u in sim.units)
            st[m] = (plane.tobytes(), units, tick_digest(capture_snapshot(sim)))
            if m == "cpu":
                if plane.any():
                    plane_ticks += 1
                v = g.swarm_larva_valid == 1
                act = v & (g.swarm_larva_ai_state != swarm.AI_COMA) & \
                    (g.swarm_larva_ai_state != swarm.AI_DEAD)
                scan = np.zeros(plane.shape, dtype=np.int64)
                np.add.at(scan, (g.swarm_larva_pos_y[act] >> 16,
                                 g.swarm_larva_pos_x[act] >> 16), 1)
                if not np.array_equal(scan, plane):
                    print(f"  tick {t}: the count plane is not the host scan"); ok = False
                for u in sim.units:
                    name = u.name[2:] if u.name.startswith("Z-") else u.name   # the door renames
                    if name in ("KM", "KZ", "KX") and u.alive and name not in rose:
                        rose[name] = t
                    if name == "KX":
                        kx_level = max(kx_level, int(u.infest_q))
        _mode("cpu")
        for m in ("percall", "resident"):
            for k, name in enumerate(("count plane", "unit corpses", "tick digest")):
                if st[m][k] != st["cpu"][k]:
                    print(f"  tick {t}: {m} {name} differs from the CPU path"); ok = False
        if not ok:
            break
    print(f"  count plane non-zero on {plane_ticks} ticks, rises {rose}, "
          f"destroyed corpse's highest level {kx_level}")
    if plane_ticks <= 0 or "KM" not in rose or "KZ" not in rose:
        print("  PART 2d non-vacuity failed"); ok = False
    if "KX" in rose or kx_level != 0:
        print("  the destroyed corpse moved or rose"); ok = False
    print("  " + ("three paths equal every tick" if ok else "FAILED"))
    return ok


def part3() -> bool:
    print("PART 3: the canonical golden on the CUDA build's CPU path")
    _mode("cpu")
    from field_ab_harness import capture_trajectory
    from _xarch_perfield_digest import GOLDEN_AGGREGATE, trajectory_digest
    ok = trajectory_digest(capture_trajectory(n_steps=30)) == GOLDEN_AGGREGATE
    print("  " + ("matches GOLDEN_AGGREGATE" if ok else "GOLDEN MOVED"))
    return ok


def main() -> int:
    print(f"swarm P3 CUDA gate, HAS_CUDA={getattr(bp, 'HAS_CUDA', False)}")
    ok = True
    for part in (part1, part2, part2b, part2c, part2d, part3):
        try:
            ok = part() and ok
        except Exception as exc:   # report and fail, never hide
            import traceback
            traceback.print_exc()
            print(f"  {part.__name__} raised {exc!r}")
            ok = False
        finally:
            _mode("cpu")
    print("SWARM_P3_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
