"""#29 P4c — THE DEVICE EDIT QUEUE GATE: slot 6b's FieldEdit flush applied on
the device, against the CPU flush, tol 0, every tick.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_edit_queue.py). The CPU flush (``EditQueue.flush`` ->
``field_edit.apply_field_edit``) is THE oracle; the device queue is
``field_edit.pack_for_device`` (host: the order, the door, the region walk)
-> ``PhysicsEngine.apply_edits_resident`` (cpp/src/cuda_edit_queue.cu: the
contribution, the keyed noise, the masks, the combines, the gas-energy seam).

Every scene steps THREE identically built ``Simulation``s through the real
``Simulation.step`` (so slot 6b is the real one):

  cpu  residency OFF -- the host flush (the oracle);
  gpu  residency ON  -- the device queue serves 6b;
  ref  residency OFF, the scene's edits NOT enqueued -- so "the edits landed"
       is a measured fact: the scene's target arrays differ between cpu and
       ref on its touched tiles (a scene whose edits all landed on skipped
       tiles, or a pack that silently packed nothing, fails as VACUOUS).

and per tick holds, through cuda_s8a_check's leg driver: EVERY GameMap array
and EVERY integer book (the trace deposit / clip books, the gas-energy seam
channels and net, the engines' books) equal at tol 0, the energy closure
identity 0 on both worlds, the stamp planes (poisoned before the tick) --
plus that the device SERVED every edit-bearing tick (the runner's
``edit_device_ticks``, the resident ``edit_queue`` counter) and never fell
back to the host flush (``edit_host_ticks`` 0), except in the fallback scene.

SCENES -- one per FieldEdit policy row, each mode / region / falloff / noise /
combine the row allows:

  trace       smoke + a trace gas: ADD DISC LINEAR noisy, ADD RECT over the
              hull and space, REMOVE TILE, MAX BEAM noisy, an edit with its
              own clamp, a deposit past the int32 saturation (the clip book);
  atmosphere  ADD DISC LINEAR noisy (minted at ambient), REMOVE RECT (leaving
              at the cell's T), ADD with a clamp (the 21 / 79 clamps), ADD on
              space tiles (bulk written, no books), MAX TILE;
  wave        wave_source ADD DISC LINEAR noisy (N dT through the seam), a
              deposit past T_MAX_PHYS (the cap), REMOVE RECT, MAX TILE, a
              disc over space and the hull (skipped);
  fire        MAX DISC over furniture + wood + air (the flammable mask), ADD
              past the [0, 1] clamp, REMOVE;
  heat        ADD DISC LINEAR noisy, ADD past INT32_MAX (the saturating add),
              REMOVE, MAX;
  gas_energy  ADD TILE, a negative RECT, a TILES row across space and hull
              (the accountable mask);
  tiles       TILES with integer weights over weight_den and the blast's
              Q16 ``scale`` plane, noisy (the fixed-total deposit) on smoke,
              heat and atmosphere; TILES with float weights on fire;
  ambient     an ambient map (the ring is non-solid and outside the
              accountable set): wave (the direct T write), atmosphere (bulk
              written, no books) and gas_energy (skipped) over the ring;
  fallback    a water_depth edit beside the others: the device cannot serve
              it (the resident pre-step reads water_depth on the host), the
              whole tick is flushed on the host, equal by construction --
              the gate checks the fallback was TAKEN and named;
  overlap     (critique 5 M3, the cross-tick key) a HEAT SOURCE entity
              enqueues its deposit at slot 9e of tick t; tick t+1 enqueues,
              inside the tick (slot 5), a heat MAX on the same (field,
              source, region) -- so only the edit key (seq, spanning the
              previous tick's 9e) orders them: ADD then MAX gives max(a, m)
              = m at the source tile, MAX then ADD would give m + a. The
              heat plane right after slot 7 must read m on BOTH worlds.

PLUS
  refusal     a gas edit on a CONSERVATIVE slice raises the host's
              AssertionError on the resident path too (a host refusal);
  noise       for every TILE record the scenes packed, the device's
              contribution (the probe of the tick's own kernel K1) equals
              the CPU's ``div_round(amount_q * weight * noise_mult_q16(draw)
              * scale, den)`` on the keyed draw (tick, seq, FIELD_NOISE,
              tile) -- and with every record's key moved by one it does NOT
              (the noise is live and keyed by the edit's rank).

PROPERTY: the device's flush leaves every field and book where the host's
does, from the host's records alone. BREAKS IF a combine, a clamp, a skip
mask, a rounding, the noise counter, the seam's booking or mirror refresh, or
the order (within a tick or across the previous tick's 9e) differs.

Prints ``EDIT_QUEUE_RESULT: PASS`` / ``FAIL`` and exits 0 / 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap)

import cuda_s8a_check as s8a

H = W = 32
N_TICKS = 12


# ---------------------------------------------------------------------------
# the worlds
# ---------------------------------------------------------------------------
def _level(entities=()):
    """A hull room with a breach to space, a wood segment (solid, flammable),
    furniture (passable, flammable, a thermal solid: outside the accountable
    set) and a grille; space all round."""
    from level_loader import LevelData
    from simulation.materials import (MAT_AIR, MAT_FURNITURE, MAT_GRILLE,
                                      MAT_HULL, MAT_WOOD)
    tm = np.full((H, W), -1, dtype=np.int32)
    tm[2:H - 2, 2:W - 2] = MAT_HULL
    tm[3:H - 3, 3:W - 3] = MAT_AIR
    tm[12, 2] = MAT_AIR                     # a breach to space
    tm[20:23, 20:23] = MAT_FURNITURE
    tm[6:9, 15] = MAT_WOOD
    tm[25, 5:9] = MAT_GRILLE
    return LevelData(name="edit_queue_gate", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."),
                     entities=list(entities), wires=[])


def _ambient_level():
    """cuda_s8a_check's ambient layout (a SPACE ring routed to is_ambient
    around open air) as a Simulation level."""
    from level_loader import LevelData
    h = w = 40
    tm = np.full((h, w), 9, dtype=np.int32)
    tm[0, :] = tm[-1, :] = tm[:, 0] = tm[:, -1] = 0
    tm[1, 18:22] = 1
    tm[25:28, 20] = 2
    return LevelData(name="edit_queue_ambient", version="1", path=Path("."),
                     tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."),
                     boundary="ambient")


def _sim(level_fn, seed=11):
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    sim = Simulation(level_fn(), seed=seed, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.physics_runner.headless = False
    return sim


def _trace_id(g):
    from simulation.gases import SMOKE
    for gi in range(g.gas.shape[0]):
        if not bool(g.gases.conservative[gi]) and gi != int(SMOKE):
            return gi
    raise AssertionError("the scene needs a second trace gas")


# ---------------------------------------------------------------------------
# the scenes: (g, t) -> [FieldEdit]
# ---------------------------------------------------------------------------
def _fe():
    from simulation import field_edit as fe
    return fe


def scene_trace(g, t):
    fe = _fe()
    F, R, M, L = fe.FieldEdit, fe.Region, fe.EditMode, fe.Falloff
    gi = _trace_id(g)
    out = [
        F("smoke", R.DISC, (15, 15, 3.5), 0.6, M.ADD, L.LINEAR, noise=0.4, source_id=1),
        F("gas", R.RECT, (5, 1, 9, 12), 0.7, M.ADD, channel=gi, source_id=2),
        F("gas", R.BEAM, (20, 4, 18, 27, 1.5), 0.9, M.MAX, L.LINEAR, channel=gi,
          noise=0.2, source_id=3),
        F("smoke", R.DISC, (10, 22, 2.5), 0.9, M.ADD, clamp=(0.0, 0.5), source_id=4),
        F("smoke", R.TILE, (25, 25), 200.0, M.ADD, source_id=5),     # past the int32 cell
    ]
    if t % 2:
        out.append(F("smoke", R.TILE, (15, 15), 0.2, M.REMOVE, source_id=1))
    return out


def scene_atmosphere(g, t):
    fe = _fe()
    F, R, M, L = fe.FieldEdit, fe.Region, fe.EditMode, fe.Falloff
    out = [
        F("atmosphere", R.DISC, (15, 10, 4.0), 0.5, M.ADD, L.LINEAR, noise=0.3, source_id=1),
        F("atmosphere", R.RECT, (22, 3, 24, 12), 0.1, M.REMOVE, source_id=2),
        F("atmosphere", R.RECT, (5, 18, 7, 24), 0.4, M.ADD, clamp=(0.0, 1.2), source_id=3),
        F("atmosphere", R.RECT, (0, 0, 1, 31), 0.3, M.ADD, source_id=4),   # space tiles
        F("atmosphere", R.TILE, (17, 17), 1.5, M.MAX, source_id=5),
    ]
    return out


def scene_wave(g, t):
    fe = _fe()
    F, R, M, L = fe.FieldEdit, fe.Region, fe.EditMode, fe.Falloff
    out = [
        F("wave_source", R.DISC, (15, 20, 3.0), 300.0, M.ADD, L.LINEAR, noise=0.25,
          source_id=1),
        # (not on the furniture: a wave_source edit on a non-solid THERMAL
        # solid writes its temperature outside the solid books, so the closure
        # identity stays open on the CPU flush itself -- a host-law finding of
        # P4c, reported, not fixed; the direct-T branch is the ambient scene's)
        F("wave_source", R.TILE, (10, 10), 20000.0, M.ADD, source_id=3),  # past T_MAX_PHYS
        F("wave_source", R.RECT, (26, 10, 27, 14), 20.0, M.REMOVE, source_id=4),
        F("wave_source", R.TILE, (5, 26), 800.0, M.MAX, source_id=5),
        F("wave_source", R.DISC, (12, 3, 3.0), 100.0, M.ADD, source_id=6),  # over space
    ]
    return out


def scene_fire(g, t):
    fe = _fe()
    F, R, M, L = fe.FieldEdit, fe.Region, fe.EditMode, fe.Falloff
    out = [
        F("fire", R.DISC, (21, 21, 2.5), 0.9, M.MAX, source_id=1),
        F("fire", R.RECT, (6, 14, 8, 16), 0.7, M.ADD, source_id=2),       # wood (+ air: skipped)
    ]
    if t % 3 == 2:
        out.append(F("fire", R.TILE, (21, 20), 0.3, M.REMOVE, source_id=3))
    return out


def scene_heat(g, t):
    fe = _fe()
    F, R, M, L = fe.FieldEdit, fe.Region, fe.EditMode, fe.Falloff
    return [
        F("heat", R.DISC, (15, 15, 3.0), 2.0, M.ADD, L.LINEAR, noise=0.5, source_id=1),
        F("heat", R.TILE, (8, 8), 40000.0, M.ADD, source_id=2),          # saturates
        F("heat", R.TILE, (8, 8), 1.0, M.REMOVE, source_id=3),
        F("heat", R.RECT, (2, 10, 3, 14), 5.0, M.MAX, source_id=4),      # hull + air
    ]


def scene_gas_energy(g, t):
    fe = _fe()
    F, R, M = fe.FieldEdit, fe.Region, fe.EditMode
    tiles = tuple((12, c, 1 + (c % 3)) for c in range(0, 8))   # space, breach, hull, air
    return [
        F("gas_energy", R.TILE, (15, 15), 0.0, M.ADD, amount_q=10 ** 9, source_id=1),
        F("gas_energy", R.RECT, (18, 4, 19, 9), 0.0, M.ADD, amount_q=-(10 ** 8),
          source_id=2),
        F("gas_energy", R.TILES, tiles, 0.0, M.ADD, amount_q=3 * 10 ** 8,
          weight_den=sum(wt for _, _, wt in tiles), source_id=3),
    ]


def _scale_plane(g):
    rng = np.random.default_rng(7)
    return rng.integers(1 << 15, 3 << 15, size=g.solid.shape).astype(np.int32)


def scene_tiles(g, t):
    fe = _fe()
    F, R, M = fe.FieldEdit, fe.Region, fe.EditMode
    walk = tuple((r, c, 1 + ((r * 7 + c) % 5)) for r in range(13, 18) for c in range(4, 11))
    den = sum(wt for _, _, wt in walk)
    scale = _scale_plane(g)
    fl = tuple((r, c, 0.25 + 0.1 * ((r + c) % 4)) for r in range(20, 23) for c in range(19, 24))
    return [
        F("smoke", R.TILES, walk, 0.0, M.ADD, amount_q=1 << 30, weight_den=den,
          noise=0.3, scale=scale, source_id=1),
        F("heat", R.TILES, walk, 0.0, M.ADD, amount_q=40 << 16, weight_den=den,
          scale=scale, source_id=2),
        F("atmosphere", R.TILES, walk, 0.0, M.ADD, amount_q=1 << 16, weight_den=den,
          noise=0.5, source_id=3),
        F("fire", R.TILES, fl, 0.8, M.MAX, source_id=4),
    ]


def scene_ambient(g, t):
    fe = _fe()
    F, R, M, L = fe.FieldEdit, fe.Region, fe.EditMode, fe.Falloff
    return [
        F("wave_source", R.RECT, (0, 5, 2, 15), 150.0, M.ADD, source_id=1),
        F("atmosphere", R.DISC, (1, 30, 3.0), 0.4, M.ADD, L.LINEAR, noise=0.2, source_id=2),
        F("gas_energy", R.RECT, (0, 30, 3, 34), 0.0, M.ADD, amount_q=10 ** 9, source_id=3),
        F("smoke", R.RECT, (0, 0, 2, 39), 0.3, M.ADD, source_id=4),
    ]


def scene_fallback(g, t):
    fe = _fe()
    F, R, M = fe.FieldEdit, fe.Region, fe.EditMode
    return scene_heat(g, t) + [F("water_depth", R.TILE, (14, 6), 0.2, M.ADD, source_id=9)]


#: name -> (scene fn, level fn, the arrays its edits target)
SCENES = {
    "trace": (scene_trace, _level, ("gas",)),
    "atmosphere": (scene_atmosphere, _level, ("gas", "gas_energy")),
    "wave": (scene_wave, _level, ("temperature", "gas_energy")),
    "fire": (scene_fire, _level, ("fire",)),
    "heat": (scene_heat, _level, ("temperature",)),
    "gas_energy": (scene_gas_energy, _level, ("gas_energy",)),
    "tiles": (scene_tiles, _level, ("gas", "temperature", "fire")),
    "ambient": (scene_ambient, _ambient_level, ("temperature", "gas", "gas_energy")),
}


def _touched(g, edits):
    """The tiles a tick's edits name (their regions, unmasked)."""
    fe = _fe()
    m = np.zeros(g.solid.shape, dtype=bool)
    for e in edits:
        for (r, c, _w) in fe._iter_region(e.region, e.coords, e.falloff, g.solid.shape):
            m[r, c] = True
    return m


def _counts():
    return dict(bp.resident_stage_counters())


def run_scene(name, fallback=False) -> bool:
    if fallback:
        fn, level_fn, targets = scene_fallback, _level, ("temperature",)
    else:
        fn, level_fn, targets = SCENES[name]
    s8a._residency(False); s8a._set_backends(False)
    s_c, s_g, s_r = _sim(level_fn), _sim(level_fn), _sim(level_fn)
    landed = {k: 0 for k in targets}
    touched_box = {}

    def step(world, t):
        sim = world[2]
        edits = fn(sim.gmap, t)
        for e in edits:
            sim.edit(e)
        if sim is s_c:
            touched_box["m"] = _touched(sim.gmap, edits)
            s_r.set_paused(False)
            s_r.step()                               # the no-edit reference
        s8a._sim_step(world, t)

    def post(t, g_c, g_g):
        m = touched_box.get("m")
        for k in targets:
            a, b = getattr(g_c, k), getattr(s_r.gmap, k)
            mm = m if a.ndim == 2 else np.broadcast_to(m, a.shape)
            if np.any((a != b) & mm):
                landed[k] += 1
        return 0

    r_g = s_g.physics_runner
    d0, h0 = r_g.edit_device_ticks, r_g.edit_host_ticks
    c0 = _counts()
    ok = s8a._run_leg(f"P4c {name}", (s_c.gmap, s_c.physics_runner, s_c),
                      (s_g.gmap, r_g, s_g), N_TICKS, step,
                      {} if fallback else {"edit_queue": N_TICKS},
                      post_tick=post)
    c1 = _counts()
    served = r_g.edit_device_ticks - d0
    host = r_g.edit_host_ticks - h0
    tiles = c1["edit_queue_tiles"] - c0["edit_queue_tiles"]
    if fallback:
        if host != N_TICKS or served != 0:
            ok = False
            print(f"  {name}: the fallback was not taken on every tick "
                  f"(host {host}, device {served})")
        reason = r_g.edit_fallback_reason or ""
        if "water_depth" not in reason:
            ok = False
            print(f"  {name}: the fallback's reason does not name water_depth: {reason!r}")
        print(f"  {name}: fallback taken on {host}/{N_TICKS} ticks ({reason!r})")
    else:
        if served != N_TICKS or host != 0:
            ok = False
            print(f"  {name}: the device served {served}/{N_TICKS} ticks, the host "
                  f"{host} (expected every tick on the device)")
        if tiles <= 0:
            ok = False
            print(f"  {name}: the device applied no tile (vacuous)")
    for k, n in landed.items():
        if n == 0:
            ok = False
            print(f"  {name}: the edits never changed gmap.{k} on their tiles "
                  f"against the no-edit world (VACUOUS)")
    print(f"  {name}: device served {served} ticks, {tiles} tiles applied; edits "
          f"landed on " + ", ".join(f"{k} ({n} ticks)" for k, n in landed.items()))
    return ok


# ---------------------------------------------------------------------------
# the overlap scene (critique 5 M3)
# ---------------------------------------------------------------------------
SRC_AT = (15, 15)


def _overlap_level():
    from level_loader import EntityInstance
    from simulation.entities import REGISTRY
    fields = {f.name: f.default for f in REGISTRY["heat_source"].FIELDS}
    fields.update(x=SRC_AT[1], y=SRC_AT[0], heat_per_s=2.0)
    ent = EntityInstance(id="stove", class_name="heat_source", ordinal=0,
                         tags=(), fields=fields)
    return _level([ent])


def run_overlap() -> bool:
    from simulation import field_edit as fe
    from simulation.payloads import _SRC_PAYLOAD_HEAT
    print("P4c overlap -- a heat source's 9e deposit (tick t) and an in-tick heat "
          "MAX (tick t+1, slot 5) on the same (field, source, region):")
    s8a._residency(False); s8a._set_backends(False)
    sims = {"cpu": _sim(_overlap_level), "gpu": _sim(_overlap_level)}
    m_amount = 1.0   # heat units: above the source's per-tick deposit
    snaps = {"cpu": [], "gpu": []}
    for key, sim in sims.items():
        orig_zombie = sim._run_zombie_slot
        orig_phys = sim._step_physics

        def zombie(sim=sim, orig=orig_zombie):
            orig()
            sim.edit(fe.FieldEdit("heat", fe.Region.DISC, (SRC_AT[0], SRC_AT[1], 1.0),
                                  m_amount, fe.EditMode.MAX, fe.Falloff.LINEAR,
                                  source_id=_SRC_PAYLOAD_HEAT))

        def phys(sim=sim, orig=orig_phys, key=key):
            d = orig()
            snaps[key].append(sim.gmap.heat.copy())
            return d

        sim._run_zombie_slot = zombie
        sim._step_physics = phys

    def step(world, t):
        s8a._sim_step(world, t)

    ok = s8a._run_leg("P4c overlap", (sims["cpu"].gmap, sims["cpu"].physics_runner,
                                      sims["cpu"]),
                      (sims["gpu"].gmap, sims["gpu"].physics_runner, sims["gpu"]),
                      N_TICKS, step, {"edit_queue": N_TICKS - 1})
    m_q = fe.heat_quantize(m_amount)
    a_seen = 0
    for t, (hc, hg) in enumerate(zip(snaps["cpu"], snaps["gpu"])):
        if not np.array_equal(hc, hg):
            ok = False
            print(f"  overlap tick {t}: the heat plane after slot 7 differs "
                  f"({int(np.count_nonzero(hc != hg))} cells)")
        v = int(hc[SRC_AT])
        if t >= 1:
            # tick t >= 1 carries the source's deposit from tick t-1's 9e
            if v != m_q:
                ok = False
                print(f"  overlap tick {t}: heat at the source tile {v} != m {m_q} "
                      f"(the 9e ADD did not apply BEFORE the in-tick MAX)")
            a_seen += 1
    print(f"  overlap: heat at the source tile == m ({m_q}) on {a_seen} ticks, "
          f"device == CPU on the heat plane after slot 7 every tick")
    return ok


def _overlap_source_amount() -> int:
    """The heat source's per-tick Q16 deposit, from its runtime (so the gate
    can show m > a: MAX-then-ADD would be distinguishable)."""
    s8a._residency(False)
    sim = _sim(_overlap_level)
    sim.set_paused(False)
    sim.step()
    sim.set_paused(False)
    snap = []
    orig = sim._step_physics

    def phys():
        d = orig()
        snap.append(sim.gmap.heat.copy())
        return d
    sim._step_physics = phys
    sim.step()
    return int(snap[0][SRC_AT])


# ---------------------------------------------------------------------------
# the refusal and the noise
# ---------------------------------------------------------------------------
def run_refusal() -> bool:
    from simulation import field_edit as fe
    from simulation.gases import O2
    print("P4c refusal -- a gas edit on a CONSERVATIVE slice:")
    ok = True
    for res in (False, True):
        s8a._residency(False)
        sim = _sim(_level)
        sim.edit(fe.FieldEdit("gas", fe.Region.TILE, (15, 15), 0.1, channel=int(O2)))
        s8a._residency(res); s8a._set_backends(res)
        try:
            sim.set_paused(False)
            sim.step()
            ok = False
            print(f"  refusal: no AssertionError with residency {res}")
        except AssertionError as exc:
            if "CONSERVATIVE" not in str(exc):
                ok = False
                print(f"  refusal: the wrong AssertionError: {exc}")
        finally:
            s8a._residency(False); s8a._set_backends(False)
    print(f"  refusal: the host's AssertionError on both paths: {ok}")
    return ok


def run_noise() -> bool:
    """Every TILE record of every scene's tick-3 pack: the device's K1 equals
    the CPU's per-tile contribution (the formula apply_field_edit evaluates,
    the noise through draws.noise_mult_q16 on the keyed draw); with every key
    moved by one, the noisy records' contributions DIFFER."""
    from simulation import draws as dr
    from simulation import field_edit as fe
    from simulation import unit_fixed as ufx
    from simulation.rng_salts import Salt
    print("P4c noise -- the device's contribution (keyed noise, rounding) on every "
          "packed record == the CPU's draws.noise_mult_q16 / div_round:")
    s8a._residency(False)
    ok = True
    n_rec = n_noisy = 0
    for name, (fn, level_fn, _t) in SCENES.items():
        sim = _sim(level_fn, seed=1234 + len(name))
        g = sim.gmap
        q = fe.EditQueue()
        for e in fn(g, 3):
            q.enqueue(e)
        ordered = q.take_ordered()
        pack, reason = fe.pack_for_device(g, ordered, sim.draws)
        if pack is None:
            print(f"  {name}: not packed ({reason})")
            ok = False
            continue
        env = np.array([[pack.tick_word, pack.key[0], pack.key[1], pack.t_amb_raw]],
                       dtype=np.int64)
        dev = bp.edit_queue_contrib_probe(pack.edits, pack.tiles, env,
                                          int(Salt.FIELD_NOISE))
        # the CPU's per-tile contribution, record by record
        cpu = np.zeros(len(pack.tiles), dtype=object)
        for k, (ei, tile, weight, scale) in enumerate(pack.tiles.tolist()):
            er = pack.edits[ei]
            num = int(er[fe.E_AMOUNT]) * int(weight)
            if er[fe.E_NOISY]:
                num *= dr.noise_mult_q16(
                    sim.draws.at(Salt.FIELD_NOISE, int(tile), int(er[fe.E_KEY])),
                    int(er[fe.E_LOW]))
                n_noisy += 1
            num *= int(scale)
            cpu[k] = ufx.div_round(num, int(er[fe.E_DEN]))
        n_rec += len(pack.tiles)
        if not all(int(a) == int(b) for a, b in zip(dev.tolist(), cpu.tolist())):
            ok = False
            bad = sum(1 for a, b in zip(dev.tolist(), cpu.tolist()) if int(a) != int(b))
            print(f"  {name}: {bad}/{len(cpu)} records' contributions differ")
        # the mutation: every key + 1 must move the noisy records
        if pack.edits[:, fe.E_NOISY].any():
            mut = pack.edits.copy()
            mut[:, fe.E_KEY] += 1
            dev2 = bp.edit_queue_contrib_probe(mut, pack.tiles, env,
                                               int(Salt.FIELD_NOISE))
            noisy = pack.edits[pack.tiles[:, 0], fe.E_NOISY] != 0
            if np.array_equal(dev2[noisy], dev[noisy]):
                ok = False
                print(f"  {name}: moving every edit key by one changed no noisy "
                      f"contribution (the noise is not keyed by the rank)")
    print(f"  noise: {n_rec} records ({n_noisy} noisy) device == CPU: {ok}")
    if n_noisy == 0:
        print("  noise: no noisy record (VACUOUS)")
        ok = False
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("EDIT_QUEUE_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    results = {}
    for name in SCENES:
        print(f"P4c {name}:")
        results[name] = run_scene(name)
    print("P4c fallback:")
    results["fallback"] = run_scene("fallback", fallback=True)
    a = _overlap_source_amount()
    print(f"  (the heat source deposits {a} Q16 counts a tick at its tile)")
    results["overlap"] = run_overlap() and a > 0
    results["refusal"] = run_refusal()
    results["noise"] = run_noise()
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print("EDIT_QUEUE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
