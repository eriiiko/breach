"""#29 P5 -- BATCH INVARIANCE: env i of a batch of N == env i stepped alone.

docs/residency_batching_design_2026-11-07.md §5.2. The batched resident
engine (simulation/resident_batch.py over physics_runner.run_resident_batch)
steps slot 7 of N envs in one device chain. Its gate: for every env i, the
state env i reaches in the batch -- every GameMap array and every book of its
engine, after EVERY tick -- equals, bit for bit, the state the same env reaches
stepped alone on the played resident path (PhysicsRunner.step ->
_step_resident, the batch of one).

PROPERTY: an env's evolution is independent of the batch it rides in (its
size, its neighbours' templates / seeds / substep counts / topology events /
resets). Changes that must break it: an env-stride bug in any (N, h, w)
kernel (a pointer not rebased by its env, a stencil reading the next env), a
masked env that is not masked (an env whose n_sub is smaller than the batch's
max doing a substep it should sit out), a per-env book folded into the wrong
engine, a per-env scalar read from env 0.

LEG A, synthetic (72 x 72), N in {1, 2, 8, 64}: four templates -- a blast room
(n_sub at the cap, blast sources, the burning wood), a calm room (low n_sub),
a calm room with a standing-water pool (the water stage live in some envs
only: the per-env water mask), a smoke room (combustion + a trace cloud);
env i is template i % 4 with a per-env seeded perturbation; ONLY env 0 has
topology events (a wall breached between ticks, and one AFTER the stamp: the
withheld-rows path, where the whole batch uploads its stamp planes); env 1 is
RESET mid-batch (a fresh calm-pool world from tick RESET_AT, its reference a
fresh world stepped alone). Evidence printed: the substep counts the envs ran
on the same tick (they must differ, or the masks were never exercised).

LEG B, real simulations: ship_demo's Simulation (no match, no units, its
entities on -- vents, ducts, doors), N = 3, seeds 1 / 2 / 3; env 1 takes a
blast source at tick 4, env 2 a wall breach at tick 6.

#29 P4c: every env of both legs also carries its OWN slot-6b FieldEdits
(smoke and heat discs with noise keyed by its seed, an atmosphere rect, a
wave_source tile, a fire MAX on the wood -- place and amount from its seed,
none on every third tick of its own phase), flushed by the device edit queue
in the batch: one records upload for every env, books per env. Breaks also if
a record lands in another env's planes or an env's books fold into another.

Run through tests/test_cuda_batch_invariance.py (the cuda_harness subprocess).
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np

import breach_physics as bp

import cuda_s8a_check as s8a

K_TICKS = 30
RESET_AT = 15
N_LIST = (1, 2, 8, 64)


# ---------------------------------------------------------------------------
# the worlds (an env: a runner + a map + its host slots around slot 7)
# ---------------------------------------------------------------------------
class RawWorld:
    """A physics-only env with Simulation's three-part tick: the scripted host
    edits and the unit stamp before slot 7, the burn-through walls and the
    end-of-tick heat clear after it (s8a's _one_tick, split)."""

    def __init__(self, runner, g, dt, events=None, edit_seed=None):
        self.physics_runner = runner
        self.gmap = g
        self._tps = int(round(1.0 / dt))
        self._dt = dt
        self.paused = False
        self.t = 0
        self.events = events or {}   # tick -> ("pre" | "late", (y, x)) or ("source", ...)
        # #29 P4c: slot 6b -- this env's own FieldEdits (they differ per env:
        # place, amount, noise key = the env's seed, which ticks carry them),
        # flushed as Simulation's 6b does: the device queue on the resident
        # path, the host flush otherwise.
        self.edit_seed = edit_seed
        if edit_seed is not None:
            from simulation.draws import MatchDraws
            from simulation.field_edit import EditQueue
            self.edit_queue = EditQueue()
            self.draws = MatchDraws(int(edit_seed), clock=lambda: self.t)

    def _step_pre_physics(self):
        ev = self.events.get(self.t)
        late = None
        if ev is not None:
            kind, arg = ev
            if kind == "wall":
                self.gmap.destroy_wall(*arg)
            elif kind == "source":
                from simulation.blast_expansion import add_source
                add_source(self.gmap, arg[0], arg[1])
            elif kind == "late":
                late = arg
        self.gmap.stamp_units([])
        if late is not None:
            self.gmap.destroy_wall(*late)
        if self.edit_seed is not None:
            for e in _env_edits(self.gmap, self.edit_seed, self.t):
                self.edit_queue.enqueue(e)
            r = self.physics_runner
            if r.serves_edit_queue():
                r.serve_edit_queue(self.gmap, self.edit_queue, self.draws)
            else:
                self.edit_queue.flush(self.gmap, self.draws)
        return True

    def _step_post_physics(self, destroyed):
        for (yy, xx) in destroyed:
            self.gmap.destroy_wall(yy, xx)
        self.gmap.heat.fill(0)
        self.t += 1

    def step_alone(self):
        self._step_pre_physics()
        destroyed = self.physics_runner.step(self.gmap, self._dt)
        self._step_post_physics(destroyed)


def _env_edits(g, seed, t):
    """#29 P4c: an env's slot-6b FieldEdits on tick t -- they DIFFER per env
    (the place and amount from its seed; the noise keyed by its seed) and an
    env has none on every third tick of its own phase, so a batch tick mixes
    envs with and without a queue."""
    from simulation.field_edit import EditMode, Falloff, FieldEdit, Region
    if (t + seed) % 3 == 0:
        return []
    y = 20 + (seed * 7 + t) % 25
    x = 20 + (seed * 13 + 2 * t) % 25
    a = 0.2 + 0.05 * (seed % 5)
    return [
        FieldEdit("smoke", Region.DISC, (y, x, 3.0), a, EditMode.ADD, Falloff.LINEAR,
                  noise=0.4, source_id=1),
        FieldEdit("heat", Region.DISC, (y + 2, x - 2, 2.5), 3.0 * a, EditMode.ADD,
                  Falloff.LINEAR, noise=0.2, source_id=2),
        FieldEdit("atmosphere", Region.RECT, (y - 2, x - 3, y - 1, x + 3), a,
                  EditMode.ADD, source_id=3),
        FieldEdit("wave_source", Region.TILE, (y, x + 4), 40.0 * a, EditMode.ADD,
                  source_id=4),
        FieldEdit("fire", Region.RECT, (46, 49, 49, 51), 0.5, EditMode.MAX, source_id=5),
    ]


def _perturb(g, seed):
    """The env's seed: a seeded O2 perturbation on a few open cells."""
    from simulation import atmosphere_fixed
    from simulation.gases import O2
    rng = np.random.default_rng(seed)
    # (small: a calm env's substep count stays below the cap)
    ys, xs = np.nonzero(~g.solid & ~g.is_vacuum)
    pick = rng.choice(len(ys), size=12, replace=False)
    sel = np.zeros(g.solid.shape, dtype=bool)
    for k in pick:
        g.gas[O2, ys[k], xs[k]] += int(atmosphere_fixed.quantize_scalar(
            float(rng.uniform(0.001, 0.02))))
        sel[ys[k], xs[k]] = True
    g.reseed_gas_energy(sel & g._gas_energy_accountable())


def _build(H, W, *, blast, pool, fire, smoke, wood, breach):
    """s8a's scenario room (a hull-ringed room breached to a vacuum band, a wood
    wall segment), its features switched per template; every temperature and
    bulk-gas write re-seeds gas_energy on the touched cells (the scenario-
    builder idiom, GameMap.reseed_gas_energy)."""
    from pathlib import Path

    from config import CFG
    from level_loader import LevelData
    from simulation import atmosphere_fixed, fire_fixed, water_fixed
    from simulation.gamemap import GameMap
    from simulation.gases import O2
    from simulation.physics_runner import PhysicsRunner

    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:H - 2, 2:W - 2] = 1
    tm[3:H - 3, 3:W - 3] = 4
    if breach:
        tm[H // 2 - 2:H // 2 + 2, W - 3] = 4  # breach the east hull to the vacuum band
    tm[46:50, 50] = 2                        # a wood wall segment
    level = LevelData(name="p5_batch", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    q = atmosphere_fixed.quantize_scalar
    touched = np.zeros((H, W), dtype=bool)
    if blast:
        g.temperature[8:18, 8:18] += q(5000.0)
        g.gas[O2, 10:16, 10:16] += q(4.0)
        touched[8:18, 8:18] = True
    if pool:
        g.water_depth[H - 8:H - 4, 6:W // 2] = water_fixed.quantize_scalar(0.4)
    if fire:
        g.fire[12:15, 12:15] = fire_fixed.quantize_scalar(0.8)
    if smoke:
        trace_ids = [gi for gi in range(g.gas.shape[0])
                     if not bool(g.gases.conservative[gi])]
        g.gas[trace_ids[0], 20:40, 20:40] += q(0.5)
    if wood:
        s8a._light_wood(g, [(47, 50), (48, 50)])
    if touched.any():
        g.reseed_gas_energy(touched & g._gas_energy_accountable())
    runner = PhysicsRunner(bp)
    g.bind_physics_engine(runner.engine)
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, dt


_FEATURES = {
    "blast": dict(blast=True, pool=False, fire=True, smoke=True, wood=True, breach=True),
    "calm":  dict(blast=False, pool=False, fire=False, smoke=False, wood=False, breach=False),
    "pool":  dict(blast=False, pool=True, fire=False, smoke=False, wood=False, breach=False),
    "smoke": dict(blast=False, pool=False, fire=False, smoke=True, wood=True, breach=True),
}


def make_world(template, seed, events=False):
    """An env of `template`, its seeded perturbation, and -- for the blast
    template with `events` -- the topology events (a wall breached between
    ticks; a wall breached AFTER the stamp: the withheld-rows path)."""
    s8a._residency(False)
    runner, g, dt = _build(72, 72, **_FEATURES[template])
    ev = {}
    if template == "blast":
        from simulation import gas_fixed
        ev[5] = ("source", ([(y, x) for y in range(10, 16) for x in range(10, 16)],
                            gas_fixed.quantize_q16_scalar(4.0)))
        if events:
            ev[12] = ("wall", (6, 6))
            ev[20] = ("late", (46, 50))
    _perturb(g, seed)
    return RawWorld(runner, g, dt, ev, edit_seed=seed)


TEMPLATES = ("blast", "calm", "pool", "smoke")


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------
def _fingerprint(world):
    g, r = world.gmap, world.physics_runner
    arrays = {k: hashlib.sha1(np.ascontiguousarray(v).view(np.uint8)).hexdigest()
              for k, v in sorted(s8a._arrays(g).items())}
    books = repr(sorted(s8a._all_books(g, r).items()))
    return arrays, hashlib.sha1(books.encode()).hexdigest(), books


def _diff(tag, fa, fb):
    (aa, ba, bka), (ab, bb, bkb) = fa, fb
    out = [k for k in aa if aa[k] != ab.get(k)]
    msg = f"{tag}: arrays differ {out}" if out else ""
    if ba != bb:
        import ast
        da, db = dict(ast.literal_eval(bka)), dict(ast.literal_eval(bkb))
        bk = [k for k in da if da[k] != db.get(k)]
        msg += f" books differ {bk[:8]}"
    return msg


def run_alone(world, n_ticks):
    fps = []
    s8a._residency(True); s8a._set_backends(True)
    try:
        for _ in range(n_ticks):
            world.step_alone()
            fps.append(_fingerprint(world))
    finally:
        s8a._residency(False); s8a._set_backends(False)
    return fps


def _free(world):
    g = world.gmap
    for k in ("_dev", "_batch"):
        g.__dict__.pop(k, None)
    world.physics_runner.__dict__.pop("_batch_water_before", None)
    world.physics_runner._d_water_before = None


def leg_a(N) -> bool:
    import cupy as cp
    from simulation.resident_batch import ResidentBatch
    print(f"LEG A, N = {N}: {K_TICKS} ticks, templates {TEMPLATES} (env i = i % 4), "
          f"env 0 with topology events, env 1 reset at tick {RESET_AT}"
          if N > 1 else f"LEG A, N = 1: {K_TICKS} ticks")
    specs = [(TEMPLATES[i % 4], 1000 + i, i == 0) for i in range(N)]
    # the references, one env at a time (alone; the device memory freed)
    refs = []
    for (tpl, seed, ev) in specs:
        w = make_world(tpl, seed, ev)
        refs.append(run_alone(w, K_TICKS))
        _free(w)
        del w
        cp.get_default_memory_pool().free_all_blocks()
    reset_ref = None
    if N > 1:
        w = make_world("pool", 4242, False)
        reset_ref = run_alone(w, K_TICKS - RESET_AT)
        _free(w)
        del w
        cp.get_default_memory_pool().free_all_blocks()
    # the batch
    worlds = [make_world(tpl, seed, ev) for (tpl, seed, ev) in specs]
    s8a._residency(True); s8a._set_backends(True)
    bad = 0
    nsub_by_tick = []
    try:
        batch = ResidentBatch(worlds)
        for t in range(K_TICKS):
            if N > 1 and t == RESET_AT:
                fresh = make_world("pool", 4242, False)
                s8a._residency(True)   # make_world builds with residency off
                batch.reset_env(1, fresh)
                worlds[1] = fresh
            batch.step()
            nsub_by_tick.append([int(w.physics_runner.eos.dbg_last_n_sub) for w in worlds])
            for i, w in enumerate(worlds):
                fp = _fingerprint(w)
                if N > 1 and i == 1 and t >= RESET_AT:
                    ref = reset_ref[t - RESET_AT]
                else:
                    ref = refs[i][t]
                if fp[0] != ref[0] or fp[1] != ref[1]:
                    bad += 1
                    if bad <= 6:
                        print("  " + _diff(f"N={N} env {i} ({specs[i][0]}) tick {t}", fp, ref))
    finally:
        s8a._residency(False); s8a._set_backends(False)
    distinct = [len(set(row)) for row in nsub_by_tick]
    mixed = sum(1 for d in distinct if d > 1)
    if N > 1:
        sample = nsub_by_tick[min(3, K_TICKS - 1)][:8]
        print(f"  substep counts on one tick (tick 3, envs 0..7): {sample}; ticks with "
              f"DIFFERENT n_sub across envs: {mixed}/{K_TICKS}")
        if mixed == 0:
            print("  FAIL: n_sub never differed across envs -- the masks were not exercised")
            bad += 1
    if bad == 0:
        print(f"  N={N}: every env == alone, every tick, every array and book (tol 0)")
    for w in worlds:
        _free(w)
    del batch, worlds
    cp.get_default_memory_pool().free_all_blocks()
    return bad == 0


# ---------------------------------------------------------------------------
# LEG B: real simulations (entities on)
# ---------------------------------------------------------------------------
def _ship_sim(seed):
    from level_loader import load as load_level
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    s8a._residency(False)
    sim = Simulation(load_level("ship_demo"), seed=seed, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.physics_runner.headless = True
    sim.set_paused(False)
    return sim


def _ship_events(sim, e, t):
    from simulation import gas_fixed
    from simulation.blast_expansion import add_source
    g = sim.gmap
    # #29 P4c: each env's own FieldEdits at an open tile of its own choosing
    from simulation.field_edit import EditMode, Falloff, FieldEdit, Region
    if (t + e) % 3 != 0:
        ys, xs = np.nonzero(~g.solid & ~g.is_vacuum)
        k = (len(ys) * (e + 1)) // 5
        y, x = int(ys[k]), int(xs[k])
        a = 0.2 + 0.1 * e
        sim.edit(FieldEdit("smoke", Region.DISC, (y, x, 3.0), a, EditMode.ADD,
                           Falloff.LINEAR, noise=0.4, source_id=1))
        sim.edit(FieldEdit("heat", Region.DISC, (y, x, 2.0), 3.0 * a, EditMode.ADD,
                           Falloff.LINEAR, noise=0.3, source_id=2))
        sim.edit(FieldEdit("atmosphere", Region.RECT, (y - 1, x - 2, y + 1, x + 2), a,
                           EditMode.ADD, source_id=3))
        sim.edit(FieldEdit("wave_source", Region.TILE, (y, x), 40.0 * a, EditMode.ADD,
                           source_id=4))
    if e == 1 and t == 4:
        ys, xs = np.nonzero(~g.solid & ~g.is_vacuum)
        k = len(ys) // 2
        add_source(g, [(int(ys[k]), int(xs[k]))], gas_fixed.quantize_q16_scalar(3.0))
    if e == 2 and t == 6:
        ys, xs = np.nonzero(g.solid)
        k = len(ys) // 3
        g.destroy_wall(int(ys[k]), int(xs[k]))


def _sim_fp(sim):
    class _W:
        pass
    w = _W()
    w.gmap, w.physics_runner = sim.gmap, sim.physics_runner
    return _fingerprint(w)


def leg_b() -> bool:
    import cupy as cp
    from simulation.resident_batch import ResidentBatch
    n, ticks = 3, 20
    print(f"LEG B: ship_demo Simulations (no match, entities on), N = {n}, seeds "
          f"1/2/3, env 1 a blast source at tick 4, env 2 a wall breach at tick 6, "
          f"{ticks} ticks")
    refs = []
    for e in range(n):
        sim = _ship_sim(e + 1)
        fps = []
        s8a._residency(True); s8a._set_backends(True)
        try:
            for t in range(ticks):
                _ship_events(sim, e, t)
                sim.step()
                fps.append(_sim_fp(sim))
        finally:
            s8a._residency(False); s8a._set_backends(False)
        refs.append(fps)
        sim.gmap.__dict__.pop("_dev", None)
        del sim
        cp.get_default_memory_pool().free_all_blocks()
    sims = [_ship_sim(e + 1) for e in range(n)]
    bad = 0
    s8a._residency(True); s8a._set_backends(True)
    try:
        batch = ResidentBatch(sims)
        for t in range(ticks):
            for e, sim in enumerate(sims):
                _ship_events(sim, e, t)
            batch.step()
            for e, sim in enumerate(sims):
                fp = _sim_fp(sim)
                if fp[0] != refs[e][t][0] or fp[1] != refs[e][t][1]:
                    bad += 1
                    if bad <= 6:
                        print("  " + _diff(f"ship env {e} tick {t}", fp, refs[e][t]))
    finally:
        s8a._residency(False); s8a._set_backends(False)
    if bad == 0:
        print(f"  ship_demo: every env == alone, every tick (tol 0)")
    return bad == 0


def main() -> int:
    print(f"batch invariance (#29 P5), HAS_CUDA={getattr(bp, 'HAS_CUDA', False)}")
    ok = True
    for N in N_LIST:
        ok = leg_a(N) and ok
    ok = leg_b() and ok
    print("BATCH_INVARIANCE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
