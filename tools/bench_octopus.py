"""tools/bench_octopus.py -- what an octopus costs, per tick, BY PART (#106 P6e).

REPORTED, never gated (Erik's ruling: cost is not a design input until the
final result is seen; these are the numbers he judges by). Nothing here has a
threshold.

    C:/Users/steen/anaconda3/python.exe tools/bench_octopus.py
    C:/Users/steen/anaconda3/python.exe tools/bench_octopus.py --cuda [--resident]
    C:/Users/steen/anaconda3/python.exe tools/bench_octopus.py \\
        --scenarios crawl,ship --sim-seconds 30 --repeat 2 --json out.json

Scenarios (``--scenarios``, default all):

* ``idle`` / ``crawl`` / ``swim`` / ``grab`` -- one octopus in a flooded
  30 x 30 room (the P6d2 gate room, built by ``tests/_creature_scenarios.
  write_room``) driven by its unit's intents the way the gates drive it:
  nothing (idle); MOVE_DIR round a square, 3 s per side (crawl); the
  umbrella swim (MOVE_DIR at ORDER_MOVE_COVER) round the same square from
  8 m up (swim); the demo prop, 0.5 s of crawl, USE, then the square (grab).
* ``hunt`` -- the ``creature`` match controller hunting two idle marines in
  the same room, through ``run_match`` (the one headless match path).
* ``octopi`` -- ``--octopi`` (default 1,2,4,8) animals crawling round the
  square together in a flooded 56 x 36 room (a 2 x 4 grid, pitch 9 tiles).
* ``ship`` -- ``tools/gen_ship.py --seed 1 --size medium --vents`` (the #125
  tick-budget ship), its default match (4 v 4 bots, 4 zombies) plus ONE
  octopus through a ``[[creature]]`` match row (the only spot on that ship
  where the rest pose fits: the zombies' room), through ``run_match``. The
  horde STANDS (``zombie_controller = "idle"``, with and without the
  octopus): on its own controller it bites the octopus dead by tick 43 (it
  shares their room), and #125 measured the horde at ~0 ms on these ships.

Every scenario also runs WITHOUT its octopus (the baseline: same room, same
match, no ``[[creature]]``), each baseline right before its octopus run, so
the rest of ``sim.step`` is measured on the same map. On CUDA a process slows
from run to run (#125's repeat slowdown), so GPU numbers are taken with the
octopus run and its baseline in SEPARATE processes: ``--no-baseline`` and
``--baseline-only``, one scenario each.

Method: ``creature_system.set_tick_laps`` attaches a part timer to slot 5b
(OFF by default: a no-op method call per part, no clock read, no state;
``--verify-inert`` re-runs each scenario with it detached and compares the
final ``tick_digest``). Each lap is a ``time.perf_counter_ns`` read between
the real calls, in the real order:

    setup      the unit lookup and the footprint rects (once per tick)
    hub        route_intents + hub_step + after_solver (creature_primitives)
    inputs     write_inputs (other units' rects, topology on a change)
    chain      arm_chain.tick -> the C++ ``arm_chain_tick`` binding
    unit       write_unit's pose write-back
    hit_shape  build_hit_shape (the P6d3 hit seam)

``slot5b`` is their sum. ``record`` is ``serialize_record`` over every body,
timed after the step -- what a recorder or a per-tick digest pays (a headless
match pays it once at the end). ``step`` is the whole ``sim.step()``,
``rest`` = step - slot5b, ``ctrl`` the controllers' share in a match (the
``bench_bot_matches.TickTimer`` split: tick 1 dropped, watcher time
excluded). The first ``--warmup`` ticks are dropped too.

``--chain-micro`` also times the solver call ALONE on a copy of a crawling
octopus's buffers: at substeps 16 / 12 / 8 (the count overridden for TIMING
only) and one call over N stacked animals against N single calls.

Also reported: the chain's ns per weighted op against P6c's G7 tally (28098
weighted ops per arm-substep x arms x substeps = ~3.6 M per octopus-tick),
and the solver's ms per arm-substep -- the P1 ceiling's "1 ns/op" assumption
against the measurement, with no verdict.

Pure core (``PartLaps``, ``summarize``, ``markdown``) + scenario runners +
thin CLI. The CUDA path is ``bench_bot_matches.setup_cuda`` (``tools/
run_on_cuda.py``'s backend list); the chain is host-side on every backend.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tests", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from bench_bot_matches import TickTimer, setup_cuda  # noqa: E402

#: The slot-5b parts, in call order.
PARTS = ("setup", "hub", "inputs", "chain", "unit", "hit_shape")
#: P6c's G7 tally (docs/octopus_integer_reference/report_p6c.md).
WEIGHTED_OPS_PER_ARM_SUBSTEP = 28098
#: The scenarios, in report order.
SCENARIOS = ("idle", "crawl", "swim", "grab", "hunt", "octopi", "ship")

GATE_ROOM = 30                 # the P6d2 gate room, tiles per side
SQUARE_SIDE_S = 3.0            # the square's side, s of MOVE_DIR
SWIM_Z0_M = 8.0                # the swim's start height (P5z / _creature_scenarios.swim)
GRAB_USE_TICK = 12             # 0.5 s of crawl, then USE (_creature_scenarios.grab)
PACK_W, PACK_H, PACK_PITCH = 56, 36, 9   # the --octopi room and its 2 x 4 grid


# ===========================================================================
# The pure core
# ===========================================================================
class PartLaps:
    """Slot 5b's part timer (``creature_system.set_tick_laps``): ``lap(part)``
    charges the ns since the previous lap to ``part``; "enter" opens a tick
    (laps outside an open tick -- a spawn's ``write_unit`` -- are ignored).
    ``take()`` returns this tick's ns per part and closes it."""

    def __init__(self, clock=time.perf_counter_ns):
        self._clock = clock
        self._mark = None
        self._cur = dict.fromkeys(PARTS, 0)

    def lap(self, part: str) -> None:
        now = self._clock()
        if part == "enter":
            self._mark = now
            return
        if self._mark is None:
            return
        self._cur[part] += now - self._mark
        self._mark = now

    def take(self) -> dict:
        out, self._cur, self._mark = self._cur, dict.fromkeys(PARTS, 0), None
        return out


@dataclass
class Run:
    """One run of one scenario: per-tick ns samples (after the warm-up)."""
    scenario: str
    n_octopi: int
    ticks: list = field(default_factory=list)   # per tick: {part: ns, step, ctrl, record}
    digest: str = ""
    meta: dict = field(default_factory=dict)


def _stats_ms(ns) -> dict:
    import numpy as np
    a = np.asarray(ns, dtype=np.float64) / 1e6
    if a.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def summarize(runs, baseline_runs=(), *, tps: float, arms: int, substeps: int) -> dict:
    """The report of one scenario from its runs (repeats pooled) and its
    baseline runs (the same scenario without the octopus): per part and
    per column mean / p50 / p95 / max ms, the per-octopus means, the share of
    a tick, the real-time ratio, the chain's ns per weighted op."""
    rows = [t for r in runs for t in r.ticks]
    n = runs[0].n_octopi if runs else 0
    cols = {}
    for k in PARTS:
        cols[k] = _stats_ms([t[k] for t in rows])
    slot = [sum(t[k] for k in PARTS) for t in rows]
    cols["slot5b"] = _stats_ms(slot)
    cols["record"] = _stats_ms([t["record"] for t in rows])
    cols["step"] = _stats_ms([t["step"] for t in rows])
    cols["rest"] = _stats_ms([t["step"] - s for t, s in zip(rows, slot)])
    cols["ctrl"] = _stats_ms([t["ctrl"] for t in rows])
    total = [t["step"] + t["ctrl"] for t in rows]
    cols["total"] = _stats_ms(total)
    tick_ms = 1e3 / tps
    # per octopus: over the octopus-ticks that ran (a dead creature is frozen
    # and costs nothing; a tick's "n" is its living creature count)
    alive = sum(t.get("n", n) for t in rows)
    per = {k: sum(t[k] for t in rows) / 1e6 / max(1, alive) for k in PARTS + ("record",)}
    per["slot5b"] = sum(slot) / 1e6 / max(1, alive)
    ops = WEIGHTED_OPS_PER_ARM_SUBSTEP * arms * substeps
    chain_ms = per["chain"]
    out = {
        "scenario": runs[0].scenario if runs else "", "n_octopi": n,
        "ticks": len(rows), "repeats": len(runs), "octopus_ticks": alive,
        "cols": cols,
        "per_octopus_ms": per,
        "share_of_tick": per["slot5b"] / tick_ms,
        "tick_ms": tick_ms,
        "realtime": (1e3 / cols["total"]["mean"]) / tps if cols["total"]["mean"] else 0.0,
        "repeat_slot5b_mean_ms": [
            _stats_ms([sum(t[k] for k in PARTS) for t in r.ticks])["mean"] for r in runs],
        "weighted_ops_per_octopus_tick": ops,
        "chain_ns_per_op": chain_ms * 1e6 / ops if ops else 0.0,
        "chain_ms_per_arm_substep": chain_ms / (arms * substeps) if arms * substeps else 0.0,
        "digests": sorted({r.digest for r in runs}),
    }
    base = [t for r in baseline_runs for t in r.ticks]
    if base:
        bstep = _stats_ms([t["step"] for t in base])
        btot = _stats_ms([t["step"] + t["ctrl"] for t in base])
        out["baseline"] = {"step": bstep, "ctrl": _stats_ms([t["ctrl"] for t in base]),
                           "total": btot,
                           "realtime": (1e3 / btot["mean"]) / tps if btot["mean"] else 0.0}
        out["delta_step_ms"] = cols["step"]["mean"] - bstep["mean"]
        out["delta_total_ms"] = cols["total"]["mean"] - btot["mean"]
    return out


def _f(x: float) -> str:
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}"


def markdown(reports, backend: str) -> str:
    """The reports as markdown tables: the per-octopus split, the tick, and
    the per-part distributions."""
    L = [f"### {backend}", "",
         "Per octopus per tick, mean ms (slot 5b by part; record = serialize_record, "
         "paid only when a recorder / per-tick digest asks):", "",
         "| scenario | N | setup | hub | inputs | chain | unit | hit_shape | **slot5b** "
         "| record | share of 41.7 ms | chain ns/op | chain ms/arm-substep |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        p = r["per_octopus_ms"]
        L.append(f"| {r['scenario']} | {r['n_octopi']} | "
                 + " | ".join(_f(p[k]) for k in PARTS)
                 + f" | **{_f(p['slot5b'])}** | {_f(p['record'])} | "
                 f"{100 * r['share_of_tick']:.2f} % | {r['chain_ns_per_op']:.3f} | "
                 f"{r['chain_ms_per_arm_substep']:.4f} |")
    L += ["", "The whole tick, ms (mean; step p95 in brackets). baseline = the same "
          "scenario without the octopus:", "",
          "| scenario | N | ticks | slot5b mean / p95 / max | step | rest of step "
          "| ctrl | baseline step | step - baseline | total - baseline | real-time x "
          "(baseline) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        c = r["cols"]
        b = r.get("baseline")
        bs = f"{_f(b['step']['mean'])}" if b else "-"
        ds = f"{r['delta_step_ms']:+.2f}" if b else "-"
        dt = f"{r['delta_total_ms']:+.2f}" if b else "-"
        rt = f"{r['realtime']:.2f}" + (f" ({b['realtime']:.2f})" if b else "")
        L.append(f"| {r['scenario']} | {r['n_octopi']} | {r['ticks']} | "
                 f"{_f(c['slot5b']['mean'])} / {_f(c['slot5b']['p95'])} / "
                 f"{_f(c['slot5b']['max'])} | {_f(c['step']['mean'])} "
                 f"({_f(c['step']['p95'])}) | {_f(c['rest']['mean'])} | "
                 f"{_f(c['ctrl']['mean'])} | {bs} | {ds} | {dt} | {rt} |")
    L += ["", "Per part, all octopi of the tick, ms mean / p50 / p95 / max:", "",
          "| scenario | N | " + " | ".join(PARTS + ("slot5b",)) + " |",
          "|---|---|" + "---|" * (len(PARTS) + 1)]
    for r in reports:
        c = r["cols"]
        L.append(f"| {r['scenario']} | {r['n_octopi']} | " + " | ".join(
            "/".join(_f(c[k][s]) for s in ("mean", "p50", "p95", "max"))
            for k in PARTS + ("slot5b",)) + " |")
    return "\n".join(L) + "\n"


# ===========================================================================
# The scenarios
# ===========================================================================
def _square_deg(k: int, tps: int) -> float:
    """The square's heading at tick ``k``: East, North, West, South, 3 s each."""
    return 90.0 * ((k // int(round(SQUARE_SIDE_S * tps))) % 4)


def _move(sim, unit, deg: float, mode: int = 0) -> None:
    from control_source import quantize_stick_direction
    r = math.radians(deg)
    dx, dy, _ = quantize_stick_direction(math.cos(r), -math.sin(r), deadzone=0.0)
    sim.set_move_dir(unit.id, dx, dy, int(mode))


def _lift(body, z_m: float) -> None:
    """``_creature_scenarios.Scenario.lift``: the swim's start in mid water."""
    from simulation import arm_chain as AC
    from simulation import creature_fixed as CF
    body.bufs.anchored[...] = 0
    dz = CF.chain_from_m(z_m)
    body.bufs.p[0, ..., 2] += dz
    body.bufs.body[0, 0, AC.BS_Z] += dz


def _pack_spawns(n: int) -> list:
    """``n`` octopi on the --octopi room's 2 x 4 grid (footprint top-left)."""
    from simulation.match import CreatureSpawn
    if not 1 <= n <= 8:
        raise ValueError(f"--octopi takes 1..8 animals, not {n}")
    out = []
    for i in range(n):
        cx = 11 + PACK_PITCH * (i % 4)
        cy = 12 + PACK_PITCH * (i // 4)
        out.append(CreatureSpawn("octopus", cx - 1.5, cy - 1.5))
    return out


def _timed_step(sim, laps, bodies) -> dict:
    t0 = time.perf_counter_ns()
    sim.step()
    t1 = time.perf_counter_ns()
    row = laps.take() if laps is not None else dict.fromkeys(PARTS, 0)
    row["step"] = t1 - t0
    row["ctrl"] = 0
    row["record"] = _time_record(bodies)
    row["n"] = len(bodies)
    return row


def _time_record(bodies) -> int:
    from simulation import creature_system as CS
    t0 = time.perf_counter_ns()
    for b in bodies:
        CS.serialize_record(b)
    return time.perf_counter_ns() - t0


def _final_digest(sim) -> str:
    from field_ab_harness import capture_snapshot
    from field_digest import tick_digest
    return tick_digest(capture_snapshot(sim))


def run_room(kind: str, *, ticks: int, warmup: int, n_octopi: int = 1,
             octopus: bool = True, timing: bool = True, seed: int = 7,
             tmp: Path | None = None) -> Run:
    """One room scenario (idle / crawl / swim / grab, or ``octopi`` with
    ``n_octopi``) for ``warmup + ticks`` ticks; ``octopus=False`` is its
    baseline (the same room, no creature). Steps the sim directly, the way
    ``_creature_scenarios.Scenario`` does (not a match)."""
    import breach_physics as bp
    import _creature_scenarios as SC
    from config import CFG
    from simulation import Simulation
    from simulation import creature_system as CS
    from simulation.match import CreatureSpawn
    from simulation.orders import ORDER_MOVE_COVER
    from simulation.ruleset import ContinuousRealtime
    tps = int(CFG.clock.ticks_per_second)
    # a fresh folder per run (repeats and baselines reuse the level name)
    tmp = Path(tempfile.mkdtemp(prefix="room_", dir=None if tmp is None else str(tmp)))
    if kind == "octopi":
        w, h = PACK_W, PACK_H
        rows = _pack_spawns(n_octopi)
    else:
        w = h = GATE_ROOM
        rows = [CreatureSpawn("octopus", w / 2.0 - 1.5, h / 2.0 - 1.5)]
        n_octopi = 1
    name = f"{kind}_{n_octopi}_{int(octopus)}_{int(timing)}"
    level = SC.write_room(tmp, name, w=w, h=h, creatures=tuple(rows) if octopus else ())
    sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    units = sim.spawn_creatures(level.creatures) if octopus else []
    bodies = [sim._creature_system.bodies[u.id] for u in units]
    if kind == "swim":
        for b in bodies:
            _lift(b, SWIM_Z0_M)
    if kind == "grab":
        for b in bodies:
            b.add_prop()
    laps = PartLaps() if timing else None
    prev = CS.set_tick_laps(laps)
    run = Run(kind, n_octopi if octopus else 0,
              meta={"room": f"{w}x{h}", "octopus": octopus, "timing": timing})
    try:
        for k in range(warmup + ticks):
            for u in units:
                if kind == "grab" and k == GRAB_USE_TICK:
                    sim.use_intent(u.id)
                if kind in ("crawl", "octopi", "grab"):
                    _move(sim, u, _square_deg(k, tps))
                elif kind == "swim":
                    _move(sim, u, _square_deg(k, tps), ORDER_MOVE_COVER)
            sim.set_paused(False)
            row = _timed_step(sim, laps, bodies)
            if k >= warmup:
                run.ticks.append(row)
    finally:
        CS.set_tick_laps(prev)
    run.digest = _final_digest(sim)
    return run


def chain_micro(*, settle_ticks: int = 72, calls: int = 200, batch=(1, 2, 4, 8),
                substeps=(16, 12, 8), tmp: Path | None = None) -> dict:
    """The solver call OUT of the sim, on a COPY of a crawling octopus's
    buffers (``settle_ticks`` of the crawl scenario first): ms per call at the
    row's substeps and with the row's substep count overridden (TIMING ONLY --
    the other row constants keep the shipped substeps' dt, so the motion is
    not a valid 12- or 8-substep animal; the cost scales with the count, which
    is what this measures), and one call over N stacked copies of the buffers
    against N single calls (the batching lever). Never touches a sim's state."""
    import numpy as np
    from dataclasses import fields as dc_fields
    import breach_physics as bp
    from simulation import arm_chain as AC
    tmp = Path(tempfile.mkdtemp(prefix="micro_")) if tmp is None else Path(tmp)
    _sim, bodies = _crawl_bodies(settle_ticks, tmp)
    body = bodies[0]
    row, row_node = body.row.copy(), body.solver_row_node.copy()

    def per_call_ms(r, bufs_list) -> float:
        work = [b.copy() for b in bufs_list]
        for b in work:                     # one untimed call: caches warm
            AC.tick(r, row_node, b, bp)
        work = [b.copy() for b in bufs_list]
        t0 = time.perf_counter_ns()
        for _ in range(calls):
            for b in work:
                AC.tick(r, row_node, b, bp)
        return (time.perf_counter_ns() - t0) / 1e6 / calls

    out = {"settle_ticks": settle_ticks, "calls": calls, "substeps": {}, "batch": {}}
    for s in substeps:
        r = row.copy()
        r[AC.ROW_N_SUB] = int(s)
        out["substeps"][int(s)] = per_call_ms(r, [body.bufs])
    for n in batch:
        stacked = AC.ArmChainBuffers(**{
            f.name: np.ascontiguousarray(np.concatenate([getattr(body.bufs, f.name)] * n))
            for f in dc_fields(AC.ArmChainBuffers)})
        one_call = per_call_ms(row, [stacked])
        n_calls = per_call_ms(row, [body.bufs] * n)
        out["batch"][int(n)] = {"one_call_ms": one_call, "n_calls_ms": n_calls}
    return out


def _crawl_bodies(ticks: int, tmp: Path):
    """A crawling octopus in the gate room after ``ticks`` ticks (the micro's
    state)."""
    import breach_physics as bp
    import _creature_scenarios as SC
    from config import CFG
    from simulation import Simulation
    from simulation.match import CreatureSpawn
    from simulation.ruleset import ContinuousRealtime
    tps = int(CFG.clock.ticks_per_second)
    level = SC.write_room(Path(tempfile.mkdtemp(prefix="room_", dir=str(tmp))), "micro",
                          w=GATE_ROOM, h=GATE_ROOM,
                          creatures=(CreatureSpawn("octopus", GATE_ROOM / 2.0 - 1.5,
                                                   GATE_ROOM / 2.0 - 1.5),))
    sim = Simulation(level, seed=7, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    units = sim.spawn_creatures(level.creatures)
    for k in range(int(ticks)):
        for u in units:
            _move(sim, u, _square_deg(k, tps))
        sim.set_paused(False)
        sim.step()
    return sim, [sim._creature_system.bodies[u.id] for u in units]


def micro_markdown(m: dict) -> str:
    s16 = m["substeps"].get(16)
    L = ["The solver call alone (a copy of a crawling octopus's buffers, "
         f"{m['calls']} calls each):", "",
         "| substeps | ms per call | vs 16 |", "|---|---|---|"]
    for s, ms in sorted(m["substeps"].items(), reverse=True):
        L.append(f"| {s} | {ms:.3f} | {ms / s16:.2f} |" if s16 else f"| {s} | {ms:.3f} | - |")
    L += ["", "| N animals | one (N, 8, n, 3) call, ms | N single calls, ms | per animal, batched "
          "| batched / single |", "|---|---|---|---|---|"]
    for n, b in sorted(m["batch"].items()):
        L.append(f"| {n} | {b['one_call_ms']:.3f} | {b['n_calls_ms']:.3f} | "
                 f"{b['one_call_ms'] / n:.3f} | {b['one_call_ms'] / b['n_calls_ms']:.3f} |")
    return "\n".join(L) + "\n"


class _MatchWatch:
    """``run_match``'s ``on_tick`` inner watcher (inside a ``TickTimer``):
    this tick's slot-5b laps and the record's cost."""

    def __init__(self, laps):
        self.laps = laps
        self.rows = []

    def __call__(self, sim):
        from simulation.creature_system import is_creature
        row = self.laps.take() if self.laps is not None else dict.fromkeys(PARTS, 0)
        bodies = sim._creature_system.ordered_bodies()
        row["record"] = _time_record(bodies)
        row["n"] = sum(1 for u in sim.units if u.alive and is_creature(u))
        self.rows.append(row)


def run_match_scenario(level, spec, *, ticks: int, warmup: int, tps: float,
                       timing: bool = True, scenario: str = "match") -> Run:
    """A match scenario through ``run_match`` (the one headless match loop),
    its time limit set to ``warmup + ticks`` ticks (+1 for the TickTimer's
    dropped first tick); per tick the TickTimer's step / ctrl split and the
    watcher's laps."""
    from run_match import run_match
    from simulation import creature_system as CS
    spec = replace(spec, time_limit_s=(warmup + ticks + 1) / float(tps))
    laps = PartLaps() if timing else None
    watch = _MatchWatch(laps)
    timer = TickTimer(watch)
    prev = CS.set_tick_laps(laps)
    try:
        o = run_match(level, spec, on_tick=timer)
    finally:
        CS.set_tick_laps(prev)
    rows = watch.rows[1:]       # TickTimer drops tick 1
    for row, st, ct in zip(rows, timer.step_s, timer.ctrl_s):
        row["step"] = int(st * 1e9)
        row["ctrl"] = int(ct * 1e9)
    rows = rows[:len(timer.step_s)][warmup:]
    n = len(spec.creatures)
    run = Run(scenario, n, ticks=rows, digest=o.tick_digest,
              meta={"result": o.result.outcome, "ticks_run": o.result.ticks,
                    "creatures_alive_min": min((r["n"] for r in rows), default=0)})
    return run


def hunt_level(tmp: Path, octopus: bool = True):
    """The gate room with two idle marines (one per team) and, with
    ``octopus``, one octopus on the ``creature`` controller hunting them."""
    import _creature_scenarios as SC
    from simulation.match import CreatureSpawn, MatchSpec, MatchTeam, SpawnRoom
    team = MatchTeam(n=1, loadout="rifleman", env_profile="default", controller="idle")
    rooms = [SpawnRoom(team=0, x=20, y=3, w=8, h=8, max_per_team=1),
             SpawnRoom(team=1, x=20, y=19, w=8, h=8, max_per_team=1)]
    creatures = (CreatureSpawn("octopus", 10.5, 13.5),) if octopus else ()
    spec = MatchSpec(teams=(team, team), seed=3, creatures=creatures)
    name = f"hunt_{int(octopus)}"
    return SC.write_room(tmp, name, w=GATE_ROOM, h=GATE_ROOM, rooms=rooms, match=spec)


#: The one spot on gen_ship seed 1 medium where the octopus's rest pose fits
#: (every node on open floor): the zombies' room. Footprint top-left, heading.
SHIP_OCTOPUS = (48.0, 26.0, 0.0)


def ship_level(tmp: Path, octopus: bool = True):
    """``gen_ship --seed 1 --size medium --vents`` (the #125 ship) with its
    default match (bots v bots, the horde) and, with ``octopus``, one
    ``[[creature]]`` row written through ``level_lib.write_match``."""
    import gen_ship
    import level_lib
    import level_loader
    from simulation.match import CreatureSpawn
    plan = gen_ship.generate(1, "medium", vents=True)
    name = f"ship_s1_medium_{int(octopus)}"
    out = gen_ship.write(plan, Path(tmp) / name)
    # the horde stands: it shares the octopus's room and would eat it (module doc)
    spec = replace(plan.match_spec(), zombie_controller="idle")
    if octopus:
        x, y, hd = SHIP_OCTOPUS
        spec = replace(spec, creatures=(CreatureSpawn("octopus", x, y, heading=hd),))
    level_lib.write_match(out, spec)
    return level_loader.load(name, str(tmp))


# ===========================================================================
# The CLI
# ===========================================================================
def _machine() -> str:
    import subprocess
    cpu = platform.processor()
    try:
        txt = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True,
                             text=True, timeout=10).stdout
        lines = [x.strip() for x in txt.splitlines() if x.strip() and x.strip() != "Name"]
        if lines:
            cpu = lines[0]
    except Exception:  # noqa: BLE001 -- wmic is optional
        pass
    return f"{platform.node()} / {cpu} / Python {platform.python_version()}"


def _ints(text: str) -> list:
    return [int(s) for s in text.split(",") if s.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench_octopus", description=__doc__.split("\n")[0])
    ap.add_argument("--scenarios", default=",".join(SCENARIOS),
                    help=f"comma list of {', '.join(SCENARIOS)}")
    ap.add_argument("--octopi", default="1,2,4,8",
                    help="animal counts for the 'octopi' scenario (1..8)")
    ap.add_argument("--sim-seconds", type=float, default=20.0,
                    help="timed sim seconds per run (after the warm-up)")
    ap.add_argument("--warmup", type=int, default=24, help="ticks dropped first")
    ap.add_argument("--repeat", type=int, default=1, help="runs per scenario (pooled)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the runs without the octopus")
    ap.add_argument("--baseline-only", action="store_true",
                    help="run only the baselines (no octopus): on CUDA, take the "
                         "octopus run and its baseline in separate processes")
    ap.add_argument("--verify-inert", action="store_true",
                    help="re-run each scenario with the timer detached; the "
                         "final tick_digest must be identical")
    ap.add_argument("--chain-micro", action="store_true",
                    help="also time the solver call alone: substeps 16/12/8 "
                         "(timing only) and N batched animals vs N calls")
    ap.add_argument("--json", default=None, help="write the reports here")
    ap.add_argument("--cuda", action="store_true",
                    help="run on the CUDA build, every live backend on")
    ap.add_argument("--cuda-build", default=str(ROOT / "cpp" / "build_cuda"))
    ap.add_argument("--resident", action="store_true",
                    help="with --cuda: GPU field residency on")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.cuda:
        setup_cuda(Path(args.cuda_build), args.resident)
    import breach_physics as bp
    from config import CFG
    from simulation.creatures import get_creature
    tps = int(CFG.clock.ticks_per_second)
    cdef = get_creature("octopus")
    arms, substeps = int(cdef.body.arms), int(cdef.solver.substeps)
    ticks = int(round(args.sim_seconds * tps))
    backend = "cpu"
    if args.cuda:
        backend = "cuda+resident" if args.resident else "cuda"
    chosen = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    for s in chosen:
        if s not in SCENARIOS:
            raise SystemExit(f"unknown scenario {s!r} (have {', '.join(SCENARIOS)})")
    machine = _machine()
    print(f"bench_octopus backend={backend} module={Path(bp.__file__).parent} "
          f"tps={tps} arms={arms} substeps={substeps} ticks={ticks} "
          f"warmup={args.warmup} repeat={args.repeat}\nmachine: {machine}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="bench_octopus_"))
    reports, inert = [], []

    def one(label, n, make_run, make_base):
        # baseline and octopus runs INTERLEAVED in pairs (baseline first): a
        # CUDA process slows from run to run (#125's repeat slowdown; here up
        # to +50 % on the resident step by a process's fourth run), so a
        # baseline taken after every octopus run would charge the drift to
        # the octopus. For GPU numbers run one scenario per process.
        runs, base = [], []
        if args.baseline_only:
            base = [make_base() for _ in range(args.repeat)]
            rep = summarize(base, tps=tps, arms=arms, substeps=substeps)
            rep["scenario"], rep["meta"], rep["baseline_only"] = label, base[0].meta, True
            reports.append(rep)
            print(f"  {label} N={n} baseline: step mean "
                  f"{rep['cols']['step']['mean']:.2f} ms", flush=True)
            return
        for _ in range(args.repeat):
            if not args.no_baseline:
                base.append(make_base())
            t0 = time.perf_counter()
            runs.append(make_run(True))
            print(f"  {label} N={n}: {len(runs[-1].ticks)} ticks in "
                  f"{time.perf_counter() - t0:.1f} s, digest {runs[-1].digest[:12]} "
                  f"{runs[-1].meta}", flush=True)
        if args.verify_inert:
            off = make_run(False)
            ok = off.digest == runs[0].digest
            inert.append((label, n, ok))
            print(f"  {label} N={n}: timer detached digest {off.digest[:12]} "
                  f"{'IDENTICAL' if ok else 'DIFFERS'}", flush=True)
        rep = summarize(runs, base, tps=tps, arms=arms, substeps=substeps)
        rep["scenario"], rep["meta"] = label, runs[0].meta
        reports.append(rep)

    for s in chosen:
        if s in ("idle", "crawl", "swim", "grab"):
            one(s, 1,
                lambda timing, s=s: run_room(s, ticks=ticks, warmup=args.warmup,
                                             timing=timing, tmp=tmp),
                lambda s=s: run_room(s, ticks=ticks, warmup=args.warmup,
                                     octopus=False, tmp=tmp))
        elif s == "octopi":
            for n in _ints(args.octopi):
                one("octopi", n,
                    lambda timing, n=n: run_room("octopi", ticks=ticks, warmup=args.warmup,
                                                 n_octopi=n, timing=timing, tmp=tmp),
                    lambda n=n: run_room("octopi", ticks=ticks, warmup=args.warmup,
                                         n_octopi=n, octopus=False, tmp=tmp))
        elif s in ("hunt", "ship"):
            make = hunt_level if s == "hunt" else ship_level
            sub = tmp / f"{s}_{len(reports)}"
            sub.mkdir()
            lv, lv0 = make(sub, True), (None if args.no_baseline else make(sub, False))
            one(s, 1,
                lambda timing, lv=lv, s=s: run_match_scenario(
                    lv, lv.match, ticks=ticks, warmup=args.warmup, tps=tps,
                    timing=timing, scenario=s),
                lambda lv0=lv0, s=s: run_match_scenario(
                    lv0, lv0.match, ticks=ticks, warmup=args.warmup, tps=tps,
                    timing=True, scenario=s + "_base"))
    micro = None
    if args.chain_micro:
        micro = chain_micro(tmp=tmp)
    print()
    print(markdown(reports, backend))
    if micro is not None:
        print(micro_markdown(micro))
    if inert:
        bad = [x for x in inert if not x[2]]
        print(f"timer inert: {len(inert) - len(bad)}/{len(inert)} digests identical"
              + (f"; DIFFER: {bad}" if bad else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"backend": backend, "machine": machine, "tps": tps, "arms": arms,
             "substeps": substeps, "ticks": ticks, "warmup": args.warmup,
             "repeat": args.repeat, "inert": inert, "reports": reports,
             "chain_micro": micro},
            indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 1 if any(not x[2] for x in inert) else 0


if __name__ == "__main__":
    sys.exit(main())
