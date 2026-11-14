"""tools/tick_stage_profile.py -- where does a HEADLESS tick's time go, stage by stage? (#29)

The timer and report half of ``tools/bench_bot_matches.py --profile`` and
``tools/bench_resident_scaling.py``: NOT a runner. The window's frame
profiler is ``tools/frame_profile.py`` (``e2e_drive --profile``); this one is
its headless sibling, per TICK and per STAGE instead of per frame, and it
never imports ``renderer/``.

    C:/Users/steen/anaconda3/envs/data/python.exe tools/bench_bot_matches.py \\
        --level ship_s1_medium --levels-dir <scratch> --seeds 1 --timing \\
        --sim-seconds 60 --build C:/Users/steen/projects/breach/cpp/build_cuda \\
        --cuda --resident --profile --profile-sync

What it does: wraps wall-clock timers (``time.perf_counter``) around the
callables each numbered slot of ``Simulation.step`` invokes and around the
stages of slot 7 (``PhysicsRunner.step`` / ``_step_resident``), at CLASS or
MODULE level -- the sim's code is untouched, no sim state is read or written.
Per measured tick it keeps the seconds of every label (0 when a stage did not
run that tick), so the report gives a per-tick MEAN and P95 for each.

- **Slots** (``SLOT_WRAPS``): one label per slot; a slot that is inline code
  in ``step`` (the event clear, the slot-9 burn-through loop, the end-of-tick
  plane wipes) lands in ``(rest of step)`` = step - the sum of the slots.
- **Slot 7** (``PHYSICS_WRAPS``): the resident stages (``GameMap.from_host``
  / ``to_host``, ``bp.water_substeps_resident`` + ``engine.step_water_tail``,
  ``engine.run_substeps_resident``, ``engine.run_trace_tail_resident``) and
  the stages both paths share (``_run_combustion``, ``_run_sky_exchange``,
  ``engine.step_tail``), plus the cpu / per-call path's ``_step_water`` and
  ``engine.run_substeps``; ``(rest of 7)`` = slot 7 - its stages (the host
  pre-physics: ambient args, the water pre-step, lazy binds, device pointers).
- **The bus** (``from_host`` / ``to_host``): calls and bytes per tick by field
  name (``arr.nbytes`` of the host mirror copied). Copies made INSIDE the C++
  per-call twins are invisible here (see ``--nvtx`` and Nsight Systems).
- **sync** (``sync=True``): ``cupy.cuda.Device().synchronize()`` after every
  stage marked ``device`` -- a resident launch core returns before the device
  finishes, so without it the device time lands on whichever later stage
  blocks first (``to_host``). With it each stage carries the device work it
  launched, and the attributed sum can exceed the plain (unsynced) tick.
- **nvtx** (``nvtx=True``): an NVTX range per stage and per tick, and
  ``cudaProfilerStart`` / ``Stop`` around ticks ``skip+1 .. skip+nvtx_ticks``,
  for ``nsys profile --capture-range=cudaProfilerApi``.
"""
from __future__ import annotations

import importlib
import time
from collections import defaultdict

STEP_LABEL = "step"
REST_STEP = "(rest of step)"
SLOT7 = "7 physics"
# #29 P5 (design §9 P5): THE APPLIER ROW, RESERVED -- the per-env serial
# applier (§5 item 6) is P6's; the row is printed (empty) so every report from
# P5 on carries it and P6 fills it without a format change.
APPLIER_LABEL = "  7 applier (per-env serial, P6 -- reserved)"
REST_7 = "  (rest of 7: host pre-physics)"

# (module, qualified name, label): the callables each conductor slot invokes,
# in slot order. Several callables may share one label (summed).
SLOT_WRAPS = (
    ("simulation.gamemap", "GameMap.reset_trace_books", "1 events + round head"),
    ("simulation.ruleset", "ContinuousRealtime.on_round_start", "1 events + round head"),
    ("simulation.ruleset", "TwoPhaseWEGO.on_round_start", "1 events + round head"),
    # #29 P6.1 (design 4.4b decision 5): every slot has a label -- a twin row
    # names its slot by one (1a, 1b', 1c, 4a, 9, 9e(d'), the tick-end clears,
    # 10', 11 had none; their time used to land in "(rest of step)").
    ("simulation.unit_impulse", "integrate_all", "1a integrate + turn"),
    ("simulation.simulation", "Simulation._consume_direct_intents", "1b intents"),
    ("simulation.unit_orders", "apply_requests", "1b' orders"),
    ("simulation.exposure", "decay_units", "1c exposure decay"),
    ("simulation.simulation", "Simulation._update_projectiles", "2 projectiles"),
    ("simulation.simulation", "tick_statuses", "2b statuses"),
    ("simulation.simulation", "Simulation._update_player_movement", "3 movement"),
    ("simulation.simulation", "process_shooting", "4 shooting"),
    ("simulation.simulation", "process_melee_intents", "4a melee"),
    ("simulation.simulation", "process_sprays", "4b sprays"),
    ("simulation.simultaneous", "SimultaneousEffects.resolve", "4c resolve"),
    ("simulation.simulation", "Simulation._run_zombie_slot", "5 zombie AI"),
    ("simulation.gamemap", "GameMap.stamp_units", "6 stamp_units"),
    ("simulation.field_edit", "EditQueue.flush", "6b FieldEdit flush"),
    # #29 P4c: on the resident path 6b packs the queue for the device edit
    # queue (or flushes on the host: the fallback, inside this call).
    ("simulation.physics_runner", "PhysicsRunner.serve_edit_queue", "6b FieldEdit flush"),
    ("simulation.physics_runner", "PhysicsRunner.step", SLOT7),
    ("simulation.simulation", "Simulation._burn_through_walls", "9 burn-through walls"),
    ("simulation.gamemap", "GameMap.find_burst_walls", "9b over-pressure walls"),
    ("simulation.simulation", "apply_environmental_damage", "9c unit heat damage"),
    ("simulation.simulation", "apply_wind_force", "9c2 wind on units"),
    ("simulation.simulation", "apply_teargas_blind", "9c3 gas rows"),
    ("simulation.simulation", "apply_poison_dose", "9c3 gas rows"),
    # #29 PX: the poison law, right after the gas row (slot 9c3').
    ("simulation.poison", "tick", "9c3 gas rows"),
    ("simulation.simulation", "apply_temperature_ignition", "9d ignition"),
    ("simulation.simulation", "Simulation._signal_emit", "9e entities"),
    ("simulation.simulation", "sample_sensors", "9e entities"),
    ("simulation.simulation", "sweep_logic_nodes", "9e entities"),
    ("simulation.simulation", "Simulation._resolve_door_inputs", "9e entities"),
    ("simulation.simulation", "sweep_pumps", "9e entities"),
    ("simulation.simulation", "sweep_vents", "9e entities"),
    ("simulation.simulation", "sweep_doors", "9e entities"),
    ("simulation.simulation", "sweep_timed_charges", "9e entities"),
    ("simulation.charges", "tick", "9e(d') planted charges"),
    ("simulation.simulation", "sweep_heat_sources", "9e entities"),
    ("simulation.simulation", "Simulation._clear_tick_buffers", "tick-end clears"),
    ("simulation.unit_orders", "expire", "10' order expiry"),
    ("simulation.ruleset", "ContinuousRealtime.on_tick_end", "10 tick end (ruleset)"),
    ("simulation.ruleset", "TwoPhaseWEGO.on_tick_end", "10 tick end (ruleset)"),
    ("simulation.team_picture", "TeamPicture.update", "11 team picture"),
)

# (module or "bp", qualified name, label, device): the stages of slot 7.
# "bp" is the imported breach_physics (the build the run uses).
PHYSICS_WRAPS = (
    ("simulation.gamemap", "GameMap.from_host", "  7 H2D GameMap.from_host", True),
    ("bp", "PhysicsEngine.step_water_tail", "  7 water tail (host)", True),
    ("simulation.physics_runner", "PhysicsRunner._step_water", "  7 water (cpu/per-call)", True),
    ("bp", "PhysicsEngine.run_substeps_resident", "  7 EOS run_substeps_resident", True),
    ("bp", "PhysicsEngine.run_substeps", "  7 EOS run_substeps (cpu/per-call)", True),
    ("bp", "PhysicsEngine.run_trace_tail_resident", "  7 trace tail (resident)", True),
    ("simulation.gamemap", "GameMap.to_host", "  7 D2H GameMap.to_host", True),
    ("simulation.physics_runner", "PhysicsRunner._run_combustion", "  7 combustion (host)", True),
    ("simulation.physics_runner", "PhysicsRunner._run_sky_exchange", "  7 sky exchange (host)", False),
    ("bp", "PhysicsEngine.step_tail", "  7 step_tail (ripple+fire+temp+sweep)", True),
    # #29 P4a: the resident tick's device stages (the brackets gone): the
    # entry-phase light upload, the water tail, combustion, the tail (fire +
    # sweep + temperature, ending with the books D2H), the sky bracket's host
    # pass on ambient maps, and the host ripple after the exit download.
    ("bp", "PhysicsEngine.upload_light_resident", "  7 light inputs (resident)", True),
    # #29 P4b: slot 6's stamp on the device (the five planes from the host's rows).
    ("bp", "PhysicsEngine.stamp_units_resident", "  7 stamp (resident)", True),
    # #29 P4c: slot 6b's flush applied on the device (the edit queue).
    ("bp", "PhysicsEngine.apply_edits_resident", "  7 edits (resident)", True),
    # #29 P5: the water stage is one resident driver (substeps + W5/W3).
    ("bp", "PhysicsEngine.step_water_resident", "  7 water (resident: substeps + W5/W3)", True),
    ("bp", "PhysicsEngine.run_combustion_resident", "  7 combustion (resident)", True),
    ("bp", "PhysicsEngine.step_tail_resident", "  7 tail (resident: fire+sweep+temp+books)", True),
    ("bp", "PhysicsEngine.step_ripple", "  7 ripple (host)", False),
)

BUS_LABELS = {"GameMap.from_host": "H2D", "GameMap.to_host": "D2H"}


def _resolve(owner, qualname):
    """``(container, attr)`` for ``qualname`` under ``owner`` (a module)."""
    parts = qualname.split(".")
    obj = owner
    for p in parts[:-1]:
        obj = getattr(obj, p)
    return obj, parts[-1]


def _stats(xs):
    import numpy as np
    a = np.asarray(xs, dtype=np.float64) * 1e3
    if a.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


class TickStageProfiler:
    """Per-tick, per-stage wall-clock timers patched in at class / module level.

    ``install(bp)`` patches; ``uninstall()`` restores; ``reset()`` empties the
    records (between matches); ``summary()`` / ``format_report()`` read them.
    Ticks ``1 .. skip`` (warm-up: residency allocation, first-touch scratch)
    are not recorded."""

    def __init__(self, skip=2, sync=False, nvtx=False, nvtx_ticks=100):
        self.skip = int(skip)
        self.sync = bool(sync)
        self.nvtx = bool(nvtx)
        self.nvtx_ticks = int(nvtx_ticks)
        self._orig = []           # (container, attr, original)
        self._captured = False    # the nvtx window opens once per process
        self._sync_fn = None
        self._nvtx_mod = None
        self._runtime = None
        self.reset()

    # -- records -----------------------------------------------------------
    def reset(self):
        self.tick = 0             # Simulation.step calls since reset
        self.records = []         # per measured tick: {label: seconds}
        self.calls = []           # per measured tick: {label: calls}
        self.bus = []             # per measured tick: {(dir, field): [calls, bytes]}
        # #29 P4a: per measured tick, the slot-7 stage calls IN CALL ORDER, a
        # bus call as (label, dir, n_fields, bytes), a stage as (label, None,
        # 0, 0) -- the bus gate reads where the transfers sit in the chain.
        self.seq = []
        self._cur = defaultdict(float)
        self._cur_calls = defaultdict(int)
        self._cur_bus = defaultdict(lambda: [0, 0])
        self._cur_seq = []
        self._capturing = False

    # -- install -----------------------------------------------------------
    def set_sync(self, on: bool):
        self.sync = bool(on)

    def install(self, bp, device=False):
        """Patch the timers in. ``device``: the run uses the GPU (``sync``
        and ``nvtx`` need CuPy then; on the CPU path CuPy is never imported)."""
        if device and (self.sync or self.nvtx):
            import cupy
            dev = cupy.cuda.Device()
            self._sync_fn = dev.synchronize
            self._runtime = cupy.cuda.runtime
            if self.nvtx:
                import cupy.cuda.nvtx as nvtx
                self._nvtx_mod = nvtx
        from simulation import Simulation
        self._patch(Simulation, "step", STEP_LABEL, device=device, tick=True)
        for mod, qual, label in SLOT_WRAPS:
            try:
                cont, attr = _resolve(importlib.import_module(mod), qual)
                getattr(cont, attr)
            except AttributeError:
                continue          # a slot callable this tree does not have
            # A class is patched only where it DEFINES the method, so an
            # inherited one is never timed twice.
            if not isinstance(cont, type) or attr in cont.__dict__:
                self._patch(cont, attr, label, device=(label == SLOT7 and device))
        for mod, qual, label, dev in PHYSICS_WRAPS:
            owner = bp if mod == "bp" else importlib.import_module(mod)
            try:
                cont, attr = _resolve(owner, qual)
                getattr(cont, attr)
            except AttributeError:
                continue          # a stage this build does not have
            self._patch(cont, attr, label, device=(dev and device),
                        bus=BUS_LABELS.get(qual), phys=True)

    def uninstall(self):
        for cont, attr, orig in reversed(self._orig):
            setattr(cont, attr, orig)
        self._orig.clear()

    def _patch(self, cont, attr, label, device=False, tick=False, bus=None,
               phys=False):
        orig = getattr(cont, attr)
        prof = self

        def timed(*a, **k):
            if tick:
                prof._tick_begin()
            nv = prof._nvtx_mod if prof._capturing else None
            if nv is not None:
                nv.RangePush(label)
            if bus is not None:
                gm = a[0]
                names = a[1] if len(a) > 1 else k.get("names")
                if names is None:
                    names = (gm._RESIDENT_FIELD_NAMES if bus == "H2D"
                             else gm._RESIDENT_SYNCED)
                nbytes = 0
                for n in names:
                    rec = prof._cur_bus[(bus, n)]
                    rec[0] += 1
                    rec[1] += int(getattr(gm, n).nbytes)
                    nbytes += int(getattr(gm, n).nbytes)
                # #29 P4b: the field NAMES ride along (a fifth element), so a
                # gate can say which planes a tick's upload carried.
                prof._cur_seq.append((label, bus, len(names), nbytes,
                                      tuple(names)))
            elif phys:
                prof._cur_seq.append((label, None, 0, 0, ()))
            t0 = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                if device and prof.sync and prof._sync_fn is not None:
                    prof._sync_fn()
                dt = time.perf_counter() - t0
                prof._cur[label] += dt
                prof._cur_calls[label] += 1
                if nv is not None:
                    nv.RangePop()
                if tick:
                    prof._tick_end()
        timed.__name__ = getattr(orig, "__name__", attr)
        timed.__wrapped__ = orig
        setattr(cont, attr, timed)
        self._orig.append((cont, attr, orig))

    # -- tick boundaries ---------------------------------------------------
    def _tick_begin(self):
        self.tick += 1
        self._cur = defaultdict(float)
        self._cur_calls = defaultdict(int)
        self._cur_bus = defaultdict(lambda: [0, 0])
        self._cur_seq = []
        if self.nvtx and self._runtime is not None:
            if self.tick == self.skip + 1 and not self._captured:
                self._runtime.profilerStart()
                self._capturing = True
                self._captured = True
            if self._capturing:
                self._nvtx_mod.RangePush(f"tick {self.tick}")

    def _tick_end(self):
        if self._capturing:
            self._nvtx_mod.RangePop()
            if self.tick >= self.skip + self.nvtx_ticks:
                self._runtime.profilerStop()
                self._capturing = False
        if self.tick > self.skip:
            self.records.append(dict(self._cur))
            self.calls.append(dict(self._cur_calls))
            self.bus.append({k: list(v) for k, v in self._cur_bus.items()})
            self.seq.append(list(self._cur_seq))

    # -- the report ----------------------------------------------------------
    def summary(self) -> dict:
        """Per-label mean / p95 / max ms per tick and calls per tick, the two
        derived rests, and the bus per field (calls and bytes per tick)."""
        n = len(self.records)
        labels = []
        for mod, qual, label in SLOT_WRAPS:
            if label not in labels:
                labels.append(label)
        phys = []
        for _m, _q, label, _d in PHYSICS_WRAPS:
            if label not in phys:
                phys.append(label)
        out = {"ticks": n, "skip": self.skip, "sync": self.sync, "stages": {}}

        def series(label):
            return [r.get(label, 0.0) for r in self.records]

        def put(label, xs, calls=None):
            st = _stats(xs)
            if calls is not None:
                st["calls"] = (sum(c.get(calls, 0) for c in self.calls)
                               / max(1, n))
            out["stages"][label] = st

        put(STEP_LABEL, series(STEP_LABEL), STEP_LABEL)
        slot_sum = [0.0] * n
        for label in labels:
            xs = series(label)
            if label == SLOT7:
                put(label, xs, label)
                p_sum = [0.0] * n
                for pl in phys:
                    pxs = series(pl)
                    put(pl, pxs, pl)
                    p_sum = [a + b for a, b in zip(p_sum, pxs)]
                put(APPLIER_LABEL, [0.0] * n)
                put(REST_7, [a - b for a, b in zip(xs, p_sum)])
            else:
                put(label, xs, label)
            slot_sum = [a + b for a, b in zip(slot_sum, xs)]
        put(REST_STEP, [a - b for a, b in zip(series(STEP_LABEL), slot_sum)])
        bus = defaultdict(lambda: [0, 0])
        for t in self.bus:
            for k, (c, b) in t.items():
                bus[k][0] += c
                bus[k][1] += b
        out["bus"] = {f"{d} {f}": {"calls": c / max(1, n), "bytes": b / max(1, n)}
                      for (d, f), (c, b) in sorted(bus.items())}
        return out

    def format_report(self, total_ms=None) -> str:
        """The stage table (ms per tick) and the bus table. ``total_ms``: the
        plain tick time measured elsewhere, printed beside the step for scale."""
        s = self.summary()
        lines = [f"stage profile: {s['ticks']} ticks (skipped {s['skip']}), "
                 f"sync={'on' if s['sync'] else 'off'}"
                 + (f", plain total {total_ms:.2f} ms" if total_ms else ""),
                 f"{'stage':44s} {'mean ms':>8s} {'p95 ms':>8s} {'calls/tick':>10s}"]
        for label, st in s["stages"].items():
            if st["mean"] == 0.0 and st["p95"] == 0.0 and label not in (
                    STEP_LABEL, SLOT7, APPLIER_LABEL):
                continue
            c = st.get("calls")
            lines.append(f"{label:44s} {st['mean']:8.3f} {st['p95']:8.3f} "
                         + (f"{c:10.2f}" if c is not None else f"{'':10s}"))
        if s["bus"]:
            lines.append(f"{'bus field':44s} {'calls/tick':>10s} {'KiB/tick':>10s}")
            tot = defaultdict(float)
            for key, v in s["bus"].items():
                tot[key.split()[0]] += v["bytes"]
                lines.append(f"{key:44s} {v['calls']:10.2f} {v['bytes'] / 1024:10.1f}")
            for d, b in sorted(tot.items()):
                lines.append(f"{'total ' + d:44s} {'':10s} {b / 1024:10.1f}")
        return "\n".join(lines)
