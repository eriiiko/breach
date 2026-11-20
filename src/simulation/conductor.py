"""THE DEVICE CONDUCTOR (#29 P7e.4, residency design §4.4c "P7e RE-CUT"): ONE
function that runs a whole reference tick's unit side ON THE DEVICE at N = 1,
in ``Simulation.step()``'s slot order, with every piece of state resident
ACROSS ticks -- the unit store and its tables, the record lists, the object
and entity tables, the edit door, the horde cache and the tick row -- chained
with the resident physics tick (slot 7) on the same device buffers
(``ResidentBatch.buffers``).

CLASS T (training only): ``Simulation.step()`` and the played resident tick
are untouched; the conductor is what P7e.5 batches and P8 trains on. It never
drives a played game.

THE SHAPE (a choice where the design was silent: the walk is a Python
function over the ``unit_kernels`` faces, every face one or a few launches on
the tick stream, rather than a C++ orchestration; its cost is the Python per
face, measured in the P7e.4 as-built note)::

    c = DeviceConductor(sim)          # ONE upload of everything, at build
    c.latch(intents)                  # THE LATCHES: the tick's intents
    c.tick()                          # the walk: slots 1 .. 11
    c.download()                      # THE GATE'S DOWNLOAD (only when asked)

``sim`` is the TWIN: a fully built ``Simulation`` whose state seeds the device
once. From then on the device is the truth; the twin's host objects are
stale until :meth:`download` lands the device state on them (the gate's
digest reads them). The twin's host scalars (``tick``, ``real_time``) are the
conductor's own clock, advanced at slot 10 beside the device tick row.

THE LATCHES (the host's only writes per tick): :meth:`latch` takes a tick's
intents as ``(kind, args)`` in issue order -- exactly what the intent tap
records (``Simulation.attach_intent_log``, ``match_replay.IntentLog``), args
normalised as the sim's own methods normalise them -- and turns them into
(1) a COLUMN SCATTER into the store's latch columns (:data:`LATCH_COLUMNS`:
each intent method writes exactly one ``Unit`` attribute, a function of its
arguments alone; the last write of a (unit, column) in the tick wins, as on
the host), never a whole-row pack; (2) the ORDER TABLE (``order_intent``,
written onto the rows at slot 1b' by ``unit_kernels.apply_orders``); (3) the
ZOMBIE REQUEST SET (``drive_zombies``, the HORDE latch). The tick row
advances ON THE DEVICE at slot 10.

THE REFUSALS AT BUILD (by name, :class:`ConductorRefused`): a sim without a
physics runner, a ruleset that is not the continuous path (the shooting and
object doors), a map with water (the water pre-step reads the host mirror),
an ambient map (the sky bracket is a host bracket), requested light, a
non-empty edit queue window (the device door starts at the host's seq with
no rows), a unit carrying an optics override (the stamp's table refusal).

NOT PART OF THE PLAYED TICK.
"""
from __future__ import annotations

import numpy as np


class ConductorRefused(RuntimeError):
    """A sim the device conductor does not serve (named at its build)."""


def _move(a):
    from simulation import intents
    return intents.MoveDirIntent(int(a[1]), int(a[2]), int(a[3]))


def _aim(a):
    from simulation import intents
    return intents.AimIntent(int(a[1]), int(a[2]))


def _throw(a):
    from simulation import intents
    return intents.ThrowIntent(int(a[1]), int(a[2]), int(a[3]),
                               None if a[4] is None else int(a[4]), int(a[5]))


#: THE ACTIONS LATCH: intent kind -> (the ONE ``Unit`` column it writes, its
#: value from the tap's normalised args). Each row is the method body's one
#: write (``Simulation.set_move_dir`` .. ``set_flashlight``); the methods read
#: no unit state but the unit's existence (``get_unit``), and ``_possessed_ids``
#: arms no device slot (``cpp/src/cuda_unit_verbs.cu``: every living marine
#: consumes its own columns).
LATCH_COLUMNS = {
    "set_move_dir": ("live_move_dir", _move),
    "clear_move_dir": ("live_move_dir", lambda a: None),
    "set_aim": ("live_aim", _aim),
    "set_trigger": ("live_trigger", lambda a: bool(a[1])),
    "throw_grenade_intent": ("pending_throw", _throw),
    "use_intent": ("pending_use", lambda a: True),
    "reload_intent": ("pending_reload", lambda a: True),
    "shove_intent": ("pending_shove", lambda a: True),
    "melee_intent": ("pending_melee", lambda a: True),
    "set_flashlight": ("flashlight_on", lambda a: not bool(a[1])),
    "plant_charge_intent": ("pending_plant", lambda a: int(a[1])),
}
#: The latches that are not a unit column.
OTHER_LATCHES = ("order_intent", "drive_zombies")


class DeviceConductor:
    """The device tick of ONE sim (N = 1); see the module docstring."""

    def __init__(self, sim, bp=None):
        import cupy as cp
        if bp is None:
            import breach_physics as bp
        if not getattr(bp, "HAS_CUDA", False):
            raise ConductorRefused("device conductor: the CUDA build is required")
        from simulation import (applier, attack_lists, edit_door, horde_cache,
                                object_tables as ot, unit_draws, unit_events,
                                unit_kernels as uk, unit_requests, unit_schema as us,
                                unit_store, unit_tables)
        from simulation.resident_batch import ResidentBatch
        self.cp, self.bp, self.sim = cp, bp, sim
        self._refuse(sim)
        uk.shooting_door(sim)
        uk.refuse_wego([sim])
        g = sim.gmap
        self.h, self.w = int(g._h), int(g._w)
        self.n_gases = int(g.gas.shape[0])
        self.o2 = int(g.gases.name_to_id["o2"])
        self.n2 = int(g.gases.name_to_id["inert_n2"])
        self.sim_time = 1.0 / float(sim._tps)
        # -- the planes: the twin's map made resident (ONE upload, here) -----
        self.batch = ResidentBatch([sim])
        f = dict(self.batch.buffers)
        for name in applier.non_resident_planes():
            # the render-only ripple pair (host-computed by design, W6a): the
            # seal's 0.0 write needs a device plane; never read back
            f[name] = cp.asarray(np.ascontiguousarray(getattr(g, name)))[None].copy()
        self.fields = f
        self.flight = cp.zeros((1, self.h, self.w), dtype=cp.bool_)
        self.src_out = cp.zeros((1, self.h, self.w), dtype=cp.int32)
        # -- the units ----------------------------------------------------------
        self.U = max(1, len(sim.units))
        self.uids = [int(u.id) for u in sim.units]
        self.row_of = {uid: r for r, uid in enumerate(self.uids)}
        self.store = unit_store.DeviceStore(1, self.U, bp=bp,
                                            tables=unit_tables.values_batch([sim]))
        self.store.upload(unit_store.pack(sim.units, self.U)[None])
        lay = unit_store.layout()
        self._lay = lay
        mem = cp.cuda.UnownedMemory(self.store.device_ptr, 8 * int(np.prod(self.store.shape)),
                                    self.store)
        self._rows = cp.ndarray(self.store.shape, dtype=cp.int64,
                                memptr=cp.cuda.MemoryPointer(mem, 0))
        self._zombie = [bool(u.is_zombie) for u in sim.units]
        # -- the tick row (the latch written ONCE, here; advanced on the device)
        self.trow = unit_draws.TickRow.of_sims([sim])
        # -- the record lists ---------------------------------------------------
        U = self.U
        self.ring = unit_events.EventRing(1, U, bp=bp)
        self.req = unit_requests.RequestList(1, U, bp=bp)
        self.eff = applier.DeviceApplier(1, applier.E_CAP, bp=bp, n_units=U)
        self.det = applier.DeviceApplier(1, applier.E_CAP, bp=bp, n_units=U)
        self.aux = applier.DeviceApplier(1, applier.E_CAP, bp=bp, n_units=U)
        self.cells = attack_lists.cell_buffer(1, bp=bp)
        self.claims = attack_lists.claims_list(1, bp=bp)
        self.door = edit_door.DeviceEditDoor(1, bp=bp)
        self.flush = edit_door.DeviceEditFlush(self.door, bp=bp)
        # -- the object and entity tables, staged ONCE from the twin ------------
        DT = ot.DeviceTable
        self.rounds = DT(us.ROUND_TABLE, 1, n_units=U, row_cap=uk.round_row_cap(), bp=bp)
        self.noise = DT(us.NOISE_TABLE, 1, n_units=U, row_cap=1, bp=bp)
        self.blast_log = DT(us.BLAST_LOG_TABLE, 1, n_units=U, row_cap=1, bp=bp)
        self.grenades = DT(us.GRENADE_TABLE, 1, bp=bp)
        self.charges = DT(us.CHARGE_TABLE, 1, bp=bp)
        self.doors, self.door_tiles = uk.door_tables(1, bp=bp)
        self.ducts, self.vents = uk.vent_tables(1, bp=bp)
        self.sources, self.source_tiles = uk.source_tables(1, bp=bp)
        self.timed, self.heat = uk.hazard_tables(1, bp=bp)
        self.sig = uk.signal_tables(1, bp=bp)
        self.sig["doors"] = self.doors
        w_ = lambda t, objs: np.array(ot.records(t, objs), dtype=np.int64).reshape(  # noqa: E731
            -1, ot.table_layout(t).width)
        self.rounds.stage_lists([uk.rounds_of(sim)])
        self.noise.stage_lists([uk.noise_of(sim)])
        self.blast_log.stage_lists([uk.blast_log_of(sim)])
        self.grenades.stage_lists([w_(us.GRENADE_TABLE, sim.projectiles)])
        self.charges.stage_lists([w_(us.CHARGE_TABLE, sim.planted_charges)])
        self.doors.stage_lists([uk.doors_of(sim)])
        self.door_tiles.stage_lists([uk.door_tiles_of(sim)])
        self.ducts.stage_lists([uk.ducts_of(sim)])
        self.vents.stage_lists([uk.vents_of(sim)])
        self.sources.stage_lists([uk.sources_of(sim)])
        self.source_tiles.stage_lists([uk.source_tiles_of(sim)])
        self.timed.stage_lists([uk.timed_charges_of(sim)])
        self.heat.stage_lists([uk.heat_sources_of(sim)])
        for ln in uk.SIGNAL_LISTS:
            self.sig[ln].stage_lists([uk.SIGNAL_RECORDS[ln](sim)])
        self.cache = horde_cache.DeviceHordeCache(1, U, persistent=True)
        self.cache.stage_lists([horde_cache.cache_records(sim)])
        self.cache.begin()
        self.door.begin()
        self.door.set_seq([int(sim.edit_queue._seq)])
        self.claims.begin()
        self.cells.begin()
        self.walk = uk.WalkLists(cells=self.cells, door=self.door, claims=self.claims,
                                 noise=self.noise, blast_log=self.blast_log,
                                 sources=self.sources, source_tiles=self.source_tiles)
        self.box_max = uk.spray_box_max(self.store, self.h, self.w)
        self.stamp_scratch = None
        # -- the latches' host side ---------------------------------------------
        self._orders: dict = {}
        self._zreqs = None if sim._zombie_requests is None else set(sim._zombie_requests)
        # -- the 9e gates: the level's entity classes (load constants) ---------
        self.has_bus = sim._signal_bus is not None
        self.has_ducts = bool(sim._ducts)
        self.has_doors = bool(sim._doors)
        self.has_timed = bool(sim._timed_charges)
        self.has_heat = bool(sim._heat_sources)
        self.has_pumps = bool(sim._pumps)
        self.ticks = 0
        self.last_books: dict = {}

    # ------------------------------------------------------------------
    # The build's refusals
    # ------------------------------------------------------------------
    @staticmethod
    def _refuse(sim):
        from simulation.ruleset import ContinuousRealtime
        r = sim.physics_runner
        if r is None:
            raise ConductorRefused("device conductor: the sim has no physics runner")
        if not isinstance(sim.ruleset, ContinuousRealtime) or sim.ruleset.drives_units:
            raise ConductorRefused(f"device conductor: the ruleset "
                                   f"{type(sim.ruleset).__name__} is not the continuous path")
        g = sim.gmap
        if g.water_sources or g.water_depth.any():
            raise ConductorRefused("device conductor: the map holds water (the water "
                                   "pre-step and its dormancy read the host mirror)")
        if r._ambient_mask(g) is not None:
            raise ConductorRefused("device conductor: an ambient map (the sky exchange "
                                   "is a host bracket, ACCEPTED GAP of the resident tick)")
        if r.engine.light_requested:
            raise ConductorRefused("device conductor: light is requested (render state, "
                                   "until #81)")
        if sim.edit_queue._edits:
            raise ConductorRefused("device conductor: the edit queue holds "
                                   f"{len(sim.edit_queue._edits)} edit(s): the device "
                                   "door starts empty")
        if sim.paused:
            raise ConductorRefused("device conductor: the sim is paused")

    # ------------------------------------------------------------------
    # THE LATCHES
    # ------------------------------------------------------------------
    def latch(self, intents) -> None:
        """This tick's intents (``(kind, args)`` in issue order, args as the
        tap records them) into the device latches: the column scatter, the
        order table, the zombie request set. Refuses an unknown kind by name.
        A unit id the sim does not hold is skipped (``get_unit`` -> None)."""
        from simulation import unit_orders, unit_schema as us
        cols: dict = {}
        for kind, args in intents:
            args = tuple(args)
            if kind in LATCH_COLUMNS:
                col, val = LATCH_COLUMNS[kind]
                r = self.row_of.get(int(args[0]))
                if r is None:
                    continue
                cols[(r, col)] = val(args)
            elif kind == "order_intent":
                uid, what, where, ticks, target = (int(x) for x in args)
                until = unit_orders.NO_LIMIT if ticks == 0 else self.sim.tick + ticks
                self._orders[uid] = (what, where, until, target)
            elif kind == "drive_zombies":
                if self._zreqs is None:
                    raise ConductorRefused("device conductor: drive_zombies on a sim whose "
                                           "horde the conductor drives")
                self._zreqs.update(int(i) for i in args[0])
            else:
                raise ConductorRefused(f"device conductor: intent {kind!r} has no latch")
        if not cols:
            return
        idx, vals = [], []
        W = self.store.shape[2]
        for (r, name), v in cols.items():
            c = self._lay.column(name)
            enc = us.COLUMN_BY_NAME[name].encode(v)
            enc = tuple(enc) if isinstance(enc, tuple) else (enc,)
            if len(enc) != c.width:
                raise ConductorRefused(f"device conductor: latch {name!r} encodes "
                                       f"{len(enc)} ints, its column is {c.width} wide")
            for j, x in enumerate(enc):
                idx.append(r * W + c.offset + j)
                vals.append(int(x))
        cp = self.cp
        flat = self._rows.reshape(-1)
        flat[cp.asarray(np.asarray(idx, dtype=np.int64))] = cp.asarray(
            np.asarray(vals, dtype=np.int64))

    def _order_table(self):
        from simulation import unit_kernels as uk
        rows = [(uid,) + tuple(self._orders[uid]) for uid in sorted(self._orders)]
        self._orders.clear()
        return uk.pad_requests([np.array(rows, dtype=np.int64).reshape(-1, uk.REQUEST_WIDTH)])

    def _horde_latch(self):
        from simulation import applier, horde_cache
        row = np.zeros(len(applier.HORDE_LATCH) + self.U, dtype=np.int64)
        m = applier.HORDE_LATCH.index("mode")
        modes = horde_cache._MODE
        if self._zreqs is None:
            row[m] = modes["CONDUCTOR"]
        elif not self._zreqs:
            row[m] = modes["EMPTY"]
        else:
            row[m] = modes["SET"]
            base = len(applier.HORDE_LATCH)
            for r, uid in enumerate(self.uids):
                if uid in self._zreqs:
                    row[base + r] = 1
            self._zreqs.clear()
        return [row]

    # ------------------------------------------------------------------
    # THE WALK
    # ------------------------------------------------------------------
    def _planes(self, names):
        return {n: self.fields[n] for n in names}

    def tick(self) -> None:
        """ONE reference tick on the device (``Simulation.step()``'s slots in
        order, one call line each)."""
        from simulation import physics_runner as pr, unit_kernels as uk
        from simulation import edit_door as _ed  # noqa: F401
        sim, st, f, tr = self.sim, self.store, self.fields, self.trow
        lists = self.walk
        # 1. the event ring begun; the trace books are host books (reset)
        self.ring.begin()
        sim.gmap.reset_trace_books()
        # 1a. integrate + turn
        uk.integrate_turn(st)
        # 1b. the hand verbs; their table writes (throws, plants, uses)
        self.req.begin()
        uk.flight_plane(st, self._planes(uk.FLIGHT_SOURCES), out=self.flight)
        uk.verbs_device(st, self.req, tr, self.doors, self.door_tiles, self.charges,
                        {"flight": self.flight})
        uk.table_appends(st, self.req, self.grenades, self.charges)
        uk.door_uses(st, self.req, self.doors)
        # 1b'. the commander's orders
        uk.apply_orders(st, self._order_table())
        # 1c. exposure decays
        uk.exposure_decay(st)
        # 2. the grenades fly (the flight plane as the slot found it), the
        # fuse-outs detonate at once
        uk.flight_plane(st, self._planes(uk.FLIGHT_SOURCES), out=self.flight)
        self.det.begin()
        uk.grenades(st, self.grenades, self.det, tr, {"flight": self.flight})
        uk.detonate(st, self.det, f, self.ring, lists, tick_row=tr)
        # 2'. the rounds in flight march, then their landing
        self.eff.begin()
        uk.rounds_march(st, self.eff, self.rounds, tr, self._planes(uk.SHOOT_FIELDS))
        uk.resolve(st, self.eff, f, self.ring, lists, tick_row=tr)
        # 2b. statuses
        uk.status_tick(st, self.ring)
        # 3. movement
        uk.movement(st, self._planes(uk.MOVEMENT_FIELDS))
        # 4. shooting, 4a. melee, 4b. sprays (phase A, one effect list)
        self.eff.begin()
        self.cells.begin()
        uk.shooting(st, self.eff, self.rounds, self.noise, tr, self._planes(uk.SHOOT_FIELDS))
        uk.melee(st, self.eff, tr)
        uk.sprays(st, self.eff, self.cells, {"solid": f["solid"]}, box_max=self.box_max)
        # 4c. phase B
        uk.resolve(st, self.eff, f, self.ring, lists, tick_row=tr)
        # 5. the horde (drains the noise list)
        uk.horde(st, self.aux, f, self.ring, self.noise, self.doors, self.door_tiles,
                 self.cache, self._horde_latch(), tr)
        # 6. the stamp from the store
        _r, self.stamp_scratch = uk.stamp(st, self._planes(n for n, _d in uk.STAMP_FIELDS),
                                          self.stamp_scratch)
        # 6b. the flush from the door
        self.flush.run(tr, st.tables, self._planes(n for n, _d in _ed.FLUSH_PLANES),
                       self.n_gases, self.o2, self.n2)
        self.flush.fold([sim.gmap], [sim.draws])
        # 7. the physics head on the device (the blast source), then slot 7
        self.aux.begin()
        uk.take_sources(st, self.aux, f, lists, source_out=self.src_out)
        pr.run_resident_conductor(sim.physics_runner, sim.gmap, self.sim_time,
                                  int(self.src_out.data.ptr))
        # 9. burn-through (the resident fire's list ranked on the device)
        uk.burn_through_device(st, self.eff, f,
                               uk.BurnList.resident(self.bp, 1, self.h, self.w), tick_row=tr)
        # 9b. over-pressure walls
        uk.burst_walls(st, self.eff, f, tick_row=tr)
        # 9c. heat, 9c2. wind, 9c3. gas rows (+ poison, bite, corpse, stress)
        uk.heat_damage(st, self.ring, self._planes(uk.HEAT_FIELDS), self.claims)
        uk.wind_impulse(st, self._planes(uk.WIND_FIELDS))
        uk.gas_rows(st, self.ring, self._planes(uk.GAS_FIELDS), self.claims)
        # 9d. ignition
        uk.ignition(st, self._planes(n for n, _d in uk.IGNITION_FIELDS))
        # 9e. the entities
        sig = self.sig
        if self.has_bus:
            uk.signals_emit(st, self.aux, f, sig, tr)
            uk.signals_logic(st, self.aux, sig)
            uk.signals_door_inputs(st, self.aux, sig)
            if self.has_pumps:
                uk.signals_pumps(st, self.aux, f, sig)
        if self.has_ducts:
            uk.sweep_vents(st, self.aux, f, self.ducts, self.vents)
        if self.has_doors:
            uk.sweep_doors(st, self.aux, f, self.doors, self.door_tiles)
        if self.has_timed:
            self.aux.begin()
            uk.timed_charges(st, self.aux, f, self.ring, lists, self.timed)
        self.det.begin()
        uk.charges_tick(st, self.charges, self.det, tr)
        uk.detonate(st, self.det, f, self.ring, lists, tick_row=tr)
        if self.has_heat:
            self.aux.begin()
            uk.heat_sources(st, self.aux, f, self.ring, lists, self.heat)
        else:
            self.blast_log.begin()
        if self.has_bus:
            uk.signals_swap(st, self.aux, sig)
        # the tick-end clears: the heat plane, the claims list
        uk.tick_end_clears(self.bp, {"heat": f["heat"]})
        self.claims.begin()
        # 10. the tick advances (the device row and the twin's host clock)
        self._advance()
        # 10'. orders expire; the ruleset's tick end (conversion, the prune)
        uk.expire_orders(st, tr)
        uk.convert_zombies(st)
        uk.prune_grenades(self.grenades)
        # 11. the team picture: host, a match only, in no digest (not run)
        self.ticks += 1

    def _advance(self) -> None:
        from simulation.unit_draws import TICK_ROW_KEY, _M32
        b = self.trow.buf
        b[:, 0] += 1
        k = TICK_ROW_KEY + 2
        b[:, k] = (b[:, k] + 1) & _M32
        self.sim.tick += 1
        self.sim.real_time += self.sim_time

    # ------------------------------------------------------------------
    # THE GATE'S DOWNLOAD
    # ------------------------------------------------------------------
    def download(self) -> None:
        """Land the device state on the twin's host objects (what the digest,
        ``field_ab_harness.capture_snapshot``, reads): the unit rows (and the
        conversion's host-only "Z-" name), the resident planes, the tick's
        synced events, the planted charges and every entity table on the
        sim's own runtimes."""
        from simulation import (object_tables as ot, swarm, unit_events,
                                unit_kernels as uk, unit_schema as us, unit_store)
        from simulation.gamemap import GameMap
        sim = self.sim
        g = sim.gmap
        rows = self.store.download()[0]
        unit_store.unpack(rows, sim.units)
        for k, u in enumerate(sim.units):
            if u.is_zombie and not self._zombie[k] and not u.name.startswith("Z-"):
                u.name = f"Z-{u.name}"
            self._zombie[k] = bool(u.is_zombie)
        sw = set(swarm.SWARM_RESIDENT_NAMES) | set(swarm.SWARM_PLANE_NAMES)
        names = [n for n in GameMap._RESIDENT_FIELD_NAMES if n not in sw]
        g.to_host(names + list(swarm.resident_exit_names(g)))
        recs = self.ring.records()[0]
        sim.tick_events[:] = [unit_events.event_of(r) for r in recs]
        crow, ccnt, _ = self.charges.download()
        sim.planted_charges[:] = ot.unpack(us.CHARGE_TABLE, crow[0], int(ccnt[0]))
        if self.has_doors:
            uk.doors_to_host(sim, self.doors.records()[0], self.door_tiles.records()[0])
        if self.has_ducts:
            uk.ducts_to_host(sim, self.ducts.records()[0])
            uk.vents_to_host(sim, self.vents.records()[0])
        if self.has_bus:
            uk.signals_to_host(sim, self.sig["signals"].records()[0])
            uk.nodes_to_host(sim, self.sig["nodes"].records()[0])
            uk.pumps_to_host(sim, self.sig["pumps"].records()[0])
        if self.has_timed:
            uk.timed_charges_to_host(sim, self.timed.records()[0])
        if self.has_heat:
            uk.heat_sources_to_host(sim, self.heat.records()[0])

    def nondigest_records(self) -> dict:
        """The cross-tick state in NO digest, for the gate: the rounds, the
        grenades, the noise list, the blast sources, the edit door's window
        and the horde cache, each as the reference's record function writes
        it (a download)."""
        from simulation import edit_door as ed
        e, t = self.door.rows()[0]
        return {"rounds": self.rounds.records()[0], "grenades": self.grenades.records()[0],
                "noise": self.noise.records()[0], "sources": self.sources.records()[0],
                "source_tiles": self.source_tiles.records()[0],
                "door": (e[:, :ed.EDIT_COLS].tolist(), t.tolist()),
                "horde": self.cache.records()[0]}


def reference_records(sim) -> dict:
    """The reference's own records of :meth:`DeviceConductor.nondigest_records`'
    state (the record functions over the host lists)."""
    from simulation import edit_door as ed, horde_cache, object_tables as ot
    from simulation import unit_kernels as uk, unit_schema as us
    w = ot.table_layout(us.GRENADE_TABLE).width
    gren = np.array(ot.records(us.GRENADE_TABLE, sim.projectiles), dtype=np.int64).reshape(-1, w)
    e, t = ed.door_rows(sim.gmap, sim.edit_queue._edits)
    tup = lambda a: [tuple(int(x) for x in r) for r in np.asarray(a)]  # noqa: E731
    return {"rounds": tup(uk.rounds_of(sim)), "grenades": tup(gren),
            "noise": tup(uk.noise_of(sim)), "sources": tup(uk.sources_of(sim)),
            "source_tiles": tup(uk.source_tiles_of(sim)),
            "door": (np.asarray(e).reshape(-1, ed.EDIT_COLS).tolist(),
                     np.asarray(t).reshape(-1, ed.TILE_COLS).tolist()),
            "horde": horde_cache.cache_records(sim)}


__all__ = ["ConductorRefused", "LATCH_COLUMNS", "OTHER_LATCHES", "DeviceConductor",
           "reference_records"]
