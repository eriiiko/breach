"""THE DEVICE CONDUCTOR (#29 P7e.4 / P7e.5, residency design §4.4c "P7e
RE-CUT"): ONE function that runs a whole reference tick of N homogeneous envs
ON THE DEVICE in ONE walk, in ``Simulation.step()``'s slot order, every face
launched once for all N envs, with every piece of state resident ACROSS ticks
-- the unit store ``(N, U, W)`` and its tables, the record lists, the object
and entity tables, the edit door, the horde cache and the tick row, per env
-- chained with the batched resident physics tick (slot 7) on the same device
buffers (``ResidentBatch.buffers``). N = 1 is one sim: never a second copy.

CLASS T (training only): ``Simulation.step()`` and the played resident tick
are untouched; the conductor is what P8 trains on. It never drives a played
game.

THE SHAPE (a choice where the design was silent: the walk is a Python
function over the ``unit_kernels`` faces, every face one or a few launches on
the tick stream, rather than a C++ orchestration; its cost is the Python per
face, measured by ``tools/bench_conductor.py``)::

    c = DeviceConductor(sims)         # ONE upload of everything, at build
    c.latch(intents, env=e)           # THE LATCHES: env e's intents this tick
    c.tick()                          # the walk: slots 1 .. 11, every env
    c.download()                      # THE GATE'S DOWNLOAD (only when asked)

``sims`` are the TWINS (one ``Simulation`` or a list; one level, as the
batched resident engine requires: grid, gas table, tilt, tick rate, space
maps), whose state seeds the device once. From then on the device is the
truth; a twin's host objects are stale until :meth:`download` lands the
device state on them (the gate's digest reads them). Each twin's host scalars
(``tick``, ``real_time``) are the conductor's clock, advanced at slot 10
beside the device tick row. The unit store pads the envs to the largest unit
count with all-zero rows.

THE LATCHES (the host's only writes per tick): :meth:`latch` takes one env's
intents as ``(kind, args)`` in issue order -- exactly what the intent tap
records (``Simulation.attach_intent_log``, ``match_replay.IntentLog``), args
normalised as the sim's own methods normalise them -- and turns them into
(1) a COLUMN SCATTER into the store's latch columns (:data:`LATCH_COLUMNS`:
each intent method writes exactly one ``Unit`` attribute, a function of its
arguments alone; the last write of a (unit, column) in the tick wins, as on
the host), every env's in ONE scatter at the tick's head, never a whole-row
pack; (2) the env's ORDER TABLE (``order_intent``, written onto the rows at
slot 1b' by ``unit_kernels.apply_orders``); (3) the env's ZOMBIE REQUEST SET
(``drive_zombies``, the HORDE latch). The tick row advances ON THE DEVICE at
slot 10. An env given no intents latches nothing: an env whose match has
ended keeps ticking (physics, its zombies, its timers) with no intents until
the caller replaces it (a per-env reset is P8's; see the P7e.5 as-built).

THE ONE READ: every face's counts are deferred into one arena
(``cuda_unit_books.cu``) and so is the physics tail's books D2H
(``bp.resident_tail_defer``): the tick's host syncs are the EOS stage's
``max_n_sub`` and the books read, two, whatever N. A refusal is raised there
by name WITH ITS ENV (``.env``) and refuses the BATCH: the tick is partly
written in every env and the caller stops them all (no face was shown unable
to let one env's fault reach another's state, so no env continues).

THE REFUSALS AT BUILD (by name, :class:`ConductorRefused`, with the env): a
sim without a physics runner, a ruleset that is not the continuous path (the
shooting and object doors), a map with water (the water pre-step reads the
host mirror), an ambient map (the sky bracket is a host bracket), requested
light, a non-empty edit queue window (the device door starts at the host's
seq with no rows), a paused sim, a heterogeneous batch, envs that disagree on
the stamp path, a unit carrying an optics override (the stamp's table
refusal).

NOT PART OF THE PLAYED TICK.
"""
from __future__ import annotations

import numpy as np


class ConductorRefused(RuntimeError):
    """A sim the device conductor does not serve (named at its build)."""


class ConductorFault(RuntimeError):
    """A tick in which a face's operands left its proved range (named at the
    one books read: the slot, the count)."""


class ConductorOverflow(RuntimeError):
    """A tick in which a record list (the ring, the request list, an effect
    list, an object or entity table, the cell buffer, the claims list, the
    edit door) dropped records past its capacity (named at the books read)."""


#: The deferred arena's capacity in int64 words: every face's counts of one
#: tick hold about 260 words per env (measured at N = 64: 16448), so the
#: arena is sized per env (a capacity, raised in P7e.5 from a flat 2^14 that
#: N = 64 crossed), never below the N = 1 floor.
BOOKS_CAP = 1 << 14
BOOKS_CAP_PER_ENV = 1 << 10


def _refusing_slots() -> dict:
    """The call lines whose kernel's REFUSED count is a refusal by name (the
    face raises it at its launch); every other refused count is a book
    (``store.last_refused``: the plants a full table refuses, the doses the
    body's store clips)."""
    from simulation.unit_kernels import ShotRefused, SpraySlotRefused
    return {"4 shooting": ShotRefused, "4b sprays": SpraySlotRefused}


def counts_by_env(words, n_env: int) -> np.ndarray:
    """A slot's deferred ``(coverage, faults, refused)`` counts as ``(3, N)``:
    the store's count block is laid out count-major (``cuda_unit_store.cu``
    ``end_counts``: coverage of every env, then faults, then refused), so env
    e's fault is ``[1, e]``."""
    return np.asarray(words, dtype=np.int64).reshape(3, int(n_env))


def _book_refusals() -> tuple:
    """The applier books that refuse the run by name, in the walk face's order."""
    from simulation import applier as a
    return (("dropped", a.EffectOverflow), ("refused", a.EffectRefused),
            ("bulk_refused", a.BulkGasOverflow), ("ledger_refused", a.LedgerRangeRefused),
            ("signal_refused", a.SignalRangeRefused), ("horde_refused", a.HordeCacheOverflow),
            ("heat_refused", a.HeatChainRefused), ("faults", a.ApplierFault))


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
#: The solver backends slot 7's resident chain runs on (run_on_cuda's set): the
#: conductor turns them on around its slot 7 and restores what it found, so a
#: host re-simulation in the same process keeps its own path.
SLOT7_BACKENDS = ("temperature", "water", "fire", "radiation", "bulk_flux",
                  "sl_advection", "mg_solve", "kick_compression", "combustion")

#: The latches that are not a unit column.
OTHER_LATCHES = ("order_intent", "drive_zombies")


class DeviceConductor:
    """The device tick of N homogeneous sims (one level; N = 1 is one sim);
    see the module docstring."""

    def __init__(self, sims, bp=None):
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
        sims = list(sims) if isinstance(sims, (list, tuple)) else [sims]
        if not sims:
            raise ConductorRefused("device conductor: at least one sim")
        self.cp, self.bp = cp, bp
        self.sims = sims
        self.sim = sims[0]                   # env 0 (the N = 1 twin)
        self.N = N = len(sims)
        for e, s in enumerate(sims):
            self._refuse(s, e)
            uk.shooting_door(s)
        uk.refuse_wego(sims)
        g = sims[0].gmap
        self.h, self.w = int(g._h), int(g._w)
        self.n_gases = int(g.gas.shape[0])
        self.o2 = int(g.gases.name_to_id["o2"])
        self.n2 = int(g.gases.name_to_id["inert_n2"])
        self.sim_time = 1.0 / float(sims[0]._tps)
        # -- the planes: every env's map made resident (ONE upload, here) -----
        # (ResidentBatch refuses a heterogeneous batch by name: grid, gas
        # table, tilt, tick rate, an ambient map)
        try:
            self.batch = ResidentBatch(sims)
        except ValueError as ex:
            raise ConductorRefused(f"device conductor: {ex}") from ex
        f = dict(self.batch.buffers)
        for name in applier.non_resident_planes():
            # the render-only ripple pair (host-computed by design, W6a): the
            # seal's 0.0 write needs a device plane; never read back
            f[name] = cp.asarray(np.ascontiguousarray(
                np.stack([getattr(s.gmap, name) for s in sims])))
        # On the C++ stamp path the host's `obstacles` IS `solid`
        # (GameMap._update_caches aliases them; the C++ stamp writes in place,
        # so a destroyed wall opens both at once): the device does the same
        # (the stamp's `obstacles` output is the `solid` buffer). The finding
        # P7e.2 reported, mirrored, not changed. One answer for the batch.
        alias = [s.gmap.obstacles is s.gmap.solid for s in sims]
        if len(set(alias)) != 1:
            raise ConductorRefused("device conductor: the envs disagree on the stamp path "
                                   f"(obstacles is solid: {alias})")
        self.obstacles_is_solid = alias[0]
        if self.obstacles_is_solid:
            f["obstacles"] = f["solid"]
        self.fields = f
        self.flight = cp.zeros((N, self.h, self.w), dtype=cp.bool_)
        self.src_out = cp.zeros((N, self.h, self.w), dtype=cp.int32)
        # -- the units: one (N, U, W) store, the short envs padded -------------
        self.U = U = max(1, max(len(s.units) for s in sims))
        self.uids = [[int(u.id) for u in s.units] for s in sims]
        self.row_of = [{uid: r for r, uid in enumerate(ids)} for ids in self.uids]
        self.store = unit_store.DeviceStore(N, U, bp=bp,
                                            tables=unit_tables.values_batch(sims))
        self.store.upload(np.stack([unit_store.pack(s.units, U) for s in sims]))
        lay = unit_store.layout()
        self._lay = lay
        mem = cp.cuda.UnownedMemory(self.store.device_ptr, 8 * int(np.prod(self.store.shape)),
                                    self.store)
        self._rows = cp.ndarray(self.store.shape, dtype=cp.int64,
                                memptr=cp.cuda.MemoryPointer(mem, 0))
        self._zombie = [[bool(u.is_zombie) for u in s.units] for s in sims]
        # -- the tick row (the latch written ONCE, here; advanced on the device)
        self.trow = unit_draws.TickRow.of_sims(sims)
        # -- the record lists ---------------------------------------------------
        self.ring = unit_events.EventRing(N, U, bp=bp)
        self.req = unit_requests.RequestList(N, U, bp=bp)
        self.eff = applier.DeviceApplier(N, applier.E_CAP, bp=bp, n_units=U)
        self.det = applier.DeviceApplier(N, applier.E_CAP, bp=bp, n_units=U)
        self.aux = applier.DeviceApplier(N, applier.E_CAP, bp=bp, n_units=U)
        self.cells = attack_lists.cell_buffer(N, bp=bp)
        self.claims = attack_lists.claims_list(N, bp=bp)
        self.door = edit_door.DeviceEditDoor(N, bp=bp)
        self.flush = edit_door.DeviceEditFlush(self.door, bp=bp)
        # -- the object and entity tables, staged ONCE from the sims -----------
        DT = ot.DeviceTable
        self.rounds = DT(us.ROUND_TABLE, N, n_units=U, row_cap=uk.round_row_cap(), bp=bp)
        self.noise = DT(us.NOISE_TABLE, N, n_units=U, row_cap=1, bp=bp)
        self.blast_log = DT(us.BLAST_LOG_TABLE, N, n_units=U, row_cap=1, bp=bp)
        self.grenades = DT(us.GRENADE_TABLE, N, bp=bp)
        self.charges = DT(us.CHARGE_TABLE, N, bp=bp)
        self.doors, self.door_tiles = uk.door_tables(N, bp=bp)
        self.ducts, self.vents = uk.vent_tables(N, bp=bp)
        self.sources, self.source_tiles = uk.source_tables(N, bp=bp)
        self.timed, self.heat = uk.hazard_tables(N, bp=bp)
        self.sig = uk.signal_tables(N, bp=bp)
        self.sig["doors"] = self.doors
        # (ONE list of the tables and their record functions, used by the
        # build for every env and by reset_env for one: never two lists)
        per_env = [self._table_records(s) for s in sims]
        for k, (tab, _r) in enumerate(per_env[0]):
            tab.stage_lists([recs[k][1] for recs in per_env])
        self.cache = horde_cache.DeviceHordeCache(N, U, persistent=True)
        self.cache.stage_lists([horde_cache.cache_records(s) for s in sims])
        self.cache.begin()
        self.door.begin()
        self.door.set_seq([int(s.edit_queue._seq) for s in sims])
        self.claims.begin()
        self.cells.begin()
        self.walk = uk.WalkLists(cells=self.cells, door=self.door, claims=self.claims,
                                 noise=self.noise, blast_log=self.blast_log,
                                 sources=self.sources, source_tiles=self.source_tiles)
        self.box_max = uk.spray_box_max(self.store, self.h, self.w)
        self.stamp_scratch = None
        # -- the latches' host side (per env) -----------------------------------
        self._cols: dict = {}                # (env, row, column) -> value, this tick
        self._orders = [dict() for _ in sims]
        self._zreqs = [None if s._zombie_requests is None else set(s._zombie_requests)
                       for s in sims]
        # -- the 9e gates: the level's entity classes (load constants; a slot
        # runs when ANY env holds its class -- an env without rows is a no-op
        # of the face, held by the batch invariance gate) ------------------------
        self._entity_gates()
        self.ticks = 0
        self.last_books: dict = {}           # book -> (N,) this tick's applier books
        self.last_profile = None
        self.swarm_death_rows = [None] * N
        self.obs = None                      # the device observation (P8.0), when attached
        self.decoder = None                  # the device decoder (P8.6a), when attached
        self._dec_uids = [frozenset() for _ in sims]   # P8.6a: the decoder-driven units
        self.ended_now = np.zeros(len(self.sims), dtype=bool)   # P8.7: the envs that ended this tick

    def attach_observation(self, obs) -> None:
        """#29 P8.0: the device observation (``simulation.device_obs``) whose
        blocks the walk builds at its end on a decision tick (call line 11').
        One per conductor."""
        if self.obs is not None and self.obs is not obs:
            raise ConductorRefused("device conductor: an observation is already attached")
        self.obs = obs

    def attach_decoder(self, dec) -> None:
        """#29 P8.6a: the device decoder (``simulation.device_decoder``): it
        decodes every decoder-driven row at the walk's head (call line 0),
        writing the latch columns straight into the store, and serves the
        horde's latch from the card at slot 5. THE TWO WAYS OF DRIVING: a team
        is the decoder's or the tap's (:meth:`latch`), never both -- a tap
        column intent for a decoder-driven unit, or ``drive_zombies`` for a
        horde the card serves, is refused by name; the commander's
        ``order_intent`` stays the tap's (the order table). One per conductor."""
        if self.decoder is not None and self.decoder is not dec:
            raise ConductorRefused("device conductor: a decoder is already attached")
        if self.obs is None or dec.obs is not self.obs:
            raise ConductorRefused("device conductor: the decoder's observation is not the "
                                   "attached one")
        self.decoder = dec
        drive = dec.drive.get()
        self._dec_uids = [frozenset(int(u.id) for t in range(drive.shape[1]) if drive[e, t]
                                    for u in s.roster(t) if u is not None)
                          for e, s in enumerate(self.sims)]

    # ------------------------------------------------------------------
    # The object and entity tables: ONE list (the build and the reset)
    # ------------------------------------------------------------------
    def _table_records(self, s) -> list:
        """Every object and entity table with env ``s``'s host records, in
        staging order: the build stages each table for every env from it,
        :meth:`reset_env` one env. A new table is a row HERE (its per-env
        staging comes with it)."""
        from simulation import object_tables as ot, unit_kernels as uk, unit_schema as us
        w_ = lambda t, objs: np.array(ot.records(t, objs), dtype=np.int64).reshape(  # noqa: E731
            -1, ot.table_layout(t).width)
        out = [(self.rounds, uk.rounds_of(s)), (self.noise, uk.noise_of(s)),
               (self.blast_log, uk.blast_log_of(s)),
               (self.grenades, w_(us.GRENADE_TABLE, s.projectiles)),
               (self.charges, w_(us.CHARGE_TABLE, s.planted_charges)),
               (self.doors, uk.doors_of(s)), (self.door_tiles, uk.door_tiles_of(s)),
               (self.ducts, uk.ducts_of(s)), (self.vents, uk.vents_of(s)),
               (self.sources, uk.sources_of(s)), (self.source_tiles, uk.source_tiles_of(s)),
               (self.timed, uk.timed_charges_of(s)), (self.heat, uk.heat_sources_of(s))]
        out += [(self.sig[ln], uk.SIGNAL_RECORDS[ln](s)) for ln in uk.SIGNAL_LISTS]
        return out

    def _entity_gates(self) -> None:
        """The 9e gates from the envs' entity classes (a slot runs when ANY
        env holds its class; the others run its face as a no-op)."""
        sims = self.sims
        self.has_bus = any(s._signal_bus is not None for s in sims)
        self.has_ducts = any(bool(s._ducts) for s in sims)
        self.has_doors = any(bool(s._doors) for s in sims)
        self.has_timed = any(bool(s._timed_charges) for s in sims)
        self.has_heat = any(bool(s._heat_sources) for s in sims)
        self.has_pumps = any(bool(s._pumps) for s in sims)

    # ------------------------------------------------------------------
    # #29 P8.9a: THE PER-ENV RESET
    # ------------------------------------------------------------------
    def reset_env(self, e: int, sim, spec=None) -> dict:
        """THE ONE RESET ENTRY: env ``e``'s next episode from ``sim`` -- a
        host ``Simulation`` at its tick 0, built as the reference builds an
        episode (``MatchSession``; equal to the reference by construction) --
        restaged into env ``e``'s slice of every device structure the build
        stages, every other env untouched and never paused. Called BETWEEN
        ticks (never inside the walk). Every refusal is checked, by name
        (:class:`ConductorRefused` / the structure's own), BEFORE anything is
        written: the new sim must fit the batch (grid, gas table, tilt, tick
        rate, stamp path, U, every table's and the horde cache's capacity) and
        the build's refusals hold for it. ``sim`` becomes ``sims[e]`` (the
        twin the gate's download lands on). Returns the restage's host work
        (``{"ms", "uploads"}``; ``"obs"`` the observation side's). With a
        device observation attached (P8.9b) its side is restaged too and the
        new episode's ``spec`` (its ``MatchSpec``) is required: the team
        picture, the books, the decoder's slot state, the per-env obs rows,
        the ended latch cleared, then the tick-0 observation built into the
        live buffers for env ``e`` alone (``DeviceObservation.reset_env``)."""
        import time as _time
        from simulation import applier, horde_cache, unit_draws, unit_kernels as uk
        from simulation import unit_store, unit_tables
        t0 = _time.perf_counter()
        cp = self.cp
        e = int(e)
        if not 0 <= e < self.N:
            raise IndexError(f"device conductor: env {e} of {self.N}")
        who = f"device conductor reset (env {e})"
        # -- every refusal first: nothing is written before the last one ------
        self._refuse(sim, e)
        uk.shooting_door(sim)
        uk.refuse_wego([sim])
        try:
            self.batch.check_reset(e, sim)
        except ValueError as ex:
            raise ConductorRefused(f"{who}: {ex}") from ex
        if (sim.gmap.obstacles is sim.gmap.solid) != self.obstacles_is_solid:
            raise ConductorRefused(f"{who}: the new sim's stamp path differs from the "
                                   f"batch's (obstacles is solid: {self.obstacles_is_solid})")
        if len(sim.units) > self.U:
            raise ConductorRefused(f"{who}: {len(sim.units)} units do not fit the batch's "
                                   f"U = {self.U}")
        planes = {}
        for name in applier.non_resident_planes():
            a = np.ascontiguousarray(getattr(sim.gmap, name))
            if a.shape != tuple(self.fields[name].shape[1:]) or a.dtype != self.fields[name].dtype:
                raise ConductorRefused(f"{who}: plane {name!r} is {a.dtype} {a.shape}, the "
                                       f"batch's {self.fields[name].dtype} "
                                       f"{tuple(self.fields[name].shape[1:])}")
            planes[name] = a
        rows = unit_store.pack(sim.units, self.U)
        tab = unit_tables.values(sim)               # its ranges refused by name
        trow = unit_draws.tick_row(sim)
        staged = [(t, t.env_rows(e, r)) for t, r in self._table_records(sim)]
        cache_rc = horde_cache.cache_records(sim)
        self.cache.check_env(e, cache_rc)
        obs_prep = None
        if self.obs is not None:
            obs_prep = self.obs.reset_prepare(e, sim, spec)   # P8.9b: its refusals too
        # -- the writes: env e's slice of every structure, on the tick stream /
        # the null stream (ordered with it), no host sync ----------------------
        uploads = 0
        self.batch.reset_env(e, sim)                # every resident plane, the swarm
        uploads += len(self.batch.buffers)
        for name, a in planes.items():
            self.fields[name][e] = cp.asarray(a)
            uploads += 1
        self.store.upload_env(e, rows, tables=tab)
        self.trow.write_env(e, trow)
        uploads += 3
        for t, rc in staged:
            t.stage_env_rows(e, rc)
            uploads += 1
        self.cache.stage_env(e, cache_rc)
        uploads += 4
        self.door.reset_env(e, int(sim.edit_queue._seq))
        uploads += 1
        # -- the host side of env e: the twin, the row maps, the latches -------
        self.sims[e] = sim
        self.sim = self.sims[0]
        self.uids[e] = [int(u.id) for u in sim.units]
        self.row_of[e] = {uid: r for r, uid in enumerate(self.uids[e])}
        self._zombie[e] = [bool(u.is_zombie) for u in sim.units]
        self._cols = {k: v for k, v in self._cols.items() if k[0] != e}
        self._orders[e] = {}
        self._zreqs[e] = (None if sim._zombie_requests is None
                          else set(sim._zombie_requests))
        self._entity_gates()
        self.box_max = max(self.box_max, uk.spray_box_of(tab[None, :], self.h, self.w))
        self.swarm_death_rows[e] = None
        self.ended_now[e] = False
        work = {"ms": 1000.0 * (_time.perf_counter() - t0), "uploads": uploads}
        if self.obs is not None:
            work["obs"] = self.obs.reset_env(e, sim, obs_prep)
            work["ms"] = 1000.0 * (_time.perf_counter() - t0)
        return work

    # ------------------------------------------------------------------
    # The build's refusals
    # ------------------------------------------------------------------
    @staticmethod
    def _refuse(sim, e=0):
        from simulation.ruleset import ContinuousRealtime
        who = f"device conductor (env {e})"
        r = sim.physics_runner
        if r is None:
            raise ConductorRefused(f"{who}: the sim has no physics runner")
        if not isinstance(sim.ruleset, ContinuousRealtime) or sim.ruleset.drives_units:
            raise ConductorRefused(f"{who}: the ruleset "
                                   f"{type(sim.ruleset).__name__} is not the continuous path")
        g = sim.gmap
        if g.water_sources or g.water_depth.any():
            raise ConductorRefused(f"{who}: the map holds water (the water "
                                   "pre-step and its dormancy read the host mirror)")
        if r._ambient_mask(g) is not None:
            raise ConductorRefused(f"{who}: an ambient map (the sky exchange "
                                   "is a host bracket, ACCEPTED GAP of the resident tick)")
        if r.engine.light_requested:
            raise ConductorRefused(f"{who}: light is requested (render state, until #81)")
        if sim.edit_queue._edits:
            raise ConductorRefused(f"{who}: the edit queue holds "
                                   f"{len(sim.edit_queue._edits)} edit(s): the device "
                                   "door starts empty")
        if sim.paused:
            raise ConductorRefused(f"{who}: the sim is paused")

    # ------------------------------------------------------------------
    # THE LATCHES
    # ------------------------------------------------------------------
    def latch(self, intents, env: int = 0) -> None:
        """Env ``env``'s intents this tick (``(kind, args)`` in issue order,
        args as the tap records them) into the device latches: the column
        scatter (applied ONCE for every env at :meth:`tick`'s head), the
        order table, the zombie request set. Refuses an unknown kind by name.
        A unit id the env does not hold is skipped (``get_unit`` -> None).
        An env given no intents latches nothing (its columns keep their
        values: the host's "no intent this tick")."""
        from simulation import unit_orders
        e = int(env)
        if not 0 <= e < self.N:
            raise IndexError(f"device conductor: env {e} of {self.N}")
        sim = self.sims[e]
        for kind, args in intents:
            args = tuple(args)
            if kind in LATCH_COLUMNS:
                col, val = LATCH_COLUMNS[kind]
                if int(args[0]) in self._dec_uids[e]:
                    raise ConductorRefused(f"device conductor (env {e}): {kind} for unit "
                                           f"{int(args[0])}, a decoder-driven unit (its "
                                           "columns are the device decoder's)")
                r = self.row_of[e].get(int(args[0]))
                if r is None:
                    continue
                self._cols[(e, r, col)] = val(args)
            elif kind == "order_intent":
                uid, what, where, ticks, target = (int(x) for x in args)
                until = unit_orders.NO_LIMIT if ticks == 0 else sim.tick + ticks
                self._orders[e][uid] = (what, where, until, target)
            elif kind == "drive_zombies":
                if self.decoder is not None and e in self.decoder.horde_envs:
                    raise ConductorRefused(f"device conductor (env {e}): drive_zombies on a "
                                           "horde the device decoder serves from the card")
                if self._zreqs[e] is None:
                    raise ConductorRefused(f"device conductor (env {e}): drive_zombies on "
                                           "a sim whose horde the conductor drives")
                self._zreqs[e].update(int(i) for i in args[0])
            else:
                raise ConductorRefused(f"device conductor (env {e}): intent {kind!r} has "
                                       "no latch")

    def _scatter(self) -> None:
        """The tick's column scatter: every env's latched (row, column)
        values in ONE CuPy scatter (one H2D of indices and values)."""
        from simulation import unit_schema as us
        cols, self._cols = self._cols, {}
        if not cols:
            return
        idx, vals = [], []
        U, W = self.store.shape[1], self.store.shape[2]
        for (e, r, name), v in cols.items():
            c = self._lay.column(name)
            enc = us.COLUMN_BY_NAME[name].encode(v)
            enc = tuple(enc) if isinstance(enc, tuple) else (enc,)
            if len(enc) != c.width:
                raise ConductorRefused(f"device conductor: latch {name!r} encodes "
                                       f"{len(enc)} ints, its column is {c.width} wide")
            for j, x in enumerate(enc):
                idx.append((e * U + r) * W + c.offset + j)
                vals.append(int(x))
        cp = self.cp
        flat = self._rows.reshape(-1)
        flat[cp.asarray(np.asarray(idx, dtype=np.int64))] = cp.asarray(
            np.asarray(vals, dtype=np.int64))

    def _order_table(self):
        from simulation import unit_kernels as uk
        per = []
        for o in self._orders:
            rows = [(uid,) + tuple(o[uid]) for uid in sorted(o)]
            o.clear()
            per.append(np.array(rows, dtype=np.int64).reshape(-1, uk.REQUEST_WIDTH))
        return uk.pad_requests(per)

    def _horde_latch(self):
        from simulation import applier, horde_cache
        modes = horde_cache._MODE
        m = applier.HORDE_LATCH.index("mode")
        base = len(applier.HORDE_LATCH)
        out = []
        for e, zr in enumerate(self._zreqs):
            row = np.zeros(base + self.U, dtype=np.int64)
            if zr is None:
                row[m] = modes["CONDUCTOR"]
            elif not zr:
                row[m] = modes["EMPTY"]
            else:
                row[m] = modes["SET"]
                for r, uid in enumerate(self.uids[e]):
                    if uid in zr:
                        row[base + r] = 1
                zr.clear()
            out.append(row)
        return out

    # ------------------------------------------------------------------
    # THE WALK
    # ------------------------------------------------------------------
    def _planes(self, names):
        return {n: self.fields[n] for n in names}

    def tick(self, profile: bool = False, idle=None) -> None:
        """ONE reference tick of every env on the device (``Simulation.step()``'s
        slots in order, one call line each, every face launched once for the
        N envs), its every count deferred to THE ONE BOOKS READ at its end
        (:meth:`_books_read`), where a refusal is raised by name with its env:
        a refused tick refuses the BATCH (the state is partly written in
        every env; the caller stops). ``profile``: the sub-slot profile of
        this tick lands in :attr:`last_profile` (``(label, gpu ms, host ms)``
        per call line, GPU ms between CUDA events on the tick stream).
        ``idle`` (#29 P8.10a): a host callable run once the walk is ENQUEUED
        and before the books read waits for it -- host work that overlaps
        the card's (the env's pre-build of the next episodes); it must not
        touch this conductor's state or the device."""
        from simulation import physics_runner as pr
        bp = self.bp
        self._scatter()
        if profile:
            bp.conductor_profile_begin()
            pr._PROFILE_MARK = bp.conductor_profile_mark
        bp.conductor_books_begin(max(BOOKS_CAP, BOOKS_CAP_PER_ENV * self.N))
        bp.resident_tail_defer(True)
        try:
            self._walk(bp.conductor_label)
            if idle is not None:
                idle()
            if profile:
                bp.conductor_profile_mark("books read")
        except BaseException:
            if bp.conductor_books_deferring():
                bp.conductor_books_end()
            if bp.resident_tail_pending():
                bp.resident_tail_finish()
            bp.resident_tail_defer(False)
            if profile:
                pr._PROFILE_MARK = None
                bp.conductor_profile_end()
            raise
        finally:
            bp.conductor_label("")
        # THE ONE READ: the unit side's arena AND the physics tail's books
        # (its D2H was enqueued on the same stream before it): one sync
        import time as _time
        t_read = _time.perf_counter()
        words, entries = bp.conductor_books_end()
        if profile:
            pr._PROFILE_MARK = None
            self.last_profile = list(bp.conductor_profile_end())
            self.last_profile.append(("books read", None,
                                      1000.0 * (_time.perf_counter() - t_read)))
        t_fold = _time.perf_counter()
        try:
            rows = bp.resident_tail_finish() if bp.resident_tail_pending() else None
        finally:
            bp.resident_tail_defer(False)
        if rows is not None:
            self.swarm_death_rows = list(rows)
        self.ticks += 1
        try:
            self._books_read(words, entries)
            # #29 P8.8b: the terminal observation of the envs whose match ended
            # THIS tick (the books read's flags: no sync of its own), built on
            # the tick's final state before any later tick enqueues
            if self.obs is not None:
                self.obs.terminal(self.ended_now)
        finally:
            if profile:
                self.last_profile.append(("books fold (host)", None,
                                          1000.0 * (_time.perf_counter() - t_fold)))

    def _walk(self, L) -> None:
        from simulation import physics_runner as pr, unit_kernels as uk
        from simulation import edit_door as _ed  # noqa: F401
        st, f, tr = self.store, self.fields, self.trow
        lists = self.walk
        dec = self.decoder
        # 0. the device decoder (P8.6a): the reference decodes in handle_frame
        # before sim.step -- every decoder-driven row's latch columns, then the
        # zombie controller's latch, at the tick's start; no sync
        if dec is not None and not dec.late:
            L("0 decode")
            dec.decode()
        # 1. the event ring begun; the trace books are host books (reset)
        L("1 tick head")
        self.ring.begin()
        for s in self.sims:
            s.gmap.reset_trace_books()
        # 1a. integrate + turn
        L("1a integrate + turn")
        uk.integrate_turn(st)
        # 1b. the hand verbs; their table writes (throws, plants, uses)
        L("1b intents")
        self.req.begin()
        uk.flight_plane(st, self._planes(uk.FLIGHT_SOURCES), out=self.flight)
        uk.verbs_device(st, self.req, tr, self.doors, self.door_tiles, self.charges,
                        {"flight": self.flight})
        uk.table_appends(st, self.req, self.grenades, self.charges)
        uk.door_uses(st, self.req, self.doors)
        # 1b'. the commander's orders
        L("1b' orders")
        if any(self._orders):                # dormant on an empty latch, as the host
            uk.apply_orders(st, self._order_table())
        # 1c. exposure decays
        L("1c exposure decay")
        uk.exposure_decay(st)
        # 2. the grenades fly (the flight plane as the slot found it), the
        # fuse-outs detonate at once
        L("2 grenades")
        uk.flight_plane(st, self._planes(uk.FLIGHT_SOURCES), out=self.flight)
        self.det.begin()
        uk.grenades(st, self.grenades, self.det, tr, {"flight": self.flight})
        uk.detonate(st, self.det, f, self.ring, lists, tick_row=tr)
        # 2'. the rounds in flight march, then their landing
        L("2' rounds")
        self.eff.begin()
        uk.rounds_march(st, self.eff, self.rounds, tr, self._planes(uk.SHOOT_FIELDS))
        uk.resolve(st, self.eff, f, self.ring, lists, tick_row=tr)
        # 2b. statuses
        L("2b statuses")
        uk.status_tick(st, self.ring)
        # 3. movement
        L("3 movement")
        uk.movement(st, self._planes(uk.MOVEMENT_FIELDS))
        # 4. shooting, 4a. melee, 4b. sprays (phase A, one effect list)
        L("4 shooting")
        self.eff.begin()
        self.cells.begin()
        uk.shooting(st, self.eff, self.rounds, self.noise, tr, self._planes(uk.SHOOT_FIELDS))
        L("4a melee")
        uk.melee(st, self.eff, tr)
        L("4b sprays")
        uk.sprays(st, self.eff, self.cells, {"solid": f["solid"]}, box_max=self.box_max)
        # 4c. phase B
        L("4c resolve")
        uk.resolve(st, self.eff, f, self.ring, lists, tick_row=tr)
        # 5. the horde (drains the noise list)
        L("5 zombie AI")
        hl = self._horde_latch()
        if dec is not None and dec.horde_envs:
            # P8.6a: the card's latch rows (no host list), device to device
            if not dec.horde_all:
                self.cache.set_latch(hl)
            dec.horde_latch_into(self.cache)
            hl = None
        uk.horde(st, self.aux, f, self.ring, self.noise, self.doors, self.door_tiles,
                 self.cache, hl, tr)
        # 6. the stamp from the store
        L("6 stamp_units")
        _r, self.stamp_scratch = uk.stamp(st, self._planes(n for n, _d in uk.STAMP_FIELDS),
                                          self.stamp_scratch)
        # 6b. the flush from the door
        L("6b edit flush")
        self.flush.run(tr, st.tables, self._planes(n for n, _d in _ed.FLUSH_PLANES),
                       self.n_gases, self.o2, self.n2)
        # 7. the physics head on the device (the blast source), then slot 7
        L("7 physics head (sources)")
        self.aux.begin()
        uk.take_sources(st, self.aux, f, lists, source_out=self.src_out)
        L("7 physics")
        was = [bool(getattr(self.bp, f"get_{b}_backend")()) for b in SLOT7_BACKENDS]
        for b in SLOT7_BACKENDS:
            getattr(self.bp, f"set_{b}_backend")(True)
        try:
            pr.run_resident_conductor([s.physics_runner for s in self.sims],
                                      [s.gmap for s in self.sims], self.sim_time,
                                      int(self.src_out.data.ptr), self.trow.ptr)
        finally:
            for b, on in zip(SLOT7_BACKENDS, was):
                getattr(self.bp, f"set_{b}_backend")(on)
        # 9. burn-through (the resident fire's list ranked on the device)
        L("9 burn-through walls")
        uk.burn_through_device(st, self.eff, f,
                               uk.BurnList.resident(self.bp, self.N, self.h, self.w),
                               tick_row=tr)
        # 9b. over-pressure walls
        L("9b over-pressure walls")
        uk.burst_walls(st, self.eff, f, tick_row=tr)
        # 9c. heat, 9c2. wind, 9c3. gas rows (+ poison, bite, corpse, stress)
        L("9c unit heat damage")
        uk.heat_damage(st, self.ring, self._planes(uk.HEAT_FIELDS), self.claims)
        L("9c2 wind on units")
        uk.wind_impulse(st, self._planes(uk.WIND_FIELDS))
        L("9c3 gas rows")
        uk.gas_rows(st, self.ring, self._planes(uk.GAS_FIELDS), self.claims)
        # 9d. ignition
        L("9d ignition")
        uk.ignition(st, self._planes(n for n, _d in uk.IGNITION_FIELDS))
        # 9e. the entities
        L("9e(a) signals")
        sig = self.sig
        if self.has_bus:
            uk.signals_emit(st, self.aux, f, sig, tr)
            L("9e(b) logic")
            uk.signals_logic(st, self.aux, sig)
            L("9e(c) door inputs")
            uk.signals_door_inputs(st, self.aux, sig)
            if self.has_pumps:
                L("9e(d) pumps")
                uk.signals_pumps(st, self.aux, f, sig)
        L("9e(d) vents")
        if self.has_ducts:
            uk.sweep_vents(st, self.aux, f, self.ducts, self.vents)
        L("9e(d) doors")
        if self.has_doors:
            uk.sweep_doors(st, self.aux, f, self.doors, self.door_tiles, tick_row=tr)
        L("9e(d') timed charges")
        if self.has_timed:
            self.aux.begin()
            uk.timed_charges(st, self.aux, f, self.ring, lists, self.timed, tick_row=tr)
        L("9e(d') planted charges")
        self.det.begin()
        uk.charges_tick(st, self.charges, self.det, tr)
        uk.detonate(st, self.det, f, self.ring, lists, tick_row=tr)
        L("9e(d'') heat sources")
        if self.has_heat:
            self.aux.begin()
            uk.heat_sources(st, self.aux, f, self.ring, lists, self.heat, tick_row=tr)
        else:
            # a source-free level only empties the log (the record list's
            # begin: its count to 0, on the tick stream, no host round trip)
            self.blast_log.buffer.begin()
        L("9e(e) swap")
        if self.has_bus:
            uk.signals_swap(st, self.aux, sig)
        # the tick-end clears: the heat plane, the claims list
        L("tick-end clears")
        uk.tick_end_clears(self.bp, {"heat": f["heat"]})
        self.claims.begin()
        # 10. the tick advances (the device row and the twins' host clocks)
        L("10 tick advance")
        self._advance()
        # 10'. orders expire; the ruleset's tick end (conversion, the prune)
        L("10' order expiry")
        if not (self.obs is not None and self.obs.expiry_brk):
            uk.expire_orders(st, tr)         # (THE BREAK ord_expiry, P8.6b's check: skipped)
        if self.obs is not None and (self.obs.early_vision or self.obs.early_picture):
            self.obs.vision()                # THE BREAK (P8.2's check): one slot too early
            if self.obs.early_picture:       # THE BREAK (P8.3's check): before the conversion
                self.obs.picture.update()
        if dec is not None and dec.horde_late:
            dec.horde_head()                 # THE BREAK (P8.6a's check): before the conversion
        L("10 tick end (ruleset)")
        uk.convert_zombies(st)
        uk.prune_grenades(self.grenades)
        # 11. the team picture and 11'. the device observation (when one is
        # attached; a match only, in no digest): who sees whom every tick
        # (P8.2), the device team picture's update (P8.3, the reference's
        # slot 11 itself), then the blocks on a decision tick (P8.0), on the
        # tick stream before the books read; no sync, no synced write
        if self.obs is not None:
            L("11' observation")
            self.obs.stage()
        if dec is not None and dec.late:
            dec.decode()                     # THE BREAK (P8.6a's check): after the walk

    # ------------------------------------------------------------------
    # THE ONE BOOKS READ
    # ------------------------------------------------------------------
    def _books_read(self, words, entries) -> None:
        """Every face's deferred counts (``cuda_unit_books.h``: one D2H, one
        sync), raised BY NAME as each face would have raised at its own
        launch -- the slot (the call line's label), what was crossed and THE
        ENV -- in walk order, the lowest env first within a call line (the
        first refusal wins; the exception carries ``.env``). A refusal
        refuses the batch: :meth:`tick` raised, the caller stops every env.
        The device flush's books are folded into each env's map books and
        draws here."""
        from simulation import applier, edit_door as ed
        from simulation.physics_runner import fold_edit_queue_books
        N = self.N
        w = np.asarray(words, dtype=np.int64)
        self.last_books = {}

        def first(a) -> int:
            return int(np.flatnonzero(np.asarray(a))[0])

        def refuse(cls, e, what, *a):
            ex = cls(f"device conductor (env {e}, {where}): {what}", *a)
            ex.env = e
            raise ex

        for label, kind, off, n in entries:
            v = w[int(off):int(off) + int(n)]
            where = f"tick {int(self.sims[0].tick) - 1}, slot {label!r}"
            if kind == "counts":                       # (coverage, faults, refused) x N
                c = counts_by_env(v, N)
                r_ = c[2] if label in _refusing_slots() else np.zeros(N, np.int64)
                if c[1].any() and (not r_.any() or first(c[1]) <= first(r_)):
                    e = first(c[1])
                    refuse(ConductorFault, e, f"{int(c[1, e])} row(s) left the range "
                           f"the kernel was proved for")
                if r_.any():
                    e = first(r_)
                    refuse(_refusing_slots()[label], e,
                           f"{int(c[2, e])} refused by name (a refused count)")
            elif kind == "list_dropped":
                if v.any():
                    e = first(v)
                    refuse(ConductorOverflow, e, f"a list dropped {int(v[e])} "
                           f"record(s) past its capacity")
            elif kind == "books":
                b = applier.books_dict(v.reshape(N, -1))
                bad = [(first(b[book]), book, exc) for book, exc in _book_refusals()
                       if np.asarray(b[book]).any()]
                if bad:
                    e, book, exc = min(bad, key=lambda t: t[0])
                    refuse(exc, e, f"{book} = {int(b[book][e])}")
                for k_, x in b.items():
                    self.last_books[k_] = self.last_books.get(k_, 0) + np.asarray(
                        x, dtype=np.int64)
            elif kind == "picture_faults":             # #29 P8.3: (N,) fault bits
                if v.any():
                    from simulation.device_picture import FAULT_BITS, fault_text
                    e = first(v)
                    over = {n for n, _ in FAULT_BITS if n.endswith("OVERFLOW")}
                    bits = int(v[e])
                    names = [n for i, (n, _) in enumerate(FAULT_BITS) if bits & (1 << i)]
                    refuse(ConductorOverflow if over & set(names) else ConductorFault, e,
                           f"the team picture: {fault_text(bits)}")
            elif kind == "reward_books":               # #29 P8.7: (3, N) words
                # the mismatch count and fault bits raised by name; the
                # envs whose match ended this tick (no third sync)
                wv = v.reshape(3, N)
                self.ended_now = wv[2].astype(bool)
                if self.obs is not None:
                    try:
                        self.obs.books.check_words(wv)
                    except Exception as ex:          # the env and the slot named
                        ex.args = (f"device conductor ({where}): {ex.args[0]}",) + ex.args[1:]
                        raise
            elif kind == "decoder_faults":             # #29 P8.6a: (N, FAULT_W) words
                if self.decoder is not None:
                    try:
                        self.decoder.check_words(v)
                    except ValueError as ex:          # check_actions' refusal, by name
                        ex.args = (f"device conductor ({where}): {ex.args[0]}",) + ex.args[1:]
                        raise
            elif kind in ("door_counts", "object_counts"):
                c = v.reshape(N, -1)
                f_ = c[:, 1]
                r_ = c[:, 2] if kind == "object_counts" else np.zeros(N, np.int64)
                if f_.any() and (not r_.any() or first(f_) <= first(r_)):
                    e = first(f_)
                    refuse(ConductorFault, e, f"{int(c[e, 1])} fault(s)")
                if r_.any():
                    from simulation.unit_kernels import ObjectSlotRefused
                    refuse(ObjectSlotRefused, first(r_), "a WEGO grenade")
            elif kind == "flush_books":
                slots = self.flush.slots
                books = v[:N * slots].reshape(N, slots)
                words_ = v[N * slots:N * slots + N]
                fails = books[:, self.flush.fail_slot]
                if (words_ != -1).any() or fails.any():
                    for e in range(N):
                        word = int(words_[e]) & ((1 << 64) - 1)
                        if word != (1 << 64) - 1:
                            code, row = word >> 32, word & 0xffffffff
                            name, why = ed.FLUSH_FAULTS.get(code, ("?", "an unknown code"))
                            refuse(ed.EditFlushRefused, e, f"the flush refused ({name}: "
                                   f"{why}; row {row})", e, code, row)
                        if int(fails[e]):
                            refuse(ed.EditQueueFailed, e, f"{int(fails[e])} edit result(s) "
                                   f"left their field's range")
                for e, s in enumerate(self.sims):
                    fold_edit_queue_books(self.bp.EDIT_QUEUE_LAYOUT, s.gmap, books[e],
                                          s.draws)

    def _advance(self) -> None:
        from simulation.unit_draws import TICK_ROW_KEY, _M32
        b = self.trow.buf
        b[:, 0] += 1
        k = TICK_ROW_KEY + 2
        b[:, k] = (b[:, k] + 1) & _M32
        for s in self.sims:
            s.tick += 1
            s.real_time += self.sim_time

    # ------------------------------------------------------------------
    # THE GATE'S DOWNLOAD
    # ------------------------------------------------------------------
    def download(self, envs=None) -> None:
        """Land the device state on the sims' host objects (what the digest,
        ``field_ab_harness.capture_snapshot``, reads) for ``envs`` (default
        every env): the unit rows (and the conversion's host-only "Z-" name),
        the resident planes, the tick's synced events, the planted charges and
        every entity table on the sim's own runtimes."""
        from simulation import (object_tables as ot, swarm, unit_events,
                                unit_kernels as uk, unit_schema as us, unit_store)
        from simulation.gamemap import GameMap
        envs = range(self.N) if envs is None else list(envs)
        rows = self.store.download()
        recs = self.ring.records()
        crow, ccnt, _ = self.charges.download()
        doors = (self.doors.records(), self.door_tiles.records()) if self.has_doors else None
        ducts = (self.ducts.records(), self.vents.records()) if self.has_ducts else None
        sigs = ({k: self.sig[k].records() for k in ("signals", "nodes", "pumps")}
                if self.has_bus else None)
        timed = self.timed.records() if self.has_timed else None
        heat = self.heat.records() if self.has_heat else None
        sw = set(swarm.SWARM_RESIDENT_NAMES) | set(swarm.SWARM_PLANE_NAMES)
        for e in envs:
            sim = self.sims[e]
            g = sim.gmap
            unit_store.unpack(rows[e], sim.units)
            for k, u in enumerate(sim.units):
                if u.is_zombie and not self._zombie[e][k] and not u.name.startswith("Z-"):
                    u.name = f"Z-{u.name}"
                self._zombie[e][k] = bool(u.is_zombie)
            names = [n for n in GameMap._RESIDENT_FIELD_NAMES if n not in sw]
            if self.obstacles_is_solid:
                names = [n for n in names if n != "obstacles"]   # the host alias of solid
            g.to_host(names + list(swarm.resident_exit_names(g)))
            sim.tick_events[:] = [unit_events.event_of(r) for r in recs[e]]
            sim.planted_charges[:] = ot.unpack(us.CHARGE_TABLE, crow[e], int(ccnt[e]))
            if doors is not None and sim._doors:
                uk.doors_to_host(sim, doors[0][e], doors[1][e])
            if ducts is not None and sim._ducts:
                uk.ducts_to_host(sim, ducts[0][e])
                uk.vents_to_host(sim, ducts[1][e])
            if sigs is not None and sim._signal_bus is not None:
                uk.signals_to_host(sim, sigs["signals"][e])
                uk.nodes_to_host(sim, sigs["nodes"][e])
                uk.pumps_to_host(sim, sigs["pumps"][e])
            if timed is not None and sim._timed_charges:
                uk.timed_charges_to_host(sim, timed[e])
            if heat is not None and sim._heat_sources:
                uk.heat_sources_to_host(sim, heat[e])

    def nondigest_records(self, env: int = 0) -> dict:
        """Env ``env``'s cross-tick state in NO digest, for the gate: the
        rounds, the grenades, the noise list, the blast sources, the edit
        door's window and the horde cache, each as the reference's record
        function writes it (a download)."""
        from simulation import edit_door as ed
        e_ = int(env)
        e, t = self.door.rows()[e_]
        return {"rounds": self.rounds.records()[e_], "grenades": self.grenades.records()[e_],
                "noise": self.noise.records()[e_], "sources": self.sources.records()[e_],
                "source_tiles": self.source_tiles.records()[e_],
                "door": (e[:, :ed.EDIT_COLS].tolist(), t.tolist()),
                "horde": self.cache.records()[e_]}


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
