"""THE DEVICE DECODER (#29 P8.6a, residency design §4.4c "P8 RE-CUT, VERSION
2"; inventory ``docs/residency_p8_observation_inventory_2026-11-20.md`` part
F): action spec v1's decisions as a device array, and the per-tick decoder of
the ``policy`` controller (``control_policy.AgentPolicy``) on the card,
writing STRAIGHT into the unit store's latch columns. The host decoder
(``AgentPolicy.set_decisions`` / ``_decode``, ``issue_row``, ``VERB_RULES``)
stays THE reference; this is its twin. CLASS T (training only).

THE SHAPE::

    c = DeviceConductor(sims); obs = DeviceObservation(c, ...)
    dec = DeviceDecoder(obs, specs)            # every marine team decoder-driven
    dec.set_decisions(actions)                 # (N, rows, 5) int32, CuPy / torch
    c.tick()                                   # decoded at the tick's head

THE DECISION (:meth:`DeviceDecoder.set_decisions`, once per decision, the
trainer's one device write): the decisions are copied into the decoder's own
``(N, rows, 5)`` int32 buffer (a torch tensor arrives by DLPack, no host
copy), then ONE launch on the tick stream (``bp.obs_dec_check``, one thread
per env): ``action_spec_v1.check_actions`` (an index outside its head is a
counted per-env fault, raised BY NAME -- the host's ``ValueError`` and its
message -- at the next tick's one books read), ``masked_choices`` into
:attr:`DeviceDecoder.masked` ``(N, rows)`` uint8 (the device form of
``info["masked_choices"]``, against the masks the card SERVED with the
observation the decision answers: the mask buffers as they are at the call),
the pointer turned into a game id against the ``list_uid`` buffer of that
same observation AT THE CALL (the resolution is done before any tick, so the
next observation cannot overwrite what it reads: no copy is kept), and
``set_decisions``' no-sim fallbacks.

THE DECODE (:meth:`DeviceDecoder.decode`, the conductor's tick head, before
the walk -- the reference decodes in ``handle_frame`` before ``sim.step``):
one thread per (env, row) of a decoder-driven team (``bp.obs_decode``),
``AgentPolicy._decode`` line for line over the slot state on the card (the
held throw aim with its alias of the decision's aim object, ``last_vec``,
``aimed_uid``, ``prev``), the aimed unit's centre from the DEVICE TEAM
PICTURE's record (live while seen, last seen otherwise), ``aim_vector`` and
``onto_reach_q`` in float64 through the observation's float kit, the hand
latch on the FIRST tick of a decision only (``VERB_RULES``), each issued
intent written into its ``conductor.LATCH_COLUMNS`` column in that column's
encoding (the throttle and the plant timer clamped as the sim's methods
clamp). The decoded rows' record (for the training-episode record, old P8c):
:attr:`decisions` and the slot state :attr:`state`.

THE HORDE (:meth:`DeviceDecoder.horde_head`, at the same tick head): the
zombie controller's latch -- ``ZombieControl``: "every living zombie at the
tick's start", SET with a flag per store row, EMPTY when none lives (the
slot then returns before ``begin_tick``) -- written on the card into
:attr:`horde_rows`, which the conductor's slot 5 hands the HORDE stage with
no host list. An ``idle`` horde is EMPTY every tick.

THE OBSERVATION'S COLUMNS (:meth:`DeviceDecoder.build_obs`, after the windows
on a decision tick): the vector's ``prev`` block and the list's
``prev_aim`` / ``on_next_tick`` on the aimed entry, from the slot state and
``DeviceObservation.aimed`` (which the decode writes every tick).

REFUSALS AT BUILD (:class:`DecoderRefused`, by name): a marine team whose
controller is not ``policy`` (design §12.3: device training is self-play or
against the horde), a horde controller other than ``zombies`` / ``idle``.
The tap carries only the commander's orders for a decoder-driven env
(``DeviceConductor.latch`` refuses a unit column for a decoder row, and
``drive_zombies`` for a card-driven horde, by name).
"""
from __future__ import annotations

import numpy as np

#: The decoder's per-env table (int64, ``ActionTables`` quantized once per
#: match, read from the reference's own object): the three throttles, the
#: hand-grenade fuse, the four reaches (Q16 tiles), the four plant timers,
#: the aimable list positions [AIM_LO, AIM_HI), the list's entries, the
#: plant row's clamp (Simulation.plant_charge_intent), the throw kinds, the
#: five heads' sizes.
TAB_FIELDS = ("THROTTLE0", "THROTTLE1", "THROTTLE2", "FUSE_TICKS", "REACH0", "REACH1",
              "REACH2", "REACH3", "TIMER0", "TIMER1", "TIMER2", "TIMER3", "AIM_LO", "AIM_HI",
              "N_ENTRIES", "PLANT_LO", "PLANT_HI", "N_KINDS", "SIZE0", "SIZE1", "SIZE2",
              "SIZE3", "SIZE4")
#: The per-(env, row) state record (int64): the decision (``_Decision``: feet,
#: the aim as kind / value / index with its ``last_point``, trigger, hand
#: after the fallbacks, arg; HAVE once a decision was handed over, FIRST until
#: its first decoded tick) and the slot (``_SlotState``: EXISTS once decoded,
#: the held throw aim TA -- ALIAS when it IS the decision's aim object --,
#: ``last_vec``, ``aimed_uid``, ``prev`` with HAS_PREV, the unit id).
STATE_FIELDS = ("HAVE", "FIRST", "D_FEET", "D_KIND", "D_VAL", "D_LPV", "D_LPX", "D_LPY",
                "D_IDX", "D_TRIG", "D_HAND", "D_ARG", "EXISTS", "TA", "TA_ALIAS", "TA_KIND",
                "TA_VAL", "TA_LPV", "TA_LPX", "TA_LPY", "LVX", "LVY", "AIMED", "HAS_PREV",
                "PREV0", "PREV1", "PREV2", "PREV3", "PREV4", "PREV5", "UID", "BAD")
#: An aim's kind (``control_policy._Aim.kind``).
AIM_KINDS = ("tok", "walk", "keep", "unit")
#: The fault words per env (int64): the out-of-range count, then per head the
#: first row (-1: none) and the value it sent.
FAULT_W = 1 + 2 * 5
#: The check's breaks (``OBS_BREAK``) in the kernels: ``dec_every`` (the hand
#: latch issued on every tick of a decision), ``dec_div`` (the aim vector as
#: dx / mag in place of dx * (1 / mag)).
BREAK_CODES = {"dec_every": 1, "dec_div": 2}
#: Horde modes of an env (``horde_mode``): the conductor drives it (no
#: controller), the card serves ``ZombieControl``, an ``idle`` controller.
HORDE_CONDUCTOR, HORDE_DEVICE, HORDE_IDLE = 0, 1, 2


class DecoderRefused(RuntimeError):
    """A match the device decoder does not serve (by name, at its build)."""


def constants() -> list:
    """The decoder's generated constants (``obs_layout.h`` through
    ``device_obs._constants``), each read from action spec v1's own tables,
    the policy's rules and the horde latch's layout."""
    from simulation import action_spec_v1 as a
    from simulation import applier
    from control_policy import VERB_RULES
    # the decoder issues exactly VERB_RULES' kinds with these classes and
    # gates: a changed rule must land in the kernel (refused here otherwise)
    want = {"set_move_dir": ("hold", "unless:clear_move_dir"),
            "clear_move_dir": ("hold", "flag"), "set_aim": ("hold", "always"),
            "set_trigger": ("hold", "always"), "throw_grenade_intent": ("latch", "vector"),
            "use_intent": ("latch", "flag"), "reload_intent": ("latch", "flag"),
            "shove_intent": ("latch", "flag"), "melee_intent": ("latch", "flag"),
            "set_flashlight": ("hold", "always"), "plant_charge_intent": ("latch", "nonzero")}
    if dict(VERB_RULES) != want:
        raise DecoderRefused(f"device decoder: control_policy.VERB_RULES {VERB_RULES} is not "
                             f"the rule set the decode kernel issues ({want}): the kernel "
                             f"changes in the same patch")
    out = [("OBS_DEC_N_DIRS", a.N_DIRS), ("OBS_DEC_N_SPEEDS", a.N_SPEEDS),
           ("OBS_DEC_N_ARG", a.N_ARG), ("OBS_DEC_ARG_ONTO", a.ARG_ONTO),
           ("OBS_DEC_N_FRACTIONS", a.N_FRACTIONS), ("OBS_DEC_N_HEADS", len(a.HEADS)),
           ("OBS_DEC_FAULT_W", FAULT_W)]
    out += [(f"OBS_DEC_TAB_{f}", i) for i, f in enumerate(TAB_FIELDS)]
    out.append(("OBS_DEC_TAB_W", len(TAB_FIELDS)))
    out += [(f"OBS_DEC_S_{f}", i) for i, f in enumerate(STATE_FIELDS)]
    out.append(("OBS_DEC_S_W", len(STATE_FIELDS)))
    out += [(f"OBS_DEC_AIM_{k.upper()}", i) for i, k in enumerate(AIM_KINDS)]
    out += [(f"OBS_DEC_BREAK_{k.upper()}", v) for k, v in BREAK_CODES.items()]
    modes = dict(applier.HORDE_MODES)
    out += [("OBS_DEC_HORDE_BASE", len(applier.HORDE_LATCH)),
            ("OBS_DEC_HORDE_MODE", applier.HORDE_LATCH.index("mode")),
            ("OBS_DEC_HORDE_SET", modes["SET"]), ("OBS_DEC_HORDE_EMPTY", modes["EMPTY"]),
            ("OBS_DEC_HORDE_DEVICE", HORDE_DEVICE), ("OBS_DEC_HORDE_IDLE", HORDE_IDLE)]
    return out


def compass_init() -> str:
    """``action_spec_v1.COMPASS_Q16`` as a C initialiser (x0, y0, x1, ...)."""
    from simulation import action_spec_v1 as a
    return ", ".join(str(int(v)) for xy in a.COMPASS_Q16 for v in xy)


def table_values(sim, n_slots: int) -> np.ndarray:
    """ONE env's decoder table (``(TAB_W,)`` int64) from the reference's own
    objects: ``action_spec_v1.action_tables(sim, n_slots)`` and the plant
    row's clamp as ``Simulation.plant_charge_intent`` computes it."""
    from simulation import action_spec_v1 as a
    from simulation.weapons import PLANT_WEAPON, fuse_ticks_from_seconds
    tb = a.action_tables(sim, n_slots)
    row = sim.weapons_tables.weapons.by_name[PLANT_WEAPON]
    vals = (list(tb.throttles) + [tb.fuse_ticks] + list(tb.reaches_q) + list(tb.plant_timers)
            + [tb.aim_lo, tb.aim_hi, tb.n_entries,
               fuse_ticks_from_seconds(row.fuse_min_seconds, sim._tps),
               fuse_ticks_from_seconds(row.fuse_max_seconds, sim._tps), tb.n_kinds]
            + list(tb.sizes))
    if len(vals) != len(TAB_FIELDS):
        raise DecoderRefused(f"device decoder: {len(vals)} table values for "
                             f"{len(TAB_FIELDS)} fields")
    return np.array(vals, dtype=np.int64)


def check_message(faults: np.ndarray, sizes) -> tuple | None:
    """The host's ``check_actions`` refusal from the fault words ``(N,
    FAULT_W)``: the first head (head order) with a bad index in any env, its
    lowest env and that env's first row -- ``(env, message)`` -- or None."""
    from simulation.action_spec_v1 import HEADS
    f = np.asarray(faults, dtype=np.int64)
    for h, n in enumerate(sizes):
        for e in range(f.shape[0]):
            if f[e, 0] and f[e, 1 + 2 * h] >= 0:
                r, v = int(f[e, 1 + 2 * h]), int(f[e, 2 + 2 * h])
                return e, (f"action out of range: env {e}, row {r}, head {HEADS[h]!r} "
                           f"sent {v} (size {int(n)})")
    return None


def state_form(state_row) -> dict:
    """One (env, row) state record as plain values (the gate's reading): the
    slot's ``aimed``, ``last_vec``, ``prev`` (None before the first), the
    held throw aim as (kind, value, last_point) with an alias resolved, and
    the decision aim's last_point; None for a slot never decoded."""
    s = {f: int(v) for f, v in zip(STATE_FIELDS, np.asarray(state_row).tolist())}
    if not s["EXISTS"]:
        return None
    d_lp = (s["D_LPX"], s["D_LPY"]) if s["D_LPV"] else None
    ta = None
    if s["TA"]:
        if s["TA_ALIAS"]:
            ta = (AIM_KINDS[s["D_KIND"]], s["D_VAL"] if AIM_KINDS[s["D_KIND"]] in ("tok", "unit")
                  else -1, d_lp)
        else:
            lp = (s["TA_LPX"], s["TA_LPY"]) if s["TA_LPV"] else None
            ta = (AIM_KINDS[s["TA_KIND"]], s["TA_VAL"] if AIM_KINDS[s["TA_KIND"]] in ("tok", "unit")
                  else -1, lp)
    prev = tuple(s[f"PREV{i}"] for i in range(6)) if s["HAS_PREV"] else None
    return {"uid": s["UID"], "aimed": s["AIMED"], "last_vec": (s["LVX"], s["LVY"]),
            "prev": prev, "throw_aim": ta, "aim_lp": d_lp}


def host_state_form(policy, slot: int) -> dict:
    """The reference policy's slot ``slot`` in :func:`state_form`'s form."""
    st = policy._slot.get(slot)
    if st is None:
        return None
    ta = None
    if st.throw_aim is not None:
        a = st.throw_aim
        ta = (a.kind, int(a.value) if a.kind in ("tok", "unit") else -1,
              None if a.last_point is None else tuple(int(x) for x in a.last_point))
    d = policy._dec[slot] if policy._dec is not None and slot < len(policy._dec) else None
    lp = None
    if d is not None and d.aim.last_point is not None:
        lp = tuple(int(x) for x in d.aim.last_point)
    prev = None if st.prev is None else tuple(int(x) for x in st.prev)
    return {"uid": int(st.uid), "aimed": int(st.aimed_uid),
            "last_vec": tuple(int(x) for x in st.last_vec), "prev": prev, "throw_aim": ta,
            "aim_lp": lp}


class DeviceDecoder:
    """The device decoder of a ``DeviceObservation``'s N envs (see the module
    docstring). ``specs``: each env's ``MatchSpec`` (its teams' and horde's
    controllers, refused by name unless every marine team is ``policy`` and
    the horde ``zombies`` / ``idle``)."""

    def __init__(self, obs, specs):
        import cupy as cp
        self.cp, self.obs, self.c, self.bp = cp, obs, obs.c, obs.bp
        c = obs.c
        specs = list(specs)
        if len(specs) != c.N:
            raise DecoderRefused(f"device decoder: {len(specs)} specs for {c.N} envs")
        self.N, self.T, self.S, self.n_rows = c.N, obs.T, obs.S, obs.n_rows
        drive = np.zeros((self.N, self.T), dtype=np.uint8)
        hmode = np.zeros((self.N,), dtype=np.int64)
        for e, (sim, spec) in enumerate(zip(c.sims, specs)):
            for t, team in enumerate(spec.teams):
                if team.controller != "policy":
                    raise DecoderRefused(
                        f"device decoder (env {e}): team {t} is driven by {team.controller!r}: "
                        f"on the device path every marine team is the decoder's (design "
                        f"§12.3: self-play or against the horde)")
                drive[e, t] = 1
            z = getattr(spec, "zombie_controller", "zombies")
            if sim._zombie_requests is None:
                hmode[e] = HORDE_CONDUCTOR
            elif z == "zombies":
                hmode[e] = HORDE_DEVICE
            elif z == "idle":
                hmode[e] = HORDE_IDLE
            else:
                raise DecoderRefused(f"device decoder (env {e}): the horde controller {z!r} has "
                                     f"no device form (zombies or idle)")
        self.drive = cp.asarray(drive)
        self.horde_mode_host = hmode
        self.horde_mode = cp.asarray(hmode.astype(np.int32))
        self.horde_envs = [e for e in range(self.N) if hmode[e] != HORDE_CONDUCTOR]
        self.horde_all = len(self.horde_envs) == self.N
        self.table_host = np.stack([table_values(sim, self.S) for sim in c.sims])
        sizes = {tuple(r) for r in self.table_host[:, len(TAB_FIELDS) - 5:]}
        if len(sizes) != 1:
            raise DecoderRefused(f"device decoder: the envs' head sizes {sorted(sizes)} differ")
        self.sizes = tuple(int(x) for x in sizes.pop())
        self.table = cp.asarray(self.table_host)
        self.decisions = cp.zeros((self.N, self.n_rows, 5), dtype=cp.int32)
        self.state = cp.zeros((self.N, self.n_rows, len(STATE_FIELDS)), dtype=cp.int64)
        self.state[:, :, STATE_FIELDS.index("AIMED")] = -1
        self.masked = cp.zeros((self.N, self.n_rows), dtype=cp.uint8)
        self.faults = cp.zeros((self.N, FAULT_W), dtype=cp.int64)
        self.due = cp.zeros((self.N,), dtype=cp.uint8)
        from simulation import applier
        self.horde_rows = cp.zeros((self.N, len(applier.HORDE_LATCH) + int(c.U)),
                                   dtype=cp.int64)
        self._horde_idx = cp.asarray(np.array(self.horde_envs, dtype=np.int64))
        # THE BREAK hooks (the check's), all off in a real run
        self.brk = 0
        self.late = False                    # decode after the walk
        self.horde_late = False              # the horde latch at the previous tick's end
        self.snap = None                     # a copy of the store after the decode (gate)
        self.snapping = False
        self.decodes = 0
        c.attach_decoder(self)
        for e in self.horde_envs:            # the tap no longer drives this horde: the
            c._zreqs[e] = set()              # host row is EMPTY, overwritten at slot 5
        self.horde_head()                    # the build's tick: the latch of tick 0

    # -- #29 P8.9b: the per-env reset ------------------------------------------
    def reset_prepare(self, e: int, sim, spec):
        """Env ``e``'s next episode's decoder rows from the reference's own
        objects, refused by name as the build refuses (a team not
        ``policy``, a horde controller with no device form, other head
        sizes): ``(drive row, horde mode, table row)``. Writes nothing."""
        e = int(e)
        drive = np.zeros((self.T,), dtype=np.uint8)
        for t, team in enumerate(spec.teams):
            if team.controller != "policy":
                raise DecoderRefused(
                    f"device decoder reset (env {e}): team {t} is driven by "
                    f"{team.controller!r}: on the device path every marine team is the "
                    f"decoder's")
            drive[t] = 1
        z = getattr(spec, "zombie_controller", "zombies")
        if sim._zombie_requests is None:
            mode = HORDE_CONDUCTOR
        elif z == "zombies":
            mode = HORDE_DEVICE
        elif z == "idle":
            mode = HORDE_IDLE
        else:
            raise DecoderRefused(f"device decoder reset (env {e}): the horde controller {z!r} "
                                 f"has no device form (zombies or idle)")
        tab = table_values(sim, self.S)
        sizes = tuple(int(x) for x in tab[len(TAB_FIELDS) - 5:])
        if sizes != self.sizes:
            raise DecoderRefused(f"device decoder reset (env {e}): head sizes {sizes}, the "
                                 f"batch's {self.sizes}")
        return drive, int(mode), tab

    def reset_env(self, e: int, prep) -> None:
        """Env ``e``'s decoder for its next episode: no slot ever decoded
        (the state record zero, ``aimed`` -1: a new ``AgentPolicy``), no
        pending decision (its decisions, due flag, masked choices and fault
        words zero), its drive row, horde mode and table row from ``prep``
        (:meth:`reset_prepare`); the conductor's decoder-driven units and
        horde latch for the env. Every other env untouched."""
        cp, c = self.cp, self.c
        e = int(e)
        drive, mode, tab = prep
        st = np.zeros(tuple(self.state.shape[1:]), dtype=np.int64)
        st[:, STATE_FIELDS.index("AIMED")] = -1
        self.state[e] = cp.asarray(st)
        self.decisions[e] = 0
        self.masked[e] = 0
        self.faults[e] = 0
        self.due[e] = 0
        self.drive[e] = cp.asarray(drive)
        self.horde_mode_host[e] = mode
        self.horde_mode[e] = mode
        self.horde_envs = [k for k in range(self.N) if self.horde_mode_host[k] != HORDE_CONDUCTOR]
        self.horde_all = len(self.horde_envs) == self.N
        self._horde_idx = cp.asarray(np.array(self.horde_envs, dtype=np.int64))
        self.table_host[e] = tab
        self.table[e] = cp.asarray(tab)
        sim = c.sims[e]
        c._dec_uids[e] = frozenset(int(u.id) for t in range(self.T) if drive[t]
                                   for u in sim.roster(t) if u is not None)
        if e in self.horde_envs:
            c._zreqs[e] = set()              # the card serves this horde (as at the build)
        self.horde_head()                    # the latch of the env's tick 0

    # -- the decision ----------------------------------------------------------
    def set_decisions(self, actions, envs=None, list_uid=None) -> None:
        """Hand over a decision: ``actions`` ``(N, rows, 5)`` int (CuPy, a
        torch tensor by DLPack, or numpy -- one H2D); ``envs`` an ``(N,)``
        bool mask of the envs that decide now (default all); ``list_uid``
        the ``(N, rows, E)`` the pointers resolve against (default the
        observation's buffer as it is now: the one the decision answers).
        One launch on the tick stream, no sync."""
        cp = self.cp
        if hasattr(actions, "__dlpack__") and not isinstance(actions, (np.ndarray, cp.ndarray)):
            actions = cp.from_dlpack(actions)
        a = cp.asarray(actions)
        if a.shape != self.decisions.shape:
            raise ValueError(f"device decoder: decisions must be {self.decisions.shape}, got "
                             f"{a.shape}")
        if not cp.issubdtype(a.dtype, cp.integer):
            raise ValueError(f"device decoder: decisions must be integer, got {a.dtype}")
        # an index past int32 would wrap: it is out of range whatever it is
        big = a.astype(cp.int64)
        self.decisions[...] = cp.clip(big, -(2 ** 31), 2 ** 31 - 1).astype(cp.int32)
        self.due[...] = 1 if envs is None else cp.asarray(np.asarray(envs, dtype=np.uint8))
        lu = self.obs.buffers["list_uid"] if list_uid is None else cp.ascontiguousarray(
            cp.asarray(list_uid, dtype=cp.int32))
        b = self.obs.buffers
        self.bp.obs_dec_check(
            int(self.N), int(self.n_rows), int(self.S), int(self.decisions.data.ptr),
            int(self.due.data.ptr), int(lu.data.ptr), int(self.obs.n_entries),
            int(b["mask_feet"].data.ptr), int(b["mask_aim"].data.ptr),
            int(b["mask_trigger"].data.ptr), int(b["mask_hand"].data.ptr), int(self.obs.n_kinds),
            int(self.table.data.ptr), int(self.state.data.ptr), int(self.masked.data.ptr),
            int(self.faults.data.ptr))
        self._lu_alive = lu                  # alive until the launch ran

    # -- the tick head ---------------------------------------------------------
    def decode(self) -> None:
        """The decode of every decoder-driven row of every env (the
        conductor's tick head): the latch columns written into the store, the
        slot state, ``DeviceObservation.aimed``; the fault words deferred
        into the conductor's one books read."""
        c, o = self.c, self.obs
        pic = o.picture
        self.bp.obs_decode(
            int(self.N), int(c.store.device_ptr), int(c.U), int(c.store.shape[2]),
            c.store.tables, int(o.rowmap.data.ptr), int(self.n_rows), int(self.S), int(self.T),
            int(o.table.data.ptr), int(o.table.shape[1]), int(c.trow.ptr),
            int(o.ended.data.ptr), int(pic.state.data.ptr), int(pic.W),
            int(self.table.data.ptr), int(self.drive.data.ptr), int(self.state.data.ptr),
            int(o.aimed.data.ptr), int(self.faults.data.ptr), int(self.brk))
        if not self.horde_late:
            self.horde_head()
        if self.snapping:
            self.snap = c._rows.copy()
        self.decodes += 1

    def horde_head(self) -> None:
        """The zombie controller's latch on the card (every living zombie at
        this point of the tick) into :attr:`horde_rows`."""
        if not self.horde_envs:
            return
        c = self.c
        self.bp.obs_dec_horde(int(self.N), int(c.store.device_ptr), int(c.U),
                              int(c.store.shape[2]), int(self.horde_mode.data.ptr),
                              int(self.horde_rows.data.ptr), int(self.horde_rows.shape[1]))

    def horde_latch_into(self, cache) -> None:
        """Slot 5: the card's horde rows into the HORDE stage's latch buffer
        (device to device, the envs the card serves)."""
        if self.horde_all:
            cache.latch[...] = self.horde_rows
        else:
            cache.latch[self._horde_idx] = self.horde_rows[self._horde_idx]

    # -- the observation's columns ---------------------------------------------
    def build_obs(self, store_ptr=None) -> None:
        """The vector's ``prev`` block and the list's ``prev_aim`` /
        ``on_next_tick`` for every env on its decision tick (after the list
        and the windows)."""
        build_obs(self.obs, self.state, store_ptr)

    # -- the books read --------------------------------------------------------
    def check_words(self, words) -> None:
        """The conductor's books read of the fault words ``(N, FAULT_W)``:
        an out-of-range decision raises the host's ``ValueError`` by name."""
        got = check_message(np.asarray(words, dtype=np.int64).reshape(self.N, FAULT_W),
                            self.sizes)
        if got is not None:
            e, msg = got
            ex = ValueError(msg)
            ex.env = e
            raise ex

    def host_state(self) -> np.ndarray:
        """THE GATE'S DOWNLOAD of the state records ``(N, rows, S_W)``."""
        return self.state.get()


def build_obs(o, state, store_ptr=None) -> None:
    """``bp.obs_dec_obs`` for the observation ``o`` over the decoder state
    ``state`` ``(N, rows, S_W)`` (``DeviceDecoder.state``, or a zero state
    when no decoder is attached)."""
    c = o.c
    b = o.buffers
    store = int(c.store.device_ptr) if store_ptr is None else int(store_ptr)
    o.bp.obs_dec_obs(
        int(o.N), store, int(c.U), int(c.store.shape[2]), c.store.tables,
        int(o.rowmap.data.ptr), int(o.n_rows), int(o.S), int(o.T),
        int(o.table.data.ptr), int(o.table.shape[1]), int(c.trow.ptr),
        int(o.params.data.ptr), int(o.ended.data.ptr), int(o.picture.state.data.ptr),
        int(o.picture.W), int(state.data.ptr), int(o.aimed.data.ptr),
        int(b["list_mask"].data.ptr), int(b["list_uid"].data.ptr), int(o.n_entries),
        int(b["list"].data.ptr), int(b["vector"].data.ptr))


class NullDecoderView:
    """The decoder's columns of an observation with no decoder attached (the
    tap-driven corpora): no slot ever decoded (``prev`` -1), the aimed entry
    from ``DeviceObservation.aimed``."""

    def __init__(self, o):
        import cupy as cp
        self.o = o
        self.state = cp.zeros((o.N, o.n_rows, len(STATE_FIELDS)), dtype=cp.int64)

    def build_obs(self, store_ptr=None) -> None:
        build_obs(self.o, self.state, store_ptr)


def aim_vector_host(dx, dy, divide: bool = False):
    """``action_spec_v1.aim_vector``'s arithmetic over integer offset arrays
    (numpy float64, IEEE, the same operations element for element): ``(qx,
    qy, none)``; ``divide`` the break's ``dx / mag``."""
    dx = np.asarray(dx, dtype=np.int64)
    dy = np.asarray(dy, dtype=np.int64)
    fx, fy = dx.astype(np.float64), dy.astype(np.float64)
    mag = np.sqrt(fx * fx + fy * fy)
    none = mag <= 0.0
    m = np.where(none, 1.0, mag)
    if divide:
        ux, uy = fx / m, fy / m
    else:
        inv = 1.0 / m
        ux, uy = fx * inv, fy * inv

    def q(u):
        v = u * 65536.0
        return np.where(v >= 0.0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int64)
    return np.where(none, 0, q(ux)), np.where(none, 0, q(uy)), none


__all__ = ["DeviceDecoder", "DecoderRefused", "TAB_FIELDS", "STATE_FIELDS", "AIM_KINDS",
           "FAULT_W", "BREAK_CODES", "constants", "compass_init", "table_values",
           "check_message", "state_form", "host_state_form", "aim_vector_host",
           "HORDE_CONDUCTOR", "HORDE_DEVICE", "HORDE_IDLE", "build_obs",
           "NullDecoderView"]
