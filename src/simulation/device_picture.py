"""THE DEVICE TEAM PICTURE (#29 P8.3, residency design §4.4c "P8 RE-CUT,
VERSION 2"; inventory ``docs/residency_p8_observation_inventory_2026-11-20.md``
part C): every piece of state ``simulation.team_picture.TeamPicture`` keeps,
per env, on the card, updated ONCE per tick as the last stage of the device
conductor's walk (the ``11'`` call line, after who sees whom and before the
observation's blocks: the place the reference's slot 11 runs, after the tick
advance, the order expiry and the conversion), from the unit store, the
device vision (``DeviceObservation.visible``), the tick's event ring and the
grenade / charge object tables. The host picture stays THE reference; this
is its twin (never a second picture, never a second update site).

CLASS T (training only): nothing here is synced state, no digest reads it,
and its device code writes no synced buffer.

THE STATE (:func:`layout`): ONE ``(N, PIC_W)`` int64 block per env whose
sections are, in order (T the layout's marine teams, U the store's rows):

- ``REC`` ``(T, U, REC_W)``: team t's record of the unit on store row r
  (``_records[t][uid]``; at most one record per unit per team, so the
  capacity U is the law's and can never overflow); the loadout as its table
  row (``NO_ROW`` for none, the store's own encoding of ``""``);
- ``UNIT`` ``(U, UNIT_W)``: per row the bar (``bars``, as its Q16 ``hp_q``:
  ``float(current_hp)`` is exactly ``hp_q / 65536``), the turned flag
  (``_turned``), ``death_tick``, ``motion`` (prev / now Q16 positions) and
  ``last_hit`` (tick, direction, the cause's index);
- ``DMG`` ``(T, DMG_CAP, DMG_W)`` + ``DMG_HD`` ``(T, 2)``: team t's damage
  window as ONE FIFO ring of (victim uid, tick, damage Q16) in append order
  (head, length). The host keeps one deque per victim; every entry is
  appended at the current tick and dropped by age, so the FIFO's prefix is
  exactly what every deque drops, and the deque of victim v is the FIFO's
  subsequence of v, in order (its sum is taken in that order:
  ``cuda_obs_picture.cuh``'s ``damage_sum``). A FIFO past ``DMG_CAP`` is a
  counted fault (refused at the books read), never a silent drop;
- ``OBJ`` ``(T, OCAP, 2)`` + ``OBJ_N`` / ``OBJ_NEXT`` ``(T,)``: team t's object
  tracker, (object serial, team object id) per live object it has seen, and
  its id counter (``_obj_next``); ``SEEN`` ``(T, OCAP, 2)`` + ``SEEN_N``:
  ``seen_now[t]`` as (serial, id) in the reference's order (grenades, then
  charges, each in table order). OCAP = the two tables' capacities: the
  tracker holds only LIVE objects, so it cannot overflow either;
- ``SERIAL_NEXT``, ``SIDE_G`` ``(GRENADE_CAP,)`` / ``KNOWN_G``, ``SIDE_C``
  ``(CHARGE_CAP,)`` / ``KNOWN_C``: THE STABLE OBJECT SERIAL. The host tracker
  is keyed by Python object identity; the device tables are compacted
  (``prune_grenades``, the charges' prune), so a row index is no identity.
  Every row of the two tables carries a serial in a SIDE ARRAY of this block
  (never a table column: no digest reads it), which the tables' own stable
  compaction carries with the rows (``ObjectTableBuffer.set_side``:
  ``k_table_compact`` moves side[i] with row i and recounts KNOWN, the rows
  that already hold a serial). Rows are only ever appended at the end, so
  rows ``KNOWN .. count`` are the new ones: the update numbers them from
  ``SERIAL_NEXT`` in row order, grenades first. On the host the matching
  notion is a TEST-SIDE mapping only: an object's position in
  ``sim.projectiles + sim.planted_charges`` (:func:`host_form`), which the
  host game never sees;
- ``PARAM`` (``memory_ticks``, ``window_ticks`` and the env's team count, read
  from the reference's own picture object) and ``TICK`` (the tick the
  picture is at; -1 before its first update).

THE FAULTS (``(N,)`` int64 bits, :data:`FAULT_BITS`): a skipped tick, a
damage window past its capacity, a hit on a uid the store does not hold, a
line past the device sight's range, ... ride the conductor's ONE books read
(kind ``picture_faults``) and are raised there by name with the env.

THE VECTOR: the 13 columns the column-to-patch table gives P8.3 (``hp``,
``vel_x`` / ``vel_y``, the five ``hit_*``, the four record counts), built by
``bp.obs_picture_vector`` beside P8.1's block at the decision cadence.

THE SEAM FOR P8.9b: :meth:`DevicePicture.restage` sets env e's block from a
host picture (equal by construction); the build takes either the device's
own first update (tick 0, no events) or that restage.
"""
from __future__ import annotations

import numpy as np

#: A record's fields (``team_picture.Record``; PRESENT: the record exists).
REC_FIELDS = ("PRESENT", "UID", "TEAM", "ZOMBIE", "LOADOUT", "TILE_X", "TILE_Y",
              "CENTRE_X", "CENTRE_Y", "POS_X", "POS_Y", "VEL_X", "VEL_Y", "TICK", "FACING")
#: A unit's trackers.
UNIT_FIELDS = ("BAR_SET", "BAR_Q", "TURNED", "DEATH_SET", "DEATH_TICK", "MOTION_SET",
               "PREV_X", "PREV_Y", "NOW_X", "NOW_Y", "HIT_SET", "HIT_TICK", "HIT_DX",
               "HIT_DY", "HIT_CAUSE")
#: A damage window entry.
DMG_FIELDS = ("VICTIM", "TICK", "DQ")
#: The per-env parameters (from the reference's picture object).
PARAM_FIELDS = ("MEMORY_TICKS", "WINDOW_TICKS", "N_TEAMS")
#: The damage window's capacity per (env, team): the hits a team dealt or
#: saw within one window. No law bounds it (several attackers, damage over
#: time); past it the update faults by name (a refusal, never a clamp). The
#: corpus's peak is printed by the check.
DMG_CAP = 1024
#: The fault bits (the books read names each).
FAULT_BITS = (
    ("UNKNOWN_UID", "a hit names a unit id the store does not hold (last_hit cannot keep it)"),
    ("DMG_OVERFLOW", "a team's damage window passed its capacity"),
    ("OBJ_OVERFLOW", "a team's object tracker passed its capacity"),
    ("SKIPPED", "the picture was not at the previous tick (a skipped or doubled update)"),
    ("SIGHT", "an object sight line left the device sight's range"),
    ("SERIAL", "an object table holds fewer rows than carry a serial"),
    ("TEAMS", "an env's picture has more teams than the layout"),
)
#: THE BREAKS the check switches (``brk``): 1 a record kept one tick longer
#: (``>=`` for ``>``), 2 the damage window fed the tick's hits in reverse
#: emission order, 3 the object serial replaced by the row index.
BREAK_CODES = {"": 0, "lifetime": 1, "dmg_order": 2, "serial_row": 3}


def _caps() -> tuple:
    from simulation import unit_schema as us
    from simulation.object_tables import table_layout
    return (int(table_layout(us.GRENADE_TABLE).capacity),
            int(table_layout(us.CHARGE_TABLE).capacity))


def layout(n_teams: int, n_units: int) -> list:
    """``[(section, offset, shape)]`` of one env's block, THE layout the
    kernels compute (``cuda_obs_picture.cuh``'s ``PicView``; the build checks
    the compiled one against it)."""
    T, U = int(n_teams), int(n_units)
    g, c = _caps()
    oc = g + c
    secs = [("REC", (T, U, len(REC_FIELDS))), ("UNIT", (U, len(UNIT_FIELDS))),
            ("DMG", (T, DMG_CAP, len(DMG_FIELDS))), ("DMG_HD", (T, 2)),
            ("OBJ", (T, oc, 2)), ("OBJ_N", (T,)), ("OBJ_NEXT", (T,)),
            ("SEEN", (T, oc, 2)), ("SEEN_N", (T,)), ("SERIAL_NEXT", (1,)),
            ("SIDE_G", (g,)), ("KNOWN_G", (1,)), ("SIDE_C", (c,)), ("KNOWN_C", (1,)),
            ("PARAM", (len(PARAM_FIELDS),)), ("TICK", (1,))]
    out, off = [], 0
    for name, shape in secs:
        out.append((name, off, shape))
        off += int(np.prod(shape))
    return out


def width(n_teams: int, n_units: int) -> int:
    name, off, shape = layout(n_teams, n_units)[-1]
    return off + int(np.prod(shape))


def constants() -> list:
    """The picture's generated constants (``obs_layout.h``, through
    ``device_obs._constants``)."""
    from simulation.team_picture import OBJECT_UID_BASE
    g, c = _caps()
    out = [(f"OBS_PIC_REC_{f}", i) for i, f in enumerate(REC_FIELDS)]
    out.append(("OBS_PIC_REC_W", len(REC_FIELDS)))
    out += [(f"OBS_PIC_UNIT_{f}", i) for i, f in enumerate(UNIT_FIELDS)]
    out.append(("OBS_PIC_UNIT_W", len(UNIT_FIELDS)))
    out += [(f"OBS_PIC_DMG_{f}", i) for i, f in enumerate(DMG_FIELDS)]
    out += [("OBS_PIC_DMG_W", len(DMG_FIELDS)), ("OBS_PIC_DMG_CAP", DMG_CAP)]
    out += [(f"OBS_PIC_PARAM_{f}", i) for i, f in enumerate(PARAM_FIELDS)]
    out.append(("OBS_PIC_PARAM_W", len(PARAM_FIELDS)))
    out += [(f"OBS_PIC_FAULT_{n}", 1 << i) for i, (n, _) in enumerate(FAULT_BITS)]
    out += [("OBS_PIC_GRENADE_CAP", g), ("OBS_PIC_CHARGE_CAP", c),
            ("OBS_PIC_OBJ_CAP", g + c), ("OBS_OBJECT_UID_BASE", int(OBJECT_UID_BASE))]
    out += [(f"OBS_PIC_BREAK_{k.upper()}", v) for k, v in BREAK_CODES.items() if k]
    return out


class PictureRefused(RuntimeError):
    """A batch the device picture does not serve (by name)."""


def fault_names(bits: int) -> list:
    return [n for i, (n, _) in enumerate(FAULT_BITS) if int(bits) & (1 << i)]


def fault_text(bits: int) -> str:
    return "; ".join(f"{n}: {why}" for i, (n, why) in enumerate(FAULT_BITS)
                     if int(bits) & (1 << i))


# ---------------------------------------------------------------------------
# THE NORMAL FORM: the picture's whole state as plain Python values, keyed by
# unit id and by object POSITION, so the host picture and the device block
# compare field by field (the gate) and the host can be restaged (P8.9b).
# ---------------------------------------------------------------------------
def _loadout_row(name) -> int:
    from simulation import unit_schema as us
    if not name:
        return int(us.NO_ROW)
    return int(us.table_row("loadouts", name))


def _cause_index(cause: str) -> int:
    from simulation.events import CAUSE_NAMES
    return int(list(CAUSE_NAMES).index(cause))


def host_form(picture, sim) -> dict:
    """The REFERENCE picture's normal form. Objects are named by their
    position in ``sim.projectiles + sim.planted_charges`` (THE test-side
    mapping of the device's serial; the host game holds no serial)."""
    pos = {id(o): i for i, o in enumerate(list(sim.projectiles) + list(sim.planted_charges))}
    recs = {}
    for t in picture.teams:
        recs[t] = {int(uid): (int(r.team), int(bool(r.zombie)), _loadout_row(r.loadout),
                              int(r.tile[0]), int(r.tile[1]), int(r.centre[0]),
                              int(r.centre[1]), int(r.pos_q[0]), int(r.pos_q[1]),
                              int(r.vel_q[0]), int(r.vel_q[1]), int(r.tick), int(r.facing_q))
                   for uid, r in picture._records[t].items()}
    bars = {}
    for uid, b in picture.bars.items():
        q = round(float(b) * 65536)
        if q / 65536 != float(b):
            raise PictureRefused(f"device picture: unit {uid}'s bar {b!r} is not on the Q16 grid")
        bars[int(uid)] = int(q)
    dmg = {}
    for t in picture.teams:
        dmg[t] = {int(uid): [(int(tk), _dq(d)) for tk, d in q]
                  for uid, q in picture._damage[t].items() if q}
    objs, seen = {}, {}
    for t in picture.teams:
        objs[t] = {pos[id(o)]: int(v) for o, v in picture._objects[t].items()}
        seen[t] = [(pos[id(o)], int(v)) for o, v in picture.seen_now[t]]
    return {
        "tick": None if picture.tick is None else int(picture.tick),
        "records": recs,
        "bars": bars,
        "turned": sorted(int(u) for u in picture._turned),
        "death_tick": {int(k): int(v) for k, v in picture.death_tick.items()},
        "motion": {int(k): (int(a[0]), int(a[1]), int(b[0]), int(b[1]))
                   for k, (a, b) in picture.motion.items()},
        "last_hit": {int(k): (int(v[0]), int(v[1]), int(v[2]), _cause_index(v[3]))
                     for k, v in picture.last_hit.items()},
        "damage": dmg,
        "objects": objs,
        "obj_next": {int(t): int(n) for t, n in picture._obj_next.items()},
        "seen_now": seen,
        "n_objects": len(pos),
        "n_grenades": len(sim.projectiles),
    }


def _dq(d) -> int:
    q = round(float(d) * 65536)
    if q / 65536 != float(d):
        raise PictureRefused(f"device picture: a damage {d!r} is not on the Q16 grid")
    return int(q)


def device_form(block: np.ndarray, uids, n_teams_env: int, T: int, U: int,
                g_count: int, c_count: int, key_is_row: bool = False) -> dict:
    """One env's DEVICE block (``(PIC_W,)`` int64, downloaded) as the normal
    form: rows named by the store's unit ids (``uids``, row order), objects
    by their position (grenade rows, then charge rows: the host lists'
    order) through the side arrays' serials (``key_is_row``: the tracker
    keyed by the row itself, THE BREAK ``serial_row``'s reading)."""
    sec = {n: block[o:o + int(np.prod(s))].reshape(s) for n, o, s in layout(T, U)}
    R = {f: i for i, f in enumerate(REC_FIELDS)}
    Q = {f: i for i, f in enumerate(UNIT_FIELDS)}
    n = len(uids)
    recs = {}
    for t in range(n_teams_env):
        recs[t] = {}
        for r in range(n):
            x = sec["REC"][t, r]
            if not x[R["PRESENT"]]:
                continue
            recs[t][int(x[R["UID"]])] = tuple(int(x[R[f]]) for f in REC_FIELDS[2:])
    bars, turned, death, motion, hit = {}, [], {}, {}, {}
    for r in range(n):
        x = sec["UNIT"][r]
        uid = int(uids[r])
        if x[Q["BAR_SET"]]:
            bars[uid] = int(x[Q["BAR_Q"]])
        if x[Q["TURNED"]]:
            turned.append(uid)
        if x[Q["DEATH_SET"]]:
            death[uid] = int(x[Q["DEATH_TICK"]])
        if x[Q["MOTION_SET"]]:
            motion[uid] = tuple(int(x[Q[f]]) for f in ("PREV_X", "PREV_Y", "NOW_X", "NOW_Y"))
        if x[Q["HIT_SET"]]:
            hit[uid] = tuple(int(x[Q[f]]) for f in ("HIT_TICK", "HIT_DX", "HIT_DY", "HIT_CAUSE"))
    dmg = {}
    for t in range(n_teams_env):
        head, ln = (int(v) for v in sec["DMG_HD"][t])
        d: dict = {}
        for k in range(ln):
            v, tk, dq = (int(z) for z in sec["DMG"][t, (head + k) % DMG_CAP])
            d.setdefault(v, []).append((tk, dq))
        dmg[t] = d
    g, _c = _caps()
    serial_pos = {}
    for i in range(g_count):
        serial_pos[i if key_is_row else int(sec["SIDE_G"][i])] = i
    for j in range(c_count):
        serial_pos[(g + j) if key_is_row else int(sec["SIDE_C"][j])] = g_count + j
    objs, seen, nxt = {}, {}, {}
    for t in range(n_teams_env):
        objs[t] = {}
        for k in range(int(sec["OBJ_N"][t])):
            s, v = (int(z) for z in sec["OBJ"][t, k])
            objs[t][serial_pos.get(s, ("serial", s))] = v
        seen[t] = [(serial_pos.get(int(s), ("serial", int(s))), int(v))
                   for s, v in sec["SEEN"][t, :int(sec["SEEN_N"][t])]]
        nxt[t] = int(sec["OBJ_NEXT"][t])
    tick = int(sec["TICK"][0])
    return {"tick": None if tick < 0 else tick, "records": recs, "bars": bars,
            "turned": sorted(turned), "death_tick": death, "motion": motion,
            "last_hit": hit, "damage": dmg, "objects": objs, "obj_next": nxt,
            "seen_now": seen, "n_objects": g_count + c_count, "n_grenades": g_count}


def first_difference(want: dict, got: dict):
    """The first field where two normal forms differ, as ``(path, reference,
    device)``, or None."""
    def walk(path, a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b), key=str):
                if k not in a:
                    return (path + (k,), "<absent>", b[k])
                if k not in b:
                    return (path + (k,), a[k], "<absent>")
                d = walk(path + (k,), a[k], b[k])
                if d is not None:
                    return d
            return None
        if a != b:
            return (path, a, b)
        return None
    return walk((), want, got)


def block_from_form(form: dict, uids, T: int, U: int, params, where: str = "") -> np.ndarray:
    """ONE env's device block (``(PIC_W,)`` int64) from a normal form
    (:func:`host_form`): THE conversion the restage (P8.9b's seam) uploads;
    :func:`device_form` of it is the form again. ``uids``: the store's rows'
    unit ids (row order); ``params``: the env's PARAM row. Objects get the
    serial = their position (the side arrays renumbered 0 .. n-1)."""
    hf = form
    row = {u: r for r, u in enumerate(uids)}
    blk = np.zeros((width(T, U),), dtype=np.int64)
    sec = {n: blk[o:o + int(np.prod(s))].reshape(s) for n, o, s in layout(T, U)}
    R = {f: i for i, f in enumerate(REC_FIELDS)}
    Q = {f: i for i, f in enumerate(UNIT_FIELDS)}
    for t, recs in hf["records"].items():
        for uid, vals in recs.items():
            x = sec["REC"][t, row[uid]]
            x[R["PRESENT"]], x[R["UID"]] = 1, uid
            x[2:] = vals
    for uid, b in hf["bars"].items():
        sec["UNIT"][row[uid], Q["BAR_SET"]], sec["UNIT"][row[uid], Q["BAR_Q"]] = 1, b
    for uid in hf["turned"]:
        sec["UNIT"][row[uid], Q["TURNED"]] = 1
    for uid, tk in hf["death_tick"].items():
        sec["UNIT"][row[uid], Q["DEATH_SET"]], sec["UNIT"][row[uid], Q["DEATH_TICK"]] = 1, tk
    for uid, m in hf["motion"].items():
        x = sec["UNIT"][row[uid]]
        x[Q["MOTION_SET"]] = 1
        x[[Q["PREV_X"], Q["PREV_Y"], Q["NOW_X"], Q["NOW_Y"]]] = m
    for uid, h in hf["last_hit"].items():
        if uid not in row:
            raise PictureRefused(f"device picture: {where}: a last hit on unit "
                                 f"{uid}, which the store does not hold")
        x = sec["UNIT"][row[uid]]
        x[Q["HIT_SET"]] = 1
        x[[Q["HIT_TICK"], Q["HIT_DX"], Q["HIT_DY"], Q["HIT_CAUSE"]]] = h
    for t, d in hf["damage"].items():
        # any interleaving of the victims' deques that keeps each deque's
        # order and the ticks sorted is the same window (module doc)
        ent = sorted(((tk, k, uid, dq) for uid, q in d.items()
                      for k, (tk, dq) in enumerate(q)), key=lambda z: (z[0], z[2], z[1]))
        if len(ent) > DMG_CAP:
            raise PictureRefused(f"device picture: {where}: team {t}'s damage "
                                 f"window holds {len(ent)} entries, the capacity {DMG_CAP}")
        for k, (tk, _k, uid, dq) in enumerate(ent):
            sec["DMG"][t, k] = (uid, tk, dq)
        sec["DMG_HD"][t] = (0, len(ent))
    g_cnt = int(hf["n_grenades"])
    c_cnt = int(hf["n_objects"]) - g_cnt
    sec["SIDE_G"][:g_cnt] = np.arange(g_cnt)
    sec["SIDE_C"][:c_cnt] = g_cnt + np.arange(c_cnt)
    sec["KNOWN_G"][0], sec["KNOWN_C"][0] = g_cnt, c_cnt
    sec["SERIAL_NEXT"][0] = g_cnt + c_cnt
    for t, objs in hf["objects"].items():
        for k, (p, v) in enumerate(sorted(objs.items())):
            sec["OBJ"][t, k] = (p, v)
        sec["OBJ_N"][t] = len(objs)
        sec["OBJ_NEXT"][t] = hf["obj_next"][t]
        for k, (p, v) in enumerate(hf["seen_now"][t]):
            sec["SEEN"][t, k] = (p, v)
        sec["SEEN_N"][t] = len(hf["seen_now"][t])
    sec["PARAM"][:] = params
    sec["TICK"][0] = -1 if hf["tick"] is None else hf["tick"]
    return blk


# ---------------------------------------------------------------------------
# The device picture
# ---------------------------------------------------------------------------
class DevicePicture:
    """The team picture of a device observation's N envs (see the module
    docstring). Built by ``DeviceObservation``; ``start`` takes each env's
    first state (the device's own first update, or a restage from the host
    picture)."""

    def __init__(self, obs):
        import cupy as cp
        c = obs.c
        self.obs, self.c, self.cp, self.bp = obs, c, cp, c.bp
        self.N, self.T, self.U = int(obs.N), int(obs.T), int(c.U)
        self.W = width(self.T, self.U)
        self.layout = {n: (o, s) for n, o, s in layout(self.T, self.U)}
        compiled = dict(self.bp.obs_picture_layout(self.T, self.U))
        mine = {n: int(o) for n, (o, _s) in self.layout.items()}
        mine["W"] = self.W
        if compiled != mine:
            raise PictureRefused(f"device picture: the compiled block layout {compiled} is not "
                                 f"device_picture.layout's {mine} (rebuild cpp/build_cuda)")
        params = np.zeros((self.N, len(PARAM_FIELDS)), dtype=np.int64)
        for e, sim in enumerate(c.sims):
            pic = getattr(sim, "team_picture", None)
            if pic is None:
                raise PictureRefused(f"device picture (env {e}): the sim has no team picture "
                                     f"(Simulation.start_team_picture)")
            want = tuple(range(obs.env_teams[e]))
            if tuple(pic.teams) != want:
                raise PictureRefused(f"device picture (env {e}): the picture's teams "
                                     f"{tuple(pic.teams)} are not the match's {want}")
            if len(want) > self.T:
                raise PictureRefused(f"device picture (env {e}): {len(want)} teams, the layout "
                                     f"{self.T}")
            params[e] = (int(pic.memory_ticks), int(pic.window_ticks), len(want))
        self.params_host = params
        block = np.zeros((self.N, self.W), dtype=np.int64)
        o = self.layout["PARAM"][0]
        block[:, o:o + len(PARAM_FIELDS)] = params
        block[:, self.layout["TICK"][0]] = -1
        self.state = cp.asarray(block)
        self.faults = cp.zeros((self.N,), dtype=cp.int64)
        self.brk = 0                          # THE BREAKS (BREAK_CODES), the check's
        self.updates = 0
        # the side arrays: the tables' compaction carries them with the rows
        base = int(self.state.data.ptr)
        for table, side, known in ((c.grenades, "SIDE_G", "KNOWN_G"),
                                   (c.charges, "SIDE_C", "KNOWN_C")):
            table.buffer.set_side(base + 8 * self.layout[side][0], self.W,
                                  base + 8 * self.layout[known][0], self.W)

    def section(self, name: str):
        """A section of every env's block as a CuPy view ``(N, *shape)``."""
        o, s = self.layout[name]
        n = int(np.prod(s))
        return self.state[:, o:o + n].reshape((self.N,) + tuple(s))

    # -- the update ------------------------------------------------------
    def update(self, first: bool = False, check: bool = True) -> None:
        """THE stage (the ``11'`` call line): every env's picture brought to
        the tick, on the tick stream. Inside a conductor tick the fault bits
        ride the books read; outside (the build's first update) they are
        read here (a sync) and raised by name (``check``: False only for the
        bench, which times repeated launches)."""
        o, c = self.obs, self.c
        deferred = self.bp.obs_picture(
            int(self.N), int(c.store.device_ptr), int(self.U), int(c.store.shape[2]),
            c.ring.buffer, c.grenades.buffer, c.charges.buffer,
            int(c.fields["solid"].data.ptr), int(c.fields["sight_screen"].data.ptr),
            int(c.h), int(c.w), int(c.trow.ptr), int(o.visible.data.ptr), int(self.T),
            int(self.state.data.ptr), int(self.W), int(self.faults.data.ptr), bool(first),
            int(self.brk))
        self.updates += 1
        if not deferred and check:
            self.check_faults()

    def check_faults(self) -> None:
        f = self.faults.get()
        if f.any():
            e = int(np.flatnonzero(f)[0])
            raise PictureFault(f"device picture (env {e}, tick {int(self.c.sims[e].tick)}): "
                               f"{fault_text(int(f[e]))}", env=e, bits=int(f[e]))

    def start(self, from_host=None) -> None:
        """Every env's first state: the device's own first update (the
        reference's ``start_team_picture`` update: no events) for each env
        not in ``from_host``; a restage from its host picture for each env
        in it (P8.9b's seam; the check's hand matches, whose hand state comes
        after the reference's first update)."""
        from_host = set(from_host or ())
        self.update(first=True)
        for e in sorted(from_host):
            self.restage(e)

    def restage(self, e: int, picture=None, sim=None, form=None) -> None:
        """Set env ``e``'s block from a HOST picture (default the env's own
        twin's; or its normal form ``form``, :func:`host_form`): equal to it
        by construction. The object tables must hold the host lists' rows
        (the env's tables restaged from the same sim); their side serials are
        renumbered 0 .. n-1 in row order."""
        e = int(e)
        if form is None:
            sim = self.c.sims[e] if sim is None else sim
            picture = sim.team_picture if picture is None else picture
            form = host_form(picture, sim)
        blk = block_from_form(form, self.c.uids[e], self.T, self.U, self.params_host[e],
                              where=f"restage of env {e}")
        self.state[e] = self.cp.asarray(blk)

    def reset_env(self, e: int, block, params) -> None:
        """#29 P8.9b: env ``e``'s next episode -- its PARAM row (the new
        picture's memory / window ticks and team count) and its whole block
        (``block_from_form`` of the new sim's tick-0 picture, made by the
        observation's ``reset_prepare``); every other env untouched."""
        e = int(e)
        self.params_host[e] = np.asarray(params, dtype=np.int64)
        self.state[e] = self.cp.asarray(np.asarray(block, dtype=np.int64))

    # -- readers -------------------------------------------------------------
    def damage_sums(self):
        """THE damage window's sums on the card (``damage_last_s`` for every
        (team, store row): ``(N, T, U)`` float64 CuPy), in the deque's order
        through the float kit (the gate's; P8.4b's list reads the same
        device function)."""
        out = self.cp.zeros((self.N, self.T, self.U), dtype=self.cp.float64)
        self.bp.obs_picture_damage_sums(int(self.N), int(self.c.store.device_ptr), int(self.U),
                                        int(self.c.store.shape[2]), int(self.T),
                                        int(self.state.data.ptr), int(self.W),
                                        int(out.data.ptr), False)
        return out

    def forms(self) -> list:
        """THE GATE'S DOWNLOAD (a sync): every env's normal form."""
        blk = self.state.get()
        _gr, gcnt, _ = self.c.grenades.download()
        _cr, ccnt, _ = self.c.charges.download()
        # THE BREAK serial_row keys the tracker by row: read it so, so the
        # break shows exactly where a row index is no identity (a compaction)
        row = self.brk == BREAK_CODES["serial_row"]
        return [device_form(blk[e], self.c.uids[e], self.obs.env_teams[e], self.T, self.U,
                            int(gcnt[e]), int(ccnt[e]), key_is_row=row) for e in range(self.N)]


class PictureFault(RuntimeError):
    """A picture update whose operands left its proved range or whose
    record list passed its capacity (named, with the env)."""

    def __init__(self, msg, env=0, bits=0):
        super().__init__(msg)
        self.env = int(env)
        self.bits = int(bits)


__all__ = ["REC_FIELDS", "UNIT_FIELDS", "DMG_FIELDS", "PARAM_FIELDS", "DMG_CAP", "FAULT_BITS",
           "BREAK_CODES", "layout", "width", "constants", "PictureRefused", "PictureFault",
           "fault_names", "fault_text", "host_form", "device_form", "block_from_form",
           "first_difference",
           "DevicePicture"]
