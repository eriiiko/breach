"""``control_bots_door`` -- the bots through the door (#195 1j).

THE contract is ``docs/rl_agent_rung1_note_1j_2026-11-13.md``. The layered
bots (``control_bots.BotTeam``, unchanged) decide; their verbs are CAPTURED
on a proxy (:class:`IntentCapture`) and never reach the sim; at the env's
decision ticks the controller observes and masks the team's rows through
``rl_env.observe_v1`` (the env's own builders), translates each bot's
decision into the five integers of action spec v1 -- the nearest LEGAL
choice, through the 1h inverse helpers, never a table of its own
(:func:`translate`) -- and hands them to an inner ``AgentPolicy``'s v1
decoder, which is then the ONLY thing that issues the team's soldier verbs.
The team writes its own brain's orders through the commander's two lifted
functions (``control_bots.team_order`` / ``issue_team_order``); it is not
in ``control_source.DOOR_CONTROLLERS``.

A variant is a :data:`DOOR_VARIANTS` row (design 2.7: each coarseness a
measurement). With a :class:`RecordSink` installed in the process, every
door team of the match writes one ``.npz`` IMITATION RECORD (the
observation, masks, the five integers, the losses and the fireteam rewards,
in the env's served dtypes, tied to the match by its ``meta``) beside the
match's replay recording; without one, nothing is recorded and no reward
book runs.

Headless and pyray-free; draws nothing; writes the sim only through the
decoder's verbs and the order verb.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, replace
from math import isqrt
from pathlib import Path

import numpy as np

from control_source import ControlSource

#: The record files' format; :func:`read_record` refuses any other.
RECORD_FORMAT = 1

#: The verbs the bots may call, captured (the policy's eleven).
def _policy_kinds() -> tuple:
    from control_policy import POLICY_KINDS
    return POLICY_KINDS


#: The intent entry points a bot never calls: refused by name on a capture.
REFUSED_ON_CAPTURE = ("order_intent", "drive_zombies", "apply_action")
#: The hold verbs (the last call of a tick wins) and the latches.
HOLD_KINDS = ("set_move_dir", "clear_move_dir", "set_aim", "set_trigger",
              "set_flashlight")
#: latch kind -> the hand option it becomes (throws and plants apart).
_LATCH_HAND = {"reload_intent": 1, "melee_intent": 2, "shove_intent": 3,
               "use_intent": 4}

#: The loss bits of one slot's decision (note section 3's table), in bit
#: order; per-match totals carry the same names.
LOSS_BITS = ("feet_dir", "throttle", "aim_missing", "aim_rounded",
             "aim_ambiguous", "carried", "superseded", "hand_masked",
             "throw_rounded", "trigger_throw")
_BIT = {n: 1 << i for i, n in enumerate(LOSS_BITS)}


class DoorRefused(RuntimeError):
    """The door would send something the game refuses (an index out of
    range or a masked choice), or a bot called a verb no soldier has."""


# ---------------------------------------------------------------------------
# The interception
# ---------------------------------------------------------------------------
_SIGS = {}


def _signature(kind):
    sig = _SIGS.get(kind)
    if sig is None:
        from simulation.simulation import Simulation
        sig = _SIGS[kind] = inspect.signature(getattr(Simulation, kind))
    return sig


class IntentCapture:
    """A proxy handed to a controller in place of the sim (note section 1).
    The policy verbs (``control_policy.POLICY_KINDS``) APPEND ``(tick, kind,
    args)`` to :attr:`calls` -- ``args`` bound to the entry point's own
    signature, defaults filled, normalised as the sim's tap normalises them
    (ints, bools, the throttle and the plant timer clamped) -- and, only with
    ``passthrough=True``, are re-issued to the real sim at once with the
    caller's own arguments. ``order_intent``, ``drive_zombies`` and
    ``apply_action`` raise by name. Every other attribute read AND write is
    forwarded to the real sim (``vision.state_for`` caches its one sight
    state on the sim: a write must land there)."""

    _OWN = ("_sim", "_pass", "calls")

    def __init__(self, sim, passthrough: bool = False):
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_pass", bool(passthrough))
        object.__setattr__(self, "calls", [])

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_sim"), name)

    def __setattr__(self, name, value):
        if name in IntentCapture._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_sim"), name, value)

    def _capture(self, kind, args, kwargs):
        sim = object.__getattribute__(self, "_sim")
        bound = _signature(kind).bind(None, *args, **kwargs)
        bound.apply_defaults()
        vals = list(bound.arguments.values())[1:]
        object.__getattribute__(self, "calls").append(
            (int(sim.tick), kind, _normalise(sim, kind, vals)))
        if object.__getattribute__(self, "_pass"):
            getattr(sim, kind)(*args, **kwargs)


def _normalise(sim, kind, vals) -> tuple:
    from match_replay import INTENT_KINDS
    out = []
    for v, t in zip(vals, INTENT_KINDS[kind]):
        if t == "bool":
            out.append(bool(v))
        elif t == "int?":
            out.append(None if v is None else int(v))
        else:
            out.append(int(v))
    if kind == "set_move_dir":
        from simulation.movement import THROTTLE_ONE
        out[3] = min(max(out[3], 0), THROTTLE_ONE)
    elif kind == "plant_charge_intent":
        from simulation.weapons import PLANT_WEAPON, fuse_ticks_from_seconds
        row = sim.weapons_tables.weapons.by_name[PLANT_WEAPON]
        lo = fuse_ticks_from_seconds(row.fuse_min_seconds, sim._tps)
        hi = fuse_ticks_from_seconds(row.fuse_max_seconds, sim._tps)
        out[1] = min(max(out[1], lo), hi)
    return tuple(out)


def _make_verb(kind):
    def verb(self, *args, **kwargs):
        self._capture(kind, args, kwargs)
    verb.__name__ = kind
    return verb


def _make_refused(kind):
    def verb(self, *args, **kwargs):
        raise DoorRefused(f"IntentCapture: a captured controller called "
                          f"{kind!r}, which no soldier's verb is (a bot "
                          f"never calls it)")
    verb.__name__ = kind
    return verb


for _k in _policy_kinds():
    setattr(IntentCapture, _k, _make_verb(_k))
for _k in REFUSED_ON_CAPTURE:
    setattr(IntentCapture, _k, _make_refused(_k))
del _k


# ---------------------------------------------------------------------------
# The settings and the variants
# ---------------------------------------------------------------------------
THROW_RULES = ("nearest", "direction")


@dataclass(frozen=True)
class DoorSettings:
    """One door controller's coarseness (note section 1): ``dir_stride`` 1
    = 16 directions, 2 = 8 (feet and aim); ``decision_ticks`` the decision
    rate D; ``trigger_per_tick`` re-issues the bots' trigger every tick
    through ``set_decisions``; ``throws`` ``"nearest"`` (onto an entry or by
    direction, whichever lands nearer) or ``"direction"`` (by direction and
    reach only)."""
    dir_stride: int = 1
    decision_ticks: int = 4
    trigger_per_tick: bool = False
    throws: str = "nearest"

    def __post_init__(self):
        if self.dir_stride not in (1, 2):
            raise ValueError(f"DoorSettings.dir_stride must be 1 or 2, got "
                             f"{self.dir_stride!r}")
        if int(self.decision_ticks) < 1:
            raise ValueError(f"DoorSettings.decision_ticks must be >= 1, got "
                             f"{self.decision_ticks!r}")
        if self.throws not in THROW_RULES:
            raise ValueError(f"DoorSettings.throws must be one of "
                             f"{THROW_RULES}, got {self.throws!r}")


#: name -> settings: the design run and the variants of design 2.7. The
#: names are ``control_source.DOOR_BOT_NAMES`` (registered there, lazily);
#: the two are held equal at import.
DOOR_VARIANTS = {
    "bots_door": DoorSettings(),
    "bots_door_dirs8": DoorSettings(dir_stride=2),
    "bots_door_every6": DoorSettings(decision_ticks=6),
    "bots_door_every8": DoorSettings(decision_ticks=8),
    "bots_door_trigger_tick": DoorSettings(trigger_per_tick=True),
    "bots_door_throw_dir": DoorSettings(throws="direction"),
}


def _check_names() -> None:
    from control_source import DOOR_BOT_NAMES
    if set(DOOR_BOT_NAMES) != set(DOOR_VARIANTS):
        raise RuntimeError(
            f"control_source.DOOR_BOT_NAMES {sorted(DOOR_BOT_NAMES)} and "
            f"control_bots_door.DOOR_VARIANTS {sorted(DOOR_VARIANTS)} "
            f"disagree: a variant is one row in both")


_check_names()


# ---------------------------------------------------------------------------
# The translation (note section 3): a pure function of what it is handed
# ---------------------------------------------------------------------------
@dataclass
class Latch:
    """One captured latch: its tick, kind, normalised args and the
    soldier's centre tile at capture (a throw's point is fixed there)."""
    tick: int
    kind: str
    args: tuple
    centre: tuple


@dataclass
class SlotInput:
    """What :func:`translate` reads for one living slot."""
    unit: object                 # the living unit (centre, facing; read only)
    move: tuple | None           # ("set", dx, dy, thr) / ("clear",) / None
    aim: tuple | None            # the last captured aim vector, or None
    trigger: bool | None         # the last captured trigger, or None
    window: list                 # the Latch list since the last decision
    list_uid: np.ndarray         # (E,)
    list_type: np.ndarray        # (E,)
    masks: dict                  # mask_feet / mask_aim / mask_trigger /
                                 # mask_hand rows
    records: dict                # uid -> the team picture's record centre
    tick: int


@dataclass
class SlotLoss:
    """One slot decision's losses (bits) and the sums beside them."""
    bits: int = 0
    feet_angle_q: int = 0        # |angle error| of the feet bin, Q16 rad
    carried_ticks: int = 0
    landing_q: int = 0           # a throw's landing error, Q16 tiles
    family: str = ""             # "entry" / "direction" for a throw


def _vec_of(me_centre, centre):
    from control_source import quantize_stick_direction
    dx = int(centre[0]) - int(me_centre[0])
    dy = int(centre[1]) - int(me_centre[1])
    qx, qy, mag = quantize_stick_direction(float(dx), float(dy), 0.0)
    return None if mag == 0 else (qx, qy)


def _angle_err(dx, dy, d):
    from simulation import action_spec_v1 as A
    from simulation.turning import wrap_q16
    from simulation.unit_fixed import atan2_q16
    a = atan2_q16(-int(dy), int(dx))
    bx, by = A.COMPASS_Q16[d]
    b = atan2_q16(-int(by), int(bx))
    return abs(int(wrap_q16(a - b)))


def translate(si: SlotInput, tables, settings: DoorSettings) -> tuple:
    """``((feet, aim, trigger, hand, arg), SlotLoss)`` for one living slot:
    every number from the 1h tables through the 1h helpers; every choice
    the nearest LEGAL one under ``si.masks`` (note section 3)."""
    from simulation import action_spec_v1 as A
    loss = SlotLoss()
    stride = settings.dir_stride
    me = si.unit
    mc = (int(me.center_tile_x()), int(me.center_tile_y()))
    m_aim = si.masks["mask_aim"]
    m_hand = si.masks["mask_hand"]
    # -- feet --------------------------------------------------------------
    feet = A.FEET_STAND
    move_vec = None
    if si.move is not None and si.move[0] == "set":
        _k, dx, dy, thr = si.move
        if dx != 0 or dy != 0:
            move_vec = (dx, dy)
            feet = A.feet_index(dx, dy, thr, tables, stride=stride)
            d, s = A.feet_option(feet)
            if A.COMPASS_Q16[d] != (dx, dy):
                loss.bits |= _BIT["feet_dir"]
                loss.feet_angle_q = _angle_err(dx, dy, d)
            if tables.throttles[s] != thr:
                loss.bits |= _BIT["throttle"]
    # -- hand (first: a throw decides the aim) ------------------------------
    hand, arg, chosen = A.HAND_NOTHING, 0, None
    masked_seen = False
    for lt in reversed(si.window):
        if lt.kind == "throw_grenade_intent":
            opt = A.HAND_THROW0 + lt.args[5]
        elif lt.kind == "plant_charge_intent":
            opt = A.HAND_PLANT
        else:
            opt = _LATCH_HAND[lt.kind]
        if opt < len(m_hand) and m_hand[opt]:
            hand, chosen = opt, lt
            break
        masked_seen = True
    if masked_seen:
        loss.bits |= _BIT["hand_masked"]
    if len(si.window) > 1:
        loss.bits |= _BIT["superseded"]
    if chosen is not None and chosen.tick < si.tick:
        loss.bits |= _BIT["carried"]
        loss.carried_ticks = si.tick - chosen.tick
    keep_only = bool(m_aim[A.AIM_KEEP]) and int(m_aim.sum()) == 1
    aim = None
    if hand >= A.HAND_THROW0:
        got = _throw(si, chosen, tables, settings, mc, keep_only, loss)
        if got is None:
            hand = A.HAND_NOTHING          # no legal landing: nothing
        else:
            aim, arg = got
    elif hand == A.HAND_PLANT:
        _h, arg = A.plant_choice(chosen.args[1], tables)
    # -- aim ---------------------------------------------------------------
    if aim is None:
        aim = _aim(si, tables, stride, mc, keep_only, move_vec, loss)
    # -- trigger -----------------------------------------------------------
    trig = 1 if si.trigger else 0
    if hand >= A.HAND_THROW0 and trig:
        trig = 0
        loss.bits |= _BIT["trigger_throw"]
    return (feet, aim, trig, hand, arg), loss


def _aim(si, tables, stride, mc, keep_only, move_vec, loss) -> int:
    from simulation import action_spec_v1 as A
    m_aim = si.masks["mask_aim"]
    if keep_only:
        return A.AIM_KEEP
    v = si.aim
    if v is None or (v[0] == 0 and v[1] == 0):
        return A.AIM_WALK
    v = (int(v[0]), int(v[1]))
    hits = []
    for i in range(len(si.list_uid)):
        if not m_aim[A.AIM_TOKENS + i]:
            continue
        if int(si.list_type[i]) not in A.AIMABLE_TYPES:
            continue
        c = si.records.get(int(si.list_uid[i]))
        if c is not None and _vec_of(mc, c) == v:
            hits.append(i)
    if hits:
        if len(hits) > 1:
            loss.bits |= _BIT["aim_ambiguous"]
        return A.AIM_TOKENS + hits[0]
    if move_vec is not None and v == move_vec:
        return A.AIM_WALK
    tok = A.aim_index_for_vector(v[0], v[1], stride=stride)
    if any(_vec_of(mc, c) == v for c in si.records.values()):
        loss.bits |= _BIT["aim_missing"]
    elif A.COMPASS_Q16[tok] != v:
        loss.bits |= _BIT["aim_rounded"]
    return tok


def _throw(si, lt, tables, settings, mc, keep_only, loss):
    """The legal (aim, arg) pair whose landing lies nearest the bot's
    point (Q32 tiles, exact ints; ties to the first candidate: the entries
    in list order, then direction-major, reach-minor)."""
    from simulation import action_spec_v1 as A
    if keep_only:
        return None
    m_aim = si.masks["mask_aim"]
    _uid, dx, dy, _fuse, reach, _kind = lt.args
    reach = 0 if reach is None else int(reach)
    px = (int(lt.centre[0]) << 32) + int(dx) * reach
    py = (int(lt.centre[1]) << 32) + int(dy) * reach
    best = None
    if settings.throws == "nearest":
        for i in range(len(si.list_uid)):
            if not m_aim[A.AIM_TOKENS + i]:
                continue
            if int(si.list_type[i]) not in A.AIMABLE_TYPES:
                continue
            c = si.records.get(int(si.list_uid[i]))
            if c is None:
                continue
            lx, ly = int(c[0]) << 32, int(c[1]) << 32
            d2 = (lx - px) ** 2 + (ly - py) ** 2
            if best is None or d2 < best[0]:
                best = (d2, A.AIM_TOKENS + i, A.ARG_ONTO, "entry")
    cx, cy = mc[0] << 32, mc[1] << 32
    for d in range(0, A.N_DIRS, settings.dir_stride):
        if not m_aim[d]:
            continue
        ux, uy = A.COMPASS_Q16[d]
        for j, r in enumerate(tables.reaches_q):
            lx, ly = cx + ux * r, cy + uy * r
            d2 = (lx - px) ** 2 + (ly - py) ** 2
            if best is None or d2 < best[0]:
                best = (d2, d, j, "direction")
    if best is None:
        return None
    d2, aim, arg, fam = best
    if d2 != 0:
        loss.bits |= _BIT["throw_rounded"]
        loss.landing_q = isqrt(d2) >> 16
    loss.family = fam
    return aim, arg


# ---------------------------------------------------------------------------
# The records (note section 5)
# ---------------------------------------------------------------------------
_SINK = None


def install_sink(sink) -> None:
    """Install ``sink`` as this process's record sink (``None`` removes
    it): every door controller bound while it is installed reports to it.
    One match at a time per process (the tournament worker's way)."""
    global _SINK
    _SINK = sink


def current_sink():
    return _SINK


_PROCESS = {}


def _process_ids() -> dict:
    """The build ids and environment versions (once per process)."""
    if not _PROCESS:
        import match_replay
        try:
            import breach_physics
            bp_id = breach_physics.build_id()
        except Exception:                       # pragma: no cover
            bp_id = None
        from rl_env import environment_versions
        _PROCESS.update(build=match_replay.build_id(), bp_build_id=bp_id,
                        environment=environment_versions())
    return dict(_PROCESS)


class _Recorder:
    """One door team's imitation record and its reward books (only while a
    sink records)."""

    def __init__(self, ctl, sim, spec):
        from rl_env import COUNTER_NAMES, N_OLD_COUNTERS
        from simulation.reward_v1 import RewardBooks, group_weights, \
            PROFILES
        self.ctl = ctl
        self.S = ctl.S
        self.team = ctl.team
        self.terms = dict(PROFILES["fireteam"])
        n_teams = len(spec.teams)
        self.books = RewardBooks(sim, spec, self.S, self.terms,
                                 n_teams=n_teams)
        self.G = group_weights(self.terms, COUNTER_NAMES)
        self.k0 = N_OLD_COUNTERS
        self.ci = {n: i for i, n in enumerate(COUNTER_NAMES)}
        self.id_row = {}
        for t in range(n_teams):
            for u in sim.roster(t):
                self.id_row[int(u.id)] = (t, t * self.S + int(u.slot))
        self.n_rows = n_teams * self.S
        self.old = None
        self.team_before = None
        self.open = None            # the index of the decision whose window
        self.rows = []              # is open; per decision its arrays

    def before_step(self, sim) -> None:
        self.team_before = {u.id: int(u.team) for u in sim.units}
        self.books.begin(sim)

    def after_step(self, sim) -> None:
        if self.team_before is None:
            return
        from rl_env import count_events
        ended = self.books.ended
        count_events(sim.tick_events, self.team_before, self.id_row,
                     self.ci, self.old,
                     frozen=ended if ended.any() else None)
        self.books.end(sim)
        self.team_before = None

    def start_window(self, idx: int) -> None:
        self.books.start_step()
        self.old = np.zeros((self.n_rows, self.k0), dtype=np.float32)
        self.open = idx

    def close_window(self) -> None:
        if self.open is None:
            return
        counters = np.concatenate(
            [self.old.astype(np.float64), self.books.counters], axis=1)
        reward = (counters[:, :, None] * self.G[None]).sum(axis=1)
        lo, hi = self.team * self.S, (self.team + 1) * self.S
        row = self.rows[self.open]
        row["reward"] = reward[lo:hi].astype(np.float32)
        row["rows_ended"] = self.books.rows_ended[lo:hi].copy()
        self.open = None

    def finish(self, sim, result) -> None:
        self.after_step(sim)
        self.books.finish(result)
        self.close_window()


class RecordSink:
    """The process-local sink of one match's door teams (note section 5).
    ``out_dir`` None = STATS ONLY (no file, no reward book). Door
    controllers bound while it is installed register with it;
    :meth:`close` (after the match) finishes each team's books, writes its
    record (and the match's replay file) and returns the per-team stats."""

    def __init__(self, out_dir=None):
        self.out_dir = None if out_dir is None else Path(out_dir)
        self.doors = []

    @property
    def records(self) -> bool:
        return self.out_dir is not None

    def register(self, ctl) -> None:
        self.doors.append(ctl)

    def base_name(self, spec, seed) -> str:
        c0, c1 = (t.controller for t in spec.teams[:2])
        return f"{c0}-{c1}__z{int(spec.zombies)}__s{int(seed)}"

    def close(self, sim, result, digests, *, seed, recording=None) -> dict:
        """Close every registered door team; returns ``{team: stats}``."""
        import match_replay
        out = {}
        if not self.doors:
            return out
        spec = self.doors[0].spec
        level = match_replay.level_id(sim.level)
        replay_name = replay_sha = None
        if self.records:
            d = self.out_dir / level
            d.mkdir(parents=True, exist_ok=True)
            base = self.base_name(spec, seed)
            if recording is not None:
                replay_name = base + ".replay.gz"
                match_replay.write_recording(recording, d / replay_name)
                replay_sha = hashlib.sha256(
                    (d / replay_name).read_bytes()).hexdigest()
        for ctl in self.doors:
            stats = ctl.close(sim, result)
            out[str(ctl.team)] = stats
            if self.records and ctl.recorder is not None:
                path = (self.out_dir / level
                        / f"{self.base_name(spec, seed)}__t{ctl.team}.npz")
                _write_record(path, ctl, sim, result, digests, seed,
                              replay_name, replay_sha, stats)
                stats["file"] = str(path)
                stats["bytes"] = path.stat().st_size
        return out


_ARRAY_KEYS = ("vector", "list", "list_mask", "list_uid", "near_f",
               "near_u", "far_f", "far_u", "valid", "in_match", "mask_feet",
               "mask_aim", "mask_trigger", "mask_hand", "arg_mask",
               "actions", "loss", "reward", "rows_ended")


def _write_record(path, ctl, sim, result, digests, seed, replay_name,
                  replay_sha, stats) -> None:
    import match_replay
    from simulation import action_spec_v1 as A
    rec = ctl.recorder
    rows = rec.rows
    S = ctl.S
    arrays = {"tick": np.array([r["tick"] for r in rows], dtype=np.int64)}
    for k in _ARRAY_KEYS:
        if rows:
            arrays[k] = np.stack([r[k] for r in rows])
        else:
            arrays[k] = np.zeros((0, S), dtype=np.uint8)
    lg = getattr(sim.level, "levelgen", None)
    meta = {
        "record_format": RECORD_FORMAT,
        "level_id": match_replay.level_id(sim.level),
        "levelgen": (None if lg is None else {
            "recipe_hash": lg.recipe_hash, "level_seed": lg.level_seed,
            "generator_version": lg.generator_version}),
        "seed": int(seed),
        "spec": match_replay.spec_to_dict(ctl.spec),
        "controllers": [t.controller for t in ctl.spec.teams],
        "team": ctl.team, "name": ctl.name,
        "settings": asdict(ctl.settings),
        "S": S, "E": ctl.tables.n_entries, "K": ctl.tables.n_kinds,
        "hand_names": list(A.hand_names(ctl.tables.hand_grenade_rows)),
        "decision_ticks": ctl.settings.decision_ticks,
        "obs_spec": "v1", "action_spec": A.ACTION_SPEC_NAME,
        "reward_profile": "fireteam", "reward_terms": rec.terms,
        "result": {"outcome": result.outcome, "team": result.team,
                   "ticks": result.ticks},
        "field_digest": digests[0], "tick_digest": digests[1],
        "replay": replay_name, "replay_sha256": replay_sha,
        "loss_bits": list(LOSS_BITS),
        "loss_totals": {k: v for k, v in stats.items()
                        if k not in ("file", "bytes")},
    }
    meta.update(_process_ids())
    arrays["meta"] = np.array(json.dumps(meta, sort_keys=True))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez_compressed(f, **arrays)


def read_record(path) -> tuple:
    """``(arrays, meta)`` of one record file; refuses another
    ``record_format`` by name."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"][()]))
    if meta.get("record_format") != RECORD_FORMAT:
        raise ValueError(f"{path}: record_format {meta.get('record_format')!r}"
                         f", this reader reads {RECORD_FORMAT} only")
    return arrays, meta


# ---------------------------------------------------------------------------
# The controller
# ---------------------------------------------------------------------------
def _new_stats() -> dict:
    s = {n: 0 for n in LOSS_BITS}
    s.update(decisions=0, slot_decisions=0, living_ticks=0,
             feet_angle_q=0, carried_ticks=0, landing_q=0,
             hold_diff_feet_ticks=0, hold_diff_aim_ticks=0,
             hold_diff_trigger_ticks=0, throw_entry=0, throw_direction=0,
             throws_sent=0, throws_started=0, latches_sent=0,
             masked_sent=0, orders=0)
    return s


class BotsThroughDoor(ControlSource):
    """The door bots (see the module doc and note section 1): ONE match
    controller around an inner ``BotTeam`` (decides, captured) and an inner
    ``AgentPolicy`` on action spec v1 (the team's one issuer). Built by the
    registry (``bots_door`` and its variants); :meth:`bind_match` (called by
    ``create_match_controllers``) ties it to its match; an unbound one
    refuses to run."""

    def __init__(self, team: int, settings: DoorSettings | None = None,
                 name: str = "bots_door"):
        from control_bots import BotTeam
        from control_policy import AgentPolicy
        self.team = int(team)
        self.settings = DoorSettings() if settings is None else settings
        self.name = name
        self.bots = BotTeam(self.team)
        self.policy = AgentPolicy(self.team)
        self.spec = None
        self.S = None
        self.tables = None
        self.recorder = None
        self.stats = _new_stats()
        self._last_tick = None
        self._windows = {}
        self._holds_now = {}
        self._issued_now = []
        self._dec = None            # (rows5, list_uid) of the last decision
        self._closed = False
        self.decision_ticks = []    # the ticks set_decisions was called on
        self.on_decision = None     # a test's hook: f(tick, rows5, out)

    # -- binding -------------------------------------------------------------
    def bind_match(self, sim, spec) -> None:
        from rl_env import _team_slots
        from simulation.action_spec_v1 import action_tables
        from simulation.obs_spec_v1 import ObsContext, shapes
        from simulation.action_spec_v1 import mask_shapes
        from simulation.sensor_accessor import EntityFieldAccessor
        level = sim.level
        graph = getattr(level, "rooms", None)
        if graph is None or (tuple(graph.grid.shape)
                             != tuple(level.tilemap.shape)):
            raise ValueError(
                f"controller {self.name!r} (team {self.team}) needs a room "
                f"graph at the map's shape: level {level.name!r} has none -- "
                f"action spec v1 is built on observation spec v1, which reads "
                f"the room graph")
        self.spec = spec
        self.S = S = max(_team_slots(level))
        self.accessor = EntityFieldAccessor(sim.gmap)
        self.ctx = ObsContext.build(sim, spec, S, n_teams=len(spec.teams))
        self.tables = action_tables(sim, S)
        if int(self.bots.fuse_ticks) != int(self.tables.fuse_ticks):
            raise DoorRefused(
                f"{self.name}: the bots' fuse ({self.bots.fuse_ticks} ticks) "
                f"is not the action spec's one fuse "
                f"({self.tables.fuse_ticks})")
        self.policy.configure("v1", self.tables)
        sh = dict(shapes(1, S, S))
        sh.update(mask_shapes(1, S, self.tables.n_entries,
                              self.tables.n_kinds))
        self._out = {k: np.zeros(s[1:], dtype=dt) for k, (s, dt)
                     in sh.items()}
        sink = current_sink()
        if sink is not None:
            sink.register(self)
            if sink.records:
                self.recorder = _Recorder(self, sim, spec)

    def aim_view(self) -> dict:
        """The decoder's view (``AgentPolicy.aim_view``): what the
        observation reads for this team's cols 28, 66, 71..76."""
        return self.policy.aim_view()

    # -- the tick -------------------------------------------------------------
    def handle_frame(self, sim, renderer=None) -> None:
        if self.tables is None:
            raise RuntimeError(f"{self.name} (team {self.team}) is not bound "
                               f"to a match: create_match_controllers calls "
                               f"bind_match(sim, spec) once, first")
        tick = int(sim.tick)
        if tick == self._last_tick:
            return
        self._last_tick = tick
        self._count_started(sim)
        rec = self.recorder
        if rec is not None:
            rec.after_step(sim)
        cap = IntentCapture(sim)
        self.bots.handle_frame(cap, None)
        holds = self._gather(sim, cap.calls, tick)
        self._holds_now = holds
        self._orders(sim, tick)
        D = int(self.settings.decision_ticks)
        if tick % D == 0:
            self._decide(sim, tick)
        elif self.settings.trigger_per_tick and self._dec is not None:
            self._retrigger(sim, holds)
        issued = IntentCapture(sim, passthrough=True)
        self.policy.handle_frame(issued)
        self._issued_now = issued.calls
        self._diagnose(sim, holds, issued.calls)
        if rec is not None:
            rec.before_step(sim)

    def _count_started(self, sim) -> None:
        from simulation.events import ThrowEvent
        mine = {u.id for u in sim.units if u.team == self.team}
        for ev in getattr(sim, "tick_events", ()) or ():
            if isinstance(ev, ThrowEvent) and ev.unit_id in mine:
                self.stats["throws_started"] += 1

    def _gather(self, sim, calls, tick) -> dict:
        holds = {}
        for t, kind, args in calls:
            uid = int(args[0])
            h = holds.setdefault(uid, {"move": None, "aim": None,
                                       "trigger": None})
            if kind == "set_move_dir":
                h["move"] = ("set",) + tuple(args[1:4])
            elif kind == "clear_move_dir":
                h["move"] = ("clear",)
            elif kind == "set_aim":
                h["aim"] = (args[1], args[2])
            elif kind == "set_trigger":
                h["trigger"] = bool(args[1])
            elif kind == "set_flashlight":
                continue            # reserved in v1: never sent
            else:
                u = sim.get_unit(uid)
                c = ((int(u.center_tile_x()), int(u.center_tile_y()))
                     if u is not None else (0, 0))
                self._windows.setdefault(uid, []).append(
                    Latch(t, kind, args, c))
        return holds

    def _mine(self, sim) -> list:
        return [u for u in sim.units
                if u.team == self.team and u.alive and not u.is_zombie]

    def _orders(self, sim, tick) -> None:
        from control_bots import issue_team_order, team_order
        if self.bots.brain.updated_at != tick:
            return
        mine = self._mine(sim)
        if not mine:
            return
        want = team_order(self.bots.brain, sim, mine, self.team)
        self.stats["orders"] += issue_team_order(sim, mine, want)

    def _row_units(self, sim) -> list:
        out = [None] * self.S
        for u in sim.roster(self.team):
            if u is not None and u.alive and u.team == self.team:
                out[int(u.slot)] = u
        return out

    def _decide(self, sim, tick) -> None:
        from rl_env import observe_v1
        from simulation import action_spec_v1 as A
        out = self._out
        S = self.S
        units = self._row_units(sim)
        observe_v1(self.accessor, sim, self.ctx, units, out,
                   self.policy.aim_view(), self.tables,
                   self.settings.decision_ticks)
        valid = np.zeros(S, dtype=np.uint8)
        in_match = np.zeros(S, dtype=np.uint8)
        for u in sim.roster(self.team):
            if u is None:
                continue
            in_match[u.slot] = 1
            if u.alive and u.team == self.team:
                valid[u.slot] = 1
        out["valid"][...] = valid
        out["in_match"][...] = in_match
        records = {uid: (int(r.centre[0]), int(r.centre[1])) for uid, r
                   in sim.team_picture.records(self.team).items()}
        rows5 = np.zeros((S, len(A.HEADS)), dtype=np.int64)
        rows5[:, 1] = A.AIM_WALK
        losses = np.zeros(S, dtype=np.uint16)
        lst = out["list"]
        holds = self._holds_now
        for s, u in enumerate(units):
            if u is None:
                continue
            h = holds.get(int(u.id), {})
            si = SlotInput(
                unit=u, move=h.get("move"), aim=h.get("aim"),
                trigger=h.get("trigger"),
                window=self._windows.get(int(u.id), []),
                list_uid=out["list_uid"][s], list_type=lst[s, :, 0],
                masks={k: out[k][s] for k in A.MASK_KEYS},
                records=records, tick=tick)
            five, loss = translate(si, self.tables, self.settings)
            rows5[s] = five
            losses[s] = loss.bits & 0xFFFF
            self._account(loss, five)
        self._check(tick, rows5)
        self.stats["decisions"] += 1
        self._windows = {}
        lu = np.array(out["list_uid"], copy=True)
        self.decision_ticks.append(tick)
        rec = self.recorder
        if rec is not None:
            rec.close_window()
            row = {k: np.array(out[k], copy=True) for k in
                   ("vector", "list", "list_mask", "list_uid", "near_f",
                    "near_u", "far_f", "far_u", "valid", "in_match")
                   + A.MASK_KEYS}
            row["tick"] = tick
            row["actions"] = rows5.astype(np.int16)
            row["arg_mask"] = A.arg_mask(rows5[:, 3], rows5[:, 1])
            row["loss"] = losses
            row["reward"] = np.zeros((S, 5), dtype=np.float32)
            row["rows_ended"] = np.zeros(S, dtype=bool)
            rec.rows.append(row)
            rec.start_window(len(rec.rows) - 1)
        if self.on_decision is not None:
            self.on_decision(tick, rows5.copy(), out)
        self.policy.set_decisions(rows5, lu)
        self._dec = (rows5.copy(), lu)

    def _account(self, loss, five) -> None:
        from simulation import action_spec_v1 as A
        st = self.stats
        st["slot_decisions"] += 1
        for n in LOSS_BITS:
            if loss.bits & _BIT[n]:
                st[n] += 1
        st["feet_angle_q"] += loss.feet_angle_q
        st["carried_ticks"] += loss.carried_ticks
        st["landing_q"] += loss.landing_q
        if loss.family:
            st["throw_" + loss.family] += 1
        if five[3] != A.HAND_NOTHING:
            st["latches_sent"] += 1

    def _check(self, tick, rows5) -> None:
        """Gate 1, raised: every index in range and nothing masked."""
        from simulation import action_spec_v1 as A
        out = self._out
        try:
            A.check_actions(rows5[None], self.tables.sizes)
        except ValueError as e:
            raise DoorRefused(f"{self.name} tick {tick}: {e}") from None
        masks = {k: out[k][None] for k in A.MASK_KEYS}
        mc = A.masked_choices(rows5[None], masks)[0]
        if mc.any():
            s = int(np.flatnonzero(mc)[0])
            heads = [A.HEADS[h] for h, k in enumerate(A.MASK_KEYS)
                     if not out[k][s][rows5[s, h]]]
            am = A.arg_mask(rows5[s, 3], rows5[s, 1])
            if not am[rows5[s, 4]]:
                heads.append("arg")
            self.stats["masked_sent"] += int(mc.sum())
            raise DoorRefused(
                f"{self.name} tick {tick}: slot {s} would send a masked "
                f"choice on head(s) {heads}: {rows5[s].tolist()}")

    def _retrigger(self, sim, holds) -> None:
        """The trigger variant: the decision's feet and aim, THIS tick's
        captured trigger, hand nothing, on the stored list_uid."""
        rows5, lu = self._dec
        rows = rows5.copy()
        rows[:, 3] = 0
        rows[:, 4] = 0
        for s, u in enumerate(self._row_units(sim)):
            if u is None:
                continue
            t = holds.get(int(u.id), {}).get("trigger")
            rows[s, 2] = 1 if t else 0
        self.policy.set_decisions(rows, lu)

    def _diagnose(self, sim, holds, issued) -> None:
        got = {}
        for _t, kind, args in issued:
            g = got.setdefault(int(args[0]), {"move": ("clear",),
                                              "aim": (0, 0),
                                              "trigger": False})
            if kind == "set_move_dir":
                g["move"] = ("set",) + tuple(args[1:4])
            elif kind == "clear_move_dir":
                g["move"] = ("clear",)
            elif kind == "set_aim":
                g["aim"] = (args[1], args[2])
            elif kind == "set_trigger":
                g["trigger"] = bool(args[1])
            elif kind == "throw_grenade_intent":
                self.stats["throws_sent"] += 1
        st = self.stats
        for u in self._mine(sim):
            st["living_ticks"] += 1
            h = holds.get(int(u.id), {})
            g = got.get(int(u.id))
            if g is None:
                continue
            mv = h.get("move") or ("clear",)
            if mv != g["move"]:
                st["hold_diff_feet_ticks"] += 1
            if (h.get("aim") or (0, 0)) != g["aim"]:
                st["hold_diff_aim_ticks"] += 1
            if bool(h.get("trigger")) != g["trigger"]:
                st["hold_diff_trigger_ticks"] += 1

    # -- closing ---------------------------------------------------------------
    def close(self, sim, result) -> dict:
        """After the match: count the last step's thrown grenades, finish
        the books (with records) and return the stats."""
        if not self._closed:
            self._closed = True
            self._count_started(sim)
            if self.recorder is not None:
                self.recorder.finish(sim, result)
        return dict(self.stats)


def create(team: int, name: str = "bots_door") -> BotsThroughDoor:
    """The registry's factory for ``name`` (a :data:`DOOR_VARIANTS` row)."""
    return BotsThroughDoor(team, DOOR_VARIANTS[name], name)


__all__ = [
    "BotsThroughDoor", "IntentCapture", "DoorSettings", "DOOR_VARIANTS",
    "DoorRefused", "translate", "SlotInput", "SlotLoss", "Latch",
    "RecordSink", "install_sink", "current_sink", "read_record",
    "RECORD_FORMAT", "LOSS_BITS", "create",
]
