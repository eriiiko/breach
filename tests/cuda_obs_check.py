"""#29 P8 -- THE DEVICE OBSERVATION'S SHARED CHECK (``simulation.device_obs``):
the observation built on the card inside the device conductor's tick, held
EQUAL, as raw bits, to the HOST observation (``obs_spec_v1.observe_rows`` and
``rl_env.EnvRunner.row_flags``, THE reference) at every decision tick. Every
P8 patch extends this one script (a part per patch, never a new script).

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_obs.py).

THE CORPUS, played ONCE per subprocess (:func:`play`): a REFERENCE match runs
on the host CPU path with its own controllers (the bots), its intents taken
from its own tap per tick (``Simulation.attach_intent_log``), and at every
decision tick (tick 0, then every :data:`DECIMATION` ticks) the host builds
the whole observation the env would serve (an ``ObsContext`` per episode,
the rows' units, the team picture the session keeps). Kept: the tick's
intents and the OWNED arrays of each decision. (a) the pinned ship_demo
matches ``bots`` and ``bots_v1`` (``tests/_pinned_matches``); (b) a training
room (``tools/gen_training_rooms.py``'s ``range_door``, written to a temp
folder: a door, ``all_round``; no cover: the crates are ``range_hall``'s,
P8.2's :func:`corpus_cover` plays it as it is and with a crate moved onto
the line of fire); (d) the hand match (P8.1: a ship_demo
bots match cut at :data:`HAND_TICKS` with a hand state at tick 0 and hand
intents, :func:`hand_setup` / :func:`hand_script`, for the owned columns (a)
and (b) never move); and, for the batch, more seeds of (a) and (b).

OWNERSHIP (``tests/obs_column_owners.toml``, the column-to-patch table): a
gate compares exactly the columns / planes owned by :data:`PATCH` and every
earlier patch, and counts per owned column how many cells over the corpus
are not the default (0); an owned column holding one value over its whole
corpus FAILS (a gate on a column that never moved is vacuous).

THE GATE (:func:`gate`): the conductor on fresh twins of the played matches
(N envs, one level), fed each env's tapped intents per tick (an env whose
match ended ticks on with none); a ``DeviceObservation`` attached; at every
decision tick of every env its owned device arrays compared with the
reference's, raw bits (``view`` as unsigned ints: -0.0 and NaN count); the
first difference printed (tick, env, key, plane, index) and FAILS.

PART GATE (P8.0) -- corpus (a), (b) and (d) at N = 1.
PART BATCH (P8.0) -- N = 2 and 4 of ship_demo and of the training room,
different seeds per env (envs end apart): every env equal to ITS reference
at every one of its decision ticks (so equal to itself alone).
PART HANDOVER (P8.0) -- the tensors torch gets (``DeviceObservation.torch``,
``team_rows``) share the CuPy buffers' memory (data pointer identity); the
stream ordering the hand-over rests on, read from the runtime: the tick
stream is BLOCKING, CuPy and torch work on the legacy default stream; a
torch read straight after a tick (no explicit sync) equals the synced value.
PART SYNCS (P8.0) -- a conductor tick with the observation attached makes
only the conductor's named syncs (``max_n_sub``, the books read), the same
count at N = 1 and 2, and the blocks were built during the count.
PART ENDED (P8.0) -- an env whose ended latch is set keeps its buffers (the
blocks skip it) while its neighbour is rebuilt.

PART FLOATKIT (P8.1) -- the observation's float kit on the card
(``cpp/src/cuda_obs_float.cuh``): double -> float16 exhaustively equal to
numpy, double -> float32 equal to numpy, a*b + c never contracted, the
squash's served float32 equal to the host's (CUDA's log; the float64
difference count is printed).

P8.2 (no served column: the sight every later block reads) --
- inside every GATE / BATCH run: who sees whom (``DeviceObservation.vision``:
  ``seen_by``, ``visible_by_team``) equal to the reference's VisionState
  (``vision.state_for``, the team picture's) at EVERY tick, both sight rules,
  the per-rule census (pair-ticks, pairs ever seen / not seen / both) judged
  by part GATE; the line of fire (``lof_pairs``, ``line_of_fire.solid_in_way``
  over the cover table) on every (unit, unit) pair at every decision tick,
  the pairs a cover decided counted (> 0 over the corpus, else FAIL); a
  cover's alive unchanged over every match (the table is staged once);
- PART SIGHT: the bulk forms against ``rays_clear`` / ``window_clear``;
- PART WALL: the wall's first device gate (critique finding 22).

P8.3 (the device team picture, ``simulation.device_picture``) --
- inside every GATE / BATCH run: the picture's WHOLE state
  (``device_picture.device_form`` of the card's block) equal to the
  reference picture's (``host_form``, objects named by their position in
  ``projectiles + planted_charges``) at EVERY tick, field by field, and the
  damage window's sums (``damage_last_s``) raw-bit equal; the coverage of
  every case (:data:`PICTURE_CASES`, each > 0 else FAIL) counted off the
  reference; the door hand match (:func:`corpus_door`) for the grenade lost
  and re-seen; the 13 vector columns P8.3 owns in the census;
- PART RESTAGE: the seam a per-env reset uses.

P8.4a (the lanes, ``aim_of``, the line of fire, ``ticks_to_within``:
``DeviceObservation.aims``) --
- inside every GATE / BATCH run: the served lanes plane (``near_u`` /
  ``lanes``) with the owned columns, raw bits; and the intermediates the list
  reads compared with the REFERENCE'S OWN values (:class:`AimTap`, a
  test-side tap of the builder's calls; the builder unchanged) at every
  decision tick: each team's fire lanes, each own lane, each ``aim_of``
  answer, each (soldier, seen enemy)'s ``solid_in_way`` / ``mate_in_way`` /
  ``ticks_to_within``; the coverage of every case (:data:`AIM_CASES`, each >
  0 else FAIL), the cap case from the turn hand match (:func:`corpus_turn`);
- PART WALL extended: the unseen enemy's row perturbed (its trigger
  flipped too) moves none of the team's lanes, lanes plane, aims, lines of
  fire or turns.

P8.4b (the list: ``DeviceObservation.build_list``) --
- inside every GATE / BATCH run: ``list`` (every entry column but the
  decoder's ``on_next_tick`` / ``prev_aim`` and the order block),
  ``list_mask`` and ``list_uid`` raw-bit equal as served columns, every
  owned column non-constant over the corpus; no list fault; the coverage
  (:data:`LIST_CASES`: an entry of every block, a grenade and a charge, a
  door in reach and out of it, a block overflowing its capacity, counted off
  the reference with :class:`ListTap`), each > 0 else FAIL;
- PART WALL extended: the unseen enemy's row perturbed moves no entry, mask
  or uid of the team's list.

P8.4c (the four masks: ``DeviceObservation.build_masks``) --
- inside every GATE / BATCH run: ``mask_feet``, ``mask_aim``,
  ``mask_trigger``, ``mask_hand`` raw-bit equal to the reference's
  (``action_spec_v1.build_masks`` over the host list, ``rl_env.observe_v1``'s
  call), no mask fault; the coverage (:data:`MASK_CASES`: every hand bit,
  every fixed aim token and an aimable entry both set and clear over the
  corpus, the wind-up's KEEP-only case, a use with a door in reach; the bits
  the spec fixes -- stand, the first trigger option, nothing always legal,
  the flashlight always masked -- exempt by name), counted off the
  reference, each > 0 else FAIL;
- PART WALL extended: the unseen enemy's row perturbed moves no mask of the
  team.

P8.5a (the windows: ``DeviceObservation.build_windows``) --
- inside every GATE / BATCH run: the eight physics planes of ``near_f`` and
  ``far_f``, ``near_u`` watched / sees_target, ``far_u`` mates / enemies /
  records and the vector's ``air`` block and ``felt_temp`` raw-bit equal, no
  window fault; ``sees_target`` FROM the window TO the aimed entry's recorded
  centre with the aims of :func:`aim_rule` (the decoder's stand-in, given to
  the reference's ``observe_rows`` and written into
  ``DeviceObservation.aimed`` before the tick that builds); the coverage
  (:data:`WIN_CASES`: a watched tile, a tile that sees the target, a near
  window over each edge of the map, a far block straddling the map's edge, a
  far cell off the block grid, an aimed entry found and not found), counted
  off the reference, each > 0 else FAIL; the cells past the squash's knee per
  float plane printed (none at all FAILS); the fire hand match
  (:func:`corpus_fire`) for the fire plane;
- PART WALL extended: the unseen enemy's row perturbed moves no window plane
  of the team, nor its air block or felt temperature.

THE BREAKS (``OBS_BREAK``), each must turn a part red: ``win_divisor``,
``win_anchor``, ``win_half``, ``win_argmax`` (P8.5a, part GATE: each gas plane
divided by the neighbouring density row; the far window's block anchor one
block off; the float16 cast through float32; the far wind's argmax taking the
last largest load) and ``win_profile`` (the profile read one row further:
today's config holds one env profile and one species with equal floats, so
it CANNOT turn red; it is run to show it); ``mask_swap``,
``mask_reach`` (P8.4c, part GATE: the reload and shove bits' gates swapped;
the use mask's reach one tile wider); ``list_tie``,
``list_cap`` (P8.4b, part GATE: the sort ties by the higher id; the door
block one entry short of its capacity); ``lane_trunc``,
``lane_wide`` (P8.4a, part GATE: C's truncating division in place of the
floor division in the lane walk and the rasteriser; the lane walk one tile
wide of its line); ``early_picture``,
``lifetime``, ``dmg_order``, ``serial_row`` (P8.3, part GATE: the picture
before the conversion, a record kept one tick longer, the tick's hits fed
to the damage window in reverse, the object serial replaced by the row
index), ``dmg_sum`` (P8.3: the window's sums in reverse order -- every
entry is on the Q16 grid, so the float64 sums are exact and this break
CANNOT turn red; it is run to show it); ``cone`` / ``early``
/ ``cover`` (P8.2, part GATE: the cone a degree wider, who sees whom before
the conversion, every cover box one tile off), ``forms`` (P8.2, part SIGHT:
the two has_los forms swapped); ``anchor`` (the
blocks read the units one tile off: the window's anchor wrong), ``stale``
(the blocks are not rebuilt after tick 0), ``constant`` (an owned column
forced constant on both sides: the constancy guard), ``divisor`` (P8.1: the
obs table's weapon block shifted one row, so ``rounds`` / ``range`` / ...
divide by the neighbouring row), ``half_f32`` (P8.1: double -> float16 via
float32, double rounding), ``fused`` (P8.1: the kit's a*b + c as an fma).

PROPERTY: the device observation IS the host observation, value for value,
on every owned column, at every decision tick, in a batch as alone; the
torch tensors are the buffers; the observation adds no host sync. BREAKS IF
a block reads the wrong anchor, a row map entry, another env's slice, a
stale state, or a plane outside the reference's fill rules.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src", _ROOT / "tools", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cuda_conductor_check as cc  # noqa: E402
from simulation import conductor as cd  # noqa: E402
from simulation import device_obs as dobs  # noqa: E402
from simulation import action_spec_v1 as _aspec  # noqa: E402

FAILS: list = []
BREAK = os.environ.get("OBS_BREAK", "")
#: The patch this script's gates are at: every column owned by it or an
#: earlier patch of the version-2 table is compared.
PATCH = "P8.6a"
#: The decision cadence of the corpus (design 13.1's decimation, a run file's
#: ``[run] decimation``; the device observation takes it as its argument).
DECIMATION = 4
#: The seeds of the batch's envs (per corpus: different seeds, envs end apart).
BATCH_SHIP = [(None, 11), ("bots_v1", 12), (None, 13), ("bots_v1", 14)]
BATCH_ROOM = [21, 22, 23, 24]
ROOM = "range_door"
OWNERS = _ROOT / "tests" / "obs_column_owners.toml"


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", str(msg).encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Ownership (the column-to-patch table)
# ---------------------------------------------------------------------------
def owned(patch: str = PATCH) -> dict:
    """``key -> tuple of plane / column names`` (None: the whole key) owned
    by ``patch`` and every earlier patch of the table."""
    t = tomllib.loads(OWNERS.read_text(encoding="utf-8"))
    order = t["patches"]
    upto = set(order[:order.index(patch) + 1])
    out = {}
    for key, who in t["keys"].items():
        if who in upto:
            out[key] = None
    for key, cols in t["columns"].items():
        names = tuple(n for n, who in cols.items() if who in upto)
        if names:
            out[key] = names
    return out


def owned_obs(patch: str = PATCH) -> dict:
    """The owned keys that are OBSERVATION buffers (``DeviceObservation.buffers``);
    the reward side (``device_obs.REWARD_KEYS``, P8.7) is compared by
    :func:`books_equal` from ``DeviceBooks.served``."""
    return {k: v for k, v in owned(patch).items() if k not in dobs.REWARD_KEYS}


def _plane_index(key: str, name: str) -> int:
    from simulation import obs_spec_v1 as o
    planes = {"near_f": o.NEAR_F_PLANES, "near_u": o.NEAR_U_PLANES,
              "far_f": o.FAR_F_PLANES, "far_u": o.FAR_U_PLANES}
    if key in planes:
        return planes[key].index(name)
    cols = {"vector": o.VECTOR_COLUMNS, "list": o.ENTRY_COLUMNS}[key]
    return next(f.pos for f in cols if f.name == name)


def owned_parts(arrs: dict, own: dict, env_axis: bool) -> dict:
    """The owned slices of an observation dict, ``(key, name) -> array``
    (the row axis first; ``env_axis``: drop a leading env axis first is the
    caller's)."""
    out = {}
    for key, names in own.items():
        a = arrs[key]
        if names is None:
            out[(key, None)] = np.array(a, copy=True)
            continue
        for n in names:
            i = _plane_index(key, n)
            out[(key, n)] = np.array(a[:, i] if key not in ("vector", "list")
                                     else a[..., i], copy=True)
    return out


def raw(a) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


# ---------------------------------------------------------------------------
# THE CORPUS: each match played ONCE on the host (the reference)
# ---------------------------------------------------------------------------
class Played:
    """One reference match: its level, spec, seed, (T, S), per tick its
    intents, per decision tick the owned arrays of the host observation."""

    def __init__(self, name, level, spec, seed, T, S, setup=None):
        self.name, self.level, self.spec, self.seed = name, level, spec, seed
        self.T, self.S = T, S
        self.setup = setup                    # corpus (d): the hand state at tick 0
        self.intents = []                     # per tick k: [(kind, args)]
        self.obs = {}                         # tick -> {(key, name): array}
        self.end = 0
        # P8.2: the reference's VisionState every tick, its line of fire over
        # every (unit, unit) pair at every decision tick, the sight rule
        self.vision = {}                      # tick -> (seen_by, visible_by_team)
        self.lof = {}                         # tick -> (U, U) uint8 (sim.units order)
        self.lof_cover = {}                   # tick -> pairs a cover decided (walk clear)
        self.rule = "cone"
        self.uids = []
        self.cover_alive = None               # (at tick 0, at the end)
        # P8.3: the reference team picture's normal form every tick, its
        # damage window's sums per (team, unit) every tick, the coverage
        self.picture = {}                     # tick -> device_picture.host_form
        self.dsum = {}                        # tick -> (T, U) float64
        self.coverage = {}                    # case -> count
        self.dmg_peak = 0                     # the largest damage window of a team
        # P8.4a: the reference's own intermediates at every decision tick (the
        # test-side tap, :class:`AimTap`), and their coverage
        self.aims = {}                        # tick -> AimTap record
        self.aim_cov = {k: 0 for k in AIM_CASES}
        # P8.4b: the list's coverage, counted off the reference
        self.list_cov = {k: 0 for k in LIST_CASES}
        # P8.4c: the masks' coverage, counted off the reference
        self.mask_cov = {}
        # P8.5a: per decision tick the rows' aimed uids (aim_rule), the
        # windows' coverage and per float plane its cells past the squash's
        # knee, counted off the reference
        self.aimed = {}                       # tick -> (rows,) int32
        self.win_cov = {k: 0 for k in WIN_CASES}
        self.knee = {}                        # (key, plane) -> cells with |x| > 1
        # P8.7: per named reward profile the reference books' normal form every
        # tick, the served arrays at every step's end (decision or ending
        # tick), whether the match ENDED (terminal, not cut), the coverage
        self.books = {}                       # profile -> tick -> device_books.host_form
        self.served = {}                      # profile -> tick -> served dict
        self.terminal = False
        self.result = None                    # (outcome, team, ticks) when terminal
        self.book_cov = {}                    # profile -> case -> count
        # P8.6a (corpus (c), a policy run): the driver, per decision tick the
        # decisions, per tick the reference's latch columns / slot states /
        # horde latch after the controllers' handle_frame and its digests after
        # the step, the decoder's coverage
        self.driver = None
        self.decisions = {}                   # tick -> (rows, 5) int32
        self.dec_obs = {}                     # tick -> {key: array} (list_uid, masks)
        self.ref_latch = {}                   # tick -> (U, LW) int64
        self.ref_slots = {}                   # tick -> {row: host_state_form}
        self.ref_horde = {}                   # tick -> latch row
        self.digests = {}                     # tick (after the step) -> digests
        self.dec_cov = {}


_PLAYED: dict = {}
_ROOMS_DIR = None


def ship_level():
    import _pinned_matches as pm
    return pm.ship_demo()


def room_level(name: str = ROOM):
    global _ROOMS_DIR
    import gen_training_rooms as TR
    import level_loader
    if _ROOMS_DIR is None:
        _ROOMS_DIR = tempfile.mkdtemp(prefix="obs_rooms_")
    if not (Path(_ROOMS_DIR) / name / "level.toml").exists():
        for d in TR.drafts():
            if d.name == name:
                TR.write(d, _ROOMS_DIR)
    return level_loader.load(name, _ROOMS_DIR)


def aim_rule(sim, units, tick: int) -> tuple:
    """P8.5a: THE CHECK'S STAND-IN FOR THE DECODER'S AIM (``observe_rows``'s
    ``aims``: per unit id the uid the body aimed at), until P8.6a's decoder
    writes ``DeviceObservation.aimed``: from the second decision on, row i's
    soldier aims at choice ``(tick // DECIMATION + i) % n`` of its team's
    record uids (sorted) followed by one mate's uid (a mate's place: no
    target) -- so ``sees_target`` moves, and a record dropping out of the
    list or a mate's place reads none. Returns ``(aims, aimed)``: the dict
    the reference takes and the ``(rows,)`` int32 uids the device is fed
    (-1: none)."""
    aims, aimed = {}, np.full(len(units), -1, dtype=np.int32)
    if tick < DECIMATION:
        return aims, aimed
    for i, u in enumerate(units):
        if u is None:
            continue
        t = int(u.team)
        recs = sorted(int(x) for x in sim.team_picture.records(t))
        mates = sorted(int(m.id) for m in sim.units
                       if m.alive and not m.is_zombie and int(m.team) == t and m is not u)
        choices = recs + mates[:1]
        if not choices:
            continue
        uid = choices[(tick // DECIMATION + i) % len(choices)]
        aims[int(u.id)] = (uid, None)
        aimed[i] = uid
    return aims, aimed


def host_observation(session, ctx, T, S, aimed_out=None, aims_of=None) -> dict:
    """The env's served observation of ``session`` now, built by THE
    reference (``EnvRunner.row_units`` / ``row_flags`` over the session,
    ``obs_spec_v1.observe_rows`` with :func:`aim_rule`'s aims), one env's
    ``(U, ...)`` arrays; ``aimed_out`` (a list) receives the rows' aimed
    uids. ``aims_of`` (P8.6a, corpus (c)): ``(sim, units) -> (aims, aimed)``
    in place of :func:`aim_rule` (the policies' own ``aim_view``)."""
    import rl_env
    from simulation import obs_spec_v1 as o
    me = SimpleNamespace(session=session, layout=SimpleNamespace(n_slots=S, n_rows=T * S))
    shapes = o.shapes(1, T * S, S)
    out = {k: np.zeros(sh[1:], dtype=dt) for k, (sh, dt) in shapes.items()}
    units = rl_env.EnvRunner.row_units(me)
    aims, aimed = (aim_rule(session.sim, units, int(session.sim.tick)) if aims_of is None
                   else aims_of(session.sim, units))
    if aimed_out is not None:
        aimed_out.append(aimed)
    o.observe_rows(session.sim, ctx, units, out, aims=aims)
    # P8.4c: the four masks at the START of the decision tick, from the list
    # just built (rl_env.observe_v1's own call of build_masks)
    from simulation import action_spec_v1 as a
    tables = a.action_tables(session.sim, S)
    for k, (sh, dt) in a.mask_shapes(1, T * S, tables.n_entries, tables.n_kinds).items():
        out[k] = np.zeros(sh[1:], dtype=dt)
    a.build_masks(session.sim, units, out["list"][:, :, 0], out["list_mask"],
                  out["list_uid"], tables, DECIMATION, out)
    valid, in_match = rl_env.EnvRunner.row_flags(me)
    out["valid"][...] = valid
    out["in_match"][...] = in_match
    return out


def host_lof(sim):
    """THE reference line of fire (``line_of_fire.solid_in_way_many``) from
    every unit's centre tile to every unit's (``sim.units`` order, dead and
    turned included): ``(U, U)`` uint8, and the count of pairs a COVER decided
    (the physical walk clear, a live cover crossing the segment)."""
    from simulation import line_of_fire, vision
    cs = [(int(u.center_tile_x()), int(u.center_tile_y())) for u in sim.units]
    n = len(cs)
    out = np.zeros((n, n), dtype=np.uint8)
    pairs = np.array([(ay, ax, by, bx) for ax, ay in cs for bx, by in cs],
                     dtype=np.int32).reshape(-1, 4)
    walk = vision.rays_clear(sim, pairs, physical=True).reshape(n, n) if n else None
    for a in range(n):
        out[a] = np.array(line_of_fire.solid_in_way_many(sim, cs[a], cs), dtype=np.uint8)
    cover = int(np.count_nonzero(out.astype(bool) & walk)) if n else 0
    return out, cover


def play(name, level, spec, seed, setup=None, script=None, max_ticks=None,
         driver=None) -> Played:
    """Play the match on the host once (cached): its intents per tick, the
    owned host observation at every decision tick. Corpus (d): ``setup``
    (session) puts the hand state in at tick 0 (the twin gets the same);
    ``script`` (sim) issues hand intents before each tick (the tap records
    them with the controllers', so the twin is fed them); ``max_ticks`` ends
    the play early. ``driver`` (P8.6a, corpus (c)): a :class:`PolicyDriver`
    whose ``EnvRunner`` session is played (its ``policy`` teams decoding the
    driver's decisions; ``spec`` / ``seed`` are the session's)."""
    key = (name, seed)
    if key in _PLAYED:
        return _PLAYED[key]
    import rl_env
    from simulation.match_session import MatchSession
    from simulation.obs_spec_v1 import ObsContext
    t0 = time.perf_counter()
    s = (MatchSession(level, spec, seed=seed, breach_physics=bp) if driver is None
         else driver.session)
    if setup is not None:
        setup(s)
    sp = s.spec
    T, S = len(sp.teams), max(rl_env._team_slots(level))
    p = Played(name, level, sp if driver is not None else spec, seed, T, S, setup=setup)
    p.driver = driver
    ctx = ObsContext.build(s.sim, sp, S, n_teams=T)
    own = owned_obs()
    ref = RefBooks(s, T, S)                   # P8.7: the env's books, per named profile
    tap = cc.Tap()
    s.sim.attach_intent_log(tap)
    t_obs = 0.0

    def observe():
        nonlocal t_obs
        ts = time.perf_counter()
        aimed = []
        with AimTap(p.aim_cov) as at, ListTap(p.list_cov, ctx.caps):
            host = host_observation(s, ctx, T, S, aimed,
                                    None if driver is None else driver.aims_of)
        p.aimed[int(s.sim.tick)] = aimed[0]
        if driver is not None:
            driver.observed(p, s, host)
        count_windows(p, s, host)
        p.obs[int(s.sim.tick)] = owned_parts(host, own, False)
        count_list(p.list_cov, host, len(ctx.hand_rows))
        count_masks(p.mask_cov, host)
        p.aims[int(s.sim.tick)] = at.rec
        p.lof[int(s.sim.tick)], p.lof_cover[int(s.sim.tick)] = host_lof(s.sim)
        t_obs += time.perf_counter() - ts

    def see():
        from simulation import vision
        vs = vision.state_for(s.sim)          # the team picture's (slot 11) cached state
        p.vision[int(s.sim.tick)] = (dict(vs.seen_by), dict(vs.visible_by_team))
        picture_snapshot(p, s.sim, cov)

    cov = PictureCoverage()

    p.rule = str(getattr(s.sim, "sight", "cone"))
    p.uids = [int(u.id) for u in s.sim.units]
    alive0 = [bool(c.alive) for c in (s.sim.cover or ())]
    see()
    observe()
    ref.record(p, s, False)
    tap.take()                                # the setup's own calls are no intents
    while not s.is_terminal() and (max_ticks is None or int(s.sim.tick) < max_ticks):
        if script is not None:
            script(s.sim)
        if driver is not None:
            driver.decide(p, s)
        ref.before(s)
        if driver is None:
            s.tick()
        else:
            driver.tick(p, s)
        ref.after(s)
        ref.record(p, s, s.is_terminal())
        p.intents.append(tap.take() if driver is None else driver.feed(tap.take()))
        see()
        if int(s.sim.tick) % DECIMATION == 0:
            observe()
    s.sim.attach_intent_log(None)
    if driver is not None:
        driver.finish(p, s)
    p.coverage = dict(cov.counts)
    p.dmg_peak = cov.dmg_peak
    p.cover_alive = (alive0, [bool(c.alive) for c in (s.sim.cover or ())])
    p.end = int(s.sim.tick)
    p.terminal = bool(s.is_terminal())
    if p.terminal:
        r_ = s.result()
        p.result = (dpic_outcome(r_.outcome), -1 if r_.team is None else int(r_.team),
                    int(r_.ticks))
    p.book_cov = ref.coverage()
    _PLAYED[key] = p
    print(f"  played {name}@{seed}: {p.end} ticks, {len(p.obs)} decisions "
          f"({time.perf_counter() - t0:.1f} s, host observation {1000 * t_obs / len(p.obs):.1f} "
          f"ms per decision)")
    return p


def twin(p: Played):
    from simulation.match_session import MatchSession
    s = MatchSession(p.level, p.spec, seed=p.seed, breach_physics=bp)
    if p.setup is not None:
        p.setup(s)
    return s


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------
class Census:
    """Per owned (key, name): cells over the corpus not the default, the
    distinct values seen (capped)."""

    def __init__(self):
        self.nondefault: dict = {}
        self.values: dict = {}

    def add(self, parts: dict, valid) -> None:
        rows = np.asarray(valid, dtype=bool)
        for k, a in parts.items():
            a = np.asarray(a)
            sel = a if k[1] is None else a[rows]
            r = raw(sel).ravel()
            self.nondefault[k] = self.nondefault.get(k, 0) + int(np.count_nonzero(r))
            vs = self.values.setdefault(k, set())
            if len(vs) < 4:
                vs.update(int(x) for x in np.unique(r)[:4])

    def judge(self, label: str, strict: bool = True) -> None:
        for k in sorted(self.nondefault, key=str):
            n, vs = self.nondefault[k], self.values.get(k, set())
            print(f"    {label} {k[0]}{'' if k[1] is None else '/' + k[1]}: {n} non-default "
                  f"cells, {len(vs)}{'+' if len(vs) >= 4 else ''} distinct value(s)")
            if strict and len(vs) <= 1:
                fail(f"{label}: the owned column {k} is constant over its corpus "
                     f"(the gate on it would be vacuous)")


def _anchor_break(obs):
    """THE BREAK ``anchor``: the blocks read every unit one tile off."""
    from simulation import unit_store
    build = obs.build
    xq = unit_store.layout().column("x_q").offset

    def broken(store_ptr=None):
        rows = obs.c._rows.copy()
        rows[:, :, xq] += 1 << 16
        obs._break_rows = rows                     # alive until the launch ran
        build(store_ptr=int(rows.data.ptr))
    obs.build = broken


def _divisor_break(obs):
    """THE BREAK ``divisor`` (P8.1): the obs table's weapon block shifted one
    row, so a column's divisor (``rounds`` / ``mag_size``, ``range``, ...)
    is read from the NEIGHBOURING weapon row; must turn the gate red."""
    import cupy as cp
    off = {n: o for n, o, _ in dobs.table_layout()}
    ww = len(dobs.WEAPON_FIELDS)
    nw = int(bp_layout("OBS_TAB_N_WEAPONS"))
    t = obs.table_host.copy()
    blk = t[:, off["WEAPON"]:off["WEAPON"] + nw * ww].reshape(len(t), nw, ww)
    blk[:] = np.roll(blk, -1, axis=1)
    obs.table = cp.asarray(t)


def bp_layout(name: str) -> int:
    """A generated constant of the observation's layout (device_obs's own list)."""
    return dict(dobs._constants())[name]


def gate(label, played: list, census: Census | None = None, profile: str = "fireteam",
         books_census: dict | None = None) -> bool:
    """The conductor over twins of ``played`` (one level), each env fed its
    tapped intents; every env's owned device arrays equal to its reference
    at each of its decision ticks; P8.7: the books' whole state every tick
    and the served reward side at every step's end, under the named reward
    ``profile``. Returns True when equal."""
    own = owned_obs()
    sessions = [twin(p) for p in played]
    c = cd.DeviceConductor([s.sim for s in sessions], bp=bp)
    T = max(p.T for p in played)
    S = max(p.S for p in played)
    obs = dobs.DeviceObservation(c, [s.spec for s in sessions], S, DECIMATION, n_teams=T,
                                 picture_from_host=[e for e, p in enumerate(played)
                                                    if p.setup is not None],
                                 reward_profile=profile)
    _books_breaks(obs)
    _picture_breaks(obs)
    dec = None
    if any(p.driver is not None for p in played):    # P8.6a: corpus (c), decoder-driven
        if not all(p.driver is not None for p in played):
            raise ValueError("gate: a batch mixes policy runs and tap-driven matches")
        dec = decoder_for(obs, sessions)
    if BREAK == "anchor":                          # from the first decision after tick 0
        _anchor_break(obs)
    if BREAK == "stale":                           # nothing rebuilt after tick 0
        obs.skip_build = True
    if BREAK == "divisor":                         # rounds over the NEXT weapon row's magazine
        _divisor_break(obs)
    _sight_breaks(obs)
    if BREAK in dobs.AIM_BREAK_CODES:              # P8.4a: the lane breaks
        obs.aims_brk = dobs.AIM_BREAK_CODES[BREAK]
    if BREAK in dobs.LIST_BREAK_CODES:             # P8.4b: the sort's tie, a capacity
        obs.list_brk = dobs.LIST_BREAK_CODES[BREAK]
    if BREAK in dobs.MASK_BREAK_CODES:             # P8.4c: two gates swapped, the reach wide
        obs.masks_brk = dobs.MASK_BREAK_CODES[BREAK]
    if BREAK in dobs.WIN_BREAK_CODES:              # P8.5a: profile row, anchor, half, argmax
        obs.win_brk = dobs.WIN_BREAK_CODES[BREAK]
    import cupy as cp
    n_cmp = 0
    end = max(p.end for p in played)
    t0 = time.perf_counter()
    for k in range(end + 1):
        tick = int(c.sims[0].tick)
        # P8.7: the books every tick (an ended env's frozen at its final state)
        if not books_equal(label, obs, played, tick, profile, books_census):
            return False
        # the picture first: its break (early_picture) moves vision with it
        if not picture_equal(label, obs, played, tick):
            return False
        if not vision_equal(label, obs, played, tick):
            return False
        if tick % DECIMATION == 0 and not lof_equal(label, obs, played, tick):
            return False
        if tick % DECIMATION == 0 and not aims_equal(label, obs, played, tick):
            return False
        if tick % DECIMATION == 0:
            lf = obs.list_faults.get()
            if lf.any():
                fail(f"{label}: tick {tick}: list fault bits {lf.tolist()}")
                return False
            mf = obs.mask_faults.get()
            if mf.any():
                fail(f"{label}: tick {tick}: mask fault bits {mf.tolist()}")
                return False
            wf = obs.win_faults.get()
            if wf.any():
                fail(f"{label}: tick {tick}: window fault bits {wf.tolist()}")
                return False
        if tick % DECIMATION == 0:
            got = obs.host(own)
            for e, p in enumerate(played):
                if tick > p.end:
                    continue
                want = p.obs[tick]
                mine = {kk: (got[kk[0]][e] if kk[1] is None else
                             got[kk[0]][e][..., _plane_index(kk[0], kk[1])]
                             if kk[0] in ("vector", "list") else
                             got[kk[0]][e][:, _plane_index(kk[0], kk[1])])
                        for kk in want}
                if BREAK == "constant":
                    for kk in want:
                        if kk == ("near_u", "wall"):
                            want[kk] = np.zeros_like(want[kk])
                            mine[kk] = np.zeros_like(mine[kk])
                for kk, w in want.items():
                    g = mine[kk]
                    if w.shape != g.shape or not np.array_equal(raw(w), raw(g)):
                        idx = np.argwhere(raw(w) != raw(g)) if w.shape == g.shape else []
                        at = tuple(int(x) for x in idx[0]) if len(idx) else "shape"
                        fail(f"{label}: tick {tick}, env {e} ({p.name}@{p.seed}), {kk[0]}"
                             f"{'' if kk[1] is None else ' column/plane ' + kk[1]}: {len(idx)} "
                             f"value(s) differ, first at (row, ...) {at}: reference "
                             f"{w[tuple(idx[0])] if len(idx) else w.shape} device "
                             f"{g[tuple(idx[0])] if len(idx) else g.shape}"
                             + (f" (bits 0x{int(raw(w)[tuple(idx[0])]):x} vs "
                                f"0x{int(raw(g)[tuple(idx[0])]):x})" if len(idx) else ""))
                        if np.issubdtype(w.dtype, np.floating):
                            nan_w, nan_g = int(np.isnan(w).sum()), int(np.isnan(g).sum())
                            if nan_w or nan_g:
                                print(f"    NaNs: reference {nan_w}, device {nan_g}")
                        return False
                if census is not None:
                    census.add(want, want[("valid", None)] if ("valid", None) in want
                               else np.ones(p.T * p.S, bool))
                n_cmp += 1
        if k == end:
            break
        for e, p in enumerate(played):
            c.latch(p.intents[k] if k < len(p.intents) else (), env=e)
        if dec is not None:
            # P8.6a: the decisions handed over before the decision's first tick
            if not decoder_tick(label, c, obs, dec, played, tick, k):
                return False
            continue
        if (tick + 1) % DECIMATION == 0:
            # P8.5a: the decoder's stand-in -- each env's rows aimed as its
            # reference was at that decision (aim_rule), before the tick builds
            feed = np.full((len(played), obs.n_rows), -1, dtype=np.int32)
            for e, p in enumerate(played):
                a = p.aimed.get(tick + 1)
                if a is not None:
                    for i, uid in enumerate(a):
                        t_, s_ = divmod(i, p.S)
                        feed[e, t_ * S + s_] = uid
            obs.aimed[...] = cp.asarray(feed)
        c.tick()
    print(f"  {label}: {n_cmp} env-decisions equal (N = {len(played)}, {end} ticks, "
          f"{obs.builds} block launches, {1000 * (time.perf_counter() - t0) / max(1, end):.1f} "
          f"ms per tick incl. the gate's downloads)")
    return True


# ---------------------------------------------------------------------------
# P8.2: who sees whom, every tick; the line of fire over the cover table
# ---------------------------------------------------------------------------
#: Per sight rule: pair-ticks compared, the (match, observer, target) pairs
#: ever seen / ever not seen (o != t); per gate label the line-of-fire pairs
#: and those a cover decided. Filled by every gate; judged by part GATE.
VSTATS: dict = {}
LOFSTATS: dict = {}


def _sight_breaks(obs):
    """THE P8.2 BREAKS: ``cone`` (the cone's half-angle one degree wider in
    the obs table), ``early`` (who sees whom computed one slot too early:
    before the ruleset's tick end, the conversion), ``cover`` (every cover
    box shifted one tile in y)."""
    import cupy as cp
    if BREAK == "cone":
        from config import CFG
        from simulation import vision
        off = {n: o for n, o, _ in dobs.table_layout()}["CONE_COS_BITS"]
        t = obs.table.get()
        t[:, off] = dobs._f64_bits(vision.cos_threshold(
            float(CFG.onephase.vision_cone_half_deg) + 1.0))
        obs.table = cp.asarray(t)
        obs.vision()                               # tick 0 again, broken
    if BREAK == "early":
        obs.early_vision = True
    if BREAK == "cover":
        cov = obs.cover_host.copy()
        for f in ("Y0_BITS", "Y1_BITS"):
            i = dobs.COVER_FIELDS.index(f)
            vals = np.ascontiguousarray(cov[..., i]).view(np.float64) + 1.0
            cov[..., i] = vals.view(np.int64)
        obs.cover = cp.asarray(cov)


def _teams_of_slot(obs):
    from simulation.factions import ZOMBIE_TEAM
    return list(range(obs.T)) + [int(ZOMBIE_TEAM)]


def vision_equal(label, obs, played, tick) -> bool:
    """Every env's device who-sees-whom equal to its reference's VisionState
    at ``tick`` (``seen_by`` and ``visible_by_team``, every pair), no fault;
    the per-rule census filled."""
    sees = obs.sees.get()
    vis = obs.visible.get()
    faults = obs.vision_faults.get()
    slots = _teams_of_slot(obs)
    for e, p in enumerate(played):
        if tick > p.end:
            continue
        if int(faults[e]):
            fail(f"{label}: tick {tick}, env {e}: vision fault bits {int(faults[e])}")
            return False
        uids = obs.c.uids[e]
        if uids != p.uids:
            fail(f"{label}: env {e}: the store's rows are not the reference's units")
            return False
        want_seen, want_vis = p.vision[tick]
        bad_teams = [t for t in want_vis if t not in slots]
        if bad_teams:
            fail(f"{label}: tick {tick}, env {e}: reference teams {bad_teams} have no slot")
            return False
        got_seen, got_vis = dobs.vision_state(sees[e], vis[e], uids, slots)
        n = len(uids)
        ref = np.zeros((n, n), dtype=bool)
        row = {u: r for r, u in enumerate(uids)}
        for tid, obsv in want_seen.items():
            for oid in obsv:
                ref[row[oid], row[tid]] = True
        dev = sees[e][:n, :n].astype(bool)
        if got_seen != want_seen or got_vis != want_vis or not np.array_equal(ref, dev):
            idx = np.argwhere(ref != dev)
            at = (f"pair (observer {uids[idx[0][0]]}, target {uids[idx[0][1]]}): reference "
                  f"{bool(ref[tuple(idx[0])])} device {bool(dev[tuple(idx[0])])}"
                  if len(idx) else "the visible sets")
            fail(f"{label}: tick {tick}, env {e} ({p.name}@{p.seed}, rule {p.rule}): who sees "
                 f"whom differs on {len(idx)} pair(s), first {at}; visible reference "
                 f"{want_vis} device {got_vis}")
            return False
        st = VSTATS.setdefault(p.rule, {"pair_ticks": 0, "true": set(), "false": set(),
                                        "seen_ticks": 0})
        st["pair_ticks"] += n * (n - 1)
        st["seen_ticks"] += int(np.count_nonzero(ref))
        key = (p.name, p.seed, e)
        off = ~np.eye(n, dtype=bool)
        st["true"].update((key, int(a), int(b)) for a, b in np.argwhere(ref & off))
        st["false"].update((key, int(a), int(b)) for a, b in np.argwhere(~ref & off))
    return True


def lof_equal(label, obs, played, tick) -> bool:
    """``line_of_fire.solid_in_way`` on the card over every (unit, unit)
    pair equal to the reference at a decision tick; the pairs a cover decided
    counted."""
    out, faults = obs.lof_pairs()
    out, faults = out.get(), faults.get()
    for e, p in enumerate(played):
        if tick > p.end:
            continue
        if int(faults[e]):
            fail(f"{label}: tick {tick}, env {e}: line-of-fire fault bits {int(faults[e])}")
            return False
        want = p.lof[tick]
        n = want.shape[0]
        got = out[e][:n, :n]
        if not np.array_equal(want, got):
            idx = np.argwhere(want != got)
            a, b = (int(x) for x in idx[0])
            fail(f"{label}: tick {tick}, env {e} ({p.name}@{p.seed}): solid_in_way differs on "
                 f"{len(idx)} pair(s), first (unit {p.uids[a]} -> unit {p.uids[b]}): reference "
                 f"{int(want[a, b])} device {int(got[a, b])}")
            return False
        st = LOFSTATS.setdefault(label, {"pairs": 0, "blocked": 0, "cover": 0, "covers": 0})
        st["pairs"] += n * n
        st["blocked"] += int(np.count_nonzero(want))
        st["cover"] += int(p.lof_cover[tick])
        st["covers"] = int(obs.cover_count_host[e])
    return True


def judge_vision() -> None:
    """The census of who sees whom per rule and of the line of fire: a rule
    whose pairs were never both seen and unseen, or a corpus where no cover
    ever decided a pair, FAILS (a vacuous gate)."""
    for rule, st in sorted(VSTATS.items()):
        both = len(st["true"] & st["false"])
        print(f"    vision rule {rule}: {st['pair_ticks']} pair-ticks compared, "
              f"{st['seen_ticks']} seen; {len(st['true'])} pairs ever seen, "
              f"{len(st['false'])} ever not seen, {both} both")
        if not st["true"] or not st["false"] or both == 0:
            fail(f"vision rule {rule}: no pair was both seen and unseen over the corpus "
                 f"(the gate would be vacuous)")
    for rule in ("cone", "all_round"):
        if rule not in VSTATS:
            fail(f"vision: no corpus match runs the {rule!r} rule")
    for label, st in sorted(LOFSTATS.items()):
        print(f"    line of fire {label}: {st['pairs']} pairs compared, {st['blocked']} "
              f"blocked, {st['cover']} by a cover ({st['covers']} cover entities)")
    if not any(st["cover"] > 0 for st in LOFSTATS.values()):
        fail("line of fire: no cover decided any pair over the corpus (vacuous)")


# ---------------------------------------------------------------------------
# P8.4a: the lanes, aim_of, the line of fire, ticks_to_within -- compared with
# the REFERENCE'S OWN intermediate values, taken by a test-side tap of the
# builder's calls (the builder itself unchanged)
# ---------------------------------------------------------------------------
#: The coverage cases the intermediates' corpus must reach (each > 0, else
#: FAIL): a team's lanes crossing one of its mates (the cell under a mate's
#: centre held by another owner or several), a cell held by several owners, a
#: line of fire blocked by a wall, by a cover crate (the walk clear), by a
#: mate's body, a clear one; aim_of finding nobody, a mate, a unit of another
#: side; ticks_to_within 0, between, at the cap.
AIM_CASES = ("lane_crosses_mate", "lane_several", "lof_wall", "lof_cover", "lof_mate",
             "lof_clear", "aim_none", "aim_mate", "aim_other", "ttw_zero", "ttw_between",
             "ttw_cap")
#: Per gate label: decisions whose intermediates compared equal, values compared.
AIMSTATS: dict = {}


class AimTap:
    """The reference's own P8.4a intermediates of ONE host observation
    (``observe_rows``), taken by wrapping the builder's calls for its
    duration and restoring them after (the builder is not changed): per team
    the fire lanes it read (``TeamKnowledge.lanes``), per (team, mate) the
    own lane (``lane_of``: grid == uid), per (team, unit) ``aim_of``'s uid,
    per (team, soldier, seen enemy) ``solid_in_way`` / ``mate_in_way`` and
    the ``ticks_to_within`` of ``_enemy_geometry``. A call is attributed to
    the soldier ``build_list`` is building and to every seen enemy whose
    recorded centre is the call's end point (equal geometry, equal value).
    ``cov``: the coverage counts (:data:`AIM_CASES`), added to."""

    def __init__(self, cov):
        self.cov = cov
        self.rec = {"lanes": {}, "own": {}, "aim": {}, "solid": {}, "mate": {}, "ttw": {}}
        self.know = None                      # build_list's view and soldier
        self.me = None
        self.geo = None                       # _enemy_geometry's (dx, dy)
        self._saved = []

    def _patch(self, obj, name, new):
        self._saved.append((obj, name, obj.__dict__[name] if isinstance(obj, type)
                            else getattr(obj, name)))
        setattr(obj, name, new)

    def _targets(self, x, y):
        """The seen enemies of the soldier's team whose record sits at (x, y)."""
        k = self.know
        return [uid for uid in k.seen if tuple(int(v) for v in k.records[uid].centre) == (x, y)]

    def __enter__(self):
        from simulation import line_of_fire, turning, vision
        from simulation import obs_spec_v1 as o
        tap, rec = self, self.rec
        orig_build, orig_aim = o.build_list, o.aim_of
        orig_solid, orig_mate = line_of_fire.solid_in_way, line_of_fire.mate_in_way
        orig_geo, orig_ttw = o._enemy_geometry, turning.ticks_to_within
        orig_lanes, orig_lane_of = o.TeamKnowledge.lanes, o.TeamKnowledge.lane_of

        def build_list(know, u, *a, **kw):
            tap.know, tap.me = know, int(u.id)
            try:
                return orig_build(know, u, *a, **kw)
            finally:
                tap.know = tap.me = None

        def aim_of(know, ctx, u, me_id=-1):
            res = orig_aim(know, ctx, u, me_id)
            key = (int(know.team), int(u.id))
            if key not in rec["aim"]:
                rec["aim"][key] = int(res[1])
                hit = int(res[1])
                tap.cov["aim_none" if hit < 0 else "aim_mate" if hit in know.mates
                        else "aim_other"] += 1
            return res

        def solid_in_way(sim, a, b):
            res = orig_solid(sim, a, b)
            if tap.know is not None:
                ax, ay, bx, by = int(a[0]), int(a[1]), int(b[0]), int(b[1])
                walk = bool(vision.rays_clear(sim, [(ay, ax, by, bx)], physical=True)[0])
                for euid in tap._targets(bx, by):
                    rec["solid"][(int(tap.know.team), tap.me, euid)] = bool(res)
                tap.cov["lof_wall" if not walk else "lof_cover" if res else "lof_clear"] += 1
            return res

        def mate_in_way(a, b, mates, clear_tiles, fp):
            res = orig_mate(a, b, mates, clear_tiles, fp)
            if tap.know is not None:
                for euid in tap._targets(int(b[0]), int(b[1])):
                    rec["mate"][(int(tap.know.team), tap.me, euid)] = bool(res)
                tap.cov["lof_mate"] += int(bool(res))
            return res

        def enemy_geometry(e, aimer, u, dx, dy, ctx):
            tap.geo = (int(dx), int(dy))
            try:
                return orig_geo(e, aimer, u, dx, dy, ctx)
            finally:
                tap.geo = None

        def ticks_to_within(unit, target_q, a, w, dw, tol_q, cap):
            res = orig_ttw(unit, target_q, a, w, dw, tol_q, cap)
            if tap.know is not None and tap.geo is not None:
                cx, cy = int(unit.center_tile_x()), int(unit.center_tile_y())
                for euid in tap._targets(cx + tap.geo[0], cy + tap.geo[1]):
                    rec["ttw"][(int(tap.know.team), tap.me, euid)] = int(res)
                tap.cov["ttw_zero" if res == 0 else "ttw_cap" if res >= int(cap)
                        else "ttw_between"] += 1
            return res

        def lanes(know):
            grid = orig_lanes.fget(know)
            t = int(know.team)
            if t not in rec["lanes"]:
                rec["lanes"][t] = np.array(grid, dtype=np.int32)
                cell = know.lane_dials[0]
                tap.cov["lane_several"] += int(np.count_nonzero(grid == -2))
                for uid, m in know.mates.items():
                    g = int(grid[int(m.center_tile_y()) // cell, int(m.center_tile_x()) // cell])
                    tap.cov["lane_crosses_mate"] += int(g not in (-1, uid))
            return grid

        def lane_of(know, uid):
            grid = orig_lane_of(know, uid)
            rec["own"].setdefault((int(know.team), int(uid)),
                                  np.ascontiguousarray(grid == int(uid)).astype(np.uint8))
            return grid

        self._patch(o, "build_list", build_list)
        self._patch(o, "aim_of", aim_of)
        self._patch(line_of_fire, "solid_in_way", solid_in_way)
        self._patch(line_of_fire, "mate_in_way", mate_in_way)
        self._patch(o, "_enemy_geometry", enemy_geometry)
        self._patch(turning, "ticks_to_within", ticks_to_within)
        self._patch(o.TeamKnowledge, "lanes", property(lanes))
        self._patch(o.TeamKnowledge, "lane_of", lane_of)
        return self

    def __exit__(self, *exc):
        for obj, name, old in reversed(self._saved):
            setattr(obj, name, old)
        self._saved = []
        return False


def aims_equal(label, obs, played, tick) -> bool:
    """Every env's P8.4a intermediates equal to the reference's own (the
    tap's) at a decision tick: each team grid the reference read, each own
    lane, each aim_of answer, each (soldier, seen enemy)'s line of fire and
    ticks_to_within; no fault; every value the reference asked is asked on
    the card."""
    grid, own, aim = obs.lane_grid.get(), obs.lane_own.get(), obs.aim.get()
    lof, ttw, faults = obs.lof.get(), obs.ttw.get(), obs.aim_faults.get()
    rowmap = obs.rowmap.get()
    codes = dobs.AIM_CODES

    def bad(e, p, what, want, got):
        fail(f"{label}: tick {tick}, env {e} ({p.name}@{p.seed}): {what}: reference {want!r} "
             f"device {got!r}")
        return False

    for e, p in enumerate(played):
        if tick > p.end:
            continue
        if int(faults[e]):
            return bad(e, p, "aim fault bits", 0, int(faults[e]))
        ref = p.aims[tick]
        row = {u: r for r, u in enumerate(obs.c.uids[e])}
        slot = {int(rowmap[e, r]): r for r in range(obs.n_rows) if int(rowmap[e, r]) >= 0}
        n = 0
        for t, g in ref["lanes"].items():
            dg = grid[e, t]
            if dg.shape != g.shape or not np.array_equal(dg, g):
                at = (tuple(int(x) for x in np.argwhere(dg != g)[0]) if dg.shape == g.shape
                      else "shape")
                return bad(e, p, f"team {t}'s fire lanes at cell {at}",
                           g[at] if at != "shape" else g.shape,
                           dg[at] if at != "shape" else dg.shape)
            n += g.size
        for (t, uid), m in ref["own"].items():
            dm = own[e, row[uid]]
            if not np.array_equal(dm, m):
                at = tuple(int(x) for x in np.argwhere(dm != m)[0])
                return bad(e, p, f"team {t}, unit {uid}'s own lane at cell {at}", int(m[at]),
                           int(dm[at]))
            n += m.size
        for (t, uid), hit in ref["aim"].items():
            got = int(aim[e, t, row[uid]])
            if got != hit:
                return bad(e, p, f"aim_of(team {t}, unit {uid})", hit, got)
            n += 1
        for name, bit in (("solid", codes["LOF_SOLID"]), ("mate", codes["LOF_MATE"])):
            for (t, me, euid), want in ref[name].items():
                v = int(lof[e, slot[row[me]], row[euid]])
                if not v & codes["LOF_ASKED"] or bool(v & bit) != want:
                    return bad(e, p, f"{name}_in_way(team {t}, soldier {me} -> enemy {euid}) "
                               f"[bits {v}]", want, bool(v & bit))
                n += 1
        for (t, me, euid), want in ref["ttw"].items():
            got = int(ttw[e, slot[row[me]], row[euid]])
            if got != want:
                return bad(e, p, f"ticks_to_within(team {t}, soldier {me} -> enemy {euid})",
                           want, got)
            n += 1
        st = AIMSTATS.setdefault(label, {"decisions": 0, "values": 0, "aims": 0, "lofs": 0,
                                         "ttws": 0, "lanes": 0})
        st["decisions"] += 1
        st["values"] += n
        st["aims"] += len(ref["aim"])
        st["lofs"] += len(ref["solid"])
        st["ttws"] += len(ref["ttw"])
        st["lanes"] += len(ref["lanes"])
    return True


def judge_aims(played) -> None:
    """The intermediates' census: per gate the decisions and values
    compared; per case its count over the corpus, each > 0 (a case never
    reached FAILS)."""
    for label, st in sorted(AIMSTATS.items()):
        print(f"    aims {label}: {st['decisions']} env-decisions equal, {st['values']} values "
              f"({st['lanes']} team lanes, {st['aims']} aim_of, {st['lofs']} lines of fire, "
              f"{st['ttws']} ticks_to_within)")
    total = {k: 0 for k in AIM_CASES}
    for p in played:
        for k, n in p.aim_cov.items():
            total[k] += n
    print("    aims coverage: " + ", ".join(f"{k} {n}" for k, n in total.items()))
    for k, n in total.items():
        if n == 0:
            fail(f"aims coverage: the case {k!r} is never reached over the corpus (vacuous)")


# ---------------------------------------------------------------------------
# P8.4b: the list's coverage (its columns are compared as served columns)
# ---------------------------------------------------------------------------
#: The cases the list's gate must reach (each > 0 over the corpus, else FAIL):
#: an entry of every block (the record entry, a grenade, a charge), a door in
#: the use verb's reach and one out of it, a block whose candidates overflow
#: its capacity (the host cuts after the sort; the card stops at the cap).
LIST_CASES = ("mate", "seen", "record", "grenade", "charge", "door_in_reach",
              "door_out_of_reach", "overflow")
_BLOCKS = ("mates", "seen", "records", "objects", "doors")


class ListTap:
    """Counts, for ONE host observation, every block whose candidates
    (``obs_spec_v1._candidates``, wrapped for the duration and restored)
    outnumber its capacity: ``cov["overflow"]`` and ``cov["overflow_<block>"]``
    (the latter informative). The builder is not changed."""

    def __init__(self, cov, caps):
        self.cov, self.caps = cov, tuple(caps)
        self._saved = None

    def __enter__(self):
        from simulation import obs_spec_v1 as o
        orig, tap = o._candidates, self

        def candidates(know, u):
            res = orig(know, u)
            for name, block, cap in zip(_BLOCKS, res, tap.caps):
                if len(block) > cap:
                    tap.cov["overflow"] += 1
                    k = f"overflow_{name}"
                    tap.cov[k] = tap.cov.get(k, 0) + 1
            return res

        self._saved = orig
        o._candidates = candidates
        return self

    def __exit__(self, *exc):
        from simulation import obs_spec_v1 as o
        o._candidates = self._saved
        return False


def count_list(cov, host, n_hand) -> None:
    """The entry cases of one host observation (the served arrays)."""
    from simulation import obs_spec_v1 as o
    col = {f.name: f.pos for f in o.ENTRY_COLUMNS}
    lst, mask = host["list"], host["list_mask"].astype(bool)
    typ = lst[..., col["type"]]
    on = lambda t: mask & (typ == t)  # noqa: E731
    cov["mate"] += int(on(o.TYPE_MATE).sum())
    cov["seen"] += int(on(o.TYPE_SEEN).sum())
    cov["record"] += int(on(o.TYPE_RECORD).sum())
    kind = lst[..., col["obj_kind"]]
    cov["grenade"] += int((on(o.TYPE_OBJECT) & (kind >= 0) & (kind < n_hand)).sum())
    cov["charge"] += int((on(o.TYPE_OBJECT) & (kind >= n_hand)).sum())
    reach = lst[..., col["in_reach"]]
    cov["door_in_reach"] += int((on(o.TYPE_DOOR) & (reach == 1.0)).sum())
    cov["door_out_of_reach"] += int((on(o.TYPE_DOOR) & (reach == 0.0)).sum())


def judge_list(played) -> None:
    total = {}
    for p in played:
        for k, n in p.list_cov.items():
            total[k] = total.get(k, 0) + n
    print("    list coverage: " + ", ".join(f"{k} {n}" for k, n in sorted(total.items())))
    for k in LIST_CASES:
        if not total.get(k):
            fail(f"list coverage: the case {k!r} is never reached over the corpus (vacuous)")


# ---------------------------------------------------------------------------
# P8.4c: the masks' coverage (the masks are compared as served keys)
# ---------------------------------------------------------------------------
#: The bits the spec FIXES (never both set and clear, by the reference's own
#: rule): stand, the first trigger option and "nothing" always legal, the
#: flashlight always masked (reserved, design 2.6).
MASK_FIXED = {("mask_feet", _aspec.FEET_STAND), ("mask_trigger", 0),
              ("mask_hand", _aspec.HAND_NOTHING), ("mask_hand", _aspec.HAND_FLASHLIGHT)}
#: The named cases (each > 0 else FAIL) beside the per-bit "set and clear".
MASK_CASES = ("aim_keep_only", "aim_entry_set", "aim_entry_clear", "use_door_set",
              "default_row")


def count_masks(cov, host) -> None:
    """Per mask bit of one host observation: the rows it is set and clear
    on (``cov[(key, bit)] = [set, clear]``; the aim's entry positions pooled
    as ``("mask_aim", "entry")``), and the named cases."""
    from simulation import action_spec_v1 as a
    for key in a.MASK_KEYS:
        m = np.asarray(host[key]).astype(bool)
        n_fixed = a.AIM_TOKENS if key == "mask_aim" else m.shape[1]
        for j in range(n_fixed):
            c = cov.setdefault((key, j), [0, 0])
            c[0] += int(m[:, j].sum())
            c[1] += int((~m[:, j]).sum())
        if key == "mask_aim":
            ent = m[:, a.AIM_TOKENS:]
            c = cov.setdefault((key, "entry"), [0, 0])
            c[0] += int(ent.sum())
            c[1] += int((~ent).sum())
    aim = np.asarray(host["mask_aim"]).astype(bool)
    keep = aim[:, a.AIM_KEEP] & ~aim[:, a.AIM_WALK]
    valid = np.asarray(host["valid"]).astype(bool)
    cov["aim_keep_only"] = cov.get("aim_keep_only", 0) + int((keep & valid).sum())
    ent = aim[:, a.AIM_TOKENS:][valid & ~keep]
    cov["aim_entry_set"] = cov.get("aim_entry_set", 0) + int(ent.any(axis=1).sum())
    cov["aim_entry_clear"] = cov.get("aim_entry_clear", 0) + int((~ent).any(axis=1).sum())
    hand = np.asarray(host["mask_hand"]).astype(bool)
    cov["use_door_set"] = cov.get("use_door_set", 0) + int(hand[:, a.HAND_USE].sum())
    cov["default_row"] = cov.get("default_row", 0) + int((~valid).sum())


def judge_masks(played) -> None:
    total = {}
    for p in played:
        for k, n in p.mask_cov.items():
            if isinstance(n, list):
                t = total.setdefault(k, [0, 0])
                t[0] += n[0]
                t[1] += n[1]
            else:
                total[k] = total.get(k, 0) + n
    bits = sorted((k for k in total if isinstance(total[k], list)), key=str)
    print("    mask coverage (set / clear): " + ", ".join(
        f"{k[0][5:]}[{k[1]}] {total[k][0]}/{total[k][1]}" for k in bits))
    print("    mask cases: " + ", ".join(f"{k} {total.get(k, 0)}" for k in MASK_CASES))
    for k in bits:
        if k in MASK_FIXED:
            continue
        if not total[k][0] or not total[k][1]:
            fail(f"mask coverage: {k[0]} bit {k[1]} is never both set and clear over the "
                 f"corpus (set {total[k][0]}, clear {total[k][1]}): the gate on it is vacuous")
    for k in MASK_CASES:
        if not total.get(k):
            fail(f"mask coverage: the case {k!r} is never reached over the corpus (vacuous)")


# ---------------------------------------------------------------------------
# P8.5a: the windows' coverage, counted off the reference
# ---------------------------------------------------------------------------
#: The cases the windows' corpus must reach (each > 0, else FAIL): a watched
#: tile, a tile that sees the aimed target, a near window hanging over each
#: edge of the map, a far window cell on a block that straddles the map's
#: edge (part pad, part map), a far window cell off the block grid.
WIN_CASES = ("watched", "sees_target", "near_over_left", "near_over_right", "near_over_top",
             "near_over_bottom", "far_block_straddles", "far_off_grid", "aimed_found",
             "aimed_not_found")
#: The float planes whose served cells past the squash's knee are counted.
_KNEE_KEYS = ("near_f", "far_f")


def count_windows(p, session, host) -> None:
    """Add one host observation's window coverage to ``p.win_cov`` and its
    cells past the squash's knee (|x| > 1, the log path) to ``p.knee``."""
    import rl_env
    from simulation import obs_spec_v1 as o
    sim = session.sim
    me = SimpleNamespace(session=session, layout=SimpleNamespace(n_slots=p.S, n_rows=p.T * p.S))
    units = rl_env.EnvRunner.row_units(me)
    h, w = sim.gmap.solid.shape
    hb, wb = -(-h // o.BLOCK), -(-w // o.BLOCK)
    cov = p.win_cov
    nu = host["near_u"]
    cov["watched"] += int(np.count_nonzero(nu[:, o.NEAR_U_PLANES.index("watched")]))
    cov["sees_target"] += int(np.count_nonzero(nu[:, o.NEAR_U_PLANES.index("sees_target")]))
    straddle_y = h % o.BLOCK != 0
    straddle_x = w % o.BLOCK != 0
    for i, u in enumerate(units):
        if u is None:
            continue
        cx, cy = int(u.center_tile_x()), int(u.center_tile_y())
        cov["near_over_left"] += int(cx - o.HALF < 0)
        cov["near_over_right"] += int(cx + o.HALF >= w)
        cov["near_over_top"] += int(cy - o.HALF < 0)
        cov["near_over_bottom"] += int(cy + o.HALF >= h)
        by0, bx0 = cy // o.BLOCK - o.HALF, cx // o.BLOCK - o.HALF
        bi = np.arange(by0, by0 + o.WIN)
        bj = np.arange(bx0, bx0 + o.WIN)
        on_i = (bi >= 0) & (bi < hb)
        on_j = (bj >= 0) & (bj < wb)
        cov["far_off_grid"] += int(o.WIN * o.WIN - on_i.sum() * on_j.sum())
        n = 0
        if straddle_y and (hb - 1) in bi:
            n += int(on_j.sum())
        if straddle_x and (wb - 1) in bj:
            n += int(on_i.sum())
        cov["far_block_straddles"] += n
        if int(p.aimed[max(p.aimed)][i]) >= 0:
            # found: build_list marked the aimed entry (prev_aim, col 66) at a
            # place past the mates' block, so sees_target is FROM the window
            # TO its recorded centre; else none (not listed, or a mate)
            ent = host["list"][i, p.S - 1:p.S - 1 + o.CAPS_FIXED[0] + o.CAPS_FIXED[1], 66]
            cov["aimed_found" if bool(np.any(ent)) else "aimed_not_found"] += 1
    for key in _KNEE_KEYS:
        names = o.NEAR_F_PLANES if key == "near_f" else o.FAR_F_PLANES
        a = np.asarray(host[key], dtype=np.float32)
        for k, name in enumerate(names):
            if name not in o.PHYSICS_PLANES:
                continue
            p.knee[(key, name)] = p.knee.get((key, name), 0) + int(
                np.count_nonzero(np.abs(a[:, k]) > 1.0))


def judge_windows(played) -> None:
    total = {k: 0 for k in WIN_CASES}
    knee = {}
    for p in played:
        for k in WIN_CASES:
            total[k] += p.win_cov[k]
        for k, n in p.knee.items():
            knee[k] = knee.get(k, 0) + n
    print("    window cases: " + ", ".join(f"{k} {total[k]}" for k in WIN_CASES))
    print("    cells past the squash's knee (|x| > 1): " + ", ".join(
        f"{k[0]}/{k[1]} {n}" for k, n in sorted(knee.items())))
    for k in WIN_CASES:
        if not total[k]:
            fail(f"window coverage: the case {k!r} is never reached over the corpus (vacuous)")
    for name in ("temperature", "fire", "poison", "teargas", "smoke", "wind_x", "wind_y"):
        if not (knee.get(("near_f", name), 0) or knee.get(("far_f", name), 0)):
            print(f"    NOTE: plane {name} never passes the squash's knee over the corpus")
    if not any(knee.values()):
        fail("window coverage: no float plane passes the squash's knee (the log path is never "
             "served)")


# ---------------------------------------------------------------------------
# P8.3: the device team picture, its WHOLE state every tick
# ---------------------------------------------------------------------------
#: The coverage cases the picture's corpus must reach (each > 0, else FAIL).
PICTURE_CASES = (
    "record_created", "record_refreshed_after_gap", "record_expired", "death_seen",
    "death_unseen", "conversion_with_record", "grenade_tracked", "grenade_lost",
    "grenade_reseen", "grenade_survives_compaction", "charge_tracked", "damage_added",
    "damage_dropped")
#: Per gate label: ticks whose whole picture compared equal, env-ticks.
PICSTATS: dict = {}


class PictureCoverage:
    """The coverage of the picture's cases, read off the REFERENCE picture
    tick by tick (the device equals it every tick, so its coverage is the
    same): records created, refreshed after a gap, expired; deaths seen and
    unseen; a conversion with a record of the marine; grenades tracked, lost,
    re-seen, carried through a compaction; a charge tracked; the damage
    window adding and dropping."""

    def __init__(self):
        self.counts = {k: 0 for k in PICTURE_CASES}
        self.prev = None                    # the previous tick's host form
        self.prev_seen = None               # team -> set of seen object ids
        self.prev_tracked = None            # team -> {object id: list position}
        self.dmg_peak = 0

    def add(self, sim, form) -> None:
        pic = sim.team_picture
        t = form["tick"]
        objs = list(sim.projectiles) + list(sim.planted_charges)
        pos = {id(o): i for i, o in enumerate(objs)}
        n_g = len(sim.projectiles)
        seen = {tm: {id(o) for o, _ in pic.seen_now[tm]} for tm in pic.teams}
        tracked = {tm: {id(o): pos[id(o)] for o in pic._objects[tm]} for tm in pic.teams}
        for tm in pic.teams:
            n_dmg = sum(len(q) for q in pic._damage[tm].values())
            self.dmg_peak = max(self.dmg_peak, n_dmg)
        c = self.counts
        for tm in pic.teams:
            for i in tracked[tm].values():
                c["grenade_tracked" if i < n_g else "charge_tracked"] += 1
        prev = self.prev
        if prev is not None and t == prev["tick"] + 1:
            for tm in pic.teams:
                pr, cr = prev["records"][tm], form["records"][tm]
                for uid, r in cr.items():
                    if uid not in pr:
                        c["record_created"] += 1
                    elif r[11] == t and pr[uid][11] < t - 1:
                        c["record_refreshed_after_gap"] += 1
                for uid, r in pr.items():
                    if uid in cr:
                        continue
                    if r[11] == t - 1:
                        c["death_seen"] += 1
                    elif t - r[11] > pic.memory_ticks:
                        c["record_expired"] += 1
                for oid, i in tracked[tm].items():
                    if i >= n_g:
                        continue
                    if oid in self.prev_seen[tm] and oid not in seen[tm]:
                        c["grenade_lost"] += 1
                    if (oid in seen[tm] and oid not in self.prev_seen[tm]
                            and oid in self.prev_tracked[tm]):
                        c["grenade_reseen"] += 1
                    j = self.prev_tracked[tm].get(oid)
                    if j is not None and j != i:
                        c["grenade_survives_compaction"] += 1
                pd, cd_ = prev["damage"][tm], form["damage"][tm]
                c["damage_added"] += sum(1 for q in cd_.values() for tk, _ in q if tk == t)
                c["damage_dropped"] += sum(1 for q in pd.values() for tk, _ in q
                                           if tk <= t - pic.window_ticks)
            for uid, dt in form["death_tick"].items():
                if dt != t or uid in prev["death_tick"]:
                    continue
                if uid in form["turned"] and uid not in prev["turned"]:
                    continue                                 # a conversion, below
                if any(uid in form["records"][tm] and form["records"][tm][uid][11] < t - 1
                       for tm in pic.teams):
                    c["death_unseen"] += 1
            for uid in form["turned"]:
                if uid in prev["turned"] or uid not in prev["motion"]:
                    continue                                 # not a marine turning now
                if any(uid in prev["records"][tm] and prev["records"][tm][uid][1] == 0
                       for tm in pic.teams):
                    c["conversion_with_record"] += 1
        self.prev, self.prev_seen, self.prev_tracked = form, seen, tracked


def picture_snapshot(p, sim, cov) -> None:
    """The reference picture at this tick (slot 11 ran): its normal form,
    its damage window's sums per (team, unit) (``damage_last_s``), the
    coverage."""
    from simulation import device_picture as dpic
    pic = sim.team_picture
    form = dpic.host_form(pic, sim)
    p.picture[int(sim.tick)] = form
    uids = [int(u.id) for u in sim.units]
    sums = np.zeros((len(pic.teams), len(uids)), dtype=np.float64)
    for t in pic.teams:
        for r, uid in enumerate(uids):
            sums[t, r] = pic.damage_last_s(t, uid)
    p.dsum[int(sim.tick)] = sums
    cov.add(sim, form)


def _picture_breaks(obs) -> None:
    """THE P8.3 BREAKS: ``early_picture`` (vision and the picture before the
    conversion), ``lifetime`` (a record kept one tick longer), ``dmg_order``
    (the damage window fed the tick's hits in reverse emission order),
    ``serial_row`` (the object serial replaced by the row index), ``dmg_sum``
    (the window's sums taken in reverse order)."""
    from simulation import device_picture as dpic
    if BREAK == "early_picture":
        obs.early_picture = True
    if BREAK in dpic.BREAK_CODES and BREAK:
        obs.picture.brk = dpic.BREAK_CODES[BREAK]


def picture_equal(label, obs, played, tick) -> bool:
    """Every env's device picture equal to its reference's at ``tick``,
    field by field (records, bars, turned, death ticks, motion, last hits,
    the damage window, the object tracker by object position, the id
    counters, seen_now), and the damage window's sums raw-bit equal."""
    from simulation import device_picture as dpic
    forms = obs.picture.forms()
    sums = obs.picture.damage_sums().get()
    if BREAK == "dmg_sum":                               # the order turned round
        import cupy as cp
        out = cp.zeros_like(cp.asarray(sums))
        obs.bp.obs_picture_damage_sums(int(obs.N), int(obs.c.store.device_ptr), int(obs.c.U),
                                       int(obs.c.store.shape[2]), int(obs.T),
                                       int(obs.picture.state.data.ptr), int(obs.picture.W),
                                       int(out.data.ptr), True)
        sums = out.get()
    for e, p in enumerate(played):
        if tick > p.end:
            continue
        want = p.picture[tick]
        got = forms[e]
        d = dpic.first_difference(want, got)
        if d is not None:
            path, a, b = d
            fail(f"{label}: tick {tick}, env {e} ({p.name}@{p.seed}): the team picture differs "
                 f"at {'/'.join(str(x) for x in path)}: reference {a!r} device {b!r}")
            return False
        ws = p.dsum[tick]
        gs = sums[e][:ws.shape[0], :ws.shape[1]]
        if not np.array_equal(raw(ws), raw(gs)):
            t_, r_ = (int(x) for x in np.argwhere(raw(ws) != raw(gs))[0])
            fail(f"{label}: tick {tick}, env {e} ({p.name}@{p.seed}): damage_last_s(team {t_}, "
                 f"unit {p.uids[r_]}) reference {ws[t_, r_]!r} device {gs[t_, r_]!r}")
            return False
        st = PICSTATS.setdefault(label, {"env_ticks": 0, "records": 0, "objects": 0})
        st["env_ticks"] += 1
        st["records"] += sum(len(r) for r in want["records"].values())
        st["objects"] += sum(len(o) for o in want["objects"].values())
    return True


def judge_picture(played) -> None:
    """The picture's census: per gate the env-ticks compared; per case its
    count over the corpus, each > 0 (a case never reached FAILS)."""
    for label, st in sorted(PICSTATS.items()):
        print(f"    picture {label}: {st['env_ticks']} env-ticks equal (whole state), "
              f"{st['records']} record-ticks, {st['objects']} tracked-object-ticks")
    total = {k: 0 for k in PICTURE_CASES}
    peak = 0
    for p in played:
        for k, n in p.coverage.items():
            total[k] += n
        peak = max(peak, p.dmg_peak)
    print("    picture coverage: " + ", ".join(f"{k} {n}" for k, n in total.items())
          + f"; damage window peak {peak} entries (capacity "
          + f"{__import__('simulation.device_picture', fromlist=['DMG_CAP']).DMG_CAP})")
    for k, n in total.items():
        if n == 0:
            fail(f"picture coverage: the case {k!r} is never reached over the corpus (vacuous)")


# ---------------------------------------------------------------------------
# P8.7: THE DEVICE BOOKS (reward, termination, the ended latch)
# ---------------------------------------------------------------------------
def book_profiles() -> tuple:
    """The named profiles the reference books keep: every one of
    ``reward_v1.PROFILES`` (the device weighs with the obs table's rows of the
    gate's profile)."""
    from simulation.reward_v1 import PROFILES
    return tuple(PROFILES)


#: The coverage cases of the books (fireteam): each term's counters moved
#: (the served counters non-zero in some (row, step)), the two ends, a hit
#: past the unit's health (the clip that makes the walk's order matter).
BOOK_CASES = ("won", "eliminated", "clock", "bars_enemy", "bars_lost", "kills_enemy", "died",
              "bars_zombie", "share_bars", "share_enemy_bars", "share_alive",
              "share_enemy_alive", "share_zombie_bars", "bars_team", "team_kills",
              "end_elimination", "end_clock", "overkill")
BOOKSTATS: dict = {}
#: The served reward keys' census over part GATE's corpus: key -> (non-default
#: values, distinct raw values seen, capped).
BOOK_CENSUS: dict = {}


def dpic_outcome(outcome: str) -> int:
    from simulation import device_books as dbk
    return int(dbk.OUTCOME_CODES[outcome])


class RefBooks:
    """THE REFERENCE BOOKS of a played match, as ``rl_env.EnvRunner.step``
    drives them (its loop body, line for line): per named profile one
    ``RewardBooks`` and the step's old counters; ``start_step`` on a step's
    first tick, ``begin`` before the tick, ``count_events`` (the rows whose
    records ended frozen) and ``end`` after it, ``finish`` with the result at
    the end; the served arrays (``EnvRunner._reward``, the shared memory's
    float32 / uint8 casts) at every step's end."""

    def __init__(self, session, T, S):
        import rl_env
        from simulation import reward_v1 as rv
        sim, spec = session.sim, session.spec
        self.T, self.S = T, S
        self.ci = {n: i for i, n in enumerate(rl_env.COUNTER_NAMES)}
        self.k0 = rl_env.N_OLD_COUNTERS
        self.id_row = {}
        for t in range(len(spec.teams)):
            for u in sim.roster(t):
                self.id_row[u.id] = (t, t * S + u.slot)
        self.books, self.old, self.G = {}, {}, {}
        for prof in book_profiles():
            terms = rv.profile_terms(prof)
            self.books[prof] = rv.RewardBooks(sim, spec, S, terms, n_teams=T)
            self.old[prof] = np.zeros((T * S, self.k0), dtype=np.float32)
            self.G[prof] = rv.group_weights(terms, rl_env.COUNTER_NAMES)
        self.cov = {k: 0 for k in BOOK_CASES}
        self.team_before = {}

    def before(self, s) -> None:
        sim = s.sim
        if int(sim.tick) % DECIMATION == 0:          # a step's first tick
            for prof in self.books:
                self.books[prof].start_step()
                self.old[prof].fill(0.0)
        self.team_before = {u.id: int(u.team) for u in sim.units}
        for b in self.books.values():
            b.begin(sim)

    def after(self, s) -> None:
        import rl_env
        sim = s.sim
        term = s.is_terminal()
        res = s.result() if term else None
        for prof, b in self.books.items():
            ended = b.ended
            rl_env.count_events(sim.tick_events, self.team_before, self.id_row, self.ci,
                                self.old[prof], frozen=ended if ended.any() else None)
            b.end(sim)
            if term:
                b.finish(res)
        if term:
            self.cov["end_elimination" if len(s.rules.living_teams(sim)) < 2
                     else "end_clock"] += 1

    def record(self, p, s, term: bool) -> None:
        import rl_env
        from simulation import device_books as dbk
        tick = int(s.sim.tick)
        for prof, b in self.books.items():
            old = self.old[prof]
            p.books.setdefault(prof, {})[tick] = dbk.host_form(b, old)
            if tick > 0 and (tick % DECIMATION == 0 or term):
                counters = np.concatenate([old.astype(np.float64), b.counters], axis=1)
                reward = rl_env.EnvRunner._reward(SimpleNamespace(_G=self.G[prof]), old,
                                                  counters)
                p.served.setdefault(prof, {})[tick] = {
                    "reward": np.asarray(reward).astype(np.float32),
                    "counters": counters.astype(np.float32),
                    "team_balance": b.d_balance.astype(np.float32),
                    "team_uncredited": b.rest.astype(np.float32),
                    "team_outcome": b.outcome.astype(np.float32),
                    "rows_ended": b.rows_ended.astype(np.uint8),
                    "terminated": np.array([1 if term else 0], dtype=np.uint8),
                    "truncated": np.array([0], dtype=np.uint8),
                }
                if prof == "fireteam":
                    for name in BOOK_CASES:
                        if name in self.ci:
                            self.cov[name] += int(np.count_nonzero(counters[:, self.ci[name]]))

    def coverage(self) -> dict:
        st = self.books["fireteam"].stats
        out = dict(self.cov)
        out["overkill"] = int(st["overkill_hits"])
        out["_stats"] = dict(st)
        return out


def _books_breaks(obs) -> None:
    """THE P8.7 BREAKS: ``bk_order`` (the tick's events walked in reverse),
    ``bk_freeze`` (the books not frozen at the end), ``bk_late`` (termination
    one tick late) -- the kernel's switches -- and ``bk_weight`` (the terms
    read one row further in the obs table: each term's weight from the
    neighbouring row)."""
    import cupy as cp
    from simulation import device_books as dbk
    if BREAK in dbk.BREAK_CODES and BREAK:
        obs.books.brk = dbk.BREAK_CODES[BREAK]
    if BREAK == "bk_weight":
        off = {n: o for n, o, _ in dobs.table_layout()}["BOOK_TERMS"]
        n = len(dbk.TERM_NAMES)
        t = obs.table.get()
        t[:, off:off + n] = np.roll(t[:, off:off + n], -1, axis=1)
        obs.table = cp.asarray(t)


def books_equal(label, obs, played, tick, profile="fireteam", census=None) -> bool:
    """Every env's device books equal to its reference's at ``tick`` (the
    whole state, floats as raw bits; an env whose match ENDED compared, from
    its ending tick on, with its FINAL books: frozen), the ended latch set
    exactly from the ending tick, the ending tick and the result on the card,
    and at every step's end the served arrays raw-bit equal (an ended env's
    stay the ending step's)."""
    from simulation import device_books as dbk
    from simulation import device_picture as dpic
    forms = obs.books.forms()
    sc = obs.books.scalars()
    served = obs.books.host_served()
    latch = obs.ended.get()
    S_ = {f: i for i, f in enumerate(dbk.SCALAR_FIELDS)}
    for e, p in enumerate(played):
        if tick <= p.end:
            at = tick
        elif p.terminal:
            at = p.end
        else:
            continue
        where = f"{label}: tick {tick}, env {e} ({p.name}@{p.seed})"
        d = dpic.first_difference(p.books[profile][at], forms[e])
        if d is not None:
            path, a, b = d
            if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                i = next(i for i in range(len(a)) if a[i] != b[i])
                path, a, b = path + (i,), a[i], b[i]
            fail(f"{where}: the books differ at {'/'.join(str(x) for x in path)}: "
                 f"reference {a!r} device {b!r}")
            return False
        ended_ref = p.terminal and tick >= p.end
        if bool(latch[e]) != ended_ref:
            fail(f"{where}: the ended latch is {int(latch[e])}, the reference "
                 f"{'ended' if ended_ref else 'runs'} (end {p.end}, terminal {p.terminal})")
            return False
        if ended_ref:
            got = (int(sc[e, S_["ENDED_TICK"]]), int(sc[e, S_["RES_OUTCOME"]]),
                   int(sc[e, S_["RES_TEAM"]]), int(sc[e, S_["RES_TICKS"]]),
                   int(sc[e, S_["TERMINATED"]]), int(sc[e, S_["TRUNCATED"]]))
            want = (p.end,) + tuple(p.result) + (1, 0)
            if got != want:
                fail(f"{where}: (ending tick, outcome, team, ticks, terminated, truncated) "
                     f"device {got}, reference {want}")
                return False
        st = BOOKSTATS.setdefault(label, {"env_ticks": 0, "served": 0, "frozen": 0})
        st["env_ticks"] += 1
        if tick > p.end:
            st["frozen"] += 1
        sv = p.served.get(profile, {}).get(at)
        if sv is None or not (tick == at or ended_ref):
            continue
        for k, w in sv.items():
            g = served[k][e]
            g = g.reshape(w.shape) if g.size == w.size else g
            if w.shape != g.shape or not np.array_equal(raw(w), raw(g)):
                idx = np.argwhere(raw(w) != raw(g)) if w.shape == g.shape else []
                at_ = tuple(int(x) for x in idx[0]) if len(idx) else "shape"
                fail(f"{where}: served {k}: {len(idx)} value(s) differ, first at {at_}: "
                     f"reference {w[tuple(idx[0])] if len(idx) else w.shape} device "
                     f"{g[tuple(idx[0])] if len(idx) else g.shape}")
                return False
            if census is not None and tick == at:
                n_, vs = census.get(k, (0, set()))
                r_ = raw(w).ravel()
                if len(vs) < 4:
                    vs.update(int(x) for x in np.unique(r_)[:4])
                census[k] = (n_ + int(np.count_nonzero(r_)), vs)
        st["served"] += 1
    return True


def judge_books(played, census) -> None:
    """The books' census: per gate the env-ticks compared (frozen included)
    and the steps served; per case its count over the corpus, each > 0 (a case
    never reached FAILS); every served reward key non-constant over the corpus
    but ``truncated`` (0 under every named profile: the clock is an ending,
    design 6.5 -- exempt by name)."""
    for label, st in sorted(BOOKSTATS.items()):
        print(f"    books {label}: {st['env_ticks']} env-ticks equal (whole state; "
              f"{st['frozen']} frozen after the end), {st['served']} steps served equal")
    total = {k: 0 for k in BOOK_CASES}
    stats = {}
    for p in played:
        for k, n in p.book_cov.items():
            if k == "_stats":
                for kk, vv in n.items():
                    stats[kk] = stats.get(kk, 0) + vv
            else:
                total[k] += n
    print("    books coverage: " + ", ".join(f"{k} {n}" for k, n in total.items()))
    print("    books stats: " + ", ".join(f"{k} {n}" for k, n in sorted(stats.items())))
    outcomes = {}
    for p in played:
        if p.terminal:
            outcomes[p.result[0]] = outcomes.get(p.result[0], 0) + 1
    print(f"    books results (outcome code: matches): {outcomes}")
    for k, n in total.items():
        if n == 0:
            fail(f"books coverage: the case {k!r} is never reached over the corpus (vacuous)")
    for k in dobs.REWARD_KEYS:
        n_, vs = census.get(k, (0, set()))
        print(f"    books census {k}: {n_} non-default values, {len(vs)}"
              f"{'+' if len(vs) >= 4 else ''} distinct")
        if k != "truncated" and len(vs) <= 1:
            fail(f"books census: the served key {k!r} is constant over its corpus")


#: P8.7's clock hand match: the room with a door, both teams idle (nothing
#: shoots), its match's time limit cut to CLOCK_S seconds: the match ends by
#: the CLOCK (a draw with both teams standing: the clock term, termination by
#: the limit), which no played match reaches.
CLOCK_SEED = 79
CLOCK_S = 2.0


def corpus_clock():
    from dataclasses import replace
    lv = room_level()
    spec = replace(lv.match, time_limit_s=CLOCK_S,
                   teams=tuple(replace(t, controller="idle") for t in lv.match.teams))
    return [play("clock_hand", lv, spec, CLOCK_SEED)]


def part_books():
    """P8.7 beyond part GATE: the "range" profile (zombie / friendly-fire
    terms written 0) gated on (a) bots, (b) and the clock match as "fireteam"
    is; the "counters" profile refused by name on the device."""
    from simulation import device_books as dbk
    for p in corpus_a()[:1] + corpus_b() + corpus_clock():
        gate(f"books range {p.name}", [p], profile="range")
    try:
        dbk.profile_terms("counters")
    except dbk.BooksRefused as ex:
        print(f"  books: 'counters' refused by name ({str(ex)[:90]}...)")
    else:
        fail("books: the 'counters' profile was not refused")


def corpus_a():
    lv = ship_level()
    import _pinned_matches as pm
    return [play(w, lv, pm.spec_of(lv, w), None) for w in pm.WHICH]


def corpus_b():
    lv = room_level()
    return [play(ROOM, lv, None, None)]


# -- corpus (d): the smallest hand match for the owned columns (a) and (b)
# never move (P8.1: a second weapon, no loadout, no melee row, every status
# kind, the hand-grenade kinds the shipped loadout lacks, a plant, a blow's
# cooldown). A ship_demo bots match, cut short, with a hand state at tick 0
# (both the reference and the twin get it) and hand intents (tapped, so the
# twin is fed them).
HAND_SEED = 31
HAND_TICKS = 320
#: The hand loadout: the [marine] loadout row with every hand-grenade kind of
#: the weapon table's family (these counts, in the family's order).
HAND_GRENADES = (2, 1, 2, 1, 1, 3)


def hand_kit():
    """Corpus (d)'s loadout: the reference loadout row replaced IN THE LIVE
    TABLES by one carrying every hand-grenade kind (a frozen row: a
    ``dataclasses.replace``). ``Simulation`` construction rebuilds the
    loadout tables from the config (``loadouts.rebuild_tables``), so the
    row is put in after each build (the reference's and the twin's) and is
    gone at the next one: the host builders and the obs table read it while
    the session plays."""
    from dataclasses import replace
    from config import CFG
    from simulation.loadouts import get_tables
    from simulation.weapons import get_tables as wt
    by_name = get_tables().loadouts.by_name
    name = str(CFG.marine.loadout)
    rows = wt().hand_grenade_rows
    by_name[name] = replace(by_name[name],
                            grenades={r: HAND_GRENADES[k] for k, r in enumerate(rows)})


def _marines(sim):
    return sorted((u for u in sim.units if not u.is_zombie), key=lambda u: int(u.id))


def hand_setup(session):
    """Corpus (d)'s state at tick 0: the hand loadout (:func:`hand_kit`) and
    its grenades in every marine's hands; marine 0 without a loadout or a
    melee row (class -1, no charge kind, no grenade kinds, no melee); marine
    1 with a second weapon (its magazines belong to the loadout's weapon:
    spares untracked) and a blow just struck (the melee cooldown); marine 4
    with a third; every status kind 1..8 on a marine of its own, durations
    apart (DoTs gentle)."""
    from simulation import status as stt
    hand_kit()
    m = _marines(session.sim)
    for u in m:
        u.grenade_counts = list(HAND_GRENADES)
    m[0].loadout_name = None
    m[0].melee_id = ""
    m[1].weapon_id = "mp11_pdw"
    m[1].last_blow_tick = 0
    m[4].weapon_id = "lr50"
    for k in range(1, stt.N_STATUS_KINDS):
        stt.apply_status(m[k % len(m)], k, 0.05, 40 + 30 * k)


def hand_script(sim):
    """Corpus (d)'s hand intents: a plant, throws of every kind but the frag,
    a reload, a shove."""
    from simulation import unit_fixed
    m = _marines(sim)
    t = int(sim.tick)
    one = unit_fixed.FP_ONE
    alive = lambda u: u.alive and not u.is_zombie  # noqa: E731
    if t == 2 and alive(m[6]):
        sim.plant_charge_intent(int(m[6].id), 10 ** 6)
    for tk, who, kind in ((6, 7, 2), (10, 5, 3), (14, 4, 5), (30, 7, 1), (44, 5, 4)):
        if t == tk and alive(m[who]):
            sim.throw_grenade_intent(int(m[who].id), one, 0, 60, kind=kind)
    if t == 20 and alive(m[5]):
        sim.reload_intent(int(m[5].id))
    if t == 24 and alive(m[7]):
        sim.shove_intent(int(m[7].id))


#: P8.2's cover hand match: the training room whose crates are cover
#: (``range_hall``: two crates, ``blocks_los`` false, placed OFF the line of
#: fire), with crate 0 moved at tick 0 onto the line between the two spawns
#: (the reference and the twin alike), so the learner bot's line of fire to
#: the target crosses it (the room's own crates never cross a pair: counted).
COVER_ROOM = "range_hall"


def hall_setup(session):
    """The cover hand state at tick 0: crate 0 of ``sim.cover`` made a 3 x 1
    tile box on the segment between the two units' centre tiles."""
    sim = session.sim
    a, b = sorted(sim.units, key=lambda u: int(u.id))[:2]
    ax, ay = a.center_tile_x(), a.center_tile_y()
    bx, by = b.center_tile_x(), b.center_tile_y()
    mx, my = (ax + bx) // 2, (ay + by) // 2
    c = sim.cover[0]
    c.x0, c.x1 = float(mx - 1), float(mx + 2)
    c.y0, c.y1 = float(my), float(my + 1)       # one tile tall: a one-tile shift clears it


def corpus_d():
    lv = ship_level()
    import _pinned_matches as pm
    return [play("hand", lv, pm.spec_of(lv, "bots"), HAND_SEED, setup=hand_setup,
                 script=hand_script, max_ticks=HAND_TICKS)]


def corpus_cover():
    """P8.2: the training room with cover as it is, and the cover hand match."""
    lv = room_level(COVER_ROOM)
    return [play(COVER_ROOM, lv, None, None),
            play("hall_cover", lv, None, 51, setup=hall_setup)]


#: P8.2: the all_round rule over many units (the training rooms are 1 v 1, so
#: no team there ever has a seen AND an unseen enemy): a ship_demo bots match
#: whose spec names ``sight = "all_round"``, cut at this many ticks.
ROUND_TICKS = 480


def corpus_round():
    from dataclasses import replace

    import _pinned_matches as pm
    lv = ship_level()
    spec = replace(lv.match, sight="all_round")       # the "bots" match is the level's own
    return [play("ship_all_round", lv, spec, 61, max_ticks=ROUND_TICKS)]


#: P8.3's door hand match: the training room with a door (``range_door``),
#: both teams ``idle`` (only the hand intents move anything), the two units
#: placed at tick 0 on either side of the open door; unit 0 lobs a short-fuse
#: grenade, then a long-fuse one (the first detonates and the table's
#: compaction moves the second down a row while both teams track it); unit 1,
#: beside the door, shuts it (team 1 loses the grenade) and opens it again
#: (team 1 re-sees it under the same id).
DOOR_SEED = 71
DOOR_TICKS = 150
DOOR_THROWS = ((2, 20, 0), (14, 280, 1))   # (tick, fuse ticks, kind: a frag, then a smoke)
DOOR_USES = (60, 95)


def door_setup(session):
    from simulation import unit_fixed
    one = unit_fixed.FP_ONE
    a, b = sorted(session.sim.units, key=lambda u: int(u.id))[:2]
    a.set_pos_q(16 * one, 16 * one)
    b.set_pos_q(23 * one, 17 * one)
    # the hand state is tick 0's: the spawn's vision and picture are retaken
    # from it (the reference and the twin alike), as a hand-built sim does
    session.sim._vision_cache = None
    session.sim.start_team_picture(range(len(session.spec.teams)))


def door_script(sim):
    from simulation import unit_fixed
    one = unit_fixed.FP_ONE
    a, b = sorted(sim.units, key=lambda u: int(u.id))[:2]
    t = int(sim.tick)
    for tk, fuse, kind in DOOR_THROWS:
        if t == tk and a.alive:
            sim.throw_grenade_intent(int(a.id), one, 0, fuse, reach_q=3 * one, kind=kind)
    if t in DOOR_USES and b.alive:
        sim.use_intent(int(b.id))


def corpus_door():
    from dataclasses import replace
    lv = room_level()
    spec = replace(lv.match, teams=tuple(replace(t, controller="idle") for t in lv.match.teams))
    return [play("door_grenade", lv, spec, DOOR_SEED, setup=door_setup, script=door_script,
                 max_ticks=DOOR_TICKS)]


#: P8.4a's turn hand match: the room with a door, both teams idle (nothing
#: turns anyone), the two units placed as in the door match, unit 1 facing
#: AWAY from unit 0 and turning at the slowest the trait door allows (hook
#: TURN = ONE / 8): its ticks_to_within toward the enemy it sees behind it is
#: the cap (4 x the class half turn), a case no played match reaches.
TURN_SEED = 73
TURN_TICKS = 40


def turn_setup(session):
    from simulation import turning, unit_fixed, unit_traits
    b = sorted(session.sim.units, key=lambda u: int(u.id))[1]
    turning.place(b, 0)                                 # east: unit 0 stands west
    unit_traits.compile_unit(session.sim, b, {"TURN": unit_fixed.FP_ONE // 8})
    door_setup(session)                                 # placed, vision and picture retaken


def corpus_turn():
    from dataclasses import replace
    lv = room_level()
    spec = replace(lv.match, teams=tuple(replace(t, controller="idle") for t in lv.match.teams))
    return [play("slow_turn", lv, spec, TURN_SEED, setup=turn_setup, max_ticks=TURN_TICKS)]


#: P8.4b's list hand match: ship_demo, ``all_round``, both teams ``idle``
#: (only the hand intents move anything), cut at :data:`LIST_TICKS`. At tick 0
#: (the reference and the twin alike): marine A of team 0 stands beside a door
#: made a HULL door (horde.is_hull_door: the wall tile beside its span marked
#: vacuum; no corpus level has a hull door), that door made WIRED (a level wire to it: the
#: observation reads ``sim.level.wires``; the door drive table was built at
#: load, so the play is unchanged), a mate B and an enemy E within the frag
#: radius of each other and of A, a REMOTE charge (det_tick None) of A's at
#: A's feet; then A takes a hold order with E as its priority target, and B
#: walks at half throttle. The columns no played match moves (``hull``,
#: ``wired``, ``remote``, ``priority``, ``throttle``, ``frag_catch``).
LIST_SEED = 75
LIST_TICKS = 24


def _free(sim, x, y) -> bool:
    g = sim.gmap
    h, w = g.solid.shape
    return 0 <= x < w and 0 <= y < h and not g.solid[y, x] and not g.is_vacuum[y, x]         and not g.is_ambient[y, x]


def list_setup(session):
    from dataclasses import replace as dc_replace
    from types import SimpleNamespace as NS

    from simulation import horde, unit_fixed
    from simulation.charges import PlantedCharge
    sim = session.sim
    one = unit_fixed.FP_ONE
    a, b = [u for u in sorted(sim.units, key=lambda u: int(u.id))
            if not u.is_zombie and int(u.team) == 0][:2]
    enemy = [u for u in sorted(sim.units, key=lambda u: int(u.id))
             if not u.is_zombie and int(u.team) == 1][0]
    spot = None
    for d in sim._doors:
        if not d.alive:
            continue
        span = {(int(ty), int(tx)) for ty, tx in d.span}
        ring = sorted({(ty + dy, tx + dx) for ty, tx in span
                       for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0))} - span)
        wall = [(y, x) for y, x in ring if sim.gmap.solid[y, x]]
        for y, x in ring:
            if not wall or not _free(sim, x, y):
                continue
            near = [(x + ox, y + oy) for oy in range(-2, 3) for ox in range(-2, 3)
                    if (ox, oy) != (0, 0) and _free(sim, x + ox, y + oy)]
            if len(near) >= 2:
                spot = (d, x, y, near, wall[0])
                break
        if spot:
            break
    assert spot is not None, "list hand match: no door with a wall and room beside it"
    # the door made a HULL door: the wall tile beside its span marked vacuum
    # (no level of the corpus has one; ship_demo's hull is walls, not doors)
    sim.gmap.is_vacuum[spot[4][0], spot[4][1]] = True
    assert horde.is_hull_door(sim.gmap, spot[0].span)
    door, x, y, near, _ = spot
    a.set_pos_q(x * one, y * one)
    (bx, by), (ex, ey) = near[0], near[1]
    b.set_pos_q(bx * one, by * one)
    enemy.set_pos_q(ex * one, ey * one)
    wire = NS(input="open", target_ordinal=int(door.ordinal))
    sim.level = dc_replace(sim.level, wires=tuple(getattr(sim.level, "wires", ()) or ()) + (wire,))
    ammo = sim.weapons_tables.charge_rows[0]
    sim.planted_charges.append(PlantedCharge(x, y, ammo, int(a.id), det_tick=None))
    sim._vision_cache = None
    sim.start_team_picture(range(len(session.spec.teams)))


def list_script(sim):
    from simulation import room_graph, unit_fixed
    from simulation.unit_orders import graph_of
    a, b = [u for u in sorted(sim.units, key=lambda u: int(u.id))
            if not u.is_zombie and int(u.team) == 0][:2]
    enemy = [u for u in sorted(sim.units, key=lambda u: int(u.id))
             if not u.is_zombie and int(u.team) == 1][0]
    one = unit_fixed.FP_ONE
    if int(sim.tick) == 0 and a.alive:
        where = int(room_graph.place_at(graph_of(sim), a.center_tile_y(), a.center_tile_x()))
        sim.order_intent(int(a.id), 2, where, 0, int(enemy.id))
    if b.alive and int(sim.tick) in (2, 10):
        sim.set_move_dir(int(b.id), 0, one, one // 2)


def corpus_list():
    from dataclasses import replace
    lv = ship_level()
    spec = replace(lv.match, sight="all_round",
                   teams=tuple(replace(t, controller="idle") for t in lv.match.teams))
    return [play("list_hand", lv, spec, LIST_SEED, setup=list_setup, script=list_script,
                 max_ticks=LIST_TICKS)]


#: P8.5a's fire hand match: no played match lights a fire near a soldier (the
#: hand match's incendiary lands on nothing that burns), so the fire plane and
#: the air block's fire column are constant over the rest of the corpus. The
#: room with a door, both teams idle, the units placed as in the door match,
#: and at tick 0 (the reference and the twin alike) a fire started under unit
#: 0 the way the dev key does it (``debug_keys.debug_ignite``: the patch made
#: flammable, seeded, its solid tiles heated to their ignition point -- a fire
#: is started by delivering heat), then the fire burns for FIRE_TICKS.
FIRE_SEED = 77
FIRE_TICKS = 60


def fire_setup(session):
    import debug_keys
    door_setup(session)
    a = sorted(session.sim.units, key=lambda u: int(u.id))[0]
    tile = (int(a.center_tile_x()), int(a.center_tile_y()))
    debug_keys.debug_ignite(session.sim, SimpleNamespace(mouse_to_tile=lambda: tile))


def corpus_fire():
    from dataclasses import replace
    lv = room_level()
    spec = replace(lv.match, teams=tuple(replace(t, controller="idle") for t in lv.match.teams))
    return [play("fire_hand", lv, spec, FIRE_SEED, setup=fire_setup, max_ticks=FIRE_TICKS)]


def part_gate():
    ca, cb, cdn = Census(), Census(), Census()
    bc = BOOK_CENSUS
    for p in corpus_a():
        gate(f"(a) {p.name}", [p], ca, books_census=bc)
    for p in corpus_b():
        gate(f"(b) {p.name}", [p], cb, books_census=bc)
    for p in corpus_d():
        gate(f"(d) {p.name}", [p], cdn, books_census=bc)
    for p in corpus_cover():
        gate(f"(b/d cover) {p.name}", [p], cb, books_census=bc)
    for p in corpus_round():
        gate(f"(d all_round) {p.name}", [p], cdn, books_census=bc)
    for p in corpus_door():
        gate(f"(d door) {p.name}", [p], cdn, books_census=bc)
    for p in corpus_turn():
        gate(f"(d turn) {p.name}", [p], cdn, books_census=bc)
    for p in corpus_list():
        gate(f"(d list) {p.name}", [p], cdn, books_census=bc)
    for p in corpus_fire():
        gate(f"(d fire) {p.name}", [p], cdn, books_census=bc)
    for p in corpus_clock():
        gate(f"(d clock) {p.name}", [p], cdn, books_census=bc)
    ccn = Census()                                  # P8.6a: corpus (c), decoder-driven
    for p in corpus_c():
        gate(f"(c) {p.name}", [p], ccn, books_census=bc)
    ca.judge("census (a)", strict=False)
    cb.judge("census (b)", strict=False)
    cdn.judge("census (d)", strict=False)
    both = Census()
    ccn.judge("census (c)", strict=False)
    for src in (ca, cb, cdn, ccn):
        for k, n in src.nondefault.items():
            both.nondefault[k] = both.nondefault.get(k, 0) + n
            both.values.setdefault(k, set()).update(src.values.get(k, set()))
    both.judge("census (a)+(b)+(d)")
    judge_vision()
    judge_aims(corpus_a() + corpus_b() + corpus_d() + corpus_cover() + corpus_round()
               + corpus_door() + corpus_turn())
    judge_picture(corpus_a() + corpus_b() + corpus_d() + corpus_cover() + corpus_round()
                  + corpus_door())
    judge_list(corpus_a() + corpus_b() + corpus_d() + corpus_cover() + corpus_round()
               + corpus_door() + corpus_turn() + corpus_list())
    judge_masks(corpus_a() + corpus_b() + corpus_d() + corpus_cover() + corpus_round()
                + corpus_door() + corpus_turn() + corpus_list())
    judge_windows(corpus_a() + corpus_b() + corpus_d() + corpus_cover() + corpus_round()
                  + corpus_door() + corpus_turn() + corpus_list() + corpus_fire())
    judge_decoder(corpus_c())
    judge_books(corpus_a() + corpus_b() + corpus_d() + corpus_cover() + corpus_round()
                + corpus_door() + corpus_turn() + corpus_list() + corpus_fire()
                + corpus_clock(), BOOK_CENSUS)
    for p in corpus_a() + corpus_b() + corpus_d() + corpus_cover() + corpus_round():
        a0, a1 = p.cover_alive
        if a0 != a1:
            fail(f"cover: a cover's alive changed on the continuous path in {p.name} "
                 f"({a0} -> {a1}): the staged cover table would be stale")
        else:
            print(f"    cover {p.name}@{p.seed}: {len(a0)} cover entities, alive unchanged "
                  f"over {p.end} ticks")


def part_batch(n_list=(2, 4)):
    lv = ship_level()
    import _pinned_matches as pm
    ships = [play(w or "bots", lv, pm.spec_of(lv, w or "bots"), s) for w, s in BATCH_SHIP]
    rl = room_level()
    rooms = [play(ROOM, rl, None, s) for s in BATCH_ROOM]
    for group, ps in (("ship_demo", ships), ("room", rooms)):
        ends = [p.end for p in ps]
        print(f"  {group} ends per env: {ends}")
        if len(set(ends[:max(n_list)])) < 2:
            print(f"  NOTE: every {group} env ends at the same tick")
        for n in n_list:
            gate(f"batch {group} N={n}", ps[:n])
    # P8.7: the ended envs' frozen books (compared every tick after their end)
    frozen = 0
    for label, st in sorted(BOOKSTATS.items()):
        if label.startswith("batch"):
            frozen += st["frozen"]
            print(f"    books {label}: {st['env_ticks']} env-ticks equal ({st['frozen']} frozen "
                  f"after the env's end), {st['served']} steps served equal")
    if frozen == 0:
        fail("batch: no env's books were compared frozen after its end (vacuous)")


def part_handover():
    import cupy as cp
    import torch
    p = corpus_b()[0]
    s = twin(p)
    c = cd.DeviceConductor(s.sim, bp=bp)
    obs = dobs.DeviceObservation(c, [s.spec], p.S, DECIMATION, n_teams=p.T)
    ts = obs.torch()
    bad = [k for k, v in obs.buffers.items() if ts[k].data_ptr() != int(v.data.ptr)
           or str(ts[k].dtype).split(".")[-1] != str(np.dtype(v.dtype)) or tuple(ts[k].shape) != tuple(v.shape)]
    if bad:
        fail(f"handover: torch tensors that are not their CuPy buffer: {bad}")
    else:
        print(f"  handover: {len(ts)} torch tensors share their CuPy buffers' pointers, "
              f"dtypes and shapes")
    for t in range(p.T):
        rows = obs.team_rows(t)
        for k, v in rows.items():
            base = obs.buffers[k]
            want = int(base.data.ptr) + t * p.S * int(base.strides[1])
            if v.data_ptr() != want or v.shape[1] != p.S:
                fail(f"handover: team_rows({t})[{k!r}] is not a view of the buffer")
                break
    so = dobs.stream_ordering(bp)
    print(f"  stream ordering: {so}")
    if so["tick_stream_flags"] != 0:
        fail(f"handover: the tick stream is not blocking (flags {so['tick_stream_flags']})")
    if so["cupy_stream"] != 0 or so["torch_stream"] != 0:
        fail(f"handover: CuPy / torch are not on the legacy default stream: {so}")
    # a torch read straight after the tick that builds a decision, no sync
    for k in range(DECIMATION):
        c.latch(p.intents[k], env=0)
        c.tick()
    tw = ts["near_u"].to(torch.int64).sum().item()
    cp.cuda.runtime.deviceSynchronize()
    ref = int(obs.buffers["near_u"].astype(cp.int64).sum().get())
    if tw != ref or ref == 0:
        fail(f"handover: torch read {tw} after the tick, the synced buffer holds {ref}")
    else:
        print(f"  handover: torch's unsynced read after the tick == the synced buffer ({ref})")


def part_syncs():
    p = corpus_b()[0]
    counts = {}
    for n in (1, 2):
        sessions = [twin(p) for _ in range(n)]
        c = cd.DeviceConductor([x.sim for x in sessions], bp=bp)
        obs = dobs.DeviceObservation(c, [x.spec for x in sessions], p.S, DECIMATION,
                                     n_teams=p.T)
        sc = cc.SyncCount()
        per = []
        b0 = obs.builds
        try:
            for k in range(min(60, len(p.intents))):
                for e in range(n):
                    c.latch(p.intents[k], env=e)
                had = any(c._orders)
                sc.py = []
                bp.host_sync_log_begin()
                try:
                    c.tick()
                finally:
                    log = list(bp.host_sync_log_end()) + sc.py
                per.append((k, log, had))
        finally:
            sc.close()
        steady = [lg for k, lg, o in per[1:] if not o]
        names = sorted({x.partition(" | ")[2] for lg in steady for x in lg})
        counts[n] = [len(lg) for lg in steady]
        print(f"  N={n}: {np.mean(counts[n]):.2f} host syncs per steady tick, named {names}, "
              f"{obs.builds - b0} block launches during the count")
        if obs.builds - b0 == 0:
            fail(f"syncs N={n}: no block was built during the count (vacuous)")
        extra = [x for x in names if x not in ("max_n_sub", "books_read")]
        if extra:
            fail(f"syncs N={n}: a sync outside the conductor's named reads: {extra}")
    if counts[1] != counts[2]:
        fail(f"syncs: the per-tick count depends on N ({counts[1][:6]} vs {counts[2][:6]})")


def part_ended():
    import cupy as cp
    p = corpus_b()[0]
    sessions = [twin(p), twin(p)]
    c = cd.DeviceConductor([x.sim for x in sessions], bp=bp)
    obs = dobs.DeviceObservation(c, [x.spec for x in sessions], p.S, DECIMATION, n_teams=p.T)
    keep = {k: obs.buffers[k][1].copy() for k in dobs.BUILT if k in obs.buffers}
    books_kept = obs.books.state[1].copy()       # P8.7: a set latch freezes the books too
    obs.ended[1] = 1
    moved = False
    n_dec = 0
    for k in range(min(len(p.intents), 12 * DECIMATION)):
        for e in range(2):
            c.latch(p.intents[k], env=e)
        c.tick()
        tick = int(c.sims[0].tick)
        if tick % DECIMATION == 0:
            n_dec += 1
            w = p.obs[tick]
            if not np.array_equal(raw(obs.buffers["near_u"][0, :, 0].get()),
                                  raw(w[("near_u", "wall")])):
                fail(f"ended: env 0 (latch clear) is not its reference at tick {tick}")
                return
            if any(not bool(cp.array_equal(obs.buffers[k_][1], keep[k_])) for k_ in keep):
                moved = True
    if not bool(cp.array_equal(obs.books.state[1], books_kept)):
        fail("ended: the books of the env whose ended latch is set moved")
    if moved:
        fail("ended: the env whose ended latch is set was rebuilt")
    else:
        print(f"  ended: env 1 (latch set) kept its buffers over {n_dec} decisions; env 0 "
              f"equal to its reference")
    ref_moved = any(not np.array_equal(p.obs[0][kk], p.obs[max(p.obs)][kk])
                    for kk in p.obs[0] if kk[0] in ("near_u", "far_f", "far_u"))
    if not ref_moved:
        print("  NOTE: the reference planes did not move over the match")


def half_corpus() -> np.ndarray:
    """EVERY float16 rounding case as float64: every float16 value (all 2^16
    bit patterns, inf and NaN included), every midpoint of two neighbouring
    finite halves and its two float64 neighbours, the overflow boundary
    (65520 and its neighbours), values past it, the underflow boundary
    (2^-25 and its neighbours, 2^-26, float64 subnormals), signed zeros,
    NaN payloads; both signs."""
    h = np.arange(0x7c00, dtype=np.uint16).view(np.float16).astype(np.float64)
    mids = (h[:-1] + h[1:]) / 2.0                         # exact in float64
    edge = np.array([65504.0, 65520.0, 65536.0, 1e5, 1e300, np.inf, 2.0 ** -24,
                     2.0 ** -25, 2.0 ** -26, 3 * 2.0 ** -26, 5e-324, 2.2e-308, 0.0])
    pos = np.concatenate([h, mids, edge])
    pos = np.concatenate([pos, np.nextafter(pos, np.inf), np.nextafter(pos, -np.inf)])
    pos = pos[pos >= 0.0]
    allh = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(
        np.float64)
    nans = np.array([0x7ff8000000000000, 0x7ff0000000000001, 0x7ff4000000000000,
                     0x7ff00400_00000000, 0xfff8000000000000, 0x7fffffffffffffff],
                    dtype=np.uint64).view(np.float64)
    return np.concatenate([pos, -pos, allh, nans])


def part_floatkit():
    """The observation's float kit on the card (``cpp/src/cuda_obs_float.cuh``):
    the hand double -> float16 conversion EXHAUSTIVELY equal to numpy's direct
    conversion; the double -> float32 cast equal to numpy's; a*b + c never
    contracted (numpy rounds twice); the squash's served float32 equal to the
    host's ``obs_spec_v1.squash`` on a dense sweep (CUDA's log against the
    host's libm: the count is reported). Breaks: ``half_f32`` (the conversion
    through float32, double rounding), ``fused`` (an explicit fma, what
    contraction would build)."""
    from simulation import obs_spec_v1 as o
    rng = np.random.default_rng(81)
    # -- float16 --
    xs = half_corpus()
    with np.errstate(over="ignore", invalid="ignore"):
        want = xs.astype(np.float16).view(np.uint16)
    got = np.asarray(bp.obs_probe_half(xs, via_f32=(BREAK == "half_f32")))
    bad = np.flatnonzero(got != want)
    print(f"  float16: {len(xs)} values (every half, every midpoint +- 1 ulp, the edges, "
          f"NaNs): {len(bad)} differ from numpy")
    if len(bad):
        i = bad[0]
        fail(f"floatkit: double -> float16 differs on {len(bad)} of {len(xs)}: first "
             f"{xs[i]!r} (0x{int(np.float64(xs[i]).view(np.uint64)):016x}) numpy "
             f"0x{int(want[i]):04x} device 0x{int(got[i]):04x}")
    # -- float32 --
    rnd = np.concatenate([rng.uniform(-70000, 70000, 200000),
                          np.exp(rng.uniform(-750, 700, 200000)) * rng.choice([-1, 1], 200000)])
    xf = np.concatenate([xs, rnd, np.nextafter(rnd, np.inf)])
    with np.errstate(over="ignore", invalid="ignore"):
        wf = xf.astype(np.float32).view(np.uint32)
    gf = np.asarray(bp.obs_probe_f32(xf)).view(np.uint32)
    badf = np.flatnonzero(gf != wf)
    print(f"  float32: {len(xf)} values: {len(badf)} differ from numpy")
    if len(badf):
        i = badf[0]
        fail(f"floatkit: double -> float32 differs on {len(badf)}: first {xf[i]!r} numpy "
             f"0x{int(wf[i]):08x} device 0x{int(gf[i]):08x}")
    # -- no contraction: c = -(a*b rounded), so a*b + c is 0 unrounded-twice and
    # the product's rounding error when fused --
    a = rng.uniform(0.5, 2.0, 100000)
    b = rng.uniform(0.5, 2.0, 100000)
    c = -(a * b)
    wm = (a * b) + c                                       # numpy: two roundings
    gm = np.asarray(bp.obs_probe_muladd(a, b, c, fused=(BREAK == "fused")))
    fused = np.asarray(bp.obs_probe_muladd(a, b, c, fused=True))
    badm = int(np.count_nonzero(gm.view(np.uint64) != wm.view(np.uint64)))
    disc = int(np.count_nonzero(fused.view(np.uint64) != wm.view(np.uint64)))
    print(f"  a*b + c: {len(a)} triples, {badm} differ from numpy's two roundings; the "
          f"fused form differs on {disc} (the inputs discriminate)")
    if badm:
        fail(f"floatkit: a*b + c is contracted on {badm} of {len(a)} triples")
    if disc < len(a) // 2:
        fail(f"floatkit: the contraction probe's inputs do not discriminate ({disc})")
    # -- the squash: CUDA's log against the host's (the served float32) --
    sx = np.concatenate([np.exp(rng.uniform(0.0, 3.2, 300000)),
                         -np.exp(rng.uniform(0.0, 3.2, 100000)),
                         rng.uniform(-1.0, 1.0, 10000),
                         np.array([1.0, -1.0, np.e ** 3, np.nextafter(np.e ** 3, 0), 1e9]),
                         np.arange(1, 4097, dtype=np.float64) / 16.0])
    g64, g32 = bp.obs_probe_squash(sx)
    w64 = np.array([o.squash(x) for x in sx], dtype=np.float64)
    w32 = w64.astype(np.float32)
    d64 = int(np.count_nonzero(np.asarray(g64).view(np.uint64) != w64.view(np.uint64)))
    d32 = int(np.count_nonzero(np.asarray(g32).view(np.uint32) != w32.view(np.uint32)))
    print(f"  squash: {len(sx)} values: float64 differs from the host's on {d64}, the "
          f"served float32 on {d32}")
    if d32:
        fail(f"floatkit: the squash's served float32 differs from the host's on {d32} of "
             f"{len(sx)} values (CUDA's log vs the host's: the shared-module trigger)")


# ---------------------------------------------------------------------------
# P8.2: PART SIGHT -- the bulk forms against rays_clear / window_clear
# ---------------------------------------------------------------------------
SIGHT_PAIRS = 30000
SIGHT_WINDOWS = 120
WIN = 25


def _ship_states():
    """Three ship_demo states of one shape (one launch): the level as it
    starts (every door shut), half the doors OPENED (``want_open`` and the
    door sweep, on a scratch session), and grille tiles (open to gas, closed
    to sight: the sight screen set) stamped on 3% of the open tiles of a
    scratch copy. Returns (sims, label)."""
    import _pinned_matches as pm
    from simulation import door_system
    from simulation.entities.door import DOOR_OPEN
    from simulation.match_session import MatchSession
    lv = ship_level()
    spec = pm.spec_of(lv, "bots")
    a, b, g = (MatchSession(lv, spec, seed=41, breach_physics=bp) for _ in range(3))
    _toggle_doors(b.sim, every=2)
    opened = sum(int(d.state == DOOR_OPEN) for d in b.sim._doors)
    rng = np.random.default_rng(82)
    open_tiles = np.argwhere(~g.sim.gmap.solid)
    pick = open_tiles[rng.choice(len(open_tiles), max(1, len(open_tiles) // 33), replace=False)]
    g.sim.gmap.sight_screen[pick[:, 0], pick[:, 1]] = True
    print(f"  ship_demo states: shut doors / {opened} of {len(b.sim._doors)} doors opened / "
          f"{len(pick)} grille tiles")
    return [a.sim, b.sim, g.sim], "ship_demo"


def _toggle_doors(sim, every=1) -> int:
    """Flip every ``every``-th door (an open one shut, a shut one open)
    through its latch and the door sweep (a scratch session); returns how
    many flipped."""
    from simulation import door_system
    from simulation.entities.door import DOOR_OPEN
    doors = list(sim._doors)[::every]
    before = [d.state for d in doors]
    for d in doors:
        d.want_open = d.state != DOOR_OPEN
    door_system.sweep_doors(sim)
    return sum(int(d.state != s) for d, s in zip(doors, before))


def _room_states():
    """The training room with a door (as it starts: open; and shut) and the
    one with cover crates (``blocks_los`` false: rays_clear ignores them)."""
    from simulation.match_session import MatchSession
    lv = room_level()
    a, b = (MatchSession(lv, None, seed=42, breach_physics=bp) for _ in range(2))
    flipped = _toggle_doors(b.sim)
    print(f"  {ROOM} states: as it starts / {flipped} of {len(b.sim._doors)} door(s) flipped")
    return [a.sim, b.sim], ROOM


def _hall_states():
    from simulation.match_session import MatchSession
    lv = room_level(COVER_ROOM)
    a = MatchSession(lv, None, seed=43, breach_physics=bp)
    print(f"  {COVER_ROOM} state: {len(a.sim.cover)} cover entities")
    return [a.sim], COVER_ROOM


def _hot_points(sims):
    """Tiles where a line's answer turns: door span tiles and grille tiles."""
    pts = []
    for s in sims:
        for d in s._doors:
            pts += [tuple(t) for t in d.span]
        pts += [tuple(t) for t in np.argwhere(s.gmap.sight_screen)]
    return np.array(sorted(set(pts)), dtype=np.int64).reshape(-1, 2)


def _pairs(rng, h, w, hot, n):
    """``n`` (y0, x0, y1, x1) pairs: uniform over the map and a 3-tile ring
    outside it (out of bounds), lines across the hot tiles, and pairs on one
    tile."""
    k = n // 3
    uni = np.stack([rng.integers(-3, h + 3, k), rng.integers(-3, w + 3, k),
                    rng.integers(-3, h + 3, k), rng.integers(-3, w + 3, k)], axis=1)
    if len(hot):
        c = hot[rng.integers(0, len(hot), k)]
        near = np.concatenate([c + rng.integers(-10, 11, (k, 2)),
                               c + rng.integers(-10, 11, (k, 2))], axis=1)
    else:
        near = uni.copy()
    m = n - 2 * k
    one = rng.integers(-2, max(h, w) + 2, (m, 2))
    same = np.concatenate([one, one], axis=1)
    return np.concatenate([uni, near, same]).astype(np.int32)


def _sight_group(sims, label, rng) -> None:
    import cupy as cp
    from simulation import vision
    h, w = sims[0].gmap.solid.shape
    solid = cp.asarray(np.stack([np.ascontiguousarray(s.gmap.solid) for s in sims]))
    screen = cp.asarray(np.stack([np.ascontiguousarray(s.gmap.sight_screen) for s in sims]))
    hot = _hot_points(sims)
    pairs = _pairs(rng, h, w, hot, SIGHT_PAIRS)
    host = {}
    for physical in (False, True):
        dev_phys = (not physical) if BREAK == "forms" else physical
        stack = cp.asarray(np.stack([pairs] * len(sims)))
        got, faults = dobs.rays_clear_device(bp, solid, screen, stack, dev_phys)
        got, faults = got.get(), faults.get()
        for e, s in enumerate(sims):
            want = vision.rays_clear(s, pairs, physical=physical).astype(np.uint8)
            host[(e, physical)] = want
            if int(faults[e]) or not np.array_equal(want, got[e]):
                bad = np.flatnonzero(want != got[e])
                i = int(bad[0]) if len(bad) else -1
                fail(f"sight {label}: env {e} rays_clear(physical={physical}) differs on "
                     f"{len(bad)} of {len(pairs)} pairs (faults {int(faults[e])})"
                     + (f", first {tuple(int(x) for x in pairs[i])}: reference {want[i]} "
                        f"device {got[e][i]}" if i >= 0 else ""))
                return
    oob = int(np.count_nonzero((pairs[:, [0, 2]] < 0).any(1) | (pairs[:, [0, 2]] >= h).any(1)
                               | (pairs[:, [1, 3]] < 0).any(1) | (pairs[:, [1, 3]] >= w).any(1)))
    same = int(np.count_nonzero((pairs[:, 0] == pairs[:, 2]) & (pairs[:, 1] == pairs[:, 3])))
    forms = sum(int(np.count_nonzero(host[(e, False)] != host[(e, True)]))
                for e in range(len(sims)))
    doors = (int(np.count_nonzero(host[(0, True)] != host[(1, True)]))
             if len(sims) > 1 else None)
    print(f"  sight {label}: {len(sims)} envs x {len(pairs)} pairs equal in both forms "
          f"({oob} with an end outside the map, {same} on one tile; {doors} pairs the "
          f"opened doors turn, {forms} env-pairs the two forms answer apart)")
    if label == "ship_demo" and forms == 0:
        fail("sight ship_demo: no pair where the two forms differ (the grille case is vacuous)")
    if doors == 0:
        fail(f"sight {label}: no pair turned by flipping the doors (vacuous)")
    # -- windows --
    n = SIGHT_WINDOWS
    cy, cx = rng.integers(-2, h + 2, n), rng.integers(-2, w + 2, n)
    q = np.stack([cy, cx, cy - WIN // 2 + rng.integers(-15, 16, n),
                  cx - WIN // 2 + rng.integers(-15, 16, n)], axis=1).astype(np.int32)
    if len(hot):                                    # half of them centred near a hot tile
        hh = hot[rng.integers(0, len(hot), n // 2)] + rng.integers(-6, 7, (n // 2, 2))
        q[: n // 2, 0], q[: n // 2, 1] = hh[:, 0], hh[:, 1]
        q[: n // 2, 2], q[: n // 2, 3] = hh[:, 0] - WIN // 2, hh[:, 1] - WIN // 2
    cells = 0
    for to_window in (False, True):
        for physical in (False, True):
            dev_phys = (not physical) if BREAK == "forms" else physical
            got, faults = dobs.window_clear_device(bp, solid, screen,
                                                   cp.asarray(np.stack([q] * len(sims))),
                                                   WIN, WIN, to_window, dev_phys)
            got, faults = got.get(), faults.get()
            for e, s in enumerate(sims):
                for j, (y, x, wy, wx) in enumerate(q.tolist()):
                    want = vision.window_clear(s, y, x, wy, wx, WIN, WIN, to_window=to_window,
                                               physical=physical)
                    cells += want.size
                    if int(faults[e]) or not np.array_equal(want.astype(np.uint8), got[e, j]):
                        bad = np.argwhere(want.astype(np.uint8) != got[e, j])
                        fail(f"sight {label}: env {e} window_clear(({y}, {x}), ({wy}, {wx}), "
                             f"to_window={to_window}, physical={physical}) differs on "
                             f"{len(bad)} cells, first {tuple(int(v) for v in bad[0])}"
                             if len(bad) else f"sight {label}: env {e} window faults")
                        return
    print(f"  sight {label}: {len(sims)} envs x {n} windows x 4 (both directions, both forms) "
          f"equal ({cells} cells)")


def part_sight():
    """vision.rays_clear / window_clear on the card (P8.2) equal to the host's
    on dense random pairs and windows over real maps: ship_demo with its
    doors shut, half opened, and grille tiles; the training room (crates
    that do not block sight) shut and opened. Break ``forms``: the two
    has_los forms swapped."""
    rng = np.random.default_rng(8202)
    for states in (_ship_states, _room_states, _hall_states):
        sims, label = states()
        _sight_group(sims, label, rng)


# ---------------------------------------------------------------------------
# P8.2: PART WALL -- the wall's first device gate (critique finding 22)
# ---------------------------------------------------------------------------
def part_wall():
    """Who a team sees depends on an enemy's row only through that enemy
    standing in view: perturbing the store row (facing, hp, turn rate,
    effort, exposure) of an enemy NO member of team t sees leaves t's
    visible set unchanged; the same perturbation of a SEEN enemy leaves it
    unchanged too, and killing the seen enemy removes it. Corpus (a) bots
    (cone) and the all_round ship match, at decision ticks."""
    import cupy as cp
    from simulation import turning
    from simulation import unit_store
    lay = unit_store.layout()
    col = {n: lay.column(n).offset for n in ("team", "life_state", "facing_q",
                                            "current_hp", "omega_q", "effort", "exposure",
                                            "live_trigger")}
    aims_done = {}
    col["valid"] = lay.valid
    dead = 1                                         # unit_schema._LIFE_CHOICES.index("dead")
    total = {}
    for p in corpus_a()[:1] + corpus_round():
        s = twin(p)
        c = cd.DeviceConductor(s.sim, bp=bp)
        obs = dobs.DeviceObservation(c, [s.spec], p.S, DECIMATION, n_teams=p.T)
        done = 0
        for k in range(len(p.intents)):
            c.latch(p.intents[k], env=0)
            c.tick()
            if int(c.sims[0].tick) % DECIMATION or done >= 6:
                continue
            rows = c._rows.get()[0]
            vis = obs.visible.get()[0]
            for t in range(p.T):
                enemy = [r for r in range(len(p.uids)) if rows[r, col["valid"]]
                         and rows[r, col["life_state"]] == 0 and rows[r, col["team"]] != t]
                seen = [r for r in enemy if vis[t, r]]
                unseen = [r for r in enemy if not vis[t, r]]
                if not seen or not unseen:
                    continue

                def run(edit):
                    pert = c._rows.copy()
                    edit(pert)
                    obs.vision(store_ptr=int(pert.data.ptr))
                    out = obs.visible.get()[0, t].copy()
                    del pert
                    return out

                def perturb(r):
                    def f(a):
                        f0 = int(a[0, r, col["facing_q"]])
                        f1 = f0 + turning.PI_Q16 // 2       # a quarter turn, in (-PI, PI]
                        a[0, r, col["facing_q"]] = f1 - 2 * turning.PI_Q16 \
                            if f1 > turning.PI_Q16 else f1
                        a[0, r, col["current_hp"]] = max(1, int(a[0, r, col["current_hp"]]) // 3)
                        a[0, r, col["omega_q"]] += 12345
                        a[0, r, col["effort"]] += 1 << 15
                        a[0, r, col["exposure"]] += 7
                    return f

                u, sv = unseen[0], seen[0]
                if not np.array_equal(run(perturb(u)), vis[t]):
                    fail(f"wall {p.name}: tick {int(c.sims[0].tick)}, team {t}: perturbing "
                         f"unseen enemy {p.uids[u]}'s row changed the team's visible set")
                    return
                # P8.4a: nothing team t's soldiers are told of their lanes, aims,
                # lines of fire or turns moves with the unseen enemy's row
                base = team_aims(obs, t, p.S)
                pert = c._rows.copy()
                perturb(u)(pert)
                pert[0, u, col["live_trigger"]] = 1 - int(pert[0, u, col["live_trigger"]])
                obs.vision(store_ptr=int(pert.data.ptr))
                obs.aims(store_ptr=int(pert.data.ptr))
                obs.build_list(store_ptr=int(pert.data.ptr))
                obs.build_masks(store_ptr=int(pert.data.ptr))
                obs.build_windows(store_ptr=int(pert.data.ptr))
                moved = [k for k, v in team_aims(obs, t, p.S).items()
                         if not np.array_equal(v, base[k])]
                del pert
                obs.vision()
                obs.aims()
                obs.build_list()
                obs.build_masks()
                obs.build_windows()
                if moved:
                    fail(f"wall {p.name}: tick {int(c.sims[0].tick)}, team {t}: perturbing "
                         f"unseen enemy {p.uids[u]}'s row moved the team's {moved}")
                    return
                aims_done[p.rule] = aims_done.get(p.rule, 0) + 1
                if not np.array_equal(run(perturb(sv)), vis[t]):
                    fail(f"wall {p.name}: tick {int(c.sims[0].tick)}, team {t}: perturbing seen "
                         f"enemy {p.uids[sv]}'s facing / hp changed whether it is seen")
                    return

                def kill(a):
                    a[0, sv, col["life_state"]] = dead
                killed = run(kill)
                if killed[sv] or np.array_equal(killed, vis[t]):
                    fail(f"wall {p.name}: tick {int(c.sims[0].tick)}, team {t}: killing seen "
                         f"enemy {p.uids[sv]} did not remove it from the visible set")
                    return
                done += 1
            obs.vision()                             # the real store again
        total[p.rule] = total.get(p.rule, 0) + done
        print(f"  wall {p.name} ({p.rule}): {done} (team, tick) cases with a seen and an "
              f"unseen enemy: unseen perturbed -> unchanged, seen perturbed -> unchanged, "
              f"seen killed -> removed")
    for rule in ("cone", "all_round"):
        if not total.get(rule):
            fail(f"wall: no case under the {rule!r} rule (vacuous)")
        if not aims_done.get(rule):
            fail(f"wall: no P8.4a case under the {rule!r} rule (vacuous)")
    print(f"  wall P8.4a / P8.4b / P8.4c: {aims_done} (team, tick) cases: the unseen enemy's "
          f"row perturbed (and its trigger flipped) -> the team's lanes, lanes plane, aims, "
          f"lines of fire, turns, list (entries, mask, uids) and four masks unchanged")


def team_aims(obs, t, S) -> dict:
    """Team t's P8.4a answers of env 0 (the WALL's reading): its fire lanes,
    the lanes plane of its rows, its aim_of answers, its rows' lines of fire
    and ticks_to_within; P8.4b: its rows' list, list mask and list uids;
    P8.4c: its rows' four masks."""
    rows = slice(t * S, (t + 1) * S)
    from simulation import obs_spec_v1 as o
    k = o.NEAR_U_PLANES.index("lanes")
    return {"lane_grid": obs.lane_grid[0, t].get(), "lanes_plane":
            obs.buffers["near_u"][0, rows, k].get(), "aim": obs.aim[0, t].get(),
            "lof": obs.lof[0, rows].get(), "ttw": obs.ttw[0, rows].get(),
            # P8.4b: the list of the team's rows, its mask and its uids
            "list": obs.buffers["list"][0, rows].get(),
            "list_mask": obs.buffers["list_mask"][0, rows].get(),
            "list_uid": obs.buffers["list_uid"][0, rows].get(),
            # P8.4c: the four masks of the team's rows
            **{k: obs.buffers[k][0, rows].get()
               for k in ("mask_feet", "mask_aim", "mask_trigger", "mask_hand")},
            # P8.5a: every window plane of the team's rows, its air block and
            # felt temperature
            **{k: obs.buffers[k][0, rows].get() for k in ("near_f", "near_u", "far_f", "far_u")},
            "air_felt": obs.buffers["vector"][0, rows][..., list(range(58, 66)) + [96]].get()}


# ---------------------------------------------------------------------------
# P8.3: PART RESTAGE -- the seam P8.9b resets through
# ---------------------------------------------------------------------------
RESTAGE_TICK = 60


def part_restage():
    """Env e's device picture set from a HOST picture's normal form mid-match
    (``DevicePicture.restage``: the seam a per-env reset uses) and played on:
    a batch of two hand matches, env 1's block scrambled at
    :data:`RESTAGE_TICK` and restaged from its reference there, env 0
    untouched; both envs' whole picture equal to their references at every
    later tick."""
    import cupy as cp
    p = corpus_d()[0]
    played = [p, p]
    sessions = [twin(x) for x in played]
    c = cd.DeviceConductor([x.sim for x in sessions], bp=bp)
    obs = dobs.DeviceObservation(c, [x.spec for x in sessions], p.S, DECIMATION, n_teams=p.T,
                                 picture_from_host=[0, 1])
    n = 0
    for k in range(p.end + 1):
        tick = int(c.sims[0].tick)
        if tick == RESTAGE_TICK:
            obs.picture.state[1] = cp.asarray(np.arange(obs.picture.W, dtype=np.int64) * 7919)
            obs.picture.restage(1, form=p.picture[tick])
        if not picture_equal("restage", obs, played, tick):
            return
        n += 1
        if k == p.end:
            break
        for e in range(2):
            c.latch(p.intents[k] if k < len(p.intents) else (), env=e)
        c.tick()
    print(f"  restage: env 1 scrambled and restaged at tick {RESTAGE_TICK}; both envs' whole "
          f"picture equal to the reference over {n} ticks")


# ---------------------------------------------------------------------------
# P8.6a: THE DEVICE DECODER -- corpus (c), generated live (critique finding 9)
# ---------------------------------------------------------------------------
#: Corpus (c): (label, level, run seed, tick cap). An in-process
#: ``rl_env.EnvRunner`` per run (both teams ``policy``, v1 actions, the
#: commander on), its policies decoding seeded masked-random decisions.
DEC_RUNS = (("policy crowd 102", "ship_demo", 102, 600, True),
            ("policy crowd 104", "ship_demo", 104, 600, True),
            ("policy crowd 105", "ship_demo", 105, 600, True),
            ("policy crowd 110", "ship_demo", 110, 600, True),
            ("policy crowd 116", "ship_demo", 116, 600, True),
            ("policy ship_demo 133", "ship_demo", 133, 600, False),
            ("policy room 83", ROOM, 83, 400, False), ("policy room 87", ROOM, 87, 400, False))
#: Run seeds are chosen: a seed whose play opens or closes a door beside gas
#: a blast heated past the topology seam's proved range (|T_abs| < 2^28,
#: cpp/src/cuda_topology.cuh) is refused by the device DOORS stage
#: (ApplierFault at '9e(d) doors') on the tap path as well -- a finding of
#: P8.6a (e.g. crowd 100, ship_demo 81), not the decoder's; 11 of 30 seeds
#: scanned hit it.
#: The decision kit at tick 0 (the runner's session and the twin alike): the
#: hand loadout with these counts of every hand-grenade kind, in every
#: marine's hands, so every throw kind with every reach occurs.
DEC_GRENADES = (3, 3, 3, 3, 3, 3)
#: Per row and decision, the chance that one head is sent a random in-range
#: index whatever its mask (a masked choice to count; the fallbacks).
DEC_P_FORCE = 0.04
#: The decoder's coverage cases (each > 0 over corpus (c), else FAIL).
DEC_CASES = ("aim_seen", "aim_record", "aim_token", "aim_walk", "aim_keep", "trigger_0",
             "trigger_1", "reload", "melee", "shove", "use",
             *(f"throw{k}_arg{a}" for k in range(6) for a in range(5)),
             *(f"plant{a}" for a in range(4)), "fb_flashlight", "fb_onto_noaim",
             "fb_plant_onto", "throw_aim_held", "aim_lost", "aim_dead", "masked",
             "zombie_dies", "zombie_rises", "horde_empty_tick")
#: P8.6a's breaks: (OBS_BREAK, where) -- the kernels' codes, the late decode,
#: the late horde latch, the next observation's list_uid.
DEC_BREAKS = ("dec_every", "dec_div", "dec_late", "dec_horde_late", "dec_next_uid")


def dec_kit(session):
    """Corpus (c)'s state at tick 0: :func:`hand_kit`'s loadout row with
    :data:`DEC_GRENADES`, and those grenades in every marine's hands."""
    from dataclasses import replace
    from config import CFG
    from simulation.loadouts import get_tables
    from simulation.weapons import get_tables as wt
    by_name = get_tables().loadouts.by_name
    name = str(CFG.marine.loadout)
    rows = wt().hand_grenade_rows
    by_name[name] = replace(by_name[name],
                            grenades={r: DEC_GRENADES[k] for k, r in enumerate(rows)})
    for lo in {str(t.loadout) for t in session.spec.teams}:
        if lo in by_name:
            by_name[lo] = replace(by_name[lo],
                                  grenades={r: DEC_GRENADES[k] for k, r in enumerate(rows)})
    for u in session.sim.units:
        if not u.is_zombie:
            u.grenade_counts = list(DEC_GRENADES[:len(rows)])


def dec_crowd(session):
    """:func:`dec_kit`, and every unit (both teams and the horde) put at
    tick 0 into the open floor around team 0's first marine, two tiles
    apart (the reference and the twin alike): enemies seen and aimed at,
    zombies that bite and are shot, from the first decisions. The spawn's
    vision and picture are retaken from it."""
    from simulation import unit_fixed
    dec_kit(session)
    sim = session.sim
    one = unit_fixed.FP_ONE
    units = sorted(sim.units, key=lambda u: int(u.id))
    a = next(u for u in units if not u.is_zombie and int(u.team) == 0)
    cx, cy = int(a.tile_x), int(a.tile_y)
    taken = set()

    def fits(x, y, fp):
        cells = [(x + i, y + j) for j in range(fp) for i in range(fp)]
        ring = {(x + i, y + j) for j in range(-1, fp + 1) for i in range(-1, fp + 1)}
        return all(_free(sim, cx_, cy_) for cx_, cy_ in cells) and not (ring & taken)
    spots = sorted(((x, y) for y in range(cy - 12, cy + 13) for x in range(cx - 12, cx + 13)),
                   key=lambda t: ((t[0] - cx) ** 2 + (t[1] - cy) ** 2, t[1], t[0]))
    for u in units:
        fp = int(u.footprint)
        for x, y in spots:
            if fits(x, y, fp):
                u.set_pos_q(x * one, y * one)
                taken.update((x + i, y + j) for j in range(fp) for i in range(fp))
                break
    sim._vision_cache = None
    sim.start_team_picture(range(len(session.spec.teams)))


def latch_cols() -> list:
    """``(name, offset, width)`` of every store column ``LATCH_COLUMNS``
    writes (the decoder's outputs), in a fixed order."""
    from simulation import unit_store
    lay = unit_store.layout()
    names = []
    for col, _ in cd.LATCH_COLUMNS.values():
        if col not in names:
            names.append(col)
    return [(n, lay.column(n).offset, lay.column(n).width) for n in names]


def ref_latch_array(sim) -> np.ndarray:
    """Every unit's latch columns as the store encodes them (``sim.units``
    order = store rows)."""
    from simulation import unit_schema as us
    out = []
    for u in sim.units:
        row = []
        for n, _, w in latch_cols():
            enc = us.COLUMN_BY_NAME[n].encode(getattr(u, n))
            row += list(enc) if isinstance(enc, (tuple, list)) else [enc]
        out.append(row)
    return np.array(out, dtype=np.int64)


def dev_latch_array(rows) -> np.ndarray:
    """The same columns from a downloaded ``(U, W)`` store slice."""
    return np.concatenate([rows[:, o:o + w] for _, o, w in latch_cols()], axis=1)


class PolicyDriver:
    """One corpus (c) run: an in-process ``EnvRunner`` (the real
    ``AgentPolicy`` on v1 actions for every team, the commander, the horde),
    reset once; the play drives its session tick by tick as
    ``EnvRunner.step`` does (the decisions handed to each policy with the
    ``list_uid`` they were drawn on, then the ticks), recording what the
    gate compares."""

    def __init__(self, level, run_seed: int):
        import rl_env
        raw = {"run": {"seed": int(run_seed), "n_envs": 1, "decimation": DECIMATION,
                       "levels": [level.name], "policy": "policy",
                       "levels_dir": tempfile.gettempdir()},
               "obs": {"spec": "v1", "full_view": False},
               "reward": {"profile": "fireteam"}, "action": {"spec": "v1"}}
        run = rl_env.parse_run(raw, "corpus (c) run")
        layout = rl_env.make_layout(run, [level])
        self.runner = rl_env.EnvRunner(0, run, layout, [level], breach_physics=bp)
        self.runner.reset()
        self.session = self.runner.session
        self.seed = self.runner.seed
        self.policies = dict(self.runner.policies)
        self.S, self.T = layout.n_slots, layout.n_teams
        self.sizes = layout.head_sizes
        self.rng = np.random.default_rng(int(run_seed))
        self.host = None
        self.cov = {k: 0 for k in DEC_CASES}
        self.feet_seen = set()
        self.pol_uids = set()
        self.full = []

    # -- the observation and the decision ------------------------------------
    def aims_of(self, sim, units):
        aims = {}
        for c in self.session.controllers:
            view = getattr(c, "aim_view", None)
            if view is not None:
                aims.update(view())
        aimed = np.full(len(units), -1, dtype=np.int32)
        for i, u in enumerate(units):
            if u is not None:
                aimed[i] = aims.get(int(u.id), (-1, None))[0]
        return aims, aimed

    def observed(self, p, s, host):
        self.host = host
        keep = ("list_uid",) + tuple(_aspec.MASK_KEYS)
        d = {k: np.array(host[k], copy=True) for k in keep}
        d["list_type"] = np.array(host["list"][:, :, 0], copy=True)
        d["valid"] = np.array(host["valid"], copy=True)
        p.dec_obs[int(s.sim.tick)] = d

    def draw(self, host) -> np.ndarray:
        """Masked-random decisions (``tools/rl_smoke.py``'s drawer: each head
        uniform among its legal options, the argument among ``arg_mask``'s),
        then per row with :data:`DEC_P_FORCE` one head forced to a random
        in-range index."""
        rng = self.rng
        n = host["list_uid"].shape[0]
        out = np.zeros((n, 5), dtype=np.int32)
        for r in range(n):
            for h, k in enumerate(_aspec.MASK_KEYS):
                legal = np.flatnonzero(host[k][r])
                out[r, (0, 1, 2, 3)[h]] = int(rng.choice(legal))
            am = _aspec.arg_mask(out[r, 3], out[r, 1])
            out[r, 4] = int(rng.choice(np.flatnonzero(am)))
            if rng.random() < DEC_P_FORCE:
                h = int(rng.integers(5))
                out[r, h] = int(rng.integers(self.sizes[h]))
            # the no-sim fallbacks, forced now and then (masked choices too):
            # a throw onto the aimed unit with a compass aim, a plant "onto"
            f = rng.random()
            if f < DEC_P_FORCE / 2:
                out[r, 1] = int(rng.integers(_aspec.AIM_WALK))
                out[r, 3] = _aspec.HAND_THROW0 + int(rng.integers(self.sizes[3]
                                                                   - _aspec.HAND_THROW0))
                out[r, 4] = _aspec.ARG_ONTO
            elif f < DEC_P_FORCE:
                out[r, 3] = _aspec.HAND_PLANT
                out[r, 4] = _aspec.ARG_ONTO
            elif f < 1.5 * DEC_P_FORCE:
                out[r, 3] = _aspec.HAND_FLASHLIGHT
        return out

    def decide(self, p, s):
        tick = int(s.sim.tick)
        if tick % DECIMATION != 0:
            return
        a = self.draw(self.host)
        _aspec.check_actions(a[None], self.sizes)
        p.decisions[tick] = a
        masks = {k: self.host[k][None] for k in _aspec.MASK_KEYS}
        p.masked = getattr(p, "masked", {})
        p.masked[tick] = _aspec.masked_choices(a[None], masks)[0]
        self.cov["masked"] += int(np.count_nonzero(p.masked[tick]))
        lu = self.host["list_uid"]
        S = self.S
        for team, pol in self.policies.items():
            pol.set_decisions(a[team * S:(team + 1) * S], lu[team * S:(team + 1) * S])
        # coverage of the decisions on the rows that decode them
        typ = p.dec_obs[tick]["list_type"]
        A = _aspec
        for r in range(a.shape[0]):
            if not p.dec_obs[tick]["valid"][r]:
                continue
            feet, aim, trig, hand, arg = (int(x) for x in a[r])
            self.feet_seen.add(feet)
            self.cov[f"trigger_{trig}"] += 1
            if aim < A.AIM_WALK:
                self.cov["aim_token"] += 1
            elif aim == A.AIM_WALK:
                self.cov["aim_walk"] += 1
            elif aim == A.AIM_KEEP:
                self.cov["aim_keep"] += 1
            else:
                ty = int(typ[r, aim - A.AIM_TOKENS])
                if ty == 1:
                    self.cov["aim_seen"] += 1
                elif ty == 2:
                    self.cov["aim_record"] += 1
            unit_aim = (aim >= A.AIM_TOKENS and int(lu[r, aim - A.AIM_TOKENS]) >= 0
                        and int(typ[r, aim - A.AIM_TOKENS]) in A.AIMABLE_TYPES)
            if hand == A.HAND_FLASHLIGHT:
                self.cov["fb_flashlight"] += 1
            elif hand >= A.HAND_THROW0 and arg == A.ARG_ONTO and not unit_aim:
                self.cov["fb_onto_noaim"] += 1
            elif hand == A.HAND_PLANT and arg == A.ARG_ONTO:
                self.cov["fb_plant_onto"] += 1

    # -- the tick: MatchSession.tick, the reference's state taken between ----
    def tick(self, p, s):
        from simulation import device_decoder as ddec, horde_cache
        sim = s.sim
        tick = int(sim.tick)
        zomb0 = {int(u.id) for u in sim.units if u.is_zombie}
        living_z0 = {int(u.id) for u in sim.units if u.is_zombie and u.alive}
        prey0 = any(u.alive and not u.is_zombie for u in sim.units)
        for c in s.controllers:
            c.handle_frame(sim, None)
        p.ref_latch[tick] = ref_latch_array(sim)
        S = self.S
        p.ref_slots[tick] = {t * S + k: ddec.host_state_form(pol, k)
                             for t, pol in self.policies.items() for k in range(S)}
        p.ref_horde[tick] = horde_cache.latch_of(sim, len(sim.units))
        first = tick % DECIMATION == 0
        for t, pol in self.policies.items():
            recs = sim.team_picture.records(t)
            for k, st in pol._slot.items():
                u = sim.roster(t)[k] if k < len(sim.roster(t)) else None
                if u is None or not u.alive or u.team != t:
                    continue
                self.pol_uids.add(int(u.id))
                if st.aimed_uid >= 0:
                    if recs.get(st.aimed_uid) is None:
                        self.cov["aim_lost"] += 1
                    tgt = sim.get_unit(st.aimed_uid)
                    if tgt is not None and not tgt.alive:
                        self.cov["aim_dead"] += 1
                if st.throw_aim is not None and not first:
                    self.cov["throw_aim_held"] += 1
                if first and st.prev is not None:
                    hand, arg = int(st.prev[4]), int(st.prev[5])
                    A = _aspec
                    names = {A.HAND_RELOAD: "reload", A.HAND_MELEE: "melee",
                             A.HAND_SHOVE: "shove", A.HAND_USE: "use"}
                    if hand in names:
                        self.cov[names[hand]] += 1
                    elif hand == A.HAND_PLANT:
                        self.cov[f"plant{arg}"] += 1
                    elif hand >= A.HAND_THROW0:
                        self.cov[f"throw{hand - A.HAND_THROW0}_arg{arg}"] += 1
        if not living_z0 and prey0 and sim._zombie_requests is not None:
            # a tick the controller's horde latch is EMPTY while the
            # conductor's own call would still begin_tick (K8's difference)
            self.cov["horde_empty_tick"] += 1
        sim.step()
        if s._recorder is not None:
            s._recorder.after_step(sim)
        p.digests[int(sim.tick)] = cc.digests(sim)
        living_z1 = {int(u.id) for u in sim.units if u.is_zombie and u.alive}
        self.cov["zombie_dies"] += len(living_z0 - living_z1)
        self.cov["zombie_rises"] += len({int(u.id) for u in sim.units if u.is_zombie} - zomb0)

    def feed(self, intents):
        """The tap's intents the conductor is fed for a decoder-driven env:
        only the commander's orders (the decoder writes the policy units'
        columns, the card serves the horde)."""
        self.full.append(list(intents))      # the whole tap (diagnosis only)
        out = []
        for kind, args in intents:
            if kind == "order_intent":
                out.append((kind, args))
            elif kind == "drive_zombies":
                continue
            elif kind in cd.LATCH_COLUMNS and int(args[0]) in self.pol_uids:
                continue
            else:
                out.append((kind, args))
        return out

    def finish(self, p, s):
        self.cov["feet_options"] = len(self.feet_seen)
        p.dec_cov = dict(self.cov)


def corpus_c():
    """P8.6a's corpus: the policy runs of :data:`DEC_RUNS`, played once."""
    out = []
    for label, lvname, run_seed, cap, crowd in DEC_RUNS:
        lv = ship_level() if lvname == "ship_demo" else room_level(lvname)
        key = (label, -run_seed)
        if key not in _PLAYED:
            d = PolicyDriver(lv, run_seed)
            play(label, lv, d.session.spec, d.seed, setup=dec_crowd if crowd else dec_kit,
                 max_ticks=cap, driver=d)
            _PLAYED[key] = _PLAYED[(label, d.seed)]
        out.append(_PLAYED[key])
    return out


def decoder_for(obs, sessions):
    """The device decoder of a corpus (c) gate, with the check's break."""
    from simulation import device_decoder as ddec
    dec = ddec.DeviceDecoder(obs, [s.spec for s in sessions])
    if BREAK in ddec.BREAK_CODES:
        dec.brk = ddec.BREAK_CODES[BREAK]
    dec.late = BREAK == "dec_late"
    dec.horde_late = BREAK == "dec_horde_late"
    dec.snapping = True
    return dec


#: Per gate label: ticks whose latch columns / slot states / horde latch /
#: digests were compared equal, decisions handed over.
DECSTATS: dict = {}


def decoder_tick(label, c, obs, dec, played, tick, k) -> bool:
    """One tick of a decoder-driven gate: the decisions handed over on a
    decision tick (against the observation's own list_uid; ``dec_next_uid``:
    the NEXT observation's), the conductor's tick, then per env the store's
    latch columns after the decode, the slot states, the horde latch and the
    digests after the tick equal to the reference's."""
    from simulation import device_decoder as ddec
    import cupy as cp
    st = DECSTATS.setdefault(label, {"ticks": 0, "decisions": 0, "slots": 0, "horde": 0})
    N, n_rows = len(played), obs.n_rows
    if tick % DECIMATION == 0:
        acts = np.zeros((N, n_rows, 5), dtype=np.int32)
        due = np.zeros(N, dtype=bool)
        lu = None
        for e, p in enumerate(played):
            a = p.decisions.get(tick)
            if a is not None:
                acts[e, :a.shape[0]] = a
                due[e] = True
        if BREAK == "dec_next_uid":
            lu = obs.buffers["list_uid"].get()
            for e, p in enumerate(played):
                nxt = p.dec_obs.get(tick + DECIMATION)
                if nxt is not None:
                    lu[e, :nxt["list_uid"].shape[0]] = nxt["list_uid"]
        if due.any():
            dec.set_decisions(cp.asarray(acts), envs=due, list_uid=lu)
            st["decisions"] += int(due.sum())
            got = dec.masked.get()
            for e, p in enumerate(played):
                if due[e] and not np.array_equal(got[e, :p.decisions[tick].shape[0]],
                                                 p.masked[tick]):
                    fail(f"{label}: tick {tick}, env {e}: masked_choices "
                         f"{got[e].tolist()} vs the host's {p.masked[tick].tolist()}")
                    return False
    c.tick()
    rows = dec.snap.get()
    state = dec.host_state()
    hl = c.cache.latch.get()
    for e, p in enumerate(played):
        if k >= len(p.intents):
            continue
        want = p.ref_latch[tick]
        got = dev_latch_array(rows[e, :want.shape[0]])
        if not np.array_equal(want, got):
            idx = np.argwhere(want != got)[0]
            names = [n for n, _, w in latch_cols() for _ in range(w)]
            fail(f"{label}: tick {tick}, env {e}: store latch column {names[idx[1]]} of row "
                 f"{idx[0]} is {int(got[tuple(idx)])}, the reference's "
                 f"{int(want[tuple(idx)])} (after the decode)")
            return False
        for r, hs in p.ref_slots[tick].items():
            ds = ddec.state_form(state[e, r])
            if hs != ds:
                fail(f"{label}: tick {tick}, env {e}, row {r}: slot state {ds} vs the "
                     f"reference's {hs}")
                return False
            st["slots"] += hs is not None
        if e in dec.horde_envs:
            hw = p.ref_horde[tick]
            if not np.array_equal(hl[e, :len(hw)], hw) or hl[e, len(hw):].any():
                fail(f"{label}: tick {tick}, env {e}: horde latch {hl[e].tolist()} vs the "
                     f"zombie controller's {hw.tolist()}")
                return False
            st["horde"] += 1
        c.download([e])
        dg = cc.digests(c.sims[e])
        if dg != p.digests[tick + 1]:
            fail(f"{label}: tick {tick + 1}, env {e}: digests {dg} vs the reference's "
                 f"{p.digests[tick + 1]} (the decoded match diverged)")
            return False
        st["ticks"] += 1
    return True


def judge_decoder(played) -> None:
    """Corpus (c)'s coverage, each case > 0 (every feet option too)."""
    tot = {k: 0 for k in DEC_CASES}
    feet = 0
    for p in played:
        for k in DEC_CASES:
            tot[k] += p.dec_cov.get(k, 0)
        feet = max(feet, p.dec_cov.get("feet_options", 0))
    print(f"    decoder coverage: {tot}, feet options seen {feet} of {_aspec.N_FEET}")
    for k, n in tot.items():
        if n == 0:
            fail(f"decoder coverage: case {k!r} never occurred over corpus (c)")
    if feet < _aspec.N_FEET:
        fail(f"decoder coverage: {feet} of {_aspec.N_FEET} feet options occurred")
    for label, s in sorted(DECSTATS.items()):
        print(f"    decoder {label}: {s['ticks']} ticks equal (latch columns after the decode, "
              f"digests), {s['slots']} slot-ticks, {s['horde']} horde latches, "
              f"{s['decisions']} decisions")


def part_decoder():
    """P8.6a beyond part GATE: the refusals by name, an out-of-range decision
    raised at the books read with the host's message, the host syncs with the
    decoder (N = 1 and 2), the aim vector / onto reach sweep."""
    import cupy as cp
    from simulation import device_decoder as ddec
    # -- a team the decoder does not drive is refused at the build
    p = corpus_b()[0]
    s = twin(p)
    c = cd.DeviceConductor(s.sim, bp=bp)
    obs = dobs.DeviceObservation(c, [s.spec], p.S, DECIMATION, n_teams=p.T)
    try:
        ddec.DeviceDecoder(obs, [s.spec])
    except ddec.DecoderRefused as ex:
        print(f"  decoder: a {s.spec.teams[0].controller!r} team refused by name ({str(ex)[:80]}...)")
    else:
        fail("decoder: a bots-driven team was not refused at the build")
    # -- an out-of-range decision: the host's ValueError at the books read
    q = corpus_c()[1]
    s = twin(q)
    c = cd.DeviceConductor(s.sim, bp=bp)
    obs = dobs.DeviceObservation(c, [s.spec], q.S, DECIMATION, n_teams=q.T)
    dec = ddec.DeviceDecoder(obs, [s.spec])
    a = np.array(q.decisions[0], copy=True)
    a[1, 3] = dec.sizes[3]                       # the hand head one past its size
    try:
        _aspec.check_actions(a[None], dec.sizes)
    except ValueError as ex:
        host_msg = str(ex)
    dec.set_decisions(cp.asarray(a[None]))
    try:
        c.tick()
    except ValueError as ex:
        if host_msg in str(ex) and getattr(ex, "env", None) == 0:
            print(f"  decoder: out of range raised at the books read: {str(ex)[:120]}")
        else:
            fail(f"decoder: out-of-range message {str(ex)!r} is not the host's {host_msg!r}")
    else:
        fail("decoder: an out-of-range decision was not refused at the books read")
    # -- a tap column intent for a decoder unit is refused by name
    s = twin(q)
    c = cd.DeviceConductor(s.sim, bp=bp)
    obs = dobs.DeviceObservation(c, [s.spec], q.S, DECIMATION, n_teams=q.T)
    dec = ddec.DeviceDecoder(obs, [s.spec])
    try:
        c.latch([("set_aim", (int(s.sim.units[0].id), 65536, 0))], env=0)
    except cd.ConductorRefused as ex:
        print(f"  decoder: a tap intent for a decoder unit refused ({str(ex)[:80]}...)")
    else:
        fail("decoder: a tap column intent for a decoder-driven unit was accepted")
    # -- the host syncs and H2D copies per steady tick, N = 1 and 2
    counts = {}
    h2d = {}
    for n in (1, 2):
        sessions = [twin(q) for _ in range(n)]
        c = cd.DeviceConductor([x.sim for x in sessions], bp=bp)
        obs = dobs.DeviceObservation(c, [x.spec for x in sessions], q.S, DECIMATION,
                                     n_teams=q.T)
        dec = ddec.DeviceDecoder(obs, [x.spec for x in sessions])
        sc = cc.SyncCount()
        per, ups = [], []
        real_asarray = cp.asarray
        up = []

        def asarray(x, *k_, **kw):
            if isinstance(x, (np.ndarray, list, tuple)):
                up.append(1)
            return real_asarray(x, *k_, **kw)
        try:
            for k in range(min(40, len(q.intents))):
                tick = int(c.sims[0].tick)
                if tick % DECIMATION == 0:
                    dec.set_decisions(cp.asarray(np.stack([q.decisions[tick]] * n)))
                orders = [x for x in q.intents[k] if x[0] == "order_intent"]
                for e in range(n):
                    c.latch(q.intents[k], env=e)
                sc.py = []
                up.clear()
                cp.asarray = asarray
                bp.host_sync_log_begin()
                try:
                    c.tick()
                finally:
                    log = list(bp.host_sync_log_end()) + sc.py
                    cp.asarray = real_asarray
                per.append((k, log, bool(orders)))
                ups.append((k, len(up), bool(orders)))
        finally:
            sc.close()
        steady = [lg for k, lg, o in per[1:] if not o]
        names = sorted({x.partition(" | ")[2] for lg in steady for x in lg})
        counts[n] = [len(lg) for lg in steady]
        h2d[n] = [u for k, u, o in ups[1:] if not o]
        print(f"  decoder syncs N={n}: {np.mean(counts[n]):.2f} host syncs per steady tick, "
              f"named {names}; host-to-device uploads from Python per steady tick: "
              f"{sorted(set(h2d[n]))} ({dec.decodes} decodes)")
        extra = [x for x in names if x not in ("max_n_sub", "books_read")]
        if extra:
            fail(f"decoder syncs N={n}: a sync outside the conductor's named reads: {extra}")
        if any(h2d[n]):
            fail(f"decoder syncs N={n}: a steady tick uploaded from the host {h2d[n]}")
    if counts[1] != counts[2]:
        fail(f"decoder syncs: the per-tick count depends on N ({counts[1][:6]} vs {counts[2][:6]})")
    # -- the floats: aim_vector / onto_reach_q on the card against the host's
    # arithmetic over every offset a map of the corpus allows
    R = max(int(x) for x in (ship_level().tilemap.shape))
    ax = np.arange(-R, R + 1, dtype=np.int64)
    dx, dy = (v.ravel() for v in np.meshgrid(ax, ax))
    out = cp.zeros((len(dx), 4), dtype=cp.int64)
    ddx, ddy = cp.asarray(dx), cp.asarray(dy)
    bp.obs_dec_probe(len(dx), int(ddx.data.ptr), int(ddy.data.ptr), int(out.data.ptr),
                     1 if BREAK == "dec_div" else 0)
    got = out.get()
    qx, qy, none = ddec.aim_vector_host(dx, dy)
    import math
    from simulation.unit_fixed import quantize_scalar
    reach = np.array([quantize_scalar(math.sqrt(float(a * a + b * b)))
                      for a, b in zip(dx[::97], dy[::97])], dtype=np.int64)
    dqx, dqy, _ = ddec.aim_vector_host(dx, dy, divide=True)
    alt = int(np.count_nonzero((dqx != qx) | (dqy != qy)))
    bad = int(np.count_nonzero((got[:, 0] != qx) | (got[:, 1] != qy) | (got[:, 2] != none)))
    bad_r = int(np.count_nonzero(got[::97, 3] != reach))
    # the host arithmetic is aim_vector itself (a sample through the function)
    from types import SimpleNamespace as NS
    me = NS(center_tile_x=lambda: 0, center_tile_y=lambda: 0)
    samp = [(int(a), int(b)) for a, b in zip(dx[::211], dy[::211])]
    host_bad = sum(1 for (a, b), i in zip(samp, range(0, len(dx), 211))
                   if (_aspec.aim_vector(me, (a, b)) or (0, 0)) != (int(qx[i]), int(qy[i])))
    print(f"  decoder floats: {len(dx)} offsets (|dx|, |dy| <= {R}): aim vector differs on {bad}, "
          f"the onto reach on {bad_r} of {len(reach)}; aim_vector itself vs its numpy form "
          f"differs on {host_bad} of {len(samp)}; dx / mag in place of dx * (1 / mag) moves "
          f"the quantized vector on {alt}")
    if bad or bad_r or host_bad:
        fail(f"decoder floats: the card's aim vector / reach differ from the host's "
             f"({bad} / {bad_r} / {host_bad})")


def part_dec_gate():
    """P8.6a's corpus (c) alone (part GATE runs it too, with the census)."""
    for p in corpus_c():
        try:
            gate(f"(c) {p.name}", [p])
        except Exception as ex:                    # a refusal names its slot: a finding
            fail(f"(c) {p.name}: {type(ex).__name__}: {ex}")
    judge_decoder(corpus_c())


PARTS = {"gate": part_gate, "batch": part_batch, "handover": part_handover,
         "syncs": part_syncs, "ended": part_ended, "floatkit": part_floatkit,
         "sight": part_sight, "wall": part_wall, "restage": part_restage,
         "books": part_books, "decoder": part_decoder, "dec_gate": part_dec_gate}


def main(parts=None) -> int:
    from simulation import physics_runner as pr
    pr.set_residency(False)               # the reference runs the CPU path (the oracle)
    t0 = time.perf_counter()
    print(f"OBS check at {PATCH}: owned {owned()}")
    for name in (parts or list(PARTS)):
        print(f"PART {name}")
        PARTS[name]()
    print(f"{time.perf_counter() - t0:.1f} s")
    print("OBS_RESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
