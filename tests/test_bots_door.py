"""#195 1j -- the bots through the door (``src/control_bots_door.py``).

The layered bots drive a team through action spec v1: an inner ``BotTeam``
decides on a capture proxy, an inner ``AgentPolicy``'s v1 decoder is the
team's one issuer, the translation is the nearest LEGAL choice through the
1h helpers, and the imitation records are the env's own arrays.

Note: ``docs/rl_agent_rung1_note_1j_2026-11-13.md``. Every test's docstring
names the property it protects and the change that must break it. No test
pins a count, an order or a set of something designed to grow.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import control_bots  # noqa: E402
import control_bots_door as door  # noqa: E402
import control_source  # noqa: E402
import level_lib  # noqa: E402
import level_loader  # noqa: E402
import match_replay as mr  # noqa: E402
import rl_env  # noqa: E402
from control_source import ControlSource  # noqa: E402
from simulation import action_spec_v1 as A  # noqa: E402
from simulation.match_session import MatchSession  # noqa: E402
from simulation.turning import PI_Q16, wrap_q16  # noqa: E402
from simulation.unit_fixed import atan2_q16  # noqa: E402

POLICY_KINDS = door._policy_kinds()
LATCHES = {"throw_grenade_intent", "use_intent", "reload_intent",
           "shove_intent", "melee_intent", "plant_charge_intent"}


def _ship_demo():
    return level_loader.load("ship_demo", str(ROOT / "levels"))


def _spec(lvl, a="bots_door", b="bots", *, zombies=None, limit=40.0,
          sight="all_round"):
    t0, t1 = lvl.match.teams
    sp = replace(lvl.match, teams=(replace(t0, controller=a),
                                   replace(t1, controller=b)),
                 time_limit_s=float(limit), sight=sight)
    if zombies is not None:
        sp = replace(sp, zombies=int(zombies))
    return sp


class _Tap:
    """The intent tap's sink, forwarding to the one attached before it (a
    recording's log) so both see every call."""

    def __init__(self, prev=None):
        self.calls = []
        self.prev = prev

    def record(self, tick, kind, args):
        self.calls.append((int(tick), kind, tuple(args)))
        if self.prev is not None:
            self.prev.record(tick, kind, args)


def _door_of(ses):
    return [c for c in ses.controllers
            if isinstance(c, door.BotsThroughDoor)]


class _OrderCapture(door.IntentCapture):
    """A capture proxy that also keeps ``order_intent`` calls (the shadow
    commander's), instead of refusing them."""

    def order_intent(self, *args):
        self.calls.append((int(self.tick), "order_intent",
                           tuple(int(a) for a in args)))


class _Played:
    """One door match driven tick by tick, everything the tests read."""


def _play(lvl, spec, *, seed=1, sink=None, record=False, shadow=False,
          max_ticks=None, digests=True, watch=None):
    """Run ``spec`` on ``lvl`` to its end through a ``MatchSession`` (the
    run_match loop), with a tap; per tick: the digests, the door team's
    issued verbs, its captured holds and (``shadow``) a shadow commander's
    orders on a capture proxy."""
    p = _Played()
    door.install_sink(sink)
    try:
        ses = MatchSession(lvl, spec, seed=seed, breach_physics=bp)
    finally:
        door.install_sink(None)
    if record:
        ses.record(20)
    tap = _Tap(ses.sim._intent_log)
    ses.sim.attach_intent_log(tap)
    ctl = _door_of(ses)
    sh = control_bots.BotCommander(ctl[0].team) if shadow else None
    p.digests, p.issued, p.holds, p.shadow_orders = [], {}, {}, {}
    p.watched, p.centres = [], {}
    while not ses.is_terminal():
        if max_ticks is not None and ses.sim.tick >= max_ticks:
            break
        t = int(ses.sim.tick)
        if sh is not None:
            cap = _OrderCapture(ses.sim)
            sh.handle_frame(cap, None)
            p.shadow_orders[t] = [a for _t, k, a in cap.calls
                                  if k == "order_intent"]
        if ctl:
            tm = ctl[0].team
            p.centres[t] = (
                {uid: (int(r.centre[0]), int(r.centre[1])) for uid, r
                 in ses.sim.team_picture.records(tm).items()},
                {u.id: (u.center_tile_x(), u.center_tile_y())
                 for u in ses.sim.units if u.team == tm})
        ses.tick()
        for c in ctl:
            p.issued[(c.team, t)] = list(c._issued_now)
            p.holds[(c.team, t)] = dict(c._holds_now)
        if digests:
            p.digests.append(ses.digests())
        if watch is not None:
            p.watched.append(watch(ses.sim))
    p.result = ses.result()
    p.recording = ses.finish() if ses.is_terminal() else None
    p.sim, p.ses, p.tap, p.ctl = ses.sim, ses, tap, ctl
    p.stats = None
    if sink is not None:
        p.stats = sink.close(ses.sim, p.result, ses.digests(), seed=seed,
                             recording=p.recording)
    return p


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """ONE recorded door match on ship_demo (all-round sight, horde in, a
    40 s clock): bots_door (team 0) against bots, records on, a shadow
    commander beside it."""
    out = tmp_path_factory.mktemp("records")
    lvl = _ship_demo()
    spec = _spec(lvl)
    sink = door.RecordSink(out)
    p = _play(lvl, spec, sink=sink, record=True, shadow=True)
    p.lvl, p.spec, p.out = lvl, spec, out
    files = sorted(out.rglob("*.npz"))
    assert len(files) == 1
    p.arrays, p.meta = door.read_record(files[0])
    p.file = files[0]
    return p


# ---------------------------------------------------------------------------
# The capture
# ---------------------------------------------------------------------------
class _Through(ControlSource):
    """A BotTeam driven through the pass-through capture."""

    def __init__(self, team):
        self.bots = control_bots.BotTeam(team)

    def handle_frame(self, sim, renderer=None):
        self.bots.handle_frame(door.IntentCapture(sim, passthrough=True),
                               None)


def test_the_capture_is_lossless():
    """Property: a ``BotTeam`` driven through ``IntentCapture(passthrough=
    True)`` plays a ship_demo bots-vs-bots segment (600 ticks) with the same
    ``(field_digest, tick_digest)`` on every tick as a plain ``BotTeam``.
    Breaks if the proxy drops, reorders or alters a verb, or a forwarded
    read or write diverges (a second sight cache)."""
    lvl = _ship_demo()
    spec = _spec(lvl, "bots", "bots", limit=300.0)
    runs = []
    for wrap in (False, True):
        ses = MatchSession(lvl, spec, seed=3, breach_physics=bp)
        if wrap:
            ses.controllers[0] = _Through(0)
        d = []
        while not ses.is_terminal() and ses.sim.tick < 600:
            ses.tick()
            d.append(ses.digests())
        runs.append(d)
    assert len(runs[0]) == len(runs[1]) > 100
    for t, (a, b) in enumerate(zip(*runs)):
        assert a == b, f"diverged at tick {t}"


def test_the_capture_refuses_the_verbs_no_soldier_has():
    """Property: a captured controller calling ``order_intent``,
    ``drive_zombies`` or ``apply_action`` is refused by name, and a
    captured verb never reaches the sim's tap without ``passthrough``.
    Breaks if the proxy forwards a verb."""
    lvl = _ship_demo()
    ses = MatchSession(lvl, _spec(lvl, "bots", "bots"), seed=1,
                       breach_physics=bp)
    tap = _Tap()
    ses.sim.attach_intent_log(tap)
    cap = door.IntentCapture(ses.sim)
    u = ses.sim.roster(0)[0]
    for kind in door.REFUSED_ON_CAPTURE:
        with pytest.raises(door.DoorRefused, match=kind):
            getattr(cap, kind)(u.id)
    cap.set_trigger(u.id, True)
    cap.throw_grenade_intent(u.id, 65536, 0, 10, reach_q=3 << 16, kind=0)
    assert tap.calls == []
    assert [k for _t, k, _a in cap.calls] == ["set_trigger",
                                              "throw_grenade_intent"]
    assert cap.calls[1][2] == (u.id, 65536, 0, 10, 3 << 16, 0)
    assert u.live_trigger is not True


def test_capture_only_every_soldier_verb_is_the_decoders(recorded):
    """Property: with the intent tap attached, every soldier verb the sim
    received for the door team equals, tick for tick and in order, what the
    decoder issued (the pass-through capture's list) -- no inner bot's
    capture leaks. Breaks if a verb leaks past the proxy."""
    p = recorded
    team = p.ctl[0].team
    ids = {u.id for u in p.sim.units if u.team == team or
           u.id in {r.id for r in p.sim.roster(team)}}
    got = {}
    for t, k, a in p.tap.calls:
        if k in POLICY_KINDS and a[0] in ids:
            got.setdefault(t, []).append((k, a))
    for (tm, t), calls in p.issued.items():
        want = [(k, a) for _t, k, a in calls]
        assert got.get(t, []) == want, t
    assert set(got) <= {t for (_tm, t) in p.issued}


# ---------------------------------------------------------------------------
# Nothing masked; the cadence; the latches
# ---------------------------------------------------------------------------
def test_nothing_masked_is_sent_and_the_records_say_so(recorded):
    """Property: every recorded decision's five integers are in range and
    legal under the recorded masks and the realised argument mask, and the
    match's stats count no masked choice (the controller raises
    ``DoorRefused`` before sending one). Breaks if the translator sends a
    masked option or ignores ``arg_mask``."""
    a = recorded.arrays
    acts = a["actions"].astype(np.int64)
    assert len(acts) > 50
    A.check_actions(acts, recorded.ctl[0].tables.sizes)
    masks = {k: a[k] for k in A.MASK_KEYS}
    assert int(A.masked_choices(acts, masks).sum()) == 0
    am = A.arg_mask(acts[..., 3], acts[..., 1])
    assert (np.take_along_axis(am, acts[..., 4:5], -1) == 1).all()
    assert (a["arg_mask"] == am).all()
    st = recorded.stats[str(recorded.ctl[0].team)]
    assert st["masked_sent"] == 0 and st["slot_decisions"] > 0


def test_latches_leave_only_on_decision_ticks_and_holds_every_tick(
        recorded):
    """Property: a soldier latch of the door team reaches the game only on
    a decision tick (T % D == 0), and ``set_decisions`` was called exactly
    on the decision ticks the match passed. Breaks if a latch leaks off a
    decision tick or the cadence is wired wrong."""
    p = recorded
    c = p.ctl[0]
    D = c.settings.decision_ticks
    for (tm, t), calls in p.issued.items():
        for _t, k, _a in calls:
            if k in LATCHES:
                assert t % D == 0, (t, k)
    ticks = sorted(t for (_tm, t) in p.issued)
    assert c.decision_ticks == [t for t in ticks if t % D == 0]
    assert list(p.arrays["tick"]) == c.decision_ticks


@pytest.mark.parametrize("name", ["bots_door_every6", "bots_door_every8",
                                  "bots_door_trigger_tick"])
def test_the_cadence_and_the_trigger_variant(name):
    """Property: under the D = 6 / 8 variants ``set_decisions`` runs exactly
    on the ticks T % D == 0; under ``bots_door_trigger_tick`` the issued
    trigger equals the bot's captured trigger on every tick that is not a
    throw's decision tick, and no latch leaves off a decision tick. Breaks
    if the rate or the trigger variant is wired wrong."""
    lvl = _ship_demo()
    p = _play(lvl, _spec(lvl, name, "bots", limit=300.0), seed=2,
              max_ticks=400, digests=False)
    c = p.ctl[0]
    D = c.settings.decision_ticks
    ticks = sorted(t for (_tm, t) in p.issued)
    assert c.decision_ticks == [t for t in ticks if t % D == 0]
    for (tm, t), calls in p.issued.items():
        for _t, k, _a in calls:
            if k in LATCHES:
                assert t % D == 0
        if not c.settings.trigger_per_tick:
            continue
        throws = {a[0] for _t, k, a in calls
                  if k == "throw_grenade_intent"}
        holds = p.holds[(tm, t)]
        for _t, k, a in calls:
            if k == "set_trigger" and a[0] not in throws:
                assert a[1] == bool(holds.get(a[0], {}).get("trigger")), t


# ---------------------------------------------------------------------------
# The translation, on its own (synthetic slots)
# ---------------------------------------------------------------------------
class _U:
    def __init__(self, cx, cy):
        self._c = (cx, cy)

    def center_tile_x(self):
        return self._c[0]

    def center_tile_y(self):
        return self._c[1]


def _tables():
    return A.ActionTables(
        n_slots=4, n_entries=37, hand_grenade_rows=("a", "b", "c"),
        throttles=(0, 32768, 65536), fuse_ticks=72,
        reaches_q=tuple((j + 1) * (12 << 16) // 4 for j in range(4)),
        plant_timers=(120, 240, 480, 1440), aim_lo=3, aim_hi=19)


def _masks(tb, *, entries=(), hand=(0,), keep_only=False):
    sz = tb.sizes
    m = {"mask_feet": np.ones(sz[0], np.uint8),
         "mask_aim": np.zeros(sz[1], np.uint8),
         "mask_trigger": np.ones(2, np.uint8),
         "mask_hand": np.zeros(sz[3], np.uint8)}
    if keep_only:
        m["mask_aim"][A.AIM_KEEP] = 1
    else:
        m["mask_aim"][:A.AIM_KEEP] = 1
        for i in entries:
            m["mask_aim"][A.AIM_TOKENS + i] = 1
    for h in hand:
        m["mask_hand"][h] = 1
    return m


def _slot(tb, masks, *, me=(10, 10), move=None, aim=None, trigger=None,
          window=(), entries=None, tick=8):
    uid = np.full(tb.n_entries, -1, np.int32)
    typ = np.zeros(tb.n_entries, np.float32)
    rec = {}
    for i, (u, c, t) in (entries or {}).items():
        uid[i], typ[i] = u, t
        rec[u] = c
    return door.SlotInput(unit=_U(*me), move=move, aim=aim, trigger=trigger,
                          window=list(window), list_uid=uid, list_type=typ,
                          masks=masks, records=rec, tick=tick)


def _bin_ref(dx, dy, stride):
    """The test's own nearest bin: the minimum wrapped angular distance
    over every ``stride``-th compass vector's own atan2 (ties to the
    counter-clockwise one)."""
    a = atan2_q16(-dy, dx)
    best = None
    for d in range(0, 16, stride):
        bx, by = A.COMPASS_Q16[d]
        e = int(wrap_q16(a - atan2_q16(-by, bx)))
        key = (abs(e), 0 if e <= 0 else 1)
        if best is None or key < best[0]:
            best = (key, d)
    return best[1]


def test_feet_is_the_nearest_bin_and_throttle():
    """Property: the sent feet option is the compass bin nearest the bot's
    move by ``atan2_q16`` among every ``stride``-th (the test's own
    minimum, within one Q16 count of a tie), the throttle the nearest of
    walk / half / full, a cleared or absent move is stand. Breaks on a
    second rounding or a second table."""
    tb = _tables()
    rng = np.random.default_rng(5)
    from control_source import quantize_stick_direction
    for stride in (1, 2):
        st = door.DoorSettings(dir_stride=stride)
        for _ in range(400):
            x, y = rng.uniform(-1, 1, 2)
            dx, dy, mag = quantize_stick_direction(float(x), float(y), 0.0)
            if mag == 0:
                continue
            thr = int(rng.integers(0, 65537))
            five, _ = door.translate(
                _slot(tb, _masks(tb), move=("set", dx, dy, thr)), tb, st)
            d, s = A.feet_option(five[0])
            a = atan2_q16(-dy, dx)
            ref = _bin_ref(dx, dy, stride)
            if d != ref:                      # only a near tie may differ
                e1 = abs(int(wrap_q16(a - atan2_q16(-A.COMPASS_Q16[d][1],
                                                    A.COMPASS_Q16[d][0]))))
                e2 = abs(int(wrap_q16(a - atan2_q16(
                    -A.COMPASS_Q16[ref][1], A.COMPASS_Q16[ref][0]))))
                assert abs(e1 - e2) <= 2, (dx, dy, d, ref)
            assert d % stride == 0
            assert s == min(range(3), key=lambda j: abs(tb.throttles[j]
                                                         - thr))
        for mv in (None, ("clear",), ("set", 0, 0, 0)):
            five, _ = door.translate(_slot(tb, _masks(tb), move=mv), tb, st)
            assert five[0] == A.FEET_STAND


def test_the_stride_keeps_16_bins_and_coarsens_to_8():
    """Property: at ``stride = 1`` the helpers return the 1h formula's bin
    bit for bit; at 2 every bin is even and within half a coarse bin (pi/8,
    plus one count) of the vector. Breaks if the stride changes the
    16-bin chooser."""
    rng = np.random.default_rng(11)
    for _ in range(2000):
        dx, dy = (int(v) for v in rng.integers(-65536, 65537, 2))
        if dx == 0 and dy == 0:
            continue
        a = atan2_q16(-dy, dx)
        old = ((a * 8 + (PI_Q16 >> 1)) // PI_Q16) % 16
        assert A._nearest_dir(dx, dy) == old
        assert A.aim_index_for_vector(dx, dy) == old
        d2 = A.aim_index_for_vector(dx, dy, stride=2)
        assert d2 % 2 == 0
        b = atan2_q16(-A.COMPASS_Q16[d2][1], A.COMPASS_Q16[d2][0])
        assert abs(int(wrap_q16(a - b))) <= (PI_Q16 >> 3) + 1


def test_the_aim_entry_walk_and_compass_rules():
    """Property: a captured aim equal to ``aim_vector`` of an aimable legal
    entry's record centre sends that entry (the first, a second on the same
    bearing counted ambiguous); an aim equal to the move or no aim sends
    "where I walk"; a unit the list does not hold falls to the nearest
    token, counted missing; any other vector the nearest token, counted
    rounded; a keep-only mask sends keep. Breaks if unit aims become
    tokens, the walk is rounded to a token, or the keep mask is ignored."""
    tb, st = _tables(), door.DoorSettings()
    ent = {3: (77, (20, 10), 1.0), 5: (78, (10, 30), 2.0)}
    m = _masks(tb, entries=(3, 5))
    v = A.aim_vector(_U(10, 10), (20, 10))
    five, loss = door.translate(_slot(tb, m, aim=v, entries=ent), tb, st)
    assert five[1] == A.AIM_TOKENS + 3 and loss.bits == 0
    two = dict(ent)
    two[6] = (79, (30, 10), 2.0)                  # the same bearing as 3
    five, loss = door.translate(_slot(tb, _masks(tb, entries=(3, 5, 6)),
                                      aim=v, entries=two,
                                      move=("set",) + v + (0,)), tb, st)
    assert five[1] == A.AIM_TOKENS + 3       # an entry wins over the walk
    assert loss.bits & door._BIT["aim_ambiguous"]
    w = (46341, 46341)
    five, _ = door.translate(_slot(tb, m, aim=w, move=("set",) + w + (0,)),
                             tb, st)
    assert five[1] == A.AIM_WALK
    five, _ = door.translate(_slot(tb, m), tb, st)
    assert five[1] == A.AIM_WALK
    # entry 3 masked: the bot's target is missing from the aimable list
    five, loss = door.translate(_slot(tb, _masks(tb, entries=(5,)), aim=v,
                                      entries=ent), tb, st)
    assert five[1] == 0 and loss.bits & door._BIT["aim_missing"]
    odd = A.aim_vector(_U(0, 0), (7, -3))
    five, loss = door.translate(_slot(tb, m, aim=odd), tb, st)
    assert five[1] == A.aim_index_for_vector(*odd)
    assert loss.bits & door._BIT["aim_rounded"]
    five, _ = door.translate(_slot(tb, _masks(tb, keep_only=True), aim=v,
                                   entries=ent), tb, st)
    assert five[1] == A.AIM_KEEP


def _latch(t, kind, args, c=(10, 10)):
    return door.Latch(t, kind, args, c)


def test_the_latch_window_newest_legal_and_counted():
    """Property: the hand is the newest latch of the window that is legal
    at the decision tick; a newer masked one gives way to it (counted
    ``hand_masked``), two latches count ``superseded``, a latch from before
    the tick counts ``carried`` with its delay; none legal sends nothing.
    Breaks if a carried latch is lost, the mask is not read, or the newest
    is not preferred."""
    tb, st = _tables(), door.DoorSettings()
    win = [_latch(5, "use_intent", (1,)), _latch(6, "reload_intent", (1,)),
           _latch(8, "melee_intent", (1,))]
    five, loss = door.translate(_slot(tb, _masks(tb, hand=(0, 1, 4)),
                                      window=win), tb, st)
    assert five[3] == A.HAND_RELOAD and five[4] == 0
    b = loss.bits
    assert b & door._BIT["hand_masked"] and b & door._BIT["superseded"]
    assert b & door._BIT["carried"] and loss.carried_ticks == 2
    five, loss = door.translate(_slot(tb, _masks(tb, hand=(0,)),
                                      window=win), tb, st)
    assert five[3] == A.HAND_NOTHING
    five, loss = door.translate(_slot(tb, _masks(tb, hand=(0, 2)),
                                      window=win[2:]), tb, st)
    assert five[3] == A.HAND_MELEE and loss.bits == 0


def test_the_throw_lands_nearest_and_releases_the_trigger():
    """Property: on a decision whose hand is a throw, (aim, arg) is the
    legal pair of the two families whose landing lies nearest the bot's
    point (recomputed exhaustively here), the trigger is 0 (counted when
    the bot held it), the kind is the bot's; under ``throws = "direction"``
    only compass aims are sent. Breaks if the throw's aim is the fire
    target's or the argument is not the nearest."""
    tb = _tables()
    ent = {2: (90, (17, 13), 1.0), 4: (91, (4, 22), 2.0)}
    rng = np.random.default_rng(3)
    for rule in ("nearest", "direction"):
        st = door.DoorSettings(throws=rule)
        for _ in range(60):
            from control_source import quantize_stick_direction
            x, y = rng.uniform(-1, 1, 2)
            dx, dy, _m = quantize_stick_direction(float(x), float(y), 0.0)
            reach = int(rng.integers(1 << 16, 12 << 16))
            kind = int(rng.integers(0, 3))
            m = _masks(tb, entries=(2, 4), hand=(0, A.HAND_THROW0 + kind))
            fire = A.aim_vector(_U(10, 10), (17, 13))
            five, loss = door.translate(_slot(
                tb, m, aim=fire, trigger=True, entries=ent,
                window=[_latch(8, "throw_grenade_intent",
                               (1, dx, dy, 72, reach, kind))]), tb, st)
            assert five[3] == A.HAND_THROW0 + kind and five[2] == 0
            assert loss.bits & door._BIT["trigger_throw"]
            px, py = (10 << 32) + dx * reach, (10 << 32) + dy * reach
            cands = []
            if rule == "nearest":
                for i, (_u, c, _t) in ent.items():
                    cands.append((((c[0] << 32) - px) ** 2
                                  + ((c[1] << 32) - py) ** 2,
                                  A.AIM_TOKENS + i, A.ARG_ONTO))
            for d in range(16):
                for j, r in enumerate(tb.reaches_q):
                    lx = (10 << 32) + A.COMPASS_Q16[d][0] * r
                    ly = (10 << 32) + A.COMPASS_Q16[d][1] * r
                    cands.append(((lx - px) ** 2 + (ly - py) ** 2, d, j))
            best = min(c[0] for c in cands)
            got = [c for c in cands if (c[1], c[2]) == (five[1], five[4])]
            assert got and got[0][0] == best
            if rule == "direction":
                assert five[1] < A.AIM_WALK
            am = A.arg_mask(five[3], five[1])
            assert am[five[4]]


# ---------------------------------------------------------------------------
# On the match: the aim tracks, the wind-up keeps, the orders agree
# ---------------------------------------------------------------------------
def test_the_aim_at_an_entry_tracks_its_record_every_tick(recorded):
    """Property: on every tick of a decision whose aim is list entry X (no
    throw), the decoder's issued ``set_aim`` equals ``aim_vector`` from the
    soldier's centre to X's record centre ON THAT TICK (the record
    followed, never a kept list position or a frozen vector); such ticks
    occur, and on some the record moved within the decision. Breaks if unit
    aims become compass tokens or a position is kept."""
    p = recorded
    a = p.arrays
    D = p.ctl[0].settings.decision_ticks
    team = p.ctl[0].team
    roster = p.sim.roster(team)
    checked = moved = 0
    for i, t0 in enumerate(a["tick"]):
        t0 = int(t0)
        for s in range(a["actions"].shape[1]):
            aim = int(a["actions"][i, s, 1])
            if aim < A.AIM_TOKENS or a["actions"][i, s, 3] >= A.HAND_THROW0:
                continue
            uid = int(a["list_uid"][i, s, aim - A.AIM_TOKENS])
            me = roster[s]
            first = None
            for t in range(t0, t0 + D):
                if t not in p.centres:
                    break
                recs, mine = p.centres[t]
                if uid not in recs or me.id not in mine:
                    break
                vecs = [args[1:] for _t, k, args in p.issued[(team, t)]
                        if k == "set_aim" and args[0] == me.id]
                if not vecs:
                    break
                c, m = recs[uid], mine[me.id]
                dx, dy = c[0] - m[0], c[1] - m[1]
                if dx == 0 and dy == 0:
                    break
                want = A.aim_vector(_U(*m), c)
                assert vecs[-1] == want, (t, s, uid)
                checked += 1
                first = c if first is None else first
                moved += c != first
    assert checked > 20 and moved > 0


def test_the_wind_up_keeps_the_throws_aim(recorded):
    """Property: while a throw winds up past a decision's last tick the
    sent aim is "keep"; on the ticks of the throw's wind-up the issued aim
    stays the throw decision's. Breaks if a later decision's aim steers the
    grenade."""
    p = recorded
    a = p.arrays
    D = p.ctl[0].settings.decision_ticks
    acts = a["actions"]
    keep = acts[..., 1] == A.AIM_KEEP
    # keep is sent exactly where the recorded mask offered only keep
    only = (a["mask_aim"].sum(-1) == 1) & (a["mask_aim"][..., A.AIM_KEEP]
                                           == 1)
    assert (keep == only).all()
    throws = np.argwhere(acts[..., 3] >= A.HAND_THROW0)
    if len(throws) == 0:
        pytest.skip("no throw in the recorded match")
    assert keep.any()          # a 7-tick wind-up outlasts a 4-tick decision
    team = p.ctl[0].team
    for i, s in throws:
        t0 = int(a["tick"][i])
        uid = p.sim.roster(team)[s].id
        aims = [args[1:] for t in range(t0, t0 + D)
                for _t, k, args in p.issued.get((team, t), [])
                if k == "set_aim" and args[0] == uid]
        if int(acts[i, s, 1]) < A.AIM_WALK:
            assert all(v == A.COMPASS_Q16[int(acts[i, s, 1])] for v in aims)


def test_the_door_team_writes_its_own_brains_orders(recorded):
    """Property: a ``bots_door`` team gets no ``commander`` from
    ``create_match_controllers``; its ``order_intent`` calls equal, tick for
    tick, those of a shadow ``BotCommander`` run beside it on a capture
    proxy (note section 2: the two choices agree today). Breaks if
    ``bots_door`` joins ``DOOR_CONTROLLERS``, or the lifted functions drift
    from the commander's."""
    p = recorded
    assert not any(isinstance(c, control_bots.BotCommander)
                   for c in p.ses.controllers)
    assert "bots_door" not in control_source.DOOR_CONTROLLERS
    got = {}
    for t, k, a in p.tap.calls:
        if k == "order_intent":
            got.setdefault(t, []).append(a)
    assert got, "no order was issued"
    for t, want in p.shadow_orders.items():
        assert got.get(t, []) == want, t


# ---------------------------------------------------------------------------
# The records
# ---------------------------------------------------------------------------
def test_the_records_drive_the_game(recorded):
    """Property: the recorded match replayed with its door team replaced by
    a plain ``policy`` team fed the recorded ``actions`` and ``list_uid`` at
    each recorded tick plays with the same digests on every tick (the policy
    team gets the commander, so this also holds section 2's agreement).
    Breaks if a record misses something that drove the game."""
    p = recorded
    a = p.arrays
    team = p.ctl[0].team
    t0, t1 = p.spec.teams
    spec = replace(p.spec, teams=(replace(t0, controller="policy"), t1))
    ses = MatchSession(p.lvl, spec, seed=1, breach_physics=bp)
    pol = ses.controllers[team]
    pol.configure("v1", A.action_tables(ses.sim, p.meta["S"]))
    at = {int(t): i for i, t in enumerate(a["tick"])}
    n = 0
    while not ses.is_terminal():
        i = at.get(int(ses.sim.tick))
        if i is not None:
            pol.set_decisions(a["actions"][i].astype(np.int64),
                              a["list_uid"][i])
        ses.tick()
        assert ses.digests() == p.digests[n], f"diverged at tick {n}"
        n += 1
    assert n == len(p.digests)


def test_the_records_are_deterministic_and_change_nothing(recorded,
                                                          tmp_path):
    """Property: the same match again writes equal arrays and the same
    digests; the same match with no sink (no records, no books) plays the
    same digests on every tick. Breaks on hidden state, or if the sink or
    the books write to the sim."""
    p = recorded
    again = _play(p.lvl, p.spec, sink=door.RecordSink(tmp_path),
                  record=True)
    files = sorted(tmp_path.rglob("*.npz"))
    arr, meta = door.read_record(files[0])
    assert set(arr) == set(p.arrays)
    for k in arr:
        assert np.array_equal(arr[k], p.arrays[k]), k
    assert again.digests == p.digests
    bare = _play(p.lvl, p.spec)
    assert bare.digests == p.digests


def test_the_records_tie_to_their_match(recorded, tmp_path):
    """Property: ``meta`` holds the level id, seed, spec, controllers, team,
    build ids, result, digests and the replay file's sha256; the replay
    file re-simulates to the meta's final digests; ``read_record`` refuses
    another ``record_format``. Breaks if a record cannot be tied to its
    match."""
    import hashlib
    import json
    from run_match import run_match
    m = recorded.meta
    for k in ("level_id", "seed", "spec", "controllers", "team", "build",
              "bp_build_id", "result", "field_digest", "tick_digest",
              "replay", "replay_sha256", "settings", "S", "E", "K",
              "hand_names", "environment", "loss_totals"):
        assert k in m, k
    assert m["level_id"] == "ship_demo" and m["team"] == 0
    assert m["controllers"] == ["bots_door", "bots"]
    rp = recorded.file.parent / m["replay"]
    assert hashlib.sha256(rp.read_bytes()).hexdigest() == m["replay_sha256"]
    o = run_match(recorded.lvl, replay=mr.read_recording(rp))
    assert (o.field_digest, o.tick_digest) == (m["field_digest"],
                                               m["tick_digest"])
    assert o.result.ticks == m["result"]["ticks"]
    arrays = dict(recorded.arrays)
    bad = dict(m, record_format=99)
    arrays["meta"] = np.array(json.dumps(bad))
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="record_format"):
        door.read_record(path)


def _short_ship(tmp_path, limit=20.0):
    import shutil
    dst = tmp_path / "ship_demo"
    shutil.copytree(ROOT / "levels" / "ship_demo", dst)
    lv = level_loader.load("ship_demo", str(tmp_path))
    level_lib.write_match(dst, replace(lv.match, time_limit_s=float(limit)))
    return dst


def test_the_records_are_the_envs(tmp_path):
    """Property: an in-process ``EnvRunner`` on a run with ``policy =
    ["bots_door", "idle"]``, ``[obs] spec = "v1"``, ``[action] spec =
    "v1"``, ``[reward] profile = "fireteam"``, decimation 4, to the match's
    end: at every step the door team's recorded observation, masks, row
    flags and (one step later) five rewards and ``rows_ended`` equal the
    env's team-0 block, array for array. Breaks if the controller observes
    at another moment, assembles its own observation, or computes its own
    reward."""
    _short_ship(tmp_path)
    p = tmp_path / "run.toml"
    p.write_text(
        "[run]\nseed = 5\nn_envs = 1\ndecimation = 4\n"
        "levels = [\"ship_demo\"]\npolicy = [\"bots_door\", \"idle\"]\n"
        f"levels_dir = '{tmp_path.as_posix()}'\nsight = \"all_round\"\n"
        "[obs]\nspec = \"v1\"\nfull_view = false\n"
        "[action]\nspec = \"v1\"\n[reward]\nprofile = \"fireteam\"\n",
        encoding="utf-8")
    run = rl_env.load_run(p)
    levels = rl_env.load_levels(run)
    lay = rl_env.make_layout(run, levels)
    r = rl_env.EnvRunner(0, run, lay, levels, breach_physics=bp)
    sink = door.RecordSink(tmp_path / "rec")
    door.install_sink(sink)
    try:
        r.reset()
    finally:
        door.install_sink(None)
    S = lay.n_slots
    out = {k: np.zeros(s[1:], d) for k, (s, d) in lay.obs_shapes().items()}
    acts = np.zeros((lay.n_rows, 5), np.int64)
    obs, rew, ended = [], [], []
    while True:
        r.observe(out)
        obs.append({k: np.array(v[:S], copy=True) for k, v in out.items()})
        res = r.step(acts)
        rew.append(res["reward"][:S].astype(np.float32))
        ended.append(res["rows_ended"][:S].copy())
        if res["ended"]:
            break
    sim = r.session.sim
    ctl = _door_of(r.session)[0]
    sink.close(sim, r.session.result(), r.session.digests(), seed=r.seed)
    arr, meta = door.read_record(sorted((tmp_path / "rec").rglob("*.npz"))[0])
    assert len(arr["tick"]) == len(obs) > 50
    for i in range(len(obs)):
        for k, v in obs[i].items():
            assert np.array_equal(arr[k][i], v), (i, k)
        assert np.array_equal(arr["reward"][i], rew[i]), i
        assert np.array_equal(arr["rows_ended"][i], ended[i]), i
    assert np.abs(arr["reward"]).sum() > 0
    assert ctl.stats["masked_sent"] == 0


# ---------------------------------------------------------------------------
# The door box and the refusals
# ---------------------------------------------------------------------------
def _door_box(tmp_path):
    """Two rooms split by a wall with a closed door entity in it, a room
    graph, team 0 west and an ``idle`` team 1 east behind a screen wall (no
    line of sight from the west room through the door), no horde."""
    from level_loader import LevelData
    from simulation.match import MatchSpec, MatchTeam, SpawnRoom
    from simulation.materials import MAT_AIR, MAT_HULL
    from test_horde_behaviours import _door_inst
    h, w, wx = 16, 44, 22
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    tm[1:h - 1, wx] = MAT_HULL
    tm[6:9, wx] = MAT_AIR
    tm[1:11, 30] = MAT_HULL         # a screen: no sight from the door east
    rooms = [dict(id=0, x=1, y=1, w=wx - 1, h=h - 2, kind="room"),
             dict(id=1, x=wx + 1, y=1, w=w - wx - 2, h=h - 2, kind="room")]
    openings = [dict(id=0, a=0, b=1, kind="door", x=wx, y=6, w=1, h=3)]
    grid = np.full((h, w), -1, dtype=np.int32)
    grid[1:h - 1, 1:wx] = 0
    grid[1:h - 1, wx + 1:w - 1] = 1
    grid[1:11, 30] = -1
    rg = level_loader.parse_room_graph(rooms, openings, grid, (h, w), "box")
    team = MatchTeam(n=1, loadout="rifleman", env_profile="default",
                     controller="bots_door")
    spec = MatchSpec(teams=(team, replace(team, controller="idle")),
                     seed=1, zombies=0, time_limit_s=60.0,
                     sight="all_round")
    return LevelData(
        name="doorbox", version="3", path=tmp_path / "doorbox",
        tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."),
        entities=[_door_inst("d", 0, wx, 6)], rooms=rg,
        spawn_rooms=[SpawnRoom(team=0, x=4, y=4, w=8, h=8, max_per_team=1),
                     SpawnRoom(team=1, x=34, y=1, w=8, h=8,
                               max_per_team=1)],
        match=spec)


def test_the_door_bot_opens_the_door_and_walks_through(tmp_path):
    """Property: in the door box a door bot's ``use`` latch reaches the game
    on a decision tick, the door opens and the bot walks through, nothing
    masked sent. Breaks if ``use`` is not translated."""
    from simulation.entities.door import DOOR_OPEN
    lvl = _door_box(tmp_path)
    p = _play(lvl, lvl.match, seed=1, max_ticks=1500, digests=False,
              watch=lambda sim: sim.roster(0)[0].tile_x)
    uses = [t for t, k, _a in p.tap.calls if k == "use_intent"]
    assert uses and all(t % 4 == 0 for t in uses)
    assert p.sim._doors[0].state == DOOR_OPEN
    assert max(p.watched) > 22               # its body went through
    assert p.ctl[0].stats["masked_sent"] == 0


def test_the_refusals(tmp_path):
    """Property: ``bots_door`` on a level without a room graph is refused
    by name at ``bind_match``; an unbound door controller refuses to run;
    the variant names registered in ``control_source`` are the
    ``DOOR_VARIANTS`` rows and every one is nameable; ``--records`` inside
    the checkout is refused. Breaks if the door runs where spec v1 cannot
    be built, or a variant is registered in one place only."""
    lvl = _door_box(tmp_path)
    bare = replace(lvl, rooms=None)
    with pytest.raises(ValueError, match="room graph"):
        MatchSession(bare, bare.match, breach_physics=bp)
    c = door.BotsThroughDoor(0)
    with pytest.raises(RuntimeError, match="bind_match"):
        c.handle_frame(MatchSession(lvl, replace(lvl.match, teams=tuple(
            replace(t, controller="idle") for t in lvl.match.teams)),
            breach_physics=bp).sim)
    assert set(control_source.DOOR_BOT_NAMES) == set(door.DOOR_VARIANTS)
    assert set(door.DOOR_VARIANTS) <= set(
        control_source.match_file_controllers())
    import bot_tournament
    with pytest.raises(SystemExit):
        bot_tournament.main(["--a", "bots_door", "--b", "bots", "--seeds",
                             "1", "--records", str(ROOT / "levels")])
