"""#195 1h -- action spec v1, its decoder and its masks
(``simulation/action_spec_v1.py``, ``control_policy.AgentPolicy``'s v1
decoder, ``rl_env``'s ``[action] spec``; the contract
``docs/rl_agent_rung1_note_1h_2026-11-12.md``).

Each test names the property it protects and the change that must break it.
No test pins a count or order of something built to grow: the hand table is
checked AGAINST ``hand_grenade_rows``, never as a list of six.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tools", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import control_policy as cp  # noqa: E402
import rl_env  # noqa: E402
from control_bots import BotTeam  # noqa: E402
from control_source import quantize_stick_direction  # noqa: E402
from run_match import run_match  # noqa: E402
from simulation import action_spec_v1 as A  # noqa: E402
from simulation import obs_spec_v1 as O  # noqa: E402
from simulation import verb_gates  # noqa: E402
from simulation.damage import KINETIC, DamagePacket, apply_packet  # noqa: E402
from simulation.match import MatchSpec  # noqa: E402
from simulation.movement import THROTTLE_ONE  # noqa: E402
from simulation.sensor_accessor import EntityFieldAccessor  # noqa: E402
from simulation.turning import PI_Q16, wrap_q16  # noqa: E402
from simulation.turning import place as place_facing  # noqa: E402
from simulation.unit_fixed import atan2_q16, quantize_scalar  # noqa: E402
from simulation.weapons import PLANT_WEAPON  # noqa: E402
from test_bots_v1 import _level  # noqa: E402
from test_obs_spec_v1 import (IDLE, PAIR_SPEC, _graph_box, _rows,  # noqa: E402
                              _sim)

ONE = 1 << 16
S2 = 2                     # PAIR_SPEC's slots in the hand-built scenes


# ---------------------------------------------------------------------------
# a hand-built scene driven by the v1 decoder (no env): observe -> decide ->
# tick, with the intent tap
# ---------------------------------------------------------------------------
class _Tap:
    """The intent tap's sink: every call, ``(tick, kind, args)``."""

    def __init__(self):
        self.calls = []

    def record(self, tick, kind, args):
        self.calls.append((int(tick), kind, tuple(args)))

    def of(self, uid, kind=None, tick=None):
        return [c for c in self.calls if c[2] and c[2][0] == uid
                and (kind is None or c[1] == kind)
                and (tick is None or c[0] == tick)]


class _Scene:
    """A sim on a hand-built level, each team on a v1 ``AgentPolicy``; the
    observation, the masks and the aims as the env builds them."""

    def __init__(self, sim, spec=PAIR_SPEC, S=S2, decimation=4):
        self.sim, self.spec, self.S, self.d = sim, spec, S, decimation
        self.ctx = O.ObsContext.build(sim, spec, S)
        self.tables = A.action_tables(sim, S)
        self.pols = {}
        for t in range(len(spec.teams)):
            p = cp.AgentPolicy(t)
            p.configure("v1", self.tables)
            self.pols[t] = p
        self.tap = _Tap()
        sim.attach_intent_log(self.tap)
        self.out = None

    def observe(self):
        sim = self.sim
        U = len(self.spec.teams) * self.S
        out = {k: np.zeros(s[1:], d) for k, (s, d)
               in O.shapes(1, U, self.S).items()}
        out.update({k: np.zeros(s[1:], d) for k, (s, d) in A.mask_shapes(
            1, U, self.tables.n_entries, self.tables.n_kinds).items()})
        aims = {}
        for p in self.pols.values():
            aims.update(p.aim_view())
        units = _rows(sim, self.spec, self.S)
        EntityFieldAccessor(sim.gmap).observe("v1", units, out, sim=sim,
                                              context=self.ctx, aims=aims)
        A.build_masks(sim, units, out["list"][:, :, 0], out["list_mask"],
                      out["list_uid"], self.tables, self.d, out)
        self.out = out
        return out

    def decide(self, rows: dict):
        """``rows``: team -> ``(S, 5)``; the other teams decide nothing."""
        out = self.observe()
        for t, r in rows.items():
            S = self.S
            self.pols[t].set_decisions(np.asarray(r),
                                       out["list_uid"][t * S:(t + 1) * S])
        return out

    def tick(self, n=1):
        for _ in range(n):
            for p in self.pols.values():
                p.handle_frame(self.sim)
            self.sim.step()


def _row(feet=A.FEET_STAND, aim=A.AIM_WALK, trig=0, hand=A.HAND_NOTHING,
         arg=0):
    return [feet, aim, trig, hand, arg]


def _rows_for(scene, team, by_slot):
    r = np.array([_row() for _ in range(scene.S)])
    for s, v in by_slot.items():
        r[s] = v
    return {team: r}


def _entry_of(out, i, uid):
    k = np.flatnonzero((out["list_uid"][i] == uid) & (out["list_mask"][i] == 1))
    assert len(k), (uid, out["list_uid"][i])
    return int(k[0])


def _aims(tap, uid):
    """Per tick the ``set_aim`` vector the tap holds for ``uid``."""
    return {t: (a[1], a[2]) for t, k, a in tap.calls
            if k == "set_aim" and a[0] == uid}


# ---------------------------------------------------------------------------
# section 1: the tables
# ---------------------------------------------------------------------------
def test_the_hand_table_is_the_fixed_verbs_then_one_throw_per_row(tmp_path):
    """Property (note 1): the hand table is the fixed verbs, then one throw
    per ``hand_grenade_rows`` row in config order; appending a grenade row
    (monkeypatched tables) appends ONE entry and moves no other index.
    Breaks if the throws are not last, or a kind is hard-coded."""
    sim, _ = _sim(_level(tmp_path, "ht"), [(0, 10, 10)])
    tb = A.action_tables(sim, S2)
    rows = sim.weapons_tables.hand_grenade_rows
    assert tb.hand_names == A.FIXED_HANDS + tuple(f"throw:{r}" for r in rows)
    assert tb.sizes[3] == A.HAND_THROW0 + len(rows)
    for k, r in enumerate(rows):
        assert tb.hand_names[A.HAND_THROW0 + k] == f"throw:{r}"
    real = sim.weapons_tables
    sim.weapons_tables = types.SimpleNamespace(
        weapons=real.weapons, hand_grenade_rows=tuple(rows) + ("grenade_new",))
    try:
        tb2 = A.action_tables(sim, S2)
    finally:
        sim.weapons_tables = real
    assert tb2.hand_names[:len(tb.hand_names)] == tb.hand_names
    assert tb2.hand_names[-1] == "throw:grenade_new"
    assert tb2.sizes[3] == tb.sizes[3] + 1
    assert tb2.sizes[:3] == tb.sizes[:3] and tb2.sizes[4] == tb.sizes[4]


def test_the_compass_and_the_feet_index():
    """Property (note 1): ``COMPASS_Q16[d]`` equals the stick quantizer of
    ``(cos, -sin)(d * 22.5 deg)`` and ``atan2_q16`` of it lies in bin ``d``;
    every feet index decodes to its ``(d, s)`` and back. Breaks if the
    table, the convention (y down) or the index formula drifts."""
    bin_half = PI_Q16 // 16
    for d, (qx, qy) in enumerate(A.COMPASS_Q16):
        a = math.radians(d * 22.5)
        assert (qx, qy) == quantize_stick_direction(math.cos(a),
                                                    -math.sin(a), 0.0)[:2]
        ang = atan2_q16(-qy, qx)
        centre = d * (PI_Q16 // 8)
        assert abs(wrap_q16(ang - centre)) <= bin_half, (d, ang, centre)
    assert A.feet_option(A.FEET_STAND) is None
    seen = set()
    for i in range(1, A.N_FEET):
        d, s = A.feet_option(i)
        assert 0 <= d < A.N_DIRS and 0 <= s < A.N_SPEEDS
        assert 1 + A.N_SPEEDS * d + s == i
        seen.add((d, s))
    assert len(seen) == A.N_FEET - 1


def test_walk_half_full_send_the_throttle_rows_speeds(tmp_path):
    """Property (note 1, the Throttle row): feet speeds walk / half / full
    send throttle 0, ``THROTTLE_ONE // 2``, ``THROTTLE_ONE`` (read from the
    tap), along the compass vector. Breaks if a speed is a literal of
    another scale."""
    sim, (me,) = _sim(_level(tmp_path, "thr"), [(0, 20, 20)])
    sc = _Scene(sim)
    for s, want in enumerate((0, THROTTLE_ONE // 2, THROTTLE_ONE)):
        sc.decide(_rows_for(sc, 0, {0: _row(feet=1 + A.N_SPEEDS * 4 + s)}))
        t = sim.tick
        sc.tick()
        (call,) = sc.tap.of(me.id, "set_move_dir", t)
        assert call[2][1:] == (*A.COMPASS_Q16[4], want)


def test_the_reaches_and_onto_the_target(tmp_path, monkeypatch):
    """Property (note 1): the four reaches are ``(j+1)/4`` of the hand
    grenade row's ``max_throw_range`` (a monkeypatched range moves them);
    "onto target" sends the bots' ``_lob`` distance to the record's centre
    on the issuing tick. Breaks on a tile literal, or the reach read on
    another tick."""
    sim, (me, foe) = _sim(_level(tmp_path, "rch"), [(0, 10, 20), (1, 22, 24)],
                          sight="all_round")
    hand = sim.weapons_tables.weapons.by_name["hand_grenade"]
    tb = A.action_tables(sim, S2)
    assert tb.reaches_q == tuple((j + 1) * (int(hand.max_throw_range) << 16)
                                 // 4 for j in range(4))
    monkeypatch.setattr(hand, "max_throw_range", 12)
    assert A.action_tables(sim, S2).reaches_q == tuple(
        (j + 1) * (12 << 16) // 4 for j in range(4))
    monkeypatch.undo()
    sc = _Scene(sim)
    sim.step()                              # the picture sees the foe
    sim.set_move_dir(foe.id, 0, -ONE, THROTTLE_ONE)   # it walks on
    out = sc.observe()
    k = _entry_of(out, 0, foe.id)
    sc.decide(_rows_for(sc, 0, {0: _row(aim=A.AIM_TOKENS + k,
                                         hand=A.HAND_THROW0, arg=A.ARG_ONTO)}))
    t = sim.tick
    rec = sim.team_picture.records(0)[foe.id]
    want = BotTeam._lob(None, (me.center_tile_x(), me.center_tile_y()),
                        rec.centre)[1]
    sc.tick()
    (call,) = sc.tap.of(me.id, "throw_grenade_intent", t)
    assert call[2][4] == want
    assert call[2][3] == tb.fuse_ticks
    for j in range(4):
        sc.decide(_rows_for(sc, 0, {0: _row(hand=A.HAND_THROW0 + 1, arg=j)}))
        t = sim.tick
        sc.tick()
        (call,) = sc.tap.of(me.id, "throw_grenade_intent", t)
        assert call[2][4] == tb.reaches_q[j] and call[2][5] == 1


def test_the_plant_timers(tmp_path, monkeypatch):
    """Property (note 1): the plant timers are the row's ``lo`` and ``hi``
    at the ends, strictly increasing, each the integer cube root (rounded
    to nearest, exactly) of ``lo^(3-j) hi^j`` (a monkeypatched row moves
    them); the plant the game starts carries the timer. Breaks if seconds
    are hard-coded, or a float ``pow`` rounds differently."""
    from simulation.weapons import fuse_ticks_from_seconds
    sim, (me,) = _sim(_level(tmp_path, "pt"), [(0, 20, 20)])
    tps = sim._tps
    c4 = sim.weapons_tables.weapons.by_name[PLANT_WEAPON]
    lo = fuse_ticks_from_seconds(c4.fuse_min_seconds, tps)
    hi = fuse_ticks_from_seconds(c4.fuse_max_seconds, tps)
    tb = A.action_tables(sim, S2)
    t = tb.plant_timers
    assert t[0] == lo and t[-1] == hi
    assert all(a < b for a, b in zip(t, t[1:]))
    for j, v in enumerate(t):
        n = lo ** (3 - j) * hi ** j
        assert (2 * v - 1) ** 3 <= 8 * n < (2 * v + 1) ** 3, (j, v, n)
    for n in list(range(0, 2000)) + [10 ** 18 + 7, 3 ** 60]:
        r = A.icbrt_round(n)
        assert (2 * r - 1) ** 3 <= 8 * n < (2 * r + 1) ** 3 or n == 0
    monkeypatch.setattr(c4, "fuse_max_seconds", 30.0)
    t2 = A.action_tables(sim, S2).plant_timers
    assert t2[-1] == fuse_ticks_from_seconds(30.0, tps) and t2 != t
    monkeypatch.undo()
    sc = _Scene(sim)
    sc.decide(_rows_for(sc, 0, {0: _row(hand=A.HAND_PLANT, arg=2)}))
    sc.tick()
    assert me.plant_done_tick >= 0 and int(me.plant_fuse_ticks) == t[2]


# ---------------------------------------------------------------------------
# section 2: the decoder
# ---------------------------------------------------------------------------
def test_the_decoder_holds_a_uid_across_a_resort(tmp_path):
    """Property (note 2, lens 4 minor 4): across a decision in which the
    list re-sorts (two enemies swap places), the issued ``set_aim`` keeps
    pointing at the SAME unit's record, every tick. Breaks if the decoder
    keeps a list position."""
    sim, (me, e1, e2) = _sim(_level(tmp_path, "rs"),
                             [(0, 10, 20), (1, 16, 16), (1, 24, 24)],
                             sight="all_round")
    sc = _Scene(sim)
    sim.step()
    out = sc.observe()
    k1 = _entry_of(out, 0, e1.id)
    assert k1 < _entry_of(out, 0, e2.id)        # e1 the nearer: first
    sc.decide(_rows_for(sc, 0, {0: _row(aim=A.AIM_TOKENS + k1)}))
    sim.set_move_dir(e1.id, ONE, 0, THROTTLE_ONE)
    sim.set_move_dir(e2.id, -ONE, 0, THROTTLE_ONE)
    differs = 0
    _record_centres(sim)
    for _ in range(60):
        t = sim.tick
        sc.tick()
        # the aim issued at t was resolved on the records as they stood at t
        got = _aims(sc.tap, me.id)[t]
        assert got == A.aim_vector(me, _CENTRES[t][e1.id]), t
        differs += got != A.aim_vector(me, _CENTRES[t][e2.id])
        _record_centres(sim)
    out = sc.observe()
    assert _entry_of(out, 0, e1.id) > _entry_of(out, 0, e2.id)   # re-sorted
    assert differs > 0


_CENTRES: dict = {}


@pytest.fixture(autouse=True)
def _centres_reset():
    _CENTRES.clear()
    yield
    _CENTRES.clear()


def _record_centres(sim):
    _CENTRES[sim.tick] = {u: r.centre for u, r in
                          sim.team_picture.records(0).items()}


def test_the_aim_follows_the_record_then_keeps_the_last_point(tmp_path):
    """Property (note 2): against an enemy walking out of view behind a
    wall, each tick's ``set_aim`` (the tap) equals ``aim_vector`` to THAT
    tick's record centre; after the record expires it keeps the last point.
    Breaks if the aim is computed once per decision, or reads the live
    unit."""
    sim, (me, foe) = _sim(_level(tmp_path, "fol", wall_x=30, gap=(19, 24)),
                          [(0, 10, 20), (1, 34, 20)], sight="all_round",
                          memory_ticks=12)
    sc = _Scene(sim)
    sim.step()
    _record_centres(sim)
    out = sc.observe()
    k = _entry_of(out, 0, foe.id)
    sc.decide(_rows_for(sc, 0, {0: _row(aim=A.AIM_TOKENS + k)}))
    sim.set_move_dir(foe.id, 0, -ONE, THROTTLE_ONE)       # north, behind
    last, phases = None, set()
    for _ in range(80):
        t = sim.tick
        sc.tick()
        cent = _CENTRES[t].get(foe.id)
        if cent is not None:
            last = cent
            phases.add("record")
        else:
            phases.add("expired")
        assert _aims(sc.tap, me.id)[t] == A.aim_vector(me, last), t
        _record_centres(sim)
    assert phases == {"record", "expired"}
    assert (foe.center_tile_x(), foe.center_tile_y()) != last


def _hide_pair(tmp_path, name):
    """Two identical sims: me facing a foe through a gap; the foe walks
    north behind the wall in both."""
    lvl = _level(tmp_path, name, wall_x=30, gap=(19, 24))
    out = []
    for _ in range(2):
        sim, units = _sim(lvl, [(0, 10, 20), (1, 34, 20)], sight="all_round",
                          memory_ticks=400)
        out.append((sim, units, _Scene(sim)))
    return out


def _fill_bytes(out):
    """Team 0's rows of what 1h fills (vector cols 71-76, list cols 28 and
    66, ``sees_target``) and of the list's ids, as bytes. The rest of the
    row is 1g's family's (``test_obs_spec_v1``): a body walking in the air
    moves the air, so over a long scene the soldier's own impulse columns
    (``effort``) and the physics planes carry the hidden walker's imprint
    -- 1g's ACCEPTED GAP (Erik, 2026-11-12), not a decoder leak."""
    rows = slice(0, S2)
    return (out["vector"][rows, 71:77].tobytes(),
            out["list"][rows][:, :, [28, 66]].tobytes(),
            out["near_u"][rows, 3].tobytes(),
            out["list_uid"][rows].tobytes(), out["list_mask"][rows].tobytes())


def test_the_decoder_never_reads_a_hidden_unit(tmp_path):
    """Property (note 2, 1g's counterfactual (b) with the decoder on): two
    sims differing only by an unseen move of a remembered enemy issue the
    same intents for team 0 tick for tick, and the columns the decoder
    fills (and the list ids) stay byte-equal for team 0. Breaks if the
    decoder reads ``sim.units`` / a live position, or the fill leaks
    one."""
    pair = _hide_pair(tmp_path, "hid")
    for sim, (me, foe), sc in pair:
        sim.step()
        sim.set_move_dir(foe.id, 0, -ONE, THROTTLE_ONE)
    from simulation import vision
    for _ in range(80):                      # out of sight in both
        for sim, _, sc in pair:
            sim.step()
        if all(u[1].id not in vision.state_for(sim).visible_by_team.get(0, ())
               for sim, u, _ in pair):
            break
    for sim, (me, foe), sc in pair:
        assert foe.id not in vision.state_for(sim).visible_by_team.get(0, ())
        assert foe.id in sim.team_picture.records(0)
    pair[1][0].set_move_dir(pair[1][1][1].id, ONE, 0, THROTTLE_ONE)  # unseen
    for d in range(8):
        for sim, (me, foe), sc in pair:
            out = sc.observe()
            k = _entry_of(out, 0, foe.id)
            sc.decide(_rows_for(sc, 0, {0: _row(aim=A.AIM_TOKENS + k,
                                                 trig=1)}))
            sc.tick(4)
        a, b = (sc.tap.of(u[0].id) for _, u, sc in pair)
        assert a == b, d
        oa, ob = (sc.observe() for _, _, sc in pair)
        assert _fill_bytes(oa) == _fill_bytes(ob), d
    assert pair[0][1][1].x != pair[1][1][1].x


def test_hold_and_latch(tmp_path):
    """Property (note 2, lens 3 m2): over a decision, move / aim / trigger
    are issued on EVERY tick and the hand verb exactly ONCE, on the first
    tick (the tap). Breaks if the decoder re-sets rows every tick or
    bypasses ``VERB_RULES``."""
    sim, (me,) = _sim(_level(tmp_path, "hl"), [(0, 20, 20)])
    sc = _Scene(sim)
    sc.decide(_rows_for(sc, 0, {0: _row(feet=5, aim=3, trig=1,
                                         hand=A.HAND_RELOAD)}))
    t0 = sim.tick
    sc.tick(4)
    for kind in ("set_move_dir", "set_aim", "set_trigger"):
        assert sorted(c[0] for c in sc.tap.of(me.id, kind)) == [
            t0, t0 + 1, t0 + 2, t0 + 3], kind
    assert [c[0] for c in sc.tap.of(me.id, "reload_intent")] == [t0]


def test_the_throws_aim_is_held_through_the_wind_up(tmp_path):
    """Property (note 2, design 2.1, lens 3 M2): a throw at decision k
    toward entry X, decision k+1 choosing a compass token (north): every
    ``set_aim`` through the release tick aims at X, the grenade leaves
    along the facing that aim produced (east, not north), and the token
    applies from the tick after the release; ``mask_aim`` at k+1 is "keep"
    alone exactly when the release is at or after the decision's last tick.
    Breaks if the next decision's aim steers the grenade."""
    sim, (me, foe) = _sim(_level(tmp_path, "ho"), [(0, 10, 20), (1, 26, 20)],
                          sight="all_round")
    place_facing(me, 0)
    sc = _Scene(sim, decimation=6)
    sim.step()
    out = sc.observe()
    k = _entry_of(out, 0, foe.id)
    sc.decide(_rows_for(sc, 0, {0: _row(aim=A.AIM_TOKENS + k,
                                         hand=A.HAND_THROW0, arg=1)}))
    t0 = sim.tick
    sc.tick()
    rel = int(me.throw_release_tick)
    assert rel > t0 + 1
    sc.tick(5)                                         # decision k: 6 ticks
    # decision k+1 at T = t0 + 6: keep-alone iff rel >= T + d - 1
    T = sim.tick
    for d in (2, 6):
        o = sc.observe()
        A.build_masks(sim, _rows(sim, PAIR_SPEC, S2), o["list"][:, :, 0],
                      o["list_mask"], o["list_uid"], sc.tables, d, o)
        keep_alone = (o["mask_aim"][0].sum() == 1
                      and o["mask_aim"][0][A.AIM_KEEP] == 1)
        assert keep_alone == (rel >= T + d - 1), (d, rel, T)
    north = 4
    sc.decide(_rows_for(sc, 0, {0: _row(aim=north)}))
    n_before = len(sim.projectiles)
    sc.tick(6)
    aims = _aims(sc.tap, me.id)
    rec = sim.team_picture.records(0)[foe.id]
    for t in range(t0, rel + 1):
        assert aims[t] == A.aim_vector(me, rec.centre), t
    for t in range(rel + 1, sim.tick):
        assert aims[t] == A.COMPASS_Q16[north], t
    g = sim.projectiles[-1]
    assert len(sim.projectiles) > n_before or g is not None
    assert g.target_fx - g.start_fx > 0
    assert abs(g.target_fy - g.start_fy) < (g.target_fx - g.start_fx) / 4


def test_a_refused_throw_holds_nothing(tmp_path):
    """Property (note 2): a throw forced with 0 of its kind issues on every
    tick the same aim as the "nothing" row (the next decision's token is
    obeyed at once). Breaks if the hold keys on "I sent a throw" instead of
    the game's ``throw_release_tick``."""
    seqs = []
    for hand in (A.HAND_THROW0, A.HAND_NOTHING):
        sim, (me, foe) = _sim(_level(tmp_path, f"rf{hand}"),
                              [(0, 10, 20), (1, 26, 20)], sight="all_round")
        me.grenade_counts[0] = 0
        sc = _Scene(sim)
        sim.step()
        out = sc.observe()
        k = _entry_of(out, 0, foe.id)
        sc.decide(_rows_for(sc, 0, {0: _row(aim=A.AIM_TOKENS + k, hand=hand)}))
        sc.tick(4)
        assert me.throw_release_tick < 0
        sc.decide(_rows_for(sc, 0, {0: _row(aim=4)}))
        sc.tick(4)
        seqs.append(sorted(_aims(sc.tap, me.id).items()))
    assert seqs[0] == seqs[1]


def test_the_fallbacks(tmp_path):
    """Property (note 2): a masked aim (a mate, an empty slot, "keep" with
    no wind-up) decodes as "where I walk" (``set_aim(0, 0)``); a masked
    argument (a throw "onto target" with a token aim, a plant with arg 4)
    and the reserved flashlight decode as hand "nothing" (the tap holds no
    hand call; the flashlight stays held ON). Breaks if a masked index is
    decoded as something else."""
    sim, (me, mate, foe) = _sim(_level(tmp_path, "fb"),
                                [(0, 10, 20), (0, 14, 26), (1, 30, 20)],
                                sight="all_round")
    sc = _Scene(sim)
    sim.step()
    out = sc.observe()
    km = _entry_of(out, 0, mate.id)
    empty = int(np.flatnonzero(out["list_mask"][0] == 0)[0])
    for aim in (A.AIM_TOKENS + km, A.AIM_TOKENS + empty, A.AIM_KEEP):
        assert out["mask_aim"][0][aim] == 0
        sc.decide(_rows_for(sc, 0, {0: _row(aim=3)}))
        sc.tick()                           # aim somewhere first
        sc.decide(_rows_for(sc, 0, {0: _row(aim=aim)}))
        t = sim.tick
        sc.tick()
        assert _aims(sc.tap, me.id)[t] == (0, 0), aim
    hand_kinds = ("throw_grenade_intent", "plant_charge_intent",
                  "reload_intent", "melee_intent", "shove_intent",
                  "use_intent")
    for row in (_row(aim=2, hand=A.HAND_THROW0, arg=A.ARG_ONTO),
                _row(hand=A.HAND_PLANT, arg=A.ARG_ONTO),
                _row(hand=A.HAND_FLASHLIGHT)):
        sc.decide(_rows_for(sc, 0, {0: row}))
        t = sim.tick
        sc.tick()
        assert not [c for c in sc.tap.of(me.id, tick=t) if c[1] in hand_kinds]
        (fl,) = sc.tap.of(me.id, "set_flashlight", t)
        assert fl[2][1] is False
    assert me.flashlight_on


def test_an_invalid_row_issues_nothing(tmp_path, monkeypatch):
    """Property (note 3, 1e's note): an invalid row (dead, turned, an empty
    roster place) has exactly ONE legal option per head (the defaults), and
    NOTHING is issued for its unit whatever is sent (the tap holds no call
    with its id after it left; its ``pending_*`` stay at default). Breaks
    if a dead row issues a latch."""
    sim, (me, mate, foe) = _sim(_level(tmp_path, "inv"),
                                [(0, 10, 20), (0, 14, 26), (1, 30, 20)],
                                sight="all_round")
    sc = _Scene(sim)
    sim.step()
    apply_packet(me, DamagePacket(10_000, KINETIC, -1), sim.tick_events,
                 source="burning")
    apply_packet(mate, DamagePacket(10_000, KINETIC, -1), sim.tick_events,
                 source="melee", mark_killed_by_zombie=True)
    sim.step()
    assert not me.alive and mate.is_zombie
    out = sc.observe()
    U = len(PAIR_SPEC.teams) * S2
    for i in (0, 1, U - 1):                     # dead, turned, empty place
        for key, default in zip(A.MASK_KEYS, (A.FEET_STAND, A.AIM_WALK, 0,
                                              A.HAND_NOTHING)):
            m = out[key][i]
            assert m.sum() == 1 and m[default] == 1, (i, key)
    n_calls = len(sc.tap.calls)
    for hand in range(1, sc.tables.sizes[3]):
        sc.decide(_rows_for(sc, 0, {0: _row(feet=7, aim=5, trig=1, hand=hand,
                                             arg=1),
                                     1: _row(feet=7, aim=5, trig=1, hand=hand,
                                             arg=1)}))
        sc.tick()
    assert not [c for c in sc.tap.calls[n_calls:]
                if c[2] and c[2][0] in (me.id, mate.id)]
    for u in (me, mate):
        for name in ("pending_throw", "pending_reload", "pending_shove",
                     "pending_melee", "pending_plant", "pending_use"):
            assert getattr(u, name) in (None, False), (u.name, name)


# ---------------------------------------------------------------------------
# section 3: the masks
# ---------------------------------------------------------------------------
def _farm_runner(tmp_path, n=1, decimation=4, level="ship_demo",
                 levels_dir=None, record=False):
    import _1h_forced as F
    p = tmp_path / "run.toml"
    p.write_text(F.run_text(1, level, decimation, levels_dir),
                 encoding="utf-8")
    run = rl_env.load_run(p)
    levels = rl_env.load_levels(run)
    layout = rl_env.make_layout(run, levels)
    r = rl_env.EnvRunner(0, run, layout, levels, breach_physics=bp,
                         record=record, checkpoint_every_ticks=24)
    out = {k: np.zeros(s[1:], d) for k, (s, d) in layout.obs_shapes().items()}
    r.reset()
    r.observe(out)
    return r, layout, out


def _random_legal(rng, out, n_rows):
    import _1h_forced as F
    return np.array([F._draw(rng, out, i, 0.5) for i in range(n_rows)],
                    dtype=np.int32)


def test_the_mask_is_the_gate(tmp_path, monkeypatch):
    """Property (note 3, design 2.4 / 12.3 item 7): over a played match, for
    every living row and every hand entry the mask bit equals the 1e gate
    called at the same moment; with a gate monkeypatched to answer False
    the bit follows. Also: no head is ever all-masked, and the realised
    argument mask always has a legal option. Breaks if the mask copies a
    gate, or a gate state leaves a head with nothing legal (a NaN
    softmax)."""
    r, layout, out = _farm_runner(tmp_path)
    rng = np.random.default_rng(5)
    tb = r.act_tables
    checked = 0
    for _ in range(60):
        sim = r.session.sim
        T = sim.tick
        for i, u in enumerate(r.row_units()):
            for k in A.MASK_KEYS:
                assert out[k][i].any(), (k, i)
            if u is None:
                continue
            hm = out["mask_hand"][i]
            want = [1, verb_gates.can_reload(sim, u),
                    verb_gates.can_melee(u, T, sim.weapons_tables,
                                         at_tick_start=True),
                    verb_gates.can_shove(sim, u), verb_gates.can_use(sim, u),
                    verb_gates.can_plant(sim, u), 0] + [
                verb_gates.can_throw(sim, u, k) for k in range(tb.n_kinds)]
            assert list(hm) == [int(bool(w)) for w in want], (T, i)
            checked += 1
        acts = _random_legal(rng, out, layout.n_rows)
        for i in range(layout.n_rows):
            assert A.arg_mask(acts[i, 3], acts[i, 1]).any()
        if r.step(acts)["ended"]:
            r.reset()
        r.observe(out)
    assert checked > 100
    for name, idx in (("can_shove", A.HAND_SHOVE), ("can_reload",
                                                    A.HAND_RELOAD)):
        monkeypatch.setattr(verb_gates, name, lambda *a, **k: False)
        r.observe(out)
        assert not out["mask_hand"][:, idx].any(), name
        monkeypatch.undo()


def test_the_aim_mask_reads_the_picture_not_liveness(tmp_path, monkeypatch):
    """Property (note 3, design 2.4): an entry is legal iff it is a type-1
    or type-2 entry with ``list_mask`` 1, and a record of an enemy that
    died UNSEEN stays legal. Breaks if the mask reads liveness."""
    sim, (me, foe) = _sim(_level(tmp_path, "am", wall_x=30, gap=(19, 24)),
                          [(0, 10, 20), (1, 34, 20)], sight="all_round",
                          memory_ticks=60)
    sc = _Scene(sim)
    sim.step()
    sim.set_move_dir(foe.id, 0, -ONE, THROTTLE_ONE)
    for _ in range(14):
        sim.step()
    apply_packet(foe, DamagePacket(10_000, KINETIC, -1), sim.tick_events,
                 source="burning")
    sim.step()
    assert not foe.alive and foe.id in sim.team_picture.records(0)
    out = sc.observe()
    for i in range(len(PAIR_SPEC.teams) * S2):
        if out["mask_aim"][i].sum() == 0:
            continue
        lt = out["list"][i, :, 0].astype(int)
        want = (out["list_mask"][i] == 1) & np.isin(lt, (1, 2))
        assert np.array_equal(out["mask_aim"][i][A.AIM_TOKENS:] == 1, want)
    k = _entry_of(out, 0, foe.id)
    assert out["mask_aim"][0][A.AIM_TOKENS + k] == 1


def test_the_argument_mask_and_the_heads_used():
    """Property (note 3, design 8.2, lens 4 minor 2): ``arg_mask(hand,
    aim)``: "onto target" legal iff a throw and ``aim >= 18``; a plant
    0..3; any other verb 0 alone; ``heads_used`` is "more than one legal"
    and is False for exactly 8.2's three cases (the argument of a verb that
    takes none, the aim while a throw holds it, every head of a row that
    cannot act). Breaks if the argument's legality ignores the drawn aim
    or verb."""
    n_hand = A.HAND_THROW0 + 3
    for hand in range(n_hand):
        for aim in (0, A.AIM_WALK, A.AIM_KEEP, A.AIM_TOKENS,
                    A.AIM_TOKENS + 5):
            m = A.arg_mask(hand, aim)
            if hand >= A.HAND_THROW0:
                assert list(m[:4]) == [1] * 4
                assert m[A.ARG_ONTO] == (aim >= A.AIM_TOKENS)
            elif hand == A.HAND_PLANT:
                assert list(m) == [1, 1, 1, 1, 0]
            else:
                assert list(m) == [1, 0, 0, 0, 0]
    hands = np.arange(n_hand)[:, None].repeat(4, axis=1)
    aims = np.array([0, 16, 17, 20])[None, :].repeat(n_hand, axis=0)
    assert A.arg_mask(hands, aims).shape == (n_hand, 4, A.N_ARG)
    # 8.2's three cases are heads with ONE legal option
    U = 3
    masks = {k: np.zeros((U, n), np.uint8) for k, n in zip(
        A.MASK_KEYS, (A.N_FEET, 30, 2, n_hand))}
    out = {k: v[0:3] for k, v in masks.items()}
    A.default_masks(out, 0)                    # a row that cannot act
    for i in (1, 2):
        out["mask_feet"][i] = 1
        out["mask_trigger"][i] = 1
        out["mask_hand"][i, :3] = 1
        out["mask_aim"][i, :A.AIM_KEEP] = 1
    out["mask_aim"][2] = 0                     # the aim held by a wind-up
    out["mask_aim"][2, A.AIM_KEEP] = 1
    argm = A.arg_mask(np.array([0, A.HAND_RELOAD, A.HAND_THROW0]),
                      np.array([A.AIM_WALK, 3, A.AIM_KEEP]))
    used = A.heads_used(masks, argm)
    assert not any(used[h][0] for h in A.HEADS)
    assert used["feet"][1] and used["aim"][1] and not used["arg"][1]
    assert not used["aim"][2] and used["arg"][2]


# ---------------------------------------------------------------------------
# section 4: the observation fill
# ---------------------------------------------------------------------------
def test_the_observation_fill(tmp_path):
    """Property (note 4): col 66 marks exactly the entry whose uid the body
    aimed at on the last tick (none for a token); col 28 equals the bots'
    ``_on_target`` for it; ``sees_target`` equals ``has_los`` from each
    window tile to the entry's centre; cols 71..76 equal the decoded
    previous choices, -1 before the first decision; with no decoder view
    (the "raw" action spec) all of them read the none values. Breaks if a
    column is filled from a list position, or from the raw rows."""
    sim, (me, foe) = _sim(_level(tmp_path, "of", wall_x=30, gap=(19, 24)),
                          [(0, 10, 20), (1, 22, 22)], sight="all_round")
    sc = _Scene(sim)
    sim.step()
    out = sc.observe()
    assert (out["vector"][0, 71:77] == -1).all()
    assert not out["list"][0, :, 66].any() and not out["near_u"][0, 3].any()
    k = _entry_of(out, 0, foe.id)
    sc.decide(_rows_for(sc, 0, {0: _row(feet=1 + 3 * 2 + 1,
                                         aim=A.AIM_TOKENS + k, trig=1,
                                         hand=A.HAND_THROW0 + 1, arg=2)}))
    sc.tick(4)
    out = sc.observe()
    assert tuple(out["vector"][0, 71:77]) == (2, 1, A.AIM_TOKENS, 1,
                                              A.HAND_THROW0 + 1, 2)
    k = _entry_of(out, 0, foe.id)
    assert np.flatnonzero(out["list"][0, :, 66]).tolist() == [k]
    rec = sim.team_picture.records(0)[foe.id]
    v = A.aim_vector(me, rec.centre)
    after = sim.facing_after_turn_q16(me, v[0], v[1])
    spread = quantize_scalar(math.radians(float(
        sim.weapons_tables.weapons.by_name[me.weapon_id].spread_deg)))
    on = abs(wrap_q16(atan2_q16(-v[1], v[0]) - after)) <= spread
    assert out["list"][0, k, 28] == float(on)
    cx, cy = me.center_tile_x(), me.center_tile_y()
    y0, x0 = cy - O.HALF, cx - O.HALF
    tx, ty = rec.centre
    plane = out["near_u"][0, 3]
    for j in range(O.WIN):
        for i in range(O.WIN):
            y, x = y0 + j, x0 + i
            h, w = sim.gmap.solid.shape
            want = sim.gmap.has_los(y, x, ty, tx) if (
                0 <= y < h and 0 <= x < w) else None
            if want is not None:
                assert plane[j, i] == want, (y, x)
    assert plane.any() and not plane.all()
    # a token aim (after the throw's held aim let go): no entry marked, no
    # plane; prev records the token
    sc.decide(_rows_for(sc, 0, {0: _row(aim=7)}))
    sc.tick(8)
    out = sc.observe()
    assert not out["list"][0, :, 66].any() and not out["near_u"][0, 3].any()
    assert tuple(out["vector"][0, 71:77]) == (16, -1, 7, 0, 0, -1)
    # the raw spec: no decoder view, the none values
    raw = {k2: np.zeros(s[1:], d) for k2, (s, d)
           in O.shapes(1, 4, S2).items()}
    EntityFieldAccessor(sim.gmap).observe("v1", _rows(sim, PAIR_SPEC, S2),
                                          raw, sim=sim, context=sc.ctx)
    assert (raw["vector"][0, 71:77] == -1).all()
    assert not raw["list"][0, :, 66].any() and not raw["list"][0, :, 28].any()
    assert not raw["near_u"][0, 3].any()


# ---------------------------------------------------------------------------
# the run file door, the farm, the inverse helpers, the replay
# ---------------------------------------------------------------------------
def _run_file(path, levels_dir, *, obs="v1", action='"v1"', extra=""):
    path.write_text(
        "[run]\nseed = 3\nn_envs = 2\ndecimation = 4\nlevels = [\"abox\"]\n"
        f"policy = \"policy\"\nlevels_dir = '{Path(levels_dir).as_posix()}'\n"
        f"[obs]\nspec = \"{obs}\"\n"
        + ("" if obs == "v1" else
           "channels = [\"smoke\"]\ncrop_radius = 2\nvector = [\"hp\"]\n")
        + (f"[action]\nspec = {action}\n{extra}" if action else "")
        + "[reward]\nkills = 1.0\n", encoding="utf-8")
    return path


def test_the_run_file_door_and_the_farm(tmp_path):
    """Property (note 1, 3): ``[action] spec`` unknown is refused by name;
    ``"v1"`` with ``[obs] spec = "crop"`` refused; under ``"v1"`` the
    actions are ``(N, U, 5)``, the masks are ``obs`` keys with ``final_``
    twins (and ``team_rows`` hands them out), an out-of-range index is
    refused naming env, row and head, and ``info["masked_choices"]`` counts
    a forced masked option; a run without the table is ``"raw"``. Breaks if
    the door accepts what the decoder cannot read."""
    spec = MatchSpec(teams=(IDLE, IDLE), seed=1, zombies=0, time_limit_s=3.0)
    _graph_box(tmp_path, "abox", spec)
    with pytest.raises(ValueError, match=r"\[action\] spec must be one of"):
        rl_env.load_run(_run_file(tmp_path / "a.toml", tmp_path,
                                  action='"v2"'))
    with pytest.raises(ValueError, match="needs \\[obs\\] spec 'v1'"):
        rl_env.load_run(_run_file(tmp_path / "b.toml", tmp_path, obs="crop"))
    with pytest.raises(ValueError, match="unknown key"):
        rl_env.load_run(_run_file(tmp_path / "c.toml", tmp_path,
                                  extra="heads = 5\n"))
    assert rl_env.load_run(_run_file(tmp_path / "d.toml", tmp_path,
                                     action=None)).action_spec == "raw"
    run = rl_env.load_run(_run_file(tmp_path / "e.toml", tmp_path))
    with rl_env.BatchedEnv(run, n_workers=1) as env:
        N, U = env.n_envs, env.layout.n_rows
        sizes = env.layout.head_sizes
        assert env._a["actions"].shape == (N, U, 5)
        obs, _ = env.reset()
        for k in A.MASK_KEYS:
            assert obs[k].shape[:2] == (N, U)
        acts = np.zeros((N, U, 5), np.int32)
        acts[..., 1] = A.AIM_WALK
        bad = acts.copy()
        bad[1, 2, 3] = sizes[3]
        with pytest.raises(ValueError, match=r"env 1, row 2, head 'hand'"):
            env.step(bad)
        forced = acts.copy()
        forced[0, 0, 3] = A.HAND_FLASHLIGHT        # reserved: always masked
        forced[0, 0, 1] = A.AIM_KEEP               # masked outside a hold
        obs, _, _, _, info = env.step(forced)
        mc = info["masked_choices"]
        assert mc.shape == (N, U) and mc.dtype == np.uint8
        assert mc[0, 0] == 2 and mc.sum() == 2 + mc[:, 2:].sum()
        assert set(info["final_obs"]) == set(obs)
        rows = env.team_rows(obs, 1)
        assert set(A.MASK_KEYS) <= set(rows)
        assert np.shares_memory(rows["mask_hand"], obs["mask_hand"])


def test_the_inverse_helpers_round_trip(tmp_path):
    """Property (note 8): the inverse helpers round-trip with the decoder's
    tables -- ``feet_index`` of a decoded feet option is that index, the
    nearest compass token of a compass vector is its index, a zero vector
    is "where I walk", the entry of a unit is found by uid, a throw's reach
    and a plant's timer map back to their index. Breaks if 1j would need a
    second table."""
    sim, _ = _sim(_level(tmp_path, "inv8"), [(0, 10, 10)])
    tb = A.action_tables(sim, S2)
    for i in range(A.N_FEET):
        opt = A.feet_option(i)
        if opt is None:
            assert A.feet_index(None, None, 0, tb) == i
            assert A.feet_index(0, 0, ONE, tb) == i
        else:
            dx, dy = A.COMPASS_Q16[opt[0]]
            assert A.feet_index(dx, dy, tb.throttles[opt[1]], tb) == i
    for d, (dx, dy) in enumerate(A.COMPASS_Q16):
        assert A.aim_index_for_vector(dx, dy) == d
    assert A.aim_index_for_vector(0, 0) == A.AIM_WALK
    uids = np.array([5, -1, 7, 9, -1])
    types = np.array([0, 0, 1, 2, 3])
    assert A.aim_index_for_unit(7, uids, types) == A.AIM_TOKENS + 2
    assert A.aim_index_for_unit(9, uids, types) == A.AIM_TOKENS + 3
    assert A.aim_index_for_unit(5, uids, types) is None       # a mate
    for k in range(tb.n_kinds):
        for j, r in enumerate(tb.reaches_q):
            assert A.throw_choice(k, r, False, tb) == (A.HAND_THROW0 + k, j)
        assert A.throw_choice(k, 0, True, tb) == (A.HAND_THROW0 + k,
                                                  A.ARG_ONTO)
    for j, t in enumerate(tb.plant_timers):
        assert A.plant_choice(t, tb) == (A.HAND_PLANT, j)


def test_a_recorded_v1_match_replays(tmp_path):
    """Property (note 6): a v1 policy match recorded through the tap
    (``EnvRunner(record=True)``, random legal rows) replays with every
    checkpoint equal (``run_match(replay=)``); the format is unchanged.
    Breaks if the decoder issues outside the tap, or the format moved."""
    import match_replay
    spec = MatchSpec(teams=(IDLE, IDLE), seed=1, zombies=2, time_limit_s=6.0)
    lv = _graph_box(tmp_path, "rbox", spec)
    r, layout, out = _farm_runner(tmp_path, level="rbox",
                                  levels_dir=tmp_path, record=True)
    rng = np.random.default_rng(2)
    for _ in range(400):
        res = r.step(_random_legal(rng, out, layout.n_rows))
        if res["ended"]:
            break
        r.observe(out)
    assert res["ended"]
    rec = r.last_recording
    # v9 since 183-no-fire-windup (a game rule, not the v1 spec's doing)
    assert match_replay.FORMAT_VERSION == 9
    kinds = {k for v in rec.intents.values() for k, _ in v}
    assert "set_aim" in kinds and len(rec.checkpoints) > 3
    o = run_match(lv, replay=rec)
    assert o.result.ticks == max(rec.checkpoints)


# ---------------------------------------------------------------------------
# section 5: the gate
# ---------------------------------------------------------------------------
def test_forced_masked_choices_do_nothing():
    """Property (note 5): for every masked option of every head on a living
    row, sending it leaves ``field_digest`` and ``tick_digest`` where the
    option's declared fallback leaves them, tick for tick, and anything
    sent on an invalid row leaves them where the default row does
    (``tests/_1h_forced.py``, here one ship_demo seed and a short zombie
    box, every masked hand entry forced at least once). Breaks if the mask
    and the verb read their gate at different moments, an effect lies
    outside a gate, a fallback differs from the decoder's, or the dead-row
    skip goes."""
    import _1h_forced as F
    got = F.main((("ship_demo", 1, 120), ("zbox", 1, 50)), floor=1,
                 verbose=False,
                 require=(("throw", "kind0"), ("reload", "never_bound"),
                          ("use", "no_door"), ("flashlight", "reserved"),
                          ("invalid", "dead")))
    assert got["decisions"] == 170


def test_a_blow_on_the_sweep_tick_is_offered_exactly_when_it_lands(tmp_path):
    """Property (note 3 / 5, the MOMENT): a unit whose stun the coming sweep
    releases, an enemy in its arc: the melee bit built at the start of the
    tick says what the blow issued then does -- 1, and the blow lands; with
    the stun lasting one more tick, 0, and the forced blow does nothing.
    Breaks if the mask reads the gate at the verb's moment
    (``at_tick_start=False``: masked, yet it lands) -- the deliberate break
    (i) of the note's section 5, which random play reaches too rarely for
    the driver to see."""
    from simulation.status import STUNNED, apply_status
    for extra, lands in ((0, True), (1, False)):
        sim, (me, foe) = _sim(_level(tmp_path, f"sw{extra}"),
                              [(0, 10, 20), (1, 13, 20)])
        place_facing(me, 0)
        assert me.melee_id
        st = apply_status(me, STUNNED, 0, 1 + extra)
        sc = _Scene(sim)
        sim.step()                       # the sweep counts it to `extra`
        assert st.remaining_ticks == extra
        out = sc.observe()
        assert out["mask_hand"][0][A.HAND_MELEE] == int(lands)
        hp = foe.current_hp
        sc.decide(_rows_for(sc, 0, {0: _row(hand=A.HAND_MELEE)}))
        sc.tick()
        assert (foe.current_hp < hp) == lands, (extra, hp, foe.current_hp)
