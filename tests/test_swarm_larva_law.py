"""The larva law's properties (arc #63 P3, docs/swarm_P3_impl.md §11
T-P3-1..14, v4.7 G-d), on the REAL C++ CPU entry
(``bp.swarm_species_step("larva", ...)``) over synthetic numpy arrays (test
inputs, never ``gmap.temperature`` / ``gmap.solid``) and explicit table
values (``_swarm_testkit.DEFAULT_LARVA``), never the config. The RAW
reader's oracle is ``T[y >> 16, x >> 16]``.

Every scene that runs the CPU twin declares the law branches it must take
(``bp.swarm_branch_tally``) and asserts each count > 0 (G-d).

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_swarm_larva_law.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

import _swarm_testkit as tk  # noqa: E402
from simulation import swarm, swarm_fixed  # noqa: E402
from simulation.draws import MatchDraws  # noqa: E402
from simulation.rng_salts import Salt  # noqa: E402

ONE = tk.FP_ONE
PI = tk.PI_Q16
SEED = 0x1234_5678_9ABC   # the worlds' key = (seed lo, seed hi)
KEY = (SEED & 0xFFFFFFFF, SEED >> 32)


def _branches_taken(*names):
    tally = bp.swarm_branch_tally()
    missing = [n for n in names if tally[n] <= 0]
    assert not missing, f"scene never took the law branches {missing} (tally {tally})"


@pytest.fixture(autouse=True)
def _reset_tally():
    bp.swarm_branch_tally(reset=True)
    yield


def _world(h, w, max_units, n_env=1, **kw):
    return tk.larva_world(h, w, max_units, n_env, seed=KEY, **kw)


def _tick(world, t):
    world.env[:, swarm.ENV_TICK] = t & 0xFFFFFFFF
    return tk.step_cpu(world)


def _tiles(world, e=0):
    v = world.valid[e] == 1
    return (world.pos_x[e][v] >> 16), (world.pos_y[e][v] >> 16), v


# ---------------------------------------------------------------------------
# T-P3-1
# ---------------------------------------------------------------------------

def test_no_tunnelling_at_max_speed_and_corner_cut_is_allowed():
    """(a) At step_q = FP_ONE - 1 a larva never enters a solid tile and never
    crosses a 1-thick wall line (axis-aligned, and a 4-connected staircase),
    from many sub-tile offsets and all 64 headings k * PI / 32, over 200
    ticks. (b) The accepted corner cut: between two orthogonally solid tiles
    a diagonal move lands in the open diagonal tile.

    Must break if the probe skips its target tile, tests the pre-move tile,
    or the step reaches a whole tile (R1 relaxed) -- or (b) if someone
    "fixes" the accepted corner cut."""
    # (a1) axis-aligned wall at column 16 spanning every row.
    h = w = 32
    solid = np.zeros((h, w), dtype=bool)
    solid[:, 16] = True
    wd = _world(h, w, 2048, solid=solid, step_q=ONE - 1, p_base_q=6000, p_alarm_q=6000)
    slot = 0
    offsets = (0, 1, 16384, 32768, 49152, ONE - 1)
    for k in range(64):
        for ox in offsets[::2]:
            for side_tx in (15, 17):
                tk.place(wd, 0, slot, (side_tx << 16) + ox, ((2 + (slot % 28)) << 16) + offsets[k % 6],
                         k * PI // 32)
                slot += 1
    start_side = (wd.pos_x[0][: slot] >> 16) > 16
    for t in range(200):
        _tick(wd, t)
        tx, ty, v = _tiles(wd)
        assert not solid[ty, tx].any(), f"tick {t}: a larva stands on a wall tile"
        side = (wd.pos_x[0][:slot] >> 16) > 16
        assert np.array_equal(side[v[:slot]], start_side[v[:slot]]), (
            f"tick {t}: a larva crossed the axis-aligned wall")
    _branches_taken("probe_solid", "moved", "tumble_taken", "sweep_solid")

    # (a2) a 4-connected staircase: solid where tx == ty or tx == ty + 1.
    h = w = 40
    yy, xx = np.mgrid[0:h, 0:w]
    solid = (xx == yy) | (xx == yy + 1)
    wd = _world(h, w, 2048, solid=solid, step_q=ONE - 1, p_base_q=6000, p_alarm_q=6000)
    slot = 0
    for k in range(64):
        for (tx0, ty0) in ((5, 12), (12, 25), (20, 8), (30, 18)):
            tk.place(wd, 0, slot, (tx0 << 16) + offsets[k % 6], (ty0 << 16) + offsets[(k + 3) % 6],
                     k * PI // 32)
            slot += 1
    def below(world):
        return (world.pos_x[0][:slot] >> 16) < (world.pos_y[0][:slot] >> 16)
    start = below(wd)
    for t in range(200):
        _tick(wd, t)
        tx, ty, v = _tiles(wd)
        assert not solid[ty, tx].any(), f"tick {t}: a larva stands in the staircase"
        assert np.array_equal(below(wd)[v[:slot]], start[v[:slot]]), (
            f"tick {t}: a larva crossed the 4-connected staircase")

    # (b) the corner cut.
    tx, ty = 5, 5
    solid = np.zeros((12, 12), dtype=bool)
    solid[ty, tx + 1] = True
    solid[ty + 1, tx] = True
    wd = _world(12, 12, 4, solid=solid, step_q=1000, p_base_q=0, p_alarm_q=0)
    tk.place(wd, 0, 0, ((tx + 1) << 16) - 1, ((ty + 1) << 16) - 1, PI // 4)
    _tick(wd, 0)
    assert (int(wd.pos_x[0, 0]) >> 16, int(wd.pos_y[0, 0]) >> 16) == (tx + 1, ty + 1)


# ---------------------------------------------------------------------------
# T-P3-2
# ---------------------------------------------------------------------------

def test_out_of_grid_is_blocked_and_takes_the_wall_turn():
    """On a grid with NO solid tile, larvae one step from each edge heading
    out stay inside [0, w<<16) x [0, h<<16) every tick, and the edge hit is
    the wall-turn path: that tick the heading changes and the position does
    not.

    Must break if the probe clamps to the grid instead of refusing, or reads
    an out-of-grid tile."""
    h, w = 10, 14
    step = 3000
    wd = _world(h, w, 64, step_q=step, p_base_q=0, p_alarm_q=0)
    starts = [
        (step // 2, (5 << 16) + 100, PI),                        # west edge, heading west
        ((w << 16) - 1 - step // 2, (4 << 16) + 7, 0),           # east edge, heading east
        ((6 << 16) + 3, step // 2, -PI // 2),                    # north (y = 0), heading -y
        ((7 << 16) + 9, (h << 16) - 1 - step // 2, PI // 2),     # south, heading +y
    ]
    for s, (x, y, hd) in enumerate(starts):
        tk.place(wd, 0, s, x, y, hd)
    before = (wd.pos_x[0, :4].copy(), wd.pos_y[0, :4].copy(), wd.heading[0, :4].copy())
    _tick(wd, 0)
    assert np.array_equal(wd.pos_x[0, :4], before[0])
    assert np.array_equal(wd.pos_y[0, :4], before[1])
    assert (wd.heading[0, :4] != before[2]).all(), "an edge hit must turn"
    _branches_taken("probe_edge", "sweep_off_grid")
    wd.tables[:, 9] = wd.tables[:, 10] = 20000      # tumble a lot from here on
    for t in range(1, 400):
        _tick(wd, t)
        assert ((wd.pos_x[0, :4] >= 0) & (wd.pos_x[0, :4] < (w << 16))).all()
        assert ((wd.pos_y[0, :4] >= 0) & (wd.pos_y[0, :4] < (h << 16))).all()


# ---------------------------------------------------------------------------
# T-P3-3
# ---------------------------------------------------------------------------

def test_every_write_keeps_the_store_valid():
    """A 500-larva, 300-tick run with high tumble rates, interior walls and a
    field straddling every threshold: after every tick the REAL
    swarm.check_invariants is clean (canonical headings, in-grid positions,
    states, the empty-slot form).

    Must break on a raw h +- m store, an int32 wrap, TWO_PI_Q16 as the
    period, a move out of the grid or a dead slot left non-zero."""
    rng = np.random.default_rng(3)
    h, w = 24, 30
    solid = rng.random((h, w)) < 0.12
    T = rng.integers(-30 * ONE, 60 * ONE, size=(h, w)).astype(np.int32)
    wd = _world(h, w, 600, T=T, solid=solid, p_base_q=15000, p_alarm_q=40000, step_q=9000)
    open_t = np.argwhere(~solid)
    for s in range(500):
        ty, tx = open_t[rng.integers(len(open_t))]
        tk.place(wd, 0, s, (int(tx) << 16) + int(rng.integers(ONE)),
                 (int(ty) << 16) + int(rng.integers(ONE)), int(rng.integers(-PI + 1, PI + 1)),
                 state=int(rng.choice([swarm.AI_RUN, swarm.AI_EAT, swarm.AI_COMA])))
    for t in range(300):
        _tick(wd, t)
        viol = swarm.check_invariants(tk.as_store_namespace(wd, 0))
        assert viol == [], f"tick {t}: {viol}"
    _branches_taken("wrap_neg_remainder", "died", "run_to_coma", "coma_to_run")


# ---------------------------------------------------------------------------
# T-P3-4
# ---------------------------------------------------------------------------

def test_dist_walked_wraps_mod_2_32_and_only_on_moves():
    """dist_walked = 2**32 - 3 plus a step of 10 reads 7 after an unblocked
    tick (uint32 wrap), and is unchanged after a blocked tick.

    Must break on signed or overflowing accumulation, or on accumulating
    a blocked tick."""
    wd = _world(10, 10, 4, step_q=10, p_base_q=0, p_alarm_q=0)
    tk.place(wd, 0, 0, (4 << 16) + 100, (4 << 16) + 100, 0, dist=2 ** 32 - 3)
    _tick(wd, 0)
    assert int(wd.dist_walked[0, 0]) == 7
    wd2 = _world(10, 10, 4, step_q=10, p_base_q=0, p_alarm_q=0)
    tk.place(wd2, 0, 0, (10 << 16) - 3, (4 << 16) + 100, 0, dist=2 ** 32 - 3)
    _tick(wd2, 0)
    assert int(wd2.dist_walked[0, 0]) == 2 ** 32 - 3
    assert int(wd2.pos_x[0, 0]) == (10 << 16) - 3


# ---------------------------------------------------------------------------
# T-P3-5, T-P3-6
# ---------------------------------------------------------------------------

def test_coma_hysteresis_both_halves():
    """On flat fields at T_ctmin - 1, every value in [T_ctmin, T_coma_exit]
    and T_coma_exit + 1 (one env each): RUN -> COMA only below T_ctmin, COMA
    -> RUN only above T_coma_exit, inside the band each state holds; a COMA
    larva's position, heading and dist_walked are unchanged, and its columns
    are identical under two different seeds (no draw reaches it).

    Must break with a single threshold, <= for <, or sensing, drawing or
    moving in COMA."""
    ctmin, cexit = -100, -90
    values = list(range(ctmin - 1, cexit + 2))
    n = len(values)
    T = np.zeros((n, 8, 8), dtype=np.int32)
    for e, v in enumerate(values):
        T[e] = v
    outs = []
    for seed in ((1, 2), (99, 7)):
        wd = tk.larva_world(8, 8, 2, n, T=T, seed=seed, T_ctmin_q=ctmin, T_coma_exit_q=cexit,
                            p_base_q=30000, p_alarm_q=30000)
        for e in range(n):
            tk.place(wd, e, 0, (3 << 16) + 5, (3 << 16) + 7, 1000, state=swarm.AI_RUN)
            tk.place(wd, e, 1, (4 << 16) + 5, (4 << 16) + 7, -2000, state=swarm.AI_COMA)
        tk.step_cpu(wd)
        outs.append(wd)
        for e, v in enumerate(values):
            run_after = int(wd.ai_state[e, 0])
            coma_after = int(wd.ai_state[e, 1])
            assert run_after == (swarm.AI_COMA if v < ctmin else swarm.AI_RUN), v
            assert coma_after == (swarm.AI_RUN if v > cexit else swarm.AI_COMA), v
            assert int(wd.pos_x[e, 1]) == (4 << 16) + 5 and int(wd.pos_y[e, 1]) == (4 << 16) + 7
            assert int(wd.heading[e, 1]) == -2000 and int(wd.dist_walked[e, 1]) == 0
    for c in swarm.ROSTER:
        assert np.array_equal(getattr(outs[0], c.name)[:, 1], getattr(outs[1], c.name)[:, 1]), (
            f"a COMA larva's {c.name} depends on the seed: a draw reached it")
    _branches_taken("run_to_coma", "coma_held", "coma_to_run")


def test_waking_consumes_the_tick():
    """A COMA larva whose field rises above T_coma_exit becomes RUN with its
    position unchanged that tick, and moves on the next.

    Must break if waking and moving happen in one tick."""
    wd = _world(8, 8, 2, p_base_q=0, p_alarm_q=0, step_q=2000)
    wd.T[:] = -50 * ONE
    tk.place(wd, 0, 0, (3 << 16) + 5, (3 << 16) + 5, 0, state=swarm.AI_COMA)
    _tick(wd, 0)
    assert int(wd.ai_state[0, 0]) == swarm.AI_COMA
    wd.T[:] = 0
    x0 = int(wd.pos_x[0, 0])
    _tick(wd, 1)
    assert int(wd.ai_state[0, 0]) == swarm.AI_RUN and int(wd.pos_x[0, 0]) == x0
    _tick(wd, 2)
    assert int(wd.pos_x[0, 0]) == x0 + 2000


# ---------------------------------------------------------------------------
# T-P3-7, T-P3-8
# ---------------------------------------------------------------------------

def test_kill_reads_the_post_move_tile_in_every_state():
    """A lethal block (T_lethal_hot + 1 tile of counts) in a safe field: (a) a
    RUN larva half a step before the block heading in dies on the tick it
    crosses, its record at the POST-move position (lethal by the RAW oracle,
    the pre-move tile safe); (b) a larva inside, half a step from the edge
    heading out, survives; (c) a COMA larva whose tile is made lethal dies.
    Each record carries the larva's id and the heading it moved along.

    Must break on pre-move sampling, a COMA exemption, or a record built
    from the zeroed slot."""
    step = 2000
    T = np.zeros((10, 20), dtype=np.int32)
    T[:, 10:14] = tk.DEFAULT_LARVA["T_lethal_hot_q"] + ONE
    wd = _world(10, 20, 8, T=T, step_q=step, p_base_q=0, p_alarm_q=0)
    ua = tk.place(wd, 0, 0, (10 << 16) - step // 2, (3 << 16) + 9, 0)
    ub = tk.place(wd, 0, 1, (14 << 16) - step // 2, (5 << 16) + 9, 0)
    uc = tk.place(wd, 0, 2, (2 << 16) + 9, (7 << 16) + 9, 1234, state=swarm.AI_COMA)
    wd.T[0, 7, 2] = tk.DEFAULT_LARVA["T_lethal_hot_q"] + 1
    assert int(T[3, ((10 << 16) - step // 2) >> 16]) < tk.DEFAULT_LARVA["T_lethal_hot_q"]
    deaths = _tick(wd, 0)
    assert deaths == 2
    rec = wd.death[0]
    assert list(rec[0]) == [ua, (10 << 16) - step // 2 + step, (3 << 16) + 9, 0]
    assert int(wd.T[0, rec[0][2] >> 16, rec[0][1] >> 16]) > tk.DEFAULT_LARVA["T_lethal_hot_q"]
    assert int(wd.valid[0, 1]) == 1 and int(wd.unit_id[0, 1]) == ub
    assert list(rec[2]) == [uc, (2 << 16) + 9, (7 << 16) + 9, 1234]
    _branches_taken("died", "survived", "moved")


def test_death_writes_the_empty_form_and_records_every_slot_below_hw():
    """After a kill all 9 columns of the slot equal a never-spawned slot's;
    every slot in [0, hw) without a death (an invalid slot included) has a
    record of 4 zeros though the records were pre-filled with garbage;
    record slots >= hw keep their garbage.

    Must break if a death leaves valid = 0 with other columns set, a stale
    record survives below hw, or a record is written beyond hw."""
    T = np.zeros((8, 8), dtype=np.int32)
    T[2, 2] = tk.DEFAULT_LARVA["T_lethal_hot_q"] + 5
    wd = _world(8, 8, 10, T=T, p_base_q=0, p_alarm_q=0, step_q=100)
    tk.place(wd, 0, 0, (2 << 16) + 300, (2 << 16) + 300, 0)
    tk.place(wd, 0, 2, (5 << 16) + 300, (5 << 16) + 300, 0)
    tk.place(wd, 0, 4, (6 << 16) + 300, (5 << 16) + 300, 0)     # hw = 5; slots 1, 3 empty
    _tick(wd, 0)
    for c in swarm.ROSTER:
        assert int(getattr(wd, c.name)[0, 0]) == 0, c.name
    assert (wd.death[0, 1:5] == 0).all()
    assert (wd.death[0, 5:] == -7).all()
    assert swarm.check_invariants(tk.as_store_namespace(wd, 0)) == []
    _branches_taken("slot_invalid", "died")


# ---------------------------------------------------------------------------
# T-P3-9, T-P3-10, T-P3-11
# ---------------------------------------------------------------------------

def _binomial_band(count, n, p):
    sd = np.sqrt(p * (1 - p) / n)
    return abs(count / n - p) <= 4 * sd


def test_a_blocked_sweep_point_is_no_reading():
    """(a) K larvae half a tile from a wall, facing it (both sweep points in
    the wall, the probe target open), on a flat comfortable field whose wall
    tiles hold a far worse temperature: over K x J independent single-tick
    trials the tumble frequency is p_base's (binomial 4 sigma) and outside
    p_alarm's band. (b) A wall hit with one side open: the wall turn goes to
    the open (PLUS) side with frequency turn_bias.

    Must break if a blocked point counts as worst (the alarm rate) or as a
    reading (the wall's own temperature), or if the side choice ignores
    blockage."""
    K, J = 2000, 40
    p_base, p_alarm = 3000, 30000
    T = np.zeros((6, 12), dtype=np.int32)
    solid = np.zeros((6, 12), dtype=bool)
    solid[:, 9] = True
    T[:, 9] = 40 * ONE              # a reading of the wall would be "worse"
    wd = _world(6, 12, K, T=T, solid=solid, p_base_q=p_base, p_alarm_q=p_alarm, step_q=1000)
    for s in range(K):
        tk.place(wd, 0, s, (9 << 16) - ONE // 2, (2 << 16) + ONE // 2, 0)
    base = wd.copy()
    tumbles = 0
    for j in range(J):
        w = base.copy()
        _tick(w, j)
        assert (w.pos_x[0] == base.pos_x[0] + 1000).all(), "the probe target must be open"
        tumbles += int((w.heading[0] != base.heading[0]).sum())
    assert _binomial_band(tumbles, K * J, p_base / ONE)
    assert not _binomial_band(tumbles, K * J, p_alarm / ONE)
    _branches_taken("sweep_solid", "pref_none_blocked", "alarm_off", "tumble_taken")

    # (b) one sweep side open, the probe blocked.
    step = 2000
    solid = np.zeros((20, 20), dtype=bool)
    solid[:, 11] = True
    solid[11, 11] = False
    bias = tk.DEFAULT_LARVA["turn_bias_q"]
    wd = _world(20, 20, K, solid=solid, p_base_q=0, p_alarm_q=0, step_q=step)
    for s in range(K):
        tk.place(wd, 0, s, (11 << 16) - step // 2, (10 << 16) + ONE // 2, 0)
    base = wd.copy()
    plus = 0
    for j in range(J):
        w = base.copy()
        _tick(w, j)
        assert (w.pos_x[0] == base.pos_x[0]).all(), "every larva must hit the wall"
        plus += int((w.heading[0] > 0).sum())
    assert _binomial_band(plus, K * J, bias / ONE)
    _branches_taken("pref_plus", "turn_toward", "turn_away", "probe_solid")


def _draw_words(salt, uid, tick):
    d = MatchDraws(SEED)
    d.set_tick(tick)
    return bp.philox32_4x32_10(*d.at(salt, uid, 0).counter(), *d.key)


def test_the_wall_turn_wins_over_the_tumble():
    """With p_base = p_alarm = FP_ONE (always tumble) and a blocked probe
    (both sweep points blocked too: no preference), the written heading is
    turn_heading on the WALL_TURN block's words (from the sim's own draws,
    MatchDraws.at(Salt.SWARM_WALL_TURN, uid, 0)), never the TUMBLE block's.

    Must break if the tumble overrides the wall turn, or both are applied."""
    K = 300
    solid = np.zeros((6, 12), dtype=bool)
    solid[:, 9] = True
    step = 2000
    wd = _world(6, 12, K, solid=solid, p_base_q=ONE, p_alarm_q=ONE, step_q=step)
    for s in range(K):
        tk.place(wd, 0, s, (9 << 16) - step // 2, (2 << 16) + ONE // 2, 0)
    tick = 77
    _tick(wd, tick)
    P = tk.DEFAULT_LARVA
    differs = 0
    for s in range(K):
        uid = int(wd.unit_id[0, s])
        ww = _draw_words(Salt.SWARM_WALL_TURN, uid, tick)
        wt = _draw_words(Salt.SWARM_TUMBLE, uid, tick)
        want = tk.turn_heading_ref(0, 0, ww[0], ww[1], P["turn_bias_q"], P["turn_min_q"],
                                   P["turn_max_q"])
        tumb = tk.turn_heading_ref(0, 0, wt[1], wt[2], P["turn_bias_q"], P["turn_min_q"],
                                   P["turn_max_q"])
        assert int(wd.heading[0, s]) == want
        differs += int(tumb != want)
    assert differs > K // 2
    _branches_taken("pref_none_blocked", "turn_by_coin", "tumble_taken", "probe_solid")


def test_the_draw_layout_is_the_draw_table():
    """For RUN larvae in open space (no preference on a flat field) the
    tumble decision and the new heading are the values computed from the
    TUMBLE block (tick, 0, Salt.SWARM_TUMBLE, uid) under the key: word 0
    through philox32_uniform_q16 against p, words 1 / 2 through the turn
    draw and swarm_fixed.wrap_heading_q16.

    Must break on a salt swap, a draw_index other than 0, a `%` or
    non-integer mapping, or a word reorder."""
    K = 400
    p = 30000
    wd = _world(30, 30, K, p_base_q=p, p_alarm_q=p, step_q=500)
    rng = np.random.default_rng(11)
    for s in range(K):
        tk.place(wd, 0, s, (15 << 16) + int(rng.integers(ONE)), (15 << 16) + int(rng.integers(ONE)),
                 int(rng.integers(-PI + 1, PI + 1)))
    h0 = wd.heading[0].copy()
    tick = 4242
    _tick(wd, tick)
    P = tk.DEFAULT_LARVA
    n_tumble = 0
    for s in range(K):
        uid = int(wd.unit_id[0, s])
        w = _draw_words(Salt.SWARM_TUMBLE, uid, tick)
        tumble = bp.philox32_uniform_q16(w[0]) < p
        want = (tk.turn_heading_ref(int(h0[s]), 0, w[1], w[2], P["turn_bias_q"],
                                    P["turn_min_q"], P["turn_max_q"]) if tumble else int(h0[s]))
        assert int(wd.heading[0, s]) == want, s
        n_tumble += tumble
    assert 0 < n_tumble < K
    _branches_taken("pref_none_equal", "turn_by_coin", "tumble_not_taken")


# ---------------------------------------------------------------------------
# T-P3-12, T-P3-13, T-P3-14
# ---------------------------------------------------------------------------

def test_per_slot_independence():
    """Larva A's full trajectory (all 9 columns, every tick) alone equals its
    trajectory with 200 other larvae in other slots (same uid, same slot).

    Must break on any cross-slot read, a shared counter, or a draw keyed on
    the slot or the population."""
    rng = np.random.default_rng(5)
    solid = rng.random((20, 20)) < 0.1
    solid[10, 10] = False
    T = rng.integers(-5 * ONE, 5 * ONE, size=(20, 20)).astype(np.int32)
    alone = _world(20, 20, 256, T=T, solid=solid, p_base_q=8000, p_alarm_q=20000, step_q=6000)
    crowd = alone.copy()
    tk.place(alone, 0, 50, (10 << 16) + 7, (10 << 16) + 9, 333, uid=777)
    tk.place(crowd, 0, 50, (10 << 16) + 7, (10 << 16) + 9, 333, uid=777)
    open_t = np.argwhere(~solid)
    others = [s for s in range(256) if s != 50][:200]
    for i, s in enumerate(others):
        ty, tx = open_t[rng.integers(len(open_t))]
        tk.place(crowd, 0, s, (int(tx) << 16) + 3, (int(ty) << 16) + 5,
                 int(rng.integers(-PI + 1, PI + 1)), uid=1000 + i)
    for t in range(300):
        _tick(alone, t)
        _tick(crowd, t)
        for c in swarm.ROSTER:
            assert getattr(alone, c.name)[0, 50] == getattr(crowd, c.name)[0, 50], (t, c.name)


def test_every_counter_and_key_word_reaches_the_draws():
    """Changing only the tick base, only the key's low word, only its high
    word, or only the unit id changes a larva's heading sequence over 500
    ticks.

    Must break if a counter or key word is ignored."""
    def run(tick0=0, seed=KEY, uid=5):
        wd = tk.larva_world(40, 40, 2, 1, seed=seed, p_base_q=6000, p_alarm_q=6000, step_q=3000)
        tk.place(wd, 0, 0, (20 << 16), (20 << 16), 0, uid=uid)
        seq = []
        for t in range(500):
            wd.env[:, swarm.ENV_TICK] = (tick0 + t) & 0xFFFFFFFF
            tk.step_cpu(wd)
            seq.append(int(wd.heading[0, 0]))
        return seq
    ref = run()
    assert run(tick0=1000) != ref
    assert run(seed=(KEY[0] ^ 1, KEY[1])) != ref
    assert run(seed=(KEY[0], KEY[1] ^ 1)) != ref
    assert run(uid=6) != ref


def test_invalid_and_out_of_hw_slots_are_inert():
    """Garbage in every column of the slots >= hw survives untouched; valid
    = 0 slots below hw stay all-zero; hw = 0 touches nothing and returns 0.

    Must break if the loop bound is the capacity or empty slots are
    processed."""
    rng = np.random.default_rng(9)
    wd = _world(8, 8, 32, p_base_q=20000, p_alarm_q=20000, step_q=4000)
    for s in (0, 3, 5):
        tk.place(wd, 0, s, (3 << 16) + s, (3 << 16) + s, s * 100)
    for c in swarm.ROSTER:
        arr = getattr(wd, c.name)
        arr[0, 10:] = rng.integers(1, 100, size=arr[0, 10:].shape).astype(arr.dtype)
    snap = wd.copy()
    for t in range(20):
        _tick(wd, t)
    for c in swarm.ROSTER:
        assert np.array_equal(getattr(wd, c.name)[0, 6:], getattr(snap, c.name)[0, 6:]), c.name
        assert (getattr(wd, c.name)[0, [1, 2, 4]] == 0).all(), c.name
    wd0 = _world(8, 8, 8)
    for c in swarm.ROSTER:
        getattr(wd0, c.name)[0] = 9
    snap0 = wd0.copy()
    assert _tick(wd0, 0) == 0
    for name in [c.name for c in swarm.ROSTER] + ["death"]:
        assert np.array_equal(getattr(wd0, name), getattr(snap0, name)), name


# ---------------------------------------------------------------------------
# G-d: every branch of the law, in one CPU scene
# ---------------------------------------------------------------------------

def test_every_law_branch_is_taken_by_the_fuzz_scene():
    """A seeded fuzz scene (walls, edges, a field straddling every threshold,
    every state, every turn case) takes EVERY branch of the larva law
    (bp.SWARM_LARVA_BRANCHES) at least once on the CPU twin -- so the
    branch tally is not vacuous and the CUDA gate's PART 1 can lean on it.

    Must break when a forcer stops reaching a branch, or a branch is added
    to the law without a scene that takes it."""
    rng = np.random.default_rng(2026)
    for trial in range(6):
        h, w = int(rng.integers(6, 20)), int(rng.integers(6, 20))
        solid = rng.random((h, w)) < 0.2
        # a few levels only, so equal comfort on both sides happens too
        T = (rng.integers(-4, 8, size=(h, w)) * 10 * ONE).astype(np.int32)
        wd = _world(h, w, 300, T=T, solid=solid, p_base_q=int(rng.integers(0, ONE)),
                    p_alarm_q=ONE, step_q=int(rng.integers(1, ONE)))
        for s in range(300):
            tk.place(wd, 0, s, int(rng.integers(0, w << 16)), int(rng.integers(0, h << 16)),
                     int(rng.integers(-PI + 1, PI + 1)),
                     state=int(rng.choice([swarm.AI_RUN, swarm.AI_EAT, swarm.AI_COMA])))
        wd.valid[0, ::7] = 0
        for c in swarm.ROSTER:
            getattr(wd, c.name)[0, ::7] = 0
        for t in range(30):
            _tick(wd, t)
    _branches_taken(*bp.SWARM_LARVA_BRANCHES)
