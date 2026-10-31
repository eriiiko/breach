"""Creature primitives (#106 P6d2): the hub's state machines -- reach, crawl,
swim, grab, idle -- ported from the float prototypes P1-P5 / P5z
(``prototypes/octopus_motion/``: ``crawl_sim.py``, ``swim_sim.py``,
``grab_sim.py``, ``idle_sim.py`` and their ``*3d.py`` game-scale versions)
into integers over the ``[creatures.*]`` row's dials.

Design: docs/octopus_p6_sim_port_design_2026-10-30.md sections 2, 4.3, 4.5.
ONE module owns the state machines; ``creature_system.CreatureBody`` owns the
state they read and write (hashed in the ``__creature__`` record) and calls
:func:`hub_step` before its one solver call and :func:`after_solver` after it.

The rules every primitive keeps
-------------------------------
* It writes ONLY the solver's per-tick inputs for its arms -- ``act`` (or a
  stiffening wave through the ``arm_cfg`` wave columns, its ``wave_clock``
  zeroed at the start), ``rest_scale``, ``bend_target``, the anchors
  (``anchored`` / ``anchor_pos``), the grabs, the roots' body-frame places
  (``root_local``: the aim and the lift of an arm's root, the swimming ring)
  -- and the swim jet, a VELOCITY add on the crown before the solver call.
  Nothing writes a position: the solver is the only mover of nodes and of the
  crown.
* It reads the solver's report after a call (``anchored`` cleared by a pop,
  ``grab_force``) and the node positions, never re-deriving a contact the
  solver reports. (The grab's touch of the PROP is a distance test here: the
  solver reports node contacts with the floor and the walls only.)
* All state is integer; every timer an ELAPSED-tick counter (``arm_ticks``,
  ``decision_ticks``, ``swim_ticks``), never ``sim.tick``; every random draw
  through the body's own xorshift64 (never ``sim.rng``), arms in index order.
* Decisions happen on the row's comb, ``crawl.decision_ticks`` (0.1 s -> 2
  ticks at 24 Hz; the float prototypes decided every 3 ticks, see the run
  log).

The goal table (what the intents ask for -> which primitive runs)
----------------------------------------------------------------
==========================================  ==============================
intent on the creature unit                 primitive
==========================================  ==============================
none                                        idle (seated; explore, touch, limp)
MOVE_DIR (any speed mode but the two below) crawl toward the direction
MOVE_DIR at ``ORDER_MOVE_COVER``            swim: the umbrella (the cruise)
MOVE_DIR at ``ORDER_SPRINT``                swim: the jet train (the escape)
AIM (with or without a MOVE_DIR)            reach: the best-placed arm
USE (latched; needs a creature-owned prop)  grab the prop: reach, wrap,
                                            hold, lift, then carry it
==========================================  ==============================

While the grab arm holds the prop the other arms crawl (any MOVE_DIR) and
the escape becomes the CARRY JETS (``carry_jet_dv`` impulses on the escape
cycle) instead of a swim: the carry of P4. A swim does not steer in v1: the
jet pushes along the heading (mantle first), the umbrella's thrust is along
it. A creature in the LIMP gait (``CreatureBody.set_limp``) runs no primitive.
"""
from __future__ import annotations

import numpy as np

from simulation import arm_chain as AC
from simulation import creature_fixed as CF
from simulation import fixedpoint_twins as K
from simulation.orders import ORDER_MOVE_COVER, ORDER_SPRINT

ONE = 1 << 16
PI = K.PI_Q16
TWO_PI = 2 * PI

# ---------------------------------------------------------------------------
# The states (record constants: a uint8 per arm, one per body)
# ---------------------------------------------------------------------------
#: Per-arm states. LIMP: no primitive (the LIMP gait). Crawl (P2): READY,
#: STANCE (anchored, pushing), RECOVER. Idle (P5): STILL (seated), DECIDE,
#: EXPLORE (the reach wave), TOUCH, RELAX (limp, then seated again). Reach
#: (P1): REACH. Swim (P3): SWIM. Grab (P4): GRAB_REACH, GRAB_WRAP, GRAB_HOLD,
#: CARRY.
(ARM_LIMP, ARM_READY, ARM_STANCE, ARM_RECOVER, ARM_STILL, ARM_DECIDE, ARM_EXPLORE,
 ARM_TOUCH, ARM_RELAX, ARM_REACH, ARM_SWIM, ARM_GRAB_REACH, ARM_GRAB_WRAP, ARM_GRAB_HOLD,
 ARM_CARRY) = range(15)
ARM_STATE_NAMES = ("limp", "ready", "stance", "recover", "still", "decide", "explore",
                   "touch", "relax", "reach", "swim", "grab_reach", "grab_wrap",
                   "grab_hold", "carry")
GRAB_STATES = (ARM_GRAB_REACH, ARM_GRAB_WRAP, ARM_GRAB_HOLD, ARM_CARRY)
SEALED_STATES = (ARM_GRAB_WRAP, ARM_GRAB_HOLD, ARM_CARRY)
IDLE_GESTURE = (ARM_DECIDE, ARM_EXPLORE, ARM_TOUCH, ARM_RELAX)

#: The body's gait: which primitive drives the arms no task holds.
GAIT_LIMP, GAIT_IDLE, GAIT_CRAWL, GAIT_SWIM = range(4)
GAIT_NAMES = ("limp", "idle", "crawl", "swim")

#: The travel goal (``goal_kind``, from MOVE_DIR's speed mode).
GOAL_NONE, GOAL_MOVE_DIR, GOAL_CRUISE, GOAL_ESCAPE = range(4)
#: MOVE_DIR's speed mode -> the travel goal; every other mode crawls.
SPEED_MODE_GOAL = {ORDER_MOVE_COVER: GOAL_CRUISE, ORDER_SPRINT: GOAL_ESCAPE}

#: The swim's phase clock (one per body): the umbrella OPEN (slow) / CLOSE
#: (fast), the JET and its COAST.
SWIM_OPEN, SWIM_CLOSE, SWIM_JET, SWIM_COAST = range(4)


def _q16(x: float) -> int:
    return CF.q16_from_scalar(float(x))


def _rad_q16(deg: float) -> int:
    return CF.q16_from_scalar(float(deg) * (np.pi / 180.0))


def _wrap(a: int) -> int:
    """A Q16 angle into (-pi, pi]."""
    a = (a + PI) % TWO_PI - PI
    return PI if a == -PI else a


def _ease(f_num: int, f_den: int) -> int:
    """``0.5 (1 - cos(pi f))`` in Q16 for f = f_num / f_den clamped to [0, 1]:
    the prototypes' cosine ease, through the kit's cos."""
    f = min(max(f_num, 0), f_den)
    ang = K.div_round_away(PI * f, f_den)
    return K.shr_round_away(ONE - K.cos_q16(ang), 1)


def _lerp(a: int, b: int, num: int, den: int) -> int:
    """``a + (b - a) min(num, den) / den``, rounded half away."""
    return a + K.div_round_away((b - a) * min(max(num, 0), den), den)


# ---------------------------------------------------------------------------
# The row's dials, quantized once per body (derived from the row: the row
# hash covers them, never stored)
# ---------------------------------------------------------------------------
class PrimDials:
    """Every primitive dial in the integer currencies the hub computes in:
    Q16 fractions / radians / scale factors, Q24 lengths and speeds (the chain
    currency), ticks (the row's ``*_ticks``), substeps for the waves."""

    def __init__(self, cdef):
        b, c, sw, g, i, r = cdef.body, cdef.crawl, cdef.swim, cdef.grab, cdef.idle, cdef.reach
        self.A, self.n = int(b.arms), int(b.nodes_per_arm)
        self.n_sub = int(cdef.solver.substeps)
        self.seg_q24 = CF.chain_from_m(b.arm_length_m / (self.n - 1))
        self.rim_q24 = CF.chain_from_m(b.body_radius_m)
        L_q24 = CF.chain_from_m(b.arm_length_m)
        # per arm: the body-frame angle (the body frame is y-down like the
        # grid, so a screen-CCW row angle is its negative: rest_pose's rule),
        # frontness / backness = (1 +- cos) / 2
        self.phi = [_q16(-float(d) * (np.pi / 180.0)) for d in b.arm_angles_deg]
        cos_phi = [K.cos_q16(p) for p in self.phi]
        self.front = [K.shr_round_away(ONE + cp, 1) for cp in cos_phi]
        self.back = [K.shr_round_away(ONE - cp, 1) for cp in cos_phi]
        # the stiffening wave's front width (one segment): set_wave's door
        self.w_q24 = CF.chain_from_m(r.front_width_m)
        if 8 * self.w_q24 > 3 * L_q24 or self.w_q24 < 1:
            raise ValueError("creature_primitives: reach.front_width_m must be in (0, 3 L / 8]")
        # reach (P1)
        self.reach_a = _q16(r.a_ca)
        self.reach_tact = int(r.t_act_substeps)
        # crawl (P2)
        self.anchor_nodes = tuple(int(k) for k in c.anchor_nodes)
        self.n_anc = max(self.anchor_nodes) + 1            # bends / segments that push
        self.e_max = _q16(c.e_max)
        self.push_ticks, self.push_jit = int(c.push_ticks), int(c.push_jitter_ticks)
        self.recover_ticks, self.recover_jit = int(c.recover_ticks), int(c.recover_jitter_ticks)
        self.comb = int(c.decision_ticks)
        self.p_recruit = _q16(c.p_recruit)
        self.max_stance = int(c.max_stance)
        self.s_min = _q16(c.s_min)
        self.back_bias = _q16(c.back_bias)
        self.nb_bonus = _q16(c.neighbour_bonus)
        self.stance_act = _q16(c.stance_activation)
        self.steer_gain = _q16(c.steer_gain)
        self.steer_min = _q16(c.steer_min_dist)            # Q16 m
        self.steer_k = -(-int(c.steer_window_ticks) // self.comb)   # ring samples
        self.grip_q8 = AC.grip_q8(c.grip_limit)
        # swim (P3)
        self.open_rad, self.closed_rad = _rad_q16(sw.open_deg), _rad_q16(sw.closed_deg)
        self.bundled_rad = _rad_q16(sw.bundled_deg)
        self.open_ticks, self.close_ticks = int(sw.open_ticks), int(sw.close_ticks)
        self.swim_act = _q16(sw.activation)
        self.jet_dv_q24 = CF.chain_from_m(sw.jet_dv)
        self.jet_ticks = int(sw.jet_ticks)
        if self.jet_ticks >= self.open_ticks:
            raise ValueError("creature_primitives: swim.jet_s must be shorter than swim.open_s "
                             "(the escape fires one jet per opening time)")
        self.jet_relax = min(ONE, K.div_round_away(4 * ONE, self.jet_ticks))
        self.ring_q24 = CF.chain_from_m(sw.root_ring_frac * b.body_radius_m)
        # the roots move between the deck pose and the swimming ring in one
        # CLOSING time (the fast stroke): no new dial, and independent of the
        # escape's period (one jet per opening time)
        self.rig_step = max(1, K.div_round_away(ONE, self.close_ticks))
        # grab (P4)
        self.grab_arm = int(g.arm)
        self.prop_at_q24 = (CF.chain_from_m(g.prop_at_m[0]), CF.chain_from_m(g.prop_at_m[1]))
        self.prop_r_q24 = CF.chain_from_m(g.prop_radius)
        self.touch_r_q24 = CF.chain_from_m(g.prop_radius + g.sucker_reach)
        self.grab_k0, self.grab_k1 = int(g.grab_nodes[0]), int(g.grab_nodes[-1])
        self.grab_tact = int(g.reach_wave_substeps)
        self.grab_reach_ticks = int(g.reach_wave_ticks)
        self.grab_stretch = _q16(g.reach_stretch)
        self.grab_reach_a = _q16(g.reach_activation)
        self.wrap_act, self.hold_act = _q16(g.wrap_activation), _q16(g.hold_activation)
        self.pull_in = _q16(g.pull_in_scale)
        self.pull_in_ticks, self.hold_ticks = int(g.pull_in_ticks), int(g.hold_ticks)
        self.wrap_ticks = int(g.wrap_ticks)
        self.grab_grip_q8 = AC.grip_q8(g.grip_limit)
        self.aim_max = _rad_q16(g.aim_max_deg)
        self.lift = _rad_q16(g.lift_pitch_deg)
        self.lift_ticks = int(g.lift_ticks)
        self.carry_dv_q24 = CF.chain_from_m(g.carry_jet_dv)
        # one segment's turn on the prop's rim (grab_sim: rest0 / radius), Q16 rad
        self.wrap_turn = K.div_round_away(self.seg_q24 << 16, self.prop_r_q24)
        # idle (P5)
        self.max_active = int(i.max_active)
        self.p_start = _q16(i.p_start)
        self.decide_ticks, self.decide_jit = int(i.hold_ticks), int(i.hold_jitter_ticks)
        self.wave_ticks, self.wave_jit = int(i.wave_ticks), int(i.wave_jitter_ticks)
        self.idle_stretch, self.idle_stretch_jit = _q16(i.stretch), _q16(i.stretch_jitter)
        self.idle_act = _q16(i.activation)
        self.touch_ticks, self.touch_jit = int(i.touch_ticks), int(i.touch_jitter_ticks)
        self.limp_ticks, self.limp_jit = int(i.limp_ticks), int(i.limp_jitter_ticks)
        self.idle_aim = _rad_q16(i.aim_deg)
        self.front_bias = _q16(i.front_bias)


# ---------------------------------------------------------------------------
# The body's own randomness (xorshift64; arms in index order)
# ---------------------------------------------------------------------------
def _draw(body) -> int:
    from simulation.creature_system import xorshift64
    body.rng, v = xorshift64(body.rng)
    return v


def _chance(body, p_q16: int) -> bool:
    """True with probability p (Q16): the top 16 bits below p."""
    return (_draw(body) >> 48) < p_q16


def _below(body, n: int) -> int:
    """A uniform integer in [0, n) (n >= 1), from the top 32 bits."""
    return ((_draw(body) >> 32) * int(n)) >> 32


def _jitter(body, base: int, jit: int) -> int:
    """``base + U{0 .. jit}`` ticks (a zero jitter draws nothing)."""
    return base + (_below(body, jit + 1) if jit > 0 else 0)


# ---------------------------------------------------------------------------
# Solver-input helpers
# ---------------------------------------------------------------------------
def _anchor(body, a: int, nodes) -> None:
    """Anchor ``nodes`` of arm ``a`` where they lie now (the sucker patch)."""
    b = body.bufs
    idx = list(nodes)
    b.anchored[0, a, idx] = 1
    b.anchor_pos[0, a, idx] = b.p[0, a, idx]


def _release(body, a: int) -> None:
    body.bufs.anchored[0, a, :] = 0


def _on_deck(body, a: int) -> list:
    """The anchor patch's nodes of arm ``a`` the solver reported on the
    FLOOR last tick (its contact report): a sucker grips the deck, never the
    water. An arm with none can neither be recruited nor seated."""
    c = body.bufs.contact[0, a]
    return [k for k in body.dials.anchor_nodes if int(c[k]) & AC.CONTACT_FLOOR]


def _seat(body, a: int) -> None:
    """Seat arm ``a`` (idle's STILL): anchor whichever of its patch lies on
    the deck; a patch in mid water stays free and is seated when it lands."""
    nodes = _on_deck(body, a)
    if nodes:
        _anchor(body, a, nodes)


def _wave_on(body, a: int, a_ca_q16: int, tact_sub: int) -> None:
    """A held stiffening wave on arm ``a`` (``arm_chain.set_wave``'s columns,
    integer: the travel time in substeps, the row's front width), its clock
    zeroed at the start."""
    cfg = body.bufs.arm_cfg[0, a]
    cfg[AC.AC_WAVE_ON] = 1
    cfg[AC.AC_WAVE_A_CA_Q16] = int(a_ca_q16)
    cfg[AC.AC_WAVE_TACT_SUB] = max(1, int(tact_sub))
    cfg[AC.AC_WAVE_W_Q24] = body.dials.w_q24
    cfg[AC.AC_WAVE_HOLD] = 1
    body.bufs.wave_clock[0, a] = 0


def _wave_off(body, a: int) -> None:
    body.bufs.arm_cfg[0, a, AC.AC_WAVE_ON] = 0


def _set_state(body, a: int, state: int, drawn: int | None = None) -> None:
    body.arm_state[a] = state
    body.arm_ticks[a] = 0
    if drawn is not None:
        body.arm_drawn[a] = int(drawn)


def _th16(body) -> int:
    return K.shr_round_away(int(body.bufs.body[0, 0, AC.BS_TH]), CF.CHAIN_SHIFT - 16)


def _unit_q16(x: int, y: int) -> tuple[int, int] | None:
    """(x, y) normalized to a Q16 unit vector, None for the zero vector."""
    x, y = int(x), int(y)
    # bring the pair to Q16-ish magnitude so the sum of squares fits int64
    while max(abs(x), abs(y)) >= (1 << 30):
        x, y = K.shr_round_away(x, 1), K.shr_round_away(y, 1)
    n = K.sqrt_q16(x * x + y * y)
    if n <= 0:
        return None
    return K.div_round_away(x << 16, n), K.div_round_away(y << 16, n)


# ---------------------------------------------------------------------------
# The current scale of an arm (what a transition starts from)
# ---------------------------------------------------------------------------
def _scale(body, a: int) -> int:
    d, s, t, L = body.dials, int(body.arm_state[a]), int(body.arm_ticks[a]), int(body.arm_drawn[a])
    st = int(body.arm_stretch[a])
    if s == ARM_STANCE:
        return _lerp(ONE, d.e_max, t, max(L, 1))
    if s in (ARM_RECOVER, ARM_RELAX):
        return _lerp(st, ONE, 2 * t, max(L, 1))
    if s == ARM_EXPLORE:
        return _lerp(ONE, st, t, max(L, 1))
    if s == ARM_TOUCH:
        return st
    return ONE


# ---------------------------------------------------------------------------
# Intents -> the goal (CreatureBody.route_intents calls this)
# ---------------------------------------------------------------------------
def route_intents(body, unit) -> None:
    """The intents of ``unit`` -> the body's goal: the travel goal from
    MOVE_DIR (its speed mode through ``SPEED_MODE_GOAL``), the reach goal from
    AIM, the grab request from a USE latch (consumed here: the conductor's
    USE slot skips a creature). A change of travel goal restarts the swim
    clock."""
    mv = getattr(unit, "live_move_dir", None)
    aim = getattr(unit, "live_aim", None)
    old = body.goal_kind
    if mv is not None and (mv.dx_q or mv.dy_q):
        g = _unit_q16(mv.dx_q, mv.dy_q)
        body.goal_kind = SPEED_MODE_GOAL.get(int(mv.speed_mode), GOAL_MOVE_DIR)
        body.goal_q16 = g
    else:
        body.goal_kind, body.goal_q16 = GOAL_NONE, (0, 0)
    if body.goal_kind != old:
        body.swim_ticks = 0
    if aim is not None and (aim.dx_q or aim.dy_q):
        body.reach_q16 = _unit_q16(aim.dx_q, aim.dy_q)
    else:
        body.reach_q16 = (0, 0)
    if getattr(unit, "pending_use", False):
        unit.pending_use = False
        if body.has_prop:
            body.grab_req = 1


# ---------------------------------------------------------------------------
# Gaits
# ---------------------------------------------------------------------------
def _carrying(body) -> bool:
    return int(body.arm_state[body.dials.grab_arm]) in GRAB_STATES


def _wanted_gait(body) -> int:
    g = body.goal_kind
    if g == GOAL_NONE:
        return GAIT_IDLE
    if g == GOAL_MOVE_DIR or _carrying(body):
        return GAIT_CRAWL
    return GAIT_SWIM


def _settle(body, a: int, gait: int) -> None:
    """Arm ``a`` leaves a task (a reach, a lost grab) into ``gait``."""
    d = body.dials
    _wave_off(body, a)
    _release(body, a)
    if gait == GAIT_IDLE:
        body.arm_stretch[a] = _scale(body, a)
        _set_state(body, a, ARM_RELAX, _jitter(body, d.limp_ticks, d.limp_jit))
    elif gait == GAIT_CRAWL:
        body.arm_stretch[a] = _scale(body, a)
        _set_state(body, a, ARM_RECOVER, _jitter(body, d.recover_ticks, d.recover_jit))
    elif gait == GAIT_SWIM:
        _set_state(body, a, ARM_SWIM)
    else:
        _set_state(body, a, ARM_LIMP)


def _enter_gait(body, gait: int) -> None:
    """Every arm no task holds moves into ``gait`` (arms in index order)."""
    d = body.dials
    for a in range(d.A):
        s = int(body.arm_state[a])
        if s in GRAB_STATES and gait != GAIT_LIMP:
            continue
        if s == ARM_REACH and gait in (GAIT_IDLE, GAIT_CRAWL):
            continue
        if gait == GAIT_LIMP:
            _wave_off(body, a)
            _release(body, a)
            _set_state(body, a, ARM_LIMP)
            continue
        if gait == GAIT_SWIM:
            _wave_off(body, a)
            _release(body, a)
            _set_state(body, a, ARM_SWIM)
            continue
        if gait == GAIT_IDLE:
            if s in (ARM_READY, ARM_LIMP):
                _set_state(body, a, ARM_STILL)
                _seat(body, a)
            elif s in (ARM_STANCE, ARM_RECOVER, ARM_SWIM, ARM_REACH):
                _settle(body, a, GAIT_IDLE)
        elif gait == GAIT_CRAWL:
            if s in (ARM_STILL, ARM_DECIDE, ARM_LIMP, ARM_SWIM):
                _wave_off(body, a)
                _release(body, a)
                _set_state(body, a, ARM_READY)
            elif s in (ARM_EXPLORE, ARM_TOUCH, ARM_RELAX, ARM_REACH):
                _settle(body, a, GAIT_CRAWL)
    if gait == GAIT_LIMP:
        _drop_all_grabs(body)
        body.grab_req = 0
    if gait == GAIT_SWIM:
        body.swim_ticks = 0
        body.swim_phase = SWIM_JET if body.goal_kind == GOAL_ESCAPE else SWIM_OPEN
    body.gait = gait


# ---------------------------------------------------------------------------
# Steering (P2) and the two lotteries
# ---------------------------------------------------------------------------
def _steer(body) -> tuple[int, int]:
    """The crawl's steering target (crawl_sim._steer_target): the wanted
    direction corrected by where the crown went over the last window --
    ``wanted + gain (wanted - recent)`` -- and the window's ring advanced by
    one sample (every comb tick, whatever the gait, like the prototype's
    per-tick log)."""
    d = body.dials
    bd = body.bufs.body[0, 0]
    bx, by = int(bd[AC.BS_X]), int(bd[AC.BS_Y])
    gx, gy = body.goal_q16
    target = (gx, gy)
    ring, n = body.steer_ring, int(body.steer_ticks)
    if n >= d.steer_k and d.steer_gain > 0 and (gx or gy):
        rx = K.shr_round_away(bx - int(ring[0, 0]), CF.CHAIN_SHIFT - 16)
        ry = K.shr_round_away(by - int(ring[0, 1]), CF.CHAIN_SHIFT - 16)
        dist = K.sqrt_q16(rx * rx + ry * ry)
        if dist >= d.steer_min:
            ux, uy = K.div_round_away(rx << 16, dist), K.div_round_away(ry << 16, dist)
            tx = gx + K.mul_shr_away(d.steer_gain, gx - ux, 16)
            ty = gy + K.mul_shr_away(d.steer_gain, gy - uy, 16)
            u = _unit_q16(tx, ty)
            if u is not None:
                target = u
    if n >= d.steer_k:
        ring[:-1] = ring[1:]
        ring[-1] = (bx, by)
    else:
        ring[n] = (bx, by)
        body.steer_ticks = n + 1
    return target


def _push_dir(body, a: int, th16: int) -> tuple[int, int]:
    """Arm ``a``'s push direction in the world (away from its root), Q16."""
    ang = th16 + body.dials.phi[a] + int(body.arm_aim[a])
    return -K.cos_q16(ang), -K.sin_q16(ang)


def _crawl_decide(body, target) -> None:
    """crawl_sim._decide: below max_stance, with probability p_recruit,
    recruit ONE ready arm by a seeded lottery weighted by alignment x
    hind-arm bias x neighbour bonus. No clock, no order."""
    d = body.dials
    st = body.arm_state
    if int((st == ARM_STANCE).sum()) >= d.max_stance:
        return
    if not _chance(body, d.p_recruit):
        return
    th16 = _th16(body)
    tx, ty = target
    weights = [0] * d.A
    for a in range(d.A):
        if int(st[a]) != ARM_READY or not _on_deck(body, a):
            continue
        dx, dy = _push_dir(body, a, th16)
        w = K.shr_round_away(dx * tx + dy * ty, 16) - d.s_min
        if w <= 0:
            continue
        w = K.mul_shr_away(w, ONE + K.mul_shr_away(d.back_bias, d.back[a], 16), 16)
        if int(st[(a - 1) % d.A]) == ARM_STANCE or int(st[(a + 1) % d.A]) == ARM_STANCE:
            w = K.mul_shr_away(w, ONE + d.nb_bonus, 16)
        weights[a] = max(0, w)
    total = sum(weights)
    if total <= 0:
        return
    r = _below(body, total)
    a = 0
    while r >= weights[a]:
        r -= weights[a]
        a += 1
    _set_state(body, a, ARM_STANCE, _jitter(body, d.push_ticks, d.push_jit))
    _anchor(body, a, _on_deck(body, a))


def _idle_decide(body) -> None:
    """idle_sim._decide: below max_active explorers, with probability
    p_start, release ONE still arm for a gesture, front arms favoured."""
    d = body.dials
    st = body.arm_state
    active = sum(1 for a in range(d.A) if int(st[a]) in IDLE_GESTURE)
    if active >= d.max_active or not _chance(body, d.p_start):
        return
    weights = [ONE + K.mul_shr_away(d.front_bias, d.front[a], 16) if int(st[a]) == ARM_STILL else 0
               for a in range(d.A)]
    total = sum(weights)
    if total <= 0:
        return
    r = _below(body, total)
    a = 0
    while r >= weights[a]:
        r -= weights[a]
        a += 1
    _release(body, a)
    _set_state(body, a, ARM_DECIDE, _jitter(body, d.decide_ticks, d.decide_jit))


# ---------------------------------------------------------------------------
# Reach (P1) and grab (P4) task starts
# ---------------------------------------------------------------------------
def _route_reach(body) -> None:
    d = body.dials
    st = body.arm_state
    reaching = [a for a in range(d.A) if int(st[a]) == ARM_REACH]
    want = (body.reach_q16 != (0, 0)) and body.gait in (GAIT_IDLE, GAIT_CRAWL)
    if want and not reaching:
        gx, gy = body.reach_q16
        th16 = _th16(body)
        best, best_cos = -1, None
        for a in range(d.A):
            if int(st[a]) in GRAB_STATES:
                continue
            ang = th16 + d.phi[a] + int(body.arm_aim[a])
            c = K.cos_q16(ang) * gx + K.sin_q16(ang) * gy
            if best_cos is None or c > best_cos:
                best, best_cos = a, c
        if best >= 0:
            _release(body, best)
            _set_state(body, best, ARM_REACH)
            _wave_on(body, best, d.reach_a, d.reach_tact)
    elif not want:
        for a in reaching:
            _settle(body, a, body.gait)


def _start_grab(body) -> None:
    d = body.dials
    a = d.grab_arm
    _release(body, a)
    _set_state(body, a, ARM_GRAB_REACH)
    _wave_on(body, a, d.grab_reach_a, d.grab_tact)


# ---------------------------------------------------------------------------
# The prop (grab) geometry
# ---------------------------------------------------------------------------
def _prop_xyz(body) -> tuple[int, int, int]:
    pb = body.bufs.body[0, 1]
    return int(pb[AC.BS_X]), int(pb[AC.BS_Y]), int(pb[AC.BS_Z])


def _touching(body, a: int) -> list:
    """The grab patch's nodes of arm ``a`` within the prop's radius + a
    sucker's reach of its centre (3-D), ascending."""
    d = body.dials
    px, py, pz = _prop_xyz(body)
    r2 = d.touch_r_q24 * d.touch_r_q24
    p = body.bufs.p[0, a]
    out = []
    for k in range(d.grab_k0, d.grab_k1 + 1):
        dx, dy, dz = int(p[k, 0]) - px, int(p[k, 1]) - py, int(p[k, 2]) - pz
        if dx * dx + dy * dy + dz * dz <= r2:
            out.append(k)
    return out


def _sealed_nodes(body, a: int) -> list:
    g = body.bufs.grabs[0]
    return sorted(int(r[AC.GR_NODE]) for r in g if int(r[AC.GR_ARM]) == a)


def _seal(body, a: int, nodes) -> None:
    """Rigid grabs of ``nodes`` to the prop in the prop's frame (grab_sim._seal),
    the grab slots kept sorted by (arm, node), empties last."""
    from simulation.creature_system import _rot
    b = body.bufs
    have = {(int(r[AC.GR_ARM]), int(r[AC.GR_NODE])) for r in b.grabs[0] if int(r[AC.GR_ARM]) >= 0}
    px, py, pz = _prop_xyz(body)
    th = int(b.body[0, 1, AC.BS_TH])
    rows = [tuple(int(v) for v in r) for r in b.grabs[0] if int(r[AC.GR_ARM]) >= 0]
    for k in nodes:
        if (a, k) in have:
            continue
        p = b.p[0, a, k]
        lx, ly = _rot(-th, int(p[0]) - px, int(p[1]) - py)
        rows.append((a, k, 1, lx, ly, int(p[2]) - pz))
    _write_grabs(body, rows)


def _write_grabs(body, rows) -> None:
    g = body.bufs.grabs
    rows = sorted(rows)
    G = g.shape[1]
    if len(rows) > G:
        raise RuntimeError("creature_primitives: more seals than grab slots")
    g[0, :, :] = 0
    g[0, :, AC.GR_ARM] = -1
    for i, r in enumerate(rows):
        g[0, i] = r


def _drop_all_grabs(body) -> None:
    if body.bufs.grabs.shape[1]:
        _write_grabs(body, [])


def _wrap_side(body, a: int, k: int) -> int:
    """+1 when the prop lies on the positive side of the arm at node k (the
    plane-B sense: x toward y), else -1 (grab_sim._wrap_side)."""
    n = body.dials.n
    p = body.bufs.p[0, a]
    dx = int(p[min(k + 1, n - 1), 0]) - int(p[max(k - 1, 0), 0])
    dy = int(p[min(k + 1, n - 1), 1]) - int(p[max(k - 1, 0), 1])
    px, py, _ = _prop_xyz(body)
    tx, ty = px - int(p[k, 0]), py - int(p[k, 1])
    return 1 if dx * ty - dy * tx > 0 else -1


def _aim_root_at_prop(body, a: int) -> None:
    """grab_sim._aim_root: the root's yaw in the body frame toward the prop,
    clipped to +-aim_max (stored as the arm's aim)."""
    from simulation.creature_system import _rot
    d = body.dials
    bd = body.bufs.body[0, 0]
    th = int(bd[AC.BS_TH])
    r0 = body.rest_root[a, 0]
    rx, ry = _rot(th, int(r0[0]), int(r0[1]))
    px, py, _ = _prop_xyz(body)
    lx, ly = _rot(-th, px - (int(bd[AC.BS_X]) + rx), py - (int(bd[AC.BS_Y]) + ry))
    while max(abs(lx), abs(ly)) >= (1 << 30):
        lx, ly = K.shr_round_away(lx, 1), K.shr_round_away(ly, 1)
    ang = _wrap(K.atan2_q16(ly, lx) - d.phi[a])
    body.arm_aim[a] = max(-d.aim_max, min(d.aim_max, ang))


def _lift_pitch(body) -> int:
    """The holding arm's root pitch: the lift's cosine ease from the start of
    the hold (grab_sim3d._drive_grab), Q16 rad."""
    d = body.dials
    s = int(body.arm_state[d.grab_arm])
    t = int(body.arm_ticks[d.grab_arm])
    if s == ARM_GRAB_HOLD:
        el = t
    elif s == ARM_CARRY:
        el = d.hold_ticks + t
    else:
        return 0
    return K.mul_shr_away(d.lift, _ease(el, d.lift_ticks), 16)


# ---------------------------------------------------------------------------
# One arm's drive (the per-tick inputs of its state)
# ---------------------------------------------------------------------------
def _drive_arm(body, a: int, act, rest, bend) -> None:
    d = body.dials
    s, t, L = int(body.arm_state[a]), int(body.arm_ticks[a]), int(body.arm_drawn[a])
    na = d.n_anc
    if s == ARM_STILL:
        if not body.bufs.anchored[0, a].any():
            _seat(body, a)                              # landed, or a seat popped
    elif s == ARM_STANCE:
        scale = _lerp(ONE, d.e_max, t, max(L, 1))
        rest[a, :na] = scale
        act[a, :na] = d.stance_act
        anchored = body.bufs.anchored[0, a, list(d.anchor_nodes)]
        if t >= L or not anchored.any():
            # release on saturation or when the patch popped (crawl_sim._release)
            _release(body, a)
            act[a, :] = 0
            body.arm_stretch[a] = scale
            _set_state(body, a, ARM_RECOVER, _jitter(body, d.recover_ticks, d.recover_jit))
    elif s == ARM_RECOVER:
        rest[a, :na] = _lerp(int(body.arm_stretch[a]), ONE, 2 * t, max(L, 1))
        if t >= L:
            rest[a, :] = ONE
            _set_state(body, a, ARM_READY)
    elif s == ARM_DECIDE:
        if t >= L:
            # the gesture starts (idle_sim: wave, stretch, aim -- drawn in that order)
            W = _jitter(body, d.wave_ticks, d.wave_jit)
            body.arm_stretch[a] = d.idle_stretch + (_below(body, d.idle_stretch_jit + 1)
                                                   if d.idle_stretch_jit > 0 else 0)
            body.arm_aim[a] = _below(body, 2 * d.idle_aim + 1) - d.idle_aim
            _set_state(body, a, ARM_EXPLORE, W)
    elif s == ARM_EXPLORE:
        if t == 1:
            # the wave starts the tick AFTER the root swings to its aim, as in
            # idle_sim (the aim lands on a passive arm, which bends at its
            # base; the same tick's wave would stiffen it first and swing the
            # whole arm)
            _wave_on(body, a, d.idle_act, L * d.n_sub)
        rest[a, :] = _lerp(ONE, int(body.arm_stretch[a]), t, max(L, 1))
        if t >= L:
            _set_state(body, a, ARM_TOUCH, _jitter(body, d.touch_ticks, d.touch_jit))
    elif s == ARM_TOUCH:
        rest[a, :] = int(body.arm_stretch[a])
        if t >= L:
            _wave_off(body, a)
            act[a, :] = 0
            _set_state(body, a, ARM_RELAX, _jitter(body, d.limp_ticks, d.limp_jit))
    elif s == ARM_RELAX:
        rest[a, :] = _lerp(int(body.arm_stretch[a]), ONE, 2 * t, max(L, 1))
        if t >= L:
            rest[a, :] = ONE
            _set_state(body, a, ARM_STILL)
            _seat(body, a)                              # stick down where it lies
    elif s == ARM_REACH:
        gx, gy = body.reach_q16
        arm_ang = _th16(body) + d.phi[a] + int(body.arm_aim[a])
        bend[a, 0, 1] = _wrap(K.atan2_q16(gy, gx) - arm_ang)
    elif s == ARM_SWIM:
        act[a, :] = d.swim_act
    elif s == ARM_GRAB_REACH:
        _aim_root_at_prop(body, a)
        scale = _lerp(ONE, d.grab_stretch, t, d.grab_reach_ticks)
        rest[a, :] = scale
        hit = _touching(body, a)
        if hit:
            _seal(body, a, hit)
            _wave_off(body, a)
            body.arm_stretch[a] = scale
            _set_state(body, a, ARM_GRAB_WRAP)
    elif s in SEALED_STATES:
        sealed = _sealed_nodes(body, a)
        if not sealed:
            return
        k = sealed[0]
        if s == ARM_GRAB_WRAP:
            body.grab_side = _wrap_side(body, a, k)
        side = int(body.grab_side)
        bend[a, max(0, k - 1):, 1] = side * d.wrap_turn
        act[a, :] = d.wrap_act
        st = int(body.arm_stretch[a])
        if s == ARM_GRAB_WRAP:
            rest[a, :k] = st
            _seal(body, a, _touching(body, a))
            if t >= d.wrap_ticks:
                _set_state(body, a, ARM_GRAB_HOLD)
        else:
            act[a, :k] = d.hold_act
            if s == ARM_GRAB_HOLD:
                rest[a, :k] = _lerp(st, d.pull_in, t, d.pull_in_ticks)
                if t >= d.hold_ticks:
                    _set_state(body, a, ARM_CARRY)
            else:
                rest[a, :k] = d.pull_in


# ---------------------------------------------------------------------------
# The swim (P3): the phase clock, the root angle, the jet
# ---------------------------------------------------------------------------
def _heading_dv(body, dv_q24: int) -> None:
    th16 = _th16(body)
    body.add_velocity_q24(K.mul_shr_away(dv_q24, K.cos_q16(th16), 16),
                          K.mul_shr_away(dv_q24, K.sin_q16(th16), 16))


def _drive_swim(body) -> None:
    """swim_sim.Swimmer._command on integers: the umbrella's root angle on one
    phase clock (open slowly, close fast: open_s / close_s), or the jet
    train (arms bundled toward bundled_deg, jet_dv spread evenly over
    jet_ticks along the heading, one jet per opening time)."""
    d = body.dials
    ph, t = int(body.swim_phase), int(body.swim_ticks)
    escape = body.goal_kind == GOAL_ESCAPE
    if escape and ph in (SWIM_OPEN, SWIM_CLOSE):
        ph, t = SWIM_JET, 1
    elif not escape and ph in (SWIM_JET, SWIM_COAST):
        ph, t = SWIM_OPEN, 1
    if ph == SWIM_OPEN:
        body.swim_phi = d.closed_rad + K.mul_shr_away(d.open_rad - d.closed_rad,
                                                      _ease(t, d.open_ticks), 16)
        if t >= d.open_ticks:
            ph, t = SWIM_CLOSE, 0
    elif ph == SWIM_CLOSE:
        body.swim_phi = d.open_rad + K.mul_shr_away(d.closed_rad - d.open_rad,
                                                    _ease(t, d.close_ticks), 16)
        if t >= d.close_ticks:
            ph, t = SWIM_OPEN, 0
    else:
        # the escape: a jet per opening time -- JET for jet_ticks (the arms
        # bundling), then COAST until the cycle's open_ticks are spent
        if ph == SWIM_COAST and t > d.open_ticks:
            ph, t = SWIM_JET, 1
        if ph == SWIM_JET:
            phi = int(body.swim_phi)
            body.swim_phi = phi + K.mul_shr_away(d.bundled_rad - phi, d.jet_relax, 16)
            if 1 <= t <= d.jet_ticks:
                _heading_dv(body, K.div_round_away(d.jet_dv_q24, d.jet_ticks))
            if t >= d.jet_ticks:
                ph = SWIM_COAST
    body.swim_phase, body.swim_ticks = ph, t


def _drive_carry_jets(body) -> None:
    """The carry's escape (P4's jets): one carry_jet_dv impulse along the
    heading per opening time while the escape is asked for."""
    d = body.dials
    t = int(body.swim_ticks)
    if t == 1:
        _heading_dv(body, d.carry_dv_q24)
    if t >= d.open_ticks:
        body.swim_ticks = 0


# ---------------------------------------------------------------------------
# The roots (aim, lift, the swimming ring)
# ---------------------------------------------------------------------------
def _write_roots(body) -> None:
    d = body.dials
    rl = body.bufs.root_local[0]
    pitch_arm, pitch = d.grab_arm, _lift_pitch(body)
    rig = int(body.swim_rig)
    seg = d.seg_q24
    for a in range(d.A):
        aim = int(body.arm_aim[a])
        p = pitch if a == pitch_arm else 0
        r0 = [int(v) for v in body.rest_root[a, 0]]
        if aim == 0 and p == 0:
            r1 = [int(v) for v in body.rest_root[a, 1]]
        else:
            yaw = d.phi[a] + aim
            cp, sp = K.cos_q16(p), K.sin_q16(p)
            r1 = [r0[0] + K.mul_shr_away(K.mul_shr_away(seg, K.cos_q16(yaw), 16), cp, 16),
                  r0[1] + K.mul_shr_away(K.mul_shr_away(seg, K.sin_q16(yaw), 16), cp, 16),
                  r0[2] + K.mul_shr_away(seg, sp, 16)]
        if rig > 0:
            # the swimming ring at the back of the head; the root direction
            # phi from the trailing axis toward the arm's place psi_a around
            # it. psi_a = phi_a - pi/2 keeps each arm on its own side (left
            # arms left, the front pair below, the back pair above): the
            # deck pose maps onto the cone mirror-symmetrically, so the rig's
            # blend turns the crown by no net torque (swim_sim3d's psi_a =
            # phi_a is the same eight places, relabelled -- it started IN
            # the cone and never blended).
            psi = d.phi[a] - (PI >> 1)
            cpsi, spsi = K.cos_q16(psi), K.sin_q16(psi)
            phi = int(body.swim_phi)
            cph, sph = K.cos_q16(phi), K.sin_q16(phi)
            s0 = [-d.rim_q24, K.mul_shr_away(d.ring_q24, cpsi, 16), K.mul_shr_away(d.ring_q24, spsi, 16)]
            s1 = [s0[0] - K.mul_shr_away(seg, cph, 16),
                  s0[1] + K.mul_shr_away(K.mul_shr_away(seg, sph, 16), cpsi, 16),
                  s0[2] + K.mul_shr_away(K.mul_shr_away(seg, sph, 16), spsi, 16)]
            r0 = [x + K.mul_shr_away(s - x, rig, 16) for x, s in zip(r0, s0)]
            r1 = [x + K.mul_shr_away(s - x, rig, 16) for x, s in zip(r1, s1)]
        rl[a, 0] = r0
        rl[a, 1] = r1


# ---------------------------------------------------------------------------
# The hub: one tick, before and after the solver call
# ---------------------------------------------------------------------------
def hub_step(body) -> None:
    """The hub's tick, before the solver call: the gait from the goal, the
    task starts (grab, reach), the decision comb (steering sample + the
    gait's lottery), the elapsed counters, every arm's drive, the swim's
    clock and jet, the roots, and the solver inputs written."""
    d = body.dials
    if body.gait != GAIT_LIMP:
        gait = _wanted_gait(body)
        if gait != body.gait:
            _enter_gait(body, gait)
        if body.grab_req and body.has_prop and not _carrying(body):
            _start_grab(body)
        _route_reach(body)
    body.decision_ticks += 1
    if body.decision_ticks >= d.comb:
        body.decision_ticks = 0
        target = _steer(body)
        if body.gait == GAIT_CRAWL:
            _crawl_decide(body, target)
        elif body.gait == GAIT_IDLE:
            _idle_decide(body)
    body.arm_ticks += 1
    act = np.zeros((d.A, d.n - 2), np.int32)
    rest = np.full((d.A, d.n - 1), ONE, np.int32)
    bend = np.zeros((d.A, d.n - 2, 2), np.int32)
    for a in range(d.A):
        _drive_arm(body, a, act, rest, bend)
    if body.gait == GAIT_SWIM:
        body.swim_ticks += 1
        _drive_swim(body)
        body.swim_rig = min(ONE, int(body.swim_rig) + d.rig_step)
    else:
        body.swim_rig = max(0, int(body.swim_rig) - d.rig_step)
        if body.goal_kind == GOAL_ESCAPE and _carrying(body):
            body.swim_ticks += 1
            _drive_carry_jets(body)
    _write_roots(body)
    b = body.bufs
    wave = b.arm_cfg[0, :, AC.AC_WAVE_ON] != 0
    b.act[0][~wave] = act[~wave]
    b.rest_scale[0] = rest
    b.bend_target[0] = bend
    body.stance = int((body.arm_state == ARM_STANCE).sum())
    swim = body.gait == GAIT_SWIM
    body.solver_row_node = body.row_node_swim if swim else body.row_node
    b.body_cfg[0, 0] = body.body_cfg_swim if swim else body.body_cfg_crawl


def after_solver(body) -> None:
    """After the solver call: a seal whose grab force exceeded the grab's
    grip limit lets go (counted in ``grab_pops``); a grab arm left with no
    seal settles back into the gait and the request is dropped."""
    d = body.dials
    b = body.bufs
    G = b.grabs.shape[1]
    if not G:
        return
    rows = [tuple(int(v) for v in b.grabs[0, g]) for g in range(G)]
    keep = []
    for g, r in enumerate(rows):
        if r[AC.GR_ARM] < 0:
            continue
        if int(b.grab_force[0, g]) > d.grab_grip_q8:
            body.grab_pops += 1
            continue
        keep.append(r)
    if len(keep) != sum(1 for r in rows if r[AC.GR_ARM] >= 0):
        _write_grabs(body, keep)
    a = d.grab_arm
    if int(body.arm_state[a]) in SEALED_STATES and not _sealed_nodes(body, a):
        body.grab_req = 0
        body.arm_aim[a] = 0
        _settle(body, a, body.gait)


__all__ = [
    "ARM_LIMP", "ARM_READY", "ARM_STANCE", "ARM_RECOVER", "ARM_STILL", "ARM_DECIDE",
    "ARM_EXPLORE", "ARM_TOUCH", "ARM_RELAX", "ARM_REACH", "ARM_SWIM", "ARM_GRAB_REACH",
    "ARM_GRAB_WRAP", "ARM_GRAB_HOLD", "ARM_CARRY", "ARM_STATE_NAMES", "GRAB_STATES",
    "GAIT_LIMP", "GAIT_IDLE", "GAIT_CRAWL", "GAIT_SWIM", "GAIT_NAMES",
    "GOAL_NONE", "GOAL_MOVE_DIR", "GOAL_CRUISE", "GOAL_ESCAPE", "SPEED_MODE_GOAL",
    "SWIM_OPEN", "SWIM_CLOSE", "SWIM_JET", "SWIM_COAST",
    "PrimDials", "route_intents", "hub_step", "after_solver",
]
