"""P5z scenarios: every simulation the 3-D gates read, as top-level functions
that return plain numpy data (so `gates3d.py` can run them in parallel
processes). Each scenario is seeded and deterministic. Scenarios marked TRACE
also return the solver's per-tick inputs and the node positions per tick, for
the integer reference's replay (`out/traces_p5z.npz`).

Author: Claude (octopus lead), 2026-10-30.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np

from arm_bank3d import DRAG_UNIFORM, World
from arm_chain import ArmChain, ChainParams, ReachCommand, hooked_pose
from arm_chain3d import Chain3D, ChainParams3D, planar_to_3d, rot_matrix
from crawl_sim import CrawlParams
from crawl_sim3d import make_octopus3d
from game_scale import mid_row
from grab_sim import CARRY
from grab_sim3d import make_grab_octopus3d
from idle_sim3d import make_idle_octopus3d
from swim_sim3d import make_swimmer3d

ROW = mid_row(write=False)
Z_REACH = 1.5            # m: the P1 arm's base height (mid water, clear of the deck)


def _timed(fn):
    def wrap(*a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        out["wall_s"] = time.perf_counter() - t0
        return out
    wrap.__name__ = fn.__name__
    return wrap


# ------------------------------------------------------------ trace recorder

class Rec:
    """Per tick: the solver inputs after the hubs wrote them, and the state after the step."""

    def __init__(self, world: World, sl: slice, bodies):
        self.sl, self.bodies = sl, np.asarray(bodies, dtype=np.int64)
        self.d = defaultdict(list)
        b = world.bank
        self.init = dict(p0=b.p[sl].copy(), body_x0=b.bx[self.bodies].copy(), body_th0=b.bth[self.bodies].copy(),
                         node_mass=b.m[sl].copy(), K_per=b.K_per[sl].copy(), K_tan=b.K_tan[sl].copy(),
                         mu=b.mu[sl].copy(), drag_mode=b.drag_mode[sl].copy(), drag_tau=b.drag_tau[sl].copy(),
                         drag_vref=b.drag_vref[sl].copy(), alive=b.alive[sl].copy(),
                         body_mass=b.bm[self.bodies].copy(), body_inertia=b.binertia[self.bodies].copy(),
                         body_radius=b.bradius[self.bodies].copy(), body_c_drag=b.bcd[self.bodies].copy(),
                         body_area=b.barea[self.bodies].copy(), body_rot_damp=b.brot_damp[self.bodies].copy(),
                         body_mu=b.bmu[self.bodies].copy(), body_drag_mode=b.bdrag_mode[self.bodies].copy(),
                         body_tau=b.btau[self.bodies].copy())
        world.recorders.append(self)

    def __call__(self, world: World, phase: str) -> None:
        b = world.bank
        if phase == "inputs":
            s = b.snapshot_inputs()
            for k in ("activation", "rest_scale", "bend_target", "anchored", "anchor_pos", "root_local", "grip"):
                self.d["in_" + k].append(s[k][self.sl])
            gb = s["grab_body"][self.sl]
            local = np.full_like(gb, -1)
            for i, bi in enumerate(self.bodies):
                local[gb == bi] = i
            self.d["in_grab_body"].append(local)
            self.d["in_grab_local"].append(s["grab_local"][self.sl])
            self.d["in_body_v"].append(s["bv"][self.bodies])
            self.d["in_body_omega"].append(s["bom"][self.bodies])
        else:
            self.d["p"].append(b.p[self.sl].copy())
            self.d["body_x"].append(b.bx[self.bodies].copy())
            self.d["body_theta"].append(b.bth[self.bodies].copy())

    def arrays(self) -> dict:
        out = {k: np.array(v) for k, v in self.d.items()}
        out.update(self.init)
        return out


def _params_record(world_or_chain) -> dict:
    prm = world_or_chain.prm
    g = world_or_chain.geom
    return dict(dt_tick=prm.dt_tick, n_sub=prm.n_sub, n_pass=prm.n_pass, k_bend=prm.k_bend,
                k_passive_frac=prm.k_passive_frac, k_dist=prm.k_dist, bend_damp_s=prm.bend_damp_s,
                c_per=prm.c_per, c_tan=prm.c_tan, g_eff=prm.g_eff, activation_table=prm.activation_table,
                arm_length_m=g.length_m, r_base_m=g.r_base_m, r_tip_m=g.r_tip_m, rho_arm=g.rho_arm,
                rho_water=g.rho_water)


# ------------------------------------------------------------ P1: one arm

def _reach_chain(table: bool, nodrag: bool = False) -> Chain3D:
    chain = ROW.chain if not nodrag else replace(ROW.chain, c_per=0.0, c_tan=0.0)
    return Chain3D(ROW.geom, replace(chain, activation_table=table))


def _hooked3d(z=Z_REACH):
    return planar_to_3d(hooked_pose(ROW.geom), origin=(0.0, 0.0, z))


@_timed
def p1_batch(table: bool) -> dict:
    """Every P1 run as one batch of pinned arms (independent: batching changes nothing):
    reach (hold), reach (fade), t_act sweep, a_ca sweep, the minimum-force map, the
    dead arm, the three failure looks. Gravity ON, base 1.5 m above the deck."""
    ch = _reach_chain(table)
    L = ROW.geom.length_m
    P0 = _hooked3d()
    groups = {}

    def add(name, n, **wave):
        sl = ch.add_arms(np.repeat(P0[None], n, 0), pinned=(0, 1))
        groups[name] = sl
        return sl

    sl = add("reach", 1)
    ch.set_waves(sl, 0.6, 0.9)
    sl = add("fade", 1)
    ch.set_waves(sl, 0.6, 0.9, hold=False)
    t_acts = [0.5, 0.7, 0.9, 1.2, 1.5]
    sl = add("tact", len(t_acts))
    for i, ta in enumerate(t_acts):
        ch.set_waves(slice(sl.start + i, sl.start + i + 1), 0.6, ta)
    a_cas = [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
    sl = add("amp", len(a_cas))
    for i, a in enumerate(a_cas):
        ch.set_waves(slice(sl.start + i, sl.start + i + 1), a, 0.9)
    ta_grid = np.array([0.1, 0.2, 0.3, 0.45, 0.6, 0.9, 1.2, 1.8, 2.5])
    a_grid = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0])
    sl = add("map", len(ta_grid) * len(a_grid))
    for ia, a in enumerate(a_grid):
        for it, ta in enumerate(ta_grid):
            i = sl.start + ia * len(ta_grid) + it
            ch.set_waves(slice(i, i + 1), a, ta)
    sl = add("dead", 1)
    omega = 3.0                     # rad/s whip about the base (the 2-D gate's), angular: unchanged by the law
    rel = P0 - P0[0]
    ch.v[sl.start] = omega * np.stack([-rel[:, 1], rel[:, 0], np.zeros(len(rel))], axis=1)
    ch.v[sl.start, :2] = 0.0
    looks = [(0.6, 0.9), (0.6, 0.1), (0.1, 0.9)]
    sl = add("looks", 3)
    for i, (a, ta) in enumerate(looks):
        ch.set_waves(slice(sl.start + i, sl.start + i + 1), a, ta)
    n_ticks = int(round(2.65 / ch.prm.dt_tick))
    P = [ch.p.copy()]
    ACT = [ch.activation.copy()]
    T = [0.0]
    for _ in range(n_ticks):
        ch.tick()
        P.append(ch.p.copy())
        ACT.append(ch.activation.copy())
        T.append(ch.time)
    out = dict(T=np.array(T), P=np.array(P), ACT=np.array(ACT), t_acts=np.array(t_acts), a_cas=np.array(a_cas),
               ta_grid=ta_grid, a_grid=a_grid, looks=np.array(looks), s_node=ch.s_node, L=L, table=table)
    out["groups"] = {k: (v.start, v.stop) for k, v in groups.items()}
    return out


@_timed
def p1_nodrag() -> dict:
    ch = _reach_chain(False, nodrag=True)
    sl = ch.add_arms(_hooked3d()[None], pinned=(0, 1))
    ch.set_waves(sl, 0.6, 0.9)
    P = [ch.p[0].copy()]
    for _ in range(int(round(1.6 / ch.prm.dt_tick))):
        ch.tick()
        P.append(ch.p[0].copy())
    return dict(P=np.array(P))


@_timed
def p1_equal() -> dict:
    """Gate 0: (a) the 3-D chain in a horizontal plane (g = 0) against the 2-D P1
    chain at the same mid-scale parameters; (b) the same reach in a TILTED plane
    (the pose and the root reference rotated together) rotated back."""
    g = ROW.geom
    c = ROW.chain
    prm2 = ChainParams(dt_tick=c.dt_tick, n_sub=c.n_sub, n_pass=c.n_pass, k_bend=c.k_bend,
                       k_passive_frac=c.k_passive_frac, k_dist=c.k_dist, bend_damp_s=c.bend_damp_s,
                       c_per=c.c_per, c_tan=c.c_tan, g_eff=0.0)
    p2 = hooked_pose(g)
    ch2 = ArmChain(g, prm2, p2)
    cmd = ReachCommand(a_ca=0.6, t_act=0.9)
    prm3 = replace(c, g_eff=0.0)
    ch3 = Chain3D(g, prm3)
    ch3.floor_on = False
    sl = ch3.add_arms(planar_to_3d(p2)[None], pinned=(0, 1))
    ch3.set_waves(sl, 0.6, 0.9)
    Rm = rot_matrix((1.0, 0.3, 0.0), np.radians(50.0)) @ rot_matrix((0, 0, 1), np.radians(20.0))
    ch3t = Chain3D(g, prm3)
    ch3t.floor_on = False
    slt = ch3t.add_arms((planar_to_3d(p2) @ Rm.T)[None], pinned=(0, 1), root_ref=Rm @ np.array([0.0, 0.0, 1.0]))
    ch3t.set_waves(slt, 0.6, 0.9)
    dmax, dtilt = 0.0, 0.0
    for _ in range(24):
        ch2.tick(cmd)
        ch3.tick()
        ch3t.tick()
        dmax = max(dmax, float(np.abs(ch3.p[0, :, :2] - ch2.p).max()), float(np.abs(ch3.p[0, :, 2]).max()))
        dtilt = max(dtilt, float(np.abs(ch3t.p[0] @ Rm - ch3.p[0]).max()))
    xs = np.linspace(-10, 10, 200001)
    from arm_chain3d import sigmoid_table, sigmoid_tanh
    tab_err = float(np.abs(sigmoid_table(xs) - sigmoid_tanh(xs)).max())
    return dict(dmax=dmax, dtilt=dtilt, tab_err=tab_err, L=g.length_m)


@_timed
def p1_trace() -> dict:
    """TRACE: the canonical reach (tanh), one arm, 1.6 s, gravity on."""
    ch = _reach_chain(False)
    sl = ch.add_arms(_hooked3d()[None], pinned=(0, 1))
    ch.set_waves(sl, 0.6, 0.9)
    P = [ch.p[0].copy()]
    for _ in range(int(round(1.6 / ch.prm.dt_tick))):
        ch.tick()
        P.append(ch.p[0].copy())
    tr = dict(p=np.array(P)[1:], p0=np.array(P)[0], pinned=np.array([0, 1]), wave_a=0.6, wave_tact=0.9,
              wave_t0=0.0, wave_w=ROW.geom.length_m / 17, wave_hold=True, root_ref=np.array([0.0, 0.0, 1.0]),
              mu=0.0, floor=True)
    tr.update(_params_record(ch))
    return dict(trace=tr)


# ------------------------------------------------------------ P2: crawl

def _crawl_log(o) -> dict:
    return dict(T=np.array(o.log_t), X=np.array(o.log_x), Z=np.array(o.log_z), theta=np.array(o.log_theta),
                S=np.array(o.log_stance), pred=np.array(o.log_pred), P=np.array(o.log_p),
                anchored=np.array(o.log_anchored), steps=np.array(o.steps, dtype=float).reshape(-1, 3),
                travel=o.travel.copy(), alive=o.bank.alive.copy(), radius=o.body.radius,
                popped_total=int(o.bank.popped.sum()))


@_timed
def p2_crawl(direction_deg: float, seed: int, seconds: float, dead=(), clocked=False, trace=False) -> dict:
    crawl = ROW.crawl
    if clocked:
        crawl = replace(crawl, push_jitter=0.0, recover_jitter=0.0, p_recruit=1.0)
    d = np.radians(direction_deg)
    o = make_octopus3d(ROW, direction=(np.cos(d), np.sin(d)), seed=seed, dead=dead, crawl=crawl)
    rec = Rec(o.world, o.bank.sl, [o.body.i]) if trace else None
    o.run(seconds)
    out = _crawl_log(o)
    if trace:
        out["trace"] = {**rec.arrays(), **_params_record(o.world.bank), "hub": "crawl", "seed": seed,
                        "direction_deg": direction_deg, "root_ref": np.array([0.0, 0.0, 1.0])}
    return out


# ------------------------------------------------------------ P3: swim

def _swim_log(s) -> dict:
    return dict(T=np.array(s.log_t), X=np.array(s.log_x), Z=np.array(s.log_z), phi=np.array(s.log_phi),
                P=np.array(s.log_p), phase=np.array(s.log_phase), radius=s.body.radius)


@_timed
def p3_swim(kind: str, trace: bool = False) -> dict:
    dt = ROW.chain.dt_tick
    if kind == "umbrella":
        s = make_swimmer3d(ROW)
        s.start_umbrella()
        s.run(10.0)
    elif kind == "linear":
        s = make_swimmer3d(ROW, linear=True)
        s.start_umbrella()
        s.run(10.0)
    elif kind == "jet":
        s = make_swimmer3d(ROW)
        rec = Rec(s.world, s.bank.sl, [s.body.i]) if trace else None
        s.start_jet()
        s.run(6.0)
    elif kind == "escape":
        s = make_swimmer3d(ROW)
        s.start_umbrella()
        jets = 0
        for _ in range(int(round(10.0 / dt))):
            s.world.tick()
            if s.mode == "close" and s.t_mode <= dt * 1.01 and s.jet_force == 0.0:
                h = np.array([np.cos(s.body.theta), np.sin(s.body.theta)])
                s.body.v += s.swim.jet_dv * 0.5 * h
                jets += 1
    elif kind == "train":
        s = make_swimmer3d(ROW)
        s.start_jet()
        nxt = 1.0
        for _ in range(int(round(10.0 / dt))):
            s.world.tick()
            if s.time >= nxt:
                s.start_jet()
                nxt += 1.0
    elif kind == "gif":
        s = make_swimmer3d(ROW)
        s.start_umbrella()
        s.run(4.2)
        s.start_jet()
        s.run(3.0)
    else:
        raise ValueError(kind)
    out = _swim_log(s)
    if kind == "escape":
        out["jets"] = jets
    if kind == "jet" and trace:
        out["trace"] = {**rec.arrays(), **_params_record(s.world.bank), "hub": "swim_jet",
                        "root_ref": np.array([0.0, 0.0, 1.0])}
    return out


# ------------------------------------------------------------ P4: grab + lift

@_timed
def p4_grab(trace: bool = True) -> dict:
    o = make_grab_octopus3d(ROW, seed=3)
    rec = Rec(o.world, o.bank.sl, [o.body.i, o.prop.i]) if trace else None
    o.run(0.5)
    o.start_reach()
    while o.phase != CARRY and o.time < 8.0:
        o.world.tick()
    t_carry = o.time
    t_hold = o.t_hold
    o.run(15.0)
    t_jets = []
    for _ in range(3):
        o.jet()
        t_jets.append(o.time)
        o.run(2.0)
    out = _crawl_log(o)
    a = o.gp.arm
    out.update(PR=np.array(o.log_prop), PR3=np.array(o.log_prop3), NG=np.array(o.log_ngrab),
               phase=np.array(o.log_phase), t_carry=t_carry, t_hold=t_hold, t_contact=o.t_grab,
               t_jets=np.array(t_jets), grab_s=np.array(o.grab_s),
               sealed_s=np.array(sorted(set(float(o.bank.s_node[k]) for (aa, k, _, _) in o.bank.grabs if aa == a))),
               arm=a, popped=int(o.bank.popped.sum()), prop_radius=o.prop.radius, L=o.geom.length_m)
    if trace:
        out["trace"] = {**rec.arrays(), **_params_record(o.world.bank), "hub": "grab", "seed": 3,
                        "root_ref": np.array([0.0, 0.0, 1.0])}
    return out


@_timed
def p4_free(seed: int = 3, seconds: float = 15.0) -> dict:
    o = make_octopus3d(ROW, direction=(1.0, 0.0), seed=seed)
    o.run(seconds)
    return _crawl_log(o)


# ------------------------------------------------------------ P5: idle

@_timed
def p5_idle(trace: bool = True) -> dict:
    o = make_idle_octopus3d(ROW, seed=5)
    rec = Rec(o.world, o.bank.sl, [o.body.i]) if trace else None
    o.run(60.0)
    out = _crawl_log(o)
    out.update(gestures=np.array(o.gestures, dtype=float).reshape(-1, 3), ex_final=np.array(o.ex),
               ex_log=np.array(o.log_ex), rest_final=o.bank.rest.copy())
    if trace:
        out["trace"] = {**rec.arrays(), **_params_record(o.world.bank), "hub": "idle", "seed": 5,
                        "root_ref": np.array([0.0, 0.0, 1.0])}
    return out


# ------------------------------------------------------------ the two new gates

@_timed
def n1_dead_arm(seconds: float = 20.0, trace: bool = True) -> dict:
    """A dead arm (no activation, no pins, no body) released 0.2 m above the deck,
    lying in Yekutieli's hooked pose, under g_eff, real drag, deck friction mu."""
    w = World(ROW.geom, ROW.chain)
    b = w.bank
    sl = b.add_arms(_hooked3d(z=0.2)[None], mu=ROW.crawl.mu_k)
    T, P, V = [], [], []
    for _ in range(int(round(seconds / ROW.chain.dt_tick))):
        w.tick()
        T.append(w.time)
        P.append(b.p[0].copy())
        V.append(b.v[0].copy())
    out = dict(T=np.array(T), P=np.array(P), V=np.array(V), p0=_hooked3d(z=0.2))
    if trace:
        out["trace"] = dict(p=np.array(P), p0=_hooked3d(z=0.2), pinned=np.array([], dtype=np.int64),
                            mu=ROW.crawl.mu_k, floor=True, root_ref=np.array([0.0, 0.0, 1.0]),
                            **_params_record(b))
    return out


@_timed
def n2_free_animal(drag: str = "uniform", seconds: float = 60.0, tau: float = 1.0, trace: bool = True) -> dict:
    """A free animal: no anchors (none at the start, none at the settle), the
    idle hub's waves on, NO deck friction (arms and body), and either the
    'uniform' symmetric drag (v <- v / (1 + dt / tau) on every node and the body:
    mass-proportional, so in exact arithmetic the horizontal momentum decays from
    zero and the centre of mass cannot move) or the real 'quadratic' drag."""
    o = make_idle_octopus3d(ROW, seed=5, seated=False, mu=0.0)
    b = o.world.bank
    if drag == "uniform":
        b.drag_mode[o.bank.sl] = DRAG_UNIFORM
        b.drag_tau[o.bank.sl] = tau
        b.bdrag_mode[o.body.i] = DRAG_UNIFORM
        b.btau[o.body.i] = tau
    rec = Rec(o.world, o.bank.sl, [o.body.i]) if trace else None
    M = b.m[o.bank.sl]
    mb = b.bm[o.body.i]

    def com():
        return (np.einsum("an,anj->j", M, b.p[o.bank.sl]) + mb * b.bx[o.body.i]) / (M.sum() + mb)

    C = [com()]
    o.world.recorders.append(lambda w, ph: C.append(com()) if ph == "state" else None)
    o.run(seconds)
    out = _crawl_log(o)
    out.update(COM=np.array(C), gestures=np.array(o.gestures, dtype=float).reshape(-1, 3), drag=drag, tau=tau)
    if trace:
        out["trace"] = {**rec.arrays(), **_params_record(b), "hub": "idle_free_" + drag, "seed": 5,
                        "root_ref": np.array([0.0, 0.0, 1.0])}
    return out


# ------------------------------------------------------------ the job list

JOBS = {
    "p1_tanh": (p1_batch, (False,)),
    "p1_table": (p1_batch, (True,)),
    "p1_nodrag": (p1_nodrag, ()),
    "p1_equal": (p1_equal, ()),
    "p1_trace": (p1_trace, ()),
    "p2_d0": (p2_crawl, (0.0, 1, 25.0), {"trace": True}),
    "p2_d70": (p2_crawl, (70.0, 1, 25.0)),
    "p2_d135": (p2_crawl, (135.0, 1, 25.0)),
    "p2_dm90": (p2_crawl, (-90.0, 1, 25.0)),
    "p2_long": (p2_crawl, (0.0, 7, 40.0)),
    "p2_clock": (p2_crawl, (0.0, 7, 40.0), {"clocked": True}),
    "p2_dead1": (p2_crawl, (0.0, 1, 25.0), {"dead": (3,)}),
    "p2_dead2": (p2_crawl, (0.0, 1, 25.0), {"dead": (3, 5)}),
    "p3_umbrella": (p3_swim, ("umbrella",)),
    "p3_linear": (p3_swim, ("linear",)),
    "p3_jet": (p3_swim, ("jet",), {"trace": True}),
    "p3_escape": (p3_swim, ("escape",)),
    "p3_train": (p3_swim, ("train",)),
    "p3_gif": (p3_swim, ("gif",)),
    "p4_grab": (p4_grab, ()),
    "p4_free": (p4_free, ()),
    "p5_idle": (p5_idle, ()),
    "n1_dead": (n1_dead_arm, ()),
    "n2_free": (n2_free_animal, ("uniform",)),
    "n2_free_quad": (n2_free_animal, ("quadratic",), {"trace": False}),
}


def run_job(name: str) -> tuple[str, dict]:
    spec = JOBS[name]
    fn, args = spec[0], spec[1]
    kw = spec[2] if len(spec) > 2 else {}
    return name, fn(*args, **kw)
