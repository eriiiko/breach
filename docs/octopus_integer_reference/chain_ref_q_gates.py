"""GATES for the arm chain's integer reference (`chain_ref_q.py`), #106 P6b + P6c.

    C:/Users/steen/anaconda3/python.exe docs/octopus_integer_reference/chain_ref_q_gates.py [--fast] [gate ...]

Eight gates (the P6b row of docs/octopus_p6_sim_port_design_2026-10-30.md
section 8, plus P6c's static-topology contacts), each printing its MEASURED
numbers and PASS / FAIL; exit 1 on any FAIL; the full run writes
`report_p6c.md` beside this file (`report_p6b.md` is P6b's run, kept as is).
Since P6c, G3 and G4 run WITH an open topology (contacts enabled, nothing
touched): the walls must be inert where there are none.

  G1 P1's seven on ints  the mid row's reach batch through the reference,
                         dequantized, measured by the SAME code as the P5z
                         gates (gates3d.profile_shape / _trace2d, arm_chain's
                         bend_kinematics / mean_local_angle_deg): a1-a4, b, c, d
  G2 dead arm sinks      released 0.2 m above the deck: touches, rests
                         (|v| < 1 mm/s over the last 5 s), never below z = 0
  G3 no slide            the free animal (P5z N2's recorded idle inputs, the
                         uniform drag, mu 0): centre of mass <= 1 mm in 60 s;
                         PAIRED with the same run on FLOOR rounding, which must
                         slide (so the gate can fail)
  G4 float vs int        P5z's reach / dead-arm / crawl traces replayed; the
                         reach gated (node RMS <= 1 cm), the others reported
  G5 determinism         two runs, identical bytes; CHECK on == CHECK off
  G6 headroom            CHECK mode at the 1.80 / 2.52 / 3.39 m arms: every
                         int64 site below 2^63, int32 state below 2^31
  G7 cost                kit calls per arm-substep, weighted as P1's gate d --
                         REPORTED, not gated (cost is not a design input, Erik's
                         ruling 1; P5z's float d said the same; P6e measures)
  G8 walls (P6c)         the static-topology contacts: inert on an open map
                         (state bytes identical with and without); an arm
                         thrown at a wall stops at the face, never penetrates
                         and never slides along it with no tangential velocity;
                         an oblique throw (gravity, floor, friction) and a
                         corner throw never end a tick inside a blocked tile; a
                         body pushed into a wall or another unit's footprint
                         stays out; the new int64 sites in CHECK mode

`--fast` runs G1's reach (a1-a3), G5, G6 and G8 on short runs -- what
tests/test_octopus_chain_reference.py runs. G3 and G4 need the P5z traces
(`prototypes/octopus_motion/gates3d.py` writes out/traces_p5z.npz, gitignored).
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import tomllib  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTO = ROOT / "prototypes" / "octopus_motion"
for _p in (HERE, PROTO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import chain_ref_q as R  # noqa: E402
from simulation import creature_fixed as CF  # noqa: E402
from simulation.creatures import build_creature, get_creature, scale_row_by_law  # noqa: E402

DT = 1.0 / 24.0
SIZES = (1.80, 2.52, 3.39)          # the art table's arms (CONTROLS.md, #112)
T_ACTS = (0.5, 0.7, 0.9, 1.2, 1.5)  # P1's a4 sweep
A_CAS = (0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)   # P1's b sweep
REACH_RMS_TOL_M = 0.01              # G4: 1 cm node RMS over the reach
TILE_M = 1.0 / 3.0                  # the game's tile (Erik's ruling: tune at 0.333 m)


# ====================================================================== scenario helpers

def creature(arm_length_m: float | None = None):
    """The shipped row, or the shipped row at another arm length by P5z's law
    (through the door: a size the door refuses cannot be run)."""
    if arm_length_m is None or abs(arm_length_m - 2.52) < 1e-9:
        return get_creature("octopus")
    with open(ROOT / "config.toml", "rb") as f:
        raw = tomllib.load(f)["creatures"]["octopus"]
    return build_creature("octopus", scale_row_by_law(raw, arm_length_m), 24)


def geometry(cdef):
    """The prototype's ArmGeometry for a row (MEASUREMENT side only)."""
    from arm_chain import ArmGeometry
    b = cdef.body
    return ArmGeometry(length_m=b.arm_length_m, r_base_m=b.radius_base_m, r_tip_m=b.radius_tip_m,
                       rho_arm=b.tissue_density, rho_water=b.water_density)


def hooked3d(cdef, z: float):
    from arm_chain import hooked_pose
    from arm_chain3d import planar_to_3d
    return planar_to_3d(hooked_pose(geometry(cdef)), origin=(0.0, 0.0, z))


def open_topology(half_m: float = 48.0):
    """An ALL-OPEN tile grid covering +-half_m around the origin: the contacts
    are enabled and nothing is ever touched (P6c's inertness condition)."""
    n = int(round(2.0 * half_m / TILE_M))
    return R.topology(np.zeros((n, n), dtype=bool), TILE_M, origin_m=(-half_m, -half_m))


#: The walls scenarios' room: 30 x 30 tiles (10 m) at the game's tile; tile
#: columns 20.. blocked (a flat wall at x = 20 tiles = 6.67 m), a 2 x 2 pillar at
#: tiles (8..9, 20..21); outside the grid is blocked too.
WALL_X_TILE = 20
PILLAR = ((8, 20), (9, 20), (8, 21), (9, 21))
FOOTPRINT = (12, 3, 15, 6)            # another unit's 3 x 3 footprint (tiles, half-open)


def wall_grid() -> np.ndarray:
    g = np.zeros((30, 30), dtype=bool)
    g[:, WALL_X_TILE:] = True
    for ix, iy in PILLAR:
        g[iy, ix] = True
    return g


def wall_topology(unit_rects=(FOOTPRINT,)):
    return R.topology(wall_grid(), TILE_M, unit_rects=unit_rects)


def inside_blocked(w, P_counts) -> bool:
    t = w.topo
    return t.blocked(t.tile(P_counts[0], 0), t.tile(P_counts[1], 1))


class Scenario:
    """A World plus its per-tick INPUTS (what a hub would write before each
    tick): `inputs(world, t)` or None. The tol-0 test drives the C++ solver
    with exactly these (tests/test_octopus_arm_chain_solver.py)."""

    def __init__(self, world, inputs=None, ticks=24, name=""):
        self.world, self.inputs, self.ticks, self.name = world, inputs, ticks, name

    def step(self, t: int) -> None:
        if self.inputs is not None:
            self.inputs(self.world, t)
        self.world.tick()


def scenario_reach(cdef, ticks=None, topo=None) -> Scenario:
    """P1's reach: pinned 0, 1; the hooked pose 1.5 L / 2.52 m up; the wave."""
    row = R.chain_row(cdef)
    w = R.World(row, topo=topo)
    w.add_arm(hooked3d(cdef, 1.5 * cdef.body.arm_length_m / 2.52), pinned=(0, 1))
    w.set_wave(0, 0.6, 0.9)
    return Scenario(w, None, ticks or int(round(1.6 / DT)), "reach")


def scenario_dead(cdef, ticks=None, topo=None) -> Scenario:
    """G2's dead arm: the hooked pose 0.2 m above the deck, no pins, the row's
    friction, real drag and g_eff -- it falls, lands, unfurls and slides."""
    row = R.chain_row(cdef)
    w = R.World(row, topo=topo)
    w.add_arm(hooked3d(cdef, 0.2), mu=cdef.tissue.mu_floor)
    return Scenario(w, None, ticks or int(round(6.0 / DT)), "dead")


def _straight(cdef, x0, y0, z0, ang):
    L = cdef.body.arm_length_m
    n = cdef.body.nodes_per_arm
    seg = R.CF.chain_to_m(R.CF.chain_from_m(L / (n - 1)))
    return np.array([[x0 + i * seg * np.cos(ang), y0 + i * seg * np.sin(ang), z0] for i in range(n)])


def scenario_wall_thrown(cdef, ticks=None) -> Scenario:
    """A straight arm laid along y, parallel to the wall, thrown at it at
    1 m/s: no gravity, no floor, no drag, mu 0 -- nothing acts along the wall."""
    from dataclasses import replace
    row = replace(R.chain_row(cdef), g_q24=0, g_dv_q24=0)
    w = R.World(row, floor_on=False, topo=wall_topology(()))
    P0 = _straight(cdef, 6.0, 2.0, 0.5, np.pi / 2)
    w.add_arm(P0, v0=[[1.0, 0.0, 0.0]] * len(P0), drag_mode=R.DRAG_UNIFORM, tau=1e9)
    w.arms[0].u_q30 = 0
    return Scenario(w, None, ticks or int(round(1.5 / DT)), "wall_thrown")


def scenario_wall_oblique(cdef, ticks=None) -> Scenario:
    """The hooked arm thrown obliquely at the wall (2.0, 0.5, 0) m/s from
    0.3 m up: gravity, the floor, the row's friction, real drag."""
    row = R.chain_row(cdef)
    w = R.World(row, topo=wall_topology(()))
    P0 = hooked3d(cdef, 0.3) + np.array([5.8, 3.0, 0.0])
    w.add_arm(P0, v0=[[2.0, 0.5, 0.0]] * len(P0), mu=cdef.tissue.mu_floor)
    return Scenario(w, None, ticks or int(round(3.0 / DT)), "wall_oblique")


def scenario_wall_corner(cdef, ticks=None) -> Scenario:
    """A straight arm thrown tip-first along its own axis (45 deg, 1.5 m/s) at
    the pillar's corner (no gravity, no floor, mu 0.3 on the faces): the -x /
    -y faces and their tie-break."""
    from dataclasses import replace
    row = replace(R.chain_row(cdef), g_q24=0, g_dv_q24=0)
    w = R.World(row, floor_on=False, topo=wall_topology(()))
    L = cdef.body.arm_length_m
    ang = np.pi / 4
    tip = np.array([PILLAR[0][0] * TILE_M - 0.3, PILLAR[0][1] * TILE_M - 0.3])
    base = tip - L * np.array([np.cos(ang), np.sin(ang)])
    P0 = _straight(cdef, base[0], base[1], 0.5, ang)
    v = 1.5 / np.sqrt(2.0)
    w.add_arm(P0, v0=[[v, v, 0.0]] * len(P0), mu=0.3)
    return Scenario(w, None, ticks or int(round(2.5 / DT)), "wall_corner")


def _body_kw(cdef):
    b = cdef.body
    return dict(mass=b.body_mass_kg, radius=b.body_radius_m, c_drag=b.body_c_drag,
                area=b.body_crawl_area_m2, rot_damp=b.body_rot_damp, mu=cdef.tissue.mu_floor,
                coll_radius=b.body_radius_m)


def scenario_body_wall(cdef, ticks=None) -> Scenario:
    """The crown alone (the row's body, its collision disc) pushed into the
    wall at 2 m/s and, beside it, a second crown pushed into another unit's
    footprint at 2 m/s (-y): gravity, floor, real drag."""
    row = R.chain_row(cdef)
    w = R.World(row, topo=wall_topology())
    w.add_body((5.9, 5.0, 0.0), **_body_kw(cdef))
    w.bodies[0].v = R._q24v((2.0, 0.0, 0.0))
    w.add_body((4.5, 2.9, 0.0), **_body_kw(cdef))
    w.bodies[1].v = R._q24v((0.0, -2.0, 0.0))
    return Scenario(w, None, ticks or int(round(2.0 / DT)), "body_wall")


def animal_room(cdef):
    """The animal's room, sized by the row: an open floor ringed by one tile of
    wall, the wall a quarter metre beyond a resting arm's tip (a stretched or
    swinging arm reaches it), the crown centred; another unit's 3 x 3 footprint
    overlapping the crown's disc by 2 cm on its -y side. Returns (topology,
    crown centre in metres)."""
    import math
    b = cdef.body
    reach = 0.6 * b.body_radius_m + b.arm_length_m + 0.25
    n = int(math.ceil(2.0 * reach / TILE_M)) + 2
    g = np.zeros((n, n), dtype=bool)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = True
    c_tile = n // 2
    cx = c_tile * TILE_M
    fy1 = c_tile - int(math.ceil(b.body_radius_m / TILE_M))          # the footprint's top edge, tiles
    cy = fy1 * TILE_M + b.body_radius_m - 0.02
    rect = (c_tile - 1, fy1 - 3, c_tile + 2, fy1)
    return R.topology(g, TILE_M, unit_rects=(rect,)), (cx, cy)


def scenario_animal(cdef, ticks=None, topo="room") -> Scenario:
    """A synthetic crawl-like animal exercising every path at once: the crown
    (its disc, the row's drag, rot damping, friction) with eight arms attached
    at their roots, lying straight 0.25 m above the floor, hub-like inputs every
    tick (activation, rest scale, bend targets), two stiffening waves (one
    released behind its front), anchors set, released and re-set (grip limits
    on two arms: pops), a 50 kg prop grabbed by arm 2's tip, the room's walls
    and another unit's footprint touching the crown (`animal_room`)."""
    import math
    b = cdef.body
    row = R.chain_row(cdef)
    room, (cx, cy) = animal_room(cdef)
    w = R.World(row, topo=room if topo == "room" else topo)
    z0 = 0.25
    w.add_body((cx, cy, z0), **_body_kw(cdef))
    n = b.nodes_per_arm
    seg = R.CF.chain_to_m(R.CF.chain_from_m(b.arm_length_m / (n - 1)))
    r0 = 0.6 * b.body_radius_m
    tips = []
    for a, deg in enumerate(b.arm_angles_deg):
        phi = math.radians(deg)
        c, s_ = math.cos(phi), math.sin(phi)
        P0 = np.array([[cx + (r0 + i * seg) * c, cy + (r0 + i * seg) * s_, z0] for i in range(n)])
        local = [[(r0 + k * seg) * c, (r0 + k * seg) * s_, 0.0] for k in (0, 1)]
        grip = 150.0 if a in (0, 4) else None
        w.add_arm(P0, mu=cdef.tissue.mu_floor, body=0, root_local=local, grip=grip)
        tips.append(P0[-1])
    gl = (0.3, 0.0, 0.0)
    w.add_body(tuple(tips[2] - np.array(gl)), mass=50.0, radius=0.3, c_drag=1.0, area=0.18,
               rot_damp=2.0, mu=0.3, coll_radius=0.3)
    w.grabs = [(2, n - 1, 1, R._q24v(gl))]
    w.set_wave(5, 0.6, 0.9, t_start_s=0.1)
    w.set_wave(6, 0.5, 0.7, t_start_s=0.4, hold=False)
    wave_arms = (5, 6)

    def inputs(world, t):
        for a in range(len(world.arms)):
            if a not in wave_arms:
                world.set_activation(a, [max(0.0, min(1.0, 0.3 + 0.3 * math.sin(0.2 * t + a + 0.1 * j)))
                                         for j in range(n - 2)])
            world.set_rest_scale(a, [1.0 + 0.08 * math.sin(0.15 * t + a) for _ in range(n - 1)])
            world.set_bend_target(a, [(0.05 * math.sin(0.1 * t + a + 0.2 * j),
                                       0.02 * math.cos(0.13 * t + a + 0.3 * j)) for j in range(n - 2)])
        if t == 6:
            for a in (0, 1, 3, 4):
                world.anchor(a, range(12, 17))
        if t == 30:
            world.release(1)
        if t == 40:
            world.anchor(1, range(10, 15))

    return Scenario(w, inputs, ticks or int(round(10.0 / DT)), "animal")


SCENARIOS = {"reach": scenario_reach, "dead": scenario_dead, "wall_thrown": scenario_wall_thrown,
             "wall_oblique": scenario_wall_oblique, "wall_corner": scenario_wall_corner,
             "body_wall": scenario_body_wall, "animal": scenario_animal}


def run_p1_arm(spec) -> dict:
    """One P1 arm (pinned 0, 1; the hooked pose 1.5 L/2.52 m above the deck;
    gravity on; quadratic drag): kind 'wave' (a_ca, t_act) or 'dead' (the 3 rad/s
    whip about the base). Returns the dequantized positions per tick, index 0 =
    the initial pose."""
    kind, a_ca, t_act, n_ticks, L, check = spec
    cdef = creature(L)
    row = R.chain_row(cdef)
    R.CHECK = bool(check)
    R.reset_bits()
    w = R.World(row)
    P0 = hooked3d(cdef, 1.5 * cdef.body.arm_length_m / 2.52)
    v0 = None
    if kind == "dead":
        rel = P0 - P0[0]
        v0 = 3.0 * np.stack([-rel[:, 1], rel[:, 0], np.zeros(len(rel))], axis=1)
        v0[:2] = 0.0
    ai = w.add_arm(P0, pinned=(0, 1), v0=v0)
    if kind == "wave":
        w.set_wave(ai, a_ca, t_act)
    P = [R.positions_m(w)[0]]
    for _ in range(n_ticks):
        w.tick()
        P.append(R.positions_m(w)[0])
    out = dict(P=np.array(P), state=w.state_bytes(), bits=R.bits_report())
    R.CHECK = False
    return out


def _com(row, w) -> np.ndarray:
    """The animal's centre of mass, dequantized (MEASUREMENT)."""
    m = np.array(row.m_q16, dtype=np.float64) / 65536.0
    P = R.positions_m(w)
    tot = m.sum() * P.shape[0]
    acc = np.einsum("n,anj->j", m, P)
    for b in w.bodies:
        mb = (1 << 48) / b.w_q32 / 65536.0
        acc = acc + mb * np.array(b.x, dtype=np.float64) / CF.CHAIN_ONE
        tot += mb
    return acc / tot


def run_free_animal(spec) -> dict:
    """G3: P5z's free animal (N2) replayed: the idle hub's recorded inputs, the
    uniform drag, mu 0. `biased` swaps every symmetric rounding of the
    reference for a FLOOR (>> and //), the variant the gate must catch."""
    seconds, biased = spec
    orig = (R.sra, R.dra, R.msa)
    if biased:
        R.sra = lambda x, s: x >> s
        R.dra = lambda n, d: n // d
        R.msa = lambda a, b, s: (a * b) >> s
    try:
        return _free_animal(seconds)
    finally:
        R.sra, R.dra, R.msa = orig


def _free_animal(seconds: float) -> dict:
    cdef = creature()
    row = R.chain_row(cdef)
    tr = R.load_traces()["free_animal"]
    C = []
    bx = []

    def on_tick(w, t):
        C.append(_com(row, w))
        bx.append([c / CF.CHAIN_ONE for c in w.bodies[0].x])

    out = R.replay(row, tr, ticks=int(round(seconds / DT)), record_every=1 << 30, on_tick=on_tick,
                   topo=open_topology())
    w = out["world"]
    C0 = _com_initial(row, tr)
    return dict(C=np.array(C), C0=C0, BX=np.array(bx), params=R.check_trace_params(row, cdef, tr),
                state=w.state_bytes())


def _com_initial(row, tr) -> np.ndarray:
    m = np.array(row.m_q16, dtype=np.float64) / 65536.0
    P = CF.chain_to_m(CF.chain_from_m_array(tr["p0"]))
    mb = float(CF.q16_from_kg(float(tr["body_mass"][0]))) / 65536.0
    acc = np.einsum("n,anj->j", m, P) + mb * CF.chain_to_m(CF.chain_from_m_array(tr["body_x0"][0]))
    return acc / (m.sum() * P.shape[0] + mb)


def run_replay(name: str) -> dict:
    """G4: replay one P5z trace through the reference; the deviation per tick."""
    cdef = creature()
    row = R.chain_row(cdef)
    tr = R.load_traces()[name]
    out = R.replay(row, tr, topo=open_topology())
    P, F = out["P"], tr["p"]
    if F.ndim == 3:
        P = P[:, 0]
    d = np.linalg.norm(P - F[:len(P)], axis=-1)
    res = dict(name=name, ticks=len(P), max=float(d.max()), rms=float(np.sqrt((d ** 2).mean())),
               per_tick_max=d.reshape(len(P), -1).max(axis=1), params=R.check_trace_params(row, cdef, tr))
    if "body_x" in tr:
        res["body_dev"] = np.linalg.norm(out["BX"][:, 0] - tr["body_x"][:len(P), 0], axis=-1)
    if name == "reach":
        res["arith"] = _reach_arith_only(P, tr, row)
    if name == "dead_arm":
        res["cent_int"] = P[-1, :, :2].mean(0)
        res["cent_float"] = F[len(P) - 1, :, :2].mean(0)
    return res


def _reach_arith_only(P_int, tr, row) -> dict:
    """The float reach re-run with t_act on the int clock's grid (whole
    substeps): the deviation that is the ARITHMETIC's, with the clock's
    quantization of the input removed."""
    from dataclasses import replace
    import scenarios3d as S
    from arm_chain3d import Chain3D
    tact_sub = CF._round_half_away(float(tr["wave_tact"]) * row.inv_dt)
    ch = Chain3D(S.ROW.geom, replace(S.ROW.chain, activation_table=True))
    sl = ch.add_arms(np.asarray(tr["p0"])[None], pinned=(0, 1))
    ch.set_waves(sl, float(tr["wave_a"]), tact_sub / row.inv_dt)
    F = []
    for _ in range(len(P_int)):
        ch.tick()
        F.append(ch.p[0].copy())
    d = np.linalg.norm(P_int - np.array(F), axis=-1)
    return dict(max=float(d.max()), rms=float(np.sqrt((d ** 2).mean())), tact_sub=tact_sub)


def run_dead_direct(seconds: float) -> dict:
    """G2: the dead arm built from the row (no trace): the hooked pose 0.2 m
    above the deck, no pins, no body, the row's mu_floor, g_eff, real drag."""
    cdef = creature()
    row = R.chain_row(cdef)
    w = R.World(row)
    w.add_arm(hooked3d(cdef, 0.2), mu=cdef.tissue.mu_floor)
    zmin_counts, Z, V = [], [], []
    for _ in range(int(round(seconds / DT))):
        w.tick()
        zmin_counts.append(min(q[2] for q in w.arms[0].p))
        Z.append(R.positions_m(w)[0][:, 2].copy())
        V.append(np.linalg.norm(R.velocities_m(w)[0], axis=1))
    return dict(zmin_counts=np.array(zmin_counts), Z=np.array(Z), V=np.array(V),
                cent=R.positions_m(w)[0][:, :2].mean(0), state=w.state_bytes())


# ====================================================================== G1: P1's seven

def g1_p1(fast: bool = False):
    from arm_chain import bend_kinematics, mean_local_angle_deg, segment_angles
    from gates3d import _trace2d, profile_shape
    cdef = creature()
    L = cdef.body.arm_length_m
    s_node = np.concatenate([[0.0], np.cumsum(np.full(cdef.body.nodes_per_arm - 1,
                                                      L / (cdef.body.nodes_per_arm - 1)))])
    n16 = int(round(1.6 / DT))
    specs = {"reach": ("wave", 0.6, 0.9, n16, None, False)}
    if not fast:
        for ta in T_ACTS:
            specs[f"tact{ta}"] = ("wave", 0.6, ta, int(round((ta + 0.8) / DT)), None, False)
        for a in A_CAS:
            specs[f"amp{a}"] = ("wave", a, 0.9, int(round(1.05 / DT)), None, False)
        specs["dead"] = ("dead", 0.0, 1.0, int(round(2.0 / DT)), None, False)
    runs = {k: run_p1_arm(v) for k, v in specs.items()}
    T = np.arange(0, 200) * DT
    lines, oks = [], []

    def gate(name, ok, note):
        oks.append(bool(ok))
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {note}")

    P = runs["reach"]["P"]
    tr0 = _trace2d(P, T, n16)
    t, s_b, pos_b, vel_b, amp_b = bend_kinematics(tr0, s_node)
    shape = profile_shape(t, vel_b)
    gate("a1 bell-shaped bend velocity",
         shape["single_peak"] and shape["rise"] >= 1.4 and shape["t_peak_frac"] >= 0.3,
         f"single peak {shape['single_peak']}, peak/early {shape['rise']:.2f}, peak at {shape['t_peak_frac']:.0%} "
         f"of the tracked time, the bend dissolves at t = {shape['t_end']:.2f} s; peak {shape.get('peak', 0):.2f} m/s; "
         f"sag to z = {P[:, :, 2].min():.3f} m")

    def straight_behind(Pk, sb):
        k = int(np.searchsorted(s_node, sb)) - 2
        return mean_local_angle_deg(Pk, upto_node=k) if k >= 2 else np.nan

    ok_idx = [i for i in range(len(t)) if np.isfinite(s_b[i]) and s_b[i] > 0.4 * L]
    behind = np.array([straight_behind(tr0.p[i], s_b[i]) for i in ok_idx])
    gate("a2 straight arm behind the bend", np.nanmax(behind) <= 3.0,
         f"mean local angle >= 2 nodes behind the bend, once past 0.4 L: max {np.nanmax(behind):.2f} deg "
         f"(limit 3), final whole-arm {mean_local_angle_deg(tr0.p[-1]):.3f} deg")

    def distal_turn(Pk, n_last=5):
        return float(np.degrees(segment_angles(Pk)[-n_last:].sum()))

    end_idx = [i for i in range(len(t)) if (np.isfinite(s_b[i]) and s_b[i] > 0.8 * L)
               or (not np.isfinite(s_b[i]) and t[i] > 0.3)]
    tip_y_min = min(tr0.p[i][-1][1] for i in end_idx)
    dturn = min(distal_turn(tr0.p[i]) for i in end_idx)
    gate("a3 tip never overshoots", tip_y_min >= -0.05 * L and dturn >= -10.0,
         f"once the bend has passed 0.8 L: min tip y {tip_y_min * 1e3:.0f} mm (limit {-0.05 * L * 1e3:.0f}), "
         f"min distal turn {dturn:.1f} deg (limit -10)")
    data = dict(reach=P)
    if fast:
        return all(oks), lines, data
    peaks = []
    for ta in T_ACTS:
        n = int(round((ta + 0.8) / DT))
        tr = _trace2d(runs[f"tact{ta}"]["P"], T, n)
        peaks.append(np.nanmax(bend_kinematics(tr, s_node)[3]))
    t_acts = np.array(T_ACTS)
    ratios = np.array(peaks) / (L / t_acts)
    natural = (t_acts >= 0.6) & (t_acts <= 1.2)
    gate("a4 peak bend speed ~ wave speed (t_act 0.6-1.2 s)",
         bool(np.all((ratios[natural] >= 0.7) & (ratios[natural] <= 1.3))),
         "peak / wave = " + ", ".join(f"{q:.2f} @ {ta} s" for q, ta in zip(ratios, t_acts)))
    n_b = int(round(1.05 / DT))
    profiles, norm_curves = {}, {}
    for a in A_CAS:
        tr = _trace2d(runs[f"amp{a}"]["P"], T, n_b)
        tt, ss, pp, vv, aa = bend_kinematics(tr, s_node)
        profiles[a] = (tt, vv, mean_local_angle_deg(tr.p[-1]))
        ok = np.isfinite(vv)
        if ok.sum() >= 5:
            tn = (tt[ok] - tt[ok][0]) / (tt[ok][-1] - tt[ok][0])
            norm_curves[a] = np.interp(np.linspace(0, 1, 50), tn, vv[ok] / vv[ok].max())
    above = [a for a in A_CAS if profiles[a][2] <= 2.0]
    if len(above) >= 3:
        ref = norm_curves[above[-1]]
        rms = {a: float(np.sqrt(np.mean((norm_curves[a] - ref) ** 2))) for a in above if a in norm_curves}
        pk = {a: float(np.nanmax(profiles[a][1])) for a in above}
        spread = (max(pk.values()) - min(pk.values())) / max(pk.values())
        gate("b kinematics invariant to a_ca above the minimum force", max(rms.values()) <= 0.15 and spread <= 0.2,
             f"straight for a_ca {list(above)}; RMS vs {above[-1]}: "
             + ", ".join(f"{a}: {q:.3f}" for a, q in rms.items()) + f"; peak spread {spread:.0%}")
    else:
        gate("b kinematics invariant to a_ca above the minimum force", False,
             "fewer than three amplitudes straightened: " + ", ".join(f"{a}: {profiles[a][2]:.1f}" for a in A_CAS))
    nd = int(round(2.0 / DT))
    trd = _trace2d(runs["dead"]["P"], T, nd)
    td, sd, pd, vd, ad = bend_kinematics(trd, s_node)
    okd = np.isfinite(vd)
    vv = vd[okd]
    i_pk = int(vv.argmax()) if len(vv) else 0
    after = vv[i_pk:]
    mono = bool(np.all(np.diff(after) <= 0.05 * vv.max())) if len(after) > 2 else False
    gate("c dead arm: bend decelerates monotonically, ends curved",
         mono and mean_local_angle_deg(trd.p[-1]) > 2.0,
         f"peak {vv.max() if len(vv) else 0:.2f} m/s, then {'monotone' if mono else 'NOT monotone'} over "
         f"{len(after)} ticks; final mean angle {mean_local_angle_deg(trd.p[-1]):.1f} deg")
    ok_d, note_d, _ = g7_cost(fast=True)
    lines.append("  [REPORTED] d operation count per arm-substep: " + note_d[0].strip())
    return all(oks), lines, data


# ====================================================================== G2: dead arm sinks

def g2_dead_arm(fast: bool = False):
    seconds = 6.0 if fast else 20.0
    r = run_dead_direct(seconds)
    T = (np.arange(len(r["Z"])) + 1) * DT
    touch = r["Z"].min(axis=1) <= 0.0
    t_touch = float(T[np.argmax(touch)]) if touch.any() else float("nan")
    last = T > T[-1] - 5.0
    vmax = float(r["V"][last].max())
    zmax = float(r["Z"][last].max())
    zmin = int(r["zmin_counts"].min())
    ok = touch.any() and vmax < 1e-3 and zmin >= -1
    lines = [f"  released 0.2 m above the deck: first node on it at {t_touch:.2f} s (float 1.7 s); over the "
             f"last 5 s of {T[-1]:.0f} s: max node speed {vmax * 1e3:.4f} mm/s (limit 1), max z "
             f"{zmax * 1e3:.4f} mm; lowest node ever {zmin} counts (limit -1); the centroid ends at "
             f"({r['cent'][0]:.3f}, {r['cent'][1]:.3f}) m (the arm unfurls and slides, as in float)"]
    return ok, lines, r


# ====================================================================== G3: no slide

def g3_specs(fast: bool):
    seconds = 10.0 if fast else 60.0
    return [(seconds, False), (min(seconds, 10.0), True)]


def g3_no_slide(fast: bool = False, results=None):
    honest, biased = results if results is not None else [run_free_animal(s) for s in g3_specs(fast)]
    seconds = len(honest["C"]) * DT
    dh = np.linalg.norm(honest["C"][:, :2] - honest["C0"][:2], axis=1)
    db = np.linalg.norm(biased["C"][:, :2] - biased["C0"][:2], axis=1)
    swing = float(np.linalg.norm(honest["BX"][:, :2] - honest["BX"][0, :2], axis=1).max())
    ok = dh.max() <= 1e-3 and db.max() > 10 * max(dh[:len(db)].max(), 1e-6)
    lines = [f"  free animal (P5z N2's idle inputs, uniform drag tau 1 s, mu 0, no anchors; an open topology: "
             f"contacts on, nothing touched), {seconds:.0f} s: "
             f"horizontal COM moved max {dh.max() * 1e3:.4f} mm (end {dh[-1] * 1e3:.4f} mm; limit 1 mm; "
             f"float 2.5e-10 mm) while the body swung up to {swing * 1e3:.1f} mm",
             f"  PAIR (must slide): the same run on FLOOR rounding (>> and //, every symmetric round "
             f"swapped) moves the COM {db.max() * 1e3:.3f} mm in {len(db) * DT:.0f} s -- "
             f"{db.max() / max(dh[:len(db)].max(), 1e-12):.0f}x the honest run's {dh[:len(db)].max() * 1e3:.4f} mm",
             f"  trace params vs row: {honest['params'] or 'equal'}"]
    return ok, lines, dict(honest=honest, biased=biased)


# ====================================================================== G4: float vs int

def g4_specs(fast: bool):
    return ["reach"] if fast else ["reach", "dead_arm", "crawl"]


def g4_float_vs_int(fast: bool = False, results=None):
    res = results if results is not None else [run_replay(n) for n in g4_specs(fast)]
    R_ = {r["name"]: r for r in res}
    r = R_["reach"]
    ok = r["rms"] <= REACH_RMS_TOL_M
    lines = ["  every replay runs WITH an open topology (P6c: contacts on, nothing touched)",
             f"  reach (1 arm, {r['ticks']} ticks, gated): node deviation max {r['max'] * 1e3:.3f} mm, RMS "
             f"{r['rms'] * 1e3:.3f} mm (tolerance {REACH_RMS_TOL_M * 1e3:.0f} mm = 0.4 % of L: the P1 gates' "
             f"finest length scale is a3's 5 % L, and P5z's table-vs-tanh was 1e-5 m); of it, the clock's "
             f"quantization of t_act (0.9 s -> {r['arith']['tact_sub']} substeps) is nearly all: against the float "
             f"run AT that t_act the int deviates max {r['arith']['max'] * 1e3:.4f} mm, RMS "
             f"{r['arith']['rms'] * 1e3:.4f} mm (the arithmetic's own share)"]
    if "dead_arm" in R_:
        d = R_["dead_arm"]
        lines.append(f"  dead arm (20 s, reported): max {d['max'] * 1e3:.2f} mm, RMS {d['rms'] * 1e3:.2f} mm; the "
                     f"curled arm unfurls and slides ~2.3 m over the deck, the int centroid ends "
                     f"{np.linalg.norm(d['cent_int'] - d['cent_float']) * 1e3:.1f} mm from the float's")
    if "crawl" in R_:
        c = R_["crawl"]
        lines.append(f"  crawl (25 s, recorded hub inputs, reported -- chaotic): nodes max {c['max'] * 1e3:.0f} mm, "
                     f"RMS {c['rms'] * 1e3:.0f} mm; body max {c['body_dev'].max() * 1e3:.0f} mm, at 1 / 5 / 25 s "
                     f"{c['body_dev'][23] * 1e3:.1f} / {c['body_dev'][119] * 1e3:.1f} / "
                     f"{c['body_dev'][-1] * 1e3:.0f} mm")
    for r_ in res:
        if r_["params"]:
            lines.append(f"  {r_['name']} trace params vs row: {r_['params']}")
    return ok, lines, R_


# ====================================================================== G5: determinism

def g5_determinism(fast: bool = False):
    n = 8 if fast else int(round(1.6 / DT))
    spec = ("wave", 0.6, 0.9, n, None, False)
    a, b = run_p1_arm(spec), run_p1_arm(spec)
    c = run_p1_arm(spec[:5] + (True,))
    ha = hashlib.sha256(a["state"]).hexdigest()[:16]
    ok = a["state"] == b["state"] == c["state"]
    lines = [f"  the reach, {n} ticks, run twice and once in CHECK mode: state bytes "
             f"{'IDENTICAL' if ok else 'DIFFER'} (sha256 {ha}..., {len(a['state'])} bytes)",
             "  (one arithmetic path: Python ints; int64 safety is G6's measurement, not a second path)"]
    return ok, lines, None


# ====================================================================== G6: headroom

def g6_headroom(fast: bool = False):
    n = 6 if fast else int(round(1.6 / DT))
    lines, ok = [], True
    worst = {}
    for L in SIZES:
        try:
            r = run_p1_arm(("wave", 1.0, 0.5, n, L, True))
        except OverflowError as e:
            ok = False
            lines.append(f"  {L:.2f} m: OVERFLOW {e}")
            continue
        for site, bits, lim in r["bits"]:
            if site not in worst or lim - bits < worst[site][1] - worst[site][0]:
                worst[site] = (bits, lim, L)
        tight = min(r["bits"], key=lambda x: x[2] - x[1])
        lines.append(f"  {L:.2f} m reach (a_ca 1.0, t_act 0.5 s: the fastest, stiffest P1 wave), {n} ticks: "
                     f"{len(r['bits'])} sites, tightest {tight[0]} {tight[1]} / {tight[2]} bits")
    R.CHECK = True
    try:
        for L in ((3.39,) if fast else SIZES):
            sc = scenario_animal(creature(L), ticks=6 if fast else 48)
            for t in range(sc.ticks):
                sc.step(t)
            for site, bits, lim in R.bits_report():
                if site not in worst or lim - bits < worst[site][1] - worst[site][0]:
                    worst[site] = (bits, lim, L)
            lines.append(f"  {L:.2f} m synthetic animal (8 arms, the crown's disc, waves, anchors + pops, a grab, "
                         f"walls + a footprint), {sc.ticks} ticks: no overflow")
            R.reset_bits()
    except OverflowError as e:
        ok = False
        lines.append(f"  animal: OVERFLOW {e}")
    finally:
        R.CHECK = False
        R.reset_bits()
    if not fast:
        R.CHECK = True
        try:
            cdef = creature()
            row = R.chain_row(cdef)
            tr = R.load_traces()["crawl"]
            R.replay(row, tr, ticks=48, record_every=1 << 30)
            for site, bits, lim in R.bits_report():
                if site not in worst or lim - bits < worst[site][1] - worst[site][0]:
                    worst[site] = (bits, lim, 2.52)
            lines.append("  2.52 m crawl replay (8 arms, the body, anchors, floor + friction), 2 s: no overflow")
        except OverflowError as e:
            ok = False
            lines.append(f"  crawl: OVERFLOW {e}")
        finally:
            R.CHECK = False
            R.reset_bits()
    lines.append("  per site, the largest magnitude seen (bits / limit, at size): " + "; ".join(
        f"{s} {b}/{lim}" for s, (b, lim, L) in sorted(worst.items())))
    return ok, lines, worst


# ====================================================================== G7: cost

#: weights (P1 gate d's): +, -, *, shift 1; a 128-bit multiply 4; a division
#: or a sqrt 8; atan2 / sin / cos 16
_W = dict(sra=2, dra=8, msa=4, isqrt=8, atan2=16, trig=16)
#: plain integer +, -, * per call site, tallied from the source (per arm-substep
#: per element): drag per node, distance per segment, frame per segment, bend
#: per plane, predict + velocity per node
_PLAIN = dict(drag=30, dist=22, frame=40, bend=95, node=24)


def g7_cost(fast: bool = False):
    counts = dict.fromkeys(_W, 0)
    orig = (R.sra, R.dra, R.msa, R.isqrt, R.K.atan2_q16, R.K.sin_q16, R.K.cos_q16)

    def wrap(key, fn):
        def f(*a):
            counts[key] += 1
            return fn(*a)
        return f

    R.sra, R.dra, R.msa, R.isqrt = (wrap("sra", orig[0]), wrap("dra", orig[1]), wrap("msa", orig[2]),
                                    wrap("isqrt", orig[3]))
    R.K.atan2_q16 = wrap("atan2", orig[4])
    try:
        n = 6
        run_p1_arm(("wave", 0.6, 0.9, n, None, False))
    finally:
        R.sra, R.dra, R.msa, R.isqrt, R.K.atan2_q16, R.K.sin_q16, R.K.cos_q16 = orig
    cdef = creature()
    nodes, nsub, npass = cdef.body.nodes_per_arm, cdef.solver.substeps, cdef.solver.passes
    subs = n * nsub
    per = {k: v / subs for k, v in counts.items()}
    plain = (nodes * (_PLAIN["drag"] + _PLAIN["node"])
             + npass * ((nodes - 1) * _PLAIN["dist"] + (nodes - 2) * (_PLAIN["frame"] + 2 * _PLAIN["bend"])))
    weighted = plain + sum(per[k] * _W[k] for k in per)
    per_oct = weighted * nsub * 8
    ok = True       # REPORTED: cost is not a design input (Erik's ruling 1); P6e measures
    over = f"{weighted / 25_000 - 1:+.0%} against" if weighted > 25_000 else "under"
    lines = [f"  REPORTED: ~{weighted:.0f} weighted ops per arm-substep ({over} the float's 25000 bound; "
             f"the integer chain pays for its 128-bit products, symmetric roundings and fine remainders): kit calls "
             + ", ".join(f"{k} {per[k]:.0f}" for k in per) + f" + ~{plain} plain +, -, * ; x {nsub} x 8 arms = "
             f"~{per_oct / 1e6:.2f} M per octopus-tick (the P5z float tally: ~18200 weighted, ~2.33 M)"]
    return ok, lines, per


# ====================================================================== G8: walls (P6c)

def g8_walls(fast: bool = False):
    import hashlib
    cdef = creature()
    lines, oks = [], []

    def gate(name, ok, note):
        oks.append(bool(ok))
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {note}")

    # (a) inert on an open map: identical state bytes with and without
    n_r = 12 if fast else int(round(1.6 / DT))
    n_d = 24 if fast else int(round(4.0 / DT))
    same = []
    for mk, n in ((scenario_reach, n_r), (scenario_dead, n_d)):
        bs = []
        for topo in (None, open_topology()):
            sc = mk(cdef, ticks=n, topo=topo)
            for t in range(n):
                sc.step(t)
            bs.append(sc.world.state_bytes())
        same.append((sc.name, n, bs[0] == bs[1], hashlib.sha256(bs[0]).hexdigest()[:12]))
    gate("a inert where there are no walls", all(x[2] for x in same),
         "; ".join(f"{nm} {n} ticks: {'IDENTICAL' if eq else 'DIFFER'} (sha256 {h}...)" for nm, n, eq, h in same))

    # (b) thrown flat at a flat wall: stops at the face, never in, never slides
    sc = scenario_wall_thrown(cdef, ticks=24 if fast else None)
    w = sc.world
    p0 = [list(q) for q in w.arms[0].p]
    face = w.topo.origin[0] + WALL_X_TILE * w.topo.tile_q24 - 1
    worst_in, slide = 0, 0
    for t in range(sc.ticks):
        sc.step(t)
        for q, q0 in zip(w.arms[0].p, p0):
            worst_in = max(worst_in, q[0] - face)
            slide = max(slide, abs(q[1] - q0[1]), abs(q[2] - q0[2]))
    last = w.arms[0]
    gap = max(face - q[0] for q in last.p)
    vx = max(abs(v[0]) for v in last.v)
    walled = all(c & R.CONTACT_WALL for c in last.contact) if not fast else True
    ok_b = worst_in <= 0 and slide == 0 and (fast or (gap <= 2 and vx <= 384 and walled))
    gate("b an arm thrown flat at a wall stops at the face, never in, never slides", ok_b,
         f"{sc.ticks} ticks at 1 m/s: deepest past the face {worst_in} counts (limit 0); along the wall "
         f"y / z moved at most {slide} counts (limit 0: no tangential velocity, no slide); at the end every node "
         f"within {gap} counts of the face, |v_x| <= {vx} counts ({vx / R.CF.CHAIN_ONE * 1e3:.4f} mm/s), "
         f"wall contact on every node: {walled}")

    # (c) oblique + (d) corner: never end a tick inside a blocked tile
    for label, mk in (("c an oblique throw (gravity, floor, friction)", scenario_wall_oblique),
                      ("d a tip-first throw at the pillar's corner", scenario_wall_corner)):
        sc = mk(cdef, ticks=36 if fast else None)
        w = sc.world
        bad, touched = 0, 0
        for t in range(sc.ticks):
            sc.step(t)
            bad += sum(inside_blocked(w, q) for q in w.arms[0].p)
            touched = max(touched, sum(1 for c in w.arms[0].contact if c & R.CONTACT_WALL))
        ok_c = bad == 0 and touched > 0
        gate(label + " never ends a tick inside a blocked tile", ok_c,
             f"{sc.ticks} ticks: node-ticks inside a blocked tile {bad} (limit 0); most nodes on a wall in one "
             f"tick {touched}")

    # (e) bodies: a crown into the wall, a crown into another unit's footprint
    sc = scenario_body_wall(cdef, ticks=24 if fast else None)
    w = sc.world
    T = w.topo.tile_q24
    wall_x = WALL_X_TILE * T
    fy1 = FOOTPRINT[3] * T
    pen_w = pen_u = -(1 << 40)
    cont = [0, 0]
    for t in range(sc.ticks):
        sc.step(t)
        b0, b1 = w.bodies
        pen_w = max(pen_w, b0.x[0] + b0.coll_r_q24 - wall_x)
        pen_u = max(pen_u, fy1 + b1.coll_r_q24 - b1.x[1])
        cont[0] |= b0.contact & R.CONTACT_WALL
        cont[1] |= b1.contact & R.CONTACT_UNIT
    ok_e = pen_w <= 16 and pen_u <= 16 and all(cont)
    gate("e a crown pushed into a wall / another unit's footprint stays out", ok_e,
         f"{sc.ticks} ticks at 2 m/s: deepest disc overlap with the wall {pen_w} counts, with the footprint "
         f"{pen_u} counts (limit 16 = 1 um; the push is fine, the rebuild takes it); contact reported "
         f"(wall, unit): {bool(cont[0])}, {bool(cont[1])}")

    # (f) the new sites in CHECK mode at the largest art size
    R.CHECK = True
    R.reset_bits()
    try:
        big = creature(3.39)
        for mk in (scenario_wall_oblique, scenario_wall_corner, scenario_body_wall):
            sc = mk(big, ticks=12 if fast else 48)
            for t in range(sc.ticks):
                sc.step(t)
        sites = [x for x in R.bits_report() if x[0].split(".")[0] in ("wall", "disc", "friction", "anchor",
                                                                     "grab", "attach", "arm")]
        gate("f the contacts' int64 sites in CHECK mode (3.39 m)", True,
             "; ".join(f"{s_} {b}/{lim}" for s_, b, lim in sites) or "no site reached")
    except OverflowError as e:
        gate("f the contacts' int64 sites in CHECK mode (3.39 m)", False, f"OVERFLOW {e}")
    finally:
        R.CHECK = False
        R.reset_bits()
    return all(oks), lines, None


# ====================================================================== main

GATES = [
    ("G1 P1's seven gates on the integer chain", g1_p1),
    ("G2 a dead arm released 0.2 m above the deck sinks to it and rests", g2_dead_arm),
    ("G3 no slide: a free animal's centre of mass", g3_no_slide),
    ("G4 float vs int: P5z's traces replayed", g4_float_vs_int),
    ("G5 determinism", g5_determinism),
    ("G6 headroom at the three art sizes", g6_headroom),
    ("G7 cost per arm-substep", g7_cost),
    ("G8 static-topology contacts (P6c)", g8_walls),
]


def _run_gate(i_fast):
    i, fast = i_fast
    name, fn = GATES[i]
    t0 = time.perf_counter()
    try:
        ok, lines, _ = fn(fast)
    except Exception as exc:  # noqa: BLE001
        import traceback
        ok, lines = False, [f"  RAISED: {type(exc).__name__}: {exc}", traceback.format_exc()]
    return name, ok, lines, time.perf_counter() - t0


def main(argv) -> int:
    fast = "--fast" in argv
    only = [a for a in argv[1:] if not a.startswith("--")]
    sel = [i for i, (name, _) in enumerate(GATES)
           if not only or any(name.lower().startswith(o.lower()) for o in only)]
    if fast:
        sel = [i for i in sel if GATES[i][0][:2] in ("G1", "G5", "G6", "G8")]
    t0 = time.perf_counter()
    workers = max(1, (os.cpu_count() or 4) - 2)
    leaf = {"G3": (run_free_animal, g3_specs, g3_no_slide), "G4": (run_replay, g4_specs, g4_float_vs_int)}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = []
        for i in sel:
            key = GATES[i][0][:2]
            if key in leaf:
                job, specs, _ = leaf[key]
                futs.append((i, [ex.submit(job, sp) for sp in specs(fast)]))
            else:
                futs.append((i, ex.submit(_run_gate, (i, fast))))
        results = []
        for i, f in futs:
            key = GATES[i][0][:2]
            if key in leaf:
                t1 = time.perf_counter()
                try:
                    ok, lines, _ = leaf[key][2](fast, [x.result() for x in f])
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    ok, lines = False, [f"  RAISED: {type(exc).__name__}: {exc}", traceback.format_exc()]
                results.append((GATES[i][0], ok, lines, time.perf_counter() - t1))
            else:
                results.append(f.result())
    all_ok = True
    for name, ok, lines, wall in results:
        print(f"\n{name}  ({wall:.0f} s)\n{'-' * len(name)}")
        for ln in lines:
            print(ln)
        print(f"  => {'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    wall = time.perf_counter() - t0
    print(f"\n{'ALL GATES PASS' if all_ok else 'SOME GATES FAILED'} in {wall:.0f} s")
    if not fast and not only:
        _write_report(results, wall)
    return 0 if all_ok else 1


def _write_report(results, wall) -> None:
    out = ["# P6c gate report -- the arm chain's integer reference (P6b's seven + the walls)\n",
           f"Generated by `chain_ref_q_gates.py`, {time.strftime('%Y-%m-%d %H:%M')}, {wall:.0f} s. "
           "Reference: `chain_ref_q.py`; row: `[creatures.octopus]` (config.toml) through the P6a door; "
           "float baseline: P5z (`prototypes/octopus_motion/out/traces_p5z.npz`).\n",
           f"**{sum(r[1] for r in results)}/{len(results)} gates pass.**\n",
           "| gate | result | measurement |", "|---|---|---|"]
    for name, ok, lines, _ in results:
        body = "<br>".join(ln.strip().replace("|", "/") for ln in lines)
        out.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {body} |")
    (HERE / "report_p6c.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"report: {HERE / 'report_p6c.md'}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
