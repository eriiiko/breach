"""The arm chain solver's Python side (#106 P6c): the BUFFER CONTRACT of
``cpp/src/arm_chain_solver.h`` -- allocation, the row's and the set-up's
quantization (the door), and the one call.

The solver (``breach_physics.arm_chain_tick``) is STATELESS: every array below
is owned by the caller (``CreatureBody``, P6d1), so the sim stays picklable;
one call is one TICK for every environment, in place. It is held at tol 0 to
the integer reference ``docs/octopus_integer_reference/chain_ref_q.py``
(``World.tick``) by ``tests/test_octopus_arm_chain_solver.py``; this module's
quantizers are held equal to the reference's own door (``chain_row``,
``World.add_arm`` / ``add_body`` / ``set_wave`` / ``set_grip``) by the same
test. Nothing here imports the reference.

Shapes (N environments -- 1 today, RL-batch habits; A arms, n nodes, B bodies,
G grab slots, U footprint slots, an (h, w) tile grid), dtypes, currencies:
see ``ArmChainBuffers`` and the header. The layout constants are LITERAL twins
of the C++ enums (``breach_physics.ARM_CHAIN_LAYOUT``), held equal by the test.

Import-light (numpy + the creature currencies); the engine is imported only by
``tick``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from simulation import creature_fixed as CF
from simulation.fixedpoint_twins import div_round_away

# ---------------------------------------------------------------------------
# The layout (twins of arm_chain_solver.h's enums)
# ---------------------------------------------------------------------------
(ROW_N_NODES, ROW_N_SUB, ROW_N_PASS, ROW_INV_DT, ROW_INV_DT2, ROW_L_Q24, ROW_K_DIST_Q16,
 ROW_GAMMA_Q16, ROW_ONEPG_Q16, ROW_ATAN2_SHIFT, ROW_G_DV_Q24, ROW_PLANE_MIN_Q32, ROW_N) = range(13)
(RN_S_NODE_Q24, RN_W_Q32, RN_K_PER_Q16, RN_K_TAN_Q16, RN_REST0_Q24, RN_K_ACT_Q16,
 RN_K_PAS_Q16, RN_N) = range(8)
WD_FLOOR_ON, WD_HAS_TOPO, WD_TILE_Q24, WD_ORIGIN_X_Q24, WD_ORIGIN_Y_Q24, WD_N = range(6)
(AC_DRAG_MODE, AC_VREF_Q24, AC_U_Q30, AC_MU_Q16, AC_ROOT_REF_X_Q30, AC_ROOT_REF_Y_Q30,
 AC_ROOT_REF_Z_Q30, AC_BODY, AC_GRIP_Q8, AC_WAVE_ON, AC_WAVE_A_CA_Q16, AC_WAVE_TACT_SUB,
 AC_WAVE_W_Q24, AC_WAVE_HOLD, AC_N) = range(15)
(BC_W_Q32, BC_WINVI_Q32, BC_CB_Q32, BC_DRAG_MODE, BC_VREF_Q24, BC_U_Q30, BC_Y_Q30, BC_MU_Q16,
 BC_GRAVITY, BC_FLOOR, BC_COLL_R_Q24, BC_N) = range(12)
BS_X, BS_Y, BS_Z, BS_VX, BS_VY, BS_VZ, BS_TH, BS_OM, BS_N = range(9)
GR_ARM, GR_NODE, GR_BODY, GR_LX, GR_LY, GR_LZ, GR_N = range(7)
DRAG_QUADRATIC, DRAG_LINEAR, DRAG_UNIFORM = 0, 1, 2
CONTACT_FLOOR, CONTACT_WALL, CONTACT_UNIT = 1, 2, 4
#: ``AC_GRIP_Q8`` below zero: the anchors hold whatever the force.
NO_GRIP = -1
U_ONE = 1 << 30
Q16_ONE = 1 << 16


def layout() -> dict:
    """Every layout constant by name (compared to the engine's twin)."""
    g = globals()
    return {k: g[k] for k in g if k.isupper() and k.split("_")[0] in
            ("ROW", "RN", "WD", "AC", "BC", "BS", "GR", "DRAG", "CONTACT")}


# ---------------------------------------------------------------------------
# The buffers
# ---------------------------------------------------------------------------
@dataclass
class ArmChainBuffers:
    """Every array of one solver call, owned by the caller (see the header for
    each one's meaning). State: p, v, act, wave_clock, anchored, body. Per-tick
    inputs: arm_cfg, pinned, root_local, act (non-wave arms), rest_scale,
    bend_target, anchored, anchor_pos, body_cfg, grabs, world, blocked,
    medium, unit_rects. Outputs (rewritten every tick): contact, popped,
    anchor_force, body_contact, grab_force."""
    world: np.ndarray          # int64 (N, WD_N)
    blocked: np.ndarray        # uint8 (N, h, w)  GameMap.solid (material + door state)
    medium: np.ndarray         # uint8 (N, h, w)  read, unused in v1 (water everywhere)
    unit_rects: np.ndarray     # int32 (N, U, 4)  other units' footprints, half-open tiles
    p: np.ndarray              # int32 (N, A, n, 3) Q24 m
    v: np.ndarray              # int32 (N, A, n, 3) Q24 m/s
    arm_cfg: np.ndarray        # int64 (N, A, AC_N)
    pinned: np.ndarray         # uint8 (N, A, n)
    root_local: np.ndarray     # int32 (N, A, 2, 3) Q24 m, body frame
    act: np.ndarray            # int32 (N, A, n-2) Q16
    rest_scale: np.ndarray     # int32 (N, A, n-1) Q16
    bend_target: np.ndarray    # int32 (N, A, n-2, 2) Q16 rad
    wave_clock: np.ndarray     # int64 (N, A) substeps since the arm's wave started
    anchored: np.ndarray       # uint8 (N, A, n)
    anchor_pos: np.ndarray     # int32 (N, A, n, 3) Q24 m
    contact: np.ndarray        # uint8 (N, A, n)  OUT CONTACT_* bits
    popped: np.ndarray         # uint8 (N, A, n)  OUT
    anchor_force: np.ndarray   # int64 (N, A, n)  OUT Q8 N
    body: np.ndarray           # int32 (N, B, BS_N) Q24 (th Q24 rad, om Q24 rad/s)
    body_cfg: np.ndarray       # int64 (N, B, BC_N)
    body_contact: np.ndarray   # uint8 (N, B)     OUT CONTACT_* bits
    grabs: np.ndarray          # int64 (N, G, GR_N) arm < 0 = empty
    grab_force: np.ndarray     # int64 (N, G)     OUT Q8 N

    def copy(self) -> "ArmChainBuffers":
        return ArmChainBuffers(**{f.name: getattr(self, f.name).copy() for f in fields(self)})


def allocate(n_env: int, arms: int, nodes: int, bodies: int = 1, grabs: int = 0,
             unit_slots: int = 0, h: int = 1, w: int = 1) -> ArmChainBuffers:
    """Zeroed buffers, born (N, arms, nodes, 3) with N = ``n_env``; rest scale
    1.0, no grip limit, no body, empty grab slots, no topology, floor on."""
    if nodes < 3:
        raise ValueError(f"arm_chain.allocate: nodes per arm must be >= 3 (got {nodes})")
    N, A, n, B, G = int(n_env), int(arms), int(nodes), int(bodies), int(grabs)
    world = np.zeros((N, WD_N), np.int64)
    world[:, WD_FLOOR_ON] = 1
    world[:, WD_TILE_Q24] = 1
    arm_cfg = np.zeros((N, A, AC_N), np.int64)
    arm_cfg[..., AC_BODY] = -1
    arm_cfg[..., AC_GRIP_Q8] = NO_GRIP
    arm_cfg[..., AC_ROOT_REF_Z_Q30] = U_ONE
    body_cfg = np.zeros((N, B, BC_N), np.int64)
    body_cfg[..., BC_W_Q32] = 1
    body_cfg[..., BC_GRAVITY] = 1
    body_cfg[..., BC_FLOOR] = 1
    gr = np.zeros((N, G, GR_N), np.int64)
    gr[..., GR_ARM] = -1
    return ArmChainBuffers(
        world=world, blocked=np.zeros((N, h, w), np.uint8), medium=np.zeros((N, h, w), np.uint8),
        unit_rects=np.zeros((N, int(unit_slots), 4), np.int32),
        p=np.zeros((N, A, n, 3), np.int32), v=np.zeros((N, A, n, 3), np.int32), arm_cfg=arm_cfg,
        pinned=np.zeros((N, A, n), np.uint8), root_local=np.zeros((N, A, 2, 3), np.int32),
        act=np.zeros((N, A, n - 2), np.int32),
        rest_scale=np.full((N, A, n - 1), Q16_ONE, np.int32),
        bend_target=np.zeros((N, A, n - 2, 2), np.int32), wave_clock=np.zeros((N, A), np.int64),
        anchored=np.zeros((N, A, n), np.uint8), anchor_pos=np.zeros((N, A, n, 3), np.int32),
        contact=np.zeros((N, A, n), np.uint8), popped=np.zeros((N, A, n), np.uint8),
        anchor_force=np.zeros((N, A, n), np.int64), body=np.zeros((N, B, BS_N), np.int32),
        body_cfg=body_cfg, body_contact=np.zeros((N, B), np.uint8), grabs=gr,
        grab_force=np.zeros((N, G), np.int64))


# ---------------------------------------------------------------------------
# The door: the row and the set-up, quantized once
# ---------------------------------------------------------------------------
def row_arrays(cdef, ticks_per_second: int) -> tuple[np.ndarray, np.ndarray]:
    """(row, row_node) for a CreatureDef: the chain constants the reference's
    ``chain_row`` quantizes, in the solver's layout."""
    b, t, s, ch = cdef.body, cdef.tissue, cdef.solver, cdef.chain
    n = int(b.nodes_per_arm)
    inv_dt = int(ticks_per_second) * int(s.substeps)
    if inv_dt * inv_dt != ch.inv_dt2:
        raise ValueError("arm_chain.row_arrays: the row's 1/dt^2 disagrees with tps x substeps")
    seg = CF.chain_from_m(b.arm_length_m / (n - 1))
    k_act, k_pas = CF.node_bend_stiffness_q16(n, b.radius_base_m, b.radius_tip_m, t.k_bend,
                                              t.k_passive_frac)
    kp, kt = CF.node_drag_k_q16(n, b.arm_length_m, b.radius_base_m, b.radius_tip_m,
                                b.water_density, t.c_drag_perp, t.c_drag_tan)
    row = np.zeros(ROW_N, np.int64)
    row[ROW_N_NODES] = n
    row[ROW_N_SUB] = int(s.substeps)
    row[ROW_N_PASS] = int(s.passes)
    row[ROW_INV_DT] = inv_dt
    row[ROW_INV_DT2] = ch.inv_dt2
    row[ROW_L_Q24] = CF.chain_from_m(b.arm_length_m)
    row[ROW_K_DIST_Q16] = ch.k_dist_q16
    row[ROW_GAMMA_Q16] = ch.bend_gamma_q16
    row[ROW_ONEPG_Q16] = Q16_ONE + ch.bend_gamma_q16
    row[ROW_ATAN2_SHIFT] = ch.atan2_shift
    row[ROW_G_DV_Q24] = div_round_away(CF.chain_from_m(t.g_eff), inv_dt)
    row[ROW_PLANE_MIN_Q32] = ch.bend_plane_min_q32
    rn = np.zeros((n, RN_N), np.int64)
    for i in range(n):
        rn[i, RN_S_NODE_Q24] = seg * i
        rn[i, RN_W_Q32] = ch.node_inv_mass_q32[i]
        rn[i, RN_K_PER_Q16] = kp[i]
        rn[i, RN_K_TAN_Q16] = kt[i]
        if i < n - 1:
            rn[i, RN_REST0_Q24] = seg
        if i < n - 2:
            rn[i, RN_K_ACT_Q16] = k_act[i]
            rn[i, RN_K_PAS_Q16] = k_pas[i]
    return row, rn


def _q16(x: float) -> int:
    return CF.q16_from_scalar(x)


def _rate_q30(tau_s: float, inv_dt: int) -> int:
    """dt / tau in Q30 (the uniform drag law), 0 for tau <= 0."""
    tau_q16 = _q16(tau_s)
    return div_round_away(1 << 46, inv_dt * tau_q16) if tau_q16 > 0 else 0


def unit_q30(vec) -> tuple[int, int, int]:
    """A direction -> a Q30 unit vector (door 3: + * / and sqrt only)."""
    n = math.sqrt(sum(float(c) ** 2 for c in vec))
    return tuple(CF._round_half_away(float(c) / n * U_ONE) for c in vec)


def grip_q8(grip_n) -> int:
    """A grip limit in N -> Q8 N; None or infinite -> NO_GRIP."""
    if grip_n is None or not math.isfinite(float(grip_n)):
        return NO_GRIP
    return CF._round_half_away(float(grip_n) * 256.0)


def arm_cfg_row(*, inv_dt: int, drag_mode: int = DRAG_QUADRATIC, vref_m_s: float = 0.0,
                tau_s: float = 1.0, mu: float = 0.0, root_ref=(0.0, 0.0, 1.0), body: int = -1,
                grip_n=None) -> np.ndarray:
    """One arm's ``arm_cfg`` row (no wave): the reference's ``add_arm`` door."""
    r = np.zeros(AC_N, np.int64)
    r[AC_DRAG_MODE] = int(drag_mode)
    r[AC_VREF_Q24] = CF.chain_from_m(vref_m_s)
    r[AC_U_Q30] = _rate_q30(tau_s, inv_dt)
    r[AC_MU_Q16] = _q16(mu)
    r[AC_ROOT_REF_X_Q30:AC_ROOT_REF_Z_Q30 + 1] = unit_q30(root_ref)
    r[AC_BODY] = int(body)
    r[AC_GRIP_Q8] = grip_q8(grip_n)
    return r


def set_wave(cfg_row: np.ndarray, *, inv_dt: int, L_q24: int, a_ca: float, t_act_s: float,
             width_m: float, hold: bool = True) -> None:
    """Write a stiffening wave into an ``arm_cfg`` row (the reference's
    ``set_wave``); the caller zeroes the arm's ``wave_clock`` at its start."""
    w_q24 = CF.chain_from_m(width_m)
    if 8 * w_q24 > 3 * L_q24:
        raise ValueError("arm_chain.set_wave: front width > 3 L / 8")
    tact = CF._round_half_away(float(t_act_s) * inv_dt)
    if tact < 1 or w_q24 < 1:
        raise ValueError("arm_chain.set_wave: t_act must be >= 1 substep and the width > 0")
    cfg_row[AC_WAVE_ON] = 1
    cfg_row[AC_WAVE_A_CA_Q16] = _q16(a_ca)
    cfg_row[AC_WAVE_TACT_SUB] = tact
    cfg_row[AC_WAVE_W_Q24] = w_q24
    cfg_row[AC_WAVE_HOLD] = int(bool(hold))


def body_cfg_row(*, inv_dt: int, water_density: float, mass_kg: float, radius_m: float,
                 c_drag: float = 1.0, area_m2: float = 0.0, rot_damp: float = 0.0, mu: float = 0.0,
                 inertia=None, gravity: bool = True, floor: bool = True,
                 drag_mode: int = DRAG_QUADRATIC, tau_s: float = 1.0, vref_m_s: float = 0.0,
                 coll_radius_m=None) -> np.ndarray:
    """One body's ``body_cfg`` row: the reference's ``add_body`` door."""
    m_q16 = CF.q16_from_kg(mass_kg)
    inertia_kg_m2 = 0.5 * mass_kg * radius_m ** 2 if inertia is None else float(inertia)
    w = CF.inv_mass_q32(m_q16)
    winv_i = CF.inv_mass_q32(CF.q16_from_kg(inertia_kg_m2))
    rho_q16 = _q16(water_density)
    kb_q16 = CF._round_half_away(0.5 * (rho_q16 / 65536.0) * c_drag * area_m2 * 65536.0)
    r = np.zeros(BC_N, np.int64)
    r[BC_W_Q32] = w
    r[BC_WINVI_Q32] = winv_i
    r[BC_CB_Q32] = div_round_away(kb_q16 * w, inv_dt << 16)
    r[BC_DRAG_MODE] = int(drag_mode)
    r[BC_VREF_Q24] = CF.chain_from_m(vref_m_s)
    u = _rate_q30(tau_s, inv_dt)
    r[BC_U_Q30] = u
    r[BC_Y_Q30] = u if int(drag_mode) == DRAG_UNIFORM else \
        div_round_away(_q16(rot_damp) * winv_i, inv_dt << 18)
    r[BC_MU_Q16] = _q16(mu)
    r[BC_GRAVITY] = int(bool(gravity))
    r[BC_FLOOR] = int(bool(floor))
    r[BC_COLL_R_Q24] = 0 if coll_radius_m is None else CF.chain_from_m(coll_radius_m)
    return r


def set_topology(bufs: ArmChainBuffers, env: int, blocked, tile_size_m: float,
                 origin_m=(0.0, 0.0)) -> None:
    """Install environment ``env``'s static topology: ``blocked`` is the map's
    ``GameMap.solid`` ((h, w) bool: material + door state -- never
    ``obstacles``), the tile side and tile (0, 0)'s corner in metres."""
    b = np.asarray(blocked)
    if b.shape != bufs.blocked.shape[1:]:
        raise ValueError(f"arm_chain.set_topology: blocked {b.shape} != the buffers' "
                         f"{bufs.blocked.shape[1:]}")
    bufs.blocked[env] = b.astype(np.uint8)
    bufs.world[env, WD_HAS_TOPO] = 1
    bufs.world[env, WD_TILE_Q24] = CF.chain_from_m(tile_size_m)
    bufs.world[env, WD_ORIGIN_X_Q24] = CF.chain_from_m(origin_m[0])
    bufs.world[env, WD_ORIGIN_Y_Q24] = CF.chain_from_m(origin_m[1])


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------
def tick(row: np.ndarray, row_node: np.ndarray, bufs: ArmChainBuffers, engine=None) -> None:
    """ONE tick of every environment's chains, in place (the C++ solver)."""
    if engine is None:
        import breach_physics as engine
    b = bufs
    engine.arm_chain_tick(row, row_node, b.world, b.blocked, b.medium, b.unit_rects, b.p, b.v,
                          b.arm_cfg, b.pinned, b.root_local, b.act, b.rest_scale, b.bend_target,
                          b.wave_clock, b.anchored, b.anchor_pos, b.contact, b.popped,
                          b.anchor_force, b.body, b.body_cfg, b.body_contact, b.grabs,
                          b.grab_force)


__all__ = [
    "ArmChainBuffers", "allocate", "row_arrays", "arm_cfg_row", "set_wave", "body_cfg_row",
    "set_topology", "unit_q30", "grip_q8", "tick", "layout", "NO_GRIP",
]
