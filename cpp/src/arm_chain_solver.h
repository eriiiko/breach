#pragma once
// ============================================================================
// ArmChainSolver -- the articulated-body solver for every creature with arms
// (#106 P6c; design docs/octopus_p6_sim_port_design_2026-10-30.md sections 2,
// 4.4 and 6).
// ============================================================================
//
// THE EXECUTABLE SPECIFICATION is the integer reference
// docs/octopus_integer_reference/chain_ref_q.py (`World.tick`): this solver is
// its transcription, held to it at TOLERANCE 0 by
// tests/test_octopus_arm_chain_solver.py (positions, velocities, body pose,
// activation, activation clocks, anchors and every output, after every tick).
// A change to the arithmetic lands in the reference FIRST, its gates are
// re-run, then this file is written against it -- never the other way round.
//
// Technique credits (what the reference transcribes; PDFs in
// docs/papers/octopus/):
//   * M. Macklin, M. Mueller, N. Chentanez, "XPBD: Position-Based Simulation
//     of Compliant Constrained Dynamics", MIG 2016 (10.1145/2994258.2994272)
//     -- the constraint projection (stiffness form, lambda reset per pass).
//   * M. Macklin, K. Storey, M. Lu, P. Terdiman, N. Chentanez, S. Jeschke,
//     M. Mueller, "Small Steps in Physics Simulation", SCA 2019
//     (10.1145/3309486.3340247) -- many substeps, few iterations (16 x 2).
//   * R. L. Bishop, "There is More than One Way to Frame a Curve", Amer.
//     Math. Monthly 82(3):246-251, 1975 -- the parallel-transported frame the
//     two signed bends per node live in.
//   * Y. Yekutieli, R. Sagiv-Zohar, R. Aharonov, Y. Engel, B. Hochner,
//     T. Flash, "Dynamic Model of the Octopus Arm. I. Biomechanics of the
//     Octopus Reaching Movement" and "II. Control of Reaching Movements",
//     J. Neurophysiol. 94:1443-1458 / 1459-1468, 2005
//     (10.1152/jn.00684.2004, 10.1152/jn.00685.2004) -- the stiffening wave
//     (the checked-in activation table) and the anisotropic drag law.
//   * M. Macklin, M. Mueller, N. Chentanez, T.-Y. Kim, "Unified Particle
//     Physics for Real-Time Applications", SIGGRAPH 2014
//     (10.1145/2601097.2601152) -- the position-based Coulomb friction on the
//     floor and the wall faces.
//
// STATELESS: the solver owns no per-creature state. Everything it reads and
// writes is a numpy buffer the caller owns (`CreatureBody`, P6d1), so the sim
// stays picklable; scratch is sized from the buffers once per call (never a
// per-arm allocation inside a substep). Pure integer (/fp:strict TU, 0/0/0 on
// the ratchet); trig and roots only through fixed_point.h.
//
// THE BUFFER CONTRACT (N environments, N = 1 today -- RL-batch habits; A arms
// per environment, n nodes per arm, B bodies, G grab slots, U footprint
// slots, an (h, w) tile grid). Currencies: Q24 = chain counts (2^24 per metre
// or m/s, `fixedpoint::CHAIN_SHIFT`), Q16 = 2^16, Q30 = unit vectors, Q32 =
// inverse masses (1/kg) and drag rates (s/m), Q8 N = forces. Every array is
// C-contiguous and of exactly the dtype named (never converted):
//
//   row        int64 (ROW_N,)            the species' chain constants (RowScalar)
//   row_node   int64 (n, RN_N)           per node (RowNodeCol)
//   world      int64 (N, WD_N)           floor on, topology on, tile, origin
//   blocked    uint8 (N, h, w)           the STATIC topology: GameMap.solid
//                                        (material + door state), never obstacles
//   medium     uint8 (N, h, w)           the medium per tile -- READ, unused in v1
//                                        (water everywhere: the row's g_eff and drag)
//   unit_rects int32 (N, U, 4)           other units' footprints, half-open tile
//                                        rects (tx0, ty0, tx1, ty1); empty if
//                                        tx1 <= tx0 or ty1 <= ty0; bodies only
//   p, v       int32 (N, A, n, 3)        IN/OUT node positions / velocities, Q24
//   arm_cfg    int64 (N, A, AC_N)        per arm (ArmCol): drag law, friction,
//                                        root reference, body, grip, the wave
//   pinned     uint8 (N, A, n)           a pinned node never moves (w = 0)
//   root_local int32 (N, A, 2, 3)        nodes 0 and 1 in the body frame, Q24
//   act        int32 (N, A, n-2)         IN/OUT activation per interior node,
//                                        Q16 (a wave arm overwrites it)
//   rest_scale int32 (N, A, n-1)         Q16 per segment (rest = rest0 x scale)
//   bend_target int32 (N, A, n-2, 2)     Q16 rad, planes A (toward nA), B (nB)
//   wave_clock int64 (N, A)              IN/OUT the arm's elapsed SUBSTEPS since
//                                        its wave started (the reference's
//                                        sub_count - t0_sub); + n_sub per tick
//   anchored   uint8 (N, A, n)           IN/OUT (a pop clears it)
//   anchor_pos int32 (N, A, n, 3)        Q24
//   contact    uint8 (N, A, n)           OUT CONTACT_* bits this tick (floor,
//                                        wall; the unit bit stays 0: nodes never
//                                        touch units in v1)
//   popped     uint8 (N, A, n)           OUT anchors popped this tick
//   anchor_force int64 (N, A, n)         OUT the largest anchor force this tick, Q8 N
//   body       int32 (N, B, BS_N)        IN/OUT x, y, z, vx, vy, vz (Q24), th (Q24
//                                        rad, wrapped into (-pi, pi]), om (Q24 rad/s)
//   body_cfg   int64 (N, B, BC_N)        per body (BodyCol)
//   body_contact uint8 (N, B)            OUT CONTACT_* bits this tick (floor,
//                                        wall = a blocked tile, unit = a footprint)
//   grabs      int64 (N, G, GR_N)        (arm, node, body, local x, y, z Q24),
//                                        sorted by (arm, node); arm < 0 = empty
//   grab_force int64 (N, G)              OUT the largest grab force this tick, Q8 N
//
// One call = one TICK (n_sub substeps x n_pass passes) for every environment.
// ============================================================================

#include <cstdint>

namespace arm_chain {

// ---- row (the species' constants, quantized once at the creature door) ----
enum RowScalar : int {
    ROW_N_NODES = 0,    // nodes per arm (>= 3)
    ROW_N_SUB,          // substeps per tick
    ROW_N_PASS,         // constraint passes per substep
    ROW_INV_DT,         // 1/dt = tps x n_sub (384)
    ROW_INV_DT2,        // 1/dt^2 (147456)
    ROW_L_Q24,          // arm length
    ROW_K_DIST_Q16,     // distance stiffness, N/m
    ROW_GAMMA_Q16,      // bend damping gamma = bend_damp_s / dt
    ROW_ONEPG_Q16,      // 1 + gamma
    ROW_ATAN2_SHIFT,    // the ONE narrowing shift of the bend's cross / dot
    ROW_G_DV_Q24,       // g_eff dt, the per-substep velocity step
    ROW_PLANE_MIN_Q32,  // the bend's in-plane guard: a plane whose in-plane segment
                        // length^2 (Q32) is below it is skipped (the door's l_min^2)
    ROW_N
};
enum RowNodeCol : int {
    RN_S_NODE_Q24 = 0,  // arc length at the node
    RN_W_Q32,           // inverse mass (unpinned)
    RN_K_PER_Q16,       // quadratic drag coefficient, kg/m, perpendicular
    RN_K_TAN_Q16,       // ... tangential
    RN_REST0_Q24,       // segment i's rest length (i < n - 1; 0 on the last row)
    RN_K_ACT_Q16,       // interior node j's active bend stiffness (j < n - 2)
    RN_K_PAS_Q16,       // ... passive
    RN_N
};
// ---- world ------------------------------------------------------------------
enum WorldCol : int {
    WD_FLOOR_ON = 0,    // the floor contact at z = 0
    WD_HAS_TOPO,        // 0: no static-topology contact at all (blocked unread)
    WD_TILE_Q24,        // a tile's side
    WD_ORIGIN_X_Q24,    // tile (0, 0)'s corner
    WD_ORIGIN_Y_Q24,
    WD_N
};
// ---- arms ---------------------------------------------------------------------
enum ArmCol : int {
    AC_DRAG_MODE = 0,   // DRAG_QUADRATIC / DRAG_LINEAR / DRAG_UNIFORM
    AC_VREF_Q24,        // the linear law's reference speed
    AC_U_Q30,           // the uniform law's dt / tau
    AC_MU_Q16,          // floor and wall friction
    AC_ROOT_REF_X_Q30,  // the Bishop frame's root reference (a unit vector)
    AC_ROOT_REF_Y_Q30,
    AC_ROOT_REF_Z_Q30,
    AC_BODY,            // the body nodes 0, 1 attach to, or -1
    AC_GRIP_Q8,         // the anchors' grip limit, Q8 N; < 0 = no limit
    AC_WAVE_ON,         // a stiffening wave writes `act`
    AC_WAVE_A_CA_Q16,   // its amplitude
    AC_WAVE_TACT_SUB,   // its travel time, substeps (> 0)
    AC_WAVE_W_Q24,      // its front width (> 0, <= 3 L / 8)
    AC_WAVE_HOLD,       // 0: the activation fades behind the front
    AC_N
};
// ---- bodies -------------------------------------------------------------------
enum BodyCol : int {
    BC_W_Q32 = 0,       // 1 / mass
    BC_WINVI_Q32,       // 1 / inertia, Q32 1/(kg m^2)
    BC_CB_Q32,          // quadratic drag rate, Q32 s/m
    BC_DRAG_MODE,
    BC_VREF_Q24,
    BC_U_Q30,
    BC_Y_Q30,           // yaw damping rate
    BC_MU_Q16,          // floor friction
    BC_GRAVITY,
    BC_FLOOR,
    BC_COLL_R_Q24,      // the collision disc's radius; <= 0 = no topology collision
    BC_N
};
enum BodyState : int { BS_X = 0, BS_Y, BS_Z, BS_VX, BS_VY, BS_VZ, BS_TH, BS_OM, BS_N };
enum GrabCol : int { GR_ARM = 0, GR_NODE, GR_BODY, GR_LX, GR_LY, GR_LZ, GR_N };

constexpr int DRAG_QUADRATIC = 0, DRAG_LINEAR = 1, DRAG_UNIFORM = 2;
constexpr int CONTACT_FLOOR = 1, CONTACT_WALL = 2, CONTACT_UNIT = 4;

// Raw views of the caller's buffers (the binding validates every shape and
// dtype before filling one; see the contract above).
struct ArmChainView {
    int N = 0, A = 0, n = 0, B = 0, G = 0, U = 0, h = 0, w = 0;
    const int64_t* row = nullptr;
    const int64_t* row_node = nullptr;
    const int64_t* world = nullptr;
    const uint8_t* blocked = nullptr;
    const uint8_t* medium = nullptr;
    const int32_t* unit_rects = nullptr;
    int32_t* p = nullptr;
    int32_t* v = nullptr;
    const int64_t* arm_cfg = nullptr;
    const uint8_t* pinned = nullptr;
    const int32_t* root_local = nullptr;
    int32_t* act = nullptr;
    const int32_t* rest_scale = nullptr;
    const int32_t* bend_target = nullptr;
    int64_t* wave_clock = nullptr;
    uint8_t* anchored = nullptr;
    const int32_t* anchor_pos = nullptr;
    uint8_t* contact = nullptr;
    uint8_t* popped = nullptr;
    int64_t* anchor_force = nullptr;
    int32_t* body = nullptr;
    const int64_t* body_cfg = nullptr;
    uint8_t* body_contact = nullptr;
    const int64_t* grabs = nullptr;
    int64_t* grab_force = nullptr;
};

// One tick for every environment (World.tick of the reference, per env).
void tick(const ArmChainView& view);

// The activation lookup (the reference's activation_s_q16), exposed for the
// binding's twin test.
int64_t activation_s_q16(int64_t x_q16);
int activation_table_samples();

}  // namespace arm_chain
