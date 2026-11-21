#pragma once
// ============================================================================
// #29 P6.4b -- THE TOPOLOGY SEAM AS DEVICE FUNCTIONS (residency design §4.3,
// "The per-env serial applier"; §5.7): GameMap.on_tile_changed and
// GameMap.destroy_wall, integer for integer, on ONE env's planes. Shared by
// every device caller that changes a tile's material: the per-env serial
// applier (cuda_applier.cu: slot 9's burn-through) now; P6b's door deaths and
// seals, P7a's chew, P7c's blasts call the same functions -- never a copy.
//
// THE ORACLE is src/simulation/gamemap.py; an arithmetic change lands there
// first, then here, and tests/cuda_unit_topology_check.py (parts SEAM, SCENES,
// CHAINED) holds the two equal at tol 0 on every plane and every book.
//
// THE PLANES (TopoEnv): one env's typed pointers into the (N, ...) arrays of
// the launch's pointer table, in the order of the GENERATED applier_layout.h
// (simulation/applier.PLANES), each in the GameMap field's own dtype and
// shape -- the resident fields as ResidentBatch.buffers hold them, so the
// device conductor (the env's device backend) hands its buffers unchanged. The float planes
// (light_atten, heat_atten, conductivity, permeability, wave_absorb) are
// written as their exact 32-bit patterns from the device tables (never a
// float here). Every per-material value comes from the env's device tables
// (unit_tables_layout.h: the mat_* rows, read from that env's own
// GameMap.materials), never an argument.
//
// THE BOOKS (int64 per env, applier_layout.h's APPLIER_BOOK_*): each seam
// write books where the Python books -- n_destruction_seed_sum (the tile's
// own bulk delta, measured), gas_energy_books' 'retire' / 'mint' /
// 'destroy_seed', solid_energy_books' 'destroy_wall' (the host's
// _solid_books_sink is the caller's), the on_tile_changed count (the host's
// _tile_serial bump).
//
// DROPPED ON THE DEVICE (host-only, no device form): render_dirty (the wall
// seam's mark), _sky_mask_dirty, _tile_serial itself (its bump is the
// tiles_changed book), the solid books' sink (the solver's term), and the
// callers' render-only Wall / DoorDestroyedEvent.
//
// A FAULT (an operand past the proved range: a donor's negative gas, a solid
// book product past int64) writes nothing and is counted; the driver refuses
// the run.
//
// Device-only header (CUDA types); included by the applier's TU and its
// TEST-ONLY probe (cuda_topology_probe.cu).
// CLASS T: nothing in the played tick includes it.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "applier_layout.h"        // GENERATED: the planes, the books
#include "fixed_point.h"           // the kit
#include "gas_energy.h"            // THE gas energy seam (shared with the edit queue)
#include "object_tables_layout.h"  // GENERATED: the door tile table's capacity (SPAN_MAX)
#include "temperature_solver.h"    // conduction::cell_capacity_q (the solid books)
#include "unit_tables_layout.h"    // GENERATED: the device tables

namespace breach_cuda {
namespace topo {

using namespace applier;
using namespace unit_tables;

// The launch's pointer table: device addresses of the (N, ...) planes, in
// applier_layout.h's APPLIER_PLANE_* order (plain C++: bindings.cpp fills it).
struct PlanePtrs {
    unsigned long long p[APPLIER_N_PLANES];
};

// One env's planes, typed. `n` = h * w; the gas planes are (n_gases, h, w).
struct TopoEnv {
    int h, w, n, n_gases, o2, n2;
    int8_t* material;
    int32_t* wall_hp;
    int32_t* gas;
    int64_t* gas_energy;
    int32_t* temperature;
    int32_t* fire;
    int32_t* atmosphere;
    bool* solid;
    bool* is_vacuum;
    bool* is_ambient;
    bool* thermal_solid;
    bool* flammable;
    bool* sight_screen;
    int32_t* heat_inv_shift;
    int32_t* face_shift;        // (h, w, 4)
    uint32_t* conductivity;     // float32 bits
    uint32_t* permeability;     // float32 bits
    uint32_t* wave_absorb;      // float32 bits
    uint32_t* heat_atten;       // float32 bits
    int32_t* heat_atten_q;
    int32_t* dyn_heat_atten_q;
    uint32_t* light_atten;      // (h, w, 3) float32 bits
    int32_t* light_atten_q;     // (h, w, 3)
    int32_t* dyn_light_atten_q; // (h, w, 3)
    int64_t* fuel_recip;
    int32_t* fire_T_ext_plane;
    int32_t* fuel_per_o2_plane;
    // #29 P6b.1: seal_tiles' solver-owned planes, the float ripple pair (bit
    // patterns) and the water depth (read only).
    int32_t* wave_p;
    int32_t* wind_x;
    int32_t* wind_y;
    int32_t* flow_vx;
    int32_t* flow_vy;
    uint32_t* ripple;           // float32 bits
    uint32_t* ripple_v;         // float32 bits
    int32_t* water_depth;
    // #29 P7c.2: the blast's pending divergence source (Q16 1/s; the strength
    // path's start_source / start_expansion and the SOURCES stage write it).
    int32_t* div_source;
    // #29 P7e.1: the env's TICK ROW tile-change count (GameMap._tile_serial's
    // device form), bumped by on_tile_changed; null: no tick row handed.
    int64_t* tiles = nullptr;

    __device__ int32_t& g(int slice, int i) const { return gas[(size_t)slice * n + i]; }
    __device__ int64_t bulk(int i) const { return (int64_t)g(o2, i) + (int64_t)g(n2, i); }
};

template <class T>
__device__ __forceinline__ T* env_plane(const PlanePtrs& P, int k, int env, size_t per_env) {
    return reinterpret_cast<T*>(P.p[k]) + (size_t)env * per_env;
}

// Env `env`'s view of the pointer table (the (N, ...) rebase, as bptr does).
__device__ inline TopoEnv env_view(const PlanePtrs& P, int env, int h, int w, int n_gases,
                                   const int64_t* tab) {
    TopoEnv E;
    E.h = h; E.w = w; E.n = h * w; E.n_gases = n_gases;
    E.o2 = (int)tab[UNIT_TAB_GAS_O2];
    E.n2 = (int)tab[UNIT_TAB_GAS_N2];
    const size_t n = (size_t)E.n;
    E.material = env_plane<int8_t>(P, APPLIER_PLANE_MATERIAL, env, n);
    E.wall_hp = env_plane<int32_t>(P, APPLIER_PLANE_WALL_HP, env, n);
    E.gas = env_plane<int32_t>(P, APPLIER_PLANE_GAS, env, n * (size_t)n_gases);
    E.gas_energy = env_plane<int64_t>(P, APPLIER_PLANE_GAS_ENERGY, env, n);
    E.temperature = env_plane<int32_t>(P, APPLIER_PLANE_TEMPERATURE, env, n);
    E.fire = env_plane<int32_t>(P, APPLIER_PLANE_FIRE, env, n);
    E.atmosphere = env_plane<int32_t>(P, APPLIER_PLANE_ATMOSPHERE, env, n);
    E.solid = env_plane<bool>(P, APPLIER_PLANE_SOLID, env, n);
    E.is_vacuum = env_plane<bool>(P, APPLIER_PLANE_IS_VACUUM, env, n);
    E.is_ambient = env_plane<bool>(P, APPLIER_PLANE_IS_AMBIENT, env, n);
    E.thermal_solid = env_plane<bool>(P, APPLIER_PLANE_THERMAL_SOLID, env, n);
    E.flammable = env_plane<bool>(P, APPLIER_PLANE_FLAMMABLE, env, n);
    E.sight_screen = env_plane<bool>(P, APPLIER_PLANE_SIGHT_SCREEN, env, n);
    E.heat_inv_shift = env_plane<int32_t>(P, APPLIER_PLANE_HEAT_INV_SHIFT, env, n);
    E.face_shift = env_plane<int32_t>(P, APPLIER_PLANE_FACE_SHIFT, env, 4 * n);
    E.conductivity = env_plane<uint32_t>(P, APPLIER_PLANE_CONDUCTIVITY, env, n);
    E.permeability = env_plane<uint32_t>(P, APPLIER_PLANE_PERMEABILITY, env, n);
    E.wave_absorb = env_plane<uint32_t>(P, APPLIER_PLANE_WAVE_ABSORB, env, n);
    E.heat_atten = env_plane<uint32_t>(P, APPLIER_PLANE_HEAT_ATTEN, env, n);
    E.heat_atten_q = env_plane<int32_t>(P, APPLIER_PLANE_HEAT_ATTEN_Q, env, n);
    E.dyn_heat_atten_q = env_plane<int32_t>(P, APPLIER_PLANE_DYN_HEAT_ATTEN_Q, env, n);
    E.light_atten = env_plane<uint32_t>(P, APPLIER_PLANE_LIGHT_ATTEN, env, 3 * n);
    E.light_atten_q = env_plane<int32_t>(P, APPLIER_PLANE_LIGHT_ATTEN_Q, env, 3 * n);
    E.dyn_light_atten_q = env_plane<int32_t>(P, APPLIER_PLANE_DYN_LIGHT_ATTEN_Q, env, 3 * n);
    E.fuel_recip = env_plane<int64_t>(P, APPLIER_PLANE_FUEL_RECIP, env, n);
    E.fire_T_ext_plane = env_plane<int32_t>(P, APPLIER_PLANE_FIRE_T_EXT_PLANE, env, n);
    E.fuel_per_o2_plane = env_plane<int32_t>(P, APPLIER_PLANE_FUEL_PER_O2_PLANE, env, n);
    E.wave_p = env_plane<int32_t>(P, APPLIER_PLANE_WAVE_P, env, n);
    E.wind_x = env_plane<int32_t>(P, APPLIER_PLANE_WIND_X, env, n);
    E.wind_y = env_plane<int32_t>(P, APPLIER_PLANE_WIND_Y, env, n);
    E.flow_vx = env_plane<int32_t>(P, APPLIER_PLANE_FLOW_VX, env, n);
    E.flow_vy = env_plane<int32_t>(P, APPLIER_PLANE_FLOW_VY, env, n);
    E.ripple = env_plane<uint32_t>(P, APPLIER_PLANE_RIPPLE, env, n);
    E.ripple_v = env_plane<uint32_t>(P, APPLIER_PLANE_RIPPLE_V, env, n);
    E.water_depth = env_plane<int32_t>(P, APPLIER_PLANE_WATER_DEPTH, env, n);
    E.div_source = env_plane<int32_t>(P, APPLIER_PLANE_DIV_SOURCE, env, n);
    return E;
}

// GameMap._FACE_DIRS: N, S, E, W (the opposite of d is d ^ 1).
__device__ __forceinline__ int face_dy(int d) { return d == 0 ? -1 : (d == 1 ? 1 : 0); }
__device__ __forceinline__ int face_dx(int d) { return d == 2 ? 1 : (d == 3 ? -1 : 0); }
// destroy_wall's 4-neighbour order ((-1, 0), (1, 0), (0, -1), (0, 1)).
__device__ __forceinline__ int nb_dy(int k) { return k == 0 ? -1 : (k == 1 ? 1 : 0); }
__device__ __forceinline__ int nb_dx(int k) { return k == 2 ? -1 : (k == 3 ? 1 : 0); }

__device__ __forceinline__ bool in_grid(const TopoEnv& E, int y, int x) {
    return 0 <= y && y < E.h && 0 <= x && x < E.w;
}

// GameMap.gas_is_accountable (the seam's one predicate, gas_energy.h).
__device__ __forceinline__ bool accountable(const TopoEnv& E, int i) {
    return gas_energy::accountable(E.solid, E.thermal_solid, E.is_vacuum, E.is_ambient, i);
}

// GameMap.gas_energy_refresh_cell.
__device__ __forceinline__ void refresh_cell(const TopoEnv& E, int i, int64_t t_amb) {
    if (!accountable(E, i)) return;
    E.temperature[i] = gas_energy::mirror_sat_q(E.gas_energy[i], E.bulk(i), t_amb);
}

// GameMap._solid_books_cell: cap_real * T on a thermal solid (the engine's
// one capacity law, conduction::cell_capacity_q), else 0. False when the
// product leaves int64 (a FAULT: Python's ints would hold it).
__device__ __forceinline__ bool solid_books_cell(const TopoEnv& E, int i, int64_t* out) {
    if (!E.thermal_solid[i]) { *out = 0; return true; }
    int64_t cap_used = 0, cap_real = 0;
    conduction::cell_capacity_q(true, E.heat_inv_shift[i], 0, 0, 0, &cap_used, &cap_real);
    const int64_t t = (int64_t)E.temperature[i];
    const int64_t lo = (int64_t)((unsigned long long)cap_real * (unsigned long long)t);
    const int64_t hi = __mul64hi(cap_real, t);
    if (hi != (lo >> 63)) return false;
    *out = lo;
    return true;
}

// GameMap._patch_face_shift: this tile's four faces AND each in-grid
// neighbour's face pointing back (the symmetric table read both ways).
__device__ inline void patch_face_shift(const TopoEnv& E, const int64_t* tab, int fy, int fx) {
    const int M = UNIT_TAB_MAT_FLAMMABLE_W;          // the material count
    const int i = fy * E.w + fx;
    const int mi = (int)E.material[i];
    for (int d = 0; d < 4; ++d) {
        const int ny = fy + face_dy(d), nx = fx + face_dx(d);
        if (in_grid(E, ny, nx)) {
            const int j = ny * E.w + nx;
            const int mn = (int)E.material[j];
            E.face_shift[(size_t)i * 4 + d] = (int32_t)tab[UNIT_TAB_MAT_FACE_SHIFT + mi * M + mn];
            E.face_shift[(size_t)j * 4 + (d ^ 1)] =
                (int32_t)tab[UNIT_TAB_MAT_FACE_SHIFT + mn * M + mi];
        } else {
            E.face_shift[(size_t)i * 4 + d] = (int32_t)tab[UNIT_TAB_MAT_NO_FACE];
        }
    }
}

// GameMap.on_tile_changed, minus its host-only side (render_dirty, the sky
// mask's dirty flag, _tile_serial -- booked as tiles_changed): every
// table-derived plane of the tile patched from its (already reassigned)
// material, the two dynamic MAX raises, the face shifts, solid from the
// permeability, the sight screen; then the MEMBERSHIP DIFF through the gas
// energy seam (retire / born at ambient). The caller guarantees the tile's
// material is a table row (0 <= m < M).
__device__ inline void on_tile_changed(const TopoEnv& E, const int64_t* tab, int fy, int fx,
                                       int64_t* B) {
    if (!in_grid(E, fy, fx)) return;
    B[APPLIER_BOOK_TILES_CHANGED] += 1;
    if (E.tiles != nullptr) *E.tiles += 1;    // #29 P7e.1: the tick row's count
    const int i = fy * E.w + fx;
    const bool was_accountable = accountable(E, i);
    const int m = (int)E.material[i];
    for (int c = 0; c < 3; ++c)
        E.light_atten[(size_t)i * 3 + c] = (uint32_t)tab[UNIT_TAB_MAT_LIGHT_ATTEN_BITS + 3 * m + c];
    E.heat_atten[i] = (uint32_t)tab[UNIT_TAB_MAT_HEAT_ATTEN_BITS + m];
    E.heat_atten_q[i] = (int32_t)tab[UNIT_TAB_MAT_HEAT_ATTEN_Q + m];
    if (E.dyn_heat_atten_q[i] < E.heat_atten_q[i]) E.dyn_heat_atten_q[i] = E.heat_atten_q[i];
    for (int c = 0; c < 3; ++c) {
        const size_t k = (size_t)i * 3 + c;
        E.light_atten_q[k] = (int32_t)tab[UNIT_TAB_MAT_LIGHT_ATTEN_Q + 3 * m + c];
        if (E.dyn_light_atten_q[k] < E.light_atten_q[k]) E.dyn_light_atten_q[k] = E.light_atten_q[k];
    }
    E.flammable[i] = tab[UNIT_TAB_MAT_FLAMMABLE + m] != 0;
    E.wall_hp[i] = (int32_t)tab[UNIT_TAB_MAT_WALL_HP_Q + m];
    E.conductivity[i] = (uint32_t)tab[UNIT_TAB_MAT_CONDUCTIVITY_BITS + m];
    E.heat_inv_shift[i] = (int32_t)tab[UNIT_TAB_MAT_HEAT_INV_SHIFT + m];
    E.thermal_solid[i] = tab[UNIT_TAB_MAT_THERMAL_SOLID + m] != 0;
    E.fuel_recip[i] = tab[UNIT_TAB_MAT_FUEL_RECIP + m];
    E.fire_T_ext_plane[i] = (int32_t)tab[UNIT_TAB_MAT_FIRE_T_EXT_Q + m];
    E.fuel_per_o2_plane[i] = (int32_t)tab[UNIT_TAB_MAT_FUEL_PER_O2_Q + m];
    patch_face_shift(E, tab, fy, fx);
    E.permeability[i] = (uint32_t)tab[UNIT_TAB_MAT_PERMEABILITY_BITS + m];
    E.wave_absorb[i] = (uint32_t)tab[UNIT_TAB_MAT_WAVE_ABSORB_BITS + m];
    E.solid[i] = tab[UNIT_TAB_MAT_SOLID + m] != 0;
    E.sight_screen[i] = tab[UNIT_TAB_MAT_BLOCKS_SIGHT + m] != 0 && !E.solid[i];
    // THE MEMBERSHIP DIFF (arc #54 P-G1b).
    const bool now_accountable = accountable(E, i);
    if (was_accountable && !now_accountable) {
        B[APPLIER_BOOK_GE_RETIRE] -= gas_energy::retire(E.gas_energy, i);
    } else if (now_accountable && !was_accountable) {
        B[APPLIER_BOOK_GE_MINT] += gas_energy::born_at_ambient(
            E.gas_energy, E.temperature, i, E.bulk(i), tab[UNIT_TAB_TOPO_T_AMB_Q]);
    }
}

// GameMap.destroy_wall, the whole law, minus the host-only side. Returns true
// when it acted (a non-air tile in bounds). SERIAL by contract: the breach
// rule reads the neighbours an earlier destroy of the same walk changed.
__device__ inline bool destroy_wall(const TopoEnv& E, const int64_t* tab, int fy, int fx,
                                    int64_t* B) {
    if (!in_grid(E, fy, fx)) return false;
    const int i = fy * E.w + fx;
    const int64_t mat_air = tab[UNIT_TAB_TOPO_MAT_AIR];
    if ((int64_t)E.material[i] == mat_air) return false;
    const bool was_hull = (int64_t)E.material[i] == tab[UNIT_TAB_TOPO_MAT_HULL];
    const bool ambient_map = tab[UNIT_TAB_TOPO_BOUNDARY_AMBIENT] != 0;
    bool* breach_mask = ambient_map ? E.is_ambient : E.is_vacuum;
    const int64_t t_amb = tab[UNIT_TAB_TOPO_T_AMB_Q];
    const int64_t n_total_q = tab[UNIT_TAB_TOPO_AMB_N_TOTAL_Q];

    // --- the operands, read before any write (all or nothing) ---------------
    // The solid books' bracket (#73 item 2).
    int64_t solid_before = 0;
    if (!solid_books_cell(E, i, &solid_before)) { B[APPLIER_BOOK_FAULTS] += 1; return false; }
    // The breach rule's reads (on_tile_changed writes none of them: it changes
    // only this tile's planes and its neighbours' facing faces).
    const bool on_edge_hull = was_hull && (fy < 1 || fy >= E.h - 1 || fx < 1 || fx >= E.w - 1);
    bool exposes = false;
    for (int k = 0; k < 4; ++k) {
        const int ny = fy + nb_dy(k), nx = fx + nb_dx(k);
        if (in_grid(E, ny, nx)) {
            const int j = ny * E.w + nx;
            if (breach_mask[j] && !E.solid[j]) exposes = true;
        }
    }
    const bool breach = breach_mask[i] || on_edge_hull || exposes;
    // The donors (4-neighbours not solid and not vacuum; is_ambient counts),
    // their composition in the one exact form (P-M3 §3.1.3).
    uint64_t sum_o2 = 0, sum_n = 0;
    if (!breach) {
        for (int k = 0; k < 4; ++k) {
            const int ny = fy + nb_dy(k), nx = fx + nb_dx(k);
            if (!in_grid(E, ny, nx)) continue;
            const int j = ny * E.w + nx;
            if (E.solid[j] || E.is_vacuum[j]) continue;
            const int64_t d_o2 = E.g(E.o2, j), d_n2 = E.g(E.n2, j);
            if (d_o2 < 0 || d_n2 < 0) { B[APPLIER_BOOK_FAULTS] += 1; return false; }
            sum_o2 += (uint64_t)d_o2;
            sum_n += (uint64_t)d_o2 + (uint64_t)d_n2;
        }
    }
    int64_t o2_q = tab[UNIT_TAB_TOPO_AMB_O2_Q];
    if (!breach && sum_n > 0) {
        // round-half-up: (n_total * sum_o2 + sum_n // 2) // sum_n, exact in
        // 64 unsigned bits for the proved ranges (n_total < 2^31, a donor's
        // planes < 2^31 each); past them a FAULT.
        uint64_t hi = 0, lo = 0;
        fixedpoint::umul64_wide((uint64_t)n_total_q, sum_o2, &hi, &lo);
        const uint64_t num = lo + sum_n / 2;
        if (hi != 0 || num < lo) { B[APPLIER_BOOK_FAULTS] += 1; return false; }
        o2_q = (int64_t)(num / sum_n);
    }
    const int64_t n_before = E.bulk(i);

    // --- the law -------------------------------------------------------------
    E.material[i] = (int8_t)mat_air;
    on_tile_changed(E, tab, fy, fx, B);
    if (on_edge_hull || exposes) breach_mask[i] = true;
    if (breach_mask[i]) {
        // A tile that joined the boundary: its gas EVACUATED here (booked by
        // the bracket), its energy retired (a no-op when the membership diff
        // already retired it).
        E.g(E.o2, i) = 0;
        E.g(E.n2, i) = 0;
        B[APPLIER_BOOK_GE_RETIRE] -= gas_energy::retire(E.gas_energy, i);
    } else {
        E.g(E.o2, i) = (int32_t)o2_q;
        E.g(E.n2, i) = (int32_t)(n_total_q - o2_q);
        // RETIRE, then THE ONE TRUE MINT (born at ambient), then the mirror.
        B[APPLIER_BOOK_GE_DESTROY_SEED] -= gas_energy::retire(E.gas_energy, i);
        if (n_total_q > 0 && accountable(E, i)) {
            const int64_t de = gas_energy::minted(n_total_q, (int32_t)t_amb);
            E.gas_energy[i] += de;
            B[APPLIER_BOOK_GE_DESTROY_SEED] += de;
        }
        refresh_cell(E, i, t_amb);
    }
    E.atmosphere[i] = (int32_t)tab[UNIT_TAB_TOPO_AMB_PIN_Q];
    E.temperature[i] = 0;
    E.fire[i] = 0;
    // The two bookings, MEASURED and signed.
    B[APPLIER_BOOK_N_DESTRUCTION_SEED] += E.bulk(i) - n_before;
    int64_t solid_after = 0;
    solid_books_cell(E, i, &solid_after);     // air: 0 (no thermal solid)
    B[APPLIER_BOOK_SOLID_DESTROY_WALL] += solid_after - solid_before;
    return true;
}

// #29 P7d (moved from k_applier's CHEW, a pure refactor): combat.chew_wall
// with the damage already quantized per call (`live` false for a
// wall_damage <= 0: the no-op) -- a no-op on air; the wall's hp less the
// damage, destroy_wall at <= 0. Returns CHEW_NOOP, CHEW_APPLIED or CHEW_FAULT
// (a tile outside the grid, where numpy would wrap or raise; a new hp outside
// int32): nothing written on a fault. The applier's CHEW and slot 5's door
// bash both call it.
enum ChewResult { CHEW_NOOP = 0, CHEW_APPLIED = 1, CHEW_FAULT = 2 };

__device__ inline int chew_wall(const TopoEnv& E, const int64_t* tab, int64_t y, int64_t x,
                                int64_t damage_q, bool live, int64_t* B) {
    if (!live) return CHEW_NOOP;
    if (y < 0 || y >= E.h || x < 0 || x >= E.w) return CHEW_FAULT;
    const int i = (int)y * E.w + (int)x;
    if ((int64_t)E.material[i] == tab[UNIT_TAB_TOPO_MAT_AIR]) return CHEW_NOOP;
    const int64_t hp = (int64_t)E.wall_hp[i] - damage_q;
    if (hp < -2147483648LL || hp > 2147483647LL) return CHEW_FAULT;   // int32's range
    E.wall_hp[i] = (int32_t)hp;
    if (hp <= 0) destroy_wall(E, tab, (int)y, (int)x, B);
    return CHEW_APPLIED;
}

// ============================================================================
// #29 P6b.1 -- SEAL / UNSEAL / CAN_SEAL, THE GAS-N PRIMITIVES and
// _APERTURE_BLOCKED as device functions (design 4.4c, the cut table's P6b.1
// row): GameMap.seal_tiles / unseal_tiles / can_seal_tiles, inject_gas_n /
// extract_gas_n_vec (extract_gas_n is its bulk sum) / transfer_gas_n and
// vent_system._aperture_blocked, integer for integer, on ONE env's planes.
// The door sweep (cuda_unit_doors.cuh, the applier's DOORS stage) is their
// first caller; P6b.2's vents and P6b.3's pumps call the primitives, never a
// copy. The oracle is src/simulation/gamemap.py (vent_system.py for the
// aperture); tests/cuda_unit_doors_check.py (part PROBE) holds them equal.
//
// A SPAN is `n` (y, x) pairs at `tiles[i * stride]`, `tiles[i * stride + 1]`,
// ROW-MAJOR SORTED (GameMap._normalize_span's order: the caller normalizes;
// a door's span is stored so). A span naming a tile twice is refused
// (duplicate_tile, the normalizer's ValueError); an unsorted span is a FAULT
// (the caller's duty); a span longer than SPAN_MAX is refused (span_too_long).
//
// THE REFUSAL PATH: every exception the Python raises BEFORE it mutates is a
// code (applier_layout.h's APPLIER_SEAM_*, simulation/applier.SEAM_REFUSALS,
// in _seal_blockers' order) and the function writes NOTHING. A FAULT -- an
// operand past the proved range, where Python's ints would carry on -- is
// found in a pre-pass and also writes nothing. Proved range (pre-checked on
// every tile the call reads or writes): gas planes >= 0 (a negative trace
// would raise mid-mutation in Python), |gas_energy| < 2^61, T_AMB < 2^28, a
// solid book within int64, and for seal_tiles / unseal_tiles THE TERM BOUND
// below on every cell whose gas moves (a sealed tile, an unseal's donor). The
// gas-N primitives (inject / extract / transfer) keep their own |T_abs| <
// 2^28 (T_BOUND): the vents' proof (cuda_unit_vents.cuh) leans on the
// extraction's.
//
// THE TERM BOUND (#29 TOPOLOGY RANGE; it replaced a donor's |T_abs| < 2^28,
// which refused blast-heated gas the physics holds: ship_demo, run seed 81,
// tick 32, T_abs 324 247 500 beside a door opened by a use). A moved parcel's
// energy is dN * T_abs with dN <= the moving cell's bulk N. TEMPERATURE IS
// NOT BOUNDED: T is whatever the int32 plane holds, [-2^31, 2^31) (a seam
// write refreshes the mirror without the rails, so between recoveries a
// mirror may sit above T_MAX_PHYS: only the dtype's range holds always), and
// |T_AMB| < 2^28, so |T_abs| < 2^31 + 2^28. The refusal is on the PRODUCT:
// a moving cell must have
//     N * (|T_abs| + 1) < TERM_BOUND = 2^60,
// computed exactly in uint64 (N < 2^32 -- two int32 planes, >= 0 -- times
// |T_abs| + 1 <= 2^31 + 2^28 is < 2^63.2 < 2^64). Every term a cell gives in
// the call is then < 2^60 (the unseal's re-read donor: see below), a cell
// takes at most four terms in and its own out, and so every store and sum
// fits int64: |E| < 2^61 plus four terms < 2^60 is < 2^63; a born-at-ambient
// E is N * T_AMB < 2^32 * 2^28 = 2^60. No expression is restated: inside the
// range each is the same int64 arithmetic, exact against Python's ints.
// WHERE THE PHYSICS SITS: after the recovery's rails |T_abs| <= (T_MAX_PHYS +
// T_AMB_K) raw = (16000 + 290) * 2^16 < 2^30, and the engine's own per-cell
// range is n_bulk < 2^30 (bulk_transport.h, design 2.1.2), so N * (|T_abs| +
// 1) < 2^60 there (T_ABS_PHYS_MAX below, asserted); a mirror transiently past
// the ceiling, up to the int32 top, still passes with N < 2^28.8 (about 450 x
// the ambient bulk at 1/3 m tiles).
// THE RE-READ DONOR: unseal reads a donor's T_abs live at each span tile it
// borders, and refreshes its mirror after each drain, which can RAISE it (by
// floor(r / N') of the remainder r of E / N). When the donor's mirror IS its
// energy (T == mirror_sat(E, N), the iron rule's state), each later term is
// < N0 * |T_abs0| + r0 < N0 * (|T_abs0| + 1), N0 / T_abs0 / r0 its pre-call
// values (the remainder only shrinks; a saturated mirror stays saturated), so
// the one product bounds every read. A donor that borders two or more span
// tiles, is accountable and is NOT a mirror of its energy is bounded by the
// plane's own range instead: N0 * (2^31 + |T_AMB| + 1) < 2^60. (The old proof
// let such a donor through with any N < 2^32: a later read could overflow.)
// Each book's running sum over many calls stays the books' own int64
// contract, as before.
//
// FLOATS: the ripple pair is written as the bit pattern of 0.0f (0u); the
// seal's `permeability > 0.0` compare is the seal_mat_ok table row (made once
// on the host from the same float the law reads). No float arithmetic here.
//
// BOOKS: each move books where gas_energy_move books (the call's channel, or
// its diag_*_from / _to_nonacct twin), each clip of the counted trace
// saturation trace_clip_<g>, the solid books seal / unseal; on_tile_changed's
// own books as ever.
// ============================================================================

//: The longest span a seal / unseal takes (the door tile side table's capacity).
constexpr int SPAN_MAX = object_tables::OBJ_DOOR_TILE_CAP;
constexpr int64_t E_BOUND = (int64_t)1 << 61;
constexpr int64_t T_BOUND = (int64_t)1 << 28;    // T_AMB; the gas-N primitives' T_abs
constexpr int64_t I32_LIMIT = (int64_t)1 << 31;   // int32 stores
//: THE TERM BOUND of seal / unseal: a moving cell's N * (|T_abs| + 1) below it.
constexpr uint64_t TERM_BOUND = (uint64_t)1 << 60;
//: The physics' ceiling in the seam's currency, (T_MAX_PHYS + T_AMB_K) raw at
//: the current dials (config.toml [physics.thermal] T_MAX_PHYS = 16000,
//: PROVISIONAL; T_AMB_K = 290): used only for the assertion that a cell at the
//: ceiling with the engine's largest bulk (2^30) is inside the term bound. If
//: the dial moves, this assertion is where the margin is re-read.
constexpr uint64_t T_ABS_PHYS_MAX = (uint64_t)(16000 + 290) << 16;
static_assert(((uint64_t)1 << 30) * (T_ABS_PHYS_MAX + 1) < TERM_BOUND,
              "a cell at the physics' ceiling with n_bulk 2^30 is inside the term bound");

static_assert(APPLIER_BOOK_TRACE_CLIP_1 == APPLIER_BOOK_TRACE_CLIP_0 + 1,
              "the trace clip books are one run, a slice each");

//: The book channels of one gas_energy_move caller (GameMap.gas_energy_move's
//: channel and its two diagnostic twins).
struct MoveBooks {
    int ch, from_nonacct, to_nonacct;
};
__device__ __forceinline__ MoveBooks seal_books() {
    return MoveBooks{APPLIER_BOOK_GE_SEAL, APPLIER_BOOK_GE_DIAG_SEAL_FROM,
                     APPLIER_BOOK_GE_DIAG_SEAL_TO};
}
__device__ __forceinline__ MoveBooks unseal_books() {
    return MoveBooks{APPLIER_BOOK_GE_UNSEAL, APPLIER_BOOK_GE_DIAG_UNSEAL_FROM,
                     APPLIER_BOOK_GE_DIAG_UNSEAL_TO};
}
__device__ __forceinline__ MoveBooks transfer_books() {
    return MoveBooks{APPLIER_BOOK_GE_PUMP_TRANSFER, APPLIER_BOOK_GE_DIAG_PUMP_TRANSFER_FROM,
                     APPLIER_BOOK_GE_DIAG_PUMP_TRANSFER_TO};
}

// GameMap._trace_slice.
__device__ __forceinline__ bool trace_slice(const int64_t* tab, int g) {
    return tab[UNIT_TAB_GAS_TRACE + g] != 0;
}

// GameMap.gas_t_abs_at.
__device__ __forceinline__ int64_t t_abs_at(const TopoEnv& E, const int64_t* tab, int i) {
    return (int64_t)E.temperature[i] + tab[UNIT_TAB_TOPO_T_AMB_Q];
}

// GameMap.gas_energy_move (the caller refreshes the mirrors).
__device__ inline void energy_move(const TopoEnv& E, int src, int dst, int64_t dn,
                                   int64_t t_abs, int64_t* B, MoveBooks mb) {
    if (dn <= 0) return;
    const int64_t de = dn * t_abs;
    if (accountable(E, src)) {
        E.gas_energy[src] -= de;
        B[mb.ch] -= de;
    } else {
        B[mb.from_nonacct] += de;
    }
    if (accountable(E, dst)) {
        E.gas_energy[dst] += de;
        B[mb.ch] += de;
    } else {
        B[mb.to_nonacct] += de;
    }
}

// GameMap.trace_store_sat on slice g of cell i (the clip booked; a negative
// result was refused by the caller's pre-pass).
__device__ __forceinline__ void trace_store_sat(const TopoEnv& E, int g, int i, int64_t add,
                                                int64_t* B) {
    int64_t clip = 0;
    E.g(g, i) = fixedpoint::trace_add_sat(E.g(g, i), add, &clip);
    B[APPLIER_BOOK_TRACE_CLIP_0 + g] += clip;
}

// A span tile's coordinates as given (int64: compared before any cast).
__device__ __forceinline__ int64_t span_y(const int64_t* t, int stride, int i) { return t[(size_t)i * stride]; }
__device__ __forceinline__ int64_t span_x(const int64_t* t, int stride, int i) { return t[(size_t)i * stride + 1]; }
__device__ __forceinline__ bool in_grid64(const TopoEnv& E, int64_t y, int64_t x) {
    return 0 <= y && y < E.h && 0 <= x && x < E.w;
}

__device__ inline bool in_span(const int64_t* t, int stride, int n, int64_t y, int64_t x) {
    for (int i = 0; i < n; ++i)
        if (span_y(t, stride, i) == y && span_x(t, stride, i) == x) return true;
    return false;
}

// GameMap._normalize_span's checks on a span the caller handed sorted: the
// length bound, a repeated tile (ValueError), the order (a FAULT); 0 = good.
// Coordinates past int32 are a FAULT (an int cast would wrap).
__device__ inline int64_t span_shape(const int64_t* t, int stride, int n) {
    if (n < 0) return APPLIER_SEAM_FAULT;
    if (n > SPAN_MAX) return APPLIER_SEAM_SPAN_TOO_LONG;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (span_y(t, stride, i) == span_y(t, stride, j) &&
                span_x(t, stride, i) == span_x(t, stride, j))
                return APPLIER_SEAM_DUPLICATE_TILE;
    for (int i = 1; i < n; ++i) {
        const int64_t py = span_y(t, stride, i - 1), px = span_x(t, stride, i - 1);
        const int64_t y = span_y(t, stride, i), x = span_x(t, stride, i);
        if (py > y || (py == y && px >= x)) return APPLIER_SEAM_FAULT;   // not row-major
    }
    return 0;
}

// GameMap._seal_receivers: the open 4-neighbours of (fy, fx) not in the span,
// N, S, E, W; returns their count, their linear indices in `rcv`.
__device__ inline int seal_receivers(const TopoEnv& E, const int64_t* t, int stride, int n,
                                     int fy, int fx, int* rcv) {
    int k = 0;
    for (int d = 0; d < 4; ++d) {
        const int ny = fy + face_dy(d), nx = fx + face_dx(d);
        if (in_grid(E, ny, nx) && !E.solid[ny * E.w + nx] && !in_span(t, stride, n, ny, nx))
            rcv[k++] = ny * E.w + nx;
    }
    return k;
}

// GameMap._seal_blockers on a normalized span: the checks in their pinned
// order, the first refusal's code (0 = the seal would succeed). `mid` < 0:
// no material argument (can_seal_tiles). Reads only.
__device__ inline int64_t seal_blockers(const TopoEnv& E, const int64_t* tab, const int64_t* t,
                                        int stride, int n, int64_t mid, bool has_mid) {
    for (int i = 0; i < n; ++i)                                     // 1. bounds
        if (!in_grid64(E, span_y(t, stride, i), span_x(t, stride, i)))
            return APPLIER_SEAM_OUT_OF_BOUNDS;
    for (int i = 0; i < n; ++i)                                     // 2. already solid
        if (E.solid[(int)span_y(t, stride, i) * E.w + (int)span_x(t, stride, i)])
            return APPLIER_SEAM_ALREADY_SOLID;
    if (has_mid) {                                                  // 3. the material
        if (mid < 0 || mid >= (int64_t)UNIT_TAB_SEAL_MAT_OK_W) return APPLIER_SEAM_UNKNOWN_MATERIAL;
        if (tab[UNIT_TAB_SEAL_MAT_OK + mid] == 0) return APPLIER_SEAM_MATERIAL_NOT_SOLID;
    }
    for (int i = 0; i < n; ++i)                                     // 4. water rule v1
        if (E.water_depth[(int)span_y(t, stride, i) * E.w + (int)span_x(t, stride, i)] != 0)
            return APPLIER_SEAM_STANDING_WATER;
    int rcv[4];
    for (int i = 0; i < n; ++i) {                                   // 5 + 6. sealed pocket
        const int fy = (int)span_y(t, stride, i), fx = (int)span_x(t, stride, i);
        if (seal_receivers(E, t, stride, n, fy, fx, rcv) != 0) continue;
        for (int g = 0; g < E.n_gases; ++g)
            if (E.g(g, fy * E.w + fx) != 0) return APPLIER_SEAM_SEALED_POCKET;
    }
    // 7. the overflow pre-check, bulk slices only: each receiver's own count
    // plus every adjacent span tile's whole load (the generous bound).
    for (int g = 0; g < E.n_gases; ++g) {
        if (trace_slice(tab, g)) continue;
        for (int i = 0; i < n; ++i) {
            const int k = seal_receivers(E, t, stride, n, (int)span_y(t, stride, i),
                                         (int)span_x(t, stride, i), rcv);
            for (int a = 0; a < k; ++a) {
                const int r = rcv[a];
                int64_t bound = E.g(g, r);
                const int ry = r / E.w, rx = r % E.w;
                for (int d = 0; d < 4; ++d) {        // the span tiles r receives from
                    const int sy = ry + face_dy(d), sx = rx + face_dx(d);
                    if (in_grid(E, sy, sx) && in_span(t, stride, n, sy, sx))
                        bound += E.g(g, sy * E.w + sx);
                }
                if (bound >= I32_LIMIT) return APPLIER_SEAM_OVERFLOW;
            }
        }
    }
    return 0;
}

// GameMap.can_seal_tiles: 0 when seal_tiles would succeed for a valid solid
// material, else the refusal (span shape first, as the normalizer raises).
__device__ inline int64_t can_seal(const TopoEnv& E, const int64_t* tab, const int64_t* t,
                                   int stride, int n) {
    const int64_t shape = span_shape(t, stride, n);
    if (shape != 0) return shape;
    return seal_blockers(E, tab, t, stride, n, -1, false);
}

// The solid books of a cell as it WILL be: cap_real(material m) * T on a
// thermal-solid m (the pre-pass's check of the product); false past int64.
__device__ inline bool solid_books_as(const int64_t* tab, int64_t m, int64_t T, int64_t* out) {
    if (tab[UNIT_TAB_MAT_THERMAL_SOLID + m] == 0) { *out = 0; return true; }
    int64_t cap_used = 0, cap_real = 0;
    conduction::cell_capacity_q(true, (int32_t)tab[UNIT_TAB_MAT_HEAT_INV_SHIFT + m], 0, 0, 0,
                                &cap_used, &cap_real);
    const int64_t lo = (int64_t)((unsigned long long)cap_real * (unsigned long long)T);
    const int64_t hi = __mul64hi(cap_real, T);
    if (hi != (lo >> 63)) return false;
    *out = lo;
    return true;
}

__device__ __forceinline__ bool e_ok(const TopoEnv& E, int i) {
    return E.gas_energy[i] > -E_BOUND && E.gas_energy[i] < E_BOUND;
}
__device__ __forceinline__ bool t_ok(int64_t t_abs) { return t_abs > -T_BOUND && t_abs < T_BOUND; }
// THE TERM BOUND on a moving cell of bulk n (>= 0, < 2^32) whose |T_abs| is
// at most `t_mag` (< 2^31 + 2^28): n * (t_mag + 1) < 2^60, exact in uint64.
__device__ __forceinline__ bool term_ok(int64_t n, uint64_t t_mag) {
    return (uint64_t)n * (t_mag + 1) < TERM_BOUND;
}
__device__ __forceinline__ uint64_t mag64(int64_t v) {
    return v < 0 ? (uint64_t)0 - (uint64_t)v : (uint64_t)v;
}
// Is cell i's mirror its energy (what gas_energy_refresh_cell would write)?
__device__ inline bool is_mirror(const TopoEnv& E, int i, int64_t t_amb) {
    const int64_t n = E.bulk(i);
    if (n < gas_energy::N_EPS_RAW) return E.temperature[i] == 0;
    return E.temperature[i] == gas_energy::mirror_sat_q(E.gas_energy[i], n, t_amb);
}
__device__ inline bool gas_nonneg(const TopoEnv& E, int i) {
    for (int g = 0; g < E.n_gases; ++g)
        if (E.g(g, i) < 0) return false;
    return true;
}

// GameMap.seal_tiles(span, mid): 0 done, else the refusal (nothing written).
// SERIAL: each tile's evacuation reads the receivers the previous tile wrote.
__device__ inline int64_t seal_tiles(const TopoEnv& E, const int64_t* tab, const int64_t* t,
                                     int stride, int n, int64_t mid, int64_t* B) {
    const int64_t shape = span_shape(t, stride, n);
    if (shape != 0) return shape;
    const int64_t why = seal_blockers(E, tab, t, stride, n, mid, true);
    if (why != 0) return why;
    const int64_t t_amb = tab[UNIT_TAB_TOPO_T_AMB_Q];
    if (!t_ok(t_amb)) return APPLIER_SEAM_FAULT;
    // --- the pre-pass: the close-T means (PRE-call thermal solids: a span
    // member closed earlier in this call never donates) and the proved range
    int64_t close_t[SPAN_MAX];
    bool has_close[SPAN_MAX];
    int rcv[4];
    for (int i = 0; i < n; ++i) {
        const int fy = (int)span_y(t, stride, i), fx = (int)span_x(t, stride, i);
        const int c = fy * E.w + fx;
        int64_t sum = 0;
        int cnt = 0;
        for (int d = 0; d < 4; ++d) {
            const int ny = fy + face_dy(d), nx = fx + face_dx(d);
            if (in_grid(E, ny, nx) && E.thermal_solid[ny * E.w + nx]) {
                sum += (int64_t)E.temperature[ny * E.w + nx];
                ++cnt;
            }
        }
        has_close[i] = cnt > 0;
        close_t[i] = cnt > 0 ? fixedpoint::floordiv_q(sum, (int64_t)cnt) : 0;
        // the term bound (a sealed tile is read once: its gas and T do not
        // change before its turn; the shares sum to at most its bulk)
        if (!gas_nonneg(E, c) || !e_ok(E, c) ||
            !term_ok(E.bulk(c), mag64(t_abs_at(E, tab, c))))
            return APPLIER_SEAM_FAULT;
        int64_t sb = 0;
        if (!solid_books_cell(E, c, &sb)) return APPLIER_SEAM_FAULT;
        const int64_t t_end = has_close[i] ? close_t[i] : (int64_t)E.temperature[c];
        if (!solid_books_as(tab, mid, t_end, &sb)) return APPLIER_SEAM_FAULT;
        const int k = seal_receivers(E, t, stride, n, fy, fx, rcv);
        for (int a = 0; a < k; ++a)
            if (!gas_nonneg(E, rcv[a]) || !e_ok(E, rcv[a])) return APPLIER_SEAM_FAULT;
    }
    // --- the law (raise-free from here: GameMap's atomicity pin)
    const MoveBooks mb = seal_books();
    for (int i = 0; i < n; ++i) {
        const int fy = (int)span_y(t, stride, i), fx = (int)span_x(t, stride, i);
        const int c = fy * E.w + fx;
        const int k = seal_receivers(E, t, stride, n, fy, fx, rcv);
        const int64_t t_abs_seal = t_abs_at(E, tab, c);
        int64_t solid_before = 0;
        solid_books_cell(E, c, &solid_before);
        int64_t bulk_share[4] = {0, 0, 0, 0};
        for (int g = 0; g < E.n_gases; ++g) {
            const int64_t nn = E.g(g, c);
            if (nn == 0) continue;
            const int64_t q = nn / k, r = nn % k;          // nn >= 0, k >= 1 (the pocket rule)
            for (int j = 0; j < k; ++j) {
                const int64_t share = q + (j < r ? 1 : 0);
                if (share == 0) continue;
                if (trace_slice(tab, g)) {
                    trace_store_sat(E, g, rcv[j], share, B);
                } else {
                    E.g(g, rcv[j]) = (int32_t)((int64_t)E.g(g, rcv[j]) + share);
                }
                if (g == E.o2 || g == E.n2) bulk_share[j] += share;
            }
            E.g(g, c) = 0;
        }
        for (int j = 0; j < k; ++j) energy_move(E, c, rcv[j], bulk_share[j], t_abs_seal, B, mb);
        for (int j = 0; j < k; ++j) refresh_cell(E, rcv[j], t_amb);
        E.material[c] = (int8_t)mid;
        on_tile_changed(E, tab, fy, fx, B);
        E.atmosphere[c] = 0;
        E.wave_p[c] = 0;
        E.wind_x[c] = 0;
        E.wind_y[c] = 0;
        E.flow_vx[c] = 0;
        E.flow_vy[c] = 0;
        E.ripple[c] = 0u;                                  // 0.0f
        E.ripple_v[c] = 0u;                                // 0.0f
        if (has_close[i]) E.temperature[c] = (int32_t)close_t[i];
        int64_t solid_after = 0;
        solid_books_cell(E, c, &solid_after);
        B[APPLIER_BOOK_SOLID_SEAL] += solid_after - solid_before;
    }
    return 0;
}

// GameMap.unseal_tiles(span): 0 done, else the refusal (nothing written).
// SERIAL: the boundary join reads the LIVE solid mask (it chains down the
// span) and a donor's holdings as an earlier tile's withdrawal left them.
__device__ inline int64_t unseal_tiles(const TopoEnv& E, const int64_t* tab, const int64_t* t,
                                       int stride, int n, int64_t* B) {
    const int64_t shape = span_shape(t, stride, n);
    if (shape != 0) return shape;
    for (int i = 0; i < n; ++i)
        if (!in_grid64(E, span_y(t, stride, i), span_x(t, stride, i)))
            return APPLIER_SEAM_OUT_OF_BOUNDS;
    for (int i = 0; i < n; ++i)
        if (!E.solid[(int)span_y(t, stride, i) * E.w + (int)span_x(t, stride, i)])
            return APPLIER_SEAM_NOT_SOLID;
    const int64_t t_amb = tab[UNIT_TAB_TOPO_T_AMB_Q];
    const int64_t mat_air = tab[UNIT_TAB_TOPO_MAT_AIR];
    if (!t_ok(t_amb) || tab[UNIT_TAB_MAT_THERMAL_SOLID + mat_air] != 0) return APPLIER_SEAM_FAULT;
    // --- the pre-pass (the proved range on every cell the call touches)
    for (int i = 0; i < n; ++i) {
        const int fy = (int)span_y(t, stride, i), fx = (int)span_x(t, stride, i);
        const int c = fy * E.w + fx;
        int64_t sb = 0;
        if (!gas_nonneg(E, c) || !e_ok(E, c) || !solid_books_cell(E, c, &sb))
            return APPLIER_SEAM_FAULT;
        for (int d = 0; d < 4; ++d) {
            const int ny = fy + face_dy(d), nx = fx + face_dx(d);
            if (!in_grid(E, ny, nx) || in_span(t, stride, n, ny, nx)) continue;
            const int j = ny * E.w + nx;
            if (E.solid[j]) continue;
            // the term bound on a donor (THE RE-READ DONOR in the header)
            if (!gas_nonneg(E, j) || !e_ok(E, j) ||
                !term_ok(E.bulk(j), mag64(t_abs_at(E, tab, j))))
                return APPLIER_SEAM_FAULT;
            int reads = 0;                                 // the span tiles j borders
            for (int e = 0; e < 4; ++e)
                if (in_grid(E, ny + face_dy(e), nx + face_dx(e)) &&
                    in_span(t, stride, n, ny + face_dy(e), nx + face_dx(e)))
                    ++reads;
            if (reads >= 2 && accountable(E, j) && !is_mirror(E, j, t_amb) &&
                !term_ok(E.bulk(j), ((uint64_t)1 << 31) + mag64(t_amb)))
                return APPLIER_SEAM_FAULT;
        }
    }
    // --- the law
    const bool ambient_map = tab[UNIT_TAB_TOPO_BOUNDARY_AMBIENT] != 0;
    bool* breach_mask = ambient_map ? E.is_ambient : E.is_vacuum;
    const MoveBooks mb = unseal_books();
    for (int i = 0; i < n; ++i) {
        const int fy = (int)span_y(t, stride, i), fx = (int)span_x(t, stride, i);
        const int c = fy * E.w + fx;
        int64_t solid_before = 0, solid_after = 0;
        solid_books_cell(E, c, &solid_before);
        E.material[c] = (int8_t)mat_air;
        on_tile_changed(E, tab, fy, fx, B);
        solid_books_cell(E, c, &solid_after);
        B[APPLIER_BOOK_SOLID_UNSEAL] += solid_after - solid_before;
        // the boundary join (the LIVE solid mask)
        bool joins = breach_mask[c];
        for (int d = 0; d < 4 && !joins; ++d) {
            const int ny = fy + face_dy(d), nx = fx + face_dx(d);
            if (in_grid(E, ny, nx) && breach_mask[ny * E.w + nx] && !E.solid[ny * E.w + nx])
                joins = true;
        }
        if (joins) {
            breach_mask[c] = true;
            continue;
        }
        // the donors: PRE-call open (a non-span neighbour's solidity never
        // changes in this call), not vacuum, not in the span; N, S, E, W
        int don[4];
        int k = 0;
        for (int d = 0; d < 4; ++d) {
            const int ny = fy + face_dy(d), nx = fx + face_dx(d);
            if (!in_grid(E, ny, nx)) continue;
            const int j = ny * E.w + nx;
            if (!E.solid[j] && !E.is_vacuum[j] && !in_span(t, stride, n, ny, nx)) don[k++] = j;
        }
        if (k == 0) continue;                              // opens empty: never mint
        int64_t t_abs_d[4];
        for (int j = 0; j < k; ++j) t_abs_d[j] = t_abs_at(E, tab, don[j]);
        int64_t bulk_take[4] = {0, 0, 0, 0};
        for (int g = 0; g < E.n_gases; ++g) {
            int64_t avail[4], take[4], sum = 0;
            for (int j = 0; j < k; ++j) {
                avail[j] = E.g(g, don[j]);
                sum += avail[j];
            }
            const int64_t target = sum / (k + 1);          // sum >= 0 (the pre-pass)
            if (target == 0) continue;
            const int64_t q = target / k, r = target % k;
            int64_t got = 0;
            for (int j = 0; j < k; ++j) {
                const int64_t want = q + (j < r ? 1 : 0);
                take[j] = want < avail[j] ? want : avail[j];
                got += take[j];
            }
            int64_t short_ = target - got;
            for (int j = 0; j < k && short_ != 0; ++j) {
                const int64_t room = avail[j] - take[j];
                const int64_t extra = short_ < room ? short_ : room;
                take[j] += extra;
                short_ -= extra;
            }
            for (int j = 0; j < k; ++j) {
                E.g(g, don[j]) = (int32_t)(avail[j] - take[j]);
                if (g == E.o2 || g == E.n2) bulk_take[j] += take[j];
            }
            E.g(g, c) = (int32_t)target;
        }
        for (int j = 0; j < k; ++j) energy_move(E, don[j], c, bulk_take[j], t_abs_d[j], B, mb);
        for (int j = 0; j < k; ++j) refresh_cell(E, don[j], t_amb);
        refresh_cell(E, c, t_amb);
        int64_t atm = 0;
        for (int j = 0; j < k; ++j) atm += (int64_t)E.atmosphere[don[j]];
        E.atmosphere[c] = (int32_t)fixedpoint::floordiv_q(atm, (int64_t)k);
        E.wave_p[c] = E.atmosphere[c];
    }
    return 0;
}

// vent_system._aperture_blocked: solid / thermal_solid / vacuum / ambient /
// flooded. The caller guarantees (fy, fx) is in the grid.
__device__ __forceinline__ bool aperture_blocked(const TopoEnv& E, int fy, int fx) {
    const int i = fy * E.w + fx;
    return E.solid[i] || E.thermal_solid[i] || E.is_vacuum[i] || E.is_ambient[i] ||
           E.water_depth[i] != 0;
}

// GameMap._inject_gas_n_core's refusal: a bulk slice past int32 (the trace
// slices take the counted saturation). Reads only.
__device__ inline bool inject_core_fits(const TopoEnv& E, const int64_t* tab, int i,
                                        const int64_t* add) {
    for (int g = 0; g < E.n_gases; ++g) {
        if (trace_slice(tab, g)) continue;
        if ((int64_t)E.g(g, i) + add[g] >= I32_LIMIT) return false;
    }
    return true;
}

// GameMap._inject_gas_n_core's write (after inject_core_fits).
__device__ inline void inject_core(const TopoEnv& E, const int64_t* tab, int i, const int64_t* add,
                                   int64_t* B) {
    for (int g = 0; g < E.n_gases; ++g) {
        if (add[g] == 0) continue;
        if (trace_slice(tab, g))
            trace_store_sat(E, g, i, add[g], B);
        else
            E.g(g, i) = (int32_t)((int64_t)E.g(g, i) + add[g]);
    }
}

//: The most gas slices a primitive's vector holds (GameMap's N_GASES).
constexpr int GAS_MAX = UNIT_TAB_GAS_TRACE_W;

// GameMap.inject_gas_n(fy, fx, dn): 0 done (a no-op for dn <= 0), else the
// refusal (out_of_bounds; overflow, the core's; fault past the range).
__device__ inline int64_t inject_gas_n(const TopoEnv& E, const int64_t* tab, int64_t fy,
                                       int64_t fx, int64_t dn, int64_t* B) {
    if (fy < 0 || fy >= E.h || fx < 0 || fx >= E.w) return APPLIER_SEAM_OUT_OF_BOUNDS;
    if (dn <= 0) return 0;
    if (dn >= ((int64_t)1 << 40) || E.n_gases > GAS_MAX) return APPLIER_SEAM_FAULT;
    const int i = (int)fy * E.w + (int)fx;
    const int64_t t_amb = tab[UNIT_TAB_TOPO_T_AMB_Q];
    if (!t_ok(t_amb) || !e_ok(E, i)) return APPLIER_SEAM_FAULT;
    const int64_t o2_add = (dn * tab[UNIT_TAB_INJECT_O2_FRAC_Q] + ((int64_t)1 << 15)) >> 16;
    int64_t add[GAS_MAX];
    for (int g = 0; g < GAS_MAX; ++g) add[g] = 0;
    add[E.o2] = o2_add;
    add[E.n2] = dn - o2_add;
    if (!inject_core_fits(E, tab, i, add)) return APPLIER_SEAM_OVERFLOW;
    inject_core(E, tab, i, add, B);
    if (accountable(E, i)) {                               // gas_energy_mint
        const int64_t de = gas_energy::minted(dn, (int32_t)t_amb);
        E.gas_energy[i] += de;
        B[APPLIER_BOOK_GE_PUMP_INJECT] += de;
    }
    refresh_cell(E, i, t_amb);
    return 0;
}

// GameMap._air_withdrawal at cell i: `remove` per slice, returns `want`.
// Reads only (the caller pre-checked the range: planes < 2^31, so every
// product < 2^63).
__device__ inline int64_t air_withdrawal(const TopoEnv& E, const int64_t* tab, int i, int64_t dn,
                                         int64_t* remove) {
    for (int g = 0; g < E.n_gases; ++g) remove[g] = 0;
    if (dn <= 0) return 0;
    const int64_t o2 = E.g(E.o2, i), n2 = E.g(E.n2, i);
    const int64_t h0 = o2 > 0 ? o2 : 0, h1 = n2 > 0 ? n2 : 0;
    const int64_t n_bulk = h0 + h1;
    if (n_bulk < gas_energy::N_EPS_RAW) return 0;
    const int64_t want = dn < n_bulk ? dn : n_bulk;
    // gas_proportional_split([h0, h1], want): the floor shares, the
    // shortfall cascaded in index order (O2 first)
    int64_t tk[2] = {(h0 * want) / n_bulk, (h1 * want) / n_bulk};
    const int64_t hold[2] = {h0, h1};
    int64_t short_ = want - tk[0] - tk[1];
    for (int g = 0; g < 2 && short_ > 0; ++g) {
        const int64_t room = hold[g] - tk[g];
        if (room <= 0) continue;
        const int64_t grab = room < short_ ? room : short_;
        tk[g] += grab;
        short_ -= grab;
    }
    remove[E.o2] = tk[0];
    remove[E.n2] = tk[1];
    for (int g = 0; g < E.n_gases; ++g) {
        if (g == E.o2 || g == E.n2) continue;
        const int64_t s = E.g(g, i);
        if (s > 0) remove[g] = (want * s) / n_bulk;        // priced by air
    }
    return want;
}

// GameMap.extract_gas_n_vec(fy, fx, dn): 0 done (`remove` the per-slice
// withdrawal, `*taken` its bulk part: extract_gas_n's return), else the
// refusal (out_of_bounds; fault past the range).
__device__ inline int64_t extract_gas_n_vec(const TopoEnv& E, const int64_t* tab, int64_t fy,
                                            int64_t fx, int64_t dn, int64_t* remove,
                                            int64_t* taken, int64_t* B) {
    *taken = 0;
    for (int g = 0; g < E.n_gases && g < GAS_MAX; ++g) remove[g] = 0;
    if (fy < 0 || fy >= E.h || fx < 0 || fx >= E.w) return APPLIER_SEAM_OUT_OF_BOUNDS;
    if (E.n_gases > GAS_MAX) return APPLIER_SEAM_FAULT;
    const int i = (int)fy * E.w + (int)fx;
    const int64_t t_amb = tab[UNIT_TAB_TOPO_T_AMB_Q];
    if (dn >= ((int64_t)1 << 40) || !t_ok(t_amb) || !e_ok(E, i) ||
        !t_ok(t_abs_at(E, tab, i)))
        return APPLIER_SEAM_FAULT;
    const int64_t want = air_withdrawal(E, tab, i, dn, remove);
    if (want == 0) return 0;
    const int64_t t_abs = t_abs_at(E, tab, i);
    for (int g = 0; g < E.n_gases; ++g)
        if (remove[g]) E.g(g, i) = (int32_t)((int64_t)E.g(g, i) - remove[g]);
    if (accountable(E, i)) {
        const int64_t de = want * t_abs;
        E.gas_energy[i] -= de;
        B[APPLIER_BOOK_GE_PUMP_EXTRACT] -= de;
        refresh_cell(E, i, t_amb);
    }
    *taken = remove[E.o2] + remove[E.n2];
    return 0;
}

// GameMap.transfer_gas_n(sy, sx, dy, dx, dn): 0 done (`*moved` the bulk total
// moved), else the refusal (out_of_bounds; overflow at the destination --
// DECIDED BEFORE ANY WRITE: the Python drains the source first and then
// raises, so on this path the device is atomic where the reference is not;
// fault past the range).
__device__ inline int64_t transfer_gas_n(const TopoEnv& E, const int64_t* tab, int64_t sy,
                                         int64_t sx, int64_t dy, int64_t dx, int64_t dn,
                                         int64_t* moved, int64_t* B) {
    *moved = 0;
    if (sy < 0 || sy >= E.h || sx < 0 || sx >= E.w) return APPLIER_SEAM_OUT_OF_BOUNDS;
    if (dy < 0 || dy >= E.h || dx < 0 || dx >= E.w) return APPLIER_SEAM_OUT_OF_BOUNDS;
    if (sy == dy && sx == dx) return 0;
    if (E.n_gases > GAS_MAX) return APPLIER_SEAM_FAULT;
    const int s = (int)sy * E.w + (int)sx, d = (int)dy * E.w + (int)dx;
    const int64_t t_amb = tab[UNIT_TAB_TOPO_T_AMB_Q];
    if (dn >= ((int64_t)1 << 40) || !t_ok(t_amb) || !e_ok(E, s) || !e_ok(E, d) ||
        !t_ok(t_abs_at(E, tab, s)) || !gas_nonneg(E, d))
        return APPLIER_SEAM_FAULT;
    int64_t remove[GAS_MAX];
    const int64_t want = air_withdrawal(E, tab, s, dn, remove);
    if (want == 0) return 0;
    if (!inject_core_fits(E, tab, d, remove)) return APPLIER_SEAM_OVERFLOW;
    const int64_t t_abs = t_abs_at(E, tab, s);              // BEFORE the drain
    for (int g = 0; g < E.n_gases; ++g)
        if (remove[g]) E.g(g, s) = (int32_t)((int64_t)E.g(g, s) - remove[g]);
    inject_core(E, tab, d, remove, B);
    energy_move(E, s, d, want, t_abs, B, transfer_books());
    refresh_cell(E, s, t_amb);
    refresh_cell(E, d, t_amb);
    *moved = want;
    return 0;
}

// ============================================================================
// #29 P6b.2 -- GameMap.inject_gas_n_vec (the vents' deposit with its rails),
// integer for integer: `add` (n_gases slices, bulk AND trace) through the
// shared mutation core (bulk loud past int32 = the overflow refusal, trace the
// counted saturation), then -- on an ACCOUNTABLE tile -- the conversion seam:
// `dE = dN_bulk * (t_dep + T_AMB)` added to the cell's energy and booked to the
// caller's channel, the mirror read with Python's floor (floordiv_q), railed
// to [t_min, t_max], a rail booked to the caller's rail channel and written
// back into E; on a NON-accountable tile the pre-#54 mass-weighted T-mix
// (floor), railed. `*t_new` / `*rail` are the Python's return (rail -1 low,
// 0 none, +1 high). A bulk-free result (N_old + dN_bulk <= 0) leaves T alone,
// a zero vector writes nothing.
//
// THE PROVED RANGE (pre-checked, a FAULT past it, nothing written): every add
// slice in [0, 2^54) (a negative one is the Python contract's violation; the
// bound keeps a stage's clip books inside int64), the cell's gas planes >= 0,
// |gas_energy| < 2^61, T_AMB < 2^28, |t_dep| < 2^30, |t_min|, |t_max| < 2^30,
// and on a non-accountable tile |T_old| < 2^30. Then (bulk slices < 2^31 each
// after the overflow refusal, so N_old, dN_bulk, N_new < 2^32): |dE| < 2^32 *
// (2^30 + 2^28) < 2^62.4, |E + dE| < 2^63, a rail's N_new * (T + T_AMB) <
// 2^62.4, and the T-mix's N_old * T_old + dN * t_dep < 2^63.
// ============================================================================

//: The book channels of one inject_gas_n_vec caller (its `channel` and
//: `rail_channel`): the vents' are pump_inject_vec / pump_rail; P7c.2's bulk
//: gas payload brings its own pair.
struct InjectVecBooks {
    int ch, rail;
};
__device__ __forceinline__ InjectVecBooks vent_inject_books() {
    return InjectVecBooks{APPLIER_BOOK_GE_PUMP_INJECT_VEC, APPLIER_BOOK_GE_PUMP_RAIL};
}

constexpr int64_t INJECT_ADD_MAX = (int64_t)1 << 54;
constexpr int64_t INJECT_T_BOUND = (int64_t)1 << 30;

__device__ inline int64_t inject_gas_n_vec(const TopoEnv& E, const int64_t* tab, int64_t fy,
                                           int64_t fx, const int64_t* add, int64_t t_dep,
                                           int64_t t_min, int64_t t_max, int64_t* t_new_out,
                                           int64_t* rail_out, int64_t* B, InjectVecBooks bk) {
    *t_new_out = 0;
    *rail_out = 0;
    if (fy < 0 || fy >= E.h || fx < 0 || fx >= E.w) return APPLIER_SEAM_OUT_OF_BOUNDS;
    if (E.n_gases > GAS_MAX) return APPLIER_SEAM_FAULT;
    const int i = (int)fy * E.w + (int)fx;
    int64_t dn_total = 0;
    for (int g = 0; g < E.n_gases; ++g) {
        if (add[g] < 0 || add[g] >= INJECT_ADD_MAX) return APPLIER_SEAM_FAULT;
        dn_total += add[g];
    }
    if (dn_total <= 0) {                                   // nothing to deposit
        *t_new_out = (int64_t)E.temperature[i];
        return 0;
    }
    const int64_t t_amb = tab[UNIT_TAB_TOPO_T_AMB_Q];
    const bool acct = accountable(E, i);
    const int64_t t_old = (int64_t)E.temperature[i];
    if (!t_ok(t_amb) || !e_ok(E, i) || !gas_nonneg(E, i) ||
        t_dep <= -INJECT_T_BOUND || t_dep >= INJECT_T_BOUND ||
        t_min <= -INJECT_T_BOUND || t_min >= INJECT_T_BOUND ||
        t_max <= -INJECT_T_BOUND || t_max >= INJECT_T_BOUND ||
        (!acct && (t_old <= -INJECT_T_BOUND || t_old >= INJECT_T_BOUND)))
        return APPLIER_SEAM_FAULT;
    if (!inject_core_fits(E, tab, i, add)) return APPLIER_SEAM_OVERFLOW;
    const int64_t n_old = E.bulk(i);
    const int64_t d_bulk = add[E.o2] + add[E.n2];
    inject_core(E, tab, i, add, B);
    const int64_t n_new = n_old + d_bulk;
    if (n_new <= 0) {                                      // nothing bulk to weight by
        *t_new_out = t_old;
        return 0;
    }
    int64_t t_new;
    int64_t rail = 0;
    if (acct) {
        const int64_t de = d_bulk * (t_dep + t_amb);
        E.gas_energy[i] += de;
        B[bk.ch] += de;
        t_new = fixedpoint::floordiv_q(E.gas_energy[i], n_new) - t_amb;
        if (t_new < t_min) {
            t_new = t_min;
            rail = -1;
        } else if (t_new > t_max) {
            t_new = t_max;
            rail = 1;
        }
        if (rail != 0) {
            const int64_t e_new = n_new * (t_new + t_amb);
            B[bk.rail] += e_new - E.gas_energy[i];
            E.gas_energy[i] = e_new;
        }
    } else {
        t_new = fixedpoint::floordiv_q(n_old * t_old + d_bulk * t_dep, n_new);
        if (t_new < t_min) {
            t_new = t_min;
            rail = -1;
        } else if (t_new > t_max) {
            t_new = t_max;
            rail = 1;
        }
    }
    E.temperature[i] = (int32_t)t_new;
    *t_new_out = t_new;
    *rail_out = rail;
    return 0;
}

}  // namespace topo
}  // namespace breach_cuda
