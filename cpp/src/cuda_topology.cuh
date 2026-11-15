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
// device backend (P8b) hands its buffers unchanged. The float planes
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
// Device-only header (CUDA types); included by the applier's TU only.
// CLASS T: nothing in the played tick includes it.
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "applier_layout.h"        // GENERATED: the planes, the books
#include "fixed_point.h"           // the kit
#include "gas_energy.h"            // THE gas energy seam (shared with the edit queue)
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

}  // namespace topo
}  // namespace breach_cuda
