// ============================================================================
// #29 P7.0a -- SLOT 9d ON THE DEVICE: ignition from temperature
// (combat.apply_temperature_ignition; residency design 4.4c, the P7.0 row).
//
// A PER-CELL law, order-free (the reference is a vectorised gather that writes
// each tile from frozen inputs): one thread per cell, born (N, h, w) (bgrid,
// blockIdx.y the env, every plane rebased per env), on the tick stream. For a
// FLAMMABLE tile (the `flammable` plane):
//
//   T <  ignition_temp_q16[material]      -> re-arm (armed = 1)
//   T >= threshold and wall_hp > 0 (hot):
//       armed, fire == 0 and X > X_ext    -> fire = seed
//       fire > 0 (after the seed)         -> disarm (armed = 0)
//
// with X = (sum_o2 * 2^16) // max(sum_tot, floor) over the OPEN (non-solid,
// non-vacuum) 4-neighbours inside the grid (sum_tot = O2 + inert N2; a tile
// with no open neighbour never ignites). Every constant is a Device tables
// row read from the object the law reads (unit_tables' ign_* rows); the gas
// planes' indices are the map's own. The law reads NO pressure factor:
// apply_temperature_ignition gates on the mole fraction alone (the fire's
// sustain also multiplies o2_pressure_factor.h's g(p) -- a reference finding
// reported by P7.0a, not changed), so neither does its twin.
//
// The planes are device pointers to (N, ...) arrays in the GameMap fields'
// own dtypes (ResidentBatch.buffers as they are): temperature int32, material
// int8, flammable / solid / is_vacuum / ignition_armed bool (1 byte), wall_hp
// int32, gas (N, G, h, w) int32, fire int32. fire and ignition_armed are
// written IN PLACE. Counts: coverage = cells changed (fire or armed); faults
// = cells whose material or a gas index left the tables (left unchanged).
//
// CLASS T (training only): nothing in the played tick calls this TU.
// ============================================================================
#include "cuda_unit_kernels.h"
#include "cuda_unit_lib.cuh"

namespace breach_cuda {

namespace {

using namespace unit_lib;

__global__ void k_ignition(const int64_t* tables, IgnitionPlanes p, unsigned long long* cnt) {
    const int n = p.h * p.w;
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    const int32_t* temperature = bptr(reinterpret_cast<const int32_t*>(p.temperature), (size_t)n);
    const int8_t* material = bptr(reinterpret_cast<const int8_t*>(p.material), (size_t)n);
    const uint8_t* flammable = bptr(reinterpret_cast<const uint8_t*>(p.flammable), (size_t)n);
    const int32_t* wall_hp = bptr(reinterpret_cast<const int32_t*>(p.wall_hp), (size_t)n);
    const uint8_t* solid = bptr(reinterpret_cast<const uint8_t*>(p.solid), (size_t)n);
    const uint8_t* vacuum = bptr(reinterpret_cast<const uint8_t*>(p.is_vacuum), (size_t)n);
    const int32_t* gas = bptr(reinterpret_cast<const int32_t*>(p.gas), (size_t)p.n_gases * n);
    int32_t* fire = bptr(reinterpret_cast<int32_t*>(p.fire), (size_t)n);
    uint8_t* armed = bptr(reinterpret_cast<uint8_t*>(p.armed), (size_t)n);
    const int64_t* tab = env_tables(tables);

    if (!flammable[i]) return;                  // only fuel re-arms, ignites or disarms
    const int m = (int)material[i];
    const int64_t o2i = tab[UNIT_TAB_IGN_GAS_O2], n2i = tab[UNIT_TAB_IGN_GAS_N2];
    if (m < 0 || m >= UNIT_TAB_IGN_TEMP_Q_W || o2i >= p.n_gases || n2i >= p.n_gases) {
        count_fault(cnt);
        return;
    }
    const int64_t t = (int64_t)temperature[i];
    if (t < tab[UNIT_TAB_IGN_TEMP_Q + m]) {     // cooled: RE-ARM
        if (!armed[i]) {
            armed[i] = 1;
            count_changed(cnt);
        }
        return;
    }
    if (!(wall_hp[i] > 0)) return;              // no fuel: never hot
    bool changed = false;
    int32_t f = fire[i];
    if (armed[i] && f == 0) {
        // the O2 mole fraction over the open 4-neighbours (N, S, E, W)
        const int y = i / p.w, x = i % p.w;
        const int32_t* o2 = gas + (size_t)o2i * n;
        const int32_t* n2 = gas + (size_t)n2i * n;
        int64_t sum_o2 = 0, sum_tot = 0, count = 0;
        const int dys[4] = {-1, 1, 0, 0}, dxs[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k) {
            const int ny = y + dys[k], nx = x + dxs[k];
            if (ny < 0 || ny >= p.h || nx < 0 || nx >= p.w) continue;
            const int j = ny * p.w + nx;
            if (solid[j] || vacuum[j]) continue;
            sum_o2 += (int64_t)o2[j];
            sum_tot += (int64_t)o2[j] + (int64_t)n2[j];
            ++count;
        }
        const int64_t floor_q = tab[UNIT_TAB_IGN_X_N_FLOOR_Q];
        const int64_t den = sum_tot > floor_q ? sum_tot : floor_q;
        const int64_t x_q16 = fixedpoint::floordiv_q(sum_o2 * 65536, den);
        if (count > 0 && x_q16 > tab[UNIT_TAB_IGN_X_EXT_Q]) {
            f = (int32_t)tab[UNIT_TAB_IGN_SEED_Q];
            if (fire[i] != f) {
                fire[i] = f;
                changed = true;
            }
        }
    }
    if (f > 0 && armed[i]) {                    // a hot tile burning: DISARM
        armed[i] = 0;
        changed = true;
    }
    if (changed) count_changed(cnt);
}

}  // namespace

void unit_ignition(UnitStoreDevice& store, const UnitTablesDevice& tables,
                   const IgnitionPlanes& planes, const UnitKernelCounts& out) {
    unit_check_pair(store, tables, "unit_ignition");
    if (planes.h < 1 || planes.w < 1 || planes.n_gases < 1)
        throw std::invalid_argument("unit_ignition: empty planes");
    if (!planes.temperature || !planes.material || !planes.flammable || !planes.wall_hp ||
        !planes.solid || !planes.is_vacuum || !planes.gas || !planes.fire || !planes.armed)
        throw std::invalid_argument("unit_ignition: a null plane");
    store.begin_counts();
    k_ignition<<<bgrid(planes.h * planes.w, store.n_env()), 256, 0, tstream()>>>(
        tables.values(), planes, store.counts_dev());
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
    store.end_counts(out.coverage, out.faults, out.refused);
}

}  // namespace breach_cuda
