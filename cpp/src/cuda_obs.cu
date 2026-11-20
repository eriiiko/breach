// ==========================================================================
// #29 P8.0: THE DEVICE OBSERVATION's kernels (see cuda_obs.h), Class T.
//
// The twin of obs_spec_v1.build_windows' structure planes (spec 3.1, 3.3,
// 3.4: near_u "wall" / "door", far_f "wall_share", far_u "door"),
// FieldPlanes' door plane, and EnvRunner.row_flags. Pure integer: the only
// float16 values written (k / 16) are bit patterns generated from numpy's
// own conversion (obs_layout.h). On the target-wide CUDA determinism floor
// (--fmad=false, the strict host floor) like every .cu.
//
// The reference's arithmetic, value for value:
//   tile_x = int(x_q / 65536) truncated toward zero (Unit.tile_x);
//   centre = tile + footprint // 2 (Unit.center_tile_x / _y);
//   near window top-left (cy - HALF, cx - HALF), cells outside the map read
//     wall OUTSIDE (1) and door 0 (crop);
//   far window top-left (cy // BLOCK - HALF, cx // BLOCK - HALF) in blocks
//     anchored at tile (0, 0) (Python floor division), a block's tiles past
//     the map padded wall 1 / door 0 (blocks(plane, fill)); a block outside
//     the block grid reads wall_share OUTSIDE (1.0) and door 0;
//   wall_share = (wall tiles of the block) / 16 as float16.
// ==========================================================================
#include "cuda_obs.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "cuda_batch.cuh"          // tstream(): THE tick stream (blocking)
#include "cuda_unit_events.h"      // UnitRecordList (the door tables' buffers)
#include "obs_layout.h"
#include "object_tables_layout.h"
#include "unit_draws_layout.h"
#include "unit_store_layout.h"

namespace breach_cuda {

using namespace obs_layout;

namespace {

__constant__ uint16_t c_share_half[OBS_BLOCK * OBS_BLOCK + 1] = OBS_WALL_SHARE_HALF_INIT;

constexpr int CELLS = OBS_WIN * OBS_WIN;

// Python's floor division by a positive divisor.
__device__ __forceinline__ int64_t floordiv(int64_t a, int64_t b) {
    const int64_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

// Unit.tile_x / tile_y: the Q16 position truncated toward zero.
__device__ __forceinline__ int64_t tile_of(int64_t q) {
    return q >= 0 ? (q >> 16) : -((-q) >> 16);
}

// The env's decision tick: (tick - episode start) a multiple of the
// decimation, and its ended latch clear.
__device__ __forceinline__ bool env_due(int e, const int64_t* trow, const int64_t* params,
                                        const uint8_t* ended) {
    if (ended[e]) return false;
    const int64_t tick = trow[(size_t)e * unit_draws::TICK_ROW_W + unit_draws::TICK_ROW_TICK];
    const int64_t t0 = params[(size_t)e * OBS_PARAM_W + OBS_PARAM_EPISODE_TICK];
    const int64_t dec = params[(size_t)e * OBS_PARAM_W + OBS_PARAM_DECIMATION];
    const int64_t d = tick - t0;
    return dec > 0 && d >= 0 && d % dec == 0;
}

// FieldPlanes.door: per living door, its code (open / closed) at every span
// tile on the map, the max (order-free, so an atomic max is exact).
__global__ void k_door_plane(const int64_t* doors, const int64_t* door_count, int door_cap,
                             int door_w, const int64_t* tiles, int tile_cap, int tile_w,
                             int32_t* plane, int h, int w, const int64_t* trow,
                             const int64_t* params, const uint8_t* ended) {
    const int e = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= door_cap || d >= door_count[e] || !env_due(e, trow, params, ended)) return;
    const int64_t* row = doors + ((size_t)e * door_cap + d) * door_w;
    if (!row[object_tables::OBJ_DOOR_COL_ALIVE]) return;
    const int64_t st = row[object_tables::OBJ_DOOR_COL_STATE];
    const int code = st == OBS_DOOR_STATE_OPEN ? OBS_DOOR_CODE_OPEN
                   : st == OBS_DOOR_STATE_CLOSED ? OBS_DOOR_CODE_CLOSED : OBS_DOOR_CODE_NONE;
    if (code == OBS_DOOR_CODE_NONE) return;
    const int64_t t0 = row[object_tables::OBJ_DOOR_COL_TILE_START];
    const int64_t tn = row[object_tables::OBJ_DOOR_COL_TILE_COUNT];
    for (int64_t k = 0; k < tn; ++k) {
        const int64_t ti = t0 + k;
        if (ti < 0 || ti >= tile_cap) continue;
        const int64_t* t = tiles + ((size_t)e * tile_cap + ti) * tile_w;
        const int64_t ty = t[object_tables::OBJ_DOOR_TILE_COL_TY];
        const int64_t tx = t[object_tables::OBJ_DOOR_TILE_COL_TX];
        if (0 <= ty && ty < h && 0 <= tx && tx < w)
            atomicMax(plane + ((size_t)e * h + ty) * w + tx, code);
    }
}

// One thread per (env, row, window cell): valid / in_match (cell 0) and the
// four structure planes at that cell of both windows.
__global__ void k_obs_block(const int64_t* store, int n_units, const int32_t* rowmap,
                            int n_rows, int n_slots, const uint8_t* solid,
                            const int32_t* door_plane, int h, int w, const int64_t* trow,
                            const int64_t* params, const uint8_t* ended, uint8_t* near_u,
                            uint16_t* far_f, uint8_t* far_u, uint8_t* valid_out,
                            uint8_t* in_match_out) {
    const int e = blockIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * CELLS || !env_due(e, trow, params, ended)) return;
    const int r = idx / CELLS, cell = idx % CELLS;
    const int i = cell / OBS_WIN, j = cell % OBS_WIN;
    const size_t er = (size_t)e * n_rows + r;
    const int sr = rowmap[er];
    bool valid = false;
    const int64_t* u = nullptr;
    if (sr >= 0 && sr < n_units) {
        u = store + ((size_t)e * n_units + sr) * unit_store::UNIT_STORE_W;
        valid = u[unit_store::UNIT_COL_LIFE_STATE] == OBS_LIFE_ALIVE &&
                u[unit_store::UNIT_COL_TEAM] == (int64_t)(r / n_slots);
    }
    if (cell == 0) {
        valid_out[er] = valid ? 1 : 0;
        in_match_out[er] = sr >= 0 ? 1 : 0;
    }
    uint8_t* nu = near_u + er * OBS_NEAR_U_PLANES * CELLS;
    uint16_t* ff = far_f + er * OBS_FAR_F_PLANES * CELLS;
    uint8_t* fu = far_u + er * OBS_FAR_U_PLANES * CELLS;
    if (!valid) {
        nu[OBS_NEAR_U_WALL * CELLS + cell] = 0;
        nu[OBS_NEAR_U_DOOR * CELLS + cell] = 0;
        ff[OBS_FAR_F_WALL_SHARE * CELLS + cell] = 0;
        fu[OBS_FAR_U_DOOR * CELLS + cell] = 0;
        return;
    }
    const int64_t fp = u[unit_store::UNIT_COL_FOOTPRINT];
    const int64_t cx = tile_of(u[unit_store::UNIT_COL_X_Q]) + floordiv(fp, 2);
    const int64_t cy = tile_of(u[unit_store::UNIT_COL_Y_Q]) + floordiv(fp, 2);
    const uint8_t* so = solid + (size_t)e * h * w;
    const int32_t* dp = door_plane + (size_t)e * h * w;
    // -- near (3.1, 3.4) --
    const int64_t y = cy - OBS_HALF + i, x = cx - OBS_HALF + j;
    const bool inside = 0 <= y && y < h && 0 <= x && x < w;
    nu[OBS_NEAR_U_WALL * CELLS + cell] =
        inside ? (so[y * w + x] ? 1 : 0) : (uint8_t)OBS_WALL_OUTSIDE;
    nu[OBS_NEAR_U_DOOR * CELLS + cell] = inside ? (uint8_t)dp[y * w + x] : 0;
    // -- far (3.3, 3.4): the block grid anchored at tile (0, 0) --
    const int64_t hb = (h + OBS_BLOCK - 1) / OBS_BLOCK, wb = (w + OBS_BLOCK - 1) / OBS_BLOCK;
    const int64_t bi = floordiv(cy, OBS_BLOCK) - OBS_HALF + i;
    const int64_t bj = floordiv(cx, OBS_BLOCK) - OBS_HALF + j;
    if (0 <= bi && bi < hb && 0 <= bj && bj < wb) {
        int walls = 0, door = 0;
        for (int a = 0; a < OBS_BLOCK; ++a) {
            const int64_t ty = bi * OBS_BLOCK + a;
            for (int b = 0; b < OBS_BLOCK; ++b) {
                const int64_t tx = bj * OBS_BLOCK + b;
                if (ty < h && tx < w) {
                    walls += so[ty * w + tx] ? 1 : 0;
                    const int dv = dp[ty * w + tx];
                    door = dv > door ? dv : door;
                } else {
                    walls += 1;                     // blocks(wall, 1): the pad is wall
                }
            }
        }
        ff[OBS_FAR_F_WALL_SHARE * CELLS + cell] = c_share_half[walls];
        fu[OBS_FAR_U_DOOR * CELLS + cell] = (uint8_t)door;
    } else {
        ff[OBS_FAR_F_WALL_SHARE * CELLS + cell] = (uint16_t)OBS_WALL_SHARE_OUTSIDE_HALF;
        fu[OBS_FAR_U_DOOR * CELLS + cell] = 0;
    }
}

void need(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("obs_block: ") + what);
}

}  // namespace

void obs_block(int n_env, std::uintptr_t store, int n_units, int store_w,
               std::uintptr_t rowmap, int n_rows, int n_slots,
               std::uintptr_t solid, int h, int w,
               UnitRecordList& doors, UnitRecordList& door_tiles, std::uintptr_t door_plane,
               std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
               std::uintptr_t near_u, std::uintptr_t far_f, std::uintptr_t far_u,
               std::uintptr_t valid, std::uintptr_t in_match) {
    need(n_env >= 1 && n_units >= 1 && n_rows >= 1 && n_slots >= 1 && h >= 1 && w >= 1,
         "a non-positive size");
    need(store_w == unit_store::UNIT_STORE_W, "the store's width is not UNIT_STORE_W");
    need(store && rowmap && solid && door_plane && tick_row && params && ended && near_u &&
         far_f && far_u && valid && in_match, "a null device pointer");
    need(doors.n_env() == n_env && door_tiles.n_env() == n_env, "the door tables' env count");
    need(doors.width() == object_tables::OBJ_DOOR_W &&
         door_tiles.width() == object_tables::OBJ_DOOR_TILE_W, "the door tables' widths");
    cudaStream_t s = tstream();
    int32_t* dp = reinterpret_cast<int32_t*>(door_plane);
    cudaMemsetAsync(dp, 0, sizeof(int32_t) * (size_t)n_env * h * w, s);
    const int64_t* trow = reinterpret_cast<const int64_t*>(tick_row);
    const int64_t* prm = reinterpret_cast<const int64_t*>(params);
    const uint8_t* end = reinterpret_cast<const uint8_t*>(ended);
    const int dcap = doors.cap();
    k_door_plane<<<dim3((dcap + 63) / 64, n_env), 64, 0, s>>>(
        doors.records_dev(), doors.count_dev(), dcap, doors.width(), door_tiles.records_dev(),
        door_tiles.cap(), door_tiles.width(), dp, h, w, trow, prm, end);
    const int cells = n_rows * CELLS;
    k_obs_block<<<dim3((cells + 255) / 256, n_env), 256, 0, s>>>(
        reinterpret_cast<const int64_t*>(store), n_units, reinterpret_cast<const int32_t*>(rowmap),
        n_rows, n_slots, reinterpret_cast<const uint8_t*>(solid), dp, h, w, trow, prm, end,
        reinterpret_cast<uint8_t*>(near_u), reinterpret_cast<uint16_t*>(far_f),
        reinterpret_cast<uint8_t*>(far_u), reinterpret_cast<uint8_t*>(valid),
        reinterpret_cast<uint8_t*>(in_match));
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("obs_block: launch failed: ") +
                                 cudaGetErrorString(err));
}

const char* obs_layout_fingerprint() { return OBS_LAYOUT_FINGERPRINT; }

}  // namespace breach_cuda
