#pragma once
// ============================================================================
// #29 P7d -- THE HORDE on the device: slot 5 (ai_zombie.update_zombies_tick
// through Simulation._run_zombie_slot) as the applier's HORDE stage, and THE
// FLOOD (horde.flood / cpp/src/horde_field.cpp) as a fan-out over the stage's
// block. Residency design 4.4c (the cut table's P7d row) and 4.9.
//
// ONE BLOCK per env (the applier's launch: blockIdx.y the env), HORDE_BT
// threads. In the reference's order:
//   the DRAIN       the env's noise list copied to the block and emptied
//                   (noise.take: every tick, driven or not);
//   the LATCH       mode EMPTY (a match's zombie controller named no id):
//                   nothing more; no living prey: nothing more -- both BEFORE
//                   the cache's begin_tick, so the cache stays standing;
//   begin_tick      the cache dropped at tick % refresh == 0 or when the
//                   map's topology serial moved since the last drop;
//   pass 1          WIDE (one thread per row): a non-awake zombie with a line
//                   of sight to a prey strictly inside the trigger radius wakes;
//   pass 2          WIDE, to the fixpoint: an awake zombie wakes every
//                   non-awake one within the propagation radius (the closure
//                   of the host's sweep, whatever its order); the awake drop
//                   their noise and anchor (horde.clear);
//   pass 2b         WIDE: hearing (each non-awake zombie the nearest drained
//                   noise, ties to the earlier), then thread 0 keeps the
//                   noise_fields_max newest targets and settles the rest;
//   pass 3          SERIAL in row order on thread 0 (a bash applied at once
//                   changes the map the next zombie reads for its step, its
//                   fear, its sight; a bite the hp the next zombie's bite
//                   reads): the bite (THE device apply_packet, immediate, its
//                   events on the ring, credited to the biter, the conversion
//                   flag), the walk (default_speed_q over the footprint, the
//                   stride clock, the descent on its field with the parity
//                   tie order, the fear of fire, the door BASH through
//                   topo::chew_wall at the bite's cadence with its noise
//                   appended to the noise list), the wander (the seedless
//                   Philox hash of (tick, id)). A field the walk needs that the
//                   cache does not hold is FLOODED on that need by the whole
//                   block (thread 0 hands the key over; the block builds the
//                   mask on the first flood of a drop, the anchor grid's
//                   passability and the BFS frontier by frontier) from the LIVE
//                   map, then thread 0 resumes the zombie where it stopped.
//
// THE FLOOD: distances are the BFS's, which no visiting order changes, so the
// frontier is processed in parallel (an atomic CAS claims a cell, an atomic
// counter appends it to the next frontier): equal cell for cell to the C++
// reference, whose FIFO order only fixes the order of equal distances.
//
// Every per-env buffer (the cache, the scratch: the pass plane, the two
// frontiers, the row flags, the hearing targets) is indexed by the env.
// CLASS T (training only): nothing in the played tick includes this header.
// ============================================================================
#include <cstdint>

#include "applier_layout.h"         // GENERATED: APPLIER_HORDE_*, the stage, the books
#include "object_tables_layout.h"   // GENERATED: the noise list, the door tables
#include "unit_tables_layout.h"     // GENERATED: the horde_* rows
#include "cuda_topology.cuh"        // TopoEnv, chew_wall, destroy_wall
#include "cuda_unit_damage.cuh"     // apply_packet, the sinks, the unit library
#include "cuda_unit_sight.cuh"      // has_los
#include "cuda_unit_draws.cuh"      // seedless_word, below, SALT_WANDER

namespace breach_cuda {
namespace horde_dev {

using namespace applier;
using namespace unit_tables;
using namespace unit_lib;

//: The block's threads (the applier launches the HORDE stage with these).
constexpr int HORDE_BT = 256;
//: horde.UNREACHED == horde::UNREACHED.
constexpr int32_t UNREACHED = 0x7FFFFFFF;
//: horde.WANDER_STAGGER.
constexpr int64_t WANDER_STAGGER = 37;
//: unit_schema.SPEED_FROM_TABLE.
constexpr int64_t SPEED_FROM_TABLE = -1;
//: The proved range of a tile coordinate the law squares (a d2 < 2^62).
constexpr int64_t TILE_MAX = (int64_t)1 << 28;

//: horde.NEIGHBOURS (N, W, E, S) and DESCENT_ORDER (by the cell's parity).
static __device__ __constant__ const int NB_DY[4] = {-1, 0, 0, 1};
static __device__ __constant__ const int NB_DX[4] = {0, -1, 1, 0};
static __device__ __constant__ const int DESC_DY[2][4] = {{-1, 1, 0, 0}, {0, 0, -1, 1}};
static __device__ __constant__ const int DESC_DX[2][4] = {{0, 0, -1, 1}, {-1, 1, 0, 0}};

// The per-env device state and scratch the stage reaches (device addresses of
// (N, ...) arrays; each rebased to env e by `env`).
struct HordeBuffers {
    int64_t* hdr = nullptr;     // (N, HDR_W)
    int64_t* keys = nullptr;    // (N, FIELD_CAP, KEY_W)
    int32_t* dist = nullptr;    // (N, FIELD_CAP, h, w)
    uint8_t* masks = nullptr;   // (N, N_MASKS, h, w)
    uint8_t* pass = nullptr;    // (N, h, w)
    uint8_t* flags = nullptr;   // (N, 2, U): players, zombies driven
    int32_t* queues = nullptr;  // (N, 2, h, w)
    int64_t* live = nullptr;    // (N, U, 3): the hearing's targets
    const int64_t* latch = nullptr;   // (N, LATCH_W + U)
    int fcap = 0;
};

// One env's view.
struct Env {
    const topo::TopoEnv* E;
    const int64_t* tab;
    int64_t* store;               // (N, U, W) (unit_row rebases by benv())
    int n_units;
    uint8_t* touched;             // (U,)
    EventSink* sink;              // the env's ring (bites)
    RecordSink* noise_out;        // the env's noise list (a strike's noise)
    const int64_t* doors;         // (door cap, OBJ_DOOR_W) of this env
    int64_t n_doors;
    const int64_t* dtiles;        // (tile cap, OBJ_DOOR_TILE_W) of this env
    int64_t n_dtiles;
    int64_t* hdr;
    int64_t* keys;
    int32_t* dist;                // (fcap, n)
    uint8_t* masks;               // (N_MASKS, n)
    uint8_t* pass;                // (n)
    uint8_t* players;             // (U)
    uint8_t* zsel;                // (U)
    int32_t* q[2];                // (n) each
    int64_t* live;                // (U, 3)
    const int64_t* latch;
    int fcap;
    int64_t tick;
    int64_t* B;
};

// The block's shared state.
struct Shared {
    int64_t noise[object_tables::OBJ_NOISE_CAP][3];
    int n_noise;
    int stop;                     // 1: the stage ends after the drain
    int bad;                      // a wide pass faulted
    int changed;                  // pass 2's fixpoint
    int cmd;                      // pass 3: CMD_DONE / CMD_FLOOD
    int64_t key[APPLIER_HORDE_KEY_W];
    int slot;                     // the field slot being flooded
    int n_cur, n_next, flip;
    // pass 3's resume point
    int r, phase;
};
constexpr int CMD_DONE = 0, CMD_FLOOD = 1;
constexpr int PH_START = 0, PH_PREY = 1, PH_PREY_FEARLESS = 2, PH_NOISE = 3;

__device__ __forceinline__ int64_t* row_of(const Env& V, int r) {
    return unit_row(V.store, V.n_units, r);
}
__device__ __forceinline__ void mark(const Env& V, int r, bool changed) {
    if (changed) V.touched[r] = 1;
}
// Unit.center_tile_x / _y.
__device__ __forceinline__ int64_t ctx(const int64_t* row) {
    return tile_of(row[UNIT_COL_X_Q]) + row[UNIT_COL_FOOTPRINT] / 2;
}
__device__ __forceinline__ int64_t cty(const int64_t* row) {
    return tile_of(row[UNIT_COL_Y_Q]) + row[UNIT_COL_FOOTPRINT] / 2;
}
// ai_zombie._d2 within the proved range (*bad outside).
__device__ __forceinline__ int64_t d2(int64_t ax, int64_t ay, int64_t bx, int64_t by,
                                      bool* bad) {
    const int64_t dx = ax - bx, dy = ay - by;
    if (dx > TILE_MAX || dx < -TILE_MAX || dy > TILE_MAX || dy < -TILE_MAX) {
        *bad = true;
        return 0;
    }
    return dx * dx + dy * dy;
}
__device__ __forceinline__ bool los(const Env& V, int64_t y1, int64_t x1, int64_t y2,
                                    int64_t x2, bool* bad) {
    return unit_sight::has_los(reinterpret_cast<const uint8_t*>(V.E->solid),
                               reinterpret_cast<const uint8_t*>(V.E->sight_screen), V.E->h,
                               V.E->w, y1, x1, y2, x2, bad);
}
__device__ __forceinline__ MobilityMap mob_map(const Env& V) {
    return MobilityMap{V.E->material, V.E->h, V.E->w, V.tab + UNIT_TAB_MATERIAL_MOBILITY,
                       UNIT_TAB_MATERIAL_MOBILITY_W};
}

// horde.clear on a row.
__device__ inline void horde_clear(int64_t* row, bool& ch) {
    put(row, UNIT_COL_ZOMBIE_NOISE_Y, -1, ch);
    put(row, UNIT_COL_ZOMBIE_NOISE_X, -1, ch);
    put(row, UNIT_COL_ZOMBIE_NOISE_TICKS, 0, ch);
    put(row, UNIT_COL_ZOMBIE_ANCHOR_Y, -1, ch);
    put(row, UNIT_COL_ZOMBIE_ANCHOR_X, -1, ch);
}
// horde.settle on a row.
__device__ inline void horde_settle(const Env& V, int64_t* row, bool& ch) {
    put(row, UNIT_COL_ZOMBIE_NOISE_Y, -1, ch);
    put(row, UNIT_COL_ZOMBIE_NOISE_X, -1, ch);
    put(row, UNIT_COL_ZOMBIE_NOISE_TICKS, 0, ch);
    if (V.tab[UNIT_TAB_HORDE_WANDER]) {
        put(row, UNIT_COL_ZOMBIE_ANCHOR_Y, tile_of(row[UNIT_COL_Y_Q]), ch);
        put(row, UNIT_COL_ZOMBIE_ANCHOR_X, tile_of(row[UNIT_COL_X_Q]), ch);
    }
}

// ---- the cache ------------------------------------------------------------------
// The slot of `key` in the cache, or -1.
__device__ inline int find_field(const Env& V, const int64_t* key) {
    const int64_t n = V.hdr[APPLIER_HORDE_HDR_N_FIELDS];
    for (int s = 0; s < n; ++s) {
        const int64_t* k = V.keys + (size_t)s * APPLIER_HORDE_KEY_W;
        bool same = true;
        for (int c = 0; c < APPLIER_HORDE_KEY_W; ++c)
            if (k[c] != key[c]) same = false;
        if (same) return s;
    }
    return -1;
}
__device__ __forceinline__ int32_t field_at(const Env& V, int slot, int64_t y, int64_t x) {
    const topo::TopoEnv& E = *V.E;
    if (y < 0 || y >= E.h || x < 0 || x >= E.w) return UNREACHED;
    return V.dist[(size_t)slot * E.n + (size_t)y * E.w + (size_t)x];
}

// horde.is_hull_door over the door's span (rows a .. a + c of the tile table).
__device__ inline bool hull_door(const Env& V, int64_t a, int64_t c) {
    const topo::TopoEnv& E = *V.E;
    for (int64_t i = a; i < a + c; ++i) {
        const int64_t fy = V.dtiles[i * object_tables::OBJ_DOOR_TILE_W +
                                    object_tables::OBJ_DOOR_TILE_COL_TY];
        const int64_t fx = V.dtiles[i * object_tables::OBJ_DOOR_TILE_W +
                                    object_tables::OBJ_DOOR_TILE_COL_TX];
        for (int k = 0; k < 4; ++k) {
            const int64_t ny = fy + NB_DY[k], nx = fx + NB_DX[k];
            bool in_span = false;
            for (int64_t j = a; j < a + c; ++j)
                if (V.dtiles[j * object_tables::OBJ_DOOR_TILE_W +
                             object_tables::OBJ_DOOR_TILE_COL_TY] == ny &&
                    V.dtiles[j * object_tables::OBJ_DOOR_TILE_W +
                             object_tables::OBJ_DOOR_TILE_COL_TX] == nx)
                    in_span = true;
            if (in_span) continue;
            if (ny < 0 || ny >= E.h || nx < 0 || nx >= E.w) return true;
            const size_t j = (size_t)ny * E.w + (size_t)nx;
            if (E.is_vacuum[j] || E.is_ambient[j]) return true;
        }
    }
    return false;
}

// HordeState.mask: built once per drop, by the block, from the live map --
// the walkable tiles, plus the bashable doors' closed tiles (bash_doors), and
// the same minus the feared tiles (fear_fire).
__device__ inline void build_mask(const Env& V, Shared& S) {
    const topo::TopoEnv& E = *V.E;
    uint8_t* ok = V.masks + (size_t)APPLIER_HORDE_MASK_TILE_OK * E.n;
    uint8_t* okf = V.masks + (size_t)APPLIER_HORDE_MASK_TILE_OK_FEARLESS * E.n;
    uint8_t* bash = V.masks + (size_t)APPLIER_HORDE_MASK_BASHABLE * E.n;
    const MobilityMap m = mob_map(V);
    for (int i = threadIdx.x; i < E.n; i += blockDim.x) {
        bool bad = false;
        okf[i] = mobility_at(m, i / E.w, i % E.w, &bad) > 0 ? 1 : 0;
        bash[i] = 0;
        if (bad) atomicOr(&S.bad, 1);
    }
    __syncthreads();
    if (threadIdx.x == 0 && V.tab[UNIT_TAB_HORDE_BASH_DOORS]) {
        const int64_t closed = V.tab[UNIT_TAB_DOOR_STATES + 0];
        const int64_t mat_closed = V.tab[UNIT_TAB_DOOR_MAT_CLOSED];
        for (int64_t k = 0; k < V.n_doors; ++k) {
            const int64_t* d = V.doors + k * object_tables::OBJ_DOOR_W;
            if (!d[object_tables::OBJ_DOOR_COL_ALIVE] ||
                d[object_tables::OBJ_DOOR_COL_STATE] != closed)
                continue;
            const int64_t a = d[object_tables::OBJ_DOOR_COL_TILE_START];
            const int64_t c = d[object_tables::OBJ_DOOR_COL_TILE_COUNT];
            if (a < 0 || c < 0 || a + c > V.n_dtiles) {
                S.bad = 1;
                continue;
            }
            if (hull_door(V, a, c)) continue;
            for (int64_t i = a; i < a + c; ++i) {
                const int64_t fy = V.dtiles[i * object_tables::OBJ_DOOR_TILE_W +
                                            object_tables::OBJ_DOOR_TILE_COL_TY];
                const int64_t fx = V.dtiles[i * object_tables::OBJ_DOOR_TILE_W +
                                            object_tables::OBJ_DOOR_TILE_COL_TX];
                if (fy < 0 || fy >= E.h || fx < 0 || fx >= E.w) {
                    S.bad = 1;
                    continue;
                }
                const size_t j = (size_t)fy * E.w + (size_t)fx;
                if ((int64_t)E.material[j] == mat_closed) bash[j] = 1;
            }
        }
    }
    __syncthreads();
    const bool fear = V.tab[UNIT_TAB_HORDE_FEAR_FIRE] != 0;
    const int64_t fq = V.tab[UNIT_TAB_HORDE_FEAR_TEMP_Q16];
    for (int i = threadIdx.x; i < E.n; i += blockDim.x) {
        const uint8_t f = (okf[i] || bash[i]) ? 1 : 0;
        okf[i] = f;
        ok[i] = (fear && (E.fire[i] > 0 || (int64_t)E.temperature[i] >= fq)) ? 0 : f;
    }
    __syncthreads();
    if (threadIdx.x == 0) V.hdr[APPLIER_HORDE_HDR_MASK_BUILT] = 1;
    __syncthreads();
}

// horde.flood (the C++ reference's law) into `out` over `tile_ok`, by the
// block: the anchor grid's passability, every cell UNREACHED, the seeds
// (labelled 0 whether passable or not; a repeated or out-of-grid seed
// skipped) -- `seed_y` / `seed_x` read by thread 0, n_seeds of them -- then the
// frontier loop. Shared by the HORDE stage and the TEST-ONLY probe.
template <class Seeds>
__device__ inline void flood_block(const uint8_t* tile_ok, int h, int w, int fp, int32_t* out,
                                   uint8_t* pass, int32_t* q0, int32_t* q1, const Seeds& seeds,
                                   Shared& S) {
    const int n = h * w;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int y = i / w, x = i % w;
        uint8_t p = 0;
        if (fp > 0 && y + fp <= h && x + fp <= w) {
            p = 1;
            for (int yy = y; yy < y + fp && p; ++yy)
                for (int xx = x; xx < x + fp; ++xx)
                    if (!tile_ok[(size_t)yy * w + xx]) {
                        p = 0;
                        break;
                    }
        }
        pass[i] = p;
        out[i] = UNREACHED;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int nq = 0;
        if (fp > 0) {
            for (int s = 0; s < seeds.count(); ++s) {
                const int64_t y = seeds.y(s), x = seeds.x(s);
                if (y < 0 || y >= h || x < 0 || x >= w) continue;
                const int i = (int)y * w + (int)x;
                if (out[i] == 0) continue;
                out[i] = 0;
                q0[nq++] = i;
            }
        }
        S.n_cur = nq;
        S.flip = 0;
    }
    __syncthreads();
    while (S.n_cur > 0) {
        int32_t* cur = S.flip ? q1 : q0;
        int32_t* nxt = S.flip ? q0 : q1;
        const int nc = S.n_cur;
        if (threadIdx.x == 0) S.n_next = 0;
        __syncthreads();
        for (int k = threadIdx.x; k < nc; k += blockDim.x) {
            const int i = cur[k];
            const int y = i / w, x = i - y * w;
            const int32_t nd = out[i] + 1;
            const int js[4] = {y > 0 ? i - w : -1, x > 0 ? i - 1 : -1,
                               x + 1 < w ? i + 1 : -1, y + 1 < h ? i + w : -1};
            for (int q = 0; q < 4; ++q) {
                const int j = js[q];
                if (j < 0 || !pass[j]) continue;
                if (atomicCAS(reinterpret_cast<int*>(out + j), (int)UNREACHED, (int)nd) ==
                    (int)UNREACHED)
                    nxt[atomicAdd(&S.n_next, 1)] = j;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            S.n_cur = S.n_next;
            S.flip ^= 1;
        }
        __syncthreads();
    }
}

// The seeds of a field: every prey's centre in the zombie's anchor space
// (horde.anchor_of), or the one noise anchor of the key.
struct FieldSeeds {
    const Env* V;
    int64_t kind, fp, sy, sx;
    int n;            // prey: the store's rows (only players count); noise: 1
    __device__ int count() const { return n; }
    __device__ bool use(int s) const {
        return kind == APPLIER_HORDE_KIND_NOISE || V->players[s];
    }
    __device__ int64_t y(int s) const {
        if (kind == APPLIER_HORDE_KIND_NOISE) return sy;
        if (!use(s)) return -1;
        return cty(row_of(*V, s)) - fp / 2;
    }
    __device__ int64_t x(int s) const {
        if (kind == APPLIER_HORDE_KIND_NOISE) return sx;
        if (!use(s)) return -1;
        return ctx(row_of(*V, s)) - fp / 2;
    }
};

// The block floods the field of S.key into slot S.slot (the mask first, when
// this drop has none).
__device__ inline void flood_field(const Env& V, Shared& S) {
    if (V.hdr[APPLIER_HORDE_HDR_MASK_BUILT] == 0) build_mask(V, S);
    const topo::TopoEnv& E = *V.E;
    const int64_t kind = S.key[APPLIER_HORDE_KEY_KIND];
    const uint8_t* ok = V.masks + (size_t)(kind == APPLIER_HORDE_KIND_PREY_FEARLESS
                                               ? APPLIER_HORDE_MASK_TILE_OK_FEARLESS
                                               : APPLIER_HORDE_MASK_TILE_OK) * E.n;
    FieldSeeds sd{&V, kind, S.key[APPLIER_HORDE_KEY_FP], S.key[APPLIER_HORDE_KEY_SY],
                  S.key[APPLIER_HORDE_KEY_SX], kind == APPLIER_HORDE_KIND_NOISE ? 1 : V.n_units};
    flood_block(ok, E.h, E.w, (int)S.key[APPLIER_HORDE_KEY_FP],
                V.dist + (size_t)S.slot * E.n, V.pass, V.q[0], V.q[1], sd, S);
}

// ---- pass 3's pieces (thread 0) ------------------------------------------------
// Unit.face_towards(nx, ny) -> turning.set_target.
__device__ inline bool face_towards(int64_t* row, int64_t nx, int64_t ny, bool& ch) {
    const int64_t dx = (nx << 16) - row[UNIT_COL_X_Q];
    const int64_t dy = (ny << 16) - row[UNIT_COL_Y_Q];
    if (dx == 0 && dy == 0) return true;
    if (dx > 2147483647LL || dx < -2147483647LL || dy > 2147483647LL || dy < -2147483647LL)
        return false;
    put(row, UNIT_COL_FACING_TARGET_Q,
        normalize_q16(fixedpoint::atan2_q16((q16)(-dy), (q16)dx)), ch);
    return true;
}

// horde.block_feared on the live map.
__device__ inline bool block_feared(const Env& V, int64_t y, int64_t x, int64_t fp) {
    const topo::TopoEnv& E = *V.E;
    const int64_t y0 = y > 0 ? y : 0, x0 = x > 0 ? x : 0;
    int64_t y1 = y + fp > 0 ? y + fp : 0, x1 = x + fp > 0 ? x + fp : 0;
    if (y1 > E.h) y1 = E.h;
    if (x1 > E.w) x1 = E.w;
    const int64_t fq = V.tab[UNIT_TAB_HORDE_FEAR_TEMP_Q16];
    for (int64_t yy = y0; yy < y1; ++yy)
        for (int64_t xx = x0; xx < x1; ++xx) {
            const size_t i = (size_t)yy * E.w + (size_t)xx;
            if (E.fire[i] > 0 || (int64_t)E.temperature[i] >= fq) return true;
        }
    return false;
}

// horde.bash_tile on the cache's bashable mask and the live map: the first
// blocking tile row-major when every blocking tile is a bashable closed door.
__device__ inline bool bash_tile(const Env& V, int64_t y, int64_t x, int64_t fp, int64_t* ty,
                                 int64_t* tx, bool* bad) {
    const topo::TopoEnv& E = *V.E;
    if (y < 0 || x < 0 || y + fp > E.h || x + fp > E.w) return false;
    const MobilityMap m = mob_map(V);
    const uint8_t* bash = V.masks + (size_t)APPLIER_HORDE_MASK_BASHABLE * E.n;
    const int64_t mat_closed = V.tab[UNIT_TAB_DOOR_MAT_CLOSED];
    bool found = false;
    for (int64_t yy = y; yy < y + fp; ++yy)
        for (int64_t xx = x; xx < x + fp; ++xx) {
            const size_t i = (size_t)yy * E.w + (size_t)xx;
            if (mobility_at(m, yy, xx, bad) > 0) continue;
            if (bash[i] && (int64_t)E.material[i] == mat_closed) {
                if (!found) {
                    *ty = yy;
                    *tx = xx;
                    found = true;
                }
            } else {
                return false;
            }
        }
    return found;
}

constexpr int WALK_STEP = 0, WALK_BASH = 1, WALK_WAIT = 2;

// ai_zombie._bash: a strike at the bite's cadence while it can act, through
// the wall-chew seam, a noise at the struck tile.
__device__ inline bool strike(const Env& V, int r, int64_t* row, int64_t nx, int64_t ny,
                              int64_t ty, int64_t tx, bool can_act) {
    if (!can_act) return true;
    if (V.tick - row[UNIT_COL_LAST_MELEE_TICK] < V.tab[UNIT_TAB_ZOMBIE_ATTACK_COOLDOWN_TICKS])
        return true;
    bool ch = false;
    put(row, UNIT_COL_LAST_MELEE_TICK, V.tick, ch);
    if (!face_towards(row, nx, ny, ch)) return false;
    mark(V, r, ch);
    const int64_t dq = V.tab[UNIT_TAB_HORDE_BASH_DAMAGE_Q];
    const int res = topo::chew_wall(*V.E, V.tab, ty, tx, dq, dq > 0, V.B);
    if (res == topo::CHEW_FAULT) return false;
    V.B[APPLIER_BOOK_HORDE_BASHES] += 1;
    const int64_t rad = V.tab[UNIT_TAB_HORDE_BASH_LOUDNESS_TILES];
    if (rad > 0) {                                   // noise.emit: 0 is silence
        int64_t rec[object_tables::OBJ_NOISE_W] = {};
        rec[object_tables::OBJ_NOISE_COL_FY] = ty;
        rec[object_tables::OBJ_NOISE_COL_FX] = tx;
        rec[object_tables::OBJ_NOISE_COL_RADIUS] = rad;
        rec[object_tables::OBJ_NOISE_COL_LIVE] = 1;
        emit_record(*V.noise_out, rec);
    }
    return true;
}

// horde.descend on field `slot`, then the move made (ai_zombie._walk's tail):
// WALK_STEP (the clock reset, the facing, the whole-tile position), WALK_BASH
// (a strike, if its cadence allows) or WALK_WAIT. false: a fault.
__device__ inline bool descend(const Env& V, int r, int64_t* row, int slot, bool can_act,
                               int* got) {
    const int64_t y = tile_of(row[UNIT_COL_Y_Q]), x = tile_of(row[UNIT_COL_X_Q]);
    const int64_t fp = row[UNIT_COL_FOOTPRINT];
    const topo::TopoEnv& E = *V.E;
    const int32_t cur = field_at(V, slot, y, x);
    const int par = (int)((x + y) & 1);
    int64_t cv[4], cy[4], cx[4];
    int ck[4], nc = 0;
    for (int k = 0; k < 4; ++k) {
        const int64_t ny = y + DESC_DY[par][k], nx = x + DESC_DX[par][k];
        if (ny < 0 || ny >= E.h || nx < 0 || nx >= E.w) continue;
        const int32_t v = V.dist[(size_t)slot * E.n + (size_t)ny * E.w + (size_t)nx];
        if (v < cur) {
            // insertion by (value, order): the order k rises with the loop
            int p = nc++;
            while (p > 0 && cv[p - 1] > v) {
                cv[p] = cv[p - 1]; cy[p] = cy[p - 1]; cx[p] = cx[p - 1]; ck[p] = ck[p - 1];
                --p;
            }
            cv[p] = v; cy[p] = ny; cx[p] = nx; ck[p] = k;
        }
    }
    const bool fear = V.tab[UNIT_TAB_HORDE_FEAR_FIRE] != 0;
    const bool bash = V.tab[UNIT_TAB_HORDE_BASH_DOORS] != 0;
    const MobilityMap m = mob_map(V);
    bool bad = false;
    for (int c = 0; c < nc; ++c) {
        if (fear && block_feared(V, cy[c], cx[c], fp)) continue;
        if (passable_block(m, cy[c], cx[c], fp, &bad)) {
            if (bad) return false;
            bool ch = false;
            put(row, UNIT_COL_ZOMBIE_MOVE_ACCUMULATOR, 0, ch);
            if (!face_towards(row, cx[c], cy[c], ch)) return false;
            put(row, UNIT_COL_X_Q, cx[c] << 16, ch);
            put(row, UNIT_COL_Y_Q, cy[c] << 16, ch);
            mark(V, r, ch);
            *got = WALK_STEP;
            return true;
        }
        if (bad) return false;
        if (bash) {
            int64_t ty = 0, tx = 0;
            const bool hit = bash_tile(V, cy[c], cx[c], fp, &ty, &tx, &bad);
            if (bad) return false;
            if (hit) {
                *got = WALK_BASH;
                return strike(V, r, row, cx[c], cy[c], ty, tx, can_act);
            }
        }
    }
    *got = WALK_WAIT;
    return true;
}

// ai_zombie._walk's head: the footprint's cadence at `base` ticks per tile
// with the gait's hook and the effort scale; the stride clock advanced.
// *due false: the clock still runs. false: a fault.
__device__ inline bool stride(const Env& V, int r, int64_t* row, int64_t base, int gait,
                              bool* due) {
    bool bad = false;
    const int64_t fp = row[UNIT_COL_FOOTPRINT];
    if (fp < 1 || fp > FOOTPRINT_MAX) return false;
    int64_t n = 0, total = 0;
    footprint_mobility(mob_map(V), tile_of(row[UNIT_COL_Y_Q]), tile_of(row[UNIT_COL_X_Q]), fp,
                       &n, &total, &bad);
    if (bad) return false;
    const int64_t es = effort_scale_q16(row[UNIT_COL_EFFORT],
                                        V.tab[UNIT_TAB_IMPULSE_EFFORT_EDGE_Q]);
    const int64_t step = default_speed_q(n, total, base, hook(row, gait), es, &bad);
    if (bad) return false;
    const int64_t acc = row[UNIT_COL_ZOMBIE_MOVE_ACCUMULATOR] + 1;
    bool ch = false;
    put(row, UNIT_COL_ZOMBIE_MOVE_ACCUMULATOR, acc, ch);
    mark(V, r, ch);
    *due = !(acc < step);
    return true;
}

// The bite on the nearest prey (immediate apply_packet, cause melee, the
// biter credited, the conversion flag on the kill).
__device__ inline bool bite(const Env& V, int r, int64_t* row, int pr) {
    int64_t* prow = row_of(V, pr);
    Packet p;
    p.amount_q = scaled(row, UNIT_HOOK_STRIKE, V.tab[UNIT_TAB_ZOMBIE_BITE_CLASS_Q]);
    p.dtype = UNIT_DTYPE_KINETIC;
    p.attacker = row[UNIT_COL_ID];
    p.dir_x_q = centre_q(prow[UNIT_COL_X_Q], prow[UNIT_COL_FOOTPRINT]) -
                centre_q(row[UNIT_COL_X_Q], row[UNIT_COL_FOOTPRINT]);
    p.dir_y_q = centre_q(prow[UNIT_COL_Y_Q], prow[UNIT_COL_FOOTPRINT]) -
                centre_q(row[UNIT_COL_Y_Q], row[UNIT_COL_FOOTPRINT]);
    p.cause = UNIT_CAUSE_MELEE;
    p.mark_killed_by_zombie = true;
    bool ch = false, killed = false, bad = false;
    apply_packet(prow, V.tab, p, *V.sink, ch, &killed, &bad);
    if (bad) return false;
    mark(V, pr, ch);
    V.B[APPLIER_BOOK_HORDE_BITES] += 1;
    return true;
}

// ai_zombie._sees_prey: a line of sight to a prey (the slot's prey list)
// within r2 (inclusive).
__device__ inline bool sees_prey(const Env& V, const int64_t* row, int64_t r2, bool* bad) {
    const int64_t zx = ctx(row), zy = cty(row);
    for (int p = 0; p < V.n_units; ++p) {
        if (!V.players[p]) continue;
        const int64_t* pr = row_of(V, p);
        const int64_t px = ctx(pr), py = cty(pr);
        if (d2(zx, zy, px, py, bad) <= r2 && los(V, zy, zx, py, px, bad)) return true;
        if (*bad) return false;
    }
    return false;
}

// The base cadence of a walk (ai_zombie's SPEED_FROM_TABLE rule): the unit's
// own, else the sprint (awake, sprint on, a prey in sight) or the shamble.
__device__ inline int64_t base_ticks(const Env& V, const int64_t* row, bool awake, int* gait,
                                     bool* bad) {
    *gait = UNIT_HOOK_WALK;
    const int64_t own = row[UNIT_COL_SPEED_TICKS_PER_TILE];
    if (own != SPEED_FROM_TABLE) return own;
    if (awake && V.tab[UNIT_TAB_HORDE_SPRINT] &&
        sees_prey(V, row, V.tab[UNIT_TAB_HORDE_SPRINT_R2], bad)) {
        *gait = UNIT_HOOK_SPRINT;
        return V.tab[UNIT_TAB_ZOMBIE_SPRINT_TICKS];
    }
    return V.tab[UNIT_TAB_ZOMBIE_SHAMBLE_TICKS];
}

__device__ inline void prey_key(int64_t* key, int64_t fp, bool fearless) {
    key[APPLIER_HORDE_KEY_KIND] = fearless ? APPLIER_HORDE_KIND_PREY_FEARLESS
                                           : APPLIER_HORDE_KIND_PREY;
    key[APPLIER_HORDE_KEY_FP] = fp;
    key[APPLIER_HORDE_KEY_SY] = 0;
    key[APPLIER_HORDE_KEY_SX] = 0;
}

// The pass-3 outcomes of one advance step.
constexpr int ADV_NEXT = 0, ADV_NEED = 1, ADV_FAULT = 2;

// One zombie at its resume point (thread 0). ADV_NEED: S.key is the field
// it needs (S.phase where to resume); ADV_NEXT: the zombie is done.
__device__ inline int zombie_step(const Env& V, Shared& S) {
  for (;;) {      // a cached field resumes at once (no recursion on the device)
    const int r = S.r;
    int64_t* row = row_of(V, r);
    bool bad = false;
    const Flags f = composed_flags(row, V.tab, false, &bad);
    if (bad) return ADV_FAULT;
    const int64_t fp = row[UNIT_COL_FOOTPRINT];
    const int64_t ty = tile_of(row[UNIT_COL_Y_Q]), tx = tile_of(row[UNIT_COL_X_Q]);
    bool ch = false;
    if (S.phase == PH_PREY || S.phase == PH_PREY_FEARLESS) {
        int64_t key[APPLIER_HORDE_KEY_W];
        prey_key(key, fp, S.phase == PH_PREY_FEARLESS);
        const int slot = find_field(V, key);
        if (slot < 0) return ADV_FAULT;
        if (S.phase == PH_PREY && V.tab[UNIT_TAB_HORDE_FEAR_FIRE] &&
            field_at(V, slot, ty, tx) == UNREACHED) {
            prey_key(key, fp, true);           // fire closes every way: the fearless field
            if (find_field(V, key) < 0) {
                for (int c = 0; c < APPLIER_HORDE_KEY_W; ++c) S.key[c] = key[c];
                S.phase = PH_PREY_FEARLESS;
                return ADV_NEED;
            }
            S.phase = PH_PREY_FEARLESS;
            continue;
        }
        int got = WALK_WAIT;
        return descend(V, r, row, slot, f.can_act, &got) ? ADV_NEXT : ADV_FAULT;
    }
    if (S.phase == PH_NOISE) {
        int64_t key[APPLIER_HORDE_KEY_W];
        for (int c = 0; c < APPLIER_HORDE_KEY_W; ++c) key[c] = S.key[c];
        const int slot = find_field(V, key);
        if (slot < 0) return ADV_FAULT;
        const int32_t here = field_at(V, slot, ty, tx);
        if ((int64_t)here <= V.tab[UNIT_TAB_HORDE_NOISE_ARRIVE_TILES]) {
            horde_settle(V, row, ch);          // arrived, saw nobody
            mark(V, r, ch);
            return ADV_NEXT;
        }
        int gait = UNIT_HOOK_WALK;
        const int64_t base = base_ticks(V, row, false, &gait, &bad);
        if (bad) return ADV_FAULT;
        bool due = false;
        if (!stride(V, r, row, base, gait, &due)) return ADV_FAULT;
        if (!due) return ADV_NEXT;
        int got = WALK_WAIT;
        if (!descend(V, r, row, slot, f.can_act, &got)) return ADV_FAULT;
        if (got == WALK_WAIT && here == UNREACHED) {
            horde_settle(V, row, ch);          // no path to the noise
            mark(V, r, ch);
        }
        return ADV_NEXT;
    }
    // ---- PH_START ----------------------------------------------------------------
    if (row[UNIT_COL_ZOMBIE_ACTIVATED]) {
        if (!f.can_move && !f.can_act) return ADV_NEXT;
        const int64_t zx = ctx(row), zy = cty(row);
        int near = -1;
        int64_t nd2 = 0;
        for (int p = 0; p < V.n_units; ++p) {
            if (!V.players[p]) continue;
            const int64_t* pr = row_of(V, p);
            const int64_t dd = d2(zx, zy, ctx(pr), cty(pr), &bad);
            if (bad) return ADV_FAULT;
            if (near < 0 || dd < nd2) {
                near = p;
                nd2 = dd;
            }
        }
        const int64_t reach = fp + 1;
        if (nd2 <= reach * reach) {
            if (f.can_act &&
                V.tick - row[UNIT_COL_LAST_MELEE_TICK] >=
                    V.tab[UNIT_TAB_ZOMBIE_ATTACK_COOLDOWN_TICKS]) {
                put(row, UNIT_COL_LAST_MELEE_TICK, V.tick, ch);
                mark(V, r, ch);
                if (!bite(V, r, row, near)) return ADV_FAULT;
            }
            return ADV_NEXT;
        }
        if (!f.can_move || !walk_allowed(row)) return ADV_NEXT;
        int gait = UNIT_HOOK_WALK;
        const int64_t base = base_ticks(V, row, true, &gait, &bad);
        if (bad) return ADV_FAULT;
        bool due = false;
        if (!stride(V, r, row, base, gait, &due)) return ADV_FAULT;
        if (!due) return ADV_NEXT;
        prey_key(S.key, fp, false);
        S.phase = PH_PREY;
        if (find_field(V, S.key) < 0) return ADV_NEED;
        continue;
    }
    if (row[UNIT_COL_ZOMBIE_NOISE_TICKS] > 0) {
        const int64_t left = row[UNIT_COL_ZOMBIE_NOISE_TICKS] - 1;
        put(row, UNIT_COL_ZOMBIE_NOISE_TICKS, left, ch);
        mark(V, r, ch);
        if (left <= 0) {
            horde_settle(V, row, ch);          // timed out
            mark(V, r, ch);
            return ADV_NEXT;
        }
        if (!f.can_move || !walk_allowed(row)) return ADV_NEXT;
        const topo::TopoEnv& E = *V.E;
        int64_t sy = row[UNIT_COL_ZOMBIE_NOISE_Y] - fp / 2;
        int64_t sx = row[UNIT_COL_ZOMBIE_NOISE_X] - fp / 2;
        sy = sy < 0 ? 0 : (sy > E.h - 1 ? E.h - 1 : sy);
        sx = sx < 0 ? 0 : (sx > E.w - 1 ? E.w - 1 : sx);
        S.key[APPLIER_HORDE_KEY_KIND] = APPLIER_HORDE_KIND_NOISE;
        S.key[APPLIER_HORDE_KEY_FP] = fp;
        S.key[APPLIER_HORDE_KEY_SY] = sy;
        S.key[APPLIER_HORDE_KEY_SX] = sx;
        S.phase = PH_NOISE;
        if (find_field(V, S.key) < 0) return ADV_NEED;
        continue;
    }
    if (!V.tab[UNIT_TAB_HORDE_WANDER]) return ADV_NEXT;
    // DORMANT: one hashed step every period, within the leash
    if (row[UNIT_COL_ZOMBIE_ANCHOR_Y] < 0) {
        put(row, UNIT_COL_ZOMBIE_ANCHOR_Y, ty, ch);
        put(row, UNIT_COL_ZOMBIE_ANCHOR_X, tx, ch);
        mark(V, r, ch);
    }
    const int64_t id = row[UNIT_COL_ID];
    const int64_t period = V.tab[UNIT_TAB_HORDE_WANDER_PERIOD_TICKS];
    if (id < 0 || id > TILE_MAX) return ADV_FAULT;
    if ((V.tick + id * WANDER_STAGGER) % period != 0) return ADV_NEXT;
    if (!f.can_move || !walk_allowed(row)) return ADV_NEXT;
    const int64_t ay = row[UNIT_COL_ZOMBIE_ANCHOR_Y], ax = row[UNIT_COL_ZOMBIE_ANCHOR_X];
    const int64_t leash = V.tab[UNIT_TAB_HORDE_WANDER_LEASH_R2];
    const int64_t cur = d2(ty, tx, ay, ax, &bad);
    if (bad) return ADV_FAULT;
    int64_t ny, nx;
    if (cur > leash) {
        int best = -1;
        int64_t bd = 0;
        for (int k = 0; k < 4; ++k) {
            const int64_t dd = d2(ty + NB_DY[k], tx + NB_DX[k], ay, ax, &bad);
            if (dd < cur && (best < 0 || dd < bd)) {
                best = k;
                bd = dd;
            }
        }
        if (bad) return ADV_FAULT;
        if (best < 0) return ADV_NEXT;
        ny = ty + NB_DY[best];
        nx = tx + NB_DX[best];
    } else {
        const uint32_t wd = unit_draws_dev::seedless_word(V.tick, unit_draws::SALT_WANDER, id);
        const int k = (int)unit_draws_dev::below(wd, 4u);
        ny = ty + NB_DY[k];
        nx = tx + NB_DX[k];
        if (d2(ny, nx, ay, ax, &bad) > leash) return bad ? ADV_FAULT : ADV_NEXT;
    }
    if (!passable_block(mob_map(V), ny, nx, fp, &bad)) return bad ? ADV_FAULT : ADV_NEXT;
    if (bad) return ADV_FAULT;
    if (V.tab[UNIT_TAB_HORDE_FEAR_FIRE] && block_feared(V, ny, nx, fp)) return ADV_NEXT;
    if (!face_towards(row, nx, ny, ch)) return ADV_FAULT;
    put(row, UNIT_COL_X_Q, nx << 16, ch);
    put(row, UNIT_COL_Y_Q, ny << 16, ch);
    mark(V, r, ch);
    return ADV_NEXT;
  }
}

// Thread 0: pass 3 from (S.r, S.phase) until a field must be flooded (CMD_FLOOD:
// the key, the slot appended to the cache) or every zombie is done (CMD_DONE).
__device__ inline void advance(const Env& V, Shared& S) {
    while (S.r < V.n_units) {
        if (!V.zsel[S.r]) {
            S.r += 1;
            S.phase = PH_START;
            continue;
        }
        const int res = zombie_step(V, S);
        if (res == ADV_FAULT) {
            V.B[APPLIER_BOOK_FAULTS] += 1;
            S.cmd = CMD_DONE;                     // the env stops here (refused by name)
            return;
        }
        if (res == ADV_NEED) {
            const int64_t n = V.hdr[APPLIER_HORDE_HDR_N_FIELDS];
            const int64_t fp = S.key[APPLIER_HORDE_KEY_FP];
            if (n >= V.fcap) {
                V.B[APPLIER_BOOK_HORDE_REFUSED] += 1;
                S.cmd = CMD_DONE;
                return;
            }
            if (fp < 1 || fp > FOOTPRINT_MAX) {
                V.B[APPLIER_BOOK_FAULTS] += 1;
                S.cmd = CMD_DONE;
                return;
            }
            for (int c = 0; c < APPLIER_HORDE_KEY_W; ++c)
                V.keys[(size_t)n * APPLIER_HORDE_KEY_W + c] = S.key[c];
            V.hdr[APPLIER_HORDE_HDR_N_FIELDS] = n + 1;
            V.hdr[APPLIER_HORDE_HDR_FLOODS] += 1;
            V.B[APPLIER_BOOK_HORDE_FLOODS] += 1;
            S.slot = (int)n;
            S.cmd = CMD_FLOOD;
            return;                               // resume at S.phase after the flood
        }
        S.r += 1;
        S.phase = PH_START;
    }
    S.cmd = CMD_DONE;
}

// ---- the stage ------------------------------------------------------------------
__device__ inline void stage(const Env& V, Shared& S, const RecordListView& noise, int e) {
    const topo::TopoEnv& E = *V.E;
    const int U = V.n_units;
    for (int r = threadIdx.x; r < U; r += blockDim.x) V.touched[r] = 0;
    if (threadIdx.x == 0) {
        S.bad = 0;
        S.stop = 0;
        // the DRAIN (noise.take): the noises since the last slot, the list emptied
        const int64_t n = noise.count[e];
        const int64_t* L = noise.recs + (size_t)e * noise.cap * noise.width;
        if (n > object_tables::OBJ_NOISE_CAP || n > noise.cap) {
            S.bad = 1;
            S.n_noise = 0;
        } else {
            S.n_noise = (int)n;
            for (int i = 0; i < (int)n; ++i) {
                S.noise[i][0] = L[(size_t)i * noise.width + object_tables::OBJ_NOISE_COL_FY];
                S.noise[i][1] = L[(size_t)i * noise.width + object_tables::OBJ_NOISE_COL_FX];
                S.noise[i][2] = L[(size_t)i * noise.width + object_tables::OBJ_NOISE_COL_RADIUS];
            }
        }
        noise.count[e] = 0;
        // the LATCH and the prey: an empty request set, or no living prey,
        // returns before begin_tick (the cache stays standing)
        const int64_t mode = V.latch[APPLIER_HORDE_LATCH_MODE];
        const int64_t* req = V.latch + APPLIER_HORDE_LATCH_W;
        int n_prey = 0;
        for (int r = 0; r < U; ++r) {
            const int64_t* row = row_of(V, r);
            const bool ok = row_valid(row) && row_alive(row);
            V.players[r] = (ok && !row[UNIT_COL_IS_ZOMBIE]) ? 1 : 0;
            V.zsel[r] = (ok && row[UNIT_COL_IS_ZOMBIE] &&
                         (mode == APPLIER_HORDE_MODE_CONDUCTOR || req[r] != 0)) ? 1 : 0;
            n_prey += V.players[r];
        }
        if (mode == APPLIER_HORDE_MODE_EMPTY || n_prey == 0) S.stop = 1;
        if (!S.stop) {
            // HordeState.begin_tick
            const int64_t serial = V.latch[APPLIER_HORDE_LATCH_SERIAL];
            const int64_t every = V.tab[UNIT_TAB_HORDE_REFRESH_TICKS];
            if (V.tick % every == 0 || !V.hdr[APPLIER_HORDE_HDR_HAS_SERIAL] ||
                V.hdr[APPLIER_HORDE_HDR_SERIAL] != serial) {
                V.hdr[APPLIER_HORDE_HDR_N_FIELDS] = 0;
                V.hdr[APPLIER_HORDE_HDR_MASK_BUILT] = 0;
                V.hdr[APPLIER_HORDE_HDR_HAS_SERIAL] = 1;
                V.hdr[APPLIER_HORDE_HDR_SERIAL] = serial;
            }
        }
    }
    __syncthreads();
    if (S.stop) {
        if (threadIdx.x == 0 && S.bad) V.B[APPLIER_BOOK_FAULTS] += 1;
        return;
    }
    // ---- pass 1: trigger (wide) ----------------------------------------------------
    const int64_t trig = V.tab[UNIT_TAB_HORDE_TRIGGER_R2];
    for (int r = threadIdx.x; r < U; r += blockDim.x) {
        if (!V.zsel[r]) continue;
        int64_t* row = row_of(V, r);
        if (row[UNIT_COL_ZOMBIE_ACTIVATED]) continue;
        bool bad = false;
        const int64_t zx = ctx(row), zy = cty(row);
        for (int p = 0; p < U; ++p) {
            if (!V.players[p]) continue;
            const int64_t* pr = row_of(V, p);
            const int64_t px = ctx(pr), py = cty(pr);
            if (d2(zx, zy, px, py, &bad) < trig && los(V, zy, zx, py, px, &bad)) {
                bool ch = false;
                put(row, UNIT_COL_ZOMBIE_ACTIVATED, 1, ch);
                mark(V, r, ch);
                break;
            }
            if (bad) break;
        }
        if (bad) atomicOr(&S.bad, 1);
    }
    __syncthreads();
    // ---- pass 2: the chain, to its fixpoint (wide) --------------------------------
    const int64_t prop = V.tab[UNIT_TAB_HORDE_PROPAGATION_R2];
    for (int it = 0; it <= U; ++it) {
        if (threadIdx.x == 0) S.changed = 0;
        __syncthreads();
        for (int r = threadIdx.x; r < U; r += blockDim.x) {
            if (!V.zsel[r]) continue;
            int64_t* row = row_of(V, r);
            if (row[UNIT_COL_ZOMBIE_ACTIVATED]) continue;
            bool bad = false;
            for (int z = 0; z < U; ++z) {
                if (z == r || !V.zsel[z]) continue;
                const int64_t* zr = row_of(V, z);
                if (!((volatile const int64_t*)zr)[UNIT_COL_ZOMBIE_ACTIVATED]) continue;
                if (d2(ctx(zr), cty(zr), ctx(row), cty(row), &bad) < prop) {
                    bool ch = false;
                    put(row, UNIT_COL_ZOMBIE_ACTIVATED, 1, ch);
                    mark(V, r, ch);
                    S.changed = 1;
                    break;
                }
            }
            if (bad) atomicOr(&S.bad, 1);
        }
        __syncthreads();
        if (!S.changed) break;
        __syncthreads();
    }
    for (int r = threadIdx.x; r < U; r += blockDim.x) {
        if (!V.zsel[r]) continue;
        int64_t* row = row_of(V, r);
        if (row[UNIT_COL_ZOMBIE_ACTIVATED] &&
            (row[UNIT_COL_ZOMBIE_NOISE_TICKS] != 0 || row[UNIT_COL_ZOMBIE_ANCHOR_Y] >= 0)) {
            bool ch = false;
            horde_clear(row, ch);              // awake for good: no noise, no anchor
            mark(V, r, ch);
        }
    }
    __syncthreads();
    // ---- pass 2b: hearing (wide, then the keep on thread 0) ------------------------
    if (V.tab[UNIT_TAB_HORDE_HEARING] && S.n_noise > 0) {
        const int64_t tmo = V.tab[UNIT_TAB_HORDE_NOISE_TIMEOUT_TICKS];
        for (int r = threadIdx.x; r < U; r += blockDim.x) {
            if (!V.zsel[r]) continue;
            int64_t* row = row_of(V, r);
            if (row[UNIT_COL_ZOMBIE_ACTIVATED]) continue;
            bool bad = false;
            const int64_t zx = ctx(row), zy = cty(row);
            int best = -1;
            int64_t bd = 0;
            for (int i = 0; i < S.n_noise; ++i) {
                const int64_t rad = S.noise[i][2];
                const int64_t dd = d2(zx, zy, S.noise[i][1], S.noise[i][0], &bad);
                if (rad < 0 || rad > TILE_MAX) bad = true;
                if (bad) break;
                if (dd <= rad * rad && (best < 0 || dd < bd)) {
                    best = i;
                    bd = dd;
                }
            }
            if (bad) {
                atomicOr(&S.bad, 1);
                continue;
            }
            if (best >= 0) {
                bool ch = false;
                put(row, UNIT_COL_ZOMBIE_NOISE_Y, S.noise[best][0], ch);
                put(row, UNIT_COL_ZOMBIE_NOISE_X, S.noise[best][1], ch);
                put(row, UNIT_COL_ZOMBIE_NOISE_TICKS, tmo, ch);
                mark(V, r, ch);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            // the live targets: (y, x) -> the most timeout left, in first-seen order
            int nl = 0;
            for (int r = 0; r < U; ++r) {
                if (!V.zsel[r]) continue;
                const int64_t* row = row_of(V, r);
                if (row[UNIT_COL_ZOMBIE_ACTIVATED] || row[UNIT_COL_ZOMBIE_NOISE_TICKS] <= 0)
                    continue;
                const int64_t y = row[UNIT_COL_ZOMBIE_NOISE_Y], x = row[UNIT_COL_ZOMBIE_NOISE_X];
                const int64_t t = row[UNIT_COL_ZOMBIE_NOISE_TICKS];
                int k = 0;
                while (k < nl && !(V.live[k * 3] == y && V.live[k * 3 + 1] == x)) ++k;
                if (k == nl) {
                    V.live[k * 3] = y;
                    V.live[k * 3 + 1] = x;
                    V.live[k * 3 + 2] = t;
                    ++nl;
                } else if (t > V.live[k * 3 + 2]) {
                    V.live[k * 3 + 2] = t;
                }
            }
            const int64_t keep_n = V.tab[UNIT_TAB_HORDE_NOISE_FIELDS_MAX];
            if (nl > keep_n) {
                for (int r = 0; r < U; ++r) {
                    if (!V.zsel[r]) continue;
                    int64_t* row = row_of(V, r);
                    if (row[UNIT_COL_ZOMBIE_ACTIVATED] || row[UNIT_COL_ZOMBIE_NOISE_TICKS] <= 0)
                        continue;
                    const int64_t y = row[UNIT_COL_ZOMBIE_NOISE_Y];
                    const int64_t x = row[UNIT_COL_ZOMBIE_NOISE_X];
                    int k = 0;
                    while (k < nl && !(V.live[k * 3] == y && V.live[k * 3 + 1] == x)) ++k;
                    // its rank by (-ticks, y, x): kept when fewer than keep_n rank before
                    int64_t rank = 0;
                    for (int j = 0; j < nl; ++j) {
                        const int64_t tj = V.live[j * 3 + 2], tk = V.live[k * 3 + 2];
                        if (tj > tk || (tj == tk && (V.live[j * 3] < y ||
                                                     (V.live[j * 3] == y &&
                                                      V.live[j * 3 + 1] < x))))
                            ++rank;
                    }
                    if (rank >= keep_n) {
                        bool ch = false;
                        horde_settle(V, row, ch);
                        mark(V, r, ch);
                    }
                }
            }
        }
        __syncthreads();
    }
    // ---- pass 3: serial in row order; the block floods on a field's first need ----
    if (threadIdx.x == 0) {
        S.r = 0;
        S.phase = PH_START;
        S.cmd = CMD_DONE;
        if (!S.bad) advance(V, S);
    }
    __syncthreads();
    while (S.cmd == CMD_FLOOD) {
        flood_field(V, S);
        __syncthreads();
        if (threadIdx.x == 0) advance(V, S);
        __syncthreads();
    }
    if (threadIdx.x == 0 && S.bad) V.B[APPLIER_BOOK_FAULTS] += 1;
    (void)E;
}

}  // namespace horde_dev
}  // namespace breach_cuda
