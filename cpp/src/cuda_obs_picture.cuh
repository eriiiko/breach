#pragma once
// ==========================================================================
// #29 P8.3: THE DEVICE TEAM PICTURE's block and its readers (Class T). See
// src/simulation/device_picture.py (the layout, the normal form, the seam)
// and cpp/src/cuda_obs_picture.cu (the update, the vector columns). One env's
// block is ONE int64 row of PIC_W words whose sections (REC, UNIT, DMG,
// DMG_HD, OBJ, OBJ_N, OBJ_NEXT, SEEN, SEEN_N, SERIAL_NEXT, SIDE_G, KNOWN_G,
// SIDE_C, KNOWN_C, PARAM, TICK) are laid out from (T, U) here and in
// device_picture.layout alike (the build compares them). Every later block
// that reads the picture (the list, the reward's bars) reads it through
// these, never its own offsets.
// ==========================================================================
#include <cstdint>

#include <cuda_runtime.h>

#include "cuda_obs_float.cuh"
#include "obs_layout.h"

namespace breach_cuda {
namespace obs_pic {

using namespace obs_layout;

// The section offsets of one env's block (int64 words), from (T, U).
struct PicLayout {
    int64_t rec, unit, dmg, dmg_hd, obj, obj_n, obj_next, seen, seen_n, serial_next, side_g,
        known_g, side_c, known_c, param, tick, w;
};

__host__ __device__ inline PicLayout pic_layout(int T, int U) {
    PicLayout L{};
    int64_t o = 0;
    L.rec = o;          o += (int64_t)T * U * OBS_PIC_REC_W;
    L.unit = o;         o += (int64_t)U * OBS_PIC_UNIT_W;
    L.dmg = o;          o += (int64_t)T * OBS_PIC_DMG_CAP * OBS_PIC_DMG_W;
    L.dmg_hd = o;       o += (int64_t)T * 2;
    L.obj = o;          o += (int64_t)T * OBS_PIC_OBJ_CAP * 2;
    L.obj_n = o;        o += T;
    L.obj_next = o;     o += T;
    L.seen = o;         o += (int64_t)T * OBS_PIC_OBJ_CAP * 2;
    L.seen_n = o;       o += T;
    L.serial_next = o;  o += 1;
    L.side_g = o;       o += OBS_PIC_GRENADE_CAP;
    L.known_g = o;      o += 1;
    L.side_c = o;       o += OBS_PIC_CHARGE_CAP;
    L.known_c = o;      o += 1;
    L.param = o;        o += OBS_PIC_PARAM_W;
    L.tick = o;         o += 1;
    L.w = o;
    return L;
}

// Team t's record of store row r (its PRESENT word first).
__device__ __forceinline__ int64_t* rec_of(int64_t* blk, const PicLayout& L, int U, int t, int r) {
    return blk + L.rec + ((int64_t)t * U + r) * OBS_PIC_REC_W;
}
__device__ __forceinline__ const int64_t* rec_of(const int64_t* blk, const PicLayout& L, int U,
                                                 int t, int r) {
    return blk + L.rec + ((int64_t)t * U + r) * OBS_PIC_REC_W;
}
// Store row r's trackers.
__device__ __forceinline__ const int64_t* unit_of(const int64_t* blk, const PicLayout& L, int r) {
    return blk + L.unit + (int64_t)r * OBS_PIC_UNIT_W;
}

// TeamPicture.damage_last_s(team t, uid): the sum of the uid's entries of
// team t's damage window IN ITS DEQUE'S ORDER (the FIFO's subsequence of the
// uid), from 0, each damage float64(dq) / 65536 (exact), through the kit's
// unfused add. `reverse`: THE BREAK (the order turned round).
__device__ inline double damage_sum(const int64_t* blk, const PicLayout& L, int t, int64_t uid,
                                    bool reverse) {
    const int64_t head = blk[L.dmg_hd + 2 * t], len = blk[L.dmg_hd + 2 * t + 1];
    double s = 0.0;
    for (int64_t k0 = 0; k0 < len; ++k0) {
        const int64_t k = reverse ? len - 1 - k0 : k0;
        const int64_t* d = blk + L.dmg + ((int64_t)t * OBS_PIC_DMG_CAP +
                                          (head + k) % OBS_PIC_DMG_CAP) * OBS_PIC_DMG_W;
        if (d[OBS_PIC_DMG_VICTIM] != uid) continue;
        s = obs_f64::add(s, obs_f64::div((double)d[OBS_PIC_DMG_DQ], 65536.0));
    }
    return s;
}

}  // namespace obs_pic
}  // namespace breach_cuda
