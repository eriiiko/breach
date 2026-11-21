#pragma once
// ==========================================================================
// #29 P8.0: THE DEVICE OBSERVATION's kernels (src/simulation/device_obs.py),
// Class T. Observation spec v1's blocks built on the card inside the device
// conductor's tick, every env in one launch per block, on THE tick stream;
// the host builders (obs_spec_v1, rl_env.EnvRunner.row_flags) are the
// reference. The observation is not synced state: nothing here writes a
// synced buffer, and nothing syncs. Its constants come only from the
// GENERATED obs_layout.h (tools/gen_obs_layout.py).
//
// The observation's device code lives in its own TU / headers (cuda_obs.*),
// never in the unit library (cuda_unit_lib.cuh).
// ==========================================================================
#include <cstdint>

namespace breach_cuda {

class UnitRecordList;   // cuda_unit_events.h (the door and door-tile tables)

// P8.0's block, for every env whose decision tick it is (the tick row's tick
// against the per-env parameter row: the episode's first tick, the
// decimation) and whose ended latch is clear:
//   - the door plane (N, h, w) int32 scratch: per living door of the door
//     table, open 1 / closed 2 over its span tiles (the max), else 0;
//   - per row t*S+s of the row map (N, n_rows) int32 (a store row or -1):
//     valid / in_match (N, n_rows) uint8; and, at the row's unit centre, the
//     near window's wall and door planes of near_u (N, n_rows, 5, 25, 25)
//     uint8, the far window's wall share of far_f (N, n_rows, 10, 25, 25)
//     float16 (as its bit patterns) and its door plane of far_u (N, n_rows,
//     5, 25, 25) uint8. A row that is not valid reads 0 in every plane.
// Refuses (std::invalid_argument) a null pointer or a non-positive size.
void obs_block(int n_env, std::uintptr_t store, int n_units, int store_w,
               std::uintptr_t rowmap, int n_rows, int n_slots,
               std::uintptr_t solid, int h, int w,
               UnitRecordList& doors, UnitRecordList& door_tiles, std::uintptr_t door_plane,
               std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
               std::uintptr_t near_u, std::uintptr_t far_f, std::uintptr_t far_u,
               std::uintptr_t valid, std::uintptr_t in_match);

class UnitTablesDevice; // cuda_unit_store.h (the device tables)

// #29 P8.1's block, for every env on its decision tick with its ended latch
// clear: the vector (N, n_rows, OBS_V) float32 columns of the store and the
// tables (device_obs.BUILT["vector"]: body, statuses, kit, the clock, the
// team's count and ammo), from the store, the device tables, the obs table
// (N, OBS_TAB_W) int64 and the team rows (N, n_teams, OBS_TEAM_W) int64 (the
// generated obs_layout.h). Columns no patch owns yet are left as they are.
// Refuses (std::invalid_argument) a null pointer, a size or a table width
// that is not the compiled layout's.
void obs_vector(int n_env, std::uintptr_t store, int n_units, int store_w,
                const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
                int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t teams,
                std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
                std::uintptr_t vector);

// #29 P8.1 TEST-ONLY probes of the float kit (cuda_obs_float.cuh) on host
// arrays (their own allocations, a device sync): the double -> float16 bits
// (via_f32: THE BREAK, through float32), the double -> float32 cast, a*b + c
// (fused: THE BREAK, an explicit fma), the squash (float64 and its float32).
void obs_probe_half(const double* x, uint16_t* out, int n, bool via_f32);
void obs_probe_f32(const double* x, float* out, int n);
void obs_probe_muladd(const double* a, const double* b, const double* c, double* out, int n,
                      bool fused);
void obs_probe_squash(const double* x, double* out64, float* out32, int n);

// The compiled layout's fingerprint (obs_layout.h): what device_obs checks.
const char* obs_layout_fingerprint();

}  // namespace breach_cuda
