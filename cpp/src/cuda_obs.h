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
#include <string>
#include <utility>
#include <vector>

namespace breach_cuda {

class UnitRecordList;   // cuda_unit_events.h (the door and door-tile tables)
class ObjectTable;      // cuda_object_table.h (the grenade and charge tables)
class UnitEventRing;    // cuda_unit_events.h (the event ring)

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

// ---------------------------------------------------------------------------
// #29 P8.2: THE OBSERVATION'S SIGHT (cpp/src/cuda_obs_sight.cu over the device
// functions of cuda_obs_sight.cuh, which every later block that asks a line,
// who sees whom or a round's way calls). Every env in ONE launch, on the tick
// stream, no sync. `solid` / `screen`: the (N, h, w) uint8 solid and sight
// screen planes; `faults`: (N,) int32, a bit set per env on a bad operand
// (1: a line end past the device sight's range; 2: a team value that is
// neither a marine team < n_teams nor the zombie team). Each refuses
// (std::invalid_argument) a null pointer, a size or a width not the compiled
// layout's.
//
// WHO SEES WHOM (vision.compute's VisionState), under the match's sight rule
// read from the obs table (N, OBS_TAB_W): sees (N, U, U) uint8, sees[e][o][t]
// = 1 iff store row o is among seen_by[row t] (both living, teams differ,
// vision.can_see); visible (N, n_teams + 1, U) uint8, visible[e][s][t] = 1 iff
// row t is in visible_by_team[team of slot s] (slot n_teams: the zombie team).
void obs_vision(int n_env, std::uintptr_t store, int n_units, int store_w, std::uintptr_t solid,
                std::uintptr_t screen, int h, int w, std::uintptr_t obs_table, int table_w,
                int n_teams, std::uintptr_t sees, std::uintptr_t visible,
                std::uintptr_t faults);
// vision.rays_clear in bulk: pairs (N, P, 4) int32 (y0, x0, y1, x1), out
// (N, P) uint8 (1 = clear), the sight form or physical.
void obs_rays(int n_env, std::uintptr_t solid, std::uintptr_t screen, int h, int w,
              bool physical, std::uintptr_t pairs, int n_pairs, std::uintptr_t out,
              std::uintptr_t faults);
// vision.window_clear in bulk: queries (N, Q, 4) int32 (y, x, wy, wx), out
// (N, Q, wh, ww) uint8; FROM (y, x) to each cell, or to_window FROM each cell.
void obs_windows(int n_env, std::uintptr_t solid, std::uintptr_t screen, int h, int w,
                 bool physical, bool to_window, std::uintptr_t queries, int n_q, int wh, int ww,
                 std::uintptr_t out, std::uintptr_t faults);
// line_of_fire.solid_in_way between the centre tiles of every (row a, row b)
// of the store: out (N, U, U) uint8 (1 = something solid in the way), over
// the cover table (N, cover_cap, OBS_COVER_W) int64 with (N,) int32 counts.
void obs_lof_pairs(int n_env, std::uintptr_t store, int n_units, int store_w,
                   std::uintptr_t solid, int h, int w, std::uintptr_t covers, int cover_cap,
                   int cover_w, std::uintptr_t cover_count, std::uintptr_t out,
                   std::uintptr_t faults);

// ---------------------------------------------------------------------------
// #29 P8.3: THE DEVICE TEAM PICTURE (cpp/src/cuda_obs_picture.cu over the
// block layout of cuda_obs_picture.cuh; src/simulation/device_picture.py).
// `state`: (N, state_w) int64, one block per env; `faults`: (N,) int64 bits
// (obs_layout.h OBS_PIC_FAULT_*). Every launch on the tick stream, no sync.
//
// TeamPicture.update for every env: the serials of the grenade / charge
// tables' new rows, the per-unit trackers and last_hit (the ring's hits),
// then per marine team the damage window, the seen deaths, the memory, the
// sightings (the device vision's visible table (N, n_teams + 1, U)) and the
// object tracker. `first`: the start_team_picture update (no events). `brk`:
// the check's breaks (OBS_PIC_BREAK_*), 0 in every real run. Returns true
// when the fault bits were deferred into the conductor's books read; false
// outside a conductor tick (the caller reads `faults`).
bool obs_picture(int n_env, std::uintptr_t store, int n_units, int store_w, UnitEventRing& ring,
                 ObjectTable& grenades, ObjectTable& charges, std::uintptr_t solid,
                 std::uintptr_t screen, int h, int w, std::uintptr_t tick_row,
                 std::uintptr_t visible, int n_teams, std::uintptr_t state, int state_w,
                 std::uintptr_t faults, bool first, int brk);
// The vector's P8.3 columns (hp, vel_x / vel_y, hit_*, the record counts),
// every env on its decision tick with its ended latch clear, beside
// obs_vector's block.
void obs_picture_vector(int n_env, std::uintptr_t store, int n_units, int store_w,
                        std::uintptr_t rowmap, int n_rows, int n_slots, int n_teams,
                        std::uintptr_t obs_table, int table_w, std::uintptr_t teams,
                        std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
                        std::uintptr_t visible, std::uintptr_t state, int state_w,
                        std::uintptr_t vector);
// TeamPicture.damage_last_s for every (env, team, store row): out (N,
// n_teams, U) float64 (the gate's; `reverse`: THE BREAK, the order turned).
void obs_picture_damage_sums(int n_env, std::uintptr_t store, int n_units, int store_w,
                             int n_teams, std::uintptr_t state, int state_w, std::uintptr_t out,
                             bool reverse);
// The compiled block layout's section offsets for (n_teams, n_units).
std::vector<std::pair<std::string, int64_t>> obs_picture_layout(int n_teams, int n_units);

// ---------------------------------------------------------------------------
// #29 P8.4a: THE LANES, AIM_OF, THE LINE OF FIRE, TICKS_TO_WITHIN (cuda_obs.cu)
// for every env on its decision tick with its ended latch clear, on the tick
// stream, no sync: the fire lanes of every marine team (lane_grid (N, T, gh,
// gw) int32) and every lane owner's own lane (lane_own (N, U, gh, gw) uint8);
// the served lanes plane of near_u; aim_of's uid per (team, store row) (aim
// (N, T, U) int32, OBS_AIM_NOT_ASKED where the list asks nothing); the line
// of fire and ticks_to_within per (row, store row) of an enemy seen with a
// record (lof (N, n_rows, U) uint8 bits OBS_LOF_*, ttw (N, n_rows, U) int32).
// Reads the device vision's visible table, the picture's block, the cover
// table. `faults`: (N,) int32 bits OBS_AIM_FAULT_*; `brk`: the check's breaks
// (OBS_AIM_BREAK_*), 0 in every real run.
void obs_aims(int n_env, std::uintptr_t store, int n_units, int store_w,
              const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
              int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t tick_row,
              std::uintptr_t params, std::uintptr_t ended, std::uintptr_t solid,
              std::uintptr_t screen, int h, int w, std::uintptr_t covers, int cover_cap,
              int cover_w, std::uintptr_t cover_count, std::uintptr_t visible,
              std::uintptr_t state, int state_w, std::uintptr_t lane_grid,
              std::uintptr_t lane_own, int gh, int gw, std::uintptr_t aim, std::uintptr_t lof,
              std::uintptr_t ttw, std::uintptr_t near_u, std::uintptr_t faults, int brk);

// ---------------------------------------------------------------------------
// #29 P8.4b: THE LIST (cpp/src/cuda_obs_list.cu): obs_spec_v1.build_list for
// every env on its decision tick with its ended latch clear, on the tick
// stream, no sync: list (N, n_rows, n_entries, OBS_F) float32, list_mask (N,
// n_rows, n_entries) uint8, list_uid (N, n_rows, n_entries) int32; the five
// blocks each sorted by (squared centre distance, id) and cut at its
// capacity. Reads the store, the device tables, the obs table and the team
// rows, the device vision (visible (N, T + 1, U), sees (N, U, U)), the
// picture's block, P8.4a's answers (aim, lof, ttw, lane_own), the grenade /
// charge / door tables, the is_vacuum / is_ambient planes (N, h, w) and the
// wired flags (N, wired_cap) per door table row. The columns of later
// patches (on_next_tick, prev_aim, the order block) are written 0. `faults`:
// (N,) int32 bits OBS_LIST_FAULT_*; `brk`: the check's breaks
// (OBS_LIST_BREAK_*), 0 in every real run.
void obs_list(int n_env, std::uintptr_t store, int n_units, int store_w,
              const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
              int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t teams,
              std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
              std::uintptr_t visible, std::uintptr_t sees, std::uintptr_t state, int state_w,
              std::uintptr_t aim, std::uintptr_t lof, std::uintptr_t ttw, std::uintptr_t lane_own,
              int gh, int gw, ObjectTable& grenades, ObjectTable& charges, UnitRecordList& doors,
              UnitRecordList& door_tiles, std::uintptr_t is_vacuum, std::uintptr_t is_ambient,
              int h, int w, std::uintptr_t wired, int wired_cap, int n_entries,
              std::uintptr_t list, std::uintptr_t list_mask, std::uintptr_t list_uid,
              std::uintptr_t faults, int brk);

// ---------------------------------------------------------------------------
// #29 P8.4c: THE FOUR MASKS (cpp/src/cuda_obs_list.cu): action_spec_v1.
// build_masks for every env on its decision tick with its ended latch clear,
// on the tick stream, no sync, AFTER obs_list (the aim mask reads the list's
// type column and its mask): mask_feet (N, n_rows, OBS_MASK_N_FEET), mask_aim
// (N, n_rows, OBS_MASK_AIM_TOKENS + n_entries), mask_trigger (N, n_rows,
// OBS_MASK_N_TRIGGER), mask_hand (N, n_rows, OBS_MASK_HAND_THROW0 + n_kinds),
// uint8. Every hand bit is ONE unit-library gate (the use bit over the door
// tables through unit_lib::reach_key). `faults`: (N,) int32 bits
// OBS_MASK_FAULT_*; `brk`: the check's breaks (OBS_MASK_BREAK_*), 0 in every
// real run.
void obs_masks(int n_env, std::uintptr_t store, int n_units, int store_w,
               const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
               std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
               std::uintptr_t list, std::uintptr_t list_mask, int n_entries,
               UnitRecordList& doors, UnitRecordList& door_tiles, int n_kinds,
               std::uintptr_t mask_feet, std::uintptr_t mask_aim, std::uintptr_t mask_trigger,
               std::uintptr_t mask_hand, std::uintptr_t faults, int brk);

// ---------------------------------------------------------------------------
// #29 P8.8: THE FULL VIEW (cpp/src/cuda_obs_list.cu): obs_spec_v1.
// observe_full_rows for every env on its decision tick with its ended latch
// clear, on the tick stream, no sync: full_aim (N, 2, U) int32 (FullView.aim
// along each living unit's facing / facing target), full_list (N, n_rows,
// OBS_FULL_ENTRIES, OBS_FULL_F) float32, full_mask / full_uid (N, n_rows,
// OBS_FULL_ENTRIES) uint8 / int32, full_match (N, n_rows, 2 T + 3) float32,
// full_watched (N, n_rows, 25, 25) uint8, for every row whose roster place
// holds a unit (dead or turned included). Reads what the list reads (the
// list's entry builders are called, never copied) plus the cover table, the
// sight screen and the place table (N, place_cap, OBS_PLACE_W) int64 (the
// room graph's places and their centres) with its (N,) counts. `faults`: (N,)
// int32 bits OBS_FULL_FAULT_*; `brk`: the check's breaks (OBS_FULL_BREAK_*),
// 0 in every real run.
void obs_full(int n_env, std::uintptr_t store, int n_units, int store_w,
              const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
              int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t teams,
              std::uintptr_t tick_row, std::uintptr_t params, std::uintptr_t ended,
              std::uintptr_t visible, std::uintptr_t sees, std::uintptr_t state, int state_w,
              std::uintptr_t lane_own, int gh, int gw, ObjectTable& grenades,
              ObjectTable& charges, UnitRecordList& doors, UnitRecordList& door_tiles,
              std::uintptr_t is_vacuum, std::uintptr_t is_ambient, std::uintptr_t solid,
              std::uintptr_t screen, int h, int w, std::uintptr_t wired, int wired_cap,
              std::uintptr_t covers, int cover_cap, int cover_w, std::uintptr_t cover_count,
              std::uintptr_t places, int place_cap, int place_w, std::uintptr_t place_count,
              std::uintptr_t full_aim, std::uintptr_t full_list, std::uintptr_t full_mask,
              std::uintptr_t full_uid, std::uintptr_t full_match, std::uintptr_t full_watched,
              std::uintptr_t faults, int brk);

// #29 P8.5a: THE WINDOWS (cuda_obs_windows.cu) for every env on its decision
// tick with its ended latch clear, on the tick stream, no sync: the eight
// physics planes of near_f and far_f (from the RESIDENT planes temperature,
// fire, gas (n_gases, h, w), atmosphere, wind_x / wind_y, solid, through
// `far_raw` (N, 8, hb, wb) float64, the far blocks' raw reductions), near_u
// watched / sees_target (the picture's records `state`, the visible table,
// the decoder's `aimed` (N, n_rows) uid, the list just built), far_u mates /
// enemies / records, and the vector's air block and felt_temp (the heat /
// rad_flux_sweep planes through the unit library's felt temperature over the
// device tables). Faults OR into `faults` (N,) int32 (OBS_WIN_FAULT_*);
// `brk`: the check's breaks (OBS_WIN_BREAK_*), 0 in every real run.
void obs_planes(int n_env, std::uintptr_t store, int n_units, int store_w,
                const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
                int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t tick_row,
                std::uintptr_t params, std::uintptr_t ended, std::uintptr_t temperature,
                std::uintptr_t fire, std::uintptr_t gas, int n_gases, std::uintptr_t atmosphere,
                std::uintptr_t wind_x, std::uintptr_t wind_y, std::uintptr_t heat,
                std::uintptr_t rad, std::uintptr_t solid, std::uintptr_t screen, int h, int w,
                std::uintptr_t far_raw, int hb, int wb, std::uintptr_t visible,
                std::uintptr_t state, int state_w, std::uintptr_t aimed, std::uintptr_t list_uid,
                std::uintptr_t list_mask, int n_entries, std::uintptr_t near_f,
                std::uintptr_t near_u, std::uintptr_t far_f, std::uintptr_t far_u,
                std::uintptr_t vector, std::uintptr_t faults, int brk);

// ---------------------------------------------------------------------------
// #29 P8.7: THE DEVICE BOOKS (cpp/src/cuda_obs_books.cu; src/simulation/
// device_books.py). The reward books of profile v1, the old counters, the
// served reward groups, termination and the ended latch, ONE THREAD PER ENV
// over the tick's event ring in emission order, on the tick stream, after the
// team picture (`pic_state`: its bars) and the observation's blocks. `state`:
// (N, state_w) int64, one block per env; the served arrays reward (N, n_rows,
// OBS_BK_GROUPS) / counters (N, n_rows, OBS_BK_K) / team_* (N, n_teams)
// float32, rows_ended (N, n_rows) / terminated / truncated (N,) uint8 written
// on a decision tick and on the ending tick; `ended` (N,) uint8 the ended
// latch, set at the ending tick (a set env's books are frozen). `words`
// (3, N) int64: the RewardBooksMismatch count, the fault bits
// (OBS_BK_FAULT_*), ended this tick. `first`: the books' constructor (the
// snapshot, the standing, the living teams). `brk`: the check's breaks
// (OBS_BK_BREAK_*), 0 in every real run. Returns true when the words were
// deferred into the conductor's books read.
bool obs_books(int n_env, std::uintptr_t store, int n_units, int store_w, UnitEventRing& ring,
               std::uintptr_t rowmap, int n_rows, int n_slots, int n_teams,
               std::uintptr_t obs_table, int table_w, std::uintptr_t tick_row,
               std::uintptr_t params, std::uintptr_t ended, std::uintptr_t pic_state, int pic_w,
               std::uintptr_t state, int state_w, std::uintptr_t reward, std::uintptr_t counters,
               std::uintptr_t team_balance, std::uintptr_t team_uncredited,
               std::uintptr_t team_outcome, std::uintptr_t rows_ended, std::uintptr_t terminated,
               std::uintptr_t truncated, std::uintptr_t words, bool first, int brk);
// The compiled books block's section offsets for (n_teams, n_units, n_rows).
std::vector<std::pair<std::string, int64_t>> obs_books_layout(int n_teams, int n_units,
                                                              int n_rows);

// ---------------------------------------------------------------------------
// #29 P8.6a: THE DEVICE DECODER (cpp/src/cuda_obs_decoder.cu; src/simulation/
// device_decoder.py), on the tick stream, no sync. obs_dec_check: once per
// decision, the (N, n_rows, 5) int32 decisions of the envs `due` ((N,)
// uint8) range-checked into `faults` ((N, OBS_DEC_FAULT_W) int64), counted
// against the served masks into `masked` ((N, n_rows) uint8), resolved
// against `list_uid` ((N, n_rows, n_entries) int32) into the decision part of
// `state` ((N, n_rows, OBS_DEC_S_W) int64). obs_decode: every tick at the
// conductor's tick head, the latch columns of every decoder-driven row
// (`drive` (N, n_teams) uint8) written into the store, the slot part of
// `state`, `aimed` ((N, n_rows) int32); returns whether `faults` was deferred
// into the conductor's books read. obs_dec_horde: per env whose `mode` ((N,)
// int32) is the card's, the HORDE stage's latch row ((N, row_w) int64: mode,
// a flag per store row) of every living zombie. obs_dec_obs: on a decision
// tick, the vector's prev block and the list's prev_aim / on_next_tick.
// obs_dec_probe (TEST-ONLY): per offset (qx, qy, none, onto reach) into out
// (n, 4) int64; `div_brk` the check's break.
void obs_dec_check(int n_env, int n_rows, int n_slots, std::uintptr_t decisions,
                   std::uintptr_t due, std::uintptr_t list_uid, int n_entries,
                   std::uintptr_t mask_feet, std::uintptr_t mask_aim, std::uintptr_t mask_trigger,
                   std::uintptr_t mask_hand, int n_kinds, std::uintptr_t dec_table,
                   std::uintptr_t state, std::uintptr_t masked, std::uintptr_t faults);
bool obs_decode(int n_env, std::uintptr_t store, int n_units, int store_w,
                const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
                int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t tick_row,
                std::uintptr_t ended, std::uintptr_t pic_state, int pic_w, std::uintptr_t dec_table,
                std::uintptr_t drive, std::uintptr_t state, std::uintptr_t aimed,
                std::uintptr_t faults, int brk);
void obs_dec_horde(int n_env, std::uintptr_t store, int n_units, int store_w,
                   std::uintptr_t mode, std::uintptr_t rows, int row_w);
void obs_dec_obs(int n_env, std::uintptr_t store, int n_units, int store_w,
                 const UnitTablesDevice& tables, std::uintptr_t rowmap, int n_rows, int n_slots,
                 int n_teams, std::uintptr_t obs_table, int table_w, std::uintptr_t tick_row,
                 std::uintptr_t params, std::uintptr_t ended, std::uintptr_t pic_state, int pic_w,
                 std::uintptr_t state, std::uintptr_t aimed, std::uintptr_t list_mask,
                 std::uintptr_t list_uid, int n_entries, std::uintptr_t list,
                 std::uintptr_t vector);
void obs_dec_probe(int n, std::uintptr_t dx, std::uintptr_t dy, std::uintptr_t out, int div_brk);

// The compiled layout's fingerprint (obs_layout.h): what device_obs checks.
const char* obs_layout_fingerprint();

}  // namespace breach_cuda
