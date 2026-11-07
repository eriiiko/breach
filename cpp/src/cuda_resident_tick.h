#pragma once
// ============================================================================
// #29 P4a — THE RESIDENT TICK'S DRIVERS (docs/residency_batching_design_
// 2026-11-07.md §4.2, §4.3's slot-7 row, §9's P4a row).
// ============================================================================
//
// Slot 7 on the resident path used to run its tail on the host mirror: the
// water tail (W5 / W3), combustion, and step_tail's fire + radiation sweep +
// temperature, each a host bracket or a per-call twin that allocated,
// uploaded, launched, downloaded and freed every tick. These drivers run the
// same stages on the PERSISTENT device fields (GameMap's resident CuPy
// buffers, handed in as raw device addresses) through the families' launch
// cores (cuda_resident.h), in the CPU tick's order:
//
//   resident_water_tail   W5 + W3 + the seal + before := water_depth
//   (the EOS stage and the trace tail: PhysicsEngine::run_substeps_resident /
//    run_trace_tail_resident, unchanged drivers)
//   resident_combustion   the combustion core, its ignition plane gathered on
//                         the device from `material` and a per-material table
//   resident_tail         the bulk sum, fire (its early exit a device gate),
//                         the radiation sweep (heat, and light when the engine
//                         requests it), temperature
//
// with ZERO plane transfers between them. Persistent scratch, the static
// tables (the E° / L° tables, the gas optical columns, the ignition table,
// gas_conservative) and the per-env scalars live in this TU's state, keyed by
// the grid; a table or scalar is uploaded only when its host value changes
// (memcmp against the copy last uploaded) — never per tick in steady state.
// Every stage's books land in ONE device arena; resident_tail ends with ONE
// small D2H of it (and of the head of the destroyed-wall list), folds each
// book into the solver member the CPU tick writes, and returns the destroyed
// walls in row-major order. The light inputs (cones, sky, emit_mult: render
// state set before the tick) are uploaded by resident_upload_light at the
// tick's entry, beside GameMap.from_host.
//
// Plain C++ declarations (no CUDA types) so physics_engine.cpp (cl.exe) can
// include it. Compiled only when BREACH_CUDA.
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

class PhysicsEngine;
struct ResidentFields;
struct ResidentTickArgs;

namespace breach_cuda {

// #29 P5: THE ONE TICK STREAM (cuda_batch.cuh's tstream), as an opaque handle
// so plain C++ can hold it: created once per process (a blocking stream), every
// resident launch core and every in-tick copy / memset runs on it.
void* tick_stream_handle();

// #29 P5: THE TICK'S HOST SYNC POINTS go through here (the EOS stage's counted
// max(n_sub) read, the tail's books D2H) — a plain stream sync, unless the
// capture probe is on: then the segment captured since the last point is
// ended, instantiated and launched (so the tick still runs), its node count
// and the point's name recorded, and capture restarts in GLOBAL mode. Any
// operation the capture refuses (a legacy-stream copy, a sync, a malloc, a
// pageable copy) fails the CUDA call that made it.
void tick_sync_point(const char* name);
// TEST-ONLY (the capture-mode test): begin capturing the tick stream in global
// mode; end returns the segments' summary. abort ends a capture a thrown tick
// left open.
void resident_capture_begin();
struct CaptureSummary {
    int segments = 0;
    long long nodes = 0;
    std::vector<std::string> sync_points;   // the named breaks, in order
};
CaptureSummary resident_capture_end();
void resident_capture_abort();

// #29 P5: one FOLD-TABLE row as an env reads it this tick (design §4.2 "n_sub
// per env"): the host builds the table once per config (every n_sub in
// [1, N_SUB_MAX], range-checked at build — a row out of range is marked
// k_flux_q == 0 and an env that selects it raises its failure flag); the
// device pre-stage picks each env's row from that env's own reductions.
struct EnvSub {
    int32_t n_sub;        // the env's substep count, 1..N_SUB_MAX
    int32_t dt_s_q;       // quantize(dt / n_sub) — the substep dt
    int32_t k_flux_q;     // energy_flux_scalar_folds(..., n_sub).k_flux_q (0: refused row)
    int32_t c_local_q;    // the env's c_LOCAL (telemetry: dbg_last_c_local_q)
    int64_t flux_pu_cap;  // FLUX_MAG_CAP / k_flux_q
};

// #29 P5: THE BATCH. Every driver below runs over an (N, h, w) batch of
// homogeneous envs (one grid shape, one boundary type, one config): the
// planes in `f` are the batch's buffers (env 0's addresses, the envs
// contiguous; f.n_env = N) and `engs` holds the N envs' engines — engs[0]
// supplies the config (dials, tables) and each env's books fold into its own
// engine. N = 1 is the played single-instance game (`--cuda --resident`): ONE
// code path, never a second N = 1 copy. The books of every stage land per
// env in ONE device arena and come back in ONE D2H at resident_tail.
using EngineList = std::vector<const PhysicsEngine*>;

// The water stage: the substep loop (a.water_n_sub substeps of a.water_wdt)
// and W5 + W3 + the seal + the copy, on the resident planes (f.water_depth,
// f.flow_vx, f.flow_vy, f.floor_height, f.atmosphere, f.solid, f.gas,
// f.water_before, f.dyn_permeability, f.gas_energy, f.thermal_solid,
// f.is_vacuum, f.is_ambient when ambient). a.water_active: (N,) uint8, the
// envs whose water is live this tick (a dormant env does nothing); nullptr =
// every env. The books (e_water_evac_export_sum ASSIGNED, the trace mint /
// clip / neg books ADDED) are folded by the tick's resident_tail.
void resident_water(const EngineList& engs, const ResidentFields& f,
                    const ResidentTickArgs& a);

// The EOS stage over the batch (cuda_eos_resident.h's eos_step_resident: the
// device pre-stage, both masked substep loops, MG per env, the kick, the
// store, the energy flux), its books into the arena. Each env's blast source
// is its engine's eos.div_source. Returns the batch's max n_sub (the one
// counted per-tick host read).
int resident_eos(const EngineList& engs, const ResidentFields& f,
                 const ResidentTickArgs& a);

// The once-per-tick trace tail over the batch, its books into the arena
// (ADDED at the fold, after the EOS stage's assigned trace slots).
void resident_trace_tail(const EngineList& engs, const ResidentFields& f,
                         const ResidentTickArgs& a);

// The combustion core on the resident planes. `a.ignition_table` is the
// per-material ignition_temp_q16 column already narrowed to int32 (the CPU
// path's `ignition_temp_q16[material].astype(np.int32)`, gathered here with
// numpy's negative-index wrap). Skipped, as on the CPU, when the burn cap
// quantizes to <= 0 or an index is outside the gas block.
void resident_combustion(const EngineList& engs, const ResidentFields& f,
                         const ResidentTickArgs& a);

// The light inputs of THIS tick (only while the engine requests light; N == 1
// only — light is render state, refused in a batch until #81): the cone rows,
// the sky and the emit_mult rows, validated by the sweep's own host door and
// projected by its own builders, then uploaded. Called at the tick's entry;
// resident_tail's sweep reads them.
void resident_upload_light(const PhysicsEngine& eng, const ResidentFields& f,
                           const ResidentTickArgs& a);

// #29 P4b: slot 6's five stamp outputs on the resident planes (f.permeability,
// f.wave_absorb, f.heat_atten_q, f.light_atten_q in; f.obstacles,
// f.dyn_permeability, f.dyn_wave_absorb, f.dyn_heat_atten_q,
// f.dyn_light_atten_q out) from the host stamp rows of EVERY env (the env
// column set) -- one small H2D of the rows (n_rows x STAMP_ROW_COLS int32),
// the launch core, no sync. Called at the tick's entry.
void resident_stamp(const EngineList& engs, const ResidentFields& f,
                    const int32_t* rows, int n_rows);

// The bulk sum, fire, the sweep and temperature; then the ONE books D2H (every
// stage's books, every env) and the folds into each env's engine; returns
// each env's destroyed walls (row-major).
std::vector<std::vector<std::pair<int, int>>> resident_tail(
    const EngineList& engs, const ResidentFields& f, const ResidentTickArgs& a);

// Telemetry for the gates (the eos_resident_calls idiom): how many times each
// resident stage ran in this process, and how many of those did real work —
// [0] water tail calls, [1] water-tail ticks that boiled or evacuated a cell,
// [2] combustion calls, [3] fire calls, [4] fire ticks whose gate opened,
// [5] sweep calls, [6] sweep calls with light, [7] temperature calls,
// [8] books downloads, [9] water-tail ticks that boiled a cell (W5), [10]
// water-tail ticks that evacuated a cell (W3), [11] stamp ticks (#29 P4b: the
// device stamp ran -- the five planes were NOT uploaded), [12] EOS stage calls, [13]
// trace-tail calls. A gate that sees these move
// knows the device ran the stage (a silently host "device run" would make a
// bit-identity gate vacuous).
// [12] EOS stage calls, [13] trace-tail calls (#29 P5).
constexpr int RESIDENT_STAGE_COUNTERS = 14;
void resident_stage_counters(long long out[RESIDENT_STAGE_COUNTERS]);

// The sweep's Fleck plane ON DEMAND: the per-call twin copies it into the
// engine's RadiationSweep every call (its observable state, the
// fleck_plane_for_twin idiom); the resident tick keeps it on the device (it
// feeds nothing on the host), and this copies the LAST resident sweep's plane
// into the engine when a gate or a tool asks — never per tick.
void resident_sync_fleck_plane(const PhysicsEngine& eng, int h, int w);

// TEST-ONLY (the gate's PART 1g): the resident combustion's ignition-plane
// gather (rt_ignition, the kernel the tick launches) on host arrays — so the
// law `ignition_temp_q16[material].astype(int32)`, numpy's negative-index
// wrap included, is held to numpy directly. Allocates and copies: never on
// the hot path.
void resident_ignition_probe(const int8_t* material, int n,
                             const int32_t* table, int n_mat, int32_t* out);

}  // namespace breach_cuda
