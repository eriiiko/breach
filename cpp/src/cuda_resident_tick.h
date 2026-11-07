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
#include <utility>
#include <vector>

class PhysicsEngine;
struct ResidentFields;

namespace breach_cuda {

// W5 + W3 + the seal + the copy on the resident planes (f.water_depth,
// f.atmosphere, f.solid, f.gas, f.water_before, f.dyn_permeability,
// f.gas_energy, f.thermal_solid, f.is_vacuum, f.is_ambient when `ambient`).
// The books (e_water_evac_export_sum ASSIGNED, the trace mint / clip / neg
// books ADDED) are folded by the tick's resident_tail.
void resident_water_tail(const PhysicsEngine& eng, const ResidentFields& f,
                         const bool* gas_conservative, int steam_idx,
                         float sim_time, double ceiling_h, double flood_eps,
                         double ratio_cap, double boil_rate,
                         double boil_p_thresh, double steam_yield,
                         bool ambient, int32_t t_amb_raw);

// The combustion core on the resident planes. `ignition_table` is the
// per-material ignition_temp_q16 column already narrowed to int32 (the CPU
// path's `ignition_temp_q16[material].astype(np.int32)`, gathered here with
// numpy's negative-index wrap). Skipped, as on the CPU, when the burn cap
// quantizes to <= 0 or an index is outside the gas block.
void resident_combustion(const PhysicsEngine& eng, const ResidentFields& f,
                         int o2_idx, int inert_n2_idx, int black_smoke_idx,
                         int draw_r, float sim_time,
                         const int32_t* ignition_table, int n_materials,
                         bool ambient, int32_t t_amb_q);

// The light inputs of THIS tick (only while the engine requests light): the
// cone rows, the sky and the emit_mult rows, validated by the sweep's own
// host door and projected by its own builders, then uploaded. Called at the
// tick's entry; resident_tail's sweep reads them.
void resident_upload_light(const PhysicsEngine& eng, const ResidentFields& f,
                           const int32_t* gas_light_absorb_q16,
                           const int32_t* gas_light_glow_q16,
                           const int64_t* light_cones, int n_light_cones,
                           const int64_t* light_sky,
                           const int64_t* light_emit_mult, int n_light_emit_mult);

// The bulk sum, fire, the sweep and temperature; then the ONE books D2H and
// the folds; returns the destroyed walls (row-major).
std::vector<std::pair<int, int>> resident_tail(
    const PhysicsEngine& eng, const ResidentFields& f,
    const bool* gas_conservative, int o2_idx, float sim_time,
    bool ambient, int32_t t_amb_q, int32_t k_leak_q, int64_t rad_amb_vacuum_q,
    const int32_t* gas_heat_absorb_q16,
    const int32_t* gas_light_absorb_q16, const int32_t* gas_light_glow_q16);

// Telemetry for the gates (the eos_resident_calls idiom): how many times each
// resident stage ran in this process, and how many of those did real work —
// [0] water tail calls, [1] water-tail ticks that boiled or evacuated a cell,
// [2] combustion calls, [3] fire calls, [4] fire ticks whose gate opened,
// [5] sweep calls, [6] sweep calls with light, [7] temperature calls,
// [8] books downloads. A gate that sees these move knows the device ran the
// stage (a silently host "device run" would make a bit-identity gate vacuous).
constexpr int RESIDENT_STAGE_COUNTERS = 9;
void resident_stage_counters(long long out[RESIDENT_STAGE_COUNTERS]);

}  // namespace breach_cuda
