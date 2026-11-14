#pragma once
// ============================================================================
// #29 P6.2 -- THE UNIT SLOT KERNELS (residency design §4.4b, "Re-check at
// P6.2's start", item 4): one free function per ported slot, in its slot
// family's TU, over the store (UnitStoreDevice: the rows, the counts, the
// input staging) and the device tables (UnitTablesDevice). The store class
// gains no method per slot.
//
//   cuda_unit_exposure.cu    slot 1c   exposure.decay_units
//   cuda_unit_impulse.cu     slot 1a   unit_impulse.integrate_all (+ turning.step)
//                            slot 9c2  exchange.apply_wind_force (+ add_wind)
//   cuda_unit_orders.cu      slot 1b'  unit_orders.apply_requests
//                            slot 10'  unit_orders.expire
//   cuda_unit_conversion.cu  slot 10 (ruleset tail) ai_zombie.convert_marines_to_zombies
//   cuda_unit_lib_probe.cu   TEST-ONLY: the device unit library on synthetic rows
//   cuda_unit_status.cu      slot 2b   status.tick_statuses           (#29 P6.3)
//   cuda_unit_exchange.cu    slot 9c   exchange.apply_environmental_damage
//                            slot 9c3  apply_teargas_blind + apply_poison_dose
//                                      + poison.tick (one kernel)    (#29 P6.3)
//   (the device damage pipeline they share: cuda_unit_damage.cuh)
//   cuda_unit_damage_probe.cu TEST-ONLY: apply_packet on synthetic rows (#29 P6.3)
//
// Every kernel is born (N, U): bgrid(U, N), one thread per row, blockIdx.y the
// env (cuda_unit_lib.cuh's unit_row rebases), a padding row (valid 0) skipped,
// on the tick stream; integer for integer its Python law; it reads no law
// constant from its arguments (every table value comes from the device
// tables). Each writes the per-env counts (coverage = rows it changed, faults =
// rows whose operands left the proved range -- left unchanged --, refused =
// status applications refused at the stack cap) into `out` (host arrays of N
// int64 each, any may be null); the call returns after the counts landed (a
// stream sync: allowed, nothing in the played tick calls these).
//
// Plain C++ declarations (no CUDA types) so bindings.cpp (cl.exe) can include
// it. Compiled only when BREACH_CUDA. CLASS T.
#include <cstddef>
#include <cstdint>

#include "cuda_unit_store.h"
#include "cuda_unit_events.h"     // #29 P6.3: the event ring

namespace breach_cuda {

struct UnitKernelCounts {
    int64_t* coverage = nullptr;   // (N,) rows the kernel changed
    int64_t* faults = nullptr;     // (N,) rows refused: an operand out of range
    int64_t* refused = nullptr;    // (N,) status applications refused (stack cap)
};

// Field planes a slot reads, as device pointers to (N, ...) arrays in the
// RESIDENT field's own dtype and currency -- exactly as ResidentBatch holds
// them (gmap._dev[name] is buffer[e]), so the device backend hands its
// buffers unchanged: gas (N, n_gases, h, w) int32 (the bulk planes Q16),
// wind_x / wind_y (N, h, w) int32 (Q16 m/s), solid (N, h, w) bool (1 byte).
struct UnitFieldPlanes {
    std::uintptr_t gas = 0;
    std::uintptr_t wind_x = 0;
    std::uintptr_t wind_y = 0;
    std::uintptr_t solid = 0;
    int n_gases = 0;
    int h = 0;
    int w = 0;
};

// Slot 1c (exposure.decay_units / decay_step) on every valid row; decay_q is
// the tables' exposure_decay_q.
void unit_exposure_decay(UnitStoreDevice& store, const UnitTablesDevice& tables,
                         const UnitKernelCounts& out);

// Slot 1a (unit_impulse.integrate_all): every valid LIVING row, the linear
// law (unit_impulse.integrate) then omega += dw and the turning law
// (turning.step, the live aim the target while a non-zero one is held).
void unit_integrate_turn(UnitStoreDevice& store, const UnitTablesDevice& tables,
                         const UnitKernelCounts& out);

// Slot 9c2 (exchange.apply_wind_force): every valid LIVING row, the wind load
// of its footprint (GameMap.footprint_wind over the planes) as one tick of
// impulse (unit_impulse.add_wind; prone: the standing surplus).
void unit_wind_impulse(UnitStoreDevice& store, const UnitTablesDevice& tables,
                       const UnitFieldPlanes& planes, const UnitKernelCounts& out);

// Slot 1b' (unit_orders.apply_requests): `requests` is (N, n_req, 5) int64
// host rows (uid, what, where, until, target), uid -1 = none (an env with
// fewer requests pads with it); staged through the store's input slot 0.
void unit_apply_orders(UnitStoreDevice& store, const UnitTablesDevice& tables,
                       const int64_t* requests, int n_req, const UnitKernelCounts& out);

// Slot 10' (unit_orders.expire): `ticks` is (N,) int64 host, each env's
// sim.tick after the advance; staged through input slot 0.
void unit_expire_orders(UnitStoreDevice& store, const UnitTablesDevice& tables,
                        const int64_t* ticks, const UnitKernelCounts& out);

// Slot 10's ruleset tail (ai_zombie.convert_marines_to_zombies): every valid
// row killed by a zombie rises as one.
void unit_convert_zombies(UnitStoreDevice& store, const UnitTablesDevice& tables,
                          const UnitKernelCounts& out);

// TEST-ONLY (cuda_unit_lib_probe.cu): the device unit library on the store's
// rows. op 0 composed_flags (a0 = after_sweep) -> out[.., 0..4] = can_move,
// can_act, can_aim, is_prone, bad; op 1 apply_status(kind a0, mag a1,
// duration a2, source a3) -> out[.., 0] = the ApplyResult, the rows changed in
// place; op 2 sweep_statuses, in place; op 3 scaled(row, h, a0) for every
// hook h -> out[.., h]. `out` is (N, U, UNIT_LIB_PROBE_K) int64 host; a
// padding row is left alone (out row zero).
constexpr int UNIT_LIB_PROBE_K = 16;
void unit_lib_probe(UnitStoreDevice& store, const UnitTablesDevice& tables, int op,
                    int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t* out,
                    const UnitKernelCounts& counts);

// ---- #29 P6.2m: slot 3, movement (cuda_unit_movement.cu) ------------------------
// Slot 3 on the CONTINUOUS path (Simulation._update_player_movement: the
// MOVE_DIR walk + the pushed velocity as one displacement, the slide rule, a
// held-no-aim walker's facing target; a pushed row moves by pv alone) on every
// valid LIVING row. `material` is a device pointer to the (N, h, w) int8
// GameMap.material planes (the resident field, as ResidentBatch holds it); the
// mobility column is the tables' material_mobility row.
void unit_movement(UnitStoreDevice& store, const UnitTablesDevice& tables,
                   std::uintptr_t material, int h, int w, const UnitKernelCounts& out);

// TEST-ONLY (cuda_unit_movement_probe.cu): the walk's library functions on
// synthetic rows. `args` is (n_rows, UNIT_MOVEMENT_PROBE_ARGS) int64 host, `out`
// (n_rows, UNIT_MOVEMENT_PROBE_OUT) int64 host; out[.., 3] = bad. op 0
// default_speed_q(n, total, class, hook, scale) -> out[0]; op 1
// move_speed_q16(cw, cf, side, back, t, cos) -> out[0]; op 2
// effort_scale_q16(e, e0) -> out[0]; op 3 facing_cos_q16(dx, dy, facing) ->
// out[0]; op 4 over `material` (h, w) int8 and `mobility` (n_mat,) int64 host:
// passable_block(fy, fx, fp) -> out[0], footprint_mobility -> out[1] n, out[2]
// total.
constexpr int UNIT_MOVEMENT_PROBE_ARGS = 6;
constexpr int UNIT_MOVEMENT_PROBE_OUT = 4;
void unit_movement_probe(int op, const int64_t* args, int n_rows, const int8_t* material,
                         int h, int w, const int64_t* mobility, int n_mat, int64_t* out);

// ---- #29 P6.3: the slots that hurt (the device apply_packet, the event ring) ----
// Their kernels append UnitHit / UnitKilled records to `ring` (begun by the
// caller at the tick's start) in the reference's emission order; `dropped`
// (N,) is each env's ring overflow after the kernel (the Python face refuses
// a run that dropped one). `refused` also counts, for slot 9c3, the poison
// doses clipped at the body's store (poison.SATURATION's hits).
struct UnitDamageCounts : UnitKernelCounts {
    int64_t* dropped = nullptr;    // (N,) events the ring could not hold
};

// The planes the damage slots read: device pointers to (N, ...) arrays in the
// resident fields' dtypes -- heat (N, h, w) int32 (Q16 heat units),
// rad_flux_sweep (N, h, w) int64 (the sweep's fine currency), gas (N, G, h, w)
// int32 (the trace planes in the TRACE currency).
struct UnitDamagePlanes {
    std::uintptr_t heat = 0;
    std::uintptr_t rad_flux = 0;
    std::uintptr_t gas = 0;
    int n_gases = 0;
    int h = 0;
    int w = 0;
};

// Slot 2b (status.tick_statuses): every valid LIVING row's statuses swept,
// their damage over time applied, counted down.
void unit_status_tick(UnitStoreDevice& store, const UnitTablesDevice& tables,
                      UnitEventRing& ring, const UnitDamageCounts& out);

// Slot 9c (exchange.apply_environmental_damage). `claims` is (N, n_claims, 4)
// int64 host rows (channel, y, x, attacker) -- damage.TickCredit, channel
// unit_events::UNIT_CREDIT_HEAT for the heat buffer, padded with
// UNIT_CREDIT_PAD -- staged through the store's input slot 1.
void unit_heat_damage(UnitStoreDevice& store, const UnitTablesDevice& tables,
                      UnitEventRing& ring, const UnitDamagePlanes& planes,
                      const int64_t* claims, int n_claims, const UnitDamageCounts& out);

// Slot 9c3 (exchange.apply_teargas_blind, exchange.apply_poison_dose,
// poison.tick, fused per row); the claims as for 9c (a gas slice's channel is
// its index).
void unit_gas_rows(UnitStoreDevice& store, const UnitTablesDevice& tables,
                   UnitEventRing& ring, const UnitDamagePlanes& planes,
                   const int64_t* claims, int n_claims, const UnitDamageCounts& out);

// TEST-ONLY (cuda_unit_damage_probe.cu): the device apply_packet on the
// store's rows, one packet per row: `packets` is (N, U, UNIT_DAMAGE_PROBE_W)
// int64 host rows (amount_q, dtype, attacker, indirect, expose, dir_x_q,
// dir_y_q, ap_q, cause, mark_killed_by_zombie); `out` (N, U, 2) int64 host:
// the applied damage and the kill flag (0 for a faulted or padding row).
constexpr int UNIT_DAMAGE_PROBE_W = 10;
void unit_damage_probe(UnitStoreDevice& store, const UnitTablesDevice& tables,
                       UnitEventRing& ring, const int64_t* packets, int64_t* out,
                       const UnitDamageCounts& counts);

}  // namespace breach_cuda
