#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include "atmosphere_solver.h"
#include "smoke_dynamics.h"
#include "fire_simulation.h"
#include "temperature_solver.h"
#include "raycaster.h"
#include "water_solver.h"
#include "eos_solver.h"
#include "combustion.h"
#include "physics_engine.h"
#include "bulk_transport.h"  // EOS refactor P1: expose bulk_flux_transport for direct unit test
#include "sky_exchange.h"    // sky-exchange: planetside volumetric O2 replenishment (per-tick host pass)
#include "fixed_point.h"   // Bedrock cliff-patch: expose smoke_cliff_count for unit test
#ifdef BREACH_HAS_CUDA
#include "cuda_hello.h"        // CUDA-S0: hello-world map kernel + device info
#include "cuda_spike.h"        // CUDA-S8a: residency spike (raw device pointer in)
#include "cuda_resident.h"     // CUDA-S8a Path B: water/smoke resident launch cores
#include "cuda_temperature.h"  // CUDA-S1: GPU temperature solver + backend flag
#include "cuda_raycaster.h"    // CUDA-S2: GPU directional raycaster (heat bit-identical)
#include "cuda_water.h"        // CUDA-S3: GPU water solver + backend flag
#include "cuda_smoke.h"        // CUDA-S4a: GPU smoke solver + backend flag
#include "cuda_fire.h"         // CUDA-S6: GPU fire solver + backend flag
#include "cuda_sl_advection.h" // EOS P6.2: fused 3-field SL advection + backend flag
#include "cuda_bulk_transport.h"  // EOS P6.1: GPU bulk donor-cell flux + backend flag
#include "cuda_kick_compression.h"  // EOS P6.4: GPU kick + compression work + backend flag
#include "cuda_mg_solve.h"     // EOS P6.3: GPU multigrid pressure solve + backend flag
#include "cuda_eos_step.h"     // EOS P6.5: chained full-eos.step dispatch predicate
#include "cuda_eos_resident.h" // S8a Path A: resident EOS telemetry + build parity
#include "cuda_combustion.h"   // EOS P6.9b: GPU two-gather combustion + backend flag
// CUDA-S5 cuda_wave.h / CUDA-S7 cuda_atmosphere.h RETIRED in EOS P6.0 — the
// wave+diffuse solvers they mirrored were replaced by the compressible EOS
// solve in P3 (docs/eos_p6_gpu_alignment_review.md §1.11).
#endif

namespace py = pybind11;

// Helper: extract raw pointer + dimensions from a 2D numpy array
template<typename T>
static std::tuple<T*, int, int> get_2d(py::array_t<T>& arr) {
    auto a = arr.template mutable_unchecked<2>();
    return {a.mutable_data(0, 0),
            static_cast<int>(a.shape(0)),
            static_cast<int>(a.shape(1))};
}

template<typename T>
static std::tuple<const T*, int, int> get_2d_const(const py::array_t<T>& arr) {
    auto a = arr.template unchecked<2>();
    return {a.data(0, 0),
            static_cast<int>(a.shape(0)),
            static_cast<int>(a.shape(1))};
}

// Helper: extract raw pointer + (h, w) from a 3D (h, w, 3) numpy array.
// Returns the H and W dimensions (channel count is the trailing axis).
template<typename T>
static std::tuple<T*, int, int> get_3d(py::array_t<T>& arr) {
    auto a = arr.template mutable_unchecked<3>();
    return {a.mutable_data(0, 0, 0),
            static_cast<int>(a.shape(0)),
            static_cast<int>(a.shape(1))};
}

PYBIND11_MODULE(breach_physics, m) {
    m.doc() = "Breach physics engine -- C++ accelerated simulation";

    // CUDA-S0: advertise whether this .pyd was built with the GPU backend, and
    // expose the hello-world map kernel + device info. HAS_CUDA lets the
    // bit-identity harness skip the GPU leg gracefully on a CPU-only build.
#ifdef BREACH_HAS_CUDA
    m.attr("HAS_CUDA") = true;
    m.def("cuda_available", &breach_cuda::available,
          "True if a CUDA device is present and usable.");
    m.def("cuda_device_info", &breach_cuda::device_info,
          "GPU name + compute capability + runtime/driver versions.");
    m.def("cuda_map_mul_q16", &breach_cuda::map_mul_q16,
          py::arg("in_"), py::arg("factor_q16"),
          "S0 hello-world: out[i] = mul_q16(in[i], factor_q16) computed on the "
          "GPU via the shared toolkit; bit-identical to the CPU mul_q16.");

    // CUDA-S8a: the residency spike. Given a raw device address (a CuPy array's
    // int(arr.data.ptr)) as uintptr_t, add 1 to each of n int32 elements IN
    // PLACE on the GPU — no malloc, no transfer. Proves CuPy-owned device
    // memory is directly launchable from the breach .pyd (shared primary
    // context), the primitive the whole S8a resident path rests on.
    m.def("cuda_spike_add1",
          [](std::uintptr_t dev_ptr, int n) { breach_cuda::spike_add1(dev_ptr, n); },
          py::arg("dev_ptr"), py::arg("n"),
          "S8a spike: int32 in-place +1 on a raw device pointer (CuPy .data.ptr).");

    // CUDA-S8a Path B: the two RESIDENT launch-loop entries. Each takes raw
    // device addresses (CuPy `.data.ptr` as uintptr_t) for the persistent fields,
    // owns its persistent scratch internally (allocated once, keyed by (h,w) — NO
    // per-substep/per-plane malloc/H2D/D2H), and syncs once. This is what kills the
    // substep-/plane-MULTIPLIED transfer tax; the launch cores are the SHARED
    // bodies the per-call water_step/smoke_step also run, so both paths are
    // bit-identical. dev_ptr == 0 -> nullptr (e.g. no floor / space-map ambient).
    m.def("water_substeps_resident",
          [](std::uintptr_t d_depth, std::uintptr_t d_vx, std::uintptr_t d_vy,
             std::uintptr_t d_floor, std::uintptr_t d_atm, std::uintptr_t d_solid,
             int h, int w, int n_sub, float wdt, float tilt_x, float tilt_y,
             float g, float damping, float dx, float k_p, float v_max,
             float depth_eps) {
              breach_cuda::water_substeps_resident(
                  reinterpret_cast<int32_t*>(d_depth),
                  reinterpret_cast<int32_t*>(d_vx),
                  reinterpret_cast<int32_t*>(d_vy),
                  reinterpret_cast<const int32_t*>(d_floor),
                  reinterpret_cast<const int32_t*>(d_atm),
                  reinterpret_cast<const bool*>(d_solid),
                  h, w, n_sub, wdt, tilt_x, tilt_y,
                  g, damping, dx, k_p, v_max, depth_eps);
          },
          py::arg("d_depth"), py::arg("d_vx"), py::arg("d_vy"),
          py::arg("d_floor"), py::arg("d_atm"), py::arg("d_solid"),
          py::arg("h"), py::arg("w"), py::arg("n_sub"), py::arg("wdt"),
          py::arg("tilt_x"), py::arg("tilt_y"),
          py::arg("g"), py::arg("damping"), py::arg("dx"), py::arg("k_p"),
          py::arg("v_max"), py::arg("depth_eps"),
          "S8a Path B: water substep loop resident on device buffers (no per-substep "
          "transfer). Device pointers are CuPy .data.ptr uintptr_t.");

    m.def("trace_smoke_resident",
          [](std::uintptr_t d_gas_base, std::uintptr_t d_wx, std::uintptr_t d_wy,
             std::uintptr_t d_solid, std::uintptr_t d_vac, std::uintptr_t d_perm,
             std::uintptr_t d_amb,
             int h, int w, int n_gases, int inert_n2_idx,
             py::array_t<bool> gas_conservative,
             py::array_t<float> gas_diffusion,
             py::array_t<float> gas_decay,
             float dt, float advection_rate, float wind_diffusion_scale) {
              auto gc = gas_conservative.unchecked<1>();
              auto gd = gas_diffusion.unchecked<1>();
              auto gdc = gas_decay.unchecked<1>();
              breach_cuda::trace_smoke_resident(
                  reinterpret_cast<int32_t*>(d_gas_base),
                  reinterpret_cast<const int32_t*>(d_wx),
                  reinterpret_cast<const int32_t*>(d_wy),
                  reinterpret_cast<const bool*>(d_solid),
                  reinterpret_cast<const bool*>(d_vac),
                  reinterpret_cast<const float*>(d_perm),
                  reinterpret_cast<const bool*>(d_amb),
                  h, w, n_gases, inert_n2_idx,
                  gc.data(0), gd.data(0), gdc.data(0),
                  dt, advection_rate, wind_diffusion_scale);
          },
          py::arg("d_gas_base"), py::arg("d_wx"), py::arg("d_wy"),
          py::arg("d_solid"), py::arg("d_vac"), py::arg("d_perm"), py::arg("d_amb"),
          py::arg("h"), py::arg("w"), py::arg("n_gases"), py::arg("inert_n2_idx"),
          py::arg("gas_conservative"), py::arg("gas_diffusion"), py::arg("gas_decay"),
          py::arg("dt"), py::arg("advection_rate"), py::arg("wind_diffusion_scale"),
          "S8a Path B: per-tick trace-plane smoke loop + decay resident on device "
          "(no per-plane transfer). gas_conservative/diffusion/decay are host (N,) "
          "columns; field pointers are CuPy .data.ptr uintptr_t.");

    // S8a Path A: resident-EOS telemetry (the gate's vacuousness guard) + the
    // TEST-ONLY device-MG-build parity probe (gate PART 1c).
    m.def("eos_resident_calls", &breach_cuda::eos_resident_calls,
          "S8a Path A: how many ticks ran the fully resident EOS chain.");

    m.def("eos_mg_build_parity",
          [](const EOSSolver& solver,
             py::array_t<int32_t> pstar, py::array_t<int32_t> div_u,
             py::array_t<int32_t> n_total, py::array_t<int32_t> p_prev,
             py::array_t<bool> solid, py::array_t<bool> is_vacuum,
             py::array_t<float> dyn_permeability,
             float dt, py::object is_ambient, int32_t p_amb,
             py::object sponge_sigma) {
              auto [ps, h, w]   = get_2d_const(pstar);
              auto [du, h2, w2] = get_2d_const(div_u);
              auto [nt, h3, w3] = get_2d_const(n_total);
              auto [pp, h4, w4] = get_2d_const(p_prev);
              auto [sol, h5, w5] = get_2d_const(solid);
              auto [vac, h6, w6] = get_2d_const(is_vacuum);
              auto [perm, h7, w7] = get_2d_const(dyn_permeability);
              const bool* amb = nullptr;
              py::array_t<bool> amb_arr;
              if (!is_ambient.is_none()) {
                  amb_arr = is_ambient.cast<py::array_t<bool>>();
                  auto aa = amb_arr.unchecked<2>();
                  amb = aa.data(0, 0);
              }
              const int32_t* sigma = nullptr;
              py::array_t<int32_t> sigma_arr;
              if (!sponge_sigma.is_none()) {
                  sigma_arr = sponge_sigma.cast<py::array_t<int32_t>>();
                  auto sg = sigma_arr.unchecked<2>();
                  sigma = sg.data(0, 0);
              }
              std::string report;
              const long long mism = breach_cuda::eos_mg_build_parity(
                  solver, ps, du, nt, pp, sol, vac, perm, h, w, dt,
                  amb, p_amb, sigma, &report);
              return py::make_tuple(mism, report);
          },
          py::arg("solver"), py::arg("pstar"), py::arg("div_u"),
          py::arg("n_total"), py::arg("p_prev"), py::arg("solid"),
          py::arg("is_vacuum"), py::arg("dyn_permeability"), py::arg("dt"),
          py::arg("is_ambient") = py::none(), py::arg("p_amb") = 0,
          py::arg("sponge_sigma") = py::none(),
          "S8a Path A gate PART 1c (TEST-ONLY): host mg_build_levels vs the "
          "production device build on identical inputs — returns (mismatched "
          "cell count, per-level report); 0 == bit-identical.");

    // CUDA-S1: the GPU temperature solver. The backend flag switches
    // PhysicsEngine::step_tail between the CPU and GPU temperature pass (the live
    // CPU fallback stays). cuda_temperature_step runs the 3-pass solver IN PLACE
    // on `temperature` for the isolated GPU-vs-CPU bit-identity gate.
    m.def("set_temperature_backend",
          [](bool use_cuda) { breach_cuda::set_temperature_backend_cuda(use_cuda); },
          py::arg("use_cuda"),
          "Switch PhysicsEngine's temperature pass to the GPU (True) or CPU (False).");
    m.def("get_temperature_backend",
          []() { return breach_cuda::temperature_backend_is_cuda(); },
          "True if the temperature pass currently runs on the GPU.");
    m.def("cuda_temperature_step",
          // Arg order: all REQUIRED args first, then the defaulted ones last
          // (pybind11 forbids a required arg after a defaulted one).
          [](py::array_t<int32_t> temperature, py::array_t<int32_t> heat,
             py::array_t<int32_t> heat_inv_shift, py::array_t<int32_t> face_shift,
             py::array_t<bool> solid, py::array_t<bool> is_vacuum,
             py::array_t<int32_t> atmosphere,
             int no_face, int cool_shift, int cool_shift_vacuum,
             float o2_vacuum_thresh, float c_v, float n_floor_heat,
             float gas_advection_rate, float t_max_phys,
             py::object n_bulk_obj, py::object wind_x_obj, py::object wind_y_obj,
             float dt, py::object thermal_solid_obj,
             py::object cool_shift_grid_obj, int cool_shift_floor) -> py::tuple {
              auto [temp, h, w]    = get_2d(temperature);
              auto [hp, h2, w2]    = get_2d_const(heat);
              auto [shift, h3, w3] = get_2d_const(heat_inv_shift);
              auto [sol, h4, w4]   = get_2d_const(solid);
              auto [vac, h5, w5]   = get_2d_const(is_vacuum);
              auto [atm, h6, w6]   = get_2d_const(atmosphere);
              auto fa = face_shift.unchecked<3>();
              const int32_t* fs = fa.data(0, 0, 0);
              // Optional n_bulk / wind_x / wind_y — None -> nullptr (the solver
              // then falls back to the atmosphere N proxy / skips Pass 0 advect).
              // Keep the extracted arrays alive in this scope.
              const int32_t* nb = nullptr;
              const int32_t* wx = nullptr;
              const int32_t* wy = nullptr;
              py::array_t<int32_t> nb_arr, wx_arr, wy_arr;
              if (!n_bulk_obj.is_none()) {
                  nb_arr = n_bulk_obj.cast<py::array_t<int32_t>>();
                  auto [nbp, hn, wn] = get_2d_const(nb_arr);
                  nb = nbp;
              }
              if (!wind_x_obj.is_none() && !wind_y_obj.is_none()) {
                  wx_arr = wind_x_obj.cast<py::array_t<int32_t>>();
                  wy_arr = wind_y_obj.cast<py::array_t<int32_t>>();
                  auto [wxp, hx, wxw] = get_2d_const(wx_arr);
                  auto [wyp, hy, wyw] = get_2d_const(wy_arr);
                  wx = wxp;
                  wy = wyp;
              }
              // THERMAL-MASS AXIS (P2): the per-medium THERMAL mask, OPTIONAL
              // exactly like the CPU TemperatureSolver binding — None -> nullptr
              // -> the kernel falls back to `solid`, the pre-patch behaviour, so
              // every existing direct caller (tests/cuda_conduction_check) keeps
              // its meaning. The lockstep gate passes the real mask on both sides.
              const bool* tsol = nullptr;
              py::array_t<bool> tsol_arr;
              if (!thermal_solid_obj.is_none()) {
                  tsol_arr = thermal_solid_obj.cast<py::array_t<bool>>();
                  auto [tsp, ht, wt] = get_2d_const(tsol_arr);
                  tsol = tsp;
              }
              // COOL-SHIFT AXIS: the per-tile decay shift, OPTIONAL by the
              // same idiom — None -> nullptr -> the kernel uses the
              // `cool_shift` scalar for every tile (the pre-axis behaviour),
              // so every existing direct caller keeps its exact meaning.
              const int32_t* csg = nullptr;
              py::array_t<int32_t> csg_arr;
              if (!cool_shift_grid_obj.is_none()) {
                  csg_arr = cool_shift_grid_obj.cast<py::array_t<int32_t>>();
                  auto [csp, hc, wc] = get_2d_const(csg_arr);
                  csg = csp;
              }
              // P-E2a/P-E2b: the isolated GPU entry now returns
              // (t_max_phys_hits, e_cond_trunc_sum, e_cond_cap_sum,
              //  cond_limit_hits, e_cool_sum, e_vac_wipe_sum, e_ring_pin_sum,
              //  e_deposit_drop_sum) — the energy books the conduction
              // rewrite (P-E2a) and the Pass-1 attenuation drop (P-E2b) must
              // be gated on. An AUTHORIZED contract change (Appendix A
              // P-E2a, extended P-E2b): the callers are
              // tests/cuda_conduction_check.py, tests/cuda_thermal_mass_check.py
              // and tests/cuda_cool_shift_check.py, which compare all seven
              // against the CPU solver's own fields.
              int64_t cnt[breach_cuda::TEMPERATURE_ENERGY_SLOTS] = {0};
              const int64_t hits = breach_cuda::temperature_step(
                  temp, hp, shift, fs, sol, vac, atm, nb, wx, wy,
                  no_face, cool_shift, cool_shift_vacuum, o2_vacuum_thresh,
                  c_v, n_floor_heat, gas_advection_rate, t_max_phys, h, w, dt,
                  nullptr, tsol, csg, cool_shift_floor, nullptr, nullptr, cnt);
              return py::make_tuple(hits, cnt[0], cnt[1], cnt[2], cnt[3],
                                    cnt[4], cnt[5], cnt[6]);
          },
          py::arg("temperature"), py::arg("heat"), py::arg("heat_inv_shift"),
          py::arg("face_shift"), py::arg("solid"), py::arg("is_vacuum"),
          py::arg("atmosphere"),
          py::arg("no_face"), py::arg("cool_shift"), py::arg("cool_shift_vacuum"),
          py::arg("o2_vacuum_thresh"), py::arg("c_v"), py::arg("n_floor_heat"),
          py::arg("gas_advection_rate"), py::arg("t_max_phys"),
          py::arg("n_bulk") = py::none(), py::arg("wind_x") = py::none(),
          py::arg("wind_y") = py::none(), py::arg("dt") = 0.0f,
          py::arg("thermal_solid") = py::none(),   // thermal-mass axis (optional)
          py::arg("cool_shift_grid") = py::none(), // cool-shift axis (optional)
          py::arg("cool_shift_floor") = 2,         // == config SHIFT_MIN
          "P6.6 isolated: run the GPU unified temperature solver in place on "
          "`temperature` (bit-identical to TemperatureSolver.step); returns "
          "(t_max_phys_hits, e_cond_trunc_sum, e_cond_cap_sum, cond_limit_hits, "
          "e_cool_sum, e_vac_wipe_sum, e_ring_pin_sum, e_deposit_drop_sum) for "
          "this call (P-E2a + P-E2b).");

    // CUDA-S2: the GPU directional raycaster gate. Casts ONE LightSource on the
    // GPU into the (pre-zeroed) output fields, replicating the CPU cast's per-ray
    // loop via Raycaster::build_ray_list (the shared /fp:strict angle math) and
    // dispatching to breach_cuda::raycaster_cast_directional. The HEAT output is
    // bit-identical to Raycaster::cast_source_directional; the render channels
    // (light_rgb/dir/smoke_glow) are deterministic-exempt. Mirrors the numpy
    // field-unpacking of the CPU cast_source_directional binding exactly. Used by
    // the S2 bit-identity gate — NOT a live game path (the live cast is the CPU
    // method above; this isolated entry never touches it).
    m.def("cuda_raycaster_cast",
          [](const Raycaster& self,
             const LightSource& src,
             py::array_t<float> light_rgb,
             py::array_t<float> light_dx,
             py::array_t<float> light_dy,
             py::array_t<float> gas,
             py::array_t<float> gas_absorption,
             py::array_t<float> gas_scatter,
             py::array_t<float> light_atten,
             py::object heat,
             py::object smoke_glow,
             py::object heat_atten) {
              auto [lrgb, h, w]  = get_3d(light_rgb);
              auto [ldx, h2, w2] = get_2d(light_dx);
              auto [ldy, h3, w3] = get_2d(light_dy);
              auto gv = gas.unchecked<3>();
              const float* gas_field = gv.data(0, 0, 0);
              const int n_gases = static_cast<int>(gv.shape(0));
              auto ga = gas_absorption.unchecked<2>();
              const float* gabs = ga.data(0, 0);
              auto gs = gas_scatter.unchecked<2>();
              const float* gsca = gs.data(0, 0);
              auto a = light_atten.unchecked<3>();
              const float* atten = a.data(0, 0, 0);
              int32_t* heat_ptr = nullptr;
              py::array_t<int32_t> heat_arr;
              if (!heat.is_none()) {
                  heat_arr = heat.cast<py::array_t<int32_t>>();
                  auto ha = heat_arr.mutable_unchecked<2>();
                  heat_ptr = ha.mutable_data(0, 0);
              }
              float* glow_ptr = nullptr;
              py::array_t<float> glow_arr;
              if (!smoke_glow.is_none()) {
                  glow_arr = smoke_glow.cast<py::array_t<float>>();
                  auto gga = glow_arr.mutable_unchecked<3>();
                  glow_ptr = gga.mutable_data(0, 0, 0);
              }
              const float* hatten = nullptr;
              py::array_t<float> heat_atten_arr;
              if (!heat_atten.is_none()) {
                  heat_atten_arr = heat_atten.cast<py::array_t<float>>();
                  auto haa = heat_atten_arr.unchecked<2>();
                  hatten = haa.data(0, 0);
              }
              // Build the ray list in the /fp:strict TU so the angle/energy/heat
              // math (and dx=cos/dy=sin) is bit-identical to the CPU march.
              std::vector<breach_cuda::RayHD> rays = self.build_ray_list(src);
              breach_cuda::raycaster_cast_directional(
                  rays.data(), static_cast<int>(rays.size()),
                  lrgb, ldx, ldy, heat_ptr, glow_ptr,
                  gas_field, gabs, gsca, n_gases,
                  atten, hatten,
                  self.smoke_absorb_scale, self.light_cull, self.heat_cull,
                  h, w);
          },
          py::arg("raycaster"), py::arg("source"), py::arg("light_rgb"),
          py::arg("light_dx"), py::arg("light_dy"),
          py::arg("gas"), py::arg("gas_absorption"), py::arg("gas_scatter"),
          py::arg("light_atten"),
          py::arg("heat") = py::none(),
          py::arg("smoke_glow") = py::none(),
          py::arg("heat_atten") = py::none(),
          "S2 isolated: cast one LightSource on the GPU into the pre-zeroed output "
          "fields; `heat` is bit-identical to Raycaster.cast_source_directional.");

    // S8c item 1 (the fire-FPS fix): cast a SEQUENCE of LightSources in ONE
    // device march. Field-for-field identical to cuda_raycaster_cast above,
    // except it concatenates build_ray_list over every source (in this
    // /fp:strict TU) and issues a SINGLE raycaster_cast_directional — one H2D of
    // the inputs + running heat plane, one march, one D2H — instead of one
    // round-trip PER source. That collapses the per-tick transfer tax that made
    // hundreds of burning tiles run at ~3 fps (2026-07-20 B5 feel-test).
    //
    // `heat` is BYTE-IDENTICAL to the per-source cuda_raycaster_cast loop: heat
    // deposits are saturating integer atomic adds of non-negative, per-ray-
    // independent deltas (heat_atomic_sat_add, cuda_raycaster.cu:41) — order-free
    // under the monotone INT32_MAX clamp, so batching every source's rays into one
    // launch yields the identical per-cell min(base + Σdeltas, MAX). No march
    // arithmetic changes; the (x*7+y*13)%ray_count phase is still set per source
    // in Python. build_ray_list is a pure function of its source (its per-source
    // mt19937 is drawn ONLY when jitter>0; cast_fire_heat sets jitter=0), so
    // concatenation cannot perturb any ray's bits. Design +3-lens critique:
    // docs/s8c_item1_fire_heat_batch_impl_2026-07-21.md.
    //
    // RENDER CHANNELS ARE NOT BYTE-STABLE HERE: light_rgb/dx/dy/smoke_glow use
    // float atomics whose interleave order differs from the per-source launches.
    // They are determinism-EXEMPT (render-only) AND cast_fire_heat discards them
    // (smoke_glow=None, rgb/dir scratch thrown away). ONLY use this entry from a
    // caller that discards the render channels.
    m.def("cuda_raycaster_cast_batch",
          [](const Raycaster& self,
             const std::vector<LightSource>& sources,
             py::array_t<float> light_rgb,
             py::array_t<float> light_dx,
             py::array_t<float> light_dy,
             py::array_t<float> gas,
             py::array_t<float> gas_absorption,
             py::array_t<float> gas_scatter,
             py::array_t<float> light_atten,
             py::object heat,
             py::object smoke_glow,
             py::object heat_atten) {
              auto [lrgb, h, w]  = get_3d(light_rgb);
              auto [ldx, h2, w2] = get_2d(light_dx);
              auto [ldy, h3, w3] = get_2d(light_dy);
              auto gv = gas.unchecked<3>();
              const float* gas_field = gv.data(0, 0, 0);
              const int n_gases = static_cast<int>(gv.shape(0));
              auto ga = gas_absorption.unchecked<2>();
              const float* gabs = ga.data(0, 0);
              auto gs = gas_scatter.unchecked<2>();
              const float* gsca = gs.data(0, 0);
              auto a = light_atten.unchecked<3>();
              const float* atten = a.data(0, 0, 0);
              int32_t* heat_ptr = nullptr;
              py::array_t<int32_t> heat_arr;
              if (!heat.is_none()) {
                  heat_arr = heat.cast<py::array_t<int32_t>>();
                  auto ha = heat_arr.mutable_unchecked<2>();
                  heat_ptr = ha.mutable_data(0, 0);
              }
              float* glow_ptr = nullptr;
              py::array_t<float> glow_arr;
              if (!smoke_glow.is_none()) {
                  glow_arr = smoke_glow.cast<py::array_t<float>>();
                  auto gga = glow_arr.mutable_unchecked<3>();
                  glow_ptr = gga.mutable_data(0, 0, 0);
              }
              const float* hatten = nullptr;
              py::array_t<float> heat_atten_arr;
              if (!heat_atten.is_none()) {
                  heat_atten_arr = heat_atten.cast<py::array_t<float>>();
                  auto haa = heat_atten_arr.unchecked<2>();
                  hatten = haa.data(0, 0);
              }
              // Concatenate every source's rays IN SOURCE ORDER (row-major from
              // Python). Order is irrelevant to `heat` (order-free atomics) but
              // keeps the discarded render scratch aligned with the per-source
              // path. Each build_ray_list call runs in this /fp:strict TU — the
              // same place the per-source entry builds its one list.
              std::vector<breach_cuda::RayHD> rays;
              for (const auto& src : sources) {
                  std::vector<breach_cuda::RayHD> r = self.build_ray_list(src);
                  rays.insert(rays.end(), r.begin(), r.end());
              }
              // n_rays==0 guard (empty source list, or all sources fully
              // angular-culled). Python also guards via burning.any(); this is
              // defense in depth. rays.size() <= INT_MAX for any playable map
              // (8 rays/tile; int overflows only near a 16384²-all-ablaze map).
              if (rays.empty()) return;
              breach_cuda::raycaster_cast_directional(
                  rays.data(), static_cast<int>(rays.size()),
                  lrgb, ldx, ldy, heat_ptr, glow_ptr,
                  gas_field, gabs, gsca, n_gases,
                  atten, hatten,
                  self.smoke_absorb_scale, self.light_cull, self.heat_cull,
                  h, w);
          },
          py::arg("raycaster"), py::arg("sources"), py::arg("light_rgb"),
          py::arg("light_dx"), py::arg("light_dy"),
          py::arg("gas"), py::arg("gas_absorption"), py::arg("gas_scatter"),
          py::arg("light_atten"),
          py::arg("heat") = py::none(),
          py::arg("smoke_glow") = py::none(),
          py::arg("heat_atten") = py::none(),
          "S8c: cast a SEQUENCE of LightSources in ONE device march (the fire-FPS "
          "fix). `heat` is bit-identical to a per-source cuda_raycaster_cast loop "
          "(order-free saturating add). Render channels differ in float-atomic "
          "order from the per-source path and are only valid for callers that "
          "discard rgb/dir/glow (cast_fire_heat).");

    // P-R1 (docs/radiation_raycaster_extinction_ruling_2026-07-31.md
    // A4.1-A4.2): the CUDA twin of cast_from_fire_plane (the Raycaster class
    // method above). Builds the SAME per-tile source list as the CPU entry
    // point (Raycaster::build_fire_ray_list — build_fire_sources, the
    // float-parity-critical shared enumerator, folded into RayHD via
    // build_ray_list) and concatenates it — IDENTICAL to
    // cuda_raycaster_cast_batch's concatenation above, except the source list
    // is built FROM THE FIRE PLANE here instead of supplied by Python. `heat`
    // is byte-identical to cast_from_fire_plane's CPU loop (same sources,
    // same order-free saturating add) — mechanical relocation, no march/law
    // change. RENDER CHANNELS ARE NOT BYTE-STABLE HERE for the same reason as
    // cuda_raycaster_cast_batch (float-atomic interleave order) — only valid
    // for callers that discard rgb/dir/glow (cast_fire_heat).
    m.def("cuda_raycaster_cast_from_fire_plane",
          [](const Raycaster& self,
             py::array_t<int32_t> fire,
             int fire_ray_count,
             double range_base, double range_per_intensity,
             double intensity_base, double intensity_per_intensity,
             std::array<float, 3> color,
             // P-F1a: the VISIBLE-LIGHT buffers are OPTIONAL (None -> the
             // short second cast is skipped entirely). The live sim path
             // discards them -- the renderer draws fire light from its own
             // blackbody selector -- and skipping saves a WHOLE extra device
             // round-trip (upload + kernel + download), which is the dominant
             // cost of the split, not the long rays.
             py::object light_rgb,
             py::object light_dx,
             py::object light_dy,
             py::array_t<float> gas,
             py::array_t<float> gas_absorption,
             py::array_t<float> gas_scatter,
             py::array_t<float> light_atten,
             py::array_t<float> heat_atten,
             py::array_t<int32_t> temperature,
             py::array_t<int32_t> heat_inv_shift,
             py::array_t<bool> thermal_solid,
             py::array_t<int32_t> rad_net,
             py::array_t<int32_t> rad_amb,
             py::array_t<int32_t> rad_flux,
             int tick,
             py::object smoke_glow,
             double jitter) {
              auto [fp, h, w] = get_2d_const(fire);
              float* lrgb = nullptr; float* ldx = nullptr; float* ldy = nullptr;
              py::array_t<float> lrgb_a, ldx_a, ldy_a;
              const bool want_light = !light_rgb.is_none();
              if (want_light) {
                  lrgb_a = light_rgb.cast<py::array_t<float>>();
                  ldx_a  = light_dx.cast<py::array_t<float>>();
                  ldy_a  = light_dy.cast<py::array_t<float>>();
                  auto lr = lrgb_a.mutable_unchecked<3>();
                  lrgb = lr.mutable_data(0, 0, 0);
                  auto lx = ldx_a.mutable_unchecked<2>();
                  ldx = lx.mutable_data(0, 0);
                  auto ly = ldy_a.mutable_unchecked<2>();
                  ldy = ly.mutable_data(0, 0);
              }
              auto gv = gas.unchecked<3>();
              const float* gas_field = gv.data(0, 0, 0);
              const int n_gases = static_cast<int>(gv.shape(0));
              auto ga = gas_absorption.unchecked<2>();
              const float* gabs = ga.data(0, 0);
              auto gs = gas_scatter.unchecked<2>();
              const float* gsca = gs.data(0, 0);
              auto a = light_atten.unchecked<3>();
              const float* atten = a.data(0, 0, 0);
              float* glow_ptr = nullptr;
              py::array_t<float> glow_arr;
              if (!smoke_glow.is_none()) {
                  glow_arr = smoke_glow.cast<py::array_t<float>>();
                  auto gga = glow_arr.mutable_unchecked<3>();
                  glow_ptr = gga.mutable_data(0, 0, 0);
              }
              auto [hatten, h5, w5] = get_2d_const(heat_atten);
              auto [tmp, h6, w6]    = get_2d_const(temperature);
              auto [his, h7, w7]    = get_2d_const(heat_inv_shift);
              auto [tsol, h8, w8]   = get_2d_const(thermal_solid);
              auto [rnet, h9, w9]   = get_2d(rad_net);
              auto [ramb, h11, w11] = get_2d(rad_amb);
              auto [rflux, h10, w10] = get_2d(rad_flux);
              // P-F1a: TWO ray lists and TWO device casts, mirroring the CPU
              // entry point's split. `rays` is the EMISSION set (RADIATION_RANGE,
              // radiation payload) for the pure-radiation fast-path kernel;
              // `light_rays` is the SHORT visible-light set (legacy range, no
              // payload) for the UNCHANGED directional kernel. `emit_mask` is
              // the once-per-tick emitter plane rule 2 keys on (v7.1 item 13),
              // built by the same shared enumerator the CPU uses.
              std::vector<uint8_t> emit_mask;
              std::vector<breach_cuda::RayHD> light_rays;
              std::vector<breach_cuda::RayHD> rays = self.build_fire_ray_list(
                  fp, h, w, fire_ray_count,
                  range_base, range_per_intensity,
                  intensity_base, intensity_per_intensity,
                  color.data(), tmp, hatten, his, tsol, tick, jitter,
                  &emit_mask, want_light ? &light_rays : nullptr);
              // n_rays==0 guard (no emitters, or all sources fully
              // angular-culled) — defense in depth, mirrors
              // cuda_raycaster_cast_batch (Python also guards before calling).
              if (rays.empty()) return (int64_t)0;
              const int64_t contact = breach_cuda::raycaster_cast_radiation(
                  rays.data(), static_cast<int>(rays.size()),
                  hatten, self.heat_cull, h, w,
                  // The E° bake (host side, from THIS raycaster's rad_scale),
                  // the three read planes, the emitter mask, and the two signed
                  // ledgers + D3's positive-only damage sensor.
                  self.emissive_table(), tmp, his, emit_mask.data(),
                  rnet, ramb, rflux);
              if (want_light && !light_rays.empty()) {
                  breach_cuda::raycaster_cast_directional(
                      light_rays.data(), static_cast<int>(light_rays.size()),
                      lrgb, ldx, ldy, /*heat=*/nullptr, glow_ptr,
                      gas_field, gabs, gsca, n_gases,
                      atten, hatten,
                      self.smoke_absorb_scale, self.light_cull, self.heat_cull,
                      h, w);
              }
              return contact;
          },
          py::arg("raycaster"), py::arg("fire"),
          py::arg("fire_ray_count"),
          py::arg("range_base"), py::arg("range_per_intensity"),
          py::arg("intensity_base"), py::arg("intensity_per_intensity"),
          py::arg("color"),
          py::arg("light_rgb"), py::arg("light_dx"), py::arg("light_dy"),
          py::arg("gas"), py::arg("gas_absorption"), py::arg("gas_scatter"),
          py::arg("light_atten"), py::arg("heat_atten"),
          py::arg("temperature"), py::arg("heat_inv_shift"),
          py::arg("thermal_solid"), py::arg("rad_net"),
          py::arg("rad_amb"), py::arg("rad_flux"), py::arg("tick"),
          py::arg("smoke_glow") = py::none(),
          py::arg("jitter") = 0.0,
          "P-F1a: CUDA twin of cast_from_fire_plane — builds the emitter list "
          "from the fire/temperature planes in C++ and marches it in ONE "
          "batched device cast. `rad_net` is bit-identical to the CPU loop "
          "(plain signed atomicAdd == the CPU's plain signed add).");

    // CUDA-S2 LIVE: the raycaster backend flag (mirrors set_temperature_backend).
    // Unlike the 6 field solvers, the live fire->heat cast is NOT dispatched in
    // PhysicsEngine::step — it runs in Python (PhysicsRunner.cast_fire_heat, the
    // per-burning-tile source loop). So this flag is read THERE: when True, the
    // runner casts each source with cuda_raycaster_cast (build_ray_list -> the GPU
    // march) instead of Raycaster.cast_source_directional; both ACCUMULATE the
    // per-source heat into the SAME gmap.heat buffer (saturating-add on the GPU
    // side too) with the identical per-tick clear, so the synced `heat` output is
    // byte-identical (the S2 gate already proved the GPU march's heat == CPU; this
    // flag wires it into the live tick to make --cuda a full 7/7). The render
    // channels (light_rgb/dir/smoke_glow) come back to the host each call for the
    // renderer and are deterministic-exempt. CPU is the live default (flag off).
    m.def("set_raycaster_backend",
          [](bool use_cuda) { breach_cuda::set_raycaster_backend_cuda(use_cuda); },
          py::arg("use_cuda"),
          "Switch PhysicsRunner.cast_fire_heat's fire->heat ray cast to the GPU "
          "(True) or CPU (False). HEAT is bit-identical; light is render-only.");
    m.def("get_raycaster_backend",
          []() { return breach_cuda::raycaster_backend_is_cuda(); },
          "True if the live fire->heat ray cast currently runs on the GPU.");

    // CUDA-S3: the GPU water solver. The backend flag switches PhysicsEngine::
    // step_water's per-substep call between the CPU and GPU pipe-model solver
    // (the live CPU fallback stays). cuda_water_step runs the 8-pass solver IN
    // PLACE on water_depth/flow_vx/flow_vy for the isolated GPU-vs-CPU bit-
    // identity gate. floor_height/atmosphere are nullable (mirroring the live
    // WaterSolver.step binding — EOS P3: atmosphere is the integer P, wave_p
    // retired); the solver's scalar dials are passed explicitly since
    // water_step is a free function.
    m.def("set_water_backend",
          [](bool use_cuda) { breach_cuda::set_water_backend_cuda(use_cuda); },
          py::arg("use_cuda"),
          "Switch PhysicsEngine's water pass to the GPU (True) or CPU (False).");
    m.def("get_water_backend",
          []() { return breach_cuda::water_backend_is_cuda(); },
          "True if the water pass currently runs on the GPU.");
    m.def("cuda_water_step",
          [](py::array_t<int32_t> water_depth, py::array_t<int32_t> flow_vx,
             py::array_t<int32_t> flow_vy, py::object floor_height,
             py::object atmosphere, py::array_t<bool> solid,
             float dt, float tilt_x, float tilt_y,
             float g, float damping, float dx, float k_p, float v_max,
             float depth_eps) {
              auto [wd, h, w]    = get_2d(water_depth);
              auto [vx, h2, w2]  = get_2d(flow_vx);
              auto [vy, h3, w3]  = get_2d(flow_vy);
              auto [sol, h4, w4] = get_2d_const(solid);
              // Nullable fields (the live WaterSolver.step binding precedent):
              // None -> nullptr, else cast to an array kept alive in this scope.
              const int32_t* fl = nullptr;
              py::array_t<int32_t> fl_arr;
              if (!floor_height.is_none()) {
                  fl_arr = floor_height.cast<py::array_t<int32_t>>();
                  auto fa = fl_arr.unchecked<2>();
                  fl = fa.data(0, 0);
              }
              // EOS P3: `atmosphere` is the derived integer pressure P (Q16.16),
              // read via the pure-integer head term; the float wave_p bridge is
              // retired (matches WaterSolver.step).
              const int32_t* atm = nullptr;
              py::array_t<int32_t> atm_arr;
              if (!atmosphere.is_none()) {
                  atm_arr = atmosphere.cast<py::array_t<int32_t>>();
                  auto aa = atm_arr.unchecked<2>();
                  atm = aa.data(0, 0);
              }
              breach_cuda::water_step(wd, vx, vy, fl, atm, sol, h, w, dt,
                                      tilt_x, tilt_y, g, damping, dx, k_p,
                                      v_max, depth_eps);
          },
          py::arg("water_depth"), py::arg("flow_vx"), py::arg("flow_vy"),
          py::arg("floor_height") = py::none(),
          py::arg("atmosphere")   = py::none(),
          py::arg("solid"), py::arg("dt"), py::arg("tilt_x"), py::arg("tilt_y"),
          py::arg("g"), py::arg("damping"), py::arg("dx"), py::arg("k_p"),
          py::arg("v_max"), py::arg("depth_eps"),
          "S3 isolated: run the GPU water solver in place on water_depth/flow_vx/"
          "flow_vy (bit-identical to WaterSolver.step).");

    // CUDA-S4a: the GPU smoke solver. The backend flag switches PhysicsEngine::
    // run_substeps's per-gas smoke transport between the CPU SmokeDynamics::step
    // and the GPU smoke_step (the live CPU fallback stays). cuda_smoke_step runs
    // the 4-pass solver IN PLACE on `smoke` (one gas plane) for the isolated
    // GPU-vs-CPU bit-identity gate. The solver's scalar dials (d_smoke /
    // wind_diffusion_scale / advection_rate) are passed explicitly since
    // smoke_step is a free function — mirroring the live SmokeDynamics.step
    // binding's array args plus those scalars.
    // (The S4b sink_hop half of this banner went with the pass — A9, 2026-08-04.)
    m.def("set_smoke_backend",
          [](bool use_cuda) { breach_cuda::set_smoke_backend_cuda(use_cuda); },
          py::arg("use_cuda"),
          "Switch PhysicsEngine's smoke pass to the GPU (True) "
          "or CPU (False).");
    m.def("get_smoke_backend",
          []() { return breach_cuda::smoke_backend_is_cuda(); },
          "True if the smoke pass currently runs on the GPU.");
    m.def("cuda_smoke_step",
          [](py::array_t<int32_t> smoke,        // Q16.16 int32 (one gas plane)
             py::array_t<int32_t> wind_x,       // Q16.16 int32
             py::array_t<int32_t> wind_y,       // Q16.16 int32
             py::array_t<bool>  obstacles,
             py::array_t<bool>  is_wall,
             py::array_t<bool>  is_vacuum,
             py::array_t<float> permeability,
             float dt, float d_smoke,
             float wind_diffusion_scale, float advection_rate) {
              auto [sm, h, w]    = get_2d(smoke);
              auto [wx, h2, w2]  = get_2d_const(wind_x);
              auto [wy, h3, w3]  = get_2d_const(wind_y);
              auto [obs, h4, w4] = get_2d_const(obstacles);
              auto [wl, h5, w5]  = get_2d_const(is_wall);
              auto [vac, h6, w6] = get_2d_const(is_vacuum);
              auto [perm, h7, w7] = get_2d_const(permeability);
              breach_cuda::smoke_step(sm, wx, wy, obs, wl, vac, perm, h, w, dt,
                                      d_smoke, wind_diffusion_scale, advection_rate);
          },
          py::arg("smoke"), py::arg("wind_x"), py::arg("wind_y"),
          py::arg("obstacles"), py::arg("is_wall"), py::arg("is_vacuum"),
          py::arg("permeability"), py::arg("dt"), py::arg("d_smoke"),
          py::arg("wind_diffusion_scale"), py::arg("advection_rate"),
          "S4a isolated: run the GPU smoke solver in place on one gas plane "
          "(bit-identical to SmokeDynamics.step).");
    // (cuda_smoke_sink_hop DELETED — audit Patch A / A9, 2026-08-04. It
    // exposed breach_cuda::smoke_sink_hop, an orphaned GPU port whose CPU
    // twin SmokeDynamics::sink_hop went with EOS refactor P3. No Python
    // caller existed; the kernel is deleted in cuda_smoke.cu.)

    // CUDA-S5 (set_wave_backend / get_wave_backend / cuda_wave_substep) RETIRED
    // in EOS P6.0: the wave_substep solver it mirrored was deleted in P3 (the
    // compressible EOS solve replaced wave+diffuse), so the kernel had no live
    // dispatch and no non-stale caller (docs/eos_p6_gpu_alignment_review.md §1.11).

    // CUDA-S6: the GPU fire solver. The backend flag switches PhysicsEngine::
    // step_tail's fire pass between the CPU FireSimulation::step and the GPU
    // fire_step (the live CPU fallback stays). cuda_fire_step runs ONE GPU fire
    // step IN PLACE on fire/atmosphere/smoke/wall_hp (the 4 mutated fields) and
    // RETURNS the destroyed-walls list as a py::list of (y,x) tuples — mirroring
    // the live FireSimulation.step binding. The solver's scalar FireParams dials
    // are passed explicitly since fire_step is a free function.
    m.def("set_fire_backend",
          [](bool use_cuda) { breach_cuda::set_fire_backend_cuda(use_cuda); },
          py::arg("use_cuda"),
          "Switch PhysicsEngine's fire pass (FireSimulation.step) to the GPU "
          "(True) or CPU (False).");
    m.def("get_fire_backend",
          []() { return breach_cuda::fire_backend_is_cuda(); },
          "True if the fire pass currently runs on the GPU.");
    m.def("cuda_fire_step",
          [](py::array_t<int32_t> fire,         // Q16.16 int32 (intensity)
             py::array_t<int32_t> atmosphere,   // Q16.16 int32 (read-only, vestigial)
             py::array_t<int32_t> n_o2,         // Q16.16 int32 (read-only, O2 gate numerator)
             py::array_t<int32_t> n_total,      // Q16.16 int32 (read-only, O2 gate denominator)
             py::array_t<int32_t> smoke,        // Q16.16 int32 (emission scatter)
             py::array_t<int32_t> wall_hp,      // Q16.16 int32 (burn-through)
             py::array_t<int32_t> temperature,  // Q16.16 int32 (in/out type; READ
                                                 // only as of P-R2 — the plume->T
                                                 // shim write is deleted)
             py::array_t<int32_t> wind_x,       // Q16.16 int32 (read-only)
             py::array_t<int32_t> wind_y,       // Q16.16 int32 (read-only)
             py::array_t<bool>  is_wall,
             py::array_t<bool>  is_vacuum,
             py::array_t<bool>  flammable,
             float dt, float k_grow, float k_die, float fire_T_ext,
             float fire_T_span, float fuel_ref, float o2_frac_ext, float o2_frac_full,
             float I_min, float k_wind_fan, float k_wind_strip,
             float wall_damage, float temp_scale, float I_cap_per_avail,
             py::object fuel_recip,                  // FUEL-FRACTION AXIS
             py::object fire_T_ext_plane) -> py::list {  // PER-MATERIAL T_ext
              auto [f, h, w]     = get_2d(fire);
              auto [atm, h2, w2] = get_2d_const(atmosphere);
              auto [o2, h2b, w2b] = get_2d_const(n_o2);
              auto [nt, h2c, w2c] = get_2d_const(n_total);
              auto [sm, h3, w3]  = get_2d(smoke);
              auto [whp, h4, w4] = get_2d(wall_hp);
              auto [temp, h5, w5] = get_2d(temperature);   // in/out type; READ only
                                                            // as of P-R2
              auto [wx, h6, w6]  = get_2d_const(wind_x);
              auto [wy, h7, w7]  = get_2d_const(wind_y);
              auto [wl, h8, w8]  = get_2d_const(is_wall);
              auto [vac, h9, w9] = get_2d_const(is_vacuum);
              auto [fl, h10, w10] = get_2d_const(flammable);
              // FUEL-FRACTION AXIS: OPTIONAL per-tile 1/hp plane (int64).
              // None -> nullptr -> the `fuel_ref` scalar fallback, which is the
              // pre-axis law bit-for-bit. Same nullable-plane idiom the
              // cool-shift axis uses on the temperature kernel.
              const int64_t* fr = nullptr;
              py::array_t<int64_t> fr_arr;
              if (!fuel_recip.is_none()) {
                  fr_arr = fuel_recip.cast<py::array_t<int64_t>>();
                  auto fv = fr_arr.unchecked<2>();
                  fr = fv.data(0, 0);
              }
              // PER-MATERIAL T_ext (P-R3): the same nullable-plane idiom.
              const int32_t* tep = nullptr;
              py::array_t<int32_t> tep_arr;
              if (!fire_T_ext_plane.is_none()) {
                  tep_arr = fire_T_ext_plane.cast<py::array_t<int32_t>>();
                  auto tv = tep_arr.unchecked<2>();
                  tep = tv.data(0, 0);
              }
              auto destroyed = breach_cuda::fire_step(
                  f, atm, o2, nt, sm, whp, temp, wx, wy, wl, vac, fl, h, w, dt,
                  k_grow, k_die, fire_T_ext, fire_T_span, fuel_ref, o2_frac_ext,
                  o2_frac_full, I_min, k_wind_fan, k_wind_strip,
                  wall_damage, temp_scale, I_cap_per_avail,
                  fr, tep);
              py::list result;
              for (const auto& [dy, dx] : destroyed) {
                  result.append(py::make_tuple(dy, dx));
              }
              return result;
          },
          py::arg("fire"), py::arg("atmosphere"), py::arg("n_o2"), py::arg("n_total"),
          py::arg("smoke"),
          py::arg("wall_hp"), py::arg("temperature"), py::arg("wind_x"),
          py::arg("wind_y"), py::arg("is_wall"), py::arg("is_vacuum"),
          py::arg("flammable"), py::arg("dt"), py::arg("k_grow"), py::arg("k_die"),
          py::arg("fire_T_ext"), py::arg("fire_T_span"), py::arg("fuel_ref"),
          py::arg("o2_frac_ext"), py::arg("o2_frac_full"), py::arg("I_min"),
          py::arg("k_wind_fan"), py::arg("k_wind_strip"),
          py::arg("wall_damage"), py::arg("temp_scale"),
          // CAPACITY LAW (P-R3, ruling A3): `c`. Defaulted to the FireParams
          // default so every existing direct caller keeps a valid law.
          py::arg("I_cap_per_avail") = 2.53f,
          py::arg("fuel_recip") = py::none(),        // fuel-fraction axis (optional)
          py::arg("fire_T_ext_plane") = py::none(),  // per-material T_ext (optional)
          "P6.8 isolated: run ONE GPU fire step (re-derived — continuous-O2 "
          "mole-fraction gate) in place on fire/smoke/wall_hp (bit-identical to "
          "FireSimulation.step) and return the destroyed-walls list of (y,x) "
          "tuples. temperature is still a parameter but is READ ONLY as of "
          "P-R2 (the plume->T shim write is deleted).");

    // EOS P6.9b: the GPU combustion solver (the two-gather reformulation —
    // docs/eos_p6_9_combustion_design.md). The backend flag switches
    // PhysicsRunner's combustion pass between the CPU CombustionSolver::step and
    // the GPU combustion_step (live CPU fallback stays; flag-off == exact prior
    // CPU call). cuda_combustion_step runs ONE GPU combustion step IN PLACE on
    // the three mutated gas planes + temperature + wall_hp (bit-identical to
    // CombustionSolver::step) and returns the per-call rail counts
    // (heat_floor_hits, t_max_phys_hits, e_deposit_drop_sum — the last is
    // P-E2b's energy-sum twin of heat_floor_hits) as a tuple so the gate can
    // compare them against the CPU solver's member-counter deltas. The scalar
    // config dials are passed explicitly since combustion_step is a free
    // function.
    m.def("set_combustion_backend",
          [](bool use_cuda) { breach_cuda::set_combustion_backend_cuda(use_cuda); },
          py::arg("use_cuda"),
          "Switch PhysicsRunner's combustion pass (CombustionSolver.step) to the "
          "GPU (True) or CPU (False).");
    m.def("get_combustion_backend",
          []() { return breach_cuda::combustion_backend_is_cuda(); },
          "True if the combustion pass currently runs on the GPU.");
    m.def("cuda_combustion_step",
          [](py::array_t<int32_t> gas,             // (n_gases,h,w) Q16.16, MUTATED
             int o2_idx, int inert_n2_idx, int black_smoke_idx,
             py::array_t<int32_t> temperature,     // Q16.16, MUTATED (heat deposit)
             py::array_t<int32_t> wall_hp,         // Q16.16, MUTATED (fuel payment)
             py::array_t<int32_t> fire,            // Q16.16, read-only (continuous-O2 law)
             py::array_t<bool> flammable,
             py::array_t<bool> solid,
             py::array_t<bool> is_vacuum,
             py::array_t<int32_t> ignition_temp_q16,  // Q16.16, read-only
             float dt, float c_v, float n_floor_heat,
             float burn_rate, float o2_thresh_burn, float H_fuel,
             float soot_yield, float fuel_per_o2, float o2_frac_ext,
             float o2_frac_full, float T_MAX_PHYS,
             // THERMAL-MASS AXIS, P-EOS (ruling §2 site 3): the OBJECT-deposit
             // branch's inputs, both OPTIONAL (None -> the gas path == pre-patch).
             py::object thermal_solid,
             py::object heat_inv_shift,
             // P-R4: the fuel-bed deposit's plane + its split constant.
             py::object heat, float H_BED_M, int H_BED_SHIFT,
             // D1: the (max_claimants,h,w) error-feedback demand accumulator
             // (IN/OUT).
             py::object dem_acc,
             // P-O2b (design v5.2 "F-O2b"): the EXTENDED OXYGEN DRAW — radius,
             // the permeability plane the path weight rides, and the declared
             // dem_acc slot depth. draw_r == 1 is the shipped law bit for bit.
             int draw_r, py::object dyn_permeability, int max_claimants) -> py::tuple {
              auto gv = gas.mutable_unchecked<3>();
              int32_t* gas_ptr = gv.mutable_data(0, 0, 0);
              const int n_gases = static_cast<int>(gv.shape(0));
              const int h = static_cast<int>(gv.shape(1));
              const int w = static_cast<int>(gv.shape(2));
              auto [temp, h2, w2] = get_2d(temperature);
              auto [whp, h3, w3]  = get_2d(wall_hp);
              auto [f, h4b, w4b]  = get_2d_const(fire);
              auto [fl, h4, w4]   = get_2d_const(flammable);
              auto [sol, h5, w5]  = get_2d_const(solid);
              auto [vac, h6, w6]  = get_2d_const(is_vacuum);
              auto [ign, h7, w7]  = get_2d_const(ignition_temp_q16);
              const bool* tsol = nullptr;
              py::array_t<bool> tsol_arr;
              if (!thermal_solid.is_none()) {
                  tsol_arr = thermal_solid.cast<py::array_t<bool>>();
                  auto ta = tsol_arr.unchecked<2>();
                  tsol = ta.data(0, 0);
              }
              const int32_t* hshift = nullptr;
              py::array_t<int32_t> hshift_arr;
              if (!heat_inv_shift.is_none()) {
                  hshift_arr = heat_inv_shift.cast<py::array_t<int32_t>>();
                  auto ha = hshift_arr.unchecked<2>();
                  hshift = ha.data(0, 0);
              }
              int32_t* heat_ptr = nullptr;
              py::array_t<int32_t> heat_arr;
              if (!heat.is_none()) {
                  heat_arr = heat.cast<py::array_t<int32_t>>();
                  auto hh = heat_arr.mutable_unchecked<2>();
                  heat_ptr = hh.mutable_data(0, 0);
              }
              int32_t* dacc_ptr = nullptr;
              py::array_t<int32_t> dacc_arr;
              if (!dem_acc.is_none()) {
                  dacc_arr = dem_acc.cast<py::array_t<int32_t>>();
                  auto da = dacc_arr.mutable_unchecked<3>();   // (max_claimants, h, w)
                  dacc_ptr = da.mutable_data(0, 0, 0);
              }
              const float* perm_ptr = nullptr;
              py::array_t<float> perm_arr;
              if (!dyn_permeability.is_none()) {
                  perm_arr = dyn_permeability.cast<py::array_t<float>>();
                  auto pa = perm_arr.unchecked<2>();
                  perm_ptr = pa.data(0, 0);
              }
              int64_t heat_floor_hits = 0, t_max_phys_hits = 0;
              int64_t e_deposit_drop_sum = 0;   // P-E2b
              breach_cuda::combustion_step(
                  gas_ptr, n_gases, o2_idx, inert_n2_idx, black_smoke_idx,
                  temp, whp, f, fl, sol, vac, ign, h, w, dt, c_v, n_floor_heat,
                  burn_rate, o2_thresh_burn, H_fuel, soot_yield, fuel_per_o2,
                  o2_frac_ext, o2_frac_full,
                  T_MAX_PHYS, &heat_floor_hits, &t_max_phys_hits,
                  &e_deposit_drop_sum,
                  tsol, hshift, heat_ptr, H_BED_M, H_BED_SHIFT, dacc_ptr,
                  draw_r, perm_ptr, max_claimants);
              return py::make_tuple(heat_floor_hits, t_max_phys_hits,
                                    e_deposit_drop_sum);
          },
          py::arg("gas"), py::arg("o2_idx"), py::arg("inert_n2_idx"),
          py::arg("black_smoke_idx"), py::arg("temperature"), py::arg("wall_hp"),
          py::arg("fire"),
          py::arg("flammable"), py::arg("solid"), py::arg("is_vacuum"),
          py::arg("ignition_temp_q16"), py::arg("dt"), py::arg("c_v"),
          py::arg("n_floor_heat"), py::arg("burn_rate"), py::arg("o2_thresh_burn"),
          py::arg("H_fuel"), py::arg("soot_yield"), py::arg("fuel_per_o2"),
          py::arg("o2_frac_ext"), py::arg("o2_frac_full"),
          py::arg("T_MAX_PHYS"),
          py::arg("thermal_solid") = py::none(),
          py::arg("heat_inv_shift") = py::none(),
          py::arg("heat") = py::none(),
          py::arg("H_BED_M") = 0.0f,
          py::arg("H_BED_SHIFT") = 0,
          py::arg("dem_acc") = py::none(),
          py::arg("draw_r") = 1,
          py::arg("dyn_permeability") = py::none(),
          py::arg("max_claimants") = 4,
          "P6.9b isolated: run ONE GPU combustion step (the two-gather "
          "reformulation, continuous-O2 proportional demand) in place on the "
          "three gas planes + temperature + wall_hp (bit-identical to "
          "CombustionSolver.step) and return the (heat_floor_hits, "
          "t_max_phys_hits, e_deposit_drop_sum) per-call rail counts.");

    // CUDA-S7 (set_atmos_backend / get_atmos_backend / cuda_diffuse_solve)
    // RETIRED in EOS P6.0: the diffuse_solve solver it mirrored was deleted in
    // P3 (the compressible EOS solve replaced wave+diffuse), so the kernel had
    // no live dispatch and no non-stale caller
    // (docs/eos_p6_gpu_alignment_review.md §1.11).

    // EOS P6.2: the GPU fused 3-field SL advection (velocity self-advection +
    // gas-T advection — EOSSolver::step's substep-loop steps 1a/1b/1f).
    // cuda_eos_sl_advect runs the FULL substep-loop advection chain for one
    // tick IN PLACE on wind_x/wind_y/temperature and returns the chained FNV
    // digest (== EOSSolver.digest_advect for the same inputs/schedule). Used
    // by the P6.2 bit-identity gate — NOT a live game path; the engine
    // dispatch flip is P6.5 (the backend flag below exists for that wiring).
    m.def("set_sl_advection_backend",
          [](bool use_cuda) { breach_cuda::set_sl_advection_backend_cuda(use_cuda); },
          py::arg("use_cuda"),
          "Switch the EOS SL-advection pass to the GPU (True) or CPU (False). "
          "No dispatch site consumes this until P6.5 wires eos.step's GPU path.");
    m.def("get_sl_advection_backend",
          []() { return breach_cuda::sl_advection_backend_is_cuda(); },
          "True if the EOS SL-advection pass is flagged for the GPU.");
    m.def("cuda_eos_sl_advect",
          [](py::array_t<int32_t> wind_x, py::array_t<int32_t> wind_y,
             py::array_t<int32_t> temperature,
             py::array_t<bool> solid, py::array_t<bool> is_vacuum,
             py::array_t<float> dyn_permeability,
             float dt, int n_sub,
             py::object thermal_solid) -> uint64_t {   // THERMAL-MASS AXIS
              auto [wx, h, w]    = get_2d(wind_x);
              auto [wy, h2, w2]  = get_2d(wind_y);
              auto [t, h3, w3]   = get_2d(temperature);
              auto [sol, h4, w4] = get_2d_const(solid);
              auto [vac, h5, w5] = get_2d_const(is_vacuum);
              auto [pm, h6, w6]  = get_2d_const(dyn_permeability);
              const bool* tsol = nullptr;
              py::array_t<bool> tsol_arr;
              if (!thermal_solid.is_none()) {
                  tsol_arr = thermal_solid.cast<py::array_t<bool>>();
                  auto ta = tsol_arr.unchecked<2>();
                  tsol = ta.data(0, 0);
              }
              return breach_cuda::eos_sl_advect(wx, wy, t, sol, vac, pm,
                                                h, w, dt, n_sub, tsol);
          },
          py::arg("wind_x"), py::arg("wind_y"), py::arg("temperature"),
          py::arg("solid"), py::arg("is_vacuum"), py::arg("dyn_permeability"),
          py::arg("dt"), py::arg("n_sub"),
          py::arg("thermal_solid") = py::none(),
          "P6.2 isolated: run the GPU fused SL-advection substep chain in place "
          "on wind_x/wind_y/temperature (bit-identical to eos_sl_advect_ref) and "
          "return the chained FNV digest (== EOSSolver.digest_advect).");

    // EOS P6.1: the GPU bulk donor-cell flux (cuda_bulk_transport.cu — the
    // water K3-K8 precompute-then-gather pattern with per-face coefficients,
    // bit-identical to bulk_flux_transport_cached). KERNEL-GATE ONLY for now:
    // the backend flag exists so P6.5 can wire the eos_solver dispatch, but
    // nothing dispatches on it yet (docs/eos_p6_gpu_alignment_review.md §4,
    // P6.1 row). cuda_bulk_flux_transport mirrors the CPU bulk_flux_transport
    // binding (below, unconditional) argument-for-argument (legacy signature — the host-side
    // coefficient hoist runs verbatim inside the .cu) so the digest gate can
    // drive both entries with identical inputs.
    m.def("set_bulk_flux_backend",
          [](bool use_cuda) { breach_cuda::set_bulk_flux_backend_cuda(use_cuda); },
          py::arg("use_cuda"),
          "Switch the bulk donor-cell flux to the GPU (True) or CPU (False). "
          "P6.1: flag only — the engine dispatch lands in P6.5.");
    m.def("get_bulk_flux_backend",
          []() { return breach_cuda::bulk_flux_backend_is_cuda(); },
          "True if the bulk donor-cell flux backend is set to GPU (P6.1: flag "
          "only until the P6.5 engine dispatch).");
    m.def("cuda_bulk_flux_transport",
          [](py::array_t<int32_t> gas, py::array_t<bool> gas_conservative,
             py::array_t<int32_t> wind_x, py::array_t<int32_t> wind_y,
             py::array_t<bool> solid, py::array_t<bool> is_vacuum,
             py::array_t<float> dyn_permeability, float dt) {
              auto gv = gas.mutable_unchecked<3>();
              int32_t* gas_ptr = gv.mutable_data(0, 0, 0);
              const int n_gases = static_cast<int>(gv.shape(0));
              const int h = static_cast<int>(gv.shape(1));
              const int w = static_cast<int>(gv.shape(2));
              auto gc = gas_conservative.unchecked<1>();
              const bool* gcons = gc.data(0);
              auto [wx, h2, w2] = get_2d_const(wind_x);
              auto [wy, h3, w3] = get_2d_const(wind_y);
              auto [sol, h4, w4] = get_2d_const(solid);
              auto [vac, h5, w5] = get_2d_const(is_vacuum);
              auto [perm, h6, w6] = get_2d_const(dyn_permeability);
              breach_cuda::bulk_flux_transport(
                  gas_ptr, gcons, n_gases,
                  wx, wy, sol, vac, perm,
                  h, w, dt);
          }, py::arg("gas"), py::arg("gas_conservative"),
             py::arg("wind_x"), py::arg("wind_y"),
             py::arg("solid"), py::arg("is_vacuum"),
             py::arg("dyn_permeability"), py::arg("dt"),
          "P6.1 isolated: GPU donor-cell conservative flux transport of every "
          "`gas_conservative`-flagged plane, once, on the given wind field "
          "(bit-identical to bulk_flux_transport).");

    // EOS P6.4: the GPU momentum kick + compression work (EOSSolver::step's
    // post-solve tail, steps 4 + 4c). cuda_eos_kick_compression runs both
    // passes IN PLACE on wind_x/wind_y/temperature and returns the digests +
    // per-call rail counters (== eos_kick_compression_ref for the same
    // inputs). Used by the P6.4 bit-identity gate — NOT a live game path; the
    // engine dispatch flip is P6.5 (the backend flag below exists for that
    // wiring).
    m.def("set_kick_compression_backend",
          [](bool use_cuda) { breach_cuda::set_kick_compression_backend_cuda(use_cuda); },
          py::arg("use_cuda"),
          "Switch the EOS kick+compression tail to the GPU (True) or CPU "
          "(False). No dispatch site consumes this until P6.5 wires eos.step's "
          "GPU path.");
    m.def("get_kick_compression_backend",
          []() { return breach_cuda::kick_compression_backend_is_cuda(); },
          "True if the EOS kick+compression tail is flagged for the GPU.");
    m.def("cuda_eos_kick_compression",
          [](py::array_t<int32_t> wind_x, py::array_t<int32_t> wind_y,
             py::array_t<int32_t> temperature, py::array_t<int32_t> p_new,
             py::array_t<int32_t> gas, py::array_t<bool> gas_conservative,
             py::array_t<bool> solid, py::array_t<bool> is_vacuum,
             py::array_t<float> dyn_wave_absorb,
             float dt, int32_t c_local_q,
             float c_max, float dx, float adiabatic_index,
             float absorb_strength, float n_floor_solver, float t_min,
             float t_work_clamp, float t_max_phys, float u_max,
             // trace_mass_scale param RETIRED (P-T0, design §2.6)
             // P-E3 (design §2.8): interior drag + heat counterparty.
             float k_drag, float k_drag_heat_frac, float c_v,
             py::object thermal_solid) -> py::tuple {   // THERMAL-MASS AXIS
              auto [wx, h, w]    = get_2d(wind_x);
              auto [wy, h2, w2]  = get_2d(wind_y);
              auto [t, h3, w3]   = get_2d(temperature);
              auto [pn, h4, w4]  = get_2d_const(p_new);
              auto gv = gas.unchecked<3>();
              const int32_t* gas_ptr = gv.data(0, 0, 0);
              const int n_gases = static_cast<int>(gv.shape(0));
              auto gc = gas_conservative.unchecked<1>();
              const bool* gcons = gc.data(0);
              auto [sol, h5, w5] = get_2d_const(solid);
              auto [vac, h6, w6] = get_2d_const(is_vacuum);
              auto [ab, h7, w7]  = get_2d_const(dyn_wave_absorb);
              const bool* tsol = nullptr;
              py::array_t<bool> tsol_arr;
              if (!thermal_solid.is_none()) {
                  tsol_arr = thermal_solid.cast<py::array_t<bool>>();
                  auto ta = tsol_arr.unchecked<2>();
                  tsol = ta.data(0, 0);
              }
              uint64_t dig_vel = 0, dig_comp = 0;
              int64_t cnts[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
              breach_cuda::eos_kick_compression(
                  wx, wy, t, pn, gas_ptr, gcons, n_gases, sol, vac, ab,
                  h, w, dt, c_local_q,
                  c_max, dx, adiabatic_index, absorb_strength,
                  n_floor_solver, t_min, t_work_clamp, t_max_phys, u_max,
                  k_drag, k_drag_heat_frac, c_v,
                  &dig_vel, &dig_comp, cnts,   // trace_mass_scale arg RETIRED
                  nullptr, nullptr, tsol);
              return py::make_tuple(dig_vel, dig_comp, cnts[0], cnts[1],
                                    cnts[2], cnts[3], cnts[4], cnts[5],
                                    cnts[6], cnts[7], cnts[8]);
          },
          py::arg("wind_x"), py::arg("wind_y"), py::arg("temperature"),
          py::arg("p_new"), py::arg("gas"), py::arg("gas_conservative"),
          py::arg("solid"), py::arg("is_vacuum"), py::arg("dyn_wave_absorb"),
          py::arg("dt"), py::arg("c_local_q"),
          py::arg("c_max"), py::arg("dx"), py::arg("adiabatic_index"),
          py::arg("absorb_strength"), py::arg("n_floor_solver"),
          py::arg("t_min"), py::arg("t_work_clamp"), py::arg("t_max_phys"),
          py::arg("u_max"),
          py::arg("k_drag") = 0.0f, py::arg("k_drag_heat_frac") = 1.0f,
          py::arg("c_v") = 1.0f,
          py::arg("thermal_solid") = py::none(),
          "P6.4 isolated: run the GPU kick + compression-work tail in place on "
          "wind_x/wind_y/temperature; returns (digest_velocity, "
          "digest_compression, u_clamp_hits, u_max_hits, work_clamp_hits, "
          "energy_floor_hits, t_max_phys_hits, ke_drag_removed, "
          "e_drag_deposit, e_drag_drop_sum, e_drag_rail_clipped) for this "
          "call — bit-identical to eos_kick_compression_ref.");

    // EOS P6.3: the GPU multigrid Helmholtz pressure solve (cuda_mg_solve.cu
    // — per-color RB-GS launches on fine levels, gather-form restriction/
    // prolongation, the FUSED COARSE-TAIL kernel for the ≤1024-cell levels;
    // bit-identical to EOSSolver::mg_run_solve_cpu). KERNEL-GATE ONLY for
    // now: the backend flag exists so P6.5 can wire the eos_solver dispatch,
    // but nothing dispatches on it yet (docs/eos_p6_gpu_alignment_review.md
    // §4, P6.3 row). cuda_eos_mg_solve mirrors eos_mg_solve_ref (below,
    // unconditional) argument-for-argument: the per-tick hierarchy is built
    // HOST-side through the SAME EOSSolver::mg_build_levels the CPU path
    // calls (review §2.7 — build placement is digest-neutral pre-residency;
    // the build is per-tick because level-0 m derives from p* and gE/gS fold
    // the per-tick 1/N̂), then the ENTIRE iteration runs on the device.
    m.def("set_mg_solve_backend",
          [](bool use_cuda) { breach_cuda::set_mg_solve_backend_cuda(use_cuda); },
          py::arg("use_cuda"),
          "Switch the EOS multigrid pressure solve to the GPU (True) or CPU "
          "(False). No dispatch site consumes this until P6.5 wires eos.step's "
          "GPU path.");
    m.def("get_mg_solve_backend",
          []() { return breach_cuda::mg_solve_backend_is_cuda(); },
          "True if the EOS multigrid pressure solve is flagged for the GPU.");
    m.def("cuda_eos_mg_solve",
          [](const EOSSolver& solver,
             py::array_t<int32_t> pstar, py::array_t<int32_t> div_u,
             py::array_t<int32_t> n_total, py::array_t<int32_t> p_prev,
             py::array_t<bool> solid, py::array_t<bool> is_vacuum,
             py::array_t<float> dyn_permeability,
             float dt, py::array_t<int32_t> p_out) -> py::tuple {
              auto [ps, h, w]    = get_2d_const(pstar);
              auto [dv, h2, w2]  = get_2d_const(div_u);
              auto [nt, h3, w3]  = get_2d_const(n_total);
              auto [pp, h4, w4]  = get_2d_const(p_prev);
              auto [sol, h5, w5] = get_2d_const(solid);
              auto [vac, h6, w6] = get_2d_const(is_vacuum);
              auto [pm, h7, w7]  = get_2d_const(dyn_permeability);
              auto [po, h8, w8]  = get_2d(p_out);
              const int n_levels = solver.mg_build_levels(
                  ps, dv, nt, pp, sol, vac, pm, h, w, dt);
              if (n_levels <= 0)
                  return py::make_tuple((uint64_t)0, 0, 0);
              const auto& L = solver.mg_levels();
              std::vector<breach_cuda::MGLevelHostView> views(n_levels);
              for (int lv = 0; lv < n_levels; ++lv) {
                  views[lv].h = L[lv].h;
                  views[lv].w = L[lv].w;
                  views[lv].excl  = L[lv].excl.data();
                  views[lv].m     = L[lv].m.data();
                  views[lv].gE    = L[lv].gE.data();
                  views[lv].gS    = L[lv].gS.data();
                  views[lv].recip = L[lv].recip.data();
                  views[lv].b     = L[lv].b.data();
                  views[lv].P     = L[lv].P.data();
              }
              int la = 0, ln = 0;
              const uint64_t dig = breach_cuda::eos_mg_vcycle(
                  views.data(), n_levels, solver.use_multigrid,
                  solver.mg_cycles, solver.mg_nu1, solver.mg_nu2,
                  solver.mg_coarsest_sweeps, solver.S, po, &la, &ln);
              return py::make_tuple(dig, la, ln);
          },
          py::arg("solver"), py::arg("pstar"), py::arg("div_u"),
          py::arg("n_total"), py::arg("p_prev"),
          py::arg("solid"), py::arg("is_vacuum"), py::arg("dyn_permeability"),
          py::arg("dt"), py::arg("p_out"),
          "P6.3 isolated: run the multigrid pressure solve with the hierarchy "
          "built host-side (the SAME mg_build_levels the CPU calls) and the "
          "ENTIRE V-cycle iteration on the GPU; writes the solved P into "
          "p_out and returns (digest, launches_actual, launches_naive) — the "
          "digest is bit-identical to eos_mg_solve_ref / digest_helmholtz.");

    // EOS P6.5: the chained full-eos.step engine dispatch (cuda_eos_step.cu).
    // PhysicsEngine::run_substeps now routes eos.step to the GPU orchestration
    // when EVERY one of the four EOS kernel-surface flags is on (sl_advection
    // && bulk_flux && mg_solve && kick_compression — the review is silent on a
    // master flag, so they are ANDed; there is deliberately NO separate
    // setter). get_eos_step_backend exposes the dispatch predicate;
    // eos_step_cuda_calls counts the ticks that actually ran the GPU chain,
    // so the P6.5 gate can prove the dispatch FIRED (a silently-CPU "GPU run"
    // would make a bit-identity gate vacuous).
    m.def("get_eos_step_backend",
          []() { return breach_cuda::eos_step_backend_is_cuda(); },
          "True iff run_substeps will dispatch eos.step to the GPU chain "
          "(all four EOS kernel-surface backend flags are on).");
    m.def("eos_step_cuda_calls",
          []() { return breach_cuda::eos_step_cuda_calls(); },
          "How many engine ticks have run the chained GPU eos.step path "
          "(P6.5 dispatch-fired telemetry).");
#else
    m.attr("HAS_CUDA") = false;
#endif

    // S1: the water core is now int32 Q16.16 (metres / m/s, scale 2^16). Python
    // (gamemap fields, tests, the feel-regression harness) reads this flag to
    // allocate water_depth/flow_vx/flow_vy/floor_height/before as int32 and to
    // dequantize for the renderer / float bridges.
    m.attr("WATER_FIXEDPOINT") = true;
    m.attr("WATER_FP_SHIFT") = 16;
    m.attr("WATER_FP_ONE") = 65536;

    // Bedrock cliff-patch: expose the integer smoke-CFL substep-count helper so a
    // unit test (tests/test_bedrock_cliff_counts.py) can verify the SHIPPED C++
    // (the real 128-bit / _umul128 path) against the Python reference mirror — not
    // just a re-implementation. Args are the quantized Q16.16 cliff constants +
    // the Q.32 integer max|wind|^2 (exactly what run_substeps feeds the engine).
    m.def("smoke_cliff_count",
          [](int32_t c4st_q, int32_t dsmoke_q, int32_t wds_q, int64_t mws_q32) {
              return fixedpoint::smoke_cliff_count(c4st_q, dsmoke_q, wds_q, mws_q32);
          },
          py::arg("c4st_q"), py::arg("dsmoke_q"), py::arg("wds_q"),
          py::arg("mws_q32"),
          "Bedrock: integer smoke-CFL substep count "
          "n=ceil(4*sim_time*d_smoke_max*(1+wds*max_wind_sq)) from quantized inputs.");

    // EOS refactor P1 (docs/eos_refactor_design.md §2.2): expose
    // bulk_flux_transport directly (not just via PhysicsEngine::run_substeps)
    // so tests can drive it with a hand-crafted wind field / permeability
    // scene, mirroring the WaterSolver stress-conservation test pattern
    // (tests/test_water_conservation_stress.py) for the donor-cell transport
    // this function ports. `gas` is (n_gases, h, w) contiguous, Q16.16,
    // mutated in place.
    m.def("bulk_flux_transport",
          [](py::array_t<int32_t> gas, py::array_t<bool> gas_conservative,
             py::array_t<int32_t> wind_x, py::array_t<int32_t> wind_y,
             py::array_t<bool> solid, py::array_t<bool> is_vacuum,
             py::array_t<float> dyn_permeability, float dt) {
              auto gv = gas.mutable_unchecked<3>();
              int32_t* gas_ptr = gv.mutable_data(0, 0, 0);
              const int n_gases = static_cast<int>(gv.shape(0));
              const int h = static_cast<int>(gv.shape(1));
              const int w = static_cast<int>(gv.shape(2));
              auto gc = gas_conservative.unchecked<1>();
              const bool* gcons = gc.data(0);
              auto [wx, h2, w2] = get_2d_const(wind_x);
              auto [wy, h3, w3] = get_2d_const(wind_y);
              auto [sol, h4, w4] = get_2d_const(solid);
              auto [vac, h5, w5] = get_2d_const(is_vacuum);
              auto [perm, h6, w6] = get_2d_const(dyn_permeability);
              bulk_flux_transport(
                  gas_ptr, gcons, n_gases,
                  wx, wy, sol, vac, perm,
                  h, w, dt);
          }, py::arg("gas"), py::arg("gas_conservative"),
             py::arg("wind_x"), py::arg("wind_y"),
             py::arg("solid"), py::arg("is_vacuum"),
             py::arg("dyn_permeability"), py::arg("dt"),
          "EOS P1: donor-cell conservative flux transport of every "
          "`gas_conservative`-flagged plane, once, on the given wind field.");

    // sky-exchange (docs/sky_exchange_design_2026-07-24.md): the planetside
    // volumetric-O2 per-tick host pass — relax each sky-connected air tile's
    // composition toward ambient at FIXED local N_total. `gas` is (n_gases,h,w)
    // Q16.16, mutated in place (O2 + inert only); `sky_flux` is an (n_gases,)
    // int64 rail ACCUMULATED into (the caller clears it per tick). Exposed for
    // the runner tick AND for the gate a/b/c unit tests (tests/test_sky_exchange).
    m.def("sky_exchange_step",
          [](py::array_t<int32_t> gas, int o2_idx, int inert_idx,
             py::array_t<bool> sky_mask,
             int32_t o2_frac_q, int32_t lambda_q,
             py::array_t<int64_t> sky_flux) {
              auto gv = gas.mutable_unchecked<3>();
              int32_t* gas_ptr = gv.mutable_data(0, 0, 0);
              const int n_gases = static_cast<int>(gv.shape(0));
              const int h = static_cast<int>(gv.shape(1));
              const int w = static_cast<int>(gv.shape(2));
              auto [mask, hm, wm] = get_2d_const(sky_mask);
              auto fv = sky_flux.mutable_unchecked<1>();
              int64_t* flux_ptr = fv.mutable_data(0);
              sky_exchange_step(
                  gas_ptr, n_gases, o2_idx, inert_idx,
                  mask, h, w, o2_frac_q, lambda_q, flux_ptr);
          }, py::arg("gas"), py::arg("o2_idx"), py::arg("inert_idx"),
             py::arg("sky_mask"),
             py::arg("o2_frac_q"), py::arg("lambda_q"), py::arg("sky_flux"),
          "sky-exchange: per-tick composition relaxation of sky-connected air "
          "toward ambient at fixed local N_total (O2 up / inert down); "
          "sky_flux[gas] accumulates the actual applied delta (conservation rail).");

    // Q2-LIFT: the deterministic trig kit (fixed_point.h). Pure integer q16 ->
    // q16 — the cross-machine-safe replacement for the libm transcendentals in
    // the SYNCED unit state (facing atan2, combat bullet cos/sin) and the
    // raycaster ray dirs. Python quantizes its float radians at the boundary,
    // calls these, and dequantizes back (exact n/65536 doubles). Exposed in
    // BOTH the CPU and CUDA builds (plain defs, no #ifdef) so every backend's
    // Python layer computes the identical bits. Accuracy is gated by
    // tests/test_fixed_trig.py (pinned <= 9.0e-6 vs double libm).
    m.def("atan2_q16",
          [](int32_t y, int32_t x) { return fixedpoint::atan2_q16(y, x); },
          py::arg("y"), py::arg("x"),
          "Q2-LIFT: pure-integer atan2 on Q16.16 (radians out, Q16.16; range "
          "[-205887, +205887] == [-quantize(pi), +quantize(pi)]).");
    m.def("sin_q16",
          [](int32_t a) { return fixedpoint::sin_q16(a); },
          py::arg("a"),
          "Q2-LIFT: pure-integer sin on Q16.16 radians (output Q16.16 in "
          "[-65536, 65536]; accuracy pinned for |a| <= 4*pi, any int32 defined).");
    m.def("cos_q16",
          [](int32_t a) { return fixedpoint::cos_q16(a); },
          py::arg("a"),
          "Q2-LIFT: pure-integer cos on Q16.16 radians (output Q16.16 in "
          "[-65536, 65536]; accuracy pinned for |a| <= 4*pi, any int32 defined).");

    // FUEL-FRACTION AXIS (2026-07-30): the load-time reciprocal bake itself,
    // exposed so Python can be GATED against it rather than trusted to
    // reproduce it. `src/simulation/materials.fuel_recip_from_hp` bakes each
    // material's 1/hp for `GameMap.fuel_recip`, and that plane must be
    // BIT-IDENTICAL to what `fixedpoint::make_recip` would have produced in the
    // solver — a one-ULP disagreement is a determinism bug, not a rounding
    // detail. tests/test_fuel_fraction_axis.py compares the two over every
    // shipped material and a wide sweep through this entry point.
    m.def("fp_make_recip",
          [](double divisor) { return fixedpoint::make_recip(divisor); },
          py::arg("divisor"),
          "fixed_point.h make_recip: round(2^32 / divisor) as an int64, the "
          "load-time reciprocal `recip_mul` consumes. Divisor must be > 0.");

    // PER-MATERIAL fire_T_ext (P-R3, 2026-07-31 — ruling A3 ride-along): the
    // boundary cast itself, exposed for the SAME reason `fp_make_recip` above
    // is. `src/simulation/materials.quantize_q16` bakes each material's
    // `ignition_temp - Δ` into `GameMap.fire_T_ext_plane`, and that plane must
    // be BIT-IDENTICAL to what `fixedpoint::quantize` would have produced from
    // the scalar — that agreement IS the "uniform plane == scalar fallback"
    // back-compat contract. tests/test_pr3_capacity_law.py gates the two.
    m.def("fp_quantize",
          [](double v) { return fixedpoint::quantize(v); },
          py::arg("v"),
          "fixed_point.h quantize: a real value -> Q16.16 int32, "
          "round-half-away-from-zero, computed in double.");

    // P-E2b (energy-books arc, design §2.2): exposed so the n_floor_heat
    // low-dial sweep (0.05 -> 0.01, reachable down to 0.001) can be verified
    // against the ACTUAL per-cell reciprocal + the deposit multiply chain
    // both deposit sites use, rather than a re-derived Python approximation
    // (tools/e2b_floor_reciprocal_probe.py).
    m.def("fp_reciprocal_q16",
          [](int32_t denom_q) { return fixedpoint::reciprocal_q16(denom_q); },
          py::arg("denom_q"),
          "fixed_point.h reciprocal_q16: per-cell Newton reciprocal, Q16.16 "
          "in -> Q16.16 out (int64 internally, ~1 ULP accurate; self-guards "
          "denom_q <= 0 -> 0 and floors {1,2} to 3).");
    m.def("fp_recip_mul",
          [](int32_t x_q16, int64_t recip) {
              return fixedpoint::recip_mul(x_q16, recip);
          },
          py::arg("x_q16"), py::arg("recip"),
          "fixed_point.h recip_mul: x_q16 * recip >> RECIP_SHIFT via a "
          "128-bit intermediate (the make_recip reciprocal multiply, used "
          "for the deposit's .../c_v step).");
    m.def("fp_deposit_dT_wide_q16",
          [](int32_t deposit_q, int32_t recip_n_q, int64_t recip_cv) {
              return fixedpoint::deposit_dT_wide_q16(deposit_q, recip_n_q,
                                                      recip_cv);
          },
          py::arg("deposit_q"), py::arg("recip_n_q"), py::arg("recip_cv"),
          "fixed_point.h deposit_dT_wide_q16 (P-E2b): deposit/(N*c_v) chained "
          "as one 128-bit product, narrowed ONCE to int64 (not q16) — the "
          "fix for the old two-step mul_q16->recip_mul chain's silent q16 "
          "overflow at low n_floor_heat. Both deposit sites (combustion.cpp, "
          "temperature_solver.cpp Pass 1) use this; exposed so it can be "
          "verified directly rather than re-derived.");

    // S2a: the explicit WAVE state (wave_p / wave_v / wave_source) is now int32
    // Q16.16 (same 2^16 scale as water/heat). Python (gamemap fields, field
    // edits, the recorder boundary, tests) reads this flag to allocate the wave
    // fields as int32 and to quantize/dequantize at the boundaries. wave_v keeps
    // Q16.16 (the Q-S2-2 measurement: peak |wave_v| ~2674 << 32768).
    m.attr("WAVE_FIXEDPOINT") = true;
    m.attr("WAVE_FP_SHIFT") = 16;
    m.attr("WAVE_FP_ONE") = 65536;

    // S2c: the atmosphere (bulk pressure) + wind (= -grad(atm+wave_p)) are now
    // int32 Q16.16 (same 2^16 scale). This CLOSES the S2 group: the whole
    // atmosphere/wave/wind/smoke/gas group is now cross-GPU deterministic (the
    // only float bridge left is the downstream FIRE coupling, S3). Python
    // (gamemap fields, field edits, the recorder/render dequantize, tests) reads
    // these flags to allocate atmosphere/wind_x/wind_y as int32 and to
    // quantize/dequantize at the boundaries.
    m.attr("ATMOSPHERE_FIXEDPOINT") = true;
    m.attr("ATMOSPHERE_FP_SHIFT") = 16;
    m.attr("ATMOSPHERE_FP_ONE") = 65536;
    m.attr("WIND_FIXEDPOINT") = true;
    m.attr("WIND_FP_SHIFT") = 16;
    m.attr("WIND_FP_ONE") = 65536;

    // --- AtmosphereSolver (IMEX: explicit wave + implicit diffusion) ---
    py::class_<AtmosphereSolver>(m, "AtmosphereSolver")
        .def(py::init<>())
        .def_readwrite("c",                   &AtmosphereSolver::c)
        .def_readwrite("damping",             &AtmosphereSolver::damping)
        .def_readwrite("absorb_strength",     &AtmosphereSolver::absorb_strength)
        .def_readwrite("transfer",            &AtmosphereSolver::transfer)
        .def_readwrite("d_atm",               &AtmosphereSolver::d_atm)
        .def_readwrite("feed_rate",           &AtmosphereSolver::feed_rate)
        .def_readwrite("breach_rate",         &AtmosphereSolver::breach_rate)
        .def_readwrite("max_source_per_step", &AtmosphereSolver::max_source_per_step)
        .def_readwrite("gs_iters",            &AtmosphereSolver::gs_iters)
        // Patch 2a: GS-residual diagnostic (read-only). Linf of the implicit
        // operator residual normalized by max|atm|, measured inside
        // diffuse_solve after the GS sweeps but before the BC pass. Lets Python
        // read `engine.atmos.last_gs_residual` to check 8 sweeps don't under-
        // relax at the big once-per-tick dt. Nothing reads it yet.
        .def_readonly("last_gs_residual",     &AtmosphereSolver::last_gs_residual)
        .def("gs_residual", &AtmosphereSolver::gs_residual)
        .def("max_dt", &AtmosphereSolver::max_dt)
        .def("max_dt_q", &AtmosphereSolver::max_dt_q)   // Bedrock cliff-patch: Q16.16 CFL constant
        .def("step", [](const AtmosphereSolver& self,
                        py::array_t<int32_t> wave_p,        // S2a: Q16.16 int32
                        py::array_t<int32_t> wave_v,        // S2a: Q16.16 int32
                        py::array_t<int32_t> wave_source,   // S2a: Q16.16 int32
                        py::array_t<int32_t> atmosphere,    // S2c: Q16.16 int32
                        py::array_t<int32_t> wind_x,        // S2c: Q16.16 int32
                        py::array_t<int32_t> wind_y,        // S2c: Q16.16 int32
                        py::array_t<bool>  obstacles,
                        py::array_t<bool>  is_wall,
                        py::array_t<bool>  is_vacuum,
                        py::array_t<float> permeability,
                        py::array_t<float> wave_absorb,
                        float dt) {
            auto [wp, h, w]   = get_2d(wave_p);
            auto [wv, h2, w2] = get_2d(wave_v);
            auto [ws, h3, w3] = get_2d(wave_source);
            auto [atm, h4, w4] = get_2d(atmosphere);
            auto [wx, h5, w5] = get_2d(wind_x);
            auto [wy, h6, w6] = get_2d(wind_y);
            auto [obs, h7, w7] = get_2d_const(obstacles);
            auto [wl, h8, w8] = get_2d_const(is_wall);
            auto [vac, h9, w9] = get_2d_const(is_vacuum);
            auto [perm, h10, w10] = get_2d_const(permeability);
            auto [wabs, h11, w11] = get_2d_const(wave_absorb);
            self.step(wp, wv, ws, atm, wx, wy, obs, wl, vac, perm, wabs, h, w, dt);
        }, py::arg("wave_p"), py::arg("wave_v"), py::arg("wave_source"),
           py::arg("atmosphere"),
           py::arg("wind_x"), py::arg("wind_y"),
           py::arg("obstacles"), py::arg("is_wall"), py::arg("is_vacuum"),
           py::arg("permeability"),
           py::arg("wave_absorb"),
           py::arg("dt"))
        // CUDA-S5: the wave-only substep (NO diffuse_solve), exposed so the GPU
        // bit-identity gate has a CLEAN CPU reference for wave_substep ALONE (step()
        // also runs diffuse_solve, which mutates atmosphere/wave_v in the sponge BC
        // and so is NOT a wave-only oracle). Same arg list as step() minus wind_x/
        // wind_y (wave_substep writes neither).
        .def("wave_substep", [](const AtmosphereSolver& self,
                                py::array_t<int32_t> wave_p,      // Q16.16 int32
                                py::array_t<int32_t> wave_v,      // Q16.16 int32
                                py::array_t<int32_t> wave_source, // Q16.16 int32
                                py::array_t<int32_t> atmosphere,  // Q16.16 int32
                                py::array_t<bool>  obstacles,
                                py::array_t<bool>  is_wall,
                                py::array_t<bool>  is_vacuum,
                                py::array_t<float> permeability,
                                py::array_t<float> wave_absorb,
                                float dt) {
            auto [wp, h, w]   = get_2d(wave_p);
            auto [wv, h2, w2] = get_2d(wave_v);
            auto [ws, h3, w3] = get_2d(wave_source);
            auto [atm, h4, w4] = get_2d(atmosphere);
            auto [obs, h5, w5] = get_2d_const(obstacles);
            auto [wl, h6, w6] = get_2d_const(is_wall);
            auto [vac, h7, w7] = get_2d_const(is_vacuum);
            auto [perm, h8, w8] = get_2d_const(permeability);
            auto [wabs, h9, w9] = get_2d_const(wave_absorb);
            self.wave_substep(wp, wv, ws, atm, obs, wl, vac, perm, wabs, h, w, dt);
        }, py::arg("wave_p"), py::arg("wave_v"), py::arg("wave_source"),
           py::arg("atmosphere"),
           py::arg("obstacles"), py::arg("is_wall"), py::arg("is_vacuum"),
           py::arg("permeability"), py::arg("wave_absorb"),
           py::arg("dt"))
        // CUDA-S7: the implicit diffusion + sponge + wind ALONE (NO wave_substep),
        // exposed so the GPU bit-identity gate has a CLEAN CPU reference for
        // diffuse_solve (step() also runs wave_substep first). Arg order matches the
        // C++ diffuse_solve signature (atmosphere, wave_p, wave_v, wave_source,
        // wind_x, wind_y, masks, perm, dt) — note atmosphere is FIRST here (it is the
        // RB-GS target), unlike step()/wave_substep where wave_p leads.
        .def("diffuse_solve", [](const AtmosphereSolver& self,
                                 py::array_t<int32_t> atmosphere,   // Q16.16 int32
                                 py::array_t<int32_t> wave_p,       // Q16.16 int32
                                 py::array_t<int32_t> wave_v,       // Q16.16 int32
                                 py::array_t<int32_t> wave_source,  // Q16.16 int32
                                 py::array_t<int32_t> wind_x,       // Q16.16 int32
                                 py::array_t<int32_t> wind_y,       // Q16.16 int32
                                 py::array_t<bool>  obstacles,
                                 py::array_t<bool>  is_wall,
                                 py::array_t<bool>  is_vacuum,
                                 py::array_t<float> permeability,
                                 float dt) {
            auto [atm, h, w]   = get_2d(atmosphere);
            auto [wp, h2, w2]  = get_2d(wave_p);
            auto [wv, h3, w3]  = get_2d(wave_v);
            auto [ws, h4, w4]  = get_2d(wave_source);
            auto [wx, h5, w5]  = get_2d(wind_x);
            auto [wy, h6, w6]  = get_2d(wind_y);
            auto [obs, h7, w7] = get_2d_const(obstacles);
            auto [wl, h8, w8]  = get_2d_const(is_wall);
            auto [vac, h9, w9] = get_2d_const(is_vacuum);
            auto [perm, h10, w10] = get_2d_const(permeability);
            self.diffuse_solve(atm, wp, wv, ws, wx, wy, obs, wl, vac, perm, h, w, dt);
        }, py::arg("atmosphere"), py::arg("wave_p"), py::arg("wave_v"),
           py::arg("wave_source"), py::arg("wind_x"), py::arg("wind_y"),
           py::arg("obstacles"), py::arg("is_wall"), py::arg("is_vacuum"),
           py::arg("permeability"), py::arg("dt"));

    // --- SmokeDynamics (uses precomputed wind from AtmosphereSolver) ---
    py::class_<SmokeDynamics>(m, "SmokeDynamics")
        .def(py::init<>())
        .def_readwrite("d_smoke",               &SmokeDynamics::d_smoke)
        .def_readwrite("advection_rate",         &SmokeDynamics::advection_rate)
        .def_readwrite("wind_diffusion_scale",   &SmokeDynamics::wind_diffusion_scale)
        // (sink_strength / vent_hops / sink_hop DELETED — EOS refactor P3,
        // decisions.md #3: native venting replaces the BFS sink-pull.)
        .def("step", [](const SmokeDynamics& self,
                        py::array_t<int32_t> smoke,        // S2b: Q16.16 int32
                        py::array_t<int32_t> wind_x,       // S2c: Q16.16 int32
                        py::array_t<int32_t> wind_y,       // S2c: Q16.16 int32
                        py::array_t<bool>  obstacles,
                        py::array_t<bool>  is_wall,
                        py::array_t<bool>  is_vacuum,
                        py::array_t<float> permeability,
                        float dt) {
            auto [sm, h, w] = get_2d(smoke);
            auto [wx, h2, w2] = get_2d_const(wind_x);
            auto [wy, h3, w3] = get_2d_const(wind_y);
            auto [obs, h4, w4] = get_2d_const(obstacles);
            auto [wl, h5, w5] = get_2d_const(is_wall);
            auto [vac, h6, w6] = get_2d_const(is_vacuum);
            auto [perm, h7, w7] = get_2d_const(permeability);
            self.step(sm, wx, wy, obs, wl, vac, perm, h, w, dt);
        }, py::arg("smoke"), py::arg("wind_x"), py::arg("wind_y"),
           py::arg("obstacles"), py::arg("is_wall"), py::arg("is_vacuum"),
           py::arg("permeability"),
           py::arg("dt"));

    // --- FireSimulation (signed-logistic feedback; fire_design_proposal §2/§3) ---
    py::class_<FireParams>(m, "FireParams")
        .def(py::init<>())
        .def_readwrite("k_grow",         &FireParams::k_grow)
        .def_readwrite("k_die",          &FireParams::k_die)
        // CAPACITY LAW (P-R3, ruling A3): `c` — the SIZE dial (I_eq ~= c*a).
        .def_readwrite("I_cap_per_avail", &FireParams::I_cap_per_avail)
        // fire_T_ext is now the FALLBACK only — the live gate is per-material
        // (GameMap.fire_T_ext_plane, ignition_temp - ignition_to_ext_delta).
        .def_readwrite("fire_T_ext",     &FireParams::fire_T_ext)
        .def_readwrite("fire_T_span",    &FireParams::fire_T_span)
        .def_readwrite("fuel_ref",       &FireParams::fuel_ref)
        .def_readwrite("o2_frac_ext",    &FireParams::o2_frac_ext)
        // FULL-RESPONSE REFERENCE SPLIT: o2_frac_full is the span's upper end
        // (pure O2, 1.0, NOT map-overridden); o2_frac_amb is the ambient record
        // and is no longer read by the availability law.
        .def_readwrite("o2_frac_full",   &FireParams::o2_frac_full)
        .def_readwrite("o2_frac_amb",    &FireParams::o2_frac_amb)
        .def_readwrite("P_min",          &FireParams::P_min)
        .def_readwrite("P_full",         &FireParams::P_full)
        .def_readwrite("I_min",          &FireParams::I_min)
        .def_readwrite("k_wind_fan",     &FireParams::k_wind_fan)
        .def_readwrite("k_wind_strip",   &FireParams::k_wind_strip)
        .def_readwrite("p_expand_ref",   &FireParams::p_expand_ref)
        // smoke_emission RETIRED at P-S1 (2026-08-15) — the field it drove
        // was deleted from FireParams itself (fire_simulation.h), not just
        // unbound here, so an old caller gets a loud AttributeError instead
        // of silently writing a dial with nothing behind it.
        .def_readwrite("wall_damage",    &FireParams::wall_damage)
        .def_readwrite("temp_scale",     &FireParams::temp_scale);

    py::class_<FireSimulation>(m, "FireSimulation")
        .def(py::init<>())
        .def_readwrite("params", &FireSimulation::params)
        // DEBUG probe (temporary). Its former partner dbg_plume_dT (the
        // plume->T shim's traced-cell deposit) was removed with the shim
        // (P-R2 — docs/radiation_raycaster_extinction_ruling_2026-07-31.md A2).
        .def_readwrite("dbg_probe_idx", &FireSimulation::dbg_probe_idx)
        .def("step", [](const FireSimulation& self,
                        py::array_t<int32_t> fire,         // S3b: Q16.16 int32
                        py::array_t<int32_t> atmosphere,   // S2c: Q16.16 int32
                        py::array_t<int32_t> n_o2,         // EOS P4: Q16.16 int32
                        py::array_t<int32_t> n_total,      // continuous-O2 law: Q16.16 int32
                        py::array_t<int32_t> smoke,        // S2b: Q16.16 int32
                        py::array_t<int32_t> wall_hp,      // S3b: Q16.16 int32
                        py::array_t<int32_t> temperature,
                        py::array_t<int32_t> wind_x,       // S2c: Q16.16 int32
                        py::array_t<int32_t> wind_y,       // S2c: Q16.16 int32
                        py::array_t<bool>  is_wall,
                        py::array_t<bool>  is_vacuum,
                        py::array_t<bool>  flammable,
                        float dt,
                        py::object fuel_recip_obj,
                        py::object fire_T_ext_plane_obj) -> py::list {
            auto [f, h, w] = get_2d(fire);
            auto [atm, h2, w2] = get_2d_const(atmosphere);   // EOS P3: read-only (== P)
            auto [o2, h2b, w2b] = get_2d_const(n_o2);        // fraction numerator (read-only)
            auto [nt, h2c, w2c] = get_2d_const(n_total);     // fraction denominator (read-only)
            auto [sm, h3, w3] = get_2d(smoke);
            auto [whp, h4, w4] = get_2d(wall_hp);
            auto [temp, h5, w5] = get_2d(temperature);       // mutable type; READ
                                                              // only as of P-R2
            auto [wx, h6, w6] = get_2d_const(wind_x);
            auto [wy, h7, w7] = get_2d_const(wind_y);
            auto [wl, h8, w8] = get_2d_const(is_wall);
            auto [vac, h9, w9] = get_2d_const(is_vacuum);
            auto [fl, h10, w10] = get_2d_const(flammable);
            // FUEL-FRACTION AXIS (2026-07-30): the per-tile 1/hp plane is
            // OPTIONAL (default None) — None -> nullptr, and the solver then
            // normalises F by the scalar `params.fuel_ref`, which is the
            // pre-axis behaviour BIT-FOR-BIT. This is the documented
            // back-compat idiom the thermal-mass axis established on the
            // standalone TemperatureSolver binding: every shipped DIRECT caller
            // of this binding (tests, tools) keeps its exact prior meaning,
            // while the engine's step_tail always passes GameMap.fuel_recip.
            const int64_t* fr = nullptr;
            py::array_t<int64_t> fr_arr;
            if (!fuel_recip_obj.is_none()) {
                fr_arr = fuel_recip_obj.cast<py::array_t<int64_t>>();
                auto fv = fr_arr.unchecked<2>();
                fr = fv.data(0, 0);
            }
            // PER-MATERIAL T_ext (P-R3, ruling A3 ride-along): the same
            // OPTIONAL nullable-plane idiom — None -> nullptr -> the scalar
            // `params.fire_T_ext`, which is the pre-derivation law BIT-FOR-BIT.
            const int32_t* tep = nullptr;
            py::array_t<int32_t> tep_arr;
            if (!fire_T_ext_plane_obj.is_none()) {
                tep_arr = fire_T_ext_plane_obj.cast<py::array_t<int32_t>>();
                auto tv = tep_arr.unchecked<2>();
                tep = tv.data(0, 0);
            }
            auto destroyed = self.step(f, atm, o2, nt, sm, whp, temp, wx, wy,
                                       wl, vac, fl, h, w, dt, fr, tep);
            py::list result;
            for (const auto& [dy, dx] : destroyed) {
                result.append(py::make_tuple(dy, dx));
            }
            return result;
        }, py::arg("fire"), py::arg("atmosphere"), py::arg("n_o2"), py::arg("n_total"),
           py::arg("smoke"), py::arg("wall_hp"), py::arg("temperature"),
           py::arg("wind_x"), py::arg("wind_y"),
           py::arg("is_wall"), py::arg("is_vacuum"), py::arg("flammable"),
           py::arg("dt"),
           py::arg("fuel_recip") = py::none(),        // fuel-fraction axis (optional)
           py::arg("fire_T_ext_plane") = py::none()); // per-material T_ext (optional)

    // --- TemperatureSolver (heat -> temperature conversion §1 + conduction §2
    //     + ambient cooling §3; engine/06 §1–§3) ---
    py::class_<TemperatureSolver>(m, "TemperatureSolver")
        .def(py::init<>())
        // DEBUG probe (temporary, eos-p3fix-thermal-ceiling investigation).
        .def_readwrite("dbg_probe_idx",         &TemperatureSolver::dbg_probe_idx)
        .def_readonly("dbg_T_post_heat",        &TemperatureSolver::dbg_T_post_heat)
        .def_readonly("dbg_T_post_conduction",  &TemperatureSolver::dbg_T_post_conduction)
        .def_readonly("dbg_T_post_cooling",     &TemperatureSolver::dbg_T_post_cooling)
        // NO_FACE sentinel (face_shift == this -> skip the face). Bound from
        // config [physics.thermal].NO_FACE so Python and C++ never disagree.
        .def_property("no_face",
            &TemperatureSolver::get_no_face,
            &TemperatureSolver::set_no_face)
        // Ambient cooling dials (§3.3), bound from config [physics.thermal].
        .def_property("cool_shift",
            &TemperatureSolver::get_cool_shift,
            &TemperatureSolver::set_cool_shift)
        .def_property("cool_shift_vacuum",
            &TemperatureSolver::get_cool_shift_vacuum,
            &TemperatureSolver::set_cool_shift_vacuum)
        // COOL-SHIFT AXIS (2026-07-30): the low clamp applied when the global
        // vacuum offset (cool_shift - cool_shift_vacuum) is subtracted from a
        // tile's per-material shift. Bound from config [physics.thermal]
        // SHIFT_MIN — the same floor materials.py validates the column against.
        .def_property("cool_shift_floor",
            &TemperatureSolver::get_cool_shift_floor,
            &TemperatureSolver::set_cool_shift_floor)
        .def_property("o2_vacuum_thresh",
            &TemperatureSolver::get_o2_vacuum_thresh,
            &TemperatureSolver::set_o2_vacuum_thresh)
        // --- P2 gas-T dials (docs/eos_refactor_design.md §4.3, §9) ---------
        .def_property("gas_advection_rate",
            &TemperatureSolver::get_gas_advection_rate,
            &TemperatureSolver::set_gas_advection_rate)
        .def_property("c_v",
            &TemperatureSolver::get_c_v,
            &TemperatureSolver::set_c_v)
        .def_property("n_floor_heat",
            &TemperatureSolver::get_n_floor_heat,
            &TemperatureSolver::set_n_floor_heat)
        // v2.4 T_MAX_PHYS rail + counter (temperature_solver.h).
        .def_readwrite("T_MAX_PHYS",        &TemperatureSolver::T_MAX_PHYS)
        .def_readonly("t_max_phys_hits",    &TemperatureSolver::t_max_phys_hits)
        // P-F1a (v7.2): the Pass-1 LOW rail's engagement count. The radiation
        // fold is the only SIGNED path into `temperature`, so it is the only
        // one that can drive a tile below the ambient floor at 0. Justified
        // INERT by the per-term budget argument — a nonzero count inside a gate
        // scenario is a RED, which is exactly why it is readable from Python.
        .def_readonly("t_low_rail_hits",    &TemperatureSolver::t_low_rail_hits)
        // P-E2a energy books (design §2.3/§5/§7): conduction's two counted
        // residuals + the limiter's engagement count, then the three
        // open-by-design SIGNED channels (L3-6). All in RAW ENERGY counts
        // except `cond_limit_hits`; all accumulating across step() calls (the
        // ledger diffs them per tick), and shared with the CUDA path.
        .def_readonly("e_cond_trunc_sum",   &TemperatureSolver::e_cond_trunc_sum)
        .def_readonly("e_cond_cap_sum",     &TemperatureSolver::e_cond_cap_sum)
        .def_readonly("cond_limit_hits",    &TemperatureSolver::cond_limit_hits)
        .def_readonly("e_cool_sum",         &TemperatureSolver::e_cool_sum)
        .def_readonly("e_vac_wipe_sum",     &TemperatureSolver::e_vac_wipe_sum)
        .def_readonly("e_ring_pin_sum",     &TemperatureSolver::e_ring_pin_sum)
        // P-E2b (design §2.2/§2.5, L3-7): the Pass-1 attenuation-drop energy
        // sum. Same accumulate-across-step() idiom as the P-E2a six above.
        .def_readonly("e_deposit_drop_sum", &TemperatureSolver::e_deposit_drop_sum)
        // P2: wind_x/wind_y/dt are OPTIONAL (default None/0.0) so the shipped
        // direct-binding call sites (tests/test_temperature_*.py,
        // tests/cuda_s1_check.py — all pre-P2, 7 positional args) keep working
        // UNCHANGED: with wind omitted the gas-T advection pre-pass (Pass 0) is
        // a clean no-op (see temperature_solver.cpp step()'s dt<=0/null guard),
        // so those callers exercise exactly the solid convert/conduct/cool path
        // (PLUS the new gas radiation deposit of Pass 1, which is NOT gated by
        // wind/dt — a designed P2 behaviour change on any open-air cell with a
        // nonzero `heat` deposit; see the P2 gate report for the tests this
        // moves). PhysicsEngine::step_tail (physics_engine.cpp) always calls the
        // real C++ TemperatureSolver::step with real wind/dt — this optional
        // path only exists for the standalone Python binding.
        .def("step", [](const TemperatureSolver& self,
                        py::array_t<int32_t> temperature,
                        py::array_t<int32_t> heat,
                        py::array_t<int32_t> heat_inv_shift,
                        py::array_t<int32_t> face_shift,
                        py::array_t<bool>    solid,
                        py::array_t<bool>    is_vacuum,
                        py::array_t<int32_t> atmosphere,   // S3c: Q16.16 int32 (was float)
                        py::object wind_x_obj,
                        py::object wind_y_obj,
                        float dt,
                        py::object n_bulk_obj,
                        py::object thermal_solid_obj,
                        py::object cool_shift_grid_obj,
                        py::object rad_net_obj) {
            auto [temp, h, w]     = get_2d(temperature);
            auto [hp, h2, w2]     = get_2d_const(heat);
            auto [shift, h3, w3]  = get_2d_const(heat_inv_shift);
            auto [sol, h4, w4]    = get_2d_const(solid);
            auto [vac, h5, w5]    = get_2d_const(is_vacuum);
            auto [atm, h6, w6]    = get_2d_const(atmosphere);
            // face_shift is (h, w, 4) int32 — fixed dir order N,S,E,W.
            auto fa = face_shift.unchecked<3>();
            const int32_t* fs = fa.data(0, 0, 0);
            // P2: wind_x/wind_y are optional numpy arrays; None -> nullptr, and
            // step() self-guards dt<=0/null into a no-op advection pass. Keep the
            // extracted py::array_t alive in this scope (wx_arr/wy_arr) so the
            // pointers stay valid through the self.step() call below.
            const int32_t* wx = nullptr;
            const int32_t* wy = nullptr;
            py::array_t<int32_t> wx_arr, wy_arr;
            if (!wind_x_obj.is_none() && !wind_y_obj.is_none()) {
                wx_arr = wind_x_obj.cast<py::array_t<int32_t>>();
                wy_arr = wind_y_obj.cast<py::array_t<int32_t>>();
                auto [wxp, hx, wxw] = get_2d_const(wx_arr);
                auto [wyp, hy, wyw] = get_2d_const(wy_arr);
                wx = wxp;
                wy = wyp;
            }
            // EOS P3/P6.6: n_bulk is OPTIONAL (default None). None -> nullptr, so
            // the solver falls back to the atmosphere density-proxy (the shipped
            // back-compat path — all pre-P6.6 direct callers keep that behaviour).
            // The P6.6 GPU bit-identity gate passes the real O2+N2 sum here so the
            // CPU reference and the GPU kernel divide by the SAME N (the engine's
            // step_tail always passes the real sum too).
            const int32_t* nb = nullptr;
            py::array_t<int32_t> nb_arr;
            if (!n_bulk_obj.is_none()) {
                nb_arr = n_bulk_obj.cast<py::array_t<int32_t>>();
                auto [nbp, hn, wn] = get_2d_const(nb_arr);
                nb = nbp;
            }
            // THERMAL-MASS AXIS (docs/thermal_mass_axis_design_2026-07-25.md):
            // `thermal_solid` is OPTIONAL (default None) — None -> nullptr, and
            // the solver falls back to `solid`, i.e. the exact pre-patch
            // behaviour. Every shipped direct caller (tests/test_temperature_*)
            // therefore keeps its meaning unchanged; the engine's step_tail
            // always passes the real GameMap.thermal_solid.
            const bool* tsol = nullptr;
            py::array_t<bool> tsol_arr;
            if (!thermal_solid_obj.is_none()) {
                tsol_arr = thermal_solid_obj.cast<py::array_t<bool>>();
                auto [tsp, ht, wt] = get_2d_const(tsol_arr);
                tsol = tsp;
            }
            // COOL-SHIFT AXIS (2026-07-30): `cool_shift_grid` is OPTIONAL by the
            // same idiom — None -> nullptr, and Pass 3 uses the solver's scalar
            // `cool_shift` for every tile, i.e. the exact pre-axis single-global
            // behaviour every shipped direct caller (tests/test_temperature_*)
            // relies on. The engine's step_tail always passes GameMap.cool_shift.
            const int32_t* csg = nullptr;
            py::array_t<int32_t> csg_arr;
            if (!cool_shift_grid_obj.is_none()) {
                csg_arr = cool_shift_grid_obj.cast<py::array_t<int32_t>>();
                auto [csp, hc, wc] = get_2d_const(csg_arr);
                csg = csp;
            }
            // P-R4 (ruling A1.7): `rad_net` is OPTIONAL by the SAME idiom —
            // None -> nullptr and Pass 1 does no radiation fold, i.e. exactly
            // the pre-P-R4 behaviour every shipped direct caller relies on.
            const int32_t* rnet = nullptr;
            py::array_t<int32_t> rnet_arr;
            if (!rad_net_obj.is_none()) {
                rnet_arr = rad_net_obj.cast<py::array_t<int32_t>>();
                auto [rnp, hr, wr] = get_2d_const(rnet_arr);
                rnet = rnp;
            }
            self.step(temp, hp, shift, fs, sol, vac, atm, nb, wx, wy, h, w, dt,
                      nullptr, tsol, csg, rnet);
        }, py::arg("temperature"), py::arg("heat"),
           py::arg("heat_inv_shift"), py::arg("face_shift"),
           py::arg("solid"), py::arg("is_vacuum"), py::arg("atmosphere"),
           py::arg("wind_x") = py::none(), py::arg("wind_y") = py::none(),
           py::arg("dt") = 0.0f, py::arg("n_bulk") = py::none(),
           py::arg("thermal_solid") = py::none(),
           py::arg("cool_shift_grid") = py::none(),
           py::arg("rad_net") = py::none());

    // --- Raycaster ---
    py::class_<LightSource>(m, "LightSource")
        .def(py::init<>())
        .def_readwrite("x", &LightSource::x)
        .def_readwrite("y", &LightSource::y)
        .def_readwrite("max_range", &LightSource::max_range)
        .def_readwrite("ray_count", &LightSource::ray_count)
        .def_readwrite("angle_center", &LightSource::angle_center)
        .def_readwrite("angle_spread", &LightSource::angle_spread)
        .def_readwrite("intensity", &LightSource::intensity)
        .def_readwrite("heat", &LightSource::heat)
        .def_readwrite("jitter", &LightSource::jitter)
        // RGB tint exposed as a 3-tuple (r, g, b). Default white {1,1,1}.
        .def_property("color",
            [](const LightSource& s) {
                return py::make_tuple(s.color[0], s.color[1], s.color[2]);
            },
            [](LightSource& s, const std::array<float, 3>& c) {
                s.color[0] = c[0]; s.color[1] = c[1]; s.color[2] = c[2];
            });

    py::class_<Raycaster>(m, "Raycaster")
        .def(py::init<>())
        .def_readwrite("smoke_absorption", &Raycaster::smoke_absorption)
        .def_readwrite("smoke_absorb_scale", &Raycaster::smoke_absorb_scale)
        // Pure-density propagation cull floors (engine/08 §The march): per-channel
        // survival thresholds. `light_cull` = ε_rgb (render), `heat_cull` = ε_heat
        // (gameplay/damage, its own dial so heat-shield materials can diverge).
        .def_readwrite("light_cull", &Raycaster::light_cull)
        .def_readwrite("heat_cull", &Raycaster::heat_cull)
        // P-R4 radiation dials (ruling A1). `rad_scale` is the E° bake's
        // emission calibration (heat counts per K⁴); the table re-bakes lazily
        // whenever it moves, so setting the dial is enough. `T_emit_gate` is the
        // warm-emitter threshold in GAME temperature units.
        .def_readwrite("rad_scale", &Raycaster::rad_scale)
        // Canonical game-T -> Kelvin map (temperature_scale_unification design
        // §2/§3a): kelvin_ambient + k_temp_to_kelvin owned by config
        // [physics.temperature_scale], assigned here by physics_runner.
        .def_readwrite("kelvin_ambient", &Raycaster::kelvin_ambient)
        .def_readwrite("k_temp_to_kelvin", &Raycaster::k_temp_to_kelvin)
        .def_readwrite("T_emit_gate", &Raycaster::T_emit_gate)
        // P-F1a / v7 rule 4: RADIATION_RANGE — the emission ray's reach, in
        // tiles. A STABILITY-CLASS CONSTANT, not a feel dial: it must be >= the
        // grid diagonal of the largest shipping level (128x256 => 286.22, so the
        // floor is 287) or "genuinely escapes" stops meaning "left the world"
        // and the corridor leak reopens. `range_base`/`range_per_intensity` no
        // longer bound an emission ray — they are render/legacy duty and D3's
        // damage_range guard.
        .def_readonly_static("RADIATION_RANGE_MIN", &Raycaster::RADIATION_RANGE_MIN)
        .def_readwrite("radiation_range", &Raycaster::radiation_range)
        .def("bake_emissive_table", &Raycaster::bake_emissive_table,
             "P-R4: (re)bake the black-body E° table from the current "
             "rad_scale. Idempotent; the cast entry points bake lazily too.")
        .def("emissive_table", [](const Raycaster& self) {
                const int64_t* t = self.emissive_table();
                return py::array_t<int64_t>(E_TABLE_SIZE, t);
             },
             "P-F1a: a COPY of the baked E° table (E_TABLE_SIZE INT64 entries, "
             "4 game-units per bucket) — the oracle for the bake's tests. The "
             "table widened from int32 at P-F1a (L2-B3): its old INT32_MAX "
             "saturation above T_game ~ 1768 was a silent ceiling on the law.")
        // Per-channel Beer-Lambert absorption (R,G,B) — exposed as a 3-tuple.
        .def_property("smoke_absorption_rgb",
            [](const Raycaster& r) {
                return py::make_tuple(r.smoke_absorption_rgb[0],
                                      r.smoke_absorption_rgb[1],
                                      r.smoke_absorption_rgb[2]);
            },
            [](Raycaster& r, const std::array<float, 3>& c) {
                r.smoke_absorption_rgb[0] = c[0];
                r.smoke_absorption_rgb[1] = c[1];
                r.smoke_absorption_rgb[2] = c[2];
            })
        // Per-channel additive scatter/glow albedo (R,G,B) — 3-tuple.
        .def_property("smoke_scatter_albedo",
            [](const Raycaster& r) {
                return py::make_tuple(r.smoke_scatter_albedo[0],
                                      r.smoke_scatter_albedo[1],
                                      r.smoke_scatter_albedo[2]);
            },
            [](Raycaster& r, const std::array<float, 3>& c) {
                r.smoke_scatter_albedo[0] = c[0];
                r.smoke_scatter_albedo[1] = c[1];
                r.smoke_scatter_albedo[2] = c[2];
            })
        // (P-R1, 2026-07-31: update_from_fire + coarse_cluster deleted here —
        // no production caller; see raycaster.h's tombstone comment and
        // docs/radiation_raycaster_extinction_ruling_2026-07-31.md A4.2.)
        .def("cast_source_directional",
             [](const Raycaster& self,
                const LightSource& src,
                py::array_t<float> light_rgb,
                py::array_t<float> light_dx,
                py::array_t<float> light_dy,
                py::array_t<float> gas,
                py::array_t<float> gas_absorption,
                py::array_t<float> gas_scatter,
                py::array_t<float> light_atten,
                py::object heat,
                py::object smoke_glow,
                py::object heat_atten) {
            auto [lrgb, h, w]  = get_3d(light_rgb);
            auto [ldx, h2, w2] = get_2d(light_dx);
            auto [ldy, h3, w3] = get_2d(light_dy);
            // Multi-gas density fields (engine/05 §6.2): contiguous (n_gases,h,w).
            // Each gas[g] is a (h,w) plane; the march sums them density-weighted
            // with the per-gas absorption/scatter table rows below.
            auto gv = gas.unchecked<3>();
            const float* gas_field = gv.data(0, 0, 0);
            const int n_gases = static_cast<int>(gv.shape(0));
            // Per-gas per-channel tables, shape (n_gases, 3) contiguous.
            auto ga = gas_absorption.unchecked<2>();
            const float* gabs = ga.data(0, 0);
            auto gs = gas_scatter.unchecked<2>();
            const float* gsca = gs.data(0, 0);
            // Per-tile static material attenuation, shape (h, w, 3) — same
            // interleaved layout as light_rgb. Replaces the binary is_wall:
            // occlusion is now per-channel (opaque [1,1,1] == old wall stop).
            auto a = light_atten.unchecked<3>();
            const float* atten = a.data(0, 0, 0);
            // Slice-4 optional outputs. `heat` is Q16.16 int32 (h, w); the
            // sim/headless owns it. `smoke_glow` is f32 RGB (h, w, 3) — the
            // render-only god-ray buffer. Both default to None (skip the
            // deposit) so render-only callers can pass only what they need.
            // Hold the arrays alive for the duration of the cast.
            int32_t* heat_ptr = nullptr;
            py::array_t<int32_t> heat_arr;
            if (!heat.is_none()) {
                heat_arr = heat.cast<py::array_t<int32_t>>();
                auto ha = heat_arr.mutable_unchecked<2>();
                heat_ptr = ha.mutable_data(0, 0);
            }
            float* glow_ptr = nullptr;
            py::array_t<float> glow_arr;
            if (!smoke_glow.is_none()) {
                glow_arr = smoke_glow.cast<py::array_t<float>>();
                auto ga = glow_arr.mutable_unchecked<3>();
                glow_ptr = ga.mutable_data(0, 0, 0);
            }
            // Per-tile heat attenuation (h, w) f32 — the heat analogue of
            // light_atten (engine/06 §1). Optional: None -> nullptr -> heat is
            // NOT attenuated (heat survival stays 1.0 the whole march, the
            // pre-S6 behaviour). The const pointer is held alive by heat_atten_arr.
            const float* hatten = nullptr;
            py::array_t<float> heat_atten_arr;
            if (!heat_atten.is_none()) {
                heat_atten_arr = heat_atten.cast<py::array_t<float>>();
                auto haa = heat_atten_arr.unchecked<2>();
                hatten = haa.data(0, 0);
            }
            self.cast_source_directional(src, lrgb, ldx, ldy,
                                         heat_ptr, glow_ptr,
                                         gas_field, gabs, gsca, n_gases,
                                         atten, hatten, h, w);
        }, py::arg("source"), py::arg("light_rgb"),
           py::arg("light_dx"), py::arg("light_dy"),
           py::arg("gas"), py::arg("gas_absorption"), py::arg("gas_scatter"),
           py::arg("light_atten"),
           py::arg("heat") = py::none(),
           py::arg("smoke_glow") = py::none(),
           py::arg("heat_atten") = py::none())
        // P-R1 (docs/radiation_raycaster_extinction_ruling_2026-07-31.md
        // A4.1-A4.2): the whole-fire-plane cast. Replaces the old Python
        // per-tile bp.LightSource() loop in PhysicsRunner.cast_fire_heat
        // (~10 pybind attribute writes PER BURNING TILE, PER TICK, ~6000/tick
        // at 600 fires) with ONE call: enumerates fire>0 row-major in C++
        // (Raycaster::build_fire_sources) and casts each source immediately,
        // same as the old loop's cast_source_directional calls. `heat` is
        // byte-identical to that old loop — mechanical relocation, no
        // march/law change. Field-for-field identical march-input unpacking
        // to cast_source_directional above; `fire` is the extra Q16.16 int32
        // plane input, and the per-source params are dial SCALARS instead of
        // one LightSource per call.
        .def("cast_from_fire_plane",
             [](const Raycaster& self,
                py::array_t<int32_t> fire,
                int fire_ray_count,
                double range_base, double range_per_intensity,
                double intensity_base, double intensity_per_intensity,
                std::array<float, 3> color,
                // P-F1a: OPTIONAL (None -> the short visible-light second cast
                // is skipped entirely). See the CUDA twin above.
                py::object light_rgb,
                py::object light_dx,
                py::object light_dy,
                py::array_t<float> gas,
                py::array_t<float> gas_absorption,
                py::array_t<float> gas_scatter,
                py::array_t<float> light_atten,
                py::array_t<float> heat_atten,
                py::array_t<int32_t> temperature,
                py::array_t<int32_t> heat_inv_shift,
                py::array_t<bool> thermal_solid,
                py::array_t<int32_t> rad_net,
                py::array_t<int32_t> rad_amb,
                py::array_t<int32_t> rad_flux,
                int tick,
                py::object smoke_glow,
                double jitter) {
            auto [fp, h, w] = get_2d_const(fire);
            float* lrgb = nullptr; float* ldx = nullptr; float* ldy = nullptr;
            py::array_t<float> lrgb_a, ldx_a, ldy_a;
            if (!light_rgb.is_none()) {
                lrgb_a = light_rgb.cast<py::array_t<float>>();
                ldx_a  = light_dx.cast<py::array_t<float>>();
                ldy_a  = light_dy.cast<py::array_t<float>>();
                auto lr = lrgb_a.mutable_unchecked<3>();
                lrgb = lr.mutable_data(0, 0, 0);
                auto lx = ldx_a.mutable_unchecked<2>();
                ldx = lx.mutable_data(0, 0);
                auto ly = ldy_a.mutable_unchecked<2>();
                ldy = ly.mutable_data(0, 0);
            }
            auto gv = gas.unchecked<3>();
            const float* gas_field = gv.data(0, 0, 0);
            const int n_gases = static_cast<int>(gv.shape(0));
            auto ga = gas_absorption.unchecked<2>();
            const float* gabs = ga.data(0, 0);
            auto gs = gas_scatter.unchecked<2>();
            const float* gsca = gs.data(0, 0);
            auto a = light_atten.unchecked<3>();
            const float* atten = a.data(0, 0, 0);
            float* glow_ptr = nullptr;
            py::array_t<float> glow_arr;
            if (!smoke_glow.is_none()) {
                glow_arr = smoke_glow.cast<py::array_t<float>>();
                auto gga = glow_arr.mutable_unchecked<3>();
                glow_ptr = gga.mutable_data(0, 0, 0);
            }
            // P-R4: heat_atten is now REQUIRED (it IS a_x — the emissivity AND
            // the absorptivity, Kirchhoff), as are the three radiation planes
            // and the signed accumulator. `heat` is GONE from this entry point:
            // the fire has no one-way deposit any more.
            auto [hatten, h5, w5] = get_2d_const(heat_atten);
            auto [tmp, h6, w6]    = get_2d_const(temperature);
            auto [his, h7, w7]    = get_2d_const(heat_inv_shift);
            auto [tsol, h8, w8]   = get_2d_const(thermal_solid);
            auto [rnet, h9, w9]   = get_2d(rad_net);
            auto [ramb, h11, w11] = get_2d(rad_amb);
            auto [rflux, h10, w10] = get_2d(rad_flux);
            return self.cast_from_fire_plane(fp, h, w,
                                       fire_ray_count,
                                       range_base, range_per_intensity,
                                       intensity_base, intensity_per_intensity,
                                       color.data(),
                                       lrgb, ldx, ldy, glow_ptr,
                                       gas_field, gabs, gsca, n_gases,
                                       atten, hatten,
                                       tmp, his, tsol, rnet, ramb, rflux, tick,
                                       jitter);
        }, py::arg("fire"),
           py::arg("fire_ray_count"),
           py::arg("range_base"), py::arg("range_per_intensity"),
           py::arg("intensity_base"), py::arg("intensity_per_intensity"),
           py::arg("color"),
           py::arg("light_rgb"), py::arg("light_dx"), py::arg("light_dy"),
           py::arg("gas"), py::arg("gas_absorption"), py::arg("gas_scatter"),
           py::arg("light_atten"), py::arg("heat_atten"),
           py::arg("temperature"), py::arg("heat_inv_shift"),
           py::arg("thermal_solid"), py::arg("rad_net"),
           py::arg("rad_amb"), py::arg("rad_flux"), py::arg("tick"),
           py::arg("smoke_glow") = py::none(),
           py::arg("jitter") = 0.0,
           "P-F1a: enumerate the emitter set (burning tiles + thermal solids at "
           "or above T_emit_gate) row-major, build the once-per-tick emitter "
           "mask, and run the VERIFIED RADIATION BOOKS (v6.1 rules 1/3/4 as "
           "amended by v7/v7.1) as a PURE-RADIATION cast at RADIATION_RANGE, "
           "plus a second SHORT visible-light cast on the legacy range formula. "
           "rad_net is the signed tile ledger, rad_amb the per-tile SKY ledger; "
           "sum(rad_net) + sum(rad_amb) == 0 exactly, pre-fold.")
        .def_static("normalize_directions",
             [](py::array_t<float> light_dx, py::array_t<float> light_dy) {
            auto [ldx, h, w]   = get_2d(light_dx);
            auto [ldy, h2, w2] = get_2d(light_dy);
            Raycaster::normalize_directions(ldx, ldy, h, w);
        }, py::arg("light_dx"), py::arg("light_dy"));

    // --- EOSSolver (EOS refactor P3 — the compressible Kwatra solver) --------
    // Tunables bound from [physics.eos] config (physics_runner.py); step() is
    // NOT bound standalone (PhysicsEngine.run_substeps is the only call site —
    // it owns the p_prev/gas/temperature threading). Exposed so Python can set
    // c_max/S/dx/etc AND read the six digest checkpoints + energy_floor_hits
    // for the P3 determinism/telemetry gates.
    py::class_<EOSSolver>(m, "EOSSolver")
        .def(py::init<>())
        .def_readwrite("c_max",             &EOSSolver::c_max)
        .def_readwrite("dx",                &EOSSolver::dx)
        .def_readwrite("S",                 &EOSSolver::S)
        // v2.2 D-B: the multigrid schedule (frozen at the MG measurement gate)
        .def_readwrite("use_multigrid",     &EOSSolver::use_multigrid)
        .def_readwrite("mg_nu1",            &EOSSolver::mg_nu1)
        .def_readwrite("mg_nu2",            &EOSSolver::mg_nu2)
        .def_readwrite("mg_cycles",         &EOSSolver::mg_cycles)
        .def_readwrite("mg_coarsest_sweeps", &EOSSolver::mg_coarsest_sweeps)
        .def_readwrite("mg_min_dim",        &EOSSolver::mg_min_dim)
        .def_readwrite("debug_pstar_from_prev", &EOSSolver::debug_pstar_from_prev)
        .def_readwrite("N_SUB_MAX",         &EOSSolver::N_SUB_MAX)
        .def_readwrite("CFL_ADV",           &EOSSolver::CFL_ADV)
        .def_readwrite("N_FLOOR_SOLVER",    &EOSSolver::N_FLOOR_SOLVER)
        .def_readwrite("T_AMB_K",           &EOSSolver::T_AMB_K)
        .def_readwrite("C",                 &EOSSolver::C)
        .def_readwrite("S_EOS",             &EOSSolver::S_EOS)
        .def_readwrite("adiabatic_index",   &EOSSolver::adiabatic_index)
        .def_readwrite("absorb_strength",   &EOSSolver::absorb_strength)
        .def_readwrite("T_MIN",             &EOSSolver::T_MIN)
        .def_readwrite("T_WORK_CLAMP",      &EOSSolver::T_WORK_CLAMP)
        // P-E2b (design §2.4): trust-gate dial, PLUMBING ONLY — the fade
        // mechanism is P-E4's. Provably inert (nothing reads this member).
        .def_readwrite("n_work_ref",        &EOSSolver::n_work_ref)
        // P-E3 (energy-books arc, design §2.8): interior momentum drag with a
        // heat counterparty. k_drag default 0.0 -> dormant (branch on the
        // QUANTIZED kd_q, not this float); k_drag_heat_frac default 1.0
        // (RULING R2) keeps the conservation oracle EXACT through every gate.
        .def_readwrite("k_drag",            &EOSSolver::k_drag)
        .def_readwrite("k_drag_heat_frac",  &EOSSolver::k_drag_heat_frac)
        // c_v: EOSSolver's own copy of the SAME [physics.thermal] c_v gas
        // heat-capacity constant TemperatureSolver::c_v prices its deposits
        // with (physics_runner.py binds both from the one config key).
        .def_readwrite("c_v",               &EOSSolver::c_v)
        .def_readwrite("T_MAX_PHYS",        &EOSSolver::T_MAX_PHYS)     // v2.4 rail
        .def_readwrite("U_MAX",             &EOSSolver::U_MAX)          // v2.4 rail
        // trace_mass_scale binding RETIRED (P-T0, design §2.6 — the member
        // itself is gone from EOSSolver; see eos_solver.h).
        .def_readonly("energy_floor_hits",  &EOSSolver::energy_floor_hits)
        .def_readonly("u_clamp_hits",       &EOSSolver::u_clamp_hits)
        .def_readonly("work_clamp_hits",    &EOSSolver::work_clamp_hits)
        .def_readonly("t_max_phys_hits",    &EOSSolver::t_max_phys_hits) // v2.4
        .def_readonly("u_max_hits",         &EOSSolver::u_max_hits)      // v2.4
        // P-E0 (energy-books §2.5): per-tick energy-bracket deltas over
        // Σ n_bulk·T (transport substeps / step-4c). Digest-inert telemetry.
        .def_readonly("eth_transport_delta",   &EOSSolver::eth_transport_delta)
        .def_readonly("eth_compression_delta", &EOSSolver::eth_compression_delta)
        // P-E1 (energy-books design §2.1.5/§2.5): the transport law's one-way
        // guard terms + the active-flux telemetry §7's bound is scaled by.
        // Read-only, per-tick, int64 raw Q16.16² (dequant = raw / 65536²)
        // except the two counts.
        .def_readonly("e_ts_residual",         &EOSSolver::e_ts_residual)
        .def_readonly("e_wipe_sum",            &EOSSolver::e_wipe_sum)
        .def_readonly("e_floor_sum",           &EOSSolver::e_floor_sum)
        .def_readonly("n_active_flux",         &EOSSolver::n_active_flux)
        .def_readonly("n_bulk_active_sum",     &EOSSolver::n_bulk_active_sum)
        // P-E3 (design §2.8): the interior-drag oracle, PER-TICK, both
        // n-weighted, raw Q16.16^2 (the SAME "N*T" currency as the P-E1 five
        // above). Identity: ke_drag_removed == 2*c_v*(e_drag_deposit +
        // e_drag_drop_sum + e_drag_rail_clipped) within a small LSB slack.
        .def_readonly("ke_drag_removed",       &EOSSolver::ke_drag_removed)
        .def_readonly("e_drag_deposit",        &EOSSolver::e_drag_deposit)
        .def_readonly("e_drag_drop_sum",       &EOSSolver::e_drag_drop_sum)
        .def_readonly("e_drag_rail_clipped",   &EOSSolver::e_drag_rail_clipped)
        // BC (spec §5): the boundary_flux rail — per-conservative-plane int64
        // Σ(N_pre_reset − N_amb). Returned as a Python list (empty on space
        // maps). NOT folded into any digest (absence-transparent, zero golden
        // re-baseline).
        .def("boundary_flux", [](const EOSSolver& s) {
            py::list out;
            for (int64_t v : s.boundary_flux()) out.append(v);
            return out;
        })
        .def_readonly("digest_advect",      &EOSSolver::digest_advect)
        .def_readonly("digest_bulk_flux",   &EOSSolver::digest_bulk_flux)
        .def_readonly("digest_pstar",       &EOSSolver::digest_pstar)
        .def_readonly("digest_helmholtz",   &EOSSolver::digest_helmholtz)
        .def_readonly("digest_velocity",    &EOSSolver::digest_velocity)
        .def_readonly("digest_compression", &EOSSolver::digest_compression)
        // DEBUG probe (temporary, eos-p3fix-thermal-ceiling investigation).
        .def_readwrite("dbg_probe_idx",          &EOSSolver::dbg_probe_idx)
        .def_readonly("dbg_T_pre_advect",        &EOSSolver::dbg_T_pre_advect)
        .def_readonly("dbg_T_post_advect",       &EOSSolver::dbg_T_post_advect)
        .def_readonly("dbg_T_post_compression",  &EOSSolver::dbg_T_post_compression)
        // EOS P6.2: the substep count the last step() ran (gate telemetry —
        // lets the per-kernel digest gates replay the isolated advection on
        // the exact schedule the solver derived).
        .def_readonly("dbg_last_n_sub",          &EOSSolver::dbg_last_n_sub)
        // EOS P6.4: the c_LOCAL velocity cap the last step() derived (q16 raw;
        // gate telemetry — lets the P6.4 digest gate feed the isolated
        // kick+compression replay the exact per-tick cap, which is computed
        // from the PRE-advection T scan the replay cannot see).
        .def_readonly("dbg_last_c_local_q",      &EOSSolver::dbg_last_c_local_q)
        // EOS P6.3 gate telemetry: 1-D int32 copies of the last step()'s
        // solve-input caches (pstar, div_u, n_total — nothing after the
        // solve writes them; reshape to (h, w) in Python). Together with the
        // engine-visible p_prev these reconstruct the EXACT inputs the
        // pressure solve consumed, so the digest gate can replay it
        // isolated (eos_mg_solve_ref / cuda_eos_mg_solve).
        .def("dbg_mg_inputs", [](const EOSSolver& s) {
            auto mk = [](const std::vector<int32_t>& v) {
                py::array_t<int32_t> a((py::ssize_t)v.size());
                std::copy(v.begin(), v.end(), a.mutable_data());
                return a;
            };
            return py::make_tuple(mk(s.dbg_pstar_cache()),
                                  mk(s.dbg_div_u_cache()),
                                  mk(s.dbg_n_total_cache()));
        }, "P6.3: (pstar, div_u, n_total) flat int32 copies as consumed by "
           "the last step()'s pressure solve.");

    // EOS P6.2: the standalone CPU reference for the fused SL-advection
    // substep chain (eos_solver.cpp eos_sl_advect_reference — the SAME
    // file-local backtrace routine EOSSolver::step calls). Runs IN PLACE on
    // wind_x/wind_y/temperature and returns the chained FNV digest, ==
    // EOSSolver.digest_advect when fed step-1-entry state + dbg_last_n_sub.
    // Test entry only (both CPU and CUDA builds) — the live path is
    // EOSSolver::step inside PhysicsEngine::run_substeps.
    m.def("eos_sl_advect_ref",
          [](py::array_t<int32_t> wind_x, py::array_t<int32_t> wind_y,
             py::array_t<int32_t> temperature,
             py::array_t<bool> solid, py::array_t<bool> is_vacuum,
             py::array_t<float> dyn_permeability,
             float dt, int n_sub,
             py::object thermal_solid) -> uint64_t {   // THERMAL-MASS AXIS
              auto [wx, h, w]    = get_2d(wind_x);
              auto [wy, h2, w2]  = get_2d(wind_y);
              auto [t, h3, w3]   = get_2d(temperature);
              auto [sol, h4, w4] = get_2d_const(solid);
              auto [vac, h5, w5] = get_2d_const(is_vacuum);
              auto [pm, h6, w6]  = get_2d_const(dyn_permeability);
              const bool* tsol = nullptr;
              py::array_t<bool> tsol_arr;
              if (!thermal_solid.is_none()) {
                  tsol_arr = thermal_solid.cast<py::array_t<bool>>();
                  auto ta = tsol_arr.unchecked<2>();
                  tsol = ta.data(0, 0);
              }
              return eos_sl_advect_reference(wx, wy, t, sol, vac, pm,
                                             h, w, dt, n_sub, nullptr, tsol);
          },
          py::arg("wind_x"), py::arg("wind_y"), py::arg("temperature"),
          py::arg("solid"), py::arg("is_vacuum"), py::arg("dyn_permeability"),
          py::arg("dt"), py::arg("n_sub"),
          py::arg("thermal_solid") = py::none(),
          "P6.2 CPU reference: replay EOSSolver::step's SL-advection substep "
          "chain in place on wind_x/wind_y/temperature; returns the chained "
          "FNV digest (== EOSSolver.digest_advect for the same inputs).");

    // EOS P6.4: the standalone CPU reference for the momentum kick +
    // compression work (eos_solver.cpp eos_kick_compression_reference — the
    // step-4/4c loops copied line for line, same file-local helpers). Runs
    // IN PLACE on wind_x/wind_y/temperature and returns (digest_velocity,
    // digest_compression, u_clamp_hits, u_max_hits, work_clamp_hits,
    // energy_floor_hits, t_max_phys_hits) — the digests == EOSSolver's own
    // when fed the reconstructed step-4-entry state (post-advection u/T via
    // eos_sl_advect_ref + dbg_last_n_sub, post-tick atmosphere as p_new, the
    // post-tick gas planes, dbg_last_c_local_q); the counters are per-call
    // (the solver's members are cumulative — gates compare per-tick deltas).
    // Test entry only (both CPU and CUDA builds) — the live path is
    // EOSSolver::step inside PhysicsEngine::run_substeps.
    m.def("eos_kick_compression_ref",
          [](py::array_t<int32_t> wind_x, py::array_t<int32_t> wind_y,
             py::array_t<int32_t> temperature, py::array_t<int32_t> p_new,
             py::array_t<int32_t> gas, py::array_t<bool> gas_conservative,
             py::array_t<bool> solid, py::array_t<bool> is_vacuum,
             py::array_t<float> dyn_wave_absorb,
             float dt, int32_t c_local_q,
             float c_max, float dx, float adiabatic_index,
             float absorb_strength, float n_floor_solver, float t_min,
             float t_work_clamp, float t_max_phys, float u_max,
             // trace_mass_scale param RETIRED (P-T0, design §2.6)
             // P-E3 (design §2.8): interior drag + heat counterparty.
             float k_drag, float k_drag_heat_frac, float c_v,
             py::object thermal_solid,                  // THERMAL-MASS AXIS
             // A6: the ambient/planetside path. `is_ambient` was hard-coded
             // nullptr at the call below, so no caller could reach the
             // reference's ambient branch at all; `sponge_udamp` is the B3c
             // band restored in eos_solver.cpp. Both default None -> pre-A6.
             py::object is_ambient, py::object sponge_udamp) -> py::tuple {
              auto [wx, h, w]    = get_2d(wind_x);
              auto [wy, h2, w2]  = get_2d(wind_y);
              auto [t, h3, w3]   = get_2d(temperature);
              auto [pn, h4, w4]  = get_2d_const(p_new);
              auto gv = gas.unchecked<3>();
              const int32_t* gas_ptr = gv.data(0, 0, 0);
              const int n_gases = static_cast<int>(gv.shape(0));
              auto gc = gas_conservative.unchecked<1>();
              const bool* gcons = gc.data(0);
              auto [sol, h5, w5] = get_2d_const(solid);
              auto [vac, h6, w6] = get_2d_const(is_vacuum);
              auto [ab, h7, w7]  = get_2d_const(dyn_wave_absorb);
              const bool* tsol = nullptr;
              py::array_t<bool> tsol_arr;
              if (!thermal_solid.is_none()) {
                  tsol_arr = thermal_solid.cast<py::array_t<bool>>();
                  auto ta = tsol_arr.unchecked<2>();
                  tsol = ta.data(0, 0);
              }
              const bool* amb = nullptr;
              py::array_t<bool> amb_arr;
              if (!is_ambient.is_none()) {
                  amb_arr = is_ambient.cast<py::array_t<bool>>();
                  auto aa = amb_arr.unchecked<2>();
                  amb = aa.data(0, 0);
              }
              const int32_t* sud = nullptr;
              py::array_t<int32_t> sud_arr;
              if (!sponge_udamp.is_none()) {
                  sud_arr = sponge_udamp.cast<py::array_t<int32_t>>();
                  auto sa = sud_arr.unchecked<2>();
                  sud = sa.data(0, 0);
              }
              uint64_t dig_vel = 0, dig_comp = 0;
              int64_t cnts[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
              eos_kick_compression_reference(
                  wx, wy, t, pn, gas_ptr, gcons, n_gases, sol, vac, ab,
                  h, w, dt, c_local_q,
                  c_max, dx, adiabatic_index, absorb_strength,
                  n_floor_solver, t_min, t_work_clamp, t_max_phys, u_max,
                  k_drag, k_drag_heat_frac, c_v,
                  &dig_vel, &dig_comp, cnts, amb, tsol, sud);   // trace_mass_scale arg RETIRED
              return py::make_tuple(dig_vel, dig_comp, cnts[0], cnts[1],
                                    cnts[2], cnts[3], cnts[4], cnts[5],
                                    cnts[6], cnts[7], cnts[8]);
          },
          py::arg("wind_x"), py::arg("wind_y"), py::arg("temperature"),
          py::arg("p_new"), py::arg("gas"), py::arg("gas_conservative"),
          py::arg("solid"), py::arg("is_vacuum"), py::arg("dyn_wave_absorb"),
          py::arg("dt"), py::arg("c_local_q"),
          py::arg("c_max"), py::arg("dx"), py::arg("adiabatic_index"),
          py::arg("absorb_strength"), py::arg("n_floor_solver"),
          py::arg("t_min"), py::arg("t_work_clamp"), py::arg("t_max_phys"),
          py::arg("u_max"),
          py::arg("k_drag") = 0.0f, py::arg("k_drag_heat_frac") = 1.0f,
          py::arg("c_v") = 1.0f,
          py::arg("thermal_solid") = py::none(),
          py::arg("is_ambient") = py::none(),
          py::arg("sponge_udamp") = py::none(),
          "P6.4 CPU reference: replay EOSSolver::step's kick + compression-"
          "work tail in place on wind_x/wind_y/temperature; returns "
          "(digest_velocity, digest_compression, u_clamp_hits, u_max_hits, "
          "work_clamp_hits, energy_floor_hits, t_max_phys_hits, "
          "ke_drag_removed, e_drag_deposit, e_drag_drop_sum, "
          "e_drag_rail_clipped) for this call.");

    // EOS P6.3: the standalone CPU reference for the multigrid pressure
    // solve (eos_solver.cpp eos_mg_solve_reference — drives the SAME
    // mg_build_levels + mg_run_solve_cpu the live step() calls). Writes the
    // solved P into p_out and returns the FNV digest, ==
    // EOSSolver.digest_helmholtz when fed the solve inputs of a real tick
    // (dbg_mg_inputs + the engine's p_prev). Takes the solver instance for
    // the config surface (dx/c_max/gamma/N_FLOOR_SOLVER + the frozen MG
    // schedule). Test entry only (both CPU and CUDA builds) — the live path
    // remains EOSSolver::step inside PhysicsEngine::run_substeps.
    m.def("eos_mg_solve_ref",
          [](const EOSSolver& solver,
             py::array_t<int32_t> pstar, py::array_t<int32_t> div_u,
             py::array_t<int32_t> n_total, py::array_t<int32_t> p_prev,
             py::array_t<bool> solid, py::array_t<bool> is_vacuum,
             py::array_t<float> dyn_permeability,
             float dt, py::array_t<int32_t> p_out) -> uint64_t {
              auto [ps, h, w]    = get_2d_const(pstar);
              auto [dv, h2, w2]  = get_2d_const(div_u);
              auto [nt, h3, w3]  = get_2d_const(n_total);
              auto [pp, h4, w4]  = get_2d_const(p_prev);
              auto [sol, h5, w5] = get_2d_const(solid);
              auto [vac, h6, w6] = get_2d_const(is_vacuum);
              auto [pm, h7, w7]  = get_2d_const(dyn_permeability);
              auto [po, h8, w8]  = get_2d(p_out);
              return eos_mg_solve_reference(solver, ps, dv, nt, pp,
                                            sol, vac, pm, h, w, dt, po);
          },
          py::arg("solver"), py::arg("pstar"), py::arg("div_u"),
          py::arg("n_total"), py::arg("p_prev"),
          py::arg("solid"), py::arg("is_vacuum"), py::arg("dyn_permeability"),
          py::arg("dt"), py::arg("p_out"),
          "P6.3 CPU reference: replay EOSSolver::step's pressure solve on "
          "given solve inputs; writes the solved P into p_out and returns "
          "the FNV digest (== EOSSolver.digest_helmholtz for the same "
          "inputs).");

    // --- CombustionSolver (EOS refactor P4 — combustion on real O2, design
    //     §5). Own pass, run once per tick AFTER eos.step materializes P. ---
    py::class_<CombustionSolver>(m, "CombustionSolver")
        .def(py::init<>())
        .def_readwrite("burn_rate",         &CombustionSolver::burn_rate)
        .def_readwrite("o2_frac_ext",       &CombustionSolver::o2_frac_ext)
        // FULL-RESPONSE REFERENCE SPLIT — twin of FireParams::o2_frac_full.
        .def_readwrite("o2_frac_full",      &CombustionSolver::o2_frac_full)
        .def_readwrite("o2_frac_amb",       &CombustionSolver::o2_frac_amb)
        .def_readwrite("o2_thresh_burn",    &CombustionSolver::o2_thresh_burn)
        .def_readwrite("H_fuel",            &CombustionSolver::H_fuel)
        .def_readwrite("soot_yield",        &CombustionSolver::soot_yield)
        .def_readwrite("fuel_per_o2",       &CombustionSolver::fuel_per_o2)   // v2.5 P5.1
        // P-R4: the FUEL-BED deposit's split constant (H_bed = M * 2^SHIFT).
        .def_readwrite("H_BED_M",           &CombustionSolver::H_BED_M)
        .def_readwrite("H_BED_SHIFT",       &CombustionSolver::H_BED_SHIFT)
        .def_readwrite("o2_thresh_breathe", &CombustionSolver::o2_thresh_breathe)
        .def_readwrite("T_MAX_PHYS",        &CombustionSolver::T_MAX_PHYS)     // v2.4 rail
        .def_readonly("heat_floor_hits",    &CombustionSolver::heat_floor_hits)
        .def_readonly("t_max_phys_hits",    &CombustionSolver::t_max_phys_hits) // v2.4
        // P-E2b (design §2.2/§2.5): the energy-sum twin of heat_floor_hits.
        .def_readonly("e_deposit_drop_sum", &CombustionSolver::e_deposit_drop_sum)
        .def("step", [](const CombustionSolver& self,
                        py::array_t<int32_t> gas,             // (n_gases,h,w) Q16.16
                        int o2_idx, int inert_n2_idx, int black_smoke_idx,
                        py::array_t<int32_t> temperature,     // Q16.16, mutated
                        py::array_t<int32_t> wall_hp,         // Q16.16, MUTATED (v2.5
                                                              // P5.1: ember-scale fuel
                                                              // depletion, 1-LSB floor)
                        py::array_t<int32_t> fire,            // Q16.16, read-only
                        py::array_t<bool> flammable,
                        py::array_t<bool> solid,
                        py::array_t<bool> is_vacuum,
                        py::array_t<int32_t> ignition_temp_q16,   // Q16.16, read-only
                        float dt, float c_v, float n_floor_heat,
                        // THERMAL-MASS AXIS, P-EOS (ruling §2 site 3): on a
                        // thermal_solid burn site the aggregate deposit converts
                        // via heat_inv_shift (the OBJECT path). Both OPTIONAL —
                        // None -> nullptr -> every site takes the gas path, so
                        // every shipped direct caller keeps its exact meaning.
                        py::object thermal_solid,
                        py::object heat_inv_shift,
                        // P-R4: the `heat[]` plane the fuel-bed deposit lands
                        // in. OPTIONAL (None -> no H_bed == pre-P-R4).
                        py::object heat,
                        // D1: the (max_claimants,h,w) error-feedback demand
                        // accumulator. OPTIONAL (None -> the pre-D1 chained
                        // truncation).
                        py::object dem_acc,
                        // P-O2b (design v5.2 "F-O2b"): the EXTENDED OXYGEN
                        // DRAW. draw_r == 1 (the default) is the shipped 4-face
                        // law BIT FOR BIT, so every direct caller that does not
                        // opt in is unmoved. `dyn_permeability` is the float
                        // plane the path weight multiplies through (quantized
                        // once at pass entry); None -> permeability 1.0.
                        int draw_r,
                        py::object dyn_permeability,
                        int max_claimants) {
            auto gv = gas.mutable_unchecked<3>();
            int32_t* gas_ptr = gv.mutable_data(0, 0, 0);
            const int n_gases = static_cast<int>(gv.shape(0));
            const int h = static_cast<int>(gv.shape(1));
            const int w = static_cast<int>(gv.shape(2));
            auto [temp, h2, w2] = get_2d(temperature);
            auto [whp, h3, w3]  = get_2d(wall_hp);   // v2.5 P5.1: mutable
            auto [f, h4, w4]    = get_2d_const(fire);
            auto [fl, h5, w5]   = get_2d_const(flammable);
            auto [sol, h6, w6]  = get_2d_const(solid);
            auto [vac, h7, w7]  = get_2d_const(is_vacuum);
            auto [ign, h8, w8]  = get_2d_const(ignition_temp_q16);
            const bool* tsol = nullptr;
            py::array_t<bool> tsol_arr;
            if (!thermal_solid.is_none()) {
                tsol_arr = thermal_solid.cast<py::array_t<bool>>();
                auto ta = tsol_arr.unchecked<2>();
                tsol = ta.data(0, 0);
            }
            const int32_t* hshift = nullptr;
            py::array_t<int32_t> hshift_arr;
            if (!heat_inv_shift.is_none()) {
                hshift_arr = heat_inv_shift.cast<py::array_t<int32_t>>();
                auto ha = hshift_arr.unchecked<2>();
                hshift = ha.data(0, 0);
            }
            int32_t* heat_ptr = nullptr;
            py::array_t<int32_t> heat_arr;
            if (!heat.is_none()) {
                heat_arr = heat.cast<py::array_t<int32_t>>();
                auto hh = heat_arr.mutable_unchecked<2>();
                heat_ptr = hh.mutable_data(0, 0);
            }
            int32_t* dacc_ptr = nullptr;
            py::array_t<int32_t> dacc_arr;
            if (!dem_acc.is_none()) {
                dacc_arr = dem_acc.cast<py::array_t<int32_t>>();
                auto da = dacc_arr.mutable_unchecked<3>();   // (max_claimants, h, w)
                dacc_ptr = da.mutable_data(0, 0, 0);
            }
            const float* perm_ptr = nullptr;
            py::array_t<float> perm_arr;
            if (!dyn_permeability.is_none()) {
                perm_arr = dyn_permeability.cast<py::array_t<float>>();
                auto pa = perm_arr.unchecked<2>();
                perm_ptr = pa.data(0, 0);
            }
            self.step(gas_ptr, n_gases, o2_idx, inert_n2_idx, black_smoke_idx,
                     temp, whp, f, fl, sol, vac, ign, h, w, dt, c_v, n_floor_heat,
                     tsol, hshift, heat_ptr, dacc_ptr,
                     draw_r, perm_ptr, max_claimants);
        }, py::arg("gas"), py::arg("o2_idx"), py::arg("inert_n2_idx"),
           py::arg("black_smoke_idx"), py::arg("temperature"), py::arg("wall_hp"),
           py::arg("fire"), py::arg("flammable"), py::arg("solid"),
           py::arg("is_vacuum"), py::arg("ignition_temp_q16"),
           py::arg("dt"), py::arg("c_v"), py::arg("n_floor_heat"),
           py::arg("thermal_solid") = py::none(),
           py::arg("heat_inv_shift") = py::none(),
           py::arg("heat") = py::none(),
           py::arg("dem_acc") = py::none(),
           py::arg("draw_r") = 1,
           py::arg("dyn_permeability") = py::none(),
           py::arg("max_claimants") = 4);

    // --- WaterSolver (pipe model: damped velocity + donor-cell upwind flux;
    //     engine/07 §2, water_implementation_plan Step W1) ---
    py::class_<WaterSolver>(m, "WaterSolver")
        .def(py::init<>())
        .def_readwrite("g",         &WaterSolver::g)
        .def_readwrite("damping",   &WaterSolver::damping)
        .def_readwrite("dx",        &WaterSolver::dx)
        .def_readwrite("k_p",       &WaterSolver::k_p)
        .def_readwrite("v_max",     &WaterSolver::v_max)
        .def_readwrite("depth_eps", &WaterSolver::depth_eps)
        .def_readwrite("h_ref",     &WaterSolver::h_ref)
        .def_readwrite("gamma_r",   &WaterSolver::gamma_r)
        .def_readwrite("h_cap",     &WaterSolver::h_cap)
        .def_readwrite("k_amp",     &WaterSolver::k_amp)
        .def_readwrite("k_splash",  &WaterSolver::k_splash)
        .def("max_dt", &WaterSolver::max_dt)
        .def("max_dt_q", &WaterSolver::max_dt_q)   // S1: Q16.16 CFL constant
        .def("ripple_max_dt", &WaterSolver::ripple_max_dt)
        .def("step", [](const WaterSolver& self,
                        py::array_t<int32_t> water_depth,   // S1: Q16.16 int32
                        py::array_t<int32_t> flow_vx,       // S1: Q16.16 int32
                        py::array_t<int32_t> flow_vy,       // S1: Q16.16 int32
                        py::object floor_height,            // Q16.16 int32 (nullable)
                        py::object atmosphere,              // EOS P3: Q16.16 int32 == P (nullable)
                        py::array_t<bool> solid,
                        float dt, float tilt_x, float tilt_y) {
            auto [wd, h, w]    = get_2d(water_depth);
            auto [vx, h2, w2]  = get_2d(flow_vx);
            auto [vy, h3, w3]  = get_2d(flow_vy);
            auto [sol, h4, w4] = get_2d_const(solid);
            // Nullable fields (cast_source_directional precedent): None ->
            // nullptr, else cast to an array kept alive in this scope.
            // floor_height None -> flat zero (Q16.16 int32); atmosphere None
            // -> no head term (and with k_p == 0 it is never read). EOS P3:
            // the wave_p head-term FLOAT BRIDGE arg is RETIRED (see file
            // header) — atmosphere is now the integer P read directly.
            const int32_t* fl = nullptr;
            py::array_t<int32_t> fl_arr;
            if (!floor_height.is_none()) {
                fl_arr = floor_height.cast<py::array_t<int32_t>>();
                auto fa = fl_arr.unchecked<2>();
                fl = fa.data(0, 0);
            }
            const int32_t* atm = nullptr;
            py::array_t<int32_t> atm_arr;
            if (!atmosphere.is_none()) {
                atm_arr = atmosphere.cast<py::array_t<int32_t>>();
                auto aa = atm_arr.unchecked<2>();
                atm = aa.data(0, 0);
            }
            self.step(wd, vx, vy, fl, atm, sol, h, w, dt, tilt_x, tilt_y);
        }, py::arg("water_depth"), py::arg("flow_vx"), py::arg("flow_vy"),
           py::arg("floor_height") = py::none(),
           py::arg("atmosphere")   = py::none(),
           py::arg("solid"), py::arg("dt"),
           py::arg("tilt_x"), py::arg("tilt_y"))
        // W6a ripple: the VISUAL-ONLY surface wave (canon §6). water_depth /
        // atmosphere / p_prev / solid are read-only — the ripple never feeds
        // back into transport. EOS P3: the splash source is the per-tick
        // pressure transient |P - P_prev| (design §6); both nullable (None
        // -> no splash source, never read).
        .def("step_ripple", [](const WaterSolver& self,
                               py::array_t<float> ripple,
                               py::array_t<float> ripple_v,
                               py::array_t<int32_t> water_depth,   // S1: Q16.16 int32
                               py::object atmosphere,
                               py::object p_prev,
                               py::array_t<bool> solid,
                               float dt) {
            auto [r, h, w]     = get_2d(ripple);
            auto [rv, h2, w2]  = get_2d(ripple_v);
            auto [wd, h3, w3]  = get_2d_const(water_depth);
            auto [sol, h4, w4] = get_2d_const(solid);
            const int32_t* atm = nullptr;
            py::array_t<int32_t> atm_arr;
            if (!atmosphere.is_none()) {
                atm_arr = atmosphere.cast<py::array_t<int32_t>>();
                auto aa = atm_arr.unchecked<2>();
                atm = aa.data(0, 0);
            }
            const int32_t* pp = nullptr;
            py::array_t<int32_t> pp_arr;
            if (!p_prev.is_none()) {
                pp_arr = p_prev.cast<py::array_t<int32_t>>();
                auto pa = pp_arr.unchecked<2>();
                pp = pa.data(0, 0);
            }
            self.step_ripple(r, rv, wd, atm, pp, sol, h, w, dt);
        }, py::arg("ripple"), py::arg("ripple_v"), py::arg("water_depth"),
           py::arg("atmosphere") = py::none(),
           py::arg("p_prev") = py::none(),
           py::arg("solid"), py::arg("dt"));

    // --- PhysicsEngine (Patch 1 S3) — owns the solver instances ---------------
    // The scaffold for the unification: holds the six solvers and exposes each
    // as a reference (reference_internal ties the solver's lifetime to the engine
    // + returns the SAME held instance, so Python set-params + step() act on the
    // engine's solver). PhysicsRunner uses engine.<solver> instead of building
    // them itself — same objects, same calls, bit-identical. The per-tick step()
    // moves in here in S4.
    py::class_<PhysicsEngine>(m, "PhysicsEngine")
        .def(py::init<>())
        .def_property_readonly("atmos",
            [](PhysicsEngine& e) -> AtmosphereSolver& { return e.atmos; },
            py::return_value_policy::reference_internal)
        .def_property_readonly("smoke",
            [](PhysicsEngine& e) -> SmokeDynamics& { return e.smoke; },
            py::return_value_policy::reference_internal)
        .def_property_readonly("fire",
            [](PhysicsEngine& e) -> FireSimulation& { return e.fire; },
            py::return_value_policy::reference_internal)
        .def_property_readonly("temperature",
            [](PhysicsEngine& e) -> TemperatureSolver& { return e.temperature; },
            py::return_value_policy::reference_internal)
        .def_property_readonly("raycaster",
            [](PhysicsEngine& e) -> Raycaster& { return e.raycaster; },
            py::return_value_policy::reference_internal)
        .def_property_readonly("water",
            [](PhysicsEngine& e) -> WaterSolver& { return e.water; },
            py::return_value_policy::reference_internal)
        .def_property_readonly("eos",
            [](PhysicsEngine& e) -> EOSSolver& { return e.eos; },
            py::return_value_policy::reference_internal)
        .def_property_readonly("combustion",
            [](PhysicsEngine& e) -> CombustionSolver& { return e.combustion; },
            py::return_value_policy::reference_internal)
        // --- Patch 1 S4a: the per-tick TAIL ---------------------------------
        // step_tail moves the three trailing pure-solver-call steps of
        // PhysicsRunner.step (ripple, fire, temperature — after the IMEX substep
        // loop) into C++. Pointer extraction mirrors the FireSimulation /
        // TemperatureSolver / WaterSolver.step_ripple bindings above. `gmap.solid`
        // is passed once and used as BOTH the ripple `solid` and the fire/temp
        // `is_wall`/`solid` arg (the Python passes gmap.solid to all three).
        // `temperature` is the one numpy array, extracted both const (the fire
        // read) and mutable (the temperature-solver write). Returns the (y, x)
        // burn-through list exactly as FireSimulation.step does.
        .def("step_tail", [](const PhysicsEngine& self,
                             // ripple group
                             py::array_t<float> ripple,
                             py::array_t<float> ripple_v,
                             py::array_t<int32_t> water_depth,   // S1: Q16.16 int32
                             py::array_t<int32_t> wave_p,        // S2a: Q16.16 int32
                             py::array_t<bool>  solid,
                             // fire group
                             py::array_t<int32_t> fire,          // S3b: Q16.16 int32 (integer logistic)
                             py::array_t<int32_t> atmosphere,   // S2c: Q16.16 int32 (fire reads + plume-writes)
                             py::array_t<int32_t> smoke,         // S2b: Q16.16 int32
                             py::array_t<int32_t> wall_hp,       // S3b: Q16.16 int32
                             py::array_t<int32_t> temperature,
                             py::array_t<int32_t> wind_x,        // S2c: Q16.16 int32
                             py::array_t<int32_t> wind_y,        // S2c: Q16.16 int32
                             py::array_t<bool>  is_vacuum,
                             py::array_t<bool>  flammable,
                             // temperature group
                             py::array_t<int32_t> heat,
                             py::array_t<int32_t> heat_inv_shift,
                             py::array_t<int32_t> face_shift,
                             // THERMAL-MASS AXIS: the per-medium THERMAL mask
                             // (thermal_mass > 0) — REQUIRED, deliberately not
                             // nullable here, so a caller can never silently
                             // fall the engine back to the flow mask `solid`.
                             py::array_t<bool> thermal_solid,
                             // COOL-SHIFT AXIS: the per-tile ambient-decay
                             // shift (GameMap.cool_shift) — REQUIRED for the
                             // same reason as thermal_solid above: the live
                             // engine must never silently fall back to the
                             // single global COOL_SHIFT.
                             py::array_t<int32_t> cool_shift_grid,
                             // FUEL-FRACTION AXIS: the per-tile 1/hp plane
                             // (GameMap.fuel_recip) the fire logistic's fuel
                             // term reads — REQUIRED for the same reason
                             // thermal_solid/cool_shift_grid are: the live
                             // engine must never silently fall back to the
                             // single global [physics.fire] fuel_ref.
                             py::array_t<int64_t> fuel_recip,
                             // PER-MATERIAL T_ext (P-R3, ruling A3 ride-along):
                             // the per-tile Q16.16 extinction temperature the
                             // fire logistic's `hot` gate reads — REQUIRED for
                             // the same reason fuel_recip is: the live engine
                             // must never silently fall back to a global that
                             // sits above both shipped ignition temps.
                             py::array_t<int32_t> fire_T_ext_plane,
                             // EOS P3: bulk-N source (Pass-1 heat divisor)
                             py::array_t<int32_t> gas,
                             py::array_t<bool> gas_conservative,
                             int o2_idx,                         // EOS P4
                             float sim_time,
                             py::object is_ambient,                 // BC
                             py::object rad_net) -> py::list {      // P-R4
            // ripple group
            auto [rip, h, w]    = get_2d(ripple);
            auto [ripv, h2, w2] = get_2d(ripple_v);
            auto [wd, h3, w3]   = get_2d_const(water_depth);
            auto [wp, h4, w4]   = get_2d_const(wave_p);
            auto [sol, h5, w5]  = get_2d_const(solid);
            // fire group
            auto [f, h6, w6]    = get_2d(fire);
            auto [atm, h7, w7]  = get_2d(atmosphere);
            auto [sm, h8, w8]   = get_2d(smoke);
            auto [whp, h9, w9]  = get_2d(wall_hp);
            // temperature is read const by fire AND mutated by temperature.step —
            // extract a mutable pointer (one mutable_unchecked view) and use it
            // for both; aliasing is exactly the Python behaviour (same array).
            auto [temp, h10, w10] = get_2d(temperature);
            auto [wx, h11, w11] = get_2d_const(wind_x);
            auto [wy, h12, w12] = get_2d_const(wind_y);
            auto [vac, h13, w13] = get_2d_const(is_vacuum);
            auto [fl, h14, w14] = get_2d_const(flammable);
            // temperature group
            auto [hp, h15, w15]    = get_2d_const(heat);
            auto [shift, h16, w16] = get_2d_const(heat_inv_shift);
            // face_shift is (h, w, 4) int32 — mirror the TemperatureSolver binding.
            auto fa = face_shift.unchecked<3>();
            const int32_t* fs = fa.data(0, 0, 0);
            // THERMAL-MASS AXIS: the per-medium thermal mask.
            auto [tsol, h17, w17] = get_2d_const(thermal_solid);
            // COOL-SHIFT AXIS: the per-tile ambient-decay shift.
            auto [csg, h18, w18] = get_2d_const(cool_shift_grid);
            // FUEL-FRACTION AXIS: the per-tile 1/hp plane (int64 — a
            // RECIP_SHIFT=32 reciprocal does not fit int32).
            auto fr_v = fuel_recip.unchecked<2>();
            const int64_t* fr = fr_v.data(0, 0);
            // PER-MATERIAL T_ext: the per-tile Q16.16 extinction temperature.
            auto tep_v = fire_T_ext_plane.unchecked<2>();
            const int32_t* tep = tep_v.data(0, 0);
            // EOS P3: (N,h,w) gas + the conservative flags — step_tail sums
            // the bulk planes for the temperature Pass-1 N divisor.
            auto gv = gas.unchecked<3>();
            const int32_t* gas_ptr = gv.data(0, 0, 0);
            const int n_gases = static_cast<int>(gv.shape(0));
            auto gc = gas_conservative.unchecked<1>();
            const bool* gcons = gc.data(0);
            // BC nullable is_ambient (None on space maps -> nullptr).
            const bool* amb = nullptr;
            py::array_t<bool> amb_arr;
            if (!is_ambient.is_none()) {
                amb_arr = is_ambient.cast<py::array_t<bool>>();
                auto aa = amb_arr.unchecked<2>();
                amb = aa.data(0, 0);
            }

            // P-R4: the nullable SIGNED radiation accumulator (None -> nullptr,
            // i.e. no fold — the same idiom as is_ambient above).
            const int32_t* rnet = nullptr;
            py::array_t<int32_t> rnet_arr;
            if (!rad_net.is_none()) {
                rnet_arr = rad_net.cast<py::array_t<int32_t>>();
                auto ra = rnet_arr.unchecked<2>();
                rnet = ra.data(0, 0);
            }

            auto destroyed = self.step_tail(
                rip, ripv, wd, wp, sol,
                f, atm, sm, whp, temp, wx, wy, vac, fl,
                temp, hp, shift, fs, tsol, csg, fr, tep,
                gas_ptr, gcons, n_gases, o2_idx,
                h, w, sim_time, amb, rnet);
            py::list result;
            for (const auto& [dy, dx] : destroyed) {
                result.append(py::make_tuple(dy, dx));
            }
            return result;
        }, py::arg("ripple"), py::arg("ripple_v"),
           py::arg("water_depth"), py::arg("wave_p"), py::arg("solid"),
           py::arg("fire"), py::arg("atmosphere"), py::arg("smoke"),
           py::arg("wall_hp"), py::arg("temperature"),
           py::arg("wind_x"), py::arg("wind_y"),
           py::arg("is_vacuum"), py::arg("flammable"),
           py::arg("heat"), py::arg("heat_inv_shift"), py::arg("face_shift"),
           py::arg("thermal_solid"),             // thermal-mass axis (required)
           py::arg("cool_shift_grid"),           // cool-shift axis (required)
           py::arg("fuel_recip"),                // fuel-fraction axis (required)
           py::arg("fire_T_ext_plane"),          // per-material T_ext (required)
           py::arg("gas"), py::arg("gas_conservative"), py::arg("o2_idx"),
           py::arg("sim_time"),
           py::arg("is_ambient") = py::none(),   // BC (default None = space map)
           py::arg("rad_net") = py::none())      // P-R4 (default None = no fold)
        // --- Patch 1 S4b: the IMEX atmosphere/smoke substep loop ------------
        // run_substeps moves the per-tick IMEX substep block of PhysicsRunner.step
        // (between _step_water and step_tail) into C++. Pointer extraction mirrors
        // the AtmosphereSolver.step / SmokeDynamics.step bindings above. `gas` is
        // the (N, h, w) contiguous density array (each plane (h, w) is one gas);
        // `gas_diffusion` is the (N,) per-gas base-diffusion column. sink_x/sink_y
        // are fetched Python-side (gmap.sink_fields() — a lazy BFS) and passed in.
        // `solid` is passed once and used as both the atmos/smoke `is_wall`. The
        // n / dt_actual / dt_smoke precision matching (the integer cliff + the
        // double-until-the-solver-boundary contract) lives in C++ (run_substeps).
        // EOS refactor P3: `wave_p` is repurposed as p_prev; `wave_v`/
        // `wave_source`/`sink_x`/`sink_y` are RETIRED (no longer accepted —
        // sink_hop's BFS loop is deleted); `temperature` is a new required arg.
        .def("run_substeps", [](PhysicsEngine& self,
                                py::array_t<int32_t> p_prev,        // was wave_p
                                py::array_t<int32_t> atmosphere,    // S2c: Q16.16 int32
                                py::array_t<int32_t> wind_x,        // S2c: Q16.16 int32
                                py::array_t<int32_t> wind_y,        // S2c: Q16.16 int32
                                py::array_t<int32_t> temperature,   // EOS P3
                                py::array_t<bool>  obstacles,
                                py::array_t<bool>  solid,
                                py::array_t<bool>  is_vacuum,
                                py::array_t<float> dyn_permeability,
                                py::array_t<float> dyn_wave_absorb,
                                py::array_t<int32_t> gas,           // S2b: Q16.16 int32
                                py::array_t<float> gas_diffusion,
                                py::array_t<bool> gas_conservative, // EOS P1
                                py::array_t<float> gas_decay,       // EOS P4
                                int inert_n2_idx,                   // EOS P4
                                float sim_time,
                                // BC (boundary_conditions_spec_2026-07-19): the
                                // planetside AMBIENT ring. All None on space maps
                                // -> nullptr -> byte-identical (dormancy by branch).
                                py::object is_ambient,
                                py::object n_amb,
                                int32_t p_amb,
                                py::object sponge_sigma,
                                py::object sponge_udamp,
                                bool do_traces,     // S8a Path B
                                // THERMAL-MASS AXIS, P-EOS: the per-medium
                                // THERMAL mask (GameMap.thermal_solid).
                                // OPTIONAL (None -> nullptr -> the pre-patch
                                // byte-identical path), mirroring the
                                // TemperatureSolver.step binding.
                                py::object thermal_solid) {
            auto [pp, h, w]    = get_2d(p_prev);
            auto [atm, h4, w4] = get_2d(atmosphere);
            auto [wx, h5, w5]  = get_2d(wind_x);
            auto [wy, h6, w6]  = get_2d(wind_y);
            auto [temp, h6b, w6b] = get_2d(temperature);
            auto [obs, h7, w7] = get_2d_const(obstacles);
            auto [sol, h8, w8] = get_2d_const(solid);
            auto [vac, h9, w9] = get_2d_const(is_vacuum);
            auto [perm, h10, w10] = get_2d_const(dyn_permeability);
            auto [wabs, h11, w11] = get_2d_const(dyn_wave_absorb);
            // gas: (N, h, w) contiguous — pass the base pointer + N; run_substeps
            // strides by plane (h*w) internally. h/w come from the 2D fields above.
            auto gv = gas.mutable_unchecked<3>();
            int32_t* gas_ptr = gv.mutable_data(0, 0, 0);        // S2b: Q16.16 int32
            const int n_gases = static_cast<int>(gv.shape(0));
            // gas_diffusion: (N,) float32 — the per-gas base-diffusion column.
            auto gd = gas_diffusion.unchecked<1>();
            const float* gdiff = gd.data(0);
            // gas_conservative: (N,) bool — EOS P1's bulk-species flag
            // (simulation.gases.GasTable.conservative), true only for O2/inert_N2.
            auto gc = gas_conservative.unchecked<1>();
            const bool* gcons = gc.data(0);
            // gas_decay: (N,) float32 — EOS P4's per-gas trace decay column
            // (simulation.gases.GasTable.decay), applied once per tick after
            // each trace plane's own advection. The decayed count simply
            // VANISHES (P-T0, design §2.6 — the P4 decay->inert_N2 credit is
            // DELETED; decay is no longer "oxidation into bulk", just loss).
            auto gdc = gas_decay.unchecked<1>();
            const float* gdecay = gdc.data(0);
            // BC nullable extraction (the WaterSolver.step precedent, above):
            // None -> nullptr; else cast to an array kept alive in this scope.
            const bool* amb = nullptr;
            py::array_t<bool> amb_arr;
            if (!is_ambient.is_none()) {
                amb_arr = is_ambient.cast<py::array_t<bool>>();
                auto aa = amb_arr.unchecked<2>();
                amb = aa.data(0, 0);
            }
            const int32_t* namb = nullptr;
            py::array_t<int32_t> namb_arr;
            if (!n_amb.is_none()) {
                namb_arr = n_amb.cast<py::array_t<int32_t>>();
                auto na = namb_arr.unchecked<1>();
                namb = na.data(0);
            }
            const int32_t* sponge = nullptr;
            py::array_t<int32_t> sponge_arr;
            if (!sponge_sigma.is_none()) {
                sponge_arr = sponge_sigma.cast<py::array_t<int32_t>>();
                auto sp = sponge_arr.unchecked<2>();
                sponge = sp.data(0, 0);
            }
            const int32_t* udamp = nullptr;
            py::array_t<int32_t> udamp_arr;
            if (!sponge_udamp.is_none()) {
                udamp_arr = sponge_udamp.cast<py::array_t<int32_t>>();
                auto up = udamp_arr.unchecked<2>();
                udamp = up.data(0, 0);
            }
            // THERMAL-MASS AXIS, P-EOS: nullable extraction, the BC precedent.
            const bool* tsol = nullptr;
            py::array_t<bool> tsol_arr;
            if (!thermal_solid.is_none()) {
                tsol_arr = thermal_solid.cast<py::array_t<bool>>();
                auto ta = tsol_arr.unchecked<2>();
                tsol = ta.data(0, 0);
            }
            self.run_substeps(
                pp, atm, wx, wy, temp,
                obs, sol, vac, perm, wabs,
                gas_ptr, gdiff, n_gases, gcons,
                gdecay, inert_n2_idx,
                h, w, sim_time,
                amb, namb, p_amb, sponge, udamp, do_traces, tsol);
        }, py::arg("p_prev"),
           py::arg("atmosphere"), py::arg("wind_x"), py::arg("wind_y"),
           py::arg("temperature"),
           py::arg("obstacles"), py::arg("solid"), py::arg("is_vacuum"),
           py::arg("dyn_permeability"), py::arg("dyn_wave_absorb"),
           py::arg("gas"), py::arg("gas_diffusion"), py::arg("gas_conservative"),
           py::arg("gas_decay"), py::arg("inert_n2_idx"),
           py::arg("sim_time"),
           // BC: default None/0 -> the pre-BC call site (space maps) is unchanged.
           py::arg("is_ambient") = py::none(),
           py::arg("n_amb") = py::none(),
           py::arg("p_amb") = 0,
           py::arg("sponge_sigma") = py::none(),
           py::arg("sponge_udamp") = py::none(),
           py::arg("do_traces") = true,   // S8a Path B (default = prior behaviour)
           // THERMAL-MASS AXIS, P-EOS: default None -> pre-patch behaviour, so
           // every existing direct caller keeps its exact meaning; the live
           // runner always passes GameMap.thermal_solid.
           py::arg("thermal_solid") = py::none())
        // --- S8a Path A: the fully device-resident EOS stage ----------------
        // Host mirrors feed the shared pre-stage (all reductions consume
        // tick-entry state == the authoritative mirror) + telemetry; device
        // pointers are CuPy .data.ptr uintptr_t (0 == nullptr for the ambient
        // statics). Bound on every build — the method throws without CUDA.
        .def("run_substeps_resident", [](PhysicsEngine& self,
                                py::array_t<int32_t> p_prev,
                                py::array_t<int32_t> atmosphere,
                                py::array_t<int32_t> wind_x,
                                py::array_t<int32_t> wind_y,
                                py::array_t<int32_t> temperature,
                                py::array_t<bool>  solid,
                                py::array_t<bool>  is_vacuum,
                                py::array_t<float> dyn_permeability,
                                py::array_t<float> dyn_wave_absorb,
                                py::array_t<int32_t> gas,
                                py::array_t<bool> gas_conservative,
                                float sim_time,
                                py::object is_ambient,
                                py::object n_amb,
                                int32_t p_amb,
                                std::uintptr_t d_atmosphere,
                                std::uintptr_t d_wave_p,
                                std::uintptr_t d_wind_x,
                                std::uintptr_t d_wind_y,
                                std::uintptr_t d_temperature,
                                std::uintptr_t d_gas,
                                std::uintptr_t d_solid,
                                std::uintptr_t d_is_vacuum,
                                std::uintptr_t d_dyn_permeability,
                                std::uintptr_t d_is_ambient,
                                std::uintptr_t d_sponge_sigma,
                                std::uintptr_t d_sponge_udamp,
                                // THERMAL-MASS AXIS, P-EOS: the mirror + the
                                // device copy of the per-medium THERMAL mask.
                                py::object thermal_solid,
                                std::uintptr_t d_thermal_solid) {
            auto [pp, h, w]    = get_2d(p_prev);
            auto [atm, h4, w4] = get_2d_const(atmosphere);
            auto [wx, h5, w5]  = get_2d_const(wind_x);
            auto [wy, h6, w6]  = get_2d_const(wind_y);
            auto [temp, h6b, w6b] = get_2d_const(temperature);
            auto [sol, h8, w8] = get_2d_const(solid);
            auto [vac, h9, w9] = get_2d_const(is_vacuum);
            auto [perm, h10, w10] = get_2d_const(dyn_permeability);
            auto [wabs, h11, w11] = get_2d_const(dyn_wave_absorb);
            auto gv = gas.unchecked<3>();
            const int32_t* gas_ptr = gv.data(0, 0, 0);
            const int n_gases = static_cast<int>(gv.shape(0));
            auto gc = gas_conservative.unchecked<1>();
            const bool* gcons = gc.data(0);
            const bool* amb = nullptr;
            py::array_t<bool> amb_arr;
            if (!is_ambient.is_none()) {
                amb_arr = is_ambient.cast<py::array_t<bool>>();
                auto aa = amb_arr.unchecked<2>();
                amb = aa.data(0, 0);
            }
            const int32_t* namb = nullptr;
            py::array_t<int32_t> namb_arr;
            if (!n_amb.is_none()) {
                namb_arr = n_amb.cast<py::array_t<int32_t>>();
                auto na = namb_arr.unchecked<1>();
                namb = na.data(0);
            }
            // THERMAL-MASS AXIS, P-EOS: nullable mirror extraction (BC idiom).
            const bool* tsol = nullptr;
            py::array_t<bool> tsol_arr;
            if (!thermal_solid.is_none()) {
                tsol_arr = thermal_solid.cast<py::array_t<bool>>();
                auto ta = tsol_arr.unchecked<2>();
                tsol = ta.data(0, 0);
            }
            self.run_substeps_resident(
                pp, atm, wx, wy, temp,
                sol, vac, perm, wabs,
                gas_ptr, n_gases, gcons,
                h, w, sim_time,
                amb, namb, p_amb,
                d_atmosphere, d_wave_p, d_wind_x, d_wind_y,
                d_temperature, d_gas, d_solid, d_is_vacuum,
                d_dyn_permeability, d_is_ambient,
                d_sponge_sigma, d_sponge_udamp,
                tsol, d_thermal_solid);
        }, py::arg("p_prev"), py::arg("atmosphere"),
           py::arg("wind_x"), py::arg("wind_y"), py::arg("temperature"),
           py::arg("solid"), py::arg("is_vacuum"),
           py::arg("dyn_permeability"), py::arg("dyn_wave_absorb"),
           py::arg("gas"), py::arg("gas_conservative"),
           py::arg("sim_time"),
           py::arg("is_ambient") = py::none(),
           py::arg("n_amb") = py::none(),
           py::arg("p_amb") = 0,
           py::arg("d_atmosphere") = 0, py::arg("d_wave_p") = 0,
           py::arg("d_wind_x") = 0, py::arg("d_wind_y") = 0,
           py::arg("d_temperature") = 0, py::arg("d_gas") = 0,
           py::arg("d_solid") = 0, py::arg("d_is_vacuum") = 0,
           py::arg("d_dyn_permeability") = 0,
           py::arg("d_is_ambient") = 0,
           py::arg("d_sponge_sigma") = 0, py::arg("d_sponge_udamp") = 0,
           // THERMAL-MASS AXIS, P-EOS: default None/0 -> the pre-patch path.
           py::arg("thermal_solid") = py::none(),
           py::arg("d_thermal_solid") = 0)
        // --- Patch 1 S4c: the water-layer array arithmetic ------------------
        // step_water moves the array-op body of PhysicsRunner._step_water into
        // C++ (substep loop + W5 flash-boil + W3 displacement/seal + the final
        // copyto). Pointer extraction mirrors the WaterSolver.step binding above:
        // floor_height/atmosphere/wave_p are passed here as plain arrays (the
        // Python call site always passes all three — they are nullable only in the
        // standalone WaterSolver.step binding). `gas` is the (N, h, w) contiguous
        // density array; step_water strides to the steam slice internally via
        // steam_idx. `before` is the runner's _water_depth_before snapshot — READ
        // by the W3 displacement and MUTATED by the final copyto (passed in, the
        // runner keeps owning it across ticks). The W3/W5 scalar params arrive as
        // Python doubles (config-bound floats) and are cast to float32 INSIDE
        // step_water at numpy's exact cast points — the whole bit-identity hinge.
        // KEPT IN PYTHON (the runner does these, then calls step_water only when
        // NOT dormant): the lazy-init (before-seed, water.dx bind, steam_idx
        // resolve), the dormancy early-out, and the sparse source-holds loop.
        // EOS refactor P3: `wave_p` arg RETIRED (the head bridge is gone —
        // `atmosphere` is read directly as the integer P); `n_gases` is
        // derived from `gas`'s shape and threaded to step_water for the W3
        // occupancy-transition evacuation loop (every gas plane, not just
        // the W5 steam slice).
        .def("step_water", [](const PhysicsEngine& self,
                              py::array_t<int32_t> water_depth,    // S1: Q16.16 int32
                              py::array_t<int32_t> flow_vx,        // S1: Q16.16 int32
                              py::array_t<int32_t> flow_vy,        // S1: Q16.16 int32
                              py::array_t<int32_t> floor_height,   // S1: Q16.16 int32
                              py::array_t<int32_t> atmosphere,     // S2c: Q16.16 int32 == P
                              py::array_t<bool>  solid,
                              py::array_t<int32_t> gas,            // S2b: Q16.16 int32
                              py::array_t<int32_t> before,         // S1: Q16.16 int32 snapshot
                              py::array_t<float> dyn_permeability, // float (FLOAT BRIDGE: seal)
                              int steam_idx, float tilt_x, float tilt_y,
                              float sim_time,
                              double ceiling_h, double flood_eps,
                              double ratio_cap, double boil_rate,
                              double boil_p_thresh, double steam_yield) {
            auto [wd, h, w]    = get_2d(water_depth);
            auto [vx, h2, w2]  = get_2d(flow_vx);
            auto [vy, h3, w3]  = get_2d(flow_vy);
            auto [fl, h4, w4]  = get_2d_const(floor_height);
            auto [atm, h5, w5] = get_2d(atmosphere);
            auto [sol, h7, w7] = get_2d_const(solid);
            auto [bef, h8, w8] = get_2d(before);
            auto [perm, h9, w9] = get_2d(dyn_permeability);
            // gas: (N, h, w) contiguous — pass the base pointer + N; step_water
            // strides by plane (h*w) internally (steam_idx AND the W3 evac loop).
            auto gv = gas.mutable_unchecked<3>();
            int32_t* gas_ptr = gv.mutable_data(0, 0, 0);        // S2b: Q16.16 int32
            const int n_gases = static_cast<int>(gv.shape(0));
            self.step_water(
                wd, vx, vy, fl, atm, sol,
                gas_ptr, n_gases, bef, perm,
                steam_idx, tilt_x, tilt_y,
                h, w, sim_time,
                ceiling_h, flood_eps, ratio_cap,
                boil_rate, boil_p_thresh, steam_yield);
        }, py::arg("water_depth"), py::arg("flow_vx"), py::arg("flow_vy"),
           py::arg("floor_height"), py::arg("atmosphere"),
           py::arg("solid"), py::arg("gas"), py::arg("before"),
           py::arg("dyn_permeability"), py::arg("steam_idx"),
           py::arg("tilt_x"), py::arg("tilt_y"), py::arg("sim_time"),
           py::arg("ceiling_h"), py::arg("flood_eps"), py::arg("ratio_cap"),
           py::arg("boil_rate"), py::arg("boil_p_thresh"), py::arg("steam_yield"))
        // --- S8a Path B: the water substep COUNT (the integer cliff) ---------
        // Exposes the exact n = max(1, ceil_div(quantize(sim_time), max_dt_q))
        // step_water computes internally, so the resident path (which runs the
        // substep loop on device) uses the BIT-IDENTICAL count. wdt = sim_time/n
        // is formed Python-side and cast to float32 at the pybind boundary — the
        // same (float)((double)sim_time / n) step_water performs.
        .def("water_substep_count", [](const PhysicsEngine& self, float sim_time) {
            using namespace fixedpoint;
            return std::max(1, ceil_div(quantize((double)sim_time),
                                        self.water.max_dt_q()));
        }, py::arg("sim_time"))
        // --- S8a Path B: the water HOST TAIL, split out of step_water --------
        // The W5 flash-boil + W3 displacement + copyto, WITHOUT the substep loop.
        // The resident path runs the substep loop on device (water_substeps_resident)
        // then calls this on the mirror — bit-identical to the monolithic step_water.
        .def("step_water_tail", [](const PhysicsEngine& self,
                                   py::array_t<int32_t> water_depth,
                                   py::array_t<int32_t> atmosphere,
                                   py::array_t<bool>  solid,
                                   py::array_t<int32_t> gas,
                                   py::array_t<int32_t> before,
                                   py::array_t<float> dyn_permeability,
                                   int steam_idx, float sim_time,
                                   double ceiling_h, double flood_eps,
                                   double ratio_cap, double boil_rate,
                                   double boil_p_thresh, double steam_yield) {
            auto [wd, h, w]    = get_2d(water_depth);
            auto [atm, h5, w5] = get_2d(atmosphere);
            auto [sol, h7, w7] = get_2d_const(solid);
            auto [bef, h8, w8] = get_2d(before);
            auto [perm, h9, w9] = get_2d(dyn_permeability);
            auto gv = gas.mutable_unchecked<3>();
            int32_t* gas_ptr = gv.mutable_data(0, 0, 0);
            const int n_gases = static_cast<int>(gv.shape(0));
            self.step_water_tail(
                wd, atm, sol, gas_ptr, n_gases, bef, perm,
                steam_idx, h, w, sim_time,
                ceiling_h, flood_eps, ratio_cap,
                boil_rate, boil_p_thresh, steam_yield);
        }, py::arg("water_depth"), py::arg("atmosphere"), py::arg("solid"),
           py::arg("gas"), py::arg("before"), py::arg("dyn_permeability"),
           py::arg("steam_idx"), py::arg("sim_time"),
           py::arg("ceiling_h"), py::arg("flood_eps"), py::arg("ratio_cap"),
           py::arg("boil_rate"), py::arg("boil_p_thresh"), py::arg("steam_yield"))
        // --- stamp_units: the per-tick dynamic-field rebuild ----------------
        // Moves the FIELD-REBUILD half of GameMap.stamp_units into C++ (the unit
        // iteration / occupied_tiles() / alive-filter / bounds-check / defaults
        // stay Python, flattened into per-row arrays). Static (h,w) grids
        // permeability/wave_absorb + (h,w,3) light_atten are read; the dyn_*
        // targets + the (h,w) bool obstacles are written IN-PLACE. The flat stamp
        // arrays (ys/xs int32; perm/wabsorb/atten_{r,g,b} float32) carry one row
        // per stamped footprint tile. PURE-STRUCTURE, 0-ULP (copies + compare +
        // min/max only). The atmosphere-refill bit stays Python (Q1, locked).
        .def("stamp_units", [](const PhysicsEngine& self,
                               py::array_t<float> permeability,
                               py::array_t<float> wave_absorb,
                               py::array_t<float> light_atten,
                               py::array_t<float> dyn_permeability,
                               py::array_t<float> dyn_wave_absorb,
                               py::array_t<float> dyn_light_atten,
                               py::array_t<bool>  obstacles,
                               py::array_t<int32_t> ys,
                               py::array_t<int32_t> xs,
                               py::array_t<float> perm,
                               py::array_t<float> wabsorb,
                               py::array_t<float> atten_r,
                               py::array_t<float> atten_g,
                               py::array_t<float> atten_b) {
            auto [pm, h, w]    = get_2d_const(permeability);
            auto [wa, h2, w2]  = get_2d_const(wave_absorb);
            auto [dpm, h3, w3] = get_2d(dyn_permeability);
            auto [dwa, h4, w4] = get_2d(dyn_wave_absorb);
            auto [obs, h5, w5] = get_2d(obstacles);
            // light_atten / dyn_light_atten are (h, w, 3) f32 — pass the base
            // pointer; the loop strides the trailing channel axis internally.
            auto la_v  = light_atten.unchecked<3>();
            const float* la = la_v.data(0, 0, 0);
            auto dla_v = dyn_light_atten.mutable_unchecked<3>();
            float* dla = dla_v.mutable_data(0, 0, 0);
            // Flat per-row stamp arrays (1D). Empty arrays (no living units) are
            // valid — n_stamp == 0 -> the stamp loop is a no-op (reset only).
            auto ys_v = ys.unchecked<1>();
            auto xs_v = xs.unchecked<1>();
            const int n_stamp = static_cast<int>(ys_v.shape(0));
            const int32_t* ys_p = (n_stamp > 0) ? ys_v.data(0) : nullptr;
            const int32_t* xs_p = (n_stamp > 0) ? xs_v.data(0) : nullptr;
            auto perm_v = perm.unchecked<1>();
            auto wabs_v = wabsorb.unchecked<1>();
            auto ar_v   = atten_r.unchecked<1>();
            auto ag_v   = atten_g.unchecked<1>();
            auto ab_v   = atten_b.unchecked<1>();
            const float* perm_p = (n_stamp > 0) ? perm_v.data(0) : nullptr;
            const float* wabs_p = (n_stamp > 0) ? wabs_v.data(0) : nullptr;
            const float* ar_p   = (n_stamp > 0) ? ar_v.data(0) : nullptr;
            const float* ag_p   = (n_stamp > 0) ? ag_v.data(0) : nullptr;
            const float* ab_p   = (n_stamp > 0) ? ab_v.data(0) : nullptr;
            self.stamp_units(pm, wa, la, dpm, dwa, dla, obs,
                             ys_p, xs_p, perm_p, wabs_p, ar_p, ag_p, ab_p,
                             n_stamp, h, w);
        }, py::arg("permeability"), py::arg("wave_absorb"), py::arg("light_atten"),
           py::arg("dyn_permeability"), py::arg("dyn_wave_absorb"),
           py::arg("dyn_light_atten"), py::arg("obstacles"),
           py::arg("ys"), py::arg("xs"), py::arg("perm"), py::arg("wabsorb"),
           py::arg("atten_r"), py::arg("atten_g"), py::arg("atten_b"));
}
