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
#include "physics_engine.h"

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
        .def("max_dt", &AtmosphereSolver::max_dt)
        .def("step", [](const AtmosphereSolver& self,
                        py::array_t<float> wave_p,
                        py::array_t<float> wave_v,
                        py::array_t<float> wave_source,
                        py::array_t<float> atmosphere,
                        py::array_t<float> wind_x,
                        py::array_t<float> wind_y,
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
           py::arg("dt"));

    // --- SmokeDynamics (uses precomputed wind from AtmosphereSolver) ---
    py::class_<SmokeDynamics>(m, "SmokeDynamics")
        .def(py::init<>())
        .def_readwrite("d_smoke",               &SmokeDynamics::d_smoke)
        .def_readwrite("advection_rate",         &SmokeDynamics::advection_rate)
        .def_readwrite("dt_scale",               &SmokeDynamics::dt_scale)
        .def_readwrite("wind_diffusion_scale",   &SmokeDynamics::wind_diffusion_scale)
        .def_readwrite("sink_strength",          &SmokeDynamics::sink_strength)
        .def("step", [](const SmokeDynamics& self,
                        py::array_t<float> smoke,
                        py::array_t<float> wind_x,
                        py::array_t<float> wind_y,
                        py::array_t<float> sink_x,
                        py::array_t<float> sink_y,
                        py::array_t<bool>  obstacles,
                        py::array_t<bool>  is_wall,
                        py::array_t<bool>  is_vacuum,
                        py::array_t<float> permeability,
                        float dt) {
            auto [sm, h, w] = get_2d(smoke);
            auto [wx, h2, w2] = get_2d_const(wind_x);
            auto [wy, h3, w3] = get_2d_const(wind_y);
            auto [skx, h4b, w4b] = get_2d_const(sink_x);
            auto [sky, h4c, w4c] = get_2d_const(sink_y);
            auto [obs, h4, w4] = get_2d_const(obstacles);
            auto [wl, h5, w5] = get_2d_const(is_wall);
            auto [vac, h6, w6] = get_2d_const(is_vacuum);
            auto [perm, h7, w7] = get_2d_const(permeability);
            self.step(sm, wx, wy, skx, sky, obs, wl, vac, perm, h, w, dt);
        }, py::arg("smoke"), py::arg("wind_x"), py::arg("wind_y"),
           py::arg("sink_x"), py::arg("sink_y"),
           py::arg("obstacles"), py::arg("is_wall"), py::arg("is_vacuum"),
           py::arg("permeability"),
           py::arg("dt"));

    // --- FireSimulation (signed-logistic feedback; fire_design_proposal §2/§3) ---
    py::class_<FireParams>(m, "FireParams")
        .def(py::init<>())
        .def_readwrite("k_grow",         &FireParams::k_grow)
        .def_readwrite("k_die",          &FireParams::k_die)
        .def_readwrite("fire_T_ext",     &FireParams::fire_T_ext)
        .def_readwrite("fire_T_span",    &FireParams::fire_T_span)
        .def_readwrite("fuel_ref",       &FireParams::fuel_ref)
        .def_readwrite("P_min",          &FireParams::P_min)
        .def_readwrite("P_full",         &FireParams::P_full)
        .def_readwrite("I_min",          &FireParams::I_min)
        .def_readwrite("k_wind_fan",     &FireParams::k_wind_fan)
        .def_readwrite("k_wind_strip",   &FireParams::k_wind_strip)
        .def_readwrite("fire_pressure_gain", &FireParams::fire_pressure_gain)
        .def_readwrite("p_expand_ref",   &FireParams::p_expand_ref)
        .def_readwrite("smoke_emission", &FireParams::smoke_emission)
        .def_readwrite("wall_damage",    &FireParams::wall_damage)
        .def_readwrite("temp_scale",     &FireParams::temp_scale);

    py::class_<FireSimulation>(m, "FireSimulation")
        .def(py::init<>())
        .def_readwrite("params", &FireSimulation::params)
        .def("step", [](const FireSimulation& self,
                        py::array_t<float> fire,
                        py::array_t<float> atmosphere,
                        py::array_t<float> smoke,
                        py::array_t<float> wall_hp,
                        py::array_t<int32_t> temperature,
                        py::array_t<float> wind_x,
                        py::array_t<float> wind_y,
                        py::array_t<bool>  is_wall,
                        py::array_t<bool>  is_vacuum,
                        py::array_t<bool>  flammable,
                        float dt) -> py::list {
            auto [f, h, w] = get_2d(fire);
            auto [atm, h2, w2] = get_2d(atmosphere);
            auto [sm, h3, w3] = get_2d(smoke);
            auto [whp, h4, w4] = get_2d(wall_hp);
            auto [temp, h5, w5] = get_2d_const(temperature);
            auto [wx, h6, w6] = get_2d_const(wind_x);
            auto [wy, h7, w7] = get_2d_const(wind_y);
            auto [wl, h8, w8] = get_2d_const(is_wall);
            auto [vac, h9, w9] = get_2d_const(is_vacuum);
            auto [fl, h10, w10] = get_2d_const(flammable);
            auto destroyed = self.step(f, atm, sm, whp, temp, wx, wy,
                                       wl, vac, fl, h, w, dt);
            py::list result;
            for (const auto& [dy, dx] : destroyed) {
                result.append(py::make_tuple(dy, dx));
            }
            return result;
        }, py::arg("fire"), py::arg("atmosphere"), py::arg("smoke"),
           py::arg("wall_hp"), py::arg("temperature"),
           py::arg("wind_x"), py::arg("wind_y"),
           py::arg("is_wall"), py::arg("is_vacuum"), py::arg("flammable"),
           py::arg("dt"));

    // --- TemperatureSolver (heat -> temperature conversion §1 + conduction §2
    //     + ambient cooling §3; engine/06 §1–§3) ---
    py::class_<TemperatureSolver>(m, "TemperatureSolver")
        .def(py::init<>())
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
        .def_property("o2_vacuum_thresh",
            &TemperatureSolver::get_o2_vacuum_thresh,
            &TemperatureSolver::set_o2_vacuum_thresh)
        .def("step", [](const TemperatureSolver& self,
                        py::array_t<int32_t> temperature,
                        py::array_t<int32_t> heat,
                        py::array_t<int32_t> heat_inv_shift,
                        py::array_t<int32_t> face_shift,
                        py::array_t<bool>    solid,
                        py::array_t<bool>    is_vacuum,
                        py::array_t<float>   atmosphere) {
            auto [temp, h, w]     = get_2d(temperature);
            auto [hp, h2, w2]     = get_2d_const(heat);
            auto [shift, h3, w3]  = get_2d_const(heat_inv_shift);
            auto [sol, h4, w4]    = get_2d_const(solid);
            auto [vac, h5, w5]    = get_2d_const(is_vacuum);
            auto [atm, h6, w6]    = get_2d_const(atmosphere);
            // face_shift is (h, w, 4) int32 — fixed dir order N,S,E,W.
            auto fa = face_shift.unchecked<3>();
            const int32_t* fs = fa.data(0, 0, 0);
            self.step(temp, hp, shift, fs, sol, vac, atm, h, w);
        }, py::arg("temperature"), py::arg("heat"),
           py::arg("heat_inv_shift"), py::arg("face_shift"),
           py::arg("solid"), py::arg("is_vacuum"), py::arg("atmosphere"));

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
        .def_readwrite("coarse_cluster",   &Raycaster::coarse_cluster)
        .def("update_from_fire", [](const Raycaster& self,
                                     py::array_t<float> light_map,
                                     py::array_t<float> fire,
                                     py::array_t<float> smoke,
                                     py::array_t<bool>  is_wall) {
            auto [lm, h, w] = get_2d(light_map);
            auto [f, h2, w2] = get_2d_const(fire);
            auto [sm, h3, w3] = get_2d_const(smoke);
            auto [wl, h4, w4] = get_2d_const(is_wall);
            self.update_from_fire(lm, f, sm, wl, h, w);
        }, py::arg("light_map"), py::arg("fire"),
           py::arg("smoke"), py::arg("is_wall"))
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
        .def_static("normalize_directions",
             [](py::array_t<float> light_dx, py::array_t<float> light_dy) {
            auto [ldx, h, w]   = get_2d(light_dx);
            auto [ldy, h2, w2] = get_2d(light_dy);
            Raycaster::normalize_directions(ldx, ldy, h, w);
        }, py::arg("light_dx"), py::arg("light_dy"));

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
        .def("ripple_max_dt", &WaterSolver::ripple_max_dt)
        .def("step", [](const WaterSolver& self,
                        py::array_t<float> water_depth,
                        py::array_t<float> flow_vx,
                        py::array_t<float> flow_vy,
                        py::object floor_height,
                        py::object atmosphere,
                        py::object wave_p,
                        py::array_t<bool> solid,
                        float dt, float tilt_x, float tilt_y) {
            auto [wd, h, w]    = get_2d(water_depth);
            auto [vx, h2, w2]  = get_2d(flow_vx);
            auto [vy, h3, w3]  = get_2d(flow_vy);
            auto [sol, h4, w4] = get_2d_const(solid);
            // Nullable fields (cast_source_directional precedent): None ->
            // nullptr, else cast to an array kept alive in this scope.
            // floor_height None -> flat zero; atmosphere/wave_p None -> no
            // head term (and with k_p == 0 they are never read at all).
            const float* fl = nullptr;
            py::array_t<float> fl_arr;
            if (!floor_height.is_none()) {
                fl_arr = floor_height.cast<py::array_t<float>>();
                auto fa = fl_arr.unchecked<2>();
                fl = fa.data(0, 0);
            }
            const float* atm = nullptr;
            py::array_t<float> atm_arr;
            if (!atmosphere.is_none()) {
                atm_arr = atmosphere.cast<py::array_t<float>>();
                auto aa = atm_arr.unchecked<2>();
                atm = aa.data(0, 0);
            }
            const float* wp = nullptr;
            py::array_t<float> wp_arr;
            if (!wave_p.is_none()) {
                wp_arr = wave_p.cast<py::array_t<float>>();
                auto wa = wp_arr.unchecked<2>();
                wp = wa.data(0, 0);
            }
            self.step(wd, vx, vy, fl, atm, wp, sol, h, w, dt, tilt_x, tilt_y);
        }, py::arg("water_depth"), py::arg("flow_vx"), py::arg("flow_vy"),
           py::arg("floor_height") = py::none(),
           py::arg("atmosphere")   = py::none(),
           py::arg("wave_p")       = py::none(),
           py::arg("solid"), py::arg("dt"),
           py::arg("tilt_x"), py::arg("tilt_y"))
        // W6a ripple: the VISUAL-ONLY surface wave (canon §6). water_depth /
        // wave_p / solid are read-only — the ripple never feeds back into
        // transport. wave_p nullable (None -> no splash source, never read).
        .def("step_ripple", [](const WaterSolver& self,
                               py::array_t<float> ripple,
                               py::array_t<float> ripple_v,
                               py::array_t<float> water_depth,
                               py::object wave_p,
                               py::array_t<bool> solid,
                               float dt) {
            auto [r, h, w]     = get_2d(ripple);
            auto [rv, h2, w2]  = get_2d(ripple_v);
            auto [wd, h3, w3]  = get_2d_const(water_depth);
            auto [sol, h4, w4] = get_2d_const(solid);
            const float* wp = nullptr;
            py::array_t<float> wp_arr;
            if (!wave_p.is_none()) {
                wp_arr = wave_p.cast<py::array_t<float>>();
                auto wa = wp_arr.unchecked<2>();
                wp = wa.data(0, 0);
            }
            self.step_ripple(r, rv, wd, wp, sol, h, w, dt);
        }, py::arg("ripple"), py::arg("ripple_v"), py::arg("water_depth"),
           py::arg("wave_p") = py::none(),
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
                             py::array_t<float> water_depth,
                             py::array_t<float> wave_p,
                             py::array_t<bool>  solid,
                             // fire group
                             py::array_t<float> fire,
                             py::array_t<float> atmosphere,
                             py::array_t<float> smoke,
                             py::array_t<float> wall_hp,
                             py::array_t<int32_t> temperature,
                             py::array_t<float> wind_x,
                             py::array_t<float> wind_y,
                             py::array_t<bool>  is_vacuum,
                             py::array_t<bool>  flammable,
                             // temperature group
                             py::array_t<int32_t> heat,
                             py::array_t<int32_t> heat_inv_shift,
                             py::array_t<int32_t> face_shift,
                             float sim_time) -> py::list {
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

            auto destroyed = self.step_tail(
                rip, ripv, wd, wp, sol,
                f, atm, sm, whp, temp, wx, wy, vac, fl,
                temp, hp, shift, fs,
                h, w, sim_time);
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
           py::arg("sim_time"));
}
