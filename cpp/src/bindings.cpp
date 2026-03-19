#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "atmosphere_solver.h"
#include "smoke_dynamics.h"
#include "fire_simulation.h"
#include "raycaster.h"

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

PYBIND11_MODULE(breach_physics, m) {
    m.doc() = "Breach physics engine -- C++ accelerated simulation";

    // --- AtmosphereSolver (IMEX: explicit wave + implicit diffusion) ---
    py::class_<AtmosphereSolver>(m, "AtmosphereSolver")
        .def(py::init<>())
        .def_readwrite("c",                   &AtmosphereSolver::c)
        .def_readwrite("damping",             &AtmosphereSolver::damping)
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
            self.step(wp, wv, ws, atm, wx, wy, obs, wl, vac, h, w, dt);
        }, py::arg("wave_p"), py::arg("wave_v"), py::arg("wave_source"),
           py::arg("atmosphere"),
           py::arg("wind_x"), py::arg("wind_y"),
           py::arg("obstacles"), py::arg("is_wall"), py::arg("is_vacuum"),
           py::arg("dt"));

    // --- SmokeDynamics ---
    py::class_<SmokeDynamics>(m, "SmokeDynamics")
        .def(py::init<>())
        .def_readwrite("d_smoke",        &SmokeDynamics::d_smoke)
        .def_readwrite("advection_rate", &SmokeDynamics::advection_rate)
        .def_readwrite("wave_advection", &SmokeDynamics::wave_advection)
        .def("step", [](const SmokeDynamics& self,
                        py::array_t<float> smoke,
                        py::array_t<float> atmosphere,
                        py::array_t<float> wave_p,
                        py::array_t<bool>  obstacles,
                        py::array_t<bool>  is_wall,
                        py::array_t<bool>  is_vacuum,
                        float dt) {
            auto [sm, h, w] = get_2d(smoke);
            auto [atm, h2, w2] = get_2d_const(atmosphere);
            auto [wp, h3, w3] = get_2d_const(wave_p);
            auto [obs, h4, w4] = get_2d_const(obstacles);
            auto [wl, h5, w5] = get_2d_const(is_wall);
            auto [vac, h6, w6] = get_2d_const(is_vacuum);
            self.step(sm, atm, wp, obs, wl, vac, h, w, dt);
        }, py::arg("smoke"), py::arg("atmosphere"), py::arg("wave_p"),
           py::arg("obstacles"), py::arg("is_wall"), py::arg("is_vacuum"),
           py::arg("dt"));

    // --- FireSimulation ---
    py::class_<FireParams>(m, "FireParams")
        .def(py::init<>())
        .def_readwrite("spread_rate",    &FireParams::spread_rate)
        .def_readwrite("o2_threshold",   &FireParams::o2_threshold)
        .def_readwrite("o2_consumption", &FireParams::o2_consumption)
        .def_readwrite("smoke_emission", &FireParams::smoke_emission)
        .def_readwrite("wall_damage",    &FireParams::wall_damage)
        .def_readwrite("k_wind_thresh",  &FireParams::k_wind_thresh)
        .def_readwrite("k_wind_net",     &FireParams::k_wind_net);

    py::class_<FireSimulation>(m, "FireSimulation")
        .def(py::init<>())
        .def_readwrite("params", &FireSimulation::params)
        .def("step", [](const FireSimulation& self,
                        py::array_t<float> fire,
                        py::array_t<float> atmosphere,
                        py::array_t<float> smoke,
                        py::array_t<float> wall_hp,
                        py::array_t<bool>  is_wall,
                        py::array_t<bool>  flammable,
                        float dt) -> py::list {
            auto [f, h, w] = get_2d(fire);
            auto [atm, h2, w2] = get_2d(atmosphere);
            auto [sm, h3, w3] = get_2d(smoke);
            auto [whp, h4, w4] = get_2d(wall_hp);
            auto [wl, h5, w5] = get_2d_const(is_wall);
            auto [fl, h6, w6] = get_2d_const(flammable);
            auto destroyed = self.step(f, atm, sm, whp, wl, fl, h, w, dt);
            py::list result;
            for (const auto& [dy, dx] : destroyed) {
                result.append(py::make_tuple(dy, dx));
            }
            return result;
        }, py::arg("fire"), py::arg("atmosphere"), py::arg("smoke"),
           py::arg("wall_hp"), py::arg("is_wall"), py::arg("flammable"),
           py::arg("dt"));

    // --- Raycaster ---
    py::class_<Raycaster>(m, "Raycaster")
        .def(py::init<>())
        .def_readwrite("smoke_absorption", &Raycaster::smoke_absorption)
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
           py::arg("smoke"), py::arg("is_wall"));
}
