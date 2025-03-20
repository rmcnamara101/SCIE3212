#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "simulation.hpp"

PYBIND11_MODULE(cpp_simulation, m) {
    py::class_<SimulationCore>(m, "SimulationCore")
        .def(py::init<const py::array_t<double>&,
                     const py::array_t<double>&,
                     const py::array_t<double>&,
                     const py::array_t<double>&,
                     const py::array_t<double>&,
                     const py::array_t<double>&,
                     double,
                     double,
                     const py::dict&>())
        .def("step_rk4", &SimulationCore::step_rk4)
        .def("get_state", &SimulationCore::get_state);
}
