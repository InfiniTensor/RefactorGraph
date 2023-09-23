#include "compiler.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace refactor::python_ffi {

    PYBIND11_MODULE(python_ffi, m) {
        using policy = py::return_value_policy;

        // clang-format off
        py::class_<Compiler, std::shared_ptr<Compiler>>(m, "Compiler")
            .def("substitute"      , &Compiler::substitute   , policy::automatic )
            .def("set_input"       , &Compiler::setInput     , policy::automatic )
            .def("check_variables" , &Compiler::fillEdgeInfo , policy::move      )
            .def("get_tensor"      , &Compiler::getTensor    , policy::move      );
        // clang-format on
    }

}// namespace refactor::python_ffi
