#include "compiler.h"
#include "import.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace refactor::python_ffi {

    PYBIND11_MODULE(python_ffi, m) {
        using policy = py::return_value_policy;
        using namespace frontend;

        // clang-format off

        py::class_<Compiler , std::shared_ptr<Compiler> >(m, "Compiler" )
            .def("substitute"      , &Compiler::substitute   , policy::automatic )
            .def("set_input"       , &Compiler::setInput     , policy::automatic )
            .def("check_variables" , &Compiler::fillEdgeInfo , policy::move      )
            .def("get_tensor"      , &Compiler::getTensor    , policy::move      );

        py::class_<Tensor   , std::shared_ptr<Tensor>   >(m, "Tensor"   );
        py::class_<Operator , std::shared_ptr<Operator> >(m, "Operator" );
        py::class_<Graph    , std::shared_ptr<Graph>    >(m, "Graph"    );

        m   .def("make_node"       , &makeOp                 , policy::move      )
            .def("make_tensor"     , &makeTensor             , policy::move      )
            .def("make_data"       , &makeTensorWithData     , policy::move      )
            .def("make_graph"      , &makeGraph              , policy::move      );

        // clang-format on
    }

}// namespace refactor::python_ffi
