#include "communication/operators.h"
#include "compiler.h"
#include "import.h"
#include "onnx/operators.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>// keep this line to convert stl types

namespace py = pybind11;

namespace refactor::python_ffi {

    PYBIND11_MODULE(python_ffi, m) {
        using policy = py::return_value_policy;
        using namespace frontend;

        onnx::register_();
        communication::register_();

        // clang-format off

        py::class_<Compiler , std::shared_ptr<Compiler> >(m, "Compiler" )
            .def("substitute"      , &Compiler::substitute   , policy::automatic )
            .def("set_input"       , &Compiler::setInput     , policy::automatic )
            .def("check_variables" , &Compiler::fillEdgeInfo , policy::move      )
            .def("get_tensor"      , &Compiler::getTensor    , policy::move      )
            .def("lower"           , &Compiler::lower        , policy::move      );

        py::class_<Tensor   , std::shared_ptr<Tensor>   >(m, "Tensor"   );
        py::class_<Operator , std::shared_ptr<Operator> >(m, "Operator" );

        m   .def("_make_operator"  , &makeOp                 , policy::move      )
            .def("_make_tensor"    , &makeTensor             , policy::move      )
            .def("_make_data"      , &makeTensorWithData     , policy::move      )
            .def("_make_compiler"  , &makeCompiler           , policy::move      );

        // clang-format on

        // TODO 临时测试用
        py::class_<computation::Graph, std::shared_ptr<computation::Graph>>(m, "ComputationGraph");
    }

}// namespace refactor::python_ffi
