#include "communication/operators.h"
#include "import.h"
#include "onnx/operators.h"
#include <pybind11/stl.h>// keep this line to convert stl types

namespace py = pybind11;

namespace refactor::python_ffi {

    PYBIND11_MODULE(python_ffi, m) {
        using return_ = py::return_value_policy;
        using namespace frontend;

        onnx::register_();
        communication::register_();

        // clang-format off

        py::class_<Tensor   , Arc<Tensor>  >(m, "Tensor"   );
        py::class_<OpBox    , Arc<OpBox>   >(m, "Operator" );

        m   .def("config_log"      , &configLog                  , return_::automatic )
            .def("_make_operator"  , &makeOp                     , return_::move      )
            .def("_make_tensor"    , &makeTensor                 , return_::move      )
            .def("_make_data"      , &makeTensorWithData         , return_::move      )
            .def("_make_data_ex"   , &makeTensorWithExternalData , return_::move      )
            .def("_make_compiler"  , &makeCompiler               , return_::move      );

        py::class_<Compiler , Arc<Compiler>>(m, "Compiler" )
            .def("substitute"      , &Compiler::substitute       , return_::automatic )
            .def("set_input"       , &Compiler::setInput         , return_::automatic )
            .def("check_variables" , &Compiler::fillEdgeInfo     , return_::move      )
            .def("get_tensor"      , &Compiler::getTensor        , return_::move      )
            .def("compile"         , &Compiler::compile          , return_::move      );

        py::class_<Executor , Arc<Executor>>(m, "Executor" )
            .def("setInput"        , &Executor::setInput         , return_::automatic )
            .def("prepare"         , &Executor::prepare          , return_::move      );

        // clang-format on
    }

}// namespace refactor::python_ffi
