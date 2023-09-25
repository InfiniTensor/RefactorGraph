﻿#include "communication/operators.h"
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

        py::class_<Tensor   , std::shared_ptr<Tensor>   >(m, "Tensor"   );
        py::class_<Operator , std::shared_ptr<Operator> >(m, "Operator" );

        m   .def("config_log"      , &configLog              , return_::automatic )
            .def("_make_operator"  , &makeOp                 , return_::move      )
            .def("_make_tensor"    , &makeTensor             , return_::move      )
            .def("_make_data"      , &makeTensorWithData     , return_::move      )
            .def("_make_compiler"  , &makeCompiler           , return_::move      );

        py::class_<Compiler , std::shared_ptr<Compiler> >(m, "Compiler" )
            .def("substitute"      , &Compiler::substitute   , return_::automatic )
            .def("set_input"       , &Compiler::setInput     , return_::automatic )
            .def("check_variables" , &Compiler::fillEdgeInfo , return_::move      )
            .def("get_tensor"      , &Compiler::getTensor    , return_::move      )
            .def("compile"         , &Compiler::compile      , return_::move      );

        py::class_<Executor , std::shared_ptr<Executor> >(m, "Executor" );

        // clang-format on
    }

}// namespace refactor::python_ffi