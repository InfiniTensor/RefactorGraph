#include "communication/operators.h"
#include "hardware/device.h"
#include "import.h"
#include "onnx/operators.h"
#include <pybind11/stl.h>// keep this line to convert stl types

namespace py = pybind11;

namespace refactor::python_ffi {
    using namespace hardware;

    PYBIND11_MODULE(python_ffi, m) {
        using return_ = py::return_value_policy;
        using namespace frontend;

        onnx::register_();
        communication::register_();

        // clang-format off

        py::class_<Tensor      , Arc<Tensor>      >(m, "Tensor"      );
        py::class_<OpBox       , Arc<OpBox>       >(m, "Operator"    );
        py::class_<Device      , Arc<Device>      >(m, "Device"      );

        m   .def("config_log"      , &configLog                  , return_::automatic )
            .def("find_device"     , &findDevice                 , return_::move      )
            .def("_make_operator"  , &makeOp                     , return_::move      )
            .def("_make_tensor"    , &makeTensor                 , return_::move      )
            .def("_make_data"      , &makeTensorWithData         , return_::move      )
            .def("_make_data_ex"   , &makeTensorWithExternalData , return_::move      )
            .def("_make_compiler"  , &makeCompiler               , return_::move      );

        py::class_<Compiler , Arc<Compiler>>(m, "Compiler" )
            .def("substitute"      , &Compiler::substitute       , return_::automatic )
            .def("set_input"       , &Compiler::setInput         , return_::automatic )
            .def("check_variables" , &Compiler::fillEdgeInfo     , return_::move      )
            .def("zero_inputs"     , &Compiler::zeroInputs       , return_::move      )
            .def("get_tensor"      , &Compiler::getTensor        , return_::move      )
            .def("compile"         , &Compiler::compile          , return_::move      )
            .def("compile_on"      , &Compiler::compileOn        , return_::move      );

        py::class_<Executor , Arc<Executor>>(m, "Executor" )
            .def("dispatch"        , &Executor::dispatch         , return_::automatic )
            .def("set_input"       , &Executor::setInput         , return_::automatic )
            .def("get_output"      , &Executor::getOutput        , return_::move      )
            .def("run"             , &Executor::run              , return_::automatic )
            .def("bench"           , &Executor::bench            , return_::automatic )
            .def("trace"           , &Executor::trace            , return_::automatic )
            .def("dbg"             , &Executor::debugInfo        , return_::automatic );

        // clang-format on
    }

}// namespace refactor::python_ffi
