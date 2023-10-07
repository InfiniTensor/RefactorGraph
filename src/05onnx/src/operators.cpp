#include "onnx/operators.h"
#include "frontend/operator.h"
#include "operators/common.h"

#include "operators/simple_binary.hh"

namespace refactor::onnx {

    void register_() {
        // clang-format off
        Operator::register_<SimpleBinary>("onnx::Add");
        Operator::register_<SimpleBinary>("onnx::Sub");
        Operator::register_<SimpleBinary>("onnx::Mul");
        Operator::register_<SimpleBinary>("onnx::Div");
        // clang-format on
    }

}// namespace refactor::onnx
