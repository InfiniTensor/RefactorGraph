#include "onnx/operators.h"
#include "frontend/operator.h"
#include "operators/common.h"

#include "operators/binary.hh"

namespace refactor::onnx {

    void register_() {
        // clang-format off
        Operator::register_<Binary>("onnx::Add");
        Operator::register_<Binary>("onnx::Sub");
        Operator::register_<Binary>("onnx::Mul");
        Operator::register_<Binary>("onnx::Div");
        // clang-format on
    }

}// namespace refactor::onnx
