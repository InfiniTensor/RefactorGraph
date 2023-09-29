#include "communication/operators.h"
#include "frontend/operator.h"
#include "operators/common.h"

namespace refactor::communication {
    using namespace frontend;

    void register_() {
        // clang-format off
        OpType::register_("onnx::AllReduceSum"  , inferAllReduce , unreachableLower);
        OpType::register_("onnx::AllReduceProd" , inferAllReduce , unreachableLower);
        OpType::register_("onnx::AllReduceMin"  , inferAllReduce , unreachableLower);
        OpType::register_("onnx::AllReduceMax"  , inferAllReduce , unreachableLower);
        OpType::register_("onnx::AllReduceAvg"  , inferAllReduce , unreachableLower);
        OpType::register_("onnx::AllGather"     , inferAllGather , unreachableLower);
        // clang-format on
    }

}// namespace refactor::communication
