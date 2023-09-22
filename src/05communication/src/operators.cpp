#include "communication/operators.h"
#include "frontend/operator.h"
#include "operators/common.h"

namespace refactor::communication {
    using namespace frontend;

    void register_() {
        // clang-format off
        OpType::register_("onnx::AllReduceSum"  , inferAllReduce , lowerAllReduce);
        OpType::register_("onnx::AllReduceProd" , inferAllReduce , lowerAllReduce);
        OpType::register_("onnx::AllReduceMin"  , inferAllReduce , lowerAllReduce);
        OpType::register_("onnx::AllReduceMax"  , inferAllReduce , lowerAllReduce);
        OpType::register_("onnx::AllReduceAvg"  , inferAllReduce , lowerAllReduce);
        OpType::register_("onnx::AllGather"     , inferAllGather , lowerAllGather);
        // clang-format on
    }

}// namespace refactor::communication
