#include "communication/operators.h"
#include "computation/operator.h"
#include "infer/infer.h"

namespace refactor::communication {
    using namespace computation;

    void register_() {
        OpType::register_("onnx::AllReduceSum", inferAllReduce);
        OpType::register_("onnx::AllReduceProd", inferAllReduce);
        OpType::register_("onnx::AllReduceMin", inferAllReduce);
        OpType::register_("onnx::AllReduceMax", inferAllReduce);
        OpType::register_("onnx::AllReduceAvg", inferAllReduce);
        OpType::register_("onnx::AllGather", inferAllGather);
    }

}// namespace refactor::communication
