#include "communication/operators.h"
#include "operators/all_gather.hh"
#include "operators/all_reduce.hh"

namespace refactor::communication {
    using namespace frontend;

    void register_() {
#define REGISTER(NAME, CLASS) Operator::register_<CLASS>("onnx::" #NAME)
        // clang-format off
        REGISTER(AllReduceAvg , AllReduce);
        REGISTER(AllReduceSum , AllReduce);
        REGISTER(AllReduceMin , AllReduce);
        REGISTER(AllReduceMax , AllReduce);
        REGISTER(AllReduceProd, AllReduce);
        REGISTER(AllGather    , AllGather);
        // clang-format on
#undef REGISTER
    }

}// namespace refactor::communication
